"""Deutsch-Jozsa: the notebook's two predicates on QUMO, each decided by one query.

Reproduces: Classiq library - Deutsch-Jozsa Algorithm
Source: https://github.com/Classiq/classiq-library/blob/main/algorithms/foundational/deutsch_jozsa/deutsch_jozsa.ipynb (Classiq Library, MIT)

The notebook runs H^n on x, a phase oracle (-1)^f(x), then H^n, and reads constant or balanced from the measured x: the outcome 0 alone means constant, a single other outcome means balanced. Its two predicates are f(x) = x > 7 on 4 qubits and f(x) = max(x ^ 4, x & 3) >= 4 on 3 qubits; both are balanced, and the notebook prints "The function is balanced" for each. This file runs the same circuit on QUMO for both, sampling 1000 shots (the notebook passes no shot count). Checks: the phase oracle's sign on every basis input against the predicate, the exact output law against the Walsh-Hadamard formula of the notebook's Technical Notes (a point mass at 8 and at 4), the ancilla returning to |0>, and the sampled decision. The file also runs the two constant functions on 4 qubits (f = 0 and f = 1), which the notebook does not run, so the constant branch of the post-processing is exercised against the analytic P(0) = 1.

Deviations: the notebook synthesizes its arithmetic predicates into reversible circuits (Classiq phase_oracle(predicate)). QUMO has no such synthesis here: the predicate is evaluated classically on every basis state, and the oracle is phase_oracle(n, marked) from qumo.algorithms.library, built from X and MCZ gates (the Grover construction). Its action is still exactly (-1)^f(x). The notebook's max_width=11 synthesis bound, its job links and its show() circuit links are not reproduced.

Run: .venv/bin/mojo run -I . examples/frameworks/classiq/deutsch_jozsa.mojo
example-meta: {"framework": "classiq", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from std.bit import pop_count
from qumo.ir.circuit import Circuit
from qumo.sim.runner import evolve, sample, bitstring
from qumo.algorithms.library import phase_oracle, num_mcz_ancillas
from qumo.testing.checks import CheckLog


def parity(x: Int) -> Int:
    return Int(pop_count(x)) & 1


def show(xs: List[Int]) -> String:
    var s = String("[")
    for i in range(len(xs)):
        if i > 0:
            s += ", "
        s += String(xs[i])
    return s + "]"


# ---------------------------------------------------------------------------
# The notebook's predicates. A basis index x is the integer held by the register.
# ---------------------------------------------------------------------------
def simple_f(x: Int) -> Bool:
    """The notebook's simple_predicate: x > 7."""
    return x > 7


def complex_f(x: Int) -> Bool:
    """The notebook's complex_predicate: max(x ^ 4, x & 3) >= 4."""
    return max(x ^ 4, x & 3) >= 4


def zero_f(x: Int) -> Bool:
    """Constant 0 (an extension: the notebook runs only balanced functions)."""
    return False


def one_f(x: Int) -> Bool:
    """Constant 1 (an extension: the notebook runs only balanced functions)."""
    return True


def marked_items[F: def(Int) -> Bool](n: Int, pred: F) -> List[Int]:
    """Basis states x in [0, 2^n) where the predicate holds: the phase oracle's marked set."""
    var out = List[Int]()
    for x in range(1 << n):
        if pred(x):
            out.append(x)
    return out^


def dj_circuit(n: Int, marked: List[Int], measure: Bool) raises -> Circuit:
    """The notebook's deutsch_jozsa: H^n on x (qubits 0..n-1), the phase oracle (its clean
    ancillas return to |0>), H^n, then optionally measure x into clbit q = qubit q."""
    var c = Circuit(n + num_mcz_ancillas(n), name="deutsch_jozsa")
    for q in range(n):
        c.h(q)
    c.compose(phase_oracle(n, marked))
    for q in range(n):
        c.h(q)
    if measure:
        for q in range(n):
            c.measure(q, q)
    return c^


def oracle_worst[F: def(Int) -> Bool](n: Int, marked: List[Int], pred: F) raises -> Float64:
    """Worst deviation of the phase oracle from |x> -> (-1)^f(x) |x> over all 2^n inputs:
    the real and imaginary parts of amplitude x, and the weight on every other basis state."""
    var nanc = num_mcz_ancillas(n)
    var worst = 0.0
    for x in range(1 << n):
        var prep = Circuit(n + nanc, name="basis_prep")
        for q in range(n):
            if ((x >> q) & 1) == 1:
                prep.x(q)
        prep.compose(phase_oracle(n, marked))
        var sv = evolve(prep)
        var probs = sv.probabilities()
        var want = -1.0 if pred(x) else 1.0
        var off = 0.0
        for i in range(len(probs)):
            if i != x:
                off += probs[i]
        worst = max(worst, max(abs(sv.re[x] - want), max(abs(sv.im[x]), off)))
    return worst


def full_probs(n: Int, marked: List[Int]) raises -> List[Float64]:
    """Exact output probabilities of dj_circuit(n, marked) over the register and the ancillas."""
    return evolve(dj_circuit(n, marked, False)).probabilities()


def marginal(probs: List[Float64], n: Int) -> List[Float64]:
    """Probability of each value of the register x, with the ancillas traced out."""
    var mask = (1 << n) - 1
    var law = List[Float64](length=1 << n, fill=0.0)
    for i in range(len(probs)):
        law[i & mask] += probs[i]
    return law^


def ancilla_mass(probs: List[Float64], n: Int) -> Float64:
    """Probability that any ancilla (qubit n and above) reads 1."""
    var out = 0.0
    for i in range(len(probs)):
        if (i >> n) != 0:
            out += probs[i]
    return out


def ideal_law[F: def(Int) -> Bool](n: Int, pred: F) -> List[Float64]:
    """Walsh-Hadamard formula of the Technical Notes, by brute force:
    P(k) = |2^-n sum_j (-1)^(f(j) + j.k)|^2, with j.k the parity of j AND k."""
    var big = 1 << n
    var law = List[Float64](length=big, fill=0.0)
    for k in range(big):
        var amp = 0.0
        for j in range(big):
            var sign = 1.0
            if pred(j):
                sign = -sign
            if parity(j & k) == 1:
                sign = -sign
            amp += sign
        amp /= Float64(big)
        law[k] = amp * amp
    return law^


def law_deviation(a: List[Float64], b: List[Float64]) -> Float64:
    var dev = 0.0
    for k in range(len(a)):
        dev = max(dev, abs(a[k] - b[k]))
    return dev


def decide(keys: List[String], n: Int) -> String:
    """The notebook's post_process_deutsch_jozsa, applied to the distinct measured outcomes."""
    if len(keys) == 1:
        if keys[0] == bitstring(0, n):
            return "constant"
        return "balanced"
    return "cannot decide"


def main() raises:
    var log = CheckLog("classiq/deutsch_jozsa")
    var shots = 1000
    var seed = UInt64(2025)

    # ---- A: f(x) = x > 7 on n = 4 qubits (the notebook's simple example) ----
    var n_a = 4
    var marked_a = marked_items(n_a, simple_f)
    print("A: f(x) = x > 7, n = 4, marked items", show(marked_a))
    _ = log.check("A f is balanced: 8 of 16 inputs are marked", len(marked_a) == 8, "marked=" + String(len(marked_a)))
    _ = log.close("A phase oracle equals (-1)^f(x) on all 16 basis inputs (max deviation)", oracle_worst(n_a, marked_a, simple_f), 0.0, 1e-12)
    var probs_a = full_probs(n_a, marked_a)
    var law_a = marginal(probs_a, n_a)
    _ = log.close("A output law equals the Walsh-Hadamard formula (max deviation)", law_deviation(law_a, ideal_law(n_a, simple_f)), 0.0, 1e-12)
    _ = log.close("A P(x = 0) = 0 for a balanced f", law_a[0], 0.0, 1e-12)
    _ = log.close("A P(x = 8) = 1 (exact; the notebook measures 8)", law_a[8], 1.0, 1e-12)
    _ = log.close("A ancilla returns to |0> (P(ancilla = 1))", ancilla_mass(probs_a, n_a), 0.0, 1e-12)
    var res_a = sample(dj_circuit(n_a, marked_a, True), shots, seed=seed)
    var dec_a = decide(res_a.sorted_keys(), n_a)
    print("   A: The function is", dec_a, "| most frequent outcome", res_a.most_frequent(), "from", shots, "shots")
    _ = log.check("A sampled decision is balanced (notebook: The function is balanced)", dec_a == "balanced", "decision=" + dec_a)
    _ = log.close("A sampled P(outcome 1000 = 8) = 1.0", res_a.probability(bitstring(8, n_a)), 1.0, 1e-12)

    # ---- B: f(x) = max(x ^ 4, x & 3) >= 4 on n = 3 qubits (the notebook's complex example) ----
    var n_b = 3
    var marked_b = marked_items(n_b, complex_f)
    print("B: f(x) = max(x ^ 4, x & 3) >= 4, n = 3, marked items", show(marked_b))
    _ = log.check("B f is balanced: 4 of 8 inputs are marked", len(marked_b) == 4, "marked=" + String(len(marked_b)))
    _ = log.close("B phase oracle equals (-1)^f(x) on all 8 basis inputs (max deviation)", oracle_worst(n_b, marked_b, complex_f), 0.0, 1e-12)
    var probs_b = full_probs(n_b, marked_b)
    var law_b = marginal(probs_b, n_b)
    _ = log.close("B output law equals the Walsh-Hadamard formula (max deviation)", law_deviation(law_b, ideal_law(n_b, complex_f)), 0.0, 1e-12)
    _ = log.close("B P(x = 0) = 0 for a balanced f", law_b[0], 0.0, 1e-12)
    _ = log.close("B P(x = 4) = 1 (exact; the notebook measures 4)", law_b[4], 1.0, 1e-12)
    var res_b = sample(dj_circuit(n_b, marked_b, True), shots, seed=seed)
    var dec_b = decide(res_b.sorted_keys(), n_b)
    print("   B: The function is", dec_b, "| most frequent outcome", res_b.most_frequent(), "from", shots, "shots")
    _ = log.check("B sampled decision is balanced (notebook: The function is balanced)", dec_b == "balanced", "decision=" + dec_b)
    _ = log.close("B sampled P(outcome 100 = 4) = 1.0", res_b.probability(bitstring(4, n_b)), 1.0, 1e-12)

    # ---- C: the two constant functions on n = 4 (extension: exercises the constant branch) ----
    var zero_marked = marked_items(4, zero_f)
    var one_marked = marked_items(4, one_f)
    _ = log.close("C f = 0: phase oracle is +1 on all 16 basis inputs (max deviation)", oracle_worst(4, zero_marked, zero_f), 0.0, 1e-12)
    _ = log.close("C f = 1: phase oracle is -1 on all 16 basis inputs (max deviation)", oracle_worst(4, one_marked, one_f), 0.0, 1e-12)
    var law_zero = marginal(full_probs(4, zero_marked), 4)
    var law_one = marginal(full_probs(4, one_marked), 4)
    _ = log.close("C f = 0: P(x = 0) = 1 (constant)", law_zero[0], 1.0, 1e-12)
    _ = log.close("C f = 1: P(x = 0) = 1 (constant)", law_one[0], 1.0, 1e-12)
    var res_zero = sample(dj_circuit(4, zero_marked, True), shots, seed=seed)
    var res_one = sample(dj_circuit(4, one_marked, True), shots, seed=seed)
    var dec_zero = decide(res_zero.sorted_keys(), 4)
    var dec_one = decide(res_one.sorted_keys(), 4)
    _ = log.check("C sampled decision for f = 0 is constant", dec_zero == "constant", "decision=" + dec_zero)
    _ = log.check("C sampled decision for f = 1 is constant", dec_one == "constant", "decision=" + dec_one)

    log.finish()
