"""Grover search: the Classiq notebook's 3-SAT and MaxCut instances on QUMO.

Reproduces: Classiq library - Grover's Search Algorithm
Source: https://github.com/Classiq/classiq-library/blob/main/algorithms/search_and_optimization/grover/grover.ipynb (Classiq Library, MIT)

The notebook finds the solutions of two 3-SAT formulas (2 of 8 and 2 of 16 assignments) and the cuts of size >= 4 of a 5-node graph (10 of 32) by repeating Grover iterations until a marked item is sampled above a threshold. QUMO marks the same basis states (the predicate's truth table) for phase_oracle; the checks compare the solution sets, the selected iteration counts, exact success probabilities with sin^2((2r+1) theta), sampled success, and the MaxCut optimum (5).

One note on the notebook text: its stopping rule says a marked item is sampled above 0.1 after one repetition. The exact probability of each marked item after one repetition is 0.0957, below that threshold, so the stop there is a sampled event; this file checks the seeded stop it produces, not the threshold as an exact property.

Run: .venv/bin/mojo run -I . examples/frameworks/classiq/grover.mojo
example-meta: {"framework": "classiq", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from std.math import asin, ceil, pi, sin, sqrt
from qumo.sim.runner import evolve, sample
from qumo.algorithms.library import grover, bitstring_to_int
from qumo.testing.checks import CheckLog


def bit(x: Int, i: Int) -> Int:
    return (x >> i) & 1


# ---------------------------------------------------------------------------
# The notebook's predicates. A basis index x holds the variables: bit i is x_i.
# ---------------------------------------------------------------------------
def small_3sat(x: Int) -> Bool:
    var x0 = bit(x, 0) == 1
    var x1 = bit(x, 1) == 1
    var x2 = bit(x, 2) == 1
    return (
        (x0 or x1 or x2)
        and (not x0 or x1 or x2)
        and (not x0 or not x1 or not x2)
        and (not x0 or not x1 or x2)
        and (x0 or x1 or not x2)
        and (not x0 or x1 or not x2)
    )


def large_3sat(x: Int) -> Bool:
    var x0 = bit(x, 0) == 1
    var x1 = bit(x, 1) == 1
    var x2 = bit(x, 2) == 1
    var x3 = bit(x, 3) == 1
    return (
        (x1 or x2 or x3)
        and (not x0 or x1 or x2)
        and (not x0 or x1 or not x2)
        and (not x0 or not x1 or x2)
        and (x0 or not x1 or not x2)
        and (x0 or not x1 or x2)
        and (not x0 or not x1 or not x3)
        and (not x0 or not x1 or x3)
        and (not x1 or not x2 or not x3)
        and (x1 or not x2 or x3)
        and (x0 or not x2 or x3)
        and (x0 or not x1 or not x3)
        and (not x0 or not x1 or not x2)
    )


def cut_size(x: Int) -> Int:
    """Number of edges of the notebook's 5-node graph whose endpoints differ."""
    var ei: List[Int] = [0, 0, 1, 1, 2, 3]
    var ej: List[Int] = [1, 2, 2, 3, 4, 4]
    var c = 0
    for k in range(len(ei)):
        if bit(x, ei[k]) != bit(x, ej[k]):
            c += 1
    return c


def cut_at_least_4(x: Int) -> Bool:
    return cut_size(x) >= 4


def max_cut() -> Int:
    var best = 0
    for x in range(32):
        var c = cut_size(x)
        if c > best:
            best = c
    return best


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def truth_table[F: def(Int) -> Bool](n: Int, pred: F) -> List[Int]:
    """Basis states x in [0, 2^n) where the predicate holds: the marked items."""
    var out = List[Int]()
    for x in range(1 << n):
        if pred(x):
            out.append(x)
    return out^


def items_from_keys(keys: List[String]) raises -> List[Int]:
    """Notebook bitstrings (leftmost char = highest qubit) to basis indices."""
    var out = List[Int]()
    for k in keys:
        out.append(bitstring_to_int(k))
    return out^


def is_marked(x: Int, marked: List[Int]) -> Bool:
    for m in marked:
        if m == x:
            return True
    return False


def same_set(a: List[Int], b: List[Int]) -> Bool:
    if len(a) != len(b):
        return False
    for x in a:
        if not is_marked(x, b):
            return False
    return True


def show(xs: List[Int]) -> String:
    var s = String("[")
    for i in range(len(xs)):
        if i > 0:
            s += ", "
        s += String(xs[i])
    return s + "]"


def success_probability(num_items: Int, k: Int, r: Int) -> Float64:
    """Analytic sin^2((2r+1) theta) with sin(theta) = sqrt(k / N)."""
    var theta = asin(sqrt(Float64(k) / Float64(num_items)))
    var s = sin(Float64(2 * r + 1) * theta)
    return s * s


def exact_after(n: Int, marked: List[Int], r: Int) raises -> Tuple[Float64, Float64, Float64, Float64]:
    """Exact state after r iterations: (total probability on the marked items,
    smallest and largest probability of one marked item, probability on the
    clean search space). The last value is 1 when the ancillas return to |0>."""
    var sv = evolve(grover(n, marked, r))
    var total = 0.0
    var lo = 1.0
    var hi = 0.0
    for m in marked:
        var p = sv.re[m] * sv.re[m] + sv.im[m] * sv.im[m]
        total += p
        lo = min(lo, p)
        hi = max(hi, p)
    var inside = 0.0
    for i in range(1 << n):
        inside += sv.re[i] * sv.re[i] + sv.im[i] * sv.im[i]
    return (total, lo, hi, inside)


def sampled_marked_fraction(n: Int, marked: List[Int], r: Int, shots: Int, seed: UInt64) raises -> Float64:
    var res = sample(grover(n, marked, r, True), shots, seed)
    var hits = 0
    for kv in res.counts.items():
        if is_marked(bitstring_to_int(kv.key), marked):
            hits += kv.value
    return Float64(hits) / Float64(shots)


def repeat_until_success(n: Int, marked: List[Int], threshold: Float64, shots: Int, seed: UInt64) raises -> Tuple[Int, Bool]:
    """The notebook's repeat_grover_until_success with max_iter = 4: run
    r = ceil(pi/4 sqrt(2^i)) iterations, skip a repeated r, and stop when a
    marked item is sampled with frequency above `threshold`.
    Returns (r of the last run, found)."""
    var r = 0
    var r_previous = -1
    for i in range(4):
        r = Int(ceil(pi / 4.0 * sqrt(Float64(1 << i))))
        if r == r_previous:
            continue
        r_previous = r
        var res = sample(grover(n, marked, r, True), shots, seed)
        for kv in res.counts.items():
            var p = Float64(kv.value) / Float64(shots)
            if p > threshold and is_marked(bitstring_to_int(kv.key), marked):
                return (r, True)
    return (r, False)


def main() raises:
    var log = CheckLog("classiq/grover")
    var shots = 1000
    var seed = UInt64(1234)

    # ---- A: small 3-SAT, N = 8, the notebook's two solutions ----
    var small_marked = truth_table(3, small_3sat)
    var small_ref = items_from_keys(["010", "110"])
    print("A: small 3-SAT, n = 3, marked items", show(small_marked))
    _ = log.check("small 3-SAT marked set equals the notebook solutions", same_set(small_marked, small_ref))
    var a1 = exact_after(3, small_marked, 1)
    _ = log.close("small 3-SAT success probability at r = 1", a1[0], success_probability(8, 2, 1), 1e-9)
    var small_run = repeat_until_success(3, small_marked, 0.45, shots, seed)
    print("   loop selects r =", small_run[0], " found =", small_run[1])
    _ = log.check("small 3-SAT loop stops at r = 1 (notebook: 1 repetition)", small_run[1] and small_run[0] == 1)

    # ---- B: larger 3-SAT, N = 16, the notebook's two solutions ----
    var large_marked = truth_table(4, large_3sat)
    var large_ref = items_from_keys(["1000", "1100"])
    print("B: large 3-SAT, n = 4, marked items", show(large_marked))
    _ = log.check("large 3-SAT marked set equals the notebook solutions", same_set(large_marked, large_ref))
    var b1 = exact_after(4, large_marked, 1)
    _ = log.close("large 3-SAT success probability at r = 1", b1[0], success_probability(16, 2, 1), 1e-9)
    _ = log.check("large 3-SAT at r = 1 keeps each item below 0.45", b1[2] < 0.45)
    var b2 = exact_after(4, large_marked, 2)
    _ = log.close("large 3-SAT success probability at r = 2", b2[0], success_probability(16, 2, 2), 1e-9)
    _ = log.close("large 3-SAT ancillas clean at r = 2", b2[3], 1.0, 1e-9)
    var large_run = repeat_until_success(4, large_marked, 0.45, shots, seed)
    print("   loop selects r =", large_run[0], " found =", large_run[1])
    _ = log.check("large 3-SAT loop stops at r = 2 (notebook: 2 repetitions)", large_run[1] and large_run[0] == 2)
    var frac_large = sampled_marked_fraction(4, large_marked, 2, shots, seed)
    print("   sampled success at r = 2:", frac_large, " (notebook 0.481 + 0.467 = 0.948)")
    _ = log.within("large 3-SAT sampled success within notebook noise of 0.948", frac_large, 0.92, 0.97)

    # ---- C: MaxCut, cut size >= 4 on the 5-node graph (10 marked of 32) ----
    var cut_marked = truth_table(5, cut_at_least_4)
    var cut_ref = items_from_keys([
        "01110", "01100", "01101", "10010", "00110",
        "10110", "10011", "10001", "11001", "01001",
    ])
    print("C: MaxCut, cut >= 4, n = 5, marked items", show(cut_marked))
    _ = log.check("cut >= 4 marked set equals the notebook's ten cuts", same_set(cut_marked, cut_ref))
    _ = log.check("MaxCut optimum of the graph is 5 by brute force", max_cut() == 5)
    var c1 = exact_after(5, cut_marked, 1)
    _ = log.close("cut >= 4 success probability at r = 1", c1[0], success_probability(32, 10, 1), 1e-9)
    _ = log.close("cut >= 4 items equally likely at r = 1", c1[2], c1[1], 1e-9)
    _ = log.close("cut >= 4 ancillas clean at r = 1", c1[3], 1.0, 1e-9)
    print("   exact probability of one marked cut at r = 1:", c1[1], " notebook threshold 0.1")
    var cut_run = repeat_until_success(5, cut_marked, 0.1, shots, seed)
    print("   loop selects r =", cut_run[0], " found =", cut_run[1])
    _ = log.check("cut >= 4 loop stops at r = 1 (notebook: 1 repetition)", cut_run[1] and cut_run[0] == 1)
    var frac_cut = sampled_marked_fraction(5, cut_marked, 1, shots, seed)
    print("   sampled success at r = 1:", frac_cut, " (notebook top ten sum 0.959)")
    _ = log.within("cut >= 4 sampled success within notebook noise of 0.959", frac_cut, 0.93, 0.98)
    var top = sample(grover(5, cut_marked, 1, True), shots, seed).most_frequent()
    _ = log.check("cut >= 4 most frequent sample is a marked cut", is_marked(bitstring_to_int(top), cut_marked), "top=" + top)

    log.finish()
