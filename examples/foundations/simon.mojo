"""Simon's algorithm: the notebook's two oracles, secret strings recovered by GF(2) elimination.

Reproduces: Classiq library - Simon's Algorithm
Source: https://github.com/Classiq/classiq-library/blob/main/algorithms/foundational/simon/simon.ipynb (Classiq Library, MIT)
Framework: Classiq - https://github.com/Classiq/classiq-library

The notebook runs H^n, a 2-to-1 oracle f with f(x) = f(x XOR s), H^n, then measures x; n-1 outcomes y with y.s = 0 (mod 2) give s.
Instance A is f(x) = min(x, x XOR s) on 5 qubits with s = 6; QUMO has no arithmetic min, so the oracle is the CX network x XOR (x_h s), h the top bit of s. Instance B is the notebook's bipartite oracle on 6 qubits with p = 4 and s = 60.
Checks: both oracles on every basis input, the exact output law (uniform on s-perp, success probability 1), and sampled outcomes with the secret recovered from 50n shots.

Run: .venv/bin/mojo run -I . examples/foundations/simon.mojo
example-meta: {"framework": "classiq", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from std.bit import pop_count
from qumo.ir.circuit import Circuit
from qumo.sim.runner import evolve, sample, bitstring
from qumo.testing.checks import CheckLog


def parity(x: Int) -> Int:
    return Int(pop_count(x)) & 1


def simon_circuit(n: Int, oracle: Circuit, measure: Bool) -> Circuit:
    """H^n on x (qubits 0..n-1), the oracle on x and res (qubits n..2n-1), H^n, then optionally measure x."""
    var c = Circuit(2 * n, name="simon")
    for q in range(n):
        c.h(q)
    c.compose(oracle)
    for q in range(n):
        c.h(q)
    if measure:
        for q in range(n):
            c.measure(q, q)
    return c^


def min_oracle(n: Int, s: Int) -> Circuit:
    """|x>|0> -> |x>|min(x, x XOR s)> as CX gates.

    min(x, x XOR s) = x XOR (x_h s), h the top set bit of s: x and x XOR s agree above bit h
    and differ at bit h, and the minimum is the one whose bit h is 0.
    """
    var h = 0
    for j in range(n):
        if ((s >> j) & 1) == 1:
            h = j
    var c = Circuit(2 * n, name="min_oracle")
    for q in range(n):
        c.cx(q, n + q)
    for j in range(n):
        if ((s >> j) & 1) == 1:
            c.cx(h, n + j)
    return c^


def bipartite_oracle(n: Int, p: Int) -> Circuit:
    """The notebook's simon_qfunc_with_bipartite_s: copy x_0..x_{n-p-1}, a 0 at n-p, and x_j XOR x_{n-p} above it."""
    var c = Circuit(2 * n, name="bipartite_oracle")
    for i in range(n - p):
        c.cx(i, n + i)
    for j in range(n - p + 1, n):
        c.cx(j, n + j)
        c.cx(n - p, n + j)
    return c^


def bipartite_ref(x: Int, n: Int, p: Int) -> Int:
    """The notebook's f(x) as a classical function; bit j of the result is res_j."""
    var pivot = (x >> (n - p)) & 1
    var out = 0
    for j in range(n):
        var b = (x >> j) & 1
        if j == n - p:
            b = 0
        elif j > n - p:
            b ^= pivot
        out |= b << j
    return out


def basis_output(oracle: Circuit, n: Int, x: Int) raises -> Tuple[Int, Int, Float64]:
    """Run |x>|0> through the oracle: (x read back, res value, probability of that basis state)."""
    var prep = Circuit(2 * n, name="basis_prep")
    for q in range(n):
        if ((x >> q) & 1) == 1:
            prep.x(q)
    prep.compose(oracle)
    var probs = evolve(prep).probabilities()
    var best = 0
    for i in range(len(probs)):
        if probs[i] > probs[best]:
            best = i
    return (best & ((1 << n) - 1), best >> n, probs[best])


def two_to_one(f: List[Int], n: Int) -> Bool:
    """Every value that f takes on the 2^n inputs is taken exactly twice (or never)."""
    var counts = List[Int](length=1 << n, fill=0)
    for x in range(1 << n):
        counts[f[x]] += 1
    for v in range(1 << n):
        if counts[v] != 0 and counts[v] != 2:
            return False
    return True


def promise_holds(f: List[Int], n: Int, s: Int) -> Bool:
    """Check that f(x) = f(x XOR s) for every x."""
    for x in range(1 << n):
        if f[x] != f[x ^ s]:
            return False
    return True


def x_law(oracle: Circuit, n: Int) raises -> List[Float64]:
    """Exact distribution of x after H^n, the oracle and H^n (res traced out)."""
    var probs = evolve(simon_circuit(n, oracle, False)).probabilities()
    var law = List[Float64](length=1 << n, fill=0.0)
    var mask = (1 << n) - 1
    for i in range(len(probs)):
        law[i & mask] += probs[i]
    return law^


def law_errors(law: List[Float64], n: Int, s: Int) -> Tuple[Float64, Float64]:
    """(max deviation from the ideal law, P(y.s = 0)); the ideal law is 2^-(n-1) on s-perp and 0 elsewhere."""
    var expect = 1.0 / Float64(1 << (n - 1))
    var dev = 0.0
    var succ = 0.0
    for y in range(1 << n):
        var orth = parity(y & s) == 0
        var want = expect if orth else 0.0
        dev = max(dev, abs(law[y] - want))
        if orth:
            succ += law[y]
    return (dev, succ)


def solve_gf2(rows: List[Int], n: Int) -> Tuple[Int, Int]:
    """(secret, rank): the non-zero s with y.s = 0 (mod 2) for all rows, or 0 unless the rank is n-1."""
    var basis = List[Int]()  # reduced rows
    var pivots = List[Int]()
    for r0 in rows:
        var r = r0
        for i in range(len(basis)):
            if ((r >> pivots[i]) & 1) == 1:
                r ^= basis[i]
        if r == 0:
            continue
        var p = 0
        while ((r >> p) & 1) == 0:
            p += 1
        for i in range(len(basis)):
            if ((basis[i] >> p) & 1) == 1:
                basis[i] ^= r
        basis.append(r)
        pivots.append(p)
    var rank = len(basis)
    if rank != n - 1:
        return (0, rank)
    var is_pivot = List[Bool](length=n, fill=False)
    for p in pivots:
        is_pivot[p] = True
    var free = 0
    while is_pivot[free]:
        free += 1
    var s = 1 << free
    for i in range(len(basis)):
        if ((basis[i] >> free) & 1) == 1:
            s |= 1 << pivots[i]
    return (s, rank)


def run_sampled(oracle: Circuit, n: Int, shots: Int, seed: UInt64, s: Int) raises -> Tuple[Int, Int, Int, Int]:
    """Sample x with the oracle, then solve: (recovered secret, GF(2) rank, outcomes with y.s != 0, distinct outcomes)."""
    var res = sample(simon_circuit(n, oracle, True), shots, seed=seed)
    var rows = List[Int]()
    var bad = 0
    for y in res.memory:
        rows.append(y)
        if parity(y & s) != 0:
            bad += 1
    var sol = solve_gf2(rows, n)
    return (sol[0], sol[1], bad, len(res.counts))


def main() raises:
    var log = CheckLog("classiq/simon")

    # ---- A: f(x) = min(x, x XOR s), n = 5, s = 6 (the notebook's arithmetic oracle) ----
    var n_a = 5
    var s_a = 6
    var oracle_a = min_oracle(n_a, s_a)
    var table_a = List[Int](length=1 << n_a, fill=0)
    for x in range(1 << n_a):
        table_a[x] = min(x, x ^ s_a)
    var mismatch_a = 0
    var worst_a = 0.0
    for x in range(1 << n_a):
        var r = basis_output(oracle_a, n_a, x)
        if r[0] != x or r[1] != table_a[x]:
            mismatch_a += 1
        worst_a = max(worst_a, abs(1.0 - r[2]))
    _ = log.check("A CX oracle equals min(x, x XOR s) on all 32 basis inputs", mismatch_a == 0, "mismatches=" + String(mismatch_a))
    _ = log.close("A oracle maps each basis input to a single basis state", worst_a, 0.0, 1e-12)
    _ = log.check("A min(x, x XOR s) is 2-to-1 with f(x) = f(x XOR s)", promise_holds(table_a, n_a, s_a) and two_to_one(table_a, n_a))
    _ = log.check("A secret is the notebook's 00110", bitstring(s_a, n_a) == "00110", "s=" + String(s_a))
    var err_a = law_errors(x_law(oracle_a, n_a), n_a, s_a)
    _ = log.close("A exact output law is uniform on s-perp (max deviation)", err_a[0], 0.0, 1e-12)
    _ = log.close("A success probability P(y.s = 0)", err_a[1], 1.0, 1e-12)
    var shots_a = 50 * n_a
    var rec_a = run_sampled(oracle_a, n_a, shots_a, UInt64(7), s_a)
    print("A: secret", bitstring(s_a, n_a), "=", s_a, "| recovered", rec_a[0], "from", shots_a, "shots,", rec_a[3], "distinct outcomes, GF(2) rank", rec_a[1])
    _ = log.check("A every sampled outcome satisfies y.s = 0", rec_a[2] == 0, "violations=" + String(rec_a[2]))
    _ = log.check("A GF(2) elimination recovers s = 6", rec_a[0] == s_a and rec_a[1] == n_a - 1, "recovered=" + String(rec_a[0]))

    # ---- B: the notebook's bipartite oracle, n = 6, p = 4, s = 1^4 0^2 = 60 ----
    var n_b = 6
    var p_b = 4
    var s_b = ((1 << p_b) - 1) << (n_b - p_b)
    var oracle_b = bipartite_oracle(n_b, p_b)
    var table_b = List[Int](length=1 << n_b, fill=0)
    for x in range(1 << n_b):
        table_b[x] = bipartite_ref(x, n_b, p_b)
    var mismatch_b = 0
    var worst_b = 0.0
    for x in range(1 << n_b):
        var r = basis_output(oracle_b, n_b, x)
        if r[0] != x or r[1] != table_b[x]:
            mismatch_b += 1
        worst_b = max(worst_b, abs(1.0 - r[2]))
    _ = log.check("B bipartite oracle equals the notebook's f(x) on all 64 basis inputs", mismatch_b == 0, "mismatches=" + String(mismatch_b))
    _ = log.close("B oracle maps each basis input to a single basis state", worst_b, 0.0, 1e-12)
    _ = log.check("B f(x) = f(x XOR s) with f exactly 2-to-1", promise_holds(table_b, n_b, s_b) and two_to_one(table_b, n_b))
    _ = log.check("B secret is the notebook's 1111 00 (= 60)", bitstring(s_b, n_b) == "111100", "s=" + String(s_b))
    var err_b = law_errors(x_law(oracle_b, n_b), n_b, s_b)
    _ = log.close("B exact output law is uniform on s-perp (max deviation)", err_b[0], 0.0, 1e-12)
    _ = log.close("B success probability P(y.s = 0)", err_b[1], 1.0, 1e-12)
    var shots_b = 50 * n_b
    var rec_b = run_sampled(oracle_b, n_b, shots_b, UInt64(11), s_b)
    print("B: secret", bitstring(s_b, n_b), "=", s_b, "| recovered", rec_b[0], "from", shots_b, "shots,", rec_b[3], "distinct outcomes, GF(2) rank", rec_b[1])
    _ = log.check("B every sampled outcome satisfies y.s = 0", rec_b[2] == 0, "violations=" + String(rec_b[2]))
    _ = log.check("B GF(2) elimination recovers s = 60", rec_b[0] == s_b and rec_b[1] == n_b - 1, "recovered=" + String(rec_b[0]))

    log.finish()
