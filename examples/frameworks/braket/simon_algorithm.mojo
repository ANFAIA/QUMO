"""Simon's algorithm with a GF(2) solver (Amazon Braket example instance).

Reproduces: Amazon Braket - Simon's Algorithm
Source: https://github.com/amazon-braket/amazon-braket-examples/tree/main/examples/advanced_circuits_algorithms/Simons_Algorithm (Apache-2.0)

The notebook hides a 6-bit secret s = "101011" in a 2-to-1 function
f(x) = f(x XOR s) on 2n = 12 qubits:
- copy the input register into the output register (CNOT x_i -> y_i);
- for the "flag" qubit (the first 1 of s), apply CNOT(flag -> y_j) for every
  j with s_j = 1, so f(x) = x XOR (x_flag * s);
- H^n, oracle, H^n, measure the input register, 4n = 24 shots on a local
  simulator; Gaussian elimination over GF(2) recovers s from the outcomes.
Braket writes bitstrings with character i = qubit i, so "101011" sets qubits
0, 2, 4 and 5: s = 53 as a QUMO (little-endian) integer. The example then
repeats the experiment for further secrets, including s = 0 (a 1-to-1
function, recognised by a full-rank system).

Checks:
- every measured y satisfies y.s = 0 (mod 2) (exact, all shots, all secrets);
- the GF(2) solver recovers s for every secret (page: "Secret string: 101011");
- the exact output distribution is uniform over s-perp: P(y) = 2^-(n-1) on
  y.s = 0 and 0 elsewhere (1e-12; for s = 0 uniform 2^-n).

Run: .venv/bin/mojo run -I . examples/frameworks/braket/simon_algorithm.mojo
example-meta: {"framework": "braket", "tier": "fast", "timeout": 60, "requires": [], "args": []}
"""

from std.bit import pop_count
from qumo.ir.circuit import Circuit
from qumo.sim.runner import evolve, sample
from qumo.testing.checks import CheckLog

comptime N = 6
comptime SHOTS = 4 * N


def braket_bits_to_int(s: String) -> Int:
    """Braket bitstring (character i = qubit i) -> little-endian integer."""
    var v = 0
    var bs = s.as_bytes()
    for i in range(s.byte_length()):
        if Int(bs[i]) == ord("1"):
            v |= 1 << i
    return v


def braket_string(v: Int, n: Int) -> String:
    var s = String()
    for i in range(n):
        s += "1" if ((v >> i) & 1) == 1 else "0"
    return s


def simon_circuit(n: Int, secret: Int, measure: Bool) -> Circuit:
    var c = Circuit(2 * n, name="simon")
    for q in range(n):
        c.h(q)
    for q in range(n):
        c.cx(q, q + n)
    if secret != 0:
        var flag = 0
        while ((secret >> flag) & 1) == 0:
            flag += 1
        for j in range(n):
            if ((secret >> j) & 1) == 1:
                c.cx(flag, j + n)
    for q in range(n):
        c.h(q)
    if measure:
        for q in range(n):
            c.measure(q, q)
    return c^


def parity(x: Int) -> Int:
    return Int(pop_count(x)) & 1


def solve_gf2(rows: List[Int], n: Int, mut rank: Int) -> Int:
    """Non-trivial solution s of y.s = 0 for all rows (0 if the system has full rank).

    Gaussian elimination to reduced row-echelon form over GF(2); when the
    rank is n - 1 the unique non-zero kernel vector is read off the free
    column. rank < n - 1 is reported through `rank` (more shots needed).
    """
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
        for i in range(len(basis)):  # keep the echelon form reduced
            if ((basis[i] >> p) & 1) == 1:
                basis[i] ^= r
        basis.append(r)
        pivots.append(p)
    rank = len(basis)
    if rank != n - 1:
        return 0
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
    return s


def main() raises:
    var log = CheckLog("braket/simon_algorithm")
    var secrets: List[String] = ["101011", "110000", "000111", "011010", "000000"]
    var all_orthogonal = True
    var all_recovered = True
    var worst_dist = 0.0
    for si in range(len(secrets)):
        var s = braket_bits_to_int(secrets[si])
        # exact output distribution of the input register
        var probs = evolve(simon_circuit(N, s, False)).probabilities()
        var marg = List[Float64](length=1 << N, fill=0.0)
        for i in range(len(probs)):
            marg[i & ((1 << N) - 1)] += probs[i]
        var expect = 1.0 / Float64(1 << (N - 1)) if s != 0 else 1.0 / Float64(1 << N)
        for y in range(1 << N):
            var want = expect if (s == 0 or parity(y & s) == 0) else 0.0
            worst_dist = max(worst_dist, abs(marg[y] - want))
        # sampled run with 4n shots and the GF(2) solver
        var res = sample(simon_circuit(N, s, True), SHOTS, seed=UInt64(100 + si))
        var rows = List[Int]()
        var orth = True
        for y in res.memory:
            rows.append(y)
            if parity(y & s) != 0:
                orth = False
        var rank = 0
        var found = solve_gf2(rows, N, rank)
        var ok = found == s and (rank == N - 1 or (s == 0 and rank == N))
        print("secret", secrets[si], "(=", s, "):", len(res.counts), "distinct outcomes in", SHOTS, "shots, rank", rank,
              "-> recovered", braket_string(found, N), "OK" if ok else "FAILED")
        all_orthogonal = all_orthogonal and orth
        all_recovered = all_recovered and ok
    _ = log.check("every outcome satisfies y.s = 0", all_orthogonal)
    _ = log.check("GF(2) solver recovers every secret (page 101011)", all_recovered)
    _ = log.close("exact distribution is uniform over s-perp", worst_dist, 0.0, 1e-12)
    log.finish()
