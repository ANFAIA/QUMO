"""Shor's factoring algorithm: the notebook's order finding for N = 21, a = 11, on QUMO.

Reproduces: Classiq library - Shor's Factoring Algorithm
Source: https://github.com/Classiq/classiq-library/blob/main/algorithms/number_theory_and_cryptography/shor/shor.ipynb (Classiq Library, MIT)
Framework: Classiq - https://github.com/Classiq/classiq-library

The notebook finds the order r of 11 mod 21 by quantum phase estimation of multiplication by a mod N (the notebook calls qpe_flexible with the powered unitary a**p; the textbook schedule p = 2^j, used here, is assumed and not shown in the notebook text): a 10-qubit phase register (twice the 5 work qubits), an inverse QFT, continued-fraction post-processing with denominator limit N, and then the factors gcd(a^(r/2) +- 1, N). QUMO's order_finding implements the same QPE structure (phase qubit j controls the permutation x -> a^(2^j) x mod N of the work register, then the inverse QFT) but not the same gates: the permutations are Gray-code chains of multi-controlled X gates with clean ancillas, not Classiq's modular_multiply_constant_inplace circuit, which is not reproduced. The exact phase-register law is a sum of Fejer kernels, one per eigenphase s/r with weight 1/r; this file evaluates that sum by brute force and compares it with QUMO's statevector to 1e-12. The other checks compare QUMO with the notebook's printed output: its continued-fraction table for N = 18 and the 15 fractions it prints for N = 21, its five printed histogram rows (2048 shots, binomial z-scores against the exact law), and the factors 3 and 7. A seeded 2048-shot run of QUMO's sampler is checked against the analytic law on the six outcomes with exact probability above 0.05 only (the seed 2048 is arbitrary, the shot count, and was not tuned), and the period r = 6 is checked against a brute-force order.

Deviations from the notebook:
* Classiq's flexible_qpe and modular_multiply_constant_inplace are replaced by QUMO's order_finding. Its work register starts in |1> and only ever holds values below N, where x -> a^(2^j) x mod N is the notebook's modular multiplication; the controlled permutations use clean ancillas (3 qubits).
* The Classiq synthesis, cloud job and 2048-shot sample are not reproduced. QUMO's statevector gives the exact probabilities, and the notebook's printed histogram rows are compared with them. The notebook's output shows only five histogram rows, so those are the only counts used.
* The notebook's get_factors uses Python's ^ (XOR) where a power is meant: a ^ (r // 2) + 1 evaluates to 11 XOR 4 = 15. Its printed "3 and 7" comes out of that. This file computes a^(r/2) mod N by modular exponentiation and takes gcd(a^(r/2) +- 1, N); both give 3 and 7.
* The notebook keeps the outcomes with more than 10 counts in its own 2048-shot sample (15 outcomes). This file has no such sample for QUMO, so it keeps the outcomes whose exact Fejer probability exceeds 10/2048 (14 outcomes; the notebook's extra one is y = 343, exact P 0.00456). The period does not depend on this: the denominators of all the notebook's 15 fractions divide 6 too.
* a = 11 is fixed as in the notebook. The notebook's np.random.seed(0) and random_coprime draw are not run.

Run: .venv/bin/mojo run -I . examples/algorithms/shor.mojo
example-meta: {"framework": "classiq", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from std.math import cos, floor, pi, sin, sqrt
from qumo.ir.circuit import Circuit
from qumo.sim.runner import evolve, sample
from qumo.algorithms.library import continued_fraction_convergents, order_finding, order_from_phase
from qumo.testing.checks import CheckLog

comptime N_MOD = 21  # the number to factor
comptime A_MULT = 11  # the notebook's fixed co-prime a
comptime N_EVAL = 10  # phase register: twice the 5 work qubits (bit length of 21)
comptime GRID = 1 << N_EVAL  # phase values y / 1024
comptime SHOTS = 2048  # the notebook's shot count (its probability column is count / 2048)


def gcd(a: Int, b: Int) -> Int:
    var x = abs(a)
    var y = abs(b)
    while y != 0:
        var t = x % y
        x = y
        y = t
    return x


def lcm(a: Int, b: Int) -> Int:
    return a // gcd(a, b) * b


def pow_mod(base: Int, e: Int, m: Int) -> Int:
    """Base^e mod m by repeated squaring (e >= 0, m >= 2)."""
    var result = 1 % m
    var b = base % m
    var k = e
    while k > 0:
        if (k & 1) == 1:
            result = (result * b) % m
        b = (b * b) % m
        k = k >> 1
    return result


def brute_order(a: Int, n: Int) -> Int:
    """Smallest r >= 1 with a^r = 1 mod n, found by trying r = 1, 2, ... (gcd(a, n) = 1)."""
    var x = a % n
    var r = 1
    while x != 1:
        x = (x * a) % n
        r += 1
    return r


def grid(v: Float64) -> Int:
    """A phase in [0, 1) as its index y on the grid y / 2^N_EVAL."""
    return Int(floor(v * Float64(GRID) + 0.5))


def post_process(y: Int, n_mod: Int) raises -> Tuple[Int, Int]:
    """The notebook's continued_fraction_post_process on y / 2^N_EVAL: the last convergent p/q with q < n_mod.
    Returns (p, q), or (0, 0) when no convergent qualifies."""
    var convs = continued_fraction_convergents(y, GRID)
    var bp = 0
    var bq = 0
    for t in convs:
        if t[1] < n_mod:
            bp = t[0]
            bq = t[1]
    return (bp, bq)


def fejer_law(r: Int) -> List[Float64]:
    """Exact phase-register law by brute force.

    The work register starts in |1> = (1/sqrt r) sum_s |psi_s>, where |psi_s> is an eigenstate of
    multiplication by a with eigenphase s/r. Phase estimation of phase phi gives outcome y with
    amplitude (1/M) sum_k exp(2 pi i k (phi - y/M)), M = 2^N_EVAL, so P(y) = (1/r) sum_s |amplitude_s(y)|^2.
    """
    var out = List[Float64](length=GRID, fill=0.0)
    var m2 = Float64(GRID) * Float64(GRID)
    for y in range(GRID):
        var total = 0.0
        for s in range(r):
            var phi = Float64(s) / Float64(r)
            var re = 0.0
            var im = 0.0
            for k in range(GRID):
                var ang = 2.0 * pi * Float64(k) * (phi - Float64(y) / Float64(GRID))
                re += cos(ang)
                im += sin(ang)
            total += (re * re + im * im) / m2 / Float64(r)
        out[y] = total
    return out^


def main() raises:
    var log = CheckLog("classiq/shor")
    var r_true = brute_order(A_MULT, N_MOD)
    print("brute-force order of", A_MULT, "mod", N_MOD, "=", r_true)

    # ---- A: the notebook's continued-fraction post-processing on its printed values ----
    # N = 18 example (notebook cells 17-19): five phase results and the fractions the notebook prints
    var ex18: List[Float64] = [0.1953125, 0.39160156, 0.0, 0.59667969, 0.80078125]
    var want18_p: List[Int] = [1, 2, 0, 3, 4]
    var want18_q: List[Int] = [5, 5, 1, 5, 5]
    var bad18 = 0
    var got18 = String()
    for i in range(len(ex18)):
        var y = grid(ex18[i])
        if abs(ex18[i] * Float64(GRID) - Float64(y)) > 1e-5:
            bad18 += 1
        var fr = post_process(y, 18)
        got18 += String(fr[0]) + "/" + String(fr[1]) + " "
        if fr[0] != want18_p[i] or fr[1] != want18_q[i]:
            bad18 += 1
    _ = log.check(
        "notebook N=18 fractions 1/5 2/5 0 3/5 4/5 from QUMO convergents",
        bad18 == 0,
        "got " + got18 + "mismatches=" + String(bad18),
    )

    # N = 21: the 15 outcomes the notebook post-processes (more than 10 counts) and the fractions it prints
    var tab_phase: List[Float64] = [
        0.5, 0.0, 0.8330078125, 0.3330078125, 0.6669921875, 0.1669921875, 0.333984375, 0.666015625,
        0.166015625, 0.833984375, 0.33203125, 0.16796875, 0.83203125, 0.66796875, 0.3349609375,
    ]
    var tab_p: List[Int] = [1, 0, 5, 1, 2, 1, 1, 2, 1, 5, 1, 1, 5, 2, 1]
    var tab_q: List[Int] = [2, 1, 6, 3, 3, 6, 3, 3, 6, 6, 3, 6, 6, 3, 3]
    var bad21 = 0
    for i in range(len(tab_phase)):
        var fr = post_process(grid(tab_phase[i]), N_MOD)
        if fr[0] != tab_p[i] or fr[1] != tab_q[i]:
            bad21 += 1
    _ = log.check(
        "notebook N=21 printed fractions for 15 outcomes",
        bad21 == 0,
        "mismatches=" + String(bad21),
    )

    # ---- B: the QPE circuit, exact phase-register law ----
    var c = order_finding(A_MULT, N_MOD, N_EVAL)
    print("order_finding(11, 21, 10):", c.num_qubits, "qubits (10 phase + 5 work + 3 ancilla),", len(c), "gates")
    var probs = evolve(c).probabilities()
    var p = List[Float64](length=GRID, fill=0.0)
    var total = 0.0
    for i in range(len(probs)):
        p[i & (GRID - 1)] += probs[i]
        total += probs[i]
    _ = log.close("total probability", total, 1.0, 1e-12)

    var exact = fejer_law(r_true)
    var dev = 0.0
    for y in range(GRID):
        dev = max(dev, abs(p[y] - exact[y]))
    print("  max |P_QUMO(y) - P_Fejer(y)| over 1024 outcomes =", dev)
    _ = log.close("phase law equals the Fejer-kernel sum (max deviation)", dev, 0.0, 1e-12)

    # ---- C: the notebook's printed 2048-shot histogram rows against the exact law ----
    var nb_y: List[Int] = [512, 0, 853, 341, 683]
    var nb_count: List[Int] = [366, 324, 254, 239, 230]
    var max_z_nb = 0.0
    for i in range(len(nb_y)):
        var y = nb_y[i]
        var f = Float64(nb_count[i]) / Float64(SHOTS)
        var z = (f - exact[y]) / sqrt(exact[y] * (1.0 - exact[y]) / Float64(SHOTS))
        max_z_nb = max(max_z_nb, abs(z))
        print("  notebook y =", y, " count", nb_count[i], " exact P =", exact[y], " z =", z)
    _ = log.within("notebook 2048-shot histogram rows max |z| against exact law", max_z_nb, 0.0, 4.0)

    # QUMO's own seeded 2048-shot sampler on the phase register, against the analytic law
    var cm = c.copy()
    for j in range(N_EVAL):
        cm.measure(j, j)
    var res = sample(cm, SHOTS, seed=2048)
    var hist = List[Int](length=GRID, fill=0)
    for y in res.memory:
        hist[y] += 1
    var max_z_s = 0.0
    var peaks = 0
    for y in range(GRID):
        if exact[y] > 0.05:
            peaks += 1
            var f = Float64(hist[y]) / Float64(SHOTS)
            var z = (f - exact[y]) / sqrt(exact[y] * (1.0 - exact[y]) / Float64(SHOTS))
            max_z_s = max(max_z_s, abs(z))
    _ = log.check("six peaks above 0.05 in the analytic law", peaks == 6, "peaks=" + String(peaks))
    _ = log.within("QUMO seeded 2048-shot sample max |z| on the six peaks", max_z_s, 0.0, 4.0)

    # ---- D: period from the phase-register outcomes, then the factors ----
    # the notebook keeps outcomes with more than 10 of 2048 sampled counts; here: exact probability above 10/2048
    var threshold = 10.0 / Float64(SHOTS)
    var r_qpe = 1
    var n_freq = 0
    var divides = True
    for y in range(GRID):
        if exact[y] > threshold:
            n_freq += 1
            var fr = post_process(y, N_MOD)
            if fr[1] == 0:
                divides = False
            elif r_true % fr[1] != 0:
                divides = False
            else:
                r_qpe = lcm(r_qpe, fr[1])
    _ = log.check("denominators of the frequent outcomes divide the order", divides, "outcomes=" + String(n_freq))
    _ = log.close("period r from the frequent outcomes (lcm of denominators)", Float64(r_qpe), Float64(r_true), 0.0)

    # order_from_phase (QUMO's classical post-processing) on the notebook's five printed outcomes;
    # y = 0 carries no information and gives 0, as the notebook prints
    var ofp_y: List[Int] = [512, 0, 853, 341, 683]
    var ofp_ok = 0
    for i in range(len(ofp_y)):
        var y = ofp_y[i]
        var want = r_true
        if y == 0:
            want = 0
        if order_from_phase(y, N_EVAL, A_MULT, N_MOD) == want:
            ofp_ok += 1
    _ = log.check("order_from_phase on the printed histogram outcomes", ofp_ok == len(ofp_y), "agree=" + String(ofp_ok) + " of " + String(len(ofp_y)))

    var half = pow_mod(A_MULT, r_qpe // 2, N_MOD)
    var f_minus = gcd(half - 1, N_MOD)
    var f_plus = gcd(half + 1, N_MOD)
    print("  a^(r/2) mod N =", half, " gcd(a^(r/2) - 1, N) =", f_minus, " gcd(a^(r/2) + 1, N) =", f_plus)
    _ = log.check(
        "r is even and a^(r/2) is not -1 mod N",
        r_qpe % 2 == 0 and half != N_MOD - 1,
        "a^(r/2) mod N = " + String(half),
    )
    var factors_ok = f_minus * f_plus == N_MOD and min(f_minus, f_plus) == 3 and max(f_minus, f_plus) == 7
    _ = log.check(
        "factors of 21 from r are 3 and 7 as the notebook prints",
        factors_ok,
        String(f_minus) + " x " + String(f_plus),
    )
    log.finish()
