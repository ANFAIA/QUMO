"""Discrete logarithm: the notebook's two-register circuit for Z_5^x and Z_13^x on QUMO.

Reproduces: Classiq library - Discrete Logarithm
Framework: QUMO - https://github.com/ANFAIA/QUMO
Source: https://github.com/Classiq/classiq-library/blob/main/algorithms/number_theory_and_cryptography/discrete_log/discrete_log.ipynb (Classiq Library, MIT)

The notebook prepares two exponent registers alpha and beta of t qubits in uniform superposition, writes f(alpha, beta) = x^alpha g^beta mod N into a work register, applies inverse QFTs to alpha and beta and measures them. With beta coprime to r = N - 1, the logarithm is alpha beta^-1 mod r. Z_5^x (g = 3, x = 2, r = 4 = 2^2, t = 2) is the notebook's exact case and gives 3. Z_13^x (g = 7, x = 3, r = 12, t = 5) rounds alpha and beta to multiples of 1/r and gives 8.
The checks compare the exact (alpha, beta) law from QUMO's statevector with the law computed by brute force from the definition of the oracle and the inverse QFT, the decoded logarithms with brute-force discrete logs (3 and 8), the clean ancillas and the work register staying in Z_N^x, and seeded samples with the analytic probabilities.

Deviations from the notebook:
* The oracle is built from QUMO's append_controlled_modmul instead of Classiq's modular_multiply_constant_inplace. Each exponent qubit controls one multiplication of the work register by x^(2^j) or g^(2^j) mod N, a permutation of the register (Gray-code chains of multi-controlled X gates with clean ancillas). The work register starts at |1> and only ever holds units mod N, so the states reached agree with the notebook's oracle.
* QUMO's statevector replaces Classiq's cloud simulator. The exact law is checked directly, and sampling uses QUMO's seeded sampler: 4000 shots for Z_5^x (the notebook's count) and 10000 for Z_13^x. The notebook's job links and sample tables are not reproduced, only its printed results.
* The Z_13^x registers are integers mu, nu in [0, 32) with value mu / 32, the notebook's fixed-point registers with 5 fraction bits.
* Rounding takes the nearest integer with ties to even, as numpy's round does.
* The notebook's Z_5^x cell computes beta alpha^-1 mod 4. This file uses alpha beta^-1 mod r for both instances. Both give 3 here, because 3 is its own inverse mod 4.
* The notebook's heuristic sinc-kernel plots are not drawn, and its Diffie-Hellman text is not reproduced. The exact law is checked instead.

Run: .venv/bin/mojo run -I . examples/frameworks/classiq/discrete_log.mojo
example-meta: {"framework": "classiq", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from std.math import cos, pi, sin, sqrt
from qumo.ir.circuit import Circuit
from qumo.sim.runner import evolve, sample
from qumo.algorithms.library import append_controlled_modmul, iqft, num_mcz_ancillas
from qumo.testing.checks import CheckLog


def bit_width(n: Int) -> Int:
    """Ceil(log2 n): the smallest b with 2^b >= n."""
    var b = 0
    while (1 << b) < n:
        b += 1
    return b


def pow_mod(base: Int, e: Int, m: Int) -> Int:
    """Base^e mod m by repeated squaring (e >= 0, m >= 2)."""
    var result = 1
    var b = base % m
    var k = e
    while k > 0:
        if (k & 1) == 1:
            result = (result * b) % m
        b = (b * b) % m
        k = k >> 1
    return result


def gcd(a: Int, b: Int) -> Int:
    var x = abs(a)
    var y = abs(b)
    while y != 0:
        var t = x % y
        x = y
        y = t
    return x


def inv_mod(b: Int, r: Int) -> Int:
    """The inverse of b mod r by search (gcd(b, r) = 1), or -1."""
    for k in range(r):
        if (b * k) % r == 1:
            return k
    return -1


def round_half_even(num: Int, den: Int) -> Int:
    """The integer nearest to num / den (num >= 0, den > 0), ties to even as numpy's round."""
    var q = num // den
    var rem = num - q * den
    if 2 * rem > den or (2 * rem == den and q % 2 == 1):
        q += 1
    return q


def decode(mu: Int, nu: Int, t: Int, r: Int) -> Int:
    """The notebook's post-processing of one outcome (alpha = mu / T, beta = nu / T, T = 2^t):
    round alpha * r and beta * r to integers a and b, then a b^-1 mod r when gcd(b, r) = 1.
    Returns -1 when b is not coprime to r (the notebook discards those outcomes)."""
    var T = 1 << t
    var a = round_half_even(mu * r, T) % r
    var b = round_half_even(nu * r, T) % r
    if gcd(b, r) != 1:
        return -1
    return (a * inv_mod(b, r)) % r


def brute_dlog(n_mod: Int, g: Int, x: Int) -> Int:
    """The least s >= 0 with g^s = x mod N, by search; -1 if there is none."""
    for s in range(n_mod - 1):
        if pow_mod(g, s, n_mod) == x % n_mod:
            return s
    return -1


def dlog_circuit(n_mod: Int, g: Int, x: Int, t: Int, measure: Bool) raises -> Circuit:
    """The notebook's circuit: H on alpha and beta, f(alpha, beta) = x^alpha g^beta mod N into the work
    register (prepared in |1>), inverse QFTs on alpha and beta, and optionally a measurement of alpha and beta.
    Layout: alpha = qubits 0..t-1, beta = t..2t-1, work = 2t..2t+R-1 with R = ceil(log2 N), clean ancillas after.
    Clbit j = alpha bit j and clbit t+j = beta bit j, so a shot packs as mu + nu * 2^t."""
    var R = bit_width(n_mod)
    var nanc = num_mcz_ancillas(R + 1)
    var nclb = 2 * t if measure else 0
    var c = Circuit(2 * t + R + nanc, nclb, name="discrete_log")
    var alpha = List[Int]()
    var beta = List[Int]()
    var work = List[Int]()
    var anc = List[Int]()
    for j in range(t):
        alpha.append(j)
        beta.append(t + j)
    for k in range(R):
        work.append(2 * t + k)
    for k in range(nanc):
        anc.append(2 * t + R + k)
    for j in range(t):
        c.h(alpha[j])
        c.h(beta[j])
    c.x(work[0])  # work register |1>
    # alpha bit j controls the multiplication by x^(2^j) mod N, beta bit j by g^(2^j) mod N
    var mult = x % n_mod
    for j in range(t):
        if mult != 1:
            append_controlled_modmul(c, alpha[j], work, mult, n_mod, anc)
        mult = (mult * mult) % n_mod
    mult = g % n_mod
    for j in range(t):
        if mult != 1:
            append_controlled_modmul(c, beta[j], work, mult, n_mod, anc)
        mult = (mult * mult) % n_mod
    c.compose(iqft(t), 0)
    c.compose(iqft(t), t)
    if measure:
        for j in range(t):
            c.measure(j, j)
            c.measure(t + j, t + j)
    return c^


def fourier_law(n_mod: Int, x: Int, g: Int, t: Int) -> List[Float64]:
    """The exact law of (mu, nu) with the work register traced out, index mu * T + nu, from the definition:
    the amplitude of |mu>|nu>|v> is T^-2 sum of exp(-2 pi i (a mu + b nu) / T) over the (a, b) with
    x^a g^b = v mod N, so P(mu, nu) = sum over v of |amplitude|^2. Brute force, independent of the circuit."""
    var T = 1 << t
    var law = List[Float64](length=T * T, fill=0.0)
    var re = List[Float64](length=T * T, fill=0.0)
    var im = List[Float64](length=T * T, fill=0.0)
    var val = List[Int](length=T * T, fill=0)
    for a in range(T):
        for b in range(T):
            val[a * T + b] = (pow_mod(x, a, n_mod) * pow_mod(g, b, n_mod)) % n_mod
    var T4 = Float64(T * T) * Float64(T * T)
    for v in range(n_mod):
        for k in range(T * T):
            re[k] = 0.0
            im[k] = 0.0
        var used = False
        for a in range(T):
            for b in range(T):
                if val[a * T + b] != v:
                    continue
                used = True
                for mu in range(T):
                    for nu in range(T):
                        var ang = -2.0 * pi * Float64((a * mu + b * nu) % T) / Float64(T)
                        re[mu * T + nu] += cos(ang)
                        im[mu * T + nu] += sin(ang)
        if used:
            for k in range(T * T):
                law[k] += (re[k] * re[k] + im[k] * im[k]) / T4
    return law^


def marginal_ab(probs: List[Float64], t: Int) -> List[Float64]:
    """P(mu, nu) with the work and ancilla registers traced out, index mu * T + nu."""
    var T = 1 << t
    var mask = T - 1
    var out = List[Float64](length=T * T, fill=0.0)
    for i in range(len(probs)):
        out[(i & mask) * T + ((i >> t) & mask)] += probs[i]
    return out^


def work_and_ancilla(probs: List[Float64], t: Int, R: Int, n_mod: Int) -> Tuple[Float64, Float64]:
    """(probability that the ancillas are |0>, probability that the work register holds a unit of Z_N)."""
    var wmask = (1 << R) - 1
    var anc_shift = 2 * t + R
    var clean = 0.0
    var units = 0.0
    for i in range(len(probs)):
        var w = (i >> (2 * t)) & wmask
        if (i >> anc_shift) == 0:
            clean += probs[i]
        if w >= 1 and w < n_mod and gcd(w, n_mod) == 1:
            units += probs[i]
    return (clean, units)


def decoded_mass(law: List[Float64], t: Int, r: Int, target: Int) -> Tuple[Float64, Float64]:
    """(P(beta coprime to r), P(decoded logarithm = target)) for a law with index mu * T + nu."""
    var T = 1 << t
    var coprime = 0.0
    var hit = 0.0
    for mu in range(T):
        for nu in range(T):
            var d = decode(mu, nu, t, r)
            if d >= 0:
                coprime += law[mu * T + nu]
                if d == target:
                    hit += law[mu * T + nu]
    return (coprime, hit)


def max_dev(a: List[Float64], b: List[Float64]) -> Float64:
    var d = 0.0
    for k in range(len(a)):
        d = max(d, abs(a[k] - b[k]))
    return d


def sample_counts(n_mod: Int, g: Int, x: Int, t: Int, shots: Int, seed: UInt64) raises -> List[Int]:
    """Seeded shots of the measured (alpha, beta), as a count per index mu * T + nu."""
    var T = 1 << t
    var mask = T - 1
    var res = sample(dlog_circuit(n_mod, g, x, t, True), shots, seed=seed)
    var counts = List[Int](length=T * T, fill=0)
    for m in res.memory:
        var mu = m & mask
        var nu = (m >> t) & mask
        counts[mu * T + nu] += 1
    return counts^


def exact_law_gap(counts: List[Int], analytic: List[Float64], shots: Int) -> Tuple[Int, Float64]:
    """(shots on outcomes the analytic law forbids, largest |z| over the outcomes it allows)."""
    var forbidden = 0
    var zmax = 0.0
    var n = Float64(shots)
    for k in range(len(analytic)):
        if analytic[k] <= 1e-12:
            forbidden += counts[k]
        else:
            var p = analytic[k]
            var z = abs(Float64(counts[k]) / n - p) / sqrt(p * (1.0 - p) / n)
            zmax = max(zmax, z)
    return (forbidden, zmax)


def main() raises:
    var log = CheckLog("classiq/discrete_log")

    # ---------------- A: Z_5^x, g = 3, x = 2, r = 4 (T = r, the notebook's exact case) ----------------
    var n_a = 5
    var g_a = 3
    var x_a = 2
    var t_a = 2
    var r_a = n_a - 1
    var R_a = bit_width(n_a)
    var s_a = brute_dlog(n_a, g_a, x_a)
    var qc_a = dlog_circuit(n_a, g_a, x_a, t_a, False)
    print("A: Z_5^x, g = 3, x = 2, t = 2:", qc_a.num_qubits, "qubits,", len(qc_a), "gates,", qc_a.num_2q_gates(), "two-qubit gates")
    var probs_a = evolve(qc_a).probabilities()
    var law_a = marginal_ab(probs_a, t_a)
    var ref_a = fourier_law(n_a, x_a, g_a, t_a)
    var masses_a = work_and_ancilla(probs_a, t_a, R_a, n_a)
    var dm_a = decoded_mass(law_a, t_a, r_a, s_a)
    var dr_a = decoded_mass(ref_a, t_a, r_a, s_a)
    print("  decoded logarithm 3: P(beta coprime to 4) =", dm_a[0], " P(log = 3) =", dm_a[1])

    _ = log.check("A brute-force discrete log of 2 base 3 mod 5 is the notebook's 3", s_a == 3, "s=" + String(s_a))
    _ = log.check("A notebook check 3^3 mod 5 = 2", pow_mod(g_a, 3, n_a) == x_a)
    _ = log.close("A exact (alpha, beta) law equals the brute-force Fourier law (max deviation)", max_dev(law_a, ref_a), 0.0, 1e-10)
    _ = log.close("A ancillas return to |0> (probability)", masses_a[0], 1.0, 1e-10)
    _ = log.close("A work register stays in Z_5^x (probability)", masses_a[1], 1.0, 1e-10)
    var phi_a = 0
    for k in range(r_a):
        if gcd(k, r_a) == 1:
            phi_a += 1
    _ = log.close("A P(beta coprime to 4) = phi(4)/4", dm_a[0], Float64(phi_a) / Float64(r_a), 1e-10)
    _ = log.close("A P(decoded log = 3) matches the brute-force law", dm_a[1], dr_a[1], 1e-10)
    _ = log.close("A every coprime outcome decodes to 3 (conditional probability)", dm_a[1] / dm_a[0], 1.0, 1e-10)

    var shots_a = 4000
    var counts_a = sample_counts(n_a, g_a, x_a, t_a, shots_a, UInt64(5))
    var gap_a = exact_law_gap(counts_a, ref_a, shots_a)
    var coprime_a = 0
    var good_a = 0
    for k in range(len(ref_a)):
        var mu = k // (1 << t_a)
        var nu = k % (1 << t_a)
        var d = decode(mu, nu, t_a, r_a)
        if d >= 0:
            coprime_a += counts_a[k]
            if d == s_a:
                good_a += counts_a[k]
    print("  sampled", shots_a, "shots: coprime beta", coprime_a, " decoded 3:", good_a)
    _ = log.check("A sampled shots only land on the analytic support", gap_a[0] == 0, "forbidden=" + String(gap_a[0]))
    _ = log.within("A sampled frequencies, max |z| over the 4 support outcomes", gap_a[1], 0.0, 4.0)
    _ = log.check("A every sampled coprime outcome decodes to 3", coprime_a > 0 and good_a == coprime_a, "decoded=" + String(good_a) + " of " + String(coprime_a))

    # ---------------- B: Z_13^x, g = 7, x = 3, r = 12, t = 5 (T = 32, the notebook's second example) ----------------
    var n_b = 13
    var g_b = 7
    var x_b = 3
    var t_b = 5
    var r_b = n_b - 1
    var R_b = bit_width(n_b)
    var s_b = brute_dlog(n_b, g_b, x_b)
    var qc_b = dlog_circuit(n_b, g_b, x_b, t_b, False)
    print("B: Z_13^x, g = 7, x = 3, t = 5:", qc_b.num_qubits, "qubits,", len(qc_b), "gates,", qc_b.num_2q_gates(), "two-qubit gates")
    var probs_b = evolve(qc_b).probabilities()
    var law_b = marginal_ab(probs_b, t_b)
    var ref_b = fourier_law(n_b, x_b, g_b, t_b)
    var masses_b = work_and_ancilla(probs_b, t_b, R_b, n_b)
    var dm_b = decoded_mass(law_b, t_b, r_b, s_b)
    var dr_b = decoded_mass(ref_b, t_b, r_b, s_b)
    print("  P(beta coprime to 12) =", dm_b[0], " P(decoded log = 8, beta coprime) =", dm_b[1],
          " conditional =", dm_b[1] / dm_b[0])

    _ = log.check("B brute-force discrete log of 3 base 7 mod 13 is the notebook's 8", s_b == 8, "s=" + String(s_b))
    _ = log.check("B notebook check 7^8 mod 13 = 3", pow_mod(g_b, 8, n_b) == x_b)
    _ = log.close("B exact (alpha, beta) law equals the brute-force Fourier law (max deviation)", max_dev(law_b, ref_b), 0.0, 1e-10)
    _ = log.close("B ancillas return to |0> (probability)", masses_b[0], 1.0, 1e-10)
    _ = log.close("B work register stays in Z_13^x (probability)", masses_b[1], 1.0, 1e-10)
    _ = log.close("B P(beta coprime to 12) matches the brute-force law", dm_b[0], dr_b[0], 1e-10)
    _ = log.close("B P(decoded log = 8, beta coprime) matches the brute-force law", dm_b[1], dr_b[1], 1e-10)

    # the most probable coprime outcomes (tied in the analytic law) decode to the notebook's 8
    var best = 0.0
    for mu in range(1 << t_b):
        for nu in range(1 << t_b):
            if decode(mu, nu, t_b, r_b) >= 0:
                best = max(best, law_b[mu * (1 << t_b) + nu])
    var top = 0
    var top_all_8 = True
    for mu in range(1 << t_b):
        for nu in range(1 << t_b):
            var d = decode(mu, nu, t_b, r_b)
            if d >= 0 and law_b[mu * (1 << t_b) + nu] >= best - 1e-9:
                top += 1
                if d != 8:
                    top_all_8 = False
    print("  most probable coprime outcomes:", top, "with P =", best, "; all decode to 8:", top_all_8)
    _ = log.check("B the most probable coprime outcomes decode to the notebook's 8", top_all_8 and top > 0, "count=" + String(top))

    var shots_b = 10000
    var counts_b = sample_counts(n_b, g_b, x_b, t_b, shots_b, UInt64(13))
    var gap_b = exact_law_gap(counts_b, ref_b, shots_b)
    var coprime_b = 0
    var good_b = 0
    for mu in range(1 << t_b):
        for nu in range(1 << t_b):
            var k = mu * (1 << t_b) + nu
            var d = decode(mu, nu, t_b, r_b)
            if d >= 0:
                coprime_b += counts_b[k]
                if d == 8:
                    good_b += counts_b[k]
    var p_cop = dr_b[0]
    var p_cond = dr_b[1] / dr_b[0]
    var f_cop = Float64(coprime_b) / Float64(shots_b)
    var f_cond = Float64(good_b) / Float64(coprime_b)
    var z_cop = abs(f_cop - p_cop) / sqrt(p_cop * (1.0 - p_cop) / Float64(shots_b))
    var z_cond = abs(f_cond - p_cond) / sqrt(p_cond * (1.0 - p_cond) / Float64(coprime_b))
    print("  sampled", shots_b, "shots: coprime beta", coprime_b, " decoded 8:", good_b,
          " frequencies", f_cop, f_cond, " analytic", p_cop, p_cond)
    _ = log.within("B sampled coprime-beta frequency, |z| against the analytic value", z_cop, 0.0, 4.0)
    _ = log.within("B sampled conditional frequency of log 8, |z| against the analytic value", z_cond, 0.0, 4.0)
    _ = log.check("B sampled shots only land on outcomes the analytic law allows", gap_b[0] == 0, "forbidden=" + String(gap_b[0]))

    log.finish()
