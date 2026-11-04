"""Quantum counting: the notebook's (a + b) <= 2 oracle, counted by QPE and by iterative amplitude estimation.

Reproduces: Classiq library - Quantum Counting Using the Iterative Quantum Amplitude Estimation Algorithm
Source: https://github.com/Classiq/classiq-library/blob/main/algorithms/amplitude_amplification_and_estimation/quantum_counting/quantum_counting.ipynb (Classiq Library, MIT)

The notebook counts the six solutions of (a + b) <= 2 over 2-bit a and b (16 inputs). QPE with 5 phase qubits on the Grover operator reads out the two equal peaks 7/32 and 25/32 (the notebook's 0.21875), and 16 sin^2(pi 7/32) = 6.4393. Iterative amplitude estimation (IQAE) on an indicator qubit gives the count with a confidence interval. The checks compare the exact Grover powers with sin^2((2m+1) theta), the exact QPE distribution with the analytic QPE kernel, the sampled readouts with the same analytic values, and the IQAE count with the exact count 6 at the notebook's own tolerance 1/15.

Deviations from the notebook:
* The notebook calls Classiq's IQAE class, which QUMO does not have. The loop is written in this file, following Grinko et al. (npj Quantum Inf. 7, 52, 2021), the paper the notebook cites: the Grover power is the largest one whose angle window has no turning point of sin^2, each round turns the observed frequency into an angle window with Hoeffding bounds at confidence alpha / 2^(k+1), and each round uses 2048 shots as the notebook does. The Grover powers depend on the sampled outcomes, and QUMO's sampler is not Classiq's, so the notebook's printed counts (1325/723, 697/1351) and its estimate 5.9748 are not reproduced; the IQAE check uses the exact count 6. In the checked run the second round uses 6 Grover iterations where the notebook uses 5.
* Sign: QUMO's grover_diffusion() is I - 2|s><s|, minus the textbook reflection 2|s><s| - I. The notebook's grover_operator is the textbook -A S0 A^dag S_chi, so the QPE unitary carries global phase pi. Without it the readouts move to 9/32 and 23/32 and the count to 9.56 instead of 6.44.
* QPE is built as qumo.algorithms.library.qpe() builds it, except that the controlled powers use controlled_by(): controlled() rejects the CCX gates of the MCZ ladder inside the Grover operator.
* The two QPE peaks 7/32 and 25/32 have exactly equal probability (the state is real), so the notebook's pick of 0.21875 is one of two equal bins; the checks accept either.
* QPE sampling uses 4000 shots (the notebook does not print its count), and the IQAE and QPE seeds are QUMO's.
* No Classiq synthesis, no cloud link and no printed circuit link.

Run: .venv/bin/mojo run -I . examples/frameworks/classiq/quantum_counting.mojo
example-meta: {"framework": "classiq", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from std.math import asin, cos, floor, log, pi, sin, sqrt
from qumo.ir.circuit import Circuit
from qumo.ir.mc import controlled_by
from qumo.algorithms.library import append_mcx, append_mcz, bitstring_to_int, grover_diffusion, iqft, phase_oracle
from qumo.sim.runner import bitstring, evolve, sample
from qumo.testing.checks import CheckLog


comptime NSEARCH = 4  # x = a + 4 b: a on qubits 0, 1 and b on qubits 2, 3
comptime IND = 4  # indicator qubit: ind = (a + b <= 2)
comptime NQ = 7  # IQAE register: 5 state qubits, then 2 clean ancillas for the 5-qubit MCZ / MCX
comptime QPE_BITS = 5  # phase qubits, as in the notebook
comptime QPE_SHOTS = 4000
comptime IQAE_SHOTS = 2048  # per round, as in the notebook's counts (each sums to 2048)
comptime IQAE_ALPHA = 0.01  # failure probability of the IQAE interval
comptime IQAE_EPS = 0.03125  # 1/32, the notebook's epsilon (accuracy of a)
comptime MAX_ROUNDS = 12


def bit(x: Int, i: Int) -> Int:
    return (x >> i) & 1


def sin2(x: Float64) -> Float64:
    var s = sin(x)
    return s * s


def predicate(x: Int) -> Bool:
    """The notebook's arith_equation: a + b <= 2 for x = a + 4 b, with a and b the 2-bit values."""
    var a = x & 3
    var b = x >> 2
    return a + b <= 2


def solutions() -> List[Int]:
    var out = List[Int]()
    for x in range(16):
        if predicate(x):
            out.append(x)
    return out^


# ---------------------------------------------------------------------------
# Circuits
# ---------------------------------------------------------------------------
def ind_prep() raises -> Circuit:
    """A: H on a and b, then ind ^= (a + b <= 2) as one multi-controlled X per solution (X-conjugated on zero bits)."""
    var c = Circuit(NQ, 0, "iqae_state_prep")
    var controls = List[Int]()
    for q in range(NSEARCH):
        c.h(q)
        controls.append(q)
    var anc = List[Int]()
    anc.append(5)
    anc.append(6)
    for x in solutions():
        for q in range(NSEARCH):
            if bit(x, q) == 0:
                c.x(q)
        append_mcx(c, controls, IND, anc)
        for q in range(NSEARCH):
            if bit(x, q) == 0:
                c.x(q)
    return c^


def grover_iterate(prep: Circuit) raises -> Circuit:
    """Q = -A S0 A^dag S_chi on the 5 state qubits. S0 flips the phase of |0...0> on them, the ancillas stay |0>."""
    var c = Circuit(NQ, 0, "iqae_grover")
    var state = List[Int]()
    for q in range(5):
        state.append(q)
    var anc = List[Int]()
    anc.append(5)
    anc.append(6)
    c.z(IND)  # S_chi: -1 when the indicator is 1
    c.compose(prep.inverse())  # A^dag
    for q in range(5):
        c.x(q)
    append_mcz(c, state, anc)  # S0 = X^n (multi-controlled Z) X^n
    for q in range(5):
        c.x(q)
    c.compose(prep)  # A
    c.global_phase = pi  # textbook sign; probabilities do not depend on it
    return c^


def round_circuit(prep: Circuit, iterate: Circuit, m: Int, measure: Bool) raises -> Circuit:
    """A followed by m Grover iterations; optionally measure the indicator into clbit 0."""
    var c = Circuit(NQ, 1 if measure else 0, "iqae_round")
    c.compose(prep)
    for _ in range(m):
        c.compose(iterate)
    if measure:
        c.measure(IND, 0)
    return c^


def exact_good_probability(prep: Circuit, iterate: Circuit, m: Int) raises -> Float64:
    """Exact probability that the indicator reads 1 after A and m Grover iterations."""
    var probs = evolve(round_circuit(prep, iterate, m, False)).probabilities()
    var good = 0.0
    for i in range(len(probs)):
        if bit(i, IND) == 1:
            good += probs[i]
    return good


def qpe_unitary() raises -> Circuit:
    """The notebook's grover_operator, -A S0 A^dag S_chi with A = H^4, on the 4 search qubits and the MCZ ancilla.

    QUMO's phase_oracle() is S_chi and grover_diffusion() is A S0 A^dag = I - 2|s><s|, so their product is minus the
    textbook operator. The global phase pi restores the textbook sign, which the controlled powers in qpe() see."""
    var c = Circuit(NSEARCH + 1, 0, "grover_operator")
    c.compose(phase_oracle(NSEARCH, solutions()))
    c.compose(grover_diffusion(NSEARCH))
    c.global_phase = pi
    return c^


# ---------------------------------------------------------------------------
# QPE: exact readout distribution and its analytic reference
# ---------------------------------------------------------------------------
def build_qpe(u: Circuit, measure: Bool) raises -> Circuit:
    """QPE of u from H^4 |0>|0>: eval qubits 0..QPE_BITS-1, system qubits QPE_BITS.. (u has 5 qubits).

    The construction is qumo.algorithms.library.qpe(): H on the eval register, controlled-u^(2^j) from eval
    qubit j, then the inverse QFT. The controlled powers use controlled_by() because controlled() rejects
    the CCX gates of the MCZ ladder inside the Grover operator. The state preparation H^4 acts on the search
    qubits of the system register, so the MCZ ancilla stays |0>."""
    var m = u.num_qubits
    var c = Circuit(QPE_BITS + m, QPE_BITS if measure else 0, "qpe_grover")
    var sys_map = List[Int]()
    for q in range(m):
        sys_map.append(QPE_BITS + q)
    for q in range(NSEARCH):
        c.h(QPE_BITS + q)
    for j in range(QPE_BITS):
        c.h(j)
    for j in range(QPE_BITS):
        var ctrl = List[Int]()
        ctrl.append(m)
        var cu = controlled_by(u.power(1 << j), ctrl)
        var qmap = sys_map.copy()
        qmap.append(j)
        c.compose_mapped(cu, qmap)
    c.compose(iqft(QPE_BITS))
    if measure:
        for j in range(QPE_BITS):
            c.measure(j, j)
    return c^


def qpe_readouts(u: Circuit) raises -> List[Float64]:
    """Exact distribution of the QPE_BITS-bit readout after QPE of u, summed over the system register."""
    var probs = evolve(build_qpe(u, False)).probabilities()
    var out = List[Float64](length=1 << QPE_BITS, fill=0.0)
    var mask = (1 << QPE_BITS) - 1
    for i in range(len(probs)):
        out[i & mask] += probs[i]
    return out^


def qpe_kernel(phi: Float64, k: Int) -> Float64:
    """Ideal readout probability |(1/N) sum_j exp(2 pi i j (phi - k/N))|^2 for eigenphase phi, N = 2^QPE_BITS."""
    var nn = 1 << QPE_BITS
    var d = phi - Float64(k) / Float64(nn)
    var re = 0.0
    var im = 0.0
    for j in range(nn):
        var ang = 2.0 * pi * Float64(j) * d
        re += cos(ang)
        im += sin(ang)
    return (re * re + im * im) / Float64(nn * nn)


def analytic_readout(k: Int) -> Float64:
    """The state A|0> has weight 1/2 on each eigenvector of Q, with eigenphases +/- theta/pi, theta = asin(sqrt(6/16))/pi."""
    var phi = asin(sqrt(6.0 / 16.0)) / pi
    return 0.5 * (qpe_kernel(phi, k) + qpe_kernel(-phi, k))


# ---------------------------------------------------------------------------
# IQAE: Grinko et al. (2021) with Hoeffding intervals on the angle theta, a = sin^2(theta)
# ---------------------------------------------------------------------------
def turning_free(lo: Float64, hi: Float64) -> Bool:
    """True when no multiple of pi/2 lies strictly inside (lo, hi): sin^2 is monotone on the window."""
    var q = 0.5 * pi
    return (floor(lo / q) + 1.0) * q >= hi


def choose_power(tl: Float64, tu: Float64) -> Int:
    """Largest Grover power m with (2m+1)(tu - tl) <= pi/2 whose window (2m+1)[tl, tu] is turning-free. Power 0 always qualifies."""
    var width = tu - tl
    var m = Int(floor(pi / (4.0 * width) - 0.5))
    while m > 0:
        if turning_free(Float64(2 * m + 1) * tl, Float64(2 * m + 1) * tu):
            return m
        m -= 1
    return 0


def window_point(p: Float64, lo: Float64, hi: Float64) -> Float64:
    """The point of [lo, hi] where sin^2 equals p, clipped to the window (sin^2 is monotone there)."""
    var f_lo = sin2(lo)
    var f_hi = sin2(hi)
    var up = f_hi >= f_lo
    if up and p <= f_lo:
        return lo
    if up and p >= f_hi:
        return hi
    if (not up) and p >= f_lo:
        return lo
    if (not up) and p <= f_hi:
        return hi
    var a = lo
    var b = hi
    for _ in range(100):
        var mid = 0.5 * (a + b)
        if (sin2(mid) < p) == up:
            a = mid
        else:
            b = mid
    return 0.5 * (a + b)


def iqae(prep: Circuit, iterate: Circuit, eps: Float64, alpha: Float64, seed: UInt64,
         mut powers: List[Int], mut hits: List[Int]) raises -> Tuple[Float64, Float64]:
    """Iterate until the half-width of a's interval is at most eps. Returns the final window [tl, tu] for theta.

    Round k runs A and Q^m (m = powers[k]), measures the indicator with IQAE_SHOTS shots, and intersects the window
    with the angles whose sin^2 lies in the Hoeffding interval at confidence alpha / 2^(k+1)."""
    var tl = 0.0
    var tu = 0.5 * pi
    var k = 0
    while k < MAX_ROUNDS:
        if 0.5 * (sin2(tu) - sin2(tl)) <= eps:
            break
        var m = choose_power(tl, tu)
        var res = sample(round_circuit(prep, iterate, m, True), IQAE_SHOTS, seed + UInt64(k))
        var h = res.counts.get("1", 0)
        powers.append(m)
        hits.append(h)
        var alpha_k = alpha / Float64(1 << (k + 1))
        var d = sqrt(log(2.0 / alpha_k) / (2.0 * Float64(IQAE_SHOTS)))
        var p_hat = Float64(h) / Float64(IQAE_SHOTS)
        var p_lo = max(p_hat - d, 0.0)
        var p_hi = min(p_hat + d, 1.0)
        var scale = Float64(2 * m + 1)
        var x_lo = scale * tl
        var x_hi = scale * tu
        var xa = window_point(p_lo, x_lo, x_hi)
        var xb = window_point(p_hi, x_lo, x_hi)
        tl = max(tl, min(xa, xb) / scale)
        tu = min(tu, max(xa, xb) / scale)
        k += 1
    return (tl, tu)


def main() raises:
    var log = CheckLog("classiq/quantum_counting")
    var theta_a = asin(sqrt(6.0 / 16.0))  # sin^2(theta_a) = a = 6/16

    # ---- A: state preparation marks (a + b <= 2) on the indicator ----
    var prep = ind_prep()
    var probs0 = evolve(prep).probabilities()
    var p_good = 0.0
    var worst_a = 0.0
    for i in range(len(probs0)):
        var x = i & 15
        var ind = bit(i, IND)
        var anc = i >> 5
        if ind == 1:
            p_good += probs0[i]
        var want = 0.0
        if anc == 0 and ((ind == 1) == predicate(x)):
            want = 1.0 / 16.0
        worst_a = max(worst_a, abs(probs0[i] - want))
    print("solutions of (a + b) <= 2 among 16 inputs:", len(solutions()))
    _ = log.close("state preparation gives P(indicator = 1) = 6/16", p_good, 0.375, 1e-12)
    _ = log.close("state preparation matches the truth table on all 128 basis states", worst_a, 0.0, 1e-12)

    # ---- B: Grover iterate: Q^m A gives P(indicator = 1) = sin^2((2m+1) theta) ----
    var iterate = grover_iterate(prep)
    var worst_b = 0.0
    for m in range(6):
        var got = exact_good_probability(prep, iterate, m)
        var want = sin2(Float64(2 * m + 1) * theta_a)
        worst_b = max(worst_b, abs(got - want))
    _ = log.close("Q^m A gives P(indicator = 1) = sin^2((2m+1) theta) for m = 0 to 5", worst_b, 0.0, 1e-9)
    var probs5 = evolve(round_circuit(prep, iterate, 5, False)).probabilities()
    var leak = 0.0
    for i in range(len(probs5)):
        if (i >> 5) != 0:
            leak += probs5[i]
    _ = log.close("ancillas back in |0> after five Grover iterations", leak, 0.0, 1e-12)

    # ---- C: QPE of the Grover operator with 5 phase qubits ----
    var u = qpe_unitary()
    var readout = qpe_readouts(u)
    var worst_c = 0.0
    for k in range(1 << QPE_BITS):
        worst_c = max(worst_c, abs(readout[k] - analytic_readout(k)))
    _ = log.close("QPE readout distribution matches the analytic kernel on all 32 bins", worst_c, 0.0, 1e-9)

    var first = 0
    for k in range(1 << QPE_BITS):
        if readout[k] > readout[first]:
            first = k
    var second = -1
    for k in range(1 << QPE_BITS):
        if k != first and (second < 0 or readout[k] > readout[second]):
            second = k
    print("QPE two most probable readouts:", first, second, "probabilities", readout[first], readout[second])
    _ = log.check("two most probable QPE readouts are 7 and 25 (phases 7/32 and 25/32)",
                  (first == 7 and second == 25) or (first == 25 and second == 7),
                  "readouts=" + String(first) + "," + String(second))
    var count_qpe = 16.0 * sin2(pi * Float64(first) / Float64(1 << QPE_BITS))
    print("QPE solution count from the top readout:", count_qpe)
    _ = log.close("QPE count 16 sin^2(pi k / 32) at the top readout matches the notebook's 6.4393", count_qpe, 6.439277423870974, 1e-9)

    var meas = sample(build_qpe(u, True), QPE_SHOTS, seed=UInt64(41))
    var top_readout = bitstring_to_int(meas.most_frequent())
    _ = log.check("sampled QPE most frequent readout is 7 or 25", top_readout == 7 or top_readout == 25,
                  "readout=" + String(top_readout))
    var want7 = analytic_readout(7)
    var f7 = meas.probability(bitstring(7, QPE_BITS))
    var sigma7 = sqrt(want7 * (1.0 - want7) / Float64(QPE_SHOTS))
    _ = log.within("sampled readout 7 frequency within 4 sigma of the analytic probability", f7,
                   want7 - 4.0 * sigma7, want7 + 4.0 * sigma7)

    # ---- D: IQAE on the indicator qubit ----
    var powers = List[Int]()
    var hits = List[Int]()
    var window = iqae(prep, iterate, IQAE_EPS, IQAE_ALPHA, UInt64(2026), powers, hits)
    var tl = window[0]
    var tu = window[1]
    var est = 16.0 * sin2(0.5 * (tl + tu))
    var ci_lo = 16.0 * sin2(tl)
    var ci_hi = 16.0 * sin2(tu)
    for i in range(len(powers)):
        print("IQAE round", i, "Grover powers", powers[i], "indicator ones", hits[i], "of", IQAE_SHOTS)
    print("IQAE solution count", est, "interval", ci_lo, ci_hi)

    var f0 = Float64(hits[0]) / Float64(IQAE_SHOTS)
    var sigma0 = sqrt(0.375 * 0.625 / Float64(IQAE_SHOTS))
    _ = log.within("IQAE round 0 frequency within 4 sigma of a = 6/16", f0, 0.375 - 4.0 * sigma0, 0.375 + 4.0 * sigma0)
    _ = log.check("IQAE first round uses no Grover iteration, as in the notebook", powers[0] == 0,
                  "powers0=" + String(powers[0]))
    _ = log.close("IQAE count within the notebook tolerance 1/15 of the exact count 6", est, 6.0, 1.0 / 15.0)
    _ = log.check("IQAE confidence interval contains the exact count 6", ci_lo <= 6.0 and 6.0 <= ci_hi)

    log.finish()
