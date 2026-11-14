"""European call option pricing by amplitude estimation on a log-normal asset.

Reproduces: Classiq library - Estimating European Option Price Using Amplitude Estimation
Framework: QUMO - https://github.com/ANFAIA/QUMO
Source: https://github.com/Classiq/classiq-library/blob/main/applications/finance/option_pricing/option_pricing.ipynb (Classiq Library, MIT)

The notebook discretises a log-normal asset price (mu = 0.7, sigma = 0.13, 5 qubits, truncated at mean +- 3 stddev), loads it with a state preparation, loads the call payoff max(S - K, 0) / scaling_factor (K = 1.9) as the amplitude of an indicator qubit, and estimates P(ind = 1) = E[payoff] / scaling_factor with iterative amplitude estimation. It prints a measured payoff 0.17682654 and the exact payoff 0.17680663. QUMO builds the same A on 6 qubits (indicator = qubit 0, asset bit k = qubit 1 + k). Checks: the printed grid statistics (mean, variance, stddev, low, high), the notebook's exact expected payoff, the joint law of (asset, ind) after A against p(x) (S(x) - K)+ / scaling_factor, the exact P(ind = 1) after k Grover iterations of Q = -A S_0 A^dag S_good against sin^2((2k + 1) theta) for k = 0, 1, 2, 4, 8, and a seeded sampled amplitude estimate (maximum likelihood over those depths) against the exact payoff within 3 standard errors (Fisher information 4 x shots x (2k+1)^2 per depth), and against the notebook's printed measured payoff within the notebook's own assertion tolerance (5 x its confidence-interval width, about 0.033, 19% of the payoff). That last tolerance is a loose reproduction of the notebook assertion, not a precision test; the 3 standard error check is the meaningful test of the sampled estimate.

Deviations from the notebook:
* The notebook uses Classiq's IQAE application. QUMO has no IQAE routine, so this file runs the same Grover powers A Q^k on sampled circuits (k = 0, 1, 2, 4, 8, 1000 shots each, seed 2026) and combines them by maximum likelihood on a grid of theta (the MLAE estimator). The estimate is therefore not the notebook's random run; it is compared with the exact value and with the notebook's printed number through the tolerances above.
* inplace_prepare_state is QUMO's real amplitude embedding (a Mottonen RY tree) of sqrt(p). assign_amplitude_table under the comparator asset >= ceiling(descale(K)) is written as one multi-controlled RY per grid point above the strike, controlled on that point's bit pattern, with angle 2 asin(sqrt((S - K) / scaling_factor)). The operator is the same.
* The Grover operator is built from A as in the qmc_user_defined example: global phase pi, S_good = Z on ind, S_0 = I - 2|0><0| on the six qubits, with no ancillas.
* The notebook's lognormal pdf is computed from the closed-form density (scipy.stats.lognorm.pdf with s = sigma, scale = exp(mu)) written out directly.
* No synthesis, width constraint or circuit link; the plot is not drawn.

Run: .venv/bin/mojo run -I . examples/frameworks/classiq/option_pricing.mojo
example-meta: {"framework": "classiq", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from std.math import asin, exp, log, pi, sin, sqrt
from qumo.ir.circuit import Circuit
from qumo.ir.mc import append_mcp, append_mcrz
from qumo.algorithms.qml import append_amplitude_embedding_real
from qumo.sim.runner import evolve, sample
from qumo.testing.checks import CheckLog


comptime N = 5
comptime M = 32
comptime MU = 0.7
comptime SIGMA = 0.13
comptime K = 1.9
comptime SHOTS = 1000


struct Market(Copyable):
    var grid: List[Float64]
    var probs: List[Float64]
    var mean: Float64
    var variance: Float64
    var stddev: Float64
    var low: Float64
    var high: Float64

    def __init__(out self):
        self.mean = exp(MU + SIGMA * SIGMA / 2.0)
        self.variance = (exp(SIGMA * SIGMA) - 1.0) * exp(2.0 * MU + SIGMA * SIGMA)
        self.stddev = sqrt(self.variance)
        self.low = max(0.0, self.mean - 3.0 * self.stddev)
        self.high = self.mean + 3.0 * self.stddev
        self.grid = List[Float64]()
        self.probs = List[Float64]()
        var tot = 0.0
        for i in range(M):
            var x = self.low + (self.high - self.low) * Float64(i) / Float64(M - 1)
            self.grid.append(x)
            var z = (log(x) - MU) / SIGMA
            var pdf = exp(-0.5 * z * z) / (x * SIGMA * sqrt(2.0 * pi))
            self.probs.append(pdf)
            tot += pdf
        for i in range(M):
            self.probs[i] = self.probs[i] / tot


def add_mcry(mut c: Circuit, theta: Float64, controls: List[Int], target: Int) raises:
    """Multi-controlled RY as S H (multi-controlled RZ) H S^dag, in time order."""
    c.sdg(target)
    c.h(target)
    append_mcrz(c, theta, controls, target)
    c.h(target)
    c.s(target)


def state_prep(mk: Market, sf: Float64) raises -> Circuit:
    """A on 6 qubits: qubit 0 = ind, qubits 1..5 = asset (bit k on qubit 1 + k)."""
    var c = Circuit(N + 1, name="european_call_state_preparation")
    var amps = List[Float64]()
    for i in range(M):
        amps.append(sqrt(mk.probs[i]))
    var wires = List[Int]()
    for k in range(N - 1, -1, -1):
        wires.append(1 + k)
    append_amplitude_embedding_real(c, amps, wires)
    var controls = List[Int]()
    for k in range(N):
        controls.append(1 + k)
    for n in range(M):
        if mk.grid[n] < K:
            continue  # asset >= ceiling(descale(K)): payoff is zero below the strike
        var amp = sqrt(abs((mk.grid[n] - K) / sf))
        if amp == 0.0:
            continue
        for k in range(N):
            if ((n >> k) & 1) == 0:
                c.x(1 + k)
        add_mcry(c, 2.0 * asin(amp), controls, 0)
        for k in range(N):
            if ((n >> k) & 1) == 0:
                c.x(1 + k)
    return c^


def grover_operator(a: Circuit) raises -> Circuit:
    """Q = -A S_0 A^dag S_good (time order S_good, A^dag, S_0, A, global phase pi)."""
    var nq = N + 1
    var c = Circuit(nq, name="iqae_grover")
    c.z(0)
    c.compose(a.inverse())
    for q in range(nq):
        c.x(q)
    var ctrl = List[Int]()
    for q in range(nq - 1):
        ctrl.append(q)
    append_mcp(c, pi, ctrl, nq - 1)
    for q in range(nq):
        c.x(q)
    c.compose(a)
    c.global_phase = pi
    return c^


def a_q_k(a: Circuit, q: Circuit, k: Int, measured: Bool) raises -> Circuit:
    var c = Circuit(N + 1, 1 if measured else 0)
    c.compose(a)
    for _ in range(k):
        c.compose(q)
    if measured:
        c.measure(0, 0)
    return c^


def p_ind1(sv_probs: List[Float64]) -> Float64:
    var p = 0.0
    for i in range(len(sv_probs)):
        if (i & 1) == 1:
            p += sv_probs[i]
    return p


def main() raises:
    var log_ = CheckLog("classiq/option_pricing")
    var mk = Market()

    # ---- notebook print of get_log_normal_probabilities ----
    print("mean, variance, stddev, low, high =", mk.mean, mk.variance, mk.stddev, mk.low, mk.high)
    _ = log_.close("grid mean matches the notebook print", mk.mean, 2.030841014265948, 1e-11)
    _ = log_.close("grid variance matches the notebook print", mk.variance, 0.07029323208790372, 1e-11)
    _ = log_.close("grid stddev matches the notebook print", mk.stddev, 0.26512870853210846, 1e-11)
    _ = log_.close("grid low matches the notebook print", mk.low, 1.2354548886696226, 1e-11)
    _ = log_.close("grid high matches the notebook print", mk.high, 2.8262271398622736, 1e-11)

    # ---- exact expected payoff, brute force ----
    var sf = mk.high - K
    var expected = 0.0
    for i in range(M):
        if mk.grid[i] >= K:
            expected += (mk.grid[i] - K) * mk.probs[i]
    print("Expected Payoff:", expected)
    _ = log_.close("expected payoff matches the notebook print", expected, 0.17680663493930157, 1e-11)
    var a_exact = expected / sf
    var theta = asin(sqrt(a_exact))

    # ---- A|0>: joint law of (asset, ind), basis index = ind + 2 x ----
    var a = state_prep(mk, sf)
    var probs = evolve(a).probabilities()
    var dev = 0.0
    var p_good = 0.0
    for x in range(M):
        var f = 0.0
        if mk.grid[x] >= K:
            f = (mk.grid[x] - K) / sf
        dev = max(dev, abs(probs[1 + 2 * x] - mk.probs[x] * f))
        dev = max(dev, abs(probs[2 * x] - mk.probs[x] * (1.0 - f)))
        p_good += probs[1 + 2 * x]
    _ = log_.close("A|0> joint law of (asset, ind) matches p(x) f(x) (max deviation)", dev, 0.0, 1e-10)
    _ = log_.close("P(ind = 1) after A equals expected payoff / scaling_factor", p_good, a_exact, 1e-10)
    _ = log_.close("scaled payoff times scaling_factor equals the notebook expected payoff", p_good * sf, 0.17680663493930157, 1e-10)

    # ---- Grover powers: exact amplification law ----
    var q = grover_operator(a)
    var ks: List[Int] = [0, 1, 2, 4, 8]
    var dev_k = 0.0
    for kk in ks:
        var pk = p_ind1(evolve(a_q_k(a, q, kk, False)).probabilities())
        var s = sin(Float64(2 * kk + 1) * theta)
        dev_k = max(dev_k, abs(pk - s * s))
        print("k =", kk, " P(ind=1) =", pk, " sin^2((2k+1)theta) =", s * s)
    _ = log_.close("P(ind = 1) after A Q^k matches sin^2((2k+1) theta) for k = 0,1,2,4,8 (max deviation)", dev_k, 0.0, 1e-9)

    # ---- sampled maximum-likelihood amplitude estimate ----
    var hits = List[Int]()
    for kk in ks:
        var res = sample(a_q_k(a, q, kk, True), SHOTS, UInt64(2026))
        var h = 0
        for kv in res.counts.items():
            if kv.key == "1":
                h += kv.value
        hits.append(h)
    var best = 0.0
    var best_ll = -1.0e300
    var grid_n = 40000
    for g in range(1, grid_n):
        var t = 0.5 * pi * Float64(g) / Float64(grid_n)
        var ll = 0.0
        for j in range(len(ks)):
            var s = sin(Float64(2 * ks[j] + 1) * t)
            var p = min(max(s * s, 1e-300), 1.0 - 1e-16)
            ll += Float64(hits[j]) * log(p) + Float64(SHOTS - hits[j]) * log(1.0 - p)
        if ll > best_ll:
            best_ll = ll
            best = t
    # p_k = sin^2((2k+1) t): dp/dt = (2k+1) sin(2(2k+1) t), Fisher information 4 shots (2k+1)^2 per depth
    var fisher = 0.0
    for kk in ks:
        fisher += 4.0 * Float64(SHOTS) * Float64(2 * kk + 1) * Float64(2 * kk + 1)
    var sigma_t = 1.0 / sqrt(fisher)
    var est_a = sin(best) * sin(best)
    var sigma_payoff = sin(2.0 * best) * sigma_t * sf
    var measured = est_a * sf
    print("hits per depth (k=0,1,2,4,8):", hits[0], hits[1], hits[2], hits[3], hits[4])
    print("Measured Payoff:", measured, " standard error:", sigma_payoff)
    _ = log_.close("sampled MLE payoff matches the exact expected payoff (3 standard errors)", measured, expected, 3.0 * sigma_payoff)
    # the notebook asserts |measured - expected| <= 5 * (CI width), CI = [0.17349158, 0.1801615]
    var nb_tol = 5.0 * (0.1801615 - 0.17349158)
    _ = log_.close("sampled MLE payoff matches the notebook measured payoff 0.17682654 (notebook tolerance)", measured, 0.17682654151421245, nb_tol)
    _ = log_.check("notebook assertion |measured - expected| <= 5 x CI width holds", abs(measured - expected) <= nb_tol, "diff=" + String(abs(measured - expected)))

    log_.finish()
