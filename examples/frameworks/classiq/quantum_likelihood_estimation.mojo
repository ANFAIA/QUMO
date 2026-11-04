"""Quantum likelihood estimation on one qubit: the standard and the information-optimized Bayesian loops.

Reproduces: Classiq library - Quantum Likelihood Estimation
Source: https://github.com/Classiq/classiq-library/blob/main/algorithms/search_and_optimization/quantum_likelihood_estimation/quantum_likelihood_estimation.ipynb (Classiq Library, MIT)

The notebook identifies which of four single-qubit Hamiltonians H_j in {X, 2X, Z, 2Z} generated the data. Each step prepares psi_1 = cos(alpha)|0> + e^{i beta} sin(alpha)|1>, evolves for a time t, applies the measurement unitary U(theta, phi, pi - phi), and measures; the weights are updated with the likelihoods p_j(y) = |<y| W e^{-i H_j t} |psi_1>|^2 (Eq. (4)). The standard loop chooses t with the particle guess heuristic (PGH) and keeps the angles fixed. The optimized loop, following Ref. [3], chooses (t, alpha, beta, theta, phi) at each step by simulated annealing that minimizes the conditional entropy H(F|Y). The checks compare QUMO's circuit probabilities with the closed-form model (max error over four Hamiltonians and four settings), the outcome 0 at t = 0 and theta = 0 with cos^2(alpha), 4000-shot frequencies within four standard errors of the closed form, the PGH times with Frobenius norms worked out by hand, the Bayes posterior with the normalized analytic likelihoods, the QUMO conditional entropy with the identity H(F,Y) - H(Y) on analytic probabilities, the bounds 1 <= H(F|Y) <= 2 bits for a uniform prior (a 1e-12 floating-point slack), the annealing ranges and t_max = pi, the annealing cost against the analytic cost at its answer and against the best fixed-angle PGH time, convergence to H_0 with the notebook's seed for both loops, and over ten seeds that both loops converge to H_0 and that the optimized loop needs fewer iterations in total (the notebook's summary claim).

Deviations from the notebook:
- Classiq synthesis, calculate_state_vector and the cloud sample() are replaced by QUMO's local evolve() and sample(). Each circuit is built directly from the notebook's gates: RY(2 alpha), P(beta), the Hamiltonian evolution, and U(theta, phi, pi - phi, 0) as in w_func. The evolution is trotter(H, t, 1, order 1), the notebook's suzuki_trotter(h, t, 1, 1). With one Pauli term the product formula is exact, e^{-iHt} = RX or RZ of angle 2ct.
- The notebook's Eq. (2), W = e^{-i theta Y/2} e^{-i phi Z/2}, is not the unitary in its code. At theta = 0.9, phi = 2.2, alpha = 0.4, beta = 1.1 the outcome-0 probability is 0.844 for the code's U gate and 0.994 for Eq. (2). The checks follow the code, which is what the notebook runs.
- Random draws come from QUMO's seeded Rng, not numpy's default_rng. The notebook's iteration counts (16 and 3), its weight vectors and its optimal angles depend on numpy's stream and are not reproduced. The checks use analytic values, the notebook's own convergence asserts, and the summary's convergence claim over ten seeds.
- PGH draws two distinct hypotheses by sequential weighted draws without replacement. numpy's choice uses its own algorithm.
- The annealer keeps the notebook's schedule (8 steps, 20 neighbours per step, cooling 0.9, T starting at 1, angles wrapped modulo their ranges) and its acceptance rule. The notebook run uses SA seed 1234 in every optimized iteration; its random draws come from QUMO's Rng. The ten-seed check uses seeds 1 to 10 for both the outcome stream and the annealer.
- The minimum energy gap is 2|coeff| in closed form (the levels of coeff * sigma are plus and minus coeff), where the notebook uses eigvalsh. The entropy sums keep every term with a positive probability, where the notebook drops terms below 1e-12.
- Entropies are in bits (log base 2), as in the notebook. The code takes them as log(x) / ln 2, because std.math.log2 is about 1e-10 off on this toolchain.

Run: .venv/bin/mojo run -I . examples/frameworks/classiq/quantum_likelihood_estimation.mojo
example-meta: {"framework": "classiq", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from std.math import cos, exp, floor, log, pi, sin, sqrt
from qumo.algorithms.library import trotter
from qumo.ir.circuit import Circuit
from qumo.ir.pauli import PauliSum
from qumo.sim.rng import Rng
from qumo.sim.runner import evolve, sample
from qumo.testing.checks import CheckLog


comptime LN2 = 0.6931471805599453  # std.math.log2 is about 1e-10 off on this toolchain; log(x) / LN2 is exact to a few ulp
comptime TRUE_IDX = 0  # H_0 = X is the true Hamiltonian (the notebook's TRUE_IDX)
comptime THRESHOLD = 0.99  # both loops stop when the true weight passes this
comptime MAX_ITERS_STANDARD = 10000
comptime MAX_ITERS_OPTIMIZED = 100
comptime SA_STEPS = 8  # the notebook's optimizer_kwargs
comptime SA_NEIGHBORS = 20
comptime SA_COOLING = 0.9
comptime SLACK = 1e-12  # floating-point slack on the entropy bounds


# ---------------------------------------------------------------------------
# Small complex and 2x2 arithmetic for the closed-form side (no QUMO gates)
# ---------------------------------------------------------------------------
@fieldwise_init
struct Cx(ImplicitlyCopyable):
    var re: Float64
    var im: Float64

    def abs2(self) -> Float64:
        return self.re * self.re + self.im * self.im


@fieldwise_init
struct Mat2(ImplicitlyCopyable):
    var a: Cx  # row 0, column 0
    var b: Cx  # row 0, column 1
    var c: Cx  # row 1, column 0
    var d: Cx  # row 1, column 1


def cadd(p: Cx, q: Cx) -> Cx:
    return Cx(p.re + q.re, p.im + q.im)


def csub(p: Cx, q: Cx) -> Cx:
    return Cx(p.re - q.re, p.im - q.im)


def cmul(p: Cx, q: Cx) -> Cx:
    return Cx(p.re * q.re - p.im * q.im, p.re * q.im + p.im * q.re)


def cscale(p: Cx, s: Float64) -> Cx:
    return Cx(p.re * s, p.im * s)


# ---------------------------------------------------------------------------
# The notebook's Hamiltonians: H = coeff * sigma_axis on one qubit
# ---------------------------------------------------------------------------
@fieldwise_init
struct Ham(ImplicitlyCopyable):
    var axis: Int  # 0: sigma_x, 1: sigma_z
    var coeff: Float64


def sigma(axis: Int) -> Mat2:
    if axis == 0:
        return Mat2(Cx(0.0, 0.0), Cx(1.0, 0.0), Cx(1.0, 0.0), Cx(0.0, 0.0))
    return Mat2(Cx(1.0, 0.0), Cx(0.0, 0.0), Cx(0.0, 0.0), Cx(-1.0, 0.0))


def dense(h: Ham) -> Mat2:
    """The notebook's hamiltonian_to_matrix: the 2x2 matrix coeff * sigma_axis."""
    var s = sigma(h.axis)
    return Mat2(cscale(s.a, h.coeff), cscale(s.b, h.coeff), cscale(s.c, h.coeff), cscale(s.d, h.coeff))


def pauli_sum(h: Ham) -> PauliSum:
    """The same Hamiltonian as a QUMO PauliSum: x and z are the qubit-0 bit masks."""
    var out = PauliSum()
    if h.axis == 0:
        out.add(h.coeff, UInt64(1), UInt64(0))
    else:
        out.add(h.coeff, UInt64(0), UInt64(1))
    return out^


def hamiltonian_set() -> List[Ham]:
    """HAMILTONIAN_SET = [Pauli.X(0), 2 * Pauli.X(0), Pauli.Z(0), 2 * Pauli.Z(0)]."""
    var hs = List[Ham]()
    hs.append(Ham(0, 1.0))
    hs.append(Ham(0, 2.0))
    hs.append(Ham(1, 1.0))
    hs.append(Ham(1, 2.0))
    return hs^


def frobenius_distance(a: Ham, b: Ham) -> Float64:
    """Frobenius norm of hamiltonian_to_matrix(a) - hamiltonian_to_matrix(b), as np.linalg.norm gives it."""
    var da = dense(a)
    var db = dense(b)
    var sq = (
        csub(da.a, db.a).abs2()
        + csub(da.b, db.b).abs2()
        + csub(da.c, db.c).abs2()
        + csub(da.d, db.d).abs2()
    )
    return sqrt(sq)


def min_energy_gap(hs: List[Ham]) -> Float64:
    """Smallest positive level spacing over the set. coeff * sigma has levels -coeff and +coeff, gap 2|coeff|."""
    var gap = 2.0 * abs(hs[0].coeff)
    for k in range(1, len(hs)):
        gap = min(gap, 2.0 * abs(hs[k].coeff))
    return gap


# ---------------------------------------------------------------------------
# Closed-form model (independent of QUMO): p(y) = |<y| U3(theta, phi, pi-phi) e^{-iHt} psi_1|^2
# ---------------------------------------------------------------------------
def analytic_probs(h: Ham, t: Float64, params: List[Float64]) -> List[Float64]:
    var alpha = params[0]
    var beta = params[1]
    var theta = params[2]
    var phi = params[3]
    # psi_1 = cos(alpha)|0> + e^{i beta} sin(alpha)|1>  (Eq. (1))
    var v0 = Cx(cos(alpha), 0.0)
    var v1 = Cx(sin(alpha) * cos(beta), sin(alpha) * sin(beta))
    # e^{-iHt} = cos(ct) I - i sin(ct) sigma, with c = coeff
    var ct = h.coeff * t
    var cs = Cx(cos(ct), 0.0)
    var ms = Cx(0.0, -sin(ct))
    var sg = sigma(h.axis)
    var e00 = cadd(cs, cmul(ms, sg.a))
    var e01 = cmul(ms, sg.b)
    var e10 = cmul(ms, sg.c)
    var e11 = cadd(cs, cmul(ms, sg.d))
    var u0 = cadd(cmul(e00, v0), cmul(e01, v1))
    var u1 = cadd(cmul(e10, v0), cmul(e11, v1))
    # U(theta, phi, lam) with lam = pi - phi, global phase 0 (OpenQASM U3 entries)
    var lam = pi - phi
    var c = cos(theta / 2.0)
    var s = sin(theta / 2.0)
    var w00 = Cx(c, 0.0)
    var w01 = Cx(-s * cos(lam), -s * sin(lam))
    var w10 = Cx(s * cos(phi), s * sin(phi))
    var w11 = Cx(c * cos(phi + lam), c * sin(phi + lam))
    var a0 = cadd(cmul(w00, u0), cmul(w01, u1))
    var a1 = cadd(cmul(w10, u0), cmul(w11, u1))
    var out = List[Float64]()
    out.append(a0.abs2())
    out.append(a1.abs2())
    return out^


def analytic_table(hs: List[Ham], t: Float64, params: List[Float64]) -> List[List[Float64]]:
    var out = List[List[Float64]]()
    for j in range(len(hs)):
        out.append(analytic_probs(hs[j], t, params))
    return out^


# ---------------------------------------------------------------------------
# QUMO circuits: the notebook's main(), prepare_psi1, hamiltonian_evolution, w_func
# ---------------------------------------------------------------------------
def qle_circuit(h: Ham, t: Float64, params: List[Float64]) raises -> Circuit:
    var c = Circuit(1, name="qle")
    c.ry(2.0 * params[0], 0)  # prepare_psi1: RY(2 alpha)
    c.p(params[1], 0)  # phase(q == 1, beta)
    c.compose(trotter(pauli_sum(h), t, 1, 1, 1))  # hamiltonian_evolution: suzuki_trotter(h, t, 1, 1)
    c.u(params[2], params[3], pi - params[3], 0)  # w_func: U(theta, phi, pi - phi, 0)
    return c^


def measured_circuit(h: Ham, t: Float64, params: List[Float64]) raises -> Circuit:
    var c = qle_circuit(h, t, params)
    c.measure(0, 0)
    return c^


def qumo_probs(h: Ham, t: Float64, params: List[Float64]) raises -> List[Float64]:
    """Exact outcome law of the circuit on QUMO's statevector simulator."""
    return evolve(qle_circuit(h, t, params)).probabilities()


def qumo_table(hs: List[Ham], t: Float64, params: List[Float64]) raises -> List[List[Float64]]:
    var out = List[List[Float64]]()
    for j in range(len(hs)):
        out.append(qumo_probs(hs[j], t, params))
    return out^


def sample_outcome(h: Ham, t: Float64, params: List[Float64], seed: UInt64) raises -> Int:
    """One shot from the true system (the notebook's sample(qprogs[true], num_shots=1))."""
    return sample(measured_circuit(h, t, params), 1, seed).memory[0]


# ---------------------------------------------------------------------------
# PGH, Bayes update, conditional entropy
# ---------------------------------------------------------------------------
def pgh_time(hs: List[Ham], i: Int, j: Int) -> Float64:
    """Eq. (5): t = 1 / ||H_i - H_j||."""
    var nh = frobenius_distance(hs[i], hs[j])
    return 0.1 if nh < 1e-6 else 1.0 / nh


def draw_index(weights: List[Float64], excluded: Int, mut rng: Rng) -> Int:
    """Weighted draw of an index, skipping `excluded` (-1 skips none)."""
    var total = 0.0
    for k in range(len(weights)):
        if k != excluded:
            total += weights[k]
    var u = rng.uniform() * total
    var acc = 0.0
    var last = 0
    for k in range(len(weights)):
        if k == excluded:
            continue
        acc += weights[k]
        last = k
        if u < acc:
            return k
    return last


def pgh(hs: List[Ham], weights: List[Float64], mut rng: Rng) -> Float64:
    """PGH: two distinct hypotheses drawn from the current weights."""
    var i = draw_index(weights, -1, rng)
    var j = draw_index(weights, i, rng)
    return pgh_time(hs, i, j)


def bayes_step(weights: List[Float64], likelihoods: List[Float64]) raises -> List[Float64]:
    """Eq. (4): w_j <- w_j p_j(y) / sum_i w_i p_i(y)."""
    var out = List[Float64]()
    var total = 0.0
    for j in range(len(weights)):
        var v = max(weights[j] * likelihoods[j], 0.0)
        out.append(v)
        total += v
    if total <= 0.0:
        raise Error("bayes_step: the observed outcome has zero likelihood under every hypothesis")
    for j in range(len(out)):
        out[j] = out[j] / total
    return out^


def conditional_entropy(weights: List[Float64], probs: List[List[Float64]]) -> Float64:
    """H(F|Y) in bits = -sum_{j,y} w_j p_j(y) log2( w_j p_j(y) / p(y) ), p(y) = sum_j w_j p_j(y)."""
    var py = List[Float64](length=2, fill=0.0)
    for j in range(len(weights)):
        for y in range(2):
            py[y] += weights[j] * probs[j][y]
    var h = 0.0
    for j in range(len(weights)):
        for y in range(2):
            var joint = weights[j] * probs[j][y]
            if joint > 0.0:
                h -= joint * log(joint / py[y]) / LN2
    return h


def analytic_cost(weights: List[Float64], probs: List[List[Float64]]) -> Float64:
    """H(F|Y) = H(F,Y) - H(Y) in bits: the same quantity by a different route, for brute-force checks."""
    var h_fy = 0.0
    var py = List[Float64](length=2, fill=0.0)
    for j in range(len(weights)):
        for y in range(2):
            var q = weights[j] * probs[j][y]
            py[y] += q
            if q > 0.0:
                h_fy -= q * log(q) / LN2
    var h_y = 0.0
    for y in range(2):
        if py[y] > 0.0:
            h_y -= py[y] * log(py[y]) / LN2
    return h_fy - h_y


def uniform_weights(n: Int) -> List[Float64]:
    return List[Float64](length=n, fill=1.0 / Float64(n))


def argmax(xs: List[Float64]) -> Int:
    var best = 0
    for k in range(1, len(xs)):
        if xs[k] > xs[best]:
            best = k
    return best


def converged(weights: List[Float64], true_index: Int) -> Bool:
    """The notebook's asserts: guess_idx == TRUE_IDX and final weight of the true hypothesis >= 0.99."""
    return argmax(weights) == true_index and weights[true_index] >= THRESHOLD


# ---------------------------------------------------------------------------
# Annealing over (t, alpha, beta, theta, phi)
# ---------------------------------------------------------------------------
def wrap(x: Float64, r: Float64) -> Float64:
    """Modulo r for r > 0, as np.mod gives it."""
    return x - r * floor(x / r)


def point_cost(weights: List[Float64], hs: List[Ham], z: List[Float64]) raises -> Float64:
    """H(F|Y) at z = (t, alpha, beta, theta, phi), with probabilities from QUMO's statevector."""
    var params = List[Float64]()
    for k in range(1, len(z)):
        params.append(z[k])
    return conditional_entropy(weights, qumo_table(hs, z[0], params))


def optimize_entropy_sa(
    weights: List[Float64],
    hs: List[Ham],
    n_steps: Int,
    n_neighbors: Int,
    cooling: Float64,
    seed: UInt64,
) raises -> Tuple[Float64, List[Float64], Float64]:
    """The notebook's optimize_entropy_sa: returns (t, params, cost)."""
    var rng = Rng(seed)
    var t_max = 2.0 * pi / min_energy_gap(hs)
    var ranges = List[Float64]()
    ranges.append(t_max)
    ranges.append(pi)
    ranges.append(2.0 * pi)
    ranges.append(pi)
    ranges.append(2.0 * pi)
    var dim = len(ranges)

    var current = List[Float64]()
    for k in range(dim):
        current.append(wrap(rng.uniform() * ranges[k], ranges[k]))
    var current_cost = point_cost(weights, hs, current)
    var best = current.copy()
    var best_cost = current_cost
    var temp = 1.0

    for _ in range(n_steps):
        var neighbors = List[List[Float64]]()
        var costs = List[Float64]()
        for _ in range(n_neighbors):
            var z = List[Float64]()
            for k in range(dim):
                z.append(wrap(current[k] + temp * (2.0 * rng.uniform() - 1.0) * (ranges[k] / 2.0), ranges[k]))
            costs.append(point_cost(weights, hs, z))
            neighbors.append(z^)
        var best_idx = 0
        for b in range(1, n_neighbors):
            if costs[b] < costs[best_idx]:
                best_idx = b
        var cand_cost = costs[best_idx]
        var accept = cand_cost < current_cost
        if not accept:
            var prob = exp((current_cost - cand_cost) / max(temp, 1e-15))
            accept = rng.uniform() < prob
        if accept:
            current = neighbors[best_idx].copy()
            current_cost = cand_cost
        if current_cost < best_cost:
            best = current.copy()
            best_cost = current_cost
        temp *= cooling

    var params = List[Float64]()
    for k in range(1, dim):
        params.append(best[k])
    return (best[0], params^, best_cost)


# ---------------------------------------------------------------------------
# The two loops
# ---------------------------------------------------------------------------
def run_standard_qle(
    hs: List[Ham],
    true_index: Int,
    params: List[Float64],
    threshold: Float64,
    max_iters: Int,
    seed: UInt64,
) raises -> Tuple[List[Float64], Int]:
    """The notebook's run_qle: PGH time, one shot from the true system, Bayes update, until the true weight passes the threshold."""
    var rng = Rng(seed)
    var weights = uniform_weights(len(hs))
    var iters = 0
    while iters < max_iters:
        if weights[true_index] > threshold:
            break
        var t = pgh(hs, weights, rng)
        var outcome = sample_outcome(hs[true_index], t, params, rng.next_u64())
        var probs = qumo_table(hs, t, params)
        var like = List[Float64]()
        for j in range(len(hs)):
            like.append(probs[j][outcome])
        weights = bayes_step(weights, like)
        iters += 1
    return (weights^, iters)


def run_optimized_qle(
    hs: List[Ham],
    true_index: Int,
    threshold: Float64,
    max_iters: Int,
    seed: UInt64,
    sa_seed: UInt64,
) raises -> Tuple[List[Float64], Int]:
    """The notebook's run_optimized_qle: annealed (t, angles) minimizing H(F|Y), then the same shot and Bayes update."""
    var rng = Rng(seed)
    var weights = uniform_weights(len(hs))
    var iters = 0
    while iters < max_iters:
        if weights[true_index] > threshold:
            break
        var step = optimize_entropy_sa(weights, hs, SA_STEPS, SA_NEIGHBORS, SA_COOLING, sa_seed)
        var t = step[0]
        var params = step[1].copy()
        var outcome = sample_outcome(hs[true_index], t, params, rng.next_u64())
        var probs = qumo_table(hs, t, params)
        var like = List[Float64]()
        for j in range(len(hs)):
            like.append(probs[j][outcome])
        weights = bayes_step(weights, like)
        iters += 1
    return (weights^, iters)


def params4(a: Float64, b: Float64, c: Float64, d: Float64) -> List[Float64]:
    var p = List[Float64]()
    p.append(a)
    p.append(b)
    p.append(c)
    p.append(d)
    return p^


def main() raises:
    var log = CheckLog("classiq/quantum_likelihood_estimation")
    var hs = hamiltonian_set()
    var fixed = params4(1.2566, 0.5712, 0.7854, 2.0 * pi)  # the notebook's fixed (alpha, beta, theta, phi)
    var uni = uniform_weights(len(hs))

    # ---- A: the single-qubit model ----
    var test_t = List[Float64]()
    test_t.append(0.0)
    test_t.append(0.7)
    test_t.append(1.5)
    test_t.append(pi)
    var test_p = List[List[Float64]]()
    test_p.append(params4(0.3, 1.1, 0.9, 2.2))
    test_p.append(fixed.copy())
    test_p.append(params4(0.4, 2.5, 2.0, 5.0))
    test_p.append(params4(1.0, 0.2, 2.9, 0.6))
    var model_err = 0.0
    for k in range(len(test_t)):
        for j in range(len(hs)):
            var q = qumo_probs(hs[j], test_t[k], test_p[k])
            var r = analytic_probs(hs[j], test_t[k], test_p[k])
            model_err = max(model_err, max(abs(q[0] - r[0]), abs(q[1] - r[1])))
    _ = log.close("QUMO circuit matches the closed-form e^{-iHt} model (max error, 4 Hamiltonians x 4 settings)", model_err, 0.0, 1e-9)

    var zero_params = params4(0.4, 2.5, 0.0, 1.7)
    var zero_err = 0.0
    for j in range(len(hs)):
        var q0 = qumo_probs(hs[j], 0.0, zero_params)[0]
        zero_err = max(zero_err, abs(q0 - cos(0.4) * cos(0.4)))
    _ = log.close("at t = 0 and theta = 0 the outcome 0 has probability cos^2(alpha) for every Hamiltonian", zero_err, 0.0, 1e-9)

    var pz = analytic_probs(hs[TRUE_IDX], 0.7, fixed)[0]
    var shots = 4000
    var counted = sample(measured_circuit(hs[TRUE_IDX], 0.7, fixed), shots, UInt64(2026))
    var freq = counted.probability("0")
    var band = 4.0 * sqrt(pz * (1.0 - pz) / Float64(shots))
    _ = log.within("4000-shot frequency of outcome 0 within four standard errors of the closed form", freq, pz - band, pz + band)

    # ---- B: PGH times, annealing range, Bayes update ----
    var pa = List[Int]()
    var pb = List[Int]()
    var expected_pgh = List[Float64]()
    pa.append(0)
    pb.append(1)
    expected_pgh.append(1.0 / sqrt(2.0))  # ||X - 2X|| = ||X||_F = sqrt(2)
    pa.append(0)
    pb.append(2)
    expected_pgh.append(0.5)  # ||X - Z||_F = 2
    pa.append(0)
    pb.append(3)
    expected_pgh.append(1.0 / sqrt(10.0))  # ||X - 2Z||_F = sqrt(10)
    pa.append(1)
    pb.append(2)
    expected_pgh.append(1.0 / sqrt(10.0))  # ||2X - Z||_F = sqrt(10)
    pa.append(1)
    pb.append(3)
    expected_pgh.append(0.25)  # ||2X - 2Z||_F = 4
    pa.append(2)
    pb.append(3)
    expected_pgh.append(1.0 / sqrt(2.0))  # ||Z - 2Z||_F = sqrt(2)
    var pgh_err = 0.0
    for k in range(len(expected_pgh)):
        pgh_err = max(pgh_err, abs(pgh_time(hs, pa[k], pb[k]) - expected_pgh[k]))
    _ = log.close("PGH times 1 / ||H_i - H_j||_F match the six Frobenius norms worked out by hand", pgh_err, 0.0, 1e-12)

    _ = log.close("annealing range t_max = 2 pi / min energy gap equals pi", 2.0 * pi / min_energy_gap(hs), pi, 1e-12)

    var qt = qumo_table(hs, 0.7, fixed)
    var at = analytic_table(hs, 0.7, fixed)
    var bayes_err = 0.0
    for y in range(2):
        var lq = List[Float64]()
        var la = List[Float64]()
        for j in range(len(hs)):
            lq.append(qt[j][y])
            la.append(at[j][y])
        var post_q = bayes_step(uni, lq)
        var post_a = bayes_step(uni, la)
        for j in range(len(hs)):
            bayes_err = max(bayes_err, abs(post_q[j] - post_a[j]))
    _ = log.close("Bayes posterior from QUMO likelihoods matches the analytic posterior for both outcomes", bayes_err, 0.0, 1e-9)

    # ---- C: conditional entropy H(F|Y) ----
    var w_skew = List[Float64]()
    w_skew.append(0.4)
    w_skew.append(0.3)
    w_skew.append(0.2)
    w_skew.append(0.1)
    var cost_t = List[Float64]()
    cost_t.append(0.7)
    cost_t.append(0.3)
    cost_t.append(pi / 2.0)
    var cost_p = List[List[Float64]]()
    cost_p.append(fixed.copy())
    cost_p.append(params4(0.3, 1.1, 0.9, 2.2))
    cost_p.append(params4(1.0, 0.2, 2.9, 0.6))
    var cost_w = List[List[Float64]]()
    cost_w.append(uni.copy())
    cost_w.append(w_skew.copy())
    cost_w.append(uni.copy())
    var cost_err = 0.0
    var uniform_ok = True
    for k in range(len(cost_t)):
        var qc = conditional_entropy(cost_w[k], qumo_table(hs, cost_t[k], cost_p[k]))
        var ac = analytic_cost(cost_w[k], analytic_table(hs, cost_t[k], cost_p[k]))
        cost_err = max(cost_err, abs(qc - ac))
        if k != 1:
            if ac < 1.0 - SLACK or ac > 2.0 + SLACK:
                uniform_ok = False
    _ = log.close("QUMO conditional entropy matches H(F,Y) - H(Y) on analytic probabilities (3 settings)", cost_err, 0.0, 1e-9)
    _ = log.check("uniform-prior H(F|Y) lies in [1, 2] bits, since H(F) = 2 and H(Y) <= 1", uniform_ok)

    # ---- D: simulated annealing ----
    var sa = optimize_entropy_sa(uni, hs, SA_STEPS, SA_NEIGHBORS, SA_COOLING, UInt64(1234))
    var t_sa = sa[0]
    var p_sa = sa[1].copy()
    var in_range = (
        t_sa >= 0.0 and t_sa < pi
        and p_sa[0] >= 0.0 and p_sa[0] < pi
        and p_sa[1] >= 0.0 and p_sa[1] < 2.0 * pi
        and p_sa[2] >= 0.0 and p_sa[2] < pi
        and p_sa[3] >= 0.0 and p_sa[3] < 2.0 * pi
    )
    _ = log.check("annealing answer (t, alpha, beta, theta, phi) lies inside the notebook's ranges", in_range)
    var sa_analytic = analytic_cost(uni, analytic_table(hs, t_sa, p_sa))
    _ = log.close("annealing cost equals the analytic H(F|Y) at its answer", sa[2], sa_analytic, 1e-9)

    var fixed_best = analytic_cost(uni, analytic_table(hs, expected_pgh[0], fixed))
    for k in range(1, len(expected_pgh)):
        fixed_best = min(fixed_best, analytic_cost(uni, analytic_table(hs, expected_pgh[k], fixed)))
    _ = log.check(
        "annealed H(F|Y) is at most the best fixed-angle cost over the six PGH times",
        sa[2] <= fixed_best,
        "annealed=" + String(sa[2]) + " fixed_best=" + String(fixed_best),
    )
    print("annealing (seed 1234): t =", t_sa, " H(F|Y) =", sa[2], " fixed-angle best =", fixed_best)

    # ---- E: the two loops ----
    var std_run = run_standard_qle(hs, TRUE_IDX, fixed, THRESHOLD, MAX_ITERS_STANDARD, UInt64(1234))
    var std_w = std_run[0].copy()
    var std_iters = std_run[1]
    print("standard QLE (seed 1234): iterations =", std_iters, " weights =", std_w[0], std_w[1], std_w[2], std_w[3])
    _ = log.check("standard QLE with seed 1234 converges to H_0 (the notebook's assert)", converged(std_w, TRUE_IDX), "iters=" + String(std_iters))

    var opt_run = run_optimized_qle(hs, TRUE_IDX, THRESHOLD, MAX_ITERS_OPTIMIZED, UInt64(1234), UInt64(1234))
    var opt_w = opt_run[0].copy()
    var opt_iters = opt_run[1]
    print("optimized QLE (seed 1234): iterations =", opt_iters, " weights =", opt_w[0], opt_w[1], opt_w[2], opt_w[3])
    _ = log.check("optimized QLE with seed 1234 converges to H_0 (the notebook's assert)", converged(opt_w, TRUE_IDX), "iters=" + String(opt_iters))

    var std_total = 0
    var opt_total = 0
    var all_converged = True
    for s in range(1, 11):
        var seed = UInt64(s)
        var rs = run_standard_qle(hs, TRUE_IDX, fixed, THRESHOLD, MAX_ITERS_STANDARD, seed)
        var ro = run_optimized_qle(hs, TRUE_IDX, THRESHOLD, MAX_ITERS_OPTIMIZED, seed, seed)
        if not converged(rs[0], TRUE_IDX) or not converged(ro[0], TRUE_IDX):
            all_converged = False
        std_total += rs[1]
        opt_total += ro[1]
    print("ten seeds: total iterations standard =", std_total, " optimized =", opt_total)
    _ = log.check("both loops converge to H_0 for all ten seeds", all_converged)
    _ = log.check(
        "optimized QLE takes fewer iterations than standard QLE over ten seeds (the notebook's summary)",
        opt_total < std_total,
        "optimized=" + String(opt_total) + " standard=" + String(std_total),
    )

    log.finish()
