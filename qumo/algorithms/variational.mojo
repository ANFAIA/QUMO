"""Variational algorithm drivers: VQE, QAOA-MaxCut and parameter sweeps.

  * `VQEObjective` adapts (ansatz, Hamiltonian, gradient method) to the
    `Objective` trait; it owns an `ExpectationEngine`, so an optimisation run
    allocates its two state vectors once and reuses them for every
    evaluation (PennyLane-Lightning adjoint pipeline, Qiskit `VQE` API shape).
  * `vqe(...)` returns a `VQEResult` with the energy, optimal parameters (and
    their names), the convergence history, evaluation counters (optimiser
    calls *and* circuit executions - the true simulator cost) and wall time.
  * `qaoa_maxcut(graph, p)` optimises QAOA angles with multi-start, samples the
    optimised circuit, and reports the approximation ratio against the
    brute-force optimum (Farhi-Goldstone-Gutmann; Qiskit `QAOA`, OpenQAOA).
  * `parameter_sweep` / `landscape_2d` evaluate energy landscapes through the
    batched evaluator (one circuit, many parameter sets).

Determinism: every random choice (initial points, SPSA perturbations, shot
sampling) derives from an explicit `seed`.
"""

from std.time import perf_counter_ns
from std.math import pi, ceil
from std.builtin.sort import sort
from qumo.algorithms.qubo import qubo_energy_of_ising
from qumo.ir.circuit import Circuit
from qumo.ir.pauli import PauliSum
from qumo.sim.runner import sample, evolve_into, SampleResult, check_statevector_size
from qumo.sim.statevector import StateVector
from qumo.core.rng import Rng
from qumo.algorithms.gradients import ExpectationEngine, Grad, grad_method_name, batch_expectation
from qumo.algorithms.optimizers import Objective, Optimizer, OptimizeResult, minimize
from qumo.algorithms.hamiltonians import Graph, maxcut, check_fits
from qumo.algorithms.library import qaoa_maxcut_ansatz, bitstring_to_int


struct VQEObjective(Objective, Movable):
    """E(theta) = <ansatz(theta)|H|ansatz(theta)> with a selectable gradient method."""

    var engine: ExpectationEngine
    var method: Int

    def __init__(out self, ansatz: Circuit, h: PauliSum, grad_method: Int = Grad.ADJOINT) raises:
        self.engine = ExpectationEngine(ansatz, h)
        self.method = grad_method

    def num_params(self) -> Int:
        return self.engine.num_params

    def value(mut self, x: List[Float64]) raises -> Float64:
        return self.engine.value(x)

    def value_and_gradient(mut self, x: List[Float64], mut grad: List[Float64]) raises -> Float64:
        return self.engine.value_and_gradient(x, grad, self.method)


def cvar_of_energies(energies: List[Float64], alpha: Float64) raises -> Float64:
    """CVaR_alpha of sampled energies: the mean of the lowest ceil(alpha * N)
    values (Barkoutsos et al. 2020, Qiskit `VQE(aggregation=alpha)`).
    alpha = 1 is the plain mean; a smaller alpha never gives a larger value."""
    _check_alpha(alpha)
    if len(energies) == 0:
        raise Error("QM-E804: cvar_of_energies() needs at least one energy.")
    var s = energies.copy()
    sort(s)
    var k = Int(ceil(alpha * Float64(len(s)) - 1e-12))
    k = min(max(k, 1), len(s))
    var acc = 0.0
    for i in range(k):
        acc += s[i]
    return acc / Float64(k)


def _check_alpha(alpha: Float64) raises:
    if not (alpha > 0.0 and alpha <= 1.0):
        raise Error("QM-E804: CVaR alpha must lie in (0, 1], got " + String(alpha) + ". Hint: alpha = 1 is the plain mean.")


struct CVaRObjective(Objective, Movable):
    """CVaR_alpha of the energy distribution of ansatz(theta) for a *diagonal* H.

    Exact (infinite-shot) limit: the basis-state energies are sorted once, and
    each evaluation averages the lowest alpha of probability mass. alpha = 1
    equals <H>. CVaR is piecewise smooth, so the gradient is a central finite
    difference (use "bfgs" with care, or "nelder-mead" / "spsa")."""

    var ansatz: Circuit
    var sv: StateVector
    var energies: List[Float64]
    var order: List[Int]
    var alpha: Float64
    var step: Float64
    var nfev: Int

    def __init__(out self, ansatz: Circuit, h: PauliSum, alpha: Float64, step: Float64 = 1e-6) raises:
        _check_alpha(alpha)
        ansatz.validate()
        if not h.is_diagonal():
            raise Error("QM-E804: CVaR needs a diagonal Hamiltonian (Z / ZZ terms only). Hint: use alpha = 1 (plain VQE) for Hamiltonians with X or Y terms.")
        check_fits(h, ansatz.num_qubits)  # QM-E803, as the mean objective raises
        if ansatz.num_qubits > 24:
            raise Error("QM-E804: CVaRObjective tabulates 2^n energies; " + String(ansatz.num_qubits) + " qubits is too many.")
        self.ansatz = ansatz.copy()
        self.sv = StateVector(ansatz.num_qubits)
        self.alpha = alpha
        self.step = step
        self.nfev = 0
        var dim = 1 << ansatz.num_qubits
        self.energies = List[Float64](capacity=dim)
        self.order = List[Int](capacity=dim)
        for i in range(dim):
            self.energies.append(qubo_energy_of_ising(h, UInt64(i)))
            self.order.append(i)
        var en = self.energies.copy()

        def less(a: Int, b: Int) {imm en} -> Bool:
            return en[a] < en[b] or (en[a] == en[b] and a < b)

        sort(Span(self.order), less)

    def num_params(self) -> Int:
        return self.ansatz.num_params()

    def value(mut self, x: List[Float64]) raises -> Float64:
        evolve_into(self.sv, self.ansatz, x)
        self.nfev += 1
        var remaining = self.alpha
        var acc = 0.0
        for j in self.order:
            var p = self.sv.re[j] * self.sv.re[j] + self.sv.im[j] * self.sv.im[j]
            var w = min(p, remaining)
            acc += w * self.energies[j]
            remaining -= w
            if remaining <= 0.0:
                break
        return acc / self.alpha

    def value_and_gradient(mut self, x: List[Float64], mut grad: List[Float64]) raises -> Float64:
        var f = self.value(x)
        var xs = x.copy()
        for i in range(len(x)):
            xs[i] = x[i] + self.step
            var fp = self.value(xs)
            xs[i] = x[i] - self.step
            var fm = self.value(xs)
            xs[i] = x[i]
            grad[i] = (fp - fm) / (2.0 * self.step)
        return f


struct VQEResult(Copyable, Writable):
    var energy: Float64
    var params: List[Float64]
    var param_names: List[String]
    var history: List[Float64]
    var nit: Int
    var nfev: Int
    var ngrad: Int
    var circuit_evals: Int
    var wall_time_s: Float64
    var converged: Bool
    var message: String
    var optimizer: String
    var grad_method: String

    def __init__(out self):
        self.energy = 0.0
        self.params = List[Float64]()
        self.param_names = List[String]()
        self.history = List[Float64]()
        self.nit = 0
        self.nfev = 0
        self.ngrad = 0
        self.circuit_evals = 0
        self.wall_time_s = 0.0
        self.converged = False
        self.message = String()
        self.optimizer = String()
        self.grad_method = String()

    def write_to(self, mut w: Some[Writer]):
        w.write("VQEResult(energy=", self.energy, ", optimizer=", self.optimizer, ", grad=", self.grad_method,
                ", nit=", self.nit, ", nfev=", self.nfev, ", ngrad=", self.ngrad,
                ", circuit_evals=", self.circuit_evals, ", time_s=", self.wall_time_s,
                ", converged=", self.converged, ")")

    def to_json(self) -> String:
        """One JSON object with the fields of the `qumo_vqe` tool (energy, params,
        iterations, circuit_evaluations, converged, optimizer, gradient,
        wall_time_s) plus param_names, function_evaluations,
        gradient_evaluations, message and the energy history."""
        var s = String("{\"energy\": ") + _json_num(self.energy) + ", \"params\": " + _json_nums(self.params)
        s += ", \"param_names\": ["
        for i in range(len(self.param_names)):
            if i > 0:
                s += ", "
            s += _json_str(self.param_names[i])
        s += "], \"iterations\": " + String(self.nit) + ", \"function_evaluations\": " + String(self.nfev)
        s += ", \"gradient_evaluations\": " + String(self.ngrad) + ", \"circuit_evaluations\": " + String(self.circuit_evals)
        s += ", \"converged\": " + ("true" if self.converged else "false")
        s += ", \"optimizer\": " + _json_str(self.optimizer) + ", \"gradient\": " + _json_str(self.grad_method)
        s += ", \"message\": " + _json_str(self.message) + ", \"wall_time_s\": " + _json_num(self.wall_time_s)
        s += ", \"history\": " + _json_nums(self.history) + "}"
        return s^


def _json_num(v: Float64) -> String:
    """A JSON number; NaN and infinities, which JSON cannot hold, become null."""
    if v != v or v - v != 0.0:
        return "null"
    return String(v)


def _json_nums(xs: List[Float64]) -> String:
    var s = String("[")
    for i in range(len(xs)):
        if i > 0:
            s += ", "
        s += _json_num(xs[i])
    return s + "]"


def _json_str(text: String) -> String:
    """`text` as a JSON string (backslash, quote, newline and tab escaped)."""
    return "\"" + text.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n").replace("\t", "\\t") + "\""


def default_initial_point(num_params: Int, seed: UInt64 = 7, scale: Float64 = 0.1) -> List[Float64]:
    """Small seeded random angles in [-scale, scale] (breaks symmetric saddles, deterministic)."""
    var rng = Rng(seed)
    var x = List[Float64](length=num_params, fill=0.0)
    for i in range(num_params):
        x[i] = scale * (2.0 * rng.uniform() - 1.0)
    return x^


def _finish(var opt: OptimizeResult, ansatz: Circuit, evals: Int, t0: Int, grad_method: Int) -> VQEResult:
    var r = VQEResult()
    r.energy = opt.fun
    r.params = opt.x.copy()
    r.param_names = ansatz.param_names.copy()
    r.history = opt.history.copy()
    r.nit = opt.nit
    r.nfev = opt.nfev
    r.ngrad = opt.ngrad
    r.circuit_evals = evals
    r.wall_time_s = Float64(perf_counter_ns() - t0) * 1e-9
    r.converged = opt.converged
    r.message = opt.message
    r.optimizer = opt.method
    r.grad_method = grad_method_name(grad_method)
    return r^


def vqe_with[Opt: Optimizer](ansatz: Circuit, h: PauliSum, optimizer: Opt, x0: List[Float64] = List[Float64](), grad_method: Int = Grad.ADJOINT) raises -> VQEResult:
    """VQE with an explicit optimiser instance, e.g. vqe_with(ansatz, H, Adam(lr=0.05)).

    x0 empty -> default_initial_point(num_params). Returns the best energy found.
    """
    var t0 = perf_counter_ns()
    var obj = VQEObjective(ansatz, h, grad_method)
    var start = x0.copy() if len(x0) > 0 else default_initial_point(obj.num_params())
    var opt = optimizer.minimize(obj, start)
    return _finish(opt^, ansatz, obj.engine.nfev, t0, grad_method)


def _vqe_cvar(ansatz: Circuit, h: PauliSum, method: String, x0: List[Float64], maxiter: Int, tol: Float64, seed: UInt64, alpha: Float64) raises -> VQEResult:
    var t0 = perf_counter_ns()
    var obj = CVaRObjective(ansatz, h, alpha)
    var start = x0.copy() if len(x0) > 0 else default_initial_point(obj.num_params(), seed)
    var opt = minimize(obj, start, method, maxiter, tol, 0.05, seed)
    var r = _finish(opt^, ansatz, obj.nfev, t0, Grad.FINITE_DIFF)
    return r^


def vqe(ansatz: Circuit, h: PauliSum, method: String = "bfgs", x0: List[Float64] = List[Float64](), maxiter: Int = 200, tol: Float64 = 1e-8, grad_method: Int = Grad.ADJOINT, seed: UInt64 = 7, alpha: Float64 = 1.0) raises -> VQEResult:
    """Variational quantum eigensolver (Peruzzo et al. 2014).

    ansatz: parametric circuit (all its symbolic parameters are optimised).
    h: Hamiltonian (PauliSum) on <= ansatz.num_qubits qubits.
    method: optimiser name ("bfgs", "adam", "gradient-descent", "spsa", "nelder-mead").
    grad_method: Grad.ADJOINT (default), Grad.PARAM_SHIFT or Grad.FINITE_DIFF.
    x0 empty -> default_initial_point(num_params, seed).
    alpha: CVaR level in (0, 1] (Barkoutsos et al. 2020). 1.0 (default) is the
        plain expectation value; alpha < 1 minimises the mean of the lowest
        alpha of the energy distribution and needs a diagonal h. `energy` is
        then the CVaR value; grad_method is ignored (finite differences).
    """
    _check_alpha(alpha)
    if alpha < 1.0:
        return _vqe_cvar(ansatz, h, method, x0, maxiter, tol, seed, alpha)
    var t0 = perf_counter_ns()
    var obj = VQEObjective(ansatz, h, grad_method)
    var start = x0.copy() if len(x0) > 0 else default_initial_point(obj.num_params(), seed)
    var opt = minimize(obj, start, method, maxiter, tol, 0.05, seed)
    return _finish(opt^, ansatz, obj.engine.nfev, t0, grad_method)


# ---------------------------------------------------------------------------
# QAOA MaxCut
# ---------------------------------------------------------------------------
struct QAOAResult(Copyable, Writable):
    var p: Int
    var energy: Float64
    var expected_cut: Float64
    var max_cut: Float64
    var approx_ratio: Float64
    var gammas: List[Float64]
    var betas: List[Float64]
    var best_bitstring: String
    var best_cut: Float64
    var sampled_ratio: Float64
    var prob_optimal: Float64
    var most_frequent: String
    var shots: Int
    var history: List[Float64]
    var nfev: Int
    var circuit_evals: Int
    var wall_time_s: Float64
    var optimizer: String

    def __init__(out self, p: Int):
        self.p = p
        self.energy = 0.0
        self.expected_cut = 0.0
        self.max_cut = 0.0
        self.approx_ratio = 0.0
        self.gammas = List[Float64]()
        self.betas = List[Float64]()
        self.best_bitstring = String()
        self.best_cut = 0.0
        self.sampled_ratio = 0.0
        self.prob_optimal = 0.0
        self.most_frequent = String()
        self.shots = 0
        self.history = List[Float64]()
        self.nfev = 0
        self.circuit_evals = 0
        self.wall_time_s = 0.0
        self.optimizer = String()

    def write_to(self, mut w: Some[Writer]):
        w.write("QAOAResult(p=", self.p, ", expected_cut=", self.expected_cut, ", max_cut=", self.max_cut,
                ", approx_ratio=", self.approx_ratio, ", best=", self.best_bitstring, " (cut ", self.best_cut,
                "), P(optimal)=", self.prob_optimal, ", gammas=")
        _write_list(w, self.gammas)
        w.write(", betas=")
        _write_list(w, self.betas)
        w.write(", time_s=", self.wall_time_s, ")")


def _write_list(mut w: Some[Writer], v: List[Float64]):
    w.write("[")
    for i in range(len(v)):
        if i > 0:
            w.write(", ")
        w.write(v[i])
    w.write("]")


def qaoa_maxcut(graph: Graph, p: Int = 1, method: String = "bfgs", restarts: Int = 4, shots: Int = 1024, seed: UInt64 = 7, maxiter: Int = 200, grad_method: Int = Grad.ADJOINT, alpha: Float64 = 1.0) raises -> QAOAResult:
    """Solve MaxCut on `graph` with depth-p QAOA.

    1. Build qaoa_maxcut_ansatz(graph, p) and H = maxcut(graph) (<H> = -<cut>).
    2. Minimise <H> from `restarts` initial points (a linear ramp, then seeded
       random angles), keep the best.
    3. Sample the optimised circuit (`shots`, `seed`), report the best sampled
       cut, the exact probability of measuring an optimal cut, and the
       approximation ratio <cut> / maxcut against brute force.
    Params order: gamma[0..p-1], beta[0..p-1].
    alpha in (0, 1]: CVaR level (Barkoutsos et al. 2020). 1.0 (default) minimises
    <H>; alpha < 1 minimises the mean of the best alpha-fraction of the cut
    distribution (finite-difference gradients; grad_method is ignored). `energy`
    is then the CVaR value while `expected_cut` stays the exact mean cut.
    """
    _check_alpha(alpha)
    graph.check_max_cut_size()  # the reference runs after the optimisation: fail first
    check_statevector_size(graph.num_nodes)
    if alpha < 1.0:
        return _qaoa_maxcut_cvar(graph, p, method, restarts, shots, seed, maxiter, alpha)
    var t0 = perf_counter_ns()
    var ansatz = qaoa_maxcut_ansatz(graph, p)
    var h = maxcut(graph)
    var obj = VQEObjective(ansatz, h, grad_method)
    var starts = _qaoa_starts(p, restarts, seed)
    var best_f = 1e300
    var best_x = List[Float64]()
    var best_hist = List[Float64]()
    var nfev = 0
    for r in range(len(starts)):
        var opt = minimize(obj, starts[r], method, maxiter, 1e-8, 0.05, seed + UInt64(r))
        nfev += opt.nfev
        if opt.fun < best_f:
            best_f = opt.fun
            best_x = opt.x.copy()
            best_hist = opt.history.copy()
    return _qaoa_report(graph, p, ansatz, best_f, best_f, best_x, best_hist^, nfev, method, shots, seed, obj.engine.nfev, t0)


def _qaoa_maxcut_cvar(graph: Graph, p: Int, method: String, restarts: Int, shots: Int, seed: UInt64, maxiter: Int, alpha: Float64) raises -> QAOAResult:
    var t0 = perf_counter_ns()
    var ansatz = qaoa_maxcut_ansatz(graph, p)
    var h = maxcut(graph)
    var obj = CVaRObjective(ansatz, h, alpha)
    var starts = _qaoa_starts(p, restarts, seed)
    var best_f = 1e300
    var best_x = List[Float64]()
    var best_hist = List[Float64]()
    var nfev = 0
    for r in range(len(starts)):
        var opt = minimize(obj, starts[r], method, maxiter, 1e-8, 0.05, seed + UInt64(r))
        nfev += opt.nfev
        if opt.fun < best_f:
            best_f = opt.fun
            best_x = opt.x.copy()
            best_hist = opt.history.copy()
    var mean_obj = VQEObjective(ansatz, h)
    var mean_e = mean_obj.value(best_x)
    return _qaoa_report(graph, p, ansatz, best_f, mean_e, best_x, best_hist^, nfev, method, shots, seed, obj.nfev, t0)


def _qaoa_starts(p: Int, restarts: Int, seed: UInt64) -> List[List[Float64]]:
    """Restart 0 is a linear ramp, the others seeded random angles."""
    var rng = Rng(seed)
    var out = List[List[Float64]]()
    for r in range(max(restarts, 1)):
        var x0 = List[Float64](length=2 * p, fill=0.0)
        for l in range(p):
            var frac = (Float64(l) + 0.5) / Float64(p)
            if r == 0:
                x0[l] = 0.6 * frac  # gamma ramps up
                x0[p + l] = 0.6 * (1.0 - frac)  # beta ramps down
            else:
                x0[l] = pi * rng.uniform()
                x0[p + l] = 0.5 * pi * rng.uniform()
        out.append(x0^)
    return out^


def _qaoa_report(graph: Graph, p: Int, ansatz: Circuit, best_f: Float64, mean_e: Float64, best_x: List[Float64], var best_hist: List[Float64], nfev: Int, method: String, shots: Int, seed: UInt64, evals: Int, t0: Int) raises -> QAOAResult:
    var res = QAOAResult(p)
    res.energy = best_f
    res.expected_cut = -mean_e
    var mc = graph.max_cut()
    res.max_cut = mc[0]
    res.approx_ratio = res.expected_cut / mc[0] if mc[0] > 0 else 1.0
    for l in range(p):
        res.gammas.append(best_x[l])
        res.betas.append(best_x[p + l])
    res.history = best_hist^
    res.nfev = nfev
    res.optimizer = method
    # exact probability of optimal cuts
    var sv = StateVector(graph.num_nodes)
    evolve_into(sv, ansatz, best_x)
    var popt = 0.0
    for i in range(sv.size()):
        if graph.cut_value(i) >= mc[0] - 1e-9:
            popt += sv.re[i] * sv.re[i] + sv.im[i] * sv.im[i]
    res.prob_optimal = popt
    # shot sampling
    var bound = ansatz.bind(best_x)
    bound.measure_all()
    var counts = sample(bound, shots, seed)
    res.shots = shots
    res.most_frequent = counts.most_frequent()
    var best_cut = -1.0
    for k in counts.sorted_keys():
        var cv = graph.cut_value(bitstring_to_int(k))
        if cv > best_cut:
            best_cut = cv
            res.best_bitstring = k
    res.best_cut = best_cut
    res.sampled_ratio = best_cut / mc[0] if mc[0] > 0 else 1.0
    res.circuit_evals = evals
    res.wall_time_s = Float64(perf_counter_ns() - t0) * 1e-9
    return res^


# ---------------------------------------------------------------------------
# Sweeps / landscapes
# ---------------------------------------------------------------------------
def linspace(a: Float64, b: Float64, n: Int) -> List[Float64]:
    var out = List[Float64](length=max(n, 0), fill=a)
    if n > 1:
        for i in range(n):
            out[i] = a + (b - a) * Float64(i) / Float64(n - 1)
    return out^


def parameter_sweep(circuit: Circuit, h: PauliSum, index: Int, values: List[Float64], base: List[Float64]) raises -> List[Float64]:
    """E(theta) along parameter `index` (other parameters fixed at `base`)."""
    if index < 0 or index >= circuit.num_params():
        raise Error("QM-E804: parameter index " + String(index) + " out of range for " + circuit.param_names_str() + ".")
    var sets = List[List[Float64]](capacity=len(values))
    for v in values:
        var x = base.copy()
        x[index] = v
        sets.append(x^)
    return batch_expectation(circuit, h, sets)


def landscape_2d(circuit: Circuit, h: PauliSum, i: Int, j: Int, values_i: List[Float64], values_j: List[Float64], base: List[Float64]) raises -> List[Float64]:
    """Row-major grid E[a * len(values_j) + b] = E(theta_i = values_i[a], theta_j = values_j[b])."""
    var np = circuit.num_params()
    if i < 0 or i >= np or j < 0 or j >= np:
        raise Error("QM-E804: landscape_2d() parameter indices out of range for " + circuit.param_names_str() + ".")
    var sets = List[List[Float64]](capacity=len(values_i) * len(values_j))
    for a in values_i:
        for b in values_j:
            var x = base.copy()
            x[i] = a
            x[j] = b
            sets.append(x^)
    return batch_expectation(circuit, h, sets)
