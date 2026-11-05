"""1D Ising chain solved with QAOA (CVaR objective), exact on QUMO.

Reproduces: Classiq library - Ising Model
Source: https://github.com/Classiq/classiq-library/blob/main/applications/physical_systems/ising_model/ising_model.ipynb (Classiq Library, MIT)

The notebook minimises the periodic 1D Ising Hamiltonian H = sum over the six bonds of [-J s_i s_j - (h/2)(s_i + s_j)], s = 2z - 1, for J = 10, h = -20, n = 6. It runs a 5-layer QAOA with Classiq's CombinatorialProblem (maxiter 100, CVaR quantile 0.7) and prints the five lowest-cost samples: the all-zero state at cost -180 (probability 0.698) and single-spin flips at cost -100. This file checks the Hamiltonian against the printed costs by brute force over all 64 states (ground state z = 000000 at -180, the six single flips at -100, the printed best solution), checks that the QAOA cost layer built from RZZ and RZ gates applies the phase exp(-i gamma H / J) to every basis state, then optimises a 5-layer QAOA with the CVaR(0.7) objective on exact probabilities and checks that the most probable state is the brute-force ground state and that its probability beats the uniform value 1/64 and a 0.5 floor, and that the optimised expected energy beats the uniform average.

Deviations from the notebook:
- Classiq compiles the Pyomo model into its own phase primitive and mixer, and picks its own initial angles. Here the cost layer is RZZ(-2 gamma) on every bond plus RZ(-4 gamma) on every site (the phase exp(-i gamma H / J) up to a global phase, so the energy is scaled by 1/J), the mixer is RX(2 beta) on every qubit, and the start is a linear ramp. The circuit is not the notebook's circuit.
- The optimiser is QUMO's Nelder-Mead (at most 400 iterations) on the exact CVaR with quantile 0.7, the mean energy of the lowest 70 percent of the probability mass. The notebook uses a shot-based Classiq optimiser with maxiter = 100. The budgets are not the same.
- The notebook's 0.697754 is one shot-sampled estimate at its own optimum; it is not a target. The check here is on the exact probability of the ground state at this file's optimum, against the uniform value and a 0.5 floor.
- Pyomo and the penalty_factor (unused, the problem has no constraints) are replaced by a plain energy function over the 64 basis states. Qubit q holds z_q; basis index bit q is qubit q.

Run: .venv/bin/mojo run -I . examples/frameworks/classiq/ising_model.mojo
example-meta: {"framework": "classiq", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from std.math import cos, sin
from qumo.ir.circuit import Circuit
from qumo.sim.runner import evolve
from qumo.algorithms.optimizers import NelderMead, Objective
from qumo.testing.checks import CheckLog

comptime N = 6
comptime DIM = 64
comptime J = 10.0
comptime HF = -20.0
comptime LAYERS = 5
comptime QUANTILE = 0.7


def spin(x: Int, q: Int) -> Float64:
    """Spin s = 2 z - 1 for qubit q of basis index x."""
    return Float64(2 * ((x >> q) & 1) - 1)


def bond_energy(x: Int, i: Int, j: Int) -> Float64:
    """The notebook's E(i, j)."""
    return -J * spin(x, i) * spin(x, j) - (HF / 2.0) * (spin(x, i) + spin(x, j))


def energy(x: Int) -> Float64:
    """Periodic chain: bond (0, n-1) first, then (i, i+1), as in ising_model_1d(periodic='True')."""
    var e = bond_energy(x, 0, N - 1)
    for i in range(N - 1):
        e += bond_energy(x, i, i + 1)
    return e


def cost_layer(mut c: Circuit, gamma: Float64):
    """The phase exp(-i gamma H / J), up to a global phase. s_i s_j = Z_i Z_j and s_i = -Z_i, so
    H = -J sum_bonds Z_i Z_j + h sum_i Z_i; exp(-i g c ZZ) = RZZ(2 g c), exp(-i g c Z) = RZ(2 g c)."""
    for i in range(N):
        c.rzz(-2.0 * gamma, i, (i + 1) % N)
    for i in range(N):
        c.rz(2.0 * gamma * HF / J, i)


def qaoa_circuit(params: List[Float64]) -> Circuit:
    """Params are gammas (LAYERS), then betas (LAYERS)."""
    var c = Circuit(N, name="ising_qaoa")
    for q in range(N):
        c.h(q)
    for l in range(LAYERS):
        cost_layer(c, params[l])
        for q in range(N):
            c.rx(2.0 * params[LAYERS + l], q)
    return c^


def energies() -> List[Float64]:
    var out = List[Float64](length=DIM, fill=0.0)
    for x in range(DIM):
        out[x] = energy(x)
    return out^


def sorted_order(e: List[Float64]) -> List[Int]:
    """Basis indices by increasing energy (insertion sort, ties by index)."""
    var idx = List[Int]()
    for x in range(len(e)):
        idx.append(x)
    for a in range(1, len(idx)):
        var v = idx[a]
        var b = a - 1
        while b >= 0 and e[idx[b]] > e[v]:
            idx[b + 1] = idx[b]
            b -= 1
        idx[b + 1] = v
    return idx^


def cvar(pr: List[Float64], e: List[Float64], order: List[Int], alpha: Float64) -> Float64:
    """Mean energy of the lowest `alpha` share of the probability mass."""
    var need = alpha
    var acc = 0.0
    for x in order:
        var take = min(pr[x], need)
        acc += take * e[x]
        need -= take
        if need <= 0.0:
            break
    return acc / alpha


struct CvarObjective(Objective, Movable):
    var e: List[Float64]
    var order: List[Int]
    var nfev: Int

    def __init__(out self, var e: List[Float64]):
        self.order = sorted_order(e)
        self.e = e^
        self.nfev = 0

    def num_params(self) -> Int:
        return 2 * LAYERS

    def value(mut self, x: List[Float64]) raises -> Float64:
        self.nfev += 1
        var pr = evolve(qaoa_circuit(x)).probabilities()
        return cvar(pr, self.e, self.order, QUANTILE)

    def value_and_gradient(mut self, x: List[Float64], mut grad: List[Float64]) raises -> Float64:
        var f0 = self.value(x)
        var xp = x.copy()
        var h = 1e-6
        for i in range(len(x)):
            xp[i] = x[i] + h
            grad[i] = (self.value(xp) - f0) / h
            xp[i] = x[i]
        return f0


def main() raises:
    var log = CheckLog("classiq/ising_model")
    var e = energies()
    var order = sorted_order(e)

    # ---- Hamiltonian against the notebook's printed costs ----
    var n_ground = 0
    var n_m100 = 0
    for x in range(DIM):
        if e[x] == -180.0:
            n_ground += 1
        if e[x] == -100.0:
            n_m100 += 1
    print("lowest energy", e[order[0]], " states at -180:", n_ground, " states at -100:", n_m100)
    _ = log.check("all-zero state z = 000000 has cost -180 (notebook best solution)", e[0] == -180.0 and order[0] == 0, "e0=" + String(e[0]))
    _ = log.check("the ground state is unique", n_ground == 1)
    var singles_ok = True
    for q in range(N):
        if e[1 << q] != -100.0:
            singles_ok = False
    _ = log.check("each of the six single-spin flips has cost -100 (notebook rows 2-5)", singles_ok)
    _ = log.check("the single flips are the only states at -100", n_m100 == N)

    # ---- Cost layer phases vs exp(-i gamma H / J) on all 64 basis states ----
    var g = 0.37
    var probe = Circuit(N, name="ising_cost_probe")
    for q in range(N):
        probe.h(q)
    cost_layer(probe, g)
    var sv = evolve(probe)
    var dev = 0.0
    for x in range(DIM):
        var d = -g * (e[x] - e[0]) / J  # phase of state x relative to state 0
        var rr = sv.re[x] * sv.re[0] + sv.im[x] * sv.im[0]
        var ii = sv.im[x] * sv.re[0] - sv.re[x] * sv.im[0]
        dev = max(dev, abs(rr - cos(d) / 64.0))
        dev = max(dev, abs(ii - sin(d) / 64.0))
    _ = log.close("cost layer applies exp(-i gamma H / J) on all 64 states", dev, 0.0, 1e-12)

    # ---- QAOA with CVaR(0.7), 5 layers, from a linear ramp ----
    var obj = CvarObjective(e.copy())
    var x0 = List[Float64]()
    for l in range(LAYERS):
        x0.append(0.1 + 0.4 * Float64(l) / Float64(LAYERS - 1))
    for l in range(LAYERS):
        x0.append(0.6 - 0.4 * Float64(l) / Float64(LAYERS - 1))
    var start_cvar = obj.value(x0)
    var res = NelderMead(maxiter=400, initial_step=0.3).minimize(obj, x0)
    var pr = evolve(qaoa_circuit(res.x)).probabilities()
    var best = 0
    var mean_e = 0.0
    var uni_e = 0.0
    for x in range(DIM):
        if pr[x] > pr[best]:
            best = x
        mean_e += pr[x] * e[x]
        uni_e += e[x] / 64.0
    var final_cvar = cvar(pr, e, order, QUANTILE)
    print("CVaR start", start_cvar, " final", final_cvar, " nfev", obj.nfev)
    print("P(ground) =", pr[0], " <E> =", mean_e, " uniform <E> =", uni_e, " (notebook P(ground) = 0.697754)")
    _ = log.check("optimisation lowers the CVaR objective", final_cvar < start_cvar, "start=" + String(start_cvar) + " final=" + String(final_cvar))
    _ = log.check("most probable state is the brute-force ground state z = 000000", best == 0, "best=" + String(best))
    _ = log.check("ground-state probability beats the uniform value 1/64", pr[0] > 1.0 / 64.0, "p=" + String(pr[0]))
    _ = log.check("ground-state probability exceeds 0.5 (notebook reports 0.698)", pr[0] > 0.5, "p=" + String(pr[0]))
    _ = log.check("optimised <E> beats the uniform average", mean_e < uni_e, "mean=" + String(mean_e) + " uniform=" + String(uni_e))
    var single_best = 0.0
    for q in range(N):
        single_best = max(single_best, pr[1 << q])
    _ = log.check("ground state is more probable than any single-flip state", pr[0] > single_best)

    log.finish()
