"""Set cover by CVaR-QAOA on a penalty cost, exact on QUMO.

Reproduces: Classiq library - Set Cover Problem
Source: https://github.com/Classiq/classiq-library/blob/main/applications/optimization/set_cover/set_cover.ipynb (Classiq Library, MIT)

The notebook covers the universe {1..10} with the fewest of eight subsets. It writes a Pyomo model (one binary variable per subset, a "covered at least once" constraint per element, objective = number of chosen subsets), lets Classiq's CombinatorialProblem turn it into a 3-layer QAOA with penalty_factor 10, optimises with CVaR (quantile 0.7, at most 60 iterations) and prints the five lowest-cost samples (all cost 23: three subsets and one uncovered element) next to a classical solution with four subsets. This file builds the same eight-qubit QAOA on QUMO's state vector. It checks the brute-force optimum (four subsets, attained by seven covers including the notebook's classical answer), the cost model against the notebook's five printed samples (cost 23, three subsets, exactly one uncovered element, and 23 is the lowest cost among infeasible states), the cost layer's phases against the analytic formula on all 256 basis states, and a CVaR Nelder-Mead run against the uniform-state baselines and the brute-force optimum.

Deviations from the notebook:
- The penalty is inferred, not read from the notebook. An element can be uncovered by at most one unit (the constraint sum is a non-negative integer), so every penalty form reduces to penalty * uncovered. The notebook's printed cost 23 = 3 subsets + 20 for one uncovered element fixes the weight at 20 = 2 * penalty_factor. This file uses cost(x) = |x| + 20 * (number of uncovered elements); Classiq's exact internal convention is not shown in the notebook.
- The cost layer is exp(-i gamma cost(x) / 20), built from the Walsh-Hadamard (Z-parity) expansion of the cost: CNOT ladder, RZ, CNOT ladder per parity. Classiq compiles its own phase primitive and scaling. Gamma is rescaled by 1/20 here.
- The mixer is RX(beta) on every qubit, three layers, as in a standard QAOA. The notebook does not print Classiq's mixer or its optimised angles, so its sampled distribution is not reproduced; only its printed samples and classical solution are checked.
- The optimiser is QUMO's Nelder-Mead (at most 60 iterations) on the exact CVaR of the state-vector distribution (mean cost of the lowest 70 percent of probability mass). Classiq uses its own classical optimiser on shot-sampled CVaR, and its reading of quantile = 0.7 as the kept fraction is assumed.
- The classical reference is brute force over all 256 assignments instead of the couenne solver.

Run: .venv/bin/mojo run -I . examples/frameworks/classiq/set_cover.mojo
example-meta: {"framework": "classiq", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from std.math import cos, sin
from qumo.ir.circuit import Circuit
from qumo.sim.runner import evolve
from qumo.algorithms.optimizers import NelderMead, Objective
from qumo.testing.checks import CheckLog

comptime NSETS = 8
comptime DIM = 256
comptime PENALTY_W = 20.0
comptime LAYERS = 3
comptime QUANTILE = 0.7


def sub_sets() -> List[List[Int]]:
    var s = List[List[Int]]()
    s.append([1, 2, 3, 4])
    s.append([2, 3, 4, 5])
    s.append([6, 7])
    s.append([8, 9, 10])
    s.append([1, 6, 8])
    s.append([3, 7, 9])
    s.append([4, 7, 10])
    s.append([2, 5, 8])
    return s^


def bit(x: Int, q: Int) -> Int:
    return (x >> q) & 1


def popcount(x: Int) -> Int:
    var c = 0
    var y = x
    while y > 0:
        c += y & 1
        y = y >> 1
    return c


def uncovered(x: Int) -> Int:
    """Elements 1..10 not covered by the subsets chosen in x (bit i = subset i)."""
    var ss = sub_sets()
    var n = 0
    for e in range(1, 11):
        var cov = False
        for i in range(NSETS):
            if bit(x, i) == 1:
                for v in ss[i]:
                    if v == e:
                        cov = True
        if not cov:
            n += 1
    return n


def cost_of(x: Int) -> Float64:
    return Float64(popcount(x)) + PENALTY_W * Float64(uncovered(x))


def cost_table() -> List[Float64]:
    var t = List[Float64](length=DIM, fill=0.0)
    for x in range(DIM):
        t[x] = cost_of(x)
    return t^


def walsh(cost: List[Float64]) -> List[Float64]:
    """Return w[s] with cost(x) = sum_s w[s] (-1)^popcount(s & x)."""
    var w = List[Float64](length=DIM, fill=0.0)
    for s in range(DIM):
        var acc = 0.0
        for x in range(DIM):
            if popcount(s & x) % 2 == 0:
                acc += cost[x]
            else:
                acc -= cost[x]
        w[s] = acc / Float64(DIM)
    return w^


def cost_layer(mut c: Circuit, w: List[Float64], gamma: Float64):
    """Apply exp(-i gamma cost(x) / 20) up to a global phase: one parity RZ per nonzero Walsh term."""
    for s in range(1, DIM):
        if abs(w[s]) < 1e-12:
            continue
        var qs = List[Int]()
        for q in range(NSETS):
            if bit(s, q) == 1:
                qs.append(q)
        var tgt = qs[len(qs) - 1]
        for k in range(len(qs) - 1):
            c.cx(qs[k], tgt)
        c.rz(2.0 * gamma * w[s] / PENALTY_W, tgt)
        for k in range(len(qs) - 1):
            c.cx(qs[len(qs) - 2 - k], tgt)


def qaoa_circuit(w: List[Float64], params: List[Float64]) -> Circuit:
    """Build the ansatz from params = gammas (3 layers), then betas (3 layers)."""
    var c = Circuit(NSETS, name="set_cover_qaoa")
    for q in range(NSETS):
        c.h(q)
    for l in range(LAYERS):
        cost_layer(c, w, params[l])
        for q in range(NSETS):
            c.rx(params[LAYERS + l], q)
    return c^


def cvar(probs: List[Float64], cost: List[Float64], alpha: Float64) -> Float64:
    """Mean cost over the lowest `alpha` fraction of probability mass."""
    var order = List[Int]()
    for i in range(len(cost)):
        order.append(i)
    # insertion sort by cost
    for i in range(1, len(order)):
        var key = order[i]
        var j = i - 1
        while j >= 0 and cost[order[j]] > cost[key]:
            order[j + 1] = order[j]
            j -= 1
        order[j + 1] = key
    var remaining = alpha
    var acc = 0.0
    for k in range(len(order)):
        var i = order[k]
        var take = probs[i]
        if take > remaining:
            take = remaining
        acc += take * cost[i]
        remaining -= take
        if remaining <= 0.0:
            break
    return acc / alpha


struct CvarObjective(Objective, Movable):
    var w: List[Float64]
    var cost: List[Float64]
    var nfev: Int

    def __init__(out self, var w: List[Float64], var cost: List[Float64]):
        self.w = w^
        self.cost = cost^
        self.nfev = 0

    def num_params(self) -> Int:
        return 2 * LAYERS

    def value(mut self, x: List[Float64]) raises -> Float64:
        self.nfev += 1
        var pr = evolve(qaoa_circuit(self.w, x)).probabilities()
        return cvar(pr, self.cost, QUANTILE)

    def value_and_gradient(mut self, x: List[Float64], mut grad: List[Float64]) raises -> Float64:
        var f0 = self.value(x)
        var xp = x.copy()
        var h = 1e-6
        for i in range(len(x)):
            xp[i] = x[i] + h
            grad[i] = (self.value(xp) - f0) / h
            xp[i] = x[i]
        return f0


def vec_to_x(v: List[Int]) -> Int:
    var x = 0
    for q in range(len(v)):
        x = x | (v[q] << q)
    return x


def main() raises:
    var log = CheckLog("classiq/set_cover")
    var cost = cost_table()
    var w = walsh(cost)

    # ---- Brute force and the notebook's printed output ----
    var best = 1000000.0
    var n_best = 0
    var min_infeasible = 1000000.0
    for x in range(DIM):
        if uncovered(x) == 0:
            if cost[x] < best:
                best = cost[x]
                n_best = 1
            elif cost[x] == best:
                n_best += 1
        else:
            min_infeasible = min(min_infeasible, cost[x])
    var classical = vec_to_x([1, 1, 1, 1, 0, 0, 0, 0])
    print("brute-force minimum cover size", best, "attained by", n_best, "assignment(s)")
    _ = log.check("minimum cover has 4 subsets, attained by 7 covers", best == 4.0 and n_best == 7)
    _ = log.check("the notebook's classical solution [1,1,1,1,0,0,0,0] is one of them", uncovered(classical) == 0 and popcount(classical) == 4)

    var printed = List[List[Int]]()
    printed.append([0, 1, 0, 1, 1, 0, 0, 0])
    printed.append([0, 1, 0, 1, 1, 0, 0, 0])
    printed.append([0, 1, 0, 0, 1, 0, 1, 0])
    printed.append([1, 0, 1, 1, 0, 0, 0, 0])
    printed.append([0, 1, 0, 0, 1, 0, 1, 0])
    for v in printed:
        var x = vec_to_x(v)
        _ = log.close("notebook sample cost is 23", cost[x], 23.0, 0.0)
        _ = log.check("notebook sample: 3 subsets and exactly one uncovered element", popcount(x) == 3 and uncovered(x) == 1)
    _ = log.close("23 is the lowest cost among infeasible states", min_infeasible, 23.0, 0.0)

    # ---- Cost layer phases against exp(-i gamma cost / 20) on all 256 states ----
    var g = 0.37
    var probe = Circuit(NSETS, name="cost_probe")
    for q in range(NSETS):
        probe.h(q)
    cost_layer(probe, w, g)
    var sv = evolve(probe)
    var dev = 0.0
    for x in range(DIM):
        var d = -g * (cost[x] - cost[0]) / PENALTY_W
        var rr = sv.re[x] * sv.re[0] + sv.im[x] * sv.im[0]
        var ii = sv.im[x] * sv.re[0] - sv.re[x] * sv.im[0]
        dev = max(dev, abs(rr - cos(d) / Float64(DIM)))
        dev = max(dev, abs(ii - sin(d) / Float64(DIM)))
    _ = log.close("cost layer phases match exp(-i gamma cost / 20) on all 256 states", dev, 0.0, 1e-12)

    # ---- CVaR-QAOA ----
    var uniform_mean = 0.0
    var uni = List[Float64](length=DIM, fill=1.0 / Float64(DIM))
    for x in range(DIM):
        uniform_mean += cost[x] / Float64(DIM)
    var uniform_cvar = cvar(uni, cost, QUANTILE)
    var obj = CvarObjective(w.copy(), cost.copy())
    var x0: List[Float64] = [0.2, 0.5, 0.8, 0.8, 0.5, 0.2]
    var opt = NelderMead(maxiter=60, initial_step=0.5).minimize(obj, x0)
    var pr = evolve(qaoa_circuit(w, opt.x)).probabilities()
    var mean_c = 0.0
    var p_opt = 0.0
    var p_feas = 0.0
    var uni_feas = 0.0
    for x in range(DIM):
        mean_c += pr[x] * cost[x]
        if uncovered(x) == 0 and popcount(x) == 4:
            p_opt += pr[x]
        if uncovered(x) == 0:
            p_feas += pr[x]
            uni_feas += 1.0 / Float64(DIM)
    print("uniform: mean cost", uniform_mean, " CVaR", uniform_cvar, " P(feasible)", uni_feas)
    print("QAOA:    mean cost", mean_c, " CVaR", opt.fun, " P(feasible)", p_feas, " P(optimum)", p_opt, " nfev", obj.nfev)
    _ = log.check("optimised CVaR beats the uniform state", opt.fun < uniform_cvar, "qaoa=" + String(opt.fun) + " uniform=" + String(uniform_cvar))
    _ = log.check("optimised mean cost beats the uniform state", mean_c < uniform_mean, "qaoa=" + String(mean_c) + " uniform=" + String(uniform_mean))
    _ = log.check("optimised P(minimum cover) beats the uniform 7/256", p_opt > 7.0 / 256.0, "p=" + String(p_opt))
    _ = log.check("optimised P(feasible) beats the uniform state", p_feas > uni_feas, "p=" + String(p_feas) + " uniform=" + String(uni_feas))

    log.finish()
