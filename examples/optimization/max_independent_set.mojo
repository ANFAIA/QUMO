"""Maximum independent set with CVaR-QAOA on an 8-node graph, exact on QUMO.

Reproduces: Classiq library - Max Independent Set
Source: https://github.com/Classiq/classiq-library/blob/main/applications/optimization/max_independent_set/max_independent_set.ipynb (Classiq Library, MIT)
Framework: Classiq - https://github.com/Classiq/classiq-library

The notebook states max sum x_v subject to x_u + x_v <= 1 on every edge as a Pyomo model, lets Classiq's CombinatorialProblem turn it into a 3-layer QAOA, optimises it with CVaR (quantile 0.7, at most 60 iterations), samples it, and compares with the classical solver Couenne (independent set of size 3). The graph is networkx fast_gnp_random_graph(8, 0.4, seed=12345); its 14 edges are read from the notebook's printed constraint list. This file builds the same 3-layer QAOA on QUMO's state vector with a penalised diagonal cost, optimises the exact CVaR with Nelder-Mead, and checks: the edge count and the printed edges; the brute-force maximum independent set size 3 (the notebook's classical answer {0, 1, 3} is one of them, and so are the five sampled sets it prints, whose cost is -3); the cost layer's phases against exp(-i gamma cost) on all 256 basis states; that the penalised cost is minimised exactly on the maximum independent sets; and that the optimised QAOA beats the uniform state in probability of a maximum independent set, with the best of 1000 samples being a maximum independent set.

Deviations from the notebook:
- Classiq builds its own penalised Hamiltonian from the Pyomo model and does not print the penalty. Here cost(x) = -|x| + 2 * (number of edges with both ends selected). Any penalty above 1 makes the minimum of the cost the maximum independent set; 2 is this file's choice.
- The cost layer is RZ and RZZ rotations from the Ising form of that cost, equal to exp(-i gamma cost) up to a global phase (phase check on all 256 states). Classiq compiles its own primitive.
- The mixer is RX(2 beta) on every qubit, the standard QAOA mixer; the starting state is H on every qubit. Classiq's mixer convention is not printed.
- The optimiser is QUMO's Nelder-Mead (at most 60 iterations, initial step 0.5) on the exact CVaR of the state vector with the lowest-cost 70 percent of the probability mass, from a linear-ramp start. The notebook uses Classiq's own optimiser on shot-sampled estimates, and its quantile convention is not printed, so the tail fraction 0.7 is an interpretation. The notebook's sampled probabilities and angles are not printed in a form that can be reused, so the sample probabilities are not compared with it.
- The notebook plots are not reproduced. The classical solver Couenne is replaced by brute force over the 256 bit strings.

Run: .venv/bin/mojo run -I . examples/optimization/max_independent_set.mojo
example-meta: {"framework": "classiq", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from std.math import cos, sin
from qumo.ir.circuit import Circuit
from qumo.sim.runner import evolve, sample
from qumo.algorithms.library import bitstring_to_int
from qumo.algorithms.optimizers import NelderMead, Objective
from qumo.testing.checks import CheckLog

comptime N = 8
comptime PENALTY = 2.0
comptime LAYERS = 3
comptime ALPHA = 0.7


def bit(x: Int, q: Int) -> Int:
    return (x >> q) & 1


def edges() -> List[Tuple[Int, Int]]:
    """The 14 edges printed in the notebook's independent_rule constraint."""
    var e = List[Tuple[Int, Int]]()
    e.append((0, 2))
    e.append((0, 4))
    e.append((0, 6))
    e.append((0, 7))
    e.append((1, 2))
    e.append((1, 4))
    e.append((1, 5))
    e.append((2, 4))
    e.append((2, 5))
    e.append((2, 6))
    e.append((3, 4))
    e.append((3, 5))
    e.append((3, 6))
    e.append((5, 6))
    return e^


def set_size(x: Int) -> Int:
    var s = 0
    for q in range(N):
        s += bit(x, q)
    return s


def violations(x: Int) -> Int:
    var v = 0
    for e in edges():
        if bit(x, e[0]) == 1 and bit(x, e[1]) == 1:
            v += 1
    return v


def cost(x: Int) -> Float64:
    return -Float64(set_size(x)) + PENALTY * Float64(violations(x))


def vec_to_x(v: List[Int]) -> Int:
    var x = 0
    for q in range(len(v)):
        x = x | (v[q] << q)
    return x


def cost_layer(mut c: Circuit, gamma: Float64):
    """Phase exp(-i gamma cost) up to a global phase.
    -x_i = (Z_i - 1)/2 gives RZ(gamma) ; P x_i x_j = P (1 - Z_i - Z_j + Z_i Z_j)/4
    gives RZ(-gamma P/2) on each end and RZZ(gamma P/2)."""
    var deg = List[Int](length=N, fill=0)
    for e in edges():
        deg[e[0]] += 1
        deg[e[1]] += 1
        c.rzz(gamma * PENALTY / 2.0, e[0], e[1])
    for q in range(N):
        c.rz(gamma - gamma * PENALTY / 2.0 * Float64(deg[q]), q)


def qaoa_circuit(params: List[Float64]) -> Circuit:
    """Notebook layout: params = gammas (3 layers), then betas (3 layers)."""
    var c = Circuit(N, name="mis_qaoa")
    for q in range(N):
        c.h(q)
    for l in range(LAYERS):
        cost_layer(c, params[l])
        for q in range(N):
            c.rx(2.0 * params[LAYERS + l], q)
    return c^


def cvar(pr: List[Float64], costs: List[Float64], alpha: Float64) -> Float64:
    """Mean cost over the lowest-cost fraction alpha of the probability mass."""
    var idx = List[Int]()
    for i in range(len(pr)):
        idx.append(i)
    # insertion sort by cost
    for i in range(1, len(idx)):
        var k = idx[i]
        var j = i - 1
        while j >= 0 and costs[idx[j]] > costs[k]:
            idx[j + 1] = idx[j]
            j -= 1
        idx[j + 1] = k
    var left = alpha
    var acc = 0.0
    for t in range(len(idx)):
        var w = min(pr[idx[t]], left)
        acc += w * costs[idx[t]]
        left -= w
        if left <= 0.0:
            break
    return acc / alpha


struct CvarObjective(Objective, Movable):
    var costs: List[Float64]
    var nfev: Int

    def __init__(out self, var costs: List[Float64]):
        self.costs = costs^
        self.nfev = 0

    def num_params(self) -> Int:
        return 2 * LAYERS

    def value(mut self, x: List[Float64]) raises -> Float64:
        self.nfev += 1
        return cvar(evolve(qaoa_circuit(x)).probabilities(), self.costs, ALPHA)

    def value_and_gradient(mut self, x: List[Float64], mut grad: List[Float64]) raises -> Float64:
        var f0 = self.value(x)
        var xp = x.copy()
        for i in range(len(x)):
            xp[i] = x[i] + 1e-6
            grad[i] = (self.value(xp) - f0) / 1e-6
            xp[i] = x[i]
        return f0


def main() raises:
    var log = CheckLog("classiq/max_independent_set")
    var dim = 1 << N
    var costs = List[Float64](length=dim, fill=0.0)
    for x in range(dim):
        costs[x] = cost(x)

    # ---- Instance and brute force ----
    _ = log.check("graph has the notebook's 14 edges", len(edges()) == 14, "edges=" + String(len(edges())))
    var best = 0
    for x in range(dim):
        if violations(x) == 0:
            best = max(best, set_size(x))
    var mis = List[Int]()
    for x in range(dim):
        if violations(x) == 0 and set_size(x) == best:
            mis.append(x)
    print("brute force: maximum independent set size", best, "with", len(mis), "maximum sets")
    _ = log.check("brute-force maximum independent set size is 3 (notebook: classical size 3)", best == 3, "best=" + String(best))

    var classical: List[Int] = [1, 1, 0, 1, 0, 0, 0, 0]
    var cx = vec_to_x(classical)
    _ = log.check("notebook's classical set {0, 1, 3} is independent and size 3", violations(cx) == 0 and set_size(cx) == 3)
    var printed = List[List[Int]]()
    printed.append([0, 1, 0, 1, 0, 0, 0, 1])
    printed.append([0, 0, 0, 0, 1, 1, 0, 1])
    printed.append([1, 1, 0, 1, 0, 0, 0, 0])
    printed.append([0, 1, 0, 0, 0, 0, 1, 1])
    printed.append([0, 0, 1, 1, 0, 0, 0, 1])
    var all_ok = True
    for v in printed:
        var x = vec_to_x(v)
        if violations(x) != 0 or set_size(x) != 3 or cost(x) != -3.0:
            all_ok = False
    _ = log.check("the five sampled sets printed by the notebook are independent, size 3, cost -3", all_ok)

    # ---- Penalised cost minimised exactly on the maximum independent sets ----
    var cmin = 0.0
    for x in range(dim):
        cmin = min(cmin, costs[x])
    var argmin_ok = cmin == -3.0
    for x in range(dim):
        var is_mis = violations(x) == 0 and set_size(x) == 3
        if (costs[x] == cmin) != is_mis:
            argmin_ok = False
    _ = log.check("penalised cost minimum -3 is attained exactly on the maximum independent sets", argmin_ok)

    # ---- Cost layer phases against exp(-i gamma cost) on all 256 states ----
    var g = 0.73
    var probe = Circuit(N, name="mis_cost_probe")
    for q in range(N):
        probe.h(q)
    cost_layer(probe, g)
    var sv = evolve(probe)
    var dev = 0.0
    for x in range(dim):
        var d = -g * (costs[x] - costs[0])
        var rr = sv.re[x] * sv.re[0] + sv.im[x] * sv.im[0]
        var ii = sv.im[x] * sv.re[0] - sv.re[x] * sv.im[0]
        dev = max(dev, abs(rr - cos(d) / Float64(dim)))
        dev = max(dev, abs(ii - sin(d) / Float64(dim)))
    _ = log.close("cost layer phases match exp(-i gamma cost) on all 256 states", dev, 0.0, 1e-12)

    # ---- CVaR-QAOA ----
    var obj = CvarObjective(costs.copy())
    var x0: List[Float64] = [0.1, 0.2, 0.3, 0.3, 0.2, 0.1]
    var opt = NelderMead(maxiter=60, initial_step=0.5).minimize(obj, x0)
    var pr = evolve(qaoa_circuit(opt.x)).probabilities()
    var p_mis = 0.0
    var e_cost = 0.0
    var e_uni = 0.0
    for x in range(dim):
        e_uni += costs[x] / Float64(dim)
        e_cost += pr[x] * costs[x]
        if violations(x) == 0 and set_size(x) == 3:
            p_mis += pr[x]
    var uni_mis = Float64(len(mis)) / Float64(dim)
    var cvar0 = cvar(evolve(qaoa_circuit(x0)).probabilities(), costs, ALPHA)
    print("optimised: P(max independent set) =", p_mis, " uniform =", uni_mis, " <cost> =", e_cost, " nfev =", obj.nfev)
    _ = log.check("optimised CVaR improves on the start", opt.fun < cvar0, "cvar=" + String(opt.fun) + " start=" + String(cvar0))
    _ = log.check("optimised probability of a maximum independent set beats uniform", p_mis > uni_mis, "p=" + String(p_mis) + " uniform=" + String(uni_mis))
    _ = log.check("optimised mean cost beats uniform", e_cost < e_uni, "cost=" + String(e_cost) + " uniform=" + String(e_uni))

    var meas = qaoa_circuit(opt.x)
    meas.measure_all()
    var res = sample(meas, 1000, UInt64(7))
    var best_cost = 0.0
    var best_x = 0
    for kv in res.counts.items():
        var x = bitstring_to_int(kv.key)
        if costs[x] < best_cost:
            best_cost = costs[x]
            best_x = x
    print("best sampled set size", set_size(best_x), "cost", best_cost)
    _ = log.check("best of 1000 samples is a maximum independent set (size 3, no violated edge)", best_cost == -3.0 and violations(best_x) == 0 and set_size(best_x) == 3)

    log.finish()
