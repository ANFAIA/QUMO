"""Max clique with CVaR-QAOA on a 7-node random graph, exact on QUMO.

Reproduces: Classiq library - Max Clique Problem
Source: https://github.com/Classiq/classiq-library/blob/main/applications/optimization/max_clique/max_clique.ipynb (Classiq Library, MIT)

The notebook builds a Pyomo model on networkx.erdos_renyi_graph(7, 0.6, seed=79): binary x_i per node, the constraint x^T C x == 0 (C marks the non-edges, so no two chosen nodes may be non-adjacent) and the objective max sum x_i. Classiq turns it into a 3-layer QAOA and minimises the CVaR of the cost (quantile 0.7, at most 50 iterations). It prints the five cheapest samples (cost -4 for {1,2,3,5} and {0,1,2,3}, then cost -3), and a classical solver returns the clique [1, 1, 1, 1, 0, 0, 0]. This file rebuilds the same graph, writes the same problem as a diagonal cost Hamiltonian, runs the 3-layer QAOA on the QUMO state vector, minimises the exact CVaR, and checks: the edge list (15 edges) reproduces the notebook's printed cliques and costs, the brute-force maximum clique has size 4 with exactly the two cliques the notebook shows, no 5-clique exists, the cost layer's phases against the analytic formula on all 128 basis states, and that the optimised circuit beats the uniform state in expected cost, CVaR and probability of a maximum clique, and that 1000 samples contain a maximum clique.

Deviations from the notebook:
- networkx is not available here. The graph is generated with Python's random.Random(79) over the node pairs in lexicographic order, which is how networkx.gnp_random_graph draws edges; the 15 edges are written as a constant. The checks against the notebook's printed cliques and costs confirm that this is the notebook's graph.
- Classiq's translator turns the equality constraint into a penalty with its own weight and its own cost normalisation, which the notebook does not print. This file uses cost(x) = -sum x_i + 2 * (number of chosen non-adjacent pairs). A weight above 1 makes the minimum of the cost the maximum clique, and for a clique the cost is -size, as in the notebook's table.
- The cost layer is one RZ per node and one RZZ per non-edge, equal to exp(-i gamma cost) up to a global phase. The mixer is RX(beta) on every qubit. Classiq's mixer parametrisation is not printed, so the angles are not comparable and the notebook's sampled probabilities (0.0078 for each best clique, close to uniform) are not compared.
- The optimiser is QUMO's Nelder-Mead on the exact CVaR (alpha = 0.7) from the state vector, at most 300 iterations, from a linear-ramp start. The notebook uses a shot-sampled CVaR with at most 50 iterations of its own optimiser.

Run: .venv/bin/mojo run -I . examples/frameworks/classiq/max_clique.mojo
example-meta: {"framework": "classiq", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from std.math import cos, sin
from qumo.ir.circuit import Circuit
from qumo.sim.runner import evolve, sample
from qumo.algorithms.library import bitstring_to_int
from qumo.algorithms.optimizers import NelderMead, Objective
from qumo.testing.checks import CheckLog

comptime N = 7
comptime LAYERS = 3
comptime PENALTY = 2.0
comptime ALPHA = 0.7


def bit(x: Int, q: Int) -> Int:
    return (x >> q) & 1


def edge_list() -> List[Tuple[Int, Int]]:
    """The graph networkx.erdos_renyi_graph(7, 0.6, seed=79) returns."""
    var e: List[Int] = [0, 1, 0, 2, 0, 3, 0, 4, 0, 6, 1, 2, 1, 3, 1, 5, 1, 6, 2, 3, 2, 5, 3, 4, 3, 5, 4, 5, 5, 6]
    var out = List[Tuple[Int, Int]]()
    for k in range(len(e) // 2):
        out.append((e[2 * k], e[2 * k + 1]))
    return out^


def adjacent(i: Int, j: Int) -> Bool:
    for e in edge_list():
        if (e[0] == i and e[1] == j) or (e[0] == j and e[1] == i):
            return True
    return False


def non_edges() -> List[Tuple[Int, Int]]:
    var out = List[Tuple[Int, Int]]()
    for i in range(N):
        for j in range(i + 1, N):
            if not adjacent(i, j):
                out.append((i, j))
    return out^


def size_of(x: Int) -> Int:
    var s = 0
    for q in range(N):
        s += bit(x, q)
    return s


def violations(x: Int) -> Int:
    var v = 0
    for p in non_edges():
        if bit(x, p[0]) == 1 and bit(x, p[1]) == 1:
            v += 1
    return v


def cost_of(x: Int) -> Float64:
    return -Float64(size_of(x)) + PENALTY * Float64(violations(x))


def cost_table() -> List[Float64]:
    var out = List[Float64](length=1 << N, fill=0.0)
    for x in range(1 << N):
        out[x] = cost_of(x)
    return out^


def vec_to_x(v: List[Int]) -> Int:
    var x = 0
    for q in range(len(v)):
        x = x | (v[q] << q)
    return x


# ---------------------------------------------------------------------------
# Circuit: cost = const + sum h_i z_i + sum J_ij z_i z_j with x_i = (1 - z_i) / 2.
# -sum x_i gives h_i = +1/2; P x_i x_j gives P/4 (1 - z_i - z_j + z_i z_j).
# exp(-i gamma cost) = RZ(2 gamma h_i) per node, RZZ(2 gamma J) per non-edge.
# ---------------------------------------------------------------------------
def cost_layer(mut c: Circuit, gamma: Float64):
    var h = List[Float64](length=N, fill=0.5)
    for p in non_edges():
        h[p[0]] -= PENALTY / 4.0
        h[p[1]] -= PENALTY / 4.0
        c.rzz(2.0 * gamma * PENALTY / 4.0, p[0], p[1])
    for q in range(N):
        c.rz(2.0 * gamma * h[q], q)


def qaoa_circuit(params: List[Float64]) -> Circuit:
    """Ansatz with params = gammas (LAYERS), then betas (LAYERS)."""
    var c = Circuit(N, name="max_clique_qaoa")
    for q in range(N):
        c.h(q)
    for l in range(LAYERS):
        cost_layer(c, params[l])
        for q in range(N):
            c.rx(2.0 * params[LAYERS + l], q)
    return c^


def cvar(probs: List[Float64], cost: List[Float64], alpha: Float64) -> Float64:
    """Mean cost of the lowest `alpha` probability mass (costs are integers here)."""
    var mass = List[Float64](length=128, fill=0.0)
    for x in range(len(probs)):
        mass[Int(cost[x]) + 8] += probs[x]
    var remaining = alpha
    var acc = 0.0
    for k in range(128):
        if remaining <= 0.0:
            break
        var take = min(mass[k], remaining)
        acc += take * Float64(k - 8)
        remaining -= take
    return acc / alpha


struct CvarObjective(Objective, Movable):
    var cost: List[Float64]
    var nfev: Int

    def __init__(out self, var cost: List[Float64]):
        self.cost = cost^
        self.nfev = 0

    def num_params(self) -> Int:
        return 2 * LAYERS

    def value(mut self, x: List[Float64]) raises -> Float64:
        self.nfev += 1
        return cvar(evolve(qaoa_circuit(x)).probabilities(), self.cost, ALPHA)

    def value_and_gradient(mut self, x: List[Float64], mut grad: List[Float64]) raises -> Float64:
        var f0 = self.value(x)
        var xp = x.copy()
        for i in range(len(x)):
            xp[i] = x[i] + 1e-6
            grad[i] = (self.value(xp) - f0) / 1e-6
            xp[i] = x[i]
        return f0


def main() raises:
    var log = CheckLog("classiq/max_clique")
    var cost = cost_table()
    var dim = 1 << N

    # ---- Graph and brute force ----
    var n_edges = len(edge_list())
    var best_size = 0
    var best_set = List[Int]()
    var has_five = False
    for x in range(dim):
        if violations(x) == 0:
            var s = size_of(x)
            if s > best_size:
                best_size = s
                best_set = List[Int]()
            if s == best_size:
                best_set.append(x)
            if s >= 5:
                has_five = True
    print("graph:", n_edges, "edges; brute-force max clique size", best_size, "with", len(best_set), "cliques")
    _ = log.check("graph has 15 edges", n_edges == 15)
    _ = log.check("brute-force maximum clique has size 4", best_size == 4)
    _ = log.check("no clique of size 5 exists", not has_five)

    # The notebook's five cheapest samples and its classical solution.
    var nb_vecs = List[List[Int]]()
    nb_vecs.append([0, 1, 1, 1, 0, 1, 0])
    nb_vecs.append([1, 1, 1, 1, 0, 0, 0])
    nb_vecs.append([0, 1, 1, 0, 0, 1, 0])
    nb_vecs.append([1, 0, 1, 1, 0, 0, 0])
    nb_vecs.append([0, 1, 0, 1, 0, 1, 0])
    var nb_costs: List[Int] = [-4, -4, -3, -3, -3]
    for i in range(5):
        var x = vec_to_x(nb_vecs[i])
        _ = log.check(
            "notebook sample " + String(i) + " is a clique with cost " + String(nb_costs[i]),
            violations(x) == 0 and Int(cost[x]) == nb_costs[i],
        )
    var x_first = vec_to_x(nb_vecs[0])
    var x_second = vec_to_x(nb_vecs[1])
    _ = log.check(
        "maximum cliques are exactly the notebook's {1,2,3,5} and {0,1,2,3}",
        len(best_set) == 2 and ((best_set[0] == x_first and best_set[1] == x_second) or (best_set[0] == x_second and best_set[1] == x_first)),
    )
    var min_cost = 0.0
    var n_min = 0
    for x in range(dim):
        if cost[x] < min_cost:
            min_cost = cost[x]
            n_min = 0
        if cost[x] == min_cost:
            n_min += 1
    _ = log.check("cost minimum is -4 and only the two maximum cliques reach it", min_cost == -4.0 and n_min == 2)

    # ---- Cost layer phases against exp(-i gamma cost(x)) on all 128 states ----
    var g = 0.37
    var probe = Circuit(N, name="max_clique_cost_probe")
    for q in range(N):
        probe.h(q)
    cost_layer(probe, g)
    var sv = evolve(probe)
    var dev = 0.0
    for x in range(dim):
        var d = -g * (cost[x] - cost[0])
        var rr = sv.re[x] * sv.re[0] + sv.im[x] * sv.im[0]
        var ii = sv.im[x] * sv.re[0] - sv.re[x] * sv.im[0]
        dev = max(dev, abs(rr - cos(d) / Float64(dim)))
        dev = max(dev, abs(ii - sin(d) / Float64(dim)))
    _ = log.close("cost layer phases match exp(-i gamma cost) on all 128 states", dev, 0.0, 1e-12)

    # ---- CVaR-QAOA ----
    var uniform = List[Float64](length=dim, fill=1.0 / Float64(dim))
    var uniform_exp = 0.0
    for x in range(dim):
        uniform_exp += cost[x] / Float64(dim)
    var uniform_cvar = cvar(uniform, cost, ALPHA)
    var uniform_opt = Float64(len(best_set)) / Float64(dim)

    var obj = CvarObjective(cost.copy())
    var x0: List[Float64] = [0.1, 0.2, 0.3, 0.3, 0.2, 0.1]
    var res = NelderMead(maxiter=300, initial_step=0.4).minimize(obj, x0)
    var pr = evolve(qaoa_circuit(res.x)).probabilities()
    var opt_cvar = cvar(pr, cost, ALPHA)
    var opt_exp = 0.0
    var p_opt = 0.0
    for x in range(dim):
        opt_exp += pr[x] * cost[x]
    for x in best_set:
        p_opt += pr[x]
    print("uniform: <cost> =", uniform_exp, " CVaR =", uniform_cvar, " P(max clique) =", uniform_opt)
    print("QAOA   : <cost> =", opt_exp, " CVaR =", opt_cvar, " P(max clique) =", p_opt, " nfev =", obj.nfev)
    _ = log.check("optimised CVaR beats the uniform state", opt_cvar < uniform_cvar - 0.1, "cvar=" + String(opt_cvar) + " uniform=" + String(uniform_cvar))
    _ = log.check("optimised expected cost beats the uniform state", opt_exp < uniform_exp - 0.1, "exp=" + String(opt_exp) + " uniform=" + String(uniform_exp))
    _ = log.check("optimised P(maximum clique) beats the uniform state", p_opt > 2.0 * uniform_opt, "p=" + String(p_opt) + " uniform=" + String(uniform_opt))

    var meas = qaoa_circuit(res.x)
    meas.measure_all()
    var out = sample(meas, 1000, UInt64(79))
    var best_sample = 0
    var best_sample_cost = 1.0e9
    for kv in out.counts.items():
        var x = bitstring_to_int(kv.key)
        if cost[x] < best_sample_cost:
            best_sample_cost = cost[x]
            best_sample = x
    print("best sampled cost", best_sample_cost, "with nodes of mask", best_sample)
    _ = log.check("best of 1000 samples is a maximum clique", violations(best_sample) == 0 and size_of(best_sample) == best_size)

    log.finish()
