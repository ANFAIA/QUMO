"""QAOA with a CVaR objective: optimise the best part of the shots, not their average.

Reproduces: QUMO - CVaR objective for VQE and QAOA (docs/ALGORITHMS.md, "CVaR objective")
Source: https://arxiv.org/abs/1907.04769 (Barkoutsos, Nannicini, Robert, Tavernelli, Woerner, "Improving Variational Quantum Optimization using CVaR", Quantum 4, 256, 2020)
Source: https://github.com/ANFAIA/QUMO/blob/main/docs/ALGORITHMS.md
Framework: QUMO - https://github.com/ANFAIA/QUMO

A plain QAOA run minimises the average energy of every bitstring the circuit produces. For an
optimisation problem only the best bitstrings matter, so CVaR (conditional value at risk)
averages the lowest `alpha` fraction of the probability mass instead: alpha = 1 is the plain
average, a smaller alpha looks only at the best tail.

    r = qaoa_maxcut(graph, p, alpha=0.25)      # one argument more than the plain run

This prints, for MaxCut on a weighted 8-node graph and alpha = 1, 0.5, 0.25:

  objective    the value that was minimised (the CVaR of the energy; the mean for alpha = 1)
  <cut>        the average cut of the final circuit
  P(optimal)   the probability of sampling a best cut

Then it shows the limit of the method on a 5-node ring. There the starting state already
holds 10 optimal strings of 32, so for alpha <= 0.3125 the objective starts at its minimum
and the optimiser has nothing to follow. CVaR is a different objective, not a guaranteed gain.

Checks (each against something computed without the code under test):
- `cvar_of_energies` on four hand-computed cases;
- `CVaRObjective` at fixed angles equals a CVaR computed here from the state vector and
  `Graph.cut_value`, for three values of alpha (1e-9);
- alpha = 1 equals the plain expectation value, and the value never rises as alpha shrinks;
- alpha -> 0 gives minus the brute-force maximum cut;
- `qaoa_maxcut(alpha=1.0)` returns the same energy as the call without alpha;
- for alpha < 1 the minimised value lies between -max_cut and the mean energy -<cut>;
- on the ring, the objective at the starting angles is already -max_cut for alpha = 0.2.

Run: .venv/bin/mojo run -I . examples/optimization/qaoa_cvar.mojo
example-meta: {"framework": "qumo", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from qumo.algorithms.hamiltonians import Graph, maxcut
from qumo.algorithms.library import qaoa_maxcut_ansatz
from qumo.algorithms.variational import CVaRObjective, VQEObjective, cvar_of_energies, qaoa_maxcut
from qumo.sim.runner import evolve
from qumo.testing.checks import CheckLog


def reference_cvar(g: Graph, x: List[Float64], alpha: Float64) raises -> Float64:
    """CVaR of the energy -cut(z), written out here: probabilities from the state vector,
    states visited from the largest cut down, the last one taken in part."""
    var sv = evolve(qaoa_maxcut_ansatz(g, len(x) // 2), x)
    var dim = 1 << g.num_nodes
    var used = List[Bool](length=dim, fill=False)
    var remaining = alpha
    var acc = 0.0
    while remaining > 1e-15:
        var best = -1
        var best_cut = -1.0
        for z in range(dim):
            if not used[z] and g.cut_value(z) > best_cut:
                best = z
                best_cut = g.cut_value(z)
        if best < 0:
            break
        used[best] = True
        var p = sv.re[best] * sv.re[best] + sv.im[best] * sv.im[best]
        var w = min(p, remaining)
        acc += w * (-best_cut)
        remaining -= w
    return acc / alpha


def main() raises:
    var log = CheckLog("qumo/qaoa_cvar")

    # 1. The sampled form: the mean of the lowest ceil(alpha * N) energies.
    var shots: List[Float64] = [3.0, -1.0, 2.0, 0.0]
    _ = log.close("sampled CVaR, alpha = 1 is the mean", cvar_of_energies(shots, 1.0), 1.0, 1e-12)
    _ = log.close("sampled CVaR, alpha = 0.5 averages the two lowest", cvar_of_energies(shots, 0.5), -0.5, 1e-12)
    _ = log.close("sampled CVaR, alpha = 0.25 is the lowest", cvar_of_energies(shots, 0.25), -1.0, 1e-12)
    _ = log.close("sampled CVaR, alpha = 0.3 rounds up to two shots", cvar_of_energies(shots, 0.3), -0.5, 1e-12)

    # 2. The exact form, at fixed angles, against a CVaR computed in this file.
    var g = Graph.parse("0-1 0-2 0-3 1-4 1-5 2-4:1.5 2-6 3-5 3-6:0.5 4-7 5-7 6-7:2")
    var h = maxcut(g)
    var mc = g.max_cut()
    var ansatz = qaoa_maxcut_ansatz(g, 2)
    var x: List[Float64] = [0.4, 0.7, 0.6, 0.3]  # gamma_0, gamma_1, beta_0, beta_1
    print("MaxCut on weighted", g, " max cut =", mc[0])
    var mean_obj = VQEObjective(ansatz, h)
    var mean = mean_obj.value(x)
    var prev = mean
    var falls = True
    for a in [1.0, 0.5, 0.1]:
        var obj = CVaRObjective(ansatz, h, a)
        var v = obj.value(x)
        _ = log.close("exact CVaR at fixed angles, alpha = " + String(a), v, reference_cvar(g, x, a), 1e-9)
        falls = falls and v <= prev + 1e-12
        prev = v
    var one = CVaRObjective(ansatz, h, 1.0)
    _ = log.close("alpha = 1 is the expectation value", one.value(x), mean, 1e-10)
    _ = log.check("the value never rises as alpha shrinks", falls)
    var tiny = CVaRObjective(ansatz, h, 1e-9)
    _ = log.close("alpha -> 0 is minus the maximum cut", tiny.value(x), -mc[0], 1e-9)

    # 3. The driver: the same call with one more argument.
    var plain = qaoa_maxcut(g, 2, method="nelder-mead", restarts=4, seed=3)
    var same = qaoa_maxcut(g, 2, method="nelder-mead", restarts=4, seed=3, alpha=1.0)
    _ = log.close("alpha = 1.0 is the plain run", same.energy, plain.energy, 0.0)
    print("  alpha = 1.0   objective =", plain.energy, " <cut> =", plain.expected_cut, "/", plain.max_cut,
          " P(optimal) =", plain.prob_optimal)
    for a in [0.5, 0.25]:
        var r = qaoa_maxcut(g, 2, method="nelder-mead", restarts=4, seed=3, alpha=a)
        print("  alpha =", a, "  objective =", r.energy, " <cut> =", r.expected_cut, "/", r.max_cut,
              " P(optimal) =", r.prob_optimal)
        _ = log.within("alpha = " + String(a) + ": objective between -max_cut and the mean energy",
                       r.energy, -r.max_cut - 1e-9, -r.expected_cut + 1e-9)

    # 4. Where it does not help: the objective is flat from the start.
    var ring = Graph.ring(5)
    var ring_max = ring.max_cut()
    var start: List[Float64] = [0.0, 0.0]  # no cost layer: the uniform superposition
    var flat = CVaRObjective(qaoa_maxcut_ansatz(ring, 1), maxcut(ring), 0.2)
    _ = log.close("5-node ring: alpha = 0.2 is already minimal at the start", flat.value(start), -ring_max[0], 1e-9)
    var ring_mean = qaoa_maxcut(ring, 1, seed=3)
    var ring_cvar = qaoa_maxcut(ring, 1, method="nelder-mead", seed=3, alpha=0.2)
    print("5-node ring, p = 1: P(optimal) with the mean =", ring_mean.prob_optimal,
          " with alpha = 0.2 =", ring_cvar.prob_optimal, " (the uniform state gives 0.3125)")
    log.finish()
