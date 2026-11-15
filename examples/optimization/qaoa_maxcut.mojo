"""QAOA for MaxCut: how close to the best cut, as the circuit gets deeper.

MaxCut: split the nodes of a graph into two groups so that as many edges (or as much edge
weight) as possible cross between the groups. It is hard in general, so QAOA is a standard
test of a variational quantum algorithm: a circuit with p layers whose angles a classical
optimiser tunes. For p = 1, 2, 3 this prints:

  <cut> / max   the average cut the circuit produces over the best possible cut (from brute force)
  ratio         that quotient: 1.0 would be perfect; it should grow with p
  P(optimal)    the probability of sampling a best cut
  best sample   the best cut found in 2048 shots

It runs on an 8-node ring and then on a weighted 8-node graph ("a-b:w" is an edge with weight w).

Run: .venv/bin/mojo run -I . examples/optimization/qaoa_maxcut.mojo

Framework: QUMO - https://github.com/ANFAIA/QUMO
"""

# Algorithms are one explicit import away: they live in qumo.algorithms, not in the root.
from qumo.algorithms.hamiltonians import Graph
from qumo.algorithms.variational import qaoa_maxcut


def main() raises:
    # 1. An 8-node ring: nodes 0-1-2-...-7-0. The best cut has 8 edges (alternate the groups).
    var ring = Graph.ring(8)
    print("MaxCut on", ring)
    for p in range(1, 4):
        var r = qaoa_maxcut(ring, p, restarts=4, shots=2048, seed=7)
        print("  p =", p, " <cut> =", r.expected_cut, "/", r.max_cut, " ratio =", r.approx_ratio,
              " P(optimal) =", r.prob_optimal, " best sample =", r.best_bitstring, "(cut", r.best_cut, ")",
              " time =", r.wall_time_s * 1e3, "ms")

    # 2. A weighted graph, written as edges "a-b" or "a-b:weight".
    var g = Graph.parse("0-1 0-2 0-3 1-4 1-5 2-4:1.5 2-6 3-5 3-6:0.5 4-7 5-7 6-7:2")
    print("MaxCut on weighted", g)
    for p in range(1, 4):
        var r = qaoa_maxcut(g, p, restarts=6, shots=2048, seed=3)
        print("  p =", p, " ratio =", r.approx_ratio, " P(optimal) =", r.prob_optimal,
              " best sample cut =", r.best_cut, "/", r.max_cut, " gammas =", r.gammas[0], "... betas =", r.betas[0], "...")
