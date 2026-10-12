"""QAOA for MaxCut: approximation ratio vs depth p, with brute-force reference.

Run: .venv/bin/mojo run -I . examples/qaoa_maxcut.mojo
"""

from qumo.algorithms.hamiltonians import Graph
from qumo.algorithms.variational import qaoa_maxcut


def main() raises:
    var ring = Graph.ring(8)
    print("MaxCut on", ring)
    for p in range(1, 4):
        var r = qaoa_maxcut(ring, p, restarts=4, shots=2048, seed=7)
        print("  p =", p, " <cut> =", r.expected_cut, "/", r.max_cut, " ratio =", r.approx_ratio,
              " P(optimal) =", r.prob_optimal, " best sample =", r.best_bitstring, "(cut", r.best_cut, ")",
              " time =", r.wall_time_s * 1e3, "ms")

    var g = Graph.parse("0-1 0-2 0-3 1-4 1-5 2-4:1.5 2-6 3-5 3-6:0.5 4-7 5-7 6-7:2")
    print("MaxCut on weighted", g)
    for p in range(1, 4):
        var r = qaoa_maxcut(g, p, restarts=6, shots=2048, seed=3)
        print("  p =", p, " ratio =", r.approx_ratio, " P(optimal) =", r.prob_optimal,
              " best sample cut =", r.best_cut, "/", r.max_cut, " gammas =", r.gammas[0], "... betas =", r.betas[0], "...")
