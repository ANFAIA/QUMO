"""Grover search: success probability vs problem size (exact + sampled).

Run: .venv/bin/mojo run -I . examples/grover.mojo
"""

from qumo.sim.runner import evolve, sample, bitstring
from qumo.algorithms.library import (
    grover, grover_optimal_iterations, grover_success_probability, num_mcz_ancillas, bitstring_to_int,
)


def main() raises:
    print("Grover search (phase oracle via CCX ladder + clean ancillas)")
    for n in range(3, 9):
        var marked: List[Int] = [(37 * n + 11) % (1 << n)]
        var k = grover_optimal_iterations(n, 1)
        var c = grover(n, marked)
        var sv = evolve(c)
        var m = marked[0]
        var p = sv.re[m] * sv.re[m] + sv.im[m] * sv.im[m]
        var cm = grover(n, marked, measure=True)
        var r = sample(cm, 1000, seed=UInt64(n))
        var found = bitstring_to_int(r.most_frequent())
        print("  n =", n, " N =", 1 << n, " marked =", m, " iterations =", k, " qubits =", c.num_qubits,
              "(", num_mcz_ancillas(n), "ancillas)  gates =", len(c),
              " P(exact) =", p, " P(theory) =", grover_success_probability(n, 1, k),
              " sampled top =", found, "OK" if found == m else "MISS")
    var multi: List[Int] = [3, 12, 21, 30]
    var c5 = grover(5, multi, measure=True)
    var r5 = sample(c5, 2000, seed=9)
    var hits = 0
    for x in multi:
        hits += r5.counts.get(bitstring(x, 5), 0)
    print("  4 marked items in N=32:", grover_optimal_iterations(5, 4), "iterations, sampled success =", Float64(hits) / 2000.0)

