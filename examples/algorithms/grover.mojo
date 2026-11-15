"""Grover search: find one marked item among N = 2^n in about sqrt(N) steps.

A classical search checks items one by one: about N/2 tries on average. Grover's algorithm
finds the marked item with high probability in about (pi/4) sqrt(N) iterations. For each size
n = 3 ... 8 qubits this prints:

  P(exact)    probability of the marked item, read from the exact state vector
  P(theory)   the textbook value sin^2((2k+1) arcsin(1/sqrt(N))) after k iterations: it matches
  sampled top the most frequent result of 1000 shots; "OK" means it is the marked item

It ends with a search for four marked items among 32.

Run: .venv/bin/mojo run -I . examples/algorithms/grover.mojo

Framework: QUMO - https://github.com/ANFAIA/QUMO
"""

from qumo import evolve, sample
from qumo.sim.runner import bitstring
# Algorithms are one explicit import away: they live in qumo.algorithms, not in the root.
from qumo.algorithms.library import (
    grover, grover_optimal_iterations, grover_success_probability, num_mcz_ancillas, bitstring_to_int,
)


def main() raises:
    print("Grover search (phase oracle via CCX ladder + clean ancillas)")
    for n in range(3, 9):
        # Pick a marked item (any number below 2^n) and the best number of iterations for it.
        var marked: List[Int] = [(37 * n + 11) % (1 << n)]
        var k = grover_optimal_iterations(n, 1)
        var c = grover(n, marked)

        # Exact: the probability of the marked item in the final state vector.
        var sv = evolve(c)
        var m = marked[0]
        var p = sv.re[m] * sv.re[m] + sv.im[m] * sv.im[m]

        # Sampled: add measurements and run 1000 shots, as a device would.
        var cm = grover(n, marked, measure=True)
        var r = sample(cm, 1000, seed=UInt64(n))
        var found = bitstring_to_int(r.most_frequent())

        print("  n =", n, " N =", 1 << n, " marked =", m, " iterations =", k, " qubits =", c.num_qubits,
              "(", num_mcz_ancillas(n), "ancillas)  gates =", len(c),
              " P(exact) =", p, " P(theory) =", grover_success_probability(n, 1, k),
              " sampled top =", found, "OK" if found == m else "MISS")

    # Several marked items: fewer iterations are needed (about sqrt(N / M)).
    var multi: List[Int] = [3, 12, 21, 30]
    var c5 = grover(5, multi, measure=True)
    var r5 = sample(c5, 2000, seed=9)
    var hits = 0
    for x in multi:
        hits += r5.counts.get(bitstring(x, 5), 0)
    print("  4 marked items in N=32:", grover_optimal_iterations(5, 4), "iterations, sampled success =", Float64(hits) / 2000.0)
