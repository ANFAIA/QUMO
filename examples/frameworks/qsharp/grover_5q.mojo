"""Grover search on 5 qubits, marked |01010>, optimal iteration count (Q# sample).

Reproduces: Q# (Microsoft QDK) - Grover's search sample
Source: https://github.com/microsoft/qdk/blob/main/samples/algorithms/Grover.qs (MIT)

The Q# sample searches 2^5 = 32 items for the marked state |01010> (Q#
lists qubit 0 first: qubits 1 and 3 are 1, i.e. the QUMO integer 10; the
string is a palindrome, so QUMO's key "01010" names the same state). It
computes iterations = Round(pi / (4 asin(1/sqrt(N))) - 1/2) = 4, uses an
oracle that flips a |-> output qubit with a multi-controlled X, a diffusion
that reflects about the uniform superposition, and measures all qubits; the
sample comments that the result is [Zero, One, Zero, One, Zero] "with very
high probability".

QUMO's `grover(n, marked)` builds the equivalent phase-oracle form (an MCZ
with X conjugation instead of the kickback qubit, Toffoli V-chain with
num_mcz_ancillas(5) = 2 clean ancillas), and `grover_optimal_iterations` /
`grover_success_probability` give the textbook numbers.

Checks:
- grover_optimal_iterations(5, 1) = 4 (Q#'s formula, page);
- exact P(01010) after 4 iterations = sin^2(9 asin(1/sqrt(32))) = grover_success_probability (1e-12);
- 1,000 sampled shots: frequency of 01010 within 4 sigma, and it is the most frequent outcome;
- the success probability oscillates with the iteration count: exact P equals
  sin^2((2k+1) theta) for k = 0..12 (1e-12) and peaks at k = 4.

Run: .venv/bin/mojo run -I . examples/frameworks/qsharp/grover_5q.mojo
example-meta: {"framework": "qsharp", "tier": "fast", "timeout": 60, "requires": [], "args": []}
"""

from std.math import sqrt, asin, sin, pi
from qumo.sim.runner import evolve, sample
from qumo.algorithms.library import (
    grover, grover_optimal_iterations, grover_success_probability, bitstring_to_int,
)
from qumo.ai.checks import CheckLog

comptime N = 5
comptime SHOTS = 1000


def qsharp_bits_to_int(s: String) -> Int:
    """Q# result array order (index 0 = qubit 0) -> integer."""
    var v = 0
    var bs = s.as_bytes()
    for i in range(s.byte_length()):
        if Int(bs[i]) == ord("1"):
            v |= 1 << i
    return v


def marked_probability(iterations: Int, marked: Int) raises -> Float64:
    var marks: List[Int] = [marked]
    var probs = evolve(grover(N, marks, iterations)).probabilities()
    var p = 0.0
    for i in range(len(probs)):
        if (i & ((1 << N) - 1)) == marked:
            p += probs[i]
    return p


def main() raises:
    var log = CheckLog("qsharp/grover_5q")
    var marked = qsharp_bits_to_int("01010")
    var k_opt = grover_optimal_iterations(N, 1)
    var theta = asin(1.0 / sqrt(32.0))
    var qsharp_k = Int(pi / (4.0 * theta) - 0.5 + 0.5)  # Round(x - 1/2) as in the sample
    print("marked |01010> = integer", marked, "; optimal iterations =", k_opt, "(Q# formula:", qsharp_k, ")")
    _ = log.check("optimal iterations = 4 (page)", k_opt == 4 and qsharp_k == 4, "k=" + String(k_opt))

    var p_exact = marked_probability(k_opt, marked)
    var p_formula = sin(9.0 * theta) ** 2
    print("P(01010) after 4 iterations: state vector", p_exact, " sin^2(9 asin(1/sqrt 32)) =", p_formula)
    _ = log.close("P(marked) equals sin^2(9 theta)", p_exact, p_formula, 1e-12)
    _ = log.close("grover_success_probability matches", grover_success_probability(N, 1, k_opt), p_formula, 1e-12)

    var marks: List[Int] = [marked]
    var c = grover(N, marks, k_opt, measure=True)
    print("circuit:", c.num_qubits, "qubits (5 search + 2 ancillas),", len(c), "instructions")
    var r = sample(c, SHOTS, seed=42)
    var freq = r.probability("01010")
    var sigma = sqrt(p_formula * (1.0 - p_formula) / Float64(SHOTS))
    print("sampled", SHOTS, "shots: 01010 ->", r.counts.get("01010", 0), " most frequent", r.most_frequent())
    _ = log.within("sampled frequency within 4 sigma", abs(freq - p_formula) / sigma, 0.0, 4.0)
    _ = log.check("01010 is the most frequent outcome (page)", r.most_frequent() == "01010", r.most_frequent())

    print("oscillation of P(marked) with the iteration count:")
    var osc_err = 0.0
    var best_k = 0
    var best_p = 0.0
    for k in range(13):
        var pk = marked_probability(k, marked)
        var fk = sin((2.0 * Float64(k) + 1.0) * theta) ** 2
        osc_err = max(osc_err, abs(pk - fk))
        if pk > best_p:
            best_p = pk
            best_k = k
        var bar = String("")
        for _ in range(Int(pk * 50.0)):
            bar += "#"
        print("  k =", k, "  P =", pk, "  ", bar)
    _ = log.close("P(k) = sin^2((2k+1) theta) for k = 0..12", osc_err, 0.0, 1e-12)
    _ = log.check("the maximum over k = 0..12 is at k = 4", best_k == 4, "best k=" + String(best_k))
    log.finish()
