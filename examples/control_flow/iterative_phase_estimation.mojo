"""Iterative phase estimation: one ancilla, and a gate angle computed from its own results.

Reproduces: Dobsicek, Johansson, Shumeiko, Wendin - Arbitrary accuracy iterative phase estimation algorithm as a two qubit benchmark
Source: https://arxiv.org/abs/quant-ph/0610214
Framework: QUMO - https://github.com/ANFAIA/QUMO

Textbook phase estimation of U = P(2 pi phi) on its eigenstate |1> takes one ancilla
per bit. The iterative form takes one: it reuses the ancilla n times, least
significant bit first, and each round rotates it by a feedback angle that depends on
the bits read so far. That angle `w` is known only while the shot runs, so it is a
classical variable feeding a gate:

    for i in 0..n-1:  reset a; h a; ctrl-U^(2^(n-1-i)); p(w) a; h a; b_i = measure a
                      w = w / 2 - b_i * pi / 2

The estimate m = sum b_i 2^i is distributed as in textbook estimation:

    P(m) = | 2^-n sum_{k < 2^n} exp(2 pi i k (phi - m / 2^n)) |^2

Checks:
- phi = 45/256 and phi = 5/16 are exact in 8 bits: every shot reads the right value;
- for phi = 0.3, the estimates follow P(m) above, in total variation under the
  Weissman bound for delta = 1e-6, fixed before running, and so do those of the
  textbook circuit with 8 ancillas (`qumo.algorithms.library.qpe`);
- with the feedback removed, the law is measurably wrong: the variable is doing the work;
- the program compiles for a line and gives the same law. Every block here starts
  with a reset, so verify_program checks its layouts and structure, not its gates:
  the law is what shows the compiled gates are right.

Run: .venv/bin/mojo run -I . examples/control_flow/iterative_phase_estimation.mojo
example-meta: {"framework": "qumo", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from std.math import cos, log, pi, sin, sqrt

from qumo.algorithms.library import bitstring_to_int, qpe
from qumo.compiler.program_compile import compact_qubits, compile_program
from qumo.compiler.target import parse_target
from qumo.ir.circuit import Circuit
from qumo.ir.program import Program
from qumo.ir.qasm import load_any
from qumo.sim.program_runner import sample_program
from qumo.sim.runner import sample
from qumo.testing.checks import CheckLog

comptime BITS = 8
comptime SHOTS = 40000


def ipe_program(feedback: Bool) -> String:
    var update = "  if (c[0]) {\n    w = w / 2 - pi / 2;\n  } else {\n    w = w / 2;\n  }\n" if feedback else ""
    return (
        "OPENQASM 3.0;\ninclude \"stdgates.inc\";\ninput float phi;\nqubit[2] q;\nbit[1] c;\n"
        + "output uint[8] estimate = 0;\nfloat w = 0.0;\nx q[1];\n"
        + "for int i in [0:" + String(BITS - 1) + "] {\n"
        + "  reset q[0];\n  h q[0];\n"
        + "  cp(2 * pi * phi * 2 ** (" + String(BITS - 1) + " - i)) q[0], q[1];\n"
        + "  p(w) q[0];\n  h q[0];\n  c[0] = measure q[0];\n"
        + "  estimate = estimate | (uint[8](c[0]) << uint[8](i));\n"
        + update
        + "}\n"
    )


def textbook_law(phi: Float64) -> List[Float64]:
    var n = 1 << BITS
    var out = List[Float64](capacity=n)
    for m in range(n):
        var re = 0.0
        var im = 0.0
        var step = 2.0 * pi * (phi - Float64(m) / Float64(n))
        for k in range(n):
            re += cos(step * Float64(k))
            im += sin(step * Float64(k))
        out.append((re * re + im * im) / Float64(n * n))
    return out^


def weissman(outcomes: Int, shots: Int) -> Float64:
    """Total variation of an empirical law past this has probability below 1e-6."""
    return sqrt((Float64(outcomes) * log(2.0) + log(1e6)) / (2.0 * Float64(shots)))


def distance(values: List[Int], law: List[Float64]) -> Float64:
    var hist = List[Int](length=len(law), fill=0)
    for v in values:
        hist[v] += 1
    var t = 0.0
    for m in range(len(law)):
        t += abs(Float64(hist[m]) / Float64(len(values)) - law[m])
    return t / 2.0


def estimates(p: Program, phi: Float64, seed: UInt64) raises -> List[Int]:
    var params: List[Float64] = [phi]
    return sample_program(p, SHOTS, seed, params).int_output("estimate")


def main() raises:
    var log_ = CheckLog("iterative_phase_estimation")
    var p = load_any(ipe_program(True))
    _ = log_.check("a program, not a circuit", p.has_runtime_region())

    var exact: List[Int] = [45, 80]  # 45/256 and 5/16 = 80/256
    for want in exact:
        var got = estimates(p, Float64(want) / Float64(1 << BITS), 21)
        var hits = 0
        for v in got:
            if v == want:
                hits += 1
        _ = log_.check("phi = " + String(want) + "/256 reads exactly", hits == SHOTS, "hits=" + String(hits))

    var phi = 0.3
    var law = textbook_law(phi)
    var limit = weissman(1 << BITS, SHOTS)
    _ = log_.within("iterative estimates follow the textbook law, total variation", distance(estimates(p, phi, 22), law), 0.0, limit)

    var u = Circuit(1)
    u.p(2.0 * pi * phi, 0)
    var prep = Circuit(1)
    prep.x(0)
    var textbook = sample(qpe(u, BITS, prep.copy(), measure=True), SHOTS, seed=23)
    var values = List[Int](capacity=SHOTS)
    for ref k in textbook.sorted_keys():
        for _ in range(textbook.counts.get(k, 0)):
            values.append(bitstring_to_int(k))
    _ = log_.within("eight-ancilla circuit follows the same law, total variation", distance(values, law), 0.0, limit)

    var blind = load_any(ipe_program(False))
    _ = log_.within("without feedback the law is wrong, total variation", distance(estimates(blind, phi, 24), law), 0.2, 1.0)

    var compiled = compile_program(p, parse_target("line:3/rz,sx,x,cz"), 2, 7)
    _ = log_.check("layouts and structure verified", compiled.blocks_unchecked > 0, String(compiled))
    _ = log_.within("compiled program follows the law, total variation", distance(estimates(compact_qubits(compiled.program), phi, 25), law), 0.0, limit)
    log_.finish()
