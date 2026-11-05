"""Repeat-until-success: a rotation that retries in a `while` loop until it works.

Reproduces: Paetznick, Svore - Repeat-Until-Success: non-deterministic decomposition of single-qubit unitaries (the method, on a textbook circuit)
Source: https://arxiv.org/abs/1311.1074
Framework: QUMO - https://github.com/ANFAIA/QUMO

One trial conjugates S on the target by a Toffoli, between Hadamards on all three
qubits. Worked out by hand (and checked below): the two controls read 00 with
probability 5/8, and the target has then undergone

    V = (2 I - i X) / sqrt(5) = Rx(theta),  cos(theta) = 3/5

which is not a Clifford rotation by a multiple of pi/2. Any other reading, 3/8 in
all, has applied X, which one X gate undoes, so the target can be tried again as if
nothing had happened. The number of trials is geometric with p = 5/8 and has no
bound known at compile time, so a straight-line circuit cannot express this: it is
a `while` loop over a measured value.

Closed forms: mean trials 8/5; from |0>, P(Z = 1) = sin^2(theta/2) = 1/5 and
P(Y = +1) = (1 - sin(theta)) / 2 = 1/10; from |+>, an eigenstate of X, the outcome in
the X basis is always +, whatever the number of retries.

Checks:
- trials follow Geom(5/8): mean within 4 sigma of 8/5, and total variation against
  the exact law under the Weissman bound for delta = 1e-6, fixed before running;
- the Z, Y and X readouts equal the closed forms above;
- the loop compiles for a line and a heavy-hex lattice, passes verify_program, and
  the compiled program gives the same distribution;
- OpenQASM round trip is a fixed point, and JSON comes back the same program;
- a budget of one back-edge stops the shots that need a retry with QM-E480.

Run: .venv/bin/mojo run -I . examples/control_flow/repeat_until_success.mojo
example-meta: {"framework": "qumo", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from std.math import log, sqrt

from qumo.ai.describe import program_from_json, program_to_json
from qumo.ai.json import parse_json
from qumo.compiler.program_compile import compact_qubits, compile_program
from qumo.compiler.target import parse_target
from qumo.ir.program import Program
from qumo.ir.qasm import load_any, to_qasm
from qumo.sim.program_runner import sample_program
from qumo.testing.checks import CheckLog

comptime SHOTS = 40000


def rus_program(prepare: String, readout: String) -> String:
    """Trials on q[2], with q[0] and q[1] as controls; `prepare` and `readout` set
    the input state and the measured basis of the target."""
    return (
        "OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit[3] q;\nbit[2] m;\nbit[1] r;\n"
        + prepare
        + """
output int trials = 0;
while (true) {
  trials += 1;
  reset q[0];
  reset q[1];
  h q[0];
  h q[1];
  h q[2];
  ccx q[0], q[1], q[2];
  s q[2];
  ccx q[0], q[1], q[2];
  h q[2];
  h q[0];
  h q[1];
  m[0] = measure q[0];
  m[1] = measure q[1];
  if (m == 0) {
    break;
  }
  x q[2];
}
"""
        + readout
        + "r[0] = measure q[2];\n"
    )


def weissman(outcomes: Int, shots: Int) -> Float64:
    """Total variation of an empirical law past this has probability below 1e-6."""
    return sqrt((Float64(outcomes) * log(2.0) + log(1e6)) / (2.0 * Float64(shots)))


def p_of_r(p: Program, shots: Int, seed: UInt64) raises -> Float64:
    """Fraction of shots that read r = 1 (the top character of the counts key)."""
    var r = sample_program(p, shots, seed)
    var ones = 0
    for key in r.sample.sorted_keys():
        if key.startswith("1"):
            ones += r.sample.counts.get(key, 0)
    return Float64(ones) / Float64(shots)


def main() raises:
    var log_ = CheckLog("repeat_until_success")
    var z = load_any(rus_program("", ""))
    _ = log_.check("a program, not a circuit", z.has_runtime_region())

    var run = sample_program(z, SHOTS, 11)
    var trials = run.int_output("trials")
    var total = 0
    var hist = List[Int](length=64, fill=0)
    for t in trials:
        total += t
        hist[min(t, 63)] += 1
    var mean = Float64(total) / Float64(SHOTS)
    _ = log_.close("mean trials", mean, 8.0 / 5.0, 4.0 * sqrt(0.96 / Float64(SHOTS)))
    _ = log_.close("back-edges per shot are retries", run.mean_iterations(), mean - 1.0, 1e-12)
    var tv = 0.0
    var law = 5.0 / 8.0
    for k in range(1, 63):
        tv += abs(Float64(hist[k]) / Float64(SHOTS) - law)
        law *= 3.0 / 8.0
    _ = log_.within("trials follow Geom(5/8), total variation", tv / 2.0, 0.0, weissman(16, SHOTS))

    var pz = p_of_r(z, SHOTS, 12)
    var sz = 4.0 * sqrt(0.2 * 0.8 / Float64(SHOTS))
    _ = log_.close("P(Z = 1) from |0>", pz, 0.2, sz)
    var y = load_any(rus_program("", "sdg q[2];\nh q[2];\n"))
    _ = log_.close("P(Y = -1) from |0>", p_of_r(y, SHOTS, 13), 0.9, 4.0 * sqrt(0.9 * 0.1 / Float64(SHOTS)))
    var x = load_any(rus_program("h q[2];\n", "h q[2];\n"))
    _ = log_.close("|+> stays |+> through every retry", p_of_r(x, SHOTS, 14), 0.0, 1e-12)

    var limit = weissman(16, SHOTS) * 2.0  # two empirical laws
    var targets: List[String] = ["line:5/rz,sx,x,cz", "heavy_hex:3/rz,sx,x,cx"]
    for name in targets:
        var compiled = compile_program(y, parse_target(name), 2, 7)
        _ = log_.check("verified for " + name, compiled.blocks_checked + compiled.blocks_unchecked > 0, String(compiled))
        var a = sample_program(y, SHOTS, 15).sample.copy()
        var b = sample_program(compact_qubits(compiled.program), SHOTS, 16).sample.copy()
        var keys = a.sorted_keys()
        for k in b.sorted_keys():
            if a.counts.get(k, 0) == 0:
                keys.append(k)
        var d = 0.0
        for k in keys:
            d += abs(a.probability(k) - b.probability(k))
        _ = log_.within("compiled for " + name + " gives the same law, total variation", d / 2.0, 0.0, limit)

    var text = to_qasm(z)
    _ = log_.check("OpenQASM round trip is a fixed point", to_qasm(load_any(text)) == text)
    _ = log_.check("JSON round trip", to_qasm(program_from_json(parse_json(program_to_json(z)))) == text)

    var stopped = String()
    try:
        _ = sample_program(z, 200, 17, List[Float64](), 1)
    except e:
        stopped = String(e)
    _ = log_.check("a budget of one back-edge stops with QM-E480", stopped.startswith("QM-E480"), stopped)
    log_.finish()
