"""Hello, QUMO: build a Bell pair, look at it, sample it, compile it.

Reproduces: QUMO - a Bell pair, the standard first quantum circuit (step 1 of the tutorial)
Source: https://github.com/ANFAIA/QUMO/blob/main/docs/TUTORIAL.md
Framework: QUMO - https://github.com/ANFAIA/QUMO

The first Mojo program to read. Five short steps, each printing what it found:

  1. build a 2-qubit circuit (H, then CX: a Bell pair)
  2. draw it
  3. get the exact state vector and one expectation value
  4. sample it, as a real device would
  5. compile it for a 2x2 grid of superconducting qubits

A Bell pair is two qubits that always agree: you read 00 half of the time and 11 the
other half, never 01 or 10. Expected output (the counts depend on the seed):

    {"00": 484, "11": 516}

Run:  .venv/bin/mojo run -I . examples/foundations/hello_bell.mojo
The first run takes a few seconds: Mojo compiles the file, then runs it. The harmless
"Failed to initialize Crashpad" line can be ignored.

example-meta: {"framework": "qumo", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

# One import brings everything a first program needs. The same names also live in the
# modules that define them (qumo.ir.circuit, qumo.sim.runner, qumo.compiler.pass_manager,
# ...); see qumo/ir if you want to work on the circuit representation on its own.
from qumo import Circuit, statevector, expectation, parse_pauli_sum, sample_auto, parse_target, transpile
from qumo.testing.checks import CheckLog


def main() raises:
    var log = CheckLog("hello_bell")

    # 1. Build the circuit. All qubits start in |0>.
    var c = Circuit(2)
    c.h(0)             # put qubit 0 in an equal superposition of 0 and 1
    c.cx(0, 1)         # flip qubit 1 when qubit 0 is 1: the two qubits are now entangled
    c.measure_all()    # read both qubits at the end
    print("1. circuit with", len(c), "operations")

    # 2. Draw it.
    print("2. drawing")
    print(c.draw())

    # 3. The exact state, with no sampling noise. The final measurements are ignored here.
    var sv = statevector(c)
    print("3. state vector")
    print(sv)
    var zz = expectation(c, parse_pauli_sum("Z0 Z1"))
    print("   <Z0 Z1> =", zz, "(+1: the two qubits always agree)")

    # 4. Sample it. The seed makes the counts repeatable.
    var r = sample_auto(c, 1000, seed=7)
    print("4. 1000 shots:", r)
    var p00 = r.probability("00")

    # 5. Compile for a device: a 2x2 grid whose native two-qubit gate is CZ.
    var out = transpile(c, parse_target("grid:2x2/sc_cz"))
    print("5. compiled for grid:2x2/sc_cz:")
    print(out)

    # What this file checks, so the example runner can verify it.
    _ = log.close("<Z0 Z1>", zz, 1.0, 1e-9)
    _ = log.within("P(00)", p00, 0.45, 0.55)
    _ = log.check("only 00 and 11 appear", r.counts.get("00", 0) + r.counts.get("11", 0) == 1000)
    _ = log.within("estimated fidelity of the compiled circuit", out.est_fidelity, 0.9, 1.0)
    log.finish()
