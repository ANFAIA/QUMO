"""The circuit IR on its own: build, inspect and convert a circuit, with no simulator or compiler.

Reproduces: QUMO - the circuit IR used on its own, without simulator or compiler
Source: https://github.com/ANFAIA/QUMO/blob/main/qumo/ir/__init__.mojo
Framework: QUMO - https://github.com/ANFAIA/QUMO

For people who work on the circuit representation itself: a parser, an exporter, a
transformation, a visualiser, a new front end. Everything here comes from `qumo.ir`, which
depends on nothing but the JSON helpers in `qumo.core`. Nothing from the simulator
(`qumo.sim`), the compiler (`qumo.compiler`) or the algorithms is imported, so this is the part
of QUMO you can read, test and change in isolation.

This file imports with the full module paths on purpose. The same names are also available
from the package root, which redirects to these modules:

    from qumo import Circuit, parse_qasm, to_qasm        # the short spelling for a first program

Use whichever you prefer; they are the same objects. The full paths say where each thing is
defined, and they keep an IR-only program from pulling in the rest of QUMO by accident.

What the module files hold (all under qumo/ir/):

    circuit.mojo   Circuit (the list of operations), Instruction
    gates.mojo     Op (the gate set), arities, names
    param.mojo     symbolic parameters
    pauli.mojo     Pauli strings and sums
    qasm.mojo      OpenQASM 3 in and out (qasm_lexer, qasm_parser, qasm_emit); qasm2.mojo for OpenQASM 2
    program.mojo   Program: circuits with while loops and classical variables
    draw.mojo      the text drawing

Run:  .venv/bin/mojo run -I . examples/foundations/ir_only.mojo

example-meta: {"framework": "qumo", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from qumo.ir.circuit import Circuit
from qumo.ir.gates import Op, op_name
from qumo.ir.qasm import parse_qasm, to_qasm
from qumo.ir.qasm2 import to_qasm2
from qumo.testing.checks import CheckLog


def main() raises:
    var log = CheckLog("ir_only")

    # A circuit is a list of instructions: build it, then look at the list.
    var c = Circuit(3)
    c.h(0)
    c.cx(0, 1)
    c.cx(1, 2)
    print("a GHZ circuit:", c.num_qubits, "qubits,", len(c), "operations, depth", c.depth())
    for i in range(len(c)):
        var ins = c.ops[i]
        var qubits = "qubit " + String(ins.q0)
        if ins.q1 >= 0:  # two-qubit gates also name a second qubit
            qubits = "qubits " + String(ins.q0) + " and " + String(ins.q1)
        print("  op", i, ":", op_name(ins.kind), "on", qubits)
    print(c.draw())

    # Text in, circuit out, and back. The IR is the same whichever way you got it.
    var qasm3 = to_qasm(c)
    print("OpenQASM 3:")
    print(qasm3)
    var again = parse_qasm(qasm3)
    print("OpenQASM 2:")
    print(to_qasm2(c))

    _ = log.check("the circuit has 3 operations", len(c) == 3, "len=" + String(len(c)))
    _ = log.check("2 of them are CX", c.count(Op.CX) == 2)
    _ = log.check("OpenQASM 3 round trip is exact", to_qasm(again) == qasm3)
    log.finish()
