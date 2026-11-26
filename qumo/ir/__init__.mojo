"""The circuit intermediate representation (IR): what a circuit is, with nothing about running it.

This package is the part of QUMO to read, test and change on its own. It depends only on the
JSON helpers in `qumo.core`; it imports nothing from the simulators, the compiler, the
algorithms, the GPU code or the hardware clients. If you write a parser, an exporter, a
transformation, a visualiser or a new front end, you can work against `qumo.ir` alone.

## Two ways to import

The package root `qumo` redirects to this package for the names a first program needs:

    from qumo import Circuit, Op, parse_qasm, to_qasm, PauliSum, parse_pauli_sum

The full module paths below are the same objects, and they say where each thing is defined:

    from qumo.ir.circuit import Circuit
    from qumo.ir.qasm import parse_qasm, to_qasm

Use the short form in scripts and notebooks. Use the full form in IR code, where it keeps an
IR-only program from pulling in the rest of QUMO by accident (`examples/foundations/ir_only.mojo`
is a small runnable example). The root `qumo` also re-exports the simulator and compiler entry
points; those are not part of the IR.

## The modules

Circuits and gates:

    circuit.mojo       Circuit, the list of operations, and Instruction: the type everything else uses
    gates.mojo         the gate set: opcodes (Op), arities, names, unitary matrices
    gateset.mojo       gate sets and the exact rewrite rules for multi-qubit gates
    mc.mojo            multi-controlled gates, expanded into existing gates when built
    param.mojo         symbolic parameters (an angle that is bound later)
    pauli.mojo         Pauli strings and sums (observables, Hamiltonians), up to 64 qubits
    kraus.mojo         noise channels beyond Pauli ones, as Kraus operators
    pauli_noise.mojo   what each Pauli noise opcode draws (one definition for every sampler)
    clifford_table.mojo  Clifford conjugation of Paulis (G^dag P G, G P G^dag), sign included
    draw.mojo          the text drawing behind `Circuit.draw()` and `qumo draw`

Text formats:

    qasm.mojo          OpenQASM 3 (and 2 on input) <-> Circuit: the facade over the three files below
    qasm_lexer.mojo    tokens, diagnostics helpers and the builtin gate table
    qasm_parser.mojo   tokens -> Circuit or Program, and the public parsing functions
    qasm_emit.mojo     Circuit or Program -> OpenQASM 3 text
    qasm2.mojo         the OpenQASM 2 emitter, for interoperability
    noise_inject.mojo  insert `#pragma qumo` noise channels into compiled OpenQASM 2 text
    json.mojo          circuit and program JSON (`qumo.circuit.v1`, `qumo.program.v1`): write and read back

Programs with control flow and neutral-atom programs:

    program.mojo       Program: circuits with `while` loops, classical variables and branches
    classical.mojo     the classical values of a Program: types, expressions, evaluator
    analog.mojo        analog neutral-atom programs: an atom register driven by Rydberg pulses

The CLI and the MCP server work on this same IR, so a circuit built here can be simulated,
compiled, drawn or exported without conversion.
"""
