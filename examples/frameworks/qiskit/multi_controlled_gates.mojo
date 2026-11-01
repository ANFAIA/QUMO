"""Multi-controlled gates in OpenQASM 3 and from the builder API.

Reproduces: Qiskit - synth_mcx_noaux_v24, synth_mcx_1_clean_kg24, synth_mcx_n_dirty_i15 and `ctrl(k) @` export
Source: https://quantum.cloud.ibm.com/docs/en/api/qiskit/synthesis#multipliers (Qiskit docs, Apache-2.0)

`ctrl(4) @ x` is valid OpenQASM 3 that QUMO used to reject. `Instruction` holds at
most three qubits, so multi-controlled gates expand to ordinary gates when built.
Reference CX counts come from Qiskit 2.5.2 (bench/crosscheck/tier1.py mcx).

Checks:
- `ctrl(4) @ x` flips the target exactly on |1111>;
- CX counts with no ancilla at 5 controls: 62 (Qiskit's v24 needs 84);
- with one clean ancilla: 6 (k - 1), as Qiskit's kg24;
- with k - 2 dirty ancillas: 8 k - 6, as Qiskit's i15;
- idle qubits of the register are borrowed by the parser.

Run: .venv/bin/mojo run -I . examples/frameworks/qiskit/multi_controlled_gates.mojo
example-meta: {"framework": "qiskit", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from qumo.ir.circuit import Circuit
from qumo.ir.mc import append_mcx, _two_qubit_cost
from qumo.ir.qasm import parse_qasm
from qumo.sim.runner import evolve
from qumo.testing.checks import CheckLog


def range_list(a: Int, b: Int) -> List[Int]:
    var out = List[Int]()
    for i in range(a, b):
        out.append(i)
    return out^


def main() raises:
    var log = CheckLog("multi_controlled_gates")
    var c = parse_qasm("OPENQASM 3.0; include \"stdgates.inc\"; qubit[5] q; x q[0:3]; ctrl(4) @ x q[0], q[1], q[2], q[3], q[4];")
    var s = evolve(c)
    _ = log.close("ctrl(4) @ x fires on |1111>", s.re[31] * s.re[31] + s.im[31] * s.im[31], 1.0, 1e-9)
    var noaux = Circuit(6)
    append_mcx(noaux, range_list(0, 5), 5)
    _ = log.close("5 controls, no ancilla", Float64(_two_qubit_cost(noaux)), 62.0, 0.0)
    for k in [5, 10, 20]:
        var clean = Circuit(k + 2)
        append_mcx(clean, range_list(0, k), k, [k + 1], "clean", True)
        _ = log.close("1 clean ancilla, k=" + String(k), Float64(_two_qubit_cost(clean)), Float64(6 * (k - 1)), 0.0)
        var dirty = Circuit(2 * k)
        append_mcx(dirty, range_list(0, k), k, range_list(k + 1, 2 * k - 1), "dirty", False)
        _ = log.close("k-2 dirty ancillas, k=" + String(k), Float64(_two_qubit_cost(dirty)), Float64(8 * k - 6), 0.0)
    var spare = parse_qasm("OPENQASM 3.0; include \"stdgates.inc\"; qubit[15] q; ctrl(8) @ x q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];")
    _ = log.close("the parser borrows idle qubits", Float64(_two_qubit_cost(spare)), 58.0, 0.0)
    print(parse_qasm("OPENQASM 3.0; include \"stdgates.inc\"; qubit[4] q; ctrl(3) @ x q[0], q[1], q[2], q[3];").draw())
    log.finish()
