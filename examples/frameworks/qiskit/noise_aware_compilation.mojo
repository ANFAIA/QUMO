"""Calibration data reaches the compiler: noise-aware layout and angle bounds.

Reproduces: Qiskit - VF2PostLayout, Target angle bounds and WrapAngles
Source: https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.transpiler.passes.VF2PostLayout (Qiskit docs, Apache-2.0)

A device whose couplers differ by a factor of ten: `post_layout` renames physical
qubits to the embedding with the best estimated fidelity, under the same gates. On
uniform targets nothing changes. `rzz[0:pi/2]` in the basis keeps angles in the
range a device accepts (IBM's fractional RZZ) through Pauli corrections.

Checks:
- a 3-qubit chain lands on the two good couplers;
- same two-qubit count with and without calibration;
- estimated success rises by the expected factor;
- every rzz angle ends up inside [0, pi/2] and the state does not change;
- emulating the compiled circuit tracks the estimate.

Run: .venv/bin/mojo run -I . examples/frameworks/qiskit/noise_aware_compilation.mojo
example-meta: {"framework": "qiskit", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from std.math import pi
from qumo.ir.circuit import Circuit
from qumo.ir.gates import Op
from qumo.compiler.target import Target, parse_target
from qumo.compiler.pass_manager import transpile, preset_pipeline
from qumo.exec.emulate import emulate
from qumo.sim.runner import evolve
from qumo.testing.checks import CheckLog


def main() raises:
    var log = CheckLog("noise_aware_compilation")
    var uniform = parse_target("superconducting_grid:2x3")
    var cal = uniform.copy()
    for i in range(cal.num_edges()):
        cal.error_2q[i] = 0.05
    cal.set_edge_error(1, 2, 0.003)
    cal.set_edge_error(2, 5, 0.003)
    var c = Circuit(3)
    c.h(0)
    c.cx(0, 1)
    c.cx(1, 2)
    c.measure_all()
    print(preset_pipeline(2, cal))
    var a = transpile(c, uniform, 2, 7)
    var b = transpile(c, cal, 2, 7)
    _ = log.close("same two-qubit count", Float64(b.circuit.num_2q_gates()), Float64(a.circuit.num_2q_gates()), 0.0)
    var before = cal.estimate_fidelity(a.circuit)
    var after = cal.estimate_fidelity(b.circuit)
    _ = log.check("post_layout chose the good couplers", after > before * 1.05, String(before) + " -> " + String(after))
    var r = emulate(b.circuit, cal, 20000, 3)
    var keep = r.probability("000") + r.probability("111")
    _ = log.close("emulated GHZ weight tracks the estimate", keep, after, 0.05)

    var t = parse_target("line:3/rz,sx,x,cz,rzz[0:pi/2]")
    var angles = Circuit(3)
    angles.h(0)
    angles.rzz(2.5, 0, 1)
    angles.rzz(-0.7, 1, 2)
    var w = transpile(angles, t, 2, 7)
    var bad = 0
    for g in w.circuit.ops:
        if g.kind == Op.RZZ and (g.p0.value() < -1e-12 or g.p0.value() > pi / 2 + 1e-12):
            bad += 1
    _ = log.close("angles outside the bounds", Float64(bad), 0.0, 0.0)
    _ = log.close("the state is unchanged", evolve(angles).fidelity(evolve(w.circuit)), 1.0, 1e-9)
    log.finish()
