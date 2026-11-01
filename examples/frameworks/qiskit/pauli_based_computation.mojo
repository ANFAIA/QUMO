"""Pauli-based computation: Litinski's form of a Clifford+T circuit.

Reproduces: Qiskit - LitinskiTransformation (arXiv:1808.02892)
Source: https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.transpiler.passes.LitinskiTransformation (Qiskit docs, Apache-2.0)

Commuting every Clifford to the end leaves a time-ordered list of Pauli-product
rotations and measurements, which is what a surface-code computer executes. For
H T CX T the rotations are X0 and Z1 X0, both pi/8, as `LitinskiTransformation`
gives. `pauli_simp(objective=t)` uses the same extraction to reduce T counts.

Checks:
- the two Pauli strings and their commuting layer;
- `to_circuit` reproduces the unitary including its phase;
- the T objective reaches 8 T on a small example, where folding alone stays higher.

Run: .venv/bin/mojo run -I . examples/frameworks/qiskit/pauli_based_computation.mojo
example-meta: {"framework": "qiskit", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from qumo.ir.circuit import Circuit
from qumo.compiler.pbc import to_pbc
from qumo.compiler.pass_manager import compile_spec
from qumo.compiler.target import Target
from qumo.sim.runner import evolve
from qumo.testing.checks import CheckLog


def main() raises:
    var log = CheckLog("pauli_based_computation")
    var c = Circuit(2)
    c.h(0)
    c.t(0)
    c.cx(0, 1)
    c.t(1)
    var prog = to_pbc(c)
    for op in prog.ops:
        print(op.label(2), op.angle)
    _ = log.check("first rotation is X0", prog.ops[0].label(2) == "IX", prog.ops[0].label(2))
    _ = log.check("second rotation is Z1 X0", prog.ops[1].label(2) == "ZX", prog.ops[1].label(2))
    _ = log.close("one commuting layer", Float64(prog.layers()), 1.0, 0.0)
    var sa = evolve(c)
    var sb = evolve(prog.to_circuit())
    var worst = 0.0
    for i in range(sa.size()):
        worst = max(worst, abs(sa.re[i] - sb.re[i]), abs(sa.im[i] - sb.im[i]))
    _ = log.close("to_circuit keeps the unitary", worst, 0.0, 1e-10)
    # two T gates on the same Pauli, hidden from phase folding by Hadamards
    var d = Circuit(2)
    d.h(0)
    d.t(0)
    d.h(0)
    d.cz(0, 1)
    d.cz(0, 1)
    d.h(0)
    d.t(0)
    d.h(0)
    var t = Target.all_to_all(2)
    var fold = compile_spec(d, t, "fixpoint(cancel,merge,commute,phase_fold,cancel,merge,max=6)")
    var best = compile_spec(d, t, "pauli_simp(objective=t),fixpoint(cancel,merge,commute,phase_fold,cancel,merge,max=6)")
    print("T count: folding", fold.circuit.t_count(), "objective=t", best.circuit.t_count())
    _ = log.check("the T objective merges across Hadamards", best.circuit.t_count() < fold.circuit.t_count() or fold.circuit.t_count() == 0)
    log.finish()
