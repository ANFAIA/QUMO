"""`conformance_error`: does a circuit fit a target as the hardware would check it.

Basis, coupling edges, angle bounds and width, from the `Target` data alone. It was in
`qumo/compiler/pass_manager.mojo`, which still exports it; it is here so that the device emulator
(`qumo.noise.emulate`) can check conformance without importing the compiler.
"""

from qumo.ir.gates import op_arity, op_is_unitary
from qumo.ir.circuit import Circuit
from qumo.device.target import Target
from qumo.device.format import basis_str


def conformance_error(c: Circuit, target: Target) -> String:
    """First violation of basis / connectivity, or "" when the circuit conforms."""
    for i in range(len(c.ops)):
        var g = c.ops[i]
        if not op_is_unitary(g.kind):
            continue
        if not target.supports(g.kind):
            return "instruction #" + String(i) + " '" + String(g) + "' is not in the basis [" + basis_str(target.basis) + "]"
        var a = op_arity(g.kind)
        if a >= 2 and not target.full:
            if a > 2:
                return "instruction #" + String(i) + " '" + String(g) + "' acts on 3 qubits (no 3q coupling)"
            if not target.are_connected(g.q0, g.q1):
                return "instruction #" + String(i) + " '" + String(g) + "' is not on a coupling edge of " + target.name
        if target.angle_bound[g.kind] and not g.p0.is_symbolic():
            var v = g.p0.value()
            if v < target.angle_lo[g.kind] - 1e-9 or v > target.angle_hi[g.kind] + 1e-9:
                return "instruction #" + String(i) + " '" + String(g) + "' has its angle outside [" + String(target.angle_lo[g.kind]) + ", " + String(target.angle_hi[g.kind]) + "]"
        for s in range(a):
            if g.qubit(s) >= target.num_qubits:
                return "instruction #" + String(i) + " uses qubit " + String(g.qubit(s)) + " beyond the target size"
    return ""
