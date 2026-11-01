"""`emulate`: run a compiled circuit on a device's noise, no device needed.

The neutral counterpart of Braket's local emulators: conformance is checked as
the hardware would (basis, coupling map, angle bounds), `NoiseModel.from_target`
turns the target's calibration into channels, and the cheapest exact sampler
runs. Unused physical qubits are dropped first, so a few-qubit circuit compiled
for a 127-qubit device simulates in 2^k, not 2^127.

Nothing here touches a real device.
"""

from qumo.ir.gates import Op, op_arity, op_is_unitary, op_is_noise
from qumo.ir.circuit import Circuit
from qumo.compiler.target import Target
from qumo.compiler.pass_manager import conformance_error
from qumo.qec.characterization import NoiseModel, apply_noise_model
from qumo.sim.runner import SampleResult, sample
from qumo.sim.ptsbe import sample_ptsbe
from qumo.exec.dispatch import choose_method_cpu, sample_frame
from qumo.sim.stabilizer import sample_stabilizer


def active_qubits(c: Circuit) -> List[Int]:
    """Physical qubits `c` uses, ascending."""
    var used = List[Bool](length=c.num_qubits, fill=False)
    for g in c.ops:
        if op_is_unitary(g.kind) or g.kind == Op.MEASURE or g.kind == Op.RESET or op_is_noise(g.kind):
            for j in range(op_arity(g.kind)):
                used[g.qubit(j)] = True
    var out = List[Int]()
    for q in range(c.num_qubits):
        if used[q]:
            out.append(q)
    return out^


def compact(c: Circuit, qubits: List[Int]) -> Circuit:
    """`c` on `qubits` renumbered 0, 1, ... (classical bits unchanged)."""
    var position = List[Int](length=c.num_qubits, fill=-1)
    for i in range(len(qubits)):
        position[qubits[i]] = i
    var out = Circuit(len(qubits), c.num_clbits, c.name)
    out.compose_mapped(c, position)
    return out^


def noisy_circuit(c: Circuit, target: Target) raises -> Circuit:
    """Compacted `c` with the target's noise channels inserted."""
    var err = conformance_error(c, target)
    if err.byte_length() > 0:
        raise Error("QM-E615: circuit does not conform to target '" + target.name + "': " + err + ". Hint: transpile it for the target first.")
    var qubits = active_qubits(c)
    var model = NoiseModel.from_target(target).restricted(qubits)
    return apply_noise_model(compact(c, qubits), model)


def emulate(c: Circuit, target: Target, shots: Int, seed: UInt64 = 1234) raises -> SampleResult:
    """Sample `c`, already compiled for `target`, under the target's noise.

    Pauli-frame or tableau sampling for Clifford circuits, pattern-grouped state
    vector for static ones, per-shot trajectories otherwise. `method` in the
    result says which ran."""
    var noisy = noisy_circuit(c, target)
    var m = choose_method_cpu(noisy, shots)
    if m == "frame":
        return sample_frame(noisy, shots, seed)
    if m == "stabilizer":
        return sample_stabilizer(noisy, shots, seed)
    if noisy.is_dynamic():
        return sample(noisy, shots, seed)
    return sample_ptsbe(noisy, shots, seed)
