# Build a Bell circuit, optimize it for a 2-qubit line device (rz, sx, x, cx), simulate it.
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
t = transpile(qc, basis_gates=["rz", "sx", "x", "cx"], coupling_map=[[0, 1], [1, 0]], optimization_level=2, seed_transpiler=7)
print("compiled:", dict(t.count_ops()))
print({str(k): int(v) for k, v in sorted(Statevector(t).sample_counts(1000).items())})
