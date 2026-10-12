"""Baseline: same layered circuit on Qiskit Aer statevector (CPU, fp64)."""
import sys, time
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def layered(n, depth):
    c = QuantumCircuit(n)
    for d in range(depth):
        for q in range(n):
            c.ry(0.1 + 0.01 * (q + d), q)
            c.rz(0.2 + 0.03 * (q * d), q)
        for q in range(d % 2, n - 1, 2):
            c.cx(q, q + 1)
    c.save_statevector()
    return c


nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 26
sim = AerSimulator(method="statevector", precision="double")
print("qubits,seconds(aer)")
for n in range(16, nmax + 1, 2):
    c = layered(n, 10)
    t0 = time.perf_counter()
    sim.run(c).result()
    print(n, ",", time.perf_counter() - t0)
