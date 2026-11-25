# Build a Bell circuit, optimize it for the CZ gate set, simulate it.
import cirq

a, b = cirq.LineQubit.range(2)
c = cirq.optimize_for_target_gateset(cirq.Circuit(cirq.H(a), cirq.CNOT(a, b)), gateset=cirq.CZTargetGateset())
print("compiled:", len(list(c.all_operations())), "ops,", sum(1 for o in c.all_operations() if len(o.qubits) == 2), "two-qubit")
c.append(cirq.measure(a, b, key="m"))
h = cirq.Simulator(seed=7).run(c, repetitions=1000).histogram(key="m")
print({format(k, "02b"): int(v) for k, v in sorted(h.items())})
