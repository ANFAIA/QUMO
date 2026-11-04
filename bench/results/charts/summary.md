| level | SDK | records | ok | unsupported | failed | median compile s | p90 compile s | idle MB | median peak MB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| default | QUMO | 704 | 704 | 0 | 0 | 0.0006 | 0.0050 | 97 | 139 |
| default | Qiskit | 704 | 704 | 0 | 0 | 0.0037 | 0.0076 | 189 | 226 |
| default | TKET | 704 | 704 | 0 | 0 | 0.1351 | 1.3676 | 284 | 286 |
| default | Cirq | 704 | 379 | 320 | 5 | 0.0596 | 0.3817 | 231 | 228 |
| max | QUMO | 704 | 704 | 0 | 0 | 0.0012 | 0.0128 | 97 | 139 |
| max | Qiskit | 704 | 704 | 0 | 0 | 0.0038 | 0.0092 | 189 | 226 |
| max | TKET | 704 | 704 | 0 | 0 | 0.2580 | 2.2581 | 284 | 286 |
| max | Cirq | 704 | 379 | 320 | 5 | 0.0713 | 0.4505 | 231 | 228 |

## Memory

Records: 5632 of 5632 expected. Run complete.

Bare python: 19 MB. Python with `from qiskit import qasm2`: 71 MB. Every worker imports it, so this part of each idle footprint is shared, not SDK-specific.

| SDK | records | idle MB | idle minus shared MB | median peak MB | growth median / p90 / max MB | base RSS, first 10% / last 10% MB |
|---|---:|---:|---:|---:|---:|---:|
| QUMO | 1408 | 97 | 26 | 139 | 0.0 / 0.0 / 96 | 100 / 134 |
| Qiskit | 1408 | 189 | 118 | 226 | 0.0 / 0.1 / 104 | 196 / 187 |
| TKET | 1408 | 284 | 213 | 286 | 0.0 / 0.0 / 226 | 290 / 280 |
| Cirq | 758 | 231 | 161 | 228 | 0.0 / 0.1 / 162 | 239 / 221 |

Mean change of base RSS from one job to the next, by the qubits of the earlier job (MB):

- QUMO: after <= 8 qubits -0.03; after 9-14 qubits +0.20.
- Qiskit: after <= 8 qubits +0.05; after 9-14 qubits +0.15.
- TKET: after <= 8 qubits -0.02; after 9-14 qubits +0.26.
- Cirq: after <= 8 qubits -0.02; after 9-14 qubits +0.26.

Reading:
- Inside the timed compile call the resident set barely grows; medians are in the growth column.
- Between jobs the base RSS moves in steps. After 9-14 qubit jobs the mean step is +0.15 to +0.26 MB across SDKs, the jobs whose statevector verification runs between timed calls. The harness, not one compiler, is the likely source.
- Net change of base RSS from the first to the last 10% of jobs, MB: QUMO +34, Qiskit -9, TKET -10, Cirq -18. The change is not uniform across SDKs. Not isolated yet: that needs a fresh process per job or a run without verification.
- Idle footprints compare fairly only after subtracting the shared qiskit import.
