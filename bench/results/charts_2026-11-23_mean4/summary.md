| level | SDK | records | ok | unsupported | failed | median compile s | p90 compile s | idle MB | median peak MB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| default | QUMO | 704 | 704 | 0 | 0 | 0.0005 | 0.0044 | 83 | 227 |
| default | Qiskit | 704 | 704 | 0 | 0 | 0.0027 | 0.0059 | 160 | 411 |
| default | TKET | 704 | 704 | 0 | 0 | 0.0947 | 1.0237 | 253 | 532 |
| default | Cirq | 704 | 379 | 320 | 5 | 0.0520 | 0.3407 | 222 | 346 |
| max | QUMO | 704 | 704 | 0 | 0 | 0.0011 | 0.0117 | 83 | 227 |
| max | Qiskit | 704 | 704 | 0 | 0 | 0.0029 | 0.0078 | 160 | 411 |
| max | TKET | 704 | 704 | 0 | 0 | 0.1877 | 1.8291 | 253 | 532 |
| max | Cirq | 704 | 379 | 320 | 5 | 0.0594 | 0.3991 | 222 | 346 |

## Memory

Records: 5632 of 5632 expected. Run complete.

Bare python: 18 MB. Python with `from qiskit import qasm2`: 70 MB. Every worker imports it, so this part of each idle footprint is shared, not SDK-specific.

| SDK | records | idle MB | idle minus shared MB | median peak MB | growth median / p90 / max MB | base RSS, first 10% / last 10% MB |
|---|---:|---:|---:|---:|---:|---:|
| QUMO | 1408 | 83 | 13 | 227 | 0.0 / 0.0 / 92 | 83 / 220 |
| Qiskit | 1408 | 160 | 90 | 411 | 0.0 / 0.1 / 105 | 160 / 403 |
| TKET | 1408 | 253 | 183 | 532 | 0.0 / 0.0 / 245 | 255 / 527 |
| Cirq | 758 | 222 | 151 | 346 | 0.0 / 0.1 / 141 | 226 / 341 |

Mean change of base RSS from one job to the next, by the qubits of the earlier job (MB):

- QUMO: after <= 8 qubits +0.02; after 9-14 qubits +0.36.
- Qiskit: after <= 8 qubits +0.01; after 9-14 qubits +0.63.
- TKET: after <= 8 qubits +0.02; after 9-14 qubits +0.66.
- Cirq: after <= 8 qubits +0.07; after 9-14 qubits +0.60.

Reading:
- Inside the timed compile call the resident set barely grows; medians are in the growth column.
- Between jobs the base RSS moves in steps. After 9-14 qubit jobs the mean step is +0.36 to +0.66 MB across SDKs, the jobs whose statevector verification runs between timed calls. The harness, not one compiler, is the likely source.
- Net change of base RSS from the first to the last 10% of jobs, MB: QUMO +138, Qiskit +243, TKET +273, Cirq +115. The change is not uniform across SDKs. Not isolated yet: that needs a fresh process per job or a run without verification.
- Idle footprints compare fairly only after subtracting the shared qiskit import.
