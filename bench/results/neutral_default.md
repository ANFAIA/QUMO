# Hardware-neutral compiler benchmark — results

Records: 2816 · SDKs: cirq, qiskit, qumo, tket · groups: 704 · level: default

Ratios are geometric means vs. the best SDK in each (circuit, target, level) group: **2Q** and **time** lower is better (1.00 = best), **ESP** higher is better (1.00 = best).

| target | SDK | ok | 2Q ratio | ESP ratio | time ratio | median wall (ms) | wins | ties | unsupported | incorrect | invalid | error | timeout |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ion_all_to_all | cirq | 0 | nan | nan | nan | nan | 0 | 0 | 64 | 0 | 0 | 0 | 0 |
| ion_all_to_all | qiskit | 64 | 1.128 | 0.816 | 11.4 | 5.0 | 0 | 33 | 0 | 0 | 0 | 0 | 0 |
| ion_all_to_all | qumo | 64 | 1.048 | 0.998 | 1.0 | 0.4 | 24 | 33 | 0 | 0 | 0 | 0 | 0 |
| ion_all_to_all | tket | 64 | 1.073 | 0.841 | 384.1 | 188.8 | 7 | 33 | 0 | 0 | 0 | 0 | 0 |
| ion_zone_ring | cirq | 0 | nan | nan | nan | nan | 0 | 0 | 64 | 0 | 0 | 0 | 0 |
| ion_zone_ring | qiskit | 64 | 1.147 | 0.755 | 8.9 | 5.0 | 1 | 24 | 0 | 0 | 0 | 0 | 0 |
| ion_zone_ring | qumo | 64 | 1.053 | 0.997 | 1.0 | 0.6 | 29 | 27 | 0 | 0 | 0 | 0 | 0 |
| ion_zone_ring | tket | 64 | 1.192 | 0.739 | 328.8 | 192.2 | 7 | 26 | 0 | 0 | 0 | 0 | 0 |
| linear_cx | cirq | 0 | nan | nan | nan | nan | 0 | 0 | 64 | 0 | 0 | 0 | 0 |
| linear_cx | qiskit | 64 | 1.115 | 0.360 | 5.4 | 5.8 | 6 | 22 | 0 | 0 | 0 | 0 | 0 |
| linear_cx | qumo | 64 | 1.042 | 0.935 | 1.0 | 1.3 | 31 | 22 | 0 | 0 | 0 | 0 | 0 |
| linear_cx | tket | 64 | 1.260 | 0.128 | 175.8 | 192.7 | 5 | 15 | 0 | 0 | 0 | 0 | 0 |
| neutral_atom | cirq | 64 | 1.335 | 0.427 | 122.6 | 97.7 | 0 | 22 | 0 | 0 | 0 | 0 | 0 |
| neutral_atom | qiskit | 64 | 1.122 | 0.575 | 10.8 | 6.2 | 0 | 43 | 0 | 0 | 0 | 0 | 0 |
| neutral_atom | qumo | 64 | 1.059 | 0.970 | 1.0 | 0.4 | 8 | 43 | 0 | 0 | 0 | 0 | 0 |
| neutral_atom | tket | 64 | 1.067 | 0.720 | 324.6 | 200.0 | 13 | 39 | 0 | 0 | 0 | 0 | 0 |
| neutral_atom_reconfigurable | cirq | 64 | 1.283 | 0.490 | 158.3 | 95.9 | 0 | 25 | 0 | 0 | 0 | 0 | 0 |
| neutral_atom_reconfigurable | qiskit | 64 | 1.122 | 0.628 | 11.7 | 4.7 | 0 | 44 | 0 | 0 | 0 | 0 | 0 |
| neutral_atom_reconfigurable | qumo | 64 | 1.060 | 0.959 | 1.0 | 0.4 | 5 | 44 | 0 | 0 | 0 | 0 | 0 |
| neutral_atom_reconfigurable | tket | 64 | 1.051 | 0.721 | 403.7 | 184.7 | 15 | 44 | 0 | 0 | 0 | 0 | 0 |
| sc_diagonal_grid | cirq | 59 | 1.344 | 0.312 | 68.3 | 69.2 | 1 | 7 | 0 | 0 | 0 | 5 | 0 |
| sc_diagonal_grid | qiskit | 64 | 1.126 | 0.493 | 5.5 | 6.1 | 4 | 17 | 0 | 0 | 0 | 0 | 0 |
| sc_diagonal_grid | qumo | 64 | 1.044 | 0.982 | 1.0 | 1.3 | 35 | 17 | 0 | 0 | 0 | 0 | 0 |
| sc_diagonal_grid | tket | 64 | 1.178 | 0.492 | 208.4 | 255.5 | 7 | 15 | 0 | 0 | 0 | 0 | 0 |
| sc_grid | cirq | 64 | 1.268 | 0.319 | 70.6 | 110.2 | 0 | 19 | 0 | 0 | 0 | 0 | 0 |
| sc_grid | qiskit | 64 | 1.134 | 0.499 | 5.9 | 6.1 | 4 | 23 | 0 | 0 | 0 | 0 | 0 |
| sc_grid | qumo | 64 | 1.045 | 0.982 | 1.0 | 1.3 | 28 | 24 | 0 | 0 | 0 | 0 | 0 |
| sc_grid | tket | 64 | 1.189 | 0.522 | 245.5 | 394.3 | 7 | 18 | 0 | 0 | 0 | 0 | 0 |
| sc_heavy_hex | cirq | 64 | 1.288 | 0.344 | 65.9 | 107.9 | 0 | 19 | 0 | 0 | 0 | 0 | 0 |
| sc_heavy_hex | qiskit | 64 | 1.117 | 0.527 | 5.5 | 7.1 | 7 | 21 | 0 | 0 | 0 | 0 | 0 |
| sc_heavy_hex | qumo | 64 | 1.038 | 0.987 | 1.0 | 1.5 | 28 | 24 | 0 | 0 | 0 | 0 | 0 |
| sc_heavy_hex | tket | 63 | 1.216 | 0.467 | 211.2 | 436.2 | 5 | 16 | 0 | 0 | 0 | 0 | 1 |
| sc_octagonal | cirq | 64 | 1.280 | 0.316 | 72.9 | 106.7 | 2 | 17 | 0 | 0 | 0 | 0 | 0 |
| sc_octagonal | qiskit | 64 | 1.123 | 0.514 | 6.1 | 6.6 | 3 | 25 | 0 | 0 | 0 | 0 | 0 |
| sc_octagonal | qumo | 64 | 1.040 | 0.986 | 1.0 | 1.3 | 28 | 26 | 0 | 0 | 0 | 0 | 0 |
| sc_octagonal | tket | 64 | 1.220 | 0.406 | 228.0 | 218.1 | 5 | 16 | 0 | 0 | 0 | 0 | 0 |
| torus | cirq | 0 | nan | nan | nan | nan | 0 | 0 | 64 | 0 | 0 | 0 | 0 |
| torus | qiskit | 64 | 1.116 | 0.365 | 7.6 | 5.9 | 4 | 29 | 0 | 0 | 0 | 0 | 0 |
| torus | qumo | 64 | 1.044 | 0.802 | 1.0 | 1.0 | 24 | 29 | 0 | 0 | 0 | 0 | 0 |
| torus | tket | 64 | 1.124 | 0.197 | 323.0 | 458.9 | 7 | 28 | 0 | 0 | 0 | 0 | 0 |
| tree | cirq | 0 | nan | nan | nan | nan | 0 | 0 | 64 | 0 | 0 | 0 | 0 |
| tree | qiskit | 64 | 1.107 | 0.364 | 5.6 | 5.9 | 9 | 21 | 0 | 0 | 0 | 0 | 0 |
| tree | qumo | 64 | 1.047 | 0.920 | 1.0 | 1.3 | 28 | 21 | 0 | 0 | 0 | 0 | 0 |
| tree | tket | 64 | 1.220 | 0.152 | 205.9 | 200.0 | 6 | 15 | 0 | 0 | 0 | 0 | 0 |

## Overall (all targets)

| SDK | 2Q ratio | ESP ratio | time ratio | wins | ties |
|---|---:|---:|---:|---:|---:|
| cirq | 1.299 | 0.363 | 87.8 | 3 | 109 |
| qiskit | 1.123 | 0.517 | 7.3 | 38 | 302 |
| qumo | 1.047 | 0.955 | 1.0 | 268 | 310 |
| tket | 1.161 | 0.413 | 266.3 | 84 | 265 |
