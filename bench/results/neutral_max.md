# Hardware-neutral compiler benchmark — results

Records: 2816 · SDKs: cirq, qiskit, qumo, tket · groups: 704 · level: max

Ratios are geometric means vs. the best SDK in each (circuit, target, level) group: **2Q** and **time** lower is better (1.00 = best), **ESP** higher is better (1.00 = best).

| target | SDK | ok | 2Q ratio | ESP ratio | time ratio | median wall (ms) | wins | ties | unsupported | incorrect | invalid | error | timeout |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ion_all_to_all | cirq | 0 | nan | nan | nan | nan | 0 | 0 | 64 | 0 | 0 | 0 | 0 |
| ion_all_to_all | qiskit | 64 | 1.131 | 0.817 | 5.5 | 5.4 | 0 | 33 | 0 | 0 | 0 | 0 | 0 |
| ion_all_to_all | qumo | 64 | 1.051 | 0.998 | 1.0 | 0.9 | 21 | 35 | 0 | 0 | 0 | 0 | 0 |
| ion_all_to_all | tket | 63 | 1.067 | 0.850 | 317.5 | 298.4 | 8 | 35 | 0 | 0 | 0 | 0 | 1 |
| ion_zone_ring | cirq | 0 | nan | nan | nan | nan | 0 | 0 | 64 | 0 | 0 | 0 | 0 |
| ion_zone_ring | qiskit | 64 | 1.151 | 0.750 | 4.8 | 5.2 | 1 | 24 | 0 | 0 | 0 | 0 | 0 |
| ion_zone_ring | qumo | 64 | 1.056 | 0.998 | 1.0 | 1.3 | 29 | 27 | 0 | 0 | 0 | 0 | 0 |
| ion_zone_ring | tket | 63 | 1.181 | 0.803 | 271.1 | 293.3 | 7 | 26 | 0 | 0 | 0 | 0 | 1 |
| linear_cx | cirq | 0 | nan | nan | nan | nan | 0 | 0 | 64 | 0 | 0 | 0 | 0 |
| linear_cx | qiskit | 64 | 1.120 | 0.314 | 2.6 | 5.7 | 5 | 22 | 0 | 0 | 0 | 0 | 0 |
| linear_cx | qumo | 64 | 1.043 | 0.901 | 1.2 | 3.2 | 29 | 25 | 0 | 0 | 0 | 0 | 0 |
| linear_cx | tket | 63 | 1.153 | 0.228 | 164.9 | 390.0 | 5 | 20 | 0 | 0 | 0 | 0 | 1 |
| neutral_atom | cirq | 64 | 1.336 | 0.467 | 66.2 | 111.5 | 0 | 22 | 0 | 0 | 0 | 0 | 0 |
| neutral_atom | qiskit | 64 | 1.125 | 0.586 | 6.6 | 6.5 | 0 | 42 | 0 | 0 | 0 | 0 | 0 |
| neutral_atom | qumo | 64 | 1.058 | 0.992 | 1.0 | 1.0 | 11 | 42 | 0 | 0 | 0 | 0 | 0 |
| neutral_atom | tket | 63 | 1.069 | 0.712 | 292.4 | 370.9 | 11 | 39 | 0 | 0 | 0 | 0 | 1 |
| neutral_atom_reconfigurable | cirq | 64 | 1.283 | 0.523 | 87.4 | 108.6 | 0 | 25 | 0 | 0 | 0 | 0 | 0 |
| neutral_atom_reconfigurable | qiskit | 64 | 1.122 | 0.648 | 5.9 | 5.1 | 0 | 44 | 0 | 0 | 0 | 0 | 0 |
| neutral_atom_reconfigurable | qumo | 64 | 1.059 | 0.991 | 1.0 | 0.8 | 7 | 44 | 0 | 0 | 0 | 0 | 0 |
| neutral_atom_reconfigurable | tket | 63 | 1.053 | 0.715 | 359.0 | 350.2 | 13 | 44 | 0 | 0 | 0 | 0 | 1 |
| sc_diagonal_grid | cirq | 59 | 1.358 | 0.323 | 36.8 | 78.7 | 1 | 7 | 0 | 0 | 0 | 5 | 0 |
| sc_diagonal_grid | qiskit | 64 | 1.138 | 0.492 | 2.7 | 6.3 | 2 | 18 | 0 | 0 | 0 | 0 | 0 |
| sc_diagonal_grid | qumo | 64 | 1.036 | 0.989 | 1.1 | 3.2 | 33 | 19 | 0 | 0 | 0 | 0 | 0 |
| sc_diagonal_grid | tket | 63 | 1.113 | 0.548 | 170.6 | 391.7 | 9 | 18 | 0 | 0 | 0 | 0 | 1 |
| sc_grid | cirq | 64 | 1.283 | 0.339 | 39.0 | 123.9 | 0 | 18 | 0 | 0 | 0 | 0 | 0 |
| sc_grid | qiskit | 64 | 1.150 | 0.478 | 3.0 | 6.2 | 1 | 21 | 0 | 0 | 0 | 0 | 0 |
| sc_grid | qumo | 64 | 1.041 | 0.980 | 1.1 | 3.1 | 34 | 23 | 0 | 0 | 0 | 0 | 0 |
| sc_grid | tket | 63 | 1.154 | 0.546 | 195.1 | 543.7 | 6 | 18 | 0 | 0 | 0 | 0 | 1 |
| sc_heavy_hex | cirq | 64 | 1.302 | 0.365 | 34.5 | 119.5 | 0 | 19 | 0 | 0 | 0 | 0 | 0 |
| sc_heavy_hex | qiskit | 64 | 1.128 | 0.502 | 2.7 | 6.9 | 4 | 21 | 0 | 0 | 0 | 0 | 0 |
| sc_heavy_hex | qumo | 64 | 1.043 | 0.991 | 1.1 | 3.7 | 23 | 24 | 0 | 0 | 0 | 0 | 0 |
| sc_heavy_hex | tket | 63 | 1.132 | 0.530 | 168.8 | 724.0 | 13 | 16 | 0 | 0 | 0 | 0 | 1 |
| sc_octagonal | cirq | 64 | 1.293 | 0.324 | 38.9 | 120.0 | 2 | 17 | 0 | 0 | 0 | 0 | 0 |
| sc_octagonal | qiskit | 64 | 1.132 | 0.468 | 2.9 | 6.1 | 3 | 21 | 0 | 0 | 0 | 0 | 0 |
| sc_octagonal | qumo | 64 | 1.044 | 0.979 | 1.1 | 3.5 | 26 | 27 | 0 | 0 | 0 | 0 | 0 |
| sc_octagonal | tket | 63 | 1.119 | 0.535 | 184.0 | 419.2 | 6 | 22 | 0 | 0 | 0 | 0 | 1 |
| torus | cirq | 0 | nan | nan | nan | nan | 0 | 0 | 64 | 0 | 0 | 0 | 0 |
| torus | qiskit | 64 | 1.143 | 0.283 | 3.8 | 6.0 | 2 | 25 | 0 | 0 | 0 | 0 | 0 |
| torus | qumo | 64 | 1.038 | 0.889 | 1.1 | 2.0 | 28 | 28 | 0 | 0 | 0 | 0 | 0 |
| torus | tket | 63 | 1.138 | 0.222 | 261.0 | 573.0 | 6 | 26 | 0 | 0 | 0 | 0 | 1 |
| tree | cirq | 0 | nan | nan | nan | nan | 0 | 0 | 64 | 0 | 0 | 0 | 0 |
| tree | qiskit | 64 | 1.126 | 0.318 | 2.7 | 5.9 | 6 | 18 | 0 | 0 | 0 | 0 | 0 |
| tree | qumo | 64 | 1.051 | 0.891 | 1.1 | 3.4 | 28 | 20 | 0 | 0 | 0 | 0 | 0 |
| tree | tket | 63 | 1.150 | 0.243 | 175.1 | 397.7 | 10 | 17 | 0 | 0 | 0 | 0 | 1 |

## Overall (all targets)

| SDK | 2Q ratio | ESP ratio | time ratio | wins | ties |
|---|---:|---:|---:|---:|---:|
| cirq | 1.308 | 0.384 | 47.4 | 3 | 108 |
| qiskit | 1.133 | 0.487 | 3.7 | 24 | 289 |
| qumo | 1.047 | 0.963 | 1.1 | 269 | 314 |
| tket | 1.120 | 0.487 | 223.8 | 94 | 281 |
