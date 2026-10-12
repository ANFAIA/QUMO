### tket vs qiskit (both levels; ratio tket/qiskit, >1 means qiskit better)

| target | n | 2Q ratio | time ratio | tket fewer 2Q | tie | qiskit fewer 2Q |
|---|---:|---:|---:|---:|---:|---:|
| sc_heavy_hex | 126 | 1.043 | 47.5 | 28 | 40 | 58 |
| sc_grid | 127 | 1.026 | 51.7 | 21 | 44 | 62 |
| sc_diagonal_grid | 127 | 1.012 | 48.7 | 30 | 41 | 56 |
| sc_octagonal | 127 | 1.036 | 48.4 | 25 | 38 | 64 |
| ion_all_to_all | 127 | 0.947 | 43.5 | 27 | 98 | 2 |
| ion_zone_ring | 127 | 1.033 | 45.0 | 27 | 52 | 48 |
| neutral_atom | 127 | 0.951 | 35.9 | 31 | 83 | 13 |
| neutral_atom_reconfigurable | 127 | 0.937 | 45.2 | 35 | 92 | 0 |
| linear_cx | 127 | 1.078 | 44.9 | 23 | 37 | 67 |
| tree | 127 | 1.061 | 49.0 | 21 | 35 | 71 |
| torus | 127 | 1.001 | 53.5 | 22 | 63 | 42 |
| **all** | 1396 | **1.010** | 46.4 | 290 | 623 | 483 |

### cirq vs qiskit (both levels; ratio cirq/qiskit, >1 means qiskit better)

| target | n | 2Q ratio | time ratio | cirq fewer 2Q | tie | qiskit fewer 2Q |
|---|---:|---:|---:|---:|---:|---:|
| sc_heavy_hex | 128 | 1.153 | 12.3 | 9 | 48 | 71 |
| sc_grid | 128 | 1.117 | 12.5 | 14 | 46 | 68 |
| sc_diagonal_grid | 118 | 1.203 | 12.3 | 2 | 22 | 94 |
| sc_octagonal | 128 | 1.141 | 12.7 | 6 | 50 | 72 |
| ion_all_to_all | 0 | – | – | – | – | – |
| ion_zone_ring | 0 | – | – | – | – | – |
| neutral_atom | 128 | 1.189 | 10.7 | 0 | 58 | 70 |
| neutral_atom_reconfigurable | 128 | 1.144 | 14.2 | 0 | 66 | 62 |
| linear_cx | 0 | – | – | – | – | – |
| tree | 0 | – | – | – | – | – |
| torus | 0 | – | – | – | – | – |
| **all** | 758 | **1.157** | 12.4 | 31 | 290 | 437 |

