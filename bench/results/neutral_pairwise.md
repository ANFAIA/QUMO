Verified by state equivalence: 4970 of 4970 successful compilations (the rest exceed 14 logical / 20 active physical qubits).

### QUMO vs qiskit (pairwise, both verified-ok)

| modality | level | n | 2Q ratio QUMO/other | ESP ratio QUMO/other | time ratio QUMO/other | QUMO fewer 2Q | tie | other fewer 2Q |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| abstract | default | 192 | 0.939 | 44.54 | 0.167 | 85 | 86 | 21 |
| abstract | max | 192 | 0.924 | 54.22 | 0.384 | 95 | 82 | 15 |
| neutral atom | default | 128 | 0.944 | 1.61 | 0.089 | 18 | 110 | 0 |
| neutral atom | max | 128 | 0.942 | 1.61 | 0.164 | 20 | 108 | 0 |
| superconducting | default | 256 | 0.926 | 2.39 | 0.175 | 131 | 102 | 23 |
| superconducting | max | 256 | 0.916 | 2.56 | 0.392 | 145 | 101 | 10 |
| trapped ion | default | 128 | 0.924 | 1.27 | 0.100 | 56 | 71 | 1 |
| trapped ion | max | 128 | 0.924 | 1.27 | 0.200 | 56 | 71 | 1 |
| **all** | | 1408 | **0.928** | 4.58 | 0.201 | 606 | 731 | 71 |

### QUMO vs tket (pairwise, both verified-ok)

| modality | level | n | 2Q ratio QUMO/other | ESP ratio QUMO/other | time ratio QUMO/other | QUMO fewer 2Q | tie | other fewer 2Q |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| abstract | default | 192 | 0.870 | 168.59 | 0.004 | 115 | 58 | 19 |
| abstract | max | 189 | 0.911 | 84.22 | 0.006 | 105 | 63 | 21 |
| neutral atom | default | 128 | 1.000 | 1.34 | 0.003 | 17 | 83 | 28 |
| neutral atom | max | 126 | 0.998 | 1.39 | 0.003 | 19 | 83 | 24 |
| superconducting | default | 255 | 0.868 | 2.86 | 0.005 | 165 | 65 | 25 |
| superconducting | max | 252 | 0.922 | 2.28 | 0.006 | 142 | 76 | 34 |
| trapped ion | default | 128 | 0.929 | 1.27 | 0.003 | 54 | 60 | 14 |
| trapped ion | max | 126 | 0.940 | 1.21 | 0.003 | 49 | 62 | 15 |
| **all** | | 1396 | **0.919** | 5.71 | 0.004 | 666 | 550 | 180 |

### QUMO vs cirq (pairwise, both verified-ok)

| modality | level | n | 2Q ratio QUMO/other | ESP ratio QUMO/other | time ratio QUMO/other | QUMO fewer 2Q | tie | other fewer 2Q |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| neutral atom | default | 128 | 0.809 | 2.11 | 0.007 | 72 | 56 | 0 |
| neutral atom | max | 128 | 0.808 | 2.01 | 0.013 | 73 | 55 | 0 |
| superconducting | default | 251 | 0.804 | 3.81 | 0.014 | 175 | 72 | 4 |
| superconducting | max | 251 | 0.796 | 3.53 | 0.030 | 176 | 72 | 3 |
| **all** | | 758 | **0.803** | 3.02 | 0.016 | 496 | 255 | 7 |

### Status counts (64 circuits x 11 targets x 2 levels = 1408 per SDK)

| SDK | ok (verified or valid) | unsupported | timeout (>120 s) | error | incorrect | invalid |
|---|---:|---:|---:|---:|---:|---:|
| qumo | 1408 | 0 | 0 | 0 | 0 | 0 |
| qiskit | 1408 | 0 | 0 | 0 | 0 | 0 |
| tket | 1396 | 0 | 12 | 0 | 0 | 0 |
| cirq | 758 | 640 | 0 | 10 | 0 | 0 |

Versions: {'qiskit': '2.5.2', 'tket': '2.18.5', 'cirq': '1.7.0', 'qumo': '0.1.0'}
