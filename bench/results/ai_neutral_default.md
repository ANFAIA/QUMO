# QUMO AI transpiler — hardware-neutral benchmark

Records: 5632 · groups: 1408 · levels: default, max · qumo-ai statuses: ok 1408

qumo-ai = `transpile_ai` (default -> AI level 2, max -> AI level 3, a portfolio with O3). Ratios are geometric means of (x+1)/(y+1) over (circuit, target, level) groups where both results are `ok` (statevector-verified up to 14 qubits); < 1 means qumo-ai is better. W/T/L = qumo-ai has fewer / equal / more 2q gates.


## Level `default`

| family | vs | pairs | 2q ratio | 2q-depth ratio | W/T/L | median ms (qumo-ai) | median ms (ref) | time ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| abstract | qumo | 192 | 0.991 | 0.992 | 31/157/4 | 6.68 | 1.61 | 4.11 |
| abstract | qiskit | 192 | 0.870 | 0.868 | 108/79/5 | 6.68 | 5.89 | 0.75 |
| abstract | tket | 192 | 0.806 | 0.815 | 119/58/15 | 6.68 | 207.88 | 0.02 |
| neutral atom | qumo | 128 | 0.998 | 0.996 | 6/122/0 | 0.44 | 0.30 | 1.88 |
| neutral atom | qiskit | 128 | 0.941 | 0.951 | 19/109/0 | 0.44 | 5.02 | 0.13 |
| neutral atom | tket | 128 | 0.997 | 1.008 | 18/83/27 | 0.44 | 191.64 | 0.00 |
| superconducting | qumo | 256 | 0.991 | 0.994 | 44/205/7 | 9.78 | 1.74 | 4.71 |
| superconducting | qiskit | 256 | 0.851 | 0.847 | 156/93/7 | 9.78 | 6.37 | 0.96 |
| superconducting | tket | 255 | 0.798 | 0.797 | 170/63/22 | 9.75 | 258.89 | 0.02 |
| trapped ion | qumo | 128 | 0.998 | 0.997 | 7/119/2 | 0.57 | 0.42 | 1.86 |
| trapped ion | qiskit | 128 | 0.918 | 0.917 | 56/71/1 | 0.57 | 5.02 | 0.17 |
| trapped ion | tket | 128 | 0.923 | 0.934 | 54/60/14 | 0.57 | 188.81 | 0.00 |
| **overall** | qumo | 704 | 0.994 | 0.994 | 88/603/13 | 2.04 | 0.89 | 3.24 |
| **overall** | qiskit | 704 | 0.884 | 0.883 | 339/352/13 | 2.04 | 5.81 | 0.45 |
| **overall** | tket | 703 | 0.856 | 0.861 | 361/264/78 | 1.96 | 207.88 | 0.01 |

### Techniques (level `default`)

| technique | jobs run | blocks | proposed | verified | replaced | replaced / blocks | saved 2q cost | rejected (not equal) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| linear_synth | 704 | 63524 | 23599 | 6967 | 7505 | 0.118 | 24771 | 0 |
| perm_synth | 704 | 428 | 60 | 13 | 30 | 0.070 | 6 | 0 |
| learned routing (kept vs SABRE) | 312 | | | | 45 | 0.144 | | |

Portfolio winner is the AI pipeline in 704 of 704 qumo-ai jobs.

