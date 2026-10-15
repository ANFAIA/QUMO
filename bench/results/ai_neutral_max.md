# QUMO AI transpiler — hardware-neutral benchmark

Records: 5632 · groups: 1408 · levels: default, max · qumo-ai statuses: ok 1408

qumo-ai = `transpile_ai` (default -> AI level 2, max -> AI level 3, a portfolio with O3). Ratios are geometric means of (x+1)/(y+1) over (circuit, target, level) groups where both results are `ok` (statevector-verified up to 14 qubits); < 1 means qumo-ai is better. W/T/L = qumo-ai has fewer / equal / more 2q gates.


## Level `max`

| family | vs | pairs | 2q ratio | 2q-depth ratio | W/T/L | median ms (qumo-ai) | median ms (ref) | time ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| abstract | qumo | 192 | 0.951 | 0.942 | 55/137/0 | 41.61 | 4.65 | 10.28 |
| abstract | qiskit | 192 | 0.822 | 0.813 | 122/68/2 | 41.61 | 5.83 | 4.79 |
| abstract | tket | 189 | 0.812 | 0.818 | 118/59/12 | 41.29 | 419.95 | 0.07 |
| neutral atom | qumo | 128 | 0.946 | 0.921 | 36/92/0 | 11.61 | 0.67 | 13.71 |
| neutral atom | qiskit | 128 | 0.889 | 0.876 | 42/86/0 | 11.61 | 5.35 | 1.81 |
| neutral atom | tket | 126 | 0.943 | 0.930 | 40/73/13 | 11.44 | 368.70 | 0.03 |
| superconducting | qumo | 256 | 0.955 | 0.955 | 74/182/0 | 54.75 | 4.73 | 11.11 |
| superconducting | qiskit | 256 | 0.809 | 0.803 | 168/87/1 | 54.75 | 6.35 | 5.52 |
| superconducting | tket | 252 | 0.817 | 0.817 | 163/70/19 | 47.43 | 430.10 | 0.09 |
| trapped ion | qumo | 128 | 0.963 | 0.955 | 23/105/0 | 8.65 | 0.91 | 8.77 |
| trapped ion | qiskit | 128 | 0.884 | 0.872 | 62/65/1 | 8.65 | 5.37 | 1.54 |
| trapped ion | tket | 126 | 0.901 | 0.898 | 56/60/10 | 8.61 | 298.37 | 0.03 |
| **overall** | qumo | 704 | 0.954 | 0.945 | 188/516/0 | 30.94 | 2.01 | 10.83 |
| **overall** | qiskit | 704 | 0.840 | 0.831 | 394/306/4 | 30.94 | 5.89 | 3.44 |
| **overall** | tket | 693 | 0.852 | 0.851 | 377/262/54 | 28.77 | 397.69 | 0.06 |

### Techniques (level `max`)

| technique | jobs run | blocks | proposed | verified | replaced | replaced / blocks | saved 2q cost | rejected (not equal) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| linear_synth | 484 | 135654 | 23751 | 5238 | 12970 | 0.096 | 9254 | 0 |
| perm_synth | 484 | 501 | 76 | 11 | 27 | 0.054 | 0 | 0 |
| clifford_synth | 484 | 660270 | 43519 | 9291 | 24786 | 0.038 | 14172 | 0 |
| pauli_network | 484 | 357402 | 39120 | 3366 | 11884 | 0.033 | 1331 | 0 |
| learned routing (kept vs SABRE) | 222 | | | | 45 | 0.203 | | |

Portfolio winner is the AI pipeline in 484 of 704 qumo-ai jobs (the rest: the O3/O2 heuristic member won).
