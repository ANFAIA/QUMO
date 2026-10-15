# AI transpiler stress set (Clifford / linear / permutation / Pauli-network)

26 generated circuits (up to 60 qubits) x targets sc_heavy_hex, sc_grid, king_grid, linear_cx x levels default,max. Every row verified exactly (Pauli-sum Heisenberg propagation / tableau through the reported layouts; `verified_by` field). Ratios: geometric mean of (x+1)/(y+1) over rows where both are ok; qumo-ai < 1 is better; W/L = fewer / more 2q gates. Raw rows: `ai_stress.jsonl` (sdks qumo, qumo-ai, qiskit, tket).

| family | level | vs | rows | 2q ratio | 2q-depth ratio | W/L | median time ratio |
|---|---|---|---:|---:|---:|---:|---:|
| clifford | default | qumo | 16 | 0.934 | 0.975 | 15/0 | 17.2 |
| clifford | default | qiskit | 16 | 0.892 | 0.906 | 16/0 | 23.7 |
| clifford | default | tket | 16 | 0.762 | 0.772 | 16/0 | 0.2 |
| clifford | max | qumo | 16 | 0.869 | 0.834 | 14/0 | 21.6 |
| clifford | max | qiskit | 16 | 0.819 | 0.768 | 15/1 | 77.1 |
| clifford | max | tket | 16 | 0.716 | 0.688 | 16/0 | 0.4 |
| linear | default | qumo | 16 | 0.788 | 0.817 | 16/0 | 21.5 |
| linear | default | qiskit | 16 | 0.733 | 0.744 | 16/0 | 17.6 |
| linear | default | tket | 15 | 0.683 | 0.661 | 14/1 | 0.2 |
| linear | max | qumo | 16 | 0.663 | 0.623 | 15/0 | 25.4 |
| linear | max | qiskit | 16 | 0.607 | 0.562 | 16/0 | 57.4 |
| linear | max | tket | 12 | 0.551 | 0.493 | 12/0 | 0.4 |
| perm | default | qumo | 16 | 1.000 | 1.000 | 0/0 | 0.9 |
| perm | default | qiskit | 16 | 1.000 | 1.000 | 0/0 | 0.0 |
| perm | default | tket | 16 | 1.000 | 1.000 | 0/0 | 0.0 |
| perm | max | qumo | 16 | 1.000 | 1.000 | 0/0 | 1.1 |
| perm | max | qiskit | 16 | 1.000 | 1.000 | 0/0 | 0.0 |
| perm | max | tket | 16 | 1.000 | 1.000 | 0/0 | 0.0 |
| swap | default | qumo | 16 | 1.000 | 1.000 | 0/0 | 1.0 |
| swap | default | qiskit | 16 | 1.000 | 1.000 | 0/0 | 0.0 |
| swap | default | tket | 16 | 1.000 | 1.000 | 0/0 | 0.0 |
| swap | max | qumo | 16 | 1.000 | 1.000 | 0/0 | 1.1 |
| swap | max | qiskit | 16 | 1.000 | 1.000 | 0/0 | 0.0 |
| swap | max | tket | 16 | 1.000 | 1.000 | 0/0 | 0.0 |
| trotter | default | qumo | 12 | 0.918 | 0.947 | 10/1 | 10.6 |
| trotter | default | qiskit | 12 | 0.843 | 0.859 | 12/0 | 5.7 |
| trotter | default | tket | 12 | 0.720 | 0.758 | 12/0 | 0.0 |
| trotter | max | qumo | 12 | 0.879 | 0.853 | 12/0 | 25.7 |
| trotter | max | qiskit | 12 | 0.789 | 0.762 | 12/0 | 26.0 |
| trotter | max | tket | 12 | 0.702 | 0.717 | 12/0 | 0.1 |
| ucc | default | qumo | 12 | 0.914 | 0.915 | 12/0 | 8.3 |
| ucc | default | qiskit | 12 | 0.735 | 0.715 | 12/0 | 5.9 |
| ucc | default | tket | 12 | 0.614 | 0.611 | 12/0 | 0.0 |
| ucc | max | qumo | 12 | 0.888 | 0.746 | 11/0 | 14.6 |
| ucc | max | qiskit | 12 | 0.710 | 0.589 | 12/0 | 25.8 |
| ucc | max | tket | 12 | 0.619 | 0.540 | 12/0 | 0.1 |
