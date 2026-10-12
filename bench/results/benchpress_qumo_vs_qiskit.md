## Status counts

| SDK | passed | failed | skipped | xfailed |
|---|---:|---:|---:|---:|
| qumo | 1044 | 0 | 22 | 0 |
| qiskit | 1044 | 0 | 22 | 0 |

## qumo vs qiskit

Ratios are qumo/qiskit: **time < 1 ⇒ qumo faster**, **2Q < 1 ⇒ qumo fewer two-qubit gates**.

| category | n | time ratio | 2Q ratio | qumo fewer 2Q | tie | qiskit fewer 2Q |
|---|---:|---:|---:|---:|---:|---:|
| test_BV_100_transpile | 1 | 0.58 | 1.005 | 0 | 0 | 1 |
| test_BVlike_simplification_transpile | 1 | 0.16 | 1.000 | 0 | 1 | 0 |
| test_DTC100_set_build | 1 | 0.11 | 1.000 | 0 | 1 | 0 |
| test_DTC100_twirling | 1 | 0.54 | – | 0 | 0 | 0 |
| test_QAOA_100_transpile | 1 | 1.37 | 0.975 | 1 | 0 | 0 |
| test_QASMBench_large · all-to-all | 58 | 0.05 | 1.018 | 1 | 48 | 9 |
| test_QASMBench_large · heavy-hex | 58 | 0.37 | 0.637 | 37 | 13 | 8 |
| test_QASMBench_large · linear | 58 | 0.65 | 0.855 | 34 | 17 | 7 |
| test_QASMBench_large · square | 58 | 0.52 | 0.819 | 36 | 17 | 5 |
| test_QASMBench_medium · all-to-all | 23 | 0.11 | 1.000 | 3 | 18 | 2 |
| test_QASMBench_medium · heavy-hex | 23 | 0.54 | 0.896 | 13 | 5 | 5 |
| test_QASMBench_medium · linear | 23 | 0.72 | 0.973 | 12 | 6 | 5 |
| test_QASMBench_medium · square | 23 | 0.61 | 0.937 | 16 | 6 | 1 |
| test_QASMBench_small · all-to-all | 42 | 0.18 | 0.926 | 5 | 37 | 0 |
| test_QASMBench_small · heavy-hex | 42 | 0.28 | 0.920 | 16 | 25 | 1 |
| test_QASMBench_small · linear | 42 | 0.28 | 0.922 | 16 | 26 | 0 |
| test_QASMBench_small · square | 42 | 0.27 | 0.910 | 15 | 25 | 2 |
| test_QFT_100_transpile | 1 | 1.72 | 0.886 | 1 | 0 | 0 |
| test_QV100_basis_change | 1 | 0.31 | – | 0 | 0 | 0 |
| test_QV100_build | 1 | 0.72 | – | 0 | 0 | 0 |
| test_QV100_qasm2_import | 1 | 0.21 | 1.000 | 0 | 1 | 0 |
| test_QV_100_transpile | 1 | 1.23 | 1.017 | 0 | 0 | 1 |
| test_bigint_qasm2_import | 1 | 0.26 | 1.000 | 0 | 1 | 0 |
| test_circSU2_100_transpile | 1 | 0.07 | 1.000 | 0 | 1 | 0 |
| test_circSU2_89_transpile | 1 | 0.02 | 0.841 | 1 | 0 | 0 |
| test_clifford_100_transpile | 1 | 1.71 | 1.008 | 0 | 0 | 1 |
| test_clifford_build | 1 | 0.00 | – | 0 | 0 | 0 |
| test_feynman_transpile | 50 | 0.71 | 0.975 | 32 | 8 | 10 |
| test_hamiltonians · all-to-all | 100 | 0.30 | 0.921 | 42 | 50 | 8 |
| test_hamiltonians · heavy-hex | 100 | 0.87 | 0.821 | 73 | 12 | 15 |
| test_hamiltonians · linear | 100 | 1.01 | 0.874 | 75 | 10 | 15 |
| test_hamiltonians · square | 100 | 0.98 | 0.826 | 76 | 13 | 11 |
| test_hamlib_hamiltonians_transpile | 81 | 0.96 | 0.913 | 55 | 9 | 17 |
| test_multi_control_circuit | 1 | 0.02 | – | 0 | 0 | 0 |
| test_multi_control_decompose | 1 | 0.13 | – | 0 | 0 | 0 |
| test_param_circSU2_100_bind | 1 | 0.06 | – | 0 | 0 | 0 |
| test_param_circSU2_100_build | 1 | 1.84 | – | 0 | 0 | 0 |
| test_random_clifford_decompose | 1 | 5.41 | – | 0 | 0 | 0 |
| test_square_heisenberg_100_transpile | 1 | 1.18 | 0.652 | 1 | 0 | 0 |
| **all** | 1044 | **0.47** | **0.878** | 561 | 350 | 124 |
