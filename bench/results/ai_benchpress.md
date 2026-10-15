# AI transpiler on the Benchpress device suite

Benchpress `qumo_gym/device_transpile` (133-qubit heavy-hex device, cz basis, level 2): qumo (`[qumo] ai = false`) vs qumo-ai (`ai = true`, `transpile_ai` level 2), same machine, same session. 140 passed / 22 skipped (circuit too large for backend) in both runs; Benchpress's own checks passed. Ratios are geometric means of (qumo-ai+1)/(qumo+1) of output 2q gates and 2q depth; time ratio is the median of mean transpile times.

| group | tests | 2q ratio | 2q-depth ratio | W/T/L (2q) | median time ratio |
|---|---:|---:|---:|---:|---:|
| feynman | 50 | 0.958 | 0.968 | 24/23/3 | 14.0 |
| all | 140 | 0.955 | 0.965 | 87/44/9 | 17.3 |
| hamiltonians | 81 | 0.950 | 0.960 | 57/18/6 | 19.9 |
| summit | 9 | 0.983 | 0.999 | 6/3/0 | 14.7 |

Largest wins: test_feynman_transpile[hwb10.qasm] 105796->82640; test_hamlib_hamiltonians_transpile[ham_ham_JW-14] 12461->10131; test_hamlib_hamiltonians_transpile[ham_bh_graph-1D-grid-pbc-qubitnodes_Lx-10_U-30_enc-unary_d-8] 28682->23329; test_feynman_transpile[hwb8.qasm] 17761->14636; test_feynman_transpile[hwb11.qasm] 310596->256681

Largest losses: test_hamlib_hamiltonians_transpile[ham_enc_gray_dvalues_d-level-16] 181->230; test_hamlib_hamiltonians_transpile[ham_bh_graph-1D-grid-pbc-qubitnodes_Lx-4_U-40_enc-gray_d-8] 4668->4805; test_feynman_transpile[vbe_adder_3.qasm] 75->76; test_hamlib_hamiltonians_transpile[ham_enc_gray_dvalues_8-8-8] 9861->9983; test_feynman_transpile[csla_mux_3.qasm] 116->117

Reference baselines (docs/COMPILER.md, docs/NEUTRAL_BENCHMARK_REPORT.md): qumo/qiskit 2q 0.928, qumo/tket 0.919 on the neutral harness; Benchpress device HamLib geometric mean qumo/qiskit 0.91.
