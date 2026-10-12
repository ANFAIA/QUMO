# QUMO on Qiskit Benchpress — results

> Run date: 2026-10-12. Same machine for every SDK (Apple Silicon, 10 cores: 4 performance + 6 efficiency, 32 GB; macOS 26).
> Benchpress @ `b695f30`, default configuration (`fake_torino` device, basis `id,sx,x,rz,cz`, optimisation level 2 for both SDKs), its own fork of `pytest-benchmark`.
> SDKs: **QUMO 0.1.0** (Mojo 1.1, called from Python through its CPython extension `build/qumo_py.so`) vs **Qiskit 2.5.2** (+ qiskit-ibm-runtime 0.50).
> Raw data: `bench/results/benchpress/{qumo,qiskit}_full.json.gz`; comparison: `bench/results/benchpress_qumo_vs_qiskit.md` (regenerate with `bench/analysis/benchpress_compare.py`).
> How to reproduce: `make py && bench/benchpress/install.sh <benchpress-checkout> && bench/run_final.sh`.

## Headline

| | QUMO | Qiskit 2.5.2 |
|---|---:|---:|
| Tests passed / failed / skipped (of 1066) | **1044 / 0 / 22** | 1044 / 0 / 22 |
| Geometric-mean time ratio QUMO/Qiskit (1044 common tests) | **0.47** (2.1× faster) | 1 |
| Geometric-mean two-qubit-gate ratio QUMO/Qiskit (1035 tests reporting 2Q counts) | **0.878** | 1 |
| Tests with fewer / equal / more 2Q gates than the other SDK | **561** / 350 / 124 | 124 / 350 / 561 |

The 22 skips are identical for both SDKs: 3 Feynman and 19 HamLib circuits larger than the 133-qubit device ("Circuit too large for given backend").

## By category

Ratios are QUMO/Qiskit (< 1 ⇒ QUMO better). "2Q" = `output_gate_count_2q` of the device's two-qubit gate, the metric Benchpress records.

| Category | n | Time | 2Q | QUMO fewer | tie | Qiskit fewer |
|---|---:|---:|---:|---:|---:|---:|
| QASMBench small · all-to-all / square / heavy-hex / linear | 4 × 42 | 0.18 / 0.27 / 0.28 / 0.28 | 0.926 / 0.910 / 0.920 / 0.922 | 52 | 113 | 3 |
| QASMBench medium · all-to-all / square / heavy-hex / linear | 4 × 23 | 0.11 / 0.61 / 0.54 / 0.72 | 1.000 / 0.937 / 0.896 / 0.973 | 44 | 35 | 13 |
| QASMBench large · all-to-all / square / heavy-hex / linear | 4 × 58 | 0.05 / 0.52 / 0.37 / 0.65 | 1.018 / 0.819 / **0.637** / 0.855 | 108 | 95 | 29 |
| HamLib Hamiltonians · all-to-all / square / heavy-hex / linear | 4 × 100 | 0.30 / 0.98 / 0.87 / 1.01 | 0.921 / 0.826 / 0.821 / 0.874 | 266 | 85 | 49 |
| HamLib on device (fake_torino) | 81 | 0.96 | 0.913 | 55 | 9 | 17 |
| Feynman on device | 50 | 0.71 | 0.975 | 32 | 8 | 10 |
| 100-qubit device suite ("summit", 9 tests) | 9 | see below | see below | 4 | 2 | 3 |
| Construction (8 tests) | 8 | 0.00–1.84 | – | – | – | – |
| Manipulation (4 tests) | 4 | 0.13–5.41 | – | – | – | – |

### 100-qubit device suite (fake_torino, 133 q heavy-hex)

| Test | QUMO 2Q | Qiskit 2Q | QUMO s | Qiskit s | Time ratio |
|---|---:|---:|---:|---:|---:|
| QFT_100 | 8 272 | 9 338 | 0.311 | 0.181 | 1.72 |
| QV_100 | 98 397 | 96 732 | 3.751 | 3.053 | 1.23 |
| circSU2_89 | 300 | 357 | 0.052 | 2.803 | **0.02** |
| circSU2_100 | 300 | 300 | 0.004 | 0.057 | 0.07 |
| BV_100 | 197 | 196 | 0.021 | 0.037 | 0.58 |
| square_heisenberg_100 | 1 083 | 1 662 | 0.057 | 0.048 | 1.18 |
| QAOA_100 | 8 576 | 8 792 | 0.221 | 0.162 | 1.37 |
| BVlike_simplification | 0 | 0 | 0.001 | 0.003 | 0.16 |
| clifford_100 | 58 442 | 57 989 | 1.065 | 0.622 | 1.71 |

*(Qiskit columns here are this machine's run; small differences to numbers quoted earlier in the project come from Qiskit's seed-dependent routing.)*

### Where Qiskit is still better
- **Large dense circuits on the device**: QV_100 (+1.7% 2Q) and clifford_100 (+0.8%), and Qiskit is 1.2–1.7× faster on QFT_100, QV_100, QAOA_100 and clifford_100 — Qiskit's Rust passes are multithreaded across all cores, while QUMO's routing trials are parallel but its optimisation passes are single-threaded.
- **QASMBench large, all-to-all**: 1.018 (9 losses, 48 ties). These circuits need no routing; Qiskit's 2-qubit peephole resynthesis finds a few more savings.
- **Manipulation**: `random_clifford_decompose` (5.4× slower — QUMO translates the given circuit gate by gate; Qiskit first resynthesises the Clifford tableau) and `param_circSU2_100_build` (1.8× slower — QUMO interns 1000 parameter names with a linear scan).

## What is *not* comparable in these numbers

Benchpress hands each SDK a different input in some tests (see [BENCHPRESS_BIAS_REPORT.md](BENCHPRESS_BIAS_REPORT.md), finding 7):
- **Hamiltonians**: Qiskit receives an abstract `PauliEvolutionGate` and synthesises it inside the timed transpile; QUMO builds its own Trotter circuit (`qumo_py.hamiltonian_circuit`) outside the timed region and resynthesises it with its `pauli_simp` pass inside. Both are "the SDK's best path from a Hamiltonian to a device circuit", which is what the test measures, but the split between untimed and timed work differs.
- **QV_100**: Qiskit compiles `QuantumVolume(100, 100, seed)` with unsynthesised `UnitaryGate`s; QUMO compiles the same instance from Benchpress' pre-synthesised `qv_N100_12345.qasm`.
- **Construction**: Qiskit's `quantum_volume` and `.control()` build lazy objects; QUMO synthesises eagerly (more work, still faster).
- **Correctness**: Benchpress does not check that outputs implement the input. QUMO's compiler is equivalence-tested in its own suite and in the neutral harness (4970/4970 verified), but Benchpress itself would not catch a wrong answer from any SDK.

## Disclosure

QUMO's compiler was improved during this project *using* some Benchpress categories as development targets (round 2 targeted the HamLib and QASMBench-small gaps shown by a first full run: HamLib all-to-all went 1.426 → 0.921, QASMBench small 1.03–1.05 → 0.91–0.93). Results on a benchmark used for tuning overstate generalisation. The first full run (before round 2) is kept in `bench/results/benchpress/qumo_full_round1.json.gz` (2Q ratio 1.070, time ratio 0.32). Independent replication on unseen circuits is needed before drawing strong conclusions.
