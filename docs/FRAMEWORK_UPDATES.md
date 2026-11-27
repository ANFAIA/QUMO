# Framework updates reviewed, 2026-11-26

What changed in the frameworks QUMO draws from, and what QUMO did about it. Status:
**done** (with the place it landed), **deferred** (worth doing, not in this round;
tier 2 or 3) or **skipped** (with the reason). "Have" means QUMO already had it.
Cross-checks live in `bench/crosscheck/tier1.py`; per-feature numbers are in the
module docs linked below.

Versions (2026-11-26): Qiskit 2.6.2, qiskit-ibm-runtime 0.52.1, Aer 0.18.4, pytket
2.18.5, tket-py 0.15.16, tket (Rust) 0.23.1, guppylang 1.2.2, guppyft 0.7.0, CUDA-Q
0.17.0, cudaq-qec 0.9.1, PennyLane 0.45.1, Catalyst 0.15.0, Cirq 1.7.0, Stim 1.16.0,
PyMatching 2.4.0, QDK 1.34.0, Braket 1.128.2, Qibo 0.3.7; also reviewed: bloqade-tsim
1.1.0, bloqade-analog 0.16.9. Toolchain: Mojo 1.2.0, MAX 26.7.0. The versions are those
of the latest review; rows from earlier rounds name the version they came from, and
"done" rows were not re-checked against the new releases. Candidates of the 2026-11-26
review are in [their own section](#candidates-of-the-2026-11-26-review).

## Language and IR

| Feature | Source | Status |
|---|---|---|
| `ctrl(k) @`, `negctrl(k) @` for any gate; MCX, MCPhase synthesis (`kg24`, `i15`, no-ancilla) | Qiskit 2.1-2.6 | done, `qumo.ir.mc`, [ALGORITHMS](ALGORITHMS.md) |
| `hp24`, `sp22` MCX | Qiskit 2.6 | deferred: hp24 beats QUMO from 7 controls without ancillas |
| Lazy multi-controlled opcode with an `mc_synth` pass | Qiskit 3.0 direction | deferred |
| `box`, `box[dur]`, annotations | Qiskit 2.1, CUDA-Q atomic regions | done: barrier on each side, annotations warn (QM-W052) |
| `switch`, Qiskit's `switch_dummy` and V1 forms | OpenQASM 3, Qiskit | done, register conditions |
| Verbatim box semantics, `Circuit.protect` | Braket, CUDA-Q 0.16 | deferred: `box` only stops motion across it |
| `delay`, `stretch`, scheduling, dynamical decoupling | Qiskit, Cirq | deferred |
| Constant `break` / `continue` | QDK 1.31 | done through `Program`: a loop with either stays a region, simulated and compiled; [CONTROL_FLOW](CONTROL_FLOW.md) |
| `while`, classical variables, CFG IR | Qiskit, Guppy; `cirq.While` exists only in 1.8.dev, unreleased (the latest Cirq release is 1.7.0) | done on the CPU, `qumo.ir.program`: regions on a tape with a derived CFG, a separate type from `Circuit`, [CONTROL_FLOW](CONTROL_FLOW.md). Hardware and GPU are deferred |
| Text circuit drawer | Cirq, Qibo, PennyLane | done, `Circuit.draw()`, `qumo draw`, `qumo_draw`, `qumo_py.draw` |
| Quantikz / LaTeX drawing | Cirq 1.7 | deferred |
| Multi-qubit Pauli operators beyond 64 qubits | Qiskit `SparseObservable` | deferred, needed for Pauli propagation |
| `defcal`, QPY, C API, Rust internals, HUGR internals | Qiskit, Quantinuum | skipped: no neutral value, or vendor formats |
| `FreeParameterExpression` math functions | Braket 1.119 | skipped: affine `Param` is deliberate, adjoint gradients rely on it |
| OpenQASM 2 `^` as the power operator; `pi`, `tau`, `euler` rejected as declared names (QM-E522) | Qiskit 2.6.x fixes (expression domain errors, constant-name collisions); research-watch 2026-11-17 | done, `qumo/ir/qasm_parser.mojo`, regression tests in `tests/test_qasm_qiskit26.mojo`. Both were QUMO bugs the tests exposed |
| QIR export | QDK, Qibo | deferred |

## Compiler

| Feature | Source | Status |
|---|---|---|
| Target angle bounds, `WrapAngles` | Qiskit 2.2 | done, `wrap_angles`, spec `rzz[0:pi/2]`; plugin no longer needs `_is_valid_rzz_pub_helper`, `FoldRzzAngle` stays as a fallback |
| Calibration plumbing, `VF2PostLayout`, fidelity scoring | Qiskit 2.2, 2.3 | done, `post_layout`, `Target.set_calibration`, [COMPILER](COMPILER.md) |
| Error-aware routing, KAK and 1q choice | Qiskit 2.5, 2.6 | deferred |
| `LitinskiTransformation`, PBC | Qiskit 2.2-2.5, PennyLane 0.44, Catalyst | done, `to_pbc`; `PPM` compilation passes (`ppr_to_ppm`, `reduce_t_depth`) deferred |
| T-count objective | Qiskit `COUNT_T`, tket T optimisation | done, `pauli_simp(objective=t)` |
| TODD (`t_opt`) | tket 0.15.10, PennyLane `todd` | deferred: largest T gain available, FT-only |
| PCOAST-style greedy synthesis | tket pg-libs | deferred |
| gridsynth, Clifford+T preset | Qiskit 2.3, CUDA-Q 0.16, Catalyst 0.14, Qibo | deferred: needs big integers |
| QM-E604 names the input gate | review finding | done |
| MCTS Pauli synthesis | Qiskit 2.6 | deferred |
| Measurement-aware cleanups, light cone | Qibo 0.3.6, Cirq 1.7 | deferred: must stay out of O0-O3 |
| QSD for 3-5 qubit unitaries | CUDA-Q 0.16, Qiskit | deferred: no n-qubit unitary op |
| `disentangle_cnot` | PennyLane 0.45 | deferred |
| Directed targets, native `gate_direction` | Cirq, Qibo | deferred |
| 3-iSWAP SU(4) | Cirq 1.8.dev | deferred, unreleased upstream |
| Multithreaded passes, foldhash | Qiskit 2.5 | skipped: Rust tuning; moves the speed baseline, rerun benchmarks against 2.6 |

## Noise and simulation

| Feature | Source | Status |
|---|---|---|
| `NoiseModel.from_target`, `emulate` | Braket 1.112 | done, `qumo.noise.emulate`, `qumo_emulate` |
| Trajectory deduplication | CUDA-Q 0.14 PTSBE | done, `sample_ptsbe`, [SIMULATION](SIMULATION.md) |
| T-REX | Cirq contrib, `qiskit-mitigation` | done, `trex_expectation`; not compared with `qiskit_mitigation.TREX` |
| M3 | `mthree` 3.0 | done, `mitigate_counts_m3`, matches to 5e-9 |
| Density matrix, Kraus channels | QDK 1.28, CUDA-Q, Cirq, Aer | done, `qumo.sim.density`, `qumo.ir.kraus` |
| Stabilizer branching | QDK 1.32 | done, `qumo.sim.branching`, with a measurement that never grows the list |
| Pauli propagation, OBP | cuPauliProp, pauli-prop, addon-obp | deferred |
| MPS | CUDA-Q, qibotn, Aer | deferred: needs a complex SVD |
| Twirling, template programs, Samplomatic model | Qiskit | deferred |
| PEC, PEA, Pauli-Lindblad learning | Qiskit | deferred |
| Native Pauli rotation kernel | Lightning 0.45 | deferred |
| Noisy GPU trajectories | QDK 1.25 | deferred |
| Qubit loss, leakage | QDK 1.25-1.32, cuStabilizer 0.5, Selene | deferred |
| Lindblad dynamics | CUDA-Q | deferred |
| Extended stabilizer, TDVP, DMRG | Aer, cuDensityMat | skipped: wait for MPS |
| FT cost of arbitrary rotations | PennyLane estimator | have (`SynthesisModel` in `qumo.compiler.resources`, commit 5a965fa) |

## QEC and FT

| Feature | Source | Status |
|---|---|---|
| Correlated (two-pass) matching | PyMatching 2.3, sinter 1.16 | done, `mwpm_correlated`, opt-in; paired counts match PyMatching, worse at d = 3 in both |
| Stim text import, `PAULI_CHANNEL_2`, `E`, `MPP` | Stim, QDK-Stim | done, `from_stim`, [QEC](QEC.md) |
| DEM parser, DEM sampler | cudaq-qec 0.7, cuStabilizer 0.3 | done, `parse_dem`, `sample_dem` |
| Exact DEM comparison (`gather_dem`, `compare_dem`) | DEMVal, Luo and Suzuki, arXiv:2611.20501; research-watch 2026-11-22 | done, `qumo.qec.dem_compare`, [QEC](QEC.md); example `examples/error_correction/dem_exact_comparison.mojo`. Oracle only: no injection reference, no label alignment |
| Exact circuit distance of a DEM (`circuit_distance`) | Cheng, Guo and Pan, arXiv:2611.08247 (SI Sec. 5.1); research-watch 2026-11-27 | done, `qumo.qec.distance`, [QEC](QEC.md); example `examples/error_correction/circuit_distance_exact.mojo`. Weight-bounded search only: no sector reduction, rank DP or time collapse, so d <= 5 on full surface-code DEMs; tropidist not compared |
| `missing_detectors`, `count_determined_measurements` | Stim 1.16 | done, `qumo.qec.lint`, matches Stim |
| Sliding-window decoder | cudaq-qec 0.5 | done, `SlidingWindowDecoder[D]` |
| `reichardt` QEC style | guppyft 0.5 | done; not compared with guppyft on Selene |
| Iceberg Pauli exponentials | guppyft 0.5 | done, verified at theta = pi/2 |
| `helix_code()` | arXiv:2609.03194 | done: definition only |
| Exact fault-tolerance check of state preparation (`fault_tolerant_state_preparation`) | Schmid and Wille, arXiv:2611.25107 (Definition 1); research-watch 2026-11-23 | done, `qumo.ft.faults`, [FT](FT.md); example `examples/ft/ft_state_prep_check.mojo`, counts reproduced with Stim 1.16. Check constructions (Block, Bare, Cat) and the Pareto search deferred; MQT QECC not compared |
| Helix architecture | Quantinuum | deferred: needs hypergraph decoding and unreadable figures |
| BP+OSD, Relay-BP, BB codes | cudaq-qec 0.5-0.8 | deferred |
| QRE v3 Pareto estimator, tiered estimator | QDK 1.28-1.33, `pennylane.estimator`, CUDA-Q Logical | deferred |
| Colour codes, chunks, flows | Stim, Chromobius | deferred |
| Exact error of small magic-state factories (3 T to CS, 8-input CCZ, 9-input T) | Balada Gaggioli and Marecek, arXiv:2610.31115; research-watch 2026-11-27 | done as an example, `examples/error_correction/magic_state_distillation_exact.mojo`, [QEC](QEC.md). Polynomials and harmful counts confirmed by state-vector enumeration; optimality claims, larger factories and 15-to-1 not checked; `resources.mojo` unchanged |
| 15-to-1 distillation gadget | guppyft PR 483 | deferred, unreleased upstream |
| Parallel fused MWPM | NV-Fusion | deferred |
| `ChainedDecoder`, DEM options | cudaq-qec | deferred |
| AI predecoders with vendor weights | NVIDIA Ising | skipped: access-controlled weights |
| Real-time FPGA and GPU links | CUDA-Q Realtime, Backline | skipped: bound to vendor hardware |

## Algorithms

| Feature | Source | Status |
|---|---|---|
| Trim SQD, configuration recovery, `partition_subsample` | qiskit-addon-sqd 0.14 | done in `sqd.mojo`; not compared with the addon |
| IQP template, `iqp_expval` | PennyLane 0.44 | done, `qumo.algorithms.iqp` |
| CVaR objective for VQE and QAOA (`alpha`) | Barkoutsos et al., arXiv:1907.04769; Qiskit `VQE(aggregation=)`; research-watch 2026-11-17 | done, `vqe(alpha=)`, `qaoa_maxcut(alpha=)`, `CVaRObjective`, [ALGORITHMS](ALGORITHMS.md); example `examples/optimization/qaoa_cvar.mojo`. Not a guaranteed gain: the example shows where it is flat |
| Dicke-state preparation | Bärtschi and Eidenbenz, arXiv:1904.07358; prompted by arXiv:2611.27640; research-watch 2026-11-22 | done, `dicke_state`, `append_dicke_state`, `dicke_amplitudes`, [ALGORITHMS](ALGORITHMS.md); example `examples/algorithms/dicke_state.mojo`. The ancilla-free O(n)-gate, O(log n)-depth construction of 2611.27640 is deferred: no gate constants; for n <= 31, and for the 31-qubit blocks used when n >= 32, the paper relies on generic exact synthesis, so no explicit circuit is given |
| Fermi-Hubbard, lattices | Qibo 0.3.6, QDK 1.33 | done, `fermi_hubbard`, `Graph.square_lattice`, `Graph.triangular_lattice` |
| `lie_closure` | Qibo 0.3.6 | done, Pauli strings only |
| PUB `split` | Braket 1.120 | done, `split_pub`, seeds follow the original index |
| MPF | qiskit-addon-mpf | done, `qumo.algorithms.mpf` |
| Suzuki orders above 2 | PennyLane, CUDA-Q Algorithms | done, `trotter(order=4, 6, ...)` |
| Optimised order-8 product formula, 21 Strang steps | Maležič and Ostmeyer, arXiv:2611.27896 (Table 3); research-watch 2026-11-27 | done, `trotter(order=8, scheme="mo8_q21")`, `mo8_q21_weights`, [ALGORITHMS](ALGORITHMS.md); example `examples/chemistry_and_physics/trotter_order8_cost.mojo`. Opt-in: Suzuki order 4 is cheaper at loose targets (1e-1, 1e-2) in the measured case (XZ chain, t = 10); where it loses depends on t. The 30-step generic scheme (Table 4) is deferred: it is not a list of Strang weights |
| First-order Trotter error: rigorous bound and leading-order estimate (`trotter_error`) | Dion and D'Anjou, arXiv:2611.33438 (eq. 6, 16, 19-20; eq. 6 is Childs et al., PRX 11, 011020); research-watch 2026-11-27 | done, `qumo.algorithms.trotter_error`, [ALGORITHMS](ALGORITHMS.md); example `examples/chemistry_and_physics/trotter_error_bound.mojo`. Order 1 only: the paper's eq. 7 (second-order bound) is exceeded in tests as printed, and eq. 18 could not be reproduced; the estimate is labelled as not a bound |
| Arithmetic, QROM | QDK 1.25-1.31, PennyLane | deferred |
| LCU, qubitization, QSVT | CUDA-Q Algorithms, PennyLane, Qibo | deferred |
| Classical shadows, tomography | Qibo 0.3.6 | deferred |
| Fermion mappers, FCIDUMP | qiskit-fermions, CUDA-Q | deferred |
| Cutting in Mojo | addon-cutting | deferred |
| Momentum QNG, `conjugate()` | PennyLane | deferred |

## Candidates of the 2026-11-26 review

Proposals only: nothing in this table is implemented. Each row keeps the file's status vocabulary (nothing is **done**; "adopt now" and "prototype" rows are **deferred** until a change lands) and adds the review's rank.

| Feature | Source and date | Status |
|---|---|---|
| Import `T`, `T_DAG`, `S[T]`, `S_DAG[T]` in `from_stim`, with raw detector sampling and early post-selection on the branching simulator; other tagged forms get a coded error (today `I[R_Z(theta=...)]` raises an uncoded one) | tsim 1.1.0; 2026-11-26 | deferred (rank: adopt now): [plan](TSIM_BLOQADE_PLAN.md#16-recommended-first-slice) |
| AHS JSON export with a capabilities-driven limits check, in the hardware plugins | bloqade-analog 0.16.9, Braket AHS schema; 2026-11-26 | deferred (rank: adopt now): [plan](TSIM_BLOQADE_PLAN.md#25-recommended-first-slice) |
| Run the IBM plugin tests on qiskit-ibm-runtime 0.52.1, then document the result or bound the version (`>=0.50.1` has no upper bound; 0.52.0 is breaking) | qiskit-ibm-runtime 0.52.0 (2026-11-09), 0.52.1 (2026-11-12); 2026-11-26 | deferred (rank: adopt now): tests not run; the module paths the plugin imports for `is_isa_circuit`, `FoldRzzAngle` and the executor `Sampler` exist at tag 0.52.1 |
| Before a Mojo 1.3 upgrade, audit `String.find()`/`rfind()` uses and `UnsafePointer` | Mojo 1.3 nightly, unreleased (2026-11-25); 2026-11-26 | deferred (rank: adopt now): not built against the nightly |
| Active-width planner (report only), then a compile-once active-state-vector sampler | tsim 1.0.0 (2026-11-04), Clifft; 2026-11-26 | deferred (rank: prototype): go/no-go on the measured width |
| Local detuning in `qumo.ir.analog` and `qumo.sim.rydberg` | bloqade-analog 0.16.9 (2026-03-02); 2026-11-26 | deferred (rank: prototype): must stay device-neutral |
| Parallel-window decoding | CUDA-Q QEC 0.9.0 (2026-11-16); 2026-11-26 | deferred (rank: prototype): QUMO has only the serial `SlidingWindowDecoder`; compare logical error rate and throughput |
| Learning DEM error rates from detector data | CUDA-Q QEC 0.9.0, closed source; 2026-11-26 | deferred (rank: prototype): needs an open published method first |
| Mixed (probabilistic) Rz synthesis | PennyLane development changelog (unreleased), arXiv:2203.10064 (not read); 2026-11-26 | deferred (rank: prototype): compare expected T count at equal channel error |
| Cross-check `pauli_simp(objective=t)` with `phase_fold` against tket Pauli-graph resynthesis | tket (Rust) 0.22.1 (2026-10-28); 2026-11-26 | deferred (rank: prototype): independent T and two-qubit counts |
| Amplitude estimation and iterative QPE as library functions | Qibo 0.3.7 (2026-11-10); 2026-11-26 | deferred (rank: prototype): `qumo.algorithms.library` has neither; some examples build their own |
| A user-facing agent skill for QUMO and an execution-verified evaluation of agents writing QUMO code | vendor agent skills (Qiskit, AWS, CUDA-Q Algorithms, tsim), arXiv:2604.08570; 2026-11-26 | deferred (rank: prototype): measure pass@1 with and without skill and feedback |
| tsim cross-check script (after the first slice) and AHS emulator cross-check (after the export); AHS import; blockade-subspace emulator | tsim 1.1.0, bloqade-analog 0.16.9; 2026-11-26 | deferred (rank: defer): optional, no CI dependency |
| Parametric ZX engine; non-Clifford fault distance and determined measurements | tsim (the last two unreleased and experimental); 2026-11-26 | deferred (rank: defer): depends on the width results |
| Native Braket AHS client or gate-model Braket provider | Braket SDK 1.128.2 (2026-11-23); 2026-11-26 | deferred (rank: defer): needs credentials and paid shots |
| Spacetime Pauli checks; fermion and boson operator packages | IBM QDC 2026 (2026-11-11); 2026-11-26 | deferred (rank: defer) |
| Gadget audit with certified fault-distance bounds | QDK `qdk.ec`, 1.33; 2026-11-26 | deferred (rank: defer): `qumo.ft.faults` covers state preparation only |
| Decoder ensembles, cancellation, task graphs, chunked DEMs | CUDA-Q QEC 0.9.0; 2026-11-26 | deferred (rank: defer): wait for a real-time use case |
| Stabilizer-rank sampling, a tsim or bloqade-analog Python bridge, GPU near-Clifford sampling, a sweep/builder language, three-level atom emulation | tsim 1.1.0, bloqade-analog 0.16.9; 2026-11-26 | skipped (rank: skip): provisional upstream, heavy dependencies or low value now |
| IBM `Calibrator`, `resource_units`, IBM Quantum Functions, pulse access | qiskit-ibm-runtime 0.52.0, QDC 2026; 2026-11-26 | skipped (rank: skip): vendor service features |
| CUDA-Q Decoding Server, EDEM, TCBP/JOBS, `custabilizer` | CUDA-Q 0.17.0, QEC 0.9.0 (2026-11-16); 2026-11-26 | skipped (rank: skip): vendor-bound, closed or declared unstable; QUMO has a GPU frame sampler (speed not compared) |

## Toolchain and ecosystem

| Change | Status |
|---|---|
| MCP tool annotations per tool (`ToolHints`); a tool without hints is no longer announced as read-only | done, `qumo/ai/tools.mojo`, [AI_NATIVE](AI_NATIVE.md); research-watch 2026-11-17 |
| Mojo 1.1 to 1.2, MAX 26.6 to 26.7 | done: [MOJO_GUIDE](MOJO_GUIDE.md) lists what moved; pins in the Makefile |
| Qiskit 2.6 baselines | deferred: README and benchmark reports still compare with 2.5.2 |
| `qiskit-aer-gpu` wheels are back | docs updated earlier; the harness fallback is still there |
| `QAOAAnsatz` deprecation | done, `qaoa_ansatz` in the reference script |
| Plugin: `[helios]` and `[local-quantinuum]` conflict, `with_minimal_opt` | documented; code change deferred |
| Qiskit Functions, serverless, IBM-native instructions, Selene plugins | skipped: service or vendor plumbing |
