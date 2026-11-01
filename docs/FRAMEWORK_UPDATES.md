# Framework updates reviewed, 2026-11-01

What changed in the frameworks QUMO draws from, and what QUMO did about it. Status:
**done** (with the place it landed), **deferred** (worth doing, not in this round;
tier 2 or 3) or **skipped** (with the reason). "Have" means QUMO already had it.
Cross-checks live in `bench/crosscheck/tier1.py`; per-feature numbers are in the
module docs linked below.

Versions: Qiskit 2.6.1, qiskit-ibm-runtime 0.51, Aer 0.18.2, pytket 2.18.5, tket-py
0.15.13, guppylang 1.2.0, guppyft 0.5.1, CUDA-Q 0.16.0, cudaq-qec 0.8.1, PennyLane
0.45.1, Catalyst 0.15, Cirq 1.7.0, Stim 1.16, PyMatching 2.4, QDK 1.33.1, Braket
1.127.7, Qibo 0.3.6. Toolchain: Mojo 1.2.0, MAX 26.7.0.

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
| `NoiseModel.from_target`, `emulate` | Braket 1.112 | done, `qumo.exec.emulate`, `qumo_emulate` |
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
| FT cost of arbitrary rotations | PennyLane estimator | have (`SynthesisModel` in `qumo.exec.resources`, commit 5a965fa) |

## QEC and FT

| Feature | Source | Status |
|---|---|---|
| Correlated (two-pass) matching | PyMatching 2.3, sinter 1.16 | done, `mwpm_correlated`, opt-in; paired counts match PyMatching, worse at d = 3 in both |
| Stim text import, `PAULI_CHANNEL_2`, `E`, `MPP` | Stim, QDK-Stim | done, `from_stim`, [QEC](QEC.md) |
| DEM parser, DEM sampler | cudaq-qec 0.7, cuStabilizer 0.3 | done, `parse_dem`, `sample_dem` |
| `missing_detectors`, `count_determined_measurements` | Stim 1.16 | done, `qumo.qec.lint`, matches Stim |
| Sliding-window decoder | cudaq-qec 0.5 | done, `SlidingWindowDecoder[D]` |
| `reichardt` QEC style | guppyft 0.5 | done; not compared with guppyft on Selene |
| Iceberg Pauli exponentials | guppyft 0.5 | done, verified at theta = pi/2 |
| `helix_code()` | arXiv:2609.03194 | done: definition only |
| Helix architecture | Quantinuum | deferred: needs hypergraph decoding and unreadable figures |
| BP+OSD, Relay-BP, BB codes | cudaq-qec 0.5-0.8 | deferred |
| QRE v3 Pareto estimator, tiered estimator | QDK 1.28-1.33, `pennylane.estimator`, CUDA-Q Logical | deferred |
| Colour codes, chunks, flows | Stim, Chromobius | deferred |
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
| Fermi-Hubbard, lattices | Qibo 0.3.6, QDK 1.33 | done, `fermi_hubbard`, `Graph.square_lattice`, `Graph.triangular_lattice` |
| `lie_closure` | Qibo 0.3.6 | done, Pauli strings only |
| PUB `split` | Braket 1.120 | done, `split_pub`, seeds follow the original index |
| MPF | qiskit-addon-mpf | done, `qumo.algorithms.mpf` |
| Suzuki orders above 2 | PennyLane, CUDA-Q Algorithms | done, `trotter(order=4, 6, ...)` |
| Arithmetic, QROM | QDK 1.25-1.31, PennyLane | deferred |
| LCU, qubitization, QSVT | CUDA-Q Algorithms, PennyLane, Qibo | deferred |
| Classical shadows, tomography | Qibo 0.3.6 | deferred |
| Fermion mappers, FCIDUMP | qiskit-fermions, CUDA-Q | deferred |
| Cutting in Mojo | addon-cutting | deferred |
| Momentum QNG, `conjugate()` | PennyLane | deferred |

## Toolchain and ecosystem

| Change | Status |
|---|---|
| Mojo 1.1 to 1.2, MAX 26.6 to 26.7 | done: [MOJO_GUIDE](MOJO_GUIDE.md) lists what moved; pins in the Makefile |
| Qiskit 2.6 baselines | deferred: README and benchmark reports still compare with 2.5.2 |
| `qiskit-aer-gpu` wheels are back | docs updated earlier; the harness fallback is still there |
| `QAOAAnsatz` deprecation | done, `qaoa_ansatz` in the reference script |
| Plugin: `[helios]` and `[local-quantinuum]` conflict, `with_minimal_opt` | documented; code change deferred |
| Qiskit Functions, serverless, IBM-native instructions, Selene plugins | skipped: service or vendor plumbing |
