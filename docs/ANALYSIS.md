# Deep analysis: the 8 reference quantum frameworks (November 2026)

> Goal: extract the best of **Qiskit, TKET, CUDA-Q, PennyLane, Cirq, Q#/QDK, Amazon Braket SDK and Qibo**, decide how to combine it, and rebuild it in **Mojo** as one stack that is efficient (speed and resource use) and **AI-native**.
>
> Versions reviewed (2026-11-26): Qiskit 2.6.2, qiskit-ibm-runtime 0.52.1, Aer 0.18.4 · pytket 2.18.5, guppylang 1.2.2, guppyft 0.7.0, hugr (Rust) 0.31.1 / hugr-py 0.18.8, tket (Rust) 0.23.1 / tket-py 0.15.16 · CUDA-Q 0.17.0, CUDA-Q QEC 0.9.1, cuQuantum SDK 26.11.0 (CUDA-Q 0.17 ships 26.09.0), `cudaq-logical` 0.2.0, CUDA-Q Algorithms 0.2.0 · PennyLane 0.45.1 and Catalyst 0.15.0 (0.46 in beta) · Cirq 1.7.0, qsim 0.22.1 · QDK 1.34.0 (`microsoft/qdk`, package `qdk`; `qsharp` is retired) · Braket SDK 1.128.2 · Qibo 0.3.7, Qibolab 0.2.18, Qibocal 0.2.6 · Stim/Sinter 1.16.0, PyMatching 2.4.0. Additional references (§14): `bloqade-tsim` 1.1.0, `bloqade-analog` 0.16.9. What QUMO took from them: [FRAMEWORK_UPDATES.md](FRAMEWORK_UPDATES.md).
>
> Performance figures quoted are vendor-reported unless stated otherwise (e.g. Benchpress is authored by IBM — see [BENCHPRESS_BIAS_REPORT.md](BENCHPRESS_BIAS_REPORT.md)).

---

## What changed in this revision

Reviewed on 2026-11-26; the previous review was 2026-11-01. "Unreleased" means seen on a development branch or in a pre-release, so it can still change. Only the statements named below, and the sections rewritten with them, were checked again; other text is carried over from earlier reviews.

**New since the last review:** releases of Qiskit, qiskit-ibm-runtime (the top-level `Sampler`/`Estimator` are now client-side), guppyft and tket (breaking), CUDA-Q 0.17 and CUDA-Q QEC 0.9, QDK 1.34, Braket SDK 1.128 and Qibo 0.3.7 (large transpiler release). PennyLane, Catalyst, Cirq, Stim, PyMatching, pytket, Mojo and MAX had no new stable release. §14 adds tsim and bloqade-analog.

**Statements that were already wrong or misleading on 2026-11-01:**

| Section | Old statement | Correction |
|---|---|---|
| §0.3 | ORBIT-Q, PennySynth and QuanBench, "2026 studies", agree that agents need a verifier loop, a stable API and current documentation; Qiskit's corpus is the most stale and Cirq the most stable | QuanBench is from October 2025. The three do not reach a joint conclusion, and we found neither corpus ranking in them. Rewritten in §0.3 |
| §1 | Aer 0.18.2 brought the GPU wheels back | `qiskit-aer-gpu` returned in 0.18.0 (2026-10-20); the release before it was 0.15.1 (2024-09-13) |
| §1 | No in-SDK stabilizer simulator or decoders | `qiskit.quantum_info.StabilizerState` is documented as a stabilizer simulator |
| §1 | C pass manager in "2.6rc1" | 2.6.0 final was released on 2026-10-16 |
| §10 | "Trained code assistants (Qiskit Code Assistant)" | IBM discontinued the service ([Qiskit/mcp-servers](https://github.com/Qiskit/mcp-servers) README) |
| §2 | pytket had 87 failures in Benchpress | The paper reports 86 of 1,032 (pytket 1.31.0, a 2024 version) |
| §2 | Only Badger listed for the Rust `tket` optimiser | `tket` (Rust) 0.22.1 had already added Pauli-graph resynthesis |
| §3 | Decoder list of CUDA-Q QEC 0.8 | It omitted `nv-fusion-decoder`, shipped in 0.8.0 |
| §6 | Density matrix through `run_qir` | `run_qir` accepts only stabilizer/Clifford, CPU and GPU; the density-matrix simulator is a separate class |
| §6 | QEC support is the resource estimator; IR pipeline without FIR transforms | `qdk.ec` and `run_qir(qodec=)` were already in 1.33; so were the FIR transforms |
| §7 | The local emulator validates and adds noise to circuits | It is documented for verbatim circuits |
| §8 | Trivial placer; 0.3.6 added Clifford+T through `pygridsynth` | There is no Trivial placer in 0.3.6; `pygridsynth` and `clifford_plus_t` were already in 0.3.5 |
| §10 | AlphaQubit 2 as an example of "neural pre-decoder + classical global decoder" | The paper describes a neural-network decoder, not a pre-decoder |
| §11.3 | Qibo chooses the simulator by circuit structure | Not supported by the Qibo code we read: the backend is selected globally. Qibo is removed from that row |

---

## 0. Executive summary

1. **No framework wins all four areas.** Qiskit leads compilation for real hardware; TKET leads algebraic optimisation (ZX, phase polynomials, Pauli graphs) and IR design (HUGR); CUDA-Q leads heterogeneous execution and GPUs; PennyLane leads automatic differentiation and inspectable pipelines; Q# leads functors and compile-time capability analysis; Braket leads device-as-data abstraction and task batching; Cirq leads API stability (QUMO's judgement) and is the usual way into the Stim/PyMatching QEC tools; Qibo leads "calibration as code" for labs.
2. **Most of them share the same language split**: a slow Python front end over a C++/Rust/MLIR core. Consequences: per-gate Python overhead, two semantic worlds (tape vs. qjit in PennyLane, kernel-Python vs. Python in CUDA-Q, pytket vs. HUGR in TKET), complex builds, and constantly changing APIs.
3. **API churn is a cost for agents.** What the studies support, each within its scope: (a) repair from execution feedback raises success a lot (QuanBench+, arXiv:2604.08570: best Pass@1 on Qiskit, Cirq and PennyLane goes from 59.5/54.8/42.9% to 83.3/76.2/66.7%; on Qiskit HumanEval, arXiv:2603.22184 reports up to 85% for an execution-feedback agent, against 60-65% zero-shot or with retrieval, and modest gains from retrieval). (b) SDK version drift is a measured failure source for Qiskit, and migration notes repair only part of it (arXiv:2607.04072). (c) For PennyLane, PennySynth (arXiv:2605.25572) reports +25 to +28 points of pass@5 for its retrieval pipeline over the same model without retrieval (authors' figures; the pipeline has an execution repair loop, and retrieval can also bring in stale names). QuanBench (arXiv:2510.16779, October 2025, Qiskit only) lists outdated APIs among failure causes and finds them the least frequent class overall. ORBIT-Q (arXiv:2607.03105) compares agents on differentiable research workflows and leaves Qiskit and Cirq out of its main evaluation. An evaluation audit (arXiv:2610.28678) finds benchmark harness defects that cause false passes and false failures, so the verifier must be isolated and checked. We found no study that ranks frameworks by API stability.
4. **Mojo closes the split**: Python-like syntax (familiar to LLMs), systems performance (SIMD, threads, GPUs via MAX), one language end to end, and an MLIR-based compiler. QUMO uses this to ship IR, compiler, simulators, QEC and the agent protocol **in one binary**.
5. **QUMO synthesis** (details in §11): a flat IR like Qiskit's `CircuitData` + Stim annotations + Q# functors; a textual, inspectable pass pipeline (PennyLane/pytket); SABRE routing (Qiskit) + phase folding (TKET); PUB-style primitives (Qiskit) with ProgramSet-style batching (Braket) and asynchronous multi-device orchestration (CUDA-Q); Pauli-frame simulation + DEM as the "narrow waist" (Stim) with swappable decoders including AI decoders (CUDA-Q QEC / AlphaQubit); noise models derived from calibration (Qibocal / Braket emulator); and an AI-native layer: OpenQASM 3 as lingua franca, stable error codes + hints, seed-level determinism, JSON self-description and an MCP server. **Hardware-agnostic by design**: no vendor is the default target.

---

## 1. Qiskit (IBM) — the de facto standard for superconducting hardware

**IR / data model**
- `QuantumCircuit` is a Python facade over Rust `CircuitData`: a flat vector of `PackedInstruction` with interned qargs/cargs; a `StandardGate` enum avoids per-gate Python objects.
- Rust `DAGCircuit` (op/in/out nodes; wires for qubits, clbits, `expr.Var`, `Stretch`). Since 2.3 control flow (`IfElseOp`, `WhileLoopOp`, `ForLoopOp`, `SwitchCaseOp`, `BoxOp`) is Rust-native.
- `Target`: single hardware description `{op: {qargs: InstructionProperties(duration, error)}}` + `dt`; since 2.2 it bounds angles of fractional gates (RZZ on Heron).
- Rust symbolic engine for parameters (`SymbolExpr`). Serialisation via QPY v18 and OpenQASM 2/3.

**Compiler**
- `StagedPassManager`: init → layout → routing → translation → optimisation → scheduling; presets `generate_preset_pass_manager(level, target)` with plugins.
- Standout passes: **SabreLayout/SabreSwap (LightSABRE in Rust)**, VF2Layout (noise-aware subgraph isomorphism), HighLevelSynthesis plugins (Clifford, MCX, `PauliEvolutionSynthesisMcts` in 2.6), `TwoQubitPeepholeOptimization` (2.5, multithreaded), gridsynth (Ross–Selinger) for RZ in discrete bases, `LitinskiTransformation`, Clifford+T and PBC presets (2.5).
- `MultiStagePassManager` (2.5) lowers progressively across IRs; a **C API** with a pass manager (`QkPassManager`, 2.6) whose passes declare typed input/output IRs.

**Execution**
- **Primitives V2 with PUBs** (*Primitive Unified Blocs*): `(circuit, parameters[, shots])` for Sampler, `(circuit, observables, parameters[, precision])` for Estimator, with numpy-style broadcasting and shaped results. Modes: job / Batch / Session.
- 2026: "directed execution": `Executor` + `samplomatic` (box-annotated circuits → template + *samplex*; randomisation/twirling happens server-side). 0.50 deprecated the server-side Sampler/Estimator. In 0.52.0 (breaking) the top-level `Sampler`/`Estimator` are the client-side ones, the server-side ones moved to `qiskit_ibm_runtime.server_side`, and per-PUB shots or precision within one job were removed. 0.52 also adds a client-side `NoiseLearnerV3` and an experimental in-session `Calibrator`.

**Simulation**: Aer (C++; statevector, density matrix, MPS, stabilizer, cuStateVec) is in reduced maintenance (critical fixes only, per its README). GPU wheels returned in 0.18.0 (October 2026); 0.18.4 is a bug-fix release.

**QEC / experimentalist**: pulse programming left the SDK in 2.0 (replaced by fractional gates and dynamic circuits). On 2026-11-11 IBM announced pulse-level access on Nighthawk r2, listing mid-circuit measurement, fast reset, dynamic circuits and soft (IQ) readout (vendor announcement; we did not verify any SDK pulse API). `StabilizerState` simulates Clifford circuits; we did not find a Pauli-frame/DEM sampler or QEC decoders in the SDK. Strong fault-tolerant compilation (PBC, Litinski, Clifford+T). 2.6.2 is fixes only, among them `WrapAngles` ignoring angle bounds inside control flow.

**Algorithms**: `qiskit.algorithms` removed in 1.0; "Qiskit patterns" (map → optimize → execute → post-process) + add-ons (SQD, circuit cutting, OBP, MPF, AQC-Tensor). QDC 2026 (November) added packages for fermions, bosons (alpha), error mitigation, noise learning and spacetime Pauli checks.

**Weaknesses**: continuous churn (1.0, 2.0, V1→V2→Executor, unstable C ABI); Python overhead for Python passes and parameter binding; stagnating Aer; LLM corpus full of dead APIs (`execute`, `Aer.get_backend`, `IBMQ`).

**Best idea to steal → the PUB**: a declarative execution unit with broadcasting and shaped results that separates *what* to compute from *how* it is batched, randomised or mitigated. Runner-up: `Target` as the single source of truth for the compiler.

---

## 2. TKET / pytket / Guppy / HUGR (Quantinuum) — the algebraic compiler

**IR**
- **pytket** (C++ core): `Circuit` as a Boost DAG with typed edges (quantum, classical, boolean, WASM), global phase and implicit wire permutations; rich boxes (`PauliExpBox`, `PhasePolyBox`, `QControlBox`, `ConjugationBox`…).
- **HUGR**: hierarchical port graph with `Module/FuncDefn/DFG/CFG/Conditional/TailLoop/Call…` nodes, Value/Static/Order/ControlFlow/Hierarchy edges, **linear qubit types** (no-cloning checked at compile time), MLIR-dialect-like extensions, lowers to LLVM.
- **Guppy 1.0** (Aug 2026): Python-embedded language with linear types, `comptime`, `daggerable`/`controllable`.

**Compiler**
- Passes as objects with pre/post-conditions (`GateSetPredicate`, `ConnectivityPredicate`), composed with `SequencePass`/`RepeatWithMetricPass`, JSON-serialisable.
- Standouts: `FullPeepholeOptimise` (Clifford + KAK + 3q squash), **`GreedyPauliSimp`** (Pauli-graph synthesis), `ZXGraphlikeOptimisation`, `PauliSimp`, `CliffordSimp`, **`PhasePolyBox`/AAS routing** (architecture-aware phase-polynomial synthesis).
- Rust `tket`: **Badger** optimiser (multithreaded search over rewrite rules from equivalence classes, port matching) and, since 0.22.1, Pauli-graph resynthesis (`pauli_graph_resynthesis`).

**Execution**: `Backend` with `backend_info`, `required_predicates`, `default_compilation_pass(level)`, asynchronous `process_circuits` → serialisable `ResultHandle`. Nexus + Helios (accepts HUGR and full Adaptive QIR; classical code runs per shot as LLVM).

**Simulation**: delegated to extensions; **Selene** (Rust, QuEST/Stim plugins) faithfully emulates real-time control.

**Weaknesses**: two parallel stacks (pytket vs. HUGR/Guppy) with partial parity; heavy churn (`tket` (Rust) 0.23.0 is breaking and moves to HUGR 0.31; guppyft 0.6.0 and 0.7.0 are breaking, the latter requiring an explicit QED policy in its Iceberg builder); in Benchpress (IBM-authored, 2024 versions) pytket 1.31.0 failed 86 of 1,032 tests and was much slower than Qiskit; Guppy has little LLM training data (but its compile errors suit repair loops). Since the last review, tket 0.23.x fixed qubit order in controlled-Toffoli expansion and global phase in modifier resolution and added `power` modifiers; guppylang 1.2.1 fixed linearity checks under dagger and control.

**Best idea to steal → linear, hierarchical IR (HUGR)** and the **phase-polynomial / Pauli-graph optimisation family** (phase folding, GreedyPauliSimp), the biggest reducers of T-count and CNOTs.

---

## 3. NVIDIA CUDA-Q — the heterogeneous platform

**IR**: single-source kernels (`@cudaq.kernel` / `__qpu__`) lowered to MLIR: **Quake** dialect (reference semantics `!quake.ref/veq` and value semantics `!quake.wire`) + **CC** (classical control). 0.15 adds `detector`, `logical_observable` and `dem_from_kernel` in the language. General operator algebra (spin, fermion, boson).

**Compiler**: `nvq++` and the Python bridge → Quake; passes `apply-op-specialization` (automatic adjoint/controlled), `memtoreg`, unrolling, decomposition, mapping; **`quake-synth`** specialises the kernel with runtime arguments (JIT); output QIR (full / base / adaptive) or OpenQASM 2. 0.16: static Clifford+T synthesis, "atomic quantum regions". 0.17: `optimization_level=0..3` and borrow checking on the value-semantics IR; breaking changes include the reworked Quake operator class and the removal of `quantum_platform::get_num_qubits`/`connectivity`; `from_qiskit`/`from_qasm` conversion in `cudaq.contrib` is deprecated.

**Execution**
- Target by string (`cudaq.set_target("nvidia", option="fp64,mgpu")`), same kernel on `qpp-cpu`, `nvidia`, `tensornet(-mps)`, `dynamics`, `stim`, `custabilizer` (0.17, GPU stabilizer sampling) or QPUs (Quantinuum, IonQ, IQM, Pasqal, QuEra, Braket…).
- `sample` / `observe` / `get_state` / `run` (+ `_async` variants with futures). **Multi-QPU**: `mqpu` creates one virtual QPU per GPU (`qpu_id=i`), `observe(..., execution=cudaq.parallel.thread|mpi)` splits Hamiltonian terms.
- **NVQLink / `device_call`**: classical calls (CPU/GPU/FPGA) from the kernel, lowered into the IR with a latency contract (authors report a 3.96 µs maximum round-trip latency).

**Simulation**: cuStateVec (fp32 default, fusion up to 10 qubits, GPU→host migration, MPI/NVLink distribution from 25 qubits), cuTensorNet, MPS, cuDensityMat; **PTSBE** (pre-sampled Kraus trajectories with shot batching for training AI decoders).

**QEC**: CUDA-Q QEC 0.9: codes (surface, colour…); decoders `nv-qldpc-decoder` (BP+OSD, GPU Relay-BP; closed source) and `nv-fusion-decoder` (parallel fused MWPM, since 0.8.0; closed source), `trt_decoder` (ONNX/TensorRT for AI decoders), PyMatching, sliding window, tensor network; **real-time decoding** inside kernels (`enqueue_syndromes`/`get_corrections`/`get_logical_measurements`). 0.9.0 adds a Decoding Server, decoding task graphs (window, parallel window, pre-decoder, ensemble) over dynamic or chunked DEMs, and in `nv-qldpc-decoder` a learned predecoder (TCBP), GPU OSD beam search (JOBS) and noise-rate learning from detector data; it requires CUDA-Q 0.17 and breaks the decoder plugin ABI. 0.9.1 speeds up batched Relay BP (vendor-reported 3.5-5.6x on two NVIDIA GPUs). CUDA-Q 0.17 also ships `cudaq-edem` (extended DEMs; its format is declared unstable).

**Algorithms**: functors `cudaq.control/adjoint`, `compute_action` (within/apply); no autodiff (finite-difference/parameter-shift gradients); CUDA-Q Solvers deprecated → **CUDA-Q Algorithms 0.2** (LCU, block encoding, qubitisation, QSVT, Trotter; 0.2 adds unary iteration, QROM, reversible arithmetic, alias sampling and phase estimation over block encodings).

**Weaknesses**: churn (spin_op, `sample` vs `run`, rewritten bridge, LLVM 16→22); "kernel Python" is not Python (no dynamic lists, limited closures) so LLMs produce invalid kernels; GPU only on Linux/NVIDIA; fragmented docs.

**Best idea to steal → the QPU as just another device** in a compiled heterogeneous program: one kernel, target by string, asynchronous multi-QPU execution and real-time classical calls as part of the IR.

---

## 4. PennyLane (+ Catalyst) — the differentiable framework

**IR**: QNode → tapes (`QuantumScript`) with operations and measurement processes; "capture" path (plxpr = JAXPR + quantum primitives) to Catalyst (default for `qjit` on the unreleased Catalyst 0.16 branch). Official alias `import pennylane as qp` (2026).

**Compiler**: `@qp.transform` transforms and `qp.CompilePipeline` with markers; `cancel_inverses`, `merge_rotations`, graph-based decomposition to a gate set; **`qp.specs`/`qp.draw(level=…)` show any stage**. Catalyst (`qp.qjit`): MLIR (`quantum`, `gradient`, `mitigation`, PBC/QEC dialects) + StableHLO + Enzyme; MLIR passes writable in Python (xDSL).

**Execution**: Device API v2 (only `execute(circuits, config)` required; preprocessing and derivative hooks) with **capabilities declared in TOML** that automatically generate the preprocessing pipeline. Mid-circuit measurement: `deferred`, `one-shot`, `tree-traversal`. 0.46 beta: **Backline** (experimental; controller + GPU/FPGA coprocessors, Triton decoders; the development changelog also lists ONNX decoders).

**Simulation**: Lightning family (C++ SIMD, Kokkos, cuStateVec, tensor); **adjoint differentiation** is the variational workhorse. The unreleased 0.46 branch removes `lightning.tensor`.

**Algorithms**: `diff_method` ∈ {backprop, adjoint, parameter-shift, finite-diff, spsa, hadamard, device}; JAX/Torch interfaces; operator arithmetic (`adjoint`, `ctrl`, `pow`); huge template library; FT resource estimator.

**Weaknesses**: per-tape Python overhead; two semantic worlds (tape vs. qjit); constant deprecations (TF, device shots, `qml`→`qp`) that poison the LLM corpus. No new stable release since the last review. The 0.46 development changelog (unreleased) removes `pennylane.noise` (including `NoiseModel` and the ZNE transforms; noise channels stay), `pulse`, `kernels` and `qnn`.

**Best idea to steal → differentiation as a first-class citizen (adjoint + parameter-shift)** and **the compilation pipeline as an inspectable object** generated from declared capabilities.

---

## 5. Cirq (Google Quantum AI) — stability and QEC

**IR**: `Circuit` = list of **`Moment`s** (operations on disjoint qubits → explicit scheduling); `FrozenCircuit`/`CircuitOperation` (repetitions, maps); sympy parameters + sweeps; insertion strategies.

**Protocols**: duck typing (`_unitary_`, `_decompose_`, `_kraus_`, `_act_on_`, `_has_stabilizer_effect_`, `_json_dict_`) with fallback chains: any object joins simulation and compilation without inheritance.

**Compiler**: transformers as pure functions (`@cirq.transformer`), `optimize_for_target_gateset` (CZ, √iSWAP), `eject_z`, `merge_k_qubit_unitaries`, dynamical decoupling, `RouteCQC`. No pass manager and no real Target.

**Execution**: `Sampler.run/run_sweep/run_batch`; Google Engine; **quantum virtual machine** with Willow/Weber noise.

**Simulation**: numpy simulators (slow beyond 20 qubits); **qsim** (C++ AVX/OpenMP, fusion, cuStateVec; on Apple Silicon only a basic non-vectorised build).

**QEC**: Cirq is a common gateway to **Stim** (bit-packed Pauli-frame simulation, `DETECTOR`/`OBSERVABLE_INCLUDE`, **DEM** as the exchange format), **PyMatching 2.4** (sparse blossom, ~1M errors/core·s, authors' figure; correlated matching; not a Google project), **Sinter** (reproducible statistics), Chromobius, Tesseract; Willow: Λ = 2.14 ± 0.02 (error suppression per increase of the distance by 2, up to d=7) and real-time decoding (63 µs mean latency at d=5). Characterisation (XEB, RB, T1, confusion matrices).

**Weaknesses**: one Python object per operation; slow sympy; weak routing; slow release cadence. No release since 1.7.0 (June 2026). Unreleased on `main` (1.8.dev): minimum Python 3.12; simulators move to `np.random.Generator`, which can change seeded outputs; `cirq_google` `Engine.run` is deprecated.

**Best idea to steal → the DEM as the narrow waist** (any circuit generator → DEM → any decoder) and Stim's **bit-packed Pauli-frame simulation**; from Cirq itself: API stability and extensible protocols.

---

## 6. Q# / Microsoft QDK — the language with the strongest guarantees

**IR**: Rust pipeline `parse → HIR → FIR → … → partial eval → RIR → QIR`, with FIR transforms (monomorphisation, defunctionalisation and others) and RCA before code generation; also compiles to WASM (playground, VS Code, linter, debugger).

**Compiler**: **RCA (Runtime Capabilities Analysis)** + **partial evaluation** against a declared **target profile** (`Base`, `Adaptive_RI`, `Adaptive_RIF`, `Adaptive` (the former `Adaptive_RIFLA`), `Unrestricted`): the user (or the LLM) gets "this construct needs Adaptive_RIF" *at compile time*, not a hardware rejection after hours in a queue. Changes merged before 1.34.0 let some programs rejected by the full Adaptive path fall back to `Adaptive_RIF` with a warning, and let RCA accept more recursion.

**Language**: `operation` vs `function`; **automatically generated functors** (`is Adj + Ctl` → `Adjoint Op`, `Controlled Op`); conjugation `within { U } apply { V }`; scoped qubits (`use q = Qubit()`), which must be released in |0⟩.

**Execution / simulation**: `qdk` package (replaces `qsharp`); sparse simulator, stabilizer (thousands of qubits, with branching for non-Clifford gates); `run_qir` offers stabilizer, CPU state vector and **GPU via wgpu/WebGPU** (cross-vendor); a density-matrix simulator is a separate class. The noiseless full-state simulator was removed in 1.32. The 1.34.0 release notes list bug fixes and performance in QIR compilation.

**QEC / resources**: **resource estimator** (QRE v3: `Application`, `Architecture`, `ISA`; T factories, Pareto frontiers); **`qdk.ec`** (since 1.33) for authoring and auditing QEC schemes, and an experimental `run_qir(qodec=, decoder=)` error-corrected pipeline.

**Weaknesses**: niche language (small LLM corpus); small local simulators; packaging churn; no autodiff.

**Best idea to steal → automatic functors (Adjoint/Controlled) + compile-time capability analysis + resource estimation before execution.**

---

## 7. Amazon Braket SDK — devices as data and task economics

**IR**: **OpenQASM 3 as the wire format** (`#pragma braket result …`, `verbatim` boxes), `FreeParameter` → `input float`. Pydantic device-property schemas (native gates, connectivity, calibration, pulse frames). AHS (analog Hamiltonian simulation) with `AtomArrangement`/`DrivingField`.

**Compiler**: deliberately thin (vendor compilers do the work); the **local emulator** (`AwsDevice(arn).emulator()`) is for **verbatim** circuits: it validates them against calibration data and simulates them on a density matrix with per-gate depolarising noise and readout error.

**Execution**: durable asynchronous tasks (`AwsQuantumTask`, results in S3); **ProgramSets** (`product()`/`zip()` of circuits × parameters × observables in one task → amortises the $0.30/task fee and queue latency); Hybrid Jobs with queue priority; cost `Tracker` (an estimate; Braket also has service-side spending limits).

**Simulation**: managed SV1 (34 qubits), DM1 (TN1 retired in SDK 1.121); modest local Python simulator.

**Weaknesses**: per-task fee + queue latency punish fine-grained loops; slow Hybrid Jobs container start-up (not re-verified); experimental dynamic circuits (IQM only, verbatim; mid-circuit outcomes are not returned). Since the last review (SDK 1.127.8 to 1.128.2): noise-model and emulator fixes, `AwsDevice.status_reason`, Python 3.14/3.15. A 2026-11-09 [security bulletin](https://aws.amazon.com/security/security-bulletins/2026-175-aws/) covers the SDK before 1.127.7 and an older schemas package. Braket notebook instances now ship an agent skill and MCP servers.

**Best idea to steal → an emulator built from the device's own data** (validation + cost before spending) and **ProgramSet batching** at the task boundary.

---

## 8. Qibo (TII) — from simulation to your own lab

**IR**: `Circuit` with a flat gate queue and lazy moments; **noise channels are gates** (`DepolarizingChannel`, `ThermalRelaxationChannel`, `ReadoutErrorChannel`, Kraus); measurement with collapse mid-circuit.

**Compiler**: through 0.3.6, a transpiler with placers (ReverseTraversal, Subgraph, StarConnectivity, Random), ShortestPaths/Sabre routers and `Unroller(NativeGates)`; 0.3.6 added an optimizer set (`InverseCancellation`, `ConsolidateBlocks`, measurement-aware cleanups). **0.3.7** adds a `transpile()` entry point with `optimization_level` 0-2, SABRE trials, directed connectivity, quantum Shannon decomposition, a refactored KAK, commutative cancellation, an early validation pass, and algorithm models (QPE, amplitude amplification and estimation, quantum counting, adders). Gate fusion for simulation. We ran no quality or speed comparison.

**Execution**: swappable backends (`set_backend`: numpy, qibojit numba/cupy/cuQuantum, qibotn, qiboml, qibolab) — one change moves from GPU simulation to **your own hardware**. **Qibolab** models the platform (IQ/DC/acquisition channels, Qblox/QM/Keysight/RFSoC instruments) with pulse sequences and hardware-loop *sweepers*.

**QEC / experimentalist**: **Qibocal**: acquire → fit → report → update protocols (spectroscopy, Rabi, Ramsey, T1/T2, DRAG, single-shot classification, RB, chevron, CZ…) declared in YAML runcards that **write calibrated parameters back into the platform**. Mitigation (ZNE, CDR, readout). No detectors/decoders.

**Weaknesses**: global mutable state (backend/transpiler) that hurts concurrency; small team; per-gate Python overhead. Qibolab 0.2.18 is mostly Qblox driver work; Qibocal has had no release since 0.2.6.

**Best idea to steal → calibration as code with a closed loop**: one platform model shared by simulation, transpiler, pulses and calibration.

---

## 9. Comparative matrix by area

Scale: ★ basic · ★★ solid · ★★★ industry reference. Every rating is QUMO's judgement from documented features, not a measurement. The last two rows are the most subjective: no study cited here ranks frameworks on them.

One rating changed in this revision: Qibo circuit optimisation, ★ to ★★, because 0.3.7 adds preset optimisation levels, commutative cancellation, quantum Shannon decomposition and a reworked KAK (feature coverage only; output quality not measured). PennyLane's removal of `pulse` is unreleased, so its experimentalist rating stays until a release.

| Area / capability | Qiskit | TKET | CUDA-Q | PennyLane | Cirq | Q# | Braket | Qibo |
|---|---|---|---|---|---|---|---|---|
| **Algorithms: composition** (functors, templates) | ★★ | ★★ | ★★★ | ★★★ | ★★ | ★★★ | ★ | ★★ |
| Automatic differentiation | ★ | – | ★ | ★★★ | ★ | – | – | ★★ |
| **Heterogeneous orchestration** (async, multi-QPU, HPC) | ★★ | ★★ | ★★★ | ★★ | ★ | ★ | ★★ | ★ |
| **IR / representation** | ★★★ | ★★★ | ★★★ | ★★ | ★★ | ★★★ | ★★ | ★ |
| **Circuit optimisation** | ★★★ | ★★★ | ★★ | ★★ | ★★ | ★★ | ★ | ★★ |
| Routing / mapping | ★★★ | ★★★ | ★★ | ★ | ★ | ★ | – | ★★ |
| **Hardware abstraction** (target as data) | ★★★ | ★★ | ★★ | ★★★ | ★★ | ★★ | ★★★ | ★★★ |
| Runtime protocols (QIR, OpenQASM 3, real time) | ★★ | ★★★ | ★★★ | ★★ | ★ | ★★★ | ★★ | ★★ |
| Simulation performance (CPU/GPU) | ★★ | ★ | ★★★ | ★★★ | ★★ | ★★ | ★★ | ★★ |
| **QEC: simulation + DEM + decoders** | ★ | ★ | ★★★ | ★★ | ★★★ (Stim/PyMatching) | ★★ | – | ★ |
| **Experimentalist** (calibration, pulses, characterisation) | ★★ | ★ | ★★ | ★★ | ★★ | ★ | ★★ | ★★★ |
| API stability (AI) (QUMO's judgement) | ★ | ★ | ★ | ★ | ★★★ | ★★ | ★★ | ★★ |
| Diagnostics for agents (QUMO's judgement) | ★★ | ★★ | ★ | ★★ | ★★ | ★★★ | ★★ | ★ |

---

## 10. Cross-cutting patterns (what the industry is learning)

1. **Native cores, thin facades.** Qiskit (Rust `CircuitData`/`DAGCircuit`), TKET (HUGR/Rust), Selene (Rust), Q# (Rust). The new frontier is **typed multi-IR pipelines** (`MultiStagePassManager`, `QkIrHandle`, HUGR extensions, MLIR dialects).
2. **Declarative execution + server-side randomisation** (IBM Executor/samplex; per-shot compiled classical code on Helios).
3. **Capabilities as data + early validation** (PennyLane TOML, Braket properties, QIR module flags, Q# RCA; Qibo 0.3.7 adds a validation pass).
4. **Real time as an IR-level call with a latency contract** (CUDA-Q `device_call`, Backline `runtime_call`), not RPC.
5. **QEC converges on DEM + swappable decoder**. Neural decoding is advancing both as a decoder in its own right (AlphaQubit 2: authors report under 1 µs per cycle on commercial accelerators, surface code up to d=11) and as a pre-decoder in front of a classical one (CUDA-Q QEC's Ising + Chromobius colour-code demo). CUDA-Q QEC 0.9 composes decoders as task graphs.
6. **AI**: MCP servers (official Qiskit, qBraid, Coda), `llms.txt` (Braket), and now **agent skills** shipped by vendors ([Qiskit/skills](https://github.com/Qiskit/skills), AWS's `amazon-braket` skill in Braket notebooks, CUDA-Q Algorithms 0.2, tsim 1.1). IBM discontinued Qiskit Code Assistant. What QUMO bets helps agents most: **an executable verifier, stability, precise diagnostics, current documentation**; §0.3 says which of these the studies measure.

---

## 11. Synthesis: what QUMO takes from each and how it combines them

### 11.1 Algorithms (composition + heterogeneous orchestration)

| Idea | Origin | In QUMO |
|---|---|---|
| Kernels = functions returning composable circuits | CUDA-Q kernels, Q# operations, PennyLane templates, Qualtran bloqs | `qumo.algorithms.library` (qft, grover, qpe, ansatz, trotter…) |
| Automatic Adjoint / Controlled / Power functors | Q#, CUDA-Q `apply-op-specialization`, OpenQASM `inv@ ctrl@ pow@` | `Circuit.inverse()`, `.controlled(q)`, `.power(k)` |
| Adjoint differentiation + parameter shift + finite differences | PennyLane / Lightning | `qumo.algorithms.gradients` |
| Cheap symbolic parameters | Qiskit `Parameter`, Braket `FreeParameter` | 24-byte affine `Param` (allocation-free binding) |
| PUB with parameter × observable broadcasting | Qiskit Primitives V2 | `qumo.exec.primitives` (Sampler/Estimator) |
| Batching at the task boundary | Braket ProgramSet, CUDA-Q `observe(execution=…)` | `qumo.exec.orchestrator` |
| Asynchronous multi-device execution | CUDA-Q `mqpu`, `*_async` | HEFT orchestrator over a CPU/GPU/QPU device pool |

### 11.2 Quantum code (circuits + optimisation)

| Idea | Origin | In QUMO |
|---|---|---|
| Flat, cache-friendly IR without per-gate objects | Qiskit `CircuitData` | `Circuit.ops: List[Instruction]` POD |
| Explicit global phase | Qiskit, pytket | `Circuit.global_phase` |
| Measurement records + DETECTOR/OBSERVABLE + noise channels in one IR | Stim, Qibo (noise as gates), CUDA-Q 0.15 (`detector`) | noise/annotation opcodes in `qumo.ir.gates` |
| Gate metadata as data (Clifford, diagonal, inverse…) | Cirq protocols, Qiskit `StandardGate` | `op_is_clifford`, `op_inverse_kind`, … + JSON `describe()` |
| Inspectable, serialisable pass pipeline | PennyLane `CompilePipeline`/`specs(level)`, pytket JSON passes | text spec `"cancel,merge,route(sabre),basis(...)"` + per-pass report |
| SABRE routing with lookahead | Qiskit LightSABRE | `qumo.compiler` route(sabre) |
| Phase folding / phase polynomials | TKET PhasePolyBox, PyZX | `qumo.compiler` phase_fold |
| Virtual-Z / Pauli pushing | Cirq `eject_z`, `eject_phased_paulis` | `qumo.compiler` |
| 1q fusion and ZYZ resynthesis | Qiskit `Optimize1qGatesDecomposition`, Cirq `merge_single_qubit_gates` | fuse1q |
| `Target` as single source of truth | Qiskit Target, Braket device properties | `qumo.device.target.Target` (modality presets, none default) |
| Capability errors at compile time | Q# RCA | validation with `QM-E…` codes before execution |

### 11.3 Quantum execution (hardware abstraction + protocols)

| Idea | Origin | In QUMO |
|---|---|---|
| Same program, swappable target | CUDA-Q `set_target`, Qibo `set_backend` (but **no global state**) | explicit backend per call + explained automatic selection |
| Simulator choice by circuit structure | QDK (Clifford → stabilizer) | auto-dispatch: Clifford → tableau/frame, else statevector |
| Static sampling vs. trajectories | CUDA-Q `sample` vs `run`, PennyLane mcm_method | `qumo.sim.runner.sample` decides and reports `method` |
| OpenQASM 3 as wire format | Braket, IBM | `qumo.ir.qasm` (parser + emitter) |
| Emulator from device calibration | Braket emulator, Cirq virtual engine | `NoiseModel.from_target` + `emulate` (`qumo.noise.emulate`) |
| Resource estimation before running | Q# QRE, PennyLane `specs`, Braket `Tracker` | `qumo.compiler.resources` (logical, NISQ hardware, fault-tolerant) |

One more row belongs to execution on analog neutral atoms: the program model `qumo.ir.analog` (Pulser convention) has a global drive only and one export, Pulser, in the hardware plugins ([§14](#14-additional-references-tsim-and-bloqade-analog)).

### 11.4 Error correction (experimentalist)

| Idea | Origin | In QUMO |
|---|---|---|
| Bit-packed Pauli-frame simulation (64 shots/word) | Stim | `qumo.sim.frame` |
| Aaronson–Gottesman tableau | Stim, Qiskit Aer, QDK | `qumo.sim.stabilizer` |
| DEM as the narrow waist | Stim, CUDA-Q QEC `dem_from_kernel` | `qumo.qec.dem` (Stim-compatible text) |
| Swappable decoders (UF, MWPM, lookup, AI) | PyMatching, CUDA-Q QEC, AlphaQubit | `qumo.qec.decoders` (common interface) |
| Reproducible statistics harness | Sinter | `qumo.qec.experiments` (threshold sweeps) |
| Calibration as code / noise from T1, T2, errors | Qibocal, Braket emulator, Qiskit `NoiseModel.from_backend` | `qumo.noise.characterization` (NoiseModel, RB, readout mitigation) |
| Active-width near-Clifford sampling | tsim, Clifft, SymFT | not in QUMO: `qumo.sim.branching` is shot-by-shot stabilizer branching ([§14](#14-additional-references-tsim-and-bloqade-analog)) |

---

## 12. Why Mojo?

| Problem shared by all 8 frameworks | How Mojo solves it |
|---|---|
| Slow Python facade + C++/Rust core (two languages, FFI, builds) | One language: Python-like syntax, compiled natively, no internal FFI |
| Per-gate/per-object overhead | POD structs in contiguous lists; no GIL, no boxing |
| SIMD/threads need C++/OpenMP | Native `SIMD[DType.float64, 8]`, `vectorize`, `parallelize` |
| GPUs need separate CUDA/HIP/wgpu code | GPU kernels in Mojo via MAX (NVIDIA/AMD/Apple) — same code base (Apple measured in fp32 only, [GPU.md](GPU.md)) |
| Compile-time specialisation (quake-synth, Catalyst) | Native `comptime` / compile-time parameters |
| Lowering to MLIR/QIR | Mojo compiles through MLIR (we have not verified a supported way to add custom dialects) |
| LLMs and Python | Familiar syntax for models; QUMO additionally speaks OpenQASM 3 and JSON |

**Toolchain status:** Mojo 1.2.0 and MAX 26.7.0 (2026-10-22), QUMO's pins, are still the latest stable releases. The Mojo compiler and tools have been open source (Apache 2.0 with LLVM exceptions) since [2026-08-18](https://www.modular.com/blog/mojo-open-source). Since MAX 26.7 the MAX source is available under Modular's own [MAX License](https://www.modular.com/blog/26-7), which is not Apache 2.0. Qualcomm [completed its acquisition of Modular](https://www.modular.com/blog/qualcomm-completes-acquisition-of-modular) (announced 2026-07-29); the announcement says Mojo and MAX continue as products and brands.

**Mojo risks** (measured during this project, except the last two): the language still changes quickly (1.0→1.1 removed `fn`, `alias`, `@parameter for`, renamed `UnsafePointer`; `parallelize` and GPU moved to `max.*`); no convenient dynamic trait objects (the [roadmap](https://mojolang.org/docs/roadmap/) lists existentials as not started; QUMO dispatches passes and backends by ID); small ecosystem. The unreleased [1.3 nightly](https://mojolang.org/releases/nightly/) removes APIs deprecated up to 1.1, including the `UnsafePointer` alias, and makes `String.find()`/`rfind()` return `Optional[Int]` instead of -1; QUMO has not been built against it. The toolchain's owner now also sells chips, which a hardware-neutral project should watch (QUMO's judgement; no change in neutrality has been observed). Mitigation: `docs/MOJO_GUIDE.md` pins the exact dialect and per-module tests guard it.

---

## 13. QUMO AI-native principles

1. **Formats models already know**: OpenQASM 3 in/out; Paulis as `"0.5*Z0 Z1 + 0.2*X0"` or `"ZZI"`; pipelines as text.
2. **Stable diagnostics**: every error carries `QM-Exxx`, a message, a location and an actionable `Hint:` (catalogue in `docs/ERRORS.md`).
3. **Determinism**: all randomness hangs off an explicit seed → bit-identical, cacheable, verifiable reruns.
4. **Self-description**: gates, errors, capabilities, passes and backends as JSON (`qumo describe`, `llms.txt`).
5. **Executable verifier in the loop**: `qumo validate`, resource estimation and local simulation before spending on hardware.
6. **Native MCP**: the same binary is an MCP server (`qumo mcp`) with tools to run, validate, compile, estimate and do QEC.
7. **Explainability**: every automatic decision (simulation method, backend, passes) is reported in the result.
8. **Stability**: small orthogonal API; names aligned with OpenQASM/Qiskit to reuse model knowledge.
9. **Hardware neutrality**: targets are data; presets are named by modality, never by vendor, and none is the default.

---

## 14. Additional references: tsim and bloqade-analog

Two QuEra repositories, reviewed as research input. They are not part of the eight frameworks and have no column in §9. Everything proposed here is **not implemented**. Details, options and the validation plan are in [TSIM_BLOQADE_PLAN.md](TSIM_BLOQADE_PLAN.md).

**[tsim](https://github.com/QuEraComputing/tsim)** (`bloqade-tsim` 1.1.0, 2026-11-12; Apache-2.0, with parts under other licences) samples noisy Clifford + non-Clifford circuits for QEC. It mirrors Stim's API and extends Stim's text format with `T`, rotations and Pauli-product rotations. It turns a circuit into a ZX diagram with noise as boolean parameters and reduces it once. Since 1.0.0 (2026-11-04) the default, `active_statevector`, runs each remaining component on a dense state vector over its *active* qubits (following Clifft and SymFT): cost grows as 2^w in the active width w, with w ≤ 30 (≤ 11 on its Apple GPU backend, in float32). The older stabilizer-rank strategies, the main subject of the [April paper](https://arxiv.org/abs/2604.01059), are now provisional and may be removed. Its `detector_error_model()` comes from Stim on a Clifford stand-in and is only guaranteed for Clifford circuits. Its speed figures are vendor-reported and we did not run it.

**QUMO today:** near-Clifford sampling is `qumo.sim.branching` (a stabilizer frame plus sparse amplitudes, redone for every shot; ≤ 512 qubits, ≤ 2^20 branches, ≤ 62 classical bits, no detector sampling). Detector sampling (`qumo.sim.frame`) and the DEM are Clifford only. `from_stim` does not import `T` or tagged gates. There is no ZX engine.

**[bloqade-analog](https://github.com/QuEraComputing/bloqade-analog)** (0.16.9, 2026-03-02; Apache-2.0) is a Python SDK for analog neutral-atom programs: a builder, an IR, a local emulator (SciPy ODE solvers, optional blockade subspace, two- or three-level atoms) and submission to QuEra Aquila through Amazon Braket. Activity since the 0.16.9 release looks like maintenance (our reading of the commit log). Release 0.16.9 builds local detuning with `ShiftingField`, which the Braket SDK marks as deprecated in favour of `LocalDetuning`.

**QUMO today:** `qumo.ir.analog` models a register with one global drive; `qumo.sim.rydberg` is a noiseless Trotter simulator for ≤ 20 atoms; the only vendor format is the Pulser export in the hardware plugins. There is no local detuning and no Braket AHS format.

**Proposed order** (details in the plan): for tsim, first import the `T` gates and sample raw detectors with the existing branching simulator, checked against an independent exact distribution; before any compile-once sampler, measure the active width QUMO could reach. For analog, first a plugin-side AHS JSON export, then native local detuning. The core stays vendor-neutral: AHS lives in `plugins/hardware`, and nothing under `qumo/` imports a vendor module.

---

### Main sources
Qiskit 2.5/2.6 release notes, [QDC 2026 developer blog](https://www.ibm.com/quantum/blog/dev-tools-qdc-2026) and [QDC 2026 blog](https://www.ibm.com/quantum/blog/qdc-2026); qiskit-ibm-runtime 0.50 and [0.52.0](https://github.com/Qiskit/qiskit-ibm-runtime/releases/tag/0.52.0); Benchpress (arXiv:2409.08844); HUGR (arXiv:2510.11420), tket/guppylang/Selene (Quantinuum GitHub); CUDA-Q 0.16 docs, [CUDA-Q 0.17.0](https://github.com/NVIDIA/cuda-quantum/releases/tag/0.17.0), [CUDA-Q QEC releases](https://github.com/NVIDIA/cudaq-qec/releases), NVQLink (arXiv:2510.25213); PennyLane 0.45 / Catalyst docs, Backline (arXiv:2609.09270); Cirq 1.7, qsim, Stim 1.16, PyMatching (sparse blossom, arXiv:2303.15933), Willow (arXiv:2408.13687), AlphaQubit 2 (arXiv:2512.07737), Relay-BP (arXiv:2506.01779); QDK releases and target profiles (learn.microsoft.com); Braket program sets, emulator and pricing (AWS); Qibo/Qibolab/Qibocal (qiboteam GitHub); QIR Adaptive Profile (qir-alliance); OpenQASM 3.1; ORBIT-Q (arXiv:2607.03105), PennySynth (arXiv:2605.25572), QuanBench (arXiv:2510.16779), QuanBench+ (arXiv:2604.08570), arXiv:2603.22184, arXiv:2607.04072, arXiv:2610.28678; PennyLane, Catalyst and Lightning development changelogs; Qibo, tket, guppyft and CUDA-Q Algorithms release notes; PyPI and crates.io for versions; tsim (arXiv:2604.01059, repository, [1.0 blog](https://bloqade.quera.com/blog/introducing-tsim-1-0-0/)) and bloqade-analog (repository).
