# Deep analysis: the 8 reference quantum frameworks (October 2026)

> Goal: extract the best of **Qiskit, TKET, CUDA-Q, PennyLane, Cirq, Q#/QDK, Amazon Braket SDK and Qibo**, decide how to combine it, and rebuild it in **Mojo** as one stack that is efficient (speed and resource use) and **AI-native**.
>
> Versions reviewed (2026-10-12): Qiskit 2.5.2 / 2.6.0rc1, qiskit-ibm-runtime 0.50, Aer 0.17.2 · pytket 2.18.5, Guppy 1.1.3, HUGR 0.30, tket (Rust) 0.22 · CUDA-Q 0.16.0, CUDA-Q QEC 0.8.0, cuQuantum 26.06 · PennyLane 0.45.1 (0.46 beta), Catalyst 0.15 · Cirq 1.7.0, qsim 0.22 · QDK 1.33.0 (`microsoft/qdk`) · Braket SDK 1.127.3 · Qibo 0.3.5, Qibolab 0.2.17, Qibocal 0.2.6 · Stim/Sinter 1.16, PyMatching 2.4.
>
> Performance figures quoted are vendor-reported unless stated otherwise (e.g. Benchpress is authored by IBM — see [BENCHPRESS_BIAS_REPORT.md](BENCHPRESS_BIAS_REPORT.md)).

---

## 0. Executive summary

1. **No framework wins all four areas.** Qiskit leads compilation for real hardware; TKET leads algebraic optimisation (ZX, phase polynomials, Pauli graphs) and IR design (HUGR); CUDA-Q leads heterogeneous execution and GPUs; PennyLane leads automatic differentiation and inspectable pipelines; Q# leads functors and compile-time capability analysis; Braket leads device-as-data abstraction and task batching; Cirq leads API stability and gives access to Google's QEC ecosystem (Stim, PyMatching); Qibo leads "calibration as code" for labs.
2. **All of them suffer the same language split**: a slow Python front end over a C++/Rust/MLIR core. Consequences: per-gate Python overhead, two semantic worlds (tape vs. qjit in PennyLane, kernel-Python vs. Python in CUDA-Q, pytket vs. HUGR in TKET), complex builds, and constantly changing APIs.
3. **API churn is the largest tax on AI.** 2026 studies (ORBIT-Q, PennySynth, QuanBench) agree: agents succeed when there is (a) an executable verifier loop, (b) a stable API, (c) current, retrievable documentation. Qiskit has the largest but most stale corpus; Cirq the most stable.
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
- `MultiStagePassManager` (2.5) lowers progressively across IRs; a **C API** with a pass manager (`QkPassManager`, 2.6rc1) whose passes declare typed input/output IRs.

**Execution**
- **Primitives V2 with PUBs** (*Primitive Unified Blocs*): `(circuit, parameters[, shots])` for Sampler, `(circuit, observables, parameters[, precision])` for Estimator, with numpy-style broadcasting and shaped results. Modes: job / Batch / Session.
- 2026: "directed execution": `Executor` + `samplomatic` (box-annotated circuits → template + *samplex*; randomisation/twirling happens server-side). In 0.50 the server-side Sampler/Estimator are deprecated in favour of client-side versions built on Executor.

**Simulation**: Aer (C++; statevector, density matrix, MPS, stabilizer, cuStateVec) is in maintenance mode (no release for 13 months; no GPU wheels since 2024).

**QEC / experimentalist**: no pulse access since 2.0 (replaced by fractional gates and dynamic circuits); no in-SDK stabilizer simulator or decoders (IBM research uses Stim + Relay-BP). Strong fault-tolerant compilation (PBC, Litinski, Clifford+T).

**Algorithms**: `qiskit.algorithms` removed in 1.0; "Qiskit patterns" (map → optimize → execute → post-process) + add-ons (SQD, circuit cutting, OBP, MPF, AQC-Tensor).

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
- Rust `tket`: **Badger** optimiser (multithreaded search over rewrite rules from equivalence classes, port matching).

**Execution**: `Backend` with `backend_info`, `required_predicates`, `default_compilation_pass(level)`, asynchronous `process_circuits` → serialisable `ResultHandle`. Nexus + Helios (accepts HUGR and full Adaptive QIR; classical code runs per shot as LLVM).

**Simulation**: delegated to extensions; **Selene** (Rust, QuEST/Stim plugins) faithfully emulates real-time control.

**Weaknesses**: two parallel stacks (pytket vs. HUGR/Guppy) with partial parity; heavy churn (Guppy 1.0, `tket` 0.15→0.22); in Benchpress (IBM-authored) pytket 1.31 had 87 failures and was much slower; Guppy has little LLM training data (but its compile errors suit repair loops).

**Best idea to steal → linear, hierarchical IR (HUGR)** and the **phase-polynomial / Pauli-graph optimisation family** (phase folding, GreedyPauliSimp), the biggest reducers of T-count and CNOTs.

---

## 3. NVIDIA CUDA-Q — the heterogeneous platform

**IR**: single-source kernels (`@cudaq.kernel` / `__qpu__`) lowered to MLIR: **Quake** dialect (reference semantics `!quake.ref/veq` and value semantics `!quake.wire`) + **CC** (classical control). 0.15 adds `detector`, `logical_observable` and `dem_from_kernel` in the language. General operator algebra (spin, fermion, boson).

**Compiler**: `nvq++` and the Python bridge → Quake; passes `apply-op-specialization` (automatic adjoint/controlled), `memtoreg`, unrolling, decomposition, mapping; **`quake-synth`** specialises the kernel with runtime arguments (JIT); output QIR (full / base / adaptive) or OpenQASM 2. 0.16: static Clifford+T synthesis, "atomic quantum regions".

**Execution**
- Target by string (`cudaq.set_target("nvidia", option="fp64,mgpu")`), same kernel on `qpp-cpu`, `nvidia`, `tensornet(-mps)`, `dynamics`, `stim` or QPUs (Quantinuum, IonQ, IQM, Pasqal, QuEra, Braket…).
- `sample` / `observe` / `get_state` / `run` (+ `_async` variants with futures). **Multi-QPU**: `mqpu` creates one virtual QPU per GPU (`qpu_id=i`), `observe(..., execution=cudaq.parallel.thread|mpi)` splits Hamiltonian terms.
- **NVQLink / `device_call`**: classical calls (CPU/GPU/FPGA) from the kernel, lowered into the IR with a latency contract (~4 µs FPGA-GPU-FPGA round trip measured).

**Simulation**: cuStateVec (fp32 default, fusion up to 10 qubits, GPU→host migration, MPI/NVLink distribution from 25 qubits), cuTensorNet, MPS, cuDensityMat; **PTSBE** (pre-sampled Kraus trajectories with shot batching for training AI decoders).

**QEC**: CUDA-Q QEC 0.8: codes (surface, colour…), decoders `nv-qldpc-decoder` (BP+OSD, GPU Relay-BP), `trt_decoder` (ONNX/TensorRT for AI decoders), PyMatching, sliding window, tensor network; **real-time decoding** inside kernels (`enqueue_syndromes`/`get_corrections`).

**Algorithms**: functors `cudaq.control/adjoint`, `compute_action` (within/apply); no autodiff (finite-difference/parameter-shift gradients); CUDA-Q Solvers deprecated → **CUDA-Q Algorithms 0.1** (LCU, block encoding, qubitisation, QSVT, Trotter).

**Weaknesses**: churn (spin_op, `sample` vs `run`, rewritten bridge, LLVM 16→22); "kernel Python" is not Python (no dynamic lists, limited closures) so LLMs produce invalid kernels; GPU only on Linux/NVIDIA; fragmented docs.

**Best idea to steal → the QPU as just another device** in a compiled heterogeneous program: one kernel, target by string, asynchronous multi-QPU execution and real-time classical calls as part of the IR.

---

## 4. PennyLane (+ Catalyst) — the differentiable framework

**IR**: QNode → tapes (`QuantumScript`) with operations and measurement processes; "capture" path (plxpr = JAXPR + quantum primitives) to Catalyst. Official alias `import pennylane as qp` (2026).

**Compiler**: `@qp.transform` transforms and `qp.CompilePipeline` with markers; `cancel_inverses`, `merge_rotations`, graph-based decomposition to a gate set; **`qp.specs`/`qp.draw(level=…)` show any stage**. Catalyst (`qp.qjit`): MLIR (`quantum`, `gradient`, `mitigation`, PBC/QEC dialects) + StableHLO + Enzyme; MLIR passes writable in Python (xDSL).

**Execution**: Device API v2 (only `execute(circuits, config)` required; preprocessing and derivative hooks) with **capabilities declared in TOML** that automatically generate the preprocessing pipeline. Mid-circuit measurement: `deferred`, `one-shot`, `tree-traversal`. 0.46 beta: **Backline** (controller + GPU/FPGA coprocessors, Triton decoders).

**Simulation**: Lightning family (C++ SIMD, Kokkos, cuStateVec, tensor); **adjoint differentiation** is the variational workhorse.

**Algorithms**: `diff_method` ∈ {backprop, adjoint, parameter-shift, finite-diff, spsa, hadamard, device}; JAX/Torch interfaces; operator arithmetic (`adjoint`, `ctrl`, `pow`); huge template library; FT resource estimator.

**Weaknesses**: per-tape Python overhead; two semantic worlds (tape vs. qjit); constant deprecations (TF, device shots, `qml`→`qp`) that poison the LLM corpus.

**Best idea to steal → differentiation as a first-class citizen (adjoint + parameter-shift)** and **the compilation pipeline as an inspectable object** generated from declared capabilities.

---

## 5. Cirq (Google Quantum AI) — stability and QEC

**IR**: `Circuit` = list of **`Moment`s** (operations on disjoint qubits → explicit scheduling); `FrozenCircuit`/`CircuitOperation` (repetitions, maps); sympy parameters + sweeps; insertion strategies.

**Protocols**: duck typing (`_unitary_`, `_decompose_`, `_kraus_`, `_act_on_`, `_has_stabilizer_effect_`, `_json_dict_`) with fallback chains: any object joins simulation and compilation without inheritance.

**Compiler**: transformers as pure functions (`@cirq.transformer`), `optimize_for_target_gateset` (CZ, √iSWAP), `eject_z`, `merge_k_qubit_unitaries`, dynamical decoupling, `RouteCQC`. No pass manager and no real Target.

**Execution**: `Sampler.run/run_sweep/run_batch`; Google Engine; **quantum virtual machine** with Willow/Weber noise.

**Simulation**: numpy simulators (slow beyond 20 qubits); **qsim** (C++ AVX/OpenMP, fusion, cuStateVec; on Apple Silicon only a basic non-vectorised build).

**QEC**: Cirq is the gateway to **Stim** (bit-packed Pauli-frame simulation, `DETECTOR`/`OBSERVABLE_INCLUDE`, **DEM** as the exchange format), **PyMatching 2.4** (sparse blossom, ~1M errors/core·s; correlated matching), **Sinter** (reproducible statistics), Chromobius, Tesseract; Willow: Λ = 2.14 at d=7 and real-time decoding (63 µs mean latency at d=5). Characterisation (XEB, RB, T1, confusion matrices).

**Weaknesses**: one Python object per operation; slow sympy; weak routing; slow release cadence.

**Best idea to steal → the DEM as the narrow waist** (any circuit generator → DEM → any decoder) and Stim's **bit-packed Pauli-frame simulation**; from Cirq itself: API stability and extensible protocols.

---

## 6. Q# / Microsoft QDK — the language with the strongest guarantees

**IR**: Rust pipeline `parse → HIR → FIR → RCA → partial eval → RIR → QIR`; also compiles to WASM (playground, VS Code, linter, debugger).

**Compiler**: **RCA (Runtime Capabilities Analysis)** + **partial evaluation** against a declared **target profile** (`Base`, `Adaptive_RI`, `Adaptive_RIF`, `Adaptive_RIFLA`, `Adaptive`, `Unrestricted`): the user (or the LLM) gets "this construct needs Adaptive_RIF" *at compile time*, not a hardware rejection after hours in a queue.

**Language**: `operation` vs `function`; **automatically generated functors** (`is Adj + Ctl` → `Adjoint Op`, `Controlled Op`); conjugation `within { U } apply { V }`; scoped qubits (`use q = Qubit()`), which must be released in |0⟩.

**Execution / simulation**: `qdk` package (replaces `qsharp`); sparse simulator, stabilizer (thousands of qubits, branching for T), full-state (~25 qubits) and **GPU via wgpu/WebGPU** (cross-vendor: Metal, Vulkan, DX12).

**QEC / resources**: **resource estimator** (QRE v3: `Application`, `Architecture`, `ISA`; T factories, Pareto frontiers), no Azure account needed.

**Weaknesses**: niche language (small LLM corpus); small local simulators; packaging churn; no autodiff.

**Best idea to steal → automatic functors (Adjoint/Controlled) + compile-time capability analysis + resource estimation before execution.**

---

## 7. Amazon Braket SDK — devices as data and task economics

**IR**: **OpenQASM 3 as the wire format** (`#pragma braket result …`, `verbatim` boxes), `FreeParameter` → `input float`. Pydantic device-property schemas (native gates, connectivity, calibration, pulse frames). AHS (analog Hamiltonian simulation) with `AtomArrangement`/`DrivingField`.

**Compiler**: deliberately thin (vendor compilers do the work); the **local emulator** (`AwsDevice(arn).emulator()`) validates connectivity/native gates and builds noise from real calibration data.

**Execution**: durable asynchronous tasks (`AwsQuantumTask`, results in S3); **ProgramSets** (`product()`/`zip()` of circuits × parameters × observables in one task → amortises the $0.30/task fee and queue latency); Hybrid Jobs with queue priority; cost `Tracker`.

**Simulation**: managed SV1 (34 qubits), DM1, TN1; modest local Python simulator.

**Weaknesses**: per-task fee + queue latency punish fine-grained loops; slow Hybrid Jobs container start-up; experimental dynamic circuits.

**Best idea to steal → an emulator built from the device's own data** (validation + cost before spending) and **ProgramSet batching** at the task boundary.

---

## 8. Qibo (TII) — from simulation to your own lab

**IR**: `Circuit` with a flat gate queue and lazy moments; **noise channels are gates** (`DepolarizingChannel`, `ThermalRelaxationChannel`, `ReadoutErrorChannel`, Kraus); measurement with collapse mid-circuit.

**Compiler**: simple transpiler (Trivial/ReverseTraversal placers, ShortestPaths/Sabre routers, `Unroller(NativeGates)`), gate fusion for simulation.

**Execution**: swappable backends (`set_backend`: numpy, qibojit numba/cupy/cuQuantum, qibotn, qiboml, qibolab) — one change moves from GPU simulation to **your own hardware**. **Qibolab** models the platform (IQ/DC/acquisition channels, Qblox/QM/Keysight/RFSoC instruments) with pulse sequences and hardware-loop *sweepers*.

**QEC / experimentalist**: **Qibocal**: acquire → fit → report → update protocols (spectroscopy, Rabi, Ramsey, T1/T2, DRAG, single-shot classification, RB, chevron, CZ…) declared in YAML runcards that **write calibrated parameters back into the platform**. Mitigation (ZNE, CDR, readout). No detectors/decoders.

**Weaknesses**: global mutable state (backend/transpiler) that hurts concurrency; simplistic compiler; small team; per-gate Python overhead.

**Best idea to steal → calibration as code with a closed loop**: one platform model shared by simulation, transpiler, pulses and calibration.

---

## 9. Comparative matrix by area

Scale: ★ basic · ★★ solid · ★★★ industry reference.

| Area / capability | Qiskit | TKET | CUDA-Q | PennyLane | Cirq | Q# | Braket | Qibo |
|---|---|---|---|---|---|---|---|---|
| **Algorithms: composition** (functors, templates) | ★★ | ★★ | ★★★ | ★★★ | ★★ | ★★★ | ★ | ★★ |
| Automatic differentiation | ★ | – | ★ | ★★★ | ★ | – | – | ★★ |
| **Heterogeneous orchestration** (async, multi-QPU, HPC) | ★★ | ★★ | ★★★ | ★★ | ★ | ★ | ★★ | ★ |
| **IR / representation** | ★★★ | ★★★ | ★★★ | ★★ | ★★ | ★★★ | ★★ | ★ |
| **Circuit optimisation** | ★★★ | ★★★ | ★★ | ★★ | ★★ | ★★ | ★ | ★ |
| Routing / mapping | ★★★ | ★★★ | ★★ | ★ | ★ | ★ | – | ★★ |
| **Hardware abstraction** (target as data) | ★★★ | ★★ | ★★ | ★★★ | ★★ | ★★ | ★★★ | ★★★ |
| Runtime protocols (QIR, OpenQASM 3, real time) | ★★ | ★★★ | ★★★ | ★★ | ★ | ★★★ | ★★ | ★★ |
| Simulation performance (CPU/GPU) | ★★ | ★ | ★★★ | ★★★ | ★★ | ★★ | ★★ | ★★ |
| **QEC: simulation + DEM + decoders** | ★ | ★ | ★★★ | ★★ | ★★★ (Stim/PyMatching) | ★★ | – | ★ |
| **Experimentalist** (calibration, pulses, characterisation) | ★★ | ★ | ★★ | ★★ | ★★ | ★ | ★★ | ★★★ |
| API stability (AI) | ★ | ★ | ★ | ★ | ★★★ | ★★ | ★★ | ★★ |
| Diagnostics for agents | ★★ | ★★ | ★ | ★★ | ★★ | ★★★ | ★★ | ★ |

---

## 10. Cross-cutting patterns (what the industry is learning)

1. **Native cores, thin facades.** Qiskit (Rust `CircuitData`/`DAGCircuit`), TKET (HUGR/Rust), Selene (Rust), Q# (Rust). The new frontier is **typed multi-IR pipelines** (`MultiStagePassManager`, `QkIrHandle`, HUGR extensions, MLIR dialects).
2. **Declarative execution + server-side randomisation** (IBM Executor/samplex; per-shot compiled classical code on Helios).
3. **Capabilities as data + early validation** (PennyLane TOML, Braket properties, QIR module flags, Q# RCA).
4. **Real time as an IR-level call with a latency contract** (CUDA-Q `device_call`, Backline `runtime_call`), not RPC.
5. **QEC converges on DEM + swappable decoder**, trending to **neural pre-decoder + classical global decoder** (AlphaQubit 2 real-time < 1 µs/cycle up to d=11; TensorRT in CUDA-Q QEC).
6. **AI**: MCP servers (official Qiskit, qBraid, Coda), `llms.txt` (Braket), trained code assistants (Qiskit Code Assistant). What helps agents most: **stability, an executable verifier, precise diagnostics, current documentation**.

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
| `Target` as single source of truth | Qiskit Target, Braket device properties | `qumo.compiler.target.Target` (modality presets, none default) |
| Capability errors at compile time | Q# RCA | validation with `QM-E…` codes before execution |

### 11.3 Quantum execution (hardware abstraction + protocols)

| Idea | Origin | In QUMO |
|---|---|---|
| Same program, swappable target | CUDA-Q `set_target`, Qibo `set_backend` (but **no global state**) | explicit backend per call + explained automatic selection |
| Simulator choice by circuit structure | Qibo/QDK (Clifford → stabilizer) | auto-dispatch: Clifford → tableau/frame, else statevector |
| Static sampling vs. trajectories | CUDA-Q `sample` vs `run`, PennyLane mcm_method | `qumo.sim.runner.sample` decides and reports `method` |
| OpenQASM 3 as wire format | Braket, IBM | `qumo.ir.qasm` (parser + emitter) |
| Emulator from device calibration | Braket emulator, Cirq virtual engine | noise model from `Target`/calibration |
| Resource estimation before running | Q# QRE, PennyLane `specs`, Braket `Tracker` | `qumo.exec.resources` (logical, NISQ hardware, fault-tolerant) |

### 11.4 Error correction (experimentalist)

| Idea | Origin | In QUMO |
|---|---|---|
| Bit-packed Pauli-frame simulation (64 shots/word) | Stim | `qumo.sim.frame` |
| Aaronson–Gottesman tableau | Stim, Qiskit Aer, QDK | `qumo.sim.stabilizer` |
| DEM as the narrow waist | Stim, CUDA-Q QEC `dem_from_kernel` | `qumo.qec.dem` (Stim-compatible text) |
| Swappable decoders (UF, MWPM, lookup, AI) | PyMatching, CUDA-Q QEC, AlphaQubit | `qumo.qec.decoders` (common interface) |
| Reproducible statistics harness | Sinter | `qumo.qec.experiments` (threshold sweeps) |
| Calibration as code / noise from T1, T2, errors | Qibocal, Braket emulator, Qiskit `NoiseModel.from_backend` | `qumo.qec.characterization` (NoiseModel, RB, readout mitigation) |

---

## 12. Why Mojo?

| Problem shared by all 8 frameworks | How Mojo solves it |
|---|---|
| Slow Python facade + C++/Rust core (two languages, FFI, builds) | One language: Python-like syntax, compiled natively, no internal FFI |
| Per-gate/per-object overhead | POD structs in contiguous lists; no GIL, no boxing |
| SIMD/threads need C++/OpenMP | Native `SIMD[DType.float64, 8]`, `vectorize`, `parallelize` |
| GPUs need separate CUDA/HIP/wgpu code | GPU kernels in Mojo via MAX (NVIDIA/AMD/Apple) — same code base |
| Compile-time specialisation (quake-synth, Catalyst) | Native `comptime` / compile-time parameters |
| Lowering to MLIR/QIR | Mojo compiles through MLIR (natural path to quantum dialects) |
| LLMs and Python | Familiar syntax for models; QUMO additionally speaks OpenQASM 3 and JSON |

**Mojo risks** (measured during this project): the language still changes quickly (1.0→1.1 removed `fn`, `alias`, `@parameter for`, renamed `UnsafePointer`; `parallelize` and GPU moved to `max.*`); no convenient dynamic trait objects (QUMO dispatches passes and backends by ID); small ecosystem. Mitigation: `docs/MOJO_GUIDE.md` pins the exact dialect and per-module tests guard it.

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

### Main sources
Qiskit 2.5/2.6 release notes and IBM blog; qiskit-ibm-runtime 0.50; Benchpress (arXiv:2409.08844); HUGR (arXiv:2510.11420), tket/guppylang/Selene (Quantinuum GitHub); CUDA-Q 0.16 docs, CUDA-Q QEC 0.8, NVQLink (arXiv:2510.25213); PennyLane 0.45 / Catalyst docs, Backline (arXiv:2609.09270); Cirq 1.7, qsim, Stim 1.16, PyMatching (sparse blossom, arXiv:2303.15933), Willow (arXiv:2408.13687), AlphaQubit 2 (arXiv:2512.07737), Relay-BP (arXiv:2506.01779); QDK releases and target profiles (learn.microsoft.com); Braket program sets, emulator and pricing (AWS); Qibo/Qibolab/Qibocal (qiboteam GitHub); QIR Adaptive Profile (qir-alliance); OpenQASM 3.1; ORBIT-Q (arXiv:2607.03105), PennySynth, QuanBench.
