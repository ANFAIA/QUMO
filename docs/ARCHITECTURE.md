# QUMO architecture

QUMO is one Mojo code base with five layers. Everything above the IR is
optional; everything shares the same `Circuit` value and the same error-code,
seeding and JSON conventions.

```
            ┌────────────────────────────── front ends ──────────────────────────────┐
            │  cli/main.mojo (qumo CLI)   qumo mcp (MCP stdio)   python/qumo_py.so     │
            └──────────────┬──────────────────────┬───────────────────────┬────────────┘
                           │  qumo.ai: JSON, tool registry, diagnostics, describe        │
   ┌───────────────────────┴───────────┐  ┌───────┴─────────────┐  ┌──────┴────────────┐
   │ qumo.algorithms                   │  │ qumo.exec           │  │ qumo.qec          │
   │ kernels, gradients, optimizers,   │  │ PUB primitives,     │  │ codes, DEM,       │
   │ VQE/QAOA, Hamiltonians            │  │ HEFT orchestrator,  │  │ decoders, sweeps, │
   └───────────────┬───────────────────┘  │ resources, tools    │  │ characterization  │
                   │                      └───────┬─────────────┘  └──────┬────────────┘
   ┌───────────────┴──────────────────────────────┴───────────────────────┴────────────┐
   │ qumo.compiler: Target (data), pass pipeline (text spec), routing, synthesis       │
   ├───────────────────────────────────────────────────────────────────────────────────┤
   │ qumo.sim: state vector (SIMD + threads), tableau, Pauli frames, RNG, runner       │
   ├───────────────────────────────────────────────────────────────────────────────────┤
   │ qumo.ir: gates (opcodes + metadata + matrices), Param, Circuit, Pauli, QASM 2/3   │
   └───────────────────────────────────────────────────────────────────────────────────┘
```

## 1. IR (`qumo/ir`)

* **`Instruction`** is a POD: opcode, up to 3 qubits, 3 affine `Param`s, clbit,
  classical condition (single bit, or a register condition via
  `cond_reg` → `Circuit.cond_table`). No heap allocation per gate.
* **`Circuit`** = flat `List[Instruction]` + parameter names + measurement
  records + detector/observable record table + register-condition table +
  `global_phase`. Builders return `None` (no chaining warnings).
* **Opcodes carry metadata** (`op_arity`, `op_num_params`, `op_is_clifford`,
  `op_is_diagonal`, `op_inverse_kind`, `op_is_controlled_1q`, …) used by the
  compiler, the simulators and `qumo describe`. One table, many consumers.
* **Functors**: `inverse()` (Adjoint), `controlled(q)` (Controlled; covers the
  gates needed by QPE / Hadamard tests), `power(k)`.
* **QEC in the same IR**: noise channels (`DEPOLARIZE1/2`, `X/Y/Z_ERROR`,
  `PAULI_CHANNEL_1`) and annotations (`DETECTOR`, `OBSERVABLE`, `TICK`) with
  Stim-style absolute measurement records.
* **Text formats**: OpenQASM 3 is canonical (`qasm.mojo`, round-trip exact,
  `#pragma qumo` for noise/detectors); OpenQASM 2 export (`qasm2.mojo`) for
  interoperability with every other SDK.

## 2. Simulation (`qumo/sim`)

* **State vector**: split re/im storage, pair-index kernels (`insert_zero`),
  8-wide SIMD when the lowest touched qubit ≥ 3, chunked multithreading
  (`run_chunks` over `max.algorithm.parallelize`). Specialised kernels for
  diagonal gates, X/CX/SWAP/CCX permutations and controlled gates; generic
  dense 4×4 for the rest. The runner fuses runs of 1q gates per wire.
* **Sampling**: static circuits evolve once and sample sorted uniforms
  (O(2ⁿ + shots log shots)); dynamic or noisy circuits run per-shot
  trajectories with split RNG streams. The result reports which method ran.
* **Stabilizer**: bit-packed inverse tableau (Stim's TableauSimulator design),
  mid-circuit measurement, reset, bit and register feed-forward, Pauli noise.
  **Pauli-frame sampler**: one reference run, then 512 shots per
  `SIMD[uint64, 8]` batch, parallel over batches (Stim's key idea; 3–4× Stim
  single-threaded on surface-code memory circuits).
* **Automatic dispatch** (`qumo.exec.dispatch`): noisy Clifford circuits with
  many shots → Pauli frames; other Clifford circuits → tableau; everything else
  → state vector. Reported in `SampleResult.method`.
* **Cache blocking**: for ≥ 19 qubits, consecutive kernels on qubits < 15 are
  applied block by block (2^15 amplitudes = 512 KiB per block, one pass over
  memory for the whole run of gates).
* **Determinism**: xoshiro256** seeded by SplitMix64; `split(k)` per worker.

## 3. Compiler (`qumo/compiler`)

* **`Target`** is data: qubits, undirected edges, basis opcodes, per-gate
  errors/durations, T1/T2. Presets are named by modality and topology (heavy-hex,
  grid, iSWAP grid, trapped ion, neutral atom, line, ring, torus, tree) and
  **none is a default**; `parse_target("grid:4x4/sc_cz")` gives one-line specs.
* **Pipelines are text** (`"cancel,merge,fuse1q,route(sabre),basis,cancel"`),
  parsed into pass IDs, executed with a `PassReport` per pass (ops / 2q / depth
  before and after, µs). Presets O0–O3 are just canonical strings.
* **Passes**: identity removal (Qiskit's 1e-12 process-infidelity rule),
  commutation-aware inverse cancellation, rotation merging, commutation through
  CX/CZ, 1q fusion + basis-aware Euler resynthesis, phase folding (phase
  polynomials over CNOT+phase regions), eject-Z (Cirq), 3q unrolling, KAK
  two-qubit block resynthesis (0–3 CX, 1 RXX/RZZ per interaction on ion
  targets, 0/1/2 iSWAPs when possible, absorbs adjacent SWAPs), routing, basis
  translation to every 2q family (cx, cz, ecr, iswap, rxx, rzz), SWAP elision by
  relabelling, conformance check.
* **Layout & routing**: parallel layout trials seeded by an exact VF2
  embedding, a relaxed VF2 embedding of the heaviest spanning tree, greedy
  weighted placements, dense and random layouts; each trial refined with
  forward/backward SABRE passes (LightSABRE lookahead and Qiskit 2.x layer
  heuristic alternate); 20 routing seeds on the winning layout; a star-to-path
  rewrite (fan-in/fan-out of a hub qubit) competes with plain routing.
  Classically-conditioned gates are routed like any gate and never moved
  across.
* **Layouts**: `CompileResult.initial_layout[l]` / `final_layout[l]` give the
  physical qubit of logical `l` before/after the circuit (length N physical
  after routing). O2/O3 may permute outputs even on all-to-all targets (SWAP
  elision), so always read results through `final_layout`.
* **O3** runs the O3 and O2 pipelines and keeps the better result.

## 4. Execution (`qumo/exec`)

* **Primitives**: `EstimatorPub(circuit, observables, parameter_sets, precision)`
  and `SamplerPub(circuit, shots, parameter_sets)`. Exact estimation on the
  state vector, or hardware-faithful shot estimation with qubit-wise
  commuting measurement groups and standard errors. Parameter sets of small
  circuits run in parallel.
* **Orchestrator**: a workflow is a DAG of sample/estimate jobs; a device pool
  mixes CPU/GPU state-vector simulators, stabilizer devices and QPUs with
  queue/shot/cost models. HEFT list scheduling assigns jobs (respecting
  capability constraints and an optional time-vs-money weight) and explains
  every decision; execution runs dependency waves in parallel.
* **Resources**: logical metrics + recommended simulator, NISQ duration /
  success probability per modality profile, surface-code fault-tolerant
  estimate (distance, factories, physical qubits, runtime).

## 5. Algorithms (`qumo/algorithms`) and QEC (`qumo/qec`)

See [ALGORITHMS.md](ALGORITHMS.md) and [QEC.md](QEC.md). Algorithms are plain
functions returning `Circuit`s (composable kernels), with adjoint differentiation
as the default gradient. QEC follows the DEM narrow waist: circuit generators →
detector error model → swappable decoders → Sinter-style statistics.

## 6. AI-native layer (`qumo/ai`, `cli/`, `python/`)

* **Tool registry**: every agent-facing operation is a tool (JSON Schema input,
  JSON output) available identically through MCP (`qumo mcp`), the CLI
  (`qumo tool <name> '<json>'`) and tests. Module tools are added in
  `qumo/ai/registry.mojo`.
* **Diagnostics**: every error is `QM-Exxx: message. Hint: …`; the registry
  turns them into structured `Diagnostic` JSON (`docs/ERRORS.md`).
* **Self-description**: `qumo describe gates|errors|capabilities`, `llms.txt`.
* **Python**: `build/qumo_py.so` (Mojo-built CPython extension) exposes
  circuits, QASM I/O, `transpile`, simulation and benchmark generators — used
  by the Benchpress gym and the neutral benchmark.

## 7. Conventions

| Convention | Rule |
|---|---|
| Qubit order | little-endian (bit k of a basis index = qubit k); bitstrings Qiskit-style (leftmost = highest clbit) |
| Errors | `QM-E0xx` circuit · `E1xx` params · `E2xx` functors · `E3xx` Pauli · `E4xx` simulation/execution · `E5xx` QASM · `E6xx` compiler · `E7xx` QEC · `E8xx` algorithms · `E9xx` interface · `W0xx` lint |
| Randomness | explicit `seed` everywhere; identical inputs → identical outputs |
| Neutrality | no vendor default; modality presets; benchmarks weigh all modalities |
| Mojo dialect | pinned in [MOJO_GUIDE.md](MOJO_GUIDE.md) |
