# QUMO architecture

QUMO is one Mojo code base with five layers (the diagram) over a shared base (`core`, `ir`, `device`),
plus three side packages (`ft`, `gpu`, `net`). Everything above the IR is optional; everything shares the
same `Circuit` value and the same error-code, seeding and JSON conventions. The import rules between
packages are in [Package layering](#package-layering). Vendor code (the clients of hardware services and
their wire formats) is not in `qumo/`: see [Vendor code](#4c-vendor-code-outside-qumo).

```
            ┌────────────────────────────── front ends ──────────────────────────────┐
            │  cli/main.mojo (qumo CLI)   qumo mcp (MCP stdio)   python/qumo_py.so     │
            └──────────────┬──────────────────────┬───────────────────────┬────────────┘
                           │  qumo.ai: JSON, tool registry, diagnostics, describe        │
   ┌───────────────────────┴───────────┐  ┌───────┴─────────────┐  ┌──────┴────────────┐
   │ qumo.algorithms                   │  │ qumo.exec           │  │ qumo.qec          │
   │ kernels, gradients, optimizers,   │  │ PUB primitives,     │  │ codes, DEM,       │
   │ VQE/QAOA, Hamiltonians            │  │ HEFT orchestrator   │  │ decoders, sweeps, │
   └───────────────┬───────────────────┘  │                     │  │ characterization  │
                   │                      └───────┬─────────────┘  └──────┬────────────┘
   ┌───────────────┴──────────────────────────────┴───────────────────────┴────────────┐
   │ qumo.compiler: pass pipeline (text spec), routing, synthesis (targets: qumo.device)│
   ├───────────────────────────────────────────────────────────────────────────────────┤
   │ qumo.noise: NoiseModel, characterization, M3  │ qumo.device: Target (data)        │
   ├───────────────────────────────────────────────────────────────────────────────────┤
   │ qumo.sim: state vector (SIMD + threads), tableau, Pauli frames, runner            │
   ├───────────────────────────────────────────────────────────────────────────────────┤
   │ qumo.ir: gates (opcodes + metadata + matrices), Param, Circuit, Pauli, QASM 2/3,  │
   │          circuit and program JSON                                                 │
   ├───────────────────────────────────────────────────────────────────────────────────┤
   │ qumo.core: JSON writer and parser, Rng (imports nothing from QUMO)                │
   └───────────────────────────────────────────────────────────────────────────────────┘
```

## Package layering

A package imports only from packages on lower rows. The right column is what each package imports
today (`qumo/*/*.mojo`, `from qumo.X` lines; the re-export shims for old paths are not counted).

| Row | Package | Role | Imports |
|---|---|---|---|
| 0 | `core` | JSON writer and parser, `Rng` | nothing from QUMO |
| 1 | `ir` | gates, `Circuit`, `Program`, Pauli, QASM, circuit/program JSON (`ir/json.mojo`) | `core` |
| 2 | `device` | `Target`: coupling graph, basis, errors, durations, presets, `parse_target`; `conformance_error` | `ir` |
| 2 | `sim` | state vector, tableau, Pauli frames, runners, density, branching, CPU simulator selection (`dispatch`) | `core`, `ir` |
| 3 | `noise` | `NoiseModel`, `apply_noise_model`, RB, T1/Ramsey, readout mitigation, M3, the device emulator (`emulate`) | `core`, `ir`, `device`, `sim` |
| 3 | `compiler` | pass manager, routing, synthesis, `Program` compilation, static resource estimation (`resources`) | `core`, `ir`, `device`, `sim` |
| 3 | `qec` | codes, DEM, decoders, sweeps | `core`, `ir`, `sim` |
| 3 | `gpu` | vendor-neutral GPU state-vector and frame back ends (`-D QUMO_GPU`); simulator selection over CPU and GPU (`dispatch`: `sample_auto`, `sample_on`) | `core`, `ir`, `sim` |
| 3 | `net` | HTTP/1.1 + TLS client (no QUMO imports). No package of `qumo/` imports it: its users are the vendor clients outside the core | nothing from QUMO |
| 4 | `algorithms` | kernels, gradients, optimizers, VQE/QAOA, mitigation | `core`, `ir`, `sim`, `noise`, `qec`, `gpu`, `exec` (one file) |
| 4 | `exec` | orchestration only: PUB primitives and the HEFT orchestrator | `ir`, `sim`, `gpu` |
| 4 | `ft` | fault-tolerant programming layer (Steane, Iceberg, Rz synthesis) | `core`, `ir`, `sim`, `qec` |
| 5 | `ai` | tool registry, the tool handlers of every module, diagnostics, describe, MCP | `core`, `ir`, `device`, `sim`, `noise`, `compiler`, `qec`, `gpu`, `algorithms`, `ft` |
| - | `bench`, `testing` | benchmark circuits, test helpers | `core`, `ir` (`testing`: nothing from QUMO) |

`core`, `ir`, `device`, `sim` and `noise` import none of `compiler`, `qec`, `ai`, `exec`, `algorithms`.
`ai` is the top row: no other package imports it, and the package graph has no cycle. The agent-tool
handlers of every module are in `qumo/ai` (`exec_tools.mojo`, `algorithm_tools.mojo`, `ft_tools.mojo`;
the first and the last were `qumo/exec/tools.mojo` and `qumo/ft/tools.mojo`), so `exec`, `ft` and
`algorithms` do not import `ai`, and `exec` no longer imports `algorithms`. Two imports go sideways or
up inside the stack, neither closes a cycle: `algorithms/mitigation.mojo` imports `exec.primitives` (the
QWC measurement helpers; `exec.primitives` itself imports only `core`, `ir` and `sim`), and `compiler`
imports `sim` (for `program_compile` and `resynth`).

Old import paths are kept for one release as one-line re-exports: `qumo.sim.rng` (now `qumo.core.rng`),
`qumo.compiler.target` and the `basis_str` / `fmt_num` names of `qumo.compiler.common` (now `qumo.device`),
`qumo.qec.characterization` and `qumo.qec.mthree` (now `qumo.noise`), and the circuit and program JSON
functions of `qumo.ai.describe` (now `qumo.ir.json`). `qumo.exec.tools` and `qumo.ft.tools` have no
re-export: one inside `exec` or `ft` would import `ai` again. Import `qumo.ai.exec_tools` and
`qumo.ai.ft_tools`.

`exec` holds orchestration only (`primitives`, `orchestrator`). What else was there went to the package
its imports put it in, and `qumo.exec.dispatch`, `qumo.exec.emulate` and `qumo.exec.resources` are
re-exports for one release (none closes a cycle; they are not counted in the table):

| Was | Is | Why there |
|---|---|---|
| `exec/dispatch.mojo`: `choose_method_cpu`, `sample_frame`, `counts_from_measurements` | `sim/dispatch.mojo` | CPU simulator selection; imports `ir` and `sim` only |
| `exec/dispatch.mojo`: `sample_auto`, `sample_on`, `choose_method`, `gpu_sample_route`, `sample_frame_gpu`, `gpu_eligible`, `frame_eligible_gpu` | `gpu/dispatch.mojo` | the selection that calls the GPU back ends; `gpu` imports `sim`, so in `sim` it would close a cycle |
| `exec/emulate.mojo` | `noise/emulate.mojo` | it is `NoiseModel.from_target` plus a CPU sampler; `noise` imports `sim`, so it cannot be in `sim` |
| `exec/resources.mojo` | `compiler/resources.mojo` | static analysis of a circuit; imports `ir` only |
| `conformance_error` of `compiler/pass_manager.mojo` (still importable from there) | `device/conformance.mojo` | so that `noise.emulate` does not import the compiler |

There are three files named `emulate.mojo`, each in its own package: `noise/emulate.mojo` (a device's
noise on the CPU), `gpu/emulate.mojo` (host emulation of the GPU kernel bodies) and `ft/emulate.mojo`.

## 1. IR (`qumo/ir`)

The IR is self-contained: `qumo/ir` imports nothing from the simulators, the compiler, the algorithms, the GPU code or the hardware clients (only the JSON helpers of `qumo/core`), so it can be read, tested and changed on its own. The package root redirects to it for the common names (`from qumo import Circuit, parse_qasm, to_qasm`); the full paths (`from qumo.ir.circuit import Circuit`) are the same objects. The file-by-file list is in the docstring of [`qumo/ir/__init__.mojo`](../qumo/ir/__init__.mojo).

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
  gates needed by QPE / Hadamard tests), `power(k)`; `qumo.ir.mc.controlled_by`
  takes any number of controls and expands to ordinary gates at build time.
* **Side tables**: what does not fit a POD instruction lives on the `Circuit`
  and the instruction holds an index: register conditions, detector records,
  Kraus channels (`Op.KRAUS1/2`), `PAULI_CHANNEL_2` probabilities, the Pauli
  products of `E`. Pauli-based programs are a separate type, `qumo.compiler.pbc`.
* **QEC in the same IR**: noise channels (`DEPOLARIZE1/2`, `X/Y/Z_ERROR`,
  `PAULI_CHANNEL_1/2`, `E`) and annotations (`DETECTOR`, `OBSERVABLE`, `TICK`) with
  Stim-style absolute measurement records.
* **`Program`** (`program.mojo`, `classical.mojo`) is `Circuit` plus what a flat list
  cannot hold: structured regions on a tape, classical variables, expressions in
  postorder. The `Circuit` is its arena: a block is a range of ops, and no opcode is
  new, because every walker of `.ops` skips what it does not know. `cfg_of` derives
  the control-flow graph, used by the interpreter and by the verifier. It is a
  separate type so that a route that was not adapted does not compile
  ([CONTROL_FLOW.md](CONTROL_FLOW.md)).
* **Text formats**: OpenQASM 3 is canonical (`qasm.mojo` re-exports the lexer `qasm_lexer.mojo`,
  the parser `qasm_parser.mojo` and the emitter `qasm_emit.mojo`; round-trip exact,
  `#pragma qumo` for noise/detectors); OpenQASM 2 export (`qasm2.mojo`) for
  interoperability with every other SDK. Two rules of the parser worth knowing: in an OpenQASM 2
  file `^` is the power operator (`rz(2^3) q[0];`), while in OpenQASM 3 it is XOR and the power is
  `**`; and `pi`, `tau` and `euler` are built-in constants that cannot be declared as a variable,
  input, register, gate, gate parameter or loop variable (QM-E522).
* **JSON**: `qumo/ir/json.mojo` writes and reads `qumo.circuit.v1` and `qumo.program.v1`
  (`circuit_to_json`, `circuit_from_json`, `program_to_json`, `program_from_json`);
  `qumo.ai.describe` keeps only the agent-facing descriptions and re-exports these for one release.
* **Shared utilities**: `qumo/core` (JSON writer and parser, and the seeded `Rng`) imports nothing from QUMO and is
  used by every layer above the IR. `GateSet` and the multi-qubit rewrite rules (`expand_rule`)
  live in `qumo/ir/gateset.mojo`, so the simulator does not depend on the compiler.

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
* **Automatic dispatch** (`qumo.sim.dispatch` for the CPU methods, `qumo.gpu.dispatch` for `sample_auto` /
  `sample_on`, which add the GPU): noisy Clifford circuits with
  many shots → Pauli frames; other Clifford circuits → tableau; everything else
  → state vector. Reported in `SampleResult.method`.
* **Cache blocking**: for ≥ 19 qubits, consecutive kernels on qubits < 15 are
  applied block by block (2^15 amplitudes = 512 KiB per block, one pass over
  memory for the whole run of gates).
* **Determinism**: xoshiro256** seeded by SplitMix64 (`Rng`, in `qumo/core/rng.mojo`); `split(k)` per worker.

* **Programs**: `program_runner.mojo` walks the graph shot by shot through a static
  `ShotBackend` (state vector, tableau, branching). A program without runtime regions
  is a circuit and goes to `sample`, as a static circuit always did.

## 3. Compiler (`qumo/compiler`)

* **`Target`** (defined in `qumo/device/target.mojo`, so the simulators, `noise` and `exec` can use it without importing the compiler; `qumo.compiler.target` re-exports it) is data: qubits, undirected edges, basis opcodes, per-gate
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

* **Programs**: `program_compile.mojo` runs the passes around `route` on each block
  and keeps one layout: a region exits with the layout it entered with, and
  `verify_program` checks the skeleton, every edge and the blocks it can.
* **Resources** (`resources.mojo`): logical metrics + recommended simulator, NISQ duration /
  success probability per modality profile, surface-code fault-tolerant
  estimate (distance, factories, physical qubits, runtime).

## 3b. AI-transpiler training (`rl/`: `qrl` and `qumo_bridge`)

An independent package next to QUMO, never imported by `qumo/` (the runtime inference stack only
reads the model files it writes). It is pure Mojo, with no Python, Rust, torch or numpy, and has
two parts (rl/README.md):

* `rl/qrl`, the core, imports nothing from `qumo.*` (`make check-independence`) and builds with
  `-I rl` alone: portable SIMD f32 kernels (`tensor`, `optim`, `rlrng`), the synthesis network
  and its PPO loss (`nn`, `loss`), native ports of the four synthesis environments (`topology`,
  `symmetry`, `gateset`, `synth_env`, over the packed-state dynamics `env_core`), the PPO
  collector, evaluator and trainer (`ppo`), the model-container formats `qumo.rlsynth.v0` /
  `qumo.rlroute.v0` with their writer, reader, INDEX writer and verifier (`container`, `model`,
  `digest`, `f16`), run directories, checkpoints and the synthesis exporter (`checkpoint`,
  `export_synth`), the router's behaviour cloning (`teacher_data`, `router_net`, `router_loss`,
  `bc`), circuit-family and topology generators (`targets`, `jobs`), the synthesis driver and
  campaign (`synth_train`, `campaign`), and its own JSON, RNG and opcodes (`json`, `rng`,
  `gates`, copies of QUMO's).
* `rl/qumo_bridge`, the only part that imports QUMO: job circuits as QUMO circuits
  (`circuits`), the router export and PPO fine-tune that play episodes in QUMO's routing
  environment (`export_route`, `route_ppo`), block evaluation against QUMO's synthesizers
  (`eval_blocks`), and the tools built on them (`route_tool`, `eval_synth`, ...). Its tests pin
  every copied piece of qrl against QUMO's original and load qrl-exported models in QUMO.

The tool mains are thin (`rl/drivers`, `rl/qumo_bridge/drivers`; `make train-ai`). Results are
independent of the core count (fixed seed tree and reduction order). Design: `rl/docs/MOJO_PORT.md`.

## 4. Execution (`qumo/exec`)

* **Primitives**: `EstimatorPub(circuit, observables, parameter_sets, precision)`
  and `SamplerPub(circuit, shots, parameter_sets)`. Exact estimation on the
  state vector, or hardware-faithful shot estimation with qubit-wise
  commuting measurement groups and standard errors. Parameter sets of small
  circuits run in parallel. In shot mode `precision` is the standard error the
  estimate stays under: shots per setting = ceil(sum over groups of (sum |coeff|)^2 /
  precision^2), at most 10,000,000 (QM-E913). The reported std is the sample
  variance of the per-shot weighted sum of a group (its terms share the shots and
  are correlated), so it is usually below that bound. Shots come from the real
  sampler: reset, noise, mid-circuit measurement and classical control are honoured,
  while the exact path (`precision = 0`) evolves one pure state. An observable wider
  than the circuit raises QM-E405 in both modes.
* **Orchestrator**: a workflow is a DAG of sample/estimate jobs; a device pool
  mixes CPU/GPU state-vector simulators, stabilizer devices and QPUs with
  queue/shot/cost models. HEFT list scheduling assigns jobs (respecting
  capability constraints and an optional time-vs-money weight) and explains
  every decision; execution runs dependency waves in parallel.

Simulator selection, the device emulator and resource estimation are not here: see the table in
[Package layering](#package-layering).

## 4b. Side packages: noise, FT, GPU, network

* **`qumo/noise`**: `NoiseModel` from calibration data, `apply_noise_model`, randomized benchmarking,
  T1/Ramsey, readout calibration and mitigation, M3, and `emulate` (a compiled circuit under its target's
  noise; `noise/emulate.mojo`). Used by `algorithms.mitigation`.
  See [QEC.md](QEC.md) (Characterization).
* **`qumo/ft`**: the fault-tolerant programming layer: `LogicalProgram`, stabilizer codes, Steane and
  Iceberg architectures, factories, Clifford+T `Rz` synthesis, verification on the tableau; its tools are
  in `qumo/ai/ft_tools.mojo`. Imports are listed in the docstring of `qumo/ft/__init__.mojo`.
* **`qumo/gpu`**: vendor-neutral GPU state-vector and Pauli-frame back ends on MAX (NVIDIA, AMD, Apple),
  compiled only with `-D QUMO_GPU`; `make gpu-check` cross-compiles them without a device.
* **`qumo/net`**: native HTTP/1.1 and TLS client (OpenSSL 3 loaded at run time). It imports nothing from
  QUMO and names no vendor. Nothing else in `qumo/` imports it: its only users are the vendor clients of
  `plugins/hardware/native/qumo_hw` (the Mojo agents have their own client, `agents/mojo/qumo_agent/net`).

## 4c. Vendor code (outside `qumo/`)

**Rule: nothing under `qumo/` imports a vendor module.** The core has no client of a hardware service and
no vendor wire format; `make check-neutrality` (`tests/check_neutrality.sh`, part of `make test` and of
the CI checks job) fails if a file under `qumo/` or `cli/` imports the vendor package or is named after
one of its clients.

Vendor code is in `plugins/hardware`:

* `plugins/hardware/native/qumo_hw` (Mojo): one REST client per service (`ionq_api`, `ibm_api`,
  `quantinuum_api`, `speqtrum_api`, `qbraid_api`, `pasqal_api`; they were `qumo/exec/*_api.mojo`),
  `pasqal_pulser` (Pulser's abstract representation of an analog program and the device specs Pasqal
  Cloud returns; it was in `qumo/ir/analog.mojo`) and `common` (the one loopback-host test they share).
  The package imports `qumo.core`, `qumo.ir` and `qumo.net`, never the other way round.
* `plugins/hardware/qumo_hardware` (Python): the provider interface, the CLI `qumo-hardware` and the
  optional vendor-SDK fallbacks ([HARDWARE_PLUGINS.md](HARDWARE_PLUGINS.md)).

`qumo_hw` is not on the import path of the core: `make cli` and `make test` build with `-I .` alone, so
a vendor import in `qumo/` does not compile there. Only the builds that bind or use the clients add
`-I plugins/hardware/native`: the `qumo_py` extension (`make py`, `make py-gpu`; `python/qumo_py.mojo`
exports the clients to the Python plugins) and the Mojo examples under `examples/hardware`.

What stays in the core is neutral: `qumo/ir/analog.mojo` is the program model of any analog Rydberg
device (register, pulses, waveforms; `qumo.sim.rydberg` simulates it) and imports nothing, `Target` is
data with presets named by modality and topology, and `qumo/net` is a generic HTTP client.

## 5. Algorithms (`qumo/algorithms`) and QEC (`qumo/qec`)

See [ALGORITHMS.md](ALGORITHMS.md) and [QEC.md](QEC.md). Algorithms are plain
functions returning `Circuit`s (composable kernels), with adjoint differentiation
as the default gradient. QEC follows the DEM narrow waist: circuit generators →
detector error model → swappable decoders → Sinter-style statistics.

## 6. AI-native layer (`qumo/ai`, `cli/`, `python/`)

* **Tool registry**: every agent-facing operation is a tool (JSON Schema input,
  JSON output) available identically through MCP (`qumo mcp`), the CLI
  (`qumo tool <name> '<json>'`) and tests. The tools over `exec`, `algorithms` and `ft` are
  `qumo/ai/exec_tools.mojo`, `algorithm_tools.mojo` and `ft_tools.mojo`, added in `qumo/ai/registry.mojo`.
* **Diagnostics**: every error is `QM-Exxx: message. Hint: …`; the registry
  turns them into structured `Diagnostic` JSON (`docs/ERRORS.md`).
* **Self-description**: `qumo describe gates|errors|capabilities`, `llms.txt`.
* **Python**: `build/qumo_py.so` (Mojo-built CPython extension) exposes
  circuits, programs, QASM I/O, `transpile`, simulation and benchmark generators —
  used by the Benchpress gym and the neutral benchmark.

## 7. Conventions

| Convention | Rule |
|---|---|
| Qubit order | little-endian (bit k of a basis index = qubit k); bitstrings Qiskit-style (leftmost = highest clbit) |
| Errors | `QM-E0xx` circuit · `E1xx` params · `E2xx` functors · `E3xx` Pauli · `E4xx` simulation/execution · `E5xx` QASM · `E6xx` compiler · `E7xx` QEC · `E8xx` algorithms · `E9xx` interface · `W0xx` lint |
| Randomness | explicit `seed` everywhere; identical inputs → identical outputs |
| Neutrality | no vendor default; modality presets; benchmarks weigh all modalities; nothing under `qumo/` imports a vendor module (`make check-neutrality`) |
| Mojo dialect | pinned in [MOJO_GUIDE.md](MOJO_GUIDE.md) |
