# QUMO's AI-native interface layer

QUMO treats LLM agents (Claude Code, MCP clients, scripted pipelines) as
first-class users, alongside humans. This document explains the design of the
interface layer (`qumo/ir/qasm.mojo`, `qumo/ai/*`, `cli/app.mojo`) and how
other QUMO modules plug their features into it.

| Piece | File | What it gives an agent |
|---|---|---|
| OpenQASM 2/3 parser + emitter | `qumo/ir/qasm.mojo` | A circuit format models already write fluently, with exact round trips |
| OpenQASM 2 emitter | `qumo/ir/qasm2.mojo` | Portable `qelib1.inc` output that every SDK parser loads (Qiskit strict and legacy, pytket, Cirq) |
| JSON | `qumo/core/json.mojo` | Streaming writer and arena parser, deterministic output, error positions |
| Diagnostics | `qumo/ai/diagnostics.mojo` | Stable codes, line/col, snippet, fix hint, lint warnings |
| Self-description | `qumo/ai/describe.mojo` | Gate set, error catalog, capabilities and circuit JSON, generated from the implementation |
| Tool registry | `qumo/ai/tools.mojo`, `qumo/ai/registry.mojo` | One implementation per operation, shared by MCP and CLI; `full_registry()` adds the module tools |
| Module tools | `qumo/exec/tools.mojo` | Compile, resources, VQE, QAOA, QEC experiments (all decoders), GPU status, device-explicit runs |
| MCP server | `qumo/ai/mcp.mojo` | JSON-RPC 2.0 over stdio (MCP 2025-06-18) |
| CLI | `cli/app.mojo` (commands, options, help; `run_cli(args)`), `cli/main.mojo` (the entry file) -> `build/qumo` | `qumo <command> [--json]`, plus generic `qumo tool <name> '<json>'` |
| CLI help for tools | `qumo/ai/cli_help.mojo` | `qumo tool <name> --help` generated from the tool's JSON Schema, "did you mean" hints |

```bash
.venv/bin/mojo build -I . -o build/qumo cli/main.mojo
./build/qumo run examples/foundations/bell.qasm --shots 1000
./build/qumo validate examples/foundations/teleport.qasm --json
./build/qumo mcp            # MCP server on stdio
```

> The entry file is `cli/main.mojo`: it calls `run_cli` of `cli/app.mojo`, prints
> the result and exits with its code. All of the interface is in `cli/app.mojo`, so
> tests call it in process (`tests/test_cli_frontdoor.mojo`). A main module named
> `qumo.mojo` would shadow the `qumo` package, and every `from qumo...` import
> would fail.

---

## 1. Why OpenQASM 3 is the LLM-facing circuit format

* **Models already know it.** OpenQASM is what Qiskit, Braket, TKET, Cirq,
  IBM, IonQ and Quantinuum all import and export, so public code and papers
  contain far more QASM than any single Python builder API. When an agent
  writes a circuit in QASM, it rarely needs to look up QUMO-specific syntax.
* **It is text.** It fits in a tool argument, diffs cleanly, and is easy to
  quote in an error message.
* **One canonical output.** `to_qasm()` emits a single normalized OpenQASM 3
  form: `stdgates.inc` names, `c[i] = measure q[j];`, readable exact angles
  (`pi/2`, `-3*pi/4`) where they re-parse to the identical `Float64` and
  shortest round-trip decimals otherwise, and `input float[64] theta;` for
  symbolic parameters. Re-emitting that output gives byte-identical text
  (`to_qasm(parse_qasm(to_qasm(c))) == to_qasm(c)`), and the simulated state
  vectors match bit for bit (both are tested).
* **Portable output.** Native gates that are missing from `stdgates.inc`
  (`sxdg`, `iswap`, `iswapdg`, `ecr`, `rxx`, `ryy`, `rzz`, `ccz`) are emitted
  with exact `gate` definitions. The tests check each definition against the
  native unitary to 1e-12. When QUMO reads such a definition back, it treats it
  as a declaration and uses the exact native opcode.
* **OpenQASM 2 for interoperability.** `qumo.ir.qasm2.to_qasm2` (MCP
  `qumo_convert` with `to: "qasm2"`, CLI `qumo convert --to qasm2`) writes
  `qelib1.inc` output for tools that only read OpenQASM 2. Without a
  definition it uses only the gates of the *original* `qelib1.inc` (the gate
  set `qiskit.qasm2.loads` accepts in strict mode). Every other QUMO gate
  (`sx sxdg swap cswap crx cry rxx ryy rzz iswap iswapdg ecr ccz`) gets a
  `gate` definition before its first use, and the body calls only `qelib1`
  gates. Bodies are exact, global phase included, so a parser that inlines
  them (Qiskit strict, pytket) and a parser that substitutes its own class for
  a known name (Qiskit `LEGACY_CUSTOM_INSTRUCTIONS`) get the same unitary.
  Standard names are kept, so gate counts by name still work downstream.
  Parameters must be bound (QM-E102). A single-bit condition on a multi-bit
  register cannot be written in OpenQASM 2 (QM-E552). The global phase, noise
  and QEC annotations become comments. Cross-check, run on all 34 unitary gate
  kinds x 3 operand layouts, 6 random 60-gate circuits and one angle-format
  circuit (109 circuits): every circuit loads in `qiskit.qasm2` strict and
  legacy modes, in pytket and in Cirq, and gives exactly QUMO's unitary,
  global phase included. The only exceptions are 7 circuits that use `u3`
  in Cirq, which differ by a global phase only, because Cirq defines
  `u3` up to phase. Before this fix, 31 of the 109 failed to load in strict
  Qiskit, and 8 failed everywhere, because `ecr` was defined through the
  undeclared `rzx`. `tests/test_qasm2_export.mojo` checks the same contract in
  Mojo.
* **QUMO extensions stay out of other tools' way.** Noise channels, QEC
  detectors and observables, ticks and original parameter names travel in
  `#pragma qumo ...` lines. Other OpenQASM tools ignore these lines; QUMO reads
  them back.

### Accepted input (what agents actually write)

* `OPENQASM 2.0;` / `3.0;`, `include` (ignored), `qreg/creg`, `qubit[n]/bit[n]`,
  single `qubit q;`, physical qubits `$0`. Multiple registers are flattened
  in declaration order.
* Every QUMO gate and the common aliases: `cnot`, `CX`, `U`, `u`, `u1`, `u2`,
  `u3`, `phase`, `cphase`, `cu1`, `cu`, `cu3`, `csx`, `toffoli`, `fredkin`,
  `id`/`i`, and `gphase`. `u2` and `u3` follow Qiskit semantics: `u3` is exactly
  `U`, with no implicit global phase.
* Constant expressions, including `pi`/`π`, `tau`, `euler`, `+ - * / % **`,
  `sin cos tan arcsin arccos arctan exp ln sqrt abs floor ceil pow mod`.
* `input float theta;` / `input angle[32] t;` declare symbolic parameters. A
  gate angle may be any expression that is *affine in one input*, such as
  `2*theta + pi/2` or `-theta/2`. It maps to QUMO's allocation-free
  `Param(coeff, index, offset)`. Products of inputs (QM-E531) and expressions
  that mix two inputs (QM-E532) are rejected with an explanation, rather than
  silently building an expression tree.
* `gate name(params) a, b { ... }`, inlined with parameter substitution.
  Modifiers `inv @`, `ctrl @`, `ctrl(n) @`, `negctrl @` and `pow(k) @` are
  applied through the IR functors `inverse()`, `controlled()` and `power()`.
* Register broadcast (`h q;`, `cx q[0], q;`, `measure q -> c;`), inclusive
  slices `q[0:2]`, index sets `q[{0,2}]` and negative indices.
* `c = measure q;`, `c[0] = measure q[0];`, `measure q[0] -> c[0];`,
  `bit[2] c = measure q;`, `reset`, `barrier`.
* Feed-forward on one bit: `if (c[0]) x q[1];`, `if (c[0] == 1) { ... } else { ... }`,
  `if (!c[0])`; on a whole register of any size: `if (c == 5)`; on several bits:
  `if (c[0] == 1 && c[2] == 0)`. The last two take no `else`.
* `const` declarations, and `for` loops over constant ranges (inclusive, as in
  OpenQASM 3) or sets, unrolled at parse time.
* `#pragma braket ...` and other foreign pragmas, `@annotations`, and
  `// /* */` comments are ignored. `delay` is ignored with warning QM-W051.

Beyond what a `Circuit` holds, `load_any` and `parse_qasm_program` read a `Program`:
`while`, `break`, `continue`, `for` and `switch` decided during the shot, classical
variables (`bit`, `bool`, `int[n]`, `uint[n]`, `float`, `angle[n]`), assignment,
conditions on any value and gate angles computed at run time. Every tool below that
says "program" goes through them; [CONTROL_FLOW.md](CONTROL_FLOW.md) has the rules.
`parse_qasm` itself stays straight-line and answers QM-E504 with a hint to them.
An integer constant with no width is exact and does not wrap at 64 bits, and QM-W055
warns where integer division is a trap for an angle.

Not supported: subroutines (QM-E504), hexadecimal, octal and binary integer literals
(QM-E500), and pulse-level `cal`/`defcal`.

### QUMO pragmas

```
#pragma qumo depolarize1(0.01) q[0]          // any noise op: x_error, y_error, z_error,
#pragma qumo depolarize2(0.02) q[0], q[1]     //   depolarize1/2, pauli_channel_1(px,py,pz)
#pragma qumo detector 0, 1                    // absolute measurement-record indices (negative = from the end)
#pragma qumo observable(0) 2
#pragma qumo tick
#pragma qumo param theta_0 "theta[0]"         // original name of a sanitized input identifier
#pragma qumo if (c[0] == 1) x_error(0.1) q[1];
```

The trailing `;` is optional on pragma lines. `pragma qumo ...` without the
`#` is accepted too.

The full table of noise channels, including `pauli_channel_2`, `correlated_error`
and Kraus channels (`kraus1`, `kraus2`), with the matching `Circuit` methods, is in
[SIMULATION.md](SIMULATION.md#noise-channels-and-the-pragma-syntax).

### Size limits

QUMO expands gate calls, unrolled loops, broadcasts and modifiers at parse time, so
a few bytes could ask for any amount of memory: `pow(2^53 - 1) @ x` was 2^53 gates.
A source from an agent or the MCP server can come from anywhere, so the text pays
for what it asks:

* **Budget, QM-E506.** 8 million items plus the length of the source in bytes. An
  item is a declared wire, a statement run (gate bodies and unrolled loops run
  again), a target of a broadcast or slice, a control, a repeat of `pow`, or a
  classical expression node that a program keeps. An operation is 136 bytes, so a
  short source stays near 1 GB of operations and under 2 GB at its peak; a long
  one pays with its length, so written-out circuits of any size still load. The
  1000000 passes of unrolled loops (QM-E570) still hold.
* **`pow` of a gate of known period** folds and costs nothing: past its period p,
  `pow(k) @ g` is g repeated `k mod p` times. x, y, z, h, cx, cy, cz, ch, swap, ecr,
  ccx, cswap and ccz have period 2; s, sdg, sx, sxdg, iswap and iswapdg 4; t and tdg 8.
  The equality is exact, global phase included. `pow(2) @ h` stays two gates, while
  `pow(4) @ h` is none. An exponent at or past 2^53 is QM-E550: as a float it is no
  longer exact, and its parity is lost.
* **Wires, QM-E507.** A source declares at most 4096 wires beyond 3 per
  operation, the rule `circuit_from_json` already had (QM-E911): `qubit[5000] q;
  h q[0];` is refused, `qubit[5000] q; h q;` is not.
* **Tableau, QM-E713.** 262144 qubits at most (32 GiB), the way the state vector
  stops at 32 qubits (QM-E404).
* **Lint** lists 100 findings of a code per circuit and counts the rest in one
  more finding.

Not limited: CPU time. `ctrl(120) @ x` (synthesis without ancillas) and
`qumo_transpile` on 20000 qubits each take more than 25 seconds, with little memory. Stim text has its own limit, 50 million instructions after `REPEAT`.

### Speed

The lexer makes one pass over UTF-8 bytes, with no regex and no per-character
`String`. Tokens are plain fixed-size records. Columns are computed only when a
diagnostic is rendered. Statement dispatch tries the gate table first, and
`q[INT]` operands take a fast path. On an M-series laptop, `mojo run` parses
about **1.1 million gates per second** (2 MB of QASM, 100k gates, in about
90 ms including lexing) and emits the same circuit in about 31 ms. See
`tests/test_qasm.mojo::test_speed`.

---

## 2. Errors are data: stable codes, locations, hints

Every QUMO error has the form

```
QM-E510: unknown gate 'cnto' (line 6, col 1). Hint: did you mean 'cnot'?
 6 | cnto q[0], q[1];
   | ^~~~
```

* **The code is the API.** Codes never change meaning, so an agent can branch
  on `QM-E5xx` (fix the QASM), `QM-E101` (pass parameters), `QM-E930` (the
  circuit is too large) and so on. Ranges: E0 circuit, E1 parameters,
  E2 functors, E3 Pauli, E4 simulation and execution (E47x GPU), E5 OpenQASM
  parsing and export, E6 compiler, E7 stabilizer and QEC, E8 algorithms,
  E9 interface (JSON, MCP, CLI, device selection; E95x hardware plugins and
  E96x benchmark harness, both Python), W0 warnings (lint, execution notes
  such as QM-W020 GPU fallback, parser warnings). See [ERRORS.md](ERRORS.md).
* **One catalogue, generated docs.** `error_catalog()` in
  `qumo/ai/diagnostics.mojo` is the single source of truth. `docs/ERRORS.md`
  is generated from it (`describe_errors_markdown()`), and
  `tests/test_mcp.mojo` fails when the file is stale. It also fails when a code
  that appears anywhere in `qumo/`, `cli/`, `python/`,
  `plugins/hardware/qumo_hardware/` or `bench/run_benchmarks.py` is missing
  from the catalogue.
* **Every error includes a hint for the fix.** Unknown names get suggestions by
  edit distance (`hx` -> `'cx', 'h' or 'rx'`; `thetta` -> `'theta'`). Arity
  errors print the gate signature (`usage: cx q[0], q[1]; (control first, then
  target)`). Errors inside gate bodies name the call site.
* **The structured form is shared everywhere.**
  `diagnostic_from_error(text)` returns a `Diagnostic`
  `{code, severity, message, hint, line, col, snippet, instruction, docs}`.
  CLI `--json`, MCP `structuredContent.error` and `qumo_validate_qasm` all
  return this same shape. A message with no QUMO code (from the stdlib, for
  example) maps to `QM-E999`. If the raising site gave no hint, the hint is
  filled in from the catalog.
* **Lint (`QM-W0xx`) catches code that is legal but probably a mistake:**
  unused qubits (each one doubles state-vector memory), gates after
  measurement (which makes the circuit dynamic), qubits measured while still
  in |0>, overwritten bits, conditions on bits that were never written,
  adjacent gates that cancel, zero rotations, unbound parameters, memory
  estimates and deep circuits. Each finding points at an instruction index.

## 3. Self-description

The `qumo_describe` tool and `qumo describe ...` command serve documents that
are generated from the same metadata the compiler and simulators use, so they
cannot drift from the implementation:

* **gates**: for every opcode, its arity, parameter names, Clifford, diagonal,
  self-inverse and symmetric flags, inverse, aliases, whether it appears in
  `stdgates.inc`, QASM syntax, description, and the matrix of each fixed 1- or
  2-qubit gate (with the stated little-endian convention).
* **errors**: the full code catalog, including codes from other modules and
  from the Python hardware plugins and benchmark harness. `format: "markdown"`
  returns the generated text of `docs/ERRORS.md`.
* **capabilities**: conventions (qubit order, bitstring order, angle
  convention, seeding), simulator methods and memory model, the supported and
  unsupported QASM subset, error ranges, and the live list of MCP tools.
* **Circuit JSON** (`qumo.circuit.v1`): `circuit_to_json` and
  `circuit_from_json` round-trip. Every tool accepts `circuit` (JSON) instead
  of `qasm`, for agents that prefer structured input.
* **Program JSON** (`qumo.program.v1`): `program_to_json` and `program_from_json`
  round-trip, wrapping circuit JSON. Malformed input is QM-E587, or QM-E583 for a
  wrong nesting; every index is checked before anything runs.
* **Circuit summary**: qubits, clbits, op counts (sorted), depth, two-qubit
  gate count, T-count, Clifford/dynamic/noise/parametric flags, state-vector
  bytes and a recommended simulator.

## 4. Determinism

* Every random process takes an explicit seed (default 1234), so identical
  input gives identical counts. This is tested at the tool level.
* JSON key order is fixed, maps (counts, op tallies) are sorted, and floats
  use the shortest round-trip representation.
* The only nondeterministic field is `timing_ms`, which is clearly separated
  from the results.

---

## 5. MCP server

`build/qumo mcp` speaks newline-delimited JSON-RPC 2.0 on stdin/stdout and
implements MCP **2025-06-18** (it also accepts 2025-03-26 and 2024-11-05).

| Method | Behaviour |
|---|---|
| `initialize` | Negotiates the protocol version and returns `serverInfo {name: "qumo"}`, `capabilities.tools`, and `instructions` (a short usage primer) |
| `notifications/*` | No reply. Every message without an `id` is a notification and gets no reply. |
| `ping` | `{}` |
| `tools/list` | Each tool's `name`, `title`, `description`, `inputSchema` (JSON Schema) and `annotations` (read-only, idempotent, closed-world) |
| `tools/call` | `{content: [{type: "text", text}], structuredContent, isError}` |

When a tool fails, the reply is a normal *result* with `isError: true`,
`structuredContent.error` set to the `Diagnostic`, and the text content
rendered rustc-style so the model can fix its input. Protocol problems are
JSON-RPC errors: -32700 parse error (with `data` set to the QM-E900
diagnostic), -32600 invalid request (including batches, which MCP 2025-06-18
removed), -32601 unknown method, and -32602 for invalid params or an unknown
tool (with `data` `{code: "QM-E912", available: [...]}`).

### Tools

| Tool | Arguments | Result |
|---|---|---|
| `qumo_run_qasm` | `qasm` \| `circuit` \| `program`, `shots`=1024, `seed`=1234, `params`, `max_qubits`=28, `max_iterations`=65536 | `counts` (sorted), `method` (`statevector:static` or `:trajectories`), `most_frequent`, `warnings`, `timing_ms`; a program adds `iterations` and `outputs`. Circuits without measurements are measured on all qubits (QM-W004). |
| `qumo_statevector` | `qasm`, `params`, `max_amplitudes`=64, `threshold`, `max_qubits`=24 | `amplitudes[{index, bitstring, re, im, probability}]`, `num_nonzero`, `truncated`, `norm` |
| `qumo_expectation` | `qasm`, `observable` (string or array), `params` | `expectation`, `results[]` |
| `qumo_validate_qasm` | `qasm` | `valid`, `errors[]` (every error the parser can reach, not only the first), `errors_complete`, `warnings[]` (lint; QM-W054 for a program), `summary` when valid. After a failing gate call, measurement or assignment the check goes on to the next statement; an error in a declaration or a block header stops it, and `errors_complete` is then `false` (later lines were not checked) |
| `qumo_describe` | `topic`: gates \| gate \| errors \| error \| capabilities, `name`, `format`: json \| markdown | self-description JSON (`markdown`, topic `errors` only: the text of docs/ERRORS.md) |
| `qumo_circuit_info` | `qasm` \| `program`, `include_instructions`, `include_qasm`, `include_lint` | `summary`, `lint`, and optionally `circuit` (`program` for a program) and `qasm` |
| `qumo_draw` | `qasm` \| `circuit` \| `program`, `width`=80, `style`: ascii \| unicode | `drawing`: one line per qubit, columns of gates (also `qumo draw file.qasm`, `Circuit.draw()`); a program draws by blocks, under indented headers |
| `qumo_convert` | `qasm` \| `circuit` \| `program`, `to`: qasm3 \| qasm2 \| json, `params` | canonical QASM 3, portable QASM 2 (both also returned as plain text content), or circuit JSON; a program gives QASM 3 or `qumo.program.v1`, and QM-E585 for QASM 2 |

`params` binds `input float` parameters. Pass either an array in declaration
order or an object such as `{"theta": 0.1}`.

A tool that needs one circuit, such as `qumo_statevector` or `qumo_expectation`,
answers QM-E585 for a program with runtime regions: it has no single final state.

Module tools (`qumo/exec/tools.mojo`, registered by `full_registry()`):

| Tool | Arguments | Result |
|---|---|---|
| `qumo_transpile` | `qasm` \| `circuit` \| `program`, `target` (e.g. `grid:4x4/sc_cz`, `heavy_hex:5/rz,sx,x,ecr`, `trapped_ion:11`, `neutral_atom:4x4`) or `edges` + `basis`, `level` 0-3, `pipeline`, `seed` | compiled QASM 3, per-pass report, layouts, swaps, estimated fidelity |
| `qumo_estimate_resources` | `qasm`, `profile`: superconducting \| trapped_ion \| neutral_atom, `physical_error`, `error_budget` | logical, NISQ and fault-tolerant estimates |
| `qumo_vqe`, `qumo_qaoa_maxcut` | Pauli-sum `hamiltonian` / edge-list `graph`, optimizer options, `seed` | energy or cut, parameters, convergence |
| `qumo_qec_memory` | `code`: surface_z \| surface_x \| repetition, `distance`, `rounds`, `p`, `shots`, `decoder`, `seed`, `max_errors` | logical error rate, std error, Wilson interval, per-round rate, timing |
| `qumo_qec_dem` | `code`, `distance`, `rounds`, `p`, `include_stim` | detector error model in Stim DEM text |
| `qumo_emulate` | `qasm` \| `circuit`, target as `qumo_transpile`, `calibration`, `compile`, `level`, `shots`=1024, `seed`=1234 | `counts` under the target's noise, `method`, `shots`, `seed`, `num_qubits`, `compiled` (whether it was transpiled first), `estimated_fidelity`, `target`; nothing touches a device. A circuit that does not fit the target is QM-E615 unless `compile` is true |
| `qumo_density` | `qasm` \| `circuit`, `observable` | exact `probabilities`, `trace`, `purity`, `expectation`; up to 15 qubits |
| `qumo_to_pbc` | `qasm` \| `circuit`, `fix_clifford` | Pauli-product rotations and measurements in the input frame, with `counts` |
| `qumo_stim_import` | `stim`, `include_dem`, `include_qasm` | sizes, `missing_detectors`, optionally the DEM and OpenQASM |
| `qumo_ft_codes` (`qumo/ft/tools.mojo`) | `code` (optional), `check_distance`, `architecture`: steane \| iceberg_<k>, `verify_gadgets` | code definitions, verified distance, architecture description, verification status of every gadget ([FT.md](FT.md)) |
| `qumo_ft_encode` | `qasm` \| `circuit`, `architecture`, `n_blocks`, QEC policy (`qec_style`, `qec_threshold`, `qec_costs`), factories, `feed_forward`, `magic_prep`, `rz_mode`, `shots`, `noise_p`, `include_logical`, `include_qasm` | logical program summary, resource report, outputs / post-selection bits, optional emulation next to the unencoded reference |
| `qumo_ft_verify` | `code`, `kind`: clifford \| state, `semantic_qasm`, `impl_qasm`, `num_blocks`, `num_ancillas` | valid + the first failing operator |
| `qumo_gpu_info` | none | whether the build has GPU support (`-D QUMO_GPU`), device name, API (CUDA, ROCm/HIP, Metal), `precision` (what `auto` runs: fp32 by default on every GPU), `precision_build_default`, `precision_env` (`QUMO_GPU_PRECISION`), `precisions` (selectable), `precision_note`, `fp64_supported`, auto-dispatch threshold, device profile |
| `qumo_run_device` | as `qumo_run_qasm`, plus `device`: auto \| cpu \| gpu and `precision`: auto \| fp32 \| fp64 | `counts`, `method`, `device_used`, `precision`, `precision_used`, `warnings` |

Decoders of `qumo_qec_memory` (QM-E707 lists them for an unknown name):
`mwpm` (the default; exact minimum-weight perfect matching by sparse
blossom, the algorithm of PyMatching 2; aliases `matching`, `pymatching`),
`sparse_blossom` (the same decoder), `mwpm_correlated` (two-pass reweighting, opt-in:
1.3 to 2.2 times fewer logical errors at d >= 5, worse at d = 3, about 3 times
slower), `mwpm_dense` (exact MWPM by all-pairs
shortest paths plus a dense blossom; small codes and reference), `uf`
(union-find; fast, slightly lower threshold) and `lookup` (lookup table;
tiny codes with at most 64 detectors). See [QEC.md](QEC.md).

`qumo_run_device` selects the simulator as follows (`qumo.exec.dispatch.sample_on`):

* `auto` uses the same dispatch as `qumo_run_qasm`. Noisy Clifford circuits
  without feed-forward go to CPU Pauli frames. Other wide, noisy or dynamic
  Clifford circuits go to the stabilizer tableau. Static non-Clifford circuits
  with at least 20 qubits go to the GPU state vector, when the build has GPU
  support and a usable device (any GPU failure falls back to the CPU).
  Everything else runs on the CPU state vector.
* `cpu` makes the same choice among CPU simulators only.
* `gpu` sends Clifford circuits without parameters or classical feed-forward
  to GPU Pauli frames (`gpu:pauli_frame`); noise, mid-circuit measurement and
  reset are allowed. Other static, noiseless circuits go to the GPU state
  vector in `precision` (`gpu:statevector:fp32` or `gpu:statevector:fp64`).
  Three cases run on the CPU, as with `cpu`, and add warning **QM-W020** with
  the reason: dynamic or noisy non-Clifford circuits, builds without
  `-D QUMO_GPU`, and machines without a usable device. The GPU path is the
  same code for every vendor (MAX kernels for CUDA, ROCm and Metal). See
  [GPU.md](GPU.md).

`precision` selects the GPU state vector's floating-point precision:

* `auto` (default): the environment variable `QUMO_GPU_PRECISION` (`fp32` or
  `fp64`) when it is set, else the build default, which is **fp32 on every
  GPU** (as CUDA-Q's `nvidia` target; `-D QUMO_GPU_FP64` makes it fp64 on
  NVIDIA and AMD).
* `fp32` or `fp64`. Apple GPUs have no fp64: `device: "gpu"` with `fp64`
  fails with **QM-E472**; with `device: "auto"` such a circuit stays on the
  CPU, which is fp64. An unknown value fails with **QM-E941**.
* It applies to GPU state vectors only: CPU state vectors are fp64, the
  stabilizer tableau and Pauli frames are exact. The result's
  `precision_used` says what ran (`fp32`, `fp64` or `exact`), and warning
  **QM-W021** flags a requested precision that did not apply.

The CLI exposes the same choice: `qumo run x.qasm --device gpu --precision
fp64` (needs a build made with `make cli-gpu`).

Warnings in every tool result have the `Diagnostic` shape
(`{code, severity, message, hint, docs}`).

### Hardware plugins (Python)

Running on quantum hardware is outside the Mojo core and the MCP server.
`plugins/hardware` (Python package `qumo_hardware`) compiles with QUMO
(`build/qumo_py.so`), translates the result 1:1 into the provider SDK, and
submits it after explicit confirmation. The providers are IBM Quantum (Qiskit
Runtime), Quantinuum (pytket and qnexus) and a `local` emulator. Its errors
use the same text and JSON shape, in range QM-E950..QM-E959 (listed in
[ERRORS.md](ERRORS.md)). Submitting to a billable backend without
confirmation is QM-E957. See [HARDWARE_PLUGINS.md](HARDWARE_PLUGINS.md).

### Client configuration

Claude Code:

```bash
claude mcp add qumo -- /absolute/path/to/QUMO/build/qumo mcp
```

or in `.mcp.json` / Claude Desktop / any MCP client:

```json
{"mcpServers": {"qumo": {"command": "/absolute/path/to/QUMO/build/qumo", "args": ["mcp"]}}}
```

### Transcript (piped session)

```
-> {"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"x","version":"1"}}}
<- {"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-06-18","capabilities":{"tools":{"listChanged":false}},"serverInfo":{"name":"qumo","title":"QUMO quantum toolkit","version":"0.1.0"},"instructions":"..."}}
-> {"jsonrpc":"2.0","method":"notifications/initialized"}
-> {"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"qumo_run_qasm","arguments":{"qasm":"OPENQASM 3.0; include \"stdgates.inc\"; qubit[2] q; bit[2] c; h q[0]; cx q[0], q[1]; c = measure q;","shots":1000,"seed":7}}}
<- {"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"{\"counts\":{\"00\":484,\"11\":516},...}"}],"structuredContent":{"counts":{"00":484,"11":516},"shots":1000,"seed":7,"method":"statevector:static",...},"isError":false}}
-> {"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"qumo_validate_qasm","arguments":{"qasm":"OPENQASM 3.0;\nqubit[2] q;\nhx q[0];"}}}
<- {"jsonrpc":"2.0","id":4,"result":{"content":[{"type":"text","text":"INVALID\nerror[QM-E510]: unknown gate 'hx'\n  --> line 3, col 1\n 3 | hx q[0];\n   | ^~\n  hint: did you mean 'cx', 'h' or 'rx'?"}],"structuredContent":{"valid":false,"errors":[{"code":"QM-E510",...,"line":3,"col":1,...}]},"isError":false}}
```

`McpServer.handle(line) -> String` is pure (it does no I/O), so the protocol
is unit-tested without stdio (`tests/test_mcp.mojo`). `serve_stdio()` is a
buffered line loop over fd 0 that handles lines of any length and a final
line without a newline. Tool latency is well under 1 ms for small circuits.

---

## 6. CLI

```
qumo run <file|-> [--shots N] [--seed S] [--params JSON] [--max-qubits N] [--max-iterations N] [--device auto|cpu|gpu] [--precision auto|fp32|fp64]
qumo statevector <file|-> [--max-amplitudes N] [--params JSON] [--max-qubits N]
qumo expect <file|-> "0.5*Z0 Z1 - X0" [more...] [--observable PAULI_SUM]... [--params JSON]
qumo transpile <file|-> [--target SPEC] [--level 0-3] [--seed S] [--basis GATES] [--pipeline SPEC] [--ai] [--ai-level 1-3] [-o FILE] [--qasm]
qumo emulate <file|-> --target SPEC [--shots N] [--seed S] [--compile|--no-compile] [--level 0-3]
qumo vqe "<hamiltonian>" [--layers N] [--entanglement E] [--optimizer O] [--gradient G] [--maxiter N] [--qubits N] [--seed S]
qumo validate <file|->                 # exit 1 when invalid; lists every error it can find
qumo info <file|-> [--qasm] [--instructions]
qumo draw <file|-> [--width N] [--unicode]
qumo describe [gates|gate <name>|errors|error <code>|capabilities|passes|models]
qumo convert <file|-> [--to qasm3|qasm2|json] [--params JSON] [-o out]
qumo tool list | qumo tool <name> '<json-args>'|- | qumo tool <name> --help
qumo mcp | version | help [command | tool <name>]
```

Every command also takes `--json`, `--pretty` and `--help`.

The CLI prints human-readable output by default. With `--json` it prints the
tool's `structuredContent` byte for byte, and `--pretty` indents it. Exit codes
are 0 for success, 1 when the operation failed (the diagnostic goes to stderr,
or to stdout as JSON with `--json`), and 2 for usage errors (QM-E921/QM-E912).
Use `-` as the file argument to read from stdin.

**Each command is a thin adapter over one tool** and adds no simulation or
compilation logic: `run` -> `qumo_run_qasm` (`qumo_run_device` with `--device`),
`transpile` -> `qumo_transpile`, `emulate` -> `qumo_emulate`, `vqe` -> `qumo_vqe`,
and so on. `qumo transpile x.qasm --target grid:2x2/sc_cz --json` therefore prints
what the MCP tool returns.

**Options are strict.** A command declares its options in `command_table()`
(`cli/app.mojo`) and the help text is generated from the same declaration. An
option the command does not declare is a usage error, never ignored:

```
$ qumo run examples/foundations/bell.qasm --shot 10
error[QM-E921]: unknown option '--shot' for 'qumo run'
  hint: did you mean '--shots'? Options of 'qumo run': --shots, --seed, --params, --max-qubits, --max-iterations, --device, --precision, --json, --pretty, --help (see 'qumo run --help')
$ echo $?
2
```

Options are written `--name value` or `--name=value`. There is no `--quiet`
option (it was accepted and ignored before; it is QM-E921 now). `qumo help
<unknown>` and an unknown command are QM-E912 with exit code 2, and
`qumo emulate` without `--target` is QM-E921.

**`expect --observable`.** Observables are positional arguments, or
`--observable "<pauli sum>"`, which may be repeated. Both forms can be mixed;
the positional ones are evaluated first.

**`emulate`.** A circuit that already fits the target runs as written. One
that does not is compiled first, and the output says so (`"compiled": true`
in JSON); `--compile` always compiles and `--no-compile` makes a circuit that
does not fit an error (QM-E615).

**Tools from the command line.** `qumo tool list` prints one line per tool.
`qumo tool <name> --help` (or `qumo help tool <name>`) prints the tool's
description, its arguments with types, defaults and ranges, and an example,
all generated from the tool's JSON Schema. Three conveniences exist only in
the CLI and change nothing for MCP clients:

* the `qumo_` prefix is optional: `qumo tool draw '{...}'` calls `qumo_draw`;
* a tool that takes OpenQASM text in `"qasm"` also takes `"file": "path"`
  (`"-"` is stdin): the CLI reads the file and passes its text as `qasm`. The
  MCP server does not read files, and `file` is not in the tool's schema;
* the JSON arguments may come from stdin: `qumo tool <name> -`.

**Seeds.** Defaults are stated once in
[GETTING_STARTED.md](GETTING_STARTED.md#cli): 1234 for sampling and `emulate`,
7 for `transpile` and `vqe`, 1 for QEC experiments.

---

## 7. Extending: register tools and commands

**A new MCP tool** (it appears automatically in `tools/list`, in
`qumo tool <name>`, in `qumo tool list` and in `describe capabilities`):

```mojo
# qumo/compiler/tools.mojo  (owned by the compiler module)
from qumo.ai.tools import ToolRegistry, ToolResult, ToolContext, tool_ok, arg_circuit, arg_int
from qumo.core.json import JsonValue, JsonWriter
from qumo.ai.describe import write_circuit_summary
from qumo.ir.qasm import to_qasm
from qumo.compiler.pass_manager import transpile
from qumo.compiler.target import parse_target

def _transpile(args: JsonValue, ctx: ToolContext) raises -> ToolResult:
    var warnings = List[String]()
    var c = arg_circuit(args, warnings)                 # qasm or circuit JSON
    var target = parse_target(args.get_string("target", "all_to_all:" + String(c.num_qubits)))
    var r = transpile(c, target, arg_int(args, "level", 2, 0, 3))
    var w = JsonWriter()
    w.begin_object()
    w.kv_string("qasm", to_qasm(r.circuit))
    w.key("summary"); write_circuit_summary(r.circuit, w)
    w.end_object()
    return tool_ok(w.finish())

def register_compiler_tools(mut reg: ToolRegistry):
    reg.add("qumo_transpile", "Transpile circuit",
            "Optimize, route and translate a circuit for a target (e.g. 'grid:4x4/sc_cz').",
            "{\"type\":\"object\",\"properties\":{\"qasm\":{\"type\":\"string\"},"
            + "\"target\":{\"type\":\"string\"},\"level\":{\"type\":\"integer\",\"minimum\":0,\"maximum\":3}},"
            + "\"additionalProperties\":false}",
            _transpile)
```

Then add one line to `full_registry()` in `qumo/ai/registry.mojo` (module
tools live there, so that `qumo.ai.tools` needs no imports from higher-level
modules):

```mojo
    register_compiler_tools(reg)   # like register_exec_tools(reg)
```

The rules for tool handlers:

* **Signature.** A handler is `def(JsonValue, ToolContext) raises thin -> ToolResult`
  (a plain top-level function). `ToolContext` carries server context, currently
  the list of tool names. A handler must not call `default_registry()`, because
  Mojo rejects function references that recursively instantiate themselves.
* **Errors.** Raise `Error("QM-Exxx: ... . Hint: ...")` from your own range.
  The registry turns it into an `isError` result.
* **Output.** Return a JSON *object* through `tool_ok(json_text)`. Add
  `ToolResult(structured, text, False)` if a plain-text rendering is more
  useful to a model, as `qumo_convert` does with the QASM text.
* **Argument helpers.** Reuse `arg_circuit`, `arg_params`, `arg_int` and
  `arg_required` so argument errors keep their uniform codes
  (QM-E910 missing argument, QM-E911 wrong type, QM-E913 bad value).

**A new CLI command:** in `command_table()` of `cli/app.mojo`, build a
`Command("name", "summary", "<positional synopsis>", max_positional, handler)`,
declare each option with `.opt("name", "METAVAR", "help")` (an empty metavar
makes a flag), add `.example(...)` lines, and append it. `handler` is
`def(CliArgs) raises thin -> CliOut`. The help text, the option check
(QM-E921) and the "did you mean" hints come from that declaration. Most
commands build a JSON argument object, call the tool, and format the
structured result; `tests/test_cli_frontdoor.mojo` shows how to test one in
process with `run_cli`. Every registered tool is also reachable, and
documented by `--help`, without a dedicated command:

```bash
qumo tool qumo_transpile '{"file": "examples/foundations/bell.qasm", "target": "grid:3x3"}' --pretty
qumo tool qumo_transpile --help
```

**New error codes:** add an `_e(c, "QM-E6xx", ...)` line to `error_catalog()`
in `qumo/ai/diagnostics.mojo`, then regenerate `docs/ERRORS.md`:

```bash
QUMO_UPDATE_DOCS=1 .venv/bin/mojo run -I . tests/test_mcp.mojo
# or: build/qumo tool qumo_describe '{"topic":"errors","format":"markdown"}' | python3 -c 'import json,sys; print(json.load(sys.stdin)["markdown"], end="")' > docs/ERRORS.md
```

`make test` fails while a raised code is missing from the catalogue or the
file is stale.

---

## 8. Limitations and next steps

* The IR flattens registers and keeps no register names, so emitted QASM
  always uses a single `q` and a single `c`. Barriers are global.
* In a `Circuit`, conditions test one bit, a whole register or an `&&` chain of bits.
  Anything else is QM-E541 in `parse_qasm`, and a `Program` takes it.
* Programs do not run on hardware or on the GPU yet ([CONTROL_FLOW.md](CONTROL_FLOW.md#limits)).
* `ctrl @` covers the gates that the IR's `controlled()` functor supports
  (everything needed for QPE and Hadamard tests). `ctrl @ ccx` and
  `ctrl @ U` with symbolic angles raise QM-E202, located at the QASM line.
* Parameters are affine in a single input (by design: binding is
  allocation-free).
* OpenQASM 2 export needs bound parameters and cannot express single-bit
  conditions on multi-bit registers, a global phase (written as a comment) or
  noise. OpenQASM 3 export writes a condition on part of the bits as an `&&` chain.
* Sampling tools dispatch automatically (Pauli frames, stabilizer tableau,
  CPU or GPU state vector). `qumo_statevector` and `qumo_expectation` always
  use the dense CPU state vector.
* Hardware execution is a Python plugin, not an MCP tool, because it needs
  provider credentials and explicit confirmation of billable jobs.
