# QUMO's AI-native interface layer

QUMO treats LLM agents (Claude Code, MCP clients, scripted pipelines) as
first-class users, alongside humans. This document explains the design of the
interface layer (`qumo/ir/qasm.mojo`, `qumo/ai/*`, `cli/main.mojo`) and how
other QUMO modules plug their features into it.

| Piece | File | What it gives an agent |
|---|---|---|
| OpenQASM 2/3 parser + emitter | `qumo/ir/qasm.mojo` | A circuit format models already write fluently, with exact round trips |
| JSON | `qumo/ai/json.mojo` | Streaming writer and arena parser, deterministic output, error positions |
| Diagnostics | `qumo/ai/diagnostics.mojo` | Stable codes, line/col, snippet, fix hint, lint warnings |
| Self-description | `qumo/ai/describe.mojo` | Gate set, error catalog, capabilities and circuit JSON, generated from the implementation |
| Tool registry | `qumo/ai/tools.mojo` | One implementation per operation, shared by MCP and CLI |
| MCP server | `qumo/ai/mcp.mojo` | JSON-RPC 2.0 over stdio (MCP 2025-06-18) |
| CLI | `cli/main.mojo` -> `build/qumo` | `qumo <command> [--json]`, plus generic `qumo tool <name> '<json>'` |

```bash
.venv/bin/mojo build -I . -o build/qumo cli/main.mojo
./build/qumo run examples/bell.qasm --shots 1000
./build/qumo validate examples/teleport.qasm --json
./build/qumo mcp            # MCP server on stdio
```

> The entry file is `cli/main.mojo`. A main module named `qumo.mojo` would
> shadow the `qumo` package, and every `from qumo...` import would fail.

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
* Single-bit feed-forward: `if (c[0]) x q[1];`, `if (c[0] == 1) { ... } else { ... }`,
  `if (!c[0])`, and QASM 2 `if (c==1)` on 1-bit registers.
* `const` declarations, and `for` loops over constant ranges (inclusive, as in
  OpenQASM 3) or sets, unrolled at parse time.
* `#pragma braket ...` and other foreign pragmas, `@annotations`, and
  `// /* */` comments are ignored. `delay` is ignored with warning QM-W051.

Not supported (each is rejected with a QM-E504 hint): subroutines,
`while`, runtime classical arithmetic, multi-bit conditions (QM-E541), and
pulse-level `cal`/`defcal`.

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
  E2 functors, E3 Pauli, E4 simulation and execution, E5 QASM, E6 compiler,
  E7 stabilizer and QEC, E8 algorithms, E9 interface (JSON, MCP, CLI),
  W0 lint. See [ERRORS.md](ERRORS.md).
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
* **errors**: the full code catalog, including codes from other modules.
* **capabilities**: conventions (qubit order, bitstring order, angle
  convention, seeding), simulator methods and memory model, the supported and
  unsupported QASM subset, error ranges, and the live list of MCP tools.
* **Circuit JSON** (`qumo.circuit.v1`): `circuit_to_json` and
  `circuit_from_json` round-trip. Every tool accepts `circuit` (JSON) instead
  of `qasm`, for agents that prefer structured input.
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
| `qumo_run_qasm` | `qasm` \| `circuit`, `shots`=1024, `seed`=1234, `params`, `max_qubits`=28 | `counts` (sorted), `method` (`statevector:static` or `:trajectories`), `most_frequent`, `warnings`, `timing_ms`. Circuits without measurements are measured on all qubits (QM-W004). |
| `qumo_statevector` | `qasm`, `params`, `max_amplitudes`=64, `threshold`, `max_qubits`=24 | `amplitudes[{index, bitstring, re, im, probability}]`, `num_nonzero`, `truncated`, `norm` |
| `qumo_expectation` | `qasm`, `observable` (string or array), `params` | `expectation`, `results[]` |
| `qumo_validate_qasm` | `qasm` | `valid`, `errors[]`, `warnings[]` (lint), `summary` |
| `qumo_describe` | `topic`: gates \| gate \| errors \| error \| capabilities, `name` | self-description JSON |
| `qumo_circuit_info` | `qasm`, `include_instructions`, `include_qasm`, `include_lint` | `summary`, `lint`, and optionally `circuit` and `qasm` |
| `qumo_convert` | `qasm` \| `circuit`, `to`: qasm3 \| json | canonical QASM 3 (also returned as plain text content) or circuit JSON |

`params` binds `input float` parameters. Pass either an array in declaration
order or an object such as `{"theta": 0.1}`.

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
qumo run <file|-> [--shots N] [--seed S] [--params JSON] [--json]
qumo statevector <file|-> [--max-amplitudes N] [--params JSON]
qumo expect <file|-> "0.5*Z0 Z1 - X0" [more...] [--params JSON]
qumo validate <file|->                 # exit 1 when invalid
qumo info <file|-> [--qasm] [--instructions]
qumo describe [gates|gate <name>|errors|error <code>|capabilities]
qumo convert <file|-> [--to qasm3|json] [-o out]
qumo tool list | qumo tool <name> '<json-args>'|-
qumo mcp | version | help [command]
```

The CLI prints human-readable output by default. With `--json` it prints the
tool's `structuredContent` byte for byte, and `--pretty` indents it. Exit codes
are 0 for success, 1 when the operation failed (the diagnostic goes to stderr,
or to stdout as JSON with `--json`), and 2 for usage errors (QM-E921/QM-E912).
Use `-` as the file argument to read from stdin.

---

## 7. Extending: register tools and commands

**A new MCP tool** (it appears automatically in `tools/list`, in
`qumo tool <name>`, in `qumo tool list` and in `describe capabilities`):

```mojo
# qumo/compiler/tools.mojo  (owned by the compiler module)
from qumo.ai.tools import ToolRegistry, ToolResult, ToolContext, tool_ok, arg_circuit, arg_int
from qumo.ai.json import JsonValue, JsonWriter
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

Then add one line to `default_registry()` in `qumo/ai/tools.mojo`:

```mojo
    register_compiler_tools(reg)   # placeholders for compiler/qec/algorithms are already there
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

**A new CLI command:** append a
`Command("name", "summary", "usage", handler)` to `command_table()` in
`cli/main.mojo`, where `handler` is `def(CliArgs) raises thin -> Int`. Most
commands build a JSON argument object, call `call_tool(...)`, and format the
structured result. Every registered tool is also reachable without a dedicated
command:

```bash
qumo tool qumo_transpile '{"qasm": "...", "target": "grid:3x3"}' --pretty
```

**New error codes:** add an `_e(c, "QM-E6xx", ...)` line to `error_catalog()`
in `qumo/ai/diagnostics.mojo`, then regenerate `docs/ERRORS.md` from
`qumo describe errors --json`.

---

## 8. Limitations and next steps

* The IR flattens registers and keeps no register names, so emitted QASM
  always uses a single `q` and a single `c`. Barriers are global.
* Conditions test one classical bit. Multi-bit register comparisons, nested
  ifs and compound conditions are rejected with QM-E541.
* `ctrl @` covers the gates that the IR's `controlled()` functor supports
  (everything needed for QPE and Hadamard tests). `ctrl @ ccx` and
  `ctrl @ U` with symbolic angles raise QM-E202, located at the QASM line.
* Parameters are affine in a single input (by design: binding is
  allocation-free).
* Only QASM 3 is emitted. There is no QASM 2 export yet.
* Tools run the dense state-vector simulator, with Clifford circuits
  recommended for `qumo.sim.stabilizer` in the summary. Stabilizer, QEC,
  compiler and algorithm tools are expected to come from their modules
  through the registry hook described above.
