# QUMO error and warning codes

Every failure in QUMO is reported with a **stable code**. Codes never change
meaning; message wording may improve. Agents should branch on the code,
show the message, and apply the hint.

```
QM-E510: unknown gate 'cnto' (line 6, col 1). Hint: did you mean 'cnot'?
 6 | cnto q[0], q[1];
   | ^~~~
```

* **Text form** (any `Error` raised by QUMO): `QM-Exxx: <message> [(line L, col C)]. [Hint: <fix>]`,
  optionally followed by a source snippet with a caret.
* **Structured form** (CLI `--json`, MCP `structuredContent.error`, `qumo_validate_qasm`):
  `qumo.ai.diagnostics.diagnostic_from_error(text)` ->
  `{"code", "severity", "message", "hint", "line", "col", "snippet", "instruction", "docs"}`
  (`line`/`col`/`snippet` only for source errors, `instruction` only for lint findings).
* Errors without a QUMO code (e.g. from the Mojo standard library) map to `QM-E999`.
* `QM-Wxxx` are lint findings from `qumo.ai.diagnostics.lint()` / the QASM parser.
  Severity is `warning` or `info` (QM-W014 is reported with severity `error`).

Machine-readable catalog: `qumo describe errors --json`, or the MCP tool
`qumo_describe {"topic": "errors"}` (single code: `{"topic": "error", "name": "QM-E510"}`).
This file is generated from that catalog (`qumo.ai.diagnostics.error_catalog()`).

## Code ranges

| Range | Owner |
|---|---|
| `QM-E0xx` | Circuit structure (qumo.ir.circuit.validate) |
| `QM-E1xx` | Symbolic parameters and binding |
| `QM-E2xx` | inverse()/controlled()/power() transforms |
| `QM-E3xx` | Pauli strings / observables |
| `QM-E4xx` | State-vector simulation and execution (qumo.sim, qumo.exec) |
| `QM-E5xx` | OpenQASM 2/3 parsing (qumo.ir.qasm) |
| `QM-E6xx` | Compiler passes, pass pipelines and targets (qumo.compiler) |
| `QM-E7xx` | Stabilizer / Pauli-frame simulation and QEC (qumo.sim.stabilizer, qumo.qec) |
| `QM-E8xx` | Algorithms library (qumo.algorithms) |
| `QM-E9xx` | JSON, MCP tools and CLI (qumo.ai, cli); QM-E999 = uncategorized |
| `QM-W0xx` | Lint warnings from qumo.ai.diagnostics.lint |

Adding a code: raise `Error("QM-Exxx: <message>. Hint: <fix>.")` from your
module (inside your range), then add one `_e(c, ...)` line to
`error_catalog()` in `qumo/ai/diagnostics.mojo` and regenerate this file.

## QM-E0xx — Circuit structure

### QM-E001

**An instruction uses a qubit index outside the circuit.**  
Fix: Valid qubits are 0..num_qubits-1; enlarge the register or fix the index.

### QM-E002

**A multi-qubit gate repeats the same qubit.**  
Fix: Each operand of a gate must be a distinct qubit.

### QM-E003

**A condition refers to a classical bit beyond the classical register.**  
Fix: Declare enough classical bits or fix the condition bit.

## QM-E1xx — Parameters

### QM-E101

**Number of parameter values does not match the circuit's symbolic parameters.**  
Fix: Pass one value per parameter, in circuit.param_names order (or a name->value object).

### QM-E102

**Operation needs a circuit without unbound parameters.**  
Fix: Call circuit.bind(values) first.

## QM-E2xx — Functors (inverse / controlled / power)

### QM-E201

**inverse() on a circuit with non-unitary operations.**  
Fix: Remove measurements/resets/noise or split the circuit.

### QM-E202

**controlled() does not support a gate, or needs bound parameters.**  
Fix: Bind parameters, or translate to the {u, cx} basis first.

### QM-E203

**The requested control qubit is already used by the circuit.**  
Fix: Pick a fresh control qubit.

## QM-E3xx — Pauli observables

### QM-E301

**Malformed Pauli expression.**  
Fix: Write e.g. '0.5*Z0 Z1 - 1.2*X0' or dense labels '0.5*ZZI'.

### QM-E302

**Pauli index >= 64.**  
Fix: QUMO Pauli strings are bit-packed into 64 qubits.

### QM-E303

**Dense Pauli labels of different lengths.**  
Fix: Pad labels with I to the same length.

## QM-E4xx — Simulation and execution

### QM-E401

**evolve()/statevector cannot apply classically-conditioned gates.**  
Fix: Use sample() (trajectories) for dynamic circuits.

### QM-E402

**Sampling a circuit without measurements.**  
Fix: Add measurements (measure_all) or use statevector/expectation.

### QM-E403

**Estimator task failed for at least one parameter set.**  
Fix: Inspect the nested error of the failing parameter set.

### QM-E404

**State vector too large for the requested operation.**  
Fix: Reduce the number of qubits or use a stabilizer method.

### QM-E450

**Invalid physical error rate.**  
Fix: Probabilities must lie in [0, 1].

### QM-E460

**No devices available in the execution pool.**  
Fix: Add Device.cpu() at least.

### QM-E461

**Job failed or is unknown.**  
Fix: Check the job id and the nested error.

### QM-E462

**Job state does not allow the operation.**  
Fix: Wait for completion or resubmit.

## QM-E5xx — OpenQASM parsing

### QM-E500

**OpenQASM syntax error (unexpected or missing token).**  
Fix: Check the token at line:col; every statement ends with ';'.

### QM-E501

**Unexpected character in OpenQASM source.**  
Fix: Remove it or check for a missing quote/comment marker.

### QM-E502

**Unterminated string literal or block comment.**  
Fix: Close strings on the same line and comments with */.

### QM-E503

**Unsupported OpenQASM version.**  
Fix: Use 'OPENQASM 3.0;' or 'OPENQASM 2.0;'.

### QM-E504

**OpenQASM feature outside QUMO's supported subset (subroutines, while, classical assignment, ...).**  
Fix: Use the gate-level subset; see 'qumo describe capabilities'.

### QM-E510

**Unknown gate name.**  
Fix: Check the suggestions, run 'qumo describe gates', or define it with 'gate name(params) a, b { ... }'.

### QM-E511

**Wrong number of gate parameters.**  
Fix: Match the gate signature, e.g. rz(theta) q[0];

### QM-E512

**Wrong number of qubit operands (including ctrl @ controls).**  
Fix: Match the gate arity; controls of ctrl @ come first.

### QM-E513

**A gate uses the same qubit twice.**  
Fix: Each operand must be a distinct qubit.

### QM-E514

**Broadcast over registers of different sizes.**  
Fix: Broadcast operands must have equal size, or index single qubits.

### QM-E520

**Undeclared or wrong-kind identifier (register, parameter, gate argument).**  
Fix: Declare it before use; check qubit vs bit registers.

### QM-E521

**Register index out of range.**  
Fix: Valid indices are 0..size-1; slices [a:b] are inclusive.

### QM-E522

**Name declared twice.**  
Fix: Use unique names for registers, inputs, constants and gates.

### QM-E523

**Invalid register size.**  
Fix: Register sizes must be positive integer constants.

### QM-E530

**A constant was required (index, size, exponent, gphase) but the expression is not constant/integer.**  
Fix: Use numeric literals, const values or loop variables.

### QM-E531

**Parameter expression is not affine (product/quotient of inputs).**  
Fix: QUMO parameters are coeff*input + offset; bind values or restructure.

### QM-E532

**Parameter expression mixes two different inputs.**  
Fix: Use one input per gate angle (split into two gates if needed).

### QM-E533

**Math domain error (division by zero, log/sqrt of invalid value).**  
Fix: Check the expression.

### QM-E534

**Unknown function or wrong arity in an expression.**  
Fix: Supported: sin cos tan arcsin arccos arctan exp ln sqrt abs floor ceil pow mod.

### QM-E540

**Measurement maps a different number of qubits and bits.**  
Fix: Qubit and bit operands must have equal size.

### QM-E541

**Unsupported condition (multi-bit register value, nested or compound if).**  
Fix: QUMO conditions test one bit: if (c[i] == 1) ...

### QM-E550

**Invalid gate modifier (non-integer pow, bad ctrl count).**  
Fix: Use inv @, ctrl(n) @, negctrl @ or pow(k) @ with integer k.

### QM-E551

**Gate definition expands recursively.**  
Fix: Gates must not call themselves.

### QM-E560

**Invalid '#pragma qumo' directive.**  
Fix: See docs/AI_NATIVE.md for detector/observable/tick/param pragmas.

### QM-E570

**Invalid or too large for-loop range.**  
Fix: Use constant inclusive ranges [a:b] / [a:step:b] or sets {a, b}.

## QM-E6xx — Compiler

### QM-E600

**Unknown pass name in a pass pipeline.**  
Fix: List passes with the pass manager docs; e.g. 'cancel,commute,fuse1q'.

### QM-E601

**Malformed pass pipeline (unbalanced parentheses / syntax).**  
Fix: Write pipelines like 'fixpoint(cancel,commute),route(sabre)'.

### QM-E602

**Bad pass argument (unknown layout/routing method, fixpoint options).**  
Fix: Check the pass argument names and values.

### QM-E603

**Basis translation impossible (no usable 2-qubit entangler or decomposition rule).**  
Fix: Include an entangler (cx, cz, ecr, iswap, rxx, rzz) in the target basis.

### QM-E604

**A 1-qubit gate cannot be expressed in the target basis.**  
Fix: Add rz+sx, u, or rx+ry style 1-qubit gates to the basis.

### QM-E605

**Circuit is wider than the target.**  
Fix: Pick a larger target or reduce the circuit width.

### QM-E606

**Interacting qubits are in disconnected parts of the coupling map.**  
Fix: Use a connected target or a layout that keeps interacting qubits together.

### QM-E607

**Routing needs gates on at most 2 qubits.**  
Fix: Run 'unroll3q' (or 'basis') before 'route'.

### QM-E609

**Optimization level out of range.**  
Fix: Use 0..3.

### QM-E610

**Unknown gate name in a basis specification.**  
Fix: Use gate names like rz,sx,x,cx,cz,ecr or a basis preset.

### QM-E611

**Invalid target specification (topology/size).**  
Fix: Use '<topology>:<size>', e.g. 'grid:4x4', 'line:20', 'heavy_hex:5'.

### QM-E612

**Invalid coupling map (edges, distance, qubit count).**  
Fix: Edges are pairs of distinct qubits; heavy_hex distance is odd.

### QM-E615

**Compiled circuit does not conform to the target (basis or connectivity).**  
Fix: Add 'route' and 'basis' passes before 'check', or use a preset level.

## QM-E7xx — Stabilizer simulation and QEC

### QM-E701

**Non-Clifford operation in a stabilizer-only path.**  
Fix: Use the state-vector simulator, or keep rotations at multiples of pi/2.

### QM-E702

**Unbound parameters in a stabilizer/Stim export path.**  
Fix: Bind parameters first.

### QM-E703

**Detectors/observables are not deterministic in the noiseless circuit.**  
Fix: Check the measurement records referenced by detectors.

### QM-E704

**Unsupported feed-forward for the stabilizer/frame simulator.**  
Fix: Only Pauli X/Y/Z conditioned on one measured bit is supported.

### QM-E705

**Register too large for the requested output.**  
Fix: Reduce clbits/observables or use another output format.

### QM-E706

**Invalid QEC code parameters.**  
Fix: Check code name, distance, rounds and basis.

### QM-E708

**Decoding failure.**  
Fix: Check the detector error model / syndrome size.

### QM-E711

**Invalid noise probabilities.**  
Fix: Probabilities must lie in [0, 1].

## QM-E8xx — Algorithms

### QM-E801

**Invalid algorithm option.**  
Fix: Check the option names/values in the function docstring.

### QM-E802

**Invalid size or index for an algorithm builder.**  
Fix: Check qubit counts and indices.

### QM-E803

**Hamiltonian/operator does not fit the register.**  
Fix: Match observable qubits to the ansatz qubits.

### QM-E804

**Parameter vector has the wrong length or index.**  
Fix: Pass one value per circuit parameter.

### QM-E805

**Circuit not supported by the chosen gradient/analysis method.**  
Fix: Use parameter-shift or bind the unsupported gates.

### QM-E806

**Unknown optimizer or gradient method.**  
Fix: Use one of the documented method names.

### QM-E807

**Problem too large for an exact/brute-force routine.**  
Fix: Reduce the size or use a variational method.

### QM-E808

**Invalid Trotterization options.**  
Fix: steps >= 1, order 1 or 2.

### QM-E809

**Invalid graph specification.**  
Fix: Edges are 'i-j' pairs of distinct non-negative nodes.

### QM-E810

**Invalid QPE options.**  
Fix: 1 <= n_eval <= 16 and a unitary on >= 1 qubit.

### QM-E820

**Invalid multi-controlled gate construction.**  
Fix: Check control/target/ancilla counts.

## QM-E9xx — Interface: JSON, MCP, CLI

### QM-E900

**Malformed JSON.**  
Fix: Fix the JSON at the reported line/col (double-quoted keys, no trailing commas).

### QM-E901

**Invalid JSON-RPC request.**  
Fix: Send {"jsonrpc":"2.0","id":1,"method":...,"params":{...}}.

### QM-E910

**Missing required tool argument.**  
Fix: See the tool's inputSchema in tools/list.

### QM-E911

**Tool argument has the wrong JSON type.**  
Fix: See the tool's inputSchema in tools/list.

### QM-E912

**Unknown tool or command.**  
Fix: List tools with tools/list or 'qumo help'.

### QM-E913

**Invalid argument value (out of range, unknown name, unknown topic).**  
Fix: Check allowed values in the inputSchema / 'qumo help <command>'.

### QM-E914

**Observable acts on qubits outside the circuit.**  
Fix: Use Pauli indices < num_qubits.

### QM-E920

**Cannot read input file.**  
Fix: Check the path; use '-' to read from stdin.

### QM-E921

**Invalid command-line usage.**  
Fix: Run 'qumo help'.

### QM-E930

**Circuit too large for dense state-vector output.**  
Fix: Use sampling, fewer qubits, or raise max_qubits explicitly.

### QM-E999

**Uncategorized internal error (no QUMO code in the message).**  
Fix: Report it with the input that triggered it.

## QM-W0xx — Lint warnings

### QM-W001

**Qubits are declared but never used.**  
Fix: Shrink the register to save simulation memory (2x per qubit).

### QM-W002

**A gate acts on a qubit after it was measured (no reset): the circuit becomes dynamic.**  
Fix: Intended for mid-circuit measurement; otherwise move measurements to the end.

### QM-W003

**A qubit is measured without any prior operation (always 0).**  
Fix: Check the qubit index of the measurement.

### QM-W004

**The circuit has no measurements.**  
Fix: Sampling tools will measure all qubits; add measurements to control the readout.

### QM-W005

**The circuit has unbound symbolic parameters.**  
Fix: Pass 'params' when running it.

### QM-W006

**Dense state-vector simulation needs a lot of memory.**  
Fix: 16 bytes * 2^n: use a stabilizer method for Clifford circuits or fewer qubits.

### QM-W007

**A classical bit is overwritten before being read.**  
Fix: Use distinct bits or read (condition on) the bit first.

### QM-W008

**A condition reads a classical bit that was never written (always 0).**  
Fix: Measure into the bit before conditioning on it.

### QM-W009

**Classical bits are never written.**  
Fix: Shrink the bit register or add measurements.

### QM-W010

**Adjacent gates cancel each other.**  
Fix: Remove both or run the optimizer (qumo.compiler).

### QM-W011

**Rotation by angle 0 (no-op).**  
Fix: Remove the gate.

### QM-W012

**Very deep circuit.**  
Fix: Consider optimization passes before simulation.

### QM-W014

**Noise probability outside [0, 1].**  
Fix: Probabilities must lie in [0, 1].

### QM-W050

**OpenQASM gate definition shadows a native gate with a different signature.**  
Fix: Rename the gate.

### QM-W051

**OpenQASM delay is ignored.**  
Fix: Model idle noise with '#pragma qumo depolarize1(p) q[i];'.
