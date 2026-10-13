# QUMO error and warning codes

<!-- GENERATED FILE, do not edit by hand. Source of truth: error_catalog() and
     error_ranges() in qumo/ai/diagnostics.mojo. Regenerate with
     QUMO_UPDATE_DOCS=1 .venv/bin/mojo run -I . tests/test_mcp.mojo
     (or build/qumo tool qumo_describe '{"topic":"errors","format":"markdown"}').
     tests/test_mcp.mojo fails when this file is stale. -->

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
* `QM-Wxxx` are warnings: lint findings from `qumo.ai.diagnostics.lint()`, execution notes
  in tool results (e.g. QM-W020, GPU fallback) and OpenQASM parser warnings. Severity is
  `warning` or `info` (QM-W014 is reported with severity `error`).
* QM-E95x (hardware plugins) and QM-E96x (benchmark harness) are raised by the Python tools
  of this repository, with the same text form and JSON shape.

Machine-readable catalog: `qumo describe errors --json`, or the MCP tool
`qumo_describe {"topic": "errors"}` (single code: `{"topic": "error", "name": "QM-E510"}`).

## Code ranges

| Range | Category | Owner |
|---|---|---|
| `QM-E0xx` | circuit | Circuit structure (qumo.ir.circuit: validate, builders) |
| `QM-E1xx` | parameters | Symbolic parameters and binding |
| `QM-E2xx` | functors | inverse()/controlled()/power() transforms |
| `QM-E3xx` | pauli | Pauli strings / observables (qumo.ir.pauli) |
| `QM-E4xx` | simulation | CPU state-vector simulation, sampling and execution (qumo.sim, qumo.exec) |
| `QM-E47x` | gpu | GPU state vector and GPU Pauli frames (qumo.gpu; builds made with -D QUMO_GPU) |
| `QM-E5xx` | qasm | OpenQASM 2/3 parsing and export (qumo.ir.qasm, qumo.ir.qasm2) |
| `QM-E6xx` | compiler | Compiler passes, pass pipelines and targets (qumo.compiler) |
| `QM-E7xx` | stabilizer | Stabilizer / Pauli-frame simulation and QEC (qumo.sim.stabilizer, qumo.sim.frame, qumo.qec) |
| `QM-E8xx` | algorithms | Algorithms library (qumo.algorithms) |
| `QM-E9xx` | interface | JSON, MCP tools, CLI and device selection (qumo.ai, qumo.exec.tools, cli); QM-E999 = uncategorized |
| `QM-E95x` | hardware | Hardware plugins (Python package plugins/hardware: IBM Quantum via Qiskit Runtime, Quantinuum via pytket/qnexus) |
| `QM-E96x` | benchmark | Benchmark harness (bench/run_benchmarks.py, Python) |
| `QM-W0xx` | warning | Warnings: lint findings QM-W001..W019 (qumo.ai.diagnostics.lint), execution notes QM-W020..W049, OpenQASM parser warnings QM-W050..W059 |

A code belongs to the narrowest range that contains it (QM-E472 is in QM-E47x, not QM-E4xx).

Adding a code: raise `Error("QM-Exxx: <message>. Hint: <fix>.")` from your module
(inside your range), add one `_e(c, ...)` line to `error_catalog()` in
`qumo/ai/diagnostics.mojo`, then regenerate this file (command in the header comment).
`make test` fails while a raised code is missing from the catalogue or this file is stale.

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

### QM-E004

**A register condition (Circuit.c_if_register, OpenQASM 2 'if(c==N)') needs matching non-empty bit and value lists.**  
Fix: Pass one value (0 or 1) per condition bit.

## QM-E1xx — Parameters

### QM-E101

**Number of parameter values does not match the circuit's symbolic parameters.**  
Fix: Pass one value per parameter, in circuit.param_names order (or a name->value object).

### QM-E102

**Operation needs a circuit without unbound parameters (e.g. OpenQASM 2 export).**  
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

### QM-E405

**Observable acts on a qubit outside the state.**  
Fix: Check Pauli indices (0-based, qubit 0 = least significant).

### QM-E406

**The sampler packs classical registers into 62 bits; the circuit has more clbits.**  
Fix: Measure fewer bits or split the register.

### QM-E407

**Invalid index-bit swap layer (StateVector.apply_swap_layer): the qa/qb lists differ in length, a bit is outside [0, num_qubits), or a bit appears twice.**  
Fix: Pass disjoint pairs of distinct in-range bits, one partner per entry.

### QM-E408

**Invalid cache-block size for execute_plan_reordered (need 1 <= nb <= num_qubits).**  
Fix: Use execute_plan(), or pass nb in [1, num_qubits].

### QM-E450

**Invalid physical error rate (out of [0, 1], or above the surface-code threshold for fault-tolerant estimates).**  
Fix: Probabilities must lie in [0, 1]; fault-tolerant estimation needs p < 1e-2.

### QM-E460

**No devices available in the execution pool.**  
Fix: Add Device.cpu() at least.

### QM-E461

**A workflow job fits no device in the pool (too many qubits).**  
Fix: Add a device with more qubits, or a stabilizer device for Clifford circuits.

### QM-E462

**A workflow job depends on a job that is not an earlier job.**  
Fix: Add jobs in dependency order; a job may only depend on earlier jobs.

## QM-E47x — GPU acceleration

See [GPU.md](GPU.md). Only GPU entry points raise these codes. The MCP tool `qumo_run_device` and `qumo.exec.dispatch.sample_on(..., "gpu")` never fail because the GPU is missing: they run on the CPU and report warning QM-W020.

### QM-E470

**GPU support is not compiled into this build.**  
Fix: Rebuild with -D QUMO_GPU (make cli-gpu / py-gpu / test-gpu); on macOS install Xcode and run 'xcodebuild -downloadComponent MetalToolchain' first.

### QM-E471

**Dynamic or noisy circuit given to the GPU state-vector sampler (it runs static circuits: terminal measurements, no noise).**  
Fix: Use qumo.exec.dispatch.sample_auto or sample_on, which fall back to CPU trajectories.

### QM-E472

**float64 requested on an Apple GPU (Metal has no double precision).**  
Fix: Use GpuStateVector[DType.float32], the default on Apple.

### QM-E473

**Sampling a GPU state with zero norm.**  
Fix: Check the circuit and the initial state.

### QM-E474

**Circuit or host state is wider than the allocated GPU state.**  
Fix: Allocate GpuStateVector(circuit.num_qubits).

### QM-E475

**The GPU executor has no kernel for an operation of the execution plan.**  
Fix: Run this circuit on the CPU state vector (qumo.sim.runner).

## QM-E5xx — OpenQASM

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

### QM-E552

**OpenQASM export cannot express a classical condition: a single-bit condition on a multi-bit register (OpenQASM 2), or a register condition that does not cover every clbit.**  
Fix: Export OpenQASM 3 (to_qasm) for single-bit conditions; condition on single bits or on the whole register.

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

See [QEC.md](QEC.md).

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

### QM-E707

**Unknown decoder name.**  
Fix: Use 'mwpm' (exact matching by sparse blossom, the default; aliases 'matching', 'pymatching'), 'sparse_blossom' (same decoder), 'mwpm_dense' (dense-blossom MWPM, small codes), 'uf' (union-find) or 'lookup', or pass your own Decoder to sample_and_decode().

### QM-E708

**Decoding failure (reserved; decoders currently count isolated defects in a failures field).**  
Fix: Check the detector error model / syndrome size.

### QM-E709

**Invalid calibration data or noise model.**  
Fix: Check T1/T2 > 0, T2 <= 2*T1, error probabilities in [0, 1].

### QM-E710

**Exponential/decay fit failed.**  
Fix: Use more sequence lengths or shots; check the data decays.

### QM-E711

**Invalid noise probabilities.**  
Fix: Probabilities must lie in [0, 1].

### QM-E712

**Internal invariant violation in the sparse-blossom decoder (never raised per shot; counted in SparseBlossomDecoder.anomalies, which must stay 0).**  
Fix: Report the syndrome; decode it with 'mwpm_dense' meanwhile.

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

## QM-E9xx — Interface

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

### QM-E940

**Unknown simulator device name (qumo.exec.dispatch.sample_on, qumo_run_device).**  
Fix: Use 'auto', 'cpu' or 'gpu'.

### QM-E999

**Uncategorized internal error (no QUMO code in the message).**  
Fix: Report it with the input that triggered it.

## QM-E95x — Hardware plugins

Raised by the Python package `qumo_hardware`, not by the Mojo core. This block mirrors `CATALOG` in plugins/hardware/qumo_hardware/errors.py (tests/test_mcp.mojo checks that every code used there is listed here). See [HARDWARE_PLUGINS.md](HARDWARE_PLUGINS.md#error-codes); `python -m qumo_hardware errors --json` prints the plugin's own catalogue. Messages never contain credentials.

### QM-E950

**The QUMO Python extension (build/qumo_py.so) could not be loaded.**  
Fix: Run `make py` at the repository root, or set QUMO_PY_PATH to the directory containing qumo_py.so.

### QM-E951

**A provider SDK needed for this operation is not installed.**  
Fix: Install the extra: pip install 'qumo-hardware[ibm]' or 'qumo-hardware[quantinuum]'.

### QM-E952

**Provider credentials are missing, expired or rejected.**  
Fix: IBM: export QISKIT_IBM_TOKEN (and QISKIT_IBM_INSTANCE) or QiskitRuntimeService.save_account(); Quantinuum: run qnexus.login() once.

### QM-E953

**Unknown provider or backend name, or the backend is not available to this account.**  
Fix: List backends with `python -m qumo_hardware backends --provider <name>`.

### QM-E954

**The circuit cannot be compiled for the backend (too many qubits, unsupported operation, parse error).**  
Fix: Reduce the circuit width or check the circuit with `build/qumo validate`.

### QM-E955

**The compiled payload failed the provider's own validation (not ISA / not native).**  
Fix: Report the circuit; retry with --provider-transpile to compare with the provider's compiler.

### QM-E956

**A shot, size or cost limit would be exceeded.**  
Fix: Lower --shots, or raise --max-cost if the cost is intended.

### QM-E957

**Submission to a remote (billable) backend needs explicit confirmation.**  
Fix: Review backend, shots and cost estimate, then re-run with --yes (CLI) or confirm=True (API).

### QM-E958

**Unknown job id (not in the local job store and not found at the provider).**  
Fix: List known jobs with `python -m qumo_hardware jobs`.

### QM-E959

**The job failed, was cancelled, or its result is not available yet.**  
Fix: Check `python -m qumo_hardware status <job>`; results exist only for completed jobs.

## QM-E96x — Benchmark harness

Raised by the Python benchmark harness, not by the Mojo core. Reported in the result JSON (`suites.<name>.reason`) and on stderr; a suite failure never aborts the run. See [BENCHMARKING_ON_CLOUD_GPUS.md](BENCHMARKING_ON_CLOUD_GPUS.md).

### QM-E960

**Unknown or invalid config, preset, suite or option.**  
Fix: Run `python bench/run_benchmarks.py --list`.

### QM-E961

**Mojo build of a benchmark binary failed.**  
Fix: Run the recorded build command by hand; GPU toolchains: docs/GPU.md.

### QM-E962

**A suite, command or build exceeded its timeout (process group killed).**  
Fix: Raise `timeout_s` in the config or use `--quick` / smaller `qubits`.

### QM-E963

**Required tool or binary missing (mojo, make, ...).**  
Fix: Run `make setup`, or install the tool.

### QM-E964

**Benchmark output could not be parsed.**  
Fix: Inspect `output_tail` in the JSON; the binary's output format changed.

### QM-E965

**Third-party baseline not installed or unusable (CUDA-Q, Aer GPU, Lightning).**  
Fix: Install it in the interpreter named by the config / `QUMO_*_PYTHON`.

### QM-E966

**Qubit width dropped: it does not fit `memory_fraction` of GPU memory or free host RAM.**  
Fix: Use a larger GPU / High-RAM runtime, or lower `qubits`.

### QM-E967

**No GPU, a GPU outside the Mojo/MAX support list, or unsupported precision (fp64 on Metal).**  
Fix: Check `nvidia-smi` / `rocminfo`; `QUMO_FORCE_GPU=1` to try anyway.

### QM-E968

**A benchmark binary (e.g. a GPU runtime error) or a baseline worker crashed or produced no result.**  
Fix: Rerun the recorded command; for GPU errors add `MODULAR_DEBUG=device-sync-mode`.

### QM-E969

**A test, `gpu-check` target or correctness check reported FAIL.**  
Fix: Rerun the failing file: `.venv/bin/mojo run -I . <file>`.

## QM-W0xx — Warnings

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

### QM-W020

**Device 'gpu' was requested but the run fell back to the CPU; the warning gives the reason (GPU support not compiled, no usable device, or a dynamic/noisy non-Clifford circuit).**  
Fix: Check qumo_gpu_info; use a build made with -D QUMO_GPU (make cli-gpu / py-gpu) on a supported GPU, or request device 'auto' / 'cpu'.

### QM-W050

**OpenQASM gate definition shadows a native gate with a different signature.**  
Fix: Rename the gate.

### QM-W051

**OpenQASM delay is ignored.**  
Fix: Model idle noise with '#pragma qumo depolarize1(p) q[i];'.
