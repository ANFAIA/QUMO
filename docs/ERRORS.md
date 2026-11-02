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
| `QM-W0xx` | warning | Warnings: lint findings QM-W001..W019 (qumo.ai.diagnostics.lint), execution notes QM-W020..W049, OpenQASM parser warnings QM-W050..W053 and QM-W055, Program lint QM-W054 |

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

### QM-E005

**Unknown circuit drawing style.**  
Fix: Use 'ascii' or 'unicode'.

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

### QM-E204

**A multi-controlled gate uses a qubit twice among its controls, target and ancillas.**  
Fix: Controls, target and ancillas must all be distinct.

### QM-E205

**Unknown multi-controlled synthesis mode.**  
Fix: Use auto, noaux, clean, dirty or split.

### QM-E206

**The multi-controlled synthesis mode needs more ancillas than were given.**  
Fix: Pass more ancillas, or use mode 'auto'.

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

### QM-E420

**split_pub needs a positive limit.**  
Fix: Pass the device limit on parameter sets per job.

### QM-E409

**sample_ptsbe needs a static circuit with Pauli noise: no resets, classical conditions, mid-circuit measurements or Kraus channels.**  
Fix: Use sample(), which runs per-shot trajectories.

### QM-E410

**Density matrix too large: 2^n x 2^n complex numbers.**  
Fix: The limit is 15 qubits (16 GiB). Use sample_ptsbe or sample() for more.

### QM-E411

**The density-matrix simulator cannot follow one classical branch: classical conditions, or sampling after a mid-circuit measurement.**  
Fix: Use sample(), which runs per-shot trajectories.

### QM-E412

**Stabilizer branching exceeded its branch limit (2^20): too many non-Clifford gates on non-commuting Paulis.**  
Fix: Reduce the number of non-Clifford gates, or use the state vector.

### QM-E413

**The stabilizer branching simulator has no decomposition for this gate.**  
Fix: Translate the circuit to Clifford gates and rotations first.

### QM-E414

**Invalid or unsupported Kraus channel: wrong shape, not trace preserving, or used on a simulator that needs Pauli noise.**  
Fix: Pass matrices with sum K^dag K = identity; run them on density_evolve() or sample().

### QM-E408

**Invalid cache-block size for execute_plan_reordered (need 1 <= nb <= num_qubits).**  
Fix: Use execute_plan(), or pass nb in [1, num_qubits].

### QM-E480

**A shot of a Program passed its budget of loop iterations (max_iterations, default 65536 back-edges per shot).**  
Fix: The loop may never exit; raise max_iterations if it is only long.

### QM-E481

**Division or remainder by zero, or an integer power with a negative exponent, while a shot of a Program ran. The message gives the shot and line.**  
Fix: Guard the divisor with an if.

### QM-E482

**A float reached an integer or a gate angle as NaN, infinite or out of range of the target type, an input value is not finite, or a for-loop range holds a uint at or above 2^63 that it cannot read.**  
Fix: Guard the value, or use a wider integer.

### QM-E483

**A simulation method that cannot run a Program ('density', 'ptsbe', 'frame': they need one fixed sequence of operations for all shots), or an unknown method name.**  
Fix: Use 'auto', 'statevector', 'tableau' or 'branching'.

### QM-E484

**An opcode that a Program block cannot run: blocks take unitary gates, measure, reset, noise, barrier and tick.**  
Fix: Report it: every other walker of the instruction list would skip such an opcode in silence.

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

### QM-E463

**A workflow job is pinned (add_sample / add_estimate device=..., qumo_py.schedule job 'device') to a device name that is not in the pool.**  
Fix: Use the name of a Device in the pool passed to plan()/run(), or device="" to let HEFT choose.

### QM-E464

**A workflow job is pinned to a device that cannot run it (more qubits than max_qubits, or a non-Clifford circuit on a Clifford-only device).**  
Fix: Pin it to a larger / general device, or unpin it so HEFT places it.

## QM-E47x — GPU acceleration

See [GPU.md](GPU.md). Only GPU entry points raise these codes. The MCP tool `qumo_run_device` and `qumo.exec.dispatch.sample_on(..., "gpu")` never fail because the GPU is missing: they run on the CPU and report warning QM-W020. They do fail with QM-E472 when precision 'fp64' is requested on an Apple GPU, which has no fp64.

### QM-E470

**GPU support is not compiled into this build.**  
Fix: Rebuild with -D QUMO_GPU (make cli-gpu / py-gpu / test-gpu); on macOS install Xcode and run 'xcodebuild -downloadComponent MetalToolchain' first.

### QM-E471

**Dynamic or noisy circuit given to the GPU state-vector sampler (it runs static circuits: terminal measurements, no noise).**  
Fix: Use qumo.exec.dispatch.sample_auto or sample_on, which fall back to CPU trajectories.

### QM-E472

**float64 requested on an Apple GPU (Metal has no double precision): precision 'fp64' (argument or QUMO_GPU_PRECISION) or GpuStateVector[DType.float64].**  
Fix: Use precision 'fp32' or 'auto' (GpuStateVector[DType.float32], the default on every GPU), or the CPU simulators, which run in fp64.

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

**OpenQASM feature outside QUMO's supported subset, or one a Circuit cannot hold: subroutines, or, in parse_qasm, while and classical assignment.**  
Fix: Parse with parse_qasm_program or load_any for while and classical variables; see 'qumo describe capabilities' for the rest.

### QM-E505

**An expression or a block nests deeper than 256 levels.**  
Fix: Split it into smaller expressions or statements; a chain of else-if arms counts as nesting too.

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

**Unsupported condition: nested if, '||', 'else' after a register or '&&' test, a body that writes a bit it still tests, or a switch on a copy of bits measured since.**  
Fix: Conditions test a bit, a whole register or an '&&' chain of bits; keep writes to tested bits at the end of the body.

### QM-E550

**Invalid gate modifier (non-integer pow, bad ctrl count).**  
Fix: Use inv @, ctrl(n) @, negctrl @ or pow(k) @ with integer k.

### QM-E551

**Gate definition expands recursively.**  
Fix: Gates must not call themselves.

### QM-E552

**OpenQASM export cannot express something: a single-bit condition on a multi-bit register (OpenQASM 2), a register condition that does not cover every clbit, or a constant that is NaN or infinite.**  
Fix: Export OpenQASM 3 (to_qasm) for single-bit conditions; condition on single bits or on the whole register; compute a non-finite value at run time.

### QM-E580

**Classical type error: a cast the OpenQASM table refuses, or operands that do not mix (int with uint, float with angle, shift of an int).**  
Fix: Cast one operand explicitly; docs/CONTROL_FLOW.md has the rules.

### QM-E581

**A classical variable may be read before it is assigned, on some path through the program.**  
Fix: Assign it on every path, for example where it is declared. Bits start at 0 and are exempt.

### QM-E582

**Classical type not representable: a width outside 1..64, float[32], or a mutable angle without a width.**  
Fix: float is binary64; write angle[n] for a variable.

### QM-E583

**Malformed control flow: break or continue outside a loop, else or case without its opener, or a region left open.**  
Fix: Check the nesting of the block.

### QM-E584

**A value known only at run time where a constant is needed: qubit index, size, modifier, const initializer.**  
Fix: Use a constant, or a for loop that can unroll.

### QM-E585

**A Program with classical variables or runtime control flow was given to an operation that needs one Circuit: exact state vector, expectation value, density matrix, OpenQASM 2, resource estimates, device emulation, encoding.**  
Fix: Sample it with qumo_run_qasm, or keep the part you need in a straight-line circuit.

### QM-E586

**Detector or observable in a program with runtime control flow: they index measurements as if each ran once, and no decoder takes a syndrome of varying length.**  
Fix: Move the QEC circuit out of the loop.

### QM-E587

**Program JSON (qumo.program.v1) that cannot be read: a wrong format, a missing field, or an index outside its table.**  
Fix: Pass the output of program_to_json or qumo_convert with to='json'; docs/CONTROL_FLOW.md has the format.

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

### QM-E616

**Unknown synthesis method or model name.**  
Fix: method=auto|exact|astar|greedy|policy; list models with describe.

### QM-E617

**AI model file missing, corrupt, or failed checksum/format check.**  
Fix: Set QUMO_MODEL_DIR or model_dir=, run make models-verify.

### QM-E618

**Block too wide for the requested synthesis method.**  
Fix: Lower max_qubits or use method=greedy.

### QM-E619

**Synthesized block not equivalent to the original (strict mode).**  
Fix: Report a bug; non-strict mode silently keeps the original.

### QM-E620

**AI model incompatible with the request (kind, size, topology or gate set).**  
Fix: Use model=auto.

### QM-E621

**Invalid target calibration or angle bounds: wrong number of values, an unknown edge, a gate without one angle, or an empty interval.**  
Fix: One value per qubit or edge; bounds look like rzz[0:pi/2].

### QM-E623

**to_pbc cannot take this circuit: unbound parameters, conditions, resets, noise or QEC annotations.**  
Fix: Bind parameters and pass a circuit of unitary gates and measurements.

### QM-E624

**A circuit with E / correlated_error was given to the compiler: its qubits live in a side table no pass reads.**  
Fix: Compile first, then add noise.

### QM-E622

**A constant angle cannot be brought into the target's angle bounds.**  
Fix: A Pauli rotation needs an interval covering [0, pi/2] up to a period; widen the bounds or drop them.

### QM-E630

**A Program cannot use the AI presets, synthesis passes or the learned router, which need a whole circuit, nor a second route or a relabeling pass inside fixpoint.**  
Fix: Use a preset level, or a pipeline with one route and none of these.

### QM-E631

**A compiled Program failed verification: a different control structure, two layouts meeting on a control-flow edge, or a block that does not equal its source.**  
Fix: Report the program; run a lower optimization level meanwhile.

### QM-E632

**No SWAP network was found to bring a region of a Program back to its entry layout.**  
Fix: Report the program; check that the target's coupling graph is what you meant.

### QM-E633

**A gate with target angle bounds takes an angle known only at run time, which cannot be wrapped into the bounds.**  
Fix: Use a gate without bounds, or drop the bounds.

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

### QM-E720

**Invalid Pauli label or Pauli size mismatch in a code definition (qumo.ft.pauli).**  
Fix: One character I/X/Y/Z per physical qubit (character i = qubit i), optionally prefixed by + or -.

### QM-E721

**Invalid stabilizer-code definition: wrong number of generators or logicals, anticommuting or dependent generators, logicals not commuting with the stabilizers or violating X_i Z_j relations.**  
Fix: List n-k independent commuting generators and k X/Z logicals with canonical commutation; the message names the offending operators.

### QM-E722

**Unknown code name or invalid code parameter (e.g. odd k for an Iceberg code).**  
Fix: Use steane, five_qubit, css_422, noncss_422, iceberg_<even k>, surface_<d>, repetition_<d> (qumo_ft_codes lists them).

### QM-E723

**Circuit not suitable for gadget verification (sizes do not match num_blocks*k / num_blocks*n, non-unitary or non-Clifford ops, conditioned ops, flag that never accepts).**  
Fix: Clifford gadgets: unitary Clifford circuits; state preparations: valid_stabilizer_state_preparation; magic states: valid_magic_state_preparation.

### QM-E724

**Gate or logical op not supported by the chosen QEC architecture (no decomposition into its logical gate set).**  
Fix: Translate to {h, s, t, cx, rz} first, or pick the architecture that implements it (Iceberg: native rotations; Steane: Clifford+T).

### QM-E725

**The architecture ran out of logical blocks (more simultaneously live logical qubits than n_blocks * k).**  
Fix: Build the architecture with a larger n_blocks (SteaneBuilder().build(n_blocks)); magic states also occupy a block while live.

### QM-E726

**Unsupported classically-conditioned logical operation (conditioned T / magic-state injection, measurement, reset, or cross-block conditioned SWAP on Iceberg).**  
Fix: Condition Clifford gates only; make non-Clifford gates unconditional (e.g. conditioned SWAP into a scratch qubit, as the comparator Rz does).

### QM-E727

**Invalid architecture option, QEC policy or malformed logical program (unknown style/cost key, bad factory sizes, use of a consumed logical handle).**  
Fix: Check qumo.ft.arch option names; logical handles come from alloc()/prep_t() and are consumed by measure/free/inject.

### QM-E728

**Invalid Rz-synthesis option (epsilon outside (0, 1), T budget outside [0, 24], bad comparator attempts or angle register).**  
Fix: epsilon 1e-2 with max_t 16 is the default; comparator angle registers hold n + 2 clbits (comparator_register_size).

### QM-E729

**Emulation request not supported (state vector wider than its limit, non-Clifford circuit on the tableau method, non-deterministic program in an error-rate sweep, shots < 1).**  
Fix: Use method 'auto', fewer logical qubits / magic_prep 'nonft' for state-vector runs, or a deterministic program for error-rate sweeps.

### QM-E732

**Invalid PAULI_CHANNEL_2 or correlated error: wrong number of probabilities or letters.**  
Fix: 15 probabilities in Stim's order; one Pauli letter per qubit.

### QM-E733

**ELSE_CORRELATED_ERROR has no detector error model.**  
Fix: Sample it with the Pauli-frame simulator.

### QM-E734

**Stim circuit text could not be imported: unknown or unsupported instruction, bad target, or too many instructions after unrolling.**  
Fix: See the message; docs/QEC.md lists what from_stim takes.

### QM-E735

**Stim detector error model text could not be parsed.**  
Fix: Use Stim's format: error(p) D0 D1 ^ D2 L0, detector, logical_observable, shift_detectors, repeat.

### QM-E736

**Invalid sliding-window setup: window, step, or missing detector round data.**  
Fix: 0 < step <= window <= rounds and (rounds - window) % step == 0; the DEM needs detector coordinates or an explicit round per detector.

### QM-E737

**missing_detectors needs a Clifford circuit with at most 20000 measurements.**  
Fix: Bind parameters and remove non-Clifford gates.

### QM-E731

**Correlated matching needs error probabilities up to 0.5.**  
Fix: Use plain 'mwpm' for this detector error model.

### QM-E730

**A computational circuit given to encode() contains noise channels, QEC annotations or unbound symbolic parameters.**  
Fix: Encode noiseless, QEC-agnostic, bound circuits; add noise with qumo.ft.emulate.add_noise; use the comparator circuit for run-time angles.

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
Fix: steps >= 1, order 1 or even.

### QM-E809

**Invalid graph specification.**  
Fix: Edges are 'i-j' pairs of distinct non-negative nodes.

### QM-E816

**Invalid SQD policy or trim settings.**  
Fix: policy 'standard' or 'trim', ratios in (0, 1].

### QM-E817

**Invalid multi-product formula: step counts must be distinct and positive, with a feasible L1 bound.**  
Fix: Use distinct steps; 'approx' needs max_l1 >= 1.

### QM-E818

**Invalid IQP model: a generator is empty or outside the register, or the weights do not match the parameter groups.**  
Fix: One weight per parameter group; generators are non-empty lists of qubits.

### QM-E819

**lie_closure passed max_dim.**  
Fix: Raise max_dim: the algebra can have dimension up to 4^n - 1.

### QM-E815

**T-REX cannot run: too many qubits, too few shots, or a calibration too close to 0.**  
Fix: Use at most 62 qubits and shots >= num_twirls; raise calibration_shots.

### QM-E810

**Invalid QPE options.**  
Fix: 1 <= n_eval <= 16 and a unitary on >= 1 qubit.

### QM-E820

**Invalid multi-controlled gate construction.**  
Fix: Check control/target/ancilla counts.

### QM-E830

**Invalid QML template, embedding or observable arguments (qumo.algorithms.qml): wrong weight/feature count, bad wires, unknown rotation, zero-norm or oversized amplitude vector, diagonal observable above 12 qubits.**  
Fix: Check the shapes in docs/QML.md (weights L*n*3 for strongly entangling layers, 4^n-1 for ArbitraryUnitary, 15/3 for QCNN conv/pool).

### QM-E831

**Shape mismatch in a QML kernel, dataset or training helper (qumo.algorithms.qkernels, datasets, qml_train).**  
Fix: Kernel matrices are row-major N x M lists; X rows must share one width; predictions and labels must have equal length.

### QM-E832

**metric_tensor()/QNG does not support the circuit: a trainable parameter in an unsupported gate (U, ...) or shared by several gates (e.g. QAOA gamma).**  
Fix: Use one parameter per rx/ry/rz/p/cp/rxx/ryy/rzz/crx/cry/crz gate, or restrict with wrt=[...].

### QM-E833

**Invalid optimizer state or settings (StepOptimizer, QNGOptimizer): the parameter or gradient length changed between steps, or a setting is out of range.**  
Fix: Call reset() before optimising a vector of another length; check lr, momentum and beta values.

### QM-E834

**Dataset file missing or malformed (load_csv, load_json_matrix).**  
Fix: Run from the repository root (examples ship their data under examples/*/data); every CSV row needs the same number of numeric columns.

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

### QM-E941

**Unknown GPU precision selector: the 'precision' argument (sample_on, qumo_run_device, Device.gpu, qumo_py.sample_device, qumo run --precision) or the QUMO_GPU_PRECISION environment variable.**  
Fix: Use 'auto' (QUMO_GPU_PRECISION when set, else the build default: fp32), 'fp32' or 'fp64'; QUMO_GPU_PRECISION accepts 'fp32' or 'fp64'.

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

### QM-W021

**A GPU precision ('fp32' / 'fp64') was requested but the simulator that ran has no such choice: CPU state vectors run in fp64, the stabilizer and Pauli-frame samplers are exact.**  
Fix: Use device 'gpu' for a GPU state vector in the requested precision, or precision 'auto'.

### QM-W050

**OpenQASM gate definition shadows a native gate with a different signature.**  
Fix: Rename the gate.

### QM-W051

**OpenQASM delay is ignored.**  
Fix: Model idle noise with '#pragma qumo depolarize1(p) q[i];'.

### QM-W052

**OpenQASM annotation is ignored.**  
Fix: Annotations carry no meaning in QUMO; the annotated statement still runs.

### QM-W053

**OpenQASM box duration is ignored.**  
Fix: QUMO circuits have no timing model; the box still keeps its gates together.

### QM-W054

**The body of a while loop cannot change its condition and has no break: the loop never runs, or runs until QM-E480.**  
Fix: Assign a variable the condition reads, measure into a bit it reads, or add a break.

### QM-W055

**An integer division loses a remainder in a constant used as an angle or a float (float t = 3 / 4 * pi is 0), or a literal-only gate argument keeps real division where OpenQASM 3 and Qiskit truncate (rz(3/4*pi) is 3 pi / 4 here, 0 there).**  
Fix: Write it in floats, such as 3.0 / 4, so that every tool reads the same value.
