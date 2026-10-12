"""Structured, machine-readable diagnostics: stable codes, locations, fix hints.

Philosophy (rustc/clang diagnostics + LSP, applied to quantum programs):

* Every failure anywhere in QUMO is an `Error` whose text starts with a stable
  code `QM-Exxx` and (usually) ends with `Hint: <how to fix>`. Codes never
  change meaning; messages may improve. Agents branch on the code, show the
  message, and apply the hint.
* `diagnostic_from_error()` turns that text back into a `Diagnostic`
  {code, severity, message, hint, line, col, snippet}, so tools/CLI/MCP all
  return the same JSON shape (`Diagnostic.to_json()`).
* `lint()` adds `QM-Wxxx` findings that are legal but probably unintended
  (unused qubits, gates after measurement, overwritten bits, cancelling
  pairs, ...), each pointing at an instruction index.
* `error_catalog()` is the single source of truth behind
  `qumo describe errors`, the `qumo_describe` MCP tool and docs/ERRORS.md.

Code ranges: E0xx circuit structure, E1xx parameters, E2xx functors,
E3xx Pauli observables, E4xx simulation/execution, E5xx OpenQASM,
E6xx compiler, E7xx stabilizer/QEC, E8xx algorithms, E9xx JSON/MCP/CLI
interface (E999 = uncategorized), W0xx lint warnings.
"""

from std.math import pi

from qumo.ir.circuit import Circuit, Instruction
from qumo.ir.gates import (
    Op, op_name, op_arity, op_num_params, op_is_unitary, op_is_noise,
    op_is_annotation, op_is_self_inverse, op_inverse_kind, op_is_symmetric,
)
from qumo.ai.json import JsonWriter


# ---------------------------------------------------------------------------
# Diagnostic
# ---------------------------------------------------------------------------
@fieldwise_init
struct Diagnostic(Copyable, Writable):
    var code: String  # "QM-E510"
    var severity: String  # "error" | "warning" | "info"
    var message: String
    var hint: String
    var line: Int  # 1-based source line, 0 = not applicable
    var col: Int
    var snippet: String  # rendered source excerpt (may be empty)
    var op_index: Int  # instruction index for circuit-level findings, -1 otherwise

    @staticmethod
    def make(code: String, severity: String, message: String, hint: String, op_index: Int = -1) -> Diagnostic:
        return Diagnostic(code, severity, message, hint, 0, 0, String(), op_index)

    def write_json(self, mut w: JsonWriter):
        w.begin_object()
        w.kv_string("code", self.code)
        w.kv_string("severity", self.severity)
        w.kv_string("message", self.message)
        w.kv_string("hint", self.hint)
        if self.line > 0:
            w.kv_int("line", self.line)
            w.kv_int("col", self.col)
        if self.snippet.byte_length() > 0:
            w.kv_string("snippet", self.snippet)
        if self.op_index >= 0:
            w.kv_int("instruction", self.op_index)
        w.kv_string("docs", "docs/ERRORS.md#" + self.code.lower())
        w.end_object()

    def to_json(self) -> String:
        var w = JsonWriter()
        self.write_json(w)
        return w.finish()

    def write_to(self, mut w: Some[Writer]):
        """rustc-style human rendering."""
        w.write(self.severity, "[", self.code, "]: ", self.message)
        if self.line > 0:
            w.write("\n  --> line ", self.line, ", col ", self.col)
        if self.op_index >= 0:
            w.write("\n  --> instruction #", self.op_index)
        if self.snippet.byte_length() > 0:
            w.write("\n", self.snippet)
        if self.hint.byte_length() > 0:
            w.write("\n  hint: ", self.hint)


def diagnostics_to_json(diags: List[Diagnostic]) -> String:
    var w = JsonWriter()
    w.begin_array()
    for d in diags:
        d.write_json(w)
    w.end_array()
    return w.finish()


# ---------------------------------------------------------------------------
# Error-text parsing
# ---------------------------------------------------------------------------
def _is_d(c: Int) -> Bool:
    return c >= 0x30 and c <= 0x39


def parse_error_code(err: String) -> String:
    """First `QM-[EWI]ddd` occurring in `err`; `QM-E999` when there is none."""
    var bs = err.as_bytes()
    var n = len(bs)
    var i = 0
    while i + 7 <= n:
        if Int(bs[i]) == ord("Q") and Int(bs[i + 1]) == ord("M") and Int(bs[i + 2]) == ord("-"):
            var k = Int(bs[i + 3])
            if (k == ord("E") or k == ord("W") or k == ord("I")) and _is_d(Int(bs[i + 4])) and _is_d(Int(bs[i + 5])) and _is_d(Int(bs[i + 6])):
                return String(unsafe_from_utf8=bs[i:i + 7])
        i += 1
    return "QM-E999"


def _severity_of(code: String) -> String:
    var bs = code.as_bytes()
    if len(bs) >= 4:
        if Int(bs[3]) == ord("W"):
            return "warning"
        if Int(bs[3]) == ord("I"):
            return "info"
    return "error"


def _parse_int_at(s: String, start: Int) -> Int:
    var bs = s.as_bytes()
    var v = 0
    var i = start
    while i < len(bs) and _is_d(Int(bs[i])):
        v = v * 10 + Int(bs[i]) - 0x30
        i += 1
    return v


def _rstrip_dot(s: String) -> String:
    var bs = s.as_bytes()
    var e = len(bs)
    while e > 0 and (Int(bs[e - 1]) == ord(".") or Int(bs[e - 1]) == ord(" ")):
        e -= 1
    return String(s[byte=0:e])


def diagnostic_from_error(err: String) -> Diagnostic:
    """Parse `QM-Exxx: message (line L, col C). Hint: fix\\n<snippet>` (every
    part optional) into a `Diagnostic`. Non-QUMO errors map to QM-E999."""
    var code = parse_error_code(err)
    var nl = err.find("\n")
    var first = String(err) if nl < 0 else String(err[byte=0:nl])
    var snippet = String() if nl < 0 else String(err[byte=nl + 1:])
    var body = first
    var prefix = code + ": "
    if first.startswith(prefix):
        body = String(first[byte=prefix.byte_length():])
    elif code == "QM-E999":
        body = "internal error: " + first
    var hint = String()
    var h = body.find(" Hint: ")
    var hl = 7
    if h < 0:
        h = body.find("Hint: ")
        hl = 6
    if h >= 0:
        hint = _rstrip_dot(String(body[byte=h + hl:]))
        var head = String(body[byte=0:h])
        body = head^
    var line = 0
    var col = 0
    var l = body.find(" (line ")
    if l >= 0:
        line = _parse_int_at(body, l + 7)
        var c = body.find(", col ", l)
        if c >= 0:
            col = _parse_int_at(body, c + 6)
        var head = String(body[byte=0:l])
        body = head^
    var msg = _rstrip_dot(body)
    if hint.byte_length() == 0:
        hint = default_hint(code)
    return Diagnostic(code, _severity_of(code), msg, hint, line, col, snippet, -1)


def default_hint(code: String) -> String:
    """Catalog hint for a code (used when the raising site gave none)."""
    for e in error_catalog():
        if e.code == code:
            return e.hint
    return ""


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------
@fieldwise_init
struct ErrorInfo(Copyable):
    var code: String
    var category: String
    var meaning: String
    var hint: String


def _e(mut out: List[ErrorInfo], code: String, cat: String, meaning: String, hint: String):
    out.append(ErrorInfo(code, cat, meaning, hint))


def error_ranges() -> List[ErrorInfo]:
    """Code ranges -> owning subsystem (codes are never reused across ranges)."""
    var r = List[ErrorInfo]()
    _e(r, "QM-E0xx", "circuit", "Circuit structure (qumo.ir.circuit.validate)", "")
    _e(r, "QM-E1xx", "parameters", "Symbolic parameters and binding", "")
    _e(r, "QM-E2xx", "functors", "inverse()/controlled()/power() transforms", "")
    _e(r, "QM-E3xx", "pauli", "Pauli strings / observables", "")
    _e(r, "QM-E4xx", "simulation", "State-vector simulation and execution (qumo.sim, qumo.exec)", "")
    _e(r, "QM-E5xx", "qasm", "OpenQASM 2/3 parsing (qumo.ir.qasm)", "")
    _e(r, "QM-E6xx", "compiler", "Compiler passes, pass pipelines and targets (qumo.compiler)", "")
    _e(r, "QM-E7xx", "stabilizer", "Stabilizer / Pauli-frame simulation and QEC (qumo.sim.stabilizer, qumo.qec)", "")
    _e(r, "QM-E8xx", "algorithms", "Algorithms library (qumo.algorithms)", "")
    _e(r, "QM-E9xx", "interface", "JSON, MCP tools and CLI (qumo.ai, cli); QM-E999 = uncategorized", "")
    _e(r, "QM-W0xx", "lint", "Lint warnings from qumo.ai.diagnostics.lint", "")
    return r^


def error_catalog() -> List[ErrorInfo]:
    """All documented codes. Keep in sync with docs/ERRORS.md."""
    var c = List[ErrorInfo]()
    # circuit structure
    _e(c, "QM-E001", "circuit", "An instruction uses a qubit index outside the circuit.", "Valid qubits are 0..num_qubits-1; enlarge the register or fix the index.")
    _e(c, "QM-E002", "circuit", "A multi-qubit gate repeats the same qubit.", "Each operand of a gate must be a distinct qubit.")
    _e(c, "QM-E003", "circuit", "A condition refers to a classical bit beyond the classical register.", "Declare enough classical bits or fix the condition bit.")
    # parameters
    _e(c, "QM-E101", "parameters", "Number of parameter values does not match the circuit's symbolic parameters.", "Pass one value per parameter, in circuit.param_names order (or a name->value object).")
    _e(c, "QM-E102", "parameters", "Operation needs a circuit without unbound parameters.", "Call circuit.bind(values) first.")
    # functors
    _e(c, "QM-E201", "functors", "inverse() on a circuit with non-unitary operations.", "Remove measurements/resets/noise or split the circuit.")
    _e(c, "QM-E202", "functors", "controlled() does not support a gate, or needs bound parameters.", "Bind parameters, or translate to the {u, cx} basis first.")
    _e(c, "QM-E203", "functors", "The requested control qubit is already used by the circuit.", "Pick a fresh control qubit.")
    # pauli
    _e(c, "QM-E301", "pauli", "Malformed Pauli expression.", "Write e.g. '0.5*Z0 Z1 - 1.2*X0' or dense labels '0.5*ZZI'.")
    _e(c, "QM-E302", "pauli", "Pauli index >= 64.", "QUMO Pauli strings are bit-packed into 64 qubits.")
    _e(c, "QM-E303", "pauli", "Dense Pauli labels of different lengths.", "Pad labels with I to the same length.")
    # simulation / execution
    _e(c, "QM-E401", "simulation", "evolve()/statevector cannot apply classically-conditioned gates.", "Use sample() (trajectories) for dynamic circuits.")
    _e(c, "QM-E402", "simulation", "Sampling a circuit without measurements.", "Add measurements (measure_all) or use statevector/expectation.")
    _e(c, "QM-E403", "simulation", "Estimator task failed for at least one parameter set.", "Inspect the nested error of the failing parameter set.")
    _e(c, "QM-E404", "simulation", "State vector too large for the requested operation.", "Reduce the number of qubits or use a stabilizer method.")
    _e(c, "QM-E450", "simulation", "Invalid physical error rate.", "Probabilities must lie in [0, 1].")
    _e(c, "QM-E460", "execution", "No devices available in the execution pool.", "Add Device.cpu() at least.")
    _e(c, "QM-E461", "execution", "Job failed or is unknown.", "Check the job id and the nested error.")
    _e(c, "QM-E462", "execution", "Job state does not allow the operation.", "Wait for completion or resubmit.")
    # qasm
    _e(c, "QM-E500", "qasm", "OpenQASM syntax error (unexpected or missing token).", "Check the token at line:col; every statement ends with ';'.")
    _e(c, "QM-E501", "qasm", "Unexpected character in OpenQASM source.", "Remove it or check for a missing quote/comment marker.")
    _e(c, "QM-E502", "qasm", "Unterminated string literal or block comment.", "Close strings on the same line and comments with */.")
    _e(c, "QM-E503", "qasm", "Unsupported OpenQASM version.", "Use 'OPENQASM 3.0;' or 'OPENQASM 2.0;'.")
    _e(c, "QM-E504", "qasm", "OpenQASM feature outside QUMO's supported subset (subroutines, while, classical assignment, ...).", "Use the gate-level subset; see 'qumo describe capabilities'.")
    _e(c, "QM-E510", "qasm", "Unknown gate name.", "Check the suggestions, run 'qumo describe gates', or define it with 'gate name(params) a, b { ... }'.")
    _e(c, "QM-E511", "qasm", "Wrong number of gate parameters.", "Match the gate signature, e.g. rz(theta) q[0];")
    _e(c, "QM-E512", "qasm", "Wrong number of qubit operands (including ctrl @ controls).", "Match the gate arity; controls of ctrl @ come first.")
    _e(c, "QM-E513", "qasm", "A gate uses the same qubit twice.", "Each operand must be a distinct qubit.")
    _e(c, "QM-E514", "qasm", "Broadcast over registers of different sizes.", "Broadcast operands must have equal size, or index single qubits.")
    _e(c, "QM-E520", "qasm", "Undeclared or wrong-kind identifier (register, parameter, gate argument).", "Declare it before use; check qubit vs bit registers.")
    _e(c, "QM-E521", "qasm", "Register index out of range.", "Valid indices are 0..size-1; slices [a:b] are inclusive.")
    _e(c, "QM-E522", "qasm", "Name declared twice.", "Use unique names for registers, inputs, constants and gates.")
    _e(c, "QM-E523", "qasm", "Invalid register size.", "Register sizes must be positive integer constants.")
    _e(c, "QM-E530", "qasm", "A constant was required (index, size, exponent, gphase) but the expression is not constant/integer.", "Use numeric literals, const values or loop variables.")
    _e(c, "QM-E531", "qasm", "Parameter expression is not affine (product/quotient of inputs).", "QUMO parameters are coeff*input + offset; bind values or restructure.")
    _e(c, "QM-E532", "qasm", "Parameter expression mixes two different inputs.", "Use one input per gate angle (split into two gates if needed).")
    _e(c, "QM-E533", "qasm", "Math domain error (division by zero, log/sqrt of invalid value).", "Check the expression.")
    _e(c, "QM-E534", "qasm", "Unknown function or wrong arity in an expression.", "Supported: sin cos tan arcsin arccos arctan exp ln sqrt abs floor ceil pow mod.")
    _e(c, "QM-E540", "qasm", "Measurement maps a different number of qubits and bits.", "Qubit and bit operands must have equal size.")
    _e(c, "QM-E541", "qasm", "Unsupported condition (multi-bit register value, nested or compound if).", "QUMO conditions test one bit: if (c[i] == 1) ...")
    _e(c, "QM-E550", "qasm", "Invalid gate modifier (non-integer pow, bad ctrl count).", "Use inv @, ctrl(n) @, negctrl @ or pow(k) @ with integer k.")
    _e(c, "QM-E551", "qasm", "Gate definition expands recursively.", "Gates must not call themselves.")
    _e(c, "QM-E560", "qasm", "Invalid '#pragma qumo' directive.", "See docs/AI_NATIVE.md for detector/observable/tick/param pragmas.")
    _e(c, "QM-E570", "qasm", "Invalid or too large for-loop range.", "Use constant inclusive ranges [a:b] / [a:step:b] or sets {a, b}.")
    # compiler
    _e(c, "QM-E600", "compiler", "Unknown pass name in a pass pipeline.", "List passes with the pass manager docs; e.g. 'cancel,commute,fuse1q'.")
    _e(c, "QM-E601", "compiler", "Malformed pass pipeline (unbalanced parentheses / syntax).", "Write pipelines like 'fixpoint(cancel,commute),route(sabre)'.")
    _e(c, "QM-E602", "compiler", "Bad pass argument (unknown layout/routing method, fixpoint options).", "Check the pass argument names and values.")
    _e(c, "QM-E603", "compiler", "Basis translation impossible (no usable 2-qubit entangler or decomposition rule).", "Include an entangler (cx, cz, ecr, iswap, rxx, rzz) in the target basis.")
    _e(c, "QM-E604", "compiler", "A 1-qubit gate cannot be expressed in the target basis.", "Add rz+sx, u, or rx+ry style 1-qubit gates to the basis.")
    _e(c, "QM-E605", "compiler", "Circuit is wider than the target.", "Pick a larger target or reduce the circuit width.")
    _e(c, "QM-E606", "compiler", "Interacting qubits are in disconnected parts of the coupling map.", "Use a connected target or a layout that keeps interacting qubits together.")
    _e(c, "QM-E607", "compiler", "Routing needs gates on at most 2 qubits.", "Run 'unroll3q' (or 'basis') before 'route'.")
    _e(c, "QM-E609", "compiler", "Optimization level out of range.", "Use 0..3.")
    _e(c, "QM-E610", "compiler", "Unknown gate name in a basis specification.", "Use gate names like rz,sx,x,cx,cz,ecr or a basis preset.")
    _e(c, "QM-E611", "compiler", "Invalid target specification (topology/size).", "Use '<topology>:<size>', e.g. 'grid:4x4', 'line:20', 'heavy_hex:5'.")
    _e(c, "QM-E612", "compiler", "Invalid coupling map (edges, distance, qubit count).", "Edges are pairs of distinct qubits; heavy_hex distance is odd.")
    _e(c, "QM-E615", "compiler", "Compiled circuit does not conform to the target (basis or connectivity).", "Add 'route' and 'basis' passes before 'check', or use a preset level.")
    # stabilizer / qec
    _e(c, "QM-E701", "stabilizer", "Non-Clifford operation in a stabilizer-only path.", "Use the state-vector simulator, or keep rotations at multiples of pi/2.")
    _e(c, "QM-E702", "stabilizer", "Unbound parameters in a stabilizer/Stim export path.", "Bind parameters first.")
    _e(c, "QM-E703", "stabilizer", "Detectors/observables are not deterministic in the noiseless circuit.", "Check the measurement records referenced by detectors.")
    _e(c, "QM-E704", "stabilizer", "Unsupported feed-forward for the stabilizer/frame simulator.", "Only Pauli X/Y/Z conditioned on one measured bit is supported.")
    _e(c, "QM-E705", "stabilizer", "Register too large for the requested output.", "Reduce clbits/observables or use another output format.")
    _e(c, "QM-E706", "qec", "Invalid QEC code parameters.", "Check code name, distance, rounds and basis.")
    _e(c, "QM-E707", "qec", "Unknown decoder name.", "Use 'mwpm', 'uf' or 'lookup'.")
    _e(c, "QM-E708", "qec", "Decoding failure (reserved; decoders currently count isolated defects in a failures field).", "Check the detector error model / syndrome size.")
    _e(c, "QM-E709", "qec", "Invalid calibration data or noise model.", "Check T1/T2 > 0, T2 <= 2*T1, error probabilities in [0, 1].")
    _e(c, "QM-E710", "qec", "Exponential/decay fit failed.", "Use more sequence lengths or shots; check the data decays.")
    _e(c, "QM-E711", "qec", "Invalid noise probabilities.", "Probabilities must lie in [0, 1].")
    # algorithms
    _e(c, "QM-E801", "algorithms", "Invalid algorithm option.", "Check the option names/values in the function docstring.")
    _e(c, "QM-E802", "algorithms", "Invalid size or index for an algorithm builder.", "Check qubit counts and indices.")
    _e(c, "QM-E803", "algorithms", "Hamiltonian/operator does not fit the register.", "Match observable qubits to the ansatz qubits.")
    _e(c, "QM-E804", "algorithms", "Parameter vector has the wrong length or index.", "Pass one value per circuit parameter.")
    _e(c, "QM-E805", "algorithms", "Circuit not supported by the chosen gradient/analysis method.", "Use parameter-shift or bind the unsupported gates.")
    _e(c, "QM-E806", "algorithms", "Unknown optimizer or gradient method.", "Use one of the documented method names.")
    _e(c, "QM-E807", "algorithms", "Problem too large for an exact/brute-force routine.", "Reduce the size or use a variational method.")
    _e(c, "QM-E808", "algorithms", "Invalid Trotterization options.", "steps >= 1, order 1 or 2.")
    _e(c, "QM-E809", "algorithms", "Invalid graph specification.", "Edges are 'i-j' pairs of distinct non-negative nodes.")
    _e(c, "QM-E810", "algorithms", "Invalid QPE options.", "1 <= n_eval <= 16 and a unitary on >= 1 qubit.")
    _e(c, "QM-E820", "algorithms", "Invalid multi-controlled gate construction.", "Check control/target/ancilla counts.")
    # interface (JSON / MCP / CLI)
    _e(c, "QM-E900", "interface", "Malformed JSON.", "Fix the JSON at the reported line/col (double-quoted keys, no trailing commas).")
    _e(c, "QM-E901", "interface", "Invalid JSON-RPC request.", "Send {\"jsonrpc\":\"2.0\",\"id\":1,\"method\":...,\"params\":{...}}.")
    _e(c, "QM-E910", "interface", "Missing required tool argument.", "See the tool's inputSchema in tools/list.")
    _e(c, "QM-E911", "interface", "Tool argument has the wrong JSON type.", "See the tool's inputSchema in tools/list.")
    _e(c, "QM-E912", "interface", "Unknown tool or command.", "List tools with tools/list or 'qumo help'.")
    _e(c, "QM-E913", "interface", "Invalid argument value (out of range, unknown name, unknown topic).", "Check allowed values in the inputSchema / 'qumo help <command>'.")
    _e(c, "QM-E914", "interface", "Observable acts on qubits outside the circuit.", "Use Pauli indices < num_qubits.")
    _e(c, "QM-E920", "interface", "Cannot read input file.", "Check the path; use '-' to read from stdin.")
    _e(c, "QM-E921", "interface", "Invalid command-line usage.", "Run 'qumo help'.")
    _e(c, "QM-E930", "interface", "Circuit too large for dense state-vector output.", "Use sampling, fewer qubits, or raise max_qubits explicitly.")
    _e(c, "QM-E999", "internal", "Uncategorized internal error (no QUMO code in the message).", "Report it with the input that triggered it.")
    # lint
    _e(c, "QM-W001", "lint", "Qubits are declared but never used.", "Shrink the register to save simulation memory (2x per qubit).")
    _e(c, "QM-W002", "lint", "A gate acts on a qubit after it was measured (no reset): the circuit becomes dynamic.", "Intended for mid-circuit measurement; otherwise move measurements to the end.")
    _e(c, "QM-W003", "lint", "A qubit is measured without any prior operation (always 0).", "Check the qubit index of the measurement.")
    _e(c, "QM-W004", "lint", "The circuit has no measurements.", "Sampling tools will measure all qubits; add measurements to control the readout.")
    _e(c, "QM-W005", "lint", "The circuit has unbound symbolic parameters.", "Pass 'params' when running it.")
    _e(c, "QM-W006", "lint", "Dense state-vector simulation needs a lot of memory.", "16 bytes * 2^n: use a stabilizer method for Clifford circuits or fewer qubits.")
    _e(c, "QM-W007", "lint", "A classical bit is overwritten before being read.", "Use distinct bits or read (condition on) the bit first.")
    _e(c, "QM-W008", "lint", "A condition reads a classical bit that was never written (always 0).", "Measure into the bit before conditioning on it.")
    _e(c, "QM-W009", "lint", "Classical bits are never written.", "Shrink the bit register or add measurements.")
    _e(c, "QM-W010", "lint", "Adjacent gates cancel each other.", "Remove both or run the optimizer (qumo.compiler).")
    _e(c, "QM-W011", "lint", "Rotation by angle 0 (no-op).", "Remove the gate.")
    _e(c, "QM-W012", "lint", "Very deep circuit.", "Consider optimization passes before simulation.")
    _e(c, "QM-W014", "lint", "Noise probability outside [0, 1].", "Probabilities must lie in [0, 1].")
    _e(c, "QM-W050", "lint", "OpenQASM gate definition shadows a native gate with a different signature.", "Rename the gate.")
    _e(c, "QM-W051", "lint", "OpenQASM delay is ignored.", "Model idle noise with '#pragma qumo depolarize1(p) q[i];'.")
    return c^


# ---------------------------------------------------------------------------
# Lint
# ---------------------------------------------------------------------------
def _qubit_list(qs: List[Int], limit: Int = 16) -> String:
    var s = String("[")
    for i in range(min(len(qs), limit)):
        if i > 0:
            s += ", "
        s += String(qs[i])
    if len(qs) > limit:
        s += ", ... (" + String(len(qs)) + " total)"
    return s + "]"


def _cancels(a: Instruction, b: Instruction) -> Bool:
    """b undoes a (same operands, no conditions)."""
    if a.cond >= 0 or b.cond >= 0:
        return False
    if not op_is_unitary(a.kind) or not op_is_unitary(b.kind) or a.kind == Op.I:
        return False
    var n = op_arity(a.kind)
    if op_arity(b.kind) != n:
        return False
    var same = True
    for j in range(n):
        if a.qubit(j) != b.qubit(j):
            same = False
    if not same and not (n == 2 and op_is_symmetric(a.kind) and a.q0 == b.q1 and a.q1 == b.q0):
        return False
    var np = op_num_params(a.kind)
    if np == 0:
        return op_inverse_kind(a.kind) == b.kind
    if a.kind != b.kind or np != 1:
        return False
    var p = a.p0
    var q = b.p0
    if p.is_symbolic() or q.is_symbolic():
        return p.index == q.index and p.coeff == -q.coeff and p.offset == -q.offset
    return p.offset == -q.offset


def lint(circuit: Circuit) -> List[Diagnostic]:
    """Static checks for legal-but-suspicious circuits (codes QM-W0xx)."""
    var out = List[Diagnostic]()
    var n = circuit.num_qubits
    var used = List[Bool](length=n, fill=False)
    var touched = List[Bool](length=n, fill=False)  # any op before measurement
    var measured = List[Bool](length=n, fill=False)
    var after_meas_reported = List[Bool](length=n, fill=False)
    var last_op = List[Int](length=n, fill=-1)
    var nc = circuit.num_clbits
    var written = List[Bool](length=nc, fill=False)
    var read_since_write = List[Bool](length=nc, fill=True)
    var cbit_writer = List[Int](length=nc, fill=-1)
    var never_touched_measured = List[Int]()
    var n_meas = 0
    for i in range(len(circuit.ops)):
        var g = circuit.ops[i]
        var k = g.kind
        if g.cond >= 0 and g.cond < nc:
            if not written[g.cond]:
                out.append(Diagnostic.make("QM-W008", "warning", "instruction #" + String(i) + " '" + op_name(k) + "' is conditioned on bit c[" + String(g.cond) + "] which has not been written yet (it is always 0)", "measure into c[" + String(g.cond) + "] before the condition", i))
            read_since_write[g.cond] = True
        if op_is_annotation(k):
            if k == Op.BARRIER:
                for q in range(n):
                    last_op[q] = -1
            continue
        var a = op_arity(k)
        for j in range(a):
            var q = g.qubit(j)
            if q >= 0 and q < n:
                used[q] = True
        if k == Op.MEASURE:
            n_meas += 1
            var q = g.q0
            if q >= 0 and q < n:
                if not touched[q]:
                    never_touched_measured.append(q)
                measured[q] = True
                last_op[q] = -1
            var cb = g.clbit
            if cb >= 0 and cb < nc:
                if written[cb] and not read_since_write[cb]:
                    out.append(Diagnostic.make("QM-W007", "warning", "measurement #" + String(i) + " overwrites bit c[" + String(cb) + "] written by instruction #" + String(cbit_writer[cb]) + " before it was read", "measure into a different bit or condition on c[" + String(cb) + "] first", i))
                written[cb] = True
                read_since_write[cb] = False
                cbit_writer[cb] = i
            continue
        if k == Op.RESET:
            if g.q0 >= 0 and g.q0 < n:
                measured[g.q0] = False
                touched[g.q0] = True
                last_op[g.q0] = -1
            continue
        if op_is_noise(k):
            var np = op_num_params(k)
            for j in range(np):
                var p = g.param(j)
                if not p.is_symbolic() and (p.offset < 0.0 or p.offset > 1.0):
                    out.append(Diagnostic.make("QM-W014", "error", "noise channel '" + op_name(k) + "' at #" + String(i) + " has probability " + String(p.offset) + " outside [0, 1]", "probabilities must lie in [0, 1]", i))
            continue
        # unitary gate
        for j in range(a):
            var q = g.qubit(j)
            if q < 0 or q >= n:
                continue
            touched[q] = True
            if measured[q] and not after_meas_reported[q]:
                after_meas_reported[q] = True
                out.append(Diagnostic.make("QM-W002", "info", "gate '" + op_name(k) + "' (#" + String(i) + ") acts on qubit " + String(q) + " after it was measured; the circuit is dynamic and runs as per-shot trajectories", "intended for mid-circuit measurement; otherwise move measurements to the end", i))
        # adjacent cancellation
        if a >= 1 and g.q0 >= 0 and g.q0 < n:
            var j0 = last_op[g.q0]
            var all_same = j0 >= 0
            for j in range(1, a):
                var q = g.qubit(j)
                if q < 0 or q >= n or last_op[q] != j0:
                    all_same = False
            if all_same and _cancels(circuit.ops[j0], g):
                out.append(Diagnostic.make("QM-W010", "info", "gates #" + String(j0) + " '" + op_name(circuit.ops[j0].kind) + "' and #" + String(i) + " '" + op_name(k) + "' cancel each other", "remove both or run the optimizer", i))
        if op_num_params(k) == 1 and (k == Op.RX or k == Op.RY or k == Op.RZ or k == Op.P or k == Op.CP or k == Op.CRX or k == Op.CRY or k == Op.CRZ or k == Op.RXX or k == Op.RYY or k == Op.RZZ):
            if not g.p0.is_symbolic() and g.p0.offset == 0.0:
                out.append(Diagnostic.make("QM-W011", "info", "'" + op_name(k) + "' at #" + String(i) + " rotates by 0 (no-op)", "remove the gate", i))
        for j in range(a):
            var q = g.qubit(j)
            if q >= 0 and q < n:
                last_op[q] = i
    var unused = List[Int]()
    for q in range(n):
        if not used[q]:
            unused.append(q)
    if len(unused) > 0:
        out.append(Diagnostic.make("QM-W001", "warning", String(len(unused)) + " qubit(s) never used: " + _qubit_list(unused), "shrink the register; every unused qubit doubles state-vector memory"))
    if len(never_touched_measured) > 0:
        out.append(Diagnostic.make("QM-W003", "info", "qubit(s) " + _qubit_list(never_touched_measured) + " are measured without any prior operation (always 0)", "check the measured qubit indices"))
    var unwritten = List[Int]()
    for b in range(nc):
        if not written[b]:
            unwritten.append(b)
    if len(unwritten) > 0 and n_meas > 0:
        out.append(Diagnostic.make("QM-W009", "info", String(len(unwritten)) + " classical bit(s) never written: " + _qubit_list(unwritten), "shrink the bit register or measure into them"))
    if n_meas == 0 and n > 0:
        out.append(Diagnostic.make("QM-W004", "info", "circuit has no measurements", "sampling tools measure all qubits automatically; add 'c = measure q;' to control readout"))
    if circuit.num_params() > 0:
        out.append(Diagnostic.make("QM-W005", "info", "circuit has " + String(circuit.num_params()) + " unbound parameter(s) " + circuit.param_names_str(), "pass 'params' (array in this order, or a name->value object) when running"))
    if n > 26:
        var gib = Float64(1 << min(n, 62)) * 16.0 / Float64(1 << 30)
        out.append(Diagnostic.make("QM-W006", "warning", "dense state-vector simulation of " + String(n) + " qubits needs " + String(gib) + " GiB", "use a stabilizer method for Clifford circuits, or fewer qubits"))
    var d = circuit.depth()
    if d > 10000:
        out.append(Diagnostic.make("QM-W012", "info", "circuit depth " + String(d) + " is very large", "run optimization passes before simulation"))
    return out^
