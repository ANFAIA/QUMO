"""Tool registry: one implementation of every agent-facing operation, exposed
through MCP (`qumo.ai.mcp`) and the CLI (`cli/qumo.mojo`) alike.

A tool = name + description + JSON-Schema `inputSchema` + a thin handler
`def(JsonValue) raises thin -> ToolResult`. Handlers take JSON arguments and
return a JSON object (`structured`), so CLI `--json` output, MCP
`structuredContent` and tests all see the same bytes.

Errors never escape: `ToolRegistry.call` converts any raised `QM-Exxx`
error into an `isError` result carrying the structured `Diagnostic`.

Extending (other QUMO modules)
------------------------------
```mojo
# qumo/compiler/tools.mojo
from qumo.ai.tools import ToolRegistry, ToolResult, ToolContext, tool_ok, arg_circuit
from qumo.core.json import JsonValue
def _transpile(args: JsonValue, ctx: ToolContext) raises -> ToolResult: ...
def register_compiler_tools(mut reg: ToolRegistry):
    reg.add("qumo_transpile", "Transpile", "Optimize/route a circuit ...",
            "{\\"type\\":\\"object\\", ...}", _transpile)
```
then add one line to `default_registry()` below. Every registered tool
appears automatically in MCP `tools/list`, `qumo tool <name> '<json>'`,
`qumo tool list` and `describe capabilities`.
"""

from std.time import perf_counter_ns
from std.math import sqrt

from qumo.ir.circuit import Circuit, check_finite_values
from qumo.ir.gates import Op
from qumo.ir.pauli import PauliSum, parse_pauli_sum
from qumo.ir.program import Program, SK
from qumo.ir.qasm import draw_program, load_any_with_warnings, to_qasm
from qumo.ir.qasm2 import to_qasm2
from qumo.sim.runner import sample, evolve, bitstring
from qumo.sim.program_runner import DEFAULT_MAX_ITERATIONS, ProgramResult, is_clifford_program, sample_program
from qumo.exec.dispatch import sample_auto, choose_method
from qumo.compiler.program_compile import program_metrics
from qumo.core.json import JsonWriter, JsonValue, JsonKind, parse_json, json_quote
from qumo.ai.diagnostics import (
    Diagnostic, diagnostic_from_error, lint, lint_program, parse_error_code,
)
from qumo.ai.describe import (
    QUMO_VERSION, describe_gates_json, describe_gate_json, describe_errors_json,
    describe_errors_markdown, describe_error_json, describe_capabilities_json, write_circuit_json,
    write_circuit_summary, circuit_from_json, describe_passes_json, describe_models_json,
    program_from_json, write_program_json,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
struct ToolResult(Movable):
    var structured: String  # JSON object text (MCP structuredContent)
    var text: String  # human/LLM-readable rendering; "" -> structured JSON
    var is_error: Bool

    def __init__(out self, var structured: String, var text: String, is_error: Bool):
        self.structured = structured^
        self.text = text^
        self.is_error = is_error

    def content_text(self) -> String:
        if self.text.byte_length() > 0:
            return self.text
        return self.structured


struct ToolContext(Copyable):
    """Server-side context handed to every tool (extensible: limits, cwd...)."""

    var tool_names: List[String]

    def __init__(out self, var tool_names: List[String]):
        self.tool_names = tool_names^


comptime ToolHandler = def(JsonValue, ToolContext) raises thin -> ToolResult


struct ToolSpec(Copyable):
    var name: String
    var title: String
    var description: String
    var input_schema: String  # JSON Schema (object) as text
    var handler: ToolHandler

    def __init__(out self, name: String, title: String, description: String, input_schema: String, handler: ToolHandler):
        self.name = name
        self.title = title
        self.description = description
        self.input_schema = input_schema
        self.handler = handler


def tool_ok(var structured: String) -> ToolResult:
    return ToolResult(structured^, String(), False)


def tool_error(err: String) -> ToolResult:
    """Structured error result from any QUMO error text."""
    var d = diagnostic_from_error(err)
    var w = JsonWriter()
    w.begin_object()
    w.key("error")
    d.write_json(w)
    w.end_object()
    return ToolResult(w.finish(), String(d), True)


struct ToolRegistry(Movable):
    var tools: List[ToolSpec]

    def __init__(out self):
        self.tools = List[ToolSpec]()

    def add(mut self, name: String, title: String, description: String, input_schema: String, handler: ToolHandler):
        """Register (or replace) a tool."""
        var i = self.find(name)
        if i >= 0:
            self.tools[i] = ToolSpec(name, title, description, input_schema, handler)
        else:
            self.tools.append(ToolSpec(name, title, description, input_schema, handler))

    def find(self, name: String) -> Int:
        for i in range(len(self.tools)):
            if self.tools[i].name == name:
                return i
        return -1

    def names(self) -> List[String]:
        var out = List[String]()
        for t in self.tools:
            out.append(t.name)
        return out^

    def call(self, name: String, args: JsonValue) -> ToolResult:
        """Run a tool; every failure becomes an `is_error` result."""
        var i = self.find(name)
        if i < 0:
            return tool_error("QM-E912: unknown tool '" + name + "'. Hint: available tools: " + String(", ").join(self.names()) + ".")
        var h = self.tools[i].handler
        var ctx = ToolContext(self.names())
        try:
            return h(args, ctx)
        except e:
            return tool_error(String(e))

    def write_list_json(self, mut w: JsonWriter):
        """MCP `tools/list` array."""
        w.begin_array()
        for t in self.tools:
            w.begin_object()
            w.kv_string("name", t.name)
            w.kv_string("title", t.title)
            w.kv_string("description", t.description)
            w.kv_raw("inputSchema", t.input_schema)
            w.key("annotations")
            w.begin_object()
            w.kv_string("title", t.title)
            w.kv_bool("readOnlyHint", True)
            w.kv_bool("destructiveHint", False)
            w.kv_bool("idempotentHint", True)
            w.kv_bool("openWorldHint", False)
            w.end_object()
            w.end_object()
        w.end_array()


def default_registry() -> ToolRegistry:
    """Core tools + (one line per module) tools contributed by other modules."""
    var reg = ToolRegistry()
    register_core_tools(reg)
    # register_compiler_tools(reg)    # qumo.compiler.tools  (transpile, optimize)
    # register_qec_tools(reg)         # qumo.qec.tools       (memory experiments, decoding)
    # register_algorithm_tools(reg)   # qumo.algorithms.tools (vqe, qaoa, resource estimation)
    return reg^


# ---------------------------------------------------------------------------
# Argument helpers (shared by tools of every module)
# ---------------------------------------------------------------------------
def arg_required(args: JsonValue, key: String, expected: String) raises -> JsonValue:
    var v = args.get(key)
    if not v.is_defined() or v.is_null():
        raise Error("QM-E910: missing required argument '" + key + "' (" + expected + "). Hint: see the tool's inputSchema.")
    return v^


def arg_int(args: JsonValue, key: String, default: Int, lo: Int, hi: Int) raises -> Int:
    var v = args.get_int(key, default)
    if v < lo or v > hi:
        raise Error("QM-E913: argument '" + key + "' = " + String(v) + " is out of range [" + String(lo) + ", " + String(hi) + "]. Hint: pick a value inside the range.")
    return v


def arg_program(args: JsonValue, mut warnings: List[String]) raises -> Program:
    """Program from `qasm` (OpenQASM 2/3 text, through `load_any`), `program`
    (qumo.program.v1 JSON) or `circuit` (qumo.circuit.v1 JSON, or a program)."""
    var q = args.get("qasm")
    if q.is_defined() and not q.is_null():
        return load_any_with_warnings(q.as_string(), warnings)
    var j = args.get("program")
    if j.is_defined() and not j.is_null():
        return program_from_json(j)
    var c = args.get("circuit")
    if c.is_defined() and not c.is_null():
        if c.is_object() and c.get_string("format", "") == "qumo.program.v1":
            return program_from_json(c)
        return Program.from_circuit(circuit_from_json(c))
    raise Error("QM-E910: missing required argument 'qasm' (OpenQASM 2/3 source text). Hint: e.g. {\"qasm\": \"OPENQASM 3.0; include \\\"stdgates.inc\\\"; qubit[2] q; h q[0]; cx q[0], q[1];\"}.")


def circuit_of(p: Program) raises -> Circuit:
    """The one Circuit a program is, or QM-E585 when it has runtime regions."""
    var c = p.try_flatten()
    if c:
        return c.take()
    raise Error("QM-E585: this operation needs one Circuit, and the input has classical variables or control flow at run time. Hint: sample it with qumo_run_qasm; a shot-by-shot program has no single final state.")


def arg_circuit(args: JsonValue, mut warnings: List[String]) raises -> Circuit:
    """`arg_program`, for a tool that needs a Circuit (QM-E585 otherwise)."""
    return circuit_of(arg_program(args, warnings))


def _names_text(names: List[String]) -> String:
    return "[" + String(", ").join(names) + "]"


def _values_of(args: JsonValue, names: List[String], owner: String) raises -> List[Float64]:
    """`params`: array (declaration order) or object name -> value."""
    var out = _named_values(args, names, owner)
    check_finite_values(out)
    return out^


def _named_values(args: JsonValue, names: List[String], owner: String) raises -> List[Float64]:
    var out = List[Float64]()
    var p = args.get("params")
    var n = len(names)
    if not p.is_defined() or p.is_null():
        if n > 0:
            raise Error("QM-E101: " + owner + " has " + String(n) + " parameter(s) " + _names_text(names) + " but no 'params' were given. Hint: pass \"params\": [values...] or {\"name\": value}.")
        return out^
    if p.is_array():
        if len(p) != n:
            raise Error("QM-E101: 'params' has " + String(len(p)) + " value(s), " + owner + " has " + String(n) + " parameter(s) " + _names_text(names) + ". Hint: one value per parameter, in this order.")
        for i in range(n):
            out.append(p.at(i).as_float())
        return out^
    if p.is_object():
        for i in range(n):
            var v = p.get(names[i])
            if not v.is_defined():
                raise Error("QM-E101: missing value for parameter '" + names[i] + "'. Hint: 'params' must name every parameter of " + _names_text(names) + ".")
            out.append(v.as_float())
        for k in p.keys():
            var found = False
            for nm in names:
                if nm == k:
                    found = True
            if not found:
                raise Error("QM-E913: unknown parameter '" + k + "' in 'params'. Hint: " + owner + " parameters are " + _names_text(names) + ".")
        return out^
    raise Error("QM-E911: field 'params': expected an array or object, got " + p.kind_name() + ". Hint: e.g. \"params\": [0.1, 0.2] or {\"theta\": 0.1}.")


def arg_params(args: JsonValue, circuit: Circuit) raises -> List[Float64]:
    """`params` of a circuit."""
    return _values_of(args, circuit.param_names, "circuit")


def arg_program_params(args: JsonValue, p: Program) raises -> List[Float64]:
    """`params` of a program: its `input` declarations, not its runtime slots."""
    return _values_of(args, p.user_param_names(), "program")


def arg_max_iterations(args: JsonValue) raises -> Int:
    return arg_int(args, "max_iterations", DEFAULT_MAX_ITERATIONS, 1, 9223372036854775807)


def _ms(t0: Int, t1: Int) -> Float64:
    return Float64(t1 - t0) / 1e6


def write_warnings(mut w: JsonWriter, warnings: List[String]):
    """`"warnings": [Diagnostic...]` from `QM-Wxxx: ... Hint: ...` texts (shared
    by every tool, so warnings always have the same JSON shape)."""
    w.key("warnings")
    w.begin_array()
    for s in warnings:
        diagnostic_from_error(s).write_json(w)
    w.end_array()


def _check_qubits(circuit: Circuit, max_qubits: Int) raises:
    if circuit.num_qubits > max_qubits:
        raise Error("QM-E930: circuit has " + String(circuit.num_qubits) + " qubits; dense state-vector simulation is limited to max_qubits=" + String(max_qubits) + " (needs " + String(Float64(1 << min(circuit.num_qubits, 62)) * 16.0 / 1073741824.0) + " GiB). Hint: raise 'max_qubits' if you have the memory, or reduce the circuit.")


# ---------------------------------------------------------------------------
# Core tools
# ---------------------------------------------------------------------------
comptime _QASM_PROP = "\"qasm\":{\"type\":\"string\",\"description\":\"(required unless circuit is given) OpenQASM 3.0 or 2.0 source, e.g. 'OPENQASM 3.0; include \\\"stdgates.inc\\\"; qubit[2] q; bit[2] c; h q[0]; cx q[0], q[1]; c = measure q;'\"}"
comptime _CIRCUIT_PROP = "\"circuit\":{\"type\":\"object\",\"description\":\"Alternative to qasm: circuit JSON in format qumo.circuit.v1 (see qumo_convert to='json')\"}"
comptime _PROGRAM_PROP = "\"program\":{\"type\":\"object\",\"description\":\"Alternative to qasm: program JSON in format qumo.program.v1 (see qumo_convert to='json')\"}"
comptime _PARAMS_PROP = "\"params\":{\"description\":\"Values for 'input float' parameters: array in declaration order or object name->value\",\"oneOf\":[{\"type\":\"array\",\"items\":{\"type\":\"number\"}},{\"type\":\"object\",\"additionalProperties\":{\"type\":\"number\"}}]}"


def write_program_outputs(mut w: JsonWriter, r: ProgramResult) raises:
    """`iterations` (back-edges per shot) and each declared output over the shots:
    its range, its mean, and its value in the first 16 shots."""
    var shots = len(r.iterations)
    var top = 0
    for v in r.iterations:
        top = max(top, v)
    w.key("iterations")
    w.begin_object()
    w.kv_float("mean", r.mean_iterations())
    w.kv_int("max", top)
    w.end_object()
    w.key("outputs")
    w.begin_object()
    for j in range(len(r.out_names)):
        w.key(r.out_names[j])
        w.begin_object()
        var total = 0.0
        if r.out_is_real[j]:
            var vals = r.real_output(r.out_names[j])
            var lo = vals[0] if shots > 0 else 0.0
            var hi = lo
            for v in vals:
                total += v
                lo = min(lo, v)
                hi = max(hi, v)
            w.kv_float("mean", total / Float64(max(shots, 1)))
            w.kv_float("min", lo)
            w.kv_float("max", hi)
            w.key("first_shots")
            w.begin_array()
            for s in range(min(shots, 16)):
                w.number(vals[s])
            w.end_array()
        elif r.out_is_unsigned[j]:
            var vals = r.uint_output(r.out_names[j])
            var lo = vals[0] if shots > 0 else UInt64(0)
            var hi = lo
            for v in vals:
                total += Float64(v)
                lo = min(lo, v)
                hi = max(hi, v)
            w.kv_float("mean", total / Float64(max(shots, 1)))
            w.key("min")
            w.unsigned(lo)
            w.key("max")
            w.unsigned(hi)
            w.key("first_shots")
            w.begin_array()
            for s in range(min(shots, 16)):
                w.unsigned(vals[s])
            w.end_array()
        else:
            var vals = r.int_output(r.out_names[j])
            var lo = vals[0] if shots > 0 else 0
            var hi = lo
            for v in vals:
                total += Float64(v)
                lo = min(lo, v)
                hi = max(hi, v)
            w.kv_float("mean", total / Float64(max(shots, 1)))
            w.kv_int("min", lo)
            w.kv_int("max", hi)
            w.key("first_shots")
            w.begin_array()
            for s in range(min(shots, 16)):
                w.integer(vals[s])
            w.end_array()
        w.end_object()
    w.end_object()


def _check_program_size(p: Program, max_qubits: Int) raises:
    if not is_clifford_program(p):  # Clifford programs run on the tableau
        _check_qubits(p.body, max_qubits)


def _run_program(p: Program, args: JsonValue, var warnings: List[String], t0: Int, t1: Int) raises -> ToolResult:
    var shots = arg_int(args, "shots", 1024, 1, 10000000)
    var seed = arg_int(args, "seed", 1234, 0, 9223372036854775807)
    _check_program_size(p, arg_int(args, "max_qubits", 28, 1, 40))
    var r = sample_program(p, shots, UInt64(seed), arg_program_params(args, p), arg_max_iterations(args))
    var t2 = perf_counter_ns()
    var w = JsonWriter()
    w.begin_object()
    w.key("counts")
    w.begin_object()
    for k in r.sample.sorted_keys():
        w.kv_int(k, r.sample.counts.get(k, 0))
    w.end_object()
    w.kv_int("shots", shots)
    w.kv_int("seed", seed)
    w.kv_string("method", r.sample.method)
    w.kv_int("num_qubits", p.num_qubits())
    w.kv_int("num_clbits", p.num_clbits())
    w.kv_string("most_frequent", r.sample.most_frequent())
    w.kv_string("bit_order", "leftmost character = highest classical bit c[n-1]")
    w.kv_bool("auto_measured", False)
    write_program_outputs(w, r)
    warnings.extend(r.warnings.copy())
    write_warnings(w, warnings)
    w.key("timing_ms")
    w.begin_object()
    w.kv_float("parse", _ms(t0, t1))
    w.kv_float("simulate", _ms(t1, t2))
    w.end_object()
    w.end_object()
    return tool_ok(w.finish())


def _run_qasm(args: JsonValue, ctx: ToolContext) raises -> ToolResult:
    var t0 = perf_counter_ns()
    var warnings = List[String]()
    var p = arg_program(args, warnings)
    var flat = p.try_flatten()
    var t1 = perf_counter_ns()
    if not flat:
        return _run_program(p, args, warnings^, t0, t1)
    var c = flat.take()
    var shots = arg_int(args, "shots", 1024, 1, 10000000)
    var seed = arg_int(args, "seed", 1234, 0, 9223372036854775807)
    var max_q = arg_int(args, "max_qubits", 28, 1, 40)
    if choose_method(c) != "stabilizer":  # Clifford circuits scale polynomially
        _check_qubits(c, max_q)
    var params = arg_params(args, c)
    var auto = False
    if c.num_measurements == 0:
        c.measure_all()
        auto = True
    var r = sample_auto(c, shots, UInt64(seed), params)
    var t2 = perf_counter_ns()
    var w = JsonWriter()
    w.begin_object()
    w.key("counts")
    w.begin_object()
    for k in r.sorted_keys():
        w.kv_int(k, r.counts.get(k, 0))
    w.end_object()
    w.kv_int("shots", shots)
    w.kv_int("seed", seed)
    w.kv_string("method", r.method)
    w.kv_int("num_qubits", c.num_qubits)
    w.kv_int("num_clbits", c.num_clbits)
    w.kv_string("most_frequent", r.most_frequent())
    w.kv_string("bit_order", "leftmost character = highest classical bit c[n-1]")
    w.kv_bool("auto_measured", auto)
    if auto:
        warnings.append("QM-W004: circuit had no measurements; all qubits were measured into c[0..n-1]. Hint: add 'c = measure q;' to control readout.")
    write_warnings(w, warnings)
    w.key("timing_ms")
    w.begin_object()
    w.kv_float("parse", _ms(t0, t1))
    w.kv_float("simulate", _ms(t1, t2))
    w.end_object()
    w.end_object()
    return tool_ok(w.finish())


def _statevector(args: JsonValue, ctx: ToolContext) raises -> ToolResult:
    var warnings = List[String]()
    var c = arg_circuit(args, warnings)
    var max_q = arg_int(args, "max_qubits", 24, 1, 40)
    _check_qubits(c, max_q)
    var max_amp = arg_int(args, "max_amplitudes", 64, 1, 1 << 20)
    var threshold = args.get_float("threshold", 1e-12)
    var params = arg_params(args, c)
    var has_meas = c.num_measurements > 0 or c.count(Op.RESET) > 0
    var sv = evolve(c, params)
    var n = c.num_qubits
    var w = JsonWriter()
    w.begin_object()
    w.kv_int("num_qubits", n)
    var shown = 0
    var nonzero = 0
    var norm = 0.0
    w.key("amplitudes")
    w.begin_array()
    for i in range(sv.size()):
        var p = sv.re[i] * sv.re[i] + sv.im[i] * sv.im[i]
        norm += p
        if p <= threshold:
            continue
        nonzero += 1
        if shown >= max_amp:
            continue
        shown += 1
        w.begin_object()
        w.kv_int("index", i)
        w.kv_string("bitstring", bitstring(i, n))
        w.kv_float("re", sv.re[i])
        w.kv_float("im", sv.im[i])
        w.kv_float("probability", p)
        w.end_object()
    w.end_array()
    w.kv_int("num_nonzero", nonzero)
    w.kv_bool("truncated", nonzero > shown)
    w.kv_float("norm", sqrt(norm))
    w.kv_string("bit_order", "bitstring leftmost character = highest qubit q[n-1]; index bit k = qubit k")
    w.kv_bool("measurements_ignored", has_meas)
    write_warnings(w, warnings)
    w.end_object()
    return tool_ok(w.finish())


def _expectation(args: JsonValue, ctx: ToolContext) raises -> ToolResult:
    var warnings = List[String]()
    var c = arg_circuit(args, warnings)
    var max_q = arg_int(args, "max_qubits", 28, 1, 40)
    _check_qubits(c, max_q)
    var params = arg_params(args, c)
    var ob = arg_required(args, "observable", "Pauli sum string or array of strings")
    var labels = List[String]()
    if ob.is_array():
        for i in range(len(ob)):
            labels.append(ob.at(i).as_string())
    else:
        labels.append(ob.as_string())
    var obs = List[PauliSum]()
    for s in labels:
        var h = parse_pauli_sum(s)
        if h.num_qubits() > c.num_qubits:
            raise Error("QM-E914: observable '" + s + "' acts on qubit " + String(h.num_qubits() - 1) + " but the circuit has " + String(c.num_qubits) + " qubit(s). Hint: Pauli indices must be < num_qubits (dense labels: one character per qubit).")
        obs.append(h^)
    var sv = evolve(c, params)
    var w = JsonWriter()
    w.begin_object()
    var vals = List[Float64]()
    for ref h in obs:
        vals.append(sv.expectation(h))
    if not ob.is_array():
        w.kv_float("expectation", vals[0])
    w.key("results")
    w.begin_array()
    for i in range(len(obs)):
        w.begin_object()
        w.kv_string("observable", labels[i])
        w.kv_string("parsed", String(obs[i]))
        w.kv_int("num_terms", len(obs[i]))
        w.kv_float("expectation", vals[i])
        w.end_object()
    w.end_array()
    w.kv_int("num_qubits", c.num_qubits)
    w.kv_string("method", "statevector:exact")
    w.kv_bool("measurements_ignored", c.num_measurements > 0)
    write_warnings(w, warnings)
    w.end_object()
    return tool_ok(w.finish())


def write_program_summary(p: Program, mut w: JsonWriter):
    """The summary of a program with runtime regions. It has no depth and no gate
    total: both depend on the shot, and `program_metrics` gives the cost per loop."""
    var loops = 0
    var branches = 0
    var blocks = 0
    for s in p.tape:
        if s.kind == SK.WHILE or s.kind == SK.FOR:
            loops += 1
        elif s.kind == SK.IF or s.kind == SK.SWITCH:
            branches += 1
        elif s.kind == SK.OPS:
            blocks += 1
    w.begin_object()
    w.kv_string("name", p.name())
    w.kv_int("num_qubits", p.num_qubits())
    w.kv_int("num_clbits", p.num_clbits())
    w.kv_int("num_instructions", len(p.body.ops))
    w.kv_bool("has_runtime_regions", True)
    w.kv_int("num_statements", len(p.tape))
    w.kv_int("num_blocks", blocks)
    w.kv_int("num_loops", loops)
    w.kv_int("num_branches", branches)
    w.kv_int("num_variables", len(p.vars))
    w.kv_string_list("params", p.user_param_names())
    w.kv_string("metrics", String(program_metrics(p)))
    if p.num_qubits() < 60:
        w.kv_float("statevector_bytes", Float64(1 << p.num_qubits()) * 16.0)
    var clifford = is_clifford_program(p)
    w.kv_bool("is_clifford", clifford)
    var sim = String("tableau" if clifford else "statevector")
    if not clifford and p.num_qubits() > 28:
        sim = "none locally (exceeds dense state-vector memory)"
    w.kv_string("recommended_simulator", sim)
    w.end_object()


comptime VALIDATE_MAX_ERRORS = 20


def _line_span(bs: Span[Byte, _], line: Int) -> Tuple[Int, Int]:
    """Byte range [start, end) of 1-based `line`, newline excluded; (-1, -1) past the end."""
    var start = 0
    var cur = 1
    var n = len(bs)
    while cur < line and start < n:
        if Int(bs[start]) == 10:
            cur += 1
        start += 1
    if cur != line:
        return (-1, -1)
    var end = start
    while end < n and Int(bs[end]) != 10:
        end += 1
    return (start, end)


def _code_of_line(source: String, line: Int) -> String:
    """`line` without its `//` comment and surrounding blanks ("" past the end)."""
    var sp = _line_span(source.as_bytes(), line)
    if sp[0] < 0:
        return String()
    var text = String(source[byte=sp[0]:sp[1]])
    var c = text.find("//")
    if c >= 0:
        var head = String(text[byte=0:c])
        text = head^
    return String(text.strip())


def _count_byte(s: String, ch: Int) -> Int:
    var n = 0
    for b in s.as_bytes():
        if Int(b) == ch:
            n += 1
    return n


comptime _NOT_SKIPPABLE = " OPENQASM include qubit bit qreg creg gate def defcal defcalgrammar cal opaque extern input output const let int uint float bool angle complex duration stretch array if else while for switch case default box "


def _skippable_statement(source: String, line: Int) -> Bool:
    """True when `line` holds exactly one whole simple statement (a gate call,
    measurement, reset, assignment, ...), so that leaving it out cannot make a
    later line fail: not a declaration, not a block header, and the statement
    neither starts on an earlier line nor shares the line with another one."""
    var code = _code_of_line(source, line)
    if not code.endswith(";") or _count_byte(code, ord(";")) != 1:
        return False
    if code.find("{") >= 0 or code.find("}") >= 0 or code.startswith("#") or code.find("/*") >= 0 or code.find("*/") >= 0:
        return False
    var bs = code.as_bytes()
    var e = 0
    while e < len(bs) and ((Int(bs[e]) >= ord("a") and Int(bs[e]) <= ord("z")) or (Int(bs[e]) >= ord("A") and Int(bs[e]) <= ord("Z")) or Int(bs[e]) == ord("_")):
        e += 1
    if e > 0 and String(_NOT_SKIPPABLE).find(" " + String(code[byte=0:e]) + " ") >= 0:
        return False
    var prev = line - 1
    while prev >= 1:
        var pc = _code_of_line(source, prev)
        if pc.byte_length() > 0:
            return pc.endswith(";") or pc.endswith("{") or pc.endswith("}") or pc.startswith("#") or pc.endswith("*/")
        prev -= 1
    return True


def _without_line(source: String, line: Int) -> String:
    """`source` with `line` emptied: later lines keep their numbers."""
    var sp = _line_span(source.as_bytes(), line)
    return String(source[byte=0:sp[0]]) + String(source[byte=sp[1]:])


def collect_parse_errors(source: String, first_error: String, mut complete: Bool) -> List[Diagnostic]:
    """Every independent parse error of `source`, the first one given.

    The parser stops at its first error, so recovery happens here, off every hot
    path: the failing line is left out and the source parsed again, at most
    `VALIDATE_MAX_ERRORS` times. That is only sound for a line that holds one
    whole simple statement (`_skippable_statement`); at a declaration, a block
    header, an error without a line, an error that does not move forward or one
    that repeats the previous message (the usual sign of a cascade), collection
    stops and `complete` is False: more errors may follow the last one listed.
    `complete` is True when the source parses once the listed lines are left out."""
    var out = List[Diagnostic]()
    var src = source
    var err = first_error
    var last_line = 0
    complete = False
    while True:
        var d = diagnostic_from_error(err)
        if len(out) > 0 and (d.line <= last_line or d.message == out[len(out) - 1].message):
            return out^
        out.append(d.copy())
        if d.line <= 0 or len(out) >= VALIDATE_MAX_ERRORS or not _skippable_statement(src, d.line):
            return out^
        last_line = d.line
        var next = _without_line(src, d.line)
        src = next^
        var scratch = List[String]()
        try:
            _ = load_any_with_warnings(src, scratch)
            complete = True
            return out^
        except e:
            err = String(e)


def _validate(args: JsonValue, ctx: ToolContext) raises -> ToolResult:
    var src = arg_required(args, "qasm", "OpenQASM source text").as_string()
    var warnings = List[String]()
    var w = JsonWriter()
    w.begin_object()
    var p: Program
    try:
        p = load_any_with_warnings(src, warnings)
    except e:
        var complete = False
        var ds = collect_parse_errors(src, String(e), complete)
        var text = String("INVALID")
        w.kv_bool("valid", False)
        w.key("errors")
        w.begin_array()
        for d in ds:
            d.write_json(w)
            text += "\n" + String(d)
        w.end_array()
        w.kv_bool("errors_complete", complete)
        write_warnings(w, warnings)
        w.end_object()
        return ToolResult(w.finish(), text, False)
    var flat = p.try_flatten()
    var diags = lint(flat.value()) if flat else lint_program(p)
    var nerr = 0
    for d in diags:
        if d.severity == "error":
            nerr += 1
    w.kv_bool("valid", nerr == 0)
    w.key("errors")
    w.begin_array()
    for d in diags:
        if d.severity == "error":
            d.write_json(w)
    w.end_array()
    w.key("warnings")
    w.begin_array()
    for s in warnings:
        diagnostic_from_error(s).write_json(w)
    for d in diags:
        if d.severity != "error":
            d.write_json(w)
    w.end_array()
    w.key("summary")
    if flat:
        write_circuit_summary(flat.value(), w)
    else:
        write_program_summary(p, w)
    w.end_object()
    return tool_ok(w.finish())


def _describe(args: JsonValue, ctx: ToolContext) raises -> ToolResult:
    var topic = args.get_string("topic", "capabilities")
    var name = args.get_string("name", "")
    var fmt = args.get_string("format", "json")
    if fmt != "json" and fmt != "markdown":
        raise Error("QM-E913: unknown format '" + fmt + "'. Hint: format is 'json' (default) or 'markdown' (topic 'errors' only).")
    if fmt == "markdown":
        if topic != "errors" or name.byte_length() > 0:
            raise Error("QM-E913: format 'markdown' is only available for topic 'errors' without a name. Hint: use format 'json' for other topics.")
        var md = describe_errors_markdown()
        var w = JsonWriter()
        w.begin_object()
        w.kv_string("topic", "errors")
        w.kv_string("format", "markdown")
        w.kv_string("markdown", md)
        w.end_object()
        return ToolResult(w.finish(), md, False)
    if topic == "gates":
        if name.byte_length() > 0:
            return tool_ok(describe_gate_json(name))
        return tool_ok(describe_gates_json())
    if topic == "gate":
        return tool_ok(describe_gate_json(name))
    if topic == "errors":
        if name.byte_length() > 0:
            return tool_ok(describe_error_json(name))
        return tool_ok(describe_errors_json())
    if topic == "error":
        return tool_ok(describe_error_json(name))
    if topic == "capabilities":
        return tool_ok(describe_capabilities_json(ctx.tool_names))
    if topic == "passes":
        return tool_ok(describe_passes_json())
    if topic == "models":
        return tool_ok(describe_models_json(name))
    raise Error("QM-E913: unknown topic '" + topic + "'. Hint: topic is one of gates, gate, errors, error, capabilities, passes, models.")


def _draw(args: JsonValue, ctx: ToolContext) raises -> ToolResult:
    var warnings = List[String]()
    var p = arg_program(args, warnings)
    var text = draw_program(p, args.get_int("width", 80), args.get_string("style", "ascii"))
    var w = JsonWriter()
    w.begin_object()
    w.kv_string("drawing", text)
    w.kv_int("num_qubits", p.num_qubits())
    w.kv_int("num_ops", len(p.body.ops))
    w.end_object()
    return ToolResult(w.finish(), text, False)


def _program_info(p: Program, args: JsonValue, warnings: List[String]) raises -> ToolResult:
    var w = JsonWriter()
    w.begin_object()
    w.key("summary")
    write_program_summary(p, w)
    if args.get_bool("include_instructions", False):
        w.key("program")
        write_program_json(p, w)
    if args.get_bool("include_qasm", False):
        w.kv_string("qasm", to_qasm(p))
    if args.get_bool("include_lint", True):
        w.key("lint")
        w.begin_array()
        for d in lint_program(p):
            d.write_json(w)
        w.end_array()
    write_warnings(w, warnings)
    w.end_object()
    return tool_ok(w.finish())


def _circuit_info(args: JsonValue, ctx: ToolContext) raises -> ToolResult:
    var warnings = List[String]()
    var p = arg_program(args, warnings)
    var flat = p.try_flatten()
    if not flat:
        return _program_info(p, args, warnings)
    var c = flat.take()
    var w = JsonWriter()
    w.begin_object()
    w.key("summary")
    write_circuit_summary(c, w)
    if args.get_bool("include_instructions", False):
        w.key("circuit")
        write_circuit_json(c, w)
    if args.get_bool("include_qasm", False):
        w.kv_string("qasm", to_qasm(c))
    if args.get_bool("include_lint", True):
        w.key("lint")
        w.begin_array()
        for d in lint(c):
            d.write_json(w)
        w.end_array()
    write_warnings(w, warnings)
    w.end_object()
    return tool_ok(w.finish())


def _convert_program(p: Program, to: String, args: JsonValue, warnings: List[String]) raises -> ToolResult:
    var v = args.get("params")
    if v.is_defined() and not v.is_null():
        raise Error("QM-E585: 'params' binds a Circuit; this input has classical variables or control flow at run time. Hint: pass the values when you run it, with qumo_run_qasm.")
    var w = JsonWriter()
    w.begin_object()
    if to == "qasm3" or to == "qasm" or to == "openqasm3":
        var text = to_qasm(p)
        w.kv_string("format", "openqasm3")
        w.kv_string("qasm", text)
        write_warnings(w, warnings)
        w.end_object()
        return ToolResult(w.finish(), text, False)
    if to == "json":
        w.kv_string("format", "qumo.program.v1")
        w.key("program")
        write_program_json(p, w)
        write_warnings(w, warnings)
        w.end_object()
        return tool_ok(w.finish())
    if to == "qasm2" or to == "openqasm2":
        raise Error("QM-E585: OpenQASM 2 has no classical variables and no loops. Hint: use to='qasm3'.")
    raise Error("QM-E913: unknown target format '" + to + "'. Hint: 'to' is 'qasm3', 'qasm2' or 'json'.")


def _convert(args: JsonValue, ctx: ToolContext) raises -> ToolResult:
    var warnings = List[String]()
    var prog = arg_program(args, warnings)
    var to = args.get_string("to", "qasm3")
    var flat = prog.try_flatten()
    if not flat:
        return _convert_program(prog, to, args, warnings)
    var c = flat.take()
    var p = args.get("params")
    if p.is_defined() and not p.is_null():
        c = c.bind(arg_params(args, c))
    var w = JsonWriter()
    w.begin_object()
    if to == "qasm2" or to == "openqasm2":
        var text2 = to_qasm2(c)
        w.kv_string("format", "openqasm2")
        w.kv_string("qasm", text2)
        write_warnings(w, warnings)
        w.end_object()
        return ToolResult(w.finish(), text2, False)
    if to == "qasm3" or to == "qasm" or to == "openqasm3":
        var text = to_qasm(c)
        w.kv_string("format", "openqasm3")
        w.kv_string("qasm", text)
        write_warnings(w, warnings)
        w.end_object()
        return ToolResult(w.finish(), text, False)
    if to == "json":
        w.kv_string("format", "qumo.circuit.v1")
        w.key("circuit")
        write_circuit_json(c, w)
        write_warnings(w, warnings)
        w.end_object()
        return tool_ok(w.finish())
    raise Error("QM-E913: unknown target format '" + to + "'. Hint: 'to' is 'qasm3', 'qasm2' or 'json'.")


def register_core_tools(mut reg: ToolRegistry):
    reg.add(
        "qumo_run_qasm", "Run circuit (sample)",
        "Simulate an OpenQASM 2/3 circuit or program and return measurement counts (Qiskit-style bitstrings, leftmost = highest bit). Deterministic for a given seed. Circuits without measurements are measured on all qubits. Supports mid-circuit measurement, reset, feed-forward and '#pragma qumo' noise (per-shot trajectories), and while, break, continue and classical variables, shot by shot: those results add 'iterations' (back-edges per shot) and 'outputs' (declared output variables).",
        "{\"type\":\"object\",\"properties\":{" + _QASM_PROP + "," + _CIRCUIT_PROP + "," + _PROGRAM_PROP + ",\"shots\":{\"type\":\"integer\",\"minimum\":1,\"maximum\":10000000,\"default\":1024},\"seed\":{\"type\":\"integer\",\"minimum\":0,\"default\":1234}," + _PARAMS_PROP + ",\"max_qubits\":{\"type\":\"integer\",\"default\":28,\"description\":\"Safety limit for dense simulation (16*2^n bytes)\"},\"max_iterations\":{\"type\":\"integer\",\"minimum\":1,\"default\":65536,\"description\":\"Loop back-edges allowed per shot; past it the run stops with QM-E480\"}},\"additionalProperties\":false}",
        _run_qasm,
    )
    reg.add(
        "qumo_statevector", "Final state vector",
        "Return the exact final state vector (amplitudes with probability > threshold, sorted by basis index) of a unitary circuit. Measurements and resets are ignored; classically-conditioned gates are rejected (QM-E401), and so is a program with runtime control flow (QM-E585).",
        "{\"type\":\"object\",\"properties\":{" + _QASM_PROP + "," + _CIRCUIT_PROP + "," + _PARAMS_PROP + ",\"max_amplitudes\":{\"type\":\"integer\",\"default\":64,\"minimum\":1},\"threshold\":{\"type\":\"number\",\"default\":1e-12},\"max_qubits\":{\"type\":\"integer\",\"default\":24}},\"additionalProperties\":false}",
        _statevector,
    )
    reg.add(
        "qumo_expectation", "Pauli expectation value",
        "Exact expectation value <psi|H|psi> of a Pauli-sum observable (e.g. '0.5*Z0 Z1 - 1.2*X0 + 0.3' or dense 'ZZI') on the circuit's final state. Pass an array of observables to evaluate several at once. A program with runtime control flow has no single final state: QM-E585.",
        "{\"type\":\"object\",\"properties\":{" + _QASM_PROP + "," + _CIRCUIT_PROP + ",\"observable\":{\"description\":\"Pauli sum string, or array of strings\",\"oneOf\":[{\"type\":\"string\"},{\"type\":\"array\",\"items\":{\"type\":\"string\"}}]}," + _PARAMS_PROP + ",\"max_qubits\":{\"type\":\"integer\",\"default\":28}},\"required\":[\"observable\"],\"additionalProperties\":false}",
        _expectation,
    )
    reg.add(
        "qumo_validate_qasm", "Validate OpenQASM",
        "Parse and lint OpenQASM 2/3 without simulating, including programs with while and classical variables. Returns valid=true/false, structured errors (stable QM-Exxx code, message, line, col, snippet, hint) and lint warnings (QM-Wxxx), plus a summary when valid. Every independent parse error is listed, not only the first: a failing gate call, measurement or assignment is skipped and parsing goes on; 'errors_complete' is false when collection had to stop (an error in a declaration or a block header, which later lines depend on), so more errors may follow the last one.",
        "{\"type\":\"object\",\"properties\":{" + _QASM_PROP + "},\"required\":[\"qasm\"],\"additionalProperties\":false}",
        _validate,
    )
    reg.add(
        "qumo_describe", "Describe QUMO",
        "Machine-readable self-description: topic 'gates' (every gate: arity, params, Clifford/diagonal flags, QASM syntax, matrix), 'gate' + name, 'errors' (error-code catalog with fix hints; format 'markdown' returns docs/ERRORS.md), 'error' + name (one code), 'capabilities' (simulators, QASM subset, conventions, tools), 'passes' (compiler pass catalog incl. the AI synthesis passes, route AI keys and presets) or 'models' (installed AI-transpiler models from the INDEX.json registries; 'name' = optional model directory).",
        "{\"type\":\"object\",\"properties\":{\"topic\":{\"type\":\"string\",\"enum\":[\"gates\",\"gate\",\"errors\",\"error\",\"capabilities\",\"passes\",\"models\"],\"default\":\"capabilities\"},\"name\":{\"type\":\"string\",\"description\":\"Gate name/alias or error code for topic gate/error; model directory for topic models\"},\"format\":{\"type\":\"string\",\"enum\":[\"json\",\"markdown\"],\"default\":\"json\",\"description\":\"'markdown' (topic 'errors' only): the generated docs/ERRORS.md text\"}},\"additionalProperties\":false}",
        _describe,
    )
    reg.add(
        "qumo_circuit_info", "Circuit summary",
        "Analyze a circuit without simulating: qubits, clbits, op counts, depth, two-qubit gate count, T-count, Clifford/dynamic/noisy flags, parameters, memory estimate, recommended simulator and lint findings. Optionally include the instruction list (JSON) and canonical OpenQASM 3. A program with runtime control flow gets blocks, loops, variables and cost per loop instead of a depth.",
        "{\"type\":\"object\",\"properties\":{" + _QASM_PROP + "," + _CIRCUIT_PROP + "," + _PROGRAM_PROP + ",\"include_instructions\":{\"type\":\"boolean\",\"default\":false},\"include_qasm\":{\"type\":\"boolean\",\"default\":false},\"include_lint\":{\"type\":\"boolean\",\"default\":true}},\"additionalProperties\":false}",
        _circuit_info,
    )
    reg.add(
        "qumo_draw", "Draw circuit",
        "Text drawing of a circuit: one line per qubit, columns of gates, @ control, + CX target, x SWAP end, M>c3 measurement into bit 3, ?c0=1 classical condition; detectors, observables and barriers take a whole column. A program draws by blocks, under indented headers. 'style' ascii or unicode; 'width' folds long circuits (0: never).",
        "{\"type\":\"object\",\"properties\":{" + _QASM_PROP + "," + _CIRCUIT_PROP + "," + _PROGRAM_PROP + ",\"width\":{\"type\":\"integer\",\"default\":80},\"style\":{\"type\":\"string\",\"enum\":[\"ascii\",\"unicode\"],\"default\":\"ascii\"}},\"additionalProperties\":false}",
        _draw,
    )
    reg.add(
        "qumo_convert", "Convert circuit",
        "Convert OpenQASM 2/3 (or circuit JSON) to canonical OpenQASM 3 (to='qasm3', round-trip exact, readable pi fractions), to portable OpenQASM 2 (to='qasm2': qelib1.inc plus exact gate definitions, loads in qiskit.qasm2 strict and legacy modes, pytket and Cirq; needs bound parameters, pass 'params' to bind) or to QUMO circuit JSON (to='json'). A program with runtime control flow goes to OpenQASM 3 or to program JSON (qumo.program.v1); OpenQASM 2 cannot hold it (QM-E585). 'params' binds 'input float' parameters of a circuit before converting.",
        "{\"type\":\"object\",\"properties\":{" + _QASM_PROP + "," + _CIRCUIT_PROP + "," + _PROGRAM_PROP + ",\"to\":{\"type\":\"string\",\"enum\":[\"qasm3\",\"qasm2\",\"json\"],\"default\":\"qasm3\"}," + _PARAMS_PROP + "},\"additionalProperties\":false}",
        _convert,
    )
