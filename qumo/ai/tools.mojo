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
from qumo.ai.json import JsonValue
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

from qumo.ir.circuit import Circuit
from qumo.ir.gates import Op
from qumo.ir.pauli import PauliSum, parse_pauli_sum
from qumo.ir.qasm import parse_qasm_with_warnings, to_qasm
from qumo.sim.runner import sample, evolve, bitstring
from qumo.exec.dispatch import sample_auto, choose_method
from qumo.ai.json import JsonWriter, JsonValue, JsonKind, parse_json, json_quote
from qumo.ai.diagnostics import (
    Diagnostic, diagnostic_from_error, lint, parse_error_code,
)
from qumo.ai.describe import (
    QUMO_VERSION, describe_gates_json, describe_gate_json, describe_errors_json,
    describe_error_json, describe_capabilities_json, write_circuit_json,
    write_circuit_summary, circuit_from_json,
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


def arg_circuit(args: JsonValue, mut warnings: List[String]) raises -> Circuit:
    """Circuit from `qasm` (OpenQASM 2/3 text) or `circuit` (qumo.circuit.v1 JSON)."""
    var q = args.get("qasm")
    if q.is_defined() and not q.is_null():
        return parse_qasm_with_warnings(q.as_string(), warnings)
    var c = args.get("circuit")
    if c.is_defined() and not c.is_null():
        return circuit_from_json(c)
    raise Error("QM-E910: missing required argument 'qasm' (OpenQASM 2/3 source text). Hint: e.g. {\"qasm\": \"OPENQASM 3.0; include \\\"stdgates.inc\\\"; qubit[2] q; h q[0]; cx q[0], q[1];\"}.")


def arg_params(args: JsonValue, circuit: Circuit) raises -> List[Float64]:
    """`params`: array (declaration order) or object name -> value."""
    var out = List[Float64]()
    var p = args.get("params")
    var n = circuit.num_params()
    if not p.is_defined() or p.is_null():
        if n > 0:
            raise Error("QM-E101: circuit has " + String(n) + " parameter(s) " + circuit.param_names_str() + " but no 'params' were given. Hint: pass \"params\": [values...] or {\"name\": value}.")
        return out^
    if p.is_array():
        if len(p) != n:
            raise Error("QM-E101: 'params' has " + String(len(p)) + " value(s), circuit has " + String(n) + " parameter(s) " + circuit.param_names_str() + ". Hint: one value per parameter, in this order.")
        for i in range(n):
            out.append(p.at(i).as_float())
        return out^
    if p.is_object():
        for i in range(n):
            var v = p.get(circuit.param_names[i])
            if not v.is_defined():
                raise Error("QM-E101: missing value for parameter '" + circuit.param_names[i] + "'. Hint: 'params' must name every parameter of " + circuit.param_names_str() + ".")
            out.append(v.as_float())
        for k in p.keys():
            var found = False
            for nm in circuit.param_names:
                if nm == k:
                    found = True
            if not found:
                raise Error("QM-E913: unknown parameter '" + k + "' in 'params'. Hint: circuit parameters are " + circuit.param_names_str() + ".")
        return out^
    raise Error("QM-E911: field 'params': expected an array or object, got " + p.kind_name() + ". Hint: e.g. \"params\": [0.1, 0.2] or {\"theta\": 0.1}.")


def _ms(t0: Int, t1: Int) -> Float64:
    return Float64(t1 - t0) / 1e6


def _write_warnings(mut w: JsonWriter, warnings: List[String]):
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
comptime _PARAMS_PROP = "\"params\":{\"description\":\"Values for 'input float' parameters: array in declaration order or object name->value\",\"oneOf\":[{\"type\":\"array\",\"items\":{\"type\":\"number\"}},{\"type\":\"object\",\"additionalProperties\":{\"type\":\"number\"}}]}"


def _run_qasm(args: JsonValue, ctx: ToolContext) raises -> ToolResult:
    var t0 = perf_counter_ns()
    var warnings = List[String]()
    var c = arg_circuit(args, warnings)
    var t1 = perf_counter_ns()
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
    _write_warnings(w, warnings)
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
    _write_warnings(w, warnings)
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
    _write_warnings(w, warnings)
    w.end_object()
    return tool_ok(w.finish())


def _validate(args: JsonValue, ctx: ToolContext) raises -> ToolResult:
    var src = arg_required(args, "qasm", "OpenQASM source text").as_string()
    var warnings = List[String]()
    var w = JsonWriter()
    w.begin_object()
    var c: Circuit
    try:
        c = parse_qasm_with_warnings(src, warnings)
    except e:
        var d = diagnostic_from_error(String(e))
        w.kv_bool("valid", False)
        w.key("errors")
        w.begin_array()
        d.write_json(w)
        w.end_array()
        _write_warnings(w, warnings)
        w.end_object()
        return ToolResult(w.finish(), "INVALID\n" + String(d), False)
    var diags = lint(c)
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
    write_circuit_summary(c, w)
    w.end_object()
    return tool_ok(w.finish())


def _describe(args: JsonValue, ctx: ToolContext) raises -> ToolResult:
    var topic = args.get_string("topic", "capabilities")
    var name = args.get_string("name", "")
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
    raise Error("QM-E913: unknown topic '" + topic + "'. Hint: topic is one of gates, gate, errors, error, capabilities.")


def _circuit_info(args: JsonValue, ctx: ToolContext) raises -> ToolResult:
    var warnings = List[String]()
    var c = arg_circuit(args, warnings)
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
    _write_warnings(w, warnings)
    w.end_object()
    return tool_ok(w.finish())


def _convert(args: JsonValue, ctx: ToolContext) raises -> ToolResult:
    var warnings = List[String]()
    var c = arg_circuit(args, warnings)
    var to = args.get_string("to", "qasm3")
    var w = JsonWriter()
    w.begin_object()
    if to == "qasm3" or to == "qasm" or to == "openqasm3":
        var text = to_qasm(c)
        w.kv_string("format", "openqasm3")
        w.kv_string("qasm", text)
        _write_warnings(w, warnings)
        w.end_object()
        return ToolResult(w.finish(), text, False)
    if to == "json":
        w.kv_string("format", "qumo.circuit.v1")
        w.key("circuit")
        write_circuit_json(c, w)
        _write_warnings(w, warnings)
        w.end_object()
        return tool_ok(w.finish())
    raise Error("QM-E913: unknown target format '" + to + "'. Hint: 'to' is 'qasm3' or 'json'.")


def register_core_tools(mut reg: ToolRegistry):
    reg.add(
        "qumo_run_qasm", "Run circuit (sample)",
        "Simulate an OpenQASM 2/3 circuit and return measurement counts (Qiskit-style bitstrings, leftmost = highest bit). Deterministic for a given seed. Circuits without measurements are measured on all qubits. Supports mid-circuit measurement, reset, single-bit feed-forward and '#pragma qumo' noise (per-shot trajectories).",
        "{\"type\":\"object\",\"properties\":{" + _QASM_PROP + "," + _CIRCUIT_PROP + ",\"shots\":{\"type\":\"integer\",\"minimum\":1,\"maximum\":10000000,\"default\":1024},\"seed\":{\"type\":\"integer\",\"minimum\":0,\"default\":1234}," + _PARAMS_PROP + ",\"max_qubits\":{\"type\":\"integer\",\"default\":28,\"description\":\"Safety limit for dense simulation (16*2^n bytes)\"}},\"additionalProperties\":false}",
        _run_qasm,
    )
    reg.add(
        "qumo_statevector", "Final state vector",
        "Return the exact final state vector (amplitudes with probability > threshold, sorted by basis index) of a unitary circuit. Measurements and resets are ignored; classically-conditioned gates are rejected (QM-E401).",
        "{\"type\":\"object\",\"properties\":{" + _QASM_PROP + "," + _CIRCUIT_PROP + "," + _PARAMS_PROP + ",\"max_amplitudes\":{\"type\":\"integer\",\"default\":64,\"minimum\":1},\"threshold\":{\"type\":\"number\",\"default\":1e-12},\"max_qubits\":{\"type\":\"integer\",\"default\":24}},\"additionalProperties\":false}",
        _statevector,
    )
    reg.add(
        "qumo_expectation", "Pauli expectation value",
        "Exact expectation value <psi|H|psi> of a Pauli-sum observable (e.g. '0.5*Z0 Z1 - 1.2*X0 + 0.3' or dense 'ZZI') on the circuit's final state. Pass an array of observables to evaluate several at once.",
        "{\"type\":\"object\",\"properties\":{" + _QASM_PROP + "," + _CIRCUIT_PROP + ",\"observable\":{\"description\":\"Pauli sum string, or array of strings\",\"oneOf\":[{\"type\":\"string\"},{\"type\":\"array\",\"items\":{\"type\":\"string\"}}]}," + _PARAMS_PROP + ",\"max_qubits\":{\"type\":\"integer\",\"default\":28}},\"required\":[\"observable\"],\"additionalProperties\":false}",
        _expectation,
    )
    reg.add(
        "qumo_validate_qasm", "Validate OpenQASM",
        "Parse and lint OpenQASM 2/3 without simulating. Returns valid=true/false, structured errors (stable QM-Exxx code, message, line, col, snippet, hint) and lint warnings (QM-Wxxx), plus a circuit summary when valid.",
        "{\"type\":\"object\",\"properties\":{" + _QASM_PROP + "},\"required\":[\"qasm\"],\"additionalProperties\":false}",
        _validate,
    )
    reg.add(
        "qumo_describe", "Describe QUMO",
        "Machine-readable self-description: topic 'gates' (every gate: arity, params, Clifford/diagonal flags, QASM syntax, matrix), 'gate' + name, 'errors' (error-code catalog with fix hints), 'error' + name (one code), or 'capabilities' (simulators, QASM subset, conventions, tools).",
        "{\"type\":\"object\",\"properties\":{\"topic\":{\"type\":\"string\",\"enum\":[\"gates\",\"gate\",\"errors\",\"error\",\"capabilities\"],\"default\":\"capabilities\"},\"name\":{\"type\":\"string\",\"description\":\"Gate name/alias or error code for topic gate/error\"}},\"additionalProperties\":false}",
        _describe,
    )
    reg.add(
        "qumo_circuit_info", "Circuit summary",
        "Analyze a circuit without simulating: qubits, clbits, op counts, depth, two-qubit gate count, T-count, Clifford/dynamic/noisy flags, parameters, memory estimate, recommended simulator and lint findings. Optionally include the instruction list (JSON) and canonical OpenQASM 3.",
        "{\"type\":\"object\",\"properties\":{" + _QASM_PROP + "," + _CIRCUIT_PROP + ",\"include_instructions\":{\"type\":\"boolean\",\"default\":false},\"include_qasm\":{\"type\":\"boolean\",\"default\":false},\"include_lint\":{\"type\":\"boolean\",\"default\":true}},\"additionalProperties\":false}",
        _circuit_info,
    )
    reg.add(
        "qumo_convert", "Convert circuit",
        "Convert OpenQASM 2/3 (or circuit JSON) to canonical OpenQASM 3 (to='qasm3', round-trip exact, readable pi fractions) or to QUMO circuit JSON (to='json').",
        "{\"type\":\"object\",\"properties\":{" + _QASM_PROP + "," + _CIRCUIT_PROP + ",\"to\":{\"type\":\"string\",\"enum\":[\"qasm3\",\"json\"],\"default\":\"qasm3\"}},\"additionalProperties\":false}",
        _convert,
    )
