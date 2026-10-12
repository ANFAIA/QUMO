"""Machine-readable self-description of QUMO (gates, errors, capabilities)
and JSON views of circuits.

Inspiration: PennyLane's device capability TOML files, Amazon Braket's
`DeviceCapabilities`, and the `llms.txt` convention - an agent should be able
to ask the toolkit what it supports instead of guessing from docs. Every
document is generated from the same metadata the compiler and simulators use
(`qumo.ir.gates` opcode tables, `qumo.ai.diagnostics.error_catalog`), so it
cannot drift from the implementation. Output is deterministic (fixed key
order, sorted maps) to keep agent transcripts diffable and cacheable.

Circuit JSON (`format: "qumo.circuit.v1"`) round-trips through
`circuit_from_json`, giving agents a structured alternative to QASM text.
"""

from std.math import pi

from qumo.ir.gates import (
    Op, op_name, op_arity, op_num_params, op_is_unitary, op_is_noise, op_is_annotation,
    op_is_clifford, op_is_diagonal, op_is_self_inverse, op_inverse_kind, op_is_symmetric,
    op_description, matrix_1q, matrix_2q, M2, M4,
)
from qumo.ir.circuit import Circuit, Instruction
from qumo.ir.param import Param
from qumo.ir.qasm import builtin_gate_table, format_param
from qumo.ai.json import JsonWriter, JsonValue, JsonKind
from qumo.ai.diagnostics import error_catalog, error_ranges

comptime QUMO_VERSION = "0.1.0"


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------
def all_opcodes() -> List[Int]:
    var out = List[Int]()
    for k in range(Op.NUM_OPS):
        if op_name(k) != "unknown":
            out.append(k)
    return out^


def op_kind_name(k: Int) -> String:
    if op_is_unitary(k): return "gate"
    if k == Op.MEASURE: return "measurement"
    if k == Op.RESET: return "reset"
    if op_is_noise(k): return "noise"
    return "annotation"


def _param_names(k: Int) -> List[String]:
    var out = List[String]()
    if k == Op.U:
        out.append("theta")
        out.append("phi")
        out.append("lambda")
    elif k == Op.PAULI_CHANNEL_1:
        out.append("px")
        out.append("py")
        out.append("pz")
    elif op_is_noise(k):
        out.append("p")
    elif k == Op.P or k == Op.CP:
        out.append("lambda")
    elif op_num_params(k) == 1:
        out.append("theta")
    return out^


def gate_qasm_example(k: Int) -> String:
    """How to write the opcode in QUMO-flavoured OpenQASM 3."""
    if k == Op.MEASURE: return "c[0] = measure q[0];"
    if k == Op.RESET: return "reset q[0];"
    if k == Op.BARRIER: return "barrier q;"
    if k == Op.DETECTOR: return "#pragma qumo detector 0, 1"
    if k == Op.OBSERVABLE: return "#pragma qumo observable(0) 2"
    if k == Op.TICK: return "#pragma qumo tick"
    var s = String()
    if op_is_noise(k):
        s += "#pragma qumo "
    s += "U" if k == Op.U else op_name(k)
    var pn = _param_names(k)
    if len(pn) > 0:
        s += "("
        for i in range(len(pn)):
            if i > 0:
                s += ", "
            s += pn[i]
        s += ")"
    for i in range(op_arity(k)):
        s += " q[" if i == 0 else ", q["
        s += String(i) + "]"
    return s + ";"


def _in_stdgates(k: Int) -> Bool:
    return op_is_unitary(k) and not (
        k == Op.SXDG or k == Op.ISWAP or k == Op.ISWAPDG or k == Op.ECR
        or k == Op.RXX or k == Op.RYY or k == Op.RZZ or k == Op.CCZ
    )


def _write_m2(mut w: JsonWriter, m: M2):
    w.begin_array()
    for r in range(2):
        w.begin_array()
        for c in range(2):
            w.begin_array()
            w.number(m.re[r * 2 + c])
            w.number(m.im[r * 2 + c])
            w.end_array()
        w.end_array()
    w.end_array()


def _write_m4(mut w: JsonWriter, m: M4):
    w.begin_array()
    for r in range(4):
        w.begin_array()
        for c in range(4):
            w.begin_array()
            w.number(m.re[r * 4 + c])
            w.number(m.im[r * 4 + c])
            w.end_array()
        w.end_array()
    w.end_array()


def write_gate_json(mut w: JsonWriter, k: Int, aliases: List[String]):
    w.begin_object()
    w.kv_string("name", op_name(k))
    w.kv_int("opcode", k)
    w.kv_string("kind", op_kind_name(k))
    w.kv_int("num_qubits", op_arity(k))
    w.kv_int("num_params", op_num_params(k))
    w.kv_string_list("param_names", _param_names(k))
    if op_is_unitary(k):
        w.kv_bool("clifford", op_is_clifford(k))
        w.kv_bool("diagonal", op_is_diagonal(k))
        w.kv_bool("self_inverse", op_is_self_inverse(k))
        w.kv_bool("symmetric", op_is_symmetric(k))
        if op_num_params(k) > 0:
            w.kv_string("inverse", op_name(k) + " with negated parameters" if k != Op.U else "U(-theta, -lambda, -phi)")
        else:
            w.kv_string("inverse", op_name(op_inverse_kind(k)))
        w.kv_bool("in_stdgates_inc", _in_stdgates(k))
    w.kv_string_list("aliases", aliases)
    w.kv_string("qasm", gate_qasm_example(k))
    w.kv_string("description", op_description(k))
    if op_is_unitary(k) and op_num_params(k) == 0:
        if op_arity(k) == 1:
            w.key("matrix")
            _write_m2(w, matrix_1q(k))
        elif op_arity(k) == 2:
            w.key("matrix")
            _write_m4(w, matrix_2q(k))
    w.end_object()


def _aliases_of(k: Int, table_keys: List[String], table_vals: List[Int]) -> List[String]:
    var out = List[String]()
    var canon = op_name(k)
    for i in range(len(table_keys)):
        if table_vals[i] == k and table_keys[i] != canon:
            out.append(table_keys[i])
    sort(out)
    return out^


def describe_gates_json() -> String:
    """Every opcode with arity, params, algebraic flags, QASM syntax, matrix."""
    var t = builtin_gate_table()
    var keys = List[String]()
    var vals = List[Int]()
    for kv in t.items():
        keys.append(kv.key)
        vals.append(kv.value)
    var w = JsonWriter()
    w.begin_object()
    w.kv_string("schema", "qumo.gates.v1")
    w.kv_string("matrix_convention", "row-major, entries [re, im]; basis index = b(q0) + 2*b(q1) (little-endian, Qiskit order); for controlled gates q0 is the control")
    w.key("gates")
    w.begin_array()
    for k in all_opcodes():
        write_gate_json(w, k, _aliases_of(k, keys, vals))
    w.end_array()
    w.key("composite_aliases")
    w.begin_object()
    w.kv_string("u2(phi, lambda)", "U(pi/2, phi, lambda) (Qiskit semantics, no extra global phase)")
    w.kv_string("u3(theta, phi, lambda)", "U(theta, phi, lambda) (Qiskit semantics, no extra global phase)")
    w.kv_string("cu(theta, phi, lambda, gamma) c, t", "p(gamma) c; ctrl @ U(theta, phi, lambda) c, t;")
    w.kv_string("cu3(theta, phi, lambda) c, t", "ctrl @ U(theta, phi, lambda) c, t;")
    w.kv_string("csx c, t", "ctrl @ sx c, t;")
    w.kv_string("gphase(gamma)", "global phase e^{i gamma}")
    w.end_object()
    w.kv_string_list("modifiers", ["inv @", "ctrl @", "ctrl(n) @", "negctrl @", "pow(k) @ (integer k)"])
    w.end_object()
    return w.finish()


def describe_gate_json(name: String) raises -> String:
    """One gate by name or alias (QM-E913 with suggestions when unknown)."""
    var t = builtin_gate_table()
    var code = t.get(name)
    var k = -1
    if code and code.value() >= 0:
        k = code.value()
    else:
        for op in all_opcodes():
            if op_name(op) == name:
                k = op
    if k < 0:
        raise Error("QM-E913: unknown gate '" + name + "'. Hint: call describe with topic 'gates' for the full list.")
    var keys = List[String]()
    var vals = List[Int]()
    for kv in t.items():
        keys.append(kv.key)
        vals.append(kv.value)
    var w = JsonWriter()
    write_gate_json(w, k, _aliases_of(k, keys, vals))
    return w.finish()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
def describe_errors_json() -> String:
    var w = JsonWriter()
    w.begin_object()
    w.kv_string("schema", "qumo.errors.v1")
    w.kv_string("format", "QM-Exxx: <message> (line L, col C). Hint: <fix>  followed by an optional source snippet; QM-Wxxx are lint findings")
    w.key("ranges")
    w.begin_array()
    for r in error_ranges():
        w.begin_object()
        w.kv_string("range", r.code)
        w.kv_string("category", r.category)
        w.kv_string("owner", r.meaning)
        w.end_object()
    w.end_array()
    w.key("codes")
    w.begin_array()
    for e in error_catalog():
        w.begin_object()
        w.kv_string("code", e.code)
        w.kv_string("category", e.category)
        w.kv_string("severity", "warning" if e.code.startswith("QM-W") else "error")
        w.kv_string("meaning", e.meaning)
        w.kv_string("hint", e.hint)
        w.end_object()
    w.end_array()
    w.end_object()
    return w.finish()


def describe_error_json(code: String) raises -> String:
    var c = code.upper()
    for e in error_catalog():
        if e.code == c:
            var w = JsonWriter()
            w.begin_object()
            w.kv_string("code", e.code)
            w.kv_string("category", e.category)
            w.kv_string("meaning", e.meaning)
            w.kv_string("hint", e.hint)
            w.end_object()
            return w.finish()
    raise Error("QM-E913: unknown error code '" + code + "'. Hint: call describe with topic 'errors' for the catalog.")


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------
def describe_capabilities_json(tools: List[String], commands: List[String] = List[String]()) -> String:
    var w = JsonWriter()
    w.begin_object()
    w.kv_string("schema", "qumo.capabilities.v1")
    w.kv_string("name", "qumo")
    w.kv_string("version", QUMO_VERSION)
    w.kv_string("description", "QUMO: a fast, AI-native quantum software stack in Mojo (IR, OpenQASM 2/3, state-vector simulation, compiler, QEC, algorithms).")
    w.key("conventions")
    w.begin_object()
    w.kv_string("qubit_order", "little-endian: bit k of a basis-state index is qubit k")
    w.kv_string("bitstrings", "Qiskit-style: leftmost character is the highest classical bit (c[n-1] ... c[0])")
    w.kv_string("angles", "radians; U(theta, phi, lambda) follows OpenQASM 3")
    w.kv_string("determinism", "every random process takes an explicit seed (default 1234); identical inputs give identical counts")
    w.kv_string("parameters", "affine symbolic parameters coeff*input + offset ('input float theta;'); bind with 'params' as an array (declaration order) or a name->value object")
    w.kv_string("observables", "Pauli sums such as '0.5*Z0 Z1 - 1.2*X0 + 0.3' or dense labels '0.5*ZZI' (leftmost = highest qubit)")
    w.end_object()
    w.key("simulators")
    w.begin_array()
    w.begin_object()
    w.kv_string("name", "statevector")
    w.kv_string_list("methods", ["statevector:static", "statevector:trajectories", "statevector:exact"])
    w.kv_string("memory", "16 * 2^n bytes (n=24: 256 MiB, n=28: 4 GiB, n=30: 16 GiB)")
    w.kv_int("max_qubits_default", 28)
    w.kv_string("selection", "static circuits: one evolution + sampling; mid-circuit measurement, reset, feed-forward or noise: per-shot trajectories")
    w.kv_string_list("features", ["all QUMO gates", "measure", "reset", "single-bit feed-forward", "Pauli noise channels", "symbolic parameters", "exact Pauli expectation values", "SIMD + multithreaded kernels"])
    w.end_object()
    w.end_array()
    w.key("qasm")
    w.begin_object()
    w.kv_string_list("input", ["OpenQASM 2.0", "OpenQASM 3.0"])
    w.kv_string("output", "OpenQASM 3.0 (stdgates.inc + exact gate definitions for non-standard natives)")
    w.kv_string_list("supported", [
        "qubit[n]/qreg, bit[n]/creg (multiple registers are flattened)",
        "all QUMO gates and aliases (cnot, toffoli, u1/u2/u3, cu1/cphase, cu, cu3, csx, U, CX, phase, id)",
        "constant expressions: + - * / % ** pi tau euler sin cos tan arcsin arccos arctan exp ln sqrt abs floor ceil",
        "input float/angle parameters with affine expressions",
        "gate definitions (inlined) and modifiers inv @, ctrl(n) @, negctrl @, pow(k) @",
        "register broadcast, slices q[a:b] (inclusive), index sets q[{0,2}], negative indices, $n physical qubits",
        "measure q -> c; c = measure q; c[i] = measure q[j]; reset; barrier; gphase",
        "if (c[i]) / if (c[i] == v) / if (!c[i]) with else; QASM2 if (creg == v) for 1-bit registers",
        "const declarations and for loops over constant ranges/sets (unrolled)",
        "#pragma qumo: noise channels, detector, observable, tick, param",
    ])
    w.kv_string_list("unsupported", [
        "def/extern subroutines", "while loops", "classical arithmetic on runtime values",
        "multi-bit register conditions", "opaque gates", "pulse-level cal/defcal", "box/delay timing (delay ignored)",
    ])
    w.end_object()
    w.key("error_codes")
    w.begin_array()
    for r in error_ranges():
        w.begin_object()
        w.kv_string("range", r.code)
        w.kv_string("owner", r.meaning)
        w.end_object()
    w.end_array()
    w.kv_string_list("mcp_tools", tools)
    if len(commands) > 0:
        w.kv_string_list("cli_commands", commands)
    w.kv_string("mcp_protocol", "2025-06-18 (stdio, newline-delimited JSON-RPC 2.0)")
    w.end_object()
    return w.finish()


# ---------------------------------------------------------------------------
# Circuits
# ---------------------------------------------------------------------------
def write_param_json(mut w: JsonWriter, p: Param, names: List[String]):
    if not p.is_symbolic():
        w.number(p.offset)
        return
    w.begin_object()
    w.kv_string("expr", format_param(p, names))
    w.kv_string("param", names[p.index] if p.index < len(names) else "$" + String(p.index))
    w.kv_float("coeff", p.coeff)
    w.kv_float("offset", p.offset)
    w.end_object()


def write_circuit_json(circuit: Circuit, mut w: JsonWriter):
    w.begin_object()
    w.kv_string("format", "qumo.circuit.v1")
    w.kv_string("name", circuit.name)
    w.kv_int("num_qubits", circuit.num_qubits)
    w.kv_int("num_clbits", circuit.num_clbits)
    w.kv_float("global_phase", circuit.global_phase)
    w.kv_string_list("params", circuit.param_names)
    w.key("instructions")
    w.begin_array()
    for g in circuit.ops:
        var k = g.kind
        w.begin_object()
        w.kv_string("op", op_name(k))
        if k == Op.DETECTOR or k == Op.OBSERVABLE:
            if k == Op.OBSERVABLE:
                w.kv_int("index", g.q2)
            w.key("records")
            w.begin_array()
            for j in range(g.q1):
                w.integer(circuit.records[g.q0 + j])
            w.end_array()
            w.end_object()
            continue
        var a = op_arity(k)
        if a > 0:
            w.key("qubits")
            w.begin_array()
            for j in range(a):
                w.integer(g.qubit(j))
            w.end_array()
        var np = op_num_params(k)
        if np > 0:
            w.key("params")
            w.begin_array()
            for j in range(np):
                write_param_json(w, g.param(j), circuit.param_names)
            w.end_array()
        if k == Op.MEASURE:
            w.kv_int("clbit", g.clbit)
        if g.cond >= 0:
            w.key("condition")
            w.begin_object()
            w.kv_int("clbit", g.cond)
            w.kv_int("value", g.cond_val)
            w.end_object()
        w.end_object()
    w.end_array()
    w.end_object()


def circuit_to_json(circuit: Circuit) -> String:
    var w = JsonWriter()
    write_circuit_json(circuit, w)
    return w.finish()


def _path_err(path: String, msg: String, hint: String) -> Error:
    return Error("QM-E911: " + path + ": " + msg + ". Hint: " + hint)


def circuit_from_json(v: JsonValue) raises -> Circuit:
    """Inverse of `circuit_to_json` (format qumo.circuit.v1). Gate names may
    be any QASM alias; params are numbers or {param, coeff, offset} objects."""
    if not v.is_object():
        raise _path_err("circuit", "expected an object", "pass the output of circuit_to_json / qumo_convert(to='json')")
    var nq = v.get_int("num_qubits", -1)
    if nq < 0:
        raise Error("QM-E910: circuit.num_qubits is required. Hint: e.g. {\"num_qubits\": 2, \"instructions\": [...]}.")
    var c = Circuit(nq, v.get_int("num_clbits", 0), v.get_string("name", "circuit"))
    c.global_phase = v.get_float("global_phase", 0.0)
    var ps = v.get("params")
    for i in range(len(ps)):
        _ = c.parameter(ps.at(i).as_string())
    var table = builtin_gate_table()
    var ins = v.get("instructions")
    if ins.is_defined() and not ins.is_array():
        raise _path_err("circuit.instructions", "expected an array", "list instruction objects like {\"op\": \"h\", \"qubits\": [0]}")
    for i in range(len(ins)):
        var path = "instructions[" + String(i) + "]"
        var g = ins.at(i)
        var nm = g.get_string("op", "")
        var k: Int
        if nm == "measure": k = Op.MEASURE
        elif nm == "reset": k = Op.RESET
        elif nm == "barrier": k = Op.BARRIER
        elif nm == "tick": k = Op.TICK
        elif nm == "detector": k = Op.DETECTOR
        elif nm == "observable": k = Op.OBSERVABLE
        else:
            var code = table.get(nm)
            if not code or code.value() < 0:
                raise Error("QM-E913: " + path + ": unknown op '" + nm + "'. Hint: use a gate name from describe(gates); composite aliases (u2, u3, cu, csx) are QASM-only.")
            k = code.value()
        if k == Op.DETECTOR or k == Op.OBSERVABLE:
            var recs = List[Int]()
            var rv = g.get("records")
            for j in range(len(rv)):
                recs.append(rv.at(j).as_int())
            if k == Op.DETECTOR:
                c.detector(recs)
            else:
                c.observable(g.get_int("index", 0), recs)
            continue
        if k == Op.BARRIER or k == Op.TICK:
            c.add(k, -1)
            continue
        var qv = g.get("qubits")
        var a = op_arity(k)
        if len(qv) != a:
            raise _path_err(path + ".qubits", "'" + nm + "' needs " + String(a) + " qubit(s), got " + String(len(qv)), "e.g. \"qubits\": [0, 1] for two-qubit gates")
        var qs = List[Int]()
        for j in range(a):
            qs.append(qv.at(j).as_int())
        if k == Op.MEASURE:
            c.measure(qs[0], g.get_int("clbit", qs[0]))
        else:
            var pv = g.get("params")
            var np = op_num_params(k)
            if len(pv) != np:
                raise _path_err(path + ".params", "'" + nm + "' needs " + String(np) + " parameter(s), got " + String(len(pv)), "pass numbers or {\"param\": name, \"coeff\": 1, \"offset\": 0}")
            var pp = List[Param]()
            for j in range(np):
                var x = pv.at(j)
                if x.is_number():
                    pp.append(Param(x.as_float()))
                elif x.is_object():
                    var pn = x.get_string("param", "")
                    var idx = -1
                    for t in range(len(c.param_names)):
                        if c.param_names[t] == pn:
                            idx = t
                    if idx < 0:
                        raise Error("QM-E913: " + path + ".params[" + String(j) + "]: unknown parameter '" + pn + "'. Hint: list it in circuit.params.")
                    pp.append(Param(x.get_float("offset", 0.0), idx, x.get_float("coeff", 1.0)))
                else:
                    raise _path_err(path + ".params[" + String(j) + "]", "expected a number or parameter object", "e.g. 0.5 or {\"param\": \"theta\", \"coeff\": 2.0, \"offset\": 0.0}")
            while len(pp) < 3:
                pp.append(Param(0.0))
            c.add(k, qs[0] if a > 0 else -1, qs[1] if a > 1 else -1, qs[2] if a > 2 else -1, pp[0], pp[1], pp[2])
        var cond = g.get("condition")
        if cond.is_object():
            c.c_if(cond.get_int("clbit", 0), cond.get_int("value", 1))
    c.validate()
    return c^


def write_circuit_summary(circuit: Circuit, mut w: JsonWriter):
    var counts = List[Int](length=Op.NUM_OPS, fill=0)
    for g in circuit.ops:
        if g.kind >= 0 and g.kind < Op.NUM_OPS:
            counts[g.kind] += 1
    var names = List[String]()
    for k in range(Op.NUM_OPS):
        if counts[k] > 0:
            names.append(op_name(k))
    sort(names)
    w.begin_object()
    w.kv_string("name", circuit.name)
    w.kv_int("num_qubits", circuit.num_qubits)
    w.kv_int("num_clbits", circuit.num_clbits)
    w.kv_int("num_instructions", len(circuit.ops))
    w.kv_int("num_gates", circuit.num_unitary_gates())
    w.kv_int("depth", circuit.depth())
    w.kv_int("two_qubit_gates", circuit.num_2q_gates())
    w.kv_int("t_count", circuit.t_count())
    w.key("ops")
    w.begin_object()
    for nm in names:
        for k in range(Op.NUM_OPS):
            if op_name(k) == nm:
                w.kv_int(nm, counts[k])
    w.end_object()
    w.kv_int("num_measurements", circuit.num_measurements)
    w.kv_bool("is_clifford", circuit.is_clifford())
    w.kv_bool("is_dynamic", circuit.is_dynamic())
    w.kv_bool("has_noise", circuit.has_noise())
    w.kv_bool("is_parametric", circuit.is_parametric())
    w.kv_string_list("params", circuit.param_names)
    w.kv_float("global_phase", circuit.global_phase)
    w.kv_int("num_detectors", circuit.num_detectors)
    w.kv_int("num_observables", circuit.num_observables)
    if circuit.num_qubits < 60:
        w.kv_float("statevector_bytes", Float64(1 << circuit.num_qubits) * 16.0)
    var sim = String("statevector")
    if circuit.num_qubits > 28 and circuit.is_clifford():
        sim = "stabilizer (qumo.sim.stabilizer)"
    elif circuit.num_qubits > 28:
        sim = "none locally (exceeds dense state-vector memory)"
    w.kv_string("recommended_simulator", sim)
    w.end_object()


def circuit_summary_json(circuit: Circuit) -> String:
    var w = JsonWriter()
    write_circuit_summary(circuit, w)
    return w.finish()
