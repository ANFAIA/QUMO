"""Circuit and program JSON: the structured alternative to OpenQASM text.

Circuit JSON (`format: "qumo.circuit.v1"`) round-trips through `circuit_from_json`.
Program JSON (`format: "qumo.program.v1"`) wraps it, with the circuit as the arena of quantum ops,
and round-trips through `program_from_json`. Output is deterministic (fixed key order) so agent
transcripts stay diffable and cacheable.

This is IR serialisation, so it lives with the IR; it imports only `qumo.ir` and the JSON helpers of
`qumo.core`. `qumo.ai.describe` re-exports these names for one release.
"""

from std.math import isinf, isnan, pi

from qumo.ir.gates import Op, op_name, op_arity, op_num_params, op_is_unitary, op_is_noise, op_is_annotation
from qumo.ir.circuit import UNUSED_WIRES, Circuit, Instruction
from qumo.ir.param import Param
from qumo.ir.kraus import KrausChannel
from qumo.ir.classical import CFunc, CNodeOp, CNode, CTypeKind, CType, type_name
from qumo.ir.program import CReg, CVar, ForRangeField, MAX_NESTING, Program, StmtKind, Stmt, TapeBuilder, check_program
from qumo.ir.qasm import builtin_gate_table, format_param
from qumo.core.json import JsonWriter, JsonValue, JsonKind, parse_json


def _kv_number(mut w: JsonWriter, key: StringSlice, v: Float64):
    """JSON has no NaN or infinity, and an affine coefficient that overflowed is one."""
    if isnan(v) or isinf(v):
        w.kv_string(key, "nan" if isnan(v) else ("inf" if v > 0.0 else "-inf"))
    else:
        w.kv_float(key, v)


def _read_number(v: JsonValue, key: String, default: Float64) raises -> Float64:
    """Inverse of `_kv_number`."""
    var x = v.get(key)
    if x.is_string():
        var s = x.as_string()
        if s == "nan" or s == "inf" or s == "-inf":
            return Float64(s)
    return v.get_float(key, default)


def write_param_json(mut w: JsonWriter, p: Param, names: List[String]):
    if not p.is_symbolic():
        w.number(p.offset)
        return
    w.begin_object()
    w.kv_string("expr", format_param(p, names))
    w.kv_string("param", names[p.index] if p.index < len(names) else "$" + String(p.index))
    _kv_number(w, "coeff", p.coeff)
    _kv_number(w, "offset", p.offset)
    w.end_object()


def _write_condition_json(mut w: JsonWriter, circuit: Circuit, g: Instruction):
    """`condition`: {clbit, value} tests one bit; a register condition adds
    `bits` and `values`, every one of which must hold."""
    if g.cond < 0:
        return
    w.key("condition")
    w.begin_object()
    w.kv_int("clbit", g.cond)
    w.kv_int("value", g.cond_val)
    if g.cond_reg >= 0:
        var n = circuit.cond_table[g.cond_reg]
        w.key("bits")
        w.begin_array()
        for j in range(n):
            w.integer(circuit.cond_table[g.cond_reg + 1 + 2 * j])
        w.end_array()
        w.key("values")
        w.begin_array()
        for j in range(n):
            w.integer(circuit.cond_table[g.cond_reg + 2 + 2 * j])
        w.end_array()
    w.end_object()


def _read_condition_json(mut c: Circuit, g: JsonValue, path: String, room: Int) raises:
    """Condition of the instruction just appended, as `_write_condition_json` wrote it.
    A bit is 0 or more (a negative one used to mean "no condition"), a value is 0 or 1
    (any other can never hold), and a bit past the register is refused at the end of
    the document, once every measure has grown it."""
    var cond = g.get("condition")
    if not cond.is_defined() or cond.is_null():
        return
    if not cond.is_object():
        raise _path_err(path + ".condition", "expected an object", "e.g. {\"clbit\": 0, \"value\": 1}")
    var bv = cond.get("bits")
    if not bv.is_defined():
        var bit = cond.get_int("clbit", 0)
        var val = cond.get_int("value", 1)
        if bit < 0 or bit >= room:
            raise _path_err(path + ".condition.clbit", "bit " + String(bit) + " is not a classical bit", "use 0 or more, below the register size")
        if val != 0 and val != 1:
            raise _path_err(path + ".condition.value", "value " + String(val) + " is not 0 or 1", "a classical bit holds 0 or 1")
        c.c_if(bit, val)
        return
    var vv = cond.get("values")
    if not bv.is_array() or not vv.is_array() or len(bv) != len(vv) or len(bv) == 0:
        raise _path_err(path + ".condition", "bits and values must be arrays of the same, non-zero length", "e.g. {\"bits\": [0, 1], \"values\": [1, 0]}")
    var bits = List[Int]()
    var vals = List[Int]()
    for j in range(len(bv)):
        var bit = bv.at(j).as_int()
        var val = vv.at(j).as_int()
        if bit < 0 or bit >= room:
            raise _path_err(path + ".condition.bits[" + String(j) + "]", "bit " + String(bit) + " is not a classical bit", "use 0 or more, below the register size")
        if val != 0 and val != 1:
            raise _path_err(path + ".condition.values[" + String(j) + "]", "value " + String(val) + " is not 0 or 1", "a classical bit holds 0 or 1")
        bits.append(bit)
        vals.append(val)
    c.c_if_register(bits, vals)


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
        if k == Op.CORRELATED_ERROR:
            w.kv_float("probability", g.p0.offset)
            w.kv_bool("else", g.q2 != 0)
            w.key("paulis")
            w.begin_array()
            for j in range(g.q1):
                w.begin_array()
                w.integer(circuit.pauli_table[g.q0 + 2 * j])
                w.integer(circuit.pauli_table[g.q0 + 2 * j + 1])
                w.end_array()
            w.end_array()
            _write_condition_json(w, circuit, g)
            w.end_object()
            continue
        var a = op_arity(k)
        if a > 0:
            w.key("qubits")
            w.begin_array()
            for j in range(a):
                w.integer(g.qubit(j))
            w.end_array()
        if k == Op.PAULI_CHANNEL_2:
            w.key("probabilities")
            w.begin_array()
            for j in range(15):
                w.number(circuit.noise_table[Int(g.p0.offset) + j])
            w.end_array()
        if k == Op.KRAUS1 or k == Op.KRAUS2:
            ref ch = circuit.kraus[Int(g.p0.offset)]
            w.kv_string("channel", ch.name)
            w.key("matrices")
            w.begin_array()
            for m in range(ch.count()):
                w.begin_array()
                for j in range(len(ch.re[m])):
                    w.number(ch.re[m][j])
                    w.number(ch.im[m][j])
                w.end_array()
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
        _write_condition_json(w, circuit, g)
        w.end_object()
    w.end_array()
    w.end_object()


def circuit_to_json(circuit: Circuit) -> String:
    var w = JsonWriter()
    write_circuit_json(circuit, w)
    return w.finish()


def _path_err(path: String, msg: String, hint: String) -> Error:
    return Error("QM-E911: " + path + ": " + msg + ". Hint: " + hint)


comptime _MAX_OBSERVABLES = 64  # the DEM and the frame simulators hold observables in a UInt64 mask


def _check_probability(x: Float64, path: String) raises:
    if not (x >= 0.0 and x <= 1.0):
        raise _path_err(path, "probability " + String(x) + " is outside [0, 1]", "a probability is between 0 and 1")


def _check_probability_sum(xs: List[Float64], path: String) raises:
    var total = 0.0
    for x in xs:
        _check_probability(x, path)
        total += x
    if total > 1.0 + 1e-9:
        raise _path_err(path, "the probabilities add up to " + String(total), "the branches of one channel add up to at most 1")


def circuit_from_json(v: JsonValue) raises -> Circuit:
    """Inverse of `circuit_to_json` (format qumo.circuit.v1). Gate names may
    be any QASM alias; params are numbers or {param, coeff, offset} objects. A size
    has to come with the instructions that use it: a short document cannot ask for
    a large allocation further on. The values an instruction carries are checked
    the same way: qubits and clbits against the register, records against the
    measurements made so far (negative counts back), an observable index below 64,
    probabilities in [0, 1], Pauli letters 1 to 3, Kraus matrices of the right size,
    condition bits of 0 or more and condition values of 0 or 1."""
    if not v.is_object():
        raise _path_err("circuit", "expected an object", "pass the output of circuit_to_json / qumo_convert(to='json')")
    var nq = v.get_int("num_qubits", -1)
    if nq < 0:
        raise Error("QM-E910: circuit.num_qubits is required. Hint: e.g. {\"num_qubits\": 2, \"instructions\": [...]}.")
    var ncl = v.get_int("num_clbits", 0)
    var listed = v.get("instructions")
    var room = UNUSED_WIRES + (3 * len(listed) if listed.is_array() else 0)
    if nq > room or ncl < 0 or ncl > room:
        raise _path_err("circuit", "num_qubits and num_clbits go past what its instructions use", "the sizes allow " + String(UNUSED_WIRES) + " unused wires")
    var c = Circuit(nq, ncl, v.get_string("name", "circuit"))
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
        elif nm == "correlated_error": k = Op.CORRELATED_ERROR
        elif nm == "pauli_channel_2": k = Op.PAULI_CHANNEL_2
        elif nm == "kraus1": k = Op.KRAUS1
        elif nm == "kraus2": k = Op.KRAUS2
        else:
            var code = table.get(nm)
            if not code or code.value() < 0:
                raise Error("QM-E913: " + path + ": unknown op '" + nm + "'. Hint: use a gate name from describe(gates); composite aliases (u2, u3, cu, csx) are QASM-only.")
            k = code.value()
        if k == Op.DETECTOR or k == Op.OBSERVABLE:
            var recs = List[Int]()
            var rv = g.get("records")
            if rv.is_defined() and not rv.is_array():
                raise _path_err(path + ".records", "expected an array", "list measurement records, e.g. [0, 1] or [-1]")
            var nmeas = c.num_measurements
            for j in range(len(rv)):
                var r = rv.at(j).as_int()
                if r < -nmeas or r >= nmeas:
                    raise _path_err(path + ".records[" + String(j) + "]", "record " + String(r) + " does not exist yet (" + String(nmeas) + " measurements so far)", "records are absolute measurement indices (0 = first), or negative to count back")
                recs.append(r + nmeas if r < 0 else r)
            if k == Op.DETECTOR:
                c.detector(recs)
            else:
                var index = g.get_int("index", 0)
                if index < 0 or index >= _MAX_OBSERVABLES:
                    raise _path_err(path + ".index", "observable index " + String(index) + " is outside [0, " + String(_MAX_OBSERVABLES) + ")", "the simulators keep observables in a 64-bit mask")
                c.observable(index, recs)
            continue
        if k == Op.BARRIER or k == Op.TICK:
            c.add(k, -1)
            continue
        if k == Op.CORRELATED_ERROR:
            var eq = List[Int]()
            var el = List[Int]()
            var pv = g.get("paulis")
            for j in range(len(pv)):
                var pq = pv.at(j).at(0).as_int()
                var pl = pv.at(j).at(1).as_int()
                if pq < 0 or pq >= nq:
                    raise _path_err(path + ".paulis[" + String(j) + "]", "qubit " + String(pq) + " is outside the " + String(nq) + " qubits", "use 0 to num_qubits - 1")
                if pl < 1 or pl > 3:
                    raise _path_err(path + ".paulis[" + String(j) + "]", "Pauli letter " + String(pl) + " is not 1, 2 or 3", "1 is X, 2 is Y, 3 is Z")
                eq.append(pq)
                el.append(pl)
            var pe = g.get_float("probability", 0.0)
            _check_probability(pe, path + ".probability")
            c.correlated_error(pe, eq, el, g.get_bool("else", False))
            _read_condition_json(c, g, path, room)
            continue
        var qv = g.get("qubits")
        var a = op_arity(k)
        if len(qv) != a:
            raise _path_err(path + ".qubits", "'" + nm + "' needs " + String(a) + " qubit(s), got " + String(len(qv)), "e.g. \"qubits\": [0, 1] for two-qubit gates")
        var qs = List[Int]()
        for j in range(a):
            var q = qv.at(j).as_int()
            if q < 0 or q >= nq:
                raise _path_err(path + ".qubits[" + String(j) + "]", "qubit " + String(q) + " is outside the " + String(nq) + " qubits", "use 0 to num_qubits - 1")
            qs.append(q)
        if k == Op.PAULI_CHANNEL_2:
            var probs = List[Float64]()
            var pv = g.get("probabilities")
            for j in range(len(pv)):
                probs.append(pv.at(j).as_float())
            _check_probability_sum(probs, path + ".probabilities")
            c.pauli_channel_2(probs, qs[0], qs[1])
        elif k == Op.KRAUS1 or k == Op.KRAUS2:
            var ch = KrausChannel(g.get_string("channel", "kraus"), a)
            var mv = g.get("matrices")
            for m in range(len(mv)):
                var re = List[Float64]()
                var im = List[Float64]()
                var row = mv.at(m)
                var dim = 1 << a
                if not row.is_array() or len(row) != 2 * dim * dim:
                    raise _path_err(path + ".matrices[" + String(m) + "]", "a " + String(a) + "-qubit Kraus operator needs " + String(2 * dim * dim) + " numbers (row-major re, im pairs)", "write each entry as two numbers: real, imaginary")
                for j in range(0, len(row), 2):
                    re.append(row.at(j).as_float())
                    im.append(row.at(j + 1).as_float())
                ch.add(re^, im^)
            c.apply_channel(ch, qs[0], qs[1] if a == 2 else -1)
        elif k == Op.MEASURE:
            var cb = g.get_int("clbit", qs[0])
            if cb < 0 or cb >= room:
                raise _path_err(path + ".clbit", "clbit " + String(cb) + " is not a classical bit", "use 0 or more, below " + String(room))
            c.measure(qs[0], cb)
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
                    pp.append(Param(_read_number(x, "offset", 0.0), idx, _read_number(x, "coeff", 1.0)))
                else:
                    raise _path_err(path + ".params[" + String(j) + "]", "expected a number or parameter object", "e.g. 0.5 or {\"param\": \"theta\", \"coeff\": 2.0, \"offset\": 0.0}")
            if k == Op.DEPOLARIZE1 or k == Op.DEPOLARIZE2 or k == Op.X_ERROR or k == Op.Y_ERROR or k == Op.Z_ERROR or k == Op.PAULI_CHANNEL_1:
                var constants = List[Float64]()
                for j in range(np):
                    if not pp[j].is_symbolic():
                        constants.append(pp[j].value())
                if k == Op.PAULI_CHANNEL_1:
                    _check_probability_sum(constants, path + ".params")
                else:
                    for x in constants:
                        _check_probability(x, path + ".params")
            while len(pp) < 3:
                pp.append(Param(0.0))
            c.add(k, qs[0] if a > 0 else -1, qs[1] if a > 1 else -1, qs[2] if a > 2 else -1, pp[0], pp[1], pp[2])
        _read_condition_json(c, g, path, room)
    if c.num_clbits < 0 or c.num_clbits > room:  # a measure grows the register
        raise _path_err("circuit", "an instruction writes past num_clbits", "keep clbit below " + String(room))
    c.validate()
    return c^


# ---------------------------------------------------------------------------
# Programs
# ---------------------------------------------------------------------------
comptime _PROGRAM_FORMAT = "qumo.program.v1"
comptime _MAX_ANALYSIS = 67108864  # blocks x stored values: the assignment analysis keeps a set per block


def _node_names() -> List[String]:
    """Index = CNodeOp code."""
    return ["const", "var", "input", "bits", "neg", "lnot", "bnot", "add", "sub", "mul", "div", "mod", "pow",
            "shl", "shr", "and", "or", "xor", "eq", "ne", "lt", "le", "gt", "ge", "cast", "fn",
            "sc_and", "sc_or", "sc_end"]


def _stmt_names() -> List[String]:
    """Index = StmtKind code."""
    return ["ops", "assign", "if", "else", "while", "for", "switch", "case", "break", "continue", "end"]


def write_program_json(p: Program, mut w: JsonWriter):
    var ops = _node_names()
    var kinds = _stmt_names()
    w.begin_object()
    w.kv_string("format", _PROGRAM_FORMAT)
    w.key("circuit")
    write_circuit_json(p.body, w)
    w.kv_int("num_store", p.num_store)
    w.kv_int_list("user_params", p.user_params)
    w.key("registers")
    w.begin_array()
    for r in p.cregs:
        w.begin_object()
        w.kv_string("name", r.name)
        w.kv_int("first", r.first)
        w.kv_int("width", r.width)
        w.end_object()
    w.end_array()
    w.key("variables")
    w.begin_array()
    for v in p.vars:
        w.begin_object()
        w.kv_string("name", v.name)
        w.kv_string("type", type_name(v.ty))
        w.kv_int("store", v.store)
        w.kv_int("slot", v.slot)
        w.kv_bool("output", v.is_output)
        w.end_object()
    w.end_array()
    w.key("nodes")
    w.begin_array()
    for n in p.nodes:
        w.begin_object()
        w.kv_string("op", ops[n.op])
        w.kv_string("type", type_name(n.ty))
        if n.x != 0: w.kv_int("x", n.x)
        if n.y != 0: w.kv_int("y", n.y)
        if n.i != 0: w.kv_int("i", n.i)
        if String(n.f) != "0.0":
            _kv_number(w, "f", n.f)
        w.end_object()
    w.end_array()
    w.key("expressions")
    w.begin_array()
    for i in range(len(p.expr_lo)):
        w.begin_array()
        w.integer(p.expr_lo[i])
        w.integer(p.expr_root[i])
        w.end_array()
    w.end_array()
    w.kv_int_list("for_ranges", p.for_ranges)
    w.kv_int_list("case_values", p.case_vals)
    w.key("statements")
    w.begin_array()
    for s in p.tape:
        w.begin_object()
        w.kv_string("kind", kinds[s.kind])
        if s.kind == StmtKind.OPS or s.kind == StmtKind.ASSIGN or s.kind == StmtKind.FOR or s.kind == StmtKind.CASE:
            w.kv_int("a", s.a)
            w.kv_int("b", s.b)
        elif s.kind == StmtKind.IF or s.kind == StmtKind.WHILE or s.kind == StmtKind.SWITCH:
            w.kv_int("a", s.a)
        if s.line != 0:
            w.kv_int("line", s.line)
        w.end_object()
    w.end_array()
    w.end_object()


def program_to_json(p: Program) -> String:
    var w = JsonWriter()
    write_program_json(p, w)
    return w.finish()


def _bad_program(msg: String) -> Error:
    return Error("QM-E587: " + msg + ". Hint: pass the output of program_to_json or qumo_convert with to='json'.")


def _json_int(v: JsonValue, what: String) raises -> Int:
    if not v.is_number():
        raise _bad_program(what + " must be an integer, got " + v.kind_name())
    try:
        return v.as_int()
    except:
        raise _bad_program(what + " must be an integer")


def _json_int_or(v: JsonValue, key: String, default: Int, what: String) raises -> Int:
    var x = v.get(key)
    if not x.is_defined():
        return default
    return _json_int(x, what + "." + key)


def _json_ints(v: JsonValue, key: String) raises -> List[Int]:
    var a = v.get(key)
    if not a.is_defined():
        return List[Int]()
    if not a.is_array():
        raise _bad_program(key + " must be an array")
    var out = List[Int](capacity=len(a))
    for i in range(len(a)):
        out.append(_json_int(a.at(i), key + "[" + String(i) + "]"))
    return out^


def _json_string(v: JsonValue, key: String, what: String) raises -> String:
    var x = v.get(key)
    if not x.is_string():
        raise _bad_program(what + "." + key + " must be a string")
    return x.as_string()


def _json_bool_or(v: JsonValue, key: String, default: Bool, what: String) raises -> Bool:
    var x = v.get(key)
    if not x.is_defined() or x.is_null():
        return default
    try:
        return x.as_bool()
    except:
        raise _bad_program(what + " has a '" + key + "' that is not true or false")


def _json_array(v: JsonValue, key: String) raises -> JsonValue:
    var a = v.get(key)
    if not a.is_array():
        raise _bad_program("'" + key + "' must be an array")
    return a^


def _json_float(v: JsonValue, what: String) raises -> Float64:
    try:
        return _read_number(v, "f", 0.0)
    except:
        raise _bad_program(what + ".f must be a number, 'nan', 'inf' or '-inf'")


def _type_from_name(s: String, what: String) raises -> CType:
    """Inverse of `type_name`."""
    if s == "bool": return CType(CTypeKind.BOOL, 1, False)
    if s == "float": return CType(CTypeKind.FLOAT, 64, True)
    if s == "float[64]": return CType(CTypeKind.FLOAT, 64, False)
    if s == "bit": return CType(CTypeKind.BIT, 1, False)
    if s == "int": return CType(CTypeKind.INT, 64, True)
    if s == "uint": return CType(CTypeKind.UINT, 64, True)
    var open = s.find("[")
    if open > 0 and s.endswith("]"):
        var nm = String(s[byte=0:open])
        var width: Int
        try:
            width = Int(String(s[byte=open + 1:s.byte_length() - 1]))
        except:
            raise _bad_program(what + ": bad type '" + s + "'")
        var tag = CTypeKind.BIT if nm == "bit" else (CTypeKind.INT if nm == "int" else (CTypeKind.UINT if nm == "uint" else (CTypeKind.ANGLE if nm == "angle" else -1)))
        if tag >= 0 and width >= 1 and width <= 64:
            return CType(tag, width, False)
    raise _bad_program(what + ": bad type '" + s + "'")


def _is_float(t: CType) -> Bool:
    return t.tag == CTypeKind.FLOAT


def _operand(last: List[Int], lo: Int, at: Int, pc: Int) raises:
    if at < lo or at >= pc or last[at - lo] < pc:
        raise _bad_program("nodes[" + String(pc) + "] has an operand it cannot read")


def _typed(p: Program, last: List[Int], lo: Int, at: Int, pc: Int, float_: Bool) raises:
    _operand(last, lo, at, pc)
    if _is_float(p.nodes[at].ty) != float_:
        raise _bad_program("nodes[" + String(pc) + "] has an operand of the wrong kind")


def _check_expression(p: Program, lo: Int, root: Int) raises:
    """What `eval_nodes` takes on trust: each operand comes before its node, in this
    expression, of the kind the node reads. A short-circuit jumps over its right
    side, so a node inside one can be read only up to its SC_END, and the jump itself
    has no value to read. `last[i]` is the last node that may read node lo + i."""
    var last = List[Int](capacity=root - lo + 1)
    var open = List[Int]()  # SC_END of each open short-circuit, innermost last
    for pc in range(lo, root + 1):
        ref n = p.nodes[pc]
        var op = n.op
        var fl = _is_float(n.ty)
        if op == CNodeOp.SC_END:
            if len(open) == 0 or open[len(open) - 1] != pc:
                raise _bad_program("nodes[" + String(pc) + "] ends a short-circuit that nothing started")
            _typed(p, last, lo, n.x, pc, False)
            _ = open.pop()
        last.append(open[len(open) - 1] if len(open) > 0 else root)
        if op == CNodeOp.CONST or op == CNodeOp.SC_END:
            pass
        elif op == CNodeOp.VAR:
            if n.x < 0 or n.x >= p.num_store:
                raise _bad_program("nodes[" + String(pc) + "] reads a variable outside the store")
        elif op == CNodeOp.INPUT:
            if n.x < 0 or n.x >= p.body.num_params() or not fl:
                raise _bad_program("nodes[" + String(pc) + "] reads a parameter outside the circuit")
        elif op == CNodeOp.BITS:
            if n.x < 0 or n.y < 1 or n.y > 64 or n.x > p.num_clbits() - n.y or fl:
                raise _bad_program("nodes[" + String(pc) + "] reads bits outside the circuit")
        elif op == CNodeOp.CAST:
            _operand(last, lo, n.x, pc)
        elif op == CNodeOp.NEG:
            _typed(p, last, lo, n.x, pc, fl)
        elif op == CNodeOp.LNOT or op == CNodeOp.BNOT:
            _typed(p, last, lo, n.x, pc, False)
        elif op == CNodeOp.ADD or op == CNodeOp.SUB or op == CNodeOp.MUL or op == CNodeOp.DIV or op == CNodeOp.MOD or op == CNodeOp.POW:
            _typed(p, last, lo, n.x, pc, fl)
            _typed(p, last, lo, n.y, pc, fl)
        elif op == CNodeOp.SHL or op == CNodeOp.SHR or op == CNodeOp.AND or op == CNodeOp.OR or op == CNodeOp.XOR:
            _typed(p, last, lo, n.x, pc, False)
            _typed(p, last, lo, n.y, pc, False)
        elif op >= CNodeOp.EQ and op <= CNodeOp.GE:
            _operand(last, lo, n.x, pc)
            _typed(p, last, lo, n.y, pc, _is_float(p.nodes[n.x].ty))
        elif op == CNodeOp.SC_AND or op == CNodeOp.SC_OR:
            _typed(p, last, lo, n.x, pc, False)
            var stop = open[len(open) - 1] if len(open) > 0 else root + 1
            if n.y <= pc or n.y >= stop or p.nodes[n.y].op != CNodeOp.SC_END:
                raise _bad_program("nodes[" + String(pc) + "] skips to a node that does not end it")
            last[pc - lo] = -1
            open.append(n.y)
        elif op == CNodeOp.FN:
            if n.i < 0 or n.i > CFunc.POPCOUNT:
                raise _bad_program("nodes[" + String(pc) + "] calls an unknown function")
            var on_ints = n.i == CFunc.POPCOUNT or (n.i == CFunc.ABS and not fl)
            _typed(p, last, lo, n.x, pc, not on_ints)
            if n.y != -1:
                _typed(p, last, lo, n.y, pc, not on_ints)
            elif n.i == CFunc.MOD:
                raise _bad_program("nodes[" + String(pc) + "] calls mod with one operand")


def _check_arena(p: Program) raises:
    """What the circuit part holds that only the program can judge: where measures
    write, and that every symbol is an input or the slot of a variable."""
    var known = List[Bool](length=p.body.num_params(), fill=False)
    for i in p.user_params:
        known[i] = True
    for v in p.vars:
        if v.slot >= 0:
            known[v.slot] = True
    for g in p.body.ops:
        if g.kind == Op.MEASURE and (g.clbit < 0 or g.clbit >= p.num_clbits()):
            raise _bad_program("a measure writes outside the circuit's clbits")
        for j in range(op_num_params(g.kind)):
            var q = g.param(j)
            if q.is_symbolic() and not known[q.index]:
                raise _bad_program("a gate reads a parameter that is neither an input nor a variable's slot")


def _check_tables(p: Program) raises:
    """Every index in the classical tables of a program read from a file. Nothing
    runs on trust in the file: a bad index here would be a crash in a shot. Bounds
    compare without adding, because a huge index would overflow the sum."""
    var nv = p.num_store
    for i in p.user_params:
        if i < 0 or i >= p.body.num_params():
            raise _bad_program("user_params names a parameter outside the circuit")
    for r in p.cregs:
        if r.first < 0 or r.width < 0 or r.first > p.num_clbits() - r.width:
            raise _bad_program("register '" + r.name + "' lies outside the circuit")
    for v in p.vars:
        if v.ty.tag == CTypeKind.BIT:
            if v.store < 0 or v.store > p.num_clbits() - v.ty.width:
                raise _bad_program("variable '" + v.name + "' lies outside the circuit")
        elif v.store < 0 or v.store >= nv:
            raise _bad_program("variable '" + v.name + "' lies outside the store")
        if v.slot < -1 or v.slot >= p.body.num_params() or (v.slot >= 0 and v.ty.tag == CTypeKind.BIT):
            raise _bad_program("variable '" + v.name + "' has a slot it cannot hold")
    if len(p.expr_lo) != len(p.expr_root):
        raise _bad_program("expressions must be [lo, root] pairs")
    for e in range(len(p.expr_lo)):
        var lo = p.expr_lo[e]
        var root = p.expr_root[e]
        if lo < 0 or root < lo or root >= len(p.nodes):
            raise _bad_program("expressions[" + String(e) + "] lies outside the nodes")
        _check_expression(p, lo, root)


def _is_float_expr(p: Program, e: Int) -> Bool:
    return _is_float(p.nodes[p.expr_root[e]].ty)


def _check_statement(p: Program, s: Stmt, at: Int) raises:
    var ne = len(p.expr_lo)
    var what = "statements[" + String(at) + "]"
    if s.kind == StmtKind.OPS:
        if s.a < 0 or s.b < s.a or s.b > len(p.body.ops):
            raise _bad_program(what + " lies outside the circuit")
    elif s.kind == StmtKind.ASSIGN:
        if s.a < 0 or s.a >= len(p.vars) or s.b < 0 or s.b >= ne:
            raise _bad_program(what + " names a variable or expression that does not exist")
        if _is_float(p.vars[s.a].ty) != _is_float_expr(p, s.b):
            raise _bad_program(what + " assigns a value of the wrong kind")
    elif s.kind == StmtKind.IF or s.kind == StmtKind.WHILE or s.kind == StmtKind.SWITCH:
        if s.a < 0 or s.a >= ne:
            raise _bad_program(what + " names an expression that does not exist")
        if _is_float_expr(p, s.a):
            raise _bad_program(what + " tests a float")
    elif s.kind == StmtKind.CASE:
        if s.a != -1 and (s.a < 0 or s.b < s.a or s.b > len(p.case_vals)):
            raise _bad_program(what + " lies outside case_values")
    elif s.kind == StmtKind.FOR:
        if s.a < 0 or s.a >= len(p.vars) or s.b < 0 or s.b > len(p.for_ranges) - ForRangeField.SIZE:
            raise _bad_program(what + " names a variable or range that does not exist")
        var tag = p.vars[s.a].ty.tag
        if tag != CTypeKind.INT and tag != CTypeKind.UINT:
            raise _bad_program(what + " counts in a variable that is not int or uint")
        var store = p.for_ranges[s.b + ForRangeField.STORE]
        if store < 0 or store > p.num_store - 3:
            raise _bad_program(what + " keeps its state outside the store")
        var flags = p.for_ranges[s.b + ForRangeField.UNSIGNED]
        if flags < 0 or flags > ForRangeField.ALL_FLAGS:
            raise _bad_program(what + " has flags that do not exist")
        var set_at = p.for_ranges[s.b + ForRangeField.SET]
        if set_at < -1:
            raise _bad_program(what + " has a set outside case_values")
        if set_at >= 0:
            if set_at >= len(p.case_vals) or p.case_vals[set_at] < 0 or p.case_vals[set_at] > len(p.case_vals) - set_at - 1:
                raise _bad_program(what + " has a set outside case_values")
        for f in range(ForRangeField.START, ForRangeField.STOP + 1):
            var e = p.for_ranges[s.b + f]
            if e == -1 and set_at >= 0:
                continue  # a set has no range
            if e < 0 or e >= ne or _is_float_expr(p, e):
                raise _bad_program(what + " has a bound that is not an integer expression")


def program_from_json(v: JsonValue) raises -> Program:
    """Inverse of `program_to_json`. Every index is checked, and the control-flow
    links are built again from the statements (QM-E583 when they do not nest)."""
    if not v.is_object():
        raise _bad_program("a program is an object")
    var format_ = v.get("format")
    if not format_.is_string() or format_.as_string() != _PROGRAM_FORMAT:
        raise _bad_program("format must be '" + _PROGRAM_FORMAT + "'")
    var arena: Circuit
    var declared: Int
    try:
        arena = circuit_from_json(v.get("circuit"))
        declared = v.get("circuit").get_int("num_clbits", 0)
    except e:
        raise _bad_program("its circuit does not load: " + String(e))
    if arena.num_clbits != declared:  # a measure past the end grew them
        raise _bad_program("a measure writes outside the circuit's clbits")
    var p = Program(arena^)
    # what the document can justify: one store per variable, three per FOR
    var justified = len(_json_array(v, "variables")) + len(_json_ints(v, "for_ranges")) // ForRangeField.SIZE * 3
    p.num_store = _json_int_or(v, "num_store", 0, "program")
    if p.num_store < 0 or p.num_store > justified:
        raise _bad_program("num_store is more than its variables and loops use")
    p.user_params = _json_ints(v, "user_params")
    var regs = _json_array(v, "registers")
    for i in range(len(regs)):
        var r = regs.at(i)
        var what = "registers[" + String(i) + "]"
        p.cregs.append(CReg(_json_string(r, "name", what), _json_int_or(r, "first", -1, what), _json_int_or(r, "width", -1, what)))
    var vars = _json_array(v, "variables")
    for i in range(len(vars)):
        var x = vars.at(i)
        var what = "variables[" + String(i) + "]"
        p.vars.append(CVar(_json_string(x, "name", what), _type_from_name(_json_string(x, "type", what), what),
                           _json_int_or(x, "store", -1, what), _json_int_or(x, "slot", -1, what), _json_bool_or(x, "output", False, what)))
    var names = _node_names()
    var nodes = _json_array(v, "nodes")
    for i in range(len(nodes)):
        var x = nodes.at(i)
        var what = "nodes[" + String(i) + "]"
        var nm = _json_string(x, "op", what)
        var op = -1
        for k in range(len(names)):
            if names[k] == nm:
                op = k
        if op < 0:
            raise _bad_program(what + ": unknown op '" + nm + "'")
        p.nodes.append(CNode(op, _type_from_name(_json_string(x, "type", what), what), _json_int_or(x, "x", 0, what),
                             _json_int_or(x, "y", 0, what), _json_int_or(x, "i", 0, what), _json_float(x, what)))
    var exprs = _json_array(v, "expressions")
    for i in range(len(exprs)):
        var what = "expressions[" + String(i) + "]"
        var e = exprs.at(i)
        if not e.is_array() or len(e) != 2:
            raise _bad_program(what + " must be [lo, root]")
        p.expr_lo.append(_json_int(e.at(0), what))
        p.expr_root.append(_json_int(e.at(1), what))
    p.for_ranges = _json_ints(v, "for_ranges")
    p.case_vals = _json_ints(v, "case_values")
    _check_tables(p)
    _check_arena(p)
    var kinds = _stmt_names()
    var stmts = _json_array(v, "statements")
    var t = TapeBuilder()
    var ops_end = 0  # blocks cover the arena in order, as every builder writes them
    var blocks = 1
    for i in range(len(stmts)):
        var x = stmts.at(i)
        var what = "statements[" + String(i) + "]"
        var nm = _json_string(x, "kind", what)
        var kind = -1
        for k in range(len(kinds)):
            if kinds[k] == nm:
                kind = k
        if kind < 0:
            raise _bad_program(what + ": unknown kind '" + nm + "'")
        var s = Stmt(kind, _json_int_or(x, "a", -1, what), _json_int_or(x, "b", -1, what), -1, -1, _json_int_or(x, "line", 0, what))
        _check_statement(p, s, i)
        if kind != StmtKind.OPS and kind != StmtKind.ASSIGN:
            blocks += 1
        if len(t.open) >= MAX_NESTING and (kind == StmtKind.IF or kind == StmtKind.WHILE or kind == StmtKind.FOR or kind == StmtKind.SWITCH):
            raise _bad_program(what + " nests past " + String(MAX_NESTING) + " levels")
        if len(t.tape) > 0 and t.tape[len(t.tape) - 1].kind == StmtKind.SWITCH and kind != StmtKind.CASE and kind != StmtKind.END:
            raise Error("QM-E583: a statement between a switch and its first case. Hint: every statement of a switch sits in a case.")
        if kind == StmtKind.OPS:
            if s.a != ops_end or (len(t.tape) > 0 and t.tape[len(t.tape) - 1].kind == StmtKind.OPS):
                raise _bad_program(what + " breaks the order of the blocks")
            ops_end = s.b
            t.tape.append(s)
        elif kind == StmtKind.ASSIGN:
            t.tape.append(s)
        elif kind == StmtKind.IF or kind == StmtKind.WHILE or kind == StmtKind.FOR or kind == StmtKind.SWITCH:
            t.begin(kind, s.a, s.b if kind == StmtKind.FOR else -1, 0, s.line)
        elif kind == StmtKind.ELSE:
            t.else_(0, s.line)
        elif kind == StmtKind.CASE:
            t.case_(s.a, s.b, 0, s.line)
        elif kind == StmtKind.BREAK or kind == StmtKind.CONTINUE:
            t.jump(kind, 0, s.line)
        else:
            t.end(0, s.line)
    if len(t.open) != 0:
        raise Error("QM-E583: a region of the program is left open. Hint: every if, while, for and switch needs its end.")
    if ops_end != len(p.body.ops):
        raise _bad_program("the statements leave ops of the circuit out")
    if p.num_store > _MAX_ANALYSIS // blocks:
        raise _bad_program("its blocks and variables are too many to analyse")
    p.tape = t.tape.copy()
    _ = check_program(p)
    return p^
