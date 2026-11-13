"""OpenQASM emitter: Circuit and Program -> OpenQASM 3 text.

Part of the OpenQASM output side, split from qumo/ir/qasm.mojo (which re-exports every name).
Depends on the lexer only (names and identifier rules); nothing depends on it except the facade.
"""

from std.math import pi, sin, cos, tan, sqrt, exp, log, floor, ceil, isnan, isinf
from qumo.ir.gates import Op, op_name, op_arity, op_num_params, op_is_noise
from qumo.ir.circuit import Circuit, Instruction
from qumo.ir.param import Param
from qumo.ir.classical import CN, CF, CT, CType, CNode, type_name, bool_type, int_type, angle_radians
from qumo.ir.program import FORR, Program, SK, Stmt
from qumo.ir.qasm_lexer import _is_digit, bits_to_decimal, builtin_gate_table


def register_condition_value(c: Circuit, g: Instruction) raises -> String:
    """`N` such that the condition reads `c == N` over the whole creg."""
    var n = c.cond_table[g.cond_reg]
    if n != c.num_clbits:
        raise Error("QM-E552: register condition covers " + String(n) + " of " + String(c.num_clbits) + " clbits; OpenQASM output has a single creg. Hint: condition on single bits or on the full register.")
    var bits = List[Int](length=c.num_clbits, fill=-1)
    for i in range(n):
        bits[c.cond_table[g.cond_reg + 1 + 2 * i]] = c.cond_table[g.cond_reg + 2 + 2 * i]
    for b in bits:
        if b < 0:
            raise Error("QM-E552: register condition does not cover every clbit.")
    return bits_to_decimal(bits)

# ===========================================================================
# Emitter
# ===========================================================================
def _reserved_words() -> List[String]:
    var w: List[String] = [
        "q", "c", "pi", "tau", "euler", "OPENQASM", "include", "qubit", "qreg", "bit", "creg",
        "input", "output", "const", "gate", "measure", "reset", "barrier", "if", "else",
        "for", "while", "in", "let", "def", "return", "int", "uint", "float", "angle",
        "bool", "complex", "duration", "stretch", "true", "false", "ctrl", "negctrl", "inv",
        "pow", "gphase", "delay", "box", "opaque", "extern", "array", "void", "sin", "cos",
        "tan", "arcsin", "arccos", "arctan", "exp", "ln", "log", "sqrt", "mod", "abs",
        "floor", "ceil", "pragma", "end", "break", "continue", "switch", "case", "default",
        "cal", "defcal", "defcalgrammar", "readonly", "mutable", "dt", "ns", "us", "ms", "s",
    ]
    var t = builtin_gate_table()
    for k in t.keys():
        w.append(k)
    return w^


def _contains(lst: List[String], s: String) -> Bool:
    for x in lst:
        if x == s:
            return True
    return False


def _sanitize_ident(name: String, reserved: List[String], used: List[String]) -> String:
    var bs = name.as_bytes()
    var out = String()
    for i in range(len(bs)):
        var c = Int(bs[i])
        if (c >= 0x61 and c <= 0x7A) or (c >= 0x41 and c <= 0x5A) or c == 0x5F or (c >= 0x30 and c <= 0x39):
            out += String(unsafe_from_utf8=bs[i:i + 1])
        else:
            var ob = out.as_bytes()
            if len(ob) == 0 or Int(ob[len(ob) - 1]) != 0x5F:
                out += "_"
    var ob = out.as_bytes()
    var e = len(ob)
    while e > 0 and Int(ob[e - 1]) == 0x5F:
        e -= 1
    var trimmed = String(out[byte=0:e])
    out = trimmed^
    if out.byte_length() == 0 or _is_digit(Int(out.as_bytes()[0])):
        out = "p_" + out
    if _contains(reserved, out) or _contains(used, out):
        var k = 1
        while _contains(reserved, out + "_" + String(k)) or _contains(used, out + "_" + String(k)):
            k += 1
        out = out + "_" + String(k)
    return out^


def format_angle(v: Float64) -> String:
    """Readable exact angle: `pi/2`, `-3*pi/4`, ... when the parser re-evaluates
    the text to the identical Float64; otherwise shortest round-trip decimal."""
    if v == 0.0:
        return "0"
    if isnan(v) or isinf(v) or abs(v) > 1e6:
        return String(v)
    var dens: List[Int] = [1, 2, 3, 4, 6, 8, 12, 16, 32, 64]
    for d in dens:
        var kf = v * Float64(d) / pi
        var k = Int(round(kf))
        if k == 0 or abs(kf - Float64(k)) > 1e-9 or abs(k) > 4096:
            continue
        var val: Float64
        var txt: String
        if d == 1:
            if k == 1:
                val = pi
                txt = "pi"
            elif k == -1:
                val = -pi
                txt = "-pi"
            else:
                val = Float64(k) * pi
                txt = String(k) + "*pi"
        else:
            if k == 1:
                val = pi / Float64(d)
                txt = "pi/" + String(d)
            elif k == -1:
                val = (-pi) / Float64(d)
                txt = "-pi/" + String(d)
            else:
                val = Float64(k) * pi / Float64(d)
                txt = String(k) + "*pi/" + String(d)
        if val == v:
            return txt^
    if v == floor(v) and abs(v) < 1e15:
        return String(Int(v))
    return String(v)


def format_param(p: Param, names: List[String]) -> String:
    """OpenQASM text of an affine parameter, e.g. `2*theta + pi/2` (exact round trip)."""
    if not p.is_symbolic():
        return format_angle(p.offset)
    var nm = names[p.index]
    var s: String
    if p.coeff == 1.0:
        s = nm
    elif p.coeff == -1.0:
        s = "-" + nm
    else:
        s = format_angle(p.coeff) + "*" + nm
    if p.offset != 0.0:
        if p.offset > 0.0:
            s += " + " + format_angle(p.offset)
        else:
            s += " - " + format_angle(-p.offset)
    return s^


def _condition_text(circuit: Circuit, g: Instruction) -> String:
    """`if (...) ` of a conditioned instruction: `c == N` over the whole
    register when that is what it tests, an `&&` chain over its bits otherwise."""
    if g.cond_reg < 0:
        return "if (c[" + String(g.cond) + "] == " + String(g.cond_val) + ") "
    try:
        return "if (c == " + register_condition_value(circuit, g) + ") "
    except:
        pass
    var s = String("if (")
    for i in range(circuit.cond_table[g.cond_reg]):
        if i > 0:
            s += " && "
        s += "c[" + String(circuit.cond_table[g.cond_reg + 1 + 2 * i]) + "] == " + String(circuit.cond_table[g.cond_reg + 2 + 2 * i])
    return s + ") "


def _gate_definition(k: Int) -> String:
    """Exact OpenQASM 3 definitions for natives missing from stdgates.inc."""
    if k == Op.SXDG: return "gate sxdg a { inv @ sx a; }"
    if k == Op.ISWAP: return "gate iswap a, b { s a; s b; h a; cx a, b; cx b, a; h b; }"
    if k == Op.ISWAPDG: return "gate iswapdg a, b { h b; cx b, a; cx a, b; h a; sdg b; sdg a; }"
    if k == Op.ECR: return "gate ecr a, b { h b; cx a, b; rz(pi/4) b; cx a, b; h b; x a; h b; cx a, b; rz(-pi/4) b; cx a, b; h b; }"
    if k == Op.RXX: return "gate rxx(θ) a, b { h a; h b; cx a, b; rz(θ) b; cx a, b; h a; h b; }"
    if k == Op.RYY: return "gate ryy(θ) a, b { rx(pi/2) a; rx(pi/2) b; cx a, b; rz(θ) b; cx a, b; rx(-pi/2) a; rx(-pi/2) b; }"
    if k == Op.RZZ: return "gate rzz(θ) a, b { cx a, b; rz(θ) b; cx a, b; }"
    if k == Op.CCZ: return "gate ccz a, b, c { h c; ccx a, b, c; h c; }"
    return ""


def _gate_defs_text(circuit: Circuit) -> String:
    """Definitions of the natives outside stdgates.inc that `circuit` uses."""
    var out = String()
    var defs: List[Int] = [Op.SXDG, Op.ISWAP, Op.ISWAPDG, Op.ECR, Op.RXX, Op.RYY, Op.RZZ, Op.CCZ]
    var used = List[Bool](length=Op.NUM_OPS, fill=False)
    for g in circuit.ops:
        if g.kind >= 0 and g.kind < Op.NUM_OPS:
            used[g.kind] = True
    if used[Op.ISWAPDG]:
        used[Op.ISWAP] = True
    for k in defs:
        if used[k]:
            out += _gate_definition(k) + "\n"
    return out^


def to_qasm(circuit: Circuit) -> String:
    """Canonical OpenQASM 3 text for `circuit` (round-trips through parse_qasm)."""
    var out = String("OPENQASM 3.0;\ninclude \"stdgates.inc\";\n")
    # gate definitions for natives outside stdgates.inc
    out += _gate_defs_text(circuit)
    # parameters
    var reserved = _reserved_words()
    var names = List[String]()
    for nm in circuit.param_names:
        var s = _sanitize_ident(nm, reserved, names)
        names.append(s)
    for i in range(len(names)):
        out += "input float[64] " + names[i] + ";\n"
    for i in range(len(names)):
        if names[i] != circuit.param_names[i]:
            out += "#pragma qumo param " + names[i] + " \"" + circuit.param_names[i].replace("\"", "'") + "\"\n"
    if circuit.num_qubits > 0:
        out += "qubit[" + String(circuit.num_qubits) + "] q;\n"
    if circuit.num_clbits > 0:
        out += "bit[" + String(circuit.num_clbits) + "] c;\n"
    if circuit.global_phase != 0.0:
        out += "gphase(" + format_angle(circuit.global_phase) + ");\n"
    _ops_text(circuit, 0, len(circuit.ops), names, out)
    return out^


def _ops_text(circuit: Circuit, lo: Int, hi: Int, names: List[String], mut out: String):
    """One line of OpenQASM per instruction of [lo, hi), appended to `out`."""
    for i in range(lo, hi):
        ref g = circuit.ops[i]
        var k = g.kind
        var cond = String()
        if g.cond >= 0:
            cond = _condition_text(circuit, g)
        if k == Op.DETECTOR or k == Op.OBSERVABLE:
            out += "#pragma qumo "
            out += "detector" if k == Op.DETECTOR else "observable(" + String(g.q2) + ")"
            for j in range(g.q1):
                out += " " if j == 0 else ", "
                out += String(circuit.records[g.q0 + j])
            out += "\n"
            continue
        if k == Op.TICK:
            out += "#pragma qumo tick\n"
            continue
        if k == Op.BARRIER:
            out += "barrier q;\n" if circuit.num_qubits > 0 else "barrier;\n"
            continue
        if k == Op.MEASURE:
            out += cond + "c[" + String(g.clbit) + "] = measure q[" + String(g.q0) + "];\n"
            continue
        if k == Op.RESET:
            out += cond + "reset q[" + String(g.q0) + "];\n"
            continue
        if k == Op.PAULI_CHANNEL_2:
            var pc2 = "#pragma qumo " + cond + "pauli_channel_2("
            var po = Int(g.p0.offset)
            for j in range(15):
                pc2 += ", " if j > 0 else ""
                pc2 += format_angle(circuit.noise_table[po + j])
            out += pc2 + ") q[" + String(g.q0) + "], q[" + String(g.q1) + "];\n"
            continue
        if k == Op.CORRELATED_ERROR:
            var er = "#pragma qumo " + cond
            er += "else_correlated_error(" if g.q2 != 0 else "correlated_error("
            er += format_angle(g.p0.offset) + ")"
            for j in range(g.q1):
                var letter = circuit.pauli_table[g.q0 + 2 * j + 1]
                er += ", " if j > 0 else " "
                er += "x" if letter == 1 else ("y" if letter == 2 else "z")
                er += " q[" + String(circuit.pauli_table[g.q0 + 2 * j]) + "]"
            out += er + ";\n"
            continue
        if k == Op.KRAUS1 or k == Op.KRAUS2:
            ref ch = circuit.kraus[Int(g.p0.offset)]
            var kr = "#pragma qumo " + cond + op_name(k) + " \"" + ch.name + "\" ("
            var first = True
            for m in range(ch.count()):
                for j in range(len(ch.re[m])):
                    kr += "" if first else ", "
                    first = False
                    kr += format_angle(ch.re[m][j]) + ", " + format_angle(ch.im[m][j])
            kr += ") q[" + String(g.q0) + "]"
            if k == Op.KRAUS2:
                kr += ", q[" + String(g.q1) + "]"
            out += kr + ";\n"
            continue
        var line = String()
        if op_is_noise(k):
            line += "#pragma qumo "
        line += cond
        line += "U" if k == Op.U else op_name(k)
        var np = op_num_params(k)
        if np > 0:
            line += "("
            for j in range(np):
                if j > 0:
                    line += ", "
                line += format_param(g.param(j), names)
            line += ")"
        var a = op_arity(k)
        for j in range(a):
            line += " q[" if j == 0 else ", q["
            line += String(g.qubit(j)) + "]"
        line += ";\n"
        out += line


# ---------------------------------------------------------------------------
# Program emitter
# ---------------------------------------------------------------------------
comptime _PR_OR = 1
comptime _PR_AND = 2
comptime _PR_BOR = 3
comptime _PR_XOR = 4
comptime _PR_BAND = 5
comptime _PR_EQ = 6
comptime _PR_REL = 7
comptime _PR_SHIFT = 8
comptime _PR_ADD = 9
comptime _PR_MUL = 10
comptime _PR_UNARY = 11
comptime _PR_POW = 12
comptime _PR_ATOM = 13
comptime _MAX_EXPR_TEXT_DEPTH = 16384  # 60000 overflows an 8 MB stack


def _op_text(op: Int) -> String:
    if op == CN.ADD: return "+"
    if op == CN.SUB: return "-"
    if op == CN.MUL: return "*"
    if op == CN.DIV: return "/"
    if op == CN.MOD: return "%"
    if op == CN.POW: return "**"
    if op == CN.SHL: return "<<"
    if op == CN.SHR: return ">>"
    if op == CN.AND: return "&"
    if op == CN.OR: return "|"
    if op == CN.XOR: return "^"
    if op == CN.EQ: return "=="
    if op == CN.NE: return "!="
    if op == CN.LT: return "<"
    if op == CN.LE: return "<="
    if op == CN.GT: return ">"
    return ">="


def _op_prec(op: Int) -> Int:
    if op == CN.ADD or op == CN.SUB: return _PR_ADD
    if op == CN.MUL or op == CN.DIV or op == CN.MOD: return _PR_MUL
    if op == CN.POW: return _PR_POW
    if op == CN.SHL or op == CN.SHR: return _PR_SHIFT
    if op == CN.AND: return _PR_BAND
    if op == CN.OR: return _PR_BOR
    if op == CN.XOR: return _PR_XOR
    if op == CN.EQ or op == CN.NE: return _PR_EQ
    return _PR_REL


def _fn_text(f: Int) -> String:
    if f == CF.SIN: return "sin"
    if f == CF.COS: return "cos"
    if f == CF.TAN: return "tan"
    if f == CF.ASIN: return "arcsin"
    if f == CF.ACOS: return "arccos"
    if f == CF.ATAN: return "arctan"
    if f == CF.EXP: return "exp"
    if f == CF.LOG: return "ln"
    if f == CF.SQRT: return "sqrt"
    if f == CF.ABS: return "abs"
    if f == CF.FLOOR: return "floor"
    if f == CF.CEIL: return "ceil"
    if f == CF.MOD: return "mod"
    return "popcount"


def _float_text(v: Float64) -> String:
    var s = String(v)
    if isnan(v) or isinf(v):
        return s^
    if s.find(".") < 0 and s.find("e") < 0:
        s += ".0"
    return s^


def _check_finite(v: Float64, names: _PNames) raises:
    if (isnan(v) or isinf(v)) and not names.drawing:
        raise Error("QM-E552: the constant " + String(v) + " has no OpenQASM literal. Hint: compute it at run time, or fix the expression that overflowed.")


struct _PNames(Movable):
    """What the expression printer needs to name things."""

    var stores: List[String]  # by storage index
    var params: List[String]  # by parameter index
    var num_clbits: Int
    var drawing: Bool  # for a reader, who can take `inf`: OpenQASM text cannot

    def __init__(out self, n: Int):
        self.stores = List[String](length=n, fill=String(""))
        self.params = List[String]()
        self.num_clbits = 0
        self.drawing = False


def _const_text(n: CNode, plain: Bool, names: _PNames) raises -> String:
    """A constant; `plain` drops the type, for where the context restores it."""
    var t = n.ty
    if t.tag == CT.BOOL:
        return "true" if n.i != 0 else "false"
    if t.tag == CT.FLOAT:
        _check_finite(n.f, names)
        return _float_text(n.f)
    var body: String
    if t.tag == CT.ANGLE:
        body = _float_text(angle_radians(n.i, t.width))
    elif t.tag == CT.INT:
        body = String(n.i)
    else:
        body = String(UInt64(n.i))
    if plain or n.y == 1:
        return body^
    return type_name(t) + "(" + body + ")"


def _is_soft_cast(p: Program, n: Int) -> Bool:
    """A cast the parser puts back by itself: bit to bool, bit[n] to uint[n]."""
    ref g = p.nodes[n]
    if g.op != CN.CAST:
        return False
    var src = p.nodes[g.x].ty
    if g.ty.tag == CT.BOOL:
        return src.tag == CT.BOOL or (src.tag == CT.BIT and src.width == 1)
    return g.ty.tag == CT.UINT and (src.tag == CT.BIT or src.tag == CT.BOOL) and src.width == g.ty.width


def _child(p: Program, n: Int, names: _PNames, want: Int, plain: Bool, depth: Int) raises -> String:
    var t = _node_text(p, n, names, plain, depth)
    if t[1] < want:
        return "(" + t[0] + ")"
    return t[0]


def _bits_text(first: Int, count: Int, names: _PNames) -> String:
    if count == names.num_clbits and first == 0 and count > 1:
        return String("c")
    if count == 1:
        return "c[" + String(first) + "]"
    return "c[" + String(first) + ":" + String(first + count - 1) + "]"


def _node_text(p: Program, n: Int, names: _PNames, plain: Bool = False, depth: Int = 0) raises -> Tuple[String, Int]:
    """Text of node `n` and the precedence of its outermost operator. This recurses
    on the expression, which the stack limits."""
    if depth > _MAX_EXPR_TEXT_DEPTH:
        raise Error("QM-E552: an expression is nested more than " + String(_MAX_EXPR_TEXT_DEPTH) + " deep, too deep to write out. Hint: split it with a variable.")
    ref g = p.nodes[n]
    var op = g.op
    if op == CN.CONST:
        var text = _const_text(g, plain, names)
        # a negative constant is a sign and a number: it binds like a unary minus
        return (text, _PR_UNARY if text.startswith("-") else _PR_ATOM)
    if op == CN.VAR:
        return (names.stores[g.x], _PR_ATOM)
    if op == CN.INPUT:
        return (names.params[g.x], _PR_ATOM)
    if op == CN.BITS:
        return (_bits_text(g.x, g.y, names), _PR_ATOM)
    if op == CN.CAST:
        if _is_soft_cast(p, n):
            return _node_text(p, g.x, names, plain, depth + 1)
        var inner = _node_text(p, g.x, names, True, depth + 1)[0]
        # the table has no bit -> float, so that step stays written
        if g.ty.tag == CT.FLOAT and _is_soft_cast(p, g.x) and p.nodes[p.nodes[g.x].x].ty.tag == CT.BIT:
            inner = type_name(p.nodes[g.x].ty) + "(" + inner + ")"
        return (type_name(g.ty) + "(" + inner + ")", _PR_ATOM)
    if op == CN.NEG or op == CN.LNOT or op == CN.BNOT:
        var mark = String("-" if op == CN.NEG else ("!" if op == CN.LNOT else "~"))
        return (mark + _child(p, g.x, names, _PR_UNARY, False, depth + 1), _PR_UNARY)
    if op == CN.FN:
        var args = _node_text(p, g.x, names, True, depth + 1)[0]
        if g.y >= 0:
            args += ", " + _node_text(p, g.y, names, True, depth + 1)[0]
        return (_fn_text(g.i) + "(" + args + ")", _PR_ATOM)
    if op == CN.SC_END:
        var m = n - 1
        while not ((p.nodes[m].op == CN.SC_AND or p.nodes[m].op == CN.SC_OR) and p.nodes[m].y == n):
            m -= 1
        var is_and = p.nodes[m].op == CN.SC_AND
        var pr = _PR_AND if is_and else _PR_OR
        var left = _child(p, p.nodes[m].x, names, pr, False, depth + 1)
        var right = _child(p, g.x, names, pr + 1, False, depth + 1)
        return (left + (" && " if is_and else " || ") + right, pr)
    # binary: a constant takes the type of its sibling back when parsed again
    var pr = _op_prec(op)
    var l = p.nodes[g.x]
    var r = p.nodes[g.y]
    var same = l.ty == r.ty or op == CN.SHL or op == CN.SHR or op == CN.POW
    var lp = same and l.op == CN.CONST and r.op != CN.CONST
    var rp = same and r.op == CN.CONST and l.op != CN.CONST
    var lwant = pr + 1 if op == CN.POW else pr
    var rwant = pr if op == CN.POW else pr + 1
    return (_child(p, g.x, names, lwant, lp, depth + 1) + " " + _op_text(op) + " " + _child(p, g.y, names, rwant, rp, depth + 1), pr)


def _value_text(p: Program, e: Int, ty: CType, names: _PNames) raises -> String:
    """An expression where the context casts to `ty`: its outer cast is implied."""
    var root = p.expr_root[e]
    ref g = p.nodes[root]
    if g.op == CN.CAST and g.ty == ty and can_imply(p.nodes[g.x].ty, ty):
        return _node_text(p, g.x, names, True)[0]
    return _node_text(p, root, names, True)[0]


def can_imply(src: CType, dst: CType) -> Bool:
    return src.tag != CT.ANGLE or dst.tag == CT.ANGLE or dst.tag == CT.BOOL or dst.tag == CT.BIT


def _for_vars(p: Program) -> List[Bool]:
    var out = List[Bool](length=len(p.vars), fill=False)
    for s in p.tape:
        if s.kind == SK.FOR:
            out[s.a] = True
    return out^


def _pad(depth: Int) -> String:
    var s = String()
    for _ in range(depth):
        s += "  "
    return s^


def to_qasm(p: Program) raises -> String:
    """OpenQASM 3 text of a program (a fixed point of parse_qasm_program + to_qasm).
    Registers merge into `q` and `c` as for a Circuit, and declarations move to
    the top; names get a suffix where scopes shadowed each other. A constant that
    is NaN or infinite has no literal: QM-E552."""
    var flat = p.try_flatten()
    if flat:
        return to_qasm(flat.value())
    ref c = p.body
    var out = String("OPENQASM 3.0;\ninclude \"stdgates.inc\";\n")
    out += _gate_defs_text(c)
    var reserved = _reserved_words()
    var used = List[String]()
    var names = _PNames(p.num_store)
    names.num_clbits = c.num_clbits
    names.params = List[String](length=c.num_params(), fill=String(""))
    for i in p.user_params:
        var s = _sanitize_ident(c.param_names[i], reserved, used)
        used.append(s)
        names.params[i] = s
        out += "input float[64] " + s + ";\n"
    for i in p.user_params:
        if names.params[i] != c.param_names[i]:
            out += "#pragma qumo param " + names.params[i] + " \"" + c.param_names[i].replace("\"", "'") + "\"\n"
    if c.num_qubits > 0:
        out += "qubit[" + String(c.num_qubits) + "] q;\n"
    if c.num_clbits > 0:
        out += "bit[" + String(c.num_clbits) + "] c;\n"
    var is_loop = _for_vars(p)
    var vnames = List[String](length=len(p.vars), fill=String(""))
    for i in range(len(p.vars)):
        ref v = p.vars[i]
        if v.ty.tag == CT.BIT:
            continue
        var s = _sanitize_ident(v.name, reserved, used)
        used.append(s)
        vnames[i] = s
        names.stores[v.store] = s
        if v.slot >= 0:
            names.params[v.slot] = s
        if not is_loop[i]:
            out += ("output " if v.is_output else "") + type_name(v.ty) + " " + s + ";\n"
    if c.global_phase != 0.0:
        out += "gphase(" + format_angle(c.global_phase) + ");\n"
    var depth = 0
    var first_case = List[Bool]()  # one entry per open switch
    var unsigned_case = List[Bool]()
    for i in range(len(p.tape)):
        var s = p.tape[i]
        var k = s.kind
        if k == SK.OPS:
            var text = String()
            _ops_text(c, s.a, s.b, names.params, text)
            var pad = _pad(depth)
            for line in text.split("\n"):
                if line.byte_length() > 0:
                    out += pad + line + "\n"
        elif k == SK.ASSIGN:
            ref v = p.vars[s.a]
            var target = vnames[s.a] if v.ty.tag != CT.BIT else _bits_text(v.store, v.ty.width, names)
            out += _pad(depth) + target + " = " + _value_text(p, s.b, v.ty, names) + ";\n"
        elif k == SK.IF:
            out += _pad(depth) + "if (" + _value_text(p, s.a, bool_type(), names) + ") {\n"
            depth += 1
        elif k == SK.ELSE:
            out += _pad(depth - 1) + "} else {\n"
        elif k == SK.WHILE:
            out += _pad(depth) + "while (" + _value_text(p, s.a, bool_type(), names) + ") {\n"
            depth += 1
        elif k == SK.FOR:
            out += _pad(depth) + _for_head(p, s, vnames[s.a], names) + " {\n"
            depth += 1
        elif k == SK.SWITCH:
            out += _pad(depth) + "switch (" + _value_text(p, s.a, int_type(64), names) + ") {\n"
            depth += 2
            first_case.append(True)
            ref root = p.nodes[p.expr_root[s.a]]
            unsigned_case.append(root.op == CN.CAST and not p.nodes[root.x].ty.is_signed())
        elif k == SK.CASE:
            if first_case[len(first_case) - 1]:
                first_case[len(first_case) - 1] = False
            else:
                out += _pad(depth - 1) + "}\n"
            if s.a < 0:
                out += _pad(depth - 1) + "default {\n"
            else:
                var vals = String()
                var unsigned_target = unsigned_case[len(unsigned_case) - 1]
                for j in range(s.a, s.b):
                    var v = p.case_vals[j]
                    vals += (", " if j > s.a else "") + (String(UInt64(v)) if unsigned_target else String(v))
                out += _pad(depth - 1) + "case " + vals + " {\n"
        elif k == SK.BREAK:
            out += _pad(depth) + "break;\n"
        elif k == SK.CONTINUE:
            out += _pad(depth) + "continue;\n"
        else:  # END
            var op = p.tape[s.end].kind
            if op == SK.SWITCH:
                _ = unsigned_case.pop()
                if not first_case.pop():
                    out += _pad(depth - 1) + "}\n"
                depth -= 2
                out += _pad(depth) + "}\n"
            else:
                depth -= 1
                out += _pad(depth) + "}\n"
    return out^


def _for_head(p: Program, s: Stmt, name: String, names: _PNames) raises -> String:
    """`for int i in [0:2]`, without the brace."""
    var head = "for " + type_name(p.vars[s.a].ty) + " " + name + " in "
    var set_off = p.for_ranges[s.b + FORR.SET]
    if set_off >= 0:
        head += "{"
        for j in range(p.case_vals[set_off]):
            head += (", " if j > 0 else "") + String(p.case_vals[set_off + 1 + j])
        return head + "}"
    var i64 = int_type(64)
    var a = _bound_text(p, s.b, FORR.START, names)
    var st = _value_text(p, p.for_ranges[s.b + FORR.STEP], i64, names)
    var b = _bound_text(p, s.b, FORR.STOP, names)
    return head + "[" + a + ":" + ("" if st == "1" else st + ":") + b + "]"


def _bound_text(p: Program, off: Int, field: Int, names: _PNames) raises -> String:
    """A range bound. On an unsigned range a constant past 2^63 is held by its bits."""
    var e = p.for_ranges[off + field]
    ref root = p.nodes[p.expr_root[e]]
    if (p.for_ranges[off + FORR.UNSIGNED] & FORR.BOUNDS_UNSIGNED) != 0 and root.op == CN.CONST:
        return String(UInt64(root.i))
    return _value_text(p, e, int_type(64), names)


def draw_program(p: Program, width: Int = 80, style: String = "ascii") raises -> String:
    """Text drawing by blocks: each block as `Circuit.draw` draws it, under headers
    in OpenQASM and indented by region. A program without runtime regions is a
    circuit and draws as one. A gate with a runtime angle shows the variable."""
    var flat = p.try_flatten()
    if flat:
        return flat.value().draw(width, style)
    var names = _PNames(p.num_store)
    names.num_clbits = p.num_clbits()
    names.drawing = True
    var shell = p.body.copy()
    shell.ops.clear()
    names.params = shell.param_names.copy()
    var out = String()
    for ref v in p.vars:
        if v.ty.tag == CT.BIT:
            continue
        names.stores[v.store] = v.name
        if v.slot >= 0:
            names.params[v.slot] = v.name
    shell.param_names = names.params.copy()
    var is_loop = _for_vars(p)
    for i in range(len(p.vars)):
        ref v = p.vars[i]
        if v.ty.tag != CT.BIT and not is_loop[i]:
            out += ("output " if v.is_output else "") + type_name(v.ty) + " " + v.name + ";\n"
    var depth = 0
    for s in p.tape:
        var k = s.kind
        if k == SK.OPS:
            if s.b == s.a:
                continue
            var block = shell.copy()
            for j in range(s.a, s.b):
                block.ops.append(p.body.ops[j])
            var pad = _pad(depth)
            var fold = max(width - 2 * depth, 20) if width > 0 else 0
            for line in block.draw(fold, style).split("\n"):
                if line.byte_length() > 0:
                    out += pad + line + "\n"
        elif k == SK.ASSIGN:
            ref v = p.vars[s.a]
            var target = v.name if v.ty.tag != CT.BIT else _bits_text(v.store, v.ty.width, names)
            out += _pad(depth) + target + " = " + _value_text(p, s.b, v.ty, names) + ";\n"
        elif k == SK.IF:
            out += _pad(depth) + "if (" + _value_text(p, s.a, bool_type(), names) + ") {\n"
            depth += 1
        elif k == SK.ELSE:
            out += _pad(depth - 1) + "} else {\n"
        elif k == SK.WHILE:
            out += _pad(depth) + "while (" + _value_text(p, s.a, bool_type(), names) + ") {\n"
            depth += 1
        elif k == SK.FOR:
            out += _pad(depth) + _for_head(p, s, p.vars[s.a].name, names) + " {\n"
            depth += 1
        elif k == SK.SWITCH:
            out += _pad(depth) + "switch (" + _value_text(p, s.a, int_type(64), names) + ") {\n"
            depth += 2
        elif k == SK.CASE:
            if s.a < 0:
                out += _pad(depth - 1) + "default:\n"
            else:
                ref root = p.nodes[p.expr_root[p.tape[p.tape[s.end].end].a]]
                var unsigned_target = root.op == CN.CAST and not p.nodes[root.x].ty.is_signed()
                var vals = String()
                for j in range(s.a, s.b):
                    var v = p.case_vals[j]
                    vals += (", " if j > s.a else "") + (String(UInt64(v)) if unsigned_target else String(v))
                out += _pad(depth - 1) + "case " + vals + ":\n"
        elif k == SK.BREAK:
            out += _pad(depth) + "break;\n"
        elif k == SK.CONTINUE:
            out += _pad(depth) + "continue;\n"
        else:  # END
            depth -= 2 if p.tape[s.end].kind == SK.SWITCH else 1
            out += _pad(depth) + "}\n"
    return out^
