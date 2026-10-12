"""OpenQASM 3.0 (+ 2.0 compatibility) <-> QUMO `Circuit`.

Why OpenQASM is QUMO's LLM-facing circuit format: it is the lingua franca of
Qiskit, Braket, TKET, Cirq and the IBM/IonQ/Quantinuum hardware APIs, so
language models have seen orders of magnitude more QASM than any Python
builder API. QUMO therefore accepts the dialects models actually emit
(QASM 2 `qreg/creg/measure ->/if(c==1)`, QASM 3 `qubit[n]/bit[n]/c[i] = measure`,
`input float theta;`, `gate` definitions, `inv @`/`ctrl @`/`pow(k) @`
modifiers, constant `for` loops, unicode `π`/`θ`) and emits one canonical
QASM 3 form that other toolchains read back.

Implementation
--------------
* Lexer: one pass over the UTF-8 bytes, no regex, no per-character String
  allocation. Tokens are POD (kind, byte span, line, numeric value); columns
  are computed lazily (UTF-8 aware) only when a diagnostic is rendered.
* Parser: recursive descent over the token array. Parameter expressions are
  evaluated eagerly to QUMO's affine `Param` (`coeff * input + offset`); a
  product/division of inputs or a mix of two inputs is rejected with a precise
  error instead of silently building an expression tree.
* User `gate` bodies are inlined by re-walking their token range with a fresh
  closed scope (params -> Param, qubit args -> global indices). Modifiers are
  applied with the IR functors `Circuit.inverse()/controlled()/power()`.
* Definitions of gates QUMO already knows natively (e.g. the `gate rzz(...)`
  blocks the emitter writes for non-`stdgates.inc` gates) are treated as
  declarations: the native, exact opcode is used. This keeps emit -> parse
  round trips bit-exact while staying portable to other tools.
* QUMO-only IR (noise channels, QEC detectors/observables, ticks, original
  parameter names) is carried in `#pragma qumo ...` lines that this parser
  reads back and other tools ignore.

Every error is `QM-E5xx: <message> (line L, col C). Hint: <fix>` followed by
the offending source line and a caret, so both humans and agents can repair
the input (`qumo.ai.diagnostics.diagnostic_from_error` parses it back into a
structured `Diagnostic`).
"""

from std.collections import Dict
from std.math import pi, sin, cos, tan, sqrt, exp, log, asin, acos, atan, floor, ceil, isnan, isinf

from qumo.ir.gates import (
    Op, op_name, op_arity, op_num_params, op_is_unitary, op_is_noise,
)
from qumo.ir.circuit import Circuit, Instruction
from qumo.ir.param import Param


# ===========================================================================
# Tokens
# ===========================================================================
def decimal_to_bits(digits: String, width: Int) -> List[Int]:
    """Little-endian bits of an arbitrarily large decimal literal; empty if it
    does not fit in `width` bits."""
    var ds = List[Int]()
    var bs = digits.as_bytes()
    for i in range(digits.byte_length()):
        var c = Int(bs[i])
        if c >= ord("0") and c <= ord("9"):
            ds.append(c - ord("0"))
    var bits = List[Int]()
    while True:
        var nonzero = False
        for d in ds:
            if d != 0:
                nonzero = True
                break
        if not nonzero:
            break
        var rem = 0
        for i in range(len(ds)):
            var cur = rem * 10 + ds[i]
            ds[i] = cur // 2
            rem = cur % 2
        bits.append(rem)
        if len(bits) > width:
            return List[Int]()
    while len(bits) < width:
        bits.append(0)
    return bits^


def bits_to_decimal(bits: List[Int]) -> String:
    """Decimal string of a little-endian bit list of any length."""
    var ds: List[Int] = [0]  # little-endian decimal digits
    var i = len(bits) - 1
    while i >= 0:
        var carry = bits[i]
        for j in range(len(ds)):
            var v = ds[j] * 2 + carry
            ds[j] = v % 10
            carry = v // 10
        if carry > 0:
            ds.append(carry)
        i -= 1
    var out = String()
    var k = len(ds) - 1
    while k >= 0:
        out += String(ds[k])
        k -= 1
    return out


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


struct TK:
    comptime EOF = 0
    comptime IDENT = 1
    comptime INT = 2
    comptime FLOAT = 3
    comptime STRING = 4
    comptime PUNCT = 5
    comptime PRAGMA = 6  # `#pragma qumo` / `pragma qumo` marker
    comptime HWQUBIT = 7  # `$3`


struct P:
    comptime SEMI = 1
    comptime COMMA = 2
    comptime LPAREN = 3
    comptime RPAREN = 4
    comptime LBRACK = 5
    comptime RBRACK = 6
    comptime LBRACE = 7
    comptime RBRACE = 8
    comptime ARROW = 9
    comptime EQEQ = 10
    comptime EQ = 11
    comptime PLUS = 12
    comptime MINUS = 13
    comptime STAR = 14
    comptime SLASH = 15
    comptime POW = 16
    comptime AT = 17
    comptime COLON = 18
    comptime BANG = 19
    comptime NEQ = 20
    comptime LT = 21
    comptime GT = 22
    comptime LE = 23
    comptime GE = 24
    comptime TILDE = 25
    comptime AMP = 26
    comptime PIPE = 27
    comptime CARET = 28
    comptime PERCENT = 29
    comptime DOT = 30
    comptime ANDAND = 31
    comptime OROR = 32
    comptime SHL = 33
    comptime SHR = 34
    comptime OPASSIGN = 35  # += -= *= /= ...


@fieldwise_init
struct Token(ImplicitlyCopyable):
    var kind: Int
    var sub: Int  # punctuation code (P.*)
    var start: Int  # byte offsets [start, end)
    var end: Int
    var line: Int
    var num: Float64  # INT / FLOAT / HWQUBIT value


# ===========================================================================
# Diagnostics helpers (no dependency on qumo.ai)
# ===========================================================================
def _col_of(bs: Span[Byte, _], offset: Int) -> Int:
    """1-based column of byte `offset`, counting UTF-8 code points."""
    var i = min(offset, len(bs))
    var ls = i
    while ls > 0 and Int(bs[ls - 1]) != 0x0A:
        ls -= 1
    var col = 1
    for j in range(ls, i):
        if (Int(bs[j]) & 0xC0) != 0x80:
            col += 1
    return col


def format_qasm_error(src: String, code: String, msg: String, offset: Int,
                      length: Int, line: Int, hint: String) -> String:
    """Render `CODE: msg (line L, col C). Hint: ...` + source line + caret."""
    var bs = src.as_bytes()
    var n = len(bs)
    var off = max(0, min(offset, n))
    var col = _col_of(bs, off)
    var s = code + ": " + msg + " (line " + String(line) + ", col " + String(col) + ")."
    if hint.byte_length() > 0:
        s += " Hint: " + hint
    # source line snippet
    var ls = off
    while ls > 0 and Int(bs[ls - 1]) != 0x0A:
        ls -= 1
    var le = off
    while le < n and Int(bs[le]) != 0x0A and Int(bs[le]) != 0x0D:
        le += 1
    if le > ls or off < n:
        var ln = String(line)
        s += "\n " + ln + " | " + String(unsafe_from_utf8=bs[ls:le])
        var pad = String()
        for _ in range(ln.byte_length()):
            pad += " "
        s += "\n " + pad + " | "
        for j in range(ls, off):
            var c = Int(bs[j])
            if (c & 0xC0) == 0x80:
                continue
            s += "\t" if c == 0x09 else " "
        var width = 0
        var j = off
        while j < min(off + max(length, 1), le):
            if (Int(bs[j]) & 0xC0) != 0x80:
                width += 1
            j += 1
        s += "^"
        for _ in range(1, max(width, 1)):
            s += "~"
    return s^


def _edit_distance(a: String, b: String) -> Int:
    var ab = a.as_bytes()
    var bb = b.as_bytes()
    var m = len(ab)
    var n = len(bb)
    var prev = List[Int](length=n + 1, fill=0)
    var cur = List[Int](length=n + 1, fill=0)
    for j in range(n + 1):
        prev[j] = j
    for i in range(1, m + 1):
        cur[0] = i
        for j in range(1, n + 1):
            var cost = 0 if ab[i - 1] == bb[j - 1] else 1
            cur[j] = min(min(prev[j] + 1, cur[j - 1] + 1), prev[j - 1] + cost)
        for j in range(n + 1):
            prev[j] = cur[j]
    return prev[n]


def suggest_names(name: String, candidates: List[String], max_out: Int = 3) -> List[String]:
    """Closest candidates by edit distance (<= 2, or <= len/3 for long names)."""
    var limit = max(2, name.byte_length() // 3)
    var best = limit + 1
    var dists = List[Int]()
    for c in candidates:
        var d = _edit_distance(name, c)
        if name.lower() == c.lower():
            d = 0
        dists.append(d)
        if d < best:
            best = d
    var out = List[String]()
    if best > limit:
        return out^
    for i in range(len(candidates)):
        if dists[i] == best:
            out.append(candidates[i])
    sort(out)
    while len(out) > max_out:
        _ = out.pop()
    return out^


def _quote_list(names: List[String]) -> String:
    var s = String()
    for i in range(len(names)):
        if i > 0:
            s += " or " if i == len(names) - 1 else ", "
        s += "'" + names[i] + "'"
    return s^


# ===========================================================================
# Lexer
# ===========================================================================
def _is_ident_start(c: Int) -> Bool:
    return (c >= 0x61 and c <= 0x7A) or (c >= 0x41 and c <= 0x5A) or c == 0x5F or c >= 0x80


def _is_ident_char(c: Int) -> Bool:
    return _is_ident_start(c) or (c >= 0x30 and c <= 0x39)


def _is_digit(c: Int) -> Bool:
    return c >= 0x30 and c <= 0x39


def _span_eq(bs: Span[Byte, _], start: Int, end: Int, w: StringSlice) -> Bool:
    var wb = w.as_bytes()
    if end - start != len(wb):
        return False
    for i in range(len(wb)):
        if bs[start + i] != wb[i]:
            return False
    return True


def _push(mut toks: List[Token], kind: Int, sub: Int, start: Int, end: Int, line: Int, num: Float64 = 0.0):
    toks.append(Token(kind, sub, start, end, line, num))


def _close_pragma(mut toks: List[Token], at: Int, line: Int):
    var n = len(toks)
    if n > 0 and toks[n - 1].kind == TK.PUNCT and toks[n - 1].sub == P.SEMI:
        return
    if n > 0 and toks[n - 1].kind == TK.PRAGMA:
        _push(toks, TK.PUNCT, P.SEMI, at, at, line)
        return
    _push(toks, TK.PUNCT, P.SEMI, at, at, line)


def _lex_error(src: String, code: String, msg: String, at: Int, line: Int, hint: String) -> Error:
    return Error(format_qasm_error(src, code, msg, at, 1, line, hint))


def tokenize_qasm(src: String) raises -> List[Token]:
    """Byte-level lexer. Comments, `#pragma` (non-qumo) and `@annotation`
    lines are dropped; `#pragma qumo` becomes a PRAGMA token followed by the
    line's tokens and a synthetic `;`."""
    var bs = src.as_bytes()
    var n = len(bs)
    var toks = List[Token](capacity=n // 3 + 8)
    var i = 0
    var line = 1
    var line_has_tok = False
    var in_pragma = False
    while i < n:
        var c = Int(bs[i])
        if c == 0x0A:
            if in_pragma:
                _close_pragma(toks, i, line)
                in_pragma = False
            line += 1
            i += 1
            line_has_tok = False
            continue
        if c == 0x20 or c == 0x09 or c == 0x0D or c == 0x0C:
            i += 1
            continue
        # comments
        if c == 0x2F and i + 1 < n and Int(bs[i + 1]) == 0x2F:
            while i < n and Int(bs[i]) != 0x0A:
                i += 1
            continue
        if c == 0x2F and i + 1 < n and Int(bs[i + 1]) == 0x2A:
            var start = i
            var sline = line
            i += 2
            var closed = False
            while i + 1 < n:
                if Int(bs[i]) == 0x2A and Int(bs[i + 1]) == 0x2F:
                    closed = True
                    i += 2
                    break
                if Int(bs[i]) == 0x0A:
                    line += 1
                i += 1
            if not closed:
                raise _lex_error(src, "QM-E502", "unterminated block comment", start, sline, "close the comment with */")
            continue
        # directives: #pragma / pragma
        if c == 0x23:
            if i + 7 <= n and _span_eq(bs, i + 1, i + 7, "pragma"):
                var j = i + 7
                while j < n and (Int(bs[j]) == 0x20 or Int(bs[j]) == 0x09):
                    j += 1
                if j + 4 <= n and _span_eq(bs, j, j + 4, "qumo") and (j + 4 == n or not _is_ident_char(Int(bs[j + 4]))):
                    _push(toks, TK.PRAGMA, 0, i, j + 4, line)
                    in_pragma = True
                    line_has_tok = True
                    i = j + 4
                    continue
                while i < n and Int(bs[i]) != 0x0A:
                    i += 1
                continue
            raise _lex_error(src, "QM-E501", "unexpected character '#'", i, line, "only '#pragma' directives are supported; use // for comments")
        if c == 0x40 and not line_has_tok:  # @annotation line
            while i < n and Int(bs[i]) != 0x0A:
                i += 1
            continue
        line_has_tok = True
        if _is_ident_start(c):
            var j = i + 1
            while j < n and _is_ident_char(Int(bs[j])):
                j += 1
            if _span_eq(bs, i, j, "pragma") and not in_pragma:
                var k = j
                while k < n and (Int(bs[k]) == 0x20 or Int(bs[k]) == 0x09):
                    k += 1
                if k + 4 <= n and _span_eq(bs, k, k + 4, "qumo") and (k + 4 == n or not _is_ident_char(Int(bs[k + 4]))):
                    _push(toks, TK.PRAGMA, 0, i, k + 4, line)
                    in_pragma = True
                    i = k + 4
                    continue
                while i < n and Int(bs[i]) != 0x0A:
                    i += 1
                continue
            _push(toks, TK.IDENT, 0, i, j, line)
            i = j
            continue
        if _is_digit(c) or (c == 0x2E and i + 1 < n and _is_digit(Int(bs[i + 1]))):
            var j = i
            var is_float = False
            var iv = 0
            while j < n and (_is_digit(Int(bs[j])) or Int(bs[j]) == 0x5F):
                if Int(bs[j]) != 0x5F:
                    iv = iv * 10 + (Int(bs[j]) - 0x30)
                j += 1
            if j < n and Int(bs[j]) == 0x2E:
                is_float = True
                j += 1
                while j < n and _is_digit(Int(bs[j])):
                    j += 1
            if j < n and (Int(bs[j]) == 0x65 or Int(bs[j]) == 0x45):
                var k = j + 1
                if k < n and (Int(bs[k]) == 0x2B or Int(bs[k]) == 0x2D):
                    k += 1
                if k < n and _is_digit(Int(bs[k])):
                    is_float = True
                    j = k
                    while j < n and _is_digit(Int(bs[j])):
                        j += 1
            if is_float:
                var v: Float64
                try:
                    v = Float64(StringSlice(unsafe_from_utf8=bs[i:j]))
                except:
                    raise _lex_error(src, "QM-E501", "malformed number", i, line, "write numbers like 0.5, 1e-3 or 3")
                _push(toks, TK.FLOAT, 0, i, j, line, v)
            else:
                _push(toks, TK.INT, 0, i, j, line, Float64(iv))
            i = j
            continue
        if c == 0x22 or c == 0x27:
            var j = i + 1
            while j < n and Int(bs[j]) != c and Int(bs[j]) != 0x0A:
                j += 1
            if j >= n or Int(bs[j]) != c:
                raise _lex_error(src, "QM-E502", "unterminated string literal", i, line, "close the string on the same line")
            _push(toks, TK.STRING, 0, i + 1, j, line)
            i = j + 1
            continue
        if c == 0x24:  # $n hardware qubit
            var j = i + 1
            var iv = 0
            while j < n and _is_digit(Int(bs[j])):
                iv = iv * 10 + (Int(bs[j]) - 0x30)
                j += 1
            if j == i + 1:
                raise _lex_error(src, "QM-E501", "'$' must be followed by a physical qubit number", i, line, "write $0, $1, ...")
            _push(toks, TK.HWQUBIT, 0, i, j, line, Float64(iv))
            i = j
            continue
        # punctuation
        var d = Int(bs[i + 1]) if i + 1 < n else 0
        var sub: Int
        var w = 1
        if c == 0x3B: sub = P.SEMI
        elif c == 0x2C: sub = P.COMMA
        elif c == 0x28: sub = P.LPAREN
        elif c == 0x29: sub = P.RPAREN
        elif c == 0x5B: sub = P.LBRACK
        elif c == 0x5D: sub = P.RBRACK
        elif c == 0x7B: sub = P.LBRACE
        elif c == 0x7D: sub = P.RBRACE
        elif c == 0x40: sub = P.AT
        elif c == 0x3A: sub = P.COLON
        elif c == 0x7E: sub = P.TILDE
        elif c == 0x2E: sub = P.DOT
        elif c == 0x2D:
            if d == 0x3E:
                sub = P.ARROW
                w = 2
            elif d == 0x3D:
                sub = P.OPASSIGN
                w = 2
            else:
                sub = P.MINUS
        elif c == 0x3D:
            if d == 0x3D:
                sub = P.EQEQ
                w = 2
            else:
                sub = P.EQ
        elif c == 0x21:
            if d == 0x3D:
                sub = P.NEQ
                w = 2
            else:
                sub = P.BANG
        elif c == 0x2B:
            if d == 0x3D:
                sub = P.OPASSIGN
                w = 2
            else:
                sub = P.PLUS
        elif c == 0x2A:
            if d == 0x2A:
                sub = P.POW
                w = 2
            elif d == 0x3D:
                sub = P.OPASSIGN
                w = 2
            else:
                sub = P.STAR
        elif c == 0x2F:
            if d == 0x3D:
                sub = P.OPASSIGN
                w = 2
            else:
                sub = P.SLASH
        elif c == 0x25: sub = P.PERCENT
        elif c == 0x3C:
            if d == 0x3D:
                sub = P.LE
                w = 2
            elif d == 0x3C:
                sub = P.SHL
                w = 2
            else:
                sub = P.LT
        elif c == 0x3E:
            if d == 0x3D:
                sub = P.GE
                w = 2
            elif d == 0x3E:
                sub = P.SHR
                w = 2
            else:
                sub = P.GT
        elif c == 0x26:
            if d == 0x26:
                sub = P.ANDAND
                w = 2
            else:
                sub = P.AMP
        elif c == 0x7C:
            if d == 0x7C:
                sub = P.OROR
                w = 2
            else:
                sub = P.PIPE
        elif c == 0x5E: sub = P.CARET
        else:
            raise _lex_error(src, "QM-E501", "unexpected character '" + String(unsafe_from_utf8=bs[i:i + 1]) + "'", i, line, "remove it or check for a missing quote/comment marker")
        _push(toks, TK.PUNCT, sub, i, i + w, line)
        i += w
    if in_pragma:
        _close_pragma(toks, n, line)
    _push(toks, TK.EOF, 0, n, n, line)
    return toks^


# ===========================================================================
# Gate table
# ===========================================================================
struct G:
    """Special (non-opcode) gate codes; natives use their opcode (>= 0)."""

    comptime U2 = -2
    comptime U3 = -3
    comptime CU = -4
    comptime CU3 = -5
    comptime CSX = -6
    comptime GPHASE = -7


def _special_nparams(code: Int) -> Int:
    if code == G.U2: return 2
    if code == G.U3: return 3
    if code == G.CU: return 4
    if code == G.CU3: return 3
    if code == G.GPHASE: return 1
    return 0


def _special_arity(code: Int) -> Int:
    if code == G.U2 or code == G.U3: return 1
    if code == G.CU or code == G.CU3 or code == G.CSX: return 2
    return 0


def builtin_gate_table() -> Dict[String, Int]:
    """Every gate name the parser understands (natives + aliases + specials)."""
    var t = Dict[String, Int]()
    for k in range(Op.CCZ + 1):
        t[op_name(k)] = k
    for k in range(Op.DEPOLARIZE1, Op.PAULI_CHANNEL_1 + 1):
        t[op_name(k)] = k
    t["i"] = Op.I
    t["U"] = Op.U
    t["CX"] = Op.CX
    t["cnot"] = Op.CX
    t["toffoli"] = Op.CCX
    t["fredkin"] = Op.CSWAP
    t["phase"] = Op.P
    t["u1"] = Op.P
    t["cphase"] = Op.CP
    t["cu1"] = Op.CP
    t["u2"] = G.U2
    t["u3"] = G.U3
    t["cu"] = G.CU
    t["cu3"] = G.CU3
    t["csx"] = G.CSX
    t["gphase"] = G.GPHASE
    return t^


# ===========================================================================
# Parser
# ===========================================================================
@fieldwise_init
struct Register(Copyable):
    var name: String
    var offset: Int
    var size: Int
    var is_qubit: Bool


struct GateDef(Copyable):
    var name: String
    var pnames: List[String]
    var qnames: List[String]
    var body_start: Int  # token index of first body token
    var body_end: Int  # token index of the closing '}'

    def __init__(out self, name: String, var pnames: List[String], var qnames: List[String], body_start: Int, body_end: Int):
        self.name = name
        self.pnames = pnames^
        self.qnames = qnames^
        self.body_start = body_start
        self.body_end = body_end


struct Scope(Copyable):
    var names: List[String]  # numeric bindings: gate params, loop variables
    var values: List[Param]
    var qnames: List[String]  # qubit bindings: gate arguments
    var qubits: List[Int]
    var closed: Bool  # gate scope: outer locals invisible

    def __init__(out self, closed: Bool):
        self.names = List[String]()
        self.values = List[Param]()
        self.qnames = List[String]()
        self.qubits = List[Int]()
        self.closed = closed


comptime _MAX_GATE_DEPTH = 64
comptime _MAX_UNROLL = 1000000


struct QasmParser(Movable):
    var src: String
    var toks: List[Token]
    var pos: Int
    var version: Int
    var circs: List[Circuit]  # [0] = result; top = current emission target
    var regs: List[Register]
    var reg_index: Dict[String, Int]
    var gate_table: Dict[String, Int]
    var gates: List[GateDef]
    var user_gates: Dict[String, Int]
    var consts: Dict[String, Param]
    var inputs: Dict[String, Int]
    var scopes: List[Scope]
    var cond_bit: Int
    var cond_val: Int
    var cond_reg: Int  # offset into circs[0].cond_table for register conditions (-1 none)
    var cond_reg_bits: List[Int]
    var cond_reg_vals: List[Int]
    var call_names: List[String]
    var call_lines: List[Int]
    var warnings: List[String]
    var depth: Int
    var unrolled: Int
    var stmt_count: Int

    def __init__(out self, src: String, name: String) raises:
        self.src = src
        self.toks = tokenize_qasm(src)
        self.pos = 0
        self.version = 3
        self.circs = List[Circuit]()
        self.circs.append(Circuit(0, 0, name))
        self.regs = List[Register]()
        self.reg_index = Dict[String, Int]()
        self.gate_table = builtin_gate_table()
        self.gates = List[GateDef]()
        self.user_gates = Dict[String, Int]()
        self.consts = Dict[String, Param]()
        self.inputs = Dict[String, Int]()
        self.scopes = List[Scope]()
        self.cond_bit = -1
        self.cond_val = 1
        self.cond_reg = -1
        self.cond_reg_bits = List[Int]()
        self.cond_reg_vals = List[Int]()
        self.call_names = List[String]()
        self.call_lines = List[Int]()
        self.warnings = List[String]()
        self.depth = 0
        self.unrolled = 0
        self.stmt_count = 0

    # ------------------------------------------------------------ tokens
    def peek(self, k: Int = 0) -> Token:
        var i = min(self.pos + k, len(self.toks) - 1)
        return self.toks[i]

    def next(mut self) -> Token:
        var t = self.toks[self.pos]
        if self.pos < len(self.toks) - 1:
            self.pos += 1
        return t

    def text(self, t: Token) -> String:
        return String(unsafe_from_utf8=self.src.as_bytes()[t.start:t.end])

    def is_word(self, t: Token, w: StringSlice) -> Bool:
        return t.kind == TK.IDENT and _span_eq(self.src.as_bytes(), t.start, t.end, w)

    def is_punct(self, t: Token, sub: Int) -> Bool:
        return t.kind == TK.PUNCT and t.sub == sub

    def describe(self, t: Token) -> String:
        if t.kind == TK.EOF:
            return "end of input"
        if t.kind == TK.PUNCT and t.start == t.end:
            return "end of pragma line"
        return "'" + self.text(t) + "'"

    # ------------------------------------------------------------ errors
    def err(self, code: String, msg: String, t: Token, hint: String = "") -> Error:
        var h = hint
        if h.byte_length() == 0:
            h = "check the syntax near this token (see 'qumo describe capabilities' for the supported OpenQASM subset)"
        var m = msg
        var nc = len(self.call_names)
        if nc > 0:
            m += " [inside gate '" + self.call_names[nc - 1] + "' called on line " + String(self.call_lines[nc - 1]) + "]"
        return Error(format_qasm_error(self.src, code, m, t.start, t.end - t.start, t.line, h))

    def warn(mut self, code: String, msg: String, t: Token, hint: String = ""):
        self.warnings.append(format_qasm_error(self.src, code, msg, t.start, t.end - t.start, t.line, hint))

    def relocate(self, e: String, t: Token) -> Error:
        """Attach a source location to an error raised by core IR code."""
        if e.find(" (line ") >= 0:
            return Error(e)
        var code = String("QM-E500")
        var body = e
        if e.startswith("QM-") and e.byte_length() >= 8 and e.find(":") == 7:
            code = String(e[byte=0:7])
            body = String(e[byte=9:])
        var hint = String()
        var h = body.find(" Hint: ")
        if h >= 0:
            hint = String(body[byte=h + 7:])
            var head = String(body[byte=0:h])
            body = head^
        var bb = body.as_bytes()
        var bl = len(bb)
        while bl > 0 and (Int(bb[bl - 1]) == 0x2E or Int(bb[bl - 1]) == 0x20):
            bl -= 1
        return self.err(code, String(body[byte=0:bl]), t, hint)

    def expect_punct(mut self, sub: Int, what: String, hint: String = "") raises -> Token:
        var t = self.peek()
        if not self.is_punct(t, sub):
            var h = hint
            var at = t
            if sub == P.SEMI:
                if h.byte_length() == 0:
                    h = "every statement ends with ';'"
                # rustc-style: point just after the previous token
                if self.pos > 0:
                    var prev = self.toks[self.pos - 1]
                    at = Token(TK.PUNCT, 0, prev.end, prev.end, prev.line, 0.0)
            elif h.byte_length() == 0:
                if sub == P.RPAREN:
                    h = "a ')' is missing, or an operator (+ - * /) or ',' is missing between terms"
                elif sub == P.RBRACK:
                    h = "a ']' is missing; indices look like q[0], q[0:2] or q[{0, 2}]"
                elif sub == P.RBRACE:
                    h = "a '}' is missing"
                else:
                    h = "check the syntax near this token (see 'qumo describe capabilities' for the supported OpenQASM subset)"
            raise self.err("QM-E500", "expected " + what + ", found " + self.describe(t), at, h)
        return self.next()

    def expect_ident(mut self, what: String) raises -> Token:
        var t = self.peek()
        if t.kind != TK.IDENT:
            raise self.err("QM-E500", "expected " + what + ", found " + self.describe(t), t, "an identifier (letters, digits, '_') is required here")
        return self.next()

    # ------------------------------------------------------------ emission
    def top(self) -> Int:
        return len(self.circs) - 1

    def emit(mut self, var inst: Instruction):
        if self.cond_bit >= 0 and len(self.circs) == 1:
            inst.cond = self.cond_bit
            inst.cond_val = self.cond_val
            inst.cond_reg = self.cond_reg
        self.circs[len(self.circs) - 1].ops.append(inst)

    def add_phase(mut self, v: Float64):
        if self.cond_bit >= 0 and len(self.circs) == 1:
            return  # a classically-conditioned global phase is unobservable
        self.circs[len(self.circs) - 1].global_phase += v

    def push_target(mut self):
        var t = self.top()
        var c = Circuit(self.circs[t].num_qubits, self.circs[t].num_clbits, "sub")
        c.param_names = self.circs[t].param_names.copy()
        self.circs.append(c^)

    def pop_target(mut self) -> Circuit:
        return self.circs.pop()

    def compose_sub(mut self, sub: Circuit):
        var t = self.top()
        var start = len(self.circs[t].ops)
        self.circs[t].compose(sub)
        if self.cond_bit >= 0 and t == 0:
            for i in range(start, len(self.circs[t].ops)):
                self.circs[t].ops[i].cond = self.cond_bit
                self.circs[t].ops[i].cond_val = self.cond_val
                self.circs[t].ops[i].cond_reg = self.cond_reg
            self.circs[t].global_phase -= sub.global_phase

    # ------------------------------------------------------------ names
    def known_value_names(self) -> List[String]:
        var out = List[String]()
        for s in self.scopes:
            for nm in s.names:
                out.append(nm)
        for k in self.consts.keys():
            out.append(k)
        for k in self.inputs.keys():
            out.append(k)
        out.append("pi")
        out.append("tau")
        out.append("euler")
        return out^

    def known_gate_names(self) -> List[String]:
        var out = List[String]()
        for k in self.gate_table.keys():
            out.append(k)
        for k in self.user_gates.keys():
            out.append(k)
        return out^

    def check_new_name(self, t: Token) raises:
        var nm = self.text(t)
        if nm in self.reg_index or nm in self.inputs or nm in self.consts:
            raise self.err("QM-E522", "name '" + nm + "' is already declared", t, "pick a unique name for each register, input and constant")

    # ------------------------------------------------------------ expressions
    def lookup_value(self, t: Token) raises -> Param:
        var nm = self.text(t)
        if nm == "pi" or nm == "π":
            return Param(pi)
        if nm == "tau" or nm == "τ":
            return Param(2.0 * pi)
        if nm == "euler" or nm == "ℇ":
            return Param(2.718281828459045)
        if nm == "true":
            return Param(1.0)
        if nm == "false":
            return Param(0.0)
        var s = len(self.scopes) - 1
        while s >= 0:
            for i in range(len(self.scopes[s].names)):
                if self.scopes[s].names[i] == nm:
                    return self.scopes[s].values[i]
            if self.scopes[s].closed:
                break
            s -= 1
        var c = self.consts.get(nm)
        if c:
            return c.value()
        var ip = self.inputs.get(nm)
        if ip:
            return Param.symbol(ip.value())
        if nm in self.reg_index:
            raise self.err("QM-E530", "register '" + nm + "' cannot be used in a numeric expression", t, "conditions use 'if (c[0] == 1)'; angles must be numbers, constants or 'input float' parameters")
        var sug = suggest_names(nm, self.known_value_names())
        var hint = String("declare it, e.g. 'input float " + nm + ";' for a symbolic parameter or 'const float " + nm + " = 0.5;'")
        if len(sug) > 0:
            hint = "did you mean " + _quote_list(sug) + "? Otherwise " + hint
        raise self.err("QM-E520", "undeclared identifier '" + nm + "'", t, hint)

    def p_add(self, a: Param, b: Param, t: Token) raises -> Param:
        if not a.is_symbolic():
            return Param(a.offset + b.offset, b.index, b.coeff)
        if not b.is_symbolic():
            return Param(a.offset + b.offset, a.index, a.coeff)
        if a.index == b.index:
            return Param(a.offset + b.offset, a.index, a.coeff + b.coeff)
        raise self.err("QM-E532", "expression combines two different input parameters", t, "QUMO parameters are affine in ONE input (e.g. 2*theta + pi/2); introduce a separate gate per parameter")

    def p_mul(self, a: Param, b: Param, t: Token) raises -> Param:
        if not a.is_symbolic() and not b.is_symbolic():
            return Param(a.offset * b.offset)
        if a.is_symbolic() and b.is_symbolic():
            raise self.err("QM-E531", "product of input parameters is not affine", t, "QUMO parameters must be coeff*input + offset; bind values first or restructure the circuit")
        if a.is_symbolic():
            return Param(a.offset * b.offset, a.index, a.coeff * b.offset)
        return Param(a.offset * b.offset, b.index, b.coeff * a.offset)

    def p_div(self, a: Param, b: Param, t: Token) raises -> Param:
        if b.is_symbolic():
            raise self.err("QM-E531", "division by an input parameter is not affine", t, "only constants may appear in denominators")
        if b.offset == 0.0:
            raise self.err("QM-E533", "division by zero", t, "check the denominator")
        if not a.is_symbolic():
            return Param(a.offset / b.offset)
        return Param(a.offset / b.offset, a.index, a.coeff / b.offset)

    def need_const(self, p: Param, t: Token, what: String) raises -> Float64:
        if p.is_symbolic():
            raise self.err("QM-E530", what + " must be a constant, but depends on an input parameter", t, "bind the parameter first or use a numeric value")
        return p.offset

    def parse_expr(mut self) raises -> Param:
        var a = self.parse_term()
        while True:
            var t = self.peek()
            if self.is_punct(t, P.PLUS):
                _ = self.next()
                var b = self.parse_term()
                a = self.p_add(a, b, t)
            elif self.is_punct(t, P.MINUS):
                _ = self.next()
                var b = self.parse_term()
                a = self.p_add(a, -b, t)
            else:
                return a

    def parse_term(mut self) raises -> Param:
        var a = self.parse_unary()
        while True:
            var t = self.peek()
            if self.is_punct(t, P.STAR):
                _ = self.next()
                var b = self.parse_unary()
                a = self.p_mul(a, b, t)
            elif self.is_punct(t, P.SLASH):
                _ = self.next()
                var b = self.parse_unary()
                a = self.p_div(a, b, t)
            elif self.is_punct(t, P.PERCENT):
                _ = self.next()
                var b = self.parse_unary()
                var x = self.need_const(a, t, "operand of %")
                var y = self.need_const(b, t, "operand of %")
                if y == 0.0:
                    raise self.err("QM-E533", "modulo by zero", t, "check the right operand")
                a = Param(x - y * floor(x / y))
            else:
                return a

    def parse_unary(mut self) raises -> Param:
        var t = self.peek()
        if self.is_punct(t, P.MINUS):
            _ = self.next()
            return -self.parse_unary()
        if self.is_punct(t, P.PLUS):
            _ = self.next()
            return self.parse_unary()
        return self.parse_power()

    def parse_power(mut self) raises -> Param:
        var a = self.parse_primary()
        var t = self.peek()
        if self.is_punct(t, P.POW):
            _ = self.next()
            var b = self.parse_unary()
            var x = self.need_const(a, t, "base of **")
            var y = self.need_const(b, t, "exponent of **")
            return Param(x ** y)
        return a

    def parse_primary(mut self) raises -> Param:
        var t = self.peek()
        if t.kind == TK.INT or t.kind == TK.FLOAT:
            _ = self.next()
            return Param(t.num)
        if self.is_punct(t, P.LPAREN):
            _ = self.next()
            var v = self.parse_expr()
            _ = self.expect_punct(P.RPAREN, "')'")
            return v
        if t.kind == TK.IDENT:
            _ = self.next()
            if self.is_punct(self.peek(), P.LPAREN):
                return self.parse_call(t)
            return self.lookup_value(t)
        raise self.err("QM-E500", "expected an expression, found " + self.describe(t), t, "angles look like 0.5, pi/2, -theta or 2*theta + 0.1")

    def parse_call(mut self, f: Token) raises -> Param:
        var nm = self.text(f)
        _ = self.expect_punct(P.LPAREN, "'('")
        var args = List[Param]()
        if not self.is_punct(self.peek(), P.RPAREN):
            args.append(self.parse_expr())
            while self.is_punct(self.peek(), P.COMMA):
                _ = self.next()
                args.append(self.parse_expr())
        _ = self.expect_punct(P.RPAREN, "')'")
        if nm == "float" or nm == "angle":
            if len(args) == 1:
                return args[0]
        if len(args) == 2 and nm == "pow":
            return Param(self.need_const(args[0], f, "argument of pow") ** self.need_const(args[1], f, "argument of pow"))
        if len(args) == 2 and nm == "mod":
            var x = self.need_const(args[0], f, "argument of mod")
            var y = self.need_const(args[1], f, "argument of mod")
            if y == 0.0:
                raise self.err("QM-E533", "modulo by zero", f, "check the right operand")
            return Param(x - y * floor(x / y))
        if len(args) != 1:
            raise self.err("QM-E534", "function '" + nm + "' called with " + String(len(args)) + " arguments", f, "supported: sin cos tan arcsin arccos arctan exp ln log sqrt abs floor ceil int float (1 arg), pow mod (2 args)")
        var x = self.need_const(args[0], f, "argument of " + nm + "()")
        if nm == "sin": return Param(sin(x))
        if nm == "cos": return Param(cos(x))
        if nm == "tan": return Param(tan(x))
        if nm == "arcsin" or nm == "asin": return Param(asin(x))
        if nm == "arccos" or nm == "acos": return Param(acos(x))
        if nm == "arctan" or nm == "atan": return Param(atan(x))
        if nm == "exp": return Param(exp(x))
        if nm == "ln" or nm == "log":
            if x <= 0.0:
                raise self.err("QM-E533", "logarithm of a non-positive number", f, "check the argument")
            return Param(log(x))
        if nm == "sqrt":
            if x < 0.0:
                raise self.err("QM-E533", "square root of a negative number", f, "check the argument")
            return Param(sqrt(x))
        if nm == "abs": return Param(abs(x))
        if nm == "floor": return Param(floor(x))
        if nm == "ceil": return Param(ceil(x))
        if nm == "int" or nm == "uint": return Param(Float64(Int(x)))
        var cands: List[String] = ["sin", "cos", "tan", "arcsin", "arccos", "arctan", "exp", "ln", "log", "sqrt", "abs", "floor", "ceil", "int", "float", "pow", "mod"]
        var sug = suggest_names(nm, cands)
        var hint = String("supported functions: sin cos tan arcsin arccos arctan exp ln sqrt abs floor ceil pow mod")
        if len(sug) > 0:
            hint = "did you mean " + _quote_list(sug) + "? " + hint
        raise self.err("QM-E534", "unknown function '" + nm + "'", f, hint)

    def parse_const_int(mut self, what: String) raises -> Int:
        var t = self.peek()
        var p = self.parse_expr()
        var v = self.need_const(p, t, what)
        if v != floor(v):
            raise self.err("QM-E530", what + " must be an integer, got " + String(v), t, "use an integer constant")
        return Int(v)

    # ------------------------------------------------------------ operands
    def parse_index_set(mut self, size: Int, reg_t: Token, mut out: List[Int]) raises -> Bool:
        """After '[': parse index / slice / set; appends absolute positions
        within the register. Returns True when it was a single index."""
        var t = self.peek()
        if self.is_punct(t, P.LBRACE):
            _ = self.next()
            while True:
                var it = self.peek()
                var v = self.parse_const_int("register index")
                out.append(self.norm_index(v, size, reg_t, it))
                if self.is_punct(self.peek(), P.COMMA):
                    _ = self.next()
                    continue
                break
            _ = self.expect_punct(P.RBRACE, "'}'")
            _ = self.expect_punct(P.RBRACK, "']'")
            return False
        var a = self.parse_const_int("register index")
        if self.is_punct(self.peek(), P.COLON):
            _ = self.next()
            var b = self.parse_const_int("slice bound")
            var step = 1
            if self.is_punct(self.peek(), P.COLON):
                _ = self.next()
                step = b
                b = self.parse_const_int("slice bound")
            _ = self.expect_punct(P.RBRACK, "']'")
            if step == 0:
                raise self.err("QM-E530", "slice step must be non-zero", t, "write q[0:2] (inclusive) or q[0:2:6]")
            var i = a
            while (step > 0 and i <= b) or (step < 0 and i >= b):
                out.append(self.norm_index(i, size, reg_t, t))
                i += step
            return False
        _ = self.expect_punct(P.RBRACK, "']'")
        out.append(self.norm_index(a, size, reg_t, t))
        return True

    def norm_index(self, v: Int, size: Int, reg_t: Token, t: Token) raises -> Int:
        var i = v
        if i < 0:
            i += size
        if i < 0 or i >= size:
            raise self.err("QM-E521", "index " + String(v) + " out of range for register '" + self.text(reg_t) + "' of size " + String(size), t, "valid indices are 0.." + String(size - 1) + " (slices like q[0:2] are inclusive)")
        return i

    def parse_qubit_operand(mut self) raises -> List[Int]:
        var out = List[Int]()
        _ = self.parse_qubit_operand_into(out)
        return out^

    def parse_qubit_operand_into(mut self, mut out: List[Int]) raises -> Int:
        """Append the global qubit indices of one operand; returns how many."""
        var t = self.peek()
        if t.kind == TK.HWQUBIT:
            _ = self.next()
            var k = Int(t.num)
            if self.circs[0].num_qubits <= k:
                for ref c in self.circs:
                    if c.num_qubits <= k:
                        c.num_qubits = k + 1
            out.append(k)
            return 1
        if t.kind != TK.IDENT:
            raise self.err("QM-E500", "expected a qubit operand, found " + self.describe(t), t, "operands look like q[0], q, q[0:2] or $0")
        _ = self.next()
        var nm = self.text(t)
        # gate-argument bindings (closed scope)
        var s = len(self.scopes) - 1
        var in_gate = False
        while s >= 0:
            for i in range(len(self.scopes[s].qnames)):
                if self.scopes[s].qnames[i] == nm:
                    if self.is_punct(self.peek(), P.LBRACK):
                        raise self.err("QM-E500", "gate argument '" + nm + "' is a single qubit and cannot be indexed", self.peek(), "use '" + nm + "' directly")
                    out.append(self.scopes[s].qubits[i])
                    return 1
            if self.scopes[s].closed:
                in_gate = True
                break
            s -= 1
        var ri = self.reg_index.get(nm)
        if not ri:
            if in_gate:
                raise self.err("QM-E520", "'" + nm + "' is not an argument of this gate", t, "gate bodies may only use their own qubit arguments")
            var cands = List[String]()
            for r in self.regs:
                if r.is_qubit:
                    cands.append(r.name)
            var sug = suggest_names(nm, cands)
            var hint = String("declare it first, e.g. 'qubit[2] " + nm + ";'")
            if len(sug) > 0:
                hint = "did you mean " + _quote_list(sug) + "?"
            raise self.err("QM-E520", "undeclared qubit register '" + nm + "'", t, hint)
        var r = ri.value()
        if not self.regs[r].is_qubit:
            raise self.err("QM-E520", "'" + nm + "' is a classical bit register, expected qubits", t, "gates act on qubit registers; bits are written by measure")
        if in_gate:
            raise self.err("QM-E520", "gate bodies cannot reference global register '" + nm + "'", t, "pass the qubit as a gate argument")
        var offset = self.regs[r].offset
        var size = self.regs[r].size
        if self.is_punct(self.peek(), P.LBRACK):
            # fast path: q[INT]
            var it = self.peek(1)
            if it.kind == TK.INT and self.is_punct(self.peek(2), P.RBRACK):
                self.pos += 3
                out.append(offset + self.norm_index(Int(it.num), size, t, it))
                return 1
            _ = self.next()
            var rel = List[Int]()
            _ = self.parse_index_set(size, t, rel)
            for x in rel:
                out.append(offset + x)
            return len(rel)
        for i in range(size):
            out.append(offset + i)
        return size

    def parse_bit_operand(mut self) raises -> List[Int]:
        var t = self.peek()
        if t.kind != TK.IDENT:
            raise self.err("QM-E500", "expected a classical bit operand, found " + self.describe(t), t, "write c[0] or c")
        _ = self.next()
        var nm = self.text(t)
        var ri = self.reg_index.get(nm)
        if not ri:
            var cands = List[String]()
            for r in self.regs:
                if not r.is_qubit:
                    cands.append(r.name)
            var sug = suggest_names(nm, cands)
            var hint = String("declare it first, e.g. 'bit[2] " + nm + ";'")
            if len(sug) > 0:
                hint = "did you mean " + _quote_list(sug) + "?"
            raise self.err("QM-E520", "undeclared bit register '" + nm + "'", t, hint)
        var reg = self.regs[ri.value()].copy()
        if reg.is_qubit:
            raise self.err("QM-E520", "'" + nm + "' is a qubit register, expected classical bits", t, "measurement results go to a 'bit[n]' / 'creg' register")
        var out = List[Int]()
        if self.is_punct(self.peek(), P.LBRACK):
            _ = self.next()
            var rel = List[Int]()
            _ = self.parse_index_set(reg.size, t, rel)
            for r in rel:
                out.append(reg.offset + r)
            return out^
        for i in range(reg.size):
            out.append(reg.offset + i)
        return out^

    # ------------------------------------------------------------ statements
    def parse_program(mut self) raises:
        while self.peek().kind != TK.EOF:
            self.parse_statement()

    def forbid_in_gate(self, t: Token) raises:
        if self.depth > 0:
            raise self.err("QM-E504", "'" + self.text(t) + "' is not allowed inside a gate body", t, "gate bodies may only contain gate calls, gphase and modifiers")

    def parse_statement(mut self) raises:
        var t = self.peek()
        self.stmt_count += 1
        if t.kind == TK.PRAGMA:
            self.parse_pragma()
            return
        if t.kind == TK.PUNCT:
            if t.sub == P.SEMI:
                _ = self.next()
                return
            if t.sub == P.LBRACE:
                _ = self.next()
                while not self.is_punct(self.peek(), P.RBRACE):
                    if self.peek().kind == TK.EOF:
                        raise self.err("QM-E500", "missing '}' to close block", t, "every '{' needs a matching '}'")
                    self.parse_statement()
                _ = self.next()
                return
            raise self.err("QM-E500", "unexpected " + self.describe(t) + " at start of statement", t, "statements start with a declaration, a gate name, measure, reset, barrier, if or for")
        if t.kind != TK.IDENT:
            raise self.err("QM-E500", "unexpected " + self.describe(t) + " at start of statement", t, "statements start with a declaration, a gate name, measure, reset, barrier, if or for")
        # fast path: the vast majority of statements are gate calls
        var nt = self.peek(1)
        if nt.kind == TK.IDENT or nt.kind == TK.HWQUBIT or self.is_punct(nt, P.LPAREN):
            var nm0 = self.text(t)
            if nm0 in self.gate_table or nm0 in self.user_gates:
                self.parse_gate_call()
                return
        if self.is_word(t, "OPENQASM"):
            self.parse_version()
        elif self.is_word(t, "include"):
            _ = self.next()
            var s = self.peek()
            if s.kind != TK.STRING:
                raise self.err("QM-E500", "expected a file name string after include", s, "write include \"stdgates.inc\";")
            _ = self.next()
            _ = self.expect_punct(P.SEMI, "';'")
        elif self.is_word(t, "qubit") or self.is_word(t, "qreg"):
            self.forbid_in_gate(t)
            self.parse_reg_decl(True)
        elif self.is_word(t, "bit") or self.is_word(t, "creg"):
            self.forbid_in_gate(t)
            self.parse_reg_decl(False)
        elif self.is_word(t, "input"):
            self.forbid_in_gate(t)
            self.parse_input()
        elif self.is_word(t, "output"):
            self.forbid_in_gate(t)
            _ = self.next()
            if self.is_word(self.peek(), "bit"):
                self.parse_reg_decl(False)
            else:
                raise self.err("QM-E504", "only 'output bit[n] c;' outputs are supported", t, "QUMO returns measured bits; drop other outputs")
        elif self.is_word(t, "const"):
            _ = self.next()
            self.parse_classical_decl(True)
        elif (self.is_word(t, "int") or self.is_word(t, "uint") or self.is_word(t, "float")
              or self.is_word(t, "angle") or self.is_word(t, "bool")):
            self.parse_classical_decl(False)
        elif self.is_word(t, "gate"):
            self.forbid_in_gate(t)
            self.parse_gate_def()
        elif self.is_word(t, "measure"):
            self.forbid_in_gate(t)
            self.parse_measure_arrow()
        elif self.is_word(t, "reset"):
            self.forbid_in_gate(t)
            _ = self.next()
            var qs = self.parse_qubit_operand()
            _ = self.expect_punct(P.SEMI, "';'")
            for q in qs:
                self.emit(Instruction.make(Op.RESET, q))
        elif self.is_word(t, "barrier"):
            _ = self.next()
            if not self.is_punct(self.peek(), P.SEMI):
                _ = self.parse_qubit_operand()
                while self.is_punct(self.peek(), P.COMMA):
                    _ = self.next()
                    _ = self.parse_qubit_operand()
            _ = self.expect_punct(P.SEMI, "';'")
            if self.depth == 0:
                self.circs[self.top()].ops.append(Instruction.make(Op.BARRIER, -1))
        elif self.is_word(t, "if"):
            self.forbid_in_gate(t)
            self.parse_if()
        elif self.is_word(t, "for"):
            self.parse_for()
        elif self.is_word(t, "delay"):
            self.parse_delay()
        elif self.is_word(t, "opaque"):
            raise self.err("QM-E504", "opaque gates are not supported", t, "define the gate with a body: gate name(params) a, b { ... }")
        elif self.is_word(t, "def") or self.is_word(t, "extern"):
            raise self.err("QM-E504", "subroutines ('" + self.text(t) + "') are not supported", t, "inline the subroutine body or define a 'gate'")
        elif self.is_word(t, "while"):
            raise self.err("QM-E504", "while loops are not supported", t, "QUMO circuits are straight-line; use 'for uint i in [0:n] { ... }' with constant bounds")
        elif (self.is_word(t, "box") or self.is_word(t, "cal") or self.is_word(t, "defcal")
              or self.is_word(t, "defcalgrammar") or self.is_word(t, "let") or self.is_word(t, "switch")
              or self.is_word(t, "return") or self.is_word(t, "break") or self.is_word(t, "continue")
              or self.is_word(t, "end") or self.is_word(t, "array") or self.is_word(t, "duration")
              or self.is_word(t, "stretch") or self.is_word(t, "complex")):
            raise self.err("QM-E504", "OpenQASM feature '" + self.text(t) + "' is not supported by QUMO", t, "QUMO supports the gate-level subset: declarations, gates, modifiers, measure, reset, barrier, if on one bit, constant for loops")
        else:
            # assignment `c = measure q;` / `c[0] = measure q[0];`
            var nm = self.text(t)
            var ri = self.reg_index.get(nm)
            if ri and not self.regs[ri.value()].is_qubit:
                self.forbid_in_gate(t)
                self.parse_measure_assign()
            elif ri:
                raise self.err("QM-E500", "statement starts with qubit register '" + nm + "'", t, "a gate name is missing, e.g. 'h " + nm + "[0];'")
            elif nm in self.consts or nm in self.inputs:
                raise self.err("QM-E504", "assignment to classical variable '" + nm + "' is not supported", t, "declare constants once with 'const float x = ...;'")
            else:
                self.parse_gate_call()

    def parse_version(mut self) raises:
        var t = self.next()
        if self.stmt_count > 1:
            raise self.err("QM-E500", "OPENQASM version must be the first statement", t, "move 'OPENQASM 3.0;' to the top of the file")
        var v = self.peek()
        if v.kind != TK.INT and v.kind != TK.FLOAT:
            raise self.err("QM-E503", "expected a version number after OPENQASM", v, "write 'OPENQASM 3.0;' or 'OPENQASM 2.0;'")
        _ = self.next()
        var major = Int(floor(v.num))
        if major != 2 and major != 3:
            raise self.err("QM-E503", "unsupported OpenQASM version " + self.text(v), v, "QUMO reads OpenQASM 2.0 and 3.x")
        self.version = major
        _ = self.expect_punct(P.SEMI, "';'")

    def parse_reg_decl(mut self, is_qubit: Bool) raises:
        var kw = self.next()
        var old_style = self.is_word(kw, "qreg") or self.is_word(kw, "creg")
        var size = 1
        var have_size = False
        if not old_style and self.is_punct(self.peek(), P.LBRACK):
            _ = self.next()
            var st = self.peek()
            size = self.parse_const_int("register size")
            _ = self.expect_punct(P.RBRACK, "']'")
            have_size = True
            if size <= 0:
                raise self.err("QM-E523", "register size must be positive, got " + String(size), st, "write qubit[2] q;")
        var nt = self.expect_ident("register name")
        self.check_new_name(nt)
        if old_style and self.is_punct(self.peek(), P.LBRACK):
            _ = self.next()
            var st = self.peek()
            size = self.parse_const_int("register size")
            _ = self.expect_punct(P.RBRACK, "']'")
            have_size = True
            if size <= 0:
                raise self.err("QM-E523", "register size must be positive, got " + String(size), st, "write qreg q[2];")
        elif not old_style and self.is_punct(self.peek(), P.LBRACK):
            raise self.err("QM-E500", "OpenQASM 3 puts the size on the type", self.peek(), "write '" + self.text(kw) + "[n] " + self.text(nt) + ";' (or qreg/creg for OpenQASM 2)")
        _ = have_size
        var init_measure = False
        if self.is_punct(self.peek(), P.EQ):
            if not is_qubit and self.is_word(self.peek(1), "measure"):
                init_measure = True  # `bit[2] c = measure q;`
            else:
                raise self.err("QM-E504", "initialized registers are only supported as 'bit[n] c = measure q;'", self.peek(), "declare 'bit[n] c;' (bits start at 0) and measure into it")
        if not init_measure:
            _ = self.expect_punct(P.SEMI, "';'")
        var nm = self.text(nt)
        var offset: Int
        if is_qubit:
            offset = self.circs[0].num_qubits
            for ref c in self.circs:
                c.num_qubits = offset + size
        else:
            offset = self.circs[0].num_clbits
            for ref c in self.circs:
                c.num_clbits = offset + size
        self.reg_index[nm] = len(self.regs)
        self.regs.append(Register(nm, offset, size, is_qubit))
        if init_measure:
            _ = self.next()  # '='
            var mt = self.next()  # 'measure'
            var qs = self.parse_qubit_operand()
            _ = self.expect_punct(P.SEMI, "';'")
            var cs = List[Int]()
            for i in range(size):
                cs.append(offset + i)
            self.emit_measures(qs, cs, mt)

    def skip_type(mut self) raises -> Token:
        """Consume a classical type (+ optional [designator]); return its token."""
        var tt = self.expect_ident("a type")
        if self.is_punct(self.peek(), P.LBRACK):
            _ = self.next()
            var depth = 1
            while depth > 0:
                var t = self.next()
                if t.kind == TK.EOF:
                    raise self.err("QM-E500", "missing ']' in type designator", tt)
                if self.is_punct(t, P.LBRACK):
                    depth += 1
                elif self.is_punct(t, P.RBRACK):
                    depth -= 1
        return tt

    def parse_input(mut self) raises:
        _ = self.next()
        var tt = self.skip_type()
        if not (self.is_word(tt, "float") or self.is_word(tt, "angle")):
            raise self.err("QM-E504", "only 'input float' / 'input angle' parameters are supported, got '" + self.text(tt) + "'", tt, "declare symbolic angles as 'input float theta;'")
        var nt = self.expect_ident("parameter name")
        self.check_new_name(nt)
        _ = self.expect_punct(P.SEMI, "';'")
        var nm = self.text(nt)
        var p = self.circs[0].parameter(nm)
        for i in range(1, len(self.circs)):
            _ = self.circs[i].parameter(nm)
        self.inputs[nm] = p.index

    def parse_classical_decl(mut self, is_const: Bool) raises:
        var tt = self.skip_type()
        if self.is_word(tt, "bit") or self.is_word(tt, "qubit"):
            raise self.err("QM-E504", "'const' registers are not supported", tt, "declare 'bit[n] c;' without const")
        var nt = self.expect_ident("variable name")
        self.check_new_name(nt)
        if not self.is_punct(self.peek(), P.EQ):
            raise self.err("QM-E504", "classical variables need an initializer", self.peek(), "write 'const float x = 0.5;' (QUMO folds classical values at parse time)")
        _ = self.next()
        var v = self.parse_expr()
        _ = self.expect_punct(P.SEMI, "';'")
        if self.is_word(tt, "int") or self.is_word(tt, "uint"):
            if not v.is_symbolic():
                v = Param(Float64(Int(v.offset)))
        _ = is_const
        self.consts[self.text(nt)] = v

    def parse_gate_def(mut self) raises:
        var gt = self.next()
        var nt = self.expect_ident("gate name")
        var nm = self.text(nt)
        var pnames = List[String]()
        var qnames = List[String]()
        if self.is_punct(self.peek(), P.LPAREN):
            _ = self.next()
            if not self.is_punct(self.peek(), P.RPAREN):
                pnames.append(self.text(self.expect_ident("parameter name")))
                while self.is_punct(self.peek(), P.COMMA):
                    _ = self.next()
                    pnames.append(self.text(self.expect_ident("parameter name")))
            _ = self.expect_punct(P.RPAREN, "')'")
        qnames.append(self.text(self.expect_ident("qubit argument name")))
        while self.is_punct(self.peek(), P.COMMA):
            _ = self.next()
            qnames.append(self.text(self.expect_ident("qubit argument name")))
        var lb = self.expect_punct(P.LBRACE, "'{' to start the gate body")
        var body_start = self.pos
        var depth = 1
        while depth > 0:
            var t = self.next()
            if t.kind == TK.EOF:
                raise self.err("QM-E500", "missing '}' to close gate '" + nm + "'", lb, "every '{' needs a matching '}'")
            if self.is_punct(t, P.LBRACE):
                depth += 1
            elif self.is_punct(t, P.RBRACE):
                depth -= 1
        var body_end = self.pos - 1
        var code = self.gate_table.get(nm)
        if code:
            var k = code.value()
            var np = _special_nparams(k) if k < 0 else op_num_params(k)
            var na = _special_arity(k) if k < 0 else op_arity(k)
            if np == len(pnames) and na == len(qnames):
                return  # declaration of a native gate: keep exact native semantics
            self.warn("QM-W050", "gate definition '" + nm + "' shadows a QUMO native gate with a different signature; the definition is used", nt, "rename the gate to avoid confusion")
            _ = self.gate_table.pop(nm)
        if nm in self.user_gates:
            raise self.err("QM-E522", "gate '" + nm + "' is already defined", nt, "rename one of the definitions")
        _ = gt
        self.user_gates[nm] = len(self.gates)
        self.gates.append(GateDef(nm, pnames^, qnames^, body_start, body_end))

    def parse_measure_arrow(mut self) raises:
        var mt = self.next()
        var qs = self.parse_qubit_operand()
        var cs = List[Int]()
        if self.is_punct(self.peek(), P.ARROW):
            _ = self.next()
            cs = self.parse_bit_operand()
        else:
            # bare `measure q;` -> fresh anonymous clbits
            var base = self.circs[0].num_clbits
            for i in range(len(qs)):
                cs.append(base + i)
        _ = self.expect_punct(P.SEMI, "';'")
        self.emit_measures(qs, cs, mt)

    def parse_measure_assign(mut self) raises:
        var ct = self.peek()
        var cs = self.parse_bit_operand()
        var eq = self.peek()
        if self.is_punct(eq, P.OPASSIGN) or not self.is_punct(eq, P.EQ):
            raise self.err("QM-E500", "expected '= measure ...' after bit operand, found " + self.describe(eq), eq, "write 'c[0] = measure q[0];'")
        _ = self.next()
        var mt = self.peek()
        if not self.is_word(mt, "measure"):
            raise self.err("QM-E504", "only measurement results can be assigned to bits", mt, "write 'c[0] = measure q[0];'")
        _ = self.next()
        var qs = self.parse_qubit_operand()
        _ = self.expect_punct(P.SEMI, "';'")
        _ = ct
        self.emit_measures(qs, cs, mt)

    def emit_measures(mut self, qs: List[Int], cs: List[Int], t: Token) raises:
        if len(qs) != len(cs):
            raise self.err("QM-E540", "measurement maps " + String(len(qs)) + " qubit(s) to " + String(len(cs)) + " bit(s)", t, "qubit and bit operands must have the same size")
        for i in range(len(qs)):
            self.circs[0].measure(qs[i], cs[i])
            if self.cond_bit >= 0:
                self.circs[0].c_if(self.cond_bit, self.cond_val)
                self.circs[0].ops[len(self.circs[0].ops) - 1].cond_reg = self.cond_reg

    def parse_condition(mut self) raises -> Tuple[Int, Int]:
        """`(c[0])`, `(!c[0])`, `(c[0] == 1)`, and whole-register `(c == N)`
        (OpenQASM 2 style, N of any size). Register conditions are recorded in
        self.cond_reg_bits / cond_reg_vals; the returned pair is the first bit."""
        var lp = self.expect_punct(P.LPAREN, "'(' after if")
        self.cond_reg_bits = List[Int]()
        self.cond_reg_vals = List[Int]()
        var negate = False
        if self.is_punct(self.peek(), P.BANG):
            _ = self.next()
            negate = True
        var bt = self.peek()
        var bits = self.parse_bit_operand()
        var val = 1
        var ct = self.peek()
        if self.is_punct(ct, P.EQEQ) or self.is_punct(ct, P.NEQ):
            _ = self.next()
            var vt = self.peek()
            if len(bits) > 1:
                if self.is_punct(ct, P.NEQ) or negate:
                    raise self.err("QM-E541", "'!=' / negation on a whole register is not supported", ct, "compare the register with '==' (OpenQASM 2 semantics)")
                if vt.kind != TK.INT:
                    raise self.err("QM-E541", "a register condition needs a non-negative integer literal", vt, "write if (c == 5)")
                _ = self.next()
                var vbits = decimal_to_bits(self.text(vt), len(bits))
                if len(vbits) == 0:
                    raise self.err("QM-E541", "condition value does not fit in the " + String(len(bits)) + "-bit register", vt, "use a value below 2^" + String(len(bits)))
                for i in range(len(bits)):
                    self.cond_reg_bits.append(bits[i])
                    self.cond_reg_vals.append(vbits[i])
                _ = self.expect_punct(P.RPAREN, "')' to close the condition")
                _ = lp
                return (bits[0], vbits[0])
            var v = self.parse_const_int("condition value")
            if v != 0 and v != 1:
                raise self.err("QM-E541", "a single bit can only be compared with 0 or 1", vt, "write if (c[0] == 1)")
            val = v
            if self.is_punct(ct, P.NEQ):
                val = 1 - val
        elif len(bits) > 1:
            raise self.err("QM-E541", "condition on a " + String(len(bits)) + "-bit register is ambiguous", bt, "index one bit: if (" + self.text(bt) + "[0]) ... or compare the register: if (" + self.text(bt) + " == 3)")
        if self.is_punct(self.peek(), P.ANDAND) or self.is_punct(self.peek(), P.OROR):
            raise self.err("QM-E541", "compound conditions are not supported", self.peek(), "nest the operations or test one bit per if")
        _ = self.expect_punct(P.RPAREN, "')' to close the condition")
        _ = lp
        if negate:
            val = 1 - val
        return (bits[0], val)

    def parse_if(mut self) raises:
        var it = self.next()
        if self.cond_bit >= 0:
            raise self.err("QM-E541", "nested if statements are not supported", it, "QUMO instructions carry one condition")
        var cv = self.parse_condition()
        self.cond_bit = cv[0]
        self.cond_val = cv[1]
        self.cond_reg = -1
        if len(self.cond_reg_bits) > 0:
            ref c0 = self.circs[0]
            self.cond_reg = len(c0.cond_table)
            c0.cond_table.append(len(self.cond_reg_bits))
            for i in range(len(self.cond_reg_bits)):
                c0.cond_table.append(self.cond_reg_bits[i])
                c0.cond_table.append(self.cond_reg_vals[i])
        self.parse_statement()
        if self.is_word(self.peek(), "else"):
            if self.cond_reg >= 0:
                raise self.err("QM-E541", "'else' after a whole-register condition is not supported", self.peek(), "test single bits when an else branch is needed")
            _ = self.next()
            self.cond_val = 1 - cv[1]
            self.parse_statement()
        self.cond_bit = -1
        self.cond_val = 1
        self.cond_reg = -1

    def skip_balanced(mut self, open: Int, close: Int, t: Token) raises:
        """At an `open` token: skip to just after its matching `close`."""
        var depth = 0
        while True:
            var x = self.next()
            if x.kind == TK.EOF:
                raise self.err("QM-E500", "unbalanced brackets", t, "check that every bracket is closed")
            if self.is_punct(x, open):
                depth += 1
            elif self.is_punct(x, close):
                depth -= 1
                if depth == 0:
                    return

    def skip_statement(mut self) raises:
        var t = self.peek()
        if self.is_punct(t, P.LBRACE):
            self.skip_balanced(P.LBRACE, P.RBRACE, t)
            return
        if self.is_word(t, "if"):
            _ = self.next()
            self.skip_balanced(P.LPAREN, P.RPAREN, t)
            self.skip_statement()
            if self.is_word(self.peek(), "else"):
                _ = self.next()
                self.skip_statement()
            return
        if self.is_word(t, "for"):
            while not self.is_word(self.peek(), "in"):
                if self.peek().kind == TK.EOF:
                    raise self.err("QM-E500", "malformed for loop", t)
                _ = self.next()
            _ = self.next()
            var r = self.peek()
            if self.is_punct(r, P.LBRACK):
                self.skip_balanced(P.LBRACK, P.RBRACK, r)
            elif self.is_punct(r, P.LBRACE):
                self.skip_balanced(P.LBRACE, P.RBRACE, r)
            self.skip_statement()
            return
        while True:
            var x = self.peek()
            if x.kind == TK.EOF:
                raise self.err("QM-E500", "missing ';'", t, "every statement ends with ';'")
            if self.is_punct(x, P.LPAREN):
                self.skip_balanced(P.LPAREN, P.RPAREN, x)
                continue
            if self.is_punct(x, P.LBRACK):
                self.skip_balanced(P.LBRACK, P.RBRACK, x)
                continue
            _ = self.next()
            if self.is_punct(x, P.SEMI):
                return

    def parse_for(mut self) raises:
        var ft = self.next()
        # optional loop-variable type
        if not self.is_word(self.peek(1), "in"):
            _ = self.skip_type()
        var vt = self.expect_ident("loop variable")
        var it = self.peek()
        if not self.is_word(it, "in"):
            raise self.err("QM-E500", "expected 'in' in for loop, found " + self.describe(it), it, "write 'for uint i in [0:3] { ... }' (ranges are inclusive)")
        _ = self.next()
        var values = List[Int]()
        var rt = self.peek()
        if self.is_punct(rt, P.LBRACK):
            _ = self.next()
            var a = self.parse_const_int("range start")
            _ = self.expect_punct(P.COLON, "':' in range", "write [start:stop] or [start:step:stop] (inclusive)")
            var b = self.parse_const_int("range bound")
            var step = 1
            if self.is_punct(self.peek(), P.COLON):
                _ = self.next()
                step = b
                b = self.parse_const_int("range stop")
            _ = self.expect_punct(P.RBRACK, "']' to close the range")
            if step == 0:
                raise self.err("QM-E570", "range step must be non-zero", rt, "write [0:2:10]")
            var i = a
            while (step > 0 and i <= b) or (step < 0 and i >= b):
                values.append(i)
                i += step
                if len(values) > _MAX_UNROLL:
                    raise self.err("QM-E570", "for loop has more than " + String(_MAX_UNROLL) + " iterations", rt, "QUMO unrolls loops at parse time; reduce the range")
        elif self.is_punct(rt, P.LBRACE):
            _ = self.next()
            values.append(self.parse_const_int("set element"))
            while self.is_punct(self.peek(), P.COMMA):
                _ = self.next()
                values.append(self.parse_const_int("set element"))
            _ = self.expect_punct(P.RBRACE, "'}' to close the set")
        else:
            raise self.err("QM-E570", "for loops need a constant range [a:b] or set {a, b}", rt, "write 'for uint i in [0:3] { ... }' (ranges are inclusive)")
        self.unrolled += len(values)
        if self.unrolled > _MAX_UNROLL:
            raise self.err("QM-E570", "too many unrolled loop iterations", ft, "QUMO unrolls loops at parse time; keep the total below " + String(_MAX_UNROLL))
        var body = self.pos
        if len(values) == 0:
            self.skip_statement()
            return
        var nm = self.text(vt)
        for v in values:
            var sc = Scope(False)
            sc.names.append(nm)
            sc.values.append(Param(Float64(v)))
            self.scopes.append(sc^)
            self.pos = body
            self.parse_statement()
            _ = self.scopes.pop()

    def parse_delay(mut self) raises:
        var dt = self.next()
        if self.is_punct(self.peek(), P.LBRACK):
            self.skip_balanced(P.LBRACK, P.RBRACK, self.peek())
        if not self.is_punct(self.peek(), P.SEMI):
            _ = self.parse_qubit_operand()
            while self.is_punct(self.peek(), P.COMMA):
                _ = self.next()
                _ = self.parse_qubit_operand()
        _ = self.expect_punct(P.SEMI, "';'")
        self.warn("QM-W051", "delay is ignored (QUMO circuits have no timing model)", dt, "model idle noise with '#pragma qumo depolarize1(p) q[0];'")

    def parse_pragma(mut self) raises:
        var pt = self.next()
        var t = self.peek()
        if self.is_word(t, "detector") or self.is_word(t, "observable"):
            self.forbid_in_gate(t)
            _ = self.next()
            var index = 0
            var is_obs = self.is_word(t, "observable")
            if is_obs:
                _ = self.expect_punct(P.LPAREN, "'(' after observable", "write '#pragma qumo observable(0) 3, 5'")
                index = self.parse_const_int("observable index")
                _ = self.expect_punct(P.RPAREN, "')'")
            var recs = List[Int]()
            if not self.is_punct(self.peek(), P.SEMI):
                while True:
                    var rt = self.peek()
                    var r = self.parse_const_int("measurement record index")
                    if r < 0:
                        r += self.circs[0].num_measurements
                    if r < 0 or r >= self.circs[0].num_measurements:
                        raise self.err("QM-E560", "measurement record " + String(r) + " does not exist yet (" + String(self.circs[0].num_measurements) + " measurements so far)", rt, "records are absolute measurement indices (0 = first measurement), or negative to count back")
                    recs.append(r)
                    if self.is_punct(self.peek(), P.COMMA):
                        _ = self.next()
                        continue
                    break
            _ = self.expect_punct(P.SEMI, "end of pragma line")
            if is_obs:
                self.circs[0].observable(index, recs)
            else:
                self.circs[0].detector(recs)
            return
        if self.is_word(t, "tick"):
            _ = self.next()
            _ = self.expect_punct(P.SEMI, "end of pragma line")
            self.circs[self.top()].ops.append(Instruction.make(Op.TICK, -1))
            return
        if self.is_word(t, "param"):
            _ = self.next()
            var it = self.expect_ident("parameter identifier")
            var st = self.peek()
            if st.kind != TK.STRING:
                raise self.err("QM-E560", "expected the original parameter name as a string", st, "write '#pragma qumo param theta_0 \"theta[0]\"'")
            _ = self.next()
            _ = self.expect_punct(P.SEMI, "end of pragma line")
            var ip = self.inputs.get(self.text(it))
            if not ip:
                raise self.err("QM-E560", "'" + self.text(it) + "' is not a declared input parameter", it, "declare 'input float " + self.text(it) + ";' before the pragma")
            var nm = self.text(st)
            for ref c in self.circs:
                if ip.value() < len(c.param_names):
                    c.param_names[ip.value()] = nm
            return
        if t.kind == TK.PUNCT and t.sub == P.SEMI:
            raise self.err("QM-E560", "empty qumo pragma", pt, "e.g. '#pragma qumo depolarize1(0.01) q[0];', '#pragma qumo detector 0, 1'")
        # any other statement (noise channels, conditioned noise, ...)
        self.parse_statement()

    # ------------------------------------------------------------ gate calls
    def parse_gate_call(mut self) raises:
        var first = self.peek()
        var mods = List[Int]()  # 1 inv, 2 ctrl, 3 negctrl, 4 pow
        var margs = List[Int]()
        var mtoks = List[Token]()
        while True:
            var t = self.peek()
            if t.kind != TK.IDENT:
                break
            var is_inv = self.is_word(t, "inv")
            var is_ctrl = self.is_word(t, "ctrl")
            var is_neg = self.is_word(t, "negctrl")
            var is_pow = self.is_word(t, "pow")
            if not (is_inv or is_ctrl or is_neg or is_pow):
                break
            # must be followed by '@' (possibly after (...))
            var save = self.pos
            _ = self.next()
            var arg = 1
            if self.is_punct(self.peek(), P.LPAREN):
                _ = self.next()
                var at = self.peek()
                if is_pow:
                    var pv = self.parse_expr()
                    var x = self.need_const(pv, at, "pow() exponent")
                    if x != floor(x):
                        raise self.err("QM-E550", "pow(" + String(x) + ") @ needs an integer exponent", at, "QUMO supports integer powers; use inv @ for -1")
                    arg = Int(x)
                elif is_inv:
                    raise self.err("QM-E550", "inv takes no argument", at, "write 'inv @ gate ...'")
                else:
                    arg = self.parse_const_int("number of controls")
                    if arg < 1:
                        raise self.err("QM-E550", "ctrl count must be >= 1", at, "write ctrl(2) @ x a, b, c;")
                _ = self.expect_punct(P.RPAREN, "')'")
            elif is_pow:
                raise self.err("QM-E550", "pow needs an exponent", self.peek(), "write 'pow(2) @ gate ...'")
            if not self.is_punct(self.peek(), P.AT):
                if is_pow and not self.is_punct(self.peek(), P.AT):
                    raise self.err("QM-E500", "expected '@' after modifier, found " + self.describe(self.peek()), self.peek(), "write 'pow(2) @ x q[0];'")
                self.pos = save
                break
            _ = self.next()
            mods.append(1 if is_inv else (2 if is_ctrl else (3 if is_neg else 4)))
            margs.append(arg)
            mtoks.append(t)
        var nt = self.expect_ident("gate name")
        var nm = self.text(nt)
        var code = 0
        var user = -1
        var gc = self.gate_table.get(nm)
        if gc:
            code = gc.value()
        else:
            var ug = self.user_gates.get(nm)
            if ug:
                user = ug.value()
            else:
                var sug = suggest_names(nm, self.known_gate_names())
                var hint = String("run 'qumo describe gates' for the gate set, or define it: gate " + nm + "(params) a, b { ... }")
                if len(sug) > 0:
                    hint = "did you mean " + _quote_list(sug) + "?"
                if nm == "measure" or nm == "reset":
                    hint = "'" + nm + "' cannot take modifiers"
                raise self.err("QM-E510", "unknown gate '" + nm + "'", nt, hint)
        var ps = List[Param]()
        if self.is_punct(self.peek(), P.LPAREN):
            _ = self.next()
            if not self.is_punct(self.peek(), P.RPAREN):
                ps.append(self.parse_expr())
                while self.is_punct(self.peek(), P.COMMA):
                    _ = self.next()
                    ps.append(self.parse_expr())
            _ = self.expect_punct(P.RPAREN, "')' to close the parameter list")
        var flat = List[Int](capacity=4)
        var lens = List[Int](capacity=4)
        if not self.is_punct(self.peek(), P.SEMI):
            lens.append(self.parse_qubit_operand_into(flat))
            while self.is_punct(self.peek(), P.COMMA):
                _ = self.next()
                lens.append(self.parse_qubit_operand_into(flat))
        _ = self.expect_punct(P.SEMI, "';' after gate operands", "separate operands with ',' and end with ';'")
        # signature checks
        var np: Int
        var na: Int
        if user >= 0:
            np = len(self.gates[user].pnames)
            na = len(self.gates[user].qnames)
        elif code < 0:
            np = _special_nparams(code)
            na = _special_arity(code)
        else:
            np = op_num_params(code)
            na = op_arity(code)
        if len(ps) != np:
            raise self.err("QM-E511", "gate '" + nm + "' takes " + String(np) + " parameter(s), got " + String(len(ps)), nt, self.usage(nm, code, user))
        var nctrl = 0
        for i in range(len(mods)):
            if mods[i] == 2 or mods[i] == 3:
                nctrl += margs[i]
        var nops = len(lens)
        if nops != na + nctrl:
            var extra = String("")
            if nctrl > 0:
                extra = " (" + String(nctrl) + " control(s) + " + String(na) + " target(s))"
            var uh = self.usage(nm, code, user)
            if nctrl > 0:
                uh += " The modifiers add " + String(nctrl) + " leading control qubit(s): list controls first, then the targets"
            raise self.err("QM-E512", "gate '" + nm + "' expects " + String(na + nctrl) + " qubit operand(s)" + extra + ", got " + String(nops), nt, uh)
        # broadcasting
        var width = 1
        for o in lens:
            if o > 1:
                if width > 1 and o != width:
                    raise self.err("QM-E514", "register operands of different sizes (" + String(width) + " vs " + String(o) + ") cannot be broadcast", nt, "broadcast operands must have equal size, or index single qubits")
                width = max(width, o)
        var qs = List[Int](capacity=nops)
        for b in range(width):
            qs.clear()
            var base = 0
            for o in lens:
                qs.append(flat[base + b] if o > 1 else flat[base])
                base += o
            for i in range(len(qs)):
                for j in range(i + 1, len(qs)):
                    if qs[i] == qs[j]:
                        raise self.err("QM-E513", "gate '" + nm + "' uses qubit " + String(qs[i]) + " more than once", nt, "each operand of a gate must be a distinct qubit")
            if len(mods) == 0:
                self.apply_base(code, user, ps, qs, nt)
            else:
                self.apply_modified(code, user, ps, qs, mods, margs, mtoks, nctrl, nt)
        _ = first

    def usage(self, nm: String, code: Int, user: Int) -> String:
        var np: Int
        var na: Int
        if user >= 0:
            np = len(self.gates[user].pnames)
            na = len(self.gates[user].qnames)
        elif code < 0:
            np = _special_nparams(code)
            na = _special_arity(code)
        else:
            np = op_num_params(code)
            na = op_arity(code)
        var s = "usage: " + nm
        if np > 0:
            s += "("
            for i in range(np):
                if i > 0:
                    s += ", "
                s += "p" + String(i)
            s += ")"
        for i in range(na):
            s += " q[" + String(i) + "]" if i == 0 else ", q[" + String(i) + "]"
        s += ";"
        if code == Op.CX or code == Op.CY or code == Op.CH or code == Op.CRX or code == Op.CRY or code == Op.CRZ or code == Op.CP:
            s += " (control first, then target)"
        return s^

    def apply_base(mut self, code: Int, user: Int, ps: List[Param], qs: List[Int], t: Token) raises:
        if user >= 0:
            self.expand_user_gate(user, ps, qs, t)
            return
        if code >= 0:
            var q0 = qs[0] if len(qs) > 0 else -1
            var q1 = qs[1] if len(qs) > 1 else -1
            var q2 = qs[2] if len(qs) > 2 else -1
            var p0 = ps[0] if len(ps) > 0 else Param(0.0)
            var p1 = ps[1] if len(ps) > 1 else Param(0.0)
            var p2 = ps[2] if len(ps) > 2 else Param(0.0)
            self.emit(Instruction.make(code, q0, q1, q2, p0, p1, p2))
            return
        if code == G.GPHASE:
            var v = self.need_const(ps[0], t, "gphase angle")
            self.add_phase(v)
            return
        if code == G.U2 or code == G.U3:
            var th = Param(pi / 2) if code == G.U2 else ps[0]
            var ph = ps[0] if code == G.U2 else ps[1]
            var la = ps[1] if code == G.U2 else ps[2]
            # Qiskit semantics: u3 == U, u2(phi, lambda) == U(pi/2, phi, lambda)
            # exactly (no implicit global phase), matching qiskit.qasm2/qasm3.
            self.emit(Instruction.make(Op.U, qs[0], -1, -1, th, ph, la))
            return
        # controlled specials: build sub-circuit and use the Controlled functor
        var c = qs[0]
        var tq = qs[1]
        self.push_target()
        if code == G.CSX:
            self.emit(Instruction.make(Op.SX, tq))
        else:
            self.emit(Instruction.make(Op.U, tq, -1, -1, ps[0], ps[1], ps[2]))
        var sub = self.pop_target()
        var ctl: Circuit
        try:
            ctl = sub.controlled(c)
        except e:
            raise self.relocate(String(e), t)
        if code == G.CU:
            self.emit(Instruction.make(Op.P, c, -1, -1, ps[3]))
        self.compose_sub(ctl)

    def apply_modified(mut self, code: Int, user: Int, ps: List[Param], qs: List[Int],
                       mods: List[Int], margs: List[Int], mtoks: List[Token], nctrl: Int, t: Token) raises:
        var base = List[Int]()
        for i in range(nctrl, len(qs)):
            base.append(qs[i])
        self.push_target()
        self.apply_base(code, user, ps, base, t)
        var sub = self.pop_target()
        var ci = nctrl
        var m = len(mods) - 1
        while m >= 0:
            var kind = mods[m]
            try:
                if kind == 1:
                    sub = sub.inverse()
                elif kind == 4:
                    sub = sub.power(margs[m])
                else:
                    var k = margs[m]
                    for j in range(ci - k, ci):
                        var cq = qs[j]
                        var ctl = sub.controlled(cq)
                        if kind == 3:
                            var w = sub.empty_like()
                            w.num_qubits = ctl.num_qubits
                            w.global_phase = 0.0
                            w.x(cq)
                            w.compose(ctl)
                            w.x(cq)
                            sub = w^
                        else:
                            sub = ctl^
                    ci -= k
            except e:
                raise self.relocate(String(e), mtoks[m])
            m -= 1
        self.compose_sub(sub)

    def expand_user_gate(mut self, gi: Int, ps: List[Param], qs: List[Int], t: Token) raises:
        if self.depth >= _MAX_GATE_DEPTH:
            raise self.err("QM-E551", "gate expansion deeper than " + String(_MAX_GATE_DEPTH) + " (recursive gate definition?)", t, "gates must not call themselves")
        var g = self.gates[gi].copy()
        var sc = Scope(True)
        sc.names = g.pnames.copy()
        sc.values = ps.copy()
        sc.qnames = g.qnames.copy()
        sc.qubits = qs.copy()
        self.scopes.append(sc^)
        self.call_names.append(g.name)
        self.call_lines.append(t.line)
        var saved = self.pos
        self.pos = g.body_start
        self.depth += 1
        while self.pos < g.body_end:
            self.parse_statement()
        self.depth -= 1
        self.pos = saved
        _ = self.scopes.pop()
        _ = self.call_names.pop()
        _ = self.call_lines.pop()

    def finish(mut self) raises -> Circuit:
        var c = self.circs.pop()
        try:
            c.validate()
        except e:
            raise Error(String(e))
        return c^


# ===========================================================================
# Public parsing API
# ===========================================================================
def parse_qasm_with_warnings(source: String, mut warnings: List[String], name: String = "qasm") raises -> Circuit:
    """Parse OpenQASM 2/3 source; parser warnings (QM-W05x) are appended."""
    var p = QasmParser(source, name)
    p.parse_program()
    for w in p.warnings:
        warnings.append(w)
    return p.finish()


def parse_qasm(source: String, name: String = "qasm") raises -> Circuit:
    """Parse OpenQASM 2.0 / 3.0 source into a `Circuit` (raises QM-E5xx)."""
    var p = QasmParser(source, name)
    p.parse_program()
    return p.finish()


def load_qasm(path: String) raises -> Circuit:
    var text: String
    try:
        with open(path, "r") as f:
            text = f.read()
    except e:
        raise Error("QM-E920: cannot read file '" + path + "': " + String(e) + ". Hint: check the path and permissions.")
    return parse_qasm(text, path)


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


def to_qasm(circuit: Circuit) -> String:
    """Canonical OpenQASM 3 text for `circuit` (round-trips through parse_qasm)."""
    var out = String("OPENQASM 3.0;\ninclude \"stdgates.inc\";\n")
    # gate definitions for natives outside stdgates.inc
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
    for g in circuit.ops:
        var k = g.kind
        var cond = String()
        if g.cond >= 0:
            if g.cond_reg >= 0:
                try:
                    cond = "if (c == " + register_condition_value(circuit, g) + ") "
                except e:
                    cond = "// QM-E552 (partial register condition dropped): "
            else:
                cond = "if (c[" + String(g.cond) + "] == " + String(g.cond_val) + ") "
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
    return out^
