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
from qumo.ir.mc import controlled_by
from qumo.ir.kraus import KrausChannel
from qumo.ir.param import Param
from qumo.ir.classical import (
    CExpr, CN, CF, CT, CType, CNode, type_name, bit_type, bool_type, int_type, uint_type, float_type,
    angle_type, ce_binary, ce_bool, ce_bnot, ce_cast, ce_explicit_cast, ce_function, ce_lnot, ce_logical, ce_neg,
    angle_radians,
)
from qumo.ir.program import CReg, FORR, Program, SK, TapeBuilder, check_program, slot_prefix


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
    comptime ANNOTATION = 8  # `@keyword payload` up to the end of the line


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


def _at_statement_start(toks: List[Token]) -> Bool:
    """True after ';', '{', '}' or another annotation: '@' starts an annotation
    there, whereas the '@' of a gate modifier always follows a modifier token."""
    if len(toks) == 0:
        return True
    ref t = toks[len(toks) - 1]
    if t.kind == TK.ANNOTATION:
        return True
    return t.kind == TK.PUNCT and (t.sub == P.SEMI or t.sub == P.LBRACE or t.sub == P.RBRACE)


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
        if c == 0x40 and (not line_has_tok or (_is_ident_start(Int(bs[i + 1]) if i + 1 < n else 0) and _at_statement_start(toks))):
            # annotation: its payload is free-form text up to the end of the line
            var j = i
            while j < n and Int(bs[j]) != 0x0A:
                j += 1
            _push(toks, TK.ANNOTATION, 0, i, j, line)
            i = j
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
            elif j - i > 18:
                # past 64 bits iv wrapped: the token's value comes from the digits
                var digits = String(unsafe_from_utf8=bs[i:j]).replace("_", "")
                var v: Float64
                try:
                    v = Float64(digits)
                except:
                    raise _lex_error(src, "QM-E501", "malformed number", i, line, "write numbers like 0.5, 1e-3 or 3")
                _push(toks, TK.INT, 0, i, j, line, v)
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
                if i + 2 < n and Int(bs[i + 2]) == 0x3D:
                    sub = P.OPASSIGN
                    w = 3
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
        elif c == 0x25:
            if d == 0x3D:
                sub = P.OPASSIGN
                w = 2
            else:
                sub = P.PERCENT
        elif c == 0x3C:
            if d == 0x3D:
                sub = P.LE
                w = 2
            elif d == 0x3C:
                sub = P.SHL
                w = 2
                if i + 2 < n and Int(bs[i + 2]) == 0x3D:
                    sub = P.OPASSIGN
                    w = 3
            else:
                sub = P.LT
        elif c == 0x3E:
            if d == 0x3D:
                sub = P.GE
                w = 2
            elif d == 0x3E:
                sub = P.SHR
                w = 2
                if i + 2 < n and Int(bs[i + 2]) == 0x3D:
                    sub = P.OPASSIGN
                    w = 3
            else:
                sub = P.GT
        elif c == 0x26:
            if d == 0x26:
                sub = P.ANDAND
                w = 2
            elif d == 0x3D:
                sub = P.OPASSIGN
                w = 2
            else:
                sub = P.AMP
        elif c == 0x7C:
            if d == 0x7C:
                sub = P.OROR
                w = 2
            elif d == 0x3D:
                sub = P.OPASSIGN
                w = 2
            else:
                sub = P.PIPE
        elif c == 0x5E:
            if d == 0x3D:
                sub = P.OPASSIGN
                w = 2
            else:
                sub = P.CARET
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
    var exprs: List[CExpr]  # typed values of `names`, for loop variables; else empty

    def __init__(out self, closed: Bool):
        self.names = List[String]()
        self.values = List[Param]()
        self.qnames = List[String]()
        self.qubits = List[Int]()
        self.closed = closed
        self.exprs = List[CExpr]()


comptime _MAX_GATE_DEPTH = 64
comptime _MAX_UNROLL = 1000000


struct NK:
    """What a name of a program scope stands for."""

    comptime VAR = 0  # index: variable id
    comptime CONST = 1  # index: into const_vals, a folded declaration or a loop value
    comptime REG = 2  # index: into regs, a bit register declared in a block


@fieldwise_init
struct _Name(Copyable):
    var name: String
    var kind: Int
    var index: Int


@fieldwise_init
struct _LoopName(Copyable):
    """A loop variable counts as declared first in the body of its loop."""

    var name: String
    var depth: Int  # block depth of the body


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
    var int_vars: Dict[String, List[Int]]  # `int x;` + `x = c;`: the bits x mirrors (for switch)
    var int_var_at: Dict[String, Int]  # len(ops) at `x = c;`: x is stale once those bits are rewritten
    var scopes: List[Scope]
    var cond_bit: Int
    var cond_val: Int
    var cond_reg: Int  # offset into circs[0].cond_table for register conditions (-1 none)
    var cond_reg_bits: List[Int]
    var cond_reg_vals: List[Int]
    var pragma_if: Bool  # parsing `#pragma qumo if (...) <pragma body>`
    var call_names: List[String]
    var call_lines: List[Int]
    var warnings: List[String]
    var depth: Int
    var unrolled: Int
    var stmt_count: Int
    # program mode (parse_qasm_program): statements go to a tape, the ops of
    # circs[0] are its arena, and classical variables exist at run time
    var program_mode: Bool
    var prog: Program
    var tb: TapeBuilder
    var reassigned: Dict[String, Bool]  # names assigned after their declaration
    var const_exprs: Dict[String, CExpr]  # typed values of global declarations (both parsers)
    var input_types: Dict[String, CType]  # `input` of a type other than float
    var names: List[_Name]  # visible names of program scopes, innermost last
    var scope_marks: List[Int]  # len(names) at each open block
    var const_vals: List[CExpr]  # values of NK.CONST names
    var scoped_regs: Dict[Int, Int]  # token of a block's `bit` declaration -> its register
    var temp_count: Int
    var block_depth: Int  # open `{`, in both parsers
    var loop_names: List[_LoopName]

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
        self.int_vars = Dict[String, List[Int]]()
        self.int_var_at = Dict[String, Int]()
        self.scopes = List[Scope]()
        self.cond_bit = -1
        self.cond_val = 1
        self.cond_reg = -1
        self.cond_reg_bits = List[Int]()
        self.cond_reg_vals = List[Int]()
        self.pragma_if = False
        self.call_names = List[String]()
        self.call_lines = List[Int]()
        self.warnings = List[String]()
        self.depth = 0
        self.unrolled = 0
        self.stmt_count = 0
        self.program_mode = False
        self.prog = Program(Circuit(0, 0, name))
        self.tb = TapeBuilder()
        self.reassigned = Dict[String, Bool]()
        self.const_exprs = Dict[String, CExpr]()
        self.input_types = Dict[String, CType]()
        self.names = List[_Name]()
        self.scope_marks = List[Int]()
        self.const_vals = List[CExpr]()
        self.scoped_regs = Dict[Int, Int]()
        self.temp_count = 0
        self.block_depth = 0
        self.loop_names = List[_LoopName]()

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

    def condition_last(mut self):
        """Attach the active `if` / `switch` condition to the op just appended
        by a Circuit builder (the channels kept in side tables bypass `emit`)."""
        if self.cond_bit >= 0 and len(self.circs) == 1:
            ref g = self.circs[0].ops[len(self.circs[0].ops) - 1]
            g.cond = self.cond_bit
            g.cond_val = self.cond_val
            g.cond_reg = self.cond_reg

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
        for e in self.names:
            out.append(e.name)
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
        for l in self.loop_names:
            if l.name == nm and l.depth == self.block_depth:
                raise self.err("QM-E522", "'" + nm + "' is the loop variable, declared first in this body", t, "pick another name, or declare it in an inner block")
        if self.program_mode:
            var inner = len(self.scope_marks) > 0
            var lo = self.scope_marks[len(self.scope_marks) - 1] if inner else 0
            for i in range(lo, len(self.names)):
                if self.names[i].name == nm:
                    raise self.err("QM-E522", "'" + nm + "' is already declared in this scope", t, "pick another name, or declare it in an inner block")
            if inner:
                return  # a block may shadow what lies outside it
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
        var k = self.find_name(nm) if self.program_mode and self.depth == 0 else -1
        if k >= 0:
            ref e = self.names[k]
            if e.kind == NK.CONST:
                return Param(_const_radians(self.const_vals[e.index]))
            raise self.err("QM-E584", "'" + nm + "' changes at run time, and a constant is needed here", t, "qubit indices, sizes and modifiers are fixed at parse time; use a constant, or a for loop that can unroll")
        var c = self.consts.get(nm)
        if c:
            return c.value()
        var ip = self.inputs.get(nm)
        if ip:
            return Param.symbol(ip.value())
        if nm in self.reg_index:
            if self.program_mode:
                raise self.err("QM-E584", "register '" + nm + "' changes at run time, and a constant is needed here", t, "qubit indices, sizes and modifiers are fixed at parse time")
            raise self.err("QM-E504", "reading register '" + nm + "' in an expression needs a value at run time", t, "parse with parse_qasm_program or load_any; conditions use 'if (c[0] == 1)'; angles must be numbers, constants or 'input float' parameters")
        if nm in self.int_vars:
            raise self.err("QM-E504", "reading '" + nm + "' in an expression needs a value at run time", t, "parse with parse_qasm_program or load_any")
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

    def literal_ahead(self, at: Int) -> Bool:
        """The token at offset `at` ends a constant: no operator follows it."""
        var n = self.peek(at)
        if n.kind != TK.PUNCT:
            return n.kind == TK.IDENT or n.kind == TK.EOF
        return n.sub == P.COMMA or n.sub == P.RPAREN or n.sub == P.RBRACK or n.sub == P.COLON or n.sub == P.RBRACE or n.sub == P.LBRACE or n.sub == P.SEMI

    def parse_const_int(mut self, what: String) raises -> Int:
        var t = self.peek()
        # a plain literal stays exact: the float path rounds above 2^53
        var neg = self.is_punct(t, P.MINUS)
        var lit = self.peek(1) if neg else t
        if lit.kind == TK.INT and self.literal_ahead(2 if neg else 1):
            var value = self.parse_c_literal(lit)
            if value.root().ty.tag == CT.UINT and not (neg and value.const_int() == -9223372036854775807 - 1):
                raise self.err("QM-E580", what + " does not fit a 64-bit int", lit, "use a smaller value")
            _ = self.next()
            if neg:
                _ = self.next()
            return 0 - value.const_int() if neg else value.const_int()
        if (self.program_mode or len(self.const_exprs) > 0 or len(self.scopes) > 0) and self.reads_typed():
            # typed constants compute in their types, exactly
            var e = self.parse_cexpr()
            if not e.is_const():
                raise self.err("QM-E584", what + " must be a constant", t, "use a constant, or a for loop that can unroll")
            if e.ty().tag != CT.FLOAT:
                if e.ty().tag == CT.UINT and e.const_int() < 0:
                    raise self.err("QM-E580", what + " does not fit a 64-bit int", t, "use a smaller value")
                return e.const_int()
            var x = e.const_float()
            if x != floor(x) or abs(x) >= 9007199254740992.0:
                raise self.err("QM-E530", what + " must be an integer, got " + String(x), t, "use an integer constant")
            return Int(x)
        var p = self.parse_expr()
        var v = self.need_const(p, t, what)
        if v != floor(v):
            raise self.err("QM-E530", what + " must be an integer, got " + String(v), t, "use an integer constant")
        if abs(v) >= 9007199254740992.0:
            raise self.err("QM-E580", what + " is too large to compute exactly", t, "write it as one integer literal")
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
        var ri = self.find_register(nm)
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
        var ri = self.find_register(nm)
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
        if t.kind == TK.ANNOTATION:
            _ = self.next()
            self.warn("QM-W052", "annotation '" + self.text(t) + "' is ignored", t, "annotations carry no meaning in QUMO; the statement after it still runs")
            return
        if t.kind == TK.PUNCT:
            if t.sub == P.SEMI:
                _ = self.next()
                return
            if t.sub == P.LBRACE:
                _ = self.next()
                self.push_scope()
                self.block_depth += 1
                while not self.is_punct(self.peek(), P.RBRACE):
                    if self.peek().kind == TK.EOF:
                        raise self.err("QM-E500", "missing '}' to close block", t, "every '{' needs a matching '}'")
                    self.parse_statement()
                _ = self.next()
                self.block_depth -= 1
                self.pop_scope()
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
            elif self.program_mode:
                self.parse_decl_program(False, True)
            else:
                raise self.err("QM-E504", "only 'output bit[n] c;' outputs are supported", t, "QUMO returns measured bits; parse with parse_qasm_program or load_any for variable outputs")
        elif self.is_word(t, "const"):
            _ = self.next()
            if self.program_mode:
                self.parse_decl_program(True, False)
            else:
                self.parse_classical_decl(True)
        elif (self.is_word(t, "int") or self.is_word(t, "uint") or self.is_word(t, "float")
              or self.is_word(t, "angle") or self.is_word(t, "bool")):
            if self.program_mode:
                self.forbid_in_gate(t)
                self.parse_decl_program(False, False)
            else:
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
            if not self.program_mode:
                raise self.err("QM-E504", "a while loop needs a Program", t, "parse with parse_qasm_program or load_any; a Circuit is straight-line")
            self.forbid_in_gate(t)
            self.parse_while()
        elif self.program_mode and (self.is_word(t, "break") or self.is_word(t, "continue")):
            self.forbid_in_gate(t)
            _ = self.next()
            _ = self.expect_punct(P.SEMI, "';'")
            try:
                self.tb.jump(SK.BREAK if self.is_word(t, "break") else SK.CONTINUE, self.ops_end(), t.line)
            except e:
                raise self.relocate(String(e), t)
        elif self.is_word(t, "box"):
            self.forbid_in_gate(t)
            self.parse_box()
        elif self.is_word(t, "switch"):
            self.forbid_in_gate(t)
            self.parse_switch()
        elif (self.is_word(t, "cal") or self.is_word(t, "defcal")
              or self.is_word(t, "defcalgrammar") or self.is_word(t, "let")
              or self.is_word(t, "return") or self.is_word(t, "break") or self.is_word(t, "continue")
              or self.is_word(t, "end") or self.is_word(t, "array") or self.is_word(t, "duration")
              or self.is_word(t, "stretch") or self.is_word(t, "complex")):
            raise self.err("QM-E504", "OpenQASM feature '" + self.text(t) + "' is not supported by QUMO" + (" outside a loop" if self.is_word(t, "break") or self.is_word(t, "continue") else ""), t, "a Circuit takes declarations, gates, modifiers, measure, reset, barrier, box, if on bits or a register, switch and constant for loops; parse_qasm_program or load_any add while, break, continue and classical variables")
        else:
            # assignment `c = measure q;` / `c[0] = measure q[0];`
            var nm = self.text(t)
            var k = self.find_name(nm) if self.program_mode else -1
            var ri = self.find_register(nm)
            if ri and not self.regs[ri.value()].is_qubit:
                self.forbid_in_gate(t)
                self.parse_measure_assign()
            elif ri:
                raise self.err("QM-E500", "statement starts with qubit register '" + nm + "'", t, "a gate name is missing, e.g. 'h " + nm + "[0];'")
            elif k >= 0 and self.names[k].kind == NK.VAR:
                self.forbid_in_gate(t)
                self.parse_var_assign()
            elif k >= 0 or self.is_loop_name(nm):
                raise self.err("QM-E504", "assignment to '" + nm + "', a constant here", t, "drop 'const', or assign to a variable; a loop variable that changes needs parse_qasm_program or load_any")
            elif nm in self.int_vars:
                self.parse_int_assign()
            elif nm in self.consts or nm in self.inputs:
                raise self.err("QM-E504", "assignment to classical variable '" + nm + "' is not supported", t, "a const or an input cannot change; other variables need parse_qasm_program or load_any")
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
        if is_qubit and self.block_depth > 0:
            raise self.err("QM-E504", "qubits are declared at global scope", kw, "move the declaration out of the block")
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
        var init_value = False
        if self.is_punct(self.peek(), P.EQ):
            if not is_qubit and self.is_word(self.peek(1), "measure"):
                init_measure = True  # `bit[2] c = measure q;`
            elif not is_qubit and self.program_mode:
                init_value = True
            else:
                raise self.err("QM-E504", "initialized registers are only supported as 'bit[n] c = measure q;'", self.peek(), "declare 'bit[n] c;' (bits start at 0) and measure into it, or parse with parse_qasm_program or load_any")
        if not init_measure and not init_value:
            _ = self.expect_punct(P.SEMI, "';'")
        var nm = self.text(nt)
        var scoped = self.program_mode and not is_qubit and len(self.scope_marks) > 0
        var known = self.scoped_regs.get(kw.start) if scoped else None
        var offset: Int
        if known:
            offset = self.regs[known.value()].offset  # the same declaration, on a later pass of an unrolled loop
        elif is_qubit:
            offset = self.circs[0].num_qubits
            for ref c in self.circs:
                c.num_qubits = offset + size
        else:
            offset = self.circs[0].num_clbits
            for ref c in self.circs:
                c.num_clbits = offset + size
        if scoped:
            if not known:
                self.scoped_regs[kw.start] = len(self.regs)
                self.regs.append(Register(nm, offset, size, False))
            self.names.append(_Name(nm, NK.REG, self.scoped_regs[kw.start]))
            if not init_measure and not init_value:
                # a block's bit starts at 0 each time it is entered
                self.assign_bits(offset, size, nm, CExpr.literal_int(0), kw)
        else:
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
        if init_value:
            var eq = self.next()
            var e = self.parse_cexpr()
            _ = self.expect_punct(P.SEMI, "';'")
            self.assign_bits(offset, size, nm, e, eq)

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
        var ty = float_type()
        var typed = False
        var tt = self.peek()
        if self.program_mode and not (self.is_word(tt, "float") or self.is_word(tt, "angle")):
            ty = self.parse_ctype(False)
            typed = True
        else:
            tt = self.skip_type()
            if not (self.is_word(tt, "float") or self.is_word(tt, "angle")):
                raise self.err("QM-E504", "only 'input float' / 'input angle' parameters are supported, got '" + self.text(tt) + "'", tt, "declare symbolic angles as 'input float theta;', or parse with parse_qasm_program or load_any")
        var nt = self.expect_ident("parameter name")
        self.check_new_name(nt)
        _ = self.expect_punct(P.SEMI, "';'")
        var nm = self.text(nt)
        var p = self.circs[0].parameter(nm)
        for i in range(1, len(self.circs)):
            _ = self.circs[i].parameter(nm)
        self.inputs[nm] = p.index
        if typed:
            self.input_types[nm] = ty
        if self.program_mode:
            self.prog.user_params.append(p.index)

    def parse_classical_decl(mut self, is_const: Bool) raises:
        var first = self.peek()
        if self.is_word(first, "bit") or self.is_word(first, "qubit"):
            raise self.err("QM-E504", "'const' registers are not supported", first, "declare 'bit[n] c;' without const")
        # a type the classical table covers decides the value, as in a program;
        # a designator it refuses (float[32]) keeps the legacy untyped constant
        var ty = float_type()
        var typed = True
        var save = self.pos
        try:
            ty = self.parse_ctype(False)
        except:
            typed = False
            self.pos = save
        var tt = first
        if not typed:
            tt = self.skip_type()
        var nt = self.expect_ident("variable name")
        self.check_new_name(nt)
        if not self.is_punct(self.peek(), P.EQ):
            if (self.is_word(tt, "int") or self.is_word(tt, "uint")) and not is_const and self.is_punct(self.peek(), P.SEMI):
                # Qiskit's `int switch_dummy; switch_dummy = c; switch (switch_dummy)`
                _ = self.next()
                self.int_vars[self.text(nt)] = List[Int]()
                return
            raise self.err("QM-E504", "classical variables need an initializer", self.peek(), "write 'const float x = 0.5;' (QUMO folds classical values at parse time)")
        var eq = self.next()
        var nm = self.text(nt)
        if typed:
            var start = self.pos
            var ok = False
            var value = CExpr()
            try:
                var e = self.parse_cexpr()
                if self.is_punct(self.peek(), P.SEMI):
                    value = ce_cast(e, ty)
                    ok = value.is_const()
            except err:
                var msg = String(err)
                if msg.startswith("QM-E580") or msg.startswith("QM-E533") or msg.startswith("QM-E482"):
                    raise self.relocate(msg, eq)
            if ok:
                _ = self.next()
                self.const_exprs[nm] = value.copy()
                self.consts[nm] = Param(_const_radians(value))
                return
            self.pos = start  # depends on an input: an affine Param, below
        var v = self.parse_expr()
        _ = self.expect_punct(P.SEMI, "';'")
        if self.is_word(tt, "int") or self.is_word(tt, "uint"):
            if not v.is_symbolic():
                v = Param(Float64(Int(v.offset)))
        _ = is_const
        self.consts[nm] = v

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
        if self.program_mode and (self.is_punct(eq, P.OPASSIGN) or (self.is_punct(eq, P.EQ) and not self.is_word(self.peek(1), "measure"))):
            self.parse_bit_assign(ct, cs)
            return
        if self.is_punct(eq, P.OPASSIGN) or not self.is_punct(eq, P.EQ):
            raise self.err("QM-E500", "expected '= measure ...' after bit operand, found " + self.describe(eq), eq, "write 'c[0] = measure q[0];'")
        _ = self.next()
        var mt = self.peek()
        if not self.is_word(mt, "measure"):
            raise self.err("QM-E504", "only measurement results can be assigned to bits", mt, "write 'c[0] = measure q[0];', or parse with parse_qasm_program or load_any")
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

    def parse_bit_test(mut self) raises -> Tuple[Int, Int]:
        """One `c[i]`, `!c[i]`, `c[i] == v` or `c[i] != v` of an `&&` chain."""
        var neg = False
        if self.is_punct(self.peek(), P.BANG):
            _ = self.next()
            neg = True
        var bt = self.peek()
        var bits = self.parse_bit_operand()
        if len(bits) != 1:
            raise self.err("QM-E541", "an '&&' condition tests single bits, not a " + String(len(bits)) + "-bit register", bt, "write if (c[0] == 1 && c[1] == 0)")
        var val = 1
        var ct = self.peek()
        if self.is_punct(ct, P.EQEQ) or self.is_punct(ct, P.NEQ):
            _ = self.next()
            var vt = self.peek()
            val = self.parse_const_int("condition value")
            if val != 0 and val != 1:
                raise self.err("QM-E541", "a single bit can only be compared with 0 or 1", vt, "write if (c[0] == 1)")
            if self.is_punct(ct, P.NEQ):
                val = 1 - val
        if neg:
            val = 1 - val
        return (bits[0], val)

    def parse_condition(mut self) raises -> Tuple[Int, Int]:
        """`(c[0])`, `(!c[0])`, `(c[0] == 1)`, whole-register `(c == N)` (OpenQASM
        2 style, N of any size), and `(c[0] == 1 && c[1] == 0)`, which is how
        to_qasm writes a condition on part of the classical bits. Register and
        `&&` conditions are recorded in self.cond_reg_bits / cond_reg_vals; the
        returned pair is the first bit."""
        var lp = self.expect_punct(P.LPAREN, "'(' after if")
        self.cond_reg_bits = List[Int]()
        self.cond_reg_vals = List[Int]()
        var negate = False
        if self.is_punct(self.peek(), P.BANG):
            _ = self.next()
            negate = True
        var bt = self.peek()
        if bt.kind != TK.IDENT or self.text(bt) in self.consts or self.text(bt) in self.int_vars or self.text(bt) in self.inputs or self.is_word(bt, "true") or self.is_word(bt, "false"):
            raise self.err("QM-E541", "a Circuit condition tests bits or a register", bt, "parse with parse_qasm_program or load_any for conditions on values")
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
                if self.is_punct(self.peek(), P.ANDAND) or self.is_punct(self.peek(), P.OROR):
                    raise self.err("QM-E541", "a whole-register test cannot be combined with other tests", self.peek(), "spell the bits out: if (c[0] == 1 && c[1] == 0)")
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
        if negate:
            val = 1 - val
        if self.is_punct(self.peek(), P.ANDAND):
            self.cond_reg_bits.append(bits[0])
            self.cond_reg_vals.append(val)
            while self.is_punct(self.peek(), P.ANDAND):
                _ = self.next()
                var test = self.parse_bit_test()
                self.cond_reg_bits.append(test[0])
                self.cond_reg_vals.append(test[1])
        if self.is_punct(self.peek(), P.OROR):
            raise self.err("QM-E541", "'||' conditions are not supported", self.peek(), "nest the operations or test one bit per if")
        _ = self.expect_punct(P.RPAREN, "')' to close the condition")
        _ = lp
        return (bits[0], val)

    def check_condition_kept(self, tested: List[Int], first: Int, t: Token) raises:
        """Conditions are evaluated gate by gate, so a write to a tested bit
        changes what the conditioned ops after it see. A write as the very last
        conditioned op is harmless; anything later would give a wrong answer."""
        ref ops = self.circs[0].ops
        var written = False
        for i in range(first, len(ops)):
            if written and ops[i].cond >= 0:
                raise self.err("QM-E541", "this 'if' writes a bit it tests, and the operations after that write would see the new value", t, "measure into another bit, move the write to the end of the body, or use parse_qasm_program or load_any, where the test runs once")
            if ops[i].kind == Op.MEASURE:
                for b in tested:
                    if ops[i].clbit == b:
                        written = True

    def parse_if(mut self) raises:
        var it = self.next()
        if self.program_mode:
            self.parse_if_region(it)
            return
        if self.cond_bit >= 0:
            raise self.err("QM-E541", "nested if statements are not supported", it, "QUMO instructions carry one condition")
        var cv = self.parse_condition()
        self.cond_bit = cv[0]
        self.cond_val = cv[1]
        self.cond_reg = -1
        var tested: List[Int] = [cv[0]]
        if len(self.cond_reg_bits) > 0:
            tested = self.cond_reg_bits.copy()
            ref c0 = self.circs[0]
            self.cond_reg = len(c0.cond_table)
            c0.cond_table.append(len(self.cond_reg_bits))
            for i in range(len(self.cond_reg_bits)):
                c0.cond_table.append(self.cond_reg_bits[i])
                c0.cond_table.append(self.cond_reg_vals[i])
        var first = len(self.circs[0].ops)
        if self.pragma_if:
            self.pragma_if = False
            self.parse_pragma_body(it)
        else:
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
        self.check_condition_kept(tested, first, it)

    def parse_int_assign(mut self) raises:
        """`x = c;` for an `int x;`: x mirrors the bits of c, for a later switch."""
        var t = self.next()
        if self.cond_bit >= 0:
            raise self.err("QM-E541", "assignment to '" + self.text(t) + "' under a condition is not supported", t, "assign it before the if")
        _ = self.expect_punct(P.EQ, "'=' in assignment")
        var rt = self.peek()
        var nm = self.text(rt)
        var ri = self.reg_index.get(nm)
        if rt.kind != TK.IDENT or not ri or self.regs[ri.value()].is_qubit:
            raise self.err("QM-E504", "an integer variable can only mirror a bit register", rt, "write '" + self.text(t) + " = c;' and use it as the target of a switch")
        self.int_vars[self.text(t)] = self.parse_bit_operand()
        self.int_var_at[self.text(t)] = len(self.circs[0].ops)
        _ = self.expect_punct(P.SEMI, "';'")

    def parse_box(mut self) raises:
        """`box [dur] { ... }`: gates cannot leave or enter, so a barrier on each side."""
        var bt = self.next()
        if self.is_punct(self.peek(), P.LBRACK):
            self.skip_balanced(P.LBRACK, P.RBRACK, self.peek())
            self.warn("QM-W053", "box duration is ignored (QUMO circuits have no timing model)", bt, "the box still keeps its gates together")
        if not self.is_punct(self.peek(), P.LBRACE):
            raise self.err("QM-E500", "expected '{' after box, found " + self.describe(self.peek()), self.peek(), "write 'box { ... }'")
        self.circs[self.top()].ops.append(Instruction.make(Op.BARRIER, -1))
        self.parse_statement()
        self.circs[self.top()].ops.append(Instruction.make(Op.BARRIER, -1))

    def parse_conditioned_body(mut self, bits: List[Int], vals: List[Int], body: Int) raises:
        """Parse the block at token `body` with every gate conditioned on bits == vals."""
        if len(bits) > 0:
            self.cond_bit = bits[0]
            self.cond_val = vals[0]
            self.cond_reg = -1
            if len(bits) > 1:
                ref c0 = self.circs[0]
                self.cond_reg = len(c0.cond_table)
                c0.cond_table.append(len(bits))
                for i in range(len(bits)):
                    c0.cond_table.append(bits[i])
                    c0.cond_table.append(vals[i])
        var t = self.toks[body]
        var first = len(self.circs[0].ops)
        self.pos = body
        self.parse_statement()
        self.cond_bit = -1
        self.cond_val = 1
        self.cond_reg = -1
        # conditions are evaluated gate by gate, so the body must not change them
        for i in range(first, len(self.circs[0].ops)):
            var g = self.circs[0].ops[i]
            if g.kind == Op.MEASURE:
                for b in bits:
                    if g.clbit == b:
                        raise self.err("QM-E541", "a switch body cannot measure into the bits it switches on", t, "measure into another register")

    def parse_switch(mut self) raises:
        """`switch (c) { case 0 { ... } case 1, 2 { ... } default { ... } }`, also
        Qiskit's older `case 1: case 2: { ... } break;`. The target is a bit
        register (register conditions, one body copy per value), an `int` that
        mirrors one, or a constant (the matching body only)."""
        var st = self.next()
        if self.program_mode:
            self.parse_switch_region(st)
            return
        if self.cond_bit >= 0:
            raise self.err("QM-E541", "nested conditions are not supported", st, "QUMO instructions carry one condition")
        _ = self.expect_punct(P.LPAREN, "'(' after switch")
        var t = self.peek()
        if (self.is_word(t, "int") or self.is_word(t, "uint")) and not (self.text(t) in self.consts):
            _ = self.skip_type()
            _ = self.expect_punct(P.LPAREN, "'(' after the cast")
            t = self.peek()
        var bits = List[Int]()
        var is_const = False
        var cval = 0
        var nm = self.text(t)
        var ri = self.reg_index.get(nm)
        if nm in self.int_vars:
            _ = self.next()
            bits = self.int_vars[nm].copy()
            if len(bits) == 0:
                raise self.err("QM-E541", "switch on '" + nm + "' before it is assigned", t, "assign it first: '" + nm + " = c;'")
            # `x = c` records no value, x reads c's bits at the switch: refuse
            # the switch if they may differ from the copy the program asked for
            ref ops = self.circs[0].ops
            for i in range(self.int_var_at[nm], len(ops)):
                if ops[i].kind == Op.MEASURE:
                    for b in bits:
                        if ops[i].clbit == b:
                            raise self.err("QM-E541", "'" + nm + "' was assigned from bits that were measured again before this switch", t, "switch right after the assignment, or on the register itself; parse_qasm_program and load_any keep a real copy")
        elif ri and not self.regs[ri.value()].is_qubit:
            bits = self.parse_bit_operand()
        else:
            cval = self.parse_const_int("switch target")
            is_const = True
        while self.is_punct(self.peek(), P.RPAREN):
            _ = self.next()
        var lb = self.expect_punct(P.LBRACE, "'{' to open the switch")
        var values = List[List[Int]]()  # values of each case
        var bodies = List[Int]()
        var default_body = -1
        while not self.is_punct(self.peek(), P.RBRACE):
            var c = self.peek()
            if c.kind == TK.EOF:
                raise self.err("QM-E500", "missing '}' to close the switch", lb, "every '{' needs a matching '}'")
            var vs = List[Int]()
            var is_default = False
            while self.is_word(self.peek(), "case") or self.is_word(self.peek(), "default"):
                if self.is_word(self.peek(), "default"):
                    _ = self.next()
                    is_default = True
                else:
                    _ = self.next()
                    vs.append(self.parse_const_int("case value"))
                    while self.is_punct(self.peek(), P.COMMA):
                        _ = self.next()
                        vs.append(self.parse_const_int("case value"))
                if self.is_punct(self.peek(), P.COLON):
                    _ = self.next()
            if len(vs) == 0 and not is_default:
                raise self.err("QM-E500", "expected 'case' or 'default', found " + self.describe(c), c, "write 'case 0 { ... }' or 'default { ... }'")
            if not self.is_punct(self.peek(), P.LBRACE):
                raise self.err("QM-E500", "expected '{' to open the case, found " + self.describe(self.peek()), self.peek(), "write 'case 0 { ... }'")
            var body = self.pos
            self.skip_balanced(P.LBRACE, P.RBRACE, self.peek())
            if self.is_word(self.peek(), "break"):
                _ = self.next()
                _ = self.expect_punct(P.SEMI, "';'")
            if is_default:
                if default_body >= 0:
                    raise self.err("QM-E541", "a switch has one default", c, "merge the two bodies")
                default_body = body
            if len(vs) > 0:
                values.append(vs^)
                bodies.append(body)
        var after = self.pos + 1
        # cases are mutually exclusive by definition; catch duplicates early
        var seen = List[Int]()
        for vs in values:
            for v in vs:
                for u in seen:
                    if u == v:
                        raise self.err("QM-E541", "case value " + String(v) + " appears twice", st, "each value belongs to one case")
                seen.append(v)
        if is_const:
            var chosen = default_body
            for i in range(len(values)):
                for v in values[i]:
                    if v == cval:
                        chosen = bodies[i]
            if chosen >= 0:
                self.parse_conditioned_body(List[Int](), List[Int](), chosen)
            self.pos = after
            return
        var covered = List[List[Int]]()
        for i in range(len(values)):
            for v in values[i]:
                var vb = decimal_to_bits(String(v), len(bits)) if v >= 0 else List[Int]()
                if len(vb) == 0:
                    raise self.err("QM-E541", "case value " + String(v) + " does not fit the " + String(len(bits)) + "-bit target", st, "use a value in 0 .. 2^" + String(len(bits)) + " - 1")
                covered.append(vb.copy())
                self.parse_conditioned_body(bits, vb, bodies[i])
        if default_body >= 0:
            # the complement of the cases as a few partial-bit conditions
            var cubes = List[List[Int]]()
            var cube = List[Int](length=len(bits), fill=-1)
            _switch_cover(covered, len(bits) - 1, cube, cubes)
            for ref q in cubes:
                var bs = List[Int]()
                var vs = List[Int]()
                for i in range(len(bits)):
                    if q[i] >= 0:
                        bs.append(bits[i])
                        vs.append(q[i])
                self.parse_conditioned_body(bs, vs, default_body)
        self.pos = after

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
            self.skip_balanced(P.LPAREN, P.RPAREN, self.peek())
            self.skip_statement()
            if self.is_word(self.peek(), "else"):
                _ = self.next()
                self.skip_statement()
            return
        if self.is_word(t, "while") or self.is_word(t, "switch"):
            _ = self.next()
            self.skip_balanced(P.LPAREN, P.RPAREN, self.peek())
            self.skip_statement()
            return
        if self.is_word(t, "box"):
            _ = self.next()
            if self.is_punct(self.peek(), P.LBRACK):
                self.skip_balanced(P.LBRACK, P.RBRACK, self.peek())
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
        var loop_ty = int_type(64, True)
        var typed = self.depth == 0
        if not self.is_word(self.peek(1), "in"):
            if self.depth == 0:
                var save = self.pos
                try:
                    loop_ty = self.parse_ctype(True)
                except:
                    self.pos = save
                    _ = self.skip_type()
                    typed = False
            else:
                _ = self.skip_type()
        var vt = self.expect_ident("loop variable")
        var it = self.peek()
        if not self.is_word(it, "in"):
            raise self.err("QM-E500", "expected 'in' in for loop, found " + self.describe(it), it, "write 'for uint i in [0:3] { ... }' (ranges are inclusive)")
        _ = self.next()
        if self.program_mode and self.depth == 0 and not (typed and (loop_ty.tag == CT.INT or loop_ty.tag == CT.UINT)):
            raise self.err("QM-E580", "a loop counts in int or uint, not " + type_name(loop_ty), vt, "write 'for int i in ...'")
        typed = typed and (loop_ty.tag == CT.INT or loop_ty.tag == CT.UINT or loop_ty.tag == CT.FLOAT)
        if self.program_mode and self.depth == 0 and self.for_needs_region(loop_ty, self.text(vt)):
            self.parse_for_region(loop_ty, vt)
            return
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
                if len(values) > _MAX_UNROLL:
                    raise self.err("QM-E570", "for loop has more than " + String(_MAX_UNROLL) + " iterations", rt, "QUMO unrolls loops at parse time; reduce the range")
                if _last_step(i, step, b):
                    break
                i += step
        elif self.is_punct(rt, P.LBRACE):
            _ = self.next()
            values.append(self.parse_const_int("set element"))
            while self.is_punct(self.peek(), P.COMMA):
                _ = self.next()
                values.append(self.parse_const_int("set element"))
            _ = self.expect_punct(P.RBRACE, "'}' to close the set")
        else:
            raise self.err("QM-E570", "for loops need a constant range [a:b] or set {a, b}", rt, "write 'for uint i in [0:3] { ... }' (ranges are inclusive)")
        if not self.program_mode and typed and self.depth == 0:
            for v in values:
                if not _value_fits(v, loop_ty):
                    raise self.err("QM-E504", "value " + String(v) + " does not fit the loop variable's type", rt, "the variable wraps while the loop counts: parse with parse_qasm_program or load_any")
        self.unrolled += len(values)
        if self.unrolled > _MAX_UNROLL:
            raise self.err("QM-E570", "too many unrolled loop iterations", ft, "QUMO unrolls loops at parse time; keep the total below " + String(_MAX_UNROLL))
        var body = self.pos
        if len(values) == 0:
            self.skip_statement()
            return
        var nm = self.text(vt)
        var named = self.depth == 0
        if named:
            self.loop_names.append(_LoopName(nm, self.block_depth + (1 if self.is_punct(self.peek(), P.LBRACE) else 0)))
        for v in values:
            self.pos = body
            if self.program_mode and self.depth == 0:
                self.push_scope()
                self.names.append(_Name(nm, NK.CONST, len(self.const_vals)))
                self.const_vals.append(_loop_value(v, loop_ty))
                self.parse_statement()
                self.pop_scope()
                continue
            var sc = Scope(False)
            sc.names.append(nm)
            sc.values.append(Param(Float64(v)))
            if named and typed:
                var value = _loop_value(v, loop_ty)
                sc.values[0] = Param(_const_radians(value))
                sc.exprs.append(value^)
            self.scopes.append(sc^)
            self.parse_statement()
            _ = self.scopes.pop()
        if named:
            _ = self.loop_names.pop()

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

    def parse_pragma_constants(mut self, what: String) raises -> List[Float64]:
        """`(a, b, ...)` of constant expressions after a pragma word."""
        var out = List[Float64]()
        _ = self.expect_punct(P.LPAREN, "'(' after " + what)
        while True:
            var vt = self.peek()
            var p = self.parse_expr()
            out.append(self.need_const(p, vt, what + " value"))
            if self.is_punct(self.peek(), P.COMMA):
                _ = self.next()
                continue
            break
        _ = self.expect_punct(P.RPAREN, "')'")
        return out^

    def parse_pragma(mut self) raises:
        var pt = self.next()
        if self.is_word(self.peek(), "if"):
            # `#pragma qumo if (...) <pragma body>`: the form to_qasm writes
            self.pragma_if = True
            self.parse_if()
            return
        self.parse_pragma_body(pt)

    def parse_pragma_body(mut self, pt: Token) raises:
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
        if self.is_word(t, "pauli_channel_2"):
            self.forbid_in_gate(t)
            _ = self.next()
            var probs = self.parse_pragma_constants("pauli_channel_2")
            if len(probs) != 15:
                raise self.err("QM-E560", "pauli_channel_2 takes 15 probabilities, got " + String(len(probs)), t, "IX, IY, IZ, XI, ... ZZ, with the left Pauli on the first qubit")
            var pa = self.parse_qubit_operand()
            _ = self.expect_punct(P.COMMA, "',' between the two qubits")
            var pb = self.parse_qubit_operand()
            _ = self.expect_punct(P.SEMI, "end of pragma line")
            if len(pa) != 1 or len(pb) != 1:
                raise self.err("QM-E560", "pauli_channel_2 takes two single qubits", t, "write q[0], q[1]")
            self.circs[self.top()].pauli_channel_2(probs, pa[0], pb[0])
            self.condition_last()
            return
        if self.is_word(t, "correlated_error") or self.is_word(t, "else_correlated_error"):
            self.forbid_in_gate(t)
            var is_else = self.is_word(t, "else_correlated_error")
            _ = self.next()
            var ep = self.parse_pragma_constants("correlated_error")
            if len(ep) != 1:
                raise self.err("QM-E560", "correlated_error takes one probability", t, "write '#pragma qumo correlated_error(0.01) x q[0], z q[1];'")
            var eq = List[Int]()
            var el = List[Int]()
            while True:
                var lt = self.peek()
                var letter = 1 if self.is_word(lt, "x") else (2 if self.is_word(lt, "y") else (3 if self.is_word(lt, "z") else 0))
                if letter == 0:
                    raise self.err("QM-E560", "expected a Pauli letter x, y or z", lt, "write '#pragma qumo correlated_error(0.01) x q[0], z q[1];'")
                _ = self.next()
                var ot = self.parse_qubit_operand()
                if len(ot) != 1:
                    raise self.err("QM-E560", "each Pauli takes one qubit", lt, "write 'x q[0]', not a whole register")
                eq.append(ot[0])
                el.append(letter)
                if self.is_punct(self.peek(), P.COMMA):
                    _ = self.next()
                    continue
                break
            _ = self.expect_punct(P.SEMI, "end of pragma line")
            if is_else and self.cond_bit >= 0:
                # an ELSE reads whether its chain fired: it must run under the
                # same condition as the error before it, or the flag is stale
                ref ops = self.circs[0].ops
                var n = len(ops)
                if n == 0 or ops[n - 1].kind != Op.CORRELATED_ERROR or ops[n - 1].cond != self.cond_bit or ops[n - 1].cond_val != self.cond_val or ops[n - 1].cond_reg != self.cond_reg:
                    raise self.err("QM-E541", "else_correlated_error under a condition must follow an error under the same condition", t, "put the whole chain in the same if")
            self.circs[self.top()].correlated_error(ep[0], eq, el, is_else)
            self.condition_last()
            return
        if self.is_word(t, "kraus1") or self.is_word(t, "kraus2"):
            self.forbid_in_gate(t)
            var knq = 1 if self.is_word(t, "kraus1") else 2
            _ = self.next()
            var kn = self.peek()
            if kn.kind != TK.STRING:
                raise self.err("QM-E560", "expected the channel name as a string", kn, "write '#pragma qumo kraus1 \"damp\" (re, im, ...) q[0];'")
            _ = self.next()
            var kv = self.parse_pragma_constants("kraus")
            var dd = (1 << knq) * (1 << knq)
            if len(kv) == 0 or len(kv) % (2 * dd) != 0:
                raise self.err("QM-E560", "kraus needs (re, im) pairs for whole " + String(1 << knq) + "x" + String(1 << knq) + " matrices", t, "matrices are row-major, one after the other")
            var ch = KrausChannel(self.text(kn), knq)
            var at = 0
            while at < len(kv):
                var re = List[Float64](capacity=dd)
                var im = List[Float64](capacity=dd)
                for j in range(dd):
                    re.append(kv[at + 2 * j])
                    im.append(kv[at + 2 * j + 1])
                ch.add(re^, im^)
                at += 2 * dd
            var ka = self.parse_qubit_operand()
            var kb = List[Int]()
            if knq == 2:
                _ = self.expect_punct(P.COMMA, "',' between the two qubits")
                kb = self.parse_qubit_operand()
            _ = self.expect_punct(P.SEMI, "end of pragma line")
            if len(ka) != 1 or len(kb) != knq - 1:
                raise self.err("QM-E560", "kraus takes single qubits", t, "write q[0] or q[0], q[1]")
            self.circs[self.top()].apply_channel(ch, ka[0], kb[0] if knq == 2 else -1)
            self.condition_last()
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
                var classical = self.program_mode or len(self.const_exprs) > 0 or len(self.scopes) > 0
                ps.append(self.parse_gate_arg() if classical else self.parse_expr())
                while self.is_punct(self.peek(), P.COMMA):
                    _ = self.next()
                    ps.append(self.parse_gate_arg() if classical else self.parse_expr())
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
                    var before = sub.copy()
                    var fallback = False
                    try:
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
                    except e:
                        if not String(e).startswith("QM-E202"):
                            raise e^
                        fallback = True
                    if fallback:
                        # past what Circuit.controlled covers: full synthesis,
                        # borrowing every qubit outside this gate as dirty
                        var ctrls = List[Int]()
                        for j in range(ci - k, ci):
                            ctrls.append(qs[j])
                        var mc = controlled_by(before, ctrls, self.idle_qubits(qs))
                        if kind == 3:
                            var w = before.empty_like()
                            w.num_qubits = mc.num_qubits
                            w.global_phase = 0.0
                            for q in ctrls:
                                w.x(q)
                            w.compose(mc)
                            for q in ctrls:
                                w.x(q)
                            sub = w^
                        else:
                            sub = mc^
                    ci -= k
            except e:
                raise self.relocate(String(e), mtoks[m])
            m -= 1
        self.compose_sub(sub)

    def idle_qubits(self, qs: List[Int]) -> List[Int]:
        """Declared qubits other than `qs` (available as dirty ancillas)."""
        var out = List[Int]()
        for q in range(self.circs[0].num_qubits):
            var used = False
            for u in qs:
                used = used or u == q
            if not used:
                out.append(q)
        return out^

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

    # ------------------------------------------------------------ program mode
    def ops_end(self) -> Int:
        return len(self.circs[0].ops)

    def push_scope(mut self):
        if self.program_mode:
            self.scope_marks.append(len(self.names))

    def pop_scope(mut self):
        if not self.program_mode:
            return
        var m = self.scope_marks.pop()
        while len(self.names) > m:
            _ = self.names.pop()

    def find_name(self, nm: String) -> Int:
        """Index in `names` of the innermost declaration of `nm`, or -1."""
        var i = len(self.names) - 1
        while i >= 0:
            if self.names[i].name == nm:
                return i
            i -= 1
        return -1

    def find_var(self, nm: String) -> Int:
        """Id of the variable `nm` names here, or -1."""
        var k = self.find_name(nm)
        if k >= 0 and self.names[k].kind == NK.VAR:
            return self.names[k].index
        return -1

    def find_register(self, nm: String) -> Optional[Int]:
        """The register `nm` names here: a name of a program scope hides the
        registers of the global scope."""
        if self.program_mode and self.depth == 0:
            var k = self.find_name(nm)
            if k >= 0:
                if self.names[k].kind == NK.REG:
                    return self.names[k].index
                return None
        return self.reg_index.get(nm)

    def is_loop_name(self, nm: String) -> Bool:
        for l in self.loop_names:
            if l.name == nm:
                return True
        return False

    def declare_var(mut self, nm: String, ty: CType, is_output: Bool) -> Int:
        var v = self.prog.add_var(nm, ty, is_output)
        self.names.append(_Name(nm, NK.VAR, v))
        return v

    def prescan(mut self):
        """Names assigned after their declaration. A declaration with a constant
        initializer that is never reassigned still folds, so `int i = 2; h q[i];`
        parses as it does for a Circuit."""
        for i in range(1, len(self.toks) - 1):
            var t = self.toks[i]
            var n = self.toks[i + 1]
            if t.kind != TK.IDENT or not (self.is_punct(n, P.EQ) or self.is_punct(n, P.OPASSIGN)):
                continue
            var prev = self.toks[i - 1]
            var declared = prev.kind == TK.IDENT and not self.is_word(prev, "else")
            if self.is_punct(prev, P.RBRACK):
                # `int[8] x = 1;` declares; `for int i in [0:2] x = 1;` does not
                var k = i - 2
                var open = 1
                while k > 0 and open > 0:
                    if self.is_punct(self.toks[k], P.RBRACK):
                        open += 1
                    elif self.is_punct(self.toks[k], P.LBRACK):
                        open -= 1
                    k -= 1
                declared = not self.is_word(self.toks[k], "in")
            if not declared:
                self.reassigned[self.text(t)] = True

    def parse_ctype(mut self, allow_bare_angle: Bool) raises -> CType:
        """A classical type with its designator (QM-E582 when not representable)."""
        var tt = self.expect_ident("a type")
        var nm = self.text(tt)
        var has = False
        var width = 0
        var wt = self.peek()
        if self.is_punct(wt, P.LBRACK):
            _ = self.next()
            wt = self.peek()
            width = self.parse_const_int("type width")
            _ = self.expect_punct(P.RBRACK, "']'")
            has = True
        if nm == "bool":
            if has:
                raise self.err("QM-E582", "bool takes no width", wt, "write 'bool b;'")
            return bool_type()
        if nm == "float":
            if has and width != 64:
                raise self.err("QM-E582", "float[" + String(width) + "] is not supported: QUMO floats are binary64", wt, "write 'float x;' or 'float[64] x;'")
            return float_type(not has)
        if nm != "int" and nm != "uint" and nm != "angle" and nm != "bit":
            raise self.err("QM-E504", "type '" + nm + "' is not supported in a Program", tt, "use bit, bool, int, uint, float or angle[n]")
        if has and (width < 1 or width > 64):
            raise self.err("QM-E582", "width " + String(width) + " is outside 1..64", wt, "use a width from 1 to 64")
        if nm == "angle":
            if not has and not allow_bare_angle:
                raise self.err("QM-E582", "a mutable angle needs a width", tt, "write 'angle[20] a;', or 'const angle a = pi/2;'")
            return angle_type(width if has else 64)
        if nm == "bit":
            return bit_type(width if has else 1)
        if nm == "int":
            return int_type(width if has else 64, not has)
        return uint_type(width if has else 64, not has)

    def rel(mut self, op: Int, a: CExpr, b: CExpr, t: Token) raises -> CExpr:
        try:
            return ce_binary(op, a, b)
        except e:
            raise self.relocate(String(e), t)

    def cast_at(self, e: CExpr, ty: CType, t: Token) raises -> CExpr:
        try:
            return ce_cast(e, ty)
        except err:
            raise self.relocate(String(err), t)

    def parse_cexpr(mut self) raises -> CExpr:
        """An expression evaluated at run time, in OpenQASM precedence; typing and
        casts come from qumo.ir.classical. `parse_expr` remains the constant path."""
        var a = self.parse_c_and()
        while self.is_punct(self.peek(), P.OROR):
            var t = self.next()
            var b = self.parse_c_and()
            try:
                a = ce_logical(False, a, b)
            except e:
                raise self.relocate(String(e), t)
        return a^

    def parse_c_and(mut self) raises -> CExpr:
        var a = self.parse_c_bor()
        while self.is_punct(self.peek(), P.ANDAND):
            var t = self.next()
            var b = self.parse_c_bor()
            try:
                a = ce_logical(True, a, b)
            except e:
                raise self.relocate(String(e), t)
        return a^

    def parse_c_bor(mut self) raises -> CExpr:
        var a = self.parse_c_xor()
        while self.is_punct(self.peek(), P.PIPE):
            var t = self.next()
            a = self.rel(CN.OR, a, self.parse_c_xor(), t)
        return a^

    def parse_c_xor(mut self) raises -> CExpr:
        var a = self.parse_c_band()
        while self.is_punct(self.peek(), P.CARET):
            var t = self.next()
            a = self.rel(CN.XOR, a, self.parse_c_band(), t)
        return a^

    def parse_c_band(mut self) raises -> CExpr:
        var a = self.parse_c_eq()
        while self.is_punct(self.peek(), P.AMP):
            var t = self.next()
            a = self.rel(CN.AND, a, self.parse_c_eq(), t)
        return a^

    def parse_c_eq(mut self) raises -> CExpr:
        var a = self.parse_c_rel()
        while True:
            var t = self.peek()
            if self.is_punct(t, P.EQEQ):
                _ = self.next()
                a = self.rel(CN.EQ, a, self.parse_c_rel(), t)
            elif self.is_punct(t, P.NEQ):
                _ = self.next()
                a = self.rel(CN.NE, a, self.parse_c_rel(), t)
            else:
                return a^

    def parse_c_rel(mut self) raises -> CExpr:
        var a = self.parse_c_shift()
        while True:
            var t = self.peek()
            var op = -1
            if self.is_punct(t, P.LT): op = CN.LT
            elif self.is_punct(t, P.LE): op = CN.LE
            elif self.is_punct(t, P.GT): op = CN.GT
            elif self.is_punct(t, P.GE): op = CN.GE
            if op < 0:
                return a^
            _ = self.next()
            a = self.rel(op, a, self.parse_c_shift(), t)

    def parse_c_shift(mut self) raises -> CExpr:
        var a = self.parse_c_add()
        while True:
            var t = self.peek()
            if self.is_punct(t, P.SHL):
                _ = self.next()
                a = self.rel(CN.SHL, a, self.parse_c_add(), t)
            elif self.is_punct(t, P.SHR):
                _ = self.next()
                a = self.rel(CN.SHR, a, self.parse_c_add(), t)
            else:
                return a^

    def parse_c_add(mut self) raises -> CExpr:
        var a = self.parse_c_mul()
        while True:
            var t = self.peek()
            if self.is_punct(t, P.PLUS):
                _ = self.next()
                a = self.rel(CN.ADD, a, self.parse_c_mul(), t)
            elif self.is_punct(t, P.MINUS):
                _ = self.next()
                a = self.rel(CN.SUB, a, self.parse_c_mul(), t)
            else:
                return a^

    def parse_c_mul(mut self) raises -> CExpr:
        var a = self.parse_c_unary()
        while True:
            var t = self.peek()
            var op = -1
            if self.is_punct(t, P.STAR): op = CN.MUL
            elif self.is_punct(t, P.SLASH): op = CN.DIV
            elif self.is_punct(t, P.PERCENT): op = CN.MOD
            if op < 0:
                return a^
            _ = self.next()
            a = self.rel(op, a, self.parse_c_unary(), t)

    def parse_c_unary(mut self) raises -> CExpr:
        var t = self.peek()
        try:
            if self.is_punct(t, P.MINUS):
                _ = self.next()
                return ce_neg(self.parse_c_unary())
            if self.is_punct(t, P.PLUS):
                _ = self.next()
                return self.parse_c_unary()
            if self.is_punct(t, P.BANG):
                _ = self.next()
                return ce_lnot(self.parse_c_unary())
            if self.is_punct(t, P.TILDE):
                _ = self.next()
                return ce_bnot(self.parse_c_unary())
        except e:
            raise self.relocate(String(e), t)
        var a = self.parse_c_atom()
        var p = self.peek()
        if self.is_punct(p, P.POW):
            _ = self.next()
            return self.rel(CN.POW, a, self.parse_c_unary(), p)
        return a^

    def parse_c_literal(mut self, t: Token) raises -> CExpr:
        """An integer token, exact up to 2^64 (the token's float is not)."""
        var v: UInt64 = 0
        var digits = self.src.as_bytes()
        for i in range(t.start, t.end):
            var d = Int(digits[i])
            if d == 0x5F:
                continue
            var nv = v * 10 + UInt64(d - 0x30)
            if nv / 10 != v:
                raise self.err("QM-E580", "integer literal does not fit 64 bits", t, "use a smaller value")
            v = nv
        if v >= UInt64(1) << 63:
            return CExpr.literal_uint(Int(v))
        return CExpr.literal_int(Int(v))

    def parse_c_atom(mut self) raises -> CExpr:
        var t = self.peek()
        if t.kind == TK.INT:
            _ = self.next()
            return self.parse_c_literal(t)
        if t.kind == TK.FLOAT:
            _ = self.next()
            return CExpr.literal_float(t.num)
        if t.kind == TK.STRING:
            _ = self.next()
            var bs = self.src.as_bytes()
            var v = 0
            var n = t.end - t.start
            if n < 1 or n > 64:
                raise self.err("QM-E582", "a bit literal has 1 to 64 bits", t, "write \"0101\"")
            for i in range(t.start, t.end):
                if Int(bs[i]) != 0x30 and Int(bs[i]) != 0x31:
                    raise self.err("QM-E500", "a bit literal holds only 0 and 1", t, "write \"0101\"")
                v = (v << 1) | (Int(bs[i]) - 0x30)
            return CExpr.typed_const(bit_type(n), v)
        if self.is_punct(t, P.LPAREN):
            _ = self.next()
            var e = self.parse_cexpr()
            _ = self.expect_punct(P.RPAREN, "')'")
            return e^
        if t.kind != TK.IDENT:
            raise self.err("QM-E500", "expected an expression, found " + self.describe(t), t, "expressions read variables, bits, numbers and casts")
        var nm = self.text(t)
        var nx = self.peek(1)
        if (nm == "int" or nm == "uint" or nm == "float" or nm == "angle" or nm == "bool" or nm == "bit") and (self.is_punct(nx, P.LPAREN) or self.is_punct(nx, P.LBRACK)):
            var ty = self.parse_ctype(True)
            _ = self.expect_punct(P.LPAREN, "'(' after the type")
            var e = self.parse_cexpr()
            _ = self.expect_punct(P.RPAREN, "')'")
            try:
                return ce_explicit_cast(e, ty)
            except err:
                raise self.relocate(String(err), t)
        if self.is_punct(nx, P.LPAREN):
            return self.parse_c_call(t)
        _ = self.next()
        if nm == "pi" or nm == "π":
            return CExpr.literal_float(pi)
        if nm == "tau" or nm == "τ":
            return CExpr.literal_float(2.0 * pi)
        if nm == "euler" or nm == "ℇ":
            return CExpr.literal_float(2.718281828459045)
        if nm == "true":
            return CExpr.literal_bool(True)
        if nm == "false":
            return CExpr.literal_bool(False)
        var k = self.find_name(nm) if self.program_mode and self.depth == 0 else -1
        if k >= 0:
            ref e = self.names[k]
            if e.kind == NK.VAR:
                ref cv = self.prog.vars[e.index]
                return CExpr.variable(cv.store, cv.ty)
            if e.kind == NK.CONST:
                return self.const_vals[e.index].copy()
            self.pos -= 1
            return self.parse_c_bits(t)
        var s = len(self.scopes) - 1
        while s >= 0:
            for i in range(len(self.scopes[s].names)):
                if self.scopes[s].names[i] == nm:
                    if len(self.scopes[s].exprs) > i:
                        return self.scopes[s].exprs[i].copy()
                    var q = self.scopes[s].values[i]
                    if q.is_symbolic():
                        raise self.err("QM-E584", "gate parameter '" + nm + "' cannot be read as a classical value", t, "gate bodies take affine angles only")
                    if q.offset == floor(q.offset) and abs(q.offset) < 9e15:
                        return CExpr.literal_int(Int(q.offset))
                    return CExpr.literal_float(q.offset)
            if self.scopes[s].closed:
                break
            s -= 1
        var ce = self.const_exprs.get(nm)
        if ce:
            return ce.value().copy()
        var ip = self.inputs.get(nm)
        if ip:
            var it = self.input_types.get(nm)
            if it:
                return self.cast_at(CExpr.input(ip.value()), it.value(), t)
            return CExpr.input(ip.value())
        var cp = self.consts.get(nm)
        if cp:
            if cp.value().is_symbolic():
                raise self.err("QM-E584", "'" + nm + "' depends on an input, and a value is needed here", t, "use the input directly")
            return CExpr.literal_float(cp.value().offset)
        if nm in self.int_vars:
            raise self.err("QM-E504", "reading '" + nm + "' in an expression needs a value at run time", t, "parse with parse_qasm_program or load_any")
        var ri = self.reg_index.get(nm)
        if ri:
            if self.regs[ri.value()].is_qubit:
                raise self.err("QM-E520", "'" + nm + "' is a qubit register, not a classical value", t, "measure it into a bit register first")
            self.pos -= 1
            return self.parse_c_bits(t)
        _ = self.lookup_value(t)  # raises the undeclared-identifier error
        return CExpr.literal_int(0)

    def parse_c_bits(mut self, t: Token) raises -> CExpr:
        var bits = self.parse_bit_operand()
        for i in range(1, len(bits)):
            if bits[i] != bits[0] + i:
                raise self.err("QM-E504", "an expression reads consecutive ascending bits", t, "read the bits one by one")
        return CExpr.bits(bits[0], len(bits))

    def parse_c_call(mut self, f: Token) raises -> CExpr:
        var nm = self.text(f)
        _ = self.next()
        _ = self.expect_punct(P.LPAREN, "'('")
        var args = List[CExpr]()
        if not self.is_punct(self.peek(), P.RPAREN):
            args.append(self.parse_cexpr())
            while self.is_punct(self.peek(), P.COMMA):
                _ = self.next()
                args.append(self.parse_cexpr())
        _ = self.expect_punct(P.RPAREN, "')'")
        var func = -1
        if nm == "sin": func = CF.SIN
        elif nm == "cos": func = CF.COS
        elif nm == "tan": func = CF.TAN
        elif nm == "arcsin" or nm == "asin": func = CF.ASIN
        elif nm == "arccos" or nm == "acos": func = CF.ACOS
        elif nm == "arctan" or nm == "atan": func = CF.ATAN
        elif nm == "exp": func = CF.EXP
        elif nm == "ln" or nm == "log": func = CF.LOG
        elif nm == "sqrt": func = CF.SQRT
        elif nm == "abs": func = CF.ABS
        elif nm == "floor": func = CF.FLOOR
        elif nm == "ceil": func = CF.CEIL
        elif nm == "popcount": func = CF.POPCOUNT
        try:
            if nm == "mod" and len(args) == 2:
                return ce_function(CF.MOD, args[0], args[1].copy())
            if nm == "pow" and len(args) == 2:
                return ce_binary(CN.POW, args[0], args[1])
            if func >= 0 and len(args) == 1:
                return ce_function(func, args[0])
        except e:
            raise self.relocate(String(e), f)
        raise self.err("QM-E534", "unknown function '" + nm + "' or wrong number of arguments (" + String(len(args)) + ")", f, "supported: sin cos tan arcsin arccos arctan exp ln sqrt abs floor ceil popcount (1 arg), pow mod (2 args)")

    def parse_decl_program(mut self, is_const: Bool, is_output: Bool) raises:
        var tt = self.peek()
        if self.is_word(tt, "bit") or self.is_word(tt, "qubit"):
            raise self.err("QM-E504", "'const' registers are not supported", tt, "declare 'bit[n] c;' without const")
        var ty = self.parse_ctype(is_const)
        var nt = self.expect_ident("variable name")
        self.check_new_name(nt)
        var nm = self.text(nt)
        if not self.is_punct(self.peek(), P.EQ):
            _ = self.expect_punct(P.SEMI, "';'")
            if is_const:
                raise self.err("QM-E584", "a const needs a value", nt, "write 'const int x = 3;'")
            _ = self.declare_var(nm, ty, is_output)
            return
        var eq = self.next()
        if self.is_word(self.peek(), "measure"):
            raise self.err("QM-E504", "only bit registers take a measurement directly", self.peek(), "measure into a bit register, then assign")
        var e = self.parse_cexpr()
        _ = self.expect_punct(P.SEMI, "';'")
        var value = self.cast_at(e, ty, eq)
        if value.is_const() and not is_output and (is_const or nm not in self.reassigned):
            if len(self.scope_marks) > 0:
                self.names.append(_Name(nm, NK.CONST, len(self.const_vals)))
                self.const_vals.append(value^)
            else:
                self.consts[nm] = Param(_const_radians(value))
                self.const_exprs[nm] = value^
            return
        if is_const and not value.is_const():
            raise self.err("QM-E584", "'" + nm + "' is const but its value is only known at run time", nt, "drop 'const'")
        var v = self.declare_var(nm, ty, is_output)
        self.tb.assign(v, self.prog.add_expr(value), self.ops_end(), nt.line)

    def parse_var_assign(mut self) raises:
        """`x = e;`, `x += e;`."""
        var t = self.next()
        var v = self.find_var(self.text(t))
        var op = self.peek()
        if not (self.is_punct(op, P.EQ) or self.is_punct(op, P.OPASSIGN)):
            raise self.err("QM-E500", "expected '=' after variable '" + self.text(t) + "', found " + self.describe(op), op, "write 'x = 1;'")
        _ = self.next()
        if self.is_word(self.peek(), "measure"):
            raise self.err("QM-E504", "only bit registers take a measurement directly", self.peek(), "measure into a bit register, then assign")
        var ty = self.prog.vars[v].ty
        var store = self.prog.vars[v].store
        var e = self.parse_cexpr()
        _ = self.expect_punct(P.SEMI, "';'")
        if self.is_punct(op, P.OPASSIGN):
            e = self.compound(op, CExpr.variable(store, ty), e)
        self.tb.assign(v, self.prog.add_expr(self.cast_at(e, ty, op)), self.ops_end(), t.line)

    def compound(mut self, op: Token, a: CExpr, b: CExpr) raises -> CExpr:
        """The value of `a op= b`."""
        var txt = self.text(op)
        var code = CN.ADD
        if txt == "-=": code = CN.SUB
        elif txt == "*=": code = CN.MUL
        elif txt == "/=": code = CN.DIV
        elif txt == "%=": code = CN.MOD
        elif txt == "**=": code = CN.POW
        elif txt == "&=": code = CN.AND
        elif txt == "|=": code = CN.OR
        elif txt == "^=": code = CN.XOR
        elif txt == "<<=": code = CN.SHL
        elif txt == ">>=": code = CN.SHR
        return self.rel(code, a, b, op)

    def parse_bit_assign(mut self, ct: Token, cs: List[Int]) raises:
        """`c[0] = e;`, `c = e;`, `c ^= e;` for the bits parsed into `cs`."""
        var op = self.next()
        for i in range(1, len(cs)):
            if cs[i] != cs[0] + i:
                raise self.err("QM-E504", "an assignment writes consecutive ascending bits", ct, "assign the bits one by one")
        var e = self.parse_cexpr()
        _ = self.expect_punct(P.SEMI, "';'")
        if self.is_punct(op, P.OPASSIGN):
            e = self.compound(op, CExpr.bits(cs[0], len(cs)), e)
        self.assign_bits(cs[0], len(cs), self.text(ct), e, op)

    def assign_bits(mut self, first: Int, width: Int, nm: String, e: CExpr, t: Token) raises:
        var v = -1
        for i in range(len(self.prog.vars)):
            ref cv = self.prog.vars[i]
            if cv.ty.tag == CT.BIT and cv.store == first and cv.ty.width == width:
                v = i
        if v < 0:
            v = self.prog.add_bit_var(nm + "[" + String(first) + "]", first, width)
        self.tb.assign(v, self.prog.add_expr(self.cast_at(e, bit_type(width), t)), self.ops_end(), t.line)

    def parse_if_region(mut self, it: Token) raises:
        self.forbid_in_gate(it)
        _ = self.expect_punct(P.LPAREN, "'(' after if")
        var e = self.parse_cexpr()
        _ = self.expect_punct(P.RPAREN, "')' to close the condition")
        self.tb.begin(SK.IF, self.prog.add_expr(self.cast_at(e, bool_type(), it)), -1, self.ops_end(), it.line)
        if self.pragma_if:
            self.pragma_if = False
            self.parse_pragma_body(it)
        else:
            self.parse_statement()
        if self.is_word(self.peek(), "else"):
            var et = self.next()
            self.tb.else_(self.ops_end(), et.line)
            self.parse_statement()
        self.tb.end(self.ops_end(), it.line)

    def parse_while(mut self) raises:
        var wt = self.next()
        _ = self.expect_punct(P.LPAREN, "'(' after while")
        var e = self.parse_cexpr()
        _ = self.expect_punct(P.RPAREN, "')' to close the condition")
        self.tb.begin(SK.WHILE, self.prog.add_expr(self.cast_at(e, bool_type(), wt)), -1, self.ops_end(), wt.line)
        self.parse_statement()
        self.tb.end(self.ops_end(), wt.line)

    def body_has_jump(mut self) raises -> Bool:
        """Does the statement at self.pos hold a break or continue of its own
        loop? Inner loops take theirs, and `case 0 { } break;` closes a case."""
        var save = self.pos
        var start = self.pos
        self.skip_statement()
        var stop = self.pos
        self.pos = start
        var found = False
        var depth = 0
        var switch_depths = List[Int]()
        while self.pos < stop:
            var t = self.peek()
            if self.is_word(t, "while") or self.is_word(t, "for"):
                self.skip_statement()
                continue
            if self.is_word(t, "switch"):
                switch_depths.append(depth + 1)
            elif self.is_punct(t, P.LBRACE):
                depth += 1
            elif self.is_punct(t, P.RBRACE):
                if len(switch_depths) > 0 and switch_depths[len(switch_depths) - 1] == depth:
                    _ = switch_depths.pop()
                depth -= 1
            elif self.is_word(t, "continue"):
                found = True
            elif self.is_word(t, "break"):
                if not (len(switch_depths) > 0 and switch_depths[len(switch_depths) - 1] == depth):
                    found = True
            _ = self.next()
        self.pos = save
        return found

    def body_assigns(mut self, nm: String) raises -> Bool:
        """Does the statement at self.pos assign `nm`?"""
        var save = self.pos
        self.skip_statement()
        var stop = self.pos
        self.pos = save
        for i in range(save, stop):
            ref t = self.toks[i]
            if t.kind == TK.IDENT and (self.is_punct(self.toks[i + 1], P.EQ) or self.is_punct(self.toks[i + 1], P.OPASSIGN)) and self.text(t) == nm:
                return True
        return False

    def for_needs_region(mut self, ty: CType, nm: String) raises -> Bool:
        """With the iterable next: a FOR stays on the tape when its bounds are known
        only at run time, its values do not fit the variable's type, or its body
        jumps or assigns the variable. Otherwise it unrolls, to the same effect."""
        var save = self.pos
        var runtime = False
        var r = self.peek()
        var fits = True
        try:
            if self.is_punct(r, P.LBRACK):
                _ = self.next()
                var a = self.parse_const_int("range start")
                _ = self.expect_punct(P.COLON, "':' in range")
                var b = self.parse_const_int("range bound")
                var step = 1
                if self.is_punct(self.peek(), P.COLON):
                    _ = self.next()
                    step = b
                    b = self.parse_const_int("range stop")
                _ = self.expect_punct(P.RBRACK, "']'")
                # empty ranges fit; otherwise the two ends bound every value
                if step != 0 and ((step > 0 and a <= b) or (step < 0 and a >= b)):
                    fits = _value_fits(a, ty) and _value_fits(b, ty)
            elif self.is_punct(r, P.LBRACE):
                _ = self.next()
                while True:
                    fits = _value_fits(self.parse_const_int("set element"), ty) and fits
                    if not self.is_punct(self.peek(), P.COMMA):
                        break
                    _ = self.next()
            self.pos = save
        except:
            runtime = True  # the classical parser reports what is really wrong
            self.pos = save
        if runtime or not fits:
            return True
        self.skip_iterable()
        var needs = self.body_has_jump() or self.body_assigns(nm)
        self.pos = save
        return needs

    def skip_iterable(mut self) raises:
        var r = self.peek()
        if self.is_punct(r, P.LBRACK):
            self.skip_balanced(P.LBRACK, P.RBRACK, r)
        elif self.is_punct(r, P.LBRACE):
            self.skip_balanced(P.LBRACE, P.RBRACE, r)
        else:
            raise self.err("QM-E570", "for loops need a range [a:b] or set {a, b}", r, "write 'for uint i in [0:3] { ... }' (ranges are inclusive)")

    def parse_for_region(mut self, ty: CType, vt: Token) raises:
        var off = len(self.prog.for_ranges)
        var i64 = int_type(64)
        var rt = self.peek()
        var starts = -1
        var steps = -1
        var stops = -1
        var set_off = -1
        if self.is_punct(rt, P.LBRACK):
            _ = self.next()
            var a = self.parse_cexpr()
            _ = self.expect_punct(P.COLON, "':' in range", "write [start:stop] or [start:step:stop] (inclusive)")
            var b = self.parse_cexpr()
            var step = CExpr.literal_int(1)
            var has_step = False
            if self.is_punct(self.peek(), P.COLON):
                _ = self.next()
                step = b^
                b = self.parse_cexpr()
                has_step = True
            _ = self.expect_punct(P.RBRACK, "']' to close the range")
            if step.ty().tag == CT.FLOAT:
                raise self.err("QM-E570", "range step must be an integer, got " + type_name(step.ty()), rt, "write [0:2:10]")
            if a.ty().tag == CT.FLOAT or b.ty().tag == CT.FLOAT:
                raise self.err("QM-E570", "range " + ("start" if a.ty().tag == CT.FLOAT else "stop") + " must be an integer, got a float", rt, "write [0:3], or cast the bound with int(x)")
            _ = has_step
            if step.is_const() and step.const_int() == 0:
                raise self.err("QM-E570", "range step must be non-zero", rt, "write [0:2:10]")
            starts = self.prog.add_expr(self.cast_at(a, i64, rt))
            steps = self.prog.add_expr(self.cast_at(step, i64, rt))
            stops = self.prog.add_expr(self.cast_at(b, i64, rt))
        else:
            _ = self.next()
            var vals = List[Int]()
            vals.append(self.parse_exact_int("set element"))
            while self.is_punct(self.peek(), P.COMMA):
                _ = self.next()
                vals.append(self.parse_exact_int("set element"))
            _ = self.expect_punct(P.RBRACE, "'}' to close the set")
            set_off = len(self.prog.case_vals)
            self.prog.case_vals.append(len(vals))
            for v in vals:
                self.prog.case_vals.append(v)
        self.prog.for_ranges.append(starts)
        self.prog.for_ranges.append(steps)
        self.prog.for_ranges.append(stops)
        self.prog.for_ranges.append(self.prog.num_store)
        self.prog.num_store += 3
        self.prog.for_ranges.append(set_off)
        self.push_scope()
        var v = self.declare_var(self.text(vt), ty, False)
        self.tb.begin(SK.FOR, v, off, self.ops_end(), vt.line)
        self.loop_names.append(_LoopName(self.text(vt), self.block_depth + (1 if self.is_punct(self.peek(), P.LBRACE) else 0)))
        self.parse_statement()
        _ = self.loop_names.pop()
        self.tb.end(self.ops_end(), vt.line)
        self.pop_scope()

    def parse_switch_region(mut self, st: Token) raises:
        self.forbid_in_gate(st)
        _ = self.expect_punct(P.LPAREN, "'(' after switch")
        var e = self.parse_cexpr()
        _ = self.expect_punct(P.RPAREN, "')'")
        if e.ty().tag == CT.FLOAT:
            raise self.err("QM-E580", "switch takes an integer or bits, not a float", st, "cast the target: switch (int(x))")
        var unsigned_target = not e.ty().is_signed()
        self.tb.begin(SK.SWITCH, self.prog.add_expr(self.cast_at(e, int_type(64), st)), -1, self.ops_end(), st.line)
        var lb = self.expect_punct(P.LBRACE, "'{' to open the switch")
        var seen = List[Int]()
        var has_default = False
        while not self.is_punct(self.peek(), P.RBRACE):
            var c = self.peek()
            if c.kind == TK.EOF:
                raise self.err("QM-E500", "missing '}' to close the switch", lb, "every '{' needs a matching '}'")
            var vs = List[Int]()
            var is_default = False
            var has_label = False
            while self.is_word(self.peek(), "case") or self.is_word(self.peek(), "default"):
                if self.is_word(self.peek(), "default"):
                    _ = self.next()
                    is_default = True
                else:
                    _ = self.next()
                    self.parse_case_value(vs, unsigned_target)
                    while self.is_punct(self.peek(), P.COMMA):
                        _ = self.next()
                        self.parse_case_value(vs, unsigned_target)
                    has_label = True
                if self.is_punct(self.peek(), P.COLON):
                    _ = self.next()
            if not has_label and not is_default:
                raise self.err("QM-E500", "expected 'case' or 'default', found " + self.describe(c), c, "write 'case 0 { ... }' or 'default { ... }'")
            if not self.is_punct(self.peek(), P.LBRACE):
                raise self.err("QM-E500", "expected '{' to open the case, found " + self.describe(self.peek()), self.peek(), "write 'case 0 { ... }'")
            for v in vs:
                for u in seen:
                    if u == v:
                        raise self.err("QM-E541", "case value " + String(v) + " appears twice", st, "each value belongs to one case")
                seen.append(v)
            if is_default:
                if has_default:
                    raise self.err("QM-E541", "a switch has one default", c, "merge the two bodies")
                has_default = True
                self.tb.case_(-1, -1, self.ops_end(), c.line)  # values on a default body are covered by it
            else:
                var lo = len(self.prog.case_vals)
                for v in vs:
                    self.prog.case_vals.append(v)
                self.tb.case_(lo, len(self.prog.case_vals), self.ops_end(), c.line)
            self.parse_statement()
            if self.is_word(self.peek(), "break"):
                _ = self.next()
                _ = self.expect_punct(P.SEMI, "';'")
        _ = self.next()
        self.tb.end(self.ops_end(), st.line)

    def parse_exact_int(mut self, what: String) raises -> Int:
        """A constant integer, exact to its 64 bits (held by its bits at or above
        2^63), which the float path of parse_const_int cannot promise."""
        var t = self.peek()
        var e = self.parse_cexpr()
        if not e.is_const() or e.ty().tag == CT.FLOAT:
            raise self.err("QM-E584", what + " must be a constant integer", t, "use an integer literal or a const")
        return e.const_int()

    def parse_case_value(mut self, mut vs: List[Int], unsigned_target: Bool) raises:
        """One case label, exact to its 64 bits. A value of the other sign could only
        match by accident of its bits (-1 is not 2^64 - 1), so it is refused."""
        var t = self.peek()
        var e = self.parse_cexpr()
        if not e.is_const() or e.ty().tag == CT.FLOAT:
            raise self.err("QM-E584", "case value must be a constant integer", t, "use an integer literal or a const")
        var v = e.const_int()
        if (unsigned_target and e.ty().is_signed() and v < 0) or (not unsigned_target and e.ty().tag == CT.UINT and v < 0):
            raise self.err("QM-E580", "this case value has the wrong sign for the switch target, so it can never match", t, "drop the case, or cast the target")
        vs.append(v)

    def slot_for(mut self, v: Int) -> Int:
        """The parameter slot that mirrors variable `v`, in every open circuit."""
        var cur = self.prog.vars[v].slot
        if cur >= 0:
            return cur
        var nm = slot_prefix() + String(v) + "_" + self.prog.vars[v].name
        var index = self.circs[0].parameter(nm).index
        for i in range(1, len(self.circs)):
            _ = self.circs[i].parameter(nm)
        self.prog.vars[v].slot = index
        return index

    def var_of_store(self, store: Int) -> Int:
        for i in range(len(self.prog.vars)):
            if self.prog.vars[i].store == store and self.prog.vars[i].ty.tag != CT.BIT:
                return i
        return -1

    def affine(mut self, e: CExpr, n: Int, mut kind: Int, mut index: Int, mut coeff: Float64, mut offset: Float64) -> Bool:
        """Node `n` of `e` as coeff * s + offset for one symbol s. kind: 0 constant,
        1 variable with storage `index`, 2 input `index`. False when it is not
        affine in float arithmetic: integer nodes wrap, so they do not qualify."""
        var g = e.nodes[n]
        kind = 0
        index = -1
        coeff = 0.0
        offset = 0.0
        if g.op == CN.CONST:
            if g.ty.tag != CT.FLOAT:
                return False
            offset = g.f
            return True
        if g.op == CN.INPUT:
            kind = 2
            index = g.x
            coeff = 1.0
            return True
        if g.op == CN.VAR:
            if g.ty.tag != CT.FLOAT:
                return False
            kind = 1
            index = g.x
            coeff = 1.0
            return True
        if g.op == CN.CAST and g.ty.tag == CT.FLOAT:
            var src = e.nodes[g.x]
            if src.op == CN.VAR and (src.ty.tag == CT.INT or src.ty.tag == CT.UINT):
                kind = 1
                index = src.x
                coeff = 1.0
                return True
            return False
        if g.op == CN.NEG:
            if not self.affine(e, g.x, kind, index, coeff, offset):
                return False
            coeff = -coeff
            offset = -offset
            return True
        if g.op == CN.ADD or g.op == CN.SUB or g.op == CN.MUL or g.op == CN.DIV:
            var k1 = 0
            var i1 = -1
            var c1 = 0.0
            var o1 = 0.0
            var k2 = 0
            var i2 = -1
            var c2 = 0.0
            var o2 = 0.0
            if g.ty.tag != CT.FLOAT or not self.affine(e, g.x, k1, i1, c1, o1) or not self.affine(e, g.y, k2, i2, c2, o2):
                return False
            if g.op == CN.ADD or g.op == CN.SUB:
                if k1 != 0 and k2 != 0 and (k1 != k2 or i1 != i2):
                    return False
                var sg = 1.0 if g.op == CN.ADD else -1.0
                kind = k1 if k1 != 0 else k2
                index = i1 if k1 != 0 else i2
                coeff = c1 + sg * c2
                offset = o1 + sg * o2
                return True
            if g.op == CN.MUL:
                if k1 != 0 and k2 != 0:
                    return False
                kind = k1 if k1 != 0 else k2
                index = i1 if k1 != 0 else i2
                if k1 != 0:
                    coeff = c1 * o2
                    offset = o1 * o2
                else:
                    coeff = c2 * o1
                    offset = o2 * o1
                return True
            if k2 != 0 or o2 == 0.0:
                return False
            kind = k1
            index = i1
            coeff = c1 / o2
            offset = o1 / o2
            return True
        return False

    def reads_typed(self) -> Bool:
        """Does the expression at self.pos name a typed constant? Its meaning is then
        the classical table's, in both parsers; a literal-only expression keeps the
        constant path's (`rz(1/2)` is 0.5, as it always was here)."""
        var open = 0
        var i = self.pos
        while i < len(self.toks):
            ref t = self.toks[i]
            if t.kind == TK.EOF:
                return False
            if t.kind == TK.PUNCT:
                if t.sub == P.LPAREN:
                    open += 1
                elif t.sub == P.RPAREN:
                    if open == 0:
                        return False
                    open -= 1
                elif open == 0 and (t.sub == P.COMMA or t.sub == P.RBRACK or t.sub == P.COLON or t.sub == P.SEMI or t.sub == P.LBRACE or t.sub == P.RBRACE):
                    return False
            elif t.kind == TK.IDENT:
                var nm = self.text(t)
                if self.program_mode:
                    var k = self.find_name(nm)
                    if k >= 0:
                        if self.names[k].kind == NK.CONST:
                            return True
                        i += 1
                        continue
                if nm in self.const_exprs or nm in self.input_types:
                    return True
                var s = len(self.scopes) - 1
                while s >= 0:
                    for j in range(len(self.scopes[s].exprs)):
                        if self.scopes[s].names[j] == nm:
                            return True
                    if self.scopes[s].closed:
                        break
                    s -= 1
            i += 1
        return False

    def parse_gate_arg(mut self) raises -> Param:
        """A gate angle. Without typed constants, whatever the constant path takes
        keeps its exact value. Otherwise the classical parser decides: an affine
        expression becomes a `Param`, and in a program anything else goes to a
        temporary assigned just before the gate."""
        if self.depth > 0:
            return self.parse_expr()
        if not self.reads_typed():
            if not self.program_mode:
                return self.parse_expr()
            var save = self.pos
            try:
                var value = self.parse_expr()
                if self.is_punct(self.peek(), P.COMMA) or self.is_punct(self.peek(), P.RPAREN):
                    return value
            except:
                pass
            self.pos = save  # not a constant expression: the classical parser reports errors
        var t = self.peek()
        var e = self.parse_cexpr()
        if e.is_const():
            return Param(_const_radians(e))
        var ty = e.ty()
        if ty.tag == CT.ANGLE and self.program_mode:
            var root = e.root()
            if root.op == CN.VAR:
                return Param.symbol(self.slot_for(self.var_of_store(root.x)))
            var a = self.temp_var(ty)
            self.tb.assign(a, self.prog.add_expr(e), self.ops_end(), t.line)
            return Param.symbol(self.slot_for(a))
        if ty.tag != CT.FLOAT:
            if ty.tag == CT.BIT or ty.tag == CT.BOOL:
                e = self.cast_at(e, uint_type(ty.width), t)
            e = self.cast_at(e, float_type(), t)
        var kind = 0
        var index = -1
        var coeff = 0.0
        var offset = 0.0
        if self.affine(e, len(e.nodes) - 1, kind, index, coeff, offset):
            if kind == 0:
                return Param(offset)
            if kind == 2:
                return Param(offset, index, coeff)
            return Param(offset, self.slot_for(self.var_of_store(index)), coeff)
        if not self.program_mode:
            raise self.err("QM-E504", "this angle needs a value at run time", t, "parse with parse_qasm_program or load_any")
        var tmp = self.temp_var(float_type())
        self.tb.assign(tmp, self.prog.add_expr(e), self.ops_end(), t.line)
        return Param.symbol(self.slot_for(tmp))

    def temp_var(mut self, ty: CType) -> Int:
        var nm = "__t" + String(self.temp_count)
        self.temp_count += 1
        return self.prog.add_var(nm, ty, False)

    def finish_program(mut self) raises -> Program:
        var c = self.circs.pop()
        c.validate()
        var line = self.peek().line
        self.tb.flush(len(c.ops), line)
        if len(self.tb.open) > 0:
            raise Error("QM-E583: a region was left open. Hint: check the braces.")
        var p = self.prog.copy()
        p.tape = self.tb.tape.copy()
        for r in self.regs:
            if not r.is_qubit:
                p.cregs.append(CReg(r.name, r.offset, r.size))
        p.body = c^
        _ = check_program(p)
        return p^

    def finish(mut self) raises -> Circuit:
        var c = self.circs.pop()
        try:
            c.validate()
        except e:
            raise Error(String(e))
        return c^


def _const_radians(e: CExpr) -> Float64:
    """A constant as parse_expr and gates read it: an angle in radians."""
    if e.ty().tag == CT.ANGLE:
        return angle_radians(e.const_int(), e.ty().width)
    return e.const_float()


def _value_fits(v: Int, ty: CType) -> Bool:
    """The int64 `v` is representable in the loop type `ty`."""
    if ty.tag == CT.FLOAT or ty.width >= 64 and ty.tag == CT.INT:
        return True
    if ty.tag == CT.INT:
        return v >= -(1 << (ty.width - 1)) and v < (1 << (ty.width - 1))
    return v >= 0 and (ty.width >= 63 or v < (1 << ty.width))


def _loop_value(v: Int, ty: CType) -> CExpr:
    """What a loop variable of type `ty` holds on the pass with value `v`."""
    if ty.tag == CT.FLOAT:
        return CExpr.typed_const(ty, 0, Float64(v))
    return CExpr.typed_const(ty, v)


def _last_step(i: Int, step: Int, stop: Int) -> Bool:
    """Is `i` the last value of the range? Compares distances, so that the value
    after it need not exist in 64 bits."""
    var room = UInt64(stop) - UInt64(i) if step > 0 else UInt64(i) - UInt64(stop)
    var stride = UInt64(step) if step > 0 else UInt64(0) - UInt64(step)
    return stride > room


def _switch_cover(values: List[List[Int]], bit: Int, mut cube: List[Int], mut out: List[List[Int]]):
    """Partial assignments covering exactly what is not in `values` (bits above
    `bit` already fixed by `cube`, -1 = free): one per empty subtree of the bit trie."""
    if len(values) == 0:
        out.append(cube.copy())
        return
    if bit < 0:
        return
    for b in range(2):
        var side = List[List[Int]]()
        for v in values:
            if v[bit] == b:
                side.append(v.copy())
        cube[bit] = b
        _switch_cover(side, bit - 1, cube, out)
    cube[bit] = -1


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


def parse_qasm_program(source: String, name: String = "qasm") raises -> Program:
    """Parse OpenQASM into a `Program`: as `parse_qasm`, plus while, break, continue,
    classical variables and conditions on any value."""
    var p = QasmParser(source, name)
    p.program_mode = True
    p.prescan()
    p.parse_program()
    return p.finish_program()


def load_any(source: String, name: String = "qasm") raises -> Program:
    """The flat parser first, so anything a Circuit can hold keeps its exact
    meaning. After QM-E504 or QM-E541 the input needs a program, and the program
    parser's verdict stands. Any other error may still be classical syntax the
    constant path cannot read (`int[8](x)` is QM-E520 there), so the program parser
    gets a try, and if it refuses too the flat diagnostic is the one reported."""
    var flat_error: String
    try:
        return Program.from_circuit(parse_qasm(source, name))
    except e:
        flat_error = String(e)
    if flat_error.startswith("QM-E504") or flat_error.startswith("QM-E541"):
        return parse_qasm_program(source, name)
    try:
        return parse_qasm_program(source, name)
    except:
        raise Error(flat_error)


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
        var g = circuit.ops[i]
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
    if s.find(".") < 0 and s.find("e") < 0:
        s += ".0"
    return s^


def _check_finite(v: Float64) raises:
    if isnan(v) or isinf(v):
        raise Error("QM-E552: the constant " + String(v) + " has no OpenQASM literal. Hint: compute it at run time, or fix the expression that overflowed.")


struct _PNames(Movable):
    """What the expression printer needs to name things."""

    var stores: List[String]  # by storage index
    var params: List[String]  # by parameter index
    var num_clbits: Int

    def __init__(out self, n: Int):
        self.stores = List[String](length=n, fill=String(""))
        self.params = List[String]()
        self.num_clbits = 0


def _const_text(n: CNode, plain: Bool) raises -> String:
    """A constant; `plain` drops the type, for where the context restores it."""
    var t = n.ty
    if t.tag == CT.BOOL:
        return "true" if n.i != 0 else "false"
    if t.tag == CT.FLOAT:
        _check_finite(n.f)
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


def _child(p: Program, n: Int, names: _PNames, want: Int, plain: Bool) raises -> String:
    var t = _node_text(p, n, names, plain)
    if t[1] < want:
        return "(" + t[0] + ")"
    return t[0]


def _bits_text(first: Int, count: Int, names: _PNames) -> String:
    if count == names.num_clbits and first == 0 and count > 1:
        return String("c")
    if count == 1:
        return "c[" + String(first) + "]"
    return "c[" + String(first) + ":" + String(first + count - 1) + "]"


def _node_text(p: Program, n: Int, names: _PNames, plain: Bool = False) raises -> Tuple[String, Int]:
    """Text of node `n` and the precedence of its outermost operator."""
    ref g = p.nodes[n]
    var op = g.op
    if op == CN.CONST:
        var text = _const_text(g, plain)
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
            return _node_text(p, g.x, names, plain)
        var inner = _node_text(p, g.x, names, True)[0]
        # the table has no bit -> float, so that step stays written
        if g.ty.tag == CT.FLOAT and _is_soft_cast(p, g.x) and p.nodes[p.nodes[g.x].x].ty.tag == CT.BIT:
            inner = type_name(p.nodes[g.x].ty) + "(" + inner + ")"
        return (type_name(g.ty) + "(" + inner + ")", _PR_ATOM)
    if op == CN.NEG or op == CN.LNOT or op == CN.BNOT:
        var mark = String("-" if op == CN.NEG else ("!" if op == CN.LNOT else "~"))
        return (mark + _child(p, g.x, names, _PR_UNARY, False), _PR_UNARY)
    if op == CN.FN:
        var args = _node_text(p, g.x, names, True)[0]
        if g.y >= 0:
            args += ", " + _node_text(p, g.y, names, True)[0]
        return (_fn_text(g.i) + "(" + args + ")", _PR_ATOM)
    if op == CN.SC_END:
        var m = n - 1
        while not ((p.nodes[m].op == CN.SC_AND or p.nodes[m].op == CN.SC_OR) and p.nodes[m].y == n):
            m -= 1
        var is_and = p.nodes[m].op == CN.SC_AND
        var pr = _PR_AND if is_and else _PR_OR
        var left = _child(p, p.nodes[m].x, names, pr, False)
        var right = _child(p, g.x, names, pr + 1, False)
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
    return (_child(p, g.x, names, lwant, lp) + " " + _op_text(op) + " " + _child(p, g.y, names, rwant, rp), pr)


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
            ref v = p.vars[s.a]
            var head = _pad(depth) + "for " + type_name(v.ty) + " " + vnames[s.a] + " in "
            var set_off = p.for_ranges[s.b + FORR.SET]
            if set_off >= 0:
                head += "{"
                for j in range(p.case_vals[set_off]):
                    head += (", " if j > 0 else "") + String(p.case_vals[set_off + 1 + j])
                head += "}"
            else:
                var i64 = int_type(64)
                var a = _value_text(p, p.for_ranges[s.b + FORR.START], i64, names)
                var st = _value_text(p, p.for_ranges[s.b + FORR.STEP], i64, names)
                var b = _value_text(p, p.for_ranges[s.b + FORR.STOP], i64, names)
                head += "[" + a + ":" + ("" if st == "1" else st + ":") + b + "]"
            out += head + " {\n"
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
