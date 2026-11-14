"""OpenQASM lexer: tokens, diagnostic helpers and the builtin gate table.

Part of the OpenQASM front end, split from qumo/ir/qasm.mojo (which re-exports every name).
Nothing here depends on the parser or the emitter.
"""

from std.collections import Dict
from qumo.ir.gates import Op, op_name


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
                if iv < 9007199254740992:  # past 2^53 it stays large, and never wraps
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
