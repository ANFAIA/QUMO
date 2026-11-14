"""POSIX extended regular expressions through libc, and a translator from the
Python `re` subset agents use (DESIGN.md 2.1.5).

`PosixRegex` wraps regcomp/regexec/regfree (REG_EXTENDED | REG_NOSUB, plus
REG_ICASE). `py_regex_to_ere` turns a Python pattern into an ERE for line search
(match / no match) and refuses what ERE cannot express with
`unsupported in POSIX ERE: <construct>`.

Remaining differences with Python: the process locale is not set, so `.` and
bracket expressions match bytes (a non-ASCII character counts as several `.`) and
`[[:alnum:]]` is ASCII-only; POSIX leftmost-longest matching does not matter for a
boolean search.
"""

from qumo_agent.native.libc import (
    CStr,
    buf_addr,
    c_regcomp,
    c_regexec,
    c_regerror,
    c_regfree,
    REG_EXTENDED,
    REG_ICASE,
    REG_NOSUB,
)


struct PosixRegex(Movable):
    """A compiled POSIX ERE (regex_t in a zeroed, 8-byte aligned 512-byte buffer)."""

    var _re: List[UInt64]
    var _ok: Bool

    def __init__(out self, ere: String, icase: Bool) raises:
        """regcomp(REG_EXTENDED | REG_NOSUB [| REG_ICASE]); raises regerror's text."""
        self._re = List[UInt64](length=64, fill=0)
        self._ok = False
        var flags = REG_EXTENDED | REG_NOSUB
        if icase:
            flags |= REG_ICASE
        var pat = CStr(ere)
        var rc = c_regcomp(Int(self._re.unsafe_ptr()), pat.addr(), flags)
        pat.keep()
        if rc != 0:
            var buf = List[UInt8](length=512, fill=0)
            var n = c_regerror(rc, Int(self._re.unsafe_ptr()), buf_addr(buf), 512)
            var text: String
            if n > 1:
                text = String(from_utf8_lossy=Span(buf)[0 : n - 1])
            else:
                text = "regcomp error " + String(rc)
            _ = len(buf)
            raise Error(text)
        self._ok = True

    def search(self, line: String) -> Bool:
        """True when the pattern matches somewhere in `line` (regexec == 0)."""
        var s = CStr(line)
        var rc = c_regexec(Int(self._re.unsafe_ptr()), s.addr(), 0, 0, 0)
        s.keep()
        _ = len(self._re)
        return rc == 0

    def __deinit__(deinit self):
        if self._ok:
            c_regfree(Int(self._re.unsafe_ptr()))
        _ = len(self._re)


# ---------------------------------------------------------------------------
# Python re -> POSIX ERE
# ---------------------------------------------------------------------------
comptime _WORD = "[:alnum:]_"
comptime _B_START = "(^|[^[:alnum:]_])"
comptime _B_END = "([^[:alnum:]_]|$)"


def _cps(s: String) -> List[Int]:
    """Code points of `s`."""
    var out = List[Int]()
    var bs = s.as_bytes()
    var i = 0
    var n = len(bs)
    while i < n:
        var b0 = Int(bs[i])
        var cp: Int
        var w: Int
        if b0 < 0x80:
            cp = b0
            w = 1
        elif b0 < 0xE0:
            cp = ((b0 & 0x1F) << 6) | (Int(bs[i + 1]) & 0x3F)
            w = 2
        elif b0 < 0xF0:
            cp = ((b0 & 0x0F) << 12) | ((Int(bs[i + 1]) & 0x3F) << 6) | (Int(bs[i + 2]) & 0x3F)
            w = 3
        else:
            cp = ((b0 & 0x07) << 18) | ((Int(bs[i + 1]) & 0x3F) << 12) | ((Int(bs[i + 2]) & 0x3F) << 6) | (Int(bs[i + 3]) & 0x3F)
            w = 4
        out.append(cp)
        i += w
    return out^


def _chr(cp: Int) -> String:
    return String(Codepoint(unsafe_unchecked_codepoint=UInt32(cp)))


def _unsupported(construct: String) -> Error:
    return Error("unsupported in POSIX ERE: " + construct)


def _is_digit(c: Int) -> Bool:
    return c >= 0x30 and c <= 0x39


def _hexval(c: Int) -> Int:
    if c >= 0x30 and c <= 0x39:
        return c - 0x30
    if c >= 0x61 and c <= 0x66:
        return c - 0x61 + 10
    if c >= 0x41 and c <= 0x46:
        return c - 0x41 + 10
    return -1


def _read_hex(p: List[Int], start: Int, count: Int) raises -> Int:
    if start + count > len(p):
        raise Error("incomplete escape at position " + String(start - 2))
    var v = 0
    for k in range(count):
        var h = _hexval(p[start + k])
        if h < 0:
            raise Error("incomplete escape at position " + String(start - 2))
        v = v * 16 + h
    return v


def _ere_literal(cp: Int) -> String:
    """`cp` as an ERE literal outside brackets (special characters bracketed)."""
    if cp == 0x5C:   # backslash
        return "\\\\"
    if cp == 0x5E:   # ^
        return "\\^"
    if cp == 0x2E or cp == 0x5B or cp == 0x28 or cp == 0x29 or cp == 0x2A or cp == 0x2B or cp == 0x3F or cp == 0x7B or cp == 0x7C or cp == 0x24:
        return "[" + _chr(cp) + "]"
    return _chr(cp)


def _escape_code(p: List[Int], mut i: Int) raises -> Int:
    """For an escape at p[i] == '\\' that denotes one character (\\t \\n \\xhh ...),
    the code point, advancing `i` past it; -1 when it is not a character escape."""
    var c = p[i + 1]
    if c == 0x74:   # t
        i += 2
        return 9
    if c == 0x6E:   # n
        i += 2
        return 10
    if c == 0x72:   # r
        i += 2
        return 13
    if c == 0x66:   # f
        i += 2
        return 12
    if c == 0x76:   # v
        i += 2
        return 11
    if c == 0x61:   # a
        i += 2
        return 7
    if c == 0x78:   # xhh
        var v = _read_hex(p, i + 2, 2)
        i += 4
        return v
    if c == 0x75:   # uhhhh
        var v = _read_hex(p, i + 2, 4)
        i += 6
        return v
    if c == 0x55:   # Uhhhhhhhh
        var v = _read_hex(p, i + 2, 8)
        i += 10
        return v
    var is_alnum = (c >= 0x30 and c <= 0x39) or (c >= 0x41 and c <= 0x5A) or (c >= 0x61 and c <= 0x7A)
    if not is_alnum:
        i += 2
        return c   # escaped punctuation, space or non-ASCII: the character itself
    return -1


def _translate_bracket(p: List[Int], mut i: Int) raises -> String:
    """Translates the Python character class starting at p[i] == '['; `i` ends past ']'."""
    var start = i
    i += 1
    var negate = False
    if i < len(p) and p[i] == 0x5E:
        negate = True
        i += 1
    var body = String()
    var has_rbracket = False
    var has_caret = False
    var has_hyphen = False
    var has_lbracket = False
    var first = True
    while True:
        if i >= len(p):
            raise Error("unterminated character set at position " + String(start))
        var c = p[i]
        if c == 0x5D and not first:   # ]
            i += 1
            break
        first = False
        if c == 0x5B and i + 1 < len(p) and (p[i + 1] == 0x3A or p[i + 1] == 0x2E or p[i + 1] == 0x3D):
            raise _unsupported("[" + _chr(p[i + 1]) + " inside [...]")
        var cp: Int
        if c == 0x5C:   # backslash
            if i + 1 >= len(p):
                raise Error("bad escape (end of pattern) at position " + String(i))
            var e = p[i + 1]
            if e == 0x64:   # \d
                body += "0-9"
                i += 2
                continue
            if e == 0x77:   # \w
                body += _WORD
                i += 2
                continue
            if e == 0x73:   # \s
                body += "[:space:]"
                i += 2
                continue
            if e == 0x44 or e == 0x57 or e == 0x53:
                raise _unsupported("\\" + _chr(e) + " inside [...]")
            if e == 0x62:   # \b is a backspace inside a class
                cp = 8
                i += 2
            elif _is_digit(e):
                raise _unsupported("\\" + _chr(e) + " inside [...]")
            else:
                var j = i
                cp = _escape_code(p, j)
                if cp < 0:
                    raise _unsupported("\\" + _chr(e))
                i = j
        else:
            cp = c
            i += 1
        # A range "a-z": cp is the start when a '-' follows that is not the last char.
        if i + 1 < len(p) and p[i] == 0x2D and p[i + 1] != 0x5D:
            var hi: Int
            if p[i + 1] == 0x5C:
                var j = i + 1
                if j + 1 >= len(p):
                    raise Error("bad escape (end of pattern) at position " + String(j))
                hi = _escape_code(p, j)
                if hi < 0:
                    raise _unsupported("\\" + _chr(p[j + 1]) + " in a range")
                i = j
            else:
                hi = p[i + 1]
                i += 2
            if hi < cp:
                raise Error("bad character range " + _chr(cp) + "-" + _chr(hi) + " at position " + String(start))
            # Range ends that are special inside an ERE bracket.
            if cp == 0x5D or cp == 0x5E or cp == 0x2D or hi == 0x5D or hi == 0x2D:
                raise _unsupported("range with ']', '^' or '-' inside [...]")
            if cp > 0x7F or hi > 0x7F:
                raise _unsupported("non-ASCII character inside [...]")
            body += _chr(cp) + "-" + _chr(hi)
            continue
        if cp == 0x5D:
            has_rbracket = True
        elif cp == 0x5E:
            has_caret = True
        elif cp == 0x2D:
            has_hyphen = True
        elif cp == 0x5B:
            has_lbracket = True
        elif cp > 0x7F:
            raise _unsupported("non-ASCII character inside [...]")
        else:
            body += _chr(cp)
    var out = String("[")
    if negate:
        out += "^"
    if has_rbracket:
        out += "]"
    if body.byte_length() == 0 and not has_rbracket and not has_lbracket and has_caret and not has_hyphen:
        # A lone caret cannot come first in an ERE bracket: "\^" outside one.
        return "\\^" if not negate else String("[^^]")
    out += body
    # '[' last (never followed by ':', '.' or '=' that way), then '^' (not first), then '-'.
    if has_lbracket:
        out += "["
    if has_caret:
        out += "^"
    if has_hyphen:
        out += "-"
    out += "]"
    return out^


def _quantifier_end(p: List[Int], i: Int) -> Int:
    """For p[i] == '{', the index past '}' when it is a Python quantifier
    ({m}, {m,}, {,n}, {m,n}); -1 when Python reads the brace literally."""
    var j = i + 1
    var digits1 = 0
    while j < len(p) and _is_digit(p[j]):
        j += 1
        digits1 += 1
    if j < len(p) and p[j] == 0x7D:
        return j + 1 if digits1 > 0 else -1
    if j >= len(p) or p[j] != 0x2C:
        return -1
    j += 1
    while j < len(p) and _is_digit(p[j]):
        j += 1
    if j < len(p) and p[j] == 0x7D:
        return j + 1
    return -1


def py_regex_to_ere(pattern: String) raises -> Tuple[String, Bool]:
    """(ERE text, icase) for a Python `re` pattern used for line search; raises
    `unsupported in POSIX ERE: <construct>` for what ERE cannot express (DESIGN.md
    2.1.5)."""
    var p = _cps(pattern)
    var i = 0
    var icase = False
    if len(p) >= 4 and p[0] == 0x28 and p[1] == 0x3F and p[2] == 0x69 and p[3] == 0x29:   # (?i)
        icase = True
        i = 4
    var body_start = i
    var out = String()
    var after_quant = False   # previous token was a quantifier
    var can_repeat = False    # previous token is something a quantifier can apply to
    while i < len(p):
        var c = p[i]
        if c == 0x5C:   # backslash
            if i + 1 >= len(p):
                raise Error("bad escape (end of pattern) at position " + String(i))
            var e = p[i + 1]
            after_quant = False
            can_repeat = True
            if e == 0x64:
                out += "[0-9]"
            elif e == 0x44:
                out += "[^0-9]"
            elif e == 0x77:
                out += "[" + _WORD + "]"
            elif e == 0x57:
                out += "[^" + _WORD + "]"
            elif e == 0x73:
                out += "[[:space:]]"
            elif e == 0x53:
                out += "[^[:space:]]"
            elif e == 0x62:   # \b
                can_repeat = False
                if i == body_start:
                    out += _B_START
                elif i + 2 == len(p):
                    out += _B_END
                else:
                    raise _unsupported("\\b")
            elif e == 0x41 or e == 0x5A or e == 0x42 or e == 0x7A or e == 0x47:   # \A \Z \B \z \G
                raise _unsupported("\\" + _chr(e))
            elif _is_digit(e):
                raise _unsupported("\\" + _chr(e))
            else:
                var j = i
                var cp = _escape_code(p, j)
                if cp < 0:
                    raise _unsupported("\\" + _chr(e))
                out += _ere_literal(cp)
                i = j
                continue
            i += 2
            continue
        if c == 0x5B:   # [
            out += _translate_bracket(p, i)
            after_quant = False
            can_repeat = True
            continue
        if c == 0x28:   # (
            after_quant = False
            can_repeat = False
            if i + 1 < len(p) and p[i + 1] == 0x3F:   # (?
                if i + 2 < len(p) and p[i + 2] == 0x3A:   # (?:
                    out += "("
                    i += 3
                    continue
                var what = String("(?")
                if i + 2 < len(p):
                    what += _chr(p[i + 2])
                    if (p[i + 2] == 0x3C or p[i + 2] == 0x50) and i + 3 < len(p):
                        what += _chr(p[i + 3])
                raise _unsupported(what)
            out += "("
            i += 1
            continue
        if c == 0x2A or c == 0x2B or c == 0x3F:   # * + ?
            if after_quant:
                if c == 0x3F:   # lazy: same match/no-match result
                    i += 1
                    after_quant = False
                    can_repeat = False
                    continue
                if c == 0x2B:
                    raise _unsupported("possessive quantifier")
                raise Error("multiple repeat at position " + String(i))
            if not can_repeat:
                raise Error("nothing to repeat at position " + String(i))
            can_repeat = False
            out += _chr(c)
            after_quant = True
            i += 1
            continue
        if c == 0x7B:   # {
            var end = _quantifier_end(p, i)
            if end < 0:
                out += "[{]"
                after_quant = False
                can_repeat = True
                i += 1
                continue
            if after_quant:
                raise Error("multiple repeat at position " + String(i))
            if not can_repeat:
                raise Error("nothing to repeat at position " + String(i))
            can_repeat = False
            var q = String("{")
            var k = i + 1
            if p[k] == 0x2C:
                q += "0"
            while k < end:
                q += _chr(p[k])
                k += 1
            out += q
            after_quant = True
            i = end
            continue
        after_quant = False
        if c == 0x5E or c == 0x24 or c == 0x7C:   # ^ $ |
            out += _chr(c)
            can_repeat = False
        elif c == 0x2E or c == 0x29:   # . )
            out += _chr(c)
            can_repeat = True
        else:
            out += _ere_literal(c)
            can_repeat = True
        i += 1
    return (out^, icase)
