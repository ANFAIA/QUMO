"""Insert `#pragma qumo` depolarizing and readout channels into compiled OpenQASM 2 text.

Native port of the text transform `inject_noise` in plugins/hardware/qumo_hardware/local.py.
Input is one statement per line, as the local provider compresses it. Output keeps every
original line and adds, for each gate, `depolarize1(p1)` after a single-qubit gate and
`depolarize2(p2)` after a two-qubit gate (`depolarize1(p2)` per qubit for wider gates), and
`x_error(pm)` before every measurement (one per measured qubit). Gate and opaque definitions
are copied unchanged; `barrier`, `reset`, declarations and existing pragmas get no channel.
Lines are split on `\\n` only.
"""

from qumo.ir.qasm import _float_text


struct NoisyText(Copyable):
    var text: String
    var depolarize1: Int
    var depolarize2: Int
    var x_error: Int

    def __init__(out self, var text: String, depolarize1: Int, depolarize2: Int, x_error: Int):
        self.text = text^
        self.depolarize1 = depolarize1
        self.depolarize2 = depolarize2
        self.x_error = x_error


def _slice(s: String, lo: Int, hi: Int) -> String:
    var bs = s.as_bytes()
    return String(unsafe_from_utf8=bs[lo:hi])


def _strip(s: String) -> String:
    return String(s.strip())


def _split_lines(text: String) -> List[String]:
    """Python `str.splitlines()` for '\\n' separators: no trailing empty line."""
    var out = List[String]()
    var bs = text.as_bytes()
    var n = len(bs)
    var start = 0
    for i in range(n):
        if bs[i] == 10:
            out.append(String(unsafe_from_utf8=bs[start:i]))
            start = i + 1
    if start < n:
        out.append(String(unsafe_from_utf8=bs[start:n]))
    return out^


def _is_word_byte(b: UInt8) -> Bool:
    return (b >= 48 and b <= 57) or (b >= 65 and b <= 90) or (b >= 97 and b <= 122) or b == 95


def _qubits(head: String) -> List[Int]:
    """Qubit indices of every `q[<digits>]` token (regex `\\bq\\[(\\d+)\\]`)."""
    var out = List[Int]()
    var bs = head.as_bytes()
    var n = len(bs)
    var i = 0
    while i + 2 < n:
        if bs[i] == 113 and bs[i + 1] == 91 and (i == 0 or not _is_word_byte(bs[i - 1])):  # 'q' '['
            var j = i + 2
            var v = 0
            var digits = 0
            while j < n and bs[j] >= 48 and bs[j] <= 57:
                v = v * 10 + Int(bs[j]) - 48
                digits += 1
                j += 1
            if digits > 0 and j < n and bs[j] == 93:  # ']'
                out.append(v)
                i = j + 1
                continue
        i += 1
    return out^


def _count(s: String, ch: UInt8) -> Int:
    var k = 0
    for b in s.as_bytes():
        if b == ch:
            k += 1
    return k


def _qref(q: Int) -> String:
    return "q[" + String(q) + "]"


def inject_noise(text: String, p1: Float64, p2: Float64, pm: Float64) -> NoisyText:
    """Add the QUMO noise pragmas to `text`; rates are per gate (p1, p2) and per measurement (pm)."""
    var out = String()
    var lines = _split_lines(text)
    var depth = 0
    var d1 = 0
    var d2 = 0
    var xe = 0
    for li in range(len(lines)):
        var raw = lines[li]
        var line = _strip(raw)
        var cut = line.find("//")
        var body = _strip(_slice(line, 0, cut)) if cut >= 0 else line
        if depth > 0 or body.startswith("gate ") or body.startswith("opaque "):
            depth += _count(body, 123) - _count(body, 125)  # '{' '}'
            out += raw + "\n"
            continue
        if (
            body.byte_length() == 0
            or body.startswith("OPENQASM") or body.startswith("include") or body.startswith("qreg")
            or body.startswith("creg") or body.startswith("#pragma") or body.startswith("barrier")
            or body.startswith("reset") or not body.endswith(";")
        ):
            out += raw + "\n"
            continue
        var stmt = body
        if stmt.startswith("if"):
            var rp = stmt.find(")")
            if rp >= 0:
                stmt = _strip(_slice(stmt, rp + 1, stmt.byte_length()))
        var arrow = stmt.find("->")
        var head = _slice(stmt, 0, arrow) if arrow >= 0 else stmt
        var qs = _qubits(head)
        if stmt.startswith("measure"):
            if pm > 0:
                for qb in qs:
                    out += "#pragma qumo x_error(" + _float_text(pm) + ") " + _qref(qb) + "\n"
                    xe += 1
            out += raw + "\n"
            continue
        out += raw + "\n"
        if len(qs) == 1 and p1 > 0:
            out += "#pragma qumo depolarize1(" + _float_text(p1) + ") " + _qref(qs[0]) + "\n"
            d1 += 1
        elif len(qs) == 2 and p2 > 0:
            out += "#pragma qumo depolarize2(" + _float_text(p2) + ") " + _qref(qs[0]) + ", " + _qref(qs[1]) + "\n"
            d2 += 1
        elif len(qs) > 2 and p2 > 0:
            for qb in qs:
                out += "#pragma qumo depolarize1(" + _float_text(p2) + ") " + _qref(qb) + "\n"
                d1 += 1
    if len(lines) == 0:
        out = "\n"
    return NoisyText(out^, d1, d2, xe)
