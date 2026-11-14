"""Insert `#pragma qumo` depolarizing and readout channels into compiled OpenQASM 2 text.

Native port of the text transform `inject_noise` in plugins/hardware/qumo_hardware/local.py.
Input is one statement per line, as the local provider compresses it. Output keeps every
original line and adds, for each gate, `depolarize1(p1)` after a single-qubit gate and
`depolarize2(p2)` after a two-qubit gate (`depolarize1(p2)` per qubit for wider gates), and
`x_error(pm)` before every measurement (one per measured qubit). Gate and opaque definitions
are copied unchanged; `barrier`, `reset`, declarations and existing pragmas get no channel.
Lines are split on `\\n` only.

Text in another layout is taken too: several statements on one line and a statement over
several lines are re-emitted one statement per line (canonical input passes through byte
for byte). A whole-register operand (`measure q -> c;`, `h q;`) gets one channel per qubit
of `qreg q`. What the pragmas cannot address raises QM-E913 instead of passing through
without noise: a register not named `q`, a multi-qubit gate broadcast over registers, a
statement with no closing `;`.
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


def _operand_text(stmt: String) -> String:
    """Operands of one statement without its `;`: after the gate name and its
    parenthesised parameters, up to `->` for a measurement."""
    var bs = stmt.as_bytes()
    var n = len(bs)
    var i = 0
    while i < n and (_is_word_byte(bs[i])):
        i += 1
    while i < n and (bs[i] == 32 or bs[i] == 9):
        i += 1
    if i < n and bs[i] == 40:  # '('
        var depth = 0
        while i < n:
            if bs[i] == 40:
                depth += 1
            elif bs[i] == 41:
                depth -= 1
                if depth == 0:
                    i += 1
                    break
            i += 1
    var end = n
    var arrow = stmt.find("->")
    if arrow >= 0:
        end = arrow
    elif n > 0 and bs[n - 1] == 59:  # ';'
        end = n - 1
    if i >= end:
        return String()
    return _strip(_slice(stmt, i, end))


def _operands(stmt: String) -> List[String]:
    var out = List[String]()
    var text = _operand_text(stmt)
    if text.byte_length() == 0:
        return out^
    for part in text.split(","):
        out.append(_strip(String(part)))
    return out^


def _resolve_qubits(stmt: String, head: String, qreg_size: Int, shown: String) raises -> List[Int]:
    """Qubits a statement acts on. `q[i]` operands are read as before; a bare `q`
    stands for every qubit of `qreg q` (one-operand statements only)."""
    var qs = _qubits(head)
    var ops = _operands(stmt)
    if len(qs) == len(ops):
        return qs^
    if len(ops) == 1 and ops[0] == "q":
        if qreg_size < 0:
            raise Error("QM-E913: inject_noise found '" + shown + "' before any 'qreg q[n];'. Hint: declare the register first.")
        var all = List[Int]()
        for i in range(qreg_size):
            all.append(i)
        return all^
    raise Error("QM-E913: inject_noise cannot place noise on '" + shown + "'. Hint: it takes operands q[i] of one register named q, or a whole register for measure and one-qubit gates; export the circuit with to_qasm2() first.")


def _emit_statement(raw: String, body: String, p1: Float64, p2: Float64, pm: Float64, mut out: String,
                    mut d1: Int, mut d2: Int, mut xe: Int, mut qreg_size: Int) raises:
    """One complete statement (`body` ends with ';'); `raw` is the line written out."""
    if body.startswith("qreg"):
        var lb = body.find("[")
        var rb = body.find("]")
        var name = _strip(_slice(body, 4, lb if lb >= 0 else body.byte_length() - 1))
        if name != "q" or qreg_size >= 0:
            raise Error("QM-E913: inject_noise takes one quantum register named q, found '" + body + "'. Hint: export the circuit with to_qasm2() first.")
        if lb >= 0 and rb > lb:
            qreg_size = Int(_strip(_slice(body, lb + 1, rb)))
        else:
            qreg_size = 1
        out += raw + "\n"
        return
    if (
        body.startswith("OPENQASM") or body.startswith("include") or body.startswith("creg")
        or body.startswith("barrier") or body.startswith("reset")
    ):
        out += raw + "\n"
        return
    var stmt = body
    if stmt.startswith("if"):
        var rp = stmt.find(")")
        if rp >= 0:
            stmt = _strip(_slice(stmt, rp + 1, stmt.byte_length()))
    var arrow = stmt.find("->")
    var head = _slice(stmt, 0, arrow) if arrow >= 0 else stmt
    var qs = _resolve_qubits(stmt, head, qreg_size, body)
    if stmt.startswith("measure"):
        if pm > 0:
            for qb in qs:
                out += "#pragma qumo x_error(" + _float_text(pm) + ") " + _qref(qb) + "\n"
                xe += 1
        out += raw + "\n"
        return
    out += raw + "\n"
    var bare = len(qs) > 1 and len(_operands(stmt)) == 1  # a one-qubit gate over the register
    if (len(qs) == 1 or bare) and p1 > 0:
        for qb in qs:
            out += "#pragma qumo depolarize1(" + _float_text(p1) + ") " + _qref(qb) + "\n"
            d1 += 1
    elif bare:
        pass
    elif len(qs) == 2 and p2 > 0:
        out += "#pragma qumo depolarize2(" + _float_text(p2) + ") " + _qref(qs[0]) + ", " + _qref(qs[1]) + "\n"
        d2 += 1
    elif len(qs) > 2 and p2 > 0:
        for qb in qs:
            out += "#pragma qumo depolarize1(" + _float_text(p2) + ") " + _qref(qb) + "\n"
            d1 += 1


def inject_noise(text: String, p1: Float64, p2: Float64, pm: Float64) raises -> NoisyText:
    """Add the QUMO noise pragmas to `text`; rates are per gate (p1, p2) and per measurement (pm)."""
    var out = String()
    var lines = _split_lines(text)
    var depth = 0
    var d1 = 0
    var d2 = 0
    var xe = 0
    var qreg_size = -1
    var pending = String()  # a statement still waiting for its ';'
    var gate_open = False  # a `gate` header whose '{' has not come yet
    for li in range(len(lines)):
        var raw = lines[li]
        var line = _strip(raw)
        var cut = line.find("//")
        var body = _strip(_slice(line, 0, cut)) if cut >= 0 else line
        if pending.byte_length() == 0 and (depth > 0 or gate_open or body.startswith("gate ") or body.startswith("opaque ")):
            depth += _count(body, 123) - _count(body, 125)  # '{' '}'
            if body.find("{") >= 0:
                gate_open = False
            elif body.startswith("gate "):
                gate_open = True  # the body opens on a later line
            if depth == 0 and body.find("}") >= 0 and not body.endswith("}"):
                raise Error("QM-E913: inject_noise found a statement after a gate definition on one line: '" + body + "'. Hint: put it on its own line.")
            out += raw + "\n"
            continue
        if body.byte_length() == 0 or body.startswith("#pragma"):
            out += raw + "\n"
            continue
        if pending.byte_length() == 0 and body.endswith(";") and _count(body, 59) == 1:
            _emit_statement(raw, body, p1, p2, pm, out, d1, d2, xe, qreg_size)
            continue
        # another layout: several statements on the line, or one over several lines
        var bs = body.as_bytes()
        var start = 0
        for i in range(len(bs)):
            if bs[i] == 59:  # ';'
                var piece = _strip(_slice(body, start, i + 1))
                var stmt = (pending + " " + piece) if pending.byte_length() > 0 else piece
                pending = String()
                start = i + 1
                if stmt.startswith("gate ") or stmt.startswith("opaque "):
                    raise Error("QM-E913: inject_noise needs a gate definition to start its own line: '" + stmt + "'. Hint: export the circuit with to_qasm2() first.")
                _emit_statement(stmt, stmt, p1, p2, pm, out, d1, d2, xe, qreg_size)
        if start < len(bs):
            var rest = _strip(_slice(body, start, len(bs)))
            if rest.byte_length() > 0:
                pending = (pending + " " + rest) if pending.byte_length() > 0 else rest
    if pending.byte_length() > 0:
        raise Error("QM-E913: inject_noise found a statement without its ';': '" + pending + "'. Hint: OpenQASM statements end with ';'.")
    if len(lines) == 0:
        out = "\n"
    return NoisyText(out^, d1, d2, xe)
