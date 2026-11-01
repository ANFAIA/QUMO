"""Text circuit drawer: `Circuit.draw()`, `qumo draw`, MCP `qumo_draw`.

Gates sit in columns (ASAP, a gate never overlaps another one's vertical span).
Each qubit is one line, with a spacer line between neighbours for vertical
connections; `@` is a control, `+` a CX target, `x` a SWAP end, `M` a measurement
with its classical bit, `?c3=1` a classical condition. Detectors, observables and
barriers take a whole column; `style="unicode"` swaps the wires for box-drawing
characters. Lines fold at `width` columns (0: never).
"""

from std.math import pi
from qumo.ir.gates import Op, op_name, op_arity, op_num_params, op_is_noise, op_is_controlled_1q, op_base_1q
from qumo.ir.param import Param
from qumo.ir.circuit import Circuit, Instruction


def _angle(v: Float64) -> String:
    """Short angle: pi fractions when exact, else 3 decimals."""
    if v == 0.0:
        return "0"
    var a = abs(v)
    var sign = String("-") if v < 0 else String("")
    for d in [1, 2, 3, 4, 6, 8, 16]:
        var r = a * Float64(d) / pi
        var n = Int(r + 0.5)
        if n > 0 and n < 32 and abs(r - Float64(n)) < 1e-10:
            var head = String("pi") if n == 1 else String(n) + "pi"
            return sign + head + ("" if d == 1 else "/" + String(d))
    var scaled = Int(a * 1000.0 + 0.5)
    var frac = String(scaled % 1000)
    while frac.byte_length() < 3:
        frac = "0" + frac
    while frac.byte_length() > 1 and frac[byte=frac.byte_length() - 1:] == "0":
        var shorter = String(frac[byte=0:frac.byte_length() - 1])
        frac = shorter^
    if scaled % 1000 == 0:
        return sign + String(scaled // 1000)
    return sign + String(scaled // 1000) + "." + frac


def _param(p: Param, names: List[String]) -> String:
    if not p.is_symbolic():
        return _angle(p.offset)
    var s = String("")
    if p.coeff == -1.0:
        s = "-"
    elif p.coeff != 1.0:
        s = _angle(p.coeff) + "*"
    s += names[p.index]
    if p.offset != 0.0:
        s += "+" + _angle(p.offset) if p.offset > 0 else _angle(p.offset)
    return s^


def _base_label(g: Instruction, names: List[String], kind: Int) -> String:
    """Gate name in capitals with its parameters."""
    var s = op_name(kind).upper()
    var n = op_num_params(kind)
    if n > 0:
        s += "(" + _param(g.p0, names)
        if n > 1:
            s += "," + _param(g.p1, names)
        if n > 2:
            s += "," + _param(g.p2, names)
        s += ")"
    return s^


struct _Column(Copyable):
    """One column: label per line (2 n - 1 lines), empty = wire."""

    var labels: List[String]
    var width: Int

    def __init__(out self, n: Int):
        self.labels = List[String](length=2 * n - 1, fill=String(""))
        self.width = 1

    def set(mut self, line: Int, label: String):
        self.labels[line] = label
        self.width = max(self.width, label.byte_length())


def _condition(c: Circuit, g: Instruction) -> String:
    if g.cond < 0:
        return ""
    if g.cond_reg < 0:
        return "?c" + String(g.cond) + "=" + String(g.cond_val)
    var s = String("?")
    var n = c.cond_table[g.cond_reg]
    for i in range(n):
        if i > 0:
            s += "&"
        s += "c" + String(c.cond_table[g.cond_reg + 1 + 2 * i]) + "=" + String(c.cond_table[g.cond_reg + 2 + 2 * i])
    return s^


def _span_set(mut col: _Column, lo: Int, hi: Int):
    """Vertical connector through every line strictly between qubits lo and hi."""
    for q in range(lo, hi):
        col.set(2 * q + 1, "|")
        if q > lo:
            if col.labels[2 * q].byte_length() == 0:
                col.set(2 * q, "|")


def _center(label: String, width: Int, fill: String) -> String:
    var pad = width - label.byte_length()
    var left = pad // 2
    var s = String("")
    for _ in range(left):
        s += fill
    s += label
    for _ in range(pad - left):
        s += fill
    return s^


def _glyph(label: String, unicode: Bool) -> String:
    if unicode:
        if label == "@":
            return "●"
        if label == "+":
            return "⊕"
        if label == "x":
            return "×"
        if label == "|":
            return "│"
    return label


def _wire_cell(label: String, width: Int, wire: String, unicode: Bool) -> String:
    """Centre `label` on a wire; glyphs are one column wide in every style."""
    var shown = _glyph(label, unicode)
    var pad = width - label.byte_length()
    var left = pad // 2
    var s = String("")
    for _ in range(left):
        s += wire
    s += shown
    for _ in range(pad - left):
        s += wire
    return s^


def draw_circuit(c: Circuit, width: Int = 80, style: String = "ascii") raises -> String:
    """Text drawing of `c`. `style`: ascii or unicode; `width`: fold column, 0 = never."""
    if style != "ascii" and style != "unicode":
        raise Error("QM-E005: unknown drawing style '" + style + "'. Hint: use 'ascii' or 'unicode'.")
    var unicode = style == "unicode"
    var n = c.num_qubits
    if n == 0:
        return "(no qubits)\n"
    var cols = List[_Column]()
    var level = List[Int](length=n, fill=0)  # first free column per qubit
    for g in c.ops:
        var col_index: Int
        var lo = 0
        var hi = n - 1
        var whole = g.kind == Op.BARRIER or g.kind == Op.TICK or g.kind == Op.DETECTOR or g.kind == Op.OBSERVABLE
        if g.kind == Op.CORRELATED_ERROR:
            lo = c.pauli_table[g.q0]
            hi = lo
            for j in range(1, g.q1):
                lo = min(lo, c.pauli_table[g.q0 + 2 * j])
                hi = max(hi, c.pauli_table[g.q0 + 2 * j])
        elif not whole:
            var a = op_arity(g.kind)
            lo = g.q0
            hi = g.q0
            for j in range(1, a):
                lo = min(lo, g.qubit(j))
                hi = max(hi, g.qubit(j))
        col_index = 0
        for q in range(lo, hi + 1):
            col_index = max(col_index, level[q])
        while len(cols) <= col_index:
            cols.append(_Column(n))
        for q in range(lo, hi + 1):
            level[q] = col_index + 1
        ref col = cols[col_index]
        var cond = _condition(c, g)
        var k = g.kind
        if k == Op.BARRIER or k == Op.TICK:
            var mark = String("|") if k == Op.BARRIER else String(":")
            for line in range(2 * n - 1):
                col.set(line, mark)
        elif k == Op.DETECTOR or k == Op.OBSERVABLE:
            var s = String("DET") if k == Op.DETECTOR else "OBS" + String(g.q2)
            s += "["
            for j in range(g.q1):
                if j > 0:
                    s += ","
                s += String(c.records[g.q0 + j])
            s += "]"
            col.set(0, s)
            _span_set(col, 0, n - 1)
        elif k == Op.MEASURE:
            col.set(2 * g.q0, "M>c" + String(g.clbit) + cond)
        elif k == Op.RESET:
            col.set(2 * g.q0, "|0>" + cond)
        elif k == Op.CORRELATED_ERROR:
            _span_set(col, lo, hi)
            for j in range(g.q1):
                var letter = c.pauli_table[g.q0 + 2 * j + 1]
                var s = String("ELSE_E") if g.q2 != 0 else String("E")
                s += "(" + _param(g.p0, c.param_names) + "):"
                s += "X" if letter == 1 else ("Y" if letter == 2 else "Z")
                col.set(2 * c.pauli_table[g.q0 + 2 * j], s)
        elif k == Op.PAULI_CHANNEL_2:
            _span_set(col, lo, hi)
            col.set(2 * g.q0, String("PC2"))
            col.set(2 * g.q1, String("PC2"))
        elif k == Op.KRAUS1 or k == Op.KRAUS2:
            var s = c.kraus[Int(g.p0.offset)].name
            col.set(2 * g.q0, s)
            if k == Op.KRAUS2:
                _span_set(col, lo, hi)
                col.set(2 * g.q1, s)
        elif op_is_noise(k):
            var s = op_name(k).upper()
            s += "(" + _param(g.p0, c.param_names)
            if k == Op.PAULI_CHANNEL_1:
                s += "," + _param(g.p1, c.param_names) + "," + _param(g.p2, c.param_names)
            s += ")"
            col.set(2 * g.q0, s)
            if k == Op.DEPOLARIZE2:
                col.set(2 * g.q1, s)
                _span_set(col, min(g.q0, g.q1), max(g.q0, g.q1))
        elif k == Op.SWAP or k == Op.CSWAP:
            var a = g.q0
            var b = g.q1
            if k == Op.CSWAP:
                a = g.q1
                b = g.q2
            _span_set(col, lo, hi)
            col.set(2 * a, "x")
            col.set(2 * b, "x")
            if k == Op.CSWAP:
                col.set(2 * g.q0, "@" + cond)
        elif k == Op.CX or k == Op.CCX:
            _span_set(col, lo, hi)
            var t = g.q1 if k == Op.CX else g.q2
            col.set(2 * t, "+" + cond)
            col.set(2 * g.q0, "@")
            if k == Op.CCX:
                col.set(2 * g.q1, "@")
        elif k == Op.CZ or k == Op.CCZ:
            _span_set(col, lo, hi)
            col.set(2 * g.q0, "@" + cond)
            col.set(2 * g.q1, "@")
            if k == Op.CCZ:
                col.set(2 * g.q2, "@")
        elif op_is_controlled_1q(k):
            _span_set(col, lo, hi)
            col.set(2 * g.q0, "@")
            col.set(2 * g.q1, _base_label(g, c.param_names, op_base_1q(k)) + cond)
        elif op_arity(k) == 1:
            col.set(2 * g.q0, _base_label(g, c.param_names, k) + cond)
        else:
            _span_set(col, lo, hi)
            var s = _base_label(g, c.param_names, k) + cond
            col.set(2 * g.q0, s)
            col.set(2 * g.q1, s)
    # names padded to one width
    var name_w = 0
    var names = List[String]()
    for q in range(n):
        var s = "q" + String(q) + ": "
        names.append(s)
        name_w = max(name_w, s.byte_length())
    var wire = String("─") if unicode else String("-")
    var out = String("")
    var start = 0
    while True:
        # one fold: as many columns as fit in `width`
        var used = name_w + 1
        var stop = start
        while stop < len(cols):
            var w = cols[stop].width + 2
            if width > 0 and used + w + 1 > width and stop > start:
                break
            used += w
            stop += 1
        for line in range(2 * n - 1):
            var row = String("")
            var is_wire = line % 2 == 0
            if is_wire:
                row += _center(names[line // 2], name_w, " ")
                row += wire
            else:
                row += _center("", name_w + 1, " ")
            for i in range(start, stop):
                ref col = cols[i]
                var label = col.labels[line]
                if is_wire:
                    row += wire + _wire_cell(label, col.width, wire, unicode) + wire
                else:
                    row += " " + _wire_cell(label, col.width, " ", unicode) + " "
            if is_wire:
                row += wire if stop >= len(cols) else ">"
            out += row.rstrip() + "\n"
        start = stop
        if start >= len(cols):
            break
        out += "\n"
    return out^
