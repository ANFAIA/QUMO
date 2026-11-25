"""PauliSum.from_list builds the same sum as repeated add_label calls."""

from qumo.ir.pauli import PauliSum, PauliTerm, parse_pauli_sum


def main() raises:
    var a = PauliSum()
    a.add_label(1.0, "ZZ")
    a.add_label(-1.0, "ZX")
    a.add_label(0.5, "XZ")
    a.add_label(0.25, "ZZ")  # merges with the first term
    var b = PauliSum.from_list([(1.0, "ZZ"), (-1.0, "ZX"), (0.5, "XZ"), (0.25, "ZZ")])
    if len(a) != len(b) or len(b) != 3:
        raise Error("FAIL: term count " + String(len(a)) + " vs " + String(len(b)))
    for i in range(len(a)):
        var p = a.terms[i].copy()
        var q = b.terms[i].copy()
        if p.x != q.x or p.z != q.z or p.coeff != q.coeff:
            raise Error("FAIL: term " + String(i) + " differs")
    var empty = PauliSum.from_list(List[Tuple[Float64, String]]())
    if len(empty) != 0:
        raise Error("FAIL: empty list should give an empty sum")
    var raised = False
    try:
        _ = PauliSum.from_list([(1.0, "ZQ")])
    except e:
        raised = String(e).startswith("QM-E301")
    if not raised:
        raise Error("FAIL: invalid label must raise QM-E301")
    _check_text_inputs()
    print("OK")


def _raises(text: String, code: String, what: String) raises:
    var msg = String()
    var raised = False
    try:
        _ = parse_pauli_sum(text)
    except e:
        raised = True
        msg = String(e)
    if not raised:
        raise Error("FAIL: '" + text + "' must raise " + code + " (" + what + ")")
    if not msg.startswith(code):
        raise Error("FAIL: '" + text + "' raised " + msg + ", wanted " + code)


def _check_text_inputs() raises:
    # labels: at most 64 characters, leftmost = highest qubit
    var h = PauliSum()
    h.add_label(1.0, "X" + "I" * 63)
    if len(h) != 1 or h.terms[0].x != (UInt64(1) << 63) or h.terms[0].z != 0:
        raise Error("FAIL: 64-character label must put X on qubit 63")
    var h1 = PauliSum()
    h1.add_label(1.0, "I" * 63 + "Z")
    if h1.terms[0].z != 1:
        raise Error("FAIL: rightmost character is qubit 0")
    var raised = False
    var long = PauliSum()
    try:
        long.add_label(1.0, "X" + "I" * 64)
    except e:
        raised = String(e).startswith("QM-E302")
    if not raised:
        raise Error("FAIL: 65-character label must raise QM-E302")
    if len(long) != 0:
        raise Error("FAIL: rejected label must not add a term")
    raised = False
    try:
        _ = PauliSum.from_list([(1.0, "Z" * 65)])
    except e:
        raised = String(e).startswith("QM-E302")
    if not raised:
        raise Error("FAIL: from_list with a 65-character label must raise QM-E302")
    var t = PauliTerm(1.0, UInt64(1) << 63, 0)
    if t.label(64) != "X" + "I" * 63:
        raise Error("FAIL: label(64)")
    if t.label(3) != "III":
        raise Error("FAIL: label(3) of a term above qubit 2")
    raised = False
    try:
        _ = t.label(65)
    except e:
        raised = String(e).startswith("QM-E302")
    if not raised:
        raise Error("FAIL: label(65) must raise QM-E302")

    # parse_pauli_sum: a qubit once per term
    _raises("X0 Z0", "QM-E301", "repeated qubit")
    _raises("Z0 Z0", "QM-E301", "repeated qubit")
    _raises("0.5*Z0 Z1 + 2*X1 Y1", "QM-E301", "repeated qubit in the second term")
    _raises("Z0 Z", "QM-E301", "indexed and dense factor on qubit 0")
    var named = String()
    try:
        _ = parse_pauli_sum("1.0 + X3 Z3")
    except e:
        named = String(e)
    if named.find("qubit 3") < 0 or named.find("X3 Z3") < 0:
        raise Error("FAIL: message must name the qubit and the term, got " + named)
    # a sign needs a term
    _raises("Z0 +", "QM-E301", "trailing sign")
    _raises("Z0 -", "QM-E301", "trailing sign")
    _raises("+", "QM-E301", "lone sign")

    # results that were already right stay the same
    var a = parse_pauli_sum("0.5 * Z0 Z1 - 1.2 * X0 Y2 + 0.3")
    if len(a) != 3 or a.terms[0].coeff != 0.5 or a.terms[1].coeff != -1.2 or a.terms[2].coeff != 0.3 or a.terms[2].x != 0 or a.terms[2].z != 0:
        raise Error("FAIL: sparse sum changed")
    var b = parse_pauli_sum("0.5*ZZI + 0.2*XIX")
    if len(b) != 2 or b.terms[0].z != 6 or b.terms[1].x != 5:
        raise Error("FAIL: dense sum changed")
    var c = parse_pauli_sum("Z0 I0 + I1 Z1 - -X2")
    if len(c) != 3 or c.terms[0].z != 1 or c.terms[1].z != 2 or c.terms[2].coeff != 1.0:
        raise Error("FAIL: identity factors or double minus changed")
    var d = parse_pauli_sum("Z1 IZ")
    if len(d) != 1 or d.terms[0].z != 3:
        raise Error("FAIL: indexed and dense factors on different qubits")
    var e = parse_pauli_sum("")
    if len(e) != 0:
        raise Error("FAIL: empty text is an empty sum")
    var f = parse_pauli_sum("2")
    if len(f) != 1 or f.terms[0].coeff != 2.0 or f.terms[0].z != 0:
        raise Error("FAIL: lone coefficient is the identity term")
