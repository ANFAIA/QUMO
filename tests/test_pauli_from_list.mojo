"""PauliSum.from_list builds the same sum as repeated add_label calls."""

from qumo.ir.pauli import PauliSum


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
    print("OK")
