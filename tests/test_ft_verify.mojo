"""Tests for qumo.ft.verify and qumo.ft.gadgets: every gadget verified, wrong gadgets caught.

Mirrors Guppy FT's tests/verifier (Steane, [[4,2,2]] CSS and non-CSS,
including the deliberately incorrect "Table 2, row 4" implementation).
"""

from std.math import pi
from std.testing import assert_equal, assert_true

from qumo.ir.circuit import Circuit
from qumo.ir.gates import Op
from qumo.ft.code import steane_code, iceberg_code, css_422_code, noncss_422_code, five_qubit_code, surface_code
from qumo.ft.verify import (
    valid_clifford_implementation, valid_stabilizer_state_preparation, is_valid_clifford_implementation,
)
from qumo.ft.gadgets import (
    verify_steane_library, verify_iceberg_library, valid_magic_state_preparation, steane_encoder_zero,
    steane_prep_zero_ft, steane_prep_h_nonft, steane_prep_t_nonft, transversal_1q, transversal_2q,
    iceberg_prep_zero,
)


def test_libraries() raises:
    var n_ok = 0
    for c in verify_steane_library(steane_code()):
        assert_true(c.report.ok, c.name + ": " + c.report.reason)
        n_ok += 1
    var ks: List[Int] = [2, 4]
    for k in ks:
        for c in verify_iceberg_library(iceberg_code(k)):
            assert_true(c.report.ok, c.name + ": " + c.report.reason)
            n_ok += 1
    assert_true(n_ok > 40)


def test_steane_wrong_gadgets() raises:
    var st = steane_code()
    # logical S is transversal S^dag; transversal S is S^dag: right operator, wrong sign
    var sem = Circuit(1)
    sem.s(0)
    var r = valid_clifford_implementation(sem, transversal_1q(Op.S, 7), st)
    assert_true(not r.ok)
    assert_true(r.reason.find("wrong sign") >= 0, r.reason)
    # CX with the blocks swapped implements CX(1, 0)
    var scx = Circuit(2)
    scx.cx(0, 1)
    var wrong = Circuit(14)
    for q in range(7):
        wrong.cx(7 + q, q)
    assert_true(not is_valid_clifford_implementation(scx, wrong, st, 2))
    # a "Hadamard" on 6 of the 7 qubits leaves the code space
    var sh = Circuit(1)
    sh.h(0)
    var h6 = Circuit(7)
    for q in range(6):
        h6.h(q)
    var r2 = valid_clifford_implementation(sh, h6, st)
    assert_true(not r2.ok and r2.reason.find("code space") >= 0, r2.reason)
    # state preparation: |0>_L circuit is not |+>_L, and a broken encoder fails
    var plus = Circuit(1)
    plus.h(0)
    assert_true(not valid_stabilizer_state_preparation(plus, steane_encoder_zero(), st).ok)
    var broken = steane_encoder_zero()
    _ = broken.ops.pop()  # drop the last CX
    assert_true(not valid_stabilizer_state_preparation(Circuit(1), broken, st).ok)
    assert_true(valid_stabilizer_state_preparation(Circuit(1), steane_prep_zero_ft(), st, 1, 1).ok)
    # magic states: T|+> verified; the |H> state before the S^dag H rotation is not T|+>
    assert_true(valid_magic_state_preparation(steane_prep_t_nonft(), st).ok)
    assert_true(not valid_magic_state_preparation(steane_prep_h_nonft(), st).ok)
    # T^dag|+> is not T|+> (theta sign)
    assert_true(not valid_magic_state_preparation(steane_prep_t_nonft(), st, -pi / 4).ok)


def test_guppy_422_cases() raises:
    var c4 = css_422_code()
    var s1 = Circuit(2)
    s1.cx(0, 1)
    var i1 = Circuit(4)
    i1.swap(3, 1)
    assert_true(is_valid_clifford_implementation(s1, i1, c4))
    var s2 = Circuit(2)
    s2.cz(0, 1)
    var i2 = Circuit(4)
    i2.s(0)
    i2.sdg(1)
    i2.sdg(2)
    i2.s(3)
    assert_true(is_valid_clifford_implementation(s2, i2, c4))
    # Bell state across two blocks
    var sb = Circuit(4)
    sb.h(0)
    sb.cx(0, 2)
    sb.h(1)
    sb.cx(1, 3)
    var ib = Circuit(8)
    ib.compose(iceberg_prep_zero(4))
    ib.compose(iceberg_prep_zero(4), 4)
    for q in range(4):
        ib.h(q)
    ib.swap(1, 2)
    for q in range(4):
        ib.cx(q, 4 + q)
    assert_true(valid_stabilizer_state_preparation(sb, ib, c4, 2).ok)
    # non-CSS [[4,2,2]]: Table 2 row 4, incorrect vs correct (Guppy FT test)
    var nc = noncss_422_code()
    var s4 = Circuit(2)
    s4.cx(0, 1)
    s4.swap(0, 1)
    var bad = Circuit(4)
    bad.h(0)
    bad.s(0)
    bad.sdg(1)
    bad.h(1)
    bad.sdg(2)
    bad.h(2)
    bad.h(3)
    bad.s(3)
    var rb = valid_clifford_implementation(s4, bad, nc)
    assert_true(not rb.ok)
    assert_true(rb.reason.find("wrong sign") >= 0, rb.reason)
    var good = Circuit(4)
    good.h(0)
    good.sdg(0)
    good.sdg(1)
    good.h(1)
    good.s(2)
    good.h(2)
    good.h(3)
    good.s(3)
    assert_true(is_valid_clifford_implementation(s4, good, nc))
    var scz = Circuit(2)
    scz.cz(0, 1)
    var icz = Circuit(4)
    icz.swap(1, 2)
    assert_true(is_valid_clifford_implementation(scz, icz, nc))


def test_other_codes() raises:
    # transversal X / Z on the perfect code and on the surface code adapter
    var f = five_qubit_code()
    var sx = Circuit(1)
    sx.x(0)
    assert_true(is_valid_clifford_implementation(sx, transversal_1q(Op.X, 5), f))
    var s3 = surface_code(3)
    var ix = Circuit(9)
    for q in range(3):
        ix.x(q)  # X_L = X on row j = 0
    assert_true(is_valid_clifford_implementation(sx, ix, s3))
    # transversal H is NOT a logical gate of the rotated surface code (X and Z checks swap)
    var sh = Circuit(1)
    sh.h(0)
    assert_true(not is_valid_clifford_implementation(sh, transversal_1q(Op.H, 9), s3))
    # clean ancilla that must come back to |0>
    var st = steane_code()
    var sem = Circuit(1)
    sem.x(0)
    var anc_ok = Circuit(8)
    anc_ok.x(7)
    for q in range(7):
        anc_ok.x(q)
    anc_ok.x(7)
    assert_true(valid_clifford_implementation(sem, anc_ok, st, 1, 1).ok)
    var anc_bad = Circuit(8)
    anc_bad.h(7)
    for q in range(7):
        anc_bad.x(q)
    assert_true(not valid_clifford_implementation(sem, anc_bad, st, 1, 1).ok)


def test_errors() raises:
    var st = steane_code()
    var hit = 0
    try:
        _ = valid_clifford_implementation(Circuit(2), Circuit(7), st)
    except e:
        assert_true(String(e).startswith("QM-E723"))
        hit += 1
    try:
        var c = Circuit(7)
        c.t(0)
        _ = valid_clifford_implementation(Circuit(1), c, st)
    except e:
        assert_true(String(e).startswith("QM-E723"))
        hit += 1
    assert_equal(hit, 2)


def main() raises:
    test_libraries()
    test_steane_wrong_gadgets()
    test_guppy_422_cases()
    test_other_codes()
    test_errors()
    print("test_ft_verify OK")
