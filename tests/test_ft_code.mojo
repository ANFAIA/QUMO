"""Tests for qumo.ft.pauli and qumo.ft.code: Pauli algebra, code library, validation errors."""

from std.testing import assert_equal, assert_true

from qumo.ft.pauli import SymPauli, StabilizerGroup, group_from
from qumo.ft.code import (
    StabilizerCode, steane_code, iceberg_code, css_422_code, noncss_422_code, five_qubit_code,
    identity_code, surface_code, repetition_code, code_by_name,
)


def _expect_error(code: String, msg: String) raises:
    assert_true(msg.startswith(code), "expected " + code + ", got: " + msg)
    assert_true(msg.find("Hint:") >= 0, "missing hint: " + msg)


def test_pauli_algebra() raises:
    var x = SymPauli.from_label("X")
    var y = SymPauli.from_label("Y")
    var z = SymPauli.from_label("Z")
    # XY = iZ, YX = -iZ, ZX = iY, XZ = -iY
    var xy = x.mul(y)
    assert_true(xy.same_pauli(z) and xy.phase == 1)
    var yx = y.mul(x)
    assert_true(yx.same_pauli(z) and yx.phase == 3)
    var xz = x.mul(z)
    assert_true(xz.same_pauli(y) and xz.phase == 3)
    assert_true(not x.commutes(z))
    var a = SymPauli.from_label("XXI")
    var b = SymPauli.from_label("ZZI")
    assert_true(a.commutes(b))
    assert_equal(SymPauli.from_label("-IZZI").label(), "-IZZI")
    assert_equal(SymPauli.from_label("XIZ").embed(2, 5).label(), "+IIXIZ")
    var bad = False
    try:
        _ = SymPauli.from_label("XQ")
    except e:
        _expect_error("QM-E720", String(e))
        bad = True
    assert_true(bad)


def test_group_membership() raises:
    var gens: List[SymPauli] = [SymPauli.from_label("XXXX"), SymPauli.from_label("ZZZZ")]
    var g = group_from(gens, 4)
    assert_equal(g.rank(), 2)
    # (XXXX)(ZZZZ) = (XZ)^4 = (-iY)^4 = +YYYY
    assert_true(g.contains(SymPauli.from_label("YYYY")))
    assert_true(not g.contains(SymPauli.from_label("-YYYY")))
    assert_true(g.contains_up_to_sign(SymPauli.from_label("-YYYY")))
    assert_true(not g.contains(SymPauli.from_label("XXII")))
    assert_true(not g.add(SymPauli.from_label("-XXXX")))  # dependent (up to sign)


def test_library() raises:
    var st = steane_code()
    assert_equal(st.n, 7)
    assert_equal(st.k, 1)
    assert_true(st.is_css())
    assert_equal(st.min_distance(4), 3)
    var f = five_qubit_code()
    assert_true(not f.is_css())
    assert_equal(f.min_distance(4), 3)
    var c4 = css_422_code()
    assert_equal(c4.min_distance(3), 2)
    var nc = noncss_422_code()
    assert_true(not nc.is_css())
    assert_equal(nc.min_distance(3), 2)
    var ks: List[Int] = [2, 4, 6]
    for k in ks:
        var ice = iceberg_code(k)
        assert_equal(ice.n, k + 2)
        assert_equal(ice.k, k)
        assert_equal(ice.min_distance(3), 2)
    # Y logical = i X Z is Hermitian and anticommutes with X and Z
    var yl = st.y_logical(0)
    assert_true(yl.is_hermitian())
    assert_true(not yl.commutes(st.x_logicals[0]) and not yl.commutes(st.z_logicals[0]))
    # adapters over qumo.qec memory circuits
    var s3 = surface_code(3)
    assert_equal(s3.n, 9)
    assert_equal(len(s3.generators), 8)
    assert_equal(s3.min_distance(3), 3)
    var s5 = surface_code(5)
    assert_equal(s5.n, 25)
    assert_equal(len(s5.generators), 24)
    var r5 = repetition_code(5)
    assert_equal(r5.n, 5)
    assert_equal(r5.min_distance(2), 1)  # phase flips are undetected
    assert_equal(identity_code(3).min_distance(1), 1)
    assert_equal(code_by_name("iceberg_4").n, 6)
    assert_equal(code_by_name("surface_3").name, "surface_3")
    assert_true(st.to_json().find("\"css\":true") >= 0)


def test_invalid_definitions() raises:
    var cases = 0
    try:
        var g: List[String] = ["XX", "ZI"]
        var xl: List[String] = ["XX"]
        var zl: List[String] = ["ZZ"]
        _ = StabilizerCode.from_strings("bad_comm", 2, 1, 1, g, xl, zl)
    except e:
        _expect_error("QM-E721", String(e))
        assert_true(String(e).find("generators") >= 0)
        cases += 1
    try:
        var g: List[String] = ["XXXX"]
        var xl: List[String] = ["XXII", "XIXI"]
        var zl: List[String] = ["IZIZ", "IIZZ"]
        _ = StabilizerCode.from_strings("bad_count", 4, 2, 2, g, xl, zl)
    except e:
        _expect_error("QM-E721", String(e))
        cases += 1
    try:
        var g: List[String] = ["XXXX", "ZZZZ"]
        var xl: List[String] = ["XIII", "XIXI"]  # X0 anticommutes with ZZZZ
        var zl: List[String] = ["IZIZ", "IIZZ"]
        _ = StabilizerCode.from_strings("bad_logical", 4, 2, 2, g, xl, zl)
    except e:
        _expect_error("QM-E721", String(e))
        cases += 1
    try:
        var g: List[String] = ["XXXX", "ZZZZ"]
        var xl: List[String] = ["XXII", "XXII"]  # same X for both qubits
        var zl: List[String] = ["IZIZ", "IIZZ"]
        _ = StabilizerCode.from_strings("bad_relations", 4, 2, 2, g, xl, zl)
    except e:
        _expect_error("QM-E721", String(e))
        cases += 1
    try:
        var g: List[String] = ["XXXX", "ZZZZ"]
        var xl: List[String] = ["XXII", "XIXI"]
        var zl: List[String] = ["IZIZ", "IIZZZ"]
        _ = StabilizerCode.from_strings("bad_size", 4, 2, 2, g, xl, zl)
    except e:
        _expect_error("QM-E720", String(e))
        cases += 1
    try:
        _ = iceberg_code(3)
    except e:
        _expect_error("QM-E722", String(e))
        cases += 1
    try:
        _ = code_by_name("toric_7")
    except e:
        _expect_error("QM-E722", String(e))
        cases += 1
    assert_equal(cases, 7)


def main() raises:
    test_pauli_algebra()
    test_group_membership()
    test_library()
    test_invalid_definitions()
    print("test_ft_code OK")
