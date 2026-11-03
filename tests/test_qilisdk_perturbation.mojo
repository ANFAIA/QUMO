"""Tests for qumo.sim.perturbation: gate-angle perturbation (qilisdk GaussianPerturbation, OffsetPerturbation)."""

from std.math import sqrt
from std.testing import assert_equal, assert_true
from std.utils.numerics import inf, nan

from qumo.ir.circuit import Circuit
from qumo.ir.gates import Op
from qumo.sim.perturbation import (
    gaussian_angle_perturbation,
    is_angle_rotation,
    offset_angle_perturbation,
)


def _mixed_circuit() -> Circuit:
    """Every kind of instruction the perturbation must treat differently."""
    var c = Circuit(3, 1, "mixed")
    c.rx(0.1, 0)
    c.cx(0, 1)
    c.ry(0.2, 1)
    c.h(2)
    c.rz(0.3, 2)
    c.cp(0.7, 0, 1)
    c.rzz(0.9, 1, 2)
    c.u(0.4, 0.5, 0.6, 0)
    c.depolarize1(0.01, 0)
    c.measure(0, 0)
    c.global_phase = 0.125
    return c^


def _rx_chain(n: Int) -> Circuit:
    var c = Circuit(1, 0, "chain")
    for _ in range(n):
        c.rx(0.0, 0)
    return c^


def _same_shape(a: Circuit, b: Circuit, what: String) raises:
    """Same instructions, operands and classical fields; angles not compared."""
    assert_equal(len(a.ops), len(b.ops), what + " op count")
    for i in range(len(a.ops)):
        var x = a.ops[i]
        var y = b.ops[i]
        assert_equal(x.kind, y.kind, what + " kind of #" + String(i))
        assert_equal(x.q0, y.q0, what + " q0 of #" + String(i))
        assert_equal(x.q1, y.q1, what + " q1 of #" + String(i))
        assert_equal(x.q2, y.q2, what + " q2 of #" + String(i))
        assert_equal(x.clbit, y.clbit, what + " clbit of #" + String(i))
        assert_equal(x.cond, y.cond, what + " cond of #" + String(i))


def _same_angles(a: Circuit, b: Circuit, what: String) raises:
    for i in range(len(a.ops)):
        assert_true(a.ops[i].p0 == b.ops[i].p0, what + " p0 of #" + String(i))
        assert_true(a.ops[i].p1 == b.ops[i].p1, what + " p1 of #" + String(i))
        assert_true(a.ops[i].p2 == b.ops[i].p2, what + " p2 of #" + String(i))


def _gaussian_code(c: Circuit, std: Float64) -> String:
    try:
        _ = gaussian_angle_perturbation(c, std, 1)
    except e:
        return String(String(e)[byte=:7])
    return ""


def _offset_code(c: Circuit, offset: Float64) -> String:
    try:
        _ = offset_angle_perturbation(c, offset)
    except e:
        return String(String(e)[byte=:7])
    return ""


def test_zero_std_and_zero_offset_are_identity() raises:
    var c = _mixed_circuit()
    var g = gaussian_angle_perturbation(c, 0.0, 7)
    var o = offset_angle_perturbation(c, 0.0)
    _same_shape(c, g, "std 0")
    _same_shape(c, o, "offset 0")
    _same_angles(c, g, "std 0")
    _same_angles(c, o, "offset 0")


def test_offset_moves_every_rotation_by_exactly_the_offset() raises:
    var c = Circuit(2, 0, "offset")
    c.rx(0.1, 0)
    c.cx(0, 1)
    c.ry(0.2, 1)
    c.h(0)
    c.rz(0.3, 0)
    var out = offset_angle_perturbation(c, 0.25)
    var a: Float64 = 0.1
    var b: Float64 = 0.2
    var d: Float64 = 0.3
    var off: Float64 = 0.25
    assert_equal(out.ops[0].p0.offset, a + off, "rx")
    assert_equal(out.ops[2].p0.offset, b + off, "ry")
    assert_equal(out.ops[4].p0.offset, d + off, "rz")
    assert_equal(out.ops[1].kind, Op.CX, "cx kept")
    assert_equal(out.ops[3].kind, Op.H, "h kept")
    _same_shape(c, out, "offset")


def test_symbolic_angle_keeps_its_coefficient() raises:
    var c = Circuit(1, 0, "symbolic")
    var theta = c.parameter("theta")
    c.rz(theta * 2.0 + 0.5, 0)
    var out = offset_angle_perturbation(c, 0.1)
    var p = out.ops[0].p0
    assert_true(p.is_symbolic(), "still symbolic")
    assert_equal(p.index, theta.index, "same symbol")
    assert_equal(p.coeff, 2.0, "coefficient kept")
    assert_true(abs(p.offset - 0.6) < 1e-12, "offset moved to 0.6, got " + String(p.offset))
    assert_equal(len(out.param_names), 1, "parameter table kept")
    assert_equal(out.param_names[0], "theta", "parameter name kept")
    assert_equal(c.ops[0].p0.offset, 0.5, "input not modified")


def test_gaussian_is_reproducible_per_seed() raises:
    var c = _rx_chain(64)
    var a = gaussian_angle_perturbation(c, 0.05, 11)
    var b = gaussian_angle_perturbation(c, 0.05, 11)
    var d = gaussian_angle_perturbation(c, 0.05, 12)
    var differs = 0
    for i in range(64):
        assert_equal(a.ops[i].p0.offset, b.ops[i].p0.offset, "same seed, angle #" + String(i))
        if a.ops[i].p0.offset != d.ops[i].p0.offset:
            differs += 1
    assert_true(differs > 0, "a different seed gives different angles")


def test_gaussian_sample_statistics() raises:
    var n = 4000
    var c = _rx_chain(n)
    var out = gaussian_angle_perturbation(c, 0.05, 7)
    var total = 0.0
    for i in range(n):
        total += out.ops[i].p0.offset
    var mean = total / Float64(n)
    var ss = 0.0
    for i in range(n):
        var e = out.ops[i].p0.offset - mean
        ss += e * e
    var sd = sqrt(ss / Float64(n - 1))
    assert_true(abs(mean) < 0.003, "sample mean " + String(mean) + " not within 0.003 of 0")
    assert_true(abs(sd - 0.05) <= 0.05 * 0.05, "sample sd " + String(sd) + " not within 5% of 0.05")


def test_bad_std_and_offset_are_refused() raises:
    var c = _rx_chain(3)
    assert_equal(_gaussian_code(c, -0.01), "QM-E415", "negative std")
    assert_equal(_gaussian_code(c, nan[DType.float64]()), "QM-E415", "NaN std")
    assert_equal(_gaussian_code(c, inf[DType.float64]()), "QM-E415", "infinite std")
    assert_equal(_offset_code(c, nan[DType.float64]()), "QM-E415", "NaN offset")
    assert_equal(_offset_code(c, -inf[DType.float64]()), "QM-E415", "infinite offset")
    assert_equal(_gaussian_code(c, 0.01), "", "a valid std is accepted")


def _check_only_rotations_moved(c: Circuit, res: Circuit, what: String) raises:
    """Everything but the rotation angles equals the input, and the output validates."""
    res.validate()
    _same_shape(c, res, what)
    assert_equal(res.global_phase, c.global_phase, what + " global phase")
    assert_equal(res.num_measurements, c.num_measurements, what + " measurements")
    assert_equal(res.num_clbits, c.num_clbits, what + " clbits")
    assert_equal(len(res.param_names), len(c.param_names), what + " parameter table")
    for i in range(len(c.ops)):
        if not is_angle_rotation(c.ops[i].kind):
            assert_true(res.ops[i].p0 == c.ops[i].p0, what + " non-rotation p0 of #" + String(i))
            assert_true(res.ops[i].p1 == c.ops[i].p1, what + " non-rotation p1 of #" + String(i))
            assert_true(res.ops[i].p2 == c.ops[i].p2, what + " non-rotation p2 of #" + String(i))


def test_perturbed_circuit_keeps_everything_else() raises:
    var c = _mixed_circuit()
    var g = gaussian_angle_perturbation(c, 0.05, 3)
    var o = offset_angle_perturbation(c, 0.25)
    _check_only_rotations_moved(c, g, "gaussian")
    _check_only_rotations_moved(c, o, "offset")
    assert_equal(c.ops[0].p0.offset, 0.1, "input not modified")
    assert_equal(c.ops[4].p0.offset, 0.3, "input rz not modified")


def main() raises:
    test_zero_std_and_zero_offset_are_identity()
    test_offset_moves_every_rotation_by_exactly_the_offset()
    test_symbolic_angle_keeps_its_coefficient()
    test_gaussian_is_reproducible_per_seed()
    test_gaussian_sample_statistics()
    test_bad_std_and_offset_are_refused()
    test_perturbed_circuit_keeps_everything_else()
    print("test_qilisdk_perturbation OK")
