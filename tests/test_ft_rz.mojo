"""Tests for qumo.ft.rz: exact / searched Clifford+T Rz and the comparator RUS Rz."""

from std.math import pi
from std.testing import assert_equal, assert_true

from qumo.ir.circuit import Circuit
from qumo.ir.gates import Op
from qumo.ft.rz import (
    synthesize_rz, exact_rz_sequence, sequence_matrix, rz_matrix, phase_distance, comparator_angle,
    comparator_register_size, append_comparator_rz, angle_register, comparator_t_count_per_attempt,
    CliffordTable,
)
from qumo.ft.emulate import comparator_rz_check
from qumo.ft.arch import SteaneBuilder
from qumo.ft.encode import compile, encode


def test_exact_and_search() raises:
    assert_equal(len(CliffordTable().mats), 24)
    for m in range(-8, 9):
        var seq = List[Int]()
        assert_true(exact_rz_sequence(Float64(m) * pi / 4, seq))
        assert_true(phase_distance(rz_matrix(Float64(m) * pi / 4), sequence_matrix(seq)) < 1e-7)  # sqrt(1 - |tr|/2) amplifies rounding
    var seq = List[Int]()
    assert_true(not exact_rz_sequence(0.3, seq))
    # searched words: the reported error is the true distance, T-count matches
    var prev = 1.0
    var budgets: List[Int] = [4, 8, 12]
    for t in budgets:
        var r = synthesize_rz(0.3, 1e-9, t)
        assert_true(r.t_count <= t)
        var tc = 0
        for g in r.gates:
            if g == Op.T or g == Op.TDG:
                tc += 1
        assert_equal(tc, r.t_count)
        assert_true(abs(phase_distance(rz_matrix(0.3), sequence_matrix(r.gates)) - r.error) < 1e-9)
        assert_true(r.error <= prev + 1e-12)
        prev = r.error
    var good = synthesize_rz(1.0, 2e-2, 16)
    assert_true(good.error <= 2e-2)
    var bad = False
    try:
        _ = synthesize_rz(0.3, 0.0, 4)
    except e:
        bad = String(e).startswith("QM-E728")
    assert_true(bad)


def test_comparator_static() raises:
    assert_equal(comparator_register_size(0.1), 5)
    assert_equal(comparator_register_size(0.01), 8)
    assert_equal(comparator_t_count_per_attempt(5), 26)
    var thetas: List[Float64] = [0.5, 1.3, -3.0 * pi / 4, 2.9, 0.0]
    for th in thetas:
        var r = comparator_rz_check(th, 0.1, 4, 16, 3)
        # exact w.r.t. the angle encoded in k, and within epsilon of theta
        assert_true(r.max_eff_error < 1e-9, String(r))
        assert_true(r.max_angle_error <= 0.1, String(r))
        assert_true(r.success_rate > 0.6, String(r))


def test_comparator_runtime_angle() raises:
    """The angle arrives as classical bits: two different angles give circuits
    that differ only in the angle-register preparation."""
    var r = comparator_rz_check(1.1, 0.1, 3, 12, 5, runtime_angle=True)
    assert_true(r.max_eff_error < 1e-9, String(r))
    assert_true(r.max_angle_error <= 0.1, String(r))
    var n = comparator_register_size(0.1)
    var c1 = Circuit(1)
    var b1 = angle_register(c1, comparator_angle(0.4, n))
    var i1 = append_comparator_rz(c1, 0, 0.4, 0.1, 3, b1)
    var c2 = Circuit(1)
    var b2 = angle_register(c2, comparator_angle(2.2, n))
    _ = append_comparator_rz(c2, 0, 2.2, 0.1, 3, b2)
    # bodies start at the first op on the target (qubit 0) and must be identical
    var s1 = 0
    while c1.ops[s1].q0 != 0:
        s1 += 1
    var s2 = 0
    while c2.ops[s2].q0 != 0:
        s2 += 1
    assert_true(s1 != s2)  # the angle registers differ
    assert_equal(len(c1.ops) - s1, len(c2.ops) - s2)
    var diff = 0
    for i in range(len(c1.ops) - s1):
        var g1 = c1.ops[s1 + i]
        var g2 = c2.ops[s2 + i]
        if g1.kind != g2.kind or g1.q0 != g2.q0 or g1.q1 != g2.q1 or g1.cond != g2.cond or g1.cond_val != g2.cond_val:
            diff += 1
    assert_equal(diff, 0)
    assert_true(i1.t_count == 3 * 2 * 7 + 3 * 4 * (n - 2))


def test_comparator_compiles_to_steane() raises:
    """Comparator mode: the RUS circuit (classical lane, conditioned SWAPs,
    conditioned Paulis) compiles to the Steane logical gate set; its logical
    semantics is the computational circuit (checked on the state vector
    elsewhere), and it encodes (resource numbers only: ~60 logical blocks)."""
    var c = Circuit(1, 1)
    c.h(0)
    c.rz(0.7, 0)
    c.h(0)
    c.measure(0, 0)
    var arch = SteaneBuilder().with_rz("comparator", 0.1, 16, 2).build(16)
    var lp = compile(arch, c)
    assert_true(lp.count(1) > 0)  # FREE ops from the attempt resets
    assert_true(lp.t_count() > 20)
    assert_true(lp.count(23) > 0)  # CSET: the classical lane became classical expressions
    var prog = encode(arch, c)
    assert_true(prog.stats.injections == lp.t_count())
    assert_true(prog.circuit.num_qubits > 40, String(prog.circuit.num_qubits))


def main() raises:
    test_exact_and_search()
    test_comparator_static()
    test_comparator_runtime_angle()
    test_comparator_compiles_to_steane()
    print("test_ft_rz OK")
