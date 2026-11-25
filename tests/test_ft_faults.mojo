"""Tests for qumo.ft.faults: exact t-fault-tolerance counts, cross-checked against the tableau.

The counts are the ones measured with Stim 1.16 (examples/ft/ft_state_prep_check_stim.py);
the tableau cross-check runs every single fault through TableauSimulator and compares the
measurement records and state-group signs with the bit-mask Pauli frame.
"""

from std.bit import pop_count
from std.testing import assert_equal, assert_true

from qumo.ir.circuit import Circuit
from qumo.sim.stabilizer import TableauSimulator
from qumo.ft.pauli import SymPauli
from qumo.ft.code import steane_code, five_qubit_code, identity_code
from qumo.ft.verify import valid_stabilizer_state_preparation
from qumo.ft.gadgets import steane_encoder_zero, steane_prep_zero_ft
from qumo.ft.faults import (
    FaultModel, Fault, FaultToleranceReport, fault_tolerant_state_preparation, count_undetected,
    reduced_weight, single_faults,
)


def _stab_flag() -> Circuit:
    """Encoder + flag on the code stabilizer Z0 Z1 Z2 Z3: valid without faults, not fault tolerant."""
    var c = Circuit(8, 1, "stab_flag")
    c.compose(steane_encoder_zero())
    c.cx(0, 7)
    c.cx(1, 7)
    c.cx(2, 7)
    c.cx(3, 7)
    c.measure(7, 0)
    return c^


def test_encoder_not_ft() raises:
    var st = steane_code()
    var r = fault_tolerant_state_preparation(Circuit(1), steane_encoder_zero(), st)
    assert_true(not r.ok)
    assert_equal(r.t, 1)
    assert_equal(r.locations, 18)
    assert_equal(r.fault_sets, 136)
    assert_equal(r.undetected, 136)
    assert_equal(r.dangerous, 29)
    assert_equal(r.operations, 8)
    assert_equal(r.cx_depth, 3)
    assert_true(r.witness.find("init q3: X") >= 0, r.witness)
    assert_true(r.witness.find("{2,3}") >= 0, r.witness)
    var rx = fault_tolerant_state_preparation(Circuit(1), steane_encoder_zero(), st, 0, -1, FaultModel.xz_only())
    assert_equal(rx.fault_sets, 61)
    assert_equal(rx.undetected, 61)
    assert_equal(rx.dangerous, 8)


def test_flagged_ft() raises:
    var st = steane_code()
    var r = fault_tolerant_state_preparation(Circuit(1), steane_prep_zero_ft(), st, 1)
    assert_true(r.ok, r.reason)
    assert_equal(r.fault_sets, 183)
    assert_equal(r.undetected, 90)
    assert_equal(r.dangerous, 0)
    assert_equal(r.witness, "")
    assert_equal(r.operations, 12)
    assert_equal(r.cx_depth, 6)
    assert_equal(r.model, "circuit_level")
    assert_true(String(r).startswith("FAULT TOLERANT"), String(r))
    assert_true(r.to_json().find("\"fault_tolerant\":true") >= 0, r.to_json())
    var rx = fault_tolerant_state_preparation(Circuit(1), steane_prep_zero_ft(), st, 1, -1, FaultModel.xz_only())
    assert_true(rx.ok)
    assert_equal(rx.fault_sets, 81)
    assert_equal(rx.undetected, 54)
    assert_equal(rx.dangerous, 0)


def test_negative_control() raises:
    var st = steane_code()
    var impl = _stab_flag()
    assert_true(valid_stabilizer_state_preparation(Circuit(1), impl, st, 1, 1).ok)
    var r = fault_tolerant_state_preparation(Circuit(1), impl, st, 1)
    assert_true(not r.ok)
    assert_equal(r.fault_sets, 198)
    assert_equal(r.undetected, 122)
    assert_equal(r.dangerous, 21)
    var rx = fault_tolerant_state_preparation(Circuit(1), _stab_flag(), st, 1, -1, FaultModel.xz_only())
    assert_equal(rx.fault_sets, 87)
    assert_equal(rx.undetected, 65)
    assert_equal(rx.dangerous, 6)


def test_count_undetected() raises:
    var st = steane_code()
    assert_equal(count_undetected(Circuit(1), steane_encoder_zero(), st, 0, 1), 136)
    assert_equal(count_undetected(Circuit(1), steane_encoder_zero(), st, 0, 2), 8331)
    assert_equal(count_undetected(Circuit(1), steane_prep_zero_ft(), st, 1, 2), 7735)
    assert_equal(count_undetected(Circuit(1), _stab_flag(), st, 1, 2), 9466)
    assert_equal(count_undetected(Circuit(1), steane_encoder_zero(), st, 0, 2, FaultModel.xz_only()), 1707)
    assert_equal(count_undetected(Circuit(1), steane_prep_zero_ft(), st, 1, 2, FaultModel.xz_only()), 1702)
    assert_equal(count_undetected(Circuit(1), _stab_flag(), st, 1, 2, FaultModel.xz_only()), 2200)


def test_reduced_weight() raises:
    var st = steane_code()
    var group = st.generators.copy()
    group.append(SymPauli.from_label("ZZZZZZZ"))
    var cases: List[String] = ["ZZZZZZZ", "IXXXIII", "IIXXIII", "ZZIIIII", "XXXXXXX", "YIIIIII"]
    var want_x: List[Int] = [0, 1, 2, 0, 3, 1]
    var want_z: List[Int] = [0, 0, 0, 1, 0, 1]
    for i in range(len(cases)):
        var r = reduced_weight(SymPauli.from_label(cases[i]), group)
        assert_equal(r[0], want_x[i], cases[i])
        assert_equal(r[1], want_z[i], cases[i])


def _bits(p: SymPauli, xpart: Bool) -> UInt64:
    var m = UInt64(0)
    for q in range(p.n):
        var b = p.x(q) if xpart else p.z(q)
        if b == 1:
            m = m | (UInt64(1) << UInt64(q))
    return m


def _fresh(impl: Circuit, nn: Int) raises -> TableauSimulator:
    var sim = TableauSimulator(nn, 0)
    for _ in range(impl.num_clbits):
        sim.creg.append(UInt8(0))
    return sim^


def _inject(mut sim: TableauSimulator, f: Fault):
    if f.p0 == 1:
        sim.x(f.q0)
    elif f.p0 == 2:
        sim.y(f.q0)
    elif f.p0 == 3:
        sim.z(f.q0)
    if f.q1 >= 0:
        if f.p1 == 1:
            sim.x(f.q1)
        elif f.p1 == 2:
            sim.y(f.q1)
        elif f.p1 == 3:
            sim.z(f.q1)


def _check_faults(impl: Circuit, nn: Int, gens: List[SymPauli]) raises -> Int:
    """Every single fault of the circuit-level model: tableau records and state signs must match
    the frame's record mask and anticommutation parities. Returns the number of faults checked."""
    var faults = single_faults(impl)
    var good = _fresh(impl, nn)
    good.run(impl, noisy=False, forced_zero=True)
    var checked = 0
    for f in faults:
        checked += 1
        if f.flip:
            assert_equal(f.x, UInt64(0))
            assert_equal(f.z, UInt64(0))
            assert_equal(f.rec, UInt64(1) << UInt64(f.meas))
            continue
        var sim = _fresh(impl, nn)
        if f.op < 0:
            _inject(sim, f)
            sim.run_range(impl, 0, len(impl.ops), False, True, List[Float64]())
        else:
            sim.run_range(impl, 0, f.op + 1, False, True, List[Float64]())
            _inject(sim, f)
            sim.run_range(impl, f.op + 1, len(impl.ops), False, True, List[Float64]())
        var mask = UInt64(0)
        for m in range(len(sim.records)):
            if sim.records[m] != good.records[m]:
                mask = mask | (UInt64(1) << UInt64(m))
        assert_equal(mask, f.rec, f.label)
        for g0 in gens:
            var g = g0.embed(0, nn)
            var qs = List[Int]()
            var ps = List[Int]()
            for q in range(nn):
                var xq = g.x(q)
                var zq = g.z(q)
                if xq == 0 and zq == 0:
                    continue
                qs.append(q)
                ps.append(2 if (xq == 1 and zq == 1) else (1 if xq == 1 else 3))
            var parity = Int(pop_count(f.x & _bits(g, False))) + Int(pop_count(f.z & _bits(g, True)))
            var expect = -1 if parity % 2 == 1 else 1
            var v = sim.expectation_sparse(qs, ps) * g.sign()
            assert_equal(v, expect, f.label)
    return checked


def test_tableau_cross_check() raises:
    var st = steane_code()
    var gens = st.generators.copy()
    gens.append(SymPauli.from_label("ZZZZZZZ"))
    var total = _check_faults(steane_encoder_zero(), 7, gens)
    total += _check_faults(steane_prep_zero_ft(), 8, gens)
    assert_equal(total, 136 + 183)


def _spec(ops: String) raises -> Circuit:
    """8-qubit circuit (qubit 7 is the flag) from 'G a[ b]' items separated by ';'. M = measure into
    the next bit, R = reset, H S SDG X Y Z on a, CX CZ on a b. Independent Stim numbers use the same list."""
    var parts = ops.split(";")
    var nm = 0
    for o in parts:
        if String(o).startswith("M "):
            nm += 1
    var c = Circuit(8, nm)
    var m = 0
    for o in parts:
        var t = String(o).split(" ")
        var g = String(t[0])
        var a = Int(String(t[1]))
        if g == "H":
            c.h(a)
        elif g == "S":
            c.s(a)
        elif g == "SDG":
            c.sdg(a)
        elif g == "X":
            c.x(a)
        elif g == "Y":
            c.y(a)
        elif g == "Z":
            c.z(a)
        elif g == "CX":
            c.cx(a, Int(String(t[2])))
        elif g == "CZ":
            c.cz(a, Int(String(t[2])))
        elif g == "R":
            c.reset(a)
        elif g == "M":
            c.measure(a, m)
            m += 1
    return c^


def _reset_all() raises -> Circuit:
    """A reset on every qubit, then exactly the ops of steane_prep_zero_ft: same faults, so same counts."""
    var c = Circuit(8, 1)
    for q in range(8):
        c.reset(q)
    c.compose(steane_prep_zero_ft())
    return c^


def _expect_counts(impl: Circuit, model: FaultModel, fs1: Int, u1: Int, fs2: Int, u2: Int,
                   order2: Int, label: String) raises:
    var st = steane_code()
    var sem = Circuit(1)
    var r1 = fault_tolerant_state_preparation(sem, impl, st, 1, 1, model)
    assert_equal(r1.fault_sets, fs1, label)
    assert_equal(r1.undetected, u1, label)
    assert_equal(r1.dangerous, 0, label)
    var r2 = fault_tolerant_state_preparation(sem, impl, st, 1, 2, model)
    assert_equal(r2.fault_sets, fs2, label)
    assert_equal(r2.undetected, u2, label)
    assert_equal(r2.dangerous, 0, label)
    assert_equal(count_undetected(sem, impl, st, 1, 2, model), order2, label)


def test_gate_coverage() raises:
    """CZ, S, SDG, Y, Z, RESET propagation, against Stim brute force."""
    var czflag = _spec("H 0;H 4;H 6;CX 0 1;CX 4 5;CX 6 3;CX 6 5;CX 4 2;CX 0 3;CX 4 1;CX 3 2;S 2;S 2;Z 2;Y 5;Y 5;SDG 6;S 6;H 7;CZ 1 7;CZ 3 7;CZ 5 7;H 7;M 7")
    _expect_counts(czflag, FaultModel.circuit_level(), 210, 109, 20964, 10478, 10369, "czflag circuit_level")
    _expect_counts(czflag, FaultModel.xz_only(), 99, 68, 4773, 2726, 2658, "czflag xz_only")
    var twoflag = _spec("H 0;H 4;H 6;CX 0 1;CX 4 5;CX 6 3;CX 6 5;CX 4 2;CX 0 3;CX 4 1;CX 3 2;CX 1 7;CX 3 7;CX 5 7;M 7;R 7;CX 0 7;CX 1 7;CX 2 7;CX 3 7;M 7")
    _expect_counts(twoflag, FaultModel.circuit_level(), 245, 97, 28551, 7840, 7743, "twoflag_reset circuit_level")
    _expect_counts(twoflag, FaultModel.xz_only(), 107, 64, 5550, 2293, 2229, "twoflag_reset xz_only")
    var ra = _reset_all()
    _expect_counts(ra, FaultModel.circuit_level(), 183, 90, 15672, 7825, 7735, "reset_all circuit_level")
    _expect_counts(ra, FaultModel.xz_only(), 81, 54, 3153, 1756, 1702, "reset_all xz_only")


def _bell(phase: Bool) -> Circuit:
    var c = Circuit(2)
    c.h(0)
    c.cx(0, 1)
    if phase:
        c.s(0)
        c.s(1)
    return c^


def test_phase_bell() raises:
    """Bell state with S on both qubits: stabilizers +YY, +ZZ (CSS, not in the X/Z-pure form)."""
    var r = fault_tolerant_state_preparation(_bell(True), _bell(True), identity_code(2), 0, 1)
    assert_true(r.ok, r.reason)
    assert_equal(r.dangerous, 0)


def test_errors() raises:
    var st = steane_code()
    var hit = 0
    try:
        _ = fault_tolerant_state_preparation(Circuit(1), Circuit(5), five_qubit_code())
    except e:
        assert_true(String(e).startswith("QM-E738"), String(e))
        hit += 1
    try:
        var c = steane_prep_zero_ft()
        c.x(0)
        c.c_if(0, 1)
        _ = fault_tolerant_state_preparation(Circuit(1), c, st, 1)
    except e:
        assert_true(String(e).startswith("QM-E738"), String(e))
        hit += 1
    try:
        var c = steane_encoder_zero()
        c.t(0)
        _ = fault_tolerant_state_preparation(Circuit(1), c, st)
    except e:
        assert_true(String(e).startswith("QM-E738"), String(e))
        hit += 1
    try:
        _ = fault_tolerant_state_preparation(Circuit(1), steane_encoder_zero(), st, 0, 3, FaultModel.circuit_level(), 1000)
    except e:
        assert_true(String(e).startswith("QM-E738"), String(e))
        hit += 1
    try:
        var y = List[SymPauli]()
        y.append(SymPauli.from_label("YYYYYYY"))
        _ = reduced_weight(SymPauli.from_label("IIIIIII"), y)
    except e:
        assert_true(String(e).startswith("QM-E738"), String(e))
        hit += 1
    try:
        var c = Circuit(8, 1)
        c.compose(steane_encoder_zero())
        c.h(7)
        c.measure(7, 0)
        _ = fault_tolerant_state_preparation(Circuit(1), c, st, 1)
    except e:
        assert_true(String(e).startswith("QM-E738"), String(e))
        hit += 1
    assert_equal(hit, 6)


def main() raises:
    test_encoder_not_ft()
    test_flagged_ft()
    test_negative_control()
    test_count_undetected()
    test_reduced_weight()
    test_tableau_cross_check()
    test_gate_coverage()
    test_phase_bell()
    test_errors()
    print("test_ft_faults OK")
