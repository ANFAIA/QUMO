"""Tests for qumo.compiler.pbc (to_pbc) and pauli_simp(objective=t)."""

from std.math import pi
from std.testing import assert_equal, assert_true
from qumo.ir.circuit import Circuit
from qumo.ir.gates import Op
from qumo.sim.runner import evolve, sample
from qumo.sim.rng import Rng
from qumo.compiler.pbc import to_pbc
from qumo.compiler.pass_manager import compile_spec, transpile
from qumo.compiler.target import Target, parse_target


def random_clifford_t(n: Int, gates: Int, seed: UInt64) -> Circuit:
    var rng = Rng(seed)
    var c = Circuit(n)
    for _ in range(gates):
        var r = rng.below(7)
        var q = rng.below(n)
        var t = (q + 1 + rng.below(n - 1)) % n
        if r == 0:
            c.h(q)
        elif r == 1:
            c.s(q)
        elif r == 2:
            c.t(q)
        elif r == 3:
            c.tdg(q)
        elif r == 4:
            c.rz(rng.uniform() * 3.0, q)
        else:
            c.cx(q, t)
    return c^


def test_unitary_equality() raises:
    for seed in range(12):
        var c = random_clifford_t(5, 60, UInt64(seed + 1))
        var prog = to_pbc(c)
        var back = prog.to_circuit()
        var sa = evolve(c)
        var sb = evolve(back)
        var worst = 0.0
        for i in range(sa.size()):
            worst = max(worst, abs(sa.re[i] - sb.re[i]), abs(sa.im[i] - sb.im[i]))
        assert_true(worst < 1e-10, "to_pbc keeps the unitary with its phase, seed " + String(seed))
    print("  rotations + final Clifford reproduce the circuit exactly")


def test_all_cliffords_move_to_the_end() raises:
    var c = Circuit(2)
    c.h(0)
    c.t(0)
    c.cx(0, 1)
    c.t(1)
    var prog = to_pbc(c)
    assert_equal(len(prog.ops), 2, "two rotations")
    # T on q0 after H is a rotation about X; T on q1 after H, CX has Pauli Z0 Z1 ... conjugated back
    assert_equal(prog.ops[0].label(2), "IX", "first T: H moved past it")
    assert_equal(prog.ops[1].label(2), "ZX", "second T: H and CX moved past it")
    var d = prog.counts()
    assert_equal(d["pi8"], 2, "two pi/8 rotations")
    assert_equal(d["max_weight"], 2, "weight")
    assert_equal(d["layers"], 1, "IX and ZX commute")
    print("  Pauli labels and counts")


def test_measurements() raises:
    # distribution equality with mid-circuit measurements
    for seed in range(4):
        var c = random_clifford_t(4, 30, UInt64(seed + 50))
        c.measure(0, 0)
        c.h(0)
        c.cx(0, 2)
        c.t(2)
        c.h(2)
        c.measure(2, 1)
        c.measure(1, 2)
        c.measure(3, 3)
        var prog = to_pbc(c, fix_clifford=False)
        assert_equal(prog.counts()["measurements"], 4, "four PPMs")
        var a = sample(c, 20000, 7)
        var b = sample(prog.to_circuit(), 20000, 9)
        var tv = 0.0
        for kv in a.counts.items():
            tv += abs(a.probability(kv.key) - b.probability(kv.key))
        for kv in b.counts.items():
            if kv.key not in a.counts:
                tv += b.probability(kv.key)
        assert_true(tv / 2 < 0.03, "same outcomes, TV " + String(tv / 2))
    print("  measurements become Pauli-product measurements")


def test_errors() raises:
    var c = Circuit(1, 1)
    c.h(0)
    c.reset(0)
    var raised = False
    try:
        _ = to_pbc(c)
    except e:
        raised = String(e).startswith("QM-E623")
    assert_true(raised, "QM-E623 on reset")
    print("  error codes")


def toffoli_ladder(n: Int) -> Circuit:
    """A reversible circuit where rotations meet across Hadamards."""
    var c = Circuit(n)
    for _ in range(3):
        for q in range(n - 2):
            c.ccx(q, q + 1, q + 2)
            c.cx(q + 2, q)
        for q in range(n - 1):
            c.ccx(q, (q + 2) % n, q + 1)
    return c^


def test_objective_t() raises:
    # a T pair separated only by Clifford conjugation: same Pauli, phase_fold cannot see it
    # through the Hadamard, the T objective merges them
    var c = Circuit(2)
    c.h(0)
    c.t(0)
    c.h(0)
    c.cx(1, 0)
    c.cx(1, 0)
    c.h(0)
    c.t(0)
    c.h(0)
    var t = Target.all_to_all(2)
    var r = compile_spec(c, t, "pauli_simp(objective=t)")
    assert_true(r.circuit.t_count() < c.t_count() or r.circuit.count(Op.T) < 2, "merged the two T gates")
    # the default objective still needs 2q savings, and O0-O3 stay as they are
    assert_true(transpile(c, t, 2).pipeline.find("objective") < 0, "not in presets")
    var ladder = toffoli_ladder(6)
    var base = compile_spec(ladder, t, "unroll3q,fixpoint(cancel,merge,commute,phase_fold,cancel,merge,max=6)")
    var obj = compile_spec(ladder, t, "unroll3q,pauli_simp(objective=t),fixpoint(cancel,merge,commute,phase_fold,cancel,merge,max=6)")
    assert_true(obj.circuit.t_count() <= base.circuit.t_count(), "objective t is not worse in T")
    var sa = evolve(ladder)
    var sb = evolve(obj.circuit)
    assert_true(sa.fidelity(sb) > 1 - 1e-9, "still the same unitary action on |0>")
    var bad = False
    try:
        _ = compile_spec(c, t, "pauli_simp(objective=x)")
    except e:
        bad = String(e).startswith("QM-E602")
    assert_true(bad, "unknown objective")
    print("  pauli_simp(objective=t): T", base.circuit.t_count(), "->", obj.circuit.t_count())


def test_e604_names_the_input_gate() raises:
    var c = Circuit(3)
    c.h(0)
    c.rz(0.3, 1)
    c.cx(0, 1)
    c.h(1)
    var raised = False
    for lvl in range(4):
        try:
            _ = transpile(c, parse_target("line:3/clifford_t"), lvl)
        except e:
            var s = String(e)
            raised = s.startswith("QM-E604") and s.find("rz(0.3) q1") >= 0 and s.find("instruction #1") >= 0 and s.find("not universal") >= 0
            assert_true(raised, "level " + String(lvl) + ": " + s)
    assert_true(raised, "QM-E604 raised")
    print("  QM-E604 names rz(0.3) q1, instruction #1")


def main() raises:
    test_unitary_equality()
    test_all_cliffords_move_to_the_end()
    test_measurements()
    test_errors()
    test_objective_t()
    test_e604_names_the_input_gate()
    print("test_pbc OK")
