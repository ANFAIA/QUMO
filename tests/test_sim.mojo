from qumo.ir.circuit import Circuit
from qumo.ir.gates import Op, matrix_1q
from qumo.ir.pauli import parse_pauli_sum, PauliSum
from qumo.sim.statevector import StateVector
from qumo.sim.runner import evolve, sample, expectation
from std.testing import assert_equal, assert_true, assert_almost_equal
from std.math import pi, sqrt


def main() raises:
    # Bell state
    var c = Circuit(2)
    c.h(0)
    c.cx(0, 1)
    var sv = evolve(c)
    assert_almost_equal(sv.re[0], 1 / sqrt(2.0))
    assert_almost_equal(sv.re[3], 1 / sqrt(2.0))
    assert_almost_equal(sv.expectation(parse_pauli_sum("Z0 Z1")), 1.0)
    assert_almost_equal(sv.expectation(parse_pauli_sum("X0 X1")), 1.0)
    assert_almost_equal(sv.expectation(parse_pauli_sum("Y0 Y1")), -1.0)
    assert_almost_equal(sv.expectation(parse_pauli_sum("0.5*ZI + 2*XX")), 2.0)
    c.measure_all()
    var r = sample(c, 2000, seed=7)
    print(r, r.method)
    assert_true(r.counts.get("01", 0) == 0 and r.counts.get("10", 0) == 0)
    assert_true(abs(r.probability("00") - 0.5) < 0.05)

    # GHZ-20 with SIMD + parallel path
    var n = 20
    var g = Circuit(n)
    g.h(0)
    for q in range(n - 1):
        g.cx(q, q + 1)
    var s2 = evolve(g)
    assert_almost_equal(s2.re[0], 1 / sqrt(2.0))
    assert_almost_equal(s2.re[(1 << n) - 1], 1 / sqrt(2.0))
    assert_almost_equal(s2.norm_sq(), 1.0)

    # every gate: U U^dagger = I on random-ish state, on high and low qubits
    var kinds = List[Int]()
    for k in range(Op.CCZ + 1):
        kinds.append(k)
    for k in kinds:
        var t = Circuit(12)
        for q in range(12):
            t.ry(0.1 * Float64(q + 1), q)
            t.rz(0.37 * Float64(q), q)
        for q in range(11):
            t.cx(q, q + 1)
        var a = 0
        var b = 11
        var cc = 5
        var probe = Circuit(12)
        probe.add(k, a, b, cc, 0.7, -0.3, 1.1)
        probe.add(k, b, cc, a, 0.2, 0.5, -0.9)
        var full = t.copy()
        full.compose(probe)
        full.compose(probe.inverse())
        var ref_ = evolve(t)
        var out = evolve(full)
        var f = ref_.fidelity(out)
        if abs(f - 1.0) > 1e-9:
            raise Error("gate inverse failed for opcode " + String(k) + " fidelity=" + String(f))
    # mid-circuit measurement + feed-forward teleportation
    var tp = Circuit(3)
    tp.ry(1.234, 0)
    tp.h(1)
    tp.cx(1, 2)
    tp.cx(0, 1)
    tp.h(0)
    tp.measure(0, 0)
    tp.measure(1, 1)
    tp.x(2)
    tp.c_if(1, 1)
    tp.z(2)
    tp.c_if(0, 1)
    tp.ry(-1.234, 2)
    tp.measure(2, 2)
    var tr = sample(tp, 500, seed=3)
    print("teleport", tr, tr.method)
    for k in tr.counts.keys():
        assert_true(k.startswith("0"), "teleported qubit must return to |0>")
    # noise: x_error p=1 flips deterministically
    var nz = Circuit(1)
    nz.x_error(1.0, 0)
    nz.measure(0, 0)
    assert_equal(sample(nz, 10).counts.get("1", 0), 10)
    print("test_sim OK")
