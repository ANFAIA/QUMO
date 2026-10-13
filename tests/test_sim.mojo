from qumo.ir.circuit import Circuit
from qumo.ir.gates import Op, matrix_1q
from qumo.ir.pauli import parse_pauli_sum, PauliSum
from qumo.sim.statevector import StateVector
from qumo.sim.runner import evolve, sample, expectation, terminal_measure_map
from qumo.sim.stabilizer import sample_stabilizer
from qumo.sim.rng import Rng
from std.testing import assert_equal, assert_true, assert_almost_equal
from std.math import pi, sqrt


def _only(c: Circuit, key: String, method: String) raises:
    """Every shot of sample() must give `key` via `method`."""
    var r = sample(c, 64, seed=3)
    assert_equal(r.method, method)
    assert_equal(r.counts.get(key, 0), 64, "expected only " + key + " via " + method + ", got " + String(r))


def _key(creg: Int, width: Int) -> String:
    var s = String()
    var b = width - 1
    while b >= 0:
        s += "1" if (creg >> b) & 1 == 1 else "0"
        b -= 1
    return s


def test_repeated_clbit_last_write_wins() raises:
    """Regression: a clbit written by several terminal MEASUREs keeps the LAST
    one (Qiskit-Aer semantics) on the static fast path, exactly like the
    per-shot trajectory path and the stabilizer sampler. The static path used
    to OR bits in (an earlier 1 survived a later 0)."""
    # reported case: x q0; m q0->c1; m q1->c1; m q0->c0  ==> '01' (Aer: 01)
    var c1 = Circuit(2, 2)
    c1.x(0)
    c1.measure(0, 1)
    c1.measure(1, 1)
    c1.measure(0, 0)
    _only(c1, "01", "statevector:static")
    var t1 = c1.copy()
    t1.x(0)  # gate after a measurement -> per-shot trajectories
    t1.x(0)
    _only(t1, "01", "statevector:trajectories")
    assert_equal(sample_stabilizer(c1, 32, seed=1).counts.get("01", 0), 32)
    # without the third measurement ==> '00' (Aer: 00)
    var c2 = Circuit(2, 2)
    c2.x(0)
    c2.measure(0, 1)
    c2.measure(1, 1)
    _only(c2, "00", "statevector:static")
    # a later 0 overwrites an earlier 1 on clbit 0 ==> '00'
    var c3 = Circuit(2, 2)
    c3.x(1)
    c3.measure(1, 0)
    c3.measure(0, 0)
    _only(c3, "00", "statevector:static")
    # non-Clifford state, overwritten value never leaks
    var c4 = Circuit(2, 2)
    c4.ry(1.1, 0)
    c4.t(0)
    c4.measure(0, 1)
    c4.measure(1, 1)
    _only(c4, "00", "statevector:static")
    # same qubit into two clbits: perfectly correlated
    var c5 = Circuit(1, 2)
    c5.h(0)
    c5.measure(0, 0)
    c5.measure(0, 1)
    var r5 = sample(c5, 400, seed=5)
    assert_equal(r5.counts.get("01", 0) + r5.counts.get("10", 0), 0)
    assert_true(r5.counts.get("00", 0) > 100 and r5.counts.get("11", 0) > 100)
    # the map itself: one source per clbit, identity fast path for measure_all
    var mm = terminal_measure_map(c1)
    assert_equal(len(mm.clbits), 2)
    assert_equal(mm.qubits[0], 0)  # c0 <- q0 (last write)
    assert_equal(mm.qubits[1], 1)  # c1 <- q1 (last write)
    assert_equal(mm.pack(0b01), 0b01)
    var ma = Circuit(3)
    ma.measure_all()
    var mi = terminal_measure_map(ma)
    assert_equal(mi.mask, 0b111)
    assert_equal(mi.pack(0b101), 0b101)
    var mr = Circuit(3, 4)
    mr.measure(2, 0)
    mr.measure(0, 3)
    var mp = terminal_measure_map(mr)
    assert_equal(mp.mask, -1)
    assert_equal(mp.pack(0b100), 0b0001)
    assert_equal(mp.pack(0b001), 0b1000)
    assert_equal(mp.pack(0b111), 0b1001)

    # randomized basis-state circuits: static == trajectories == stabilizer ==
    # last-write-wins reference
    var rng = Rng(77)
    for trial in range(150):
        var n = 1 + rng.below(5)
        var nc = 1 + rng.below(5)
        var c = Circuit(n, nc)
        var state = 0
        for q in range(n):
            if rng.below(2) == 1:
                c.x(q)
                state |= 1 << q
        var expect = 0
        var nm = 1 + rng.below(8)
        for _ in range(nm):
            var q = rng.below(n)
            var cb = rng.below(nc)
            c.measure(q, cb)
            if (state >> q) & 1 == 1:
                expect |= 1 << cb
            else:
                expect &= ~(1 << cb)
        var key = _key(expect, nc)
        _only(c, key, "statevector:static")
        var s = sample_stabilizer(c, 16, seed=1)
        assert_equal(s.counts.get(key, 0), 16, "stabilizer trial " + String(trial))
        var meas_q0 = False
        for g in c.ops:
            if g.kind == Op.MEASURE and g.q0 == 0:
                meas_q0 = True
        if meas_q0:
            var dyn = c.copy()
            dyn.x(0)
            dyn.x(0)
            _only(dyn, key, "statevector:trajectories")


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
    test_repeated_clbit_last_write_wins()
    print("test_sim OK")
