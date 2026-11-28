"""Tests for qumo.sim.mps (matrix product state simulator).

The reference is the dense state vector (`qumo.sim.runner.evolve`), a different simulator
with its own gate kernels: every amplitude, Pauli expectation values and sampled
frequencies are compared on seeded random circuits that use every 1-, 2- and 3-qubit
opcode on adjacent and non-adjacent qubits in both operand orders. Wider than the
state-vector limit (32 qubits), GHZ states have a closed-form answer.
"""

from std.math import sqrt, pi
from std.testing import assert_true, assert_equal
from qumo.core.rng import Rng
from qumo.ir.circuit import Circuit
from qumo.ir.gates import Op, matrix_1q, matrix_2q
from qumo.ir.pauli import PauliSum
from qumo.sim.runner import evolve, sample, MAX_SV_QUBITS
from qumo.sim.statevector import StateVector
from qumo.sim.mps import MPS, mps_evolve, MPS_MAX_BOND, MPS_MAX_QUBITS


def random_circuit(n: Int, gates: Int, seed: UInt64, three_q: Bool = True) raises -> Circuit:
    var rng = Rng(seed)
    var c = Circuit(n)
    var one: List[Int] = [Op.I, Op.X, Op.Y, Op.Z, Op.H, Op.S, Op.SDG, Op.T, Op.TDG, Op.SX, Op.SXDG,
                          Op.RX, Op.RY, Op.RZ, Op.P, Op.U]
    var two = List[Int]()
    for k in range(Op.CX, Op.RZZ + 1):
        two.append(k)
    var three: List[Int] = [Op.CCX, Op.CSWAP, Op.CCZ]
    for _ in range(gates):
        var u = rng.uniform()
        var a = rng.below(n)
        if n == 1 or u < 0.4:
            var k = one[rng.below(len(one))]
            c.add(k, a, -1, -1, 2 * pi * rng.uniform(), 2 * pi * rng.uniform(), 2 * pi * rng.uniform())
        elif n < 3 or not three_q or u < 0.92:
            var b = rng.below(n - 1)
            if b >= a:
                b += 1
            var k = two[rng.below(len(two))]
            c.add(k, a, b, -1, 2 * pi * rng.uniform())
        else:
            var b = rng.below(n - 1)
            if b >= a:
                b += 1
            var d = rng.below(n)
            while d == a or d == b:
                d = rng.below(n)
            c.add(three[rng.below(3)], a, b, d)
    c.global_phase = 0.3
    return c^


def state_distance(sv: StateVector, st: MPS) raises -> Float64:
    """||psi - phi||_2 over all 2^n amplitudes."""
    var d = 0.0
    for i in range(sv.size()):
        var a = st.amplitude_index(i)
        d += (a[0] - sv.re[i]) ** 2 + (a[1] - sv.im[i]) ** 2
    return sqrt(d)


def random_pauli_label(n: Int, mut rng: Rng) -> String:
    var s = String("")
    for _ in range(n):
        var k = rng.below(4)
        s += "I" if k == 0 else ("X" if k == 1 else ("Y" if k == 2 else "Z"))
    return s^


def expect_code(code: String, msg: String) raises:
    assert_true(msg.startswith(code), "expected " + code + ", got: " + msg)


def test_exact_random() raises:
    """No truncation: amplitudes and expectations equal the state vector to rounding."""
    var worst = 0.0
    var cases: List[Tuple[Int, Int]] = [(1, 12), (2, 30), (3, 40), (4, 60), (5, 80), (7, 120), (9, 160)]
    var seed: UInt64 = 1
    for cs in cases:
        for rep in range(3):
            var n = cs[0]
            var c = random_circuit(n, cs[1], seed)
            seed += 1
            var sv = evolve(c)
            var st = mps_evolve(c, max_bond=MPS_MAX_BOND, cutoff=0.0)
            var d = state_distance(sv, st)
            worst = max(worst, d)
            assert_true(d <= 1e-11, "exact n=" + String(n) + " rep " + String(rep) + ": distance " + String(d))
            assert_equal(st.discarded_weight, 0.0, "nothing truncated")
            assert_equal(st.error_bound, 0.0, "nothing truncated")
            assert_true(abs(st.norm_sq() - 1.0) <= 1e-10, "norm " + String(st.norm_sq()) + " n=" + String(n) + " d=" + String(d))
            var rng = Rng(seed * 7)
            for _ in range(4):
                var lab = random_pauli_label(n, rng)
                var h = PauliSum()
                h.add_label(1.0, lab)
                var e_sv = sv.expectation(h)
                var e1 = st.expectation_label(lab)
                var e2 = st.expectation(h)
                assert_true(abs(e1 - e_sv) <= 1e-11, "expectation " + lab + ": " + String(e1) + " vs " + String(e_sv))
                assert_true(abs(e2 - e_sv) <= 1e-11, "expectation (PauliSum) " + lab)
            # Bond dimensions never exceed the Schmidt-rank limit 2^min(k + 1, n - k - 1).
            var dims = st.bond_dims()
            for k in range(len(dims)):
                assert_true(dims[k] <= (1 << min(k + 1, n - k - 1)), "bond " + String(k))
    print("  exact random circuits: max ||psi - phi|| =", worst)


def test_bit_order_and_sampling() raises:
    """Keys and amplitudes follow QUMO's order: qubit k = bit k, leftmost key char = top qubit."""
    var c = Circuit(3)
    c.x(0)
    var st = mps_evolve(c)
    var a = st.amplitude_index(1)
    assert_true(abs(a[0] - 1.0) < 1e-15, "X on qubit 0 is index 1")
    var counts = st.sample(10, 5)
    assert_equal(counts.get("001", 0), 10, "key of X on qubit 0 is 001")
    var c2 = Circuit(3)
    c2.x(0)
    c2.measure_all()
    var res = sample(c2, 10, seed=5)
    assert_equal(res.counts.get("001", 0), 10, "the state-vector sampler uses the same key")

    # Frequencies against exact probabilities (5 sigma), on a non-trivial 6-qubit state.
    var c3 = random_circuit(6, 70, 99)
    var sv = evolve(c3)
    var st3 = mps_evolve(c3, max_bond=MPS_MAX_BOND, cutoff=0.0)
    var shots = 40000
    var cnt = st3.sample(shots, 1234)
    var cnt2 = st3.sample(shots, 1234)
    var total = 0
    for i in range(64):
        var key = String("")
        for q in range(5, -1, -1):
            key += "1" if (i >> q) & 1 else "0"
        var p = sv.re[i] ** 2 + sv.im[i] ** 2
        var f = Float64(cnt.get(key, 0)) / Float64(shots)
        var sigma = sqrt(max(p * (1 - p), 1e-6) / Float64(shots))
        assert_true(abs(f - p) <= 5 * sigma + 1e-9, "frequency of " + key + ": " + String(f) + " vs " + String(p))
        assert_equal(cnt.get(key, 0), cnt2.get(key, 0), "same seed, same counts")
        total += cnt.get(key, 0)
    assert_equal(total, shots)


def test_truncation_bound() raises:
    """With a small max_bond: ||psi - phi|| <= error_bound and ||phi||^2 = 1 - discarded_weight."""
    var hits = 0
    for seed in range(5):
        var n = 8
        var c = random_circuit(n, 150, UInt64(500 + seed), three_q=False)
        var sv = evolve(c)
        for chi in [2, 4, 8]:
            var st = mps_evolve(c, max_bond=chi, cutoff=0.0)
            var d = state_distance(sv, st)
            assert_true(d <= st.error_bound + 1e-10, "chi " + String(chi) + ": distance " + String(d) + " > bound " + String(st.error_bound))
            assert_true(abs(st.norm_sq() - (1.0 - st.discarded_weight)) <= 1e-10, "norm vs discarded weight")
            if st.truncations > 0:
                hits += 1
                assert_true(st.error_bound > 0.0, "a truncation is reported")
            # Expectation values within 4 delta.
            var rng = Rng(UInt64(seed * 31 + chi))
            for _ in range(3):
                var lab = random_pauli_label(n, rng)
                var h = PauliSum()
                h.add_label(1.0, lab)
                var e = st.expectation_label(lab)
                assert_true(abs(e - sv.expectation(h)) <= 4 * st.error_bound + 1e-10, "expectation within 4 delta")
            for k in range(n - 1):
                assert_true(st.bond_dims()[k] <= chi, "max_bond respected")
    assert_true(hits >= 10, "the small bonds truncated")
    # A cutoff truncates too, and is reported.
    var c = random_circuit(8, 150, 77, three_q=False)
    var st = mps_evolve(c, max_bond=64, cutoff=0.2)
    assert_true(st.truncations > 0 and st.error_bound > 0.0, "cutoff truncation reported")
    assert_true(state_distance(evolve(c), st) <= st.error_bound + 1e-10, "cutoff bound")


def test_wide_ghz() raises:
    """GHZ on 64 and 100 qubits (state vector limit: 32): closed-form answers."""
    for n in [64, 100]:
        assert_true(n > MAX_SV_QUBITS)
        var c = Circuit(n)
        c.h(0)
        for q in range(n - 1):
            c.cx(q, q + 1)
        var st = mps_evolve(c, max_bond=8, cutoff=1e-12)
        assert_equal(st.error_bound, 0.0)
        for d in st.bond_dims():
            assert_equal(d, 2, "GHZ bond dimension")
        var zeros = List[Int](length=n, fill=0)
        var ones = List[Int](length=n, fill=1)
        var a0 = st.amplitude(zeros)
        var a1 = st.amplitude(ones)
        var r = 1.0 / sqrt(2.0)
        assert_true(abs(a0[0] - r) < 1e-12 and abs(a0[1]) < 1e-12, "<0..0|GHZ>")
        assert_true(abs(a1[0] - r) < 1e-12 and abs(a1[1]) < 1e-12, "<1..1|GHZ>")
        var mixed = List[Int](length=n, fill=0)
        mixed[n // 2] = 1
        var am = st.amplitude(mixed)
        assert_true(abs(am[0]) < 1e-14 and abs(am[1]) < 1e-14, "other amplitudes vanish")
        var xs = String("")
        var zz = String("")
        for q in range(n):
            xs += "X"
            zz += "Z" if (q == 0 or q == n - 1) else "I"
        assert_true(abs(st.expectation_label(xs) - 1.0) < 1e-12, "<X...X> = 1")
        assert_true(abs(st.expectation_label(zz) - 1.0) < 1e-12, "<Z_0 Z_{n-1}> = 1")
        var z0 = List[Int](length=n, fill=0)
        z0[0] = 3
        assert_true(abs(st.expectation_ops(z0)) < 1e-12, "<Z_0> = 0")
        var counts = st.sample(2000, 3)
        var all0 = String("")
        var all1 = String("")
        for _ in range(n):
            all0 += "0"
            all1 += "1"
        var c0 = counts.get(all0, 0)
        var c1 = counts.get(all1, 0)
        assert_equal(c0 + c1, 2000, "only 0...0 and 1...1")
        assert_true(abs(Float64(c0) - 1000.0) <= 5 * sqrt(500.0), "balanced")
    # Long-range gates (swaps through the chain) and a 3-qubit gate on 40 qubits:
    # CX(0, q) for every q makes GHZ; CCX(0, 39, 20) then flips qubit 20 on the 1...1 branch.
    var n = 40
    var c = Circuit(n)
    c.h(0)
    for q in range(n - 1, 0, -1):
        c.cx(0, q)
    c.ccx(0, n - 1, 20)
    var st = mps_evolve(c, max_bond=16)
    var b1 = List[Int](length=n, fill=1)
    b1[20] = 0
    var amp = st.amplitude(b1)
    assert_true(abs(amp[0] - 1.0 / sqrt(2.0)) < 1e-12, "long-range CX and CCX")
    assert_true(st.max_bond_seen <= 4, "swaps keep GHZ bonds small: " + String(st.max_bond_seen))


def test_errors() raises:
    var c = Circuit(2)
    c.h(0)
    c.measure(0, 0)
    c.x(0)
    try:
        _ = mps_evolve(c)
        raise Error("mid-circuit measurement accepted")
    except e:
        expect_code("QM-E432", String(e))
    var c2 = Circuit(2)
    c2.reset(0)
    try:
        _ = mps_evolve(c2)
        raise Error("reset accepted")
    except e:
        expect_code("QM-E432", String(e))
    var c3 = Circuit(2)
    c3.depolarize1(0.1, 0)
    try:
        _ = mps_evolve(c3)
        raise Error("noise accepted")
    except e:
        expect_code("QM-E432", String(e))
    var c4 = Circuit(2)
    c4.measure(0, 0)
    c4.x(1)
    c4.c_if(0, 1)
    try:
        _ = mps_evolve(c4)
        raise Error("classical control accepted")
    except e:
        expect_code("QM-E432", String(e))
    # Terminal measurements are fine.
    var c5 = Circuit(2)
    c5.h(0)
    c5.cx(0, 1)
    c5.measure_all()
    var st = mps_evolve(c5)
    assert_true(abs(st.expectation_label("ZZ") - 1.0) < 1e-12)
    for bad in [(0, 4, 0.0), (3, 0, 0.0), (3, MPS_MAX_BOND + 1, 0.0), (3, 4, 1.0), (3, 4, -1e-3),
                (MPS_MAX_QUBITS + 1, 4, 0.0), (4096, 512, 0.0)]:
        try:
            _ = MPS(bad[0], bad[1], bad[2])
            raise Error("bad options accepted")
        except e:
            expect_code("QM-E433", String(e))
    var m = MPS(3)
    for q in [-1, 3]:
        try:
            m.apply_1q(matrix_1q(Op.H), q)
            raise Error("bad qubit accepted")
        except e:
            expect_code("QM-E434", String(e))
    try:
        m.apply_2q(matrix_2q(Op.CX), 1, 1)
        raise Error("repeated qubit accepted")
    except e:
        expect_code("QM-E434", String(e))
    try:
        _ = m.amplitude(List[Int](length=2, fill=0))
        raise Error("short bit list accepted")
    except e:
        expect_code("QM-E434", String(e))
    try:
        _ = m.expectation_label("ZZ")
        raise Error("short label accepted")
    except e:
        expect_code("QM-E434", String(e))
    try:
        _ = m.expectation_label("ZAZ")
        raise Error("bad label accepted")
    except e:
        expect_code("QM-E434", String(e))
    try:
        _ = m.sample(0, 1)
        raise Error("zero shots accepted")
    except e:
        expect_code("QM-E433", String(e))
    var h = PauliSum()
    h.add_label(1.0, "ZIII")
    try:
        _ = m.expectation(h)
        raise Error("wide observable accepted")
    except e:
        expect_code("QM-E405", String(e))
    # Review findings (2026-11-27). Sampling memory: shots * num_qubits is capped
    # (4096 qubits x 100000 shots took 1.7 GB of keys); refused before any work.
    var wide = MPS(4096, 1, 0.0)
    for shots in [100000, (1 << 27) // 4096 + 1]:
        try:
            _ = wide.sample(shots, 1)
            raise Error(String(shots) + " shots of 4096 qubits accepted")
        except e:
            expect_code("QM-E433", String(e))
    var ok = wide.sample(8, 1)
    assert_equal(ok.get(String("0") * 4096, 0), 8, "|0...0> samples one key")
    # Non-finite angles and global phase were run and gave NaN amplitudes with
    # error_bound 0 and plausible counts; they are refused (QM-E482).
    var nan = Float64(0.0) / Float64(0.0)
    var inf = Float64(1.0) / Float64(0.0)
    for which in range(4):
        var cn = Circuit(2)
        cn.h(0)
        if which == 0:
            cn.rx(nan, 1)
        elif which == 1:
            cn.u(0.1, inf, 0.2, 0)
        elif which == 2:
            cn.rzz(-inf, 0, 1)
        else:
            cn.global_phase = nan
        try:
            _ = mps_evolve(cn)
            raise Error("non-finite value accepted, case " + String(which))
        except e:
            expect_code("QM-E482", String(e))
    print("  coded errors ok")


def main() raises:
    test_exact_random()
    test_bit_order_and_sampling()
    test_truncation_bound()
    test_wide_ghz()
    test_errors()
    print("test_mps OK")
