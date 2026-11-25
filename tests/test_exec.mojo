from qumo.ir.circuit import Circuit
from qumo.ir.pauli import parse_pauli_sum, PauliSum
from qumo.exec.primitives import EstimatorPub, EstimatorResult, SamplerPub, run_estimator, run_sampler
from std.testing import assert_equal, assert_true, assert_almost_equal, assert_raises
from std.math import cos, pi, sqrt


def _bell() -> Circuit:
    var c = Circuit(2)
    c.h(0)
    c.cx(0, 1)
    return c^


def _shots_run(c: Circuit, text: String, precision: Float64, seed: UInt64) raises -> EstimatorResult:
    return run_estimator(EstimatorPub(c, [parse_pauli_sum(text)], List[List[Float64]](), precision, seed))


def test_group_std_uses_the_covariance() raises:
    """Terms of one QWC group share their shots: Z0 - Z1 on a Bell pair is 0 on every
    shot, Z0 + Z1 is +-2 (twice the single-term spread), not sqrt(2) times it."""
    var bell = _bell()
    var diff = _shots_run(bell, "Z0 - Z1", 0.02, 1)
    assert_almost_equal(diff.evs[0][0], 0.0)
    assert_true(diff.stds[0][0] < 1e-12, "Z0 - Z1 has no shot noise")
    var single = _shots_run(bell, "Z0", 0.02, 1)
    var summ = _shots_run(bell, "Z0 + Z1", 0.02, 1)
    # Z0 + Z1 asks for 4x the shots of Z0 (L = 2), so the same std 0.02 each way
    assert_equal(summ.shots_per_set, 4 * single.shots_per_set)
    # empirical spread of the estimate over seeds against the reported std
    var n = 40
    var s1 = 0.0
    var s2 = 0.0
    for k in range(n):
        var r = _shots_run(bell, "Z0 + Z1", 0.05, UInt64(100 + k))
        s1 += r.evs[0][0]
        s2 += r.evs[0][0] * r.evs[0][0]
    var mean = s1 / Float64(n)
    var spread = sqrt(max(0.0, s2 / Float64(n) - mean * mean))
    var probe = _shots_run(bell, "Z0 + Z1", 0.05, 100)
    var rep = probe.stds[0][0]
    # every shot is +2 or -2, so the exact std of the estimate is 2/sqrt(S) <= precision
    var exact_std = 2.0 / sqrt(Float64(probe.shots_per_set))
    assert_true(abs(rep - exact_std) < 0.02 * exact_std, "reported std is 2/sqrt(S)")
    assert_true(spread > 0.65 * rep and spread < 1.5 * rep, "empirical spread matches reported std")


def test_precision_is_the_standard_error() raises:
    var c = Circuit(1)
    c.h(0)
    var r = _shots_run(c, "10 * Z0", 0.01, 3)
    assert_true(r.stds[0][0] <= 0.01 * 1.0001, "10*Z with precision 0.01 stays under 0.01")
    assert_true(r.stds[0][0] > 0.009, "and is not wildly conservative")
    assert_true(abs(r.evs[0][0]) < 6 * r.stds[0][0], "10*Z on |+> is 0")
    # sum of |coeff| per group, groups add in quadrature: X0 + Y0 anti-commute = 2 groups of L=1
    var two = _shots_run(c, "X0 + Y0", 0.02, 3)
    assert_equal(two.shots_per_set, 5000)
    assert_true(two.stds[0][0] <= 0.02 * 1.0001, "two groups add in quadrature")
    with assert_raises(contains="QM-E913"):
        _ = _shots_run(c, "10 * Z0", 1e-5, 3)
    with assert_raises(contains="QM-E913"):
        _ = _shots_run(c, "Z0", Float64.MAX, 3)


def test_observable_wider_than_circuit() raises:
    var c = Circuit(1)
    c.h(0)
    with assert_raises(contains="QM-E405"):
        _ = _shots_run(c, "Z5", 0.05, 1)
    with assert_raises(contains="QM-E405"):
        _ = _shots_run(c, "Z5", 0.0, 1)


def test_shots_honour_reset_noise_and_mid_circuit_measurement() raises:
    var r = Circuit(1)
    r.x(0)
    r.reset(0)
    var a = _shots_run(r, "Z0", 0.05, 1)
    assert_almost_equal(a.evs[0][0], 1.0)  # x; reset -> |0>, sample() gives {"0": all}
    # noise: x then depolarize1(0.3): <Z> = -(1 - 4p/3) = -0.6
    var nz = Circuit(1)
    nz.x(0)
    nz.depolarize1(0.3, 0)
    var b = _shots_run(nz, "Z0", 0.01, 1)
    assert_true(abs(b.evs[0][0] + 0.6) < 5 * b.stds[0][0], "noisy value within 5 sigma of -0.6")
    # mid-circuit measurement of |+> then X basis: the measurement collapsed it, <X> = 0, <Z>=0
    var mm = Circuit(1, 1)
    mm.h(0)
    mm.measure(0, 0)
    mm.h(0)
    var d = _shots_run(mm, "Z0", 0.01, 1)  # H then measure then H: Z-basis after collapse -> <H Z H> = <X> on collapsed = 0
    assert_true(abs(d.evs[0][0]) < 5 * d.stds[0][0], "collapsed state")
    # classically controlled flip: measure |1> into c0, X conditioned on it undoes the flip
    var cc = Circuit(1, 1)
    cc.x(0)
    cc.measure(0, 0)
    cc.x(0)
    cc.c_if(0, 1)
    var e = _shots_run(cc, "Z0", 0.05, 1)
    assert_almost_equal(e.evs[0][0], 1.0)


def main() raises:
    test_group_std_uses_the_covariance()
    test_precision_is_the_standard_error()
    test_observable_wider_than_circuit()
    test_shots_honour_reset_noise_and_mid_circuit_measurement()
    var c = Circuit(2)
    var th = c.parameter("theta")
    c.ry(th, 0)
    c.cx(0, 1)
    var obs = List[PauliSum]()
    obs.append(parse_pauli_sum("Z0 Z1"))
    obs.append(parse_pauli_sum("Z0"))
    obs.append(parse_pauli_sum("X0 X1 + 0.5*Y0 Y1"))
    var sets = List[List[Float64]]()
    for k in range(8):
        var s: List[Float64] = [Float64(k) * pi / 7]
        sets.append(s^)
    var pub = EstimatorPub(c, obs, sets)
    var r = run_estimator(pub)
    print(r)
    for k in range(8):
        var t = Float64(k) * pi / 7
        assert_almost_equal(r.evs[k][0], 1.0)
        assert_almost_equal(r.evs[k][1], cos(t))
    var pub2 = EstimatorPub(c, obs, sets, precision=0.01, seed=5)
    var r2 = run_estimator(pub2)
    print(r2.method, r2.shots_per_set)
    for k in range(8):
        for o in range(3):
            assert_true(abs(r2.evs[k][o] - r.evs[k][o]) < 6 * r2.stds[k][o] + 1e-9, "shot estimate within 6 sigma")
    var m = c.copy()
    m.measure_all()
    var sp = SamplerPub(m, 1000, sets)
    var outs = run_sampler(sp)
    assert_equal(len(outs), 8)
    assert_equal(outs[0].counts.get("00", 0), 1000)
    var wide = Circuit(46)
    wide.h(0)
    var obs1 = List[PauliSum]()
    obs1.append(parse_pauli_sum("Z0"))
    var refused = False
    try:
        _ = run_estimator(EstimatorPub(wide, obs1))
    except e:
        refused = String(e).startswith("QM-E404")
    assert_true(refused, "run_estimator refuses 46 qubits with QM-E404")
    print("test_exec OK")
