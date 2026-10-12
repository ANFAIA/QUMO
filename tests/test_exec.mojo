from qumo.ir.circuit import Circuit
from qumo.ir.pauli import parse_pauli_sum, PauliSum
from qumo.exec.primitives import EstimatorPub, SamplerPub, run_estimator, run_sampler
from std.testing import assert_equal, assert_true, assert_almost_equal
from std.math import cos, pi


def main() raises:
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
    print("test_exec OK")
