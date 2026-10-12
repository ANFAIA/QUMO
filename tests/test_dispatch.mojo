from qumo.ir.circuit import Circuit
from qumo.exec.dispatch import sample_auto, choose_method
from qumo.sim.runner import sample
from std.testing import assert_equal, assert_true


def main() raises:
    # noisy GHZ: frame sampler vs state-vector trajectories agree statistically
    var c = Circuit(4)
    c.h(0)
    for q in range(3):
        c.cx(q, q + 1)
        c.depolarize2(0.02, q, q + 1)
    for q in range(4):
        c.x_error(0.03, q)
    c.measure_all()
    assert_equal(choose_method(c, 20000), "frame")
    var f = sample_auto(c, 20000, 3)
    var t = sample(c, 20000, 4)
    assert_equal(f.method, "stabilizer:pauli_frame")
    for k in t.counts.keys():
        var pf = f.probability(k)
        var pt = t.probability(k)
        assert_true(abs(pf - pt) < 0.015, "frame vs trajectories mismatch on " + k)
    # large clean Clifford -> tableau; non-Clifford -> state vector
    var g = Circuit(50)
    g.h(0)
    for q in range(49):
        g.cx(q, q + 1)
    g.measure(0, 0)
    g.measure(49, 1)
    assert_equal(choose_method(g, 100), "stabilizer")
    var r = sample_auto(g, 100, 1)
    assert_equal(r.counts.get("00", 0) + r.counts.get("11", 0), 100)
    var nc = Circuit(2)
    nc.t(0)
    nc.measure_all()
    assert_equal(choose_method(nc, 100), "statevector")
    print("test_dispatch OK")
