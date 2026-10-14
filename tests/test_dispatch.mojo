from qumo.ir.circuit import Circuit
from qumo.exec.dispatch import sample_auto, choose_method, sample_on
from qumo.sim.runner import sample
from qumo.gpu.device import gpu_compiled, GPU_PRECISION_ENV
from std.os import setenv, unsetenv
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
    # GPU precision selector: validated on every device (QM-E941); CPU runs ignore it (fp64)
    var note = String()
    for p in ["auto", "fp32", "fp64", "FP64"]:
        var rp = sample_on(nc, 200, 5, List[Float64](), "cpu", note, String(p))
        assert_equal(rp.method, "statevector:static")
        assert_equal(rp.counts.get("00", 0) + rp.counts.get("01", 0), 200)
    var raised = False
    try:
        _ = sample_on(nc, 10, 5, List[Float64](), "auto", note, "fp16")
    except e:
        raised = String(e).startswith("QM-E941")
    assert_true(raised, "QM-E941 expected for an unknown precision")
    raised = False
    try:
        _ = sample_auto(nc, 10, 5, List[Float64](), "single")
    except e:
        raised = String(e).startswith("QM-E941")
    assert_true(raised, "QM-E941 expected from sample_auto")
    raised = False
    try:  # an unknown device is reported before the precision
        _ = sample_on(nc, 10, 5, List[Float64](), "tpu", note, "fp16")
    except e:
        raised = String(e).startswith("QM-E940")
    assert_true(raised, "QM-E940 expected")
    if not gpu_compiled():
        # explicit GPU request without GPU support: CPU fallback with a note, whatever the precision
        _ = setenv(GPU_PRECISION_ENV, "fp64")
        var rg = sample_on(nc, 100, 5, List[Float64](), "gpu", note, "fp64")
        assert_equal(rg.method, "statevector:static")
        assert_true(note.find("not compiled") >= 0, note)
        _ = unsetenv(GPU_PRECISION_ENV)
    print("test_dispatch OK")
