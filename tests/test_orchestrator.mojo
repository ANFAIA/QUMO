from qumo.ir.circuit import Circuit
from qumo.ir.pauli import parse_pauli_sum
from qumo.exec.orchestrator import Workflow, Device, job_precision
from qumo.exec.resources import estimate_resources, estimate_hardware, estimate_fault_tolerant, HardwareProfile
from std.testing import assert_equal, assert_true, assert_almost_equal


def ghz(n: Int) -> Circuit:
    var c = Circuit(n)
    c.h(0)
    for q in range(n - 1):
        c.cx(q, q + 1)
    return c^


def main() raises:
    var wf = Workflow()
    var a = wf.add_estimate("zz", ghz(4), parse_pauli_sum("Z0 Z3"))
    var g = ghz(6)
    g.measure_all()
    var deps: List[Int] = [a]
    var b = wf.add_sample("ghz6", g, 1000, deps=deps)
    var t = Circuit(8)
    t.h(0)
    t.t(0)
    t.measure_all()
    _ = wf.add_sample("t8", t, 500)
    var big = ghz(40)
    big.measure_all()
    _ = wf.add_sample("ghz40-clifford", big, 100)
    var pool = List[Device]()
    pool.append(Device.cpu("cpu0"))
    pool.append(Device.stabilizer("tableau0"))
    pool.append(Device.qpu("ion-qpu", 36, 60.0, 0.001, 0.30, 0.0001))
    var sch = wf.plan(pool)
    print(sch)
    var res = wf.run(pool)
    # the 40-qubit Clifford job is dispatched to the tableau simulator
    assert_true(res[3].ok, res[3].error)
    assert_equal(res[3].samples.counts.get("0" * 40, 0) + res[3].samples.counts.get("1" * 40, 0), 100)
    assert_almost_equal(res[a].expectation, 1.0)
    assert_equal(res[b].samples.counts.get("000000", 0) + res[b].samples.counts.get("111111", 0), 1000)
    for r in res:
        assert_true(r.ok, r.error)
    var re = estimate_resources(ghz(30))
    print(re)
    var tc = Circuit(3)
    tc.ccx(0, 1, 2)
    tc.t(0)
    print(estimate_hardware(ghz(10), HardwareProfile.superconducting()))
    print(estimate_fault_tolerant(tc))
    # GPU precision: per device and per job ("auto" = the device's); unknown selectors fail in plan
    var gdev = Device.gpu("gpu0", 30, "fp64")
    assert_equal(gdev.precision, "fp64")
    assert_true(String(gdev).find("precision=fp64") >= 0, String(gdev))
    assert_equal(Device.gpu().precision, "auto")
    assert_equal(Device.cpu().precision, "auto")
    var wf2 = Workflow()
    var j0 = wf2.add_sample("auto", t, 100)
    var j1 = wf2.add_sample("fp32", t, 100, precision="fp32")
    assert_equal(job_precision(wf2.jobs[j0], gdev.precision), "fp64")
    assert_equal(job_precision(wf2.jobs[j1], gdev.precision), "fp32")
    var bad = List[Device]()
    bad.append(Device.gpu("gpu0", 30, "fp128"))
    var raised = False
    try:
        _ = wf2.plan(bad)
    except e:
        raised = String(e).startswith("QM-E941")
    assert_true(raised, "QM-E941 expected for an unknown device precision")
    var wf3 = Workflow()
    _ = wf3.add_sample("bad", t, 10, precision="half")
    raised = False
    try:
        _ = wf3.plan(pool)
    except e:
        raised = String(e).startswith("QM-E941")
    assert_true(raised, "QM-E941 expected for an unknown job precision")
    print("test_orchestrator OK")
