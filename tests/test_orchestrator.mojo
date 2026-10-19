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
    test_pinning()
    print("test_orchestrator OK")


def test_pinning() raises:
    """Device pinning: pinned jobs go to their device, unpinned ones to HEFT."""
    var pool = List[Device]()
    pool.append(Device.cpu())
    pool.append(Device.qpu("sc-qpu", 20, 5.0, 1e-4, 0.3, 1e-4))
    pool.append(Device.qpu("ion-qpu", 12, 30.0, 1e-2, 1.0, 1e-2))
    var c6 = ghz(6)
    c6.measure_all()
    var wf = Workflow()
    var j0 = wf.add_sample("dense-fragment", c6, 500, device="ion-qpu")
    var j1 = wf.add_sample("wide-fragment", c6, 500, device="sc-qpu")
    var j2 = wf.add_estimate("exact-check", ghz(6), parse_pauli_sum("Z0 Z5"))
    var sch = wf.plan(pool)
    assert_equal(pool[sch.device_of[j0]].name, "ion-qpu")
    assert_equal(pool[sch.device_of[j1]].name, "sc-qpu")
    assert_equal(pool[sch.device_of[j2]].name, "cpu0")
    var pinned_notes = 0
    for line in sch.notes:
        if "(pinned)" in line:
            pinned_notes += 1
    assert_equal(pinned_notes, 2)
    # without the pin HEFT sends the dense fragment to the faster QPU
    var wf_free = Workflow()
    var k0 = wf_free.add_sample("dense-fragment", c6, 500)
    var sfree = wf_free.plan(pool)
    assert_true(pool[sfree.device_of[k0]].name != "ion-qpu", "unpinned job is placed by HEFT")
    # pinned jobs run (QPU jobs on the local emulator, flagged emulated)
    var res = wf.run(pool)
    assert_true(res[j0].ok and res[j0].emulated and res[j0].device == "ion-qpu", "pinned job ran")
    var raised = False
    var wf_bad = Workflow()
    _ = wf_bad.add_sample("x", c6, 10, device="nope")
    try:
        _ = wf_bad.plan(pool)
    except e:
        raised = String(e).startswith("QM-E463")
    assert_true(raised, "QM-E463 for an unknown pinned device")
    raised = False
    var wf_small = Workflow()
    var big = ghz(16)
    big.measure_all()
    _ = wf_small.add_sample("too-big", big, 10, device="ion-qpu")
    try:
        _ = wf_small.plan(pool)
    except e:
        raised = String(e).startswith("QM-E464")
    assert_true(raised, "QM-E464 for a pinned device that is too small")
