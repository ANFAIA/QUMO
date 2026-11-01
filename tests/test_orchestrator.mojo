from qumo.ir.circuit import Circuit
from qumo.ir.pauli import parse_pauli_sum
from qumo.exec.orchestrator import Workflow, Device, job_precision
from std.math import pi
from qumo.exec.resources import estimate_resources, estimate_hardware, estimate_fault_tolerant, HardwareProfile, SynthesisModel
from std.testing import assert_equal, assert_true, assert_almost_equal


def ghz(n: Int) -> Circuit:
    var c = Circuit(n)
    c.h(0)
    for q in range(n - 1):
        c.cx(q, q + 1)
    return c^


def test_rotation_t_cost() raises:
    """Arbitrary rotations cost a synthesised sequence, not one T (regression:
    200 rotations used to report T=200)."""
    var c = Circuit(2)
    for _ in range(200):
        c.rz(0.3, 0)
    var r = estimate_resources(c)
    assert_equal(r.rotations, 200)
    assert_equal(r.t_exact, 0)
    assert_equal(r.t_per_rotation, 53)  # ceil(3 log2(200 / 1e-3))
    assert_equal(r.t_count, 200 * 53)
    var rus = estimate_resources(c, model=SynthesisModel.repeat_until_success())
    assert_equal(rus.t_per_rotation, 30)  # ceil(1.149 log2(200 / 1e-3) + 9.2)
    assert_true(estimate_resources(c, rotation_budget=1e-6).t_count > r.t_count)
    # exact Clifford+T stays exact; two-qubit rotations are counted too
    var d = Circuit(3)
    d.t(0)
    d.ccx(0, 1, 2)
    assert_equal(estimate_resources(d).t_count, 8)
    var e = Circuit(2)
    e.rzz(0.3, 0, 1)
    e.cp(0.3, 0, 1)
    assert_equal(estimate_resources(e).rotations, 4)  # rzz 1, cp 3
    var f = Circuit(2)
    f.rz(pi / 4, 0)  # exactly one T up to Cliffords, nothing to synthesise
    f.cp(pi / 2, 0, 1)  # CS: three T
    f.cp(pi, 0, 1)  # CZ
    var rf = estimate_resources(f)
    assert_equal(rf.t_exact, 4)
    assert_equal(rf.rotations, 0)
    # the fault-tolerant estimate pays for the synthesised T states
    assert_equal(estimate_fault_tolerant(c).logical_cycles, 200 * 48)  # budget 1e-2 / 3


def main() raises:
    test_rotation_t_cost()
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
