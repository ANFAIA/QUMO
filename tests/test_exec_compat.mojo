"""The old `qumo.exec.dispatch`, `qumo.exec.emulate` and `qumo.exec.resources` paths (kept for one release)
give the same functions as `qumo.gpu.dispatch` / `qumo.sim.dispatch`, `qumo.noise.emulate` and
`qumo.compiler.resources`."""
from std.testing import assert_equal, assert_true

from qumo.ir.circuit import Circuit
from qumo.device.target import Target
from qumo.exec.dispatch import sample_auto as old_sample_auto, choose_method as old_choose_method, choose_method_cpu as old_choose_method_cpu, sample_on as old_sample_on
from qumo.exec.emulate import emulate as old_emulate
from qumo.exec.resources import estimate_resources as old_estimate_resources, HardwareProfile as OldHardwareProfile
from qumo.gpu.dispatch import sample_auto, choose_method, sample_on
from qumo.sim.dispatch import choose_method_cpu
from qumo.noise.emulate import emulate
from qumo.compiler.resources import estimate_resources, HardwareProfile
from qumo.compiler.pass_manager import conformance_error as pm_conformance_error
from qumo.device.conformance import conformance_error


def main() raises:
    var c = Circuit(3, 3)
    c.h(0)
    c.cx(0, 1)
    c.t(2)
    c.cx(1, 2)
    c.measure_all()
    assert_equal(old_choose_method(c, 1000), choose_method(c, 1000))
    assert_equal(old_choose_method_cpu(c, 1000), choose_method_cpu(c, 1000))
    assert_equal(String(old_sample_auto(c, 500, 7)), String(sample_auto(c, 500, 7)))
    var n1 = String()
    var n2 = String()
    assert_equal(String(old_sample_on(c, 500, 7, List[Float64](), "cpu", n1)), String(sample_on(c, 500, 7, List[Float64](), "cpu", n2)))
    assert_equal(n1, n2)
    assert_equal(String(old_estimate_resources(c)), String(estimate_resources(c)))
    assert_equal(String(OldHardwareProfile.superconducting()), String(HardwareProfile.superconducting()))
    var t = Target.line(3, "rz,sx,x,cz")
    var d = Circuit(3, 3)
    d.sx(0)
    d.cz(0, 1)
    d.measure_all()
    assert_equal(String(old_emulate(d, t, 300, 5)), String(emulate(d, t, 300, 5)))
    assert_equal(conformance_error(d, t), String(""))
    assert_true(conformance_error(c, t).byte_length() > 0)
    assert_equal(pm_conformance_error(c, t), conformance_error(c, t))
    print("test_exec_compat OK")
