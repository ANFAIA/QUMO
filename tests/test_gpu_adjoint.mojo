"""Device-placed values and adjoint gradients (qumo.algorithms.gradients_device).

Default build: the GPU adjoint (doubled register + ancilla-Y reductions) runs
on the host-emulated GPU kernels in fp64 and fp32 and is checked against the
CPU adjoint (ExpectationEngine) on random circuits with every differentiable
gate (shared and affine parameters, U split into RZ RY RZ), a warm-start QAOA
circuit (which must batch its reductions: 2 per layer) and a non-diagonal
Hamiltonian. Also: expectations of many observables from one state, device
fallback notes and error codes. `-D QUMO_GPU` with a usable device: the same
gradient on the real device (fp32 tolerance).
"""

from std.math import pi, sqrt, asin
from std.sys.defines import is_defined
from std.testing import assert_true, assert_equal

from qumo.ir.circuit import Circuit
from qumo.ir.pauli import PauliSum, parse_pauli_sum
from qumo.sim.rng import Rng
from qumo.algorithms.gradients import ExpectationEngine, Grad
from qumo.algorithms.gradients_device import (
    gpu_value_and_gradient, value_and_gradient_on, expectations_on, expectations_emulated_gpu, DeviceGradient,
)
from qumo.gpu.device import gpu_available


def cpu_grad(c: Circuit, h: PauliSum, x: List[Float64], mut g: List[Float64]) raises -> Float64:
    var eng = ExpectationEngine(c, h)
    g = List[Float64](length=c.num_params(), fill=0.0)
    return eng.value_and_gradient(x, g, Grad.ADJOINT)


def max_abs_diff(a: List[Float64], b: List[Float64]) -> Float64:
    var m = 0.0
    for i in range(len(a)):
        m = max(m, abs(a[i] - b[i]))
    return m


def random_circuit(n: Int, layers: Int, seed: UInt64) raises -> Circuit:
    var rng = Rng(seed)
    var c = Circuit(n, name="rand")
    var th = c.parameters("t", 6)
    for _ in range(layers):
        for q in range(n):
            var k = rng.below(6)
            var p = th[rng.below(6)]
            var a = rng.uniform() * 2.0 - 1.0
            var pa = p
            pa.coeff = a
            pa.offset = 0.3 * rng.uniform()
            if k == 0:
                c.rx(pa, q)
            elif k == 1:
                c.ry(pa, q)
            elif k == 2:
                c.rz(pa, q)
            elif k == 3:
                c.p(pa, q)
            elif k == 4:
                c.u(pa, th[rng.below(6)], 0.4, q)
            else:
                c.h(q)
                c.t(q)
        for q in range(n - 1):
            var k = rng.below(8)
            var p = th[rng.below(6)]
            if k == 0:
                c.rxx(p, q, q + 1)
            elif k == 1:
                c.ryy(p, q + 1, q)
            elif k == 2:
                c.rzz(p, q, q + 1)
            elif k == 3:
                c.cp(p, q, q + 1)
            elif k == 4:
                c.crx(p, q + 1, q)
            elif k == 5:
                c.cry(p, q, q + 1)
            elif k == 6:
                c.crz(p, q + 1, q)
            else:
                c.cx(q, q + 1)
                c.rz(0.7, q + 1)
    return c^


def ws_qaoa(n: Int, p: Int, seed: UInt64) raises -> Tuple[Circuit, PauliSum]:
    """Warm-start QAOA (Egger et al. 2021) on a ring with chords, layered gate order."""
    var rng = Rng(seed)
    var c = Circuit(n, name="wsqaoa")
    var gam = c.parameters("gamma", p)
    var bet = c.parameters("beta", p)
    var thetas = List[Float64]()
    for _ in range(n):
        var ci = 0.25 if rng.uniform() < 0.5 else 0.75
        thetas.append(2.0 * asin(sqrt(ci)))
    var edges = List[Int]()
    for i in range(n):
        edges.append(i)
        edges.append((i + 1) % n)
    edges.append(0)
    edges.append(n // 2)
    var h = PauliSum()
    for e in range(len(edges) // 2):
        h.add(0.5, 0, (UInt64(1) << UInt64(edges[2 * e])) | (UInt64(1) << UInt64(edges[2 * e + 1])))
    for q in range(n):
        c.ry(thetas[q], q)
    for l in range(p):
        for e in range(len(edges) // 2):
            c.rzz(gam[l], edges[2 * e], edges[2 * e + 1])
        for q in range(n):
            c.ry(-thetas[q], q)
        for q in range(n):
            var b = -bet[l]
            b.coeff = -2.0
            c.rz(b, q)
        for q in range(n):
            c.ry(thetas[q], q)
    return (c^, h^)


def test_random_circuits() raises:
    var hs = parse_pauli_sum("0.7*Z0 Z1 + 0.3*X1 X2 - 0.5*Y0 Z2 + 0.2*X3 + 0.4*Z1 Y3 X2")
    for s in range(6):
        var c = random_circuit(4, 3, UInt64(100 + s))
        var rng = Rng(UInt64(7 + s))
        var x = List[Float64]()
        for _ in range(c.num_params()):
            x.append(rng.uniform() * 2.0 * pi - pi)
        var g_cpu = List[Float64]()
        var e_cpu = cpu_grad(c, hs, x, g_cpu)
        var r64 = gpu_value_and_gradient[DType.float64](c, hs, x, emulate=True)
        assert_true(abs(r64.value - e_cpu) < 1e-12, "fp64 value")
        var d64 = max_abs_diff(r64.grad, g_cpu)
        assert_true(d64 < 1e-10, String("fp64 gradient diff ") + String(d64))
        var r32 = gpu_value_and_gradient[DType.float32](c, hs, x, emulate=True)
        var d32 = max_abs_diff(r32.grad, g_cpu)
        assert_true(d32 < 2e-5, String("fp32 gradient diff ") + String(d32))
        assert_true(abs(r32.value - e_cpu) < 2e-5, "fp32 value")
        assert_equal(r64.method, "gpu-emulator:adjoint:fp64")
        assert_equal(r32.method, "gpu-emulator:adjoint:fp32")


def test_ws_qaoa_batches() raises:
    var p = 2
    var ch = ws_qaoa(8, p, 11)
    var x: List[Float64] = [0.4, -0.3, 0.7, 0.2]
    var g_cpu = List[Float64]()
    var e_cpu = cpu_grad(ch[0], ch[1], x, g_cpu)
    var r = gpu_value_and_gradient[DType.float64](ch[0], ch[1], x, emulate=True)
    assert_true(abs(r.value - e_cpu) < 1e-12, "ws-qaoa value")
    assert_true(max_abs_diff(r.grad, g_cpu) < 1e-10, "ws-qaoa gradient")
    # cost layer (commuting ZZ) and mixer RZ layer: one reduction batch each
    assert_equal(r.reductions, 2 * p)


def test_expectations_and_routing() raises:
    var c = random_circuit(5, 2, 5)
    var x = List[Float64]()
    for k in range(c.num_params()):
        x.append(0.1 * Float64(k + 1))
    var obs = List[PauliSum]()
    obs.append(parse_pauli_sum("Z0 Z1 + 0.5*X2"))
    obs.append(parse_pauli_sum("Y3 Y4 - 0.25*Z4"))
    obs.append(parse_pauli_sum("X0 Y1 Z2 X3 Y4"))
    var note = String()
    var method = String()
    var cpu = expectations_on(c, obs, x, "cpu", note, method)
    assert_equal(method, "cpu:statevector:fp64")
    var emu = expectations_emulated_gpu[DType.float64](c, obs, x)
    assert_true(max_abs_diff(cpu, emu) < 1e-12, "emulated GPU expectations")
    # device 'gpu' without a usable GPU: CPU with a note
    var vals = expectations_on(c, obs, x, "gpu", note, method)
    var on_gpu = False
    comptime if is_defined["QUMO_GPU"]():
        on_gpu = gpu_available()
    if on_gpu:
        assert_true(method.startswith("gpu:statevector"), "GPU method")
        assert_true(max_abs_diff(cpu, vals) < 1e-5, "GPU expectations")
    else:
        assert_true(note.byte_length() > 0, "fallback note")
        assert_true(max_abs_diff(cpu, vals) < 1e-14, "fallback values")
    var h = parse_pauli_sum("Z0 Z1 + 0.5*X2 - 0.3*Y3 Y4")
    var r = value_and_gradient_on(c, h, x, "gpu", note)
    var g_cpu = List[Float64]()
    var e_cpu = cpu_grad(c, h, x, g_cpu)
    if on_gpu:
        assert_true(r.method.startswith("gpu:adjoint"), "GPU adjoint ran")
        assert_true(max_abs_diff(r.grad, g_cpu) < 5e-5, "device gradient")
        print("  device GPU adjoint:", r.method, "max grad diff", max_abs_diff(r.grad, g_cpu))
    else:
        assert_equal(r.method, "cpu:adjoint:fp64")
        assert_true(note.byte_length() > 0, "gradient fallback note")
        assert_true(max_abs_diff(r.grad, g_cpu) < 1e-14, "cpu gradient")
    assert_true(abs(r.value - e_cpu) < 1e-5, "value")
    var r2 = value_and_gradient_on(c, h, x, "cpu", note)
    assert_equal(note, "")
    assert_equal(r2.method, "cpu:adjoint:fp64")
    var raised = False
    try:
        _ = value_and_gradient_on(c, h, x, "tpu", note)
    except e:
        raised = String(e).startswith("QM-E940")
    assert_true(raised, "QM-E940 for unknown device")
    raised = False
    try:
        _ = value_and_gradient_on(c, h, x, "cpu", note, "fp16")
    except e:
        raised = String(e).startswith("QM-E941")
    assert_true(raised, "QM-E941 for unknown precision")


def test_errors() raises:
    var c = Circuit(2)
    var t = c.parameter("t")
    c.rx(t, 0)
    c.measure_all()
    var h = parse_pauli_sum("Z0")
    var raised = False
    try:
        var bad: List[Float64] = [0.1, 0.2]
        _ = gpu_value_and_gradient[DType.float64](c, h, bad, emulate=True)
    except e:
        raised = String(e).startswith("QM-E804")
    assert_true(raised, "QM-E804")
    var d = Circuit(2)
    d.reset(0)
    raised = False
    try:
        _ = gpu_value_and_gradient[DType.float64](d, h, List[Float64](), emulate=True)
    except e:
        raised = String(e).startswith("QM-E805")
    assert_true(raised, "QM-E805")
    # terminal measurements are ignored; d/dt <Z0> of RX(t)|0> = -sin t
    var r = gpu_value_and_gradient[DType.float64](c, h, [0.3], emulate=True)
    from std.math import sin, cos
    assert_true(abs(r.value - cos(0.3)) < 1e-12 and abs(r.grad[0] + sin(0.3)) < 1e-12, "rx analytic")


def main() raises:
    test_random_circuits()
    test_ws_qaoa_batches()
    test_expectations_and_routing()
    test_errors()
    print("test_gpu_adjoint OK")
