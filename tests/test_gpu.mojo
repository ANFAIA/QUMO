"""GPU state vector tests.

Default build: every GPU kernel *body* runs through the host emulator over the
exact launch grid (qumo.gpu.emulate) in fp64 and fp32 and is checked against
the CPU StateVector: each gate kernel on edge qubits (0 and n-1) and tiny
states (grid tails), random circuits up to 21 qubits, probabilities,
Pauli expectation values through the emulated shared-memory tree (including
the >1024-block grid-stride regime), segment sampling and error codes.

`-D QUMO_GPU` build: additionally runs the same suites on the real device
(skipped with a message when no device is usable). `make gpu-check` compiles
this file for sm_90, gfx942 and metal:4 so every kernel is compiled for every
vendor (fp64 kernels everywhere except Apple).
"""

from std.math import cos, sin, sqrt
from std.sys.defines import is_defined
from std.testing import assert_equal, assert_true, assert_almost_equal

from qumo.ir.gates import Op, M2, M4, matrix_1q, matrix_2q, u3_matrix
from qumo.ir.circuit import Circuit
from qumo.ir.pauli import PauliSum, parse_pauli_sum
from qumo.sim.rng import Rng
from qumo.sim.statevector import StateVector
from qumo.sim.runner import evolve, sample, build_plan
from qumo.gpu.kernels import TPB, SAMPLE_SEG, MAX_RED_BLOCKS
from qumo.gpu.statevector_gpu import GpuStateVector, evolve_gpu, sample_gpu, expectation_gpu, reduction_blocks
from qumo.gpu.device import (
    gpu_compiled, gpu_available, gpu_info, query_gpu, GPU_DEFAULT_DTYPE, GPU_FP64_SUPPORTED,
    GPU_MIN_QUBITS,
)
from qumo.exec.dispatch import choose_method, sample_auto
from qumo.exec.orchestrator import Workflow, Device
from qumo.ai.registry import full_registry
from qumo.ai.json import parse_json


def max_diff(a: StateVector, b: StateVector) -> Float64:
    var m = 0.0
    for i in range(a.size()):
        m = max(m, abs(a.re[i] - b.re[i]))
        m = max(m, abs(a.im[i] - b.im[i]))
    return m


def prep_circuit(n: Int) -> Circuit:
    """A generic entangled state with non-trivial phases on every qubit."""
    var c = Circuit(n)
    for q in range(n):
        c.ry(0.3 + 0.17 * Float64(q), q)
        c.rz(0.5 - 0.11 * Float64(q), q)
    for q in range(n - 1):
        c.cx(q, q + 1)
    for q in range(n):
        c.rx(0.2 + 0.05 * Float64(q), q)
    if n > 1:
        c.cz(0, n - 1)
    return c^


def dense_m4() -> M4:
    return matrix_2q(Op.ECR) @ matrix_2q(Op.RXX, 0.6) @ M4.from_1q(u3_matrix(0.4, 1.3, -0.2), True)


def apply_case[dt: DType](k: Int, mut cpu: StateVector, mut gpu: GpuStateVector[dt], a: Int, b: Int, c: Int) raises:
    """Gate case k on both simulators (a, b, c distinct qubits)."""
    if k == 0:
        var m = u3_matrix(0.3, 1.1, -0.7)
        cpu.apply_1q(m, a)
        gpu.apply_1q(m, a)
    elif k == 1:
        cpu.apply_diag_1q(1.0, 0.0, cos(0.4), sin(0.4), a)
        gpu.apply_diag_1q(1.0, 0.0, cos(0.4), sin(0.4), a)
    elif k == 2:
        cpu.apply_diag_1q(cos(0.2), sin(0.2), cos(0.9), -sin(0.9), a)
        gpu.apply_diag_1q(cos(0.2), sin(0.2), cos(0.9), -sin(0.9), a)
    elif k == 3:
        cpu.apply_x(a)
        gpu.apply_x(a)
    elif k == 4:
        cpu.apply_cx(a, b)
        gpu.apply_cx(a, b)
    elif k == 5:
        cpu.apply_cphase(cos(0.7), sin(0.7), a, b)
        gpu.apply_cphase(cos(0.7), sin(0.7), a, b)
    elif k == 6:
        cpu.apply_cphase(-1.0, 0.0, a, b)
        gpu.apply_cphase(-1.0, 0.0, a, b)
    elif k == 7:
        cpu.apply_controlled_1q(matrix_1q(Op.RY, 0.8), a, b)
        gpu.apply_controlled_1q(matrix_1q(Op.RY, 0.8), a, b)
    elif k == 8:
        cpu.apply_rzz(0.9, a, b)
        gpu.apply_rzz(0.9, a, b)
    elif k == 9:
        cpu.apply_swap(a, b)
        gpu.apply_swap(a, b)
    elif k == 10:
        cpu.apply_2q(dense_m4(), a, b)
        gpu.apply_2q(dense_m4(), a, b)
    elif k == 11:
        cpu.apply_ccx(a, b, c)
        gpu.apply_ccx(a, b, c)
    elif k == 12:
        cpu.apply_ccz(a, b, c)
        gpu.apply_ccz(a, b, c)
    elif k == 13:
        cpu.apply_cswap(a, b, c)
        gpu.apply_cswap(a, b, c)
    elif k == 14:
        cpu.apply_global_phase(0.37)
        gpu.apply_global_phase(0.37)


comptime NUM_CASES = 15


def case_arity(k: Int) -> Int:
    if k <= 3 or k == 14:
        return 1
    if k <= 10:
        return 2
    return 3


def check_gates[dt: DType](emulate: Bool, tol: Float64) raises:
    """Every kernel on a generic 7-qubit state, operands incl. qubits 0 and n-1."""
    var n = 7
    var base = evolve(prep_circuit(n))
    var triples: List[Int] = [0, 6, 3, 6, 0, 4, 2, 5, 6, 6, 1, 0, 3, 4, 5, 1, 0, 6]
    for k in range(NUM_CASES):
        for t in range(len(triples) // 3):
            var a = triples[3 * t]
            var b = triples[3 * t + 1]
            var c = triples[3 * t + 2]
            var cpu = base.copy()
            var gpu = GpuStateVector[dt](n, emulate)
            gpu.load_host(base)
            apply_case[dt](k, cpu, gpu, a, b, c)
            var d = max_diff(cpu, gpu.to_host())
            assert_true(d < tol, "gate case " + String(k) + " on (" + String(a) + "," + String(b) + "," + String(c) + ") diff " + String(d))
    # tiny states: a single work item per launch, 255 idle tail threads
    for k in range(NUM_CASES):
        var n2 = case_arity(k)
        var b2 = evolve(prep_circuit(n2))
        var cpu = b2.copy()
        var gpu = GpuStateVector[dt](n2, emulate)
        gpu.load_host(b2)
        apply_case[dt](k, cpu, gpu, n2 - 1, 0, 1 if n2 == 3 else 0)
        var d = max_diff(cpu, gpu.to_host())
        assert_true(d < tol, "tiny-state case " + String(k) + " diff " + String(d))


def random_circuit(n: Int, ngates: Int, seed: UInt64) -> Circuit:
    """Every opcode family the GPU executor lowers, edge qubits favoured."""
    var rng = Rng(seed)
    var c = Circuit(n)
    c.global_phase = 0.21
    for _ in range(ngates):
        var a = rng.below(n)
        var r = rng.below(4)
        if r == 0:
            a = 0
        elif r == 1:
            a = n - 1
        var b = (a + 1 + rng.below(max(n - 1, 1))) % n
        var cq = 0
        while n >= 3 and (cq == a or cq == b):
            cq = rng.below(n)
        var th = rng.uniform() * 6.283185307179586 - 3.14159
        var maxk = 12 if n == 1 else (27 if n == 2 else 30)
        var k = rng.below(maxk)
        if k == 0: c.h(a)
        elif k == 1: c.x(a)
        elif k == 2: c.y(a)
        elif k == 3: c.z(a)
        elif k == 4: c.s(a)
        elif k == 5: c.t(a)
        elif k == 6: c.sx(a)
        elif k == 7: c.rx(th, a)
        elif k == 8: c.ry(th, a)
        elif k == 9: c.rz(th, a)
        elif k == 10: c.p(th, a)
        elif k == 11: c.u(th, 0.3, -0.8, a)
        elif k == 12: c.cx(a, b)
        elif k == 13: c.cz(a, b)
        elif k == 14: c.cp(th, a, b)
        elif k == 15: c.crx(th, a, b)
        elif k == 16: c.cry(th, a, b)
        elif k == 17: c.crz(th, a, b)
        elif k == 18: c.cy(a, b)
        elif k == 19: c.ch(a, b)
        elif k == 20: c.swap(a, b)
        elif k == 21: c.rzz(th, a, b)
        elif k == 22: c.rxx(th, a, b)
        elif k == 23: c.ryy(th, a, b)
        elif k == 24: c.iswap(a, b)
        elif k == 25: c.ecr(a, b)
        elif k == 26: c.cx(b, a)
        elif k == 27: c.ccx(a, b, cq)
        elif k == 28: c.ccz(a, b, cq)
        else: c.cswap(a, b, cq)
    return c^


def check_circuits[dt: DType](emulate: Bool, tol: Float64, big: Bool) raises:
    var ns: List[Int] = [1, 2, 3, 6, 11]
    var gs: List[Int] = [12, 30, 60, 150, 200]
    if big:
        ns.append(20)
        gs.append(90)
        ns.append(21)
        gs.append(50)
    for i in range(len(ns)):
        var n = ns[i]
        var c = random_circuit(n, gs[i], UInt64(100 + n))
        var cpu = evolve(c)
        var g = evolve_gpu[dt](c, emulate=emulate)
        var d = max_diff(cpu, g.to_host())
        assert_true(d < tol, "random circuit n=" + String(n) + " diff " + String(d))
        # one launch per plan op + init + global phase: no hidden host work
        var plan = build_plan(c, List[Float64]())
        assert_equal(g.launches, len(plan) + 2)
        var pg = g.probabilities()
        var pc = cpu.probabilities()
        var dp = 0.0
        for j in range(len(pc)):
            dp = max(dp, abs(pg[j] - pc[j]))
        assert_true(dp < tol, "probabilities n=" + String(n) + " diff " + String(dp))
        assert_true(abs(g.norm_sq() - 1.0) < 10 * tol, "norm n=" + String(n))


def random_pauli_sum(n: Int, nterms: Int, seed: UInt64) -> PauliSum:
    var rng = Rng(seed)
    var h = PauliSum()
    h.add(0.25, 0, 0)  # identity
    var top = UInt64(1) << UInt64(n - 1)
    h.add(-0.5, 1, 1)  # Y0
    h.add(0.75, top, 0)  # X_{n-1}
    h.add(0.6, top | 1, top)  # X0 Y_{n-1}
    for _ in range(nterms):
        var mask = (UInt64(1) << UInt64(n)) - 1
        var x = rng.next_u64() & mask
        var z = rng.next_u64() & mask
        h.add(rng.uniform() - 0.5, x, z)
    return h^


def direct_pauli_sum(sv: StateVector, x: Int, z: Int) -> Tuple[Float64, Float64]:
    """Sequential fp64 Kahan-summed reference for one reduction (no tree)."""
    var sr = 0.0
    var si = 0.0
    var cr = 0.0
    var ci = 0.0
    for i in range(sv.size()):
        var j = i ^ x
        var s = -1.0 if (pop_count_int(i & z) & 1) == 1 else 1.0
        var yr = s * (sv.re[j] * sv.re[i] + sv.im[j] * sv.im[i]) - cr
        var tr = sr + yr
        cr = (tr - sr) - yr
        sr = tr
        var yi = s * (sv.re[j] * sv.im[i] - sv.im[j] * sv.re[i]) - ci
        var ti = si + yi
        ci = (ti - si) - yi
        si = ti
    return (sr, si)


def pop_count_int(v: Int) -> Int:
    var c = 0
    var x = v
    while x != 0:
        c += x & 1
        x >>= 1
    return c


def check_expectation[dt: DType](emulate: Bool, tol: Float64, big: Bool) raises:
    var ns: List[Int] = [1, 3, 9, 13]
    if big:
        ns.append(20)
    for n in ns:
        var c = random_circuit(n, 40 + 4 * n, UInt64(7 * n + 1))
        var cpu = evolve(c)
        var g = evolve_gpu[dt](c, emulate=emulate)
        var h = random_pauli_sum(n, 12, UInt64(n))
        var e_cpu = cpu.expectation(h)
        var e_gpu = g.expectation(h)
        assert_true(abs(e_cpu - e_gpu) < 10 * tol, "expectation n=" + String(n) + ": " + String(e_cpu) + " vs " + String(e_gpu))
        var e2 = expectation_gpu[dt](c, h, emulate=emulate)
        assert_true(abs(e2 - e_gpu) < 10 * tol)
    if not big:
        return
    # reduction structure: 2^23 amplitudes -> 1024 blocks (cap) x 256 threads,
    # 32 grid-stride items per thread; compare raw sums with a sequential loop
    var n = 23
    assert_equal(reduction_blocks(1 << n), MAX_RED_BLOCKS)
    var c = Circuit(n)
    for q in range(n):
        c.h(q)
        c.rz(0.1 * Float64(q + 1), q)
    c.cx(0, n - 1)
    c.ry(0.7, n - 1)
    var g = evolve_gpu[dt](c, emulate=emulate)
    var host = g.to_host()
    var xs: List[Int] = [0, 1, (1 << (n - 1)) | 5]
    var zs: List[Int] = [0, (1 << (n - 1)) | 2, 3]
    var s = g.pauli_sums(xs, zs)
    for k in range(len(xs)):
        var ref_ = direct_pauli_sum(host, xs[k], zs[k])
        assert_true(abs(s[2 * k] - ref_[0]) < 50 * tol, "tree reduction re, term " + String(k) + ": " + String(s[2 * k]) + " vs " + String(ref_[0]))
        assert_true(abs(s[2 * k + 1] - ref_[1]) < 50 * tol, "tree reduction im, term " + String(k) + ": " + String(s[2 * k + 1]) + " vs " + String(ref_[1]))


def check_sampling[dt: DType](emulate: Bool, exact: Bool) raises:
    # n=5: one 32-amplitude segment; n=14: four 4096-amplitude segments
    var ns: List[Int] = [5, 14]
    for n in ns:
        var c = random_circuit(n, 10 * n, UInt64(31 + n))
        c.measure_all()
        var shots = 4000
        var rg = sample_gpu[dt](c, shots, 99, emulate=emulate)
        var rc = sample(c, shots, 99)
        assert_true(rg.method.startswith("gpu"), rg.method)
        assert_equal(rg.shots, shots)
        var tot = 0
        for kv in rg.counts.items():
            tot += kv.value
            var cc = rc.counts.get(kv.key, 0)
            if exact:
                assert_equal(kv.value, cc, "fp64 GPU sampling must match the CPU sampler for a seed: " + kv.key)
            else:
                assert_true(abs(kv.value - cc) <= 3, "fp32 GPU sampling far from CPU on " + kv.key)
        assert_equal(tot, shots)
        # same seed -> identical shot record (memory order included)
        var rg2 = sample_gpu[dt](c, shots, 99, emulate=emulate)
        for i in range(shots):
            assert_equal(rg.memory[i], rg2.memory[i])
    # zero-probability states are never returned: 13-qubit GHZ with phases
    var g = Circuit(13)
    g.h(0)
    for q in range(12):
        g.cx(q, q + 1)
    g.t(5)
    g.rz(0.4, 12)
    g.measure_all()
    var r = sample_gpu[dt](g, 3000, 5, emulate=emulate)
    var zeros = r.counts.get("0" * 13, 0)
    var ones = r.counts.get("1" * 13, 0)
    assert_equal(zeros + ones, 3000)
    assert_true(zeros > 1300 and ones > 1300)
    # explicit low-level API: sample indices from a basis state
    var b = GpuStateVector[dt](13, emulate)
    b.set_basis(5000)
    var idx = b.sample(100, 3)
    for i in idx:
        assert_equal(i, 5000)


def check_errors() raises:
    # dynamic circuits are not sampled on the GPU
    var c = Circuit(2)
    c.h(0)
    c.measure(0, 0)
    c.x(0)
    c.measure(0, 1)
    var raised = False
    try:
        _ = sample_gpu[DType.float64](c, 10, 1, emulate=True)
    except e:
        raised = String(e).startswith("QM-E471")
    assert_true(raised, "QM-E471 expected")
    # circuit wider than the state
    raised = False
    try:
        var g = GpuStateVector[DType.float64].emulator(2)
        g.evolve(prep_circuit(3))
    except e:
        raised = String(e).startswith("QM-E474")
    assert_true(raised, "QM-E474 expected")
    comptime if not is_defined["QUMO_GPU"]():
        # default builds: device mode is a clear, coded error, never a crash
        raised = False
        try:
            _ = GpuStateVector[DType.float32](3)
        except e:
            raised = String(e).startswith("QM-E470")
        assert_true(raised, "QM-E470 expected without -D QUMO_GPU")
        assert_true(not gpu_compiled())
        assert_true(not gpu_available())
        assert_true(not query_gpu().available)
    var js = query_gpu().to_json()
    assert_true(js.startswith("{\"compiled\":"), js)
    print(gpu_info())


def check_integration() raises:
    # dispatch: a wide non-Clifford static circuit
    var c = Circuit(GPU_MIN_QUBITS)
    for q in range(GPU_MIN_QUBITS):
        c.h(q)
    c.t(0)
    c.cx(0, GPU_MIN_QUBITS - 1)
    c.measure_all()
    var m = choose_method(c, 100)
    var r = sample_auto(c, 200, 4)
    if gpu_compiled() and gpu_available():
        assert_equal(m, "gpu")
        assert_true(r.method.startswith("gpu:statevector:"), r.method)
    else:
        assert_equal(m, "statevector")
        assert_equal(r.method, "statevector:static")
    # orchestrator: a job planned on a GPU device runs (on the GPU or on the
    # CPU with an explicit fallback note)
    var wf = Workflow()
    var t = Circuit(4)
    t.h(0)
    t.t(0)
    t.cx(0, 3)
    t.measure_all()
    _ = wf.add_sample("t4", t, 300)
    _ = wf.add_estimate("zz", prep_circuit(4), parse_pauli_sum("Z0 Z3"))
    var pool = List[Device]()
    pool.append(Device.gpu("gpu0"))
    var res = wf.run(pool)
    for jr in res:
        assert_true(jr.ok, jr.error)
        assert_equal(jr.device, "gpu0")
        if gpu_compiled() and gpu_available():
            assert_true(jr.backend.startswith("gpu:"), jr.backend)
        else:
            assert_true(jr.backend.find("fallback") >= 0, jr.backend)
    assert_almost_equal(res[1].expectation, evolve(prep_circuit(4)).expectation(parse_pauli_sum("Z0 Z3")), atol=1e-5)


def check_tools() raises:
    """MCP tools: qumo_gpu_info and qumo_run_device (device auto/cpu/gpu)."""
    var reg = full_registry()
    var info = reg.call("qumo_gpu_info", parse_json("{}"))
    assert_true(not info.is_error, info.structured)
    var ij = parse_json(info.structured)
    assert_equal(ij.get("compiled").as_bool(), gpu_compiled())
    var q = "OPENQASM 3.0; include \\\"stdgates.inc\\\"; qubit[3] q; bit[3] c; h q[0]; t q[0]; cx q[0], q[2]; c = measure q;"
    for dev in ["auto", "cpu", "gpu"]:
        var r = reg.call("qumo_run_device", parse_json("{\"qasm\":\"" + q + "\",\"shots\":200,\"seed\":3,\"device\":\"" + dev + "\"}"))
        assert_true(not r.is_error, r.structured)
        var j = parse_json(r.structured)
        var cts = j.get("counts")
        assert_equal(cts.get("000").as_int() + cts.get("101").as_int(), 200)
        var method = j.get("method").as_string()
        if dev == "gpu" and gpu_compiled() and gpu_available():
            assert_true(method.startswith("gpu:"), method)
        else:
            assert_equal(method, "statevector:static")
        if dev == "gpu" and not (gpu_compiled() and gpu_available()):
            assert_true(r.structured.find("QM-W020") >= 0, r.structured)
    var bad = reg.call("qumo_run_device", parse_json("{\"qasm\":\"" + q + "\",\"device\":\"tpu\"}"))
    assert_true(bad.is_error and bad.structured.find("QM-E940") >= 0, bad.structured)


def run_suite[dt: DType](emulate: Bool, tol: Float64, big: Bool, exact_sampling: Bool) raises:
    check_gates[dt](emulate, tol)
    check_circuits[dt](emulate, tol, big)
    check_expectation[dt](emulate, tol, big)
    check_sampling[dt](emulate, exact_sampling)


def main() raises:
    # host emulation of the exact kernel bodies and launch grids
    run_suite[DType.float64](True, 1e-12, True, True)
    run_suite[DType.float32](True, 2e-5, True, False)
    check_errors()
    check_integration()
    check_tools()
    comptime if is_defined["QUMO_GPU"]():
        if gpu_available():
            print("running device suites on:", gpu_info())
            run_suite[DType.float32](False, 2e-5, True, False)
            comptime if GPU_FP64_SUPPORTED:
                run_suite[DType.float64](False, 1e-12, True, False)
        else:
            print("no usable GPU device: device suites skipped (", gpu_info(), ")")
    print("test_gpu OK")
