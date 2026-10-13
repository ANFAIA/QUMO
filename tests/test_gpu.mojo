"""GPU state vector tests.

Default build: every GPU kernel's thread phases run through the host emulator
over the exact launch grid (qumo.gpu.emulate) in fp64 and fp32 and are checked
against the CPU StateVector: every gate on edge qubits (0 and n-1) and on tiny
states (partial register sets), random circuits up to 21 qubits (many tile
runs, register transposes, controls and diagonal gates outside the tile),
the tile scheduler itself (coverage, capacity, diagonal gates free, launch
geometry at 32 qubits), probabilities, Pauli expectation values through the
emulated shared-memory trees (including the >1024-block grid-stride regime
and batched terms), CDF-block sampling and error codes.

`-D QUMO_GPU` build: additionally runs the same suites on the real device
(skipped with a message when no device is usable, an error instead when
QUMO_GPU_REQUIRE_DEVICE=1, as `make test-gpu` sets it) with fp32-appropriate
tolerances, plus a 24-qubit random circuit and probability / expectation /
sampling reductions at that width. `make gpu-check` compiles this file for
sm_75, sm_80, sm_90, gfx942 and metal:4 so every kernel is compiled for every
vendor (fp64 kernels everywhere except Apple).
"""

from std.math import cos, sin, sqrt
from std.sys.info import size_of
from std.sys.defines import is_defined
from std.os import getenv
from std.testing import assert_equal, assert_true, assert_almost_equal

from qumo.ir.gates import Op, M2, M4, matrix_1q, matrix_2q, u3_matrix
from qumo.ir.circuit import Circuit
from qumo.ir.pauli import PauliSum, parse_pauli_sum
from qumo.sim.rng import Rng
from qumo.sim.statevector import StateVector
from qumo.sim.runner import evolve, sample, build_plan
from qumo.gpu.kernels import (
    TPB, MAX_RED_BLOCKS, MAX_GRID_BLOCKS, tile_bits, sample_seg, REG_BITS, EXP_G, SEG_DIRECT, FIN_DIRECT,
    TILE_SMEM_DYNAMIC, TILE_SAFE_THREADS, TILE_LAUNCH_BOUND,
)
from qumo.gpu.tiles import TileProgram, build_tile_program, tile_geometry, split_1q_ok
from qumo.gpu.emulate import stride_blocks
from qumo.gpu.statevector_gpu import GpuStateVector, evolve_gpu, sample_gpu, expectation_gpu, reduction_blocks
from qumo.gpu.device import (
    gpu_compiled, gpu_available, gpu_info, query_gpu, GPU_DEFAULT_DTYPE, GPU_FP64_SUPPORTED,
    GPU_MIN_QUBITS, GpuProfile, TileKernelLimits,
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


def layered_circuit(n: Int, depth: Int) -> Circuit:
    """The layered circuit of bench_sv.mojo (its tile programs use direct loads and
    stores at the ends of runs: see check_tiles)."""
    var c = Circuit(n)
    for d in range(depth):
        for q in range(n):
            c.ry(0.1 + 0.01 * Float64(q + d), q)
            c.rz(0.2 + 0.03 * Float64(q * d), q)
        for q in range(d % 2, n - 1, 2):
            c.cx(q, q + 1)
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
        # one launch per tile run (init and global phase folded in), and
        # gate blocking: far fewer launches than plan ops on wide states
        var plan = build_plan(c, List[Float64]())
        assert_equal(g.launches, g.runs_last)
        assert_true(g.ops_last <= len(plan) + 1)  # + the global phase op, - fused diagonals
        var hostp = build_tile_program[dt](plan, n, c.global_phase, g.profile.tile_bits[dt](), g.profile.low[dt](),
                                           g.profile.diag_width)
        assert_equal(g.transposes_last, hostp.num_transposes())
        assert_equal(g.runs_last, len(hostp.runs))
        assert_equal(g.ops_last, hostp.num_ops())
        assert_true(g.runs_last >= 1 and g.runs_last <= len(plan))
        if n >= 11:
            assert_true(g.runs_last * 2 < len(plan), "n=" + String(n) + ": " + String(g.runs_last) + " runs for " + String(len(plan)) + " ops")
        var pg = g.probabilities()
        var pc = cpu.probabilities()
        var dp = 0.0
        for j in range(len(pc)):
            dp = max(dp, abs(pg[j] - pc[j]))
        assert_true(dp < tol, "probabilities n=" + String(n) + " diff " + String(dp))
        assert_true(abs(g.norm_sq() - 1.0) < 10 * tol, "norm n=" + String(n))
    if big:  # direct loads / stores at run ends (see check_tiles)
        var lc = layered_circuit(20, 6)
        var dl = max_diff(evolve(lc), evolve_gpu[dt](lc, emulate=emulate).to_host())
        assert_true(dl < tol, "layered(20) diff " + String(dl))


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
    # several MEASUREs into one clbit: the LAST one wins (runner.sample, Aer);
    # q0 = 1 and q1 = 0, so clbit 0 must read q1's 0 when q1 is measured last
    for order in range(2):
        var mc = Circuit(2, 2)
        mc.x(0)
        mc.h(1)
        mc.h(1)
        if order == 0:
            mc.measure(0, 0)
            mc.measure(1, 0)
        else:
            mc.measure(1, 0)
            mc.measure(0, 0)
        mc.measure(0, 1)
        var rm = sample_gpu[dt](mc, 64, 3, emulate=emulate)
        var rmc = sample(mc, 64, 3)
        var want = "10" if order == 0 else "11"  # bitstring: clbit 1 (= q0 = 1), then clbit 0
        assert_equal(rm.counts.get(want, 0), 64, "last MEASURE into a clbit must win (order " + String(order) + ")")
        assert_equal(rmc.counts.get(want, 0), 64)
    # explicit low-level API: sample indices from a basis state
    var b = GpuStateVector[dt](13, emulate)
    b.set_basis(5000)
    var idx = b.sample(100, 3)
    for i in idx:
        assert_equal(i, 5000)


def qft(n: Int) -> Circuit:
    var c = Circuit(n)
    var j = n - 1
    while j >= 0:
        c.h(j)
        var k = j - 1
        while k >= 0:
            c.cp(3.141592653589793 / Float64(1 << (j - k)), k, j)
            k -= 1
        j -= 1
    for q in range(n // 2):
        c.swap(q, n - 1 - q)
    return c^


def check_segments[dt: DType](prog: TileProgram[dt]) raises -> Tuple[Int, Int]:
    """Packed segment records: each run's segments cover its ops in order, and
    every op's target register positions point at its local target qubits;
    direct loads / stores (no transpose at a run's ends) only with register
    sets whose local qubits are all >= low. Returns (direct loads, direct stores)."""
    var dl = 0
    var ds = 0
    for r in prog.runs:
        if r.fin == FIN_DIRECT:
            ds += 1
            assert_true(r.s1 > r.s0)
            for q in prog.seg_qubits(r.s1 - 1):
                assert_true(q >= prog.low, "direct store from a set with a low qubit")
        var nxt = r.o0
        for i in range(r.s0, r.s1):
            var rg = prog.seg_ops(i)
            assert_equal(rg[0], nxt)
            assert_true(rg[1] > rg[0], "empty segment")
            nxt = rg[1]
            var q = prog.seg_qubits(i)
            assert_equal(len(q), prog.nr)
            if prog.seg_flag(i) == SEG_DIRECT:
                dl += 1
                assert_equal(i, r.s0, "direct load only into a run's first set")
                for x in q:
                    assert_true(x >= prog.low, "direct load into a set with a low qubit")
            for j in range(1, len(q)):
                assert_true(q[j] > q[j - 1], "register set must be ascending")
            for o in range(rg[0], rg[1]):
                var kind = Int(prog.ops[8 * o])
                if kind != 3 and kind != 6:  # every kind but DG / DGT targets registers
                    assert_equal(q[Int(prog.ops[8 * o + 5])], Int(prog.ops[8 * o + 1]))
                    if kind == 1 or kind == 2:
                        assert_equal(q[Int(prog.ops[8 * o + 6])], Int(prog.ops[8 * o + 2]))
        assert_equal(nxt, r.o1 if r.s1 > r.s0 else r.o0)
    return (dl, ds)


def check_register_controls[dt: DType](emulate: Bool, tol: Float64) raises:
    """Controls and diagonal qubits inside the register set, outside it but in
    the tile, and outside the tile (device records split the masks: g0 test +
    per-register bits); every controlled kind, plus real (G1R) and complex
    (G1) 2x2 gates."""
    for n in [5, 12, 20]:
        var c = Circuit(n)
        for q in range(n):
            c.h(q)
            c.rz(0.1 * Float64(q + 1), q)
        var t = n - 1
        for rep in range(3):
            for ctl in [n - 2, n - 3, 0, 1, n // 2]:
                if ctl == t or ctl < 0:
                    continue
                c.cx(ctl, t)
                c.cry(0.3 + 0.1 * Float64(rep), ctl, t)
                c.crx(0.7, ctl, t)
                c.cp(0.4, ctl, t)
                c.crz(0.9, t, ctl)
                c.rzz(0.5, ctl, t)
                c.ry(0.2, t)
                c.u(0.3, 0.2, 0.1, t)
                if n >= 4:
                    var o = 0 if ctl != 0 else 2
                    if o != t and o != ctl:
                        c.ccx(ctl, o, t)
                        c.ccz(ctl, o, t)
                        c.cswap(o, ctl, t)
            t = n - 2 if t == n - 1 else n - 1
        var cpu = evolve(c)
        var g = evolve_gpu[dt](c, emulate=emulate)
        var d = max_diff(cpu, g.to_host())
        assert_true(d < tol, "register-set controls n=" + String(n) + " diff " + String(d))


def check_tiles() raises:
    """Host tile scheduler: coverage, capacity, free diagonal gates, geometry."""
    var tb = tile_bits[DType.float32]()
    # geometry: >= 64 tiles when possible, a single tile for small states
    var g20 = tile_geometry(20, tb)
    assert_equal(g20[0], tb)
    assert_equal(g20[1], 5)
    var g12 = tile_geometry(12, tb)
    assert_true(g12[0] <= 12 - 3)
    var g7 = tile_geometry(7, tb)
    assert_equal(g7[0], 7)
    # launch limits at 32 qubits (AMD HSA: < 2^32 work items per launch)
    var p32 = TileProgram[DType.float32](32, tb)
    assert_true(p32.num_blocks() * p32.threads_per_block() < (1 << 31))
    assert_true(stride_blocks(1 << 32) * TPB < (1 << 31))
    assert_equal(stride_blocks(1 << 32), MAX_GRID_BLOCKS)
    # every op scheduled exactly once; QFT's controlled phases are diagonal and
    # take no tile capacity, so a 20-qubit QFT needs only a few runs
    var c = qft(20)
    var plan = build_plan(c, List[Float64]())
    var prog = build_tile_program[DType.float32](plan, 20, 0.0, tb, 5, 0)  # no diagonal fusion
    assert_equal(prog.num_ops(), len(plan))
    assert_true(len(prog.runs) <= 6, "qft(20): " + String(len(prog.runs)) + " runs")
    # diagonal fusion: the QFT's controlled phases become phase tables (TK.DGT)
    var pf = build_tile_program[DType.float32](plan, 20, 0.0, tb)
    assert_equal(len(pf.runs), len(prog.runs))
    var ndg = 0
    var ndgt = 0
    for o in range(pf.num_ops()):
        if Int(pf.ops[8 * o]) == 3:
            ndg += 1
        elif Int(pf.ops[8 * o]) == 6:
            ndgt += 1
    assert_true(ndgt > 0 and pf.num_ops() * 3 < prog.num_ops() * 2,
                "qft(20) diagonal fusion: " + String(prog.num_ops()) + " -> " + String(pf.num_ops()) + " ops")
    _ = check_segments(pf)
    var covered = 0
    for r in prog.runs:
        covered += r.o1 - r.o0
        assert_true(r.s1 >= r.s0)
    assert_equal(covered, len(plan))
    _ = check_segments(prog)
    # real 1q matrices (H, RY) take the G1R kind (4 matrix values), complex ones G1 (8)
    var rc = Circuit(14)
    for q in range(14):
        rc.ry(0.1 * Float64(q + 1), q)
        rc.rx(0.2, q)
    var rp = build_tile_program[DType.float32](build_plan(rc, List[Float64]()), 14, 0.0, tb)
    var ng1 = 0
    var ng1r = 0
    for o in range(rp.num_ops()):
        if Int(rp.ops[8 * o]) == 5:
            ng1r += 1
        elif Int(rp.ops[8 * o]) == 0:
            ng1 += 1
    var rr = Circuit(14)
    for q in range(14):
        rr.h(q)
        rr.ry(0.3, q)
    var rp2 = build_tile_program[DType.float32](build_plan(rr, List[Float64]()), 14, 0.0, tb)
    var all_real = True
    for o in range(rp2.num_ops()):
        if Int(rp2.ops[8 * o]) != 5:
            all_real = False
    assert_true(all_real, "H.RY products are real: G1R expected")
    assert_true(ng1 > 0, "RX.RY products are complex: G1 expected")
    assert_equal(len(rp2.dops), 4 * rp2.num_ops())
    _ = check_segments(build_tile_program[DType.float32](build_plan(random_circuit(20, 300, UInt64(5)), List[Float64]()), 20, 0.0, tb))
    var d20 = check_segments(build_tile_program[DType.float32](build_plan(layered_circuit(20, 6), List[Float64]()), 20, 0.0, tb))
    assert_true(d20[0] > 0 and d20[1] > 0, "layered(20): no direct load / store to test")
    _ = check_segments(build_tile_program[DType.float64](build_plan(random_circuit(13, 200, UInt64(6)), List[Float64]()), 13, 0.0, tile_bits[DType.float64]()))
    # a control far outside the tile: CX(19 -> 0) and CCX(18, 19 -> 1) stay single runs
    var cx = Circuit(20)
    for q in range(20):
        cx.h(q)
    cx.cx(19, 0)
    cx.ccx(18, 19, 1)
    cx.cz(19, 17)
    cx.rzz(0.3, 19, 0)
    cx.crz(0.4, 19, 2)
    var px = build_tile_program[DType.float32](build_plan(cx, List[Float64]()), 20, 0.25, tb)
    assert_true(len(px.runs) <= 3, "controls outside the tile: " + String(len(px.runs)) + " runs")
    var cpu = evolve(cx)
    var gx = evolve_gpu[DType.float64](cx, emulate=True)
    assert_true(max_diff(cpu, gx.to_host()) < 1e-12, "controls / diagonals outside the tile")


def check_split_policy() raises:
    """Split policy: split_1q splits only gates of the form diagonal . real 2x2 (never more
    multiply-adds than the dense gate), and the scheduler counts what the
    split emits exactly (no runs cut early)."""
    var rzry = Circuit(1)
    rzry.ry(-0.4, 0)
    rzry.rz(0.9, 0)
    var rzh = Circuit(1)
    rzh.h(0)
    rzh.rz(0.3, 0)
    var u3c = Circuit(1)
    u3c.u(0.7, 0.3, -1.1, 0)
    var p1 = build_plan(rzry, List[Float64]())
    var p2 = build_plan(rzh, List[Float64]())
    var p3 = build_plan(u3c, List[Float64]())
    assert_true(split_1q_ok(p1[0].m), "RZ . RY(-0.4) must split")
    assert_true(split_1q_ok(p2[0].m), "RZ . H must split")
    assert_true(not split_1q_ok(p3[0].m), "a generic U3 must stay dense")
    # diagonal / real / X / generic-U3 gates emit the same ops with and without
    # the split: same runs (the scheduler used to count 3 ops per 1q gate)
    var n = 17
    var dc = Circuit(n)
    for r in range(8):
        for q in range(n):
            dc.rz(0.1 * Float64(q + r), q)
            dc.cp(0.3, q, (q + 5) % n)
            dc.t(q)
            dc.h(q)
            if r % 2 == 0:
                dc.u(0.7, 0.3 + 0.01 * Float64(q), -1.1, q)
            dc.x(q)
        for q in range(0, n - 1, 2):
            dc.cx(q, q + 1)
    for r in range(20):  # diagonal-heavy: runs bounded by ops, not tile capacity (22 runs; 27 with the old count)
        for q in range(n):
            dc.rz(0.2 + 0.01 * Float64(q + r), q)
        for q in range(n):
            dc.cp(0.4, q, (q + 5) % n)
    var plan = build_plan(dc, List[Float64]())
    var tb = tile_bits[DType.float64]()
    var a = build_tile_program[DType.float64](plan, n, 0.0, tb, 5, 6, False)
    var b = build_tile_program[DType.float64](plan, n, 0.0, tb, 5, 6, True)
    assert_equal(len(b.runs), len(a.runs))
    assert_equal(b.num_ops(), a.num_ops())
    var cpu = evolve(dc)
    var g = GpuStateVector[DType.float64](n, True)
    g.profile.split_1q_fp64 = True
    g.evolve(dc)
    assert_true(max_diff(cpu, g.to_host()) < 1e-12, "split policy circuit")


def check_wide[dt: DType](emulate: Bool, tol: Float64) raises:
    """24-qubit random circuit + reductions at that width (device suite)."""
    var n = 24
    var c = random_circuit(n, 120, UInt64(2024))
    var cpu = evolve(c)
    var g = evolve_gpu[dt](c, emulate=emulate)
    var d = max_diff(cpu, g.to_host())
    assert_true(d < tol, "24-qubit random circuit diff " + String(d))
    assert_true(abs(g.norm_sq() - 1.0) < 1e-4)
    var h = random_pauli_sum(n, 2 * EXP_G + 3, UInt64(77))
    var ec = cpu.expectation(h)
    var eg = g.expectation(h)
    assert_true(abs(ec - eg) < 1e-4, "24-qubit expectation " + String(ec) + " vs " + String(eg))
    var pg = g.probabilities()
    var pc = cpu.probabilities()
    var dp = 0.0
    for j in range(len(pc)):
        dp = max(dp, abs(pg[j] - pc[j]))
    assert_true(dp < tol, "24-qubit probabilities diff " + String(dp))
    var idx = g.sample(20000, 9)
    var hits = 0.0
    for i in idx:
        hits += pc[i]
    # sampled states carry real probability mass (zero-probability states never drawn)
    for i in idx:
        assert_true(pc[i] > 0.0)
    assert_true(hits > 0.0)


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
    assert_true(js.find("\"profile\":{") >= 0, js)
    var pe = GpuProfile.emulator()
    assert_true(pe.low[DType.float32]() >= 3 and pe.low[DType.float32]() <= 5)
    assert_true(pe.to_json().find("\"low_bits_fp32\":") >= 0)
    assert_true(pe.to_json().find("\"tile_bits_fp64\":") >= 0)
    # tile size policy from reported shared memory (no device needed)
    var t4 = GpuProfile("nvidia")  # T4: 64 KiB per block (opt-in) and per SM
    t4.smem_per_block = 48 * 1024
    t4.smem_per_block_optin = 64 * 1024
    t4.smem_per_multiprocessor = 64 * 1024
    assert_equal(t4._pick_tile_bits(4, tile_bits[DType.float32]()), tile_bits[DType.float32]())
    assert_equal(t4._pick_tile_bits(8, tile_bits[DType.float64]()), tile_bits[DType.float64]())
    assert_true(not t4.wants_kernel_limits(4) and not t4.wants_kernel_limits(8))
    var a100 = GpuProfile("nvidia")  # A100: 163 KiB opt-in, 164 KiB per SM
    a100.smem_per_block = 48 * 1024
    a100.smem_per_block_optin = 163 * 1024
    a100.smem_per_multiprocessor = 164 * 1024
    var tb32 = tile_bits[DType.float32]()
    var tb64 = tile_bits[DType.float64]()
    # kernel limits unknown: only blocks that launch with any register count
    var k32 = a100._pick_tile_bits(4, tb32)
    var k64 = a100._pick_tile_bits(8, tb64)
    assert_true((1 << (k32 - REG_BITS)) <= TILE_SAFE_THREADS and (1 << (k64 - REG_BITS)) <= TILE_SAFE_THREADS,
                "A100 without kernel limits: " + String(k32) + "/" + String(k64))
    # a kernel report below 256 threads shrinks even the default tile
    var small = GpuProfile("nvidia")
    small.kl_fp32 = TileKernelLimits(128, -1, 255, 0)
    assert_true((1 << (small._pick_tile_bits(4, tb32) - REG_BITS)) <= 128)
    comptime if TILE_SMEM_DYNAMIC:  # larger tiles need the dynamic shared region
        assert_true(a100.wants_kernel_limits(4), "A100 fp32: a larger tile is possible")
        # compiled kernel admits 1024 threads, occupancy unknown: shared memory decides
        a100.kl_fp32 = TileKernelLimits(1024, -1, 60, 0)
        a100.kl_fp64 = TileKernelLimits(1024, -1, 60, 0)
        assert_equal(a100._pick_tile_bits(4, tb32), tb32 + 2)
        assert_equal(a100._pick_tile_bits(8, tb64), tb64 + 2)
        # 512 threads at most (e.g. 100 registers per thread): one size less
        a100.kl_fp32 = TileKernelLimits(512, -1, 100, 0)
        var k512 = a100._pick_tile_bits(4, tb32)
        assert_true((1 << (k512 - REG_BITS)) <= 512 and k512 >= tb32, "A100 fp32 at 512 threads: " + String(k512))
        assert_true(not a100._tile_fits(4, k512 + 1) or (1 << (k512 + 1 - REG_BITS)) <= 512)
        # launchable, but < 2 resident blocks above 256 threads: stay at 256
        a100.kl_fp32 = TileKernelLimits(1024, TILE_SAFE_THREADS, 120, 0)
        assert_true((1 << (a100._pick_tile_bits(4, tb32) - REG_BITS)) <= TILE_SAFE_THREADS)
        # the env override is checked against the same launch limit
        a100.kl_fp32 = TileKernelLimits.unknown()
        assert_true(not a100._tile_fits(4, REG_BITS + 10), "1024 threads must not fit without a kernel report")
    else:
        assert_equal(a100._pick_tile_bits(4, tb32), tb32)
    comptime if TILE_LAUNCH_BOUND > 0:  # AMD: the launch bound caps the block whatever the driver reports
        a100.kl_fp32 = TileKernelLimits(1024, -1, 60, 0)
        assert_true((1 << (a100._pick_tile_bits(4, tb32) - REG_BITS)) <= TILE_LAUNCH_BOUND)
    var l4 = GpuProfile("nvidia")  # L4: 99 KiB opt-in, 100 KiB per SM
    l4.smem_per_block = 48 * 1024
    l4.smem_per_block_optin = 99 * 1024
    l4.smem_per_multiprocessor = 100 * 1024
    assert_true((1 << (l4._pick_tile_bits(4, tb32) - REG_BITS)) <= TILE_SAFE_THREADS)
    assert_true(l4.tile_smem_optin() == (99 * 1024 - 1024 if TILE_SMEM_DYNAMIC else 0))
    assert_equal(t4.tile_smem_optin(), 63 * 1024 if TILE_SMEM_DYNAMIC else 0)
    var apple = GpuProfile("apple")  # 32 KiB per threadgroup, per-core size not reported
    apple.smem_per_block = 32 * 1024
    assert_equal(apple.tile_smem_optin(), 0)
    assert_true(not apple.wants_kernel_limits(4))
    assert_equal(apple._pick_tile_bits(4, tile_bits[DType.float32]()), tile_bits[DType.float32]())
    assert_true(not apple._tile_fits(4, tile_bits[DType.float32]() + 1))
    assert_true(apple._tile_fits(4, tile_bits[DType.float32]()))
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


def check_low_bits[dt: DType](emulate: Bool, tol: Float64, big: Bool) raises:
    """Tiles with 3 or 4 always-resident low qubits (GpuProfile.low, the
    per-device knob): same states, fewer runs on wide circuits."""
    var ns: List[Int] = [6, 11]
    if big:
        ns.append(20)
    for n in ns:
        var c = random_circuit(n, 120, UInt64(300 + n))
        var cpu = evolve(c)
        var lc = layered_circuit(n, 6)
        var lcpu = evolve(lc)
        var runs5 = 0
        for lb in [5, 4, 3]:
            var g = GpuStateVector[dt](n, emulate)
            g.profile.low_fp32 = lb
            g.profile.low_fp64 = lb
            g.evolve(c)
            var d = max_diff(cpu, g.to_host())
            assert_true(d < tol, "low=" + String(lb) + " random n=" + String(n) + " diff " + String(d))
            g.reset()
            g.evolve(lc)
            var dl = max_diff(lcpu, g.to_host())
            assert_true(dl < tol, "low=" + String(lb) + " layered n=" + String(n) + " diff " + String(dl))
            if lb == 5:
                runs5 = g.runs_last
            elif n >= 20:
                assert_true(g.runs_last <= runs5, "fewer low qubits must not add runs")


def check_tile_bits[dt: DType](emulate: Bool, tol: Float64, big: Bool) raises:
    """Tile size is a runtime value (GpuProfile.tile_bits, dynamic shared
    memory sized per launch): smaller and larger tiles than the default give
    the same states (larger ones only where the device's shared memory fits
    them; the emulator has no limit)."""
    var tb0 = tile_bits[dt]()
    var n = 20 if big else 14
    var c = random_circuit(n, 150, UInt64(400 + n))
    var cpu = evolve(c)
    var lc = layered_circuit(n, 6)
    var lcpu = evolve(lc)
    for tb in [tb0 - 1, tb0 + 1, tb0 + 2]:
        var g = GpuStateVector[dt](n, emulate)
        if not emulate and not g.profile._tile_fits(size_of[dt](), tb):
            continue
        g.profile.tile_bits_fp32 = tb
        g.profile.tile_bits_fp64 = tb
        g.evolve(c)
        var d = max_diff(cpu, g.to_host())
        assert_true(d < tol, "tile bits " + String(tb) + " random n=" + String(n) + " diff " + String(d))
        g.reset()
        g.evolve(lc)
        var dl = max_diff(lcpu, g.to_host())
        assert_true(dl < tol, "tile bits " + String(tb) + " layered n=" + String(n) + " diff " + String(dl))


def check_diag_fusion[dt: DType](emulate: Bool, tol: Float64, big: Bool) raises:
    """Fused diagonal phase tables (TK.DGT) of every width against the CPU:
    QFT (controlled phases), a random circuit (RZ / CP / CRZ / RZZ / CCZ /
    controlled diagonals mixed with everything else), at widths 0 (off) .. 6,
    with and without complex 1q gates split into diagonal . real 2x2 .
    diagonal (GpuProfile.split_1q; incl. |U00| = 0 and nearly diagonal U)."""
    var n = 16 if big else 9
    var circs = List[Circuit]()
    circs.append(qft(n))
    var dc = Circuit(n)
    for q in range(n):
        dc.h(q)
    for r in range(4):
        for q in range(n):
            dc.rz(0.1 * Float64(q + r), q)
            dc.cp(0.3 + 0.05 * Float64(q), q, (q + 1 + r) % n)
            dc.rzz(0.2, q, (q + 3) % n)
            dc.crz(0.4, (q + 2) % n, q)
            if n >= 3:
                dc.ccz(q, (q + 1) % n, (q + 2) % n)
            dc.t(q)
        dc.cx(0, n - 1)
        dc.ry(0.3, r)
    circs.append(dc^)
    circs.append(random_circuit(n, 200, UInt64(500 + n)))
    circs.append(layered_circuit(n, 5))
    var uc = Circuit(n)  # complex 1q gates incl. |U00| = 0 and |U10| tiny
    for q in range(n):
        uc.u(0.7 + 0.1 * Float64(q), 0.3, -1.1, q)
        uc.x(q)
        uc.rz(0.2, q)
        uc.u(3.141592653589793, 0.4, 0.9, q)
        uc.u(1e-9, 0.5, 0.25, q)
        uc.cx(q, (q + 1) % n)
        uc.sx(q)
        uc.y(q)
    circs.append(uc^)
    var sc = Circuit(n)  # split-eligible D . R products: tiny / negative / pi angles, RZ . H, phases . X
    for q in range(n):
        sc.ry(1e-9, q)
        sc.rz(0.5 + 0.1 * Float64(q), q)
        sc.cx(q, (q + 1) % n)
        sc.ry(-0.7 - 0.05 * Float64(q), q)
        sc.rz(1.1, q)
        sc.cx((q + 2) % n, q)
        sc.h(q)
        sc.rz(-2.3, q)
        sc.cz(q, (q + 3) % n)
        sc.ry(3.141592653589793, q)
        sc.t(q)
        sc.cx(q, (q + 4) % n)
        sc.x(q)
        sc.s(q)
    circs.append(sc^)
    for ci in range(len(circs)):
        var cpu = evolve(circs[ci])
        for dw in [0, 2, 4, 6]:
            for split in [False, True]:
                var g = GpuStateVector[dt](n, emulate)
                g.profile.diag_width = dw
                g.profile.split_1q_fp32 = split
                g.profile.split_1q_fp64 = split
                g.evolve(circs[ci])
                var d = max_diff(cpu, g.to_host())
                assert_true(d < tol, "diag width " + String(dw) + " split " + String(split) + " circuit " + String(ci)
                            + " diff " + String(d))


def run_suite[dt: DType](emulate: Bool, tol: Float64, big: Bool, exact_sampling: Bool) raises:
    check_gates[dt](emulate, tol)
    check_diag_fusion[dt](emulate, tol, big)
    check_register_controls[dt](emulate, tol)
    check_low_bits[dt](emulate, tol, big)
    check_tile_bits[dt](emulate, tol, big)
    check_circuits[dt](emulate, tol, big)
    check_expectation[dt](emulate, tol, big)
    check_sampling[dt](emulate, exact_sampling)


def main() raises:
    # host emulation of the exact kernel bodies and launch grids
    check_tiles()
    check_split_policy()
    run_suite[DType.float64](True, 1e-12, True, True)
    run_suite[DType.float32](True, 2e-5, True, False)
    check_errors()
    check_integration()
    check_tools()
    comptime if is_defined["QUMO_GPU"]():
        if gpu_available():
            print("running device suites on:", gpu_info())
            run_suite[DType.float32](False, 2e-5, True, False)
            check_wide[DType.float32](False, 2e-5)
            comptime if GPU_FP64_SUPPORTED:
                run_suite[DType.float64](False, 1e-12, True, False)
                check_wide[DType.float64](False, 1e-12)
        else:
            var why = "no usable GPU device: device suites skipped (" + gpu_info() + ")"
            if getenv("QUMO_GPU_REQUIRE_DEVICE") == "1":  # make test-gpu: a skip is a failure
                raise Error("QUMO_GPU_REQUIRE_DEVICE=1 but " + why)
            print(why)
    print("test_gpu OK")
