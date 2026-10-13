"""GPU vs CPU benchmarks: state vector (circuit families) and QEC Pauli frames.

    make bench-gpu                          # = build with -D QUMO_GPU, default mode up to 26 qubits
    ./build/gpu/bench_gpu 28 5              # default mode: nmax, repetitions (medians)
    ./build/gpu/bench_gpu 28 3 families 20 26   # layered/qft/random, n = 20..28, CPU up to 26
    ./build/gpu/bench_gpu 22 5 crossover 10     # end-to-end sample_gpu vs CPU sample (dispatch threshold)
    ./build/gpu/bench_gpu 9 3 frames 1000000    # Pauli frames, d = 3..9, shots
    ./build/gpu/bench_gpu --stim 5 0.001        # the frames circuit in Stim syntax (for Stim timings)
    ./build/gpu/bench_gpu 26 3 probe            # engine cost model on this device (n = 24, 26)

Every comparison is interleaved A/B (CPU run, then GPU run, per repetition)
and reports medians (upper median for even counts), so slow drifts of a
shared machine hit both sides alike.

Default mode (the output contract of bench/run_benchmarks.py, unchanged):
  * table 1, layered circuit of bench_sv.mojo, n = 16, 18, ..., nmax: CPU
    `evolve` (fused plan, SIMD + threads + qubit reordering) vs GPU
    `reset + evolve + synchronize` on a preallocated `GpuStateVector` (device
    allocation and the one-off DeviceContext / kernel load excluded; a warm-up
    run precedes the timed runs; GPU repetitions run back to back, then the CPU
    ones), plus GPU expectation (sum of n-1 ZZ terms)
    and 10k-shot sampling; fidelity against the CPU (computed up to 26 qubits,
    -1 above, to bound host memory);
  * table 2, rotated surface-code memory circuits (d = 3, 5, 7; rounds = d;
    p = 0.001; 1M shots): CPU `FrameSimulator` vs `GpuFrameSampler`.

`families`: the three circuit families of bench_sv.mojo / bench_sv_aer.py
(gate for gate the same circuits), GPU time plus tile runs / transposes, CPU
time for n <= cpu_max (-1 above). `crossover`: end-to-end `sample_gpu`
(DeviceContext, allocation, evolve, sampling) vs CPU `sample`, the numbers
behind GPU_MIN_QUBITS. `frames`: table 2 for d = 3, 5, ..., dmax.

`probe`: the numbers that decide how to tune the tile engine on a device
(docs/GPU.md, "Probe"): device facts (multiprocessors, shared memory, warp
width, compute capability / arch, kernel registers / spills / occupancy where
the API offers them), the time of an empty tile run (one state sweep), the
cost per op kind (slope between short and long single-segment runs), the cost
of one register-set transpose, the bandwidth of tile-shaped streaming with 32,
64 and 128 B contiguous chunks, and evolve time / runs / transposes / ops of
the three families with the ratio evolve / (runs x sweep). One line per
number: `PROBE,<dtype>,<n>,<metric>,<value>,<unit>`; fp32, plus fp64 where
the device has it.

Default mode also prints `gpu_alloc_s`: GpuStateVector.on_context(ctx, n)
(allocation of the two state buffers on the existing context) + evolve +
synchronize, the timed region CUDA-Q and Aer have (they allocate and
initialise the state on every call). Freeing the state is outside the timer,
as for CUDA-Q.

Without -D QUMO_GPU the binary only reports that GPU support is missing (and
runs a tiny emulator smoke check so the harness itself stays tested).
"""

from std.time import perf_counter_ns
from std.sys import argv
from std.sys.defines import is_defined
from std.math import pi

from qumo.ir.circuit import Circuit
from qumo.ir.pauli import PauliSum
from qumo.sim.runner import evolve, sample
from qumo.sim.frame import FrameSimulator
from qumo.qec.codes import generate_code, NoiseParams
from qumo.gpu.frame_gpu import GpuFrameSampler
from qumo.gpu.statevector_gpu import (
    GpuStateVector, sample_gpu, tile_func_attribute, tile_kernel_attr, tile_kernel_occupancy,
)
from qumo.gpu.device import gpu_info, gpu_available, GPU_DEFAULT_DTYPE, GPU_FP64_SUPPORTED, precision_name
from qumo.gpu.kernels import GPtr, tile_bits, DIAG_MAX_WIDTH, REG_BITS, tile_smem_typical
from qumo.gpu.tiles import TileProgram, kop_targets
from qumo.sim.runner import KOp, K
from qumo.ir.gates import Op, M2, u3_matrix, matrix_2q, matrix_1q
from qumo.ir.circuit import Instruction
from std.bit import pop_count
from std.sys.info import size_of
from max.gpu import thread_idx, block_idx
from max.gpu.host import DeviceContext, DeviceAttribute, Attribute


# ----------------------------------------------------------------------------
# Circuit families: gate for gate those of bench/bench_sv.mojo and
# bench/bench_sv_aer.py (same angles, qubit order and 64-bit LCG stream).
# ----------------------------------------------------------------------------
def layered(n: Int, depth: Int) -> Circuit:
    var c = Circuit(n)
    for d in range(depth):
        for q in range(n):
            c.ry(0.1 + 0.01 * Float64(q + d), q)
            c.rz(0.2 + 0.03 * Float64(q * d), q)
        for q in range(d % 2, n - 1, 2):
            c.cx(q, q + 1)
    return c^


def qft(n: Int) -> Circuit:
    var c = Circuit(n)
    var j = n - 1
    while j >= 0:
        c.h(j)
        var k = j - 1
        while k >= 0:
            c.cp(pi / Float64(1 << (j - k)), k, j)
            k -= 1
        j -= 1
    for q in range(n // 2):
        c.swap(q, n - 1 - q)
    return c^


struct Lcg:
    """64-bit LCG (Knuth MMIX constants); output = state >> 33."""

    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed

    def next(mut self) -> Int:
        self.s = self.s * 6364136223846793005 + 1442695040888963407
        return Int(self.s >> 33)

    def angle(mut self) -> Float64:
        return Float64(self.next() % 1000) / 1000.0 * pi


def random_pairs(n: Int, depth: Int, seed: UInt64 = 7) -> Circuit:
    var rng = Lcg(seed)
    var c = Circuit(n)
    for _ in range(depth):
        for q in range(n):
            var th = rng.angle()
            var ph = rng.angle()
            var la = rng.angle()
            c.u(th, ph, la, q)
        var perm = List[Int]()
        for q in range(n):
            perm.append(q)
        var i = n - 1
        while i > 0:
            var j = rng.next() % (i + 1)
            var t = perm[i]
            perm[i] = perm[j]
            perm[j] = t
            i -= 1
        var k = 0
        while k + 1 < n:
            c.cx(perm[k], perm[k + 1])
            k += 2
    return c^


def make(family: String, n: Int) -> Circuit:
    if family == "qft":
        return qft(n)
    if family == "random":
        return random_pairs(n, 10)
    return layered(n, 10)


def zz_chain(n: Int) -> PauliSum:
    var h = PauliSum()
    for q in range(n - 1):
        h.add(1.0, 0, (UInt64(3) << UInt64(q)))
    return h^


def median(var xs: List[Float64]) -> Float64:
    sort(xs)
    return xs[len(xs) // 2]


def secs(t0: Int) -> Float64:
    return Float64(perf_counter_ns() - t0) / 1e9


# ----------------------------------------------------------------------------
# Default mode, table 1 (contract: bench/run_benchmarks.py parses it)
# ----------------------------------------------------------------------------
def bench_width[dt: DType](n: Int, reps: Int, emulate: Bool, cpu_max: Int = 26) raises:
    """One row of table 1. The CPU reference (timing, expectation, fidelity)
    runs only for n <= cpu_max, so wide GPU widths need no 16 B/amplitude of
    host RAM (cpu_s, speedup, abs_err_expect and fidelity are -1 above)."""
    var c = layered(n, 10)
    var h = zz_chain(n)
    var g = GpuStateVector[dt](n, emulate)
    g.evolve(c)  # warm-up (module load, allocator)
    g.synchronize()
    _ = g.expectation(h)  # warm-up of the read-out kernels: their first-call
    _ = g.sample(100, 1)  # module load must not land in gpu_expect_s / gpu_sample10k_s
    var cpu_t = List[Float64]()
    var gpu_t = List[Float64]()
    var cpu_e = 0.0
    # GPU repetitions run back to back, without CPU work in between, so they
    # are timed under the same conditions as the other engines' GPU rows
    # (CUDA-Q, Aer GPU); interleaving the multi-threaded fp64 CPU reference
    # shifted gpu_s systematically (measured on the M5).
    for _ in range(reps):
        var t1 = perf_counter_ns()
        g.reset()
        g.evolve(c)
        g.synchronize()
        gpu_t.append(secs(t1))
    if n <= cpu_max:
        _ = evolve(c)  # CPU warm-up for this width
        for _ in range(reps):
            var t0 = perf_counter_ns()
            var sv = evolve(c)
            cpu_t.append(secs(t0))
            cpu_e = sv.expectation(h)
    var runs = g.runs_last  # state sweeps (tile-run launches) of one evolve
    var method = g.method_name()
    var times = String("")
    for i in range(len(gpu_t)):
        if i > 0:
            times += ";"
        times += String(gpu_t[i])
    var t1 = perf_counter_ns()
    var gpu_e = g.expectation(h)
    var exp_s = secs(t1)
    var t2 = perf_counter_ns()
    _ = g.sample(10000, 7)
    var smp_s = secs(t2)
    var fid = -1.0
    var err = -1.0
    if n <= cpu_max:
        fid = evolve(c).fidelity(g.to_host())
        err = abs(cpu_e - gpu_e)
    # allocation-inclusive time (CUDA-Q / Aer allocate + initialise per call):
    # new state buffers on the same context + evolve + synchronize; `g` is
    # released first so the widest widths still fit in device memory
    var alloc_t = List[Float64]()
    comptime if is_defined["QUMO_GPU"]():
        if not emulate:
            var ctx = g.ctx.value()
            g = GpuStateVector[dt](1, emulate)  # frees the n-qubit buffers
            ctx.synchronize()
            for _ in range(reps):
                ctx.synchronize()  # the previous rep's frees complete outside the timer
                var t3 = perf_counter_ns()
                var ga = GpuStateVector[dt].on_context(ctx, n)
                ga.evolve(c)
                ga.synchronize()
                alloc_t.append(secs(t3))
                _ = ga.size()  # ASAP destruction: free the buffers after the timer, not inside it
    var ta = median(alloc_t^) if len(alloc_t) > 0 else -1.0
    var tg = median(gpu_t^)
    var tc = -1.0
    var ratio = -1.0
    if len(cpu_t) > 0:
        tc = median(cpu_t^)
        ratio = tc / tg
    print(n, ",", len(c), ",", tc, ",", tg, ",", ratio, ",", Float64(len(c)) / tg, ",",
          exp_s, ",", smp_s, ",", err, ",", fid, ",", method, ",", runs, ",", times, ",", ta)


# ----------------------------------------------------------------------------
# families / crossover modes
# ----------------------------------------------------------------------------
def bench_family[dt: DType](family: String, n: Int, reps: Int, cpu_max: Int) raises:
    var c = make(family, n)
    var g = GpuStateVector[dt](n)
    g.evolve(c)  # warm-up
    g.synchronize()
    var cpu_t = List[Float64]()
    var gpu_t = List[Float64]()
    for _ in range(reps):  # interleaved A/B
        if n <= cpu_max:
            var t0 = perf_counter_ns()
            var sv = evolve(c)
            cpu_t.append(secs(t0))
            _ = sv.size()
        var t1 = perf_counter_ns()
        g.reset()
        g.evolve(c)
        g.synchronize()
        gpu_t.append(secs(t1))
    var fid = -1.0
    if n <= min(cpu_max, 26):
        fid = evolve(c).fidelity(g.to_host())
    var tg = median(gpu_t^)
    var tc = -1.0
    var ratio = -1.0
    if len(cpu_t) > 0:
        tc = median(cpu_t^)
        ratio = tc / tg
    print(family, ",", n, ",", len(c), ",", g.runs_last, ",", g.transposes_last, ",", tc, ",", tg, ",",
          ratio, ",", Float64(len(c)) / tg, ",", fid, ",", g.method_name())


def bench_crossover(n: Int, reps: Int) raises:
    var c = make("layered", n)
    c.measure_all()
    _ = sample_gpu(c, 100, 1)  # warm-up: kernel load
    var cpu_t = List[Float64]()
    var gpu_t = List[Float64]()
    for r in range(reps):  # interleaved A/B, end to end
        var t0 = perf_counter_ns()
        _ = sample(c, 1000, UInt64(r + 1))
        cpu_t.append(secs(t0))
        var t1 = perf_counter_ns()
        _ = sample_gpu(c, 1000, UInt64(r + 1))
        gpu_t.append(secs(t1))
    var tc = median(cpu_t^)
    var tg = median(gpu_t^)
    print(n, ",", len(c), ",", tc, ",", tg, ",", tc / tg)


# ----------------------------------------------------------------------------
# probe mode: the cost model of the tile engine on this device
# ----------------------------------------------------------------------------
def pline(dt: String, n: Int, metric: String, value: String, unit: String = ""):
    print("PROBE,", dt, ",", n, ",", metric, ",", value, ",", unit, sep="")


def short_err(e: Error) -> String:
    """First line of an error, without commas (one CSV field)."""
    var t = String(e)
    var cut = t.find("\n")
    var u = String(t[byte=0:cut]) if cut >= 0 else t
    return u.replace(",", ";")


def gk_probe_stream[dt: DType](re: GPtr[dt], im: GPtr[dt], c: Int32, k: Int32, top: Int32, a: Scalar[dt]):
    """Tile-shaped read + write of both arrays. Block = 2^k amplitudes made of
    2^c contiguous ones (c low qubits) times 2^(k-c) chunks on the top k-c
    qubits (from `top` up: the farthest-apart chunks a tile can have);
    8 amplitudes per thread with gk_tile's natural map l = tid + j 2^(k-3).
    c = k streams whole contiguous tiles."""
    var tid = UInt32(thread_idx.x)
    var blk = UInt32(block_idx.x)
    var cc = UInt32(c)
    var sh = UInt32(k) - 3
    var cmask = (UInt32(1) << cc) - 1
    var gi = Array[UInt32, 8](fill=0)
    var vr = Array[Scalar[dt], 8](fill=0)
    var vi = Array[Scalar[dt], 8](fill=0)
    comptime for j in range(8):
        var l = tid | (UInt32(j) << sh)
        gi[j] = (l & cmask) | (blk << cc) | ((l >> cc) << UInt32(top))
    comptime for j in range(8):
        vr[j] = re[unsafe_offset=Int(gi[j])]
        vi[j] = im[unsafe_offset=Int(gi[j])]
    comptime for j in range(8):
        re[unsafe_offset=Int(gi[j])] = a * vr[j]
        im[unsafe_offset=Int(gi[j])] = a * vi[j]


def probe_device() raises:
    """Device facts (each query guarded: Metal raises 'Attributes not supported')."""
    var ctx = DeviceContext()
    pline("-", 0, "device_name", ctx.name())
    pline("-", 0, "api", ctx.api())
    try:
        pline("-", 0, "compute_capability", String(ctx.compute_capability()))
    except e:
        pline("-", 0, "compute_capability", "n/a", short_err(e))
    try:
        pline("-", 0, "arch_name", ctx.arch_name())
    except e:
        pline("-", 0, "arch_name", "n/a", short_err(e))
    var names: List[String] = ["MULTIPROCESSOR_COUNT", "MAX_SHARED_MEMORY_PER_MULTIPROCESSOR",
                               "MAX_SHARED_MEMORY_PER_BLOCK", "MAX_SHARED_MEMORY_PER_BLOCK_OPTIN", "WARP_SIZE",
                               "MAX_THREADS_PER_MULTIPROCESSOR", "MAX_REGISTERS_PER_MULTIPROCESSOR",
                               "MAX_BLOCKS_PER_MULTIPROCESSOR", "CLOCK_RATE"]
    var attrs: List[DeviceAttribute] = [DeviceAttribute.MULTIPROCESSOR_COUNT,
                                        DeviceAttribute.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
                                        DeviceAttribute.MAX_SHARED_MEMORY_PER_BLOCK,
                                        DeviceAttribute.MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, DeviceAttribute.WARP_SIZE,
                                        DeviceAttribute.MAX_THREADS_PER_MULTIPROCESSOR,
                                        DeviceAttribute.MAX_REGISTERS_PER_MULTIPROCESSOR,
                                        DeviceAttribute.MAX_BLOCKS_PER_MULTIPROCESSOR, DeviceAttribute.CLOCK_RATE]
    for i in range(len(attrs)):
        try:
            pline("-", 0, names[i], String(ctx.get_attribute(attrs[i])))
        except e:
            pline("-", 0, names[i], "n/a", short_err(e))


def probe_kernel_attrs[dt: DType](g: GpuStateVector[dt]) raises:
    """Registers / spills / static shared memory / max threads / occupancy of
    the tile kernels as the engine loads them (gk_tile or gk_tile_lb, with
    the device's shared-memory attribute), at the engine's tile size."""
    probe_kernel_attrs1[dt, False](g, "gk_tile_light")
    probe_kernel_attrs1[dt, True](g, "gk_tile_heavy")


def probe_kernel_attrs1[dt: DType, HEAVY: Bool](g: GpuStateVector[dt], kname: String) raises:
    var tag = precision_name[dt]()
    ref ctx = g.ctx.value()
    var fa = tile_func_attribute(g.profile)
    var k = g.profile.tile_bits[dt]()
    var nthr = 1 << (k - REG_BITS)
    var names: List[String] = ["NUM_REGS", "LOCAL_SIZE_BYTES", "SHARED_SIZE_BYTES", "MAX_THREADS_PER_BLOCK"]
    var attrs: List[Attribute] = [Attribute.NUM_REGS, Attribute.LOCAL_SIZE_BYTES, Attribute.SHARED_SIZE_BYTES,
                                  Attribute.MAX_THREADS_PER_BLOCK]
    for i in range(len(attrs)):
        try:
            pline(tag, 0, kname + "_" + names[i], String(tile_kernel_attr[dt, True, HEAVY](ctx, attrs[i], fa)))
        except e:
            pline(tag, 0, kname + "_" + names[i], "n/a", short_err(e))
    try:
        var sm = tile_smem_typical(size_of[dt](), k)
        pline(tag, 0, kname + "_blocks_per_sm", String(tile_kernel_occupancy[dt, True, HEAVY](ctx, nthr, sm, fa)),
              "at " + String(nthr) + " threads, " + String(sm) + " B dynamic shared memory (typical run)")
    except e:
        pline(tag, 0, kname + "_blocks_per_sm", "n/a", short_err(e))


def probe_kop(kind: String, i: Int, n: Int) -> KOp:
    """Op i of a synthetic single-segment run: targets cycle over the three
    top qubits (the natural register set of the run: no transposes)."""
    var t = n - 1 - (i % 3)
    var u = n - 1 - ((i + 1) % 3)
    if kind == "XC":  # CX with its control on a low qubit (tile, not registers)
        return KOp.make(K.CX, 1, t, Instruction.make(Op.CX, 1, t))
    if kind == "DG" or kind == "DGF":  # RZ anywhere (diagonal: no tile slot needed)
        var q = (7 * i) % n
        var op = KOp.make(K.ONE, q, -1, Instruction.make(Op.RZ, q))
        op.m = matrix_1q(Op.RZ, 0.1 + 0.01 * Float64(i))
        return op
    if kind == "SW":
        return KOp.make(K.SWAP, t, u, Instruction.make(Op.SWAP, t, u))
    if kind == "G2":
        var op = KOp.make(K.TWO, t, u, Instruction.make(Op.ECR, t, u))
        op.m4 = matrix_2q(Op.RXX, 0.3 + 0.01 * Float64(i))
        return op
    if kind == "G1R":  # real 2x2 (RY): the G1R path
        var op = KOp.make(K.ONE, t, -1, Instruction.make(Op.RY, t))
        op.m = matrix_1q(Op.RY, 0.3 + 0.01 * Float64(i))
        return op
    if kind == "TR":  # dense 1q gates switching between {0,1,2} and the top 3 every 4 ops
        t = (i % 3) if (i // 4) % 2 == 0 else t
    var op = KOp.make(K.ONE, t, -1, Instruction.make(Op.U, t))
    op.m = u3_matrix(0.3 + 0.01 * Float64(i), 1.1, -0.7)
    return op


def probe_program[dt: DType](n: Int, kind: String, nops: Int, copies: Int, tb: Int, low_bits: Int) raises -> TileProgram[dt]:
    """`copies` identical runs of `nops` synthetic ops (back-to-back launches).
    DG: diagonal gates one by one (fusion off); DGF: the same gates fused into
    phase tables (the default), i.e. the effective cost per diagonal gate."""
    var prog = TileProgram[dt](n, tb, low_bits, 0 if kind == "DG" else DIAG_MAX_WIDTH)
    var plan = List[KOp]()
    var smask = 0
    for i in range(nops):
        var op = probe_kop(kind, i, n)
        smask |= kop_targets(op)
        plan.append(op)
    smask &= ~((1 << prog.low) - 1)
    var sel = List[Int]()
    for i in range(nops):
        sel.append(i)
    for _ in range(copies):
        if nops == 0:
            prog.add_empty_run()
        else:
            prog.add_run(plan, sel, smask, Int(pop_count(smask)))
    prog.finalize()
    return prog^


def time_program[dt: DType](mut g: GpuStateVector[dt], kind: String, nops: Int, reps: Int) raises -> Tuple[Float64, Int]:
    """Median ms per run (8 back-to-back copies per sync) and transposes per run."""
    comptime COPIES = 8
    var ts = List[Float64]()
    var tr = 0
    for _ in range(reps + 1):
        var prog = probe_program[dt](g.num_qubits, kind, nops, COPIES, g.profile.tile_bits[dt](), g.profile.low[dt]())
        g.synchronize()
        var t0 = perf_counter_ns()
        g.run_program(prog^)
        g.synchronize()
        ts.append(secs(t0) * 1e3 / Float64(COPIES))
        tr = g.transposes_last // COPIES
    _ = ts.pop(0)  # first: table buffers / kernel variant warm-up
    return (median(ts^), tr)


def probe_engine[dt: DType](n: Int, reps: Int, first: Bool) raises:
    comptime if is_defined["QUMO_GPU"]():
        var tag = precision_name[dt]()
        var g = GpuStateVector[dt](n)
        var k = g.profile.tile_bits[dt]()
        g.evolve(layered(n, 1))  # materialise the state, load the kernels
        g.synchronize()
        if first:
            pline(tag, 0, "profile", g.profile.to_json().replace(",", ";"))
        var sweep = time_program[dt](g, "G1", 0, reps)[0]
        var bytes = Float64(1 << n) * 4.0 * Float64(size_of[dt]())
        pline(tag, n, "sweep_ms", String(sweep), "empty tile run (read + write of the state)")
        pline(tag, n, "sweep_GBps", String(bytes / (sweep * 1e6)))
        var kinds: List[String] = ["G1", "G1R", "XC", "DG", "DGF", "SW", "G2"]
        for kd in kinds:
            var lo = 8 if kd == "G2" else 32
            var hi = 30 if kd == "G2" else 120
            var a = time_program[dt](g, kd, lo, reps)[0]
            var b = time_program[dt](g, kd, hi, reps)[0]
            pline(tag, n, "op_" + kd + "_us", String((b - a) * 1e3 / Float64(hi - lo)),
                  "slope " + String(lo) + "->" + String(hi) + (" gates" if kd == "DGF" else " ops"))
        var a = time_program[dt](g, "G1", 64, reps)
        var b = time_program[dt](g, "TR", 64, reps)
        if b[1] > a[1]:
            pline(tag, n, "transpose_us", String((b[0] - a[0]) * 1e3 / Float64(b[1] - a[1])),
                  String(b[1] - a[1]) + " transposes over 64 dense 1q ops")
        # tile-shaped streaming: contiguous chunk size vs bandwidth
        var chunks: List[Int] = [32, 64, 128]
        var es = size_of[dt]()
        for cb in chunks:
            var c = 0
            while (1 << c) * es < cb:
                c += 1
            probe_stream[dt](g, n, k, c, reps, tag, String(cb) + "B")
        probe_stream[dt](g, n, k, k, reps, tag, "tile")
        if first:
            probe_kernel_attrs[dt](g)
        # families: evolve vs the sweep floor (runs x empty sweep)
        var fams: List[String] = ["layered", "qft", "random"]
        for f in fams:
            var c = make(f, n)
            g.evolve(c)
            g.synchronize()
            var ts = List[Float64]()
            for _ in range(reps):
                var t0 = perf_counter_ns()
                g.reset()
                g.evolve(c)
                g.synchronize()
                ts.append(secs(t0) * 1e3)
            var ev = median(ts^)
            pline(tag, n, f + "_evolve_ms", String(ev))
            pline(tag, n, f + "_runs", String(g.runs_last))
            pline(tag, n, f + "_transposes", String(g.transposes_last))
            pline(tag, n, f + "_ops", String(g.ops_last))
            pline(tag, n, f + "_evolve_over_sweep_floor", String(ev / (Float64(g.runs_last) * sweep)))


def probe_stream[dt: DType](mut g: GpuStateVector[dt], n: Int, k: Int, c: Int, reps: Int, tag: String, label: String) raises:
    comptime if is_defined["QUMO_GPU"]():
        ref ctx = g.ctx.value()
        var top = n - (k - c)
        var ts = List[Float64]()
        comptime COPIES = 8
        for _ in range(reps + 1):
            g.synchronize()
            var t0 = perf_counter_ns()
            for _ in range(COPIES):
                ctx.enqueue_function[gk_probe_stream[dt]](
                    g.dre.value(), g.dim.value(), Int32(c), Int32(k), Int32(top), Scalar[dt](1),
                    grid_dim=1 << (n - k), block_dim=1 << (k - 3))
            ctx.synchronize()
            ts.append(secs(t0) * 1e3 / Float64(COPIES))
        _ = ts.pop(0)
        var t = median(ts^)
        pline(tag, n, "stream_" + label + "_GBps", String(Float64(1 << n) * 4.0 * Float64(size_of[dt]()) / (t * 1e6)),
              "chunk " + String(1 << c) + " amplitudes")


# ----------------------------------------------------------------------------
# Pauli frames
# ----------------------------------------------------------------------------
def frames_circuit(d: Int) raises -> Circuit:
    return generate_code("surface_z", d, d, NoiseParams.uniform(0.001)).circuit.copy()


def bench_frames(d: Int, shots: Int, reps: Int, emulate: Bool) raises:
    var c = frames_circuit(d)
    var fs = FrameSimulator(c)
    var gs = GpuFrameSampler(c, emulate)
    _ = gs.sample_detectors(min(shots, 4096), 1)  # warm-up (module load, allocator)
    var cpu_t = List[Float64]()
    var gpu_t = List[Float64]()
    var ce = 0
    var ge = 0
    for r in range(reps):  # interleaved A/B
        var t0 = perf_counter_ns()
        var a = fs.sample_detectors(shots, UInt64(r + 1))
        cpu_t.append(secs(t0))
        var t1 = perf_counter_ns()
        var b = gs.sample_detectors(shots, UInt64(r + 1))
        gpu_t.append(secs(t1))
        ce = a.num_detection_events()
        ge = b.num_detection_events()
    var tc = median(cpu_t^)
    var tg = median(gpu_t^)
    print("surface_z,", d, ",", d, ",", shots, ",", tc, ",", tg, ",", tc / tg, ",",
          Float64(shots) / tc, ",", Float64(shots) / tg, ",", ce, ",", ge, ",", gs.method_name())


def frames_header():
    print("code,d,rounds,shots,cpu_s,gpu_s,speedup,cpu_shots_per_s,gpu_shots_per_s,cpu_events,gpu_events,method")


def main() raises:
    var args = argv()
    if len(args) >= 4 and String(args[1]) == "--stim":
        print(generate_code("surface_z", Int(String(args[2])), Int(String(args[2])),
                            NoiseParams.uniform(Float64(String(args[3])))).to_stim(), end="")
        return
    var nmin = 16
    var nmax = 26
    var reps = 3
    var mode = String("default")
    if len(args) > 1:
        nmax = Int(String(args[1]))
    if len(args) > 2:
        reps = Int(String(args[2]))
    if len(args) > 3:
        mode = String(args[3])
    print("#", gpu_info())
    comptime if is_defined["QUMO_GPU"]():
        if not gpu_available():
            print("# no usable GPU device")
            return
        if mode == "families":
            var lo = Int(String(args[4])) if len(args) > 4 else 20
            var cpu_max = Int(String(args[5])) if len(args) > 5 else 26
            print("family,n,gates,gpu_runs,gpu_transposes,cpu_s,gpu_s,cpu_over_gpu,gpu_gates_per_s,fidelity,method")
            var fams: List[String] = ["layered", "qft", "random"]
            for n in range(lo, nmax + 1, 2):
                for f in fams:
                    bench_family[GPU_DEFAULT_DTYPE](f, n, reps, cpu_max)
            return
        if mode == "crossover":
            var lo = Int(String(args[4])) if len(args) > 4 else 10
            print("n,gates,cpu_sample_s,gpu_sample_s,cpu_over_gpu")
            for n in range(lo, nmax + 1, 2):
                bench_crossover(n, reps)
            return
        if mode == "probe":
            probe_device()
            for n in range(max(nmax - 2, 16), nmax + 1, 2):
                probe_engine[DType.float32](n, reps, n == max(nmax - 2, 16))
                comptime if GPU_FP64_SUPPORTED:
                    probe_engine[DType.float64](n, reps, n == max(nmax - 2, 16))
            return
        if mode == "frames":
            var shots = Int(String(args[4])) if len(args) > 4 else 1000000
            frames_header()
            for d in range(3, nmax + 1, 2):
                bench_frames(d, shots, reps, False)
            return
        # default mode: args[3] = "default" (or absent), args[4] = cpu_max [26]
        var cpu_max = Int(String(args[4])) if len(args) > 4 else 26
        print("# timed: reset + evolve + synchronize on a preallocated GpuStateVector (device allocation and "
              + "DeviceContext / module load excluded, one warm-up evolve per width); gpu_runs = launches of one evolve; "
              + "gpu_alloc_s = state allocation on the same context + evolve + synchronize (freeing excluded)")
        print("qubits,gates,cpu_s,gpu_s,speedup,gpu_gates_per_s,gpu_expect_s,gpu_sample10k_s,abs_err_expect,fidelity,method,gpu_runs,gpu_times,gpu_alloc_s")
        for n in range(nmin, nmax + 1, 2):
            bench_width[GPU_DEFAULT_DTYPE](n, reps, False, cpu_max)
        frames_header()
        for d in range(3, 8, 2):
            bench_frames(d, 1000000, reps, False)
    else:
        print("# GPU support not compiled: run `make bench-gpu` (builds with -D QUMO_GPU).")
        print("# emulator smoke check (timings are NOT GPU timings):")
        print("qubits,gates,cpu_s,emu_s,ratio,emu_gates_per_s,emu_expect_s,emu_sample10k_s,abs_err_expect,fidelity,method,emu_runs,emu_times,emu_alloc_s")
        bench_width[DType.float32](12, 1, True)
        print("code,d,rounds,shots,cpu_s,emu_s,ratio,cpu_shots_per_s,emu_shots_per_s,cpu_events,emu_events,method")
        bench_frames(3, 20000, 1, True)
