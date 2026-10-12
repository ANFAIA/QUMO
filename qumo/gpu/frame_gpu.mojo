"""GPU Pauli-frame sampler for QEC: bit-packed frames, one thread per 64-shot word.

What it is
  * The GPU counterpart of `qumo.sim.frame.FrameSimulator` (Stim's Pauli-frame
    method). It runs the *same* lowered program (`FrameProgram`: frame
    primitives + noiseless tableau reference sample) and returns the *same*
    result types (`DetectorSamples`, `MeasurementSamples`, identical bit
    layout), so decoders and `qumo.qec` consume GPU samples unchanged.
  * One GPU thread owns one 64-bit word = 64 shots of every frame row
    (x/z bits per qubit, measurement-record bits). The whole circuit is ONE
    kernel launch: each thread interprets the op table (uploaded once, read
    in lock-step by every thread, so loads are broadcasts and control flow is
    uniform across a warp). Frame rows live in global memory in
    row-major-by-word layout `row * nw + word`, so consecutive threads touch
    consecutive words (fully coalesced), the layout GPU frame simulators use
    to scale to millions of shots.
  * Noise: geometric skipping over the 64 lanes of the thread's word (the
    CPU sampler's trick) but *integer-only*: for each noise op the host
    precomputes T_k = floor(2^32 (1-p)^k), k = 1..64, and a skip is the
    largest k with u < T_k (binary lifting, 7 compares) for a 32-bit uniform
    u. Pauli choices (DEPOLARIZE1/2, PAULI_CHANNEL_1) use Lemire's
    multiply-shift and 32-bit thresholds. No floating point runs on the
    device, so results are bit-identical on NVIDIA, AMD, Apple (no fp64
    needed) and in the host emulator.
  * Randomness: word g of a run uses `Rng(seed).split(g)` (xoshiro256**),
    so samples depend only on (circuit, shots, seed): not on the launch grid,
    the chunking of large runs, or the device. They are NOT bit-identical to
    the CPU FrameSimulator (which draws one stream per 512-shot batch); the
    two agree in distribution, and exactly whenever the outcome is
    deterministic (noise probabilities 0 or 1), which the tests check.
  * Large runs are chunked so the frame scratch stays under
    FRAME_CHUNK_WORDS words of device memory per launch.

Emulation (`GpuFrameSampler(circuit, emulate=True)`) runs the same thread
body for every word through `qumo.gpu.emulate.emu_launch`, so the kernel
logic is unit-tested on machines without a GPU. Device mode needs
`-D QUMO_GPU` (QM-E470 otherwise).

References: C. Gidney, "Stim: a fast stabilizer circuit simulator" (Quantum
2021) for frames, reference sample and Z-gauge randomisation; D. Lemire,
"Fast random integer generation in an interval" (TOMACS 2019).
"""

from std.math import exp, log1p
from std.time import perf_counter_ns
from std.collections import Optional
from max.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceBuffer

from qumo.ir.circuit import Circuit
from qumo.sim.rng import Rng
from qumo.sim.frame import (
    FrameProgram, FK, FW, DetectorSamples, MeasurementSamples, num_batches_for,
)
from qumo.gpu.kernels import TPB
from qumo.gpu.emulate import emu_launch, grid_blocks
from qumo.gpu.device import GPU_COMPILED, gpu_not_compiled_error

comptime U64Ptr = Pointer[UInt64, MutAnyOrigin]
comptime U32Ptr = Pointer[UInt32, MutAnyOrigin]
comptime I32Ptr = Pointer[Int32, MutAnyOrigin]

comptime OP_STRIDE = 4  # Int32 per op: kind, a, b, c (c = threshold offset for noise)
comptime THR_STRIDE = 66  # UInt32 per noise op: T_1..T_64, then PAULI_CHANNEL_1 split
comptime THR_INF = UInt32(0xFFFFFFFF)  # "always true" threshold (probability 1)
comptime FRAME_CHUNK_WORDS = 1 << 25  # 256 MiB of UInt64 frame scratch per launch


@always_inline
def _ptr[dt: DType](lst: List[Scalar[dt]]) -> Pointer[Scalar[dt], MutAnyOrigin]:
    """Mutable any-origin pointer into a host array (emulated device memory)."""
    return lst.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin()


@always_inline
def _is_noise(kind: Int) -> Bool:
    return kind == FK.XE or kind == FK.YE or kind == FK.ZE or kind == FK.DEP1 or kind == FK.DEP2 or kind == FK.PC1


# ---------------------------------------------------------------------------
# Device-side randomness (integer only)
# ---------------------------------------------------------------------------
@always_inline
def _u32(mut rng: Rng) -> UInt32:
    return UInt32(rng.next_u64() >> 32)


@always_inline
def _below(mut rng: Rng, n: UInt64) -> Int:
    """Uniform integer in [0, n) by Lemire's multiply-shift (n < 2^32)."""
    return Int((UInt64(_u32(rng)) * n) >> 32)


@always_inline
def _lt(u: UInt32, t: UInt32) -> Bool:
    """u < t, where t == THR_INF means probability one."""
    return t == THR_INF or u < t


@always_inline
def skip_lookup(u: UInt32, thr: U32Ptr, base: Int) -> Int:
    """Geometric skip capped at 64: the largest k in [0, 64] with u < T_k
    (T_0 = +inf, T_k = thr[base + k - 1] non-increasing). P(skip >= k) =
    T_k / 2^32 = (1 - p)^k up to 2^-32 rounding."""
    if u < thr[unsafe_offset=base + 63]:
        return 64
    var k = 0
    comptime for j in range(6):
        comptime s = 32 >> j
        if u < thr[unsafe_offset=base + k + s - 1]:
            k += s
    return k


@always_inline
def noise_mask(mut rng: Rng, thr: U32Ptr, base: Int) -> UInt64:
    """64 independent Bernoulli(p) lanes by geometric skipping (O(1 + 64 p) draws)."""
    var m: UInt64 = 0
    var pos = skip_lookup(_u32(rng), thr, base)
    while pos < 64:
        m |= UInt64(1) << UInt64(pos)
        pos += 1 + skip_lookup(_u32(rng), thr, base)
    return m


# ---------------------------------------------------------------------------
# Thread body: one 64-shot word through the whole frame program
# ---------------------------------------------------------------------------
@always_inline
def th_frame_word(gid: Int, nw: Int, word0: Int, seed: UInt64,
                  opi: I32Ptr, nops: Int, thr: U32Ptr, recl: I32Ptr, nq: Int, nk: Int,
                  xs: U64Ptr, zs: U64Ptr, recs: U64Ptr, dets: U64Ptr, obs: U64Ptr):
    """Propagate the 64 frames of word `word0 + gid` (rows at `row * nw + gid`).

    Same semantics as `qumo.sim.frame.FrameBuffers.run` restricted to one
    word: x bits start at 0, z bits random (Z gauge); measurement records
    store the x bit and re-randomise z; detectors/observables XOR records.
    """
    if gid >= nw:
        return
    var rng = Rng(seed).split(word0 + gid)
    for q in range(nq):
        xs[unsafe_offset=q * nw + gid] = 0
        zs[unsafe_offset=q * nw + gid] = rng.next_u64()
    for k in range(nk):
        obs[unsafe_offset=k * nw + gid] = 0
    for i in range(nops):
        var kind = Int(opi[unsafe_offset=OP_STRIDE * i])
        var a = Int(opi[unsafe_offset=OP_STRIDE * i + 1])
        var b = Int(opi[unsafe_offset=OP_STRIDE * i + 2])
        var c = Int(opi[unsafe_offset=OP_STRIDE * i + 3])
        var ia = a * nw + gid
        var ib = b * nw + gid
        if kind == FK.CX:
            xs[unsafe_offset=ib] = xs[unsafe_offset=ib] ^ xs[unsafe_offset=ia]
            zs[unsafe_offset=ia] = zs[unsafe_offset=ia] ^ zs[unsafe_offset=ib]
        elif kind == FK.H:
            var vx = xs[unsafe_offset=ia]
            xs[unsafe_offset=ia] = zs[unsafe_offset=ia]
            zs[unsafe_offset=ia] = vx
        elif kind == FK.DEP2:
            var mxa: UInt64 = 0
            var mza: UInt64 = 0
            var mxb: UInt64 = 0
            var mzb: UInt64 = 0
            var pos = skip_lookup(_u32(rng), thr, c)
            while pos < 64:
                var bit = UInt64(1) << UInt64(pos)
                var r = 1 + _below(rng, 15)
                var pa = r & 3
                var pb = r >> 2
                if pa == 1 or pa == 2: mxa |= bit
                if pa >= 2: mza |= bit
                if pb == 1 or pb == 2: mxb |= bit
                if pb >= 2: mzb |= bit
                pos += 1 + skip_lookup(_u32(rng), thr, c)
            xs[unsafe_offset=ia] = xs[unsafe_offset=ia] ^ mxa
            zs[unsafe_offset=ia] = zs[unsafe_offset=ia] ^ mza
            xs[unsafe_offset=ib] = xs[unsafe_offset=ib] ^ mxb
            zs[unsafe_offset=ib] = zs[unsafe_offset=ib] ^ mzb
        elif kind == FK.DEP1:
            var mx: UInt64 = 0
            var mz: UInt64 = 0
            var pos = skip_lookup(_u32(rng), thr, c)
            while pos < 64:
                var bit = UInt64(1) << UInt64(pos)
                var r = _below(rng, 3)  # 0 = X, 1 = Y, 2 = Z
                if r <= 1: mx |= bit
                if r >= 1: mz |= bit
                pos += 1 + skip_lookup(_u32(rng), thr, c)
            xs[unsafe_offset=ia] = xs[unsafe_offset=ia] ^ mx
            zs[unsafe_offset=ia] = zs[unsafe_offset=ia] ^ mz
        elif kind == FK.M:
            recs[unsafe_offset=b * nw + gid] = xs[unsafe_offset=ia]
            zs[unsafe_offset=ia] = zs[unsafe_offset=ia] ^ rng.next_u64()
        elif kind == FK.R:
            xs[unsafe_offset=ia] = 0
            zs[unsafe_offset=ia] = rng.next_u64()
        elif kind == FK.XE or kind == FK.YE or kind == FK.ZE:
            var m = noise_mask(rng, thr, c)
            if kind != FK.ZE:
                xs[unsafe_offset=ia] = xs[unsafe_offset=ia] ^ m
            if kind != FK.XE:
                zs[unsafe_offset=ia] = zs[unsafe_offset=ia] ^ m
        elif kind == FK.PC1:
            var mx: UInt64 = 0
            var mz: UInt64 = 0
            var tx = thr[unsafe_offset=c + 64]
            var txy = thr[unsafe_offset=c + 65]
            var pos = skip_lookup(_u32(rng), thr, c)
            while pos < 64:
                var bit = UInt64(1) << UInt64(pos)
                var u = _u32(rng)
                if _lt(u, tx):
                    mx |= bit
                elif _lt(u, txy):
                    mx |= bit
                    mz |= bit
                else:
                    mz |= bit
                pos += 1 + skip_lookup(_u32(rng), thr, c)
            xs[unsafe_offset=ia] = xs[unsafe_offset=ia] ^ mx
            zs[unsafe_offset=ia] = zs[unsafe_offset=ia] ^ mz
        elif kind == FK.DET:
            var acc: UInt64 = 0
            for j in range(b):
                acc ^= recs[unsafe_offset=Int(recl[unsafe_offset=a + j]) * nw + gid]
            dets[unsafe_offset=c * nw + gid] = acc
        elif kind == FK.OBS:
            var acc = obs[unsafe_offset=c * nw + gid]
            for j in range(b):
                acc ^= recs[unsafe_offset=Int(recl[unsafe_offset=a + j]) * nw + gid]
            obs[unsafe_offset=c * nw + gid] = acc
        elif kind == FK.S:
            zs[unsafe_offset=ia] = zs[unsafe_offset=ia] ^ xs[unsafe_offset=ia]
        elif kind == FK.SX:
            xs[unsafe_offset=ia] = xs[unsafe_offset=ia] ^ zs[unsafe_offset=ia]
        elif kind == FK.CZ:
            var xa = xs[unsafe_offset=ia]
            var xb = xs[unsafe_offset=ib]
            zs[unsafe_offset=ia] = zs[unsafe_offset=ia] ^ xb
            zs[unsafe_offset=ib] = zs[unsafe_offset=ib] ^ xa
        elif kind == FK.SWAP:
            var xa = xs[unsafe_offset=ia]
            var za = zs[unsafe_offset=ia]
            xs[unsafe_offset=ia] = xs[unsafe_offset=ib]
            zs[unsafe_offset=ia] = zs[unsafe_offset=ib]
            xs[unsafe_offset=ib] = xa
            zs[unsafe_offset=ib] = za
        elif kind == FK.FEED:
            # a = controlling record, b = qubit, c = Pauli (1 = X, 2 = Y, 3 = Z)
            var flips = recs[unsafe_offset=a * nw + gid]
            if c == 1 or c == 2:
                xs[unsafe_offset=ib] = xs[unsafe_offset=ib] ^ flips
            if c >= 2:
                zs[unsafe_offset=ib] = zs[unsafe_offset=ib] ^ flips


def gk_frame_words(opi: I32Ptr, nops: Int64, thr: U32Ptr, recl: I32Ptr, nq: Int64, nk: Int64,
                   xs: U64Ptr, zs: U64Ptr, recs: U64Ptr, dets: U64Ptr, obs: U64Ptr,
                   nw: Int64, word0: Int64, seed: UInt64):
    """GPU kernel: one thread per 64-shot word (instantiated only with -D QUMO_GPU)."""
    th_frame_word(global_idx.x, Int(nw), Int(word0), seed, opi, Int(nops), thr, recl,
                  Int(nq), Int(nk), xs, zs, recs, dets, obs)


# ---------------------------------------------------------------------------
# Host side: op table, thresholds, launches and result layout
# ---------------------------------------------------------------------------
def skip_thresholds(p: Float64, mut thr: List[UInt32]):
    """Append T_1..T_64 with T_k = floor(2^32 (1 - p)^k) (all 0 when p >= 1)."""
    for k in range(1, 65):
        var v = 0.0
        if p < 1.0:
            v = exp(Float64(k) * log1p(-p)) * 4294967296.0
        if v >= 4294967295.0:
            thr.append(THR_INF)
        elif v <= 0.0:
            thr.append(0)
        else:
            thr.append(UInt32(Int(v)))


def _threshold(frac: Float64) -> UInt32:
    """32-bit threshold for probability `frac` (THR_INF for 1)."""
    if frac >= 1.0:
        return THR_INF
    if frac <= 0.0:
        return 0
    return UInt32(min(Int(frac * 4294967296.0), 4294967294))


struct GpuFrameSampler(Movable):
    """Pauli-frame sampling of a noisy Clifford circuit on a GPU (or its emulator).

        var gs = GpuFrameSampler(circuit)              # needs -D QUMO_GPU
        var ds = gs.sample_detectors(1_000_000, seed=7) # qumo.sim.frame.DetectorSamples
        var es = GpuFrameSampler(circuit, emulate=True) # same kernel body on the CPU
    """

    var prog: FrameProgram
    var emulated: Bool
    var max_words: Int  # words (64 shots each) per kernel launch
    var opi: List[Int32]
    var thr: List[UInt32]
    var recl: List[Int32]
    var nops: Int
    var ctx: Optional[DeviceContext]
    var d_opi: Optional[DeviceBuffer[DType.int32]]
    var d_thr: Optional[DeviceBuffer[DType.uint32]]
    var d_recl: Optional[DeviceBuffer[DType.int32]]

    def __init__(out self, circuit: Circuit, emulate: Bool = False, max_words: Int = 0) raises:
        """Lower `circuit` (QM-E701 if not Clifford) and upload the op table.

        max_words > 0 caps the words per launch (testing); 0 = memory budget.
        """
        self.prog = FrameProgram(circuit)
        self.emulated = emulate
        self.opi = List[Int32]()
        self.thr = List[UInt32]()
        self.recl = List[Int32]()
        self.ctx = None
        self.d_opi = None
        self.d_thr = None
        self.d_recl = None
        self.nops = len(self.prog.ops)
        self.max_words = 1
        for fo in self.prog.ops:
            var c = fo.c
            if _is_noise(fo.kind):
                c = len(self.thr)
                skip_thresholds(fo.p, self.thr)
                if fo.kind == FK.PC1:
                    self.thr.append(_threshold(fo.px / fo.p))
                    self.thr.append(_threshold((fo.px + fo.py) / fo.p))
                else:
                    self.thr.append(0)
                    self.thr.append(0)
            self.opi.append(Int32(fo.kind))
            self.opi.append(Int32(fo.a))
            self.opi.append(Int32(fo.b))
            self.opi.append(Int32(c))
        for r in self.prog.records:
            self.recl.append(Int32(r))
        # never-empty tables (device buffers of size 0 are not portable)
        if len(self.thr) == 0:
            self.thr.append(0)
        if len(self.recl) == 0:
            self.recl.append(0)
        if len(self.opi) == 0:
            self.opi.append(0)
        var rows = self._rows()
        self.max_words = max_words if max_words > 0 else max(1, FRAME_CHUNK_WORDS // rows)
        if not emulate:
            comptime if GPU_COMPILED:
                var ctx = DeviceContext()
                var dopi = ctx.enqueue_create_buffer[DType.int32](len(self.opi))
                var dthr = ctx.enqueue_create_buffer[DType.uint32](len(self.thr))
                var drecl = ctx.enqueue_create_buffer[DType.int32](len(self.recl))
                with dopi.map_to_host() as h:
                    for i in range(len(self.opi)):
                        h[i] = self.opi[i]
                with dthr.map_to_host() as h:
                    for i in range(len(self.thr)):
                        h[i] = self.thr[i]
                with drecl.map_to_host() as h:
                    for i in range(len(self.recl)):
                        h[i] = self.recl[i]
                self.d_opi = dopi^
                self.d_thr = dthr^
                self.d_recl = drecl^
                self.ctx = ctx^
            else:
                raise gpu_not_compiled_error()

    def _rows(self) -> Int:
        """UInt64 rows of frame scratch per word: x, z per qubit, records, detectors, observables."""
        return 2 * max(1, self.prog.num_qubits) + max(1, self.prog.num_measurements) + max(1, self.prog.num_detectors) + max(1, self.prog.num_observables)

    def method_name(self) -> String:
        return "gpu-emulator:pauli_frame" if self.emulated else "gpu:pauli_frame"

    def num_kernel_launches(self, shots: Int) -> Int:
        """Launches a run of `shots` takes (one per chunk of max_words words)."""
        var words = num_batches_for(shots) * FW
        return (words + self.max_words - 1) // self.max_words

    def _run(self, shots: Int, seed: UInt64, want_meas: Bool,
             mut ds: DetectorSamples, mut ms: MeasurementSamples) raises:
        var total = num_batches_for(shots) * FW
        if total == 0:
            return
        var Q = max(1, self.prog.num_qubits)
        var M = max(1, self.prog.num_measurements)
        var D = self.prog.num_detectors
        var K = self.prog.num_observables
        var Dp = max(1, D)
        var Kp = max(1, K)
        var cw = min(total, self.max_words)
        var nq = self.prog.num_qubits
        var nops = self.nops
        if self.emulated:
            var xs = List[UInt64](length=Q * cw, fill=0)
            var zs = List[UInt64](length=Q * cw, fill=0)
            var recs = List[UInt64](length=M * cw, fill=0)
            var dets = List[UInt64](length=Dp * cw, fill=0)
            var obs = List[UInt64](length=Kp * cw, fill=0)
            var pxs = _ptr[DType.uint64](xs)
            var pzs = _ptr[DType.uint64](zs)
            var precs = _ptr[DType.uint64](recs)
            var pdets = _ptr[DType.uint64](dets)
            var pobs = _ptr[DType.uint64](obs)
            var popi = _ptr[DType.int32](self.opi)
            var pthr = _ptr[DType.uint32](self.thr)
            var precl = _ptr[DType.int32](self.recl)
            var word0 = 0
            while word0 < total:
                var nw = min(cw, total - word0)

                def f(g: Int) {imm nw, imm word0, imm seed, imm popi, imm nops, imm pthr, imm precl, imm nq, imm K,
                               imm pxs, imm pzs, imm precs, imm pdets, imm pobs}:
                    th_frame_word(g, nw, word0, seed, popi, nops, pthr, precl, nq, K, pxs, pzs, precs, pdets, pobs)

                emu_launch(f, nw)
                if want_meas:
                    self._scatter_meas(recs, nw, word0, ms)
                else:
                    self._scatter_dets(dets, obs, nw, word0, ds)
                word0 += nw
            # keep the emulated device memory alive past the last launch (ASAP destruction)
            _ = len(xs)
            _ = len(zs)
            _ = len(recs)
            _ = len(dets)
            _ = len(obs)
            return
        comptime if GPU_COMPILED:
            ref ctx = self.ctx.value()
            var dxs = ctx.enqueue_create_buffer[DType.uint64](Q * cw)
            var dzs = ctx.enqueue_create_buffer[DType.uint64](Q * cw)
            var drecs = ctx.enqueue_create_buffer[DType.uint64](M * cw)
            var ddets = ctx.enqueue_create_buffer[DType.uint64](Dp * cw)
            var dobs = ctx.enqueue_create_buffer[DType.uint64](Kp * cw)
            var word0 = 0
            while word0 < total:
                var nw = min(cw, total - word0)
                ctx.enqueue_function[gk_frame_words](
                    self.d_opi.value(), Int64(nops), self.d_thr.value(), self.d_recl.value(),
                    Int64(nq), Int64(K), dxs, dzs, drecs, ddets, dobs,
                    Int64(nw), Int64(word0), seed,
                    grid_dim=grid_blocks(nw), block_dim=TPB)
                if want_meas:
                    var hr = ctx.enqueue_create_host_buffer[DType.uint64](M * cw)
                    drecs.enqueue_copy_to(hr)
                    ctx.synchronize()
                    var tmp = List[UInt64](length=M * nw, fill=0)
                    for i in range(M * nw):
                        tmp[i] = hr[i]
                    self._scatter_meas(tmp, nw, word0, ms)
                else:
                    var hd = ctx.enqueue_create_host_buffer[DType.uint64](Dp * cw)
                    var ho = ctx.enqueue_create_host_buffer[DType.uint64](Kp * cw)
                    ddets.enqueue_copy_to(hd)
                    dobs.enqueue_copy_to(ho)
                    ctx.synchronize()
                    var td = List[UInt64](length=Dp * nw, fill=0)
                    var to = List[UInt64](length=Kp * nw, fill=0)
                    for i in range(Dp * nw):
                        td[i] = hd[i]
                    for i in range(Kp * nw):
                        to[i] = ho[i]
                    self._scatter_dets(td, to, nw, word0, ds)
                word0 += nw
        else:
            raise gpu_not_compiled_error()

    def _scatter_dets(self, dets: List[UInt64], obs: List[UInt64], nw: Int, word0: Int,
                      mut ds: DetectorSamples):
        """Chunk rows (row * nw + j) -> DetectorSamples layout ((batch * D + d) * FW + w)."""
        var D = self.prog.num_detectors
        var K = self.prog.num_observables
        for j in range(nw):
            var g = word0 + j
            var bt = g // FW
            var w = g % FW
            for d in range(D):
                ds.dets[(bt * D + d) * FW + w] = dets[d * nw + j]
            for k in range(K):
                ds.obs[(bt * K + k) * FW + w] = obs[k * nw + j]

    def _scatter_meas(self, recs: List[UInt64], nw: Int, word0: Int, mut ms: MeasurementSamples):
        """Record flips XOR the reference sample -> MeasurementSamples layout."""
        var M = self.prog.num_measurements
        for j in range(nw):
            var g = word0 + j
            var bt = g // FW
            var w = g % FW
            for m in range(M):
                var refw: UInt64 = ~UInt64(0) if self.prog.reference[m] != 0 else 0
                ms.bits[(bt * M + m) * FW + w] = recs[m * nw + j] ^ refw

    def sample_detectors(self, shots: Int, seed: UInt64 = 1234) raises -> DetectorSamples:
        """Detection events + observable flips (same layout as FrameSimulator)."""
        var out = DetectorSamples(shots, self.prog.num_detectors, self.prog.num_observables)
        var unused = MeasurementSamples(0, 0)
        var t0 = perf_counter_ns()
        self._run(shots, seed, False, out, unused)
        out.seconds = Float64(perf_counter_ns() - t0) / 1e9
        return out^

    def sample_measurements(self, shots: Int, seed: UInt64 = 1234) raises -> MeasurementSamples:
        """Raw measurement records (reference XOR frame flips) per shot."""
        var out = MeasurementSamples(shots, self.prog.num_measurements)
        var unused = DetectorSamples(0, 0, 0)
        var t0 = perf_counter_ns()
        self._run(shots, seed, True, unused, out)
        out.seconds = Float64(perf_counter_ns() - t0) / 1e9
        return out^

    def throughput(self, samples: DetectorSamples) -> Float64:
        """Circuit instructions x shots per second for a finished run."""
        return Float64(self.prog.num_circuit_ops) * Float64(samples.shots) / max(samples.seconds, 1e-12)
