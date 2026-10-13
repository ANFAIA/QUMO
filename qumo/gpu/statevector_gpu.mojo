"""GpuStateVector: the QUMO state vector on a GPU (MAX, vendor-neutral).

What it is
  * Owns the amplitudes as two device buffers (split re/im, like the CPU
    `StateVector`) plus the `DeviceContext` (one in-order stream). Precision
    is the comptime parameter `dtype` (float32 / float64; see
    `qumo.gpu.device` for the policy: fp32 on Apple, fp64 default elsewhere).
  * Gates run as register-blocked *tile runs* (`qumo.gpu.tiles`,
    `kernels.gk_tile`): the CPU runner's fused plan (`build_plan`: runs of 1q
    gates merged per wire, as in qsim / Aer) is packed into runs of gates
    whose targets fit in one 2^k-amplitude tile (controls and diagonal gates
    may act anywhere); each run is ONE launch that reads and writes the state
    once. One upload of the op / matrix / segment tables per flush; launches
    are asynchronous on one stream, with no host synchronisation between
    kernels, only when results are read back (to_host, probabilities,
    expectation, sample). Single-gate calls (`apply_1q`, `apply_cx`, ...)
    are one run each: prefer `evolve` / `execute_plan` / `run_ops`.
  * |0...0> (and any `set_basis`) is lazy: the first gate run writes the
    basis state in its load phase, so `GpuStateVector(n)` + `evolve` costs no
    separate initialisation pass. A read-out of a still-lazy state launches
    the (grid-stride) init kernel first.
  * Expectation values: terms are grouped by X mask; one launch reduces up to
    EXP_G terms with the same X mask in a single pass over the state
    (shared-memory trees, <= 1024 partials per term), ONE device->host copy
    for all terms, fp64 summation of the partials on the host
    (deterministic, no atomics).
  * Sampling never copies the state: a block per segment reduces segment
    probabilities on the GPU, the host builds the fp64 prefix over segments
    and maps the sorted uniforms of the QUMO RNG onto (segment, residual);
    then one block per *occupied* segment builds the segment's CDF in shared
    memory (coalesced load + block scan) and binary-searches it for each of
    its shots. The RNG stream use is the same as `StateVector.sample`
    (sorted uniforms, then the same Fisher-Yates shuffle), so a seed
    reproduces the same shots run to run.

Emulation mode (`GpuStateVector.emulator(n)`, or `emulate=True`): the same
object keeps host arrays and runs every kernel's thread phases through
`qumo.gpu.emulate` over the same launch grid. All host logic here (tile
scheduling, table layout, launch geometry, reductions, segment sampling) is
therefore exercised by tests on machines without a GPU. Device mode compiles
only with `-D QUMO_GPU`; otherwise it raises QM-E470.

Error codes: QM-E470 not compiled, QM-E471 dynamic/noisy circuit on the GPU
sampler, QM-E472 fp64 on Apple, QM-E473 zero-norm state, QM-E474 circuit wider
than the GPU state, QM-E475 unsupported plan op.
"""

from std.math import cos, sin, ceildiv
from std.collections import Optional, Dict
from std.collections.optional import OptionalReg
from std.bit import pop_count
from std.sys.info import size_of
from max.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceBuffer, Attribute, FuncAttribute

from qumo.ir.gates import Op, M2, M4
from qumo.ir.circuit import Circuit, Instruction
from qumo.ir.pauli import PauliSum
from qumo.sim.rng import Rng
from qumo.sim.statevector import StateVector
from qumo.sim.runner import build_plan, KOp, K, SampleResult, check_statevector_size, terminal_measure_map
from qumo.gpu.kernels import (
    TPB, RED_ITEMS, MAX_RED_BLOCKS, EXP_G, GPtr, I64Ptr, I32Ptr, tile_bits, tile_smem_bytes, tile_smem_typical,
    sample_seg, TILE_SMEM_DYNAMIC, TILE_SAFE_THREADS, TILE_LAUNCH_BOUND, REG_BITS,
    th_prob, gk_tile, gk_tile_lb, gk_prob, gk_expect, gk_segsum, gk_sample,
)
from qumo.gpu.tiles import TileProgram, build_tile_program
from qumo.gpu.emulate import (
    emu_threads, emu_tile, emu_expect, emu_segsum, emu_sample, stride_blocks,
)
from qumo.gpu.device import (
    GPU_COMPILED, GPU_DEFAULT_DTYPE, GpuProfile, TileKernelLimits, device_kernels_enabled, precision_name,
    gpu_not_compiled_error, gpu_fp64_error,
)


@always_inline
def _mptr[T: DType](lst: List[Scalar[T]]) -> Pointer[Scalar[T], MutAnyOrigin]:
    """Mutable any-origin pointer into a host array (emulated device memory)."""
    return lst.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin()


def reduction_blocks(n: Int) -> Int:
    """Blocks of a reduction over n items: ~RED_ITEMS items per thread, capped."""
    return min(MAX_RED_BLOCKS, max(1, ceildiv(n, TPB * RED_ITEMS)))


def tile_func_attribute(p: GpuProfile) -> OptionalReg[FuncAttribute]:
    """The function attribute every load of the tile kernel carries on this
    device: MAX_DYNAMIC_SHARED_SIZE_BYTES = GpuProfile.tile_smem_optin() when
    the device has a shared-memory opt-in above 48 KiB (NVIDIA Volta+), else
    none. One constant per device (not each launch's size), so whether a
    cached function is re-attributed between loads never matters."""
    var b = p.tile_smem_optin()
    if b > 0:
        return FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(UInt32(b))
    return None


@always_inline
def enqueue_tile[dt: DType, FULL: Bool, HEAVY: Bool](
    ctx: DeviceContext, re: DeviceBuffer[dt], im: DeviceBuffer[dt], ops: DeviceBuffer[DType.int32],
    mats: DeviceBuffer[dt], hp: DeviceBuffer[DType.int32], segs: DeviceBuffer[DType.int32],
    s0: Int, s1: Int, hoff: Int, nh: Int, low: Int, k: Int, nr: Int, fin: Int, init: Int,
    o0: Int, nops: Int, m0: Int, nm: Int, nblocks: Int, nthr: Int, smem: Int, fa: OptionalReg[FuncAttribute],
) raises:
    """One tile-run launch of the kernel this target uses (gk_tile, or
    gk_tile_lb when kernels.TILE_LAUNCH_BOUND > 0)."""
    comptime if TILE_LAUNCH_BOUND > 0:
        ctx.enqueue_function[gk_tile_lb[dt, FULL, HEAVY]](
            re, im, ops, mats, hp, segs, Int32(s0), Int32(s1), Int32(hoff), Int32(nh), Int32(low), Int32(k),
            Int32(nr), Int32(fin), Int64(init), Int32(o0), Int32(nops), Int32(m0), Int32(nm),
            grid_dim=nblocks, block_dim=nthr, shared_mem_bytes=smem, func_attribute=fa)
    else:
        ctx.enqueue_function[gk_tile[dt, FULL, HEAVY]](
            re, im, ops, mats, hp, segs, Int32(s0), Int32(s1), Int32(hoff), Int32(nh), Int32(low), Int32(k),
            Int32(nr), Int32(fin), Int64(init), Int32(o0), Int32(nops), Int32(m0), Int32(nm),
            grid_dim=nblocks, block_dim=nthr, shared_mem_bytes=smem, func_attribute=fa)


def tile_kernel_attr[dt: DType, FULL: Bool, HEAVY: Bool](ctx: DeviceContext, attr: Attribute,
                                                         fa: OptionalReg[FuncAttribute]) raises -> Int:
    """A driver attribute (registers, spills, max threads per block) of the
    tile kernel as _launch loads it on this device (raises on Metal)."""
    comptime if TILE_LAUNCH_BOUND > 0:
        return ctx.compile_function[gk_tile_lb[dt, FULL, HEAVY]](func_attribute=fa).get_attribute(attr)
    else:
        return ctx.compile_function[gk_tile[dt, FULL, HEAVY]](func_attribute=fa).get_attribute(attr)


def tile_kernel_occupancy[dt: DType, FULL: Bool, HEAVY: Bool](ctx: DeviceContext, threads: Int, smem: Int,
                                                              fa: OptionalReg[FuncAttribute]) raises -> Int:
    """Resident blocks per multiprocessor of the tile kernel at a block size
    and dynamic shared memory (occupancy calculator; raises on Metal)."""
    comptime if TILE_LAUNCH_BOUND > 0:
        return ctx.compile_function[gk_tile_lb[dt, FULL, HEAVY]](
            func_attribute=fa).occupancy_max_active_blocks_per_multiprocessor(threads, smem)
    else:
        return ctx.compile_function[gk_tile[dt, FULL, HEAVY]](
            func_attribute=fa).occupancy_max_active_blocks_per_multiprocessor(threads, smem)


def _limits1[dt: DType, FULL: Bool, HEAVY: Bool](ctx: DeviceContext, fa: OptionalReg[FuncAttribute],
                                                 mut kl: TileKernelLimits) raises:
    var mt = tile_kernel_attr[dt, FULL, HEAVY](ctx, Attribute.MAX_THREADS_PER_BLOCK, fa)
    kl.max_threads = mt if kl.max_threads < 0 else min(kl.max_threads, mt)
    try:
        kl.regs = max(kl.regs, tile_kernel_attr[dt, FULL, HEAVY](ctx, Attribute.NUM_REGS, fa))
        kl.local_bytes = max(kl.local_bytes, tile_kernel_attr[dt, FULL, HEAVY](ctx, Attribute.LOCAL_SIZE_BYTES, fa))
    except:
        pass


def tile_kernel_limits[dt: DType](ctx: DeviceContext, p: GpuProfile) -> TileKernelLimits:
    """Limits of the 4 compiled tile-kernel specialisations of precision dt on
    the device behind ctx (TileKernelLimits; never raises: unknown on Metal
    or on any driver error, so the profile keeps blocks of at most
    TILE_SAFE_THREADS threads). Loads the kernels the first launch would
    load anyway, with the same function attribute."""
    var kl = TileKernelLimits.unknown()
    comptime if device_kernels_enabled[dt]():
        var fa = tile_func_attribute(p)
        try:
            _limits1[dt, True, False](ctx, fa, kl)
            _limits1[dt, True, True](ctx, fa, kl)
            _limits1[dt, False, False](ctx, fa, kl)
            _limits1[dt, False, True](ctx, fa, kl)
        except:
            return TileKernelLimits.unknown()
        # largest larger-than-safe block that keeps >= 2 typical runs resident
        try:
            var es = size_of[dt]()
            var tb0 = tile_bits[dt]()
            kl.two_block_threads = TILE_SAFE_THREADS
            for k in [tb0 + 2, tb0 + 1]:
                var threads = 1 << (k - REG_BITS)
                if threads <= TILE_SAFE_THREADS or threads > kl.max_threads:
                    continue
                var sm = tile_smem_typical(es, k)
                var occ = min(tile_kernel_occupancy[dt, True, False](ctx, threads, sm, fa),
                              tile_kernel_occupancy[dt, True, True](ctx, threads, sm, fa))
                if occ >= 2:
                    kl.two_block_threads = threads
                    break
        except:
            kl.two_block_threads = -1
    return kl


def gk_init[dt: DType](re: GPtr[dt], im: GPtr[dt], index: Int64, work: Int64, nthreads: Int64):
    """|index> (grid-stride): materialises a lazy basis state before a read-out."""
    var i = global_idx.x
    while i < Int(work):
        re[unsafe_offset=i] = Scalar[dt](1) if i == Int(index) else Scalar[dt](0)
        im[unsafe_offset=i] = 0
        i += Int(nthreads)


struct GpuStateVector[dtype: DType = GPU_DEFAULT_DTYPE](Movable):
    var num_qubits: Int
    var emulated: Bool
    var ctx: Optional[DeviceContext]
    var dre: Optional[DeviceBuffer[Self.dtype]]
    var dim: Optional[DeviceBuffer[Self.dtype]]
    var hre: List[Scalar[Self.dtype]]  # emulated device memory (empty on a GPU)
    var him: List[Scalar[Self.dtype]]
    var launches: Int  # gate-run launches issued so far (diagnostics)
    var runs_last: Int  # tile runs of the last flush
    var ops_last: Int  # gates (tile ops) of the last flush
    var transposes_last: Int  # shared-memory register-set changes of the last flush
    var init_index: Int  # >= 0: state is |init_index>, not yet written (lazy)
    # tables of the last flush: kept alive until the stream is synchronised
    var t_ops: List[Int32]
    var t_mats: List[Scalar[Self.dtype]]
    var t_hp: List[Int32]
    var t_segs: List[Int32]
    var d_ops: Optional[DeviceBuffer[DType.int32]]
    var d_mats: Optional[DeviceBuffer[Self.dtype]]
    var d_hp: Optional[DeviceBuffer[DType.int32]]
    var d_segs: Optional[DeviceBuffer[DType.int32]]
    var in_flight: Bool
    var profile: GpuProfile  # per-device tuning (tile geometry), see qumo.gpu.device

    def __init__(out self, num_qubits: Int, emulate: Bool = False,
                 ctx_in: Optional[DeviceContext] = None) raises:
        """|0...0> on the GPU (emulate=False) or in the host emulator. A GPU
        state creates its own DeviceContext unless `ctx_in` gives one."""
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64, "GpuStateVector supports float32 and float64"
        check_statevector_size(num_qubits)
        self.num_qubits = num_qubits
        self.emulated = emulate
        self.ctx = None
        self.dre = None
        self.dim = None
        self.hre = List[Scalar[Self.dtype]]()
        self.him = List[Scalar[Self.dtype]]()
        self.launches = 0
        self.runs_last = 0
        self.ops_last = 0
        self.transposes_last = 0
        self.init_index = 0
        self.t_ops = List[Int32]()
        self.t_mats = List[Scalar[Self.dtype]]()
        self.t_hp = List[Int32]()
        self.t_segs = List[Int32]()
        self.d_ops = None
        self.d_mats = None
        self.d_hp = None
        self.d_segs = None
        self.in_flight = False
        self.profile = GpuProfile.emulator()
        var n = 1 << num_qubits
        if emulate:
            self.hre = List[Scalar[Self.dtype]](length=n, fill=0)
            self.him = List[Scalar[Self.dtype]](length=n, fill=0)
        else:
            comptime if device_kernels_enabled[Self.dtype]():
                var ctx = ctx_in.value() if ctx_in else DeviceContext()
                var prof = GpuProfile.of(ctx)
                # blocks above TILE_SAFE_THREADS threads only once the compiled
                # kernel reports that they launch (registers per thread)
                if prof.wants_kernel_limits(size_of[Self.dtype]()):
                    var kl = tile_kernel_limits[Self.dtype](ctx, prof)
                    comptime if Self.dtype == DType.float64:
                        prof = GpuProfile.of(ctx, TileKernelLimits.unknown(), kl)
                    else:
                        prof = GpuProfile.of(ctx, kl, TileKernelLimits.unknown())
                self.profile = prof^
                self.dre = ctx.enqueue_create_buffer[Self.dtype](n)
                self.dim = ctx.enqueue_create_buffer[Self.dtype](n)
                self.ctx = ctx^
            else:
                raise Self._no_device()

    @staticmethod
    def on_context(ctx: DeviceContext, num_qubits: Int) raises -> Self:
        """|0...0> on the device of an existing DeviceContext: only the two
        state buffers are allocated (no context creation, no kernel load)."""
        return Self(num_qubits, False, Optional[DeviceContext](ctx))

    @staticmethod
    def emulator(num_qubits: Int) raises -> Self:
        """Host-emulated GPU state (runs the kernel bodies on the CPU)."""
        return Self(num_qubits, emulate=True)

    @staticmethod
    def _no_device() -> Error:
        comptime if GPU_COMPILED:
            return gpu_fp64_error()
        else:
            return gpu_not_compiled_error()

    def size(self) -> Int:
        return 1 << self.num_qubits

    def precision(self) -> String:
        return precision_name[Self.dtype]()

    def method_name(self) -> String:
        """'gpu:statevector:fp32' (or 'gpu-emulator:...'), for SampleResult.method."""
        var p = "gpu-emulator" if self.emulated else "gpu"
        return p + ":statevector:" + self.precision()

    def device_name(self) -> String:
        if self.emulated:
            return "host-emulator"
        return self.ctx.value().name()

    def synchronize(mut self) raises:
        """Wait for all enqueued kernels (no-op when emulated)."""
        if not self.emulated:
            self.ctx.value().synchronize()
        self.in_flight = False

    @always_inline
    def _re(self) -> GPtr[Self.dtype]:
        return _mptr[Self.dtype](self.hre)

    @always_inline
    def _im(self) -> GPtr[Self.dtype]:
        return _mptr[Self.dtype](self.him)

    # ------------------------------------------------------------ init
    def reset(mut self) raises:
        """Back to |0...0> (lazy: written by the next gate run or read-out)."""
        self.set_basis(0)

    def set_basis(mut self, index: Int) raises:
        """Computational basis state |index> (lazy, see reset)."""
        if index < 0 or index >= self.size():
            raise Error("QM-E474: basis index " + String(index) + " out of range for " + String(self.num_qubits) + " qubits. Hint: use 0 <= index < 2^n.")
        self.init_index = index

    def _materialize(self) raises:
        """Write a lazy basis state before a read-out (grid-stride init kernel).

        Takes `self` immutably so read-outs work on temporaries; the lazy flag
        stays set, so a later read-out re-writes the same basis state
        (idempotent) and the next gate run still folds it into its load."""
        var index = self.init_index
        if index < 0:
            return
        var work = self.size()
        if self.emulated:
            var re = self._re()
            var im = self._im()

            def f(g: Int) {imm re, imm im, imm work, imm index}:
                if g < work:
                    re[unsafe_offset=g] = Scalar[Self.dtype](1) if g == index else Scalar[Self.dtype](0)
                    im[unsafe_offset=g] = 0

            emu_threads(f, work)
            return
        comptime if device_kernels_enabled[Self.dtype]():
            var nb = stride_blocks(work)
            self.ctx.value().enqueue_function[gk_init[Self.dtype]](
                self.dre.value(), self.dim.value(), Int64(index), Int64(work), Int64(nb * TPB),
                grid_dim=nb, block_dim=TPB)
        else:
            raise Self._no_device()

    def load_host(mut self, sv: StateVector) raises:
        """Upload a CPU state (rounded to `dtype`); sizes must match."""
        if sv.num_qubits != self.num_qubits:
            raise Error("QM-E474: host state has " + String(sv.num_qubits) + " qubits but the GPU state has " + String(self.num_qubits) + ". Hint: allocate GpuStateVector(sv.num_qubits).")
        var n = self.size()
        self.init_index = -1
        if self.emulated:
            for i in range(n):
                self.hre[i] = Scalar[Self.dtype](sv.re[i])
                self.him[i] = Scalar[Self.dtype](sv.im[i])
            return
        comptime if device_kernels_enabled[Self.dtype]():
            self.synchronize()
            with self.dre.value().map_to_host() as h:
                for i in range(n):
                    h[i] = Scalar[Self.dtype](sv.re[i])
            with self.dim.value().map_to_host() as h:
                for i in range(n):
                    h[i] = Scalar[Self.dtype](sv.im[i])
        else:
            raise Self._no_device()

    # ------------------------------------------------------------ gate runs
    def run_ops(mut self, ops: List[KOp], phase: Float64 = 0.0) raises:
        """Schedule `ops` (+ global phase) into tile runs and enqueue them.

        One table upload, one `gk_tile` launch per run, no host sync (except
        waiting for the previous flush's tables when it may still be in use).
        """
        if len(ops) == 0 and phase == 0.0:
            return
        var prog = build_tile_program[Self.dtype](ops, self.num_qubits, phase, self.profile.tile_bits[Self.dtype](),
                                                  self.profile.low[Self.dtype](), self.profile.diag_width,
                                                  self.profile.split_1q[Self.dtype]())
        self._launch(prog^)

    def run_program(mut self, var prog: TileProgram[Self.dtype]) raises:
        """Enqueue a finalized TileProgram as is (diagnostics and probes:
        `bench_gpu ... probe` builds synthetic runs with it)."""
        self._launch(prog^)

    def _launch(mut self, var prog: TileProgram[Self.dtype]) raises:
        var nruns = len(prog.runs)
        self.runs_last = nruns
        self.ops_last = prog.num_ops()
        self.transposes_last = prog.num_transposes()
        if nruns == 0:
            return
        prog.pad_tables()
        var nblocks = prog.num_blocks()
        var k = prog.k
        var low = prog.low
        var nr = prog.nr
        var nthr = prog.threads_per_block()
        if self.emulated:
            var pops = prog.dops.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin()
            var pmats = _mptr[Self.dtype](prog.mats)
            var php = prog.hp.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin()
            var psegs = prog.segs.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin()
            for r in range(nruns):
                var run = prog.runs[r]
                var init = self.init_index if r == 0 else -1
                emu_tile[Self.dtype](self._re(), self._im(), pops, pmats, php, psegs, run.s0, run.s1,
                                     run.hoff, run.nh, low, k, nr, run.fin, init,
                                     run.o0, run.o1 - run.o0, run.m0, run.m1 - run.m0, nblocks)
            self.launches += nruns
            self.init_index = -1
            _ = len(prog.dops)
            _ = len(prog.mats)
            _ = len(prog.hp)
            _ = len(prog.segs)
            return
        comptime if device_kernels_enabled[Self.dtype]():
            if self.in_flight:
                self.ctx.value().synchronize()  # old tables may still be read
            var heavies = List[Bool](capacity=nruns)
            for r in range(nruns):
                heavies.append(prog.heavy(r))
            # the host tables live in self until the next synchronisation:
            # enqueue_copy from a host pointer is asynchronous
            swap(self.t_ops, prog.dops)
            swap(self.t_mats, prog.mats)
            swap(self.t_hp, prog.hp)
            swap(self.t_segs, prog.segs)
            var pops = self.t_ops.unsafe_ptr()
            var pmats = self.t_mats.unsafe_ptr()
            var php = self.t_hp.unsafe_ptr()
            var psegs = self.t_segs.unsafe_ptr()
            ref ctx = self.ctx.value()
            var dops = ctx.enqueue_create_buffer[DType.int32](len(self.t_ops))
            var dmats = ctx.enqueue_create_buffer[Self.dtype](len(self.t_mats))
            var dhp = ctx.enqueue_create_buffer[DType.int32](len(self.t_hp))
            var dsegs = ctx.enqueue_create_buffer[DType.int32](len(self.t_segs))
            ctx.enqueue_copy(dops, pops)
            ctx.enqueue_copy(dmats, pmats)
            ctx.enqueue_copy(dhp, php)
            ctx.enqueue_copy(dsegs, psegs)
            var full = prog.full_registers()
            # dynamic shared memory per launch (kernels.tile_smem_bytes), the
            # opt-in limit above 48 KiB being one constant per device
            # (tile_func_attribute)
            var es = size_of[Self.dtype]()
            var fa = tile_func_attribute(self.profile)
            for r in range(nruns):
                var run = prog.runs[r]
                var init = self.init_index if r == 0 else -1
                var sb = 0  # static shared arrays (kernels.TILE_SMEM_DYNAMIC off)
                comptime if TILE_SMEM_DYNAMIC:
                    sb = tile_smem_bytes(es, k, low, run.o1 - run.o0, run.m1 - run.m0, run.s1 - run.s0)
                # four specialisations: FULL x HEAVY (see kernels.gk_tile)
                if full and not heavies[r]:
                    enqueue_tile[Self.dtype, True, False](
                        ctx, self.dre.value(), self.dim.value(), dops, dmats, dhp, dsegs, run.s0, run.s1, run.hoff,
                        run.nh, low, k, nr, run.fin, init, run.o0, run.o1 - run.o0, run.m0, run.m1 - run.m0,
                        nblocks, nthr, sb, fa)
                elif full:
                    enqueue_tile[Self.dtype, True, True](
                        ctx, self.dre.value(), self.dim.value(), dops, dmats, dhp, dsegs, run.s0, run.s1, run.hoff,
                        run.nh, low, k, nr, run.fin, init, run.o0, run.o1 - run.o0, run.m0, run.m1 - run.m0,
                        nblocks, nthr, sb, fa)
                elif not heavies[r]:
                    enqueue_tile[Self.dtype, False, False](
                        ctx, self.dre.value(), self.dim.value(), dops, dmats, dhp, dsegs, run.s0, run.s1, run.hoff,
                        run.nh, low, k, nr, run.fin, init, run.o0, run.o1 - run.o0, run.m0, run.m1 - run.m0,
                        nblocks, nthr, sb, fa)
                else:
                    enqueue_tile[Self.dtype, False, True](
                        ctx, self.dre.value(), self.dim.value(), dops, dmats, dhp, dsegs, run.s0, run.s1, run.hoff,
                        run.nh, low, k, nr, run.fin, init, run.o0, run.o1 - run.o0, run.m0, run.m1 - run.m0,
                        nblocks, nthr, sb, fa)
            self.d_ops = dops^
            self.d_mats = dmats^
            self.d_hp = dhp^
            self.d_segs = dsegs^
            self.in_flight = True
            self.launches += nruns
            self.init_index = -1
        else:
            raise Self._no_device()

    def _one(mut self, op: KOp) raises:
        var ops = List[KOp]()
        ops.append(op)
        self.run_ops(ops)

    # ------------------------------------------------------------ single gates
    # Each call is one tile run (one launch). Prefer evolve / execute_plan /
    # run_ops for sequences: they pack many gates into each launch.
    def apply_1q(mut self, m: M2, t: Int) raises:
        """Dense 2x2 on qubit t (diagonal / X matrices take cheaper tile ops)."""
        var op = KOp.make(K.ONE, t, -1, Instruction.make(Op.U, t))
        op.m = m
        self._one(op)

    def apply_diag_1q(mut self, d0r: Float64, d0i: Float64, d1r: Float64, d1i: Float64, t: Int) raises:
        """diag(d0, d1) on qubit t."""
        var m = M2.identity()
        m.re[0] = d0r
        m.im[0] = d0i
        m.re[3] = d1r
        m.im[3] = d1i
        self.apply_1q(m, t)

    def apply_x(mut self, t: Int) raises:
        var m = M2.identity()
        m.re[0] = 0.0
        m.re[3] = 0.0
        m.re[1] = 1.0
        m.re[2] = 1.0
        self.apply_1q(m, t)

    def apply_cx(mut self, c: Int, t: Int) raises:
        self._one(KOp.make(K.CX, c, t, Instruction.make(Op.CX, c, t)))

    def apply_cphase(mut self, pr: Float64, pi_: Float64, a: Int, b: Int) raises:
        """Multiply amplitudes with bits a and b set by (pr + i pi_) (CZ, CP)."""
        var op = KOp.make(K.CPHASE, a, b, Instruction.make(Op.CP, a, b))
        op.f0 = pr
        op.f1 = pi_
        self._one(op)

    def apply_controlled_1q(mut self, m: M2, c: Int, t: Int) raises:
        var op = KOp.make(K.C1, c, t, Instruction.make(Op.CRY, c, t))
        op.m = m
        self._one(op)

    def apply_rzz(mut self, theta: Float64, a: Int, b: Int) raises:
        """exp(-i theta/2 Z_a Z_b)."""
        var op = KOp.make(K.RZZ, a, b, Instruction.make(Op.RZZ, a, b))
        op.f0 = cos(theta / 2)
        op.f1 = sin(theta / 2)
        self._one(op)

    def apply_swap(mut self, a: Int, b: Int) raises:
        self._one(KOp.make(K.SWAP, a, b, Instruction.make(Op.SWAP, a, b)))

    def apply_2q(mut self, m: M4, q0: Int, q1: Int) raises:
        """Dense 4x4 on (q0, q1); matrix index = b(q0) + 2 b(q1)."""
        var op = KOp.make(K.TWO, q0, q1, Instruction.make(Op.ECR, q0, q1))
        op.m4 = m
        self._one(op)

    def apply_ccx(mut self, c0: Int, c1: Int, t: Int) raises:
        self._one(KOp.make(K.OTHER, max(c0, max(c1, t)), -1, Instruction.make(Op.CCX, c0, c1, t)))

    def apply_ccz(mut self, a: Int, b: Int, c: Int) raises:
        self._one(KOp.make(K.OTHER, max(a, max(b, c)), -1, Instruction.make(Op.CCZ, a, b, c)))

    def apply_cswap(mut self, c: Int, a: Int, b: Int) raises:
        self._one(KOp.make(K.OTHER, max(c, max(a, b)), -1, Instruction.make(Op.CSWAP, c, a, b)))

    def apply_global_phase(mut self, phase: Float64) raises:
        if phase == 0.0:
            return
        self.run_ops(List[KOp](), phase)

    # ------------------------------------------------------------ plans
    def apply_kop(mut self, op: KOp) raises:
        """One op of `qumo.sim.runner.build_plan` (one tile run)."""
        self._one(op)

    def execute_plan(mut self, plan: List[KOp]) raises:
        """Enqueue a fused plan as tile runs (asynchronous, no host syncs)."""
        self.run_ops(plan)

    def evolve(mut self, circuit: Circuit, params: List[Float64] = List[Float64]()) raises:
        """Apply the unitary part of `circuit` (fused plan + global phase) to
        the current state. Measurements/resets are skipped (like
        `qumo.sim.runner.evolve_into`); conditioned gates raise QM-E401."""
        circuit.validate()
        if circuit.num_qubits > self.num_qubits:
            raise Error("QM-E474: circuit has " + String(circuit.num_qubits) + " qubits but the GPU state has " + String(self.num_qubits) + ". Hint: allocate GpuStateVector(circuit.num_qubits).")
        if len(params) != circuit.num_params():
            raise Error("QM-E101: circuit has " + String(circuit.num_params()) + " parameters " + circuit.param_names_str() + " but " + String(len(params)) + " values were given.")
        var plan = build_plan(circuit, params)
        self.run_ops(plan, circuit.global_phase)

    # ------------------------------------------------------------ readout
    def to_host(self) raises -> StateVector:
        """Copy the amplitudes back (converted to float64)."""
        self._materialize()
        var sv = StateVector(self.num_qubits)
        var n = self.size()
        if self.emulated:
            for i in range(n):
                sv.re[i] = Float64(self.hre[i])
                sv.im[i] = Float64(self.him[i])
            return sv^
        comptime if device_kernels_enabled[Self.dtype]():
            ref ctx = self.ctx.value()
            var hr = ctx.enqueue_create_host_buffer[Self.dtype](n)
            var hi = ctx.enqueue_create_host_buffer[Self.dtype](n)
            self.dre.value().enqueue_copy_to(hr)
            self.dim.value().enqueue_copy_to(hi)
            ctx.synchronize()
            for i in range(n):
                sv.re[i] = Float64(hr[i])
                sv.im[i] = Float64(hi[i])
            return sv^
        else:
            raise Self._no_device()

    def probabilities(self) raises -> List[Float64]:
        """|psi_i|^2 for every basis state (computed on the device)."""
        self._materialize()
        var n = self.size()
        var out = List[Float64](length=n, fill=0.0)
        var nb = stride_blocks(n)
        var nthreads = nb * TPB
        if self.emulated:
            var tmp = List[Scalar[Self.dtype]](length=n, fill=0)
            var re = self._re()
            var im = self._im()
            var dst = _mptr[Self.dtype](tmp)

            def f(g: Int) {imm re, imm im, imm dst, imm n, imm nthreads}:
                th_prob[Self.dtype](g, nthreads, n, re, im, dst)

            emu_threads(f, nthreads)
            for i in range(n):
                out[i] = Float64(tmp[i])
            return out^
        comptime if device_kernels_enabled[Self.dtype]():
            ref ctx = self.ctx.value()
            var dp = ctx.enqueue_create_buffer[Self.dtype](n)
            ctx.enqueue_function[gk_prob[Self.dtype]](
                self.dre.value(), self.dim.value(), dp, Int64(n), Int64(nthreads),
                grid_dim=nb, block_dim=TPB)
            var hp = ctx.enqueue_create_host_buffer[Self.dtype](n)
            dp.enqueue_copy_to(hp)
            ctx.synchronize()
            for i in range(n):
                out[i] = Float64(hp[i])
            return out^
        else:
            raise Self._no_device()

    def pauli_sums(self, xs: List[Int], zs: List[Int]) raises -> List[Float64]:
        """For each (x, z) mask pair: sum_i conj(psi[i^x]) psi[i] (-1)^{|i&z|}
        as [re0, im0, re1, im1, ...] (fp64 sums of the per-block partials).

        Terms are grouped by X mask; one launch reduces up to EXP_G terms of a
        group in one pass over the state; one device->host copy for all.
        """
        self._materialize()
        var n = self.size()
        var nt = len(xs)
        var nb = reduction_blocks(n)
        var out = List[Float64](length=2 * nt, fill=0.0)
        if nt == 0:
            return out^
        # slot order: terms grouped by x (first occurrence order); launches = chunks
        var order = List[Int](capacity=nt)
        var group_of = Dict[Int, Int]()
        var groups = List[List[Int]]()
        for t in range(nt):
            var g = group_of.get(xs[t], -1)
            if g < 0:
                g = len(groups)
                group_of[xs[t]] = g
                groups.append(List[Int]())
            groups[g].append(t)
        var lx = List[Int]()  # per launch: x mask, first slot, count
        var l0 = List[Int]()
        var lc = List[Int]()
        var zslots = List[Int64](capacity=nt)
        for g in range(len(groups)):
            ref ts = groups[g]
            var j = 0
            while j < len(ts):
                var c = min(EXP_G, len(ts) - j)
                lx.append(xs[ts[j]])
                l0.append(len(order))
                lc.append(c)
                for u in range(c):
                    order.append(ts[j + u])
                    zslots.append(Int64(zs[ts[j + u]]))
                j += c
        var hpr = List[Float64](length=nt * nb, fill=0.0)
        var hpi = List[Float64](length=nt * nb, fill=0.0)
        if self.emulated:
            var pre = List[Scalar[Self.dtype]](length=nt * nb, fill=0)
            var pim = List[Scalar[Self.dtype]](length=nt * nb, fill=0)
            var pz = zslots.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin()
            for L in range(len(lx)):
                emu_expect[Self.dtype](self._re(), self._im(), lx[L], pz, l0[L], lc[L], n, nb,
                                       _mptr[Self.dtype](pre), _mptr[Self.dtype](pim), l0[L] * nb)
            for i in range(nt * nb):
                hpr[i] = Float64(pre[i])
                hpi[i] = Float64(pim[i])
            _ = len(zslots)
        else:
            comptime if device_kernels_enabled[Self.dtype]():
                ref ctx = self.ctx.value()
                var dpr = ctx.enqueue_create_buffer[Self.dtype](nt * nb)
                var dpi = ctx.enqueue_create_buffer[Self.dtype](nt * nb)
                var dz = ctx.enqueue_create_buffer[DType.int64](nt)
                ctx.enqueue_copy(dz, zslots.unsafe_ptr())
                for L in range(len(lx)):
                    ctx.enqueue_function[gk_expect[Self.dtype]](
                        self.dre.value(), self.dim.value(), Int64(lx[L]), dz, Int32(l0[L]), Int32(lc[L]),
                        Int64(n), Int64(nb * TPB), dpr, dpi, Int64(l0[L] * nb), Int64(nb),
                        grid_dim=nb, block_dim=TPB)
                var hr = ctx.enqueue_create_host_buffer[Self.dtype](nt * nb)
                var hi = ctx.enqueue_create_host_buffer[Self.dtype](nt * nb)
                dpr.enqueue_copy_to(hr)
                dpi.enqueue_copy_to(hi)
                ctx.synchronize()
                _ = len(zslots)
                for i in range(nt * nb):
                    hpr[i] = Float64(hr[i])
                    hpi[i] = Float64(hi[i])
            else:
                raise Self._no_device()
        for slot in range(nt):
            var sr = 0.0
            var si = 0.0
            for b in range(nb):
                sr += hpr[slot * nb + b]
                si += hpi[slot * nb + b]
            var t = order[slot]
            out[2 * t] = sr
            out[2 * t + 1] = si
        return out^

    def norm_sq(self) raises -> Float64:
        """<psi|psi> by the same GPU tree reduction (identity Pauli term)."""
        var xs: List[Int] = [0]
        var zs: List[Int] = [0]
        return self.pauli_sums(xs, zs)[0]

    def expectation(self, h: PauliSum) raises -> Float64:
        """<psi|H|psi> for a Pauli sum (GPU reductions, fp64 host combine)."""
        if h.num_qubits() > self.num_qubits:
            raise Error("QM-E405: observable acts on qubit " + String(h.num_qubits() - 1) + " but the state has " + String(self.num_qubits) + " qubits. Hint: check Pauli indices (0-based, qubit 0 = least significant).")
        var xs = List[Int]()
        var zs = List[Int]()
        for t in h.terms:
            xs.append(Int(t.x))
            zs.append(Int(t.z))
        var s = self.pauli_sums(xs, zs)
        var total = 0.0
        for k in range(len(h.terms)):
            ref t = h.terms[k]
            var sr = s[2 * k]
            var si = s[2 * k + 1]
            var ny = Int(pop_count(t.x & t.z)) & 3
            # multiply by i^ny, keep the real part (same convention as StateVector)
            var v = sr
            if ny == 1:
                v = -si
            elif ny == 2:
                v = -sr
            elif ny == 3:
                v = si
            total += t.coeff * v
        return total

    def segment_sums(self, seg: Int) raises -> List[Float64]:
        """Probability mass of each `seg`-amplitude segment (one block each)."""
        self._materialize()
        var n = self.size()
        var nseg = n // seg
        var out = List[Float64](length=nseg, fill=0.0)
        if self.emulated:
            var tmp = List[Scalar[Self.dtype]](length=nseg, fill=0)
            emu_segsum[Self.dtype](self._re(), self._im(), seg, nseg, _mptr[Self.dtype](tmp))
            for i in range(nseg):
                out[i] = Float64(tmp[i])
            return out^
        comptime if device_kernels_enabled[Self.dtype]():
            ref ctx = self.ctx.value()
            var ds = ctx.enqueue_create_buffer[Self.dtype](nseg)
            ctx.enqueue_function[gk_segsum[Self.dtype]](
                self.dre.value(), self.dim.value(), Int64(seg), ds,
                grid_dim=nseg, block_dim=TPB)
            var hs = ctx.enqueue_create_host_buffer[Self.dtype](nseg)
            ds.enqueue_copy_to(hs)
            ctx.synchronize()
            for i in range(nseg):
                out[i] = Float64(hs[i])
            return out^
        else:
            raise Self._no_device()

    def sample(self, shots: Int, seed: UInt64) raises -> List[Int]:
        var rng = Rng(seed)
        return self.sample_rng(shots, rng)

    def sample_rng(self, shots: Int, mut rng: Rng) raises -> List[Int]:
        """Basis-state indices of `shots` samples (two-level inverse CDF).

        Same RNG use as `StateVector.sample`: `shots` sorted uniforms, then a
        Fisher-Yates shuffle, so results are reproducible for a seed.
        """
        var out = List[Int](capacity=max(shots, 0))
        if shots <= 0:
            return out^
        var us = List[Float64](capacity=shots)
        for _ in range(shots):
            us.append(rng.uniform())
        sort(us)
        var n = self.size()
        var seg = min(sample_seg[Self.dtype](), n)
        var nseg = n // seg
        var sums = self.segment_sums(seg)  # also materialises a lazy state
        var total = 0.0
        var last_pos = -1
        for s in range(nseg):
            total += sums[s]
            if sums[s] > 0.0:
                last_pos = s
        if not (total > 0.0):
            raise Error("QM-E473: cannot sample a state with zero norm. Hint: check the circuit / initial state.")
        # sorted targets -> (segment, residual) by an fp64 prefix over segments;
        # consecutive shots in one segment form one block of the sampling kernel
        var bseg = List[Int64]()
        var blo = List[Int32]()
        var resid = List[Scalar[Self.dtype]](capacity=shots)
        var s = 0
        var cum = 0.0
        for k in range(shots):
            var target = us[k] * total
            while s < nseg - 1 and cum + sums[s] <= target:
                cum += sums[s]
                s += 1
            var seg_k = s
            var r = target - cum
            if not (sums[s] > 0.0) or seg_k > last_pos:
                seg_k = last_pos  # rounding past the end: last nonzero amplitude
                r = sums[last_pos]
            if len(bseg) == 0 or Int(bseg[len(bseg) - 1]) != seg_k:
                bseg.append(Int64(seg_k))
                blo.append(Int32(k))
            resid.append(Scalar[Self.dtype](r))
        blo.append(Int32(shots))
        var nblk = len(bseg)
        var idx = List[Int64](length=shots, fill=0)
        if self.emulated:
            var pseg = bseg.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin()
            var plo = blo.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin()
            var pidx = idx.unsafe_ptr().as_unsafe_any_origin()
            emu_sample[Self.dtype](self._re(), self._im(), seg, pseg, plo, _mptr[Self.dtype](resid), pidx, nblk)
            _ = len(bseg)
            _ = len(blo)
            _ = len(resid)
        else:
            comptime if device_kernels_enabled[Self.dtype]():
                ref ctx = self.ctx.value()
                var dseg = ctx.enqueue_create_buffer[DType.int64](nblk)
                var dlo = ctx.enqueue_create_buffer[DType.int32](nblk + 1)
                var dres = ctx.enqueue_create_buffer[Self.dtype](shots)
                var douts = ctx.enqueue_create_buffer[DType.int64](shots)
                ctx.enqueue_copy(dseg, bseg.unsafe_ptr())
                ctx.enqueue_copy(dlo, blo.unsafe_ptr())
                ctx.enqueue_copy(dres, resid.unsafe_ptr())
                ctx.enqueue_function[gk_sample[Self.dtype]](
                    self.dre.value(), self.dim.value(), Int64(seg), dseg, dlo, dres, douts,
                    grid_dim=nblk, block_dim=TPB)
                var ho = ctx.enqueue_create_host_buffer[DType.int64](shots)
                douts.enqueue_copy_to(ho)
                ctx.synchronize()
                _ = len(bseg)
                _ = len(blo)
                _ = len(resid)
                for k in range(shots):
                    idx[k] = ho[k]
            else:
                raise Self._no_device()
        for k in range(shots):
            out.append(Int(idx[k]))
        # shuffle outcomes so memory order is not sorted (same as StateVector.sample)
        var k = shots - 1
        while k > 0:
            var j = rng.below(k + 1)
            var tmp = out[k]
            out[k] = out[j]
            out[j] = tmp
            k -= 1
        return out^


# ---------------------------------------------------------------------------
# Circuit-level entry points (mirror qumo.sim.runner)
# ---------------------------------------------------------------------------
def _check_params(circuit: Circuit, params: List[Float64]) raises:
    if len(params) != circuit.num_params():
        raise Error("QM-E101: circuit has " + String(circuit.num_params()) + " parameters " + circuit.param_names_str() + " but " + String(len(params)) + " values were given.")


def evolve_gpu[dt: DType = GPU_DEFAULT_DTYPE](circuit: Circuit, params: List[Float64] = List[Float64](),
                                              emulate: Bool = False) raises -> GpuStateVector[dt]:
    """Final state of `circuit` from |0...0> on the GPU (or its emulator)."""
    circuit.validate()
    _check_params(circuit, params)
    var sv = GpuStateVector[dt](circuit.num_qubits, emulate)
    sv.evolve(circuit, params)
    return sv^


def expectation_gpu[dt: DType = GPU_DEFAULT_DTYPE](circuit: Circuit, observable: PauliSum,
                                                   params: List[Float64] = List[Float64](),
                                                   emulate: Bool = False) raises -> Float64:
    var sv = evolve_gpu[dt](circuit, params, emulate)
    return sv.expectation(observable)


def sample_gpu[dt: DType = GPU_DEFAULT_DTYPE](circuit: Circuit, shots: Int, seed: UInt64 = 1234,
                                              params: List[Float64] = List[Float64](),
                                              emulate: Bool = False) raises -> SampleResult:
    """Static sampling on the GPU (same contract as runner.sample's static path).

    SampleResult.method = 'gpu:statevector:fp32|fp64' ('gpu-emulator:...' when
    emulated). Dynamic or noisy circuits raise QM-E471 (use runner.sample).
    """
    circuit.validate()
    check_statevector_size(circuit.num_qubits)
    _check_params(circuit, params)
    if circuit.num_clbits > 62:
        raise Error("QM-E406: the sampler packs classical registers into 62 bits; circuit has " + String(circuit.num_clbits) + " clbits. Hint: measure fewer bits or split the register.")
    if circuit.num_measurements == 0:
        raise Error("QM-E402: circuit has no measurements. Hint: call measure_all() or use evolve()/expectation().")
    if circuit.is_dynamic() or circuit.has_noise():
        raise Error("QM-E471: the GPU sampler runs static circuits only (terminal measurements, no noise). Hint: use qumo.exec.dispatch.sample_auto, which falls back to CPU trajectories.")
    var rng = Rng(seed)
    var sv = GpuStateVector[dt](circuit.num_qubits, emulate)
    sv.evolve(circuit, params)
    # clbit <- qubit, last MEASURE into a clbit wins (runner.sample semantics)
    var mm = terminal_measure_map(circuit)
    var outs = sv.sample_rng(shots, rng)
    var res = SampleResult(circuit.num_clbits, shots, sv.method_name())
    for o in outs:
        res.record(mm.pack(o))
    return res^
