"""GPU availability, device info and precision policy.

Compile-time gating (the only way QUMO touches a GPU toolchain):
  * Default builds (`make test`, `make cli`, `make py`) never compile a GPU
    kernel: every launch sits inside `comptime if is_defined["QUMO_GPU"]()`.
    `gpu_compiled()` is then False and `gpu_available()` is False without
    creating a `DeviceContext`.
  * `-D QUMO_GPU` builds compile the kernels for the build's target
    accelerator (the host's GPU by default, or `--target-accelerator=...`).
    Building on a Mac needs Xcode + `xcodebuild -downloadComponent
    MetalToolchain` (the Metal compiler produces the metallib at build time).

Precision policy:
  * Apple GPUs (Metal) have no float64 -> float32, always.
  * NVIDIA / AMD -> float64 by default (closer to the CPU simulator, ~1e-12);
    `-D QUMO_GPU_FP32` makes float32 the default for speed / memory (2x the
    qubits per GiB). `GpuStateVector[DType.float32]` is always selectable.
    This is NOT CUDA-Q's choice: CUDA-Q's `nvidia` target defaults to fp32 on
    every GPU, fp64 being an option (runtime/cudaq/platform/default/
    nvidia.yml L21-32 at tag 0.16.0). On GPUs with weak fp64 (T4, L4, RTX:
    1/32 or 1/64 of the fp32 rate) QUMO's default is therefore the slower
    precision; whether to switch defaults there is a user decision.

Per-device tuning of the tile engine (`GpuProfile`): see the struct below.
"""

from std.sys.defines import is_defined, get_defined_int
from std.sys import has_apple_gpu_accelerator, has_nvidia_gpu_accelerator, has_amd_gpu_accelerator
from std.os import getenv
from max.gpu.host import DeviceContext, DeviceAttribute

from qumo.gpu.kernels import (
    LOW_BITS, LOW_MIN, REG_BITS, TILE_SMEM_DYNAMIC, DIAG_MAX_WIDTH, TILE_SAFE_THREADS, TILE_LAUNCH_BOUND, tile_bits,
    tile_smem_typical, tile_smem_worst,
)

from qumo.ai.json import JsonWriter

comptime GPU_COMPILED = is_defined["QUMO_GPU"]()
comptime GPU_IS_APPLE = has_apple_gpu_accelerator()
comptime GPU_FP64_SUPPORTED = not has_apple_gpu_accelerator()
comptime GPU_DEFAULT_DTYPE = DType.float32 if (has_apple_gpu_accelerator() or is_defined["QUMO_GPU_FP32"]()) else DType.float64

# Automatic dispatch sends static non-Clifford circuits with at least this many
# qubits to the GPU. Device-specific, from `bench_gpu N R crossover` (end to
# end: DeviceContext + allocation + evolve + 1000 shots vs the CPU sampler,
# interleaved, medians):
#   * Apple M5 (Metal, unified memory), measured: the GPU call costs ~2-6 ms
#     regardless of size up to 18 qubits; the CPU wins up to 12 qubits
#     (< 1 ms) and loses from 14 on (the CPU threaded path costs tens of ms on
#     a loaded machine). 16 keeps a margin for an idle CPU -> 16 on Apple.
#   * NVIDIA / AMD: not measured here (no device); 20 keeps the previous
#     conservative heuristic (discrete memory, larger launch latency).
# Override at build time with -D QUMO_GPU_MIN_QUBITS=<n>.
comptime GPU_MIN_QUBITS = get_defined_int["QUMO_GPU_MIN_QUBITS", 16 if has_apple_gpu_accelerator() else 20]()


# Tile-engine geometry bounds (see qumo.gpu.kernels / tiles): a tile keeps the
# `low` lowest qubits resident so that consecutive threads touch consecutive
# amplitudes; LOW_MIN..LOW_BITS is the range the kernels support (the kernel's
# offset table is sized for LOW_MIN), LOW_BITS the default.
comptime LOW_MAX = LOW_BITS


@fieldwise_init
struct TileKernelLimits(ImplicitlyCopyable, Writable):
    """What the compiled tile kernels (kernels.gk_tile, its 4
    specialisations) admit on a device, read back from the driver (NVIDIA /
    AMD; Metal reports nothing): -1 = unknown.
      * max_threads: MAX_THREADS_PER_BLOCK, minimum over the variants (a
        block larger than this fails to launch: the register count times
        the block size exceeds the register file);
      * two_block_threads: largest block size 2^(k - REG_BITS) at which the
        occupancy calculator keeps >= 2 blocks of a typical run resident per
        multiprocessor (registers and shared memory), TILE_SAFE_THREADS when
        no larger tile does;
      * regs / local_bytes: NUM_REGS and LOCAL_SIZE_BYTES (spill space),
        maximum over the variants (diagnostics, reported in the profile)."""

    var max_threads: Int
    var two_block_threads: Int
    var regs: Int
    var local_bytes: Int

    @staticmethod
    def unknown() -> TileKernelLimits:
        return TileKernelLimits(-1, -1, -1, -1)

    def write_to(self, mut w: Some[Writer]):
        w.write("max_threads=", self.max_threads, " two_block_threads=", self.two_block_threads, " regs=", self.regs,
                " local_bytes=", self.local_bytes)


def _env_int(name: String, lo: Int, hi: Int) -> Int:
    """Integer environment override in [lo, hi], or -1 when unset / invalid."""
    var v = getenv(name)
    if v.byte_length() == 0:
        return -1
    try:
        var x = Int(v)
        if x >= lo and x <= hi:
            return x
    except:
        pass
    return -1


struct GpuProfile(Copyable, Writable):
    """Per-device tuning of the tile engine, built once per DeviceContext.

    Hardware neutrality means one kernel source and *measured* parameters,
    not vendor forks: the profile records what the runtime reports (each
    query guarded: Metal raises 'Attributes not supported') and the tuning
    the engine uses on this device:
      * low_fp32 / low_fp64: always-resident low qubits of a tile (2^low
        contiguous amplitudes per chunk of a tile load; fewer = more high
        qubits per tile = fewer state sweeps, if the device streams short
        chunks at full bandwidth: `bench_gpu N R probe`, stream_*B_GBps).
      * diag_width: widest group of consecutive diagonal gates fused into one
        phase table (QUMO_GPU_DIAG_WIDTH, 0..6; <= 1 disables the fusion).
      * split_1q_fp32 / fp64: run complex 1q gates of the form diagonal .
        real 2x2 (RZ . RY products, tiles.split_1q_ok) as a real 2x2 (G1R,
        half the multiply-adds of G1) + a diagonal that fuses into a phase
        table; generic U3 gates stay dense (splitting them would add
        multiply-adds). Off by default everywhere: up to 9% slower on the
        M5 (fp32), unmeasured on the weak-fp64 GPUs it targets (weak_fp64);
        QUMO_GPU_SPLIT_1Q (both precisions) / QUMO_GPU_FP64_LEAN (fp64)
        switch it on.
      * tile_bits_fp32 / fp64: log2 amplitudes per tile (one block's shared
        memory; 2^(k - REG_BITS) threads per block). The default (11 / 10,
        16 KiB of amplitudes) grows by up to 2 when (a) the device reports
        enough shared memory for the worst-case run at that size and for 2
        typical runs per multiprocessor, and (b) the block size is known to
        launch: <= TILE_SAFE_THREADS (256), or <= the compiled kernel's
        MAX_THREADS_PER_BLOCK and with >= 2 resident blocks per
        multiprocessor by the occupancy calculator (kl_fp32 / kl_fp64,
        filled by GpuStateVector on NVIDIA / AMD when a larger tile is
        possible at all). Larger tiles = more gates per state sweep.
        Computed from published limits (not run): T4, MI300X (AMD: launch
        bound 256) and Apple keep 11 / 10; A100 / H100 / L4 get fp64 11 at
        256 threads, and fp32 12-13 / fp64 12 only if the kernel reports
        512 / 1024-thread blocks (<= 128 / 64 registers per thread).
    Defaults per vendor come from measurements on that vendor (docs/GPU.md);
    unmeasured vendors keep LOW_BITS. Environment overrides, read at
    runtime so one session can sweep values without rebuilding:
    QUMO_GPU_LOW_BITS (LOW_MIN..LOW_BITS) and QUMO_GPU_TILE_BITS (both
    precisions; ignored if the worst-case run would not fit the device's
    shared memory or the block size is not known to launch, see above)."""

    var vendor: String
    var name: String
    var api: String
    var arch: String  # NVIDIA: compute capability (75 = sm_75); AMD / Apple: arch name
    var multiprocessors: Int  # -1: not reported
    var smem_per_block: Int  # bytes of static shared memory per block, -1: not reported
    var smem_per_block_optin: Int  # NVIDIA opt-in limit for dynamic shared memory, -1: n/a
    var smem_per_multiprocessor: Int
    var warp: Int
    var low_fp32: Int
    var low_fp64: Int
    var tile_bits_fp32: Int
    var tile_bits_fp64: Int
    var kl_fp32: TileKernelLimits  # compiled tile kernel limits (GpuStateVector fills them; -1: unknown)
    var kl_fp64: TileKernelLimits
    var diag_width: Int  # widest fused diagonal phase table (qubits; <= 1: no fusion)
    var split_1q_fp32: Bool  # complex 1q gates of the form diagonal . real 2x2 as G1R + diagonal (arithmetic-bound devices)
    var split_1q_fp64: Bool
    var source: String  # where the tuning came from ("defaults", "env QUMO_GPU_LOW_BITS=4")

    def __init__(out self, vendor: String):
        """Defaults of a compile-time vendor, no device queries."""
        self.vendor = vendor
        self.name = ""
        self.api = ""
        self.arch = ""
        self.multiprocessors = -1
        self.smem_per_block = -1
        self.smem_per_block_optin = -1
        self.smem_per_multiprocessor = -1
        self.warp = 64 if vendor == "amd" else 32
        # Apple M5 (measured, docs/GPU.md "Optimisation log"): 64 B chunks stream
        # at 94% of 128 B ones and low = 4 is 0-5% faster than 5 with 15-18%
        # fewer state sweeps; 32 B chunks (low = 3) reach only 79%.
        # NVIDIA / AMD: not measured yet -> LOW_BITS (sweep QUMO_GPU_LOW_BITS
        # = 5, 4, 3 with `bench_gpu N R probe` / families on the device).
        self.low_fp32 = 4 if vendor == "apple" else LOW_BITS
        self.low_fp64 = 4 if vendor == "apple" else LOW_BITS
        self.tile_bits_fp32 = tile_bits[DType.float32]()
        self.tile_bits_fp64 = tile_bits[DType.float64]()
        self.kl_fp32 = TileKernelLimits.unknown()
        self.kl_fp64 = TileKernelLimits.unknown()
        self.diag_width = DIAG_MAX_WIDTH
        self.split_1q_fp32 = False
        self.split_1q_fp64 = False
        self.source = "defaults"
        self._apply_env()

    @staticmethod
    def emulator() -> GpuProfile:
        """Profile of the host emulator (defaults + environment overrides)."""
        var p = GpuProfile("emulator")
        p.name = "host-emulator"
        return p^

    @staticmethod
    def of(ctx: DeviceContext, kl32: TileKernelLimits = TileKernelLimits.unknown(),
           kl64: TileKernelLimits = TileKernelLimits.unknown()) -> GpuProfile:
        """Query the device behind `ctx` (never raises: missing facts are -1).
        kl32 / kl64: limits of the compiled tile kernels on this device
        (GpuStateVector reads them when wants_kernel_limits says a larger
        tile is possible; unknown: tiles stay at <= TILE_SAFE_THREADS
        threads)."""
        var p = GpuProfile(gpu_vendor())
        p.kl_fp32 = kl32
        p.kl_fp64 = kl64
        p.name = ctx.name()
        p.api = ctx.api()
        try:
            if p.vendor == "nvidia":
                p.arch = String(ctx.compute_capability())
            else:
                p.arch = ctx.arch_name()
        except:
            pass
        p.multiprocessors = _attr(ctx, DeviceAttribute.MULTIPROCESSOR_COUNT)
        p.smem_per_block = _attr(ctx, DeviceAttribute.MAX_SHARED_MEMORY_PER_BLOCK)
        p.smem_per_block_optin = _attr(ctx, DeviceAttribute.MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
        p.smem_per_multiprocessor = _attr(ctx, DeviceAttribute.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)
        var w = _attr(ctx, DeviceAttribute.WARP_SIZE)
        if w > 0:
            p.warp = w
        p.tile_bits_fp32 = p._pick_tile_bits(4, tile_bits[DType.float32]())
        p.tile_bits_fp64 = p._pick_tile_bits(8, tile_bits[DType.float64]())
        # split_1q stays off by default: on the M5 (fp32) it is up to 9%
        # slower on layered circuits (more ops; the 2x2 arithmetic is not
        # the bottleneck there) and neutral elsewhere; on weak-fp64 GPUs it
        # is the fp64 lever, but unmeasured: decide from a device run
        # (QUMO_GPU_FP64_LEAN=1 vs 0, docs/GPU.md).
        p.source = "defaults"
        p._apply_env()
        return p^

    def weak_fp64(self) -> Bool:
        """GPUs whose fp64 rate is 1/32 or 1/64 of fp32 (consumer / inference
        parts): NVIDIA sm_61, 75 (T4), 86 (A10, RTX 30), 89 (L4, RTX 40),
        120 (RTX 50); AMD RDNA (gfx10xx, gfx11xx, gfx12xx). There every engine
        is bound by fp64 multiply-adds; split_1q_fp64 (QUMO_GPU_FP64_LEAN=1)
        is the lever for them, but it stays OFF until a device run shows a
        gain (reported here for the Colab A/B only; it cuts multiply-adds
        only on RZ . RY-type circuits). Data-center parts
        (sm_60/70/80/90/100, gfx90a / gfx942) have full-rate fp64."""
        if self.vendor == "nvidia":
            return self.arch == "61" or self.arch == "75" or self.arch == "86" or self.arch == "89" or self.arch == "120"
        if self.vendor == "amd":
            return self.arch.startswith("gfx10") or self.arch.startswith("gfx11") or self.arch.startswith("gfx12")
        return False

    def split_1q[dt: DType](self) -> Bool:
        """Split complex 1q gates (TileProgram.split_1q) for precision dt."""
        comptime if dt == DType.float64:
            return self.split_1q_fp64
        else:
            return self.split_1q_fp32

    def _smem_limit(self) -> Int:
        """Shared memory one block may use (-1: unknown). With an opt-in limit
        (NVIDIA), the opt-in minus 1 KiB: the constant
        MAX_DYNAMIC_SHARED_SIZE_BYTES the tile kernel is loaded with
        (tile_smem_optin, the value MAX itself derives)."""
        if self.smem_per_block_optin > 48 * 1024:
            return self.smem_per_block_optin - 1024
        return self.smem_per_block_optin if self.smem_per_block_optin > 0 else self.smem_per_block

    def tile_smem_optin(self) -> Int:
        """MAX_DYNAMIC_SHARED_SIZE_BYTES for the tile kernel on this device:
        one value per device (0 = not needed: no opt-in above the 48 KiB
        default, or static shared arrays). A constant, not each launch's
        size, so every load of the kernel in a process carries the same
        attribute whatever tile size or run a launch has."""
        comptime if TILE_SMEM_DYNAMIC:
            if self.smem_per_block_optin > 48 * 1024:
                return self.smem_per_block_optin - 1024
        return 0

    def _kl(self, es: Int) -> TileKernelLimits:
        return self.kl_fp32 if es == 4 else self.kl_fp64

    def _thread_cap(self, es: Int) -> Int:
        """Largest gk_tile block that is known to launch for element size es:
        the compiled kernel's report when there is one, else
        TILE_SAFE_THREADS."""
        var cap = TILE_SAFE_THREADS
        var mt = self._kl(es).max_threads
        if mt > 0:
            cap = min(mt, 1024)
        comptime if TILE_LAUNCH_BOUND > 0:
            cap = min(cap, TILE_LAUNCH_BOUND)
        return cap

    def _tile_fits(self, es: Int, k: Int, thread_cap: Int = -1) -> Bool:
        """Worst-case run at tile size k fits one block: shared memory, and a
        block size known to launch (thread_cap; -1: _thread_cap(es))."""
        var cap = thread_cap if thread_cap > 0 else self._thread_cap(es)
        if k < LOW_MIN + REG_BITS or (1 << (k - REG_BITS)) > cap:
            return False
        comptime if not TILE_SMEM_DYNAMIC:  # static shared arrays: sized for the default tile
            if k > (tile_bits[DType.float32]() if es == 4 else tile_bits[DType.float64]()):
                return False
        var lim = self._smem_limit()
        return lim <= 0 or tile_smem_worst(es, k) <= lim

    def _pick_tile_bits(self, es: Int, tb0: Int, thread_cap: Int = -1) -> Int:
        """Largest k in tb0..tb0 + 2 whose worst-case run fits a block, where
        2 blocks of a typical run (tile_smem_typical) fit per multiprocessor
        and whose block size is known to launch (thread_cap, -1:
        _thread_cap(es)) with >= 2 resident blocks when the occupancy
        calculator was asked (kl.two_block_threads). tb0 when the shared
        memory is not reported; below tb0 only if tb0's block cannot launch
        (QUMO_GPU_REG_BITS < 3 builds)."""
        var cap = thread_cap if thread_cap > 0 else self._thread_cap(es)
        var best = tb0
        while best > LOW_MIN + REG_BITS and (1 << (best - REG_BITS)) > cap:
            best -= 1
        if best < tb0 or self._smem_limit() <= 0 or self.smem_per_multiprocessor <= 0:
            return best
        var two = self._kl(es).two_block_threads
        for k in range(tb0 + 1, tb0 + 3):
            var threads = 1 << (k - REG_BITS)
            if not self._tile_fits(es, k, cap) or 2 * tile_smem_typical(es, k) > self.smem_per_multiprocessor:
                break
            if thread_cap <= 0 and two > 0 and threads > two:
                break
            best = k
        return best

    def wants_kernel_limits(self, es: Int) -> Bool:
        """Would a tile larger than TILE_SAFE_THREADS threads be used if the
        compiled kernel admitted it (shared memory alone, or the
        QUMO_GPU_TILE_BITS override)? Then GpuStateVector reads the kernel's
        limits (TileKernelLimits) and rebuilds the profile with them."""
        var tb0 = tile_bits[DType.float32]() if es == 4 else tile_bits[DType.float64]()
        var cur = self._pick_tile_bits(es, tb0)
        if self._pick_tile_bits(es, tb0, 1024) > cur:
            return True
        var tb = _env_int("QUMO_GPU_TILE_BITS", LOW_MIN + REG_BITS, 16)
        return tb > 0 and (1 << (tb - REG_BITS)) > self._thread_cap(es) and self._tile_fits(es, tb, 1024)

    def _apply_env(mut self):
        var lb = _env_int("QUMO_GPU_LOW_BITS", LOW_MIN, LOW_MAX)
        if lb > 0:
            self.low_fp32 = lb
            self.low_fp64 = lb
            self.source = "env QUMO_GPU_LOW_BITS=" + String(lb)
        var dw = _env_int("QUMO_GPU_DIAG_WIDTH", 0, DIAG_MAX_WIDTH)
        if dw >= 0:
            self.diag_width = dw
            var t = "env QUMO_GPU_DIAG_WIDTH=" + String(dw)
            self.source = t if self.source == "defaults" else self.source + "; " + t
        var sp = _env_int("QUMO_GPU_SPLIT_1Q", 0, 1)
        if sp >= 0:
            self.split_1q_fp32 = sp == 1
            self.split_1q_fp64 = sp == 1
            var t = "env QUMO_GPU_SPLIT_1Q=" + String(sp)
            self.source = t if self.source == "defaults" else self.source + "; " + t
        var fl = _env_int("QUMO_GPU_FP64_LEAN", 0, 1)  # the fp64-only form of the same switch
        if fl >= 0:
            self.split_1q_fp64 = fl == 1
            var t = "env QUMO_GPU_FP64_LEAN=" + String(fl)
            self.source = t if self.source == "defaults" else self.source + "; " + t
        var tb = _env_int("QUMO_GPU_TILE_BITS", LOW_MIN + REG_BITS, 16)
        if tb > 0:
            var note = ""
            if self._tile_fits(4, tb):
                self.tile_bits_fp32 = tb
            else:
                note += " (fp32 ignored: does not fit)"
            if self._tile_fits(8, tb):
                self.tile_bits_fp64 = tb
            else:
                note += " (fp64 ignored: does not fit)"
            var t = "env QUMO_GPU_TILE_BITS=" + String(tb) + note
            self.source = t if self.source == "defaults" else self.source + "; " + t

    def tile_bits[dt: DType](self) -> Int:
        """log2 amplitudes per tile for precision dt on this device."""
        comptime if dt == DType.float64:
            return self.tile_bits_fp64
        else:
            return self.tile_bits_fp32

    def low[dt: DType](self) -> Int:
        """Always-resident low qubits of a tile for precision dt."""
        comptime if dt == DType.float64:
            return self.low_fp64
        else:
            return self.low_fp32

    def to_json(self) -> String:
        var w = JsonWriter()
        w.begin_object()
        w.kv_string("vendor", self.vendor)
        w.kv_string("name", self.name)
        w.kv_string("api", self.api)
        w.kv_string("arch", self.arch)
        w.kv_int("multiprocessors", self.multiprocessors)
        w.kv_int("smem_per_block", self.smem_per_block)
        w.kv_int("smem_per_block_optin", self.smem_per_block_optin)
        w.kv_int("smem_per_multiprocessor", self.smem_per_multiprocessor)
        w.kv_int("warp", self.warp)
        w.kv_int("low_bits_fp32", self.low_fp32)
        w.kv_int("low_bits_fp64", self.low_fp64)
        w.kv_int("tile_bits_fp32", self.tile_bits_fp32)
        w.kv_int("tile_bits_fp64", self.tile_bits_fp64)
        w.kv_int("tile_threads_fp32", 1 << (self.tile_bits_fp32 - REG_BITS))
        w.kv_int("tile_threads_fp64", 1 << (self.tile_bits_fp64 - REG_BITS))
        w.kv_int("tile_kernel_max_threads_fp32", self.kl_fp32.max_threads)
        w.kv_int("tile_kernel_max_threads_fp64", self.kl_fp64.max_threads)
        w.kv_int("tile_kernel_two_block_threads_fp32", self.kl_fp32.two_block_threads)
        w.kv_int("tile_kernel_two_block_threads_fp64", self.kl_fp64.two_block_threads)
        w.kv_int("tile_kernel_regs_fp32", self.kl_fp32.regs)
        w.kv_int("tile_kernel_regs_fp64", self.kl_fp64.regs)
        w.kv_int("tile_kernel_local_bytes_fp32", self.kl_fp32.local_bytes)
        w.kv_int("tile_kernel_local_bytes_fp64", self.kl_fp64.local_bytes)
        w.kv_int("tile_launch_bound", TILE_LAUNCH_BOUND)
        w.kv_int("diag_width", self.diag_width)
        w.kv_bool("split_1q_fp32", self.split_1q_fp32)
        w.kv_bool("split_1q_fp64", self.split_1q_fp64)
        w.kv_bool("weak_fp64", self.weak_fp64())
        w.kv_string("source", self.source)
        w.end_object()
        return w.finish()

    def write_to(self, mut w: Some[Writer]):
        w.write("GpuProfile(", self.vendor, " ", self.name, ", arch=", self.arch, ", SMs=", self.multiprocessors,
                ", smem/block=", self.smem_per_block, ", warp=", self.warp, ", low fp32/fp64=", self.low_fp32, "/",
                self.low_fp64, ", tile bits fp32/fp64=", self.tile_bits_fp32, "/", self.tile_bits_fp64, ", ",
                self.source, ")")


def _attr(ctx: DeviceContext, a: DeviceAttribute) -> Int:
    try:
        return ctx.get_attribute(a)
    except:
        return -1


def gpu_compiled() -> Bool:
    """True when this binary was built with `-D QUMO_GPU`."""
    return GPU_COMPILED


@always_inline
def device_kernels_enabled[dt: DType]() -> Bool:
    """Comptime: can kernels of precision `dt` be compiled for this target?"""
    return GPU_COMPILED and (dt == DType.float32 or (dt == DType.float64 and GPU_FP64_SUPPORTED))


def precision_name[dt: DType]() -> String:
    comptime if dt == DType.float32:
        return "fp32"
    elif dt == DType.float64:
        return "fp64"
    else:
        return String(dt)


def gpu_vendor() -> String:
    """Vendor of the compile-time target accelerator ('none' without one)."""
    comptime if has_apple_gpu_accelerator():
        return "apple"
    elif has_nvidia_gpu_accelerator():
        return "nvidia"
    elif has_amd_gpu_accelerator():
        return "amd"
    else:
        return "none"


def gpu_not_compiled_error() -> Error:
    return Error("QM-E470: GPU support is not compiled into this build. Hint: rebuild with '-D QUMO_GPU' (make cli-gpu / py-gpu / test-gpu); on macOS install Xcode and run 'xcodebuild -downloadComponent MetalToolchain' first.")


def gpu_fp64_error() -> Error:
    return Error("QM-E472: float64 is not supported on Apple GPUs (Metal has no double precision). Hint: use GpuStateVector[DType.float32] (the default on Apple).")


def gpu_available() raises -> Bool:
    """True when a GPU kernel can be launched. Never raises in practice: a
    failing DeviceContext creation means 'not available'. Without -D QUMO_GPU
    no DeviceContext is created at all."""
    comptime if GPU_COMPILED:
        try:
            var ctx = DeviceContext()
            _ = ctx.name()
            return True
        except:
            return False
    else:
        return False


struct GpuInfo(Copyable, Writable):
    var compiled: Bool
    var available: Bool
    var name: String
    var api: String
    var vendor: String
    var precision: String  # default precision of GpuStateVector
    var fp64_supported: Bool
    var min_qubits_auto: Int
    var error: String
    var profile: String  # GpuProfile JSON of the device ("{}" when none)

    def __init__(out self):
        self.compiled = GPU_COMPILED
        self.available = False
        self.name = ""
        self.api = ""
        self.vendor = gpu_vendor()
        self.precision = precision_name[GPU_DEFAULT_DTYPE]()
        self.fp64_supported = GPU_FP64_SUPPORTED
        self.min_qubits_auto = GPU_MIN_QUBITS
        self.error = ""
        self.profile = "{}"

    def to_json(self) -> String:
        var w = JsonWriter()
        w.begin_object()
        w.kv_bool("compiled", self.compiled)
        w.kv_bool("available", self.available)
        w.kv_string("name", self.name)
        w.kv_string("api", self.api)
        w.kv_string("vendor", self.vendor)
        w.kv_string("precision", self.precision)
        w.kv_bool("fp64_supported", self.fp64_supported)
        w.kv_int("min_qubits_auto", self.min_qubits_auto)
        w.kv_string("error", self.error)
        w.kv_raw("profile", self.profile)
        w.end_object()
        return w.finish()

    def write_to(self, mut w: Some[Writer]):
        if not self.compiled:
            w.write("GPU: not compiled (build with -D QUMO_GPU)")
            return
        if not self.available:
            w.write("GPU: compiled for ", self.vendor, ", no usable device (", self.error, ")")
            return
        w.write("GPU: ", self.name, " (api=", self.api, ", vendor=", self.vendor, ", precision=", self.precision, ", fp64=", self.fp64_supported, ")")


def query_gpu() -> GpuInfo:
    """Device facts for agents / users; creates a DeviceContext only when compiled."""
    var info = GpuInfo()
    comptime if GPU_COMPILED:
        try:
            var ctx = DeviceContext()
            info.name = ctx.name()
            info.api = ctx.api()
            info.available = True
            info.profile = GpuProfile.of(ctx).to_json()
        except e:
            info.error = String(e)
    else:
        info.error = "not compiled with -D QUMO_GPU"
    return info^


def gpu_info() -> String:
    """One-line human-readable GPU status (see `query_gpu()` for fields)."""
    return String(query_gpu())
