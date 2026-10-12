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

Precision policy (like CUDA-Q's nvidia target: fp32 default, fp64 option):
  * Apple GPUs (Metal) have no float64 -> float32, always.
  * NVIDIA / AMD -> float64 by default (bit-for-bit closer to the CPU
    simulator); `-D QUMO_GPU_FP32` makes float32 the default for speed /
    memory (2x the qubits per GiB). `GpuStateVector[DType.float32]` is always
    selectable explicitly.
"""

from std.sys.defines import is_defined
from std.sys import has_apple_gpu_accelerator, has_nvidia_gpu_accelerator, has_amd_gpu_accelerator
from max.gpu.host import DeviceContext

from qumo.ai.json import JsonWriter

comptime GPU_COMPILED = is_defined["QUMO_GPU"]()
comptime GPU_IS_APPLE = has_apple_gpu_accelerator()
comptime GPU_FP64_SUPPORTED = not has_apple_gpu_accelerator()
comptime GPU_DEFAULT_DTYPE = DType.float32 if (has_apple_gpu_accelerator() or is_defined["QUMO_GPU_FP32"]()) else DType.float64

# Automatic dispatch sends static non-Clifford circuits with at least this many
# qubits to the GPU. Below ~2^20 amplitudes (8 MiB fp32) a gate is a few
# microseconds of CPU time, comparable to one kernel launch, and the one-off
# DeviceContext creation is not amortised. Heuristic: not yet tuned on a GPU.
comptime GPU_MIN_QUBITS = 20


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
        except e:
            info.error = String(e)
    else:
        info.error = "not compiled with -D QUMO_GPU"
    return info^


def gpu_info() -> String:
    """One-line human-readable GPU status (see `query_gpu()` for fields)."""
    return String(query_gpu())
