"""Automatic simulator selection (explained).

Picks the cheapest exact method for a circuit, like Qibo/QDK's Clifford
backends or CUDA-Q's target choice, but per call and without global state:
  * noisy Clifford circuits without feed-forward, many shots -> bit-packed
    Pauli-frame sampling (Stim's method: one reference run + 512 shots per
    SIMD batch);
  * other Clifford circuits (any width, incl. mid-circuit measurement, reset,
    feed-forward and Pauli noise) -> stabilizer tableau trajectories;
  * static non-Clifford circuits with >= GPU_MIN_QUBITS qubits -> the GPU
    state vector (`qumo.gpu`), only in builds made with `-D QUMO_GPU` and
    when a device is usable; any GPU failure falls back to the CPU;
  * everything else -> state vector (static sampling or trajectories).
An explicit `device="gpu"` (`sample_on`) routes with `gpu_sample_route`:
Clifford circuits without feed-forward (noisy or not) -> GPU Pauli frames
(`qumo.gpu.frame_gpu`); other static, noiseless circuits -> GPU state vector;
everything else runs on the CPU and the returned note says exactly why
(no GPU build / device, > 62 classical bits, Clifford feed-forward, dynamic or
noisy non-Clifford). The returned `SampleResult.method` records which path
ran ("gpu:statevector:fp32", "gpu:statevector:fp64", "gpu:pauli_frame", ...).

GPU precision (`precision` argument of `sample_on` / `sample_auto`): "auto"
(default: QUMO_GPU_PRECISION when set, else the build default, fp32 on every
GPU), "fp32" or "fp64" (`qumo.gpu.device.resolve_gpu_precision`). It applies
to GPU state-vector runs only: the CPU simulators are fp64, the stabilizer and
Pauli-frame samplers exact. An unknown selector raises QM-E941. With
`device="gpu"`, "fp64" on an Apple GPU raises QM-E472 (Metal has no double
precision); automatic dispatch instead keeps such a circuit on the CPU (fp64).
"""

from std.sys.defines import is_defined

from qumo.ir.circuit import Circuit
from qumo.ir.gates import Op
from qumo.sim.runner import sample, SampleResult
from qumo.sim.stabilizer import sample_stabilizer
from qumo.sim.frame import FrameSimulator, MeasurementSamples
from qumo.gpu.device import (
    GPU_MIN_QUBITS, gpu_available, parse_gpu_precision, resolve_gpu_precision, gpu_precision_available,
    check_gpu_precision,
)
from qumo.gpu.statevector_gpu import sample_gpu_runtime
from qumo.gpu.frame_gpu import GpuFrameSampler

comptime STABILIZER_MIN_QUBITS = 16


comptime FRAME_MIN_SHOTS = 256


def _has_conditions(c: Circuit) -> Bool:
    for g in c.ops:
        if g.cond >= 0:
            return True
    return False


def gpu_eligible(c: Circuit) -> Bool:
    """Static, noiseless, non-Clifford (or parametric) and wide enough for the
    GPU to pay off. Pure circuit test: does not look for a device."""
    if c.num_qubits < GPU_MIN_QUBITS or c.num_clbits > 62:
        return False
    if c.num_params() == 0 and c.is_clifford():
        return False
    return not c.is_dynamic() and not c.has_noise()


def _gpu_ready(c: Circuit) -> Bool:
    """gpu_eligible(c) and a usable device (always False without -D QUMO_GPU)."""
    comptime if is_defined["QUMO_GPU"]():
        if gpu_eligible(c):
            try:
                return gpu_available()
            except:
                return False
    return False


def choose_method_cpu(c: Circuit, shots: Int = 1) -> String:
    """choose_method restricted to CPU simulators."""
    if c.num_params() == 0 and c.is_clifford():
        if c.has_noise() and shots >= FRAME_MIN_SHOTS and not _has_conditions(c) and c.num_clbits <= 62:
            return "frame"
        if c.num_qubits >= STABILIZER_MIN_QUBITS or c.has_noise() or c.is_dynamic():
            return "stabilizer"
    return "statevector"


def choose_method(c: Circuit, shots: Int = 1) -> String:
    var m = choose_method_cpu(c, shots)
    if m == "statevector" and _gpu_ready(c):
        return "gpu"
    return m


def sample_frame(c: Circuit, shots: Int, seed: UInt64) raises -> SampleResult:
    """Counts from the Pauli-frame sampler (measurement record -> clbit map)."""
    var fs = FrameSimulator(c)
    var ms = fs.sample_measurements(shots, seed)
    return counts_from_measurements(c, ms, shots, "stabilizer:pauli_frame")


def frame_eligible_gpu(c: Circuit) -> Bool:
    """Clifford, measured, no parameters and no classical conditions: the GPU
    Pauli-frame sampler handles it (noise optional)."""
    return c.num_params() == 0 and c.num_measurements > 0 and c.num_clbits <= 62 and c.is_clifford() and not _has_conditions(c)


def gpu_sample_route(c: Circuit, params: List[Float64]) -> Tuple[String, String]:
    """Where `device="gpu"` sampling of `c` runs, ignoring device availability:
    ("frames", ""), ("statevector", "") or ("cpu", reason). The reason is
    phrased to follow "device 'gpu' requested but " (QM-W020 / fallback notes)."""
    if c.num_clbits > 62:
        return ("cpu", String("the circuit has more than 62 classical bits (the GPU samplers pack registers into 62 bits): ran on the CPU"))
    if len(params) == 0 and frame_eligible_gpu(c):
        return ("frames", String(""))
    if c.num_params() == 0 and c.is_clifford() and _has_conditions(c):
        return ("cpu", String("the circuit is Clifford with classically-conditioned gates (feed-forward), which the GPU Pauli-frame sampler does not run: ran on the CPU stabilizer simulator"))
    if c.is_dynamic() or c.has_noise():
        return ("cpu", String("the circuit is dynamic (mid-circuit measurement, reset or feed-forward) or noisy and not Clifford, and the GPU state-vector sampler runs static noiseless circuits only: ran on the CPU"))
    return ("statevector", String(""))


def sample_frame_gpu(c: Circuit, shots: Int, seed: UInt64) raises -> SampleResult:
    """Counts from the GPU Pauli-frame sampler (needs -D QUMO_GPU and a device;
    QM-E470 otherwise). SampleResult.method = 'gpu:pauli_frame'."""
    var gs = GpuFrameSampler(c)
    var ms = gs.sample_measurements(shots, seed)
    return counts_from_measurements(c, ms, shots, gs.method_name())


def counts_from_measurements(c: Circuit, ms: MeasurementSamples, shots: Int, method: String) -> SampleResult:
    """Bit-packed measurement records -> classical-register counts."""
    var rec_to_clbit = List[Int](length=c.num_measurements, fill=-1)
    for g in c.ops:
        if g.kind == Op.MEASURE:
            rec_to_clbit[g.q1] = g.clbit
    var res = SampleResult(c.num_clbits, shots, method)
    for s in range(shots):
        var creg = 0
        for m in range(c.num_measurements):  # later records overwrite earlier ones
            var cb = rec_to_clbit[m]
            if cb < 0:
                continue
            if ms.get(s, m):
                creg |= 1 << cb
            else:
                creg &= ~(1 << cb)
        res.record(creg)
    return res^


def _auto_gpu_precision(precision: String) -> String:
    """The precision automatic dispatch may run a GPU state vector in, or ""
    when it must stay on the CPU: the selected precision is not available on
    this GPU (fp64 on Apple), or QUMO_GPU_PRECISION is invalid (explicit GPU
    requests raise QM-E941 for it; qumo_gpu_info reports it)."""
    try:
        var p = resolve_gpu_precision(precision)
        return p if gpu_precision_available(p) else String("")
    except:
        return ""


def sample_auto(c: Circuit, shots: Int, seed: UInt64 = 1234,
                params: List[Float64] = List[Float64](), precision: String = "auto") raises -> SampleResult:
    """Sample with the method `choose_method` selects. `precision` ("auto" |
    "fp32" | "fp64") is the precision of a GPU state-vector run; the GPU is
    used only when that precision is available there (fp64 on an Apple GPU
    stays on the fp64 CPU simulator). QM-E941 for an unknown selector."""
    _ = parse_gpu_precision(precision)
    if len(params) == 0:
        var m = choose_method(c, shots)
        if m == "frame":
            return sample_frame(c, shots, seed)
        if m == "stabilizer":
            return sample_stabilizer(c, shots, seed)
        comptime if is_defined["QUMO_GPU"]():
            if m == "gpu":
                var p = _auto_gpu_precision(precision)
                if p.byte_length() > 0:
                    try:
                        return sample_gpu_runtime(c, shots, seed, params, p)
                    except:
                        pass  # e.g. device out of memory: the CPU state vector decides
    else:
        comptime if is_defined["QUMO_GPU"]():
            if _gpu_ready(c):
                var p = _auto_gpu_precision(precision)
                if p.byte_length() > 0:
                    try:
                        return sample_gpu_runtime(c, shots, seed, params, p)
                    except:
                        pass
    return sample(c, shots, seed, params)


def _sample_cpu(c: Circuit, shots: Int, seed: UInt64, params: List[Float64]) raises -> SampleResult:
    if len(params) == 0:
        var m = choose_method_cpu(c, shots)
        if m == "frame":
            return sample_frame(c, shots, seed)
        if m == "stabilizer":
            return sample_stabilizer(c, shots, seed)
    return sample(c, shots, seed, params)


def sample_on(c: Circuit, shots: Int, seed: UInt64, params: List[Float64],
              device: String, mut note: String, precision: String = "auto") raises -> SampleResult:
    """Sample on an explicit device: 'auto' (= sample_auto), 'cpu' (never the
    GPU) or 'gpu' (routed by `gpu_sample_route`: Clifford circuits without
    feed-forward -> GPU Pauli frames, other static noiseless circuits -> GPU
    state vector in `precision`). When the GPU cannot run it (not compiled,
    no usable device, or the route is "cpu") the call runs on the CPU and
    `note` says why.

    `precision`: "auto" (QUMO_GPU_PRECISION when set, else the build
    default: fp32) | "fp32" | "fp64", for GPU state-vector runs (CPU
    simulators are fp64, Pauli frames exact). QM-E941 for an unknown value
    (argument or environment variable); QM-E472 when the GPU state vector
    would run in fp64 on an Apple GPU."""
    note = ""
    if device != "auto" and device != "cpu" and device != "gpu":
        raise Error("QM-E940: unknown device '" + device + "'. Hint: use 'auto', 'cpu' or 'gpu'.")
    _ = parse_gpu_precision(precision)
    if device == "auto":
        return sample_auto(c, shots, seed, params, precision)
    if device == "cpu":
        return _sample_cpu(c, shots, seed, params)
    comptime if is_defined["QUMO_GPU"]():
        var ok: Bool
        try:
            ok = gpu_available()
        except:
            ok = False
        if not ok:
            note = "no usable GPU device: ran on the CPU"
        else:
            var route = gpu_sample_route(c, params)
            if route[0] == "frames":
                return sample_frame_gpu(c, shots, seed)
            if route[0] == "statevector":
                var p = resolve_gpu_precision(precision)
                check_gpu_precision(p)
                return sample_gpu_runtime(c, shots, seed, params, p)
            note = route[1]
    else:
        note = "GPU support not compiled (build with -D QUMO_GPU): ran on the CPU"
    return _sample_cpu(c, shots, seed, params)
