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
The returned `SampleResult.method` records which path ran
("gpu:statevector:fp32" etc. for the GPU).
"""

from std.sys.defines import is_defined

from qumo.ir.circuit import Circuit
from qumo.ir.gates import Op
from qumo.sim.runner import sample, SampleResult
from qumo.sim.stabilizer import sample_stabilizer
from qumo.sim.frame import FrameSimulator
from qumo.gpu.device import GPU_MIN_QUBITS, gpu_available
from qumo.gpu.statevector_gpu import sample_gpu

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
    var rec_to_clbit = List[Int](length=c.num_measurements, fill=-1)
    for g in c.ops:
        if g.kind == Op.MEASURE:
            rec_to_clbit[g.q1] = g.clbit
    var res = SampleResult(c.num_clbits, shots, "stabilizer:pauli_frame")
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


def sample_auto(c: Circuit, shots: Int, seed: UInt64 = 1234,
                params: List[Float64] = List[Float64]()) raises -> SampleResult:
    """Sample with the method `choose_method` selects."""
    if len(params) == 0:
        var m = choose_method(c, shots)
        if m == "frame":
            return sample_frame(c, shots, seed)
        if m == "stabilizer":
            return sample_stabilizer(c, shots, seed)
        comptime if is_defined["QUMO_GPU"]():
            if m == "gpu":
                try:
                    return sample_gpu(c, shots, seed, params)
                except:
                    pass  # e.g. device out of memory: the CPU state vector decides
    else:
        comptime if is_defined["QUMO_GPU"]():
            if _gpu_ready(c):
                try:
                    return sample_gpu(c, shots, seed, params)
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
              device: String, mut note: String) raises -> SampleResult:
    """Sample on an explicit device: 'auto' (= sample_auto), 'cpu' (never the
    GPU) or 'gpu' (GPU state vector; falls back to the CPU and explains why in
    `note` when the GPU is not compiled/available or the circuit is dynamic or
    noisy)."""
    note = ""
    if device == "auto":
        return sample_auto(c, shots, seed, params)
    if device == "cpu":
        return _sample_cpu(c, shots, seed, params)
    if device != "gpu":
        raise Error("QM-E940: unknown device '" + device + "'. Hint: use 'auto', 'cpu' or 'gpu'.")
    comptime if is_defined["QUMO_GPU"]():
        if c.is_dynamic() or c.has_noise():
            note = "dynamic or noisy circuit: ran on the CPU (the GPU sampler is static-only)"
        else:
            var ok: Bool
            try:
                ok = gpu_available()
            except:
                ok = False
            if ok:
                return sample_gpu(c, shots, seed, params)
            note = "no usable GPU device: ran on the CPU"
    else:
        note = "GPU support not compiled (build with -D QUMO_GPU): ran on the CPU"
    return _sample_cpu(c, shots, seed, params)
