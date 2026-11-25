"""CPU simulator selection (explained): the cheapest exact CPU method for a circuit.

  * noisy Clifford circuits without feed-forward, many shots -> bit-packed
    Pauli-frame sampling (Stim's method: one reference run + 512 shots per
    SIMD batch);
  * other Clifford circuits (any width, incl. mid-circuit measurement, reset,
    feed-forward and Pauli noise) -> stabilizer tableau trajectories;
  * everything else -> state vector (static sampling or trajectories).

`choose_method_cpu` names the method, `sample_frame` and
`counts_from_measurements` turn Pauli-frame records into counts. The selection
that also knows the GPU (`sample_auto`, `sample_on`, `choose_method`) is
`qumo.gpu.dispatch`: `qumo.gpu` imports `qumo.sim`, so it cannot be here. Both
were `qumo/exec/dispatch.mojo`, which re-exports them for one release.
"""

from qumo.ir.circuit import Circuit
from qumo.ir.gates import Op
from qumo.sim.runner import sample, SampleResult
from qumo.sim.stabilizer import sample_stabilizer
from qumo.sim.frame import FrameSimulator, MeasurementSamples

comptime STABILIZER_MIN_QUBITS = 16


comptime FRAME_MIN_SHOTS = 256


def _has_conditions(c: Circuit) -> Bool:
    for g in c.ops:
        if g.cond >= 0:
            return True
    return False


def choose_method_cpu(c: Circuit, shots: Int = 1) -> String:
    """choose_method restricted to CPU simulators."""
    if c.num_params() == 0 and c.is_clifford() and not c.has_non_pauli_noise():
        if c.has_noise() and shots >= FRAME_MIN_SHOTS and not _has_conditions(c) and c.num_clbits <= 62:
            return "frame"
        if c.num_qubits >= STABILIZER_MIN_QUBITS or c.has_noise() or c.is_dynamic():
            return "stabilizer"
    return "statevector"


def sample_frame(c: Circuit, shots: Int, seed: UInt64) raises -> SampleResult:
    """Counts from the Pauli-frame sampler (measurement record -> clbit map)."""
    var fs = FrameSimulator(c)
    var ms = fs.sample_measurements(shots, seed)
    return counts_from_measurements(c, ms, shots, "stabilizer:pauli_frame")


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


def _check_measured(c: Circuit) raises:
    """QM-E402 for a circuit without measurements, whatever simulator the
    dispatch would pick (the tableau and Pauli-frame samplers would return the
    single empty outcome `{"": shots}`)."""
    if c.num_measurements == 0:
        raise Error("QM-E402: circuit has no measurements. Hint: call measure_all() or use evolve()/expectation().")


def _sample_cpu(c: Circuit, shots: Int, seed: UInt64, params: List[Float64]) raises -> SampleResult:
    if len(params) == 0:
        var m = choose_method_cpu(c, shots)
        if m == "frame" or m == "stabilizer":
            _check_measured(c)
        if m == "frame":
            return sample_frame(c, shots, seed)
        if m == "stabilizer":
            return sample_stabilizer(c, shots, seed)
    return sample(c, shots, seed, params)
