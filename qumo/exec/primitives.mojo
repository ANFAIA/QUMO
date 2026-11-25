"""Execution primitives: Sampler and Estimator over PUBs.

Best-of synthesis:
  * Qiskit Primitives V2 "PUB" (Primitive Unified Bloc): a declarative unit
    `(circuit, observables, parameter_sets[, precision|shots])` with array
    broadcasting, so *what* to compute is separate from *how* it is batched.
  * Braket ProgramSet `product()` semantics: one PUB = circuits x params x
    observables, executed as ONE task (amortises compilation / dispatch).
  * CUDA-Q `observe`: exact expectation values on simulators (precision = 0),
    or shot-based estimation with qubit-wise-commuting measurement grouping
    (precision > 0) that reproduces what real hardware would return,
    including standard errors.
Parameter sets of one PUB run in parallel when circuits are small (the state
vector kernels are already multithreaded for large ones).
"""

from std.math import sqrt, ceil
from std.time import perf_counter_ns
from std.runtime import parallelism_level
from max.algorithm import parallelize

from qumo.ir.circuit import Circuit, check_finite_values
from qumo.ir.gates import Op
from qumo.ir.pauli import PauliSum, PauliTerm
from qumo.sim.statevector import StateVector
from qumo.sim.runner import evolve, evolve_into, sample, SampleResult, check_statevector_size, check_pure_state_circuit

comptime PAR_SETS_MAX_QUBITS = 14


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------
struct EstimatorPub(Copyable):
    var circuit: Circuit
    var observables: List[PauliSum]
    var parameter_sets: List[List[Float64]]  # empty => one set with no params
    var precision: Float64  # 0 => exact; else the standard error to stay under (see _shots_for_precision)
    var seed: UInt64

    def __init__(out self, circuit: Circuit, observables: List[PauliSum],
                 parameter_sets: List[List[Float64]] = List[List[Float64]](),
                 precision: Float64 = 0.0, seed: UInt64 = 1234):
        self.circuit = circuit.copy()
        self.observables = observables.copy()
        self.parameter_sets = parameter_sets.copy()
        self.precision = precision
        self.seed = seed

    @staticmethod
    def single(circuit: Circuit, observable: PauliSum, precision: Float64 = 0.0) -> EstimatorPub:
        var obs = List[PauliSum]()
        obs.append(observable.copy())
        return EstimatorPub(circuit, obs, List[List[Float64]](), precision)

    def num_sets(self) -> Int:
        return max(1, len(self.parameter_sets))


struct EstimatorResult(Copyable, Writable):
    var evs: List[List[Float64]]  # [parameter_set][observable]
    var stds: List[List[Float64]]
    var method: String
    var shots_per_set: Int
    var seconds: Float64

    def __init__(out self, nsets: Int, nobs: Int, method: String):
        self.evs = List[List[Float64]]()
        self.stds = List[List[Float64]]()
        for _ in range(nsets):
            self.evs.append(List[Float64](length=nobs, fill=0.0))
            self.stds.append(List[Float64](length=nobs, fill=0.0))
        self.method = method
        self.shots_per_set = 0
        self.seconds = 0.0

    def write_to(self, mut w: Some[Writer]):
        w.write("EstimatorResult(method=", self.method, ", seconds=", self.seconds, ", evs=[")
        for i in range(len(self.evs)):
            if i > 0: w.write(", ")
            w.write("[")
            for j in range(len(self.evs[i])):
                if j > 0: w.write(", ")
                w.write(self.evs[i][j])
                if self.stds[i][j] > 0:
                    w.write(" ± ", self.stds[i][j])
            w.write("]")
        w.write("])")


def _check_params(c: Circuit, params: List[Float64]) raises:
    check_finite_values(params)
    if len(params) != c.num_params():
        raise Error("QM-E101: PUB parameter set has " + String(len(params)) + " values but circuit has " + String(c.num_params()) + " parameters " + c.param_names_str() + ".")


def qwc_setting(circuit: Circuit, h: PauliSum, grp: List[Int], mut base: Int) -> Circuit:
    """The circuit that measures one QWC group of `h`: the basis rotation, then every
    qubit q into clbit `base + q`. A circuit whose measurements are all terminal
    loses them (base = 0). A dynamic one (reset, mid-circuit measurement, classical
    control) keeps every operation and gets the new bits after its own."""
    var n = circuit.num_qubits
    var gx: UInt64 = 0
    var gy: UInt64 = 0
    for ti in grp:
        gx |= h.terms[ti].x
        gy |= h.terms[ti].x & h.terms[ti].z  # Y marker
    var m: Circuit
    if circuit.is_dynamic():
        m = circuit.copy()
        base = circuit.num_clbits
    else:
        m = circuit.empty_like()
        for g in circuit.ops:
            if g.kind != Op.MEASURE:
                m.append(g)
        m.num_clbits = 0
        base = 0
    for q in range(n):
        var b = UInt64(1) << UInt64(q)
        if (gx & b) != 0:
            if (gy & b) != 0:
                m.sdg(q)
            m.h(q)
    for q in range(n):
        m.measure(q, base + q)
    return m^


def qwc_group_stats(res: SampleResult, h: PauliSum, grp: List[Int], base: Int) -> Tuple[Float64, Float64]:
    """(mean, variance of the mean) of sum_t coeff_t <P_t> over the group's non-identity
    terms. All terms are read from the SAME shots, so the variance is that of the
    per-shot weighted sum, not the sum of the per-term variances: they are correlated
    (Z0 - Z1 on a Bell pair is 0 on every shot)."""
    var shots = len(res.memory)
    var s1 = 0.0
    var s2 = 0.0
    for o in res.memory:
        var v = 0.0
        for ti in grp:
            var t = h.terms[ti]
            var support = Int(t.x | t.z)
            if support == 0:
                continue
            v += -t.coeff if pop_parity((o >> base) & support) else t.coeff
        s1 += v
        s2 += v * v
    var mean = s1 / Float64(shots)
    var var1 = max(0.0, s2 / Float64(shots) - mean * mean)
    return (mean, var1 / Float64(shots))


def _estimate_shots(circuit: Circuit, h: PauliSum, params: List[Float64], shots: Int,
                    seed: UInt64, mut ev: Float64, mut std: Float64) raises:
    """Hardware-faithful estimation: one measurement setting per QWC group, sampled
    with the real sampler, so reset, noise and mid-circuit measurement count."""
    var groups = h.group_qwc()
    ev = 0.0
    var var_total = 0.0
    for gi in range(len(groups)):
        ref g = groups[gi]
        var has_support = False
        for ti in g:
            var t = h.terms[ti]
            if (t.x | t.z) == 0:
                ev += t.coeff
            else:
                has_support = True
        if not has_support:
            continue
        var base = 0
        var m = qwc_setting(circuit, h, g, base)
        var res = sample(m, shots, seed + UInt64(gi) * 1000003, params)
        var st = qwc_group_stats(res, h, g, base)
        ev += st[0]
        var_total += st[1]
    std = sqrt(var_total)


@always_inline
def pop_parity(x: Int) -> Bool:
    var v = x
    var p = 0
    while v != 0:
        v &= v - 1
        p ^= 1
    return p == 1


comptime MAX_ESTIMATOR_SHOTS = 10_000_000


def _shots_for_precision(observables: List[PauliSum], precision: Float64) raises -> Int:
    """Shots per measurement setting so that every observable's standard error is at
    most `precision`. A group's per-shot value is bounded by L = sum |coeff| of its
    non-identity terms, so its variance is at most L^2 / shots, and the observable's
    standard error is at most sqrt(sum_g L_g^2 / shots). Hence
    shots = ceil(max over observables of sum_g L_g^2 / precision^2) (at least 1).
    It is a bound, the reported std is usually smaller. QM-E913 past
    MAX_ESTIMATOR_SHOTS."""
    if not (precision > 0.0 and precision < Float64.MAX):
        raise Error("QM-E913: argument 'precision' = " + String(precision) + " is out of range. Hint: pass a finite standard error > 0, or 0 for the exact value.")
    var worst = 0.0
    for o in range(len(observables)):
        var groups = observables[o].group_qwc()
        var total = 0.0
        for gi in range(len(groups)):
            var l1 = 0.0
            for ti in groups[gi]:
                var t = observables[o].terms[ti]
                if (t.x | t.z) != 0:
                    l1 += abs(t.coeff)
            total += l1 * l1
        worst = max(worst, total)
    var need = ceil(worst / (precision * precision))
    if need > Float64(MAX_ESTIMATOR_SHOTS):
        raise Error("QM-E913: precision " + String(precision) + " needs " + String(need) + " shots per setting for these observables (limit " + String(MAX_ESTIMATOR_SHOTS) + "). Hint: raise precision, scale the observable down, or use precision=0 for the exact value.")
    return max(1, Int(need))


def run_estimator(pub: EstimatorPub) raises -> EstimatorResult:
    """Run one Estimator PUB (all parameter sets x all observables)."""
    pub.circuit.validate()
    check_statevector_size(pub.circuit.num_qubits)
    var nsets = pub.num_sets()
    var nobs = len(pub.observables)
    var exact = pub.precision <= 0.0
    if exact:
        # The shot path samples every operation, so only the exact path drops them.
        check_pure_state_circuit(pub.circuit, "run_estimator()")
    var res = EstimatorResult(nsets, nobs, "statevector:exact" if exact else "statevector:shots+qwc")
    for o in range(nobs):
        if pub.observables[o].num_qubits() > pub.circuit.num_qubits:
            raise Error("QM-E405: observable " + String(o) + " acts on qubit " + String(pub.observables[o].num_qubits() - 1) + " but the circuit has " + String(pub.circuit.num_qubits) + " qubits. Hint: check Pauli indices (0-based, qubit 0 = least significant).")
    var shots = 0
    if not exact:
        shots = _shots_for_precision(pub.observables, pub.precision)
        res.shots_per_set = shots
    for s in range(nsets):
        if len(pub.parameter_sets) > 0:
            _check_params(pub.circuit, pub.parameter_sets[s])
        elif pub.circuit.num_params() > 0:
            raise Error("QM-E102: circuit has parameters " + pub.circuit.param_names_str() + " but the PUB has no parameter_sets. Hint: pass parameter_sets=[[...]].")
    var t0 = perf_counter_ns()
    var ev_ptr = List[Float64](length=nsets * nobs, fill=0.0)
    var sd_ptr = List[Float64](length=nsets * nobs, fill=0.0)
    var evp = ev_ptr.unsafe_ptr().as_unsafe_any_origin()
    var sdp = sd_ptr.unsafe_ptr().as_unsafe_any_origin()
    var failed = List[String](length=nsets, fill=String(""))
    var fp = failed.unsafe_ptr().as_unsafe_any_origin()
    var circ_ptr = Pointer(to=pub.circuit).as_unsafe_any_origin()
    var obs_ptr = Pointer(to=pub.observables).as_unsafe_any_origin()
    var sets_ptr = Pointer(to=pub.parameter_sets).as_unsafe_any_origin()
    var seed = pub.seed

    def one(s: Int) {imm evp, imm sdp, imm fp, imm circ_ptr, imm obs_ptr, imm sets_ptr, imm nobs, imm exact, imm shots, imm seed}:
        try:
            var params = List[Float64]()
            if len(sets_ptr[]) > 0:
                params = sets_ptr[][s].copy()
            if exact:
                var sv = StateVector(circ_ptr[].num_qubits)
                evolve_into(sv, circ_ptr[], params)
                for o in range(nobs):
                    evp[unsafe_offset=s * nobs + o] = sv.expectation(obs_ptr[][o])
            else:
                for o in range(nobs):
                    var ev = 0.0
                    var sd = 0.0
                    _estimate_shots(circ_ptr[], obs_ptr[][o], params, shots, seed + UInt64(s * 7919 + o), ev, sd)
                    evp[unsafe_offset=s * nobs + o] = ev
                    sdp[unsafe_offset=s * nobs + o] = sd
        except e:
            fp[unsafe_offset=s] = String(e)

    if nsets > 1 and pub.circuit.num_qubits <= PAR_SETS_MAX_QUBITS:
        parallelize(one, nsets)
    else:
        for s in range(nsets):
            one(s)
    for s in range(nsets):
        if failed[s].byte_length() > 0:
            raise Error("QM-E403: estimator task failed for parameter set " + String(s) + ": " + failed[s])
    for s in range(nsets):
        for o in range(nobs):
            res.evs[s][o] = ev_ptr[s * nobs + o]
            res.stds[s][o] = sd_ptr[s * nobs + o]
    res.seconds = Float64(perf_counter_ns() - t0) / 1e9
    return res^


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------
struct SamplerPub(Copyable):
    var circuit: Circuit
    var parameter_sets: List[List[Float64]]
    var shots: Int
    var seed: UInt64

    def __init__(out self, circuit: Circuit, shots: Int = 1024,
                 parameter_sets: List[List[Float64]] = List[List[Float64]](), seed: UInt64 = 1234):
        self.circuit = circuit.copy()
        self.parameter_sets = parameter_sets.copy()
        self.shots = shots
        self.seed = seed

    def num_sets(self) -> Int:
        return max(1, len(self.parameter_sets))


def run_sampler(pub: SamplerPub) raises -> List[SampleResult]:
    """Run one Sampler PUB: one SampleResult per parameter set (seeded per set)."""
    var out = List[SampleResult]()
    for s in range(pub.num_sets()):
        if len(pub.parameter_sets) > 0:
            _check_params(pub.circuit, pub.parameter_sets[s])
            out.append(sample(pub.circuit, pub.shots, pub.seed + UInt64(s), pub.parameter_sets[s]))
        else:
            out.append(sample(pub.circuit, pub.shots, pub.seed + UInt64(s)))
    return out^


# ---------------------------------------------------------------------------
# Splitting a PUB for a device limit
# ---------------------------------------------------------------------------
@fieldwise_init
struct PubSlot(ImplicitlyCopyable, Writable):
    """Where one parameter set of the original PUB went."""

    var pub: Int
    var index: Int

    def write_to(self, mut w: Some[Writer]):
        w.write("(", self.pub, ", ", self.index, ")")


def _check_split(max_sets: Int) raises:
    if max_sets < 1:
        raise Error("QM-E420: split needs max_sets >= 1, got " + String(max_sets) + ". Hint: pass the device limit on parameter sets per job.")


def split_pub(pub: SamplerPub, max_sets: Int, mut slots: List[PubSlot]) raises -> List[SamplerPub]:
    """Split into PUBs of at most `max_sets` parameter sets (ProgramSet.split in
    Braket). Results stay bit-identical: run_sampler seeds set s with seed + s, and
    each part starts from the seed of its first original index. `slots[s]` says
    where original set s went."""
    _check_split(max_sets)
    var out = List[SamplerPub]()
    var total = pub.num_sets()
    var start = 0
    slots.clear()
    while start < total:
        var stop = min(start + max_sets, total)
        var sets = List[List[Float64]]()
        if len(pub.parameter_sets) > 0:
            for s in range(start, stop):
                sets.append(pub.parameter_sets[s].copy())
        for s in range(start, stop):
            slots.append(PubSlot(len(out), s - start))
        out.append(SamplerPub(pub.circuit, pub.shots, sets, pub.seed + UInt64(start)))
        start = stop
    return out^


def split_pub(pub: EstimatorPub, max_sets: Int, mut slots: List[PubSlot]) raises -> List[EstimatorPub]:
    """As for SamplerPub. Shot estimation seeds set s with seed + 7919 s + o, so a
    part starts at seed + 7919 start and matches the unsplit run bit for bit."""
    _check_split(max_sets)
    var out = List[EstimatorPub]()
    var total = pub.num_sets()
    var start = 0
    slots.clear()
    while start < total:
        var stop = min(start + max_sets, total)
        var sets = List[List[Float64]]()
        if len(pub.parameter_sets) > 0:
            for s in range(start, stop):
                sets.append(pub.parameter_sets[s].copy())
        for s in range(start, stop):
            slots.append(PubSlot(len(out), s - start))
        out.append(EstimatorPub(pub.circuit, pub.observables, sets, pub.precision, pub.seed + UInt64(start * 7919)))
        start = stop
    return out^
