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

from std.math import sqrt
from std.time import perf_counter_ns
from std.runtime import parallelism_level
from max.algorithm import parallelize

from qumo.ir.circuit import Circuit
from qumo.ir.gates import Op
from qumo.ir.pauli import PauliSum, PauliTerm
from qumo.sim.statevector import StateVector
from qumo.sim.runner import evolve, evolve_into, sample, SampleResult
from qumo.sim.rng import Rng

comptime PAR_SETS_MAX_QUBITS = 14


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------
struct EstimatorPub(Copyable):
    var circuit: Circuit
    var observables: List[PauliSum]
    var parameter_sets: List[List[Float64]]  # empty => one set with no params
    var precision: Float64  # 0 => exact; else target standard error
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
    if len(params) != c.num_params():
        raise Error("QM-E101: PUB parameter set has " + String(len(params)) + " values but circuit has " + String(c.num_params()) + " parameters " + c.param_names_str() + ".")


def _basis_rotation(term_x: UInt64, term_z: UInt64, n: Int) -> Circuit:
    """Rotate so a QWC group's Paulis become Z: X -> H, Y -> Sdg H."""
    var c = Circuit(n)
    for q in range(n):
        var b = UInt64(1) << UInt64(q)
        if (term_x & b) != 0:
            if (term_z & b) != 0:
                c.sdg(q)
            c.h(q)
    return c^


def _estimate_shots(circuit: Circuit, h: PauliSum, params: List[Float64], shots: Int,
                    seed: UInt64, mut ev: Float64, mut std: Float64) raises:
    """Hardware-faithful estimation: one measurement setting per QWC group."""
    var n = circuit.num_qubits
    var base = StateVector(n)
    evolve_into(base, circuit, params)
    var groups = h.group_qwc()
    var rng = Rng(seed)
    ev = 0.0
    var var_total = 0.0
    for gi in range(len(groups)):
        ref g = groups[gi]
        # union basis for the group
        var gx: UInt64 = 0
        var gz: UInt64 = 0
        for ti in g:
            gx |= h.terms[ti].x
            gz |= h.terms[ti].z & h.terms[ti].x  # Y marker
        var sv = base.copy()
        var rot = _basis_rotation(gx, gz, n)
        evolve_into(sv, rot, List[Float64]())
        var grng = rng.split(gi)
        var outs = sv.sample(shots, grng)
        for ti in g:
            var t = h.terms[ti]
            var support = Int(t.x | t.z)
            if support == 0:
                ev += t.coeff
                continue
            var s = 0.0
            for o in outs:
                s += -1.0 if (pop_parity(o & support)) else 1.0
            var mean = s / Float64(shots)
            ev += t.coeff * mean
            var var1 = max(0.0, 1.0 - mean * mean)
            var_total += t.coeff * t.coeff * var1 / Float64(shots)
    std = sqrt(var_total)


@always_inline
def pop_parity(x: Int) -> Bool:
    var v = x
    var p = 0
    while v != 0:
        v &= v - 1
        p ^= 1
    return p == 1


def run_estimator(pub: EstimatorPub) raises -> EstimatorResult:
    """Run one Estimator PUB (all parameter sets x all observables)."""
    pub.circuit.validate()
    var nsets = pub.num_sets()
    var nobs = len(pub.observables)
    var exact = pub.precision <= 0.0
    var res = EstimatorResult(nsets, nobs, "statevector:exact" if exact else "statevector:shots+qwc")
    var shots = 0
    if not exact:
        shots = Int(1.0 / (pub.precision * pub.precision)) + 1
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
    var failed = List[Int](length=1, fill=0)
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
        except:
            fp[unsafe_offset=0] = 1

    if nsets > 1 and pub.circuit.num_qubits <= PAR_SETS_MAX_QUBITS:
        parallelize(one, nsets)
    else:
        for s in range(nsets):
            one(s)
    if failed[0] != 0:
        raise Error("QM-E403: estimator task failed for at least one parameter set.")
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
