"""Pre-trajectory sampling (the idea of CUDA-Q PTSBE, arXiv:2504.16297).

Per-shot trajectories simulate one state per shot, although at low noise most
shots repeat the same error pattern. For Pauli noise the chance of a pattern does
not depend on the state, so all patterns can be drawn first, identical ones
grouped, and each distinct pattern evolved once and sampled for its count. Same
distribution as `sample()`: the draws are the same i.i.d. process, only the
evolution is shared. Patterns also share their error-free prefix: the simulation
walks a trie of patterns, keeping one state per depth.

The gain shrinks as sites x p grows, where every shot differs (`TrajectoryPlan`
reports `unique_fraction`). `max_trajectories` keeps the most frequent patterns
and gives their weight to the shots it dropped, which is biased; `method` says so.

Static circuits only: a condition or reset makes the rest of the shot depend on
outcomes (QM-E409).
"""

from std.math import log
from std.collections import Dict
from qumo.ir.gates import Op, op_arity, op_is_noise, op_is_pauli_noise, op_is_annotation, op_is_unitary
from qumo.ir.circuit import Circuit, Instruction
from qumo.sim.statevector import StateVector
from qumo.sim.rng import Rng
from qumo.sim.runner import (
    SampleResult, KOp, build_plan, execute_plan, _apply_pauli, terminal_measure_map, TerminalMeasureMap,
    check_statevector_size,
)


@fieldwise_init
struct TrajectoryPlan(Copyable, Writable):
    var shots: Int
    var sites: Int  # noise instructions per shot
    var unique: Int  # distinct error patterns among the shots
    var error_free: Int  # shots with no error at all

    def unique_fraction(self) -> Float64:
        return Float64(self.unique) / Float64(max(self.shots, 1))

    def write_to(self, mut w: Some[Writer]):
        w.write("TrajectoryPlan(shots=", self.shots, ", sites=", self.sites, ", unique=", self.unique,
                ", error_free=", self.error_free, ")")


struct _Sites(Movable):
    """Noise instructions: total probability + cumulative outcome thresholds."""

    var qubit0: List[Int]
    var qubit1: List[Int]
    var total: List[Float64]
    var kinds: List[Int]
    var p1: List[Float64]
    var p2: List[Float64]
    var pmax: Float64

    def __init__(out self):
        self.qubit0 = List[Int]()
        self.qubit1 = List[Int]()
        self.total = List[Float64]()
        self.kinds = List[Int]()
        self.p1 = List[Float64]()
        self.p2 = List[Float64]()
        self.pmax = 0.0

    def add(mut self, g: Instruction, params: List[Float64]):
        var p = g.p0.resolve(params)
        var a = p
        var b = 0.0
        var t = p
        if g.kind == Op.PAULI_CHANNEL_1:
            a = p
            b = g.p1.resolve(params)
            t = a + b + g.p2.resolve(params)
        self.qubit0.append(g.q0)
        self.qubit1.append(g.q1)
        self.total.append(t)
        self.kinds.append(g.kind)
        self.p1.append(a)
        self.p2.append(b)
        self.pmax = max(self.pmax, t)

    def outcome(self, i: Int, mut rng: Rng) -> Int:
        """Which Pauli, given that site i fired: letters as in runner._apply_noise."""
        var k = self.kinds[i]
        if k == Op.X_ERROR:
            return 1
        if k == Op.Y_ERROR:
            return 2
        if k == Op.Z_ERROR:
            return 3
        if k == Op.DEPOLARIZE1:
            return 1 + rng.below(3)
        if k == Op.DEPOLARIZE2:
            return 1 + rng.below(15)
        var u = rng.uniform() * self.total[i]
        if u < self.p1[i]:
            return 1
        if u < self.p1[i] + self.p2[i]:
            return 2
        return 3


def _segments(c: Circuit, params: List[Float64], mut sites: _Sites, mut plans: List[List[KOp]]) raises:
    """Split at the noise instructions: plans[i] runs before site i, plans[K] last."""
    var seg = Circuit(c.num_qubits, 0, "segment")
    for g in c.ops:
        var k = g.kind
        if op_is_noise(k):
            if not op_is_pauli_noise(k):
                raise Error("QM-E409: sample_ptsbe groups Pauli patterns, whose probability does not depend on the state, and Kraus channels do. Hint: use sample(), which runs per-shot trajectories.")
            sites.add(g, params)
            plans.append(build_plan(seg, params))
            seg = Circuit(c.num_qubits, 0, "segment")
        elif op_is_unitary(k):
            seg.append(g)
    plans.append(build_plan(seg, params))


def _draw(sites: _Sites, shots: Int, mut rng: Rng, mut errs: List[List[Int]]):
    """One sorted error list per shot: site * 16 + Pauli code, by thinned geometric skips."""
    var K = len(sites.total)
    var pmax = sites.pmax
    var log_keep = log(1.0 - pmax) if pmax > 0.0 and pmax < 1.0 else 0.0
    for _ in range(shots):
        var e = List[Int]()
        if pmax > 0.0:
            var i = 0
            while True:
                if pmax < 1.0:
                    i += Int(log(1.0 - rng.uniform()) / log_keep)
                if i >= K:
                    break
                if rng.uniform() * pmax < sites.total[i]:
                    e.append(i * 16 + sites.outcome(i, rng))
                i += 1
        errs.append(e^)


def _key(e: List[Int]) -> String:
    var s = String("")
    for x in e:
        s += String(x) + ","
    return s^


struct _Walk(Movable):
    var plans: List[List[KOp]]
    var sites: _Sites
    var errs: List[List[Int]]
    var mm: TerminalMeasureMap
    var rng: Rng
    var leaf: Int

    def __init__(out self, var plans: List[List[KOp]], var sites: _Sites, var errs: List[List[Int]],
                 var mm: TerminalMeasureMap, seed: UInt64):
        self.plans = plans^
        self.sites = sites^
        self.errs = errs^
        self.mm = mm^
        self.rng = Rng(seed)
        self.leaf = 0

    def descend(mut self, mut sv: StateVector, var seg: Int, shots: List[Int], depth: Int,
                mut res: SampleResult) raises:
        """All `shots` agree on their first `depth` errors and are at segment `seg`."""
        var groups = Dict[Int, List[Int]]()
        var done = 0
        for s in shots:
            if len(self.errs[s]) == depth:
                done += 1
            else:
                var code = self.errs[s][depth]
                if code in groups:
                    groups[code].append(s)
                else:
                    groups[code] = [s]
        var codes = List[Int]()
        for kv in groups.items():
            codes.append(kv.key)
        sort(codes)
        for code in codes:
            var site = code // 16
            while seg <= site:
                execute_plan(sv, self.plans[seg])
                seg += 1
            var branch = sv.copy()
            var o = code % 16
            if self.sites.kinds[site] == Op.DEPOLARIZE2:
                _apply_pauli(branch, o & 3, self.sites.qubit0[site])
                _apply_pauli(branch, o >> 2, self.sites.qubit1[site])
            else:
                _apply_pauli(branch, o, self.sites.qubit0[site])
            self.descend(branch, seg, groups[code], depth + 1, res)
        if done > 0:
            while seg < len(self.plans):
                execute_plan(sv, self.plans[seg])
                seg += 1
            var leaf_rng = self.rng.split(self.leaf)
            self.leaf += 1
            for o in sv.sample(done, leaf_rng):
                res.record(self.mm.pack(o))


def _static_checks(c: Circuit, params: List[Float64]) raises:
    c.validate()
    check_statevector_size(c.num_qubits)
    if len(params) != c.num_params():
        raise Error("QM-E101: circuit has " + String(c.num_params()) + " parameters " + c.param_names_str() + " but " + String(len(params)) + " values were given.")
    if c.num_measurements == 0:
        raise Error("QM-E402: circuit has no measurements. Hint: call measure_all() or use evolve()/expectation().")
    if c.num_clbits > 62:
        raise Error("QM-E406: the sampler packs classical registers into 62 bits; circuit has " + String(c.num_clbits) + " clbits. Hint: measure fewer bits or split the register.")
    # noise right before a measurement is fine; anything else after one is not
    var measured = List[Bool](length=c.num_qubits, fill=False)
    for g in c.ops:
        if g.cond >= 0 or g.kind == Op.RESET:
            raise Error("QM-E409: sample_ptsbe needs a static circuit; '" + String(g) + "' depends on the shot. Hint: use sample(), which runs per-shot trajectories.")
        if g.kind == Op.MEASURE:
            measured[g.q0] = True
        elif op_is_unitary(g.kind) or op_is_noise(g.kind):
            for j in range(op_arity(g.kind)):
                if measured[g.qubit(j)]:
                    raise Error("QM-E409: sample_ptsbe needs terminal measurements; '" + String(g) + "' follows one. Hint: use sample(), which runs per-shot trajectories.")


def plan_trajectories(c: Circuit, shots: Int, seed: UInt64 = 1234,
                      params: List[Float64] = List[Float64]()) raises -> TrajectoryPlan:
    """What `sample_ptsbe` would run, without simulating: distinct patterns among `shots`."""
    _static_checks(c, params)
    var sites = _Sites()
    var plans = List[List[KOp]]()
    _segments(c, params, sites, plans)
    var rng = Rng(seed)
    var errs = List[List[Int]]()
    _draw(sites, shots, rng, errs)
    var seen = Dict[String, Int]()
    var clean = 0
    for e in errs:
        if len(e) == 0:
            clean += 1
        seen[_key(e)] = 1
    return TrajectoryPlan(shots, len(sites.total), len(seen), clean)


def sample_ptsbe(c: Circuit, shots: Int, seed: UInt64 = 1234, max_trajectories: Int = -1,
                 params: List[Float64] = List[Float64]()) raises -> SampleResult:
    """Sample a static noisy circuit with one simulation per distinct error pattern."""
    _static_checks(c, params)
    var sites = _Sites()
    var plans = List[List[KOp]]()
    _segments(c, params, sites, plans)
    var rng = Rng(seed)
    var errs = List[List[Int]]()
    _draw(sites, shots, rng, errs)
    var method = String("statevector:ptsbe")
    if max_trajectories > 0:
        # keep the most frequent patterns; dropped shots take a kept shot's pattern
        var count = Dict[String, Int]()
        for e in errs:
            var k = _key(e)
            count[k] = count.get(k, 0) + 1
        if len(count) > max_trajectories:
            method += "(capped, biased)"
            var freq = List[Int]()
            for kv in count.items():
                freq.append(kv.value)
            sort(freq)
            var cut = freq[len(freq) - max_trajectories]
            var kept = List[Int]()
            var dropped = List[Int]()
            var slots = 0
            for i in range(len(errs)):
                var k = _key(errs[i])
                if count[k] > cut or (count[k] == cut and slots < max_trajectories):
                    kept.append(i)
                else:
                    dropped.append(i)
                if count[k] == cut:
                    slots += 1
            if len(kept) == 0:
                kept.append(0)
            for i in dropped:
                errs[i] = errs[kept[rng.below(len(kept))]].copy()
    var all_shots = List[Int](capacity=shots)
    for s in range(shots):
        all_shots.append(s)
    var walk = _Walk(plans^, sites^, errs^, terminal_measure_map(c), seed ^ 0x5851F42D4C957F2D)
    var res = SampleResult(c.num_clbits, shots, method)
    var sv = StateVector(c.num_qubits)
    walk.descend(sv, 0, all_shots, 0, res)
    return res^
