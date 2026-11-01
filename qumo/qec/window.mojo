"""Sliding-window decoding over any `Decoder`.

A memory experiment of thousands of rounds cannot wait for its last round: a
window of W rounds is decoded, only the corrections of its first C rounds (the
commit region) are kept, and the next window starts C rounds later, with the
syndrome changed by what was committed (Dennis et al., quant-ph/0110143; the
rule (rounds - window) % step == 0 is cudaq-qec's).

The `Decoder` trait returns only an observable mask, which says nothing about
which edges a decoder chose, so the window asks it through that mask. Each
window gets its own detector error model in which an error of the commit region
also flips one *virtual observable* per detector it touches after that region.
The mask then is exactly the part of the correction that crosses into the future.
With more such detectors than free mask bits, a window has several decoders, one
per chunk of virtual observables, all fed the same syndrome; this relies on the
decoder being deterministic, as every QUMO decoder is.

window >= rounds is the plain decoder on the plain model.
"""

from std.collections import Dict
from qumo.qec.dem import DetectorErrorModel, DemError
from qumo.qec.decoders import Decoder, MatchingGraph, graphlike_dem


@fieldwise_init
struct _Window(Copyable):
    var first: Int  # first of the window's entries in plan.decoders
    var chunks: Int
    var detectors: List[Int]  # local detector index -> global
    var future: List[Int]  # virtual observable index -> global detector
    var per_chunk: Int


@fieldwise_init
struct WindowPlan(Copyable):
    """Sub-models to build decoders from: one per window chunk, in order."""

    var dems: List[DetectorErrorModel]
    var windows: List[_Window]
    var num_detectors: Int
    var num_observables: Int


def detector_rounds(dem: DetectorErrorModel) raises -> List[Int]:
    """Round of each detector, from the t coordinate of the model."""
    if len(dem.detector_coords) < 3 * dem.num_detectors:
        raise Error("QM-E736: this detector error model has no detector coordinates. Hint: pass the round of each detector to plan_windows.")
    var out = List[Int](capacity=dem.num_detectors)
    for d in range(dem.num_detectors):
        out.append(Int(dem.detector_coords[3 * d + 2]))
    return out^


def _graph_pieces(dem: DetectorErrorModel) -> List[DemError]:
    """Every error as graph-like pieces of at most two detectors."""
    var out = List[DemError]()
    var g = graphlike_dem(dem)
    for e in g.errors:
        if len(e.comp_dets) > 1:
            for c in range(len(e.comp_dets)):
                out.append(DemError(e.p, e.comp_dets[c].copy(), e.comp_obs[c]))
        elif len(e.comp_dets) == 1 or len(e.dets) <= 2:
            out.append(DemError(e.p, e.dets.copy(), e.obs))
    return out^


def plan_windows(dem: DetectorErrorModel, rounds_of: List[Int], window: Int, step: Int) raises -> WindowPlan:
    """Split `dem` into overlapping windows; `rounds_of[d]` is the round of detector d."""
    if len(rounds_of) != dem.num_detectors:
        raise Error("QM-E736: rounds_of has " + String(len(rounds_of)) + " entries for " + String(dem.num_detectors) + " detectors. Hint: one round per detector.")
    var rounds = 0
    for r in rounds_of:
        rounds = max(rounds, r + 1)
    if step <= 0 or window < step or (window < rounds and (rounds - window) % step != 0):
        raise Error("QM-E736: window " + String(window) + ", step " + String(step) + " for " + String(rounds) + " rounds. Hint: 0 < step <= window and (rounds - window) % step == 0.")
    var plan = WindowPlan(List[DetectorErrorModel](), List[_Window](), dem.num_detectors, dem.num_observables)
    if window >= rounds:
        var w = _Window(0, 1, List[Int](), List[Int](), 0)
        plan.dems.append(dem.copy())
        plan.windows.append(w^)
        return plan^
    var pieces = _graph_pieces(dem)
    var K = dem.num_observables
    var free = 64 - K
    if free <= 0:
        raise Error("QM-E736: no observable bits left for the window state (" + String(K) + " observables). Hint: decode with at most 63 observables.")
    var start = 0
    while True:
        var last = start + window >= rounds
        var commit_end = rounds if last else start + step
        var stop = rounds if last else start + window
        # local detectors
        var local = Dict[Int, Int]()
        var w = _Window(len(plan.dems), 1, List[Int](), List[Int](), free)
        for d in range(dem.num_detectors):
            if rounds_of[d] >= start and rounds_of[d] < stop:
                local[d] = len(w.detectors)
                w.detectors.append(d)
        # virtual observables: detectors after the commit region touched by a committed error
        var future_index = Dict[Int, Int]()
        for e in pieces:
            var commit = False
            var settled = False
            for d in e.dets:
                commit = commit or (rounds_of[d] >= start and rounds_of[d] < commit_end)
                settled = settled or rounds_of[d] < start
            if not commit or settled:
                continue
            for d in e.dets:
                if rounds_of[d] >= commit_end and d not in future_index:
                    future_index[d] = len(w.future)
                    w.future.append(d)
        w.chunks = max(1, (len(w.future) + free - 1) // free)
        for chunk in range(w.chunks):
            var sub = DetectorErrorModel(len(w.detectors), 64 if w.chunks > 0 and len(w.future) > 0 else K)
            for e in pieces:
                var dets = List[Int]()
                var committed = False
                var settled = False  # touches a round already decided: it cannot have happened
                for d in e.dets:
                    settled = settled or rounds_of[d] < start
                    var at = local.get(d, -1)
                    if at >= 0:
                        dets.append(at)
                        committed = committed or rounds_of[d] < commit_end
                if len(dets) == 0 or settled:
                    continue
                var obs = e.obs if committed else UInt64(0)
                if committed:
                    for d in e.dets:
                        var f = future_index.get(d, -1)
                        if f >= 0 and f // free == chunk:
                            obs |= UInt64(1) << UInt64(K + f % free)
                sub.errors.append(DemError(e.p, dets^, obs))
            plan.dems.append(sub^)
        plan.windows.append(w^)
        if last:
            break
        start += step
    return plan^


struct SlidingWindowDecoder[D: Decoder](Decoder):
    """`decoders[i]` must be built from `plan.dems[i]`."""

    var plan: WindowPlan
    var decoders: List[Self.D]
    var syndrome: List[Bool]
    var fired_list: List[Int]

    def __init__(out self, var plan: WindowPlan, var decoders: List[Self.D]) raises:
        if len(decoders) != len(plan.dems):
            raise Error("QM-E736: " + String(len(decoders)) + " decoders for " + String(len(plan.dems)) + " window models. Hint: build one decoder per entry of plan.dems.")
        self.syndrome = List[Bool](length=plan.num_detectors, fill=False)
        self.plan = plan^
        self.decoders = decoders^
        self.fired_list = List[Int]()

    def name(self) -> String:
        return "sliding_window"

    def decode(mut self, fired: List[Int]) -> UInt64:
        if len(self.plan.windows) == 1 and len(self.plan.windows[0].detectors) == 0:
            return self.decoders[0].decode(fired)
        for d in fired:
            self.syndrome[d] = True
        var K = self.plan.num_observables
        var logical_mask = (UInt64(1) << UInt64(K)) - 1 if K < 64 else ~UInt64(0)
        var out = UInt64(0)
        for wi in range(len(self.plan.windows)):
            self.fired_list.clear()
            ref w = self.plan.windows[wi]
            for i in range(len(w.detectors)):
                if self.syndrome[w.detectors[i]]:
                    self.fired_list.append(i)
            for chunk in range(w.chunks):
                var mask = self.decoders[w.first + chunk].decode(self.fired_list)
                if chunk == 0:
                    out ^= mask & logical_mask
                var bits = mask >> UInt64(K)
                var i = 0
                while bits != 0:
                    if (bits & 1) != 0:
                        var d = w.future[chunk * w.per_chunk + i]
                        self.syndrome[d] = not self.syndrome[d]
                    bits >>= 1
                    i += 1
        for w in self.plan.windows:
            for d in w.detectors:
                self.syndrome[d] = False
            for d in w.future:
                self.syndrome[d] = False
        return out
