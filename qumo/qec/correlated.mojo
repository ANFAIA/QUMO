"""Correlated (two-pass) matching: `mwpm_correlated`.

Fowler's reweighting (arXiv:1310.0863) as PyMatching 2.3 does it. A Y error in a
surface code shows up as one X-graph edge and one Z-graph edge; the plain matcher
prices them separately. Two passes use that correlation:

  1. decode, and expand each matched pair into the graph edges of its path;
  2. each edge e in those paths makes the edges it fires together with likelier:
     for every error with edges e and f, P(f | e) = joint(e, f) / marginal(e),
     so f may get the lower weight log((1 - P) / P);
  3. decode again with those weights, then restore them in reverse order, since
     one edge can be lowered several times.

Opt-in: `mwpm` stays plain matching. On rotated surface codes under uniform
circuit noise the logical error rate drops by 1.3 to 2.2 at d >= 5 and gets
*worse* at d = 3 (PyMatching does the same, see docs/QEC.md), at about 3 times
the decoding time. Each thread needs its own weights, so a copy of this decoder
copies the integer graph, unlike SparseBlossomDecoder, whose threads share one.
"""

from std.collections import Dict
from std.math import log
from std.memory import ArcPointer
from qumo.qec.dem import DetectorErrorModel, DemError, combine_independent
from qumo.qec.decoders import MatchingGraph, Decoder
from qumo.qec.sparse_blossom import (
    SparseBlossomDecoder, SparseGraph, CompressedEdge, discretize_weights, weight_normalising_constant,
    SB_NUM_DISTINCT_WEIGHTS, _hpush, _hpop,
)


@fieldwise_init
struct ImpliedWeight(ImplicitlyCopyable):
    var edge: Int
    var weight: Float64


def _xor_probability(a: Float64, b: Float64) -> Float64:
    return a * (1.0 - b) + b * (1.0 - a)


def _component_edges(g: MatchingGraph, e: DemError) -> List[Int]:
    """Graph edges one DEM error fires."""
    var out = List[Int]()
    if len(e.comp_dets) > 1:
        for c in e.comp_dets:
            var v = c[1] if len(c) == 2 else g.boundary
            var i = g.edge_between(c[0], v)
            if i >= 0:
                out.append(i)
    elif len(e.dets) >= 1 and len(e.dets) <= 2:
        var v = e.dets[1] if len(e.dets) == 2 else g.boundary
        var i = g.edge_between(e.dets[0], v)
        if i >= 0:
            out.append(i)
    elif len(e.dets) > 2 and len(e.dets) <= 16:
        _ = g._decompose(e.dets, 0, e.obs, out)
    return out^


def implied_weights(dem: DetectorErrorModel, g: MatchingGraph) raises -> List[List[ImpliedWeight]]:
    """Per edge e: the edges f it makes likelier, with their conditional weight."""
    var m = g.num_edges()
    var joint = Dict[Int, Float64]()
    for err in dem.errors:
        if err.p > 0.5:
            raise Error("QM-E731: correlated matching needs error probabilities up to 0.5, got " + String(err.p) + ". Hint: use plain 'mwpm'.")
        var ids = _component_edges(g, err)
        for i in range(len(ids)):
            for j in range(len(ids)):
                var key = ids[i] * m + ids[j]
                joint[key] = _xor_probability(joint.get(key, 0.0), err.p)
    var out = List[List[ImpliedWeight]](capacity=m)
    for _ in range(m):
        out.append(List[ImpliedWeight]())
    for kv in joint.items():
        var e = kv.key // m
        var f = kv.key % m
        if e == f:
            continue
        var marginal = joint.get(e * m + e, 0.0)
        if marginal <= 0.0:
            continue
        var pi_ = min(0.5, kv.value / marginal)
        if pi_ <= 0.0:
            continue
        out[e].append(ImpliedWeight(f, log((1.0 - pi_) / pi_)))
    return out^


struct CorrelatedMatchingDecoder(Decoder):
    var matcher: SparseBlossomDecoder
    var g: ArcPointer[MatchingGraph]
    var implied: ArcPointer[List[List[Int]]]  # per edge: f0, w0, f1, w1, ... in integer units
    var base: ArcPointer[List[Int]]  # integer weight per edge
    var current: List[Int]
    var at_u: ArcPointer[List[Int]]  # edge -> adjacency slot of u, or -1
    var at_v: ArcPointer[List[Int]]
    var boundary_of: ArcPointer[List[Int]]  # edge -> detector with this boundary edge, or -1
    var log_edge: List[Int]
    var log_weight: List[Int]
    var dist: List[Int]
    var via: List[Int]
    var seen: List[Int]
    var stamp: Int
    var hk: List[Int]
    var hv: List[Int]
    var path: List[Int]

    def __init__(out self, dem: DetectorErrorModel, num_distinct_weights: Int = SB_NUM_DISTINCT_WEIGHTS) raises:
        var graph = MatchingGraph(dem)
        var raw = implied_weights(dem, graph)
        # the integer scale has to cover the implied weights too
        var wmax = 0.0
        for row in raw:
            for w in row:
                wmax = max(wmax, abs(w.weight))
        graph.wmax_extra = wmax
        self.matcher = SparseBlossomDecoder(graph, num_distinct_weights)
        var scale = weight_normalising_constant(graph, num_distinct_weights)
        var table = List[List[Int]](capacity=len(raw))
        for row in raw:
            var t = List[Int](capacity=2 * len(row))
            for w in row:
                t.append(w.edge)
                t.append(2 * Int(w.weight * scale + 0.5))
            table.append(t^)
        var wi = discretize_weights(graph, num_distinct_weights)
        self.current = wi.copy()
        var n = graph.num_nodes()
        # where each edge sits in the blossom's graph
        ref sg = self.matcher.graph[]
        var slot_u = List[Int](length=graph.num_edges(), fill=-1)
        var slot_v = List[Int](length=graph.num_edges(), fill=-1)
        var bnd = List[Int](length=graph.num_edges(), fill=-1)
        var fill = List[Int](length=sg.n + 1, fill=0)
        for v in range(sg.n + 1):
            fill[v] = sg.off[v]
        for e in range(graph.num_edges()):
            var u = graph.eu[e]
            var v = graph.ev[e]
            if v == graph.boundary or u == graph.boundary:
                bnd[e] = u if v == graph.boundary else v
            else:
                slot_u[e] = fill[u]
                fill[u] += 1
                slot_v[e] = fill[v]
                fill[v] += 1
        self.g = ArcPointer(graph^)
        self.implied = ArcPointer(table^)
        self.base = ArcPointer(wi^)
        self.at_u = ArcPointer(slot_u^)
        self.at_v = ArcPointer(slot_v^)
        self.boundary_of = ArcPointer(bnd^)
        self.log_edge = List[Int]()
        self.log_weight = List[Int]()
        self.dist = List[Int](length=n, fill=0)
        self.via = List[Int](length=n, fill=-1)
        self.seen = List[Int](length=n, fill=0)
        self.stamp = 0
        self.hk = List[Int]()
        self.hv = List[Int]()
        self.path = List[Int]()

    def __init__(out self, *, copy: Self):
        self.matcher = copy.matcher.copy()
        self.matcher.graph = ArcPointer(copy.matcher.graph[].clone())
        self.g = copy.g
        self.implied = copy.implied
        self.base = copy.base
        self.current = copy.current.copy()
        self.at_u = copy.at_u
        self.at_v = copy.at_v
        self.boundary_of = copy.boundary_of
        self.log_edge = List[Int]()
        self.log_weight = List[Int]()
        self.dist = copy.dist.copy()
        self.via = copy.via.copy()
        self.seen = copy.seen.copy()
        self.stamp = copy.stamp
        self.hk = List[Int]()
        self.hv = List[Int]()
        self.path = List[Int]()

    def name(self) -> String:
        return "mwpm_correlated"

    def _set_weight(mut self, e: Int, w: Int):
        self.current[e] = w
        var bnd = self.boundary_of[][e]
        if bnd >= 0:
            self.matcher.graph[].bw[bnd] = w
            return
        ref sg = self.matcher.graph[]
        var a = self.at_u[][e]
        var b = self.at_v[][e]
        sg.adj[a] = (w << 32) | sg.neighbour(a)
        sg.adj[b] = (w << 32) | sg.neighbour(b)

    def _shortest_path(mut self, a: Int, b: Int):
        """Edges of a shortest path a -> b (b = -1: the boundary) into self.path."""
        ref g = self.g[]
        var goal = g.boundary if b < 0 else b
        self.stamp += 1
        var stamp = self.stamp
        self.hk.clear()
        self.hv.clear()
        self.dist[a] = 0
        self.via[a] = -1
        self.seen[a] = stamp
        _hpush(self.hk, self.hv, 0, a)
        while len(self.hk) > 0:
            var top = _hpop(self.hk, self.hv)
            var d = top[0]
            var u = top[1]
            if d != self.dist[u]:
                continue
            if u == goal:
                break
            if u == g.boundary:
                continue  # a path never crosses the boundary
            for i in range(g.adj_off[u], g.adj_off[u + 1]):
                var e = g.adj_edge[i]
                var v = g.other(e, u)
                var nd = d + self.base[][e]
                if self.seen[v] != stamp or nd < self.dist[v]:
                    self.seen[v] = stamp
                    self.dist[v] = nd
                    self.via[v] = e
                    _hpush(self.hk, self.hv, nd, v)
        if self.seen[goal] != stamp:
            return
        var x = goal
        while x != a:
            var e = self.via[x]
            self.path.append(e)
            x = g.other(e, x)

    def decode(mut self, fired: List[Int]) -> UInt64:
        var edges = self.matcher.decode_to_edges(fired)
        var plain: UInt64 = 0
        self.path.clear()
        for m in edges:
            plain ^= m.obs
            self._shortest_path(m.a, m.b)
        self.log_edge.clear()
        self.log_weight.clear()
        for e in self.path:
            var table = self.implied  # a second handle: _set_weight mutates self
            ref row = table[][e]
            var i = 0
            while i < len(row):
                var f = row[i]
                var w = row[i + 1]
                if w < self.current[f]:
                    self.log_edge.append(f)
                    self.log_weight.append(self.current[f])
                    self._set_weight(f, w)
                i += 2
        if len(self.log_edge) == 0:
            return plain
        var obs = self.matcher.decode(fired)
        var i = len(self.log_edge) - 1
        while i >= 0:
            self._set_weight(self.log_edge[i], self.log_weight[i])
            i -= 1
        return obs
