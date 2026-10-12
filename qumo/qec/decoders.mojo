"""Decoders for detector error models.

Common interface (trait `Decoder`): `decode(fired_detectors) -> UInt64`
predicted observable flips (bit k = logical observable k). Decoders are
`Copyable`; batch helpers give every worker thread its own copy, so decoders
may keep mutable scratch space and need no locking. Any model that maps a
syndrome to observable flips - including a neural decoder (AlphaQubit-style,
see docs/QEC.md) - plugs in by implementing the trait.

`MatchingGraph` (PyMatching's model): nodes = detectors + one virtual
boundary node; every DEM error flipping <= 2 detectors is an edge (single
detector -> edge to the boundary) with weight log((1-p)/p) and the
observables it flips. Hyperedges (> 2 detectors, e.g. Y errors in CSS codes)
are decomposed into existing graph-like edges whose observable masks XOR to
the hyperedge's (Stim `decompose_errors` logic); undecomposable ones are
dropped and counted.

Decoders
  * `UnionFindDecoder`  - Delfosse-Nickerson almost-linear UF with weighted
    (Huang-Newman-Brown) cluster growth on integer edge lengths and
    peeling on a spanning forest of each cluster. Fast, slightly worse
    threshold than MWPM.
  * `MwpmDecoder`       - exact minimum-weight perfect matching. Shortest
    paths (+ observable parity) come from an all-pairs table precomputed once
    per graph (<= 3000 nodes, shared by worker threads) or from per-defect
    Dijkstra on bigger graphs. Defects split into independent components
    (i, j interact only if d_ij < b_i + b_j); each component is solved
    exactly: subset DP for <= 10 defects, otherwise Edmonds' blossom
    algorithm (port of van Rantwijk's O(n^3) max-weight matching) on the
    reduced complete graph w'(i, j) = min(d_ij, b_i + b_j) plus one boundary
    vertex when the count is odd (equivalent to per-defect boundary copies).
    Exact for any defect count; per-shot cost grows as O(k^3) in the largest
    component, so it is ideal for d <= 7 at p <= 0.5% (PyMatching's sparse
    blossom is the scalable version; use UF beyond that).
  * `LookupTableDecoder` - most-likely observable flip per syndrome from all
    combinations of <= `max_weight` DEM errors; tiny codes, <= 64 detectors.

Error codes: QM-E705 (size limits). Decoders never raise per shot; defects that
cannot be neutralised (disconnected graph) are counted in `failures`.
"""

from std.math import log
from std.collections import Dict
from std.bit import count_trailing_zeros
from std.memory import ArcPointer
from std.runtime import parallelism_level
from max.algorithm import parallelize

from qumo.qec.dem import DetectorErrorModel, combine_independent
from qumo.sim.frame import DetectorSamples, FW, BATCH


trait Decoder(Copyable, Deinitable):
    def decode(mut self, fired: List[Int]) -> UInt64:
        """Predicted observable-flip mask for one shot's fired detectors (sorted)."""
        ...

    def name(self) -> String:
        ...


# ---------------------------------------------------------------------------
# Matching graph
# ---------------------------------------------------------------------------
struct MatchingGraph(Copyable, Writable):
    var num_detectors: Int
    var num_observables: Int
    var boundary: Int  # node id of the virtual boundary (= num_detectors)
    var eu: List[Int]
    var ev: List[Int]
    var ep: List[Float64]
    var ew: List[Float64]
    var eobs: List[UInt64]
    var adj_off: List[Int]  # CSR over nodes
    var adj_edge: List[Int]
    var num_hyperedges: Int
    var num_dropped: Int
    var num_conflicts: Int
    var num_undetectable: Int
    var _index: Dict[Int, Int]

    def __init__(out self, dem: DetectorErrorModel):
        self.num_detectors = dem.num_detectors
        self.num_observables = dem.num_observables
        self.boundary = dem.num_detectors
        self.eu = List[Int]()
        self.ev = List[Int]()
        self.ep = List[Float64]()
        self.ew = List[Float64]()
        self.eobs = List[UInt64]()
        self.adj_off = List[Int]()
        self.adj_edge = List[Int]()
        self.num_hyperedges = 0
        self.num_dropped = 0
        self.num_conflicts = 0
        self.num_undetectable = 0
        self._index = Dict[Int, Int]()
        # pass 1: graph-like errors and Stim-style decomposed pieces
        for e in dem.errors:
            var nd = len(e.dets)
            if len(e.comp_dets) > 1:
                self.num_hyperedges += 1
                for c in range(len(e.comp_dets)):
                    ref cd = e.comp_dets[c]
                    var u = cd[0]
                    var v = cd[1] if len(cd) == 2 else self.boundary
                    self._add_edge(u, v, e.p, e.comp_obs[c])
                continue
            if nd == 0:
                if e.obs != 0:
                    self.num_undetectable += 1
                continue
            if nd > 2:
                continue
            var u = e.dets[0]
            var v = e.dets[1] if nd == 2 else self.boundary
            self._add_edge(u, v, e.p, e.obs)
        # pass 2: decompose remaining hyperedges against the existing edges
        for e in dem.errors:
            if len(e.dets) <= 2 or len(e.comp_dets) > 1:
                continue
            self.num_hyperedges += 1
            var parts = List[Int]()
            if len(e.dets) <= 16 and self._decompose(e.dets, 0, e.obs, parts):
                for idx in parts:
                    self.ep[idx] = combine_independent(self.ep[idx], e.p)
            else:
                self.num_dropped += 1
        # weights + CSR adjacency
        var nn = self.num_nodes()
        for i in range(len(self.ep)):
            self.ew.append(_weight(self.ep[i]))
        var deg = List[Int](length=nn + 1, fill=0)
        for i in range(len(self.eu)):
            deg[self.eu[i]] += 1
            deg[self.ev[i]] += 1
        self.adj_off = List[Int](length=nn + 1, fill=0)
        for v in range(nn):
            self.adj_off[v + 1] = self.adj_off[v] + deg[v]
        self.adj_edge = List[Int](length=self.adj_off[nn], fill=0)
        var fill = self.adj_off.copy()
        for i in range(len(self.eu)):
            self.adj_edge[fill[self.eu[i]]] = i
            fill[self.eu[i]] += 1
            self.adj_edge[fill[self.ev[i]]] = i
            fill[self.ev[i]] += 1

    def num_nodes(self) -> Int:
        return self.num_detectors + 1

    def num_edges(self) -> Int:
        return len(self.eu)

    def _key(self, u: Int, v: Int) -> Int:
        var a = min(u, v)
        var b = max(u, v)
        return a * (self.num_detectors + 1) + b

    def edge_between(self, u: Int, v: Int) -> Int:
        return self._index.get(self._key(u, v), -1)

    def _add_edge(mut self, u: Int, v: Int, p: Float64, obs: UInt64):
        var key = self._key(u, v)
        var idx = self._index.get(key, -1)
        if idx < 0:
            self._index[key] = len(self.eu)
            self.eu.append(min(u, v))
            self.ev.append(max(u, v))
            self.ep.append(p)
            self.eobs.append(obs)
            return
        if self.eobs[idx] == obs:
            self.ep[idx] = combine_independent(self.ep[idx], p)
        else:
            self.num_conflicts += 1
            if p > self.ep[idx]:
                self.ep[idx] = p
                self.eobs[idx] = obs

    def _decompose(self, dets: List[Int], used: Int, target: UInt64, mut parts: List[Int]) -> Bool:
        """DFS: cover `dets` by known edges (pairs or boundary) with XOR(obs) == target."""
        var i = -1
        for t in range(len(dets)):
            if ((used >> t) & 1) == 0:
                i = t
                break
        if i < 0:
            return target == 0
        var used_i = used | (1 << i)
        for j in range(i + 1, len(dets)):
            if ((used_i >> j) & 1) != 0:
                continue
            var e = self.edge_between(dets[i], dets[j])
            if e >= 0:
                parts.append(e)
                if self._decompose(dets, used_i | (1 << j), target ^ self.eobs[e], parts):
                    return True
                _ = parts.pop()
        var eb = self.edge_between(dets[i], self.boundary)
        if eb >= 0:
            parts.append(eb)
            if self._decompose(dets, used_i, target ^ self.eobs[eb], parts):
                return True
            _ = parts.pop()
        return False

    def other(self, e: Int, v: Int) -> Int:
        return self.eu[e] + self.ev[e] - v

    def graphlike_distance(self) -> Int:
        """Fewest graph-like errors forming an undetectable logical error (flipping
        observable 0); -1 if none. BFS over (node, observable parity) states."""
        var nn = self.num_nodes()
        var best = -1
        var starts = List[Int]()
        if self.adj_off[self.boundary + 1] > self.adj_off[self.boundary]:
            starts.append(self.boundary)
        else:
            for v in range(nn):
                starts.append(v)
        var dist = List[Int](length=2 * nn, fill=-1)
        var queue = List[Int]()
        for s in starts:
            for i in range(2 * nn):
                dist[i] = -1
            queue.clear()
            dist[2 * s] = 0
            queue.append(2 * s)
            var head = 0
            while head < len(queue):
                var st = queue[head]
                head += 1
                var v = st >> 1
                var par = st & 1
                if best >= 0 and dist[st] >= best:
                    break
                for a in range(self.adj_off[v], self.adj_off[v + 1]):
                    var e = self.adj_edge[a]
                    var w = self.other(e, v)
                    var np = par ^ Int(self.eobs[e] & 1)
                    var ns = 2 * w + np
                    if dist[ns] < 0:
                        dist[ns] = dist[st] + 1
                        queue.append(ns)
            if dist[2 * s + 1] >= 0 and (best < 0 or dist[2 * s + 1] < best):
                best = dist[2 * s + 1]
        return best

    def write_to(self, mut w: Some[Writer]):
        w.write("MatchingGraph(nodes=", self.num_nodes(), ", edges=", self.num_edges(),
                ", hyperedges=", self.num_hyperedges, ", dropped=", self.num_dropped,
                ", conflicts=", self.num_conflicts, ")")


def _weight(p: Float64) -> Float64:
    if p <= 0.0:
        return 50.0
    if p >= 0.5:
        return 1e-3
    return max(1e-3, log((1.0 - p) / p))


# ---------------------------------------------------------------------------
# Union-Find decoder
# ---------------------------------------------------------------------------
comptime UF_SCALE = 32.0


struct UnionFindDecoder(Decoder):
    """Weighted Union-Find decoder (Delfosse & Nickerson 2017; weighted growth)."""

    var g: MatchingGraph
    var elen: List[Int]
    var parent: List[Int]
    var csize: List[Int]
    var odd: List[UInt8]
    var grounded: List[UInt8]
    var nxt: List[Int]
    var defect: List[UInt8]
    var growth: List[Int]
    var grown: List[UInt8]
    var rate: List[Int]
    var etouched: List[UInt8]
    var touched_v: List[Int]
    var touched_e: List[Int]
    var lid: List[Int]
    var seen: List[Int]
    var stamp: Int
    var failures: Int

    def __init__(out self, var graph: MatchingGraph):
        var nn = graph.num_nodes()
        var ne = graph.num_edges()
        self.elen = List[Int](capacity=ne)
        for i in range(ne):
            self.elen.append(max(1, Int(graph.ew[i] * UF_SCALE + 0.5)))
        self.g = graph^
        self.parent = List[Int](length=nn, fill=-1)
        self.csize = List[Int](length=nn, fill=0)
        self.odd = List[UInt8](length=nn, fill=0)
        self.grounded = List[UInt8](length=nn, fill=0)
        self.nxt = List[Int](length=nn, fill=-1)
        self.defect = List[UInt8](length=nn, fill=0)
        self.growth = List[Int](length=max(1, ne), fill=0)
        self.grown = List[UInt8](length=max(1, ne), fill=0)
        self.rate = List[Int](length=max(1, ne), fill=0)
        self.etouched = List[UInt8](length=max(1, ne), fill=0)
        self.touched_v = List[Int]()
        self.touched_e = List[Int]()
        self.lid = List[Int](length=nn, fill=-1)
        self.seen = List[Int](length=nn, fill=0)
        self.stamp = 0
        self.failures = 0

    @staticmethod
    def from_dem(dem: DetectorErrorModel) -> UnionFindDecoder:
        return UnionFindDecoder(MatchingGraph(dem))

    def name(self) -> String:
        return "union_find"

    def _make(mut self, v: Int):
        if self.parent[v] < 0:
            self.parent[v] = v
            self.csize[v] = 1
            self.odd[v] = 0
            self.grounded[v] = 1 if v == self.g.boundary else 0
            self.nxt[v] = v
            self.touched_v.append(v)

    def _find(mut self, v: Int) -> Int:
        var r = v
        while self.parent[r] != r:
            r = self.parent[r]
        var c = v
        while self.parent[c] != r:
            var nx = self.parent[c]
            self.parent[c] = r
            c = nx
        return r

    def _union(mut self, a: Int, b: Int):
        var ra = self._find(a)
        var rb = self._find(b)
        if ra == rb:
            return
        if self.csize[ra] < self.csize[rb]:
            var t = ra
            ra = rb
            rb = t
        self.parent[rb] = ra
        self.csize[ra] += self.csize[rb]
        self.odd[ra] ^= self.odd[rb]
        self.grounded[ra] |= self.grounded[rb]
        var t2 = self.nxt[ra]
        self.nxt[ra] = self.nxt[rb]
        self.nxt[rb] = t2

    def _touch_edge(mut self, e: Int):
        if self.etouched[e] == 0:
            self.etouched[e] = 1
            self.touched_e.append(e)

    def _cleanup(mut self):
        for v in self.touched_v:
            self.parent[v] = -1
            self.defect[v] = 0
            self.lid[v] = -1
        for e in self.touched_e:
            self.growth[e] = 0
            self.grown[e] = 0
            self.rate[e] = 0
            self.etouched[e] = 0
        self.touched_v.clear()
        self.touched_e.clear()

    def decode(mut self, fired: List[Int]) -> UInt64:
        if len(fired) == 0:
            return 0
        for d in fired:
            self._make(d)
            self.defect[d] = 1
            self.odd[d] = 1
        var cand = fired.copy()
        var roots = List[Int]()
        var bedges = List[Int]()
        var newly = List[Int]()
        while True:
            # active clusters: odd parity and not touching the boundary
            self.stamp += 1
            roots.clear()
            for v in cand:
                var r = self._find(v)
                if self.seen[r] != self.stamp:
                    self.seen[r] = self.stamp
                    if self.odd[r] != 0 and self.grounded[r] == 0:
                        roots.append(r)
            if len(roots) == 0:
                break
            bedges.clear()
            for r in roots:
                var v = r
                while True:
                    for a in range(self.g.adj_off[v], self.g.adj_off[v + 1]):
                        var e = self.g.adj_edge[a]
                        if self.grown[e] == 0:
                            if self.rate[e] == 0:
                                bedges.append(e)
                            self.rate[e] += 1
                    v = self.nxt[v]
                    if v == r:
                        break
            if len(bedges) == 0:
                self.failures += 1  # isolated odd cluster: cannot be neutralised
                break
            var delta = -1
            for e in bedges:
                var rem = self.elen[e] - self.growth[e]
                var steps = (rem + self.rate[e] - 1) // self.rate[e]
                if delta < 0 or steps < delta:
                    delta = steps
            newly.clear()
            for e in bedges:
                self._touch_edge(e)
                self.growth[e] += delta * self.rate[e]
                self.rate[e] = 0
                if self.growth[e] >= self.elen[e]:
                    self.grown[e] = 1
                    newly.append(e)
            for e in newly:
                var u = self.g.eu[e]
                var w = self.g.ev[e]
                self._make(u)
                self._make(w)
                self._union(u, w)
            cand = roots.copy()
        var obs = self._peel()
        self._cleanup()
        return obs

    def _peel(mut self) -> UInt64:
        """Spanning forest of the grown subgraph, then leaf-to-root peeling."""
        var nv = len(self.touched_v)
        for i in range(nv):
            self.lid[self.touched_v[i]] = i
        var deg = List[Int](length=nv + 1, fill=0)
        var gl = List[Int]()
        for e in self.touched_e:
            if self.grown[e] != 0:
                gl.append(e)
                deg[self.lid[self.g.eu[e]]] += 1
                deg[self.lid[self.g.ev[e]]] += 1
        var off = List[Int](length=nv + 1, fill=0)
        for i in range(nv):
            off[i + 1] = off[i] + deg[i]
        var fillp = off.copy()
        var ladj = List[Int](length=max(1, off[nv]), fill=0)
        for e in gl:
            var a = self.lid[self.g.eu[e]]
            var b = self.lid[self.g.ev[e]]
            ladj[fillp[a]] = e
            fillp[a] += 1
            ladj[fillp[b]] = e
            fillp[b] += 1
        var pe = List[Int](length=nv, fill=-2)  # -2 unvisited, -1 root
        var order = List[Int](capacity=nv)
        var bl = self.lid[self.g.boundary] if self.parent[self.g.boundary] >= 0 else -1
        var starts = List[Int]()
        if bl >= 0:
            starts.append(bl)
        for i in range(nv):
            starts.append(i)
        for s in starts:
            if pe[s] != -2:
                continue
            pe[s] = -1
            var head = len(order)
            order.append(s)
            while head < len(order):
                var x = order[head]
                head += 1
                var vx = self.touched_v[x]
                for a in range(off[x], off[x + 1]):
                    var e = ladj[a]
                    var y = self.lid[self.g.other(e, vx)]
                    if pe[y] == -2:
                        pe[y] = e
                        order.append(y)
        var obs: UInt64 = 0
        var i = len(order) - 1
        while i >= 0:
            var x = order[i]
            i -= 1
            var vx = self.touched_v[x]
            if self.defect[vx] != 0 and pe[x] >= 0:
                var e = pe[x]
                obs ^= self.g.eobs[e]
                self.defect[vx] = 0
                var u = self.g.other(e, vx)
                self.defect[u] ^= 1
        return obs


# ---------------------------------------------------------------------------
# Exact minimum-weight perfect matching
# ---------------------------------------------------------------------------
struct _Heap(Movable, Sized):
    """Binary min-heap of (key, value) with lazy deletion."""

    var keys: List[Float64]
    var vals: List[Int]

    def __init__(out self):
        self.keys = List[Float64]()
        self.vals = List[Int]()

    def clear(mut self):
        self.keys.clear()
        self.vals.clear()

    def __len__(self) -> Int:
        return len(self.keys)

    def push(mut self, k: Float64, v: Int):
        self.keys.append(k)
        self.vals.append(v)
        var i = len(self.keys) - 1
        while i > 0:
            var p = (i - 1) >> 1
            if self.keys[p] <= self.keys[i]:
                break
            self._swap(i, p)
            i = p

    def _swap(mut self, a: Int, b: Int):
        var tk = self.keys[a]
        self.keys[a] = self.keys[b]
        self.keys[b] = tk
        var tv = self.vals[a]
        self.vals[a] = self.vals[b]
        self.vals[b] = tv

    def pop(mut self) -> Tuple[Float64, Int]:
        var k = self.keys[0]
        var v = self.vals[0]
        var last = len(self.keys) - 1
        self.keys[0] = self.keys[last]
        self.vals[0] = self.vals[last]
        _ = self.keys.pop()
        _ = self.vals.pop()
        var n = len(self.keys)
        var i = 0
        while True:
            var l = 2 * i + 1
            var r = l + 1
            var m = i
            if l < n and self.keys[l] < self.keys[m]:
                m = l
            if r < n and self.keys[r] < self.keys[m]:
                m = r
            if m == i:
                break
            self._swap(i, m)
            i = m
        return (k, v)


struct BlossomMatcher(Movable):
    """Maximum-weight (optionally max-cardinality) matching on a general graph.

    Faithful port of Joris van Rantwijk's O(n^3) implementation of Edmonds'
    blossom algorithm with primal-dual updates (the code behind
    networkx.max_weight_matching). Integer weights -> integer arithmetic
    (all duals and slacks are pre-multiplied by two).
    """

    var nvertex: Int
    var nedge: Int
    var eu: List[Int]
    var ev: List[Int]
    var ew: List[Int]
    var endpoint: List[Int]
    var neighbend: List[List[Int]]
    var mate: List[Int]
    var label: List[Int]
    var labelend: List[Int]
    var inblossom: List[Int]
    var blossomparent: List[Int]
    var blossomchilds: List[List[Int]]
    var blossombase: List[Int]
    var blossomendps: List[List[Int]]
    var bestedge: List[Int]
    var blossombestedges: List[List[Int]]
    var has_bbe: List[Bool]
    var unusedblossoms: List[Int]
    var dualvar: List[Int]
    var allowedge: List[Bool]
    var queue: List[Int]

    def __init__(out self, nvertex: Int, eu: List[Int], ev: List[Int], ew: List[Int]):
        var n = nvertex
        self.nvertex = n
        self.nedge = len(eu)
        self.eu = eu.copy()
        self.ev = ev.copy()
        self.ew = ew.copy()
        self.endpoint = List[Int](capacity=2 * self.nedge)
        for k in range(self.nedge):
            self.endpoint.append(eu[k])
            self.endpoint.append(ev[k])
        self.neighbend = List[List[Int]](length=n, fill=List[Int]())
        for k in range(self.nedge):
            self.neighbend[eu[k]].append(2 * k + 1)
            self.neighbend[ev[k]].append(2 * k)
        var maxweight = 0
        for w in ew:
            maxweight = max(maxweight, w)
        self.mate = List[Int](length=n, fill=-1)
        self.label = List[Int](length=2 * n, fill=0)
        self.labelend = List[Int](length=2 * n, fill=-1)
        self.inblossom = List[Int](capacity=n)
        for i in range(n):
            self.inblossom.append(i)
        self.blossomparent = List[Int](length=2 * n, fill=-1)
        self.blossomchilds = List[List[Int]](length=2 * n, fill=List[Int]())
        self.blossombase = List[Int](length=2 * n, fill=-1)
        for i in range(n):
            self.blossombase[i] = i
        self.blossomendps = List[List[Int]](length=2 * n, fill=List[Int]())
        self.bestedge = List[Int](length=2 * n, fill=-1)
        self.blossombestedges = List[List[Int]](length=2 * n, fill=List[Int]())
        self.has_bbe = List[Bool](length=2 * n, fill=False)
        self.unusedblossoms = List[Int]()
        for i in range(n, 2 * n):
            self.unusedblossoms.append(i)
        self.dualvar = List[Int](length=2 * n, fill=0)
        for i in range(n):
            self.dualvar[i] = maxweight
        self.allowedge = List[Bool](length=max(1, self.nedge), fill=False)
        self.queue = List[Int]()

    @always_inline
    def _slack(self, k: Int) -> Int:
        return self.dualvar[self.eu[k]] + self.dualvar[self.ev[k]] - 2 * self.ew[k]

    def _leaves(self, b: Int, mut out: List[Int]):
        if b < self.nvertex:
            out.append(b)
            return
        for t in self.blossomchilds[b]:
            if t < self.nvertex:
                out.append(t)
            else:
                self._leaves(t, out)

    def _assign_label(mut self, w: Int, t: Int, p: Int):
        var b = self.inblossom[w]
        self.label[w] = t
        self.label[b] = t
        self.labelend[w] = p
        self.labelend[b] = p
        self.bestedge[w] = -1
        self.bestedge[b] = -1
        if t == 1:
            var lv = List[Int]()
            self._leaves(b, lv)
            for v in lv:
                self.queue.append(v)
        elif t == 2:
            var base = self.blossombase[b]
            var mb = self.mate[base]
            self._assign_label(self.endpoint[mb], 1, mb ^ 1)

    def _scan_blossom(mut self, v_in: Int, w_in: Int) -> Int:
        var v = v_in
        var w = w_in
        var path = List[Int]()
        var base = -1
        while v != -1 or w != -1:
            var b = self.inblossom[v]
            if (self.label[b] & 4) != 0:
                base = self.blossombase[b]
                break
            path.append(b)
            self.label[b] = 5
            if self.labelend[b] == -1:
                v = -1
            else:
                v = self.endpoint[self.labelend[b]]
                b = self.inblossom[v]
                v = self.endpoint[self.labelend[b]]
            if w != -1:
                var t = v
                v = w
                w = t
        for b in path:
            self.label[b] = 1
        return base

    def _add_blossom(mut self, base: Int, k: Int):
        var v = self.eu[k]
        var w = self.ev[k]
        var bb = self.inblossom[base]
        var bv = self.inblossom[v]
        var bw = self.inblossom[w]
        var b = self.unusedblossoms.pop()
        self.blossombase[b] = base
        self.blossomparent[b] = -1
        self.blossomparent[bb] = b
        var path = List[Int]()
        var endps = List[Int]()
        while bv != bb:
            self.blossomparent[bv] = b
            path.append(bv)
            endps.append(self.labelend[bv])
            v = self.endpoint[self.labelend[bv]]
            bv = self.inblossom[v]
        path.append(bb)
        path.reverse()
        endps.reverse()
        endps.append(2 * k)
        while bw != bb:
            self.blossomparent[bw] = b
            path.append(bw)
            endps.append(self.labelend[bw] ^ 1)
            w = self.endpoint[self.labelend[bw]]
            bw = self.inblossom[w]
        self.label[b] = 1
        self.labelend[b] = self.labelend[bb]
        self.dualvar[b] = 0
        self.blossomchilds[b] = path.copy()
        self.blossomendps[b] = endps^
        var lv = List[Int]()
        self._leaves(b, lv)
        for x in lv:
            if self.label[self.inblossom[x]] == 2:
                self.queue.append(x)
            self.inblossom[x] = b
        var bestedgeto = List[Int](length=2 * self.nvertex, fill=-1)
        for bvv in path:
            var nblist = List[Int]()
            if not self.has_bbe[bvv]:
                var lv2 = List[Int]()
                self._leaves(bvv, lv2)
                for x in lv2:
                    for p in self.neighbend[x]:
                        nblist.append(p // 2)
            else:
                nblist = self.blossombestedges[bvv].copy()
            for kk in nblist:
                var j = self.ev[kk]
                if self.inblossom[j] == b:
                    j = self.eu[kk]
                var bj = self.inblossom[j]
                if bj != b and self.label[bj] == 1 and (bestedgeto[bj] == -1 or self._slack(kk) < self._slack(bestedgeto[bj])):
                    bestedgeto[bj] = kk
            self.blossombestedges[bvv] = List[Int]()
            self.has_bbe[bvv] = False
            self.bestedge[bvv] = -1
        var bl = List[Int]()
        for kk in bestedgeto:
            if kk != -1:
                bl.append(kk)
        self.bestedge[b] = -1
        for kk in bl:
            if self.bestedge[b] == -1 or self._slack(kk) < self._slack(self.bestedge[b]):
                self.bestedge[b] = kk
        self.blossombestedges[b] = bl^
        self.has_bbe[b] = True

    @always_inline
    def _child(self, b: Int, j: Int) -> Int:
        var n = len(self.blossomchilds[b])
        return self.blossomchilds[b][((j % n) + n) % n]

    @always_inline
    def _endp(self, b: Int, j: Int) -> Int:
        var n = len(self.blossomendps[b])
        return self.blossomendps[b][((j % n) + n) % n]

    def _index_of(self, b: Int, t: Int) -> Int:
        for i in range(len(self.blossomchilds[b])):
            if self.blossomchilds[b][i] == t:
                return i
        return -1

    def _relabel_sub(mut self, bv: Int):
        """Expansion helper: give label T to a sub-blossom reached from outside."""
        var lv = List[Int]()
        self._leaves(bv, lv)
        var v = -1
        var idx = 0
        while idx < len(lv):
            v = lv[idx]
            if self.label[v] != 0:
                break
            idx += 1
        if v >= 0 and self.label[v] != 0:
            self.label[v] = 0
            self.label[self.endpoint[self.mate[self.blossombase[bv]]]] = 0
            self._assign_label(v, 2, self.labelend[v])

    def _expand_blossom(mut self, b: Int, endstage: Bool):
        var childs = self.blossomchilds[b].copy()
        for s in childs:
            self.blossomparent[s] = -1
            if s < self.nvertex:
                self.inblossom[s] = s
            elif endstage and self.dualvar[s] == 0:
                self._expand_blossom(s, endstage)
            else:
                var lv = List[Int]()
                self._leaves(s, lv)
                for v in lv:
                    self.inblossom[v] = s
        if (not endstage) and self.label[b] == 2:
            var entrychild = self.inblossom[self.endpoint[self.labelend[b] ^ 1]]
            var j = self._index_of(b, entrychild)
            var jstep: Int
            var endptrick: Int
            if (j & 1) != 0:
                j -= len(self.blossomchilds[b])
                jstep = 1
                endptrick = 0
            else:
                jstep = -1
                endptrick = 1
            var p = self.labelend[b]
            while j != 0:
                self.label[self.endpoint[p ^ 1]] = 0
                self.label[self.endpoint[self._endp(b, j - endptrick) ^ endptrick ^ 1]] = 0
                self._assign_label(self.endpoint[p ^ 1], 2, p)
                self.allowedge[self._endp(b, j - endptrick) // 2] = True
                j += jstep
                p = self._endp(b, j - endptrick) ^ endptrick
                self.allowedge[p // 2] = True
                j += jstep
            var bv = self._child(b, j)
            self.label[self.endpoint[p ^ 1]] = 2
            self.label[bv] = 2
            self.labelend[self.endpoint[p ^ 1]] = p
            self.labelend[bv] = p
            self.bestedge[bv] = -1
            j += jstep
            while self._child(b, j) != entrychild:
                bv = self._child(b, j)
                if self.label[bv] != 1:
                    self._relabel_sub(bv)
                j += jstep
        self.label[b] = -1
        self.labelend[b] = -1
        self.blossomchilds[b] = List[Int]()
        self.blossomendps[b] = List[Int]()
        self.blossombase[b] = -1
        self.blossombestedges[b] = List[Int]()
        self.has_bbe[b] = False
        self.bestedge[b] = -1
        self.unusedblossoms.append(b)

    def _augment_blossom(mut self, b: Int, v: Int):
        var t = v
        while self.blossomparent[t] != b:
            t = self.blossomparent[t]
        if t >= self.nvertex:
            self._augment_blossom(t, v)
        var i = self._index_of(b, t)
        var j = i
        var jstep: Int
        var endptrick: Int
        if (i & 1) != 0:
            j -= len(self.blossomchilds[b])
            jstep = 1
            endptrick = 0
        else:
            jstep = -1
            endptrick = 1
        while j != 0:
            j += jstep
            t = self._child(b, j)
            var p = self._endp(b, j - endptrick) ^ endptrick
            if t >= self.nvertex:
                self._augment_blossom(t, self.endpoint[p])
            j += jstep
            t = self._child(b, j)
            if t >= self.nvertex:
                self._augment_blossom(t, self.endpoint[p ^ 1])
            self.mate[self.endpoint[p]] = p ^ 1
            self.mate[self.endpoint[p ^ 1]] = p
        # rotate so the new base is first
        var nc = len(self.blossomchilds[b])
        var rc = List[Int](capacity=nc)
        var re = List[Int](capacity=nc)
        for x in range(nc):
            rc.append(self.blossomchilds[b][(i + x) % nc])
            re.append(self.blossomendps[b][(i + x) % nc])
        self.blossomchilds[b] = rc^
        self.blossomendps[b] = re^
        self.blossombase[b] = self.blossombase[self.blossomchilds[b][0]]

    def _augment_matching(mut self, k: Int):
        var v = self.eu[k]
        var w = self.ev[k]
        for side in range(2):
            var s = v if side == 0 else w
            var p = 2 * k + 1 if side == 0 else 2 * k
            while True:
                var bs = self.inblossom[s]
                if bs >= self.nvertex:
                    self._augment_blossom(bs, s)
                self.mate[s] = p
                if self.labelend[bs] == -1:
                    break
                var t = self.endpoint[self.labelend[bs]]
                var bt = self.inblossom[t]
                s = self.endpoint[self.labelend[bt]]
                var j = self.endpoint[self.labelend[bt] ^ 1]
                if bt >= self.nvertex:
                    self._augment_blossom(bt, j)
                self.mate[j] = self.labelend[bt]
                p = self.labelend[bt] ^ 1

    def solve(mut self, maxcardinality: Bool = True) -> List[Int]:
        """Run the algorithm; returns mate[v] (partner vertex or -1)."""
        var n = self.nvertex
        if self.nedge == 0:
            return List[Int](length=n, fill=-1)
        for _ in range(n):
            for i in range(2 * n):
                self.label[i] = 0
                self.bestedge[i] = -1
            for i in range(n, 2 * n):
                self.blossombestedges[i] = List[Int]()
                self.has_bbe[i] = False
            for i in range(self.nedge):
                self.allowedge[i] = False
            self.queue.clear()
            for v in range(n):
                if self.mate[v] == -1 and self.label[self.inblossom[v]] == 0:
                    self._assign_label(v, 1, -1)
            var augmented = False
            while True:
                while len(self.queue) > 0 and not augmented:
                    var v = self.queue.pop()
                    var nbn = len(self.neighbend[v])
                    for nbi in range(nbn):
                        var p = self.neighbend[v][nbi]
                        var k = p // 2
                        var w = self.endpoint[p]
                        if self.inblossom[v] == self.inblossom[w]:
                            continue
                        var kslack = 0
                        if not self.allowedge[k]:
                            kslack = self._slack(k)
                            if kslack <= 0:
                                self.allowedge[k] = True
                        if self.allowedge[k]:
                            if self.label[self.inblossom[w]] == 0:
                                self._assign_label(w, 2, p ^ 1)
                            elif self.label[self.inblossom[w]] == 1:
                                var base = self._scan_blossom(v, w)
                                if base >= 0:
                                    self._add_blossom(base, k)
                                else:
                                    self._augment_matching(k)
                                    augmented = True
                                    break
                            elif self.label[w] == 0:
                                self.label[w] = 2
                                self.labelend[w] = p ^ 1
                        elif self.label[self.inblossom[w]] == 1:
                            var b = self.inblossom[v]
                            if self.bestedge[b] == -1 or kslack < self._slack(self.bestedge[b]):
                                self.bestedge[b] = k
                        elif self.label[w] == 0:
                            if self.bestedge[w] == -1 or kslack < self._slack(self.bestedge[w]):
                                self.bestedge[w] = k
                if augmented:
                    break
                var deltatype = -1
                var delta = 0
                var deltaedge = -1
                var deltablossom = -1
                if not maxcardinality:
                    deltatype = 1
                    delta = self.dualvar[0]
                    for v in range(n):
                        delta = min(delta, self.dualvar[v])
                for v in range(n):
                    if self.label[self.inblossom[v]] == 0 and self.bestedge[v] != -1:
                        var d = self._slack(self.bestedge[v])
                        if deltatype == -1 or d < delta:
                            delta = d
                            deltatype = 2
                            deltaedge = self.bestedge[v]
                for b in range(2 * n):
                    if self.blossomparent[b] == -1 and self.label[b] == 1 and self.bestedge[b] != -1:
                        var ks = self._slack(self.bestedge[b])
                        var d = ks // 2
                        if deltatype == -1 or d < delta:
                            delta = d
                            deltatype = 3
                            deltaedge = self.bestedge[b]
                for b in range(n, 2 * n):
                    if self.blossombase[b] >= 0 and self.blossomparent[b] == -1 and self.label[b] == 2 and (deltatype == -1 or self.dualvar[b] < delta):
                        delta = self.dualvar[b]
                        deltatype = 4
                        deltablossom = b
                if deltatype == -1:
                    deltatype = 1
                    var mn = self.dualvar[0]
                    for v in range(n):
                        mn = min(mn, self.dualvar[v])
                    delta = max(0, mn)
                for v in range(n):
                    var lb = self.label[self.inblossom[v]]
                    if lb == 1:
                        self.dualvar[v] -= delta
                    elif lb == 2:
                        self.dualvar[v] += delta
                for b in range(n, 2 * n):
                    if self.blossombase[b] >= 0 and self.blossomparent[b] == -1:
                        if self.label[b] == 1:
                            self.dualvar[b] += delta
                        elif self.label[b] == 2:
                            self.dualvar[b] -= delta
                if deltatype == 1:
                    break
                elif deltatype == 2:
                    self.allowedge[deltaedge] = True
                    var i = self.eu[deltaedge]
                    if self.label[self.inblossom[i]] == 0:
                        i = self.ev[deltaedge]
                    self.queue.append(i)
                elif deltatype == 3:
                    self.allowedge[deltaedge] = True
                    self.queue.append(self.eu[deltaedge])
                elif deltatype == 4:
                    self._expand_blossom(deltablossom, False)
            if not augmented:
                break
            for b in range(n, 2 * n):
                if self.blossomparent[b] == -1 and self.blossombase[b] >= 0 and self.label[b] == 1 and self.dualvar[b] == 0:
                    self._expand_blossom(b, True)
        var out = List[Int](length=n, fill=-1)
        for v in range(n):
            if self.mate[v] >= 0:
                out[v] = self.endpoint[self.mate[v]]
        return out^


def _mwbm_dp(k: Int, dist: List[Float64], bdist: List[Float64], mut partner: List[Int]):
    """Exact subset DP (k <= ~14): dp[mask] = cheapest way to resolve `mask`."""
    var INF = 1e300
    var nm = 1 << k
    var dp = List[Float64](length=nm, fill=INF)
    var choice = List[Int](length=nm, fill=-2)
    # only pairs with d_ij < b_i + b_j can beat sending both to the boundary
    var nbr = List[Int](length=k, fill=0)
    for i in range(k):
        for j in range(k):
            if i != j and dist[i * k + j] < bdist[i] + bdist[j]:
                nbr[i] |= 1 << j
    dp[0] = 0.0
    for mask in range(1, nm):
        var i = Int(count_trailing_zeros(mask))
        var rest = mask ^ (1 << i)
        var best = dp[rest] + bdist[i]
        var bc = -1
        var r = rest & nbr[i]
        while r != 0:
            var j = Int(count_trailing_zeros(r))
            r &= r - 1
            var c = dp[rest ^ (1 << j)] + dist[i * k + j]
            if c < best:
                best = c
                bc = j
        dp[mask] = best
        choice[mask] = bc
    var mask = nm - 1
    while mask != 0:
        var i = Int(count_trailing_zeros(mask))
        var j = choice[mask]
        if j < 0:
            partner[i] = -1
            mask ^= 1 << i
        else:
            partner[i] = j
            partner[j] = i
            mask ^= (1 << i) | (1 << j)


def _mwbm_blossom(k: Int, dist: List[Float64], bdist: List[Float64], mut partner: List[Int]):
    """Blossom on the reduced complete graph: w'(i, j) = min(d_ij, b_i + b_j)
    ("pair directly" or "both to the boundary"), plus one boundary vertex when
    k is odd. Exactly equivalent to duplicating the boundary per defect, with
    half the vertices (the O(n^3) blossom gets ~8x cheaper)."""
    var INF = 1e300
    var scale = 1000.0
    var n = k + (k & 1)
    var wmax = 0
    var eu = List[Int]()
    var ev = List[Int]()
    var wraw = List[Int]()
    for i in range(k):
        for j in range(i + 1, k):
            var w = min(dist[i * k + j], bdist[i] + bdist[j])
            if w < INF:
                eu.append(i)
                ev.append(j)
                var wi = Int(w * scale + 0.5)
                wraw.append(wi)
                wmax = max(wmax, wi)
        if (k & 1) == 1 and bdist[i] < INF:
            eu.append(i)
            ev.append(k)
            var wi = Int(bdist[i] * scale + 0.5)
            wraw.append(wi)
            wmax = max(wmax, wi)
    var ew = List[Int](capacity=len(wraw))
    for w in wraw:
        ew.append(2 * (wmax + 1 - w))  # even positive integer weights
    var bm = BlossomMatcher(n, eu, ev, ew)
    var mate = bm.solve(True)
    for i in range(k):
        var m = mate[i]
        if m >= 0 and m < k and dist[i * k + m] < bdist[i] + bdist[m]:
            partner[i] = m
        else:
            partner[i] = -1


comptime DP_MAX = 10


def min_weight_boundary_matching(k: Int, dist: List[Float64], bdist: List[Float64], force_blossom: Bool = False) -> List[Int]:
    """Exact MWPM of k defects where each defect may instead go to the boundary.

    `dist` is k*k (>= 1e300 = unreachable), `bdist` length k. Returns
    partner[i] (another defect index, or -1 = boundary). The problem is first
    split into independent components (i, j interact only if
    d_ij < b_i + b_j); each component is solved by subset DP (<= 12 defects)
    or by the blossom algorithm.
    """
    var partner = List[Int](length=k, fill=-1)
    if k == 0:
        return partner^
    # components of the interaction graph (small union-find)
    var par = List[Int](capacity=k)
    for i in range(k):
        par.append(i)
    for i in range(k):
        for j in range(i + 1, k):
            if dist[i * k + j] < bdist[i] + bdist[j]:
                var a = i
                while par[a] != a:
                    a = par[a]
                var b = j
                while par[b] != b:
                    b = par[b]
                if a != b:
                    par[b] = a
    var comp = List[Int](length=k, fill=-1)
    var members = List[List[Int]]()
    for i in range(k):
        var r = i
        while par[r] != r:
            r = par[r]
        if comp[r] < 0:
            comp[r] = len(members)
            members.append(List[Int]())
        members[comp[r]].append(i)
    for ref mem in members:
        var m = len(mem)
        if m == 1:
            partner[mem[0]] = -1
            continue
        var ld = List[Float64](length=m * m, fill=1e300)
        var lb = List[Float64](length=m, fill=1e300)
        for a in range(m):
            lb[a] = bdist[mem[a]]
            for b in range(m):
                if a != b:
                    ld[a * m + b] = dist[mem[a] * k + mem[b]]
        var lp = List[Int](length=m, fill=-1)
        if m <= DP_MAX and not force_blossom:
            _mwbm_dp(m, ld, lb, lp)
        else:
            _mwbm_blossom(m, ld, lb, lp)
        for a in range(m):
            partner[mem[a]] = mem[lp[a]] if lp[a] >= 0 else -1
    return partner^


struct _Apsp(Movable, Deinitable):
    """All-pairs shortest paths of a matching graph (shared by decoder copies)."""

    var n: Int
    var dist: List[Float32]
    var obs: List[UInt8]

    def __init__(out self, g: MatchingGraph):
        var n = g.num_nodes()
        self.n = n
        self.dist = List[Float32](length=n * n, fill=Float32(3.0e38))
        self.obs = List[UInt8](length=n * n, fill=0)
        var dptr = self.dist.unsafe_ptr().as_unsafe_any_origin()
        var optr = self.obs.unsafe_ptr().as_unsafe_any_origin()
        var nchunks = max(1, min(n, 4 * max(1, parallelism_level())))
        var per = (n + nchunks - 1) // nchunks

        def task(c: Int) {imm g, imm dptr, imm optr, imm per, imm n}:
            var dist = List[Float64](length=n, fill=1e300)
            var po = List[UInt8](length=n, fill=0)
            var heap = _Heap()
            for src in range(c * per, min(n, (c + 1) * per)):
                for v in range(n):
                    dist[v] = 1e300
                heap.clear()
                dist[src] = 0.0
                po[src] = 0
                heap.push(0.0, src)
                while len(heap) > 0:
                    var top = heap.pop()
                    var d = top[0]
                    var v = top[1]
                    if d > dist[v]:
                        continue
                    dptr[unsafe_offset=src * n + v] = Float32(d)
                    optr[unsafe_offset=src * n + v] = po[v]
                    for a in range(g.adj_off[v], g.adj_off[v + 1]):
                        var e = g.adj_edge[a]
                        var w = g.other(e, v)
                        var nd = d + g.ew[e]
                        if nd < dist[w]:
                            dist[w] = nd
                            po[w] = po[v] ^ UInt8(g.eobs[e])
                            heap.push(nd, w)

        parallelize(task, nchunks)


comptime APSP_MAX_NODES = 3000


struct MwpmDecoder(Decoder):
    """Exact minimum-weight perfect-matching decoder (see module docstring).

    Graphs with <= 3000 nodes and <= 8 observables use a precomputed
    all-pairs shortest-path table (shared between thread copies through an
    ArcPointer, like PyMatching v0); larger graphs run Dijkstra per defect.
    """

    var g: MatchingGraph
    var apsp: ArcPointer[_Apsp]
    var use_apsp: Bool
    var dist: List[Float64]
    var pobs: List[UInt64]
    var stamp: List[Int]
    var cur: Int
    var tgt: List[Int]
    var tstamp: List[Int]
    var heap: _Heap
    var failures: Int
    var cur_t: Int

    def __init__(out self, var graph: MatchingGraph, use_apsp: Bool = True):
        var nn = graph.num_nodes()
        self.use_apsp = use_apsp and nn <= APSP_MAX_NODES and graph.num_observables <= 8
        if self.use_apsp:
            self.apsp = ArcPointer(_Apsp(graph))
        else:
            var empty = MatchingGraph(DetectorErrorModel(0, 0))
            self.apsp = ArcPointer(_Apsp(empty))
        self.cur_t = 0
        self.g = graph^
        self.dist = List[Float64](length=nn, fill=0.0)
        self.pobs = List[UInt64](length=nn, fill=0)
        self.stamp = List[Int](length=nn, fill=0)
        self.cur = 0
        self.tgt = List[Int](length=nn, fill=-1)
        self.tstamp = List[Int](length=nn, fill=0)
        self.heap = _Heap()
        self.failures = 0

    def __init__(out self, *, copy: Self):
        self.g = copy.g.copy()
        self.apsp = copy.apsp
        self.use_apsp = copy.use_apsp
        self.dist = copy.dist.copy()
        self.pobs = copy.pobs.copy()
        self.stamp = copy.stamp.copy()
        self.cur = copy.cur
        self.tgt = copy.tgt.copy()
        self.tstamp = copy.tstamp.copy()
        self.heap = _Heap()
        self.failures = copy.failures
        self.cur_t = copy.cur_t

    @staticmethod
    def from_dem(dem: DetectorErrorModel) -> MwpmDecoder:
        return MwpmDecoder(MatchingGraph(dem))

    def name(self) -> String:
        return "mwpm"

    def _dijkstra(mut self, src: Int, k: Int, bound: Float64, mut dout: List[Float64], mut oout: List[UInt64], row: Int):
        """Shortest paths from `src`; fills row `row` of the k x (k+1) tables for
        every target node (tgt[] >= 0) settled within `bound`."""
        self.cur += 1
        var c = self.cur
        self.heap.clear()
        self.stamp[src] = c
        self.dist[src] = 0.0
        self.pobs[src] = 0
        self.heap.push(0.0, src)
        var found = 0
        var need = k  # other defects + boundary
        while len(self.heap) > 0:
            var top = self.heap.pop()
            var d = top[0]
            var v = top[1]
            if d > self.dist[v]:
                continue
            if d > bound:
                break
            if v != src:
                if self.tstamp[v] == self.cur_t and self.tgt[v] >= 0:
                    var j = self.tgt[v]
                    dout[row * (k + 1) + j] = d
                    oout[row * (k + 1) + j] = self.pobs[v]
                    found += 1
                    if found >= need:
                        break
            for a in range(self.g.adj_off[v], self.g.adj_off[v + 1]):
                var e = self.g.adj_edge[a]
                var w = self.g.other(e, v)
                var nd = d + self.g.ew[e]
                if self.stamp[w] != c or nd < self.dist[w]:
                    self.stamp[w] = c
                    self.dist[w] = nd
                    self.pobs[w] = self.pobs[v] ^ self.g.eobs[e]
                    self.heap.push(nd, w)

    def decode(mut self, fired: List[Int]) -> UInt64:
        var k = len(fired)
        if k == 0:
            return 0
        var INF = 1e300
        var B = self.g.boundary
        var dist = List[Float64](length=k * k, fill=INF)
        var bdist = List[Float64](length=k, fill=INF)
        var dobs = List[UInt64](length=k * k, fill=0)
        var bobs = List[UInt64](length=k, fill=0)
        if self.use_apsp:
            var ap = self.apsp.ptr()
            var n = ap[].n
            for i in range(k):
                var row = fired[i] * n
                var bd = ap[].dist[row + B]
                if bd < Float32(1.0e38):
                    bdist[i] = Float64(bd)
                bobs[i] = UInt64(ap[].obs[row + B])
                for j in range(k):
                    if j != i:
                        var dd = ap[].dist[row + fired[j]]
                        if dd < Float32(1.0e38):
                            dist[i * k + j] = Float64(dd)
                        dobs[i * k + j] = UInt64(ap[].obs[row + fired[j]])
        else:
            var dtab = List[Float64](length=k * (k + 1), fill=INF)
            var otab = List[UInt64](length=k * (k + 1), fill=0)
            self.cur_t += 1
            for i in range(k):
                self.tgt[fired[i]] = i
                self.tstamp[fired[i]] = self.cur_t
            self.tgt[B] = k
            self.tstamp[B] = self.cur_t
            var bd = List[Float64](length=(k + 1), fill=INF)
            var bo = List[UInt64](length=(k + 1), fill=0)
            self._dijkstra(B, k, INF, bd, bo, 0)
            var bmax = 0.0
            for i in range(k):
                bdist[i] = bd[i]
                bobs[i] = bo[i]
                if bd[i] < INF:
                    bmax = max(bmax, bd[i])
            for i in range(k):
                var bound = bd[i] + bmax if bd[i] < INF else INF
                self._dijkstra(fired[i], k, bound, dtab, otab, i)
            for i in range(k):
                for j in range(k):
                    if i != j:
                        if dtab[i * (k + 1) + j] <= dtab[j * (k + 1) + i]:
                            dist[i * k + j] = dtab[i * (k + 1) + j]
                            dobs[i * k + j] = otab[i * (k + 1) + j]
                        else:
                            dist[i * k + j] = dtab[j * (k + 1) + i]
                            dobs[i * k + j] = otab[j * (k + 1) + i]
            for i in range(k):
                self.tgt[fired[i]] = -1
            self.tgt[B] = -1
        var partner = min_weight_boundary_matching(k, dist, bdist)
        var obs: UInt64 = 0
        for i in range(k):
            var j = partner[i]
            if j < 0:
                if bdist[i] >= INF:
                    self.failures += 1
                obs ^= bobs[i]
            elif j > i:
                obs ^= dobs[i * k + j]
        return obs


# ---------------------------------------------------------------------------
# Lookup table
# ---------------------------------------------------------------------------
def push_combo(s: UInt64, o: UInt64, w: Float64, mut acc: Dict[UInt64, Int], mut slot_syn: List[UInt64],
               mut slot_obs: List[List[UInt64]], mut slot_p: List[List[Float64]]):
    var idx = acc.get(s, -1)
    if idx < 0:
        idx = len(slot_syn)
        acc[s] = idx
        slot_syn.append(s)
        slot_obs.append(List[UInt64]())
        slot_p.append(List[Float64]())
    for t in range(len(slot_obs[idx])):
        if slot_obs[idx][t] == o:
            slot_p[idx][t] += w
            return
    slot_obs[idx].append(o)
    slot_p[idx].append(w)


struct LookupTableDecoder(Decoder):
    """Maximum-likelihood-ish table: for every syndrome reachable by <= max_weight
    DEM errors, the observable mask with the largest total probability."""

    var table: Dict[UInt64, Int]
    var best_obs: List[UInt64]
    var best_p: List[Float64]
    var num_detectors: Int

    def __init__(out self, dem: DetectorErrorModel, max_weight: Int = 2) raises:
        if dem.num_detectors > 64:
            raise Error("QM-E705: LookupTableDecoder supports <= 64 detectors (got " + String(dem.num_detectors) + "). Hint: use 'uf' or 'mwpm' for larger codes.")
        self.num_detectors = dem.num_detectors
        self.table = Dict[UInt64, Int]()
        self.best_obs = List[UInt64]()
        self.best_p = List[Float64]()
        var syn = List[UInt64]()
        var obs = List[UInt64]()
        var odds = List[Float64]()
        for e in dem.errors:
            var s: UInt64 = 0
            for d in e.dets:
                s ^= UInt64(1) << UInt64(d)
            syn.append(s)
            obs.append(e.obs)
            odds.append(e.p / (1.0 - e.p))
        # accumulate P(syndrome, obs) over all <= max_weight subsets (relative odds)
        var acc = Dict[UInt64, Int]()  # syndrome -> slot
        var slot_syn = List[UInt64]()
        var slot_obs = List[List[UInt64]]()
        var slot_p = List[List[Float64]]()
        var n = len(syn)
        push_combo(0, 0, 1.0, acc, slot_syn, slot_obs, slot_p)
        for a in range(n):
            push_combo(syn[a], obs[a], odds[a], acc, slot_syn, slot_obs, slot_p)
            if max_weight >= 2:
                for b in range(a + 1, n):
                    push_combo(syn[a] ^ syn[b], obs[a] ^ obs[b], odds[a] * odds[b], acc, slot_syn, slot_obs, slot_p)
                    if max_weight >= 3:
                        for c in range(b + 1, n):
                            push_combo(syn[a] ^ syn[b] ^ syn[c], obs[a] ^ obs[b] ^ obs[c], odds[a] * odds[b] * odds[c], acc, slot_syn, slot_obs, slot_p)
        for idx in range(len(slot_syn)):
            var bo: UInt64 = 0
            var bp = -1.0
            for t in range(len(slot_obs[idx])):
                if slot_p[idx][t] > bp:
                    bp = slot_p[idx][t]
                    bo = slot_obs[idx][t]
            self.table[slot_syn[idx]] = len(self.best_obs)
            self.best_obs.append(bo)
            self.best_p.append(bp)

    def name(self) -> String:
        return "lookup"

    def table_size(self) -> Int:
        return len(self.best_obs)

    def decode(mut self, fired: List[Int]) -> UInt64:
        var s: UInt64 = 0
        for d in fired:
            s ^= UInt64(1) << UInt64(d)
        var idx = self.table.get(s, -1)
        if idx < 0:
            return 0
        return self.best_obs[idx]


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------
def batch_fired(dets: List[UInt64], base: Int, D: Int, mut offsets: List[Int], mut flat: List[Int]):
    """Detector-major batch words -> per-shot sorted fired lists (CSR).

    dets[base + d*FW + w] holds detector d for shots 64w..64w+63 of the batch.
    """
    for i in range(BATCH + 1):
        offsets[i] = 0
    for d in range(D):
        for w in range(FW):
            var x = dets[base + d * FW + w]
            while x != 0:
                var b = Int(count_trailing_zeros(x))
                x &= x - 1
                offsets[w * 64 + b + 1] += 1
    for i in range(BATCH):
        offsets[i + 1] += offsets[i]
    var total = offsets[BATCH]
    flat.clear()
    for _ in range(total):
        flat.append(0)
    var fillp = offsets.copy()
    for d in range(D):
        for w in range(FW):
            var x = dets[base + d * FW + w]
            while x != 0:
                var b = Int(count_trailing_zeros(x))
                x &= x - 1
                var s = w * 64 + b
                flat[fillp[s]] = d
                fillp[s] += 1


def decode_samples[D: Decoder](dec: D, samples: DetectorSamples) -> List[UInt64]:
    """Predicted observable masks for every shot (parallel over batches)."""
    var out = List[UInt64](length=max(1, samples.shots), fill=0)
    var nb = samples.num_batches
    if nb == 0:
        return out^
    var optr = out.unsafe_ptr().as_unsafe_any_origin()
    var nchunks = max(1, min(nb, 4 * max(1, parallelism_level())))
    var per = (nb + nchunks - 1) // nchunks
    var ND = samples.num_detectors
    var shots = samples.shots

    def task(c: Int) {imm dec, imm samples, imm optr, imm per, imm nb, imm ND, imm shots}:
        var local = dec.copy()
        var offsets = List[Int](length=BATCH + 1, fill=0)
        var flat = List[Int]()
        var fired = List[Int]()
        var lo = c * per
        var hi = min(nb, lo + per)
        for b in range(lo, hi):
            batch_fired(samples.dets, b * ND * FW, ND, offsets, flat)
            for s in range(BATCH):
                var shot = b * BATCH + s
                if shot >= shots:
                    break
                fired.clear()
                for t in range(offsets[s], offsets[s + 1]):
                    fired.append(flat[t])
                optr[unsafe_offset=shot] = local.decode(fired)

    parallelize(task, nchunks)
    return out^


def count_logical_errors[D: Decoder](dec: D, samples: DetectorSamples) -> Int:
    """Shots where the decoder's prediction differs from the true observable flips."""
    var pred = decode_samples(dec, samples)
    var errors = 0
    for s in range(samples.shots):
        if pred[s] != samples.observables_mask(s):
            errors += 1
    return errors
