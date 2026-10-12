"""Sparse blossom: exact minimum-weight perfect matching on the detector graph.

Port of the algorithm behind PyMatching 2 (O. Higgott & C. Gidney, "Sparse
Blossom: correcting a million errors per core second with minimum-weight
matching", Quantum 9, 1600 (2025), arXiv:2303.15933). Unlike the dense
decoder in `qumo.qec.decoders.MwpmDecoder` (all-pairs shortest paths + an
O(k^3) Edmonds blossom on the complete graph of defects), sparse blossom
runs Edmonds' primal-dual algorithm *directly on the detector graph*: the
dual variables are explored areas of the graph, so no shortest-path
computation and no complete defect graph is ever built. Per-shot work is
proportional to the area the regions sweep, which is local at low p.

Ingredients (names follow the paper / PyMatching's C++ sources):

* **Graph fill regions** (`r_*` arrays): one per detection event, plus one
  per blossom. A region's radius is a piecewise-linear function of time,
  stored as `y + slope * t` with slope +1 (outer node of an alternating
  tree: growing), -1 (inner node: shrinking) or 0 (matched, or a blossom
  child: frozen). Every detector node remembers the region that reached it
  first (`n_reg`), that region's top-level blossom (`n_top`), the source
  detection event (`n_src`), the radius of arrival and the observables
  crossed on the way (`n_obs`); its "local radius" (how far the region
  extends past the node) is `radius(top) + wrapped`, with `wrapped` the
  frozen radii of the intermediate blossom children minus the arrival
  radius (PyMatching's `wrapped_radius_cached`).
* **Integer time / discretised weights**: edge weights log((1-p)/p) are
  scaled so the largest becomes `num_distinct_weights - 1` (2^24 - 1 by
  default, as in PyMatching), rounded and doubled. Even weights make every
  collision happen at an integer time (two growing fronts close an even gap
  at rate 2), so all arithmetic is exact `Int`.
* **Event queue** (the flooder): a binary min-heap of (time, node | region)
  with PyMatching's "queued/desired time" trackers so stale entries are
  dropped lazily. Node events: a growing region reaches an empty neighbour
  (the node joins its shell), hits another region, or hits the boundary.
  Region events: a shrinking region gives back the last node of its shell
  (LIFO), a trivial inner region implodes (its two tree neighbours touch ->
  blossom), or an inner blossom reaches radius 0 and shatters.
* **Matcher**: alternating trees of (inner, outer) region pairs
  (`a_*` arrays). Region-region collisions either (a) grow a tree by a
  matched pair (inner starts shrinking, its partner starts growing),
  (b) augment when two trees meet or a tree meets the boundary or a
  boundary-matched region (the tree is re-rooted at the colliding node and
  dissolved into matches), or (c) form a blossom when a tree hits itself
  (the odd cycle through the lowest common ancestor becomes a new growing
  region whose children freeze). Inner blossoms shatter at radius 0: the
  even path between the two tree edges re-enters the tree, the rest of the
  cycle is matched in pairs.
* **Compressed edges** (`CompressedEdge`): every match / tree / blossom
  edge only stores its two endpoint detection events and the XOR of the
  observables along the region path joining them, so the final observable
  prediction is the XOR over the matching's compressed edges (blossoms are
  expanded recursively at the end) and no path is ever traced.
* **No per-shot allocation**: node state lives in graph-sized arrays that
  are reset lazily through a list of touched nodes; regions and tree nodes
  come from pools whose lists keep their capacity across shots.

Exactness: the output is a minimum-weight perfect matching with respect to
the discretised integer weights. tests/test_qec_sparse_blossom.mojo checks,
on ~18k sampled and random syndromes (surface Z/X and repetition codes,
d = 3..7, random graphs with heterogeneous weights and several observables),
primal weight == dual objective == optimum of an independent oracle (integer
APSP + dense exact matcher, `MatchingOracle`) and that every matched pair's
observable mask belongs to a shortest path between its endpoints. Since the
discretisation is PyMatching's, predictions agree shot-for-shot with
PyMatching 2 on identical syndromes up to degenerate ties
(bench/bench_qec.py, part B). `last_weight()` returns the dual objective
(= matching weight at optimality) in integer units; divide by
`2 * weight_scale()` to get nats.

Limitations: weights come from `MatchingGraph` (log((1-p)/p) clamped to
[1e-3, 50] nats, so p > 0.5 edges are not treated as negative weights as
PyMatching does); <= 64 observables; node ids < 2^32. Defects in a connected
component without boundary and with odd parity cannot be matched: they are
counted in `failures` (no exception). Out-of-range or repeated detector ids in
`fired` are skipped and counted in `ignored`.

Error codes: QM-E712 (internal invariant violation in sparse blossom,
reported through the `anomalies` counter, never raised per shot; any nonzero
value is a bug).
"""

from std.memory import ArcPointer

from qumo.qec.dem import DetectorErrorModel
from qumo.qec.decoders import Decoder, MatchingGraph, min_weight_boundary_matching

comptime SB_NUM_DISTINCT_WEIGHTS = 1 << 24
comptime _BND = -2  # neighbour slot / match partner meaning "the boundary"
comptime _NOMATCH = -1
comptime _UNMATCHABLE = -3  # root of a tree that can never be matched (failure)
comptime _INF = 1 << 62


struct _UVec[T: ImplicitlyCopyable & Deinitable](Copyable, Sized):
    """List without bounds checks on element access (hot decoder state).
    Indices are always produced by the decoder itself."""

    var d: List[Self.T]

    def __init__(out self):
        self.d = List[Self.T]()

    def __init__(out self, n: Int, fill: Self.T):
        self.d = List[Self.T](length=n, fill=fill)

    @always_inline
    def __getitem__(self, i: Int) -> Self.T:
        return self.d.unsafe_get(i)

    @always_inline
    def __setitem__(mut self, i: Int, v: Self.T):
        self.d.unsafe_get(i) = v

    @always_inline
    def __len__(self) -> Int:
        return len(self.d)

    @always_inline
    def append(mut self, v: Self.T):
        self.d.append(v)

    @always_inline
    def pop(mut self) -> Self.T:
        return self.d.pop()

    @always_inline
    def clear(mut self):
        self.d.clear()

    @always_inline
    def last(self) -> Self.T:
        return self.d.unsafe_get(len(self.d) - 1)


@fieldwise_init
struct CompressedEdge(ImplicitlyCopyable, Writable):
    """Region-path between two detection events (`b == -1`: the boundary),
    with the XOR of the observables it crosses."""

    var a: Int
    var b: Int
    var obs: UInt64

    @always_inline
    def reversed(self) -> CompressedEdge:
        return CompressedEdge(self.b, self.a, self.obs)

    def write_to(self, mut w: Some[Writer]):
        w.write("CompressedEdge(", self.a, " -> ", self.b, ", obs=", self.obs, ")")


def weight_normalising_constant(g: MatchingGraph, num_distinct_weights: Int = SB_NUM_DISTINCT_WEIGHTS) -> Float64:
    """Integer units per nat: the largest |weight| maps to num_distinct_weights - 1."""
    var wmax = 0.0
    for w in g.ew:
        wmax = max(wmax, abs(w))
    if wmax <= 0.0:
        return 1.0
    return Float64(num_distinct_weights - 1) / wmax


def discretize_weights(g: MatchingGraph, num_distinct_weights: Int = SB_NUM_DISTINCT_WEIGHTS) -> List[Int]:
    """PyMatching's discretisation: w_int = 2 * round(w * c) (always even)."""
    var c = weight_normalising_constant(g, num_distinct_weights)
    var out = List[Int](capacity=len(g.ew))
    for w in g.ew:
        out.append(2 * Int(abs(w) * c + 0.5))
    return out^


struct SparseGraph(Movable):
    """Detector graph in CSR form with integer weights. Each adjacency entry
    packs (weight << 32) | neighbour; boundary edges are stored per node
    (`bw[u] < 0`: no boundary edge)."""

    var n: Int
    var off: _UVec[Int]
    var adj: _UVec[Int]
    var aobs: _UVec[UInt64]
    var bw: _UVec[Int]
    var bobs: _UVec[UInt64]
    var scale: Float64

    def __init__(out self, g: MatchingGraph, num_distinct_weights: Int = SB_NUM_DISTINCT_WEIGHTS):
        var n = g.num_detectors
        self.n = n
        self.scale = weight_normalising_constant(g, num_distinct_weights)
        var wi = discretize_weights(g, num_distinct_weights)
        self.bw = _UVec[Int](n, -1)
        self.bobs = _UVec[UInt64](n, 0)
        var deg = List[Int](length=n + 1, fill=0)
        for e in range(g.num_edges()):
            var u = g.eu[e]
            var v = g.ev[e]
            if v == g.boundary or u == g.boundary:
                var x = u if v == g.boundary else v
                self.bw[x] = wi[e]
                self.bobs[x] = g.eobs[e]
            else:
                deg[u] += 1
                deg[v] += 1
        self.off = _UVec[Int](n + 1, 0)
        for v in range(n):
            self.off[v + 1] = self.off[v] + deg[v]
        var m = self.off[n]
        self.adj = _UVec[Int](m, 0)
        self.aobs = _UVec[UInt64](m, 0)
        var fill = List[Int](length=n + 1, fill=0)
        for v in range(n + 1):
            fill[v] = self.off[v]
        for e in range(g.num_edges()):
            var u = g.eu[e]
            var v = g.ev[e]
            if v == g.boundary or u == g.boundary:
                continue
            self.adj[fill[u]] = (wi[e] << 32) | v
            self.aobs[fill[u]] = g.eobs[e]
            fill[u] += 1
            self.adj[fill[v]] = (wi[e] << 32) | u
            self.aobs[fill[v]] = g.eobs[e]
            fill[v] += 1

    @always_inline
    def neighbour(self, a: Int) -> Int:
        return self.adj[a] & 0xFFFFFFFF

    @always_inline
    def weight(self, a: Int) -> Int:
        return self.adj[a] >> 32


comptime _EMPTY = 0  # sentinel region owning every empty node (radius 0, frozen)


@fieldwise_init
struct _Scan(ImplicitlyCopyable):
    var t: Int  # earliest event time (_INF: none)
    var slot: Int  # adjacency index, _BND or -1
    var second: Int  # next-earliest event time among the other slots


struct SparseBlossomDecoder(Decoder):
    """Exact MWPM decoder (sparse blossom, PyMatching 2's algorithm)."""

    var graph: ArcPointer[SparseGraph]
    var num_detectors: Int
    var num_observables: Int
    # ---- detector node state (graph-sized, reset lazily) ----
    var n_src: _UVec[Int]  # source detection event of the owning region (-1: empty)
    var n_reg: _UVec[Int]  # region that reached the node (innermost; _EMPTY)
    var n_top: _UVec[Int]  # top-level blossom of n_reg (_EMPTY: empty node)
    var n_arr: _UVec[Int]  # radius of n_reg when it arrived
    var n_wrap: _UVec[Int]  # cached wrapped radius (0 for empty nodes)
    var n_obs: _UVec[UInt64]  # observables crossed from the source
    var n_qt: _UVec[Int]  # queued event time
    var n_dt: _UVec[Int]  # desired event time
    var n_qh: _UVec[Bool]
    var n_dh: _UVec[Bool]
    var n_touched: _UVec[Bool]
    var touched: _UVec[Int]
    # ---- graph fill regions (pool; index 0 is the _EMPTY sentinel) ----
    var r_y: _UVec[Int]
    var r_s: _UVec[Int]
    var r_par: _UVec[Int]  # blossom parent (-1: top level)
    var r_alt: _UVec[Int]  # alternating-tree node (-1: none)
    var r_match: _UVec[Int]  # partner region, _BND, _NOMATCH or _UNMATCHABLE
    var r_medge: _UVec[CompressedEdge]  # match edge (from this region)
    var r_shell: List[_UVec[Int]]
    var r_child: List[_UVec[Int]]  # blossom cycle
    var r_cedge: List[_UVec[CompressedEdge]]  # edge child k -> child k+1
    var r_qt: _UVec[Int]
    var r_dt: _UVec[Int]
    var r_qh: _UVec[Bool]
    var r_dh: _UVec[Bool]
    var r_live: _UVec[Bool]
    var r_free: _UVec[Int]
    var r_count: Int
    # ---- alternating-tree nodes (pool) ----
    var a_inner: _UVec[Int]
    var a_outer: _UVec[Int]
    var a_parent: _UVec[Int]
    var a_i2o: _UVec[CompressedEdge]  # inner -> outer
    var a_pedge: _UVec[CompressedEdge]  # inner -> parent's outer
    var a_children: List[_UVec[Int]]
    var a_mark: _UVec[Int]
    var a_free: _UVec[Int]
    var a_count: Int
    var stamp: Int
    # ---- event queue (binary min-heap of (time, code)) ----
    var hk: _UVec[Int]
    var hv: _UVec[Int]
    var now: Int
    # ---- scratch ----
    var area: _UVec[Int]
    var stk: _UVec[Int]
    var stk2: _UVec[Int]
    var path: _UVec[Int]
    var p1: _UVec[Int]
    var p2: _UVec[Int]
    var cyc_r: _UVec[Int]
    var cyc_e: _UVec[CompressedEdge]
    var grow_l: _UVec[Int]
    var shrink_l: _UVec[Int]
    var matched: List[CompressedEdge]  # filled by decode_to_edges
    var record: Bool
    # ---- statistics (cumulative over shots, except `weight`) ----
    var failures: Int  # unmatchable defects (odd component without boundary)
    var anomalies: Int  # internal invariant violations (QM-E712; must stay 0)
    var ignored: Int  # out-of-range / repeated detector ids skipped
    var weight: Int  # dual objective of the last decode (integer units)

    def __init__(out self, graph: MatchingGraph, num_distinct_weights: Int = SB_NUM_DISTINCT_WEIGHTS):
        var nd = graph.num_detectors
        self.graph = ArcPointer(SparseGraph(graph, num_distinct_weights))
        self.num_detectors = nd
        self.num_observables = graph.num_observables
        self.n_src = _UVec[Int](nd, -1)
        self.n_reg = _UVec[Int](nd, _EMPTY)
        self.n_top = _UVec[Int](nd, _EMPTY)
        self.n_arr = _UVec[Int](nd, 0)
        self.n_wrap = _UVec[Int](nd, 0)
        self.n_obs = _UVec[UInt64](nd, 0)
        self.n_qt = _UVec[Int](nd, 0)
        self.n_dt = _UVec[Int](nd, 0)
        self.n_qh = _UVec[Bool](nd, False)
        self.n_dh = _UVec[Bool](nd, False)
        self.n_touched = _UVec[Bool](nd, False)
        self.touched = _UVec[Int]()
        self.r_y = _UVec[Int]()
        self.r_s = _UVec[Int]()
        self.r_par = _UVec[Int]()
        self.r_alt = _UVec[Int]()
        self.r_match = _UVec[Int]()
        self.r_medge = _UVec[CompressedEdge]()
        self.r_shell = List[_UVec[Int]]()
        self.r_child = List[_UVec[Int]]()
        self.r_cedge = List[_UVec[CompressedEdge]]()
        self.r_qt = _UVec[Int]()
        self.r_dt = _UVec[Int]()
        self.r_qh = _UVec[Bool]()
        self.r_dh = _UVec[Bool]()
        self.r_live = _UVec[Bool]()
        self.r_free = _UVec[Int]()
        self.r_count = 0
        self.a_inner = _UVec[Int]()
        self.a_outer = _UVec[Int]()
        self.a_parent = _UVec[Int]()
        self.a_i2o = _UVec[CompressedEdge]()
        self.a_pedge = _UVec[CompressedEdge]()
        self.a_children = List[_UVec[Int]]()
        self.a_mark = _UVec[Int]()
        self.a_free = _UVec[Int]()
        self.a_count = 0
        self.stamp = 0
        self.hk = _UVec[Int]()
        self.hv = _UVec[Int]()
        self.now = 0
        self.area = _UVec[Int]()
        self.stk = _UVec[Int]()
        self.stk2 = _UVec[Int]()
        self.path = _UVec[Int]()
        self.p1 = _UVec[Int]()
        self.p2 = _UVec[Int]()
        self.cyc_r = _UVec[Int]()
        self.cyc_e = _UVec[CompressedEdge]()
        self.grow_l = _UVec[Int]()
        self.shrink_l = _UVec[Int]()
        self.matched = List[CompressedEdge]()
        self.record = False
        self.failures = 0
        self.anomalies = 0
        self.ignored = 0
        self.weight = 0
        # region 0: the _EMPTY sentinel (radius 0, frozen, never live)
        _ = self._new_region()
        self.r_live[_EMPTY] = False

    @staticmethod
    def from_dem(dem: DetectorErrorModel) -> SparseBlossomDecoder:
        return SparseBlossomDecoder(MatchingGraph(dem))

    def name(self) -> String:
        return "sparse_blossom"

    def weight_scale(self) -> Float64:
        """Integer units per nat before doubling (weights are 2*round(w*scale))."""
        return self.graph[].scale

    def last_weight(self) -> Int:
        """Dual objective of the last decode (sum of all region radii), in
        integer units; equals the matching weight when matching succeeded."""
        return self.weight

    # ------------------------------------------------------------------
    # public entry points
    # ------------------------------------------------------------------
    def decode(mut self, fired: List[Int]) -> UInt64:
        if len(fired) == 0:
            self.weight = 0
            return 0
        self.record = False
        self._run(fired)
        var obs = self._extract()
        self._reset()
        return obs

    def decode_to_edges(mut self, fired: List[Int]) -> List[CompressedEdge]:
        """The matching itself: one `CompressedEdge(a, b, obs)` per matched pair
        of detection events (b = -1: matched to the boundary), `obs` being the
        observables flipped by the matched path. XOR of all `obs` == decode().
        Unmatchable defects (see `failures`) are omitted."""
        self.matched.clear()
        if len(fired) == 0:
            self.weight = 0
            return List[CompressedEdge]()
        self.record = True
        self._run(fired)
        _ = self._extract()
        self._reset()
        self.record = False
        return self.matched.copy()

    def decode_to_matches(mut self, fired: List[Int]) -> List[Int]:
        """Matched detection-event pairs, flattened [a0, b0, a1, b1, ...]
        (b = -1: matched to the boundary). PyMatching's
        `decode_to_matched_dets_array`. Unmatchable defects are omitted."""
        var edges = self.decode_to_edges(fired)
        var out = List[Int](capacity=2 * len(edges))
        for e in edges:
            out.append(e.a)
            out.append(e.b)
        return out^

    # ------------------------------------------------------------------
    # event queue
    # ------------------------------------------------------------------
    @always_inline
    def _push(mut self, t: Int, code: Int):
        self.hk.append(t)
        self.hv.append(code)
        var i = len(self.hk) - 1
        while i > 0:
            var p = (i - 1) >> 1
            if self.hk[p] <= t:
                break
            self.hk[i] = self.hk[p]
            self.hv[i] = self.hv[p]
            i = p
        self.hk[i] = t
        self.hv[i] = code

    @always_inline
    def _pop(mut self) -> Tuple[Int, Int]:
        var t = self.hk[0]
        var c = self.hv[0]
        var lt = self.hk.pop()
        var lc = self.hv.pop()
        var n = len(self.hk)
        if n > 0:
            var i = 0
            while True:
                var l = 2 * i + 1
                if l >= n:
                    break
                var m = l
                if l + 1 < n and self.hk[l + 1] < self.hk[l]:
                    m = l + 1
                if self.hk[m] >= lt:
                    break
                self.hk[i] = self.hk[m]
                self.hv[i] = self.hv[m]
                i = m
            self.hk[i] = lt
            self.hv[i] = lc
        return (t, c)

    @always_inline
    def _node_desire(mut self, x: Int, t: Int):
        self.n_dh[x] = True
        self.n_dt[x] = t
        if not self.n_qh[x] or self.n_qt[x] > t:
            self.n_qt[x] = t
            self.n_qh[x] = True
            self._push(t, x << 1)

    @always_inline
    def _region_desire(mut self, r: Int, t: Int):
        self.r_dh[r] = True
        self.r_dt[r] = t
        if not self.r_qh[r] or self.r_qt[r] > t:
            self.r_qt[r] = t
            self.r_qh[r] = True
            self._push(t, (r << 1) | 1)

    @always_inline
    def _node_dequeue(mut self, x: Int, t: Int) -> Bool:
        """PyMatching's QueuedEventTracker: drop stale entries, re-queue when
        the desired time moved later, fire when queued == desired."""
        if not self.n_qh[x] or self.n_qt[x] != t:
            return False
        self.n_qh[x] = False
        if not self.n_dh[x]:
            return False
        if self.n_dt[x] != t:
            self.n_qt[x] = self.n_dt[x]
            self.n_qh[x] = True
            self._push(self.n_dt[x], x << 1)
            return False
        self.n_dh[x] = False
        return True

    def _region_dequeue(mut self, r: Int, t: Int) -> Bool:
        if r <= _EMPTY or r >= self.r_count or not self.r_live[r]:
            return False
        if not self.r_qh[r] or self.r_qt[r] != t:
            return False
        self.r_qh[r] = False
        if not self.r_dh[r]:
            return False
        if self.r_dt[r] != t:
            self.r_qt[r] = self.r_dt[r]
            self.r_qh[r] = True
            self._push(self.r_dt[r], (r << 1) | 1)
            return False
        self.r_dh[r] = False
        return True

    # ------------------------------------------------------------------
    # pools
    # ------------------------------------------------------------------
    def _new_region(mut self) -> Int:
        var r: Int
        if len(self.r_free) > 0:
            r = self.r_free.pop()
        else:
            r = self.r_count
            self.r_count += 1
            if r == len(self.r_y):
                self.r_y.append(0)
                self.r_s.append(0)
                self.r_par.append(-1)
                self.r_alt.append(-1)
                self.r_match.append(_NOMATCH)
                self.r_medge.append(CompressedEdge(-1, -1, 0))
                self.r_shell.append(_UVec[Int]())
                self.r_child.append(_UVec[Int]())
                self.r_cedge.append(_UVec[CompressedEdge]())
                self.r_qt.append(0)
                self.r_dt.append(0)
                self.r_qh.append(False)
                self.r_dh.append(False)
                self.r_live.append(False)
        self.r_y[r] = 0
        self.r_s[r] = 0
        self.r_par[r] = -1
        self.r_alt[r] = -1
        self.r_match[r] = _NOMATCH
        self.r_shell[r].clear()
        self.r_child[r].clear()
        self.r_cedge[r].clear()
        self.r_qh[r] = False
        self.r_dh[r] = False
        self.r_live[r] = True
        return r

    def _free_region(mut self, r: Int):
        self.r_live[r] = False
        self.r_qh[r] = False
        self.r_dh[r] = False
        self.r_free.append(r)

    def _new_alt(mut self) -> Int:
        var a: Int
        if len(self.a_free) > 0:
            a = self.a_free.pop()
        else:
            a = self.a_count
            self.a_count += 1
            if a == len(self.a_inner):
                self.a_inner.append(-1)
                self.a_outer.append(-1)
                self.a_parent.append(-1)
                self.a_i2o.append(CompressedEdge(-1, -1, 0))
                self.a_pedge.append(CompressedEdge(-1, -1, 0))
                self.a_children.append(_UVec[Int]())
                self.a_mark.append(0)
        self.a_inner[a] = -1
        self.a_outer[a] = -1
        self.a_parent[a] = -1
        self.a_children[a].clear()
        self.a_mark[a] = 0
        return a

    def _free_alt(mut self, a: Int):
        self.a_children[a].clear()
        self.a_parent[a] = -1
        self.a_free.append(a)

    def _remove_child(mut self, p: Int, c: Int):
        var n = len(self.a_children[p])
        for i in range(n):
            if self.a_children[p][i] == c:
                self.a_children[p][i] = self.a_children[p][n - 1]
                _ = self.a_children[p].pop()
                return
        self.anomalies += 1

    @always_inline
    def _touch(mut self, x: Int):
        if not self.n_touched[x]:
            self.n_touched[x] = True
            self.touched.append(x)

    # ------------------------------------------------------------------
    # flooder: regions growing on the graph
    # ------------------------------------------------------------------
    @always_inline
    def _rad(self, r: Int) -> Int:
        return self.r_y[r] + self.r_s[r] * self.now

    def _wrapped(self, x: Int) -> Int:
        var tot = 0
        var r = self.n_reg[x]
        var top = self.n_top[x]
        while r != top:
            tot += self.r_y[r]
            r = self.r_par[r]
        return tot - self.n_arr[x]

    def _collect_area(mut self, r: Int):
        """All nodes owned by region r and its blossom descendants -> self.area."""
        self.area.clear()
        self.stk.clear()
        self.stk.append(r)
        while len(self.stk) > 0:
            var q = self.stk.pop()
            for i in range(len(self.r_shell[q])):
                self.area.append(self.r_shell[q][i])
            for i in range(len(self.r_child[q])):
                self.stk.append(self.r_child[q][i])

    @always_inline
    def _scan(self, x: Int) -> _Scan:
        """Earliest (and second-earliest) interaction time at node x.

        Local radius of a node = y + s * t with (y, s) = (r_y[top] + wrap,
        r_s[top]); empty nodes belong to the frozen zero-radius _EMPTY
        region, so the neighbour loop is branch-free: a pair (x, v) of
        different owners collides when the gap w - y1 - y2 closes at rate
        s1 + s2 (2: both growing, 1: one growing; <= 0: never)."""
        ref G = self.graph[]
        var top = self.n_top[x]
        var s1 = self.r_s[top]
        if s1 < 0:
            return _Scan(_INF, -1, _INF)
        var y1 = self.r_y[top] + self.n_wrap[x]
        var best_t = _INF
        var best = -1
        var second = _INF
        if s1 > 0:
            var bw = G.bw[x]
            if bw >= 0:
                best_t = bw - y1
                best = _BND
        for a in range(G.off[x], G.off[x + 1]):
            var packed = G.adj[a]
            var v = packed & 0xFFFFFFFF
            var vt = self.n_top[v]
            var ss = s1 + self.r_s[vt]
            var t = ((packed >> 32) - y1 - self.r_y[vt] - self.n_wrap[v]) >> ((ss & 2) >> 1)
            var valid = (vt != top) & (ss >= 1)
            t = t if valid else _INF
            second = min(second, max(best_t, t))
            best = a if t < best_t else best
            best_t = min(best_t, t)
        if best_t == _INF:
            best = -1
        return _Scan(best_t, best, second)

    def _reschedule(mut self, x: Int):
        var ev = self._scan(x)
        if ev.slot == -1:
            self.n_dh[x] = False
        else:
            var t = ev.t
            if t < self.now:
                self.anomalies += 1
                t = self.now
            self._node_desire(x, t)

    def _reschedule_area(mut self, r: Int):
        self._collect_area(r)
        for i in range(len(self.area)):
            self._reschedule(self.area[i])

    def _schedule_shrink(mut self, r: Int):
        var t = self.r_y[r]
        if len(self.r_shell[r]) > 0:
            t += self.n_wrap[self.r_shell[r].last()]
        if t < self.now:
            self.anomalies += 1
            t = self.now
        self._region_desire(r, t)

    def _set_growing(mut self, r: Int):
        var v = self._rad(r)
        self.r_s[r] = 1
        self.r_y[r] = v - self.now
        self.r_dh[r] = False
        self._reschedule_area(r)

    def _set_frozen(mut self, r: Int):
        var was_shrinking = self.r_s[r] == -1
        var v = self._rad(r)
        self.r_s[r] = 0
        self.r_y[r] = v
        self.r_dh[r] = False
        # growing -> frozen only delays this region's events (lazy re-check)
        if was_shrinking:
            self._reschedule_area(r)

    def _set_shrinking(mut self, r: Int):
        var v = self._rad(r)
        self.r_s[r] = -1
        self.r_y[r] = v + self.now
        self._schedule_shrink(r)
        self._collect_area(r)
        for i in range(len(self.area)):
            self.n_dh[self.area[i]] = False

    def _arrive(mut self, r: Int, x: Int, frm: Int, eobs: UInt64):
        """Growing top-level region r reaches the empty node x from node frm."""
        self.n_obs[x] = self.n_obs[frm] ^ eobs
        self.n_src[x] = self.n_src[frm]
        var ra = self._rad(r)
        self.n_arr[x] = ra
        self.n_reg[x] = r
        self.n_top[x] = r
        self.n_wrap[x] = -ra
        self.r_shell[r].append(x)
        self._touch(x)
        self._reschedule(x)

    def _look_at_node(mut self, x: Int):
        var ev = self._scan(x)
        var slot = ev.slot
        if slot == -1:
            return
        if ev.t > self.now:
            self._node_desire(x, ev.t)
            return
        if ev.t < self.now:
            self.anomalies += 1
        # The other slots of x are unaffected by handling this one unless x's
        # own region changes state (then the handlers reschedule or only
        # delay), so the second-earliest time is a valid lower bound.
        if ev.second < _INF:
            self._node_desire(x, max(ev.second, self.now))
        if slot == _BND:
            var e = CompressedEdge(self.n_src[x], -1, self.n_obs[x] ^ self.graph[].bobs[x])
            self._on_hit_boundary(self.n_top[x], e)
            return
        var v = self.graph[].neighbour(slot)
        var eobs = self.graph[].aobs[slot]
        if self.n_top[v] == _EMPTY:
            self._arrive(self.n_top[x], v, x, eobs)
        elif self.n_top[x] == _EMPTY:
            self._arrive(self.n_top[v], x, v, eobs)
        else:
            var e = CompressedEdge(self.n_src[x], self.n_src[v], self.n_obs[x] ^ self.n_obs[v] ^ eobs)
            self._on_hit_region(self.n_top[x], self.n_top[v], e)

    def _on_region_event(mut self, r: Int):
        """Shrinking region r: give back a node, implode, or shatter."""
        var ns = len(self.r_shell[r])
        if ns == 0:
            self._shatter(r)
        elif ns == 1 and len(self.r_child[r]) == 0:
            self._implode(r)
        else:
            var x = self.r_shell[r].pop()
            self.n_src[x] = -1
            self.n_reg[x] = _EMPTY
            self.n_top[x] = _EMPTY
            self.n_arr[x] = 0
            self.n_wrap[x] = 0
            self.n_obs[x] = 0
            self._reschedule(x)
            self._schedule_shrink(r)

    # ------------------------------------------------------------------
    # matcher: alternating trees, augmentation, blossoms
    # ------------------------------------------------------------------
    def _match(mut self, r1: Int, r2: Int, var e: CompressedEdge):
        self.r_match[r1] = r2
        self.r_medge[r1] = e
        self.r_match[r2] = r1
        self.r_medge[r2] = e.reversed()

    def _lca(mut self, a1: Int, a2: Int) -> Int:
        """Lowest common ancestor of two tree nodes, -1 if in different trees."""
        self.stamp += 1
        var s = self.stamp
        var x = a1
        while x >= 0:
            self.a_mark[x] = s
            x = self.a_parent[x]
        x = a2
        while x >= 0:
            if self.a_mark[x] == s:
                return x
            x = self.a_parent[x]
        return -1

    def _become_root(mut self, a: Int):
        """Re-root a's tree at a; inner regions shift along the old root path."""
        self.path.clear()
        var x = a
        while x >= 0:
            self.path.append(x)
            x = self.a_parent[x]
        var j = len(self.path) - 2
        while j >= 0:
            var c = self.path[j]
            var p = self.path[j + 1]
            self._remove_child(p, c)
            self.a_children[c].append(p)
            self.a_inner[p] = self.a_inner[c]
            self.a_i2o[p] = self.a_pedge[c]
            self.a_pedge[p] = self.a_i2o[c]
            self.a_parent[p] = c
            self.r_alt[self.a_inner[p]] = p
            self.a_inner[c] = -1
            self.a_parent[c] = -1
            j -= 1

    def _shatter_tree(mut self, root: Int):
        """Dissolve a tree (rooted at `root`) into matched, frozen pairs. The
        root's outer region is left for the caller to match."""
        self.stk2.clear()
        for i in range(len(self.a_children[root])):
            self.stk2.append(self.a_children[root][i])
        self.r_alt[self.a_outer[root]] = -1
        self._free_alt(root)
        while len(self.stk2) > 0:
            var c = self.stk2.pop()
            var ri = self.a_inner[c]
            var ro = self.a_outer[c]
            self._match(ri, ro, self.a_i2o[c])
            self.r_alt[ri] = -1
            self.r_alt[ro] = -1
            for i in range(len(self.a_children[c])):
                self.stk2.append(self.a_children[c][i])
            self._free_alt(c)
            self._set_frozen(ri)
            self._set_frozen(ro)

    def _on_hit_boundary(mut self, r: Int, var e: CompressedEdge):
        var a = self.r_alt[r]
        if a < 0:
            self.anomalies += 1
            self._set_frozen(r)
            return
        self._become_root(a)
        self._shatter_tree(a)
        self.r_match[r] = _BND
        self.r_medge[r] = e
        self._set_frozen(r)

    def _on_hit_region(mut self, r1: Int, r2: Int, var e: CompressedEdge):
        var a1 = self.r_alt[r1]
        var a2 = self.r_alt[r2]
        if a1 >= 0 and a2 >= 0:
            var lca = self._lca(a1, a2)
            if lca >= 0:
                self._make_blossom(r1, r2, e, lca)
            else:
                self._become_root(a1)
                self._become_root(a2)
                self._shatter_tree(a1)
                self._shatter_tree(a2)
                self._match(r1, r2, e)
                self._set_frozen(r1)
                self._set_frozen(r2)
        elif a1 >= 0:
            self._tree_hit_matched(r1, r2, e)
        elif a2 >= 0:
            self._tree_hit_matched(r2, r1, e.reversed())
        else:
            self.anomalies += 1
            self._set_frozen(r1)
            self._set_frozen(r2)

    def _tree_hit_matched(mut self, rt: Int, m: Int, var e: CompressedEdge):
        """Growing tree region rt touches the matched region m."""
        var pm = self.r_match[m]
        var a = self.r_alt[rt]
        if pm == _BND:
            # augment: m leaves the boundary and matches rt
            self._become_root(a)
            self._shatter_tree(a)
            self._match(rt, m, e)
            self._set_frozen(rt)
        elif pm > _EMPTY:
            # grow the tree by the pair (m inner, pm outer)
            var c = self._new_alt()
            self.a_inner[c] = m
            self.a_outer[c] = pm
            self.a_i2o[c] = self.r_medge[m]
            self.a_parent[c] = a
            self.a_pedge[c] = e.reversed()
            self.a_children[a].append(c)
            self.r_alt[m] = c
            self.r_alt[pm] = c
            self.r_match[m] = _NOMATCH
            self.r_match[pm] = _NOMATCH
            self._set_shrinking(m)
            self._set_growing(pm)
        else:
            self.anomalies += 1
            self._set_frozen(rt)
            self._set_frozen(m)

    def _make_blossom(mut self, r1: Int, r2: Int, var e: CompressedEdge, lca: Int):
        """Tree hits itself: the odd cycle through the LCA becomes a blossom."""
        var a1 = self.r_alt[r1]
        var a2 = self.r_alt[r2]
        self.p1.clear()
        var x = a1
        while x != lca:
            self.p1.append(x)
            x = self.a_parent[x]
        self.p2.clear()
        x = a2
        while x != lca:
            self.p2.append(x)
            x = self.a_parent[x]
        self.cyc_r.clear()
        self.cyc_e.clear()
        var olca = self.a_outer[lca]
        var n1 = len(self.p1)
        var n2 = len(self.p2)
        # cycle order: O_lca, down path 1 to r1, across e to r2, up path 2
        self.cyc_r.append(olca)
        if n1 == 0:
            self.cyc_e.append(e)
        else:
            self.cyc_e.append(self.a_pedge[self.p1[n1 - 1]].reversed())
            var i = n1 - 1
            while i >= 0:
                var nd = self.p1[i]
                self.cyc_r.append(self.a_inner[nd])
                self.cyc_e.append(self.a_i2o[nd])
                self.cyc_r.append(self.a_outer[nd])
                if i > 0:
                    self.cyc_e.append(self.a_pedge[self.p1[i - 1]].reversed())
                else:
                    self.cyc_e.append(e)
                i -= 1
        for i in range(n2):
            var nd = self.p2[i]
            self.cyc_r.append(self.a_outer[nd])
            self.cyc_e.append(self.a_i2o[nd].reversed())
            self.cyc_r.append(self.a_inner[nd])
            self.cyc_e.append(self.a_pedge[nd])
        # off-cycle subtrees hang from the LCA (now the blossom's tree node)
        if n1 > 0:
            self._remove_child(lca, self.p1[n1 - 1])
        if n2 > 0:
            self._remove_child(lca, self.p2[n2 - 1])
        for side in range(2):
            var npath = n1 if side == 0 else n2
            for i in range(npath):
                var nd = self.p1[i] if side == 0 else self.p2[i]
                var skip = -1
                if i > 0:
                    skip = self.p1[i - 1] if side == 0 else self.p2[i - 1]
                for k in range(len(self.a_children[nd])):
                    var ch = self.a_children[nd][k]
                    if ch == skip:
                        continue
                    self.a_parent[ch] = lca
                    self.a_children[lca].append(ch)
                self._free_alt(nd)
        var b = self._new_region()
        self.r_y[b] = -self.now
        self.r_s[b] = 1
        for k in range(len(self.cyc_r)):
            var c = self.cyc_r[k]
            self.r_child[b].append(c)
            self.r_cedge[b].append(self.cyc_e[k])
            self.r_par[c] = b
            self.r_alt[c] = -1
            var v = self._rad(c)
            self.r_s[c] = 0
            self.r_y[c] = v
            self.r_dh[c] = False
        self.a_outer[lca] = b
        self.r_alt[b] = lca
        self._collect_area(b)
        for i in range(len(self.area)):
            var y = self.area[i]
            self.n_top[y] = b
            self.n_wrap[y] = self._wrapped(y)
        # only after every node knows its new owner (else neighbours inside
        # the blossom would look like foreign regions)
        for i in range(len(self.area)):
            self._reschedule(self.area[i])

    def _heir(self, det: Int, b: Int) -> Int:
        """Child of blossom b containing detection event det (-1: none)."""
        var r = self.n_reg[det]
        while r > _EMPTY and self.r_par[r] != b:
            r = self.r_par[r]
        return r if r > _EMPTY else -1

    def _implode(mut self, r: Int):
        """Trivial inner region at radius 0: its tree neighbours now touch."""
        var a = self.r_alt[r]
        if a < 0 or self.a_parent[a] < 0:
            self.anomalies += 1
            self._set_frozen(r)
            return
        var p = self.a_parent[a]
        var pe = self.a_pedge[a]
        var oe = self.a_i2o[a]
        self._on_hit_region(self.a_outer[p], self.a_outer[a], CompressedEdge(pe.b, oe.b, pe.obs ^ oe.obs))

    def _shatter(mut self, b: Int):
        """Inner blossom b reached radius 0: put the even path between its
        two tree edges back into the tree and match the rest of the cycle."""
        var a = self.r_alt[b]
        if a < 0 or self.a_parent[a] < 0:
            self.anomalies += 1
            self._set_frozen(b)
            return
        var p = self.a_parent[a]
        var pe = self.a_pedge[a]
        var oe = self.a_i2o[a]
        var hin = self._heir(pe.a, b)
        var hout = self._heir(oe.a, b)
        var n = len(self.r_child[b])
        self.cyc_r.clear()
        self.cyc_e.clear()
        var iin = -1
        var iout = -1
        for k in range(n):
            var c = self.r_child[b][k]
            self.cyc_r.append(c)
            self.cyc_e.append(self.r_cedge[b][k])
            if c == hin:
                iin = k
            if c == hout:
                iout = k
        if iin < 0 or iout < 0:
            self.anomalies += 1
            self._set_frozen(b)
            return
        # children become top-level regions again
        for k in range(n):
            self.r_par[self.cyc_r[k]] = -1
        for k in range(n):
            var c = self.cyc_r[k]
            self._collect_area(c)
            for i in range(len(self.area)):
                var y = self.area[i]
                self.n_top[y] = c
                self.n_wrap[y] = self._wrapped(y)
        self._remove_child(p, a)
        self.r_alt[b] = -1
        self._free_region(b)
        self.grow_l.clear()
        self.shrink_l.clear()
        self.stk2.clear()  # matched pairs (frozen; nodes need rescheduling)
        var cur_parent = p
        var cur_pedge = pe
        var gap = (iout - iin + n) % n
        if gap % 2 == 0:
            # walk forward iin -> iout (even length)
            var k = 0
            while k < gap:
                var ii = (iin + k) % n
                var io = (iin + k + 1) % n
                var nd = self._new_alt()
                self.a_inner[nd] = self.cyc_r[ii]
                self.a_outer[nd] = self.cyc_r[io]
                self.a_i2o[nd] = self.cyc_e[ii]
                self.a_parent[nd] = cur_parent
                self.a_pedge[nd] = cur_pedge
                self.a_children[cur_parent].append(nd)
                self.r_alt[self.cyc_r[ii]] = nd
                self.r_alt[self.cyc_r[io]] = nd
                self.shrink_l.append(self.cyc_r[ii])
                self.grow_l.append(self.cyc_r[io])
                cur_parent = nd
                cur_pedge = self.cyc_e[io].reversed()
                k += 2
            var j = (iout + 1) % n
            while j != iin:
                var j2 = (j + 1) % n
                self._match(self.cyc_r[j], self.cyc_r[j2], self.cyc_e[j])
                self.stk2.append(self.cyc_r[j])
                self.stk2.append(self.cyc_r[j2])
                j = (j2 + 1) % n
        else:
            # walk backward iin -> iout (even length n - gap)
            var back = n - gap
            var k = 0
            while k < back:
                var ii = ((iin - k) % n + n) % n
                var io = ((iin - k - 1) % n + n) % n
                var nd = self._new_alt()
                self.a_inner[nd] = self.cyc_r[ii]
                self.a_outer[nd] = self.cyc_r[io]
                self.a_i2o[nd] = self.cyc_e[io].reversed()
                self.a_parent[nd] = cur_parent
                self.a_pedge[nd] = cur_pedge
                self.a_children[cur_parent].append(nd)
                self.r_alt[self.cyc_r[ii]] = nd
                self.r_alt[self.cyc_r[io]] = nd
                self.shrink_l.append(self.cyc_r[ii])
                self.grow_l.append(self.cyc_r[io])
                cur_parent = nd
                cur_pedge = self.cyc_e[((io - 1) % n + n) % n]
                k += 2
            var j = (iin + 1) % n
            while j != iout:
                var j2 = (j + 1) % n
                self._match(self.cyc_r[j], self.cyc_r[j2], self.cyc_e[j])
                self.stk2.append(self.cyc_r[j])
                self.stk2.append(self.cyc_r[j2])
                j = (j2 + 1) % n
        # the original tree node keeps its outer region and subtree
        self.a_inner[a] = hout
        self.a_parent[a] = cur_parent
        self.a_pedge[a] = cur_pedge
        self.a_children[cur_parent].append(a)
        self.r_alt[hout] = a
        self.shrink_l.append(hout)
        for i in range(len(self.shrink_l)):
            self._set_shrinking(self.shrink_l[i])
        for i in range(len(self.grow_l)):
            self._set_growing(self.grow_l[i])
        for i in range(len(self.stk2)):
            self._reschedule_area(self.stk2[i])

    # ------------------------------------------------------------------
    # driver
    # ------------------------------------------------------------------
    def _run(mut self, fired: List[Int]):
        self.now = 0
        for i in range(len(fired)):
            var d = fired[i]
            # guard the unchecked node arrays: ids out of range or repeated
            # are ignored (and counted) instead of corrupting the state
            if d < 0 or d >= self.num_detectors or self.n_src[d] >= 0:
                self.ignored += 1
                continue
            var r = self._new_region()
            self.r_s[r] = 1
            self.r_shell[r].append(d)
            self.n_src[d] = d
            self.n_reg[d] = r
            self.n_top[d] = r
            self.n_arr[d] = 0
            self.n_wrap[d] = 0
            self.n_obs[d] = 0
            self._touch(d)
            var a = self._new_alt()
            self.a_outer[a] = r
            self.r_alt[r] = a
        for i in range(len(fired)):
            var d = fired[i]
            if d >= 0 and d < self.num_detectors:
                self._reschedule(d)
        while len(self.hk) > 0:
            var top = self._pop()
            var t = top[0]
            var code = top[1]
            if (code & 1) == 0:
                var x = code >> 1
                if not self._node_dequeue(x, t):
                    continue
                self.now = t
                self._look_at_node(x)
            else:
                var r = code >> 1
                if not self._region_dequeue(r, t):
                    continue
                self.now = t
                self._on_region_event(r)

    def _expand(mut self, b: Int, det: Int) -> UInt64:
        """Observable XOR of the internal matching of blossom b when its
        external match leaves through detection event det (-1: none)."""
        var n = len(self.r_child[b])
        if n == 0:
            return 0
        var obs: UInt64 = 0
        var i = 0
        if det >= 0:
            var h = self._heir(det, b)
            for k in range(n):
                if self.r_child[b][k] == h:
                    i = k
        var j = (i + 1) % n
        while j != i:
            var j2 = (j + 1) % n
            var e = self.r_cedge[b][j]
            obs ^= e.obs
            if self.record:
                self.matched.append(e)
            obs ^= self._expand(self.r_child[b][j], e.a)
            obs ^= self._expand(self.r_child[b][j2], e.b)
            j = (j2 + 1) % n
        obs ^= self._expand(self.r_child[b][i], det)
        return obs

    def _extract(mut self) -> UInt64:
        # trees that never found a partner (odd defect count in a component
        # without boundary): match their pairs, count the root as a failure
        for r in range(1, self.r_count):
            if self.r_live[r] and self.r_par[r] == -1 and self.r_alt[r] >= 0:
                var a = self.r_alt[r]
                if self.a_parent[a] < 0 and self.a_outer[a] == r:
                    self._shatter_tree(a)
                    self.r_match[r] = _UNMATCHABLE
                    self.failures += 1
        var w = 0
        for r in range(1, self.r_count):
            if self.r_live[r]:
                w += self._rad(r)
        self.weight = w
        var obs: UInt64 = 0
        for r in range(1, self.r_count):
            if not self.r_live[r] or self.r_par[r] != -1:
                continue
            var m = self.r_match[r]
            if m == _BND:
                var e = self.r_medge[r]
                obs ^= e.obs
                if self.record:
                    self.matched.append(CompressedEdge(e.a, -1, e.obs))
                if len(self.r_child[r]) > 0:
                    obs ^= self._expand(r, e.a)
            elif m > r:
                var e = self.r_medge[r]
                obs ^= e.obs
                if self.record:
                    self.matched.append(e)
                if len(self.r_child[r]) > 0:
                    obs ^= self._expand(r, e.a)
                if len(self.r_child[m]) > 0:
                    obs ^= self._expand(m, e.b)
            elif m == _UNMATCHABLE:
                obs ^= self._expand(r, -1)
            elif m == _NOMATCH:
                self.anomalies += 1
        return obs

    def _reset(mut self):
        for i in range(len(self.touched)):
            var x = self.touched[i]
            self.n_src[x] = -1
            self.n_reg[x] = _EMPTY
            self.n_top[x] = _EMPTY
            self.n_arr[x] = 0
            self.n_wrap[x] = 0
            self.n_obs[x] = 0
            self.n_qh[x] = False
            self.n_dh[x] = False
            self.n_touched[x] = False
        self.touched.clear()
        for r in range(1, self.r_count):
            self.r_live[r] = False
        self.r_count = 1
        self.r_free.clear()
        self.a_count = 0
        self.a_free.clear()
        self.hk.clear()
        self.hv.clear()
        self.now = 0


# ---------------------------------------------------------------------------
# Verification oracle
# ---------------------------------------------------------------------------
struct MatchingOracle(Movable):
    """Exact reference on the *same* integer weights as SparseBlossomDecoder:
    integer all-pairs shortest paths (Dijkstra from every node) + the dense
    exact matcher of `qumo.qec.decoders` (subset DP / Edmonds blossom).

    Shortest paths are computed on the product graph (node, observable mask)
    restricted to the low `min(num_observables, 2)` observable bits, so the
    oracle can also tell whether a matched pair's observable mask is the mask
    of *some* shortest path between its endpoints (`path_ok`): matching
    weight == optimum plus `path_ok` on every pair proves that a decoder's
    prediction is the observable of a minimum-weight correction. O(n^2)
    memory: meant for tests and benchmarks on graphs <= a few thousand
    detectors."""

    var n: Int
    var nstates: Int  # observable masks tracked per node (1, 2 or 4)
    var dist: List[Int]  # (n + 1) x (n + 1); index n = boundary; -1 = unreachable
    var pdist: List[Int]  # (n + 1) x (n + 1) x nstates; -1 = unreachable

    def __init__(out self, g: MatchingGraph, num_distinct_weights: Int = SB_NUM_DISTINCT_WEIGHTS):
        var nn = g.num_nodes()
        self.n = g.num_detectors
        var bits = min(g.num_observables, 2)
        var S = 1 << bits
        var smask = UInt64(S - 1)
        self.nstates = S
        var wi = discretize_weights(g, num_distinct_weights)
        self.dist = List[Int](length=nn * nn, fill=-1)
        self.pdist = List[Int](length=nn * nn * S, fill=-1)
        var d = List[Int](length=nn * S, fill=-1)
        var hk = List[Int]()
        var hv = List[Int]()
        for src in range(nn):
            for v in range(nn * S):
                d[v] = -1
            hk.clear()
            hv.clear()
            d[src * S] = 0
            _hpush(hk, hv, 0, src * S)
            while len(hk) > 0:
                var top = _hpop(hk, hv)
                var dv = top[0]
                var st = top[1]
                if dv > d[st]:
                    continue
                var v = st // S
                var m = st % S
                self.pdist[(src * nn + v) * S + m] = dv
                if self.dist[src * nn + v] < 0:
                    self.dist[src * nn + v] = dv  # first settled state = shortest
                for a in range(g.adj_off[v], g.adj_off[v + 1]):
                    var e = g.adj_edge[a]
                    var u = g.other(e, v)
                    var mu = m ^ Int(g.eobs[e] & smask)
                    var su = u * S + mu
                    var nd = dv + wi[e]
                    if d[su] < 0 or nd < d[su]:
                        d[su] = nd
                        _hpush(hk, hv, nd, su)

    def path_ok(self, u: Int, v: Int, obs: UInt64) -> Bool:
        """True if some shortest u-v path (v = -1: boundary) flips exactly the
        tracked low observable bits of `obs`."""
        var nn = self.n + 1
        var vv = self.n if v < 0 else v
        var m = Int(obs & UInt64(self.nstates - 1))
        var best = self.dist[u * nn + vv]
        return best >= 0 and self.pdist[(u * nn + vv) * self.nstates + m] == best

    def distance(self, u: Int, v: Int) -> Int:
        """Shortest integer distance (v = -1: boundary); -1 if unreachable."""
        var nn = self.n + 1
        var vv = self.n if v < 0 else v
        return self.dist[u * nn + vv]

    def pairs_weight(self, pairs: List[Int]) -> Int:
        """Total shortest-path weight of a flattened pair list."""
        var w = 0
        for i in range(0, len(pairs), 2):
            w += self.distance(pairs[i], pairs[i + 1])
        return w

    def optimal_weight(self, fired: List[Int]) -> Int:
        """Exact MWPM weight of the syndrome (dense blossom / subset DP)."""
        var k = len(fired)
        if k == 0:
            return 0
        var dist = List[Float64](length=k * k, fill=1e300)
        var bd = List[Float64](length=k, fill=1e300)
        for i in range(k):
            var b = self.distance(fired[i], -1)
            if b >= 0:
                bd[i] = Float64(b)
            for j in range(k):
                if i != j:
                    var dd = self.distance(fired[i], fired[j])
                    if dd >= 0:
                        dist[i * k + j] = Float64(dd)
        var partner = min_weight_boundary_matching(k, dist, bd)
        var w = 0
        for i in range(k):
            var j = partner[i]
            if j < 0:
                w += self.distance(fired[i], -1)
            elif j > i:
                w += self.distance(fired[i], fired[j])
        return w


def _hpush(mut hk: List[Int], mut hv: List[Int], t: Int, v: Int):
    hk.append(t)
    hv.append(v)
    var i = len(hk) - 1
    while i > 0:
        var p = (i - 1) >> 1
        if hk[p] <= t:
            break
        hk[i] = hk[p]
        hv[i] = hv[p]
        i = p
    hk[i] = t
    hv[i] = v


def _hpop(mut hk: List[Int], mut hv: List[Int]) -> Tuple[Int, Int]:
    var t = hk[0]
    var c = hv[0]
    var lt = hk.pop()
    var lc = hv.pop()
    var n = len(hk)
    if n > 0:
        var i = 0
        while True:
            var l = 2 * i + 1
            if l >= n:
                break
            var m = l
            if l + 1 < n and hk[l + 1] < hk[l]:
                m = l + 1
            if hk[m] >= lt:
                break
            hk[i] = hk[m]
            hv[i] = hv[m]
            i = m
        hk[i] = lt
        hv[i] = lc
    return (t, c)
