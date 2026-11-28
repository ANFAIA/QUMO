"""Exact circuit distance of a detector error model (weight-bounded search, arXiv:2611.08247).

`circuit_distance(dem)` returns the fewest error mechanisms whose detectors cancel and whose
observables do not: min |F| with D 1_F = 0 and O 1_F != 0 over GF(2), on the full symptoms (`dets`),
hyperedges kept whole. Method of Cheng, Guo and Pan (SI Sec. 5.1): iterative deepening over a weight
threshold K; roots anchored at the smallest column of the support; branch on an unsatisfied detector;
prune by the LP-dual covering bound sum_r 1/m_r, computed in integers so rounding only weakens it;
close a node whose syndrome is zero (Lemma 5.1). Columns that flip an observable come first, so only
they are anchors (every solution contains one).

A result is exact only when a completed threshold search meets a witness of the same weight; the
witness is re-checked by `verify_witness` before it is returned. A node budget that runs out inside
threshold K returns `status = "budget"` and `lower = K`, never an upper bound. Cost grows quickly with
the distance: surface-code memory d = 3 and 5 (full DEM, both bases' detectors) are fast; d = 7 is not
in reach of this minimal search.

`circuit_distance_reduced(dem)` (and `circuit_distance_with_sectors(dem, sectors)`) returns the same
quantity with the same semantics, by the paper's reductions (Methods, "Reductions, relaxation and
lifting"; SI Sec. 5.4): a LOWER bound from row projections onto detector sectors (any sector is
sound: a solution of the model solves its projection), solved exactly by a shortest odd closed walk
when the projection is graph-like, else by the search; an UPPER bound by lifting an optimal projected
solution to the model and replaying it with `verify_witness`; and, when no lift is found, the search
on the full model from threshold K = lower bound. Exact only when the two meet. Surface-code memory
d = 7 .. 15 is certified in well under a second. `detector_sectors` (a heuristic: it changes how strong
the bound is, never whether it holds) and `projected_distance` (the projected optimum alone, a lower
bound) are public. Not implemented: rank DP, time collapse, completion tables, symmetry.
"""

from std.bit import pop_count
from qumo.qec.dem import DetectorErrorModel, DemError

comptime _SCALE = 720720  # lcm(1..16): floor(_SCALE / m) <= _SCALE / m, so the bound stays admissible


struct DistanceResult(Copyable, Writable):
    """Outcome of `circuit_distance`. `lower` is always a proven lower bound; `upper` is the weight
    of a verified witness, -1 when none was found. status: "exact" | "budget" | "above_max_weight" |
    "no_logical".

    * "exact": lower == upper == len(witness) == Delta.
    * "budget": node_limit was reached inside threshold K. lower = K, upper = -1, witness empty.
    * "above_max_weight": every threshold up to min(max_weight, number of merged columns) finished
      with no solution. lower = max_weight + 1, upper = -1 (Delta > max_weight, possibly infinite).
    * "no_logical": no merged column flips an observable, so no logical error exists at all.
      lower = max_weight + 1, upper = -1, nodes = 0.
    """

    var lower: Int
    var upper: Int
    var exact: Bool  # upper >= 0 and lower == upper
    var witness: List[Int]  # indices into dem.errors, in search order; empty unless exact
    var nodes: Int  # search nodes visited, all thresholds
    var status: String

    def __init__(out self, lower: Int, upper: Int, var witness: List[Int], nodes: Int, status: String):
        self.lower = lower
        self.upper = upper
        self.exact = upper >= 0 and lower == upper
        self.witness = witness^
        self.nodes = nodes
        self.status = status

    def write_to(self, mut w: Some[Writer]):
        w.write("DistanceResult(status=", self.status, ", lower=", self.lower, ", upper=", self.upper,
                ", weight=", len(self.witness), ", nodes=", self.nodes, ")")


def _canon(ids: List[Int]) -> List[Int]:
    """Sorted ids that occur an odd number of times (D3 D3 cancels)."""
    var s = ids.copy()
    sort(s)
    var out = List[Int]()
    var i = 0
    while i < len(s):
        var j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        if ((j - i) & 1) == 1:
            out.append(s[i])
        i = j
    return out^


def _get(v: List[UInt64], base: Int, i: Int) -> Bool:
    return ((v[base + (i >> 6)] >> UInt64(i & 63)) & 1) != 0


def _toggle(mut v: List[UInt64], base: Int, i: Int):
    v[base + (i >> 6)] ^= UInt64(1) << UInt64(i & 63)


def _clear(mut v: List[UInt64], base: Int, i: Int):
    v[base + (i >> 6)] &= ~(UInt64(1) << UInt64(i & 63))


def _arg_error(what: String) -> Error:
    return Error("QM-E740: circuit_distance " + what + ". Hint: max_weight >= 1, node_limit >= 1, at most 64 observables, detector ids >= 0, and `^` pieces that XOR to the full symptom.")


struct _Search:
    """Weight-bounded search state. All buffers are allocated in `__init__`; `_dfs` allocates nothing.

    Invariants (paper SI Sec. 5.1):
    * I1 (search tree): at a node, A = chosen[0..depth-1], syn[depth] = XOR of their detectors,
      lab[depth] = XOR of their observables, avail[depth] = U. A and U are disjoint and every j in U
      is > chosen[0].
    * I2 (completeness, Thm 5.2): if a minimum support F with |F| <= K and min F = a exists, then on
      one path from root(a): A is a subset of F and F \\ A is a subset of U.
    * I3 (admissible bound): y_r = 1/m_r is dual feasible, so any completion has size >= sum_r 1/m_r.
      m_r is the MAX over available columns; a MIN would be inadmissible.
    * I4: child buffers at depth + 1 are written only by the parent at depth; so avail[depth + 1]
      keeps its cumulative exclusions across siblings.
    * I5 (anchors): every solution contains a column with obs != 0, which are columns 0..n_log-1.
    """

    var n: Int  # columns
    var W: Int  # words per detector bitset
    var NW: Int  # words per column bitset
    var col_off: List[Int]  # CSR column -> detectors: col_det[col_off[j] : col_off[j+1]]
    var col_det: List[Int]
    var col_obs: List[UInt64]
    var det_off: List[Int]  # CSR detector -> columns, ascending column id
    var det_col: List[Int]
    var syn: List[UInt64]  # (kmax + 1) * W: syndrome at depth d lives at d * W
    var lab: List[UInt64]  # kmax + 1: observable mask at depth d
    var avail: List[UInt64]  # (kmax + 1) * NW: columns still allowed at depth d, at d * NW
    var chosen: List[Int]  # kmax + 1: chosen[t] = column picked at |A| = t + 1
    var nodes: Int
    var node_limit: Int
    var out_of_budget: Bool
    var found_depth: Int

    def __init__(out self, n: Int, nd: Int, col_dets: List[List[Int]], col_obs: List[UInt64], kmax: Int, node_limit: Int):
        var W = (nd + 63) >> 6
        if W < 1:
            W = 1
        var NW = (n + 63) >> 6
        if NW < 1:
            NW = 1
        var col_off = List[Int](length=n + 1, fill=0)
        var col_det = List[Int]()
        var cnt = List[Int](length=nd, fill=0)
        for j in range(n):
            col_off[j] = len(col_det)
            for d in col_dets[j]:
                col_det.append(d)
                cnt[d] += 1
        col_off[n] = len(col_det)

        var det_off = List[Int](length=nd + 1, fill=0)
        for d in range(nd):
            det_off[d + 1] = det_off[d] + cnt[d]
        var det_col = List[Int](length=len(col_det), fill=0)
        var fillp = List[Int](length=nd, fill=0)
        for d in range(nd):
            fillp[d] = det_off[d]
        for j in range(n):
            for u in range(col_off[j], col_off[j + 1]):
                var d = col_det[u]
                det_col[fillp[d]] = j
                fillp[d] += 1

        self.n = n
        self.W = W
        self.NW = NW
        self.col_off = col_off^
        self.col_det = col_det^
        self.col_obs = col_obs.copy()
        self.det_off = det_off^
        self.det_col = det_col^
        self.syn = List[UInt64](length=(kmax + 1) * W, fill=UInt64(0))
        self.lab = List[UInt64](length=kmax + 1, fill=UInt64(0))
        self.avail = List[UInt64](length=(kmax + 1) * NW, fill=UInt64(0))
        self.chosen = List[Int](length=kmax + 1, fill=0)
        self.nodes = 0
        self.node_limit = node_limit
        self.out_of_budget = False
        self.found_depth = 0

    def root(mut self, a: Int):
        """A = {a}, U = {a+1, ..., n-1}."""
        self.chosen[0] = a
        for w in range(self.W):
            self.syn[self.W + w] = 0
        for u in range(self.col_off[a], self.col_off[a + 1]):
            _toggle(self.syn, self.W, self.col_det[u])
        self.lab[1] = self.col_obs[a]
        for w in range(self.NW):
            self.avail[self.NW + w] = 0
        for j in range(a + 1, self.n):
            _toggle(self.avail, self.NW, j)

    def _dfs(mut self, depth: Int, K: Int) -> Bool:
        """Node at |A| = depth (1..K). True when a solution was found (found_depth, chosen set)."""
        self.nodes += 1
        if self.nodes > self.node_limit:
            self.out_of_budget = True
            return False
        var sb = depth * self.W
        var zero = True
        for w in range(self.W):
            if self.syn[sb + w] != 0:
                zero = False
                break
        if zero:  # zero syndrome: a solution iff observables flip, else close (Lemma 5.1)
            if self.lab[depth] != 0:
                self.found_depth = depth
                return True
            return False
        if depth >= K:
            return False
        var ab = depth * self.NW
        var tot = 0
        var best_r = -1
        var best_cnt = self.n + 1
        for w in range(self.W):  # every unsatisfied detector r (set Q)
            var x = self.syn[sb + w]
            while x != 0:
                var low = x & (~x + 1)
                var r = w * 64 + Int(pop_count(low - 1))
                x ^= low
                var mx = 0
                var cnt = 0
                for t in range(self.det_off[r], self.det_off[r + 1]):
                    var j = self.det_col[t]
                    if not _get(self.avail, ab, j):
                        continue
                    cnt += 1
                    var c = 0
                    for u in range(self.col_off[j], self.col_off[j + 1]):
                        if _get(self.syn, sb, self.col_det[u]):
                            c += 1
                    if c > mx:
                        mx = c
                if mx == 0:  # r can never be fixed
                    return False
                tot += _SCALE // mx  # floor: only weakens the bound
                if cnt < best_cnt:  # strict <: ties keep the smallest r
                    best_cnt = cnt
                    best_r = r
        var lb = (tot + _SCALE - 1) // _SCALE
        if depth + lb > K:
            return False
        var cb = (depth + 1) * self.W
        var nb = (depth + 1) * self.NW
        for q in range(self.NW):  # copy once; then shrink cumulatively
            self.avail[nb + q] = self.avail[ab + q]
        for t in range(self.det_off[best_r], self.det_off[best_r + 1]):
            var j = self.det_col[t]
            if not _get(self.avail, ab, j):  # test the PARENT's avail
                continue
            _clear(self.avail, nb, j)  # child excludes j_1..j_p
            self.chosen[depth] = j
            for w in range(self.W):
                self.syn[cb + w] = self.syn[sb + w]
            for u in range(self.col_off[j], self.col_off[j + 1]):
                _toggle(self.syn, cb, self.col_det[u])
            self.lab[depth + 1] = self.lab[depth] ^ self.col_obs[j]
            if self._dfs(depth + 1, K):
                return True
            if self.out_of_budget:
                return False
        return False


def verify_witness(dem: DetectorErrorModel, witness: List[Int]) -> Bool:
    """True when `witness` (indices into dem.errors) is nonempty, its indices are distinct and in range,
    its mechanisms' detectors cancel (each id an even number of times over all `dets`) and their
    observables XOR to a nonzero mask. Independent of the search: plain counting, no bitsets."""
    if len(witness) == 0:
        return False
    var idx = witness.copy()
    sort(idx)
    for t in range(len(idx)):
        if idx[t] < 0 or idx[t] >= len(dem.errors):
            return False
        if t > 0 and idx[t] == idx[t - 1]:
            return False
    var dets_all = List[Int]()
    var o: UInt64 = 0
    for i in witness:
        for d in dem.errors[i].dets:
            dets_all.append(d)
        o ^= dem.errors[i].obs
    if len(_canon(dets_all)) != 0:
        return False
    return o != 0


struct _Cols(Movable):
    """Merged columns in search order: logical columns first, then by detector count, then lexicographic.

    members[k] lists the tie keys of every input folded into column k (the first is tie[k]); tie keys are
    dem.errors indices for the full model and full-column indices for a projection."""

    var dets: List[List[Int]]
    var obs: List[UInt64]
    var tie: List[Int]
    var members: List[List[Int]]
    var n_log: Int

    def __init__(out self):
        self.dets = List[List[Int]]()
        self.obs = List[UInt64]()
        self.tie = List[Int]()
        self.members = List[List[Int]]()
        self.n_log = 0


def _sort_merge(syms: List[List[Int]], obs: List[UInt64], tie: List[Int]) -> _Cols:
    """Sort (logical first, fewer detectors, lexicographic, obs, tie) and merge equal (dets, obs) pairs."""
    var m = len(syms)
    var order = List[Int](capacity=m)
    for j in range(m):
        order.append(j)

    def less(a: Int, b: Int) {imm syms, imm obs, imm tie} -> Bool:
        if (obs[a] != 0) != (obs[b] != 0):  # logical columns first
            return obs[a] != 0
        var la = len(syms[a])
        var lb = len(syms[b])
        if la != lb:  # fewer detectors first
            return la < lb
        for t in range(la):
            if syms[a][t] != syms[b][t]:
                return syms[a][t] < syms[b][t]
        if obs[a] != obs[b]:
            return obs[a] < obs[b]
        return tie[a] < tie[b]

    sort(Span(order), less)

    # Merge equal columns (adjacent after sorting): two equal columns XOR to nothing.
    var c = _Cols()
    for t in range(m):
        var j = order[t]
        var nk = len(c.dets)
        if nk > 0 and c.obs[nk - 1] == obs[j] and c.dets[nk - 1] == syms[j]:
            c.members[nk - 1].append(tie[j])
            continue
        c.dets.append(syms[j].copy())
        c.obs.append(obs[j])
        c.tie.append(tie[j])
        var mem = List[Int]()
        mem.append(tie[j])
        c.members.append(mem^)
        if obs[j] != 0:
            c.n_log += 1
    return c^


def _check_args(dem: DetectorErrorModel, max_weight: Int, node_limit: Int) raises:
    if max_weight < 1:
        raise _arg_error("got max_weight=" + String(max_weight))
    if node_limit < 1:
        raise _arg_error("got node_limit=" + String(node_limit))
    if dem.num_observables > 64:
        raise _arg_error("got " + String(dem.num_observables) + " observables")


def _prepare(dem: DetectorErrorModel, mut nd: Int) raises -> _Cols:
    """Canonical symptoms, piece check, drop no-effect mechanisms, sort and merge. Sets nd to the
    number of detector rows (at least 1)."""
    var syms = List[List[Int]]()
    var obs = List[UInt64]()
    var orig = List[Int]()
    var max_id = -1
    for i in range(len(dem.errors)):
        ref e = dem.errors[i]
        var s = _canon(e.dets)
        for d in s:
            if d < 0:
                raise _arg_error("got detector id " + String(d) + " in error " + String(i))
        if e.num_components() > 0:
            var cat = List[Int]()
            var x: UInt64 = 0
            for c in range(e.num_components()):
                for d in e.comp_dets[c]:
                    cat.append(d)
                x ^= e.comp_obs[c]
            if _canon(cat) != s or x != e.obs:
                raise _arg_error("got error " + String(i) + " whose ^ pieces do not XOR to its symptom")
        if len(s) == 0 and e.obs == 0:
            continue
        for d in s:
            if d > max_id:
                max_id = d
        syms.append(s^)
        obs.append(e.obs)
        orig.append(i)
    nd = dem.num_detectors
    if max_id + 1 > nd:
        nd = max_id + 1
    if nd < 1:
        nd = 1
    return _sort_merge(syms, obs, orig)


struct _Found(Movable):
    """Outcome of one iterative-deepening run: status "exact" (chosen holds the solution's columns),
    "budget" (lower = threshold that was interrupted) or "above_max_weight"."""

    var status: String
    var lower: Int
    var chosen: List[Int]
    var nodes: Int

    def __init__(out self, status: String, lower: Int, var chosen: List[Int], nodes: Int):
        self.status = status
        self.lower = lower
        self.chosen = chosen^
        self.nodes = nodes


def _deepen(c: _Cols, nd: Int, max_weight: Int, node_limit: Int, start_k: Int) -> _Found:
    """Weight-bounded search, thresholds K = start_k .. min(max_weight, columns).

    start_k > 1 is sound only when Delta >= start_k is already proven (every threshold below it would
    be negative); then the first threshold with a solution is still the minimum."""
    var n = len(c.dets)
    var kmax = min(max_weight, n)
    if node_limit < 1:
        return _Found("budget", start_k, List[Int](), 0)
    var s = _Search(n, nd, c.dets, c.obs, kmax, node_limit)
    for K in range(start_k, kmax + 1):
        for a in range(c.n_log):
            s.root(a)
            if s._dfs(1, K):
                var fd = s.found_depth
                var chosen = List[Int]()
                for t in range(fd):
                    chosen.append(s.chosen[t])
                return _Found("exact", fd, chosen^, s.nodes)
            if s.out_of_budget:
                return _Found("budget", K, List[Int](), s.nodes)
    return _Found("above_max_weight", max_weight + 1, List[Int](), s.nodes)


def circuit_distance(dem: DetectorErrorModel, max_weight: Int = 32, node_limit: Int = 50_000_000) raises -> DistanceResult:
    """Exact circuit distance of `dem` by weight-bounded search (arXiv:2611.08247, SI Sec. 5.1).

    Thresholds K = 1 .. min(max_weight, columns), always from 1: no caller claim is needed, so
    every `lower` is proven. `node_limit` bounds the search nodes over all
    thresholds; running out is not an error (status "budget", lower = the threshold being searched).
    Pass `detector_error_model(c, decompose=False)`; a decomposed model gives the same value.
    For circuits past distance 5, `circuit_distance_reduced` adds a row-projection lower bound.
    """
    _check_args(dem, max_weight, node_limit)
    var nd = 1
    var c = _prepare(dem, nd)
    if c.n_log == 0:
        return DistanceResult(max_weight + 1, -1, List[Int](), 0, "no_logical")
    var f = _deepen(c, nd, max_weight, node_limit, 1)
    if f.status == "exact":
        var witness = List[Int]()
        for j in f.chosen:
            witness.append(c.tie[j])
        if not verify_witness(dem, witness):
            raise Error("QM-E740: internal: circuit_distance found a witness that does not verify. Hint: report this detector error model.")
        return DistanceResult(f.lower, f.lower, witness^, f.nodes, "exact")
    return DistanceResult(f.lower, -1, List[Int](), f.nodes, f.status)


# ---------------------------------------------------------------------------------------------
# Row projection (arXiv:2611.08247, Methods "Reductions, relaxation and lifting"; SI Sec. 5.4).
#
# Keep only the detector rows in a set S. Every e with D e = 0 and O e != 0 also has D_S e = 0 and
# O e != 0, so the feasible set only grows: Delta(D_S, O) <= Delta(D, O). The projected optimum is a
# LOWER bound on the circuit distance, whatever S is; S only decides how strong it is.
# A projected solution says nothing about the original model until it is lifted: one original column
# per projected column, chosen so that the detectors outside S cancel too. A lifted set of weight
# equal to the lower bound, re-checked by `verify_witness` on the original model, is the UPPER bound
# that makes the result exact. Otherwise only the full search on the original model reports exact.
# ---------------------------------------------------------------------------------------------


def _sector_error(what: String) -> Error:
    return Error("QM-E741: circuit_distance sector " + what + ". Hint: a sector is a nonempty list of detector ids of the model: below num_detectors, or up to the largest id a mechanism names (see detector_sectors).")


def _project(full: _Cols, keep: List[Bool]) -> _Cols:
    """Row projection of the merged full model onto the detectors with keep[d]. A projected column
    whose detectors all vanish is dropped only when it flips no observable (it is then the zero
    column); one that flips an observable stays, as a weight-1 logical of the projection."""
    var syms = List[List[Int]]()
    var obs = List[UInt64]()
    var tie = List[Int]()
    for k in range(len(full.dets)):
        var p = List[Int]()
        for d in full.dets[k]:
            if keep[d]:
                p.append(d)
        if len(p) == 0 and full.obs[k] == 0:
            continue
        syms.append(p^)
        obs.append(full.obs[k])
        tie.append(k)
    return _sort_merge(syms, obs, tie)


struct _GraphSolve(Movable):
    """Shortest odd closed walk on a graph-like projection.

    status "exact": value = Delta of the projection, witnesses = distinct optimal column sets (at most
    _MAX_WITNESSES); "none": no column set flips an observable (Delta = infinity); "budget"."""

    var status: String
    var value: Int
    var witnesses: List[List[Int]]
    var nodes: Int

    def __init__(out self):
        self.status = "none"
        self.value = -1
        self.witnesses = List[List[Int]]()
        self.nodes = 0


comptime _MAX_WITNESSES = 64


def _graph_solve(c: _Cols, nd: Int, node_limit: Int) -> _GraphSolve:
    """Exact Delta of a model whose columns all have at most 2 detectors.

    Vertices: the nd detectors and a boundary vertex B = nd; a 1-detector column is an edge to B, a
    2-detector column an edge between its detectors. B is unconstrained, but in any F with D F = 0
    every detector has even degree, so B does too (degrees sum to 2|F|): F is an even subgraph.
    For one observable bit l, label each edge with bit l of its obs. Claim: the fewest columns with
    D F = 0 and bit l of O F odd equals the shortest closed walk with an odd label sum.
    (<=) a closed walk, with edges counted mod 2, is an even subgraph with the walk's label parity
    and at most its length. (>=) an even subgraph splits into edge-disjoint cycles; the label sum is
    odd, so one cycle is odd, and a cycle is a closed walk no longer than F.
    The shortest odd closed walk through v is the BFS distance from (v, 0) to (v, 1) in the parity
    double cover; minimise over v and over the bits l (O F != 0 iff some bit is odd).
    """
    var r = _GraphSolve()
    var n = len(c.dets)
    for j in range(n):  # a column with no detector that flips an observable: Delta = 1
        if len(c.dets[j]) == 0 and c.obs[j] != 0:
            r.status = "exact"
            r.value = 1
            var w = List[Int]()
            w.append(j)
            r.witnesses.append(w^)
            return r^
    var V = nd + 1
    var deg = List[Int](length=V + 1, fill=0)
    for j in range(n):
        if len(c.dets[j]) == 1:
            deg[c.dets[j][0]] += 1
            deg[nd] += 1
        elif len(c.dets[j]) == 2:
            deg[c.dets[j][0]] += 1
            deg[c.dets[j][1]] += 1
    var off = List[Int](length=V + 1, fill=0)
    for v in range(V):
        off[v + 1] = off[v] + deg[v]
    var nbr = List[Int](length=off[V], fill=0)
    var ecol = List[Int](length=off[V], fill=0)
    var fillp = List[Int](length=V, fill=0)
    for v in range(V):
        fillp[v] = off[v]
    for j in range(n):
        var a = c.dets[j][0]
        var b = nd
        if len(c.dets[j]) == 2:
            b = c.dets[j][1]
        nbr[fillp[a]] = b
        ecol[fillp[a]] = j
        fillp[a] += 1
        nbr[fillp[b]] = a
        ecol[fillp[b]] = j
        fillp[b] += 1
    var any_obs: UInt64 = 0
    for j in range(n):
        any_obs |= c.obs[j]
    var best = -1
    var dist = List[Int](length=2 * V, fill=-1)
    var pstate = List[Int](length=2 * V, fill=-1)
    var pcol = List[Int](length=2 * V, fill=-1)
    var touched = List[Int]()
    var queue = List[Int](capacity=2 * V)
    var inset = List[Bool](length=n, fill=False)
    for l in range(64):
        if ((any_obs >> UInt64(l)) & 1) == 0:
            continue
        for v in range(V):
            if deg[v] == 0:
                continue
            for s in touched:
                dist[s] = -1
            touched.clear()
            queue.clear()
            var src = 2 * v
            var dst = 2 * v + 1
            dist[src] = 0
            touched.append(src)
            queue.append(src)
            var head = 0
            while head < len(queue):
                var cur = queue[head]
                head += 1
                r.nodes += 1
                if r.nodes > node_limit:
                    r.status = "budget"
                    return r^
                if best >= 0 and dist[cur] + 1 > best:
                    break  # BFS order: every later walk is longer than the best found
                if dist[dst] >= 0:
                    break
                var u = cur >> 1
                var par = cur & 1
                for t in range(off[u], off[u + 1]):
                    var j = ecol[t]
                    var nxt = 2 * nbr[t] + (par ^ Int((c.obs[j] >> UInt64(l)) & 1))
                    if dist[nxt] < 0:
                        dist[nxt] = dist[cur] + 1
                        pstate[nxt] = cur
                        pcol[nxt] = j
                        touched.append(nxt)
                        queue.append(nxt)
            if dist[dst] < 0:
                continue
            var length = dist[dst]
            if best >= 0 and length > best:
                continue
            # Walk back from (v, 1) to (v, 0); edges used an even number of times cancel.
            var walk = List[Int]()
            var st = dst
            while st != src:
                var j = pcol[st]
                inset[j] = not inset[j]
                walk.append(j)
                st = pstate[st]
            var w = List[Int]()
            for j in walk:
                if inset[j]:
                    w.append(j)
                    inset[j] = False
            sort(w)
            if best < 0 or length < best:
                best = length
                r.witnesses.clear()
            var dup = False
            for x in r.witnesses:
                if x == w:
                    dup = True
                    break
            if not dup and len(r.witnesses) < _MAX_WITNESSES:
                r.witnesses.append(w^)
    if best < 0:
        r.status = "none"
        return r^
    r.status = "exact"
    r.value = best
    return r^


def _cols_solution(c: _Cols, nd: Int, w: List[Int]) -> Bool:
    """Plain re-check on a column model: distinct columns, detectors cancel, observables flip."""
    if len(w) == 0:
        return False
    var cnt = List[Int](length=nd, fill=0)
    var o: UInt64 = 0
    var seen = List[Bool](length=len(c.dets), fill=False)
    for j in w:
        if seen[j]:
            return False
        seen[j] = True
        for d in c.dets[j]:
            cnt[d] += 1
        o ^= c.obs[j]
    for x in cnt:
        if (x & 1) == 1:
            return False
    return o != 0


struct _ProjSolve(Movable):
    """Projected optimum: status "exact" (value, witnesses as projected column sets), "budget"
    (lower), or "above_max_weight" (Delta of the projection > max_weight, possibly infinite)."""

    var status: String
    var lower: Int
    var witnesses: List[List[Int]]
    var nodes: Int
    var graphlike: Bool

    def __init__(out self, status: String, lower: Int, var witnesses: List[List[Int]], nodes: Int, graphlike: Bool):
        self.status = status
        self.lower = lower
        self.witnesses = witnesses^
        self.nodes = nodes
        self.graphlike = graphlike


def _solve_projection(p: _Cols, nd: Int, max_weight: Int, node_limit: Int) raises -> _ProjSolve:
    """Exact optimum of the projected model: shortest odd closed walk when every projected column has
    at most 2 detectors, the weight-bounded search otherwise. Each witness is re-checked on p."""
    var graphlike = True
    for j in range(len(p.dets)):
        if len(p.dets[j]) > 2:
            graphlike = False
            break
    if graphlike:
        var g = _graph_solve(p, nd, node_limit)
        if g.status == "budget":
            return _ProjSolve("budget", 1, List[List[Int]](), g.nodes, True)
        if g.status == "none" or g.value > max_weight:
            return _ProjSolve("above_max_weight", max_weight + 1, List[List[Int]](), g.nodes, True)
        for w in g.witnesses:
            if len(w) != g.value or not _cols_solution(p, nd, w):
                raise Error("QM-E741: internal: the odd-cycle solve returned a set that is not an optimal projected solution. Hint: report this detector error model.")
        return _ProjSolve("exact", g.value, g.witnesses.copy(), g.nodes, True)
    var f = _deepen(p, nd, max_weight, node_limit, 1)
    if f.status == "exact":
        if not _cols_solution(p, nd, f.chosen):
            raise Error("QM-E741: internal: the projected search returned a set that is not a solution. Hint: report this detector error model.")
        var ws = List[List[Int]]()
        ws.append(f.chosen.copy())
        return _ProjSolve("exact", f.lower, ws^, f.nodes, False)
    return _ProjSolve(f.status, f.lower, List[List[Int]](), f.nodes, False)


def _lift_rec(i: Int, cands: List[List[Int]], outs: List[List[UInt64]], W: Int, mut acc: List[UInt64], mut pick: List[Int], mut tries: Int, cap: Int) -> Bool:
    if i == len(cands):
        for w in range(W):
            if acc[w] != 0:
                return False
        return True
    for t in range(len(cands[i])):
        tries += 1
        if tries > cap:
            return False
        var base = t * W
        for w in range(W):
            acc[w] ^= outs[i][base + w]
        pick[i] = cands[i][t]
        if _lift_rec(i + 1, cands, outs, W, acc, pick, tries, cap):
            return True
        for w in range(W):
            acc[w] ^= outs[i][base + w]
    return False


comptime _LIFT_CAP = 1 << 16


def _lift(full: _Cols, nd: Int, keep: List[Bool], p: _Cols, pw: List[Int]) -> List[Int]:
    """Pick one full column per projected column of pw (its members) so that the detectors outside
    the sector cancel. Returns full-column indices, or an empty list when no choice was found within
    _LIFT_CAP tries. Members share the projected detectors and the observable mask, so the lifted set
    has the projected syndrome on S (zero) and the same observable flip (nonzero)."""
    var W = (nd + 63) >> 6
    var cands = List[List[Int]]()
    var outs = List[List[UInt64]]()
    for k in pw:
        var mem = p.members[k].copy()
        var cnt = List[Int]()
        for fj in mem:
            var x = 0
            for d in full.dets[fj]:
                if not keep[d]:
                    x += 1
            cnt.append(x)
        var idx = List[Int]()
        for t in range(len(mem)):
            idx.append(t)

        def by_out(a: Int, b: Int) {imm cnt, imm mem} -> Bool:  # fewest outside detectors first
            if cnt[a] != cnt[b]:
                return cnt[a] < cnt[b]
            return mem[a] < mem[b]

        sort(Span(idx), by_out)
        var cl = List[Int]()
        var ol = List[UInt64]()
        for t in idx:
            var fj = mem[t]
            cl.append(fj)
            var bits = List[UInt64](length=W, fill=UInt64(0))
            for d in full.dets[fj]:
                if not keep[d]:
                    bits[d >> 6] ^= UInt64(1) << UInt64(d & 63)
            for w in range(W):
                ol.append(bits[w])
        cands.append(cl^)
        outs.append(ol^)
    var acc = List[UInt64](length=W, fill=UInt64(0))
    var pick = List[Int](length=len(pw), fill=0)
    var tries = 0
    if _lift_rec(0, cands, outs, W, acc, pick, tries, _LIFT_CAP):
        return pick^
    return List[Int]()


def detector_sectors(dem: DetectorErrorModel) raises -> List[List[Int]]:
    """Candidate detector sectors for the row-projection lower bound (a heuristic).

    Components of the graph whose edges are the 2-detector mechanisms, leaving out an edge {a, b}
    whose observable mask is the XOR of a 1-detector mechanism on a and one on b (a Y-type fault at
    a boundary, which bridges the two bases of a CSS memory). On QUMO's CSS memory circuits the
    components are the detectors of each measurement basis. Only components of at least 2 detectors
    that are not every detector are returned, smallest detector first. The choice affects how strong
    the bound is, never whether it is sound: any detector set gives a valid lower bound.
    """
    var nd = 1
    var c = _prepare(dem, nd)
    var single = List[List[UInt64]](length=nd, fill=List[UInt64]())
    for j in range(len(c.dets)):
        if len(c.dets[j]) == 1:
            single[c.dets[j][0]].append(c.obs[j])
    var par = List[Int](length=nd, fill=0)
    for d in range(nd):
        par[d] = d

    def find(x0: Int, mut par: List[Int]) -> Int:
        var x = x0
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    var used = List[Bool](length=nd, fill=False)
    for j in range(len(c.dets)):
        for d in c.dets[j]:
            used[d] = True
        if len(c.dets[j]) != 2:
            continue
        var a = c.dets[j][0]
        var b = c.dets[j][1]
        var bridged = False
        for oa in single[a]:
            for ob in single[b]:
                if (oa ^ ob) == c.obs[j]:
                    bridged = True
        if bridged:
            continue
        var ra = find(a, par)
        var rb = find(b, par)
        if ra != rb:
            par[max(ra, rb)] = min(ra, rb)
    var n_used = 0
    for d in range(nd):
        if used[d]:
            n_used += 1
    var root_of = List[Int](length=nd, fill=-1)
    var groups = List[List[Int]]()
    for d in range(nd):
        if not used[d]:
            continue
        var rt = find(d, par)
        if root_of[rt] < 0:
            root_of[rt] = len(groups)
            groups.append(List[Int]())
        groups[root_of[rt]].append(d)
    var out = List[List[Int]]()
    for g in groups:
        if len(g) >= 2 and len(g) < n_used:
            out.append(g.copy())
    return out^


def projected_distance(dem: DetectorErrorModel, sector: List[Int], max_weight: Int = 32, node_limit: Int = 50_000_000) raises -> DistanceResult:
    """Exact optimum of the row projection of `dem` onto the detectors in `sector`: a proven LOWER
    bound on `circuit_distance(dem)` for any sector.

    Shortest odd closed walk when every projected mechanism has at most 2 detectors (polynomial), the
    weight-bounded search otherwise. status as for `circuit_distance`, but about the projection:
    "exact" means lower = upper = the projected optimum, and `witness` holds one representative
    dem.errors index per projected column. That set solves the projection; it need not solve `dem`
    (check with `verify_witness`). "budget" in the odd-walk solve gives lower = 1.
    """
    _check_args(dem, max_weight, node_limit)
    var nd = 1
    var full = _prepare(dem, nd)
    var keep = _keep_mask(sector, nd)
    if full.n_log == 0:
        return DistanceResult(max_weight + 1, -1, List[Int](), 0, "no_logical")
    var p = _project(full, keep)
    var ps = _solve_projection(p, nd, max_weight, node_limit)
    if ps.status == "exact":
        var witness = List[Int]()
        for k in ps.witnesses[0]:
            witness.append(full.tie[p.tie[k]])
        return DistanceResult(ps.lower, ps.lower, witness^, ps.nodes, "exact")
    return DistanceResult(ps.lower, -1, List[Int](), ps.nodes, ps.status)


def _keep_mask(sector: List[Int], nd: Int) raises -> List[Bool]:
    if len(sector) == 0:
        raise _sector_error("is empty")
    var keep = List[Bool](length=nd, fill=False)
    for d in sector:
        if d < 0 or d >= nd:
            raise _sector_error("has detector id " + String(d) + " outside 0.." + String(nd - 1))
        keep[d] = True
    return keep^


def circuit_distance_reduced(dem: DetectorErrorModel, max_weight: Int = 32, node_limit: Int = 50_000_000) raises -> DistanceResult:
    """`circuit_distance_with_sectors(dem, detector_sectors(dem), ...)`: exact circuit distance with a
    row-projection lower bound on the automatically found sectors. Same result semantics as
    `circuit_distance` (the witness may differ, the value cannot)."""
    _check_args(dem, max_weight, node_limit)
    return circuit_distance_with_sectors(dem, detector_sectors(dem), max_weight, node_limit)


def circuit_distance_with_sectors(dem: DetectorErrorModel, sectors: List[List[Int]], max_weight: Int = 32, node_limit: Int = 50_000_000) raises -> DistanceResult:
    """Exact circuit distance of `dem`, using row projections onto `sectors` for the lower bound.

    1. Lower bound L = max over sectors S of the projected optimum (each a proven lower bound, see
       `projected_distance`), never below 1.
    2. Upper bound: lift an optimal projected solution of weight L to `dem` (one mechanism per
       projected column, detectors outside S cancelling) and re-check it with `verify_witness`. A
       verified lift of weight L makes the result exact.
    3. Otherwise the weight-bounded search on `dem` runs from threshold K = L (thresholds below L are
       already excluded by step 1) with the remaining node budget.
    `node_limit` bounds the odd-walk states and the projected and full searches; lifting is bounded
    separately (_MAX_WITNESSES projected solutions, _LIFT_CAP tries each). status, lower, upper
    and witness keep the meaning of `circuit_distance`. An empty `sectors` list is plain
    `circuit_distance`.
    """
    _check_args(dem, max_weight, node_limit)
    var nd = 1
    var full = _prepare(dem, nd)
    var keeps = List[List[Bool]]()
    for S in sectors:
        keeps.append(_keep_mask(S, nd))
    if full.n_log == 0:
        return DistanceResult(max_weight + 1, -1, List[Int](), 0, "no_logical")
    var nodes = 0
    var lower = 1
    var projs = List[_Cols]()
    var sols = List[_ProjSolve]()
    for i in range(len(keeps)):
        var p = _project(full, keeps[i])
        var ps = _solve_projection(p, nd, max_weight, node_limit - nodes)
        nodes += ps.nodes
        if ps.status == "above_max_weight":
            return DistanceResult(max_weight + 1, -1, List[Int](), nodes, "above_max_weight")
        if ps.lower > lower:
            lower = ps.lower
        projs.append(p^)
        sols.append(ps^)
        if nodes >= node_limit:
            break
    for i in range(len(sols)):
        if sols[i].status != "exact" or sols[i].lower != lower:
            continue
        for pw in sols[i].witnesses:
            var lifted = _lift(full, nd, keeps[i], projs[i], pw)
            if len(lifted) == 0:
                continue
            var witness = List[Int]()
            for fj in lifted:
                witness.append(full.tie[fj])
            if len(witness) == lower and verify_witness(dem, witness):
                return DistanceResult(lower, lower, witness^, nodes, "exact")
    var f = _deepen(full, nd, max_weight, node_limit - nodes, lower)
    nodes += f.nodes
    if f.status == "exact":
        var witness = List[Int]()
        for j in f.chosen:
            witness.append(full.tie[j])
        if not verify_witness(dem, witness):
            raise Error("QM-E740: internal: circuit_distance found a witness that does not verify. Hint: report this detector error model.")
        return DistanceResult(f.lower, f.lower, witness^, nodes, "exact")
    return DistanceResult(f.lower, -1, List[Int](), nodes, f.status)
