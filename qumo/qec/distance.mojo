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


def circuit_distance(dem: DetectorErrorModel, max_weight: Int = 32, node_limit: Int = 50_000_000) raises -> DistanceResult:
    """Exact circuit distance of `dem` by weight-bounded search (arXiv:2611.08247, SI Sec. 5.1).

    Thresholds K = 1 .. min(max_weight, columns), always from 1: no caller claim is needed, so
    every `lower` is proven. `node_limit` bounds the search nodes over all
    thresholds; running out is not an error (status "budget", lower = the threshold being searched).
    Pass `detector_error_model(c, decompose=False)`; a decomposed model gives the same value.
    """
    if max_weight < 1:
        raise _arg_error("got max_weight=" + String(max_weight))
    if node_limit < 1:
        raise _arg_error("got node_limit=" + String(node_limit))
    if dem.num_observables > 64:
        raise _arg_error("got " + String(dem.num_observables) + " observables")

    # Preprocessing: canonical symptoms, piece check, drop no-effect mechanisms.
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

    var m = len(syms)
    var order = List[Int](capacity=m)
    for j in range(m):
        order.append(j)

    def less(a: Int, b: Int) {imm syms, imm obs, imm orig} -> Bool:
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
        return orig[a] < orig[b]

    sort(Span(order), less)

    # Merge equal columns (adjacent after sorting): two equal columns XOR to nothing.
    var col_dets = List[List[Int]]()
    var col_obs = List[UInt64]()
    var col_orig = List[Int]()
    for t in range(m):
        var j = order[t]
        var nk = len(col_dets)
        if nk > 0 and col_obs[nk - 1] == obs[j] and col_dets[nk - 1] == syms[j]:
            continue
        col_dets.append(syms[j].copy())
        col_obs.append(obs[j])
        col_orig.append(orig[j])

    var n = len(col_dets)
    var n_log = 0
    for j in range(n):
        if col_obs[j] != 0:
            n_log += 1
    if n_log == 0:
        return DistanceResult(max_weight + 1, -1, List[Int](), 0, "no_logical")

    var nd = dem.num_detectors
    if max_id + 1 > nd:
        nd = max_id + 1
    if nd < 1:
        nd = 1

    var kmax = min(max_weight, n)
    var s = _Search(n, nd, col_dets, col_obs, kmax, node_limit)
    for K in range(1, kmax + 1):
        for a in range(n_log):
            s.root(a)
            if s._dfs(1, K):
                var fd = s.found_depth
                var witness = List[Int]()
                for t in range(fd):
                    witness.append(col_orig[s.chosen[t]])
                if not verify_witness(dem, witness):
                    raise Error("QM-E740: internal: circuit_distance found a witness that does not verify. Hint: report this detector error model.")
                return DistanceResult(fd, fd, witness^, s.nodes, "exact")
            if s.out_of_budget:
                return DistanceResult(K, -1, List[Int](), s.nodes, "budget")
    return DistanceResult(max_weight + 1, -1, List[Int](), s.nodes, "above_max_weight")
