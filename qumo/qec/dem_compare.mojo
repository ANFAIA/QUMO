"""Exact comparison of two detector error models (DEMVal's oracle, arXiv:2611.20501).

A symptom class is the pair (detectors flipped, observables flipped) of one
mechanism, with `^` pieces XOR-combined into the full symptom. `gather_dem`
merges mechanisms of equal class with x + y - 2xy, drops the empty class, p <= 0
and any class whose merged probability is 0, and sorts the classes in the
extractor's order (`qumo/qec/dem.mojo`). `compare_dem` gathers both sides and
walks the two tables once:

    unmatched (u) = classes present on one side only with p > floor
    invalid (i)   = mechanisms of the raw inputs with p not in [0, 1] (NaN too)
    max_rel (delta) = max over shared classes of |pa - pb| / max(pa, pb)
    consistent     = u == 0 and i == 0 and delta <= tol

Detector and observable counts are reported apart (`same_shape`) and do not
enter `consistent`. The verdict is symmetric: it says the models differ, not
which one is wrong. It compares mechanism tables of independent errors, which
is enough for equal distributions but is not a distribution test.
"""

from qumo.qec.dem import DetectorErrorModel, DemError, combine_independent, _write_obs


struct DemMismatch(Copyable, Writable):
    """One symptom class where the two models disagree."""

    var dets: List[Int]
    var obs: UInt64
    var p_a: Float64  # 0.0 when the class is missing in a
    var p_b: Float64  # 0.0 when the class is missing in b
    var rel: Float64  # |p_a - p_b| / max(p_a, p_b); 1.0 for a one-sided class

    def __init__(out self, var dets: List[Int], obs: UInt64, p_a: Float64, p_b: Float64, rel: Float64):
        self.dets = dets^
        self.obs = obs
        self.p_a = p_a
        self.p_b = p_b
        self.rel = rel

    def write_to(self, mut w: Some[Writer]):
        var first = True
        for d in self.dets:
            if not first:
                w.write(" ")
            w.write("D", d)
            first = False
        if first:
            w.write("-")
        _write_obs(w, self.obs)
        w.write(": a=", self.p_a, " b=", self.p_b, " rel=", self.rel)


struct DemComparison(Copyable, Writable):
    """Result of `compare_dem`; `worst` holds at most `keep` violations, largest first."""

    var consistent: Bool
    var unmatched: Int  # u
    var max_rel: Float64  # delta
    var shared: Int
    var same_shape: Bool
    var invalid: Int  # i: raw mechanisms with p not in [0, 1] (NaN included)
    var worst: List[DemMismatch]

    def __init__(out self):
        self.consistent = True
        self.unmatched = 0
        self.max_rel = 0.0
        self.shared = 0
        self.same_shape = True
        self.invalid = 0
        self.worst = List[DemMismatch]()

    def write_to(self, mut w: Some[Writer]):
        w.write("DemComparison(consistent=", self.consistent, ", unmatched=", self.unmatched,
                ", invalid=", self.invalid,
                ", max_rel=", self.max_rel, ", shared=", self.shared, ", same_shape=", self.same_shape, ")")
        for m in self.worst:
            w.write("\n  ", m)


def _canonical(dets: List[Int]) -> List[Int]:
    """Sorted copy with repeated ids cancelled in pairs (D3 D3 flips nothing)."""
    var s = dets.copy()
    sort(s)
    var out = List[Int](capacity=len(s))
    var i = 0
    while i < len(s):
        if i + 1 < len(s) and s[i] == s[i + 1]:
            i += 2
            continue
        out.append(s[i])
        i += 1
    return out^


def _class_cmp(da: List[Int], oa: UInt64, db: List[Int], ob: UInt64) -> Int:
    """-1 / 0 / 1 in the order of detector_error_model's output (dem.mojo:575)."""
    var m = min(len(da), len(db))
    for t in range(m):
        if da[t] != db[t]:
            return -1 if da[t] < db[t] else 1
    if len(da) != len(db):
        return -1 if len(da) < len(db) else 1
    if oa != ob:
        return -1 if oa < ob else 1
    return 0


def gather_dem(dem: DetectorErrorModel) -> DetectorErrorModel:
    """One mechanism per symptom class, undecomposed, in the extractor's order."""
    var keys = List[DemError](capacity=len(dem.errors))
    for e in dem.errors:
        if e.p <= 0.0:
            continue
        var d = _canonical(e.dets)
        if len(d) == 0 and e.obs == 0:
            continue
        keys.append(DemError(e.p, d^, e.obs))
    var order = List[Int](capacity=len(keys))
    for j in range(len(keys)):
        order.append(j)

    def less(x: Int, y: Int) {imm keys} -> Bool:
        var c = _class_cmp(keys[x].dets, keys[x].obs, keys[y].dets, keys[y].obs)
        if c != 0:
            return c < 0
        return x < y

    sort(Span(order), less)
    var merged = DetectorErrorModel(dem.num_detectors, dem.num_observables)
    for j in order:
        ref k = keys[j]
        var n = len(merged.errors)
        if n > 0 and _class_cmp(merged.errors[n - 1].dets, merged.errors[n - 1].obs, k.dets, k.obs) == 0:
            merged.errors[n - 1].p = combine_independent(merged.errors[n - 1].p, k.p)
        else:
            merged.errors.append(DemError(k.p, k.dets.copy(), k.obs))
    var out = DetectorErrorModel(dem.num_detectors, dem.num_observables)
    out.detector_coords = dem.detector_coords.copy()
    for e in merged.errors:
        if e.p > 0.0:
            out.errors.append(e.copy())
    return out^


def compare_dem(a: DetectorErrorModel, b: DetectorErrorModel, tol: Float64 = 1e-9,
                floor: Float64 = 1e-15, keep: Int = 8) -> DemComparison:
    """DEMVal's verdict on two models: consistent iff every probability is in [0, 1]
    (NaN is not), no one-sided class above `floor`, and every shared class agrees
    to relative `tol`."""
    var r = DemComparison()
    for e in a.errors:
        if not (e.p >= 0.0 and e.p <= 1.0):
            r.invalid += 1
    for e in b.errors:
        if not (e.p >= 0.0 and e.p <= 1.0):
            r.invalid += 1
    var ga = gather_dem(a)
    var gb = gather_dem(b)
    r.same_shape = a.num_detectors == b.num_detectors and a.num_observables == b.num_observables
    var bad = List[DemMismatch]()
    var i = 0
    var j = 0
    var na = len(ga.errors)
    var nb = len(gb.errors)
    while i < na or j < nb:
        var c: Int
        if i == na:
            c = 1
        elif j == nb:
            c = -1
        else:
            c = _class_cmp(ga.errors[i].dets, ga.errors[i].obs, gb.errors[j].dets, gb.errors[j].obs)
        if c < 0:
            ref e = ga.errors[i]
            if e.p > floor:
                r.unmatched += 1
                bad.append(DemMismatch(e.dets.copy(), e.obs, e.p, 0.0, 1.0))
            i += 1
        elif c > 0:
            ref e = gb.errors[j]
            if e.p > floor:
                r.unmatched += 1
                bad.append(DemMismatch(e.dets.copy(), e.obs, 0.0, e.p, 1.0))
            j += 1
        else:
            var pa = ga.errors[i].p
            var pb = gb.errors[j].p
            var rel = abs(pa - pb) / max(pa, pb)
            r.shared += 1
            r.max_rel = max(r.max_rel, rel)
            if rel > tol:
                bad.append(DemMismatch(ga.errors[i].dets.copy(), ga.errors[i].obs, pa, pb, rel))
            i += 1
            j += 1
    r.consistent = r.unmatched == 0 and r.invalid == 0 and r.max_rel <= tol
    var order = List[Int](capacity=len(bad))
    for t in range(len(bad)):
        order.append(t)

    def worse(x: Int, y: Int) {imm bad} -> Bool:
        if bad[x].rel != bad[y].rel:
            return bad[x].rel > bad[y].rel
        return x < y

    sort(Span(order), worse)
    for t in range(min(max(keep, 0), len(order))):
        r.worst.append(bad[order[t]].copy())
    return r^
