"""Detector error models (DEM): the "narrow waist" between circuits and decoders.

A DEM lists independent error mechanisms, each with a probability and the set
of detectors / logical observables it flips (Stim's `DetectorErrorModel`):

    error(0.001) D0 D1
    error(0.0005) D3 L0
    error(0.0002) D4 D9 ^ D5 D11        <- decomposed into graph-like pieces

Extraction (Stim's ErrorAnalyzer, done backwards in one sweep):
  We walk the circuit (already lowered to frame primitives by
  `qumo.sim.frame.FrameProgram`) in *reverse*, maintaining for every qubit
  the sensitivity sets  sens_x[q], sens_z[q]  = detectors/observables that an
  X (Z) error at the current time would flip. Reverse gate rules are the
  Heisenberg conjugations (e.g. CX: sens_x[c] ^= sens_x[t],
  sens_z[t] ^= sens_z[c]); a Z-measurement adds its record's detectors to
  sens_x; resets clear both. Each noise channel met on the way contributes one
  mechanism per Pauli component with signature = XOR of the relevant sets.
  Cost is O(circuit size x average set size), i.e. linear for local codes,
  instead of re-simulating every error forwards.

Channel conversion follows Stim: DEPOLARIZE1(p) becomes independent X, Y, Z
mechanisms with q = (1 - sqrt(1 - 4p/3))/2, DEPOLARIZE2(p) 15 mechanisms with
q = (1 - (1 - 16p/15)^(1/8))/2, PAULI_CHANNEL_1 is converted from disjoint to
independent by fixed-point iteration. Identical mechanisms are merged with
p = p1(1-p2) + p2(1-p1).

Decomposition (`decompose=True`, Stim's `decompose_errors`): every mechanism
of a channel is split along the channel's single-qubit X/Z "basis errors"
into graph-like pieces (<= 2 detectors): detectors that some basis
combination flips alone become single-detector pieces, otherwise an
"irreducible pair" plus singles is used. This reproduces Stim's choice, so
`MatchingGraph` builds the same graph PyMatching builds from Stim's DEM.
Mechanisms that cannot be split locally are kept whole and decomposed later
against the global set of graph-like edges.

A Z-type sensitivity surviving into a measurement or reset means a detector
anticommutes with the circuit's stabilizers: QM-E703 (non-deterministic
detector), exactly what Stim reports.
"""

from std.math import sqrt, pow
from std.collections import Dict

from qumo.ir.circuit import Circuit
from qumo.sim.frame import FrameProgram, FrameOp, FK


struct DemError(Copyable, Writable):
    var p: Float64
    var dets: List[Int]  # sorted detector ids (full symptom)
    var obs: UInt64  # bit k = flips logical observable k
    var comp_dets: List[List[Int]]  # graph-like decomposition (empty = none)
    var comp_obs: List[UInt64]

    def __init__(out self, p: Float64, var dets: List[Int], obs: UInt64):
        self.p = p
        self.dets = dets^
        self.obs = obs
        self.comp_dets = List[List[Int]]()
        self.comp_obs = List[UInt64]()

    def is_graphlike(self) -> Bool:
        return len(self.dets) <= 2

    def num_components(self) -> Int:
        return len(self.comp_dets)

    def write_to(self, mut w: Some[Writer]):
        w.write("error(", self.p, ")")
        if len(self.comp_dets) > 1:
            for c in range(len(self.comp_dets)):
                if c > 0:
                    w.write(" ^")
                for d in self.comp_dets[c]:
                    w.write(" D", d)
                _write_obs(w, self.comp_obs[c])
            return
        for d in self.dets:
            w.write(" D", d)
        _write_obs(w, self.obs)


def _write_obs(mut w: Some[Writer], obs: UInt64):
    for k in range(64):
        if ((obs >> UInt64(k)) & 1) != 0:
            w.write(" L", k)


struct DetectorErrorModel(Copyable, Writable):
    var num_detectors: Int
    var num_observables: Int
    var errors: List[DemError]
    var detector_coords: List[Float64]  # flat (x, y, t) per detector (optional)

    def __init__(out self, num_detectors: Int, num_observables: Int):
        self.num_detectors = num_detectors
        self.num_observables = num_observables
        self.errors = List[DemError]()
        self.detector_coords = List[Float64]()

    def num_errors(self) -> Int:
        return len(self.errors)

    def total_probability(self) -> Float64:
        var s = 0.0
        for e in self.errors:
            s += e.p
        return s

    def to_text(self) -> String:
        """Stim DEM syntax (errors, then detector coordinate annotations)."""
        var s = String()
        for e in self.errors:
            s += String(e) + "\n"
        if len(self.detector_coords) >= 3 * self.num_detectors:
            for d in range(self.num_detectors):
                s += "detector(" + String(self.detector_coords[3 * d]) + ", " + String(self.detector_coords[3 * d + 1]) + ", " + String(self.detector_coords[3 * d + 2]) + ") D" + String(d) + "\n"
        else:
            if self.num_detectors > 0:
                s += "detector D" + String(self.num_detectors - 1) + "\n"
        for k in range(self.num_observables):
            s += "logical_observable L" + String(k) + "\n"
        return s

    def write_to(self, mut w: Some[Writer]):
        w.write("DetectorErrorModel(detectors=", self.num_detectors, ", observables=",
                self.num_observables, ", errors=", len(self.errors), ")")


def _xor_sorted(a: List[Int], b: List[Int]) -> List[Int]:
    """Symmetric difference of two sorted id lists."""
    var out = List[Int](capacity=len(a) + len(b))
    var i = 0
    var j = 0
    while i < len(a) and j < len(b):
        if a[i] < b[j]:
            out.append(a[i])
            i += 1
        elif a[i] > b[j]:
            out.append(b[j])
            j += 1
        else:
            i += 1
            j += 1
    while i < len(a):
        out.append(a[i])
        i += 1
    while j < len(b):
        out.append(b[j])
        j += 1
    return out^


def _toggle(mut a: List[Int], v: Int):
    """XOR a single id into a sorted list."""
    var b = List[Int]()
    b.append(v)
    a = _xor_sorted(a, b)


def depolarize1_independent(p: Float64) -> Float64:
    """Per-Pauli independent probability equivalent to DEPOLARIZE1(p)."""
    var v = 1.0 - 4.0 * p / 3.0
    if v <= 0.0:
        return 0.5
    return 0.5 - 0.5 * sqrt(v)


def depolarize2_independent(p: Float64) -> Float64:
    var v = 1.0 - 16.0 * p / 15.0
    if v <= 0.0:
        return 0.5
    return 0.5 - 0.5 * pow(v, 0.125)


def pauli_channel_independent(px: Float64, py: Float64, pz: Float64) -> Tuple[Float64, Float64, Float64]:
    """Disjoint (px, py, pz) -> independent X/Y/Z mechanism probabilities."""
    var qx = px
    var qy = py
    var qz = pz
    for _ in range(100):
        var nx = (px - (1 - qx) * qy * qz) / max(1e-300, (1 - qy) * (1 - qz))
        var ny = (py - (1 - qy) * qx * qz) / max(1e-300, (1 - qx) * (1 - qz))
        var nz = (pz - (1 - qz) * qx * qy) / max(1e-300, (1 - qx) * (1 - qy))
        qx = min(0.5, max(0.0, nx))
        qy = min(0.5, max(0.0, ny))
        qz = min(0.5, max(0.0, nz))
    return (qx, qy, qz)


@always_inline
def combine_independent(p1: Float64, p2: Float64) -> Float64:
    return p1 * (1 - p2) + p2 * (1 - p1)


def _split(sig: List[Int], D: Int, mut dets: List[Int]) -> UInt64:
    """Split an id list into detector ids (appended to `dets`) and an observable mask."""
    var obs: UInt64 = 0
    for v in sig:
        if v < D:
            dets.append(v)
        else:
            obs ^= UInt64(1) << UInt64(v - D)
    return obs


def _ndets(sig: List[Int], D: Int) -> Int:
    var n = 0
    for v in sig:
        if v < D:
            n += 1
    return n


def _contains(sorted_ids: List[Int], v: Int) -> Bool:
    for x in sorted_ids:
        if x == v:
            return True
    return False


struct _DemAccumulator:
    var num_detectors: Int
    var errors: List[DemError]
    var table: Dict[UInt64, Int]

    def __init__(out self, num_detectors: Int):
        self.num_detectors = num_detectors
        self.errors = List[DemError]()
        self.table = Dict[UInt64, Int]()

    def add(mut self, p: Float64, comps: List[List[Int]]):
        """Add a mechanism given as graph-like pieces (one piece = undecomposed)."""
        if p <= 0.0:
            return
        var D = self.num_detectors
        # full symptom
        var full = List[Int]()
        for c in comps:
            full = _xor_sorted(full, c)
        if len(full) == 0:
            return
        # canonical piece order (by first id) for hashing / printing
        var order = List[Int]()
        for i in range(len(comps)):
            order.append(i)
        for i in range(1, len(order)):
            var j = i
            while j > 0 and _less(comps[order[j]], comps[order[j - 1]]):
                var t = order[j]
                order[j] = order[j - 1]
                order[j - 1] = t
                j -= 1
        var h: UInt64 = 0xCBF29CE484222325
        for i in order:
            for v in comps[i]:
                h = (h ^ UInt64(v + 1)) * 0x100000001B3
            h = (h ^ 0xFFFF) * 0x100000001B3  # piece separator
        var dets = List[Int]()
        var obs = _split(full, D, dets)
        while True:
            var idx = self.table.get(h, -1)
            if idx < 0:
                self.table[h] = len(self.errors)
                var e = DemError(p, dets^, obs)
                if len(comps) > 1:
                    for i in order:
                        var cd = List[Int]()
                        var co = _split(comps[i], D, cd)
                        e.comp_dets.append(cd^)
                        e.comp_obs.append(co)
                self.errors.append(e^)
                return
            ref e = self.errors[idx]
            if e.obs == obs and e.dets == dets and self._same_pieces(e, comps, order):
                e.p = combine_independent(e.p, p)
                return
            h += 1

    def _same_pieces(self, e: DemError, comps: List[List[Int]], order: List[Int]) -> Bool:
        if len(comps) <= 1:
            return len(e.comp_dets) == 0
        if len(e.comp_dets) != len(comps):
            return False
        for t in range(len(order)):
            var cd = List[Int]()
            var co = _split(comps[order[t]], self.num_detectors, cd)
            if cd != e.comp_dets[t] or co != e.comp_obs[t]:
                return False
        return True


def _less(a: List[Int], b: List[Int]) -> Bool:
    var m = min(len(a), len(b))
    for t in range(m):
        if a[t] != b[t]:
            return a[t] < b[t]
    return len(a) < len(b)


struct _Channel:
    """Local (per noise channel) Stim-style decomposition helper.

    basis[i] = symptom of the i-th single-qubit basis error (X_a, Z_a, X_b, Z_b);
    combination k (bitmask over basis errors) has symptom XOR_i basis[i].
    """

    var D: Int
    var sigs: List[List[Int]]  # index k = 1 .. 2^s - 1
    var singles: List[Int]  # detectors flipped alone by some combination
    var decompose: Bool

    def __init__(out self, basis: List[List[Int]], D: Int, decompose: Bool):
        self.D = D
        self.decompose = decompose
        var s = len(basis)
        self.sigs = List[List[Int]](length=1 << s, fill=List[Int]())
        for k in range(1, 1 << s):
            var acc = List[Int]()
            for i in range(s):
                if ((k >> i) & 1) != 0:
                    acc = _xor_sorted(acc, basis[i])
            self.sigs[k] = acc^
        self.singles = List[Int]()
        for k in range(1, 1 << s):
            if _ndets(self.sigs[k], D) == 1:
                for v in self.sigs[k]:
                    if v < D and not _contains(self.singles, v):
                        self.singles.append(v)

    def _single_piece(self, d: Int, want_obs: UInt64, mut out: List[Int]) -> Bool:
        """Find a combination flipping exactly detector d with observable mask want_obs."""
        for k in range(1, len(self.sigs)):
            if _ndets(self.sigs[k], self.D) == 1 and _contains(self.sigs[k], d):
                var tmp = List[Int]()
                if _split(self.sigs[k], self.D, tmp) == want_obs:
                    out = self.sigs[k].copy()
                    return True
        return False

    def _singles_cover(self, dets: List[Int], idx: Int, target: UInt64, mut pieces: List[List[Int]]) -> Bool:
        """Cover dets[idx:] by single-detector pieces whose observables XOR to target."""
        if idx == len(dets):
            return target == 0
        var d = dets[idx]
        for k in range(1, len(self.sigs)):
            if _ndets(self.sigs[k], self.D) == 1 and _contains(self.sigs[k], d):
                var tmp = List[Int]()
                var o = _split(self.sigs[k], self.D, tmp)
                pieces.append(self.sigs[k].copy())
                if self._singles_cover(dets, idx + 1, target ^ o, pieces):
                    return True
                _ = pieces.pop()
        return False

    def pieces(self, k: Int) -> List[List[Int]]:
        """Graph-like decomposition of combination k (one piece if none applies)."""
        var whole = List[List[Int]]()
        whole.append(self.sigs[k].copy())
        if not self.decompose:
            return whole^
        var dets = List[Int]()
        var obs = _split(self.sigs[k], self.D, dets)
        var nd = len(dets)
        if nd <= 1:
            return whole^
        var all_single = True
        for d in dets:
            if not _contains(self.singles, d):
                all_single = False
        if nd == 2 and not all_single:
            return whole^  # irreducible pair
        if all_single:
            var ps = List[List[Int]]()
            if self._singles_cover(dets, 0, obs, ps):
                return ps^
            if nd == 2:
                return whole^
        # irreducible pair + singles
        for k2 in range(1, len(self.sigs)):
            var d2 = List[Int]()
            var o2 = _split(self.sigs[k2], self.D, d2)
            if len(d2) != 2:
                continue
            if _contains(self.singles, d2[0]) and _contains(self.singles, d2[1]):
                continue
            if not (_contains(dets, d2[0]) and _contains(dets, d2[1])):
                continue
            var rem = List[Int]()
            var ok = True
            for d in dets:
                if d != d2[0] and d != d2[1]:
                    if not _contains(self.singles, d):
                        ok = False
                    rem.append(d)
            if not ok:
                continue
            var ps = List[List[Int]]()
            ps.append(self.sigs[k2].copy())
            if self._singles_cover(rem, 0, obs ^ o2, ps):
                return ps^
        return whole^


def _gauge_error(ids: List[Int], num_detectors: Int, where_: String) -> Error:
    var s = String()
    for i in range(min(len(ids), 8)):
        if ids[i] < num_detectors:
            s += " D" + String(ids[i])
        else:
            s += " L" + String(ids[i] - num_detectors)
    return Error(
        "QM-E703: non-deterministic detectors/observables" + s + " (they anticommute with a "
        + where_ + "). Hint: in noiseless execution every DETECTOR/OBSERVABLE must have a fixed"
        " parity; check the first-round detectors and the logical operator's basis."
    )


def _add_1q(mut acc: _DemAccumulator, sx: List[Int], sz: List[Int], qx: Float64, qy: Float64, qz: Float64, decompose: Bool):
    var basis = List[List[Int]]()
    basis.append(sx.copy())
    basis.append(sz.copy())
    var ch = _Channel(basis, acc.num_detectors, decompose)
    if qx > 0: acc.add(qx, ch.pieces(1))
    if qy > 0: acc.add(qy, ch.pieces(3))
    if qz > 0: acc.add(qz, ch.pieces(2))


def detector_error_model(circuit: Circuit, decompose: Bool = True) raises -> DetectorErrorModel:
    """Extract the DEM of a noisy Clifford circuit with DETECTOR/OBSERVABLE annotations.

    `decompose=True` also records Stim-style graph-like decompositions
    (`A B ^ C`), which `MatchingGraph` uses to build the matching graph.
    """
    var prog = FrameProgram(circuit, with_reference=False)
    var D = prog.num_detectors
    var K = prog.num_observables
    if K > 64:
        raise Error("QM-E705: at most 64 logical observables are supported (got " + String(K) + ").")
    var n = prog.num_qubits
    var sx = List[List[Int]](length=n, fill=List[Int]())
    var sz = List[List[Int]](length=n, fill=List[Int]())
    var rec_sens = List[List[Int]](length=max(1, prog.num_measurements), fill=List[Int]())
    var acc = _DemAccumulator(D)
    var i = len(prog.ops) - 1
    var empty = List[Int]()
    while i >= 0:
        var fo = prog.ops[i]
        i -= 1
        var k = fo.kind
        if k == FK.CX:
            sx[fo.a] = _xor_sorted(sx[fo.a], sx[fo.b])
            sz[fo.b] = _xor_sorted(sz[fo.b], sz[fo.a])
        elif k == FK.H:
            var t = sx[fo.a].copy()
            sx[fo.a] = sz[fo.a].copy()
            sz[fo.a] = t^
        elif k == FK.S:
            sx[fo.a] = _xor_sorted(sx[fo.a], sz[fo.a])
        elif k == FK.SX:
            sz[fo.a] = _xor_sorted(sz[fo.a], sx[fo.a])
        elif k == FK.CZ:
            sx[fo.a] = _xor_sorted(sx[fo.a], sz[fo.b])
            sx[fo.b] = _xor_sorted(sx[fo.b], sz[fo.a])
        elif k == FK.SWAP:
            var tx = sx[fo.a].copy()
            var tz = sz[fo.a].copy()
            sx[fo.a] = sx[fo.b].copy()
            sz[fo.a] = sz[fo.b].copy()
            sx[fo.b] = tx^
            sz[fo.b] = tz^
        elif k == FK.M:
            if len(sz[fo.a]) > 0:
                raise _gauge_error(sz[fo.a], D, "Z-basis measurement of qubit " + String(fo.a))
            sx[fo.a] = _xor_sorted(sx[fo.a], rec_sens[fo.b])
        elif k == FK.R:
            if len(sz[fo.a]) > 0:
                raise _gauge_error(sz[fo.a], D, "reset of qubit " + String(fo.a))
            sx[fo.a] = empty.copy()
            sz[fo.a] = empty.copy()
        elif k == FK.DET:
            for j in range(fo.b):
                _toggle(rec_sens[prog.records[fo.a + j]], fo.c)
        elif k == FK.OBS:
            for j in range(fo.b):
                _toggle(rec_sens[prog.records[fo.a + j]], D + fo.c)
        elif k == FK.FEED:
            if fo.c == 1 or fo.c == 2:
                rec_sens[fo.a] = _xor_sorted(rec_sens[fo.a], sx[fo.b])
            if fo.c >= 2:
                rec_sens[fo.a] = _xor_sorted(rec_sens[fo.a], sz[fo.b])
        elif k == FK.XE:
            var one = List[List[Int]]()
            one.append(sx[fo.a].copy())
            acc.add(fo.p, one)
        elif k == FK.ZE:
            var one = List[List[Int]]()
            one.append(sz[fo.a].copy())
            acc.add(fo.p, one)
        elif k == FK.YE:
            _add_1q(acc, sx[fo.a], sz[fo.a], 0.0, fo.p, 0.0, decompose)
        elif k == FK.DEP1:
            var q = depolarize1_independent(fo.p)
            _add_1q(acc, sx[fo.a], sz[fo.a], q, q, q, decompose)
        elif k == FK.PC1:
            var pz = fo.p - fo.px - fo.py
            var qs = pauli_channel_independent(fo.px, fo.py, pz)
            _add_1q(acc, sx[fo.a], sz[fo.a], qs[0], qs[1], qs[2], decompose)
        elif k == FK.DEP2:
            var q = depolarize2_independent(fo.p)
            var basis = List[List[Int]]()
            basis.append(sx[fo.a].copy())
            basis.append(sz[fo.a].copy())
            basis.append(sx[fo.b].copy())
            basis.append(sz[fo.b].copy())
            var ch = _Channel(basis, D, decompose)
            for kk in range(1, 16):
                acc.add(q, ch.pieces(kk))
    for q in range(n):
        if len(sz[q]) > 0:
            raise _gauge_error(sz[q], D, "the initial |0> state of qubit " + String(q))
    var dem = DetectorErrorModel(D, K)
    # Deterministic output order: sort by detector list then observables.
    var order = List[Int](capacity=len(acc.errors))
    for j in range(len(acc.errors)):
        order.append(j)
    var errs = acc.errors.copy()

    def less(a: Int, b: Int) {imm errs} -> Bool:
        ref ea = errs[a]
        ref eb = errs[b]
        var m = min(len(ea.dets), len(eb.dets))
        for t in range(m):
            if ea.dets[t] != eb.dets[t]:
                return ea.dets[t] < eb.dets[t]
        if len(ea.dets) != len(eb.dets):
            return len(ea.dets) < len(eb.dets)
        if ea.obs != eb.obs:
            return ea.obs < eb.obs
        return len(ea.comp_dets) < len(eb.comp_dets)

    sort(Span(order), less)
    for j in order:
        dem.errors.append(errs[j].copy())
    return dem^
