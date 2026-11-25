"""Exact fault-tolerance check of post-selected state preparations.

Definition 1 of Schmid and Wille, "Generalizing and Combining Check Constructions:
Design-Space Exploration for Fault-Tolerant Quantum State Preparation",
https://arxiv.org/abs/2611.25107: a post-selected preparation is t-fault tolerant if
every set F of at most t faults is either detected (some verification measurement
differs from the fault-free run) or leaves a data error whose X part and Z part each
have weight at most |F| after reduction by the stabilizers of the prepared state.

`valid_stabilizer_state_preparation` (qumo.ft.verify) checks the fault-free circuit
only; this module enumerates every fault set of size 1..t exactly, by Pauli-frame
propagation of each single fault (64-bit masks) and XOR of the effects.

Fault model (`FaultModel`): X after each reset and after the initial |0> of every qubit
whose first operation is not a reset (that reset erases the initial fault), X/Y/Z after
each 1q gate, the 15 two-qubit Paulis after each CX/CZ, an outcome flip per measurement.
`FaultModel.xz_only()` keeps X and Z after 1q gates and XI, IX, XX, ZI, IZ, ZZ after 2q
gates. Op.I gets no fault location. Idle faults are not modelled.

Scope: CSS code and CSS state, one block, at most 64 qubits and 64 measurements,
unconditioned reset, measure, H, S, SDG, X, Y, Z, CX, CZ; deterministic verification
measurements. The exact coset reduction takes rank at most 24 per Pauli type (X and Z),
so in practice codes up to about 48 qubits: surface_code(5) works, surface_code(7) raises
QM-E738. Anything else raises QM-E738.

Error codes: QM-E738.
"""

from std.bit import pop_count, count_trailing_zeros

from qumo.ir.circuit import Circuit
from qumo.ir.gates import Op, op_name, op_is_annotation
from qumo.sim.stabilizer import TableauSimulator
from qumo.ft.pauli import SymPauli
from qumo.ft.code import StabilizerCode
from qumo.ft.verify import semantic_state_stabilizers, expand_logical, valid_stabilizer_state_preparation
from qumo.core.json import JsonWriter


struct FaultModel(Copyable, Writable):
    var after_2q: Bool  # Paulis after each CX / CZ
    var after_1q: Bool  # Paulis after each H, S, SDG, X, Y, Z
    var after_reset: Bool  # X after each RESET and after the initial |0> of every qubit not first reset
    var before_measure: Bool  # outcome flip of each measurement
    var include_y: Bool  # False: X/Z-only (2 one-qubit, 6 two-qubit Paulis)

    def __init__(out self, after_2q: Bool = True, after_1q: Bool = True, after_reset: Bool = True,
                 before_measure: Bool = True, include_y: Bool = True):
        self.after_2q = after_2q
        self.after_1q = after_1q
        self.after_reset = after_reset
        self.before_measure = before_measure
        self.include_y = include_y

    @staticmethod
    def circuit_level() -> FaultModel:
        return FaultModel()

    @staticmethod
    def xz_only() -> FaultModel:
        return FaultModel(include_y=False)

    def name(self) -> String:
        var four = self.after_2q and self.after_1q and self.after_reset and self.before_measure
        if four and self.include_y:
            return "circuit_level"
        if four:
            return "xz_only"
        return "custom"

    def write_to(self, mut w: Some[Writer]):
        w.write(self.name())


struct Fault(Copyable, Writable):
    var location: Int  # index of the fault location (0-based, order of the location list)
    var op: Int  # op index the fault follows; -1 = initial |0>; for a flip: the MEASURE op
    var q0: Int
    var p0: Int  # 0 I, 1 X, 2 Y, 3 Z on q0 (0 for a flip)
    var q1: Int  # -1 for 1q faults and flips
    var p1: Int
    var flip: Bool  # measurement outcome flip
    var meas: Int  # measurement index (record position) for a flip, else -1
    var x: UInt64  # X part of the propagated frame at the end of the circuit (all impl qubits)
    var z: UInt64  # Z part
    var rec: UInt64  # bit m set = record m flipped
    var label: String

    def __init__(out self, location: Int, op: Int, q0: Int, p0: Int, q1: Int, p1: Int, flip: Bool,
                 meas: Int, x: UInt64, z: UInt64, rec: UInt64, var label: String):
        self.location = location
        self.op = op
        self.q0 = q0
        self.p0 = p0
        self.q1 = q1
        self.p1 = p1
        self.flip = flip
        self.meas = meas
        self.x = x
        self.z = z
        self.rec = rec
        self.label = label^

    def write_to(self, mut w: Some[Writer]):
        w.write(self.label)


struct FaultToleranceReport(Copyable, Writable):
    var ok: Bool
    var reason: String  # "" when ok; else why (fault-free failure, or "k dangerous fault sets")
    var t: Int
    var model: String  # FaultModel.name()
    var locations: Int
    var single_faults: Int
    var fault_sets: Int  # sets of size 1..t enumerated
    var undetected: Int
    var dangerous: Int
    var witness: String  # first dangerous set, "" if none
    var operations: Int  # CX + CZ + MEASURE ops (paper's Definition 2)
    var cx_depth: Int

    def __init__(out self, ok: Bool, reason: String, t: Int, model: String, locations: Int,
                 single_faults: Int, fault_sets: Int, undetected: Int, dangerous: Int,
                 witness: String, operations: Int, cx_depth: Int):
        self.ok = ok
        self.reason = reason
        self.t = t
        self.model = model
        self.locations = locations
        self.single_faults = single_faults
        self.fault_sets = fault_sets
        self.undetected = undetected
        self.dangerous = dangerous
        self.witness = witness
        self.operations = operations
        self.cx_depth = cx_depth

    def to_json(self) -> String:
        var w = JsonWriter()
        w.begin_object()
        w.kv_bool("fault_tolerant", self.ok)
        w.kv_string("reason", self.reason)
        w.kv_int("t", self.t)
        w.kv_string("model", self.model)
        w.kv_int("locations", self.locations)
        w.kv_int("single_faults", self.single_faults)
        w.kv_int("fault_sets", self.fault_sets)
        w.kv_int("undetected", self.undetected)
        w.kv_int("dangerous", self.dangerous)
        w.kv_string("witness", self.witness)
        w.kv_int("operations", self.operations)
        w.kv_int("cx_depth", self.cx_depth)
        w.end_object()
        return w.finish()

    def write_to(self, mut w: Some[Writer]):
        if self.ok:
            w.write("FAULT TOLERANT (t=", self.t, ", ", self.model, "): ", self.fault_sets,
                    " fault sets, ", self.undetected, " undetected, ", self.dangerous, " dangerous")
        else:
            w.write("NOT FAULT TOLERANT (t=", self.t, ", ", self.model, "): ", self.reason)
            if self.witness.byte_length() > 0:
                w.write("; witness: ", self.witness)


# ---------------------------------------------------------------------------
# Bit helpers
# ---------------------------------------------------------------------------
def _letter(p: Int) -> String:
    if p == 1:
        return "X"
    if p == 2:
        return "Y"
    if p == 3:
        return "Z"
    return "I"


def _bit(m: UInt64, q: Int) -> UInt64:
    return (m >> UInt64(q)) & UInt64(1)


def _flip(m: UInt64, q: Int) -> UInt64:
    return m ^ (UInt64(1) << UInt64(q))


def _clear(m: UInt64, q: Int) -> UInt64:
    return m & ~(UInt64(1) << UInt64(q))


def _data_mask(nd: Int) -> UInt64:
    if nd >= 64:
        return ~UInt64(0)
    return (UInt64(1) << UInt64(nd)) - UInt64(1)  # nd < 64 here: no 1 << 64


def _pauli_masks(p: Int, q: Int) -> Tuple[UInt64, UInt64]:
    """(X mask, Z mask) of Pauli code p (0 I, 1 X, 2 Y, 3 Z) on qubit q; q < 0 gives nothing."""
    if p == 0 or q < 0:
        return (UInt64(0), UInt64(0))
    var bit = UInt64(1) << UInt64(q)
    var xm = bit if (p == 1 or p == 2) else UInt64(0)
    var zm = bit if (p == 2 or p == 3) else UInt64(0)
    return (xm, zm)


def _xz_pair(pa: Int, pb: Int) -> Bool:
    """Two-qubit Paulis of the X/Z-only model: both codes in {I, X} or both in {I, Z}."""
    var both_x = pa <= 1 and pb <= 1
    var both_z = (pa == 0 or pa == 3) and (pb == 0 or pb == 3)
    return both_x or both_z


def _mask_of(p: SymPauli, xpart: Bool) -> UInt64:
    var m = UInt64(0)
    for q in range(p.n):
        var bit = p.x(q) if xpart else p.z(q)
        if bit == 1:
            m = _flip(m, q)
    return m


def _qubit_set(m: UInt64) -> String:
    var s = String("{")
    var first = True
    for q in range(64):
        if _bit(m, q) == UInt64(1):
            if not first:
                s += ","
            s += String(q)
            first = False
    return s + "}"


# ---------------------------------------------------------------------------
# Scope, circuit scan, propagation
# ---------------------------------------------------------------------------
def _check_scope(impl: Circuit, code: StabilizerCode, num_ancillas: Int) raises:
    if not code.is_css():
        raise Error("QM-E738: code '" + code.name + "' is not CSS. Hint: the check splits X and Z errors; use a CSS code.")
    if impl.num_qubits != code.n + num_ancillas:
        raise Error("QM-E738: implementation has " + String(impl.num_qubits) + " qubits, expected n + num_ancillas = "
                    + String(code.n + num_ancillas) + ". Hint: one block; pass num_ancillas for flags.")
    if impl.num_qubits > 64 or impl.num_measurements > 64:
        raise Error("QM-E738: implementation has " + String(impl.num_qubits) + " qubits and "
                    + String(impl.num_measurements) + " measurements. Hint: the frame uses 64-bit masks; at most 64 of each.")
    for i in range(len(impl.ops)):
        var g = impl.ops[i]
        if g.cond >= 0 or g.cond_reg >= 0:
            raise Error("QM-E738: instruction #" + String(i) + " is classically conditioned. Hint: the check needs an unconditioned circuit.")
        var k = g.kind
        if k == Op.I or k == Op.X or k == Op.Y or k == Op.Z or k == Op.H or k == Op.S or k == Op.SDG:
            continue
        if k == Op.CX or k == Op.CZ or k == Op.MEASURE or k == Op.RESET or k == Op.BARRIER or k == Op.TICK:
            continue
        if op_is_annotation(k):
            continue
        raise Error("QM-E738: instruction #" + String(i) + " '" + op_name(k) + "' is not supported. Hint: reset, measure, H, S, SDG, X, Y, Z, CX, CZ.")


def _meas_index(impl: Circuit) -> List[Int]:
    """For each op: the running measurement index if it is MEASURE, else -1."""
    var out = List[Int]()
    var m = 0
    for i in range(len(impl.ops)):
        if impl.ops[i].kind == Op.MEASURE:
            out.append(m)
            m += 1
        else:
            out.append(-1)
    return out^


def _propagate(impl: Circuit, start: Int, meas: List[Int], mut x: UInt64, mut z: UInt64, mut rec: UInt64):
    """Push the frame (x, z) through ops [start, end) of the circuit, flipping records as it goes."""
    for i in range(start, len(impl.ops)):
        var g = impl.ops[i]
        var k = g.kind
        var a = g.q0
        if k == Op.H:
            var xa = _bit(x, a)
            var za = _bit(z, a)
            if xa != za:
                x = _flip(x, a)
                z = _flip(z, a)
        elif k == Op.S or k == Op.SDG:
            if _bit(x, a) == UInt64(1):
                z = _flip(z, a)
        elif k == Op.CX:
            var t = g.q1
            if _bit(x, a) == UInt64(1):
                x = _flip(x, t)
            if _bit(z, t) == UInt64(1):
                z = _flip(z, a)
        elif k == Op.CZ:
            var b = g.q1
            var xa = _bit(x, a)
            var xb = _bit(x, b)
            if xb == UInt64(1):
                z = _flip(z, a)
            if xa == UInt64(1):
                z = _flip(z, b)
        elif k == Op.MEASURE:
            if _bit(x, a) == UInt64(1):
                rec = _flip(rec, meas[i])
            z = _clear(z, a)
        elif k == Op.RESET:
            x = _clear(x, a)
            z = _clear(z, a)
        # X, Y, Z, I, annotations: nothing


def _after(impl: Circuit, meas: List[Int], loc: Int, op_index: Int, a: Int, pa: Int, b: Int, pb: Int,
           var label: String) -> Fault:
    """A fault of Paulis pa on a and pb on b, placed after op `op_index`, propagated to the end."""
    var xa = _pauli_masks(pa, a)
    var xb = _pauli_masks(pb, b)
    var x = xa[0] | xb[0]
    var z = xa[1] | xb[1]
    var rec = UInt64(0)
    _propagate(impl, op_index + 1, meas, x, z, rec)
    return Fault(loc, op_index, a, pa, b, pb, False, -1, x, z, rec, label^)


def _reset_first(impl: Circuit) -> List[Bool]:
    """For each qubit: True when its first operation is a RESET. The initial |0> fault of such a
    qubit is erased by that reset, so it gets no initial location."""
    var n = impl.num_qubits
    var seen = List[Bool](length=n, fill=False)
    var out = List[Bool](length=n, fill=False)
    for i in range(len(impl.ops)):
        var g = impl.ops[i]
        for j in range(2):
            # q1 of a MEASURE holds the measurement index, so only two-qubit gates use it
            if j == 1 and not (g.kind == Op.CX or g.kind == Op.CZ):
                continue
            var q = g.q0 if j == 0 else g.q1
            if q < 0 or q >= n or seen[q]:
                continue
            seen[q] = True
            out[q] = g.kind == Op.RESET
    return out^


def single_faults(impl: Circuit, model: FaultModel = FaultModel.circuit_level()) raises -> List[Fault]:
    """Every single fault of `model` on `impl`, in location order, with its propagated effect.
    Locations: the initial |0> of each qubit (unless its first operation is a RESET), then per op in
    order: a RESET, a 1q gate H, S, SDG, X, Y, Z, a CX or CZ, a MEASURE. Op.I, BARRIER, TICK and
    annotations get no location.
    Public (the tests cross-check it against the tableau simulator). Does not check the scope."""
    var meas = _meas_index(impl)
    var out = List[Fault]()
    var loc = 0
    if model.after_reset:
        var erased = _reset_first(impl)
        for q in range(impl.num_qubits):
            if erased[q]:
                continue
            out.append(_after(impl, meas, loc, -1, q, 1, -1, 0, "init q" + String(q) + ": X"))
            loc += 1
    for i in range(len(impl.ops)):
        var g = impl.ops[i]
        var k = g.kind
        var a = g.q0
        if k == Op.MEASURE:
            if model.before_measure:
                var m = meas[i]
                out.append(Fault(loc, i, a, 0, -1, 0, True, m, UInt64(0), UInt64(0),
                                 UInt64(1) << UInt64(m),
                                 "flip of measurement " + String(m) + " (#" + String(i) + ")"))
                loc += 1
        elif k == Op.RESET:
            if model.after_reset:
                out.append(_after(impl, meas, loc, i, a, 1, -1, 0,
                                  "after #" + String(i) + " reset(" + String(a) + "): X"))
                loc += 1
        elif k == Op.H or k == Op.S or k == Op.SDG or k == Op.X or k == Op.Y or k == Op.Z:
            if model.after_1q:
                for p in range(1, 4):
                    if p == 2 and not model.include_y:
                        continue
                    out.append(_after(impl, meas, loc, i, a, p, -1, 0,
                                      "after #" + String(i) + " " + op_name(k) + "(" + String(a) + "): " + _letter(p)))
                loc += 1
        elif k == Op.CX or k == Op.CZ:
            if model.after_2q:
                var b = g.q1
                for pa in range(4):
                    for pb in range(4):
                        if pa == 0 and pb == 0:
                            continue
                        if not model.include_y and not _xz_pair(pa, pb):
                            continue
                        out.append(_after(impl, meas, loc, i, a, pa, b, pb,
                                          "after #" + String(i) + " " + op_name(k) + "(" + String(a) + ","
                                          + String(b) + "): " + _letter(pa) + _letter(pb)))
                loc += 1
    return out^


def _cx_stats(impl: Circuit) -> Tuple[Int, Int]:
    """(operations, cx_depth): CX + CZ + MEASURE count, and the layers of two-qubit gates."""
    var level = List[Int](length=impl.num_qubits, fill=0)
    var ops = 0
    var depth = 0
    for i in range(len(impl.ops)):
        var g = impl.ops[i]
        if g.kind == Op.CX or g.kind == Op.CZ:
            var l = max(level[g.q0], level[g.q1]) + 1
            level[g.q0] = l
            level[g.q1] = l
            ops += 1
            if l > depth:
                depth = l
        elif g.kind == Op.MEASURE:
            ops += 1
    return (ops, depth)


# ---------------------------------------------------------------------------
# Reduction over the state group
# ---------------------------------------------------------------------------
def _xor_basis(masks: List[UInt64]) -> List[UInt64]:
    var piv = List[UInt64](length=64, fill=UInt64(0))
    for v0 in masks:
        var v = v0
        for b in range(63, -1, -1):
            if ((v >> UInt64(b)) & UInt64(1)) == UInt64(0):
                continue
            if piv[b] == UInt64(0):
                piv[b] = v
                break
            v ^= piv[b]
    var out = List[UInt64]()
    for b in range(64):
        if piv[b] != UInt64(0):
            out.append(piv[b])
    return out^


def _min_weight(e: UInt64, basis: List[UInt64]) -> Int:
    """Minimum popcount of e + span(basis), by Gray-code enumeration of the span."""
    var r = len(basis)
    var best = Int(pop_count(e))
    var cur = e
    for i in range(1, 1 << r):
        cur ^= basis[Int(count_trailing_zeros(UInt64(i)))]
        var c = Int(pop_count(cur))
        if c < best:
            best = c
    return best


def _eliminate(mut piv: List[UInt64], mut other: List[UInt64]) -> Int:
    """Gauss-Jordan over GF(2) on the rows (piv[r], other[r]), pivoting on the bits of `piv` only.
    Returns the pivot count r; rows r.. have a zero `piv` part, and they span the group elements
    whose `piv` part is zero (row operations are invertible, so no element is lost)."""
    var m = len(piv)
    var rank = 0
    for col in range(64):
        var sel = -1
        for r in range(rank, m):
            if ((piv[r] >> UInt64(col)) & UInt64(1)) != UInt64(0):
                sel = r
                break
        if sel < 0:
            continue
        var tp = piv[sel]
        piv[sel] = piv[rank]
        piv[rank] = tp
        var to = other[sel]
        other[sel] = other[rank]
        other[rank] = to
        for r in range(m):
            if r != rank and ((piv[r] >> UInt64(col)) & UInt64(1)) != UInt64(0):
                piv[r] = piv[r] ^ piv[rank]
                other[r] = other[r] ^ other[rank]
        rank += 1
    return rank


def _group_bases(group: List[SymPauli], what: String) raises -> Tuple[List[UInt64], List[UInt64]]:
    """X-type and Z-type bases of the group generated by `group` (identity elements skipped).
    The X-type elements are the kernel of the Z-projection: Gauss-Jordan on the Z bits (rz pivots)
    leaves them as the rows past rz with zero Z part. Symmetrically for the Z-type elements.
    The group is CSS iff its X-type and Z-type subgroups together have the rank of the whole group.
    dim(whole) = rz + dim(X-type), so that holds iff dim(Z-type) = rz."""
    var xs = List[UInt64]()
    var zs = List[UInt64]()
    for g in group:
        if g.n > 64:
            raise Error("QM-E738: state group on " + String(g.n) + " qubits; the exact reduction takes at most 64. Hint: one block.")
        if g.is_identity():
            continue
        xs.append(_mask_of(g, True))
        zs.append(_mask_of(g, False))
    var ax = xs.copy()
    var az = zs.copy()
    var rz = _eliminate(az, ax)
    var xrest = List[UInt64]()
    for r in range(rz, len(ax)):
        xrest.append(ax[r])
    var bx = xs.copy()
    var bz = zs.copy()
    var rx = _eliminate(bx, bz)
    var zrest = List[UInt64]()
    for r in range(rx, len(bz)):
        zrest.append(bz[r])
    var xb = _xor_basis(xrest)
    var zb = _xor_basis(zrest)
    if len(zb) != rz:
        raise Error("QM-E738: " + what + " is not a CSS state: its stabilizers are not generated by X-type and Z-type elements. Hint: prepare an X- or Z-basis logical state.")
    if len(xb) > 24 or len(zb) > 24:
        raise Error("QM-E738: state group too large for the exact reduction (rank " + String(len(xb)) + " X, "
                    + String(len(zb)) + " Z; limit 24). Hint: use fewer independent stabilizers.")
    return (xb^, zb^)


def reduced_weight(error: SymPauli, state_group: List[SymPauli]) raises -> Tuple[Int, Int]:
    """(X weight, Z weight) of `error` minimised over the X-type and Z-type elements of the
    group generated by `state_group` (CSS generators). Y counts in both parts."""
    if error.n > 64:
        raise Error("QM-E738: error on " + String(error.n) + " qubits; the exact reduction takes at most 64. Hint: one block.")
    for g in state_group:
        if g.n != error.n:
            raise Error("QM-E738: state group element on " + String(g.n) + " qubits, error on " + String(error.n) + ".")
    var bases = _group_bases(state_group, "the state group")
    var ex = _mask_of(error, True)
    var ez = _mask_of(error, False)
    return (_min_weight(ex, bases[0]), _min_weight(ez, bases[1]))


def _state_bases(semantic: Circuit, code: StabilizerCode) raises -> Tuple[List[UInt64], List[UInt64]]:
    """Bases of the prepared state's group: code generators plus the expanded semantic stabilizers."""
    var group = code.generators.copy()
    for s in semantic_state_stabilizers(semantic):
        group.append(expand_logical(s, code, 1))
    return _group_bases(group, "the prepared state")


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------
struct _Walk(Movable):
    var faults: List[Fault]
    var loc_start: List[Int]  # first fault index of each location, then a sentinel = len(faults)
    var xb: List[UInt64]
    var zb: List[UInt64]
    var dmask: UInt64
    var weights: Bool  # False in count_undetected
    var sets: Int
    var undetected: Int
    var dangerous: Int
    var witness: String
    var chosen: List[Int]  # fault indices of the current set

    def __init__(out self, var faults: List[Fault], var loc_start: List[Int], dmask: UInt64):
        self.faults = faults^
        self.loc_start = loc_start^
        self.xb = List[UInt64]()
        self.zb = List[UInt64]()
        self.dmask = dmask
        self.weights = False
        self.sets = 0
        self.undetected = 0
        self.dangerous = 0
        self.witness = String()
        self.chosen = List[Int]()


def _describe(w: _Walk, x: UInt64, z: UInt64, wx: Int, wz: Int) -> String:
    var s = String("{")
    for j in range(len(w.chosen)):
        if j > 0:
            s += "; "
        s += w.faults[w.chosen[j]].label
    s += "}: residual X on data " + _qubit_set(x & w.dmask) + " (reduced weight " + String(wx)
    s += "), Z on data " + _qubit_set(z & w.dmask) + " (reduced weight " + String(wz) + ")"
    return s^


def _walk(mut w: _Walk, size: Int, first_loc: Int, depth: Int, x: UInt64, z: UInt64, rec: UInt64):
    if depth == size:
        w.sets += 1
        if rec != UInt64(0):
            return
        w.undetected += 1
        if not w.weights:
            return
        var wx = _min_weight(x & w.dmask, w.xb)
        var wz = _min_weight(z & w.dmask, w.zb)
        if wx > size or wz > size:
            w.dangerous += 1
            if w.witness.byte_length() == 0:
                w.witness = _describe(w, x, z, wx, wz)
        return
    var nloc = len(w.loc_start) - 1
    for loc in range(first_loc, nloc - (size - depth - 1)):
        for f in range(w.loc_start[loc], w.loc_start[loc + 1]):
            w.chosen.append(f)
            _walk(w, size, loc + 1, depth + 1, x ^ w.faults[f].x, z ^ w.faults[f].z, rec ^ w.faults[f].rec)
            _ = w.chosen.pop()


def _prepare(impl: Circuit, code: StabilizerCode, model: FaultModel) raises -> _Walk:
    """Steps 5 and 7: the verification measurements must be deterministic without faults;
    then every single fault of the model, grouped by location."""
    var sim = TableauSimulator(impl.num_qubits, 0)
    for _ in range(impl.num_clbits):
        sim.creg.append(UInt8(0))
    for i in range(len(impl.ops)):
        var g = impl.ops[i]
        if g.kind == Op.MEASURE and not sim.is_deterministic(g.q0):
            raise Error("QM-E738: measurement at instruction #" + String(i) + " is random without faults. Hint: the check needs deterministic verification measurements (measure stabilizers of the prepared state).")
        sim.run_range(impl, i, i + 1, False, True, List[Float64]())
    var faults = single_faults(impl, model)
    var loc_start = List[Int]()
    for f in range(len(faults)):
        if f == 0 or faults[f].location != faults[f - 1].location:
            loc_start.append(f)
    loc_start.append(len(faults))
    return _Walk(faults^, loc_start^, _data_mask(code.n))


def _set_counts(loc_start: List[Int], tt: Int, max_sets: Int) -> List[Int]:
    """dp[s] = number of fault sets of exactly s faults (one per location), saturated at max_sets + 1."""
    var cap = max_sets + 1
    var dp = List[Int](length=tt + 1, fill=0)
    dp[0] = 1
    for loc in range(len(loc_start) - 1):
        var c = loc_start[loc + 1] - loc_start[loc]
        for s in range(tt, 0, -1):
            dp[s] = min(dp[s] + dp[s - 1] * c, cap)
    return dp^


def fault_tolerant_state_preparation(
    semantic: Circuit, impl: Circuit, code: StabilizerCode, num_ancillas: Int = 0, t: Int = -1,
    model: FaultModel = FaultModel.circuit_level(), max_fault_sets: Int = 5_000_000,
) raises -> FaultToleranceReport:
    """Exact t-fault-tolerance check (Definition 1 of arXiv:2611.25107). See the module docstring.
    t defaults to floor((d-1)/2). For a code of distance 2 or less that is 0: no fault set is
    checked and the verdict is vacuously ok, so pass t explicitly for such codes.
    The fault-free validity check (valid_stabilizer_state_preparation) runs first; a failure is
    returned as ok = False, not raised."""
    _check_scope(impl, code, num_ancillas)
    var tt = t if t >= 0 else (code.d - 1) // 2
    var stats = _cx_stats(impl)
    var vr = valid_stabilizer_state_preparation(semantic, impl, code, 1, num_ancillas)
    if not vr.ok:
        return FaultToleranceReport(False, "the fault-free circuit is not a valid preparation: " + vr.reason,
                                    tt, model.name(), 0, 0, 0, 0, 0, "", stats[0], stats[1])
    var w = _prepare(impl, code, model)
    var bases = _state_bases(semantic, code)
    w.xb = bases[0].copy()
    w.zb = bases[1].copy()
    var dp = _set_counts(w.loc_start, tt, max_fault_sets)
    var total = 0
    for s in range(1, tt + 1):
        total = min(total + dp[s], max_fault_sets + 1)
    if total > max_fault_sets:
        raise Error("QM-E738: t = " + String(tt) + " needs more than " + String(max_fault_sets)
                    + " fault sets (limit " + String(max_fault_sets) + "). Hint: lower t or raise max_fault_sets; the count grows as (single faults)^t / t!.")
    var locations = len(w.loc_start) - 1
    var single = len(w.faults)
    w.weights = True
    for size in range(1, tt + 1):
        _walk(w, size, 0, 0, UInt64(0), UInt64(0), UInt64(0))
    var ok = w.dangerous == 0
    var reason = String("")
    if not ok:
        reason = String(w.dangerous) + " undetected fault sets leave an error heavier than |F|"
    return FaultToleranceReport(ok, reason, tt, model.name(), locations, single, w.sets, w.undetected,
                                w.dangerous, w.witness, stats[0], stats[1])


def count_undetected(semantic: Circuit, impl: Circuit, code: StabilizerCode, num_ancillas: Int = 0,
                     order: Int = 2, model: FaultModel = FaultModel.circuit_level(),
                     max_fault_sets: Int = 5_000_000) raises -> Int:
    """Number of fault sets of exactly `order` faults that no measurement detects (the quantity the
    paper names as the cause of higher logical error rates). Same scope and errors as above.
    It does not run the fault-free validity check (valid_stabilizer_state_preparation): use
    fault_tolerant_state_preparation for a verdict."""
    if order < 1:
        raise Error("QM-E738: order must be at least 1, got " + String(order) + ". Hint: count sets of 1 or more faults.")
    _check_scope(impl, code, num_ancillas)
    var w = _prepare(impl, code, model)
    var dp = _set_counts(w.loc_start, order, max_fault_sets)
    if dp[order] > max_fault_sets:
        raise Error("QM-E738: order " + String(order) + " needs more than " + String(max_fault_sets)
                    + " fault sets (limit " + String(max_fault_sets) + "). Hint: lower the order or raise max_fault_sets.")
    w.weights = False
    _walk(w, order, 0, 0, UInt64(0), UInt64(0), UInt64(0))
    return w.undetected
