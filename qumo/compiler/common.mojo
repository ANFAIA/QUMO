"""Shared compiler infrastructure: gate sets, commutation rules, per-wire
linked lists, canonical rotation emission and small numeric helpers.

Design (best-of synthesis):
  * Commutation is decided per wire by the *axis* a gate acts with on that wire
    (Z-, X-, Y-type or "other"), the cheap sufficient rule behind Qiskit's
    CommutationAnalysis fast path and TKET's "commute through" rewrites:
    two gates commute when on every shared wire they act with the same Pauli
    axis. CX is Z-type on its control and X-type on its target, RZZ is Z-type
    on both wires, ... (proof: both gates are block diagonal in the joint
    eigenbasis of those single-wire Paulis).
  * `WireList` is an O(1)-update doubly linked list of live gates per wire over
    the *flat* instruction list (no DAG objects): passes walk predecessors on a
    wire, delete gates in place and compact once at the end - linear time.
  * Rotation families (Z: z,s,sdg,t,tdg,rz,p; X: x,sx,sxdg,rx; Y: y,ry) are
    normalised to `e^{i phase} R(angle)` ("R" convention) or `P_axis(angle)`
    ("P" convention, P_axis = H-conjugated P for X/Y) so merges are exact and
    the global phase is tracked; `emit_rotation` re-materialises a merged
    rotation as the cheapest gate allowed by the target basis.
"""

from std.math import pi, cos, sin, atan2, sqrt, floor
from qumo.ir.gates import (
    Op, M2, op_name, op_arity, op_num_params, op_is_unitary, op_is_symmetric,
    op_is_noise, op_is_annotation, op_inverse_kind, matrix_1q,
)
from qumo.ir.param import Param
from qumo.ir.circuit import Circuit, Instruction

comptime TWO_PI = 6.283185307179586476925
comptime FOUR_PI = 12.56637061435917295385
comptime HALF_PI = 1.5707963267948966192313
comptime QUARTER_PI = 0.7853981633974483096157
comptime ANG_TOL = 1e-10

# per-wire action axis
comptime AX_OTHER = 0
comptime AX_Z = 1
comptime AX_X = 2
comptime AX_Y = 3
comptime AX_ANY = 4


# ----------------------------------------------------------------------------
# Gate sets
# ----------------------------------------------------------------------------
struct GateSet(Copyable, Writable):
    """Set of allowed unitary opcodes. `any` = every gate is allowed."""

    var any: Bool
    var allowed: List[Bool]

    def __init__(out self):
        self.any = True
        self.allowed = List[Bool](length=Op.NUM_OPS, fill=True)

    def __init__(out self, kinds: List[Int]):
        if len(kinds) == 0:
            self.any = True
            self.allowed = List[Bool](length=Op.NUM_OPS, fill=True)
            return
        self.any = False
        self.allowed = List[Bool](length=Op.NUM_OPS, fill=False)
        for k in kinds:
            if k >= 0 and k < Op.NUM_OPS:
                self.allowed[k] = True

    @staticmethod
    def all_but_3q() -> GateSet:
        var g = GateSet()
        g.any = False
        g.allowed[Op.CCX] = False
        g.allowed[Op.CCZ] = False
        g.allowed[Op.CSWAP] = False
        return g^

    def has(self, k: Int) -> Bool:
        if k < 0 or k >= Op.NUM_OPS:
            return False
        if not op_is_unitary(k):
            return True
        return self.allowed[k]

    def has_z_rot(self) -> Bool:
        return self.has(Op.RZ) or self.has(Op.P)

    def kinds(self) -> List[Int]:
        var out = List[Int]()
        if self.any:
            return out^
        for k in range(Op.CCZ + 1):
            if self.allowed[k]:
                out.append(k)
        return out^

    def write_to(self, mut w: Some[Writer]):
        if self.any:
            w.write("{any}")
            return
        w.write("{")
        var first = True
        for k in range(Op.CCZ + 1):
            if self.allowed[k]:
                if not first:
                    w.write(",")
                first = False
                w.write(op_name(k))
        w.write("}")


def basis_str(kinds: List[Int]) -> String:
    if len(kinds) == 0:
        return "any"
    var s = String()
    for i in range(len(kinds)):
        if i > 0:
            s += ","
        s += op_name(kinds[i])
    return s


# ----------------------------------------------------------------------------
# Numbers
# ----------------------------------------------------------------------------
def wrap_angle(a: Float64, period: Float64) -> Float64:
    """Representative of `a` modulo `period` in (-period/2, period/2]."""
    var r = a - period * floor(a / period + 0.5)
    if r <= -period / 2:
        r += period
    if r > period / 2:
        r -= period
    return r


def near_mod(a: Float64, target: Float64, period: Float64) -> Bool:
    return abs(wrap_angle(a - target, period)) < ANG_TOL


def fmt_num(x: Float64, digits: Int = 3) -> String:
    """Compact float formatting for reports (agents parse these)."""
    if x != x:
        return "nan"
    if x == 0.0:
        return "0"
    var ax = abs(x)
    var sign = String("-") if x < 0 else String("")
    if ax >= 1e-3 and ax < 1e9:
        var scale = 1.0
        for _ in range(digits):
            scale *= 10.0
        var v = Int(ax * scale + 0.5)
        var ip = v // Int(scale)
        var fp = v - ip * Int(scale)
        var s = sign + String(ip)
        if fp > 0:
            var fs = String(fp)
            var pad = digits - fs.byte_length()
            var z = String()
            for _ in range(pad):
                z += "0"
            fs = z + fs
            # trim trailing zeros
            var bs = fs.as_bytes()
            var end = len(bs)
            while end > 0 and Int(bs[end - 1]) == ord("0"):
                end -= 1
            s += "." + String(unsafe_from_utf8=bs[0:end])
        return s
    # scientific
    var e = 0
    var m = ax
    while m >= 10.0:
        m /= 10.0
        e += 1
    while m < 1.0:
        m *= 10.0
        e -= 1
    var mi = Int(m * 100.0 + 0.5)
    if mi >= 1000:
        mi = mi // 10
        e += 1
    var ms = String(mi // 100)
    var frac = mi % 100
    if frac != 0:
        if frac % 10 == 0:
            ms += "." + String(frac // 10)
        else:
            ms += "." + (String("0") if frac < 10 else String("")) + String(frac)
    return sign + ms + "e" + String(e)


# ----------------------------------------------------------------------------
# 2x2 matrices
# ----------------------------------------------------------------------------
def m2_phase_to(a: M2, b: M2, mut phase: Float64, tol: Float64 = 1e-9) -> Bool:
    """True when a = e^{i phase} b (phase written on success)."""
    var best = 0
    var bm = 0.0
    for k in range(4):
        var v = b.re[k] * b.re[k] + b.im[k] * b.im[k]
        if v > bm:
            bm = v
            best = k
    if bm < 1e-24:
        return False
    var gr = (a.re[best] * b.re[best] + a.im[best] * b.im[best]) / bm
    var gi = (a.im[best] * b.re[best] - a.re[best] * b.im[best]) / bm
    var gn = sqrt(gr * gr + gi * gi)
    if abs(gn - 1.0) > 1e-6:
        return False
    gr /= gn
    gi /= gn
    for k in range(4):
        var er = gr * b.re[k] - gi * b.im[k] - a.re[k]
        var ei = gr * b.im[k] + gi * b.re[k] - a.im[k]
        if abs(er) > tol or abs(ei) > tol:
            return False
    phase = atan2(gi, gr)
    return True


def m2_identity_phase(m: M2, mut phase: Float64, tol: Float64 = 1e-10) -> Bool:
    """True when m = e^{i phase} I."""
    if abs(m.re[1]) > tol or abs(m.im[1]) > tol or abs(m.re[2]) > tol or abs(m.im[2]) > tol:
        return False
    if abs(m.re[0] - m.re[3]) > tol or abs(m.im[0] - m.im[3]) > tol:
        return False
    phase = atan2(m.im[0], m.re[0])
    return True


def inst_matrix_1q(g: Instruction) -> M2:
    """Matrix of a bound 1-qubit instruction."""
    return matrix_1q(g.kind, g.p0.value(), g.p1.value(), g.p2.value())


def seq_matrix_1q(seq: List[Instruction], start: Int = 0) -> M2:
    """Product (time order) of bound 1-qubit instructions seq[start:]."""
    var m = M2.identity()
    for i in range(start, len(seq)):
        m = inst_matrix_1q(seq[i]) @ m
    return m


# ----------------------------------------------------------------------------
# Instruction predicates
# ----------------------------------------------------------------------------
@always_inline
def is_plain_gate(g: Instruction) -> Bool:
    """Unconditioned unitary gate (the only thing optimisation passes rewrite)."""
    return op_is_unitary(g.kind) and g.cond < 0


@always_inline
def is_bound(g: Instruction) -> Bool:
    return not g.is_parametric()


@always_inline
def is_all_wire_op(g: Instruction) -> Bool:
    """Barrier / tick: synchronise every qubit."""
    return g.kind == Op.BARRIER or g.kind == Op.TICK


def wire_axis(k: Int, slot: Int) -> Int:
    """Pauli axis a gate commutes with on its `slot`-th wire (AX_OTHER if none)."""
    if k == Op.I:
        return AX_ANY
    if k == Op.Z or k == Op.S or k == Op.SDG or k == Op.T or k == Op.TDG or k == Op.RZ or k == Op.P:
        return AX_Z
    if k == Op.X or k == Op.SX or k == Op.SXDG or k == Op.RX:
        return AX_X
    if k == Op.Y or k == Op.RY:
        return AX_Y
    if k == Op.CZ or k == Op.CP or k == Op.CRZ or k == Op.RZZ or k == Op.CCZ:
        return AX_Z
    if k == Op.RXX:
        return AX_X
    if k == Op.RYY:
        return AX_Y
    if k == Op.CX or k == Op.CRX:
        return AX_Z if slot == 0 else AX_X
    if k == Op.CY or k == Op.CRY:
        return AX_Z if slot == 0 else AX_Y
    if k == Op.CH:
        return AX_Z if slot == 0 else AX_OTHER
    if k == Op.CCX:
        return AX_Z if slot < 2 else AX_X
    if k == Op.CSWAP:
        return AX_Z if slot == 0 else AX_OTHER
    return AX_OTHER


def same_gate(a: Instruction, b: Instruction) -> Bool:
    if a.kind != b.kind or a.q0 != b.q0:
        return False
    var ar = op_arity(a.kind)
    if ar >= 2 and a.q1 != b.q1:
        return False
    if ar >= 3 and a.q2 != b.q2:
        return False
    var np = op_num_params(a.kind)
    if np >= 1 and not (a.p0 == b.p0):
        return False
    if np >= 2 and not (a.p1 == b.p1):
        return False
    if np >= 3 and not (a.p2 == b.p2):
        return False
    return True


def commutes(a: Instruction, b: Instruction) -> Bool:
    """Sufficient commutation test (per-wire Pauli-axis rule)."""
    if not is_plain_gate(a) or not is_plain_gate(b):
        return False
    var na = op_arity(a.kind)
    var nb = op_arity(b.kind)
    var shared = False
    for i in range(na):
        var w = a.qubit(i)
        for j in range(nb):
            if b.qubit(j) == w:
                shared = True
                var ta = wire_axis(a.kind, i)
                var tb = wire_axis(b.kind, j)
                if ta == AX_ANY or tb == AX_ANY:
                    continue
                if ta == AX_OTHER or ta != tb:
                    return same_gate(a, b)
    _ = shared
    return True


def same_qubit_set(a: Instruction, b: Instruction) -> Bool:
    """Same operands up to the symmetries of the gate kinds (a, b same arity)."""
    var na = op_arity(a.kind)
    if na != op_arity(b.kind):
        return False
    if na == 1:
        return a.q0 == b.q0
    if na == 2:
        if a.q0 == b.q0 and a.q1 == b.q1:
            return True
        return a.q0 == b.q1 and a.q1 == b.q0 and op_is_symmetric(a.kind) and op_is_symmetric(b.kind)
    if a.q0 == b.q0 and a.q1 == b.q1 and a.q2 == b.q2:
        return True
    if a.kind != b.kind:
        return False
    if a.kind == Op.CCZ:
        var sa = (1 << a.q0) | (1 << a.q1) | (1 << a.q2)
        var sb = (1 << b.q0) | (1 << b.q1) | (1 << b.q2)
        return sa == sb and a.q0 < 62 and a.q1 < 62 and a.q2 < 62 and b.q0 < 62 and b.q1 < 62 and b.q2 < 62
    if a.kind == Op.CCX:
        return a.q2 == b.q2 and a.q0 == b.q1 and a.q1 == b.q0
    if a.kind == Op.CSWAP:
        return a.q0 == b.q0 and a.q1 == b.q2 and a.q2 == b.q1
    return False


# ----------------------------------------------------------------------------
# Per-wire doubly linked lists over a flat instruction list
# ----------------------------------------------------------------------------
struct WireList(Movable):
    var prev: List[Int]
    var next: List[Int]
    var top: List[Int]

    def __init__(out self, num_ops: Int, num_qubits: Int):
        self.prev = List[Int](length=3 * num_ops + 3, fill=-1)
        self.next = List[Int](length=3 * num_ops + 3, fill=-1)
        self.top = List[Int](length=num_qubits, fill=-1)

    @staticmethod
    @always_inline
    def slot(g: Instruction, w: Int) -> Int:
        if g.q0 == w:
            return 0
        if g.q1 == w:
            return 1
        return 2

    def push(mut self, ops: List[Instruction], i: Int):
        var g = ops[i]
        var a = op_arity(g.kind)
        for s in range(a):
            var w = g.qubit(s)
            var t = self.top[w]
            self.prev[3 * i + s] = t
            self.next[3 * i + s] = -1
            if t >= 0:
                self.next[3 * t + WireList.slot(ops[t], w)] = i
            self.top[w] = i

    def remove(mut self, ops: List[Instruction], i: Int):
        var g = ops[i]
        var a = op_arity(g.kind)
        for s in range(a):
            var w = g.qubit(s)
            var p = self.prev[3 * i + s]
            var n = self.next[3 * i + s]
            if p >= 0:
                self.next[3 * p + WireList.slot(ops[p], w)] = n
            if n >= 0:
                self.prev[3 * n + WireList.slot(ops[n], w)] = p
            elif self.top[w] == i:
                self.top[w] = p

    @always_inline
    def before(self, ops: List[Instruction], i: Int, w: Int) -> Int:
        return self.prev[3 * i + WireList.slot(ops[i], w)]

    def cut(mut self, w: Int):
        self.top[w] = -1

    def cut_all(mut self):
        for w in range(len(self.top)):
            self.top[w] = -1

    def cut_op(mut self, g: Instruction):
        """Make `g` an optimisation barrier on its wires."""
        if is_all_wire_op(g):
            self.cut_all()
            return
        for s in range(op_arity(g.kind)):
            var w = g.qubit(s)
            if w >= 0 and w < len(self.top):
                self.top[w] = -1


def compact(mut c: Circuit, alive: List[Bool]):
    var out = List[Instruction](capacity=len(c.ops))
    for i in range(len(c.ops)):
        if alive[i]:
            out.append(c.ops[i])
    c.ops = out^


def shell_like(c: Circuit, num_qubits: Int) -> Circuit:
    """Copy of all circuit metadata (params, condition table, records, counters,
    phase) without ops; `empty_like` keeps register-condition offsets valid."""
    var out = c.empty_like()
    out.num_qubits = num_qubits
    out.records = c.records.copy()
    out.num_measurements = c.num_measurements
    out.num_detectors = c.num_detectors
    out.num_observables = c.num_observables
    return out^


# ----------------------------------------------------------------------------
# Rotation families
# ----------------------------------------------------------------------------
def rot_axis(k: Int) -> Int:
    if k == Op.Z or k == Op.S or k == Op.SDG or k == Op.T or k == Op.TDG or k == Op.RZ or k == Op.P:
        return AX_Z
    if k == Op.X or k == Op.SX or k == Op.SXDG or k == Op.RX:
        return AX_X
    if k == Op.Y or k == Op.RY:
        return AX_Y
    return AX_OTHER


@always_inline
def rot_is_r(k: Int) -> Bool:
    """R convention (exp(-i a P/2)) vs P convention (e^{i a/2} exp(-i a P/2))."""
    return k == Op.RX or k == Op.RY or k == Op.RZ


def rot_angle(g: Instruction) -> Param:
    var k = g.kind
    if k == Op.Z or k == Op.X or k == Op.Y:
        return Param(pi)
    if k == Op.S or k == Op.SX:
        return Param(HALF_PI)
    if k == Op.SDG or k == Op.SXDG:
        return Param(-HALF_PI)
    if k == Op.T:
        return Param(QUARTER_PI)
    if k == Op.TDG:
        return Param(-QUARTER_PI)
    return g.p0


def r_kind(axis: Int) -> Int:
    if axis == AX_Z:
        return Op.RZ
    if axis == AX_X:
        return Op.RX
    return Op.RY


def _named_p(axis: Int, lam: Float64, mut k1: Int, mut k2: Int):
    """Named gate(s) equal to P_axis(lam) exactly; k1 = -1 when none."""
    k1 = -1
    k2 = -1
    if axis == AX_Z:
        if near_mod(lam, pi, TWO_PI): k1 = Op.Z
        elif near_mod(lam, HALF_PI, TWO_PI): k1 = Op.S
        elif near_mod(lam, -HALF_PI, TWO_PI): k1 = Op.SDG
        elif near_mod(lam, QUARTER_PI, TWO_PI): k1 = Op.T
        elif near_mod(lam, -QUARTER_PI, TWO_PI): k1 = Op.TDG
        elif near_mod(lam, 3 * QUARTER_PI, TWO_PI):
            k1 = Op.S
            k2 = Op.T
        elif near_mod(lam, -3 * QUARTER_PI, TWO_PI):
            k1 = Op.SDG
            k2 = Op.TDG
    elif axis == AX_X:
        if near_mod(lam, pi, TWO_PI): k1 = Op.X
        elif near_mod(lam, HALF_PI, TWO_PI): k1 = Op.SX
        elif near_mod(lam, -HALF_PI, TWO_PI): k1 = Op.SXDG
    else:
        if near_mod(lam, pi, TWO_PI): k1 = Op.Y


def _u_for(axis: Int, theta: Param, q: Int) -> Instruction:
    """U gate equal (exactly) to R_axis(theta) for X/Y, P(theta) for Z."""
    if axis == AX_Z:
        return Instruction.make(Op.U, q, p0=0.0, p1=0.0, p2=theta)
    if axis == AX_X:
        return Instruction.make(Op.U, q, p0=theta, p1=-HALF_PI, p2=HALF_PI)
    return Instruction.make(Op.U, q, p0=theta, p1=0.0, p2=0.0)


def _emit_p_const(axis: Int, lam_in: Float64, q: Int, gs: GateSet,
                  mut out: List[Instruction], mut phase: Float64) -> Bool:
    var lam = wrap_angle(lam_in, TWO_PI)
    if abs(lam) < ANG_TOL:
        return True
    var k1 = -1
    var k2 = -1
    _named_p(axis, lam, k1, k2)
    if k1 >= 0 and gs.has(k1) and (k2 < 0 or gs.has(k2)):
        out.append(Instruction.make(k1, q))
        if k2 >= 0:
            out.append(Instruction.make(k2, q))
        return True
    if axis == AX_Z and gs.has(Op.P):
        out.append(Instruction.make(Op.P, q, p0=lam))
        return True
    var rk = r_kind(axis)
    if gs.has(rk):
        out.append(Instruction.make(rk, q, p0=lam))
        phase += lam / 2
        return True
    if gs.has(Op.U):
        out.append(_u_for(axis, lam, q))
        if axis != AX_Z:
            phase += lam / 2
        return True
    return False


def emit_rotation(axis: Int, is_r: Bool, angle: Param, q: Int, gs: GateSet,
                  mut out: List[Instruction], mut phase: Float64) -> Bool:
    """Append the cheapest allowed gate(s) equal to the rotation; False if none.

    R convention: R_axis(angle). P convention: P_axis(angle) = e^{i angle/2} R_axis(angle).
    On failure neither `out` nor `phase` is modified.
    """
    var ph = 0.0
    var tmp = List[Instruction]()
    var ok = False
    if not angle.is_symbolic():
        var a = angle.value()
        if is_r:
            a = wrap_angle(a, FOUR_PI)
            if abs(a) < ANG_TOL:
                ok = True
            elif abs(abs(a) - TWO_PI) < ANG_TOL:
                ph += pi
                ok = True
            else:
                var rk = r_kind(axis)
                if gs.has(rk):
                    tmp.append(Instruction.make(rk, q, p0=a))
                    ok = True
                else:
                    # R(a) = e^{-i a/2} P(a)
                    ph -= a / 2
                    ok = _emit_p_const(axis, a, q, gs, tmp, ph)
        else:
            ok = _emit_p_const(axis, a, q, gs, tmp, ph)
    else:
        if is_r:
            var rk = r_kind(axis)
            if gs.has(rk):
                tmp.append(Instruction.make(rk, q, p0=angle))
                ok = True
            elif gs.has(Op.U) and axis != AX_Z:
                tmp.append(_u_for(axis, angle, q))
                ok = True
        elif axis == AX_Z:
            if gs.has(Op.P):
                tmp.append(Instruction.make(Op.P, q, p0=angle))
                ok = True
            elif gs.has(Op.U):
                tmp.append(_u_for(axis, angle, q))
                ok = True
    if not ok:
        return False
    for t in tmp:
        out.append(t)
    phase += ph
    return True


def combine_rotations(h: Instruction, g: Instruction, gs: GateSet,
                      mut out: List[Instruction], mut phase: Float64) -> Bool:
    """Merge two same-axis 1q rotations on one wire (h first, then g).

    The result keeps h's convention (so basis gates stay basis gates).
    """
    var axis = rot_axis(h.kind)
    if axis == AX_OTHER or axis != rot_axis(g.kind):
        return False
    var hr = rot_is_r(h.kind)
    var gr = rot_is_r(g.kind)
    var ha = rot_angle(h)
    var ga = rot_angle(g)
    var ph = 0.0
    var conv_r = hr
    if hr != gr:
        if not ga.is_symbolic():
            # convert g into h's convention
            if hr:
                ph += ga.value() / 2  # P(a) = e^{i a/2} R(a)
            else:
                ph -= ga.value() / 2  # R(a) = e^{-i a/2} P(a)
        elif not ha.is_symbolic():
            if gr:
                ph += ha.value() / 2
            else:
                ph -= ha.value() / 2
            conv_r = gr
        else:
            return False
    var ok = False
    var total = ha.try_add(ga, ok)
    if not ok:
        return False
    var tmp = List[Instruction]()
    if not emit_rotation(axis, conv_r, total, h.q0, gs, tmp, ph):
        return False
    for t in tmp:
        out.append(t)
    phase += ph
    return True
