"""Phase folding: phase-polynomial rotation merging (T-count reduction).

Idea (Amy-Maslov-Mosca "T-par", PyZX / TKET phase-polynomial passes, Feynman's
sum-over-paths phase folding): in a {CX, X, SWAP, diagonal} region every wire
carries an affine parity  x_S + c  of region variables. A diagonal phase gate
on a wire contributes e^{i a (parity)}; contributions on the *same parity*
commute with everything in between (they are all multiplicative phases of the
path sum), so they are summed into the first occurrence ("anchor") and the
others are deleted. Two rotations of T-angle on equal parities fold into one
Clifford S, so the T-count drops.

Non-linear gates (H, U, RX, SX, the target of CCX/CY/CH/CRX/CRY, ...) do not
end the analysis globally: they only give the touched wire a *fresh* variable
(sum-over-paths view: the gate's amplitude is a separate factor). Diagonal
multi-qubit gates (CZ, CP, CRZ, CCZ) leave parities untouched. RZZ(theta) is
itself a parity term (a xor b) and folds with RZ's on that parity, which can
delete two-qubit gates. Measurements, resets, conditioned ops, noise and
barriers start a new *epoch* (terms never fold across them).

Labels are exact sparse sorted variable lists; lookups go through a 64-bit
Zobrist hash (XOR of per-variable keys, updated in O(1) for CX) plus an exact
list comparison, so the pass is exact and near-linear.
"""

from std.collections import Dict
from std.math import pi
from qumo.ir.gates import Op, op_arity, op_is_unitary
from qumo.ir.param import Param
from qumo.ir.circuit import Circuit, Instruction
from qumo.compiler.common import (
    GateSet, AX_Z, FOUR_PI, TWO_PI, ANG_TOL, is_plain_gate, is_all_wire_op,
    rot_axis, rot_is_r, rot_angle, emit_rotation, wrap_angle,
)


def _zobrist(v: Int) -> UInt64:
    var z = UInt64(v) * 0x9E3779B97F4A7C15 + 0x632BE59BD9B4E019
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _symdiff(a: List[Int], b: List[Int]) -> List[Int]:
    var out = List[Int](capacity=len(a) + len(b))
    var i = 0
    var j = 0
    while i < len(a) and j < len(b):
        if a[i] < b[j]:
            out.append(a[i])
            i += 1
        elif b[j] < a[i]:
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


def _same(a: List[Int], b: List[Int]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


struct PhaseFolder(Movable):
    var labels: List[List[Int]]
    var lhash: List[UInt64]
    var lconst: List[Int]
    var nvars: Int
    var epoch: Int
    var heads: Dict[UInt64, Int]
    var t_next: List[Int]
    var t_label: List[List[Int]]
    var t_epoch: List[Int]
    var t_anchor: List[Int]
    var t_const: List[Int]
    var t_conv_r: List[Bool]
    var t_angle: List[Param]
    var t_count: List[Int]
    var phase: Float64

    def __init__(out self, nq: Int):
        self.labels = List[List[Int]]()
        self.lhash = List[UInt64]()
        self.lconst = List[Int]()
        self.nvars = 0
        self.epoch = 0
        self.heads = Dict[UInt64, Int]()
        self.t_next = List[Int]()
        self.t_label = List[List[Int]]()
        self.t_epoch = List[Int]()
        self.t_anchor = List[Int]()
        self.t_const = List[Int]()
        self.t_conv_r = List[Bool]()
        self.t_angle = List[Param]()
        self.t_count = List[Int]()
        self.phase = 0.0
        for _ in range(nq):
            var v = self.nvars
            self.nvars += 1
            var l = List[Int]()
            l.append(v)
            self.labels.append(l^)
            self.lhash.append(_zobrist(v))
            self.lconst.append(0)

    def fresh(mut self, w: Int):
        var v = self.nvars
        self.nvars += 1
        var l = List[Int]()
        l.append(v)
        self.labels[w] = l^
        self.lhash[w] = _zobrist(v)
        self.lconst[w] = 0

    def cx(mut self, c: Int, t: Int):
        self.labels[t] = _symdiff(self.labels[t], self.labels[c])
        self.lhash[t] ^= self.lhash[c]
        self.lconst[t] ^= self.lconst[c]

    def swap(mut self, a: Int, b: Int):
        var la = self.labels[a].copy()
        self.labels[a] = self.labels[b].copy()
        self.labels[b] = la^
        var h = self.lhash[a]
        self.lhash[a] = self.lhash[b]
        self.lhash[b] = h
        var c = self.lconst[a]
        self.lconst[a] = self.lconst[b]
        self.lconst[b] = c

    def find(self, label: List[Int], h: UInt64) -> Int:
        var t = self.heads.get(h, -1)
        while t >= 0:
            if self.t_epoch[t] == self.epoch and _same(self.t_label[t], label):
                return t
            t = self.t_next[t]
        return -1

    def contribute(mut self, i: Int, label: List[Int], h: UInt64, cst: Int,
                   conv_r: Bool, angle: Param) -> Bool:
        """Register phase gate i; True when it was folded into an earlier anchor."""
        if len(label) == 0:
            if angle.is_symbolic():
                return False
            var v = angle.value()
            if conv_r:
                self.phase += -v / 2 * Float64(1 - 2 * cst)
            else:
                self.phase += v * Float64(cst)
            return True
        var t = self.find(label, h)
        if t < 0:
            var nt = len(self.t_next)
            self.t_next.append(self.heads.get(h, -1))
            self.heads[h] = nt
            self.t_label.append(label.copy())
            self.t_epoch.append(self.epoch)
            self.t_anchor.append(i)
            self.t_const.append(cst)
            self.t_conv_r.append(conv_r)
            self.t_angle.append(angle)
            self.t_count.append(1)
            return False
        var ang = angle
        var ph = 0.0
        var tr = self.t_conv_r[t]
        if conv_r != tr:
            if ang.is_symbolic():
                return False
            if tr:
                ph += ang.value() / 2  # P(a) = e^{i a/2} R(a)
            else:
                ph -= ang.value() / 2  # R(a) = e^{-i a/2} P(a)
        if cst != self.t_const[t]:
            if tr:
                ang = -ang  # RZ(a) on (1 + v) == RZ(-a) on v
            else:
                if ang.is_symbolic():
                    return False
                ph += ang.value()  # P(a) on (1 + v) == e^{i a} P(-a) on v
                ang = -ang
        var ok = False
        var s = self.t_angle[t].try_add(ang, ok)
        if not ok:
            return False
        self.t_angle[t] = s
        self.t_count[t] += 1
        self.phase += ph
        return True


def phase_fold(mut c: Circuit, gs: GateSet) -> Int:
    """Fold diagonal phase rotations on equal parities; returns gates removed."""
    var n = len(c.ops)
    var nq = c.num_qubits
    var f = PhaseFolder(nq)
    var alive = List[Bool](length=n, fill=True)
    var removed = 0
    for i in range(n):
        var g = c.ops[i]
        var k = g.kind
        if is_all_wire_op(g):
            f.epoch += 1
            continue
        var a = op_arity(k)
        if a == 0:
            continue
        if not is_plain_gate(g):
            f.epoch += 1
            for s in range(a):
                var q = g.qubit(s)
                if q >= 0 and q < nq:
                    f.fresh(q)
            continue
        if k == Op.I or k == Op.CZ or k == Op.CP or k == Op.CRZ or k == Op.CCZ:
            continue
        if k == Op.X:
            f.lconst[g.q0] ^= 1
            continue
        if k == Op.CX:
            f.cx(g.q0, g.q1)
            continue
        if k == Op.SWAP:
            f.swap(g.q0, g.q1)
            continue
        if a == 1 and rot_axis(k) == AX_Z:
            var q = g.q0
            var lab = f.labels[q].copy()
            var h = f.lhash[q]
            var cst = f.lconst[q]
            if f.contribute(i, lab, h, cst, rot_is_r(k), rot_angle(g)):
                alive[i] = False
                removed += 1
            continue
        if k == Op.RZZ:
            var lab = _symdiff(f.labels[g.q0], f.labels[g.q1])
            var h = f.lhash[g.q0] ^ f.lhash[g.q1]
            var cst = f.lconst[g.q0] ^ f.lconst[g.q1]
            if f.contribute(i, lab, h, cst, True, g.p0):
                alive[i] = False
                removed += 1
            continue
        if k == Op.CCX:
            f.fresh(g.q2)
            continue
        if k == Op.CSWAP:
            f.fresh(g.q1)
            f.fresh(g.q2)
            continue
        if k == Op.CY or k == Op.CH or k == Op.CRX or k == Op.CRY:
            f.fresh(g.q1)
            continue
        for s in range(a):
            f.fresh(g.qubit(s))
    if removed == 0:
        return 0
    # rewrite anchors of terms that absorbed contributions
    var rep_start = List[Int](length=n, fill=-1)
    var rep_len = List[Int](length=n, fill=0)
    var rep = List[Instruction]()
    var anyset = GateSet()
    var phase = f.phase
    for t in range(len(f.t_anchor)):
        if f.t_count[t] < 2:
            continue
        var i = f.t_anchor[t]
        var g = c.ops[i]
        var start = len(rep)
        var total = f.t_angle[t]
        if g.kind == Op.RZZ:
            if not total.is_symbolic() and abs(wrap_angle(total.value(), FOUR_PI)) < ANG_TOL:
                pass
            elif not total.is_symbolic() and abs(abs(wrap_angle(total.value(), FOUR_PI)) - TWO_PI) < ANG_TOL:
                phase += pi
            else:
                rep.append(Instruction.make(Op.RZZ, g.q0, g.q1, -1, total))
        else:
            var tmp = List[Instruction]()
            var ph = 0.0
            if not emit_rotation(AX_Z, f.t_conv_r[t], total, g.q0, gs, tmp, ph):
                tmp = List[Instruction]()
                ph = 0.0
                if not emit_rotation(AX_Z, f.t_conv_r[t], total, g.q0, anyset, tmp, ph):
                    var kk = Op.RZ if f.t_conv_r[t] else Op.P
                    tmp.append(Instruction.make(kk, g.q0, p0=total))
            for x in tmp:
                rep.append(x)
            phase += ph
        rep_start[i] = start
        rep_len[i] = len(rep) - start
    var out = List[Instruction](capacity=n)
    for i in range(n):
        if not alive[i]:
            continue
        if rep_start[i] >= 0:
            for k in range(rep_len[i]):
                out.append(rep[rep_start[i] + k])
            continue
        out.append(c.ops[i])
    c.ops = out^
    c.global_phase += phase
    return removed
