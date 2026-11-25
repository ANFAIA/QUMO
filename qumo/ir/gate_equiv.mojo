"""Dense equality of two small gate lists up to a global phase.

Used by the OpenQASM parser to decide whether a user `gate` definition that
carries the name of a native gate really is that gate. Standalone on purpose
(the IR layer must not import the compiler): bound unitary 1q, 2q and 3q
opcodes on at most `MAX_EQUIV_QUBITS` qubits, little-endian. Anything else
(noise, measurement, classical conditions, unbound parameters) raises, and the
caller treats that as "cannot compare".
"""

from qumo.ir.circuit import Instruction
from qumo.ir.gates import Op, op_arity, matrix_1q, matrix_2q

comptime MAX_EQUIV_QUBITS = 4


def _apply(g: Instruction, n: Int, mut re: List[Float64], mut im: List[Float64]) raises:
    var k = g.kind
    if k == Op.BARRIER or k == Op.TICK or k == Op.I:
        return
    if g.cond >= 0 or k > Op.CCZ or g.is_parametric():
        raise Error("cannot compare '" + String(g) + "'")
    var dim = 1 << n
    var a = op_arity(k)
    if a == 1:
        var m = matrix_1q(k, g.p0.value(), g.p1.value(), g.p2.value())
        var bq = 1 << g.q0
        for i in range(dim):
            if (i & bq) != 0:
                continue
            var j = i | bq
            var ar = re[i]
            var ai = im[i]
            var br = re[j]
            var bi = im[j]
            re[i] = m.re[0] * ar - m.im[0] * ai + m.re[1] * br - m.im[1] * bi
            im[i] = m.re[0] * ai + m.im[0] * ar + m.re[1] * bi + m.im[1] * br
            re[j] = m.re[2] * ar - m.im[2] * ai + m.re[3] * br - m.im[3] * bi
            im[j] = m.re[2] * ai + m.im[2] * ar + m.re[3] * bi + m.im[3] * br
    elif a == 2:
        var m = matrix_2q(k, g.p0.value())
        var b0 = 1 << g.q0
        var b1 = 1 << g.q1
        for i in range(dim):
            if (i & b0) != 0 or (i & b1) != 0:
                continue
            var idx: List[Int] = [i, i | b0, i | b1, i | b0 | b1]
            var vr = List[Float64](length=4, fill=0.0)
            var vi = List[Float64](length=4, fill=0.0)
            for t in range(4):
                vr[t] = re[idx[t]]
                vi[t] = im[idx[t]]
            for r in range(4):
                var sr = 0.0
                var si = 0.0
                for c in range(4):
                    var mr = m.re[r * 4 + c]
                    var mi = m.im[r * 4 + c]
                    sr += mr * vr[c] - mi * vi[c]
                    si += mr * vi[c] + mi * vr[c]
                re[idx[r]] = sr
                im[idx[r]] = si
    elif a == 3:
        var b0 = 1 << g.q0
        var b1 = 1 << g.q1
        var b2 = 1 << g.q2
        for i in range(dim):
            if k == Op.CCZ:
                # sign flip when all three bits are set
                if (i & b0) != 0 and (i & b1) != 0 and (i & b2) != 0:
                    re[i] = -re[i]
                    im[i] = -im[i]
            elif k == Op.CCX:
                # q0, q1 control, q2 target
                if (i & b0) != 0 and (i & b1) != 0 and (i & b2) == 0:
                    var j = i | b2
                    var tr = re[i]
                    var ti = im[i]
                    re[i] = re[j]
                    im[i] = im[j]
                    re[j] = tr
                    im[j] = ti
            elif k == Op.CSWAP:
                # q0 control; swap q1 and q2
                if (i & b0) != 0 and (i & b1) != 0 and (i & b2) == 0:
                    var j = (i & ~b1) | b2
                    var tr = re[i]
                    var ti = im[i]
                    re[i] = re[j]
                    im[i] = im[j]
                    re[j] = tr
                    im[j] = ti
            else:
                raise Error("cannot compare '" + String(g) + "'")
    else:
        raise Error("cannot compare '" + String(g) + "'")


def equal_up_to_phase(a: List[Instruction], b: List[Instruction], n: Int, tol: Float64 = 1e-9) raises -> Bool:
    """Whether the two lists have the same unitary up to a global phase.
    Raises when n > MAX_EQUIV_QUBITS or a gate cannot be applied densely."""
    if n < 1 or n > MAX_EQUIV_QUBITS:
        raise Error("cannot compare on " + String(n) + " qubits")
    var dim = 1 << n
    var have = False
    var pr = 1.0
    var pi_ = 0.0
    for col in range(dim):
        var ar = List[Float64](length=dim, fill=0.0)
        var ai = List[Float64](length=dim, fill=0.0)
        ar[col] = 1.0
        var br = ar.copy()
        var bi = ai.copy()
        for g in a:
            _apply(g, n, ar, ai)
        for g in b:
            _apply(g, n, br, bi)
        if not have:
            var best = 0
            var bv = -1.0
            for i in range(dim):
                var v = ar[i] * ar[i] + ai[i] * ai[i]
                if v > bv:
                    bv = v
                    best = i
            var den = ar[best] * ar[best] + ai[best] * ai[best]
            pr = (br[best] * ar[best] + bi[best] * ai[best]) / den
            pi_ = (bi[best] * ar[best] - br[best] * ai[best]) / den
            have = True
        for i in range(dim):
            var er = pr * ar[i] - pi_ * ai[i]
            var ei = pr * ai[i] + pi_ * ar[i]
            if abs(er - br[i]) > tol or abs(ei - bi[i]) > tol:
                return False
    return True
