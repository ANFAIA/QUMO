"""Two-qubit block consolidation with exact KAK (Weyl) resynthesis.

Best-of synthesis:
  * Qiskit `ConsolidateBlocks` + `UnitarySynthesis` / TKET `KAKDecomposition`:
    maximal runs of gates confined to one qubit pair are multiplied into a 4x4
    unitary and re-expressed with the minimal number of CX (0, 1, 2 or 3).
  * Weyl decomposition in the magic basis (Kraus-Cirac / Vatan-Williams):
    U = e^{ig} (K1l x K1r) exp(i(a XX + b YY + c ZZ)) (K2l x K2r). M = U_B^T U_B
    is complex-symmetric unitary, so its real and imaginary parts commute and
    are diagonalised together by one real orthogonal P (cyclic Jacobi on a
    seeded linear combination, deterministic). Coordinates are reduced modulo
    pi/2 (the remainder is a local Pauli), the number of zero coordinates picks
    the 0/1/2/3-CX template, and constant Clifford corrections map the
    template core onto exp(i(aXX+bYY+cZZ)).
  * Safety net: every resynthesised block is multiplied out and compared with
    the original 4x4 unitary (global phase measured, not derived); on any
    mismatch the original gates are kept. A block is replaced only when it
    lowers the two-qubit cost (native entanglers of the basis count 1, iSWAP
    targets pay 2 per CX). Swaps adjacent to a gate on the same pair are
    absorbed (SWAP.CX = 2 CX, SWAP.CP = 3 CX).
  * Continuous-angle entanglers (ion MS / ZZ): the canonical core is emitted as
    one rxx/ryy/rzz per non-zero Weyl coordinate (CX.RZ.CX -> 1 RXX).
  * Speed: compile-time-unrolled 4x4 complex products and Jacobi sweeps
    (~5 us per Weyl decomposition); pure-CX blocks of cost <= 3 are skipped.
"""

from std.math import pi, cos, sin, atan2, sqrt
from qumo.ir.gates import (
    Op, M2, M4, matrix_1q, matrix_2q, op_arity, op_is_unitary, zyz_decompose,
)
from qumo.ir.circuit import Circuit, Instruction
from qumo.compiler.common import (
    GateSet, HALF_PI, QUARTER_PI, is_plain_gate, is_bound, is_all_wire_op, inst_matrix_1q,
)
from qumo.compiler.synth import synth_1q
from max.algorithm import parallelize


# ----------------------------------------------------------------------------
# 4x4 complex helpers (row-major, index = b(q_lo) + 2 b(q_hi))
# ----------------------------------------------------------------------------
def mm4(a: M4, b: M4) -> M4:
    """a @ b with compile-time indices (row-broadcast SIMD; ~10x faster than M4.__matmul__)."""
    var o = M4()
    comptime for r in range(4):
        var rr = SIMD[DType.float64, 4](0)
        var ri = SIMD[DType.float64, 4](0)
        comptime for k in range(4):
            var ar = a.re[r * 4 + k]
            var ai = a.im[r * 4 + k]
            var br = b.re.slice[4, offset=k * 4]()
            var bi = b.im.slice[4, offset=k * 4]()
            rr += ar * br - ai * bi
            ri += ar * bi + ai * br
        comptime for c in range(4):
            o.re[r * 4 + c] = rr[c]
            o.im[r * 4 + c] = ri[c]
    return o^


def m4_dag(a: M4) -> M4:
    var o = M4()
    comptime for r in range(4):
        comptime for c in range(4):
            o.re[r * 4 + c] = a.re[c * 4 + r]
            o.im[r * 4 + c] = -a.im[c * 4 + r]
    return o^


def m4_transpose(a: M4) -> M4:
    var o = M4()
    comptime for r in range(4):
        comptime for c in range(4):
            o.re[r * 4 + c] = a.re[c * 4 + r]
            o.im[r * 4 + c] = a.im[c * 4 + r]
    return o^


def m4_scale(a: M4, cr: Float64, ci: Float64) -> M4:
    var o = M4()
    o.re = a.re * cr - a.im * ci
    o.im = a.re * ci + a.im * cr
    return o^


def m4_kron(hi: M2, lo: M2) -> M4:
    """hi acts on the high bit (q_hi), lo on the low bit (q_lo)."""
    var o = M4()
    comptime for r in range(4):
        comptime for c in range(4):
            comptime a = (r >> 1) * 2 + (c >> 1)
            comptime b = (r & 1) * 2 + (c & 1)
            o.re[r * 4 + c] = hi.re[a] * lo.re[b] - hi.im[a] * lo.im[b]
            o.im[r * 4 + c] = hi.re[a] * lo.im[b] + hi.im[a] * lo.re[b]
    return o^


def m4_det(a: M4, mut dr: Float64, mut di: Float64):
    """Determinant by Gaussian elimination with partial pivoting."""
    var re = a.re
    var im = a.im
    dr = 1.0
    di = 0.0
    for col in range(4):
        var piv = col
        var best = -1.0
        for r in range(col, 4):
            var v = re[r * 4 + col] * re[r * 4 + col] + im[r * 4 + col] * im[r * 4 + col]
            if v > best:
                best = v
                piv = r
        if best < 1e-300:
            dr = 0.0
            di = 0.0
            return
        if piv != col:
            for c in range(4):
                var tr = re[col * 4 + c]
                var ti = im[col * 4 + c]
                re[col * 4 + c] = re[piv * 4 + c]
                im[col * 4 + c] = im[piv * 4 + c]
                re[piv * 4 + c] = tr
                im[piv * 4 + c] = ti
            dr = -dr
            di = -di
        var pr = re[col * 4 + col]
        var pim = im[col * 4 + col]
        var ndr = dr * pr - di * pim
        di = dr * pim + di * pr
        dr = ndr
        var den = pr * pr + pim * pim
        for r in range(col + 1, 4):
            # f = a[r][col] / pivot
            var xr = re[r * 4 + col]
            var xi = im[r * 4 + col]
            var fr = (xr * pr + xi * pim) / den
            var fi = (xi * pr - xr * pim) / den
            for c in range(col, 4):
                var yr = re[col * 4 + c]
                var yi = im[col * 4 + c]
                re[r * 4 + c] -= fr * yr - fi * yi
                im[r * 4 + c] -= fr * yi + fi * yr


def m4_phase_to(a: M4, b: M4, mut phase: Float64, tol: Float64 = 1e-8) -> Bool:
    """True when a = e^{i phase} b."""
    var best = 0
    var bm = 0.0
    for k in range(16):
        var v = b.re[k] * b.re[k] + b.im[k] * b.im[k]
        if v > bm:
            bm = v
            best = k
    if bm < 1e-20:
        return False
    var gr = (a.re[best] * b.re[best] + a.im[best] * b.im[best]) / bm
    var gi = (a.im[best] * b.re[best] - a.re[best] * b.im[best]) / bm
    var gn = sqrt(gr * gr + gi * gi)
    if abs(gn - 1.0) > 1e-6:
        return False
    gr /= gn
    gi /= gn
    for k in range(16):
        var er = gr * b.re[k] - gi * b.im[k] - a.re[k]
        var ei = gr * b.im[k] + gi * b.re[k] - a.im[k]
        if abs(er) > tol or abs(ei) > tol:
            return False
    phase = atan2(gi, gr)
    return True


def magic_basis() -> M4:
    """B = 1/sqrt2 [[1,0,0,i],[0,i,1,0],[0,i,-1,0],[1,0,0,-i]]."""
    var r = 0.70710678118654752440
    var b = M4()
    b.set(0, 0, r)
    b.set(0, 3, 0.0, r)
    b.set(1, 1, 0.0, r)
    b.set(1, 2, r)
    b.set(2, 1, 0.0, r)
    b.set(2, 2, -r)
    b.set(3, 0, r)
    b.set(3, 3, 0.0, -r)
    return b^


def jacobi_eigh(s_in: SIMD[DType.float64, 16], mut v: SIMD[DType.float64, 16]):
    """Eigenvectors (columns of v) of a real symmetric 4x4 matrix (cyclic Jacobi)."""
    var s = s_in
    v = SIMD[DType.float64, 16](0)
    comptime for k in range(4):
        v[k * 5] = 1.0
    for _ in range(60):
        var off = 0.0
        comptime for p in range(4):
            comptime for q in range(p + 1, 4):
                off += s[p * 4 + q] * s[p * 4 + q]
        if off < 1e-30:
            return
        comptime for p in range(4):
            comptime for q in range(p + 1, 4):
                var apq = s[p * 4 + q]
                if abs(apq) >= 1e-300:
                    var theta = (s[q * 4 + q] - s[p * 4 + p]) / (2 * apq)
                    var t: Float64
                    if theta >= 0:
                        t = 1.0 / (theta + sqrt(theta * theta + 1))
                    else:
                        t = -1.0 / (-theta + sqrt(theta * theta + 1))
                    var c = 1.0 / sqrt(t * t + 1)
                    var sn = t * c
                    comptime for k in range(4):
                        var skp = s[k * 4 + p]
                        var skq = s[k * 4 + q]
                        s[k * 4 + p] = c * skp - sn * skq
                        s[k * 4 + q] = sn * skp + c * skq
                    comptime for k in range(4):
                        var spk = s[p * 4 + k]
                        var sqk = s[q * 4 + k]
                        s[p * 4 + k] = c * spk - sn * sqk
                        s[q * 4 + k] = sn * spk + c * sqk
                    comptime for k in range(4):
                        var vkp = v[k * 4 + p]
                        var vkq = v[k * 4 + q]
                        v[k * 4 + p] = c * vkp - sn * vkq
                        v[k * 4 + q] = sn * vkp + c * vkq


def _det2(a: M2, mut dr: Float64, mut di: Float64):
    dr = a.re[0] * a.re[3] - a.im[0] * a.im[3] - (a.re[1] * a.re[2] - a.im[1] * a.im[2])
    di = a.re[0] * a.im[3] + a.im[0] * a.re[3] - (a.re[1] * a.im[2] + a.im[1] * a.re[2])


def _csqrt(xr: Float64, xi: Float64, mut sr: Float64, mut si: Float64):
    var m = sqrt(sqrt(xr * xr + xi * xi))
    var ang = atan2(xi, xr) / 2
    sr = m * cos(ang)
    si = m * sin(ang)


def _scale2(a: M2, cr: Float64, ci: Float64) -> M2:
    return M2(a.re * cr - a.im * ci, a.re * ci + a.im * cr)


def decompose_product(k: M4, mut hi: M2, mut lo: M2) -> Bool:
    """K = e^{i g} (hi x lo); returns False when K is not a product."""
    var r = M2(SIMD[DType.float64, 4](k.re[0], k.re[1], k.re[4], k.re[5]),
               SIMD[DType.float64, 4](k.im[0], k.im[1], k.im[4], k.im[5]))
    var dr = 0.0
    var di = 0.0
    _det2(r, dr, di)
    if dr * dr + di * di < 0.01:
        r = M2(SIMD[DType.float64, 4](k.re[8], k.re[9], k.re[12], k.re[13]),
               SIMD[DType.float64, 4](k.im[8], k.im[9], k.im[12], k.im[13]))
        _det2(r, dr, di)
    if dr * dr + di * di < 1e-12:
        return False
    var sr = 0.0
    var si = 0.0
    _csqrt(dr, di, sr, si)
    var den = sr * sr + si * si
    lo = _scale2(r, sr / den, -si / den)
    var t = mm4(k, m4_kron(M2.identity(), lo.dagger()))
    hi = M2(SIMD[DType.float64, 4](t.re[0], t.re[2], t.re[8], t.re[10]),
            SIMD[DType.float64, 4](t.im[0], t.im[2], t.im[8], t.im[10]))
    var ph = 0.0
    return m4_phase_to(k, m4_kron(hi, lo), ph, 1e-7)


struct Weyl(Copyable):
    var a: Float64
    var b: Float64
    var c: Float64
    var k1hi: M2
    var k1lo: M2
    var k2hi: M2
    var k2lo: M2

    def __init__(out self):
        self.a = 0.0
        self.b = 0.0
        self.c = 0.0
        self.k1hi = M2.identity()
        self.k1lo = M2.identity()
        self.k2hi = M2.identity()
        self.k2lo = M2.identity()


def weyl_decompose(u_in: M4, mut w: Weyl) -> Bool:
    """U = e^{ig} (k1hi x k1lo) exp(i(a XX + b YY + c ZZ)) (k2hi x k2lo)."""
    var dr = 0.0
    var di = 0.0
    m4_det(u_in, dr, di)
    if dr * dr + di * di < 1e-20:
        return False
    var g = atan2(di, dr) / 4
    var u = m4_scale(u_in, cos(-g), sin(-g))
    var bm = magic_basis()
    var bd = m4_dag(bm)
    var up = mm4(mm4(bd, u), bm)
    var mm = mm4(m4_transpose(up), up)
    var coefs: List[Float64] = [1.0, 0.0, 0.0, 1.0, 0.6, 0.8, 0.35, -0.94, 0.8, -0.6, 0.17, 0.98, -0.53, 0.85, 0.93, 0.37]
    var p = SIMD[DType.float64, 16](0)
    var ok = False
    for t in range(len(coefs) // 2):
        var s = mm.re * coefs[2 * t] + mm.im * coefs[2 * t + 1]
        jacobi_eigh(s, p)
        # check P^T M P diagonal
        var pm = M4()
        pm.re = p
        var d = mm4(mm4(m4_transpose(pm), mm), pm)
        var off = 0.0
        for r in range(4):
            for c in range(4):
                if r != c:
                    off += d.re[r * 4 + c] * d.re[r * 4 + c] + d.im[r * 4 + c] * d.im[r * 4 + c]
        if off < 1e-20:
            ok = True
            break
    if not ok:
        return False
    var pm = M4()
    pm.re = p
    # make P a rotation
    var pdr = 0.0
    var pdi = 0.0
    m4_det(pm, pdr, pdi)
    if pdr < 0:
        for r in range(4):
            pm.re[r * 4] = -pm.re[r * 4]
    var dmat = mm4(mm4(m4_transpose(pm), mm), pm)
    var th = SIMD[DType.float64, 4](0)
    for k in range(4):
        th[k] = atan2(dmat.im[k * 5], dmat.re[k * 5]) / 2
    var o1 = mm4(up, pm)
    for c in range(4):
        var cr = cos(-th[c])
        var ci = sin(-th[c])
        for r in range(4):
            var xr = o1.re[r * 4 + c]
            var xi = o1.im[r * 4 + c]
            o1.re[r * 4 + c] = xr * cr - xi * ci
            o1.im[r * 4 + c] = xr * ci + xi * cr
    var odr = 0.0
    var odi = 0.0
    m4_det(o1, odr, odi)
    if odr < 0:
        th[0] += pi
        for r in range(4):
            o1.re[r * 4] = -o1.re[r * 4]
            o1.im[r * 4] = -o1.im[r * 4]
    for k in range(16):
        if abs(o1.im[k]) > 1e-7:
            return False
    var k1 = mm4(mm4(bm, o1), bd)
    var k2 = mm4(mm4(bm, m4_transpose(pm)), bd)
    if not decompose_product(k1, w.k1hi, w.k1lo):
        return False
    if not decompose_product(k2, w.k2hi, w.k2lo):
        return False
    w.a = (th[0] + th[1] - th[2] - th[3]) / 4
    w.b = (-th[0] + th[1] - th[2] + th[3]) / 4
    w.c = (th[0] - th[1] - th[2] + th[3]) / 4
    return True


# ----------------------------------------------------------------------------
# templates
# ----------------------------------------------------------------------------
def clifford(word: String) -> M2:
    """Product of H / S applied in the order written (time order)."""
    var m = M2.identity()
    var h = matrix_1q(Op.H)
    var s = matrix_1q(Op.S)
    for ch in word.as_bytes():
        if Int(ch) == ord("H"):
            m = h @ m
        elif Int(ch) == ord("S"):
            m = s @ m
    return m


def _reduce(x: Float64, mut r: Float64) -> Int:
    """x = r + m pi/2 with r in (-pi/4, pi/4]; returns m."""
    var m = Int(round(x / HALF_PI))
    r = x - Float64(m) * HALF_PI
    if r <= -QUARTER_PI + 1e-12:
        r += HALF_PI
        m -= 1
    if r > QUARTER_PI + 1e-12:
        r -= HALF_PI
        m += 1
    return m


def _emit_1q(m: M2, q: Int, mut out: List[Instruction]):
    """Raw U gate (fuse1q / basis later pick the best form); identity skipped."""
    if abs(m.re[1]) + abs(m.im[1]) + abs(m.re[2]) + abs(m.im[2]) < 1e-13 and abs(m.re[0] - m.re[3]) + abs(m.im[0] - m.im[3]) < 1e-13:
        return
    var z = zyz_decompose(m)
    out.append(Instruction.make(Op.U, q, p0=z.theta, p1=z.phi, p2=z.lam))


def gate_m4(g: Instruction, lo: Int, hi: Int) -> M4:
    """Matrix of a bound 1q/2q gate inside the (lo, hi) pair frame."""
    var a = op_arity(g.kind)
    if a == 1:
        if g.q0 == hi:
            return m4_kron(inst_matrix_1q(g), M2.identity())
        return m4_kron(M2.identity(), inst_matrix_1q(g))
    var m = matrix_2q(g.kind, g.p0.value())
    if g.q0 == lo:
        return m^
    # operands (hi, lo): conjugate by SWAP
    var o = M4()
    for r in range(4):
        for c in range(4):
            var rs = ((r & 1) << 1) | (r >> 1)
            var cs = ((c & 1) << 1) | (c >> 1)
            o.re[r * 4 + c] = m.re[rs * 4 + cs]
            o.im[r * 4 + c] = m.im[rs * 4 + cs]
    return o^


def _iswap_core(kind: Int, z: Int, x: Float64, y: Float64, lo: Int, hi: Int,
                mut core: List[Instruction], mut a0: String, mut b0: String,
                mut r1: String, mut l1: String):
    """iSWAP templates: kind 1 = Ud(pi/4,pi/4,0)-class (1 iSWAP), kind 2 =
    Ud with a zero coordinate at z (2 iSWAPs, RX(2x) / RX(2y) in between).
    Clifford words (time order) map the core onto exp(i(aXX+bYY+cZZ))."""
    core.append(Instruction.make(Op.ISWAP, lo, hi))
    if kind == 1:
        if z == 0:
            a0 = "H"; b0 = "H"; r1 = "H"; l1 = "H"
        elif z == 1:
            a0 = "SH"; b0 = "SH"; r1 = "HSSS"; l1 = "HSSS"
        else:
            a0 = ""; b0 = ""; r1 = ""; l1 = ""
        return
    if abs(x) > 1e-12:
        core.append(Instruction.make(Op.RX, lo, p0=2 * x))
    if abs(y) > 1e-12:
        core.append(Instruction.make(Op.RX, hi, p0=2 * y))
    core.append(Instruction.make(Op.ISWAP, lo, hi))
    if z == 0:
        a0 = "HSH"; b0 = "HSSHSS"; r1 = "HSHSS"; l1 = "HSSH"
    elif z == 1:
        a0 = "HS"; b0 = "S"; r1 = "SH"; l1 = "S"
    else:
        a0 = "H"; b0 = "SHSS"; r1 = "SSH"; l1 = "HSSS"


def _iswap_synthesize(u: M4, lo: Int, hi: Int, ra: Float64, rb: Float64, rc: Float64,
                      pre_lo: M2, pre_hi: M2, post_lo: M2, post_hi: M2,
                      mut out: List[Instruction], mut phase: Float64) -> Int:
    """Native iSWAP synthesis: 0, 1, 2 iSWAPs for c = 0 classes, 4 otherwise
    (Ud(a,b,c) = Ud(a,b,0) . Ud(0,0,c), both factors commute)."""
    var tol = 1e-9
    var co = SIMD[DType.float64, 4](ra, rb, rc, 0)
    var zeros = 0
    var zpos = -1
    for k in range(3):
        if abs(co[k]) < tol:
            zeros += 1
            if zpos < 0:
                zpos = k
    # segment list (time order): kind, z, x, y
    var sk = List[Int]()
    var sz = List[Int]()
    var sx = List[Float64]()
    var sy = List[Float64]()
    if zeros == 3:
        pass
    elif zeros == 1 and abs(co[(zpos + 1) % 3] - QUARTER_PI) < tol and abs(co[(zpos + 2) % 3] - QUARTER_PI) < tol:
        sk.append(1); sz.append(zpos); sx.append(0.0); sy.append(0.0)
    elif zeros >= 1:
        var x = co[1] if zpos == 0 else co[0]
        var y = co[1] if zpos == 2 else co[2]
        sk.append(2); sz.append(zpos); sx.append(x); sy.append(y)
    else:
        sk.append(2); sz.append(0); sx.append(0.0); sy.append(rc)  # Ud(0,0,c)
        sk.append(2); sz.append(2); sx.append(ra); sy.append(rb)  # Ud(a,b,0)
    var tmp = List[Instruction]()
    var cur_lo = pre_lo
    var cur_hi = pre_hi
    var n2 = 0
    for i in range(len(sk)):
        var core = List[Instruction]()
        var a0 = String("")
        var b0 = String("")
        var r1 = String("")
        var l1 = String("")
        _iswap_core(sk[i], sz[i], sx[i], sy[i], lo, hi, core, a0, b0, r1, l1)
        cur_lo = clifford(a0) @ cur_lo
        cur_hi = clifford(b0) @ cur_hi
        _emit_1q(cur_lo, lo, tmp)
        _emit_1q(cur_hi, hi, tmp)
        for g in core:
            tmp.append(g)
            if g.kind == Op.ISWAP:
                n2 += 1
        cur_lo = clifford(r1)
        cur_hi = clifford(l1)
    _emit_1q(post_lo @ cur_lo, lo, tmp)
    _emit_1q(post_hi @ cur_hi, hi, tmp)
    var m = M4.identity()
    for g in tmp:
        m = mm4(gate_m4(g, lo, hi), m)
    var ph = 0.0
    if not m4_phase_to(u, m, ph, 1e-7):
        return -1
    for g in tmp:
        out.append(g)
    phase += ph
    return n2


def kak_synthesize(u: M4, lo: Int, hi: Int, mut out: List[Instruction], mut phase: Float64,
                   native_rotations: Bool = False, iswap: Bool = False) -> Int:
    """Minimal-CX circuit for u on (lo, hi); returns #2q gates or -1 (nothing appended).

    native_rotations: emit the canonical core as rxx/ryy/rzz rotations (one per
    non-zero Weyl coordinate) for continuous-angle entanglers (ions, ZZ).
    """
    var w = Weyl()
    if not weyl_decompose(u, w):
        return -1
    var ra = 0.0
    var rb = 0.0
    var rc = 0.0
    var ma = _reduce(w.a, ra)
    var mb = _reduce(w.b, rb)
    var mc = _reduce(w.c, rc)
    # Pauli remainder (commutes with the canonical gate): applied after K2
    var pauli = M2.identity()
    if ma % 2 != 0:
        pauli = matrix_1q(Op.X) @ pauli
    if mb % 2 != 0:
        pauli = matrix_1q(Op.Y) @ pauli
    if mc % 2 != 0:
        pauli = matrix_1q(Op.Z) @ pauli
    var pre_hi = pauli @ w.k2hi
    var pre_lo = pauli @ w.k2lo
    var post_hi = w.k1hi
    var post_lo = w.k1lo
    if iswap:
        return _iswap_synthesize(u, lo, hi, ra, rb, rc, pre_lo, pre_hi, post_lo, post_hi, out, phase)
    var tol = 1e-9
    var co = SIMD[DType.float64, 4](ra, rb, rc, 0)
    var nz = 0
    var zpos = -1
    var nzpos = -1
    for k in range(3):
        if abs(co[k]) < tol:
            nz += 1
            zpos = k
        else:
            nzpos = k
    var tmp = List[Instruction]()
    var core = List[Instruction]()
    var ncx: Int
    # (q0-pre, q1-pre, q0-post, q1-post) Clifford words of each template
    var a0 = String("")
    var b0 = String("")
    var r1 = String("")
    var l1 = String("")
    if native_rotations:
        # exp(i(a XX + b YY + c ZZ)) = RXX(-2a) RYY(-2b) RZZ(-2c), commuting factors
        ncx = 0
        if abs(ra) >= tol:
            core.append(Instruction.make(Op.RXX, lo, hi, -1, -2 * ra))
            ncx += 1
        if abs(rb) >= tol:
            core.append(Instruction.make(Op.RYY, lo, hi, -1, -2 * rb))
            ncx += 1
        if abs(rc) >= tol:
            core.append(Instruction.make(Op.RZZ, lo, hi, -1, -2 * rc))
            ncx += 1
    elif nz == 3:
        ncx = 0
    elif nz == 2 and abs(co[nzpos] - QUARTER_PI) < tol:
        ncx = 1
        core.append(Instruction.make(Op.CX, lo, hi))
        if nzpos == 0:
            a0 = "H"; b0 = ""; r1 = "HSHS"; l1 = "SHS"
        elif nzpos == 1:
            a0 = "SH"; b0 = "S"; r1 = "HSH"; l1 = "SH"
        else:
            a0 = ""; b0 = "H"; r1 = "SSS"; l1 = "HSSS"
    elif nz >= 1:
        ncx = 2
        var x: Int
        var y: Int
        if zpos == 0:
            x = 1; y = 2
            a0 = "S"; b0 = "HSSHS"; r1 = "SSS"; l1 = "HSSHS"
        elif zpos == 1:
            x = 0; y = 2
            a0 = ""; b0 = "HSSHSS"; r1 = ""; l1 = "HSSHSS"
        else:
            x = 0; y = 1
            a0 = "HSH"; b0 = "SHSSS"; r1 = "SHS"; l1 = "SHSSS"
        core.append(Instruction.make(Op.CX, lo, hi))
        core.append(Instruction.make(Op.RX, lo, p0=2 * co[x]))
        core.append(Instruction.make(Op.RZ, hi, p0=2 * co[y]))
        core.append(Instruction.make(Op.CX, lo, hi))
    else:
        ncx = 3
        a0 = "HS"; b0 = "SSHSS"; r1 = "SSHSS"; l1 = "HSHS"
        core.append(Instruction.make(Op.CX, hi, lo))
        core.append(Instruction.make(Op.RZ, lo, p0=2 * ra - HALF_PI))
        core.append(Instruction.make(Op.RY, hi, p0=2 * rb - HALF_PI))
        core.append(Instruction.make(Op.CX, lo, hi))
        core.append(Instruction.make(Op.RY, hi, p0=2 * rc - HALF_PI))
        core.append(Instruction.make(Op.CX, hi, lo))
    pre_lo = clifford(a0) @ pre_lo
    pre_hi = clifford(b0) @ pre_hi
    post_lo = post_lo @ clifford(r1)
    post_hi = post_hi @ clifford(l1)
    if ncx == 0:
        _emit_1q(post_lo @ pre_lo, lo, tmp)
        _emit_1q(post_hi @ pre_hi, hi, tmp)
    else:
        _emit_1q(pre_lo, lo, tmp)
        _emit_1q(pre_hi, hi, tmp)
        for g in core:
            tmp.append(g)
        _emit_1q(post_lo, lo, tmp)
        _emit_1q(post_hi, hi, tmp)
    # verify and measure the global phase
    var m = M4.identity()
    for g in tmp:
        m = mm4(gate_m4(g, lo, hi), m)
    var ph = 0.0
    if not m4_phase_to(u, m, ph, 1e-7):
        return -1
    for g in tmp:
        out.append(g)
    phase += ph
    return ncx


# ----------------------------------------------------------------------------
# block consolidation pass
# ----------------------------------------------------------------------------
def _cx_units(k: Int) -> Int:
    if k == Op.CX or k == Op.CZ or k == Op.CY or k == Op.CH or k == Op.ECR:
        return 1
    if k == Op.SWAP:
        return 3
    return 2


def has_continuous_entangler(gs: GateSet) -> Bool:
    return not gs.any and (gs.has(Op.RZZ) or gs.has(Op.RXX) or gs.has(Op.RYY))


def cx_unit_cost(gs: GateSet) -> Int:
    """Native 2q gates needed per CX in this basis (iSWAP family needs 2)."""
    if gs.any:
        return 1
    if gs.has(Op.CX) or gs.has(Op.CZ) or gs.has(Op.ECR) or gs.has(Op.CP):
        return 1
    if gs.has(Op.ISWAP) or gs.has(Op.ISWAPDG):
        return 2
    return 1


def cx_cost(k: Int, gs: GateSet) -> Int:
    """Two-qubit cost of a gate in native-entangler units of the target basis."""
    if not gs.any and gs.has(k):
        return 1
    if has_continuous_entangler(gs) and not gs.has(Op.CX):
        if k == Op.CP or k == Op.CRZ or k == Op.CZ or k == Op.RZZ or k == Op.RXX or k == Op.RYY or k == Op.CX:
            return 1
    return _cx_units(k) * cx_unit_cost(gs)


def min_cx_lower_bound(u: M4) -> Int:
    """Conservative lower bound (0, 2 or 3) of the CX count of a 2q unitary.

    Shende-Bullock-Markov invariant (PRA 69, 062321): with V = u / det(u)^{1/4}
    and gamma(V) = V (Y x Y) V^T (Y x Y), V needs <= 2 CX only if tr gamma(V)
    is real, <= 1 CX only if tr gamma(V) = 0 and 0 CX only if |tr| = 4. Only
    the trace is formed (16 complex products), so a block that cannot beat
    its current cost is rejected without a Weyl decomposition. Tolerances are
    loose (1e-7): a borderline block is passed on to the exact synthesis."""
    var dr = 0.0
    var di = 0.0
    m4_det(u, dr, di)
    var dn = sqrt(dr * dr + di * di)
    if dn < 1e-12:
        return 0
    # Y x Y = antidiag(-1, 1, 1, -1): tr = sum_ij U[i][3-j] s(3-j) U[3-i][j] s(3-i)
    var tr = 0.0
    var ti = 0.0
    comptime for i in range(4):
        comptime for j in range(4):
            comptime sgn = (-1.0 if (3 - j) == 0 or (3 - j) == 3 else 1.0) * (-1.0 if (3 - i) == 0 or (3 - i) == 3 else 1.0)
            var ar = u.re[i * 4 + (3 - j)]
            var ai = u.im[i * 4 + (3 - j)]
            var br = u.re[(3 - i) * 4 + j]
            var bi = u.im[(3 - i) * 4 + j]
            tr += sgn * (ar * br - ai * bi)
            ti += sgn * (ar * bi + ai * br)
    # divide by sqrt(det) (either branch: only realness / modulus are tested)
    var half = atan2(di, dr) / 2
    var cr = cos(-half) / sqrt(dn)
    var ci = sin(-half) / sqrt(dn)
    var vr = tr * cr - ti * ci
    var vi = tr * ci + ti * cr
    var tol = 1e-7
    if abs(vi) > tol:
        return 3
    var mod = sqrt(vr * vr + vi * vi)
    if mod > tol and abs(mod - 4.0) > tol:
        return 2
    return 0


comptime _SLOT = 24  # max instructions a KAK template emits per block


def consolidate_2q(mut c: Circuit, gs: GateSet, mut blocks_out: Int, threads: Bool = True) -> Int:
    """Resynthesise 2q blocks with KAK when it lowers the 2q cost; returns cost saved.

    Blocks are collected in one sweep; each candidate block's unitary is
    first screened with the Shende-Bullock-Markov trace invariant (blocks
    that cannot beat their cost are dropped without a Weyl decomposition) and
    the remaining syntheses run on all cores (`threads`; pass False when
    called from inside another parallel region). Results are independent of
    the thread count.

    `gs` is the *target* basis: it sets the cost model (native entanglers count 1,
    iSWAP needs 2 per CX) and the template family (CX templates, or one
    rxx/ryy/rzz per non-zero Weyl coordinate for continuous entanglers). The
    emitted gates are basis-agnostic (u, cx, rx/ry/rz, rxx/ryy/rzz).
    """
    var native = has_continuous_entangler(gs) and not (gs.has(Op.CX) or gs.has(Op.CZ) or gs.has(Op.ECR))
    var unit = cx_unit_cost(gs)
    var iswap_mode = not gs.any and (gs.has(Op.ISWAP) or gs.has(Op.ISWAPDG)) and not (gs.has(Op.CX) or gs.has(Op.CZ) or gs.has(Op.ECR) or native)
    var n = len(c.ops)
    var nq = c.num_qubits
    var cur = List[Int](length=nq, fill=-1)
    var blk_of = List[Int](length=n, fill=-1)
    var b_lo = List[Int]()
    var b_hi = List[Int]()
    var b_last = List[Int]()
    var b_n2 = List[Int]()
    var b_cost = List[Int]()
    var b_mixed = List[Bool]()  # contains a 2q gate that is not a 1-CX gate
    var b_exact = List[Bool]()  # first 2q gate is a CX-family gate or a SWAP (exact CX cost)
    for i in range(n):
        var g = c.ops[i]
        if is_all_wire_op(g):
            for q in range(nq):
                cur[q] = -1
            continue
        var a = op_arity(g.kind)
        if a == 0:
            continue
        if not is_plain_gate(g) or not is_bound(g) or a == 3:
            for s in range(a):
                var q = g.qubit(s)
                if q >= 0 and q < nq and cur[q] >= 0:
                    var b = cur[q]
                    cur[b_lo[b]] = -1
                    cur[b_hi[b]] = -1
            continue
        if a == 1:
            var b = cur[g.q0]
            if b >= 0:
                blk_of[i] = b
                b_last[b] = i
            continue
        var x = g.q0
        var y = g.q1
        if cur[x] >= 0 and cur[x] == cur[y]:
            var b = cur[x]
            blk_of[i] = b
            b_last[b] = i
            b_n2[b] += 1
            b_cost[b] += cx_cost(g.kind, gs)
            if _cx_units(g.kind) != 1:
                b_mixed[b] = True
            continue
        for q in [x, y]:
            if cur[q] >= 0:
                var b = cur[q]
                cur[b_lo[b]] = -1
                cur[b_hi[b]] = -1
        var nb = len(b_lo)
        b_lo.append(min(x, y))
        b_hi.append(max(x, y))
        b_last.append(i)
        b_n2.append(1)
        b_cost.append(cx_cost(g.kind, gs))
        b_mixed.append(_cx_units(g.kind) != 1)
        b_exact.append(_cx_units(g.kind) != 2)
        cur[x] = nb
        cur[y] = nb
        blk_of[i] = nb
    var nb = len(b_lo)
    # small circuits: every block is a candidate (maximal consolidation); big
    # ones skip pure-CX blocks of cost <= 3, which almost never shrink
    var small = n < 40000
    var cx_family = not native and not iswap_mode
    var cand_list = List[Int]()
    for b in range(nb):
        # pure CX-family blocks of cost <= 3 are almost never reducible (trivial
        # cases are left to cancel/commute): skip them, KAK is the costly part
        if b_cost[b] >= 4 or (b_cost[b] >= 2 and (b_mixed[b] or native or iswap_mode or small)):
            # a lone SWAP / CX-family gate (1q gates around it) is already
            # optimal for CX-family targets: its Weyl class fixes the count
            if cx_family and b_n2[b] == 1 and b_exact[b]:
                continue
            cand_list.append(b)
    var ncand = len(cand_list)
    if ncand == 0:
        return 0
    # ops of every candidate block (CSR, program order)
    var slot_of = List[Int](length=nb, fill=-1)
    for j in range(ncand):
        slot_of[cand_list[j]] = j
    var ostart = List[Int](length=ncand + 1, fill=0)
    for i in range(n):
        var b = blk_of[i]
        if b >= 0 and slot_of[b] >= 0:
            ostart[slot_of[b] + 1] += 1
    for j in range(ncand):
        ostart[j + 1] += ostart[j]
    var oidx = List[Int](length=ostart[ncand], fill=0)
    var ofill = ostart.copy()
    for i in range(n):
        var b = blk_of[i]
        if b >= 0 and slot_of[b] >= 0:
            var j = slot_of[b]
            oidx[ofill[j]] = i
            ofill[j] += 1
    # block unitaries + KAK synthesis, independent per block -> parallel
    var res_k = List[Int](length=ncand, fill=-1)
    var res_len = List[Int](length=ncand, fill=0)
    var res_ph = List[Float64](length=ncand, fill=0.0)
    var res_ops = List[Instruction](length=ncand * _SLOT, fill=Instruction.make(Op.I, 0))
    var rkp = res_k.unsafe_ptr().as_unsafe_any_origin()
    var rlp = res_len.unsafe_ptr().as_unsafe_any_origin()
    var rpp = res_ph.unsafe_ptr().as_unsafe_any_origin()
    var rop = res_ops.unsafe_ptr().as_unsafe_any_origin()
    var nchunks = 1 if (ncand < 48 or not threads) else min(ncand // 16, 256)

    def work(t: Int) {imm c, imm cand_list, imm ostart, imm oidx, imm b_lo, imm b_hi, imm b_cost, imm unit, imm native, imm iswap_mode, imm cx_family, imm ncand, imm nchunks, imm rkp, imm rlp, imm rpp, imm rop}:
        var lo_j = t * ncand // nchunks
        var hi_j = (t + 1) * ncand // nchunks
        var tmp = List[Instruction](capacity=_SLOT)
        for j in range(lo_j, hi_j):
            var b = cand_list[j]
            var m = M4.identity()
            for e in range(ostart[j], ostart[j + 1]):
                m = mm4(gate_m4(c.ops[oidx[e]], b_lo[b], b_hi[b]), m)
            if cx_family and min_cx_lower_bound(m) * unit >= b_cost[b]:
                continue  # cannot beat the current cost
            tmp.clear()
            var ph = 0.0
            var k = kak_synthesize(m, b_lo[b], b_hi[b], tmp, ph, native, iswap_mode)
            if k < 0 or len(tmp) > _SLOT:
                continue
            rkp[unsafe_offset=j] = k
            rlp[unsafe_offset=j] = len(tmp)
            rpp[unsafe_offset=j] = ph
            for e in range(len(tmp)):
                rop[unsafe_offset=j * _SLOT + e] = tmp[e]

    if nchunks == 1:
        work(0)
    else:
        parallelize(work, nchunks)
    _ = len(res_k)
    _ = len(res_len)
    _ = len(res_ph)
    _ = len(res_ops)
    var rep_start = List[Int](length=nb, fill=-1)
    var rep_len = List[Int](length=nb, fill=0)
    var rep = List[Instruction]()
    var saved = 0
    var phase = 0.0
    var replaced = 0
    for j in range(ncand):
        var b = cand_list[j]
        var k = res_k[j]
        if k >= 0 and not native and not iswap_mode:
            k *= unit
        if k < 0 or k >= b_cost[b]:
            continue
        rep_start[b] = len(rep)
        rep_len[b] = res_len[j]
        for e in range(res_len[j]):
            rep.append(res_ops[j * _SLOT + e])
        phase += res_ph[j]
        saved += b_cost[b] - k
        replaced += 1
    if replaced == 0:
        return 0
    var out = List[Instruction](capacity=n)
    for i in range(n):
        var b = blk_of[i]
        if b >= 0 and rep_start[b] >= 0:
            if i == b_last[b]:
                for k in range(rep_len[b]):
                    out.append(rep[rep_start[b] + k])
            continue
        out.append(c.ops[i])
    c.ops = out^
    c.global_phase += phase
    blocks_out = replaced
    return saved
