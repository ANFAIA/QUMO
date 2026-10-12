"""Every state-vector kernel (SIMD high-qubit, SIMD low-qubit lane-shuffle and
scalar paths) against a naive dense reference, on every qubit (pair)."""

from std.math import cos, sin
from qumo.ir.gates import Op, M2, M4, matrix_1q, matrix_2q
from qumo.sim.statevector import StateVector
from qumo.sim.rng import Rng
from std.testing import assert_true


def random_state(n: Int, mut rng: Rng) -> StateVector:
    var sv = StateVector(n)
    for i in range(sv.size()):
        sv.re[i] = rng.uniform() - 0.5
        sv.im[i] = rng.uniform() - 0.5
    return sv^


def ref_m4(mut sv: StateVector, m: M4, q0: Int, q1: Int):
    """Naive dense 4x4 on (q0, q1); matrix index = b(q0) + 2 b(q1)."""
    var b0 = 1 << q0
    var b1 = 1 << q1
    for base in range(sv.size()):
        if (base & b0) != 0 or (base & b1) != 0:
            continue
        var idx: List[Int] = [base, base | b0, base | b1, base | b0 | b1]
        var ar = List[Float64]()
        var ai = List[Float64]()
        for k in range(4):
            ar.append(sv.re[idx[k]])
            ai.append(sv.im[idx[k]])
        for r in range(4):
            var sr = 0.0
            var si = 0.0
            for c in range(4):
                sr += m.re[r * 4 + c] * ar[c] - m.im[r * 4 + c] * ai[c]
                si += m.re[r * 4 + c] * ai[c] + m.im[r * 4 + c] * ar[c]
            sv.re[idx[r]] = sr
            sv.im[idx[r]] = si


def ref_m2(mut sv: StateVector, m: M2, t: Int):
    var s = 1 << t
    for i in range(sv.size()):
        if (i & s) != 0:
            continue
        var ar = sv.re[i]
        var ai = sv.im[i]
        var br = sv.re[i | s]
        var bi = sv.im[i | s]
        sv.re[i] = m.re[0] * ar - m.im[0] * ai + m.re[1] * br - m.im[1] * bi
        sv.im[i] = m.re[0] * ai + m.im[0] * ar + m.re[1] * bi + m.im[1] * br
        sv.re[i | s] = m.re[2] * ar - m.im[2] * ai + m.re[3] * br - m.im[3] * bi
        sv.im[i | s] = m.re[2] * ai + m.im[2] * ar + m.re[3] * bi + m.im[3] * br


def max_diff(a: StateVector, b: StateVector) -> Float64:
    var d = 0.0
    for i in range(a.size()):
        d = max(d, abs(a.re[i] - b.re[i]) + abs(a.im[i] - b.im[i]))
    return d


def check(a: StateVector, b: StateVector, what: String) raises:
    var d = max_diff(a, b)
    if d > 1e-12:
        raise Error("kernel mismatch: " + what + " diff=" + String(d))


def controlled(m: M2) -> M4:
    """Control = q0 (bit 0 of the matrix index), target = q1."""
    var out = M4()
    out.set(0, 0, 1)
    out.set(2, 2, 1)
    out.set(1, 1, m.re[0], m.im[0])
    out.set(1, 3, m.re[1], m.im[1])
    out.set(3, 1, m.re[2], m.im[2])
    out.set(3, 3, m.re[3], m.im[3])
    return out^


def main() raises:
    var rng = Rng(11)
    for n in [5, 7, 10, 14]:
        var psi = random_state(n, rng)
        var u = matrix_1q(Op.U, 0.7, -1.2, 0.4)
        var d = matrix_1q(Op.P, 0.9)
        d.re[0] = cos(0.3)
        d.im[0] = -sin(0.3)
        for t in range(n):
            var a = psi.copy()
            var b = psi.copy()
            a.apply_1q(u, t)
            ref_m2(b, u, t)
            check(a, b, "1q t=" + String(t))
            a = psi.copy()
            b = psi.copy()
            a.apply_1q(d, t)
            ref_m2(b, d, t)
            check(a, b, "diag t=" + String(t))
            a = psi.copy()
            b = psi.copy()
            a.apply_x(t)
            ref_m2(b, matrix_1q(Op.X), t)
            check(a, b, "x t=" + String(t))
            a = psi.copy()
            b = psi.copy()
            a.apply_scale(0.6, -0.8)
            var sc = M2.identity()
            sc.re[0] = 0.6
            sc.im[0] = -0.8
            sc.re[3] = 0.6
            sc.im[3] = -0.8
            ref_m2(b, sc, t)
            check(a, b, "scale")
        var r4 = M4()
        for k in range(16):
            r4.re[k] = rng.uniform() - 0.5
            r4.im[k] = rng.uniform() - 0.5
        var ph = M4.identity()
        ph.set(3, 3, cos(1.1), sin(1.1))
        var rzz = M4()
        rzz.set(0, 0, cos(0.35), -sin(0.35))
        rzz.set(1, 1, cos(0.35), sin(0.35))
        rzz.set(2, 2, cos(0.35), sin(0.35))
        rzz.set(3, 3, cos(0.35), -sin(0.35))
        for q0 in range(n):
            for q1 in range(n):
                if q0 == q1:
                    continue
                var tag = " (" + String(q0) + "," + String(q1) + ") n=" + String(n)
                var a = psi.copy()
                var b = psi.copy()
                a.apply_cx(q0, q1)
                ref_m4(b, matrix_2q(Op.CX), q0, q1)
                check(a, b, "cx" + tag)
                a = psi.copy()
                b = psi.copy()
                a.apply_cphase(cos(1.1), sin(1.1), q0, q1)
                ref_m4(b, ph, q0, q1)
                check(a, b, "cphase" + tag)
                a = psi.copy()
                b = psi.copy()
                a.apply_controlled_1q(u, q0, q1)
                ref_m4(b, controlled(u), q0, q1)
                check(a, b, "c1" + tag)
                a = psi.copy()
                b = psi.copy()
                a.apply_controlled_1q(d, q0, q1)
                ref_m4(b, controlled(d), q0, q1)
                check(a, b, "c1diag" + tag)
                a = psi.copy()
                b = psi.copy()
                a.apply_rzz(0.7, q0, q1)
                ref_m4(b, rzz, q0, q1)
                check(a, b, "rzz" + tag)
                a = psi.copy()
                b = psi.copy()
                a.apply_swap(q0, q1)
                ref_m4(b, matrix_2q(Op.SWAP), q0, q1)
                check(a, b, "swap" + tag)
                a = psi.copy()
                b = psi.copy()
                a.apply_2q(r4, q0, q1)
                ref_m4(b, r4, q0, q1)
                check(a, b, "2q" + tag)
        # multi-SWAP layer == product of the individual SWAPs, random pairings
        for trial in range(12):
            var perm = List[Int]()
            for q in range(n):
                perm.append(q)
            for i in range(n - 1, 0, -1):
                var j = rng.below(i + 1)
                var tmp = perm[i]
                perm[i] = perm[j]
                perm[j] = tmp
            var k = 1 + rng.below(n // 2)
            var qa = List[Int]()
            var qb = List[Int]()
            for i in range(k):
                qa.append(perm[2 * i])
                qb.append(perm[2 * i + 1])
            var a = psi.copy()
            var b = psi.copy()
            a.apply_swap_layer(qa, qb)
            for i in range(k):
                ref_m4(b, matrix_2q(Op.SWAP), qa[i], qb[i])
            check(a, b, "swap_layer trial " + String(trial) + " n=" + String(n))
    # large state: parallel paths of the swap layer (low and high bits)
    var big = random_state(20, rng)
    var qa: List[Int] = [0, 3, 17]
    var qb: List[Int] = [19, 12, 5]
    var a = big.copy()
    var b = big.copy()
    a.apply_swap_layer(qa, qb)
    for i in range(3):
        b.apply_swap(qa[i], qb[i])
    check(a, b, "swap_layer n=20")
    print("test_sim_kernels OK")
