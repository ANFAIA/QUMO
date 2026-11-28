"""Tests for qumo.core.svd (complex one-sided Jacobi SVD).

References are built here, independently of the code under test: matrices with a known
spectrum are U0 diag(s0) V0^H, with U0 and V0 orthonormal columns from Gram-Schmidt on
seeded Gaussian matrices; the checks are reconstruction, orthonormality of U and V, the
order and sign of s, and agreement with the known singular values.
"""

from std.math import sqrt
from std.time import perf_counter_ns
from std.testing import assert_true, assert_equal
from qumo.core.rng import Rng
from qumo.core.svd import CMatrix, SVDResult, svd, check_svd_size, SVD_MAX_MIN_DIM


def random_matrix(m: Int, n: Int, mut rng: Rng) raises -> CMatrix:
    var a = CMatrix(m, n)
    for r in range(m):
        for c in range(n):
            a.set(r, c, rng.normal(), rng.normal())
    return a^


def orthonormal_columns(m: Int, k: Int, mut rng: Rng) raises -> CMatrix:
    """An m x k matrix with orthonormal columns: modified Gram-Schmidt, twice, on a Gaussian matrix."""
    var a = random_matrix(m, k, rng)
    for c in range(k):
        for _pass in range(2):
            for d in range(c):
                var dr = 0.0
                var di = 0.0
                for r in range(m):  # <a_d, a_c>
                    dr += a.get_re(r, d) * a.get_re(r, c) + a.get_im(r, d) * a.get_im(r, c)
                    di += a.get_re(r, d) * a.get_im(r, c) - a.get_im(r, d) * a.get_re(r, c)
                for r in range(m):
                    var xr = a.get_re(r, c) - (a.get_re(r, d) * dr - a.get_im(r, d) * di)
                    var xi = a.get_im(r, c) - (a.get_re(r, d) * di + a.get_im(r, d) * dr)
                    a.set(r, c, xr, xi)
        var nn = 0.0
        for r in range(m):
            nn += a.get_re(r, c) ** 2 + a.get_im(r, c) ** 2
        var inv = 1.0 / sqrt(nn)
        for r in range(m):
            a.set(r, c, a.get_re(r, c) * inv, a.get_im(r, c) * inv)
    return a^


def compose(u: CMatrix, s: List[Float64], v: CMatrix) raises -> CMatrix:
    """U diag(s) V^H."""
    var m = u.rows
    var n = v.rows
    var out = CMatrix(m, n)
    for r in range(m):
        for c in range(n):
            var sr = 0.0
            var si = 0.0
            for t in range(len(s)):
                # u[r,t] * s[t] * conj(v[c,t])
                var ar = u.get_re(r, t) * s[t]
                var ai = u.get_im(r, t) * s[t]
                var br = v.get_re(c, t)
                var bi = -v.get_im(c, t)
                sr += ar * br - ai * bi
                si += ar * bi + ai * br
            out.set(r, c, sr, si)
    return out^


def fro(a: CMatrix) -> Float64:
    var s = 0.0
    for i in range(len(a.re)):
        s += a.re[i] * a.re[i] + a.im[i] * a.im[i]
    return sqrt(s)


def diff_fro(a: CMatrix, b: CMatrix) -> Float64:
    var s = 0.0
    for i in range(len(a.re)):
        s += (a.re[i] - b.re[i]) ** 2 + (a.im[i] - b.im[i]) ** 2
    return sqrt(s)


def orthonormality_error(u: CMatrix) -> Float64:
    """The largest |(U^H U - I)_{ij}|."""
    var worst = 0.0
    for i in range(u.cols):
        for j in range(u.cols):
            var sr = 0.0
            var si = 0.0
            for r in range(u.rows):
                sr += u.get_re(r, i) * u.get_re(r, j) + u.get_im(r, i) * u.get_im(r, j)
                si += u.get_re(r, i) * u.get_im(r, j) - u.get_im(r, i) * u.get_re(r, j)
            if i == j:
                sr -= 1.0
            worst = max(worst, sqrt(sr * sr + si * si))
    return worst


def check_valid(a: CMatrix, r: SVDResult, tol: Float64, what: String) raises:
    var k = min(a.rows, a.cols)
    assert_equal(r.u.rows, a.rows, what + ": U rows")
    assert_equal(r.u.cols, k, what + ": U cols")
    assert_equal(r.v.rows, a.cols, what + ": V rows")
    assert_equal(r.v.cols, k, what + ": V cols")
    assert_equal(len(r.s), k, what + ": len(s)")
    for t in range(k):
        assert_true(r.s[t] >= 0.0, what + ": s >= 0")
        if t > 0:
            assert_true(r.s[t] <= r.s[t - 1], what + ": s non-increasing")
    var rec = compose(r.u, r.s, r.v)
    var na = fro(a)
    var err = diff_fro(a, rec)
    assert_true(err <= tol * max(na, 1e-300), what + ": reconstruction error " + String(err / max(na, 1e-300)))
    var eu = orthonormality_error(r.u)
    var ev = orthonormality_error(r.v)
    assert_true(eu <= tol, what + ": U not orthonormal, " + String(eu))
    assert_true(ev <= tol, what + ": V not orthonormal, " + String(ev))


def check_known(s0: List[Float64], r: SVDResult, abs_tol: Float64, what: String) raises:
    """Compare with s0, sorted non-increasing."""
    for t in range(len(s0)):
        var d = abs(r.s[t] - s0[t])
        assert_true(d <= abs_tol, what + ": s[" + String(t) + "] = " + String(r.s[t]) + " vs " + String(s0[t]))


def known_spectrum(m: Int, n: Int, s0: List[Float64], mut rng: Rng) raises -> CMatrix:
    var k = len(s0)
    var u0 = orthonormal_columns(m, k, rng)
    var v0 = orthonormal_columns(n, k, rng)
    return compose(u0, s0, v0)


def expect_code(m: Int, n: Int, code: String) raises:
    try:
        check_svd_size(m, n)
    except e:
        assert_true(String(e).startswith(code), String(e))
        return
    raise Error("expected " + code + " for " + String(m) + " x " + String(n))


def main() raises:
    var rng = Rng(20261127)

    # Random shapes: square, tall, wide, vectors, 1 x 1, empty.
    var shapes: List[Tuple[Int, Int]] = [(1, 1), (1, 6), (6, 1), (2, 2), (4, 4), (7, 3), (3, 7),
                                         (16, 16), (40, 25), (25, 40), (64, 64)]
    for sh in shapes:
        var a = random_matrix(sh[0], sh[1], rng)
        var r = svd(a)
        check_valid(a, r, 1e-12, "random " + String(sh[0]) + "x" + String(sh[1]))
    var empty = CMatrix(0, 5)
    var re = svd(empty)
    assert_equal(len(re.s), 0)
    assert_equal(re.v.rows, 5)

    # Real input stays exact-real-compatible: a real diagonal with known values.
    var d = CMatrix(3, 3)
    d.set(0, 0, -2.0)
    d.set(1, 1, 5.0)
    d.set(2, 2, 0.5)
    var rd = svd(d)
    check_valid(d, rd, 1e-14, "diagonal")
    var sd: List[Float64] = [5.0, 2.0, 0.5]
    check_known(sd, rd, 1e-14, "diagonal")

    # Known spectra (tall, wide, square).
    var s1: List[Float64] = [3.0, 2.5, 1.0, 0.25, 0.125]
    for sh in [(9, 5), (5, 9), (5, 5)]:
        var a = known_spectrum(sh[0], sh[1], s1, rng)
        var r = svd(a)
        check_valid(a, r, 1e-12, "known")
        check_known(s1, r, 1e-13, "known " + String(sh[0]) + "x" + String(sh[1]))

    # Rank-deficient: rank 3 in 10 x 8 and 8 x 10; the 5 trailing singular values vanish
    # to rounding, and U, V stay orthonormal.
    var s2: List[Float64] = [4.0, 1.0, 0.5]
    for sh in [(10, 8), (8, 10)]:
        var a = known_spectrum(sh[0], sh[1], s2, rng)
        var r = svd(a)
        check_valid(a, r, 1e-12, "rank 3")
        check_known(s2, r, 1e-13, "rank 3")
        for t in range(3, 8):
            assert_true(r.s[t] <= 1e-13, "rank 3: trailing s = " + String(r.s[t]))
    # Regression: a rank-2 4 x 4 matrix with two zero rows (a two-site block met by the MPS
    # simulator). The noise columns shrank into subnormal numbers, the rotations stopped
    # being unitary and s[1] came out 1e-6 off. Plus many random small rank-deficient
    # matrices, with and without zero rows.
    var s7: List[Float64] = [0.9, 0.35]
    for trial in range(300):
        var m7 = 2 + trial % 7
        var n7 = 2 + (trial // 7) % 7
        var r7 = 1 + trial % min(m7, n7)
        var sk = List[Float64]()
        for t in range(r7):
            sk.append(1.0 / Float64(t + 1))
        var a7 = known_spectrum(m7, n7, sk, rng)
        if trial % 2 == 0 and m7 >= 3:  # zero every other row
            for r in range(1, m7, 2):
                for c in range(n7):
                    a7.set(r, c, 0.0, 0.0)
            var r7b = svd(a7)
            check_valid(a7, r7b, 1e-12, "rank-deficient with zero rows")
        else:
            var r7b = svd(a7)
            check_valid(a7, r7b, 1e-12, "rank-deficient")
            check_known(sk, r7b, 1e-13, "rank-deficient")
    var u8 = orthonormal_columns(2, 2, rng)
    var v8 = orthonormal_columns(4, 2, rng)
    var b8 = compose(u8, s7, v8)
    var a8 = CMatrix(4, 4)
    for c in range(4):
        a8.set(0, c, b8.get_re(0, c), b8.get_im(0, c))
        a8.set(2, c, b8.get_re(1, c), b8.get_im(1, c))
    var r8 = svd(a8)
    check_valid(a8, r8, 1e-13, "two zero rows")
    check_known(s7, r8, 1e-14, "two zero rows")

    # Exactly zero singular values: zero columns, the zero matrix (U completed by Gram-Schmidt).
    var z = CMatrix(6, 4)
    for i in range(6):
        z.set(i, 0, Float64(i + 1), 0.5)
    var rz = svd(z)
    check_valid(z, rz, 1e-13, "one non-zero column")
    assert_equal(rz.s[1], 0.0)
    var zz = CMatrix(5, 3)
    var rzz = svd(zz)
    check_valid(zz, rzz, 1e-14, "zero matrix")
    for t in range(3):
        assert_equal(rzz.s[t], 0.0)

    # Degenerate singular values (the vectors are not unique: reconstruction and
    # orthonormality must still hold).
    var s3: List[Float64] = [1.0, 1.0, 1.0, 0.5, 0.5, 0.1]
    var a3 = known_spectrum(12, 6, s3, rng)
    var r3 = svd(a3)
    check_valid(a3, r3, 1e-12, "degenerate")
    check_known(s3, r3, 1e-13, "degenerate")
    var ru = svd(CMatrix.identity(7))
    check_valid(CMatrix.identity(7), ru, 1e-15, "identity")

    # Graded: 16 orders of magnitude. Absolute accuracy relative to s_max for a general
    # orientation; relative accuracy of every value when the columns are already
    # orthogonal (A = U0 diag(s0): the one-sided method leaves such columns untouched).
    var s4 = List[Float64]()
    for t in range(9):
        s4.append(10.0 ** (-2.0 * Float64(t)))
    var a4 = known_spectrum(11, 9, s4, rng)
    var r4 = svd(a4)
    check_valid(a4, r4, 1e-12, "graded")
    check_known(s4, r4, 1e-14, "graded")
    var u5 = orthonormal_columns(11, 9, rng)
    var a5 = CMatrix(11, 9)
    for r in range(11):
        for c in range(9):
            a5.set(r, c, u5.get_re(r, c) * s4[8 - c], u5.get_im(r, c) * s4[8 - c])
    var r5 = svd(a5)
    check_valid(a5, r5, 1e-12, "graded columns")
    for t in range(9):
        assert_true(abs(r5.s[t] - s4[t]) <= 1e-13 * s4[t], "graded columns: relative error at " + String(t))
    # Scale invariance: tiny and huge matrices.
    for sc in [1e-200, 1e200]:
        var a6 = known_spectrum(6, 4, List[Float64](s1[0:4]), rng)
        for i in range(len(a6.re)):
            a6.re[i] *= sc
            a6.im[i] *= sc
        var r6 = svd(a6)
        for t in range(4):
            assert_true(abs(r6.s[t] / sc - s1[t]) <= 1e-13, "scaled " + String(sc))

    # Review findings (2026-11-27). Subnormal largest entry: 1 / scale overflowed to inf,
    # [[1e-310]] gave s = [nan] and diag(1e-310, 1e-311) a false QM-E431.
    var t1 = CMatrix(1, 1)
    t1.set(0, 0, 1e-310)
    var rt1 = svd(t1)
    assert_true(abs(rt1.s[0] - 1e-310) <= 1e-322, "1x1 subnormal: " + String(rt1.s[0]))
    assert_true(abs(rt1.u.get_re(0, 0) * rt1.v.get_re(0, 0) - 1.0) <= 1e-15, "1x1 subnormal vectors")
    var t2 = CMatrix(2, 2)
    t2.set(0, 0, 1e-310)
    t2.set(1, 1, 0.0, 1e-311)
    var rt2 = svd(t2)
    assert_true(abs(rt2.s[0] - 1e-310) <= 1e-322 and abs(rt2.s[1] - 1e-311) <= 1e-322, "diagonal subnormal")
    assert_true(orthonormality_error(rt2.u) <= 1e-15 and orthonormality_error(rt2.v) <= 1e-15, "subnormal: orthonormal")
    var t3 = random_matrix(5, 3, rng)
    for i in range(len(t3.re)):
        t3.re[i] *= 1e-310
        t3.im[i] *= 1e-310
    var rt3 = svd(t3)
    for t in range(3):
        assert_true(rt3.s[t] == rt3.s[t] and rt3.s[t] > 0.0, "subnormal random: finite")
    # Completion of the left vectors of zero singular values was O(k^3 p) (zero 256 x 256:
    # 5 s, 512 x 512: 83 s); it is O(p k^2) now. A zero 384 x 384 matrix (about 25 s
    # before) and a rank-1 300 x 200 one must finish well inside the limit below, with U
    # orthonormal.
    var tz0 = perf_counter_ns()
    var rz384 = svd(CMatrix(384, 384))
    var r1 = CMatrix(300, 200)
    for r in range(300):
        for c in range(200):
            r1.set(r, c, Float64(r % 7 + 1) * Float64(c % 5 + 1), 0.25 * Float64(r % 7 + 1))
    var rr1 = svd(r1)
    var dtz = Float64(perf_counter_ns() - tz0) / 1e9
    assert_true(dtz < 5.0, "zero-singular-value completion took " + String(dtz) + " s")
    assert_true(orthonormality_error(rz384.u) <= 1e-13, "zero 384: U orthonormal")
    check_valid(r1, rr1, 1e-12, "rank 1, 300 x 200")
    print("  zero 384x384 + rank-1 300x200:", dtz, "s")

    # Size cap and invalid input.
    expect_code(SVD_MAX_MIN_DIM + 1, SVD_MAX_MIN_DIM + 1, "QM-E430")
    expect_code(1 << 13, 1 << 10, "QM-E430")
    expect_code(-1, 3, "QM-E430")
    check_svd_size(1 << 22, 1)
    var bad = CMatrix(2, 2)
    bad.set(0, 1, Float64(0.0) / Float64(0.0))
    try:
        _ = svd(bad)
        raise Error("NaN accepted")
    except e:
        assert_true(String(e).startswith("QM-E430"), String(e))
    try:
        _ = CMatrix(-2, 2)
        raise Error("negative size accepted")
    except e:
        assert_true(String(e).startswith("QM-E430"), String(e))

    print("test_svd OK")
