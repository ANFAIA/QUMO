"""Small linear-algebra kernels for the algorithm layer.

Two families, both allocation-free in their hot loops:
  * BLAS-1 style operations on `StateVector`s (split re/im storage, raw
    pointers) used by Lanczos, exact time evolution and adjoint gradients.
  * Dense real-symmetric eigensolver (cyclic Jacobi) and a Sturm-sequence
    bisection for symmetric tridiagonal matrices (Lanczos projections).

Jacobi is O(m^3) per sweep but unconditionally stable and gives orthonormal
eigenvectors - the right trade-off for the <= 256 x 256 reference problems
QUMO diagonalises in tests and for small Krylov projections.
"""

from std.math import sqrt
from qumo.sim.statevector import StateVector, run_chunks


# ---------------------------------------------------------------------------
# State-vector BLAS-1
# ---------------------------------------------------------------------------
def sv_dot(a: StateVector, b: StateVector) -> Tuple[Float64, Float64]:
    """<a|b> as (re, im)."""
    return a.inner(b)


def sv_norm(a: StateVector) -> Float64:
    return sqrt(a.norm_sq())


def sv_zero(mut a: StateVector):
    var rp = a.re.unsafe_ptr().as_unsafe_any_origin()
    var ip = a.im.unsafe_ptr().as_unsafe_any_origin()

    def body(lo: Int, hi: Int) {imm rp, imm ip}:
        for i in range(lo, hi):
            rp[unsafe_offset=i] = 0.0
            ip[unsafe_offset=i] = 0.0

    run_chunks(body, a.size())


def sv_assign(mut dst: StateVector, src: StateVector):
    """dst <- src (same size, no reallocation)."""
    var dr = dst.re.unsafe_ptr().as_unsafe_any_origin()
    var di = dst.im.unsafe_ptr().as_unsafe_any_origin()
    var sr = src.re.unsafe_ptr().as_unsafe_any_origin()
    var si = src.im.unsafe_ptr().as_unsafe_any_origin()

    def body(lo: Int, hi: Int) {imm dr, imm di, imm sr, imm si}:
        for i in range(lo, hi):
            dr[unsafe_offset=i] = sr[unsafe_offset=i]
            di[unsafe_offset=i] = si[unsafe_offset=i]

    run_chunks(body, dst.size())


def sv_scale(mut a: StateVector, s: Float64):
    var rp = a.re.unsafe_ptr().as_unsafe_any_origin()
    var ip = a.im.unsafe_ptr().as_unsafe_any_origin()

    def body(lo: Int, hi: Int) {imm rp, imm ip, imm s}:
        for i in range(lo, hi):
            rp[unsafe_offset=i] = rp[unsafe_offset=i] * s
            ip[unsafe_offset=i] = ip[unsafe_offset=i] * s

    run_chunks(body, a.size())


def sv_axpy(mut y: StateVector, ar: Float64, ai: Float64, x: StateVector):
    """y <- y + (ar + i ai) x."""
    var yr = y.re.unsafe_ptr().as_unsafe_any_origin()
    var yi = y.im.unsafe_ptr().as_unsafe_any_origin()
    var xr = x.re.unsafe_ptr().as_unsafe_any_origin()
    var xi = x.im.unsafe_ptr().as_unsafe_any_origin()

    def body(lo: Int, hi: Int) {imm yr, imm yi, imm xr, imm xi, imm ar, imm ai}:
        for i in range(lo, hi):
            var a = xr[unsafe_offset=i]
            var b = xi[unsafe_offset=i]
            yr[unsafe_offset=i] = yr[unsafe_offset=i] + ar * a - ai * b
            yi[unsafe_offset=i] = yi[unsafe_offset=i] + ar * b + ai * a

    run_chunks(body, y.size())


def sv_mul_minus_i(mut a: StateVector, s: Float64):
    """a <- -i s a (used by Taylor time stepping)."""
    var rp = a.re.unsafe_ptr().as_unsafe_any_origin()
    var ip = a.im.unsafe_ptr().as_unsafe_any_origin()

    def body(lo: Int, hi: Int) {imm rp, imm ip, imm s}:
        for i in range(lo, hi):
            var r = rp[unsafe_offset=i]
            var m = ip[unsafe_offset=i]
            rp[unsafe_offset=i] = s * m
            ip[unsafe_offset=i] = -s * r

    run_chunks(body, a.size())


# ---------------------------------------------------------------------------
# Dense symmetric eigensolver (cyclic Jacobi)
# ---------------------------------------------------------------------------
struct EighResult(Movable):
    """Eigen-decomposition of a real symmetric m x m matrix.

    `values` ascending; `vectors` row-major m x m with eigenvector k stored in
    column k (vectors[i * m + k]).
    """

    var values: List[Float64]
    var vectors: List[Float64]
    var m: Int

    def __init__(out self, var values: List[Float64], var vectors: List[Float64], m: Int):
        self.values = values^
        self.vectors = vectors^
        self.m = m

    def vector(self, k: Int) -> List[Float64]:
        var out = List[Float64](length=self.m, fill=0.0)
        for i in range(self.m):
            out[i] = self.vectors[i * self.m + k]
        return out^


def jacobi_eigh(a_in: List[Float64], m: Int, max_sweeps: Int = 100) -> EighResult:
    """Eigenpairs of a real symmetric row-major m x m matrix (cyclic Jacobi)."""
    var a = a_in.copy()
    var v = List[Float64](length=m * m, fill=0.0)
    for i in range(m):
        v[i * m + i] = 1.0
    var ap = a.unsafe_ptr().as_unsafe_any_origin()
    var vp = v.unsafe_ptr().as_unsafe_any_origin()
    var scale = 0.0
    for i in range(m * m):
        scale += a[i] * a[i]
    for _ in range(max_sweeps):
        var off = 0.0
        for p in range(m):
            for q in range(p + 1, m):
                off += ap[unsafe_offset=p * m + q] * ap[unsafe_offset=p * m + q]
        if off <= 1e-30 * scale or off == 0.0:
            break
        for p in range(m - 1):
            for q in range(p + 1, m):
                var apq = ap[unsafe_offset=p * m + q]
                if abs(apq) < 1e-300:
                    continue
                var app = ap[unsafe_offset=p * m + p]
                var aqq = ap[unsafe_offset=q * m + q]
                var theta = (aqq - app) / (2.0 * apq)
                var t = 1.0 / (abs(theta) + sqrt(theta * theta + 1.0))
                if theta < 0.0:
                    t = -t
                var c = 1.0 / sqrt(t * t + 1.0)
                var s = t * c
                for k in range(m):
                    var akp = ap[unsafe_offset=k * m + p]
                    var akq = ap[unsafe_offset=k * m + q]
                    ap[unsafe_offset=k * m + p] = c * akp - s * akq
                    ap[unsafe_offset=k * m + q] = s * akp + c * akq
                for k in range(m):
                    var apk = ap[unsafe_offset=p * m + k]
                    var aqk = ap[unsafe_offset=q * m + k]
                    ap[unsafe_offset=p * m + k] = c * apk - s * aqk
                    ap[unsafe_offset=q * m + k] = s * apk + c * aqk
                for k in range(m):
                    var vkp = vp[unsafe_offset=k * m + p]
                    var vkq = vp[unsafe_offset=k * m + q]
                    vp[unsafe_offset=k * m + p] = c * vkp - s * vkq
                    vp[unsafe_offset=k * m + q] = s * vkp + c * vkq
    # sort ascending (selection sort on indices; m is small)
    var order = List[Int](length=m, fill=0)
    for i in range(m):
        order[i] = i
    for i in range(m):
        var best = i
        for j in range(i + 1, m):
            if a[order[j] * m + order[j]] < a[order[best] * m + order[best]]:
                best = j
        var tmp = order[i]
        order[i] = order[best]
        order[best] = tmp
    var vals = List[Float64](length=m, fill=0.0)
    var vecs = List[Float64](length=m * m, fill=0.0)
    for k in range(m):
        var src = order[k]
        vals[k] = a[src * m + src]
        for i in range(m):
            vecs[i * m + k] = v[i * m + src]
    return EighResult(vals^, vecs^, m)


# ---------------------------------------------------------------------------
# Symmetric tridiagonal: lowest eigenvalue by Sturm bisection
# ---------------------------------------------------------------------------
def _sturm_count(alpha: List[Float64], beta: List[Float64], m: Int, x: Float64) -> Int:
    """Number of eigenvalues of T (diag alpha, offdiag beta) strictly below x."""
    var cnt = 0
    var d = alpha[0] - x
    if d < 0.0:
        cnt += 1
    for i in range(1, m):
        if d == 0.0:
            d = 1e-300
        d = alpha[i] - x - beta[i - 1] * beta[i - 1] / d
        if d < 0.0:
            cnt += 1
    return cnt


def tridiag_lowest(alpha: List[Float64], beta: List[Float64], m: Int) -> Float64:
    """Smallest eigenvalue of the leading m x m block of a symmetric tridiagonal matrix."""
    var lo = alpha[0]
    var hi = alpha[0]
    for i in range(m):
        var r = 0.0
        if i > 0:
            r += abs(beta[i - 1])
        if i < m - 1:
            r += abs(beta[i])
        lo = min(lo, alpha[i] - r)
        hi = max(hi, alpha[i] + r)
    for _ in range(200):
        var mid = 0.5 * (lo + hi)
        if mid == lo or mid == hi:
            break
        if _sturm_count(alpha, beta, m, mid) >= 1:
            hi = mid
        else:
            lo = mid
        if hi - lo <= 1e-15 * max(1.0, abs(lo)):
            break
    return 0.5 * (lo + hi)
