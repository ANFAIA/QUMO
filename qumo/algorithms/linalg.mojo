"""Small linear-algebra kernels for the algorithm layer.

Three families, the first two allocation-free in their hot loops:
  * BLAS-1 style operations on `StateVector`s (split re/im storage, raw
    pointers) used by Lanczos, exact time evolution and adjoint gradients.
  * Dense real-symmetric eigensolver (cyclic Jacobi) and a Sturm-sequence
    bisection for symmetric tridiagonal matrices (Lanczos projections).
  * The Lanczos recurrence itself (`lanczos`, `HermitianOperator`,
    `LanczosRun`): one definition for `ground_state`,
    `subspace_ground_energy` and `subspace_ground_state`.

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


# ---------------------------------------------------------------------------
# Lanczos: the one Krylov recurrence of the algorithm layer
#
# `ground_state` (2^n amplitudes, `PauliOperator`), `subspace_ground_energy`
# (sparse projected matrix, eigenvalue only) and `subspace_ground_state`
# (restarted, ELL matrix, Ritz vector) all run `lanczos` below and differ only
# in the operator, the breakdown threshold, the stopping rule and what they do
# with the projection. Vectors are complex, split in two `List[Float64]`.
# ---------------------------------------------------------------------------
trait HermitianOperator:
    """y <- H x on a complex vector stored as (re, im). y is overwritten and
    never aliases x."""

    def apply_vec(self, xr: List[Float64], xi: List[Float64], mut yr: List[Float64], mut yi: List[Float64]):
        ...


def vec_dot(ar: List[Float64], ai: List[Float64], br: List[Float64], bi: List[Float64]) -> Tuple[Float64, Float64]:
    """<a|b> as (re, im), conjugate-linear in a. Serial, in index order, so the
    sum is the same on any thread count."""
    var sr = 0.0
    var si = 0.0
    for k in range(len(ar)):
        sr += ar[k] * br[k] + ai[k] * bi[k]
        si += ar[k] * bi[k] - ai[k] * br[k]
    return (sr, si)


def vec_norm(ar: List[Float64], ai: List[Float64]) -> Float64:
    var s = 0.0
    for k in range(len(ar)):
        s += ar[k] * ar[k] + ai[k] * ai[k]
    return sqrt(s)


def vec_axpy(mut yr: List[Float64], mut yi: List[Float64], ar: Float64, ai: Float64, xr: List[Float64], xi: List[Float64]):
    """y <- y + (ar + i ai) x, element-wise (parallel above PAR_MIN_ITEMS; the
    result does not depend on the thread count)."""
    var yrp = yr.unsafe_ptr().as_unsafe_any_origin()
    var yip = yi.unsafe_ptr().as_unsafe_any_origin()
    var xrp = xr.unsafe_ptr().as_unsafe_any_origin()
    var xip = xi.unsafe_ptr().as_unsafe_any_origin()

    def body(lo: Int, hi: Int) {imm yrp, imm yip, imm xrp, imm xip, imm ar, imm ai}:
        for i in range(lo, hi):
            var a = xrp[unsafe_offset=i]
            var b = xip[unsafe_offset=i]
            yrp[unsafe_offset=i] = yrp[unsafe_offset=i] + ar * a - ai * b
            yip[unsafe_offset=i] = yip[unsafe_offset=i] + ar * b + ai * a

    run_chunks(body, len(yr))
    _ = len(yi)
    _ = len(xr)
    _ = len(xi)


struct LanczosRun(Movable):
    """One Lanczos block: T = tridiag(alpha, beta) of size `steps` in the
    orthonormal basis (vr[k], vi[k]), k < steps. len(beta) == steps - 1."""

    var alpha: List[Float64]
    var beta: List[Float64]
    var vr: List[List[Float64]]
    var vi: List[List[Float64]]
    var steps: Int

    def __init__(out self):
        self.alpha = List[Float64]()
        self.beta = List[Float64]()
        self.vr = List[List[Float64]]()
        self.vi = List[List[Float64]]()
        self.steps = 0

    def lowest(self) -> Float64:
        """Smallest Ritz value (Sturm bisection on T)."""
        return tridiag_lowest(self.alpha, self.beta, self.steps)

    def ritz_vector(self, mut xr: List[Float64], mut xi: List[Float64]):
        """x <- sum_k y_k v_k for the lowest eigenvector y of T (Jacobi).
        Not normalised: y is a unit vector and the basis orthonormal, so the
        norm is 1 up to rounding and each caller normalises as it always did."""
        var m = self.steps
        var t = List[Float64](length=m * m, fill=0.0)
        for i in range(m):
            t[i * m + i] = self.alpha[i]
            if i + 1 < m:
                t[i * m + i + 1] = self.beta[i]
                t[(i + 1) * m + i] = self.beta[i]
        var eig = jacobi_eigh(t, m)
        for i in range(len(xr)):
            xr[i] = 0.0
            xi[i] = 0.0
        for k in range(m):
            vec_axpy(xr, xi, eig.vectors[k * m + 0], 0.0, self.vr[k], self.vi[k])


def lanczos[Op: HermitianOperator](op: Op, var x0r: List[Float64], var x0i: List[Float64], max_steps: Int,
                                   breakdown: Float64, conv_tol: Float64 = 0.0, check_every: Int = 4) -> LanczosRun:
    """Lanczos with full re-orthogonalisation (two passes, Kahan/Parlett) from
    the unit vector x0, at most `max_steps` steps.

    Stops early when the next residual norm is below `breakdown` (an invariant
    subspace was found) or, if `conv_tol > 0`, when the lowest Ritz value moved
    by less than `conv_tol * max(1, |theta|)` between two checks, made every
    `check_every` steps. Memory: `steps` vectors plus one work vector.
    """
    var run = LanczosRun()
    var n = len(x0r)
    var wr = List[Float64](length=n, fill=0.0)
    var wi = List[Float64](length=n, fill=0.0)
    run.vr.append(x0r^)
    run.vi.append(x0i^)
    var theta_old = 1e300
    var k = 0
    while k < max_steps:
        op.apply_vec(run.vr[k], run.vi[k], wr, wi)
        var a = vec_dot(run.vr[k], run.vi[k], wr, wi)[0]
        run.alpha.append(a)
        vec_axpy(wr, wi, -a, 0.0, run.vr[k], run.vi[k])
        if k > 0:
            vec_axpy(wr, wi, -run.beta[k - 1], 0.0, run.vr[k - 1], run.vi[k - 1])
        for _ in range(2):
            for j in range(k + 1):
                var c = vec_dot(run.vr[j], run.vi[j], wr, wi)
                vec_axpy(wr, wi, -c[0], -c[1], run.vr[j], run.vi[j])
        var b = vec_norm(wr, wi)
        k += 1
        var done = b < breakdown or k >= max_steps
        if not done and conv_tol > 0.0 and (k % check_every == 0):
            var th = tridiag_lowest(run.alpha, run.beta, k)
            if abs(th - theta_old) < conv_tol * max(1.0, abs(th)):
                done = True
            theta_old = th
        if done:
            break
        run.beta.append(b)
        var s = 1.0 / b
        var nr = List[Float64](length=n, fill=0.0)
        var ni = List[Float64](length=n, fill=0.0)
        for i in range(n):
            nr[i] = wr[i] * s
            ni[i] = wi[i] * s
        run.vr.append(nr^)
        run.vi.append(ni^)
    run.steps = k
    return run^
