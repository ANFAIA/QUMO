"""Complex singular value decomposition by one-sided Jacobi.

`svd(a)` returns the thin SVD A = U diag(s) V^H of a complex m x n matrix: k = min(m, n),
U is m x k with orthonormal columns, s holds k singular values in non-increasing order and
V is n x k with orthonormal columns. It is the decomposition the matrix product state
simulator (`qumo.sim.mps`) truncates, and it is a plain numerical kernel: it imports nothing
from QUMO, so every layer may use it.

Method: one-sided Jacobi (Hestenes), Algorithm 4.1 of J. Demmel and K. Veselic, "Jacobi's
method is more accurate than QR", LAPACK Working Note 15 (1989), section 4,
https://www.netlib.org/lapack/lawnspdf/lawn15.pdf (SIAM J. Matrix Anal. Appl. 13 (1992)
1204-1245). Plane rotations are applied to pairs of columns of a working copy G of A (and
accumulated in V) until every pair is orthogonal to working precision; the singular values
are then the column norms of G and the left singular vectors its normalized columns.

* A^H A is never formed. Each rotation needs only the three numbers ||g_i||^2, ||g_j||^2 and
  g_i^H g_j of the pair it acts on (the 2 x 2 pivot of the paper), recomputed from the current
  columns every time.
* Complex entries: write g_i^H g_j = |c| e^{i phi}. Multiplying column j by e^{-i phi} makes the
  pivot real, and the paper's real rotation (zeta = (b - a) / (2|c|),
  t = sign(zeta) / (|zeta| + sqrt(1 + zeta^2)), cs = 1 / sqrt(1 + t^2), sn = cs t) then
  orthogonalizes the pair. The combined 2 x 2 matrix [[cs, sn], [-sn e^{-i phi}, cs e^{-i phi}]]
  is unitary, so A = G V^H holds throughout.
* Stopping rule: the paper's relative criterion |g_i^H g_j| <= tol sqrt(||g_i||^2 ||g_j||^2) for
  every pair, with tol = max(m, n) * 2^-52 here. A sweep that rotates no pair ends the
  iteration. At most SVD_MAX_SWEEPS sweeps run; a matrix that has not converged by then raises
  QM-E431 (it has not happened in the tests; Jacobi converges quadratically).
* Wide matrices (m < n) are decomposed through A^H, so the working matrix always has at least
  as many rows as columns.
* A singular value that is exactly zero has no column of G to normalize; its left singular
  vector is completed by Gram-Schmidt (two passes) against the other columns of U, so U has
  orthonormal columns also for rank-deficient input.
* The input is scaled by its largest entry modulus before the sweeps and the singular values
  scaled back, so overflow does not occur; entries smaller than about 1e-154 times the largest
  one underflow when squared and count as zero in the pivots.

Bounds (QM-E430): min(m, n) <= SVD_MAX_MIN_DIM (1024) and m * n <= SVD_MAX_ENTRIES (2^22, 64 MiB
of complex entries); entries must be finite. A sweep costs O(m n^2) flops for n <= m, so the
cap also bounds the running time.
"""

from std.math import sqrt

comptime SVD_MAX_MIN_DIM = 1024
comptime SVD_MAX_ENTRIES = 1 << 22
comptime SVD_MAX_SWEEPS = 60
comptime SVD_EPS = 2.220446049250313e-16  # 2^-52
comptime CMATRIX_MAX_ENTRIES = 1 << 28


struct CMatrix(Copyable, Writable):
    """Dense complex matrix, row-major, split real and imaginary parts."""

    var rows: Int
    var cols: Int
    var re: List[Float64]
    var im: List[Float64]

    def __init__(out self, rows: Int, cols: Int) raises:
        if rows < 0 or cols < 0 or (rows > 0 and cols > CMATRIX_MAX_ENTRIES // rows):
            raise Error("QM-E430: a complex matrix of " + String(rows) + " x " + String(cols) + " entries cannot be allocated (dimensions must be >= 0 and the product at most 2^28). Hint: pass non-negative dimensions; split the problem if it is this large.")
        self.rows = rows
        self.cols = cols
        self.re = List[Float64](length=rows * cols, fill=0.0)
        self.im = List[Float64](length=rows * cols, fill=0.0)

    @staticmethod
    def identity(n: Int) raises -> CMatrix:
        var m = CMatrix(n, n)
        for i in range(n):
            m.re[i * n + i] = 1.0
        return m^

    @always_inline
    def get_re(self, r: Int, c: Int) -> Float64:
        return self.re[r * self.cols + c]

    @always_inline
    def get_im(self, r: Int, c: Int) -> Float64:
        return self.im[r * self.cols + c]

    @always_inline
    def set(mut self, r: Int, c: Int, re: Float64, im: Float64 = 0.0):
        self.re[r * self.cols + c] = re
        self.im[r * self.cols + c] = im

    def write_to(self, mut w: Some[Writer]):
        w.write("CMatrix(", self.rows, "x", self.cols, ")")


struct SVDResult(Copyable):
    """Thin SVD A = U diag(s) V^H. U: m x k, V: n x k, k = min(m, n), s non-increasing."""

    var u: CMatrix
    var s: List[Float64]
    var v: CMatrix
    var sweeps: Int  # Jacobi sweeps run, the last one rotating no pair

    def __init__(out self, var u: CMatrix, var s: List[Float64], var v: CMatrix, sweeps: Int):
        self.u = u^
        self.s = s^
        self.v = v^
        self.sweeps = sweeps


def check_svd_size(rows: Int, cols: Int) raises:
    """Raise QM-E430 if `svd` refuses a rows x cols matrix."""
    var lo = min(rows, cols)
    if rows < 0 or cols < 0:
        raise Error("QM-E430: svd got a " + String(rows) + " x " + String(cols) + " matrix. Hint: dimensions must be >= 0.")
    if lo > SVD_MAX_MIN_DIM or (rows > 0 and cols > SVD_MAX_ENTRIES // rows):
        raise Error("QM-E430: svd of a " + String(rows) + " x " + String(cols) + " matrix is over the limit (min(m, n) <= " + String(SVD_MAX_MIN_DIM) + " and m * n <= 2^22). Hint: the one-sided Jacobi SVD costs O(m n^2) per sweep; reduce the matrix (for an MPS, a smaller max_bond).")


def _jacobi_sweeps(mut wr: List[Float64], mut wi: List[Float64], p: Int, q: Int,
                   mut vr: List[Float64], mut vi: List[Float64]) raises -> Int:
    """Orthogonalize the q columns (length p, column-major) of W; accumulate the
    rotations in the q x q column-major V. Returns the number of sweeps."""
    var tol = SVD_EPS * Float64(max(p, q))
    var gr = wr.unsafe_ptr().as_unsafe_any_origin()
    var gi = wi.unsafe_ptr().as_unsafe_any_origin()
    var pr = vr.unsafe_ptr().as_unsafe_any_origin()
    var pi_ = vi.unsafe_ptr().as_unsafe_any_origin()
    for sweep in range(SVD_MAX_SWEEPS):
        var rotated = False
        for i in range(q - 1):
            var oi = i * p
            for j in range(i + 1, q):
                var oj = j * p
                var a = 0.0
                var b = 0.0
                var cr = 0.0
                var ci = 0.0
                for k in range(p):
                    var xr = gr[unsafe_offset=oi + k]
                    var xi = gi[unsafe_offset=oi + k]
                    var yr = gr[unsafe_offset=oj + k]
                    var yi = gi[unsafe_offset=oj + k]
                    a += xr * xr + xi * xi
                    b += yr * yr + yi * yi
                    cr += xr * yr + xi * yi  # conj(x) * y
                    ci += xr * yi - xi * yr
                if a == 0.0 or b == 0.0:
                    continue
                var cabs = sqrt(cr * cr + ci * ci)
                if cabs <= tol * sqrt(a) * sqrt(b):
                    continue
                rotated = True
                var zeta = (b - a) / (2.0 * cabs)
                var t: Float64
                if abs(zeta) > 1e150:
                    t = 0.5 / zeta
                else:
                    var sg = 1.0 if zeta >= 0.0 else -1.0
                    t = sg / (abs(zeta) + sqrt(1.0 + zeta * zeta))
                var cs = 1.0 / sqrt(1.0 + t * t)
                var sn = cs * t
                var er = cr / cabs  # e^{-i phi} = conj(c) / |c|
                var ei = -ci / cabs
                for k in range(p):
                    var xr = gr[unsafe_offset=oi + k]
                    var xi = gi[unsafe_offset=oi + k]
                    var yr = gr[unsafe_offset=oj + k]
                    var yi = gi[unsafe_offset=oj + k]
                    var ypr = er * yr - ei * yi
                    var ypi = er * yi + ei * yr
                    gr[unsafe_offset=oi + k] = cs * xr - sn * ypr
                    gi[unsafe_offset=oi + k] = cs * xi - sn * ypi
                    gr[unsafe_offset=oj + k] = sn * xr + cs * ypr
                    gi[unsafe_offset=oj + k] = sn * xi + cs * ypi
                var vi0 = i * q
                var vj0 = j * q
                for k in range(q):
                    var xr = pr[unsafe_offset=vi0 + k]
                    var xi = pi_[unsafe_offset=vi0 + k]
                    var yr = pr[unsafe_offset=vj0 + k]
                    var yi = pi_[unsafe_offset=vj0 + k]
                    var ypr = er * yr - ei * yi
                    var ypi = er * yi + ei * yr
                    pr[unsafe_offset=vi0 + k] = cs * xr - sn * ypr
                    pi_[unsafe_offset=vi0 + k] = cs * xi - sn * ypi
                    pr[unsafe_offset=vj0 + k] = sn * xr + cs * ypr
                    pi_[unsafe_offset=vj0 + k] = sn * xi + cs * ypi
        if not rotated:
            _ = len(wr)
            _ = len(wi)
            _ = len(vr)
            _ = len(vi)
            return sweep + 1
    raise Error("QM-E431: the one-sided Jacobi SVD of a " + String(p) + " x " + String(q) + " matrix did not converge in " + String(SVD_MAX_SWEEPS) + " sweeps. Hint: report the matrix; check it for entries spanning more than ~300 orders of magnitude.")


def svd(a: CMatrix) raises -> SVDResult:
    """Thin SVD of `a` by one-sided Jacobi (see the module docstring)."""
    var m = a.rows
    var n = a.cols
    check_svd_size(m, n)
    var wide = m < n
    var p = n if wide else m  # rows of the working matrix (p >= q)
    var q = m if wide else n
    var k = q
    var scale = 0.0
    for i in range(m * n):
        var x = abs(a.re[i])
        var y = abs(a.im[i])
        if not (x <= 1.7976931348623157e308 and y <= 1.7976931348623157e308):
            raise Error("QM-E430: svd got a non-finite entry (NaN or infinity) at row " + String(i // max(n, 1)) + ", column " + String(i % max(n, 1)) + ". Hint: the matrix entries must be finite.")
        scale = max(scale, max(x, y))
    # Working matrix W (p x q, column-major): A, or A^H when wide.
    var wr = List[Float64](length=p * q, fill=0.0)
    var wi = List[Float64](length=p * q, fill=0.0)
    var inv = 1.0 / scale if scale > 0.0 else 1.0
    for r in range(m):
        for c in range(n):
            var x = a.re[r * n + c] * inv
            var y = a.im[r * n + c] * inv
            if wide:  # W = A^H: W[c, r] = conj(A[r, c]); column r of W
                wr[r * p + c] = x
                wi[r * p + c] = -y
            else:  # W = A: column c
                wr[c * p + r] = x
                wi[c * p + r] = y
    var vr = List[Float64](length=q * q, fill=0.0)
    var vi = List[Float64](length=q * q, fill=0.0)
    for i in range(q):
        vr[i * q + i] = 1.0
    var sweeps = 0
    if q > 0 and scale > 0.0:
        sweeps = _jacobi_sweeps(wr, wi, p, q, vr, vi)
    # Column norms and order.
    var norms = List[Float64](length=q, fill=0.0)
    for j in range(q):
        var s2 = 0.0
        for i in range(p):
            s2 += wr[j * p + i] * wr[j * p + i] + wi[j * p + i] * wi[j * p + i]
        norms[j] = sqrt(s2)
    var order = List[Int](capacity=q)
    for j in range(q):
        order.append(j)
    for i in range(1, q):  # insertion sort, descending, stable
        var x = order[i]
        var j = i - 1
        while j >= 0 and norms[order[j]] < norms[x]:
            order[j + 1] = order[j]
            j -= 1
        order[j + 1] = x
    # Left vectors of W (p x k, column-major) and right vectors (q x k).
    var ur = List[Float64](length=p * k, fill=0.0)
    var ui = List[Float64](length=p * k, fill=0.0)
    var s = List[Float64](length=k, fill=0.0)
    var zero_cols = List[Int]()
    for t in range(k):
        var j = order[t]
        var nj = norms[j]
        s[t] = nj * scale
        if nj > 0.0:
            var r = 1.0 / nj
            for i in range(p):
                ur[t * p + i] = wr[j * p + i] * r
                ui[t * p + i] = wi[j * p + i] * r
        else:
            zero_cols.append(t)
    for t in zero_cols:
        _complete_column(ur, ui, p, k, t)
    # Assemble U (m x k) and V (n x k), row-major.
    var u = CMatrix(m, k)
    var v = CMatrix(n, k)
    for t in range(k):
        var j = order[t]
        if wide:  # A = Vw diag(s) Uw^H: U = Vw (q x k = m x k), V = Uw (p x k = n x k)
            for i in range(q):
                u.set(i, t, vr[j * q + i], vi[j * q + i])
            for i in range(p):
                v.set(i, t, ur[t * p + i], ui[t * p + i])
        else:
            for i in range(p):
                u.set(i, t, ur[t * p + i], ui[t * p + i])
            for i in range(q):
                v.set(i, t, vr[j * q + i], vi[j * q + i])
    return SVDResult(u^, s^, v^, sweeps)


def _complete_column(mut ur: List[Float64], mut ui: List[Float64], p: Int, k: Int, t: Int):
    """Column t of U (zero singular value): the unit vector e_r whose component orthogonal
    to the other filled columns is largest, orthogonalized twice and normalized."""
    var best = -1.0
    var br = List[Float64](length=p, fill=0.0)
    var bi = List[Float64](length=p, fill=0.0)
    for r in range(p):
        var xr = List[Float64](length=p, fill=0.0)
        var xi = List[Float64](length=p, fill=0.0)
        xr[r] = 1.0
        for _pass in range(2):
            for c in range(k):
                if c == t:
                    continue
                var o = c * p
                var dr = 0.0
                var di = 0.0
                for i in range(p):  # u_c^H x
                    dr += ur[o + i] * xr[i] + ui[o + i] * xi[i]
                    di += ur[o + i] * xi[i] - ui[o + i] * xr[i]
                for i in range(p):  # x -= u_c (u_c^H x)
                    xr[i] -= ur[o + i] * dr - ui[o + i] * di
                    xi[i] -= ur[o + i] * di + ui[o + i] * dr
        var nn = 0.0
        for i in range(p):
            nn += xr[i] * xr[i] + xi[i] * xi[i]
        if nn > best:
            best = nn
            br = xr^
            bi = xi^
        if best > 0.5:
            break
    var inv = 1.0 / sqrt(best)
    for i in range(p):
        ur[t * p + i] = br[i] * inv
        ui[t * p + i] = bi[i] * inv
