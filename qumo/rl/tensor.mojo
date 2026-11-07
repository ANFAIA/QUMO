"""f32 matrices, GEMM family and vector kernels for the native RL stack.

Everything here is portable SIMD Mojo (no vendor intrinsics, no BLAS): a register-tiled outer-product GEMM and a few
row-wise vector kernels, all operating on raw f32 pointers (`FPtr`). MOJO_PORT.md sections 4.2 and 15 (G1).

Conventions
  * Row-major storage with an explicit leading dimension (`lda`, `ldb`, `ldc` = distance in floats between rows).
  * `gemm_*` compute `C[m, n] (+)= op(A) . op(B)`; `acc = True` accumulates into C, `acc = False` overwrites it.
      gemm_nn : C = A[m,k]   . B[k,n]      backward `dx = dz W`, forward when the weight is held transposed
      gemm_nt : C = A[m,k]   . B[n,k]^T    forward `x W^T`
      gemm_tn : C = A[k,m]^T . B[k,n]      backward `dW = dz^T x`
  * Kernels are single-threaded and re-entrant (the nt variant allocates its own scratch); threading happens one
    level up (virtual workers, MOJO_PORT.md section 3).
  * Raw-pointer code: there are no bounds checks. A kernel reads exactly the elements of the operands it is given
    (`A`: m rows of k, `B`: k rows of n, etc.) and writes exactly `C[0:m, 0:n]`; the tests surround every buffer with
    sentinel guard regions to prove it.
  * Every output element of gemm_nn / gemm_tn is a single sequential fused-multiply-add chain over the k index, so
    its value does not depend on which tile or SIMD width produced it (bit-identical across edge/interior tiles).
    gemm_nt uses the same chain except on its small-matrix path, which sums with SIMD partial sums along k.
  * Callers keep the owning lists alive with `_ = len(lst)` after the last use of a pointer taken from them
    (ASAP destruction, docs/MOJO_GUIDE.md).
"""

from std.math import fma

# The one pointer alias of the contract; every other qumo.rl module imports it from here.
comptime FPtr = MutPointer[Float32, MutAnyOrigin]
comptime LANES = 8
comptime CHUNK_ROWS = 512
comptime VW = 16

comptime _F32 = DType.float32


struct Mat(Movable, Copyable):
    """Row-major f32 matrix."""

    var rows: Int
    var cols: Int
    var data: List[Float32]

    def __init__(out self, rows: Int, cols: Int):
        self.rows = rows
        self.cols = cols
        self.data = List[Float32](length=rows * cols, fill=0.0)

    def ptr(mut self) -> FPtr:
        return self.data.unsafe_ptr().as_unsafe_any_origin()


# ------------------------------------------------------------------------------------------------ GEMM tile kernel


@always_inline
def _tile[
    MR: Int, W: Int, TA: Bool
](
    k: Int,
    a: FPtr,
    lda: Int,
    i0: Int,
    b: FPtr,
    ldb: Int,
    j0: Int,
    c: FPtr,
    ldc: Int,
    acc: Bool,
):
    """One MR x W output tile: `C[i0:i0+MR, j0:j0+W] (+)= sum_p A(i, p) * B[p, j0:j0+W]`.

    `A(i, p)` is `a[i*lda + p]` (TA False) or `a[p*lda + i]` (TA True). The MR accumulator vectors stay in registers
    for the whole k loop (4 x 16 floats for the main tile)."""
    var c0 = SIMD[_F32, W](0)
    var c1 = SIMD[_F32, W](0)
    var c2 = SIMD[_F32, W](0)
    var c3 = SIMD[_F32, W](0)
    if acc:
        c0 = c.unsafe_load[width=W](i0 * ldc + j0)
        comptime if MR > 1:
            c1 = c.unsafe_load[width=W]((i0 + 1) * ldc + j0)
        comptime if MR > 2:
            c2 = c.unsafe_load[width=W]((i0 + 2) * ldc + j0)
            c3 = c.unsafe_load[width=W]((i0 + 3) * ldc + j0)
    for p in range(k):
        var bv = b.unsafe_load[width=W](p * ldb + j0)
        comptime if TA:
            var base = p * lda + i0
            c0 = fma(SIMD[_F32, W](a[unsafe_offset=base]), bv, c0)
            comptime if MR > 1:
                c1 = fma(SIMD[_F32, W](a[unsafe_offset=base + 1]), bv, c1)
            comptime if MR > 2:
                c2 = fma(SIMD[_F32, W](a[unsafe_offset=base + 2]), bv, c2)
                c3 = fma(SIMD[_F32, W](a[unsafe_offset=base + 3]), bv, c3)
        else:
            c0 = fma(SIMD[_F32, W](a[unsafe_offset=i0 * lda + p]), bv, c0)
            comptime if MR > 1:
                c1 = fma(SIMD[_F32, W](a[unsafe_offset=(i0 + 1) * lda + p]), bv, c1)
            comptime if MR > 2:
                c2 = fma(SIMD[_F32, W](a[unsafe_offset=(i0 + 2) * lda + p]), bv, c2)
                c3 = fma(SIMD[_F32, W](a[unsafe_offset=(i0 + 3) * lda + p]), bv, c3)
    c.unsafe_store(i0 * ldc + j0, c0)
    comptime if MR > 1:
        c.unsafe_store((i0 + 1) * ldc + j0, c1)
    comptime if MR > 2:
        c.unsafe_store((i0 + 2) * ldc + j0, c2)
        c.unsafe_store((i0 + 3) * ldc + j0, c3)


@always_inline
def _row_block[
    MR: Int, TA: Bool
](
    n: Int,
    k: Int,
    a: FPtr,
    lda: Int,
    i0: Int,
    b: FPtr,
    ldb: Int,
    c: FPtr,
    ldc: Int,
    acc: Bool,
):
    """MR output rows over all n columns: 16-wide tiles, then 8, 4 and single columns for the right edge."""
    var j = 0
    while j + 16 <= n:
        _tile[MR, 16, TA](k, a, lda, i0, b, ldb, j, c, ldc, acc)
        j += 16
    if j + 8 <= n:
        _tile[MR, 8, TA](k, a, lda, i0, b, ldb, j, c, ldc, acc)
        j += 8
    if j + 4 <= n:
        _tile[MR, 4, TA](k, a, lda, i0, b, ldb, j, c, ldc, acc)
        j += 4
    while j < n:
        _tile[MR, 1, TA](k, a, lda, i0, b, ldb, j, c, ldc, acc)
        j += 1


@always_inline
def _gemm_core[
    TA: Bool
](m: Int, n: Int, k: Int, a: FPtr, lda: Int, b: FPtr, ldb: Int, c: FPtr, ldc: Int, acc: Bool):
    """Shared driver of gemm_nn (TA False) and gemm_tn (TA True): 4-row blocks, then 2 and 1 rows."""
    if m <= 0 or n <= 0:
        return
    var i = 0
    while i + 4 <= m:
        _row_block[4, TA](n, k, a, lda, i, b, ldb, c, ldc, acc)
        i += 4
    if i + 2 <= m:
        _row_block[2, TA](n, k, a, lda, i, b, ldb, c, ldc, acc)
        i += 2
    if i < m:
        _row_block[1, TA](n, k, a, lda, i, b, ldb, c, ldc, acc)


def gemm_nn(m: Int, n: Int, k: Int, a: FPtr, lda: Int, b: FPtr, ldb: Int, c: FPtr, ldc: Int, acc: Bool):
    """C[m,n] (+)= A[m,k] . B[k,n]."""
    _gemm_core[False](m, n, k, a, lda, b, ldb, c, ldc, acc)


def gemm_tn(m: Int, n: Int, k: Int, a: FPtr, lda: Int, b: FPtr, ldb: Int, c: FPtr, ldc: Int, acc: Bool):
    """C[m,n] (+)= A[k,m]^T . B[k,n]."""
    _gemm_core[True](m, n, k, a, lda, b, ldb, c, ldc, acc)


# ------------------------------------------------------------------------------------------------ gemm_nt


def transpose_into(rows: Int, cols: Int, src: FPtr, lds: Int, dst: FPtr, ldd: Int):
    """`dst[j*ldd + i] = src[i*lds + j]` for `i < rows`, `j < cols` (cache-blocked 8x8 tiles). Used by gemm_nt and by
    callers that keep a transposed copy of a weight for `gemm_nn`."""
    var ib = 0
    while ib < rows:
        var ie = min(ib + 8, rows)
        var jb = 0
        while jb < cols:
            var je = min(jb + 8, cols)
            for i in range(ib, ie):
                for j in range(jb, je):
                    dst[unsafe_offset = j * ldd + i] = src[unsafe_offset = i * lds + j]
            jb += 8
        ib += 8


def _gemm_nt_dot(m: Int, n: Int, k: Int, a: FPtr, lda: Int, b: FPtr, ldb: Int, c: FPtr, ldc: Int, acc: Bool):
    """Small-matrix path: every output is a SIMD dot product of an A row with a B row (two accumulators)."""
    var main = (k // (2 * LANES)) * (2 * LANES)
    for i in range(m):
        var ai = i * lda
        for j in range(n):
            var bj = j * ldb
            var s0 = SIMD[_F32, LANES](0)
            var s1 = SIMD[_F32, LANES](0)
            var p = 0
            while p < main:
                s0 = fma(a.unsafe_load[width=LANES](ai + p), b.unsafe_load[width=LANES](bj + p), s0)
                s1 = fma(a.unsafe_load[width=LANES](ai + p + LANES), b.unsafe_load[width=LANES](bj + p + LANES), s1)
                p += 2 * LANES
            var v = (s0 + s1).reduce_add()
            while p < k:
                v = fma(a[unsafe_offset = ai + p], b[unsafe_offset = bj + p], v)
                p += 1
            var o = i * ldc + j
            if acc:
                v += c[unsafe_offset=o]
            c[unsafe_offset=o] = v


def gemm_nt(m: Int, n: Int, k: Int, a: FPtr, lda: Int, b: FPtr, ldb: Int, c: FPtr, ldc: Int, acc: Bool):
    """C[m,n] (+)= A[m,k] . B[n,k]^T.

    For matrices with at least 8 rows and 8 columns the weight `B` is transposed into a scratch buffer and `gemm_nn`
    runs (the transpose is O(n k) against O(m n k) of work); a caller that multiplies by the same weight many times
    keeps its own transposed copy (`transpose_into`) and calls `gemm_nn` directly. Smaller shapes use dot products."""
    if m <= 0 or n <= 0:
        return
    if k <= 0:
        _gemm_core[False](m, n, 0, a, lda, b, ldb, c, ldc, acc)
        return
    if m < 8 or n < 8:
        _gemm_nt_dot(m, n, k, a, lda, b, ldb, c, ldc, acc)
        return
    var bt = List[Float32](length=k * n, fill=0.0)
    var pbt = bt.unsafe_ptr().as_unsafe_any_origin()
    transpose_into(n, k, b, ldb, pbt, n)
    _gemm_core[False](m, n, k, a, lda, pbt, n, c, ldc, acc)
    _ = len(bt)


# ------------------------------------------------------------------------------------------------ vector kernels


def add_bias_rows(x: FPtr, rows: Int, cols: Int, bias: FPtr):
    """x[i, :] += bias for every row (x is rows x cols, contiguous)."""
    var main = (cols // LANES) * LANES
    for i in range(rows):
        var base = i * cols
        var j = 0
        while j < main:
            x.unsafe_store(base + j, x.unsafe_load[width=LANES](base + j) + bias.unsafe_load[width=LANES](j))
            j += LANES
        while j < cols:
            x[unsafe_offset = base + j] = x[unsafe_offset = base + j] + bias[unsafe_offset=j]
            j += 1


def relu_inplace(x: FPtr, n: Int):
    """x = max(x, 0); a NaN stays NaN (torch)."""
    var zero = SIMD[_F32, LANES](0)
    var main = (n // LANES) * LANES
    var i = 0
    while i < main:
        var v = x.unsafe_load[width=LANES](i)
        x.unsafe_store(i, v.lt(zero).select(zero, v))
        i += LANES
    while i < n:
        if x[unsafe_offset=i] < 0.0:
            x[unsafe_offset=i] = 0.0
        i += 1


def relu_backward(dx: FPtr, pre: FPtr, n: Int):
    """dx *= (pre > 0): the gradient is 0 where the pre-activation is <= 0 (exactly 0 included, torch rule)."""
    var zero = SIMD[_F32, LANES](0)
    var main = (n // LANES) * LANES
    var i = 0
    while i < main:
        var pv = pre.unsafe_load[width=LANES](i)
        dx.unsafe_store(i, pv.le(zero).select(zero, dx.unsafe_load[width=LANES](i)))
        i += LANES
    while i < n:
        if pre[unsafe_offset=i] <= 0.0:
            dx[unsafe_offset=i] = 0.0
        i += 1


def col_sum_acc(g: FPtr, m: Int, n: Int, dst: FPtr):
    """dst[j] += sum_i g[i, j] (g is m x n contiguous). Rows are summed sequentially per column into a register
    accumulator that is added to `dst` once."""
    comptime W = 16
    var j = 0
    while j + W <= n:
        var s = SIMD[_F32, W](0)
        for i in range(m):
            s += g.unsafe_load[width=W](i * n + j)
        dst.unsafe_store(j, dst.unsafe_load[width=W](j) + s)
        j += W
    while j < n:
        var s: Float32 = 0.0
        for i in range(m):
            s += g[unsafe_offset = i * n + j]
        dst[unsafe_offset=j] = dst[unsafe_offset=j] + s
        j += 1


def saxpy(y: FPtr, x: FPtr, alpha: Float32, n: Int):
    """y += alpha * x (fused multiply-add)."""
    var av = SIMD[_F32, LANES](alpha)
    var main = (n // LANES) * LANES
    var i = 0
    while i < main:
        y.unsafe_store(i, fma(av, x.unsafe_load[width=LANES](i), y.unsafe_load[width=LANES](i)))
        i += LANES
    while i < n:
        y[unsafe_offset=i] = fma(alpha, x[unsafe_offset=i], y[unsafe_offset=i])
        i += 1


def add_inplace(dst: FPtr, src: FPtr, n: Int):
    """dst += src (exact adds; used to reduce the virtual-worker gradient buffers and for embedding row scatter)."""
    var main = (n // LANES) * LANES
    var i = 0
    while i < main:
        dst.unsafe_store(i, dst.unsafe_load[width=LANES](i) + src.unsafe_load[width=LANES](i))
        i += LANES
    while i < n:
        dst[unsafe_offset=i] = dst[unsafe_offset=i] + src[unsafe_offset=i]
        i += 1


def scale_inplace(x: FPtr, n: Int, s: Float32):
    """x *= s."""
    var sv = SIMD[_F32, LANES](s)
    var main = (n // LANES) * LANES
    var i = 0
    while i < main:
        x.unsafe_store(i, x.unsafe_load[width=LANES](i) * sv)
        i += LANES
    while i < n:
        x[unsafe_offset=i] = x[unsafe_offset=i] * s
        i += 1


def fill_f32(x: FPtr, n: Int, v: Float32):
    """x[0:n] = v."""
    var vv = SIMD[_F32, LANES](v)
    var main = (n // LANES) * LANES
    var i = 0
    while i < main:
        x.unsafe_store(i, vv)
        i += LANES
    while i < n:
        x[unsafe_offset=i] = v
        i += 1


def sumsq_f64(p: FPtr, n: Int) -> Float64:
    """Sum of squares accumulated in f64 (8 f64 lanes, fixed reduction order)."""
    var s = SIMD[DType.float64, LANES](0)
    var main = (n // LANES) * LANES
    var i = 0
    while i < main:
        var v = p.unsafe_load[width=LANES](i).cast[DType.float64]()
        s += v * v
        i += LANES
    var total = s.reduce_add()
    while i < n:
        var x = Float64(p[unsafe_offset=i])
        total += x * x
        i += 1
    return total
