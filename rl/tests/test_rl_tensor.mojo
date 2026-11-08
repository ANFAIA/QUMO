"""Tests for qrl.tensor: the three GEMM variants against the Float64 reference `gemm_ref` on random and edge
shapes (accumulate flag both ways, padded leading dimensions, 1-row and 1-col cases), tile independence (a sub-block
call is bit-identical to the same block of the full call), the vector kernels, and sentinel guard regions around
every buffer (a kernel that writes one float outside its documented region is caught).

Run once more with `-D ASSERT=all` in `make test-rl-slow` (MOJO_PORT.md 13.2).
"""

from std.memory import bitcast
from qrl.tensor import (
    FPtr,
    Mat,
    LANES,
    gemm_nn,
    gemm_nt,
    gemm_tn,
    transpose_into,
    add_bias_rows,
    relu_inplace,
    relu_backward,
    col_sum_acc,
    saxpy,
    add_inplace,
    scale_inplace,
    fill_f32,
    sumsq_f64,
)
from qrl.testutil import (
    GUARD,
    gemm_ref,
    rand_fill,
    guard_alloc,
    guard_ptr,
    guard_data,
    guard_set,
    guard_check,
    close,
    abs64,
)

comptime ATOL_BASE = 1e-6  # plus ATOL_PER_K * k: f32 accumulation error of a length-k dot product grows with k
comptime ATOL_PER_K = 1e-7
comptime RTOL = 1e-5


def _sz(rows: Int, cols: Int, ld: Int) -> Int:
    if rows <= 0 or cols <= 0:
        return 0
    return (rows - 1) * ld + cols


def _bits_equal(a: List[Float32], b: List[Float32]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            return False
    return True


def _variant_name(v: Int) -> String:
    if v == 0:
        return "gemm_nn"
    if v == 1:
        return "gemm_nt"
    return "gemm_tn"


def _call(v: Int, m: Int, n: Int, k: Int, a: FPtr, lda: Int, b: FPtr, ldb: Int, c: FPtr, ldc: Int, acc: Bool):
    if v == 0:
        gemm_nn(m, n, k, a, lda, b, ldb, c, ldc, acc)
    elif v == 1:
        gemm_nt(m, n, k, a, lda, b, ldb, c, ldc, acc)
    else:
        gemm_tn(m, n, k, a, lda, b, ldb, c, ldc, acc)


def check_gemm(v: Int, m: Int, n: Int, k: Int, acc: Bool, pad: Int, seed: UInt64) raises:
    """One GEMM call with guard regions around A, B and C, compared with `gemm_ref`."""
    var a_r = m
    var a_c = k
    var b_r = k
    var b_c = n
    var ta = False
    var tb = False
    if v == 1:
        b_r = n
        b_c = k
        tb = True
    elif v == 2:
        a_r = k
        a_c = m
        ta = True
    var lda = a_c + pad
    var ldb = b_c + pad
    var ldc = n + pad
    var na = _sz(a_r, a_c, lda)
    var nb = _sz(b_r, b_c, ldb)
    var nc = _sz(m, n, ldc)
    var what = _variant_name(v) + " m=" + String(m) + " n=" + String(n) + " k=" + String(k) + " acc=" + String(acc) + " pad=" + String(pad)

    var ga = guard_alloc(na)
    var gb = guard_alloc(nb)
    var gc = guard_alloc(nc)
    var da = List[Float32](length=na, fill=0.0)
    var db = List[Float32](length=nb, fill=0.0)
    var dc = List[Float32](length=nc, fill=0.0)
    rand_fill(da, seed)
    rand_fill(db, seed + 1)
    rand_fill(dc, seed + 2)  # initial C: accumulated into when acc, poison (overwritten / padding) otherwise
    guard_set(ga, da)
    guard_set(gb, db)
    guard_set(gc, dc)

    # Reference on private copies.
    var ref_c = dc.copy()
    var pra = da.unsafe_ptr().as_unsafe_any_origin()
    var prb = db.unsafe_ptr().as_unsafe_any_origin()
    var prc = ref_c.unsafe_ptr().as_unsafe_any_origin()
    gemm_ref(m, n, k, pra, lda, ta, prb, ldb, tb, prc, ldc, acc)

    _call(v, m, n, k, guard_ptr(ga), lda, guard_ptr(gb), ldb, guard_ptr(gc), ldc, acc)

    guard_check(ga, na, what + " (A)")
    guard_check(gb, nb, what + " (B)")
    guard_check(gc, nc, what + " (C)")
    if not _bits_equal(guard_data(ga, na), da):
        raise Error("A modified: " + what)
    if not _bits_equal(guard_data(gb, nb), db):
        raise Error("B modified: " + what)
    var got = guard_data(gc, nc)
    for i in range(m):
        for j in range(n):
            var o = i * ldc + j
            if not close(Float64(got[o]), Float64(ref_c[o]), ATOL_BASE + ATOL_PER_K * Float64(k), RTOL):
                raise Error("value mismatch " + what + " at (" + String(i) + "," + String(j) + "): " + String(got[o]) + " vs " + String(ref_c[o]))
        # the padding columns between rows must be untouched
        for j in range(n, min(ldc, nc - i * ldc)):
            var o = i * ldc + j
            if bitcast[DType.uint32](got[o]) != bitcast[DType.uint32](dc[o]):
                raise Error("padding of C modified: " + what + " at row " + String(i))
    _ = len(da) + len(db) + len(ref_c)


def test_gemm_shapes() raises:
    var ms: List[Int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 17]
    var ns: List[Int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17, 23, 31, 33, 39]
    var ks: List[Int] = [1, 2, 3, 8, 17, 29]
    var seed: UInt64 = 100
    var count = 0
    for v in range(3):
        for mi in range(len(ms)):
            for ni in range(len(ns)):
                for ki in range(len(ks)):
                    var acc = ((mi + ni + ki) % 2) == 0
                    var pad = ((mi * 7 + ni * 3 + ki) % 3) * 2  # 0, 2 or 4 padding floats per row
                    check_gemm(v, ms[mi], ns[ni], ks[ki], acc, pad, seed)
                    check_gemm(v, ms[mi], ns[ni], ks[ki], not acc, 0, seed + 7)
                    seed += 11
                    count += 2
    print("  gemm shape grid:", count, "calls")


def test_gemm_larger() raises:
    # Larger shapes that cross the 16/8/4/1 column tiles and the 4/2/1 row blocks several times, including the
    # actual layer shapes of the synthesis network (E = 512, C = 256, A = 10, value head n = 1).
    var seed: UInt64 = 5000
    for v in range(3):
        check_gemm(v, 64, 70, 50, False, 0, seed)
        check_gemm(v, 64, 70, 50, True, 3, seed + 1)
        check_gemm(v, 100, 130, 100, True, 0, seed + 2)
        check_gemm(v, 123, 37, 211, False, 1, seed + 3)
        check_gemm(v, 512, 256, 512, False, 0, seed + 4)
        check_gemm(v, 512, 10, 256, False, 0, seed + 5)
        check_gemm(v, 512, 1, 256, True, 0, seed + 6)
        check_gemm(v, 256, 512, 512, True, 0, seed + 7)
        check_gemm(v, 1, 513, 256, False, 0, seed + 8)
        check_gemm(v, 513, 1, 255, False, 0, seed + 9)
        seed += 20


def test_gemm_k_zero() raises:
    # k = 0: acc False zeroes C, acc True leaves it alone (all three variants).
    for v in range(3):
        for acc in range(2):
            var m = 5
            var n = 19
            var nc = m * n
            var gc = guard_alloc(nc)
            var dc = List[Float32](length=nc, fill=0.0)
            rand_fill(dc, 77)
            guard_set(gc, dc)
            var dummy = guard_alloc(4)
            var pd = guard_ptr(dummy)
            _call(v, m, n, 0, pd, 1, pd, 1, guard_ptr(gc), n, acc == 1)
            guard_check(gc, nc, _variant_name(v) + " k=0")
            var got = guard_data(gc, nc)
            for i in range(nc):
                var expect = dc[i] if acc == 1 else Float32(0.0)
                if got[i] != expect:
                    raise Error("k=0 mismatch for " + _variant_name(v) + " acc=" + String(acc))
            _ = len(dummy)
    # Empty outputs must not touch anything.
    var dummy2 = guard_alloc(8)
    var p2 = guard_ptr(dummy2)
    gemm_nn(0, 5, 3, p2, 3, p2, 5, p2, 5, False)
    gemm_nn(5, 0, 3, p2, 3, p2, 1, p2, 1, False)
    gemm_nt(0, 5, 3, p2, 3, p2, 3, p2, 5, True)
    gemm_tn(5, 0, 3, p2, 5, p2, 1, p2, 1, True)
    guard_check(dummy2, 8, "empty outputs")
    _ = len(dummy2)


def test_tile_independence() raises:
    # Every element of gemm_nn / gemm_tn is one sequential fma chain, so a call on a sub-block (different tile
    # positions, different edge tiles) must reproduce the full result bit for bit.
    var m = 21
    var n = 53
    var k = 37
    for v in range(2):  # nn and tn
        var a_r = m
        var a_c = k
        if v == 1:
            a_r = k
            a_c = m
        var da = List[Float32](length=a_r * a_c, fill=0.0)
        var db = List[Float32](length=k * n, fill=0.0)
        rand_fill(da, 31)
        rand_fill(db, 32)
        var full = List[Float32](length=m * n, fill=0.0)
        var pa = da.unsafe_ptr().as_unsafe_any_origin()
        var pb = db.unsafe_ptr().as_unsafe_any_origin()
        var pf = full.unsafe_ptr().as_unsafe_any_origin()
        _call(2 if v == 1 else 0, m, n, k, pa, a_c, pb, n, pf, n, False)
        var i0 = 3
        var j0 = 5
        var sm = 14
        var sn = 41
        var sub = List[Float32](length=sm * sn, fill=0.0)
        var ps = sub.unsafe_ptr().as_unsafe_any_origin()
        var pa_off = pa.unsafe_offset(i0 * a_c) if v == 0 else pa.unsafe_offset(i0)
        _call(2 if v == 1 else 0, sm, sn, k, pa_off, a_c, pb.unsafe_offset(j0), n, ps, sn, False)
        for i in range(sm):
            for j in range(sn):
                if bitcast[DType.uint32](sub[i * sn + j]) != bitcast[DType.uint32](full[(i0 + i) * n + (j0 + j)]):
                    raise Error("tile independence failed (variant " + String(v) + ") at (" + String(i) + "," + String(j) + ")")
        _ = len(da) + len(db) + len(full) + len(sub)


def test_transpose() raises:
    var shapes: List[Int] = [1, 2, 7, 8, 9, 16, 17, 33]
    for ri in range(len(shapes)):
        for ci in range(len(shapes)):
            var r = shapes[ri]
            var c = shapes[ci]
            var lds = c + 3
            var ldd = r + 2
            var ns = _sz(r, c, lds)
            var nd = _sz(c, r, ldd)
            var src = List[Float32](length=ns, fill=0.0)
            rand_fill(src, 9 + UInt64(r * 100 + c))
            var gs = guard_alloc(ns)
            var gd = guard_alloc(nd)
            guard_set(gs, src)
            transpose_into(r, c, guard_ptr(gs), lds, guard_ptr(gd), ldd)
            guard_check(gs, ns, "transpose src")
            guard_check(gd, nd, "transpose dst")
            var out = guard_data(gd, nd)
            for i in range(r):
                for j in range(c):
                    if out[j * ldd + i] != src[i * lds + j]:
                        raise Error("transpose_into wrong at (" + String(i) + "," + String(j) + ")")
            _ = len(src)


def test_vector_kernels() raises:
    var lens: List[Int] = [0, 1, 2, 7, 8, 9, 15, 16, 17, 31, 40]
    for li in range(len(lens)):
        var n = lens[li]
        # relu_inplace / relu_backward
        var x = List[Float32](length=n, fill=0.0)
        var dxv = List[Float32](length=n, fill=0.0)
        rand_fill(x, 500 + UInt64(n))
        rand_fill(dxv, 600 + UInt64(n))
        if n > 3:
            x[1] = 0.0
            x[2] = -0.0
        var gx = guard_alloc(n)
        guard_set(gx, x)
        relu_inplace(guard_ptr(gx), n)
        guard_check(gx, n, "relu_inplace")
        var rx = guard_data(gx, n)
        for i in range(n):
            var e = x[i] if x[i] > 0.0 else Float32(0.0)
            if rx[i] != e:
                raise Error("relu_inplace wrong at " + String(i))
        # relu_backward: gradient zero where pre <= 0, including exactly 0
        var gp = guard_alloc(n)
        var gd = guard_alloc(n)
        guard_set(gp, x)
        guard_set(gd, dxv)
        relu_backward(guard_ptr(gd), guard_ptr(gp), n)
        guard_check(gp, n, "relu_backward pre")
        guard_check(gd, n, "relu_backward dx")
        var rd = guard_data(gd, n)
        for i in range(n):
            var e = dxv[i] if x[i] > 0.0 else Float32(0.0)
            if rd[i] != e:
                raise Error("relu_backward wrong at " + String(i) + " n=" + String(n))
        if not _bits_equal(guard_data(gp, n), x):
            raise Error("relu_backward modified pre")

        # saxpy, add_inplace, scale_inplace, fill_f32
        var y = List[Float32](length=n, fill=0.0)
        var z = List[Float32](length=n, fill=0.0)
        rand_fill(y, 700 + UInt64(n))
        rand_fill(z, 800 + UInt64(n))
        var gy = guard_alloc(n)
        var gz = guard_alloc(n)
        guard_set(gy, y)
        guard_set(gz, z)
        saxpy(guard_ptr(gy), guard_ptr(gz), 0.37, n)
        guard_check(gy, n, "saxpy y")
        guard_check(gz, n, "saxpy x")
        var ry = guard_data(gy, n)
        for i in range(n):
            if not close(Float64(ry[i]), Float64(y[i]) + 0.37 * Float64(z[i]), 1e-6, 1e-6):
                raise Error("saxpy wrong at " + String(i))
        guard_set(gy, y)
        add_inplace(guard_ptr(gy), guard_ptr(gz), n)
        guard_check(gy, n, "add_inplace y")
        guard_check(gz, n, "add_inplace x")
        ry = guard_data(gy, n)
        for i in range(n):
            if ry[i] != y[i] + z[i]:
                raise Error("add_inplace wrong at " + String(i))
        guard_set(gy, y)
        scale_inplace(guard_ptr(gy), n, 0.25)
        guard_check(gy, n, "scale_inplace")
        ry = guard_data(gy, n)
        for i in range(n):
            if ry[i] != y[i] * 0.25:
                raise Error("scale_inplace wrong at " + String(i))
        fill_f32(guard_ptr(gy), n, 3.5)
        guard_check(gy, n, "fill_f32")
        ry = guard_data(gy, n)
        for i in range(n):
            if ry[i] != 3.5:
                raise Error("fill_f32 wrong at " + String(i))

        # sumsq_f64 against a sequential f64 sum
        var gs = guard_alloc(n)
        guard_set(gs, x)
        var ss = sumsq_f64(guard_ptr(gs), n)
        guard_check(gs, n, "sumsq_f64")
        var refss: Float64 = 0.0
        for i in range(n):
            refss += Float64(x[i]) * Float64(x[i])
        if abs64(ss - refss) > 1e-13 * (1.0 + refss):
            raise Error("sumsq_f64 wrong for n=" + String(n) + ": " + String(ss) + " vs " + String(refss))
        _ = len(x) + len(dxv) + len(y) + len(z)

    # add_bias_rows and col_sum_acc on several (rows, cols)
    var rs: List[Int] = [0, 1, 3, 8, 17]
    var cs: List[Int] = [1, 5, 8, 16, 17, 33, 50]
    for ri in range(len(rs)):
        for ci in range(len(cs)):
            var rows = rs[ri]
            var cols = cs[ci]
            var nn = rows * cols
            var base = List[Float32](length=nn, fill=0.0)
            var bias = List[Float32](length=cols, fill=0.0)
            var outv = List[Float32](length=cols, fill=0.0)
            rand_fill(base, 900 + UInt64(nn))
            rand_fill(bias, 901 + UInt64(cols))
            rand_fill(outv, 902 + UInt64(cols))
            var gb = guard_alloc(nn)
            var gbias = guard_alloc(cols)
            var gout = guard_alloc(cols)
            guard_set(gb, base)
            guard_set(gbias, bias)
            guard_set(gout, outv)
            add_bias_rows(guard_ptr(gb), rows, cols, guard_ptr(gbias))
            guard_check(gb, nn, "add_bias_rows x")
            guard_check(gbias, cols, "add_bias_rows bias")
            var rb = guard_data(gb, nn)
            for i in range(rows):
                for j in range(cols):
                    if rb[i * cols + j] != base[i * cols + j] + bias[j]:
                        raise Error("add_bias_rows wrong")
            guard_set(gb, base)
            col_sum_acc(guard_ptr(gb), rows, cols, guard_ptr(gout))
            guard_check(gb, nn, "col_sum_acc g")
            guard_check(gout, cols, "col_sum_acc out")
            var ro = guard_data(gout, cols)
            for j in range(cols):
                var s: Float64 = 0.0
                for i in range(rows):
                    s += Float64(base[i * cols + j])
                if not close(Float64(ro[j]), Float64(outv[j]) + s, 1e-5, 1e-5):
                    raise Error("col_sum_acc wrong at col " + String(j))
            _ = len(base) + len(bias) + len(outv)


def test_relu_nan_semantics() raises:
    # torch: relu(NaN) = NaN; the ReLU backward lets the gradient through where the pre-activation is NaN
    # (threshold_backward masks only x <= 0).
    var nan = Float32.MAX * 2.0 - Float32.MAX * 2.0  # inf - inf = NaN
    var g = guard_alloc(9)
    var d: List[Float32] = [nan, -1.0, 0.0, 2.0, nan, -0.0, 5.0, -5.0, 1.0]
    guard_set(g, d)
    relu_inplace(guard_ptr(g), 9)
    var r = guard_data(g, 9)
    if not (r[0] != r[0]):
        raise Error("relu_inplace must keep NaN")
    if r[1] != 0.0 or r[3] != 2.0 or r[6] != 5.0 or r[7] != 0.0:
        raise Error("relu_inplace values")
    var gp = guard_alloc(9)
    var gdx = guard_alloc(9)
    guard_set(gp, d)
    var ones = List[Float32](length=9, fill=1.0)
    guard_set(gdx, ones)
    relu_backward(guard_ptr(gdx), guard_ptr(gp), 9)
    var rd = guard_data(gdx, 9)
    # pre: NaN, -1, 0, 2, NaN, -0, 5, -5, 1  ->  1, 0, 0, 1, 1, 0, 1, 0, 1
    var expect: List[Float32] = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    for i in range(9):
        if rd[i] != expect[i]:
            raise Error("relu_backward NaN/zero semantics wrong at " + String(i))
    _ = len(d) + len(ones)


def test_mat() raises:
    var mm = Mat(3, 5)
    if mm.rows != 3 or mm.cols != 5 or len(mm.data) != 15:
        raise Error("Mat shape")
    var p = mm.ptr()
    p[unsafe_offset=7] = 2.5
    if mm.data[7] != 2.5:
        raise Error("Mat.ptr does not alias data")
    _ = len(mm.data)


def main() raises:
    test_mat()
    test_gemm_shapes()
    test_gemm_larger()
    test_gemm_k_zero()
    test_tile_independence()
    test_transpose()
    test_vector_kernels()
    test_relu_nan_semantics()
    print("test_rl_tensor OK")
