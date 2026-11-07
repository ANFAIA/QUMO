"""Test helpers for the native RL stack: fixture loading, tolerances, statistics, reference GEMM, guard regions.

Used by `tests/test_rl_*.mojo` only (never imported by library code). Tests run from the repository root
(`mojo run -I . tests/test_rl_x.mojo`), so fixtures are found under `tests/fixtures/rl/`; set the environment
variable `QUMO_RL_FIXTURES` to use another root.
"""

from std.collections import List
from std.math import sqrt, cbrt
from std.memory import bitcast
from std.os import getenv
from qumo.ai.json import JsonValue, parse_json

comptime FPtr = MutPointer[Float32, MutAnyOrigin]

comptime DEFAULT_FIXTURE_ROOT = "tests/fixtures/rl"


# ------------------------------------------------------------------------------------------------ fixture loading


def fixture_root() -> String:
    var env = getenv("QUMO_RL_FIXTURES")
    if env.byte_length() > 0:
        return env^
    return String(DEFAULT_FIXTURE_ROOT)


def fixture_path(rel: String) -> String:
    return fixture_root() + "/" + rel


def read_text(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def read_bytes(path: String) raises -> List[UInt8]:
    with open(path, "r") as f:
        return f.read_bytes()


def load_fixture_text(rel: String) raises -> String:
    return read_text(fixture_path(rel))


def load_fixture(rel: String) raises -> JsonValue:
    """Parses `tests/fixtures/rl/<rel>` with `qumo.ai.json`."""
    return parse_json(read_text(fixture_path(rel)))


def load_fixture_bytes(rel: String) raises -> List[UInt8]:
    return read_bytes(fixture_path(rel))


def fixture_size(rel: String) raises -> Int:
    """Size in bytes of a fixture file."""
    return len(read_bytes(fixture_path(rel)))


# ------------------------------------------------------------------------------------------------ JSON conversions


def ints(v: JsonValue) raises -> List[Int]:
    """JSON array of integers -> List[Int]."""
    var out = List[Int](capacity=len(v))
    for i in range(len(v)):
        out.append(v.at(i).as_int())
    return out^


def floats(v: JsonValue) raises -> List[Float64]:
    """JSON array of numbers -> List[Float64]."""
    var out = List[Float64](capacity=len(v))
    for i in range(len(v)):
        out.append(v.at(i).as_float())
    return out^


def floats32(v: JsonValue) raises -> List[Float32]:
    """JSON array of numbers -> List[Float32] (the f32 rounding of the parsed double)."""
    var out = List[Float32](capacity=len(v))
    for i in range(len(v)):
        out.append(Float32(v.at(i).as_float()))
    return out^


def strings(v: JsonValue) raises -> List[String]:
    var out = List[String](capacity=len(v))
    for i in range(len(v)):
        out.append(v.at(i).as_string())
    return out^


def ints2(v: JsonValue) raises -> List[List[Int]]:
    """JSON array of integer arrays -> List[List[Int]]."""
    var out = List[List[Int]](capacity=len(v))
    for i in range(len(v)):
        out.append(ints(v.at(i)))
    return out^


def floats2(v: JsonValue) raises -> List[List[Float64]]:
    var out = List[List[Float64]](capacity=len(v))
    for i in range(len(v)):
        out.append(floats(v.at(i)))
    return out^


def _flatten_into(v: JsonValue, mut out: List[Float64]) raises:
    if v.is_array():
        for i in range(len(v)):
            _flatten_into(v.at(i), out)
    else:
        out.append(v.as_float())


def flat_floats(v: JsonValue) raises -> List[Float64]:
    """Flattens arbitrarily nested JSON arrays of numbers, row-major."""
    var out = List[Float64]()
    _flatten_into(v, out)
    return out^


def flat_floats32(v: JsonValue) raises -> List[Float32]:
    var d = flat_floats(v)
    var out = List[Float32](capacity=len(d))
    for i in range(len(d)):
        out.append(Float32(d[i]))
    return out^


# ------------------------------------------------------------------------------------------------ tolerances


def abs64(x: Float64) -> Float64:
    if x < 0.0:
        return -x
    return x


def close(a: Float64, b: Float64, atol: Float64, rtol: Float64) -> Bool:
    """`|a - b| <= atol + rtol * |b|`. NaN is never close; equal infinities are."""
    if a == b:
        return True
    var d = abs64(a - b)
    if d != d:
        return False
    return d <= atol + rtol * abs64(b)


def max_abs_diff(a: List[Float64], b: List[Float64]) -> Float64:
    """Largest elementwise `|a - b|` (NaN counts as infinity); lengths must agree."""
    if len(a) != len(b):
        return Float64.MAX
    var m: Float64 = 0.0
    for i in range(len(a)):
        var d = abs64(a[i] - b[i])
        if d != d:
            return Float64.MAX
        if d > m:
            m = d
    return m


def max_abs_diff32(a: List[Float32], b: List[Float32]) -> Float64:
    if len(a) != len(b):
        return Float64.MAX
    var m: Float64 = 0.0
    for i in range(len(a)):
        var d = abs64(Float64(a[i]) - Float64(b[i]))
        if d != d:
            return Float64.MAX
        if d > m:
            m = d
    return m


def max_rel_diff32(a: List[Float32], b: List[Float32], floor: Float64 = 1e-12) -> Float64:
    """Largest `|a - b| / max(|b|, floor)`."""
    if len(a) != len(b):
        return Float64.MAX
    var m: Float64 = 0.0
    for i in range(len(a)):
        var den = abs64(Float64(b[i]))
        if den < floor:
            den = floor
        var d = abs64(Float64(a[i]) - Float64(b[i])) / den
        if d != d:
            return Float64.MAX
        if d > m:
            m = d
    return m


def assert_close(a: Float64, b: Float64, atol: Float64, rtol: Float64, what: String) raises:
    if not close(a, b, atol, rtol):
        raise Error(
            "assert_close failed: " + what + ": " + String(a) + " vs " + String(b) + " (atol " + String(atol) + ", rtol " + String(rtol) + ")"
        )


def assert_list_close(a: List[Float64], b: List[Float64], atol: Float64, rtol: Float64, what: String) raises:
    if len(a) != len(b):
        raise Error("assert_list_close failed: " + what + ": length " + String(len(a)) + " vs " + String(len(b)))
    for i in range(len(a)):
        if not close(a[i], b[i], atol, rtol):
            raise Error(
                "assert_list_close failed: " + what + "[" + String(i) + "]: " + String(a[i]) + " vs " + String(b[i])
                + " (atol " + String(atol) + ", rtol " + String(rtol) + ")"
            )


def assert_list_close32(a: List[Float32], b: List[Float32], atol: Float64, rtol: Float64, what: String) raises:
    if len(a) != len(b):
        raise Error("assert_list_close32 failed: " + what + ": length " + String(len(a)) + " vs " + String(len(b)))
    for i in range(len(a)):
        if not close(Float64(a[i]), Float64(b[i]), atol, rtol):
            raise Error(
                "assert_list_close32 failed: " + what + "[" + String(i) + "]: " + String(a[i]) + " vs " + String(b[i])
                + " (atol " + String(atol) + ", rtol " + String(rtol) + ")"
            )


def assert_ints_equal(a: List[Int], b: List[Int], what: String) raises:
    if len(a) != len(b):
        raise Error("assert_ints_equal failed: " + what + ": length " + String(len(a)) + " vs " + String(len(b)))
    for i in range(len(a)):
        if a[i] != b[i]:
            raise Error("assert_ints_equal failed: " + what + "[" + String(i) + "]: " + String(a[i]) + " vs " + String(b[i]))


def f32_bits(x: Float32) -> UInt32:
    return bitcast[DType.uint32](x)


# ------------------------------------------------------------------------------------------------ statistics


def two_sample_z(m1: Float64, se1: Float64, m2: Float64, se2: Float64) -> Float64:
    """`(m1 - m2) / sqrt(se1^2 + se2^2)`; 0 when both standard errors vanish and the means agree."""
    var den = sqrt(se1 * se1 + se2 * se2)
    if den == 0.0:
        if m1 == m2:
            return 0.0
        return Float64.MAX
    return (m1 - m2) / den


def cluster_mean_se(sums: List[Float64], counts: List[Float64]) -> Tuple[Float64, Float64]:
    """Ratio estimator over clusters (episodes): mean = sum S / sum L,
    se = sqrt(sum (S_e - mean * L_e)^2) / sum L. Same estimator as `ppo/rollout_stats.json`."""
    var tot: Float64 = 0.0
    var ssum: Float64 = 0.0
    for i in range(len(sums)):
        tot += counts[i]
        ssum += sums[i]
    if tot == 0.0:
        return (0.0, 0.0)
    var m = ssum / tot
    var acc: Float64 = 0.0
    for i in range(len(sums)):
        var d = sums[i] - m * counts[i]
        acc += d * d
    return (m, sqrt(acc) / tot)


struct Chi2(Copyable):
    """Pearson chi-square result: statistic, degrees of freedom, and the Wilson-Hilferty z-score."""

    var stat: Float64
    var dof: Int
    var z: Float64

    def __init__(out self, stat: Float64, dof: Int):
        self.stat = stat
        self.dof = dof
        self.z = chi_square_z(stat, dof)


def chi_square_z(stat: Float64, dof: Int) -> Float64:
    """Standard-normal score of a chi-square statistic (Wilson-Hilferty cube-root approximation).
    A statistic at the mean has z about 0; a test passes at `z <= 4.0` (the one-sided 4-sigma rule of the plan)."""
    if dof <= 0:
        return 0.0
    var k = Float64(dof)
    var t = 2.0 / (9.0 * k)
    return (cbrt(stat / k) - (1.0 - t)) / sqrt(t)


def chi_square_gof(observed: List[Int], probs: List[Float64], min_expected: Float64 = 5.0) raises -> Chi2:
    """Goodness of fit of `observed` counts against cell probabilities `probs` (sum to 1).
    Cells with expected count below `min_expected` are pooled into one rest cell (merged with the smallest
    remaining cell if the rest is still below the threshold). dof = pooled cells - 1."""
    if len(observed) != len(probs):
        raise Error("chi_square_gof: length mismatch")
    var n: Float64 = 0.0
    for i in range(len(observed)):
        n += Float64(observed[i])
    var obs = List[Float64]()
    var exp = List[Float64]()
    var rest_o: Float64 = 0.0
    var rest_e: Float64 = 0.0
    for i in range(len(observed)):
        var e = probs[i] * n
        if e >= min_expected:
            obs.append(Float64(observed[i]))
            exp.append(e)
        else:
            rest_o += Float64(observed[i])
            rest_e += e
    if rest_e > 0.0 or rest_o > 0.0:
        if rest_e >= min_expected or len(exp) == 0:
            obs.append(rest_o)
            exp.append(rest_e)
        else:
            # merge into the smallest regular cell
            var j = 0
            for i in range(len(exp)):
                if exp[i] < exp[j]:
                    j = i
            obs[j] += rest_o
            exp[j] += rest_e
    var stat: Float64 = 0.0
    for i in range(len(exp)):
        if exp[i] > 0.0:
            var d = obs[i] - exp[i]
            stat += d * d / exp[i]
    return Chi2(stat, len(exp) - 1)


def chi_square_homogeneity(a: List[Int], b: List[Int], min_expected: Float64 = 5.0) raises -> Chi2:
    """Homogeneity of two count histograms over the same cells. Cells where either sample's expected count is below
    `min_expected` are pooled into one rest cell. dof = pooled cells - 1."""
    if len(a) != len(b):
        raise Error("chi_square_homogeneity: length mismatch")
    var na: Float64 = 0.0
    var nb: Float64 = 0.0
    for i in range(len(a)):
        na += Float64(a[i])
        nb += Float64(b[i])
    var ca = List[Float64]()
    var cb = List[Float64]()
    var ra: Float64 = 0.0
    var rb: Float64 = 0.0
    for i in range(len(a)):
        var tot = Float64(a[i] + b[i])
        var ea = tot * na / (na + nb)
        var eb = tot * nb / (na + nb)
        if ea >= min_expected and eb >= min_expected:
            ca.append(Float64(a[i]))
            cb.append(Float64(b[i]))
        else:
            ra += Float64(a[i])
            rb += Float64(b[i])
    if ra + rb > 0.0:
        var ea = (ra + rb) * na / (na + nb)
        var eb = (ra + rb) * nb / (na + nb)
        if (ea >= min_expected and eb >= min_expected) or len(ca) == 0:
            ca.append(ra)
            cb.append(rb)
        else:
            # merge into the cell with the fewest observations
            var j = 0
            for i in range(len(ca)):
                if ca[i] + cb[i] < ca[j] + cb[j]:
                    j = i
            ca[j] += ra
            cb[j] += rb
    var stat: Float64 = 0.0
    for i in range(len(ca)):
        var tot = ca[i] + cb[i]
        if tot > 0.0:
            var ea = tot * na / (na + nb)
            var eb = tot * nb / (na + nb)
            var da = ca[i] - ea
            var db = cb[i] - eb
            stat += da * da / ea + db * db / eb
    return Chi2(stat, len(ca) - 1)


def ecdf_at(sorted_vals: List[Float64], x: Float64) -> Float64:
    """Fraction of `sorted_vals` (ascending) that are `<= x`."""
    var lo = 0
    var hi = len(sorted_vals)
    while lo < hi:
        var mid = (lo + hi) // 2
        if sorted_vals[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    if len(sorted_vals) == 0:
        return 0.0
    return Float64(lo) / Float64(len(sorted_vals))


def ks_against_knots(sample: List[Float64], knots_x: List[Float64], knots_f: List[Float64]) -> Float64:
    """`max_k |F_sample(x_k) - F_k|` for a reference ECDF given at knots (`route/jobs_stats.json` form)."""
    var s = sample.copy()
    sort(s)
    var d: Float64 = 0.0
    for k in range(len(knots_x)):
        var g = abs64(ecdf_at(s, knots_x[k]) - knots_f[k])
        if g > d:
            d = g
    return d


def ks_critical(n_ref: Int, n_new: Int, c: Float64 = 2.28) -> Float64:
    """Two-sample KS critical value `c * sqrt(1/n_ref + 1/n_new)`; `c = 2.28` is about 4 sigma (alpha 6e-5)."""
    return c * sqrt(1.0 / Float64(n_ref) + 1.0 / Float64(n_new))


# ------------------------------------------------------------------------------------------------ reference GEMM


def gemm_ref(
    m: Int,
    n: Int,
    k: Int,
    a: FPtr,
    lda: Int,
    ta: Bool,
    b: FPtr,
    ldb: Int,
    tb: Bool,
    c: FPtr,
    ldc: Int,
    acc: Bool,
):
    """Naive triple loop with Float64 accumulation: `C[m,n] (+)= op(A)[m,k] . op(B)[k,n]`.

    `ta = False`: A is stored `[m, k]` (element `a[i * lda + p]`); `ta = True`: stored `[k, m]` (`a[p * lda + i]`).
    `tb = False`: B is stored `[k, n]` (`b[p * ldb + j]`); `tb = True`: stored `[n, k]` (`b[j * ldb + p]`).
    The three production variants are `gemm_nt` (tb), `gemm_nn` (neither) and `gemm_tn` (ta)."""
    for i in range(m):
        for j in range(n):
            var s: Float64 = 0.0
            for p in range(k):
                var av: Float32
                if ta:
                    av = a[unsafe_offset = p * lda + i]
                else:
                    av = a[unsafe_offset = i * lda + p]
                var bv: Float32
                if tb:
                    bv = b[unsafe_offset = j * ldb + p]
                else:
                    bv = b[unsafe_offset = p * ldb + j]
                s += Float64(av) * Float64(bv)
            if acc:
                c[unsafe_offset = i * ldc + j] = Float32(Float64(c[unsafe_offset = i * ldc + j]) + s)
            else:
                c[unsafe_offset = i * ldc + j] = Float32(s)


def rand_fill(mut v: List[Float32], seed: UInt64, lo: Float32 = -1.0, hi: Float32 = 1.0):
    """Fills `v` with deterministic uniform values in `[lo, hi)` (xoshiro `Rng`)."""
    from qumo.sim.rng import Rng

    var r = Rng(seed)
    for i in range(len(v)):
        v[i] = lo + Float32(r.uniform()) * (hi - lo)


# ------------------------------------------------------------------------------------------------ guard regions

comptime GUARD = 64
comptime _SENTINEL_BITS: UInt32 = 0xBEEFCAFE


def guard_alloc(n: Int, fill: Float32 = 0.0) -> List[Float32]:
    """Allocates `GUARD + n + GUARD` floats: `n` payload values (set to `fill`) between two sentinel regions.
    Pass `guard_ptr(buf)` to the kernel under test and call `guard_check(buf, n, ...)` afterwards."""
    var sentinel = bitcast[DType.float32](_SENTINEL_BITS)
    var buf = List[Float32](length=GUARD + n + GUARD, fill=sentinel)
    for i in range(n):
        buf[GUARD + i] = fill
    return buf^


def guard_ptr(mut buf: List[Float32]) -> FPtr:
    """Pointer to the first payload element of a guarded buffer. Keep `buf` alive (e.g. `_ = len(buf)`) after the call."""
    return buf.unsafe_ptr().as_unsafe_any_origin().unsafe_offset(GUARD)


def guard_data(buf: List[Float32], n: Int) -> List[Float32]:
    """Copy of the `n` payload values."""
    var out = List[Float32](capacity=n)
    for i in range(n):
        out.append(buf[GUARD + i])
    return out^


def guard_set(mut buf: List[Float32], data: List[Float32]):
    """Writes `data` into the payload (does not touch the guards)."""
    for i in range(len(data)):
        buf[GUARD + i] = data[i]


def guard_check(buf: List[Float32], n: Int, what: String) raises:
    """Raises if either guard region of a `guard_alloc(n)` buffer changed (compared bit for bit)."""
    for i in range(GUARD):
        if bitcast[DType.uint32](buf[i]) != _SENTINEL_BITS:
            raise Error("guard_check failed: " + what + ": low guard overwritten at offset " + String(i - GUARD))
        if bitcast[DType.uint32](buf[GUARD + n + i]) != _SENTINEL_BITS:
            raise Error("guard_check failed: " + what + ": high guard overwritten at offset " + String(n + i))
