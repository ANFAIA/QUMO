"""Tests for qumo.algorithms.trotter_error (first-order Trotter bound and estimate).

The reference is dense linear algebra written here: the Trotter product is the
product of the exact single-term exponentials cos(tau c) I - i sin(tau c) P, the
exact evolution is a scaled Taylor series with squaring, and the spectral norm
of their difference is the square root of the largest eigenvalue of E^dag E.
Nothing here calls trotter_error to build its own reference.
"""

from std.math import sqrt, cos, sin
from std.testing import assert_true, assert_almost_equal, assert_equal
from qumo.core.rng import Rng
from qumo.ir.pauli import PauliTerm, PauliSum, pauli_half_commutator
from qumo.ir.circuit import Circuit
from qumo.sim.statevector import StateVector
from qumo.sim.runner import evolve_into
from qumo.algorithms.hamiltonians import dense_matrix
from qumo.algorithms.linalg import jacobi_eigh
from qumo.algorithms.library import trotter
from qumo.algorithms.trotter_error import trotter_error, trotter_commutators, TrotterError


# ---------------------------------------------------------------------------
# Dense complex matrices (row-major, separate re / im)
# ---------------------------------------------------------------------------
struct CMat(Copyable, Movable):
    var re: List[Float64]
    var im: List[Float64]
    var d: Int

    def __init__(out self, d: Int):
        self.d = d
        self.re = List[Float64](length=d * d, fill=0.0)
        self.im = List[Float64](length=d * d, fill=0.0)

    @staticmethod
    def identity(d: Int) -> CMat:
        var m = CMat(d)
        for i in range(d):
            m.re[i * d + i] = 1.0
        return m^

    @staticmethod
    def of(h: PauliSum, n: Int) raises -> CMat:
        var mats = dense_matrix(h, n)
        var m = CMat(1 << n)
        m.re = mats[0].copy()
        m.im = mats[1].copy()
        return m^


def matmul(a: CMat, b: CMat) -> CMat:
    var d = a.d
    var c = CMat(d)
    for i in range(d):
        for k in range(d):
            var ar = a.re[i * d + k]
            var ai = a.im[i * d + k]
            if ar == 0.0 and ai == 0.0:
                continue
            for j in range(d):
                var br = b.re[k * d + j]
                var bi = b.im[k * d + j]
                c.re[i * d + j] += ar * br - ai * bi
                c.im[i * d + j] += ar * bi + ai * br
    return c^


def axpy(a: CMat, sr: Float64, si: Float64, b: CMat) -> CMat:
    """Return a + (sr + i si) b."""
    var c = a.copy()
    for k in range(a.d * a.d):
        c.re[k] += sr * b.re[k] - si * b.im[k]
        c.im[k] += sr * b.im[k] + si * b.re[k]
    return c^


def spectral_norm(e: CMat) -> Float64:
    """Spectral norm sqrt(lambda_max(E^dag E)) through the real symmetric embedding of E^dag E."""
    var d = e.d
    var g = CMat(d)  # E^dag E
    for i in range(d):
        for j in range(d):
            var sr = 0.0
            var si = 0.0
            for k in range(d):
                var ar = e.re[k * d + i]
                var ai = -e.im[k * d + i]
                var br = e.re[k * d + j]
                var bi = e.im[k * d + j]
                sr += ar * br - ai * bi
                si += ar * bi + ai * br
            g.re[i * d + j] = sr
            g.im[i * d + j] = si
    var m = 2 * d
    var big = List[Float64](length=m * m, fill=0.0)
    for r in range(d):
        for c in range(d):
            var a = 0.5 * (g.re[r * d + c] + g.re[c * d + r])
            var b = 0.5 * (g.im[r * d + c] - g.im[c * d + r])
            big[r * m + c] = a
            big[(r + d) * m + c + d] = a
            big[r * m + c + d] = -b
            big[(r + d) * m + c] = b
    var ev = jacobi_eigh(big, m).values.copy()
    return sqrt(max(ev[m - 1], 0.0))


def hermitian_norm(a: CMat) -> Float64:
    return spectral_norm(a)


def expm_minus_i(h: CMat, t: Float64, lam: Float64) -> CMat:
    """Return exp(-i H t) by Taylor series on t / 2^s (||H t|| / 2^s <= 0.25), then squaring."""
    var s = 0
    var x = abs(t) * lam
    while x > 0.25:
        x *= 0.5
        s += 1
    var dt = t / Float64(1 << s)
    var acc = CMat.identity(h.d)
    var term = CMat.identity(h.d)
    for k in range(1, 40):
        var nt = matmul(h, term)
        # term <- (-i dt / k) H term
        var z = CMat(h.d)
        term = axpy(z, 0.0, -dt / Float64(k), nt)
        acc = axpy(acc, 1.0, 0.0, term)
    for _ in range(s):
        acc = matmul(acc, acc)
    return acc^


def trotter_dense(h: PauliSum, n: Int, t: Float64, r: Int) raises -> CMat:
    """(e^{-i tau H_m} ... e^{-i tau H_1})^r with exact single-Pauli exponentials, terms[0] first."""
    var d = 1 << n
    var tau = t / Float64(r)
    var step = CMat.identity(d)
    for term in h.terms:
        var unit = PauliSum()
        unit.add(1.0, term.x, term.z)
        var p = CMat.of(unit, n)
        var th = tau * term.coeff
        var e = CMat.identity(d)
        for k in range(d * d):
            e.re[k] = cos(th) * e.re[k]
        e = axpy(e, 0.0, -sin(th), p)
        step = matmul(e, step)
    var out = CMat.identity(d)
    for _ in range(r):
        out = matmul(step, out)
    return out^


def exact_error(h: PauliSum, n: Int, t: Float64, r: Int) raises -> Float64:
    var lam = 0.0
    for term in h.terms:
        lam += abs(term.coeff)
    var u = expm_minus_i(CMat.of(h, n), t, lam)
    var s = trotter_dense(h, n, t, r)
    return spectral_norm(axpy(u, -1.0, 0.0, s))


def random_sum(mut rng: Rng, n: Int, m: Int) -> PauliSum:
    """Return m distinct non-identity Pauli strings on n qubits, Gaussian coefficients."""
    var h = PauliSum()
    var full = (UInt64(1) << UInt64(n)) - 1
    while len(h.terms) < m:
        var x = rng.next_u64() & full
        var z = rng.next_u64() & full
        if x == 0 and z == 0:
            continue
        var dup = False
        for t in h.terms:
            if t.x == x and t.z == z:
                dup = True
        if dup:
            continue
        h.terms.append(PauliTerm(rng.normal(), x, z))
    return h^


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_half_commutator_dense() raises:
    """-(i/2)[aP, bQ] against dense products: all 16 single-qubit pairs and random 3-qubit strings."""
    var rng = Rng(11)
    var cases = List[Tuple[PauliTerm, PauliTerm, Int]]()
    for i in range(4):
        for j in range(4):
            var xi = UInt64(1 if (i == 1 or i == 2) else 0)
            var zi = UInt64(1 if (i == 2 or i == 3) else 0)
            var xj = UInt64(1 if (j == 1 or j == 2) else 0)
            var zj = UInt64(1 if (j == 2 or j == 3) else 0)
            cases.append((PauliTerm(0.7, xi, zi), PauliTerm(-1.3, xj, zj), 1))
    for _ in range(60):
        var a = PauliTerm(rng.normal(), rng.next_u64() & 7, rng.next_u64() & 7)
        var b = PauliTerm(rng.normal(), rng.next_u64() & 7, rng.next_u64() & 7)
        cases.append((a, b, 3))
    for ref cs in cases:
        var n = cs[2]
        var sa = PauliSum()
        sa.terms.append(cs[0])
        var sb = PauliSum()
        sb.terms.append(cs[1])
        var a = CMat.of(sa, n)
        var b = CMat.of(sb, n)
        var comm = axpy(matmul(a, b), -1.0, 0.0, matmul(b, a))
        var want = axpy(CMat(a.d), 0.0, -0.5, comm)  # -(i/2)[A, B]
        var got_t = pauli_half_commutator(cs[0], cs[1])
        var sg = PauliSum()
        sg.terms.append(got_t)
        var got = CMat.of(sg, n)
        for k in range(a.d * a.d):
            assert_almost_equal(got.re[k], want.re[k], atol=1e-13)
            assert_almost_equal(got.im[k], want.im[k], atol=1e-13)


def test_commuting_terms_give_zero() raises:
    var h = PauliSum()
    h.add_label(0.9, "ZZI")
    h.add_label(-0.4, "IZZ")
    h.add_label(0.3, "XXX")  # commutes with both ZZ strings
    h.add_label(1.1, "ZIZ")
    for which in range(2):
        var e = trotter_error(h, 2.0, 3, norm="exact" if which == 0 else "l1")
        assert_equal(e.bound, 0.0)
        assert_equal(e.estimate, 0.0)
    var e = trotter_error(h, 2.0, 3)
    assert_equal(e.norm, String("exact"))
    assert_true(exact_error(h, 3, 2.0, 3) < 1e-12)


def test_two_terms_by_hand() raises:
    """H = a X0 + b Z0: -(i/2)[bZ, aX] = a b Y, so alpha1 = ||Phi2|| = a b and bound = a b t^2 / r."""
    var a = 0.7
    var b = 0.4
    var h = PauliSum()
    h.add(a, 1, 0)
    h.add(b, 0, 1)
    for which in range(2):
        var e = trotter_error(h, 1.5, 4, norm="exact" if which == 0 else "l1")
        assert_almost_equal(e.alpha1, a * b, atol=1e-11)
        assert_almost_equal(e.phi2, a * b, atol=1e-11)
        assert_almost_equal(e.bound, a * b * 1.5 * 1.5 / 4.0, atol=1e-11)
        assert_almost_equal(e.estimate, a * b * 1.5 * 1.5 / 4.0, atol=1e-11)
        assert_almost_equal(e.bound_per_step, a * b * (1.5 / 4.0) * (1.5 / 4.0), atol=1e-11)
    # the bound is saturated at small tau for two terms: exact / bound -> 1
    var t = 1e-3
    var ex = exact_error(h, 1, t, 1)
    var e = trotter_error(h, t, 1)
    assert_almost_equal(ex / e.bound, 1.0, atol=1e-3)


def test_three_terms_by_hand() raises:
    """H = 1 X0 + 2 Z0 + 3 Y0 in this order.

    C_1 = -(i/2)[2Z, X] = 2 Y; C_2 = -(i/2)[3Y, X + 2Z] = -3 Z + 6 X.
    exact: alpha1 = 2 + 3 sqrt(5), ||Phi2|| = ||2Y - 3Z + 6X|| = 7.
    l1:    alpha1 = 2 + 9 = 11,    ||Phi2|| = 2 + 3 + 6 = 11.
    Dropping either commutator of C_2, or halving any factor, changes these values.
    """
    var h = PauliSum()
    h.add(1.0, 1, 0)
    h.add(2.0, 0, 1)
    h.add(3.0, 1, 1)
    var cs = trotter_commutators(h)
    assert_equal(len(cs), 3)
    assert_equal(len(cs[0]), 0)
    assert_equal(len(cs[1]), 1)
    assert_equal(len(cs[2]), 2)
    var ex = trotter_error(h, 2.0, 5, norm="exact")
    assert_almost_equal(ex.alpha1, 2.0 + 3.0 * sqrt(5.0), atol=1e-10)
    assert_almost_equal(ex.phi2, 7.0, atol=1e-10)
    assert_almost_equal(ex.bound, (2.0 + 3.0 * sqrt(5.0)) * 4.0 / 5.0, atol=1e-10)
    assert_almost_equal(ex.estimate, 7.0 * 4.0 / 5.0, atol=1e-10)
    var l1 = trotter_error(h, 2.0, 5, norm="l1")
    assert_almost_equal(l1.alpha1, 11.0, atol=1e-12)
    assert_almost_equal(l1.phi2, 11.0, atol=1e-12)
    # reversed order: C_1 = -(i/2)[2Z, 3Y] = -6 X, C_2 = -(i/2)[X, 3Y + 2Z] = 3 Z - 2 Y
    var r = PauliSum()
    r.add(3.0, 1, 1)
    r.add(2.0, 0, 1)
    r.add(1.0, 1, 0)
    var er = trotter_error(r, 2.0, 5, norm="exact")
    assert_almost_equal(er.alpha1, 6.0 + sqrt(13.0), atol=1e-10)
    assert_almost_equal(er.phi2, 7.0, atol=1e-10)


def test_guide_hamiltonian() raises:
    """0.3 Z0Z1 + 0.7 X0 + 0.2 X1: alpha1 = 0.3 * 0.7 + 0.3 * 0.2 = 0.27 (the guide example's 0.54 / 2)."""
    var h = PauliSum()
    h.add(0.3, 0, 3)
    h.add(0.7, 1, 0)
    h.add(0.2, 2, 0)
    var e = trotter_error(h, 2.0, 30)
    assert_almost_equal(e.alpha1, 0.27, atol=1e-11)
    assert_almost_equal(e.phi2, 0.27, atol=1e-11)
    assert_almost_equal(e.bound, 0.27 * 4.0 / 30.0, atol=1e-11)


def test_term_order_matches_trotter() raises:
    """The dense reference product is the circuit trotter() builds (terms[0] first), column by column."""
    var rng = Rng(5)
    var n = 3
    var h = random_sum(rng, n, 7)
    var t = 0.8
    var r = 2
    var s = trotter_dense(h, n, t, r)
    var c = trotter(h, t, r)
    var d = 1 << n
    for j in range(d):
        var sv = StateVector(n)
        sv.re[0] = 0.0
        sv.re[j] = 1.0
        evolve_into(sv, c, List[Float64]())
        for i in range(d):
            assert_almost_equal(sv.re[i], s.re[i * d + j], atol=1e-12)
            assert_almost_equal(sv.im[i], s.im[i * d + j], atol=1e-12)


def dense_alpha1_phi2(h: PauliSum, n: Int) raises -> Tuple[Float64, Float64]:
    """Return alpha1 and ||Phi2|| from dense commutators of the dense term matrices (independent of the Pauli algebra)."""
    var d = 1 << n
    var mats = List[CMat]()
    for t in h.terms:
        var s = PauliSum()
        s.terms.append(t)
        mats.append(CMat.of(s, n))
    var alpha1 = 0.0
    var total = CMat(d)
    for p in range(len(mats)):
        var acc = CMat(d)
        for q in range(p):
            var comm = axpy(matmul(mats[p], mats[q]), -1.0, 0.0, matmul(mats[q], mats[p]))
            acc = axpy(acc, 0.0, -0.5, comm)
        alpha1 += hermitian_norm(acc)
        total = axpy(total, 1.0, 0.0, acc)
    return (alpha1, hermitian_norm(total))


def test_random_against_exact() raises:
    """Seeded random Pauli sums: bound >= exact error everywhere; estimate -> exact error as tau -> 0."""
    var rng = Rng(2026)
    var times: List[Float64] = [0.1, 1.0, 3.0, 6.0]  # in units of 1 / ||H||_1
    var step_counts: List[Int] = [1, 2, 5]
    var cases = 0
    var worst_bound = 0.0  # max exact / bound
    var worst_est = 0.0  # max exact / estimate
    for k in range(48):
        var n = 2 + (k % 3)
        var m = 3 + Int(rng.next_u64() % UInt64(4 * n - 2))
        var h = random_sum(rng, n, m)
        var lam = 0.0
        for t in h.terms:
            lam += abs(t.coeff)
        # alpha1 and ||Phi2|| against dense commutators
        var ref_vals = dense_alpha1_phi2(h, n)
        var e1 = trotter_error(h, 1.0, 1, norm="exact")
        assert_almost_equal(e1.alpha1, ref_vals[0], atol=1e-9 * (1.0 + ref_vals[0]))
        assert_almost_equal(e1.phi2, ref_vals[1], atol=1e-9 * (1.0 + ref_vals[1]))
        assert_true(e1.alpha1 >= ref_vals[0] - 1e-13 * (1.0 + ref_vals[0]))  # slack or the L1 cap keeps it an upper bound
        assert_true(e1.phi2 <= e1.alpha1)
        var el = trotter_error(h, 1.0, 1, norm="l1")
        assert_true(el.alpha1 >= e1.alpha1 - 1e-12)
        for tl in times:
            var t = tl / lam
            for r in step_counts:
                var actual = exact_error(h, n, t, r)
                var be = trotter_error(h, t, r, norm="exact")
                var bl = trotter_error(h, t, r, norm="l1")
                assert_true(actual <= be.bound * (1.0 + 1e-9) + 1e-12, "exact error above the bound")
                assert_true(be.bound <= bl.bound * (1.0 + 1e-12) + 1e-15)
                worst_bound = max(worst_bound, actual / be.bound)
                worst_est = max(worst_est, actual / be.estimate)
                cases += 1
        # leading order: exact / estimate -> 1 (tau ||H||_1 = 1e-4)
        var ts = 1e-4 / lam
        var small = exact_error(h, n, ts, 1)
        var es = trotter_error(h, ts, 1)
        assert_almost_equal(small / es.estimate, 1.0, atol=1e-3)
    assert_equal(cases, 48 * 12)
    print("  random cases:", cases, " max exact/bound:", worst_bound, " max exact/estimate:", worst_est)
    assert_true(worst_bound <= 1.0 + 1e-9)


def test_options() raises:
    var h = PauliSum()
    h.add(1.0, 1, 0)
    h.add(1.0, 0, 1)
    var raised = 0
    try:
        _ = trotter_error(h, 1.0, 0)
    except e:
        if String(e).startswith("QM-E808"):
            raised += 1
    try:
        _ = trotter_error(h, 1.0, 4, order=2)
    except e:
        if String(e).startswith("QM-E808"):
            raised += 1
    try:
        _ = trotter_error(h, 1.0, 4, norm="lanczos")
    except e:
        if String(e).startswith("QM-E801"):
            raised += 1
    var big = PauliSum()
    big.add(1.0, UInt64(1) << 8, 0)
    big.add(1.0, 0, UInt64(1) << 8)
    try:
        _ = trotter_error(big, 1.0, 4, norm="exact")
    except e:
        if String(e).startswith("QM-E807"):
            raised += 1
    assert_equal(raised, 4)
    var auto = trotter_error(big, 1.0, 4)
    assert_equal(auto.norm, String("l1"))
    assert_almost_equal(auto.bound, 1.0 / 4.0, atol=1e-12)  # C_1 = Y8, alpha1 = 1, 4 (1/4)^2
    # negative time: same error as positive time
    var neg = trotter_error(h, -1.0, 2)
    var pos = trotter_error(h, 1.0, 2)
    assert_equal(neg.bound, pos.bound)


def main() raises:
    test_half_commutator_dense()
    test_commuting_terms_give_zero()
    test_two_terms_by_hand()
    test_three_terms_by_hand()
    test_guide_hamiltonian()
    test_term_order_matches_trotter()
    test_random_against_exact()
    test_options()
    print("test_trotter_error OK")
