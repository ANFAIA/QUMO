"""Trotter error of a first-order product formula: the rigorous bound, the estimate and the true error.

Reproduces: Dion, D'Anjou - first-order commutator bound (eq. 6) and leading-order phase estimate (eq. 16, 19-20)
Source: https://arxiv.org/abs/2611.33438
Framework: QUMO - https://github.com/ANFAIA/QUMO

trotter_error(h, t, r) returns two numbers for trotter(h, t, r, order=1). `bound` = r alpha_1 (t/r)^2
is the commutator bound of Childs et al. (PRX 11, 011020, 2021), a guaranteed upper limit on the
spectral-norm error. `estimate` = r ||Phi_2|| (t/r)^2 is the leading term of the phase operator of
Dion and D'Anjou (arXiv:2611.33438): correct to leading order in t/r, but not a bound. On an open
XXZ chain with a field (6 qubits, 21 terms in the builder's order) the two differ by a factor 2.9, so
the steps needed for an error of 1e-3 differ by that factor too. Over many steps the true error is
below the estimate here (by 12% at 6000 steps): the step errors do not add up fully. The true error is computed here from
the dense unitaries: e^{-iHt} column by column with exact_evolve, and the Trotter step from the
trotter() circuit on every basis state, raised to the r-th power.
Checks: alpha_1 and ||Phi_2|| of a 2-term and a 3-term Hamiltonian against values derived by hand;
the guide example's 0.27; on the chain, the true error is below the bound at every step count tried,
the step count from the bound meets the target, the estimate matches the true one-step error to 1%
at small t/r, the true minimum step count is at most the estimate's, which is at most the bound's,
and the bound overstates the true minimum by more than 2x.

Run: .venv/bin/mojo run -I . examples/chemistry_and_physics/trotter_error_bound.mojo
example-meta: {"framework": "qumo", "tier": "fast", "timeout": 180, "requires": [], "args": []}
"""

from std.math import sqrt, ceil
from qumo.ir.pauli import PauliSum
from qumo.sim.runner import evolve_into
from qumo.sim.statevector import StateVector
from qumo.algorithms.hamiltonians import heisenberg_xxz, exact_evolve
from qumo.algorithms.library import trotter
from qumo.algorithms.linalg import jacobi_eigh
from qumo.algorithms.trotter_error import trotter_error
from qumo.testing.checks import CheckLog

comptime N = 6
comptime TIME = 1.0
comptime TARGET = 1e-3


# ---------------------------------------------------------------------------
# Dense unitaries (row-major, separate re / im), dimension 2^N
# ---------------------------------------------------------------------------
def basis(n: Int, j: Int) -> StateVector:
    var sv = StateVector(n)
    sv.re[0] = 0.0
    sv.re[j] = 1.0
    return sv^


def exact_unitary(h: PauliSum, n: Int, t: Float64) raises -> Tuple[List[Float64], List[Float64]]:
    var d = 1 << n
    var re = List[Float64](length=d * d, fill=0.0)
    var im = List[Float64](length=d * d, fill=0.0)
    for j in range(d):
        var col = exact_evolve(h, basis(n, j), t)
        for i in range(d):
            re[i * d + j] = col.re[i]
            im[i * d + j] = col.im[i]
    return (re^, im^)


def trotter_step(h: PauliSum, n: Int, tau: Float64) raises -> Tuple[List[Float64], List[Float64]]:
    """One first-order step, the circuit trotter(h, tau, 1) applied to every basis state."""
    var c = trotter(h, tau, 1, num_qubits=n)
    var d = 1 << n
    var re = List[Float64](length=d * d, fill=0.0)
    var im = List[Float64](length=d * d, fill=0.0)
    for j in range(d):
        var sv = basis(n, j)
        evolve_into(sv, c, List[Float64]())
        for i in range(d):
            re[i * d + j] = sv.re[i]
            im[i * d + j] = sv.im[i]
    return (re^, im^)


def matmul(a: Tuple[List[Float64], List[Float64]], b: Tuple[List[Float64], List[Float64]], d: Int) -> Tuple[List[Float64], List[Float64]]:
    var re = List[Float64](length=d * d, fill=0.0)
    var im = List[Float64](length=d * d, fill=0.0)
    for i in range(d):
        for k in range(d):
            var ar = a[0][i * d + k]
            var ai = a[1][i * d + k]
            for j in range(d):
                re[i * d + j] += ar * b[0][k * d + j] - ai * b[1][k * d + j]
                im[i * d + j] += ar * b[1][k * d + j] + ai * b[0][k * d + j]
    return (re^, im^)


def matpow(a: Tuple[List[Float64], List[Float64]], d: Int, r: Int) -> Tuple[List[Float64], List[Float64]]:
    var re = List[Float64](length=d * d, fill=0.0)
    for i in range(d):
        re[i * d + i] = 1.0
    var out = (re^, List[Float64](length=d * d, fill=0.0))
    var base = (a[0].copy(), a[1].copy())
    var k = r
    while k > 0:
        if (k & 1) == 1:
            out = matmul(out, base, d)
        k >>= 1
        if k > 0:
            base = matmul(base, base, d)
    return out^


def spectral_norm_diff(u: Tuple[List[Float64], List[Float64]], s: Tuple[List[Float64], List[Float64]], d: Int) -> Float64:
    """||U - S|| = sqrt(lambda_max(E^dag E)), through the real symmetric embedding."""
    var er = List[Float64](length=d * d, fill=0.0)
    var ei = List[Float64](length=d * d, fill=0.0)
    for k in range(d * d):
        er[k] = u[0][k] - s[0][k]
        ei[k] = u[1][k] - s[1][k]
    var m = 2 * d
    var big = List[Float64](length=m * m, fill=0.0)
    for i in range(d):
        for j in range(d):
            var gr = 0.0
            var gi = 0.0
            for k in range(d):
                var ar = er[k * d + i]
                var ai = -ei[k * d + i]
                gr += ar * er[k * d + j] - ai * ei[k * d + j]
                gi += ar * ei[k * d + j] + ai * er[k * d + j]
            big[i * m + j] = gr
            big[(i + d) * m + j + d] = gr
            big[i * m + j + d] = -gi
            big[(i + d) * m + j] = gi
    var ev = jacobi_eigh(big, m).values.copy()
    return sqrt(max(ev[m - 1], 0.0))


def true_error(h: PauliSum, u: Tuple[List[Float64], List[Float64]], n: Int, r: Int) raises -> Float64:
    var d = 1 << n
    var s = matpow(trotter_step(h, n, TIME / Float64(r)), d, r)
    return spectral_norm_diff(u, s, d)


def main() raises:
    var log = CheckLog("chemistry_and_physics/trotter_error_bound")

    # Hand-derived small cases. H = a X0 + b Z0: -(i/2)[bZ, aX] = ab Y, so alpha_1 = ||Phi_2|| = ab.
    var two = PauliSum()
    two.add(0.7, 1, 0)
    two.add(0.4, 0, 1)
    var e2 = trotter_error(two, 1.0, 1)
    _ = log.close("two terms alpha_1 = ab", e2.alpha1, 0.28, 1e-11)
    _ = log.close("two terms estimate = bound", e2.estimate, e2.bound, 1e-11)
    # H = X0 + 2 Z0 + 3 Y0: C_1 = 2Y, C_2 = -3Z + 6X; alpha_1 = 2 + 3 sqrt 5, ||Phi_2|| = ||6X + 2Y - 3Z|| = 7.
    var three = PauliSum()
    three.add(1.0, 1, 0)
    three.add(2.0, 0, 1)
    three.add(3.0, 1, 1)
    var e3 = trotter_error(three, 1.0, 1)
    _ = log.close("three terms alpha_1 = 2 + 3 sqrt 5", e3.alpha1, 2.0 + 3.0 * sqrt(5.0), 1e-10)
    _ = log.close("three terms norm of Phi_2 = 7", e3.phi2, 7.0, 1e-10)
    # The guide example (hamiltonian_simulation_guide.mojo): its pairwise sum 0.54 / 2 is already alpha_1.
    var guide = PauliSum()
    guide.add(0.3, 0, 3)
    guide.add(0.7, 1, 0)
    guide.add(0.2, 2, 0)
    _ = log.close("guide Hamiltonian alpha_1 = 0.27", trotter_error(guide, 1.0, 1).alpha1, 0.27, 1e-11)

    # XXZ chain, J = 1, Delta = 0.5, field 0.3, open boundary: 5 bonds x (XX, YY, ZZ), then 6 fields.
    var h = heisenberg_xxz(N, 1.0, 0.5, False, 0.3)
    var e = trotter_error(h, TIME, 1)
    var el = trotter_error(h, TIME, 1, norm="l1")
    print("XXZ chain, n =", N, ", terms =", len(h.terms), ", t =", TIME)
    print("  alpha_1 =", e.alpha1, " (l1:", el.alpha1, ")  ||Phi_2|| =", e.phi2, "  bound / estimate =", e.ratio())
    _ = log.check("exact-norm alpha_1 at most the l1 one", e.alpha1 <= el.alpha1, String(e.alpha1) + " vs " + String(el.alpha1))
    _ = log.check("estimate at most the bound", e.phi2 <= e.alpha1)

    var r_bound = Int(ceil(e.alpha1 * TIME * TIME / TARGET))
    var r_est = Int(ceil(e.phi2 * TIME * TIME / TARGET))
    var u = exact_unitary(h, N, TIME)
    # smallest r with true error <= TARGET (bisection; the error decreases with r here)
    var lo = 1
    var hi = r_bound
    while lo < hi:
        var mid = (lo + hi) // 2
        if true_error(h, u, N, mid) <= TARGET:
            hi = mid
        else:
            lo = mid + 1
    var r_true = lo
    print("  steps for error", TARGET, ": bound", r_bound, ", estimate", r_est, ", true", r_true)

    var below = True
    var detail = String()
    var rs: List[Int] = [1, 2, 5, 20, 100, r_true, r_est, r_bound]
    for r in rs:
        var actual = true_error(h, u, N, r)
        var b = trotter_error(h, TIME, r)
        print("  r =", r, " true", actual, " bound", b.bound, " estimate", b.estimate)
        if actual > b.bound:
            below = False
            detail += " r=" + String(r)
    _ = log.check("true error below the bound for every r tried", below, detail)
    _ = log.check("bound's step count meets the target", true_error(h, u, N, r_bound) <= TARGET)
    # One step: the estimate is the leading term, so it matches the true one-step error at small tau.
    var tau = TIME / Float64(r_true)
    var one = spectral_norm_diff(exact_unitary(h, N, tau), trotter_step(h, N, tau), 1 << N)
    _ = log.within("one-step true error over the estimate at tau = t / r_true", one / trotter_error(h, tau, 1).estimate, 0.99, 1.01)
    # Many steps: the step errors do not add up fully, so the true error ends below the estimate here.
    _ = log.check("true minimum at most the estimate's step count at most the bound's", r_true <= r_est and r_est <= r_bound,
                  String(r_true) + ", " + String(r_est) + ", " + String(r_bound))
    _ = log.check("bound overstates the step count by more than 2x", Float64(r_bound) > 2.0 * Float64(r_true),
                  String(r_bound) + " vs " + String(r_true))
    log.finish()
