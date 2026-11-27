"""Error of the first-order product formula: a rigorous bound and a leading-order estimate.

For H = H_1 + ... + H_m, `trotter(h, t, r, order=1)` applies, r times,
S(tau) = e^{-i tau H_m} ... e^{-i tau H_1} with tau = t / r (terms[0] first). With

    C_p = -(i/2) sum_{q < p} [H_p, H_q]      (a Hermitian Pauli sum, real coefficients)

`trotter_error` returns two different numbers:

* `bound` = r alpha_1 tau^2, alpha_1 = sum_p ||C_p||. A rigorous upper bound on
  ||e^{-iHt} - S(tau)^r|| (spectral norm) for every t and r. It is the
  commutator bound of Childs, Su, Tran, Wiebe and Zhu (PRX 11, 011020, 2021),
  eq. 6 of Dion and D'Anjou (arXiv:2611.33438). Proof sketch: with
    A_p = H_1 + ... + H_p, S_p = e^{-i tau H_p} S_{p-1} and U_p = e^{-i tau A_p},
    ||S_p - U_p|| <= ||S_{p-1} - U_{p-1}|| + ||e^{-i tau H_p} e^{-i tau A_{p-1}} - e^{-i tau A_p}||,
  and the two-exponential error is at most (tau^2 / 2) ||[H_p, A_{p-1}]|| = tau^2 ||C_p||;
  r steps add at most r times the one-step error.
* `estimate` = r ||C_1 + ... + C_m|| tau^2 = r ||Phi_2|| tau^2, the leading term of the
  phase operator of arXiv:2611.33438 (eq. 16 and 19-20). It is NOT a bound: it is
  exact to leading order in tau, and the true error can exceed it (by up to 1.6x
  at tau ||H|| = 2 in a sample of 300 random Hamiltonians, and 4x in an outlier,
  see docs/ALGORITHMS.md). estimate <= bound always (triangle inequality); they are
  equal when m = 2.

Norms (`norm`): "l1" is the sum of |coefficients| of the Pauli sum, an upper
bound at any size. "exact" is max |eigenvalue| by dense diagonalisation
(n <= 7); the bound adds a slack of 1e-12 times the L1 norm to cover the rounding
of the eigensolver, and never exceeds the L1 value. "auto" is "exact" for n <= 7
and "l1" above. No iterative (Lanczos) norm is used: a Ritz value is a lower
bound on the norm and would make the bound unsafe.

Only order 1 is covered. The paper's second-order formulas are not implemented:
its eq. 7 (second-order bound), as printed with the ordering of its eq. 4, is
exceeded by the true error in tests, and its eq. 18 (Phi_3 of the second-order
formula = 1/4 of the first-order one) could not be reproduced.

Error codes: QM-E808 (steps < 1, order other than 1), QM-E801 (unknown norm),
QM-E807 (exact norm above 7 qubits), QM-E803 (num_qubits smaller than H).
"""

from qumo.ir.pauli import PauliTerm, PauliSum, pauli_half_commutator
from qumo.algorithms.hamiltonians import exact_spectrum

comptime TROTTER_ERROR_EXACT_MAX_QUBITS = 7  # exact_spectrum's limit
comptime TROTTER_ERROR_NORM_SLACK = 1e-12  # relative to the L1 norm, added to an exact norm in the bound


@fieldwise_init
struct TrotterError(Copyable, Movable, Writable):
    """Result of `trotter_error`. `bound` is rigorous; `estimate` is not a bound."""

    var bound: Float64  # rigorous: ||e^{-iHt} - S(t/r)^r|| <= bound
    var estimate: Float64  # leading-order estimate of the same error (can be exceeded)
    var bound_per_step: Float64  # alpha_1 tau^2
    var estimate_per_step: Float64  # ||Phi_2|| tau^2
    var alpha1: Float64  # sum_p ||C_p|| (with the bound's slack)
    var phi2: Float64  # ||sum_p C_p||
    var time: Float64
    var steps: Int
    var norm: String  # "exact" or "l1", as used

    def ratio(self) -> Float64:
        """bound / estimate (>= 1; how loose the bound is at small tau). 1 when both are 0."""
        if self.estimate <= 0.0:
            return 1.0
        return self.bound / self.estimate

    def write_to(self, mut w: Some[Writer]):
        w.write("TrotterError(bound=", self.bound, ", estimate=", self.estimate, " [not a bound], alpha1=", self.alpha1,
                ", phi2=", self.phi2, ", steps=", self.steps, ", norm=", self.norm, ")")


def trotter_commutators(h: PauliSum) -> List[PauliSum]:
    """C_p = -(i/2) sum_{q < p} [H_p, H_q] for p = 0 .. m-1, in trotter()'s term order.

    C_0 is empty. Terms whose coefficients cancel exactly are removed.
    """
    var out = List[PauliSum](capacity=len(h.terms))
    for p in range(len(h.terms)):
        var c = PauliSum()
        for q in range(p):
            var t = pauli_half_commutator(h.terms[p], h.terms[q])
            if t.coeff != 0.0:
                c.add(t.coeff, t.x, t.z)
        var kept = PauliSum()
        for t in c.terms:
            if t.coeff != 0.0:
                kept.terms.append(t)
        out.append(kept^)
    return out^


def _l1(h: PauliSum) -> Float64:
    var s = 0.0
    for t in h.terms:
        s += abs(t.coeff)
    return s


def pauli_sum_norm(h: PauliSum, num_qubits: Int, norm: String, upper: Bool) raises -> Float64:
    """Spectral norm of a Hermitian Pauli sum: "l1" (an upper bound) or "exact" (dense, n <= 7).

    With upper=True the exact value is raised by 1e-12 times the L1 norm (the
    eigensolver's rounding) so it stays an upper bound. An exact value is
    always capped at the L1 norm.
    """
    var l1 = _l1(h)
    if l1 == 0.0:
        return 0.0
    if norm == "l1":
        return l1
    if norm != "exact":
        raise Error("QM-E801: unknown norm '" + norm + "'. Hint: use 'auto', 'exact' (dense, n <= 7) or 'l1'.")
    if num_qubits > TROTTER_ERROR_EXACT_MAX_QUBITS:
        raise Error("QM-E807: the exact norm is limited to " + String(TROTTER_ERROR_EXACT_MAX_QUBITS) + " qubits, got " + String(num_qubits) + ". Hint: use norm='l1' (a looser upper bound at any size).")
    var ev = exact_spectrum(h, num_qubits)
    var v = max(abs(ev[0]), abs(ev[len(ev) - 1]))
    if upper:
        v += TROTTER_ERROR_NORM_SLACK * l1
    return min(v, l1)  # the L1 norm is an upper bound on the true norm


def trotter_error(h: PauliSum, time: Float64, steps: Int, order: Int = 1, num_qubits: Int = -1, norm: String = "auto") raises -> TrotterError:
    """Rigorous bound and leading-order estimate of the error of trotter(h, time, steps, order=1).

    Returns TrotterError: `bound` = steps * alpha_1 * tau^2 is an upper bound on
    ||e^{-iHt} - S(tau)^steps|| (tau = |time| / steps); `estimate` =
    steps * ||Phi_2|| * tau^2 is correct to leading order in tau but is not a bound.
    The term order is trotter()'s (terms[0] applied first); reordering the terms
    changes both numbers. norm: "auto" (exact for n <= 7, else l1), "exact", "l1".
    Order 2 and higher are refused (QM-E808): no rigorous bound is implemented for them.
    """
    if steps < 1:
        raise Error("QM-E808: trotter_error() needs steps >= 1, got " + String(steps) + ".")
    if order != 1:
        raise Error("QM-E808: trotter_error() covers order 1 (Lie-Trotter) only, got order " + String(order) + ". Hint: no rigorous bound for order 2 or higher is implemented; the order-1 bound does not apply to Strang or Suzuki circuits.")
    var n = num_qubits if num_qubits > 0 else max(h.num_qubits(), 1)
    if h.num_qubits() > n:
        raise Error("QM-E803: the Hamiltonian acts on " + String(h.num_qubits()) + " qubits but num_qubits = " + String(n) + ".")
    var used = norm
    if norm == "auto":
        used = "exact" if n <= TROTTER_ERROR_EXACT_MAX_QUBITS else "l1"
    elif norm != "exact" and norm != "l1":
        raise Error("QM-E801: unknown norm '" + norm + "'. Hint: use 'auto', 'exact' (dense, n <= 7) or 'l1'.")
    var cs = trotter_commutators(h)
    var alpha1 = 0.0
    var total = PauliSum()
    for ref c in cs:
        alpha1 += pauli_sum_norm(c, n, used, True)
        for t in c.terms:
            total.add(t.coeff, t.x, t.z)
    var phi2 = pauli_sum_norm(total, n, used, False)
    var tau = abs(time) / Float64(steps)
    var bps = alpha1 * tau * tau
    var eps = phi2 * tau * tau
    return TrotterError(Float64(steps) * bps, Float64(steps) * eps, bps, eps, alpha1, phi2, time, steps, used)
