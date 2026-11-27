"""Schmidt coefficients of a simulated state from the complex SVD.

Reproduces: Demmel, Veselic - Jacobi's method is more accurate than QR (one-sided Jacobi SVD, Algorithm 4.1)
Source: https://www.netlib.org/lapack/lawnspdf/lawn15.pdf
Framework: QUMO - https://github.com/ANFAIA/QUMO

A pure state of qubits A = {0, 1, 2} and B = {3, 4, 5} is the 8 x 8 matrix
M[a, b] = <a + 8 b | psi> (little-endian: qubit k is bit k of the basis index), and
its singular values are the Schmidt coefficients of the cut A | B. The circuit makes
three pairs (k, k + 3) in cos(t_k / 2)|00> + sin(t_k / 2)|11> and then scrambles A
and B with seeded random local layers. Local unitaries do not change Schmidt
coefficients, so they must be the eight products of one cos(t_k / 2) or sin(t_k / 2)
per pair: the reference is that closed form, not a library call.

`qumo.core.svd.svd` is that paper's one-sided Jacobi SVD (LAPACK Working Note 15,
section 4), extended to complex entries; nothing of the paper's accuracy analysis is
checked here.

Checks:
- the eight singular values equal the sorted closed-form products;
- the entanglement entropy -sum s^2 log s^2 equals the sum of the pair entropies;
- U diag(s) V^H rebuilds M, and U, V have orthonormal columns;
- a product state (no CX) has Schmidt rank 1.

Run: .venv/bin/mojo run -I . examples/simulation/schmidt_svd.mojo
example-meta: {"framework": "qumo", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from std.math import sqrt, cos, sin, log, pi
from qumo.ir.circuit import Circuit
from qumo.core.rng import Rng
from qumo.core.svd import CMatrix, svd
from qumo.sim.runner import evolve
from qumo.testing.checks import CheckLog


def scramble(mut c: Circuit, lo: Int, mut rng: Rng) raises:
    """Two random layers on qubits lo, lo + 1, lo + 2 only."""
    for _layer in range(2):
        for q in range(lo, lo + 3):
            c.rx(2 * pi * rng.uniform(), q)
            c.rz(2 * pi * rng.uniform(), q)
        c.cx(lo, lo + 1)
        c.cx(lo + 1, lo + 2)


def amplitude_matrix(c: Circuit) raises -> CMatrix:
    var sv = evolve(c)
    var m = CMatrix(8, 8)
    for a in range(8):
        for b in range(8):
            m.set(a, b, sv.re[a + 8 * b], sv.im[a + 8 * b])
    return m^


def entropy(ps: List[Float64]) -> Float64:
    var h = 0.0
    for p in ps:
        if p > 1e-300:
            h -= p * log(p)
    return h


def main() raises:
    var log_ = CheckLog("simulation/schmidt_svd")
    var t: List[Float64] = [0.7, 1.9, 2.6]
    var c = Circuit(6)
    for k in range(3):
        c.ry(t[k], k)
        c.cx(k, k + 3)
    var rng = Rng(11)
    scramble(c, 0, rng)
    scramble(c, 3, rng)
    var m = amplitude_matrix(c)
    var r = svd(m)

    # Closed form: products of cos/sin(t_k / 2), sorted descending.
    var closed = List[Float64]()
    for mask in range(8):
        var v = 1.0
        for k in range(3):
            v *= sin(t[k] / 2) if (mask >> k) & 1 else cos(t[k] / 2)
        closed.append(abs(v))
    for i in range(1, 8):
        var x = closed[i]
        var j = i - 1
        while j >= 0 and closed[j] < x:
            closed[j + 1] = closed[j]
            j -= 1
        closed[j + 1] = x
    var worst = 0.0
    for i in range(8):
        worst = max(worst, abs(r.s[i] - closed[i]))
        print("  s[" + String(i) + "] =", r.s[i], " closed form", closed[i])
    _ = log_.close("Schmidt coefficients match the closed form (max deviation)", worst, 0.0, 1e-13)

    var p = List[Float64]()
    for i in range(8):
        p.append(r.s[i] * r.s[i])
    var h_pairs = 0.0
    for k in range(3):
        var pairs: List[Float64] = [cos(t[k] / 2) ** 2, sin(t[k] / 2) ** 2]
        h_pairs += entropy(pairs)
    _ = log_.close("entanglement entropy equals the sum over the three pairs", entropy(p), h_pairs, 1e-12)

    var rec_err = 0.0
    var orth_err = 0.0
    for a in range(8):
        for b in range(8):
            var sr = 0.0
            var si = 0.0
            var gr = 0.0
            var gi = 0.0
            var hr = 0.0
            var hi = 0.0
            for k in range(8):
                var ur = r.u.get_re(a, k) * r.s[k]
                var ui = r.u.get_im(a, k) * r.s[k]
                var vr = r.v.get_re(b, k)
                var vi = -r.v.get_im(b, k)
                sr += ur * vr - ui * vi
                si += ur * vi + ui * vr
                # (U^H U)[a, b] and (V^H V)[a, b]
                gr += r.u.get_re(k, a) * r.u.get_re(k, b) + r.u.get_im(k, a) * r.u.get_im(k, b)
                gi += r.u.get_re(k, a) * r.u.get_im(k, b) - r.u.get_im(k, a) * r.u.get_re(k, b)
                hr += r.v.get_re(k, a) * r.v.get_re(k, b) + r.v.get_im(k, a) * r.v.get_im(k, b)
                hi += r.v.get_re(k, a) * r.v.get_im(k, b) - r.v.get_im(k, a) * r.v.get_re(k, b)
            rec_err = max(rec_err, sqrt((sr - m.get_re(a, b)) ** 2 + (si - m.get_im(a, b)) ** 2))
            var d = 1.0 if a == b else 0.0
            orth_err = max(orth_err, sqrt((gr - d) ** 2 + gi * gi))
            orth_err = max(orth_err, sqrt((hr - d) ** 2 + hi * hi))
    _ = log_.close("U diag(s) V^H rebuilds the amplitude matrix (max entry error)", rec_err, 0.0, 1e-13)
    _ = log_.close("U and V have orthonormal columns (max entry error)", orth_err, 0.0, 1e-13)

    var c2 = Circuit(6)
    var rng2 = Rng(5)
    scramble(c2, 0, rng2)
    scramble(c2, 3, rng2)
    var r2 = svd(amplitude_matrix(c2))
    var rank = 0
    for i in range(8):
        if r2.s[i] > 1e-12:
            rank += 1
    _ = log_.check("a product state has Schmidt rank 1", rank == 1, "rank " + String(rank))
    print("  Jacobi sweeps:", r.sweeps, "(entangled),", r2.sweeps, "(product)")
    log_.finish()
