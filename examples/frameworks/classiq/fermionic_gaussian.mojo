"""Fermionic Gaussian states: the notebook's Slater determinant and general quadratic ground state, prepared on QUMO.

Reproduces: Classiq library - Preparation of Fermionic Gaussian States
Source: https://github.com/Classiq/classiq-library/blob/main/algorithms/quantum_state_preparation/fermionic_gaussian/fermionic_gaussian.ipynb (Classiq Library, MIT)

The notebook prepares ground states of quadratic fermionic Hamiltonians under the Jordan-Wigner map. Part 1 is a number-conserving one-body matrix h (the notebook's periodic 4-site chain with two spin species, 8 modes) with one particle: its lowest eigenvector is prepared from the state with mode 0 occupied by adjacent two-mode rotations. Part 2 is the notebook's general quadratic Hamiltonian on two modes, with a pairing term: its Majorana matrix A is brought to canonical form by a real orthogonal frame, the frame is written as planar rotations between adjacent Majoranas (an RZ on a mode, an RXX across neighbouring modes), and these gates act on the Fock state whose occupations fix the signs of the canonical energies. The checks compare the QUMO states with exact diagonalization (of the one-body h in Part 1, and of the Fock-space Hamiltonian in Part 2: fidelity 1, plus the Part 2 amplitude magnitudes and relative sign), check the canonical energies against the Fock spectrum, and compare the notebook's printed values: the Majorana matrix A, the canonical energies 1.292 and 0.511, the Fock spectrum, and the ground-state amplitudes.

Deviations from the notebook: (1) Part 1 is the one-particle case the notebook runs (N = 1). The rotation sequence is a direct sweep over adjacent entries instead of OpenFermion's givens_decomposition, and the two-mode Givens gate is CX, controlled-RY, CX (as in hhl.mojo) instead of phased_givens; the phases are not needed because h is real and N = 1. (2) Part 2 is the notebook's two-mode instance. The canonical frame comes from QUMO's Jacobi eigensolver applied to A^T A instead of scipy's real Schur decomposition, and the gates are RZ and RXX planar Majorana rotations instead of OpenFermion's double-Givens rotations and particle-hole X gates. The prepared state equals the notebook's up to a global phase. The canonical energies are assumed non-degenerate, which holds for this instance. (3) Classiq's synthesize, the transpiled depth printed by the notebook (25) and calculate_state_vector are replaced by QUMO's evolve; the depth is not reproduced, and the notebook's amplitude plot is not drawn. (4) The notebook's first 4 x 4 Givens demonstration is classical and is not run.

Run: .venv/bin/mojo run -I . examples/frameworks/classiq/fermionic_gaussian.mojo
example-meta: {"framework": "classiq", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from std.math import atan2, exp, sqrt
from qumo.ir.circuit import Circuit
from qumo.algorithms.linalg import jacobi_eigh
from qumo.sim.runner import evolve
from qumo.testing.checks import CheckLog

comptime TOLERANCE = 1e-10  # the notebook's own TOLERANCE


# ---------------------------------------------------------------------------
# Dense row-major helpers
# ---------------------------------------------------------------------------
def _matmul(a: List[Float64], b: List[Float64], m: Int) -> List[Float64]:
    var out = List[Float64](length=m * m, fill=0.0)
    for i in range(m):
        for k in range(m):
            var aik = a[i * m + k]
            for j in range(m):
                out[i * m + j] += aik * b[k * m + j]
    return out^


def _transpose(a: List[Float64], m: Int) -> List[Float64]:
    var out = List[Float64](length=m * m, fill=0.0)
    for i in range(m):
        for j in range(m):
            out[j * m + i] = a[i * m + j]
    return out^


def _matvec(a: List[Float64], x: List[Float64], m: Int) -> List[Float64]:
    var out = List[Float64](length=m, fill=0.0)
    for i in range(m):
        var s = 0.0
        for j in range(m):
            s += a[i * m + j] * x[j]
        out[i] = s
    return out^


def _det(a: List[Float64], m: Int) -> Float64:
    """Determinant by Gaussian elimination with partial pivoting."""
    var w = a.copy()
    var det = 1.0
    for c in range(m):
        var piv = c
        for r in range(c + 1, m):
            if abs(w[r * m + c]) > abs(w[piv * m + c]):
                piv = r
        if w[piv * m + c] == 0.0:
            return 0.0
        if piv != c:
            for k in range(m):
                var t = w[c * m + k]
                w[c * m + k] = w[piv * m + k]
                w[piv * m + k] = t
            det = -det
        det *= w[c * m + c]
        for r in range(c + 1, m):
            var f = w[r * m + c] / w[c * m + c]
            for k in range(c, m):
                w[r * m + k] -= f * w[c * m + k]
    return det


# ---------------------------------------------------------------------------
# Part 1: Slater determinant, one particle on 8 modes
# ---------------------------------------------------------------------------
def _qubit_idx(site: Int, spin: Int, L: Int) -> Int:
    return 2 * (site % L) + (spin % 2)


def single_electron_h(L: Int, J: Float64, lam: Float64, mean: Float64, std: Float64) -> List[Float64]:
    """The notebook's h = kinetic_energy(L, J) + spin_potential(L, spin=0, (lam, mean, std)), row-major 2L x 2L."""
    var M = 2 * L
    var K = List[Float64](length=M * M, fill=0.0)
    for site in range(L):
        for spin in range(2):
            K[_qubit_idx(site, spin, L) * M + _qubit_idx(site + 1, spin, L)] = -J
    var h = List[Float64](length=M * M, fill=0.0)
    for i in range(M):
        for j in range(M):
            h[i * M + j] = K[i * M + j] + K[j * M + i]  # 2 * sym(K)
    for site in range(L):
        var mu = _qubit_idx(site, 0, L)
        var d = Float64(site) - mean
        h[mu * M + mu] += -lam * exp(-(d * d) / (2.0 * std * std))
    return h^


def _excitation_givens(mut c: Circuit, p: Int, q: Int, theta: Float64) raises:
    """Rotation of the one-excitation pair {|e_p>, |e_q>}, identity on the other two-qubit basis states:
    |e_p> -> cos(t/2)|e_p> + sin(t/2)|e_q>, |e_q> -> -sin(t/2)|e_p> + cos(t/2)|e_q>.
    Built as CX(p -> q), controlled-RY(-t) with control q and target p, then CX(p -> q)."""
    c.cx(p, q)
    c.cry(-theta, q, p)
    c.cx(p, q)


# ---------------------------------------------------------------------------
# Part 2: general quadratic Hamiltonian on n modes
# ---------------------------------------------------------------------------
def majorana_matrix(h: List[Float64], delta: List[Float64], n: Int) -> List[Float64]:
    """A (2n x 2n, real antisymmetric) with H = tr(h)/2 + (i/4) sum_kl A_kl m_k m_l, where the Majoranas are
    m_{2j} = Z..Z X_j and m_{2j+1} = Z..Z Y_j (so c_j = (m_{2j} + i m_{2j+1})/2), and
    H = sum_mn h_mn c_m^+ c_n + sum_{m<n} Delta_mn (c_m^+ c_n^+ + c_n c_m)."""
    var m2 = 2 * n
    var U = List[Float64](length=m2 * m2, fill=0.0)
    for mu in range(n):
        U[(2 * mu) * m2 + 2 * mu + 1] += h[mu * n + mu]
    for mu in range(n):
        for nu in range(mu + 1, n):
            var t = h[mu * n + nu]
            var d = delta[mu * n + nu]
            U[(2 * mu) * m2 + 2 * nu + 1] += t - d
            U[(2 * nu) * m2 + 2 * mu + 1] += t + d
    var A = List[Float64](length=m2 * m2, fill=0.0)
    for k in range(m2):
        for l in range(m2):
            A[k * m2 + l] = U[k * m2 + l] - U[l * m2 + k]
    return A^


def _annihilator(mu: Int, n: Int) -> List[Float64]:
    """Jordan-Wigner annihilator c_mu on n qubits (bit k of the basis index is mode k), as a real matrix."""
    var dim = 1 << n
    var c = List[Float64](length=dim * dim, fill=0.0)
    for s in range(dim):
        if ((s >> mu) & 1) == 1:
            var sign = 1.0
            for k in range(mu):
                if ((s >> k) & 1) == 1:
                    sign = -sign
            c[(s ^ (1 << mu)) * dim + s] = sign
    return c^


def fock_hamiltonian(h: List[Float64], delta: List[Float64], n: Int) -> List[Float64]:
    """H as a dense real symmetric 2^n x 2^n matrix, built from the Jordan-Wigner annihilators (no Majoranas)."""
    var dim = 1 << n
    var cs = List[List[Float64]]()
    for mu in range(n):
        cs.append(_annihilator(mu, n))
    var H = List[Float64](length=dim * dim, fill=0.0)
    for mu in range(n):
        for nu in range(n):
            var term = _matmul(_transpose(cs[mu], dim), cs[nu], dim)
            for i in range(dim * dim):
                H[i] += h[mu * n + nu] * term[i]
    for mu in range(n):
        for nu in range(mu + 1, n):
            var create_pair = _matmul(_transpose(cs[mu], dim), _transpose(cs[nu], dim), dim)
            var annihilate_pair = _matmul(cs[nu], cs[mu], dim)
            var d = delta[mu * n + nu]
            for i in range(dim * dim):
                H[i] += d * (create_pair[i] + annihilate_pair[i])
    return H^


def main() raises:
    var log = CheckLog("classiq/fermionic_gaussian")

    # ======================= Part 1: Slater determinant, one particle =======================
    var L = 4
    var M = 2 * L
    var h1 = single_electron_h(L, 1.0, 10.0, Float64(L) / 2.0, Float64(L) / 6.0)
    var eig1 = jacobi_eigh(h1, M)
    var v = eig1.vector(0)  # exact single-particle ground state (the notebook's exact_ground_state)
    var e0 = eig1.values[0]
    var res1 = 0.0
    for i in range(M):
        var hv = 0.0
        for j in range(M):
            hv += h1[i * M + j] * v[j]
        res1 = max(res1, abs(hv - e0 * v[i]))
    _ = log.close("slater ground vector satisfies h v = e0 v", res1, 0.0, TOLERANCE)

    # Classical sweep: adjacent rotations on (j-1, j) zero the amplitudes from the last mode down, leaving mode 0.
    var w = v.copy()
    var rot_p = List[Int]()
    var rot_t = List[Float64]()
    var jj = M - 1
    while jj > 0:
        var p = jj - 1
        var r = sqrt(w[p] * w[p] + w[jj] * w[jj])
        if r > 0.0:
            var cs = w[p] / r
            var sn = w[jj] / r
            var wp = cs * w[p] + sn * w[jj]
            var wq = -sn * w[p] + cs * w[jj]
            w[p] = wp
            w[jj] = wq
            rot_p.append(p)
            rot_t.append(2.0 * atan2(sn, cs))
        jj -= 1
    _ = log.close("sweep leaves the whole weight on mode 0", abs(w[0]), 1.0, TOLERANCE)
    print("Slater rotations:", len(rot_p), " (two-mode Givens on", M, "modes)")

    # Circuit: X on mode 0, then the rotations in reverse order of the sweep (the inverse of the sweep).
    var circ1 = Circuit(M, name="slater_one_particle")
    circ1.x(0)
    var k = len(rot_p) - 1
    while k >= 0:
        _excitation_givens(circ1, rot_p[k], rot_p[k] + 1, rot_t[k])
        k -= 1
    var sv1 = evolve(circ1)

    var ov1_re = 0.0
    var ov1_im = 0.0
    var inside1 = 0.0
    for mu in range(M):
        var idx = 1 << mu
        ov1_re += v[mu] * sv1.re[idx]
        ov1_im += v[mu] * sv1.im[idx]
        inside1 += sv1.re[idx] * sv1.re[idx] + sv1.im[idx] * sv1.im[idx]
    var total1 = 0.0
    for i in range(1 << M):
        total1 += sv1.re[i] * sv1.re[i] + sv1.im[i] * sv1.im[i]
    var fid1 = ov1_re * ov1_re + ov1_im * ov1_im
    print("Slater fidelity with the exact ground state:", fid1)
    _ = log.close("slater fidelity with the exact ground state", fid1, 1.0, TOLERANCE)
    _ = log.close("slater weight outside the one-particle sector", total1 - inside1, 0.0, TOLERANCE)

    # ======================= Part 2: general quadratic Hamiltonian, two modes =======================
    var n = 2
    var m2 = 2 * n
    var h2: List[Float64] = [1.0, 0.5, 0.5, -0.5]
    var delta: List[Float64] = [0.0, 0.3, -0.3, 0.0]
    var const_part = (h2[0] + h2[3]) / 2.0  # tr(h)/2

    # Majorana matrix A. The notebook prints A in the ordering (x_0, x_1, p_0, p_1); ours is (x_0, p_0, x_1, p_1).
    var A = majorana_matrix(h2, delta, n)
    var printed_A: List[Float64] = [
        0.0, 0.0, 1.0, 0.2,
        0.0, 0.0, 0.8, -0.5,
        -1.0, -0.8, 0.0, 0.0,
        -0.2, 0.5, 0.0, 0.0,
    ]
    var perm: List[Int] = [0, 2, 1, 3]
    var err_A = 0.0
    for i in range(m2):
        for j in range(m2):
            err_A = max(err_A, abs(A[perm[i] * m2 + perm[j]] - printed_A[i * m2 + j]))
    _ = log.close("majorana matrix A matches the notebook print", err_A, 0.0, TOLERANCE)

    # Canonical frame: pairs (v_k, A v_k / b_k) from the eigenvectors of -A^2 = A^T A, one pair per energy b_k.
    var S = _matmul(_transpose(A, m2), A, m2)
    var eigS = jacobi_eigh(S, m2)
    var Z = List[Float64](length=m2 * m2, fill=0.0)
    var bs = List[Float64]()
    for kp in range(n):
        var col = m2 - 1 - 2 * kp
        var b = sqrt(eigS.values[col])
        var vk = eigS.vector(col)
        var uk = _matvec(A, vk, m2)
        for r in range(m2):
            Z[r * m2 + 2 * kp] = vk[r]
            Z[r * m2 + 2 * kp + 1] = uk[r] / b
        bs.append(b)
    if _det(Z, m2) < 0.0:
        # A frame of det -1 is not a Gaussian unitary; flipping one v flips the sign of its canonical energy.
        for r in range(m2):
            Z[r * m2 + m2 - 2] = -Z[r * m2 + m2 - 2]
    var ZT = _transpose(Z, m2)
    var Ac = _matmul(_matmul(ZT, A, m2), Z, m2)
    var offblock = 0.0
    for i in range(m2):
        for j in range(m2):
            if (i // 2) != (j // 2):
                offblock = max(offblock, abs(Ac[i * m2 + j]))
    _ = log.close("canonical frame Z^T A Z is block diagonal", offblock, 0.0, TOLERANCE)
    var bp = List[Float64]()  # signed canonical energies b'_k: Z^T A Z = blockdiag(-b'_k [[0,1],[-1,0]])
    for kp in range(n):
        bp.append(-Ac[(2 * kp) * m2 + 2 * kp + 1])
    print("canonical energies:", bp[0], bp[1])
    _ = log.close("canonical energy 1 matches the notebook E (1.292)", abs(bp[0]), 1.292, 5e-4)
    _ = log.close("canonical energy 2 matches the notebook E (0.511)", abs(bp[1]), 0.511, 5e-4)

    # Frame as planar rotations: M = Z^T is reduced to the identity by adjacent-plane rotations (QR-style sweep).
    var W = ZT.copy()
    var rp = List[Int]()
    var rt = List[Float64]()
    var jc = 0
    while jc < m2 - 1:
        var kk = m2 - 1
        while kk > jc:
            var p = kk - 1
            var r = sqrt(W[p * m2 + jc] * W[p * m2 + jc] + W[kk * m2 + jc] * W[kk * m2 + jc])
            if r > 0.0:
                var cs = W[p * m2 + jc] / r
                var sn = -W[kk * m2 + jc] / r
                for col in range(m2):
                    var wp = W[p * m2 + col]
                    var wq = W[kk * m2 + col]
                    W[p * m2 + col] = cs * wp - sn * wq
                    W[kk * m2 + col] = sn * wp + cs * wq
                rp.append(p)
                rt.append(atan2(sn, cs))
            kk -= 1
        jc += 1
    var err_W = 0.0
    for i in range(m2):
        for j in range(m2):
            var target = 1.0 if i == j else 0.0
            err_W = max(err_W, abs(W[i * m2 + j] - target))
    _ = log.close("reduced frame is the identity", err_W, 0.0, TOLERANCE)
    print("planar Majorana rotations:", len(rp))

    # Circuit: occupations of the canonical modes (X where b'_k > 0), then the planar rotations in sweep order.
    # Plane (2j, 2j+1) is the mode j phase rotation RZ on qubit j; plane (2j+1, 2j+2) is RXX on qubits (j, j+1).
    var circ2 = Circuit(n, name="fermionic_gaussian")
    for mu in range(n):
        if bp[mu] > 0.0:
            circ2.x(mu)
    for g in range(len(rp)):
        var p = rp[g]
        if p % 2 == 0:
            circ2.rz(rt[g], p // 2)
        else:
            circ2.rxx(rt[g], p // 2, p // 2 + 1)
    var sv2 = evolve(circ2)

    # Exact reference: the Fock-space Hamiltonian diagonalized directly.
    var dim = 1 << n
    var H2 = fock_hamiltonian(h2, delta, n)
    var eigF = jacobi_eigh(H2, dim)
    var printed_E: List[Float64] = [-0.651388, -0.140512, 0.640512, 1.151388]
    for lev in range(dim):
        _ = log.close(
            "fock spectrum level " + String(lev) + " matches the notebook print",
            eigF.values[lev], printed_E[lev], 5e-7,
        )
    var b1 = bs[0]
    var b2 = bs[1]
    _ = log.close("fock ground energy equals tr(h)/2 - (b1 + b2)/2", eigF.values[0], const_part - (b1 + b2) / 2.0, TOLERANCE)
    _ = log.close("fock level 2 equals tr(h)/2 - (b1 - b2)/2", eigF.values[1], const_part - (b1 - b2) / 2.0, TOLERANCE)
    _ = log.close("fock level 3 equals tr(h)/2 + (b1 - b2)/2", eigF.values[2], const_part + (b1 - b2) / 2.0, TOLERANCE)
    _ = log.close("fock top level equals tr(h)/2 + (b1 + b2)/2", eigF.values[3], const_part + (b1 + b2) / 2.0, TOLERANCE)

    var gs = eigF.vector(0)
    var ov2_re = 0.0
    var ov2_im = 0.0
    for i in range(dim):
        ov2_re += gs[i] * sv2.re[i]
        ov2_im += gs[i] * sv2.im[i]
    var fid2 = ov2_re * ov2_re + ov2_im * ov2_im
    print("pairing-case fidelity with the exact ground state:", fid2)
    _ = log.close("pairing ground state fidelity with the exact Fock ground state", fid2, 1.0, TOLERANCE)

    var printed_gs_abs: List[Float64] = [0.0, 0.289784, 0.957092, 0.0]
    for i in range(dim):
        var mag = sqrt(sv2.re[i] * sv2.re[i] + sv2.im[i] * sv2.im[i])
        _ = log.close(
            "ground state amplitude " + String(i) + " magnitude matches the notebook print",
            mag, printed_gs_abs[i], 5e-7,
        )
    # Relative phase of the two odd-sector amplitudes: Re(psi_1 conj(psi_2)) < 0, as the notebook prints.
    var rel_re = sv2.re[1] * sv2.re[2] + sv2.im[1] * sv2.im[2]
    var rel_im = sv2.im[1] * sv2.re[2] - sv2.re[1] * sv2.im[2]
    _ = log.check(
        "odd-sector amplitudes have the notebook relative sign",
        rel_re < 0.0 and abs(rel_im) < TOLERANCE,
        "re=" + String(rel_re) + " im=" + String(rel_im),
    )

    log.finish()
