"""Hamiltonian simulation with Suzuki-Trotter, exponentiate and qDRIFT for two coupled spins.

Reproduces: Classiq library - Hamiltonian Simulation
Source: https://github.com/Classiq/classiq-library/blob/main/algorithms/hamiltonian_simulation/hamiltonian_simulation_guide/hamiltonian_simulation_guide.ipynb (Classiq Library, MIT)
Framework: Classiq - https://github.com/Classiq/classiq-library

The notebook evolves H = 0.3 Z0 Z1 + 0.7 X0 + 0.2 X1 from |00> and plots the magnetization M(t) = (<Z0> + <Z1>)/2 on 100 times in [0, 2]: first-order Suzuki-Trotter with 30 steps, exponentiate (one step), and qDRIFT with N = 288 random Pauli rotations, against the exact evolution. The notebook prints no numbers, so the checks use analytic values and dense brute-force matrices, not the figure. QUMO builds the same circuits with trotter() and append_pauli_exponential(). The exact qDRIFT ensemble average is computed on the density-matrix simulator, with one two-qubit mixed-unitary channel per step (Kraus operators sqrt(p_j) exp(-i lambda t P_j / N)). A seeded sample of 2000 qDRIFT circuits per time is checked against the exact ensemble average.
Checks: the Hamiltonian and observable encodings; QUMO exact evolution, the 10-step amplitudes, and the 30-step Trotter and one-step exponentiate curves against dense products; the Lie-Trotter commutator bound and the qDRIFT bound 2 lambda^2 t^2 / N on the grid; the sampling frequencies of the qDRIFT terms; every sampled circuit against the dense product of its sequence; and the sample means at t = 0.5, 1, 2 within four standard errors.

Deviations from the notebook:
- Classiq synthesis, observe and its cloud simulator are replaced by QUMO's statevector and density-matrix simulators. exponentiate() is trotter(H, t, steps=1), which is the notebook's order-1, one-repetition default.
- qDRIFT is not in the QUMO library. The sampler (probabilities h_j / lambda with lambda = 1.2, QUMO Rng seed 2026) and the circuit assembly are in this file, built from existing Pauli-exponential gates. The notebook sets no seed for its random sampling, so its curve is one draw from the ensemble; the density-matrix check computes that ensemble exactly.
- The block-encoding section (qubitization, QSVT, GQSP) is text only in the notebook and is not reproduced.
- The notebook's term order (ZZ, X0, X1) is kept, since the Trotter error depends on it. Qubit 0 is the least significant bit, as in QUMO; the dense reference uses the same convention.
- The bounds are analytic: the Lie-Trotter bound is (t^2 / 2r) times the sum of pairwise commutator norms (0.54 here), doubled for an expectation of an observable with norm 1, and the qDRIFT bound is the notebook's own 2 lambda^2 t^2 / N. A 1e-12 floating-point slack applies to both.
- The sampled-mean tolerance is four standard errors of the sample mean, fixed before the run; the seed is fixed, so the run is deterministic.

Run: .venv/bin/mojo run -I . examples/chemistry_and_physics/hamiltonian_simulation_guide.mojo
example-meta: {"framework": "classiq", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from std.math import cos, sin, sqrt
from qumo.ir.circuit import Circuit
from qumo.ir.kraus import KrausChannel
from qumo.ir.param import Param
from qumo.ir.pauli import PauliSum, PauliTerm
from qumo.sim.density import density_evolve
from qumo.sim.rng import Rng
from qumo.sim.runner import evolve
from qumo.sim.statevector import StateVector
from qumo.algorithms.hamiltonians import dense_matrix, exact_evolve
from qumo.algorithms.library import trotter, append_pauli_exponential
from qumo.testing.checks import CheckLog


comptime TERMS = 3  # H = sum_j h_j P_j with P_0 = Z0 Z1, P_1 = X0, P_2 = X1
comptime LAMBDA = 1.2  # lambda = sum_j |h_j| = 0.3 + 0.7 + 0.2
comptime QDRIFT_N = 288  # the notebook's qDRIFT depth
comptime SAMPLES = 2000  # seeded qDRIFT circuits per sampled time
comptime SEED = 2026
comptime GRID = 100  # np.linspace(0, 2, 100)
comptime TROTTER_STEPS = 30  # the notebook's magnetization figure
comptime COMMUTATOR_SUM = 0.54  # sum over pairs j < k of ||[h_j P_j, h_k P_k]|| = 2 (0.3 * 0.7) + 2 (0.3 * 0.2)
comptime ROUNDING = 1e-12


# ---------------------------------------------------------------------------
# Notebook objects as QUMO objects
# ---------------------------------------------------------------------------
def coefficient(which: Int) -> Float64:
    if which == 0:
        return 0.3
    if which == 1:
        return 0.7
    return 0.2


def hamiltonian() -> PauliSum:
    """0.3 Z0 Z1 + 0.7 X0 + 0.2 X1 in the notebook's term order (x and z are qubit bit masks)."""
    var h = PauliSum()
    h.add(0.3, UInt64(0), UInt64(3))
    h.add(0.7, UInt64(1), UInt64(0))
    h.add(0.2, UInt64(2), UInt64(0))
    return h^


def magnetization_observable() -> PauliSum:
    """M = 0.5 Z0 + 0.5 Z1 (the notebook's magnetization_HAMILTONIAN)."""
    var m = PauliSum()
    m.add(0.5, UInt64(0), UInt64(1))
    m.add(0.5, UInt64(0), UInt64(2))
    return m^


def pauli_term(which: Int) -> PauliTerm:
    """Unit-coefficient term of the sampled qDRIFT step: masks of Z0 Z1, X0, X1."""
    if which == 0:
        return PauliTerm(1.0, UInt64(0), UInt64(3))
    if which == 1:
        return PauliTerm(1.0, UInt64(1), UInt64(0))
    return PauliTerm(1.0, UInt64(2), UInt64(0))


# ---------------------------------------------------------------------------
# Brute-force reference: real 4x4 Pauli strings and complex 4x4 algebra, written
# from the 2x2 Pauli matrices (independent of QUMO's bit-packed code paths)
# ---------------------------------------------------------------------------
def kron(a: List[Float64], b: List[Float64]) -> List[Float64]:
    """Real 4x4 a (x) b: a acts on qubit 1 (the high bit), b on qubit 0 (the low bit)."""
    var out = List[Float64](length=16, fill=0.0)
    for r1 in range(2):
        for r0 in range(2):
            for c1 in range(2):
                for c0 in range(2):
                    out[(2 * r1 + r0) * 4 + 2 * c1 + c0] = a[2 * r1 + c1] * b[2 * r0 + c0]
    return out^


def pauli_string(which: Int) -> List[Float64]:
    var i2: List[Float64] = [1.0, 0.0, 0.0, 1.0]
    var x2: List[Float64] = [0.0, 1.0, 1.0, 0.0]
    var z2: List[Float64] = [1.0, 0.0, 0.0, -1.0]
    if which == 0:
        return kron(z2, z2)
    if which == 1:
        return kron(i2, x2)
    return kron(x2, i2)


def hamiltonian_dense() -> List[Float64]:
    var h = List[Float64](length=16, fill=0.0)
    for j in range(TERMS):
        var p = pauli_string(j)
        for k in range(16):
            h[k] += coefficient(j) * p[k]
    return h^


def mag_weight(i: Int) -> Float64:
    """Diagonal of M = (Z0 + Z1) / 2 in the basis |b1 b0>: +1 on |00>, 0 on |01> and |10>, -1 on |11>."""
    if i == 0:
        return 1.0
    if i == 3:
        return -1.0
    return 0.0


def magnetization_state(vr: List[Float64], vi: List[Float64]) -> Float64:
    var m = 0.0
    for i in range(4):
        m += mag_weight(i) * (vr[i] * vr[i] + vi[i] * vi[i])
    return m


def magnetization_rho(rr: List[Float64], ri: List[Float64]) -> Float64:
    var m = 0.0
    for i in range(4):
        m += mag_weight(i) * rr[i * 4 + i]
    return m


def rotation(which: Int, theta: Float64) -> Tuple[List[Float64], List[Float64]]:
    """Unitary exp(-i theta P_j) = cos(theta) I - i sin(theta) P_j, since P_j^2 = I."""
    var p = pauli_string(which)
    var re = List[Float64](length=16, fill=0.0)
    var im = List[Float64](length=16, fill=0.0)
    for i in range(4):
        re[i * 4 + i] = cos(theta)
    for k in range(16):
        im[k] = -sin(theta) * p[k]
    return (re^, im^)


def apply_to_state(ur: List[Float64], ui: List[Float64], vr: List[Float64], vi: List[Float64]) -> Tuple[List[Float64], List[Float64]]:
    var out_r = List[Float64](length=4, fill=0.0)
    var out_i = List[Float64](length=4, fill=0.0)
    for r in range(4):
        for c in range(4):
            out_r[r] += ur[r * 4 + c] * vr[c] - ui[r * 4 + c] * vi[c]
            out_i[r] += ur[r * 4 + c] * vi[c] + ui[r * 4 + c] * vr[c]
    return (out_r^, out_i^)


def cmul(ar: List[Float64], ai: List[Float64], br: List[Float64], bi: List[Float64]) -> Tuple[List[Float64], List[Float64]]:
    var cr = List[Float64](length=16, fill=0.0)
    var ci = List[Float64](length=16, fill=0.0)
    for r in range(4):
        for c in range(4):
            var sr = 0.0
            var si = 0.0
            for k in range(4):
                sr += ar[r * 4 + k] * br[k * 4 + c] - ai[r * 4 + k] * bi[k * 4 + c]
                si += ar[r * 4 + k] * bi[k * 4 + c] + ai[r * 4 + k] * br[k * 4 + c]
            cr[r * 4 + c] = sr
            ci[r * 4 + c] = si
    return (cr^, ci^)


def dagger(ar: List[Float64], ai: List[Float64]) -> Tuple[List[Float64], List[Float64]]:
    var dr = List[Float64](length=16, fill=0.0)
    var di = List[Float64](length=16, fill=0.0)
    for r in range(4):
        for c in range(4):
            dr[r * 4 + c] = ar[c * 4 + r]
            di[r * 4 + c] = -ai[c * 4 + r]
    return (dr^, di^)


def exact_state(t: Float64) -> Tuple[List[Float64], List[Float64]]:
    """Exact state, the Taylor series of exp(-i H t) applied to |00> with the dense H; 60 terms cover ||H|| t <= 2.4 to machine precision."""
    var h = hamiltonian_dense()
    var zeros = List[Float64](length=16, fill=0.0)
    var sr = List[Float64](length=4, fill=0.0)
    sr[0] = 1.0
    var si = List[Float64](length=4, fill=0.0)
    var tr = sr.copy()
    var ti = si.copy()
    for k in range(1, 60):
        var w = apply_to_state(h, zeros, tr, ti)
        var c = t / Float64(k)
        var nr = List[Float64](length=4, fill=0.0)
        var ni = List[Float64](length=4, fill=0.0)
        for i in range(4):
            nr[i] = c * w[1][i]  # term <- (-i c) H term
            ni[i] = -c * w[0][i]
        tr = nr^
        ti = ni^
        for i in range(4):
            sr[i] += tr[i]
            si[i] += ti[i]
    return (sr^, si^)


def trotter_state(t: Float64, steps: Int) -> Tuple[List[Float64], List[Float64]]:
    """(prod_j exp(-i h_j P_j t / r))^r |00>, the first-order product in the notebook's term order."""
    var vr = List[Float64](length=4, fill=0.0)
    vr[0] = 1.0
    var vi = List[Float64](length=4, fill=0.0)
    var dt = t / Float64(steps)
    for _ in range(steps):
        for j in range(TERMS):
            var u = rotation(j, coefficient(j) * dt)
            var nxt = apply_to_state(u[0], u[1], vr, vi)
            vr = nxt[0].copy()
            vi = nxt[1].copy()
    return (vr^, vi^)


def qdrift_ensemble_dense(t: Float64) -> Float64:
    """Tr(M Phi^N(rho_0)) with Phi(rho) = sum_j p_j U_j rho U_j^dag, p_j = h_j / lambda, U_j = exp(-i (lambda t / N) P_j).

    This is the exact expectation over the qDRIFT sampling, since the N draws are independent."""
    var theta = LAMBDA * t / Float64(QDRIFT_N)
    var rr = List[Float64](length=16, fill=0.0)
    rr[0] = 1.0
    var ri = List[Float64](length=16, fill=0.0)
    for _ in range(QDRIFT_N):
        var nr = List[Float64](length=16, fill=0.0)
        var ni = List[Float64](length=16, fill=0.0)
        for j in range(TERMS):
            var u = rotation(j, theta)
            var ud = dagger(u[0], u[1])
            var a = cmul(u[0], u[1], rr, ri)
            var b = cmul(a[0], a[1], ud[0], ud[1])
            var p = coefficient(j) / LAMBDA
            for k in range(16):
                nr[k] += p * b[0][k]
                ni[k] += p * b[1][k]
        rr = nr^
        ri = ni^
    return magnetization_rho(rr, ri)


# ---------------------------------------------------------------------------
# QUMO side: the same circuits and the density-matrix ensemble
# ---------------------------------------------------------------------------
def qdrift_channel(t: Float64) raises -> KrausChannel:
    """One qDRIFT step as a single two-qubit channel: Kraus operators sqrt(p_j) U_j, so that sum_j p_j U_j rho U_j^dag."""
    var theta = LAMBDA * t / Float64(QDRIFT_N)
    var ch = KrausChannel("qdrift_step", 2)
    for j in range(TERMS):
        var u = rotation(j, theta)
        var w = sqrt(coefficient(j) / LAMBDA)
        var re = List[Float64](length=16, fill=0.0)
        var im = List[Float64](length=16, fill=0.0)
        for k in range(16):
            re[k] = w * u[0][k]
            im[k] = w * u[1][k]
        ch.add(re^, im^)
    return ch^


def qdrift_ensemble_qumo(t: Float64) raises -> Float64:
    """Exact qDRIFT ensemble average on QUMO's density-matrix simulator: N copies of the step channel."""
    var ch = qdrift_channel(t)
    var c = Circuit(2, name="qdrift_ensemble")
    for _ in range(QDRIFT_N):
        c.apply_channel(ch, 0, 1)
    return density_evolve(c).expectation(magnetization_observable())


def draw_term(mut rng: Rng) -> Int:
    """Sample j with probability h_j / lambda: cumulative thresholds 0.3, 1.0 and 1.2 on u * lambda."""
    var u = rng.uniform() * LAMBDA
    if u < 0.3:
        return 0
    if u < 1.0:
        return 1
    return 2


def sequence_circuit(seq: List[Int], t: Float64) raises -> Circuit:
    """The sampled qDRIFT circuit: exp(-i (lambda t / N) P_j) for each j in the sequence, in order."""
    var c = Circuit(2, name="qdrift_sample")
    var step = Param(LAMBDA * t / Float64(QDRIFT_N))
    for j in seq:
        append_pauli_exponential(c, pauli_term(j), step)
    return c^


def sequence_state(seq: List[Int], t: Float64) -> Tuple[List[Float64], List[Float64]]:
    """Dense state of the same sampled sequence, one 4x4 unitary at a time."""
    var vr = List[Float64](length=4, fill=0.0)
    vr[0] = 1.0
    var vi = List[Float64](length=4, fill=0.0)
    var theta = LAMBDA * t / Float64(QDRIFT_N)
    for j in seq:
        var u = rotation(j, theta)
        var nxt = apply_to_state(u[0], u[1], vr, vi)
        vr = nxt[0].copy()
        vi = nxt[1].copy()
    return (vr^, vi^)


def main() raises:
    var log = CheckLog("classiq/hamiltonian_simulation_guide")
    var h = hamiltonian()
    var obs = magnetization_observable()

    # ---- encodings ----
    var dense_h = dense_matrix(h, 2)
    var ref_h = hamiltonian_dense()
    var h_dev = 0.0
    var h_imag = 0.0
    for k in range(16):
        h_dev = max(h_dev, abs(dense_h[0][k] - ref_h[k]))
        h_imag = max(h_imag, abs(dense_h[1][k]))
    _ = log.close("H dense matrix equals 0.3 ZZ + 0.7 X0 + 0.2 X1", h_dev, 0.0, 1e-12)
    _ = log.close("H dense matrix is real", h_imag, 0.0, 1e-12)
    var dense_m = dense_matrix(obs, 2)
    var m_dev = 0.0
    for k in range(16):
        var r = k // 4
        var c = k % 4
        var want = mag_weight(r) if r == c else 0.0
        m_dev = max(m_dev, abs(dense_m[0][k] - want))
    _ = log.close("observable 0.5 Z0 + 0.5 Z1 is diag(1, 0, 0, -1)", m_dev, 0.0, 1e-12)

    # ---- grid: exact, Trotter, exponentiate, qDRIFT ----
    var dev_exact = 0.0
    var dev_trot = 0.0
    var dev_exp = 0.0
    var dev_qd = 0.0
    var err_trot = 0.0
    var err_exp = 0.0
    var err_qd = 0.0
    var viol_trot = 0
    var viol_qd = 0
    var m_zero = 0.0
    var m_end = List[Float64](length=4, fill=0.0)
    for k in range(GRID):
        var t = 2.0 * Float64(k) / Float64(GRID - 1)
        var exact = exact_state(t)
        var m_exact = magnetization_state(exact[0], exact[1])
        var m_exact_q = exact_evolve(h, StateVector(2), t).expectation(obs)
        dev_exact = max(dev_exact, abs(m_exact_q - m_exact))
        if k == 0:
            m_zero = m_exact_q

        var trot_ref = trotter_state(t, TROTTER_STEPS)
        var m_trot = magnetization_state(trot_ref[0], trot_ref[1])
        var m_trot_q = evolve(trotter(h, t, TROTTER_STEPS, 1, 2)).expectation(obs)
        dev_trot = max(dev_trot, abs(m_trot_q - m_trot))
        var e_trot = abs(m_trot_q - m_exact)
        err_trot = max(err_trot, e_trot)
        var trot_bound = COMMUTATOR_SUM * t * t / Float64(TROTTER_STEPS)  # 2 * (t^2 / 2r) * sum of commutator norms
        if e_trot > trot_bound + ROUNDING:
            viol_trot += 1

        var exp_ref = trotter_state(t, 1)
        var m_exp = magnetization_state(exp_ref[0], exp_ref[1])
        var m_exp_q = evolve(trotter(h, t, 1, 1, 2)).expectation(obs)
        dev_exp = max(dev_exp, abs(m_exp_q - m_exp))
        err_exp = max(err_exp, abs(m_exp_q - m_exact))

        var m_qd_q = qdrift_ensemble_qumo(t)
        var m_qd = qdrift_ensemble_dense(t)
        dev_qd = max(dev_qd, abs(m_qd_q - m_qd))
        var e_qd = abs(m_qd_q - m_exact)
        err_qd = max(err_qd, e_qd)
        var qd_bound = 2.0 * LAMBDA * LAMBDA * t * t / Float64(QDRIFT_N)  # Campbell's bound with ||M|| = 1
        if e_qd > qd_bound + ROUNDING:
            viol_qd += 1

        if k == GRID - 1:
            m_end[0] = m_exact_q
            m_end[1] = m_trot_q
            m_end[2] = m_exp_q
            m_end[3] = m_qd_q

    _ = log.close("QUMO exact evolution has M(0) = 1", m_zero, 1.0, 1e-12)
    _ = log.close("QUMO exact evolution matches dense exp(-iHt) on the 100-point grid (max dM)", dev_exact, 0.0, 1e-9)

    # ---- Suzuki-Trotter, the notebook's first cell: 10 steps at t = 1 (amplitudes) ----
    var psi10 = evolve(trotter(h, 1.0, 10, 1, 2))
    var ref10 = trotter_state(1.0, 10)
    var amp_dev = 0.0
    for i in range(4):
        amp_dev = max(amp_dev, abs(psi10.re[i] - ref10[0][i]))
        amp_dev = max(amp_dev, abs(psi10.im[i] - ref10[1][i]))
    _ = log.close("Trotter 10 steps at t = 1 equals the dense product (max amplitude)", amp_dev, 0.0, 1e-9)

    _ = log.close("Trotter 30 steps matches the dense product on the grid (max dM)", dev_trot, 0.0, 1e-9)
    _ = log.check("Trotter 30 steps within the Lie-Trotter commutator bound on the grid", viol_trot == 0, "violations=" + String(viol_trot) + " max error=" + String(err_trot))
    _ = log.close("exponentiate (one step) matches the dense product on the grid (max dM)", dev_exp, 0.0, 1e-9)
    _ = log.check("one-step exponentiate is less accurate than 30 Trotter steps", err_exp > err_trot, "max error exponentiate=" + String(err_exp) + " trotter=" + String(err_trot))

    # ---- qDRIFT: exact ensemble on the density matrix, then seeded samples ----
    _ = log.close("qDRIFT ensemble on the density matrix matches dense Phi^N on the grid (max dM)", dev_qd, 0.0, 1e-9)
    _ = log.check("qDRIFT ensemble within 2 lambda^2 t^2 / N of exact evolution on the grid", viol_qd == 0, "violations=" + String(viol_qd) + " max error=" + String(err_qd))

    var rng = Rng(UInt64(SEED))
    var counts = List[Int](length=TERMS, fill=0)
    var draws = 0
    var seq_dev = 0.0
    var sample_times: List[Float64] = [0.5, 1.0, 2.0]
    var means = List[Float64](length=len(sample_times), fill=0.0)
    var ses = List[Float64](length=len(sample_times), fill=0.0)
    for ti in range(len(sample_times)):
        var t = sample_times[ti]
        var total = 0.0
        var total_sq = 0.0
        for _ in range(SAMPLES):
            var seq = List[Int](capacity=QDRIFT_N)
            for _ in range(QDRIFT_N):
                var j = draw_term(rng)
                seq.append(j)
                counts[j] += 1
            draws += QDRIFT_N
            var q = evolve(sequence_circuit(seq, t)).expectation(obs)
            var dense = sequence_state(seq, t)
            seq_dev = max(seq_dev, abs(q - magnetization_state(dense[0], dense[1])))
            total += q
            total_sq += q * q
        var n = Float64(SAMPLES)
        var mean = total / n
        var sample_var = (total_sq - n * mean * mean) / (n - 1.0)
        means[ti] = mean
        ses[ti] = sqrt(sample_var / n)

    _ = log.close("every sampled qDRIFT circuit equals the dense product of its sequence (max dM)", seq_dev, 0.0, 1e-9)

    var worst_z = 0.0
    var total_draws = Float64(draws)
    for j in range(TERMS):
        var p = coefficient(j) / LAMBDA
        var f = Float64(counts[j]) / total_draws
        var sigma = sqrt(p * (1.0 - p) / total_draws)
        worst_z = max(worst_z, abs(f - p) / sigma)
    _ = log.check("sampled qDRIFT terms follow h_j / lambda within four sigma", worst_z <= 4.0, "worst z=" + String(worst_z) + " draws=" + String(draws))

    for ti in range(len(sample_times)):
        var t = sample_times[ti]
        var exact_ens = qdrift_ensemble_dense(t)
        var label = "seeded qDRIFT sample mean at t = " + String(t) + " within four standard errors of the ensemble"
        _ = log.close(label, means[ti], exact_ens, 4.0 * ses[ti])

    print("M(2) exact:", m_end[0], "Trotter 30:", m_end[1], "exponentiate:", m_end[2], "qDRIFT ensemble:", m_end[3])
    print("max error over grid: Trotter 30:", err_trot, "exponentiate:", err_exp, "qDRIFT ensemble:", err_qd)

    log.finish()
