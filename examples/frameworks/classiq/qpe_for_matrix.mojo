"""Quantum phase estimation of a 2-qubit Hermitian matrix: eigenvalues from a 7-qubit phase register.

Reproduces: Classiq library - Quantum Phase Estimation for Solving Matrix Eigenvalues
Source: https://github.com/Classiq/classiq-library/blob/main/algorithms/quantum_phase_estimation/qpe_for_matrix/qpe_for_matrix.ipynb (Classiq Library, MIT)

The notebook normalises M by c = 2 sum|Pauli coefficients| and runs QPE on U = exp(2 pi i M / c) (its exact model); the two most probable bins give 0.8993 and 0.4038 for the eigenvalues 0.9 and 0.4. QUMO has no matrix-gate primitive, so the controlled powers U^(2^j) are built exactly in the eigenbasis of M (diagonal phase gates), with the basis change folded into the state preparation. The initial vector is the notebook's np.random.rand(4) with seed 8, copied from its printed output. The notebook's Suzuki-Trotter model is not ported. Checks: the Pauli expansion, the normalisation and resolution, the 128-bin distribution against the analytic QPE kernel, the notebook's printed estimates, and 10000 sampled shots.

Run: .venv/bin/mojo run -I . examples/frameworks/classiq/qpe_for_matrix.mojo
example-meta: {"framework": "classiq", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from std.math import pi, sqrt, cos, sin
from qumo.ir.circuit import Circuit
from qumo.ir.pauli import PauliSum
from qumo.ir.mc import append_mcp
from qumo.algorithms.library import iqft
from qumo.algorithms.qml import append_amplitude_embedding_real
from qumo.algorithms.linalg import jacobi_eigh
from qumo.algorithms.hamiltonians import dense_matrix, exact_spectrum
from qumo.sim.runner import evolve, sample, bitstring
from qumo.testing.checks import CheckLog


def _popcount(v: Int) -> Int:
    var x = v
    var n = 0
    while x != 0:
        n += x & 1
        x >>= 1
    return n


def pauli_from_matrix(m: List[Float64], n: Int) -> PauliSum:
    """Pauli expansion M = sum_P c_P P of a real symmetric 2^n x 2^n matrix.

    P = i^{|x&z|} X^x Z^z (the PauliTerm convention), c_P = Tr(P M) / 2^n.
    Strings with an odd number of Y are skipped: they are imaginary and
    antisymmetric, so their coefficient is zero for a real symmetric M."""
    var dim = 1 << n
    var h = PauliSum()
    for x in range(dim):
        for z in range(dim):
            var ny = _popcount(x & z)
            if ny % 2 == 1:
                continue
            var ph = 1.0 if (ny // 2) % 2 == 0 else -1.0
            var tr = 0.0
            for s in range(dim):
                var sign = 1.0 if _popcount(z & s) % 2 == 0 else -1.0
                tr += ph * sign * m[s * dim + (s ^ x)]
            var coeff = tr / Float64(dim)
            if abs(coeff) > 1e-14:
                h.add(coeff, UInt64(x), UInt64(z))
    return h^


def build_qpe(n_eval: Int, amps: List[Float64], phases: List[Float64], measure: Bool) raises -> Circuit:
    """QPE of U = V diag(exp(i phases[k])) V^T on 2 system qubits.

    Layout: eval qubits 0..n_eval-1 (little-endian, readout k ~ theta * 2^n_eval),
    system qubits n_eval, n_eval+1. amps = V^T psi (the eigenbasis change of
    U^(2^j) cancels between neighbours, so only the state preparation keeps it).
    Controlled U^(2^j) from eval qubit j: the diagonal phase
    phi(k) = g + a s0 + b s1 + c s0 s1 (s0, s1 the bits of k) is exact for four
    eigenphases, so it needs one P, two CP and one doubly controlled phase."""
    var nsys = 2
    var nclb = n_eval if measure else 0
    var circ = Circuit(n_eval + nsys, nclb, "qpe_exact")
    var wires = List[Int]()
    wires.append(n_eval + 1)
    wires.append(n_eval)
    append_amplitude_embedding_real(circ, amps, wires)
    for j in range(n_eval):
        circ.h(j)
    var g = phases[0]
    var a = phases[1] - phases[0]
    var b = phases[2] - phases[0]
    var cc = phases[3] - phases[1] - phases[2] + phases[0]
    for j in range(n_eval):
        var f = Float64(1 << j)
        circ.p(f * g, j)
        circ.cp(f * a, j, n_eval)
        circ.cp(f * b, j, n_eval + 1)
        var ctrl = List[Int]()
        ctrl.append(j)
        ctrl.append(n_eval)
        append_mcp(circ, f * cc, ctrl, n_eval + 1)
    circ.compose(iqft(n_eval))
    if measure:
        for j in range(n_eval):
            circ.measure(j, j)
    return circ^


def alpha_sq(theta: Float64, k: Int, n: Int) -> Float64:
    """Ideal QPE kernel |(1/N) sum_j exp(2 pi i j (theta - k/N))|^2, N = 2^n, by direct summation."""
    var nn = 1 << n
    var d = theta - Float64(k) / Float64(nn)
    var re = 0.0
    var im = 0.0
    for j in range(nn):
        var ang = 2.0 * pi * Float64(j) * d
        re += cos(ang)
        im += sin(ang)
    return (re * re + im * im) / Float64(nn * nn)


def main() raises:
    var log = CheckLog("classiq/qpe_for_matrix")
    comptime N_QPE = 7
    comptime SHOTS = 10000
    var nk = 1 << N_QPE

    # The matrix of the notebook (symmetrised as in the notebook).
    var m_raw: List[Float64] = [
        0.38891555, 0.23315811, 0.21499372, 0.06119557,
        0.23315811, 0.44435328, 0.25197881, -0.13087919,
        0.21499372, 0.25197881, 0.44116509, -0.01961855,
        0.06119557, -0.13087919, -0.01961855, 0.32556608,
    ]
    var m = List[Float64](length=16, fill=0.0)
    for i in range(4):
        for j in range(4):
            m[i * 4 + j] = 0.5 * (m_raw[i * 4 + j] + m_raw[j * 4 + i])

    # ---- Pauli expansion and normalisation (notebook: matrix_to_pauli_operator, get_normalization) ----
    var h = pauli_from_matrix(m, 2)
    var s_abs = 0.0
    for t in h.terms:
        s_abs += abs(t.coeff)
    var cnorm = 2.0 * s_abs
    var dm = dense_matrix(h, 2)
    var recon = 0.0
    for i in range(16):
        recon = max(recon, abs(dm[0][i] - m[i]))
        recon = max(recon, abs(dm[1][i]))
    print("Pauli terms:", len(h.terms), " normalisation c =", cnorm)
    _ = log.close("Pauli expansion reproduces M", recon, 0.0, 1e-12)
    _ = log.close("normalisation matches the notebook", cnorm, 2.34912912, 1e-9)

    # ---- Eigen-decomposition: classical reference, and the Pauli-route spectrum ----
    var eig = jacobi_eigh(m, 4)
    var spec = exact_spectrum(h, 2)
    var ref_ev: List[Float64] = [0.1, 0.2, 0.4, 0.9]
    var ev_err = 0.0
    for k in range(4):
        ev_err = max(ev_err, abs(eig.values[k] - ref_ev[k]))
        ev_err = max(ev_err, abs(spec[k] - ref_ev[k]))
    print("eigenvalues of M:", eig.values[0], eig.values[1], eig.values[2], eig.values[3])
    _ = log.close("eigenvalues of M are 0.1 0.2 0.4 0.9", ev_err, 0.0, 1e-7)

    # ---- Initial vector: notebook's np.random.seed(8); np.random.rand(4), copied from its output ----
    var iv: List[Float64] = [0.8734294, 0.96854066, 0.86919454, 0.53085569]
    var nrm = 0.0
    for x in iv:
        nrm += x * x
    nrm = sqrt(nrm)
    var psi = List[Float64]()
    for x in iv:
        psi.append(x / nrm)

    # Eigenphases theta_k = lambda_k / c and the amplitudes V^T psi in the eigenbasis.
    var theta = List[Float64]()
    var phases = List[Float64]()
    for k in range(4):
        theta.append(eig.values[k] / cnorm)
        phases.append(2.0 * pi * theta[k])
    var amps = List[Float64]()
    for k in range(4):
        var v = eig.vector(k)
        var s = 0.0
        for i in range(4):
            s += v[i] * psi[i]
        amps.append(s)

    # ---- Build the QPE circuits and run them ----
    var qpe_exact = build_qpe(N_QPE, amps, phases, False)
    var qpe_meas = build_qpe(N_QPE, amps, phases, True)
    print("QPE circuit:", qpe_exact.num_qubits, "qubits,", len(qpe_exact), "gates")

    var sv = evolve(qpe_exact)
    var p_sim = List[Float64](length=nk, fill=0.0)
    for s in range(4):
        for k in range(nk):
            var idx = k + (s << N_QPE)
            p_sim[k] += sv.re[idx] * sv.re[idx] + sv.im[idx] * sv.im[idx]

    # Analytic reference, independent of the circuit: P(k) = sum_i |<v_i|psi>|^2 kernel(theta_i, k).
    var w = List[Float64]()
    for i in range(4):
        var v = eig.vector(i)
        var d = 0.0
        for q in range(4):
            d += v[q] * psi[q]
        w.append(d * d)
    var p_ref = List[Float64](length=nk, fill=0.0)
    for k in range(nk):
        for i in range(4):
            p_ref[k] += w[i] * alpha_sq(theta[i], k, N_QPE)

    var dev = 0.0
    var tot = 0.0
    for k in range(nk):
        dev = max(dev, abs(p_sim[k] - p_ref[k]))
        tot += p_sim[k]
    _ = log.close("QPE distribution matches the analytic kernel", dev, 0.0, 1e-9)
    _ = log.close("QPE distribution sums to one", tot, 1.0, 1e-12)

    var b1 = 0
    for k in range(1, nk):
        if p_sim[k] > p_sim[b1]:
            b1 = k
    var b2 = -1
    for k in range(nk):
        if k != b1 and (b2 < 0 or p_sim[k] > p_sim[b2]):
            b2 = k

    var res_w = 2.0 * cnorm / Float64(nk)
    var est1 = Float64(b1) * cnorm / Float64(nk)
    var est2 = Float64(b2) * cnorm / Float64(nk)
    print("exact model, top bins: k =", b1, "estimate", est1, "probability", p_sim[b1])
    print("exact model, second:   k =", b2, "estimate", est2, "probability", p_sim[b2])
    print("analytic probability of the top bin:", p_ref[b1], " resolution:", res_w)

    _ = log.check("most probable bin is k = 49 (eigenvalue 0.9)", b1 == 49, "k=" + String(b1))
    _ = log.check("second bin is k = 22 (eigenvalue 0.4)", b2 == 22, "k=" + String(b2))
    _ = log.close("estimate 0.8993 matches the notebook", est1, 0.8992759912499999, 1e-9)
    _ = log.close("estimate 0.4038 matches the notebook", est2, 0.40375656749999994, 1e-9)
    _ = log.close("resolution matches the notebook", res_w, 0.036705142499999996, 1e-9)
    _ = log.within("eigenvalue 0.9 lies in its bin", eig.values[3], est1 - res_w, est1 + res_w)
    _ = log.within("eigenvalue 0.4 lies in its bin", eig.values[2], est2 - res_w, est2 + res_w)

    # ---- Sampling with 10000 shots, as in the notebook ----
    var shots = sample(qpe_meas, SHOTS, UInt64(8))
    var p_samp = List[Float64](length=nk, fill=0.0)
    for k in range(nk):
        p_samp[k] = Float64(shots.counts.get(bitstring(k, N_QPE), 0)) / Float64(SHOTS)
    var sb1 = 0
    for k in range(1, nk):
        if p_samp[k] > p_samp[sb1]:
            sb1 = k
    var sb2 = -1
    for k in range(nk):
        if k != sb1 and (sb2 < 0 or p_samp[k] > p_samp[sb2]):
            sb2 = k
    print("sampled top bins:", sb1, sb2, " P(49) sampled", p_samp[49], "exact", p_sim[49])
    _ = log.check("sampled top two bins are 49 and 22", sb1 == 49 and sb2 == 22, "k=" + String(sb1) + "," + String(sb2))
    _ = log.close("sampled probability of bin 49 matches the exact one", p_samp[49], p_sim[49], 0.02)
    log.finish()
