"""HHL linear solver: the notebook's exact-simulation model for A x = b on a 4 x 4 matrix, reproduced on QUMO.

Reproduces: Classiq library - HHL Algorithm
Source: https://github.com/Classiq/classiq-library/blob/main/algorithms/quantum_linear_solvers/hhl/hhl.ipynb (Classiq Library, MIT)
Framework: Classiq - https://github.com/Classiq/classiq-library

The notebook solves A x = b for a 4 x 4 symmetric A and the normalised b = (1, 2, 4, 3) / sqrt(30), with 4-bit phase estimation of the exact unitary U = exp(2 pi i A), controlled inversion of the estimated eigenvalues, and postselection on estimator = 0 and indicator = 1. QUMO builds the same circuit on 7 qubits. The memory register is held in the eigenbasis of A, where the controlled powers U^(2^j) are exact diagonal phase gates; the basis change V is a Givens network of two-level rotations, applied after state preparation (which loads V^T b) and once more after the inverse QPE. The checks compare the classical solution, the eigenvalues and the eigenvector residuals, the V network against A, and the notebook's printed quantum solution (normalised and phase-corrected) and fidelity.

Deviations from the notebook: the Suzuki-Trotter comparison (the second circuit, R0 = 4, REPS_SCALING_FACTOR = 1.8) is not ported, since its numbers depend on Classiq's product-formula gate order. QUMO has no matrix-gate primitive, so U^(2^j) = V D^(2^j) V^T is built from the eigenphases of A; this is the same unitary the notebook's scipy expm produces. qpe_flexible is written out as H, controlled phases and the inverse QFT. assign_amplitude_table is replaced by 15 multi-controlled RY gates with the same amplitudes C / lambda-tilde (C = 1/16). The notebook rounds its quantum solution to 5 decimals before printing, so the checks allow 5e-6 for that rounding.

Run: .venv/bin/mojo run -I . examples/algorithms/hhl.mojo
example-meta: {"framework": "classiq", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from std.math import asin, atan2, cos, pi, sin, sqrt
from qumo.ir.circuit import Circuit
from qumo.ir.mc import append_mcp, controlled_by
from qumo.algorithms.library import iqft
from qumo.algorithms.qml import append_amplitude_embedding_real
from qumo.algorithms.linalg import jacobi_eigh
from qumo.sim.runner import evolve
from qumo.testing.checks import CheckLog

comptime N_EST = 4  # estimator (phase) qubits: precision m = 4
# Qubit layout: estimator 0..3, memory 4..5 (memory value = q4 + 2 q5), indicator 6.


def _solve(a: List[Float64], b: List[Float64], n: Int) -> List[Float64]:
    """Gaussian elimination with partial pivoting: the classical reference, independent of QUMO."""
    var m = a.copy()
    var x = b.copy()
    for col in range(n):
        var piv = col
        for r in range(col + 1, n):
            if abs(m[r * n + col]) > abs(m[piv * n + col]):
                piv = r
        if piv != col:
            for k in range(n):
                var t = m[col * n + k]
                m[col * n + k] = m[piv * n + k]
                m[piv * n + k] = t
            var tx = x[col]
            x[col] = x[piv]
            x[piv] = tx
        for r in range(col + 1, n):
            var f = m[r * n + col] / m[col * n + col]
            for k in range(col, n):
                m[r * n + k] -= f * m[col * n + k]
            x[r] -= f * x[col]
    var sol = List[Float64](length=n, fill=0.0)
    for ri in range(n):
        var r = n - 1 - ri
        var s = x[r]
        for k in range(r + 1, n):
            s -= m[r * n + k] * sol[k]
        sol[r] = s / m[r * n + r]
    return sol^


def _cx_map(x: Int) -> Int:
    """The permutation of CX(0 -> 1) on two qubits: swaps 1 and 3, fixes 0 and 2."""
    return x ^ ((x & 1) << 1)


def _two_level_ry(mut g: Circuit, a: Int, b: Int, theta: Float64) raises:
    """On the 2-qubit basis: M|a> = cos(t/2)|a> + sin(t/2)|b>, M|b> = -sin(t/2)|a> + cos(t/2)|b>, identity elsewhere."""
    var diff = a ^ b
    if diff == 3:
        # a and b differ in both bits: M = P M' P, with P = CX(0 -> 1) and M' acting on P(a), P(b), which differ in one bit.
        var pa = _cx_map(a)
        var pb = _cx_map(b)
        g.cx(0, 1)
        _two_level_ry(g, pa, pb, theta)
        g.cx(0, 1)
        return
    var q = 0 if diff == 1 else 1
    var o = 1 - q
    var v = (a >> o) & 1
    # RY(theta) acts on |0_q>, |1_q>; when a is the |1_q> member the rotation is RY(-theta).
    var th = theta if ((a >> q) & 1) == 0 else -theta
    if v == 0:
        g.x(o)
        g.cry(th, o, q)
        g.x(o)
    else:
        g.cry(th, o, q)


def _givens_plan(v: List[Float64], mut ra: List[Int], mut rb: List[Int], mut rt: List[Float64]) -> Float64:
    """Two-level real rotations G_1..G_L with G_L ... G_1 V = diag(1, 1, 1, sigma), V row-major 4 x 4.

    Returns sigma = det V. Each G acts on rows (j, i) of the working matrix; the circuit applies their transposes."""
    var w = v.copy()
    for j in range(3):
        for i in range(j + 1, 4):
            var xr = w[j * 4 + j]
            var xi = w[i * 4 + j]
            if xi == 0.0 and xr >= 0.0:
                continue
            var h = sqrt(xr * xr + xi * xi)
            var c = xr / h
            var s = xi / h
            for k in range(4):
                var rj = w[j * 4 + k]
                var ri = w[i * 4 + k]
                w[j * 4 + k] = c * rj + s * ri
                w[i * 4 + k] = -s * rj + c * ri
            ra.append(j)
            rb.append(i)
            rt.append(2.0 * atan2(s, c))
    return w[15]


def _v_gate(ra: List[Int], rb: List[Int], rt: List[Float64], sigma: Float64) raises -> Circuit:
    """The basis change V = G_1^T ... G_L^T diag(1, 1, 1, sigma) on two memory qubits (diagonal first, then the transposes)."""
    var g = Circuit(2, name="eigenbasis_change")
    if sigma < 0.0:
        g.cz(0, 1)
    var k = len(ra) - 1
    while k >= 0:
        _two_level_ry(g, ra[k], rb[k], rt[k])
        k -= 1
    return g^


def _qpe_eigen(phases: List[Float64]) raises -> Circuit:
    """QPE of the diagonal unitary D = diag(exp(i phases[k])) on the memory eigen-index k (qubits N_EST, N_EST+1).

    Controlled D^(2^j) from estimator qubit j: the global part is a phase on the control, and the four eigenphases are
    phi(k) = g + a s0 + b s1 + c s0 s1 (s0, s1 the bits of k), so each power costs one P, two CP and one doubly controlled phase."""
    var c = Circuit(N_EST + 2, name="qpe_eigen")
    for j in range(N_EST):
        c.h(j)
    var g = phases[0]
    var a = phases[1] - phases[0]
    var b = phases[2] - phases[0]
    var cc = phases[3] - phases[1] - phases[2] + phases[0]
    for j in range(N_EST):
        var f = Float64(1 << j)
        c.p(f * g, j)
        c.cp(f * a, j, N_EST)
        c.cp(f * b, j, N_EST + 1)
        var ctrl = List[Int]()
        ctrl.append(j)
        ctrl.append(N_EST)
        append_mcp(c, f * cc, ctrl, N_EST + 1)
    c.compose(iqft(N_EST))
    return c^


def main() raises:
    var log = CheckLog("classiq/hhl")

    # ---- The notebook's matrix and right-hand side ----
    var a: List[Float64] = [
        0.28, -0.01, 0.02, -0.1,
        -0.01, 0.5, -0.22, -0.07,
        0.02, -0.22, 0.43, -0.05,
        -0.1, -0.07, -0.05, 0.42,
    ]
    var b_raw: List[Float64] = [1.0, 2.0, 4.0, 3.0]
    var nb = 0.0
    for x in b_raw:
        nb += x * x
    nb = sqrt(nb)
    var b = List[Float64]()
    for x in b_raw:
        b.append(x / nb)

    # ---- Classical reference: solve A x = b without QUMO ----
    var x_ref = _solve(a, b, 4)
    var x_printed: List[Float64] = [1.3814374, 2.50585064, 3.19890483, 2.43147877]
    for i in range(4):
        _ = log.close("classical solution component " + String(i) + " matches the notebook", x_ref[i], x_printed[i], 1e-8)

    # ---- Eigen-decomposition of A (QUMO's Jacobi solver gives the circuit parameters) ----
    var eig = jacobi_eigh(a, 4)
    var lam_printed: List[Float64] = [0.1923052128566636, 0.2518970844513994, 0.49592342121467137, 0.6898742814772654]
    for k in range(4):
        _ = log.close("eigenvalue " + String(k) + " matches the notebook", eig.values[k], lam_printed[k], 1e-10)
    var res_eig = 0.0
    for k in range(4):
        var v = eig.vector(k)
        for i in range(4):
            var av = 0.0
            for j in range(4):
                av += a[i * 4 + j] * v[j]
            res_eig = max(res_eig, abs(av - eig.values[k] * v[i]))
    _ = log.close("eigenvectors satisfy A v = lambda v", res_eig, 0.0, 1e-12)

    # Eigenbasis amplitudes of b (state preparation loads V^T b) and eigenphases of U = exp(2 pi i A).
    var amps = List[Float64]()
    var phases = List[Float64]()
    for k in range(4):
        var v = eig.vector(k)
        var s = 0.0
        for i in range(4):
            s += v[i] * b[i]
        amps.append(s)
        phases.append(2.0 * pi * eig.values[k])

    # The basis change V (eigenvector k in column k) as a Givens network.
    var ra = List[Int]()
    var rb = List[Int]()
    var rt = List[Float64]()
    var sigma = _givens_plan(eig.vectors, ra, rb, rt)
    print("Givens rotations:", len(ra), " det V =", sigma)
    var vg = _v_gate(ra, rb, rt, sigma)

    # The network maps |k> to eigenvector k of A.
    var res_v = 0.0
    var nrm_v = 0.0
    for k in range(4):
        var prep = Circuit(2, name="basis_input")
        if (k & 1) == 1:
            prep.x(0)
        if ((k >> 1) & 1) == 1:
            prep.x(1)
        prep.compose(vg)
        var out = evolve(prep)
        var col = List[Float64]()
        for i in range(4):
            col.append(out.re[i])
        for i in range(4):
            var av = 0.0
            for j in range(4):
                av += a[i * 4 + j] * col[j]
            res_v = max(res_v, abs(av - eig.values[k] * col[i]))
        var nk = 0.0
        for i in range(4):
            nk += col[i] * col[i]
        nrm_v = max(nrm_v, abs(nk - 1.0))
    _ = log.close("V network maps |k> to an eigenvector of A", res_v, 0.0, 1e-9)
    _ = log.close("V network output is normalised", nrm_v, 0.0, 1e-12)

    # ---- HHL circuit: load, QPE, inversion, inverse QPE, basis change back ----
    var circ = Circuit(N_EST + 3, name="hhl")
    var wires = List[Int]()
    wires.append(5)
    wires.append(4)
    append_amplitude_embedding_real(circ, amps, wires)
    var qpe = _qpe_eigen(phases)
    circ.compose(qpe)
    for p in range(1, 1 << N_EST):
        var lam = Float64(p) / Float64(1 << N_EST)  # estimator value p / 2^m
        var f = (1.0 / Float64(1 << N_EST)) / lam  # C / lambda-tilde: the notebook's lookup table
        var sub = Circuit(N_EST + 3, name="inversion")
        sub.ry(2.0 * asin(f), N_EST + 2)
        var ctrl = List[Int]()
        for j in range(N_EST):
            ctrl.append(j)
        # controlled_by fires when every estimator qubit is 1, so flip the zero bits of p to select estimator = p.
        for j in range(N_EST):
            if ((p >> j) & 1) == 0:
                circ.x(j)
        circ.compose(controlled_by(sub, ctrl))
        for j in range(N_EST):
            if ((p >> j) & 1) == 0:
                circ.x(j)
    circ.compose(qpe.inverse())
    circ.compose(vg, N_EST)

    var sv = evolve(circ)
    # Postselect estimator = 0 and indicator = 1 (basis index = estimator + 16 * memory + 64 * indicator);
    # qsol = amplitude / (1 / 2^precision) = 16 * amplitude, as in the notebook.
    var qre = List[Float64](length=4, fill=0.0)
    var qim = List[Float64](length=4, fill=0.0)
    for m in range(4):
        var idx = 16 * m + 64
        qre[m] = 16.0 * sv.re[idx]
        qim[m] = 16.0 * sv.im[idx]

    # Global phase exactly as the notebook: angle(vdot(A qsol, b)); the solution is the real part.
    var sre = 0.0
    var sim = 0.0
    for i in range(4):
        var rr = 0.0
        var ri = 0.0
        for j in range(4):
            rr += a[i * 4 + j] * qre[j]
            ri += a[i * 4 + j] * qim[j]
        sre += rr * b[i]
        sim -= ri * b[i]
    var theta_g = atan2(sim, sre)
    var corr = List[Float64]()
    for j in range(4):
        corr.append(cos(theta_g) * qre[j] - sin(theta_g) * qim[j])

    var qnorm = 0.0
    for j in range(4):
        qnorm += qre[j] * qre[j] + qim[j] * qim[j]
    qnorm = sqrt(qnorm)
    var normd = List[Float64]()
    for j in range(4):
        normd.append(sqrt(qre[j] * qre[j] + qim[j] * qim[j]) / qnorm)

    var cnorm = 0.0
    var corr_norm = 0.0
    for j in range(4):
        cnorm += x_ref[j] * x_ref[j]
        corr_norm += corr[j] * corr[j]
    cnorm = sqrt(cnorm)
    corr_norm = sqrt(corr_norm)
    var dot = 0.0
    for j in range(4):
        dot += (x_ref[j] / cnorm) * (corr[j] / corr_norm)
    var fidelity = dot * dot

    print("normalised quantum solution:", normd[0], normd[1], normd[2], normd[3])
    print("corrected quantum solution:", corr[0], corr[1], corr[2], corr[3])
    print("fidelity:", fidelity)

    # The notebook prints these after rounding qsol to 5 decimals: each printed value is within 5e-6 of the exact one.
    var q_norm_printed: List[Float64] = [0.2840631, 0.50843613, 0.64683772, 0.4923432]
    var corr_printed: List[Float64] = [1.43559, 2.56952, 3.26897, 2.48819]
    for j in range(4):
        _ = log.close("normalised quantum solution component " + String(j) + " matches the notebook", normd[j], q_norm_printed[j], 5e-6)
    for j in range(4):
        _ = log.close("corrected quantum solution component " + String(j) + " matches the notebook", corr[j], corr_printed[j], 5e-6)
    # The notebook prints the fidelity as 100.0 %, i.e. it rounds to [0.99995, 1.00005].
    _ = log.within("fidelity rounds to the notebook's 100.0 %", fidelity, 0.99995, 1.00005)
    log.finish()
