"""Qubitization QPE for the H2 molecular energy: the notebook's walk operator and both QPE designs, from QUMO gates.

Reproduces: Classiq library - Qubitization Based Quantum Phase Estimation (QPE) for Solving Molecular Energies
Source: https://github.com/Classiq/classiq-library/blob/main/algorithms/quantum_phase_estimation/qpe_with_qubitization/qpe_for_molecule_with_qubitization.ipynb (Classiq Library, MIT)

The notebook takes H2 (STO-3G, 0.735 Angstrom), block-encodes its 15-term Pauli Hamiltonian with scaling s = sum |c| = 1.98507 as U = PREP^dag SEL PREP, forms the walk W = (2 Pi_0 - I) U, and runs 5-bit QPE on W from the Hartree-Fock state. This file builds the same objects from QUMO gates on 13 qubits (data q0-q3, LCU block q4-q7, phase q8-q12) and simulates them exactly. Both QPE designs of the notebook are built: the naive one with controlled W^(2^j), and the optimized one, which applies W^(2^i) uncontrolled inside the phase flip I - 2 Pi(phase bit i+1 = 0, block = 0). The checks compare the block encoding with H/s on all 16 data inputs, U^2 = I (amplitudes, so a sign or phase would be caught), the walk powers with T_k(H/s) for k = 1, 2, 3, 15, both output distributions with closed-form QPE kernels built from the Hamiltonian's own eigendecomposition, and the notebook's printed FCI energy, scaling, error bound, most probable energy and block-zero probabilities.

Deviations from the notebook: the 15 Pauli coefficients are copied from its printed output (openfermion and PySCF are not run), and the FCI energy is recomputed by Jacobi diagonalisation of a 16 x 16 matrix that QUMO rebuilds from the printed Pauli strings (the notebook prints no matrix). The Hartree-Fock state is the lowest-energy two-electron basis state of that matrix (qubits 0 and 2 occupied, consistent with the equal positive Z0 and Z2 coefficients), inferred and not the notebook's get_hf_state call. The circuits and the closed-form weights are both derived from that one inferred index hf, and the energy checks at the end (printed most probable energy, FCI gap) are what tie it to the notebook. There is no Classiq synthesis, prepare_select, qubitize or cloud simulator: PREP is QUMO's Mottonen real amplitude embedding, SEL is written as controlled Pauli gates with controlled_by, and the minus sign of the walk is an explicit -I. The notebook's transpiled depths (26347 and 6670) come from Classiq's transpiler and are not reproduced. The notebook's block-zero probabilities are sampled; the exact QUMO probabilities are compared with the closed-form kernel at 1e-12, and the printed values with three binomial standard deviations. The notebook never states its shot count (sample() takes no argument): the printed values 1019/2048 and 1018/2048 are multiples of 2^-11, so N is a multiple of 2048 and the band uses the smallest, N = 2048 (an assumption, not a figure the notebook states); a larger N would only narrow the true band. The notebook's phase_var is a signed QNum fraction, while here the readout y is unsigned in [0, 32) with energy s cos(2 pi y / 32); the two agree because cos is even and 2 pi periodic. The block-zero probability checks are weak: they do not identify the Hartree-Fock state (the other five two-electron basis states also pass them); the energy checks do, and only index 5 reproduces the printed most probable energy.

Run: .venv/bin/mojo run -I . examples/frameworks/classiq/qpe_for_molecule_with_qubitization.mojo
example-meta: {"framework": "classiq", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from std.math import acos, cos, pi, sin, sqrt
from qumo.algorithms.library import iqft
from qumo.algorithms.qml import append_amplitude_embedding_real
from qumo.ir.circuit import Circuit
from qumo.ir.mc import controlled_by
from qumo.sim.runner import evolve
from qumo.testing.checks import CheckLog


comptime NQ = 13  # data q0-q3, LCU block q4-q7, phase q8-q12
comptime NDATA = 4
comptime NBLK = 4
comptime BLK0 = 4
comptime PH0 = 8
comptime NPH = 5
comptime NTERM = 15
comptime NOTEBOOK_S = 1.9850721353060017
comptime NOTEBOOK_FCI = -1.1373060357534004
comptime NOTEBOOK_QPE_ERR = 0.07603996508423008
comptime NOTEBOOK_E_MAX = -1.1028469887726742
comptime NOTEBOOK_P0_NAIVE = 0.49755859375
comptime NOTEBOOK_P0_OPT = 0.4970703125
comptime SHOT_TOL = 0.0331786  # 3 sqrt(0.25 / 2048): binomial band at p = 1/2 for N = 2048, the smallest shot count the printed values allow


def pauli_letters() -> List[Int]:
    """Letters of the notebook's 15 terms on data qubits q0..q3, term t at 4 t + q: 0 = I, 1 = X, 2 = Y, 3 = Z."""
    var out: List[Int] = [
        0, 0, 0, 0,  # identity
        1, 1, 1, 1,  # X0 X1 X2 X3
        1, 1, 2, 2,  # X0 X1 Y2 Y3
        2, 2, 1, 1,  # Y0 Y1 X2 X3
        2, 2, 2, 2,  # Y0 Y1 Y2 Y3
        3, 0, 0, 0,  # Z0
        3, 3, 0, 0,  # Z0 Z1
        3, 0, 3, 0,  # Z0 Z2
        3, 0, 0, 3,  # Z0 Z3
        0, 3, 0, 0,  # Z1
        0, 3, 3, 0,  # Z1 Z2
        0, 3, 0, 3,  # Z1 Z3
        0, 0, 3, 0,  # Z2
        0, 0, 3, 3,  # Z2 Z3
        0, 0, 0, 3,  # Z3
    ]
    return out^


def pauli_coefs() -> List[Float64]:
    """The notebook's printed coefficients, in the same term order."""
    var out: List[Float64] = [
        -0.09057898608834856, 0.04523279994605784, 0.04523279994605784, 0.04523279994605784,
        0.04523279994605784, 0.17218393261915546, 0.12091263261776633, 0.16892753870087918,
        0.16614543256382416, -0.22575349222402352, 0.16614543256382416, 0.174643430683004,
        0.17218393261915552, 0.12091263261776633, -0.22575349222402352,
    ]
    return out^


def hamiltonian_matrix() raises -> List[Float64]:
    """Row-major 16 x 16 matrix m[y * 16 + x] = <y| H |x>, built from the Pauli strings.

    For a Pauli string, P|x> = (i^{nY} (-1)^{number of Y and Z qubits set in x}) |x xor flip>, where flip marks X and Y.
    """
    var letters = pauli_letters()
    var coefs = pauli_coefs()
    var m = List[Float64](length=16 * 16, fill=0.0)
    for t in range(NTERM):
        var flip = 0
        var n_y = 0
        for q in range(NDATA):
            var l = letters[4 * t + q]
            if l == 1 or l == 2:
                flip = flip | (1 << q)
            if l == 2:
                n_y += 1
        if n_y % 2 == 1:
            raise Error("QM-E830: odd number of Y gives an imaginary term; the printed Hamiltonian is real.")
        var base = coefs[t]
        if (n_y // 2) % 2 == 1:
            base = -base
        for x in range(16):
            var sgn = base
            for q in range(NDATA):
                var l = letters[4 * t + q]
                if (l == 2 or l == 3) and ((x >> q) & 1) == 1:
                    sgn = -sgn
            m[(x ^ flip) * 16 + x] += sgn
    return m^


def jacobi_eigh(a_in: List[Float64], n: Int) -> Tuple[List[Float64], List[Float64]]:
    """Cyclic Jacobi rotations for a real symmetric n x n matrix (row-major).

    Returns the eigenvalues and the row-major eigenvector matrix v: eigenvector k is column k, v[i * n + k] its component i."""
    var a = a_in.copy()
    var v = List[Float64](length=n * n, fill=0.0)
    for i in range(n):
        v[i * n + i] = 1.0
    for _ in range(60):
        var off = 0.0
        for p in range(n):
            for q in range(n):
                if p != q:
                    off += a[p * n + q] * a[p * n + q]
        if off < 1e-28:
            break
        for p in range(n):
            for q in range(p + 1, n):
                var apq = a[p * n + q]
                if abs(apq) < 1e-18:
                    continue
                var theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq)
                var t = (1.0 if theta >= 0.0 else -1.0) / (abs(theta) + sqrt(theta * theta + 1.0))
                var c = 1.0 / sqrt(t * t + 1.0)
                var s = t * c
                for k in range(n):
                    var akp = a[k * n + p]
                    var akq = a[k * n + q]
                    a[k * n + p] = c * akp - s * akq
                    a[k * n + q] = s * akp + c * akq
                for k in range(n):
                    var apk = a[p * n + k]
                    var aqk = a[q * n + k]
                    a[p * n + k] = c * apk - s * aqk
                    a[q * n + k] = s * apk + c * aqk
                for k in range(n):
                    var vkp = v[k * n + p]
                    var vkq = v[k * n + q]
                    v[k * n + p] = c * vkp - s * vkq
                    v[k * n + q] = s * vkp + c * vkq
    var ev = List[Float64]()
    for i in range(n):
        ev.append(a[i * n + i])
    return (ev^, v^)


def mat_vec(m: List[Float64], s: Float64, v: List[Float64]) -> List[Float64]:
    """(M / s) v for the row-major 16 x 16 matrix M."""
    var out = List[Float64](length=16, fill=0.0)
    for y in range(16):
        var acc = 0.0
        for x in range(16):
            acc += m[y * 16 + x] * v[x]
        out[y] = acc / s
    return out^


def cheb_columns(m: List[Float64], s: Float64, x: Int, kmax: Int) -> List[List[Float64]]:
    """T_k(M / s) e_x for k = 0..kmax, by the three-term recurrence."""
    var e = List[Float64](length=16, fill=0.0)
    e[x] = 1.0
    var out = List[List[Float64]]()
    out.append(e.copy())
    out.append(mat_vec(m, s, e))
    for k in range(2, kmax + 1):
        var hv = mat_vec(m, s, out[k - 1])
        var nxt = List[Float64](length=16, fill=0.0)
        for y in range(16):
            nxt[y] = 2.0 * hv[y] - out[k - 2][y]
        out.append(nxt^)
    return out^


def qpe_amp(mu: Float64, y: Int) -> Tuple[Float64, Float64]:
    """Amplitude a_y(mu) = (1/32) sum_x exp(i x (mu - 2 pi y / 32)): readout y of the 5-qubit QPE of eigenphase mu."""
    var d = mu - 2.0 * pi * Float64(y) / 32.0
    var re = 0.0
    var im = 0.0
    for x in range(32):
        re += cos(Float64(x) * d)
        im += sin(Float64(x) * d)
    return (re / 32.0, im / 32.0)


def block_wires() -> List[Int]:
    """Amplitude wires of the block register: wires[0] = q7 carries block bit 3 (the most significant)."""
    var w: List[Int] = [7, 6, 5, 4]
    return w^


def phase_wires() -> List[Int]:
    var w: List[Int] = [8, 9, 10, 11, 12]
    return w^


def lcu_amplitudes() -> List[Float64]:
    """PREP|0> = sum_j sqrt(|c_j| / s) |j>, with block value 15 unused."""
    var coefs = pauli_coefs()
    var s = 0.0
    for t in range(NTERM):
        s += abs(coefs[t])
    var amp = List[Float64]()
    for t in range(NTERM):
        amp.append(sqrt(abs(coefs[t]) / s))
    amp.append(0.0)
    return amp^


def append_uh(mut c: Circuit) raises:
    """U_H = PREP^dag SEL PREP on the data and block registers.

    SEL applies sgn_j P_j on the data when the block equals j. The zero bits of j are X-conjugated so the
    controls fire on that value only. A negative sign is -I, written as Z X Z X on data qubit 0.
    """
    var prep = Circuit(NQ, name="prep")
    append_amplitude_embedding_real(prep, lcu_amplitudes(), block_wires())
    c.compose(prep)
    var letters = pauli_letters()
    var coefs = pauli_coefs()
    var ctrl: List[Int] = [BLK0, BLK0 + 1, BLK0 + 2, BLK0 + 3]
    for j in range(NTERM):
        for k in range(NBLK):
            if ((j >> k) & 1) == 0:
                c.x(BLK0 + k)
        var sub = Circuit(NQ, name="select")
        for q in range(NDATA):
            var l = letters[4 * j + q]
            if l == 1:
                sub.x(q)
            elif l == 2:
                sub.y(q)
            elif l == 3:
                sub.z(q)
        if coefs[j] < 0.0:
            sub.z(0)
            sub.x(0)
            sub.z(0)
            sub.x(0)
        c.compose(controlled_by(sub, ctrl))
        for k in range(NBLK):
            if ((j >> k) & 1) == 0:
                c.x(BLK0 + k)
    c.compose(prep.inverse())


def append_walk(mut c: Circuit) raises:
    """W = (2 Pi_0 - I) U_H, the notebook's walk: U_H, then X^4 MCZ X^4 with the minus sign of my_reflect_about_zero."""
    append_uh(c)
    for k in range(NBLK):
        c.x(BLK0 + k)
    var sub = Circuit(NQ, name="mcz")
    sub.z(BLK0 + 3)
    var ctrl: List[Int] = [BLK0, BLK0 + 1, BLK0 + 2]
    c.compose(controlled_by(sub, ctrl))
    for k in range(NBLK):
        c.x(BLK0 + k)
    # (I - 2 Pi_0) was just applied; the walk needs (2 Pi_0 - I) = -(I - 2 Pi_0): a global -I, as Z X Z X on q0.
    c.z(0)
    c.x(0)
    c.z(0)
    c.x(0)


def append_v(mut c: Circuit, phase_q: Int) raises:
    """I - 2 Pi(phase qubit = 0 and block = 0): the notebook's control(phase==0, control(block==0, phase(pi)))."""
    var qs: List[Int] = [phase_q, BLK0, BLK0 + 1, BLK0 + 2, BLK0 + 3]
    for q in qs:
        c.x(q)
    var sub = Circuit(NQ, name="v")
    sub.z(BLK0 + 3)
    var ctrl: List[Int] = [phase_q, BLK0, BLK0 + 1, BLK0 + 2]
    c.compose(controlled_by(sub, ctrl))
    for q in qs:
        c.x(q)


def basis_prep(x: Int) raises -> Circuit:
    """|x> on the data register (q0 is bit 0), block and phase in |0>."""
    var c = Circuit(NQ, name="basis")
    for q in range(NDATA):
        if ((x >> q) & 1) == 1:
            c.x(q)
    return c^


def qpe_naive(walk: Circuit, hf: Int) raises -> Circuit:
    """Reference basis state hf (data qubit q set when bit q of hf is 1), H on the phase register, controlled W^(2^j) from phase qubit j, inverse QFT."""
    var c = Circuit(NQ, name="qpe_naive")
    for q in range(NDATA):
        if ((hf >> q) & 1) == 1:
            c.x(q)
    for j in range(NPH):
        c.h(PH0 + j)
    for j in range(NPH):
        var ctrl: List[Int] = [PH0 + j]
        c.compose(controlled_by(walk.power(1 << j), ctrl))
    c.compose_mapped(iqft(NPH), phase_wires())
    return c^


def qpe_optimized(walk: Circuit, hf: Int) raises -> Circuit:
    """The notebook's optimized design: controlled W on phase qubit 0, then for i = 0..3 the phase flip V_i,
    W^(2^i) uncontrolled, and V_i again. V_i W^k V_i = W^(+k) when phase qubit i+1 is 1 and W^(-k) when it is 0."""
    var c = Circuit(NQ, name="qpe_optimized")
    for q in range(NDATA):
        if ((hf >> q) & 1) == 1:
            c.x(q)
    for j in range(NPH):
        c.h(PH0 + j)
    var ctrl0: List[Int] = [PH0]
    c.compose(controlled_by(walk, ctrl0))
    for i in range(NPH - 1):
        append_v(c, PH0 + i + 1)
        c.compose(walk.power(1 << i))
        append_v(c, PH0 + i + 1)
    c.compose_mapped(iqft(NPH), phase_wires())
    return c^


def phase_and_block_zero(probs: List[Float64]) -> Tuple[List[Float64], List[Float64]]:
    """Marginal over the phase register, and the phase distribution restricted to block = 0."""
    var p_ph = List[Float64](length=32, fill=0.0)
    var p0 = List[Float64](length=32, fill=0.0)
    for idx in range(len(probs)):
        var y = idx >> 8
        var blk = (idx >> 4) & 15
        p_ph[y] += probs[idx]
        if blk == 0:
            p0[y] += probs[idx]
    return (p_ph^, p0^)


def energy_of(s: Float64, y: Int) -> Float64:
    """Energy of readout y: s cos(2 pi y / 32), as the notebook's post-processing."""
    return s * cos(2.0 * pi * Float64(y) / 32.0)


def argmax(xs: List[Float64]) -> Int:
    var best = 0
    for i in range(len(xs)):
        if xs[i] > xs[best]:
            best = i
    return best


def main() raises:
    var log = CheckLog("classiq/qpe_for_molecule_with_qubitization")
    var coefs = pauli_coefs()
    var s = 0.0
    for t in range(NTERM):
        s += abs(coefs[t])
    _ = log.close("scaling s = sum of |coefficients| is the notebook's 1.9850721353060017", s, NOTEBOOK_S, 1e-12)
    var bits = 0
    while (1 << bits) <= NTERM - 1:
        bits += 1
    _ = log.check("block register has 4 qubits, as the notebook's (num_terms - 1).bit_length()", bits == NBLK, "bits=" + String(bits))
    var qpe_err = 2.0 * pi * s * sin(2.0 * pi / 32.0) / 32.0
    _ = log.close("QPE error 2 pi s sin(2 pi / 32) / 32 is the notebook's 0.07603996508423008", qpe_err, NOTEBOOK_QPE_ERR, 1e-12)

    # ---- Classical reference: the printed Hamiltonian, diagonalised by Jacobi rotations ----
    var m = hamiltonian_matrix()
    var asym = 0.0
    for y in range(16):
        for x in range(16):
            asym = max(asym, abs(m[y * 16 + x] - m[x * 16 + y]))
    _ = log.close("the 16 x 16 matrix rebuilt from the Pauli strings is symmetric", asym, 0.0, 1e-15)
    var eig = jacobi_eigh(m, 16)
    var lam = eig[0].copy()
    var vec = eig[1].copy()
    var resid = 0.0
    var orth = 0.0
    for k in range(16):
        for i in range(16):
            var mv = 0.0
            for j in range(16):
                mv += m[i * 16 + j] * vec[j * 16 + k]
            resid = max(resid, abs(mv - lam[k] * vec[i * 16 + k]))
        for l in range(16):
            var dot = 0.0
            for i in range(16):
                dot += vec[i * 16 + k] * vec[i * 16 + l]
            orth = max(orth, abs(dot - (1.0 if k == l else 0.0)))
    _ = log.close("Jacobi eigenpairs of the matrix have residual below 1e-12", resid, 0.0, 1e-12)
    _ = log.close("Jacobi eigenvectors are orthonormal", orth, 0.0, 1e-12)
    var fci = lam[0]
    for k in range(16):
        fci = min(fci, lam[k])
    _ = log.close("classical ground energy from the printed Hamiltonian is the notebook's FCI -1.1373060357534004", fci, NOTEBOOK_FCI, 1e-12)

    var hf = -1
    var hf_e = 1.0e9
    for x in range(16):
        var pc = 0
        for q in range(NDATA):
            pc += (x >> q) & 1
        if pc == 2 and m[x * 16 + x] < hf_e:
            hf_e = m[x * 16 + x]
            hf = x
    _ = log.check("Hartree-Fock state is the lowest two-electron basis state, index 5 with qubits 0 and 2", hf == 5, "index=" + String(hf))

    # Overlaps of |HF> with the eigenvectors of H, and the walk phases theta_k = arccos(lambda_k / s).
    var w = List[Float64]()
    var theta = List[Float64]()
    for k in range(16):
        var ck = vec[hf * 16 + k]
        w.append(ck * ck)
        var a = max(-1.0, min(1.0, lam[k] / s))
        theta.append(acos(a))
    var wsum = 0.0
    for k in range(16):
        wsum += w[k]
    _ = log.close("eigenvector weights of |HF> sum to one", wsum, 1.0, 1e-12)

    # ---- Block encoding and walk, against the classical matrix ----
    var uh = Circuit(NQ, name="uh")
    append_uh(uh)
    var err_be = 0.0
    for x in range(16):
        var prep = basis_prep(x)
        prep.compose(uh)
        var sv = evolve(prep)
        for y in range(16):
            err_be = max(err_be, abs(sv.re[y] - m[y * 16 + x] / s))
            err_be = max(err_be, abs(sv.im[y]))
    _ = log.close("U_H on block 0 equals H / s on all 16 data basis inputs (max error)", err_be, 0.0, 1e-12)

    var uu = Circuit(NQ, name="uu")
    uu.compose(uh)
    uu.compose(uh)
    var err_u2 = 0.0
    for x in range(16):
        var prep = basis_prep(x)
        prep.compose(uu)
        var sv2 = evolve(prep)
        for y in range(len(sv2.re)):
            var want = 1.0 if y == x else 0.0
            err_u2 = max(err_u2, abs(sv2.re[y] - want))
            err_u2 = max(err_u2, abs(sv2.im[y]))
    _ = log.close("U_H squared is the identity on all 16 data inputs (max amplitude error, phase included)", err_u2, 0.0, 1e-12)

    var walk = Circuit(NQ, name="walk")
    append_walk(walk)
    var cheb_all = List[List[List[Float64]]]()
    for x in range(16):
        cheb_all.append(cheb_columns(m, s, x, 15))
    for k in [1, 2, 3, 15]:
        var wk = walk.power(k)
        var err_k = 0.0
        for x in range(16):
            var prep = basis_prep(x)
            prep.compose(wk)
            var sv = evolve(prep)
            for y in range(16):
                err_k = max(err_k, abs(sv.re[y] - cheb_all[x][k][y]))
                err_k = max(err_k, abs(sv.im[y]))
        _ = log.close("walk power W^" + String(k) + " on block 0 equals T_" + String(k) + "(H / s) (max error)", err_k, 0.0, 1e-12)

    # ---- Closed-form QPE distributions from the eigendecomposition and |HF> ----
    var ref_ph = List[Float64](length=32, fill=0.0)
    var ref_n = List[Float64](length=32, fill=0.0)
    var ref_o = List[Float64](length=32, fill=0.0)
    for k in range(16):
        var th = theta[k]
        var cs = cos(15.0 * th)
        var sn = sin(15.0 * th)
        for y in range(32):
            var ap = qpe_amp(th, y)
            var am = qpe_amp(-th, y)
            var pa = ap[0] * ap[0] + ap[1] * ap[1]
            var pm = am[0] * am[0] + am[1] * am[1]
            ref_ph[y] += 0.5 * w[k] * (pa + pm)
            var sre = ap[0] + am[0]
            var sim = ap[1] + am[1]
            ref_n[y] += 0.25 * w[k] * (sre * sre + sim * sim)
            # naive: the two eigenbranches interfere in block 0. Optimized: W^(-15) on the input gives e^(-15 i theta) on e_+ and e^(+15 i theta) on e_-.
            var gre = cs * (ap[0] + am[0]) + sn * (ap[1] - am[1])
            var gim = cs * (ap[1] + am[1]) + sn * (am[0] - ap[0])
            ref_o[y] += 0.25 * w[k] * (gre * gre + gim * gim)
    var ref_ph_sum = 0.0
    for y in range(32):
        ref_ph_sum += ref_ph[y]
    _ = log.close("closed-form QPE phase distribution sums to one", ref_ph_sum, 1.0, 1e-12)

    # ---- QUMO simulation of the two QPE designs ----
    var pr_n = evolve(qpe_naive(walk, hf)).probabilities()
    var split_n = phase_and_block_zero(pr_n)
    var ph_n = split_n[0].copy()
    var p0_n = split_n[1].copy()
    var pr_o = evolve(qpe_optimized(walk, hf)).probabilities()
    var split_o = phase_and_block_zero(pr_o)
    var ph_o = split_o[0].copy()
    var p0_o = split_o[1].copy()

    var norm_n = 0.0
    var norm_o = 0.0
    for i in range(len(pr_n)):
        norm_n += pr_n[i]
        norm_o += pr_o[i]
    _ = log.close("naive QPE state is normalised", norm_n, 1.0, 1e-12)
    _ = log.close("optimized QPE state is normalised", norm_o, 1.0, 1e-12)

    var dev_ph_n = 0.0
    var dev_ph_o = 0.0
    var dev_p0_n = 0.0
    var dev_p0_o = 0.0
    for y in range(32):
        dev_ph_n = max(dev_ph_n, abs(ph_n[y] - ref_ph[y]))
        dev_ph_o = max(dev_ph_o, abs(ph_o[y] - ref_ph[y]))
        dev_p0_n = max(dev_p0_n, abs(p0_n[y] - ref_n[y]))
        dev_p0_o = max(dev_p0_o, abs(p0_o[y] - ref_o[y]))
    _ = log.close("naive QPE phase distribution matches the closed form (max error)", dev_ph_n, 0.0, 1e-12)
    _ = log.close("optimized QPE phase distribution matches the closed form (max error)", dev_ph_o, 0.0, 1e-12)
    _ = log.close("naive QPE block-zero distribution matches the closed form (max error)", dev_p0_n, 0.0, 1e-12)
    _ = log.close("optimized QPE block-zero distribution matches the closed form (max error)", dev_p0_o, 0.0, 1e-12)

    var total_n = 0.0
    var total_o = 0.0
    for y in range(32):
        total_n += p0_n[y]
        total_o += p0_o[y]
    print("block-zero probability: naive", total_n, " optimized", total_o)
    _ = log.close("naive block-zero probability matches the notebook's printed 0.49755859375 within 3 sd at N = 2048", total_n, NOTEBOOK_P0_NAIVE, SHOT_TOL)
    _ = log.close("optimized block-zero probability matches the notebook's printed 0.4970703125 within 3 sd at N = 2048", total_o, NOTEBOOK_P0_OPT, SHOT_TOL)

    var y_n = argmax(p0_n)
    var y_o = argmax(p0_o)
    var e_n = energy_of(s, y_n)
    var e_o = energy_of(s, y_o)
    print("most probable block-zero bin: naive", y_n, "energy", e_n, " optimized", y_o, "energy", e_o)
    _ = log.check("most probable bin of both designs is y = 11 or 21 (phase +-11/32)", (y_n == 11 or y_n == 21) and (y_o == 11 or y_o == 21), "naive=" + String(y_n) + " optimized=" + String(y_o))
    _ = log.close("naive energy with maximal probability is the notebook's -1.1028469887726742", e_n, NOTEBOOK_E_MAX, 1e-12)
    _ = log.close("optimized energy with maximal probability is the notebook's -1.1028469887726742", e_o, NOTEBOOK_E_MAX, 1e-12)
    _ = log.check("maximal-probability energy lies within the QPE error of the FCI energy", abs(e_n - fci) < qpe_err and abs(e_o - fci) < qpe_err, "gap=" + String(abs(e_n - fci)))

    log.finish()
