"""Qubitization Hamiltonian simulation: the notebook's Chebyshev LCU of walk-operator powers, built from QUMO gates.

Reproduces: Classiq library - Hamiltonian Simulation with Qubitization (Chebyshev LCU)
Source: https://github.com/Classiq/classiq-library/blob/main/algorithms/hamiltonian_simulation/hamiltonian_simulation_with_block_encoding/hamiltonian_simulation_qubitization.ipynb (Classiq Library, MIT)
Framework: Classiq - https://github.com/Classiq/classiq-library

The notebook block-encodes H = 0.4 I + 0.1 Z1 + 0.05 X0 X1 + 0.2 Z0 Z1 with scaling 0.75, forms the Szegedy walk W = Pi U_H (Pi reflects about |0> on the block), and writes exp(-iHt), t = 22, as a linear combination of the powers W^k, k <= 33, with Jacobi-Anger weights. This file builds the same circuits from QUMO gates (QUMO's amplitude embedding and controlled_by helpers are used for PREP and the controlled powers; everything else is explicit gates): U_H = PREP^dag SEL PREP with a Toffoli-controlled Pauli select, the controlled powers W^(2^i) from controlled_by(), and the complex weights (-i)^k c_k split into a signed real PREP_R, phases i^k on the exponent bits, and PREP_L^dag built from |c_k|. The checks compare the block encoding with H psi / 0.75, every walk power W^k, k = 1..33, with (-1)^k T_k(H/0.75) on all four data basis inputs, the sum of |weights| with the notebook's printed 5.6264790337731325, the projected LCU state with the degree-33 Chebyshev sum, and the rescaled state with exp(-iHt) psi from a Taylor series, at the notebook's tolerance 1e-7. A second run uses the notebook's own random state, recovered from its printed H psi (cell 21, 8 digits; real, unit norm to 2e-9), and compares the block encoding with the printed cell-21 vector and the rescaled LCU state with the printed cell-32 exp(-iHt) psi.

Deviations from the notebook: no Classiq synthesis, BlockEncoding, qubitize or prepare_select, and no cloud simulator; the circuits run on the QUMO statevector simulator. The walk uses Pi = I - 2|0><0| (R = X X CZ X X), the opposite sign of the notebook's eq. (2) convention Pi = 2|0><0| - I, so the top-left block of W^k here is (-1)^k T_k(H/0.75); the LCU compensates with the phases i^k on the exponent bits (-i)^k (-1)^k = i^k, so the evolved state is unchanged. The Jacobi-Anger weights come from Bessel's integral J_n(z) = (1/2 pi) int cos(n tau - z sin tau) by the trapezoid rule, not from poly_jacobi_anger_*; the degree is fixed at 33, the value the notebook prints. The notebook's initial state is np.random.rand, unseeded, so it cannot be regenerated from the source; the main run uses a fixed real vector with a negative entry against the exact evolution, and the second run uses the state recovered from the printed values (limit: that state is only known to about 4e-8, which sets the 1e-7 tolerance of the cell-32 comparison). The reference exp(-iHt) psi is a Taylor series on the 4-dimensional space, not scipy.linalg.expm. Register layout (little-endian): data q0-q1, block of H q2-q3, exponent q4-q9.

Run: .venv/bin/mojo run -I . examples/chemistry_and_physics/hamiltonian_simulation_qubitization.mojo
example-meta: {"framework": "classiq", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from std.math import cos, pi, sin, sqrt
from qumo.algorithms.qml import append_amplitude_embedding_real
from qumo.ir.circuit import Circuit
from qumo.ir.mc import controlled_by
from qumo.sim.runner import evolve
from qumo.testing.checks import CheckLog


comptime NQ = 10  # data q0-q1, block q2-q3, exponent q4-q9
comptime NEXP = 6
comptime DEGREE = 33
comptime EVOLUTION_TIME = 22.0
comptime EPS = 1e-7
comptime NOTEBOOK_WEIGHT_SUM = 5.6264790337731325


def alphas() -> List[Float64]:
    """Pauli weights of H for the terms I, Z1, X0 X1, Z0 Z1; term j is selected by block index j."""
    var out: List[Float64] = [0.4, 0.1, 0.05, 0.2]
    return out^


def alpha_bar() -> Float64:
    """Sum of |alpha_j|: the block-encoding scaling factor."""
    var total = 0.0
    var al = alphas()
    for a in al:
        total += a
    return total


def h_apply(v: List[Float64]) -> List[Float64]:
    """H v for H = 0.4 I + 0.1 Z1 + 0.05 X0 X1 + 0.2 Z0 Z1 on basis states |x>, x = q0 + 2 q1."""
    var out = List[Float64](length=4, fill=0.0)
    for x in range(4):
        var b0 = x & 1
        var b1 = (x >> 1) & 1
        var z1 = Float64(1 - 2 * b1)
        var z01 = Float64((1 - 2 * b0) * (1 - 2 * b1))
        out[x] = (0.4 + 0.1 * z1 + 0.2 * z01) * v[x] + 0.05 * v[x ^ 3]
    return out^


def cheb_vectors(v0: List[Float64], kmax: Int) -> List[List[Float64]]:
    """T_k(H / alpha_bar) v0 for k = 0..kmax, by the three-term recurrence on real 4-vectors."""
    var ab = alpha_bar()
    var out = List[List[Float64]]()
    out.append(v0.copy())
    var first = h_apply(v0)
    for x in range(4):
        first[x] = first[x] / ab
    out.append(first^)
    for k in range(1, kmax):
        var hv = h_apply(out[k])
        var nxt = List[Float64](length=4, fill=0.0)
        for x in range(4):
            nxt[x] = 2.0 * hv[x] / ab - out[k - 1][x]
        out.append(nxt^)
    return out^


def cheb_sum(coefs: List[Float64], cheb: List[List[Float64]]) -> Tuple[List[Float64], List[Float64]]:
    """Returns the sum over k of c_k (-i)^k T_k(H / alpha_bar) v0, given cheb[k] = T_k(H / alpha_bar) v0."""
    var sr = List[Float64](length=4, fill=0.0)
    var si = List[Float64](length=4, fill=0.0)
    for k in range(len(coefs)):
        var m = k % 4
        var cr = 1.0 if m == 0 else (-1.0 if m == 2 else 0.0)
        var ci = -1.0 if m == 1 else (1.0 if m == 3 else 0.0)
        for x in range(4):
            sr[x] += coefs[k] * cr * cheb[k][x]
            si[x] += coefs[k] * ci * cheb[k][x]
    return (sr^, si^)


def bessel_j(n: Int, x: Float64) -> Float64:
    """J_n(x) for integer n from Bessel's integral (1/2 pi) int_0^{2 pi} cos(n tau - x sin tau) d tau, trapezoid rule."""
    var total = 0.0
    var npts = 1024
    for j in range(npts):
        var th = 2.0 * pi * Float64(j) / Float64(npts)
        total += cos(Float64(n) * th - x * sin(th))
    return total / Float64(npts)


def jacobi_anger_coefs(z: Float64, degree: Int) -> List[Float64]:
    """Returns c_k = eps_k J_k(z) (eps_0 = 1, eps_k = 2), so exp(-i z y) = sum_k (-i)^k c_k T_k(y) on [-1, 1]."""
    var out = List[Float64]()
    for k in range(degree + 1):
        var eps = 1.0 if k == 0 else 2.0
        out.append(eps * bessel_j(k, z))
    return out^


def exact_evolve(psi: List[Float64], t: Float64) -> Tuple[List[Float64], List[Float64]]:
    """Returns exp(-i H t) psi from 64 Taylor steps of order 12 each (brute force; H is real and 4 x 4)."""
    var steps = 64
    var dt = t / Float64(steps)
    var ar = psi.copy()
    var ai = List[Float64](length=4, fill=0.0)
    for _ in range(steps):
        var sum_r = ar.copy()
        var sum_i = ai.copy()
        var w_r = ar.copy()
        var w_i = ai.copy()
        for n in range(1, 13):
            var hr = h_apply(w_r)
            var hi = h_apply(w_i)
            var f = dt / Float64(n)
            var nr = List[Float64](length=4, fill=0.0)
            var ni = List[Float64](length=4, fill=0.0)
            for x in range(4):
                # multiply by (-i f): (hr + i hi) -> f hi - i f hr
                nr[x] = f * hi[x]
                ni[x] = -f * hr[x]
            for x in range(4):
                sum_r[x] += nr[x]
                sum_i[x] += ni[x]
            w_r = nr^
            w_i = ni^
        ar = sum_r^
        ai = sum_i^
    return (ar^, ai^)


def data_wires() -> List[Int]:
    """Amplitude wires for the data register: wires[0] = q1 is the most significant bit."""
    var w = List[Int]()
    w.append(1)
    w.append(0)
    return w^


def exp_wires() -> List[Int]:
    """Amplitude wires for the exponent register q4-q9: wires[0] = q9 is the most significant bit."""
    var w = List[Int]()
    for i in range(NEXP):
        w.append(4 + (NEXP - 1 - i))
    return w^


def append_uh(mut c: Circuit) raises:
    """U_H = PREP^dag SEL PREP on data (q0, q1) and block (q2, q3).

    PREP maps the block to sum_j sqrt(alpha_j / alpha_bar) |j>, with block index j = q2 + 2 q3.
    SEL applies P_j on the data when the block equals j: term 1 is Z1, term 2 is X0 X1, term 3 is Z0 Z1,
    term 0 is the identity. Zero bits of j are X-conjugated around each controlled term.
    """
    var al = alphas()
    var ab = alpha_bar()
    var amp = List[Float64]()
    for a in al:
        amp.append(sqrt(a / ab))
    var prep = Circuit(NQ, name="prep_h")
    var wires_h: List[Int] = [3, 2]
    append_amplitude_embedding_real(prep, amp, wires_h)
    c.compose(prep)
    # j = 1 (q2 = 1, q3 = 0): Z on q1
    c.x(3)
    c.ccz(2, 3, 1)
    c.x(3)
    # j = 2 (q2 = 0, q3 = 1): X on q0 and q1
    c.x(2)
    c.ccx(2, 3, 0)
    c.ccx(2, 3, 1)
    c.x(2)
    # j = 3 (q2 = 1, q3 = 1): Z on q0 and q1
    c.ccz(2, 3, 0)
    c.ccz(2, 3, 1)
    c.compose(prep.inverse())


def append_walk(mut c: Circuit) raises:
    """Szegedy walk W = Pi U_H with Pi = R = I - 2|0><0| on the block (q2, q3): R = X X CZ X X."""
    append_uh(c)
    c.x(2)
    c.x(3)
    c.cz(2, 3)
    c.x(2)
    c.x(3)


def lcu_circuit(psi: List[Float64], coefs: List[Float64], walk: Circuit) raises -> Circuit:
    """PREP_L^dag SEL PREP_R on the exponent register, after the data state psi.

    PREP_R|0> = sum_k s_k sqrt(|c_k| / C) |k> (signed), PREP_L|0> = sum_k sqrt(|c_k| / C) |k>, C = sum_k |c_k|.
    The projected coefficient of W^k is then (c_k / C) i^k, with i^k from S on bit 0 and Z on bit 1 of k
    (i^4 = 1 covers the higher bits). Each W^(2^i) is controlled on exponent bit i.
    """
    var total = 0.0
    for c in coefs:
        total += abs(c)
    var amp_r = List[Float64]()
    var amp_l = List[Float64]()
    for c in coefs:
        var mag = sqrt(abs(c) / total)
        amp_l.append(mag)
        amp_r.append(mag if c >= 0.0 else -mag)
    var out = Circuit(NQ, name="lcu_cheb")
    append_amplitude_embedding_real(out, psi, data_wires())
    append_amplitude_embedding_real(out, amp_r, exp_wires())
    out.s(4)
    out.z(5)
    for i in range(NEXP):
        var ctrl = List[Int]()
        ctrl.append(4 + i)
        out.compose(controlled_by(walk.power(1 << i), ctrl))
    var pl = Circuit(NQ, name="prep_l")
    append_amplitude_embedding_real(pl, amp_l, exp_wires())
    out.compose(pl.inverse())
    return out^


def main() raises:
    var log = CheckLog("classiq/hamiltonian_simulation_qubitization")
    var ab = alpha_bar()
    var z = EVOLUTION_TIME * ab

    # Fixed real initial data state with a negative entry.
    var raw: List[Float64] = [0.3, -0.7, 0.2, 0.6]
    var norm2 = 0.0
    for v in raw:
        norm2 += v * v
    var psi = List[Float64]()
    for v in raw:
        psi.append(v / sqrt(norm2))

    _ = log.close("block-encoding scaling is the notebook's 0.75", ab, 0.75, 1e-12)

    # 1. U_H on the data state, projected onto block 0, equals H psi / 0.75.
    var uh = Circuit(NQ, name="uh_check")
    append_amplitude_embedding_real(uh, psi, data_wires())
    append_uh(uh)
    var sv_uh = evolve(uh)
    var h_psi = h_apply(psi)
    var err_uh = 0.0
    for x in range(4):
        err_uh = max(err_uh, abs(sv_uh.re[x] - h_psi[x] / ab))
        err_uh = max(err_uh, abs(sv_uh.im[x]))
    _ = log.close("U_H on block 0 equals H psi / 0.75 (max error)", err_uh, 0.0, 1e-12)

    # 2. Walk powers: top-left block of W^k is (-1)^k T_k(H / 0.75), on all four data basis inputs.
    var walk = Circuit(NQ, name="walk")
    append_walk(walk)
    var cheb_basis = List[List[List[Float64]]]()
    for x in range(4):
        var e = List[Float64](length=4, fill=0.0)
        e[x] = 1.0
        cheb_basis.append(cheb_vectors(e, DEGREE))
    for k in range(1, DEGREE + 1):
        var worst = 0.0
        for x in range(4):
            var prep = Circuit(NQ, name="basis_prep")
            if (x & 1) == 1:
                prep.x(0)
            if (x & 2) == 2:
                prep.x(1)
            prep.compose(walk.power(k))
            var sv_w = evolve(prep)
            var sgn = 1.0 if k % 2 == 0 else -1.0
            for xo in range(4):
                var want = sgn * cheb_basis[x][k][xo]
                worst = max(worst, max(abs(sv_w.re[xo] - want), abs(sv_w.im[xo])))
        _ = log.close("walk power W^" + String(k) + " top-left block is (-1)^k T_k(H/0.75) (max error)", worst, 0.0, 1e-11)

    # 3. Jacobi-Anger weights: sum of |c_k| is the notebook's scaling factor; 34 weights need 6 qubits.
    var coefs = jacobi_anger_coefs(z, DEGREE)
    var weight_sum = 0.0
    for c in coefs:
        weight_sum += abs(c)
    _ = log.close("sum of Jacobi-Anger weights equals the notebook's 5.6264790337731325", weight_sum, NOTEBOOK_WEIGHT_SUM, 1e-12)
    var exp_bits = 0
    while (1 << exp_bits) < len(coefs):
        exp_bits += 1
    _ = log.check("exponent register holds the 34 weights in 6 qubits (notebook block size 6)", exp_bits == NEXP, "bits=" + String(exp_bits))

    # 4. The LCU projected onto zero ancillas, times the weight sum, against the Chebyshev sum and exp(-iHt) psi.
    var lcu = lcu_circuit(psi, coefs, walk)
    var sv_l = evolve(lcu)
    var pr = List[Float64](length=4, fill=0.0)
    var pim = List[Float64](length=4, fill=0.0)
    for x in range(4):
        pr[x] = weight_sum * sv_l.re[x]
        pim[x] = weight_sum * sv_l.im[x]
    var cheb_psi = cheb_vectors(psi, DEGREE)
    var sum_ref = cheb_sum(coefs, cheb_psi)
    var err_sum = 0.0
    for x in range(4):
        err_sum = max(err_sum, max(abs(pr[x] - sum_ref[0][x]), abs(pim[x] - sum_ref[1][x])))
    _ = log.close("LCU projection times the weight sum equals the degree-33 Chebyshev sum (max error)", err_sum, 0.0, 1e-12)

    var exact = exact_evolve(psi, EVOLUTION_TIME)
    var dist = 0.0
    var inner_re = 0.0
    var inner_im = 0.0
    var norm_e = 0.0
    var norm_r = 0.0
    for x in range(4):
        var er = exact[0][x]
        var ei = exact[1][x]
        dist += (pr[x] - er) * (pr[x] - er) + (pim[x] - ei) * (pim[x] - ei)
        inner_re += er * pr[x] + ei * pim[x]
        inner_im += er * pim[x] - ei * pr[x]
        norm_e += er * er + ei * ei
        norm_r += pr[x] * pr[x] + pim[x] * pim[x]
    var dist_norm = sqrt(dist)
    var overlap = sqrt(inner_re * inner_re + inner_im * inner_im) / sqrt(norm_e * norm_r)
    print("weight sum", weight_sum, "| rescaled vs exact distance", dist_norm, "| overlap", overlap)
    _ = log.close("rescaled LCU state matches exp(-iHt) psi within the notebook's 1e-7", dist_norm, 0.0, EPS)
    _ = log.close("overlap of the rescaled LCU state with exp(-iHt) psi is 1", overlap, 1.0, 1e-12)

    # 5. The notebook's own initial state, recovered from its printed H psi (cell 21).
    var nb_psi: List[Float64] = [0.63334593, 0.45804844, 0.15101498, 0.60519339]
    var nb_h: List[Float64] = [0.47360182, 0.14496528, 0.03800392, 0.33426399]
    var nb_exp_re: List[Float64] = [-0.70865551, 0.39552212, 0.07788126, -0.28234406]
    var nb_exp_im: List[Float64] = [-0.1340818, -0.24595992, -0.09805291, 0.40926443]
    var uh2 = Circuit(NQ, name="uh_nb")
    append_amplitude_embedding_real(uh2, nb_psi, data_wires())
    append_uh(uh2)
    var sv_u2 = evolve(uh2)
    var err_nb_h = 0.0
    for x in range(4):
        err_nb_h = max(err_nb_h, abs(sv_u2.re[x] * ab - nb_h[x]))
    _ = log.close("notebook state: U_H block 0 times 0.75 equals the printed cell-21 H psi (max error)", err_nb_h, 0.0, 1e-8)
    var sv_l2 = evolve(lcu_circuit(nb_psi, coefs, walk))
    var dist2 = 0.0
    for x in range(4):
        var dr = weight_sum * sv_l2.re[x] - nb_exp_re[x]
        var di = weight_sum * sv_l2.im[x] - nb_exp_im[x]
        dist2 += dr * dr + di * di
    _ = log.close("notebook state: rescaled LCU state matches the printed cell-32 exp(-iHt) psi within 1e-7", sqrt(dist2), 0.0, EPS)

    log.finish()
