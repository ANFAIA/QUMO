"""H2 potential energy curve: UCCSD VQE against Hartree-Fock and exact (FCI) energies, from QUMO gates.

Reproduces: Classiq library - Creating a Molecule's Potential Energy Curve
Framework: QUMO - https://github.com/ANFAIA/QUMO
Source: https://github.com/Classiq/classiq-library/blob/main/applications/chemistry/molecular_energy_curve/molecular_energy_curve.ipynb (Classiq Library, MIT)

The notebook sweeps the H-H distance over 12 points (0.2 to 1.0 Angstrom in 5 steps, 1.4 to 3.5 in 7 steps), builds the STO-3G Hamiltonian of H2 with openfermion and PySCF, runs a first-order Trotterised UCCSD VQE (singles and doubles) from the Hartree-Fock state, starting from all-zero parameters, and plots it against the Hartree-Fock and FCI energies. It prints no energies, only the distances. This file does the same sweep in QUMO: the STO-3G integrals are computed here from the s-type Gaussian primitives (overlap, kinetic, nuclear attraction and two-electron integrals through the Boys function), the 4-qubit Jordan-Wigner Hamiltonian is assembled from the molecular-orbital integrals, and the UCCSD ansatz uccsd_ansatz(4, 2) (QUMO's h2_ansatz, 3 parameters: two singles and one double) is optimised with BFGS from all-zero parameters. The checks use references that do not come from QUMO's VQE: the six integrals and the HF and FCI energies at 0.735 Angstrom against the constants QUMO stores from PySCF; the assembled Hamiltonian at 0.735 Angstrom against QUMO's stored h2_sto3g(4) (same lowest eigenvalue to 1e-7, the precision of the stored constants); the lowest eigenvalue of the assembled Hamiltonian (exact diagonalisation) against the closed-form value at every distance; the Hartree-Fock energy at every distance against 2 h_gg + J_gg + E_nuc; the FCI energy at every distance against the closed-form lowest eigenvalue of the 2 x 2 singlet configuration-interaction matrix [[2 h_gg + J_gg, K_gu], [K_gu, 2 h_uu + J_uu]] + E_nuc; the VQE energy against that value at every distance; the variational bound HF >= VQE; the curve minimum near 0.74 Angstrom; the dissociation limit, with an independent one-centre calculation of the STO-3G hydrogen atom energy (-0.4665819 Ha, so two atoms give -0.9331637 Ha) compared with the VQE at 30 Angstrom, and the limit HF - FCI = (AA|AA)/2 - 1/(2R) (R in bohr) from the one-centre two-electron integral; and the qualitative claim of the notebook that the HF error grows with distance while VQE stays on the exact curve.

Deviations from the notebook: openfermion and PySCF are not run; the integrals come from this file's own Gaussian integral code (the standard STO-3G exponents and contraction coefficients of hydrogen, no scaling), checked against PySCF-derived constants at 0.735 Angstrom and, through the atomic limit, at large distance; the other distances rest on that code and on the closed-form 2 x 2 CI formula built from the same integrals, so they are not checked against PySCF. The Hamiltonian is the full 4-qubit Jordan-Wigner one, not the Z2-tapered mapper of the notebook (Z2SymTaperMapper), so the ansatz is the untapered UCCSD with 3 parameters; the parameter count of the notebook's tapered ansatz is not printed and is not reproduced or checked. The ansatz is exactly simulated and the optimiser is BFGS with adjoint gradients rather than Classiq's variational_minimize with 500 iterations. The notebook's per-point durations and its plot are not reproduced.

Run: .venv/bin/mojo run -I . examples/frameworks/classiq/molecular_energy_curve.mojo
example-meta: {"framework": "classiq", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from std.math import exp, sqrt, pi, erf
from qumo.algorithms.library import h2_ansatz
from qumo.algorithms.hamiltonians import (
    h2_sto3g, ground_energy,
    H2_H_GG, H2_H_UU, H2_J_GG, H2_J_UU, H2_J_GU, H2_K_GU,
    H2_NUCLEAR_REPULSION, H2_EXACT_ENERGY, H2_HF_ENERGY, H2_BOND_LENGTH,
)
from qumo.ir.pauli import PauliSum
from qumo.algorithms.variational import vqe
from qumo.testing.checks import CheckLog

comptime ANGSTROM_TO_BOHR = 1.0 / 0.529177210903
comptime NPRIM = 3
comptime NPTS = 12


def prim_exp(i: Int) -> Float64:
    if i == 0:
        return 3.42525091
    if i == 1:
        return 0.62391373
    return 0.16885540


def prim_coef(i: Int) -> Float64:
    if i == 0:
        return 0.15432897
    if i == 1:
        return 0.53532814
    return 0.44463454


def boys0(t: Float64) -> Float64:
    if t < 1e-10:
        return 1.0 - t / 3.0
    return 0.5 * sqrt(pi / t) * erf(sqrt(t))


def norm(a: Float64) -> Float64:
    return (2.0 * a / pi) ** 0.75


def centre(k: Int, r: Float64) -> Float64:
    """Z coordinate (bohr) of AO k: atom 0 at the origin, atom 1 at r."""
    return 0.0 if k == 0 else r


def ao_overlap(k: Int, l: Int, r: Float64) -> Float64:
    var s = 0.0
    var d2 = (centre(k, r) - centre(l, r)) ** 2
    for i in range(NPRIM):
        for j in range(NPRIM):
            var a = prim_exp(i)
            var b = prim_exp(j)
            var p = a + b
            s += prim_coef(i) * prim_coef(j) * norm(a) * norm(b) * (pi / p) ** 1.5 * exp(-a * b / p * d2)
    return s


def ao_core(k: Int, l: Int, r: Float64) -> Float64:
    """Kinetic + nuclear attraction (both protons, Z = 1) matrix element."""
    var zk = centre(k, r)
    var zl = centre(l, r)
    var d2 = (zk - zl) ** 2
    var h = 0.0
    for i in range(NPRIM):
        for j in range(NPRIM):
            var a = prim_exp(i)
            var b = prim_exp(j)
            var p = a + b
            var mu = a * b / p
            var pref = prim_coef(i) * prim_coef(j) * norm(a) * norm(b)
            var ov = (pi / p) ** 1.5 * exp(-mu * d2)
            h += pref * mu * (3.0 - 2.0 * mu * d2) * ov
            var zp = (a * zk + b * zl) / p
            for nuc in range(2):
                var zc = 0.0 if nuc == 0 else r
                h -= pref * 2.0 * pi / p * exp(-mu * d2) * boys0(p * (zp - zc) ** 2)
    return h


def ao_eri(k: Int, l: Int, m: Int, n: Int, r: Float64) -> Float64:
    """Chemist-notation (kl|mn) over s-type contracted Gaussians."""
    var zk = centre(k, r)
    var zl = centre(l, r)
    var zm = centre(m, r)
    var zn = centre(n, r)
    var out = 0.0
    for i in range(NPRIM):
        for j in range(NPRIM):
            var a = prim_exp(i)
            var b = prim_exp(j)
            var p = a + b
            var zp = (a * zk + b * zl) / p
            var kab = exp(-a * b / p * (zk - zl) ** 2)
            for u in range(NPRIM):
                for v in range(NPRIM):
                    var c = prim_exp(u)
                    var d = prim_exp(v)
                    var q = c + d
                    var zq = (c * zm + d * zn) / q
                    var kcd = exp(-c * d / q * (zm - zn) ** 2)
                    var pref = prim_coef(i) * prim_coef(j) * prim_coef(u) * prim_coef(v) * norm(a) * norm(b) * norm(c) * norm(d)
                    out += pref * 2.0 * pi ** 2.5 / (p * q * sqrt(p + q)) * kab * kcd * boys0(p * q / (p + q) * (zp - zq) ** 2)
    return out


def atom_energy() -> Float64:
    """One-centre STO-3G hydrogen atom energy: kinetic + attraction of its own proton."""
    var h = 0.0
    for i in range(NPRIM):
        for j in range(NPRIM):
            var a = prim_exp(i)
            var b = prim_exp(j)
            var p = a + b
            var mu = a * b / p
            var pref = prim_coef(i) * prim_coef(j) * norm(a) * norm(b)
            h += pref * (mu * 3.0 * (pi / p) ** 1.5 - 2.0 * pi / p)
    return h


def atom_eri() -> Float64:
    """One-centre (AA|AA) over the contracted s Gaussian."""
    var out = 0.0
    for i in range(NPRIM):
        for j in range(NPRIM):
            for u in range(NPRIM):
                for v in range(NPRIM):
                    var a = prim_exp(i)
                    var b = prim_exp(j)
                    var c = prim_exp(u)
                    var d = prim_exp(v)
                    var p = a + b
                    var q = c + d
                    var pref = prim_coef(i) * prim_coef(j) * prim_coef(u) * prim_coef(v) * norm(a) * norm(b) * norm(c) * norm(d)
                    out += pref * 2.0 * pi ** 2.5 / (p * q * sqrt(p + q))
    return out


struct H2Integrals(Copyable):
    var hg: Float64
    var hu: Float64
    var jg: Float64
    var ju: Float64
    var jgu: Float64
    var kgu: Float64
    var enuc: Float64

    def __init__(out self, r_angstrom: Float64):
        var r = r_angstrom * ANGSTROM_TO_BOHR
        var s = ao_overlap(0, 1, r)
        var cg = 1.0 / sqrt(2.0 * (1.0 + s))
        var cu = 1.0 / sqrt(2.0 * (1.0 - s))
        var h00 = ao_core(0, 0, r)
        var h01 = ao_core(0, 1, r)
        self.hg = 2.0 * cg * cg * (h00 + h01)
        self.hu = 2.0 * cu * cu * (h00 - h01)
        var cgv = List[Float64]()
        cgv.append(cg)
        cgv.append(cg)
        var cuv = List[Float64]()
        cuv.append(cu)
        cuv.append(-cu)
        var eri = List[Float64]()
        for k in range(2):
            for l in range(2):
                for m in range(2):
                    for n in range(2):
                        eri.append(ao_eri(k, l, m, n, r))
        self.jg = 0.0
        self.ju = 0.0
        self.jgu = 0.0
        self.kgu = 0.0
        for k in range(2):
            for l in range(2):
                for m in range(2):
                    for n in range(2):
                        var e = eri[k * 8 + l * 4 + m * 2 + n]
                        self.jg += cgv[k] * cgv[l] * cgv[m] * cgv[n] * e
                        self.ju += cuv[k] * cuv[l] * cuv[m] * cuv[n] * e
                        self.jgu += cgv[k] * cgv[l] * cuv[m] * cuv[n] * e
                        self.kgu += cgv[k] * cuv[l] * cgv[m] * cuv[n] * e
        self.enuc = 1.0 / r


def bit(q: Int) -> UInt64:
    return UInt64(1) << UInt64(q)


def jw_hamiltonian(p: H2Integrals) raises -> PauliSum:
    """4-qubit Jordan-Wigner H2 (q0 g-up, q1 g-down, q2 u-up, q3 u-down); compared with h2_sto3g(4) in main."""
    var h = PauliSum()
    h.add(p.hg + p.hu + p.jg / 4 + p.ju / 4 + p.jgu - p.kgu / 2 + p.enuc, 0, 0)
    var zg = -p.hg / 2 - p.jg / 4 - p.jgu / 2 + p.kgu / 4
    var zu = -p.hu / 2 - p.ju / 4 - p.jgu / 2 + p.kgu / 4
    h.add(zg, 0, bit(0))
    h.add(zg, 0, bit(1))
    h.add(zu, 0, bit(2))
    h.add(zu, 0, bit(3))
    h.add(p.jg / 4, 0, bit(0) | bit(1))
    h.add(p.ju / 4, 0, bit(2) | bit(3))
    h.add((p.jgu - p.kgu) / 4, 0, bit(0) | bit(2))
    h.add((p.jgu - p.kgu) / 4, 0, bit(1) | bit(3))
    h.add(p.jgu / 4, 0, bit(0) | bit(3))
    h.add(p.jgu / 4, 0, bit(1) | bit(2))
    var c = p.kgu / 4
    h.add_label(-c, "YYXX")
    h.add_label(c, "XYYX")
    h.add_label(c, "YXXY")
    h.add_label(-c, "XXYY")
    return h^


def linspace(start: Float64, stop: Float64, num: Int, mut out: List[Float64]):
    for i in range(num):
        out.append(start + (stop - start) * Float64(i) / Float64(num - 1))


def main() raises:
    var chk = CheckLog("molecular_energy_curve")

    # 1. integrals against the PySCF-derived constants at 0.735 Angstrom
    var p0 = H2Integrals(H2_BOND_LENGTH)
    _ = chk.close("h_gg at 0.735", p0.hg, H2_H_GG, 1e-6)
    _ = chk.close("h_uu at 0.735", p0.hu, H2_H_UU, 1e-6)
    _ = chk.close("J_gg at 0.735", p0.jg, H2_J_GG, 1e-6)
    _ = chk.close("J_uu at 0.735", p0.ju, H2_J_UU, 1e-6)
    _ = chk.close("J_gu at 0.735", p0.jgu, H2_J_GU, 1e-6)
    _ = chk.close("K_gu at 0.735", p0.kgu, H2_K_GU, 1e-6)
    _ = chk.close("E_nuc at 0.735", p0.enuc, H2_NUCLEAR_REPULSION, 1e-6)
    var hf0 = 2 * p0.hg + p0.jg + p0.enuc
    _ = chk.close("HF energy at 0.735", hf0, H2_HF_ENERGY, 1e-6)
    var a0 = 2 * p0.hg + p0.jg
    var b0 = 2 * p0.hu + p0.ju
    var fci0 = (a0 + b0) / 2 - sqrt(((a0 - b0) / 2) ** 2 + p0.kgu ** 2) + p0.enuc
    _ = chk.close("FCI energy at 0.735", fci0, H2_EXACT_ENERGY, 1e-6)

    var hq = jw_hamiltonian(p0)
    var href = h2_sto3g(4)
    _ = chk.close("lowest eigenvalue of assembled H vs h2_sto3g(4) (stored constants have 8 digits)", ground_energy(hq), ground_energy(href), 1e-7)
    _ = chk.close("lowest eigenvalue of assembled H vs PySCF FCI", ground_energy(hq), H2_EXACT_ENERGY, 1e-6)

    # 2. the sweep of the notebook
    var dist = List[Float64]()
    linspace(0.20, 1.0, 5, dist)
    linspace(1.4, 3.5, 7, dist)
    _ = chk.check("12 sampling points", len(dist) == NPTS)
    _ = chk.close("last distance", dist[NPTS - 1], 3.5, 1e-12)
    _ = chk.close("distance 8", dist[7], 2.1, 1e-12)

    var vqe_e = List[Float64]()
    var hf_e = List[Float64]()
    var fci_e = List[Float64]()
    var worst = 0.0
    var worst_spec = 0.0
    var hf_below_vqe = True
    var x0 = List[Float64](length=3, fill=0.0)
    for i in range(NPTS):
        var p = H2Integrals(dist[i])
        var h = jw_hamiltonian(p)
        var ans = h2_ansatz()
        if i == 0:
            _ = chk.check("UCCSD ansatz has 3 parameters", ans.num_params() == 3)
        var res = vqe(ans, h, "bfgs", x0=x0, maxiter=300, tol=1e-10)
        var hf = 2 * p.hg + p.jg + p.enuc
        var a = 2 * p.hg + p.jg
        var b = 2 * p.hu + p.ju
        var fci = (a + b) / 2 - sqrt(((a - b) / 2) ** 2 + p.kgu ** 2) + p.enuc
        vqe_e.append(res.energy)
        hf_e.append(hf)
        fci_e.append(fci)
        var err = abs(res.energy - fci)
        if err > worst:
            worst = err
        var serr = abs(ground_energy(h) - fci)
        if serr > worst_spec:
            worst_spec = serr
        if hf < res.energy - 1e-9:
            hf_below_vqe = False
        print("  R =", dist[i], "A  VQE =", res.energy, " HF =", hf, " FCI =", fci)
    _ = chk.within("max |VQE - FCI| over the 12 points", worst, 0.0, 1e-9)
    _ = chk.within("max |lowest eigenvalue of H - closed-form FCI| over the 12 points", worst_spec, 0.0, 1e-9)
    _ = chk.check("HF >= VQE at every distance (variational)", hf_below_vqe)

    # 3. HF error grows with distance (claim of the notebook) up to the analytic limit (AA|AA)/2
    var hf_gap_grows = True
    for i in range(1, NPTS):
        if dist[i] > 1.0 and (hf_e[i] - fci_e[i]) < (hf_e[i - 1] - fci_e[i - 1]) - 1e-9:
            hf_gap_grows = False
    _ = chk.check("HF error grows with distance beyond 1 A", hf_gap_grows)
    var eatom = atom_energy()
    _ = chk.close("one-centre STO-3G H atom energy (literature -0.4665819)", eatom, -0.4665819, 1e-6)
    var jaa = atom_eri()
    _ = chk.check("HF error at 3.5 A below the dissociation limit (AA|AA)/2", hf_e[NPTS - 1] - fci_e[NPTS - 1] < jaa / 2)
    var pf = H2Integrals(30.0)
    var hf_far = 2 * pf.hg + pf.jg + pf.enuc
    var vf = vqe(h2_ansatz(), jw_hamiltonian(pf), "bfgs", x0=x0, maxiter=300, tol=1e-10)
    _ = chk.close("VQE at 30 A equals two H atoms", vf.energy, 2.0 * eatom, 1e-6)
    _ = chk.close("HF - VQE at 30 A equals (AA|AA)/2 - 1/(2R)", hf_far - vf.energy, jaa / 2 - 0.5 / (30.0 * ANGSTROM_TO_BOHR), 1e-6)

    # 4. curve minimum near 0.74 Angstrom: lowest sampled point is 0.8, below 0.6 and 1.0
    var imin = 0
    for i in range(NPTS):
        if vqe_e[i] < vqe_e[imin]:
            imin = i
    _ = chk.check("lowest sampled VQE energy at 0.8 A", imin == 3)
    _ = chk.check("repulsive wall at 0.2 A (positive energy)", vqe_e[0] > 0.0)

    chk.finish()
