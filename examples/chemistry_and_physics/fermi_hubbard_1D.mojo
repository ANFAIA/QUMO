"""One-dimensional Fermi-Hubbard model: Slater-determinant state, five-stage Trotter quench, charge and spin densities.

Reproduces: Classiq library - One-Dimensional Fermi-Hubbard Model
Source: https://github.com/Classiq/classiq-library/blob/main/applications/physical_systems/fermi_hubbard_model_1D/fermi_hubbard_1D.ipynb (Classiq Library, MIT)
Framework: Classiq - https://github.com/Classiq/classiq-library

The notebook prepares the ground state of a non-interacting Hubbard chain with a Gaussian potential on the spin-up species (N = 4 sites, 8 Jordan-Wigner qubits, spin-up on qubits 0..3 and spin-down on 4..7) as a Slater determinant of adjacent-mode Givens rotations, quenches with a first-order Trotter step tau = 0.1 (J = 1, U = 3) built from the five stages hop-odd, interact-odd, fermionic mode swap, interact-even, hop-even + inverse swap, and measures the charge and spin densities and spreads at num_iter = 0..9. It then validates the circuit in the two exactly solvable limits: U = 0 with one electron and J = 0 at half filling.
This file rebuilds that circuit and checks it against independent references. The notebook's plots of the interacting run are not reproduced and no printed values exist for them; the notebook prints only the one-electron fidelity 1.0 and finite-shot J = 0 deviations of 4.00e-02 and 5.03e-02. Note the sign issue of deviation (3): the notebook's interacting circuit realises U = -3, its text describes U = +3. This file runs both and checks each against the exact evolution of its own sign.
Checks: the Slater state against the determinant amplitudes det Q[:, S] of the occupied orbitals (fidelity 1; one-electron case also against the eigenvector; the two occupied orbitals are one spin-up and one spin-down), the t = 0 densities against sum_k Q_k,mu^2, K(-pi/2) acting as an iSWAP, the interacting Trotter densities and spreads at every step against an independent exact evolution of the 256-dimensional Fock-space Hamiltonian (fermionic sign rules, Taylor series) for U = +3 and for U = -3, the first-order error halving at tau/2 and tau/4, particle number conservation, the U = 0 occupations against the closed-form standing-wave solution, and for J = 0 both the frozen densities and the full state against exp(-i U t D(s)) times the Slater amplitudes (densities alone cannot see the interaction layers; the full-state check does).
Limits: the absolute Trotter tolerances (0.005 for U = +3 at tau = 0.1, 0.01 for U = -3 at tau/4, 0.035 for U = 0) are accuracy statements chosen from the observed convergence, not rigorous Trotter bounds (the commutator bound is vacuous for U = 3); the first-order scaling checks guard against wrong angles. The charge/spin spreads are compared with the exact ones; this file makes no independent claim about spin-charge separation beyond that the exact and Trotter spreads agree and U = 3 differs from U = 0.

Deviations from the notebook: (1) OpenFermion's givens_decomposition and Classiq's unitary gate are replaced by a real adjacent-column Givens sweep (h is real) and the gate CX, controlled-RY, CX (as in fermionic_gaussian.mojo); the circuits differ in depth and the transpiled depth 13 is not reproduced. (2) observe() with finite-shot sampling is replaced by exact expectation values of the same Pauli observables. (3) Interaction sign: the main run applies exp(-i tau U n_up n_down) (controlled phase of -tau U), the repulsive U = +3 the notebook describes. The notebook's code, P(phi) = phase(phi n_up n_down), applies exp(+i phi n n) (its own comment says diag(1,1,1,e^{-i phi}), which is the other sign), i.e. attractive U = -3. The densities of the two signs are not equal (they differ by up to 0.26 for this state), so the notebook's plots belong to U = -3; this file also runs that circuit (U = -3) and verifies it against the exact U = -3 evolution. (4) The notebook's prose says P(phi = tau U / 2) but its code uses phi = tau U; the code value is used. (5) K(theta) = exp(-i theta/2 (XX + YY)) is RXX(theta) RYY(theta) instead of suzuki_trotter. (6) The spread uses the notebook's own centre (N+1)/2 for 0-indexed sites. (7) The notebook's plots are not drawn, and its Classiq-library eigenvalue sanity check for U = 0 is replaced by the check of the occupations.

Run: .venv/bin/mojo run -I . examples/chemistry_and_physics/fermi_hubbard_1D.mojo
example-meta: {"framework": "classiq", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from std.math import atan2, cos, exp, sin, sqrt
from std.bit import pop_count
from qumo.ir.circuit import Circuit
from qumo.ir.pauli import PauliSum
from qumo.algorithms.linalg import jacobi_eigh
from qumo.sim.runner import evolve
from qumo.sim.statevector import StateVector
from qumo.testing.checks import CheckLog

comptime PI = 3.141592653589793


# ---------------------------------------------------------------------------
# Lattice bookkeeping (the notebook's qubit_idx: spin-up 0..N-1, spin-down N..2N-1)
# ---------------------------------------------------------------------------
def qubit_idx(site: Int, spin: Int, n: Int) -> Int:
    return site + spin * n


def single_electron_h(n: Int, J: Float64, lam: Float64, mean: Float64, std: Float64) -> List[Float64]:
    """The notebook's kinetic_energy + spin_potential(spin=0): open chain hopping -J on both spins, Gaussian well on spin-up."""
    var m = 2 * n
    var h = List[Float64](length=m * m, fill=0.0)
    for site in range(n - 1):
        for spin in range(2):
            var a = qubit_idx(site, spin, n)
            var b = qubit_idx(site + 1, spin, n)
            h[a * m + b] = -J
            h[b * m + a] = -J
    for site in range(n):
        var mu = qubit_idx(site, 0, n)
        var d = Float64(site) - mean
        h[mu * m + mu] += -lam * exp(-(d * d) / (2.0 * std * std))
    return h^


# ---------------------------------------------------------------------------
# Slater determinant: orbitals, circuit, exact amplitudes
# ---------------------------------------------------------------------------
def orbitals(h: List[Float64], m: Int, nelec: Int) -> List[Float64]:
    """Q (nelec x m, row-major): the nelec lowest eigenvectors of h as rows (the notebook's Q = Qbar.T[:Nelec])."""
    var eig = jacobi_eigh(h, m)
    var q = List[Float64](length=nelec * m, fill=0.0)
    for k in range(nelec):
        var v = eig.vector(k)
        for mu in range(m):
            q[k * m + mu] = v[mu]
    return q^


def excitation_givens(mut c: Circuit, p: Int, q: Int, theta: Float64):
    """On the one-excitation pair: |e_p> -> cos(t/2)|e_p> + sin(t/2)|e_q>, |e_q> -> -sin(t/2)|e_p> + cos(t/2)|e_q>;
    identity on |00> and |11>. CX(p -> q), controlled-RY(-t) (control q, target p), CX(p -> q)."""
    c.cx(p, q)
    c.cry(-theta, q, p)
    c.cx(p, q)


def prepare_slater(mut c: Circuit, q_in: List[Float64], m: Int, nelec: Int):
    """X on modes 0..nelec-1, then the inverse of the column sweep that reduces Q to (D, 0)."""
    var q = q_in.copy()
    var rp = List[Int]()
    var rt = List[Float64]()
    for r in range(nelec):
        var col = m - 1
        while col > r:
            var p = col - 1
            var a = q[r * m + p]
            var b = q[r * m + col]
            var rad = sqrt(a * a + b * b)
            if rad > 1e-14:
                var cs = a / rad
                var sn = b / rad
                for k in range(nelec):
                    var x = q[k * m + p]
                    var y = q[k * m + col]
                    q[k * m + p] = cs * x + sn * y
                    q[k * m + col] = -sn * x + cs * y
                rp.append(p)
                rt.append(2.0 * atan2(sn, cs))
            col -= 1
    for i in range(nelec):
        c.x(i)
    var g = len(rp) - 1
    while g >= 0:
        excitation_givens(c, rp[g], rp[g] + 1, rt[g])
        g -= 1


def det(a: List[Float64], m: Int) -> Float64:
    var w = a.copy()
    var d = 1.0
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
            d = -d
        d *= w[c * m + c]
        for r in range(c + 1, m):
            var f = w[r * m + c] / w[c * m + c]
            for k in range(c, m):
                w[r * m + k] -= f * w[c * m + k]
    return d


def slater_amplitudes(q: List[Float64], m: Int, nelec: Int) -> List[Float64]:
    """Fock amplitudes of prod_k (sum_mu Q_k,mu c^+_mu)|0>: det Q[:, S] for the occupied set S in ascending order."""
    var amp = List[Float64](length=1 << m, fill=0.0)
    for s in range(1 << m):
        if Int(pop_count(s)) != nelec:
            continue
        var cols = List[Int]()
        for mu in range(m):
            if ((s >> mu) & 1) == 1:
                cols.append(mu)
        var sub = List[Float64](length=nelec * nelec, fill=0.0)
        for k in range(nelec):
            for j in range(nelec):
                sub[k * nelec + j] = q[k * m + cols[j]]
        amp[s] = det(sub, nelec)
    return amp^


def fidelity_with(sv: StateVector, amp: List[Float64]) -> Float64:
    var re = 0.0
    var im = 0.0
    for s in range(len(amp)):
        re += amp[s] * sv.re[s]
        im += amp[s] * sv.im[s]
    return re * re + im * im


# ---------------------------------------------------------------------------
# Trotter step (the notebook's K, P, hop, interact, trotter_step)
# ---------------------------------------------------------------------------
def K(mut c: Circuit, theta: Float64, a: Int, b: Int):
    """K = exp(-i theta/2 (XX + YY)); XX and YY commute, so it is RXX(theta) RYY(theta)."""
    c.rxx(theta, a, b)
    c.ryy(theta, a, b)


def hop(mut c: Circuit, pairs: List[Tuple[Int, Int]], theta: Float64, n: Int):
    for spin in range(2):
        for pr in pairs:
            K(c, theta, qubit_idx(pr[0], spin, n), qubit_idx(pr[1], spin, n))


def interact(mut c: Circuit, sites: List[Int], phi: Float64, n: Int):
    """P = exp(-i phi n_up n_down) on each listed site, as a controlled phase of -phi."""
    for site in sites:
        c.cp(-phi, qubit_idx(site, 0, n), qubit_idx(site, 1, n))


struct Lattice(Copyable, Movable):
    var n: Int
    var odd_pairs: List[Tuple[Int, Int]]
    var even_pairs: List[Tuple[Int, Int]]
    var odd_sites: List[Int]
    var even_sites: List[Int]
    var swapped_even_sites: List[Int]

    def __init__(out self, n: Int):
        self.n = n
        self.odd_pairs = List[Tuple[Int, Int]]()
        self.even_pairs = List[Tuple[Int, Int]]()
        self.odd_sites = List[Int]()
        self.even_sites = List[Int]()
        self.swapped_even_sites = List[Int]()
        for i in range(0, n - 1, 2):
            self.odd_pairs.append((i, i + 1))
        for i in range(1, n - 1, 2):
            self.even_pairs.append((i, i + 1))
        for i in range(0, n, 2):
            self.odd_sites.append(i)
        for i in range(1, n, 2):
            self.even_sites.append(i)
        # Where the even sites sit after the swap on the even pairs.
        var perm = List[Int]()
        for i in range(n):
            perm.append(i)
        for pr in self.even_pairs:
            var t = perm[pr[0]]
            perm[pr[0]] = perm[pr[1]]
            perm[pr[1]] = t
        var inv = List[Int](length=n, fill=0)
        for pos in range(n):
            inv[perm[pos]] = pos
        for s in self.even_sites:
            self.swapped_even_sites.append(inv[s])
        # sorted ascending
        for i in range(len(self.swapped_even_sites)):
            for j in range(i + 1, len(self.swapped_even_sites)):
                if self.swapped_even_sites[j] < self.swapped_even_sites[i]:
                    var t2 = self.swapped_even_sites[i]
                    self.swapped_even_sites[i] = self.swapped_even_sites[j]
                    self.swapped_even_sites[j] = t2


def trotter_step(mut c: Circuit, lat: Lattice, tau: Float64, J: Float64, U: Float64):
    hop(c, lat.odd_pairs, -tau * J, lat.n)  # stage 1
    interact(c, lat.odd_sites, tau * U, lat.n)  # stage 2
    hop(c, lat.even_pairs, -PI / 2.0, lat.n)  # stage 3: fermionic mode swap
    interact(c, lat.swapped_even_sites, tau * U, lat.n)  # stage 4
    hop(c, lat.even_pairs, -tau * J + PI / 2.0, lat.n)  # stage 5: even hop + inverse swap


def build_circuit(q: List[Float64], lat: Lattice, nelec: Int, num_iter: Int, tau: Float64, J: Float64, U: Float64) -> Circuit:
    var m = 2 * lat.n
    var c = Circuit(m, name="fermi_hubbard")
    prepare_slater(c, q, m, nelec)
    for _ in range(num_iter):
        trotter_step(c, lat, tau, J, U)
    return c^


# ---------------------------------------------------------------------------
# Observables
# ---------------------------------------------------------------------------
def density_observables(n: Int) -> List[PauliSum]:
    """The notebook's charge_density(site) for site in 0..n-1, then spin_density(site): (I - Z)/2 sums and (Z_down - Z_up)/2."""
    var obs = List[PauliSum]()
    for site in range(n):
        var s = PauliSum()
        for spin in range(2):
            s.add(0.5, UInt64(0), UInt64(0))
            s.add(-0.5, UInt64(0), UInt64(1) << UInt64(qubit_idx(site, spin, n)))
        obs.append(s^)
    for site in range(n):
        var s = PauliSum()
        s.add(0.5, UInt64(0), UInt64(1) << UInt64(qubit_idx(site, 1, n)))
        s.add(-0.5, UInt64(0), UInt64(1) << UInt64(qubit_idx(site, 0, n)))
        obs.append(s^)
    return obs^


def measure(sv: StateVector, obs: List[PauliSum]) raises -> List[Float64]:
    var out = List[Float64]()
    for i in range(len(obs)):
        out.append(sv.expectation(obs[i]))
    return out^


def spread(density: List[Float64], offset: Int, n: Int) -> Float64:
    """The notebook's spread: sum_j |j - (N+1)/2| rho_j."""
    var s = 0.0
    for site in range(n):
        s += abs(Float64(site) - Float64(n + 1) / 2.0) * density[offset + site]
    return s


# ---------------------------------------------------------------------------
# Independent exact reference: Fock-space Hamiltonian from fermionic sign rules
# ---------------------------------------------------------------------------
struct SparseH(Movable):
    var rows: List[Int]
    var cols: List[Int]
    var vals: List[Float64]

    def __init__(out self):
        self.rows = List[Int]()
        self.cols = List[Int]()
        self.vals = List[Float64]()


def build_hamiltonian(n: Int, J: Float64, U: Float64) -> SparseH:
    """H = -J sum_{j,s} (c+_j c_{j+1} + h.c.) + U sum_j n_j,up n_j,dn on 2n modes (mode = site + spin * n).
    c+_i c_j on a basis state flips bits j and i with sign (-1)^(occupied modes strictly between i and j)."""
    var h = SparseH()
    var m = 2 * n
    for s in range(1 << m):
        var diag = 0.0
        for site in range(n):
            if ((s >> qubit_idx(site, 0, n)) & 1) == 1 and ((s >> qubit_idx(site, 1, n)) & 1) == 1:
                diag += U
        if diag != 0.0:
            h.rows.append(s)
            h.cols.append(s)
            h.vals.append(diag)
        for spin in range(2):
            for site in range(n - 1):
                var i = qubit_idx(site, spin, n)
                var j = qubit_idx(site + 1, spin, n)
                var bi = (s >> i) & 1
                var bj = (s >> j) & 1
                if bi != bj:
                    var t = s ^ (1 << i) ^ (1 << j)
                    var between = Int(pop_count((s >> (i + 1)) & ((1 << (j - i - 1)) - 1)))
                    var sign = -1.0 if (between & 1) == 1 else 1.0
                    h.rows.append(t)
                    h.cols.append(s)
                    h.vals.append(-J * sign)
    return h^


def taylor_step(h: SparseH, mut re: List[Float64], mut im: List[Float64], dt: Float64):
    """Advance psi <- exp(-i H dt) psi by a degree-16 Taylor series (|H dt| is below 1 here)."""
    var dim = len(re)
    var tr = re.copy()
    var ti = im.copy()
    for k in range(1, 17):
        var nr = List[Float64](length=dim, fill=0.0)
        var ni = List[Float64](length=dim, fill=0.0)
        for e in range(len(h.vals)):
            nr[h.rows[e]] += h.vals[e] * tr[h.cols[e]]
            ni[h.rows[e]] += h.vals[e] * ti[h.cols[e]]
        # term_k = (-i dt / k) H term_{k-1}:  (a + ib) * (-i c) = c b - i c a
        var f = dt / Float64(k)
        for s in range(dim):
            tr[s] = f * ni[s]
            ti[s] = -f * nr[s]
            re[s] += tr[s]
            im[s] += ti[s]


def occupations(re: List[Float64], im: List[Float64], m: Int) -> List[Float64]:
    var occ = List[Float64](length=m, fill=0.0)
    for s in range(len(re)):
        var p = re[s] * re[s] + im[s] * im[s]
        for mu in range(m):
            if ((s >> mu) & 1) == 1:
                occ[mu] += p
    return occ^


def densities_from_occupations(occ: List[Float64], n: Int) -> List[Float64]:
    """[rho+_0..rho+_{n-1}, rho-_0..rho-_{n-1}] with rho+ = n_up + n_dn and rho- = n_up - n_dn = (Z_dn - Z_up)/2."""
    var out = List[Float64](length=2 * n, fill=0.0)
    for site in range(n):
        var up = occ[qubit_idx(site, 0, n)]
        var dn = occ[qubit_idx(site, 1, n)]
        out[site] = up + dn
        out[n + site] = up - dn
    return out^


def max_abs_diff(a: List[Float64], b: List[Float64]) -> Float64:
    var d = 0.0
    for i in range(len(a)):
        d = max(d, abs(a[i] - b[i]))
    return d


def exact_trajectory(
    amp: List[Float64], n: Int, J: Float64, U: Float64, tau: Float64, steps: Int
) -> List[List[Float64]]:
    """Densities of the exactly evolved Slater state at t = k tau for k = 0..steps-1."""
    var m = 2 * n
    var h = build_hamiltonian(n, J, U)
    var re = amp.copy()
    var im = List[Float64](length=len(amp), fill=0.0)
    var out = List[List[Float64]]()
    for _ in range(steps):
        out.append(densities_from_occupations(occupations(re, im, m), n))
        for _ in range(4):
            taylor_step(h, re, im, tau / 4.0)
    return out^


def main() raises:
    var log = CheckLog("classiq/fermi_hubbard_1D")
    var n = 4
    var m = 2 * n
    var J = 1.0
    var tau = 0.1 / J
    var num_iter = 10
    var obs = density_observables(n)

    # ---------------------------------------------------- gate convention: K(-pi/2) is an iSWAP
    var kc = Circuit(2, name="kgate")
    kc.x(0)  # |01> in (q1 q0) order: qubit 0 set
    K(kc, -PI / 2.0, 0, 1)
    var ksv = evolve(kc)
    _ = log.close("K(-pi/2)|10> -> i|01>: real part", ksv.re[2], 0.0, 1e-12)
    _ = log.close("K(-pi/2)|10> -> i|01>: imaginary part", ksv.im[2], 1.0, 1e-12)

    # ---------------------------------------------------- Slater state, one electron
    var lam = 3.0
    var mean = Float64(n - 1) / 2.0
    var std = 0.5
    var h0 = single_electron_h(n, J, lam, mean, std)
    var q1 = orbitals(h0, m, 1)
    var c1 = Circuit(m, name="slater1")
    prepare_slater(c1, q1, m, 1)
    var sv1 = evolve(c1)
    var eig = jacobi_eigh(h0, m)
    var fid1 = fidelity_with(sv1, slater_amplitudes(q1, m, 1))
    print("one-electron fidelity with the exact ground state:", fid1)
    _ = log.close("one-electron Slater fidelity (notebook prints 1.0000000000)", fid1, 1.0, 1e-10)
    var res1 = 0.0
    for mu in range(m):
        var hv = 0.0
        for nu in range(m):
            hv += h0[mu * m + nu] * q1[nu]
        res1 = max(res1, abs(hv - eig.values[0] * q1[mu]))
    _ = log.close("orbital is an eigenvector of h", res1, 0.0, 1e-10)
    print("orbital energies (lowest four):", eig.values[0], eig.values[1], eig.values[2], eig.values[3])
    _ = log.check("non-degenerate gap above the two occupied orbitals (the Slater determinant is unique)", eig.values[2] - eig.values[1] > 1e-3)

    # ---------------------------------------------------- Slater state at quarter filling, initial densities
    var nelec = n // 2
    var q2 = orbitals(h0, m, nelec)
    var amp2 = slater_amplitudes(q2, m, nelec)
    var c2 = Circuit(m, name="slater2")
    prepare_slater(c2, q2, m, nelec)
    var sv2 = evolve(c2)
    var fid2 = fidelity_with(sv2, amp2)
    print("quarter-filling fidelity with the determinant state:", fid2)
    _ = log.close("two-electron Slater fidelity with det Q[:, S] amplitudes", fid2, 1.0, 1e-10)
    # N_up = N_dn = 1: h is block diagonal in spin, so each occupied orbital is spin pure; one must be up, one down.
    var w_up_total = 0.0
    var purity = 0.0
    for k in range(nelec):
        var w_up = 0.0
        for site in range(n):
            w_up += q2[k * m + qubit_idx(site, 0, n)] * q2[k * m + qubit_idx(site, 0, n)]
        w_up_total += w_up
        purity = max(purity, min(w_up, 1.0 - w_up))
    _ = log.close("occupied orbitals are spin pure", purity, 0.0, 1e-10)
    _ = log.close("N_up = N_dn = 1: spin-up weight of the occupied orbitals", w_up_total, 1.0, 1e-10)
    var dens0 = measure(sv2, obs)
    var ref0 = List[Float64](length=2 * n, fill=0.0)
    for site in range(n):
        var up = 0.0
        var dn = 0.0
        for k in range(nelec):
            up += q2[k * m + qubit_idx(site, 0, n)] * q2[k * m + qubit_idx(site, 0, n)]
            dn += q2[k * m + qubit_idx(site, 1, n)] * q2[k * m + qubit_idx(site, 1, n)]
        ref0[site] = up + dn
        ref0[n + site] = up - dn
    _ = log.close("initial charge and spin densities equal sum_k Q^2", max_abs_diff(dens0, ref0), 0.0, 1e-10)
    var peak = 0
    for site in range(n):
        if dens0[site] > dens0[peak]:
            peak = site
    _ = log.check("initial charge density peaks at the well centre (site 1 or 2)", peak == 1 or peak == 2)
    var total0 = 0.0
    for site in range(n):
        total0 += dens0[site]
    _ = log.close("initial electron number", total0, Float64(nelec), 1e-10)

    # ---------------------------------------------------- interacting quench: Trotter versus exact evolution
    var U = 3.0 * J
    var lat = Lattice(n)
    var swapped_ok = len(lat.swapped_even_sites) == 2 and lat.swapped_even_sites[0] == 2 and lat.swapped_even_sites[1] == 3
    _ = log.check("even sites sit at positions 2, 3 after the mode swap", swapped_ok)
    var exact = exact_trajectory(amp2, n, J, U, tau, num_iter)
    var err_main = 0.0
    var err_spread = 0.0
    var charge_spread = List[Float64]()
    var spin_spread = List[Float64]()
    var number_err = 0.0
    for it in range(num_iter):
        var svt = evolve(build_circuit(q2, lat, nelec, it, tau, J, U))
        var d = measure(svt, obs)
        err_main = max(err_main, max_abs_diff(d, exact[it]))
        var ks_c = spread(d, 0, n)
        var ks_s = spread(d, n, n)
        err_spread = max(err_spread, abs(ks_c - spread(exact[it], 0, n)))
        err_spread = max(err_spread, abs(ks_s - spread(exact[it], n, n)))
        charge_spread.append(ks_c)
        spin_spread.append(ks_s)
        var tot = 0.0
        for site in range(n):
            tot += d[site]
        number_err = max(number_err, abs(tot - Float64(nelec)))
    print("max density error of the Trotter run (tau = 0.1):", err_main)
    _ = log.close("electron number conserved at every step", number_err, 0.0, 1e-10)

    # Same final times with half and a quarter of the step: a first-order scheme has its error divided by 2 each time.
    var err_half = 0.0
    var err_quarter = 0.0
    for it in range(num_iter):
        var svh = evolve(build_circuit(q2, lat, nelec, 2 * it, tau / 2.0, J, U))
        err_half = max(err_half, max_abs_diff(measure(svh, obs), exact[it]))
        var svq = evolve(build_circuit(q2, lat, nelec, 4 * it, tau / 4.0, J, U))
        err_quarter = max(err_quarter, max_abs_diff(measure(svq, obs), exact[it]))
    print("max density error with tau/2:", err_half, " tau/4:", err_quarter)
    # The absolute bound is an accuracy statement chosen from observed convergence, not a rigorous Trotter bound
    # (the commutator bound is vacuous for U = 3); the order checks below are the guard against wrong angles or layers.
    _ = log.within("Trotter densities vs exact evolution, all steps (tau = 0.1)", err_main, 0.0, 0.005)
    _ = log.within("tau -> tau/2 cuts the Trotter error by about 2 (first order)", err_main / err_half, 1.7, 2.3)
    _ = log.within("tau/2 -> tau/4 cuts the Trotter error by about 2 (first order)", err_half / err_quarter, 1.7, 2.3)
    _ = log.within("Trotter spreads vs exact evolution", err_spread, 0.0, 0.005)
    var sep = abs(charge_spread[num_iter - 1] - spin_spread[num_iter - 1])
    var sep_exact = abs(spread(exact[num_iter - 1], 0, n) - spread(exact[num_iter - 1], n, n))
    print("charge/spin spread at the last step:", charge_spread[num_iter - 1], spin_spread[num_iter - 1])
    _ = log.close("charge-spin spread gap follows the exact gap", sep, sep_exact, 0.005)

    # The interaction matters: the exact U = 0 quench of the same state differs from U = 3 by far more than the Trotter error.
    var exact_free = exact_trajectory(amp2, n, J, 0.0, tau, num_iter)
    var u_effect = max_abs_diff(exact[num_iter - 1], exact_free[num_iter - 1])
    print("exact densities, U = 3 vs U = 0 at the last step differ by:", u_effect)
    _ = log.check("U = 3 changes the densities by more than 10 times the Trotter error", u_effect > 10.0 * err_main)

    # Sign of the interaction. The notebook's P(phi) = phase(phi n_up n_down) is exp(+i phi n n), the circuit of U = -3;
    # the file's cp(-phi) above is U = +3. Run the notebook's sign and compare with the exact U = -3 evolution.
    var exact_neg = exact_trajectory(amp2, n, J, -U, tau, num_iter)
    var err_neg = 0.0
    for it in range(num_iter):
        var svn = evolve(build_circuit(q2, lat, nelec, it, tau, J, -U))
        err_neg = max(err_neg, max_abs_diff(measure(svn, obs), exact_neg[it]))
    var err_neg_half = 0.0
    var err_neg_quarter = 0.0
    for it in range(num_iter):
        err_neg_half = max(err_neg_half, max_abs_diff(measure(evolve(build_circuit(q2, lat, nelec, 2 * it, tau / 2.0, J, -U)), obs), exact_neg[it]))
        err_neg_quarter = max(err_neg_quarter, max_abs_diff(measure(evolve(build_circuit(q2, lat, nelec, 4 * it, tau / 4.0, J, -U)), obs), exact_neg[it]))
    print("U = -3 errors tau, tau/2, tau/4:", err_neg, err_neg_half, err_neg_quarter)
    var sign_effect = 0.0
    for it in range(num_iter):
        sign_effect = max(sign_effect, max_abs_diff(exact_neg[it], exact[it]))
    print("notebook-sign circuit (U = -3) vs exact U = -3:", err_neg, " exact U = +3 vs U = -3 differ by:", sign_effect)
    # The attractive quench moves faster, so its first-order error is about 12 times larger at tau = 0.1: check the
    # convergence order and the accuracy at tau/4 (chosen from the observed convergence), not the tau = 0.1 value.
    _ = log.within("U = -3: tau -> tau/2 cuts the Trotter error by about 2", err_neg / err_neg_half, 1.7, 2.3)
    _ = log.within("U = -3: tau/2 -> tau/4 cuts the Trotter error by about 2", err_neg_half / err_neg_quarter, 1.7, 2.3)
    _ = log.within("notebook-sign circuit (exp(+i phi n n)) at tau/4 vs exact U = -3 evolution", err_neg_quarter, 0.0, 0.01)
    _ = log.check("the two signs differ by more than 10 times the U = -3 error at tau/4", sign_effect > 10.0 * err_neg_quarter)

    # ---------------------------------------------------- limit U = 0, one electron: standing waves
    var steps0 = 11
    var h_loc = List[Float64](length=m * m, fill=0.0)
    h_loc[qubit_idx(0, 0, n) * m + qubit_idx(0, 0, n)] = -1.0
    var q_loc = orbitals(h_loc, m, 1)
    var amp_loc = slater_amplitudes(q_loc, m, 1)
    var exact_loc = exact_trajectory(amp_loc, n, J, 0.0, tau, steps0)
    var err_ni = 0.0
    var err_ref = 0.0
    var err_ni_half = 0.0
    for it in range(steps0):
        var t = Float64(it) * tau
        var svt = evolve(build_circuit(q_loc, lat, 1, it, tau, J, 0.0))
        var svh = evolve(build_circuit(q_loc, lat, 1, 2 * it, tau / 2.0, J, 0.0))
        for site in range(n):
            # <j|psi(t)> = sum_k phi_k(j) phi_k(0) e^{-i E_k t}, phi_k(j) = sqrt(2/(N+1)) sin(pi k (j+1)/(N+1))
            var ar = 0.0
            var ai = 0.0
            for k in range(1, n + 1):
                var phi_j = sqrt(2.0 / Float64(n + 1)) * sin(PI * Float64(k * (site + 1)) / Float64(n + 1))
                var phi_0 = sqrt(2.0 / Float64(n + 1)) * sin(PI * Float64(k) / Float64(n + 1))
                var ek = -2.0 * J * cos(PI * Float64(k) / Float64(n + 1))
                ar += phi_j * phi_0 * cos(ek * t)
                ai += -phi_j * phi_0 * sin(ek * t)
            var analytic = ar * ar + ai * ai
            var got = svt.expectation(obs[site])  # charge density = spin-up occupation here
            var got_half = svh.expectation(obs[site])
            err_ni = max(err_ni, abs(got - analytic))
            err_ni_half = max(err_ni_half, abs(got_half - analytic))
            err_ref = max(err_ref, abs(exact_loc[it][site] - analytic))
    print("U = 0 Trotter error vs analytic:", err_ni, " with tau/2:", err_ni_half)
    _ = log.close("exact Fock reference reproduces the standing-wave formula (validates the reference)", err_ref, 0.0, 1e-10)
    # Accuracy bound chosen from observed convergence (the error is first order in tau); the ratio is the guard.
    _ = log.within("U = 0 Trotter occupations vs analytic standing waves (tau = 0.1)", err_ni, 0.0, 0.035)
    _ = log.within("U = 0: halving tau cuts the error by about 2 (first order)", err_ni / err_ni_half, 1.7, 2.3)

    # ---------------------------------------------------- limit J = 0: densities frozen, Trotter exact
    var lam_vh = 10.0
    var std_vh = Float64(n) / 6.0
    var h_vh = single_electron_h(n, 1.0, lam_vh, Float64(n - 1) / 2.0, std_vh)
    var q_vh = orbitals(h_vh, m, n)
    var amp_vh = slater_amplitudes(q_vh, m, n)
    var sv_vh0 = evolve(build_circuit(q_vh, lat, n, 0, 0.3, 0.0, 2.0))
    _ = log.close("half-filling Slater fidelity with det amplitudes", fidelity_with(sv_vh0, amp_vh), 1.0, 1e-10)
    var ref_vh = List[Float64](length=2 * n, fill=0.0)
    for site in range(n):
        var up = 0.0
        var dn = 0.0
        for k in range(n):
            up += q_vh[k * m + qubit_idx(site, 0, n)] * q_vh[k * m + qubit_idx(site, 0, n)]
            dn += q_vh[k * m + qubit_idx(site, 1, n)] * q_vh[k * m + qubit_idx(site, 1, n)]
        ref_vh[site] = up + dn
        ref_vh[n + site] = up - dn
    var frozen = 0.0
    var init_err = max_abs_diff(measure(sv_vh0, obs), ref_vh)
    for it in range(11):
        var svt = evolve(build_circuit(q_vh, lat, n, it, 0.3, 0.0, 2.0))
        frozen = max(frozen, max_abs_diff(measure(svt, obs), ref_vh))
    print("J = 0 max density deviation from frozen values:", frozen)
    _ = log.close("J = 0 initial densities equal sum_k Q^2", init_err, 0.0, 1e-10)
    _ = log.close("J = 0 densities frozen for 10 Trotter steps of 0.3 (notebook: 4-5e-2 from shot noise)", frozen, 0.0, 1e-10)

    # Densities are blind to diagonal phases, so also check the full state: at J = 0 the exact evolution multiplies the
    # amplitude of Fock state s by exp(-i U t D(s)), D(s) = number of doubly occupied sites. This sees both interaction layers.
    var steps_j0 = 10
    var u_j0 = 2.0
    var t_j0 = Float64(steps_j0) * 0.3
    var svj = evolve(build_circuit(q_vh, lat, n, steps_j0, 0.3, 0.0, u_j0))
    var ov_re = 0.0
    var ov_im = 0.0
    for s in range(len(amp_vh)):
        var dbl = 0
        for site in range(n):
            if ((s >> qubit_idx(site, 0, n)) & 1) == 1 and ((s >> qubit_idx(site, 1, n)) & 1) == 1:
                dbl += 1
        var ph = -u_j0 * t_j0 * Float64(dbl)
        # <ref|psi> with ref_s = amp_s e^{i ph}: conj(ref) * psi
        var cr = amp_vh[s] * cos(ph)
        var ci = -amp_vh[s] * sin(ph)
        ov_re += cr * svj.re[s] - ci * svj.im[s]
        ov_im += cr * svj.im[s] + ci * svj.re[s]
    _ = log.close("J = 0: full state equals the Slater state times exp(-i U t D(s)) (fidelity)", ov_re * ov_re + ov_im * ov_im, 1.0, 1e-10)

    log.finish()
