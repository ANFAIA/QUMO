"""Subspace diagonalisation for sample-based quantum diagonalisation (SQD / SKQD).

The classical "HPC" half of SQD (Robledo-Moreno et al., Sci. Adv. 2025, the
IBM + RIKEN Fugaku workflow; `qiskit-addon-sqd`): project a qubit Hamiltonian
onto the span of sampled computational-basis states and find its lowest
eigenpair. `qumo.algorithms.hamiltonians.subspace_ground_energy` does this for
<= 4096 states and returns the energy only; this module scales further and
returns the eigenvector, which configuration recovery needs (orbital
occupations of the current ground-state estimate):

* the projected matrix is stored row-wise in ELL form, one slot per distinct
  X mask of the Hamiltonian (all diagonal terms share one slot), built in
  parallel over rows with O(m * groups * log m) lookups - no 2^n work;
* thick-restart-free restarted Lanczos: blocks of `krylov` steps with full
  re-orthogonalisation, restarted from the Ritz vector until the residual
  ||H x - theta x|| <= tol * max(1, |theta|); memory = krylov + 3 vectors of
  the subspace size, matrix-vector products parallel over rows;
* `occupations` = <n_q> = sum_k |c_k|^2 bit_q(b_k) of the Ritz vector.

Variational: the result is >= the exact ground energy and equals it when the
basis spans the ground state's symmetry sector. Limits: 1 <= num_qubits <= 48,
1 <= m <= 2^21 states (QM-E807 above; QM-E802 for duplicates or states
outside [0, 2^n)).
"""

from std.math import sqrt
from std.bit import pop_count

from qumo.ir.pauli import PauliSum
from qumo.sim.rng import Rng
from qumo.sim.statevector import run_chunks
from qumo.algorithms.hamiltonians import check_fits
from qumo.algorithms.linalg import jacobi_eigh

comptime SQD_MAX_STATES = 1 << 21
comptime _IDX_BITS = 21


struct SubspaceGroundState(Movable, Writable):
    """Lowest eigenpair of H projected onto a basis-state subspace."""

    var energy: Float64
    var re: List[Float64]  # Ritz vector, coefficient k belongs to basis_states[k]
    var im: List[Float64]
    var occupations: List[Float64]  # <n_q> per qubit
    var dimension: Int
    var nonzeros: Int  # stored matrix entries (ELL slots in use)
    var matvecs: Int
    var restarts: Int
    var residual: Float64
    var converged: Bool

    def __init__(out self, m: Int, n: Int):
        self.energy = 0.0
        self.re = List[Float64](length=m, fill=0.0)
        self.im = List[Float64](length=m, fill=0.0)
        self.occupations = List[Float64](length=n, fill=0.0)
        self.dimension = m
        self.nonzeros = 0
        self.matvecs = 0
        self.restarts = 0
        self.residual = 0.0
        self.converged = False

    def write_to(self, mut w: Some[Writer]):
        w.write("SubspaceGroundState(energy=", self.energy, ", dim=", self.dimension, ", matvecs=", self.matvecs,
                ", residual=", self.residual, ", converged=", self.converged, ")")


struct _Ell(Movable):
    """Projected Hamiltonian, row-wise: slot (i, g) = <b_i| sum_{t in group g} P_t |b_col>."""

    var m: Int
    var width: Int
    var col: List[Int]
    var vre: List[Float64]
    var vim: List[Float64]

    def __init__(out self, h: PauliSum, states: List[Int], num_qubits: Int) raises:
        var m = len(states)
        self.m = m
        # sorted lookup table: key * 2^21 + caller index
        var packed = List[Int](capacity=m)
        var lim = 1 << num_qubits
        for i in range(m):
            var s = states[i]
            if s < 0 or s >= lim:
                raise Error("QM-E802: basis state " + String(s) + " is outside [0, 2^" + String(num_qubits) + ").")
            packed.append((s << _IDX_BITS) | i)
        sort(packed)
        var keys = List[Int](capacity=m)
        var pos = List[Int](capacity=m)
        for k in range(m):
            var s = packed[k] >> _IDX_BITS
            if k > 0 and s == keys[k - 1]:
                raise Error("QM-E802: basis state " + String(s) + " is listed twice. Hint: de-duplicate the sampled bitstrings first.")
            keys.append(s)
            pos.append(packed[k] & ((1 << _IDX_BITS) - 1))
        _ = packed^
        # group terms by X mask (diagonal terms -> one slot)
        var gx = List[Int]()
        var gstart = List[Int]()
        var tx = List[Int]()
        var tz = List[Int]()
        var tc = List[Float64]()
        var order = List[Int]()
        for t in range(len(h.terms)):
            order.append(t)
        # stable grouping by x without sorting PauliTerm structs
        var seen = List[Int]()
        for t in range(len(h.terms)):
            var x = Int(h.terms[t].x)
            var found = False
            for v in seen:
                if v == x:
                    found = True
                    break
            if not found:
                seen.append(x)
        for x in seen:
            gx.append(x)
            gstart.append(len(tx))
            for t in range(len(h.terms)):
                if Int(h.terms[t].x) == x:
                    tx.append(x)
                    tz.append(Int(h.terms[t].z))
                    tc.append(h.terms[t].coeff)
        gstart.append(len(tx))
        var wdt = len(gx)
        self.width = wdt
        self.col = List[Int](length=m * wdt, fill=-1)
        self.vre = List[Float64](length=m * wdt, fill=0.0)
        self.vim = List[Float64](length=m * wdt, fill=0.0)
        var cp = self.col.unsafe_ptr().as_unsafe_any_origin()
        var rp = self.vre.unsafe_ptr().as_unsafe_any_origin()
        var ip = self.vim.unsafe_ptr().as_unsafe_any_origin()
        var sp = states.unsafe_ptr().as_unsafe_any_origin()
        var kp = keys.unsafe_ptr().as_unsafe_any_origin()
        var pp = pos.unsafe_ptr().as_unsafe_any_origin()
        var gxp = gx.unsafe_ptr().as_unsafe_any_origin()
        var gsp = gstart.unsafe_ptr().as_unsafe_any_origin()
        var tzp = tz.unsafe_ptr().as_unsafe_any_origin()
        var tcp = tc.unsafe_ptr().as_unsafe_any_origin()
        var txp = tx.unsafe_ptr().as_unsafe_any_origin()

        def body(lo: Int, hi: Int) {imm cp, imm rp, imm ip, imm sp, imm kp, imm pp, imm gxp, imm gsp, imm tzp,
                                    imm tcp, imm txp, imm wdt, imm m}:
            for i in range(lo, hi):
                var bi = sp[unsafe_offset=i]
                for g in range(wdt):
                    var bj = bi ^ gxp[unsafe_offset=g]
                    # binary search bj
                    var a = 0
                    var b = m - 1
                    var j = -1
                    while a <= b:
                        var mid = (a + b) // 2
                        var v = kp[unsafe_offset=mid]
                        if v == bj:
                            j = pp[unsafe_offset=mid]
                            break
                        if v < bj:
                            a = mid + 1
                        else:
                            b = mid - 1
                    if j < 0:
                        continue
                    # <b_i| P |b_j> = coeff (-1)^{|b_j & z|} i^{ny}
                    var sr = 0.0
                    var si = 0.0
                    for t in range(gsp[unsafe_offset=g], gsp[unsafe_offset=g + 1]):
                        var z = tzp[unsafe_offset=t]
                        var c = tcp[unsafe_offset=t]
                        if (pop_count(bj & z) & 1) == 1:
                            c = -c
                        var ny = Int(pop_count(txp[unsafe_offset=t] & z)) & 3
                        if ny == 0:
                            sr += c
                        elif ny == 1:
                            si += c
                        elif ny == 2:
                            sr -= c
                        else:
                            si -= c
                    if sr == 0.0 and si == 0.0:
                        continue
                    cp[unsafe_offset=i * wdt + g] = j
                    rp[unsafe_offset=i * wdt + g] = sr
                    ip[unsafe_offset=i * wdt + g] = si

        run_chunks(body, m)
        _ = len(states)
        _ = len(keys)
        _ = len(pos)
        _ = len(gx)
        _ = len(gstart)
        _ = len(tz)
        _ = len(tc)
        _ = len(tx)

    def nonzeros(self) -> Int:
        var c = 0
        for j in self.col:
            if j >= 0:
                c += 1
        return c

    def apply(self, xr: List[Float64], xi: List[Float64], mut yr: List[Float64], mut yi: List[Float64]):
        """y <- H x (rows in parallel)."""
        var cp = self.col.unsafe_ptr().as_unsafe_any_origin()
        var rp = self.vre.unsafe_ptr().as_unsafe_any_origin()
        var ip = self.vim.unsafe_ptr().as_unsafe_any_origin()
        var xrp = xr.unsafe_ptr().as_unsafe_any_origin()
        var xip = xi.unsafe_ptr().as_unsafe_any_origin()
        var yrp = yr.unsafe_ptr().as_unsafe_any_origin()
        var yip = yi.unsafe_ptr().as_unsafe_any_origin()
        var wdt = self.width

        def body(lo: Int, hi: Int) {imm cp, imm rp, imm ip, imm xrp, imm xip, imm yrp, imm yip, imm wdt}:
            for i in range(lo, hi):
                var ar = 0.0
                var ai = 0.0
                for g in range(wdt):
                    var j = cp[unsafe_offset=i * wdt + g]
                    if j < 0:
                        continue
                    var hr = rp[unsafe_offset=i * wdt + g]
                    var hi_ = ip[unsafe_offset=i * wdt + g]
                    var vr = xrp[unsafe_offset=j]
                    var vi = xip[unsafe_offset=j]
                    ar += hr * vr - hi_ * vi
                    ai += hr * vi + hi_ * vr
                yrp[unsafe_offset=i] = ar
                yip[unsafe_offset=i] = ai

        run_chunks(body, self.m)
        _ = len(self.col)
        _ = len(xr)
        _ = len(xi)
        _ = len(yr)
        _ = len(yi)


def _dot(ar: List[Float64], ai: List[Float64], br: List[Float64], bi: List[Float64]) -> Tuple[Float64, Float64]:
    """<a|b> (conjugate-linear in a)."""
    var sr = 0.0
    var si = 0.0
    for k in range(len(ar)):
        sr += ar[k] * br[k] + ai[k] * bi[k]
        si += ar[k] * bi[k] - ai[k] * br[k]
    return (sr, si)


def _norm(ar: List[Float64], ai: List[Float64]) -> Float64:
    var s = 0.0
    for k in range(len(ar)):
        s += ar[k] * ar[k] + ai[k] * ai[k]
    return sqrt(s)


def subspace_ground_state(h: PauliSum, basis_states: List[Int], num_qubits: Int, tol: Float64 = 1e-9,
                          krylov: Int = 40, max_restarts: Int = 60, seed: UInt64 = 2024) raises -> SubspaceGroundState:
    """Lowest eigenpair of H restricted to span{|b> : b in basis_states}
    (see the module docstring). Deterministic for a seed."""
    check_fits(h, num_qubits)
    var m = len(basis_states)
    if m == 0:
        raise Error("QM-E802: subspace_ground_state() needs at least one basis state.")
    if num_qubits < 1 or num_qubits > 48:
        raise Error("QM-E802: subspace_ground_state() needs 1 <= num_qubits <= 48, got " + String(num_qubits) + ".")
    if m > SQD_MAX_STATES:
        raise Error("QM-E807: subspace_ground_state() is limited to " + String(SQD_MAX_STATES) + " basis states, got " + String(m) + ". Hint: subsample the configurations (samples_per_batch) or restrict to a symmetry sector.")
    if krylov < 2 or max_restarts < 1 or not (tol > 0.0):
        raise Error("QM-E801: subspace_ground_state() needs krylov >= 2, max_restarts >= 1 and tol > 0.")
    var A = _Ell(h, basis_states, num_qubits)
    var res = SubspaceGroundState(m, num_qubits)
    res.nonzeros = A.nonzeros()
    var rng = Rng(seed)
    var xr = List[Float64](length=m, fill=0.0)
    var xi = List[Float64](length=m, fill=0.0)
    for k in range(m):
        xr[k] = rng.uniform() - 0.5
        xi[k] = rng.uniform() - 0.5
    var nrm = _norm(xr, xi)
    for k in range(m):
        xr[k] /= nrm
        xi[k] /= nrm
    var wr = List[Float64](length=m, fill=0.0)
    var wi = List[Float64](length=m, fill=0.0)
    var K = min(krylov, m)
    var theta = 0.0
    for restart in range(max_restarts):
        res.restarts = restart + 1
        var Vr = List[List[Float64]]()
        var Vi = List[List[Float64]]()
        Vr.append(xr.copy())
        Vi.append(xi.copy())
        var alpha = List[Float64]()
        var beta = List[Float64]()
        for k in range(K):
            A.apply(Vr[k], Vi[k], wr, wi)
            res.matvecs += 1
            var a = _dot(Vr[k], Vi[k], wr, wi)[0]
            alpha.append(a)
            for t in range(m):
                wr[t] -= a * Vr[k][t]
                wi[t] -= a * Vi[k][t]
                if k > 0:
                    wr[t] -= beta[k - 1] * Vr[k - 1][t]
                    wi[t] -= beta[k - 1] * Vi[k - 1][t]
            for _ in range(2):  # full re-orthogonalisation
                for j in range(k + 1):
                    var c = _dot(Vr[j], Vi[j], wr, wi)
                    for t in range(m):
                        var ur = Vr[j][t]
                        var ui = Vi[j][t]
                        wr[t] -= c[0] * ur - c[1] * ui
                        wi[t] -= c[0] * ui + c[1] * ur
            var b = _norm(wr, wi)
            if k == K - 1 or b < 1e-12:
                break
            beta.append(b)
            var nr = List[Float64](length=m, fill=0.0)
            var ni = List[Float64](length=m, fill=0.0)
            for t in range(m):
                nr[t] = wr[t] / b
                ni[t] = wi[t] / b
            Vr.append(nr^)
            Vi.append(ni^)
        var kk = len(alpha)
        var T = List[Float64](length=kk * kk, fill=0.0)
        for j in range(kk):
            T[j * kk + j] = alpha[j]
            if j + 1 < kk:
                T[j * kk + j + 1] = beta[j]
                T[(j + 1) * kk + j] = beta[j]
        var eig = jacobi_eigh(T, kk)
        for t in range(m):
            xr[t] = 0.0
            xi[t] = 0.0
        for j in range(kk):
            var y = eig.vectors[j * kk + 0]
            for t in range(m):
                xr[t] += y * Vr[j][t]
                xi[t] += y * Vi[j][t]
        nrm = _norm(xr, xi)
        for t in range(m):
            xr[t] /= nrm
            xi[t] /= nrm
        A.apply(xr, xi, wr, wi)
        res.matvecs += 1
        var rq = _dot(xr, xi, wr, wi)[0]
        theta = rq
        var r2 = 0.0
        for t in range(m):
            var dr = wr[t] - rq * xr[t]
            var di = wi[t] - rq * xi[t]
            r2 += dr * dr + di * di
        res.residual = sqrt(r2)
        if res.residual <= tol * max(1.0, abs(theta)) or kk == m:
            res.converged = res.residual <= max(tol * max(1.0, abs(theta)), 1e-8)
            break
    res.energy = theta
    for t in range(m):
        res.re[t] = xr[t]
        res.im[t] = xi[t]
        var p = xr[t] * xr[t] + xi[t] * xi[t]
        var b = basis_states[t]
        for q in range(num_qubits):
            if (b >> q) & 1 == 1:
                res.occupations[q] += p
    return res^


def product_basis(alpha_strings: List[Int], beta_strings: List[Int], num_orbitals: Int) raises -> List[Int]:
    """Determinant basis {a + (b << norb)} for every alpha string a and beta
    string b (qiskit-addon-sqd's CI-string product). Inputs must be distinct
    and < 2^norb."""
    var out = List[Int](capacity=len(alpha_strings) * len(beta_strings))
    var lim = 1 << num_orbitals
    for b in beta_strings:
        if b < 0 or b >= lim:
            raise Error("QM-E802: beta string " + String(b) + " is outside [0, 2^" + String(num_orbitals) + ").")
        for a in alpha_strings:
            if a < 0 or a >= lim:
                raise Error("QM-E802: alpha string " + String(a) + " is outside [0, 2^" + String(num_orbitals) + ").")
            out.append(a | (b << num_orbitals))
    return out^
