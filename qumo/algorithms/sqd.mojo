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

from std.collections import Dict
from std.math import sqrt, ceil
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


# ---------------------------------------------------------------------------
# Sampling side of SQD: subsampling, configuration recovery, Trim SQD
# ---------------------------------------------------------------------------
def partition_subsample(configs: List[Int], samples_per_batch: Int, num_batches: Int, mut rng: Rng) raises -> List[List[Int]]:
    """`num_batches` disjoint batches from one shuffled pool: no entry of `configs`
    lands in two batches (qiskit-addon-sqd's `partition_subsample`). With fewer
    configurations than requested the batches come out smaller, none empty."""
    if samples_per_batch < 1 or num_batches < 1:
        raise Error("QM-E802: partition_subsample needs samples_per_batch >= 1 and num_batches >= 1. Hint: got " + String(samples_per_batch) + ", " + String(num_batches) + ".")
    if len(configs) < num_batches:
        raise Error("QM-E802: " + String(len(configs)) + " configurations cannot fill " + String(num_batches) + " disjoint batches. Hint: sample more, or use fewer batches.")
    var pool = configs.copy()
    var i = len(pool) - 1
    while i > 0:
        var j = rng.below(i + 1)
        var t = pool[i]
        pool[i] = pool[j]
        pool[j] = t
        i -= 1
    var out = List[List[Int]]()
    var per = min(samples_per_batch, len(pool) // num_batches)
    for b in range(num_batches):
        var batch = List[Int](capacity=per)
        for k in range(per):
            batch.append(pool[b * per + k])
        out.append(batch^)
    return out^


def _relu_weight(y: Float64, filling: Float64, delta: Float64 = 0.01) -> Float64:
    """Modified ReLU of Robledo-Moreno et al. (2025), supplementary S-CR."""
    if y <= filling:
        return delta * y / max(filling, 1e-12)
    return delta + (1.0 - delta) * (y - filling) / max(1.0 - filling, 1e-12)


def recover_half(var bits: Int, n_target: Int, occ: List[Float64], offset: Int, num_orbitals: Int, mut rng: Rng) -> Int:
    """One spin half of a configuration with n_target electrons: flip orbitals with
    probabilities from the occupations occ[offset + i] of the current estimate."""
    var k = Int(pop_count(bits)) - n_target
    if k == 0:
        return bits
    var filling = Float64(n_target) / Float64(num_orbitals)
    var cand = List[Int]()
    var w = List[Float64]()
    for i in range(num_orbitals):
        var set = (bits >> i) & 1 == 1
        if (k > 0) == set:
            var o = occ[offset + i]
            cand.append(i)
            w.append(1e-12 + _relu_weight(abs(1.0 - o) if k > 0 else abs(o), filling))
    for _ in range(abs(k)):
        var total = 0.0
        for x in w:
            total += x
        var u = rng.uniform() * total
        var pick = len(cand) - 1
        for i in range(len(cand)):
            u -= w[i]
            if u < 0.0:
                pick = i
                break
        bits ^= 1 << cand[pick]
        w[pick] = 0.0
    return bits


def recover_configurations(configs: List[Int], num_orbitals: Int, n_alpha: Int, n_beta: Int,
                           occupations: List[Float64], mut rng: Rng) raises -> List[Int]:
    """Configuration recovery: push noisy bitstrings back into the (n_alpha, n_beta)
    sector. `occupations` per qubit, alpha orbitals first (`SubspaceGroundState.occupations`)."""
    if len(occupations) != 2 * num_orbitals:
        raise Error("QM-E802: recover_configurations needs " + String(2 * num_orbitals) + " occupations, got " + String(len(occupations)) + ". Hint: one per spin orbital.")
    var mask = (1 << num_orbitals) - 1
    var out = List[Int](capacity=len(configs))
    for x in configs:
        var a = recover_half(x & mask, n_alpha, occupations, 0, num_orbitals, rng)
        var b = recover_half(x >> num_orbitals, n_beta, occupations, num_orbitals, num_orbitals, rng)
        out.append(a | (b << num_orbitals))
    return out^


def _unique_sorted(xs: List[Int]) -> List[Int]:
    var s = xs.copy()
    sort(s)
    var out = List[Int]()
    for i in range(len(s)):
        if i == 0 or s[i] != s[i - 1]:
            out.append(s[i])
    return out^


def _strings_of(configs: List[Int], num_orbitals: Int, mut alpha: List[Int], mut beta: List[Int]):
    var mask = (1 << num_orbitals) - 1
    for x in configs:
        alpha.append(x & mask)
        beta.append(x >> num_orbitals)


@fieldwise_init
struct _Ranked(ImplicitlyCopyable):
    var weight: Float64
    var string: Int


def _top_strings(strings: List[Int], res: SubspaceGroundState, basis: List[Int], num_orbitals: Int,
                 upper: Bool, ratio: Float64, threshold: Float64, max_strings: Int) -> List[Int]:
    """CI strings of one spin sector by marginal weight: sum |c|^2 over the other sector."""
    var index = Dict[Int, Int]()
    var weight = List[Float64](length=len(strings), fill=0.0)
    for i in range(len(strings)):
        index[strings[i]] = i
    var mask = (1 << num_orbitals) - 1
    for k in range(len(basis)):
        var s = basis[k] >> num_orbitals if upper else basis[k] & mask
        var at = index.get(s, -1)
        if at >= 0:
            weight[at] += res.re[k] * res.re[k] + res.im[k] * res.im[k]
    var order = List[Int]()
    for i in range(len(strings)):
        if weight[i] >= threshold * threshold:
            order.append(i)

    def heavier(a: Int, b: Int) {imm weight} -> Bool:
        return weight[a] > weight[b]

    sort(Span(order), heavier)
    var keep = Int(ceil(ratio * Float64(len(order))))
    if max_strings > 0:
        keep = min(keep, max_strings)
    keep = max(min(keep, len(order)), 1)
    var out = List[Int](capacity=keep)
    for i in range(min(keep, len(order))):
        out.append(strings[order[i]])
    return out^


struct SqdIteration(Movable, Writable):
    var energy: Float64
    var occupations: List[Float64]
    var alpha: List[Int]  # strings of the best subspace (standard) or the carryover (trim)
    var beta: List[Int]
    var dimension: Int  # subspace size of the reported energy
    var diagonalizations: Int

    def __init__(out self):
        self.energy = 1e300
        self.occupations = List[Float64]()
        self.alpha = List[Int]()
        self.beta = List[Int]()
        self.dimension = 0
        self.diagonalizations = 0

    def write_to(self, mut w: Some[Writer]):
        w.write("SqdIteration(energy=", self.energy, ", dimension=", self.dimension, ", diagonalizations=", self.diagonalizations, ")")


def _solve_strings(h: PauliSum, alpha: List[Int], beta: List[Int], num_orbitals: Int, mut basis: List[Int]) raises -> SubspaceGroundState:
    basis = product_basis(alpha, beta, num_orbitals)
    return subspace_ground_state(h, basis, 2 * num_orbitals)


def sqd_iteration(h: PauliSum, configs: List[Int], num_orbitals: Int, samples_per_batch: Int,
                  num_batches: Int, seed: UInt64 = 1, policy: String = "standard", trim_ratio: Float64 = 0.1,
                  max_strings_per_trim: Int = -1, carryover_threshold: Float64 = 1e-4,
                  carryover_ratio: Float64 = 0.1, max_carryover: Int = -1,
                  carryover_alpha: List[Int] = List[Int](), carryover_beta: List[Int] = List[Int]()) raises -> SqdIteration:
    """One SQD iteration over bitstrings in the right sector (alpha low, beta high).

    policy "standard": diagonalise each batch, keep the lowest energy; carries the
    best subspace's strings. policy "trim" (Trim SQD, qiskit-addon-sqd 0.14): the
    batches are disjoint and only *screen*: each is diagonalised, its heaviest strings
    (trim_ratio of each sector, at most max_strings_per_trim) go into one merged
    subspace, whose energy is the result, and its own heaviest strings seed the next
    iteration through carryover_alpha / carryover_beta. An iteration costs
    num_batches + 1 diagonalisations, each of about the size of the merged one."""
    if policy != "standard" and policy != "trim":
        raise Error("QM-E816: unknown SQD policy '" + policy + "'. Hint: use 'standard' or 'trim'.")
    if not (trim_ratio > 0.0 and trim_ratio <= 1.0) or not (carryover_ratio > 0.0 and carryover_ratio <= 1.0):
        raise Error("QM-E816: trim_ratio and carryover_ratio must lie in (0, 1]. Hint: 0.1 keeps a tenth of the strings.")
    var rng = Rng(seed)
    var batches = partition_subsample(configs, samples_per_batch, num_batches, rng)
    var out = SqdIteration()
    var merged_a = List[Int]()
    var merged_b = List[Int]()
    for batch in batches:
        var a = carryover_alpha.copy()
        var b = carryover_beta.copy()
        _strings_of(batch, num_orbitals, a, b)
        var alpha = _unique_sorted(a)
        var beta = _unique_sorted(b)
        var basis = List[Int]()
        var r = _solve_strings(h, alpha, beta, num_orbitals, basis)
        out.diagonalizations += 1
        if policy == "standard":
            if r.energy < out.energy:
                out.energy = r.energy
                out.occupations = r.occupations.copy()
                out.alpha = _top_strings(alpha, r, basis, num_orbitals, False, 1.0, carryover_threshold, -1)
                out.beta = _top_strings(beta, r, basis, num_orbitals, True, 1.0, carryover_threshold, -1)
                out.dimension = len(basis)
        else:
            merged_a.extend(_top_strings(alpha, r, basis, num_orbitals, False, trim_ratio, 0.0, max_strings_per_trim))
            merged_b.extend(_top_strings(beta, r, basis, num_orbitals, True, trim_ratio, 0.0, max_strings_per_trim))
    if policy == "trim":
        var alpha = _unique_sorted(merged_a)
        var beta = _unique_sorted(merged_b)
        var basis = List[Int]()
        var r = _solve_strings(h, alpha, beta, num_orbitals, basis)
        out.diagonalizations += 1
        out.energy = r.energy
        out.occupations = r.occupations.copy()
        out.dimension = len(basis)
        out.alpha = _top_strings(alpha, r, basis, num_orbitals, False, carryover_ratio, carryover_threshold, max_carryover)
        out.beta = _top_strings(beta, r, basis, num_orbitals, True, carryover_ratio, carryover_threshold, max_carryover)
    return out^
