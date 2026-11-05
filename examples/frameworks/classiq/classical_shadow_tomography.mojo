"""Classical shadow tomography of the Phi^- Bell state with Pauli and random-Clifford snapshots.

Reproduces: Classiq library - Classical Shadow Tomography
Source: https://github.com/Classiq/classiq-library/blob/main/algorithms/metrology/classical_shadow_tomography.ipynb (Classiq Library, MIT)

The notebook prepares the Phi^- Bell state and takes one snapshot per random measurement: a random X, Y or Z basis on each
qubit (Pauli ensemble), or a random depth-20 {H, S, CX} Clifford circuit (Clifford ensemble). Each outcome b is inverted through
the measurement channel, giving 3 U^dag |b><b| U - I per qubit, or (2^n + 1) U^dag |b><b| U - I for the Clifford ensemble, and
<Z x Z> is estimated by median of means. The checks: the Bell matrix; the QUMO outcome law of the nine Pauli basis pairs; the exact
ensemble average of the Pauli snapshots (it is the Bell matrix, <ZZ> = 1, E[X^2] = 9); the sample counts of Eq. (5) for accuracy
0.2 and failure probability 0.01 (81065 Pauli, 108086 Clifford, K = 10); and Monte Carlo runs of those sizes through QUMO's
sampler, whose median of means lies within 0.2 of <ZZ> = 1 and whose mean lies within five standard errors of the exact value
(1 for Pauli; 1.03697 for the notebook's depth-20 Clifford sampler, from an independent enumeration, see below; 1 for an extra
depth-60 run). The sampled Clifford unitaries are checked to be unitary, and every sampled snapshot to have trace one.

Deviations from the notebook:
- Classiq synthesis, cloud sampling and prepare_bell_state are replaced by QUMO circuits. Phi^- is H on qubit 0, CX(0, 1), then Z on
  qubit 0, which gives the notebook's printed density matrix.
- The basis changes are fixed gates, H for X and S then H for Y (the matrix H S of the notebook's reconstruction), not the
  parametric RZ and RY binding. The 2x2 unitaries are read back from QUMO's statevector.
- numpy's seed 552 and Classiq's random_seed are replaced by QUMO's Rng, so the notebook's printed sampled values (<ZZ> 0.8100
  from 400 Pauli snapshots, 1.0 from 50 Clifford snapshots, Frobenius distances 0.1997 and 0.5958) are not reproduced. The checks
  compare with analytic values and with the notebook's own accuracy target instead.
- The Clifford sampler is the notebook's: H or S on a uniform qubit, or CX on a uniformly ordered pair. It draws from Rng, and the
  unitary is taken from QUMO statevector columns instead of dense numpy matrices.
- The notebook's Clifford sampler is biased, and the notebook's claim that depth 20 is past the mixing time is wrong. Every depth-d
  walk over {H0, H1, S0, S1, CX(0,1), CX(1,0)} stays in one coset of the 11520-element two-qubit Clifford group (mod global
  phase) fixed by d mod 2, so it reaches only 5760 elements and never samples the whole group, at any depth. Exact enumeration of
  the walk (numpy, independent of QUMO, 2x2 and 4x4 matrices of the six gates; the script is not in the repository) gives the exact
  expectation of the <ZZ> estimator on the Bell state: 1.036973 at depth 20, 1.000649 at depth 40, 1.0000115 at depth 60, and
  1.0000000000 as the uniform average over the 5760 reachable elements. The depth-20 estimator therefore has bias 0.037 (small
  against the notebook's accuracy 0.2, so the notebook's accuracy conclusion survives). The Monte Carlo check at the Eq. (5)
  size runs the notebook's depth 20 and compares its sample mean with 1.036973 (five standard errors), and also checks the median
  of means against 1 to 0.2. An extra, separately labeled run at depth 60 checks the mean against 1.0000115. That depth only
  converges within the half-group the walk can reach; it is not the full Clifford group. The 50-snapshot run keeps depth 20.
- The Eq. (5) shadow-norm constants are computed from the observable as in the notebook's error_bound_pauli and
  error_bound_clifford (Pauli: (|c| 3^(weight/2))^2; Clifford: 3 tr(O0^2) with O0 = O - tr(O) I / 4), not hard-coded.
- Median of means uses K chunks of floor(N/K) snapshots (Eq. (1)). The notebook's slicing range(0, N, N // K) gives K + 1 chunks
  when K does not divide N, which happens at N = 81065.
- The notebook's 400 Pauli and 50 Clifford runs are repeated at their own sizes and printed. Their random values are not checked.

Run: .venv/bin/mojo run -I . examples/frameworks/classiq/classical_shadow_tomography.mojo
example-meta: {"framework": "classiq", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from std.math import ceil, log, sqrt
from qumo.ir.circuit import Circuit
from qumo.sim.rng import Rng
from qumo.sim.runner import evolve, evolve_into, sample
from qumo.sim.statevector import StateVector
from qumo.testing.checks import CheckLog


# ---- complex numbers; 2x2 and 4x4 matrices are row-major List[Cx] ----


@fieldwise_init
struct Cx(ImplicitlyCopyable):
    var re: Float64
    var im: Float64


def cadd(a: Cx, b: Cx) -> Cx:
    return Cx(a.re + b.re, a.im + b.im)


def cmul(a: Cx, b: Cx) -> Cx:
    return Cx(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)


def cconj(a: Cx) -> Cx:
    return Cx(a.re, -a.im)


def cscale(a: Cx, s: Float64) -> Cx:
    return Cx(a.re * s, a.im * s)


def cabs2(a: Cx) -> Float64:
    return a.re * a.re + a.im * a.im


def zeros(n: Int) -> List[Cx]:
    return List[Cx](length=n, fill=Cx(0.0, 0.0))


# ---- circuits: Bell state, basis changes, random Clifford gate sequences ----


def append_bell(mut c: Circuit):
    """Phi^- = (|00> - |11>)/sqrt(2) on qubits 0 and 1: H, CX(0, 1), then Z on qubit 0."""
    c.h(0)
    c.cx(0, 1)
    c.z(0)


def append_basis(mut c: Circuit, q: Int, basis: Int):
    """Turn a Z measurement into X (basis 0: H), Y (basis 1: S then H, the matrix H S) or Z (basis 2: nothing)."""
    if basis == 1:
        c.s(q)
    if basis != 2:
        c.h(q)


def sample_clifford(mut rng: Rng, depth: Int) -> List[Int]:
    """The notebook's sample_clifford_circuit: a step is H or S on a uniform qubit, or CX on a uniformly ordered pair.
    Stored as (kind, q1, q2) triples: kind 0 = H, 1 = S, 2 = CX(q1, q2)."""
    var seq = List[Int]()
    for _ in range(depth):
        var kind = rng.below(3)
        if kind < 2:
            seq.append(kind)
            seq.append(rng.below(2))
            seq.append(-1)
        else:
            var ctrl = rng.below(2)
            seq.append(2)
            seq.append(ctrl)
            seq.append(1 - ctrl)
    return seq^


def append_seq(mut c: Circuit, seq: List[Int]):
    var i = 0
    while i < len(seq):
        if seq[i] == 0:
            c.h(seq[i + 1])
        elif seq[i] == 1:
            c.s(seq[i + 1])
        else:
            c.cx(seq[i + 1], seq[i + 2])
        i += 3


def basis_unitary(basis: Int) raises -> List[Cx]:
    """2x2 unitary of a basis change, read back from QUMO: column j is the output for input |j>."""
    var c = Circuit(1, 0, name="basis")
    append_basis(c, 0, basis)
    var u = zeros(4)
    for j in range(2):
        var sv = StateVector.basis(1, j)
        evolve_into(sv, c, List[Float64]())
        for i in range(2):
            u[i * 2 + j] = Cx(sv.re[i], sv.im[i])
    return u^


def clifford_unitary(seq: List[Int]) raises -> List[Cx]:
    """4x4 unitary of a gate sequence, read back from QUMO: column j is U|j>, qubit 0 the low bit."""
    var c = Circuit(2, 0, name="clifford")
    append_seq(c, seq)
    var u = zeros(16)
    for j in range(4):
        var sv = StateVector.basis(2, j)
        evolve_into(sv, c, List[Float64]())
        for i in range(4):
            u[i * 4 + j] = Cx(sv.re[i], sv.im[i])
    return u^


# ---- snapshot estimators and reconstructions ----


def pauli_factor(u: List[Cx], b: Int) -> List[Cx]:
    """One qubit's factor 3 U^dag |b><b| U - I, with u the basis unitary (row b is <b|U)."""
    var a = zeros(4)
    for r in range(2):
        for c in range(2):
            var t = cscale(cmul(cconj(u[b * 2 + r]), u[b * 2 + c]), 3.0)
            if r == c:
                t = Cx(t.re - 1.0, t.im)
            a[r * 2 + c] = t
    return a^


def trace2(a: List[Cx]) -> Float64:
    return a[0].re + a[3].re


def add_pauli_snapshot(mut rho: List[Cx], a0: List[Cx], a1: List[Cx], w: Float64 = 1.0):
    """Add w * (A_0 (x) A_1) to rho, where qubit 0 is the low bit of the basis index, as in QUMO."""
    for r in range(4):
        for c in range(4):
            var t = cmul(a0[(r & 1) * 2 + (c & 1)], a1[((r >> 1) & 1) * 2 + ((c >> 1) & 1)])
            rho[r * 4 + c] = cadd(rho[r * 4 + c], cscale(t, w))


def zz_sign(j: Int) -> Float64:
    """Eigenvalue of Z x Z on basis state j."""
    var parity = (j & 1) ^ ((j >> 1) & 1)
    return 1.0 if parity == 0 else -1.0


def clifford_zz(u: List[Cx], b: Int) -> Float64:
    """Estimator tr(ZZ rho_hat) = 5 <b|U ZZ U^dag|b> - tr(ZZ) = 5 sum_j |U_bj|^2 z_j, because tr(ZZ) = 0."""
    var s = 0.0
    for j in range(4):
        s += cabs2(u[b * 4 + j]) * zz_sign(j)
    return 5.0 * s


def add_clifford_snapshot(mut rho: List[Cx], u: List[Cx], b: Int):
    """Add 5 U^dag |b><b| U - I to rho, with entries 5 conj(U_br) U_bc - delta_rc."""
    for r in range(4):
        for c in range(4):
            var t = cscale(cmul(cconj(u[b * 4 + r]), u[b * 4 + c]), 5.0)
            if r == c:
                t = Cx(t.re - 1.0, t.im)
            rho[r * 4 + c] = cadd(rho[r * 4 + c], t)


def zz_pauli(b0: Int, b1: Int, bit0: Int, bit1: Int) -> Float64:
    """Pauli-shadow term for Z x Z, as in the notebook's estimate_observable_pauli: zero unless both bases
    are Z, and then +3 or -3 per qubit depending on the outcome."""
    if b0 != 2 or b1 != 2:
        return 0.0
    var s0 = 3.0 if bit0 == 0 else -3.0
    var s1 = 3.0 if bit1 == 0 else -3.0
    return s0 * s1


# ---- statistics ----


def mean(xs: List[Float64]) -> Float64:
    var s = 0.0
    for x in xs:
        s += x
    return s / Float64(len(xs))


def sample_var(xs: List[Float64]) -> Float64:
    var m = mean(xs)
    var s = 0.0
    for x in xs:
        s += (x - m) * (x - m)
    return s / Float64(len(xs) - 1)


def median(xs: List[Float64]) -> Float64:
    var ys = xs.copy()
    sort(ys)
    var n = len(ys)
    if n % 2 == 1:
        return ys[n // 2]
    return 0.5 * (ys[n // 2 - 1] + ys[n // 2])


def median_of_means(xs: List[Float64], k: Int) -> Float64:
    """Eqs. (1)-(2): K chunks of floor(N/K) snapshots, then the median of the K chunk means."""
    var chunk = len(xs) // k
    var means = List[Float64]()
    for c in range(k):
        var s = 0.0
        for i in range(c * chunk, (c + 1) * chunk):
            s += xs[i]
        means.append(s / Float64(chunk))
    return median(means)


# ---- matrix helpers ----


def max_dev(a: List[Cx], b: List[Cx]) -> Float64:
    var d = 0.0
    for i in range(len(a)):
        d = max(d, sqrt(cabs2(Cx(a[i].re - b[i].re, a[i].im - b[i].im))))
    return d


def frob_dev(a: List[Cx], b: List[Cx]) -> Float64:
    var s = 0.0
    for i in range(len(a)):
        s += cabs2(Cx(a[i].re - b[i].re, a[i].im - b[i].im))
    return sqrt(s)


def trace4(a: List[Cx]) -> Float64:
    return a[0].re + a[5].re + a[10].re + a[15].re


def scale_all(a: List[Cx], s: Float64) -> List[Cx]:
    var out = zeros(len(a))
    for i in range(len(a)):
        out[i] = cscale(a[i], s)
    return out^


def notebook_bell() -> List[Cx]:
    """The notebook's printed Phi^- density matrix [[0.5, 0, 0, -0.5], [0, 0, 0, 0], [0, 0, 0, 0], [-0.5, 0, 0, 0.5]]."""
    var out = zeros(16)
    out[0] = Cx(0.5, 0.0)
    out[3] = Cx(-0.5, 0.0)
    out[12] = Cx(-0.5, 0.0)
    out[15] = Cx(0.5, 0.0)
    return out^


def bell_matrix() raises -> List[Cx]:
    """Density matrix of the QUMO-prepared Bell state."""
    var c = Circuit(2, 0, name="bell")
    append_bell(c)
    var sv = evolve(c)
    var rho = zeros(16)
    for r in range(4):
        for col in range(4):
            rho[r * 4 + col] = cmul(Cx(sv.re[r], sv.im[r]), cconj(Cx(sv.re[col], sv.im[col])))
    return rho^


# ---- Monte Carlo runs and the exact Pauli ensemble ----


@fieldwise_init
struct ShadowRun(Movable):
    var xs: List[Float64]  # Z x Z estimator of each snapshot
    var rho: List[Cx]  # sum of the snapshot reconstructions
    var trace_dev: Float64  # max |tr(rho_hat) - 1| over snapshots
    var unit_dev: Float64  # max |U^dag U - I| over the sampled Clifford unitaries (0 for Pauli)


def pauli_run(n: Int, seed: UInt64, bmats: List[List[Cx]]) raises -> ShadowRun:
    """Run N Pauli snapshots. Each draws a basis per qubit, then runs the rotated Bell state through QUMO's sampler for one shot."""
    var rng = Rng(seed)
    var xs = List[Float64](capacity=n)
    var rho = zeros(16)
    var td = 0.0
    for _ in range(n):
        var b0 = rng.below(3)
        var b1 = rng.below(3)
        var c = Circuit(2, 2, name="pauli_shadow")
        append_bell(c)
        append_basis(c, 0, b0)
        append_basis(c, 1, b1)
        c.measure(0, 0)
        c.measure(1, 1)
        var res = sample(c, 1, seed=UInt64(rng.next_u64()))
        var bit0 = res.memory[0] & 1
        var bit1 = (res.memory[0] >> 1) & 1
        xs.append(zz_pauli(b0, b1, bit0, bit1))
        var a0 = pauli_factor(bmats[b0], bit0)
        var a1 = pauli_factor(bmats[b1], bit1)
        add_pauli_snapshot(rho, a0, a1)
        td = max(td, abs(trace2(a0) * trace2(a1) - 1.0))
    return ShadowRun(xs^, rho^, td, 0.0)


def clifford_run(n: Int, seed: UInt64, depth: Int) raises -> ShadowRun:
    """Run N Clifford snapshots. Each draws its own random circuit of the given depth, applies it after the Bell state, and takes one QUMO shot (depth 20 in the notebook)."""
    var rng = Rng(seed)
    var xs = List[Float64](capacity=n)
    var rho = zeros(16)
    var td = 0.0
    var ud = 0.0
    for _ in range(n):
        var seq = sample_clifford(rng, depth)
        var u = clifford_unitary(seq)
        for j in range(4):
            for k in range(4):
                var s = Cx(0.0, 0.0)
                for i in range(4):
                    s = cadd(s, cmul(cconj(u[i * 4 + j]), u[i * 4 + k]))
                var target = 1.0 if j == k else 0.0
                ud = max(ud, max(abs(s.re - target), abs(s.im)))
        var c = Circuit(2, 2, name="clifford_shadow")
        append_bell(c)
        append_seq(c, seq)
        c.measure(0, 0)
        c.measure(1, 1)
        var b = sample(c, 1, seed=UInt64(rng.next_u64())).memory[0]
        xs.append(clifford_zz(u, b))
        add_clifford_snapshot(rho, u, b)
        var row = 0.0
        for j in range(4):
            row += cabs2(u[b * 4 + j])
        td = max(td, abs(5.0 * row - 4.0 - 1.0))
    return ShadowRun(xs^, rho^, td, ud)


@fieldwise_init
struct PauliExact(Movable):
    var rho: List[Cx]  # exact ensemble average of the Pauli snapshot reconstructions
    var ezz: Float64  # exact E[Z x Z estimator]
    var ex2: Float64  # exact E[(Z x Z estimator)^2]
    var law_dev: Float64  # max deviation of the QUMO outcome law from the analytic one


def analytic_law(b0: Int, b1: Int) -> List[Float64]:
    """Outcome law of Phi^- measured in bases (b0, b1). Correlators: <XX> = -1, <YY> = <ZZ> = +1, all others and the
    marginals zero. So matched X-X pairs give odd outcome parity, matched Y-Y and Z-Z pairs even parity, and
    mismatched pairs are uniform."""
    var p = List[Float64]()
    for i in range(4):
        var parity = (i & 1) ^ ((i >> 1) & 1)
        var v = 0.25
        if b0 == b1:
            var odd = b0 == 0
            v = 0.5 if (parity == 1) == odd else 0.0
        p.append(v)
    return p^


def pauli_exact(bmats: List[List[Cx]]) raises -> PauliExact:
    """Exact enumeration over the nine basis pairs (each with weight 1/9) and the outcomes of each, with QUMO's probabilities."""
    var rho = zeros(16)
    var ezz = 0.0
    var ex2 = 0.0
    var law_dev = 0.0
    for b0 in range(3):
        for b1 in range(3):
            var c = Circuit(2, 0, name="pauli_exact")
            append_bell(c)
            append_basis(c, 0, b0)
            append_basis(c, 1, b1)
            var probs = evolve(c).probabilities()
            var want = analytic_law(b0, b1)
            for i in range(4):
                law_dev = max(law_dev, abs(probs[i] - want[i]))
                var w = probs[i] / 9.0
                var bit0 = i & 1
                var bit1 = (i >> 1) & 1
                var a0 = pauli_factor(bmats[b0], bit0)
                var a1 = pauli_factor(bmats[b1], bit1)
                add_pauli_snapshot(rho, a0, a1, w)
                var x = zz_pauli(b0, b1, bit0, bit1)
                ezz += w * x
                ex2 += w * x * x
    return PauliExact(rho^, ezz, ex2, law_dev)


def main() raises:
    var chk = CheckLog("classiq/shadow")
    var bell_ref = notebook_bell()

    # 1. The QUMO Bell state is the notebook's printed Phi^- matrix.
    var rho_bell = bell_matrix()
    _ = chk.close("Phi- density matrix matches the notebook's printed matrix (max deviation)", max_dev(rho_bell, bell_ref), 0.0, 1e-12)

    # 2. Pauli ensemble, exact: QUMO's basis circuits, the average snapshot, and the ZZ moments.
    var bmats = List[List[Cx]]()
    for k in range(3):
        bmats.append(basis_unitary(k))
    var ex = pauli_exact(bmats)
    _ = chk.close("Pauli basis circuits give the analytic outcome law for all nine pairs (max deviation)", ex.law_dev, 0.0, 1e-12)
    _ = chk.close("exact average of the Pauli snapshots is the Bell matrix (max deviation)", max_dev(ex.rho, bell_ref), 0.0, 1e-12)
    _ = chk.close("exact average of the Pauli snapshots has trace one", trace4(ex.rho), 1.0, 1e-12)
    _ = chk.close("exact Pauli-ensemble <ZZ> equals tr(ZZ rho) = 1", ex.ezz, 1.0, 1e-12)
    _ = chk.close("exact E[(ZZ estimator)^2] for Pauli snapshots is 9", ex.ex2, 9.0, 1e-12)
    var var_p = ex.ex2 - ex.ezz * ex.ezz

    # 3. Sample sizes of Eq. (5): accuracy 0.2, failure probability 0.01, one observable ZZ (M = 1).
    var eps = 0.2
    var kdiv = 2.0 * log(2.0 / 0.01)
    var weight = 2  # ZZ has two non-identity Paulis, coefficient 1
    var pauli_norm2 = pow(1.0 * pow(3.0, Float64(weight) / 2.0), 2.0)
    var tr_o = 0.0  # tr(ZZ) and tr(ZZ^2) from the eigenvalues of Z x Z
    var tr_o2 = 0.0
    for j in range(4):
        tr_o += zz_sign(j)
        tr_o2 += zz_sign(j) * zz_sign(j)
    var clif_norm2 = 3.0 * (tr_o2 - tr_o * tr_o / 4.0)  # 3 tr(O0^2), O0 = O - tr(O) I / 4
    var n_pauli = Int(ceil(34.0 * pauli_norm2 / (eps * eps) * kdiv))
    var n_clif = Int(ceil(34.0 * clif_norm2 / (eps * eps) * kdiv))
    _ = chk.check("Eq. (5) Pauli sample count is the notebook's 81065", n_pauli == 81065, "N*K=" + String(n_pauli))
    _ = chk.check("Eq. (5) Clifford sample count is the notebook's 108086", n_clif == 108086, "N*K=" + String(n_clif))
    _ = chk.check("Eq. (5) divisions K = 10", Int(kdiv) == 10, "K=" + String(kdiv))

    # 4. The notebook's own sizes: 400 Pauli snapshots (div 2) and 50 Clifford snapshots (div 10), depth 20.
    var small_p = pauli_run(400, UInt64(552), bmats)
    var rho_p = scale_all(small_p.rho, 1.0 / 400.0)
    print("notebook size, Pauli 400 snapshots: <ZZ> =", median_of_means(small_p.xs, 2), " Frobenius distance to Phi- =", frob_dev(rho_p, bell_ref))
    var small_c = clifford_run(50, UInt64(553), 20)
    var rho_c = scale_all(small_c.rho, 1.0 / 50.0)
    print("notebook size, Clifford 50 snapshots: <ZZ> =", median_of_means(small_c.xs, 10), " Frobenius distance to Phi- =", frob_dev(rho_c, bell_ref))
    _ = chk.check("Pauli snapshots have trace one (400 snapshots)", small_p.trace_dev <= 1e-12, "max deviation=" + String(small_p.trace_dev))
    _ = chk.check("Clifford snapshots have trace one (50 snapshots)", small_c.trace_dev <= 1e-12, "max deviation=" + String(small_c.trace_dev))
    _ = chk.check("sampled Clifford unitaries are unitary (50 circuits)", small_c.unit_dev <= 1e-12, "max deviation=" + String(small_c.unit_dev))

    # 5. Monte Carlo at the Eq. (5) sizes, through QUMO's sampler: <ZZ> = 1 to the notebook's accuracy 0.2.
    var big_p = pauli_run(n_pauli, UInt64(554), bmats)
    var est_bp = median_of_means(big_p.xs, 10)
    var mean_bp = mean(big_p.xs)
    print("Pauli", len(big_p.xs), "snapshots: median of 10 means =", est_bp, " sample mean =", mean_bp)
    _ = chk.close("Pauli median of 10 means of <ZZ> at the Eq. (5) size is within 0.2 of 1", est_bp, 1.0, 0.2)
    _ = chk.close("Pauli sample mean of <ZZ> is within 5 standard errors of 1", mean_bp, 1.0, 5.0 * sqrt(var_p / Float64(len(big_p.xs))))
    _ = chk.check("Pauli snapshots have trace one at the Eq. (5) size", big_p.trace_dev <= 1e-12, "max deviation=" + String(big_p.trace_dev))

    # The notebook's depth 20 at the Eq. (5) size: biased sampler, exact expectation 1.0369728090 from the independent
    # numpy enumeration of the gate walk (see the deviation note).
    var nb_c = clifford_run(n_clif, UInt64(556), 20)
    var est_nb = median_of_means(nb_c.xs, 10)
    var mean_nb = mean(nb_c.xs)
    var se_nb = sqrt(sample_var(nb_c.xs) / Float64(len(nb_c.xs)))
    print("Clifford depth 20 (notebook),", len(nb_c.xs), "snapshots: median of 10 means =", est_nb, " sample mean =", mean_nb, " standard error =", se_nb)
    _ = chk.close("Clifford depth-20 median of 10 means of <ZZ> at the Eq. (5) size is within 0.2 of 1", est_nb, 1.0, 0.2)
    _ = chk.close("Clifford depth-20 sample mean of <ZZ> is within 5 standard errors of the exact walk value 1.0369728090", mean_nb, 1.0369728090, 5.0 * se_nb)
    _ = chk.check("Clifford depth-20 snapshots have trace one at the Eq. (5) size", nb_c.trace_dev <= 1e-12, "max deviation=" + String(nb_c.trace_dev))
    _ = chk.check("sampled depth-20 Clifford unitaries are unitary at the Eq. (5) size", nb_c.unit_dev <= 1e-12, "max deviation=" + String(nb_c.unit_dev))

    # Extra, not in the notebook: depth 60, where the walk's exact expectation is 1.0000114510 (half-group only).
    var big_c = clifford_run(n_clif, UInt64(555), 60)
    var est_bc = median_of_means(big_c.xs, 10)
    var mean_bc = mean(big_c.xs)
    var se_bc = sqrt(sample_var(big_c.xs) / Float64(len(big_c.xs)))
    print("Clifford depth 60 (extra),", len(big_c.xs), "snapshots: median of 10 means =", est_bc, " sample mean =", mean_bc, " standard error =", se_bc)
    _ = chk.close("extra: Clifford depth-60 median of 10 means of <ZZ> is within 0.2 of 1", est_bc, 1.0, 0.2)
    _ = chk.close("extra: Clifford depth-60 sample mean of <ZZ> is within 5 standard errors of the exact walk value 1.0000114510", mean_bc, 1.0000114510, 5.0 * se_bc)
    _ = chk.check("extra: Clifford depth-60 snapshots have trace one", big_c.trace_dev <= 1e-12, "max deviation=" + String(big_c.trace_dev))
    _ = chk.check("extra: sampled depth-60 Clifford unitaries are unitary", big_c.unit_dev <= 1e-12, "max deviation=" + String(big_c.unit_dev))

    chk.finish()
