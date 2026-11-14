"""VarQITE for Max-Cut: Y-Z counteradiabatic-style ansatz, G theta_dot = D, exact expectations.

Reproduces: Classiq library - Variational Quantum Imaginary Time Evolution (VarQITE) for Combinatorial Problems
Framework: Classiq - https://github.com/Classiq/classiq-library
Source: https://github.com/Classiq/classiq-library/blob/main/applications/optimization/variational_quantum_imaginary_time_evolution/variational_quantum_imaginary_time_evolution.ipynb (Classiq Library, MIT)

The notebook prepares |+>^V, applies exp(-i theta_j Z_a Y_b) for every node pair (V(V-1)/2 angles, pairs ordered by vertex weight, Z on the lighter node), and iterates theta <- theta + dtau * theta_dot with G theta_dot = D (SVD pseudo-inverse, cutoff 0.01 s_max), G_{a,j} = Re<psi|P_a d_j psi>, D_a = <P_a (E - H_c)>, E the previous energy, H_c the CS-convention Max-Cut Hamiltonian. This example runs the same loop on a 4-node complete graph with QUMO's state vector.
Checks: G (parameter-shift on QUMO circuits) and D (Pauli products on the QUMO state) against an independent plain-list state-vector reference that differentiates the ansatz analytically; the pseudo-inverse against the Penrose conditions on the real G (full column rank) and, for the 0.01 s_max cutoff, against a matrix built from chosen singular vectors and values (one case just under the cutoff that must be dropped, one just over that must be kept); that every G of the 14-step loop is full column rank (smallest s/s_max above the cutoff, so the cutoff is inactive in this run); the whole 14-step trajectory of angles and energies against the same loop run on the reference; the Hamiltonian ground energy against a brute-force cut enumeration; and that the energy stays above -maxcut and that the final angles have lower energy than |+>^4 (energy of the final angles recomputed by the reference).
Limits: the trajectory checks compare QUMO with a reference loop written for this example (same loop and pinv, independent state, G and D); they do not compare with a printed notebook run, because the notebook graph is random. That the energy falls is an observation for this graph, not a theorem: with E_tau the previous energy the notebook's own energies are non-monotonic under shot noise. The cutoff is verified on the constructed matrix, not on the loop's G.

Deviations from the notebook: (1) the notebook's graph is a random Newman-Watts-Strogatz graph (networkx is not available here), so this uses a fixed K4 with weights 0.85 0.42 0.71 0.64 0.27 0.93; the notebook's printed numbers belong to its own random weights and cannot be compared, the references are analytic. (2) Expectation values are exact instead of 128-shot estimates, so the trajectory is deterministic and has no shot noise. (3) The notebook's code divides the parameter-shift difference by 4. For exp(-i theta P) the difference f(theta+pi/4) - f(theta-pi/4) equals d f/d theta, and G = Re<psi|P_a d_j psi> = d<P_a>/d theta_j / 2 whatever the sign convention, so this uses the factor 1/2 (the notebook's stated definition, checked below against the analytic derivative). If Classiq's single_pauli_exponent is exp(-i coefficient P), which this example assumes but does not verify, the notebook's G is half of this one and its theta_dot, hence its step, is twice as large as here; the shape of the trajectory is the same but the notebook's energies are not reproduced. (4) D is computed from Pauli products (E<P_a> - sum_b c_b <P_a P_b>) instead of per-shot sums. (5) The SVD is done through the eigen-decomposition of G^T G (Jacobi), since QUMO has no linear-algebra library; the cutoff 0.01 s_max is kept. (6) The partition is read from the signs of <Z_i> as in the notebook (<Z_i> <= 0 goes to group 1, > 0 to group 2; the cost is symmetric under swapping groups).

Run: .venv/bin/mojo run -I . examples/frameworks/classiq/variational_quantum_imaginary_time_evolution.mojo
example-meta: {"framework": "classiq", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from std.math import sqrt, cos, sin, pi
from qumo.ir.circuit import Circuit
from qumo.ir.pauli import PauliTerm
from qumo.bench.circuits import pauli_evolution
from qumo.sim.runner import evolve
from qumo.testing.checks import CheckLog


struct Problem(Copyable, Movable):
    var n: Int
    var eu: List[Int]
    var ev: List[Int]
    var ew: List[Float64]
    # ansatz pairs: Z on za[j], Y on yb[j]
    var za: List[Int]
    var yb: List[Int]

    def __init__(out self, n: Int):
        self.n = n
        self.eu = List[Int]()
        self.ev = List[Int]()
        self.ew = List[Float64]()
        self.za = List[Int]()
        self.yb = List[Int]()

    def total_weight(self) -> Float64:
        var s = 0.0
        for w in self.ew:
            s += w
        return s

    def num_angles(self) -> Int:
        return len(self.za)


def build_problem() -> Problem:
    var p = Problem(4)
    var us: List[Int] = [0, 0, 0, 1, 1, 2]
    var vs: List[Int] = [1, 2, 3, 2, 3, 3]
    var ws: List[Float64] = [0.85, 0.42, 0.71, 0.64, 0.27, 0.93]
    for i in range(6):
        p.eu.append(us[i])
        p.ev.append(vs[i])
        p.ew.append(ws[i])
    # vertex weights, then node pairs sorted by (min, max) of the pair's vertex weights (stable, like Python sorted)
    var vw = List[Float64](length=p.n, fill=0.0)
    for e in range(len(p.ew)):
        vw[p.eu[e]] += p.ew[e]
        vw[p.ev[e]] += p.ew[e]
    var pa = List[Int]()
    var pb = List[Int]()
    for i in range(p.n):
        for j in range(i + 1, p.n):
            pa.append(i)
            pb.append(j)
    var m = len(pa)
    for i in range(1, m):
        var ka = pa[i]
        var kb = pb[i]
        var klo = min(vw[ka], vw[kb])
        var khi = max(vw[ka], vw[kb])
        var j = i - 1
        while j >= 0:
            var lo = min(vw[pa[j]], vw[pb[j]])
            var hi = max(vw[pa[j]], vw[pb[j]])
            if lo > klo or (lo == klo and hi > khi):
                pa[j + 1] = pa[j]
                pb[j + 1] = pb[j]
                j -= 1
            else:
                break
        pa[j + 1] = ka
        pb[j + 1] = kb
    for k in range(m):
        if vw[pa[k]] < vw[pb[k]]:
            p.za.append(pa[k])
            p.yb.append(pb[k])
        else:
            p.za.append(pb[k])
            p.yb.append(pa[k])
    return p^


def parity(x: Int) -> Int:
    var c = 0
    var v = x
    while v != 0:
        c += v & 1
        v >>= 1
    return c & 1


def zsign(mask: Int, z: Int) -> Float64:
    return -1.0 if parity(mask & z) == 1 else 1.0


# ---- Hamiltonian: term 0..E-1 = (w/2) Z_u Z_v, term E = -W/2 identity (CS convention) ----
def ham_masks(p: Problem) -> List[Int]:
    var out = List[Int]()
    for e in range(len(p.ew)):
        out.append((1 << p.eu[e]) | (1 << p.ev[e]))
    out.append(0)
    return out^


def ham_coeffs(p: Problem) -> List[Float64]:
    var out = List[Float64]()
    for e in range(len(p.ew)):
        out.append(p.ew[e] / 2.0)
    out.append(-p.total_weight() / 2.0)
    return out^


def h_of_state(masks: List[Int], coeffs: List[Float64], z: Int) -> Float64:
    var s = 0.0
    for a in range(len(masks)):
        s += coeffs[a] * zsign(masks[a], z)
    return s


# ---- QUMO side ----
def ansatz(p: Problem, thetas: List[Float64]) -> Circuit:
    """|+>^n then exp(-i theta_j Z_za Y_yb) = pauli_evolution(angle = 2 theta_j) in the pair order."""
    var c = Circuit(p.n, name="maxcut_ansatz")
    for q in range(p.n):
        c.h(q)
    for j in range(p.num_angles()):
        var x = UInt64(1) << UInt64(p.yb[j])
        var z = (UInt64(1) << UInt64(p.yb[j])) | (UInt64(1) << UInt64(p.za[j]))
        pauli_evolution(c, PauliTerm(1.0, x, z), 2.0 * thetas[j])
    return c^


def qumo_zexp(p: Problem, thetas: List[Float64], masks: List[Int]) raises -> List[Float64]:
    var sv = evolve(ansatz(p, thetas))
    var out = List[Float64]()
    for m in masks:
        out.append(sv.expectation_term(UInt64(0), UInt64(m)))
    return out^


def qumo_G(p: Problem, thetas: List[Float64], masks: List[Int]) raises -> List[Float64]:
    """Row-major (num terms x num angles), parameter shift with +-pi/4 and factor 1/2."""
    var na = p.num_angles()
    var G = List[Float64](length=len(masks) * na, fill=0.0)
    var th = thetas.copy()
    for j in range(na):
        th[j] = thetas[j] + pi / 4.0
        var fp = qumo_zexp(p, th, masks)
        th[j] = thetas[j] - pi / 4.0
        var fm = qumo_zexp(p, th, masks)
        th[j] = thetas[j]
        for a in range(len(masks)):
            G[a * na + j] = (fp[a] - fm[a]) / 2.0
    return G^


def qumo_D(p: Problem, thetas: List[Float64], masks: List[Int], coeffs: List[Float64], prev_e: Float64) raises -> Tuple[List[Float64], Float64]:
    """D_a = E <P_a> - sum_b c_b <P_a P_b>; also returns <H_c>."""
    var sv = evolve(ansatz(p, thetas))
    var D = List[Float64]()
    for a in range(len(masks)):
        var d = prev_e * sv.expectation_term(UInt64(0), UInt64(masks[a]))
        for b in range(len(masks)):
            d -= coeffs[b] * sv.expectation_term(UInt64(0), UInt64(masks[a] ^ masks[b]))
        D.append(d)
    var e = 0.0
    for b in range(len(masks)):
        e += coeffs[b] * sv.expectation_term(UInt64(0), UInt64(masks[b]))
    return (D^, e)


# ---- independent reference: plain-list complex state vector ----
struct RefState(Copyable, Movable):
    var re: List[Float64]
    var im: List[Float64]

    def __init__(out self, n: Int):
        var N = 1 << n
        self.re = List[Float64](length=N, fill=1.0 / sqrt(Float64(N)))
        self.im = List[Float64](length=N, fill=0.0)


def apply_zy(st: RefState, a: Int, b: Int) -> RefState:
    """(Z_a Y_b) psi: out[i] = (-1)^{i_a} * (i if i_b else -i) * psi[i ^ 2^b]."""
    var N = len(st.re)
    var out = RefState(1)
    out.re = List[Float64](length=N, fill=0.0)
    out.im = List[Float64](length=N, fill=0.0)
    for i in range(N):
        var j = i ^ (1 << b)
        var s = -1.0 if ((i >> a) & 1) == 1 else 1.0
        var t = 1.0 if ((i >> b) & 1) == 1 else -1.0
        # (re + i im) * (i s t) = s t (-im + i re)
        out.re[i] = s * t * (-st.im[j])
        out.im[i] = s * t * st.re[j]
    return out^


def ref_state(p: Problem, thetas: List[Float64], deriv: Int) -> RefState:
    """psi(theta), or d psi / d theta_deriv, built from exp(-i th P) = cos th - i sin th P."""
    var st = RefState(p.n)
    for k in range(p.num_angles()):
        var pp = apply_zy(st, p.za[k], p.yb[k])
        var c = cos(thetas[k])
        var s = sin(thetas[k])
        var N = len(st.re)
        var nr = List[Float64](length=N, fill=0.0)
        var ni = List[Float64](length=N, fill=0.0)
        for i in range(N):
            # c psi - i s P psi
            nr[i] = c * st.re[i] + s * pp.im[i]
            ni[i] = c * st.im[i] - s * pp.re[i]
        st.re = nr^
        st.im = ni^
        if k == deriv:
            var q = apply_zy(st, p.za[k], p.yb[k])
            # -i P psi
            for i in range(N):
                st.re[i] = q.im[i]
                st.im[i] = -q.re[i]
    return st^


def ref_G(p: Problem, thetas: List[Float64], masks: List[Int]) -> List[Float64]:
    var na = p.num_angles()
    var G = List[Float64](length=len(masks) * na, fill=0.0)
    var psi = ref_state(p, thetas, -1)
    for j in range(na):
        var d = ref_state(p, thetas, j)
        for a in range(len(masks)):
            var s = 0.0
            for i in range(len(psi.re)):
                s += zsign(masks[a], i) * (psi.re[i] * d.re[i] + psi.im[i] * d.im[i])
            G[a * na + j] = s
    return G^


def ref_D(p: Problem, thetas: List[Float64], masks: List[Int], coeffs: List[Float64], prev_e: Float64) -> Tuple[List[Float64], Float64]:
    var psi = ref_state(p, thetas, -1)
    var D = List[Float64](length=len(masks), fill=0.0)
    var e = 0.0
    for i in range(len(psi.re)):
        var prob = psi.re[i] * psi.re[i] + psi.im[i] * psi.im[i]
        var h = h_of_state(masks, coeffs, i)
        e += prob * h
        for a in range(len(masks)):
            D[a] += prob * zsign(masks[a], i) * (prev_e - h)
    return (D^, e)


# ---- linear algebra: pseudo-inverse through the eigen-decomposition of G^T G ----
def jacobi_eig(var A: List[Float64], n: Int) -> Tuple[List[Float64], List[Float64]]:
    """Symmetric eigenproblem by cyclic Jacobi: (eigenvalues, V row-major with eigenvectors in columns)."""
    var V = List[Float64](length=n * n, fill=0.0)
    for i in range(n):
        V[i * n + i] = 1.0
    for _sweep in range(60):
        var off = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                off += A[i * n + j] * A[i * n + j]
        if off < 1e-30:
            break
        for pq_p in range(n):
            for pq_q in range(pq_p + 1, n):
                var apq = A[pq_p * n + pq_q]
                if abs(apq) < 1e-300:
                    continue
                var theta = (A[pq_q * n + pq_q] - A[pq_p * n + pq_p]) / (2.0 * apq)
                var t = 1.0 / (abs(theta) + sqrt(theta * theta + 1.0))
                if theta < 0.0:
                    t = -t
                var c = 1.0 / sqrt(t * t + 1.0)
                var s = t * c
                for k in range(n):
                    var akp = A[k * n + pq_p]
                    var akq = A[k * n + pq_q]
                    A[k * n + pq_p] = c * akp - s * akq
                    A[k * n + pq_q] = s * akp + c * akq
                for k in range(n):
                    var apk = A[pq_p * n + k]
                    var aqk = A[pq_q * n + k]
                    A[pq_p * n + k] = c * apk - s * aqk
                    A[pq_q * n + k] = s * apk + c * aqk
                for k in range(n):
                    var vkp = V[k * n + pq_p]
                    var vkq = V[k * n + pq_q]
                    V[k * n + pq_p] = c * vkp - s * vkq
                    V[k * n + pq_q] = s * vkp + c * vkq
    var ev = List[Float64]()
    for i in range(n):
        ev.append(A[i * n + i])
    return (ev^, V^)


def pinv(G: List[Float64], m: Int, n: Int, threshold: Float64) -> List[Float64]:
    """Moore-Penrose pseudo-inverse (n x m, row-major) dropping singular values below threshold * s_max."""
    var A = List[Float64](length=n * n, fill=0.0)
    for i in range(n):
        for j in range(n):
            var s = 0.0
            for r in range(m):
                s += G[r * n + i] * G[r * n + j]
            A[i * n + j] = s
    var eg = jacobi_eig(A^, n)
    var smax = 0.0
    for k in range(n):
        smax = max(smax, sqrt(max(eg[0][k], 0.0)))
    var P = List[Float64](length=n * m, fill=0.0)
    for k in range(n):
        var sk = sqrt(max(eg[0][k], 0.0))
        if sk <= 1e-12 or sk < threshold * smax:
            continue
        # P += (1/s^2) v_k v_k^T G^T
        for i in range(n):
            for r in range(m):
                var gv = 0.0
                for j in range(n):
                    gv += eg[1][j * n + k] * G[r * n + j]
                P[i * m + r] += eg[1][i * n + k] * gv / (sk * sk)
    return P^


def min_sv_ratio(G: List[Float64], m: Int, n: Int) -> Float64:
    """Smallest singular value over the largest one (through eigenvalues of G^T G)."""
    var A = List[Float64](length=n * n, fill=0.0)
    for i in range(n):
        for j in range(n):
            var s = 0.0
            for r in range(m):
                s += G[r * n + i] * G[r * n + j]
            A[i * n + j] = s
    var eg = jacobi_eig(A^, n)
    var lo = 1e300
    var hi = 0.0
    for k in range(n):
        var sk = sqrt(max(eg[0][k], 0.0))
        lo = min(lo, sk)
        hi = max(hi, sk)
    return lo / hi


def built_matrix(s: List[Float64], keep: List[Bool]) -> Tuple[List[Float64], List[Float64]]:
    """4x3 G = sum_k s_k u_k v_k^T from chosen orthonormal u, v, and its expected pseudo-inverse sum over kept k of v_k u_k^T / s_k (3x4)."""
    var r2 = sqrt(0.5)
    var u: List[Float64] = [0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, -0.5]
    var v: List[Float64] = [r2, r2, 0.0, r2, -r2, 0.0, 0.0, 0.0, 1.0]
    var G = List[Float64](length=12, fill=0.0)
    var P = List[Float64](length=12, fill=0.0)
    for k in range(3):
        for i in range(4):
            for j in range(3):
                G[i * 3 + j] += s[k] * u[k * 4 + i] * v[k * 3 + j]
                if keep[k]:
                    P[j * 4 + i] += v[k * 3 + j] * u[k * 4 + i] / s[k]
    return (G^, P^)


def matvec(P: List[Float64], rows: Int, cols: Int, v: List[Float64]) -> List[Float64]:
    var out = List[Float64](length=rows, fill=0.0)
    for i in range(rows):
        for j in range(cols):
            out[i] += P[i * cols + j] * v[j]
    return out^


def matmul(A: List[Float64], B: List[Float64], r: Int, k: Int, c: Int) -> List[Float64]:
    var out = List[Float64](length=r * c, fill=0.0)
    for i in range(r):
        for j in range(c):
            var s = 0.0
            for t in range(k):
                s += A[i * k + t] * B[t * c + j]
            out[i * c + j] = s
    return out^


def max_abs_diff(a: List[Float64], b: List[Float64]) -> Float64:
    var d = 0.0
    for i in range(len(a)):
        d = max(d, abs(a[i] - b[i]))
    return d


# ---- Max-Cut helpers ----
def cut_value(p: Problem, side: Int) -> Float64:
    """Sum of weights of edges whose end nodes lie on different sides (bit i of side = node i)."""
    var s = 0.0
    for e in range(len(p.ew)):
        if ((side >> p.eu[e]) & 1) != ((side >> p.ev[e]) & 1):
            s += p.ew[e]
    return s


def brute_force_maxcut(p: Problem) -> Float64:
    var best = 0.0
    for side in range(1 << p.n):
        best = max(best, cut_value(p, side))
    return best


def partition_cost(p: Problem, zexp_single: List[Float64]) -> Float64:
    """The notebook's partition from signs of <Z_i> (group 1 where <Z_i> <= 0), CS cost -sum of cut weights."""
    var side = 0
    for i in range(p.n):
        if zexp_single[i] <= 0.0:
            side |= 1 << i
    return -cut_value(p, side)


struct Run(Movable):
    var thetas: List[Float64]
    var energies: List[Float64]
    var costs: List[Float64]
    var ratios: List[Float64]

    def __init__(out self):
        self.thetas = List[Float64]()
        self.energies = List[Float64]()
        self.costs = List[Float64]()
        self.ratios = List[Float64]()


def var_it(p: Problem, use_qumo: Bool, iters: Int, delta_tau: Float64, verbose: Bool) raises -> Run:
    var masks = ham_masks(p)
    var coeffs = ham_coeffs(p)
    var singles = List[Int]()
    for i in range(p.n):
        singles.append(1 << i)
    var na = p.num_angles()
    var run = Run()
    for _ in range(na):
        run.thetas.append(0.0)
    var energy = 0.0
    for it in range(1, iters + 1):
        var G: List[Float64]
        var D: List[Float64]
        var e: Float64
        var zs: List[Float64]
        if use_qumo:
            G = qumo_G(p, run.thetas, masks)
            var dd = qumo_D(p, run.thetas, masks, coeffs, energy)
            D = dd[0].copy()
            e = dd[1]
            zs = qumo_zexp(p, run.thetas, singles)
        else:
            G = ref_G(p, run.thetas, masks)
            var dd = ref_D(p, run.thetas, masks, coeffs, energy)
            D = dd[0].copy()
            e = dd[1]
            var psi = ref_state(p, run.thetas, -1)
            zs = List[Float64]()
            for i in range(p.n):
                var s = 0.0
                for k in range(len(psi.re)):
                    s += zsign(1 << i, k) * (psi.re[k] * psi.re[k] + psi.im[k] * psi.im[k])
                zs.append(s)
        run.ratios.append(min_sv_ratio(G, len(masks), na))
        var P = pinv(G, len(masks), na, 0.01)
        var dot = matvec(P, na, len(masks), D)
        var step = delta_tau / sqrt(Float64(it))
        for j in range(na):
            run.thetas[j] += step * dot[j]
        energy = e
        var cost = partition_cost(p, zs)
        run.energies.append(e)
        run.costs.append(cost)
        if verbose:
            print("iteration", it, ": energy =", e, ", maxcut_cost =", cost)
    return run^


def main() raises:
    var log = CheckLog("classiq/variational_quantum_imaginary_time_evolution")
    var p = build_problem()
    var masks = ham_masks(p)
    var coeffs = ham_coeffs(p)
    var na = p.num_angles()
    print("pairs (Z node, Y node):")
    for j in range(na):
        print(" ", j, ":", p.za[j], p.yb[j])

    # ---- problem: Hamiltonian ground energy = -maxcut ----
    var maxcut = brute_force_maxcut(p)
    var e0 = 1e9
    for z in range(1 << p.n):
        e0 = min(e0, h_of_state(masks, coeffs, z))
    _ = log.check("ansatz has V(V-1)/2 = 6 rotations", na == 6, "na=" + String(na))
    _ = log.close("ground energy of H_c equals -maxcut (brute-force cut enumeration)", e0, -maxcut, 1e-12)
    var zeros = List[Float64](length=na, fill=0.0)
    var e_start = qumo_D(p, zeros, masks, coeffs, 0.0)[1]
    _ = log.close("<H_c> in |+>^4 is -W/2", e_start, -p.total_weight() / 2.0, 1e-12)

    # ---- state, G and D against the independent reference at a generic angle vector ----
    var th: List[Float64] = [0.31, -0.22, 0.47, 0.13, -0.35, 0.28]
    var qs = evolve(ansatz(p, th))
    var rs = ref_state(p, th, -1)
    var amp_err = 0.0
    # global phase of pauli_evolution on non-identity terms is zero, so amplitudes agree directly
    for i in range(1 << p.n):
        amp_err = max(amp_err, abs(qs.re[i] - rs.re[i]) + abs(qs.im[i] - rs.im[i]))
    _ = log.close("QUMO ansatz state equals the reference state (max amplitude error)", amp_err, 0.0, 1e-12)
    var Gq = qumo_G(p, th, masks)
    var Gr = ref_G(p, th, masks)
    _ = log.close("G: parameter shift on QUMO vs analytic Re<psi|P_a d_j psi> (max abs diff)", max_abs_diff(Gq, Gr), 0.0, 1e-12)
    var Gq0 = qumo_G(p, zeros, masks)
    var Gr0 = ref_G(p, zeros, masks)
    _ = log.close("G at theta = 0 vs analytic (max abs diff)", max_abs_diff(Gq0, Gr0), 0.0, 1e-12)
    var dq = qumo_D(p, th, masks, coeffs, -1.3)
    var dr = ref_D(p, th, masks, coeffs, -1.3)
    _ = log.close("D vs reference sum over probabilities (max abs diff)", max_abs_diff(dq[0], dr[0]), 0.0, 1e-12)
    _ = log.close("<H_c> from Pauli expectations vs reference", dq[1], dr[1], 1e-12)

    # ---- pseudo-inverse: Penrose conditions on the real (full column rank) G, no truncation ----
    var Pg = pinv(Gr, len(masks), na, 1e-9)
    var GPG = matmul(matmul(Gr, Pg, len(masks), na, len(masks)), Gr, len(masks), len(masks), na)
    var PGP = matmul(matmul(Pg, Gr, na, len(masks), na), Pg, na, na, len(masks))
    _ = log.close("pseudo-inverse: G G+ G = G (max abs diff)", max_abs_diff(GPG, Gr), 0.0, 1e-9)
    _ = log.close("pseudo-inverse: G+ G G+ = G+ (max abs diff)", max_abs_diff(PGP, Pg), 0.0, 1e-9)
    var GP = matmul(Gr, Pg, len(masks), na, len(masks))
    var sym = 0.0
    for i in range(len(masks)):
        for j in range(len(masks)):
            sym = max(sym, abs(GP[i * len(masks) + j] - GP[j * len(masks) + i]))
    _ = log.close("pseudo-inverse: G G+ is symmetric", sym, 0.0, 1e-9)

    # ---- the 0.01 s_max cutoff on a matrix with known singular values (3, 1, s3) ----
    var bm = built_matrix([3.0, 1.0, 0.02], [True, True, False])
    var pdrop = pinv(bm[0], 4, 3, 0.01)
    _ = log.close("cutoff: s3/s_max = 0.0067 < 0.01 is dropped (max abs diff vs expected pinv)", max_abs_diff(pdrop, bm[1]), 0.0, 1e-9)
    var bk = built_matrix([3.0, 1.0, 0.04], [True, True, True])
    var pkeep = pinv(bk[0], 4, 3, 0.01)
    _ = log.close("cutoff: s3/s_max = 0.0133 > 0.01 is kept (max abs diff vs expected pinv)", max_abs_diff(pkeep, bk[1]), 0.0, 1e-9)
    _ = log.close("singular value ratio of the built matrix (s3/s1 = 0.04/3)", min_sv_ratio(bk[0], 4, 3), 0.04 / 3.0, 1e-9)

    # ---- the VarQITE loop: 14 steps, QUMO against the all-reference run ----
    var iters = 14
    var tau = 0.1
    var qrun = var_it(p, True, iters, tau, True)
    var rrun = var_it(p, False, iters, tau, False)
    _ = log.close("14-step angle trajectory: QUMO loop vs reference loop (max abs diff)", max_abs_diff(qrun.thetas, rrun.thetas), 0.0, 1e-9)
    _ = log.close("14-step energy history: QUMO loop vs reference loop (max abs diff)", max_abs_diff(qrun.energies, rrun.energies), 0.0, 1e-9)
    var e_final = qrun.energies[iters - 1]
    var lowest = 1e9
    for e in qrun.energies:
        lowest = min(lowest, e)
        _ = log.check("energy stays above the ground energy -maxcut (variational bound)", e >= -maxcut - 1e-12, "e=" + String(e))
    # energy of the angles after the last update, from the independent reference state
    var rfin = ref_state(p, qrun.thetas, -1)
    var e_after = 0.0
    for i in range(len(rfin.re)):
        e_after += (rfin.re[i] * rfin.re[i] + rfin.im[i] * rfin.im[i]) * h_of_state(masks, coeffs, i)
    print("optimal maxcut", maxcut, "| <H> start", e_start, "-> step 14", e_final, "-> after last update", e_after, "| lowest", lowest)
    _ = log.check("energy after the 14 updates (reference state) is below the |+>^4 value", e_after < e_start, "start=" + String(e_start) + " after=" + String(e_after))
    _ = log.check("energy after the 14 updates stays above -maxcut", e_after >= -maxcut - 1e-12, "after=" + String(e_after))
    var min_ratio = 1e9
    for r in qrun.ratios:
        min_ratio = min(min_ratio, r)
    print("smallest s/s_max of G over the loop:", min_ratio)
    _ = log.check("every G in the loop is full column rank: s_min/s_max > 0.01, the cutoff discards nothing", min_ratio > 0.01, "min ratio=" + String(min_ratio))
    var best_cost = 0.0
    for c in qrun.costs:
        best_cost = min(best_cost, c)
        _ = log.check("partition cost never beats the optimum -maxcut", c >= -maxcut - 1e-12, "cost=" + String(c))
    print("best partition cost over the run", best_cost)

    log.finish()
