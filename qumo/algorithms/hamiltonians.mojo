"""Problem Hamiltonians and exact reference solvers.

Builders return `PauliSum`s (bit-packed symplectic Pauli terms, see
`qumo.ir.pauli`) so they plug directly into `StateVector.expectation`,
gradients, VQE/QAOA and Trotterisation:

  * `maxcut(graph)`                 - Ising form of weighted MaxCut (QAOA, Farhi et al.)
  * `transverse_field_ising(n,J,h)` - 1D TFIM chain (open / periodic)
  * `heisenberg_xxz(n,J,delta)`     - XXZ chain
  * `h2_sto3g(num_qubits=4|2)`      - H2 / STO-3G at R = 0.735 A, generated from
                                      the molecular-orbital integrals (Qiskit
                                      Nature / PySCF values), Jordan-Wigner
                                      (4 qubits, interleaved spin orbitals
                                      0=g-up 1=g-down 2=u-up 3=u-down) or
                                      parity-reduced (2 qubits, Qiskit tutorial form).

Exact references (the "ground truth" agents verify against):
  * `PauliOperator`      - H|psi> with the diagonal part pre-summed into one
                           vector (one fused pass for all Z-strings) and
                           off-diagonal terms applied on the fly; O(2^n) memory.
  * `ground_state(h, n)` - Lanczos with full re-orthogonalisation on the
                           state vector (scales to ~18 qubits).
  * `exact_spectrum(h, n)` - dense diagonalisation (Jacobi) for n <= 7.
  * `exact_evolve(h, psi, t)` - exp(-iHt)|psi> by scaled Taylor series
                           (reference for Trotter error studies).

Graphs: `Graph` keeps weighted edges, parses "0-1 1-2:0.5" (AI-friendly text),
and brute-forces MaxCut for the approximation-ratio reference.

Error codes: QM-E803 (Hamiltonian/qubit mismatch), QM-E807 (problem too large
for exact solver), QM-E809 (graph errors).
"""

from std.math import sqrt, ceil
from std.bit import pop_count
from std.runtime import parallelism_level
from max.algorithm import parallelize
from qumo.ir.pauli import PauliTerm, PauliSum
from qumo.sim.statevector import StateVector, run_chunks, PAR_MIN_ITEMS
from qumo.sim.rng import Rng
from qumo.algorithms.linalg import (
    sv_dot, sv_norm, sv_zero, sv_assign, sv_scale, sv_axpy, sv_mul_minus_i,
    jacobi_eigh, tridiag_lowest,
)


# ---------------------------------------------------------------------------
# Graphs (MaxCut / QAOA inputs)
# ---------------------------------------------------------------------------
@fieldwise_init
struct Edge(ImplicitlyCopyable, Writable):
    var u: Int
    var v: Int
    var w: Float64

    def write_to(self, mut w: Some[Writer]):
        w.write(self.u, "-", self.v)
        if self.w != 1.0:
            w.write(":", self.w)


struct Graph(Copyable, Writable, Sized):
    """Undirected weighted graph on nodes 0..num_nodes-1 (node k <-> qubit k)."""

    var num_nodes: Int
    var edges: List[Edge]

    def __init__(out self, num_nodes: Int):
        self.num_nodes = num_nodes
        self.edges = List[Edge]()

    def __len__(self) -> Int:
        return len(self.edges)

    def add_edge(mut self, u: Int, v: Int, w: Float64 = 1.0) raises:
        if u == v:
            raise Error("QM-E809: self-loop " + String(u) + "-" + String(v) + " is not allowed. Hint: MaxCut edges join two distinct nodes.")
        if u < 0 or v < 0:
            raise Error("QM-E809: negative node index in edge " + String(u) + "-" + String(v) + ".")
        if max(u, v) >= self.num_nodes:
            self.num_nodes = max(u, v) + 1
        self.edges.append(Edge(min(u, v), max(u, v), w))

    def total_weight(self) -> Float64:
        var s = 0.0
        for e in self.edges:
            s += e.w
        return s

    @staticmethod
    def ring(n: Int, w: Float64 = 1.0) raises -> Graph:
        """Cycle graph 0-1-...-(n-1)-0 (n >= 3)."""
        if n < 3:
            raise Error("QM-E809: ring() needs n >= 3, got " + String(n) + ". Hint: use Graph.path(n) for n = 2.")
        var g = Graph(n)
        for i in range(n):
            g.add_edge(i, (i + 1) % n, w)
        return g^

    @staticmethod
    def path(n: Int, w: Float64 = 1.0) raises -> Graph:
        var g = Graph(n)
        for i in range(n - 1):
            g.add_edge(i, i + 1, w)
        return g^

    @staticmethod
    def complete(n: Int, w: Float64 = 1.0) raises -> Graph:
        var g = Graph(n)
        for i in range(n):
            for j in range(i + 1, n):
                g.add_edge(i, j, w)
        return g^

    @staticmethod
    def parse(text: String) raises -> Graph:
        """Parse an edge list such as "0-1 1-2 2-0:0.5" or "0-1,1-2" (weight after ':')."""
        var g = Graph(0)
        var norm = text.replace(",", " ").replace(";", " ")
        for tok_slice in norm.split():
            var tok = String(tok_slice)
            if tok.byte_length() == 0:
                continue
            var w = 1.0
            var body = tok
            var colon = tok.find(":")
            if colon >= 0:
                w = Float64(String(tok[byte=colon + 1:]))
                body = String(tok[byte=:colon])
            var dash = body.find("-")
            if dash <= 0:
                raise Error("QM-E809: cannot parse edge '" + tok + "'. Hint: write edges as 'u-v' or 'u-v:weight', e.g. '0-1 1-2:0.5'.")
            var u = Int(String(body[byte=:dash]))
            var v = Int(String(body[byte=dash + 1:]))
            g.add_edge(u, v, w)
        return g^

    def cut_value(self, assignment: Int) -> Float64:
        """Weight of edges crossing the partition given by the bits of `assignment`."""
        var s = 0.0
        for e in self.edges:
            if ((assignment >> e.u) & 1) != ((assignment >> e.v) & 1):
                s += e.w
        return s

    def max_cut(self) raises -> Tuple[Float64, Int]:
        """Brute-force maximum cut (value, assignment) - reference for approximation ratios."""
        if self.num_nodes > 26:
            raise Error("QM-E807: brute-force max_cut() limited to 26 nodes, graph has " + String(self.num_nodes) + ". Hint: use QAOA/heuristics for larger graphs.")
        var best = -1.0
        var arg = 0
        # node n-1 fixed to 0 (cut symmetry) halves the search
        var limit = 1 << max(self.num_nodes - 1, 0)
        for a in range(limit):
            var c = self.cut_value(a)
            if c > best:
                best = c
                arg = a
        return (best, arg)

    def write_to(self, mut w: Some[Writer]):
        w.write("Graph(n=", self.num_nodes, ", edges=[")
        for i in range(len(self.edges)):
            if i > 0:
                w.write(" ")
            w.write(self.edges[i])
        w.write("])")


# ---------------------------------------------------------------------------
# Hamiltonian builders
# ---------------------------------------------------------------------------
def _bit(q: Int) -> UInt64:
    return UInt64(1) << UInt64(q)


def maxcut(graph: Graph, include_offset: Bool = True) -> PauliSum:
    """MaxCut cost as an Ising Hamiltonian to MINIMISE.

    H = sum_{(u,v,w)} w/2 (Z_u Z_v - 1)   =>   <z|H|z> = -cut(z).
    Ground energy = -maxcut. With include_offset=False only the ZZ part is
    returned (offset = -total_weight/2).
    """
    var h = PauliSum()
    for e in graph.edges:
        h.add(0.5 * e.w, 0, _bit(e.u) | _bit(e.v))
    if include_offset and len(graph.edges) > 0:
        h.add(-0.5 * graph.total_weight(), 0, 0)
    return h^


def transverse_field_ising(n: Int, j: Float64 = 1.0, h: Float64 = 1.0, periodic: Bool = False) raises -> PauliSum:
    """H = -J sum_i Z_i Z_{i+1} - h sum_i X_i on a 1D chain of n qubits."""
    if n < 1 or n > 64:
        raise Error("QM-E802: transverse_field_ising() needs 1 <= n <= 64, got " + String(n) + ".")
    var out = PauliSum()
    var nb = n if (periodic and n > 2) else n - 1
    for i in range(nb):
        out.add(-j, 0, _bit(i) | _bit((i + 1) % n))
    for i in range(n):
        out.add(-h, _bit(i), 0)
    return out^


def heisenberg_xxz(n: Int, j: Float64 = 1.0, delta: Float64 = 1.0, periodic: Bool = False, field: Float64 = 0.0) raises -> PauliSum:
    """H = J sum_i (X_i X_{i+1} + Y_i Y_{i+1} + delta Z_i Z_{i+1}) + field sum_i Z_i."""
    if n < 2 or n > 64:
        raise Error("QM-E802: heisenberg_xxz() needs 2 <= n <= 64, got " + String(n) + ".")
    var out = PauliSum()
    var nb = n if (periodic and n > 2) else n - 1
    for i in range(nb):
        var m = _bit(i) | _bit((i + 1) % n)
        out.add(j, m, 0)
        out.add(j, m, m)
        out.add(j * delta, 0, m)
    if field != 0.0:
        for i in range(n):
            out.add(field, 0, _bit(i))
    return out^


# H2 / STO-3G at R = 0.735 Angstrom: molecular-orbital integrals (Hartree), orbitals
# g = sigma_g (bonding), u = sigma_u (antibonding); chemist notation (pq|rs).
comptime H2_BOND_LENGTH = 0.735
comptime H2_H_GG = -1.2563390730032498
comptime H2_H_UU = -0.47189600728114184
comptime H2_J_GG = 0.6757101548035161
comptime H2_J_UU = 0.6985737227320210
comptime H2_J_GU = 0.6645817302552968
comptime H2_K_GU = 0.18093119978423156
comptime H2_NUCLEAR_REPULSION = 0.7199689944489797
comptime H2_EXACT_ENERGY = -1.1373060357533993  # FCI total energy (incl. nuclear repulsion)
comptime H2_HF_ENERGY = -1.1169989967540036  # Hartree-Fock total energy


def h2_sto3g(num_qubits: Int = 4, include_nuclear_repulsion: Bool = True) raises -> PauliSum:
    """Molecular hydrogen, STO-3G, R = 0.735 A.

    num_qubits = 4: Jordan-Wigner, interleaved spin orbitals
        (q0 = g-up, q1 = g-down, q2 = u-up, q3 = u-down); HF state = |0011>
        (basis index 3). 15 Pauli terms; FCI energy -1.137306 Ha.
    num_qubits = 2: parity mapping + two-qubit (symmetry) reduction, identical
        to the Qiskit VQE tutorial operator (II, IZ, ZI, ZZ, XX).

    Both are generated from the same six MO integrals (constants H2_*), so they
    share the exact ground energy H2_EXACT_ENERGY (electronic -1.857275 Ha +
    nuclear repulsion 0.719969 Ha).
    """
    var e_nuc = H2_NUCLEAR_REPULSION if include_nuclear_repulsion else 0.0
    var hg = H2_H_GG
    var hu = H2_H_UU
    var jg = H2_J_GG
    var ju = H2_J_UU
    var jgu = H2_J_GU
    var k = H2_K_GU
    var h = PauliSum()
    if num_qubits == 4:
        # n_p = (1 - Z_p)/2 expansion of the second-quantised Hamiltonian
        h.add(hg + hu + jg / 4 + ju / 4 + jgu - k / 2 + e_nuc, 0, 0)
        var zg = -hg / 2 - jg / 4 - jgu / 2 + k / 4
        var zu = -hu / 2 - ju / 4 - jgu / 2 + k / 4
        h.add(zg, 0, _bit(0))
        h.add(zg, 0, _bit(1))
        h.add(zu, 0, _bit(2))
        h.add(zu, 0, _bit(3))
        h.add(jg / 4, 0, _bit(0) | _bit(1))
        h.add(ju / 4, 0, _bit(2) | _bit(3))
        h.add((jgu - k) / 4, 0, _bit(0) | _bit(2))  # same spin
        h.add((jgu - k) / 4, 0, _bit(1) | _bit(3))
        h.add(jgu / 4, 0, _bit(0) | _bit(3))  # opposite spin
        h.add(jgu / 4, 0, _bit(1) | _bit(2))
        # exchange + pair hopping (qubit order 0,1,2,3)
        var c = k / 4
        h.add_label(-c, "YYXX")  # X0 X1 Y2 Y3  (dense label: leftmost = q3)
        h.add_label(c, "XYYX")   # X0 Y1 Y2 X3
        h.add_label(c, "YXXY")   # Y0 X1 X2 Y3
        h.add_label(-c, "XXYY")  # Y0 Y1 X2 X3
        return h^
    if num_qubits == 2:
        var e_gg = 2 * hg + jg
        var e_uu = 2 * hu + ju
        var e_ou = hg + hu + jgu
        var a = (e_uu - e_gg) / 4
        var b = (e_ou - (e_gg + e_uu) / 2) / 2
        var c0 = (e_ou + (e_gg + e_uu) / 2) / 2
        h.add(c0 + e_nuc, 0, 0)
        h.add(a, 0, _bit(0))
        h.add(-a, 0, _bit(1))
        h.add(b, 0, _bit(0) | _bit(1))
        h.add(k, _bit(0) | _bit(1), 0)
        return h^
    raise Error("QM-E802: h2_sto3g() supports num_qubits = 4 (Jordan-Wigner) or 2 (parity-reduced), got " + String(num_qubits) + ".")


def check_fits(h: PauliSum, num_qubits: Int) raises:
    var hq = h.num_qubits()
    if hq > num_qubits:
        raise Error("QM-E803: Hamiltonian acts on " + String(hq) + " qubits but the circuit/state has " + String(num_qubits) + ". Hint: build the circuit on at least " + String(hq) + " qubits.")


# ---------------------------------------------------------------------------
# PauliOperator: fast H|psi> and <psi|H|psi>
# ---------------------------------------------------------------------------
struct PauliOperator(Copyable):
    """A PauliSum prepared for repeated application to n-qubit states.

    Diagonal terms (pure Z strings + identity) are pre-summed into one dense
    vector `diag` (built once, O(#Z-terms * 2^n)); each application then costs
    one fused pass for all of them plus one pass per off-diagonal term.
    """

    var num_qubits: Int
    var diag: List[Float64]
    var has_diag: Bool
    var off: List[PauliTerm]

    def __init__(out self, h: PauliSum, num_qubits: Int) raises:
        check_fits(h, num_qubits)
        self.num_qubits = num_qubits
        self.off = List[PauliTerm]()
        self.has_diag = False
        var n = 1 << num_qubits
        var zs = List[PauliTerm]()
        for t in h.terms:
            if t.x == 0:
                zs.append(t)
            else:
                self.off.append(t)
        self.has_diag = len(zs) > 0
        self.diag = List[Float64](length=n if self.has_diag else 0, fill=0.0)
        if self.has_diag:
            var dp = self.diag.unsafe_ptr().as_unsafe_any_origin()
            var zp = zs.unsafe_ptr().as_unsafe_any_origin()
            var nz = len(zs)

            def body(lo: Int, hi: Int) {imm dp, imm zp, imm nz}:
                for i in range(lo, hi):
                    var s = 0.0
                    for k in range(nz):
                        var t = zp[unsafe_offset=k]
                        if (pop_count(i & Int(t.z)) & 1) == 1:
                            s -= t.coeff
                        else:
                            s += t.coeff
                    dp[unsafe_offset=i] = s

            run_chunks(body, n)
            _ = len(zs)  # keep `zs` alive: raw pointers do not extend lifetimes (ASAP destruction)

    def apply(self, src: StateVector, mut dst: StateVector):
        """dst <- H src  (dst is overwritten and must not alias src)."""
        var n = src.size()
        var sr = src.re.unsafe_ptr().as_unsafe_any_origin()
        var si = src.im.unsafe_ptr().as_unsafe_any_origin()
        var dr = dst.re.unsafe_ptr().as_unsafe_any_origin()
        var di = dst.im.unsafe_ptr().as_unsafe_any_origin()
        if self.has_diag:
            var dp = self.diag.unsafe_ptr().as_unsafe_any_origin()

            def dbody(lo: Int, hi: Int) {imm sr, imm si, imm dr, imm di, imm dp}:
                for i in range(lo, hi):
                    var d = dp[unsafe_offset=i]
                    dr[unsafe_offset=i] = d * sr[unsafe_offset=i]
                    di[unsafe_offset=i] = d * si[unsafe_offset=i]

            run_chunks(dbody, n)
        else:
            sv_zero(dst)
        for t in self.off:
            var xm = Int(t.x)
            var zm = Int(t.z)
            var ny = Int(pop_count(t.x & t.z)) & 3
            # factor = coeff * i^ny
            var fr = t.coeff
            var fi = 0.0
            if ny == 1:
                fr = 0.0
                fi = t.coeff
            elif ny == 2:
                fr = -t.coeff
            elif ny == 3:
                fr = 0.0
                fi = -t.coeff

            def obody(lo: Int, hi: Int) {imm sr, imm si, imm dr, imm di, imm xm, imm zm, imm fr, imm fi}:
                for i in range(lo, hi):
                    var a = sr[unsafe_offset=i]
                    var b = si[unsafe_offset=i]
                    var cr = fr
                    var ci = fi
                    if (pop_count(i & zm) & 1) == 1:
                        cr = -fr
                        ci = -fi
                    var j = i ^ xm
                    dr[unsafe_offset=j] = dr[unsafe_offset=j] + cr * a - ci * b
                    di[unsafe_offset=j] = di[unsafe_offset=j] + cr * b + ci * a

            run_chunks(obody, n)

    def expectation(self, mut psi: StateVector) -> Float64:
        """<psi|H|psi> without allocating a second state."""
        var total = 0.0
        if self.has_diag:
            var rp = psi.re.unsafe_ptr().as_unsafe_any_origin()
            var ip = psi.im.unsafe_ptr().as_unsafe_any_origin()
            var dp = self.diag.unsafe_ptr().as_unsafe_any_origin()
            var n = psi.size()
            var nchunks = 1 if n < PAR_MIN_ITEMS else parallelism_level() * 2
            var chunk = (n + nchunks - 1) // nchunks
            var partial = List[Float64](length=nchunks, fill=0.0)
            var pp = partial.unsafe_ptr().as_unsafe_any_origin()

            def task(c: Int) {imm rp, imm ip, imm dp, imm chunk, imm n, imm pp}:
                var lo = c * chunk
                var hi = min(lo + chunk, n)
                var s = 0.0
                for i in range(lo, hi):
                    var a = rp[unsafe_offset=i]
                    var b = ip[unsafe_offset=i]
                    s += dp[unsafe_offset=i] * (a * a + b * b)
                pp[unsafe_offset=c] = s

            if nchunks == 1:
                task(0)
            else:
                parallelize(task, nchunks)
            for c in range(nchunks):
                total += partial[c]
        for t in self.off:
            total += t.coeff * psi.expectation_term(t.x, t.z)
        return total


def apply_hamiltonian(h: PauliSum, src: StateVector, mut dst: StateVector) raises:
    """dst <- H src for a PauliSum (one-shot convenience; prefer PauliOperator in loops)."""
    var op = PauliOperator(h, src.num_qubits)
    op.apply(src, dst)


# ---------------------------------------------------------------------------
# Exact solvers
# ---------------------------------------------------------------------------
struct GroundState(Movable, Writable):
    var energy: Float64
    var state: StateVector
    var iterations: Int
    var residual: Float64

    def __init__(out self, energy: Float64, var state: StateVector, iterations: Int, residual: Float64):
        self.energy = energy
        self.state = state^
        self.iterations = iterations
        self.residual = residual

    def write_to(self, mut w: Some[Writer]):
        w.write("GroundState(energy=", self.energy, ", lanczos_iters=", self.iterations, ", residual=", self.residual, ")")


def ground_state(h: PauliSum, num_qubits: Int, max_iter: Int = 160, tol: Float64 = 1e-12, seed: UInt64 = 2024) raises -> GroundState:
    """Lowest eigenpair of H by Lanczos with full re-orthogonalisation.

    Memory: (iterations + 3) state vectors - fine up to ~18 qubits. The start
    vector is a seeded random state (deterministic results). Returns the
    Rayleigh-quotient energy, the normalised ground state and the residual
    ||H psi - E psi||.
    """
    if num_qubits > 20:
        raise Error("QM-E807: ground_state() is limited to 20 qubits (Lanczos stores the Krylov basis), got " + String(num_qubits) + ". Hint: use VQE or DMRG-style methods for larger systems.")
    var op = PauliOperator(h, num_qubits)
    var dim = 1 << num_qubits
    var m_max = min(max_iter, dim)
    var rng = Rng(seed)
    var v0 = StateVector(num_qubits)
    for i in range(dim):
        v0.re[i] = rng.uniform() - 0.5
        v0.im[i] = rng.uniform() - 0.5
    sv_scale(v0, 1.0 / sv_norm(v0))
    var basis = List[StateVector]()
    basis.append(v0^)
    var alpha = List[Float64]()
    var beta = List[Float64]()
    var w = StateVector(num_qubits)
    var theta_old = 1e300
    var m = 0
    while m < m_max:
        op.apply(basis[m], w)
        var a = sv_dot(basis[m], w)[0]
        alpha.append(a)
        sv_axpy(w, -a, 0.0, basis[m])
        if m > 0:
            sv_axpy(w, -beta[m - 1], 0.0, basis[m - 1])
        # full re-orthogonalisation (twice is enough - Kahan/Parlett)
        for _ in range(2):
            for k in range(m + 1):
                var c = sv_dot(basis[k], w)
                sv_axpy(w, -c[0], -c[1], basis[k])
        var b = sv_norm(w)
        m += 1
        var done = b < 1e-10 or m >= m_max
        if not done and (m % 4 == 0):
            var th = tridiag_lowest(alpha, beta, m)
            if abs(th - theta_old) < tol * max(1.0, abs(th)):
                done = True
            theta_old = th
        if done:
            break
        beta.append(b)
        var nv = StateVector(num_qubits)
        sv_assign(nv, w)
        sv_scale(nv, 1.0 / b)
        basis.append(nv^)
    # Ritz vector from the projected tridiagonal matrix
    var t = List[Float64](length=m * m, fill=0.0)
    for i in range(m):
        t[i * m + i] = alpha[i]
        if i + 1 < m:
            t[i * m + i + 1] = beta[i]
            t[(i + 1) * m + i] = beta[i]
    var eig = jacobi_eigh(t, m)
    var psi = StateVector(num_qubits)
    sv_zero(psi)
    for k in range(m):
        sv_axpy(psi, eig.vectors[k * m + 0], 0.0, basis[k])
    sv_scale(psi, 1.0 / sv_norm(psi))
    op.apply(psi, w)
    var e = sv_dot(psi, w)[0]
    sv_axpy(w, -e, 0.0, psi)
    var res = sv_norm(w)
    return GroundState(e, psi^, m, res)


def ground_energy(h: PauliSum, num_qubits: Int = -1) raises -> Float64:
    """Exact ground-state energy (Lanczos). num_qubits defaults to h.num_qubits()."""
    var n = num_qubits if num_qubits > 0 else max(h.num_qubits(), 1)
    return ground_state(h, n).energy


def dense_matrix(h: PauliSum, num_qubits: Int) raises -> Tuple[List[Float64], List[Float64]]:
    """Dense row-major (re, im) matrix of H, size 2^n x 2^n (n <= 12)."""
    check_fits(h, num_qubits)
    if num_qubits > 12:
        raise Error("QM-E807: dense_matrix() limited to 12 qubits, got " + String(num_qubits) + ".")
    var dim = 1 << num_qubits
    var re = List[Float64](length=dim * dim, fill=0.0)
    var im = List[Float64](length=dim * dim, fill=0.0)
    for t in h.terms:
        var xm = Int(t.x)
        var zm = Int(t.z)
        var ny = Int(pop_count(t.x & t.z)) & 3
        for i in range(dim):
            var s = t.coeff
            if (pop_count(i & zm) & 1) == 1:
                s = -s
            var j = i ^ xm
            # H[j, i] += s * i^ny
            if ny == 0:
                re[j * dim + i] += s
            elif ny == 1:
                im[j * dim + i] += s
            elif ny == 2:
                re[j * dim + i] -= s
            else:
                im[j * dim + i] -= s
    return (re^, im^)


def exact_spectrum(h: PauliSum, num_qubits: Int) raises -> List[Float64]:
    """All eigenvalues (ascending) by dense Jacobi diagonalisation (n <= 7).

    Complex Hermitian H = A + iB is diagonalised through its real symmetric
    embedding [[A, -B], [B, A]] (each eigenvalue appears twice); purely real
    Hamiltonians skip the embedding.
    """
    if num_qubits > 7:
        raise Error("QM-E807: exact_spectrum() limited to 7 qubits, got " + String(num_qubits) + ". Hint: use ground_state() (Lanczos) for the lowest eigenvalue.")
    var mats = dense_matrix(h, num_qubits)
    var dim = 1 << num_qubits
    var is_real = True
    for v in mats[1]:
        if abs(v) > 1e-15:
            is_real = False
            break
    if is_real:
        return jacobi_eigh(mats[0], dim).values.copy()
    var m = 2 * dim
    var big = List[Float64](length=m * m, fill=0.0)
    for r in range(dim):
        for c in range(dim):
            var a = mats[0][r * dim + c]
            var b = mats[1][r * dim + c]
            big[r * m + c] = a
            big[(r + dim) * m + c + dim] = a
            big[r * m + c + dim] = -b
            big[(r + dim) * m + c] = b
    var ev = jacobi_eigh(big, m).values.copy()
    var out = List[Float64](capacity=dim)
    for k in range(dim):
        out.append(0.5 * (ev[2 * k] + ev[2 * k + 1]))
    return out^


def exact_evolve(h: PauliSum, psi: StateVector, time: Float64, tol: Float64 = 1e-15) raises -> StateVector:
    """exp(-i H t)|psi> by a scaled Taylor series (reference for Trotter studies).

    The time is cut into slices with ||H|| dt <= 0.5 (||H|| bounded by the sum of
    |coefficients|); each slice is summed until the next term is below `tol`.
    """
    var op = PauliOperator(h, psi.num_qubits)
    var norm = 0.0
    for t in h.terms:
        norm += abs(t.coeff)
    var slices = max(1, Int(ceil(norm * abs(time) / 0.5)))
    var dt = time / Float64(slices)
    var acc = psi.copy()
    var term = StateVector(psi.num_qubits)
    var tmp = StateVector(psi.num_qubits)
    for _ in range(slices):
        sv_assign(term, acc)
        var k = 1
        while k < 60:
            op.apply(term, tmp)
            # term <- (-i dt / k) H term
            sv_assign(term, tmp)
            sv_mul_minus_i(term, dt / Float64(k))
            sv_axpy(acc, 1.0, 0.0, term)
            if sv_norm(term) < tol:
                break
            k += 1
    return acc^


# ---------------------------------------------------------------------------
# Subspace projection: sample-based quantum diagonalisation (SQD / SKQD)
# ---------------------------------------------------------------------------
comptime SUBSPACE_MAX_STATES = 4096  # projected_hamiltonian / subspace_ground_energy limit
comptime SUBSPACE_DENSE_MAX = 192  # above this, subspace_ground_energy uses sparse Lanczos


struct _Subspace(Movable):
    var states: List[Int]  # basis states in the caller's order
    var keys: List[Int]  # sorted copy for binary search
    var pos: List[Int]  # pos[k] = caller index of keys[k]

    def __init__(out self, basis_states: List[Int], num_qubits: Int, what: String) raises:
        var m = len(basis_states)
        if m == 0:
            raise Error("QM-E802: " + what + " needs at least one basis state.")
        if m > SUBSPACE_MAX_STATES:
            raise Error("QM-E807: " + what + " is limited to " + String(SUBSPACE_MAX_STATES) + " basis states, got " + String(m) + ". Hint: keep the most frequent bitstrings (configuration recovery) or a symmetry sector.")
        if num_qubits < 1 or num_qubits > 48:
            raise Error("QM-E802: " + what + " needs 1 <= num_qubits <= 48, got " + String(num_qubits) + ".")
        var lim = 1 << num_qubits
        var packed = List[Int](capacity=m)
        for i in range(m):
            var s = basis_states[i]
            if s < 0 or s >= lim:
                raise Error("QM-E802: basis state " + String(s) + " is outside [0, 2^" + String(num_qubits) + ").")
            packed.append(s * 8192 + i)  # m <= 4096 < 2^13
        sort(packed)
        self.states = basis_states.copy()
        self.keys = List[Int](capacity=m)
        self.pos = List[Int](capacity=m)
        for k in range(m):
            var s = packed[k] // 8192
            if k > 0 and s == self.keys[k - 1]:
                raise Error("QM-E802: basis state " + String(s) + " is listed twice. Hint: de-duplicate the sampled bitstrings first.")
            self.keys.append(s)
            self.pos.append(packed[k] % 8192)

    def find(self, s: Int) -> Int:
        """Caller index of basis state s, or -1."""
        var lo = 0
        var hi = len(self.keys) - 1
        while lo <= hi:
            var mid = (lo + hi) // 2
            var v = self.keys[mid]
            if v == s:
                return self.pos[mid]
            if v < s:
                lo = mid + 1
            else:
                hi = mid - 1
        return -1


def _subspace_coo(h: PauliSum, sub: _Subspace, mut rows: List[Int], mut cols: List[Int],
                  mut vre: List[Float64], mut vim: List[Float64]):
    """Non-zero entries <b_i|H|b_j> (one per Pauli term and source state)."""
    for j in range(len(sub.states)):
        var b = sub.states[j]
        for t in h.terms:
            var i = sub.find(b ^ Int(t.x))
            if i < 0:
                continue
            var s = t.coeff
            if (pop_count(b & Int(t.z)) & 1) == 1:
                s = -s
            var ny = Int(pop_count(t.x & t.z)) & 3
            rows.append(i)
            cols.append(j)
            if ny == 0:
                vre.append(s)
                vim.append(0.0)
            elif ny == 1:
                vre.append(0.0)
                vim.append(s)
            elif ny == 2:
                vre.append(-s)
                vim.append(0.0)
            else:
                vre.append(0.0)
                vim.append(-s)


def projected_hamiltonian(h: PauliSum, basis_states: List[Int], num_qubits: Int) raises -> Tuple[List[Float64], List[Float64]]:
    """Dense (re, im) row-major m x m matrix <b_i|H|b_j> on the given basis states.

    Each Pauli string maps a basis state to exactly one basis state with a
    phase, so the cost is O(m * terms * log m) (no 2^n work). Equals
    `dense_matrix(h, n)` restricted to rows/columns `basis_states` (in that
    order). QM-E807 above 4096 states; QM-E802 for duplicates / out-of-range.
    """
    check_fits(h, num_qubits)
    var sub = _Subspace(basis_states, num_qubits, "projected_hamiltonian()")
    var m = len(basis_states)
    var rows = List[Int]()
    var cols = List[Int]()
    var vre = List[Float64]()
    var vim = List[Float64]()
    _subspace_coo(h, sub, rows, cols, vre, vim)
    var re = List[Float64](length=m * m, fill=0.0)
    var im = List[Float64](length=m * m, fill=0.0)
    for k in range(len(rows)):
        re[rows[k] * m + cols[k]] += vre[k]
        im[rows[k] * m + cols[k]] += vim[k]
    return (re^, im^)


def _dense_lowest(re: List[Float64], im: List[Float64], m: Int) -> Float64:
    var is_real = True
    for v in im:
        if abs(v) > 1e-15:
            is_real = False
            break
    if is_real:
        return jacobi_eigh(re, m).values[0]
    var mm = 2 * m
    var big = List[Float64](length=mm * mm, fill=0.0)
    for r in range(m):
        for c in range(m):
            var a = re[r * m + c]
            var b = im[r * m + c]
            big[r * mm + c] = a
            big[(r + m) * mm + c + m] = a
            big[r * mm + c + m] = -b
            big[(r + m) * mm + c] = b
    return jacobi_eigh(big, mm).values[0]


def subspace_ground_energy(h: PauliSum, basis_states: List[Int], num_qubits: Int,
                           max_iter: Int = 300, tol: Float64 = 1e-13) raises -> Float64:
    """Lowest eigenvalue of H projected onto span{|b> : b in basis_states}.

    Variational: >= the true ground energy, equal to it when the span contains
    the ground state (e.g. the full basis or the right symmetry sector).
    m <= 192: dense `jacobi_eigh` (complex H through the 2m x 2m real
    embedding [[A, -B], [B, A]]). Larger m: Lanczos with full
    re-orthogonalisation on the sparse projected matrix (complex arithmetic,
    seeded start vector, converged to `tol` relative), so 4096 states cost
    milliseconds instead of a dense O(m^3) sweep.
    """
    check_fits(h, num_qubits)
    var sub = _Subspace(basis_states, num_qubits, "subspace_ground_energy()")
    var m = len(basis_states)
    var rows = List[Int]()
    var cols = List[Int]()
    var vre = List[Float64]()
    var vim = List[Float64]()
    _subspace_coo(h, sub, rows, cols, vre, vim)
    if m <= SUBSPACE_DENSE_MAX:
        var re = List[Float64](length=m * m, fill=0.0)
        var im = List[Float64](length=m * m, fill=0.0)
        for k in range(len(rows)):
            re[rows[k] * m + cols[k]] += vre[k]
            im[rows[k] * m + cols[k]] += vim[k]
        return _dense_lowest(re, im, m)
    # --- sparse Lanczos (complex Hermitian), full re-orthogonalisation ---
    var nnz = len(rows)
    var rng = Rng(2024)
    var bre = List[List[Float64]]()
    var bim = List[List[Float64]]()
    var x_re = List[Float64](length=m, fill=0.0)
    var x_im = List[Float64](length=m, fill=0.0)
    var nrm = 0.0
    for i in range(m):
        x_re[i] = rng.uniform() - 0.5
        x_im[i] = rng.uniform() - 0.5
        nrm += x_re[i] * x_re[i] + x_im[i] * x_im[i]
    nrm = sqrt(nrm)
    for i in range(m):
        x_re[i] /= nrm
        x_im[i] /= nrm
    bre.append(x_re^)
    bim.append(x_im^)
    var alpha = List[Float64]()
    var beta = List[Float64]()
    var w_re = List[Float64](length=m, fill=0.0)
    var w_im = List[Float64](length=m, fill=0.0)
    var m_max = min(max_iter, m)
    var theta_old = 1e300
    var k = 0
    while k < m_max:
        for i in range(m):
            w_re[i] = 0.0
            w_im[i] = 0.0
        for e in range(nnz):
            var r = rows[e]
            var c = cols[e]
            var hr = vre[e]
            var hi = vim[e]
            var xr = bre[k][c]
            var xi = bim[k][c]
            w_re[r] += hr * xr - hi * xi
            w_im[r] += hr * xi + hi * xr
        var a = 0.0
        for i in range(m):
            a += bre[k][i] * w_re[i] + bim[k][i] * w_im[i]
        alpha.append(a)
        for i in range(m):
            w_re[i] -= a * bre[k][i]
            w_im[i] -= a * bim[k][i]
            if k > 0:
                w_re[i] -= beta[k - 1] * bre[k - 1][i]
                w_im[i] -= beta[k - 1] * bim[k - 1][i]
        for _ in range(2):
            for j in range(k + 1):
                var cr = 0.0
                var ci = 0.0
                for i in range(m):
                    cr += bre[j][i] * w_re[i] + bim[j][i] * w_im[i]
                    ci += bre[j][i] * w_im[i] - bim[j][i] * w_re[i]
                for i in range(m):
                    var ur = bre[j][i]
                    var ui = bim[j][i]
                    w_re[i] -= cr * ur - ci * ui
                    w_im[i] -= cr * ui + ci * ur
        var b = 0.0
        for i in range(m):
            b += w_re[i] * w_re[i] + w_im[i] * w_im[i]
        b = sqrt(b)
        k += 1
        var done = b < 1e-10 or k >= m_max
        if not done and k % 4 == 0:
            var th = tridiag_lowest(alpha, beta, k)
            if abs(th - theta_old) < tol * max(1.0, abs(th)):
                done = True
            theta_old = th
        if done:
            break
        beta.append(b)
        var nr = List[Float64](length=m, fill=0.0)
        var ni = List[Float64](length=m, fill=0.0)
        for i in range(m):
            nr[i] = w_re[i] / b
            ni[i] = w_im[i] / b
        bre.append(nr^)
        bim.append(ni^)
    return tridiag_lowest(alpha, beta, k)
