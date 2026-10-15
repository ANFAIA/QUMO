"""Block algebra for collect-and-resynthesize passes (AI-transpiler front).

Canonical, exact descriptions of the four block kinds that QUMO resynthesizes
(the native, hardware-neutral counterpart of qiskit-ibm-transpiler's
`AICliffordSynthesis` / `AILinearFunctionSynthesis` / `AIPermutationSynthesis`
/ `AIPauliNetworkSynthesis` inputs):

  * `Perm`     - a qubit permutation (SWAP networks),
  * `GF2Mat`   - an invertible linear map over GF(2) (CX / SWAP circuits),
  * `SympTab`  - a signed Aaronson-Gottesman forward tableau (Clifford circuits),
  * `PauliNet` - a list of Pauli rotations pulled back to the block input frame
                 followed by a final Clifford (Pauli-rotation networks),

plus `SubGraph`, the coupling graph induced by a block's physical qubits on
local indices 0..n-1. Synthesizers (`block_synth.mojo`, `policy_synth.mojo`)
take one of these problems and a `SubGraph` and return instructions on local
qubits; the caller verifies a proposal by recomputing the problem object from
the emitted ops and comparing with `==` (exact algebra equality).

Conventions (fixed contract, shared by all AI-transpiler fronts):

  * `Perm.p[i]`: after the circuit, the state of input qubit i sits on qubit
    p[i]. SWAP(a, b) exchanges the positions a and b.
  * `GF2Mat.rows[i]`: bit mask of row i of M, the map x -> M x on basis
    states. CX(c, t) does rows[t] ^= rows[c]; SWAP swaps rows. n <= 64.
  * `SympTab`: forward tableau of a Clifford C. Row i < n is C X_i C^dag, row
    n + i is C Z_i C^dag, each a Hermitian Pauli (-1)^r prod_j P_j with
    P_j = X (x=1, z=0), Z (0, 1), Y (1, 1). n <= 64. Equality with signs is
    equality of the Clifford up to a global phase.
  * `PauliNet`: the block equals prod_k exp(-i theta_k sign_k P_k / 2) (k in
    order, first rotation applied first, P_k in the input frame) followed by
    the Clifford `final`.

Global phase is never part of these objects: use `phase_offset` (dense
statevector, n <= 14) to fix it after an exact replacement.
"""

from std.math import pi, atan2, sqrt
from std.bit import pop_count
from qumo.ir.gates import Op, op_arity, op_inverse_kind, op_is_unitary, matrix_1q, matrix_2q, M2, M4
from qumo.ir.circuit import Instruction
from qumo.ir.param import Param
from qumo.compiler.common import GateSet
from qumo.compiler.kak import cx_cost
from qumo.compiler.target import Target


# ----------------------------------------------------------------------------
# Kinds
# ----------------------------------------------------------------------------
struct BlockKind:
    comptime CLIFFORD = 0
    comptime LINEAR = 1
    comptime PERM = 2
    comptime PAULI = 3


def block_kind_name(kind: Int) -> String:
    if kind == BlockKind.CLIFFORD: return "clifford"
    if kind == BlockKind.LINEAR: return "linear"
    if kind == BlockKind.PERM: return "perm"
    if kind == BlockKind.PAULI: return "pauli"
    return "unknown"


@always_inline
def _bit(j: Int) -> UInt64:
    return UInt64(1) << UInt64(j)


@always_inline
def _has(m: UInt64, j: Int) -> Bool:
    return ((m >> UInt64(j)) & 1) != 0


# ----------------------------------------------------------------------------
# SubGraph
# ----------------------------------------------------------------------------
struct SubGraph(Copyable, Writable):
    """Coupling graph on local qubits 0..n-1 (undirected, simple)."""

    var n: Int
    var edges: List[Int]  # flat local pairs (a, b), a < b, sorted
    var dist: List[Int]  # n*n BFS hop counts, -1 if disconnected

    def __init__(out self, n: Int, edges: List[Int]):
        """Build from flat local pairs (any order; duplicates / self loops dropped)."""
        self.n = n
        var seen = List[Bool](length=max(n * n, 1), fill=False)
        var pairs = List[Int]()
        for i in range(len(edges) // 2):
            var a = edges[2 * i]
            var b = edges[2 * i + 1]
            if a == b or a < 0 or b < 0 or a >= n or b >= n:
                continue
            var lo = min(a, b)
            var hi = max(a, b)
            if seen[lo * n + hi]:
                continue
            seen[lo * n + hi] = True
            pairs.append(lo * n + hi)
        sort(pairs)
        self.edges = List[Int](capacity=2 * len(pairs))
        for k in pairs:
            self.edges.append(k // n)
            self.edges.append(k % n)
        self.dist = List[Int](length=n * n, fill=-1)
        var queue = List[Int](length=max(n, 1), fill=0)
        for s in range(n):
            self.dist[s * n + s] = 0
            var head = 0
            var tail = 1
            queue[0] = s
            while head < tail:
                var u = queue[head]
                head += 1
                var du = self.dist[s * n + u]
                for v in range(n):
                    if self.dist[s * n + v] < 0 and seen[min(u, v) * n + max(u, v)] and u != v:
                        self.dist[s * n + v] = du + 1
                        queue[tail] = v
                        tail += 1

    def __init__(out self, *, copy: Self):
        self.n = copy.n
        self.edges = copy.edges.copy()
        self.dist = copy.dist.copy()

    @staticmethod
    def complete(n: Int) -> SubGraph:
        var e = List[Int]()
        for a in range(n):
            for b in range(a + 1, n):
                e.append(a)
                e.append(b)
        return SubGraph(n, e)

    @staticmethod
    def induced(target: Target, phys: List[Int]) -> SubGraph:
        """Subgraph of `target` induced by physical qubits `phys` (local i = phys[i])."""
        var n = len(phys)
        if target.full:
            return SubGraph.complete(n)
        var e = List[Int]()
        for a in range(n):
            for b in range(a + 1, n):
                if target.are_connected(phys[a], phys[b]):
                    e.append(a)
                    e.append(b)
        return SubGraph(n, e)

    def num_edges(self) -> Int:
        return len(self.edges) // 2

    def has_edge(self, a: Int, b: Int) -> Bool:
        if a < 0 or b < 0 or a >= self.n or b >= self.n or a == b:
            return False
        return self.dist[a * self.n + b] == 1

    def distance(self, a: Int, b: Int) -> Int:
        return self.dist[a * self.n + b]

    def connected(self) -> Bool:
        for v in range(self.n):
            if self.dist[v] < 0:
                return False
        return True

    def neighbors(self, a: Int) -> List[Int]:
        var out = List[Int]()
        for b in range(self.n):
            if self.dist[a * self.n + b] == 1:
                out.append(b)
        return out^

    def is_complete(self) -> Bool:
        return self.num_edges() == self.n * (self.n - 1) // 2

    def write_to(self, mut w: Some[Writer]):
        w.write("SubGraph(n=", self.n, ", edges=[")
        for i in range(len(self.edges) // 2):
            if i > 0:
                w.write(", ")
            w.write(self.edges[2 * i], "-", self.edges[2 * i + 1])
        w.write("])")


# ----------------------------------------------------------------------------
# Instruction helpers
# ----------------------------------------------------------------------------
@always_inline
def g1(kind: Int, q: Int) -> Instruction:
    return Instruction.make(kind, q)


@always_inline
def g2(kind: Int, a: Int, b: Int) -> Instruction:
    return Instruction.make(kind, a, b)


def is_rotation_kind(k: Int) -> Bool:
    return k == Op.RX or k == Op.RY or k == Op.RZ or k == Op.RZZ or k == Op.RXX or k == Op.RYY


def inverse_inst(g: Instruction) -> Instruction:
    """Adjoint of a gate (rotations negate their angle)."""
    var o = g
    var k = g.kind
    if k == Op.RX or k == Op.RY or k == Op.RZ or k == Op.P or k == Op.RZZ or k == Op.RXX or k == Op.RYY or k == Op.CP or k == Op.CRX or k == Op.CRY or k == Op.CRZ:
        o.p0 = -g.p0
        return o
    o.kind = op_inverse_kind(k)
    return o


def inverse_ops(ops: List[Instruction]) -> List[Instruction]:
    """Adjoint circuit: reversed order, every gate inverted."""
    var out = List[Instruction](capacity=len(ops))
    var i = len(ops) - 1
    while i >= 0:
        out.append(inverse_inst(ops[i]))
        i -= 1
    return out^


def local_ops(ops: List[Instruction], qubits: List[Int]) -> List[Instruction]:
    """Relabel wire indices to local indices (local i = position of the wire in
    `qubits`). A qubit outside `qubits` maps to -1 (the caller's bug; the
    verification of the block then fails)."""
    var mx = 0
    for q in qubits:
        mx = max(mx, q)
    var lut = List[Int](length=mx + 1, fill=-1)
    for i in range(len(qubits)):
        lut[qubits[i]] = i
    var out = List[Instruction](capacity=len(ops))
    for g in ops:
        var o = g
        var a = op_arity(g.kind)
        if a >= 1:
            o.q0 = lut[g.q0] if g.q0 >= 0 and g.q0 <= mx else -1
        if a >= 2:
            o.q1 = lut[g.q1] if g.q1 >= 0 and g.q1 <= mx else -1
        if a >= 3:
            o.q2 = lut[g.q2] if g.q2 >= 0 and g.q2 <= mx else -1
        out.append(o)
    return out^


def global_ops(ops: List[Instruction], qubits: List[Int]) -> List[Instruction]:
    """Relabel local indices back to wires: local i -> qubits[i]."""
    var out = List[Instruction](capacity=len(ops))
    for g in ops:
        var o = g
        var a = op_arity(g.kind)
        if a >= 1:
            o.q0 = qubits[g.q0]
        if a >= 2:
            o.q1 = qubits[g.q1]
        if a >= 3:
            o.q2 = qubits[g.q2]
        out.append(o)
    return out^


# ----------------------------------------------------------------------------
# Costs
# ----------------------------------------------------------------------------
def ops_cost(ops: List[Instruction], gs: GateSet) -> Int:
    """Two-qubit cost in native-entangler (CX-unit) cost of the target gate set
    (`kak.cx_cost`): native entangler 1, CX-class 1 per CX unit, SWAP 3 unless
    native, iSWAP-family targets 2 per CX. Vendor neutral by construction."""
    var c = 0
    for g in ops:
        if op_arity(g.kind) == 2 and op_is_unitary(g.kind):
            c += cx_cost(g.kind, gs)
    return c


def ops_cx_count(ops: List[Instruction]) -> Int:
    """Two-qubit cost in plain CX units (SWAP = 3, other 2q gates = 1)."""
    var c = 0
    for g in ops:
        if op_arity(g.kind) == 2 and op_is_unitary(g.kind):
            c += 3 if g.kind == Op.SWAP else 1
    return c


def ops_depth2q(ops: List[Instruction], n: Int) -> Int:
    """Number of layers of two-qubit gates (1q gates ignored)."""
    var mx = n
    for g in ops:
        var a = op_arity(g.kind)
        if a >= 1: mx = max(mx, g.q0 + 1)
        if a >= 2: mx = max(mx, g.q1 + 1)
    var lv = List[Int](length=mx, fill=0)
    var d = 0
    for g in ops:
        if op_arity(g.kind) == 2 and op_is_unitary(g.kind):
            var l = max(lv[g.q0], lv[g.q1]) + 1
            lv[g.q0] = l
            lv[g.q1] = l
            d = max(d, l)
    return d


def ops_on_edges(ops: List[Instruction], g: SubGraph) -> Bool:
    """True when every multi-qubit gate acts on an edge of `g`."""
    for o in ops:
        var a = op_arity(o.kind)
        if a >= 3:
            return False
        if a == 2 and not g.has_edge(o.q0, o.q1):
            return False
        if a >= 1 and (o.q0 < 0 or o.q0 >= g.n):
            return False
    return True


# ----------------------------------------------------------------------------
# Perm
# ----------------------------------------------------------------------------
struct Perm(Copyable, Writable):
    """Qubit permutation: after the circuit, the state of input qubit i is on qubit p[i]."""

    var p: List[Int]

    def __init__(out self, p: List[Int]):
        self.p = p.copy()

    def __init__(out self, *, copy: Self):
        self.p = copy.p.copy()

    @staticmethod
    def identity(n: Int) -> Perm:
        var p = List[Int](capacity=n)
        for i in range(n):
            p.append(i)
        return Perm(p)

    @staticmethod
    def from_ops(ops: List[Instruction], n: Int) raises -> Perm:
        """Permutation of a SWAP circuit (identity / barrier ops are skipped)."""
        var r = Perm.identity(n)
        for g in ops:
            if g.kind == Op.SWAP:
                if g.q0 < 0 or g.q1 < 0 or g.q0 >= n or g.q1 >= n:
                    raise Error("QM-E618: SWAP on qubit outside the " + String(n) + "-qubit block. Hint: relabel the block with local_ops first.")
                r.apply_swap(g.q0, g.q1)
            elif g.kind == Op.I or g.kind == Op.BARRIER or g.kind == Op.TICK:
                continue
            else:
                raise Error("QM-E618: gate '" + String(g) + "' is not a permutation gate. Hint: permutation blocks may only contain SWAP (and identity) gates.")
        return r^

    def size(self) -> Int:
        return len(self.p)

    def apply_swap(mut self, a: Int, b: Int):
        """Append SWAP(a, b): the states sitting on a and b exchange places."""
        for i in range(len(self.p)):
            if self.p[i] == a:
                self.p[i] = b
            elif self.p[i] == b:
                self.p[i] = a

    def compose(self, other: Perm) -> Perm:
        """`self` followed by `other`."""
        var out = List[Int](capacity=len(self.p))
        for i in range(len(self.p)):
            out.append(other.p[self.p[i]])
        return Perm(out)

    def inverse(self) -> Perm:
        var out = List[Int](length=len(self.p), fill=0)
        for i in range(len(self.p)):
            out[self.p[i]] = i
        return Perm(out)

    def is_identity(self) -> Bool:
        for i in range(len(self.p)):
            if self.p[i] != i:
                return False
        return True

    def is_valid(self) -> Bool:
        var n = len(self.p)
        var seen = List[Bool](length=n, fill=False)
        for v in self.p:
            if v < 0 or v >= n or seen[v]:
                return False
            seen[v] = True
        return True

    def __eq__(self, o: Self) -> Bool:
        if len(self.p) != len(o.p):
            return False
        for i in range(len(self.p)):
            if self.p[i] != o.p[i]:
                return False
        return True

    def __ne__(self, o: Self) -> Bool:
        return not self.__eq__(o)

    def write_to(self, mut w: Some[Writer]):
        w.write("Perm[")
        for i in range(len(self.p)):
            if i > 0:
                w.write(",")
            w.write(self.p[i])
        w.write("]")


# ----------------------------------------------------------------------------
# GF2Mat
# ----------------------------------------------------------------------------
struct GF2Mat(Copyable, Writable):
    """Invertible n x n matrix over GF(2); row i is a bit mask (bit j = M[i][j])."""

    var n: Int
    var rows: List[UInt64]

    def __init__(out self, n: Int, rows: List[UInt64]):
        self.n = n
        self.rows = rows.copy()

    def __init__(out self, *, copy: Self):
        self.n = copy.n
        self.rows = copy.rows.copy()

    @staticmethod
    def identity(n: Int) -> GF2Mat:
        var r = List[UInt64](capacity=n)
        for i in range(n):
            r.append(_bit(i))
        return GF2Mat(n, r)

    @staticmethod
    def from_perm(p: Perm) -> GF2Mat:
        """Linear map of a permutation: output bit p[i] = input bit i."""
        var n = p.size()
        var r = List[UInt64](length=n, fill=0)
        for i in range(n):
            r[p.p[i]] = _bit(i)
        return GF2Mat(n, r)

    @staticmethod
    def from_ops(ops: List[Instruction], n: Int) raises -> GF2Mat:
        """Linear map of a CX / SWAP circuit (identity / barrier ops are skipped)."""
        if n > 64:
            raise Error("QM-E618: linear block of " + String(n) + " qubits exceeds 64. Hint: collect blocks of at most 64 qubits.")
        var m = GF2Mat.identity(n)
        for g in ops:
            if g.kind == Op.CX:
                if g.q0 < 0 or g.q1 < 0 or g.q0 >= n or g.q1 >= n:
                    raise Error("QM-E618: CX on qubit outside the " + String(n) + "-qubit block. Hint: relabel the block with local_ops first.")
                m.apply_cx(g.q0, g.q1)
            elif g.kind == Op.SWAP:
                if g.q0 < 0 or g.q1 < 0 or g.q0 >= n or g.q1 >= n:
                    raise Error("QM-E618: SWAP on qubit outside the " + String(n) + "-qubit block. Hint: relabel the block with local_ops first.")
                m.apply_swap(g.q0, g.q1)
            elif g.kind == Op.I or g.kind == Op.BARRIER or g.kind == Op.TICK:
                continue
            else:
                raise Error("QM-E618: gate '" + String(g) + "' is not a linear (CX / SWAP) gate. Hint: linear-function blocks may only contain CX and SWAP.")
        return m^

    @always_inline
    def apply_cx(mut self, c: Int, t: Int):
        self.rows[t] ^= self.rows[c]

    @always_inline
    def apply_swap(mut self, a: Int, b: Int):
        var t = self.rows[a]
        self.rows[a] = self.rows[b]
        self.rows[b] = t

    def get(self, i: Int, j: Int) -> Bool:
        return _has(self.rows[i], j)

    def compose(self, other: GF2Mat) -> GF2Mat:
        """`self` followed by `other` (the matrix product other * self)."""
        var out = List[UInt64](length=self.n, fill=0)
        for i in range(self.n):
            var acc: UInt64 = 0
            var r = other.rows[i]
            for j in range(self.n):
                if _has(r, j):
                    acc ^= self.rows[j]
            out[i] = acc
        return GF2Mat(self.n, out)

    def transpose(self) -> GF2Mat:
        var out = List[UInt64](length=self.n, fill=0)
        for i in range(self.n):
            for j in range(self.n):
                if _has(self.rows[i], j):
                    out[j] |= _bit(i)
        return GF2Mat(self.n, out)

    def inverse(self) raises -> GF2Mat:
        """Gauss-Jordan inverse over GF(2) (QM-E618 when singular)."""
        var a = self.rows.copy()
        var b = GF2Mat.identity(self.n).rows.copy()
        for col in range(self.n):
            var piv = -1
            for r in range(col, self.n):
                if _has(a[r], col):
                    piv = r
                    break
            if piv < 0:
                raise Error("QM-E618: GF(2) matrix is singular. Hint: a linear-function block must be invertible (built from CX / SWAP).")
            if piv != col:
                var t = a[piv]; a[piv] = a[col]; a[col] = t
                var u = b[piv]; b[piv] = b[col]; b[col] = u
            for r in range(self.n):
                if r != col and _has(a[r], col):
                    a[r] ^= a[col]
                    b[r] ^= b[col]
        return GF2Mat(self.n, b)

    def is_identity(self) -> Bool:
        for i in range(self.n):
            if self.rows[i] != _bit(i):
                return False
        return True

    def is_permutation(self) -> Bool:
        for i in range(self.n):
            if pop_count(self.rows[i]) != 1:
                return False
        var acc: UInt64 = 0
        for i in range(self.n):
            acc |= self.rows[i]
        return Int(pop_count(acc)) == self.n

    def to_perm(self) -> Perm:
        """Permutation of a permutation matrix (call `is_permutation` first)."""
        var p = List[Int](length=self.n, fill=0)
        for r in range(self.n):
            for j in range(self.n):
                if _has(self.rows[r], j):
                    p[j] = r
        return Perm(p)

    def __eq__(self, o: Self) -> Bool:
        if self.n != o.n:
            return False
        for i in range(self.n):
            if self.rows[i] != o.rows[i]:
                return False
        return True

    def __ne__(self, o: Self) -> Bool:
        return not self.__eq__(o)

    def write_to(self, mut w: Some[Writer]):
        w.write("GF2Mat(")
        for i in range(self.n):
            if i > 0:
                w.write("/")
            for j in range(self.n):
                w.write("1" if _has(self.rows[i], j) else "0")
        w.write(")")


# ----------------------------------------------------------------------------
# Pauli helpers (Hermitian form (x, z, r) <-> XZ form i^e X^x Z^z)
# ----------------------------------------------------------------------------
@always_inline
def herm_to_e(x: UInt64, z: UInt64, r: UInt8) -> Int:
    """(-1)^r prod P_j  ==  i^e X^x Z^z  with e = 2r + |x & z|  (Y = i X Z)."""
    return (2 * Int(r) + Int(pop_count(x & z))) & 3


@always_inline
def e_to_herm(x: UInt64, z: UInt64, e: Int) -> UInt8:
    """Inverse of herm_to_e; e - |x&z| is even for a Hermitian Pauli."""
    var d = (e - Int(pop_count(x & z))) & 3
    return UInt8(d >> 1)


@always_inline
def xz_mul(mut ax: UInt64, mut az: UInt64, mut ae: Int, bx: UInt64, bz: UInt64, be: Int):
    """a <- a . b in XZ form: X^x1 Z^z1 X^x2 Z^z2 = (-1)^{|z1 & x2|} X^{x1^x2} Z^{z1^z2}."""
    ae = (ae + be + 2 * Int(pop_count(az & bx))) & 3
    ax ^= bx
    az ^= bz


@always_inline
def paulis_anticommute(x1: UInt64, z1: UInt64, x2: UInt64, z2: UInt64) -> Bool:
    return ((pop_count(x1 & z2) + pop_count(z1 & x2)) & 1) != 0


# Per-row Clifford conjugation rules (Aaronson-Gottesman, P -> G P G^dag).
@always_inline
def _row_h(mut x: UInt64, mut z: UInt64, mut r: UInt8, a: Int):
    var xa = (x >> UInt64(a)) & 1
    var za = (z >> UInt64(a)) & 1
    r ^= UInt8(xa & za)
    var m = _bit(a)
    x = (x & ~m) | (za << UInt64(a))
    z = (z & ~m) | (xa << UInt64(a))


@always_inline
def _row_s(mut x: UInt64, mut z: UInt64, mut r: UInt8, a: Int):
    var xa = (x >> UInt64(a)) & 1
    var za = (z >> UInt64(a)) & 1
    r ^= UInt8(xa & za)
    z ^= xa << UInt64(a)


@always_inline
def _row_sdg(mut x: UInt64, mut z: UInt64, mut r: UInt8, a: Int):
    var xa = (x >> UInt64(a)) & 1
    var za = (z >> UInt64(a)) & 1
    r ^= UInt8(xa & (za ^ 1))
    z ^= xa << UInt64(a)


@always_inline
def _row_sx(mut x: UInt64, mut z: UInt64, mut r: UInt8, a: Int):
    var xa = (x >> UInt64(a)) & 1
    var za = (z >> UInt64(a)) & 1
    r ^= UInt8(za & (xa ^ 1))
    x ^= za << UInt64(a)


@always_inline
def _row_sxdg(mut x: UInt64, mut z: UInt64, mut r: UInt8, a: Int):
    var xa = (x >> UInt64(a)) & 1
    var za = (z >> UInt64(a)) & 1
    r ^= UInt8(xa & za)
    x ^= za << UInt64(a)


@always_inline
def _row_cx(mut x: UInt64, mut z: UInt64, mut r: UInt8, c: Int, t: Int):
    var xc = (x >> UInt64(c)) & 1
    var zc = (z >> UInt64(c)) & 1
    var xt = (x >> UInt64(t)) & 1
    var zt = (z >> UInt64(t)) & 1
    r ^= UInt8(xc & zt & (xt ^ zc ^ 1))
    x ^= xc << UInt64(t)
    z ^= zt << UInt64(c)


@always_inline
def _row_swap(mut x: UInt64, mut z: UInt64, a: Int, b: Int):
    var xa = (x >> UInt64(a)) & 1
    var xb = (x >> UInt64(b)) & 1
    var za = (z >> UInt64(a)) & 1
    var zb = (z >> UInt64(b)) & 1
    var ma = _bit(a)
    var mb = _bit(b)
    x = (x & ~(ma | mb)) | (xa << UInt64(b)) | (xb << UInt64(a))
    z = (z & ~(ma | mb)) | (za << UInt64(b)) | (zb << UInt64(a))


def _quarter_turns(g: Instruction) raises -> Int:
    """Rotation angle as a multiple k (mod 4) of pi/2 (QM-E701 otherwise)."""
    if g.p0.is_symbolic():
        raise Error("QM-E701: symbolic rotation '" + String(g) + "' is not Clifford. Hint: bind parameters or keep the gate out of Clifford blocks.")
    var r = g.p0.value() / (pi / 2)
    var k = round(r)
    if abs(r - k) > 1e-9:
        raise Error("QM-E701: gate '" + String(g) + "' is not Clifford. Hint: only rotations by multiples of pi/2 are Clifford.")
    return ((Int(k) % 4) + 4) % 4


def clifford_expand(g: Instruction, mut out: List[Instruction]) raises:
    """Rewrite a Clifford gate into {H, S, SDG, SX, SXDG, X, Y, Z, CX, SWAP}
    (exact up to global phase). ISWAP / ISWAPDG / ECR are handled numerically
    by `SympTab.apply` and are not expanded here (QM-E701)."""
    var k = g.kind
    var a = g.q0
    var b = g.q1
    if k == Op.H or k == Op.S or k == Op.SDG or k == Op.SX or k == Op.SXDG or k == Op.X or k == Op.Y or k == Op.Z or k == Op.CX or k == Op.SWAP:
        out.append(g)
        return
    if k == Op.I:
        return
    if k == Op.CZ:
        out.append(g1(Op.H, b)); out.append(g2(Op.CX, a, b)); out.append(g1(Op.H, b))
        return
    if k == Op.CY:
        out.append(g1(Op.SDG, b)); out.append(g2(Op.CX, a, b)); out.append(g1(Op.S, b))
        return
    if k == Op.RZ or k == Op.P:
        var t = _quarter_turns(g)
        if t == 1: out.append(g1(Op.S, a))
        elif t == 2: out.append(g1(Op.Z, a))
        elif t == 3: out.append(g1(Op.SDG, a))
        return
    if k == Op.RX:
        var t = _quarter_turns(g)
        if t == 1: out.append(g1(Op.SX, a))
        elif t == 2: out.append(g1(Op.X, a))
        elif t == 3: out.append(g1(Op.SXDG, a))
        return
    if k == Op.RY:
        var t = _quarter_turns(g)
        if t == 1:
            out.append(g1(Op.Z, a)); out.append(g1(Op.H, a))
        elif t == 2:
            out.append(g1(Op.Y, a))
        elif t == 3:
            out.append(g1(Op.H, a)); out.append(g1(Op.Z, a))
        return
    if k == Op.RZZ or k == Op.RXX or k == Op.RYY:
        var t = _quarter_turns(g)
        if t == 0:
            return
        var pre = List[Instruction]()
        var post = List[Instruction]()
        if k == Op.RXX:
            pre.append(g1(Op.H, a)); pre.append(g1(Op.H, b))
            post.append(g1(Op.H, a)); post.append(g1(Op.H, b))
        elif k == Op.RYY:
            # Y = S X S^dag and X = H Z H:  Y = (S H) Z (S H)^dag
            pre.append(g1(Op.SDG, a)); pre.append(g1(Op.SDG, b))
            pre.append(g1(Op.H, a)); pre.append(g1(Op.H, b))
            post.append(g1(Op.H, a)); post.append(g1(Op.H, b))
            post.append(g1(Op.S, a)); post.append(g1(Op.S, b))
        for o in pre:
            out.append(o)
        if t == 2:
            out.append(g1(Op.Z, a)); out.append(g1(Op.Z, b))
        else:
            out.append(g2(Op.CX, a, b))
            out.append(g1(Op.S if t == 1 else Op.SDG, b))
            out.append(g2(Op.CX, a, b))
        for o in post:
            out.append(o)
        return
    raise Error("QM-E701: gate '" + String(g) + "' is not a supported Clifford gate. Hint: Clifford blocks accept H, S, SDG, SX, SXDG, X, Y, Z, CX, CY, CZ, SWAP, ISWAP, ECR and rotations by multiples of pi/2.")


struct _LocalImages(Copyable):
    """XZ-form images of X_q0, Z_q0, X_q1, Z_q1 under a 2q Clifford (numeric)."""
    var x: List[UInt64]
    var z: List[UInt64]
    var e: List[Int]

    def __init__(out self):
        self.x = List[UInt64](length=4, fill=0)
        self.z = List[UInt64](length=4, fill=0)
        self.e = List[Int](length=4, fill=0)


def _pauli_m4(x: Int, z: Int) -> M4:
    """X^x Z^z on two qubits (bit 0 = q0) as a 4x4 matrix (index b0 + 2 b1)."""
    var m = M4()
    for col in range(4):
        # Z^z acts first: phase (-1)^{popcount(z & col)}; then X^x flips bits.
        var ph = 1.0 if (pop_count(UInt64(z & col)) & 1) == 0 else -1.0
        var row = col ^ x
        m.set(row, col, ph, 0.0)
    return m^


def _m4_dag(a: M4) -> M4:
    var o = M4()
    for r in range(4):
        for c in range(4):
            o.set(r, c, a.re[c * 4 + r], -a.im[c * 4 + r])
    return o^


def _numeric_images(g: Instruction) raises -> _LocalImages:
    """Images U P U^dag of the four local generators of a 2q Clifford gate,
    found by Pauli decomposition of the 4x4 matrix (ISWAP, ISWAPDG, ECR, ...)."""
    var u = matrix_2q(g.kind, g.p0.value())
    var ud = _m4_dag(u)
    var out = _LocalImages()
    var gens: List[Int] = [1, 0, 0, 1, 2, 0, 0, 2]  # (x, z) for X0, Z0, X1, Z1
    for k in range(4):
        var p = _pauli_m4(gens[2 * k], gens[2 * k + 1])
        var m = u @ p @ ud
        var found = False
        for qx in range(4):
            for qz in range(4):
                var q = _pauli_m4(qx, qz)
                # c = tr(Q^dag M) / 4
                var cr = 0.0
                var ci = 0.0
                for r in range(4):
                    for c in range(4):
                        # (Q^dag)[r][c] = conj(Q[c][r]); Q is real
                        var qv = q.re[c * 4 + r]
                        cr += qv * m.re[c * 4 + r]
                        ci += qv * m.im[c * 4 + r]
                cr /= 4.0
                ci /= 4.0
                if cr * cr + ci * ci > 0.99:
                    var e: Int
                    if abs(cr - 1.0) < 1e-6: e = 0
                    elif abs(ci - 1.0) < 1e-6: e = 1
                    elif abs(cr + 1.0) < 1e-6: e = 2
                    elif abs(ci + 1.0) < 1e-6: e = 3
                    else:
                        raise Error("QM-E701: gate '" + String(g) + "' is not Clifford. Hint: only Clifford gates may enter a tableau.")
                    var gx: UInt64 = 0
                    var gz: UInt64 = 0
                    if (qx & 1) != 0: gx |= _bit(g.q0)
                    if (qx & 2) != 0: gx |= _bit(g.q1)
                    if (qz & 1) != 0: gz |= _bit(g.q0)
                    if (qz & 2) != 0: gz |= _bit(g.q1)
                    out.x[k] = gx
                    out.z[k] = gz
                    out.e[k] = e
                    found = True
                    break
            if found:
                break
        if not found:
            raise Error("QM-E701: gate '" + String(g) + "' is not Clifford. Hint: only Clifford gates may enter a tableau.")
    return out^


# ----------------------------------------------------------------------------
# SympTab
# ----------------------------------------------------------------------------
struct SympTab(Copyable, Writable):
    """Signed forward stabilizer tableau of a Clifford C (n <= 64).

    Row i < n is C X_i C^dag and row n + i is C Z_i C^dag, as Hermitian Paulis
    (x mask, z mask, sign bit r)."""

    var n: Int
    var x: List[UInt64]
    var z: List[UInt64]
    var r: List[UInt8]

    def __init__(out self, n: Int):
        """Identity tableau."""
        self.n = n
        self.x = List[UInt64](length=2 * n, fill=0)
        self.z = List[UInt64](length=2 * n, fill=0)
        self.r = List[UInt8](length=2 * n, fill=0)
        for i in range(n):
            self.x[i] = _bit(i)
            self.z[n + i] = _bit(i)

    def __init__(out self, *, copy: Self):
        self.n = copy.n
        self.x = copy.x.copy()
        self.z = copy.z.copy()
        self.r = copy.r.copy()

    @staticmethod
    def identity(n: Int) -> SympTab:
        return SympTab(n)

    @staticmethod
    def from_ops(ops: List[Instruction], n: Int) raises -> SympTab:
        """Tableau of a Clifford circuit (QM-E701 on a non-Clifford gate)."""
        if n > 64:
            raise Error("QM-E618: Clifford block of " + String(n) + " qubits exceeds 64. Hint: collect blocks of at most 64 qubits.")
        var t = SympTab(n)
        for g in ops:
            t.apply(g)
        return t^

    @staticmethod
    def from_linear(m: GF2Mat) raises -> SympTab:
        """Tableau of the CX circuit implementing x -> M x:
        C X_i C^dag = X^{column i of M}, C Z_i C^dag = Z^{row i of M^-1}."""
        var n = m.n
        var t = SympTab(n)
        var inv = m.inverse()
        for i in range(n):
            var col: UInt64 = 0
            for r in range(n):
                if _has(m.rows[r], i):
                    col |= _bit(r)
            t.x[i] = col
            t.z[i] = 0
            t.x[n + i] = 0
            t.z[n + i] = inv.rows[i]
        return t^

    def _check_q(self, g: Instruction) raises:
        var a = op_arity(g.kind)
        if (a >= 1 and (g.q0 < 0 or g.q0 >= self.n)) or (a >= 2 and (g.q1 < 0 or g.q1 >= self.n)):
            raise Error("QM-E618: gate '" + String(g) + "' acts outside the " + String(self.n) + "-qubit block. Hint: relabel the block with local_ops first.")
        if a >= 3:
            raise Error("QM-E701: 3-qubit gate '" + String(g) + "' in a Clifford block. Hint: run unroll3q first.")

    def apply(mut self, g: Instruction) raises:
        """Append gate g after C: every row P -> g P g^dag."""
        var k = g.kind
        if k == Op.BARRIER or k == Op.TICK or k == Op.I:
            return
        self._check_q(g)
        var m = 2 * self.n
        if k == Op.H:
            for i in range(m): _row_h(self.x[i], self.z[i], self.r[i], g.q0)
        elif k == Op.S:
            for i in range(m): _row_s(self.x[i], self.z[i], self.r[i], g.q0)
        elif k == Op.SDG:
            for i in range(m): _row_sdg(self.x[i], self.z[i], self.r[i], g.q0)
        elif k == Op.SX:
            for i in range(m): _row_sx(self.x[i], self.z[i], self.r[i], g.q0)
        elif k == Op.SXDG:
            for i in range(m): _row_sxdg(self.x[i], self.z[i], self.r[i], g.q0)
        elif k == Op.X:
            for i in range(m): self.r[i] ^= UInt8((self.z[i] >> UInt64(g.q0)) & 1)
        elif k == Op.Z:
            for i in range(m): self.r[i] ^= UInt8((self.x[i] >> UInt64(g.q0)) & 1)
        elif k == Op.Y:
            for i in range(m): self.r[i] ^= UInt8(((self.x[i] ^ self.z[i]) >> UInt64(g.q0)) & 1)
        elif k == Op.CX:
            for i in range(m): _row_cx(self.x[i], self.z[i], self.r[i], g.q0, g.q1)
        elif k == Op.SWAP:
            for i in range(m): _row_swap(self.x[i], self.z[i], g.q0, g.q1)
        elif k == Op.ISWAP or k == Op.ISWAPDG or k == Op.ECR:
            var im = _numeric_images(g)
            self._apply_images(g.q0, g.q1, im)
        else:
            var seq = List[Instruction]()
            clifford_expand(g, seq)
            for o in seq:
                self.apply(o)

    def _apply_images(mut self, a: Int, b: Int, im: _LocalImages):
        var mask = _bit(a) | _bit(b)
        for i in range(2 * self.n):
            var x = self.x[i]
            var z = self.z[i]
            var e = herm_to_e(x, z, self.r[i])
            # X^x Z^z = (X^{x_o} Z^{z_o}) (X^{x_g} Z^{z_g}); outer part is unchanged.
            var ox = x & ~mask
            var oz = z & ~mask
            var acc_x = ox
            var acc_z = oz
            var acc_e = e
            if _has(x, a): xz_mul(acc_x, acc_z, acc_e, im.x[0], im.z[0], im.e[0])
            if _has(x, b): xz_mul(acc_x, acc_z, acc_e, im.x[2], im.z[2], im.e[2])
            if _has(z, a): xz_mul(acc_x, acc_z, acc_e, im.x[1], im.z[1], im.e[1])
            if _has(z, b): xz_mul(acc_x, acc_z, acc_e, im.x[3], im.z[3], im.e[3])
            self.x[i] = acc_x
            self.z[i] = acc_z
            self.r[i] = e_to_herm(acc_x, acc_z, acc_e)

    def apply_ops(mut self, ops: List[Instruction]) raises:
        for g in ops:
            self.apply(g)

    def image(self, px: UInt64, pz: UInt64, pr: UInt8, mut ox: UInt64, mut oz: UInt64, mut orr: UInt8):
        """C P C^dag for a Hermitian Pauli P = (px, pz, pr)."""
        var e = herm_to_e(px, pz, pr)
        var ax: UInt64 = 0
        var az: UInt64 = 0
        var ae = e
        for j in range(self.n):
            if _has(px, j):
                xz_mul(ax, az, ae, self.x[j], self.z[j], herm_to_e(self.x[j], self.z[j], self.r[j]))
        for j in range(self.n):
            if _has(pz, j):
                var row = self.n + j
                xz_mul(ax, az, ae, self.x[row], self.z[row], herm_to_e(self.x[row], self.z[row], self.r[row]))
        ox = ax
        oz = az
        orr = e_to_herm(ax, az, ae)

    def prepend(mut self, g: Instruction) raises:
        """Prepend gate g before C (the tableau of C . g)."""
        var k = g.kind
        if k == Op.BARRIER or k == Op.TICK or k == Op.I:
            return
        self._check_q(g)
        var loc = SympTab(self.n)
        loc.apply(g)
        var qs = List[Int]()
        qs.append(g.q0)
        if op_arity(k) == 2:
            qs.append(g.q1)
        var nx = List[UInt64]()
        var nz = List[UInt64]()
        var nr = List[UInt8]()
        var rows = List[Int]()
        for q in qs:
            rows.append(q)
            rows.append(self.n + q)
        for row in rows:
            var ox: UInt64 = 0
            var oz: UInt64 = 0
            var orr: UInt8 = 0
            self.image(loc.x[row], loc.z[row], loc.r[row], ox, oz, orr)
            nx.append(ox)
            nz.append(oz)
            nr.append(orr)
        for i in range(len(rows)):
            self.x[rows[i]] = nx[i]
            self.z[rows[i]] = nz[i]
            self.r[rows[i]] = nr[i]

    def compose(self, other: SympTab) -> SympTab:
        """`self` followed by `other` (the Clifford other . self)."""
        var out = SympTab(self.n)
        for i in range(2 * self.n):
            var ox: UInt64 = 0
            var oz: UInt64 = 0
            var orr: UInt8 = 0
            other.image(self.x[i], self.z[i], self.r[i], ox, oz, orr)
            out.x[i] = ox
            out.z[i] = oz
            out.r[i] = orr
        return out^

    def inverse(self) -> SympTab:
        """Tableau of C^dag: symplectic inverse Omega M^T Omega, signs fixed by
        requiring C (C^dag P C) C^dag = P on every generator."""
        var n = self.n
        var out = SympTab(n)
        for i in range(n):
            var ax: UInt64 = 0
            var az: UInt64 = 0
            var bx: UInt64 = 0
            var bz: UInt64 = 0
            for j in range(n):
                if _has(self.z[n + j], i): ax |= _bit(j)
                if _has(self.z[j], i): az |= _bit(j)
                if _has(self.x[n + j], i): bx |= _bit(j)
                if _has(self.x[j], i): bz |= _bit(j)
            out.x[i] = ax
            out.z[i] = az
            out.x[n + i] = bx
            out.z[n + i] = bz
        for row in range(2 * n):
            var ox: UInt64 = 0
            var oz: UInt64 = 0
            var orr: UInt8 = 0
            self.image(out.x[row], out.z[row], 0, ox, oz, orr)
            out.r[row] = orr
        return out^

    def is_identity(self) -> Bool:
        for i in range(self.n):
            if self.x[i] != _bit(i) or self.z[i] != 0 or self.r[i] != 0:
                return False
            if self.x[self.n + i] != 0 or self.z[self.n + i] != _bit(i) or self.r[self.n + i] != 0:
                return False
        return True

    def is_linear(self) -> Bool:
        """True when C is a CX circuit up to Pauli signs (X rows X-only, Z rows Z-only)."""
        for i in range(self.n):
            if self.z[i] != 0 or self.x[self.n + i] != 0:
                return False
        return True

    def has_signs(self) -> Bool:
        for v in self.r:
            if v != 0:
                return True
        return False

    def linear_part(self) -> GF2Mat:
        """GF(2) matrix M of a linear tableau (column i of M = X-row i)."""
        var rows = List[UInt64](length=self.n, fill=0)
        for i in range(self.n):
            for r in range(self.n):
                if _has(self.x[i], r):
                    rows[r] |= _bit(i)
        return GF2Mat(self.n, rows)

    def is_local(self) -> Bool:
        """True when every generator of qubit j maps to a Pauli on qubit j only."""
        for i in range(self.n):
            var m = ~_bit(i)
            if (self.x[i] & m) != 0 or (self.z[i] & m) != 0:
                return False
            if (self.x[self.n + i] & m) != 0 or (self.z[self.n + i] & m) != 0:
                return False
        return True

    def eq_ignore_signs(self, o: SympTab) -> Bool:
        if self.n != o.n:
            return False
        for i in range(2 * self.n):
            if self.x[i] != o.x[i] or self.z[i] != o.z[i]:
                return False
        return True

    def __eq__(self, o: Self) -> Bool:
        if not self.eq_ignore_signs(o):
            return False
        for i in range(2 * self.n):
            if self.r[i] != o.r[i]:
                return False
        return True

    def __ne__(self, o: Self) -> Bool:
        return not self.__eq__(o)

    def write_to(self, mut w: Some[Writer]):
        w.write("SympTab(n=", self.n)
        for i in range(2 * self.n):
            w.write(" ", "X" if i < self.n else "Z", i % max(self.n, 1), "->", "-" if self.r[i] != 0 else "+")
            for j in range(self.n):
                var xb = _has(self.x[i], j)
                var zb = _has(self.z[i], j)
                w.write("Y" if xb and zb else ("X" if xb else ("Z" if zb else "I")))
        w.write(")")


# ----------------------------------------------------------------------------
# Pauli networks
# ----------------------------------------------------------------------------
@fieldwise_init
struct PauliRot(ImplicitlyCopyable, Writable):
    """Rotation exp(-i theta sign P / 2) in the block input frame; theta is the
    parameter of src_ops[src] (P = (x, z) as a Hermitian Pauli, sign = +-1)."""

    var x: UInt64
    var z: UInt64
    var sign: Int
    var src: Int

    def weight(self) -> Int:
        return Int(pop_count(self.x | self.z))

    def write_to(self, mut w: Some[Writer]):
        w.write("PauliRot(", "-" if self.sign < 0 else "+")
        for j in range(64):
            if _bit(j) > (self.x | self.z):
                break
            var xb = _has(self.x, j)
            var zb = _has(self.z, j)
            w.write("Y" if xb and zb else ("X" if xb else ("Z" if zb else "I")))
        w.write(", src=", self.src, ")")


def rotation_pauli(g: Instruction, mut x: UInt64, mut z: UInt64) -> Bool:
    """Pauli generator of RX / RY / RZ / RXX / RYY / RZZ (exp(-i theta P / 2))."""
    var k = g.kind
    x = 0
    z = 0
    if k == Op.RX: x = _bit(g.q0)
    elif k == Op.RZ: z = _bit(g.q0)
    elif k == Op.RY:
        x = _bit(g.q0); z = _bit(g.q0)
    elif k == Op.RXX:
        x = _bit(g.q0) | _bit(g.q1)
    elif k == Op.RZZ:
        z = _bit(g.q0) | _bit(g.q1)
    elif k == Op.RYY:
        x = _bit(g.q0) | _bit(g.q1)
        z = x
    else:
        return False
    return True


struct PauliNet(Copyable, Writable):
    """Block = (prod_k exp(-i theta_k sign_k P_k / 2), first k first) then `final`."""

    var n: Int
    var rots: List[PauliRot]
    var final: SympTab

    def __init__(out self, n: Int):
        self.n = n
        self.rots = List[PauliRot]()
        self.final = SympTab(n)

    def __init__(out self, *, copy: Self):
        self.n = copy.n
        self.rots = copy.rots.copy()
        self.final = copy.final.copy()

    @staticmethod
    def from_ops(ops: List[Instruction], n: Int) raises -> PauliNet:
        """Pull every non-Clifford Pauli rotation back through its Clifford
        prefix C (P -> C^dag P C). Rotations by multiples of pi/2 are Clifford
        and go into `final`; other gates raise QM-E701."""
        if n > 64:
            raise Error("QM-E618: Pauli block of " + String(n) + " qubits exceeds 64. Hint: collect blocks of at most 64 qubits.")
        var net = PauliNet(n)
        var inv = SympTab(n)  # tableau of C^dag
        for k in range(len(ops)):
            var g = ops[k]
            var kind = g.kind
            if kind == Op.BARRIER or kind == Op.TICK or kind == Op.I:
                continue
            if is_rotation_kind(kind) and not g.is_clifford():
                var px: UInt64 = 0
                var pz: UInt64 = 0
                _ = rotation_pauli(g, px, pz)
                if (op_arity(kind) >= 1 and (g.q0 < 0 or g.q0 >= n)) or (op_arity(kind) == 2 and (g.q1 < 0 or g.q1 >= n)):
                    raise Error("QM-E618: rotation '" + String(g) + "' acts outside the " + String(n) + "-qubit block. Hint: relabel the block with local_ops first.")
                var ox: UInt64 = 0
                var oz: UInt64 = 0
                var orr: UInt8 = 0
                inv.image(px, pz, 0, ox, oz, orr)
                net.rots.append(PauliRot(ox, oz, -1 if orr != 0 else 1, k))
                continue
            if not g.is_clifford():
                raise Error("QM-E701: gate '" + String(g) + "' is neither Clifford nor a Pauli rotation. Hint: Pauli-network blocks accept Clifford gates plus RX, RY, RZ, RXX, RYY, RZZ.")
            net.final.apply(g)
            inv.prepend(inverse_inst(g))
        return net^

    def write_to(self, mut w: Some[Writer]):
        w.write("PauliNet(n=", self.n, ", rots=", len(self.rots), ")")


def emit_rotation(rot: PauliRot, src_ops: List[Instruction], q: Int, axis: Int, sign: Int) -> Instruction:
    """Single-qubit rotation for `rot` once its Pauli has been mapped to `axis`
    (0 = X, 1 = Y, 2 = Z) on local qubit q. `sign` (+-1) is the sign of the
    mapped Pauli relative to (rot.x, rot.z); the emitted angle is
    rot.sign * sign * theta of src_ops[rot.src]."""
    var src = src_ops[rot.src]
    var kind = Op.RZ
    if axis == 0: kind = Op.RX
    elif axis == 1: kind = Op.RY
    var p = src.p0
    if rot.sign * sign < 0:
        p = -p
    return Instruction.make(kind, q, p0=p)


# ----------------------------------------------------------------------------
# Dense checks (tests and phase fixing; n <= 14)
# ----------------------------------------------------------------------------
def apply_ops_dense(ops: List[Instruction], n: Int, mut re: List[Float64], mut im: List[Float64]) raises:
    """Apply bound 1q / 2q gates to a dense state (little-endian)."""
    var dim = 1 << n
    for g in ops:
        var k = g.kind
        if k == Op.BARRIER or k == Op.TICK or k == Op.I:
            continue
        if g.is_parametric():
            raise Error("QM-E618: symbolic gate '" + String(g) + "' in a dense check. Hint: bind parameters first.")
        var a = op_arity(k)
        if a == 1:
            var m = matrix_1q(k, g.p0.value(), g.p1.value(), g.p2.value())
            var bq = 1 << g.q0
            for i in range(dim):
                if (i & bq) != 0:
                    continue
                var j = i | bq
                var ar = re[i]; var ai = im[i]
                var br = re[j]; var bi = im[j]
                re[i] = m.re[0] * ar - m.im[0] * ai + m.re[1] * br - m.im[1] * bi
                im[i] = m.re[0] * ai + m.im[0] * ar + m.re[1] * bi + m.im[1] * br
                re[j] = m.re[2] * ar - m.im[2] * ai + m.re[3] * br - m.im[3] * bi
                im[j] = m.re[2] * ai + m.im[2] * ar + m.re[3] * bi + m.im[3] * br
        elif a == 2:
            var m = matrix_2q(k, g.p0.value())
            var b0 = 1 << g.q0
            var b1 = 1 << g.q1
            for i in range(dim):
                if (i & b0) != 0 or (i & b1) != 0:
                    continue
                var idx = List[Int](length=4, fill=0)
                idx[0] = i
                idx[1] = i | b0
                idx[2] = i | b1
                idx[3] = i | b0 | b1
                var vr = List[Float64](length=4, fill=0.0)
                var vi = List[Float64](length=4, fill=0.0)
                for t in range(4):
                    vr[t] = re[idx[t]]
                    vi[t] = im[idx[t]]
                for r in range(4):
                    var sr = 0.0
                    var si = 0.0
                    for c in range(4):
                        var mr = m.re[r * 4 + c]
                        var mi = m.im[r * 4 + c]
                        sr += mr * vr[c] - mi * vi[c]
                        si += mr * vi[c] + mi * vr[c]
                    re[idx[r]] = sr
                    im[idx[r]] = si
        else:
            raise Error("QM-E618: gate '" + String(g) + "' unsupported in a dense check. Hint: only 1q / 2q gates.")


def _test_state(n: Int, mut re: List[Float64], mut im: List[Float64]):
    """Deterministic generic state (non-zero overlap with every basis state)."""
    var dim = 1 << n
    re = List[Float64](length=dim, fill=0.0)
    im = List[Float64](length=dim, fill=0.0)
    var s: UInt64 = 0x9E3779B97F4A7C15
    var nrm = 0.0
    for i in range(dim):
        s = s * 6364136223846793005 + 1442695040888963407
        var a = Float64(Int((s >> 33) & 0xFFFF)) / 65536.0 - 0.5
        s = s * 6364136223846793005 + 1442695040888963407
        var b = Float64(Int((s >> 33) & 0xFFFF)) / 65536.0 - 0.5
        re[i] = a
        im[i] = b
        nrm += a * a + b * b
    var inv = 1.0 / sqrt(nrm)
    for i in range(dim):
        re[i] *= inv
        im[i] *= inv


def phase_offset(ref_ops: List[Instruction], new_ops: List[Instruction], n: Int) raises -> Float64:
    """Global phase phi with new = e^{i phi} ref, measured on one generic state
    (valid once equality up to phase has been verified exactly). n <= 14."""
    if n > 14:
        raise Error("QM-E618: phase_offset limited to 14 qubits (got " + String(n) + "). Hint: fix the phase another way for wide blocks.")
    var ar = List[Float64]()
    var ai = List[Float64]()
    _test_state(n, ar, ai)
    var br = ar.copy()
    var bi = ai.copy()
    apply_ops_dense(ref_ops, n, ar, ai)
    apply_ops_dense(new_ops, n, br, bi)
    var sr = 0.0
    var si = 0.0
    for i in range(1 << n):
        # conj(a) * b
        sr += ar[i] * br[i] + ai[i] * bi[i]
        si += ar[i] * bi[i] - ai[i] * br[i]
    return atan2(si, sr)


def dense_equiv(ref_ops: List[Instruction], new_ops: List[Instruction], n: Int, mut phase: Float64, tol: Float64 = 1e-8) raises -> Bool:
    """Unitary equality up to global phase, checked column by column (n <= 10)."""
    if n > 10:
        raise Error("QM-E618: dense_equiv limited to 10 qubits (got " + String(n) + "). Hint: use the algebraic checks for wide blocks.")
    var dim = 1 << n
    var have = False
    var pr = 1.0
    var pi_ = 0.0
    for col in range(dim):
        var ar = List[Float64](length=dim, fill=0.0)
        var ai = List[Float64](length=dim, fill=0.0)
        ar[col] = 1.0
        var br = ar.copy()
        var bi = ai.copy()
        apply_ops_dense(ref_ops, n, ar, ai)
        apply_ops_dense(new_ops, n, br, bi)
        if not have:
            # pick the largest entry of the reference column to fix the phase
            var best = 0
            var bv = -1.0
            for i in range(dim):
                var v = ar[i] * ar[i] + ai[i] * ai[i]
                if v > bv:
                    bv = v
                    best = i
            # ratio b/a
            var den = ar[best] * ar[best] + ai[best] * ai[best]
            pr = (br[best] * ar[best] + bi[best] * ai[best]) / den
            pi_ = (bi[best] * ar[best] - br[best] * ai[best]) / den
            have = True
        for i in range(dim):
            var er = pr * ar[i] - pi_ * ai[i]
            var ei = pr * ai[i] + pi_ * ar[i]
            if abs(er - br[i]) > tol or abs(ei - bi[i]) > tol:
                return False
    phase = atan2(pi_, pr)
    return True
