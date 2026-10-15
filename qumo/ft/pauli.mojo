"""Symplectic Pauli operators and stabilizer-group algebra for `qumo.ft`.

`SymPauli` is an n-qubit Pauli operator `i^phase * P_0 x P_1 x ... x P_{n-1}`
with one (x, z) bit pair per qubit ((1,1) is Y, the Hermitian convention used by
`qumo.sim.stabilizer`). Unlike `qumo.ir.pauli.PauliTerm` it is not limited to
64 qubits, carries an exact phase, and is meant for the small exact algebra
that QEC code definitions and gadget verification need.

`StabilizerGroup` keeps a set of commuting generators in row-echelon form over
GF(2) *while tracking the group products with their phases*, so
`contains(P)` answers "is P (with its sign) an element of the group", which is
what code validation and Clifford-gadget verification reduce to.

Label convention (code definitions): character i of a label is qubit i
(leftmost = qubit 0), the convention of Guppy FT's `StabilizerCode` and Stim.
QUMO's dense Pauli labels elsewhere are Qiskit-style (leftmost = highest
qubit); `qumo.ft` never mixes the two.

Idea from Guppy FT (Apache-2.0), files src/guppyft/code_def.py and
src/guppyft/verify/_expansion.py (zixy `SignTerm` algebra); re-derived here.

Error codes: QM-E720 (invalid Pauli label or size mismatch).
"""

from std.collections import List


def pauli_char(x: Int, z: Int) -> String:
    if x == 0 and z == 0:
        return "I"
    if x == 1 and z == 0:
        return "X"
    if x == 1 and z == 1:
        return "Y"
    return "Z"


def single_product_phase(x1: Int, z1: Int, x2: Int, z2: Int) -> Int:
    """log_i phase of P1 * P2 for single-qubit Hermitian Paulis (Y convention):
    XY = iZ, YZ = iX, ZX = iY (+1) and the reverse orders give -1 (= 3 mod 4)."""
    var a = x1 + 2 * z1  # 1 = X, 3 = Y, 2 = Z
    var b = x2 + 2 * z2
    if a == 0 or b == 0 or a == b:
        return 0
    # cyclic order X -> Y -> Z -> X
    if (a == 1 and b == 3) or (a == 3 and b == 2) or (a == 2 and b == 1):
        return 1
    return 3


struct SymPauli(Copyable, Writable):
    """`i^phase * prod_q P_q` on `n` qubits (phase in 0..3)."""

    var n: Int
    var xs: List[UInt8]
    var zs: List[UInt8]
    var phase: Int

    def __init__(out self, n: Int):
        self.n = n
        self.xs = List[UInt8](length=n, fill=0)
        self.zs = List[UInt8](length=n, fill=0)
        self.phase = 0

    @staticmethod
    def identity(n: Int) -> SymPauli:
        return SymPauli(n)

    @staticmethod
    def single(n: Int, q: Int, p: String) raises -> SymPauli:
        """One-qubit Pauli `p` ("X", "Y" or "Z") on qubit q of n."""
        var out = SymPauli(n)
        out.set(q, p)
        return out^

    @staticmethod
    def from_label(label: String) raises -> SymPauli:
        """Parse "+XZZXI", "-IZZI" or "XXX" (char i = qubit i)."""
        var bs = label.as_bytes()
        var start = 0
        var sign = 0
        if len(bs) > 0 and (Int(bs[0]) == ord("+") or Int(bs[0]) == ord("-")):
            if Int(bs[0]) == ord("-"):
                sign = 2
            start = 1
        var n = len(bs) - start
        if n <= 0:
            raise Error("QM-E720: empty Pauli label '" + label + "'. Hint: write one of I, X, Y, Z per qubit, e.g. 'XXXXIII'.")
        var out = SymPauli(n)
        for i in range(n):
            var ch = Int(bs[start + i])
            if ch == ord("I"):
                continue
            elif ch == ord("X"):
                out.xs[i] = 1
            elif ch == ord("Y"):
                out.xs[i] = 1
                out.zs[i] = 1
            elif ch == ord("Z"):
                out.zs[i] = 1
            else:
                raise Error("QM-E720: invalid character in Pauli label '" + label + "'. Hint: use only I, X, Y, Z (optionally prefixed by + or -).")
        out.phase = sign
        return out^

    def set(mut self, q: Int, p: String) raises:
        if p == "I":
            self.xs[q] = 0
            self.zs[q] = 0
        elif p == "X":
            self.xs[q] = 1
            self.zs[q] = 0
        elif p == "Y":
            self.xs[q] = 1
            self.zs[q] = 1
        elif p == "Z":
            self.xs[q] = 0
            self.zs[q] = 1
        else:
            raise Error("QM-E720: invalid Pauli '" + p + "'. Hint: use I, X, Y or Z.")

    def get(self, q: Int) -> String:
        return pauli_char(Int(self.xs[q]), Int(self.zs[q]))

    def x(self, q: Int) -> Int:
        return Int(self.xs[q])

    def z(self, q: Int) -> Int:
        return Int(self.zs[q])

    def weight(self) -> Int:
        var w = 0
        for q in range(self.n):
            if self.xs[q] != 0 or self.zs[q] != 0:
                w += 1
        return w

    def is_identity(self) -> Bool:
        """True when the Pauli part is I (any phase)."""
        return self.weight() == 0

    def is_x_type(self) -> Bool:
        """No Z/Y component (X or I on every qubit)."""
        for q in range(self.n):
            if self.zs[q] != 0:
                return False
        return True

    def is_z_type(self) -> Bool:
        for q in range(self.n):
            if self.xs[q] != 0:
                return False
        return True

    def is_hermitian(self) -> Bool:
        return self.phase == 0 or self.phase == 2

    def sign(self) -> Int:
        """+1 / -1 for Hermitian operators, 0 for +-i."""
        if self.phase == 0:
            return 1
        if self.phase == 2:
            return -1
        return 0

    def same_pauli(self, o: SymPauli) -> Bool:
        """Equal up to phase."""
        if self.n != o.n:
            return False
        for q in range(self.n):
            if self.xs[q] != o.xs[q] or self.zs[q] != o.zs[q]:
                return False
        return True

    def equals(self, o: SymPauli) -> Bool:
        return self.same_pauli(o) and self.phase == o.phase

    def commutes(self, o: SymPauli) -> Bool:
        var s = 0
        var m = min(self.n, o.n)
        for q in range(m):
            s += Int(self.xs[q] & o.zs[q]) + Int(self.zs[q] & o.xs[q])
        return (s & 1) == 0

    def mul(self, o: SymPauli) raises -> SymPauli:
        """self * o (operator product, phases tracked)."""
        if self.n != o.n:
            raise Error("QM-E720: Pauli size mismatch (" + String(self.n) + " vs " + String(o.n) + " qubits). Hint: embed both operators on the same register first.")
        var out = SymPauli(self.n)
        var ph = self.phase + o.phase
        for q in range(self.n):
            var x1 = Int(self.xs[q])
            var z1 = Int(self.zs[q])
            var x2 = Int(o.xs[q])
            var z2 = Int(o.zs[q])
            ph += single_product_phase(x1, z1, x2, z2)
            out.xs[q] = UInt8(x1 ^ x2)
            out.zs[q] = UInt8(z1 ^ z2)
        out.phase = ph & 3
        return out^

    def times_phase(self, k: Int) -> SymPauli:
        var out = self.copy()
        out.phase = (self.phase + k) & 3
        return out^

    def embed(self, offset: Int, total: Int) -> SymPauli:
        """Place this operator on qubits offset..offset+n-1 of a `total`-qubit register."""
        var out = SymPauli(total)
        for q in range(self.n):
            out.xs[offset + q] = self.xs[q]
            out.zs[offset + q] = self.zs[q]
        out.phase = self.phase
        return out^

    def restrict(self, offset: Int, count: Int) -> SymPauli:
        """Qubits offset..offset+count-1 (phase kept)."""
        var out = SymPauli(count)
        for q in range(count):
            out.xs[q] = self.xs[offset + q]
            out.zs[q] = self.zs[offset + q]
        out.phase = self.phase
        return out^

    def support(self) -> List[Int]:
        var out = List[Int]()
        for q in range(self.n):
            if self.xs[q] != 0 or self.zs[q] != 0:
                out.append(q)
        return out^

    def label(self) -> String:
        """"+XZZX" style (char i = qubit i); phases +-i print as "+i"/"-i"."""
        var s: String
        if self.phase == 0:
            s = "+"
        elif self.phase == 1:
            s = "+i"
        elif self.phase == 2:
            s = "-"
        else:
            s = "-i"
        for q in range(self.n):
            s += pauli_char(Int(self.xs[q]), Int(self.zs[q]))
        return s

    def write_to(self, mut w: Some[Writer]):
        w.write(self.label())


def pauli_list_from_labels(labels: List[String]) raises -> List[SymPauli]:
    var out = List[SymPauli]()
    for l in labels:
        out.append(SymPauli.from_label(l))
    return out^


struct StabilizerGroup(Copyable):
    """Commuting Pauli generators in GF(2) row-echelon form, phases tracked.

    Rows are genuine group elements (products of the inserted generators with
    their phases), so reducing an operator P by the rows leaves `i^phi * I`
    exactly when P is in the group up to that phase."""

    var n: Int
    var rows: List[SymPauli]
    var pivots: List[Int]  # 0..n-1: x part of qubit q; n..2n-1: z part of qubit q-n

    def __init__(out self, n: Int):
        self.n = n
        self.rows = List[SymPauli]()
        self.pivots = List[Int]()

    def _bit(self, p: SymPauli, col: Int) -> Int:
        if col < self.n:
            return Int(p.xs[col])
        return Int(p.zs[col - self.n])

    def reduce(self, p: SymPauli) raises -> SymPauli:
        """p multiplied by the rows needed to clear every pivot column."""
        var v = p.copy()
        for i in range(len(self.rows)):
            if self._bit(v, self.pivots[i]) == 1:
                v = v.mul(self.rows[i])
        return v^

    def add(mut self, g: SymPauli) raises -> Bool:
        """Insert a generator; False when it is dependent (up to phase)."""
        var v = self.reduce(g)
        for col in range(2 * self.n):
            if self._bit(v, col) == 1:
                self.rows.append(v^)
                self.pivots.append(col)
                return True
        return False

    def rank(self) -> Int:
        return len(self.rows)

    def contains(self, p: SymPauli) raises -> Bool:
        """p (with its sign) is an element of the group."""
        var v = self.reduce(p)
        return v.is_identity() and v.phase == 0

    def contains_up_to_sign(self, p: SymPauli) raises -> Bool:
        var v = self.reduce(p)
        return v.is_identity()

    def residual_phase(self, p: SymPauli) raises -> Int:
        """phase of reduce(p) when p is in the group up to a phase, else -1."""
        var v = self.reduce(p)
        if not v.is_identity():
            return -1
        return v.phase


def group_from(gens: List[SymPauli], n: Int) raises -> StabilizerGroup:
    var g = StabilizerGroup(n)
    for p in gens:
        _ = g.add(p)
    return g^
