"""Pauli strings and Hamiltonians (bit-packed symplectic form, <= 64 qubits).

A Pauli term is coeff * i^{|x&z|} X^x Z^z  (so a qubit with x=z=1 is Y).
Parsing accepts the two notations LLMs emit most often:
    "0.5 * Z0 Z1 - 1.2 * X0 Y2 + 0.3"   (sparse, indexed)
    "0.5*ZZI + 0.2*XIX"                 (dense, leftmost char = highest qubit,
                                         Qiskit SparsePauliOp convention)
"""

from std.bit import pop_count


@fieldwise_init
struct PauliTerm(ImplicitlyCopyable, Writable):
    var coeff: Float64
    var x: UInt64
    var z: UInt64

    def weight(self) -> Int:
        return Int(pop_count(self.x | self.z))

    def num_y(self) -> Int:
        return Int(pop_count(self.x & self.z))

    def is_diagonal(self) -> Bool:
        return self.x == 0

    def max_qubit(self) -> Int:
        var m = self.x | self.z
        var q = -1
        var i = 0
        while m != 0:
            if (m & 1) == 1:
                q = i
            m >>= 1
            i += 1
        return q

    def commutes_with(self, o: PauliTerm) -> Bool:
        var s = pop_count(self.x & o.z) + pop_count(self.z & o.x)
        return (s & 1) == 0

    def qubit_wise_commutes(self, o: PauliTerm) -> Bool:
        """Qubit-wise commuting => measurable in the same basis setting."""
        var supp = (self.x | self.z) & (o.x | o.z)
        return ((self.x ^ o.x) & supp) == 0 and ((self.z ^ o.z) & supp) == 0

    def label(self, n: Int) -> String:
        """Dense label, leftmost = qubit n-1."""
        var s = String()
        var q = n - 1
        while q >= 0:
            var b = UInt64(1) << UInt64(q)
            var hx = (self.x & b) != 0
            var hz = (self.z & b) != 0
            if hx and hz: s += "Y"
            elif hx: s += "X"
            elif hz: s += "Z"
            else: s += "I"
            q -= 1
        return s

    def write_to(self, mut w: Some[Writer]):
        w.write(self.coeff)
        if self.x == 0 and self.z == 0:
            return
        w.write(" *")
        for q in range(64):
            var b = UInt64(1) << UInt64(q)
            var hx = (self.x & b) != 0
            var hz = (self.z & b) != 0
            if hx and hz: w.write(" Y", q)
            elif hx: w.write(" X", q)
            elif hz: w.write(" Z", q)


struct PauliSum(Copyable, Writable, Sized):
    var terms: List[PauliTerm]

    def __init__(out self):
        self.terms = List[PauliTerm]()

    def __len__(self) -> Int:
        return len(self.terms)

    def add(mut self, coeff: Float64, x: UInt64, z: UInt64):
        for ref t in self.terms:
            if t.x == x and t.z == z:
                t.coeff += coeff
                return
        self.terms.append(PauliTerm(coeff, x, z))

    def add_label(mut self, coeff: Float64, label: String) raises:
        """Add a dense label term, e.g. add_label(0.5, "XZI")."""
        var n = label.byte_length()
        var bs = label.as_bytes()
        var x: UInt64 = 0
        var z: UInt64 = 0
        for i in range(n):
            var q = UInt64(n - 1 - i)
            var ch = Int(bs[i])
            if ch == ord("X"): x |= UInt64(1) << q
            elif ch == ord("Z"): z |= UInt64(1) << q
            elif ch == ord("Y"):
                x |= UInt64(1) << q
                z |= UInt64(1) << q
            elif ch != ord("I"):
                raise Error("QM-E301: invalid Pauli character in '" + label + "'. Hint: use only I, X, Y, Z.")
        self.add(coeff, x, z)

    def num_qubits(self) -> Int:
        var m = -1
        for t in self.terms:
            m = max(m, t.max_qubit())
        return m + 1

    def __add__(self, o: PauliSum) -> PauliSum:
        var r = self.copy()
        for t in o.terms:
            r.add(t.coeff, t.x, t.z)
        return r^

    def scaled(self, a: Float64) -> PauliSum:
        var r = self.copy()
        for ref t in r.terms:
            t.coeff *= a
        return r^

    def is_diagonal(self) -> Bool:
        for t in self.terms:
            if t.x != 0:
                return False
        return True

    def group_qwc(self) -> List[List[Int]]:
        """Greedy qubit-wise-commuting grouping (measurement-shot reduction)."""
        var groups = List[List[Int]]()
        for i in range(len(self.terms)):
            var placed = False
            for ref g in groups:
                var ok = True
                for j in g:
                    if not self.terms[i].qubit_wise_commutes(self.terms[j]):
                        ok = False
                        break
                if ok:
                    g.append(i)
                    placed = True
                    break
            if not placed:
                var ng = List[Int]()
                ng.append(i)
                groups.append(ng^)
        return groups^

    def write_to(self, mut w: Some[Writer]):
        for i in range(len(self.terms)):
            if i > 0:
                w.write(" + ")
            w.write(self.terms[i])


def _is_space(c: Int) -> Bool:
    return c == 32 or c == 9 or c == 10 or c == 13


def parse_pauli_sum(text: String) raises -> PauliSum:
    """Parse "0.5 * Z0 Z1 - 1.2 * X0 Y2 + 0.3" or "0.5*ZZI + XIX"."""
    var out = PauliSum()
    var bs = text.as_bytes()
    var n = text.byte_length()
    var i = 0
    var sign: Float64
    while i < n:
        # skip spaces and leading sign(s)
        while i < n and _is_space(Int(bs[i])):
            i += 1
        if i >= n:
            break
        sign = 1.0
        while i < n and (Int(bs[i]) == ord("+") or Int(bs[i]) == ord("-") or _is_space(Int(bs[i]))):
            if Int(bs[i]) == ord("-"):
                sign = -sign
            i += 1
        # optional coefficient
        var coeff = 1.0
        var start = i
        while i < n and (
            (Int(bs[i]) >= ord("0") and Int(bs[i]) <= ord("9")) or Int(bs[i]) == ord(".")
            or Int(bs[i]) == ord("e") or Int(bs[i]) == ord("E")
            or ((Int(bs[i]) == ord("-") or Int(bs[i]) == ord("+")) and i > start and (Int(bs[i - 1]) == ord("e") or Int(bs[i - 1]) == ord("E")))
        ):
            i += 1
        if i > start:
            coeff = Float64(String(unsafe_from_utf8=bs[start:i]))
        while i < n and (_is_space(Int(bs[i])) or Int(bs[i]) == ord("*")):
            i += 1
        # Pauli factors until next top-level + / -
        var x: UInt64 = 0
        var z: UInt64 = 0
        var dense = String()
        while i < n and Int(bs[i]) != ord("+") and Int(bs[i]) != ord("-"):
            var ch = Int(bs[i])
            if _is_space(ch) or ch == ord("*"):
                i += 1
                continue
            if ch == ord("X") or ch == ord("Y") or ch == ord("Z") or ch == ord("I"):
                # indexed form if followed by digit
                if i + 1 < n and Int(bs[i + 1]) >= ord("0") and Int(bs[i + 1]) <= ord("9"):
                    var j = i + 1
                    while j < n and Int(bs[j]) >= ord("0") and Int(bs[j]) <= ord("9"):
                        j += 1
                    var q = Int(String(unsafe_from_utf8=bs[i + 1:j]))
                    if q >= 64:
                        raise Error("QM-E302: Pauli index " + String(q) + " >= 64 is not supported.")
                    var b = UInt64(1) << UInt64(q)
                    if ch == ord("X"): x |= b
                    elif ch == ord("Z"): z |= b
                    elif ch == ord("Y"):
                        x |= b
                        z |= b
                    i = j
                else:
                    dense += String(unsafe_from_utf8=bs[i:i + 1])
                    i += 1
                continue
            raise Error("QM-E301: unexpected character at offset " + String(i) + " in Pauli expression. Hint: write e.g. '0.5*Z0 Z1 + 0.2*X0'.")
        if dense.byte_length() > 0:
            var tmp = PauliSum()
            tmp.add_label(1.0, dense)
            x |= tmp.terms[0].x
            z |= tmp.terms[0].z
        out.add(sign * coeff, x, z)
    return out^
