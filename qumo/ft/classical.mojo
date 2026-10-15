"""Classical expressions over measurement bits, for static feed-forward.

Guppy FT programs run classical logic (decoders, `if` on decoded bits, retry
counters of state factories) on the device's classical runtime. QUMO circuits
are static instruction lists whose only classical primitive is a condition
"bit == v" (or an AND of such literals, `c_if_register`). `qumo.ft` therefore
represents every classical value it needs as an exclusive-or of terms

    CExpr = T_1 xor T_2 xor ...,  T = clbit | AND of literals (minterm) | 1

(an ESOP form, which can express any Boolean function) and evaluates it
*inside the circuit* only when a quantum operation must depend on it, with a
"classical lane": a dedicated qubit kept in a computational basis state
(reset; one X conditioned on each term; measure into a fresh clbit). The lane
never interacts with data qubits and is excluded from noise, so it is an exact
stand-in for a classical register on any QUMO backend (state vector, tableau,
OpenQASM 3 export). Values that are only reported (logical outputs) or only
feed Pauli corrections stay symbolic until the end of the program: this is how
QUMO "defers measurement reads as late as possible" in a static IR.

Idea from Guppy FT (Apache-2.0): docs/architecture_dev.md ("defer calling
read() on measurements as late as possible") and src/guppyft/std/state_factory.py
(`PreBlock.force_check`); the ESOP lane is QUMO's static-IR realisation.
"""


struct CTerm(Copyable, Writable):
    """kind 0: clbit lits[0] (== 1); kind 1: AND_i (lits[2i] == lits[2i+1]); kind 2: constant 1."""

    var kind: Int
    var lits: List[Int]

    def __init__(out self, kind: Int, var lits: List[Int]):
        self.kind = kind
        self.lits = lits^

    @staticmethod
    def clbit(c: Int) -> CTerm:
        var l = List[Int]()
        l.append(c)
        return CTerm(0, l^)

    @staticmethod
    def one() -> CTerm:
        return CTerm(2, List[Int]())

    @staticmethod
    def minterm(bits: List[Int], vals: List[Int]) -> CTerm:
        """AND over (bits[i] == vals[i]); stored sorted by bit."""
        var idx = List[Int]()
        for i in range(len(bits)):
            idx.append(i)
        # insertion sort by bit (lists are tiny)
        for i in range(1, len(idx)):
            var j = i
            while j > 0 and bits[idx[j - 1]] > bits[idx[j]]:
                var t = idx[j - 1]
                idx[j - 1] = idx[j]
                idx[j] = t
                j -= 1
        var l = List[Int]()
        for i in idx:
            l.append(bits[i])
            l.append(vals[i])
        if len(bits) == 1 and vals[0] == 1:
            return CTerm.clbit(bits[0])
        return CTerm(1, l^)

    def same(self, o: CTerm) -> Bool:
        if self.kind != o.kind or len(self.lits) != len(o.lits):
            return False
        for i in range(len(self.lits)):
            if self.lits[i] != o.lits[i]:
                return False
        return True

    def bits(self) -> List[Int]:
        var out = List[Int]()
        if self.kind == 0:
            out.append(self.lits[0])
        elif self.kind == 1:
            for i in range(len(self.lits) // 2):
                out.append(self.lits[2 * i])
        return out^

    def vals(self) -> List[Int]:
        var out = List[Int]()
        if self.kind == 0:
            out.append(1)
        elif self.kind == 1:
            for i in range(len(self.lits) // 2):
                out.append(self.lits[2 * i + 1])
        return out^

    def eval(self, creg: List[UInt8]) -> Int:
        if self.kind == 2:
            return 1
        if self.kind == 0:
            return Int(creg[self.lits[0]])
        for i in range(len(self.lits) // 2):
            if Int(creg[self.lits[2 * i]]) != self.lits[2 * i + 1]:
                return 0
        return 1

    def write_to(self, mut w: Some[Writer]):
        if self.kind == 2:
            w.write("1")
        elif self.kind == 0:
            w.write("c", self.lits[0])
        else:
            w.write("(")
            for i in range(len(self.lits) // 2):
                if i > 0:
                    w.write("&")
                if self.lits[2 * i + 1] == 0:
                    w.write("!")
                w.write("c", self.lits[2 * i])
            w.write(")")


struct CExpr(Copyable, Writable):
    """XOR of CTerms; the empty expression is the constant 0."""

    var terms: List[CTerm]

    def __init__(out self):
        self.terms = List[CTerm]()

    @staticmethod
    def zero() -> CExpr:
        return CExpr()

    @staticmethod
    def one() -> CExpr:
        var e = CExpr()
        e.terms.append(CTerm.one())
        return e^

    @staticmethod
    def bit(c: Int) -> CExpr:
        var e = CExpr()
        e.terms.append(CTerm.clbit(c))
        return e^

    @staticmethod
    def of(t: CTerm) -> CExpr:
        var e = CExpr()
        e.terms.append(t.copy())
        return e^

    def is_zero(self) -> Bool:
        return len(self.terms) == 0

    def xor_term(mut self, t: CTerm):
        for i in range(len(self.terms)):
            if self.terms[i].same(t):
                _ = self.terms.pop(i)
                return
        self.terms.append(t.copy())

    def xor(mut self, o: CExpr):
        for t in o.terms:
            self.xor_term(t)

    def xored(self, o: CExpr) -> CExpr:
        var e = self.copy()
        e.xor(o)
        return e^

    def single_condition(self) -> Bool:
        """Directly usable as a circuit condition (one clbit or one minterm)."""
        return len(self.terms) == 1 and self.terms[0].kind != 2

    def eval(self, creg: List[UInt8]) -> Int:
        var v = 0
        for t in self.terms:
            v ^= t.eval(creg)
        return v

    def write_to(self, mut w: Some[Writer]):
        if len(self.terms) == 0:
            w.write("0")
            return
        for i in range(len(self.terms)):
            if i > 0:
                w.write(" ^ ")
            w.write(self.terms[i])
