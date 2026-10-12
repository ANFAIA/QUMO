"""Lightweight symbolic parameters.

A `Param` is an affine expression `coeff * theta[index] + offset`. That is
enough for variational algorithms (sharing, scaling, parameter-shift and
adjoint gradients need d(angle)/d(theta) = coeff) while staying a 24-byte
POD: binding a parameter vector never allocates, unlike general symbolic
expression trees (Qiskit ParameterExpression / sympy).
"""


@fieldwise_init
struct Param(ImplicitlyCopyable, Writable, Equatable):
    var offset: Float64
    var index: Int  # -1 => constant
    var coeff: Float64

    @implicit
    def __init__(out self, value: Float64):
        self.offset = value
        self.index = -1
        self.coeff = 0.0

    @implicit
    def __init__(out self, value: FloatLiteral):
        self.offset = Float64(value)
        self.index = -1
        self.coeff = 0.0

    @implicit
    def __init__(out self, value: IntLiteral):
        self.offset = Float64(value)
        self.index = -1
        self.coeff = 0.0

    @staticmethod
    def symbol(index: Int) -> Param:
        return Param(0.0, index, 1.0)

    def is_symbolic(self) -> Bool:
        return self.index >= 0

    def resolve(self, values: List[Float64]) -> Float64:
        if self.index < 0:
            return self.offset
        return self.coeff * values[self.index] + self.offset

    def value(self) -> Float64:
        """Constant value (offset) - only meaningful when not symbolic."""
        return self.offset

    def __neg__(self) -> Param:
        return Param(-self.offset, self.index, -self.coeff)

    def __add__(self, o: Float64) -> Param:
        return Param(self.offset + o, self.index, self.coeff)

    def __sub__(self, o: Float64) -> Param:
        return Param(self.offset - o, self.index, self.coeff)

    def __mul__(self, o: Float64) -> Param:
        return Param(self.offset * o, self.index, self.coeff * o)

    def __rmul__(self, o: Float64) -> Param:
        return self * o

    def try_add(self, o: Param, mut ok: Bool) -> Param:
        """Sum of two params when representable (same symbol or constants)."""
        if self.index < 0:
            ok = True
            return Param(self.offset + o.offset, o.index, o.coeff)
        if o.index < 0 or o.index == self.index:
            ok = True
            return Param(self.offset + o.offset, self.index, self.coeff + o.coeff)
        ok = False
        return self

    def __eq__(self, o: Param) -> Bool:
        return self.offset == o.offset and self.index == o.index and self.coeff == o.coeff

    def write_to(self, mut w: Some[Writer]):
        if self.index < 0:
            w.write(self.offset)
            return
        if self.coeff != 1.0:
            w.write(self.coeff, "*")
        w.write("$", self.index)
        if self.offset != 0.0:
            w.write("+", self.offset)
