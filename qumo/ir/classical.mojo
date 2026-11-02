"""Classical values of a `Program`: types, expressions and their evaluator.

Types are `bit[n]`, `bool`, `int[n]`, `uint[n]` (1 <= n <= 64), `float` (binary64)
and `angle[n]` (an unsigned n-bit fraction of a full turn). The bits of a value
live in an `Int`: masked for `bit`, `bool`, `uint` and `angle`, sign-extended for
`int`. Arithmetic is modulo 2^n in two's complement.

An expression is a list of `CNode` in postorder, so one forward scan evaluates it
without recursion, and a shot allocates nothing: each node's result goes to a
slot of `CStore`. The builders here (`ce_binary`, `ce_cast`, ...) do the typing
and insert every `CAST`, so the evaluator has no implicit conversions and a node
always reads operands of the type it expects. `&&` and `||` are the one place
the scan jumps: `SC_AND` and `SC_OR` skip the right operand.

The rules the OpenQASM 3 specification leaves open, fixed here (and in
docs/CONTROL_FLOW.md):
  * `/` and `%` truncate toward zero, literals included; `%` has the sign of the
    dividend; dividing by zero raises QM-E481 at run time, QM-E533 when the divisor
    is constant. An expression folds to what it would compute at run time, with one
    exception: an integer constant with no width (a literal, `const int`, an unrolled
    `int` loop variable) is exact. It does not wrap at 64 bits: a result past them
    goes on as a float, a negative exponent gives a float, and negating an unsized
    `uint` gives the negative number. See `_exact_binary`.
  * Shifts apply to bit, uint and angle; shifting by the width or more gives 0.
  * `int` with `uint` is QM-E580, except for literals.
  * `bool` and `bit` promote to a 64-bit `int` in arithmetic, as in C99, so
    `true + 1` is 2 and `uint[8]` 255 plus `true` is 256. Next to a 64-bit `uint`
    they stay in it.
  * A literal takes the type of the other operand when it fits. A literal too wide
    for it is QM-E580 in arithmetic and bitwise operators; a comparison widens the
    operand, so it stays exact, and a shift count is free.
  * Float to int truncates; NaN, infinity or out of range raises QM-E482.
  * `angle` to int, uint or float is refused, as in the specification's cast
    table. As a gate argument an angle[n] with bits v is 2 pi v / 2^n radians.
  * Float to angle[n] rounds to the nearest step, a tie to the even step, modulo a turn.
"""

from std.bit import pop_count
from std.math import sin, cos, tan, asin, acos, atan, exp, log, sqrt, floor, ceil, isnan, isinf, pi


struct CT:
    comptime BIT = 0
    comptime BOOL = 1
    comptime UINT = 2
    comptime INT = 3
    comptime FLOAT = 4
    comptime ANGLE = 5


@fieldwise_init
struct CType(ImplicitlyCopyable, Equatable):
    var tag: Int  # CT.*
    var width: Int  # 1..64; FLOAT is 64
    var bare: Bool  # declared without a width: 64-bit, emitted as written

    def __eq__(self, o: CType) -> Bool:
        return self.tag == o.tag and self.width == o.width

    def is_integer(self) -> Bool:
        """Held in the integer slots (everything but float)."""
        return self.tag != CT.FLOAT

    def is_signed(self) -> Bool:
        return self.tag == CT.INT


def bit_type(width: Int = 1) -> CType:
    return CType(CT.BIT, width, False)


def bool_type() -> CType:
    return CType(CT.BOOL, 1, False)


def uint_type(width: Int = 64, bare: Bool = False) -> CType:
    return CType(CT.UINT, width, bare)


def int_type(width: Int = 64, bare: Bool = False) -> CType:
    return CType(CT.INT, width, bare)


def float_type(bare: Bool = True) -> CType:
    return CType(CT.FLOAT, 64, bare)


def angle_type(width: Int) -> CType:
    return CType(CT.ANGLE, width, False)


def type_name(t: CType) -> String:
    """OpenQASM spelling of a type."""
    if t.tag == CT.BOOL:
        return "bool"
    if t.tag == CT.FLOAT:
        return "float" if t.bare else "float[64]"
    var nm = String("bit" if t.tag == CT.BIT else ("uint" if t.tag == CT.UINT else ("int" if t.tag == CT.INT else "angle")))
    if t.bare or (t.tag == CT.BIT and t.width == 1):
        return nm^
    return nm + "[" + String(t.width) + "]"


struct CN:
    """Expression node operations (the operands of a node are `x` and `y`)."""

    comptime CONST = 0  # i, or f for float; y = 1 marks an untyped literal
    comptime VAR = 1  # x = storage index
    comptime INPUT = 2  # x = index into the shot's parameters
    comptime BITS = 3  # x = first clbit, y = count
    comptime NEG = 4
    comptime LNOT = 5
    comptime BNOT = 6
    comptime ADD = 7
    comptime SUB = 8
    comptime MUL = 9
    comptime DIV = 10
    comptime MOD = 11
    comptime POW = 12
    comptime SHL = 13
    comptime SHR = 14
    comptime AND = 15
    comptime OR = 16
    comptime XOR = 17
    comptime EQ = 18
    comptime NE = 19
    comptime LT = 20
    comptime LE = 21
    comptime GT = 22
    comptime GE = 23
    comptime CAST = 24  # x, to `ty`; the source type is that of node x
    comptime FN = 25  # i = CF.*, x, y (second operand or -1)
    comptime SC_AND = 26  # x = left operand; y = its SC_END
    comptime SC_OR = 27
    comptime SC_END = 28  # x = right operand


struct CF:
    """Functions of `CN.FN`."""

    comptime SIN = 0
    comptime COS = 1
    comptime TAN = 2
    comptime ASIN = 3
    comptime ACOS = 4
    comptime ATAN = 5
    comptime EXP = 6
    comptime LOG = 7
    comptime SQRT = 8
    comptime ABS = 9
    comptime FLOOR = 10
    comptime CEIL = 11
    comptime MOD = 12
    comptime POPCOUNT = 13


@fieldwise_init
struct CNode(ImplicitlyCopyable):
    var op: Int
    var ty: CType
    var x: Int  # operand node, storage index or first clbit
    var y: Int  # operand node, bit count or SC_END node
    var i: Int
    var f: Float64


struct CEval:
    """Status of an evaluation."""

    comptime OK = 0
    comptime DIV_ZERO = 1  # QM-E481
    comptime FLOAT_RANGE = 2  # QM-E482
    comptime NEG_POWER = 3  # QM-E481


trait BitReader:
    def get_bit(self, c: Int) -> Int:
        ...


@fieldwise_init
struct NoBits(BitReader, ImplicitlyCopyable):
    """For constant expressions, which read no bits."""

    def get_bit(self, c: Int) -> Int:
        return 0


struct CStore(Movable):
    """Classical state of one shot; allocated once, reused for every shot."""

    var ints: List[Int]  # by storage index
    var reals: List[Float64]
    var ti: List[Int]  # node results
    var tf: List[Float64]
    var params: List[Float64]  # user values, then runtime-angle slots

    def __init__(out self, num_vars: Int, num_nodes: Int, var params: List[Float64]):
        self.ints = List[Int](length=num_vars, fill=0)
        self.reals = List[Float64](length=num_vars, fill=0.0)
        self.ti = List[Int](length=num_nodes, fill=0)
        self.tf = List[Float64](length=num_nodes, fill=0.0)
        self.params = params^

    def begin_shot(mut self, base: List[Float64]):
        """Nothing survives from the shot before: variables back to zero, and the
        parameters to `base` (input values, runtime-angle slots at zero)."""
        for i in range(len(self.ints)):
            self.ints[i] = 0
            self.reals[i] = 0.0
        for i in range(len(base)):
            self.params[i] = base[i]


@always_inline
def wrap_int(v: Int, ty: CType) -> Int:
    """Reduce to the type's width: masked, or sign-extended for `int`."""
    var w = ty.width
    if w >= 64:
        return v
    if ty.tag == CT.INT:
        var s = 64 - w
        return (v << s) >> s
    return v & ((1 << w) - 1)


def _sdiv(a: Int, b: Int) -> Int:
    """Truncating signed division; INT64_MIN / -1 wraps instead of trapping."""
    var ua = UInt64(0) - UInt64(a) if a < 0 else UInt64(a)
    var ub = UInt64(0) - UInt64(b) if b < 0 else UInt64(b)
    var q = ua / ub
    return Int(UInt64(0) - q) if (a < 0) != (b < 0) else Int(q)


def _srem(a: Int, b: Int) -> Int:
    """Remainder of `_sdiv`: the sign of the dividend."""
    var ua = UInt64(0) - UInt64(a) if a < 0 else UInt64(a)
    var ub = UInt64(0) - UInt64(b) if b < 0 else UInt64(b)
    var r = ua % ub
    return Int(UInt64(0) - r) if a < 0 else Int(r)


def _ipow(a: Int, e: Int, ty: CType) -> Int:
    var out = 1
    var base = a
    var k = UInt64(e)  # a uint exponent can use all 64 bits
    while k != 0:
        if (k & 1) != 0:
            out = wrap_int(out * base, ty)
        base = wrap_int(base * base, ty)
        k >>= 1
    return wrap_int(out, ty)


def _two_pow(w: Int) -> Float64:
    return 18446744073709551616.0 if w >= 64 else Float64(UInt64(1) << UInt64(w))


def _float_to_int(x: Float64, ty: CType, mut out: Int) -> Bool:
    """Truncate into `ty`; False when NaN, infinite or out of range."""
    if isnan(x) or isinf(x):
        return False
    var t = -floor(-x) if x < 0.0 else floor(x)
    var w = ty.width
    if ty.tag == CT.INT:
        var top = _two_pow(w - 1)
        if t < -top or t >= top:
            return False
        out = Int(t)
        return True
    if t < 0.0 or t >= _two_pow(w):
        return False
    out = Int(UInt64(t))
    return True


def _float_to_angle(x: Float64, width: Int, mut out: Int) -> Bool:
    """Nearest step of a turn; False when NaN or infinite."""
    if isnan(x) or isinf(x):
        return False
    var turns = x / (2.0 * pi)
    var frac = turns - floor(turns)
    var scale = _two_pow(width)
    var t = frac * scale
    var v = floor(t)
    var d = t - v
    if d > 0.5 or (d == 0.5 and v / 2.0 != floor(v / 2.0)):
        v += 1.0
    out = 0 if v >= scale else Int(UInt64(v))
    return True


def _fn_real(f: Int, a: Float64, b: Float64) -> Float64:
    if f == CF.SIN: return sin(a)
    if f == CF.COS: return cos(a)
    if f == CF.TAN: return tan(a)
    if f == CF.ASIN: return asin(a)
    if f == CF.ACOS: return acos(a)
    if f == CF.ATAN: return atan(a)
    if f == CF.EXP: return exp(a)
    if f == CF.LOG: return log(a)
    if f == CF.SQRT: return sqrt(a)
    if f == CF.ABS: return abs(a)
    if f == CF.FLOOR: return floor(a)
    if f == CF.CEIL: return ceil(a)
    return a - b * floor(a / b)  # CF.MOD


def eval_nodes[B: BitReader](nodes: List[CNode], lo: Int, root: Int, mut st: CStore, bits: B) -> Int:
    """Evaluate nodes lo..root in order, leaving each result in st.ti / st.tf
    by node index. Returns a `CEval` status."""
    var pc = lo
    while pc <= root:
        ref n = nodes[pc]
        var op = n.op
        var ty = n.ty
        var fl = ty.tag == CT.FLOAT
        if op == CN.CONST:
            if fl:
                st.tf[pc] = n.f
            else:
                st.ti[pc] = n.i
        elif op == CN.VAR:
            if fl:
                st.tf[pc] = st.reals[n.x]
            else:
                st.ti[pc] = st.ints[n.x]
        elif op == CN.BITS:
            var v = 0
            for j in range(n.y):
                v |= bits.get_bit(n.x + j) << j
            st.ti[pc] = v
        elif op == CN.INPUT:
            st.tf[pc] = st.params[n.x]
        elif op == CN.CAST:
            var src = nodes[n.x].ty
            if src.tag == CT.FLOAT:
                var x = st.tf[n.x]
                if ty.tag == CT.FLOAT:
                    st.tf[pc] = x
                elif ty.tag == CT.BOOL:
                    st.ti[pc] = 0 if x == 0.0 else 1
                elif ty.tag == CT.ANGLE:
                    var v = 0
                    if not _float_to_angle(x, ty.width, v):
                        return CEval.FLOAT_RANGE
                    st.ti[pc] = v
                else:
                    var v = 0
                    if not _float_to_int(x, ty, v):
                        return CEval.FLOAT_RANGE
                    st.ti[pc] = v
            else:
                var v = st.ti[n.x]
                if ty.tag == CT.FLOAT:
                    st.tf[pc] = Float64(v) if src.is_signed() else Float64(UInt64(v))
                elif ty.tag == CT.BOOL:
                    st.ti[pc] = 0 if v == 0 else 1
                elif ty.tag == CT.ANGLE and src.tag == CT.ANGLE and src.width != ty.width:
                    # keep the turn: pad low bits, or drop them
                    if ty.width > src.width:
                        st.ti[pc] = wrap_int(v << (ty.width - src.width), ty)
                    else:
                        st.ti[pc] = Int(UInt64(v) >> UInt64(src.width - ty.width))
                else:
                    st.ti[pc] = wrap_int(v, ty)
        elif op == CN.ADD or op == CN.SUB or op == CN.MUL:
            if fl:
                var a = st.tf[n.x]
                var b = st.tf[n.y]
                st.tf[pc] = a + b if op == CN.ADD else (a - b if op == CN.SUB else a * b)
            else:
                var a = st.ti[n.x]
                var b = st.ti[n.y]
                st.ti[pc] = wrap_int(a + b if op == CN.ADD else (a - b if op == CN.SUB else a * b), ty)
        elif op == CN.DIV or op == CN.MOD:
            var src = nodes[n.x].ty
            if fl:
                var a = st.tf[n.x]
                var b = st.tf[n.y]
                if b == 0.0:
                    return CEval.DIV_ZERO
                if op == CN.DIV:
                    st.tf[pc] = a / b
                else:
                    var q = a / b
                    st.tf[pc] = a - b * (-floor(-q) if q < 0.0 else floor(q))
            else:
                var a = st.ti[n.x]
                var b = st.ti[n.y]
                if b == 0:
                    return CEval.DIV_ZERO
                var v: Int
                if src.is_signed():
                    v = _sdiv(a, b) if op == CN.DIV else _srem(a, b)
                else:
                    var ua = UInt64(a)
                    var ub = UInt64(b)
                    v = Int(ua / ub) if op == CN.DIV else Int(ua % ub)
                st.ti[pc] = wrap_int(v, ty)
        elif op == CN.POW:
            if fl:
                st.tf[pc] = st.tf[n.x] ** st.tf[n.y]
            else:
                if nodes[n.y].ty.is_signed() and st.ti[n.y] < 0:
                    return CEval.NEG_POWER
                st.ti[pc] = _ipow(st.ti[n.x], st.ti[n.y], ty)
        elif op == CN.SHL or op == CN.SHR:
            var a = st.ti[n.x]
            var s = UInt64(st.ti[n.y])
            if s >= UInt64(ty.width):
                st.ti[pc] = 0
            elif op == CN.SHL:
                st.ti[pc] = wrap_int(a << Int(s), ty)
            else:
                st.ti[pc] = wrap_int(Int(UInt64(a) >> s), ty)
        elif op == CN.AND:
            st.ti[pc] = st.ti[n.x] & st.ti[n.y]
        elif op == CN.OR:
            st.ti[pc] = st.ti[n.x] | st.ti[n.y]
        elif op == CN.XOR:
            st.ti[pc] = st.ti[n.x] ^ st.ti[n.y]
        elif op >= CN.EQ and op <= CN.GE:
            var src = nodes[n.x].ty
            var r: Bool
            if src.tag == CT.FLOAT:
                var a = st.tf[n.x]
                var b = st.tf[n.y]
                if op == CN.EQ: r = a == b
                elif op == CN.NE: r = a != b
                elif op == CN.LT: r = a < b
                elif op == CN.LE: r = a <= b
                elif op == CN.GT: r = a > b
                else: r = a >= b
            elif src.is_signed():
                var a = st.ti[n.x]
                var b = st.ti[n.y]
                if op == CN.EQ: r = a == b
                elif op == CN.NE: r = a != b
                elif op == CN.LT: r = a < b
                elif op == CN.LE: r = a <= b
                elif op == CN.GT: r = a > b
                else: r = a >= b
            else:
                var a = UInt64(st.ti[n.x])
                var b = UInt64(st.ti[n.y])
                if op == CN.EQ: r = a == b
                elif op == CN.NE: r = a != b
                elif op == CN.LT: r = a < b
                elif op == CN.LE: r = a <= b
                elif op == CN.GT: r = a > b
                else: r = a >= b
            st.ti[pc] = 1 if r else 0
        elif op == CN.NEG:
            if fl:
                st.tf[pc] = -st.tf[n.x]
            else:
                st.ti[pc] = wrap_int(0 - st.ti[n.x], ty)
        elif op == CN.LNOT:
            st.ti[pc] = 1 - st.ti[n.x]
        elif op == CN.BNOT:
            st.ti[pc] = wrap_int(~st.ti[n.x], ty)
        elif op == CN.SC_AND:
            if st.ti[n.x] == 0:
                st.ti[n.y] = 0
                pc = n.y + 1
                continue
        elif op == CN.SC_OR:
            if st.ti[n.x] != 0:
                st.ti[n.y] = 1
                pc = n.y + 1
                continue
        elif op == CN.SC_END:
            st.ti[pc] = st.ti[n.x]
        elif op == CN.FN:
            if n.i == CF.POPCOUNT:
                st.ti[pc] = Int(pop_count(UInt64(st.ti[n.x])))
            elif n.i == CF.ABS and not fl:
                var a = st.ti[n.x]
                st.ti[pc] = wrap_int(0 - a if a < 0 else a, ty)
            else:
                var d = st.tf[n.y] if n.y >= 0 else 0.0
                if n.i == CF.MOD and d == 0.0:
                    return CEval.DIV_ZERO
                st.tf[pc] = _fn_real(n.i, st.tf[n.x], d)
        pc += 1
    return CEval.OK


# ---------------------------------------------------------------------------
# Typed expressions
# ---------------------------------------------------------------------------
struct CExpr(Copyable):
    """A self-contained expression under construction: postorder nodes whose
    operands index this list. `Program.add_expr` relocates it into the program."""

    var nodes: List[CNode]

    def __init__(out self):
        self.nodes = List[CNode]()

    def root(self) -> CNode:
        return self.nodes[len(self.nodes) - 1]

    def ty(self) -> CType:
        return self.root().ty

    def is_const(self) -> Bool:
        return len(self.nodes) == 1 and self.root().op == CN.CONST

    def is_literal(self) -> Bool:
        return self.is_const() and self.root().y == 1

    def const_int(self) -> Int:
        return self.root().i

    def const_float(self) -> Float64:
        var n = self.root()
        return n.f if n.ty.tag == CT.FLOAT else (Float64(n.i) if n.ty.is_signed() else Float64(UInt64(n.i)))

    @staticmethod
    def _leaf(op: Int, ty: CType, x: Int = 0, y: Int = 0, i: Int = 0, f: Float64 = 0.0) -> CExpr:
        var e = CExpr()
        e.nodes.append(CNode(op, ty, x, y, i, f))
        return e^

    @staticmethod
    def literal_int(v: Int) -> CExpr:
        return CExpr._leaf(CN.CONST, int_type(64, True), 0, 1, v)

    @staticmethod
    def literal_uint(v: Int) -> CExpr:
        """A literal at or above 2^63, held by its bits."""
        return CExpr._leaf(CN.CONST, uint_type(64, True), 0, 1, v)

    @staticmethod
    def literal_float(v: Float64) -> CExpr:
        return CExpr._leaf(CN.CONST, float_type(), 0, 1, 0, v)

    @staticmethod
    def literal_bool(v: Bool) -> CExpr:
        return CExpr._leaf(CN.CONST, bool_type(), 0, 1, 1 if v else 0)

    @staticmethod
    def typed_const(ty: CType, i: Int, f: Float64 = 0.0) -> CExpr:
        return CExpr._leaf(CN.CONST, ty, 0, 0, i, f)

    @staticmethod
    def variable(store: Int, ty: CType) -> CExpr:
        return CExpr._leaf(CN.VAR, ty, store)

    @staticmethod
    def input(index: Int) -> CExpr:
        return CExpr._leaf(CN.INPUT, float_type(), index)

    @staticmethod
    def bits(first: Int, count: Int) -> CExpr:
        return CExpr._leaf(CN.BITS, bit_type(count), first, count)


def _append(mut out: CExpr, e: CExpr) -> Int:
    """Copy `e` into `out`, relocating its operands; returns its root index."""
    var base = len(out.nodes)
    for n in e.nodes:
        var c = n
        if c.op >= CN.NEG:
            c.x += base
            if c.y >= 0:
                c.y += base
        out.nodes.append(c)
    return len(out.nodes) - 1


def _fold(var e: CExpr) raises -> CExpr:
    """Replace an all-constant expression by one CONST node."""
    for n in e.nodes:
        if n.op == CN.VAR or n.op == CN.INPUT or n.op == CN.BITS:
            return e^
    var st = CStore(0, len(e.nodes), List[Float64]())
    var root = len(e.nodes) - 1
    var status = eval_nodes(e.nodes, 0, root, st, NoBits())
    if status == CEval.DIV_ZERO:
        raise Error("QM-E533: division by zero in a constant expression. Hint: check the denominator.")
    if status == CEval.NEG_POWER:
        raise Error("QM-E533: integer power with a negative exponent. Hint: use a float base.")
    if status == CEval.FLOAT_RANGE:
        raise Error("QM-E482: constant float is NaN, infinite or out of range for this integer type. Hint: check the value.")
    var literal = True
    for n in e.nodes:
        if n.op == CN.CONST and n.y != 1:
            literal = False
    var ty = e.root().ty
    if ty.tag == CT.FLOAT:
        return CExpr._leaf(CN.CONST, ty, 0, 1 if literal else 0, 0, st.tf[root])
    return CExpr._leaf(CN.CONST, ty, 0, 1 if literal else 0, st.ti[root])


def _unary(op: Int, ty: CType, a: CExpr, i: Int = 0, y: Int = -1) raises -> CExpr:
    var e = a.copy()
    e.nodes.append(CNode(op, ty, len(e.nodes) - 1, y, i, 0.0))
    return _fold(e^)


def _pair(op: Int, ty: CType, a: CExpr, b: CExpr, i: Int = 0) raises -> CExpr:
    var e = CExpr()
    var ra = _append(e, a)
    var rb = _append(e, b)
    e.nodes.append(CNode(op, ty, ra, rb, i, 0.0))
    return _fold(e^)


def can_cast(src: CType, dst: CType) -> Bool:
    """The specification's table of allowed casts, among the supported types."""
    var s = src.tag
    var d = dst.tag
    if s == d:
        return True
    if s == CT.ANGLE:
        return d == CT.BOOL or (d == CT.BIT and src.width == dst.width)
    if d == CT.ANGLE:
        return s == CT.FLOAT or (s == CT.BIT and src.width == dst.width)
    if s == CT.FLOAT:
        return d != CT.BIT
    if s == CT.BIT:
        return d != CT.FLOAT
    return True


def ce_cast(a: CExpr, ty: CType) raises -> CExpr:
    """`a` as `ty`, by the cast table (QM-E580 when refused). A sized target wraps; an
    unsized value does not wrap into an unsized target (QM-E482 when out of range)."""
    var src = a.ty()
    if _is_unsized(a) and ty.bare and (ty.tag == CT.INT or ty.tag == CT.UINT):
        var v = _exact_of(a)
        if (ty.tag == CT.INT and v >= (Int128(1) << 63)) or (ty.tag == CT.UINT and v < 0):
            raise Error("QM-E482: the constant " + String(v) + " is out of range for " + type_name(ty) + ". Hint: use a type that holds it, or cast to a sized type to wrap.")
    if src == ty:
        if a.is_const():
            var e = a.copy()
            e.nodes[0].ty = ty  # keeps the declared spelling
            e.nodes[0].y = 0
            return e^
        return a.copy()
    if not can_cast(src, ty):
        raise Error("QM-E580: cannot convert " + type_name(src) + " to " + type_name(ty) + ". Hint: the allowed casts are the table in docs/CONTROL_FLOW.md; angle converts to bool or to a bit register of its width only.")
    var r = _unary(CN.CAST, ty, a)
    if r.is_const():
        r.nodes[0].y = 0
    return r^


def ce_explicit_cast(a: CExpr, ty: CType) raises -> CExpr:
    """A cast written in the source: `bit[m](x)` also needs x to have m bits."""
    var src = a.ty()
    if ty.tag == CT.BIT and src.tag != CT.FLOAT:
        var fits = src.width == ty.width
        if a.is_literal():
            fits = a.const_int() >= 0 and _fits(a.const_int(), src.tag == CT.UINT, uint_type(ty.width))
        if not fits:
            raise Error("QM-E580: bit[" + String(ty.width) + "] takes a value of " + String(ty.width) + " bits, not " + type_name(src) + ". Hint: cast through uint[" + String(ty.width) + "] first.")
    return ce_cast(a, ty)


def ce_bool(a: CExpr) raises -> CExpr:
    """`a` as a condition."""
    return ce_cast(a, bool_type())


def _fits(v: Int, literal_unsigned: Bool, ty: CType) -> Bool:
    """The integer literal `v` is representable in `ty`. An unsigned literal is at
    or above 2^63 and held by its bits, so `v` is negative as an Int."""
    if literal_unsigned:
        return ty.tag != CT.INT and ty.width >= 64
    if ty.tag == CT.INT:
        return ty.width >= 64 or (v >= -(1 << (ty.width - 1)) and v < (1 << (ty.width - 1)))
    if v < 0:
        return False
    return ty.width >= 63 or v < (1 << ty.width)


def _width_for(v: Int, unsigned_: Bool, signed_target: Bool) -> Int:
    """Smallest width of the target's kind that holds the literal."""
    if unsigned_:
        return 64
    var w = 1
    while w < 64 and not (v < (1 << w) if not signed_target else (v >= -(1 << (w - 1)) and v < (1 << (w - 1)))):
        w += 1
    return w


# ---------------------------------------------------------------------------
# Exact integer constants
# ---------------------------------------------------------------------------
def _is_unsized(e: CExpr) -> Bool:
    """An integer constant with no width. Its value is exact: it is not an int64."""
    if not e.is_const() or not e.ty().bare:
        return False
    return e.ty().tag == CT.INT or e.ty().tag == CT.UINT


def _exact_operand(e: CExpr) -> Bool:
    return _is_unsized(e) or (e.is_const() and _is_bitlike(e.ty()))


def _exact_of(e: CExpr) -> Int128:
    var n = e.root()
    return Int128(n.i) if n.ty.is_signed() else Int128(UInt64(n.i))


def _real_const(v: Float64, literal: Bool) -> CExpr:
    return CExpr._leaf(CN.CONST, float_type(), 0, 1 if literal else 0, 0, v)


def _exact_const(v: Int128, unsigned_: Bool, literal: Bool) -> CExpr:
    """`v` as a constant. Past the 64 bits of int and uint it goes on as a float."""
    if v < -(Int128(1) << 63) or v >= (Int128(1) << 64):
        return _real_const(Float64(v), literal)
    var y = 1 if literal else 0
    if v >= (Int128(1) << 63):
        return CExpr._leaf(CN.CONST, uint_type(64, True), 0, y, Int(UInt64(v)))
    return CExpr._leaf(CN.CONST, uint_type(64, True) if unsigned_ and v >= 0 else int_type(64, True), 0, y, Int(v))


comptime _EXACT_LIMIT = 1.0e30  # below it a product of two 65-bit values fits Int128


def _exact_pow(x: Int128, e: Int128, unsigned_: Bool, literal: Bool) -> CExpr:
    if e < 0:
        return _real_const(Float64(x) ** Float64(e), literal)
    if x == 0 or x == 1 or e == 0:
        return _exact_const(Int128(1) if e == 0 else x, unsigned_, literal)
    if x == -1:
        return _exact_const(Int128(1) if e % 2 == 0 else Int128(-1), unsigned_, literal)
    var r = Int128(1)
    var left = e
    while left > 0:
        if abs(Float64(r) * Float64(x)) >= _EXACT_LIMIT:
            return _real_const(Float64(x) ** Float64(e), literal)
        r *= x
        left -= 1
    return _exact_const(r, unsigned_, literal)


def _exact_shift(op: Int, x: Int128, n: Int128, literal: Bool) -> CExpr:
    if op == CN.SHR:
        return _exact_const(Int128(0) if n >= 64 else x >> n, True, literal)
    if n < 126 and x < (Int128(1) << (126 - n)):
        return _exact_const(x << n, True, literal)
    return _real_const(Float64(x) * Float64(2.0) ** Float64(n), literal)


def _exact_binary(op: Int, a: CExpr, b: CExpr) raises -> CExpr:
    """`a op b` on integer constants with no width, in exact arithmetic. This is the
    one place where folding differs from run time, which would wrap or refuse."""
    var x = _exact_of(a)
    var y = _exact_of(b)
    var unsigned_ = a.ty().tag == CT.UINT or b.ty().tag == CT.UINT
    var literal = a.is_literal() and b.is_literal()
    if op == CN.ADD:
        return _exact_const(x + y, unsigned_, literal)
    if op == CN.SUB:
        return _exact_const(x - y, unsigned_, literal)
    if op == CN.MUL:
        var f = Float64(x) * Float64(y)
        if abs(f) >= _EXACT_LIMIT:
            return _real_const(f, literal)
        return _exact_const(x * y, unsigned_, literal)
    if op == CN.DIV or op == CN.MOD:
        if y == 0:
            raise Error("QM-E533: division by a constant zero. Hint: check the divisor.")
        var q = x / y  # truncates, as the table says
        return _exact_const(q if op == CN.DIV else x - q * y, unsigned_, literal)
    if op == CN.POW:
        return _exact_pow(x, y, unsigned_, literal)
    if op == CN.SHL or op == CN.SHR:
        return _exact_shift(op, x, y, literal)
    if op == CN.AND or op == CN.OR or op == CN.XOR:
        var ux = UInt64(x)
        var uy = UInt64(y)
        var bits = ux & uy if op == CN.AND else (ux | uy if op == CN.OR else ux ^ uy)
        if x < 0 or y < 0:
            return _exact_const(Int128(Int(bits)), False, literal)
        return _exact_const(Int128(bits), unsigned_, literal)
    var r: Bool
    if op == CN.EQ: r = x == y
    elif op == CN.NE: r = x != y
    elif op == CN.LT: r = x < y
    elif op == CN.LE: r = x <= y
    elif op == CN.GT: r = x > y
    else: r = x >= y
    return CExpr._leaf(CN.CONST, bool_type(), 0, 1 if literal else 0, 1 if r else 0)


def _exact_pair(op: Int, a: CExpr, b: CExpr) -> Bool:
    if not _exact_operand(a) or not _exact_operand(b) or not (_is_unsized(a) or _is_unsized(b)):
        return False
    if op == CN.SHL or op == CN.SHR:
        # a shift keeps the width of what it shifts, and `int` does not shift
        return _is_unsized(a) and (a.ty().tag == CT.UINT or a.is_literal()) and _exact_of(a) >= 0 and _exact_of(b) >= 0
    return True


def never_negative(e: CExpr) -> Bool:
    """A range bound that cannot be negative, so that it may pass 2^63."""
    var t = e.ty()
    if e.is_const() and t.is_signed():
        return e.const_int() >= 0
    return t.tag == CT.UINT or t.tag == CT.BIT or t.tag == CT.BOOL


def truncates(a: CExpr, b: CExpr) -> Bool:
    """Does `a / b` lose a remainder, on integer constants with no width?"""
    if not _exact_pair(CN.DIV, a, b):
        return False
    var y = _exact_of(b)
    if y == 0:
        return False
    var x = _exact_of(a)
    return x - (x / y) * y != 0


comptime _ARITH = 0
comptime _BITWISE = 1
comptime _COMPARE = 2


def _is_bitlike(t: CType) -> Bool:
    return t.tag == CT.BOOL or t.tag == CT.BIT


def _promoted(t: CType, mode: Int) -> CType:
    """bool and bit[n]: a 64-bit int in arithmetic, unsigned bits otherwise."""
    if _is_bitlike(t):
        return int_type(64) if mode == _ARITH else uint_type(t.width)
    return t


def _common(a: CExpr, b: CExpr, mut ca: CExpr, mut cb: CExpr, mode: Int) raises -> CType:
    """Bring two numeric operands to one type; `ca` and `cb` receive them cast."""
    var oa = a.ty()
    var ob = b.ty()
    var ta = _promoted(oa, mode)
    var tb = _promoted(ob, mode)
    var la = a.is_literal()
    var lb = b.is_literal()
    if mode == _ARITH and (_is_bitlike(oa) != _is_bitlike(ob)):
        # a 64-bit int holds any sum with a narrow operand; only a 64-bit uint does not
        var other = tb if _is_bitlike(oa) else ta
        if other.tag == CT.INT or (other.tag == CT.UINT and other.width < 64):
            ca = _to_common(a, int_type(64))
            cb = _to_common(b, int_type(64))
            return int_type(64)
    # bit and bool hold plain non-negative values, which either kind of integer takes
    if _is_bitlike(oa) and not lb and (tb.tag == CT.UINT or (tb.tag == CT.INT and mode != _ARITH)):
        ta = uint_type(max(tb.width, oa.width)) if tb.tag == CT.UINT else int_type(min(max(tb.width, oa.width + 1), 64))
    elif _is_bitlike(ob) and not la and (ta.tag == CT.UINT or (ta.tag == CT.INT and mode != _ARITH)):
        tb = uint_type(max(ta.width, ob.width)) if ta.tag == CT.UINT else int_type(min(max(ta.width, ob.width + 1), 64))
    var common: CType
    if ta.tag == CT.FLOAT or tb.tag == CT.FLOAT:
        if ta.tag == CT.ANGLE and lb:
            common = ta
        elif tb.tag == CT.ANGLE and la:
            common = tb
        elif ta.tag == CT.ANGLE or tb.tag == CT.ANGLE:
            raise Error("QM-E580: float and angle do not mix. Hint: cast the float, angle[n](x).")
        else:
            common = float_type()
    elif la and lb:
        common = uint_type(64, True) if ta.tag == CT.UINT or tb.tag == CT.UINT else int_type(64, True)
    elif la or lb:
        var lit = a.root() if la else b.root()
        var t = tb if la else ta
        var us = lit.ty.tag == CT.UINT
        if not _fits(lit.i, us, t):
            if t.tag == CT.UINT and lit.i < 0 and not us:
                raise Error("QM-E580: the negative literal " + String(lit.i) + " does not mix with " + type_name(t) + ". Hint: cast one side.")
            if t.tag == CT.INT and us:
                raise Error("QM-E580: an unsigned literal above the int range does not mix with " + type_name(t) + ". Hint: cast one side.")
            if mode != _COMPARE:
                raise Error("QM-E580: the literal does not fit " + type_name(t) + ". Hint: cast the operand to a wider type.")
            t.width = max(t.width, _width_for(lit.i, us, t.tag == CT.INT))
            t.bare = False
        common = t
    elif ta.tag != tb.tag:
        raise Error("QM-E580: " + type_name(a.ty()) + " and " + type_name(b.ty()) + " do not mix. Hint: cast one operand, e.g. int[32](x).")
    elif ta.tag == CT.ANGLE:
        if ta.width != tb.width:
            raise Error("QM-E580: angles of different widths do not mix. Hint: cast one side.")
        common = ta
    else:
        common = ta if ta.width >= tb.width else tb
    ca = _to_common(a, common)
    cb = _to_common(b, common)
    return common


def _to_common(a: CExpr, ty: CType) raises -> CExpr:
    var t = a.ty()
    if t == ty:
        return a.copy()
    var out: CExpr
    if t.tag == CT.BIT and ty.tag == CT.FLOAT:
        out = ce_cast(ce_cast(a, uint_type(t.width)), ty)  # the table has no bit -> float
    else:
        out = ce_cast(a, ty)
    if a.is_literal():
        out.nodes[0].y = 1  # a literal promoted by an operator stays a literal
    return out^


def ce_binary(op: Int, a: CExpr, b: CExpr) raises -> CExpr:
    """Typed arithmetic, bitwise, shift or comparison node."""
    if _exact_pair(op, a, b):
        return _exact_binary(op, a, b)
    var ta = a.ty()
    var tb = b.ty()
    var ca = CExpr()
    var cb = CExpr()
    if op == CN.SHL or op == CN.SHR:
        if a.is_literal() and ta.tag == CT.INT and a.const_int() >= 0:
            return ce_binary(op, CExpr.literal_uint(a.const_int()), b)
        if ta.tag == CT.INT or ta.tag == CT.FLOAT or ta.tag == CT.BOOL:
            raise Error("QM-E580: shifts apply to bit, uint and angle, not " + type_name(ta) + ". Hint: cast to uint[n] first.")
        if not (tb.tag == CT.UINT or tb.tag == CT.BIT or (b.is_literal() and tb.tag == CT.INT and b.const_int() >= 0)):
            raise Error("QM-E580: a shift count is a uint or a non-negative literal, not " + type_name(tb) + ". Hint: cast it to uint.")
        return _pair(op, ta, a, _to_shift_count(b))
    if op == CN.AND or op == CN.OR or op == CN.XOR:
        if (ta.tag == CT.INT and not a.is_literal()) or ta.tag == CT.FLOAT or (tb.tag == CT.INT and not b.is_literal()) or tb.tag == CT.FLOAT:
            raise Error("QM-E580: bitwise operators apply to bit, uint and angle. Hint: cast to uint[n] first.")
        if ta.tag == CT.BIT and tb.tag == CT.BIT:
            if ta.width != tb.width:
                raise Error("QM-E580: bit registers of different widths in a bitwise operator. Hint: cast one to the other's width.")
            return _pair(op, ta, a, b)
        var t = _common(a, b, ca, cb, _BITWISE)
        return _pair(op, t, ca, cb)
    if op >= CN.EQ and op <= CN.GE:
        if ta.tag == CT.BOOL and tb.tag == CT.BOOL:
            if op != CN.EQ and op != CN.NE:
                raise Error("QM-E580: bool values are compared with == and != only. Hint: cast to uint.")
            return _pair(op, bool_type(), a, b)
        _ = _common(a, b, ca, cb, _COMPARE)
        return _pair(op, bool_type(), ca, cb)
    # arithmetic
    if ta.tag == CT.ANGLE or tb.tag == CT.ANGLE:
        return _angle_arith(op, a, b)
    if op == CN.POW:
        if ta.tag == CT.FLOAT or tb.tag == CT.FLOAT:
            var t = _common(a, b, ca, cb, _ARITH)
            return _pair(op, t, ca, cb)
        # the exponent keeps its own type: its sign is read at run time
        var base = _to_common(a, _promoted(ta, _ARITH))
        return _pair(op, base.ty(), base, _exponent(b))
    var t = _common(a, b, ca, cb, _ARITH)
    if (op == CN.DIV or op == CN.MOD) and _is_zero(cb):
        raise Error("QM-E533: division by a constant zero. Hint: check the divisor.")
    return _pair(op, t, ca, cb)


def _is_zero(e: CExpr) -> Bool:
    return e.is_const() and e.root().f == 0.0 and e.root().i == 0


def _to_shift_count(b: CExpr) raises -> CExpr:
    if b.ty().tag == CT.INT:
        return ce_cast(b, uint_type(64))
    return b.copy()


def _exponent(b: CExpr) raises -> CExpr:
    if _is_bitlike(b.ty()):
        return ce_cast(b, uint_type(b.ty().width))
    return b.copy()


def _angle_arith(op: Int, a: CExpr, b: CExpr) raises -> CExpr:
    """angle +- angle; angle * uint; uint * angle; angle / uint; angle / angle -> uint."""
    var ta = a.ty()
    var tb = b.ty()
    var angle = ta if ta.tag == CT.ANGLE else tb
    if op == CN.ADD or op == CN.SUB:
        if a.is_literal() and ta.tag == CT.FLOAT:
            return _pair(op, angle, ce_cast(a, angle), b)
        if b.is_literal() and tb.tag == CT.FLOAT:
            return _pair(op, angle, a, ce_cast(b, angle))
        if ta != tb:
            raise Error("QM-E580: + and - take two angles of one width, not " + type_name(ta) + " and " + type_name(tb) + ". Hint: cast one side.")
        return _pair(op, angle, a, b)
    if op == CN.MUL:
        var scale = b.copy() if ta.tag == CT.ANGLE else a.copy()
        var ang = a.copy() if ta.tag == CT.ANGLE else b.copy()
        var ts = scale.ty()
        if ts.tag == CT.ANGLE or ts.tag == CT.FLOAT or ts.tag == CT.INT and not scale.is_literal():
            raise Error("QM-E580: an angle multiplies by a uint, not " + type_name(ts) + ". Hint: cast it to uint[" + String(angle.width) + "].")
        return _pair(op, angle, ang, _angle_scale(scale, angle))
    if op == CN.DIV and ta.tag == CT.ANGLE:
        if _is_zero(b):
            raise Error("QM-E533: division by a constant zero. Hint: check the divisor.")
        if tb.tag == CT.ANGLE:
            if ta != tb:
                raise Error("QM-E580: angles of different widths do not divide. Hint: cast one side.")
            return _pair(op, uint_type(ta.width), a, b)
        # the divisor must keep its value: no truncation to the angle's width
        if tb.tag == CT.FLOAT or (b.is_literal() and not _fits(b.const_int(), False, uint_type(ta.width))) or (not b.is_literal() and (tb.tag == CT.INT or tb.width != ta.width)):
            raise Error("QM-E580: an angle divides by a uint of its width or an angle, not " + type_name(tb) + ". Hint: cast it to uint[" + String(angle.width) + "].")
        return _pair(op, angle, a, _angle_scale(b, angle))
    raise Error("QM-E580: this operator does not apply to " + type_name(angle) + ". Hint: angles take + - * / and shifts.")


def _angle_scale(s: CExpr, angle: CType) raises -> CExpr:
    """A scale factor, as a uint of the angle's bit pattern width."""
    if s.is_const():
        if s.ty().is_signed() and s.const_int() < 0:
            raise Error("QM-E580: an angle scales by a non-negative value. Hint: negate the angle instead.")
    return ce_cast(s, uint_type(angle.width))


def ce_neg(a: CExpr) raises -> CExpr:
    if _is_unsized(a):
        return _exact_const(Int128(0) - _exact_of(a), False, a.is_literal())
    var t = _promoted(a.ty(), _ARITH)
    if a.is_literal():
        var n = a.root()
        if t.tag == CT.FLOAT:
            return CExpr.literal_float(-n.f)
        return CExpr.literal_int(0 - n.i)
    return _unary(CN.NEG, t, _to_common(a, t))


def ce_lnot(a: CExpr) raises -> CExpr:
    return _unary(CN.LNOT, bool_type(), ce_bool(a))


def ce_bnot(a: CExpr) raises -> CExpr:
    var t = a.ty()
    if t.tag == CT.INT or t.tag == CT.FLOAT or t.tag == CT.BOOL:
        raise Error("QM-E580: ~ applies to bit, uint and angle, not " + type_name(t) + ". Hint: for a bool write !x.")
    return _unary(CN.BNOT, t, a)


def ce_logical(is_and: Bool, a: CExpr, b: CExpr) raises -> CExpr:
    """`a && b` or `a || b`; the right operand runs only when it decides."""
    var la = ce_bool(a)
    var lb = ce_bool(b)
    if la.is_const():
        var v = la.const_int() != 0
        if v != is_and:
            return la^  # false && x, true || x
        return lb^
    var e = CExpr()
    var ra = _append(e, la)
    var sc = len(e.nodes)
    e.nodes.append(CNode(CN.SC_AND if is_and else CN.SC_OR, bool_type(), ra, -1, 0, 0.0))
    var rb = _append(e, lb)
    var end = len(e.nodes)
    e.nodes[sc].y = end
    e.nodes.append(CNode(CN.SC_END, bool_type(), rb, -1, 0, 0.0))
    return e^


def ce_function(f: Int, a: CExpr, b: Optional[CExpr] = None) raises -> CExpr:
    """sin cos ... of a float, abs of a number, popcount of bits, mod of two floats."""
    var t = a.ty()
    if f == CF.POPCOUNT:
        if t.tag != CT.BIT and t.tag != CT.UINT:
            raise Error("QM-E580: popcount takes bit or uint, not " + type_name(t) + ". Hint: cast the operand.")
        return _unary(CN.FN, uint_type(64), a, f)
    if f == CF.ABS and t.tag != CT.FLOAT:
        if _is_unsized(a):
            var v = _exact_of(a)
            return _exact_const(Int128(0) - v if v < 0 else v, t.tag == CT.UINT, a.is_literal())
        if t.tag == CT.INT:
            return _unary(CN.FN, t, a, f)
        return a.copy()
    var fa = _to_common(a, float_type())
    if f == CF.MOD:
        var e = CExpr()
        var ra = _append(e, fa)
        var rb = _append(e, _to_common(b.value(), float_type()))
        e.nodes.append(CNode(CN.FN, float_type(), ra, rb, f, 0.0))
        return _fold(e^)
    return _unary(CN.FN, float_type(), fa, f)


def angle_radians(bits: Int, width: Int) -> Float64:
    """Gate argument of an angle[width] holding `bits`."""
    return 2.0 * pi * Float64(UInt64(bits)) / _two_pow(width)
