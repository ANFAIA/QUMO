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
  * `/` and `%` truncate toward zero; `%` has the sign of the dividend; dividing
    by zero raises QM-E481 at run time, QM-E533 when both operands are constant.
  * Shifts apply to bit, uint and angle; shifting by the width or more gives 0.
  * `int` with `uint` is QM-E580, except for literals.
  * A literal takes the type of the other operand when it fits, otherwise that
    operand widens to the smallest width that holds the literal.
  * Float to int truncates; NaN, infinity or out of range raises QM-E482.
  * `angle` to int, uint or float is refused, as in the specification's cast
    table. As a gate argument an angle[n] with bits v is 2 pi v / 2^n radians.
  * Float to angle[n] rounds to the nearest step, a tie going up, modulo a turn.
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
    var k = e
    while k > 0:
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
    var v = floor(frac * scale + 0.5)
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
                st.tf[pc] = _fn_real(n.i, st.tf[n.x], st.tf[n.y] if n.y >= 0 else 0.0)
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
    """`a` as `ty`, by the cast table (QM-E580 when refused)."""
    var src = a.ty()
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


def ce_bool(a: CExpr) raises -> CExpr:
    """`a` as a condition."""
    return ce_cast(a, bool_type())


def _fits(v: Int, literal_unsigned: Bool, ty: CType) -> Bool:
    """The integer literal `v` is representable in `ty`."""
    if ty.tag == CT.INT:
        if literal_unsigned:
            return False
        return ty.width >= 64 or (v >= -(1 << (ty.width - 1)) and v < (1 << (ty.width - 1)))
    if v < 0 and not literal_unsigned:
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


def _uintish(t: CType) -> CType:
    """bool and bit[n] count as uint in arithmetic and comparisons."""
    if t.tag == CT.BOOL or t.tag == CT.BIT:
        return uint_type(t.width)
    return t


def _common(a: CExpr, b: CExpr, mut ca: CExpr, mut cb: CExpr) raises -> CType:
    """Bring two numeric operands to one type; `ca` and `cb` receive them cast."""
    var ta = _uintish(a.ty())
    var tb = _uintish(b.ty())
    var la = a.is_literal()
    var lb = b.is_literal()
    # bit and bool hold plain non-negative values: an int takes them as they are
    if a.ty().tag != ta.tag and tb.tag == CT.INT:
        ta = int_type(max(tb.width, ta.width + 1))
    elif b.ty().tag != tb.tag and ta.tag == CT.INT:
        tb = int_type(max(ta.width, tb.width + 1))
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
    var ta = a.ty()
    var tb = b.ty()
    var ca = CExpr()
    var cb = CExpr()
    if a.is_literal() and b.is_literal() and ta.tag == CT.INT and tb.tag == CT.INT:
        var x = a.const_int()
        var y = b.const_int()
        if op == CN.DIV and y != 0 and x % y != 0:
            return CExpr.literal_float(Float64(x) / Float64(y))
        if op == CN.POW and y < 0:
            return CExpr.literal_float(Float64(x) ** Float64(y))
    if op == CN.SHL or op == CN.SHR:
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
        var t = _common(a, b, ca, cb)
        return _pair(op, t, ca, cb)
    if op >= CN.EQ and op <= CN.GE:
        if ta.tag == CT.BOOL and tb.tag == CT.BOOL:
            if op != CN.EQ and op != CN.NE:
                raise Error("QM-E580: bool values are compared with == and != only. Hint: cast to uint.")
            return _pair(op, bool_type(), a, b)
        _ = _common(a, b, ca, cb)
        return _pair(op, bool_type(), ca, cb)
    # arithmetic
    if ta.tag == CT.ANGLE or tb.tag == CT.ANGLE:
        return _angle_arith(op, a, b)
    if op == CN.POW:
        if ta.tag == CT.FLOAT or tb.tag == CT.FLOAT:
            var t = _common(a, b, ca, cb)
            return _pair(op, t, ca, cb)
        # the exponent keeps its own type: its sign is read at run time
        var base = _to_common(a, _uintish(ta))
        return _pair(op, base.ty(), base, _exponent(b))
    var t = _common(a, b, ca, cb)
    return _pair(op, t, ca, cb)


def _to_shift_count(b: CExpr) raises -> CExpr:
    if b.ty().tag == CT.INT:
        return ce_cast(b, uint_type(64))
    return b.copy()


def _exponent(b: CExpr) raises -> CExpr:
    if b.ty().tag == CT.BOOL or b.ty().tag == CT.BIT:
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
    var t = _uintish(a.ty())
    if a.is_literal():
        var n = a.root()
        if t.tag == CT.FLOAT:
            return CExpr.literal_float(-n.f)
        if t.tag == CT.UINT and n.i != -9223372036854775807 - 1:
            raise Error("QM-E580: the literal does not fit a 64-bit int once negated. Hint: use a smaller value.")
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
