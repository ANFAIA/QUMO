"""Logical-level IR: programs written in an architecture's logical gate set.

This is the middle of Guppy FT's three abstraction levels
(computational -> **logical** -> physical). A `LogicalProgram` is a flat list
of `LogicalInstr` acting on *logical qubit handles* (fresh ids from `alloc()`,
like Guppy FT's `steane_logical.Qubit()`), with classical bits for decoded
measurement outcomes and the same classical-condition model as
`qumo.ir.circuit` (single bit `c_if`, AND of bits `c_if_register`).

Advanced users write logical programs directly (Guppy FT notebook
"Writing logical programs directly"):

```mojo
var lp = LogicalProgram("custom_t")
var q = lp.alloc()          # |0>_L
lp.h(q)
var m = lp.prep_t()         # T|+>_L magic state
lp.qec_cycle(m)             # explicit QEC cycles before injection
lp.qec_cycle(q)
lp.inject_t(q, m)           # consumes m
lp.measure(q, 0)
```

and `implement_ops(arch, lp)` links every logical op to its physical gadget.
`compile(arch, circuit)` produces the same IR from a computational circuit.

Ops (LOp): ALLOC, FREE, X, Y, Z, H, S, SDG, CX, CZ, SWAP (free relabel),
T, TDG (composite: magic-state injection on Steane, a native rotation on
Iceberg), RZ, RX, RZZ, RXX (native rotations, Iceberg), MEASURE (Z basis,
destructive for the logical qubit, decoded), QEC (QEC cycle on the qubit's
block), PREP_T (fresh T|+>_L), INJECT_T / INJECT_TDG (consume a magic qubit),
BARRIER.

Idea from Guppy FT (Apache-2.0), files src/guppyft/code/steane/logical.py and
src/guppyft/code/iceberg/logical.py (logical APIs) and the HUGR extensions
under extensions/src/*/ops.rs; re-derived as a QUMO-native IR.

Error codes: QM-E727 (malformed logical program).
"""

from qumo.ai.json import JsonWriter
from qumo.ft.classical import CExpr


struct LOp:
    comptime ALLOC = 0
    comptime FREE = 1
    comptime X = 2
    comptime Y = 3
    comptime Z = 4
    comptime H = 5
    comptime S = 6
    comptime SDG = 7
    comptime CX = 8
    comptime CZ = 9
    comptime SWAP = 10
    comptime T = 11
    comptime TDG = 12
    comptime RZ = 13
    comptime RX = 14
    comptime RZZ = 15
    comptime RXX = 16
    comptime MEASURE = 17
    comptime QEC = 18
    comptime PREP_T = 19
    comptime INJECT_T = 20
    comptime INJECT_TDG = 21
    comptime BARRIER = 22
    comptime CSET = 23  # classical assignment: clbit := cexprs[q0] (XOR form over logical clbits)
    comptime NUM = 24


def lop_name(k: Int) -> String:
    var names: List[String] = [
        "alloc", "free", "x", "y", "z", "h", "s", "sdg", "cx", "cz", "swap", "t", "tdg",
        "rz", "rx", "rzz", "rxx", "measure", "qec_cycle", "prep_t", "inject_t", "inject_tdg", "barrier", "cset",
    ]
    if k >= 0 and k < len(names):
        return names[k]
    return "?"


def lop_arity(k: Int) -> Int:
    if k == LOp.CX or k == LOp.CZ or k == LOp.SWAP or k == LOp.RZZ or k == LOp.RXX or k == LOp.INJECT_T or k == LOp.INJECT_TDG:
        return 2
    if k == LOp.BARRIER or k == LOp.CSET:
        return 0
    return 1


def lop_is_pauli(k: Int) -> Bool:
    return k == LOp.X or k == LOp.Y or k == LOp.Z


def lop_is_clifford_gate(k: Int) -> Bool:
    return k == LOp.X or k == LOp.Y or k == LOp.Z or k == LOp.H or k == LOp.S or k == LOp.SDG or k == LOp.CX or k == LOp.CZ or k == LOp.SWAP


@fieldwise_init
struct LogicalInstr(ImplicitlyCopyable, Writable):
    var kind: Int
    var q0: Int
    var q1: Int
    var angle: Float64
    var clbit: Int
    var cond: Int  # -1 = unconditional; first bit of a register condition otherwise
    var cond_val: Int
    var cond_reg: Int  # offset into LogicalProgram.cond_table or -1

    @staticmethod
    def make(kind: Int, q0: Int, q1: Int = -1, angle: Float64 = 0.0) -> LogicalInstr:
        return LogicalInstr(kind, q0, q1, angle, -1, -1, 0, -1)

    def write_to(self, mut w: Some[Writer]):
        if self.cond_reg >= 0:
            w.write("if(creg#", self.cond_reg, ") ")
        elif self.cond >= 0:
            w.write("if(c", self.cond, "==", self.cond_val, ") ")
        w.write(lop_name(self.kind))
        if self.kind == LOp.RZ or self.kind == LOp.RX or self.kind == LOp.RZZ or self.kind == LOp.RXX:
            w.write("(", self.angle, ")")
        var a = lop_arity(self.kind)
        if a >= 1:
            w.write(" L", self.q0)
        if a >= 2:
            w.write(", L", self.q1)
        if self.kind == LOp.MEASURE or self.kind == LOp.CSET:
            w.write(" -> c", self.clbit)


struct LogicalProgram(Copyable, Writable, Sized):
    var name: String
    var num_logical: Int  # handles issued
    var num_clbits: Int
    var ops: List[LogicalInstr]
    var cond_table: List[Int]  # [count, bit0, val0, ...] per register condition
    var notes: List[String]  # compile notes (decompositions, approximations)
    var cexprs: List[CExpr]  # CSET payloads, over logical clbits

    def __init__(out self, name: String = "logical"):
        self.name = name
        self.num_logical = 0
        self.num_clbits = 0
        self.ops = List[LogicalInstr]()
        self.cond_table = List[Int]()
        self.notes = List[String]()
        self.cexprs = List[CExpr]()

    def __len__(self) -> Int:
        return len(self.ops)

    # ---------------------------------------------------------------- builders
    def alloc(mut self) -> Int:
        """Fresh logical qubit in |0>_L (Guppy FT `Qubit()`)."""
        var q = self.num_logical
        self.num_logical += 1
        self.ops.append(LogicalInstr.make(LOp.ALLOC, q))
        return q

    def free(mut self, q: Int):
        self.ops.append(LogicalInstr.make(LOp.FREE, q))

    def gate(mut self, kind: Int, q0: Int, q1: Int = -1, angle: Float64 = 0.0):
        self.ops.append(LogicalInstr.make(kind, q0, q1, angle))

    def x(mut self, q: Int): self.gate(LOp.X, q)
    def y(mut self, q: Int): self.gate(LOp.Y, q)
    def z(mut self, q: Int): self.gate(LOp.Z, q)
    def h(mut self, q: Int): self.gate(LOp.H, q)
    def s(mut self, q: Int): self.gate(LOp.S, q)
    def sdg(mut self, q: Int): self.gate(LOp.SDG, q)
    def t(mut self, q: Int): self.gate(LOp.T, q)
    def tdg(mut self, q: Int): self.gate(LOp.TDG, q)
    def cx(mut self, c: Int, t: Int): self.gate(LOp.CX, c, t)
    def cz(mut self, a: Int, b: Int): self.gate(LOp.CZ, a, b)
    def swap(mut self, a: Int, b: Int): self.gate(LOp.SWAP, a, b)
    def rz(mut self, theta: Float64, q: Int): self.gate(LOp.RZ, q, -1, theta)
    def rx(mut self, theta: Float64, q: Int): self.gate(LOp.RX, q, -1, theta)
    def rzz(mut self, theta: Float64, a: Int, b: Int): self.gate(LOp.RZZ, a, b, theta)
    def rxx(mut self, theta: Float64, a: Int, b: Int): self.gate(LOp.RXX, a, b, theta)
    def qec_cycle(mut self, q: Int): self.gate(LOp.QEC, q)
    def barrier(mut self): self.gate(LOp.BARRIER, -1)

    def measure(mut self, q: Int, c: Int):
        """Destructive Z measurement; the decoded outcome goes to clbit c."""
        var g = LogicalInstr.make(LOp.MEASURE, q)
        g.clbit = c
        if c >= self.num_clbits:
            self.num_clbits = c + 1
        self.ops.append(g)

    def set_bit(mut self, c: Int, e: CExpr):
        """Classical assignment c := e (e is an XOR of clbits / minterms / 1 over
        logical clbits). `compile` emits it for classical-only qubits."""
        var g = LogicalInstr.make(LOp.CSET, len(self.cexprs))
        g.clbit = c
        self.cexprs.append(e.copy())
        if c >= self.num_clbits:
            self.num_clbits = c + 1
        self.ops.append(g)

    def prep_t(mut self) -> Int:
        """Fresh logical qubit in T|+>_L (magic state)."""
        var q = self.num_logical
        self.num_logical += 1
        self.ops.append(LogicalInstr.make(LOp.PREP_T, q))
        return q

    def inject_t(mut self, q: Int, magic: Int):
        self.gate(LOp.INJECT_T, q, magic)

    def inject_tdg(mut self, q: Int, magic: Int):
        self.gate(LOp.INJECT_TDG, q, magic)

    def c_if(mut self, clbit: Int, value: Int):
        """Condition the most recent instruction on a decoded classical bit."""
        self.ops[len(self.ops) - 1].cond = clbit
        self.ops[len(self.ops) - 1].cond_val = value

    def c_if_register(mut self, bits: List[Int], values: List[Int]) raises:
        if len(bits) == 0 or len(bits) != len(values):
            raise Error("QM-E727: register condition needs matching non-empty bit/value lists. Hint: one value (0/1) per bit.")
        var off = len(self.cond_table)
        self.cond_table.append(len(bits))
        for i in range(len(bits)):
            self.cond_table.append(bits[i])
            self.cond_table.append(values[i])
        ref g = self.ops[len(self.ops) - 1]
        g.cond = bits[0]
        g.cond_val = values[0]
        g.cond_reg = off

    def condition_bits(self, g: LogicalInstr) -> List[Int]:
        var out = List[Int]()
        if g.cond < 0:
            return out^
        if g.cond_reg < 0:
            out.append(g.cond)
            return out^
        for i in range(self.cond_table[g.cond_reg]):
            out.append(self.cond_table[g.cond_reg + 1 + 2 * i])
        return out^

    def condition_vals(self, g: LogicalInstr) -> List[Int]:
        var out = List[Int]()
        if g.cond < 0:
            return out^
        if g.cond_reg < 0:
            out.append(g.cond_val)
            return out^
        for i in range(self.cond_table[g.cond_reg]):
            out.append(self.cond_table[g.cond_reg + 2 + 2 * i])
        return out^

    # ---------------------------------------------------------------- analysis
    def count(self, kind: Int) -> Int:
        var n = 0
        for g in self.ops:
            if g.kind == kind:
                n += 1
        return n

    def t_count(self) -> Int:
        return self.count(LOp.T) + self.count(LOp.TDG) + self.count(LOp.INJECT_T) + self.count(LOp.INJECT_TDG)

    def validate(self) raises:
        """Use-after-free / unknown handle / double alloc checks (QM-E727)."""
        var state = List[Int](length=self.num_logical, fill=0)  # 0 unborn, 1 live, 2 dead
        for i in range(len(self.ops)):
            var g = self.ops[i]
            if g.kind == LOp.ALLOC or g.kind == LOp.PREP_T:
                if g.q0 < 0 or g.q0 >= self.num_logical or state[g.q0] != 0:
                    raise Error("QM-E727: logical instruction #" + String(i) + " re-allocates handle L" + String(g.q0) + ". Hint: use alloc()/prep_t() to obtain fresh handles.")
                state[g.q0] = 1
                continue
            var a = lop_arity(g.kind)
            for j in range(a):
                var q = g.q0 if j == 0 else g.q1
                if q < 0 or q >= self.num_logical or state[q] != 1:
                    raise Error("QM-E727: logical instruction #" + String(i) + " '" + lop_name(g.kind) + "' uses handle L" + String(q) + " which is not live. Hint: allocate before use; MEASURE, FREE and INJECT_T (magic operand) consume a qubit.")
            if a == 2 and g.q0 == g.q1:
                raise Error("QM-E727: logical instruction #" + String(i) + " repeats handle L" + String(g.q0) + ".")
            if g.cond >= 0 and (g.kind == LOp.ALLOC or g.kind == LOp.PREP_T or g.kind == LOp.FREE or g.kind == LOp.MEASURE or g.kind == LOp.QEC):
                raise Error("QM-E727: logical instruction #" + String(i) + " '" + lop_name(g.kind) + "' cannot be classically conditioned. Hint: condition gates only.")
            if g.kind == LOp.MEASURE or g.kind == LOp.FREE:
                state[g.q0] = 2
            if g.kind == LOp.INJECT_T or g.kind == LOp.INJECT_TDG:
                state[g.q1] = 2

    def summary_json(self, mut w: JsonWriter):
        w.begin_object()
        w.kv_string("name", self.name)
        w.kv_int("logical_qubits", self.num_logical)
        w.kv_int("logical_ops", len(self.ops))
        w.kv_int("clbits", self.num_clbits)
        w.key("op_counts")
        w.begin_object()
        for k in range(LOp.NUM):
            var c = self.count(k)
            if c > 0:
                w.kv_int(lop_name(k), c)
        w.end_object()
        w.kv_string_list("notes", self.notes)
        w.end_object()

    def write_to(self, mut w: Some[Writer]):
        w.write("logical program ", self.name, " handles=", self.num_logical, " clbits=", self.num_clbits, "\n")
        for g in self.ops:
            w.write("  ", g, "\n")
