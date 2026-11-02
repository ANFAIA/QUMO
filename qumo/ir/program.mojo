"""`Program`: quantum circuits with classical variables and control flow.

A `Circuit` is a flat list without jumps, and every walker of `.ops` skips the
opcodes it does not know, so a `while` inside a `Circuit` would be dropped in
silence. Loops therefore live in a separate type, one the compiler rejects on any
route that was not adapted to it.

    body   arena `Circuit`: quantum blocks are ranges [lo, hi) of its ops, so the
           side tables (conditions, Kraus, noise) stay valid and no opcode is new
    tape   structured statements in preorder with `end` links: OPS ASSIGN IF ELSE
           WHILE FOR SWITCH CASE BREAK CONTINUE END. There is no goto
    vars   classical variables; `nodes` holds their expressions (qumo.ir.classical)

`cfg_of` derives the control-flow graph from the tape; the interpreter and the
compiler's verifier both run on it. `check_program` does what a `Circuit`'s
validate does, plus the opcode whitelist and the definite-assignment analysis.

A gate angle known only at run time is a symbolic `Param` on a slot of
`body.param_names` with a reserved name. The block that uses it sees one value:
an ASSIGN ends the block, so `apply_instruction(sv, g, params)` is unchanged.
Bits stay in each simulator; `bit` variables are clbits.
"""

from std.math import isinf, isnan

from qumo.ir.circuit import Circuit, Instruction
from qumo.ir.gates import Op, op_name, op_is_unitary, op_is_noise
from qumo.ir.param import Param
from qumo.ir.classical import (
    CExpr, CNode, CN, CT, CType, bit_type, int_type, ce_bool, ce_cast, never_negative,
)


comptime MAX_NESTING = 256  # blocks and expressions, in source and in a program file


struct SK:
    """Statement kinds of the tape."""

    comptime OPS = 0  # a = first op, b = end op
    comptime ASSIGN = 1  # a = target variable, b = expression
    comptime IF = 2  # a = expression, alt = ELSE or -1, end = END
    comptime ELSE = 3  # end = END
    comptime WHILE = 4  # a = expression, end = END
    comptime FOR = 5  # a = variable, b = offset into for_ranges, end = END
    comptime SWITCH = 6  # a = expression, alt = first CASE, end = END
    comptime CASE = 7  # a, b = range of case_vals (a = -1: default), alt = next CASE or END, end = END
    comptime BREAK = 8  # end = target loop
    comptime CONTINUE = 9  # end = target loop
    comptime END = 10  # end = opener


@fieldwise_init
struct Stmt(ImplicitlyCopyable):
    var kind: Int
    var a: Int
    var b: Int
    var alt: Int
    var end: Int
    var line: Int  # source line, 0 when built by API


@fieldwise_init
struct CVar(Copyable):
    var name: String
    var ty: CType
    var store: Int  # index in the shot's ints / reals; first clbit for `bit`
    var slot: Int  # body.param_names index when used as a runtime angle, else -1
    var is_output: Bool


@fieldwise_init
struct CReg(Copyable):
    var name: String
    var first: Int
    var width: Int


struct FORR:
    """Fields of one entry of `Program.for_ranges`."""

    comptime START = 0  # expression, or -1 for a set
    comptime STEP = 1  # expression
    comptime STOP = 2  # expression
    comptime STORE = 3  # 3 hidden ints: current value or set position, step, stop
    comptime SET = 4  # offset in case_vals of [count, values...], or -1
    comptime UNSIGNED = 5  # 1: the bounds cannot be negative, so they compare unsigned
    comptime SIZE = 6


struct Program(Copyable, Writable):
    var body: Circuit  # arena; no public API returns it
    var tape: List[Stmt]
    var vars: List[CVar]
    var nodes: List[CNode]
    var expr_lo: List[Int]
    var expr_root: List[Int]
    var for_ranges: List[Int]
    var case_vals: List[Int]
    var cregs: List[CReg]
    var user_params: List[Int]  # body.param_names indices that are `input`
    var num_store: Int  # size of the shot's ints and reals

    def __init__(out self, var body: Circuit):
        self.body = body^
        self.tape = List[Stmt]()
        self.vars = List[CVar]()
        self.nodes = List[CNode]()
        self.expr_lo = List[Int]()
        self.expr_root = List[Int]()
        self.for_ranges = List[Int]()
        self.case_vals = List[Int]()
        self.cregs = List[CReg]()
        self.user_params = List[Int]()
        self.num_store = 0

    @staticmethod
    def from_circuit(c: Circuit) -> Program:
        """The same circuit as one block; `sample_program` then equals `sample`."""
        var p = Program(c.copy())
        if len(c.ops) > 0:
            p.tape.append(Stmt(SK.OPS, 0, len(c.ops), -1, -1, 0))
        if c.num_clbits > 0:
            p.cregs.append(CReg("c", 0, c.num_clbits))
        for i in range(c.num_params()):
            p.user_params.append(i)
        return p^

    # ---------------------------------------------------------------- reading
    def name(self) -> String:
        return self.body.name

    def num_qubits(self) -> Int:
        return self.body.num_qubits

    def num_clbits(self) -> Int:
        return self.body.num_clbits

    def num_params(self) -> Int:
        return len(self.user_params)

    def has_runtime_region(self) -> Bool:
        """Anything but straight-line blocks on the tape."""
        for s in self.tape:
            if s.kind != SK.OPS:
                return True
        return False

    def try_flatten(self) -> Optional[Circuit]:
        """The plain circuit of a program without runtime regions, else None."""
        if self.has_runtime_region():
            return None
        for v in self.vars:
            if v.is_output and v.ty.tag != CT.BIT:
                return None
        return self.body.copy()

    def user_param_names(self) -> List[String]:
        var out = List[String]()
        for i in self.user_params:
            out.append(self.body.param_names[i])
        return out^

    def var_index(self, name: String) -> Int:
        for i in range(len(self.vars)):
            if self.vars[i].name == name:
                return i
        return -1

    # --------------------------------------------------------------- building
    def add_var(mut self, name: String, ty: CType, is_output: Bool = False) -> Int:
        """A scalar variable with its own storage."""
        self.vars.append(CVar(name, ty, self.num_store, -1, is_output))
        self.num_store += 1
        return len(self.vars) - 1

    def add_bit_var(mut self, name: String, first: Int, width: Int, is_output: Bool = False) -> Int:
        """A view of clbits first..first+width-1 as an assignable `bit[width]`."""
        self.vars.append(CVar(name, bit_type(width), first, -1, is_output))
        return len(self.vars) - 1

    def add_expr(mut self, e: CExpr) -> Int:
        """Move an expression into the program; returns its id."""
        var base = len(self.nodes)
        for n in e.nodes:
            var c = n
            if c.op >= CN.NEG:
                c.x += base
                if c.y >= 0:
                    c.y += base
            self.nodes.append(c)
        self.expr_lo.append(base)
        self.expr_root.append(len(self.nodes) - 1)
        return len(self.expr_lo) - 1

    def slot_of(mut self, var_id: Int) -> Int:
        """The parameter slot that mirrors a variable used as a gate angle."""
        if self.vars[var_id].slot < 0:
            var p = self.body.parameter(_slot_name(var_id, self.vars[var_id].name))
            self.vars[var_id].slot = p.index
        return self.vars[var_id].slot

    # ------------------------------------------------------------------ write
    def write_to(self, mut w: Some[Writer]):
        w.write("program ", self.body.name, " qubits=", self.body.num_qubits, " clbits=", self.body.num_clbits,
                " vars=", len(self.vars), " statements=", len(self.tape))
        w.write("\n")
        var depth = 1
        for s in self.tape:
            if s.kind == SK.END or s.kind == SK.ELSE or s.kind == SK.CASE:
                depth -= 1
            for _ in range(depth):
                w.write("  ")
            w.write(_kind_name(s.kind))
            if s.kind == SK.OPS:
                w.write(" ", s.b - s.a, " ops")
            elif s.kind == SK.ASSIGN:
                w.write(" ", self.vars[s.a].name)
            w.write("\n")
            if s.kind == SK.IF or s.kind == SK.ELSE or s.kind == SK.WHILE or s.kind == SK.FOR or s.kind == SK.SWITCH or s.kind == SK.CASE:
                depth += 1


def slot_prefix() -> String:
    """Parameter names starting with this are runtime-angle slots, not inputs."""
    return "__qumo_cv"


def _slot_name(var_id: Int, name: String) -> String:
    return slot_prefix() + String(var_id) + "_" + name


def _kind_name(k: Int) -> String:
    if k == SK.OPS: return "ops"
    if k == SK.ASSIGN: return "assign"
    if k == SK.IF: return "if"
    if k == SK.ELSE: return "else"
    if k == SK.WHILE: return "while"
    if k == SK.FOR: return "for"
    if k == SK.SWITCH: return "switch"
    if k == SK.CASE: return "case"
    if k == SK.BREAK: return "break"
    if k == SK.CONTINUE: return "continue"
    return "end"


# ---------------------------------------------------------------------------
# Tape construction, shared by ProgramBuilder and the OpenQASM parser
# ---------------------------------------------------------------------------
struct TapeBuilder(Movable):
    """Appends statements; `ops_end` is the arena's op count at each call, so the
    ops since the last control statement become one OPS block."""

    var tape: List[Stmt]
    var open: List[Int]  # opener of each open region, innermost last
    var ops_lo: Int

    def __init__(out self):
        self.tape = List[Stmt]()
        self.open = List[Int]()
        self.ops_lo = 0

    def flush(mut self, ops_end: Int, line: Int):
        """Close the current block of quantum ops."""
        if ops_end > self.ops_lo:
            self.tape.append(Stmt(SK.OPS, self.ops_lo, ops_end, -1, -1, line))
        self.ops_lo = ops_end

    def _push(mut self, kind: Int, a: Int, b: Int, alt: Int, end: Int, line: Int) -> Int:
        self.tape.append(Stmt(kind, a, b, alt, end, line))
        return len(self.tape) - 1

    def assign(mut self, target: Int, expr: Int, ops_end: Int, line: Int):
        self.flush(ops_end, line)
        _ = self._push(SK.ASSIGN, target, expr, -1, -1, line)

    def begin(mut self, kind: Int, a: Int, b: Int, ops_end: Int, line: Int):
        """Open IF, WHILE, FOR or SWITCH."""
        self.flush(ops_end, line)
        self.open.append(self._push(kind, a, b, -1, -1, line))

    def else_(mut self, ops_end: Int, line: Int) raises:
        self.flush(ops_end, line)
        if len(self.open) == 0 or self.tape[self.open[len(self.open) - 1]].kind != SK.IF:
            raise Error("QM-E583: else without an open if. Hint: open the region with begin_if.")
        var op = self.open[len(self.open) - 1]
        if self.tape[op].alt >= 0:
            raise Error("QM-E583: an if takes one else. Hint: close the region with end.")
        var at = self._push(SK.ELSE, -1, -1, -1, -1, line)
        self.tape[op].alt = at

    def case_(mut self, lo: Int, hi: Int, ops_end: Int, line: Int) raises:
        """A CASE of the open switch: values case_vals[lo:hi], or default with lo = -1."""
        self.flush(ops_end, line)
        if len(self.open) == 0 or self.tape[self.open[len(self.open) - 1]].kind != SK.SWITCH:
            raise Error("QM-E583: case without an open switch. Hint: open the region with begin_switch.")
        var at = self._push(SK.CASE, lo, hi, -1, -1, line)
        var i = self.open[len(self.open) - 1]
        var last = i
        while True:
            var nx = self.tape[last].alt
            if nx < 0:
                break
            last = nx
        self.tape[last].alt = at

    def jump(mut self, kind: Int, ops_end: Int, line: Int) raises:
        """BREAK or CONTINUE to the innermost loop."""
        self.flush(ops_end, line)
        var i = len(self.open) - 1
        while i >= 0:
            var k = self.tape[self.open[i]].kind
            if k == SK.WHILE or k == SK.FOR:
                _ = self._push(kind, -1, -1, -1, self.open[i], line)
                return
            i -= 1
        raise Error("QM-E583: " + _kind_name(kind) + " outside a loop. Hint: it must sit inside a while or for.")

    def end(mut self, ops_end: Int, line: Int) raises:
        self.flush(ops_end, line)
        if len(self.open) == 0:
            raise Error("QM-E583: end without an open region. Hint: every end closes one begin_if, begin_while, begin_for or begin_switch.")
        var op = self.open.pop()
        var at = self._push(SK.END, -1, -1, -1, op, line)
        self.tape[op].end = at
        var k = self.tape[op].kind
        if k == SK.IF and self.tape[op].alt >= 0:
            self.tape[self.tape[op].alt].end = at
        elif k == SK.SWITCH:
            var c = self.tape[op].alt
            while c >= 0:
                self.tape[c].end = at
                var nx = self.tape[c].alt
                if nx < 0:
                    self.tape[c].alt = at
                    break
                c = nx


# ---------------------------------------------------------------------------
# Control-flow graph
# ---------------------------------------------------------------------------
struct TERM:
    comptime JMP = 0  # to t0
    comptime BR = 1  # to t0 if `cond` holds, else t1; cond < 0: FOR at tape -1 - cond
    comptime SWITCH = 2  # value of `cond`: keys in sw_keys[t0:t1], default sw_bb[t1]
    comptime HALT = 3


@fieldwise_init
struct BB(ImplicitlyCopyable):
    """A basic block: the tape range [lo, hi) holds only OPS and ASSIGN."""

    var lo: Int
    var hi: Int
    var term: Int
    var cond: Int
    var t0: Int
    var t1: Int
    var back: Bool  # the edge to t0 closes a loop
    var init: Int  # after the block: start the FOR at this tape index (-1 none)
    var step: Int  # after the block: advance the FOR at this tape index (-1 none)
    var line: Int


struct Cfg(Copyable):
    var blocks: List[BB]
    var sw_keys: List[Int]  # SWITCH case values
    var sw_bb: List[Int]  # the block of each key; one more entry holds the default
    var slot_ops: List[List[Int]]  # by tape index: the ops of an OPS block with runtime angles

    def __init__(out self):
        self.blocks = List[BB]()
        self.sw_keys = List[Int]()
        self.sw_bb = List[Int]()
        self.slot_ops = List[List[Int]]()


comptime _AT = 0  # the block that starts at a tape position
comptime _HEAD = 1  # the test of the loop at a tape position
comptime _LATCH = 2  # the advance of the FOR at a tape position


def cfg_of(p: Program) -> Cfg:
    """Basic blocks of the tape. Targets are first recorded as a kind and a tape
    position, then resolved once every block exists."""
    var n = len(p.tape)
    var cfg = Cfg()
    var at = List[Int](length=n + 2, fill=-1)
    var head = List[Int](length=n + 1, fill=-1)
    var latch = List[Int](length=n + 1, fill=-1)
    var scale = n + 2
    var lo = 0
    var mirrored = List[Bool](length=p.body.num_params(), fill=False)
    for v in p.vars:
        if v.slot >= 0:
            mirrored[v.slot] = True
    for i in range(n):
        var s = p.tape[i]
        var k = s.kind
        var reads = List[Int]()
        if k == SK.OPS:
            for j in range(s.a, s.b):
                ref g = p.body.ops[j]
                if (g.p0.index >= 0 and mirrored[g.p0.index]) or (g.p1.index >= 0 and mirrored[g.p1.index]) or (g.p2.index >= 0 and mirrored[g.p2.index]):
                    reads.append(j)
        cfg.slot_ops.append(reads^)
        if k == SK.OPS or k == SK.ASSIGN:
            continue
        var close = True
        var b = BB(lo, i, TERM.JMP, -1, (scale * _AT + i + 1), -1, False, -1, -1, s.line)
        if k == SK.IF:
            b.term = TERM.BR
            b.cond = s.a
            b.t0 = (scale * _AT + i + 1)
            b.t1 = (scale * _AT + (s.alt if s.alt >= 0 else s.end) + 1)
        elif k == SK.ELSE or k == SK.CASE:
            if k == SK.CASE and lo == i and i > 0 and p.tape[i - 1].kind == SK.SWITCH:
                close = False  # the first CASE only labels a position
            b.t0 = (scale * _AT + s.end + 1)
        elif k == SK.WHILE or k == SK.FOR:
            if k == SK.FOR:
                b.init = i
            b.t0 = (scale * _HEAD + i)
        elif k == SK.SWITCH:
            b.term = TERM.SWITCH
            b.cond = s.a
            b.t0 = len(cfg.sw_keys)
            var c = s.alt
            var dflt = s.end + 1
            while c >= 0 and c < n and p.tape[c].kind == SK.CASE:
                var cs = p.tape[c]
                if cs.a < 0:
                    dflt = c + 1
                else:
                    for j in range(cs.a, cs.b):
                        cfg.sw_keys.append(p.case_vals[j])
                        cfg.sw_bb.append((scale * _AT + c + 1))
                c = cs.alt
            b.t1 = len(cfg.sw_keys)
            cfg.sw_keys.append(0)  # keeps sw_keys and sw_bb aligned
            cfg.sw_bb.append((scale * _AT + dflt))
        elif k == SK.BREAK:
            b.t0 = (scale * _AT + p.tape[s.end].end + 1)
        elif k == SK.CONTINUE:
            var loop = p.tape[s.end]
            b.t0 = scale * (_LATCH if loop.kind == SK.FOR else _HEAD) + s.end
            b.back = loop.kind == SK.WHILE
        else:  # END
            var op = p.tape[s.end]
            if op.kind == SK.WHILE:
                b.t0 = (scale * _HEAD + s.end)
                b.back = True
            elif op.kind == SK.FOR:
                b.t0 = (scale * _LATCH + s.end)
            else:
                b.t0 = (scale * _AT + i + 1)
        if close:
            if at[lo] < 0:
                at[lo] = len(cfg.blocks)
            cfg.blocks.append(b)
        if k == SK.WHILE or k == SK.FOR:
            head[i] = len(cfg.blocks)
            var cond = s.a if k == SK.WHILE else -1 - i
            cfg.blocks.append(BB(i, i, TERM.BR, cond, (scale * _AT + i + 1), (scale * _AT + s.end + 1), False, -1, -1, s.line))
        if k == SK.END and p.tape[s.end].kind == SK.FOR:
            latch[s.end] = len(cfg.blocks)
            cfg.blocks.append(BB(i, i, TERM.JMP, -1, (scale * _HEAD + s.end), -1, True, -1, s.end, s.line))
        lo = i + 1
    if at[lo] < 0:
        at[lo] = len(cfg.blocks)
    cfg.blocks.append(BB(lo, n, TERM.HALT, -1, -1, -1, False, -1, -1, 0))
    for ref b in cfg.blocks:
        if b.term == TERM.JMP or b.term == TERM.BR:
            b.t0 = _resolve(b.t0, scale, at, head, latch)
        if b.term == TERM.BR:
            b.t1 = _resolve(b.t1, scale, at, head, latch)
    for ref t in cfg.sw_bb:
        t = _resolve(t, scale, at, head, latch)
    return cfg^


def _resolve(t: Int, scale: Int, at: List[Int], head: List[Int], latch: List[Int]) -> Int:
    var where = t // scale
    var pos = t % scale
    if where == _HEAD:
        return head[pos]
    if where == _LATCH:
        return latch[pos]
    return at[pos]


def _reads(p: Program, expr: Int, mut out: List[Int]):
    """Storage indices of the scalar variables an expression reads."""
    for i in range(p.expr_lo[expr], p.expr_root[expr] + 1):
        if p.nodes[i].op == CN.VAR:
            out.append(p.nodes[i].x)


def slot_stores(p: Program) -> List[Int]:
    """By parameter index: the storage of the variable that slot mirrors, or -1."""
    var out = List[Int](length=p.body.num_params(), fill=-1)
    for v in p.vars:
        if v.slot >= 0:
            out[v.slot] = v.store
    return out^


def _gate_reads(p: Program, lo: Int, hi: Int, slots: List[Int], mut out: List[Int]):
    """Variables the gates of ops [lo, hi) read through their parameter slots."""
    for i in range(lo, hi):
        ref g = p.body.ops[i]
        if g.p0.index >= 0 and slots[g.p0.index] >= 0:
            out.append(slots[g.p0.index])
        if g.p1.index >= 0 and slots[g.p1.index] >= 0:
            out.append(slots[g.p1.index])
        if g.p2.index >= 0 and slots[g.p2.index] >= 0:
            out.append(slots[g.p2.index])


def _block_reads(p: Program, b: BB, i: Int, slots: List[Int], mut out: List[Int]):
    """Reads of tape statement i, or of the terminator when i = b.hi."""
    if i < b.hi:
        if p.tape[i].kind == SK.ASSIGN:
            _reads(p, p.tape[i].b, out)
        elif p.tape[i].kind == SK.OPS:
            _gate_reads(p, p.tape[i].a, p.tape[i].b, slots, out)
        return
    if b.init >= 0:
        var off = p.tape[b.init].b
        for f in range(FORR.START, FORR.STOP + 1):
            var e = p.for_ranges[off + f]
            if e >= 0:
                _reads(p, e, out)
    if (b.term == TERM.BR or b.term == TERM.SWITCH) and b.cond >= 0:
        _reads(p, b.cond, out)


def _check_assigned(p: Program, cfg: Cfg) raises:
    """Definite assignment over the graph (QM-E581): a variable is read only
    where every path to that point assigned it. `bit`s start at 0 and are exempt."""
    var nb = len(cfg.blocks)
    var ns = p.num_store
    if ns == 0:
        return
    var preds = List[List[Int]]()
    for _ in range(nb):
        preds.append(List[Int]())
    for b in range(nb):
        ref bb = cfg.blocks[b]
        if bb.term == TERM.JMP or bb.term == TERM.BR:
            preds[bb.t0].append(b)
        if bb.term == TERM.BR:
            preds[bb.t1].append(b)
        if bb.term == TERM.SWITCH:
            for j in range(bb.t0, bb.t1 + 1):
                preds[cfg.sw_bb[j]].append(b)
    # IN sets start full, the entry empty, and shrink to the greatest fixed point
    var inn = List[List[Bool]]()
    for b in range(nb):
        inn.append(List[Bool](length=ns, fill=b != 0))
    var reachable = List[Bool](length=nb, fill=False)
    reachable[0] = True
    var slots = slot_stores(p)
    var changed = True
    while changed:
        changed = False
        for b in range(nb):
            if not reachable[b]:
                continue
            var cur = inn[b].copy()
            _apply_defs(p, cfg.blocks[b], cur)
            ref bb = cfg.blocks[b]
            var succs = List[Int]()
            if bb.term == TERM.JMP or bb.term == TERM.BR:
                succs.append(bb.t0)
            if bb.term == TERM.BR:
                succs.append(bb.t1)
            if bb.term == TERM.SWITCH:
                for j in range(bb.t0, bb.t1 + 1):
                    succs.append(cfg.sw_bb[j])
            for t in succs:
                if not reachable[t]:
                    reachable[t] = True
                    changed = True
                # a FOR assigns its variable on the edge into the body, not on exit
                var into_body = -1
                if bb.term == TERM.BR and bb.cond < 0 and t == bb.t0:
                    into_body = p.vars[p.tape[-1 - bb.cond].a].store
                for v in range(ns):
                    if inn[t][v] and not cur[v] and v != into_body:
                        inn[t][v] = False
                        changed = True
    for b in range(nb):
        if not reachable[b]:
            continue
        ref bb = cfg.blocks[b]
        var cur = inn[b].copy()
        for i in range(bb.lo, bb.hi + 1):
            var reads = List[Int]()
            _block_reads(p, bb, i, slots, reads)
            for r in reads:
                if not cur[r]:
                    _raise_unassigned(p, r, bb.line if i == bb.hi else p.tape[i].line)
            if i < bb.hi and p.tape[i].kind == SK.ASSIGN:
                var t = p.vars[p.tape[i].a].copy()
                if t.ty.tag != CT.BIT:
                    cur[t.store] = True
        if bb.term == TERM.HALT:
            for v in p.vars:
                if v.is_output and v.ty.tag != CT.BIT and not cur[v.store]:
                    raise Error("QM-E581: output '" + v.name + "' is not assigned on every path. Hint: initialize it where it is declared.")


def _apply_defs(p: Program, b: BB, mut cur: List[Bool]):
    for i in range(b.lo, b.hi):
        ref s = p.tape[i]
        if s.kind == SK.ASSIGN and p.vars[s.a].ty.tag != CT.BIT:
            cur[p.vars[s.a].store] = True


def _raise_unassigned(p: Program, store: Int, line: Int) raises:
    var name = String("a variable")
    for v in p.vars:
        if v.ty.tag != CT.BIT and v.store == store:
            name = "'" + v.name + "'"
    raise Error("QM-E581: " + name + " may be read before it is assigned (line " + String(line) + "). Hint: assign it on every path, for example where it is declared.")


def _reads_bits(first: List[Int], last: List[Int], lo: Int, hi: Int) -> Bool:
    for i in range(len(first)):
        if lo <= last[i] and first[i] <= hi:
            return True
    return False


def _can_leave(p: Program, at: Int) -> Bool:
    """Can the body of the WHILE at tape index `at` end it? A `break` can, and so can
    anything that writes a variable or a bit its condition reads."""
    ref head = p.tape[at]
    var stores = List[Int]()
    _reads(p, head.a, stores)
    var first = List[Int]()  # the condition's bit fields, [first[i], last[i]]
    var last = List[Int]()
    for i in range(p.expr_lo[head.a], p.expr_root[head.a] + 1):
        ref n = p.nodes[i]
        if n.op == CN.BITS:
            first.append(n.x)
            last.append(n.x + n.y - 1)
    for j in range(at + 1, head.end):
        ref s = p.tape[j]
        if s.kind == SK.BREAK and s.end == at:
            return True
        if s.kind == SK.OPS:
            for i in range(s.a, s.b):
                ref g = p.body.ops[i]
                if g.kind == Op.MEASURE and _reads_bits(first, last, g.clbit, g.clbit):
                    return True
        elif s.kind == SK.ASSIGN or s.kind == SK.FOR:
            ref v = p.vars[s.a]
            if v.ty.tag == CT.BIT:
                if _reads_bits(first, last, v.store, v.store + v.ty.width - 1):
                    return True
            else:
                for r in stores:
                    if r == v.store:
                        return True
    return False


def stuck_loops(p: Program) -> List[Int]:
    """Tape index of every WHILE that never runs or never ends: nothing in its
    body writes what the condition reads, and no `break` leaves it."""
    var out = List[Int]()
    for i in range(len(p.tape)):
        if p.tape[i].kind == SK.WHILE and not _can_leave(p, i):
            out.append(i)
    return out^


def is_block_op(k: Int) -> Bool:
    """Opcodes the program interpreter runs inside a block; anything else is
    QM-E484, because a walker that meets an unknown opcode skips it."""
    return op_is_unitary(k) or op_is_noise(k) or k == Op.MEASURE or k == Op.RESET or k == Op.BARRIER or k == Op.TICK


def check_program(p: Program) raises -> Cfg:
    """Everything a run or a compilation needs to hold; returns the graph."""
    p.body.validate()
    var runtime = p.has_runtime_region()
    for s in p.tape:
        if s.kind != SK.OPS:
            continue
        for i in range(s.a, s.b):
            ref g = p.body.ops[i]
            var k = g.kind
            for f in range(3):
                var q = g.param(f)
                if isnan(q.offset) or isinf(q.offset) or isnan(q.coeff):
                    raise Error("QM-E482: gate '" + op_name(k) + "' has a constant angle that is NaN or infinite (line " + String(s.line) + "). Hint: check the expression that overflowed.")
            if k == Op.DETECTOR or k == Op.OBSERVABLE:
                if runtime:
                    raise Error("QM-E586: " + op_name(k) + " in a program with runtime control flow. Hint: detectors index measurements as if each ran once, and no decoder takes a syndrome of varying length; move the QEC circuit out of the loop.")
            elif not is_block_op(k):
                raise Error("QM-E484: opcode " + String(k) + " ('" + op_name(k) + "') at op #" + String(i) + " cannot run in a program block. Hint: blocks take unitary gates, measure, reset, noise, barrier and tick.")
    var cfg = cfg_of(p)
    _check_assigned(p, cfg)
    return cfg^


def assigned_value(e: CExpr, ty: CType) raises -> CExpr:
    """`e` converted for assignment to a variable of type `ty` (the cast table)."""
    return ce_cast(e, ty)


# ---------------------------------------------------------------------------
# Building a Program by API
# ---------------------------------------------------------------------------
struct ProgramBuilder(Movable):
    """Program construction from Mojo; add gates to `body`.

        var b = ProgramBuilder(1, 1)
        b.begin_while(ce_lnot(b.bit(0)))
        b.body.reset(0)
        b.body.h(0)
        b.body.measure(0, 0)
        b.end()
        var p = b.finish()
    """

    var body: Circuit
    var p: Program
    var tb: TapeBuilder

    def __init__(out self, num_qubits: Int, num_clbits: Int = 0, name: String = "program"):
        self.body = Circuit(num_qubits, num_clbits, name)
        self.p = Program(Circuit(0, 0, name))
        self.tb = TapeBuilder()

    def _end(self) -> Int:
        return len(self.body.ops)

    def declare(mut self, name: String, ty: CType, is_output: Bool = False) -> Int:
        """A scalar variable; returns its id."""
        return self.p.add_var(name, ty, is_output)

    def read(self, var_id: Int) -> CExpr:
        ref v = self.p.vars[var_id]
        if v.ty.tag == CT.BIT:
            return CExpr.bits(v.store, v.ty.width)
        return CExpr.variable(v.store, v.ty)

    def bit(self, clbit: Int) -> CExpr:
        return CExpr.bits(clbit, 1)

    def assign(mut self, var_id: Int, e: CExpr) raises:
        """var = e; `e` is converted to the variable's type."""
        var ty = self.p.vars[var_id].ty
        self.tb.assign(var_id, self.p.add_expr(assigned_value(e, ty)), self._end(), 0)

    def assign_bits(mut self, first: Int, width: Int, e: CExpr) raises:
        """c[first .. first+width-1] = e."""
        var v = self.p.add_bit_var("c[" + String(first) + "]", first, width)
        self.assign(v, e)

    def angle(mut self, var_id: Int) -> Param:
        """A gate angle that reads the variable when the gate runs."""
        var name = _slot_name(var_id, self.p.vars[var_id].name)
        self.p.vars[var_id].slot = self.body.parameter(name).index
        return Param.symbol(self.p.vars[var_id].slot)

    def input(mut self, name: String) -> Param:
        """A parameter bound when the program runs."""
        var q = self.body.parameter(name)
        self.p.user_params.append(q.index)
        return q

    def begin_if(mut self, cond: CExpr) raises:
        self.tb.begin(SK.IF, self.p.add_expr(ce_bool(cond)), -1, self._end(), 0)

    def else_(mut self) raises:
        self.tb.else_(self._end(), 0)

    def begin_while(mut self, cond: CExpr) raises:
        self.tb.begin(SK.WHILE, self.p.add_expr(ce_bool(cond)), -1, self._end(), 0)

    def begin_for(mut self, var_id: Int, start: CExpr, step: CExpr, stop: CExpr) raises:
        """for var in [start:step:stop], inclusive as in OpenQASM."""
        var tag = self.p.vars[var_id].ty.tag
        if tag != CT.INT and tag != CT.UINT:
            raise Error("QM-E580: a for loop counts in int or uint. Hint: declare the loop variable with one of them.")
        var off = len(self.p.for_ranges)
        self.p.for_ranges.append(self.p.add_expr(ce_cast(start, int_type(64))))
        self.p.for_ranges.append(self.p.add_expr(ce_cast(step, int_type(64))))
        self.p.for_ranges.append(self.p.add_expr(ce_cast(stop, int_type(64))))
        self.p.for_ranges.append(self.p.num_store)
        self.p.num_store += 3
        self.p.for_ranges.append(-1)
        self.p.for_ranges.append(1 if never_negative(start) and never_negative(stop) else 0)
        self.tb.begin(SK.FOR, var_id, off, self._end(), 0)

    def begin_switch(mut self, e: CExpr) raises:
        self.tb.begin(SK.SWITCH, self.p.add_expr(ce_cast(e, int_type(64))), -1, self._end(), 0)

    def case_(mut self, values: List[Int]) raises:
        var lo = len(self.p.case_vals)
        for v in values:
            self.p.case_vals.append(v)
        self.tb.case_(lo, len(self.p.case_vals), self._end(), 0)

    def default_(mut self) raises:
        self.tb.case_(-1, -1, self._end(), 0)

    def break_(mut self) raises:
        self.tb.jump(SK.BREAK, self._end(), 0)

    def continue_(mut self) raises:
        self.tb.jump(SK.CONTINUE, self._end(), 0)

    def end(mut self) raises:
        self.tb.end(self._end(), 0)

    def finish(mut self) raises -> Program:
        self.tb.flush(self._end(), 0)
        if len(self.tb.open) > 0:
            raise Error("QM-E583: " + String(len(self.tb.open)) + " region(s) left open. Hint: close each with end().")
        if self.body.num_clbits > 0:
            self.p.cregs.append(CReg("c", 0, self.body.num_clbits))
        var out = self.p.copy()
        out.body = self.body.copy()
        out.tape = self.tb.tape.copy()
        for i in range(out.body.num_params()):
            if not out.body.param_names[i].startswith(slot_prefix()):
                var known = False
                for u in out.user_params:
                    known = known or u == i
                if not known:
                    out.user_params.append(i)
        _ = check_program(out)
        return out^
