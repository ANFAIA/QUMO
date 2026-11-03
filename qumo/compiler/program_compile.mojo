"""Compile a `Program` for a target.

The preset pipeline text is cut at its `route(`. The passes before it run on each
block, on logical qubits, the passes after it on each block, on physical qubits,
and between them the program gets one layout and each block is routed from the
layout it inherits.

  * One layout. `plan_route` sees a summary circuit: the 2q gates of every block,
    repeated for blocks inside loops. If VF2 embeds the summary, no block needs a
    SWAP anywhere.
  * The invariant: a region exits with its entry layout. Blocks route forward and
    move qubits, so every end of a branch, every end of a loop body and every
    BREAK and CONTINUE ends in a SWAP network back to the layout of its region
    (for a jump, of the loop it leaves or repeats). The network goes in with the
    block, before KAK and basis translation, so they can absorb it. Ancillas are
    free to end anywhere: only the qubits of the program must return.
  * `elide_swaps` and `post_layout` relabel qubits from their point onward, and
    star pre-routing changes the wire of a qubit along a block: none of them can
    work on one block at a time, so they do not run. The AI presets, synthesis
    passes and learned router need a whole circuit: QM-E630.
  * A runtime angle cannot be wrapped into a target's angle bounds: QM-E633.

`verify_program` then checks the result on its own (QM-E631): the same statements
in the same order and the same classical tables, a compiled block at the place of
every reachable source block, l_out(u) == l_in(v) on every edge of `cfg_of`,
l_out = perm(l_in) in each block, and each block against its source through the
layouts, with runtime angles set to arbitrary values. That comparison runs both
through the state-vector kernels on two seeded random states: they must come out
equal up to one global phase. A measurement enters it by the principle of deferred
measurement, as a CX onto a fresh ancilla, and the ancillas of the two sides are
paired by (clbit, nth write to that clbit). A compiled block with a reset, noise or
a conditional gate that its source lacks is QM-E631. Blocks whose source has them,
and blocks over the budget in gates x 2^wires, are counted as not checked.

`ProgramMetrics` reports `base + sum per_iteration_i * N_i`, with the costlier branch
of each IF. There is no global depth: a single number would mislead.
"""

from std.math import sqrt

from qumo.ir.circuit import Circuit, Instruction
from qumo.ir.gates import Op, op_arity, op_is_unitary, op_name
from qumo.ir.classical import CN
from qumo.ir.param import Param
from qumo.ir.program import FORR, Program, SK, Stmt, TERM, TapeBuilder, cfg_of, check_program
from qumo.compiler.ai_route import AIRouteOptions
from qumo.compiler.basis import unroll_3q
from qumo.compiler.blockalg import Perm, SubGraph
from qumo.compiler.block_synth import synth_perm
from qumo.compiler.common import reduce_angle, shell_like
from qumo.compiler.pass_manager import (
    PassId, PassManager, is_synth_pass, parse_pipeline, preset_pipeline, route_options, split_top_level, transpile,
)
from qumo.compiler.routing import RouteOptions, plan_route, route_circuit
from qumo.compiler.target import Target
from qumo.sim.rng import Rng
from qumo.sim.runner import apply_instruction
from qumo.sim.statevector import StateVector

comptime _SUMMARY_LOOP_WEIGHT = 4  # a loop body counts this many times per nesting level
comptime _SUMMARY_MAX_WEIGHT = 16
# What verify_program spends on a block, in gates x 2^wires x 2 random states. Over
# either budget a block is counted as unchecked. Wires are what the budget buys, up
# to the memory of one state.
comptime _CHECK_BLOCK_BUDGET = 1 << 27
comptime _CHECK_TOTAL_BUDGET = 1 << 29
comptime _CHECK_MAX_WIRES = 22
comptime _CHECK_TOLERANCE = 1e-7


@fieldwise_init
struct LoopCost(Copyable):
    var line: Int
    var depth: Int  # 1 for a loop outside any other
    var gates: Int  # per iteration, costlier branch of each IF
    var two_q: Int
    var iterations: Int  # known from constant bounds, else -1


struct ProgramMetrics(Copyable, Writable):
    """Gates of a run: base + sum over loops of per-iteration cost times N_i."""

    var base_gates: Int
    var base_2q: Int
    var loops: List[LoopCost]

    def __init__(out self):
        self.base_gates = 0
        self.base_2q = 0
        self.loops = List[LoopCost]()

    def gates_for(self, iterations: List[Int]) -> Int:
        """Total gates when loop i runs iterations[i] times, nested runs included."""
        var t = self.base_gates
        for i in range(min(len(self.loops), len(iterations))):
            t += self.loops[i].gates * iterations[i]
        return t

    def write_to(self, mut w: Some[Writer]):
        w.write("gates=", self.base_gates, " 2q=", self.base_2q)
        for l in self.loops:
            w.write(" + N(line ", l.line, ")*[", l.gates, ", 2q ", l.two_q, "]")


@fieldwise_init
struct ProgramBlock(Copyable):
    """What verify_program needs about one compiled OPS block."""

    var src_lo: Int  # its source ops, -1 when it only restores a layout
    var src_hi: Int
    var l_in: List[Int]  # logical -> physical on entry
    var l_out: List[Int]
    var perm: List[Int]  # data on physical x ends on physical perm[x]


struct ProgramCompileResult(Copyable, Writable):
    var program: Program
    var initial_layout: List[Int]
    var final_layout: List[Int]
    var metrics: ProgramMetrics
    var blocks: List[ProgramBlock]  # one per OPS statement of `program`
    var pipeline: String
    var target_name: String
    var restored_swaps: Int  # SWAPs added to return to region layouts
    var blocks_checked: Int  # blocks that equal their source on random states
    var blocks_unchecked: Int  # with a reset, noise or a conditional gate, or over budget

    def __init__(out self, var program: Program):
        self.program = program^
        self.initial_layout = List[Int]()
        self.final_layout = List[Int]()
        self.metrics = ProgramMetrics()
        self.blocks = List[ProgramBlock]()
        self.pipeline = String()
        self.target_name = String()
        self.restored_swaps = 0
        self.blocks_checked = 0
        self.blocks_unchecked = 0

    def write_to(self, mut w: Some[Writer]):
        w.write("compiled for ", self.target_name, ": ", self.metrics, ", restored swaps ", self.restored_swaps,
                ", blocks checked ", self.blocks_checked, ", unchecked ", self.blocks_unchecked)


def _leaf(p: Program, lo: Int, hi: Int, num_qubits: Int) -> Circuit:
    var c = shell_like(p.body, num_qubits)
    for i in range(lo, hi):
        c.ops.append(p.body.ops[i])
    return c^


def _relabels(spec: String) raises -> Bool:
    """Does the pipeline text hold a pass that renames qubits for what follows it?"""
    for q in parse_pipeline(spec):
        if q.id == PassId.ROUTE or q.id == PassId.ELIDE_SWAPS or q.id == PassId.POST_LAYOUT:
            return True
        if q.id == PassId.FIXPOINT:
            var inner = String()
            for a in q.args:
                if not a.startswith("max="):
                    inner += ("," if inner.byte_length() > 0 else "") + a
            if _relabels(inner):
                return True
    return False


def _split_pipeline(spec: String, mut pre: String, mut route: String, mut post: String) raises:
    """Cut the text at its route; a top-level elide_swaps or post_layout is dropped.
    A second route, or a relabeling pass inside fixpoint, would rename qubits behind
    the layouts this compiler tracks, so they are refused."""
    var parsed = parse_pipeline(spec)
    var items = split_top_level(spec)
    var seen = False
    for i in range(len(items)):
        ref it = items[i]
        var id = parsed[i].id
        if id == PassId.ROUTE:
            if seen:
                raise Error("QM-E630: a pipeline for a Program takes one route. Hint: drop the second route.")
            route = it
            seen = True
            continue
        if id == PassId.ELIDE_SWAPS or id == PassId.POST_LAYOUT:
            continue
        if _relabels(it):
            raise Error("QM-E630: '" + it + "' renames qubits inside a repeated pass, which a Program cannot follow. Hint: take route, elide_swaps and post_layout out of fixpoint.")
        if seen:
            post += ("," if post.byte_length() > 0 else "") + it
        else:
            pre += ("," if pre.byte_length() > 0 else "") + it


def _check_no_ai(spec: String) raises:
    for s in parse_pipeline(spec):
        if is_synth_pass(s.id):
            raise Error("QM-E630: pass '" + String(s) + "' is not available for a Program. Hint: the AI passes need a whole circuit; use a preset level or a pipeline without them.")
        for a in s.args:
            if a.startswith("policy=") and a != "policy=off":
                raise Error("QM-E630: the learned router is not available for a Program. Hint: use route without a policy.")


def _restore_perm(cur: List[Int], want: List[Int], n: Int, N: Int, label: List[Int]) -> List[Int]:
    """Bijection of physical qubits taking the n program qubits from `cur` to
    `want`. The others hold |0> and go wherever is left, staying put if they can,
    and always inside their component of the coupling graph."""
    var p = List[Int](length=N, fill=-1)
    var taken = List[Bool](length=N, fill=False)
    for l in range(n):
        p[cur[l]] = want[l]
        taken[want[l]] = True
    for x in range(N):
        if p[x] < 0 and not taken[x]:
            p[x] = x
            taken[x] = True
    for x in range(N):
        if p[x] >= 0:
            continue
        for y in range(N):
            if not taken[y] and label[y] == label[x]:
                p[x] = y
                taken[y] = True
                break
        if p[x] < 0:  # a program qubit left its component: _swap_network reports it
            for y in range(N):
                if not taken[y]:
                    p[x] = y
                    taken[y] = True
                    break
    return p^


def _components(t: Target) -> List[Int]:
    var label = List[Int](length=t.num_qubits, fill=-1)
    var count = 0
    for s in range(t.num_qubits):
        if label[s] >= 0:
            continue
        var stack: List[Int] = [s]
        label[s] = count
        while len(stack) > 0:
            var u = stack.pop()
            for j in range(0, len(t.edges), 2):
                var a = t.edges[j]
                var b = t.edges[j + 1]
                var v = b if a == u else (a if b == u else -1)
                if v >= 0 and label[v] < 0:
                    label[v] = count
                    stack.append(v)
        count += 1
    return label^


def _swap_network(t: Target, p: List[Int], seed: Int) raises -> List[Instruction]:
    """SWAPs on the coupling edges realizing `p`, one component at a time, since
    token swapping needs a connected graph. Routing only moves qubits along edges,
    so a permutation that leaves its component means a bug: QM-E632."""
    var out = List[Instruction]()
    var label = _components(t)
    var ncomp = 0
    for v in label:
        ncomp = max(ncomp, v + 1)
    for c in range(ncomp):
        var ids = List[Int]()
        var local = List[Int](length=t.num_qubits, fill=-1)
        for x in range(t.num_qubits):
            if label[x] == c:
                local[x] = len(ids)
                ids.append(x)
        var m = len(ids)
        var q = List[Int](length=m, fill=0)
        var moved = False
        for i in range(m):
            var d = p[ids[i]]
            if label[d] != c:
                raise Error("QM-E632: a layout restore would move physical qubit " + String(ids[i]) + " to another component of the coupling graph. Hint: report this; routing never does that.")
            q[i] = local[d]
            moved = moved or d != ids[i]
        if not moved:
            continue
        var edges = List[Int]()
        for j in range(0, len(t.edges), 2):
            if label[t.edges[j]] == c:
                edges.append(local[t.edges[j]])
                edges.append(local[t.edges[j + 1]])
        var perm = Perm(q)
        var ops = synth_perm(perm, SubGraph(m, edges), "greedy", 0, seed)
        var got = Perm.from_ops(ops, m)
        for i in range(m):
            if got.p[i] != q[i]:
                raise Error("QM-E632: no SWAP network found to restore a layout on " + t.name + ". Hint: report this circuit.")
        for g in ops:
            out.append(Instruction.make(Op.SWAP, ids[g.q0], ids[g.q1]))
    return out^


@fieldwise_init
struct _Region(Copyable):
    var opener: Int
    var entry: List[Int]  # full layout at the region's entry
    var dead: Bool  # unreachable on entry
    var first_case: Bool


struct _Compiler(Movable):
    """Walks the source tape; `pending` holds the block being assembled."""

    var src: Program
    var target: Target
    var opts: RouteOptions
    var routes: Bool
    var n: Int
    var N: Int
    var seed: Int
    var post: Optional[PassManager]
    var arena: Circuit
    var tb: TapeBuilder
    var cur: List[Int]
    var pending: Circuit
    var has_pending: Bool
    var dead: Bool
    var regions: List[_Region]
    var blocks: List[ProgramBlock]
    var block: ProgramBlock
    var restored: Int

    def __init__(out self, src: Program, target: Target, opts: RouteOptions, routes: Bool, N: Int, seed: Int,
                 var post: Optional[PassManager], var initial: List[Int]):
        self.src = src.copy()
        self.target = target.copy()
        self.opts = opts.copy()
        self.routes = routes
        self.n = src.num_qubits()
        self.N = N
        self.seed = seed
        self.post = post^
        self.arena = shell_like(src.body, N)
        self.arena.num_measurements = 0
        self.arena.records = List[Int]()
        self.arena.kraus.clear()
        self.arena.noise_table.clear()
        self.arena.pauli_table.clear()
        self.arena.cond_table.clear()
        self.arena.global_phase = 0.0
        self.tb = TapeBuilder()
        self.cur = initial^
        self.pending = shell_like(src.body, N)
        self.has_pending = False
        self.dead = False
        self.regions = List[_Region]()
        self.blocks = List[ProgramBlock]()
        var ident = List[Int](capacity=N)
        for i in range(N):
            ident.append(i)
        self.block = ProgramBlock(-1, -1, self.cur.copy(), self.cur.copy(), ident^)
        self.restored = 0

    def _fresh_block(mut self):
        var ident = List[Int](capacity=self.N)
        for i in range(self.N):
            ident.append(i)
        self.block = ProgramBlock(-1, -1, self.cur.copy(), self.cur.copy(), ident^)

    def flush(mut self, line: Int) raises:
        """Finish the pending block: post-routing passes, then into the arena."""
        if self.has_pending:
            var out = self.pending.copy()
            if self.post and len(self.pending.ops) > 0:
                out = self.post.value().run(self.pending, self.target).circuit.copy()
            var qmap = List[Int](capacity=self.N)
            for i in range(self.N):
                qmap.append(i)
            var phase = self.arena.global_phase
            var before = len(self.arena.ops)
            self.arena.compose_mapped(out, qmap)
            self.arena.global_phase = phase
            var moved = False
            for x in range(self.N):
                moved = moved or self.block.perm[x] != x
            if len(self.arena.ops) > before or moved or self.block.src_lo >= 0:
                self.block.l_out = self.cur.copy()
                self.blocks.append(self.block.copy())
                if len(self.arena.ops) == before:
                    # gates can cancel, and a source SWAP with a restore SWAP: the block
                    # stays, for what it does to the layout and to tie it to its source
                    self.tb.tape.append(Stmt(SK.OPS, before, before, -1, -1, line))
        self.tb.flush(len(self.arena.ops), line)
        self.pending = shell_like(self.src.body, self.N)
        self.has_pending = False
        self._fresh_block()

    def add_leaf(mut self, var leaf: Circuit, lo: Int, hi: Int) raises:
        if self.dead:
            return
        self.block.src_lo = lo
        self.block.src_hi = hi
        self.block.l_in = self.cur.copy()
        if self.routes:
            var o = self.opts.copy()
            o.initial_layout = self.cur.copy()
            var il = List[Int]()
            var fl = List[Int]()
            var note = String()
            _ = route_circuit(leaf, self.target, o, il, fl, note)
            self.block.l_in = il.copy()
            for l in range(self.N):
                self.block.perm[il[l]] = fl[l]
            self.cur = fl^
        self.pending = leaf^
        self.has_pending = True

    def restore(mut self, want: List[Int]) raises:
        """SWAPs at the end of the pending block, back to the layout `want`."""
        if self.dead or not self.routes:
            return
        var same = True
        for l in range(self.n):
            same = same and self.cur[l] == want[l]
        if same:
            return
        var p = _restore_perm(self.cur, want, self.n, self.N, _components(self.target))
        var net = _swap_network(self.target, p, self.seed)
        self.restored += len(net)
        for g in net:
            self.pending.ops.append(g)
        self.has_pending = True
        for x in range(self.N):
            self.block.perm[x] = p[self.block.perm[x]]
        for l in range(self.N):
            self.cur[l] = p[self.cur[l]]

    def enter(mut self, opener: Int):
        self.regions.append(_Region(opener, self.cur.copy(), self.dead, True))

    def rejoin(mut self, r: _Region):
        """A branch ended: what follows starts from the region's entry layout."""
        self.cur = r.entry.copy()
        self.dead = r.dead
        self._fresh_block()

    def loop_entry(self, loop: Int) raises -> List[Int]:
        var i = len(self.regions) - 1
        while i >= 0:
            if self.regions[i].opener == loop:
                return self.regions[i].entry.copy()
            i -= 1
        raise Error("QM-E632: break or continue outside its loop. Hint: report this program.")

    def run(mut self, pre: List[Circuit]) raises:
        var k = 0
        for i in range(len(self.src.tape)):
            var s = self.src.tape[i]
            var kind = s.kind
            if kind == SK.OPS:
                self.add_leaf(pre[k].copy(), s.a, s.b)
                k += 1
            elif kind == SK.ASSIGN:
                self.flush(s.line)
                self.tb.assign(s.a, s.b, len(self.arena.ops), s.line)
            elif kind == SK.IF or kind == SK.WHILE or kind == SK.SWITCH:
                self.flush(s.line)
                self.tb.begin(kind, s.a, -1, len(self.arena.ops), s.line)
                self.enter(i)
            elif kind == SK.FOR:
                self.flush(s.line)
                self.tb.begin(kind, s.a, s.b, len(self.arena.ops), s.line)
                self.enter(i)
            elif kind == SK.ELSE:
                var r = self.regions[len(self.regions) - 1].copy()
                self.restore(r.entry)
                self.flush(s.line)
                self.tb.else_(len(self.arena.ops), s.line)
                self.rejoin(r)
            elif kind == SK.CASE:
                ref top = self.regions[len(self.regions) - 1]
                var first = top.first_case
                top.first_case = False
                var r = top.copy()
                if not first:
                    self.restore(r.entry)
                self.flush(s.line)
                self.tb.case_(s.a, s.b, len(self.arena.ops), s.line)
                self.rejoin(r)
            elif kind == SK.BREAK or kind == SK.CONTINUE:
                self.restore(self.loop_entry(s.end))
                self.flush(s.line)
                self.tb.jump(kind, len(self.arena.ops), s.line)
                self.dead = True
            else:  # END
                var r = self.regions.pop()
                self.restore(r.entry)
                self.flush(s.line)
                self.tb.end(len(self.arena.ops), s.line)
                self.rejoin(r)
        self.flush(0)


def _summary(p: Program, pre: List[Circuit], n: Int) -> Circuit:
    """Every block's 2q gates, repeated by loop depth, for one global layout."""
    var c = Circuit(n, 0, "summary")
    var depth = 0
    var k = 0
    for s in p.tape:
        if s.kind == SK.WHILE or s.kind == SK.FOR:
            depth += 1
        elif s.kind == SK.END and (p.tape[s.end].kind == SK.WHILE or p.tape[s.end].kind == SK.FOR):
            depth -= 1
        elif s.kind == SK.OPS:
            var w = 1
            for _ in range(depth):
                w = min(w * _SUMMARY_LOOP_WEIGHT, _SUMMARY_MAX_WEIGHT)
            for _ in range(w):
                for g in pre[k].ops:
                    if op_is_unitary(g.kind) and op_arity(g.kind) == 2:
                        c.ops.append(Instruction.make(Op.CX, g.q0, g.q1))
            k += 1
    return c^


def _check_angle_bounds(c: Circuit, t: Target) raises:
    for g in c.ops:
        if t.angle_bound[g.kind] and g.p0.is_symbolic():
            raise Error("QM-E633: runtime angle on '" + op_name(g.kind) + "', whose angle has bounds on " + t.name + ". Hint: a value known only at run time cannot be wrapped into the bounds; use a gate without them.")


def _skeleton(p: Program) -> List[Stmt]:
    """The tape without OPS; `end` links renumbered to match."""
    var at = List[Int](length=len(p.tape), fill=-1)
    var n = 0
    for i in range(len(p.tape)):
        if p.tape[i].kind != SK.OPS:
            at[i] = n
            n += 1
    var out = List[Stmt]()
    for s in p.tape:
        if s.kind != SK.OPS:
            out.append(Stmt(s.kind, s.a, s.b, -1, at[s.end] if s.end >= 0 else -1, 0))
    return out^


def _bound_angle(p: Param, values: List[Float64]) -> Param:
    """A number, reduced as `reduce_large_angles` does: one side of the comparison was
    compiled, and a huge angle the other way round would differ in its rounding."""
    if p.is_symbolic():
        return Param(reduce_angle(p.coeff * values[p.index] + reduce_angle(p.offset)))
    return Param(reduce_angle(p.offset))


def _bound_ops(ops: List[Instruction], values: List[Float64]) -> List[Instruction]:
    var out = List[Instruction](capacity=len(ops))
    for g in ops:
        var h = g
        h.p0 = _bound_angle(g.p0, values)
        h.p1 = _bound_angle(g.p1, values)
        h.p2 = _bound_angle(g.p2, values)
        if g.kind != Op.BARRIER and g.kind != Op.TICK:
            out.append(h)
    return out^


comptime _HAS_RESET = 1
comptime _HAS_NOISE = 2
comptime _HAS_CONDITION = 4


def _block_kinds(c: Circuit, lo: Int, hi: Int) -> Int:
    """What keeps a block from being one unitary plus measurements. Pairing a reset or
    a noise channel across a reordering could raise false alarms, and a conditional
    gate is not one unitary."""
    var kinds = 0
    for i in range(lo, hi):
        ref g = c.ops[i]
        if g.cond >= 0:
            kinds |= _HAS_CONDITION
        if g.kind == Op.RESET:
            kinds |= _HAS_RESET
        elif not op_is_unitary(g.kind) and g.kind != Op.MEASURE and g.kind != Op.BARRIER and g.kind != Op.TICK:
            kinds |= _HAS_NOISE
    return kinds


def _touch(g: Instruction, mut index: List[Int], mut count: Int):
    for j in range(op_arity(g.kind)):
        var q = g.qubit(j)
        if index[q] < 0:
            index[q] = count
            count += 1


def _relabel(g: Instruction, index: List[Int]) -> Instruction:
    var h = g
    var a = op_arity(g.kind)
    if a >= 1: h.q0 = index[g.q0]
    if a >= 2: h.q1 = index[g.q1]
    if a >= 3: h.q2 = index[g.q2]
    return h


def _deferred(g: Instruction, N: Int, mut keys: List[Int], mut writes: List[Int]) -> Instruction:
    """A measurement as a CX onto its ancilla. Ancillas are qubits N, N + 1, ...,
    one per (clbit, nth write to that clbit), the same pair on both sides of the
    comparison, so the two blocks may order their measurements differently."""
    var nth = 0
    for k in range(len(writes)):
        if writes[k] == g.clbit:
            nth += 1
    writes.append(g.clbit)
    var key = (g.clbit + 1) * 1000003 + nth
    for k in range(len(keys)):
        if keys[k] == key:
            return Instruction.make(Op.CX, g.q0, N + k)
    keys.append(key)
    return Instruction.make(Op.CX, g.q0, N + len(keys) - 1)


def _wires(ops: List[Instruction], index: List[Int], values: List[Float64]) raises -> List[Instruction]:
    """Relabelled to wires 0..count-1, and bound."""
    var out = List[Instruction](capacity=len(ops))
    for g in _bound_ops(ops, values):
        out.append(_relabel(g, index))
    return out^


def _random_state(count: Int, mut rng: Rng) -> StateVector:
    var sv = StateVector(count)
    var norm = 0.0
    for i in range(sv.size()):
        sv.re[i] = rng.normal()
        sv.im[i] = rng.normal()
        norm += sv.re[i] * sv.re[i] + sv.im[i] * sv.im[i]
    norm = sqrt(norm)
    for i in range(sv.size()):
        sv.re[i] /= norm
        sv.im[i] /= norm
    return sv^


def _same_on_random_states(want: List[Instruction], got: List[Instruction], count: Int, seed: UInt64) -> Bool:
    """Do two gate lists act alike on two random states, up to one global phase? They
    differ on a random state with probability 1 if they differ at all. Each state
    must come out as z times the other side, with the same unit complex z, which is
    both overlaps equal; distance rather than overlap, so that a small angle shows."""
    var rng = Rng(seed)
    var none = List[Float64]()
    var have = False
    var zr = 0.0
    var zi = 0.0
    for _ in range(2):
        var a = _random_state(count, rng)
        var b = a.copy()
        for g in want:
            apply_instruction(a, g, none)
        for g in got:
            apply_instruction(b, g, none)
        if not have:
            for i in range(a.size()):
                zr += a.re[i] * b.re[i] + a.im[i] * b.im[i]
                zi += a.re[i] * b.im[i] - a.im[i] * b.re[i]
            have = True
        var distance = 0.0
        for i in range(a.size()):
            var dr = zr * a.re[i] - zi * a.im[i] - b.re[i]
            var di = zr * a.im[i] + zi * a.re[i] - b.im[i]
            distance += dr * dr + di * di
        if distance > _CHECK_TOLERANCE * _CHECK_TOLERANCE:
            return False
    return True


def _check_block(src: Program, built: Program, b: ProgramBlock, lo: Int, hi: Int, values: List[Float64], mut spent: Int, seed: UInt64) raises -> Bool:
    """A compiled block against its source plus the layout change, on random states.
    False when the block is outside what that can check, or over budget."""
    var made = _block_kinds(built.body, lo, hi)
    var given = _block_kinds(src.body, b.src_lo, b.src_hi) if b.src_lo >= 0 else 0
    if (made & ~given) != 0:
        raise Error("QM-E631: compiled block at op " + String(lo) + " holds a reset, noise or a conditional gate that its source does not. Hint: report this program; run with a lower optimization level meanwhile.")
    if made != 0 or given != 0:
        return False
    var N = built.num_qubits()
    var keys = List[Int]()
    var expected = List[Instruction]()
    if b.src_lo >= 0:
        var writes = List[Int]()
        for i in range(b.src_lo, b.src_hi):
            ref g = src.body.ops[i]
            if not op_is_unitary(g.kind) and g.kind != Op.MEASURE:
                continue
            var m = g
            var a = op_arity(g.kind)
            if a >= 1: m.q0 = b.l_in[g.q0]
            if a >= 2 and g.kind != Op.MEASURE: m.q1 = b.l_in[g.q1]
            if a >= 3: m.q2 = b.l_in[g.q2]
            expected.append(_deferred(m, N, keys, writes) if g.kind == Op.MEASURE else m)
    var done = List[Bool](length=N, fill=False)
    for x in range(N):
        if done[x] or b.perm[x] == x:
            continue
        var y = b.perm[x]
        done[x] = True
        while y != x:
            expected.append(Instruction.make(Op.SWAP, x, y))
            done[y] = True
            y = b.perm[y]
    var got = List[Instruction]()
    var writes = List[Int]()
    for i in range(lo, hi):
        ref g = built.body.ops[i]
        if op_is_unitary(g.kind):
            got.append(g)
        elif g.kind == Op.MEASURE:
            got.append(_deferred(g, N, keys, writes))
    var index = List[Int](length=N + len(keys), fill=-1)
    var count = 0
    for g in expected:
        _touch(g, index, count)
    for g in got:
        _touch(g, index, count)
    if count == 0:
        return True
    if count > _CHECK_MAX_WIRES:
        return False
    var cost = (len(expected) + len(got)) * (2 << count)
    if cost > _CHECK_BLOCK_BUDGET or spent + cost > _CHECK_TOTAL_BUDGET:
        return False
    spent += cost
    if not _same_on_random_states(_wires(expected, index, values), _wires(got, index, values), count, seed):
        raise Error("QM-E631: compiled block at op " + String(lo) + " does not match its source. Hint: report this program; run with a lower optimization level meanwhile.")
    return True


def _reachable_ops(p: Program) -> List[Bool]:
    """By tape index: can a shot reach this statement?"""
    var cfg = cfg_of(p)
    var seen = List[Bool](length=len(cfg.blocks), fill=False)
    var out = List[Bool](length=len(p.tape), fill=False)
    var work: List[Int] = [0]
    seen[0] = True
    while len(work) > 0:
        var u = work.pop()
        ref bb = cfg.blocks[u]
        for i in range(bb.lo, bb.hi):
            out[i] = True
        var succ = List[Int]()
        if bb.term == TERM.JMP or bb.term == TERM.BR:
            succ.append(bb.t0)
        if bb.term == TERM.BR:
            succ.append(bb.t1)
        if bb.term == TERM.SWITCH:
            for j in range(bb.t0, bb.t1 + 1):
                succ.append(cfg.sw_bb[j])
        for v in succ:
            if not seen[v]:
                seen[v] = True
                work.append(v)
    return out^


def _check_blocks_cover_source(src: Program, built: Program, blocks: List[ProgramBlock]) raises:
    """Walk both tapes: the skeletons are equal, so between two statements each holds
    at most one OPS, and every source block a shot can reach has its compiled block at
    that position, with the same source range. Nothing else may claim a source."""
    var reach = _reachable_ops(src)
    var j = 0
    var k = 0
    for i in range(len(src.tape)):
        ref s = src.tape[i]
        var here = j < len(built.tape) and built.tape[j].kind == SK.OPS
        if s.kind == SK.OPS:
            if here:
                if k >= len(blocks) or blocks[k].src_lo != s.a or blocks[k].src_hi != s.b:
                    raise Error("QM-E631: the compiled block at tape " + String(j) + " does not come from the source block at tape " + String(i) + ". Hint: report this program.")
                j += 1
                k += 1
            elif reach[i]:
                raise Error("QM-E631: the source block at tape " + String(i) + " has no compiled block. Hint: report this program.")
            continue
        if here:
            if k >= len(blocks) or blocks[k].src_lo >= 0:
                raise Error("QM-E631: the compiled block at tape " + String(j) + " claims a source it does not sit at. Hint: report this program.")
            j += 1
            k += 1
        j += 1
    if j < len(built.tape):
        if built.tape[j].kind != SK.OPS or k >= len(blocks) or blocks[k].src_lo >= 0 or j + 1 != len(built.tape):
            raise Error("QM-E631: the compiled program ends with statements its source does not have. Hint: report this program.")


def _same_tables(a: Program, other: Program) -> Bool:
    """Classical tables pass through compiling untouched."""
    if len(a.nodes) != len(other.nodes) or a.expr_lo != other.expr_lo or a.expr_root != other.expr_root or a.for_ranges != other.for_ranges or a.case_vals != other.case_vals:
        return False
    for i in range(len(a.nodes)):
        ref u = a.nodes[i]
        ref v = other.nodes[i]
        if u.op != v.op or u.x != v.x or u.y != v.y or u.i != v.i or u.ty != v.ty or String(u.f) != String(v.f):
            return False
    return True


def verify_program(src: Program, res: ProgramCompileResult) raises -> Tuple[Int, Int]:
    """QM-E631 unless the compiled program keeps the source's skeleton and classical
    tables, every reachable source block has its compiled block, and every edge joins
    equal layouts. Returns the blocks checked and not checked on random states."""
    ref out = res.program
    ref blocks = res.blocks
    var a = _skeleton(src)
    var b = _skeleton(out)
    var same = len(a) == len(b)
    if same:
        for i in range(len(a)):
            same = same and a[i].kind == b[i].kind and a[i].a == b[i].a and a[i].b == b[i].b and a[i].end == b[i].end
    if not same or not _same_tables(src, out):
        raise Error("QM-E631: compiled program has a different control structure. Hint: report this program.")
    _check_blocks_cover_source(src, out, blocks)
    var cfg = cfg_of(out)
    var n = src.num_qubits()
    var nb = len(cfg.blocks)
    var seen = List[Bool](length=nb, fill=False)
    var enter = List[List[Int]]()
    for _ in range(nb):
        enter.append(List[Int]())
    var k = 0
    var block_of = List[Int](length=len(out.tape), fill=-1)
    for i in range(len(out.tape)):
        if out.tape[i].kind == SK.OPS:
            block_of[i] = k
            k += 1
    if k != len(blocks):
        raise Error("QM-E631: " + String(k) + " blocks in the compiled tape, " + String(len(blocks)) + " recorded. Hint: report this program.")
    seen[0] = True
    enter[0] = res.initial_layout.copy()
    var work: List[Int] = [0]
    while len(work) > 0:
        var u = work.pop()
        var cur = enter[u].copy()
        ref bb = cfg.blocks[u]
        for i in range(bb.lo, bb.hi):
            if out.tape[i].kind != SK.OPS:
                continue
            ref blk = blocks[block_of[i]]
            for l in range(n):
                if blk.l_in[l] != cur[l]:
                    raise Error("QM-E631: block at tape " + String(i) + " starts from a layout other than the one that reaches it. Hint: report this program.")
            cur = blk.l_out.copy()
        var succ = List[Int]()
        if bb.term == TERM.JMP or bb.term == TERM.BR:
            succ.append(bb.t0)
        if bb.term == TERM.BR:
            succ.append(bb.t1)
        if bb.term == TERM.SWITCH:
            for j in range(bb.t0, bb.t1 + 1):
                succ.append(cfg.sw_bb[j])
        for v in succ:
            if not seen[v]:
                seen[v] = True
                enter[v] = cur.copy()
                work.append(v)
            else:
                for l in range(n):
                    if enter[v][l] != cur[l]:
                        raise Error("QM-E631: control-flow edge from block " + String(u) + " to " + String(v) + " joins two layouts (line " + String(bb.line) + "). Hint: report this program.")
    for blk in blocks:
        for l in range(n):
            if blk.perm[blk.l_in[l]] != blk.l_out[l]:
                raise Error("QM-E631: a block's permutation does not take its entry layout to its exit layout (qubit " + String(l) + "). Hint: report this program.")
    var values = List[Float64](capacity=out.body.num_params())
    for i in range(out.body.num_params()):
        values.append(0.37 + 0.11 * Float64(i))
    var checked = 0
    var unchecked = 0
    var spent = 0
    k = 0
    for s in out.tape:
        if s.kind != SK.OPS:
            continue
        if _check_block(src, out, blocks[k], s.a, s.b, values, spent, UInt64(k + 1)):
            checked += 1
        else:
            unchecked += 1
        k += 1
    return (checked, unchecked)


def program_metrics(p: Program) -> ProgramMetrics:
    """Cost of the costlier path through each branch, per loop."""
    var m = ProgramMetrics()
    var kinds: List[Int] = [SK.OPS]  # frame 0: the program
    var g: List[Int] = [0]
    var t: List[Int] = [0]
    var best_g: List[Int] = [0]
    var best_t: List[Int] = [0]
    var slot_of_loop = List[Int](length=len(p.tape), fill=-1)
    for i in range(len(p.tape)):
        var s = p.tape[i]
        var top = len(g) - 1
        if s.kind == SK.OPS:
            for j in range(s.a, s.b):
                ref op = p.body.ops[j]
                if op_is_unitary(op.kind) and op.kind != Op.I:
                    g[top] += 1
                    if op_arity(op.kind) == 2:
                        t[top] += 1
        elif s.kind == SK.IF or s.kind == SK.WHILE or s.kind == SK.FOR or s.kind == SK.SWITCH:
            kinds.append(s.kind)
            g.append(0)
            t.append(0)
            best_g.append(0)
            best_t.append(0)
            if s.kind == SK.WHILE or s.kind == SK.FOR:
                slot_of_loop[i] = len(m.loops)
                var depth = 0
                for k in kinds:
                    if k == SK.WHILE or k == SK.FOR:
                        depth += 1
                m.loops.append(LoopCost(s.line, depth, 0, 0, _static_iterations(p, s)))
        elif s.kind == SK.ELSE or s.kind == SK.CASE:
            best_g[top] = max(best_g[top], g[top])
            best_t[top] = max(best_t[top], t[top])
            g[top] = 0
            t[top] = 0
        elif s.kind == SK.END:
            var kind = kinds.pop()
            var cg = g.pop()
            var ct = t.pop()
            var bg = best_g.pop()
            var bt = best_t.pop()
            if kind == SK.WHILE or kind == SK.FOR:
                ref l = m.loops[slot_of_loop[s.end]]
                l.gates = cg
                l.two_q = ct
            else:
                g[top - 1] += max(bg, cg)
                t[top - 1] += max(bt, ct)
    m.base_gates = g[0]
    m.base_2q = t[0]
    return m^


def _const_expr(p: Program, e: Int, mut out: Int) -> Bool:
    if p.expr_lo[e] != p.expr_root[e]:
        return False
    ref n = p.nodes[p.expr_root[e]]
    if n.op != CN.CONST:
        return False
    out = n.i
    return True


def _static_iterations(p: Program, s: Stmt) -> Int:
    if s.kind != SK.FOR:
        return -1
    var off = s.b
    var set_off = p.for_ranges[off + FORR.SET]
    if set_off >= 0:
        return p.case_vals[set_off]
    var a = 0
    var st = 0
    var b = 0
    if not (_const_expr(p, p.for_ranges[off + FORR.START], a) and _const_expr(p, p.for_ranges[off + FORR.STEP], st) and _const_expr(p, p.for_ranges[off + FORR.STOP], b)) or st == 0:
        return -1
    if (p.for_ranges[off + FORR.UNSIGNED] & FORR.BOUNDS_UNSIGNED) != 0 and (a < 0 or b < 0):
        return -1  # past 2^63: held by its bits, and past what this count needs
    return max(0, (b - a) // st + 1)


def compile_spec_program(p: Program, target: Target, spec: String, seed: Int = 7) raises -> ProgramCompileResult:
    """Compile with an explicit text pipeline."""
    _ = check_program(p)
    _check_no_ai(spec)
    var n = p.num_qubits()
    if n > target.num_qubits:
        raise Error("QM-E605: program has " + String(n) + " qubits but target '" + target.name + "' has " + String(target.num_qubits) + ". Hint: pick a larger target.")
    var pre_spec = String()
    var route_spec = String()
    var post_spec = String()
    _split_pipeline(spec, pre_spec, route_spec, post_spec)
    var routes = route_spec.byte_length() > 0 and not target.full
    var N = target.num_qubits if routes else n
    var opts = RouteOptions()
    if route_spec.byte_length() > 0:
        var aopts = AIRouteOptions()
        var star = False
        opts = route_options(parse_pipeline(route_spec)[0].args, seed, aopts, star)
    # logical passes, block by block
    var pre = List[Circuit]()
    var pm_pre = PassManager(pre_spec, seed) if pre_spec.byte_length() > 0 else PassManager("remove_identities", seed)
    pm_pre.exact = True
    for s in p.tape:
        if s.kind != SK.OPS:
            continue
        var c = _leaf(p, s.a, s.b, n)
        if pre_spec.byte_length() > 0:
            c = pm_pre.run(c, target).circuit.copy()
        if routes:
            _ = unroll_3q(c)
        pre.append(c^)
    # one layout for the whole program
    var layout = List[Int](capacity=N)
    if routes:
        var sum = _summary(p, pre, n)
        var plan = plan_route(sum, target, opts)
        layout = plan.layout.copy()
    else:
        for i in range(N):
            layout.append(i)
    var post: Optional[PassManager] = None
    if post_spec.byte_length() > 0:
        var pm_post = PassManager(post_spec, seed)
        pm_post.exact = True
        post = pm_post^
    var comp = _Compiler(p, target, opts, routes, N, seed, post^, layout.copy())
    comp.run(pre)
    var out = p.copy()
    out.body = comp.arena.copy()
    out.body.num_clbits = p.num_clbits()
    out.body.num_qubits = N
    out.tape = comp.tb.tape.copy()
    _check_angle_bounds(out.body, target)
    var res = ProgramCompileResult(out^)
    res.initial_layout = layout^
    res.final_layout = comp.cur.copy()
    res.pipeline = spec
    res.target_name = target.name
    res.restored_swaps = comp.restored
    res.blocks = comp.blocks.copy()
    _ = check_program(res.program)
    var counts = verify_program(p, res)
    res.blocks_checked = counts[0]
    res.blocks_unchecked = counts[1]
    res.metrics = program_metrics(res.program)
    return res^


def compile_program(p: Program, target: Target, level: Int = 2, seed: Int = 7, ai: Bool = False) raises -> ProgramCompileResult:
    """Compile for `target` at preset level 0..3. A program without runtime regions
    is a plain circuit and goes through `transpile`."""
    if ai:
        raise Error("QM-E630: the AI presets are not available for a Program. Hint: use compile_program(p, target, level), or flatten the program first.")
    var flat = p.try_flatten()
    if flat:
        var r = transpile(flat.value(), target, level, seed)
        var res = ProgramCompileResult(Program.from_circuit(r.circuit))
        res.initial_layout = r.initial_layout.copy()
        res.final_layout = r.final_layout.copy()
        res.pipeline = r.pipeline
        res.target_name = target.name
        res.metrics = program_metrics(res.program)
        return res^
    return compile_spec_program(p, target, preset_pipeline(level, target), seed)


def compact_qubits(p: Program) -> Program:
    """The same program on just the qubits it touches, renumbered in order. A
    compiled program lives on the whole device; this makes it simulable."""
    var used = List[Bool](length=p.num_qubits(), fill=False)
    for g in p.body.ops:
        for j in range(op_arity(g.kind)):
            if g.qubit(j) >= 0:
                used[g.qubit(j)] = True
    var index = List[Int](length=p.num_qubits(), fill=-1)
    var count = 0
    for q in range(p.num_qubits()):
        if used[q]:
            index[q] = count
            count += 1
    var out = p.copy()
    out.body.num_qubits = max(count, 1)
    for ref g in out.body.ops:
        var a = op_arity(g.kind)
        if a >= 1 and g.q0 >= 0: g.q0 = index[g.q0]
        if a >= 2: g.q1 = index[g.q1]
        if a >= 3: g.q2 = index[g.q2]
    return out^
