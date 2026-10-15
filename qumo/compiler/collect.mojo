"""Block collectors for collect-and-resynthesize.

Native, hardware-neutral version of qiskit-ibm-transpiler 0.18.0's
`CollectCliffords`, `CollectLinearFunctions`, `CollectPermutations` and
`CollectPauliNetworks` (ai/collection.py, Apache-2.0; re-implemented from the
algorithm description, no code copied).

Algorithm (Qiskit `BlockCollector.collect_all_matching_blocks` with IBM's
greedy LIFO `collect_matching_block`):
  1. Dependency DAG over the flat op list. `commute=False`: plain per-wire
     order. `commute=True`: commutation-aware (Qiskit `dag_to_dagdependency`):
     on every wire a gate acts with a Pauli axis (`common.wire_axis`); gates
     acting with the same axis on a wire commute there, so consecutive
     same-axis gates on a wire form a *run* and a gate only depends on the
     previous run of its wire. Two gates get no path between them exactly when
     every shared wire agrees on the axis, i.e. when `common.commutes` (without
     its identical-gate special case) says they commute, so every linear
     extension of the DAG has the same unitary. Runs are capped at
     `_RUN_CAP` gates (a cap only adds edges, which is conservative).
     Measurements, resets, conditioned gates, noise channels, barriers and
     annotations act with axis "other" on their wires and are chained on one
     extra classical wire, so their relative order never changes.
  2. Alternating sweeps over the in-degree-0 frontier: first every
     non-matching node that becomes ready is taken (it closes blocks), then
     one matching block is grown greedily: pending nodes are popped LIFO; a
     node is accepted when `block_gate_ok` holds and the union of block qubits
     stays <= `max_width`; its successors whose in-degree drops to 0 are
     pushed; rejected nodes go back to the frontier. `from_back=True` runs the
     same sweep on the reversed DAG.
  3. Blocks are split into the connected components of their qubit-interaction
     graph, then blocks with fewer than `min_gates` gates are dropped.

Convexity: the concatenation of the sweeps is a topological order of the DAG
in which every block is a contiguous segment (split components interleave
only with each other, on disjoint qubits). So every block is convex and can be
replaced in place by emitting the ops in that order with each block's
replacement at the position of its first op (`collect_blocks_ordered`
returns that order; `resynth.mojo` splices with it).
"""

from qumo.ir.circuit import Circuit, Instruction
from qumo.ir.gates import Op, op_arity, op_is_unitary
from qumo.compiler.common import (
    is_plain_gate, is_bound, is_all_wire_op, wire_axis, AX_OTHER, AX_ANY,
)

# Block kinds (mirror of blockalg.BlockKind; kept local so this module does
# not depend on the synthesizers).
comptime KIND_CLIFFORD = 0
comptime KIND_LINEAR = 1
comptime KIND_PERM = 2
comptime KIND_PAULI = 3

comptime _RUN_CAP = 32  # max gates of one same-axis run per wire (edge bound)


struct Block(Copyable):
    """A collected block.

    kind:   KIND_CLIFFORD / KIND_LINEAR / KIND_PERM / KIND_PAULI.
    qubits: sorted wire indices of `c.ops` touched by the block.
    ops:    indices into `c.ops`, in a valid execution order.
    first / last: `ops[0]` / `ops[len(ops) - 1]` (index into `c.ops`).
    """

    var kind: Int
    var qubits: List[Int]
    var ops: List[Int]
    var first: Int
    var last: Int

    def __init__(out self, kind: Int):
        self.kind = kind
        self.qubits = List[Int]()
        self.ops = List[Int]()
        self.first = -1
        self.last = -1

    def num_gates(self) -> Int:
        return len(self.ops)

    def width(self) -> Int:
        return len(self.qubits)


def block_gate_ok(g: Instruction, kind: Int) -> Bool:
    """True when `g` may belong to a block of `kind`.

    Only unconditioned, bound unitary gates on 1 or 2 qubits qualify.
      CLIFFORD: x y z h s sdg sx sxdg cx cy cz swap, and rx/ry/rz by multiples of pi/2.
      LINEAR:   cx swap.
      PERM:     swap.
      PAULI:    cx swap cz h s sdg sx sxdg x y z rz rx ry rzz rxx ryy.
    """
    if not is_plain_gate(g) or not is_bound(g):
        return False
    var k = g.kind
    if kind == KIND_PERM:
        return k == Op.SWAP
    if kind == KIND_LINEAR:
        return k == Op.CX or k == Op.SWAP
    if kind == KIND_CLIFFORD:
        if (k == Op.X or k == Op.Y or k == Op.Z or k == Op.H or k == Op.S or k == Op.SDG
                or k == Op.SX or k == Op.SXDG or k == Op.CX or k == Op.CY or k == Op.CZ or k == Op.SWAP):
            return True
        if k == Op.RX or k == Op.RY or k == Op.RZ:
            return g.is_clifford()
        return False
    if kind == KIND_PAULI:
        return (k == Op.CX or k == Op.SWAP or k == Op.CZ or k == Op.H or k == Op.S or k == Op.SDG
                or k == Op.SX or k == Op.SXDG or k == Op.X or k == Op.Y or k == Op.Z
                or k == Op.RZ or k == Op.RX or k == Op.RY
                or k == Op.RZZ or k == Op.RXX or k == Op.RYY)
    return False


struct _Dag(Movable):
    """Dependency DAG in CSR form (edge lists may hold duplicates; in-degrees
    count them consistently)."""

    var n: Int
    var pstart: List[Int]
    var preds: List[Int]
    var sstart: List[Int]
    var succs: List[Int]

    def __init__(out self, c: Circuit, commute: Bool):
        var n = len(c.ops)
        var nq = c.num_qubits
        var nw = nq + 1  # + classical chain wire
        var cw = nq
        self.n = n
        self.pstart = List[Int](length=n + 1, fill=0)
        self.preds = List[Int]()
        var run_axis = List[Int](length=nw, fill=-1)
        # per wire: current and previous run, flat buffers of _RUN_CAP slots
        var cur = List[Int](length=nw * _RUN_CAP, fill=0)
        var prv = List[Int](length=nw * _RUN_CAP, fill=0)
        var ncur = List[Int](length=nw, fill=0)
        var nprv = List[Int](length=nw, fill=0)
        var wires = List[Int]()
        var axes = List[Int]()
        for i in range(n):
            var g = c.ops[i]
            wires.clear()
            axes.clear()
            var a = op_arity(g.kind)
            if is_all_wire_op(g):
                for w in range(nw):
                    wires.append(w)
                    axes.append(AX_OTHER)
            elif a == 0:
                wires.append(cw)
                axes.append(AX_OTHER)
            elif is_plain_gate(g):
                for s in range(a):
                    var w = g.qubit(s)
                    var ax = AX_OTHER
                    if commute:
                        ax = wire_axis(g.kind, s)
                        if ax == AX_ANY:
                            ax = AX_OTHER
                    wires.append(w)
                    axes.append(ax)
            else:
                for s in range(a):
                    wires.append(g.qubit(s))
                    axes.append(AX_OTHER)
                wires.append(cw)
                axes.append(AX_OTHER)
            for k in range(len(wires)):
                var w = wires[k]
                if w < 0 or w >= nw:
                    continue
                var ax = axes[k]
                var base = w * _RUN_CAP
                if not (ax != AX_OTHER and ax == run_axis[w] and ncur[w] < _RUN_CAP):
                    # start a new run; the current one becomes the previous one
                    for e in range(ncur[w]):
                        prv[base + e] = cur[base + e]
                    nprv[w] = ncur[w]
                    ncur[w] = 0
                    run_axis[w] = ax
                # depends on the previous run (commutes with the current one)
                for e in range(nprv[w]):
                    self.preds.append(prv[base + e])
                cur[base + ncur[w]] = i
                ncur[w] += 1
            self.pstart[i + 1] = len(self.preds)
        # successors (CSR, increasing predecessor order)
        self.sstart = List[Int](length=n + 1, fill=0)
        for e in range(len(self.preds)):
            self.sstart[self.preds[e] + 1] += 1
        for i in range(n):
            self.sstart[i + 1] += self.sstart[i]
        self.succs = List[Int](length=len(self.preds), fill=0)
        var fill = self.sstart.copy()
        for i in range(n):
            for e in range(self.pstart[i], self.pstart[i + 1]):
                var p = self.preds[e]
                self.succs[fill[p]] = i
                fill[p] += 1


def collect_blocks_ordered(c: Circuit, kind: Int, min_gates: Int, max_width: Int,
                           from_back: Bool, commute: Bool, mut order: List[Int]) -> List[Block]:
    """`collect_blocks`, also returning `order`: a topological order of all ops
    of `c` in which every returned block is contiguous (up to interleaving with
    the other components split from the same sweep, on disjoint qubits)."""
    var n = len(c.ops)
    var nq = c.num_qubits
    order = List[Int](capacity=n)
    var out = List[Block]()
    var ok = List[Bool](length=n, fill=False)
    var nok = 0
    if max_width >= 1:
        for i in range(n):
            var g = c.ops[i]
            if block_gate_ok(g, kind) and op_arity(g.kind) <= max_width:
                ok[i] = True
                nok += 1
    if nok < max(min_gates, 1):
        # nothing to collect: no DAG needed
        for i in range(n):
            order.append(i)
        return out^
    var dag = _Dag(c, commute)
    # forward: edges pred -> succ; from_back: the reversed DAG
    var indeg = List[Int](length=n, fill=0)
    for i in range(n):
        if from_back:
            indeg[i] = dag.sstart[i + 1] - dag.sstart[i]
        else:
            indeg[i] = dag.pstart[i + 1] - dag.pstart[i]
    # frontier, ordered so that LIFO pops the earliest op first (forward)
    var pending = List[Int]()
    if from_back:
        for i in range(n):
            if indeg[i] == 0:
                pending.append(i)
    else:
        for i in range(n - 1, -1, -1):
            if indeg[i] == 0:
                pending.append(i)
    var done = 0
    # raw blocks, flat: block k = raw_flat[raw_start[k]:raw_start[k + 1]]
    var raw_flat = List[Int](capacity=nok)
    var raw_start = List[Int]()
    raw_start.append(0)
    var inq = List[Bool](length=nq, fill=False)
    var touched = List[Int]()
    var stack = _Stack(n)
    var nxt = List[Int]()
    while done < n:
        # (a) non-matching sweep (buffers are reused: no per-phase allocation)
        stack.load(pending)
        while stack.size > 0:
            var i = stack.pop()
            if ok[i]:
                pending.append(i)
                continue
            order.append(i)
            done += 1
            nxt.clear()
            _release(dag, i, from_back, indeg, nxt)
            _push_ready(stack, nxt, from_back)
        pending.reverse()  # rejected nodes: earliest popped first next time
        if done >= n:
            break
        # (b) one matching block, greedy LIFO with the width cap
        stack.load(pending)
        var blk_lo = len(raw_flat)
        var width = 0
        while stack.size > 0:
            var i = stack.pop()
            if not ok[i]:
                pending.append(i)
                continue
            var g = c.ops[i]
            var a = op_arity(g.kind)
            var add = 0
            for s in range(a):
                var q = g.qubit(s)
                if not inq[q]:
                    var dup = False
                    for s2 in range(s):
                        if g.qubit(s2) == q:
                            dup = True
                    if not dup:
                        add += 1
            if width + add > max_width:
                pending.append(i)
                continue
            for s in range(a):
                var q = g.qubit(s)
                if not inq[q]:
                    inq[q] = True
                    touched.append(q)
            width += add
            raw_flat.append(i)
            order.append(i)
            done += 1
            nxt.clear()
            _release(dag, i, from_back, indeg, nxt)
            _push_ready(stack, nxt, from_back)
        pending.reverse()
        for q in touched:
            inq[q] = False
        touched.clear()
        if len(raw_flat) > blk_lo:
            raw_start.append(len(raw_flat))
    if from_back:
        order.reverse()
    # split into connected components, filter by size
    var parent = List[Int](length=nq, fill=-1)
    var bl = List[Block]()
    var nraw = len(raw_start) - 1
    var rb = List[Int]()
    for bi in range(nraw):
        var k0 = nraw - 1 - bi if from_back else bi
        rb.clear()
        if from_back:
            for e in range(raw_start[k0 + 1] - 1, raw_start[k0] - 1, -1):
                rb.append(raw_flat[e])
        else:
            for e in range(raw_start[k0], raw_start[k0 + 1]):
                rb.append(raw_flat[e])
        var qs = List[Int]()
        for i in rb:
            var g = c.ops[i]
            for s in range(op_arity(g.kind)):
                var q = g.qubit(s)
                if parent[q] == -1:
                    parent[q] = q
                    qs.append(q)
        for i in rb:
            var g = c.ops[i]
            if op_arity(g.kind) == 2:
                var ra = _find(parent, g.q0)
                var rbq = _find(parent, g.q1)
                if ra != rbq:
                    parent[max(ra, rbq)] = min(ra, rbq)
        # components in order of their first op
        var comp_root = List[Int]()
        var comp_idx = List[Block]()
        for i in rb:
            var r = _find(parent, c.ops[i].q0)
            var ci = -1
            for k in range(len(comp_root)):
                if comp_root[k] == r:
                    ci = k
                    break
            if ci < 0:
                ci = len(comp_root)
                comp_root.append(r)
                comp_idx.append(Block(kind))
            comp_idx[ci].ops.append(i)
        for q in qs:
            var r = _find(parent, q)
            for k in range(len(comp_root)):
                if comp_root[k] == r:
                    comp_idx[k].qubits.append(q)
        for q in qs:
            parent[q] = -1
        for k in range(len(comp_idx)):
            if len(comp_idx[k].ops) < min_gates:
                continue
            sort(comp_idx[k].qubits)
            comp_idx[k].first = comp_idx[k].ops[0]
            comp_idx[k].last = comp_idx[k].ops[len(comp_idx[k].ops) - 1]
            bl.append(comp_idx[k].copy())
    return bl^


def _find(mut parent: List[Int], x: Int) -> Int:
    var r = x
    while parent[r] != r:
        r = parent[r]
    var y = x
    while parent[y] != r:
        var nx = parent[y]
        parent[y] = r
        y = nx
    return r


struct _Stack(Movable):
    """LIFO of op indices with a fixed buffer (List.pop may reallocate)."""

    var data: List[Int]
    var size: Int

    def __init__(out self, cap: Int):
        self.data = List[Int](length=max(cap, 1), fill=0)
        self.size = 0

    @always_inline
    def push(mut self, x: Int):
        if self.size == len(self.data):
            self.data.append(x)
        else:
            self.data[self.size] = x
        self.size += 1

    @always_inline
    def pop(mut self) -> Int:
        self.size -= 1
        return self.data[self.size]

    def load(mut self, mut src: List[Int]):
        """self := src (pushed in order, so src's last element pops first); src := []."""
        self.size = 0
        for k in range(len(src)):
            self.push(src[k])
        src.clear()


def _push_ready(mut stack: _Stack, ready: List[Int], from_back: Bool):
    """Push newly ready nodes so that the next pop is the earliest op
    (forward) or the latest op (from_back)."""
    if from_back:
        for k in range(len(ready)):
            stack.push(ready[k])
    else:
        for k in range(len(ready) - 1, -1, -1):
            stack.push(ready[k])


def _release(dag: _Dag, i: Int, from_back: Bool, mut indeg: List[Int], mut ready: List[Int]):
    """Drop node i from the DAG; append the nodes that become ready (in edge order)."""
    if from_back:
        for e in range(dag.pstart[i], dag.pstart[i + 1]):
            var p = dag.preds[e]
            indeg[p] -= 1
            if indeg[p] == 0:
                ready.append(p)
    else:
        for e in range(dag.sstart[i], dag.sstart[i + 1]):
            var s = dag.succs[e]
            indeg[s] -= 1
            if indeg[s] == 0:
                ready.append(s)


def collect_blocks(c: Circuit, kind: Int, min_gates: Int, max_width: Int, from_back: Bool, commute: Bool) -> List[Block]:
    """Collect maximal blocks of `kind` (see the module docstring).

    min_gates: minimum number of gates of a returned block (after splitting).
    max_width: maximum number of qubits of a block.
    from_back: grow blocks from the end of the circuit (reversed DAG).
    commute:   commutation-aware dependency DAG (gates may be pulled through
               commuting gates into a block).
    Blocks are returned in execution order (of their first op); each block's
    `ops` are in a valid execution order.
    """
    var order = List[Int]()
    return collect_blocks_ordered(c, kind, min_gates, max_width, from_back, commute, order)
