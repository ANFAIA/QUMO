"""Layout + routing: VF2 exact layout, SabreLayout trials, LightSABRE routing.

Best-of synthesis:
  * Qiskit `VF2Layout`: embed the interaction graph into the coupling graph
    (subgraph monomorphism; VF2++ BFS order, Warnsdorff candidate order,
    degree pruning and distance pruning d_coupling <= d_interaction). Success
    means zero swaps.
  * Relaxed VF2 (new): when the full graph does not embed (odd rings on the
    bipartite heavy-hex lattice, a few long-range edges), its max-weight
    spanning forest is embedded from many start qubits and the embedding whose
    dropped edges are shortest seeds SABRE.
  * TKET `GraphPlacement`-like greedy weighted embeddings (Prim order, sum of
    w * distance, temporal-partner edges) from farthest-point start qubits.
  * Qiskit `SabreLayout` / LightSABRE (Zou et al. 2024): seeded layout trials
    run in parallel (`max.algorithm.parallelize`), forward/backward refinement,
    then `swap_trials` routing seeds on the chosen layout, cheapest emitted.
    Cost = max(decay) * (front + lookahead) evaluated incrementally per
    candidate swap; release valve after 10 * N fruitless swaps. Two lookahead
    flavours are mixed across trials (`heuristic=auto`): the paper's extended
    set (next 20 2q gates, size-normalised) and Qiskit 2.x's single lookahead
    layer (constant scaling, weight 0.5 / N).
  * Trials run on the 2q skeleton (1q gates, measurements and back-to-back
    repeats on one pair never change a SABRE decision), the final pass on the
    full circuit with the winning seed (the lookahead ignores repeats, so the
    decisions match).
  * Qiskit `StarPreRouting` as a portfolio candidate (`star_preroute`): runs of
    2q blocks sharing a centre become a swap chain (star -> path); the pass
    manager plans both variants and keeps the cheaper.
  * Deterministic for a given seed (`qumo.sim.rng.Rng`). Non-unitary ops keep
    their program order through a classical chain in the DAG; barriers
    synchronise all wires; conditioned 2q gates are routed like any gate.

Layout convention: `l2p[l]` = physical qubit holding circuit wire l. Wires
n..N-1 are ancillas (start in |0>). The routed circuit acts on N physical
qubits; `final_layout[l]` says where logical l ends up after the SWAPs.
"""

from std.collections import Dict
from qumo.ir.gates import Op, op_arity, op_is_unitary, op_is_noise
from qumo.ir.circuit import Circuit, Instruction
from qumo.compiler.target import Target
from qumo.compiler.common import (
    shell_like, is_all_wire_op, is_bound, wire_axis, AX_Z, AX_ANY, AX_OTHER,
)
from qumo.ir.gates import M4
from qumo.compiler.kak import mm4, gate_m4
from qumo.sim.rng import Rng
from max.algorithm import parallelize


struct RouteOptions(Copyable, Writable):
    var layout: String  # auto | sabre | dense | trivial | random | vf2
    var seed: Int
    var trials: Int
    var iterations: Int
    var lookahead: Int
    var weight: Float64
    var decay: Float64
    var vf2_steps: Int
    var valve: Int  # swaps without progress before the release valve (0 = 10 * num_qubits)
    var initial_layout: List[Int]  # user layout (logical -> physical); used when non-empty
    var heuristic: String  # "layer" (Qiskit 2.x: sum_F d + w/N sum_L1 d) | "extended" (LightSABRE paper) | "auto" (trials alternate)
    var swap_trials: Int  # routing seeds tried on the chosen layout (best kept)
    var absorb: Float64  # score factor for swaps KAK can absorb (< 1 favours them)
    var absorb_alt: Float64  # factor used by every other swap trial (portfolio)
    var mirror: Bool  # mirror-gate routing: apply free / cheap SWAPs absorbed by the gate just executed on that pair
    var mirror_auto: Bool  # resolve `mirror` per circuit: free mirrors only when most 2q runs are >= 3-CX blocks (QV-like)
    var mirror_gain: Float64  # heuristic gain (front-distance units) a 1-CX mirror must reach (free mirrors need > 0)
    var commute: Bool  # commutation-aware routing DAG (commuting units may be routed in any order)
    var commute_auto: Bool  # decide per circuit (see `resolve_commute`)

    def __init__(out self):
        self.layout = "auto"
        self.seed = 7
        self.trials = 4
        self.iterations = 2
        self.lookahead = 20
        self.weight = 0.5
        self.decay = 0.001
        self.vf2_steps = 20000
        self.valve = 0
        self.initial_layout = List[Int]()
        self.heuristic = "auto"
        self.swap_trials = 1
        self.absorb = 1.0
        self.absorb_alt = 0.9
        self.mirror = True
        self.mirror_auto = True
        self.mirror_gain = 1.0
        self.commute = True
        self.commute_auto = True

    def write_to(self, mut w: Some[Writer]):
        w.write("layout=", self.layout, ",seed=", self.seed, ",trials=", self.trials,
                ",iterations=", self.iterations, ",lookahead=", self.lookahead)


def kind_cost_2q(k: Int) -> Int:
    """CX-count of a 2q gate kind (CX-like 1, SWAP 3, CP / RZZ / iSWAP / ... 2)."""
    if k == Op.CX or k == Op.CZ or k == Op.CY or k == Op.CH or k == Op.ECR:
        return 1
    if k == Op.SWAP:
        return 100  # sentinel: runs holding a SWAP (e.g. star pre-routing) are not mirrored
    return 2


# ----------------------------------------------------------------------------
# Routing units: same-pair runs as atomic nodes, per-wire commutation axes
# ----------------------------------------------------------------------------
comptime _GROUP_CAP = 16  # max members of one commuting group (bounds DAG edges)


def _combine_axis(cur: Int, ax: Int) -> Int:
    """Fold one more gate's wire axis into a group axis (AX_ANY = no constraint)."""
    if ax == AX_ANY:
        return cur
    if cur == AX_ANY:
        return ax
    if cur != ax:
        return AX_OTHER
    return cur


def _unit_is_diagonal(ops: List[Instruction], idx: List[Int], lo: Int, hi: Int, a: Int, b: Int) -> Bool:
    """True when the ops idx[lo:hi] (all on wires a, b, bound) multiply to a
    diagonal 4x4 unitary (e.g. cx . rz . cx = rzz): such a run commutes with
    every Z-type gate on both wires."""
    var m = M4.identity()
    var qlo = min(a, b)
    var qhi = max(a, b)
    for e in range(lo, hi):
        var g = ops[idx[e]]
        if not is_bound(g):
            return False
        m = mm4(gate_m4(g, qlo, qhi), m)
    var off = 0.0
    comptime for r in range(4):
        comptime for c in range(4):
            comptime if r != c:
                off += abs(m.re[r * 4 + c]) + abs(m.im[r * 4 + c])
    return off < 1e-10


struct RUnits(Movable):
    """Program-ordered routing units over a flat op list.

    A unit is either a maximal same-pair run (2q gates on one pair, the 1q
    gates between them on those wires included; executed atomically by the
    router) or a single other op (1q gate outside runs, measurement, reset,
    barrier, conditioned gate). Each unit carries the Pauli axis it commutes
    with on each wire (`AX_Z` for diagonal runs such as cp / rzz / cx.rz.cx,
    `AX_Z`/`AX_X` for cx control / target, ...; `AX_OTHER` when nothing is
    known), which the commutation-aware DAG uses: consecutive units with the
    same axis on a wire need not keep their program order on that wire
    (Qiskit `CommutationAnalysis` applied to routing; TKET / 2QAN-style
    freedom for QAOA and Trotter circuits whose diagonal terms commute).
    """

    var n: Int
    var start: List[Int]  # CSR into idx (op indices in program order)
    var idx: List[Int]
    var is_run: List[Bool]
    var a: List[Int]  # run: first member's q0 / q1; other: -1
    var b: List[Int]
    var ax_a: List[Int]
    var ax_b: List[Int]
    var cost: List[Int]  # run: summed CX cost (>= 100 when it holds a SWAP)

    def __init__(out self, ops: List[Instruction], nq: Int, commute: Bool):
        var n_ops = len(ops)
        # pass 1: runs (same-pair 2q gates, consecutive on both wires; any
        # non-unitary / conditioned / 3q op or barrier on a wire breaks them)
        var last2 = List[Int](length=nq, fill=-1)
        var run_of = List[Int](length=n_ops, fill=-1)  # run id of a member / an inside 1q op
        var cand = List[Int](length=n_ops, fill=-1)  # 1q op: run open on its wire at that time
        var r_first = List[Int]()
        var r_last = List[Int]()
        var r_a = List[Int]()
        var r_b = List[Int]()
        for i in range(n_ops):
            var g = ops[i]
            var k = g.kind
            if is_all_wire_op(g):
                for w in range(nq):
                    last2[w] = -1
                continue
            var ar = op_arity(k)
            if not op_is_unitary(k) or g.cond >= 0 or ar >= 3:
                for s in range(ar):
                    var w = g.qubit(s)
                    if w >= 0 and w < nq:
                        last2[w] = -1
                continue
            if ar == 1:
                cand[i] = last2[g.q0]
                continue
            if ar != 2:
                continue
            var r = last2[g.q0]
            if r >= 0 and r == last2[g.q1]:
                run_of[i] = r
                r_last[r] = i
                continue
            r = len(r_first)
            r_first.append(i)
            r_last.append(i)
            r_a.append(g.q0)
            r_b.append(g.q1)
            run_of[i] = r
            last2[g.q0] = r
            last2[g.q1] = r
        var nr = len(r_first)
        for i in range(n_ops):
            var r = cand[i]
            if r >= 0 and i < r_last[r]:
                run_of[i] = r  # 1q op inside the run's span: part of the unit
        # pass 2: per-run op lists (CSR, program order)
        var r_start = List[Int](length=nr + 1, fill=0)
        for i in range(n_ops):
            if run_of[i] >= 0:
                r_start[run_of[i] + 1] += 1
        for r in range(nr):
            r_start[r + 1] += r_start[r]
        var r_idx = List[Int](length=r_start[nr], fill=0)
        var fillp = r_start.copy()
        for i in range(n_ops):
            var r = run_of[i]
            if r >= 0:
                r_idx[fillp[r]] = i
                fillp[r] += 1
        # pass 3: units in order of their first op
        self.n = 0
        self.start = List[Int]()
        self.start.append(0)
        self.idx = List[Int](capacity=n_ops)
        self.is_run = List[Bool]()
        self.a = List[Int]()
        self.b = List[Int]()
        self.ax_a = List[Int]()
        self.ax_b = List[Int]()
        self.cost = List[Int]()
        for i in range(n_ops):
            var r = run_of[i]
            if r >= 0:
                if i != r_first[r]:
                    continue
                var ra = r_a[r]
                var rb = r_b[r]
                var xa = AX_ANY
                var xb = AX_ANY
                var cst = 0
                for e in range(r_start[r], r_start[r + 1]):
                    var g = ops[r_idx[e]]
                    self.idx.append(r_idx[e])
                    var ar = op_arity(g.kind)
                    if ar == 2:
                        cst += kind_cost_2q(g.kind)
                        for s in range(2):
                            var ax = wire_axis(g.kind, s)
                            if g.qubit(s) == ra:
                                xa = _combine_axis(xa, ax)
                            else:
                                xb = _combine_axis(xb, ax)
                    else:
                        var ax = wire_axis(g.kind, 0)
                        if g.q0 == ra:
                            xa = _combine_axis(xa, ax)
                        else:
                            xb = _combine_axis(xb, ax)
                if not commute:
                    xa = AX_OTHER
                    xb = AX_OTHER
                elif (xa == AX_OTHER or xb == AX_OTHER) and cst < 100 and r_start[r + 1] - r_start[r] <= 24:
                    if _unit_is_diagonal(ops, r_idx, r_start[r], r_start[r + 1], ra, rb):
                        xa = AX_Z
                        xb = AX_Z
                if xa == AX_ANY:
                    xa = AX_OTHER
                if xb == AX_ANY:
                    xb = AX_OTHER
                self.is_run.append(True)
                self.a.append(ra)
                self.b.append(rb)
                self.ax_a.append(xa)
                self.ax_b.append(xb)
                self.cost.append(cst)
            else:
                var g = ops[i]
                self.idx.append(i)
                var xa = AX_OTHER
                if commute and op_is_unitary(g.kind) and g.cond < 0 and op_arity(g.kind) == 1:
                    xa = wire_axis(g.kind, 0)
                self.is_run.append(False)
                self.a.append(-1)
                self.b.append(-1)
                self.ax_a.append(xa)
                self.ax_b.append(AX_OTHER)
                self.cost.append(0)
            self.start.append(len(self.idx))
            self.n += 1


struct _GroupDag(Movable):
    """Edge builder for the commutation-aware DAG (per-wire axis groups).

    On each wire the node sequence is cut into maximal groups of equal axis
    (AX_OTHER nodes are singleton groups); a node depends on every member of
    the previous group on each of its wires. Groups are contiguous in the
    per-wire sequence, so two boundaries per wire describe them."""

    var seq: List[List[Int]]
    var cur_lo: List[Int]
    var prev_lo: List[Int]
    var cur_ax: List[Int]
    var src: List[Int]
    var dst: List[Int]
    var joins: Int  # nodes that joined an existing group (commutation used)

    def __init__(out self, nq: Int, cap: Int):
        self.seq = List[List[Int]]()
        for _ in range(nq):
            self.seq.append(List[Int]())
        self.cur_lo = List[Int](length=nq, fill=0)
        self.prev_lo = List[Int](length=nq, fill=0)
        self.cur_ax = List[Int](length=nq, fill=AX_OTHER)
        self.src = List[Int](capacity=cap)
        self.dst = List[Int](capacity=cap)
        self.joins = 0

    def add(mut self, v: Int, w: Int, ax: Int):
        if ax == AX_ANY:
            return  # identity: no constraint
        var ln = len(self.seq[w])
        if ax != AX_OTHER and ax == self.cur_ax[w] and ln - self.cur_lo[w] < _GROUP_CAP:
            self.joins += 1
            for e in range(self.prev_lo[w], self.cur_lo[w]):
                self.src.append(self.seq[w][e])
                self.dst.append(v)
        else:
            for e in range(self.cur_lo[w], ln):
                self.src.append(self.seq[w][e])
                self.dst.append(v)
            self.prev_lo[w] = self.cur_lo[w]
            self.cur_lo[w] = ln
            self.cur_ax[w] = ax
        self.seq[w].append(v)


# ----------------------------------------------------------------------------
# DAG over routing units (CSR successors)
# ----------------------------------------------------------------------------
struct RDag(Movable):
    """Routing DAG over `RUnits` (CSR successors).

    Full mode (final, emitting pass): one node per unit. Skeleton mode
    (layout / swap trials): one node per 2q unit plus, on a wire, a marker
    node only where the 1q gates between two 2q units change which units may
    commute (`_marker_needed`), so both modes impose the same order on the 2q
    units and a seed routes identically in both. Edges come from per-wire
    commutation groups (`_GroupDag`); measurements, resets, conditioned gates
    and annotations also keep their program order (classical chain), and
    barriers synchronise every wire (full mode)."""

    var n: Int
    var succ_start: List[Int]
    var succ: List[Int]
    var npred: List[Int]
    var is2q: List[Bool]
    var a: List[Int]
    var b: List[Int]
    var look: List[Bool]  # 2q node that counts for the lookahead
    var mc: List[Int]  # extra 2q cost of mirroring the pair right after this unit (0: >= 3-CX block, 1, 9: never)
    var emit_start: List[Int]  # full mode: CSR of op indices emitted by each node
    var emit_ops: List[Int]
    var n2q: Int
    var commuting: Float64  # (wire, node) group joins per 2q node: how much order freedom commutation adds

    def __init__(out self, ops: List[Instruction], u: RUnits, nq: Int, reverse: Bool, skeleton: Bool):
        # ---- node list in program order -------------------------------
        var nd_unit = List[Int]()  # unit id, -1 for a marker
        var nd_w0 = List[Int]()  # marker wire
        var nd_ax0 = List[Int]()  # marker axis
        var pend = List[Int](length=nq, fill=AX_ANY)  # skeleton: combined axis of 1q ops since the last 2q unit
        var gax = List[Int](length=nq, fill=AX_OTHER)  # skeleton: current group axis per wire
        for t in range(u.n):
            var first = ops[u.idx[u.start[t]]]
            var two = u.is_run[t] or (op_is_unitary(first.kind) and op_arity(first.kind) == 2)
            if not skeleton:
                nd_unit.append(t)
                nd_w0.append(-1)
                nd_ax0.append(AX_OTHER)
                continue
            if not two:
                # fold into the pending segment of its wires
                if is_all_wire_op(first):
                    for w in range(nq):
                        pend[w] = _combine_axis(pend[w], AX_OTHER)
                else:
                    var ar = op_arity(first.kind)
                    for s in range(ar):
                        var w = first.qubit(s)
                        if w >= 0 and w < nq:
                            pend[w] = _combine_axis(pend[w], u.ax_a[t] if ar == 1 else AX_OTHER)
                continue
            var wa = u.a[t] if u.is_run[t] else first.q0
            var wb = u.b[t] if u.is_run[t] else first.q1
            var xa = u.ax_a[t] if u.is_run[t] else AX_OTHER
            var xb = u.ax_b[t] if u.is_run[t] else AX_OTHER
            for side in range(2):
                var w = wa if side == 0 else wb
                var an = xa if side == 0 else xb
                if pend[w] != AX_ANY:
                    var am = pend[w]
                    var ap = gax[w]
                    if am != ap and an == ap and ap != AX_OTHER:
                        nd_unit.append(-1)
                        nd_w0.append(w)
                        nd_ax0.append(am)
                        gax[w] = am
                    pend[w] = AX_ANY
                gax[w] = an
            nd_unit.append(t)
            nd_w0.append(-1)
            nd_ax0.append(AX_OTHER)
        var n = len(nd_unit)
        self.n = n
        self.npred = List[Int](length=n, fill=0)
        self.is2q = List[Bool](length=n, fill=False)
        self.a = List[Int](length=n, fill=-1)
        self.b = List[Int](length=n, fill=-1)
        self.look = List[Bool](length=n, fill=False)
        self.mc = List[Int](length=n, fill=9)
        self.emit_start = List[Int](length=n + 1, fill=0)
        self.emit_ops = List[Int]()
        var gd = _GroupDag(nq, 4 * n)
        var last_cl = -1
        for v in range(n):
            var pn = n - 1 - v if reverse else v
            var t = nd_unit[pn]
            if t < 0:
                gd.add(v, nd_w0[pn], nd_ax0[pn])
                self.emit_start[v + 1] = len(self.emit_ops)
                continue
            if not skeleton:
                if reverse:
                    var e = u.start[t + 1] - 1
                    while e >= u.start[t]:
                        self.emit_ops.append(u.idx[e])
                        e -= 1
                else:
                    for e in range(u.start[t], u.start[t + 1]):
                        self.emit_ops.append(u.idx[e])
            self.emit_start[v + 1] = len(self.emit_ops)
            var first = ops[u.idx[u.start[t]]]
            if u.is_run[t]:
                self.is2q[v] = True
                self.look[v] = True
                self.a[v] = u.a[t]
                self.b[v] = u.b[t]
                var cst = u.cost[t]
                self.mc[v] = 9 if cst >= 100 else (0 if cst >= 3 else 1)
                gd.add(v, u.a[t], u.ax_a[t])
                gd.add(v, u.b[t], u.ax_b[t])
                continue
            var k = first.kind
            var chain = False
            if is_all_wire_op(first):
                for w in range(nq):
                    gd.add(v, w, AX_OTHER)
                chain = True
            else:
                var ar = op_arity(k)
                for s in range(ar):
                    var w = first.qubit(s)
                    if w >= 0 and w < nq:
                        gd.add(v, w, u.ax_a[t] if ar == 1 else AX_OTHER)
                if not op_is_unitary(k) and not op_is_noise(k):
                    chain = True
                if first.cond >= 0:
                    chain = True
                if op_is_unitary(k) and ar == 2:
                    # conditioned 2q gates need adjacency too (routing SWAPs stay unconditioned)
                    self.is2q[v] = True
                    self.look[v] = True
                    self.a[v] = first.q0
                    self.b[v] = first.q1
            if chain and not skeleton:
                if last_cl >= 0:
                    gd.src.append(last_cl)
                    gd.dst.append(v)
                last_cl = v
        self.n2q = 0
        for v in range(n):
            if self.is2q[v]:
                self.n2q += 1
        self.commuting = Float64(gd.joins) / Float64(max(self.n2q, 1))
        self.succ_start = List[Int](length=n + 1, fill=0)
        for e in range(len(gd.src)):
            self.succ_start[gd.src[e] + 1] += 1
            self.npred[gd.dst[e]] += 1
        for v in range(n):
            self.succ_start[v + 1] += self.succ_start[v]
        self.succ = List[Int](length=len(gd.src), fill=0)
        var fillp = self.succ_start.copy()
        for e in range(len(gd.src)):
            self.succ[fillp[gd.src[e]]] = gd.dst[e]
            fillp[gd.src[e]] += 1


# ----------------------------------------------------------------------------
# SABRE core
# ----------------------------------------------------------------------------
def _emit_mapped(g: Instruction, l2p: List[Int], mut out: List[Instruction]):
    var x = g
    var ar = op_arity(g.kind)
    if ar >= 1 and x.q0 >= 0:
        x.q0 = l2p[x.q0]
    if ar >= 2:
        x.q1 = l2p[x.q1]
    if ar >= 3:
        x.q2 = l2p[x.q2]
    out.append(x)


def sabre_pass(ops: List[Instruction], dag: RDag, target: Target, mut l2p: List[Int],
               mut p2l: List[Int], seed: UInt64, emit: Bool, mut out: List[Instruction],
               opts: RouteOptions,
               mut cost: Int, mut mirrors: Int) -> Int:
    """Route one pass; updates l2p/p2l to the final layout; returns #swaps
    emitted, mirrors included (-1 when two interacting qubits sit in
    disconnected components).

    The front layer may hold several 2q units on one qubit (commuting units
    of the commutation-aware DAG), kept in per-qubit linked lists; a
    candidate swap is scored over every front / lookahead unit on its two
    qubits.

    `cost` estimates the routing overhead in CX units after KAK: 3 per plain
    SWAP, and for a SWAP on a pair whose last event was a gate on that same
    pair (KAK merges them into one block) 0 when that block already costs
    >= 3 CX (a generic SU(4): G.SWAP is again one SU(4)) or 1 otherwise
    (CX.SWAP = 2 CX, CP.SWAP = 3 CX).

    Mirror-gate routing (MIRAGE, McKinney et al. 2024): before a swap is
    chosen, every SWAP on the pair of a gate that has just executed is
    scored with the same heuristic; when it lowers the cost (free mirrors)
    or lowers the front distance by >= `mirror_gain` (1-CX mirrors) it is
    applied without counting as a routing step. `mirrors` counts them.

    Determinism across the skeleton and the full DAG: the extended set is
    built level by level and cut in DAG order, and ties are broken over
    candidates sorted by physical qubits, so a seed routes identically in
    the trials and in the final pass."""
    var absorbed = 0
    var acc = 0
    mirrors = 0
    var n = dag.n
    var nphys = target.num_qubits
    var inf = nphys + 1
    var remaining = dag.npred.copy()
    var stack = List[Int](capacity=64)
    var v = n - 1
    while v >= 0:
        if remaining[v] == 0:
            stack.append(v)
        v -= 1
    var front = List[Int]()
    var front_pos = List[Int](length=n, fill=-1)
    # per logical qubit: linked list of its front units (slot e = 2u + side)
    var fhead = List[Int](length=nphys, fill=-1)
    var fnx = List[Int](length=2 * n + 2, fill=-1)
    var fpv = List[Int](length=2 * n + 2, fill=-1)
    var pstamp = List[Int](length=nphys, fill=-1)
    var fcnt = List[Int](length=nphys, fill=0)  # front units per logical qubit
    var nmulti = 0  # logical qubits with >= 2 front units (commuting units)
    var fmin = List[Int](length=nphys, fill=0)
    var fmin2 = List[Int](length=nphys, fill=0)
    var fcmin = List[Int](length=nphys, fill=0)
    var qs = List[Int](length=nphys, fill=-1)
    var cmin = List[Int](length=nphys, fill=0)
    var cat = List[Int](length=nphys, fill=0)
    var qstamp = 0
    var decay = List[Float64](length=nphys, fill=1.0)
    var decayed = List[Int]()
    # last 2q event per physical qubit (id; swaps get negative ids): a swap on a
    # pair whose last event was a gate on that same pair is absorbed by KAK
    var last2q = List[Int](length=nphys, fill=0)
    var last_mc = List[Int](length=nphys, fill=1)  # mirror cost of that last gate
    var ev = 0
    var ext_a = List[Int]()
    var ext_b = List[Int]()
    var ext_v = List[Int]()
    var bfs = List[Int]()
    var now = List[Int]()
    var dec = List[Int]()
    var rel = List[Int]()
    var e_first = List[Int](length=nphys, fill=-1)
    var e_next = List[Int](length=2 * max(opts.lookahead, n) + 2, fill=-1)
    var rng = Rng(seed)
    var swaps = 0
    var since_progress = 0
    var swaps_since_decay = 0
    var step = 0
    var valve = opts.valve if opts.valve > 0 else 10 * nphys
    var layered = opts.heuristic == "layer"
    var look_w = opts.weight / Float64(nphys)
    var valves = 0
    var cand_p = List[Int]()
    var cand_x = List[Int]()
    var ties = List[Int]()
    var mirror_on = opts.mirror
    var elig = List[Int]()  # physical pairs of executed 2q gates (mirror candidates)
    while True:
        var progressed = False
        while len(stack) > 0:
            var u = stack.pop()
            if dag.is2q[u]:
                var pa = l2p[dag.a[u]]
                var pb = l2p[dag.b[u]]
                if not target.are_connected(pa, pb):
                    if target.distance(pa, pb) >= inf:
                        cost = -1
                        return -1
                    # front_add(u)
                    front_pos[u] = len(front)
                    front.append(u)
                    for side in range(2):
                        var lq = dag.a[u] if side == 0 else dag.b[u]
                        var e = 2 * u + side
                        var h = fhead[lq]
                        fnx[e] = h
                        fpv[e] = -1
                        if h >= 0:
                            fpv[h] = e
                        fhead[lq] = e
                        fcnt[lq] += 1
                        if fcnt[lq] == 2:
                            nmulti += 1
                    continue
            if emit:
                for e in range(dag.emit_start[u], dag.emit_start[u + 1]):
                    _emit_mapped(ops[dag.emit_ops[e]], l2p, out)
            if dag.is2q[u]:
                ev += 1
                var qa = l2p[dag.a[u]]
                var qb = l2p[dag.b[u]]
                last2q[qa] = ev
                last2q[qb] = ev
                last_mc[qa] = dag.mc[u]
                last_mc[qb] = dag.mc[u]
                if mirror_on and dag.look[u]:
                    elig.append(qa)
                    elig.append(qb)
            progressed = True
            for e in range(dag.succ_start[u], dag.succ_start[u + 1]):
                var s = dag.succ[e]
                remaining[s] -= 1
                if remaining[s] == 0:
                    stack.append(s)
        if len(front) == 0:
            break
        step += 1
        if progressed:
            since_progress = 0
            for p in decayed:
                decay[p] = 1.0
            decayed.clear()
            swaps_since_decay = 0
            # rebuild the extended set: the next `lookahead` 2q gates in
            # topological order assuming the front layer were executed
            # (LightSABRE: decrement predecessor counts, restore afterwards).
            # Built level by level and cut in DAG order, so it does not depend
            # on the order gates entered the front: the 2q skeleton (trials)
            # and the full circuit (final pass) see the same set.
            for i in range(len(ext_a)):
                e_first[ext_a[i]] = -1
                e_first[ext_b[i]] = -1
            ext_a.clear()
            ext_b.clear()
            ext_v.clear()
            bfs.clear()
            dec.clear()
            for f in front:
                bfs.append(f)
            var lvl_lo = 0
            var lvl_hi = len(bfs)
            while lvl_lo < lvl_hi:
                for i_vis in range(lvl_lo, lvl_hi):
                    now.clear()
                    now.append(bfs[i_vis])
                    var j_vis = 0
                    while j_vis < len(now):
                        var u = now[j_vis]
                        j_vis += 1
                        for e in range(dag.succ_start[u], dag.succ_start[u + 1]):
                            var s = dag.succ[e]
                            remaining[s] -= 1
                            dec.append(s)
                            if remaining[s] == 0:
                                if dag.look[s]:
                                    ext_v.append(s)
                                    if not layered:
                                        bfs.append(s)
                                else:
                                    now.append(s)
                if layered or len(ext_v) >= opts.lookahead:
                    break
                lvl_lo = lvl_hi
                lvl_hi = len(bfs)
            for s in dec:
                remaining[s] += 1
            # (layer heuristic with commuting fronts: the next layer can hold
            # hundreds of units; it is cut the same way)
            if (not layered or nmulti > 0) and len(ext_v) > opts.lookahead:
                # keep the first `lookahead` in DAG (program) order
                for i in range(1, len(ext_v)):
                    var x = ext_v[i]
                    var j = i - 1
                    while j >= 0 and ext_v[j] > x:
                        ext_v[j + 1] = ext_v[j]
                        j -= 1
                    ext_v[j + 1] = x
                ext_v.resize(opts.lookahead, 0)
            for s in ext_v:
                ext_a.append(dag.a[s])
                ext_b.append(dag.b[s])
            for i in range(len(ext_a)):
                e_next[2 * i] = e_first[ext_a[i]]
                e_first[ext_a[i]] = 2 * i
                e_next[2 * i + 1] = e_first[ext_b[i]]
                e_first[ext_b[i]] = 2 * i + 1
        # release valve: route the closest front gate along a shortest path
        if since_progress > valve:
            valves += 1
            var best_u = front[0]
            var best_d = inf + 1
            var best_k = 1 << 60
            for f in front:
                var d = target.distance(l2p[dag.a[f]], l2p[dag.b[f]])
                var kf = min(l2p[dag.a[f]], l2p[dag.b[f]]) * nphys + max(l2p[dag.a[f]], l2p[dag.b[f]])
                if d < best_d or (d == best_d and kf < best_k):
                    best_d = d
                    best_u = f
                    best_k = kf
            var pa = l2p[dag.a[best_u]]
            var pb = l2p[dag.b[best_u]]
            while target.distance(pa, pb) > 1:
                var nxt = -1
                var dcur = target.distance(pa, pb)
                for k in range(target.adj_start[pa], target.adj_start[pa + 1]):
                    var x = target.adj[k]
                    if target.distance(x, pb) == dcur - 1:
                        nxt = x
                        break
                var la = p2l[pa]
                var lx = p2l[nxt]
                l2p[la] = nxt
                l2p[lx] = pa
                p2l[pa] = lx
                p2l[nxt] = la
                if emit:
                    out.append(Instruction.make(Op.SWAP, pa, nxt))
                if last2q[pa] > 0 and last2q[pa] == last2q[nxt]:
                    acc += last_mc[pa] if last_mc[pa] < 9 else 0
                    absorbed += 1
                else:
                    acc += 3
                ev += 1
                last2q[pa] = -ev
                last2q[nxt] = -ev
                swaps += 1
                pa = nxt
            # release every front gate that is now executable (best_u included)
            rel.clear()
            for f in front:
                if target.are_connected(l2p[dag.a[f]], l2p[dag.b[f]]):
                    rel.append(f)
            for u in rel:
                _front_remove(u, dag, front, front_pos, fhead, fnx, fpv)
                for side in range(2):
                    var lq = dag.a[u] if side == 0 else dag.b[u]
                    fcnt[lq] -= 1
                    if fcnt[lq] == 1:
                        nmulti -= 1
                stack.append(u)
            since_progress = 0
            continue
        # score candidate swaps (raw pointers: this is the hot loop)
        var dp = target.dist.unsafe_ptr()
        var l2pp = l2p.unsafe_ptr()
        var dap = dag.a.unsafe_ptr()
        var dbp = dag.b.unsafe_ptr()
        var eap = ext_a.unsafe_ptr()
        var ebp = ext_b.unsafe_ptr()
        var p2lp = p2l.unsafe_ptr()
        var fhp = fhead.unsafe_ptr()
        var adjp = target.adj.unsafe_ptr()
        var adjs = target.adj_start.unsafe_ptr()
        var l2qp = last2q.unsafe_ptr()
        var multi = nmulti > 0
        var fscale = 1.0
        var fsum = 0
        var nfq = 0  # distinct front qubits (multi mode)
        if multi:
            # nearest-front-partner sum (see `_front_min_delta`)
            fscale = 0.5
            for f in front:
                for side in range(2):
                    var lq = dap[unsafe_offset=f] if side == 0 else dbp[unsafe_offset=f]
                    fmin[lq] = 1 << 30
                    fmin2[lq] = 1 << 30
                    fcmin[lq] = 0
            for f in front:
                var d = dp[unsafe_offset=l2pp[unsafe_offset=dap[unsafe_offset=f]] * nphys + l2pp[unsafe_offset=dbp[unsafe_offset=f]]]
                for side in range(2):
                    var lq = dap[unsafe_offset=f] if side == 0 else dbp[unsafe_offset=f]
                    if d < fmin[lq]:
                        fmin2[lq] = fmin[lq]
                        fmin[lq] = d
                        fcmin[lq] = 1
                    elif d == fmin[lq]:
                        fcmin[lq] += 1
                    elif d < fmin2[lq]:
                        fmin2[lq] = d
            qstamp += 1
            for f in front:
                for side in range(2):
                    var lq = dap[unsafe_offset=f] if side == 0 else dbp[unsafe_offset=f]
                    if qs[lq] != qstamp:
                        qs[lq] = qstamp
                        fsum += fmin[lq]
                        nfq += 1
        else:
            for f in front:
                fsum += dp[unsafe_offset=l2pp[unsafe_offset=dap[unsafe_offset=f]] * nphys + l2pp[unsafe_offset=dbp[unsafe_offset=f]]]
        var esum = 0
        for i in range(len(ext_a)):
            esum += dp[unsafe_offset=l2pp[unsafe_offset=eap[unsafe_offset=i]] * nphys + l2pp[unsafe_offset=ebp[unsafe_offset=i]]]
        var nf = Float64(len(front))
        var ne = Float64(len(ext_a))
        # weights of the front / lookahead distance sums
        var wf: Float64
        var we: Float64
        if layered:
            wf = 1.0  # Qiskit 2.x: constant-scaled basic + one lookahead layer (w / N)
            we = look_w
        else:
            # average front distance: per unit, or per front qubit's nearest
            # unit when qubits hold several commuting units
            wf = 2.0 / Float64(nfq) if multi else 1.0 / nf
            we = opts.weight / ne if ne > 0 else 0.0
        var best_mirror_p = -1
        var best_mirror_x = -1
        var best_mval = 0.0
        if mirror_on and len(elig) > 0:
            # mirror candidates: pairs whose last event is the gate just executed
            # there (free when that block is a >= 3-CX unitary, else 1 CX)
            var keep = 0
            for i in range(len(elig) // 2):
                var p = elig[2 * i]
                var x = elig[2 * i + 1]
                var lp = l2qp[unsafe_offset=p]
                if lp <= 0 or lp != l2qp[unsafe_offset=x]:
                    continue  # superseded by a later gate / swap
                elig[2 * keep] = p
                elig[2 * keep + 1] = x
                keep += 1
                var df = 0
                var de = 0
                _swap_delta(p, x, l2p, p2l, fhead, fnx, dag, ext_a, ext_b, e_first, e_next, target.dist, nphys, not multi, df, de)
                if multi:
                    qstamp += 1
                    df = _front_min_delta(p, x, p2l[p], p2l[x], fhead, fnx, dag, l2p, target.dist, nphys, fmin, fcmin, fmin2, qs, cmin, cat, qstamp)
                # gain in front-distance units
                var gain = -(fscale * Float64(df) + we / wf * Float64(de))
                var mc = last_mc[p]
                if (mc == 0 and gain > 1e-9) or (mc == 1 and gain >= opts.mirror_gain - 1e-9):
                    var val = 3.0 * gain - Float64(mc)
                    # ties -> lowest physical qubit (independent of execution order)
                    if best_mirror_p < 0 or val > best_mval + 1e-9 or (val > best_mval - 1e-9 and min(p, x) < min(best_mirror_p, best_mirror_x)):
                        best_mval = val
                        best_mirror_p = p
                        best_mirror_x = x
            elig.resize(2 * keep, 0)
        var is_mirror = best_mirror_p >= 0
        var sp: Int
        var sx: Int
        if is_mirror:
            sp = best_mirror_p
            sx = best_mirror_x
        else:
            cand_p.clear()
            cand_x.clear()
            for f in front:
                for side in range(2):
                    var lq = dap[unsafe_offset=f] if side == 0 else dbp[unsafe_offset=f]
                    var p = l2pp[unsafe_offset=lq]
                    if pstamp[p] == step:
                        continue  # qubit already expanded (several front units on it)
                    pstamp[p] = step
                    for k in range(adjs[unsafe_offset=p], adjs[unsafe_offset=p + 1]):
                        var x = adjp[unsafe_offset=k]
                        # a swap between two front qubits is generated from both ends: keep one
                        if x < p and fhp[unsafe_offset=p2lp[unsafe_offset=x]] >= 0:
                            continue
                        cand_p.append(p)
                        cand_x.append(x)
            var best = 1e300
            ties.clear()
            var cpp = cand_p.unsafe_ptr()
            var cxp = cand_x.unsafe_ptr()
            for ci in range(len(cand_p)):
                var p = cpp[unsafe_offset=ci]
                var x = cxp[unsafe_offset=ci]
                var df = 0
                var de = 0
                _swap_delta(p, x, l2p, p2l, fhead, fnx, dag, ext_a, ext_b, e_first, e_next, target.dist, nphys, not multi, df, de)
                if multi:
                    qstamp += 1
                    df = _front_min_delta(p, x, p2l[p], p2l[x], fhead, fnx, dag, l2p, target.dist, nphys, fmin, fcmin, fmin2, qs, cmin, cat, qstamp)
                var h = (wf * fscale * Float64(fsum + df) + we * Float64(esum + de)) * max(decay[p], decay[x])
                var lp = l2qp[unsafe_offset=p]
                if lp > 0 and lp == l2qp[unsafe_offset=x]:
                    h *= opts.absorb
                if h < best - 1e-10:
                    best = h
                    ties.clear()
                    ties.append(ci)
                elif h <= best + 1e-10:
                    ties.append(ci)
            if len(ties) > 1:
                # canonical order (the candidate list follows the front order,
                # which differs between the skeleton and the full circuit)
                for i in range(1, len(ties)):
                    var t = ties[i]
                    var kt = min(cand_p[t], cand_x[t]) * nphys + max(cand_p[t], cand_x[t])
                    var j = i - 1
                    while j >= 0 and min(cand_p[ties[j]], cand_x[ties[j]]) * nphys + max(cand_p[ties[j]], cand_x[ties[j]]) > kt:
                        ties[j + 1] = ties[j]
                        j -= 1
                    ties[j + 1] = t
            var pick = ties[0] if len(ties) == 1 else ties[rng.below(len(ties))]
            sp = cand_p[pick]
            sx = cand_x[pick]
        var l1 = p2l[sp]
        var l2 = p2l[sx]
        l2p[l1] = sx
        l2p[l2] = sp
        p2l[sp] = l2
        p2l[sx] = l1
        if emit:
            out.append(Instruction.make(Op.SWAP, sp, sx))
        if last2q[sp] > 0 and last2q[sp] == last2q[sx]:
            absorbed += 1
            acc += last_mc[sp] if last_mc[sp] < 9 else 0
        else:
            acc += 3
        ev += 1
        last2q[sp] = -ev
        last2q[sx] = -ev
        swaps += 1
        if is_mirror:
            mirrors += 1
        else:
            since_progress += 1
            if decay[sp] == 1.0:
                decayed.append(sp)
            if decay[sx] == 1.0:
                decayed.append(sx)
            decay[sp] += opts.decay
            decay[sx] += opts.decay
            swaps_since_decay += 1
            if swaps_since_decay >= 5:
                for p in decayed:
                    decay[p] = 1.0
                decayed.clear()
                swaps_since_decay = 0
        # front gates that became executable
        rel.clear()
        for side in range(2):
            var lq = l1 if side == 0 else l2
            var e = fhead[lq]
            while e >= 0:
                var u = e >> 1
                e = fnx[e]
                if side == 1 and (dag.a[u] == l1 or dag.b[u] == l1):
                    continue  # already seen from l1
                if target.are_connected(l2p[dag.a[u]], l2p[dag.b[u]]):
                    rel.append(u)
        for u in rel:
            _front_remove(u, dag, front, front_pos, fhead, fnx, fpv)
            for side in range(2):
                var lq = dag.a[u] if side == 0 else dag.b[u]
                fcnt[lq] -= 1
                if fcnt[lq] == 1:
                    nmulti -= 1
            stack.append(u)
    if emit:
        _ = valves
    _ = absorbed
    cost = acc
    return swaps


@always_inline
def _front_remove(u: Int, dag: RDag, mut front: List[Int], mut front_pos: List[Int],
                  mut fhead: List[Int], mut fnx: List[Int], mut fpv: List[Int]):
    """Drop unit u from the front list and from both qubits' linked lists."""
    var pos = front_pos[u]
    var lst = front.pop()
    if lst != u:
        front[pos] = lst
        front_pos[lst] = pos
    front_pos[u] = -1
    for side in range(2):
        var lq = dag.a[u] if side == 0 else dag.b[u]
        var e = 2 * u + side
        var p = fpv[e]
        var nx = fnx[e]
        if p >= 0:
            fnx[p] = nx
        else:
            fhead[lq] = nx
        if nx >= 0:
            fpv[nx] = p
        fnx[e] = -1
        fpv[e] = -1


def _front_min_delta(p: Int, x: Int, la: Int, lb: Int, fhead: List[Int], fnx: List[Int],
                     dag: RDag, l2p: List[Int], dist: List[Int], nphys: Int,
                     fmin: List[Int], fcmin: List[Int], fmin2: List[Int],
                     mut qs: List[Int], mut cmin: List[Int], mut cat: List[Int], stamp: Int) -> Int:
    """Change of sum_q min_{front units u on q} d(u) for the swap (p, x).

    With commuting front units several gates wait on one qubit; summing
    all of their distances pulls a hub qubit towards every partner at once
    (plain SABRE oscillates). Counting, per qubit, only its nearest front
    partner makes the router serve partners nearest-first; it equals twice
    the plain sum when every qubit has a single front unit.

    Only units on la / lb change distance: la and lb rescan their units;
    a partner r combines the new distances of its changed units with its
    cached minimum (`fmin`, achieved `fcmin` times; `fmin2` = next value)
    over the unchanged ones, so a candidate costs O(units on la and lb)."""
    var dp = dist.unsafe_ptr()
    var l2pp = l2p.unsafe_ptr()
    var fnp = fnx.unsafe_ptr()
    var fhp = fhead.unsafe_ptr()
    var dap = dag.a.unsafe_ptr()
    var dbp = dag.b.unsafe_ptr()
    var fmp = fmin.unsafe_ptr()
    var fcp = fcmin.unsafe_ptr()
    var f2p = fmin2.unsafe_ptr()
    var qsp = qs.unsafe_ptr()
    var cmp = cmin.unsafe_ptr()
    var cap = cat.unsafe_ptr()
    var df = 0
    var nla = 1 << 30
    var nlb = 1 << 30
    # pass 1: new distances of every unit on la / lb, folded per qubit
    for side in range(2):
        var lq = la if side == 0 else lb
        var e = fhp[unsafe_offset=lq]
        while e >= 0:
            var u = e >> 1
            e = fnp[unsafe_offset=e]
            var qa = dap[unsafe_offset=u]
            var qb = dbp[unsafe_offset=u]
            var r = qb if qa == lq else qa
            if side == 1 and r == la:
                continue  # unit (la, lb): seen from la
            var oa = l2pp[unsafe_offset=qa]
            var ob = l2pp[unsafe_offset=qb]
            var na = x if qa == la else (p if qa == lb else oa)
            var nb = x if qb == la else (p if qb == lb else ob)
            var dn = dp[unsafe_offset=na * nphys + nb]
            if side == 0 or r == la:
                nla = min(nla, dn)
            if side == 1 or r == lb:
                nlb = min(nlb, dn)
            if r != la and r != lb:
                var do = dp[unsafe_offset=oa * nphys + ob]
                var hit = 1 if do == fmp[unsafe_offset=r] else 0
                if qsp[unsafe_offset=r] != stamp:
                    qsp[unsafe_offset=r] = stamp
                    cmp[unsafe_offset=r] = dn
                    cap[unsafe_offset=r] = hit
                else:
                    cmp[unsafe_offset=r] = min(cmp[unsafe_offset=r], dn)
                    cap[unsafe_offset=r] += hit
    if fhp[unsafe_offset=la] >= 0:
        df += nla - fmp[unsafe_offset=la]
    if fhp[unsafe_offset=lb] >= 0:
        df += nlb - fmp[unsafe_offset=lb]
    # pass 2: partners (each once)
    for side in range(2):
        var lq = la if side == 0 else lb
        var e = fhp[unsafe_offset=lq]
        while e >= 0:
            var u = e >> 1
            e = fnp[unsafe_offset=e]
            var qa = dap[unsafe_offset=u]
            var r = dbp[unsafe_offset=u] if qa == lq else qa
            if qsp[unsafe_offset=r] != stamp:
                continue
            qsp[unsafe_offset=r] = -1 - stamp
            var rest = fmp[unsafe_offset=r] if cap[unsafe_offset=r] < fcp[unsafe_offset=r] else f2p[unsafe_offset=r]
            df += min(cmp[unsafe_offset=r], rest) - fmp[unsafe_offset=r]
    return df


@always_inline
def _swap_delta(p: Int, x: Int, l2p: List[Int], p2l: List[Int], fhead: List[Int], fnx: List[Int],
                dag: RDag, ext_a: List[Int], ext_b: List[Int], e_first: List[Int],
                e_next: List[Int], dist: List[Int], nphys: Int, front_part: Bool,
                mut df: Int, mut de: Int):
    """Change of the front (df) and lookahead (de) distance sums if physical
    qubits p and x are swapped (incremental: only units on their wires);
    `front_part` = False skips df (computed by `_front_min_delta`)."""
    var dp = dist.unsafe_ptr()
    var l2pp = l2p.unsafe_ptr()
    var p2lp = p2l.unsafe_ptr()
    var fhp = fhead.unsafe_ptr()
    var fnp = fnx.unsafe_ptr()
    var dap = dag.a.unsafe_ptr()
    var dbp = dag.b.unsafe_ptr()
    var eap = ext_a.unsafe_ptr()
    var ebp = ext_b.unsafe_ptr()
    var efp = e_first.unsafe_ptr()
    var enp = e_next.unsafe_ptr()
    var la = p2lp[unsafe_offset=p]
    var lb = p2lp[unsafe_offset=x]
    df = 0
    de = 0
    for side in range(2):
        if not front_part:
            break
        var lq = la if side == 0 else lb
        var e = fhp[unsafe_offset=lq]
        while e >= 0:
            var u = e >> 1
            e = fnp[unsafe_offset=e]
            var qa = dap[unsafe_offset=u]
            var qb = dbp[unsafe_offset=u]
            if side == 1 and (qa == la or qb == la):
                continue
            var oa = l2pp[unsafe_offset=qa]
            var ob = l2pp[unsafe_offset=qb]
            var na = x if qa == la else (p if qa == lb else oa)
            var nb = x if qb == la else (p if qb == lb else ob)
            df += dp[unsafe_offset=na * nphys + nb] - dp[unsafe_offset=oa * nphys + ob]
    for side in range(2):
        var lq = la if side == 0 else lb
        var slot = efp[unsafe_offset=lq]
        while slot >= 0:
            var i = slot >> 1
            slot = enp[unsafe_offset=slot]
            var qa = eap[unsafe_offset=i]
            var qb = ebp[unsafe_offset=i]
            if side == 1 and (qa == la or qb == la):
                continue
            var oa = l2pp[unsafe_offset=qa]
            var ob = l2pp[unsafe_offset=qb]
            var na = x if qa == la else (p if qa == lb else oa)
            var nb = x if qb == la else (p if qb == lb else ob)
            de += dp[unsafe_offset=na * nphys + nb] - dp[unsafe_offset=oa * nphys + ob]


# ----------------------------------------------------------------------------
# layouts
# ----------------------------------------------------------------------------
def _complete_layout(mut l2p: List[Int], nphys: Int):
    """Fill ancilla wires (value -1) with unused physical qubits in order."""
    var used = List[Bool](length=nphys, fill=False)
    for p in l2p:
        if p >= 0:
            used[p] = True
    var nxt = 0
    for l in range(len(l2p)):
        if l2p[l] < 0:
            while used[nxt]:
                nxt += 1
            l2p[l] = nxt
            used[nxt] = True


def _inverse(l2p: List[Int]) -> List[Int]:
    var p2l = List[Int](length=len(l2p), fill=-1)
    for l in range(len(l2p)):
        p2l[l2p[l]] = l
    return p2l^


def interaction_degrees(ops: List[Instruction], n: Int) -> List[Int]:
    var deg = List[Int](length=n, fill=0)
    for g in ops:
        if op_is_unitary(g.kind) and op_arity(g.kind) == 2:
            deg[g.q0] += 1
            deg[g.q1] += 1
    return deg^


def dense_layout(ops: List[Instruction], n: Int, target: Target) -> List[Int]:
    """BFS-grown connected region from the best-connected / lowest-error qubit."""
    var N = target.num_qubits
    var l2p = List[Int](length=N, fill=-1)
    if target.full:
        for l in range(N):
            l2p[l] = l
        return l2p^
    var start = 0
    var best = -1.0
    for p in range(N):
        var score = Float64(target.degree(p)) - target.error_1q[p] - target.readout_error[p]
        if score > best + 1e-12:
            best = score
            start = p
    var order = List[Int](capacity=N)
    var seen = List[Bool](length=N, fill=False)
    seen[start] = True
    order.append(start)
    var head = 0
    while head < len(order):
        var u = order[head]
        head += 1
        for k in range(target.adj_start[u], target.adj_start[u + 1]):
            var x = target.adj[k]
            if not seen[x]:
                seen[x] = True
                order.append(x)
    for p in range(N):
        if not seen[p]:
            order.append(p)
    # most-interacting logical qubits get the earliest (central) slots
    var deg = interaction_degrees(ops, n)
    var logs = List[Int](capacity=n)
    for l in range(n):
        logs.append(l)
    for i in range(1, n):
        var x = logs[i]
        var j = i - 1
        while j >= 0 and deg[logs[j]] < deg[x]:
            logs[j + 1] = logs[j]
            j -= 1
        logs[j + 1] = x
    for i in range(n):
        l2p[logs[i]] = order[i]
    _complete_layout(l2p, N)
    return l2p^


def vf2_layout(ops: List[Instruction], n: Int, target: Target, max_steps: Int,
               mut l2p: List[Int], relax: List[Int] = List[Int](), relax_d: Int = 0,
               root_phys: Int = -1) -> Bool:
    """Exact embedding of the interaction graph (subgraph monomorphism).

    VF2++-style: BFS order from the highest-degree vertex, candidates are
    neighbours of an already-mapped neighbour's image tried in Warnsdorff
    order (fewest free neighbours first, which threads paths / rings along
    the boundary), degree pruning, and distance pruning
    d_coupling(p_l, p_m) <= d_interaction(l, m) for every mapped m.
    `relax` (flat pairs) are extra soft edges that must land within `relax_d`
    hops (used for edges dropped from a relaxed query, e.g. a ring's closure).
    """
    var N = target.num_qubits
    var seen = Dict[Int, Bool]()
    var nbr_start = List[Int](length=n + 1, fill=0)
    var ea = List[Int]()
    var eb = List[Int]()
    for g in ops:
        if op_is_unitary(g.kind) and op_arity(g.kind) == 2:
            var a = min(g.q0, g.q1)
            var b = max(g.q0, g.q1)
            var key = a * n + b
            if key in seen:
                continue
            seen[key] = True
            ea.append(a)
            eb.append(b)
        elif op_is_unitary(g.kind) and op_arity(g.kind) == 3:
            return False
    var m = len(ea)
    if m > target.num_edges():
        return False
    for i in range(m):
        nbr_start[ea[i] + 1] += 1
        nbr_start[eb[i] + 1] += 1
    for l in range(n):
        nbr_start[l + 1] += nbr_start[l]
    var nbr = List[Int](length=2 * m, fill=0)
    var fillp = nbr_start.copy()
    for i in range(m):
        nbr[fillp[ea[i]]] = eb[i]
        fillp[ea[i]] += 1
        nbr[fillp[eb[i]]] = ea[i]
        fillp[eb[i]] += 1
    var maxdeg_t = 0
    for p in range(N):
        maxdeg_t = max(maxdeg_t, target.degree(p))
    for l in range(n):
        if nbr_start[l + 1] - nbr_start[l] > maxdeg_t:
            return False
    # interaction-graph distances (for pruning)
    var inf = 4 * n + 4
    var di = List[Int](length=n * n, fill=inf)
    var queue = List[Int](length=n, fill=0)
    for s in range(n):
        di[s * n + s] = 0
        var head = 0
        var tail = 1
        queue[0] = s
        while head < tail:
            var u = queue[head]
            head += 1
            for k in range(nbr_start[u], nbr_start[u + 1]):
                var x = nbr[k]
                if di[s * n + x] > di[s * n + u] + 1:
                    di[s * n + x] = di[s * n + u] + 1
                    queue[tail] = x
                    tail += 1
    # soft edges shorten interaction distances (so the pruning steers e.g. a
    # ring's two ends back towards each other)
    if relax_d > 0:
        for _ in range(2):
            for r in range(len(relax) // 2):
                var ra = relax[2 * r]
                var rb = relax[2 * r + 1]
                for x in range(n):
                    var dxa = di[x * n + ra]
                    var dxb = di[x * n + rb]
                    for y in range(n):
                        var v1 = dxa + relax_d + di[rb * n + y]
                        var v2 = dxb + relax_d + di[ra * n + y]
                        var cur = di[x * n + y]
                        if v1 < cur:
                            cur = v1
                        if v2 < cur:
                            cur = v2
                        di[x * n + y] = cur
    # order: BFS from max-degree vertex of each component
    var order = List[Int](capacity=n)
    var placed = List[Bool](length=n, fill=False)
    var parent = List[Int](length=n, fill=-1)
    while len(order) < n:
        var root = -1
        var rd = -1
        for l in range(n):
            if not placed[l] and nbr_start[l + 1] - nbr_start[l] > rd:
                rd = nbr_start[l + 1] - nbr_start[l]
                root = l
        if len(order) == 0 and len(relax) >= 2:
            # start at an end of a soft edge: the walk then has to come back
            root = relax[0]
            rd = nbr_start[root + 1] - nbr_start[root]
        if rd == 0:
            for l in range(n):
                if not placed[l]:
                    placed[l] = True
                    order.append(l)
            break
        placed[root] = True
        order.append(root)
        var head = len(order) - 1
        while head < len(order):
            var u = order[head]
            head += 1
            for k in range(nbr_start[u], nbr_start[u + 1]):
                var x = nbr[k]
                if not placed[x]:
                    placed[x] = True
                    parent[x] = u
                    order.append(x)
    var assign = List[Int](length=n, fill=-1)
    var used = List[Bool](length=N, fill=False)
    var cands = List[List[Int]]()
    for _ in range(n):
        cands.append(List[Int]())
    var cand_idx = List[Int](length=n + 1, fill=0)
    var built = List[Bool](length=n + 1, fill=False)
    var depth = 0
    var steps = 0
    while depth >= 0 and depth < n:
        steps += 1
        if steps > max_steps:
            return False
        var l = order[depth]
        var deg_l = nbr_start[l + 1] - nbr_start[l]
        if deg_l == 0:
            depth += 1
            if depth < n:
                cand_idx[depth] = 0
                built[depth] = False
            continue
        if assign[l] >= 0:
            used[assign[l]] = False
            assign[l] = -1
        if not built[depth]:
            # candidate list in Warnsdorff order
            var lst = List[Int]()
            var anchor = assign[parent[l]] if parent[l] >= 0 else -1
            if anchor >= 0:
                for k in range(target.adj_start[anchor], target.adj_start[anchor + 1]):
                    lst.append(target.adj[k])
            elif depth == 0 and root_phys >= 0:
                lst.append(root_phys)
            else:
                for p in range(N):
                    lst.append(p)
            var key = List[Int](length=len(lst), fill=0)
            for i in range(len(lst)):
                var free = 0
                var p = lst[i]
                for k in range(target.adj_start[p], target.adj_start[p + 1]):
                    if not used[target.adj[k]]:
                        free += 1
                key[i] = free * (N + 1) + p
            for i in range(1, len(lst)):
                var kv = key[i]
                var pv = lst[i]
                var j = i - 1
                while j >= 0 and key[j] > kv:
                    key[j + 1] = key[j]
                    lst[j + 1] = lst[j]
                    j -= 1
                key[j + 1] = kv
                lst[j + 1] = pv
            cands[depth] = lst^
            built[depth] = True
            cand_idx[depth] = 0
        var found = False
        while cand_idx[depth] < len(cands[depth]):
            var p = cands[depth][cand_idx[depth]]
            cand_idx[depth] += 1
            if used[p] or target.degree(p) < deg_l:
                continue
            var ok = True
            for k in range(nbr_start[l], nbr_start[l + 1]):
                var a = assign[nbr[k]]
                if a >= 0 and not target.are_connected(p, a):
                    ok = False
                    break
            if ok and relax_d > 0:
                for r in range(len(relax) // 2):
                    var ra = relax[2 * r]
                    var rb = relax[2 * r + 1]
                    var other = rb if ra == l else (ra if rb == l else -1)
                    if other >= 0 and assign[other] >= 0 and target.distance(p, assign[other]) > relax_d:
                        ok = False
                        break
            if ok:
                # distance pruning against every mapped vertex of this component
                for d2 in range(depth):
                    var mq = order[d2]
                    var pm = assign[mq]
                    if pm >= 0 and di[l * n + mq] < inf and target.distance(p, pm) > di[l * n + mq]:
                        ok = False
                        break
            if ok:
                assign[l] = p
                used[p] = True
                found = True
                break
        if found:
            depth += 1
            if depth < n:
                cand_idx[depth] = 0
                built[depth] = False
        else:
            built[depth] = False
            depth -= 1
            while depth >= 0 and nbr_start[order[depth] + 1] - nbr_start[order[depth]] == 0:
                depth -= 1
    if depth < 0:
        return False
    l2p = List[Int](length=N, fill=-1)
    for l in range(n):
        if assign[l] >= 0:
            l2p[l] = assign[l]
    _complete_layout(l2p, N)
    return True


def relaxed_vf2_seed(ops: List[Instruction], n: Int, target: Target, maxdeg: Int,
                     steps: Int) -> List[Int]:
    """Multi-start embedding of the max-weight spanning forest; the start whose
    dropped edges end up shortest (sum of hop distances) wins. Empty if none."""
    var dropped = List[Int]()
    var tree = spanning_path_ops(ops, n, maxdeg, dropped)
    var best = List[Int]()
    if len(tree) == 0:
        return best^
    var N = target.num_qubits
    var bscore = -1
    var tries = min(N, 160)
    var fails = 0
    for r in range(tries):
        var l = List[Int]()
        if not vf2_layout(tree, n, target, max(steps // 8, 2000), l, dropped, 0, (r * 7919) % N):
            fails += 1
            if fails > 24 and len(best) == 0:
                break
            continue
        var score = 0
        for i in range(len(dropped) // 2):
            score += target.distance(l[dropped[2 * i]], l[dropped[2 * i + 1]])
        if bscore < 0 or score < bscore:
            bscore = score
            best = l^
            if score <= len(dropped) // 2:
                break  # every dropped edge already adjacent
    return best^


def spanning_path_ops(ops: List[Instruction], n: Int, maxdeg: Int, mut dropped: List[Int]) -> List[Instruction]:
    """Max-weight spanning forest of the interaction graph (Kruskal), as CX ops.

    Used as a relaxed VF2 query: when the full interaction graph does not
    embed (odd rings on bipartite lattices, a few long-range edges) its
    heaviest spanning forest often does and gives an excellent SABRE seed.
    """
    var w = Dict[Int, Int]()
    for g in ops:
        if op_is_unitary(g.kind) and op_arity(g.kind) == 2:
            var k = min(g.q0, g.q1) * n + max(g.q0, g.q1)
            w[k] = w.get(k, 0) + 1
    var keys = List[Int]()
    var wts = List[Int]()
    for kv in w.items():
        keys.append(kv.key)
        wts.append(kv.value)
    # sort edges by weight desc, then key asc (insertion sort is fine for sparse graphs;
    # use a counting pass when large)
    var idx = List[Int](length=len(keys), fill=0)
    for i in range(len(keys)):
        idx[i] = i
    for i in range(1, len(idx)):
        var v = idx[i]
        var j = i - 1
        while j >= 0 and (wts[idx[j]] < wts[v] or (wts[idx[j]] == wts[v] and keys[idx[j]] > keys[v])):
            idx[j + 1] = idx[j]
            j -= 1
        idx[j + 1] = v
    var parent = List[Int](length=n, fill=0)
    var deg = List[Int](length=n, fill=0)
    for i in range(n):
        parent[i] = i
    var out = List[Instruction]()
    for t in idx:
        var a = keys[t] // n
        var b = keys[t] % n
        if deg[a] >= maxdeg or deg[b] >= maxdeg:
            dropped.append(a)
            dropped.append(b)
            continue  # keep the tree embeddable (degree bound of the coupling graph)
        var ra = a
        while parent[ra] != ra:
            ra = parent[ra]
        var rb = b
        while parent[rb] != rb:
            rb = parent[rb]
        if ra == rb:
            dropped.append(a)
            dropped.append(b)
            continue
        parent[ra] = rb
        deg[a] += 1
        deg[b] += 1
        out.append(Instruction.make(Op.CX, a, b))
    return out^


struct InteractionGraph(Movable):
    """Weighted interaction graph (CSR): 2q gate counts plus temporal-partner edges."""

    var n: Int
    var start: List[Int]
    var nbr: List[Int]
    var w: List[Float64]
    var total: List[Float64]

    def __init__(out self, ops: List[Instruction], n: Int, temporal: Float64):
        self.n = n
        var key_w = Dict[Int, Float64]()
        var last = List[Int](length=n, fill=-1)
        for g in ops:
            if not op_is_unitary(g.kind) or op_arity(g.kind) != 2:
                continue
            var a = g.q0
            var b = g.q1
            var k = min(a, b) * n + max(a, b)
            key_w[k] = key_w.get(k, 0.0) + 1.0
            if temporal > 0.0:
                for side in range(2):
                    var q = a if side == 0 else b
                    var other = b if side == 0 else a
                    var lp = last[q]
                    if lp >= 0 and lp != other:
                        var k2 = min(lp, other) * n + max(lp, other)
                        key_w[k2] = key_w.get(k2, 0.0) + temporal
            last[a] = b
            last[b] = a
        var deg = List[Int](length=n + 1, fill=0)
        var ka = List[Int]()
        var kb = List[Int]()
        var kw = List[Float64]()
        for kv in key_w.items():
            var a = kv.key // n
            var b = kv.key % n
            ka.append(a)
            kb.append(b)
            kw.append(kv.value)
            deg[a + 1] += 1
            deg[b + 1] += 1
        for i in range(n):
            deg[i + 1] += deg[i]
        self.start = deg.copy()
        self.nbr = List[Int](length=2 * len(ka), fill=0)
        self.w = List[Float64](length=2 * len(ka), fill=0.0)
        self.total = List[Float64](length=n, fill=0.0)
        var fillp = deg^
        for e in range(len(ka)):
            var a = ka[e]
            var b = kb[e]
            self.nbr[fillp[a]] = b
            self.w[fillp[a]] = kw[e]
            fillp[a] += 1
            self.nbr[fillp[b]] = a
            self.w[fillp[b]] = kw[e]
            fillp[b] += 1
            self.total[a] += kw[e]
            self.total[b] += kw[e]


def greedy_layout(ig: InteractionGraph, target: Target, start_phys: Int) -> List[Int]:
    """Prim-ordered weighted embedding (TKET GraphPlacement-like, no backtracking).

    Logical qubits are placed in order of strongest connection to the placed
    set; each goes to the free physical qubit minimising sum w * distance to
    its placed neighbours (Warnsdorff tie-break: fewest free neighbours), so
    chains become snakes, rings become near-closed paths and dense clusters
    stay compact.
    """
    var n = ig.n
    var N = target.num_qubits
    var l2p = List[Int](length=N, fill=-1)
    var used = List[Bool](length=N, fill=False)
    var conn = List[Float64](length=n, fill=0.0)
    var placed = List[Bool](length=n, fill=False)
    var cost = List[Float64](length=N, fill=0.0)
    var touched = List[Int]()
    for _ in range(n):
        var l = -1
        var bc = -1.0
        for x in range(n):
            if not placed[x] and conn[x] > bc + 1e-12:
                bc = conn[x]
                l = x
        if bc <= 0.0:
            # new component: heaviest remaining logical
            var bt = -1.0
            for x in range(n):
                if not placed[x] and ig.total[x] > bt + 1e-12:
                    bt = ig.total[x]
                    l = x
            if bt <= 0.0:
                break  # only isolated qubits left
        var best_p = -1
        if bc <= 0.0:
            # nearest free physical to the trial's start qubit
            var bd = N + 2
            for p in range(N):
                if not used[p] and target.distance(start_phys, p) < bd:
                    bd = target.distance(start_phys, p)
                    best_p = p
        else:
            var best_cost = 1e300
            var best_free = 1 << 30
            for p in range(N):
                if used[p]:
                    continue
                var cst = 0.0
                for k in range(ig.start[l], ig.start[l + 1]):
                    var m = ig.nbr[k]
                    if placed[m]:
                        cst += ig.w[k] * Float64(target.distance(p, l2p[m]))
                if cst > best_cost + 1e-9:
                    continue
                var free = 0
                for k in range(target.adj_start[p], target.adj_start[p + 1]):
                    if not used[target.adj[k]]:
                        free += 1
                if cst < best_cost - 1e-9 or free < best_free:
                    best_cost = cst
                    best_free = free
                    best_p = p
        if best_p < 0:
            break
        l2p[l] = best_p
        used[best_p] = True
        placed[l] = True
        for k in range(ig.start[l], ig.start[l + 1]):
            conn[ig.nbr[k]] += ig.w[k]
    _ = cost
    _ = touched
    _complete_layout(l2p, N)
    return l2p^


def _random_layout(N: Int, mut rng: Rng) -> List[Int]:
    var l2p = List[Int](length=N, fill=0)
    for i in range(N):
        l2p[i] = i
    var k = N - 1
    while k > 0:
        var j = rng.below(k + 1)
        var t = l2p[k]
        l2p[k] = l2p[j]
        l2p[j] = t
        k -= 1
    return l2p^


def _layout_trial(t: Int, ops: List[Instruction], fwd: RDag, rev: RDag, target: Target,
                  opts: RouteOptions, n: Int, random_only: Bool, ig: InteractionGraph,
                  starts: List[Int], seed_layout: List[Int], mut start: List[Int]) -> Int:
    """One SABRE layout trial: seed layout, forward/backward refinement, score.

    Seeds rotate between greedy weighted embeddings from different start
    qubits, the dense layout and random permutations (SabreLayout trials).
    """
    var N = target.num_qubits
    var trng = Rng(UInt64(opts.seed)).split(t)
    var o = opts.copy()
    o.heuristic = trial_heuristic(opts.heuristic, t)
    var l2p: List[Int]
    if (t == 0 or t == 2) and len(seed_layout) == N and not random_only:
        l2p = seed_layout.copy()  # relaxed-VF2 embedding of the spanning forest
    elif random_only or t % 4 == 3:
        l2p = _random_layout(N, trng)
    elif t == 1:
        l2p = dense_layout(ops, n, target)
    else:
        l2p = greedy_layout(ig, target, starts[t % len(starts)])
    var p2l = _inverse(l2p)
    var scratch = List[Instruction]()
    if not random_only:
        for _ in range(max(opts.iterations, 0)):
            var c1 = 0
            var m1 = 0
            var s1 = sabre_pass(ops, fwd, target, l2p, p2l, trng.next_u64(), False, scratch, o, c1, m1)
            if s1 < 0:
                break
            var c2 = 0
            var m2 = 0
            var s2 = sabre_pass(ops, rev, target, l2p, p2l, trng.next_u64(), False, scratch, o, c2, m2)
            if s2 < 0:
                break
    start = l2p.copy()
    var cst = 0
    var mm = 0
    var sw = sabre_pass(ops, fwd, target, l2p, p2l, UInt64(opts.seed) + 17, False, scratch, o, cst, mm)
    return cst if sw >= 0 else -1


def resolve_mirror(opts: RouteOptions, run_cost: List[Int]) -> RouteOptions:
    """`mirror_auto`: mirror-gate routing pays off when most 2q runs are
    generic >= 3-CX blocks (QV, SU(4)-structured circuits: mirroring them is
    free); on CX / CP-structured circuits the 1-CX mirrors were measured to
    cost more than they save. Auto therefore enables *free* mirrors only, and
    only when >= 50% of the routing skeleton's runs are such blocks."""
    var o = opts.copy()
    if not opts.mirror_auto:
        return o^
    var free = 0
    for k in run_cost:
        if k >= 3 and k < 100:
            free += 1
    o.mirror = len(run_cost) > 0 and 2 * free >= len(run_cost)
    o.mirror_gain = 1e9  # free mirrors only
    return o^


def resolve_commute(opts: RouteOptions, u: RUnits) -> Bool:
    """`commute_auto`: commutation-aware routing pays off on circuits whose
    2q units are mostly diagonal (QAOA cost layers, ZZ Trotter steps: cp /
    rzz / cx.rz.cx commute and the router may serve them nearest-first);
    on CX networks (Clifford) and QFT the program order was measured to be
    the better routing order, so auto enables it only when >= 50% of the
    runs are diagonal (Z, Z) units."""
    if not opts.commute_auto:
        return opts.commute
    var runs = 0
    var diag = 0
    for t in range(u.n):
        if u.is_run[t]:
            runs += 1
            if u.ax_a[t] == AX_Z and u.ax_b[t] == AX_Z:
                diag += 1
    return runs > 0 and 2 * diag >= runs


def trial_heuristic(h: String, t: Int) -> String:
    """`auto`: alternate LightSABRE extended-set and Qiskit-2.x layer heuristics."""
    if h != "auto":
        return h
    return "layer" if t % 2 == 1 else "extended"


def routing_skeleton(ops: List[Instruction], n: Int) -> List[Instruction]:
    """2q gates only, with back-to-back repeats on the same pair collapsed."""
    var rc = List[Int]()
    return routing_skeleton(ops, n, rc)


def routing_skeleton(ops: List[Instruction], n: Int, mut run_cost: List[Int]) -> List[Instruction]:
    """2q gates only, back-to-back repeats on the same pair collapsed;
    `run_cost[i]` = summed CX cost of the run skeleton gate i stands for."""
    var last = List[Int](length=n, fill=-1)  # index into out of last 2q gate per wire
    var out = List[Instruction]()
    run_cost.clear()
    for g in ops:
        if not op_is_unitary(g.kind) or op_arity(g.kind) != 2:
            continue
        var la = last[g.q0]
        if la >= 0 and la == last[g.q1]:
            run_cost[la] += kind_cost_2q(g.kind)
            continue  # same pair as the previous 2q gate on both wires
        last[g.q0] = len(out)
        last[g.q1] = len(out)
        out.append(Instruction.make(Op.CX, g.q0, g.q1))
        run_cost.append(kind_cost_2q(g.kind))
    return out^


def _start_qubits(target: Target, k: Int) -> List[Int]:
    """Spread-out seed qubits: the most central one, then farthest-point sampling."""
    var N = target.num_qubits
    var out = List[Int]()
    var best = 0
    var bs = -1
    for p in range(N):
        var s = 0
        for q in range(N):
            s += target.distance(p, q)
        if bs < 0 or s < bs:
            bs = s
            best = p
    out.append(best)
    var mind = List[Int](length=N, fill=1 << 30)
    while len(out) < max(k, 1):
        var last = out[len(out) - 1]
        var far = -1
        var fd = -1
        for p in range(N):
            mind[p] = min(mind[p], target.distance(p, last))
            if mind[p] > fd:
                fd = mind[p]
                far = p
        if fd <= 0:
            break
        out.append(far)
    return out^


struct RoutePlan(Copyable):
    """Chosen initial layout (wire -> physical, length N) + skeleton swap estimate."""

    var layout: List[Int]
    var est_swaps: Int  # estimated routing cost in CX units (3 per swap, 1 if absorbable)
    var vf2: Bool
    var trivial: Bool  # all-to-all target: nothing to route
    var note: String
    var heuristic: String  # heuristic of the winning trial (used by finish_route)
    var mirror: Bool  # mirror-gate routing (resolved, see `resolve_mirror`)
    var mirror_gain: Float64
    var commute: Bool  # commutation-aware DAG (resolved, see `resolve_commute`)
    var reduced: Bool  # commuting fronts: trial counts were reduced

    def __init__(out self):
        self.layout = List[Int]()
        self.est_swaps = 0
        self.vf2 = False
        self.trivial = False
        self.note = String()
        self.heuristic = "extended"
        self.mirror = False
        self.mirror_gain = 1.0
        self.commute = False
        self.reduced = False


def _check_routable(c: Circuit, target: Target) raises:
    var n = c.num_qubits
    var N = target.num_qubits
    if n > N:
        raise Error("QM-E605: circuit has " + String(n) + " qubits but target '" + target.name + "' has only " + String(N) + ". Hint: pick a larger target (e.g. Target.grid / heavy_hex) or reduce the circuit width.")
    for g in c.ops:
        if op_is_unitary(g.kind) and op_arity(g.kind) == 3:
            raise Error("QM-E607: routing needs gates on at most 2 qubits; found '" + String(g) + "'. Hint: run 'unroll3q' (or 'basis') before 'route'.")


def plan_route(c: Circuit, target: Target, opts: RouteOptions) raises -> RoutePlan:
    """Layout stage: VF2 exact embedding, else parallel SABRE layout trials.

    Layout and swap decisions are computed on the 2q skeleton (1q gates,
    measurements and back-to-back repeats on one pair never change a SABRE
    decision), so this is cheap even for 1q-heavy circuits.
    """
    _check_routable(c, target)
    var plan = RoutePlan()
    var n = c.num_qubits
    var N = target.num_qubits
    if target.full:
        plan.trivial = True
        plan.layout = List[Int](length=n, fill=0)
        for i in range(n):
            plan.layout[i] = i
        plan.note = "all-to-all target: no routing needed"
        return plan^
    var lay = opts.layout
    var src = List[Int]()
    var skel = routing_skeleton(c.ops, n, src)
    var units = RUnits(c.ops, n, opts.commute or opts.commute_auto)
    plan.commute = resolve_commute(opts, units)
    var units_off = RUnits(c.ops, n, False)
    var sfwd = RDag(c.ops, units if plan.commute else units_off, n, False, True)
    var ropts = resolve_mirror(opts, src)
    plan.mirror = ropts.mirror
    plan.mirror_gain = ropts.mirror_gain
    # Routing variants. Large circuits: the per-circuit rules above. Small
    # ones (<= 300 2q units, trials cost microseconds) run a portfolio:
    # commutation-aware DAG on / off (when units can commute) x free mirror
    # gates on / off (when >= 3-CX blocks exist); every variant gets the full
    # set of layout trials (same seeds, paired) and the cheapest estimate wins.
    var v_comm = List[Bool]()
    var v_mir = List[Bool]()
    var any_comm = False
    var any_free = False
    for t in range(units.n):
        if units.is_run[t]:
            if units.ax_a[t] != AX_OTHER or units.ax_b[t] != AX_OTHER:
                any_comm = True
            if units.cost[t] >= 3 and units.cost[t] < 100:
                any_free = True
    var small = sfwd.n2q <= 300
    for ci in range(2):
        var cm = ci == 1
        if not (small and opts.commute_auto and any_comm) and cm != plan.commute:
            continue
        for mi in range(2):
            var mr = mi == 1
            if not (small and opts.mirror_auto and any_free) and mr != plan.mirror:
                continue
            v_comm.append(cm)
            v_mir.append(mr)
    var nv = len(v_comm)
    if nv > 1:
        ropts.mirror_gain = 1e9  # portfolio mirrors are free ones only
    if nv == 1 and plan.commute and sfwd.commuting > 0.25:
        # commuting fronts make a SABRE pass several times costlier (many
        # front units per qubit); the extra order freedom more than pays for
        # fewer trials (QAOA: -20% 2q with 2/5 of the trials)
        ropts.trials = max(4, ropts.trials * 2 // 5)
        ropts.iterations = min(ropts.iterations, 1)
        plan.reduced = True
    var best = List[Int]()
    var fixed = len(opts.initial_layout) > 0
    if fixed:
        best = List[Int](length=N, fill=-1)
        for l in range(min(n, len(opts.initial_layout))):
            best[l] = opts.initial_layout[l]
        _complete_layout(best, N)
        lay = "given"
    if not fixed and (lay == "auto" or lay == "vf2"):
        var l2p = List[Int]()
        if vf2_layout(skel, n, target, opts.vf2_steps, l2p):
            best = l2p^
            plan.vf2 = True
        elif lay == "vf2":
            plan.note = "vf2 found no exact embedding; "
    var est = -1
    if not plan.vf2 and not fixed:
        if lay == "trivial":
            best = List[Int](length=N, fill=0)
            for i in range(N):
                best[i] = i
        elif lay == "dense":
            best = dense_layout(c.ops, n, target)
        else:
            # SabreLayout: independent seeded trials (greedy / dense / random
            # seeds, forward-backward refinement) in parallel, best kept
            var ops = skel.copy()
            var f_on = RDag(c.ops, units, n, False, True)
            var r_on = RDag(c.ops, units, n, True, True)
            var f_off = RDag(c.ops, units_off, n, False, True)
            var r_off = RDag(c.ops, units_off, n, True, True)
            var TB = max(ropts.trials, 1)
            var T = TB * nv
            var sws = List[Int](length=T, fill=0)
            var lays = List[Int](length=T * N, fill=0)
            var swp = sws.unsafe_ptr().as_unsafe_any_origin()
            var lyp = lays.unsafe_ptr().as_unsafe_any_origin()
            var random_only = lay == "random"
            var ig = InteractionGraph(ops, n, 0.5)
            var starts = _start_qubits(target, TB)
            var seed_layout = List[Int]()
            if not random_only:
                var maxdeg = 0
                for p in range(N):
                    maxdeg = max(maxdeg, target.degree(p))
                seed_layout = relaxed_vf2_seed(ops, n, target, maxdeg, opts.vf2_steps)

            def task(t: Int) {imm ops, imm f_on, imm r_on, imm f_off, imm r_off, imm target, imm ropts, imm n, imm N, imm random_only, imm swp, imm lyp, imm ig, imm starts, imm seed_layout, imm nv, imm v_comm, imm v_mir}:
                var start = List[Int]()
                var v = t % nv
                var o = ropts.copy()
                o.mirror = v_mir[v]
                var sw: Int
                if v_comm[v]:
                    sw = _layout_trial(t // nv, ops, f_on, r_on, target, o, n, random_only, ig, starts, seed_layout, start)
                else:
                    sw = _layout_trial(t // nv, ops, f_off, r_off, target, o, n, random_only, ig, starts, seed_layout, start)
                swp[unsafe_offset=t] = sw
                for i in range(N):
                    lyp[unsafe_offset=t * N + i] = start[i]

            if T == 1:
                task(0)
            else:
                parallelize(task, T)
            var bt = -1
            for t in range(T):
                if sws[t] >= 0 and (bt < 0 or sws[t] < sws[bt]):
                    bt = t
            if bt < 0:
                raise Error("QM-E606: interacting qubits ended up in disconnected parts of the coupling map of '" + target.name + "'. Hint: use a connected coupling graph (Target.from_edges) or reduce the circuit to one component.")
            best = List[Int](length=N, fill=0)
            for i in range(N):
                best[i] = lays[bt * N + i]
            est = sws[bt]
            plan.heuristic = trial_heuristic(opts.heuristic, bt // nv)
            plan.commute = v_comm[bt % nv]
            plan.mirror = v_mir[bt % nv]
            if nv > 1:
                plan.mirror_gain = 1e9
    if plan.vf2:
        est = 0
    elif est < 0:
        var l = best.copy()
        var pl = _inverse(l)
        var scratch = List[Instruction]()
        var o = opts.copy()
        o.heuristic = trial_heuristic(opts.heuristic, 0)
        o.mirror = plan.mirror
        o.mirror_gain = plan.mirror_gain
        plan.heuristic = o.heuristic
        var cst = 0
        var mm = 0
        est = sabre_pass(skel, sfwd, target, l, pl, UInt64(opts.seed) + 17, False, scratch, o, cst, mm)
        if est >= 0:
            est = cst
        if est < 0:
            raise Error("QM-E606: interacting qubits are in disconnected parts of the coupling map of '" + target.name + "'. Hint: use a connected coupling graph or a layout that keeps interacting qubits in one component.")
    plan.layout = best^
    plan.est_swaps = est
    plan.note += "vf2 exact layout" if plan.vf2 else "layout=" + lay
    return plan^


def has_exact_layout(c: Circuit, target: Target, vf2_steps: Int) -> Bool:
    """True when VF2 embeds the interaction graph (routing needs no swap)."""
    if target.full:
        return True
    var l2p = List[Int]()
    return vf2_layout(routing_skeleton(c.ops, c.num_qubits), c.num_qubits, target, vf2_steps, l2p)


def finish_route(mut c: Circuit, target: Target, opts: RouteOptions, plan: RoutePlan,
                 mut init_layout: List[Int], mut final_layout: List[Int],
                 mut note: String) raises -> Int:
    """Routing stage: swap trials on the skeleton, then emit the cheapest seed."""
    var n = c.num_qubits
    var N = target.num_qubits
    note = plan.note
    if plan.trivial:
        init_layout = plan.layout.copy()
        final_layout = plan.layout.copy()
        return 0
    var units = RUnits(c.ops, n, plan.commute)
    var fwd = RDag(c.ops, units, n, False, False)
    var best = plan.layout.copy()
    init_layout = best.copy()
    var hopts = opts.copy()
    hopts.heuristic = plan.heuristic
    hopts.mirror = plan.mirror
    hopts.mirror_gain = plan.mirror_gain
    if len(c.ops) > 6000:
        # absorbable-swap portfolio only pays off on small circuits; on large
        # ones the cost estimate is too noisy, plain swap count wins
        hopts.absorb_alt = hopts.absorb
    var best_seed = UInt64(opts.seed) + 17
    var ST = max(opts.swap_trials, 1)
    if plan.reduced:
        ST = max(4, ST * 2 // 5)
    if ST > 1 and not plan.vf2:
        var sfwd = RDag(c.ops, units, n, False, True)
        var sres = List[Int](length=ST, fill=0)
        var srp = sres.unsafe_ptr().as_unsafe_any_origin()
        var seed0 = UInt64(opts.seed) + 17

        def strial(k: Int) {imm c, imm sfwd, imm target, imm hopts, imm best, imm srp, imm seed0}:
            var l = best.copy()
            var pl = _inverse(l)
            var scratch = List[Instruction]()
            var o = hopts.copy()
            if k % 2 == 1:
                o.absorb = hopts.absorb_alt
            var cst = 0
            var mm = 0
            var sw = sabre_pass(c.ops, sfwd, target, l, pl, seed0 + UInt64(k) * 7919, False, scratch, o, cst, mm)
            srp[unsafe_offset=k] = cst if sw >= 0 else -1

        parallelize(strial, ST)
        var bk = 0
        for k in range(ST):
            if sres[k] >= 0 and (sres[bk] < 0 or sres[k] < sres[bk]):
                bk = k
        best_seed = seed0 + UInt64(bk) * 7919
        if bk % 2 == 1:
            hopts.absorb = hopts.absorb_alt
    var l2p = best^
    var p2l = _inverse(l2p)
    var out_ops = List[Instruction](capacity=len(c.ops) + len(c.ops) // 2)
    var fcost = 0
    var nmir = 0
    var swaps = sabre_pass(c.ops, fwd, target, l2p, p2l, best_seed, True, out_ops, hopts, fcost, nmir)
    if swaps < 0:
        raise Error("QM-E606: interacting qubits are in disconnected parts of the coupling map of '" + target.name + "'. Hint: use a connected coupling graph or a layout that keeps interacting qubits in one component.")
    final_layout = l2p^
    var slid = slide_swaps(out_ops, target)
    var out = shell_like(c, N)
    out.ops = out_ops^
    c = out^
    note += ", swaps=" + String(swaps)
    if nmir > 0:
        note += " (" + String(nmir) + " mirrors)"
    if slid > 0:
        note += " (" + String(slid) + " slid onto 2q blocks)"
    return swaps


def route_circuit(mut c: Circuit, target: Target, opts: RouteOptions,
                  mut init_layout: List[Int], mut final_layout: List[Int],
                  mut note: String) raises -> Int:
    """Map circuit wires onto physical qubits and insert SWAPs. Returns #swaps."""
    var plan = plan_route(c, target, opts)
    return finish_route(c, target, opts, plan, init_layout, final_layout, note)


# ----------------------------------------------------------------------------
# Star pre-routing (Qiskit StarPreRouting): star -> line by absorbed swaps
# ----------------------------------------------------------------------------
def star_preroute(c: Circuit, min_leaves: Int, mut out: Circuit, mut wire_map: List[Int]) -> Int:
    """`star_preroute` without the run statistics (see the 5-argument form)."""
    var cov = 0.0
    return star_preroute(c, min_leaves, out, wire_map, cov)


def star_preroute(c: Circuit, min_leaves: Int, mut out: Circuit, mut wire_map: List[Int],
                  mut long_cov: Float64) -> Int:
    """Rewrite runs of 2q blocks sharing a centre qubit into a swap chain.

    For blocks B1(c,q1), B2(c,q2), ..., Bk(c,qk) (each a run of gates on one
    pair, partners distinct) a SWAP(c, qi) is appended after every Bi but the
    last, and all later operations are relabelled, so the centre "walks" along
    q1 -> q2 -> ... and the interaction graph becomes a path. Each inserted SWAP
    sits next to a gate on the same pair, so KAK later absorbs it (SWAP.CX = 2
    CX). Semantics are exact: the wire permutation is returned in `wire_map`
    (original wire w ends on wire wire_map[w]). Returns #swaps inserted.
    `long_cov` = fraction of 2q blocks inside runs of >= 5 blocks (QFT ~0.95,
    synthesised Cliffords ~0.5, QV / QAOA / lattice Trotter steps < 0.05):
    the pass manager uses it to decide which variants are worth planning.
    """
    var n = len(c.ops)
    var nq = c.num_qubits
    # 1. pair-blocks in 2q program order (-1 entries = hard breaks)
    var bl_a = List[Int]()
    var bl_b = List[Int]()
    var bl_last = List[Int]()
    for i in range(n):
        var g = c.ops[i]
        var a = op_arity(g.kind)
        if is_all_wire_op(g) or (a >= 1 and not (op_is_unitary(g.kind) and g.cond < 0)) or (op_is_unitary(g.kind) and a == 3):
            bl_a.append(-1)
            bl_b.append(-1)
            bl_last.append(i)
            continue
        if a != 2:
            continue
        var lo = min(g.q0, g.q1)
        var hi = max(g.q0, g.q1)
        var nb = len(bl_a)
        if nb > 0 and bl_a[nb - 1] == lo and bl_b[nb - 1] == hi:
            bl_last[nb - 1] = i
            continue
        bl_a.append(lo)
        bl_b.append(hi)
        bl_last.append(i)
    # 2. runs sharing a centre with distinct partners
    var swap_after = List[Int](length=n, fill=-1)  # op index -> partner (centre in swap_c)
    var swap_c = List[Int](length=n, fill=-1)
    var used = List[Int](length=nq, fill=-1)  # run id that used this partner
    var nbk = len(bl_a)
    var b = 0
    var run_id = 0
    var inserted = 0
    var long_blocks = 0
    while b < nbk:
        if bl_a[b] < 0 or b + 1 >= nbk or bl_a[b + 1] < 0:
            b += 1
            continue
        # centre = qubit shared by blocks b and b+1
        var cen = -1
        if bl_a[b] == bl_a[b + 1] or bl_a[b] == bl_b[b + 1]:
            cen = bl_a[b]
        elif bl_b[b] == bl_a[b + 1] or bl_b[b] == bl_b[b + 1]:
            cen = bl_b[b]
        if cen < 0:
            b += 1
            continue
        run_id += 1
        var e = b
        while e < nbk and bl_a[e] >= 0 and (bl_a[e] == cen or bl_b[e] == cen):
            var other = bl_b[e] if bl_a[e] == cen else bl_a[e]
            if used[other] == run_id:
                break
            used[other] = run_id
            e += 1
        var k = e - b
        if k >= min_leaves:
            if k >= 5:
                long_blocks += k
            for j in range(b, e - 1):
                var other = bl_b[j] if bl_a[j] == cen else bl_a[j]
                swap_after[bl_last[j]] = other
                swap_c[bl_last[j]] = cen
                inserted += 1
            b = e
        else:
            b += 1
    var nreal = 0
    for j in range(nbk):
        if bl_a[j] >= 0:
            nreal += 1
    long_cov = Float64(long_blocks) / Float64(max(nreal, 1))
    wire_map = List[Int](length=nq, fill=0)
    for w in range(nq):
        wire_map[w] = w
    out = shell_like(c, nq)
    if inserted == 0:
        out.ops = c.ops.copy()
        return 0
    var ops = List[Instruction](capacity=n + inserted)
    for i in range(n):
        var x = c.ops[i]
        var a = op_arity(x.kind)
        if a >= 1 and x.q0 >= 0:
            x.q0 = wire_map[x.q0]
        if a >= 2:
            x.q1 = wire_map[x.q1]
        if a >= 3:
            x.q2 = wire_map[x.q2]
        ops.append(x)
        if swap_after[i] >= 0:
            var cw = swap_c[i]
            var pw = swap_after[i]
            ops.append(Instruction.make(Op.SWAP, wire_map[cw], wire_map[pw]))
            var t = wire_map[cw]
            wire_map[cw] = wire_map[pw]
            wire_map[pw] = t
    out.ops = ops^
    return inserted


# ----------------------------------------------------------------------------
# Swap sliding: move each routing SWAP next to a 2q gate on the same pair
# ----------------------------------------------------------------------------
def _touches(g: Instruction, q: Int) -> Bool:
    var a = op_arity(g.kind)
    if a >= 1 and g.q0 == q:
        return True
    if a >= 2 and g.q1 == q:
        return True
    if a >= 3 and g.q2 == q:
        return True
    return False


def _relabel(mut g: Instruction, a: Int, b: Int):
    var ar = op_arity(g.kind)
    if ar >= 1:
        if g.q0 == a: g.q0 = b
        elif g.q0 == b: g.q0 = a
    if ar >= 2:
        if g.q1 == a: g.q1 = b
        elif g.q1 == b: g.q1 = a
    if ar >= 3:
        if g.q2 == a: g.q2 = b
        elif g.q2 == b: g.q2 = a


def slide_swaps(mut ops: List[Instruction], target: Target, limit: Int = 256) -> Int:
    """SWAP.G == G'.SWAP with G' = G relabelled a <-> b. Each routing SWAP(a,b)
    is slid backwards (else forwards) through the gates on wires a / b until it
    meets a 2q gate on the same pair, as long as every relabelled 2q gate stays
    on a coupling edge; the 2q block {G, SWAP} is then absorbed by KAK (cx +
    swap = 2 cx, cp + swap = 3 cx, swap + swap = nothing). Returns #moved."""
    var n = len(ops)
    var nxt = List[Int](length=n, fill=-1)
    var prv = List[Int](length=n, fill=-1)
    for i in range(n):
        nxt[i] = i + 1 if i + 1 < n else -1
        prv[i] = i - 1
    var head = 0 if n > 0 else -1
    var moved = 0
    var touched = List[Int]()
    for s in range(n):
        var sw = ops[s]
        if sw.kind != Op.SWAP or sw.cond >= 0:
            continue
        var a = sw.q0
        var b = sw.q1
        var done = False
        for direction in range(2):
            if done:
                break
            touched.clear()
            var j = prv[s] if direction == 0 else nxt[s]
            var steps = 0
            var found = -1
            while j >= 0 and steps < limit:
                var g = ops[j]
                steps += 1
                if g.kind == Op.BARRIER or g.kind == Op.TICK:
                    break
                var ta = _touches(g, a)
                var tb = _touches(g, b)
                if ta or tb:
                    var ar = op_arity(g.kind)
                    if ta and tb and ar == 2 and op_is_unitary(g.kind) and g.cond < 0:
                        found = j
                        break
                    if ar >= 2 and not (ta and tb):
                        # the gate moves to the other end of the swap
                        var other = -1
                        for k in range(ar):
                            var q = g.qubit(k)
                            if q != a and q != b:
                                other = q
                        var mapped = b if ta else a
                        if ar > 2 or (op_is_unitary(g.kind) and not target.are_connected(mapped, other)):
                            break
                    touched.append(j)
                j = prv[j] if direction == 0 else nxt[j]
            if found < 0 or len(touched) == 0:
                if found >= 0:
                    done = True  # already adjacent
                continue
            for t in touched:
                _relabel(ops[t], a, b)
            # unlink s
            var ps = prv[s]
            var ns = nxt[s]
            if ps >= 0:
                nxt[ps] = ns
            else:
                head = ns
            if ns >= 0:
                prv[ns] = ps
            if direction == 0:
                # insert after `found`
                var nf = nxt[found]
                nxt[found] = s
                prv[s] = found
                nxt[s] = nf
                if nf >= 0:
                    prv[nf] = s
            else:
                # insert before `found`
                var pf = prv[found]
                prv[found] = s
                nxt[s] = found
                prv[s] = pf
                if pf >= 0:
                    nxt[pf] = s
                else:
                    head = s
            moved += 1
            done = True
    if moved > 0:
        var out = List[Instruction](capacity=n)
        var i = head
        while i >= 0:
            out.append(ops[i])
            i = nxt[i]
        ops = out^
    return moved
