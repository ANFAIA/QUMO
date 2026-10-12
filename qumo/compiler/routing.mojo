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
from qumo.compiler.common import shell_like, is_all_wire_op
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
    var mirror_auto: Bool  # portfolio: trials alternate mirror on / off, the cheapest estimate wins
    var mirror_gain: Float64  # heuristic gain (front-distance units) a 1-CX mirror must reach (free mirrors need > 0)

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
# DAG over the flat op list (CSR successors)
# ----------------------------------------------------------------------------
struct RDag(Movable):
    var n: Int
    var succ_start: List[Int]
    var succ: List[Int]
    var npred: List[Int]
    var is2q: List[Bool]
    var a: List[Int]
    var b: List[Int]
    var op_index: List[Int]
    var look: List[Bool]  # 2q gate that counts for the lookahead (not a same-pair repeat)
    var mc: List[Int]  # extra 2q cost of mirroring the pair right after this gate (0: >= 3-CX block, else 1)

    def __init__(out self, ops: List[Instruction], nq: Int, reverse: Bool,
                 run_cost: List[Int] = List[Int]()):
        """`run_cost[i]` (optional, per op) = 2q cost of the same-pair run op i
        stands for (routing skeleton); otherwise runs are accumulated here."""
        var n = len(ops)
        self.n = n
        self.npred = List[Int](length=n, fill=0)
        self.is2q = List[Bool](length=n, fill=False)
        self.a = List[Int](length=n, fill=-1)
        self.b = List[Int](length=n, fill=-1)
        self.op_index = List[Int](length=n, fill=0)
        self.look = List[Bool](length=n, fill=False)
        self.mc = List[Int](length=n, fill=1)
        var cum = List[Int](length=n, fill=0)
        var have_rc = len(run_cost) == n
        var last2 = List[Int](length=nq, fill=-1)
        var src = List[Int](capacity=2 * n)
        var dst = List[Int](capacity=2 * n)
        var last = List[Int](length=nq, fill=-1)
        var last_cl = -1
        for v in range(n):
            var oi = n - 1 - v if reverse else v
            self.op_index[v] = oi
            var g = ops[oi]
            var k = g.kind
            var chain = False
            if k == Op.BARRIER or k == Op.TICK:
                for w in range(nq):
                    if last[w] >= 0:
                        src.append(last[w])
                        dst.append(v)
                    last[w] = v
                chain = True
            else:
                var ar = op_arity(k)
                for s in range(ar):
                    var w = g.qubit(s)
                    if last[w] >= 0:
                        src.append(last[w])
                        dst.append(v)
                    last[w] = v
                if not op_is_unitary(k) and not op_is_noise(k):
                    chain = True
                if g.cond >= 0:
                    chain = True
                if op_is_unitary(k) and ar == 2:
                    # conditioned gates need adjacency too (routing SWAPs stay unconditioned)
                    self.is2q[v] = True
                    self.a[v] = g.q0
                    self.b[v] = g.q1
                    var pa = last2[g.q0]
                    var rep = pa >= 0 and pa == last2[g.q1]
                    self.look[v] = not rep
                    var tot = run_cost[oi] if have_rc else kind_cost_2q(k) + (cum[pa] if rep else 0)
                    cum[v] = tot
                    self.mc[v] = 9 if tot >= 100 else (0 if tot >= 3 else 1)
                    last2[g.q0] = v
                    last2[g.q1] = v
            if chain:
                if last_cl >= 0:
                    src.append(last_cl)
                    dst.append(v)
                last_cl = v
        self.succ_start = List[Int](length=n + 1, fill=0)
        for e in range(len(src)):
            self.succ_start[src[e] + 1] += 1
            self.npred[dst[e]] += 1
        for v in range(n):
            self.succ_start[v + 1] += self.succ_start[v]
        self.succ = List[Int](length=len(src), fill=0)
        var fillp = self.succ_start.copy()
        for e in range(len(src)):
            self.succ[fillp[src[e]]] = dst[e]
            fillp[src[e]] += 1


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


@always_inline
def _swap_delta(p: Int, x: Int, l2p: List[Int], p2l: List[Int], front_of_log: List[Int],
                dag: RDag, ext_a: List[Int], ext_b: List[Int], e_first: List[Int],
                e_next: List[Int], dist: List[Int], nphys: Int, mut df: Int, mut de: Int):
    """Change of the front (df) and lookahead (de) distance sums if physical
    qubits p and x are swapped (incremental: only gates on their wires)."""
    var dp = dist.unsafe_ptr()
    var l2pp = l2p.unsafe_ptr()
    var p2lp = p2l.unsafe_ptr()
    var folp = front_of_log.unsafe_ptr()
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
    var g1 = folp[unsafe_offset=la]
    var g2 = folp[unsafe_offset=lb]
    if g1 >= 0:
        var qa = dap[unsafe_offset=g1]
        var qb = dbp[unsafe_offset=g1]
        var oa = l2pp[unsafe_offset=qa]
        var ob = l2pp[unsafe_offset=qb]
        var na = x if qa == la else (p if qa == lb else oa)
        var nb = x if qb == la else (p if qb == lb else ob)
        df += dp[unsafe_offset=na * nphys + nb] - dp[unsafe_offset=oa * nphys + ob]
    if g2 >= 0 and g2 != g1:
        var qa = dap[unsafe_offset=g2]
        var qb = dbp[unsafe_offset=g2]
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


def sabre_pass(ops: List[Instruction], dag: RDag, target: Target, mut l2p: List[Int],
               mut p2l: List[Int], seed: UInt64, emit: Bool, mut out: List[Instruction],
               opts: RouteOptions,
               mut cost: Int, mut mirrors: Int) -> Int:
    """Route one pass; updates l2p/p2l to the final layout; returns #swaps
    emitted, mirrors included (-1 when two interacting qubits sit in
    disconnected components).

    `cost` estimates the routing overhead in CX units after KAK: 3 per plain
    SWAP, and for a SWAP on a pair whose last event was a gate on that same
    pair (KAK merges them into one block) 0 when that block already costs
    >= 3 CX (a generic SU(4): G.SWAP is again one SU(4)) or 1 otherwise
    (CX.SWAP = 2 CX, CP.SWAP = 3 CX).

    Mirror-gate routing (MIRAGE, McKinney et al. 2024): before a swap is
    chosen, every SWAP on the pair of a gate that has just executed is
    scored with the same heuristic; when it lowers the cost (free mirrors)
    or lowers the front distance by >= `mirror_gain` (1-CX mirrors) it is
    applied without counting as a routing step. `mirrors` counts them."""
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
    var front_of_log = List[Int](length=nphys, fill=-1)
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
                    front_pos[u] = len(front)
                    front.append(u)
                    front_of_log[dag.a[u]] = u
                    front_of_log[dag.b[u]] = u
                    continue
            if emit:
                _emit_mapped(ops[dag.op_index[u]], l2p, out)
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
            if not layered and len(ext_v) > opts.lookahead:
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
                var kf = min(l2p[dag.a[f]], l2p[dag.b[f]])
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
            # now executable
            var lst = front.pop()
            if lst != best_u:
                var pos = front_pos[best_u]
                front[pos] = lst
                front_pos[lst] = pos
            front_pos[best_u] = -1
            front_of_log[dag.a[best_u]] = -1
            front_of_log[dag.b[best_u]] = -1
            stack.append(best_u)
            # any other front gate made adjacent by the walk is released too
            var fi = 0
            while fi < len(front):
                var u = front[fi]
                if target.are_connected(l2p[dag.a[u]], l2p[dag.b[u]]):
                    var lst2 = front.pop()
                    if lst2 != u:
                        front[fi] = lst2
                        front_pos[lst2] = fi
                    front_pos[u] = -1
                    front_of_log[dag.a[u]] = -1
                    front_of_log[dag.b[u]] = -1
                    stack.append(u)
                    continue
                fi += 1
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
        var folp = front_of_log.unsafe_ptr()
        var adjp = target.adj.unsafe_ptr()
        var adjs = target.adj_start.unsafe_ptr()
        var l2qp = last2q.unsafe_ptr()
        var fsum = 0
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
            wf = 1.0 / nf
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
                _swap_delta(p, x, l2p, p2l, front_of_log, dag, ext_a, ext_b, e_first, e_next, target.dist, nphys, df, de)
                # gain in front-distance units
                var gain = -(Float64(df) + we / wf * Float64(de))
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
                    for k in range(adjs[unsafe_offset=p], adjs[unsafe_offset=p + 1]):
                        var x = adjp[unsafe_offset=k]
                        # a swap between two front qubits is generated from both ends: keep one
                        if x < p and folp[unsafe_offset=p2lp[unsafe_offset=x]] >= 0:
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
                _swap_delta(p, x, l2p, p2l, front_of_log, dag, ext_a, ext_b, e_first, e_next, target.dist, nphys, df, de)
                var h = (wf * Float64(fsum + df) + we * Float64(esum + de)) * max(decay[p], decay[x])
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
        if l1 >= 0:
            l2p[l1] = sx
        if l2 >= 0:
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
        for side in range(2):
            var lq = l1 if side == 0 else l2
            if lq < 0:
                continue
            var u = front_of_log[lq]
            if u < 0:
                continue
            if target.are_connected(l2p[dag.a[u]], l2p[dag.b[u]]):
                var lst = front.pop()
                if lst != u:
                    var pos = front_pos[u]
                    front[pos] = lst
                    front_pos[lst] = pos
                front_pos[u] = -1
                front_of_log[dag.a[u]] = -1
                front_of_log[dag.b[u]] = -1
                stack.append(u)
    if emit:
        _ = valves
    _ = absorbed
    cost = acc
    return swaps


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
    o.mirror = trial_mirror(opts, t)
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


def trial_mirror(opts: RouteOptions, t: Int) -> Bool:
    """`mirror_auto`: trials 2, 3 (mod 4) route with mirror gates, 0, 1 without
    (crossed with the heuristic alternation)."""
    if not opts.mirror_auto:
        return opts.mirror
    return t % 4 >= 2


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
    var mirror: Bool  # mirror-gate routing in the winning trial

    def __init__(out self):
        self.layout = List[Int]()
        self.est_swaps = 0
        self.vf2 = False
        self.trivial = False
        self.note = String()
        self.heuristic = "extended"
        self.mirror = False


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
    var sfwd = RDag(skel, n, False, src)
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
            var rev = RDag(ops, n, True, src)
            var T = max(opts.trials, 1)
            var sws = List[Int](length=T, fill=0)
            var lays = List[Int](length=T * N, fill=0)
            var swp = sws.unsafe_ptr().as_unsafe_any_origin()
            var lyp = lays.unsafe_ptr().as_unsafe_any_origin()
            var random_only = lay == "random"
            var ig = InteractionGraph(ops, n, 0.5)
            var starts = _start_qubits(target, T)
            var seed_layout = List[Int]()
            if not random_only:
                var maxdeg = 0
                for p in range(N):
                    maxdeg = max(maxdeg, target.degree(p))
                seed_layout = relaxed_vf2_seed(ops, n, target, maxdeg, opts.vf2_steps)

            def task(t: Int) {imm ops, imm sfwd, imm rev, imm target, imm opts, imm n, imm N, imm random_only, imm swp, imm lyp, imm ig, imm starts, imm seed_layout}:
                var start = List[Int]()
                var sw = _layout_trial(t, ops, sfwd, rev, target, opts, n, random_only, ig, starts, seed_layout, start)
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
            plan.heuristic = trial_heuristic(opts.heuristic, bt)
            plan.mirror = trial_mirror(opts, bt)
    if plan.vf2:
        est = 0
    elif est < 0:
        var l = best.copy()
        var pl = _inverse(l)
        var scratch = List[Instruction]()
        var o = opts.copy()
        o.heuristic = trial_heuristic(opts.heuristic, 0)
        o.mirror = trial_mirror(opts, 0)
        plan.heuristic = o.heuristic
        plan.mirror = o.mirror
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
    var fwd = RDag(c.ops, n, False)
    var best = plan.layout.copy()
    init_layout = best.copy()
    var hopts = opts.copy()
    hopts.heuristic = plan.heuristic
    hopts.mirror = plan.mirror
    if len(c.ops) > 6000:
        # absorbable-swap portfolio only pays off on small circuits; on large
        # ones the cost estimate is too noisy, plain swap count wins
        hopts.absorb_alt = hopts.absorb
    var best_seed = UInt64(opts.seed) + 17
    var ST = max(opts.swap_trials, 1)
    if ST > 1 and not plan.vf2:
        var src = List[Int]()
        var skel = routing_skeleton(c.ops, n, src)
        var sfwd = RDag(skel, n, False, src)
        var sres = List[Int](length=ST, fill=0)
        var srp = sres.unsafe_ptr().as_unsafe_any_origin()
        var seed0 = UInt64(opts.seed) + 17

        def strial(k: Int) {imm skel, imm sfwd, imm target, imm hopts, imm best, imm srp, imm seed0}:
            var l = best.copy()
            var pl = _inverse(l)
            var scratch = List[Instruction]()
            var o = hopts.copy()
            if k % 2 == 1:
                o.absorb = hopts.absorb_alt
            if hopts.mirror_auto and k % 3 == 2:
                o.mirror = not hopts.mirror
            var cst = 0
            var mm = 0
            var sw = sabre_pass(skel, sfwd, target, l, pl, seed0 + UInt64(k) * 7919, False, scratch, o, cst, mm)
            srp[unsafe_offset=k] = cst if sw >= 0 else -1

        parallelize(strial, ST)
        var bk = 0
        for k in range(ST):
            if sres[k] >= 0 and (sres[bk] < 0 or sres[k] < sres[bk]):
                bk = k
        best_seed = seed0 + UInt64(bk) * 7919
        if bk % 2 == 1:
            hopts.absorb = hopts.absorb_alt
        if hopts.mirror_auto and bk % 3 == 2:
            hopts.mirror = not hopts.mirror
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
