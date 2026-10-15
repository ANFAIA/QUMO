"""Learned-policy layout and routing (hardware-neutral AIRouting).

The algorithm is derived from qiskit-ibm-transpiler 0.18.0 `rust/src/routing`
(Apache-2.0): an RL policy picks one SWAP at a time among the "active swaps"
(coupling edges touching the qubits of the next HORIZON topological layers),
from a sign-encoded distance-change observation, with a tiny 128 -> H -> 16
MLP; episodes are stochastic (seeded shuffles), run in parallel, and wrapped
in a bidirectional layout search. Re-implemented from a written spec; no IBM
code or weights are used. QUMO's router models are trained locally
(`training/ai_transpiler/route`, format `qumo.rlroute.v0`).

QUMO additions:
  * Determinism: every shuffle uses `Rng(seed).split(...)`; results do not
    depend on the thread count; the effort budget is counted in environment
    steps (not wall time), so a budget hit is reproducible too.
  * Progress fallback: after 2n steps without progress (no gate executed and
    the front-distance sum not decreasing) the shortest-path greedy SWAP of the
    closest front gate is taken until a gate executes, so every episode ends.
  * Exactness: the routed circuit is verified structurally (every 2q op on a
    coupling edge; undoing the SWAPs with the layouts replays every input op
    exactly once in a dependency-consistent order) before it is returned.
  * Portfolio: with `portfolio=True` QUMO's SABRE (`route_circuit`) also runs
    and the cheaper result by `routed_cost` wins (never worse than SABRE).
  * Modes: `policy=<id>|auto` (MLP argmax), `scored[:<id>]` (ablation: the
    policy's log-probabilities as a bonus on a SABRE-style score of the same
    candidates), `heuristic` (no model: SABRE-style score only), `off`.

Layout conventions are those of `routing.route_circuit`: `init_layout[w]` /
`final_layout[w]` = physical qubit of wire w before / after (length N, wires
n..N-1 are ancillas).
"""

from std.collections import Dict
from std.os import getenv
from std.os.path import exists
from std.memory import bitcast
from std.math import exp, log
from std.bit import count_trailing_zeros
from qumo.ir.gates import Op, op_arity, op_is_unitary
from qumo.ir.circuit import Circuit, Instruction
from qumo.compiler.target import Target
from qumo.compiler.common import shell_like, is_all_wire_op
from qumo.compiler.basis import unroll_3q
from qumo.compiler.routing import (
    RouteOptions, kind_cost_2q, route_circuit, plan_route, finish_route,
    vf2_layout, dense_layout, greedy_layout, InteractionGraph, routing_skeleton,
    slide_swaps,
)
from qumo.ai.json import parse_json, JsonValue
from qumo.compiler.policy import (
    load_model_file, model_search_dirs, read_model_index, MODEL_FORMAT_ROUTE, MODEL_KIND_ROUTING,
)
from qumo.sim.rng import Rng
from max.algorithm import parallelize


comptime HORIZON = 8
comptime NUM_ACTIVE_SWAPS = 16
comptime OBS_SIZE = HORIZON * NUM_ACTIVE_SWAPS  # 128

# node kinds of the routing problem
comptime NK_SINGLE = 0  # plain 1q gate (always executable)
comptime NK_UNIT = 1  # run of plain gates on one wire pair (needs adjacency)
comptime NK_FENCE = 2  # non-plain op (measure / reset / conditioned / noise / annotation)
comptime NK_BARRIER = 3  # all-wire fence

# output item kinds
comptime IT_NODE = 0
comptime IT_SWAP = 1
comptime IT_FUSED = 2  # routing SWAP fused with a single CX: CX(a,b) CX(b,a)

# decision modes
comptime MODE_POLICY = 0
comptime MODE_SCORED = 1
comptime MODE_HEURISTIC = 2

# Effort per level. runs / its / reps follow IBM's OP_LEVELS (parallel
# episodes, bidirectional iterations, restarts); QUMO adds two deterministic
# caps so compile time stays proportionate on QUMO-scale circuits: a step
# budget (max_ms * STEPS_PER_MS environment steps; max_ms defaults 300 / 2000
# / 10000) and an episode cap (64 / 192 / 1024 episodes). Both are counted in
# work units, never wall time, so results are reproducible.
comptime STEPS_PER_MS = 400  # env steps per ms on one Apple M5 core (policy mode, measured)


def level_effort(level: Int) -> List[Int]:
    """[runs, its, reps, max_ms, max_episodes] of an effort level (1..3)."""
    if level <= 1:
        return [8, 2, 2, 300, 64]
    if level == 2:
        return [16, 16, 8, 2000, 192]
    return [32, 16, 8, 10000, 1024]


struct AIRouteOptions(Copyable):
    var policy: String  # auto | off | heuristic | scored[:<id>] | <model id>
    var level: Int  # 1..3 (effort)
    var layout_mode: String  # optimize | improve | keep
    var seed_layout: List[Int]  # wire -> physical (used by improve / optimize)
    var seed: Int
    var model_dir: String
    var portfolio: Bool  # also run SABRE and keep the cheaper result
    var max_ms: Int  # 0 = per-level default (converted to a deterministic step budget)

    def __init__(out self):
        self.policy = "auto"
        self.level = 2
        self.layout_mode = "optimize"
        self.seed_layout = List[Int]()
        self.seed = 7
        self.model_dir = String()
        self.portfolio = True
        self.max_ms = 0


# ----------------------------------------------------------------------------
# Cost of a routed circuit
# ----------------------------------------------------------------------------
def _gate_cx(k: Int) -> Int:
    return 3 if k == Op.SWAP else kind_cost_2q(k)


def routed_cost(c: Circuit, target: Target) -> Tuple[Int, Int]:
    """(2q cost in CX units, 2q depth) of a routed circuit.

    CX-like gates cost 1, SWAP 3, other 2q kinds 2 (`routing.kind_cost_2q`).
    A run of consecutive 2q gates on the same pair (1q gates in between
    allowed) is one two-qubit unitary that KAK resynthesises with at most 3
    CX, so a run costs min(3, sum): this charges a SWAP next to a 2q gate on
    the same pair what it costs after consolidation, for SABRE and the policy
    alike. The 2q depth counts every 2q gate as one layer (a SWAP as 3).
    """
    var nq = max(c.num_qubits, target.num_qubits, 1)
    var run = List[Int](length=nq, fill=-1)
    var rsum = List[Int]()
    var rq = List[Int]()  # run wires (2 per run)
    var lvl = List[Int](length=nq, fill=0)
    var depth = 0
    var cost = 0
    for g in c.ops:
        var ar = op_arity(g.kind)
        if is_all_wire_op(g):
            for w in range(nq):
                var r = run[w]
                if r >= 0:
                    cost += min(3, rsum[r])
                    run[rq[2 * r]] = -1
                    run[rq[2 * r + 1]] = -1
            continue
        var plain = op_is_unitary(g.kind) and g.cond < 0
        if not plain or ar != 2:
            if plain and ar == 1:
                continue
            for s in range(ar):
                var w = g.qubit(s)
                if w < 0 or w >= nq:
                    continue
                var r = run[w]
                if r >= 0:
                    cost += min(3, rsum[r])
                    run[rq[2 * r]] = -1
                    run[rq[2 * r + 1]] = -1
            if op_is_unitary(g.kind) and ar == 2:
                cost += _gate_cx(g.kind)
                var d0 = max(lvl[g.q0], lvl[g.q1]) + (3 if g.kind == Op.SWAP else 1)
                lvl[g.q0] = d0
                lvl[g.q1] = d0
                depth = max(depth, d0)
            continue
        var a = g.q0
        var b = g.q1
        var cx = _gate_cx(g.kind)
        var d = max(lvl[a], lvl[b]) + (3 if g.kind == Op.SWAP else 1)
        lvl[a] = d
        lvl[b] = d
        depth = max(depth, d)
        var ra = run[a]
        if ra >= 0 and ra == run[b]:
            rsum[ra] += cx
            continue
        for side in range(2):
            var w = a if side == 0 else b
            var r = run[w]
            if r >= 0:
                cost += min(3, rsum[r])
                run[rq[2 * r]] = -1
                run[rq[2 * r + 1]] = -1
        var nr = len(rsum)
        rsum.append(cx)
        rq.append(a)
        rq.append(b)
        run[a] = nr
        run[b] = nr
    for w in range(nq):
        var r = run[w]
        if r >= 0:
            cost += min(3, rsum[r])
            run[rq[2 * r]] = -1
            run[rq[2 * r + 1]] = -1
    return (cost, depth)


def _full_score(c: Circuit, target: Target) -> List[Int]:
    """Lexicographic score (cx units, 2q depth, depth, gates)."""
    var rc = routed_cost(c, target)
    return [rc[0], rc[1], c.depth(), c.num_unitary_gates()]


def _better(a: List[Int], b: List[Int]) -> Bool:
    """a strictly better than b (lexicographic; empty b = no result yet)."""
    if len(b) == 0:
        return len(a) > 0
    if len(a) == 0:
        return False
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return a[i] < b[i]
    return False


# ----------------------------------------------------------------------------
# Routing problem: ops grouped into nodes with a dependency DAG
# ----------------------------------------------------------------------------
struct RProblem(Copyable):
    """Circuit as routing nodes.

    Plain 2q gates on one wire pair and the plain 1q gates following them on
    those wires (while the pair stays open) form a UNIT (one routing decision,
    like IBM's MakeBlocks); other plain 1q gates are SINGLE nodes; barriers
    fence every wire; measure / reset / conditioned / noise / annotation ops
    are FENCE nodes that also keep their program order through a classical
    chain. Node ids follow program order.
    """

    var n: Int  # circuit wires
    var ops: List[Instruction]
    var kind: List[Int]
    var wa: List[Int]  # first wire (-1 if none); for 2-wire nodes wa < wb
    var wb: List[Int]  # second wire (-1 if none)
    var op_start: List[Int]  # CSR of op indices per node
    var op_idx: List[Int]
    var succ_start: List[Int]  # CSR successors (sorted, unique)
    var succ: List[Int]
    var indeg: List[Int]
    var succ2_start: List[Int]  # 2q-only DAG (between 2-wire nodes)
    var succ2: List[Int]
    var npred2: List[Int]
    var n2q: Int  # number of 2-wire nodes
    var cx_total: Int  # summed CX cost of the input 2q gates
    var ucost: List[Int]  # summed CX cost of the 2q gates of each node

    def __init__(out self, ops: List[Instruction], n: Int):
        self.n = n
        self.ops = ops.copy()
        self.kind = List[Int]()
        self.wa = List[Int]()
        self.wb = List[Int]()
        self.op_start = List[Int]()
        self.op_idx = List[Int]()
        self.succ_start = List[Int]()
        self.succ = List[Int]()
        self.indeg = List[Int]()
        self.succ2_start = List[Int]()
        self.succ2 = List[Int]()
        self.npred2 = List[Int]()
        self.n2q = 0
        self.cx_total = 0
        self.ucost = List[Int]()
        # pass 1: nodes (ops lists in temporary per-node lists via linked chains)
        var node_ops = List[List[Int]]()
        var preds = List[List[Int]]()
        var last = List[Int](length=max(n, 1), fill=-1)
        var last2 = List[Int](length=max(n, 1), fill=-1)
        var open = List[Int](length=max(n, 1), fill=-1)
        var last_cl = -1
        var preds2 = List[List[Int]]()
        for i in range(len(ops)):
            var g = ops[i]
            var ar = op_arity(g.kind)
            var plain = op_is_unitary(g.kind) and g.cond < 0
            if op_is_unitary(g.kind) and ar == 2:
                self.cx_total += _gate_cx(g.kind)
            if is_all_wire_op(g):
                for w in range(n):
                    self._close(open, w)
                var u = len(self.kind)
                self.kind.append(NK_BARRIER)
                self.wa.append(-1)
                self.wb.append(-1)
                node_ops.append([i])
                var pl = List[Int]()
                for w in range(n):
                    _add_unique(pl, last[w])
                    last[w] = u
                preds.append(pl^)
                preds2.append(List[Int]())
                continue
            if plain and ar == 2:
                var a = g.q0
                var b = g.q1
                if open[a] >= 0 and open[a] == open[b]:
                    node_ops[open[a]].append(i)
                    continue
                self._close(open, a)
                self._close(open, b)
                var u = len(self.kind)
                self.kind.append(NK_UNIT)
                self.wa.append(min(a, b))
                self.wb.append(max(a, b))
                node_ops.append([i])
                var pl = List[Int]()
                _add_unique(pl, last[a])
                _add_unique(pl, last[b])
                preds.append(pl^)
                var p2 = List[Int]()
                _add_unique(p2, last2[a])
                _add_unique(p2, last2[b])
                preds2.append(p2^)
                last[a] = u
                last[b] = u
                last2[a] = u
                last2[b] = u
                open[a] = u
                open[b] = u
                continue
            if plain and ar == 1:
                var a = g.q0
                if open[a] >= 0:
                    node_ops[open[a]].append(i)
                    continue
                var u = len(self.kind)
                self.kind.append(NK_SINGLE)
                self.wa.append(a)
                self.wb.append(-1)
                node_ops.append([i])
                var pl = List[Int]()
                _add_unique(pl, last[a])
                preds.append(pl^)
                preds2.append(List[Int]())
                last[a] = u
                continue
            # fence: non-plain op (or a plain op of another arity)
            var ws = List[Int]()
            for s in range(ar):
                var w = g.qubit(s)
                if w >= 0 and w < n:
                    _add_unique(ws, w)
            for w in ws:
                self._close(open, w)
            var u = len(self.kind)
            self.kind.append(NK_FENCE)
            if len(ws) >= 2:
                self.wa.append(min(ws[0], ws[1]))
                self.wb.append(max(ws[0], ws[1]))
            elif len(ws) == 1:
                self.wa.append(ws[0])
                self.wb.append(-1)
            else:
                self.wa.append(-1)
                self.wb.append(-1)
            node_ops.append([i])
            var pl = List[Int]()
            for w in ws:
                _add_unique(pl, last[w])
                last[w] = u
            _add_unique(pl, last_cl)
            last_cl = u
            preds.append(pl^)
            var p2 = List[Int]()
            if len(ws) >= 2:
                for w in ws:
                    _add_unique(p2, last2[w])
                    last2[w] = u
            preds2.append(p2^)
        var nn = len(self.kind)
        for u in range(nn):
            var cst = 0
            for k in node_ops[u]:
                var g = ops[k]
                if op_is_unitary(g.kind) and op_arity(g.kind) == 2:
                    cst += _gate_cx(g.kind)
            self.ucost.append(cst)
        # CSR ops
        self.op_start = List[Int](length=nn + 1, fill=0)
        for u in range(nn):
            self.op_start[u + 1] = self.op_start[u] + len(node_ops[u])
        for u in range(nn):
            for k in node_ops[u]:
                self.op_idx.append(k)
        # successors (sorted unique; preds are earlier nodes)
        var nsucc = List[Int](length=nn, fill=0)
        self.indeg = List[Int](length=nn, fill=0)
        self.npred2 = List[Int](length=nn, fill=0)
        var nsucc2 = List[Int](length=nn, fill=0)
        for u in range(nn):
            self.indeg[u] = len(preds[u])
            for p in preds[u]:
                nsucc[p] += 1
            self.npred2[u] = len(preds2[u])
            for p in preds2[u]:
                nsucc2[p] += 1
            if self.wb[u] >= 0:
                self.n2q += 1
        self.succ_start = List[Int](length=nn + 1, fill=0)
        self.succ2_start = List[Int](length=nn + 1, fill=0)
        for u in range(nn):
            self.succ_start[u + 1] = self.succ_start[u] + nsucc[u]
            self.succ2_start[u + 1] = self.succ2_start[u] + nsucc2[u]
        self.succ = List[Int](length=self.succ_start[nn], fill=0)
        self.succ2 = List[Int](length=self.succ2_start[nn], fill=0)
        var fill1 = List[Int](length=nn, fill=0)
        var fill2 = List[Int](length=nn, fill=0)
        for u in range(nn):  # u ascending -> successor lists come out sorted
            for p in preds[u]:
                self.succ[self.succ_start[p] + fill1[p]] = u
                fill1[p] += 1
            for p in preds2[u]:
                self.succ2[self.succ2_start[p] + fill2[p]] = u
                fill2[p] += 1

    def _close(self, mut open: List[Int], w: Int):
        var u = open[w]
        if u < 0:
            return
        open[self.wa[u]] = -1
        open[self.wb[u]] = -1

    def num_nodes(self) -> Int:
        return len(self.kind)

    def is2(self, u: Int) -> Bool:
        return self.wb[u] >= 0

    def is_single_cx(self, u: Int) -> Bool:
        if self.kind[u] != NK_UNIT or self.op_start[u + 1] - self.op_start[u] != 1:
            return False
        return self.ops[self.op_idx[self.op_start[u]]].kind == Op.CX


def _add_unique(mut l: List[Int], v: Int):
    if v < 0:
        return
    for x in l:
        if x == v:
            return
    l.append(v)


def reversed_ops(ops: List[Instruction]) -> List[Instruction]:
    var out = List[Instruction](capacity=len(ops))
    for i in range(len(ops) - 1, -1, -1):
        out.append(ops[i])
    return out^


# ----------------------------------------------------------------------------
# Router model (qumo.rlroute.v0): acts = b0 + sum_i sign(obs_i) E[i,:];
# logits = W1 relu(acts) + b1
# ----------------------------------------------------------------------------
struct RouteModel(Copyable):
    var ok: Bool
    var id: String
    var dir: String
    var hidden: Int
    var emb: List[Float32]  # [OBS_SIZE, H] row-major
    var b0: List[Float32]  # [H]
    var w1: List[Float32]  # [16, H]
    var b1: List[Float32]  # [16]
    var manifest: String

    def __init__(out self):
        self.ok = False
        self.id = String()
        self.dir = String()
        self.hidden = 0
        self.emb = List[Float32]()
        self.b0 = List[Float32]()
        self.w1 = List[Float32]()
        self.b1 = List[Float32]()
        self.manifest = String()

    @staticmethod
    def random(hidden: Int, seed: Int) -> RouteModel:
        """Random-weight model (tests / mechanics without a trained model)."""
        var m = RouteModel()
        var rng = Rng(UInt64(seed))
        m.ok = True
        m.id = "random_h" + String(hidden)
        m.hidden = hidden
        for _ in range(OBS_SIZE * hidden):
            m.emb.append(Float32(rng.uniform() - 0.5))
        for _ in range(hidden):
            m.b0.append(Float32(rng.uniform() - 0.5))
        for _ in range(NUM_ACTIVE_SWAPS * hidden):
            m.w1.append(Float32(rng.uniform() - 0.5))
        for _ in range(NUM_ACTIVE_SWAPS):
            m.b1.append(Float32(rng.uniform() - 0.5))
        return m^

    def forward(self, obs: List[Int], mut hid: List[Float64], mut logits: List[Float64]):
        """Logits (Float64 accumulation, fixed summation order i = 0..127)."""
        var H = self.hidden
        for h in range(H):
            hid[h] = Float64(self.b0[h])
        var pe = self.emb.unsafe_ptr()
        for i in range(OBS_SIZE):
            var o = obs[i]
            if o == 0:
                continue
            var base = i * H
            if o > 0:
                for h in range(H):
                    hid[h] += Float64(pe[unsafe_offset=base + h])
            else:
                for h in range(H):
                    hid[h] -= Float64(pe[unsafe_offset=base + h])
        var pw = self.w1.unsafe_ptr()
        for k in range(NUM_ACTIVE_SWAPS):
            var acc = Float64(self.b1[k])
            var base = k * H
            for h in range(H):
                var v = hid[h]
                if v > 0.0:
                    acc += Float64(pw[unsafe_offset=base + h]) * v
            logits[k] = acc
        _ = len(self.emb)
        _ = len(self.w1)


def load_route_model(dir: String) raises -> RouteModel:
    """Load a `qumo.rlroute.v0` model directory (manifest.json + weights.bin)
    through `policy.load_model_file` (QM-E617 on a missing / corrupt file or
    checksum mismatch); QM-E620 when it is not a router model of this
    architecture (sign embedding 128 -> H, relu, -> 16 logits).

    Tensors (training/ai_transpiler/common/export.py naming):
    `embeddings.weight` [128, H] (stored transposed, row i = embedding of
    observation feature i), `embeddings.bias` [H], `action.0.weight` [16, H],
    `action.0.bias` [16]; optional `value.0.*` (PPO critic, unused here).
    """
    var mf = load_model_file(dir)
    if mf.format != MODEL_FORMAT_ROUTE or mf.kind_code != MODEL_KIND_ROUTING:
        raise Error("QM-E620: model '" + mf.id + "' at '" + dir + "' is a " + mf.kind + " model (" + mf.format + "), not a router model. Hint: pass policy=<id> of a model listed in models/ai_transpiler/route/INDEX.json.")
    var man = parse_json(mf.manifest)
    var arch = man.get("arch")
    var hz = arch.get_int("horizon", -1)
    var na = arch.get_int("num_active_swaps", -1)
    var H = arch.get_int("hidden", -1)
    if mf.obs_size != OBS_SIZE or hz != HORIZON or na != NUM_ACTIVE_SWAPS or H <= 0 or H > 4096:
        raise Error("QM-E620: router model '" + mf.id + "' has obs=" + String(mf.obs_size) + ", horizon=" + String(hz) + ", actions=" + String(na) + ", hidden=" + String(H) + "; this router needs obs=128, horizon=8, actions=16. Hint: retrain / re-export it as qumo.rlroute.v0 (training/ai_transpiler/route).")
    var m = RouteModel()
    m.emb = mf.tensor("embeddings.weight")
    m.b0 = mf.tensor("embeddings.bias")
    m.w1 = mf.tensor("action.0.weight")
    m.b1 = mf.tensor("action.0.bias")
    if len(m.emb) != OBS_SIZE * H or len(m.b0) != H or len(m.w1) != NUM_ACTIVE_SWAPS * H or len(m.b1) != NUM_ACTIVE_SWAPS:
        raise Error("QM-E620: router model '" + mf.id + "' tensor shapes do not match hidden=" + String(H) + ". Hint: re-export the model with training/ai_transpiler/route/export_route.py.")
    m.hidden = H
    m.id = mf.id
    m.dir = dir
    m.manifest = mf.manifest
    m.ok = True
    return m^


def find_route_model(request: String, dirs: List[String]) raises -> String:
    """Directory of the requested router model ('' when 'auto' finds none).

    Rows of `<root>/route/INDEX.json` (kind "routing") in root order; 'auto'
    takes the first row marked "default" (else the first row); an id must
    match a row (or name a model directory directly)."""
    for d in dirs:
        var rows = read_model_index(d, "route")
        var first = String()
        for r in rows:
            if r.kind_code != MODEL_KIND_ROUTING:
                continue
            if request == "auto":
                if first.byte_length() == 0:
                    first = r.path
                continue
            if r.id == request:
                return r.path
        if request == "auto":
            var dflt = _index_default(d)
            if dflt.byte_length() > 0:
                return dflt
            if first.byte_length() > 0:
                return first
    if request == "auto":
        return String()
    if exists(request + "/manifest.json"):
        return request
    var s = String("[")
    for i in range(len(dirs)):
        if i > 0:
            s += ", "
        s += dirs[i]
    raise Error("QM-E617: router model '" + request + "' not found in model roots " + s + "]. Hint: list models/ai_transpiler/route/INDEX.json, pass model_dir=<root> or set QUMO_MODEL_DIR (policy=heuristic needs no model).")


def _index_default(root: String) -> String:
    """Path of the INDEX.json row marked "default": true under <root>/route."""
    var ipath = root + "/route/INDEX.json"
    if not exists(ipath):
        return String()
    try:
        var text: String
        with open(ipath, "r") as f:
            text = f.read()
        var j = parse_json(text)
        for i in range(len(j)):
            var r = j.at(i)
            if r.get_string("kind", "") == "routing" and r.get_bool("default", False):
                return root + "/route/" + r.get("dir").as_string()
    except:
        pass
    return String()


# ----------------------------------------------------------------------------
# Episode: one stochastic routing run
# ----------------------------------------------------------------------------
struct Episode(Copyable):
    var N: Int
    var l2p: List[Int]
    var p2l: List[Int]
    var init_p2l: List[Int]
    var indeg: List[Int]
    var rem2: List[Int]
    var done: List[Bool]
    var front: List[Int]  # ready, not executed nodes
    var front2: List[Int]  # 2-wire nodes whose 2q predecessors are executed
    var executed: Int
    var exec2q: Int
    var steps: Int
    var noprog: Int
    var best_fsum: Int
    var fallback: Bool
    var n_fallback: Int
    var virtual_ok: Bool
    var failed: Bool
    var rng: Rng
    # output items
    var it_kind: List[Int]
    var it_node: List[Int]
    var it_w0: List[Int]
    var it_w1: List[Int]
    var it_prev0: List[Int]
    var it_prev1: List[Int]
    var it_order: List[Int]
    var it_dead: List[Bool]
    var it_credit: List[Int]  # absorption credit held by a SWAP item (CX units)
    var it_partner: List[Int]  # unit item that absorbed this SWAP (or SWAP absorbed by this unit), -1 none
    var it_ops: List[Instruction]  # flat
    var it_op_start: List[Int]
    var it_op_len: List[Int]
    var last: List[Int]  # last live item per physical qubit (-1 none)
    var last_barrier: Int
    var swaps_live: Int
    var n_fused: Int
    var n_virtual: Int
    var n_cancel: Int
    var absorb_credit: Int  # CX units saved by SWAPs next to a 2q unit on the same pair (KAK merges them)
    var n_absorbed: Int
    var swap_count: Int  # swaps chosen (before simplification)
    # scratch
    var layer_nodes: List[Int]
    var layer_start: List[Int]
    var active: List[Int]  # edge indices
    var obs: List[Int]
    var tmp: List[Int]
    var stamp: List[Int]
    var stamp_id: Int
    var hid: List[Float64]
    var logits: List[Float64]
    var actions: List[Int]  # chosen edge per step (trace)
    var adj_e: List[Int]  # edge index per adjacency entry
    var touch: List[UInt32]  # scratch: active-swap bitmask per physical qubit
    var sg: List[Int]  # scratch: sign(obs)
    var exe_buf: List[Int]
    var keep_buf: List[Int]
    var decay: List[Float64]
    var since_reset: Int
    var f_now: Int  # front-layer distance sum
    var e_now: Int  # lookahead (layers 1..7) distance sum
    var sw_p: List[Int]  # endpoints of active swap k
    var sw_x: List[Int]
    var rec_ops: Bool  # record output instructions (False: costs / decisions only, for training rollouts)

    def __init__(out self, prob: RProblem, target: Target, l2p0: List[Int], seed: UInt64, virtual_ok: Bool):
        var N = target.num_qubits
        self.N = N
        self.l2p = l2p0.copy()
        self.p2l = List[Int](length=N, fill=-1)
        for w in range(N):
            self.p2l[self.l2p[w]] = w
        self.init_p2l = self.p2l.copy()
        self.indeg = prob.indeg.copy()
        self.rem2 = prob.npred2.copy()
        var nn = prob.num_nodes()
        self.done = List[Bool](length=nn, fill=False)
        self.front = List[Int]()
        self.front2 = List[Int]()
        for u in range(nn):
            if self.indeg[u] == 0:
                self.front.append(u)
            if prob.is2(u) and self.rem2[u] == 0:
                self.front2.append(u)
        self.executed = 0
        self.exec2q = 0
        self.steps = 0
        self.noprog = 0
        self.best_fsum = 1 << 60
        self.fallback = False
        self.n_fallback = 0
        self.virtual_ok = virtual_ok
        self.failed = False
        self.rng = Rng(seed)
        self.it_kind = List[Int]()
        self.it_node = List[Int]()
        self.it_w0 = List[Int]()
        self.it_w1 = List[Int]()
        self.it_prev0 = List[Int]()
        self.it_prev1 = List[Int]()
        self.it_order = List[Int]()
        self.it_dead = List[Bool]()
        self.it_credit = List[Int]()
        self.it_partner = List[Int]()
        self.it_ops = List[Instruction]()
        self.it_op_start = List[Int]()
        self.it_op_len = List[Int]()
        self.last = List[Int](length=N, fill=-1)
        self.last_barrier = -1
        self.swaps_live = 0
        self.n_fused = 0
        self.n_virtual = 0
        self.n_cancel = 0
        self.absorb_credit = 0
        self.n_absorbed = 0
        self.swap_count = 0
        self.layer_nodes = List[Int]()
        self.layer_start = List[Int]()
        self.active = List[Int]()
        self.obs = List[Int](length=OBS_SIZE, fill=0)
        self.tmp = List[Int](length=nn, fill=0)
        self.stamp = List[Int](length=nn, fill=-1)
        self.stamp_id = 0
        self.hid = List[Float64]()
        self.logits = List[Float64](length=NUM_ACTIVE_SWAPS, fill=0.0)
        self.actions = List[Int]()
        self.adj_e = adjacency_edges(target)
        self.rec_ops = True
        self.decay = List[Float64](length=N, fill=1.0)
        self.since_reset = 0
        self.f_now = 0
        self.e_now = 0
        self.sw_p = List[Int](length=NUM_ACTIVE_SWAPS, fill=0)
        self.sw_x = List[Int](length=NUM_ACTIVE_SWAPS, fill=0)
        self.touch = List[UInt32](length=N, fill=0)
        self.sg = List[Int](length=OBS_SIZE, fill=0)
        self.exe_buf = List[Int]()
        self.keep_buf = List[Int]()

    def overhead(self) -> Int:
        """Routing overhead in CX units so far: 3 per live SWAP, 1 per SWAP fused
        with a CX, minus the credit of SWAPs merged into an adjacent 2q unit on
        the same pair (a run costs at most 3 CX after KAK: credit = min(3, unit cost))."""
        return 3 * self.swaps_live + self.n_fused - self.absorb_credit

    # ---- output items ------------------------------------------------------
    def last_before(self, it: Int, w: Int) -> Int:
        """Previous item of `it` on wire w."""
        return self._prev_on(it, w)

    def _prev_on(self, i: Int, p: Int) -> Int:
        return self.it_prev0[i] if self.it_w0[i] == p else self.it_prev1[i]

    def _new_item(mut self, kind: Int, node: Int, w0: Int, w1: Int) -> Int:
        var i = len(self.it_kind)
        self.it_kind.append(kind)
        self.it_node.append(node)
        self.it_w0.append(w0)
        self.it_w1.append(w1)
        self.it_prev0.append(self.last[w0] if w0 >= 0 else -1)
        self.it_prev1.append(self.last[w1] if w1 >= 0 else -1)
        self.it_order.append(0)
        self.it_dead.append(False)
        self.it_credit.append(0)
        self.it_partner.append(-1)
        self.it_op_start.append(len(self.it_ops))
        self.it_op_len.append(0)
        if w0 >= 0:
            self.last[w0] = i
        if w1 >= 0:
            self.last[w1] = i
        return i

    def _passable(self, prob: RProblem, i: Int) -> Bool:
        """Item a SWAP may pass through by relabelling (plain 2q unitary)."""
        var k = self.it_kind[i]
        if k == IT_FUSED:
            return True
        return k == IT_NODE and prob.kind[self.it_node[i]] == NK_UNIT

    def _relabel_item(mut self, i: Int, p: Int, x: Int):
        var s = self.it_op_start[i]
        for k in range(s, s + self.it_op_len[i]):
            var g = self.it_ops[k]
            if g.q0 == p:
                g.q0 = x
            elif g.q0 == x:
                g.q0 = p
            if op_arity(g.kind) >= 2:
                if g.q1 == p:
                    g.q1 = x
                elif g.q1 == x:
                    g.q1 = p
            self.it_ops[k] = g

    def _chain_ok(self, start: Int, w: Int, lo: Int, hi: Int, mut items: List[Int]) -> Bool:
        """Walk wire w's chain back to the circuit start; True when every item
        touches only lo / hi (1-wire items or items on exactly {lo, hi})."""
        var i = start
        while i >= 0:
            var a = self.it_w0[i]
            var b = self.it_w1[i]
            if a < 0:
                return False  # barrier
            if b >= 0 and not (a == lo and b == hi):
                return False
            items.append(i)
            i = self._prev_on(i, w)
        return True

    def _try_virtual(mut self, lo: Int, hi: Int) -> Bool:
        """A SWAP whose two wires carry only ops on {lo, hi} since the start
        becomes a layout change: relabel that history and swap the chains."""
        var items = List[Int]()
        if not self._chain_ok(self.last[lo], lo, lo, hi, items):
            return False
        if not self._chain_ok(self.last[hi], hi, lo, hi, items):
            return False
        sort(items)
        var prev = -1
        for i in items:
            if i == prev:
                continue
            prev = i
            self._relabel_item(i, lo, hi)
            if self.it_w1[i] >= 0:
                var t = self.it_prev0[i]
                self.it_prev0[i] = self.it_prev1[i]
                self.it_prev1[i] = t
            else:
                self.it_w0[i] = hi if self.it_w0[i] == lo else lo
        var t = self.last[lo]
        self.last[lo] = self.last[hi]
        self.last[hi] = t
        var a = self.init_p2l[lo]
        self.init_p2l[lo] = self.init_p2l[hi]
        self.init_p2l[hi] = a
        self.n_virtual += 1
        return True

    def push_swap(mut self, prob: RProblem, p: Int, x: Int):
        """Append routing SWAP(p, x) with simplification, in this order:
        (1) SWAP.SWAP cancellation, also through plain 2q units on the pair
        (the SWAP passes through them by relabelling); (2) virtual swap: when
        both wires carry only ops on {p, x} since the start, the history is
        relabelled and the initial layout changes instead; (3) fusion with a
        single CX just executed on the pair (CX.SWAP = CX(t,c) CX(c,t));
        (4) a plain SWAP item."""
        var lo = min(p, x)
        var hi = max(p, x)
        var walk = List[Int]()
        var i = self.last[lo]
        var j = self.last[hi]
        var outcome = 0  # 2 cancel
        while True:
            if i != j or i < 0:
                break
            if self.it_kind[i] == IT_SWAP:
                outcome = 2
                break
            if not self._passable(prob, i):
                break
            walk.append(i)
            var pi = self._prev_on(i, lo)
            var pj = self._prev_on(i, hi)
            i = pi
            j = pj
        if outcome == 2:
            for t in walk:
                self._relabel_item(t, lo, hi)
            var s = i
            self.it_dead[s] = True
            if self.it_credit[s] > 0:
                self.absorb_credit -= self.it_credit[s]
                self.n_absorbed -= 1
                self.it_partner[self.it_partner[s]] = -1
                self.it_credit[s] = 0
            if len(walk) == 0:
                self.last[lo] = self._prev_on(s, lo)
                self.last[hi] = self._prev_on(s, hi)
            else:
                var e = walk[len(walk) - 1]
                var pl = self._prev_on(s, lo)
                var ph = self._prev_on(s, hi)
                # e's wires are {lo, hi}: w0 = lo, w1 = hi
                self.it_prev0[e] = pl
                self.it_prev1[e] = ph
            self.swaps_live -= 1
            self.n_cancel += 1
            return
        if self.virtual_ok and self._try_virtual(lo, hi):
            return
        var k = self.last[lo]
        if k >= 0 and k == self.last[hi] and self.it_kind[k] == IT_NODE and prob.is_single_cx(self.it_node[k]):
            # CX(c,t) then SWAP == CX(t,c) CX(c,t)
            self.it_kind[k] = IT_FUSED
            self.it_order[k] = 1
            if self.rec_ops:
                var m = self.it_ops[self.it_op_start[k]]
                var r = m
                r.q0 = m.q1
                r.q1 = m.q0
                self.it_op_start[k] = len(self.it_ops)
                self.it_op_len[k] = 2
                self.it_ops.append(r)
                self.it_ops.append(m)
            self.n_fused += 1
            return
        var it = self._new_item(IT_SWAP, -1, lo, hi)
        if k >= 0 and k == self.last_before(it, hi) and self.it_kind[k] == IT_NODE and prob.kind[self.it_node[k]] == NK_UNIT and self.it_partner[k] < 0:
            var cr = min(3, prob.ucost[self.it_node[k]])
            self.it_credit[it] = cr
            self.it_partner[it] = k
            self.it_partner[k] = it
            self.absorb_credit += cr
            self.n_absorbed += 1
        if self.rec_ops:
            self.it_ops.append(Instruction.make(Op.SWAP, lo, hi))
            self.it_op_len[it] = 1
        self.swaps_live += 1

    def push_node(mut self, prob: RProblem, u: Int):
        var kind = prob.kind[u]
        if kind == NK_BARRIER:
            var it = self._new_item(IT_NODE, u, -1, -1)
            for p in range(self.N):
                self.last[p] = it
            self.last_barrier = it
            self._append_ops(prob, u, it)
            return
        var w0 = -1
        var w1 = -1
        if prob.wa[u] >= 0:
            w0 = self.l2p[prob.wa[u]]
        if prob.wb[u] >= 0:
            w1 = self.l2p[prob.wb[u]]
            if w1 < w0:
                var t = w0
                w0 = w1
                w1 = t
        if w1 >= 0 and prob.is_single_cx(u):
            var k = self.last[w0]
            if k >= 0 and k == self.last[w1] and self.it_kind[k] == IT_SWAP:
                # SWAP then CX(c,t) == CX(c,t) CX(t,c)
                self.it_kind[k] = IT_FUSED
                self.it_node[k] = u
                self.it_order[k] = 0
                if self.rec_ops:
                    var m = prob.ops[prob.op_idx[prob.op_start[u]]]
                    m.q0 = self.l2p[m.q0]
                    m.q1 = self.l2p[m.q1]
                    var r = m
                    r.q0 = m.q1
                    r.q1 = m.q0
                    self.it_op_start[k] = len(self.it_ops)
                    self.it_op_len[k] = 2
                    self.it_ops.append(m)
                    self.it_ops.append(r)
                self.swaps_live -= 1
                self.n_fused += 1
                return
        var it = self._new_item(IT_NODE, u, w0, w1)
        if w1 >= 0 and kind == NK_UNIT:
            var k = self.it_prev0[it]
            if k >= 0 and k == self.it_prev1[it] and self.it_kind[k] == IT_SWAP and self.it_partner[k] < 0:
                var cr = min(3, prob.ucost[u])
                self.it_credit[k] = cr
                self.it_partner[k] = it
                self.it_partner[it] = k
                self.absorb_credit += cr
                self.n_absorbed += 1
        self._append_ops(prob, u, it)

    def _append_ops(mut self, prob: RProblem, u: Int, it: Int):
        if not self.rec_ops:
            return
        var cnt = 0
        for e in range(prob.op_start[u], prob.op_start[u + 1]):
            var g = prob.ops[prob.op_idx[e]]
            var ar = op_arity(g.kind)
            if ar >= 1 and g.q0 >= 0:
                g.q0 = self.l2p[g.q0]
            if ar >= 2 and g.q1 >= 0:
                g.q1 = self.l2p[g.q1]
            if ar >= 3 and g.q2 >= 0:
                g.q2 = self.l2p[g.q2]
            self.it_ops.append(g)
            cnt += 1
        self.it_op_len[it] = cnt

    # ---- environment dynamics ---------------------------------------------
    def dist_nodes(self, prob: RProblem, target: Target, u: Int) -> Int:
        return target.dist[self.l2p[prob.wa[u]] * self.N + self.l2p[prob.wb[u]]]

    def execute(mut self, prob: RProblem, target: Target) -> Int:
        """Execute every ready node whose wires are adjacent (1q / fences
        always); passes in ascending node order until nothing executes."""
        var count = 0
        while True:
            if len(self.front) > 1:
                sort(self.front)
            self.exe_buf.clear()
            self.keep_buf.clear()
            for u in self.front:
                if not prob.is2(u) or self.dist_nodes(prob, target, u) == 1:
                    self.exe_buf.append(u)
                else:
                    self.keep_buf.append(u)
            if len(self.exe_buf) == 0:
                break
            var tmpl = self.front^
            self.front = self.keep_buf^
            self.keep_buf = tmpl^
            var exe = self.exe_buf.copy()
            for u in exe:
                self.push_node(prob, u)
                self.done[u] = True
                self.executed += 1
                count += 1
                for e in range(prob.succ_start[u], prob.succ_start[u + 1]):
                    var s = prob.succ[e]
                    self.indeg[s] -= 1
                    if self.indeg[s] == 0:
                        self.front.append(s)
                if prob.is2(u):
                    self.exec2q += 1
                    for e in range(prob.succ2_start[u], prob.succ2_start[u + 1]):
                        var s = prob.succ2[e]
                        self.rem2[s] -= 1
                        if self.rem2[s] == 0:
                            self.front2.append(s)
            if self.since_reset > 0:
                self._reset_decay()
            if len(self.front2) > 0:
                var k2 = List[Int]()
                for u in self.front2:
                    if not self.done[u]:
                        k2.append(u)
                self.front2 = k2^
        return count

    def fsum(self, prob: RProblem, target: Target) -> Int:
        var s = 0
        for u in self.front2:
            s += self.dist_nodes(prob, target, u) - 1
        return s


    def build_layers(mut self, prob: RProblem):
        """Up to HORIZON topological layers of the remaining 2q DAG (Kahn),
        each sorted by node id then shuffled with the episode Rng."""
        self.layer_nodes.clear()
        self.layer_start.clear()
        self.stamp_id += 1
        var cur = self.front2.copy()
        sort(cur)
        var d = 0
        while d < HORIZON and len(cur) > 0:
            self.layer_start.append(len(self.layer_nodes))
            for u in cur:
                self.layer_nodes.append(u)
            var nxt = List[Int]()
            for u in cur:
                for e in range(prob.succ2_start[u], prob.succ2_start[u + 1]):
                    var s = prob.succ2[e]
                    if self.stamp[s] != self.stamp_id:
                        self.stamp[s] = self.stamp_id
                        self.tmp[s] = self.rem2[s]
                    self.tmp[s] -= 1
                    if self.tmp[s] == 0:
                        nxt.append(s)
            sort(nxt)
            cur = nxt^
            d += 1
        self.layer_start.append(len(self.layer_nodes))
        for l in range(len(self.layer_start) - 1):
            shuffle_range(self.rng, self.layer_nodes, self.layer_start[l], self.layer_start[l + 1])

    def build_active(mut self, prob: RProblem, target: Target):
        """Coupling edges touching the qubits of the layer gates (in layer
        order), at most NUM_ACTIVE_SWAPS, then shuffled."""
        self.active.clear()
        for t in range(len(self.layer_nodes)):
            if len(self.active) >= NUM_ACTIVE_SWAPS:
                break
            var u = self.layer_nodes[t]
            for side in range(2):
                if len(self.active) >= NUM_ACTIVE_SWAPS:
                    break
                var q = self.l2p[prob.wa[u]] if side == 0 else self.l2p[prob.wb[u]]
                for k in range(target.adj_start[q], target.adj_start[q + 1]):
                    var e = self.adj_e[k]
                    var seen = False
                    for a in self.active:
                        if a == e:
                            seen = True
                            break
                    if not seen:
                        self.active.append(e)
                        if len(self.active) >= NUM_ACTIVE_SWAPS:
                            break
        shuffle_range(self.rng, self.active, 0, len(self.active))

    def build_obs(mut self, prob: RProblem, target: Target):
        """obs[d + HORIZON * k] = sum over gates of layer d of the distance
        change if active swap k were applied."""
        for i in range(OBS_SIZE):
            self.obs[i] = 0
        var N = self.N
        var A = len(self.active)
        # bitmask of the active swaps touching each physical qubit
        for k in range(A):
            var e = self.active[k]
            self.touch[target.edges[2 * e]] |= UInt32(1) << UInt32(k)
            self.touch[target.edges[2 * e + 1]] |= UInt32(1) << UInt32(k)
        for k in range(A):
            var e = self.active[k]
            self.sw_p[k] = target.edges[2 * e]
            self.sw_x[k] = target.edges[2 * e + 1]
        self.f_now = 0
        self.e_now = 0
        var pd = target.dist.unsafe_ptr()
        for d in range(len(self.layer_start) - 1):
            for t in range(self.layer_start[d], self.layer_start[d + 1]):
                var u = self.layer_nodes[t]
                var pa = self.l2p[prob.wa[u]]
                var pb = self.l2p[prob.wb[u]]
                if d == 0:
                    self.f_now += pd[unsafe_offset=pa * N + pb]
                else:
                    self.e_now += pd[unsafe_offset=pa * N + pb]
                var m = self.touch[pa] | self.touch[pb]
                if m == 0:
                    continue
                var base = pd[unsafe_offset=pa * N + pb]
                while m != 0:
                    var k = Int(count_trailing_zeros(m))
                    m &= m - 1
                    var e = self.active[k]
                    var p = target.edges[2 * e]
                    var x = target.edges[2 * e + 1]
                    var qa = x if pa == p else (p if pa == x else pa)
                    var qb = x if pb == p else (p if pb == x else pb)
                    self.obs[d + HORIZON * k] += pd[unsafe_offset=qa * N + qb] - base
        _ = len(target.dist)
        for k in range(A):
            var e = self.active[k]
            self.touch[target.edges[2 * e]] = 0
            self.touch[target.edges[2 * e + 1]] = 0

    def heuristic_score(self, k: Int, layers: Int) -> Float64:
        """SABRE-style score of active swap k (lower is better):
        max(decay_p, decay_x) * (front distance sum after the swap + 0.5 *
        mean lookahead distance after the swap), from the observation plus the
        current sums (decay +0.001 per swap on a qubit, reset after a gate
        executes or every 5 swaps, as in SABRE)."""
        var f = Float64(self.f_now + self.obs[HORIZON * k])
        var la = Float64(self.e_now)
        var cnt = 0
        for d in range(1, HORIZON):
            la += Float64(self.obs[d + HORIZON * k])
        for d in range(1, layers):
            cnt += self.layer_start[d + 1] - self.layer_start[d]
        if cnt > 0:
            la = la / Float64(cnt)
        return max(self.decay[self.sw_p[k]], self.decay[self.sw_x[k]]) * (f + 0.5 * la)

    def choose(mut self, prob: RProblem, target: Target, model: RouteModel, mode: Int, beta: Float64) -> Int:
        """Index into `active` of the chosen swap."""
        var A = len(self.active)
        var layers = len(self.layer_start) - 1
        if mode == MODE_HEURISTIC or not model.ok:
            var best = 0
            var bs = 0.0
            for k in range(A):
                var s = self.heuristic_score(k, layers)
                if k == 0 or s < bs:
                    bs = s
                    best = k
            return best
        for i in range(OBS_SIZE):
            var o = self.obs[i]
            self.sg[i] = 1 if o > 0 else (-1 if o < 0 else 0)
        if len(self.hid) != model.hidden:
            self.hid = List[Float64](length=model.hidden, fill=0.0)
        model.forward(self.sg, self.hid, self.logits)
        if mode == MODE_POLICY:
            var best = 0
            for k in range(1, A):
                if self.logits[k] > self.logits[best]:
                    best = k
            return best
        # scored: SABRE score minus beta * log-softmax of the policy
        var mx = self.logits[0]
        for k in range(1, A):
            mx = max(mx, self.logits[k])
        var z = 0.0
        for k in range(A):
            z += exp(self.logits[k] - mx)
        var lz = log(z) + mx
        var best = 0
        var bs = 0.0
        for k in range(A):
            var s = self.heuristic_score(k, layers) - beta * (self.logits[k] - lz)
            if k == 0 or s < bs:
                bs = s
                best = k
        return best

    def fallback_swap(self, prob: RProblem, target: Target) -> Int:
        """Edge index of the shortest-path swap for the closest front gate
        (-1 when no front gate can get closer: disconnected coupling map)."""
        var bu = -1
        var bd = 0
        var f2 = self.front2.copy()
        sort(f2)
        for u in f2:
            var d = self.dist_nodes(prob, target, u)
            if d > 1 and (bu < 0 or d < bd):
                bu = u
                bd = d
        if bu < 0:
            return -1
        var pa = self.l2p[prob.wa[bu]]
        var pb = self.l2p[prob.wb[bu]]
        if bd > self.N:
            return -1
        for k in range(target.adj_start[pa], target.adj_start[pa + 1]):
            var y = target.adj[k]
            if target.dist[y * self.N + pb] == bd - 1:
                return self.adj_e[k]
        return -1

    def apply_swap(mut self, prob: RProblem, target: Target, e: Int):
        var p = target.edges[2 * e]
        var x = target.edges[2 * e + 1]
        var lp = self.p2l[p]
        var lx = self.p2l[x]
        self.p2l[p] = lx
        self.p2l[x] = lp
        self.l2p[lx] = p
        self.l2p[lp] = x
        self.swap_count += 1
        self.since_reset += 1
        if self.since_reset >= 5:
            self._reset_decay()
        else:
            self.decay[p] += 0.001
            self.decay[x] += 0.001
        self.push_swap(prob, p, x)

    def _reset_decay(mut self):
        for i in range(self.N):
            self.decay[i] = 1.0
        self.since_reset = 0

    def finished(self, prob: RProblem) -> Bool:
        return self.executed == prob.num_nodes()

    def step(mut self, prob: RProblem, target: Target, model: RouteModel, mode: Int, beta: Float64, limit: Int) -> Bool:
        """Run one decision (after executing ready gates). False when done."""
        var ex = self.execute(prob, target)
        if self.finished(prob):
            return False
        var f = self.fsum(prob, target)
        if ex > 0:
            self.noprog = 0
            self.best_fsum = f
            self.fallback = False
        else:
            if f < self.best_fsum:
                self.best_fsum = f
                self.noprog = 0
            else:
                self.noprog += 1
            if self.noprog >= limit:
                self.fallback = True
        self.steps += 1
        self.build_layers(prob)
        self.build_active(prob, target)
        var e: Int
        if self.fallback:
            e = self.fallback_swap(prob, target)
            self.n_fallback += 1
            if e < 0:
                self.failed = True
                return False
        elif len(self.active) == 0:
            e = self.fallback_swap(prob, target)
            self.n_fallback += 1
            if e < 0:
                self.failed = True
                return False
        else:
            self.build_obs(prob, target)
            var k = self.choose(prob, target, model, mode, beta)
            e = self.active[k]
        self.actions.append(e)
        self.apply_swap(prob, target, e)
        return True

    def run(mut self, prob: RProblem, target: Target, model: RouteModel, mode: Int, beta: Float64, max_steps: Int):
        var limit = max(4, 2 * prob.n)
        while self.step(prob, target, model, mode, beta, limit):
            if self.steps > max_steps:
                self.failed = True
                return

    # ---- results ----------------------------------------------------------
    def live_items(self) -> List[Int]:
        var out = List[Int]()
        for i in range(len(self.it_kind)):
            if not self.it_dead[i]:
                out.append(i)
        return out^

    def emit_ops(self) -> List[Instruction]:
        var out = List[Instruction](capacity=len(self.it_ops))
        for i in range(len(self.it_kind)):
            if self.it_dead[i]:
                continue
            var s = self.it_op_start[i]
            for k in range(s, s + self.it_op_len[i]):
                out.append(self.it_ops[k])
        return out^

    def init_l2p(self) -> List[Int]:
        var l = List[Int](length=self.N, fill=0)
        for p in range(self.N):
            l[self.init_p2l[p]] = p
        return l^


def shuffle_range(mut rng: Rng, mut l: List[Int], lo: Int, hi: Int):
    """Fisher-Yates on l[lo:hi] (i from hi-1 down to lo+1, j = lo + below(i-lo+1))."""
    var i = hi - 1
    while i > lo:
        var j = lo + rng.below(i - lo + 1)
        var t = l[i]
        l[i] = l[j]
        l[j] = t
        i -= 1


def adjacency_edges(target: Target) -> List[Int]:
    """Edge index of every CSR adjacency entry (parallel to target.adj)."""
    var out = List[Int](length=len(target.adj), fill=-1)
    var N = target.num_qubits
    for q in range(N):
        for k in range(target.adj_start[q], target.adj_start[q + 1]):
            var y = target.adj[k]
            try:
                out[k] = target.edge_index[min(q, y) * N + max(q, y)]
            except:
                pass
    return out^


def episode_max_steps(prob: RProblem, target: Target) -> Int:
    return 1000 + 64 * (prob.n2q + 1) * max(2, target.diameter())


# ----------------------------------------------------------------------------
# Structural verification (exactness of a routed result)
# ----------------------------------------------------------------------------
def _same_op(a: Instruction, b: Instruction) -> Bool:
    if a.kind != b.kind or a.cond != b.cond or a.cond_val != b.cond_val or a.cond_reg != b.cond_reg or a.clbit != b.clbit:
        return False
    var ar = op_arity(a.kind)
    if ar >= 1 and a.q0 != b.q0:
        return False
    if ar >= 2 and a.q1 != b.q1:
        return False
    if ar >= 3 and a.q2 != b.q2:
        return False
    return a.p0 == b.p0 and a.p1 == b.p1 and a.p2 == b.p2


def verify_episode(prob: RProblem, ep: Episode, target: Target, mut why: String) -> Bool:
    """Replay the output items: SWAP / fused items update the mapping, every
    other op is mapped back to wires and must equal its source op; every
    source op appears exactly once, in an order consistent with the
    dependency DAG; every 2q op lies on a coupling edge; the replayed final
    mapping equals the episode's final layout."""
    var N = ep.N
    var p2l = ep.init_p2l.copy()
    var seen = List[Bool](length=len(prob.ops), fill=False)
    var nn = prob.num_nodes()
    var pos = List[Int](length=nn, fill=-1)
    var t = 0
    for i in range(len(ep.it_kind)):
        if ep.it_dead[i]:
            continue
        var s = ep.it_op_start[i]
        var L = ep.it_op_len[i]
        for k in range(s, s + L):
            var g = ep.it_ops[k]
            if op_arity(g.kind) == 2 and (op_is_unitary(g.kind) or g.q0 >= 0):
                if not target.are_connected(g.q0, g.q1):
                    why = "2q op off the coupling map: " + String(g)
                    return False
        var kind = ep.it_kind[i]
        if kind == IT_SWAP:
            if L != 1 or ep.it_ops[s].kind != Op.SWAP:
                why = "malformed swap item"
                return False
            var a = ep.it_ops[s].q0
            var b = ep.it_ops[s].q1
            var la = p2l[a]
            p2l[a] = p2l[b]
            p2l[b] = la
            continue
        var u = ep.it_node[i]
        if u < 0 or u >= nn or pos[u] >= 0:
            why = "node missing or emitted twice"
            return False
        pos[u] = t
        t += 1
        if kind == IT_FUSED:
            if L != 2 or not prob.is_single_cx(u):
                why = "malformed fused item"
                return False
            var e1 = ep.it_ops[s]
            var e2 = ep.it_ops[s + 1]
            if e1.kind != Op.CX or e2.kind != Op.CX or e1.q0 != e2.q1 or e1.q1 != e2.q0:
                why = "fused item is not CX(a,b) CX(b,a)"
                return False
            var src = prob.ops[prob.op_idx[prob.op_start[u]]]
            var m: Instruction
            if ep.it_order[i] == 0:
                # SWAP(e1 pair) then CX: mapping changes first, CX == e1
                var la = p2l[e1.q0]
                p2l[e1.q0] = p2l[e1.q1]
                p2l[e1.q1] = la
                m = e1
            else:
                m = e2
            var back = m
            back.q0 = p2l[m.q0]
            back.q1 = p2l[m.q1]
            if not _same_op(back, src):
                why = "fused CX does not map back to its source op"
                return False
            seen[prob.op_idx[prob.op_start[u]]] = True
            if ep.it_order[i] == 1:
                var la = p2l[e1.q0]
                p2l[e1.q0] = p2l[e1.q1]
                p2l[e1.q1] = la
            continue
        if L != prob.op_start[u + 1] - prob.op_start[u]:
            why = "node op count changed"
            return False
        for k in range(L):
            var g = ep.it_ops[s + k]
            var src = prob.ops[prob.op_idx[prob.op_start[u] + k]]
            var back = g
            var ar = op_arity(g.kind)
            if ar >= 1 and g.q0 >= 0:
                back.q0 = p2l[g.q0]
            if ar >= 2 and g.q1 >= 0:
                back.q1 = p2l[g.q1]
            if ar >= 3 and g.q2 >= 0:
                back.q2 = p2l[g.q2]
            if not _same_op(back, src):
                why = "op does not map back to its source: " + String(g) + " vs " + String(src)
                return False
            seen[prob.op_idx[prob.op_start[u] + k]] = True
    for k in range(len(seen)):
        if not seen[k]:
            why = "source op #" + String(k) + " missing"
            return False
    for u in range(nn):
        if pos[u] < 0:
            why = "node missing"
            return False
        for e in range(prob.succ_start[u], prob.succ_start[u + 1]):
            if pos[prob.succ[e]] <= pos[u]:
                why = "dependency order violated"
                return False
    for p in range(N):
        if p2l[p] != ep.p2l[p]:
            why = "final layout mismatch"
            return False
    return True


# ----------------------------------------------------------------------------
# Search: parallel episodes + bidirectional layout iterations
# ----------------------------------------------------------------------------
struct BatchResult(Copyable):
    var best_run: Int
    var score: List[Int]
    var final_l2p: List[Int]
    var steps: Int

    def __init__(out self):
        self.best_run = -1
        self.score = List[Int]()
        self.final_l2p = List[Int]()
        self.steps = 0


def _episode_seed(base: Int, a: Int, b: Int, c: Int, r: Int) -> UInt64:
    var g = Rng(UInt64(base)).split(a * 1000003 + b * 1009 + c).split(r)
    return g.next_u64()


def _episode_score(prob: RProblem, ep: Episode, target: Target, shell: Circuit) -> List[Int]:
    var c = shell.copy()
    c.ops = ep.emit_ops()
    return _full_score(c, target)


def run_batch(prob: RProblem, target: Target, l2p: List[Int], model: RouteModel, mode: Int, beta: Float64,
              runs: Int, base: Int, a: Int, b: Int, c: Int, virtual_ok: Bool, shell: Circuit) -> BatchResult:
    """`runs` independent episodes from one initial layout (in parallel);
    the best by (cx units, 2q depth, depth, gates), lowest run on ties."""
    var R = max(runs, 1)
    var N = target.num_qubits
    var scores = List[Int](length=R * 4, fill=-1)
    var finals = List[Int](length=R * N, fill=0)
    var nsteps = List[Int](length=R, fill=0)
    var sp = scores.unsafe_ptr().as_unsafe_any_origin()
    var fp = finals.unsafe_ptr().as_unsafe_any_origin()
    var np = nsteps.unsafe_ptr().as_unsafe_any_origin()
    var maxs = episode_max_steps(prob, target)

    def task(r: Int) {imm prob, imm target, imm l2p, imm model, imm mode, imm beta, imm base, imm a, imm b, imm c, imm virtual_ok, imm shell, imm sp, imm fp, imm np, imm N, imm maxs}:
        var ep = Episode(prob, target, l2p, _episode_seed(base, a, b, c, r), virtual_ok)
        ep.run(prob, target, model, mode, beta, maxs)
        np[unsafe_offset=r] = ep.steps
        if ep.failed:
            return
        var sc = _episode_score(prob, ep, target, shell)
        for i in range(4):
            sp[unsafe_offset=r * 4 + i] = sc[i]
        for w in range(N):
            fp[unsafe_offset=r * N + w] = ep.l2p[w]

    if R == 1:
        task(0)
    else:
        parallelize(task, R)
    _ = len(scores)
    _ = len(finals)
    _ = len(nsteps)
    var res = BatchResult()
    for r in range(R):
        res.steps += nsteps[r]
        if scores[r * 4] < 0:
            continue
        var sc = List[Int]()
        for i in range(4):
            sc.append(scores[r * 4 + i])
        if res.best_run < 0 or _better(sc, res.score):
            res.best_run = r
            res.score = sc^
    if res.best_run >= 0:
        for w in range(N):
            res.final_l2p.append(finals[res.best_run * N + w])
    return res^


struct SearchBest(Copyable):
    var score: List[Int]
    var l2p: List[Int]
    var key: List[Int]  # (a, b, c, run) seed key of the winning episode
    var episodes: Int
    var steps: Int

    def __init__(out self):
        self.score = List[Int]()
        self.l2p = List[Int]()
        self.key = List[Int]()
        self.episodes = 0
        self.steps = 0


def _layout_key(l: List[Int]) -> Int:
    var h: UInt64 = 0xCBF29CE484222325
    for v in l:
        h ^= UInt64(v + 1)
        h *= 0x100000001B3
    return Int(h & 0x3FFFFFFFFFFFFFFF)


def layout_search(fwd: RProblem, rev: RProblem, target: Target, seeds: List[List[Int]], model: RouteModel,
                  mode: Int, beta: Float64, runs: Int, its: Int, reps: Int, budget: Int, base: Int,
                  bidirectional: Bool, virtual_ok: Bool, shell: Circuit, max_episodes: Int = 1 << 30) -> SearchBest:
    """For each restart and seed layout: route forward from L, then (odd
    iterations) route the reversed circuit and feed its final placement back
    as the next L; forward results compete (lexicographic score). Stops when
    a layout repeats or the deterministic step budget is spent."""
    var best = SearchBest()
    var rshell = shell.copy()
    for rep in range(max(reps, 1)):
        for si in range(len(seeds)):
            var L = seeds[si].copy()
            var seen = List[Int]()
            var nit = max(its, 1) if bidirectional else 1
            if bidirectional and max_episodes < (1 << 30):
                # spread the episode cap over the seed layouts (>= one forward/reverse pair each)
                nit = max(2, min(nit, max_episodes // max(1, max(runs, 1) * len(seeds))))
            for it in range(nit):
                if (best.steps > budget or best.episodes >= max_episodes) and len(best.score) > 0:
                    return best^
                var reverse = it % 2 == 1
                var br: BatchResult
                if reverse:
                    br = run_batch(rev, target, L, model, mode, beta, runs, base, rep, si, it, False, rshell)
                else:
                    br = run_batch(fwd, target, L, model, mode, beta, runs, base, rep, si, it, virtual_ok, shell)
                best.steps += br.steps
                best.episodes += max(runs, 1)
                if br.best_run < 0:
                    break
                if not reverse and _better(br.score, best.score):
                    best.score = br.score.copy()
                    best.l2p = L.copy()
                    best.key = [rep, si, it, br.best_run]
                var k = _layout_key(br.final_l2p)
                var rep_seen = False
                for s in seen:
                    if s == k:
                        rep_seen = True
                if rep_seen:
                    break
                seen.append(k)
                L = br.final_l2p.copy()
    return best^


# ----------------------------------------------------------------------------
# Entry points
# ----------------------------------------------------------------------------
def _check_routable(c: Circuit, target: Target) raises:
    var n = c.num_qubits
    var N = target.num_qubits
    if n > N:
        raise Error("QM-E605: circuit has " + String(n) + " qubits but target '" + target.name + "' has only " + String(N) + ". Hint: pick a larger target (e.g. Target.grid / heavy_hex) or reduce the circuit width.")


def _complete(l2p: List[Int], n: Int, N: Int) -> List[Int]:
    """Wire -> physical of length N from the first n entries (ancillas fill
    the unused physical qubits in order); invalid input -> trivial layout."""
    var out = List[Int](length=N, fill=-1)
    var used = List[Bool](length=N, fill=False)
    var ok = True
    for w in range(min(n, len(l2p))):
        var p = l2p[w]
        if p < 0 or p >= N or used[p]:
            ok = False
            break
        out[w] = p
        used[p] = True
    if not ok:
        for w in range(N):
            out[w] = w
        return out^
    var nxt = 0
    for w in range(N):
        if out[w] < 0:
            while used[nxt]:
                nxt += 1
            out[w] = nxt
            used[nxt] = True
    return out^


def _random_layout(N: Int, mut rng: Rng) -> List[Int]:
    var l = List[Int](length=N, fill=0)
    for i in range(N):
        l[i] = i
    var i = N - 1
    while i > 0:
        var j = rng.below(i + 1)
        var t = l[i]
        l[i] = l[j]
        l[j] = t
        i -= 1
    return l^


def _conforms(c: Circuit, target: Target) -> Bool:
    if target.full:
        return True
    for g in c.ops:
        if op_is_unitary(g.kind) and op_arity(g.kind) == 2:
            if not target.are_connected(g.q0, g.q1):
                return False
        if op_is_unitary(g.kind) and op_arity(g.kind) == 3:
            return False
    return True


def _mode_of(policy: String) -> Int:
    if policy == "heuristic":
        return MODE_HEURISTIC
    if policy == "scored" or policy.startswith("scored:"):
        return MODE_SCORED
    return MODE_POLICY


def resolve_route_model(aopts: AIRouteOptions) raises -> RouteModel:
    """Model for the policy string ('' id / not ok when none is needed or
    'auto' finds none). Raises QM-E617 / QM-E620 for an explicit request."""
    var p = aopts.policy
    if p == "off" or p == "heuristic":
        return RouteModel()
    var req = String("auto")
    if p.startswith("scored:"):
        req = String(p[byte=7:])
    elif p != "scored" and p != "auto":
        req = p
    var dirs = model_search_dirs(aopts.model_dir)
    var d = find_route_model(req, dirs)
    if d.byte_length() == 0:
        return RouteModel()
    return load_route_model(d)


def ai_route_with_model(c: Circuit, target: Target, ropts: RouteOptions, aopts: AIRouteOptions, model: RouteModel,
                        mut out: Circuit, mut init_layout: List[Int], mut final_layout: List[Int],
                        mut note: String) raises -> Bool:
    """`ai_route` with an already-loaded model (tests, benchmarks, training)."""
    _check_routable(c, target)
    var cc = c.copy()
    _ = unroll_3q(cc)
    var n = cc.num_qubits
    var N = target.num_qubits
    if target.full:
        out = cc^
        init_layout = List[Int](length=n, fill=0)
        for i in range(n):
            init_layout[i] = i
        final_layout = init_layout.copy()
        note = "all-to-all target: no routing needed"
        return True
    var mode = _mode_of(aopts.policy)
    var use_ai = aopts.policy != "off" and (model.ok or mode == MODE_HEURISTIC)
    # SABRE candidate (portfolio, layout seed, or the fallback when no model)
    var have_sabre = False
    var s_out = Circuit(1)
    var s_il = List[Int]()
    var s_fl = List[Int]()
    var s_note = String()
    var s_layout = List[Int]()
    var keep = aopts.layout_mode == "keep"
    var ro = ropts.copy()
    if keep:
        ro.initial_layout = aopts.seed_layout.copy()
        if len(ro.initial_layout) == 0:
            ro.layout = "trivial"
    elif aopts.layout_mode == "improve" and len(aopts.seed_layout) > 0:
        ro.initial_layout = aopts.seed_layout.copy()
    if not use_ai and not aopts.portfolio:
        note = "ai_route: no proposal (" + ("policy=off" if aopts.policy == "off" else "no router model found for policy=" + aopts.policy) + ")"
        return False
    if aopts.portfolio or not use_ai or (aopts.layout_mode == "optimize" and len(aopts.seed_layout) == 0):
        var plan = plan_route(cc, target, ro)
        s_layout = plan.layout.copy()
        if aopts.portfolio or not use_ai:
            s_out = cc.copy()
            _ = finish_route(s_out, target, ro, plan, s_il, s_fl, s_note)
            have_sabre = _conforms(s_out, target)
    if not use_ai:
        if not have_sabre:
            return False
        out = s_out^
        init_layout = s_il^
        final_layout = s_fl^
        var why = "policy=off" if aopts.policy == "off" else "no router model found (policy=" + aopts.policy + ")"
        note = "ai_route: " + why + "; ran SABRE: " + s_note
        return True
    # seeds
    var lvl = min(max(aopts.level, 1), 3)
    var eff = level_effort(lvl)
    var runs = eff[0]
    var its = eff[1]
    var reps = eff[2]
    var max_ms = aopts.max_ms if aopts.max_ms > 0 else eff[3]
    var budget = max_ms * STEPS_PER_MS
    var seeds = List[List[Int]]()
    var bidir = True
    var vf2 = False
    if keep:
        seeds.append(_complete(aopts.seed_layout, n, N))
        bidir = False
    elif aopts.layout_mode == "improve":
        seeds.append(_complete(aopts.seed_layout, n, N))
    else:
        var skel = routing_skeleton(cc.ops, n)
        var lv = List[Int]()
        if vf2_layout(skel, n, target, ro.vf2_steps, lv):
            seeds.append(_complete(lv, n, N))
            vf2 = True
            bidir = False
            reps = 1
            runs = 1
        else:
            if len(aopts.seed_layout) > 0:
                seeds.append(_complete(aopts.seed_layout, n, N))
            if len(s_layout) > 0:
                seeds.append(_complete(s_layout, n, N))
            seeds.append(_complete(dense_layout(cc.ops, n, target), n, N))
            var ig = InteractionGraph(skel, n, 0.5)
            seeds.append(_complete(greedy_layout(ig, target, 0), n, N))
            var rng = Rng(UInt64(aopts.seed) + 101)
            seeds.append(_random_layout(N, rng))
    var fwd = RProblem(cc.ops, n)
    var rev = RProblem(reversed_ops(cc.ops), n)
    var shell = shell_like(cc, N)
    var best = layout_search(fwd, rev, target, seeds, model, mode, 1.0, runs, its, reps, budget, aopts.seed,
                             bidir, not keep, shell, eff[4])
    var ai_ok = False
    var a_out = Circuit(1)
    var a_il = List[Int]()
    var a_fl = List[Int]()
    var a_note = String()
    if len(best.key) == 4:
        var k = best.key.copy()
        var ep = Episode(fwd, target, best.l2p, _episode_seed(aopts.seed, k[0], k[1], k[2], k[3]), not keep)
        ep.run(fwd, target, model, mode, 1.0, episode_max_steps(fwd, target))
        var why = String()
        if not ep.failed and verify_episode(fwd, ep, target, why):
            a_out = shell.copy()
            a_out.ops = ep.emit_ops()
            _ = slide_swaps(a_out.ops, target)
            if _conforms(a_out, target):
                ai_ok = True
                a_il = ep.init_l2p()
                a_fl = ep.l2p.copy()
                a_note = "policy=" + (model.id if model.ok and mode != MODE_HEURISTIC else "heuristic") + ("(scored)" if mode == MODE_SCORED else "") + " swaps=" + String(ep.swaps_live + ep.n_fused) + " (virtual " + String(ep.n_virtual) + ", cancelled " + String(ep.n_cancel) + ", fused " + String(ep.n_fused) + ", fallback steps " + String(ep.n_fallback) + "), episodes=" + String(best.episodes) + (", vf2 layout" if vf2 else "")
        else:
            a_note = "ai route rejected by verification (" + why + ")"
    if not ai_ok and not have_sabre:
        if aopts.portfolio:
            return False
        # no SABRE run yet: run it now as the safe fallback
        var plan = plan_route(cc, target, ro)
        s_out = cc.copy()
        _ = finish_route(s_out, target, ro, plan, s_il, s_fl, s_note)
        have_sabre = _conforms(s_out, target)
        if not have_sabre:
            return False
        out = s_out^
        init_layout = s_il^
        final_layout = s_fl^
        note = "ai_route: " + a_note + "; fell back to SABRE: " + s_note
        return True
    if ai_ok and have_sabre:
        var ca = routed_cost(a_out, target)
        var cs = routed_cost(s_out, target)
        var sa: List[Int] = [ca[0], ca[1]]
        var ss: List[Int] = [cs[0], cs[1]]
        if not _better(sa, ss):
            # ties keep SABRE (the established result)
            out = s_out^
            init_layout = s_il^
            final_layout = s_fl^
            var how = "tie, SABRE kept" if not _better(ss, sa) else "SABRE won"
            note = "ai_route portfolio: " + how + " (cx " + String(cs[0]) + " vs policy " + String(ca[0]) + ", 2q depth " + String(cs[1]) + " vs " + String(ca[1]) + "); " + a_note
            return True
        out = a_out^
        init_layout = a_il^
        final_layout = a_fl^
        note = "ai_route portfolio: policy won (cx " + String(ca[0]) + " vs SABRE " + String(cs[0]) + ", 2q depth " + String(ca[1]) + " vs " + String(cs[1]) + "); " + a_note
        return True
    if ai_ok:
        out = a_out^
        init_layout = a_il^
        final_layout = a_fl^
        note = "ai_route: " + a_note
        return True
    out = s_out^
    init_layout = s_il^
    final_layout = s_fl^
    note = "ai_route: " + a_note + "; SABRE result kept: " + s_note
    return True


def ai_route(c: Circuit, target: Target, ropts: RouteOptions, aopts: AIRouteOptions,
             mut out: Circuit, mut init_layout: List[Int], mut final_layout: List[Int],
             mut note: String) raises -> Bool:
    """Learned-policy layout + routing (see the module docstring).

    Returns True only with a verified physical circuit in `out` (on N
    physical qubits; `init_layout` / `final_layout` as in
    `routing.route_circuit`). policy=auto without a model, or policy=off,
    runs QUMO's SABRE and says so in `note`. Raises QM-E617 / QM-E620 when an
    explicitly requested model is missing / corrupt / incompatible.
    """
    var model = RouteModel()
    if not target.full:
        model = resolve_route_model(aopts)
    return ai_route_with_model(c, target, ropts, aopts, model, out, init_layout, final_layout, note)
