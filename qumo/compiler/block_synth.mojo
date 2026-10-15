"""Exact and greedy topology-aware block synthesizers.

Native, hardware-neutral re-implementation of the synthesis side of
qiskit-ibm-transpiler 0.18.0 (`AIPermutationSynthesis`,
`AILinearFunctionSynthesis`, `AICliffordSynthesis`, `AIPauliNetworkSynthesis`,
Apache-2.0, used as a reference only): instead of a learned policy these use
exact search for small blocks and topology-aware greedy elimination for the
rest. A learned policy (`policy_synth.mojo`) is one more proposer next to these.

Contract (all functions):
  * input: a canonical problem (`Perm`, `GF2Mat`, `SympTab`, `PauliNet`) and a
    `SubGraph` g on local qubits 0..g.n-1;
  * output: instructions on local qubits whose every 2q gate lies on an edge
    of g, from {H, S, SDG, SX, SXDG, X, Y, Z, CX, SWAP, RX, RY, RZ}; the
    output is exactly the problem (Clifford: tableau with signs; i.e. equal up
    to global phase). An empty list means "no proposal" (also returned for an
    identity problem - callers test `is_identity()` first);
  * `method`: "exact" (optimal, or empty when the node budget runs out),
    "astar" (A* / bidirectional search with the given budget), "greedy",
    "auto" (greedy portfolio, plus exact search when the block is small and
    the default ~1-2 ms budget suffices; best result wins);
  * `budget`: node-expansion cap (0 = method default); `seed`: deterministic
    tie-breaking for the randomized greedy trials (`Rng(seed).split(k)`).

Costs compare (2q count in CX units with SWAP = 3, 2q depth, gate count).

Algorithms (see docs/AI_TRANSPILER.md for complexity and references):
  * permutation: A* over token placements with the consistent heuristic
    ceil(sum_v dist(v, dest(token at v)) / 2); greedy approximate token
    swapping (happy swaps, swap cycles, unhappy swap - Miltzow et al. 2016)
    with a spanning-tree fallback;
  * linear function: bidirectional BFS (optimal) / A* (h = rows of M that
    differ from the identity: one CX changes one row, so h is consistent)
    over GF(2) matrices; greedy = topology-aware Steiner-tree elimination
    (Kissinger-Meijer-van de Griend; Wu et al. RowCol) with non-cut pivots;
  * Clifford: bidirectional BFS over left cosets of the local Clifford group
    (|Sp(6,2)| / 6^3 = 6720 classes for 3 qubits) giving the optimal CX
    count, then exact sign repair; greedy = per-qubit decoupling of (X_q, Z_q)
    with Steiner-tree CX ladders, pivots chosen among non-cut vertices;
  * Pauli network: greedy front-layer elimination (qiskit-gym semantics),
    then the residual Clifford by the Clifford synthesizer.
"""

from std.bit import pop_count
from qumo.ir.gates import Op, op_arity
from qumo.ir.circuit import Instruction
from qumo.sim.rng import Rng
from qumo.compiler.blockalg import (
    BlockKind, SubGraph, Perm, GF2Mat, SympTab, PauliRot, PauliNet, emit_rotation,
    g1, g2, inverse_ops, ops_cx_count, ops_depth2q, paulis_anticommute,
)

comptime AUTO_BUDGET = 20000
comptime ASTAR_BUDGET = 200000
comptime EXACT_BUDGET = 2000000


# ----------------------------------------------------------------------------
# Common helpers
# ----------------------------------------------------------------------------
@always_inline
def _bit(j: Int) -> UInt64:
    return UInt64(1) << UInt64(j)


@always_inline
def _has(m: UInt64, j: Int) -> Bool:
    return ((m >> UInt64(j)) & 1) != 0


def _check_method(method: String) raises:
    if method != "exact" and method != "astar" and method != "greedy" and method != "auto":
        raise Error("QM-E616: unknown synthesis method '" + method + "'. Hint: use exact, astar, greedy or auto (policy is handled by policy_synth).")


def _check_width(kind: Int, n: Int, method: String) raises:
    _check_method(method)
    if method == "exact" and not exact_supported(kind, n):
        _too_wide(kind, n, method)
    if method == "astar" and not _astar_supported(kind, n):
        _too_wide(kind, n, method)


def _too_wide(kind: Int, n: Int, method: String) raises:
    var name = "clifford" if kind == BlockKind.CLIFFORD else ("linear" if kind == BlockKind.LINEAR else ("perm" if kind == BlockKind.PERM else "pauli"))
    raise Error("QM-E618: block too wide for method '" + method + "' (" + name + ", " + String(n) + " qubits). Hint: use method=auto or greedy, or lower max_qubits.")


def exact_supported(kind: Int, n: Int) -> Bool:
    """Whether method=exact can (within its default budget) be optimal."""
    if kind == BlockKind.PERM:
        return n <= 10
    if kind == BlockKind.LINEAR:
        return n <= 5
    if kind == BlockKind.CLIFFORD:
        return n <= 4
    return False


def _astar_supported(kind: Int, n: Int) -> Bool:
    if kind == BlockKind.PERM:
        return n <= 16
    if kind == BlockKind.LINEAR:
        return n <= 8
    if kind == BlockKind.CLIFFORD:
        return n <= 4
    return False


def _budget(method: String, budget: Int) -> Int:
    if budget > 0:
        return budget
    if method == "exact":
        return EXACT_BUDGET
    if method == "astar":
        return ASTAR_BUDGET
    return AUTO_BUDGET


def _better(a: List[Instruction], b: List[Instruction], n: Int) -> Bool:
    """a strictly better than b: (2q count, 2q depth, gate count)."""
    var ca = ops_cx_count(a)
    var cb = ops_cx_count(b)
    if ca != cb:
        return ca < cb
    var da = ops_depth2q(a, n)
    var db = ops_depth2q(b, n)
    if da != db:
        return da < db
    return len(a) < len(b)


def _reverse_flip_cx(ops: List[Instruction]) -> List[Instruction]:
    """Circuit of M from a CX circuit of M^T: reversed with every CX flipped."""
    var out = List[Instruction](capacity=len(ops))
    var i = len(ops) - 1
    while i >= 0:
        var o = ops[i]
        if o.kind == Op.CX:
            out.append(g2(Op.CX, o.q1, o.q0))
        else:
            out.append(o)
        i -= 1
    return out^


struct _Adj(Copyable):
    var start: List[Int]
    var nb: List[Int]

    def __init__(out self, g: SubGraph):
        var n = g.n
        self.start = List[Int](length=n + 1, fill=0)
        self.nb = List[Int]()
        for a in range(n):
            self.start[a] = len(self.nb)
            for b in range(n):
                if g.dist[a * n + b] == 1:
                    self.nb.append(b)
        self.start[n] = len(self.nb)

    def __init__(out self, *, copy: Self):
        self.start = copy.start.copy()
        self.nb = copy.nb.copy()


struct _U64Map(Movable):
    """Open-addressing hash map UInt64 -> Int (key ~0 is reserved)."""

    var keys: List[UInt64]
    var vals: List[Int]
    var bits: Int
    var count: Int

    def __init__(out self, bits: Int = 12):
        self.bits = bits
        self.keys = List[UInt64](length=1 << bits, fill=~UInt64(0))
        self.vals = List[Int](length=1 << bits, fill=-1)
        self.count = 0

    @always_inline
    def _slot(self, k: UInt64) -> Int:
        var h = (k ^ (k >> 29)) * 0x9E3779B97F4A7C15
        return Int(h >> UInt64(64 - self.bits))

    def get(self, k: UInt64) -> Int:
        var mask = (1 << self.bits) - 1
        var i = self._slot(k)
        while True:
            var kk = self.keys[i]
            if kk == k:
                return self.vals[i]
            if kk == ~UInt64(0):
                return -1
            i = (i + 1) & mask

    def put(mut self, k: UInt64, v: Int):
        if 2 * (self.count + 1) > (1 << self.bits):
            self._grow()
        var mask = (1 << self.bits) - 1
        var i = self._slot(k)
        while True:
            var kk = self.keys[i]
            if kk == k:
                self.vals[i] = v
                return
            if kk == ~UInt64(0):
                self.keys[i] = k
                self.vals[i] = v
                self.count += 1
                return
            i = (i + 1) & mask

    def _grow(mut self):
        var ok = self.keys.copy()
        var ov = self.vals.copy()
        self.bits += 1
        self.keys = List[UInt64](length=1 << self.bits, fill=~UInt64(0))
        self.vals = List[Int](length=1 << self.bits, fill=-1)
        self.count = 0
        for i in range(len(ok)):
            if ok[i] != ~UInt64(0):
                self.put(ok[i], ov[i])


struct _Heap(Movable, Sized):
    """Binary min-heap of (priority, payload) integer pairs."""

    var pr: List[Int]
    var pl: List[Int]

    def __init__(out self):
        self.pr = List[Int]()
        self.pl = List[Int]()

    def __len__(self) -> Int:
        return len(self.pr)

    def push(mut self, p: Int, v: Int):
        self.pr.append(p)
        self.pl.append(v)
        var i = len(self.pr) - 1
        while i > 0:
            var par = (i - 1) // 2
            if self.pr[par] < self.pr[i] or (self.pr[par] == self.pr[i] and self.pl[par] <= self.pl[i]):
                break
            var tp = self.pr[par]; self.pr[par] = self.pr[i]; self.pr[i] = tp
            var tv = self.pl[par]; self.pl[par] = self.pl[i]; self.pl[i] = tv
            i = par

    def pop(mut self) -> Int:
        var top = self.pl[0]
        var last = len(self.pr) - 1
        self.pr[0] = self.pr[last]
        self.pl[0] = self.pl[last]
        _ = self.pr.pop()
        _ = self.pl.pop()
        var n = len(self.pr)
        var i = 0
        while True:
            var l = 2 * i + 1
            var r = l + 1
            var m = i
            if l < n and (self.pr[l] < self.pr[m] or (self.pr[l] == self.pr[m] and self.pl[l] < self.pl[m])):
                m = l
            if r < n and (self.pr[r] < self.pr[m] or (self.pr[r] == self.pr[m] and self.pl[r] < self.pl[m])):
                m = r
            if m == i:
                break
            var tp = self.pr[m]; self.pr[m] = self.pr[i]; self.pr[i] = tp
            var tv = self.pl[m]; self.pl[m] = self.pl[i]; self.pl[i] = tv
            i = m
        return top


# ----------------------------------------------------------------------------
# Steiner trees on the alive part of a graph
# ----------------------------------------------------------------------------
struct _Tree(Movable):
    var parent: List[Int]  # -2 = not in tree, -1 = root
    var order: List[Int]  # tree nodes by decreasing depth (children first; root last)
    var size: Int  # number of tree edges

    def __init__(out self, n: Int):
        self.parent = List[Int](length=n, fill=-2)
        self.order = List[Int]()
        self.size = 0


def _steiner(adj: _Adj, alive: List[Bool], terms: List[Int], root: Int, mut rng: Rng, randomize: Bool) -> _Tree:
    """Shortest-path Steiner-tree heuristic: grow from `root`, repeatedly
    attaching the nearest remaining terminal (ties: lowest index, or random)."""
    var n = len(alive)
    var t = _Tree(n)
    t.parent[root] = -1
    var in_tree = List[Bool](length=n, fill=False)
    in_tree[root] = True
    var need = List[Bool](length=n, fill=False)
    var remaining = 0
    for v in terms:
        if not in_tree[v] and not need[v]:
            need[v] = True
            remaining += 1
    var dist = List[Int](length=n, fill=-1)
    var pred = List[Int](length=n, fill=-1)
    var queue = List[Int](length=n, fill=0)
    while remaining > 0:
        for v in range(n):
            dist[v] = -1
            pred[v] = -1
        var head = 0
        var tail = 0
        for v in range(n):
            if in_tree[v]:
                dist[v] = 0
                queue[tail] = v
                tail += 1
        while head < tail:
            var u = queue[head]
            head += 1
            for k in range(adj.start[u], adj.start[u + 1]):
                var w = adj.nb[k]
                if alive[w] and dist[w] < 0:
                    dist[w] = dist[u] + 1
                    pred[w] = u
                    queue[tail] = w
                    tail += 1
        var best = -1
        var bd = n + 1
        var nties = 0
        for v in range(n):
            if need[v] and dist[v] >= 0:
                if dist[v] < bd:
                    bd = dist[v]
                    best = v
                    nties = 1
                elif dist[v] == bd and randomize:
                    nties += 1
                    if rng.below(nties) == 0:
                        best = v
        if best < 0:
            t.size = -1
            return t^
        var v = best
        while not in_tree[v]:
            in_tree[v] = True
            t.parent[v] = pred[v]
            if need[v]:
                need[v] = False
                remaining -= 1
            t.size += 1
            v = pred[v]
    # order by decreasing depth
    var depth = List[Int](length=n, fill=-1)
    depth[root] = 0
    var changed = True
    while changed:
        changed = False
        for v in range(n):
            if t.parent[v] >= 0 and depth[v] < 0 and depth[t.parent[v]] >= 0:
                depth[v] = depth[t.parent[v]] + 1
                changed = True
    var maxd = 0
    for v in range(n):
        maxd = max(maxd, depth[v])
    var d = maxd
    while d >= 0:
        for v in range(n):
            if t.parent[v] != -2 and depth[v] == d:
                t.order.append(v)
        d -= 1
    return t^


def _is_noncut(adj: _Adj, alive: List[Bool], v: Int) -> Bool:
    """v alive and removing it keeps the alive vertices connected."""
    var n = len(alive)
    var cnt = 0
    var start = -1
    for u in range(n):
        if alive[u] and u != v:
            cnt += 1
            if start < 0:
                start = u
    if cnt == 0:
        return True
    var seen = List[Bool](length=n, fill=False)
    var queue = List[Int](length=n, fill=0)
    seen[start] = True
    queue[0] = start
    var head = 0
    var tail = 1
    while head < tail:
        var u = queue[head]
        head += 1
        for k in range(adj.start[u], adj.start[u + 1]):
            var w = adj.nb[k]
            if alive[w] and w != v and not seen[w]:
                seen[w] = True
                queue[tail] = w
                tail += 1
    return tail == cnt


# ----------------------------------------------------------------------------
# Permutations
# ----------------------------------------------------------------------------
def _perm_h(arr: List[Int], dest: List[Int], g: SubGraph) -> Int:
    var s = 0
    for v in range(len(arr)):
        s += g.dist[v * g.n + dest[arr[v]]]
    return s


def _perm_astar(p: Perm, g: SubGraph, budget: Int, mut out: List[Instruction]) -> Bool:
    """Optimal SWAP count by A* (consistent heuristic); False when over budget."""
    var n = g.n
    if n > 16:
        return False
    var dest = p.p.copy()
    var m = g.num_edges()
    var arr = List[Int](length=n, fill=0)
    var start: UInt64 = 0
    for v in range(n):
        arr[v] = v
        start |= UInt64(v) << UInt64(4 * v)
    var h0 = _perm_h(arr, dest, g)
    var states = List[UInt64]()
    var parent = List[Int]()
    var move = List[Int]()
    var gc = List[Int]()
    var hs = List[Int]()
    var best = _U64Map(12)
    var heap = _Heap()
    states.append(start); parent.append(-1); move.append(-1); gc.append(0); hs.append(h0)
    best.put(start, 0)
    heap.push(((h0 + 1) // 2) * 4096, 0)
    var expansions = 0
    var goal = -1
    while len(heap) > 0:
        var node = heap.pop()
        var gnode = gc[node]
        if best.get(states[node]) < gnode:
            continue
        if hs[node] == 0:
            goal = node
            break
        expansions += 1
        if expansions > budget:
            return False
        var key = states[node]
        for v in range(n):
            arr[v] = Int((key >> UInt64(4 * v)) & 15)
        var hcur = hs[node]
        for e in range(m):
            var a = g.edges[2 * e]
            var b = g.edges[2 * e + 1]
            var ta = arr[a]
            var tb = arr[b]
            var dh = g.dist[b * n + dest[ta]] + g.dist[a * n + dest[tb]] - g.dist[a * n + dest[ta]] - g.dist[b * n + dest[tb]]
            var nk = key & ~((UInt64(15) << UInt64(4 * a)) | (UInt64(15) << UInt64(4 * b)))
            nk |= (UInt64(tb) << UInt64(4 * a)) | (UInt64(ta) << UInt64(4 * b))
            var ng = gnode + 1
            var old = best.get(nk)
            if old >= 0 and old <= ng:
                continue
            best.put(nk, ng)
            var nh = hcur + dh
            states.append(nk); parent.append(node); move.append(e); gc.append(ng); hs.append(nh)
            var f = ng + (nh + 1) // 2
            heap.push(f * 4096 + (4095 - min(ng, 4095)), len(states) - 1)
    if goal < 0:
        return False
    var seq = List[Int]()
    var cur = goal
    while parent[cur] >= 0:
        seq.append(move[cur])
        cur = parent[cur]
    out.clear()
    var i = len(seq) - 1
    while i >= 0:
        out.append(g2(Op.SWAP, g.edges[2 * seq[i]], g.edges[2 * seq[i] + 1]))
        i -= 1
    return True


def _perm_tree_route(p: Perm, g: SubGraph, adj: _Adj, mut out: List[Instruction]) -> Bool:
    """Always-terminating fallback: on a BFS spanning tree, repeatedly pick a
    leaf, bring the token destined to it along the tree path, then drop it."""
    var n = g.n
    var arr = List[Int](length=n, fill=0)
    var pos = List[Int](length=n, fill=0)
    for v in range(n):
        arr[v] = v
        pos[v] = v
    var dest_of = List[Int](length=n, fill=0)  # token that must end on vertex v
    for i in range(n):
        dest_of[p.p[i]] = i
    var alive = List[Bool](length=n, fill=True)
    for _ in range(n):
        # leaf = alive non-cut vertex with the largest index (deterministic)
        var leaf = -1
        var v = n - 1
        while v >= 0:
            if alive[v] and _is_noncut(adj, alive, v):
                leaf = v
                break
            v -= 1
        if leaf < 0:
            return False
        var tok = dest_of[leaf]
        # BFS path in the alive graph from pos[tok] to leaf
        var src = pos[tok]
        if src != leaf:
            var pred = List[Int](length=n, fill=-1)
            var seen = List[Bool](length=n, fill=False)
            var queue = List[Int](length=n, fill=0)
            seen[src] = True
            queue[0] = src
            var head = 0
            var tail = 1
            while head < tail:
                var u = queue[head]
                head += 1
                for k in range(adj.start[u], adj.start[u + 1]):
                    var w = adj.nb[k]
                    if alive[w] and not seen[w]:
                        seen[w] = True
                        pred[w] = u
                        queue[tail] = w
                        tail += 1
            if not seen[leaf]:
                return False
            var path = List[Int]()
            var c = leaf
            while c != src:
                path.append(c)
                c = pred[c]
            path.append(src)
            # path = [leaf, ..., src]; move the token from src towards leaf
            var i = len(path) - 1
            while i > 0:
                var a = path[i]
                var b = path[i - 1]
                out.append(g2(Op.SWAP, min(a, b), max(a, b)))
                var ta = arr[a]
                var tb = arr[b]
                arr[a] = tb
                arr[b] = ta
                pos[tb] = a
                pos[ta] = b
                i -= 1
        alive[leaf] = False
    return True


def _perm_greedy(p: Perm, g: SubGraph, adj: _Adj, mut rng: Rng, randomize: Bool, mut out: List[Instruction]) -> Bool:
    """Approximate token swapping (Miltzow, Narins, Okamoto, Rote, Thomas,
    Uno, ESA 2016): happy swaps first, then swap cycles of the 'wants to move
    to' digraph, otherwise one unhappy swap at the end of a chain. Falls back
    to `_perm_tree_route` if an iteration cap is hit."""
    var n = g.n
    var arr = List[Int](length=n, fill=0)
    for v in range(n):
        arr[v] = v
    var dest = p.p.copy()
    var m = g.num_edges()
    out.clear()
    var total = 0
    for v in range(n):
        total += g.dist[v * n + dest[v]]
    var cap = 4 * total + 4 * n * n + 16
    var iters = 0
    while True:
        var done = True
        for v in range(n):
            if dest[arr[v]] != v:
                done = False
                break
        if done:
            return True
        iters += 1
        if iters > cap:
            break
        # 1) happy swap: both tokens get closer
        var pick = -1
        var nties = 0
        for e in range(m):
            var a = g.edges[2 * e]
            var b = g.edges[2 * e + 1]
            var da = dest[arr[a]]
            var db = dest[arr[b]]
            if g.dist[b * n + da] < g.dist[a * n + da] and g.dist[a * n + db] < g.dist[b * n + db]:
                if not randomize:
                    pick = e
                    break
                nties += 1
                if rng.below(nties) == 0:
                    pick = e
        if pick >= 0:
            var a = g.edges[2 * pick]
            var b = g.edges[2 * pick + 1]
            out.append(g2(Op.SWAP, a, b))
            var t = arr[a]; arr[a] = arr[b]; arr[b] = t
            continue
        # 2) follow the 'wants' digraph from an unhappy vertex
        var startv = -1
        var cnt = 0
        for v in range(n):
            if dest[arr[v]] != v:
                if not randomize:
                    startv = v
                    break
                cnt += 1
                if rng.below(cnt) == 0:
                    startv = v
        var onpath = List[Int](length=n, fill=-1)
        var path = List[Int]()
        var v = startv
        var cycle_at = -1
        while True:
            onpath[v] = len(path)
            path.append(v)
            if dest[arr[v]] == v:
                break  # reached a vertex whose token is home
            # successor: neighbour closer to this token's destination
            var d = dest[arr[v]]
            var nxt = -1
            var c2 = 0
            for k in range(adj.start[v], adj.start[v + 1]):
                var u = adj.nb[k]
                if g.dist[u * n + d] < g.dist[v * n + d]:
                    if not randomize:
                        nxt = u
                        break
                    c2 += 1
                    if rng.below(c2) == 0:
                        nxt = u
            if onpath[nxt] >= 0:
                cycle_at = onpath[nxt]
                break
            v = nxt
        if cycle_at >= 0:
            # rotate the cycle path[cycle_at..]: every token moves one step closer
            var cyc = List[Int]()
            for i in range(cycle_at, len(path)):
                cyc.append(path[i])
            var k = len(cyc)
            var i = k - 2
            while i >= 0:
                var a = cyc[i]
                var b = cyc[i + 1]
                out.append(g2(Op.SWAP, min(a, b), max(a, b)))
                var t = arr[a]; arr[a] = arr[b]; arr[b] = t
                i -= 1
        else:
            # unhappy swap: last unhappy vertex of the chain with the home vertex
            var w = path[len(path) - 1]
            var u = path[len(path) - 2]
            out.append(g2(Op.SWAP, min(u, w), max(u, w)))
            var t = arr[u]; arr[u] = arr[w]; arr[w] = t
    out.clear()
    return _perm_tree_route(p, g, adj, out)


def synth_perm(p: Perm, g: SubGraph, method: String, budget: Int, seed: Int) raises -> List[Instruction]:
    """SWAP network on the edges of g realizing permutation p."""
    _check_width(BlockKind.PERM, g.n, method)
    var n = g.n
    var out = List[Instruction]()
    if p.size() != n:
        raise Error("QM-E618: permutation of " + String(p.size()) + " qubits on a " + String(n) + "-qubit subgraph. Hint: the problem and the subgraph must have the same size.")
    if p.is_identity():
        return out^
    if not g.connected():
        return out^
    var b = _budget(method, budget)
    if method == "exact":
        if not exact_supported(BlockKind.PERM, n):
            _too_wide(BlockKind.PERM, n, method)
        if not _perm_astar(p, g, b, out):
            out.clear()
        return out^
    if method == "astar":
        if not _astar_supported(BlockKind.PERM, n):
            _too_wide(BlockKind.PERM, n, method)
        if not _perm_astar(p, g, b, out):
            out.clear()
        return out^
    var adj = _Adj(g)
    var rng = Rng(UInt64(seed))
    var best = List[Instruction]()
    _ = _perm_greedy(p, g, adj, rng, False, best)
    var pinv = p.inverse()
    var trials = 1 if method == "greedy" else 4
    for t in range(trials):
        var r = rng.split(t + 1)
        var cand = List[Instruction]()
        if t % 2 == 0:
            if _perm_greedy(pinv, g, adj, r, t > 0, cand):
                cand = inverse_ops(cand)
            else:
                cand.clear()
        else:
            _ = _perm_greedy(p, g, adj, r, True, cand)
        if len(cand) > 0 and _better(cand, best, n):
            best = cand^
    if method == "auto" and n <= 10:
        var arr = List[Int](length=n, fill=0)
        for v in range(n):
            arr[v] = v
        var lb = (_perm_h(arr, p.p, g) + 1) // 2
        if len(best) > lb:
            var ex = List[Instruction]()
            if _perm_astar(p, g, b, ex) and _better(ex, best, n):
                best = ex^
    return best^


# ----------------------------------------------------------------------------
# Clifford: local sign repair and greedy elimination
# ----------------------------------------------------------------------------
def _word(w: Int, q: Int, mut out: List[Instruction]):
    """Local Clifford words (circuit order) for the 6 phase-free classes."""
    if w == 1:
        out.append(g1(Op.H, q))
    elif w == 2:
        out.append(g1(Op.S, q))
    elif w == 3:
        out.append(g1(Op.SX, q))
    elif w == 4:
        out.append(g1(Op.H, q)); out.append(g1(Op.S, q))
    elif w == 5:
        out.append(g1(Op.S, q)); out.append(g1(Op.H, q))


def synth_local(t: SympTab, mut out: List[Instruction]) raises:
    """Exact (signed) synthesis of a local Clifford tableau, qubit by qubit."""
    var n = t.n
    if not t.is_local():
        raise Error("QM-E619: internal error, tableau is not local after elimination. Hint: report this as a QUMO bug.")
    # candidate single-qubit circuits, shortest first
    var cands = List[List[Instruction]]()
    var paulis: List[Int] = [-1, Op.X, Op.Z, Op.Y]
    var singles: List[Int] = [Op.H, Op.S, Op.SDG, Op.SX, Op.SXDG]
    cands.append(List[Instruction]())
    for k in [Op.X, Op.Y, Op.Z]:
        var c = List[Instruction]()
        c.append(g1(k, 0))
        cands.append(c^)
    for k in singles:
        var c = List[Instruction]()
        c.append(g1(k, 0))
        cands.append(c^)
    for pk in paulis:
        for w in range(1, 6):
            var c = List[Instruction]()
            if pk >= 0:
                c.append(g1(pk, 0))
            _word(w, 0, c)
            cands.append(c^)
    var tabs = List[SympTab]()
    for c in cands:
        tabs.append(SympTab.from_ops(c, 1))
    for j in range(n):
        var want = SympTab(1)
        want.x[0] = (t.x[j] >> UInt64(j)) & 1
        want.z[0] = (t.z[j] >> UInt64(j)) & 1
        want.r[0] = t.r[j]
        want.x[1] = (t.x[n + j] >> UInt64(j)) & 1
        want.z[1] = (t.z[n + j] >> UInt64(j)) & 1
        want.r[1] = t.r[n + j]
        var found = False
        for ci in range(len(cands)):
            if tabs[ci] == want:
                for o in cands[ci]:
                    out.append(g1(o.kind, j))
                found = True
                break
        if not found:
            raise Error("QM-E619: internal error, no local Clifford matches qubit " + String(j) + ". Hint: report this as a QUMO bug.")


@always_inline
def _ap(mut t: SympTab, mut w: List[Instruction], g: Instruction) raises:
    t.apply(g)
    w.append(g)


def _x_ladder(mut t: SympTab, adj: _Adj, alive: List[Bool], q: Int, row: Int, root_fill: Bool, mut w: List[Instruction], mut rng: Rng, randomize: Bool) raises -> Bool:
    """Make row `row` (an X-type Pauli on the alive qubits after local basis
    changes) equal to +-X_q with a Steiner CX tree rooted at q."""
    var n = t.n
    for j in range(n):
        if not alive[j] or (j == q and not root_fill):
            continue
        var xb = _has(t.x[row], j)
        var zb = _has(t.z[row], j)
        if zb and not xb:
            _ap(t, w, g1(Op.H, j))
        elif zb and xb:
            _ap(t, w, g1(Op.SDG, j))
    var terms = List[Int]()
    for j in range(n):
        if _has(t.x[row], j) or _has(t.z[row], j):
            terms.append(j)
    var tr = _steiner(adj, alive, terms, q, rng, randomize)
    if tr.size < 0:
        return False
    for v in tr.order:
        if (tr.parent[v] >= 0 or root_fill) and not _has(t.x[row], v):
            # fill from a tree child that carries X: CX(u, v) maps X_u -> X_u X_v
            for u in tr.order:
                if tr.parent[u] == v and _has(t.x[row], u):
                    _ap(t, w, g2(Op.CX, u, v))
                    break
    for u in tr.order:
        if tr.parent[u] >= 0:
            _ap(t, w, g2(Op.CX, tr.parent[u], u))  # X_p X_u -> X_p
    return True


def _z_ladder_tab(mut t: SympTab, adj: _Adj, alive: List[Bool], q: Int, row: Int, root_fill: Bool, mut w: List[Instruction], mut rng: Rng, randomize: Bool) raises -> Bool:
    """Make row `row` (a Z-type Pauli on the alive qubits after local basis
    changes) equal to +-Z_q with a Steiner CX tree rooted at q."""
    var n = t.n
    for j in range(n):
        if not alive[j] or (j == q and not root_fill):
            continue
        var xb = _has(t.x[row], j)
        var zb = _has(t.z[row], j)
        if xb and not zb:
            _ap(t, w, g1(Op.H, j))
        elif xb and zb:
            _ap(t, w, g1(Op.SX, j))
    var terms = List[Int]()
    for j in range(n):
        if _has(t.x[row], j) or _has(t.z[row], j):
            terms.append(j)
    var tr = _steiner(adj, alive, terms, q, rng, randomize)
    if tr.size < 0:
        return False
    for v in tr.order:
        if (tr.parent[v] >= 0 or root_fill) and not _has(t.z[row], v):
            # fill from a tree child that carries Z: CX(v, u) maps Z_u -> Z_v Z_u
            for u in tr.order:
                if tr.parent[u] == v and _has(t.z[row], u):
                    _ap(t, w, g2(Op.CX, v, u))
                    break
    for u in tr.order:
        if tr.parent[u] >= 0:
            _ap(t, w, g2(Op.CX, u, tr.parent[u]))  # Z_u Z_p -> Z_p
    return True


def _reduce_qubit(mut t: SympTab, adj: _Adj, alive: List[Bool], q: Int, zfirst: Bool, mut w: List[Instruction], mut rng: Rng, randomize: Bool) raises -> Bool:
    """Apply gates (recorded in w) until C X_q C^dag = +-X_q and C Z_q C^dag = +-Z_q.

    X-first: X row -> X_q (any root basis), then Z row -> Z_q with CX whose
    target is q (X_q fixed); the root's Y -> Z by SX (fixes X). Z-first is the
    mirror image (root Y -> X by SDG, which fixes Z)."""
    var n = t.n
    var rx = q
    var rz = n + q
    if not zfirst:
        if not _x_ladder(t, adj, alive, q, rx, True, w, rng, randomize):
            return False
        if _has(t.x[rz], q):
            _ap(t, w, g1(Op.SX, q))  # Y -> Z, X fixed
        return _z_ladder_tab(t, adj, alive, q, rz, False, w, rng, randomize)
    if not _z_ladder_tab(t, adj, alive, q, rz, True, w, rng, randomize):
        return False
    if _has(t.z[rx], q):
        _ap(t, w, g1(Op.SDG, q))  # Y -> X, Z fixed
    return _x_ladder(t, adj, alive, q, rx, False, w, rng, randomize)


def _rest_weight(t: SympTab, alive: List[Bool]) -> Int:
    """Sum over the alive generators of (support size - 1): a cheap proxy for
    the CX work left after the current elimination step."""
    var n = t.n
    var mask: UInt64 = 0
    for j in range(n):
        if alive[j]:
            mask |= _bit(j)
    var s = 0
    for j in range(n):
        if not alive[j]:
            continue
        s += Int(pop_count((t.x[j] | t.z[j]) & mask)) - 1
        s += Int(pop_count((t.x[n + j] | t.z[n + j]) & mask)) - 1
    return s


def _greedy_clifford(t0: SympTab, g: SubGraph, adj: _Adj, mut rng: Rng, randomize: Bool, mut out: List[Instruction]) raises -> Bool:
    """Greedy topology-aware Clifford synthesis: decouple one non-cut qubit at a
    time; pivot and elimination order (X-first / Z-first) are chosen by
    one-step simulation (CX count, then the remaining generator weight) for
    n <= 24, then signs are repaired by a local Clifford."""
    var n = t0.n
    var t = t0.copy()
    var w = List[Instruction]()
    var alive = List[Bool](length=n, fill=True)
    for _ in range(n):
        var bestq = -1
        var bestz = False
        var bestc = 1 << 60
        var nties = 0
        var best_t = SympTab(0)
        var best_w = List[Instruction]()
        var have_best = False
        for q in range(n):
            if not alive[q] or not _is_noncut(adj, alive, q):
                continue
            for zf in range(2):
                var c: Int
                if n <= 24:
                    var tt = t.copy()
                    var ww = List[Instruction]()
                    var r2 = rng.copy()
                    if not _reduce_qubit(tt, adj, alive, q, zf == 1, ww, r2, False):
                        continue
                    alive[q] = False
                    c = (ops_cx_count(ww) << 16) + _rest_weight(tt, alive)
                    alive[q] = True
                    var take = False
                    if c < bestc:
                        take = True
                        nties = 1
                    elif c == bestc and randomize:
                        nties += 1
                        take = rng.below(nties) == 0
                    if take:
                        bestc = c
                        bestq = q
                        bestz = zf == 1
                        best_t = tt^
                        best_w = ww^
                        have_best = True
                else:
                    if zf == 1:
                        continue
                    c = Int(pop_count(t.x[q] | t.z[q] | t.x[n + q] | t.z[n + q]))
                    if c < bestc:
                        bestc = c
                        bestq = q
                        nties = 1
                    elif c == bestc and randomize:
                        nties += 1
                        if rng.below(nties) == 0:
                            bestq = q
        if bestq < 0:
            return False
        if have_best:
            t = best_t^
            for o in best_w:
                w.append(o)
        elif not _reduce_qubit(t, adj, alive, bestq, bestz, w, rng, randomize):
            return False
        alive[bestq] = False
    out.clear()
    synth_local(t, out)
    var inv = inverse_ops(w)
    for o in inv:
        out.append(o)
    return True


# ----------------------------------------------------------------------------
# Clifford: optimal CX count by bidirectional BFS over local-Clifford cosets
# ----------------------------------------------------------------------------
@always_inline
def _rows_key(rows: List[Int], bits: Int) -> UInt64:
    var k: UInt64 = 0
    for i in range(len(rows)):
        k |= UInt64(rows[i]) << UInt64(bits * i)
    return k


@always_inline
def _key_rows(k: UInt64, nrows: Int, bits: Int, mut rows: List[Int]):
    var mask = (UInt64(1) << UInt64(bits)) - 1
    for i in range(nrows):
        rows[i] = Int((k >> UInt64(bits * i)) & mask)


def _local_cols(mut rows: List[Int], n: Int, j: Int, opt: Int):
    """Apply local class `opt` (0..5: I, H, S, SX, H.S, S.H) to qubit j's
    (x_j, z_j) column pair of every phase-free row (x in bits 0..n-1, z in n..2n-1)."""
    if opt == 0:
        return
    for i in range(len(rows)):
        var r = rows[i]
        var a = (r >> j) & 1
        var b = (r >> (n + j)) & 1
        var na: Int
        var nb: Int
        if opt == 1:
            na = b; nb = a
        elif opt == 2:
            na = a; nb = a ^ b
        elif opt == 3:
            na = a ^ b; nb = b
        elif opt == 4:
            na = b; nb = a ^ b
        else:
            na = a ^ b; nb = a
        r = r & ~((1 << j) | (1 << (n + j)))
        r |= (na << j) | (nb << (n + j))
        rows[i] = r


@always_inline
def _cx_rows(mut rows: List[Int], n: Int, c: Int, t: Int):
    for i in range(len(rows)):
        var r = rows[i]
        var xc = (r >> c) & 1
        var zt = (r >> (n + t)) & 1
        r ^= xc << t
        r ^= zt << (n + c)
        rows[i] = r


def _canon(mut rows: List[Int], n: Int) -> UInt64:
    """Canonical representative of the left coset {L C : L local}: each qubit's
    column pair independently takes its lexicographically smallest image."""
    var nr = len(rows)
    for j in range(n):
        var u = 0
        var v = 0
        for i in range(nr):
            u |= ((rows[i] >> j) & 1) << i
            v |= ((rows[i] >> (n + j)) & 1) << i
        var best = -1
        var bopt = 0
        for opt in range(6):
            var a: Int
            var b: Int
            if opt == 0:
                a = u; b = v
            elif opt == 1:
                a = v; b = u
            elif opt == 2:
                a = u; b = u ^ v
            elif opt == 3:
                a = u ^ v; b = v
            elif opt == 4:
                a = v; b = u ^ v
            else:
                a = u ^ v; b = u
            var key = (a << nr) | b
            if best < 0 or key < best:
                best = key
                bopt = opt
        _local_cols(rows, n, j, bopt)
    return _rows_key(rows, 2 * n)


def _tab_rows(t: SympTab) -> List[Int]:
    var n = t.n
    var rows = List[Int](length=2 * n, fill=0)
    for i in range(2 * n):
        rows[i] = Int(t.x[i]) | (Int(t.z[i]) << n)
    return rows^


def _cliff_moves(g: SubGraph) -> List[Int]:
    """Moves (edge, la, lb) encoded as e * 36 + la * 6 + lb."""
    var mv = List[Int]()
    for e in range(g.num_edges()):
        for la in range(6):
            for lb in range(6):
                mv.append(e * 36 + la * 6 + lb)
    return mv^


@always_inline
def _apply_move(mut rows: List[Int], n: Int, g: SubGraph, mv: Int):
    var e = mv // 36
    var la = (mv // 6) % 6
    var lb = mv % 6
    var a = g.edges[2 * e]
    var b = g.edges[2 * e + 1]
    _local_cols(rows, n, a, la)
    _local_cols(rows, n, b, lb)
    _cx_rows(rows, n, a, b)


def _clifford_bfs(t: SympTab, g: SubGraph, budget: Int, mut out: List[Instruction]) raises -> Bool:
    """Optimal CX count Clifford synthesis (n <= 4) by bidirectional BFS in the
    coset graph; gates are recovered by replaying the coset path on the actual
    tableau, then the remaining local Clifford (with signs) is synthesized."""
    var n = t.n
    var nr = 2 * n
    var bits = 2 * n
    var rows = _tab_rows(t)
    var k0 = _canon(rows, n)
    var idr = _tab_rows(SympTab(n))
    var kid = _canon(idr, n)
    var moves = _cliff_moves(g)
    var path = List[UInt64]()
    if k0 == kid:
        path.append(k0)
    else:
        # side 0 from C, side 1 from identity
        var keys0 = List[UInt64](); var par0 = List[Int]()
        var keys1 = List[UInt64](); var par1 = List[Int]()
        var map0 = _U64Map(10)
        var map1 = _U64Map(10)
        keys0.append(k0); par0.append(-1); map0.put(k0, 0)
        keys1.append(kid); par1.append(-1); map1.put(kid, 0)
        var lo0 = 0; var hi0 = 1
        var lo1 = 0; var hi1 = 1
        var expansions = 0
        var meet0 = -1
        var meet1 = -1
        var tmp = List[Int](length=nr, fill=0)
        var cur = List[Int](length=nr, fill=0)
        while meet0 < 0:
            if hi0 == lo0 or hi1 == lo1:
                return False
            var side = 0 if (hi0 - lo0) <= (hi1 - lo1) else 1
            if side == 0:
                for node in range(lo0, hi0):
                    expansions += 1
                    if expansions > budget:
                        return False
                    _key_rows(keys0[node], nr, bits, cur)
                    for mv in moves:
                        for i in range(nr):
                            tmp[i] = cur[i]
                        _apply_move(tmp, n, g, mv)
                        var k = _canon(tmp, n)
                        if map0.get(k) >= 0:
                            continue
                        keys0.append(k); par0.append(node); map0.put(k, len(keys0) - 1)
                        var o = map1.get(k)
                        if o >= 0 and meet0 < 0:
                            meet0 = len(keys0) - 1
                            meet1 = o
                lo0 = hi0
                hi0 = len(keys0)
            else:
                for node in range(lo1, hi1):
                    expansions += 1
                    if expansions > budget:
                        return False
                    _key_rows(keys1[node], nr, bits, cur)
                    for mv in moves:
                        for i in range(nr):
                            tmp[i] = cur[i]
                        _apply_move(tmp, n, g, mv)
                        var k = _canon(tmp, n)
                        if map1.get(k) >= 0:
                            continue
                        keys1.append(k); par1.append(node); map1.put(k, len(keys1) - 1)
                        var o = map0.get(k)
                        if o >= 0 and meet0 < 0:
                            meet0 = o
                            meet1 = len(keys1) - 1
                lo1 = hi1
                hi1 = len(keys1)
        # Both frontiers are BFS layers, so the first meeting found while
        # expanding a complete layer is a shortest path (all meeting nodes
        # of this layer have the same total length).
        var a = List[UInt64]()
        var c = meet0
        while c >= 0:
            a.append(keys0[c])
            c = par0[c]
        var i = len(a) - 1
        while i >= 0:
            path.append(a[i])
            i -= 1
        c = par1[meet1]
        while c >= 0:
            path.append(keys1[c])
            c = par1[c]
    # replay on the actual (phase-free) tableau
    var w = List[Instruction]()
    var act = _tab_rows(t)
    var tmp2 = List[Int](length=nr, fill=0)
    for step in range(len(path) - 1):
        var found = False
        for mv in moves:
            for i in range(nr):
                tmp2[i] = act[i]
            _apply_move(tmp2, n, g, mv)
            var tc = tmp2.copy()
            if _canon(tc, n) == path[step + 1]:
                var e = mv // 36
                var a2 = g.edges[2 * e]
                var b2 = g.edges[2 * e + 1]
                _word((mv // 6) % 6, a2, w)
                _word(mv % 6, b2, w)
                w.append(g2(Op.CX, a2, b2))
                for i in range(nr):
                    act[i] = tmp2[i]
                found = True
                break
        if not found:
            raise Error("QM-E619: internal error, Clifford coset path replay failed. Hint: report this as a QUMO bug.")
    var tf = t.copy()
    tf.apply_ops(w)
    out.clear()
    synth_local(tf, out)
    var inv = inverse_ops(w)
    for o in inv:
        out.append(o)
    return True


def synth_clifford(t: SympTab, g: SubGraph, method: String, budget: Int, seed: Int) raises -> List[Instruction]:
    """Clifford circuit on the edges of g with exactly tableau t (signs included)."""
    _check_width(BlockKind.CLIFFORD, g.n, method)
    var n = g.n
    var out = List[Instruction]()
    if t.n != n:
        raise Error("QM-E618: Clifford of " + String(t.n) + " qubits on a " + String(n) + "-qubit subgraph. Hint: the problem and the subgraph must have the same size.")
    if t.is_identity():
        return out^
    var b = _budget(method, budget)
    if t.is_local():
        synth_local(t, out)
        return out^
    if not g.connected():
        return out^
    if method == "exact" or method == "astar":
        if not exact_supported(BlockKind.CLIFFORD, n):
            _too_wide(BlockKind.CLIFFORD, n, method)
        if not _clifford_bfs(t, g, b, out):
            out.clear()
        return out^
    var adj = _Adj(g)
    var rng = Rng(UInt64(seed))
    var best = List[Instruction]()
    var r0 = rng.split(0)
    if not _greedy_clifford(t, g, adj, r0, False, best):
        best.clear()
    # inverse trick: synthesize C^-1 and invert
    var tinv = t.inverse()
    var trials = 1 if method == "greedy" else (6 if n <= 6 else (3 if n <= 12 else 2))
    for k in range(trials):
        var r = rng.split(k + 1)
        var cand = List[Instruction]()
        var ok: Bool
        if k % 2 == 0:
            ok = _greedy_clifford(tinv, g, adj, r, k > 0, cand)
            if ok:
                cand = inverse_ops(cand)
        else:
            ok = _greedy_clifford(t, g, adj, r, True, cand)
        if ok and (len(best) == 0 or _better(cand, best, n)):
            best = cand^
    # CX-only part: the linear synthesizer (Pauli signs repaired afterwards)
    if t.is_linear():
        var lin = synth_linear(t.linear_part(), g, "greedy" if method == "greedy" else "auto", budget, seed)
        var tl = SympTab.from_ops(lin, n)
        # signs: t = (lin then P) for a Pauli P; find P from the tableau difference
        var diff = tl.inverse().compose(t)
        if diff.is_local():
            var cand = lin.copy()
            synth_local(diff, cand)
            if len(lin) > 0 and (len(best) == 0 or _better(cand, best, n)):
                best = cand^
    if method == "auto" and n <= 3:
        var ex = List[Instruction]()
        if _clifford_bfs(t, g, min(b, 400), ex) and (len(best) == 0 or _better(ex, best, n)):
            best = ex^
    return best^


# ----------------------------------------------------------------------------
# Linear functions
# ----------------------------------------------------------------------------
@always_inline
def _lf_key(rows: List[UInt64], n: Int) -> UInt64:
    var k: UInt64 = 0
    for i in range(n):
        k |= rows[i] << UInt64(n * i)
    return k


@always_inline
def _lf_cx(k: UInt64, n: Int, c: Int, t: Int) -> UInt64:
    var mask = (UInt64(1) << UInt64(n)) - 1
    var rc = (k >> UInt64(n * c)) & mask
    return k ^ (rc << UInt64(n * t))


def _lf_moves(g: SubGraph) -> List[Int]:
    var mv = List[Int]()
    for e in range(g.num_edges()):
        mv.append(g.edges[2 * e] * 64 + g.edges[2 * e + 1])
        mv.append(g.edges[2 * e + 1] * 64 + g.edges[2 * e])
    return mv^


def _lf_bibfs(m: GF2Mat, g: SubGraph, budget: Int, mut out: List[Instruction]) -> Bool:
    """Optimal CX count by bidirectional BFS (n <= 8 keys; practical n <= 5)."""
    var n = m.n
    if n * n > 64:
        return False
    var moves = _lf_moves(g)
    var k0 = _lf_key(m.rows, n)
    var kid = _lf_key(GF2Mat.identity(n).rows, n)
    out.clear()
    if k0 == kid:
        return True
    var keys0 = List[UInt64](); var par0 = List[Int](); var mv0 = List[Int]()
    var keys1 = List[UInt64](); var par1 = List[Int](); var mv1 = List[Int]()
    var map0 = _U64Map(12)
    var map1 = _U64Map(12)
    keys0.append(k0); par0.append(-1); mv0.append(-1); map0.put(k0, 0)
    keys1.append(kid); par1.append(-1); mv1.append(-1); map1.put(kid, 0)
    var lo0 = 0; var hi0 = 1
    var lo1 = 0; var hi1 = 1
    var expansions = 0
    var meet0 = -1
    var meet1 = -1
    while meet0 < 0:
        if hi0 == lo0 or hi1 == lo1:
            return False
        var side = 0 if (hi0 - lo0) <= (hi1 - lo1) else 1
        if side == 0:
            for node in range(lo0, hi0):
                expansions += 1
                if expansions > budget:
                    return False
                var key = keys0[node]
                for mv in moves:
                    var k = _lf_cx(key, n, mv // 64, mv % 64)
                    if map0.get(k) >= 0:
                        continue
                    keys0.append(k); par0.append(node); mv0.append(mv); map0.put(k, len(keys0) - 1)
                    var o = map1.get(k)
                    if o >= 0 and meet0 < 0:
                        meet0 = len(keys0) - 1
                        meet1 = o
            lo0 = hi0
            hi0 = len(keys0)
        else:
            for node in range(lo1, hi1):
                expansions += 1
                if expansions > budget:
                    return False
                var key = keys1[node]
                for mv in moves:
                    var k = _lf_cx(key, n, mv // 64, mv % 64)
                    if map1.get(k) >= 0:
                        continue
                    keys1.append(k); par1.append(node); mv1.append(mv); map1.put(k, len(keys1) - 1)
                    var o = map0.get(k)
                    if o >= 0 and meet0 < 0:
                        meet0 = o
                        meet1 = len(keys1) - 1
            lo1 = hi1
            hi1 = len(keys1)
    # forward moves f1..fk (M -> S), backward b1..bl (I -> S): circuit = b1..bl, fk..f1
    var fwd = List[Int]()
    var c = meet0
    while par0[c] >= 0:
        fwd.append(mv0[c])  # collected fk, ..., f1
        c = par0[c]
    var bwd = List[Int]()
    c = meet1
    while par1[c] >= 0:
        bwd.append(mv1[c])  # collected bl, ..., b1
        c = par1[c]
    var i = len(bwd) - 1
    while i >= 0:
        out.append(g2(Op.CX, bwd[i] // 64, bwd[i] % 64))
        i -= 1
    for mv in fwd:
        out.append(g2(Op.CX, mv // 64, mv % 64))
    return True


def _lf_h(k: UInt64, n: Int) -> Int:
    var mask = (UInt64(1) << UInt64(n)) - 1
    var h = 0
    for i in range(n):
        if ((k >> UInt64(n * i)) & mask) != _bit(i):
            h += 1
    return h


def _lf_astar(m: GF2Mat, g: SubGraph, budget: Int, mut out: List[Instruction]) -> Bool:
    """A* from M to I; h = rows of M differing from the identity (consistent)."""
    var n = m.n
    if n * n > 64:
        return False
    var moves = _lf_moves(g)
    var k0 = _lf_key(m.rows, n)
    var keys = List[UInt64](); var par = List[Int](); var mvs = List[Int](); var gc = List[Int]()
    var best = _U64Map(12)
    var heap = _Heap()
    keys.append(k0); par.append(-1); mvs.append(-1); gc.append(0)
    best.put(k0, 0)
    heap.push(_lf_h(k0, n) * 4096, 0)
    var expansions = 0
    var goal = -1
    while len(heap) > 0:
        var node = heap.pop()
        var gn = gc[node]
        var key = keys[node]
        if best.get(key) < gn:
            continue
        var h = _lf_h(key, n)
        if h == 0:
            goal = node
            break
        expansions += 1
        if expansions > budget:
            return False
        for mv in moves:
            var k = _lf_cx(key, n, mv // 64, mv % 64)
            var ng = gn + 1
            var old = best.get(k)
            if old >= 0 and old <= ng:
                continue
            best.put(k, ng)
            keys.append(k); par.append(node); mvs.append(mv); gc.append(ng)
            heap.push((ng + _lf_h(k, n)) * 4096 + (4095 - min(ng, 4095)), len(keys) - 1)
    if goal < 0:
        return False
    # moves f1..fk with E_fk..E_f1 M = I  ->  circuit fk..f1
    out.clear()
    var c = goal
    while par[c] >= 0:
        out.append(g2(Op.CX, mvs[c] // 64, mvs[c] % 64))
        c = par[c]
    return True


def _lf_greedy(m: GF2Mat, g: SubGraph, adj: _Adj, mut rng: Rng, randomize: Bool, mut out: List[Instruction]) raises -> Bool:
    """Steiner-Gauss / RowCol elimination: the greedy Clifford elimination on
    the tableau of a linear map only ever emits CX (no 1q gates, no signs)."""
    var t = SympTab.from_linear(m)
    return _greedy_clifford(t, g, adj, rng, randomize, out)


def synth_linear(m: GF2Mat, g: SubGraph, method: String, budget: Int, seed: Int) raises -> List[Instruction]:
    """CX circuit on the edges of g realizing x -> M x."""
    _check_width(BlockKind.LINEAR, g.n, method)
    var n = g.n
    var out = List[Instruction]()
    if m.n != n:
        raise Error("QM-E618: linear function of " + String(m.n) + " qubits on a " + String(n) + "-qubit subgraph. Hint: the problem and the subgraph must have the same size.")
    if m.is_identity():
        return out^
    if not g.connected():
        return out^
    var b = _budget(method, budget)
    if method == "exact":
        if not exact_supported(BlockKind.LINEAR, n):
            _too_wide(BlockKind.LINEAR, n, method)
        if not _lf_bibfs(m, g, b, out):
            out.clear()
        return out^
    if method == "astar":
        if not _astar_supported(BlockKind.LINEAR, n):
            _too_wide(BlockKind.LINEAR, n, method)
        if not _lf_astar(m, g, b, out):
            out.clear()
        return out^
    var adj = _Adj(g)
    var rng = Rng(UInt64(seed))
    var best = List[Instruction]()
    var r0 = rng.split(0)
    if not _lf_greedy(m, g, adj, r0, False, best):
        best.clear()
    var mt = m.transpose()
    var minv = m.inverse()
    var trials = 2 if method == "greedy" else (8 if n <= 6 else (5 if n <= 12 else 3))
    for k in range(trials):
        var r = rng.split(k + 1)
        var cand = List[Instruction]()
        var rnd = k >= 3
        var which = k % 3
        var ok: Bool
        if which == 0:
            ok = _lf_greedy(mt, g, adj, r, rnd, cand)
            if ok:
                cand = _reverse_flip_cx(cand)
        elif which == 1:
            ok = _lf_greedy(minv, g, adj, r, rnd, cand)
            if ok:
                cand = inverse_ops(cand)
        else:
            ok = _lf_greedy(m, g, adj, r, True, cand)
        if ok and (len(best) == 0 or _better(cand, best, n)):
            best = cand^
    if method == "auto" and n <= 5:
        # n <= 4: the full space is small (|GL(4,2)| = 20160); n = 5: a small
        # budget keeps auto near 1 ms (deep instances fall back to greedy).
        var ex = List[Instruction]()
        var bb = b if n <= 4 else min(b, 3000)
        if _lf_bibfs(m, g, bb, ex) and len(ex) > 0 and (len(best) == 0 or _better(ex, best, n)):
            best = ex^
    return best^


# ----------------------------------------------------------------------------
# Pauli networks
# ----------------------------------------------------------------------------
def _conj_row(mut x: UInt64, mut z: UInt64, mut r: UInt8, gt: Instruction) raises:
    var k = gt.kind
    if k == Op.H:
        var a = gt.q0
        var xa = (x >> UInt64(a)) & 1
        var za = (z >> UInt64(a)) & 1
        r ^= UInt8(xa & za)
        x = (x & ~_bit(a)) | (za << UInt64(a))
        z = (z & ~_bit(a)) | (xa << UInt64(a))
    elif k == Op.S:
        var a = gt.q0
        var xa = (x >> UInt64(a)) & 1
        var za = (z >> UInt64(a)) & 1
        r ^= UInt8(xa & za)
        z ^= xa << UInt64(a)
    elif k == Op.SX:
        var a = gt.q0
        var xa = (x >> UInt64(a)) & 1
        var za = (z >> UInt64(a)) & 1
        r ^= UInt8(za & (xa ^ 1))
        x ^= za << UInt64(a)
    elif k == Op.CX:
        var c = gt.q0
        var tt = gt.q1
        var xc = (x >> UInt64(c)) & 1
        var zc = (z >> UInt64(c)) & 1
        var xt = (x >> UInt64(tt)) & 1
        var zt = (z >> UInt64(tt)) & 1
        r ^= UInt8(xc & zt & (xt ^ zc ^ 1))
        x ^= xc << UInt64(tt)
        z ^= zt << UInt64(c)
    else:
        raise Error("QM-E619: internal error, unexpected gate in Pauli-network greedy. Hint: report this as a QUMO bug.")


def _to_z_words(xb: Bool, zb: Bool, mut w0: Int, mut w1: Int):
    """The two local classes mapping the qubit's Pauli to Z (phase-free)."""
    if zb and not xb:
        w0 = 0; w1 = 2  # I, S
    elif xb and not zb:
        w0 = 1; w1 = 4  # H, H.S
    else:
        w0 = 5; w1 = 3  # S.H, SX


struct _PNState(Copyable):
    var x: List[UInt64]
    var z: List[UInt64]
    var r: List[UInt8]

    def __init__(out self, m: Int):
        self.x = List[UInt64](length=m, fill=0)
        self.z = List[UInt64](length=m, fill=0)
        self.r = List[UInt8](length=m, fill=0)

    def __init__(out self, *, copy: Self):
        self.x = copy.x.copy()
        self.z = copy.z.copy()
        self.r = copy.r.copy()


def _pn_greedy(pn: PauliNet, g: SubGraph, adj: _Adj, src_ops: List[Instruction], mut rng: Rng, randomize: Bool, cliff_method: String, budget: Int, seed: Int, mut out: List[Instruction]) raises -> Bool:
    var n = pn.n
    var m = len(pn.rots)
    var st = _PNState(m)
    for k in range(m):
        st.x[k] = pn.rots[k].x
        st.z[k] = pn.rots[k].z
    var done = List[Bool](length=m, fill=False)
    var anti = List[Bool](length=m * m, fill=False)
    for j in range(m):
        for k in range(m):
            anti[j * m + k] = paulis_anticommute(pn.rots[j].x, pn.rots[j].z, pn.rots[k].x, pn.rots[k].z)
    var v = SympTab(n)
    out.clear()
    var remaining = m
    var target = -1
    var guard = 0
    var guard_cap = 64 * (m + 1) * (n + 1) + 64
    while True:
        # emit every front-layer rotation of weight 1 (fixpoint)
        var progress = True
        while progress:
            progress = False
            for k in range(m):
                if done[k]:
                    continue
                var front = True
                for j in range(k):
                    if not done[j] and anti[j * m + k]:
                        front = False
                        break
                if not front:
                    continue
                var sup = st.x[k] | st.z[k]
                if pop_count(sup) == 1:
                    var q = 0
                    while not _has(sup, q):
                        q += 1
                    var xb = _has(st.x[k], q)
                    var zb = _has(st.z[k], q)
                    var axis = 1 if (xb and zb) else (0 if xb else 2)
                    out.append(emit_rotation(pn.rots[k], src_ops, q, axis, -1 if st.r[k] != 0 else 1))
                    done[k] = True
                    remaining -= 1
                    progress = True
                    if k == target:
                        target = -1
        if remaining == 0:
            break
        guard += 1
        if guard > guard_cap:
            return False
        # choose (or keep) the target: front rotation with the smallest Steiner tree
        if target < 0:
            var bestc = 1 << 30
            var nties = 0
            for k in range(m):
                if done[k]:
                    continue
                var front = True
                for j in range(k):
                    if not done[j] and anti[j * m + k]:
                        front = False
                        break
                if not front:
                    continue
                var terms = List[Int]()
                var sup = st.x[k] | st.z[k]
                for q in range(n):
                    if _has(sup, q):
                        terms.append(q)
                var alive = List[Bool](length=n, fill=True)
                var tr = _steiner(adj, alive, terms, terms[0], rng, False)
                if tr.size < 0:
                    return False
                if tr.size < bestc:
                    bestc = tr.size
                    target = k
                    nties = 1
                elif tr.size == bestc and randomize:
                    nties += 1
                    if rng.below(nties) == 0:
                        target = k
        # candidate moves for the target
        var tx = st.x[target]
        var tz = st.z[target]
        var sup = tx | tz
        var cands = List[List[Instruction]]()
        for e in range(g.num_edges()):
            var a = g.edges[2 * e]
            var b = g.edges[2 * e + 1]
            if not (_has(sup, a) and _has(sup, b)):
                continue
            var a0 = 0; var a1 = 0; var b0 = 0; var b1 = 0
            _to_z_words(_has(tx, a), _has(tz, a), a0, a1)
            _to_z_words(_has(tx, b), _has(tz, b), b0, b1)
            for wa in [a0, a1]:
                for wb in [b0, b1]:
                    for dirn in range(2):
                        var c = List[Instruction]()
                        _word(wa, a, c)
                        _word(wb, b, c)
                        if dirn == 0:
                            c.append(g2(Op.CX, a, b))
                        else:
                            c.append(g2(Op.CX, b, a))
                        cands.append(c^)
        if len(cands) == 0:
            # move a leaf of the target's Steiner tree one step towards the tree
            var terms = List[Int]()
            for q in range(n):
                if _has(sup, q):
                    terms.append(q)
            var alive = List[Bool](length=n, fill=True)
            var tr = _steiner(adj, alive, terms, terms[0], rng, False)
            if tr.size < 0:
                return False
            for u in tr.order:
                var p = tr.parent[u]
                if p < 0 or not _has(sup, u) or _has(sup, p):
                    continue
                var u0 = 0; var u1 = 0
                _to_z_words(_has(tx, u), _has(tz, u), u0, u1)
                for wu in [u0, u1]:
                    var c = List[Instruction]()
                    _word(wu, u, c)
                    c.append(g2(Op.CX, p, u))  # Z_u -> Z_p Z_u
                    c.append(g2(Op.CX, u, p))  # Z_p Z_u -> Z_p
                    cands.append(c^)
        if len(cands) == 0:
            return False
        # score: weights of the pending rotations after the move (front x3)
        var bests = 1 << 30
        var pick = 0
        var nt = 0
        for ci in range(len(cands)):
            var s = 0
            for k in range(m):
                if done[k]:
                    continue
                var x = st.x[k]
                var z = st.z[k]
                var r = st.r[k]
                for gt in cands[ci]:
                    _conj_row(x, z, r, gt)
                var front = True
                for j in range(k):
                    if not done[j] and anti[j * m + k]:
                        front = False
                        break
                var wgt = Int(pop_count(x | z))
                s += 3 * wgt if front else wgt
            if s < bests:
                bests = s
                pick = ci
                nt = 1
            elif s == bests and randomize:
                nt += 1
                if rng.below(nt) == 0:
                    pick = ci
        for gt in cands[pick]:
            out.append(gt)
            v.apply(gt)
            for k in range(m):
                if not done[k]:
                    _conj_row(st.x[k], st.z[k], st.r[k], gt)
    # residual Clifford: final . V^-1
    var res = v.inverse().compose(pn.final)
    if not res.is_identity():
        var cl = synth_clifford(res, g, cliff_method, budget, seed)
        if len(cl) == 0 and cliff_method != "greedy":
            cl = synth_clifford(res, g, "greedy", budget, seed)
        if len(cl) == 0:
            return False
        for o in cl:
            out.append(o)
    return True


def _inverse_pair(a: Instruction, b: Instruction) -> Bool:
    """b undoes a exactly (same operands)."""
    var ka = a.kind
    var kb = b.kind
    if op_arity(ka) == 1:
        if a.q0 != b.q0:
            return False
        if (ka == Op.H or ka == Op.X or ka == Op.Y or ka == Op.Z) and kb == ka:
            return True
        if (ka == Op.S and kb == Op.SDG) or (ka == Op.SDG and kb == Op.S):
            return True
        if (ka == Op.SX and kb == Op.SXDG) or (ka == Op.SXDG and kb == Op.SX):
            return True
        return False
    if ka != kb:
        return False
    if ka == Op.CX:
        return a.q0 == b.q0 and a.q1 == b.q1
    if ka == Op.CZ or ka == Op.SWAP:
        return (a.q0 == b.q0 and a.q1 == b.q1) or (a.q0 == b.q1 and a.q1 == b.q0)
    return False


def peephole_cancel(ops: List[Instruction]) -> List[Instruction]:
    """Remove adjacent (per wire) inverse pairs of Clifford gates; cascades."""
    var out = List[Instruction](capacity=len(ops))
    for o in ops:
        var ar = op_arity(o.kind)
        var j = len(out) - 1
        var hit = -1
        while j >= 0:
            var p = out[j]
            if p.acts_on(o.q0) or (ar == 2 and p.acts_on(o.q1)):
                hit = j
                break
            j -= 1
        if hit >= 0 and op_arity(out[hit].kind) == ar and _inverse_pair(out[hit], o):
            # for 2q gates the hit must also be the last gate on the other wire
            var ok = True
            if ar == 2:
                var k = len(out) - 1
                while k > hit:
                    if out[k].acts_on(o.q0) or out[k].acts_on(o.q1):
                        ok = False
                        break
                    k -= 1
            if ok:
                _ = out.pop(hit)
                continue
        out.append(o)
    return out^


def _z_ladder(x0: UInt64, z0: UInt64, adj: _Adj, n: Int, root: Int, mut rng: Rng, mut pre: List[Instruction], mut sign: Int) raises -> Bool:
    """Clifford L (appended to pre, on edges) with L P L^dag = sign * Z_root."""
    var x = x0
    var z = z0
    var r: UInt8 = 0
    for j in range(n):
        var xb = _has(x, j)
        var zb = _has(z, j)
        if xb and zb:
            var gt = g1(Op.SX, j)
            _conj_row(x, z, r, gt)
            pre.append(gt)
        elif xb:
            var gt = g1(Op.H, j)
            _conj_row(x, z, r, gt)
            pre.append(gt)
    var terms = List[Int]()
    for j in range(n):
        if _has(z, j):
            terms.append(j)
    var alive = List[Bool](length=n, fill=True)
    var tr = _steiner(adj, alive, terms, root, rng, False)
    if tr.size < 0:
        return False
    for v in tr.order:
        if not _has(z, v):
            for u in tr.order:
                if tr.parent[u] == v and _has(z, u):
                    var gt = g2(Op.CX, v, u)
                    _conj_row(x, z, r, gt)
                    pre.append(gt)
                    break
    for u in tr.order:
        if tr.parent[u] >= 0:
            var gt = g2(Op.CX, u, tr.parent[u])
            _conj_row(x, z, r, gt)
            pre.append(gt)
    if x != 0 or z != _bit(root):
        raise Error("QM-E619: internal error, Pauli ladder did not reach Z_root. Hint: report this as a QUMO bug.")
    sign = -1 if r != 0 else 1
    return True


def _pn_gadgets(pn: PauliNet, g: SubGraph, adj: _Adj, src_ops: List[Instruction], mut rng: Rng, randomize: Bool, cliff_method: String, budget: Int, seed: Int, mut out: List[Instruction]) raises -> Bool:
    """One topology-aware gadget per rotation (basis change + Steiner CX ladder
    + RZ + uncompute), next rotation = front-layer rotation sharing the most
    single-qubit Paulis with the previous one, then the final Clifford; mirrored
    ladders of consecutive gadgets cancel in `peephole_cancel`."""
    var n = pn.n
    var m = len(pn.rots)
    var done = List[Bool](length=m, fill=False)
    var anti = List[Bool](length=m * m, fill=False)
    for j in range(m):
        for k in range(m):
            anti[j * m + k] = paulis_anticommute(pn.rots[j].x, pn.rots[j].z, pn.rots[k].x, pn.rots[k].z)
    var px: UInt64 = 0
    var pz: UInt64 = 0
    var proot = -1
    var raw = List[Instruction]()
    for _ in range(m):
        var pick = -1
        var bs = -1
        var nt = 0
        for k in range(m):
            if done[k]:
                continue
            var front = True
            for j in range(k):
                if not done[j] and anti[j * m + k]:
                    front = False
                    break
            if not front:
                continue
            var rx = pn.rots[k].x
            var rz = pn.rots[k].z
            var same = ~((rx ^ px) | (rz ^ pz)) & (rx | rz)
            var sc = Int(pop_count(same))
            if sc > bs:
                bs = sc
                pick = k
                nt = 1
            elif sc == bs and randomize:
                nt += 1
                if rng.below(nt) == 0:
                    pick = k
        var rot = pn.rots[pick]
        var sup = rot.x | rot.z
        var root = -1
        if proot >= 0 and _has(sup, proot):
            root = proot
        else:
            var bd = 1 << 30
            for q in range(n):
                if not _has(sup, q):
                    continue
                var d = 0
                for u in range(n):
                    if _has(sup, u):
                        d += g.dist[q * n + u]
                if d < bd:
                    bd = d
                    root = q
        var pre = List[Instruction]()
        var sign = 1
        if not _z_ladder(rot.x, rot.z, adj, n, root, rng, pre, sign):
            return False
        for o in pre:
            raw.append(o)
        raw.append(emit_rotation(rot, src_ops, root, 2, sign))
        for o in inverse_ops(pre):
            raw.append(o)
        done[pick] = True
        px = rot.x
        pz = rot.z
        proot = root
    if not pn.final.is_identity():
        var cl = synth_clifford(pn.final, g, cliff_method, budget, seed)
        if len(cl) == 0 and cliff_method != "greedy":
            cl = synth_clifford(pn.final, g, "greedy", budget, seed)
        if len(cl) == 0:
            return False
        for o in cl:
            raw.append(o)
    out = peephole_cancel(raw)
    return True


def synth_pauli_network(pn: PauliNet, g: SubGraph, method: String, budget: Int, seed: Int, src_ops: List[Instruction]) raises -> List[Instruction]:
    """Pauli-rotation network on the edges of g: the rotations (angles copied
    from src_ops) in a commutation-respecting order, then the residual
    Clifford. Equal to the block up to global phase. exact / astar apply
    the respective method to the residual Clifford only."""
    _check_method(method)
    var n = g.n
    var out = List[Instruction]()
    if pn.n != n:
        raise Error("QM-E618: Pauli network of " + String(pn.n) + " qubits on a " + String(n) + "-qubit subgraph. Hint: the problem and the subgraph must have the same size.")
    if len(pn.rots) == 0:
        return synth_clifford(pn.final, g, "greedy" if method == "greedy" else "auto", budget, seed)
    if not g.connected():
        return out^
    var cm = method
    if (cm == "exact" or cm == "astar") and not exact_supported(BlockKind.CLIFFORD, n):
        cm = "auto"
    var adj = _Adj(g)
    var rng = Rng(UInt64(seed))
    var r0 = rng.split(0)
    var best = List[Instruction]()
    if not _pn_greedy(pn, g, adj, src_ops, r0, False, cm, budget, seed, best):
        best.clear()
    var rg = rng.split(100)
    var gad = List[Instruction]()
    if _pn_gadgets(pn, g, adj, src_ops, rg, False, cm, budget, seed, gad):
        if len(best) == 0 or _better(gad, best, n):
            best = gad^
    var trials = 0 if (method == "greedy" or n > 9) else 3
    for k in range(trials):
        var r = rng.split(k + 1)
        var cand = List[Instruction]()
        if _pn_greedy(pn, g, adj, src_ops, r, True, cm, budget, seed + k + 1, cand):
            if len(best) == 0 or _better(cand, best, n):
                best = cand^
        var r2 = rng.split(k + 101)
        var cand2 = List[Instruction]()
        if _pn_gadgets(pn, g, adj, src_ops, r2, True, cm, budget, seed + k + 1, cand2):
            if len(best) == 0 or _better(cand2, best, n):
                best = cand2^
    return best^
