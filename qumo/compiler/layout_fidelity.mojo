"""Noise-aware passes: `post_layout` (calibration-driven placement).

`post_layout` runs after routing and basis translation, on physical qubits. The
gates stay the same, only their qubits are renamed, so every embedding of the
circuit's interaction graph into the coupling graph is valid and the search can
simply take the one with the best estimated fidelity. The estimate is the one of
`Target.estimate_fidelity`, in negative log form so it adds up over gates:

    cost(mapping) = sum over gates and measurements of -log(1 - error)

Branch and bound over the used qubits (Qiskit's VF2PostLayout idea, with its
two-sided `call_limit`): qubits are visited so each one touches an assigned one,
candidates in order of cost, and a branch stops when its partial cost plus an
optimistic rest (every remaining gate at the best error of the device) reaches
the incumbent. The incumbent starts at the identity, so the pass never lowers the
estimate, and a uniform target has nothing to gain and keeps its output exactly.
"""

from std.math import log
from qumo.ir.gates import Op, op_arity, op_is_unitary
from qumo.ir.circuit import Circuit
from qumo.compiler.target import Target


def _log_keep(e: Float64) -> Float64:
    return -log(max(1.0 - e, 1e-12))


struct _Search(Movable):
    var nv: Int  # used physical qubits, in visit order
    var order: List[Int]  # visit index -> old physical qubit
    var nbr: List[List[Int]]  # per visit index: (earlier visit index, gate count) pairs
    var node_cost: List[List[Float64]]  # [visit index][new physical] = 1q + readout cost
    var edge_count: List[Int]  # per visit index, gates towards earlier indices
    var best_rest: List[Float64]  # optimistic cost of visit indices >= i
    var edge_log: List[Float64]  # N*N, inf-like when not coupled
    var coupled: List[Bool]  # N*N
    var N: Int
    var best: List[Int]
    var best_cost: Float64
    var calls: Int
    var limit: Int
    var cur: List[Int]
    var used: List[Bool]

    def __init__(out self, N: Int, limit: Int):
        self.nv = 0
        self.order = List[Int]()
        self.nbr = List[List[Int]]()
        self.node_cost = List[List[Float64]]()
        self.edge_count = List[Int]()
        self.best_rest = List[Float64]()
        self.edge_log = List[Float64]()
        self.coupled = List[Bool]()
        self.N = N
        self.best = List[Int]()
        self.best_cost = 1e300
        self.calls = 0
        self.limit = limit
        self.cur = List[Int]()
        self.used = List[Bool](length=N, fill=False)

    def dfs(mut self, i: Int, partial: Float64):
        if self.calls >= self.limit:
            return
        self.calls += 1
        if i == self.nv:
            if partial < self.best_cost - 1e-12:
                self.best_cost = partial
                self.best = self.cur.copy()
            return
        # candidates: free, coupled to every placed neighbour
        var cand = List[Int]()
        var cost = List[Float64]()
        var n = len(self.nbr[i]) // 2
        for p in range(self.N):
            if self.used[p]:
                continue
            var c = self.node_cost[i][p]
            var ok = True
            for j in range(n):
                var r = self.cur[self.nbr[i][2 * j]]
                if not self.coupled[p * self.N + r]:
                    ok = False
                    break
                c += Float64(self.nbr[i][2 * j + 1]) * self.edge_log[p * self.N + r]
            if ok and partial + c + self.best_rest[i + 1] < self.best_cost - 1e-12:
                var k = len(cand)
                cand.append(p)
                cost.append(c)
                # insertion sort: cheapest first
                while k > 0 and cost[k - 1] > cost[k]:
                    var a = cost[k - 1]
                    cost[k - 1] = cost[k]
                    cost[k] = a
                    var b = cand[k - 1]
                    cand[k - 1] = cand[k]
                    cand[k] = b
                    k -= 1
        for k in range(len(cand)):
            var p = cand[k]
            if partial + cost[k] + self.best_rest[i + 1] >= self.best_cost - 1e-12:
                break
            self.used[p] = True
            self.cur[i] = p
            self.dfs(i + 1, partial + cost[k])
            self.used[p] = False


def post_layout(mut c: Circuit, target: Target, mut wire_map: List[Int], call_limit: Int = 200000) raises -> Float64:
    """Rename the physical qubits of `c` to the best embedding found.

    `wire_map[old] = new` on return. Returns the gain in estimated log fidelity
    (0.0 when the identity stays, which leaves `c` untouched)."""
    var N = target.num_qubits
    wire_map = List[Int](capacity=N)
    for q in range(N):
        wire_map.append(q)
    if c.num_qubits != N:
        return 0.0
    # gate counts per used qubit and pair
    var n1 = List[Float64](length=N, fill=0.0)  # summed per-gate costs are position dependent: count gates
    var n_meas = List[Int](length=N, fill=0)
    var pairs = List[Int](length=N * N, fill=0)
    var touched = List[Bool](length=N, fill=False)
    for g in c.ops:
        if g.kind == Op.MEASURE:
            n_meas[g.q0] += 1
            touched[g.q0] = True
            continue
        if not op_is_unitary(g.kind) or g.kind == Op.I:
            if g.kind == Op.RESET:
                touched[g.q0] = True
            continue
        var a = op_arity(g.kind)
        if a > 2:
            return 0.0
        touched[g.q0] = True
        if a == 2:
            touched[g.q1] = True
            pairs[g.q0 * N + g.q1] += 1
            pairs[g.q1 * N + g.q0] += 1
        elif target.duration_ns[g.kind] > 0.0:
            n1[g.q0] += 1.0
    var s = _Search(N, call_limit)
    # visit order: BFS by interaction, highest degree first
    var deg = List[Int](length=N, fill=0)
    for a in range(N):
        for b in range(N):
            deg[a] += pairs[a * N + b]
    var seen = List[Bool](length=N, fill=False)
    while True:
        var start = -1
        for q in range(N):
            if touched[q] and not seen[q] and (start < 0 or deg[q] > deg[start]):
                start = q
        if start < 0:
            break
        seen[start] = True
        s.order.append(start)
        var head = len(s.order) - 1
        while head < len(s.order):
            var u = s.order[head]
            head += 1
            for v in range(N):
                if pairs[u * N + v] > 0 and not seen[v]:
                    seen[v] = True
                    s.order.append(v)
    s.nv = len(s.order)
    if s.nv == 0:
        return 0.0
    # costs
    s.edge_log = List[Float64](length=N * N, fill=0.0)
    s.coupled = List[Bool](length=N * N, fill=False)
    var best_edge = 1e300
    if target.full:
        var e = _log_keep(target.default_error_2q)
        best_edge = e
        for i in range(N * N):
            s.edge_log[i] = e
            s.coupled[i] = True
    else:
        for i in range(target.num_edges()):
            var a = target.edges[2 * i]
            var b = target.edges[2 * i + 1]
            var e = _log_keep(target.error_2q[i])
            best_edge = min(best_edge, e)
            s.edge_log[a * N + b] = e
            s.edge_log[b * N + a] = e
            s.coupled[a * N + b] = True
            s.coupled[b * N + a] = True
    var visit = List[Int](length=N, fill=-1)
    for i in range(s.nv):
        visit[s.order[i]] = i
    s.best_rest = List[Float64](length=s.nv + 2, fill=0.0)
    var identity = 0.0
    for i in range(s.nv):
        var old = s.order[i]
        var row = List[Float64](capacity=N)
        var best_node = 1e300
        for p in range(N):
            var v = n1[old] * _log_keep(target.error_1q[p]) + Float64(n_meas[old]) * _log_keep(target.readout_error[p])
            row.append(v)
            best_node = min(best_node, v)
        identity += row[old]
        s.node_cost.append(row^)
        var nb = List[Int]()
        var cnt = 0
        for j in range(i):
            var other = s.order[j]
            var count = pairs[old * N + other]
            if count > 0:
                nb.append(j)
                nb.append(count)
                cnt += count
                identity += Float64(count) * s.edge_log[old * N + other]
        s.nbr.append(nb^)
        s.edge_count.append(cnt)
        s.best_rest[i] = best_node + Float64(cnt) * best_edge
    var i = s.nv - 1
    while i >= 0:
        s.best_rest[i] += s.best_rest[i + 1]
        i -= 1
    s.best_cost = identity
    s.cur = List[Int](length=s.nv, fill=-1)
    s.dfs(0, 0.0)
    if len(s.best) == 0:
        return 0.0
    # permutation of all physical qubits: unused ones fill the free slots
    var taken = List[Bool](length=N, fill=False)
    var assigned = List[Int](length=N, fill=-1)
    for i2 in range(s.nv):
        assigned[s.order[i2]] = s.best[i2]
        taken[s.best[i2]] = True
    var free = 0
    for q in range(N):
        if assigned[q] < 0:
            while taken[free]:
                free += 1
            assigned[q] = free
            taken[free] = True
    for ref g in c.ops:
        var a = op_arity(g.kind)
        if a >= 1 and g.q0 >= 0:
            g.q0 = assigned[g.q0]
        if a >= 2:
            g.q1 = assigned[g.q1]
    wire_map = assigned^
    return identity - s.best_cost
