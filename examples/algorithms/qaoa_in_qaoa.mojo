"""QAOA-in-QAOA (QAOA^2): divide-and-conquer MaxCut on a 40-node random graph, on QUMO.

Reproduces: Classiq library - Hybrid Classical-Quantum Simulation of MaxCut Using QAOA-in-QAOA
Framework: Classiq - https://github.com/Classiq/classiq-library
Source: https://github.com/Classiq/classiq-library/blob/main/applications/optimization/qaoa_in_qaoa/qaoa_in_qaoa.ipynb (Classiq Library, MIT)

The notebook splits a G(40, 0.2) random graph into communities of at most 6 nodes, solves MaxCut on each community with a depth-2 QAOA (6 qubits, 20 optimiser iterations), builds a weighted community graph whose edge weight is (+w if the endpoints currently share a side, -w otherwise), solves MaxCut on that graph the same way (recursing while it has more than 6 nodes), and flips whole communities to merge the partial solutions. This file runs the same pipeline with QUMO state-vector QAOA. The graph is the notebook's own: Python's Mersenne Twister (random.seed(246)) is reimplemented here and draws the same 158 edges the notebook prints. Checks: the 158 edge count; the communities partition the 40 nodes with at most 6 per community; for every QAOA instance the optimised expected cut equals a QUMO-independent dense state-vector simulation of the same ansatz with the true signed weights (tolerance 1e-9, so a cost layer that ignores or mis-signs weights fails), beats the uniform-state value (sum of weights / 2), and the best of 1000 sampled bitstrings equals the brute-force optimum of that instance (a weak check: sampling alone would usually meet it); the notebook's printed 10 communities; at every merge level the change in the cut of the full graph equals the cut of the community graph under the merge solution (an identity computed on different graphs); the final cut lies between the random-partition expectation |E|/2 and |E| and is not below the cut before merging. These last bounds are sanity checks only: the notebook prints no cut value, so the final cut is not compared with a reference, and a merge identity check covers the bookkeeping, not QAOA quality (that is covered by the per-instance independent-simulation check). The final cut is reported next to the notebook's greedy one-exchange baseline.

Deviations from the notebook:
- The notebook prints no cut values and no per-community sizes beyond "10 communities, first of size 6", so those are the only printed references; the checks above are independent ones.
- Communities come from a QUMO-side greedy modularity merge (Clauset-Newman-Moore style: merge the pair with the largest modularity gain while it is positive, ties to the lowest index) instead of networkx greedy_modularity_communities, whose tie breaking differs. Communities larger than 6 nodes are split recursively; if the recursion cannot split one, it is cut into chunks of 6 in node order (the notebook splits randomly after 100 levels). On the signed community graphs the modularity uses absolute weights. The community count matches the notebook (10) but order and sizes differ (the notebook's first community has 6 nodes, here 4).
- The optimiser is QUMO's Nelder-Mead (20 iterations, initial step 0.5) on exact state-vector expectations; the notebook uses Classiq's optimiser with 20 iterations on sampled estimates. The cost layer is one RZZ per weighted edge (the notebook's phase up to a global phase) and the mixer is RX on every qubit. Depth 2, 6 qubits as in the notebook.
- The solution of each instance is the sampled bitstring with the highest cut (ties by count), from 1000 shots with a fixed seed, as the notebook sorts by cost then count. An edgeless community gets all zeros instead of a random assignment (no edge, so the same cut).
- The notebook's Goemans-Williamson comparison is commented out in the code and is not reproduced; its random-partition baseline is replaced by the exact expectation |E|/2, and its one-exchange baseline is a local search written here (start all zeros, flip any node that raises the cut).

Run: .venv/bin/mojo run -I . examples/frameworks/classiq/qaoa_in_qaoa.mojo
example-meta: {"framework": "classiq", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from std.math import cos, sin, sqrt
from qumo.ir.circuit import Circuit
from qumo.sim.runner import evolve, sample
from qumo.algorithms.library import bitstring_to_int
from qumo.algorithms.optimizers import NelderMead, Objective
from qumo.testing.checks import CheckLog


# ---------------------------------------------------------------------------
# CPython-compatible Mersenne Twister: random.seed(int) and random.random()
# ---------------------------------------------------------------------------
struct PyRandom(Movable):
    var mt: List[UInt64]
    var idx: Int

    def __init__(out self, seed: Int):
        self.mt = List[UInt64](length=624, fill=0)
        self.idx = 624
        var mask = UInt64(0xFFFFFFFF)
        self.mt[0] = UInt64(19650218)
        for i in range(1, 624):
            var p = self.mt[i - 1]
            self.mt[i] = (UInt64(1812433253) * (p ^ (p >> 30)) + UInt64(i)) & mask
        # init_by_array with the one-word key [seed]
        var key = UInt64(seed)
        var i = 1
        for _ in range(624):
            var p = self.mt[i - 1]
            self.mt[i] = ((self.mt[i] ^ ((p ^ (p >> 30)) * UInt64(1664525))) + key) & mask
            i += 1
            if i >= 624:
                self.mt[0] = self.mt[623]
                i = 1
        for _ in range(623):
            var p = self.mt[i - 1]
            self.mt[i] = ((self.mt[i] ^ ((p ^ (p >> 30)) * UInt64(1566083941))) - UInt64(i)) & mask
            i += 1
            if i >= 624:
                self.mt[0] = self.mt[623]
                i = 1
        self.mt[0] = UInt64(0x80000000)

    def _refill(mut self):
        for k in range(624):
            var y = (self.mt[k] & UInt64(0x80000000)) | (self.mt[(k + 1) % 624] & UInt64(0x7FFFFFFF))
            var v = self.mt[(k + 397) % 624] ^ (y >> 1)
            if (y & 1) == 1:
                v = v ^ UInt64(0x9908B0DF)
            self.mt[k] = v
        self.idx = 0

    def next32(mut self) -> UInt64:
        if self.idx >= 624:
            self._refill()
        var y = self.mt[self.idx]
        self.idx += 1
        y = y ^ (y >> 11)
        y = y ^ ((y << 7) & UInt64(0x9D2C5680))
        y = y ^ ((y << 15) & UInt64(0xEFC60000))
        y = y ^ (y >> 18)
        return y & UInt64(0xFFFFFFFF)

    def random(mut self) -> Float64:
        var a = self.next32() >> 5
        var b = self.next32() >> 6
        return Float64(a * UInt64(67108864) + b) / 9007199254740992.0


# ---------------------------------------------------------------------------
# Weighted graph (dense; weight 0 = no edge)
# ---------------------------------------------------------------------------
struct Graph(Copyable, Movable):
    var n: Int
    var w: List[Float64]

    def __init__(out self, n: Int):
        self.n = n
        self.w = List[Float64](length=n * n, fill=0.0)

    def get(self, i: Int, j: Int) -> Float64:
        return self.w[i * self.n + j]

    def set_edge(mut self, i: Int, j: Int, v: Float64):
        self.w[i * self.n + j] = v
        self.w[j * self.n + i] = v

    def num_edges(self) -> Int:
        var c = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.get(i, j) != 0.0:
                    c += 1
        return c

    def total_weight(self) -> Float64:
        var s = 0.0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                s += self.get(i, j)
        return s


def induced(g: Graph, nodes: List[Int]) -> Graph:
    var s = Graph(len(nodes))
    for a in range(len(nodes)):
        for b in range(a + 1, len(nodes)):
            var v = g.get(nodes[a], nodes[b])
            if v != 0.0:
                s.set_edge(a, b, v)
    return s^


def calc_cut(g: Graph, sol: List[Int]) -> Float64:
    var c = 0.0
    for i in range(g.n):
        for j in range(i + 1, g.n):
            if sol[i] != sol[j]:
                c += g.get(i, j)
    return c


def abs_f(x: Float64) -> Float64:
    if x < 0.0:
        return -x
    return x


# ---------------------------------------------------------------------------
# Community detection: greedy modularity merge, then split to <= max_nodes
# ---------------------------------------------------------------------------
def greedy_modularity(g: Graph) -> List[List[Int]]:
    var n = g.n
    var cw = List[Float64](length=n * n, fill=0.0)
    var deg = List[Float64](length=n, fill=0.0)
    var m2 = 0.0
    for i in range(n):
        for j in range(n):
            var a = abs_f(g.get(i, j))
            cw[i * n + j] = a
            deg[i] += a
            m2 += a
    var members = List[List[Int]]()
    var active = List[Bool]()
    for i in range(n):
        var one = List[Int]()
        one.append(i)
        members.append(one^)
        active.append(True)
    if m2 > 0.0:
        while True:
            var bi = -1
            var bj = -1
            var best = 1e-12
            for i in range(n):
                if not active[i]:
                    continue
                for j in range(i + 1, n):
                    if not active[j] or cw[i * n + j] <= 0.0:
                        continue
                    var dq = 2.0 * (cw[i * n + j] / m2 - (deg[i] / m2) * (deg[j] / m2))
                    if dq > best:
                        best = dq
                        bi = i
                        bj = j
            if bi < 0:
                break
            for k in range(n):
                cw[bi * n + k] += cw[bj * n + k]
                cw[k * n + bi] += cw[k * n + bj]
            cw[bi * n + bi] = 0.0
            deg[bi] += deg[bj]
            active[bj] = False
            var moved = members[bj].copy()
            for x in moved:
                members[bi].append(x)
    var out = List[List[Int]]()
    for i in range(n):
        if active[i]:
            out.append(members[i].copy())
    return out^


def sorted_copy(xs: List[Int]) -> List[Int]:
    var out = xs.copy()
    for i in range(1, len(out)):
        var j = i
        while j > 0 and out[j] < out[j - 1]:
            var t = out[j]
            out[j] = out[j - 1]
            out[j - 1] = t
            j -= 1
    return out^


def split_to_communities(g: Graph, max_nodes: Int) -> List[List[Int]]:
    var out = List[List[Int]]()
    var comms = greedy_modularity(g)
    for c in comms:
        var cs = sorted_copy(c)
        if len(cs) <= max_nodes:
            out.append(cs^)
            continue
        var sub = induced(g, cs)
        var parts = greedy_modularity(sub)
        if len(parts) == 1:
            var cur = List[Int]()
            for x in cs:
                cur.append(x)
                if len(cur) == max_nodes:
                    out.append(cur.copy())
                    cur = List[Int]()
            if len(cur) > 0:
                out.append(cur^)
        else:
            var inner = split_to_communities(sub, max_nodes)
            for p in inner:
                var mapped = List[Int]()
                for x in p:
                    mapped.append(cs[x])
                out.append(mapped^)
    return out^


def communities_to_graph(g: Graph, comms: List[List[Int]], sol: List[Int]) -> Graph:
    """Edge weight between communities: +w for an edge whose ends share a side, -w otherwise."""
    var h = Graph(len(comms))
    for i in range(len(comms)):
        for j in range(i + 1, len(comms)):
            var wt = 0.0
            for a in comms[i]:
                for b in comms[j]:
                    var v = g.get(a, b)
                    if v != 0.0:
                        if sol[a] == sol[b]:
                            wt += v
                        else:
                            wt -= v
            if wt != 0.0:
                h.set_edge(i, j, wt)
    return h^


# ---------------------------------------------------------------------------
# Depth-2 QAOA for weighted MaxCut on a small graph
# ---------------------------------------------------------------------------
comptime LAYERS = 2


def qaoa_circuit(g: Graph, params: List[Float64]) -> Circuit:
    """Ansatz with params = gammas (LAYERS), then betas (LAYERS)."""
    var c = Circuit(g.n, name="qaoa_maxcut")
    for q in range(g.n):
        c.h(q)
    for l in range(LAYERS):
        for i in range(g.n):
            for j in range(i + 1, g.n):
                var v = g.get(i, j)
                if v != 0.0:
                    c.rzz(params[l] * v, i, j)
        for q in range(g.n):
            c.rx(params[LAYERS + l], q)
    return c^


def cut_of_index(g: Graph, x: Int) -> Float64:
    var c = 0.0
    for i in range(g.n):
        for j in range(i + 1, g.n):
            if ((x >> i) & 1) != ((x >> j) & 1):
                c += g.get(i, j)
    return c


struct CutObjective(Objective, Movable):
    var g: Graph
    var cost: List[Float64]  # cut value of every basis state
    var nfev: Int

    def __init__(out self, var g: Graph):
        self.g = g^
        self.cost = List[Float64](length=1 << self.g.n, fill=0.0)
        for x in range(1 << self.g.n):
            self.cost[x] = cut_of_index(self.g, x)
        self.nfev = 0

    def num_params(self) -> Int:
        return 2 * LAYERS

    def value(mut self, x: List[Float64]) raises -> Float64:
        self.nfev += 1
        var pr = evolve(qaoa_circuit(self.g, x)).probabilities()
        var e = 0.0
        for i in range(len(pr)):
            e += pr[i] * self.cost[i]
        return -e

    def value_and_gradient(mut self, x: List[Float64], mut grad: List[Float64]) raises -> Float64:
        var f0 = self.value(x)
        var xp = x.copy()
        for i in range(len(x)):
            xp[i] = x[i] + 1e-6
            grad[i] = (self.value(xp) - f0) / 1e-6
            xp[i] = x[i]
        return f0


def reference_expected_cut(g: Graph, params: List[Float64]) -> Float64:
    """QUMO-independent dense state-vector simulation of the depth-2 ansatz.

    Written from the gate definitions only (H, RZZ(t) = exp(-i t/2 Z.Z), RX(b) = exp(-i b/2 X)),
    with the true signed edge weights, and returns <cut>.
    """
    var dim = 1 << g.n
    var re = List[Float64](length=dim, fill=1.0 / sqrt(Float64(dim)))
    var im = List[Float64](length=dim, fill=0.0)
    for l in range(LAYERS):
        for i in range(g.n):
            for j in range(i + 1, g.n):
                var v = g.get(i, j)
                if v == 0.0:
                    continue
                var th = params[l] * v
                for x in range(dim):
                    var zz = 1.0
                    if ((x >> i) & 1) != ((x >> j) & 1):
                        zz = -1.0
                    var ph = -0.5 * th * zz
                    var cr = cos(ph)
                    var ci = sin(ph)
                    var a = re[x]
                    var b = im[x]
                    re[x] = a * cr - b * ci
                    im[x] = a * ci + b * cr
        var c = cos(0.5 * params[LAYERS + l])
        var sn = sin(0.5 * params[LAYERS + l])
        for q in range(g.n):
            for x in range(dim):
                if ((x >> q) & 1) == 0:
                    var y = x | (1 << q)
                    var x0 = re[x]
                    var y0 = im[x]
                    var x1 = re[y]
                    var y1 = im[y]
                    re[x] = c * x0 + sn * y1
                    im[x] = c * y0 - sn * x1
                    re[y] = c * x1 + sn * y0
                    im[y] = c * y1 - sn * x0
    var e = 0.0
    for x in range(dim):
        e += (re[x] * re[x] + im[x] * im[x]) * cut_of_index(g, x)
    return e


def exec_qaoa(g: Graph, seed: UInt64, mut log: CheckLog, label: String) raises -> List[Int]:
    """Notebook's exec_qaoa_classiq_sort_res: QAOA, sample, return the best-cut sampled solution."""
    var sol = List[Int](length=g.n, fill=0)
    if g.n <= 1 or g.num_edges() == 0:
        return sol^
    var obj = CutObjective(g.copy())
    var x0: List[Float64] = [0.3, 0.6, 0.8, 0.4]
    var res = NelderMead(maxiter=20, initial_step=0.5).minimize(obj, x0)
    var exp_cut = -res.fun
    var uniform = g.total_weight() / 2.0
    var best_brute = obj.cost[0]
    for x in range(1 << g.n):
        if obj.cost[x] > best_brute:
            best_brute = obj.cost[x]
    var circ = qaoa_circuit(g, res.x)
    circ.measure_all()
    var counts = sample(circ, 1000, seed).counts.copy()
    var best_cost = -1e18
    var best_count = -1
    var best_x = 0
    for kv in counts.items():
        var x = bitstring_to_int(kv.key)
        var cst = obj.cost[x]
        var cnt = Int(kv.value)
        if cst > best_cost or (cst == best_cost and cnt > best_count):
            best_cost = cst
            best_count = cnt
            best_x = x
    for q in range(g.n):
        sol[q] = (best_x >> q) & 1
    var ref_cut = reference_expected_cut(g, res.x)
    _ = log.check(label + ": <cut> " + String(exp_cut) + " = independent state-vector value " + String(ref_cut), abs_f(exp_cut - ref_cut) < 1e-9)
    _ = log.check(label + ": <cut> " + String(exp_cut) + " beats uniform " + String(uniform), exp_cut > uniform)
    _ = log.check(label + ": best sampled cut " + String(best_cost) + " = brute-force optimum " + String(best_brute), best_cost == best_brute)
    return sol^


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def one_exchange(g: Graph) -> Float64:
    var sol = List[Int](length=g.n, fill=0)
    var improved = True
    while improved:
        improved = False
        for v in range(g.n):
            var same = 0.0
            var diff = 0.0
            for u in range(g.n):
                var w = g.get(v, u)
                if w != 0.0:
                    if sol[u] == sol[v]:
                        same += w
                    else:
                        diff += w
            if same > diff:
                sol[v] = 1 - sol[v]
                improved = True
    return calc_cut(g, sol)


def flip_basics(mut full: List[Int], basics: List[List[Int]], hsol: List[Int]):
    for k in range(len(hsol)):
        if hsol[k] == 1:
            for bn in basics[k]:
                full[bn] = 1 - full[bn]


def merge_basics(basics: List[List[Int]], comms: List[List[Int]]) -> List[List[Int]]:
    var out = List[List[Int]]()
    for c in comms:
        var flat = List[Int]()
        for old in c:
            for bn in basics[old]:
                flat.append(bn)
        out.append(flat^)
    return out^


def main() raises:
    var log = CheckLog("classiq/qaoa_in_qaoa")
    comptime MAX_QUBITS = 6
    comptime TOTAL_NODES = 40

    # Notebook cell 2: random.seed(246); G = nx.erdos_renyi_graph(40, 0.2).
    var rng = PyRandom(246)
    var G = Graph(TOTAL_NODES)
    for i in range(TOTAL_NODES):
        for j in range(i + 1, TOTAL_NODES):
            if rng.random() < 0.2:
                G.set_edge(i, j, 1.0)
    var n_edges = G.num_edges()
    print("number of edges:", n_edges)
    _ = log.check("random graph has the notebook's printed 158 edges", n_edges == 158, "edges=" + String(n_edges))

    var comms = split_to_communities(G, MAX_QUBITS)
    print("number of communities:", len(comms))
    var seen = List[Int](length=TOTAL_NODES, fill=0)
    var max_size = 0
    for c in comms:
        max_size = max(max_size, len(c))
        for x in c:
            seen[x] += 1
    var covered = True
    for v in range(TOTAL_NODES):
        if seen[v] != 1:
            covered = False
    _ = log.check("notebook's printed 10 communities", len(comms) == 10, "communities=" + String(len(comms)))
    _ = log.check("communities partition the 40 nodes", covered)
    _ = log.check("every community has at most 6 nodes (largest " + String(max_size) + ")", max_size <= MAX_QUBITS)

    var full = List[Int](length=TOTAL_NODES, fill=0)
    var seed = UInt64(10)
    for ci in range(len(comms)):
        var sub = induced(G, comms[ci])
        var s = exec_qaoa(sub, seed + UInt64(ci), log, "community " + String(ci) + " (" + String(len(comms[ci])) + " nodes, " + String(sub.num_edges()) + " edges)")
        for k in range(len(comms[ci])):
            full[comms[ci][k]] = s[k]

    var H = communities_to_graph(G, comms, full)
    var basics = List[List[Int]]()
    for c in comms:
        basics.append(c.copy())
    var cut_now = calc_cut(G, full)
    var cut0 = cut_now
    print("cut after the community solves:", cut_now)

    var level = 0
    var n_cur = len(comms)
    var all_ok = True
    var levels = 0
    if n_cur > MAX_QUBITS:
        while n_cur > MAX_QUBITS:
            var comms2 = split_to_communities(H, MAX_QUBITS)
            n_cur = len(comms2)
            var inter = List[Int](length=H.n, fill=0)
            for ci in range(len(comms2)):
                var sub = induced(H, comms2[ci])
                var s = exec_qaoa(sub, UInt64(100 + 10 * level + ci), log, "level " + String(level + 1) + " group " + String(ci) + " (" + String(len(comms2[ci])) + " nodes)")
                for k in range(len(comms2[ci])):
                    inter[comms2[ci][k]] = s[k]
            var gain = calc_cut(H, inter)
            flip_basics(full, basics, inter)
            var after = calc_cut(G, full)
            if after != cut_now + gain:
                all_ok = False
            cut_now = after
            levels += 1
            H = communities_to_graph(H, comms2, inter)
            basics = merge_basics(basics, comms2)
            level += 1
    var fin = exec_qaoa(H, UInt64(900), log, "top level (" + String(H.n) + " nodes)")
    var gain_top = calc_cut(H, fin)
    flip_basics(full, basics, fin)
    var after_top = calc_cut(G, full)
    if after_top != cut_now + gain_top:
        all_ok = False
    _ = cut_now
    _ = log.check("every merge level: change in cut(G) equals cut of the community graph (" + String(levels + 1) + " levels)", all_ok)

    var final_cut = calc_cut(G, full)
    var greedy = one_exchange(G)
    print("CUT OF QAOA SQUARE", final_cut)
    print("cut before merging:", cut0)
    print("greedy one exchange strategy:", greedy)
    print("random partition expectation:", Float64(n_edges) / 2.0)
    _ = log.check("final cut " + String(final_cut) + " beats the random-partition expectation " + String(Float64(n_edges) / 2.0) + " (sanity bound)", final_cut > Float64(n_edges) / 2.0)
    _ = log.check("final cut does not exceed |E| = 158", final_cut <= Float64(n_edges))
    _ = log.check("merging does not lower the cut of the community solves (" + String(cut0) + " -> " + String(final_cut) + ")", final_cut >= cut0)

    log.finish()
