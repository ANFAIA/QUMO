"""Python mirror of QUMO's learned-routing environment (qumo/compiler/ai_route.mojo).

Bit-for-bit mirror of the Mojo `RProblem` / `Episode`: node grouping (units,
singles, fences, barriers, classical chain, 2q DAG), the execution loop,
HORIZON=8 topological layers sorted then shuffled with QUMO's xoshiro256**
`Rng`, NUM_ACTIVE_SWAPS=16 active swaps, the 128-int observation, the progress
fallback, and the output simplification bookkeeping (cancel / virtual / fused
SWAPs) that defines the routing overhead in CX units (3 per live SWAP, 1 per
SWAP fused with a CX). Used for PPO fine-tuning, evaluation and to produce the
parity test vectors that the Mojo test suite replays.

The algorithm is derived from qiskit-ibm-transpiler 0.18.0 rust/src/routing
(Apache-2.0), re-implemented from a written spec (no IBM code).

Circuit format: list of (kind, a, b) tuples with kind one of
  "cx", "cz", "swap", "u2" (any other plain 2q gate)  -> 2q plain gate
  "u1" (any plain 1q gate)                              -> 1q plain gate
  "measure" (a)                                         -> fence on wire a
  "barrier"                                             -> all-wire fence
"""
from __future__ import annotations

import numpy as np

HORIZON = 8
NUM_ACTIVE_SWAPS = 16
OBS_SIZE = HORIZON * NUM_ACTIVE_SWAPS
M64 = (1 << 64) - 1

NK_SINGLE, NK_UNIT, NK_FENCE, NK_BARRIER = 0, 1, 2, 3
IT_NODE, IT_SWAP, IT_FUSED = 0, 1, 2
MODE_POLICY, MODE_SCORED, MODE_HEURISTIC = 0, 1, 2


# ----------------------------------------------------------------------------- Rng
def _splitmix(x):
    x = (x + 0x9E3779B97F4A7C15) & M64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return x, z ^ (z >> 31)


def _rotl(x, k):
    return ((x << k) | (x >> (64 - k))) & M64


class Rng:
    """qumo.sim.rng.Rng (xoshiro256** seeded by SplitMix64), bit-exact."""

    __slots__ = ("s0", "s1", "s2", "s3")

    def __init__(self, seed):
        sm = seed & M64
        sm, self.s0 = _splitmix(sm)
        sm, self.s1 = _splitmix(sm)
        sm, self.s2 = _splitmix(sm)
        sm, self.s3 = _splitmix(sm)

    def next_u64(self):
        result = (_rotl((self.s1 * 5) & M64, 7) * 9) & M64
        t = (self.s1 << 17) & M64
        self.s2 ^= self.s0
        self.s3 ^= self.s1
        self.s1 ^= self.s2
        self.s0 ^= self.s3
        self.s2 ^= t
        self.s3 = _rotl(self.s3, 45)
        return result

    def below(self, n):
        return self.next_u64() % n

    def uniform(self):
        return (self.next_u64() >> 11) * (1.0 / 9007199254740992.0)

    def split(self, stream):
        return Rng(self.s0 ^ ((stream * 0xD1B54A32D192ED03) & M64) ^ self.s3)


def episode_seed(base, a, b, c, r):
    """Mojo `_episode_seed`: Rng(base).split(a*1000003 + b*1009 + c).split(r).next_u64()."""
    return Rng(base).split(a * 1000003 + b * 1009 + c).split(r).next_u64()


def shuffle_range(rng, lst, lo, hi):
    i = hi - 1
    while i > lo:
        j = lo + rng.below(i - lo + 1)
        lst[i], lst[j] = lst[j], lst[i]
        i -= 1


# ----------------------------------------------------------------------------- Target
class RTarget:
    """Coupling graph as qumo Target: deduplicated (a<b) edges in insertion
    order, sorted CSR adjacency, BFS hop distances (unreachable = N+1)."""

    def __init__(self, n, edges, name="custom"):
        self.name = name
        self.N = n
        self.edges = []
        self.edge_index = {}
        for a, b in edges:
            lo, hi = min(a, b), max(a, b)
            if lo == hi or (lo, hi) in self.edge_index:
                continue
            self.edge_index[(lo, hi)] = len(self.edges)
            self.edges.append((lo, hi))
        self.adj = [[] for _ in range(n)]
        for a, b in self.edges:
            self.adj[a].append(b)
            self.adj[b].append(a)
        for q in range(n):
            self.adj[q].sort()
        self.adj_e = [[self.edge_index[(min(q, y), max(q, y))] for y in self.adj[q]] for q in range(n)]
        inf = n + 1
        D = np.full((n, n), inf, dtype=np.int64)
        for s in range(n):
            D[s, s] = 0
            fr = [s]
            while fr:
                nx = []
                for u in fr:
                    for v in self.adj[u]:
                        if D[s, v] == inf:
                            D[s, v] = D[s, u] + 1
                            nx.append(v)
                fr = nx
        self.D = D
        self.Dl = D.tolist()
        self.diameter = int(max((d for row in self.Dl for d in row if d < inf), default=0))

    def flat_edges(self):
        return [x for e in self.edges for x in e]


# ----------------------------------------------------------------------------- Problem
class RProblem:
    """Mirror of the Mojo RProblem (node ids follow program order)."""

    def __init__(self, ops, n):
        self.n = n
        self.ops = list(ops)
        kind, wa, wb, nops, preds, preds2 = [], [], [], [], [], []
        last = [-1] * max(n, 1)
        last2 = [-1] * max(n, 1)
        opn = [-1] * max(n, 1)
        last_cl = -1

        def close(w):
            u = opn[w]
            if u < 0:
                return
            opn[wa[u]] = -1
            opn[wb[u]] = -1

        def add_unique(lst, v):
            if v >= 0 and v not in lst:
                lst.append(v)

        for i, (k, a, b) in enumerate(self.ops):
            if k == "barrier":
                for w in range(n):
                    close(w)
                u = len(kind)
                kind.append(NK_BARRIER); wa.append(-1); wb.append(-1); nops.append([i])
                pl = []
                for w in range(n):
                    add_unique(pl, last[w])
                    last[w] = u
                preds.append(pl); preds2.append([])
                continue
            if k in ("cx", "cz", "swap", "u2"):
                if opn[a] >= 0 and opn[a] == opn[b]:
                    nops[opn[a]].append(i)
                    continue
                close(a); close(b)
                u = len(kind)
                kind.append(NK_UNIT); wa.append(min(a, b)); wb.append(max(a, b)); nops.append([i])
                pl = []
                add_unique(pl, last[a]); add_unique(pl, last[b])
                preds.append(pl)
                p2 = []
                add_unique(p2, last2[a]); add_unique(p2, last2[b])
                preds2.append(p2)
                last[a] = last[b] = u
                last2[a] = last2[b] = u
                opn[a] = opn[b] = u
                continue
            if k == "u1":
                if opn[a] >= 0:
                    nops[opn[a]].append(i)
                    continue
                u = len(kind)
                kind.append(NK_SINGLE); wa.append(a); wb.append(-1); nops.append([i])
                pl = []
                add_unique(pl, last[a])
                preds.append(pl); preds2.append([])
                last[a] = u
                continue
            if k == "measure":
                ws = [a]
            else:
                raise ValueError(k)
            for w in ws:
                close(w)
            u = len(kind)
            kind.append(NK_FENCE); wa.append(ws[0]); wb.append(-1); nops.append([i])
            pl = []
            for w in ws:
                add_unique(pl, last[w])
                last[w] = u
            add_unique(pl, last_cl)
            last_cl = u
            preds.append(pl); preds2.append([])
        nn = len(kind)
        self.kind, self.wa, self.wb, self.nops = kind, wa, wb, nops
        self.indeg = [len(p) for p in preds]
        self.npred2 = [len(p) for p in preds2]
        self.succ = [[] for _ in range(nn)]
        self.succ2 = [[] for _ in range(nn)]
        for u in range(nn):
            for p in preds[u]:
                self.succ[p].append(u)
            for p in preds2[u]:
                self.succ2[p].append(u)
        self.n2q = sum(1 for u in range(nn) if wb[u] >= 0)
        cxu = {"cx": 1, "cz": 1, "swap": 3, "u2": 2}
        self.ucost = [sum(cxu.get(self.ops[i][0], 0) for i in nops[u]) for u in range(nn)]
        self.single_cx = [kind[u] == NK_UNIT and len(nops[u]) == 1 and self.ops[nops[u][0]][0] == "cx" for u in range(nn)]

    def num_nodes(self):
        return len(self.kind)


def reversed_ops(ops):
    return list(reversed(ops))


def max_steps(prob, tgt):
    return 1000 + 64 * (prob.n2q + 1) * max(2, tgt.diameter)


# ----------------------------------------------------------------------------- Model
class RouteMLP:
    """acts = b0 + sum_i sign(obs_i) E[i,:]; logits = W1 relu(acts) + b1 (float64)."""

    def __init__(self, emb, b0, w1, b1):
        self.emb = np.asarray(emb, dtype=np.float64)  # [128, H]
        self.b0 = np.asarray(b0, dtype=np.float64)
        self.w1 = np.asarray(w1, dtype=np.float64)  # [16, H]
        self.b1 = np.asarray(b1, dtype=np.float64)

    def logits(self, obs):
        s = np.sign(np.asarray(obs, dtype=np.float64))
        h = self.b0 + s @ self.emb
        return self.w1 @ np.maximum(h, 0.0) + self.b1


# ----------------------------------------------------------------------------- Episode
class Episode:
    def __init__(self, prob: RProblem, tgt: RTarget, l2p0, seed, virtual_ok=True):
        self.prob, self.tgt = prob, tgt
        N = tgt.N
        self.N = N
        self.l2p = list(l2p0)
        self.p2l = [-1] * N
        for w in range(N):
            self.p2l[self.l2p[w]] = w
        self.init_p2l = list(self.p2l)
        self.indeg = list(prob.indeg)
        self.rem2 = list(prob.npred2)
        nn = prob.num_nodes()
        self.done = [False] * nn
        self.front = [u for u in range(nn) if self.indeg[u] == 0]
        self.front2 = [u for u in range(nn) if prob.wb[u] >= 0 and self.rem2[u] == 0]
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
        # output items: kind, node, w0, w1, prev0, prev1, order, dead, single_cx_node
        self.it = []
        self.last = [-1] * N
        self.swaps_live = 0
        self.n_fused = 0
        self.n_virtual = 0
        self.n_cancel = 0
        self.absorb_credit = 0
        self.n_absorbed = 0
        self.swap_count = 0
        self.layers = []
        self.active = []
        self.obs = [0] * OBS_SIZE
        self.actions = []
        self.decay = [1.0] * N
        self.since_reset = 0
        self.f_now = 0
        self.e_now = 0

    # -- items ----------------------------------------------------------------
    def overhead(self):
        return 3 * self.swaps_live + self.n_fused - self.absorb_credit

    def _prev_on(self, i, p):
        it = self.it[i]
        return it[4] if it[2] == p else it[5]

    def _new_item(self, kind, node, w0, w1):
        i = len(self.it)
        self.it.append([kind, node, w0, w1,
                        self.last[w0] if w0 >= 0 else -1,
                        self.last[w1] if w1 >= 0 else -1, 0, False, 0, -1])
        if w0 >= 0:
            self.last[w0] = i
        if w1 >= 0:
            self.last[w1] = i
        return i

    def _passable(self, i):
        k = self.it[i][0]
        if k == IT_FUSED:
            return True
        return k == IT_NODE and self.prob.kind[self.it[i][1]] == NK_UNIT

    def _chain_ok(self, start, w, lo, hi, items):
        i = start
        while i >= 0:
            a, b = self.it[i][2], self.it[i][3]
            if a < 0:
                return False
            if b >= 0 and not (a == lo and b == hi):
                return False
            items.append(i)
            i = self._prev_on(i, w)
        return True

    def _try_virtual(self, lo, hi):
        items = []
        if not self._chain_ok(self.last[lo], lo, lo, hi, items):
            return False
        if not self._chain_ok(self.last[hi], hi, lo, hi, items):
            return False
        for i in sorted(set(items)):
            it = self.it[i]
            if it[3] >= 0:
                it[4], it[5] = it[5], it[4]
            else:
                it[2] = hi if it[2] == lo else lo
        self.last[lo], self.last[hi] = self.last[hi], self.last[lo]
        self.init_p2l[lo], self.init_p2l[hi] = self.init_p2l[hi], self.init_p2l[lo]
        self.n_virtual += 1
        return True

    def push_swap(self, p, x):
        lo, hi = min(p, x), max(p, x)
        walk = []
        i, j = self.last[lo], self.last[hi]
        outcome = 0
        while True:
            if i != j or i < 0:
                break
            if self.it[i][0] == IT_SWAP:
                outcome = 2
                break
            if not self._passable(i):
                break
            walk.append(i)
            pi, pj = self._prev_on(i, lo), self._prev_on(i, hi)
            i, j = pi, pj
        if outcome == 2:
            s = i
            self.it[s][7] = True
            if self.it[s][8] > 0:
                self.absorb_credit -= self.it[s][8]
                self.n_absorbed -= 1
                self.it[self.it[s][9]][9] = -1
                self.it[s][8] = 0
            if not walk:
                self.last[lo] = self._prev_on(s, lo)
                self.last[hi] = self._prev_on(s, hi)
            else:
                e = walk[-1]
                pl, ph = self._prev_on(s, lo), self._prev_on(s, hi)
                self.it[e][4] = pl
                self.it[e][5] = ph
            self.swaps_live -= 1
            self.n_cancel += 1
            return
        if self.virtual_ok and self._try_virtual(lo, hi):
            return
        k = self.last[lo]
        if k >= 0 and k == self.last[hi] and self.it[k][0] == IT_NODE and self.prob.single_cx[self.it[k][1]]:
            self.it[k][0] = IT_FUSED
            self.it[k][6] = 1
            self.n_fused += 1
            return
        it = self._new_item(IT_SWAP, -1, lo, hi)
        if k >= 0 and k == self._prev_on(it, hi) and self.it[k][0] == IT_NODE and self.prob.kind[self.it[k][1]] == NK_UNIT and self.it[k][9] < 0:
            cr = min(3, self.prob.ucost[self.it[k][1]])
            self.it[it][8] = cr
            self.it[it][9] = k
            self.it[k][9] = it
            self.absorb_credit += cr
            self.n_absorbed += 1
        self.swaps_live += 1

    def push_node(self, u):
        prob = self.prob
        if prob.kind[u] == NK_BARRIER:
            it = self._new_item(IT_NODE, u, -1, -1)
            for p in range(self.N):
                self.last[p] = it
            return
        w0 = self.l2p[prob.wa[u]] if prob.wa[u] >= 0 else -1
        w1 = -1
        if prob.wb[u] >= 0:
            w1 = self.l2p[prob.wb[u]]
            if w1 < w0:
                w0, w1 = w1, w0
        if w1 >= 0 and prob.single_cx[u]:
            k = self.last[w0]
            if k >= 0 and k == self.last[w1] and self.it[k][0] == IT_SWAP:
                self.it[k][0] = IT_FUSED
                self.it[k][1] = u
                self.it[k][6] = 0
                self.swaps_live -= 1
                self.n_fused += 1
                return
        it = self._new_item(IT_NODE, u, w0, w1)
        if w1 >= 0 and prob.kind[u] == NK_UNIT:
            k = self.it[it][4]
            if k >= 0 and k == self.it[it][5] and self.it[k][0] == IT_SWAP and self.it[k][9] < 0:
                cr = min(3, prob.ucost[u])
                self.it[k][8] = cr
                self.it[k][9] = it
                self.it[it][9] = k
                self.absorb_credit += cr
                self.n_absorbed += 1

    # -- dynamics -------------------------------------------------------------
    def _dist(self, u):
        return self.tgt.Dl[self.l2p[self.prob.wa[u]]][self.l2p[self.prob.wb[u]]]

    def execute(self):
        prob = self.prob
        count = 0
        while True:
            self.front.sort()
            exe, keep = [], []
            for u in self.front:
                if prob.wb[u] < 0 or self._dist(u) == 1:
                    exe.append(u)
                else:
                    keep.append(u)
            if not exe:
                break
            self.front = keep
            for u in exe:
                self.push_node(u)
                self.done[u] = True
                self.executed += 1
                count += 1
                for s in prob.succ[u]:
                    self.indeg[s] -= 1
                    if self.indeg[s] == 0:
                        self.front.append(s)
                if prob.wb[u] >= 0:
                    self.exec2q += 1
                    for s in prob.succ2[u]:
                        self.rem2[s] -= 1
                        if self.rem2[s] == 0:
                            self.front2.append(s)
            if self.since_reset > 0:
                self._reset_decay()
            if self.front2:
                self.front2 = [u for u in self.front2 if not self.done[u]]
        return count

    def fsum(self):
        return sum(self._dist(u) - 1 for u in self.front2)

    def build_layers(self):
        prob = self.prob
        layers = []
        cur = sorted(self.front2)
        tmp = {}
        d = 0
        while d < HORIZON and cur:
            layers.append(list(cur))
            nxt = []
            for u in cur:
                for s in prob.succ2[u]:
                    if s not in tmp:
                        tmp[s] = self.rem2[s]
                    tmp[s] -= 1
                    if tmp[s] == 0:
                        nxt.append(s)
            cur = sorted(nxt)
            d += 1
        for L in layers:
            shuffle_range(self.rng, L, 0, len(L))
        self.layers = layers

    def build_active(self):
        prob, tgt = self.prob, self.tgt
        act = []
        seen = set()
        for L in self.layers:
            for u in L:
                if len(act) >= NUM_ACTIVE_SWAPS:
                    break
                for q in (self.l2p[prob.wa[u]], self.l2p[prob.wb[u]]):
                    if len(act) >= NUM_ACTIVE_SWAPS:
                        break
                    for e in tgt.adj_e[q]:
                        if e not in seen:
                            seen.add(e)
                            act.append(e)
                            if len(act) >= NUM_ACTIVE_SWAPS:
                                break
            if len(act) >= NUM_ACTIVE_SWAPS:
                break
        shuffle_range(self.rng, act, 0, len(act))
        self.active = act

    def build_obs(self):
        prob, tgt = self.prob, self.tgt
        D = tgt.Dl
        obs = [0] * OBS_SIZE
        l2p = self.l2p
        lay_q = [[(l2p[prob.wa[u]], l2p[prob.wb[u]]) for u in L] for L in self.layers]
        for k, e in enumerate(self.active):
            p, x = tgt.edges[e]
            for d, L in enumerate(lay_q):
                acc = 0
                for pa, pb in L:
                    if pa != p and pa != x and pb != p and pb != x:
                        continue
                    qa = x if pa == p else (p if pa == x else pa)
                    qb = x if pb == p else (p if pb == x else pb)
                    acc += D[qa][qb] - D[pa][pb]
                obs[d + HORIZON * k] = acc
        self.obs = obs
        self.f_now = sum(D[pa][pb] for pa, pb in lay_q[0]) if lay_q else 0
        self.e_now = sum(D[pa][pb] for L in lay_q[1:] for pa, pb in L)

    def heuristic_scores(self):
        """max(decay_p, decay_x) * (front sum after + 0.5 * mean lookahead after)."""
        nl = len(self.layers)
        cnt = sum(len(self.layers[d]) for d in range(1, nl))
        out = []
        for k, e in enumerate(self.active):
            f = float(self.f_now + self.obs[HORIZON * k])
            la = float(self.e_now)
            for d in range(1, HORIZON):
                la += float(self.obs[d + HORIZON * k])
            if cnt > 0:
                la = la / float(cnt)
            p, x = self.tgt.edges[e]
            out.append(max(self.decay[p], self.decay[x]) * (f + 0.5 * la))
        return out

    def choose(self, model, mode, beta=1.0):
        A = len(self.active)
        if mode == MODE_HEURISTIC or model is None:
            hs = self.heuristic_scores()
            best = 0
            for k in range(A):
                if k == 0 or hs[k] < hs[best]:
                    best = k
            return best
        lg = model.logits(self.obs)
        if mode == MODE_POLICY:
            best = 0
            for k in range(1, A):
                if lg[k] > lg[best]:
                    best = k
            return best
        mx = max(lg[:A])
        z = sum(np.exp(lg[k] - mx) for k in range(A))
        lz = np.log(z) + mx
        hs = self.heuristic_scores()
        best, bs = 0, 0.0
        for k in range(A):
            s = hs[k] - beta * (lg[k] - lz)
            if k == 0 or s < bs:
                bs, best = s, k
        return best

    def fallback_swap(self):
        prob, tgt = self.prob, self.tgt
        bu, bd = -1, 0
        for u in sorted(self.front2):
            d = self._dist(u)
            if d > 1 and (bu < 0 or d < bd):
                bu, bd = u, d
        if bu < 0 or bd > self.N:
            return -1
        pa, pb = self.l2p[prob.wa[bu]], self.l2p[prob.wb[bu]]
        for y, e in zip(tgt.adj[pa], tgt.adj_e[pa]):
            if tgt.Dl[y][pb] == bd - 1:
                return e
        return -1

    def apply_swap(self, e):
        p, x = self.tgt.edges[e]
        lp, lx = self.p2l[p], self.p2l[x]
        self.p2l[p], self.p2l[x] = lx, lp
        self.l2p[lx] = p
        self.l2p[lp] = x
        self.swap_count += 1
        self.since_reset += 1
        if self.since_reset >= 5:
            self._reset_decay()
        else:
            self.decay[p] += 0.001
            self.decay[x] += 0.001
        self.push_swap(p, x)

    def _reset_decay(self):
        self.decay = [1.0] * self.N
        self.since_reset = 0

    def finished(self):
        return self.executed == self.prob.num_nodes()

    def pre_decision(self, limit):
        """Execute ready gates and prepare the decision state. Returns the
        number of gates executed, or -1 when the episode is over."""
        ex = self.execute()
        if self.finished():
            return -1
        f = self.fsum()
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
        self.build_layers()
        self.build_active()
        if not self.fallback and self.active:
            self.build_obs()
        return ex

    def needs_policy(self):
        return not self.fallback and len(self.active) > 0

    def act(self, e):
        self.actions.append(e)
        self.apply_swap(e)

    def step(self, model, mode, beta, limit):
        ex = self.pre_decision(limit)
        if ex < 0:
            return False
        if not self.needs_policy():
            e = self.fallback_swap()
            self.n_fallback += 1
            if e < 0:
                self.failed = True
                return False
        else:
            e = self.active[self.choose(model, mode, beta)]
        self.act(e)
        return True

    def run(self, model, mode, beta=1.0, max_steps_=None):
        limit = max(4, 2 * self.prob.n)
        ms = max_steps_ if max_steps_ is not None else max_steps(self.prob, self.tgt)
        while self.step(model, mode, beta, limit):
            if self.steps > ms:
                self.failed = True
                return

    def init_l2p(self):
        l = [0] * self.N
        for p in range(self.N):
            l[self.init_p2l[p]] = p
        return l


def complete_layout(l2p, n, N):
    out = [-1] * N
    used = [False] * N
    for w in range(min(n, len(l2p))):
        out[w] = l2p[w]
        used[l2p[w]] = True
    nxt = 0
    for w in range(N):
        if out[w] < 0:
            while used[nxt]:
                nxt += 1
            out[w] = nxt
            used[nxt] = True
    return out


def random_layout(N, rng: Rng):
    l = list(range(N))
    i = N - 1
    while i > 0:
        j = rng.below(i + 1)
        l[i], l[j] = l[j], l[i]
        i -= 1
    return l
