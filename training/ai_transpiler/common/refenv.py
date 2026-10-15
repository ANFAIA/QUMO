"""Library-free reference of QUMO's policy inference (qumo/compiler/policy*.mojo).

Pure numpy + stdlib. Used by the exporters to write test vectors and greedy
rollouts into each model manifest, by eval.py, and by the fixture generator.
The Mojo implementation must reproduce these numbers (logits within 1e-5,
rollout action sequences exactly).

Environments (packed state = list of Python ints, same layout as the Mojo
`_Env`): see the docstring of qumo/compiler/policy_synth.mojo. They mirror
qiskit-gym 0.4.1's Rust environments (Apache-2.0) without the random
inversions / random symmetries used during training.
"""
from __future__ import annotations

import math

import numpy as np

H, S, SDG, SX, SXDG, X, Y, Z, CX, CZ, SWAP = "H", "S", "SDG", "SX", "SXDG", "X", "Y", "Z", "CX", "CZ", "SWAP"


# ---------------------------------------------------------------- policy
class Policy:
    """twisterl BasicPolicy forward in float32 (first layer stored [OBS, EMB])."""

    def __init__(self, tensors, obs_perms=(), act_perms=()):
        self.WeT = np.ascontiguousarray(tensors["embeddings.weight"], dtype=np.float32)
        self.be = np.asarray(tensors["embeddings.bias"], dtype=np.float32)
        self.common = self._seq(tensors, "common")
        self.action = self._seq(tensors, "action")
        self.value = self._seq(tensors, "value")
        self.obs_perms = [np.asarray(p, dtype=np.int64) for p in obs_perms]
        self.act_perms = [np.asarray(p, dtype=np.int64) for p in act_perms]
        self.obs_size = self.WeT.shape[0]

    @staticmethod
    def _seq(t, prefix):
        idx = sorted({int(k.split(".")[1]) for k in t if k.startswith(prefix + ".")})
        return [(np.asarray(t[f"{prefix}.{j}.weight"], dtype=np.float32),
                 np.asarray(t[f"{prefix}.{j}.bias"], dtype=np.float32)) for j in idx]

    def raw(self, obs, perm=None):
        obs = np.asarray(obs, dtype=np.int64)
        if perm is not None and len(obs):
            obs = self.obs_perms[perm][obs]
        h = self.be.copy()
        for i in obs:
            if 0 <= i < self.obs_size:
                h += self.WeT[i]
        h = np.maximum(h, 0).astype(np.float32)
        for W, b in self.common:
            h = np.maximum(W @ h + b, 0).astype(np.float32)
        x = h
        for k, (W, b) in enumerate(self.action):
            x = (W @ x + b).astype(np.float32)
            if k + 1 < len(self.action):
                x = np.maximum(x, 0)
        v = h
        for k, (W, b) in enumerate(self.value):
            v = (W @ v + b).astype(np.float32)
            if k + 1 < len(self.value):
                v = np.maximum(v, 0)
        if perm is not None:
            x = x[self.act_perms[perm]]
        return x.astype(np.float32), float(v[0]) if len(self.value) else 0.0

    def ensemble(self, obs):
        P = len(self.obs_perms)
        if P == 0:
            return self.raw(obs)
        acc = np.zeros(self.action[-1][0].shape[0], dtype=np.float32)
        val = np.float32(0)
        for p in range(P):
            l, v = self.raw(obs, p)
            acc = (acc + l / np.float32(P)).astype(np.float32)
            val = np.float32(val + np.float32(v) / np.float32(P))
        return acc, float(val)


# ---------------------------------------------------------------- Pauli algebra
def pauli_conj(kind, a, b, x, z, s):
    """g P g^dag for a Hermitian Pauli (masks x, z; sign bit s). Returns (x, z, s)."""
    bit = lambda m, q: (m >> q) & 1
    if kind == H:
        xa, za = bit(x, a), bit(z, a)
        s ^= xa & za
        x = (x & ~(1 << a)) | (za << a)
        z = (z & ~(1 << a)) | (xa << a)
    elif kind == S:
        xa, za = bit(x, a), bit(z, a)
        s ^= xa & za
        z ^= xa << a
    elif kind == SDG:
        xa, za = bit(x, a), bit(z, a)
        s ^= xa & (1 - za)
        z ^= xa << a
    elif kind == SX:
        xa, za = bit(x, a), bit(z, a)
        s ^= za & (1 - xa)
        x ^= za << a
    elif kind == SXDG:
        xa, za = bit(x, a), bit(z, a)
        s ^= za & xa
        x ^= za << a
    elif kind == X:
        s ^= bit(z, a)
    elif kind == Z:
        s ^= bit(x, a)
    elif kind == Y:
        s ^= bit(x, a) ^ bit(z, a)
    elif kind == CX:
        xc, zc, xt, zt = bit(x, a), bit(z, a), bit(x, b), bit(z, b)
        if xc and zt and (xt == zc):
            s ^= 1
        x ^= xc << b
        z ^= zt << a
    elif kind == CZ:
        x, z, s = pauli_conj(H, b, -1, x, z, s)
        x, z, s = pauli_conj(CX, a, b, x, z, s)
        x, z, s = pauli_conj(H, b, -1, x, z, s)
    elif kind == SWAP:
        xa, za, xb, zb = bit(x, a), bit(z, a), bit(x, b), bit(z, b)
        x = (x & ~((1 << a) | (1 << b))) | (xb << a) | (xa << b)
        z = (z & ~((1 << a) | (1 << b))) | (zb << a) | (za << b)
    else:
        raise ValueError(kind)
    return x, z, s


def gf2_inverse(rows, n):
    a = list(rows)
    inv = [1 << i for i in range(n)]
    for col in range(n):
        piv = next((r for r in range(col, n) if (a[r] >> col) & 1), None)
        if piv is None:
            return None
        a[piv], a[col] = a[col], a[piv]
        inv[piv], inv[col] = inv[col], inv[piv]
        for r in range(n):
            if r != col and (a[r] >> col) & 1:
                a[r] ^= a[col]
                inv[r] ^= inv[col]
    return inv


# ---------------------------------------------------------------- environments
class Env:
    """kind in {'permutation','linear_function','clifford','pauli_network'};
    actions: list of (gate, q0, q1) with q1 = -1 for 1q gates."""

    def __init__(self, kind, n, actions, max_rot=0):
        self.kind, self.n, self.actions, self.max_rot = kind, n, [tuple(a) for a in actions], max_rot
        self.R = 0
        self.anti = []

    # pauli layout
    def _rx(self): return 2 * self.n
    def _rz(self): return 2 * self.n + self.R
    def _rs(self): return 2 * self.n + 2 * self.R
    def _cnt(self): return 2 * self.n + 3 * self.R
    def _nodes(self): return 2 * self.n + 3 * self.R + 1

    def solved(self, st):
        n = self.n
        if self.kind == "permutation":
            return all(st[i] == i for i in range(n))
        rows = n if self.kind == "linear_function" else 2 * n
        if any(st[i] != (1 << i) for i in range(rows)):
            return False
        if self.kind == "pauli_network":
            return st[self._cnt()] == 0
        return True

    def observe(self, st):
        n = self.n
        if self.kind == "permutation":
            return [i * n + st[i] for i in range(n)]
        if self.kind == "linear_function":
            return [i * n + j for i in range(n) for j in range(n) if (st[i] >> j) & 1]
        d = 2 * n
        if self.kind == "clifford":
            return [i * d + j for i in range(d) for j in range(d) if (st[i] >> j) & 1]
        C = d + self.max_rot
        cnt = st[self._cnt()]
        shown = min(cnt, self.max_rot)
        out = []
        for r in range(d):
            for j in range(d):
                if (st[r] >> j) & 1:
                    out.append(r * C + j)
            for k in range(shown):
                ri = st[self._nodes() + k]
                bits = st[self._rx() + ri] if r < n else st[self._rz() + ri]
                if (bits >> (r if r < n else r - n)) & 1:
                    out.append(r * C + d + k)
        return out

    def _cliff_rows(self, st, k, a, b):
        n = self.n
        if k == H:
            st[a], st[n + a] = st[n + a], st[a]
        elif k in (S, SDG):
            st[n + a] ^= st[a]
        elif k in (SX, SXDG):
            st[a] ^= st[n + a]
        elif k == CX:
            st[b] ^= st[a]
            st[n + a] ^= st[n + b]
        elif k == CZ:
            st[n + a] ^= st[b]
            st[n + b] ^= st[a]
        elif k == SWAP:
            st[a], st[b] = st[b], st[a]
            st[n + a], st[n + b] = st[n + b], st[n + a]

    def step(self, st, act):
        st = list(st)
        self.apply(st, act, None)
        return st

    def apply(self, st, act, emit):
        k, a, b = self.actions[act]
        if self.kind == "permutation":
            st[a], st[b] = st[b], st[a]
            if emit is not None:
                emit.append((SWAP, a, b))
            return
        if self.kind == "linear_function":
            if k == CX:
                st[b] ^= st[a]
            elif k == SWAP:
                st[a], st[b] = st[b], st[a]
            if emit is not None:
                emit.append((k, a, b))
            return
        if self.kind == "clifford":
            self._cliff_rows(st, k, a, b)
            if emit is not None:
                emit.append((k, a, b))
            return
        if k == CX:
            self._pcx(st, b, a, emit)
        elif k == CZ:
            self._p1q(st, H, b, None)
            self._pcx(st, a, b, None)
            self._p1q(st, H, b, None)
            if emit is not None:
                emit.append((CZ, a, b))
            self.clean(st, emit)
        elif k == SWAP:
            self._pcx(st, b, a, emit)
            self._pcx(st, a, b, emit)
            self._pcx(st, b, a, emit)
        else:
            self._p1q(st, k, a, emit)

    def _rots(self, st, k, a, b):
        for ri in range(self.R):
            x, z, s = pauli_conj(k, a, b, st[self._rx() + ri], st[self._rz() + ri], st[self._rs() + ri])
            st[self._rx() + ri], st[self._rz() + ri], st[self._rs() + ri] = x, z, s

    def _p1q(self, st, k, q, emit):
        self._cliff_rows(st, k, q, -1)
        self._rots(st, k, q, -1)
        if emit is not None:
            emit.append((k, q, -1))

    def _pcx(self, st, c, t, emit):
        self._cliff_rows(st, CX, c, t)
        self._rots(st, CX, c, t)
        if emit is not None:
            emit.append((CX, c, t))
        self.clean(st, emit)

    def clean(self, st, emit):
        removed = True
        while removed:
            removed = False
            cnt = st[self._cnt()]
            nodes = [st[self._nodes() + p] for p in range(cnt)]
            kill = [False] * cnt
            for pos in range(cnt):
                ri = nodes[pos]
                if any(rj < ri and self.anti[ri][rj] for rj in nodes):
                    continue
                x, z = st[self._rx() + ri], st[self._rz() + ri]
                w = bin(x | z).count("1")
                if w <= 1:
                    if emit is not None and w == 1:
                        q = (x | z).bit_length() - 1
                        axis = 2 if not (x >> q) & 1 else (1 if (z >> q) & 1 else 0)
                        emit.append(("ROT", ri, q, axis, st[self._rs() + ri]))
                    st[self._rx() + ri] = 0
                    st[self._rz() + ri] = 0
                    kill[pos] = True
                    removed = True
            if removed:
                last = cnt
                for pos in range(cnt - 1, -1, -1):
                    if kill[pos]:
                        last -= 1
                        st[self._nodes() + pos] = st[self._nodes() + last]
                        st[self._nodes() + last] = 0
                st[self._cnt()] = last

    def inverse(self, st):
        n = self.n
        if self.kind == "permutation":
            out = [0] * n
            for i in range(n):
                out[st[i]] = i
            return out
        if self.kind == "linear_function":
            return gf2_inverse(st, n)
        if self.kind == "clifford":
            return gf2_inverse(st, 2 * n)
        return None

    def key(self, st):
        return tuple(st)

    # -------------------------------------------- pauli start state
    def pauli_start(self, tab_rows, rots):
        """tab_rows: 2n row masks of the Clifford state; rots: list of (x, z, sign_bit)."""
        self.R = len(rots)
        self.anti = [[bin((rots[i][0] & rots[j][1]) ^ (rots[i][1] & rots[j][0])).count("1") & 1 == 1
                      for j in range(self.R)] for i in range(self.R)]
        st = list(tab_rows) + [r[0] for r in rots] + [r[1] for r in rots] + [0] * self.R + [self.R] + list(range(self.R))
        return st


# ---------------------------------------------------------------- search
def ordered_actions(logits):
    """Descending logits; |delta| <= 1e-6 * max(1, |l|) is a tie (lower index first)."""
    order = []
    for a in range(len(logits)):
        la = np.float32(logits[a])
        pos = len(order)
        while pos > 0:
            lb = np.float32(logits[order[pos - 1]])
            tol = np.float32(1e-6) * max(np.float32(1.0), abs(lb))
            if lb - la >= -tol:
                break
            pos -= 1
        order.insert(pos, a)
    return order


def log_softmax(logits):
    l = np.asarray(logits, dtype=np.float64)
    mx = l.max()
    return l - (mx + math.log(np.exp(l - mx).sum()))


def beam(env, pol, start, width, ensemble, max_depth=128):
    """Mirror of policy_synth.mojo `_beam`; returns the list of solutions of the
    first solving depth (each an action list)."""
    if env.solved(start):
        return [[]]
    visited = {env.key(start)}
    frontier = [(0.0, list(start), [])]
    for _ in range(max_depth):
        cands = []  # (score, state, acts)
        ckey = {}
        sols = []
        for score, st, acts in frontier:
            obs = env.observe(st)
            logits, _ = pol.ensemble(obs) if ensemble else pol.raw(obs)
            lp = log_softmax(logits)
            taken = 0
            for a in ordered_actions(logits):
                if taken >= width:
                    break
                ns = env.step(st, a)
                if env.solved(ns):
                    sols.append(acts + [a])
                    taken += 1
                    continue
                k = env.key(ns)
                if k in visited:
                    continue
                sc = score + float(lp[a])
                if k in ckey:
                    i = ckey[k]
                    if sc > cands[i][0]:
                        cands[i] = (sc, cands[i][1], acts + [a])
                else:
                    ckey[k] = len(cands)
                    cands.append((sc, ns, acts + [a]))
                taken += 1
        if sols or not cands:
            return sols
        idxs = []
        for i in range(len(cands)):
            pos = len(idxs)
            while pos > 0 and cands[idxs[pos - 1]][0] < cands[i][0]:
                pos -= 1
            idxs.insert(pos, i)
        for c in cands:
            visited.add(env.key(c[1]))
        frontier = [cands[i] for i in idxs[:width]]
    return []


def greedy_trace(env, pol, start, max_depth=128):
    """The path of the width-1 ensemble beam (greedy, visited states skipped),
    returned even when it does not solve: (actions, solved)."""
    st = list(start)
    visited = {env.key(st)}
    acts = []
    for _ in range(max_depth):
        if env.solved(st):
            return acts, True
        logits, _ = pol.ensemble(env.observe(st))
        nxt = None
        for a in ordered_actions(logits):
            ns = env.step(st, a)
            if env.solved(ns) or env.key(ns) not in visited:
                nxt = (a, ns)
                break
        if nxt is None:
            return acts, False
        acts.append(nxt[0])
        st = nxt[1]
        visited.add(env.key(st))
    return acts, env.solved(st)


def greedy_rollout(env, pol, start, max_depth=128):
    sols = beam(env, pol, start, 1, True, max_depth)
    return sols[0] if sols else None


def emit_actions(env, start, acts):
    st = list(start)
    emit = []
    for a in acts:
        env.apply(st, a, emit)
    return st, emit


def cost2(emit):
    c = 0
    ng = 0
    for e in emit:
        if e[0] == "ROT":
            continue
        ng += 1
        c += 3 if e[0] == SWAP else (1 if e[0] in (CX, CZ) else 0)
    return c, ng
