# Copyright 2026 The QUMO Authors. Licensed under the Apache License, Version 2.0 (see LICENSE).
# Copied from QUMO qumo/compiler/policy_synth.mojo (Pauli conjugation, SignedTab, gf2_inverse, PolicyEnv, the
# log-softmax action order, the width-1 beam and greedy_trace) so that qrl imports nothing from `qumo.*`. The packed
# state dynamics are pinned by the goldens under tests/fixtures/envs and must stay identical to QUMO's inference
# environment: rl/qumo_bridge/tests/test_bridge_parity.mojo replays random action sequences through both.
"""Packed-state dynamics of the four synthesis environments (permutation, linear function, Clifford, Pauli network).

`PolicyEnv` is the reference gym's (version 0.4.1) environment semantics over a packed state (`List[UInt64]`), the part of QUMO's
inference environment that training shares: `qrl.synth_env.TrainEnv` wraps it (reset generators, reward, metrics,
symmetries) and `qrl.export_synth` uses `greedy_trace` / `policy_rollout` with `qrl.model.MLPPolicy` to embed the
greedy rollouts of an exported model. The search drivers (sampling, multi-beam, inversion trick, proposal
verification and block emission) stay in QUMO.

* permutation: state s; SWAP(a, b) swaps s[a], s[b]; obs = {i*n + s[i]}; solved when s is the identity.
* linear function: state = M^-1 (GF(2), row i = bit mask); CX(c, t): row t ^= row c; obs = set bits; solved at I.
* Clifford: the 2n x 2n symplectic matrix of C^-1, phases ignored; gates left-multiply.
* Pauli network: the Clifford state plus one column per rotation, a rotation DAG cleaned after every CX-type action
  (weight <= 1 front-layer rotations are emitted and removed), signs tracked exactly.

See QUMO's qumo/compiler/policy_synth.mojo docstring for the full description of every rule.
"""

from std.collections import List, Dict
from std.math import exp, log
from std.bit import pop_count
from qrl.gates import Op
from qrl.model import ModelFile, MLPPolicy, MODEL_KIND_CLIFFORD, MODEL_KIND_LINEAR, MODEL_KIND_PERM, MODEL_KIND_PAULI, fnv_words


# ============================================================ Pauli algebra
def _bit(m: UInt64, q: Int) -> Bool:
    return ((m >> UInt64(q)) & 1) == 1


def _flip(m: UInt64, q: Int) -> UInt64:
    return m ^ (UInt64(1) << UInt64(q))


def _set(m: UInt64, q: Int, v: Bool) -> UInt64:
    var b = UInt64(1) << UInt64(q)
    if v:
        return m | b
    return m & ~b


def pauli_conj(kind: Int, a: Int, b: Int, mut x: UInt64, mut z: UInt64, mut s: Int):
    """P -> g P g^dag for a Hermitian Pauli (x, z masks, Y = x&z, sign bit s)."""
    if kind == Op.H:
        var xa = _bit(x, a)
        var za = _bit(z, a)
        if xa and za:
            s ^= 1
        x = _set(x, a, za)
        z = _set(z, a, xa)
    elif kind == Op.S:
        var xa = _bit(x, a)
        var za = _bit(z, a)
        if xa and za:
            s ^= 1
        if xa:
            z = _flip(z, a)
    elif kind == Op.SDG:
        var xa = _bit(x, a)
        var za = _bit(z, a)
        if xa and not za:
            s ^= 1
        if xa:
            z = _flip(z, a)
    elif kind == Op.SX:
        var xa = _bit(x, a)
        var za = _bit(z, a)
        if za and not xa:
            s ^= 1
        if za:
            x = _flip(x, a)
    elif kind == Op.SXDG:
        var xa = _bit(x, a)
        var za = _bit(z, a)
        if za and xa:
            s ^= 1
        if za:
            x = _flip(x, a)
    elif kind == Op.X:
        if _bit(z, a):
            s ^= 1
    elif kind == Op.Z:
        if _bit(x, a):
            s ^= 1
    elif kind == Op.Y:
        if _bit(x, a) != _bit(z, a):
            s ^= 1
    elif kind == Op.CX:
        var xc = _bit(x, a)
        var zc = _bit(z, a)
        var xt = _bit(x, b)
        var zt = _bit(z, b)
        if xc and zt and (xt == zc):
            s ^= 1
        if xc:
            x = _flip(x, b)
        if zt:
            z = _flip(z, a)
    elif kind == Op.CZ:
        pauli_conj(Op.H, b, -1, x, z, s)
        pauli_conj(Op.CX, a, b, x, z, s)
        pauli_conj(Op.H, b, -1, x, z, s)
    elif kind == Op.SWAP:
        var xa = _bit(x, a)
        var za = _bit(z, a)
        x = _set(_set(x, a, _bit(x, b)), b, xa)
        z = _set(_set(z, a, _bit(z, b)), b, za)


struct SignedTab(Copyable):
    """Forward tableau with signs: rows 0..n-1 = C X_i C^dag, n..2n-1 = C Z_i C^dag."""

    var n: Int
    var x: List[UInt64]
    var z: List[UInt64]
    var s: List[Int]

    def __init__(out self, n: Int):
        self.n = n
        self.x = List[UInt64](length=2 * n, fill=0)
        self.z = List[UInt64](length=2 * n, fill=0)
        self.s = List[Int](length=2 * n, fill=0)
        for i in range(n):
            self.x[i] = UInt64(1) << UInt64(i)
            self.z[n + i] = UInt64(1) << UInt64(i)

    def apply(mut self, kind: Int, a: Int, b: Int):
        for r in range(2 * self.n):
            pauli_conj(kind, a, b, self.x[r], self.z[r], self.s[r])


def _symp_matrix(t: SignedTab) -> List[UInt64]:
    """Column map of the tableau: row = coordinate (x_0..x_{n-1}, z_0..), bit = generator."""
    var n = t.n
    var m = List[UInt64](length=2 * n, fill=0)
    for g in range(2 * n):
        for q in range(n):
            if _bit(t.x[g], q):
                m[q] |= UInt64(1) << UInt64(g)
            if _bit(t.z[g], q):
                m[n + q] |= UInt64(1) << UInt64(g)
    return m^


def gf2_inverse(rows: List[UInt64], n: Int, mut inv: List[UInt64]) -> Bool:
    """Gauss-Jordan inverse over GF(2) of an n x n matrix (row i = bit mask)."""
    var a = rows.copy()
    inv = List[UInt64](length=n, fill=0)
    for i in range(n):
        inv[i] = UInt64(1) << UInt64(i)
    for col in range(n):
        var piv = -1
        for r in range(col, n):
            if _bit(a[r], col):
                piv = r
                break
        if piv < 0:
            return False
        if piv != col:
            var t = a[piv]
            a[piv] = a[col]
            a[col] = t
            var u = inv[piv]
            inv[piv] = inv[col]
            inv[col] = u
        for r in range(n):
            if r != col and _bit(a[r], col):
                a[r] ^= a[col]
                inv[r] ^= inv[col]
    return True


# ============================================================ environments
struct PolicyEnv(Copyable):
    """Reference-gym environment semantics over a packed state (List[UInt64]).

    Layout: PERM: n words s[i]; LINEAR: n row masks; CLIFFORD: 2n row masks;
    PAULI: [2n tableau rows][R rx][R rz][R sign][1 count][R node list]."""

    var kind: Int
    var n: Int
    var act_k: List[Int]
    var act_a: List[Int]
    var act_b: List[Int]
    var R: Int  # number of rotations (Pauli)
    var max_rot: Int  # observed rotation columns (Pauli)
    var anti: List[Bool]  # R x R anticommutation (Pauli)

    def __init__(out self, kind: Int, n: Int, m: ModelFile):
        self.kind = kind
        self.n = n
        self.act_k = List[Int]()
        self.act_a = List[Int]()
        self.act_b = List[Int]()
        for i in range(m.num_actions()):
            self.act_k.append(m.actions[3 * i])
            self.act_a.append(m.actions[3 * i + 1])
            self.act_b.append(m.actions[3 * i + 2])
        self.R = 0
        self.max_rot = m.max_rotations
        self.anti = List[Bool]()

    def num_actions(self) -> Int:
        return len(self.act_k)

    def init_pauli(mut self, R: Int, st: List[UInt64]):
        """Set the rotation count and anticommutation table from a packed Pauli
        start state (rotation columns still in the input frame)."""
        self.R = R
        self.anti = List[Bool](length=R * R, fill=False)
        var bx = 2 * self.n
        var bz = 2 * self.n + R
        for i in range(R):
            for j in range(R):
                var c = pop_count((st[bx + i] & st[bz + j]) ^ (st[bz + i] & st[bx + j]))
                self.anti[i * R + j] = (c & 1) == 1

    # ---------------------------------------------------- Pauli layout helpers
    def _rx(self) -> Int:
        return 2 * self.n

    def _rz(self) -> Int:
        return 2 * self.n + self.R

    def _rs(self) -> Int:
        return 2 * self.n + 2 * self.R

    def _cnt(self) -> Int:
        return 2 * self.n + 3 * self.R

    def _nodes(self) -> Int:
        return 2 * self.n + 3 * self.R + 1

    # ---------------------------------------------------- core
    def solved(self, st: List[UInt64]) -> Bool:
        if self.kind == MODEL_KIND_PERM:
            for i in range(self.n):
                if st[i] != UInt64(i):
                    return False
            return True
        var rows = self.n if self.kind == MODEL_KIND_LINEAR else 2 * self.n
        for i in range(rows):
            if st[i] != UInt64(1) << UInt64(i):
                return False
        if self.kind == MODEL_KIND_PAULI:
            return st[self._cnt()] == 0
        return True

    def observe(self, st: List[UInt64], mut obs: List[Int]):
        obs.clear()
        var n = self.n
        if self.kind == MODEL_KIND_PERM:
            for i in range(n):
                obs.append(i * n + Int(st[i]))
            return
        if self.kind == MODEL_KIND_LINEAR:
            for i in range(n):
                for j in range(n):
                    if _bit(st[i], j):
                        obs.append(i * n + j)
            return
        var d = 2 * n
        if self.kind == MODEL_KIND_CLIFFORD:
            for i in range(d):
                for j in range(d):
                    if _bit(st[i], j):
                        obs.append(i * d + j)
            return
        # PAULI: dense rows of width C = 2n + max_rot
        var C = d + self.max_rot
        var cnt = Int(st[self._cnt()])
        var shown = min(cnt, self.max_rot)
        for r in range(d):
            for j in range(d):
                if _bit(st[r], j):
                    obs.append(r * C + j)
            for k in range(shown):
                var ri = Int(st[self._nodes() + k])
                var bits = st[self._rx() + ri] if r < n else st[self._rz() + ri]
                if _bit(bits, r if r < n else r - n):
                    obs.append(r * C + d + k)

    def _row_xor(self, mut st: List[UInt64], dst: Int, src: Int):
        st[dst] ^= st[src]

    def _row_swap(self, mut st: List[UInt64], a: Int, b: Int):
        var t = st[a]
        st[a] = st[b]
        st[b] = t

    def _cliff_rows(self, mut st: List[UInt64], kind: Int, a: Int, b: Int):
        """Clifford-env row operation for a gate (phases ignored)."""
        var n = self.n
        if kind == Op.H:
            self._row_swap(st, a, n + a)
        elif kind == Op.S or kind == Op.SDG:
            self._row_xor(st, n + a, a)
        elif kind == Op.SX or kind == Op.SXDG:
            self._row_xor(st, a, n + a)
        elif kind == Op.CX:
            self._row_xor(st, b, a)
            self._row_xor(st, n + a, n + b)
        elif kind == Op.CZ:
            self._row_xor(st, n + a, b)
            self._row_xor(st, n + b, a)
        elif kind == Op.SWAP:
            self._row_swap(st, a, b)
            self._row_swap(st, n + a, n + b)

    def step(self, st: List[UInt64], act: Int) -> List[UInt64]:
        var out = st.copy()
        var emit = List[Int]()
        self.apply(out, act, emit, False)
        return out^

    def apply(self, mut st: List[UInt64], act: Int, mut emit: List[Int], record: Bool):
        """Apply action `act` in place. With `record`, append emitted gates as
        triples (opcode, q0, q1), and emitted rotations as (-1 - rot, qubit,
        axis + 4 * sign_bit) (Pauli only)."""
        var k = self.act_k[act]
        var a = self.act_a[act]
        var b = self.act_b[act]
        if self.kind == MODEL_KIND_PERM:
            var t = st[a]
            st[a] = st[b]
            st[b] = t
            if record:
                emit.append(Op.SWAP)
                emit.append(a)
                emit.append(b)
            return
        if self.kind == MODEL_KIND_LINEAR:
            if k == Op.CX:
                st[b] ^= st[a]
            elif k == Op.SWAP:
                self._row_swap(st, a, b)
            if record:
                emit.append(k)
                emit.append(a)
                emit.append(b)
            return
        if self.kind == MODEL_KIND_CLIFFORD:
            self._cliff_rows(st, k, a, b)
            if record:
                emit.append(k)
                emit.append(a)
                emit.append(b)
            return
        # PAULI
        if k == Op.CX:
            self._pauli_cx(st, b, a, emit, record)
        elif k == Op.CZ:
            self._pauli_1q(st, Op.H, b, emit, False)
            self._pauli_cx(st, a, b, emit, False)
            self._pauli_1q(st, Op.H, b, emit, False)
            if record:
                emit.append(Op.CZ)
                emit.append(a)
                emit.append(b)
            self._clean(st, emit, record)
        elif k == Op.SWAP:
            self._pauli_cx(st, b, a, emit, record)
            self._pauli_cx(st, a, b, emit, record)
            self._pauli_cx(st, b, a, emit, record)
        else:
            self._pauli_1q(st, k, a, emit, record)

    def _pauli_1q(self, mut st: List[UInt64], k: Int, q: Int, mut emit: List[Int], record: Bool):
        self._cliff_rows(st, k, q, -1)
        for ri in range(self.R):
            var x = st[self._rx() + ri]
            var z = st[self._rz() + ri]
            var s = Int(st[self._rs() + ri])
            pauli_conj(k, q, -1, x, z, s)
            st[self._rx() + ri] = x
            st[self._rz() + ri] = z
            st[self._rs() + ri] = UInt64(s)
        if record:
            emit.append(k)
            emit.append(q)
            emit.append(-1)

    def _pauli_cx(self, mut st: List[UInt64], c: Int, t: Int, mut emit: List[Int], record: Bool):
        """Physical CX(c, t) on tableau and rotations, then clean (emits when recording)."""
        self._cliff_rows(st, Op.CX, c, t)
        for ri in range(self.R):
            var x = st[self._rx() + ri]
            var z = st[self._rz() + ri]
            var s = Int(st[self._rs() + ri])
            pauli_conj(Op.CX, c, t, x, z, s)
            st[self._rx() + ri] = x
            st[self._rz() + ri] = z
            st[self._rs() + ri] = UInt64(s)
        if record:
            emit.append(Op.CX)
            emit.append(c)
            emit.append(t)
        self._clean(st, emit, record)

    def _clean(self, mut st: List[UInt64], mut emit: List[Int], record: Bool):
        """Reference-gym `clean_and_return_with_phases`: emit + remove trivial
        front-layer rotations until none is left (petgraph swap-remove order)."""
        var n = self.n
        var removed = True
        while removed:
            removed = False
            var cnt = Int(st[self._cnt()])
            var kill = List[Bool](length=cnt, fill=False)
            for pos in range(cnt):
                var ri = Int(st[self._nodes() + pos])
                var front = True
                for p2 in range(cnt):
                    var rj = Int(st[self._nodes() + p2])
                    if rj < ri and self.anti[ri * self.R + rj]:
                        front = False
                        break
                if not front:
                    continue
                var x = st[self._rx() + ri]
                var z = st[self._rz() + ri]
                var w = pop_count(x | z)
                if w <= 1:
                    if record and w == 1:
                        var q = 0
                        while not _bit(x | z, q):
                            q += 1
                        var axis = 2
                        if _bit(x, q):
                            axis = 1 if _bit(z, q) else 0
                        emit.append(-1 - ri)
                        emit.append(q)
                        emit.append(axis + 4 * Int(st[self._rs() + ri]))
                    st[self._rx() + ri] = 0
                    st[self._rz() + ri] = 0
                    kill[pos] = True
                    removed = True
            if removed:
                # retain_nodes: reverse index order, swap_remove
                var last = cnt
                for pos in range(cnt - 1, -1, -1):
                    if kill[pos]:
                        last -= 1
                        st[self._nodes() + pos] = st[self._nodes() + last]
                        st[self._nodes() + last] = 0
                st[self._cnt()] = UInt64(last)
        _ = n

    def inverse(self, st: List[UInt64], mut out: List[UInt64]) -> Bool:
        """State of U^-1 (permutation / linear / Clifford only)."""
        if self.kind == MODEL_KIND_PERM:
            out = List[UInt64](length=self.n, fill=0)
            for i in range(self.n):
                out[Int(st[i])] = UInt64(i)
            return True
        if self.kind == MODEL_KIND_LINEAR:
            return gf2_inverse(st, self.n, out)
        if self.kind == MODEL_KIND_CLIFFORD:
            return gf2_inverse(st, 2 * self.n, out)
        return False

    def key(self, st: List[UInt64]) -> UInt64:
        return fnv_words(st)


# ============================================================ search
def _order(logits: List[Float32], off: Int, A: Int, mut lp: List[Float64], mut order: List[Int]):
    """Log-softmax of logits[off:off+A] and the action order (descending;
    |delta| <= 1e-6 * max(1, |l|) is a tie -> lower index first)."""
    var mx = Float64(logits[off])
    for a in range(A):
        mx = max(mx, Float64(logits[off + a]))
    var se = 0.0
    for a in range(A):
        se += exp(Float64(logits[off + a]) - mx)
    var lse = mx + log(se)
    lp = List[Float64](length=A, fill=0.0)
    for a in range(A):
        lp[a] = Float64(logits[off + a]) - lse
    order = List[Int]()
    for a in range(A):
        var la = logits[off + a]
        var pos = len(order)
        while pos > 0:
            var lb = logits[off + order[pos - 1]]
            var tol = Float32(1e-6) * max(Float32(1.0), abs(lb))
            if lb - la >= -tol:
                break
            pos -= 1
        order.insert(pos, a)


def _logits(env: PolicyEnv, pol: MLPPolicy, states: List[List[UInt64]], ensemble: Bool, mut flat: List[Float32]):
    """Policy logits for every state, flat [len(states), A] (identity frame or
    symmetry ensemble). Identity forwards are batched."""
    var A = pol.num_actions
    if ensemble and pol.num_symmetries() > 0:
        flat = List[Float32](capacity=len(states) * A)
        var obs = List[Int]()
        var lg = List[Float32]()
        for st in states:
            env.observe(st, obs)
            _ = pol.forward_ensemble(obs, lg)
            for a in range(A):
                flat.append(lg[a])
        return
    var obsl = List[List[Int]]()
    var perms = List[Int]()
    for st in states:
        var obs = List[Int]()
        env.observe(st, obs)
        obsl.append(obs^)
        perms.append(-1)
    var vals = List[Float32]()
    pol.raw_batch(obsl, perms, flat, vals)


def _logp(env: PolicyEnv, pol: MLPPolicy, st: List[UInt64], ensemble: Bool, mut lp: List[Float64], mut order: List[Int]):
    var sts = List[List[UInt64]]()
    sts.append(st.copy())
    var flat = List[Float32]()
    _logits(env, pol, sts, ensemble, flat)
    _order(flat, 0, pol.num_actions, lp, order)


def _beam(env: PolicyEnv, pol: MLPPolicy, max_depth: Int, start: List[UInt64], width: Int, ensemble: Bool,
          mut sols: List[List[Int]]):
    """Beam search on cumulative log-softmax with duplicate-state removal and a
    global visited set; appends every solution of the first solving depth."""
    if env.solved(start):
        sols.append(List[Int]())
        return
    var visited = Dict[UInt64, Bool]()
    visited[env.key(start)] = True
    var f_state = List[List[UInt64]]()
    var f_score = List[Float64]()
    var f_acts = List[List[Int]]()
    f_state.append(start.copy())
    f_score.append(0.0)
    f_acts.append(List[Int]())
    var lp = List[Float64]()
    var order = List[Int]()
    for _ in range(max_depth):
        var c_state = List[List[UInt64]]()
        var c_score = List[Float64]()
        var c_acts = List[List[Int]]()
        var c_key = Dict[UInt64, Int]()
        var found = False
        var flat = List[Float32]()
        _logits(env, pol, f_state, ensemble, flat)
        for fi in range(len(f_state)):
            _order(flat, fi * pol.num_actions, pol.num_actions, lp, order)
            var taken = 0
            for oi in range(len(order)):
                if taken >= width:
                    break
                var a = order[oi]
                var ns = env.step(f_state[fi], a)
                if env.solved(ns):
                    var acts = f_acts[fi].copy()
                    acts.append(a)
                    sols.append(acts^)
                    found = True
                    taken += 1
                    continue
                var k = env.key(ns)
                if k in visited:
                    continue
                var sc = f_score[fi] + lp[a]
                var idx = c_key.get(k, -1)
                if idx >= 0:
                    if sc > c_score[idx]:
                        c_score[idx] = sc
                        var acts = f_acts[fi].copy()
                        acts.append(a)
                        c_acts[idx] = acts^
                else:
                    c_key[k] = len(c_state)
                    c_state.append(ns^)
                    c_score.append(sc)
                    var acts = f_acts[fi].copy()
                    acts.append(a)
                    c_acts.append(acts^)
                taken += 1
        if found or len(c_state) == 0:
            return
        var idxs = List[Int]()
        for i in range(len(c_state)):
            var pos = len(idxs)
            while pos > 0 and c_score[idxs[pos - 1]] < c_score[i]:
                pos -= 1
            idxs.insert(pos, i)
        for i in range(len(c_state)):
            visited[env.key(c_state[i])] = True
        f_state = List[List[UInt64]]()
        f_score = List[Float64]()
        f_acts = List[List[Int]]()
        for j in range(min(width, len(idxs))):
            var i = idxs[j]
            f_state.append(c_state[i].copy())
            f_score.append(c_score[i])
            f_acts.append(c_acts[i].copy())


def greedy_trace(env: PolicyEnv, pol: MLPPolicy, max_depth: Int, start: List[UInt64], mut solved: Bool) -> List[Int]:
    """The path of the width-1 ensemble beam (greedy, visited states skipped),
    returned even when it does not solve (parity reference for refenv.greedy_trace)."""
    var st = start.copy()
    var visited = Dict[UInt64, Bool]()
    visited[env.key(st)] = True
    var acts = List[Int]()
    var lp = List[Float64]()
    var order = List[Int]()
    solved = False
    for _ in range(max_depth):
        if env.solved(st):
            solved = True
            return acts^
        _logp(env, pol, st, True, lp, order)
        var pick = -1
        for oi in range(len(order)):
            var ns = env.step(st, order[oi])
            if env.solved(ns) or not (env.key(ns) in visited):
                pick = order[oi]
                st = ns^
                break
        if pick < 0:
            return acts^
        acts.append(pick)
        visited[env.key(st)] = True
    solved = env.solved(st)
    return acts^


def policy_rollout(env: PolicyEnv, pol: MLPPolicy, max_depth: Int, start: List[UInt64]) -> List[Int]:
    """Deterministic greedy rollout with the symmetry ensemble and a visited set
    (beam width 1); the reference behaviour checked by the manifests' rollouts."""
    var sols = List[List[Int]]()
    _beam(env, pol, max_depth, start, 1, True, sols)
    if len(sols) == 0:
        return List[Int]()
    return sols[0].copy()


