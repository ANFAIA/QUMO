"""Policy-guided, topology-aware block synthesis (permutation, linear function,
Clifford, Pauli network).

Native re-implementation of the inference side of qiskit-ibm-transpiler's
AI*Synthesis passes (arXiv:2405.13196, Pauli networks arXiv:2503.14448) with
QUMO's own models (see `policy.mojo`). The environments reproduce qiskit-gym
0.4.1's Rust envs (Apache-2.0) exactly, so a model trained there sees the same
observations here:

* permutation: state s (s = Perm.p of the block, i.e. qiskit-gym's
  argsort(pattern)); SWAP(a, b) swaps s[a], s[b]; obs = {i*n + s[i]};
  solved when s is the identity. Emits SWAP(a, b).
* linear function: state = M^-1 (GF(2), row i = bit mask); CX(c, t):
  row t ^= row c; obs = {i*n + j : bit j of row i}; solved at I. Emits CX(c, t).
* Clifford: state = the 2n x 2n symplectic matrix of C^-1 (rows = Pauli
  coordinates x_0..x_{n-1}, z_0..z_{n-1}; columns = generators), phases
  ignored; gates left-multiply (H: swap rows q, n+q; S/Sdg: row n+q ^= row q;
  SX/SXdg: row q ^= row n+q; CX(c,t): row t ^= row c, row n+c ^= row n+t;
  CZ(a,b): row n+a ^= row b, row n+b ^= row a; SWAP: swap both pairs).
  Signs are fixed afterwards with one Pauli layer (qiskit-gym `_solve_phases`).
* Pauli network: the Clifford state above plus one column per rotation (the
  rotation's Pauli in the current frame). Rotations form a DAG (an edge when
  two rotations anticommute, later -> earlier); after every CX-type action
  every front-layer rotation of weight <= 1 is emitted as a 1-qubit rotation
  and removed, repeated to a fixpoint (`clean_and_return_with_phases`). The
  gym action CX(i, j) performs row i ^= row j, row n+j ^= row n+i, i.e. the
  physical gate CX(control j, target i) (its Python side reverses the qargs);
  CZ = H.CX.H and SWAP = 3 alternating CX, with cleaning after each CX. The
  observation is the 2n x (2n + max_rotations) matrix with the first
  max_rotations active rotations in petgraph node order (swap-remove).
  Signs are tracked exactly (Aaronson-Gottesman conjugation rules), so the
  emitted rotation angle is sign * theta of the source rotation.

Search (deterministic, seeded; twisterl's own inference is not seedable):
beam search of width `beams` on cumulative log-softmax with duplicate-state
removal and a global visited set (width 1 = greedy with the symmetry
ensemble); plus `searches` sampled rollouts with Rng(seed).split(k); plus the
inversion trick (solve U and U^-1, reverse the latter) for permutation /
linear / Clifford; step cap = the model's max_depth. Tie-break: highest
logit, |delta| <= 1e-6 * max(1, |l|) counts as a tie, then the lowest action
index. The cheapest successful sequence wins (2q cost: CX = CZ = 1,
SWAP = 3; then gate count; then search order).

Every result is re-verified here (permutation / matrix / signed tableau /
rotation list) and must use only edges of the block graph; anything else is
dropped (empty list = no proposal). Callers verify again before splicing.
"""

from std.collections import List, Dict
from std.math import exp, log
from std.bit import pop_count
from qumo.ir.circuit import Instruction
from qumo.ir.gates import Op
from qumo.sim.rng import Rng
from qumo.compiler.blockalg import (
    SubGraph, BlockKind, Perm, GF2Mat, SympTab, PauliRot, PauliNet, emit_rotation,
)
from qumo.compiler.policy import (
    ModelFile, MLPPolicy, ModelMatch, load_model_file, find_model, find_model_rows,
    ModelIndexEntry, load_synth_index,
    MODEL_KIND_CLIFFORD, MODEL_KIND_LINEAR, MODEL_KIND_PERM, MODEL_KIND_PAULI,
    model_kind_name, fnv_words,
)


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
    """qiskit-gym environment semantics over a packed state (List[UInt64]).

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
        """qiskit-gym `clean_and_return_with_phases`: emit + remove trivial
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


def _sample(env: PolicyEnv, pol: MLPPolicy, max_depth: Int, start: List[UInt64], seed: Int, k: Int,
            mut sols: List[List[Int]]):
    """One sampled rollout (softmax policy, visited states masked), Rng(seed).split(k)."""
    var rng = Rng(UInt64(seed)).split(k)
    var st = start.copy()
    var visited = Dict[UInt64, Bool]()
    visited[env.key(st)] = True
    var acts = List[Int]()
    var lp = List[Float64]()
    var order = List[Int]()
    for _ in range(max_depth):
        if env.solved(st):
            sols.append(acts^)
            return
        _logp(env, pol, st, False, lp, order)
        var A = len(lp)
        var w = List[Float64](length=A, fill=0.0)
        var nxt = List[List[UInt64]]()
        var tot = 0.0
        for a in range(A):
            var ns = env.step(st, a)
            if env.solved(ns) or not (env.key(ns) in visited):
                w[a] = exp(lp[a])
                tot += w[a]
            nxt.append(ns^)
        if tot <= 0.0:
            return
        var u = rng.uniform() * tot
        var pick = -1
        var acc = 0.0
        for oi in range(A):
            var a = order[oi]
            if w[a] <= 0.0:
                continue
            acc += w[a]
            pick = a
            if u < acc:
                break
        st = nxt[pick].copy()
        visited[env.key(st)] = True
        acts.append(pick)
    if env.solved(st):
        sols.append(acts^)


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


# ============================================================ driver helpers
def _inverse_gate(k: Int) -> Int:
    if k == Op.S:
        return Op.SDG
    if k == Op.SDG:
        return Op.S
    if k == Op.SX:
        return Op.SXDG
    if k == Op.SXDG:
        return Op.SX
    return k


def _gate_cost2(k: Int) -> Int:
    if k == Op.SWAP:
        return 3
    if k == Op.CX or k == Op.CZ:
        return 1
    return 0


def _emit_cost(emit: List[Int]) -> Tuple[Int, Int]:
    var c2 = 0
    var ng = 0
    for i in range(len(emit) // 3):
        if emit[3 * i] >= 0:
            c2 += _gate_cost2(emit[3 * i])
            ng += 1
    return (c2, ng)


def _better(a: Tuple[Int, Int], b: Tuple[Int, Int]) -> Bool:
    if a[0] != b[0]:
        return a[0] < b[0]
    return a[1] < b[1]


def search_best(env: PolicyEnv, pol: MLPPolicy, max_depth: Int, start: List[UInt64], beams: Int, searches: Int,
                seed: Int, allow_inverse: Bool, mut best: List[Int]) -> Bool:
    """Run every search mode on `start` (and on the inverse when allowed) and
    return the cheapest solution as emitted gate triples (model labels)."""
    var starts = List[List[UInt64]]()
    starts.append(start.copy())
    if allow_inverse:
        var inv = List[UInt64]()
        if env.inverse(start, inv):
            starts.append(inv^)
    var found = False
    var bc = (0, 0)
    var cap = max_depth  # depth cap, tightened by the best solution so far
    var plain = env.kind == MODEL_KIND_PERM or env.kind == MODEL_KIND_LINEAR
    var w = max(beams, 1)
    # runs: beam on every start first, then the sampled rollouts
    var run_si = List[Int]()
    var run_k = List[Int]()
    for si in range(len(starts)):
        run_si.append(si)
        run_k.append(-1)
    for k in range(max(searches, 0)):
        for si in range(len(starts)):
            run_si.append(si)
            run_k.append(k)
    for ri in range(len(run_si)):
        var si = run_si[ri]
        var sols = List[List[Int]]()
        if cap <= 0:
            break
        if run_k[ri] < 0:
            _beam(env, pol, cap, starts[si], w, w == 1, sols)
        else:
            _sample(env, pol, cap, starts[si], seed + 7919 * si, run_k[ri], sols)
        for s in sols:
            var st = starts[si].copy()
            var emit = List[Int]()
            for a in s:
                env.apply(st, a, emit, True)
            if not env.solved(st):
                continue
            var c = _emit_cost(emit)
            if found and not _better(c, bc):
                continue
            found = True
            bc = c
            # perm / linear: every action costs the same, so only strictly
            # shorter sequences can win; Clifford / Pauli: 1q actions are free
            var L = len(s)
            cap = min(cap, L - 1 if plain else L + L // 2)
            best = List[Int]()
            if si == 1:
                var m = len(emit) // 3
                for i in range(m - 1, -1, -1):
                    best.append(_inverse_gate(emit[3 * i]))
                    best.append(emit[3 * i + 1])
                    best.append(emit[3 * i + 2])
            else:
                best = emit^
    return found


struct PolicySession(Movable):
    """Model cache for many blocks: loaded models / policies are kept by
    directory. The frozen one-shot functions (`policy_synth_*`) create a
    throw-away session per call; passes should keep one session per run."""

    var dirs: List[String]
    var rows: List[ModelIndexEntry]
    var model_dirs: List[String]
    var models: List[ModelFile]
    var policies: List[MLPPolicy]

    def __init__(out self, dirs: List[String]):
        self.dirs = dirs.copy()
        self.rows = load_synth_index(dirs)
        self.model_dirs = List[String]()
        self.models = List[ModelFile]()
        self.policies = List[MLPPolicy]()

    def available(self, kind: Int, g: SubGraph, model: String) -> Bool:
        """True when a model of `kind` matches `g` (no loading, no raising)."""
        if model == "off":
            return False
        return find_model_rows(kind, g, model, self.rows).ok

    def load(mut self, dir: String) raises -> Int:
        for i in range(len(self.model_dirs)):
            if self.model_dirs[i] == dir:
                return i
        var m = load_model_file(dir)
        var p = MLPPolicy.from_model(m)
        self.model_dirs.append(dir)
        self.models.append(m^)
        self.policies.append(p^)
        return len(self.model_dirs) - 1

    def match(self, kind: Int, g: SubGraph, model: String) raises -> ModelMatch:
        """find_model; QM-E616 when an explicit model id does not match."""
        if model == "off":
            return ModelMatch()
        var mm = find_model_rows(kind, g, model, self.rows)
        if not mm.ok and model != "auto":
            raise Error("QM-E616: unknown or incompatible synthesis model '" + model + "' for a " + String(g.n) + "-qubit " + model_kind_name(kind) + " block. Hint: use model=auto, model=off, or a model id listed in <model_dir>/synth/INDEX.json whose graph embeds into the block graph.")
        return mm^

    def loaded_index(self, dir: String) -> Int:
        for i in range(len(self.model_dirs)):
            if self.model_dirs[i] == dir:
                return i
        return -1

    def preload(mut self, kind: Int) raises -> Int:
        """Load every indexed model of `kind` (call before using the read-only
        `propose_*` methods from parallel workers). Returns the count."""
        var c = 0
        for r in self.rows:
            if r.kind_code == kind:
                _ = self.load(r.path)
                c += 1
        return c

    def env_for(self, kind: Int, mm: ModelMatch, mut mi: Int) raises -> PolicyEnv:
        """Environment for a matched, already-loaded model (mi = -1 when not loaded)."""
        mi = self.loaded_index(mm.dir)
        if mi < 0:
            return PolicyEnv(kind, 0, ModelFile())
        if self.models[mi].kind_code != kind:
            raise Error("QM-E620: model '" + mm.id + "' is a " + self.models[mi].kind + " model, not " + model_kind_name(kind) + ". Hint: fix INDEX.json or pick a model of the right kind.")
        if self.models[mi].num_qubits != len(mm.perm):
            raise Error("QM-E620: model '" + mm.id + "' has " + String(self.models[mi].num_qubits) + " qubits, the block " + String(len(mm.perm)) + ". Hint: fix INDEX.json (num_qubits) or re-export the model.")
        var env = PolicyEnv(kind, self.models[mi].num_qubits, self.models[mi])
        for i in range(env.num_actions()):
            var a = env.act_a[i]
            var b = env.act_b[i]
            if a < 0 or a >= env.n or b >= env.n:
                raise Error("QM-E620: model '" + mm.id + "' has an action on a qubit outside 0.." + String(env.n - 1) + ". Hint: re-export the model.")
            if kind == MODEL_KIND_PERM and env.act_k[i] != Op.SWAP:
                raise Error("QM-E620: permutation model '" + mm.id + "' has a non-SWAP action. Hint: re-export the model.")
            if kind == MODEL_KIND_LINEAR and env.act_k[i] != Op.CX and env.act_k[i] != Op.SWAP:
                raise Error("QM-E620: linear-function model '" + mm.id + "' has an action outside {CX, SWAP}. Hint: re-export the model.")
        if kind == MODEL_KIND_PAULI and self.models[mi].max_rotations <= 0:
            raise Error("QM-E620: Pauli-network model '" + mm.id + "' has no obs.max_rotations. Hint: re-export the model.")
        return env^

    # ------------------------------------------------------------ loading wrappers
    def _ensure(mut self, kind: Int, g: SubGraph, model: String) raises:
        var mm = self.match(kind, g, model)
        if mm.ok:
            _ = self.load(mm.dir)

    def synth_perm(mut self, p: Perm, g: SubGraph, model: String, beams: Int, searches: Int,
                   seed: Int) raises -> List[Instruction]:
        """Policy proposal for a permutation block (loads the model on first use)."""
        self._ensure(MODEL_KIND_PERM, g, model)
        return self.propose_perm(p, g, model, beams, searches, seed)

    def synth_linear(mut self, mat: GF2Mat, g: SubGraph, model: String, beams: Int, searches: Int,
                     seed: Int) raises -> List[Instruction]:
        self._ensure(MODEL_KIND_LINEAR, g, model)
        return self.propose_linear(mat, g, model, beams, searches, seed)

    def synth_clifford(mut self, t: SympTab, g: SubGraph, model: String, beams: Int, searches: Int,
                       seed: Int) raises -> List[Instruction]:
        self._ensure(MODEL_KIND_CLIFFORD, g, model)
        return self.propose_clifford(t, g, model, beams, searches, seed)

    def synth_pauli_network(mut self, pn: PauliNet, g: SubGraph, model: String, beams: Int, searches: Int,
                            seed: Int, src_ops: List[Instruction]) raises -> List[Instruction]:
        self._ensure(MODEL_KIND_PAULI, g, model)
        return self.propose_pauli_network(pn, g, model, beams, searches, seed, src_ops)

    # ------------------------------------------------------------ per kind (read-only)
    def propose_perm(self, p: Perm, g: SubGraph, model: String, beams: Int, searches: Int,
                   seed: Int) raises -> List[Instruction]:
        var n = len(p.p)
        if n != g.n or n < 2:
            return List[Instruction]()
        var mm = self.match(MODEL_KIND_PERM, g, model)
        if not mm.ok:
            return List[Instruction]()
        var sigma = mm.perm.copy()
        var sinv = _inv_perm(sigma)
        var start = List[UInt64](length=n, fill=0)
        for m in range(n):
            start[m] = UInt64(sinv[p.p[sigma[m]]])
        var mi = 0
        var env = self.env_for(MODEL_KIND_PERM, mm, mi)
        if mi < 0:
            return List[Instruction]()
        var gates = List[Int]()
        if not search_best(env, self.policies[mi], self.models[mi].max_depth, start, beams, searches, seed, True, gates):
            return List[Instruction]()
        var ops = _to_block_ops(gates, sigma)
        if not _on_edges(ops, g) or not _perm_matches(ops, p.p):
            return List[Instruction]()
        return ops^

    def propose_linear(self, mat: GF2Mat, g: SubGraph, model: String, beams: Int, searches: Int,
                     seed: Int) raises -> List[Instruction]:
        var n = mat.n
        if n != g.n or n < 2:
            return List[Instruction]()
        var mm = self.match(MODEL_KIND_LINEAR, g, model)
        if not mm.ok:
            return List[Instruction]()
        var sigma = mm.perm.copy()
        var sinv = _inv_perm(sigma)
        var rel = List[UInt64](length=n, fill=0)
        for a in range(n):
            rel[a] = _relabel_mask(mat.rows[sigma[a]], sinv)
        var start = List[UInt64]()
        if not gf2_inverse(rel, n, start):
            return List[Instruction]()
        var mi = 0
        var env = self.env_for(MODEL_KIND_LINEAR, mm, mi)
        if mi < 0:
            return List[Instruction]()
        var gates = List[Int]()
        if not search_best(env, self.policies[mi], self.models[mi].max_depth, start, beams, searches, seed, True, gates):
            return List[Instruction]()
        var ops = _to_block_ops(gates, sigma)
        if not _on_edges(ops, g) or not _linear_matches(ops, mat.rows, n):
            return List[Instruction]()
        return ops^

    def propose_clifford(self, t: SympTab, g: SubGraph, model: String, beams: Int, searches: Int,
                       seed: Int) raises -> List[Instruction]:
        var n = len(t.x) // 2
        if n != g.n or n < 1:
            return List[Instruction]()
        var mm = self.match(MODEL_KIND_CLIFFORD, g, model)
        if not mm.ok:
            return List[Instruction]()
        var sigma = mm.perm.copy()
        var sinv = _inv_perm(sigma)
        var target = _tab_from_symptab(t, sinv, sigma)
        var start = List[UInt64]()
        if not gf2_inverse(_symp_matrix(target), 2 * n, start):
            return List[Instruction]()
        var mi = 0
        var env = self.env_for(MODEL_KIND_CLIFFORD, mm, mi)
        if mi < 0:
            return List[Instruction]()
        var gates = List[Int]()
        if not search_best(env, self.policies[mi], self.models[mi].max_depth, start, beams, searches, seed, True, gates):
            return List[Instruction]()
        if not _pauli_fix(target, gates, n):
            return List[Instruction]()
        var ops = _to_block_ops(gates, sigma)
        if not _on_edges(ops, g) or not _clifford_matches(ops, t, n):
            return List[Instruction]()
        return ops^

    def propose_pauli_network(self, pn: PauliNet, g: SubGraph, model: String, beams: Int, searches: Int,
                            seed: Int, src_ops: List[Instruction]) raises -> List[Instruction]:
        var n = pn.n
        var R = len(pn.rots)
        if n != g.n or n < 1:
            return List[Instruction]()
        var mm = self.match(MODEL_KIND_PAULI, g, model)
        if not mm.ok:
            return List[Instruction]()
        var sigma = mm.perm.copy()
        var sinv = _inv_perm(sigma)
        var target = _tab_from_symptab(pn.final, sinv, sigma)
        var tab = List[UInt64]()
        if not gf2_inverse(_symp_matrix(target), 2 * n, tab):
            return List[Instruction]()
        var mi = 0
        var env = self.env_for(MODEL_KIND_PAULI, mm, mi)
        if mi < 0:
            return List[Instruction]()
        env.R = R
        env.anti = List[Bool](length=R * R, fill=False)
        for i in range(R):
            for j in range(R):
                var c = pop_count((pn.rots[i].x & pn.rots[j].z) ^ (pn.rots[i].z & pn.rots[j].x))
                env.anti[i * R + j] = (c & 1) == 1
        var start = tab^
        for i in range(R):
            start.append(_relabel_mask(pn.rots[i].x, sinv))
        for i in range(R):
            start.append(_relabel_mask(pn.rots[i].z, sinv))
        for _ in range(R):
            start.append(0)
        start.append(UInt64(R))
        for i in range(R):
            start.append(UInt64(i))
        var pre = List[Int]()
        env._clean(start, pre, True)
        var gates = List[Int]()
        if not search_best(env, self.policies[mi], self.models[mi].max_depth, start, beams, searches, seed, False, gates):
            return List[Instruction]()
        var all = pre^
        for v in gates:
            all.append(v)
        if not _pauli_fix(target, all, n):
            return List[Instruction]()
        var ops = List[Instruction]()
        var rot_of = List[Int]()
        var rels = List[Int]()
        var emitted = List[Bool](length=R, fill=False)
        for i in range(len(all) // 3):
            var k = all[3 * i]
            if k >= 0:
                var b = all[3 * i + 2]
                if b >= 0:
                    ops.append(Instruction.make(k, sigma[all[3 * i + 1]], sigma[b]))
                else:
                    ops.append(Instruction.make(k, sigma[all[3 * i + 1]]))
                rot_of.append(-1)
                rels.append(0)
            else:
                var ri = -1 - k
                var q = sigma[all[3 * i + 1]]
                var v = all[3 * i + 2]
                var axis = v & 3
                var rel = -1 if (v >> 2) == 1 else 1
                if emitted[ri]:
                    return List[Instruction]()
                emitted[ri] = True
                ops.append(emit_rotation(pn.rots[ri], src_ops, q, axis, rel))
                rot_of.append(ri)
                rels.append(v >> 2)
        for i in range(R):
            if not emitted[i]:
                return List[Instruction]()
        if not _on_edges(ops, g) or not _pauli_net_matches(ops, rot_of, rels, pn, n):
            return List[Instruction]()
        return ops^


# ============================================================ verification
def _perm_matches(ops: List[Instruction], p: List[Int]) -> Bool:
    """Perm convention: after the circuit the state of qubit i is on qubit p[i]."""
    var n = len(p)
    var loc = List[Int](length=n, fill=0)
    var who = List[Int](length=n, fill=0)
    for i in range(n):
        loc[i] = i
        who[i] = i
    for op in ops:
        if op.kind != Op.SWAP:
            return False
        var a = op.q0
        var b = op.q1
        var wa = who[a]
        var wb = who[b]
        who[a] = wb
        who[b] = wa
        loc[wa] = b
        loc[wb] = a
    for i in range(n):
        if loc[i] != p[i]:
            return False
    return True


def _linear_matches(ops: List[Instruction], rows: List[UInt64], n: Int) -> Bool:
    var m = List[UInt64](length=n, fill=0)
    for i in range(n):
        m[i] = UInt64(1) << UInt64(i)
    for op in ops:
        if op.kind == Op.CX:
            m[op.q1] ^= m[op.q0]
        elif op.kind == Op.SWAP:
            var t = m[op.q0]
            m[op.q0] = m[op.q1]
            m[op.q1] = t
        else:
            return False
    for i in range(n):
        if m[i] != rows[i]:
            return False
    return True


def _clifford_matches(ops: List[Instruction], t: SympTab, n: Int) -> Bool:
    var s = SignedTab(n)
    for op in ops:
        s.apply(op.kind, op.q0, op.q1)
    for r in range(2 * n):
        if s.x[r] != t.x[r] or s.z[r] != t.z[r] or s.s[r] != (Int(t.r[r]) & 1):
            return False
    return True


def _pauli_net_matches(ops: List[Instruction], rot_of: List[Int], rels: List[Int], pn: PauliNet, n: Int) -> Bool:
    """Clifford part (ops with rot_of == -1) equals pn.final (signed); rotation
    rot_of[i], conjugated forward through the Clifford gates before op i, is
    the emitted single-qubit Pauli with relative sign bit rels[i]; every
    rotation is emitted once and anticommuting rotations keep their order."""
    var s = SignedTab(n)
    var R = len(pn.rots)
    var pos = List[Int](length=R, fill=-1)
    var heads = List[Int]()  # indices of Clifford ops so far
    for i in range(len(ops)):
        var op = ops[i]
        var ri = rot_of[i]
        if ri < 0:
            s.apply(op.kind, op.q0, op.q1)
            heads.append(i)
            continue
        if ri >= R or pos[ri] >= 0:
            return False
        pos[ri] = i
        var x = pn.rots[ri].x
        var z = pn.rots[ri].z
        var sg = 0
        for j in heads:
            pauli_conj(ops[j].kind, ops[j].q0, ops[j].q1, x, z, sg)
        var k = op.kind
        var q = op.q0
        var want_x = UInt64(1) << UInt64(q) if (k == Op.RX or k == Op.RY) else UInt64(0)
        var want_z = UInt64(1) << UInt64(q) if (k == Op.RZ or k == Op.RY) else UInt64(0)
        if x != want_x or z != want_z or sg != rels[i]:
            return False
    for ri in range(R):
        if pos[ri] < 0:
            return False
        for rj in range(ri):
            var c = pop_count((pn.rots[ri].x & pn.rots[rj].z) ^ (pn.rots[ri].z & pn.rots[rj].x))
            if (c & 1) == 1 and pos[rj] > pos[ri]:
                return False
    var t = pn.final.copy()
    for r in range(2 * n):
        if s.x[r] != t.x[r] or s.z[r] != t.z[r] or s.s[r] != (Int(t.r[r]) & 1):
            return False
    return True


def _on_edges(ops: List[Instruction], g: SubGraph) -> Bool:
    for op in ops:
        if op.kind == Op.CX or op.kind == Op.CZ or op.kind == Op.SWAP:
            var a = min(op.q0, op.q1)
            var b = max(op.q0, op.q1)
            var ok = False
            for e in range(len(g.edges) // 2):
                if g.edges[2 * e] == a and g.edges[2 * e + 1] == b:
                    ok = True
                    break
            if not ok:
                return False
    return True


def _inv_perm(p: List[Int]) -> List[Int]:
    var q = List[Int](length=len(p), fill=0)
    for i in range(len(p)):
        q[p[i]] = i
    return q^


def _relabel_mask(m: UInt64, sigma_inv: List[Int]) -> UInt64:
    """Block-qubit bit mask -> model-qubit bit mask."""
    var out = UInt64(0)
    for q in range(len(sigma_inv)):
        if _bit(m, q):
            out |= UInt64(1) << UInt64(sigma_inv[q])
    return out


def _to_block_ops(gates: List[Int], sigma: List[Int]) -> List[Instruction]:
    var out = List[Instruction]()
    for i in range(len(gates) // 3):
        var k = gates[3 * i]
        var a = sigma[gates[3 * i + 1]]
        var b = gates[3 * i + 2]
        if b >= 0:
            out.append(Instruction.make(k, a, sigma[b]))
        else:
            out.append(Instruction.make(k, a))
    return out^


def _pauli_fix(target: SignedTab, mut gates: List[Int], n: Int) -> Bool:
    """Append to `gates` the Pauli layer P with (gates then P) == target (signed).
    Returns False when the unsigned tableaus differ."""
    var t = SignedTab(n)
    var m0 = len(gates) // 3
    for i in range(m0):
        if gates[3 * i] >= 0:
            t.apply(gates[3 * i], gates[3 * i + 1], gates[3 * i + 2])
    var px = UInt64(0)
    var pz = UInt64(0)
    for r in range(2 * n):
        if t.x[r] != target.x[r] or t.z[r] != target.z[r]:
            return False
    for i in range(n):
        if t.s[n + i] != target.s[n + i]:  # Z_i row sign differs -> include image of X_i
            px ^= t.x[i]
            pz ^= t.z[i]
        if t.s[i] != target.s[i]:  # X_i row sign differs -> include image of Z_i
            px ^= t.x[n + i]
            pz ^= t.z[n + i]
    for q in range(n):
        var xq = _bit(px, q)
        var zq = _bit(pz, q)
        if xq and zq:
            gates.append(Op.Y)
        elif xq:
            gates.append(Op.X)
        elif zq:
            gates.append(Op.Z)
        else:
            continue
        gates.append(q)
        gates.append(-1)
        t.apply(gates[len(gates) - 3], q, -1)
    for r in range(2 * n):
        if t.s[r] != target.s[r]:
            return False
    return True


def _tab_from_symptab(t: SympTab, sigma_inv: List[Int], sigma: List[Int]) -> SignedTab:
    """Block SympTab (forward) -> SignedTab in model labels."""
    var n = len(sigma)
    var out = SignedTab(n)
    for m in range(n):
        var b = sigma[m]
        out.x[m] = _relabel_mask(t.x[b], sigma_inv)
        out.z[m] = _relabel_mask(t.z[b], sigma_inv)
        out.s[m] = Int(t.r[b]) & 1
        out.x[n + m] = _relabel_mask(t.x[n + b], sigma_inv)
        out.z[n + m] = _relabel_mask(t.z[n + b], sigma_inv)
        out.s[n + m] = Int(t.r[n + b]) & 1
    return out^


# ============================================================ public API
def policy_available(kind: Int, g: SubGraph, model: String, dirs: List[String]) -> Bool:
    """True when a model of `kind` matches the block graph `g` (see find_model)."""
    if model == "off":
        return False
    return find_model(kind, g, model, dirs).ok


def policy_synth_perm(p: Perm, g: SubGraph, model: String, beams: Int, searches: Int, seed: Int,
                      dirs: List[String]) raises -> List[Instruction]:
    var s = PolicySession(dirs)
    return s.synth_perm(p, g, model, beams, searches, seed)


def policy_synth_linear(m: GF2Mat, g: SubGraph, model: String, beams: Int, searches: Int, seed: Int,
                        dirs: List[String]) raises -> List[Instruction]:
    var s = PolicySession(dirs)
    return s.synth_linear(m, g, model, beams, searches, seed)


def policy_synth_clifford(t: SympTab, g: SubGraph, model: String, beams: Int, searches: Int, seed: Int,
                          dirs: List[String]) raises -> List[Instruction]:
    var s = PolicySession(dirs)
    return s.synth_clifford(t, g, model, beams, searches, seed)


def policy_synth_pauli_network(pn: PauliNet, g: SubGraph, model: String, beams: Int, searches: Int, seed: Int,
                               dirs: List[String], src_ops: List[Instruction]) raises -> List[Instruction]:
    var s = PolicySession(dirs)
    return s.synth_pauli_network(pn, g, model, beams, searches, seed, src_ops)
