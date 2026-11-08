"""Training environment: reset / step / reward / metrics / masks over PolicyEnv (reference-gym semantics).

`TrainEnv` wraps one `qumo.compiler.policy_synth.PolicyEnv` (the packed-state dynamics of all four kinds, which the
goldens pin) and adds what the inference environment deliberately lacks (MOJO_PORT.md 6.1, envs.md sections 3-10):

* the reset generators (plain kinds: `difficulty` uniform random actions from the identity; Pauli networks: the
  random Clifford tableau and the rotation generator, including the label-mirroring quirk of the reference),
* the metrics tracker, the penalty chain and the reward, with every product behind `mulf` (the FMA barrier of
  MOJO_PORT.md section 3 rule 9; the unfused value is what the reference computes),
* the random inversion of the plain kinds (probability 1/2 after the gate, never at reset),
* the hidden symmetry of the Pauli environment (`observe()` draws one automorphism, `step()` un-permutes the action),
* the horizon (`depth`), the success flag and the all-true / all-false action mask.

Random numbers come from one seeded `Rng` owned by the environment (`reseed` installs the per-episode generator).
The draw order inside one episode is part of the determinism contract: `reset()` draws first, then one inversion bit
per `step()` (plain kinds) or one symmetry index per `observe()` (Pauli networks).

Errors never abort a parallel task: an out-of-range action under `add_perms` is clamped to action 0 and recorded in
`error_code()` (QM-E642), which the caller raises after the parallel region (MOJO_PORT.md section 3 rule 7).
"""

from std.collections import List
from std.memory import ArcPointer
from qrl.model import (
    MODEL_KIND_CLIFFORD,
    MODEL_KIND_LINEAR,
    MODEL_KIND_PAULI,
    MODEL_KIND_PERM,
)
from qrl.env_core import PolicyEnv
from qrl.gates import Op
from qrl.gateset import EnvConfig, obs_size_of, shell_model, validate_actions
from qrl.rlrng import mulf, uniform_f32, uniform_k24
from qrl.symmetry import Twists, pauli_qubit_perms_and_act_perms
from qrl.symmetry import twists as plain_twists
from qrl.rng import Rng

# integer thresholds of the Pauli tableau generator on k = uniform draw in [0, 2^24) (envs.md 9.7)
comptime K_CX_MIN = 5033166  # u > 0.3f32  <=>  k >= 5033166
comptime K_H_MIN = 2516583  # u > 0.15f32 <=>  k >= 2516583 (and k < K_CX_MIN)


# ------------------------------------------------------------------------------------------------ reward chain


def penalty_f32(w0: Float32, w1: Float32, w2: Float32, w3: Float32, d0: Int, d1: Int, d2: Int, d3: Int) -> Float32:
    """`((w0*d0 + w1*d1) + w2*d2) + w3*d3` in f32 with the integer increments converted to f32. Each product goes
    through `mulf` and each sum is a separate statement, so no multiply-add is fused (section 3, rule 9)."""
    var p0 = mulf(w0, Float32(d0))
    var p1 = mulf(w1, Float32(d1))
    var s = p0 + p1
    var p2 = mulf(w2, Float32(d2))
    s = s + p2
    var p3 = mulf(w3, Float32(d3))
    s = s + p3
    return s


def reward_f32(achieved: Float32, penalty: Float32, layer_reward: Float32, removed: Int) -> Float32:
    """`(achieved - penalty) + (layer_reward * f32(removed))` in f32 without fusion. `removed` is the number of
    rotations cleaned by the step (Pauli networks) and 0 for the plain kinds, where the reward is just
    `achieved - penalty`."""
    var base = achieved - penalty
    var bonus = mulf(layer_reward, Float32(removed))
    return base + bonus


# ------------------------------------------------------------------------------------------------ metrics


struct Metrics(Copyable):
    """The reference `MetricsTracker`: gate count, CX count and the sets of distinct (CX-)layer indices."""

    var n: Int
    var n_cnots: Int
    var n_gates: Int
    var n_layers: Int  # |distinct layer indices|
    var n_cnot_layers: Int
    var last_gates: List[Int]
    var last_cxs: List[Int]
    var layer_seen: List[Bool]
    var cnot_layer_seen: List[Bool]

    def __init__(out self, n: Int):
        self.n = n
        self.n_cnots = 0
        self.n_gates = 0
        self.n_layers = 0
        self.n_cnot_layers = 0
        self.last_gates = List[Int](length=n, fill=-1)
        self.last_cxs = List[Int](length=n, fill=-1)
        self.layer_seen = List[Bool]()
        self.cnot_layer_seen = List[Bool]()

    def reset(mut self):
        self.n_cnots = 0
        self.n_gates = 0
        self.n_layers = 0
        self.n_cnot_layers = 0
        for i in range(self.n):
            self.last_gates[i] = -1
            self.last_cxs[i] = -1
        self.layer_seen.clear()
        self.cnot_layer_seen.clear()

    def _cx(mut self, c: Int, t: Int):
        if c == t or c < 0 or t < 0 or c >= self.n or t >= self.n:
            return
        self.n_cnots += 1
        self.n_gates += 1
        var lg = max(self.last_gates[c], self.last_gates[t]) + 1
        self.last_gates[c] = lg
        self.last_gates[t] = lg
        while len(self.layer_seen) <= lg:
            self.layer_seen.append(False)
        if not self.layer_seen[lg]:
            self.layer_seen[lg] = True
            self.n_layers += 1
        var lk = max(self.last_cxs[c], self.last_cxs[t]) + 1
        self.last_cxs[c] = lk
        self.last_cxs[t] = lk
        while len(self.cnot_layer_seen) <= lk:
            self.cnot_layer_seen.append(False)
        if not self.cnot_layer_seen[lk]:
            self.cnot_layer_seen[lk] = True
            self.n_cnot_layers += 1

    def _one(mut self, t: Int):
        if t < 0 or t >= self.n:
            return
        self.n_gates += 1
        var lg = self.last_gates[t] + 1
        self.last_gates[t] = lg
        while len(self.layer_seen) <= lg:
            self.layer_seen.append(False)
        if not self.layer_seen[lg]:
            self.layer_seen[lg] = True
            self.n_layers += 1

    def apply_gate(mut self, op: Int, a: Int, b: Int):
        """Account for one gate of the gateset (H, S, CX, SWAP; CZ and the other 1-qubit gates follow the
        reference decomposition for completeness)."""
        if op == Op.CX:
            self._cx(a, b)
        elif op == Op.SWAP:
            self._cx(a, b)
            self._cx(b, a)
            self._cx(a, b)
        elif op == Op.CZ:
            self._one(b)
            self._cx(a, b)
            self._one(b)
        else:
            self._one(a)


# ------------------------------------------------------------------------------------------------ shared tables


struct EnvTables(Copyable):
    """Read-only tables shared (behind an ArcPointer) by every copy of one environment."""

    var valid_pairs: List[Int]  # flat (q0, q1) of every CX action in gateset order (Pauli generator)
    var all_dists: List[Int]  # sorted distinct graph distances >= 1 between pairs a < b
    var dist_pairs: List[List[Int]]  # dist_pairs[d] = flat (a, b), a < b, at distance d, lexicographic
    var qperms: List[List[Int]]  # Pauli hidden symmetry: automorphisms (perm[i] = image of i)
    var act_perms: List[List[Int]]  # Pauli: act_perms[p][model action] = real action
    var obs_map: List[List[Int]]  # Pauli: obs_map[p][flat index in the p = 0 view] = flat index in the p view
    var plain: Twists  # plain kinds with add_perms: the twists handed to the policy

    def __init__(out self):
        self.valid_pairs = List[Int]()
        self.all_dists = List[Int]()
        self.dist_pairs = List[List[Int]]()
        self.qperms = List[List[Int]]()
        self.act_perms = List[List[Int]]()
        self.obs_map = List[List[Int]]()
        self.plain = Twists()


def _build_tables(cfg: EnvConfig) -> EnvTables:
    var t = EnvTables()
    var n = cfg.n
    if cfg.kind != MODEL_KIND_PAULI:
        if cfg.add_perms:
            t.plain = plain_twists(cfg.kind, n, cfg.actions)
        return t^
    for i in range(len(cfg.actions) // 3):
        if cfg.actions[3 * i] == Op.CX:
            t.valid_pairs.append(cfg.actions[3 * i + 1])
            t.valid_pairs.append(cfg.actions[3 * i + 2])
    # BFS distances on the undirected graph of the CX pairs
    var adj = List[List[Int]]()
    for _ in range(n):
        adj.append(List[Int]())
    for p in range(len(t.valid_pairs) // 2):
        var a = t.valid_pairs[2 * p]
        var b = t.valid_pairs[2 * p + 1]
        adj[a].append(b)
        adj[b].append(a)
    for _ in range(n + 1):
        t.dist_pairs.append(List[Int]())
    var seen_d = List[Bool](length=n + 1, fill=False)
    for a in range(n):
        var dist = List[Int](length=n, fill=-1)
        var queue = List[Int]()
        dist[a] = 0
        queue.append(a)
        var head = 0
        while head < len(queue):
            var u = queue[head]
            head += 1
            for k in range(len(adj[u])):
                var w = adj[u][k]
                if dist[w] < 0:
                    dist[w] = dist[u] + 1
                    queue.append(w)
        for b in range(a + 1, n):
            var d = dist[b]
            if d >= 1:
                t.dist_pairs[d].append(a)
                t.dist_pairs[d].append(b)
                seen_d[d] = True
    for d in range(1, n + 1):
        if seen_d[d]:
            t.all_dists.append(d)
    if cfg.add_perms:
        var tw = pauli_qubit_perms_and_act_perms(n, cfg.actions)
        t.qperms = tw.obs_perms.copy()
        t.act_perms = tw.act_perms.copy()
        var cols = 2 * n + cfg.max_rotations
        for p in range(len(t.qperms)):
            var perm = t.qperms[p].copy()
            # new[r][c] = old[m(r)][m(c)] for c < 2n, new[r][c] = old[m(r)][c] for the rotation columns, with
            # m(i) = perm[i] (i < n), n + perm[i - n]; an old entry (ro, co) therefore lands at (m^-1(ro), m^-1(co))
            var minv = List[Int](length=2 * n, fill=0)
            for i in range(n):
                minv[perm[i]] = i
                minv[n + perm[i]] = n + i
            var mp = List[Int](length=2 * n * cols, fill=0)
            for ro in range(2 * n):
                for co in range(cols):
                    var cc = minv[co] if co < 2 * n else co
                    mp[ro * cols + co] = minv[ro] * cols + cc
            t.obs_map.append(mp^)
    return t^


# ------------------------------------------------------------------------------------------------ the environment


struct TrainEnv(Copyable):
    var cfg: EnvConfig
    var seed: UInt64
    var pe: PolicyEnv
    var tabs: ArcPointer[EnvTables]
    var st: List[UInt64]  # packed state in the PolicyEnv layout
    var rng: Rng
    var metrics: Metrics
    var difficulty: Int
    var depth: Int
    var success_flag: Bool
    var reward_value: Float32
    var cur_perm: Int  # Pauli: symmetry drawn by the last observe(), used by step()
    var last_actual: Int  # Pauli: action actually executed by the last step() (after un-permuting)
    var err: Int
    var emit: List[Int]
    var inv_scratch: List[UInt64]

    def __init__(out self, cfg: EnvConfig, seed: UInt64) raises:
        validate_actions(cfg)
        self.cfg = cfg.copy()
        self.seed = seed
        self.pe = PolicyEnv(cfg.kind, cfg.n, shell_model(cfg))
        self.tabs = ArcPointer(_build_tables(cfg))
        self.st = List[UInt64]()
        self.rng = Rng(seed)
        self.metrics = Metrics(cfg.n)
        self.difficulty = cfg.difficulty
        self.depth = 1
        self.success_flag = True
        self.reward_value = 1.0
        self.cur_perm = 0
        self.last_actual = -1
        self.err = 0
        self.emit = List[Int]()
        self.inv_scratch = List[UInt64]()
        self.st = self._identity_state()

    # ------------------------------------------------------------------ basic accessors

    def num_actions(self) -> Int:
        return len(self.cfg.actions) // 3

    def obs_size(self) -> Int:
        return obs_size_of(self.cfg)

    def reseed(mut self, seed: UInt64):
        """Install the per-episode generator."""
        self.seed = seed
        self.rng = Rng(seed)

    def set_difficulty(mut self, d: Int):
        """Takes effect at the next reset()."""
        self.difficulty = d if d >= 0 else 0

    def all_allowed(self) -> Bool:
        """`masks() == [not success] * A`."""
        return not self.success_flag

    def is_final(self) -> Bool:
        return self.depth == 0 or self.success_flag

    def success(self) -> Bool:
        return self.success_flag

    def reward(self) -> Float32:
        return self.reward_value

    def error_code(self) -> Int:
        """0 = ok; set (never raised) inside tasks, MOJO_PORT.md section 3 rule 7."""
        return self.err

    def horizon(self) -> Int:
        """Remaining steps (`depth`): `min(depth_slope * difficulty, max_depth)` after reset(), `max_depth` after
        set_state, decremented (saturating) by every step."""
        return self.depth

    def num_syms(self) -> Int:
        """Pauli networks: number of hidden symmetries (0 without add_perms)."""
        return len(self.tabs[].qperms)

    def current_perm(self) -> Int:
        """Pauli networks: symmetry index drawn by the last observe() (0 initially)."""
        return self.cur_perm

    def last_actual_action(self) -> Int:
        """Pauli networks: gateset action executed by the last step() (after un-permuting), -1 before any."""
        return self.last_actual

    def pauli_qubit_perm(self, p: Int) -> List[Int]:
        return self.tabs[].qperms[p].copy()

    def pauli_act_perm(self, p: Int) -> List[Int]:
        """`act_perms[p][model action]` = the real action (the table of `symmetry.pauli_qubit_perms_and_act_perms`)."""
        return self.tabs[].act_perms[p].copy()

    def pauli_obs_map(self, p: Int) -> List[Int]:
        """Flat index map of symmetry p: entry `i` is where observation index `i` of the identity view lands in the
        view of symmetry p (rows and tableau columns permuted, rotation columns in place)."""
        return self.tabs[].obs_map[p].copy()

    def state_words(self) -> List[UInt64]:
        """Copy of the packed state (PolicyEnv layout), for tests and diagnostics."""
        return self.st.copy()

    def pauli_labels(self) -> List[String]:
        """Pauli networks: the labels of the active rotations in node order, character `k` <-> qubit `n - 1 - k`
        (leftmost character = highest qubit). Empty for the other kinds."""
        var out = List[String]()
        if self.cfg.kind != MODEL_KIND_PAULI:
            return out^
        var n = self.cfg.n
        var cnt = Int(self.st[self.pe._cnt()])
        for pos in range(cnt):
            var ri = Int(self.st[self.pe._nodes() + pos])
            var x = self.st[self.pe._rx() + ri]
            var z = self.st[self.pe._rz() + ri]
            var s = String()
            for k in range(n):
                var q = UInt64(n - 1 - k)
                var xb = ((x >> q) & 1) == 1
                var zb = ((z >> q) & 1) == 1
                if xb and zb:
                    s += "Y"
                elif xb:
                    s += "X"
                elif zb:
                    s += "Z"
                else:
                    s += "I"
            out.append(s^)
        return out^

    def twists(self) -> Twists:
        """Plain kinds: `symmetry.twists` (empty without add_perms); Pauli networks: empty (the symmetry is
        internal, the policy gets no obs / act perms)."""
        if self.cfg.kind == MODEL_KIND_PAULI or not self.cfg.add_perms:
            return Twists()
        return self.tabs[].plain.copy()

    # ------------------------------------------------------------------ state construction

    def _identity_state(self) -> List[UInt64]:
        var n = self.cfg.n
        var kind = self.cfg.kind
        var s = List[UInt64]()
        if kind == MODEL_KIND_PERM:
            for i in range(n):
                s.append(UInt64(i))
        elif kind == MODEL_KIND_LINEAR:
            for i in range(n):
                s.append(UInt64(1) << UInt64(i))
        else:
            for i in range(2 * n):
                s.append(UInt64(1) << UInt64(i))
            if kind == MODEL_KIND_PAULI:
                s.append(0)  # rotation count; PolicyEnv layout with R = 0
        return s^

    def _assemble_pauli(mut self, rows: List[UInt64], rx: List[UInt64], rz: List[UInt64]):
        """Install a Pauli state from the tableau rows and the rotation masks (all rotations active, in order, with
        sign bit 0), and rebuild the anticommutation table."""
        var n = self.cfg.n
        var R = len(rx)
        var s = List[UInt64](capacity=2 * n + 4 * R + 1)
        for i in range(2 * n):
            s.append(rows[i])
        for i in range(R):
            s.append(rx[i])
        for i in range(R):
            s.append(rz[i])
        for _ in range(R):
            s.append(0)
        s.append(UInt64(R))
        for i in range(R):
            s.append(UInt64(i))
        self.pe.init_pauli(R, s)
        self.st = s^

    def _reset_internals(mut self):
        self.success_flag = self.pe.solved(self.st)
        self.metrics.reset()
        self.reward_value = 1.0 if self.success_flag else 0.0
        self.cur_perm = 0
        self.last_actual = -1

    # ------------------------------------------------------------------ reset generators

    def reset(mut self):
        """Plain kinds: the identity scrambled by `difficulty` uniform random actions (no inversion, no rejection).
        Pauli networks: random Clifford tableau plus the rotation list of `generate_paulis`. Then
        `depth = min(depth_slope * difficulty, max_depth)` and the internals are reset."""
        if self.cfg.kind == MODEL_KIND_PAULI:
            self._reset_pauli()
        else:
            self.st = self._identity_state()
            var A = self.num_actions()
            for _ in range(self.difficulty):
                var a = self.rng.below(A)
                self.emit.clear()
                self.pe.apply(self.st, a, self.emit, False)
        var h = self.cfg.depth_slope * self.difficulty
        self.depth = h if h < self.cfg.max_depth else self.cfg.max_depth
        self._reset_internals()

    def _random_tableau(mut self, d: Int) -> List[UInt64]:
        """`random_clifford_tableau(d)`: H / S / CX scramble of the 2n x 2n identity with the integer thresholds on
        k = u * 2^24. Nothing happens for d == 0 or when the gateset has no CX."""
        var n = self.cfg.n
        var rows = List[UInt64](capacity=2 * n)
        for i in range(2 * n):
            rows.append(UInt64(1) << UInt64(i))
        var npairs = len(self.tabs[].valid_pairs) // 2
        if d <= 0 or npairs == 0:
            return rows^
        for _ in range(d):
            var k = uniform_k24(self.rng)
            if k >= K_CX_MIN:
                var pi = self.rng.below(npairs)
                var q0 = self.tabs[].valid_pairs[2 * pi]
                var q1 = self.tabs[].valid_pairs[2 * pi + 1]
                rows[q1] ^= rows[q0]
                rows[n + q0] ^= rows[n + q1]
            elif k >= K_H_MIN:
                var q = self.rng.below(n)
                var t = rows[q]
                rows[q] = rows[n + q]
                rows[n + q] = t
            else:
                var q = self.rng.below(n)
                rows[n + q] ^= rows[q]
        return rows^

    def _pauli_under_diff(mut self, budget: Int, mut layer: List[Int]) -> Int:
        """`pauli_under_diff(D)`: fills `layer` (length n; 0 = I, 1 = X, 2 = Y, 3 = Z at generator position q) and
        returns the cost `D - pd`, or -1 when no distance fits the budget (the reference's `None`)."""
        var n = self.cfg.n
        var valid = List[Int]()
        for i in range(len(self.tabs[].all_dists)):
            if self.tabs[].all_dists[i] <= budget:
                valid.append(self.tabs[].all_dists[i])
        if len(valid) == 0:
            return -1
        var inq = List[Bool](length=n, fill=False)
        var pd = budget
        var nd = valid[self.rng.below(len(valid))]
        var npr = len(self.tabs[].dist_pairs[nd]) // 2
        var pick = self.rng.below(npr)
        inq[self.tabs[].dist_pairs[nd][2 * pick]] = True
        inq[self.tabs[].dist_pairs[nd][2 * pick + 1]] = True
        pd = pd - nd
        if pd < 0:
            pd = 0
        while True:
            var vd = List[Int]()
            for i in range(len(valid)):
                if valid[i] <= pd:
                    vd.append(valid[i])
            var nrem = 0
            for q in range(n):
                if not inq[q]:
                    nrem += 1
            if pd == 0 or len(vd) == 0 or nrem == 0:
                break
            if uniform_f32(self.rng) <= self.cfg.num_qubits_decay:
                break
            var d2 = vd[self.rng.below(len(vd))]
            var vp = List[Int]()  # flat pairs of distance d2 touching the chosen qubits
            var total = len(self.tabs[].dist_pairs[d2]) // 2
            for i in range(total):
                var a = self.tabs[].dist_pairs[d2][2 * i]
                var b = self.tabs[].dist_pairs[d2][2 * i + 1]
                if inq[a] or inq[b]:
                    vp.append(a)
                    vp.append(b)
            if len(vp) == 0:
                continue  # the stop coin is flipped again on the next pass
            var j = self.rng.below(len(vp) // 2)
            inq[vp[2 * j]] = True
            inq[vp[2 * j + 1]] = True
            pd = pd - d2
            if pd < 0:
                pd = 0
        for q in range(n):
            if inq[q]:
                layer[q] = 1 + self.rng.below(3)
            else:
                layer[q] = 0
        return budget - pd

    def _reset_pauli(mut self):
        var n = self.cfg.n
        var rx = List[UInt64]()
        var rz = List[UInt64]()
        var rem = self.difficulty // self.cfg.pauli_diff_scale
        while rem > 0 and len(rx) < self.cfg.final_pauli_layers:
            var layer = List[Int](length=n, fill=0)
            var cost = self._pauli_under_diff(rem, layer)
            if cost < 0:
                break
            # the label is parsed with the leftmost-is-highest-qubit convention: character k <-> qubit n - 1 - k
            var x: UInt64 = 0
            var z: UInt64 = 0
            for k in range(n):
                var bit = UInt64(1) << UInt64(n - 1 - k)
                var c = layer[k]
                if c == 1 or c == 2:
                    x |= bit
                if c == 2 or c == 3:
                    z |= bit
            rx.append(x)
            rz.append(z)
            var dec = cost if cost > 1 else 1
            rem = rem - dec
            if rem < 0:
                rem = 0
        var rows = self._random_tableau(self.difficulty)
        self._assemble_pauli(rows, rx, rz)

    # ------------------------------------------------------------------ set_state (golden tests, warm starts)

    def _finish_set_state(mut self):
        self.depth = self.cfg.max_depth
        self._reset_internals()

    def set_state_perm(mut self, s: List[Int]):
        """Permutation state `s` (a permutation of 0..n-1); anything else is ignored and sets `error_code()`."""
        var n = self.cfg.n
        if self.cfg.kind != MODEL_KIND_PERM or len(s) != n:
            self.err = 642
            return
        var seen = List[Bool](length=n, fill=False)
        for i in range(n):
            if s[i] < 0 or s[i] >= n or seen[s[i]]:
                self.err = 642
                return
            seen[s[i]] = True
        var out = List[UInt64](capacity=n)
        for i in range(n):
            out.append(UInt64(s[i]))
        self.st = out^
        self._finish_set_state()

    def set_state_bits(mut self, v: List[Int]):
        """LF / Clifford flat 0/1 matrix, row-major (`bit(i, j) = v[i * w + j] > 0`, `w = n` or `2n`)."""
        var kind = self.cfg.kind
        if kind != MODEL_KIND_LINEAR and kind != MODEL_KIND_CLIFFORD:
            self.err = 642
            return
        var w = self.cfg.n if kind == MODEL_KIND_LINEAR else 2 * self.cfg.n
        if len(v) != w * w:
            self.err = 642
            return
        var out = List[UInt64](length=w, fill=0)
        for i in range(w):
            for j in range(w):
                if v[i * w + j] > 0:
                    out[i] |= UInt64(1) << UInt64(j)
        self.st = out^
        self._finish_set_state()

    def set_state_pauli(mut self, v: List[Int]) raises:
        """envs.md 9.6 vector: `[R, tableau (4n^2 entries, row-major, > 0 -> 1), then R times: len, ord(ch) * len]`.
        An empty vector does nothing. Only the first `max_rotations` labels are kept; no initial clean() (a
        weight-1 rotation stays in the DAG until some later CX cleans it). Labels may carry a phase prefix
        (`-`, `i`, `-i`); the last n characters are the Pauli letters, character k <-> qubit n - 1 - k."""
        if self.cfg.kind != MODEL_KIND_PAULI:
            raise Error("QM-E642: set_state_pauli on a non-Pauli environment. Hint: use set_state_perm / set_state_bits.")
        if len(v) == 0:
            return
        var n = self.cfg.n
        var d = 2 * n
        var R = v[0]
        if R < 0 or len(v) < 1 + d * d:
            raise Error("QM-E642: malformed Pauli state vector (header or tableau too short). Hint: see envs.md 9.6.")
        var rows = List[UInt64](length=d, fill=0)
        for i in range(d):
            for j in range(d):
                if v[1 + i * d + j] > 0:
                    rows[i] |= UInt64(1) << UInt64(j)
        var pos = 1 + d * d
        var keep = R if R < self.cfg.max_rotations else self.cfg.max_rotations
        var rx = List[UInt64]()
        var rz = List[UInt64]()
        for r in range(R):
            if pos >= len(v):
                raise Error("QM-E642: malformed Pauli state vector (missing rotation label). Hint: see envs.md 9.6.")
            var L = v[pos]
            pos += 1
            if L < n or pos + L > len(v):
                raise Error("QM-E642: malformed Pauli state vector (bad label length). Hint: see envs.md 9.6.")
            if r < keep:
                var x: UInt64 = 0
                var z: UInt64 = 0
                for k in range(n):
                    var c = v[pos + (L - n) + k]
                    var bit = UInt64(1) << UInt64(n - 1 - k)
                    if c == ord("X"):
                        x |= bit
                    elif c == ord("Z"):
                        z |= bit
                    elif c == ord("Y"):
                        x |= bit
                        z |= bit
                    elif c != ord("I"):
                        raise Error("QM-E642: malformed Pauli state vector (letter outside I, X, Y, Z). Hint: see envs.md 9.6.")
                rx.append(x)
                rz.append(z)
            pos += L
        self._assemble_pauli(rows, rx, rz)
        self._finish_set_state()

    # ------------------------------------------------------------------ observation

    def _pauli_map_into(self, p: Int, mut obs: List[Int]):
        ref mp = self.tabs[].obs_map[p]
        for i in range(len(obs)):
            obs[i] = mp[obs[i]]
        sort(obs)

    def observe(mut self, mut obs: List[Int]):
        """Sorted ascending sparse indices of the 1 entries. Pauli networks with add_perms draw one symmetry
        uniformly, store it for step() and permute the indices; call exactly once before each step()."""
        self.pe.observe(self.st, obs)
        if self.cfg.kind == MODEL_KIND_PAULI:
            var ns = len(self.tabs[].qperms)
            if ns > 0:
                var p = self.rng.below(ns)
                self.cur_perm = p
                self._pauli_map_into(p, obs)

    def observe_perm(mut self, p: Int, mut obs: List[Int]):
        """Pauli networks: the observation of explicit symmetry `p` (stored for step()); a test hook that draws
        nothing. An out-of-range `p` sets `error_code()` and returns the unpermuted view."""
        self.pe.observe(self.st, obs)
        if self.cfg.kind != MODEL_KIND_PAULI or p < 0 or p >= len(self.tabs[].qperms):
            self.err = 642
            return
        self.cur_perm = p
        self._pauli_map_into(p, obs)

    # ------------------------------------------------------------------ step

    def _metrics_penalty(mut self, act: Int) -> Float32:
        var op = self.cfg.actions[3 * act]
        var a = self.cfg.actions[3 * act + 1]
        var b = self.cfg.actions[3 * act + 2]
        var c0 = self.metrics.n_cnots
        var c1 = self.metrics.n_cnot_layers
        var c2 = self.metrics.n_layers
        var c3 = self.metrics.n_gates
        self.metrics.apply_gate(op, a, b)
        return penalty_f32(
            self.cfg.w_cnots,
            self.cfg.w_layers_cnots,
            self.cfg.w_layers,
            self.cfg.w_gates,
            self.metrics.n_cnots - c0,
            self.metrics.n_cnot_layers - c1,
            self.metrics.n_layers - c2,
            self.metrics.n_gates - c3,
        )

    def _random_invert(mut self):
        """With probability 1/2 replace the state by its inverse (plain kinds, after the gate, before solved())."""
        if (self.rng.next_u64() >> 63) == 0:
            return
        if self.pe.inverse(self.st, self.inv_scratch):
            self.st = self.inv_scratch.copy()
        else:
            self.err = 642  # singular matrix: only reachable through a garbage set_state

    def step(mut self, a: Int):
        """Apply action `a`. With add_perms an `a` outside 0..A-1 is clamped to 0 and sets `error_code()` (the
        reference panics there); without add_perms it is a silent no-op. Always decrements `depth` (saturating)."""
        var A = self.num_actions()
        var penalty: Float32 = 0.0
        var removed = 0
        var act = a
        var valid = a >= 0 and a < A
        if not valid and self.cfg.add_perms:
            act = 0
            valid = True
            self.err = 642
        if self.cfg.kind == MODEL_KIND_PAULI:
            if valid and len(self.tabs[].qperms) > 0:
                act = self.tabs[].act_perms[self.cur_perm][act]
            self.last_actual = act if valid else -1
            if valid and act >= 0 and act < A:
                penalty = self._metrics_penalty(act)
                var before = Int(self.st[self.pe._cnt()])
                self.emit.clear()
                self.pe.apply(self.st, act, self.emit, False)
                removed = before - Int(self.st[self.pe._cnt()])
        else:
            if valid:
                penalty = self._metrics_penalty(act)
                self.emit.clear()
                self.pe.apply(self.st, act, self.emit, False)
            if self.cfg.add_inverts:
                self._random_invert()
        if self.depth > 0:
            self.depth -= 1
        self.success_flag = self.pe.solved(self.st)
        var achieved: Float32 = 1.0 if self.success_flag else 0.0
        if self.cfg.kind == MODEL_KIND_PAULI:
            self.reward_value = reward_f32(achieved, penalty, self.cfg.pauli_layer_reward, removed)
        else:
            self.reward_value = achieved - penalty
