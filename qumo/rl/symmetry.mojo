"""Graph automorphisms and the observation / action permutations ("twists") derived from them (MOJO_PORT.md 4.6).

Conventions (pinned by `tests/fixtures/rl/envs/gateset_twists.json`, `envs/pauli_twists.json` and the nine shipped
manifests):

* an automorphism is `perm` with `perm[i]` = image of qubit `i`; the list of all of them is sorted
  lexicographically (identity first);
* plain kinds (permutation / linear function / Clifford): `obs_perms[p][i]` is the permuted observation index and
  `act_perms[p][a]` the action whose logit is read into slot `a` (`logits_out[a] = raw[act_perms[p][a]]`), where
  `act_perms[p][a]` is the index of the gate obtained by relabelling the qubits of action `a` through `perm`;
* Pauli training env (`pauli_qubit_perms_and_act_perms`): `obs_perms[p]` is the qubit permutation `perm` itself and
  `act_perms[p][a]` the same *direct* relabelling (model action `a` acts on the real qubits `perm[...]`);
* Pauli export ensemble (`pauli_twists`): `obs_perms[p]` is a dense-index map over the `2n x (2n + max_rot)` tableau
  layout built from the *inverse* of the row/column permutation, and `act_perms[p]` is the *inverse* of the direct
  action permutation. The two views differ exactly on non-involutive automorphisms (ring:4, ring:5, star:4, H7).

The graph is recovered from the 2-qubit actions of the gateset, so callers pass the same flat `actions` triples
`(opcode, q0, q1)` (`q1 = -1` for 1-qubit gates) that a model container carries.
"""

from std.collections import Dict, List
from qumo.compiler.policy import MODEL_KIND_CLIFFORD, MODEL_KIND_LINEAR, MODEL_KIND_PAULI, MODEL_KIND_PERM
from qumo.ir.gates import Op


@fieldwise_init
struct Twists(Copyable):
    """Parallel lists: `obs_perms[p]` and `act_perms[p]` describe symmetry `p`."""

    var obs_perms: List[List[Int]]
    var act_perms: List[List[Int]]

    def __init__(out self):
        self.obs_perms = List[List[Int]]()
        self.act_perms = List[List[Int]]()

    def count(self) -> Int:
        return len(self.act_perms)


# ------------------------------------------------------------------------------------------------ automorphisms


def automorphisms(n: Int, edges: List[Int]) -> List[List[Int]]:
    """All automorphisms of the undirected graph on `n` vertices with flat `edges` (`perm[i]` = image of `i`),
    sorted lexicographically. Self loops, duplicate edges and out-of-range endpoints are ignored. A graph with no
    edge has all `n!` permutations, so callers must not ask for large edgeless graphs."""
    var out = List[List[Int]]()
    if n <= 0:
        return out^
    var adj = List[Bool](length=n * n, fill=False)
    var ne = len(edges) // 2
    for e in range(ne):
        var a = edges[2 * e]
        var b = edges[2 * e + 1]
        if a < 0 or a >= n or b < 0 or b >= n or a == b:
            continue
        adj[a * n + b] = True
        adj[b * n + a] = True
    var deg = List[Int](length=n, fill=0)
    for i in range(n):
        var d = 0
        for j in range(n):
            if adj[i * n + j]:
                d += 1
        deg[i] = d

    # Iterative backtracking. Position i takes the smallest unused image v that has the right degree and keeps
    # adjacency consistent with the images already chosen; trying v in increasing order makes the enumeration
    # lexicographic, so `out` needs no sort.
    var perm = List[Int](length=n, fill=-1)
    var used = List[Bool](length=n, fill=False)
    var next_v = List[Int](length=n, fill=0)  # next candidate image to try at each depth
    var i = 0
    while i >= 0:
        if i == n:
            out.append(perm.copy())
            i -= 1
            # release the image held at depth i so it can advance
            used[perm[i]] = False
            perm[i] = -1
            continue
        var placed = False
        var v = next_v[i]
        while v < n:
            if (not used[v]) and deg[i] == deg[v]:
                var ok = True
                for j in range(i):
                    if adj[i * n + j] != adj[v * n + perm[j]]:
                        ok = False
                        break
                if ok:
                    perm[i] = v
                    used[v] = True
                    next_v[i] = v + 1
                    placed = True
                    break
            v += 1
        if placed:
            i += 1
            if i < n:
                next_v[i] = 0
        else:
            next_v[i] = 0
            i -= 1
            if i >= 0:
                used[perm[i]] = False
                perm[i] = -1
    return out^


# ------------------------------------------------------------------------------------------------ helpers


def _coupling_edges(n: Int, actions: List[Int]) -> List[Int]:
    """Flat unique undirected pairs (a < b) of every 2-qubit action (CX, CZ, SWAP, any action with q1 >= 0)."""
    var seen = List[Bool](length=n * n, fill=False)
    var out = List[Int]()
    var na = len(actions) // 3
    for i in range(na):
        var a = actions[3 * i + 1]
        var b = actions[3 * i + 2]
        if a < 0 or b < 0 or a >= n or b >= n or a == b:
            continue
        var lo = a if a < b else b
        var hi = b if a < b else a
        if not seen[lo * n + hi]:
            seen[lo * n + hi] = True
            out.append(lo)
            out.append(hi)
    return out^


def _action_key(op: Int, a: Int, b: Int) -> Int:
    """Dictionary key of an action: SWAP is unordered (min, max), every other gate keeps its operand order; a
    1-qubit gate has `b = -1`."""
    var x = a
    var y = b
    if op == Op.SWAP and b >= 0 and b < a:
        x = b
        y = a
    return (op * 8192 + x) * 8192 + (y + 1)


def _action_perm(n: Int, actions: List[Int], perm: List[Int], mut out: List[Int]) -> Bool:
    """`out[a]` = index of the action obtained by relabelling every qubit q -> perm[q]. False when some image is not
    in the gateset (the automorphism is then not usable). If the gateset repeats an action the later index wins,
    as a dictionary insert does."""
    var keymap = Dict[Int, Int]()
    var na = len(actions) // 3
    for i in range(na):
        keymap[_action_key(actions[3 * i], actions[3 * i + 1], actions[3 * i + 2])] = i
    out = List[Int](capacity=na)
    for i in range(na):
        var op = actions[3 * i]
        var a = actions[3 * i + 1]
        var b = actions[3 * i + 2]
        if a < 0 or a >= n or b >= n:
            return False
        var pa = perm[a]
        var pb = -1
        if b >= 0:
            pb = perm[b]
        var image = keymap.get(_action_key(op, pa, pb), -1)
        if image < 0:
            return False
        out.append(image)
    return True


def _invert(p: List[Int]) -> List[Int]:
    var inv = List[Int](length=len(p), fill=0)
    for i in range(len(p)):
        inv[p[i]] = i
    return inv^


# ------------------------------------------------------------------------------------------------ twists


def twists(kind: Int, n: Int, actions: List[Int]) -> Twists:
    """Plain kinds (permutation / linear / Clifford): the ensemble maps; Pauli and any other kind: empty.
    `actions` = flat triples `(opcode, q0, q1)`.

    The coupling graph is the set of 2-qubit actions. Automorphisms are visited in lexicographic order; one whose
    relabelled action is missing from the gateset is skipped. (The identity always has an action permutation, so
    the "fall back to identity when nothing is kept" rule of qiskit-gym cannot trigger and is not needed.)"""
    var res = Twists()
    if kind != MODEL_KIND_PERM and kind != MODEL_KIND_LINEAR and kind != MODEL_KIND_CLIFFORD:
        return res^
    var autos = automorphisms(n, _coupling_edges(n, actions))
    for p in range(len(autos)):
        var perm = autos[p].copy()
        var ap = List[Int]()
        if not _action_perm(n, actions, perm, ap):
            continue
        var op: List[Int]
        if kind == MODEL_KIND_CLIFFORD:
            var d = 2 * n
            var m = List[Int](length=d, fill=0)
            for i in range(d):
                m[i] = perm[i] if i < n else n + perm[i - n]
            op = List[Int](capacity=d * d)
            for r in range(d):
                for c in range(d):
                    op.append(m[r] * d + m[c])
        else:
            op = List[Int](capacity=n * n)
            for r in range(n):
                for c in range(n):
                    op.append(perm[r] * n + perm[c])
        res.obs_perms.append(op^)
        res.act_perms.append(ap^)
    return res^


def pauli_twists(n: Int, actions: List[Int], max_rot: Int) -> Twists:
    """Inference-side ensemble of a Pauli model (reproduces the manifest `symmetries`).

    `obs_perms[p]` has length `2n * (2n + max_rot)` and maps an actual dense index to the permuted-frame index
    (tableau rows and columns permuted by the inverse of `[perm, n + perm]`, rotation columns left in place);
    `act_perms[p][a_actual]` = model action = the inverse of the direct action permutation."""
    var res = Twists()
    var autos = automorphisms(n, _coupling_edges(n, actions))
    var cols = 2 * n + max_rot
    for p in range(len(autos)):
        var perm = autos[p].copy()
        var direct = List[Int]()
        if not _action_perm(n, actions, perm, direct):
            continue
        var pr = List[Int](length=2 * n, fill=0)
        for i in range(n):
            pr[i] = perm[i]
            pr[n + i] = n + perm[i]
        var pr_inv = _invert(pr)
        var op = List[Int](capacity=2 * n * cols)
        for r in range(2 * n):
            for c in range(cols):
                var cc = pr_inv[c] if c < 2 * n else c
                op.append(pr_inv[r] * cols + cc)
        res.obs_perms.append(op^)
        res.act_perms.append(_invert(direct))
    return res^


def pauli_qubit_perms_and_act_perms(n: Int, actions: List[Int]) -> Twists:
    """Training-env tables of the Pauli network environment: `obs_perms` holds the qubit permutations (the raw
    automorphisms, `perm[i]` = image of qubit `i`) and `act_perms` the direct action permutations (the
    environment un-permutes the model's action through them). Automorphisms without an action permutation are
    skipped."""
    var res = Twists()
    var autos = automorphisms(n, _coupling_edges(n, actions))
    for p in range(len(autos)):
        var ap = List[Int]()
        if not _action_perm(n, actions, autos[p], ap):
            continue
        res.obs_perms.append(autos[p].copy())
        res.act_perms.append(ap^)
    return res^
