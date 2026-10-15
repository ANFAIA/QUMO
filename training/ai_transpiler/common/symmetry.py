"""Coupling-graph automorphisms -> observation / action permutations.

Same conventions as qiskit-gym 0.4.1 `envs/symmetry.rs` (Apache-2.0) as seen
by twisterl's policy: obs' = obs_perms[p][i]; logits_out[a] = raw[act_perms[p][a]].
For the Pauli-network env (which permutes internally) the equivalent
ensemble maps are produced by `pauli_twists` (see export_model.py).
"""
from __future__ import annotations


def automorphisms(n, edges):
    """All automorphisms of the undirected graph (perm[i] = image of i), sorted."""
    adj = [[False] * n for _ in range(n)]
    for a, b in edges:
        adj[a][b] = adj[b][a] = True
    deg = [sum(r) for r in adj]
    out = []
    perm = [-1] * n
    used = [False] * n

    def rec(i):
        if i == n:
            out.append(list(perm))
            return
        for v in range(n):
            if used[v] or deg[i] != deg[v]:
                continue
            if all(adj[i][j] == adj[v][perm[j]] for j in range(i)):
                perm[i] = v
                used[v] = True
                rec(i + 1)
                used[v] = False
                perm[i] = -1

    rec(0)
    return sorted(out)


def _action_perm(actions, perm):
    keymap = {}
    for i, (g, a, b) in enumerate(actions):
        keymap[(g, min(a, b), max(a, b)) if g == "SWAP" else (g, a, b)] = i
    out = []
    for g, a, b in actions:
        pa, pb = perm[a], (perm[b] if b >= 0 else -1)
        k = (g, min(pa, pb), max(pa, pb)) if g == "SWAP" else (g, pa, pb)
        if k not in keymap:
            return None
        out.append(keymap[k])
    return out


def twists(kind, n, actions, edges, max_rot=0):
    """(obs_perms, act_perms) for kind in permutation / linear_function / clifford / pauli_network."""
    obs_perms, act_perms = [], []
    for perm in automorphisms(n, edges):
        if kind == "pauli_network":
            continue
        ap = _action_perm(actions, perm)
        if ap is None:
            continue
        if kind in ("permutation", "linear_function"):
            op = [perm[r] * n + perm[c] for r in range(n) for c in range(n)]
        else:
            d = 2 * n
            mp = lambda r: perm[r] if r < n else n + perm[r - n]
            op = [mp(r) * d + mp(c) for r in range(d) for c in range(d)]
        obs_perms.append(op)
        act_perms.append(ap)
    if kind == "pauli_network":
        return pauli_twists(n, actions, max_rot, edges)
    return obs_perms, act_perms


def pauli_twists(n, actions, max_rot, edges):
    """Ensemble maps for the Pauli env: obs_perm maps an actual dense index to the
    permuted-frame index (rows / tableau columns permuted, rotation columns not);
    act_perm[a_actual] = model action (inverse of qiskit-gym's action perm)."""
    C = 2 * n + max_rot
    obs_perms, act_perms = [], []
    for perm in automorphisms(n, edges):
        A = _action_perm(actions, perm)
        if A is None:
            continue
        pr = [perm[i] for i in range(n)] + [n + perm[i] for i in range(n)]
        pr_inv = [0] * (2 * n)
        for i, v in enumerate(pr):
            pr_inv[v] = i
        op = []
        for r in range(2 * n):
            for c in range(C):
                cc = pr_inv[c] if c < 2 * n else c
                op.append(pr_inv[r] * C + cc)
        ainv = [0] * len(A)
        for i, v in enumerate(A):
            ainv[v] = i
        obs_perms.append(op)
        act_perms.append(ainv)
    return obs_perms, act_perms
