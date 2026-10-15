"""Seeded circuit and topology distributions for router training / evaluation.

Topologies are QUMO's hardware-neutral families (edge builders shared with
bench/neutral/targets.py): grid, heavy-hex, diagonal grid, octagonal, line,
ring, tree, torus, neutral-atom radius-2 grid, ion zone ring.

Circuit families (all generated, never taken from bench/neutral/circuits,
which is the benchmark test set and stays fully held out):
  rand2q   random 2q layers (random partial matchings)
  qv       quantum-volume-like (random pairings, 3-CX blocks with 1q gates)
  qft      QFT-like all-pairs controlled-phase ladder
  trotter  Heisenberg / Ising Trotter steps on a logical line or grid lattice
  qaoa     QAOA on random 3-regular graphs (cx rz cx per edge)
  sparse   random sparse long-range CX circuits with measurements

Ops are (kind, a, b) tuples understood by route_env.RProblem and
route_tool.mojo: cx cz swap u2 u1 measure barrier.
"""
from __future__ import annotations

import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "bench", "neutral"))
import targets as nt  # noqa: E402  (bench/neutral/targets.py edge builders)


def topology(kind, size):
    """(name, n, edges) of a topology family at a size parameter."""
    if kind == "grid":
        r, c = size
        e = []
        for i in range(r):
            for j in range(c):
                q = i * c + j
                if j + 1 < c:
                    e.append((q, q + 1))
                if i + 1 < r:
                    e.append((q, q + c))
        return f"grid{r}x{c}", r * c, e
    if kind == "heavy_hex":
        n, e = nt.heavy_hex_edges(size)
        return f"heavy_hex{n}", n, [tuple(x) for x in e]
    if kind == "diag":
        n, e = nt.diagonal_grid_edges(size)
        return f"diag{n}", n, e
    if kind == "octagonal":
        n, e = nt.octagonal_edges(size)
        return f"oct{n}", n, e
    if kind == "line":
        return f"line{size}", size, [(i, i + 1) for i in range(size - 1)]
    if kind == "ring":
        return f"ring{size}", size, [(i, (i + 1) % size) for i in range(size)]
    if kind == "tree":
        n, e = nt.tree_edges(size)
        return f"tree{n}", n, e
    if kind == "torus":
        n, e = nt.torus_edges(size)
        return f"torus{n}", n, e
    if kind == "atom2":
        n, e = nt.grid_edges(size, 2.0)
        return f"atom_r2_{n}", n, e
    if kind == "ionring":
        n, e = nt.ion_zone_ring_edges(size)
        return f"ionring{n}", n, e
    raise ValueError(kind)


TRAIN_TOPOLOGIES = [
    ("grid", (3, 3)), ("grid", (4, 4)), ("grid", (5, 5)), ("grid", (6, 6)), ("grid", (3, 5)),
    ("heavy_hex", 19), ("heavy_hex", 57),
    ("diag", 16), ("diag", 25),
    ("octagonal", 16), ("octagonal", 32),
    ("line", 8), ("line", 12), ("line", 20),
    ("ring", 8), ("ring", 14), ("ring", 20),
    ("tree", 15), ("tree", 31),
    ("torus", 16), ("torus", 25),
    ("atom2", 16), ("atom2", 25),
    ("ionring", 12), ("ionring", 16),
]

# families used for the per-family gains table
FAMILY_OF = {"grid": "grid", "heavy_hex": "heavy_hex", "diag": "diag_grid", "octagonal": "octagonal",
             "line": "line", "ring": "ring", "tree": "tree", "torus": "torus", "atom2": "neutral_atom",
             "ionring": "ion_ring"}


def _u1(ops, rng, q):
    if rng.random() < 0.5:
        ops.append(("u1", q, -1))


def rand2q(n, depth, rng):
    ops = []
    for _ in range(depth):
        qs = list(range(n))
        rng.shuffle(qs)
        for k in range(0, n - 1, 2):
            if rng.random() < 0.7:
                a, b = qs[k], qs[k + 1]
                _u1(ops, rng, a)
                ops.append(("cx", a, b) if rng.random() < 0.8 else ("cz", a, b))
    return ops


def qv(n, depth, rng):
    ops = []
    for _ in range(depth):
        qs = list(range(n))
        rng.shuffle(qs)
        for k in range(0, n - 1, 2):
            a, b = qs[k], qs[k + 1]
            for _ in range(3):
                ops.append(("u1", a, -1))
                ops.append(("cx", a, b) if rng.random() < 0.5 else ("cx", b, a))
    return ops


def qft(n, rng):
    ops = []
    for i in range(n):
        ops.append(("u1", i, -1))
        for j in range(i + 1, n):
            ops.append(("u2", j, i))
    return ops


def trotter(n, steps, rng):
    if rng.random() < 0.5:
        edges = [(i, i + 1) for i in range(n - 1)]
    else:
        c = max(2, int(math.isqrt(n)))
        edges = []
        for q in range(n):
            if (q % c) + 1 < c and q + 1 < n:
                edges.append((q, q + 1))
            if q + c < n:
                edges.append((q, q + c))
    perm = list(range(n))
    rng.shuffle(perm)  # logical labelling is arbitrary
    ops = []
    for _ in range(steps):
        for a, b in edges:
            a, b = perm[a], perm[b]
            for _ in range(rng.choice((1, 3))):
                ops.append(("u2", a, b))
        for q in range(n):
            ops.append(("u1", q, -1))
    return ops


def qaoa(n, layers, rng):
    if n < 4:
        n = 4
    # random 3-regular-ish graph via configuration pairing
    for _ in range(50):
        stubs = [q for q in range(n) for _ in range(3)]
        rng.shuffle(stubs)
        edges = set()
        ok = True
        for k in range(0, len(stubs) - 1, 2):
            a, b = stubs[k], stubs[k + 1]
            if a == b or (min(a, b), max(a, b)) in edges:
                ok = False
                break
            edges.add((min(a, b), max(a, b)))
        if ok:
            break
    edges = sorted(edges)
    ops = [("u1", q, -1) for q in range(n)]
    for _ in range(layers):
        el = list(edges)
        rng.shuffle(el)
        for a, b in el:
            ops.append(("cx", a, b))
            ops.append(("u1", b, -1))
            ops.append(("cx", a, b))
        for q in range(n):
            ops.append(("u1", q, -1))
    return ops, n


def sparse(n, m, rng):
    ops = []
    for _ in range(m):
        a = rng.randrange(n)
        b = (a + 1 + rng.randrange(n - 1)) % n
        _u1(ops, rng, a)
        ops.append(("cx", a, b))
    for q in range(n):
        if rng.random() < 0.3:
            ops.append(("measure", q, -1))
    return ops


FAMILIES = ["rand2q", "qv", "qft", "trotter", "qaoa", "sparse"]


def make_circuit(family, n, rng, scale=1.0):
    """(name-suffix, n, ops); scale shrinks/grows depth (curriculum)."""
    if family == "rand2q":
        d = max(2, int(rng.randint(3, 3 + n) * scale))
        return f"rand2q_n{n}_d{d}", n, rand2q(n, d, rng)
    if family == "qv":
        d = max(1, int(rng.randint(2, max(3, n // 2 + 1)) * scale))
        return f"qv_n{n}_d{d}", n, qv(n, d, rng)
    if family == "qft":
        m = max(3, min(n, int(n * min(1.0, scale + 0.3))))
        return f"qft_n{m}", m, qft(m, rng)
    if family == "trotter":
        s = max(1, int(rng.randint(1, 3) * scale))
        return f"trotter_n{n}_s{s}", n, trotter(n, s, rng)
    if family == "qaoa":
        p = max(1, int(rng.randint(1, 2) * scale))
        ops, m = qaoa(n, p, rng)
        return f"qaoa_n{m}_p{p}", m, ops
    if family == "sparse":
        m = max(4, int(rng.randint(n, 4 * n) * scale))
        return f"sparse_n{n}_m{m}", n, sparse(n, m, rng)
    raise ValueError(family)


def build_jobs(seed, n_jobs, topologies=None, families=None, max_n=30, scale=1.0, min_frac=0.4):
    """Jobs dict for route_tool.mojo: random (topology, family, size) mixes."""
    rng = random.Random(seed)
    topologies = topologies or TRAIN_TOPOLOGIES
    families = families or FAMILIES
    tdefs = [topology(k, s) for k, s in topologies]
    targets = [{"name": nm, "n": n, "edges": [list(e) for e in es], "family": FAMILY_OF[k]}
               for (k, _), (nm, n, es) in zip(topologies, tdefs)]
    circuits, jobs = [], []
    for j in range(n_jobs):
        ti = rng.randrange(len(targets))
        N = targets[ti]["n"]
        hi = min(N, max_n)
        lo = max(3, min(hi, int(N * min_frac)))
        n = rng.randint(lo, hi)
        fam = families[rng.randrange(len(families))]
        name, m, ops = make_circuit(fam, n, rng, scale)
        if m > N:
            continue
        circuits.append({"name": f"{name}_s{seed}_{j}", "n": m, "ops": [list(o) for o in ops], "family": fam})
        jobs.append([len(circuits) - 1, ti, seed * 100000 + j])
    return {"targets": targets, "circuits": circuits, "jobs": jobs}
