"""Coupling-graph specs shared by the QUMO AI-transpiler training scripts.

A topology spec is a short string:

    line:N            path 0-1-...-(N-1)
    ring:N            cycle
    grid:RxC          square lattice, row-major indices
    king:RxC          square lattice plus diagonals (neutral-atom radius-1.5 / king graph)
    complete:N        all-to-all (ions, reconfigurable atoms)
    star:N            qubit 0 coupled to every other qubit
    heavyhex_frag:NAME  heavy-hex fragments (see HEAVYHEX_FRAGMENTS)
    edges:0-1,1-2     explicit undirected edge list

`parse_topology(spec)` returns (num_qubits, edges) with edges as sorted (a, b)
pairs, a < b. Everything here is pure Python (no qiskit import) so the Mojo
tests and exporters can reuse it.
"""
from __future__ import annotations

# Fragments of the heavy-hex lattice that show up as induced subgraphs of
# collected blocks on QUMO's heavy-hex targets.
HEAVYHEX_FRAGMENTS = {
    # 7-qubit "H" shape (Falcon-style unit): 0-1-2, 1-3, 3-5, 4-5-6
    "H7": (7, [(0, 1), (1, 2), (1, 3), (3, 5), (4, 5), (5, 6)]),
    # the 12-qubit heavy-hex cell: a 12-cycle
    "cell12": (12, [(i, (i + 1) % 12) for i in range(12)]),
    # 12-ring-with-spokes: a 10-cycle with two pendant qubits on opposite sides
    # (a heavy-hex cell missing two corners, plus the two outgoing spokes)
    "ring10s2": (12, [(i, (i + 1) % 10) for i in range(10)] + [(0, 10), (5, 11)]),
    # 5-qubit "T" fragment (a degree-3 node with arms 2, 1, 1)
    "T5": (5, [(0, 1), (1, 2), (2, 3), (2, 4)]),
}


def _norm(edges):
    out = sorted({(min(a, b), max(a, b)) for a, b in edges if a != b})
    return out


def parse_topology(spec: str):
    if ":" not in spec:
        raise ValueError(f"bad topology spec {spec!r}: expected kind:arg")
    kind, arg = spec.split(":", 1)
    if kind == "line":
        n = int(arg)
        return n, _norm([(i, i + 1) for i in range(n - 1)])
    if kind == "ring":
        n = int(arg)
        return n, _norm([(i, (i + 1) % n) for i in range(n)])
    if kind in ("grid", "king"):
        r, c = map(int, arg.lower().split("x"))
        e = []
        for i in range(r):
            for j in range(c):
                q = i * c + j
                if j + 1 < c:
                    e.append((q, q + 1))
                if i + 1 < r:
                    e.append((q, q + c))
                if kind == "king" and i + 1 < r:
                    if j + 1 < c:
                        e.append((q, q + c + 1))
                    if j > 0:
                        e.append((q, q + c - 1))
        return r * c, _norm(e)
    if kind == "complete":
        n = int(arg)
        return n, _norm([(a, b) for a in range(n) for b in range(a + 1, n)])
    if kind == "star":
        n = int(arg)
        return n, _norm([(0, b) for b in range(1, n)])
    if kind == "heavyhex_frag":
        if arg not in HEAVYHEX_FRAGMENTS:
            raise ValueError(f"unknown heavy-hex fragment {arg!r}; known: {sorted(HEAVYHEX_FRAGMENTS)}")
        n, e = HEAVYHEX_FRAGMENTS[arg]
        return n, _norm(e)
    if kind == "edges":
        e = []
        for tok in arg.split(","):
            a, b = tok.split("-")
            e.append((int(a), int(b)))
        e = _norm(e)
        n = max(max(p) for p in e) + 1
        return n, e
    raise ValueError(f"unknown topology kind {kind!r}")


def topology_slug(spec: str) -> str:
    """File-system friendly name: 'grid:2x3' -> 'grid2x3'."""
    kind, arg = spec.split(":", 1)
    if kind == "edges":
        import hashlib
        return "edges" + hashlib.sha1(arg.encode()).hexdigest()[:8]
    if kind == "heavyhex_frag":
        return "hh" + arg
    return kind + arg.lower()


def is_connected(n, edges):
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    seen = {0}
    st = [0]
    while st:
        v = st.pop()
        for w in adj[v]:
            if w not in seen:
                seen.add(w)
                st.append(w)
    return len(seen) == n


def wl_hash(n, edges, iters=3):
    """Weisfeiler-Lehman colour-refinement hash (prefilter only; the Mojo
    loader recomputes it with the same rule and confirms with VF2).

    Rule (must match qumo/compiler/policy.mojo `wl_hash`):
      c0[v] = degree(v)
      c_{k+1}[v] = fnv1a64( c_k[v], sorted(c_k[w] for w in N(v)) )
      hash = fnv1a64( n, |E|, sorted(c_K) ) as 16 hex digits
    """
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    col = [len(adj[v]) for v in range(n)]
    for _ in range(iters):
        col = [fnv_words([col[v]] + sorted(col[w] for w in adj[v])) for v in range(n)]
    h = fnv_words([n, len(edges)] + sorted(col))
    return f"{h:016x}"


FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MASK64 = (1 << 64) - 1


def fnv1a64(data: bytes, h: int = FNV_OFFSET) -> int:
    for b in data:
        h ^= b
        h = (h * FNV_PRIME) & MASK64
    return h


def fnv_words(words):
    """FNV-1a over the little-endian 8-byte encoding of each (unsigned 64-bit) word."""
    h = FNV_OFFSET
    for w in words:
        w &= MASK64
        for k in range(8):
            h ^= (w >> (8 * k)) & 0xFF
            h = (h * FNV_PRIME) & MASK64
    return h
