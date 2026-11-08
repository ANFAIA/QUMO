"""Coupling-topology spec parser, slug and connectivity (MOJO_PORT.md 4.6, component C3).

A topology spec is a short string:

    line:N              path 0-1-...-(N-1)
    ring:N              cycle
    grid:RxC            square lattice, row-major indices
    king:RxC            square lattice plus both diagonals (king graph)
    complete:N          all-to-all
    star:N              qubit 0 coupled to every other qubit
    heavyhex_frag:NAME  heavy-hex fragments H7, cell12, ring10s2, T5
    edges:0-1,1-2       explicit undirected edge list (n = largest index + 1)

`parse_topology` returns the qubit count and the undirected edges as a flat list `[a0, b0, a1, b1, ...]` with
`a < b`, sorted lexicographically and without duplicates or self loops (the same normalisation as the earlier helper
this module replaces). Bad specs raise QM-E641.

The Weisfeiler-Lehman hash is not re-implemented here: use `qumo.compiler.policy.wl_hash(n, edges)`.
"""

from std.collections import List
from qrl.model import fnv_words, hex16


# Resource guards: a spec naming more qubits than this is almost certainly a typo. `complete` stores
# N(N-1)/2 edges, so it gets a tighter bound.
comptime MAX_QUBITS = 4096
comptime MAX_COMPLETE = 1024
# Edge key = a * KEY_BASE + b, valid because every index is < MAX_QUBITS <= KEY_BASE.
comptime KEY_BASE = 4096


@fieldwise_init
struct Topology(Copyable):
    """Qubit count and flat undirected edge pairs `[a0, b0, a1, b1, ...]` with a < b, sorted, unique."""

    var n: Int
    var edges: List[Int]

    def __init__(out self):
        self.n = 0
        self.edges = List[Int]()

    def num_edges(self) -> Int:
        return len(self.edges) // 2


def _bad(spec: String, why: String) -> Error:
    return Error(
        "QM-E641: bad topology spec '"
        + spec
        + "': "
        + why
        + ". Hint: use line:N, ring:N, grid:RxC, king:RxC, complete:N, star:N, heavyhex_frag:{H7,cell12,ring10s2,T5}"
        + " or edges:a-b,c-d."
    )


def _parse_count(spec: String, text: String, what: String, lo: Int, hi: Int) raises -> Int:
    var t = String(text.strip())
    var v: Int
    try:
        v = Int(t)
    except:
        raise _bad(spec, what + " '" + t + "' is not an integer")
    if v < lo or v > hi:
        raise _bad(spec, what + " " + String(v) + " is outside [" + String(lo) + ", " + String(hi) + "]")
    return v


def _normalize(pairs: List[Int]) -> List[Int]:
    """Flat pairs (any order, any duplicates, self loops allowed) -> sorted unique flat pairs with a < b.
    Every index must be in [0, MAX_QUBITS)."""
    var keys = List[Int]()
    var ne = len(pairs) // 2
    for e in range(ne):
        var a = pairs[2 * e]
        var b = pairs[2 * e + 1]
        if a == b:
            continue
        var lo = a if a < b else b
        var hi = b if a < b else a
        keys.append(lo * KEY_BASE + hi)
    sort(keys)
    var out = List[Int](capacity=2 * len(keys))
    var prev = -1
    for i in range(len(keys)):
        var k = keys[i]
        if k == prev:
            continue
        prev = k
        out.append(k // KEY_BASE)
        out.append(k % KEY_BASE)
    return out^


def _fragment(name: String, mut n: Int, mut raw: List[Int]) -> Bool:
    """Heavy-hex fragments that show up as induced subgraphs of collected blocks on heavy-hex targets."""
    if name == "H7":
        # 7-qubit "H" shape (Falcon-style unit): 0-1-2, 1-3, 3-5, 4-5-6
        n = 7
        raw = [0, 1, 1, 2, 1, 3, 3, 5, 4, 5, 5, 6]
        return True
    if name == "cell12":
        # the 12-qubit heavy-hex cell: a 12-cycle
        n = 12
        raw = List[Int]()
        for i in range(12):
            raw.append(i)
            raw.append((i + 1) % 12)
        return True
    if name == "ring10s2":
        # a 10-cycle with two pendant qubits on opposite sides (heavy-hex cell missing two corners, plus the spokes)
        n = 12
        raw = List[Int]()
        for i in range(10):
            raw.append(i)
            raw.append((i + 1) % 10)
        raw.append(0)
        raw.append(10)
        raw.append(5)
        raw.append(11)
        return True
    if name == "T5":
        # 5-qubit "T" fragment (a degree-3 node with arms 2, 1, 1)
        n = 5
        raw = [0, 1, 1, 2, 2, 3, 2, 4]
        return True
    return False


def _split_dims(spec: String, arg: String) raises -> Tuple[Int, Int]:
    var low = arg.lower()
    var parts = low.split("x")
    if len(parts) != 2:
        raise _bad(spec, "expected RxC (for example grid:2x3)")
    var r = _parse_count(spec, String(parts[0]), "rows", 1, MAX_QUBITS)
    var c = _parse_count(spec, String(parts[1]), "columns", 1, MAX_QUBITS)
    if r * c > MAX_QUBITS:
        raise _bad(spec, String(r * c) + " qubits exceed the limit of " + String(MAX_QUBITS))
    return (r, c)


def parse_topology(spec: String) raises -> Topology:
    """`line:N`, `ring:N`, `grid:RxC`, `king:RxC`, `complete:N`, `star:N`, `heavyhex_frag:NAME`, `edges:a-b,c-d`.

    Edges are returned sorted with a < b and no duplicates. QM-E641 on a bad spec."""
    var colon = spec.find(":")
    if colon < 0:
        raise _bad(spec, "expected kind:arg")
    var kind = String(spec[byte=0:colon])
    var arg = String(spec[byte = colon + 1 :])
    var n = 0
    var raw = List[Int]()
    if kind == "line":
        n = _parse_count(spec, arg, "qubit count", 1, MAX_QUBITS)
        for i in range(n - 1):
            raw.append(i)
            raw.append(i + 1)
    elif kind == "ring":
        n = _parse_count(spec, arg, "qubit count", 1, MAX_QUBITS)
        # ring:1 has no edge and ring:2 a single edge, exactly as the earlier helper (self loops dropped, pairs unique)
        for i in range(n):
            raw.append(i)
            raw.append((i + 1) % n)
    elif kind == "grid" or kind == "king":
        var dims = _split_dims(spec, arg)
        var r = dims[0]
        var c = dims[1]
        n = r * c
        for i in range(r):
            for j in range(c):
                var q = i * c + j
                if j + 1 < c:
                    raw.append(q)
                    raw.append(q + 1)
                if i + 1 < r:
                    raw.append(q)
                    raw.append(q + c)
                if kind == "king" and i + 1 < r:
                    if j + 1 < c:
                        raw.append(q)
                        raw.append(q + c + 1)
                    if j > 0:
                        raw.append(q)
                        raw.append(q + c - 1)
    elif kind == "complete":
        n = _parse_count(spec, arg, "qubit count", 1, MAX_COMPLETE)
        for a in range(n):
            for b in range(a + 1, n):
                raw.append(a)
                raw.append(b)
    elif kind == "star":
        n = _parse_count(spec, arg, "qubit count", 1, MAX_QUBITS)
        for b in range(1, n):
            raw.append(0)
            raw.append(b)
    elif kind == "heavyhex_frag":
        if not _fragment(String(arg.strip()), n, raw):
            raise _bad(spec, "unknown heavy-hex fragment '" + arg + "'; known: H7, T5, cell12, ring10s2")
    elif kind == "edges":
        var toks = arg.split(",")
        for ti in range(len(toks)):
            var tok = String(toks[ti].strip())
            var ends = tok.split("-")
            if len(ends) != 2:
                raise _bad(spec, "edge token '" + tok + "' is not a-b")
            raw.append(_parse_count(spec, String(ends[0]), "qubit index", 0, MAX_QUBITS - 1))
            raw.append(_parse_count(spec, String(ends[1]), "qubit index", 0, MAX_QUBITS - 1))
    else:
        raise _bad(spec, "unknown topology kind '" + kind + "'")

    var edges = _normalize(raw)
    if kind == "edges":
        if len(edges) == 0:
            raise _bad(spec, "the edge list has no edge between two distinct qubits")
        # n = largest index + 1 (isolated trailing qubits cannot be expressed this way)
        var top = 0
        for i in range(len(edges)):
            if edges[i] > top:
                top = edges[i]
        n = top + 1
    return Topology(n, edges^)


def topology_slug(spec: String) raises -> String:
    """File-system friendly name: `grid:2x3` -> `grid2x3`, `heavyhex_frag:H7` -> `hhH7`.

    `edges:` specs get `edges` plus 8 hex digits of the FNV-1a hash of the normalised edge list (this replaces the
    SHA-1 prefix of the earlier helper and is only used for directory names)."""
    var colon = spec.find(":")
    if colon < 0:
        raise _bad(spec, "expected kind:arg")
    var kind = String(spec[byte=0:colon])
    var arg = String(spec[byte = colon + 1 :])
    if kind == "edges":
        var t = parse_topology(spec)
        var words = List[UInt64]()
        words.append(UInt64(t.n))
        words.append(UInt64(t.num_edges()))
        for i in range(len(t.edges)):
            words.append(UInt64(t.edges[i]))
        var h = hex16(fnv_words(words))
        return "edges" + String(h[byte=0:8])
    if kind == "heavyhex_frag":
        return "hh" + arg
    return kind + arg.lower()


def is_connected(n: Int, edges: List[Int]) -> Bool:
    """True iff the graph on `n` vertices with flat undirected `edges` is connected. A graph with no vertex is not
    connected (as in the earlier helper); an edge naming a vertex outside [0, n) makes the graph invalid -> False."""
    if n <= 0:
        return False
    var adj = List[List[Int]]()
    for _ in range(n):
        adj.append(List[Int]())
    var ne = len(edges) // 2
    for e in range(ne):
        var a = edges[2 * e]
        var b = edges[2 * e + 1]
        if a < 0 or a >= n or b < 0 or b >= n:
            return False
        adj[a].append(b)
        adj[b].append(a)
    var seen = List[Bool](length=n, fill=False)
    var stack = List[Int]()
    seen[0] = True
    stack.append(0)
    var count = 1
    while len(stack) > 0:
        var v = stack.pop()
        for k in range(len(adj[v])):
            var w = adj[v][k]
            if not seen[w]:
                seen[w] = True
                count += 1
                stack.append(w)
    return count == n
