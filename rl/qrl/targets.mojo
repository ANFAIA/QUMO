"""The 25 training and 10 evaluation neutral topologies (ordered edge lists, verbatim emission order).

Vendor-neutral coupling graphs for router training: grid, heavy-hex, diagonal grid, octagonal, line, ring, tree,
torus, neutral-atom radius-2 grid and ion zone ring. Every generator emits its edges in the order of the
reference builders of `bench/neutral/targets.py` (that order is the active-swap order of the router, so it is part
of the contract, and `tests/fixtures/topologies.json` pins it for all 35 topologies).

Edges are a flat list `[a0, b0, a1, b1, ...]` (pairs, not necessarily sorted inside a pair: ring wrap-around and
tree edges keep the builders' orientation). Names follow the router jobs files (`grid4x4`, `heavy_hex19`, `diag18`,
`oct16`, `line8`, `ring14`, `tree15`, `torus16`, `atom_r2_16`, `ionring12`): the number in a name is the real qubit
count of the generated graph, which can exceed the requested minimum.

The heavy-hex lattice comes from `heavy_hex_lattice_edges` below, a copy of QUMO's `Target.heavy_hex_edges`
(the reference `heavy_hex_graph(d)` order; qrl must not import QUMO, and rl/qumo_bridge/tests/test_bridge_parity.mojo
checks the two stay identical). Pairs are re-oriented to `(min, max)` and the order of the lattice builder is kept.
"""

from std.collections import List
from std.math import sqrt

# Router job-file `family` label of each builder kind.
comptime KIND_GRID = "grid"
comptime KIND_HEAVY_HEX = "heavy_hex"
comptime KIND_DIAG = "diag"
comptime KIND_OCTAGONAL = "octagonal"
comptime KIND_LINE = "line"
comptime KIND_RING = "ring"
comptime KIND_TREE = "tree"
comptime KIND_TORUS = "torus"
comptime KIND_ATOM2 = "atom2"
comptime KIND_IONRING = "ionring"


struct RTopo(Copyable):
    """Name, qubit count, flat undirected edge pairs in emission order, and the job-file family label."""

    var name: String
    var n: Int
    var edges: List[Int]
    var family: String

    def __init__(out self):
        self.name = String()
        self.n = 0
        self.edges = List[Int]()
        self.family = String()

    def __init__(out self, name: String, n: Int, var edges: List[Int], family: String = String()):
        self.name = name
        self.n = n
        self.edges = edges^
        self.family = family

    def num_edges(self) -> Int:
        return len(self.edges) // 2


def family_of_kind(kind: String) raises -> String:
    """Label used in the `family` field of a jobs-file target (per-family gain tables)."""
    if kind == KIND_GRID:
        return "grid"
    if kind == KIND_HEAVY_HEX:
        return "heavy_hex"
    if kind == KIND_DIAG:
        return "diag_grid"
    if kind == KIND_OCTAGONAL:
        return "octagonal"
    if kind == KIND_LINE:
        return "line"
    if kind == KIND_RING:
        return "ring"
    if kind == KIND_TREE:
        return "tree"
    if kind == KIND_TORUS:
        return "torus"
    if kind == KIND_ATOM2:
        return "neutral_atom"
    if kind == KIND_IONRING:
        return "ion_ring"
    raise Error("QM-E640: unknown neutral topology kind '" + kind + "'. Hint: use grid heavy_hex diag octagonal line ring tree torus atom2 ionring.")


def _ceil_sqrt(x: Int) -> Int:
    """Smallest s with s * s >= x (x >= 0)."""
    var s = Int(sqrt(Float64(max(x, 0))))
    while s * s < x:
        s += 1
    while s > 0 and (s - 1) * (s - 1) >= x:
        s -= 1
    return s


def _push(mut e: List[Int], a: Int, b: Int):
    e.append(a)
    e.append(b)


def _sorted_unique_pairs(keys: List[Int], n: Int) -> List[Int]:
    """Flat `(a, b)` pairs, sorted lexicographically and deduplicated, from keys `a * n + b`."""
    var ks = keys.copy()
    sort(ks)
    var e = List[Int]()
    var last = -1
    for i in range(len(ks)):
        if ks[i] != last:
            _push(e, ks[i] // n, ks[i] % n)
            last = ks[i]
    return e^


def _pair_key(a: Int, b: Int, n: Int) -> Int:
    return min(a, b) * n + max(a, b)


def line_edges(size: Int) -> List[Int]:
    var e = List[Int]()
    for i in range(size - 1):
        _push(e, i, i + 1)
    return e^


def ring_edges(size: Int) -> List[Int]:
    """`(i, (i + 1) mod size)` for every i: the closing edge is `(size - 1, 0)`."""
    var e = List[Int]()
    for i in range(size):
        _push(e, i, (i + 1) % size)
    return e^


def grid_rect_edges(rows: Int, cols: Int) -> List[Int]:
    """Row-major `rows x cols` lattice: per qubit the right neighbour first, then the one below."""
    var e = List[Int]()
    for i in range(rows):
        for j in range(cols):
            var q = i * cols + j
            if j + 1 < cols:
                _push(e, q, q + 1)
            if i + 1 < rows:
                _push(e, q, q + cols)
    return e^


def heavy_hex_lattice_edges(d: Int) raises -> List[Int]:
    """Edges of the reference `heavy_hex_graph(d)` (d odd): (5d^2 - 2d - 1)/2 nodes. Copied from QUMO's
    `qumo.compiler.target.Target.heavy_hex_edges` (Apache-2.0, same repository), same emission order and error."""
    if d < 1 or d % 2 == 0:
        raise Error("QM-E612: heavy_hex distance must be odd and >= 1, got " + String(d) + ". Hint: use d = 3, 5, 7, ... ((5d^2-2d-1)/2 qubits).")
    var e = List[Int]()
    if d == 1:
        return e^
    var num_data = d * d
    var num_syn = (d - 1) * (d + 1) // 2
    var syn0 = num_data
    var flag0 = num_data + num_syn
    var chunk = (d + 1) // 2
    # data <-> flag along each row
    for i in range(d):
        for j in range(d - 1):
            var f = flag0 + i * (d - 1) + j
            e.append(i * d + j)
            e.append(f)
            e.append(f)
            e.append(i * d + j + 1)
    # data <-> syndrome at the row boundaries (left for even row pairs,
    # right for odd ones), exactly as the reference heavy_hex_graph
    for i in range(d - 1):
        var base = syn0 + i * chunk
        if i % 2 == 0:
            e.append(i * d)
            e.append(base)
            e.append((i + 1) * d)
            e.append(base)
        else:
            var s = base + chunk - 1
            e.append(i * d + (d - 1))
            e.append(s)
            e.append((i + 1) * d + (d - 1))
            e.append(s)
    # flag <-> syndrome
    for i in range(d - 1):
        var base = syn0 + i * chunk
        if i % 2 == 0:
            for j in range(1, chunk):
                e.append(base + j)
                e.append(flag0 + i * (d - 1) + 2 * (j - 1) + 1)
                e.append(base + j)
                e.append(flag0 + (i + 1) * (d - 1) + 2 * (j - 1) + 1)
        else:
            for j in range(chunk - 1):
                e.append(base + j)
                e.append(flag0 + i * (d - 1) + 2 * j)
                e.append(base + j)
                e.append(flag0 + (i + 1) * (d - 1) + 2 * j)
    return e^


def heavy_hex_n_edges(min_qubits: Int) raises -> Tuple[Int, List[Int]]:
    """Smallest odd distance d >= 3 whose heavy-hex lattice has >= `min_qubits` qubits; pairs as `(min, max)`."""
    var d = 3
    while (5 * d * d - 2 * d - 1) // 2 < min_qubits:
        d += 2
    var raw = heavy_hex_lattice_edges(d)
    var e = List[Int]()
    var i = 0
    while i + 1 < len(raw):
        _push(e, min(raw[i], raw[i + 1]), max(raw[i], raw[i + 1]))
        i += 2
    return (((5 * d * d - 2 * d - 1) // 2, e^))


def square_grid_edges(min_qubits: Int, radius: Float64) -> Tuple[Int, List[Int]]:
    """Square lattice with side ceil(sqrt(min)); all i < j (row-major) at distance <= radius (+1e-9)."""
    var side = _ceil_sqrt(min_qubits)
    var n = side * side
    var e = List[Int]()
    var lim = radius + 1e-9
    for i in range(n):
        var r1 = i // side
        var c1 = i % side
        for j in range(i + 1, n):
            var dr = Float64(r1 - j // side)
            var dc = Float64(c1 - j % side)
            if sqrt(dr * dr + dc * dc) <= lim:
                _push(e, i, j)
    return (n, e^)


def diagonal_grid_edges(min_qubits: Int) -> Tuple[Int, List[Int]]:
    """Rotated square lattice: sites with r + c even in an m x m board, each coupled to its two lower diagonals."""
    var m = 2
    while True:
        var cnt = 0
        for r in range(m):
            for c in range(m):
                if (r + c) % 2 == 0:
                    cnt += 1
        if cnt >= min_qubits:
            break
        m += 1
    var index = List[Int](length=m * m, fill=-1)
    var n = 0
    for r in range(m):
        for c in range(m):
            if (r + c) % 2 == 0:
                index[r * m + c] = n
                n += 1
    var e = List[Int]()
    for r in range(m):
        for c in range(m):
            var i = index[r * m + c]
            if i < 0:
                continue
            # (dr, dc) in ((1, 1), (1, -1)), in this order
            if r + 1 < m and c + 1 < m:
                var j = index[(r + 1) * m + c + 1]
                if j >= 0:
                    _push(e, i, j)
            if r + 1 < m and c - 1 >= 0:
                var j2 = index[(r + 1) * m + c - 1]
                if j2 >= 0:
                    _push(e, i, j2)
    return (n, e^)


def octagonal_edges(min_qubits: Int) -> Tuple[Int, List[Int]]:
    """Rings of 8 qubits tiled in a row-major grid of octagons; facing sides share two couplers."""
    var k = max(1, (min_qubits + 7) // 8)
    var cols = _ceil_sqrt(k)
    var rows = (k + cols - 1) // cols
    var e = List[Int]()
    for o in range(rows * cols):
        for i in range(8):
            _push(e, 8 * o + i, 8 * o + (i + 1) % 8)
    for r in range(rows):
        for c in range(cols):
            var o = r * cols + c
            if c + 1 < cols:
                _push(e, 8 * o + 1, 8 * (o + 1) + 6)
                _push(e, 8 * o + 2, 8 * (o + 1) + 5)
            if r + 1 < rows:
                var d = o + cols
                _push(e, 8 * o + 4, 8 * d + 7)
                _push(e, 8 * o + 3, 8 * d + 0)
    return (8 * rows * cols, e^)


def ion_zone_ring_edges(min_qubits: Int, ions_per_zone: Int = 2) -> Tuple[Int, List[Int]]:
    """Ring of gate zones; ions couple inside their zone and to the ions of the next zone. Sorted pairs."""
    var zones = max(3, (min_qubits + ions_per_zone - 1) // ions_per_zone)
    var n = zones * ions_per_zone
    var keys = List[Int]()
    for z in range(zones):
        for ka in range(ions_per_zone):
            var a = z * ions_per_zone + ka
            for kb in range(ions_per_zone):
                var b = z * ions_per_zone + kb
                if a != b:
                    keys.append(_pair_key(a, b, n))
            for kb in range(ions_per_zone):
                var b2 = ((z + 1) % zones) * ions_per_zone + kb
                if a != b2:
                    keys.append(_pair_key(a, b2, n))
    return (n, _sorted_unique_pairs(keys, n))


def tree_edges(min_qubits: Int) -> Tuple[Int, List[Int]]:
    """Complete binary tree with the smallest level count whose 2^(l+1) - 1 nodes cover `min_qubits`;
    edges `(i, (i - 1) // 2)` (child first)."""
    var levels = 1
    while (1 << (levels + 1)) - 1 < min_qubits:
        levels += 1
    var n = (1 << (levels + 1)) - 1
    var e = List[Int]()
    for i in range(1, n):
        _push(e, i, (i - 1) // 2)
    return (n, e^)


def torus_edges(min_qubits: Int) -> Tuple[Int, List[Int]]:
    """Square torus with side max(3, ceil(sqrt(min))). Sorted pairs."""
    var side = max(3, _ceil_sqrt(min_qubits))
    var n = side * side
    var keys = List[Int]()
    for r in range(side):
        for c in range(side):
            var i = r * side + c
            keys.append(_pair_key(i, r * side + (c + 1) % side, n))
            keys.append(_pair_key(i, ((r + 1) % side) * side + c, n))
    return (n, _sorted_unique_pairs(keys, n))


def neutral_topology(kind: String, param: Int, param2: Int = 0) raises -> RTopo:
    """One topology of the router training families.

    `kind`: grid (rows `param`, columns `param2`; a square when `param2` is 0), heavy_hex, diag, octagonal, line,
    ring, tree, torus, atom2 (square lattice, interaction radius 2), ionring (2 ions per zone). For every kind
    but grid and line/ring, `param` is the minimum qubit count.
    """
    var fam = family_of_kind(kind)
    if kind == KIND_GRID:
        var cols = param2 if param2 > 0 else param
        if param < 1 or cols < 1:
            raise Error("QM-E640: grid topology needs rows >= 1 and columns >= 1, got " + String(param) + "x" + String(cols) + ". Hint: pass a positive size.")
        return RTopo("grid" + String(param) + "x" + String(cols), param * cols, grid_rect_edges(param, cols), fam)
    if kind == KIND_HEAVY_HEX:
        var hh = heavy_hex_n_edges(param)
        return RTopo("heavy_hex" + String(hh[0]), hh[0], hh[1].copy(), fam)
    if kind == KIND_DIAG:
        var dg = diagonal_grid_edges(param)
        return RTopo("diag" + String(dg[0]), dg[0], dg[1].copy(), fam)
    if kind == KIND_OCTAGONAL:
        var oc = octagonal_edges(param)
        return RTopo("oct" + String(oc[0]), oc[0], oc[1].copy(), fam)
    if kind == KIND_LINE:
        if param < 2:
            raise Error("QM-E640: line topology needs at least 2 qubits, got " + String(param) + ". Hint: pass size >= 2.")
        return RTopo("line" + String(param), param, line_edges(param), fam)
    if kind == KIND_RING:
        if param < 3:
            raise Error("QM-E640: ring topology needs at least 3 qubits, got " + String(param) + ". Hint: pass size >= 3.")
        return RTopo("ring" + String(param), param, ring_edges(param), fam)
    if kind == KIND_TREE:
        var tr = tree_edges(param)
        return RTopo("tree" + String(tr[0]), tr[0], tr[1].copy(), fam)
    if kind == KIND_TORUS:
        var to = torus_edges(param)
        return RTopo("torus" + String(to[0]), to[0], to[1].copy(), fam)
    if kind == KIND_ATOM2:
        var at = square_grid_edges(param, 2.0)
        return RTopo("atom_r2_" + String(at[0]), at[0], at[1].copy(), fam)
    # KIND_IONRING (family_of_kind already rejected every unknown kind)
    var io = ion_zone_ring_edges(param)
    return RTopo("ionring" + String(io[0]), io[0], io[1].copy(), fam)


def train_topology_kinds() -> List[String]:
    var k: List[String] = [
        "grid", "grid", "grid", "grid", "grid",
        "heavy_hex", "heavy_hex",
        "diag", "diag",
        "octagonal", "octagonal",
        "line", "line", "line",
        "ring", "ring", "ring",
        "tree", "tree",
        "torus", "torus",
        "atom2", "atom2",
        "ionring", "ionring",
    ]
    return k^


def train_topology_params() -> List[Int]:
    """`param, param2` per training topology, flat (grid: rows, cols; the others: minimum qubits, 0)."""
    var p: List[Int] = [
        3, 3, 4, 4, 5, 5, 6, 6, 3, 5,
        19, 0, 57, 0,
        16, 0, 25, 0,
        16, 0, 32, 0,
        8, 0, 12, 0, 20, 0,
        8, 0, 14, 0, 20, 0,
        15, 0, 31, 0,
        16, 0, 25, 0,
        16, 0, 25, 0,
        12, 0, 16, 0,
    ]
    return p^


def train_topologies() raises -> List[RTopo]:
    """The 25 training topologies (grid 3x3 4x4 5x5 6x6 3x5; heavy-hex 19 57; diag 16 25; octagonal 16 32; line 8 12 20;
    ring 8 14 20; tree 15 31; torus 16 25; atom2 16 25; ionring 12 16)."""
    var kinds = train_topology_kinds()
    var ps = train_topology_params()
    var out = List[RTopo]()
    for i in range(len(kinds)):
        out.append(neutral_topology(kinds[i], ps[2 * i], ps[2 * i + 1]))
    return out^


def eval_topologies() raises -> List[RTopo]:
    """The 10 held-out evaluation topologies: grid 4x4, heavy-hex 19, diag 16, octagonal 16, line 16, ring 16,
    tree 15, torus 16, atom2 16, ionring 16."""
    var out = List[RTopo]()
    out.append(neutral_topology("grid", 4, 4))
    out.append(neutral_topology("heavy_hex", 19))
    out.append(neutral_topology("diag", 16))
    out.append(neutral_topology("octagonal", 16))
    out.append(neutral_topology("line", 16))
    out.append(neutral_topology("ring", 16))
    out.append(neutral_topology("tree", 15))
    out.append(neutral_topology("torus", 16))
    out.append(neutral_topology("atom2", 16))
    out.append(neutral_topology("ionring", 16))
    return out^


def topology_by_name(name: String) raises -> RTopo:
    """A training or evaluation topology by its job-file name (e.g. `grid4x4`, `ring14`)."""
    var tr = train_topologies()
    for i in range(len(tr)):
        if tr[i].name == name:
            return tr[i].copy()
    var ev = eval_topologies()
    for i in range(len(ev)):
        if ev[i].name == name:
            return ev[i].copy()
    raise Error("QM-E640: unknown topology '" + name + "'. Hint: use a name from the train or eval sets (grid4x4, heavy_hex19, diag18, oct16, line16, ring16, tree15, torus16, atom_r2_16, ionring16, ...).")
