"""Vendor-neutral compilation targets, one per hardware modality.

Every target is plain data: number of qubits, undirected coupling edges, a
native basis (1q gates + exactly one 2q gate family), and a simple noise
profile used to compute an estimated success probability (ESP). No modality
is privileged; each SDK compiles to the same data.

Noise profiles are order-of-magnitude public figures for each modality
(2025-2026 devices), used only to weight 1q/2q/time trade-offs consistently.
"""
import math
from dataclasses import dataclass, field

import rustworkx as rx


@dataclass
class NoiseProfile:
    err_1q: float
    err_2q: float
    t_1q_ns: float
    t_2q_ns: float
    t1_us: float
    t2_us: float
    virtual_z: bool = True  # rz is a frame change (zero time/error)


@dataclass
class Target:
    name: str
    modality: str
    num_qubits: int
    edges: list  # undirected (a, b) pairs
    basis_1q: list
    gate_2q: str
    noise: NoiseProfile
    description: str = ""
    edge_set: set = field(default_factory=set)

    def __post_init__(self):
        self.edge_set = {(a, b) for a, b in self.edges} | {(b, a) for a, b in self.edges}

    @property
    def basis(self):
        return list(self.basis_1q) + [self.gate_2q]

    @property
    def all_to_all(self):
        n = self.num_qubits
        return len(self.edges) >= n * (n - 1) // 2


SC = NoiseProfile(err_1q=2.5e-4, err_2q=3e-3, t_1q_ns=30, t_2q_ns=70, t1_us=250, t2_us=150)
ION = NoiseProfile(err_1q=3e-5, err_2q=1e-3, t_1q_ns=5_000, t_2q_ns=250_000, t1_us=1e8, t2_us=1e6)
ATOM = NoiseProfile(err_1q=5e-4, err_2q=5e-3, t_1q_ns=1_000, t_2q_ns=400, t1_us=4e6, t2_us=1e6)
GENERIC = NoiseProfile(err_1q=1e-3, err_2q=1e-2, t_1q_ns=50, t_2q_ns=300, t1_us=100, t2_us=100, virtual_z=False)


def heavy_hex_edges(min_qubits):
    """Smallest odd-distance rustworkx heavy-hex lattice with >= min_qubits (as Benchpress)."""
    d = 3
    while (5 * d * d - 2 * d - 1) // 2 < min_qubits:
        d += 2
    g = rx.generators.heavy_hex_graph(d)
    return len(g), list(g.edge_list())


def grid_edges(min_qubits, radius=1.0):
    """Square lattice; radius > 1 adds longer-range couplings (neutral atoms)."""
    side = math.ceil(math.sqrt(min_qubits))
    coords = [(r, c) for r in range(side) for c in range(side)]
    edges = []
    for i, (r1, c1) in enumerate(coords):
        for j in range(i + 1, len(coords)):
            r2, c2 = coords[j]
            if math.hypot(r1 - r2, c1 - c2) <= radius + 1e-9:
                edges.append((i, j))
    return len(coords), edges


def diagonal_grid_edges(min_qubits):
    """Rotated (diagonal) square lattice: each qubit couples to its 4 diagonal
    neighbours, diamond-shaped boundary (Sycamore/Willow-style layout)."""
    m = 2
    while True:
        coords = [(r, c) for r in range(m) for c in range(m) if (r + c) % 2 == 0]
        if len(coords) >= min_qubits:
            break
        m += 1
    index = {rc: i for i, rc in enumerate(coords)}
    edges = []
    for (r, c), i in index.items():
        for dr, dc in ((1, 1), (1, -1)):
            j = index.get((r + dr, c + dc))
            if j is not None:
                edges.append((i, j))
    return len(coords), edges


def octagonal_edges(min_qubits):
    """Rings of 8 qubits tiled in a row-major grid of octagons; neighbouring
    octagons share 2 couplers on each facing side (Aspen-style lattice)."""
    k = max(1, math.ceil(min_qubits / 8))
    cols = math.ceil(math.sqrt(k))
    rows = math.ceil(k / cols)
    edges = []
    def q(o, i):
        return 8 * o + i
    for o in range(rows * cols):
        for i in range(8):
            edges.append((q(o, i), q(o, (i + 1) % 8)))
    for r in range(rows):
        for c in range(cols):
            o = r * cols + c
            if c + 1 < cols:  # right side (1,2) couples to left side (6,5) of right neighbour
                edges += [(q(o, 1), q(o + 1, 6)), (q(o, 2), q(o + 1, 5))]
            if r + 1 < rows:  # bottom side (3,4) couples to top side (0,7) of lower neighbour
                d = o + cols
                edges += [(q(o, 4), q(d, 7)), (q(o, 3), q(d, 0))]
    return 8 * rows * cols, edges


def ion_zone_ring_edges(min_qubits, ions_per_zone=2):
    """QCCD-style trap: gate zones arranged in a ring; ions can interact inside a
    zone or with ions in the adjacent zones (shuttling = inserted swaps)."""
    zones = max(3, math.ceil(min_qubits / ions_per_zone))
    n = zones * ions_per_zone
    edges = set()
    for z in range(zones):
        members = [z * ions_per_zone + k for k in range(ions_per_zone)]
        nxt = [((z + 1) % zones) * ions_per_zone + k for k in range(ions_per_zone)]
        for a in members:
            for b in members + nxt:
                if a != b:
                    edges.add((min(a, b), max(a, b)))
    return n, sorted(edges)


def tree_edges(min_qubits):
    """Complete binary tree (Benchpress FlexibleBackend 'tree')."""
    levels = 1
    while 2 ** (levels + 1) - 1 < min_qubits:
        levels += 1
    n = 2 ** (levels + 1) - 1
    return n, [(i, (i - 1) // 2) for i in range(1, n)]


def torus_edges(min_qubits):
    side = max(3, math.ceil(math.sqrt(min_qubits)))
    n = side * side
    edges = set()
    for r in range(side):
        for c in range(side):
            i = r * side + c
            for j in (r * side + (c + 1) % side, ((r + 1) % side) * side + c):
                edges.add((min(i, j), max(i, j)))
    return n, sorted(edges)


def make_target(kind, min_qubits):
    if kind == "sc_heavy_hex":
        n, e = heavy_hex_edges(min_qubits)
        return Target(kind, "superconducting", n, e, ["rz", "sx", "x"], "cz", SC,
                      "heavy-hex lattice, rz/sx/x + cz")
    if kind == "sc_grid":
        n, e = grid_edges(min_qubits, 1.0)
        return Target(kind, "superconducting", n, e, ["rz", "rx", "ry"], "cz", SC,
                      "square lattice, rz/rx/ry + cz")
    if kind == "ion_all_to_all":
        n = max(min_qubits, 2)
        e = [(a, b) for a in range(n) for b in range(a + 1, n)]
        return Target(kind, "trapped_ion", n, e, ["rz", "rx", "ry"], "rxx", ION,
                      "all-to-all, rz/rx/ry + rxx (Molmer-Sorensen)")
    if kind == "neutral_atom":
        n, e = grid_edges(min_qubits, 2.0)
        return Target(kind, "neutral_atom", n, e, ["rz", "rx", "ry"], "cz", ATOM,
                      "square lattice, interaction radius 2, rz/rx/ry + cz")
    if kind == "sc_diagonal_grid":
        n, e = diagonal_grid_edges(min_qubits)
        return Target(kind, "superconducting", n, e, ["rz", "rx", "ry"], "cz", SC,
                      "rotated square lattice with diagonal couplers (Sycamore/Willow-style), rz/rx/ry + cz")
    if kind == "sc_octagonal":
        n, e = octagonal_edges(min_qubits)
        return Target(kind, "superconducting", n, e, ["rz", "rx", "ry"], "cz", SC,
                      "tiled octagon rings (Aspen-style), rz/rx/ry + cz")
    if kind == "ion_zone_ring":
        n, e = ion_zone_ring_edges(min_qubits)
        return Target(kind, "trapped_ion", n, e, ["rz", "rx", "ry"], "rzz", ION,
                      "QCCD-style ring of 2-ion gate zones, interactions within/adjacent zones (shuttling as swaps), rz/rx/ry + rzz")
    if kind == "neutral_atom_reconfigurable":
        n = max(min_qubits, 2)
        e = [(a, b) for a in range(n) for b in range(a + 1, n)]
        return Target(kind, "neutral_atom", n, e, ["rz", "rx", "ry"], "cz", ATOM,
                      "movable atoms (AOD transport): any pair can interact, rz/rx/ry + cz")
    if kind == "tree":
        n, e = tree_edges(min_qubits)
        return Target(kind, "abstract", n, e, ["rz", "sx", "x"], "cx", GENERIC, "binary tree, rz/sx/x + cx")
    if kind == "torus":
        n, e = torus_edges(min_qubits)
        return Target(kind, "abstract", n, e, ["rz", "sx", "x"], "cx", GENERIC, "2D torus, rz/sx/x + cx")
    if kind == "linear_cx":
        n = max(min_qubits, 2)
        e = [(i, i + 1) for i in range(n - 1)]
        return Target(kind, "abstract", n, e, ["rz", "sx", "x"], "cx", GENERIC,
                      "linear chain, rz/sx/x + cx")
    raise ValueError(kind)


# Benchpress' abstract set is all-to-all, square, heavy-hex, linear. The rest
# are modalities/topologies Benchpress does not evaluate.
ALL_TARGETS = [
    "sc_heavy_hex", "sc_grid", "sc_diagonal_grid", "sc_octagonal",
    "ion_all_to_all", "ion_zone_ring",
    "neutral_atom", "neutral_atom_reconfigurable",
    "linear_cx", "tree", "torus",
]
