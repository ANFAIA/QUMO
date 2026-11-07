"""Coupling-topology spec parser, slug and connectivity.

STUB (W-1): signatures follow MOJO_PORT.md 4.6 and 16 (C3); section 5 does not list them, so
they are PROVISIONAL until C3 lands. Bodies raise or abort with a "STUB" message.
"""

from std.os import abort


@fieldwise_init
struct Topology(Copyable):
    """Qubit count and flat undirected edge pairs `[a0, b0, a1, b1, ...]` with a < b, sorted."""

    var n: Int
    var edges: List[Int]

    def __init__(out self):
        self.n = 0
        self.edges = List[Int]()


def parse_topology(spec: String) raises -> Topology:
    """`line:N`, `ring:N`, `grid:RxC`, `king:RxC`, `complete:N`, `star:N`, `heavyhex_frag:NAME`, `edges:a-b,c-d`.
    QM-E641 on a bad spec."""
    raise Error("QM-E640: qumo.rl.topology.parse_topology is a STUB (component C3)")


def topology_slug(spec: String) raises -> String:
    raise Error("QM-E640: qumo.rl.topology.topology_slug is a STUB (component C3)")


def is_connected(n: Int, edges: List[Int]) -> Bool:
    abort("qumo.rl.topology.is_connected: STUB (component C3)")
