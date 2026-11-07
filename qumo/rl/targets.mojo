"""The 25 training and 10 evaluation neutral topologies (ordered edge lists, verbatim emission order).

STUB (W-1): signatures follow MOJO_PORT.md 16 (C9); section 5 does not list them, so they are
PROVISIONAL until C9 lands. Bodies raise with a "STUB" message.
"""


@fieldwise_init
struct RTopo(Copyable):
    """Name, qubit count and flat undirected edge pairs in the generator's emission order."""

    var name: String
    var n: Int
    var edges: List[Int]

    def __init__(out self):
        self.name = String()
        self.n = 0
        self.edges = List[Int]()


def neutral_topology(name: String, param: Int, param2: Int = 0) raises -> RTopo:
    """line, ring, grid (param x param2), heavy_hex, diag grid, octagonal, tree, torus, atom2, ion zone ring."""
    raise Error("QM-E640: qumo.rl.targets.neutral_topology is a STUB (component C9)")


def train_topologies() raises -> List[RTopo]:
    """The 25 training topologies."""
    raise Error("QM-E640: qumo.rl.targets.train_topologies is a STUB (component C9)")


def eval_topologies() raises -> List[RTopo]:
    """The 10 evaluation topologies."""
    raise Error("QM-E640: qumo.rl.targets.eval_topologies is a STUB (component C9)")
