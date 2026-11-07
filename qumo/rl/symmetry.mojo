"""Graph automorphisms and the observation / action twists derived from them.

STUB (W-1): `Twists` and `twists` follow MOJO_PORT.md sections 5 and 16 (C3); `automorphisms`,
`pauli_twists` and `pauli_qubit_perms_and_act_perms` follow 4.6 and are PROVISIONAL until C3 lands.
Bodies abort with a "STUB" message.
"""

from std.os import abort


@fieldwise_init
struct Twists(Copyable):
    """Parallel lists: `obs_perms[p]` and `act_perms[p]` describe symmetry `p`."""

    var obs_perms: List[List[Int]]
    var act_perms: List[List[Int]]

    def __init__(out self):
        self.obs_perms = List[List[Int]]()
        self.act_perms = List[List[Int]]()


def automorphisms(n: Int, edges: List[Int]) -> List[List[Int]]:
    """All automorphisms of the graph (flat edge pairs), sorted lexicographically."""
    abort("qumo.rl.symmetry.automorphisms: STUB (component C3)")


def twists(kind: Int, n: Int, actions: List[Int]) -> Twists:
    """Plain kinds (permutation / linear / Clifford): the ensemble maps; Pauli: empty. `actions` = flat triples."""
    abort("qumo.rl.symmetry.twists: STUB (component C3)")


def pauli_twists(n: Int, actions: List[Int], max_rot: Int) -> Twists:
    """Inference-side ensemble of a Pauli model (reproduces the manifest `symmetries`)."""
    abort("qumo.rl.symmetry.pauli_twists: STUB (component C3)")


def pauli_qubit_perms_and_act_perms(n: Int, actions: List[Int]) -> Twists:
    """Training-env tables: `obs_perms` holds the qubit permutations, `act_perms` the action permutations."""
    abort("qumo.rl.symmetry.pauli_qubit_perms_and_act_perms: STUB (component C3)")
