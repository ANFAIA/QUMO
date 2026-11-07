"""Environment configuration, gateset construction and the metadata-only ModelFile for PolicyEnv.

STUB (W-1): the signatures of MOJO_PORT.md section 5 only. Component C6 replaces
this file; every behavioural body raises or returns a zero value marked STUB.
"""

from std.os import abort
from qumo.compiler.policy import ModelFile, MODEL_KIND_PERM


@fieldwise_init
struct EnvConfig(Copyable):
    var kind: Int  # MODEL_KIND_PERM | LINEAR | CLIFFORD | PAULI
    var n: Int
    var actions: List[Int]  # flat (opcode, q0, q1|-1) in gateset order
    var difficulty: Int  # 1
    var depth_slope: Int  # 2
    var max_depth: Int  # 128
    var w_cnots: Float32  # metrics weights: n_cnots 0.01
    var w_layers_cnots: Float32  # 0.0
    var w_layers: Float32  # 0.0
    var w_gates: Float32  # 0.0001
    var add_inverts: Bool  # True (plain kinds)
    var add_perms: Bool  # True
    var max_rotations: Int  # 5 (Pauli)
    var pauli_diff_scale: Int  # 16
    var num_qubits_decay: Float32  # 0.5
    var final_pauli_layers: Int  # 7
    var pauli_layer_reward: Float32  # 0.01

    def __init__(out self):
        """The qiskit-gym defaults of MOJO_PORT.md 19.1 (kind permutation, no qubits, no actions)."""
        self.kind = MODEL_KIND_PERM
        self.n = 0
        self.actions = List[Int]()
        self.difficulty = 1
        self.depth_slope = 2
        self.max_depth = 128
        self.w_cnots = 0.01
        self.w_layers_cnots = 0.0
        self.w_layers = 0.0
        self.w_gates = 0.0001
        self.add_inverts = True
        self.add_perms = True
        self.max_rotations = 5
        self.pauli_diff_scale = 16
        self.num_qubits_decay = 0.5
        self.final_pauli_layers = 7
        self.pauli_layer_reward = 0.01


def build_gateset(kind: Int, topo_n: Int, edges: List[Int]) raises -> List[Int]:
    """Flat triples, qiskit-gym order. The environment's n is (max qubit over the edges) + 1; QM-E641 if it
    differs from topo_n (a trailing isolated qubit). Emitted gate names are only {H, S, CX, SWAP} (6.1)."""
    raise Error("QM-E640: qumo.rl.gateset.build_gateset is a STUB (component C6)")


def validate_actions(cfg: EnvConfig) raises:
    """QM-E642 for any gate outside {H, S, CX, SWAP} or bad qubit ids."""
    raise Error("QM-E640: qumo.rl.gateset.validate_actions is a STUB (component C6)")


def shell_model(cfg: EnvConfig) -> ModelFile:
    """Metadata-only ModelFile for PolicyEnv (num_qubits, actions, max_rotations set)."""
    # STUB: returns the default (empty) ModelFile.
    return ModelFile()
