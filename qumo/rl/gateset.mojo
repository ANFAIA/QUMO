"""Environment configuration, gateset construction and the metadata-only ModelFile for PolicyEnv.

The gateset follows the reference gym construction used by the trainer (MOJO_PORT.md 6.1, envs.md section 2):

* the environment's qubit count is `max qubit over the edges + 1` (a trailing isolated qubit is not counted, so
  `build_gateset` raises QM-E641 when that differs from the topology's `n`);
* permutation: one `SWAP(a, b)` per undirected edge (`a < b`, lexicographic order);
* linear function: `CX(a, b)` and `CX(b, a)` for every edge, all directed pairs sorted lexicographically;
* Clifford and Pauli network: `H(0..n-1)`, `S(0..n-1)`, then the sorted directed `CX` list of the linear case.

Only the gates {H, S, CX, SWAP} are ever emitted and `TrainEnv` accepts nothing else (`validate_actions`, QM-E642):
the shipped step goldens contain only these four gates, and the Pauli branch of `PolicyEnv` is not verified for the
others (MOJO_PORT.md 6.1, correction C-8).
"""

from std.collections import List
from qumo.compiler.policy import (
    ModelFile,
    MODEL_KIND_CLIFFORD,
    MODEL_KIND_LINEAR,
    MODEL_KIND_PAULI,
    MODEL_KIND_PERM,
    model_kind_name,
)
from qumo.ir.gates import Op

comptime MAX_ENV_QUBITS = 4096
comptime MAX_PACKED_ROWS = 64  # tableau rows are UInt64 masks: 2n <= 64 for Clifford / Pauli, n <= 64 for linear
comptime MAX_ROTATION_LAYERS = 64


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
        """The reference gym defaults of MOJO_PORT.md 19.1 (kind permutation, no qubits, no actions)."""
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


def _e641(why: String) -> Error:
    return Error("QM-E641: " + why + ". Hint: pass a connected coupling graph whose highest qubit is n-1.")


def _e642(why: String) -> Error:
    return Error(
        "QM-E642: invalid training environment: "
        + why
        + ". Hint: the environment supports the gates H, S, CX and SWAP on valid qubits only."
    )


def obs_size_of(cfg: EnvConfig) -> Int:
    """Width of the (sparse) observation space: n^2 for permutation and linear, 4 n^2 for Clifford and
    2n (2n + max_rotations) for Pauli networks."""
    var n = cfg.n
    if cfg.kind == MODEL_KIND_CLIFFORD:
        return 4 * n * n
    if cfg.kind == MODEL_KIND_PAULI:
        return 2 * n * (2 * n + cfg.max_rotations)
    return n * n


def build_gateset(kind: Int, topo_n: Int, edges: List[Int]) raises -> List[Int]:
    """Flat triples `(opcode, q0, q1)` in reference-gym order (`q1 = -1` for 1-qubit gates).

    `edges` are flat undirected pairs; they are oriented `a < b`, sorted lexicographically and de-duplicated (a
    parsed `Topology` is already in that form). The environment's n is (max qubit over the edges) + 1; QM-E641 if it
    differs from `topo_n`, if there is no edge, or an edge is a self loop or has a negative endpoint. QM-E642 for a
    kind other than the four synthesis kinds."""
    if kind != MODEL_KIND_PERM and kind != MODEL_KIND_LINEAR and kind != MODEL_KIND_CLIFFORD and kind != MODEL_KIND_PAULI:
        raise _e642("kind " + String(kind) + " is not a synthesis kind")
    if len(edges) % 2 != 0:
        raise _e641("odd length of the flat edge list")
    var keys = List[Int]()
    var hi_q = -1
    for e in range(len(edges) // 2):
        var a = edges[2 * e]
        var b = edges[2 * e + 1]
        if a < 0 or b < 0:
            raise _e641("negative qubit in edge")
        if a == b:
            raise _e641("self loop on qubit " + String(a))
        var lo = a if a < b else b
        var hi = b if a < b else a
        keys.append(lo * MAX_ENV_QUBITS + hi)
        if hi > hi_q:
            hi_q = hi
    if len(keys) == 0:
        raise _e641("the coupling graph has no edge")
    var n = hi_q + 1
    if n != topo_n:
        raise _e641(
            "the highest qubit of the edges is "
            + String(hi_q)
            + " (environment n = "
            + String(n)
            + ") but the topology has n = "
            + String(topo_n)
            + " (trailing isolated qubit)"
        )
    if n > MAX_ENV_QUBITS:
        raise _e641("more than " + String(MAX_ENV_QUBITS) + " qubits")
    sort(keys)
    var pairs = List[Int]()  # unique, ascending (a < b)
    for i in range(len(keys)):
        if i == 0 or keys[i] != keys[i - 1]:
            pairs.append(keys[i])
    var out = List[Int]()
    if kind == MODEL_KIND_PERM:
        for i in range(len(pairs)):
            out.append(Op.SWAP)
            out.append(pairs[i] // MAX_ENV_QUBITS)
            out.append(pairs[i] % MAX_ENV_QUBITS)
        return out^
    if kind == MODEL_KIND_CLIFFORD or kind == MODEL_KIND_PAULI:
        for q in range(n):
            out.append(Op.H)
            out.append(q)
            out.append(-1)
        for q in range(n):
            out.append(Op.S)
            out.append(q)
            out.append(-1)
    var directed = List[Int]()
    for i in range(len(pairs)):
        var a = pairs[i] // MAX_ENV_QUBITS
        var b = pairs[i] % MAX_ENV_QUBITS
        directed.append(a * MAX_ENV_QUBITS + b)
        directed.append(b * MAX_ENV_QUBITS + a)
    sort(directed)
    for i in range(len(directed)):
        out.append(Op.CX)
        out.append(directed[i] // MAX_ENV_QUBITS)
        out.append(directed[i] % MAX_ENV_QUBITS)
    return out^


def validate_actions(cfg: EnvConfig) raises:
    """QM-E642 for any gate outside {H, S, CX, SWAP}, bad qubit ids, a permutation gateset that is not all SWAP, or a
    configuration the packed representation cannot hold. Called by `TrainEnv.__init__`."""
    var kind = cfg.kind
    if kind != MODEL_KIND_PERM and kind != MODEL_KIND_LINEAR and kind != MODEL_KIND_CLIFFORD and kind != MODEL_KIND_PAULI:
        raise _e642("kind " + String(kind) + " is not a synthesis kind")
    var n = cfg.n
    if n < 1 or n > MAX_ENV_QUBITS:
        raise _e642("n = " + String(n) + " is outside 1.." + String(MAX_ENV_QUBITS))
    if (kind == MODEL_KIND_CLIFFORD or kind == MODEL_KIND_PAULI) and 2 * n > MAX_PACKED_ROWS:
        raise _e642("n = " + String(n) + " exceeds the packed tableau limit of " + String(MAX_PACKED_ROWS // 2) + " qubits")
    if kind == MODEL_KIND_LINEAR and n > MAX_PACKED_ROWS:
        raise _e642("n = " + String(n) + " exceeds the packed matrix limit of " + String(MAX_PACKED_ROWS) + " qubits")
    if kind == MODEL_KIND_PAULI and n > MAX_PACKED_ROWS // 2:
        raise _e642("n = " + String(n) + " exceeds the Pauli limit")
    if len(cfg.actions) % 3 != 0 or len(cfg.actions) == 0:
        raise _e642("the action list must hold at least one (opcode, q0, q1) triple")
    for i in range(len(cfg.actions) // 3):
        var op = cfg.actions[3 * i]
        var a = cfg.actions[3 * i + 1]
        var b = cfg.actions[3 * i + 2]
        if op != Op.H and op != Op.S and op != Op.CX and op != Op.SWAP:
            raise _e642("action " + String(i) + " uses opcode " + String(op) + ", outside {H, S, CX, SWAP}")
        if a < 0 or a >= n:
            raise _e642("action " + String(i) + " has qubit " + String(a) + " outside 0.." + String(n - 1))
        if op == Op.H or op == Op.S:
            if b != -1:
                raise _e642("action " + String(i) + " is a 1-qubit gate with a second operand")
        else:
            if b < 0 or b >= n:
                raise _e642("action " + String(i) + " has qubit " + String(b) + " outside 0.." + String(n - 1))
            if a == b:
                raise _e642("action " + String(i) + " acts twice on qubit " + String(a))
        if kind == MODEL_KIND_PERM and op != Op.SWAP:
            raise _e642("a permutation gateset holds SWAP actions only (action " + String(i) + ")")
    if cfg.difficulty < 0:
        raise _e642("difficulty must be >= 0")
    if cfg.depth_slope < 0 or cfg.max_depth < 0:
        raise _e642("depth_slope and max_depth must be >= 0")
    if kind == MODEL_KIND_PAULI:
        if cfg.max_rotations < 0 or cfg.max_rotations > MAX_ROTATION_LAYERS:
            raise _e642("max_rotations must be in 0.." + String(MAX_ROTATION_LAYERS))
        if cfg.final_pauli_layers < 0 or cfg.final_pauli_layers > MAX_ROTATION_LAYERS:
            raise _e642("final_pauli_layers must be in 0.." + String(MAX_ROTATION_LAYERS))
        if cfg.pauli_diff_scale < 1:
            raise _e642("pauli_diff_scale must be >= 1")


def shell_model(cfg: EnvConfig) -> ModelFile:
    """Metadata-only ModelFile for PolicyEnv: num_qubits, actions, kind, max_depth and (Pauli) max_rotations set,
    no tensors."""
    var m = ModelFile()
    m.kind_code = cfg.kind
    m.kind = model_kind_name(cfg.kind)
    m.num_qubits = cfg.n
    m.actions = cfg.actions.copy()
    m.max_depth = cfg.max_depth
    m.max_rotations = cfg.max_rotations if cfg.kind == MODEL_KIND_PAULI else 0
    m.obs_size = obs_size_of(cfg)
    return m^
