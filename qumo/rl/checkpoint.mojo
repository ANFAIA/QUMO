"""Run-directory layout, f32 checkpoints, warm-start resolution and the read-only tensor-file importer.

MOJO_PORT.md 7.5 and 9.1, component C11a. A training run lives in one directory:

    RUN/run.json      the immutable run configuration (`RunInfo`): kind, topology, gateset, architecture, seed,
                      warm-start origin, budget, plus the driver's own configuration as an opaque JSON object
    RUN/best/         f32 model container of the best weights so far (valid `qumo.rlsynth.v0`, loads with
                      `load_model_file` and `BasicNet.from_model`, exports with `export_synth`)
    RUN/last/         f32 container of the latest weights plus the resumable training state (see below)
    RUN/state.json    human readable mirror of the training state stored in `last/` (never read back)
    RUN/log.jsonl     one JSON object per iteration (`append_log_line`, `truncate_log`)
    RUN/summary.json  final record (`RunSummary`); `finished` is true only after the last checkpoint

Atomicity. Every checkpoint directory is written by `container.write_model_dir`, i.e. staged in `<dir>.tmp`, read back,
then swapped in with two renames, so a reader never sees a half-written checkpoint and `recover_dir` repairs a kill
between the renames. The resumable training state is *inside* `last/`: the Adam moments are two extra f32 tensors
(`adam.m`, `adam.v`, flat) and the counters (iteration, difficulty, best tuple, elapsed time, Adam step count, seed,
config hash) are the manifest member `train_state`. Weights, optimizer and counters therefore commit as one unit, which
is what makes a resumed `--iters` run bit-identical to an uninterrupted one (T-RESUME). Plan 7.5 names this state
`state.json` + `adam.bin` as separate files; two extra files could not commit atomically with `last/`, so they are
folded into it, and `state.json` is only a mirror for humans and lane logs. Readers other than `BasicNet.from_model` and
`MLPPolicy.from_model` never see the extra tensors, and those two look tensors up by name.

Warm start. `resolve_init` implements the three cases of 7.5 (`auto`, a run directory, any model container).

Importer. `safetensors_import` reads the weight file of an old run (a tensor file: header length, JSON header, raw
data; F32, F16, BF16 and F64 are decoded to f32); it never writes the format. `basic_mlp_tensors` orders and
transposes the reference state dict into container tensors.

Errors: QM-E640 usage, QM-E644 container mismatch (from `container`), QM-E647 incompatible or unreadable warm-start
source / tensor file.
"""

from std.collections import List
from std.memory import bitcast
from std.os import makedirs
from std.os.path import exists, isdir, isfile
from qumo.ai.json import JsonValue, parse_json, json_quote, json_number, pretty_json
from qumo.compiler.policy import (
    ModelFile,
    gate_code,
    load_model_file,
    model_kind_code,
    model_kind_name,
    MODEL_KIND_CLIFFORD,
    MODEL_KIND_LINEAR,
    MODEL_KIND_PAULI,
    MODEL_KIND_PERM,
)
from qumo.ir.gates import op_name
from qumo.rl.container import (
    TensorOut,
    ModelDirInfo,
    write_model_dir,
    write_text_atomic,
    recover_dir,
    remove_tree,
)
from qumo.rl.digest import sha256_hex
from qumo.rl.gateset import EnvConfig, build_gateset, obs_size_of
from qumo.rl.nn import BasicNet, NetSpec
from qumo.rl.optim import Adam
from qumo.rl.tensor import CHUNK_ROWS, VW
from qumo.rl.symmetry import twists
from qumo.rl.topology import parse_topology

comptime RUN_FORMAT = "qumo.rlrun.v0"
comptime ENGINE = "qumo.rl (Mojo)"
comptime PRODUCER = "trained by the native QUMO stack (qumo.rl, Mojo): PPO on a native port of the reference synthesis environments"
comptime RUN_FILE = "run.json"
comptime STATE_FILE = "state.json"
comptime LOG_FILE = "log.jsonl"
comptime SUMMARY_FILE = "summary.json"
comptime BEST_DIR = "best"
comptime LAST_DIR = "last"
comptime ADAM_M = "adam.m"
comptime ADAM_V = "adam.v"


def _e640(what: String) -> Error:
    return Error("QM-E640: " + what + ". Hint: see training/ai_transpiler/README.md for the run directory layout.")


def _e647(what: String) -> Error:
    return Error("QM-E647: " + what + ". Hint: the source must be a run directory, a model container or a tensor file of the same architecture.")


# ------------------------------------------------------------------------------------------------ JSON text helpers


def ints_json(v: List[Int]) -> String:
    """`[1,2,3]` (compact)."""
    var s = String("[")
    for i in range(len(v)):
        if i > 0:
            s += ","
        s += String(v[i])
    s += "]"
    return s^


def perms_json(p: List[List[Int]]) -> String:
    """`[[..],[..]]` (compact)."""
    var s = String("[")
    for i in range(len(p)):
        if i > 0:
            s += ","
        s += ints_json(p[i])
    s += "]"
    return s^


def pairs_json(flat: List[Int]) -> String:
    """Flat pairs `[a0,b0,a1,b1]` -> `[[a0,b0],[a1,b1]]`."""
    var s = String("[")
    for i in range(len(flat) // 2):
        if i > 0:
            s += ","
        s += "[" + String(flat[2 * i]) + "," + String(flat[2 * i + 1]) + "]"
    s += "]"
    return s^


def actions_json(flat: List[Int]) -> String:
    """Flat `(opcode, q0, q1)` triples -> `[["SWAP",0,1],["H",2,-1],...]` (gate names in upper case, as in the
    shipped manifests)."""
    var s = String("[")
    for i in range(len(flat) // 3):
        if i > 0:
            s += ","
        s += (
            "["
            + json_quote(op_name(flat[3 * i]).upper())
            + ","
            + String(flat[3 * i + 1])
            + ","
            + String(flat[3 * i + 2])
            + "]"
        )
    s += "]"
    return s^


def obs_layout_text(kind: Int, n: Int, max_rot: Int) -> String:
    """The `obs.layout` string of the shipped manifests."""
    if kind == MODEL_KIND_PERM or kind == MODEL_KIND_LINEAR:
        return String(n) + "x" + String(n) + " row-major: perm i*n+state[i] / matrix bits"
    if kind == MODEL_KIND_CLIFFORD:
        return String(2 * n) + "x" + String(2 * n) + " symplectic of C^-1, rows = coords (x..., z...), cols = generators"
    return (
        String(2 * n)
        + "x"
        + String(2 * n + max_rot)
        + ": tableau of C^-1 + first "
        + String(max_rot)
        + " active rotation columns"
    )


def obs_json(kind: Int, n: Int, max_rot: Int, obs_size: Int) -> String:
    return (
        '{"encoding":"sparse_index","size":'
        + String(obs_size)
        + ',"layout":'
        + json_quote(obs_layout_text(kind, n, max_rot))
        + ',"max_rotations":'
        + String(max_rot)
        + "}"
    )


def arch_json(emb: Int, common: List[Int]) -> String:
    return (
        '{"type":"basic_mlp","emb":'
        + String(emb)
        + ',"common":'
        + ints_json(common)
        + ',"act":"relu","first_layer_transposed":true}'
    )


def symmetries_json(obs_perms: List[List[Int]], act_perms: List[List[Int]]) -> String:
    return '{"obs_perms":' + perms_json(obs_perms) + ',"act_perms":' + perms_json(act_perms) + "}"


def _basename(path: String) -> String:
    var bs = path.as_bytes()
    var end = len(bs)
    while end > 0 and Int(bs[end - 1]) == ord("/"):
        end -= 1
    var i = end - 1
    while i >= 0 and Int(bs[i]) != ord("/"):
        i -= 1
    return String(unsafe_from_utf8=bs[i + 1 : end])


def _parent(path: String) -> String:
    var bs = path.as_bytes()
    var end = len(bs)
    while end > 0 and Int(bs[end - 1]) == ord("/"):
        end -= 1
    var i = end - 1
    while i >= 0 and Int(bs[i]) != ord("/"):
        i -= 1
    if i < 0:
        return String(".")
    if i == 0:
        return String("/")
    return String(unsafe_from_utf8=bs[0:i])


def _strip_slashes(path: String) -> String:
    var bs = path.as_bytes()
    var end = len(bs)
    while end > 1 and Int(bs[end - 1]) == ord("/"):
        end -= 1
    return String(unsafe_from_utf8=bs[0:end])


def _read_text(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def _get_u64(v: JsonValue, key: String, default: UInt64) raises -> UInt64:
    if not v.has(key) or v.get(key).is_null():
        return default
    return UInt64(Int(v.get(key).number_text()))


# ------------------------------------------------------------------------------------------------ run.json


struct RunInfo(Copyable):
    """Content of `run.json`: everything that identifies a run and does not change while it trains.

    `config_json` is the driver's own configuration (PPO constants, evaluation settings, ...) as one JSON object text;
    it is stored verbatim and hashed into `config_hash_of`. `init` is the warm-start source as typed by the user (empty
    for a fresh run) and `init_final_difficulty` the difficulty it ended at (-1 when unknown)."""

    var id: String
    var kind: Int  # MODEL_KIND_PERM | LINEAR | CLIFFORD | PAULI
    var topology: String  # spec string such as "grid:3x3"
    var num_qubits: Int
    var edges: List[Int]  # flat sorted pairs a < b
    var actions: List[Int]  # flat (opcode, q0, q1|-1) triples in gateset order
    var max_rotations: Int  # Pauli networks, else 0
    var max_depth: Int
    var emb: Int
    var common: List[Int]
    var seed: Int
    var start_iteration: Int
    var start_difficulty: Int
    var cores: Int
    var chunk_rows: Int
    var vw: Int
    var budget_s: Float64  # 0 in `--iters` mode
    var iters: Int  # 0 in `--budget-s` mode
    var init: String
    var init_final_difficulty: Int
    var config_json: String

    def __init__(out self):
        self.id = String()
        self.kind = MODEL_KIND_PERM
        self.topology = String()
        self.num_qubits = 0
        self.edges = List[Int]()
        self.actions = List[Int]()
        self.max_rotations = 0
        self.max_depth = 128
        self.emb = 512
        self.common = List[Int]()
        self.common.append(256)
        self.seed = 1
        self.start_iteration = 0
        self.start_difficulty = 1
        self.cores = 4
        self.chunk_rows = CHUNK_ROWS
        self.vw = VW
        self.budget_s = 0.0
        self.iters = 0
        self.init = String()
        self.init_final_difficulty = -1
        self.config_json = String()

    def obs_size(self) -> Int:
        var cfg = EnvConfig()
        cfg.kind = self.kind
        cfg.n = self.num_qubits
        cfg.max_rotations = self.max_rotations
        return obs_size_of(cfg)

    def env_config(self) -> EnvConfig:
        """An `EnvConfig` with this run's kind, size, gateset and depth (the other constants at their defaults)."""
        var cfg = EnvConfig()
        cfg.kind = self.kind
        cfg.n = self.num_qubits
        cfg.actions = self.actions.copy()
        cfg.max_depth = self.max_depth
        cfg.max_rotations = self.max_rotations if self.kind == MODEL_KIND_PAULI else 0
        return cfg^


def new_run_info(id: String, kind: Int, topology: String) raises -> RunInfo:
    """A `RunInfo` for `kind` on the topology spec `topology` (QM-E641 on a bad spec): edges, gateset and the
    reference defaults (max_depth 128, Pauli max_rotations 5, emb 512, common [256]); callers adjust the rest."""
    var t = parse_topology(topology)
    var info = RunInfo()
    info.id = id
    info.kind = kind
    info.topology = topology
    info.num_qubits = t.n
    info.edges = t.edges.copy()
    info.actions = build_gateset(kind, t.n, t.edges)
    info.max_rotations = 5 if kind == MODEL_KIND_PAULI else 0
    return info^


def net_spec_for(info: RunInfo) -> NetSpec:
    """The `NetSpec` of this run: observation and action sizes from the gateset, the run's `emb` / `common`, one
    Linear per head, and the symmetry tables of `symmetry.twists` for plain kinds (Pauli networks train without
    policy-side symmetries: theirs is hidden inside the environment)."""
    var spec = NetSpec()
    spec.obs_size = info.obs_size()
    spec.num_actions = len(info.actions) // 3
    spec.emb = info.emb
    spec.common = info.common.copy()
    if info.kind != MODEL_KIND_PAULI:
        var tw = twists(info.kind, info.num_qubits, info.actions)
        spec.obs_perms = tw.obs_perms.copy()
        spec.act_perms = tw.act_perms.copy()
    return spec^


def run_json_text(info: RunInfo) -> String:
    """Compact JSON of `run.json`."""
    var cfg = info.config_json
    if cfg.byte_length() == 0:
        cfg = String("{}")
    var s = String("{")
    s += '"format":"' + RUN_FORMAT + '","id":' + json_quote(info.id)
    s += ',"engine":' + json_quote(ENGINE)
    s += ',"kind":' + json_quote(model_kind_name(info.kind))
    s += ',"topology":' + json_quote(info.topology)
    s += ',"num_qubits":' + String(info.num_qubits)
    s += ',"edges":' + pairs_json(info.edges)
    s += ',"actions":' + actions_json(info.actions)
    s += ',"max_rotations":' + String(info.max_rotations)
    s += ',"max_depth":' + String(info.max_depth)
    s += ',"arch":{"emb":' + String(info.emb) + ',"common":' + ints_json(info.common) + "}"
    s += ',"seed":' + String(info.seed)
    s += ',"start_iteration":' + String(info.start_iteration)
    s += ',"start_difficulty":' + String(info.start_difficulty)
    s += ',"cores":' + String(info.cores)
    s += ',"chunk_rows":' + String(info.chunk_rows)
    s += ',"vw":' + String(info.vw)
    s += ',"budget_s":' + json_number(info.budget_s)
    s += ',"iters":' + String(info.iters)
    s += ',"init":' + json_quote(info.init)
    s += ',"init_final_difficulty":' + String(info.init_final_difficulty)
    s += ',"config":' + cfg
    s += "}"
    return s^


def config_hash_of(info: RunInfo) -> String:
    """16 hex digits identifying what determines a run's results (SHA-256 prefix of `run_json_text` with `cores`,
    `budget_s` and `iters` blanked: the thread count never changes results and a resumed run may be given a larger
    budget). `--resume` refuses a `last/` whose recorded hash differs from the hash of the configuration it is asked
    to continue with; the driver's own constants belong in `config_json` so that they are covered."""
    var key = info.copy()
    key.cores = 0
    key.budget_s = 0.0
    key.iters = 0
    var h = sha256_hex(List[UInt8](run_json_text(key).as_bytes()))
    return String(unsafe_from_utf8=h.as_bytes()[0:16])


def write_run_json(run_dir: String, info: RunInfo) raises:
    """Creates `run_dir` if needed and writes `run.json` atomically (indent 1)."""
    makedirs(_strip_slashes(run_dir), exist_ok=True)
    write_text_atomic(_strip_slashes(run_dir) + "/" + RUN_FILE, pretty_json(run_json_text(info), 1) + "\n")


def read_run_json(run_dir: String) raises -> RunInfo:
    """Parses `run.json` (QM-E647 when missing or malformed). Unknown members are ignored."""
    var path = _strip_slashes(run_dir) + "/" + RUN_FILE
    if not isfile(path):
        raise _e647("'" + run_dir + "' has no run.json")
    var info = RunInfo()
    try:
        var j = parse_json(_read_text(path))
        info.id = j.get_string("id", "")
        info.kind = model_kind_code(j.get_string("kind", ""))
        if info.kind < 0 or info.kind > MODEL_KIND_PAULI:
            raise Error("unknown kind '" + j.get_string("kind", "") + "'")
        info.topology = j.get_string("topology", "")
        info.num_qubits = j.get_int("num_qubits", 0)
        info.edges = List[Int]()
        if j.has("edges"):
            var ej = j.get("edges")
            for i in range(len(ej)):
                info.edges.append(ej.at(i).at(0).as_int())
                info.edges.append(ej.at(i).at(1).as_int())
        info.actions = List[Int]()
        if j.has("actions"):
            var aj = j.get("actions")
            for i in range(len(aj)):
                var g = aj.at(i)
                var code = gate_code(g.at(0).as_string())
                if code < 0:
                    raise Error("unknown gate '" + g.at(0).as_string() + "'")
                info.actions.append(code)
                info.actions.append(g.at(1).as_int())
                info.actions.append(g.at(2).as_int() if len(g) > 2 else -1)
        info.max_rotations = j.get_int("max_rotations", 0)
        info.max_depth = j.get_int("max_depth", 128)
        if j.has("arch"):
            info.emb = j.get("arch").get_int("emb", 512)
            info.common = List[Int]()
            if j.get("arch").has("common"):
                var cj = j.get("arch").get("common")
                for i in range(len(cj)):
                    info.common.append(cj.at(i).as_int())
        info.seed = j.get_int("seed", 1)
        info.start_iteration = j.get_int("start_iteration", 0)
        info.start_difficulty = j.get_int("start_difficulty", 1)
        info.cores = j.get_int("cores", 1)
        info.chunk_rows = j.get_int("chunk_rows", CHUNK_ROWS)
        info.vw = j.get_int("vw", VW)
        info.budget_s = j.get_float("budget_s", 0.0)
        info.iters = j.get_int("iters", 0)
        info.init = j.get_string("init", "")
        info.init_final_difficulty = j.get_int("init_final_difficulty", -1)
        if j.has("config") and not j.get("config").is_null():
            info.config_json = j.get("config").to_json()
    except e:
        raise _e647("run.json of '" + run_dir + "' is malformed (" + String(e) + ")")
    return info^


# ------------------------------------------------------------------------------------------------ train state


struct TrainState(Copyable):
    """The resumable counters of a run, stored in `last/` (manifest member `train_state`).

    `it` is the number of iterations this run has completed (a resumed run continues from it; the global iteration
    key of the seed tree is `start_iteration + it`). `best_*` is the lexicographic (difficulty, success, reward) of the
    best checkpoint (`have_best` false before the first). `adam_t` is the optimizer step counter."""

    var it: Int
    var start_iteration: Int
    var difficulty: Int
    var have_best: Bool
    var best_difficulty: Int
    var best_success: Float32
    var best_reward: Float32
    var elapsed_s: Float64
    var seed: Int
    var config_hash: String
    var adam_t: Int

    def __init__(out self):
        self.it = 0
        self.start_iteration = 0
        self.difficulty = 1
        self.have_best = False
        self.best_difficulty = 0
        self.best_success = 0.0
        self.best_reward = 0.0
        self.elapsed_s = 0.0
        self.seed = 1
        self.config_hash = String()
        self.adam_t = 0


def train_state_json(st: TrainState) -> String:
    """Compact JSON object of a `TrainState`; floats are f32 values printed as shortest round-trip doubles, so
    reading them back and casting to f32 is exact."""
    var s = String("{")
    s += '"it":' + String(st.it)
    s += ',"start_iteration":' + String(st.start_iteration)
    s += ',"difficulty":' + String(st.difficulty)
    s += ',"have_best":' + ("true" if st.have_best else "false")
    s += ',"best_difficulty":' + String(st.best_difficulty)
    s += ',"best_success":' + json_number(Float64(st.best_success))
    s += ',"best_reward":' + json_number(Float64(st.best_reward))
    s += ',"elapsed_s":' + json_number(st.elapsed_s)
    s += ',"seed":' + String(st.seed)
    s += ',"config_hash":' + json_quote(st.config_hash)
    s += ',"adam_t":' + String(st.adam_t)
    s += "}"
    return s^


def _parse_train_state(j: JsonValue) raises -> TrainState:
    var st = TrainState()
    st.it = j.get_int("it", 0)
    st.start_iteration = j.get_int("start_iteration", 0)
    st.difficulty = j.get_int("difficulty", 1)
    st.have_best = j.get_bool("have_best", False)
    st.best_difficulty = j.get_int("best_difficulty", 0)
    st.best_success = Float32(j.get_float("best_success", 0.0))
    st.best_reward = Float32(j.get_float("best_reward", 0.0))
    st.elapsed_s = j.get_float("elapsed_s", 0.0)
    st.seed = j.get_int("seed", 1)
    st.config_hash = j.get_string("config_hash", "")
    st.adam_t = j.get_int("adam_t", 0)
    return st^


def read_train_state(m: ModelFile) raises -> TrainState:
    """The `train_state` of a checkpoint container (QM-E647 if it has none, e.g. a shipped model)."""
    var j = parse_json(m.manifest)
    if not j.has("train_state"):
        raise _e647("'" + m.dir + "' carries no train_state (not a run checkpoint)")
    return _parse_train_state(j.get("train_state"))


# ------------------------------------------------------------------------------------------------ checkpoints


def checkpoint_head(info: RunInfo, net: BasicNet) -> String:
    """Manifest members before `tensors` of a run checkpoint: the same description a shipped model carries
    (kind, qubits, edges, gateset, observation layout, architecture, the *net's own* symmetries, which for Pauli
    networks are none because the symmetry is hidden inside the environment), minus training metadata."""
    var s = String()
    s += '"format":"qumo.rlsynth.v0","id":' + json_quote(info.id)
    s += ',"kind":' + json_quote(model_kind_name(info.kind))
    s += ',"num_qubits":' + String(info.num_qubits)
    s += ',"edges":' + pairs_json(info.edges)
    s += ',"actions":' + actions_json(info.actions)
    s += ',"obs":' + obs_json(info.kind, info.num_qubits, info.max_rotations, net.spec.obs_size)
    s += ',"max_depth":' + String(info.max_depth)
    s += ',"arch":' + arch_json(net.spec.emb, net.spec.common)
    s += ',"symmetries":' + symmetries_json(net.spec.obs_perms, net.spec.act_perms)
    s += ',"license":"Apache-2.0"'
    return s^


def net_tensors(net: BasicNet) -> List[TensorOut]:
    """The network's parameters as f32 container tensors in registration order (embeddings.weight [OBS, E] ...);
    biases are 1-D `[n]` like the shipped models."""
    var out = List[TensorOut]()
    for i in range(net.ps.count()):
        var name = net.ps.names[i]
        var rows = net.ps.rows_of(i)
        var cols = net.ps.cols_of(i)
        var off = net.ps.offsets[i]
        var shape = List[Int]()
        if name.endswith(".bias"):
            shape.append(rows * cols)
        else:
            shape.append(rows)
            shape.append(cols)
        var data = List[Float32](capacity=rows * cols)
        for k in range(rows * cols):
            data.append(net.ps.data[off + k])
        out.append(TensorOut(name^, shape^, data^, String("f32")))
    return out^


def write_checkpoint(
    dir: String,
    net: BasicNet,
    manifest_head: String,
    manifest_tail: String,
    extra: List[TensorOut] = List[TensorOut](),
) raises -> ModelDirInfo:
    """f32 model-container checkpoint of `net` (plus optional `extra` f32 tensors), written atomically through
    `container.write_model_dir` (staged, read back, swapped in)."""
    var tensors = net_tensors(net)
    for i in range(len(extra)):
        tensors.append(extra[i].copy())
    return write_model_dir(dir, manifest_head, tensors, "f32", List[String](), manifest_tail)


def read_checkpoint(dir: String) raises -> ModelFile:
    """Loads a checkpoint or any model container (a leftover `<dir>.old` / `<dir>.tmp` is repaired first); callers
    build the network with `BasicNet.from_model`."""
    recover_dir(dir)
    return load_model_file(dir)


def has_checkpoint(dir: String) raises -> Bool:
    recover_dir(dir)
    return isfile(_strip_slashes(dir) + "/manifest.json")


def write_best(run_dir: String, net: BasicNet, info: RunInfo, st: TrainState) raises -> ModelDirInfo:
    """Writes `RUN/best/`."""
    var tail = '"train_state":' + train_state_json(st)
    return write_checkpoint(
        _strip_slashes(run_dir) + "/" + BEST_DIR, net, checkpoint_head(info, net), tail
    )


def write_last(
    run_dir: String, net: BasicNet, opt: Adam, info: RunInfo, st: TrainState
) raises -> ModelDirInfo:
    """Writes `RUN/last/` with the Adam moments as the extra tensors `adam.m`, `adam.v` and the counters as
    `train_state` (`st.adam_t` is overwritten with the optimizer's own step count), then the `state.json` mirror."""
    var s = st.copy()
    s.adam_t = opt.steps_taken()
    var extra = List[TensorOut]()
    var shape = List[Int]()
    shape.append(len(opt.m))
    extra.append(TensorOut(String(ADAM_M), shape.copy(), opt.m.copy(), String("f32")))
    extra.append(TensorOut(String(ADAM_V), shape^, opt.v.copy(), String("f32")))
    var tail = '"train_state":' + train_state_json(s)
    var rd = _strip_slashes(run_dir)
    var res = write_checkpoint(rd + "/" + LAST_DIR, net, checkpoint_head(info, net), tail, extra)
    write_text_atomic(rd + "/" + STATE_FILE, pretty_json(train_state_json(s), 1) + "\n")
    return res^


def load_adam_state(mut opt: Adam, m: ModelFile) raises:
    """Restores the step counter and both moments from a `last/` checkpoint into `opt`. Register `set_frozen` and
    `mark_sparse_rows` on `opt` first (the touched-row flags are recomputed from the moments). QM-E647 on a missing
    state or a size mismatch."""
    if not m.has_tensor(ADAM_M) or not m.has_tensor(ADAM_V):
        raise _e647("'" + m.dir + "' has no Adam state (is it a last/ checkpoint?)")
    var st = read_train_state(m)
    try:
        opt.load_state(st.adam_t, m.tensor(ADAM_M), m.tensor(ADAM_V))
    except e:
        raise _e647("Adam state of '" + m.dir + "' does not fit the optimizer (" + String(e) + ")")


# ------------------------------------------------------------------------------------------------ summary and log


struct RunSummary(Copyable):
    """Content of `summary.json`. `iterations` counts the iterations of this run (resumes included);
    `finished` is true only after the final checkpoint was written."""

    var finished: Bool
    var iterations: Int
    var start_iteration: Int
    var final_difficulty: Int
    var best_difficulty: Int
    var best_success: Float64
    var best_reward: Float64
    var wall_s: Float64
    var budget_s: Float64
    var cores: Int
    var seed: Int
    var topology: String
    var init: String

    def __init__(out self):
        self.finished = False
        self.iterations = 0
        self.start_iteration = 0
        self.final_difficulty = 1
        self.best_difficulty = 0
        self.best_success = 0.0
        self.best_reward = 0.0
        self.wall_s = 0.0
        self.budget_s = 0.0
        self.cores = 1
        self.seed = 1
        self.topology = String()
        self.init = String()


def summary_json(s: RunSummary) -> String:
    var t = String("{")
    t += '"engine":' + json_quote(ENGINE)
    t += ',"finished":' + ("true" if s.finished else "false")
    t += ',"iterations":' + String(s.iterations)
    t += ',"start_iteration":' + String(s.start_iteration)
    t += ',"final_difficulty":' + String(s.final_difficulty)
    t += ',"best_difficulty":' + String(s.best_difficulty)
    t += ',"best_success":' + json_number(s.best_success)
    t += ',"best_reward":' + json_number(s.best_reward)
    t += ',"wall_s":' + json_number(s.wall_s)
    t += ',"budget_s":' + json_number(s.budget_s)
    t += ',"cores":' + String(s.cores)
    t += ',"seed":' + String(s.seed)
    t += ',"topology":' + json_quote(s.topology)
    t += ',"init":' + json_quote(s.init)
    t += "}"
    return t^


def write_summary(run_dir: String, s: RunSummary) raises:
    makedirs(_strip_slashes(run_dir), exist_ok=True)
    write_text_atomic(_strip_slashes(run_dir) + "/" + SUMMARY_FILE, pretty_json(summary_json(s), 1) + "\n")


def has_summary(run_dir: String) -> Bool:
    return isfile(_strip_slashes(run_dir) + "/" + SUMMARY_FILE)


def read_summary(run_dir: String) raises -> RunSummary:
    """Parses `summary.json` (QM-E647 when missing or malformed). Old-format summaries (`topology`, `iterations`,
    `final_difficulty`, `wall_s`, ... without `finished`) are accepted; `finished` then defaults to true."""
    var path = _strip_slashes(run_dir) + "/" + SUMMARY_FILE
    if not isfile(path):
        raise _e647("'" + run_dir + "' has no summary.json")
    var s = RunSummary()
    try:
        var j = parse_json(_read_text(path))
        s.finished = j.get_bool("finished", True)
        s.iterations = j.get_int("iterations", 0)
        s.start_iteration = j.get_int("start_iteration", 0)
        s.final_difficulty = j.get_int("final_difficulty", 1)
        s.best_difficulty = j.get_int("best_difficulty", 0)
        s.best_success = j.get_float("best_success", 0.0)
        s.best_reward = j.get_float("best_reward", 0.0)
        s.wall_s = j.get_float("wall_s", 0.0)
        s.budget_s = j.get_float("budget_s", 0.0)
        s.cores = j.get_int("cores", 1)
        s.seed = j.get_int("seed", 1)
        s.topology = j.get_string("topology", "")
        s.init = j.get_string("init", "")
    except e:
        raise _e647("summary.json of '" + run_dir + "' is malformed (" + String(e) + ")")
    return s^


def append_log_line(run_dir: String, line: String) raises:
    """Appends one line (a JSON object, no newline inside) to `RUN/log.jsonl`."""
    makedirs(_strip_slashes(run_dir), exist_ok=True)
    with open(_strip_slashes(run_dir) + "/" + LOG_FILE, "a") as f:
        f.write(line)
        f.write("\n")


def truncate_log(run_dir: String, keep_lines: Int) raises:
    """Keeps only the first `keep_lines` lines of `RUN/log.jsonl` (a resumed run drops the lines of iterations that
    were logged after its last checkpoint). A missing file is fine."""
    var path = _strip_slashes(run_dir) + "/" + LOG_FILE
    if not isfile(path):
        return
    var text = _read_text(path)
    var bs = text.as_bytes()
    var seen = 0
    var cut = len(bs)
    for i in range(len(bs)):
        if Int(bs[i]) == 0x0A:
            seen += 1
            if seen == keep_lines:
                cut = i + 1
                break
    if keep_lines <= 0:
        cut = 0
    if cut >= len(bs):
        return
    write_text_atomic(path, String(unsafe_from_utf8=bs[0:cut]))


def log_line_count(run_dir: String) raises -> Int:
    var path = _strip_slashes(run_dir) + "/" + LOG_FILE
    if not isfile(path):
        return 0
    var bs = _read_text(path).as_bytes()
    var n = 0
    for i in range(len(bs)):
        if Int(bs[i]) == 0x0A:
            n += 1
    return n


# ------------------------------------------------------------------------------------------------ warm start


struct WarmStart(Copyable):
    """Where a run starts from (plan 7.5). `model_dir` holds the weights to load; `start_difficulty` and
    `start_iteration` are the values the driver should use unless the user overrode them."""

    var model_dir: String
    var start_difficulty: Int
    var start_iteration: Int
    var is_run: Bool

    def __init__(out self):
        self.model_dir = String()
        self.start_difficulty = 1
        self.start_iteration = 0
        self.is_run = False


def _run_iterations(run_dir: String) raises -> Int:
    """Iterations a run directory completed: from summary.json, else from the train_state of last/."""
    if has_summary(run_dir):
        return read_summary(run_dir).iterations
    if has_checkpoint(_strip_slashes(run_dir) + "/" + LAST_DIR):
        return read_train_state(load_model_file(_strip_slashes(run_dir) + "/" + LAST_DIR)).it
    return 0


def resolve_init(
    init: String, out_dir: String, start_diff: Int = -1, start_iteration: Int = -1
) raises -> WarmStart:
    """Resolves `--init`. `start_diff` / `start_iteration` are the explicit `--start-diff` / `--start-iteration`
    values, -1 when not given.

    * `auto`: `out_dir` must be named `X__cont`; the source is the sibling run `X` (its `best/`). The run starts at
      the sibling's `summary.final_difficulty` and `start_iteration = sibling.start_iteration + sibling.iterations`;
      a missing or unreadable sibling summary is QM-E647 (the old trainer silently fell back to difficulty 1).
    * a run directory (it has `run.json`): its `best/` is used; difficulty 1 unless `--start-diff`;
      `start_iteration` is the run's own offset plus its iterations.
    * any other directory (a shipped f16 model, a bare container): difficulty 1 and iteration 0 unless given."""
    var ws = WarmStart()
    if init == "auto":
        var base = _basename(out_dir)
        if not base.endswith("__cont") or base.byte_length() <= 6:
            raise _e647("--init auto needs an output directory named X__cont, got '" + out_dir + "'")
        var sibling = _parent(out_dir) + "/" + String(unsafe_from_utf8=base.as_bytes()[0 : base.byte_length() - 6])
        if not has_summary(sibling):
            raise _e647("--init auto: the sibling run '" + sibling + "' has no readable summary.json")
        var summ = read_summary(sibling)
        var off = 0
        if isfile(sibling + "/" + RUN_FILE):
            off = read_run_json(sibling).start_iteration
        ws.model_dir = sibling + "/" + BEST_DIR
        ws.is_run = True
        ws.start_difficulty = summ.final_difficulty
        ws.start_iteration = off + summ.iterations
    elif isfile(_strip_slashes(init) + "/" + RUN_FILE):
        var rd = _strip_slashes(init)
        ws.model_dir = rd + "/" + BEST_DIR
        ws.is_run = True
        ws.start_difficulty = 1
        ws.start_iteration = read_run_json(rd).start_iteration + _run_iterations(rd)
    else:
        if not isdir(_strip_slashes(init)) and not isdir(_strip_slashes(init) + ".old"):
            raise _e647("--init '" + init + "' is not a directory")
        ws.model_dir = _strip_slashes(init)
        ws.is_run = False
        ws.start_difficulty = 1
        ws.start_iteration = 0
    if start_diff >= 0:
        ws.start_difficulty = start_diff
    if start_iteration >= 0:
        ws.start_iteration = start_iteration
    return ws^


# ------------------------------------------------------------------------------------------------ tensor-file importer


@fieldwise_init
struct SafeTensor(Copyable):
    """One tensor read from a tensor file (decoded to f32, row-major, in the file's own layout)."""

    var name: String
    var shape: List[Int]
    var data: List[Float32]


def _u64_le(b: Span[UInt8, _], off: Int) -> UInt64:
    var v: UInt64 = 0
    for k in range(8):
        v |= UInt64(b[off + k]) << UInt64(8 * k)
    return v


def _f32_from_bits(u: UInt32) -> Float32:
    return bitcast[DType.float32](u)


def _f16_from_bits(h: UInt16) -> Float32:
    return Float32(bitcast[DType.float16](h))


def safetensors_parse(bytes: List[UInt8]) raises -> List[SafeTensor]:
    """Parses the bytes of a tensor file (the reference trainer's weight format): 8-byte little-endian header length, JSON header (tensor name ->
    dtype, shape, data_offsets; an optional `__metadata__` member is skipped), then the raw data. F32, F16, BF16
    and F64 are decoded to f32; the tensors come back in order of increasing data offset. QM-E647 on a truncated
    file, a malformed header, an offset outside the data, a size that does not match the shape or any other dtype."""
    var n = len(bytes)
    if n < 8:
        raise _e647("tensor file is shorter than its 8-byte header length")
    var hlen_u = _u64_le(Span(bytes), 0)
    if hlen_u > UInt64(n - 8):
        raise _e647("tensor file header length " + String(hlen_u) + " exceeds the file size " + String(n))
    var hlen = Int(hlen_u)
    var header = String(unsafe_from_utf8=Span(bytes)[8 : 8 + hlen])
    var data_start = 8 + hlen
    var data_len = n - data_start
    var names = List[String]()
    var starts = List[Int]()
    var out = List[SafeTensor]()
    try:
        var j = parse_json(header)
        var keys = j.keys()
        var raw = List[SafeTensor]()
        for ki in range(len(keys)):
            var key = keys[ki]
            if key == "__metadata__":
                continue
            var t = j.get(key)
            var dtype = t.get("dtype").as_string()
            var shape = List[Int]()
            var sh = t.get("shape")
            var cnt = 1
            for i in range(len(sh)):
                var d = sh.at(i).as_int()
                if d < 0:
                    raise Error("negative dimension in '" + key + "'")
                shape.append(d)
                cnt *= d
            var offs = t.get("data_offsets")
            var a = offs.at(0).as_int()
            var b = offs.at(1).as_int()
            var esz: Int
            if dtype == "F32":
                esz = 4
            elif dtype == "F16" or dtype == "BF16":
                esz = 2
            elif dtype == "F64":
                esz = 8
            else:
                raise Error("tensor '" + key + "' has unsupported dtype " + dtype)
            if a < 0 or b < a or b > data_len:
                raise Error("tensor '" + key + "' lies outside the data section")
            if b - a != cnt * esz:
                raise Error("tensor '" + key + "' has " + String(b - a) + " bytes for shape of " + String(cnt) + " elements")
            var vals = List[Float32](capacity=cnt)
            var base = data_start + a
            var sp = Span(bytes)
            for e in range(cnt):
                if esz == 4:
                    var u = UInt32(sp[base + 4 * e]) | (UInt32(sp[base + 4 * e + 1]) << 8) | (UInt32(sp[base + 4 * e + 2]) << 16) | (UInt32(sp[base + 4 * e + 3]) << 24)
                    vals.append(_f32_from_bits(u))
                elif esz == 8:
                    var u64 = _u64_le(sp, base + 8 * e)
                    vals.append(Float32(bitcast[DType.float64](u64)))
                else:
                    var h = UInt16(sp[base + 2 * e]) | (UInt16(sp[base + 2 * e + 1]) << 8)
                    if dtype == "F16":
                        vals.append(_f16_from_bits(h))
                    else:
                        vals.append(_f32_from_bits(UInt32(h) << 16))
            names.append(key)
            starts.append(a)
            raw.append(SafeTensor(key, shape^, vals^))
        # storage order: increasing data offset (stable)
        var used = List[Bool](length=len(raw), fill=False)
        for _ in range(len(raw)):
            var best = -1
            for i in range(len(raw)):
                if not used[i] and (best < 0 or starts[i] < starts[best]):
                    best = i
            used[best] = True
            out.append(raw[best].copy())
    except e:
        raise _e647("tensor file header is malformed (" + String(e) + ")")
    return out^


def safetensors_import(path: String) raises -> List[SafeTensor]:
    """Read-only importer for the weight files of old runs (f32, f16, bf16, f64 decoded to f32); see `safetensors_parse`."""
    if not isfile(path):
        raise _e647("tensor file '" + path + "' does not exist")
    var bytes: List[UInt8]
    with open(path, "r") as f:
        bytes = f.read_bytes()
    return safetensors_parse(bytes)


def _find_tensor(st: List[SafeTensor], name: String) -> Int:
    for i in range(len(st)):
        if st[i].name == name:
            return i
    return -1


def basic_mlp_tensors(st: List[SafeTensor]) raises -> List[TensorOut]:
    """Orders and transposes a reference state dict (`embeddings.weight` is [E, OBS]) into container tensors:
    `embeddings.weight` transposed to [OBS, E], `embeddings.bias`, then `common.k`, `action.k`, `value.k` (weight
    [out, in] and bias [out]) with consecutive layer numbers in increasing index order. QM-E647 when a part is
    missing or not 2-D / 1-D as expected."""
    var out = List[TensorOut]()
    var ei = _find_tensor(st, "embeddings.weight")
    var bi = _find_tensor(st, "embeddings.bias")
    if ei < 0 or bi < 0:
        raise _e647("state dict has no embeddings.weight / embeddings.bias")
    if len(st[ei].shape) != 2:
        raise _e647("embeddings.weight is not 2-D")
    var E = st[ei].shape[0]
    var OBS = st[ei].shape[1]
    var wt = List[Float32](length=E * OBS, fill=0.0)
    for r in range(E):
        for c in range(OBS):
            wt[c * E + r] = st[ei].data[r * OBS + c]
    var s1 = List[Int]()
    s1.append(OBS)
    s1.append(E)
    out.append(TensorOut(String("embeddings.weight"), s1^, wt^, String("f16")))
    var b1 = List[Int]()
    b1.append(len(st[bi].data))
    out.append(TensorOut(String("embeddings.bias"), b1^, st[bi].data.copy(), String("f32")))
    var prefixes: List[String] = ["common", "action", "value"]
    for pi in range(3):
        var prefix = prefixes[pi]
        var idxs = List[Int]()
        for i in range(len(st)):
            var nm = st[i].name
            if nm.startswith(prefix + ".") and nm.endswith(".weight"):
                var mid = String(unsafe_from_utf8=nm.as_bytes()[prefix.byte_length() + 1 : nm.byte_length() - 7])
                idxs.append(Int(mid))
        sort(idxs)
        for k in range(len(idxs)):
            var wn = prefix + "." + String(idxs[k]) + ".weight"
            var bn = prefix + "." + String(idxs[k]) + ".bias"
            var wi = _find_tensor(st, wn)
            var bj = _find_tensor(st, bn)
            if wi < 0 or bj < 0:
                raise _e647("state dict is missing '" + wn + "' or '" + bn + "'")
            if len(st[wi].shape) != 2:
                raise _e647("'" + wn + "' is not 2-D")
            out.append(
                TensorOut(
                    prefix + "." + String(k) + ".weight", st[wi].shape.copy(), st[wi].data.copy(), String("f16")
                )
            )
            var bs = List[Int]()
            bs.append(len(st[bj].data))
            out.append(TensorOut(prefix + "." + String(k) + ".bias", bs^, st[bj].data.copy(), String("f32")))
    return out^
