"""Synthesis export: a run directory to a `qumo.rlsynth.v0` model (f16 weights, symmetries, test vectors, rollouts,
INDEX), the independent pre-write checks, `check_pauli_ensemble` and the random-weight fixture generator.

MOJO_PORT.md 9.2 and 4.6, component C11a. `export_synth_run` reads `RUN/best` (or `last`), the gateset and topology from
`RUN/run.json` and the totals from `RUN/summary.json`, and produces `<root>/<id>/{manifest.json, weights.bin}`:

1. The symmetry ensemble is recomputed from the gateset (`symmetry.twists` for plain kinds, `symmetry.pauli_twists` for
   Pauli networks, whose training symmetry is hidden inside the environment). For plain kinds it must equal the
   symmetries stored in the checkpoint (the ones used in training).
2. The model is staged in `<root>/<id>.tmp` (f16 for matrices, f32 for biases, as the container rules say). Nothing in
   `<root>` is touched until every check passed, and `INDEX.json` is rebuilt last.
3. Independent checks (QM-E644 on any failure): on observations drawn from reachable start states, the training-layout
   `BasicNet.forward_one` (built from the pre-write f32 tensors rounded to the stored precision), the inference
   `MLPPolicy` (read from the written directory) and a naive Float64 forward (plain loops over the decoded tensors, no
   shared kernel) must agree to `1e-5 * max(1, |x|)` in logits and value, for every twist and for the ensemble average;
   for Pauli models `check_pauli_ensemble` runs on the exported symmetries.
4. Test vectors (6) and rollouts (2) are computed from the *stored* weights (`load_model_file` + `MLPPolicy` +
   `PolicyEnv`), the manifest is rewritten with them as its tail, and the finished directory is replayed
   (`verify_exported_model`) before it is swapped into place.

All random draws come from `Rng(seed_for(EXPORT_SEED, TAG_EXPORT_VEC, stream))`, so an export is a pure function of the
checkpoint. `generate_policy_fixtures` writes the tiny random-weight models of `tests/fixtures/ai/policy` (the committed
fixtures were made by the legacy generator and stay frozen; this one is the native replacement).
"""

from std.collections import List
from std.os import makedirs
from std.os.path import exists, isfile
from qrl.json import JsonValue, parse_json, json_quote, json_number
from qrl.model import (
    ModelFile,
    MLPPolicy,
    load_model_file,
    f16_to_f32,
    model_kind_name,
    MODEL_KIND_CLIFFORD,
    MODEL_KIND_LINEAR,
    MODEL_KIND_PAULI,
    MODEL_KIND_PERM,
)
from qrl.env_core import PolicyEnv, greedy_trace
from qrl.rng import Rng
from qrl.checkpoint import (
    RunInfo,
    RunSummary,
    TrainState,
    PRODUCER,
    ENGINE,
    read_run_json,
    read_summary,
    has_summary,
    read_checkpoint,
    read_train_state,
    ints_json,
    perms_json,
    pairs_json,
    actions_json,
    obs_json,
    arch_json,
    symmetries_json,
    _basename,
    _parent,
    _read_text,
)
from qrl.container import (
    TensorOut,
    ModelDirInfo,
    write_model_dir,
    atomic_replace_dir,
    recover_dir,
    remove_tree,
    rebuild_index,
)
from qrl.f16 import f32_to_f16_bits, max_abs_f32
from qrl.gateset import EnvConfig, build_gateset, obs_size_of
from qrl.nn import BasicNet, ActorScratch
from qrl.rlrng import TAG_EXPORT_VEC, seed_for
from qrl.symmetry import Twists, pauli_twists, twists
from qrl.synth_env import TrainEnv
from qrl.topology import parse_topology

comptime EXPORT_SEED: UInt64 = 20261014  # the seed of the legacy exporter's vector stream
comptime NUM_VECTORS = 6
comptime NUM_ROLLOUTS = 2
comptime CHECK_TOL = 1e-5
comptime PAULI_TOL = 1e-4  # both views sum the embedding rows in different orders: f32 rounding only
comptime F16_WARN = 60000.0


def _e644(what: String) -> Error:
    return Error(
        "QM-E644: " + what + ". Hint: nothing was written to the model root; check the run's checkpoint and gateset."
    )


def _e640(what: String) -> Error:
    return Error("QM-E640: " + what + ". Hint: export_synth RUN [--which best|last] --id ID --out ROOT [--dtype f16|f32] [--eval-json F] [--default].")


# ------------------------------------------------------------------------------------------------ small helpers


def _as_u64(text: String) raises -> UInt64:
    var v: UInt64 = 0
    var bs = text.as_bytes()
    if len(bs) == 0:
        raise Error("empty integer")
    for i in range(len(bs)):
        var c = Int(bs[i])
        if c < ord("0") or c > ord("9"):
            raise Error("not an unsigned integer: " + text)
        v = v * 10 + UInt64(c - ord("0"))
    return v


def _u64_list_json(v: List[UInt64]) -> String:
    var s = String("[")
    for i in range(len(v)):
        if i > 0:
            s += ","
        s += String(v[i])
    s += "]"
    return s^


def _f32_list_json(v: List[Float32]) -> String:
    var s = String("[")
    for i in range(len(v)):
        if i > 0:
            s += ","
        s += json_number(Float64(v[i]))
    s += "]"
    return s^


def _round_value(x: Float32, dtype: String, name: String) -> Float32:
    """The value a tensor element has after storage: f16 round trip for matrices of an f16 export, else unchanged."""
    if dtype == "f16" and not name.endswith(".bias"):
        return f16_to_f32(f32_to_f16_bits(x))
    return x


def _shape_cols(shape: List[Int]) -> Int:
    return shape[len(shape) - 1]


def _same_lists(a: List[List[Int]], b: List[List[Int]]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if len(a[i]) != len(b[i]):
            return False
        for j in range(len(a[i])):
            if a[i][j] != b[i][j]:
                return False
    return True


def _view_from_tensors(
    tensors: List[TensorOut],
    dtype: String,
    obs_perms: List[List[Int]],
    act_perms: List[List[Int]],
    obs_size: Int,
    actions: List[Int],
) -> ModelFile:
    """An in-memory `ModelFile` over the pre-write tensors, rounded to the stored precision (f16 matrices when
    `dtype` is f16), carrying the export symmetries: the training-layout side of the independent check."""
    var m = ModelFile()
    m.dir = String("<pre-write tensors>")
    m.obs_size = obs_size
    m.actions = actions.copy()
    m.obs_perms = obs_perms.copy()
    m.act_perms = act_perms.copy()
    for i in range(len(tensors)):
        var name = tensors[i].name
        var f32_forced = tensors[i].dtype == "f32"
        var data = List[Float32](capacity=len(tensors[i].data))
        for k in range(len(tensors[i].data)):
            if f32_forced:
                data.append(tensors[i].data[k])
            else:
                data.append(_round_value(tensors[i].data[k], dtype, name))
        m.names.append(name)
        m.shapes.append(tensors[i].shape.copy())
        m.tensors.append(data^)
    return m^


# ------------------------------------------------------------------------------------------------ naive Float64 forward


def _dense64(
    w: List[Float32], b: List[Float32], n_in: Int, n_out: Int, x: List[Float64], relu: Bool
) -> List[Float64]:
    var y = List[Float64](length=n_out, fill=0.0)
    for o in range(n_out):
        var acc = Float64(b[o])
        for i in range(n_in):
            acc += Float64(w[o * n_in + i]) * x[i]
        if relu and acc < 0.0:
            acc = 0.0
        y[o] = acc
    return y^


def _chain64(m: ModelFile, prefix: String, x: List[Float64], final_relu: Bool) raises -> List[Float64]:
    var cur = x.copy()
    var k = 0
    while m.has_tensor(prefix + "." + String(k) + ".weight"):
        var wn = prefix + "." + String(k) + ".weight"
        var sh = m.shape(wn)
        var more = m.has_tensor(prefix + "." + String(k + 1) + ".weight")
        cur = _dense64(m.tensor(wn), m.tensor(prefix + "." + String(k) + ".bias"), sh[1], sh[0], cur, final_relu or more)
        k += 1
    return cur^


def naive_forward_f64(m: ModelFile, idx: List[Int], twist: Int, mut logits: List[Float64]) raises -> Float64:
    """Reference forward over the decoded tensors in plain Float64 loops, sharing no code with `BasicNet` or
    `MLPPolicy`: embedding bag over `idx` (relabelled by `obs_perms[twist]` when `twist >= 0`), ReLU, the common
    layers, the action chain (ReLU between hidden layers) and the value chain; `logits[a] = raw[act_perms[twist][a]]`.
    Returns the value."""
    var esh = m.shape("embeddings.weight")
    var OBS = esh[0]
    var E = esh[1]
    var we = m.tensor("embeddings.weight")
    var h = List[Float64](length=E, fill=0.0)
    var be = m.tensor("embeddings.bias")
    for e in range(E):
        h[e] = Float64(be[e])
    for j in range(len(idx)):
        var i = idx[j]
        if twist >= 0:
            i = m.obs_perms[twist][i]
        if i < 0 or i >= OBS:
            continue
        for e in range(E):
            h[e] += Float64(we[i * E + e])
    for e in range(E):
        if h[e] < 0.0:
            h[e] = 0.0
    var trunk = _chain64(m, "common", h, True)
    var raw = _chain64(m, "action", trunk, False)
    var val = _chain64(m, "value", trunk, False)
    var A = len(raw)
    logits = List[Float64](length=A, fill=0.0)
    for a in range(A):
        logits[a] = raw[m.act_perms[twist][a]] if twist >= 0 else raw[a]
    return val[0] if len(val) > 0 else 0.0


# ------------------------------------------------------------------------------------------------ random starts


def _random_start(m: ModelFile, mut rng: Rng, depth: Int, mut env: PolicyEnv) raises -> List[UInt64]:
    """A reachable start state: a random walk of `depth` actions from the identity (Pauli networks: a random
    Clifford walk plus 1..max_rot+2 random rotations in the input frame, `pauli_start`, then `clean`). For Pauli the
    env is initialised with the state's rotation table."""
    var kind = m.kind_code
    var n = m.num_qubits
    var A = m.num_actions()
    var st = List[UInt64]()
    if kind == MODEL_KIND_PERM:
        for i in range(n):
            st.append(UInt64(i))
        for _ in range(depth):
            st = env.step(st, rng.below(A))
        return st^
    if kind == MODEL_KIND_LINEAR:
        for i in range(n):
            st.append(UInt64(1) << UInt64(i))
        for _ in range(depth):
            st = env.step(st, rng.below(A))
        return st^
    if kind == MODEL_KIND_CLIFFORD:
        for i in range(2 * n):
            st.append(UInt64(1) << UInt64(i))
        for _ in range(depth):
            st = env.step(st, rng.below(A))
        return st^
    # Pauli network
    var R = 1 + rng.below(m.max_rotations + 2)
    var rx = List[UInt64]()
    var rz = List[UInt64]()
    for _ in range(R):
        var x: UInt64 = 0
        var z: UInt64 = 0
        while (x | z) == 0:
            x = UInt64(rng.below(1 << n))
            z = UInt64(rng.below(1 << n))
        rx.append(x)
        rz.append(z)
    var cenv = PolicyEnv(MODEL_KIND_CLIFFORD, n, m)
    var tab = List[UInt64]()
    for i in range(2 * n):
        tab.append(UInt64(1) << UInt64(i))
    for _ in range(depth):
        tab = cenv.step(tab, rng.below(A))
    for i in range(2 * n):
        st.append(tab[i])
    for r in range(R):
        st.append(rx[r])
    for r in range(R):
        st.append(rz[r])
    for _ in range(R):
        st.append(0)
    st.append(UInt64(R))
    for r in range(R):
        st.append(UInt64(r))
    env.init_pauli(R, st)
    var emit = List[Int]()
    env._clean(st, emit, False)
    return st^


def check_observations(m: ModelFile, count: Int, stream: Int) raises -> List[List[Int]]:
    """`count` observations (sparse index lists) of reachable start states of random depth, from the export
    stream `stream`."""
    var rng = Rng(seed_for(EXPORT_SEED, TAG_EXPORT_VEC, stream))
    var out = List[List[Int]]()
    for i in range(count):
        var env = PolicyEnv(m.kind_code, m.num_qubits, m)
        var st = _random_start(m, rng, 2 + 3 * i, env)
        var obs = List[Int]()
        env.observe(st, obs)
        out.append(obs^)
    return out^


# ------------------------------------------------------------------------------------------------ vectors and rollouts


def _vectors_json(m: ModelFile, pol: MLPPolicy, count: Int, depth0: Int, dstep: Int, mut rng: Rng) raises -> String:
    var s = String("[")
    for i in range(count):
        var env = PolicyEnv(m.kind_code, m.num_qubits, m)
        var st = _random_start(m, rng, depth0 + dstep * i, env)
        var obs = List[Int]()
        env.observe(st, obs)
        var lg = List[Float32]()
        var v = pol.forward(obs, lg)
        var le = List[Float32]()
        var ve = pol.forward_ensemble(obs, le)
        if i > 0:
            s += ","
        s += '{"obs":' + ints_json(obs)
        s += ',"logits":' + _f32_list_json(lg)
        s += ',"value":' + json_number(Float64(v))
        s += ',"logits_ensemble":' + _f32_list_json(le)
        s += ',"value_ensemble":' + json_number(Float64(ve)) + "}"
    s += "]"
    return s^


def _rollouts_json(
    m: ModelFile, pol: MLPPolicy, count: Int, depth0: Int, dstep: Int, max_depth: Int, mut rng: Rng
) raises -> String:
    var s = String("[")
    for i in range(count):
        var env = PolicyEnv(m.kind_code, m.num_qubits, m)
        var st = _random_start(m, rng, depth0 + dstep * i, env)
        var ok = False
        var acts = greedy_trace(env, pol, max_depth, st, ok)
        if i > 0:
            s += ","
        s += '{"input":' + _u64_list_json(st)
        s += ',"R":' + String(env.R if m.kind_code == MODEL_KIND_PAULI else 0)
        s += ',"actions":' + ints_json(acts)
        s += ',"solved":' + ("true" if ok else "false") + "}"
    s += "]"
    return s^


def verify_exported_model(dir: String) raises -> Float64:
    """Replays the embedded test vectors (identity and ensemble forward, 1e-5) and rollouts (greedy trace, exact)
    of a written model through `load_model_file` + `MLPPolicy` + `PolicyEnv`, the way `tests/test_policy.mojo`
    does. QM-E644 on any difference. Returns the worst forward difference."""
    var m = load_model_file(dir)
    var pol = MLPPolicy.from_model(m)
    var man = parse_json(m.manifest)
    if not man.has("test_vectors") or not man.has("rollouts"):
        raise _e644("'" + dir + "' has no test_vectors / rollouts to replay")
    var worst = 0.0
    var tv = man.get("test_vectors")
    for t in range(len(tv)):
        var oj = tv.at(t).get("obs")
        var obs = List[Int]()
        for i in range(len(oj)):
            obs.append(oj.at(i).as_int())
        var lg = List[Float32]()
        var v = pol.forward(obs, lg)
        var rl = tv.at(t).get("logits")
        if len(rl) != len(lg):
            raise _e644("test vector " + String(t) + " has the wrong number of logits")
        for a in range(len(lg)):
            worst = max(worst, abs(Float64(lg[a]) - rl.at(a).as_float()))
        worst = max(worst, abs(Float64(v) - tv.at(t).get("value").as_float()))
        var le = List[Float32]()
        var ve = pol.forward_ensemble(obs, le)
        var re = tv.at(t).get("logits_ensemble")
        for a in range(len(le)):
            worst = max(worst, abs(Float64(le[a]) - re.at(a).as_float()))
        worst = max(worst, abs(Float64(ve) - tv.at(t).get("value_ensemble").as_float()))
    if worst > CHECK_TOL:
        raise _e644("embedded test vectors of '" + dir + "' replay with error " + String(worst))
    var ro = man.get("rollouts")
    for r in range(len(ro)):
        var ij = ro.at(r).get("input")
        var st = List[UInt64]()
        for j in range(len(ij)):
            st.append(_as_u64(ij.at(j).number_text()))
        var env = PolicyEnv(m.kind_code, m.num_qubits, m)
        if m.kind_code == MODEL_KIND_PAULI:
            env.init_pauli(ro.at(r).get("R").as_int(), st)
        var ok = False
        var acts = greedy_trace(env, pol, m.max_depth, st, ok)
        var aj = ro.at(r).get("actions")
        if len(acts) != len(aj):
            raise _e644("rollout " + String(r) + " of '" + dir + "' replays with " + String(len(acts)) + " actions, stored " + String(len(aj)))
        for j in range(len(acts)):
            if acts[j] != aj.at(j).as_int():
                raise _e644("rollout " + String(r) + " of '" + dir + "' differs at step " + String(j))
        if ro.at(r).has("solved") and ok != ro.at(r).get("solved").as_bool():
            raise _e644("rollout " + String(r) + " of '" + dir + "' replays with a different solved flag")
    return worst


# ------------------------------------------------------------------------------------------------ independent checks


def _tol(x: Float64) -> Float64:
    return CHECK_TOL * max(1.0, abs(x))


def check_independent_forward(
    view: ModelFile, written: ModelFile, observations: List[List[Int]]
) raises:
    """The independent pre-write check of 9.2 (QM-E644 on failure). `view` is the pre-write tensors in the training
    layout (`BasicNet.from_model`), `written` the directory as the inference side reads it (`MLPPolicy`, and a naive
    Float64 forward over its decoded tensors). For each observation and each twist (none, then every symmetry) and
    for the ensemble mean over symmetries, the three results must agree in logits and value to `1e-5 * max(1, |x|)`.
    A tensor that was transposed or reordered on the way to the disk makes `view` and `written` disagree."""
    var net: BasicNet
    var pol: MLPPolicy
    try:
        net = BasicNet.from_model(view)
        pol = MLPPolicy.from_model(written)
    except e:
        raise _e644("the model cannot be read by both the training and the inference side (" + String(e) + ")")
    var P = len(written.obs_perms)
    if net.spec.num_twists() != P:
        raise _e644("training view has " + String(net.spec.num_twists()) + " symmetries, written model " + String(P))
    var A = pol.num_actions
    var sc = ActorScratch(net.spec)
    for oi in range(len(observations)):
        var idx = observations[oi].copy()
        var idx32 = List[Int32](capacity=len(idx))
        for q in range(len(idx)):
            idx32.append(Int32(idx[q]))
        var ens_a = List[Float64](length=A, fill=0.0)
        var ens_c = List[Float64](length=A, fill=0.0)
        var ens_va = 0.0
        var ens_vc = 0.0
        for tw in range(-1, P):
            var la = List[Float32](length=A, fill=0.0)
            var va = net.forward_one(Span(idx32), tw, sc, la.unsafe_ptr().as_unsafe_any_origin())
            _ = len(la)
            var lb = List[Float32]()
            var vb: Float32
            if tw < 0:
                vb = pol.forward(idx, lb)
            else:
                vb = pol.forward_sym(idx, tw, lb)
            var lc = List[Float64]()
            var vc = naive_forward_f64(written, idx, tw, lc)
            for a in range(A):
                var want = lc[a]
                if abs(Float64(la[a]) - want) > _tol(want) or abs(Float64(lb[a]) - want) > _tol(want):
                    raise _e644(
                        "independent forward check failed at observation "
                        + String(oi)
                        + ", twist "
                        + String(tw)
                        + ", action "
                        + String(a)
                        + ": BasicNet "
                        + String(la[a])
                        + ", MLPPolicy "
                        + String(lb[a])
                        + ", Float64 reference "
                        + String(want)
                    )
            if abs(Float64(va) - vc) > _tol(vc) or abs(Float64(vb) - vc) > _tol(vc):
                raise _e644(
                    "independent forward check failed at observation "
                    + String(oi)
                    + ", twist "
                    + String(tw)
                    + ": value BasicNet "
                    + String(va)
                    + ", MLPPolicy "
                    + String(vb)
                    + ", Float64 reference "
                    + String(vc)
                )
            if tw >= 0:
                for a in range(A):
                    ens_a[a] += Float64(la[a]) / Float64(P)
                    ens_c[a] += lc[a] / Float64(P)
                ens_va += Float64(va) / Float64(P)
                ens_vc += vc / Float64(P)
        if P > 0:
            var le = List[Float32]()
            var ve = pol.forward_ensemble(idx, le)
            for a in range(A):
                if abs(ens_a[a] - ens_c[a]) > _tol(ens_c[a]) or abs(Float64(le[a]) - ens_c[a]) > _tol(ens_c[a]):
                    raise _e644(
                        "independent ensemble check failed at observation "
                        + String(oi)
                        + ", action "
                        + String(a)
                        + ": BasicNet mean "
                        + String(ens_a[a])
                        + ", MLPPolicy ensemble "
                        + String(le[a])
                        + ", Float64 mean "
                        + String(ens_c[a])
                    )
            if abs(ens_va - ens_vc) > _tol(ens_vc) or abs(Float64(ve) - ens_vc) > _tol(ens_vc):
                raise _e644("independent ensemble check failed at observation " + String(oi) + ": value")


# ------------------------------------------------------------------------------------------------ Pauli ensemble check


def _random_policy_model(shell: ModelFile, emb: Int, hidden: Int, mut rng: Rng) -> ModelFile:
    """A random-weight `ModelFile` with the shell's gateset, observation size and symmetries and a one-hidden-layer
    trunk; the weights are normal with scales large enough for distinct logits."""
    var m = ModelFile()
    m.dir = String("<random policy>")
    m.kind_code = shell.kind_code
    m.kind = shell.kind
    m.num_qubits = shell.num_qubits
    m.actions = shell.actions.copy()
    m.max_rotations = shell.max_rotations
    m.max_depth = shell.max_depth
    m.obs_size = shell.obs_size
    m.obs_perms = shell.obs_perms.copy()
    m.act_perms = shell.act_perms.copy()
    var A = shell.num_actions()
    var names: List[String] = [
        "embeddings.weight",
        "embeddings.bias",
        "common.0.weight",
        "common.0.bias",
        "action.0.weight",
        "action.0.bias",
        "value.0.weight",
        "value.0.bias",
    ]
    var rows: List[Int] = [shell.obs_size, 1, hidden, 1, A, 1, 1, 1]
    var cols: List[Int] = [emb, emb, emb, hidden, hidden, A, hidden, 1]
    var scale: List[Float64] = [0.6, 0.2, 0.5, 0.2, 0.9, 0.3, 0.5, 0.2]
    for t in range(len(names)):
        var cnt = rows[t] * cols[t]
        var data = List[Float32](capacity=cnt)
        for _ in range(cnt):
            data.append(Float32(rng.normal() * scale[t]))
        var sh = List[Int]()
        if names[t].endswith(".bias"):
            sh.append(cnt)
        else:
            sh.append(rows[t])
            sh.append(cols[t])
        m.names.append(names[t])
        m.shapes.append(sh^)
        m.tensors.append(data^)
    return m^


def check_pauli_ensemble(m: ModelFile, seed: UInt64) raises:
    """Behavioural self-check of the exported Pauli symmetries against `TrainEnv`'s hidden-permutation view (4.6),
    QM-E644 on a mismatch. `m` supplies the gateset, qubit count, `max_rotations` and the export symmetries
    (`obs_perms`, `act_perms`); the weights of the check are random and generated here, so the test is not blunted by
    a degenerate trained policy.

    For random Pauli states (`TrainEnv.reset` at several difficulties) and every symmetry `p`: the training view feeds
    the policy the observation `observe_perm(p)` and un-permutes its action through the env table
    (`L_train[act_perm_env[p][a]] = raw[a]`); the inference view feeds the identity observation through
    `forward_sym(p)` (`obs_perms[p]`, `L[a] = raw[act_perms[p][a]]`). Both logit vectors and both values must agree to
    1e-4 (the two views add the embedding rows in different orders, so only f32 rounding differs). A wrong direction of either table fails on non-involutive automorphisms (ring:4, ring:5, star:4)."""
    if m.kind_code != MODEL_KIND_PAULI:
        raise _e640("check_pauli_ensemble needs a Pauli-network model, got kind " + m.kind)
    var cfg = EnvConfig()
    cfg.kind = MODEL_KIND_PAULI
    cfg.n = m.num_qubits
    cfg.actions = m.actions.copy()
    cfg.max_rotations = m.max_rotations
    var env = TrainEnv(cfg, seed)
    var P = env.num_syms()
    if P != len(m.obs_perms) or P != len(m.act_perms):
        raise _e644(
            "Pauli ensemble check: the model has " + String(len(m.obs_perms)) + " symmetries, the environment " + String(P)
        )
    var rng = Rng(seed_for(seed, TAG_EXPORT_VEC, 7))
    var shell = ModelFile()
    shell.kind_code = m.kind_code
    shell.kind = m.kind
    shell.num_qubits = m.num_qubits
    shell.actions = m.actions.copy()
    shell.max_rotations = m.max_rotations
    shell.max_depth = m.max_depth
    shell.obs_size = obs_size_of(cfg)
    shell.obs_perms = m.obs_perms.copy()
    shell.act_perms = m.act_perms.copy()
    var rm = _random_policy_model(shell, 32, 32, rng)
    var pol = MLPPolicy.from_model(rm)
    var A = env.num_actions()
    var diffs: List[Int] = [2, 8, 16, 33, 64, 128]
    var distinct = False
    for si in range(len(diffs)):
        env.reseed(seed_for(seed, TAG_EXPORT_VEC, 100 + si))
        env.set_difficulty(diffs[si])
        env.reset()
        var obs_id = List[Int]()
        env.pe.observe(env.st, obs_id)
        var first = List[Float32]()
        for p in range(P):
            var obs_p = List[Int]()
            env.observe_perm(p, obs_p)
            var raw = List[Float32]()
            var v_train = pol.forward(obs_p, raw)
            var ap = env.pauli_act_perm(p)
            var l_train = List[Float32](length=A, fill=0.0)
            for a in range(A):
                l_train[ap[a]] = raw[a]
            var l_exp = List[Float32]()
            var v_exp = pol.forward_sym(obs_id, p, l_exp)
            for a in range(A):
                if abs(Float64(l_train[a]) - Float64(l_exp[a])) > PAULI_TOL * max(1.0, abs(Float64(l_exp[a]))):
                    raise _e644(
                        "Pauli ensemble check failed: symmetry "
                        + String(p)
                        + " of "
                        + String(P)
                        + ", action "
                        + String(a)
                        + ": training view "
                        + String(l_train[a])
                        + ", export view "
                        + String(l_exp[a])
                    )
            if abs(Float64(v_train) - Float64(v_exp)) > PAULI_TOL * max(1.0, abs(Float64(v_exp))):
                raise _e644(
                    "Pauli ensemble check failed: symmetry " + String(p) + " value " + String(v_train) + " vs " + String(v_exp)
                )
            if p == 0:
                first = l_exp.copy()
            else:
                for a in range(A):
                    if abs(Float64(first[a]) - Float64(l_exp[a])) > 1e-4:
                        distinct = True
    if P > 1 and not distinct:
        raise _e644("Pauli ensemble check is degenerate: every symmetry gave the same logits")
    if env.error_code() != 0:
        raise _e644("Pauli ensemble check: the environment reported error " + String(env.error_code()))


# ------------------------------------------------------------------------------------------------ writing a model


def _model_head(
    id: String,
    kind: Int,
    n: Int,
    edges: List[Int],
    actions: List[Int],
    max_rot: Int,
    obs_size: Int,
    max_depth: Int,
    emb: Int,
    common: List[Int],
    sym: Twists,
    training_json: String,
) -> String:
    """Manifest members before `tensors`, in the order of the shipped models."""
    var s = String()
    s += '"format":"qumo.rlsynth.v0","id":' + json_quote(id)
    s += ',"kind":' + json_quote(model_kind_name(kind))
    s += ',"num_qubits":' + String(n)
    s += ',"edges":' + pairs_json(edges)
    s += ',"actions":' + actions_json(actions)
    s += ',"obs":' + obs_json(kind, n, max_rot, obs_size)
    s += ',"max_depth":' + String(max_depth)
    s += ',"arch":' + arch_json(emb, common)
    s += ',"symmetries":' + symmetries_json(sym.obs_perms, sym.act_perms)
    s += ',"training":' + training_json
    s += ',"license":"Apache-2.0"'
    return s^


def _symmetries_for(kind: Int, n: Int, actions: List[Int], max_rot: Int) -> Twists:
    if kind == MODEL_KIND_PAULI:
        return pauli_twists(n, actions, max_rot)
    return twists(kind, n, actions)


def _stage_check_write(
    final_dir: String,
    head: String,
    tensors: List[TensorOut],
    dtype: String,
    kind: Int,
    n: Int,
    actions: List[Int],
    obs_size: Int,
    sym: Twists,
    max_depth: Int,
    n_vec: Int,
    vec_depth0: Int,
    vec_dstep: Int,
    n_roll: Int,
    roll_depth0: Int,
    roll_dstep: Int,
) raises -> ModelDirInfo:
    """Stage `final_dir` in `<final_dir>.tmp`, run the independent checks, compute vectors and rollouts from the
    stored weights, write the finished manifest into the stage, replay it, and only then swap it into `final_dir`.
    Nothing outside `<final_dir>.tmp` is touched before everything passed; on any failure the stage is removed."""
    var stage = final_dir + ".tmp"
    recover_dir(final_dir)
    remove_tree(stage)
    var f32_names = List[String]()
    try:
        _ = write_model_dir(stage, head, tensors, dtype, f32_names, "")
        var written = load_model_file(stage)
        if not _same_lists(written.obs_perms, sym.obs_perms) or not _same_lists(written.act_perms, sym.act_perms):
            raise _e644("the symmetries read back from '" + stage + "' differ from the ones written")
        if written.obs_size != obs_size:
            raise _e644("observation size " + String(written.obs_size) + " read back, expected " + String(obs_size))
        var view = _view_from_tensors(tensors, dtype, sym.obs_perms, sym.act_perms, obs_size, actions)
        var obs = check_observations(written, 8, 1)
        check_independent_forward(view, written, obs)
        if kind == MODEL_KIND_PAULI and len(sym.obs_perms) > 0:
            check_pauli_ensemble(written, seed_for(EXPORT_SEED, TAG_EXPORT_VEC, 2))
        var pol = MLPPolicy.from_model(written)
        var rng = Rng(seed_for(EXPORT_SEED, TAG_EXPORT_VEC, 0))
        var tv = _vectors_json(written, pol, n_vec, vec_depth0, vec_dstep, rng)
        var ro = _rollouts_json(written, pol, n_roll, roll_depth0, roll_dstep, max_depth, rng)
        var tail = '"test_vectors":' + tv + ',"rollouts":' + ro
        var info = write_model_dir(stage, head, tensors, dtype, f32_names, tail)
        _ = verify_exported_model(stage)
        atomic_replace_dir(stage, final_dir)
        return info^
    except e:
        remove_tree(stage)
        raise Error(String(e))


# ------------------------------------------------------------------------------------------------ the exporter


def _training_json(
    info: RunInfo,
    summ: RunSummary,
    have_summary: Bool,
    which: String,
    cks: TrainState,
    have_state: Bool,
    eval_json: String,
) raises -> String:
    """The `training` object. Totals come from `summary.json`; a run without one (killed before its end) falls back
    to the train state of the exported checkpoint."""
    var iterations = summ.iterations if have_summary else cks.it
    var wall = summ.wall_s if have_summary else cks.elapsed_s
    var final_diff = summ.final_difficulty if have_summary else (cks.difficulty if have_state else info.start_difficulty)
    var s = String("{")
    s += '"producer":' + json_quote(PRODUCER)
    s += ',"engine":' + json_quote(ENGINE)
    s += ',"algo":"PPO"'
    s += ',"seed":' + String(info.seed)
    s += ',"budget_s":' + json_number(summ.budget_s if have_summary else info.budget_s)
    s += ',"wall_s":' + json_number(wall)
    s += ',"cores":' + String(info.cores)
    s += ',"iterations":' + String(iterations)
    s += ',"final_difficulty":' + String(final_diff)
    s += ',"topology_spec":' + json_quote(info.topology)
    s += ',"chunk_rows":' + String(info.chunk_rows)
    s += ',"vw":' + String(info.vw)
    s += ',"start_iteration":' + String(info.start_iteration)
    s += ',"checkpoint":' + json_quote(which)
    s += ',"eval":' + eval_json
    if info.init.byte_length() > 0:
        var bdir = info.init
        var base = _basename(bdir)
        if base == "best" or base == "last":
            bdir = _parent(bdir)
        try:
            if has_summary(bdir):
                var bs = read_summary(bdir)
                s += (
                    ',"continued_from":{"run":'
                    + json_quote(_basename(bdir))
                    + ',"budget_s":'
                    + json_number(bs.budget_s)
                    + ',"wall_s":'
                    + json_number(bs.wall_s)
                    + ',"final_difficulty":'
                    + String(bs.final_difficulty)
                    + "}"
                )
                s += ',"total_budget_s":' + json_number(bs.budget_s + (summ.budget_s if have_summary else info.budget_s))
                s += ',"total_wall_s":' + json_number(bs.wall_s + wall)
        except:
            pass
    s += "}"
    return s^


def export_synth_run(
    run_dir: String,
    which: String,
    id: String,
    root: String,
    dtype: String,
    make_default: Bool,
    eval_json: String = "",
) raises -> ModelDirInfo:
    """`which` is "best" or "last"; writes `<root>/<id>/` and updates `<root>/INDEX.json` (with a `default` column
    marking `id` when `make_default`). `eval_json` is the path of an evaluation JSON to embed under
    `training.eval` ("" for `{}`). QM-E644 when a pre-write check fails (the model root is left untouched), QM-E645
    when a weight does not fit f16 (export with `dtype` f32)."""
    if which != "best" and which != "last":
        raise _e640("--which must be best or last, got '" + which + "'")
    if dtype != "f16" and dtype != "f32":
        raise _e640("--dtype must be f16 or f32, got '" + dtype + "'")
    if id.byte_length() == 0 or id.find("/") >= 0:
        raise _e640("--id must be a plain directory name, got '" + id + "'")
    var rd = run_dir
    var ck = read_checkpoint(rd + "/" + which)
    var info = read_run_json(rd)
    var kind = info.kind
    var n = info.num_qubits
    if ck.num_qubits != n or len(ck.actions) != len(info.actions):
        raise _e644("run.json and the " + which + " checkpoint describe different gatesets")
    for i in range(len(info.actions)):
        if ck.actions[i] != info.actions[i]:
            raise _e644("run.json and the " + which + " checkpoint describe different gatesets")
    var max_rot = info.max_rotations if kind == MODEL_KIND_PAULI else 0
    var cfg = info.env_config()
    var obs_size = obs_size_of(cfg)
    var sym = _symmetries_for(kind, n, info.actions, max_rot)
    if kind != MODEL_KIND_PAULI:
        if not _same_lists(ck.obs_perms, sym.obs_perms) or not _same_lists(ck.act_perms, sym.act_perms):
            raise _e644("the symmetries stored in the checkpoint differ from the ones derived from the gateset")
    if not ck.has_tensor("embeddings.weight"):
        raise _e644("checkpoint has no embeddings.weight")
    var esh = ck.shape("embeddings.weight")
    if len(esh) != 2 or esh[0] != obs_size:
        raise _e644(
            "observation size of the checkpoint ("
            + String(esh[0] if len(esh) > 0 else -1)
            + ") does not match the environment ("
            + String(obs_size)
            + ")"
        )
    var emb = esh[1]
    var common = List[Int]()
    var k = 0
    while ck.has_tensor("common." + String(k) + ".weight"):
        common.append(ck.shape("common." + String(k) + ".weight")[0])
        k += 1
    var tensors = List[TensorOut]()
    for i in range(len(ck.names)):
        var name = ck.names[i]
        if name == "adam.m" or name == "adam.v":
            continue
        var shape = ck.shapes[i].copy()
        if name.endswith(".bias") and len(shape) == 2:
            var cnt = shape[0] * shape[1]
            shape = List[Int]()
            shape.append(cnt)
        if dtype == "f16" and not name.endswith(".bias"):
            var mx = max_abs_f32(Span(ck.tensors[i]))
            if Float64(mx) > F16_WARN:
                print(
                    "export_synth: warning: tensor "
                    + name
                    + " has max|w| = "
                    + String(mx)
                    + ", close to the f16 limit 65504"
                )
        tensors.append(TensorOut(name, shape^, ck.tensors[i].copy(), String("f16")))
    var have_summary = has_summary(rd)
    var summ = read_summary(rd) if have_summary else RunSummary()
    var cks = TrainState()
    var have_state = False
    try:
        cks = read_train_state(ck)
        have_state = True
    except:
        pass
    var ev = String("{}")
    if eval_json.byte_length() > 0:
        if not isfile(eval_json):
            raise _e640("--eval-json '" + eval_json + "' does not exist")
        ev = parse_json(_read_text(eval_json)).to_json()
    var head = _model_head(
        id,
        kind,
        n,
        info.edges,
        info.actions,
        max_rot,
        obs_size,
        info.max_depth,
        emb,
        common,
        sym,
        _training_json(info, summ, have_summary, which, cks, have_state, ev),
    )
    var final_dir = root + "/" + id
    var res = _stage_check_write(
        final_dir,
        head,
        tensors,
        dtype,
        kind,
        n,
        info.actions,
        obs_size,
        sym,
        info.max_depth,
        NUM_VECTORS,
        3,
        4,
        NUM_ROLLOUTS,
        6,
        6,
    )
    rebuild_index(root, "", id if make_default else String())
    return res^


# ------------------------------------------------------------------------------------------------ fixture generator


def _copy_file(src: String, dst: String) raises:
    var bytes: List[UInt8]
    with open(src, "r") as f:
        bytes = f.read_bytes()
    with open(dst, "w") as f:
        f.write_bytes(Span(bytes))


def generate_policy_fixtures(out_dir: String) raises -> List[String]:
    """Writes the tiny random-weight fixture models (`fx_perm_line3`, `fx_perm_star4`, `fx_lf_line3`,
    `fx_cliff_line2`, `fx_pauli_line3`; emb 16, common [16], max_depth 40, normal weights from the Mojo `Rng`) to
    `<out_dir>/synth/<id>/` with `<out_dir>/synth/INDEX.json`, plus the corrupted copies `<out_dir>/bad_checksum`
    and `<out_dir>/bad_format` the loader tests use. Returns the ids. The committed fixtures stay frozen; running this into
    a scratch directory only proves the generator works."""
    var ids: List[String] = ["fx_perm_line3", "fx_perm_star4", "fx_lf_line3", "fx_cliff_line2", "fx_pauli_line3"]
    var kinds: List[Int] = [MODEL_KIND_PERM, MODEL_KIND_PERM, MODEL_KIND_LINEAR, MODEL_KIND_CLIFFORD, MODEL_KIND_PAULI]
    var topos: List[String] = ["line:3", "star:4", "line:3", "line:2", "line:3"]
    var dtypes: List[String] = ["f32", "f16", "f16", "f16", "f16"]
    var max_rots: List[Int] = [0, 0, 0, 0, 2]
    var root = out_dir + "/synth"
    makedirs(root, exist_ok=True)
    var emb = 16
    var hidden = 16
    var made = List[String]()
    for k in range(len(ids)):
        var t = parse_topology(topos[k])
        var actions = build_gateset(kinds[k], t.n, t.edges)
        var cfg = EnvConfig()
        cfg.kind = kinds[k]
        cfg.n = t.n
        cfg.max_rotations = max_rots[k]
        var obs_size = obs_size_of(cfg)
        var A = len(actions) // 3
        var rng = Rng(UInt64(1000 + k))
        var names: List[String] = [
            "embeddings.weight",
            "embeddings.bias",
            "common.0.weight",
            "common.0.bias",
            "action.0.weight",
            "action.0.bias",
            "value.0.weight",
            "value.0.bias",
        ]
        var rows: List[Int] = [obs_size, 1, hidden, 1, A, 1, 1, 1]
        var cols: List[Int] = [emb, emb, emb, hidden, hidden, A, hidden, 1]
        var scale: List[Float64] = [0.5, 0.1, 0.4, 0.1, 0.6, 0.1, 0.3, 0.1]
        var tensors = List[TensorOut]()
        for q in range(len(names)):
            var cnt = rows[q] * cols[q]
            var data = List[Float32](capacity=cnt)
            for _ in range(cnt):
                data.append(Float32(rng.normal() * scale[q]))
            var sh = List[Int]()
            if names[q].endswith(".bias"):
                sh.append(cnt)
            else:
                sh.append(rows[q])
                sh.append(cols[q])
            tensors.append(TensorOut(names[q], sh^, data^, String("f16")))
        var sym = _symmetries_for(kinds[k], t.n, actions, max_rots[k])
        var common = List[Int]()
        common.append(hidden)
        var training = (
            '{"producer":"random fixture (tests only)","algo":"none","seed":'
            + String(1000 + k)
            + ',"budget_s":0,"wall_s":0,"cores":0,"final_difficulty":0,"eval":{}}'
        )
        var head = _model_head(ids[k], kinds[k], t.n, t.edges, actions, max_rots[k], obs_size, 40, emb, common, sym, training)
        _ = _stage_check_write(
            root + "/" + ids[k], head, tensors, dtypes[k], kinds[k], t.n, actions, obs_size, sym, 40, 4, 2, 3, 2, 3, 2
        )
        made.append(ids[k])
    rebuild_index(root, "", String())
    # corrupted copies (outside synth/, never indexed): a flipped weight byte, an unknown format
    var src = root + "/fx_perm_line3"
    for name in ["bad_checksum", "bad_format"]:
        var d = out_dir + "/" + name
        remove_tree(d)
        makedirs(d, exist_ok=True)
        _copy_file(src + "/manifest.json", d + "/manifest.json")
        _copy_file(src + "/weights.bin", d + "/weights.bin")
    var wb: List[UInt8]
    with open(out_dir + "/bad_checksum/weights.bin", "r") as f:
        wb = f.read_bytes()
    wb[100] = wb[100] ^ UInt8(1)
    with open(out_dir + "/bad_checksum/weights.bin", "w") as f:
        f.write_bytes(Span(wb))
    var mt = _read_text(out_dir + "/bad_format/manifest.json")
    with open(out_dir + "/bad_format/manifest.json", "w") as f:
        f.write(String(mt.replace("qumo.rlsynth.v0", "qumo.rlsynth.v9")))
    return made^
