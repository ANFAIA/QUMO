"""Router export: a run directory to a `qumo.rlroute.v0` model with test vectors, parity episodes and INDEX handling.

`export_route_run(run, root, id, make_default, episodes, meta_path)` turns a router run directory (the f32 container
that `bc.write_bc_run` or the router PPO writes, or any router container such as a shipped f16 model) into
`<root>/<id>/`:

  * weights: `embeddings.weight` [128, H] and `action.0.weight` [16, H] (and `value.0.weight` [1, H] when the run
    has a value head) in f16, every `*.bias` in f32, 64-byte aligned (`container.write_model_dir`);
  * manifest members, in the layout of the shipped model: format, id, kind, num_qubits, obs, arch, derived_from,
    producer, license, training, test_vectors;
  * `training`: the run manifest's `training` object, merged with `train_log.json` (without its `log`), then with
    `<run>/training_meta.json` and with the file named by `meta_path` (later files win);
  * `test_vectors.obs_logits`: 6 observations (3 sparse integer vectors in [-3, 3] and 3 Bernoulli(0.5) vectors)
    with the 16 logits of a Float64 forward over the STORED (f16-rounded) weights;
  * `test_vectors.episodes`: `episodes` routing episodes (jobs from `build_jobs(424242, N, grid 3x3 + heavy-hex 19,
    max_n = 9)`, a seeded layout and router seed each) played through `Episode` in policy mode with the stored
    weights; the action trace, overhead, final layout, fallback count and failure flag are embedded so
    `tests/test_ai_route.mojo` style replays are bit exact.

Independent checks run before anything is written into `<root>/<id>`: the vectors, made by the runtime
`RouteModel.forward`, are re-derived by a naive Float64 forward over the decoded tensors and by the training
network (`RouterNet`, f32, training layout); any disagreement aborts with QM-E644. Every parity episode must pass
`verify_episode`. The staged copy lives in `<root>/<id>.export.tmp` (skipped by the INDEX writer); the final
directory is only replaced once all checks have passed, and `INDEX.json` is rewritten last.

INDEX default handling (defect 7.4 of the old exporter): with `make_default` the exported id becomes the default;
without it the previous default is preserved.

Error codes: QM-E640 usage (bad id, malformed meta file), QM-E644 container / vector / parity mismatch, QM-E645
f16 overflow (raised by the writer), QM-E647 run directory is not a router container (from `RouterNet.from_model`).
"""

from std.collections import List
from std.os import makedirs
from std.os.path import exists, isfile
from qrl.json import JsonValue, parse_json, json_quote, json_number
from qrl.rng import Rng
from qrl.model import ModelFile, load_model_file
from qumo.compiler.ai_route import (
    RouteModel,
    load_route_model,
    RProblem,
    Episode,
    episode_max_steps,
    verify_episode,
    MODE_POLICY,
    NUM_ACTIVE_SWAPS,
    OBS_SIZE,
)
from qrl.container import (
    TensorOut,
    ModelDirInfo,
    write_model_dir,
    recover_dir,
    rebuild_index,
    remove_tree,
)
from qrl.jobs import build_jobs, op_name, JobSet
from qumo_bridge.circuits import circuit_from_json, target_from_json
from qrl.targets import neutral_topology, RTopo
from qrl.rlrng import seed_for, randperm, TAG_EXPORT_VEC
from qrl.router_net import RouterNet, RouterCache, SPtr

comptime N_OBS_VECTORS = 6
comptime EXPORT_OBS_SEED: UInt64 = 0
comptime EXPORT_LAYOUT_SEED: UInt64 = 7
comptime EPISODE_JOBS_SEED = 424242
comptime EPISODE_MAX_QUBITS = 9
comptime DEFAULT_EPISODES = 2

comptime DERIVED_FROM = (
    "IBM Quantum transpiler AI routing (Apache-2.0, rust/src/routing of release 0.18.0): algorithm and network shape"
    " only; weights trained by QUMO"
)
comptime PRODUCER = (
    "QUMO training/ai_transpiler/route (native Mojo stack qumo.rl: behaviour cloning / DAgger from a rollout"
    " teacher, optional PPO)"
)
comptime FORWARD_TEXT = (
    "logits = W1 relu(b0 + sum_i sign(obs_i) E[i,:]) + b1; argmax, lowest index on ties, over valid active swaps"
)


def _e640(what: String) -> Error:
    return Error("QM-E640: " + what + ". Hint: usage is export_route RUN --root ROOT --id ID [--default] [--episodes N] [--meta FILE].")


def _e644(what: String) -> Error:
    return Error("QM-E644: " + what + ". Hint: the model under <root>/<id> and INDEX.json were left untouched; re-run the export.")


# ------------------------------------------------------------------------------------------------ small helpers


def _read_text(path: String) raises -> String:
    var text: String
    with open(path, "r") as f:
        text = f.read()
    return text^


def _sign(v: Int) -> Int:
    if v > 0:
        return 1
    if v < 0:
        return -1
    return 0


def _valid_id(id: String) -> Bool:
    if id.byte_length() == 0 or id.startswith(".") or id.endswith(".tmp") or id.endswith(".old"):
        return False
    for b in id.as_bytes():
        var c = Int(b)
        var ok = (c >= ord("a") and c <= ord("z")) or (c >= ord("A") and c <= ord("Z")) or (c >= ord("0") and c <= ord("9"))
        if not (ok or c == ord("_") or c == ord("-") or c == ord(".")):
            return False
    return True


def _int_list(xs: List[Int]) -> String:
    var s = String("[")
    for i in range(len(xs)):
        if i > 0:
            s += ","
        s += String(xs[i])
    s += "]"
    return s^


def _float_list(xs: List[Float64]) -> String:
    var s = String("[")
    for i in range(len(xs)):
        if i > 0:
            s += ","
        s += json_number(xs[i])
    s += "]"
    return s^


def _put_member(mut keys: List[String], mut texts: List[String], key: String, text: String):
    """Sets member `key` to the JSON text `text`: replaced in place when present, appended otherwise."""
    for i in range(len(keys)):
        if keys[i] == key:
            texts[i] = text
            return
    keys.append(key)
    texts.append(text)


def _merge_object(mut keys: List[String], mut texts: List[String], obj: JsonValue, skip: String) raises:
    for k in obj.keys():
        if k == skip:
            continue
        _put_member(keys, texts, k, obj.get(k).to_json())


def _merge_file(mut keys: List[String], mut texts: List[String], path: String, what: String, skip: String) raises:
    var j = parse_json(_read_text(path))
    if not j.is_object():
        raise _e640(what + " '" + path + "' is not a JSON object")
    _merge_object(keys, texts, j, skip)


def _training_json(run_dir: String, mf: ModelFile, meta_path: String) raises -> String:
    """`training` object: run manifest, then train_log.json minus `log`, then run training_meta.json, then `meta_path`."""
    var keys = List[String]()
    var texts = List[String]()
    var man = parse_json(mf.manifest)
    if man.has("training") and man.get("training").is_object():
        _merge_object(keys, texts, man.get("training"), "")
    var lp = run_dir + "/train_log.json"
    if isfile(lp):
        _merge_file(keys, texts, lp, "train_log.json", "log")
    var rp = run_dir + "/training_meta.json"
    if isfile(rp):
        _merge_file(keys, texts, rp, "training_meta.json", "")
    if meta_path.byte_length() > 0:
        if not isfile(meta_path):
            raise _e640("meta file '" + meta_path + "' does not exist")
        _merge_file(keys, texts, meta_path, "meta file", "")
    var has_engine = False
    var has_algo = False
    for k in keys:
        if k == "engine":
            has_engine = True
        if k == "algo":
            has_algo = True
    var out = String("{")
    var first = True
    if not has_engine:
        out += '"engine":"qumo.rl (Mojo)"'
        first = False
    if not has_algo:
        if not first:
            out += ","
        out += '"algo":"behaviour cloning + DAgger"'
        first = False
    for i in range(len(keys)):
        if not first:
            out += ","
        out += json_quote(keys[i]) + ":" + texts[i]
        first = False
    out += "}"
    return out^


# ------------------------------------------------------------------------------------------------ vectors


def route_test_observations() -> List[List[Int]]:
    """The 6 observation vectors of `test_vectors.obs_logits` (a pure function of the constants above): three vectors of
    integers uniform in [-3, 3] with each entry zeroed with probability 1/2, then three Bernoulli(1/2) {0, 1} vectors."""
    var rng = Rng(seed_for(EXPORT_OBS_SEED, TAG_EXPORT_VEC, 0))
    var out = List[List[Int]]()
    for v in range(N_OBS_VECTORS):
        var obs = List[Int](length=OBS_SIZE, fill=0)
        if v < 3:
            for i in range(OBS_SIZE):
                obs[i] = rng.below(7) - 3
            for i in range(OBS_SIZE):
                if rng.uniform() < 0.5:
                    obs[i] = 0
        else:
            for i in range(OBS_SIZE):
                obs[i] = 1 if rng.uniform() < 0.5 else 0
        out.append(obs^)
    return out^


def route_logits(model: RouteModel, obs: List[Int]) -> List[Float64]:
    """Float64 logits of the runtime router for a raw observation (the sign is taken here)."""
    var sg = List[Int](length=OBS_SIZE, fill=0)
    for i in range(OBS_SIZE):
        sg[i] = _sign(obs[i])
    var hid = List[Float64](length=model.hidden, fill=0.0)
    var lg = List[Float64](length=NUM_ACTIVE_SWAPS, fill=0.0)
    model.forward(sg, hid, lg)
    return lg^


def _naive_logits(
    emb: List[Float32], b0: List[Float32], w1: List[Float32], b1: List[Float32], H: Int, obs: List[Int]
) -> List[Float64]:
    """Plain-loop Float64 forward (hidden unit outer, observation inner): shares no code with `RouteModel.forward`."""
    var hid = List[Float64](length=H, fill=0.0)
    for h in range(H):
        var acc = Float64(b0[h])
        for i in range(OBS_SIZE):
            acc += Float64(_sign(obs[i])) * Float64(emb[i * H + h])
        hid[h] = acc if acc > 0.0 else 0.0
    var out = List[Float64](length=NUM_ACTIVE_SWAPS, fill=0.0)
    for k in range(NUM_ACTIVE_SWAPS):
        var acc = Float64(b1[k])
        for h in range(H):
            acc += Float64(w1[k * H + h]) * hid[h]
        out[k] = acc
    return out^


def check_route_vectors(mf: ModelFile, obs: List[List[Int]], logits: List[List[Float64]]) raises:
    """Independent re-derivation of the embedded logits (QM-E644 on a disagreement): a naive Float64 forward over
    the decoded tensors must agree to 1e-9, and the training network (`RouterNet`, f32 accumulation, training
    layout) to 1e-4 absolute plus 1e-4 relative. Binds the training layout to the inference layout."""
    if len(obs) != len(logits):
        raise _e644("vector count mismatch")
    var emb = mf.tensor("embeddings.weight")
    var b0 = mf.tensor("embeddings.bias")
    var w1 = mf.tensor("action.0.weight")
    var b1 = mf.tensor("action.0.bias")
    var H = len(b0)
    var net = RouterNet.from_model(mf)
    var rows = len(obs)
    var sg = List[Int8](length=rows * OBS_SIZE, fill=0)
    for r in range(rows):
        for i in range(OBS_SIZE):
            sg[r * OBS_SIZE + i] = Int8(_sign(obs[r][i]))
    var cc = RouterCache(max(rows, 1), H)
    var sp = sg.unsafe_ptr().as_unsafe_any_origin()
    net.forward_chunk(sp, rows, cc, False)
    _ = len(sg)
    for r in range(rows):
        var naive = _naive_logits(emb, b0, w1, b1, H, obs[r])
        for k in range(NUM_ACTIVE_SWAPS):
            var want = logits[r][k]
            if abs(naive[k] - want) > 1e-9:
                raise _e644(
                    "obs vector " + String(r) + " logit " + String(k) + ": stored " + String(want)
                    + " but the independent Float64 forward gives " + String(naive[k])
                )
            var f32v = Float64(cc.z[r * NUM_ACTIVE_SWAPS + k])
            if abs(f32v - want) > 1e-4 + 1e-4 * abs(want):
                raise _e644(
                    "obs vector " + String(r) + " logit " + String(k) + ": stored " + String(want)
                    + " but the training network gives " + String(f32v)
                )


def _circuit_json(c_n: Int, ops: List[Int]) -> String:
    var s = String('{"n":' + String(c_n) + ',"ops":[')
    var o = 0
    while o + 2 < len(ops):
        if o > 0:
            s += ","
        s += '["' + op_name(ops[o]) + '",' + String(ops[o + 1]) + "," + String(ops[o + 2]) + "]"
        o += 3
    s += "]}"
    return s^


def _target_json(t: RTopo) -> String:
    var s = String('{"n":' + String(t.n) + ',"edges":[')
    var k = 0
    while k + 1 < len(t.edges):
        if k > 0:
            s += ","
        s += "[" + String(t.edges[k]) + "," + String(t.edges[k + 1]) + "]"
        k += 2
    s += "]}"
    return s^


def route_episode_vectors(model: RouteModel, episodes: Int) raises -> String:
    """JSON array of `episodes` parity episodes (see the module docstring). Every episode that did not fail must pass
    the structural verification (QM-E644 otherwise)."""
    if episodes <= 0:
        return String("[]")
    var topos = List[RTopo]()
    topos.append(neutral_topology("grid", 3, 3))
    topos.append(neutral_topology("heavy_hex", 19))
    var js = build_jobs(EPISODE_JOBS_SEED, episodes, topos, List[String](), EPISODE_MAX_QUBITS, 1.0, 0.4)
    var rng = Rng(seed_for(EXPORT_LAYOUT_SEED, TAG_EXPORT_VEC, 1))
    var out = String("[")
    for j in range(js.num_jobs()):
        ref jc = js.circuits[js.jobs[3 * j]]
        ref jt = js.targets[js.jobs[3 * j + 1]]
        var cjson = _circuit_json(jc.n, jc.ops)
        var tjson = _target_json(jt)
        var layout = randperm(rng, jt.n)
        var seed = Int(rng.next_u64() >> 24)  # 40 bits
        var c = circuit_from_json(parse_json(cjson))
        var t = target_from_json(parse_json(tjson))
        var prob = RProblem(c.ops, c.num_qubits)
        var ep = Episode(prob, t, layout, UInt64(seed), True)
        ep.run(prob, t, model, MODE_POLICY, 1.0, episode_max_steps(prob, t))
        if not ep.failed:
            var why = String()
            if not verify_episode(prob, ep, t, why):
                raise _e644("parity episode " + String(j) + " fails verification: " + why)
        if j > 0:
            out += ","
        out += '{"circuit":' + cjson + ',"target":' + tjson + ',"layout":' + _int_list(layout)
        out += ',"seed":' + String(seed) + ',"actions":' + _int_list(ep.actions)
        out += ',"overhead":' + String(ep.overhead()) + ',"final":' + _int_list(ep.l2p)
        out += ',"fallback":' + String(ep.n_fallback) + ',"failed":' + ("true" if ep.failed else "false") + "}"
    out += "]"
    return out^


# ------------------------------------------------------------------------------------------------ export


def _manifest_head(id: String, hidden: Int, value_head: Bool, derived_from: String, training: String, vectors: String) -> String:
    var h = String()
    h += '"format":"qumo.rlroute.v0","id":' + json_quote(id) + ',"kind":"routing","num_qubits":0'
    h += ',"obs":{"size":128,"layout":"route_sign_h8_k16","encoding":"sign(obs[d + 8k])"}'
    h += (
        ',"arch":{"type":"sign_embedding_mlp","horizon":8,"num_active_swaps":16,"hidden":' + String(hidden)
        + ',"activation":"relu","value_head":' + ("true" if value_head else "false")
        + ',"first_layer_transposed":true,"forward":' + json_quote(FORWARD_TEXT) + "}"
    )
    h += ',"derived_from":' + json_quote(derived_from)
    h += ',"producer":' + json_quote(PRODUCER)
    h += ',"license":"Apache-2.0"'
    h += ',"training":' + training
    if vectors.byte_length() > 0:
        h += ',"test_vectors":' + vectors
    return h^


def _export_tensors(net: RouterNet, value_head: Bool) -> List[TensorOut]:
    var all = net.to_tensors()
    var out = List[TensorOut]()
    var n = len(all) if value_head else 4
    for i in range(n):
        var t = all[i].copy()
        t.dtype = "f16"  # the writer's default rules: weights in f16, every *.bias in f32
        out.append(t^)
    return out^


def export_route_run(
    run_dir: String, root: String, id: String, make_default: Bool, episodes: Int, meta_path: String
) raises -> ModelDirInfo:
    """Writes `<root>/<id>/` (see the module docstring) and rebuilds `<root>/INDEX.json`. `make_default` False keeps the
    previous default of the INDEX. Returns the info of the written container (weights size, fnv1a64, sha256,
    manifest size)."""
    if not _valid_id(id):
        raise _e640("model id '" + id + "' is empty or has characters outside [A-Za-z0-9_.-], a leading '.', or ends in .tmp / .old")
    if episodes < 0:
        raise _e640("--episodes must be >= 0")
    var rd = run_dir
    while rd.byte_length() > 1 and rd.endswith("/"):
        var t = String(rd[byte = : rd.byte_length() - 1])
        rd = t^
    var rt = root
    while rt.byte_length() > 1 and rt.endswith("/"):
        var t = String(rt[byte = : rt.byte_length() - 1])
        rt = t^
    recover_dir(rd)
    var mf = load_model_file(rd)
    var net = RouterNet.from_model(mf)  # QM-E647 unless it is a router container
    var value_head = mf.has_tensor("value.0.weight") and mf.has_tensor("value.0.bias")
    var man = parse_json(mf.manifest)
    var derived = man.get_string("derived_from", DERIVED_FROM)
    var training = _training_json(rd, mf, meta_path)
    var tensors = _export_tensors(net, value_head)
    var final_dir = rt + "/" + id
    var stage = final_dir + ".export.tmp"
    if not exists(rt):
        makedirs(rt, exist_ok=True)
    if exists(stage) or exists(stage + ".tmp") or exists(stage + ".old"):
        remove_tree(stage)
        remove_tree(stage + ".tmp")
        remove_tree(stage + ".old")

    var info: ModelDirInfo
    try:
        # 1. stage the weights (f16 weights, f32 biases) and decode what was stored
        var sinfo = write_model_dir(stage, _manifest_head(id, net.hidden, value_head, derived, training, String()), tensors, "f16", List[String](), "")
        var staged_sha = sinfo.sha256
        var stored = load_model_file(stage)
        var rm = load_route_model(stage)
        # 2. vectors from the stored weights, independently re-derived before they are trusted
        var obs = route_test_observations()
        var lg = List[List[Float64]]()
        var vobs = String("[")
        for v in range(len(obs)):
            var l = route_logits(rm, obs[v])
            if v > 0:
                vobs += ","
            vobs += '{"obs":' + _int_list(obs[v]) + ',"logits":' + _float_list(l) + "}"
            lg.append(l^)
        vobs += "]"
        check_route_vectors(stored, obs, lg)
        var eps = route_episode_vectors(rm, episodes)
        var vectors = '{"obs_logits":' + vobs + ',"episodes":' + eps + "}"
        # 3. the final directory, replaced atomically only now
        info = write_model_dir(final_dir, _manifest_head(id, net.hidden, value_head, derived, training, vectors), tensors, "f16", List[String](), "")
        if info.sha256 != staged_sha:
            raise _e644("the final weights.bin differs from the staged one")
        _ = load_route_model(final_dir)  # the loader checks the manifest, the fnv1a64 and the shapes
    except e:
        try:
            remove_tree(stage)
        except:
            pass
        raise Error(String(e))
    remove_tree(stage)
    rebuild_index(rt, "", id if make_default else String())
    return info^
