"""Tests for qumo_bridge.export_route (MOJO_PORT.md 13.2 `test_rl_export_route`, acceptance A7 router half).

* a tiny behaviour-cloning run is exported: the directory loads through `load_route_model` (checksum, shapes), the
  weights are exactly the f16 rounding of the run's f32 weights (biases stay f32), the manifest carries the merged
  `training` object (run manifest, train_log.json without `log`, training_meta.json, --meta file), the 6 embedded
  obs -> logits vectors replay at 1e-5 through `RouteModel`, and the parity episodes replay exactly through
  `Episode` (action trace, overhead, final layout, fallback count) and pass `verify_episode`
* INDEX handling: the default is set with `make_default`, preserved when exporting another model (or re-exporting
  one) without it, moved when asked; rows are sorted; `bytes` and `sha256` match the files; `verify_models` is clean
* the shipped router, used as a run, re-exports to a byte-identical weights.bin (the independent oracle: the sha256 of
  the committed INDEX) and the exporter's Float64 forward reproduces the Python-made vectors of the shipped manifest
* determinism (two exports are byte-identical), hidden sizes that are not SIMD multiples, a run without a value head,
  `--episodes 0`
* failures change nothing: bad id, bad meta file, a synthesis container as run (QM-E647), an f16 overflow (QM-E645)
  leave the previous model directory and INDEX.json byte-identical and no staging directory behind
* the independent vector check (`check_route_vectors`) rejects a perturbed logit (QM-E644)
"""

from std.collections import List
from std.os import makedirs, listdir
from std.os.path import exists, isfile, isdir
from qrl.json import JsonValue, parse_json
from qrl.rng import Rng
from qumo.ir.circuit import Circuit
from qumo.compiler.target import Target
from qrl.model import load_model_file, f16_to_f32
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
from qrl.container import TensorOut, write_model_dir, verify_models, remove_tree
from qrl.f16 import f32_to_f16_bits
from qumo_bridge.circuits import circuit_from_json, target_from_json
from qrl.router_net import RouterNet
from qrl.bc import BCConfig, BCResult, BCEpochLog, bc_manifest_head, write_bc_run
from qumo_bridge.export_route import export_route_run, check_route_vectors, route_test_observations
from qrl.testutil import read_text, read_bytes

comptime WORK = "build/test_bridge_export_route"
comptime SHIPPED = "models/ai_transpiler/route/route_mlp_h64_v1"
comptime SHIPPED_INDEX = "models/ai_transpiler/route/INDEX.json"


def check(cond: Bool, what: String) raises:
    if not cond:
        raise Error("FAIL: " + what)


def expect_error(msg: String, code: String, what: String) raises:
    if not msg.startswith(code):
        raise Error("FAIL: " + what + ": expected " + code + ", got '" + msg + "'")


def fresh(path: String) raises:
    remove_tree(path)
    makedirs(path, exist_ok=True)


def write_text(path: String, text: String) raises:
    with open(path, "w") as f:
        f.write(text)


def sign_of(v: Int) -> Int:
    if v > 0:
        return 1
    if v < 0:
        return -1
    return 0


def make_run(dir: String, hidden: Int, seed: Int, scale: Float32, value_head: Bool = True) raises:
    """A BC run directory of a random network (weights scaled up so the policy is decisive)."""
    var net = RouterNet(hidden, UInt64(seed))
    for i in range(len(net.params)):
        net.params[i] = net.params[i] * scale
    var res = BCResult()
    res.samples = 1234
    res.val = 100
    res.wall_s = 1.5
    res.log.append(BCEpochLog(0, 1.5, 0.3, 2.0, 0.1))
    res.log.append(BCEpochLog(1, 1.2, 0.4, 1.5, 0.2))
    var cfg = BCConfig()
    cfg.hidden = hidden
    cfg.epochs = 2
    cfg.seed = UInt64(seed)
    var data: List[String] = ["d0.bin", "dir/d1.bin"]
    if value_head:
        write_bc_run(dir, net, res, cfg, data, "")
    else:
        var all = net.to_tensors()
        var four = List[TensorOut]()
        for i in range(4):
            four.append(all[i].copy())
        var head = bc_manifest_head("novalue", res, cfg, data, hidden, "")
        _ = write_model_dir(dir, head, four, "f32", List[String](), "")
        write_text(dir + "/train_log.json", '{"samples":1234,"val":100,"log":[{"epoch":0}]}')


def index_rows(root: String) raises -> JsonValue:
    return parse_json(read_text(root + "/INDEX.json"))


def index_default_ids(root: String) raises -> String:
    """Comma separated ids of the rows marked default, ids of all rows after a '|'."""
    var j = index_rows(root)
    var d = String()
    var a = String()
    for i in range(len(j)):
        var r = j.at(i)
        if r.get_bool("default", False):
            d += r.get("id").as_string() + ","
        a += r.get("id").as_string() + ","
    return d + "|" + a


def manifest_of(root: String, id: String) raises -> JsonValue:
    return parse_json(read_text(root + "/" + id + "/manifest.json"))


def dir_names(root: String) raises -> String:
    var names = List[String]()
    for n in listdir(root):
        names.append(n)
    sort(names)
    var s = String()
    for n in names:
        s += n + ","
    return s^


def replay_vectors(model: RouteModel, man: JsonValue) raises -> Int:
    """Replays obs -> logits at 1e-5 and the parity episodes exactly; returns the number of episodes."""
    var tv = man.get("test_vectors")
    var ol = tv.get("obs_logits")
    var hid = List[Float64](length=model.hidden, fill=0.0)
    var lg = List[Float64](length=NUM_ACTIVE_SWAPS, fill=0.0)
    for i in range(len(ol)):
        var v = ol.at(i)
        var obs = List[Int]()
        var oj = v.get("obs")
        check(len(oj) == OBS_SIZE, "obs vector length")
        for k in range(OBS_SIZE):
            obs.append(sign_of(oj.at(k).as_int()))
        model.forward(obs, hid, lg)
        var want = v.get("logits")
        check(len(want) == NUM_ACTIVE_SWAPS, "logit count")
        for k in range(NUM_ACTIVE_SWAPS):
            var d = abs(lg[k] - want.at(k).as_float())
            check(d < 1e-5, "obs -> logits replay (diff " + String(d) + ")")
    var eps = tv.get("episodes")
    for i in range(len(eps)):
        var e = eps.at(i)
        var c = circuit_from_json(e.get("circuit"))
        var t = target_from_json(e.get("target"))
        var lay = List[Int]()
        var lj = e.get("layout")
        for k in range(len(lj)):
            lay.append(lj.at(k).as_int())
        var prob = RProblem(c.ops, c.num_qubits)
        var ep = Episode(prob, t, lay, UInt64(e.get("seed").as_int()), True)
        ep.run(prob, t, model, MODE_POLICY, 1.0, episode_max_steps(prob, t))
        var acts = e.get("actions")
        check(len(ep.actions) == len(acts), "episode " + String(i) + " length")
        for k in range(len(acts)):
            check(ep.actions[k] == acts.at(k).as_int(), "episode " + String(i) + " action " + String(k))
        check(ep.overhead() == e.get("overhead").as_int(), "episode overhead")
        check(ep.n_fallback == e.get("fallback").as_int(), "episode fallback count")
        check(ep.failed == e.get("failed").as_bool(), "episode failed flag")
        var fj = e.get("final")
        check(len(fj) == len(ep.l2p), "final layout length")
        for k in range(len(fj)):
            check(ep.l2p[k] == fj.at(k).as_int(), "episode final layout")
        if not ep.failed:
            var why = String()
            check(verify_episode(prob, ep, t, why), "episode verification: " + why)
    return len(eps)


# ------------------------------------------------------------------------------------------------ tests


def test_basic_export() raises:
    var run = WORK + "/run_a"
    var root = WORK + "/root_a"
    fresh(WORK + "/basic")
    make_run(run, 16, 3, 3.0)
    write_text(run + "/training_meta.json", '{"machine":"test box","algo":"custom algo"}')
    write_text(WORK + "/meta_a.json", '{"epochs_run":99,"budget_cap_s":1}')
    remove_tree(root)
    var info = export_route_run(run, root, "route_a", True, 3, WORK + "/meta_a.json")
    var final = root + "/route_a"
    check(isfile(final + "/manifest.json") and isfile(final + "/weights.bin"), "files exist")
    check(not exists(final + ".export.tmp") and not exists(final + ".tmp") and not exists(final + ".old"), "no staging directories")
    check(dir_names(root) == "INDEX.json,route_a,", "root holds only the model and the INDEX: " + dir_names(root))
    # loads through the runtime loader (checksum, shapes)
    var rm = load_route_model(final)
    check(rm.ok and rm.hidden == 16 and rm.id == "route_a", "load_route_model")
    var man = manifest_of(root, "route_a")
    check(man.get("format").as_string() == "qumo.rlroute.v0", "format")
    check(man.get("kind").as_string() == "routing" and man.get_int("num_qubits", -1) == 0, "kind / num_qubits")
    check(man.get("arch").get_bool("value_head", False), "value_head flag")
    check(man.get("arch").get_int("hidden", 0) == 16, "hidden")
    check(man.get("license").as_string() == "Apache-2.0", "license")
    check(man.get("weights_bytes").as_int() == info.weights_bytes, "weights_bytes")
    check(man.get("fnv1a64").as_string() == info.fnv1a64 and man.get("sha256").as_string() == info.sha256, "digests")
    # weights are the f16 rounding of the run (weights) and exact (biases)
    var rf = load_model_file(run)
    var xf = load_model_file(final)
    var names: List[String] = ["embeddings.weight", "embeddings.bias", "action.0.weight", "action.0.bias", "value.0.weight", "value.0.bias"]
    var tens = man.get("tensors")
    check(len(tens) == 6, "six tensors")
    for t in range(6):
        check(tens.at(t).get("name").as_string() == names[t], "tensor order " + names[t])
        var src = rf.tensor(names[t])
        var dst = xf.tensor(names[t])
        check(len(src) == len(dst), "tensor size " + names[t])
        var is_bias = names[t].endswith(".bias")
        check(tens.at(t).get("dtype").as_string() == ("f32" if is_bias else "f16"), "dtype " + names[t])
        for k in range(len(src)):
            var want = src[k] if is_bias else f16_to_f32(f32_to_f16_bits(src[k]))
            check(dst[k] == want, "stored value of " + names[t] + " at " + String(k))
    # training object
    var tr = man.get("training")
    check(tr.is_object(), "training is an object")
    check(not tr.has("log"), "no per-epoch log in training")
    check(tr.get_int("samples", 0) == 1234 and tr.get_int("val", 0) == 100, "samples / val from the run")
    check(tr.get_string("engine", "") == "qumo.rl (Mojo)", "engine")
    check(tr.get_string("algo", "") == "custom algo", "training_meta.json overrides the run algo")
    check(tr.get_string("machine", "") == "test box", "training_meta.json member")
    check(tr.get_int("epochs_run", 0) == 99 and tr.get_int("budget_cap_s", 0) == 1, "--meta file wins")
    check(tr.get_int("chunk_rows", 0) == 128 and tr.get_int("vw", 0) == 4, "chunk_rows / vw from the run")
    var dj = tr.get("data")
    check(len(dj) == 2 and dj.at(0).as_string() == "d0.bin" and dj.at(1).as_string() == "d1.bin", "data basenames")
    # vectors: structure
    var tv = man.get("test_vectors")
    var ol = tv.get("obs_logits")
    check(len(ol) == 6, "six obs vectors")
    var saw_neg = False
    var saw_zero = False
    for v in range(6):
        var oj = ol.at(v).get("obs")
        for k in range(OBS_SIZE):
            var o = oj.at(k).as_int()
            check(o >= -3 and o <= 3, "obs range")
            if v >= 3:
                check(o == 0 or o == 1, "Bernoulli vectors are 0/1")
            if v < 3 and o < 0:
                saw_neg = True
            if o == 0:
                saw_zero = True
    check(saw_neg and saw_zero, "integer vectors have negative and zero entries")
    # vectors: replay (obs -> logits at 1e-5, episodes exactly)
    var n_eps = replay_vectors(rm, man)
    check(n_eps == 3, "three parity episodes, got " + String(n_eps))
    # the observation vectors are the documented pure function
    var obs_fn = route_test_observations()
    for v in range(6):
        var oj = ol.at(v).get("obs")
        for k in range(OBS_SIZE):
            check(oj.at(k).as_int() == obs_fn[v][k], "obs vector is route_test_observations()")
    # INDEX
    var idx = index_rows(root)
    check(len(idx) == 1, "one INDEX row")
    var row = idx.at(0)
    check(row.get("id").as_string() == "route_a" and row.get("dir").as_string() == "route_a", "INDEX id / dir")
    check(row.get("kind").as_string() == "routing" and row.get_int("num_qubits", -1) == 0, "INDEX kind")
    check(row.get_bool("default", False), "INDEX default")
    check(row.get("sha256").as_string() == info.sha256, "INDEX sha256")
    check(row.get_int("bytes", 0) == info.weights_bytes + info.manifest_bytes, "INDEX bytes")
    var rep = verify_models(root)
    check(rep.n_problems == 0 and rep.n_files == 1, "verify_models clean: " + rep.lines[len(rep.lines) - 1])
    print("  basic export, f16 rounding, merged training, vectors and episodes replay, INDEX, verify OK")


def test_determinism_and_shapes() raises:
    var root = WORK + "/root_det"
    remove_tree(root)
    make_run(WORK + "/run_det", 13, 5, 3.0)
    var i1 = export_route_run(WORK + "/run_det", root, "m", True, 2, String())
    var m1 = read_text(root + "/m/manifest.json")
    var w1 = read_bytes(root + "/m/weights.bin")
    var i2 = export_route_run(WORK + "/run_det", root, "m", True, 2, String())
    var m2 = read_text(root + "/m/manifest.json")
    var w2 = read_bytes(root + "/m/weights.bin")
    check(m1 == m2, "manifest bytes identical on re-export")
    check(len(w1) == len(w2), "weights size")
    for k in range(len(w1)):
        check(w1[k] == w2[k], "weights byte")
    check(i1.sha256 == i2.sha256 and i1.fnv1a64 == i2.fnv1a64, "digests identical")
    var rm = load_route_model(root + "/m")
    check(rm.hidden == 13, "hidden 13 (not a SIMD multiple)")
    _ = replay_vectors(rm, parse_json(m2))
    # no episodes
    var i3 = export_route_run(WORK + "/run_det", root, "noeps", False, 0, String())
    var man = manifest_of(root, "noeps")
    check(len(man.get("test_vectors").get("episodes")) == 0, "--episodes 0 gives an empty list")
    check(len(man.get("test_vectors").get("obs_logits")) == 6, "vectors still present")
    check(i3.weights_bytes == i1.weights_bytes, "same weights size")
    # a run without a value head
    var rootv = WORK + "/root_nv"
    remove_tree(rootv)
    make_run(WORK + "/run_nv", 8, 9, 3.0, False)
    _ = export_route_run(WORK + "/run_nv", rootv, "nv", True, 1, String())
    var mv = manifest_of(rootv, "nv")
    check(not mv.get("arch").get_bool("value_head", True), "value_head false")
    check(len(mv.get("tensors")) == 4, "four tensors without a value head")
    var rmv = load_route_model(rootv + "/nv")
    _ = replay_vectors(rmv, mv)
    check(mv.get("training").get_int("samples", 0) == 1234, "training from train_log.json when the manifest has none of it")
    print("  determinism, hidden 13, --episodes 0, run without value head OK")


def test_index_default() raises:
    var root = WORK + "/root_idx"
    remove_tree(root)
    make_run(WORK + "/run_i1", 8, 1, 3.0)
    make_run(WORK + "/run_i2", 8, 2, 3.0)
    _ = export_route_run(WORK + "/run_i1", root, "r1", True, 1, String())
    check(index_default_ids(root) == "r1,|r1,", "r1 default: " + index_default_ids(root))
    _ = export_route_run(WORK + "/run_i2", root, "r2", False, 1, String())
    check(index_default_ids(root) == "r1,|r1,r2,", "default preserved when exporting r2 without --default: " + index_default_ids(root))
    _ = export_route_run(WORK + "/run_i1", root, "r1", False, 1, String())
    check(index_default_ids(root) == "r1,|r1,r2,", "default preserved when re-exporting r1: " + index_default_ids(root))
    _ = export_route_run(WORK + "/run_i2", root, "r2", True, 1, String())
    check(index_default_ids(root) == "r2,|r1,r2,", "default moved to r2: " + index_default_ids(root))
    _ = export_route_run(WORK + "/run_i1", root, "r1", False, 1, String())
    check(index_default_ids(root) == "r2,|r1,r2,", "r2 stays default after re-exporting r1: " + index_default_ids(root))
    # an id that sorts first, exported without --default, keeps the default row
    _ = export_route_run(WORK + "/run_i1", root, "a0", False, 0, String())
    check(index_default_ids(root) == "r2,|a0,r1,r2,", "rows sorted by directory, default kept: " + index_default_ids(root))
    var rep = verify_models(root)
    check(rep.n_problems == 0 and rep.n_files == 3, "verify_models: " + rep.lines[len(rep.lines) - 1])
    # a root with no INDEX yet and no --default: every row is non default
    var root2 = WORK + "/root_idx2"
    remove_tree(root2)
    _ = export_route_run(WORK + "/run_i1", root2, "solo", False, 0, String())
    check(index_default_ids(root2) == "|solo,", "no default asked, none set: " + index_default_ids(root2))
    print("  INDEX default set / preserved / moved, rows sorted, verify OK")


def test_shipped_roundtrip() raises:
    if not isfile(SHIPPED + "/manifest.json"):
        print("  shipped router not found: round trip skipped")
        return
    var root = WORK + "/root_ship"
    remove_tree(root)
    var info = export_route_run(SHIPPED, root, "ship_copy", False, 2, String())
    # weights.bin is byte-identical to the committed model, hence the committed INDEX digest
    var a = read_bytes(SHIPPED + "/weights.bin")
    var b = read_bytes(root + "/ship_copy/weights.bin")
    check(len(a) == len(b), "weights size equals the shipped one")
    for k in range(len(a)):
        check(a[k] == b[k], "weights.bin byte " + String(k))
    var sidx = parse_json(read_text(SHIPPED_INDEX))
    check(sidx.at(0).get("sha256").as_string() == info.sha256, "sha256 equals the committed INDEX digest")
    var sman = parse_json(read_text(SHIPPED + "/manifest.json"))
    var man = manifest_of(root, "ship_copy")
    check(man.get("training").get_int("samples", 0) == sman.get("training").get_int("samples", -1), "shipped training preserved")
    check(man.get_string("derived_from", "") == sman.get_string("derived_from", "x"), "derived_from preserved from the run manifest")
    check(man.get("fnv1a64").as_string() == sman.get("fnv1a64").as_string(), "fnv1a64 equals the shipped one")
    # the exporter's Float64 forward reproduces the Python-made vectors of the shipped manifest
    var rm = load_route_model(root + "/ship_copy")
    var sol = sman.get("test_vectors").get("obs_logits")
    var hid = List[Float64](length=rm.hidden, fill=0.0)
    var lg = List[Float64](length=NUM_ACTIVE_SWAPS, fill=0.0)
    var worst = 0.0
    for v in range(len(sol)):
        var obs = List[Int]()
        for k in range(OBS_SIZE):
            obs.append(sign_of(sol.at(v).get("obs").at(k).as_int()))
        rm.forward(obs, hid, lg)
        for k in range(NUM_ACTIVE_SWAPS):
            worst = max(worst, abs(lg[k] - sol.at(v).get("logits").at(k).as_float()))
    check(worst < 1e-6, "Python-made shipped vectors replay through the exported copy (max " + String(worst) + ")")
    _ = replay_vectors(rm, man)
    print("  shipped router re-exports byte-identically (sha256", info.sha256, ")")


def unchanged(root: String, idx_before: String, man_before: String, w_before: List[UInt8], names_before: String, what: String) raises:
    check(read_text(root + "/INDEX.json") == idx_before, what + ": INDEX.json changed")
    check(read_text(root + "/keep/manifest.json") == man_before, what + ": manifest changed")
    var w = read_bytes(root + "/keep/weights.bin")
    check(len(w) == len(w_before), what + ": weights size changed")
    for k in range(len(w)):
        check(w[k] == w_before[k], what + ": weights changed")
    check(dir_names(root) == names_before, what + ": directory listing changed: " + dir_names(root))


def test_failures_change_nothing() raises:
    var root = WORK + "/root_fail"
    remove_tree(root)
    make_run(WORK + "/run_f", 8, 4, 3.0)
    _ = export_route_run(WORK + "/run_f", root, "keep", True, 1, String())
    var idx_before = read_text(root + "/INDEX.json")
    var man_before = read_text(root + "/keep/manifest.json")
    var w_before = read_bytes(root + "/keep/weights.bin")
    var names_before = dir_names(root)

    # bad ids
    var bad: List[String] = ["", "a/b", ".hidden", "x.tmp", "y.old", "sp ace", "ü"]
    for b in bad:
        var raised = False
        try:
            _ = export_route_run(WORK + "/run_f", root, b, False, 0, String())
        except e:
            raised = True
            expect_error(String(e), "QM-E640", "bad id '" + b + "'")
        check(raised, "bad id '" + b + "' must raise")
    unchanged(root, idx_before, man_before, w_before, names_before, "bad id")
    # negative episode count
    var raised = False
    try:
        _ = export_route_run(WORK + "/run_f", root, "keep", True, -1, String())
    except e:
        raised = True
        expect_error(String(e), "QM-E640", "negative episodes")
    check(raised, "negative episodes must raise")
    unchanged(root, idx_before, man_before, w_before, names_before, "negative episodes")
    # a meta file that is not an object, and a missing one
    write_text(WORK + "/meta_bad.json", "[1,2]")
    raised = False
    try:
        _ = export_route_run(WORK + "/run_f", root, "keep", True, 1, WORK + "/meta_bad.json")
    except e:
        raised = True
        expect_error(String(e), "QM-E640", "meta array")
    check(raised, "meta array must raise")
    raised = False
    try:
        _ = export_route_run(WORK + "/run_f", root, "keep", True, 1, WORK + "/meta_missing.json")
    except e:
        raised = True
        expect_error(String(e), "QM-E640", "meta missing")
    check(raised, "missing meta must raise")
    unchanged(root, idx_before, man_before, w_before, names_before, "bad meta")
    # a missing run directory
    raised = False
    try:
        _ = export_route_run(WORK + "/no_such_run", root, "keep", True, 1, String())
    except e:
        raised = True
    check(raised, "missing run must raise")
    unchanged(root, idx_before, man_before, w_before, names_before, "missing run")
    # a synthesis container is not a router
    raised = False
    try:
        _ = export_route_run("tests/fixtures/ai/policy/synth/fx_perm_line3", root, "keep", True, 1, String())
    except e:
        raised = True
        expect_error(String(e), "QM-E647", "synthesis container as run")
    check(raised, "synthesis container must raise")
    unchanged(root, idx_before, man_before, w_before, names_before, "synthesis container")
    # f16 overflow: nothing is replaced, no staging directory is left
    var net = RouterNet(8, 11)
    net.params[5] = 70000.0
    var res = BCResult()
    var cfg = BCConfig()
    cfg.hidden = 8
    var data: List[String] = ["d.bin"]
    write_bc_run(WORK + "/run_big", net, res, cfg, data, "")
    raised = False
    try:
        _ = export_route_run(WORK + "/run_big", root, "keep", True, 1, String())
    except e:
        raised = True
        expect_error(String(e), "QM-E645", "f16 overflow")
    check(raised, "f16 overflow must raise")
    unchanged(root, idx_before, man_before, w_before, names_before, "f16 overflow")
    # the same overflow into a model id that does not exist yet leaves no directory
    raised = False
    try:
        _ = export_route_run(WORK + "/run_big", root, "fresh", True, 1, String())
    except e:
        raised = True
    check(raised, "f16 overflow into a new id must raise")
    unchanged(root, idx_before, man_before, w_before, names_before, "f16 overflow, new id")
    print("  bad id / meta / run / container / f16 overflow leave the model and INDEX untouched OK")


def test_vector_check() raises:
    var root = WORK + "/root_chk"
    remove_tree(root)
    make_run(WORK + "/run_c", 8, 6, 3.0)
    _ = export_route_run(WORK + "/run_c", root, "chk", True, 0, String())
    var mf = load_model_file(root + "/chk")
    var rm = load_route_model(root + "/chk")
    var obs = route_test_observations()
    var good = List[List[Float64]]()
    var hid = List[Float64](length=rm.hidden, fill=0.0)
    for v in range(len(obs)):
        var sg = List[Int]()
        for k in range(OBS_SIZE):
            sg.append(sign_of(obs[v][k]))
        var lg = List[Float64](length=NUM_ACTIVE_SWAPS, fill=0.0)
        rm.forward(sg, hid, lg)
        good.append(lg^)
    check_route_vectors(mf, obs, good)
    var bad = List[List[Float64]]()
    for v in range(len(good)):
        bad.append(good[v].copy())
    bad[2][7] += 1e-3
    var raised = False
    try:
        check_route_vectors(mf, obs, bad)
    except e:
        raised = True
        expect_error(String(e), "QM-E644", "perturbed logit")
    check(raised, "a perturbed logit must be rejected")
    # a transposed head is caught as well: swap two columns of the stored logits
    var swapped = List[List[Float64]]()
    for v in range(len(good)):
        swapped.append(good[v].copy())
    var t = swapped[0][0]
    swapped[0][0] = swapped[0][1]
    swapped[0][1] = t
    raised = False
    try:
        check_route_vectors(mf, obs, swapped)
    except e:
        raised = True
    check(raised, "swapped logits must be rejected")
    print("  independent vector check accepts good and rejects perturbed logits OK")


def main() raises:
    makedirs(WORK, exist_ok=True)
    test_basic_export()
    test_determinism_and_shapes()
    test_index_default()
    test_shipped_roundtrip()
    test_failures_change_nothing()
    test_vector_check()
    print("test_bridge_export_route OK")
