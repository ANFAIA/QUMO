"""Tests for qrl.checkpoint and qrl.export_synth (component C11a).

What is pinned, and against what:
  * run layout and checkpoints: `run.json`, `TrainState`, `best/` and `last/` round trip bit for bit (weights,
    Adam moments and step counter, the f32 best tuple); a resumed Adam run continues bit-identically to an
    uninterrupted one (sparse rows included); a kill between the directory renames is repaired; log truncation;
    `summary.json`; the warm-start rules of plan 7.5 (`auto`, a run directory, a bare container, QM-E647 cases);
  * the exporter against the nine shipped synth manifests: for every shipped model, exporting a random-weight run
    on the same topology reproduces the shipped `edges`, `actions`, `obs`, `max_depth`, `arch`, `symmetries`,
    `tensors` and `weights_bytes` exactly (Python-made oracle), so the manifest assembly, the layout strings and the
    container rules are bound to the real files;
  * the exported directories: checksum ok through `load_model_file`, embedded vectors replayed through `MLPPolicy` at
    1e-5, rollouts replayed exactly, `verify_models` reports 0 problems, INDEX rows and the `default` column,
    determinism (two exports are byte-identical), the f16 overflow path (QM-E645, root untouched);
  * the independent pre-write checks abort (QM-E644) on a deliberately transposed embedding tensor, and pass on the
    honest one;
  * `check_pauli_ensemble` on the six `envs/pauli_twists.json` topologies with the Python-dumped export tables, and
    a mutation check: feeding the direct instead of the inverse action tables fails exactly on the topologies with
    non-involutive automorphisms;
  * the safetensors importer on `ppo/tiny_policy.safetensors` (values equal the `policy_forward.json` weights, and
    the imported network reproduces the torch logits), on synthetic F16 / BF16 / F64 files and on malformed files;
  * `generate_policy_fixtures` (random-weight fixtures from the Mojo RNG) writes models that load and replay.
"""

from std.os import makedirs
from std.os.path import exists, isdir, isfile
from qrl.json import JsonValue, parse_json
from qrl.model import (
    ModelFile,
    MLPPolicy,
    load_model_file,
    gate_code,
    MODEL_KIND_PAULI,
    MODEL_KIND_PERM,
    MODEL_KIND_LINEAR,
    MODEL_KIND_CLIFFORD,
    model_kind_code,
)
from qrl.checkpoint import (
    RunInfo,
    RunSummary,
    TrainState,
    WarmStart,
    new_run_info,
    net_spec_for,
    run_json_text,
    config_hash_of,
    write_run_json,
    read_run_json,
    checkpoint_head,
    perms_json,
    net_tensors,
    write_checkpoint,
    read_checkpoint,
    has_checkpoint,
    write_best,
    write_last,
    read_train_state,
    load_adam_state,
    write_summary,
    read_summary,
    append_log_line,
    truncate_log,
    log_line_count,
    resolve_init,
    safetensors_import,
    safetensors_parse,
    basic_mlp_tensors,
    SafeTensor,
)
from qrl.container import (
    TensorOut,
    write_model_dir,
    remove_tree,
    verify_models,
    atomic_replace_dir_stop,
    recover_dir,
)
from qrl.export_synth import (
    export_synth_run,
    verify_exported_model,
    check_pauli_ensemble,
    check_independent_forward,
    check_observations,
    generate_policy_fixtures,
    naive_forward_f64,
    _view_from_tensors,
    _symmetries_for,
)
from qrl.nn import BasicNet, ActorScratch
from qrl.optim import Adam
from qrl.symmetry import pauli_qubit_perms_and_act_perms
from qrl.testutil import (
    load_fixture,
    load_fixture_bytes,
    floats32,
    ints,
    ints2,
    f32_bits,
    rand_fill,
    assert_list_close32,
    model_root,
    has_model_root,
    skip_without_models,
)

comptime WORK = "build/test_rl_export"


def expect(cond: Bool, msg: String) raises:
    if not cond:
        raise Error("test_rl_export: " + msg)


def expect_error(prefix: String, text: String, what: String) raises:
    if not text.startswith(prefix):
        raise Error("test_rl_export: " + what + ": expected " + prefix + ", got: " + text)


def same_bits(a: List[Float32], b: List[Float32]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if f32_bits(a[i]) != f32_bits(b[i]):
            return False
    return True


def read_text(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def make_run(
    run_dir: String, id: String, kind: Int, topo: String, emb: Int, common: List[Int], seed: Int
) raises -> RunInfo:
    """A run directory holding a random-weight network: run.json, best/ and a summary."""
    var info = new_run_info(id, kind, topo)
    info.emb = emb
    info.common = common.copy()
    info.seed = seed
    info.cores = 4
    info.budget_s = 12.5
    var net = BasicNet(net_spec_for(info), UInt64(seed))
    write_run_json(run_dir, info)
    var st = TrainState()
    st.it = 7
    st.difficulty = 5
    st.seed = seed
    st.config_hash = config_hash_of(info)
    _ = write_best(run_dir, net, info, st)
    var s = RunSummary()
    s.finished = True
    s.iterations = 7
    s.final_difficulty = 5
    s.wall_s = 3.5
    s.budget_s = 12.5
    s.cores = 4
    s.seed = seed
    s.topology = topo
    write_summary(run_dir, s)
    return info^


# ------------------------------------------------------------------------------------------------ checkpoints


def test_checkpoint_resume() raises:
    var dir = String(WORK) + "/ckpt"
    var info = new_run_info("ck", MODEL_KIND_PERM, "line:3")
    info.emb = 24
    var common: List[Int] = [16, 12]  # two common layers: consecutive numbering
    info.common = common.copy()
    info.config_json = String('{"lr":0.0003,"gamma":0.995}')
    write_run_json(dir, info)
    var back = read_run_json(dir)
    expect(run_json_text(back) == run_json_text(info), "run.json does not round trip")
    expect(config_hash_of(back) == config_hash_of(info), "config hash changed by a round trip")
    expect(back.kind == MODEL_KIND_PERM and back.num_qubits == 3 and len(back.actions) == 6, "run.json fields")
    var other = info.copy()
    other.seed = 2
    expect(config_hash_of(other) != config_hash_of(info), "config hash ignores the seed")
    var moved = info.copy()
    moved.cores = 1
    moved.budget_s = 99.0
    moved.iters = 3
    expect(config_hash_of(moved) == config_hash_of(info), "config hash depends on cores / budget / iters")
    var cfg2 = info.copy()
    cfg2.config_json = String('{"lr":0.001}')
    expect(config_hash_of(cfg2) != config_hash_of(info), "config hash ignores the driver configuration")

    var spec = net_spec_for(info)
    expect(spec.num_twists() == 2, "line:3 has 2 symmetries")
    var net = BasicNet(spec, 3)
    var E = spec.emb
    var OBS = spec.obs_size
    var opt = Adam(net.param_count(), 1e-3)
    opt.mark_sparse_rows(net.ps.offset("embeddings.weight"), E, OBS)
    var grads = List[Float32](length=net.param_count(), fill=0.0)
    var emb_off = net.ps.offset("embeddings.weight")
    # phase 1: embedding rows 5..8 receive no gradient (they stay untouched in the sparse region)
    for step in range(3):
        rand_fill(grads, UInt64(10 + step))
        for r in range(5, OBS):
            for c in range(E):
                grads[emb_off + r * E + c] = 0.0
        opt.step(net.ps.data, grads, 1e-3)
    var snap = net.ps.data.copy()  # the weights at step 3, which last/ must hold
    var st = TrainState()
    st.it = 3
    st.start_iteration = 40
    st.difficulty = 4
    st.have_best = True
    st.best_difficulty = 4
    st.best_success = Float32(0.8349999785423279)
    st.best_reward = Float32(-0.123456789)
    st.elapsed_s = 12.375
    st.seed = 3
    st.config_hash = config_hash_of(info)
    _ = write_last(dir, net, opt, info, st)
    _ = write_best(dir, net, info, st)
    expect(isfile(dir + "/state.json"), "state.json mirror missing")

    # uninterrupted continuation (rows 5.. now get gradients)
    var net_a = BasicNet(spec, 3, False)
    var ck = read_checkpoint(dir + "/last")
    net_a = BasicNet.from_model(ck)
    var opt_a = Adam(net_a.param_count(), 1e-3)
    opt_a.mark_sparse_rows(net_a.ps.offset("embeddings.weight"), E, OBS)
    load_adam_state(opt_a, ck)
    var rs = read_train_state(ck)
    expect(rs.it == 3 and rs.start_iteration == 40 and rs.difficulty == 4, "train state counters")
    expect(rs.have_best and rs.best_difficulty == 4, "best flag")
    expect(f32_bits(rs.best_success) == f32_bits(st.best_success), "best_success not f32-exact")
    expect(f32_bits(rs.best_reward) == f32_bits(st.best_reward), "best_reward not f32-exact")
    expect(rs.elapsed_s == 12.375 and rs.seed == 3 and rs.config_hash == st.config_hash, "train state fields")
    expect(rs.adam_t == 3 and opt_a.steps_taken() == 3, "adam step counter")
    expect(same_bits(net_a.ps.data, net.ps.data), "last/ weights are not bit-exact")
    expect(same_bits(opt_a.m, opt.m) and same_bits(opt_a.v, opt.v), "Adam moments are not bit-exact")
    expect(opt_a.touched_rows() == opt.touched_rows() and opt.touched_rows() == 5, "touched rows after resume")
    for step in range(3, 7):
        rand_fill(grads, UInt64(10 + step))
        opt.step(net.ps.data, grads, 1e-3)
        opt_a.step(net_a.ps.data, grads, 1e-3)
    expect(same_bits(net_a.ps.data, net.ps.data), "a resumed run diverged from the uninterrupted run")
    expect(opt_a.steps_taken() == 7 and opt.steps_taken() == 7, "steps after the continuation")

    # best/ carries the train state too, and no Adam tensors
    var bm = read_checkpoint(dir + "/best")
    expect(not bm.has_tensor("adam.m"), "best/ must not carry optimizer tensors")
    expect(read_train_state(bm).start_iteration == 40, "best/ train_state")
    var bnet = BasicNet.from_model(bm)
    expect(bnet.spec.num_twists() == 2 and len(bnet.spec.common) == 2, "best/ architecture and symmetries")
    # two common layers are named consecutively and shipped-style: biases are 1-D, matrices f32 here
    expect(bm.has_tensor("common.1.weight") and bm.shape("common.1.bias").__len__() == 1, "tensor naming / bias shape")
    # a load_adam_state on a best/ checkpoint is rejected
    var raised = False
    try:
        load_adam_state(opt_a, bm)
    except e:
        raised = String(e).startswith("QM-E647")
    expect(raised, "load_adam_state accepted a checkpoint without optimizer state")
    # an optimizer of the wrong size is rejected
    var small = Adam(10, 1e-3)
    raised = False
    try:
        load_adam_state(small, ck)
    except e:
        raised = String(e).startswith("QM-E647")
    expect(raised, "load_adam_state accepted a mismatching optimizer")
    # a shipped model has no train state
    if has_model_root():
        raised = False
        try:
            _ = read_train_state(load_model_file(model_root() + "/synth/perm_grid3x3"))
        except e:
            raised = String(e).startswith("QM-E647")
        expect(raised, "a shipped model has no train_state")

    # a kill between the two renames of a checkpoint swap: last/ is renamed to last.old; recovery restores it
    var tmp = dir + "/last_next"
    makedirs(tmp, exist_ok=True)
    with open(tmp + "/marker", "w") as f:
        f.write("new")
    atomic_replace_dir_stop(tmp, dir + "/last", 1)
    expect(not isdir(dir + "/last") and isdir(dir + "/last.old"), "stop after step 1")
    var again = read_checkpoint(dir + "/last")
    expect(has_checkpoint(dir + "/last"), "recover_dir did not restore last/")
    expect(same_bits(BasicNet.from_model(again).ps.data, snap), "the restored last/ is not the step-3 checkpoint")
    expect(not isdir(dir + "/last.old"), "last.old not consumed by recovery")
    remove_tree(tmp)
    print("  checkpoints, resume and recovery OK")


def test_run_files() raises:
    var dir = String(WORK) + "/files"
    remove_tree(dir)
    write_run_json(dir, new_run_info("rf", MODEL_KIND_PAULI, "line:4"))
    var ri = read_run_json(dir)
    expect(ri.max_rotations == 5 and ri.obs_size() == 104 and len(ri.actions) == 3 * 14, "pauli line:4 run info")
    expect(ri.env_config().n == 4 and ri.env_config().max_rotations == 5, "env_config")
    # log lines
    for i in range(5):
        append_log_line(dir, '{"it":' + String(i) + "}")
    expect(log_line_count(dir) == 5, "log line count")
    truncate_log(dir, 3)
    expect(log_line_count(dir) == 3, "truncate_log")
    expect(read_text(dir + "/log.jsonl") == '{"it":0}\n{"it":1}\n{"it":2}\n', "log content after truncation")
    truncate_log(dir, 10)
    expect(log_line_count(dir) == 3, "truncate_log beyond the end keeps everything")
    truncate_log(dir, 0)
    expect(log_line_count(dir) == 0, "truncate_log to zero")
    # summary
    var s = RunSummary()
    s.finished = True
    s.iterations = 335
    s.final_difficulty = 208
    s.best_success = 0.8525
    s.wall_s = 2100.0
    s.topology = String("line:4")
    write_summary(dir, s)
    var s2 = read_summary(dir)
    expect(s2.finished and s2.iterations == 335 and s2.final_difficulty == 208 and s2.wall_s == 2100.0, "summary round trip")
    expect(s2.best_success == 0.8525 and s2.topology == "line:4", "summary floats and strings")
    # old-format summary (no `finished`) counts as finished
    with open(dir + "/summary.json", "w") as f:
        f.write('{"seed":1,"budget_s":600.0,"wall_s":601.1,"cores":4,"iterations":976,"final_difficulty":256,"topology":"grid:3x3"}')
    var old = read_summary(dir)
    expect(old.finished and old.iterations == 976 and old.final_difficulty == 256, "old summary format")
    var raised = False
    try:
        _ = read_summary(String(WORK) + "/no_such_run")
    except e:
        raised = String(e).startswith("QM-E647")
    expect(raised, "missing summary raises QM-E647")
    print("  run.json, log.jsonl and summary.json OK")


def test_resolve_init() raises:
    var base = String(WORK) + "/warm"
    remove_tree(base)
    var common: List[Int] = [8]
    var info = make_run(base + "/X", "X", MODEL_KIND_PERM, "line:3", 8, common, 1)
    info.start_iteration = 100
    write_run_json(base + "/X", info)  # offset 100, 7 iterations in the summary
    var w = resolve_init("auto", base + "/X__cont")
    expect(w.model_dir == base + "/X/best" and w.is_run, "auto source")
    expect(w.start_difficulty == 5 and w.start_iteration == 107, "auto difficulty / iteration offset")
    var w2 = resolve_init("auto", base + "/X__cont", 9, 3)
    expect(w2.start_difficulty == 9 and w2.start_iteration == 3, "auto with explicit overrides")
    var raised = False
    try:
        _ = resolve_init("auto", base + "/Y")
    except e:
        raised = String(e).startswith("QM-E647")
    expect(raised, "auto needs an X__cont directory")
    raised = False
    try:
        _ = resolve_init("auto", base + "/Z__cont")
    except e:
        raised = String(e).startswith("QM-E647")
    expect(raised, "auto with a missing sibling summary raises QM-E647")
    var w3 = resolve_init(base + "/X", base + "/next")
    expect(w3.model_dir == base + "/X/best" and w3.is_run and w3.start_difficulty == 1 and w3.start_iteration == 107, "run dir init")
    var w4 = resolve_init(base + "/X", base + "/next", 6, -1)
    expect(w4.start_difficulty == 6 and w4.start_iteration == 107, "run dir init with --start-diff")
    if has_model_root():
        var w5 = resolve_init(model_root() + "/synth/perm_grid3x3", base + "/next")
        expect(not w5.is_run and w5.start_difficulty == 1 and w5.start_iteration == 0, "bare container init")
        expect(w5.model_dir == model_root() + "/synth/perm_grid3x3", "bare container path")
    raised = False
    try:
        _ = resolve_init(base + "/missing", base + "/next")
    except e:
        raised = String(e).startswith("QM-E647")
    expect(raised, "a missing init directory raises QM-E647")
    # a run without a summary: iterations come from last/'s train state
    var info2 = new_run_info("W", MODEL_KIND_PERM, "line:3")
    info2.emb = 8
    write_run_json(base + "/W", info2)
    var net = BasicNet(net_spec_for(info2), 2)
    var opt = Adam(net.param_count(), 1e-3)
    var st = TrainState()
    st.it = 12
    _ = write_best(base + "/W", net, info2, st)
    _ = write_last(base + "/W", net, opt, info2, st)
    var w6 = resolve_init(base + "/W", base + "/next")
    expect(w6.start_iteration == 12, "iterations from the last/ train state")
    print("  warm-start resolution OK")


# ------------------------------------------------------------------------------------------------ exporter


def json_eq(a: JsonValue, b: JsonValue) -> Bool:
    return a.to_json() == b.to_json()


def test_export_matches_shipped() raises:
    """Export a random-weight run on every shipped model's topology; everything except weights, training and the
    vectors must equal the shipped manifest."""
    if skip_without_models("test_export_matches_shipped"):
        return
    var rows = parse_json(read_text(model_root() + "/synth/INDEX.json"))
    var work = String(WORK) + "/ship"
    remove_tree(work)
    var root = work + "/root"
    var count = 0
    for r in range(len(rows)):
        var dir = rows.at(r).get_string("dir", "")
        var sm = parse_json(read_text(model_root() + "/synth/" + dir + "/manifest.json"))
        var topo = sm.get("training").get_string("topology_spec", "")
        var kind = model_kind_code(sm.get_string("kind", ""))
        var common: List[Int] = [256]
        var info = make_run(work + "/run_" + dir, dir, kind, topo, 512, common, 1 + r)
        var res = export_synth_run(work + "/run_" + dir, "best", dir, root, "f16", False)
        var mm = parse_json(read_text(root + "/" + dir + "/manifest.json"))
        var keys: List[String] = ["format", "id", "kind", "num_qubits", "edges", "actions", "obs", "max_depth", "arch", "symmetries", "tensors", "weights_file", "weights_bytes", "license"]
        for k in keys:
            expect(json_eq(mm.get(k), sm.get(k)), dir + ": manifest member '" + k + "' differs from the shipped one")
        expect(res.weights_bytes == sm.get_int("weights_bytes", -1), dir + ": weights_bytes")
        # tensors are f16 matrices and f32 biases like the shipped file (dtype is part of `tensors` above)
        expect(mm.get("test_vectors").__len__() == 6 and mm.get("rollouts").__len__() == 2, dir + ": vectors / rollouts count")
        expect(mm.get("training").get_string("engine", "") == "qumo.rl (Mojo)", dir + ": engine")
        expect(mm.get("training").get_string("topology_spec", "") == topo, dir + ": topology_spec")
        expect(mm.get("training").get_int("iterations", -1) == 7, dir + ": iterations from the summary")
        _ = verify_exported_model(root + "/" + dir)
        count += 1
    expect(count == 9, "expected nine shipped synth models")
    var rep = verify_models(root)
    expect(rep.n_problems == 0 and rep.n_index == 1 and rep.n_files == 9, "verify_models on the exported root: " + rep.lines[len(rep.lines) - 1])
    var idx = parse_json(read_text(root + "/INDEX.json"))
    expect(len(idx) == 9 and not idx.at(0).has("default"), "INDEX rows without a default column")
    # the shipped INDEX rows carry the same topology fields as ours
    for r in range(len(idx)):
        var mine = idx.at(r)
        for sr in range(len(rows)):
            if rows.at(sr).get_string("id", "") == mine.get_string("id", ""):
                expect(json_eq(rows.at(sr).get("wl_hash"), mine.get("wl_hash")), "INDEX wl_hash of " + mine.get_string("id", ""))
                expect(json_eq(rows.at(sr).get("edges"), mine.get("edges")), "INDEX edges")
                expect(rows.at(sr).get_int("bytes", 0) > 0, "shipped INDEX bytes")
    print("  export reproduces the shipped manifest structure of 9 models, verify_models clean OK")


def test_naive_forward_vs_shipped_vectors() raises:
    """The Float64 reference of the exporter's checks reproduces the independent (numpy-made) test vectors of the
    nine shipped models, for the identity forward and for the ensemble mean over symmetries."""
    if skip_without_models("test_naive_forward_vs_shipped_vectors"):
        return
    var rows = parse_json(read_text(model_root() + "/synth/INDEX.json"))
    var worst = 0.0
    for r in range(len(rows)):
        var dir = model_root() + "/synth/" + rows.at(r).get_string("dir", "")
        var m = load_model_file(dir)
        var man = parse_json(m.manifest)
        var tv = man.get("test_vectors")
        var P = len(m.obs_perms)
        for t in range(len(tv)):
            var obs = ints(tv.at(t).get("obs"))
            var l0 = List[Float64]()
            var v0 = naive_forward_f64(m, obs, -1, l0)
            var want = tv.at(t).get("logits")
            for a in range(len(l0)):
                worst = max(worst, abs(l0[a] - want.at(a).as_float()))
            worst = max(worst, abs(v0 - tv.at(t).get_float("value", 0.0)))
            var ens = List[Float64](length=len(l0), fill=0.0)
            var ve = 0.0
            for p in range(P):
                var lp = List[Float64]()
                var vp = naive_forward_f64(m, obs, p, lp)
                for a in range(len(lp)):
                    ens[a] += lp[a] / Float64(P)
                ve += vp / Float64(P)
            var wante = tv.at(t).get("logits_ensemble")
            for a in range(len(ens)):
                worst = max(worst, abs(ens[a] - wante.at(a).as_float()))
            worst = max(worst, abs(ve - tv.at(t).get_float("value_ensemble", 0.0)))
    expect(worst < 2e-5, "naive Float64 forward differs from the shipped vectors by " + String(worst))
    print("  naive Float64 forward reproduces the shipped vectors of 9 models (worst " + String(worst) + ") OK")


def test_export_determinism_and_default() raises:
    var work = String(WORK) + "/det"
    remove_tree(work)
    var common: List[Int] = [16]
    _ = make_run(work + "/run", "det", MODEL_KIND_LINEAR, "ring:4", 24, common, 5)
    var a = export_synth_run(work + "/run", "best", "m1", work + "/root", "f16", False)
    var b = export_synth_run(work + "/run", "best", "m1", work + "/root", "f16", False)
    expect(a.sha256 == b.sha256 and a.manifest_bytes == b.manifest_bytes, "re-export changed the bytes")
    var t1 = read_text(work + "/root/m1/manifest.json")
    var c = export_synth_run(work + "/run", "best", "m2", work + "/root", "f16", True)
    expect(c.sha256 == a.sha256, "same checkpoint, same weights")
    var t2 = read_text(work + "/root/m2/manifest.json")
    expect(t1.replace('"id":"m1"', '"id":"m2"') == t2, "manifests of two exports differ beyond the id")
    var idx = parse_json(read_text(work + "/root/INDEX.json"))
    expect(len(idx) == 2, "two INDEX rows")
    expect(idx.at(0).get_bool("default", True) == False and idx.at(1).get_bool("default", False) == True, "default column")
    # dtype f32 keeps every tensor f32 and the model replays
    var d = export_synth_run(work + "/run", "best", "m3", work + "/root", "f32", False)
    var man = parse_json(read_text(work + "/root/m3/manifest.json"))
    for i in range(len(man.get("tensors"))):
        expect(man.get("tensors").at(i).get_string("dtype", "") == "f32", "f32 export has an f16 tensor")
    expect(d.weights_bytes > a.weights_bytes, "f32 weights are larger")
    _ = verify_exported_model(work + "/root/m3")
    # the stage never survives
    expect(not isdir(work + "/root/m1.tmp") and not isdir(work + "/root/m3.tmp"), "stage directory left behind")
    # last/ can be exported too: write one
    var info = read_run_json(work + "/run")
    var net = BasicNet(net_spec_for(info), 5)
    var opt = Adam(net.param_count(), 1e-3)
    _ = write_last(work + "/run", net, opt, info, TrainState())
    var e = export_synth_run(work + "/run", "last", "m4", work + "/root", "f16", False)
    expect(e.sha256 == a.sha256, "last/ exports the same weights (same seed)")
    var raised = False
    try:
        _ = export_synth_run(work + "/run", "middle", "m5", work + "/root", "f16", False)
    except e2:
        raised = String(e2).startswith("QM-E640")
    expect(raised, "--which middle is a usage error")
    raised = False
    try:
        _ = export_synth_run(work + "/nope", "best", "m5", work + "/root", "f16", False)
    except e3:
        raised = True
    expect(raised, "a missing run directory raises")
    print("  export determinism, default column, f32 export OK")


def test_export_overflow_leaves_root_untouched() raises:
    var work = String(WORK) + "/ovf"
    remove_tree(work)
    var common: List[Int] = [8]
    var info = make_run(work + "/run", "ovf", MODEL_KIND_PERM, "line:3", 8, common, 2)
    # poison one weight beyond the f16 range, keeping the checkpoint valid as f32
    var net = BasicNet.from_model(read_checkpoint(work + "/run/best"))
    net.ps.data[3] = 1.0e6
    _ = write_best(work + "/run", net, info, TrainState())
    var raised = False
    try:
        _ = export_synth_run(work + "/run", "best", "big", work + "/root", "f16", False)
    except e:
        raised = String(e).startswith("QM-E645")
    expect(raised, "f16 overflow must raise QM-E645")
    expect(not isdir(work + "/root/big") and not isdir(work + "/root/big.tmp"), "failed export left a directory")
    expect(not isfile(work + "/root/INDEX.json"), "failed export touched INDEX.json")
    # the same checkpoint exports fine as f32
    _ = export_synth_run(work + "/run", "best", "big", work + "/root", "f32", False)
    expect(isfile(work + "/root/INDEX.json"), "f32 export wrote the index")
    print("  f16 overflow aborts with QM-E645 and writes nothing OK")


def test_independent_checks_abort() raises:
    var work = String(WORK) + "/indep"
    remove_tree(work)
    var common: List[Int] = [10]
    var info = new_run_info("ic", MODEL_KIND_PERM, "line:3")
    info.emb = 8
    info.common = common.copy()
    var spec = net_spec_for(info)
    var net = BasicNet(spec, 11)
    var tensors = net_tensors(net)
    var sym = _symmetries_for(info.kind, info.num_qubits, info.actions, 0)
    var head = checkpoint_head(info, net)
    var obs_size = spec.obs_size
    # honest: written f16, view = pre-write tensors rounded to f16
    _ = write_model_dir(work + "/good", head, tensors, "f16", List[String](), "")
    var good = load_model_file(work + "/good")
    var view = _view_from_tensors(tensors, "f16", sym.obs_perms, sym.act_perms, obs_size, info.actions)
    var obs = check_observations(good, 6, 1)
    check_independent_forward(view, good, obs)
    # the naive Float64 forward agrees with MLPPolicy on this model too
    var pol = MLPPolicy.from_model(good)
    var lg = List[Float32]()
    var v = pol.forward(obs[0], lg)
    var l64 = List[Float64]()
    var v64 = naive_forward_f64(good, obs[0], -1, l64)
    for a in range(len(lg)):
        expect(abs(Float64(lg[a]) - l64[a]) < 1e-5, "naive forward logits")
    expect(abs(Float64(v) - v64) < 1e-5, "naive forward value")
    # deliberately transposed: the embedding matrix is written in the reference [E, OBS] layout but declared [OBS, E]
    var bad = List[TensorOut]()
    for i in range(len(tensors)):
        var t = tensors[i].copy()
        if t.name == "embeddings.weight":
            var E = spec.emb
            var swapped = List[Float32](length=len(t.data), fill=0.0)
            for r in range(obs_size):
                for c in range(E):
                    swapped[c * obs_size + r] = t.data[r * E + c]
            t.data = swapped^
        bad.append(t^)
    _ = write_model_dir(work + "/bad", head, bad, "f16", List[String](), "")
    var written_bad = load_model_file(work + "/bad")
    var raised = False
    var msg = String()
    try:
        check_independent_forward(view, written_bad, obs)
    except e:
        msg = String(e)
        raised = msg.startswith("QM-E644")
    expect(raised, "a transposed embedding tensor must abort with QM-E644: " + msg)
    # two layers swapped in the written model (tensor order bug): also caught
    var swapped_names = List[TensorOut]()
    for i in range(len(tensors)):
        var t = tensors[i].copy()
        if t.name == "action.0.weight":
            var A = len(info.actions) // 3
            var H = 10
            var twist = List[Float32](length=len(t.data), fill=0.0)
            for r in range(A):
                for c in range(H):
                    twist[((r + 1) % A) * H + c] = t.data[r * H + c]
            t.data = twist^
        swapped_names.append(t^)
    _ = write_model_dir(work + "/bad2", head, swapped_names, "f16", List[String](), "")
    raised = False
    try:
        check_independent_forward(view, load_model_file(work + "/bad2"), obs)
    except e2:
        raised = String(e2).startswith("QM-E644")
    expect(raised, "rotated action rows must abort with QM-E644")
    # wrong symmetry tables in the written model (an act_perm direction bug) are caught on a non-involutive graph
    var ring = new_run_info("rg", MODEL_KIND_PERM, "ring:4")
    ring.emb = 8
    ring.common = common.copy()
    var rspec = net_spec_for(ring)
    var rnet = BasicNet(rspec, 4)
    var rten = net_tensors(rnet)
    var rsym = _symmetries_for(ring.kind, ring.num_qubits, ring.actions, 0)
    var inv_act = List[List[Int]]()
    for p in range(len(rsym.act_perms)):
        var ap = rsym.act_perms[p].copy()
        var inv = List[Int](length=len(ap), fill=0)
        for a in range(len(ap)):
            inv[ap[a]] = a
        inv_act.append(inv^)
    # the wrong head: same obs_perms, inverted act_perms
    var rhead = checkpoint_head(ring, rnet)
    var cut = rhead.find('"symmetries":')
    var cut2 = rhead.find(',"license"')
    var wh = String(rhead[byte=0:cut]) + '"symmetries":{"obs_perms":' + perms_json(rsym.obs_perms) + ',"act_perms":' + perms_json(inv_act) + "}" + String(rhead[byte=cut2:])
    _ = write_model_dir(work + "/bad3", wh, rten, "f16", List[String](), "")
    var rview = _view_from_tensors(rten, "f16", rsym.obs_perms, rsym.act_perms, rspec.obs_size, ring.actions)
    var robs = check_observations(load_model_file(work + "/bad3"), 6, 1)
    var inv_differs = False
    for p in range(len(inv_act)):
        for a in range(len(inv_act[p])):
            if inv_act[p][a] != rsym.act_perms[p][a]:
                inv_differs = True
    expect(inv_differs, "ring:4 must have a non-involutive automorphism")
    raised = False
    try:
        check_independent_forward(rview, load_model_file(work + "/bad3"), robs)
    except e3:
        raised = String(e3).startswith("QM-E644")
    expect(raised, "inverted act_perms must abort with QM-E644 on ring:4")
    print("  independent pre-write checks: honest model passes, transposed / rotated / inverted ones abort OK")


# ------------------------------------------------------------------------------------------------ Pauli ensemble


def actions_flat(v: JsonValue) raises -> List[Int]:
    var out = List[Int]()
    for i in range(len(v)):
        out.append(gate_code(v.at(i).at(0).as_string()))
        out.append(v.at(i).at(1).as_int())
        out.append(v.at(i).at(2).as_int())
    return out^


def pauli_shell(c: JsonValue, act_override: Optional[List[List[Int]]]) raises -> ModelFile:
    var m = ModelFile()
    m.dir = String("<pauli fixture>")
    m.kind_code = MODEL_KIND_PAULI
    m.kind = String("pauli_network")
    m.num_qubits = c.get_int("num_qubits", 0)
    m.actions = actions_flat(c.get("actions"))
    m.max_rotations = c.get_int("max_rotations", 5)
    m.obs_size = c.get_int("obs_size", 0)
    m.obs_perms = ints2(c.get("obs_perms"))
    m.act_perms = ints2(c.get("act_perms"))
    if act_override:
        m.act_perms = act_override.value().copy()
    return m^


def test_pauli_ensemble() raises:
    var fx = load_fixture("envs/pauli_twists.json")
    var cases = fx.get("cases")
    var failed_mutations = 0
    var expected_failures = 0
    for i in range(len(cases)):
        var c = cases.at(i)
        var m = pauli_shell(c, None)
        check_pauli_ensemble(m, UInt64(77 + i))  # the Python-dumped export tables agree with TrainEnv
        # the same check on the tables our own symmetry module produces
        var mine = _symmetries_for(MODEL_KIND_PAULI, m.num_qubits, m.actions, m.max_rotations)
        var m2 = pauli_shell(c, None)
        m2.obs_perms = mine.obs_perms.copy()
        m2.act_perms = mine.act_perms.copy()
        check_pauli_ensemble(m2, UInt64(5 + i))
        # mutation: the direct (env-side) action tables instead of their inverses
        var direct = pauli_qubit_perms_and_act_perms(m.num_qubits, m.actions)
        var non_involutive = False
        for p in range(len(direct.act_perms)):
            var ap = direct.act_perms[p].copy()
            for a in range(len(ap)):
                if ap[ap[a]] != a:
                    non_involutive = True
        var wrong = pauli_shell(c, Optional[List[List[Int]]](direct.act_perms.copy()))
        var raised = False
        try:
            check_pauli_ensemble(wrong, UInt64(5 + i))
        except e:
            raised = String(e).startswith("QM-E644")
        expect(raised == non_involutive, c.get_string("topology", "") + ": the direct-table mutation must fail exactly on non-involutive graphs")
        if non_involutive:
            expected_failures += 1
        if raised:
            failed_mutations += 1
        # mutation: identity observation maps (obs_perms replaced by the identity) must fail wherever a symmetry moves obs
        var ident = pauli_shell(c, None)
        for p in range(len(ident.obs_perms)):
            for q in range(len(ident.obs_perms[p])):
                ident.obs_perms[p][q] = q
        var raised2 = False
        try:
            check_pauli_ensemble(ident, UInt64(5 + i))
        except e:
            raised2 = String(e).startswith("QM-E644")
        expect(raised2, c.get_string("topology", "") + ": identity obs maps must fail the ensemble check")
    expect(expected_failures == 4 and failed_mutations == 4, "ring:5, ring:4, star:4 and H7 are the non-involutive topologies")
    # a plain-kind model is a usage error
    var plain = ModelFile()
    plain.kind_code = MODEL_KIND_PERM
    plain.kind = String("permutation")
    var raised3 = False
    try:
        check_pauli_ensemble(plain, 1)
    except e:
        raised3 = String(e).startswith("QM-E640")
    expect(raised3, "check_pauli_ensemble on a non-Pauli model")
    print("  check_pauli_ensemble on 6 topologies, direction and identity mutations OK")


# ------------------------------------------------------------------------------------------------ safetensors


def build_safetensors(header: String, data: List[UInt8]) -> List[UInt8]:
    var out = List[UInt8]()
    var n = header.byte_length()
    for k in range(8):
        out.append(UInt8((n >> (8 * k)) & 0xFF))
    for b in header.as_bytes():
        out.append(b)
    for b in data:
        out.append(b)
    return out^


def test_safetensors() raises:
    # the real file written by the reference stack
    var sts = safetensors_import(String("tests/fixtures/ppo/tiny_policy.safetensors"))
    expect(len(sts) == 8, "tiny_policy.safetensors has 8 tensors")
    var fw = load_fixture("ppo/policy_forward.json")
    var weights = fw.get("weights")
    var seen = 0
    for i in range(len(sts)):
        var name = sts[i].name
        expect(weights.has(name), "unexpected tensor " + name)
        var want = floats32(weights.get(name).get("data"))
        var wshape = ints(weights.get(name).get("shape"))
        expect(len(wshape) == len(sts[i].shape), name + " rank")
        for d in range(len(wshape)):
            expect(wshape[d] == sts[i].shape[d], name + " shape")
        assert_list_close32(sts[i].data, want, 0.0, 0.0, name)  # exact
        seen += 1
    expect(seen == 8, "all tensors compared")
    # storage order is increasing data offset (the header lists names alphabetically)
    expect(sts[0].name == "action.0.bias" and sts[len(sts) - 1].name == "value.0.weight", "storage order")
    # container tensors: order, transposition, 1-D biases
    var ct = basic_mlp_tensors(sts)
    expect(len(ct) == 8, "container tensor count")
    var want_names: List[String] = ["embeddings.weight", "embeddings.bias", "common.0.weight", "common.0.bias", "action.0.weight", "action.0.bias", "value.0.weight", "value.0.bias"]
    for i in range(8):
        expect(ct[i].name == want_names[i], "container order " + ct[i].name)
    expect(ct[0].shape[0] == 9 and ct[0].shape[1] == 8, "embeddings.weight is transposed to [OBS, E]")
    expect(len(ct[1].shape) == 1 and ct[1].shape[0] == 8, "embeddings.bias is 1-D")
    var torch_w = floats32(weights.get("embeddings.weight").get("data"))  # [E=8, OBS=9]
    for r in range(8):
        for c in range(9):
            expect(f32_bits(ct[0].data[c * 8 + r]) == f32_bits(torch_w[r * 9 + c]), "transposition element")
    # the imported network reproduces the torch logits (binds importer + transposition + BasicNet layout)
    var m = ModelFile()
    m.dir = String("<tiny>")
    m.obs_size = 9
    for i in range(len(ct)):
        m.names.append(ct[i].name)
        m.shapes.append(ct[i].shape.copy())
        m.tensors.append(ct[i].data.copy())
    var net = BasicNet.from_model(m)
    var sc = ActorScratch(net.spec)
    var cases = fw.get("cases")
    var checked = 0
    for ci in range(len(cases)):
        var c = cases.at(ci)
        if not c.get("twist").is_null():
            continue
        var obs = ints(c.get("obs"))
        var idx = List[Int32]()
        for q in range(len(obs)):
            idx.append(Int32(obs[q]))
        var lg = List[Float32](length=4, fill=0.0)
        var v = net.forward_one(Span(idx), -1, sc, lg.unsafe_ptr().as_unsafe_any_origin())
        _ = len(lg)
        var want = floats32(c.get("raw_logits"))
        assert_list_close32(lg, want, 1e-5, 1e-5, "imported net logits")
        expect(abs(Float64(v) - c.get_float("value", 0.0)) < 1e-5, "imported net value")
        checked += 1
    expect(checked >= 5, "twist-free cases checked")

    # synthetic files: F16, BF16, F64 decode; a metadata member is skipped
    var h = String(
        '{"__metadata__":{"format":"pt"},"a":{"dtype":"F16","shape":[2],"data_offsets":[0,4]},'
        '"b":{"dtype":"BF16","shape":[2],"data_offsets":[4,8]},"c":{"dtype":"F64","shape":[1],"data_offsets":[8,16]}}   '
    )
    var d = List[UInt8]()
    # f16 1.0 = 0x3C00, f16 -2.5 = 0xC100 ; bf16 1.5 = 0x3FC0, bf16 -0.5 = 0xBF00 ; f64 0.1
    var raw: List[Int] = [0x00, 0x3C, 0x00, 0xC1, 0xC0, 0x3F, 0x00, 0xBF, 0x9A, 0x99, 0x99, 0x99, 0x99, 0x99, 0xB9, 0x3F]
    for x in raw:
        d.append(UInt8(x))
    var parsed = safetensors_parse(build_safetensors(h, d))
    expect(len(parsed) == 3, "three synthetic tensors")
    expect(parsed[0].name == "a" and parsed[0].data[0] == 1.0 and parsed[0].data[1] == -2.5, "F16 decode")
    expect(parsed[1].name == "b" and parsed[1].data[0] == 1.5 and parsed[1].data[1] == -0.5, "BF16 decode")
    expect(parsed[2].name == "c" and f32_bits(parsed[2].data[0]) == f32_bits(Float32(0.1)), "F64 decode")
    # malformed files raise QM-E647
    var bad_cases = List[List[UInt8]]()
    bad_cases.append(List[UInt8]())  # empty
    var huge = List[UInt8]()
    for _ in range(8):
        huge.append(UInt8(0xFF))
    bad_cases.append(huge^)  # header length beyond the file
    bad_cases.append(build_safetensors(String('{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}'), List[UInt8]()))  # data missing
    bad_cases.append(build_safetensors(String('{"a":{"dtype":"F32","shape":[3],"data_offsets":[0,8]}}'), d.copy()))  # size / shape
    bad_cases.append(build_safetensors(String('{"a":{"dtype":"I64","shape":[1],"data_offsets":[0,8]}}'), d.copy()))  # dtype
    bad_cases.append(build_safetensors(String("not json"), d.copy()))
    bad_cases.append(build_safetensors(String('{"a":{"dtype":"F32","shape":[1],"data_offsets":[8,4]}}'), d.copy()))  # reversed
    for i in range(len(bad_cases)):
        var raised = False
        try:
            _ = safetensors_parse(bad_cases[i])
        except e:
            raised = String(e).startswith("QM-E647")
        expect(raised, "malformed safetensors case " + String(i) + " must raise QM-E647")
    var raised = False
    try:
        _ = safetensors_import(String("tests/fixtures/ppo/does_not_exist.safetensors"))
    except e:
        raised = String(e).startswith("QM-E647")
    expect(raised, "missing safetensors file")
    # a state dict without a value head part raises
    var partial = List[SafeTensor]()
    partial.append(sts[0].copy())
    raised = False
    try:
        _ = basic_mlp_tensors(partial)
    except e:
        raised = String(e).startswith("QM-E647")
    expect(raised, "basic_mlp_tensors without embeddings")
    print("  safetensors importer OK")


# ------------------------------------------------------------------------------------------------ fixtures generator


def test_generate_fixtures() raises:
    var out = String(WORK) + "/fx"
    remove_tree(out)
    var ids = generate_policy_fixtures(out)
    expect(len(ids) == 5, "five fixture models")
    for i in range(len(ids)):
        var worst = verify_exported_model(out + "/synth/" + ids[i])
        expect(worst <= 1e-5, "fixture " + ids[i] + " replay")
    var rep = verify_models(out + "/synth")
    expect(rep.n_problems == 0 and rep.n_files == 5, "verify_models on generated fixtures")
    var raised = False
    try:
        _ = load_model_file(out + "/bad_checksum")
    except e:
        raised = String(e).startswith("QM-E617")
    expect(raised, "bad_checksum copy must fail the checksum")
    raised = False
    try:
        _ = load_model_file(out + "/bad_format")
    except e:
        raised = String(e).startswith("QM-E617")
    expect(raised, "bad_format copy must fail the format check")
    # a second run reproduces the same bytes (deterministic generator)
    var a = read_text(out + "/synth/fx_pauli_line3/manifest.json")
    _ = generate_policy_fixtures(out)
    expect(read_text(out + "/synth/fx_pauli_line3/manifest.json") == a, "fixture generator is not deterministic")
    # (QUMO's committed copies under tests/fixtures/ai/policy: rl/qumo_bridge/tests/test_bridge_policy.mojo)
    print("  random-weight fixture generator OK")


def main() raises:
    makedirs(String(WORK), exist_ok=True)
    test_checkpoint_resume()
    test_run_files()
    test_resolve_init()
    test_export_matches_shipped()
    test_naive_forward_vs_shipped_vectors()
    test_export_determinism_and_default()
    test_export_overflow_leaves_root_untouched()
    test_independent_checks_abort()
    test_pauli_ensemble()
    test_safetensors()
    test_generate_fixtures()
    print("test_rl_export OK")
