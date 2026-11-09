"""Bridge test of the model formats: models qrl exports load in QUMO's runtime and give identical logits.

qrl owns `qumo.rlsynth.v0` and carries its own reader (`qrl.model`); QUMO reads the same containers with its own copy
(`qumo.compiler.policy`, `qumo.compiler.policy_synth`). For three sets of synthesis models

  * the five random-weight fixtures of `qrl.export_synth.generate_policy_fixtures` (every kind, f32 and f16),
  * three models exported by `export_synth_run` from training run directories (random BasicNet weights, one per
    plain kind; f16), and
  * every model QUMO ships (models/ai_transpiler/synth),

this checks that both loaders decode the same manifest fields and the same tensors bit for bit, that both
`MLPPolicy` forwards give bit-identical logits and values (identity frame, every symmetry, the ensemble, and the
batched path) on the embedded test vectors and on random observations, that the embedded vectors and greedy
rollouts replay through QUMO exactly as `tests/test_policy.mojo` replays them, and that QUMO's greedy trace equals
qrl's. It also checks that QUMO's committed fixtures (tests/fixtures/ai/policy) have the structure the generator
writes. Run from the QUMO root with `-I . -I rl` (rl/Makefile `test-bridge`).
"""

from std.os import makedirs
from std.os.path import exists
from std.memory import bitcast
from qrl.json import JsonValue, parse_json
from qrl.rng import Rng
from qrl.model import (
    load_model_file as qrl_load,
    read_model_index,
    MLPPolicy as QrlPolicy,
    MODEL_KIND_LINEAR,
    MODEL_KIND_PAULI,
    MODEL_KIND_PERM,
    MODEL_KIND_CLIFFORD,
)
from qrl.env_core import PolicyEnv as QrlEnv, greedy_trace as qrl_greedy
from qrl.export_synth import generate_policy_fixtures, export_synth_run
from qrl.checkpoint import TrainState, new_run_info, net_spec_for, write_run_json, config_hash_of, write_best
from qrl.container import remove_tree
from qrl.nn import BasicNet
from qrl.testutil import model_root, read_text
from qumo.compiler.policy import load_model_file as qumo_load, MLPPolicy as QumoPolicy
from qumo.compiler.policy_synth import PolicyEnv as QumoEnv, greedy_trace as qumo_greedy

comptime WORK = "build/test_bridge_policy"
comptime QUMO_FX = "tests/fixtures/ai/policy"


def check(cond: Bool, msg: String) raises:
    if not cond:
        raise Error("test_bridge_policy: " + msg)


def _ints(v: JsonValue) raises -> List[Int]:
    var out = List[Int]()
    for i in range(len(v)):
        out.append(v.at(i).as_int())
    return out^


def _same_f32(a: List[Float32], b: List[Float32]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            return False
    return True


def _same_ints(a: List[Int], b: List[Int]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def _random_obs(obs_size: Int, mut rng: Rng) -> List[Int]:
    """A sorted sparse observation (distinct indices, as the environments produce)."""
    var out = List[Int]()
    for i in range(obs_size):
        if rng.below(4) == 0:
            out.append(i)
    return out^


def check_model(dir: String) raises -> Int:
    """Every check above for one model directory; returns the number of logit vectors compared."""
    var mq = qrl_load(dir)
    var mu = qumo_load(dir)
    check(mq.id == mu.id and mq.kind == mu.kind and mq.kind_code == mu.kind_code and mq.format == mu.format, dir + ": identity")
    check(mq.num_qubits == mu.num_qubits and mq.obs_size == mu.obs_size and mq.max_depth == mu.max_depth, dir + ": sizes")
    check(mq.max_rotations == mu.max_rotations and mq.obs_layout == mu.obs_layout, dir + ": obs")
    check(_same_ints(mq.actions, mu.actions) and _same_ints(mq.edges, mu.edges), dir + ": actions / edges")
    check(len(mq.obs_perms) == len(mu.obs_perms) and len(mq.act_perms) == len(mu.act_perms), dir + ": symmetry count")
    for p in range(len(mq.obs_perms)):
        check(_same_ints(mq.obs_perms[p], mu.obs_perms[p]) and _same_ints(mq.act_perms[p], mu.act_perms[p]), dir + ": symmetry " + String(p))
    check(len(mq.names) == len(mu.names), dir + ": tensor count")
    for t in range(len(mq.names)):
        check(mq.names[t] == mu.names[t] and _same_ints(mq.shapes[t], mu.shapes[t]), dir + ": tensor " + mq.names[t])
        check(_same_f32(mq.tensors[t], mu.tensors[t]), dir + ": decoded tensor " + mq.names[t] + " differs")
    check(mq.manifest == mu.manifest, dir + ": manifest text")

    var pq = QrlPolicy.from_model(mq)
    var pu = QumoPolicy.from_model(mu)
    check(pq.num_actions == pu.num_actions and pq.num_symmetries() == pu.num_symmetries(), dir + ": policy shape")
    var man = parse_json(mq.manifest)
    var obs_list = List[List[Int]]()
    var tv = man.get("test_vectors")
    for t in range(len(tv)):
        obs_list.append(_ints(tv.at(t).get("obs")))
    var rng = Rng(UInt64(dir.byte_length()) * 7919 + 3)
    for _ in range(12):
        obs_list.append(_random_obs(mq.obs_size, rng))
    var compared = 0
    for o in range(len(obs_list)):
        var lq = List[Float32]()
        var lu = List[Float32]()
        var vq = pq.forward(obs_list[o], lq)
        var vu = pu.forward(obs_list[o], lu)
        check(_same_f32(lq, lu) and bitcast[DType.uint32](vq) == bitcast[DType.uint32](vu), dir + ": identity forward, obs " + String(o))
        var eq = List[Float32]()
        var eu = List[Float32]()
        vq = pq.forward_ensemble(obs_list[o], eq)
        vu = pu.forward_ensemble(obs_list[o], eu)
        check(_same_f32(eq, eu) and bitcast[DType.uint32](vq) == bitcast[DType.uint32](vu), dir + ": ensemble forward, obs " + String(o))
        compared += 2
        for p in range(pq.num_symmetries()):
            var sq = List[Float32]()
            var su = List[Float32]()
            _ = pq.forward_sym(obs_list[o], p, sq)
            _ = pu.forward_sym(obs_list[o], p, su)
            check(_same_f32(sq, su), dir + ": symmetry " + String(p) + " forward, obs " + String(o))
            compared += 1
        # the embedded vectors replay through QUMO within the tolerance of tests/test_policy.mojo
        if o < len(tv):
            var want = tv.at(o).get("logits")
            for a in range(len(lu)):
                check(abs(Float64(lu[a]) - want.at(a).as_float()) <= 1e-5, dir + ": QUMO vs embedded logits")
            var wante = tv.at(o).get("logits_ensemble")
            for a in range(len(eu)):
                check(abs(Float64(eu[a]) - wante.at(a).as_float()) <= 1e-5, dir + ": QUMO vs embedded ensemble logits")
    # the batched identity path
    var perms = List[Int](length=len(obs_list), fill=-1)
    var bq = List[Float32]()
    var bu = List[Float32]()
    var bvq = List[Float32]()
    var bvu = List[Float32]()
    pq.raw_batch(obs_list, perms, bq, bvq)
    pu.raw_batch(obs_list, perms, bu, bvu)
    check(_same_f32(bq, bu) and _same_f32(bvq, bvu), dir + ": raw_batch")
    compared += len(obs_list)

    # rollouts: QUMO's greedy trace reproduces the embedded actions, and equals qrl's
    var ro = man.get("rollouts")
    for r in range(len(ro)):
        var ij = ro.at(r).get("input")
        var st = List[UInt64]()
        for j in range(len(ij)):
            st.append(UInt64(ij.at(j).as_int()))
        var eu = QumoEnv(mu.kind_code, mu.num_qubits, mu)
        var eq = QrlEnv(mq.kind_code, mq.num_qubits, mq)
        if mq.kind_code == MODEL_KIND_PAULI:
            eu.init_pauli(ro.at(r).get("R").as_int(), st)
            eq.init_pauli(ro.at(r).get("R").as_int(), st)
        var oku = False
        var okq = False
        var au = qumo_greedy(eu, pu, mu.max_depth, st, oku)
        var aq = qrl_greedy(eq, pq, mq.max_depth, st, okq)
        check(_same_ints(au, aq) and oku == okq, dir + ": greedy trace of rollout " + String(r) + " differs between QUMO and qrl")
        check(_same_ints(au, _ints(ro.at(r).get("actions"))), dir + ": QUMO greedy trace vs embedded rollout " + String(r))
        if ro.at(r).has("solved"):
            check(oku == ro.at(r).get("solved").as_bool(), dir + ": rollout solved flag")
    return compared


def make_run(run_dir: String, id: String, kind: Int, topo: String, seed: Int) raises:
    var info = new_run_info(id, kind, topo)
    info.emb = 24
    var common: List[Int] = [16]
    info.common = common.copy()
    info.seed = seed
    var net = BasicNet(net_spec_for(info), UInt64(seed))
    write_run_json(run_dir, info)
    var st = TrainState()
    st.it = 3
    st.difficulty = 2
    st.seed = seed
    st.config_hash = config_hash_of(info)
    _ = write_best(run_dir, net, info, st)


def test_generated_fixtures() raises:
    var out = String(WORK) + "/fx"
    remove_tree(out)
    var ids = generate_policy_fixtures(out)
    var n = 0
    for id in ids:
        n += check_model(out + "/synth/" + id)
    # QUMO's committed fixtures have the structure the generator writes (kinds, actions, symmetries, sizes)
    for id in ids:
        var m_old = parse_json(read_text(String(QUMO_FX) + "/synth/" + id + "/manifest.json"))
        var m_new = parse_json(read_text(out + "/synth/" + id + "/manifest.json"))
        for k in ["kind", "num_qubits", "edges", "actions", "symmetries", "max_depth", "arch"]:
            check(m_old.get(k).to_json() == m_new.get(k).to_json(), id + ": member " + String(k) + " differs from the committed fixture")
        n += check_model(String(QUMO_FX) + "/synth/" + id)
    print("  generated + committed fixtures: " + String(2 * len(ids)) + " models, " + String(n) + " logit vectors identical")


def test_exported_runs() raises:
    var root = String(WORK) + "/exported"
    remove_tree(root)
    var kinds: List[Int] = [MODEL_KIND_PERM, MODEL_KIND_LINEAR, MODEL_KIND_CLIFFORD]
    var topos: List[String] = ["ring:4", "line:4", "line:3"]
    var names: List[String] = ["x_perm_ring4", "x_lf_line4", "x_cliff_line3"]
    var n = 0
    for i in range(3):
        var run = String(WORK) + "/run_" + names[i]
        remove_tree(run)
        make_run(run, names[i], kinds[i], topos[i], 11 + i)
        _ = export_synth_run(run, "best", names[i], root, "f16", i == 0)
        n += check_model(root + "/" + names[i])
    print("  export_synth_run: 3 models, " + String(n) + " logit vectors identical in QUMO")


def test_shipped_models() raises:
    var rows = read_model_index(model_root(), "synth")
    check(len(rows) > 0, "no shipped synthesis model under " + model_root())
    var n = 0
    for r in rows:
        n += check_model(r.path)
    print("  shipped models: " + String(len(rows)) + " models, " + String(n) + " logit vectors identical")


def main() raises:
    makedirs(String(WORK), exist_ok=True)
    test_generated_fixtures()
    test_exported_runs()
    test_shipped_models()
    print("test_bridge_policy OK")
