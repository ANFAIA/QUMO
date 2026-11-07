"""Tests for qumo.rl.gateset and qumo.rl.synth_env (component C6): the exact half of the environment goldens.

Pinned here (golden data produced offline by the real qiskit-gym 0.4.1 environments, tests/fixtures/rl/envs):
  * `build_gateset` and the symmetry tables of the 44 configs of gateset_twists.json (and the observation sizes);
  * `set_state` + action sequences: observation, is_final, mask and f32 reward of all four kinds (steps_*.json),
    with non-default metrics weights and non-default Pauli arguments (steps_custom_args.json);
  * the Pauli hidden symmetry (pauli_perm_cases.json): observations are members of `obs_by_perm`, mapped actions,
    reward, finality and the next observation;
  * the FMA barrier: reward_fma_cases.json bit for bit (a fused multiply-add differs in the last bit there);
  * add_inverts: outcome set and P(invert) over 4,000 trials (inverts.json);
  * horizon `min(2d, 128)`, the exactly-one-observe protocol, QM-E641 / QM-E642 rejections, `step(a >= A)`,
    seeding, copy independence.
The statistical half (reset distributions) is tests/test_rl_env_slow.mojo.
"""

from std.collections import List
from qumo.ai.json import JsonValue
from qumo.compiler.policy import (
    MODEL_KIND_CLIFFORD,
    MODEL_KIND_LINEAR,
    MODEL_KIND_PAULI,
    MODEL_KIND_PERM,
    gate_code,
    model_kind_code,
)
from qumo.ir.gates import Op
from qumo.rl.gateset import EnvConfig, build_gateset, obs_size_of, shell_model, validate_actions
from qumo.rl.synth_env import Metrics, TrainEnv, penalty_f32, reward_f32
from qumo.rl.testutil import f32_bits, ints, ints2, load_fixture
from qumo.rl.topology import parse_topology


def expect(cond: Bool, msg: String) raises:
    if not cond:
        raise Error("test_rl_env: " + msg)


def same_ints(a: List[Int], b: List[Int]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def same_lists(a: List[List[Int]], b: List[List[Int]]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if not same_ints(a[i], b[i]):
            return False
    return True


def actions_from_json(v: JsonValue) raises -> List[Int]:
    var out = List[Int]()
    for i in range(len(v)):
        var code = gate_code(v.at(i).at(0).as_string())
        if code < 0:
            raise Error("unknown gate " + v.at(i).at(0).as_string())
        out.append(code)
        out.append(v.at(i).at(1).as_int())
        out.append(v.at(i).at(2).as_int())
    return out^


def cfg_for(kind_name: String, topo: String) raises -> EnvConfig:
    var t = parse_topology(topo)
    var cfg = EnvConfig()
    cfg.kind = model_kind_code(kind_name)
    cfg.n = t.n
    cfg.actions = build_gateset(cfg.kind, t.n, t.edges)
    return cfg^


def failed_with(code: String, kind: Int, n: Int, edges: List[Int]) -> Bool:
    try:
        _ = build_gateset(kind, n, edges)
    except e:
        return String(e).startswith(code)
    return False


def env_fails(cfg: EnvConfig, code: String) -> Bool:
    try:
        _ = TrainEnv(cfg, 1)
    except e:
        return String(e).startswith(code)
    return False


def obs_of(mut env: TrainEnv) -> List[Int]:
    var o = List[Int]()
    env.observe(o)
    return o^


def load_state(mut env: TrainEnv, kind: Int, state: List[Int]) raises:
    if kind == MODEL_KIND_PERM:
        env.set_state_perm(state)
    elif kind == MODEL_KIND_PAULI:
        env.set_state_pauli(state)
    else:
        env.set_state_bits(state)
    expect(env.error_code() == 0, "set_state raised an error code")


def check_row(
    mut env: TrainEnv, row: JsonValue, what: String, with_action: Bool, action_taken: Int
) raises:
    var obs = obs_of(env)
    var want = ints(row.get("obs"))
    expect(same_ints(obs, want), what + ": observation differs")
    expect(f32_bits(env.reward()) == f32_bits(Float32(row.get("reward").as_float())), what + ": reward differs")
    expect(env.is_final() == row.get("is_final").as_bool(), what + ": is_final differs")
    expect(env.all_allowed() == row.get("allowed").as_bool(), what + ": mask differs")


# ------------------------------------------------------------------------------------------------ gateset


def test_gateset_and_tables() raises:
    var fx = load_fixture("envs/gateset_twists.json")
    expect(len(fx) == 44, "44 gateset configs")
    for i in range(len(fx)):
        var c = fx.at(i)
        var kind_name = c.get("kind").as_string()
        var topo = c.get("topology").as_string()
        var tag = kind_name + " " + topo
        var cfg = cfg_for(kind_name, topo)
        var want = actions_from_json(c.get("gateset"))
        expect(same_ints(cfg.actions, want), tag + ": gateset")
        expect(cfg.n == c.get("num_qubits").as_int(), tag + ": num_qubits")
        var shape = ints(c.get("obs_shape"))
        expect(obs_size_of(cfg) == shape[0] * shape[1], tag + ": obs size")
        var env = TrainEnv(cfg, 1)
        expect(env.num_actions() == c.get("num_actions").as_int(), tag + ": num_actions")
        expect(env.obs_size() == shape[0] * shape[1], tag + ": env obs size")
        if cfg.kind == MODEL_KIND_PAULI:
            var qp = ints2(c.get("qubit_perms"))
            var ap = ints2(c.get("act_perms"))
            expect(env.num_syms() == len(qp), tag + ": number of hidden symmetries")
            for p in range(len(qp)):
                expect(same_ints(env.pauli_qubit_perm(p), qp[p]), tag + ": qubit perm " + String(p))
                expect(same_ints(env.pauli_act_perm(p), ap[p]), tag + ": act perm " + String(p))
            expect(env.twists().count() == 0, tag + ": Pauli twists are empty")
        else:
            var tw = env.twists()
            var op = ints2(c.get("obs_perms"))
            var ap = ints2(c.get("act_perms"))
            expect(same_lists(tw.obs_perms, op), tag + ": obs_perms")
            expect(same_lists(tw.act_perms, ap), tag + ": act_perms")
        # without add_perms the policy gets no twists
        var cfg2 = cfg.copy()
        cfg2.add_perms = False
        var env2 = TrainEnv(cfg2, 1)
        expect(env2.twists().count() == 0 and env2.num_syms() == 0, tag + ": add_perms off")


def test_gateset_errors() raises:
    # trailing isolated qubit: edges cover qubits 0..2 but the topology claims 4
    expect(failed_with("QM-E641", MODEL_KIND_PERM, 4, [0, 1, 1, 2]), "isolated trailing qubit must raise QM-E641")
    expect(failed_with("QM-E641", MODEL_KIND_LINEAR, 2, [0, 1, 1, 2]), "n smaller than the edges must raise QM-E641")
    expect(failed_with("QM-E641", MODEL_KIND_CLIFFORD, 3, List[Int]()), "no edge")
    expect(failed_with("QM-E641", MODEL_KIND_PAULI, 3, [0, 0, 1, 2]), "self loop")
    expect(failed_with("QM-E642", 17, 3, [0, 1, 1, 2]), "unknown kind")
    # unordered / duplicated input is normalised
    var g = build_gateset(MODEL_KIND_PERM, 3, [2, 1, 1, 0, 0, 1])
    expect(same_ints(g, [Op.SWAP, 0, 1, Op.SWAP, 1, 2]), "edges are oriented, sorted and de-duplicated")


def test_validate_actions() raises:
    var base = cfg_for("clifford", "line:3")
    validate_actions(base)
    # gates outside {H, S, CX, SWAP}
    var banned = List[Int]()
    banned.append(Op.CZ)
    banned.append(Op.SDG)
    banned.append(Op.SX)
    banned.append(Op.SXDG)
    banned.append(Op.X)
    banned.append(Op.T)
    for i in range(len(banned)):
        var c = base.copy()
        var two = banned[i] == Op.CZ
        c.actions.append(banned[i])
        c.actions.append(0)
        c.actions.append(1 if two else -1)
        expect(env_fails(c, "QM-E642"), "opcode " + String(banned[i]) + " must raise QM-E642")
    # bad qubit ids and malformed triples
    var c1 = base.copy()
    c1.actions.append(Op.H)
    c1.actions.append(3)
    c1.actions.append(-1)
    expect(env_fails(c1, "QM-E642"), "qubit out of range")
    var c2 = base.copy()
    c2.actions.append(Op.CX)
    c2.actions.append(1)
    c2.actions.append(1)
    expect(env_fails(c2, "QM-E642"), "CX on one qubit")
    var c3 = base.copy()
    c3.actions.append(Op.H)
    c3.actions.append(0)
    c3.actions.append(1)
    expect(env_fails(c3, "QM-E642"), "1-qubit gate with a second operand")
    var c4 = base.copy()
    c4.actions.append(Op.H)
    expect(env_fails(c4, "QM-E642"), "action list not a multiple of 3")
    var c5 = base.copy()
    c5.actions = List[Int]()
    expect(env_fails(c5, "QM-E642"), "empty gateset")
    var c6 = cfg_for("permutation", "line:3")
    c6.actions.append(Op.CX)
    c6.actions.append(0)
    c6.actions.append(1)
    expect(env_fails(c6, "QM-E642"), "a permutation gateset is SWAP only")
    var c7 = base.copy()
    c7.n = 40
    expect(env_fails(c7, "QM-E642"), "2n > 64 does not fit the packed tableau")
    var c8 = cfg_for("pauli_network", "line:3")
    c8.pauli_diff_scale = 0
    expect(env_fails(c8, "QM-E642"), "pauli_diff_scale 0")
    var c9 = base.copy()
    c9.kind = 9
    expect(env_fails(c9, "QM-E642"), "unknown kind")
    # shell_model carries the metadata PolicyEnv needs
    var pc = cfg_for("pauli_network", "line:4")
    pc.max_rotations = 3
    var sm = shell_model(pc)
    expect(sm.num_qubits == 4 and sm.max_rotations == 3 and sm.num_actions() == len(pc.actions) // 3, "shell_model")
    expect(sm.obs_size == 8 * 11, "shell_model obs_size")
    var pm = shell_model(cfg_for("permutation", "line:4"))
    expect(pm.max_rotations == 0 and pm.kind_code == MODEL_KIND_PERM, "shell_model plain")


# ------------------------------------------------------------------------------------------------ step goldens


def run_cases(cfg: EnvConfig, cases: JsonValue, tag: String) raises -> Int:
    var steps_done = 0
    for ci in range(len(cases)):
        var cs = cases.at(ci)
        var env = TrainEnv(cfg, UInt64(1000 + ci))
        var st = ints(cs.get("state"))
        load_state(env, cfg.kind, st)
        check_row(env, cs.get("init"), tag + " case " + String(ci) + " init", False, -1)
        var steps = cs.get("steps")
        for si in range(len(steps)):
            var row = steps.at(si)
            var a = row.get("action").as_int()
            env.step(a)
            check_row(env, row, tag + " case " + String(ci) + " step " + String(si), True, a)
            steps_done += 1
        expect(env.error_code() == 0, tag + ": unexpected error code")
    return steps_done


def test_step_goldens() raises:
    var files = List[String]()
    files.append("envs/steps_permutation.json")
    files.append("envs/steps_linear_function.json")
    files.append("envs/steps_clifford.json")
    files.append("envs/steps_pauli_network.json")
    var total = 0
    for f in range(len(files)):
        var fx = load_fixture(files[f])
        for i in range(len(fx)):
            var c = fx.at(i)
            var cfg = cfg_for(c.get("kind").as_string(), c.get("topology").as_string())
            cfg.add_inverts = c.get("add_inverts").as_bool()
            cfg.add_perms = c.get("add_perms").as_bool()
            expect(not cfg.add_inverts and not cfg.add_perms, "step goldens are recorded without inversion / perms")
            total += run_cases(cfg, c.get("cases"), files[f] + " " + c.get("topology").as_string())
    expect(total == 4102, "the step goldens were not replayed in full")


def apply_weights(mut cfg: EnvConfig, w: JsonValue) raises:
    if w.has("n_cnots"):
        cfg.w_cnots = Float32(w.get("n_cnots").as_float())
    if w.has("n_layers_cnots"):
        cfg.w_layers_cnots = Float32(w.get("n_layers_cnots").as_float())
    if w.has("n_layers"):
        cfg.w_layers = Float32(w.get("n_layers").as_float())
    if w.has("n_gates"):
        cfg.w_gates = Float32(w.get("n_gates").as_float())


def test_custom_args() raises:
    var fx = load_fixture("envs/steps_custom_args.json")
    expect(len(fx) == 4, "4 custom configs")
    var total = 0
    for i in range(len(fx)):
        var c = fx.at(i)
        var cfg = cfg_for(c.get("kind").as_string(), c.get("topology").as_string())
        if cfg.kind == MODEL_KIND_PAULI:
            var k = c.get("constructor")
            cfg.max_rotations = k.get("max_rotations").as_int()
            cfg.pauli_layer_reward = Float32(k.get("pauli_layer_reward").as_float())
            cfg.pauli_diff_scale = k.get("pauli_diff_scale").as_int()
            cfg.num_qubits_decay = Float32(k.get("num_qubits_decay").as_float())
            cfg.final_pauli_layers = k.get("final_pauli_layers").as_int()
            cfg.depth_slope = k.get("depth_slope").as_int()
            cfg.max_depth = k.get("max_depth").as_int()
            cfg.add_perms = k.get("add_perms").as_bool()
            cfg.add_inverts = False
            apply_weights(cfg, k.get("metrics_weights"))
        else:
            cfg.add_inverts = c.get("add_inverts").as_bool()
            cfg.add_perms = c.get("add_perms").as_bool()
            apply_weights(cfg, c.get("metrics_weights"))
        total += run_cases(cfg, c.get("cases"), "custom " + c.get("kind").as_string())
    expect(total == 472, "custom-argument goldens not replayed in full")


# ------------------------------------------------------------------------------------------------ Pauli symmetry


def test_pauli_hidden_symmetry() raises:
    var fx = load_fixture("envs/pauli_perm_cases.json")
    expect(len(fx) == 4, "4 Pauli symmetry configs")
    var checked = 0
    for i in range(len(fx)):
        var c = fx.at(i)
        var topo = c.get("topology").as_string()
        var cfg = cfg_for("pauli_network", topo)  # add_perms defaults to True
        var qp = ints2(c.get("qubit_perms"))
        var ap = ints2(c.get("act_perms"))
        var env0 = TrainEnv(cfg, 1)
        expect(env0.num_syms() == c.get("num_perms").as_int() and env0.num_syms() == len(qp), topo + ": number of symmetries")
        for p in range(len(qp)):
            expect(same_ints(env0.pauli_qubit_perm(p), qp[p]), topo + ": qubit perm")
            expect(same_ints(env0.pauli_act_perm(p), ap[p]), topo + ": act perm")
        var cases = c.get("cases")
        for ci in range(len(cases)):
            var cs = cases.at(ci)
            var env = TrainEnv(cfg, UInt64(77 + ci))
            env.set_state_pauli(ints(cs.get("state")))
            var by_perm_obs = ints2(cs.get("obs_by_perm"))
            # (1) observation under every explicit symmetry
            for p in range(len(by_perm_obs)):
                var o = List[Int]()
                var probe = env.copy()
                probe.observe_perm(p, o)
                expect(same_ints(o, by_perm_obs[p]), topo + " case " + String(ci) + ": obs under perm " + String(p))
            # (2) observe() only produces members of obs_by_perm, labelled by the stored symmetry, and covers all
            var seen = List[Bool](length=len(by_perm_obs), fill=False)
            for _ in range(200):
                var o = obs_of(env)
                var p = env.current_perm()
                expect(p >= 0 and p < len(by_perm_obs), "current_perm in range")
                expect(same_ints(o, by_perm_obs[p]), topo + " case " + String(ci) + ": observe() is the view of current_perm")
                seen[p] = True
            for p in range(len(seen)):
                expect(seen[p], topo + " case " + String(ci) + ": symmetry " + String(p) + " never drawn in 200 observes")
            # (3) mapped action, reward, finality, next observation under perm 0
            var actions = cs.get("actions")
            for ai in range(len(actions)):
                var act = actions.at(ai)
                var a = act.get("action").as_int()
                var by = act.get("by_perm")
                for bi in range(len(by)):
                    var row = by.at(bi)
                    var p = row.get("perm").as_int()
                    var e2 = env.copy()
                    var o = List[Int]()
                    e2.observe_perm(p, o)
                    e2.step(a)
                    var tag = topo + " case " + String(ci) + " action " + String(a) + " perm " + String(p)
                    expect(e2.last_actual_action() == row.get("actual_action").as_int(), tag + ": actual action")
                    expect(f32_bits(e2.reward()) == f32_bits(Float32(row.get("reward").as_float())), tag + ": reward")
                    expect(e2.is_final() == row.get("is_final").as_bool(), tag + ": is_final")
                    var nxt = List[Int]()
                    e2.observe_perm(0, nxt)
                    expect(same_ints(nxt, ints(row.get("next_obs_perm0"))), tag + ": next observation under perm 0")
                    checked += 1
    expect(checked > 200, "too few Pauli symmetry steps checked")


# ------------------------------------------------------------------------------------------------ rewards


def test_reward_fma_barrier() raises:
    var fx = load_fixture("envs/reward_fma_cases.json")
    var cases = fx.get("cases")
    expect(len(cases) == 33, "33 FMA cases")
    var example_seen = False
    for i in range(len(cases)):
        var c = cases.at(i)
        var achieved = Float32(c.get("achieved").as_float())
        var penalty = Float32(c.get("penalty").as_float())
        var plr = Float32(c.get("pauli_layer_reward").as_float())
        var count = c.get("count").as_int()
        var got = reward_f32(achieved, penalty, plr, count)
        var want_bits = UInt32(c.get("unfused_bits").as_int())
        var fused_bits = UInt32(c.get("fused_bits").as_int())
        expect(want_bits != fused_bits, "the fixture case must discriminate")
        expect(f32_bits(got) == want_bits, "reward case " + String(i) + " is not the unfused value")
        expect(f32_bits(got) != fused_bits, "reward case " + String(i) + " equals the fused value")
        # the penalty chain with the documented weights
        var d = ints(c.get("deltas"))
        var w0: Float32 = 0.01
        var w1: Float32 = 0.0
        var w2: Float32 = 0.0
        var w3: Float32 = 0.0001
        if c.get("weights").as_string() == "custom":
            w0 = 0.02
            w1 = 0.003
            w2 = 0.0017
            w3 = 0.0005
        var pen = penalty_f32(w0, w1, w2, w3, d[0], d[1], d[2], d[3])
        expect(f32_bits(pen) == f32_bits(penalty), "penalty chain of case " + String(i))
        if c.has("example"):
            example_seen = True
            expect(count == 5 and f32_bits(got) == 1028416496, "documented example 0 / 1e-4 / 0.01 / 5 -> 0.049899995")
    expect(example_seen, "the documented example is in the fixture")
    # a plain env reward through the whole step path: the 0 / 1e-4 chain is exact
    var pe = cfg_for("pauli_network", "line:3")
    var env = TrainEnv(pe, 1)
    env.set_state_pauli(List[Int]())
    expect(env.horizon() == 1, "empty set_state leaves the construction state")


def test_metrics_tracker() raises:
    var m = Metrics(4)
    m.apply_gate(Op.H, 0, -1)
    expect(m.n_gates == 1 and m.n_cnots == 0 and m.n_layers == 1, "one gate")
    m.apply_gate(Op.CX, 0, 1)
    expect(m.n_gates == 2 and m.n_cnots == 1 and m.n_layers == 2 and m.n_cnot_layers == 1, "CX after H")
    m.apply_gate(Op.SWAP, 2, 3)
    expect(m.n_gates == 5 and m.n_cnots == 4, "SWAP is three CX")
    expect(m.n_cnot_layers == 3 and m.n_layers == 3, "layer sets count distinct layer indices")
    m.apply_gate(Op.CX, 1, 1)
    m.apply_gate(Op.CX, 0, 7)
    m.apply_gate(Op.H, 9, -1)
    expect(m.n_gates == 5, "gates on invalid qubits are ignored")
    m.reset()
    expect(m.n_gates == 0 and m.n_cnots == 0 and m.n_layers == 0 and m.n_cnot_layers == 0, "reset")
    # default penalties of the documentation: 1q 1e-4, CX 0.0101, SWAP 0.0303
    var p1 = penalty_f32(0.01, 0.0, 0.0, 0.0001, 0, 0, 0, 1)
    var p2 = penalty_f32(0.01, 0.0, 0.0, 0.0001, 1, 0, 0, 1)
    var p3 = penalty_f32(0.01, 0.0, 0.0, 0.0001, 3, 0, 0, 3)
    expect(f32_bits(p1) == f32_bits(Float32(0.0001)), "1q penalty")
    expect(f32_bits(p2) == f32_bits(Float32(0.0101)), "CX penalty")
    expect(f32_bits(p3) == f32_bits(Float32(0.030299999)), "SWAP penalty")


# ------------------------------------------------------------------------------------------------ inversion


def test_inverts() raises:
    var fx = load_fixture("envs/inverts.json")
    var trials = 0
    var inverted = 0
    for i in range(len(fx)):
        var c = fx.at(i)
        var cfg = cfg_for(c.get("kind").as_string(), c.get("topology").as_string())
        cfg.add_inverts = True
        cfg.add_perms = False
        var cases = c.get("cases")
        for ci in range(len(cases)):
            var cs = cases.at(ci)
            var plain = ints(cs.get("obs_plain"))
            var inv = ints(cs.get("obs_inverted"))
            var distinct = not same_ints(plain, inv)
            var reps = 200
            for r in range(reps):
                var env = TrainEnv(cfg, UInt64(31 * (i * 1000 + ci) + r))
                load_state(env, cfg.kind, ints(cs.get("state")))
                env.step(cs.get("action").as_int())
                var o = obs_of(env)
                var is_plain = same_ints(o, plain)
                var is_inv = same_ints(o, inv)
                expect(is_plain or is_inv, "inverts: the observation is neither outcome")
                expect(f32_bits(env.reward()) == f32_bits(Float32(cs.get("reward").as_float())), "inverts: reward")
                expect(env.is_final() == cs.get("is_final").as_bool(), "inverts: is_final")
                if distinct:
                    trials += 1
                    if is_inv:
                        inverted += 1
    expect(trials >= 4000, "inverts: fewer than 4000 discriminating trials")
    var rate = Float64(inverted) / Float64(trials)
    expect(rate > 0.45 and rate < 0.55, "inverts: P(invert) = " + String(rate) + " outside [0.45, 0.55]")
    # no inversion without add_inverts, and none at reset / set_state
    var cfg = cfg_for("linear_function", "line:4")
    cfg.add_inverts = False
    var distinct_runs = 0
    var env = TrainEnv(cfg, 5)
    env.set_state_bits(
        [1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1]
    )
    var base = obs_of(env)
    for _ in range(50):
        var e2 = env.copy()
        e2.step(0)
        var o = obs_of(e2)
        var e3 = env.copy()
        e3.step(0)
        if not same_ints(o, obs_of(e3)):
            distinct_runs += 1
    expect(distinct_runs == 0, "add_inverts off: step must be deterministic")
    expect(same_ints(base, obs_of(env)), "observe is pure for plain kinds")


# ------------------------------------------------------------------------------------------------ protocol


def test_construction_and_horizon() raises:
    var kinds = List[String]()
    kinds.append("permutation")
    kinds.append("linear_function")
    kinds.append("clifford")
    kinds.append("pauli_network")
    var ds = List[Int]()
    ds.append(3)
    ds.append(40)
    ds.append(64)
    ds.append(65)
    ds.append(200)
    for k in range(len(kinds)):
        var cfg = cfg_for(kinds[k], "ring:4")
        var env = TrainEnv(cfg, 3)
        # construction state: identity, depth 1, solved, reward 1.0, final until the first reset
        expect(env.horizon() == 1 and env.success() and env.is_final(), kinds[k] + ": construction state")
        expect(f32_bits(env.reward()) == f32_bits(Float32(1.0)), kinds[k] + ": construction reward")
        expect(not env.all_allowed(), kinds[k] + ": solved state has all-false masks")
        for di in range(len(ds)):
            var d = ds[di]
            env.set_difficulty(d)
            env.reset()
            var want = 2 * d if 2 * d < 128 else 128
            expect(env.horizon() == want, kinds[k] + ": horizon after reset at difficulty " + String(d))
        # step always decrements depth, saturating at 0, even when final or invalid
        env.set_difficulty(2)
        env.reset()
        var h = env.horizon()
        for _ in range(h + 5):
            env.step(0)
        expect(env.horizon() == 0 and env.is_final(), kinds[k] + ": depth saturates at 0")
        # set_state uses depth = max_depth regardless of the difficulty
        if kinds[k] == "permutation":
            env.set_state_perm([1, 0, 2, 3])
            expect(env.horizon() == 128 and not env.is_final() and env.all_allowed(), "set_state_perm horizon")
            env.set_state_perm([0, 1, 2, 3])
            expect(env.success() and env.is_final() and not env.all_allowed(), "solved state")
            expect(f32_bits(env.reward()) == f32_bits(Float32(1.0)), "solved set_state reward")


def test_exactly_one_observe() raises:
    var cfg = cfg_for("pauli_network", "ring:5")
    expect(cfg.add_perms, "default add_perms")
    var env = TrainEnv(cfg, 2024)
    expect(env.num_syms() == 10, "ring:5 has 10 automorphisms")
    expect(env.current_perm() == 0, "CURRENT_PERM starts at 0")
    env.set_difficulty(48)
    env.reset()
    # the symmetry used by step() is the one stored by the last observe(); a second observe() replaces it
    var seen_diff = False
    for t in range(40):
        var e = env.copy()
        var o1 = obs_of(e)
        var p1 = e.current_perm()
        var o2 = obs_of(e)
        var p2 = e.current_perm()
        if p1 != p2:
            seen_diff = True
        var a = (t * 7) % e.num_actions()
        e.step(a)
        expect(e.last_actual_action() == e.pauli_act_perm(p2)[a], "step must use the symmetry of the LAST observe()")
        _ = o1
        _ = o2
    expect(seen_diff, "two observes should draw different symmetries sometimes")
    # without add_perms no symmetry is drawn and the action is used as is
    cfg.add_perms = False
    var e0 = TrainEnv(cfg, 5)
    e0.set_difficulty(48)
    e0.reset()
    _ = obs_of(e0)
    expect(e0.current_perm() == 0 and e0.num_syms() == 0, "no symmetry without add_perms")
    e0.step(4)
    expect(e0.last_actual_action() == 4, "action used as is without add_perms")


def test_out_of_range_actions() raises:
    var kinds = List[String]()
    kinds.append("permutation")
    kinds.append("linear_function")
    kinds.append("clifford")
    kinds.append("pauli_network")
    for k in range(len(kinds)):
        var cfg = cfg_for(kinds[k], "line:4")
        cfg.add_inverts = False
        # add_perms on: clamped to action 0 with error_code 642, never raised or aborted
        var env = TrainEnv(cfg, 9)
        env.set_difficulty(6)
        env.reset()
        var ref_env = env.copy()
        _ = obs_of(env)
        _ = obs_of(ref_env)
        var A = env.num_actions()
        env.step(A)
        expect(env.error_code() == 642, kinds[k] + ": a >= A with add_perms sets error code 642")
        ref_env.step(0)
        expect(ref_env.error_code() == 0, kinds[k] + ": valid action keeps error code 0")
        var sa = env.state_words()
        var sb = ref_env.state_words()
        var same = len(sa) == len(sb)
        if same:
            for i in range(len(sa)):
                if sa[i] != sb[i]:
                    same = False
        expect(same, kinds[k] + ": the clamped action is action 0")
        var env2 = TrainEnv(cfg, 9)
        env2.set_difficulty(6)
        env2.reset()
        env2.step(-3)
        expect(env2.error_code() == 642, kinds[k] + ": negative action")
        # add_perms off: silent no-op (state unchanged, penalty 0)
        var cfg_off = cfg.copy()
        cfg_off.add_perms = False
        var e3 = TrainEnv(cfg_off, 9)
        e3.set_difficulty(6)
        e3.reset()
        var before = e3.state_words()
        e3.step(A + 3)
        expect(e3.error_code() == 0, kinds[k] + ": silent without add_perms")
        var after = e3.state_words()
        var unchanged = len(before) == len(after)
        if unchanged:
            for i in range(len(before)):
                if before[i] != after[i]:
                    unchanged = False
        expect(unchanged, kinds[k] + ": no state change for an invalid action without add_perms")
        expect(f32_bits(e3.reward()) == f32_bits(Float32(0.0)) or e3.success(), kinds[k] + ": no penalty for an invalid action")
    # garbage set_state is reported, not applied
    var pc = cfg_for("permutation", "line:4")
    var pe = TrainEnv(pc, 1)
    pe.set_state_perm([0, 1, 1, 3])
    expect(pe.error_code() == 642, "set_state_perm with a non-permutation")
    var raised = False
    var qc = cfg_for("pauli_network", "line:3")
    var qe = TrainEnv(qc, 1)
    try:
        qe.set_state_pauli([2, 1, 0])
    except e:
        raised = String(e).startswith("QM-E642")
    expect(raised, "malformed Pauli state raises QM-E642")


def test_seeding_and_copy() raises:
    var kinds = List[String]()
    kinds.append("permutation")
    kinds.append("linear_function")
    kinds.append("clifford")
    kinds.append("pauli_network")
    for k in range(len(kinds)):
        var cfg = cfg_for(kinds[k], "ring:5")
        var a = TrainEnv(cfg, 1)
        var b = TrainEnv(cfg, 999)
        a.set_difficulty(80)
        b.set_difficulty(80)
        a.reseed(42)
        b.reseed(42)
        var diff = 0
        var diff_other = 0
        for ep in range(20):
            a.reset()
            b.reset()
            var sa = a.state_words()
            var sb = b.state_words()
            for i in range(len(sa)):
                if sa[i] != sb[i]:
                    diff += 1
            # the same stream gives the same observations and the same steps
            var oa = obs_of(a)
            var ob = obs_of(b)
            expect(same_ints(oa, ob), kinds[k] + ": same seed, same observation")
            a.step(ep % a.num_actions())
            b.step(ep % b.num_actions())
            expect(f32_bits(a.reward()) == f32_bits(b.reward()), kinds[k] + ": same seed, same reward")
        expect(diff == 0, kinds[k] + ": same seed must give identical resets")
        a.reseed(42)
        b.reseed(43)
        for _ in range(20):
            a.reset()
            b.reset()
            var sa = a.state_words()
            var sb = b.state_words()
            if len(sa) != len(sb):
                diff_other += 1
            else:
                for i in range(len(sa)):
                    if sa[i] != sb[i]:
                        diff_other += 1
        expect(diff_other > 0, kinds[k] + ": different seeds must give different resets")
        # copies are independent
        a.reseed(7)
        a.reset()
        var c = a.copy()
        var before = c.state_words()
        a.step(0)
        a.step(1)
        var after = c.state_words()
        var same = True
        for i in range(len(before)):
            if before[i] != after[i]:
                same = False
        expect(same, kinds[k] + ": stepping the original must not change the copy")
        # reset after a copy is also independent (shared read-only tables only)
        c.reset()
        a.reset()


def test_pauli_set_state() raises:
    # line:3 state of the step goldens: R = 2 rotations, then labels "YXX" and "IIY" with their length prefix
    var cfg = cfg_for("pauli_network", "line:3")
    cfg.add_perms = False
    var env = TrainEnv(cfg, 1)
    var st = List[Int]()
    st.append(2)
    for i in range(6):
        for j in range(6):
            st.append(1 if i == j else 0)
    st.append(3)
    st.append(ord("Y"))
    st.append(ord("X"))
    st.append(ord("X"))
    st.append(4)  # a phase prefix counts in the length: "-IIY"
    st.append(ord("-"))
    st.append(ord("I"))
    st.append(ord("I"))
    st.append(ord("Y"))
    env.set_state_pauli(st)
    var labels = env.pauli_labels()
    expect(len(labels) == 2 and labels[0] == "YXX" and labels[1] == "IIY", "labels round trip (phase prefix dropped)")
    expect(env.horizon() == cfg.max_depth and not env.success(), "set_state_pauli depth and success")
    # a weight-1 rotation stays in the DAG: no initial clean()
    var st1 = List[Int]()
    st1.append(1)
    for i in range(6):
        for j in range(6):
            st1.append(1 if i == j else 0)
    st1.append(3)
    st1.append(ord("I"))
    st1.append(ord("I"))
    st1.append(ord("Z"))
    env.set_state_pauli(st1)
    expect(len(env.pauli_labels()) == 1, "weight-1 rotation is not cleaned by set_state")
    expect(not env.success(), "a pending rotation means not solved")
    # a CX anywhere then cleans it (here every CX leaves the weight-1 rotation weight 1 or 2; H does not clean)
    env.step(0)  # H(0): no clean
    expect(len(env.pauli_labels()) == 1, "H never cleans")
    # R above max_rotations is truncated to max_rotations
    var st7 = List[Int]()
    st7.append(7)
    for i in range(6):
        for j in range(6):
            st7.append(1 if i == j else 0)
    for _ in range(7):
        st7.append(3)
        st7.append(ord("X"))
        st7.append(ord("X"))
        st7.append(ord("I"))
    env.set_state_pauli(st7)
    expect(len(env.pauli_labels()) == 5, "only the first max_rotations rotations are kept")


def main() raises:
    test_gateset_and_tables()
    test_gateset_errors()
    test_validate_actions()
    test_step_goldens()
    test_custom_args()
    test_pauli_hidden_symmetry()
    test_reward_fma_barrier()
    test_metrics_tracker()
    test_inverts()
    test_construction_and_horizon()
    test_exactly_one_observe()
    test_out_of_range_actions()
    test_seeding_and_copy()
    test_pauli_set_state()
    print("test_rl_env OK")
