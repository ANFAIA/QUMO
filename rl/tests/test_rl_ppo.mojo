"""Tests for qrl.ppo: the update, the collector, the evaluator and the trainer (MOJO_PORT.md 7, 13.2 `test_rl_ppo`).

Pinned against data produced by the real packages (torch 2.14.1 / twisterl 0.5.1):
  * `ppo/ppo_update.json` through the integrated `ppo_update` (Adam, chunks, virtual workers): the loss dict of the
    last epoch and the weights after 1, 2 and 10 epochs, and the `normalize_advantage` variant;
independent checks that need no fixture:
  * T-DET: same seed, cores 1, 2, 4 and 8 (clamped), three iterations of the trainer on `permutation line:3`, `linear
    ring:4` and `pauli line:3`: identical weight bytes and identical log lines; a rerun is identical;
  * `ppo_update` is bit-identical for 1, 2, 4, 8 cores on the golden batch replicated to 5,012 rows (79 chunks), and the
    sparse-row Adam equals the dense one bit for bit;
  * stage seeds: iteration 0 of a warm-start stage uses other streams and collects other start states than iteration 0
    of the first stage, and equals the same global iteration of the first stage;
  * collector self-consistency on the four kinds: a replay of every episode against the environment (observation
    before the action, reward of the state itself, terminal row recorded, its action never executed, mask all `-1e10`
    iff solved), recorded logits equal `forward_one` bit for bit where allowed, the update's forward with the row's twist
    reproduces them, GAE equals the f64 closed form and the terminal row has no bootstrap;
  * the promotion rule (170 / 200 promotes, 169 does not), the evaluation order and accounting, the NaN guard (QM-E643,
    weights and Adam state restored) and the status slots (QM-E642, deterministic for any core count).
"""

from std.memory import bitcast
from std.math import exp
from qrl.json import JsonValue
from qrl.model import model_kind_code
from qrl.rng import Rng
from qrl.topology import parse_topology
from qrl.gateset import EnvConfig, build_gateset
from qrl.synth_env import TrainEnv
from qrl.tensor import FPtr
from qrl.optim import Adam
from qrl.nn import NetSpec, BasicNet, ChunkCache, ActorScratch
from qrl.loss import PPOLossCfg, IPtr, ppo_chunk_loss, old_logp_row
from qrl.ppo import (
    PPOConfig,
    Rollout,
    EvalResult,
    LossStats,
    StepReport,
    Trainer,
    collect,
    evaluate,
    ppo_update,
    effective_cores,
    collect_env_seed,
    collect_actor_seed,
    eval_better,
    first_status,
    MASKED_LOGIT,
)
from qrl.testutil import (
    load_fixture,
    flat_floats32,
    floats32,
    ints,
    ints2,
    assert_close,
    assert_list_close32,
    f32_bits,
)


# ------------------------------------------------------------------------------------------------ builders


def make_env(kind_name: String, topo: String, difficulty: Int) raises -> TrainEnv:
    var t = parse_topology(topo)
    var cfg = EnvConfig()
    cfg.kind = model_kind_code(kind_name)
    cfg.n = t.n
    cfg.actions = build_gateset(cfg.kind, t.n, t.edges)
    cfg.difficulty = difficulty
    return TrainEnv(cfg, 1)


def make_spec(env: TrainEnv, emb: Int, common: List[Int]) -> NetSpec:
    var tw = env.twists()
    var s = NetSpec()
    s.obs_size = env.obs_size()
    s.num_actions = env.num_actions()
    s.emb = emb
    s.common = common.copy()
    s.obs_perms = tw.obs_perms.copy()
    s.act_perms = tw.act_perms.copy()
    return s^


def make_net(env: TrainEnv, emb: Int, seed: UInt64) raises -> BasicNet:
    var common: List[Int] = [8]
    return BasicNet(make_spec(env, emb, common), seed)


def small_cfg(cores: Int, episodes: Int = 40) -> PPOConfig:
    var c = PPOConfig()
    c.num_cores = cores
    c.num_episodes = episodes
    c.eval_episodes = 24
    c.eval_sampled_searches = 3
    c.eval_sampled_every = 1
    c.chunk_rows = 16
    c.seed = 5
    return c^


def bits_equal(a: List[Float32], b: List[Float32]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            return False
    return True


def rollouts_equal(a: Rollout, b: Rollout) -> Bool:
    if a.n != b.n or a.num_episodes != b.num_episodes or a.status != b.status:
        return False
    if len(a.sb.idx) != len(b.sb.idx) or len(a.sb.off) != len(b.sb.off):
        return False
    for i in range(len(a.sb.idx)):
        if a.sb.idx[i] != b.sb.idx[i]:
            return False
    for i in range(len(a.sb.off)):
        if a.sb.off[i] != b.sb.off[i]:
            return False
    for i in range(a.n):
        if a.sb.perm[i] != b.sb.perm[i] or a.acts[i] != b.acts[i] or a.allowed[i] != b.allowed[i]:
            return False
    return (
        bits_equal(a.logits, b.logits)
        and bits_equal(a.advs, b.advs)
        and bits_equal(a.rets, b.rets)
        and bits_equal(a.old_lp, b.old_lp)
        and bits_equal(a.values, b.values)
        and bits_equal(a.rewards, b.rewards)
    )


def _cf(l: List[Float32]) -> FPtr:
    return l.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin()


def _ci(l: List[Int32]) -> IPtr:
    return l.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin()


# ------------------------------------------------------------------------------------------------ golden batch


def tiny_spec() raises -> NetSpec:
    var fx = load_fixture("ppo/policy_forward.json")
    var s = NetSpec()
    s.obs_size = fx.get("obs_size").as_int()
    s.num_actions = fx.get("num_actions").as_int()
    s.emb = fx.get("emb").as_int()
    s.common = ints(fx.get("common"))
    s.obs_perms = ints2(fx.get("obs_perms"))
    s.act_perms = ints2(fx.get("act_perms"))
    return s^


def set_param_torch(mut net: BasicNet, i: Int, vals: List[Float32]) raises:
    """Writes tensor `i` (registration order) from torch-layout values; `embeddings.weight` is stored transposed."""
    var off = net.ps.offsets[i]
    if len(vals) != net.ps.size_of(i):
        raise Error("tensor " + String(i) + ": wrong number of values")
    if i == 0:
        var obs = net.ps.rows_of(0)
        var e = net.ps.cols_of(0)
        for o in range(e):
            for k in range(obs):
                net.ps.data[off + k * e + o] = vals[o * obs + k]
    else:
        for j in range(len(vals)):
            net.ps.data[off + j] = vals[j]


def torch_layout(net: BasicNet, i: Int) -> List[Float32]:
    var off = net.ps.offsets[i]
    var n = net.ps.size_of(i)
    var out = List[Float32](length=n, fill=0.0)
    if i == 0:
        var obs = net.ps.rows_of(0)
        var e = net.ps.cols_of(0)
        for o in range(e):
            for k in range(obs):
                out[o * obs + k] = net.ps.data[off + k * e + o]
    else:
        for j in range(n):
            out[j] = net.ps.data[off + j]
    return out^


def tensor_keys() raises -> List[String]:
    var keys = List[String]()
    var order = load_fixture("ppo/policy_forward.json").get("tensor_order")
    for i in range(len(order)):
        keys.append(order.at(i).as_string())
    return keys^


def tiny_net() raises -> BasicNet:
    var fx = load_fixture("ppo/policy_forward.json")
    var net = BasicNet(tiny_spec(), 0, False)
    var keys = tensor_keys()
    for i in range(len(keys)):
        set_param_torch(net, i, flat_floats32(fx.get("weights").get(keys[i]).get("data")))
    return net^


def golden_rollout(reps: Int = 1) raises -> Rollout:
    """The 14-row batch of `ppo_update.json` as a Rollout (repeated `reps` times)."""
    var fx = load_fixture("ppo/ppo_update.json")
    var b = fx.get("batch")
    var obs = ints2(b.get("obs"))
    var perm = ints(b.get("perm"))
    var acts = ints(b.get("actions"))
    var advs = floats32(b.get("advs"))
    var rets = floats32(b.get("rets"))
    var olp = floats32(b.get("old_logp"))
    var lg = b.get("old_logits_masked")
    var ro = Rollout()
    for _ in range(reps):
        for r in range(len(obs)):
            ro.sb.append_row_ints(obs[r], perm[r])
            var row = floats32(lg.at(r))
            for a in range(len(row)):
                ro.logits.append(row[a])
            ro.acts.append(Int32(acts[r]))
            ro.advs.append(advs[r])
            ro.rets.append(rets[r])
            ro.old_lp.append(olp[r])
            ro.allowed.append(1)
            ro.values.append(0.0)
            ro.rewards.append(0.0)
    ro.n = len(obs) * reps
    ro.num_episodes = 1
    return ro^


def golden_cfg(epochs: Int) raises -> PPOConfig:
    var fx = load_fixture("ppo/ppo_update.json")
    var tj = fx.get("config").get("training")
    var c = PPOConfig()
    c.num_epochs = epochs
    c.vf_coef = Float32(tj.get("vf_coef").as_float())
    c.ent_coef = Float32(tj.get("ent_coef").as_float())
    c.clip_ratio = Float32(tj.get("clip_ratio").as_float())
    c.lr = fx.get("config").get("optimizer").get("lr").as_float()
    c.num_cores = 1
    c.chunk_rows = 512
    return c^


def test_update_golden() raises:
    var fx = load_fixture("ppo/ppo_update.json")
    var keys = tensor_keys()
    var epochs = fx.get("epochs")
    var checked = 0
    for k in range(len(epochs)):
        var ep = epochs.at(k)
        var n_ep = k + 1
        if not (n_ep == 1 or n_ep == 2 or n_ep == 10):
            continue
        var net = tiny_net()
        var cfg = golden_cfg(n_ep)
        var opt = Adam(net.param_count(), cfg.lr)
        var ro = golden_rollout()
        var ls = ppo_update(net, opt, ro, cfg)
        var want = ep.get("losses")
        assert_close(ls.value, want.get("value").as_float(), 2e-5, 2e-5, "value loss after " + String(n_ep) + " epochs")
        assert_close(ls.policy, want.get("policy").as_float(), 2e-5, 2e-5, "policy loss after " + String(n_ep) + " epochs")
        assert_close(ls.entropy, want.get("entropy").as_float(), 2e-5, 2e-5, "entropy after " + String(n_ep) + " epochs")
        assert_close(ls.total, want.get("total").as_float(), 2e-5, 2e-5, "total loss after " + String(n_ep) + " epochs")
        if opt.steps_taken() != n_ep:
            raise Error("Adam took " + String(opt.steps_taken()) + " steps, expected " + String(n_ep))
        for i in range(len(keys)):
            var ww = flat_floats32(ep.get("weights_after").get(keys[i]).get("data"))
            assert_list_close32(torch_layout(net, i), ww, 3e-5, 1e-4, "weights after " + String(n_ep) + " epochs " + keys[i])
        checked += 1
    assert_close(Float64(checked), 3.0, 0.0, 0.0, "golden epoch counts checked")
    print("  ppo_update vs the golden: losses and weights after 1, 2 and 10 epochs OK")


def test_normalize_variant() raises:
    var fx = load_fixture("ppo/ppo_update.json")
    var keys = tensor_keys()
    var net = tiny_net()
    var cfg = golden_cfg(1)
    cfg.normalize_advantage = True
    var opt = Adam(net.param_count(), cfg.lr)
    var ro = golden_rollout()
    var ls = ppo_update(net, opt, ro, cfg)
    var want = fx.get("normalize_advantage_variant").get("loss")
    assert_close(ls.policy, want.get("policy").as_float(), 2e-5, 2e-5, "normalised policy loss")
    assert_close(ls.value, want.get("value").as_float(), 2e-5, 2e-5, "normalised value loss")
    assert_close(ls.total, want.get("total").as_float(), 2e-5, 2e-5, "normalised total loss")
    var wa = fx.get("normalize_advantage_variant").get("weights_after_step1")
    for i in range(len(keys)):
        var ww = flat_floats32(wa.get(keys[i]).get("data"))
        assert_list_close32(torch_layout(net, i), ww, 3e-5, 1e-4, "normalised weights " + keys[i])
    # the rollout's own advantages are untouched (the normalisation works on a copy)
    var fresh = golden_rollout()
    if not bits_equal(ro.advs, fresh.advs):
        raise Error("ppo_update modified Rollout.advs")
    print("  ppo_update with normalize_advantage OK")


def test_update_thread_independence() raises:
    var base = tiny_net()
    var ro = golden_rollout(358)  # 5,012 rows
    assert_close(Float64(ro.n), 5012.0, 0.0, 0.0, "replicated rows")
    var ref_w = List[Float32]()
    var ref_l = LossStats(0.0, 0.0, 0.0, 0.0)
    var counts: List[Int] = [1, 2, 4, 8, 16]
    for ci in range(len(counts)):
        var net = BasicNet(tiny_spec(), 0, False)
        net.ps.data = base.ps.data.copy()
        var cfg = golden_cfg(3)
        cfg.num_cores = counts[ci]
        cfg.chunk_rows = 64  # 79 chunks over 16 virtual workers
        var opt = Adam(net.param_count(), cfg.lr)
        var ls = ppo_update(net, opt, ro, cfg)
        if ci == 0:
            ref_w = net.ps.data.copy()
            ref_l = ls.copy()
        else:
            if not bits_equal(net.ps.data, ref_w):
                raise Error("ppo_update weights differ between 1 and " + String(counts[ci]) + " cores")
            if (
                bitcast[DType.uint64](ls.total) != bitcast[DType.uint64](ref_l.total)
                or bitcast[DType.uint64](ls.policy) != bitcast[DType.uint64](ref_l.policy)
                or bitcast[DType.uint64](ls.value) != bitcast[DType.uint64](ref_l.value)
                or bitcast[DType.uint64](ls.entropy) != bitcast[DType.uint64](ref_l.entropy)
            ):
                raise Error("ppo_update losses differ between 1 and " + String(counts[ci]) + " cores")
    # the replicated batch is a pure repetition: the same update as the 14-row batch (different summation order)
    var net14 = tiny_net()
    var cfg14 = golden_cfg(3)
    var opt14 = Adam(net14.param_count(), cfg14.lr)
    var ro14 = golden_rollout(1)
    _ = ppo_update(net14, opt14, ro14, cfg14)
    assert_list_close32(ref_w, net14.ps.data, 3e-5, 1e-3, "replicated batch update vs the 14-row update")
    print("  ppo_update identical for 1, 2, 4, 8, 16 cores (5,012 rows, 79 chunks) OK")


# ------------------------------------------------------------------------------------------------ trainer determinism


struct RunOut(Movable):
    var weights: List[Float32]
    var log: String
    var difficulty: Int

    def __init__(out self):
        self.weights = List[Float32]()
        self.log = String()
        self.difficulty = 0


def run_trainer(
    kind: String, topo: String, cores: Int, iters: Int, start_diff: Int, start_iter: Int = 0, episodes: Int = 40
) raises -> RunOut:
    var env = make_env(kind, topo, start_diff)
    var net = make_net(env, 16, 7)
    var cfg = small_cfg(cores, episodes)
    cfg.start_iteration = start_iter
    var tr = Trainer(env, net^, cfg)
    var out = RunOut()
    for i in range(iters):
        var rep = tr.learn_step()
        var pr = tr.promote_if_ready(rep)
        out.log += (
            String(i) + "|" + String(rep.difficulty) + "|" + String(f32_bits(rep.det.success)) + "|"
            + String(f32_bits(rep.det.reward)) + "|" + String(f32_bits(rep.sampled.success)) + "|"
            + String(f32_bits(rep.sampled.reward)) + "|" + String(bitcast[DType.uint64](rep.loss.total)) + "|"
            + String(bitcast[DType.uint64](rep.loss.policy)) + "|" + String(bitcast[DType.uint64](rep.loss.value)) + "|"
            + String(bitcast[DType.uint64](rep.loss.entropy)) + "|" + String(rep.rows) + "|" + String(pr) + "\n"
        )
    out.weights = tr.net.ps.data.copy()
    out.difficulty = tr.env.difficulty
    return out^


def test_t_det() raises:
    var kinds: List[String] = ["permutation", "linear_function", "pauli_network"]
    var topos: List[String] = ["line:3", "ring:4", "line:3"]
    var diffs: List[Int] = [3, 3, 4]
    var iters: List[Int] = [3, 3, 2]
    var asked: List[Int] = [1, 2, 4, 8]
    for k in range(len(kinds)):
        var ref_out = run_trainer(kinds[k], topos[k], 1, iters[k], diffs[k])
        for c in range(1, len(asked)):
            var o = run_trainer(kinds[k], topos[k], asked[c], iters[k], diffs[k])
            if o.log != ref_out.log:
                raise Error("T-DET: " + kinds[k] + ": log lines differ between 1 and " + String(asked[c]) + " cores")
            if not bits_equal(o.weights, ref_out.weights):
                raise Error("T-DET: " + kinds[k] + ": weights differ between 1 and " + String(asked[c]) + " cores")
        var again = run_trainer(kinds[k], topos[k], 4, iters[k], diffs[k])
        if again.log != ref_out.log or not bits_equal(again.weights, ref_out.weights):
            raise Error("T-DET: " + kinds[k] + ": a rerun is not identical")
        # the run actually trained something (weights moved) and the log has one line per iteration
        var start = make_net(make_env(kinds[k], topos[k], diffs[k]), 16, 7)
        if bits_equal(start.ps.data, ref_out.weights):
            raise Error("T-DET: " + kinds[k] + ": the weights did not change")
    if effective_cores(8) != min(8, effective_cores(1000)):
        raise Error("effective_cores(8) is not clamped to the pool")
    print("  T-DET: cores 1, 2, 4, 8 and a rerun give identical weights and log lines (3 kinds) OK")


def test_stage_seeds() raises:
    var env = make_env("permutation", "line:4", 4)
    var net = make_net(env, 12, 3)
    var c1 = small_cfg(2, 48)
    var c2 = small_cfg(2, 48)
    c2.start_iteration = 3
    for e in range(60):
        if collect_env_seed(c1, 0, e) == collect_env_seed(c2, 0, e):
            raise Error("stage 2 iteration 0 uses the env stream of stage 1 iteration 0 (episode " + String(e) + ")")
        if collect_actor_seed(c1, 0, e) == collect_actor_seed(c2, 0, e):
            raise Error("stage 2 iteration 0 uses the actor stream of stage 1 iteration 0")
        if collect_env_seed(c2, 0, e) != collect_env_seed(c1, 3, e) or collect_actor_seed(c2, 0, e) != collect_actor_seed(c1, 3, e):
            raise Error("the global iteration key is not start_iteration + iteration")
        if collect_env_seed(c1, 0, e) == collect_actor_seed(c1, 0, e):
            raise Error("env and actor streams coincide")
    var r1 = collect(env, net, c1, 0)
    var r2 = collect(env, net, c2, 0)
    var r3 = collect(env, net, c1, 3)
    if not rollouts_equal(r2, r3):
        raise Error("stage 2 iteration 0 is not the same rollout as stage 1 iteration 3")
    if rollouts_equal(r1, r2):
        raise Error("stage 2 iteration 0 replayed stage 1 iteration 0")
    # different start states: count episodes whose first observation differs
    var diff_start = 0
    var rs1 = 0
    var rs2 = 0
    for e in range(c1.num_episodes):
        var a0 = Int(r1.sb.off[rs1])
        var a1 = Int(r1.sb.off[rs1 + 1])
        var b0 = Int(r2.sb.off[rs2])
        var b1 = Int(r2.sb.off[rs2 + 1])
        var same = (a1 - a0) == (b1 - b0)
        if same:
            for q in range(a1 - a0):
                if r1.sb.idx[a0 + q] != r2.sb.idx[b0 + q]:
                    same = False
        if not same:
            diff_start += 1
        rs1 += Int(r1.ep_len[e])
        rs2 += Int(r2.ep_len[e])
    if diff_start < 10:
        raise Error("only " + String(diff_start) + " of 48 start states differ between the two stages")
    print("  stage seeds: other streams and other start states for a warm-start stage (" + String(diff_start) + " of 48) OK")


# ------------------------------------------------------------------------------------------------ collector


def check_collector(kind: String, topo: String, difficulty: Int, episodes: Int) raises:
    var env = make_env(kind, topo, difficulty)
    var net = BasicNet(make_spec(env, 12, [8]), 21)
    var cfg = small_cfg(4, episodes)
    cfg.seed = 11
    cfg.gamma = 0.97
    cfg.lam = 0.9
    var it = 2
    var ro = collect(env, net, cfg, it)
    var A = env.num_actions()
    var T = len(net.spec.obs_perms)
    if ro.status != 0:
        raise Error(kind + ": collector status " + String(ro.status))
    if ro.num_episodes != episodes or len(ro.ep_len) != episodes:
        raise Error(kind + ": episode count")
    var total = 0
    for e in range(episodes):
        total += Int(ro.ep_len[e])
    if total != ro.n or len(ro.acts) != ro.n or len(ro.logits) != ro.n * A or len(ro.sb.off) != ro.n + 1:
        raise Error(kind + ": row counts are inconsistent")
    var sc = ActorScratch(net.spec)
    var lg = List[Float32](length=A, fill=0.0)
    var lp = lg.unsafe_ptr().as_unsafe_any_origin()
    var cc = ChunkCache(net.spec, 64)
    var row0 = 0
    var n_solved = 0
    var n_masked_rows = 0
    var max_diff_twist = 0.0
    var twist_seen = List[Int](length=max(T, 1) + 1, fill=0)
    var obs = List[Int]()
    for e in range(episodes):
        var L = Int(ro.ep_len[e])
        if L < 1 or L > env.cfg.max_depth + 1:
            raise Error(kind + ": episode length " + String(L))
        # (1) replay against the environment
        var ev = env.copy()
        ev.reseed(collect_env_seed(cfg, it, e))
        ev.reset()
        for r in range(L):
            var row = row0 + r
            ev.observe(obs)
            var q0 = Int(ro.sb.off[row])
            var q1 = Int(ro.sb.off[row + 1])
            if q1 - q0 != len(obs):
                raise Error(kind + ": episode " + String(e) + " row " + String(r) + ": observation length differs from the replay")
            for q in range(len(obs)):
                if Int(ro.sb.idx[q0 + q]) != obs[q]:
                    raise Error(kind + ": episode " + String(e) + " row " + String(r) + ": observation differs from the replay")
            if f32_bits(ro.rewards[row]) != f32_bits(ev.reward()):
                raise Error(kind + ": episode " + String(e) + " row " + String(r) + ": reward is not the reward of the state itself")
            var allowed = ro.allowed[row] == 1
            if allowed != ev.all_allowed():
                raise Error(kind + ": mask flag differs from the replay")
            var last = r == L - 1
            if ev.is_final() != last:
                raise Error(kind + ": episode " + String(e) + ": the terminal row is not the first final state (row " + String(r) + ")")
            # (2) recorded logits equal forward_one with the row's twist, bit for bit where allowed
            var t = Int(ro.sb.perm[row])
            if (T == 0) != (t == -1) or t >= max(T, 1):
                raise Error(kind + ": bad twist " + String(t))
            twist_seen[t + 1 if T > 0 else 0] += 1
            var span = Span(ro.sb.idx)[q0:q1]
            var v = net.forward_one(span, t, sc, lp)
            if f32_bits(v) != f32_bits(ro.values[row]):
                raise Error(kind + ": recorded value differs from forward_one")
            if allowed:
                for a in range(A):
                    if f32_bits(lg[a]) != f32_bits(ro.logits[row * A + a]):
                        raise Error(kind + ": recorded logits differ from forward_one (episode " + String(e) + " row " + String(r) + ")")
                if T > 1:
                    var t2 = (t + 1) % T
                    _ = net.forward_one(span, t2, sc, lp)
                    for a in range(A):
                        var d = abs(Float64(lg[a]) - Float64(ro.logits[row * A + a]))
                        if d > max_diff_twist:
                            max_diff_twist = d
            else:
                n_masked_rows += 1
                for a in range(A):
                    if ro.logits[row * A + a] != MASKED_LOGIT:
                        raise Error(kind + ": a solved state must carry all-masked logits")
            # old_lp is computed from the recorded masked logits
            var ol = old_logp_row(_cf(ro.logits).unsafe_offset(row * A), A, Int(ro.acts[row]))
            if f32_bits(ol) != f32_bits(ro.old_lp[row]):
                raise Error(kind + ": old_lp is not l_a - logsumexp(l) of the recorded logits")
            if Int(ro.acts[row]) < 0 or Int(ro.acts[row]) >= A:
                raise Error(kind + ": action out of range")
            if last:
                # the mask is all -1e10 iff the terminal state is solved; the terminal action is not executed
                if allowed == ev.success():
                    raise Error(kind + ": terminal mask flag does not match success")
                if ev.success():
                    n_solved += 1
            else:
                ev.step(Int(ro.acts[row]))
        if ev.error_code() != 0:
            raise Error(kind + ": replay raised an environment error")
        # (3) GAE: f64 closed form, terminal row without bootstrap
        var nxt = 0.0
        var tt = L - 1
        var gam = Float64(cfg.gamma)
        var lam = Float64(cfg.lam)
        while tt >= 0:
            var row = row0 + tt
            var delta = Float64(ro.rewards[row]) - Float64(ro.values[row])
            if tt + 1 < L:
                delta = Float64(ro.rewards[row]) + gam * Float64(ro.values[row0 + tt + 1]) - Float64(ro.values[row])
            nxt = delta + gam * lam * nxt
            assert_close(Float64(ro.advs[row]), nxt, 3e-5, 3e-5, kind + " advantage")
            assert_close(Float64(ro.rets[row]), nxt + Float64(ro.values[row]), 3e-5, 3e-5, kind + " return")
            tt -= 1
        var lrow = row0 + L - 1
        if f32_bits(ro.rets[lrow]) != f32_bits(ro.rewards[lrow]):
            raise Error(kind + ": terminal return bootstraps from something")
        if f32_bits(ro.advs[lrow]) != f32_bits(ro.rewards[lrow] - ro.values[lrow]):
            raise Error(kind + ": terminal advantage is not r - v")
        row0 += L
    # (4) the update's forward with the recorded twist reproduces the recorded logits (rows allowed)
    var done = 0
    var worst = 0.0
    while done < ro.n:
        var hi = min(done + 64, ro.n)
        net.forward_chunk(ro.sb, done, hi, cc)
        for r in range(hi - done):
            var row = done + r
            if ro.allowed[row] == 1:
                for a in range(A):
                    var d = abs(Float64(cc.logits[r * A + a]) - Float64(ro.logits[row * A + a]))
                    if d > worst:
                        worst = d
            var dv = abs(Float64(cc.values[r]) - Float64(ro.values[row]))
            if dv > 1e-5:
                raise Error(kind + ": the update's forward value differs from the collector's")
        done = hi
    if worst > 1e-5:
        raise Error(kind + ": the update's forward differs from the recorded logits by " + String(worst))
    if T > 1 and max_diff_twist < 1e-4:
        raise Error(kind + ": the twist check has no power (all twists agree)")
    if n_masked_rows != n_solved:
        raise Error(kind + ": masked rows and solved terminal states disagree")
    print("  collector " + kind + " " + topo + " d=" + String(difficulty) + ": " + String(ro.n) + " rows, " + String(n_solved) + " solved episodes OK")


def test_collector_consistency() raises:
    check_collector("permutation", "line:4", 4, 60)
    check_collector("linear_function", "ring:4", 3, 50)
    check_collector("clifford", "line:3", 3, 50)
    check_collector("pauli_network", "line:3", 4, 150)


# ------------------------------------------------------------------------------------------------ evaluate


def test_evaluate() raises:
    # difficulty 0: every episode starts solved, so success is 1 and the summed reward is the reward of that one state
    var env0 = make_env("permutation", "line:3", 0)
    var net0 = make_net(env0, 12, 4)
    var cfg = small_cfg(2, 30)
    var r0 = evaluate(env0, net0, cfg, 0, True, 1, 30)
    if f32_bits(r0.success) != f32_bits(1.0) or f32_bits(r0.reward) != f32_bits(1.0) or r0.status != 0:
        raise Error("evaluate at difficulty 0: expected (1, 1), got (" + String(r0.success) + ", " + String(r0.reward) + ")")
    # independent of the core count (bits), deterministic and sampled
    var env = make_env("linear_function", "ring:4", 3)
    var net = make_net(env, 12, 4)
    var ref_d = evaluate(env, net, cfg, 1, True, 1, 60)
    var ref_s = evaluate(env, net, cfg, 1, False, 4, 60)
    var counts: List[Int] = [1, 2, 4, 8]
    for ci in range(len(counts)):
        var c2 = small_cfg(counts[ci], 30)
        var d = evaluate(env, net, c2, 1, True, 1, 60)
        var s = evaluate(env, net, c2, 1, False, 4, 60)
        if f32_bits(d.success) != f32_bits(ref_d.success) or f32_bits(d.reward) != f32_bits(ref_d.reward):
            raise Error("deterministic evaluate depends on the core count")
        if f32_bits(s.success) != f32_bits(ref_s.success) or f32_bits(s.reward) != f32_bits(ref_s.reward):
            raise Error("sampled evaluate depends on the core count")
    # best of k searches contains search 0, so success never decreases with the number of searches
    var s1 = evaluate(env, net, cfg, 3, False, 1, 200)
    var s10 = evaluate(env, net, cfg, 3, False, 10, 200)
    if s10.success < s1.success:
        raise Error("sampled evaluate: 10 searches succeeded less often than 1 (" + String(s10.success) + " < " + String(s1.success) + ")")
    if s10.success <= s1.success:
        raise Error("sampled evaluate: 10 searches found nothing more than 1 (the search loop has no effect)")
    # the success rate is an f32 quotient of an integer count
    var q = Float32(Int(s10.success * 200.0 + 0.5)) / 200.0
    if f32_bits(q) != f32_bits(s10.success):
        raise Error("success is not n_ok / E in f32")
    # candidate order: success first, then the summed reward; the first of equal candidates wins
    if not eval_better(1.0, -5.0, 0.0, 3.0):
        raise Error("success must dominate the reward")
    if eval_better(0.0, 9.0, 1.0, -9.0):
        raise Error("a failure must not beat a success")
    if not eval_better(1.0, 0.5, 1.0, 0.4) or eval_better(1.0, 0.4, 1.0, 0.4) or eval_better(1.0, 0.3, 1.0, 0.4):
        raise Error("reward tie-break")
    print("  evaluate: accounting at difficulty 0, core independence, best-of-k order OK")


def test_promotion() raises:
    var env = make_env("permutation", "line:3", 1)
    var cfg = small_cfg(1)
    var tr = Trainer(env, make_net(env, 8, 1), cfg)
    var z = LossStats(0.0, 0.0, 0.0, 0.0)
    var rep170 = StepReport(EvalResult(Float32(170) / Float32(200), 0.0, 0), EvalResult(0.0, 0.0, 0), False, 1, z.copy(), 0, 0.0)
    var rep169 = StepReport(EvalResult(Float32(169) / Float32(200), 0.0, 0), EvalResult(0.0, 0.0, 0), False, 1, z.copy(), 0, 0.0)
    if tr.promote_if_ready(rep169) or tr.env.difficulty != 1:
        raise Error("169 / 200 must not promote")
    if not tr.promote_if_ready(rep170) or tr.env.difficulty != 2:
        raise Error("170 / 200 must promote (f32 0.85 is above 0.85)")
    var full = StepReport(EvalResult(1.0, 0.0, 0), EvalResult(0.0, 0.0, 0), False, 2, z.copy(), 0, 0.0)
    if not tr.promote_if_ready(full) or tr.env.difficulty != 3:
        raise Error("one promotion per call")
    # diff_max stops the curriculum, a promotion is never undone
    var cfg2 = small_cfg(1)
    cfg2.diff_max = 2
    var env2 = make_env("permutation", "line:3", 2)
    var tr2 = Trainer(env2, make_net(env2, 8, 1), cfg2)
    if tr2.promote_if_ready(full) or tr2.env.difficulty != 2:
        raise Error("difficulty must not exceed diff_max")
    var zero = StepReport(EvalResult(0.0, 0.0, 0), EvalResult(0.0, 0.0, 0), False, 2, z.copy(), 0, 0.0)
    if tr.promote_if_ready(zero) or tr.env.difficulty != 3:
        raise Error("a failing evaluation must not demote")
    print("  promotion rule (170 / 200 promotes, 169 does not, diff_max, no demotion) OK")


# ------------------------------------------------------------------------------------------------ update details


def test_sparse_adam_equals_dense() raises:
    """The trainer's Adam skips never-touched embedding rows; it must equal the dense optimiser bit for bit."""
    var env = make_env("pauli_network", "line:3", 4)
    var net_a = BasicNet(make_spec(env, 12, [8]), 5)
    var net_b = BasicNet(make_spec(env, 12, [8]), 5)
    var cfg = small_cfg(2, 40)
    cfg.num_epochs = 4
    var ro = collect(env, net_a, cfg, 0)
    var dense = Adam(net_a.param_count(), cfg.lr)
    var sparse = Adam(net_b.param_count(), cfg.lr)
    sparse.mark_sparse_rows(net_b.ps.offsets[0], net_b.ps.cols_of(0), net_b.ps.rows_of(0))
    for _ in range(3):
        _ = ppo_update(net_a, dense, ro, cfg)
        _ = ppo_update(net_b, sparse, ro, cfg)
    if not bits_equal(net_a.ps.data, net_b.ps.data):
        raise Error("sparse-row Adam differs from dense Adam")
    if sparse.touched_rows() == 0 or sparse.touched_rows() >= net_b.ps.rows_of(0):
        raise Error("touched row count: the test needs untouched rows to have power (" + String(sparse.touched_rows()) + ")")
    print("  sparse-row Adam equals dense Adam bit for bit (" + String(sparse.touched_rows()) + " of " + String(net_b.ps.rows_of(0)) + " rows touched) OK")


def test_mask_new_logits() raises:
    """`mask_new_logits` masks the new logits of solved rows: reproduce one epoch by hand and compare bit for bit."""
    var env = make_env("permutation", "line:4", 4)
    var net_a = make_net(env, 12, 9)
    var net_b = make_net(env, 12, 9)
    var cfg = small_cfg(1, 30)
    cfg.num_epochs = 1
    var ro = collect(env, net_a, cfg, 0)
    cfg.chunk_rows = ro.n
    cfg.mask_new_logits = True
    var masked_rows = 0
    for r in range(ro.n):
        if ro.allowed[r] == 0:
            masked_rows += 1
    if masked_rows == 0:
        raise Error("the rollout has no solved rows; the mask test has no power")
    var opt_a = Adam(net_a.param_count(), cfg.lr)
    _ = ppo_update(net_a, opt_a, ro, cfg)
    # by hand
    var n = ro.n
    var A = net_b.spec.num_actions
    var cc = ChunkCache(net_b.spec, n)
    net_b.forward_chunk(ro.sb, 0, n, cc)
    for r in range(n):
        if ro.allowed[r] == 0:
            for a in range(A):
                cc.logits[r * A + a] = MASKED_LOGIT
    var dz = List[Float32](length=n * A, fill=0.0)
    var dv = List[Float32](length=n, fill=0.0)
    var lc = PPOLossCfg(cfg.clip_ratio, cfg.vf_coef, cfg.ent_coef)
    _ = ppo_chunk_loss(
        cc.logits.unsafe_ptr().as_unsafe_any_origin(), cc.values.unsafe_ptr().as_unsafe_any_origin(), n, A,
        _ci(ro.acts), _cf(ro.advs), _cf(ro.rets), _cf(ro.old_lp), lc, Float32(1.0) / Float32(n),
        dz.unsafe_ptr().as_unsafe_any_origin(), dv.unsafe_ptr().as_unsafe_any_origin(),
    )
    for r in range(n):
        if ro.allowed[r] == 0:
            for a in range(A):
                dz[r * A + a] = 0.0
    var grad = List[Float32](length=net_b.param_count(), fill=0.0)
    net_b.backward_chunk(ro.sb, 0, n, cc, dz.unsafe_ptr().as_unsafe_any_origin(), dv.unsafe_ptr().as_unsafe_any_origin(), grad.unsafe_ptr().as_unsafe_any_origin())
    var opt_b = Adam(net_b.param_count(), cfg.lr)
    opt_b.step(net_b.ps.data, grad, cfg.lr)
    if not bits_equal(net_a.ps.data, net_b.ps.data):
        raise Error("mask_new_logits does not reproduce the hand-written masked update")
    # and it is a real switch: the faithful update differs
    var net_c = make_net(env, 12, 9)
    var cfg_f = small_cfg(1, 30)
    cfg_f.num_epochs = 1
    cfg_f.chunk_rows = ro.n
    var opt_c = Adam(net_c.param_count(), cfg_f.lr)
    _ = ppo_update(net_c, opt_c, ro, cfg_f)
    if bits_equal(net_a.ps.data, net_c.ps.data):
        raise Error("mask_new_logits changed nothing")
    print("  mask_new_logits reproduces the hand-written masked epoch (" + String(masked_rows) + " solved rows) OK")


# ------------------------------------------------------------------------------------------------ guards


def has_text(msg: String, needle: String) -> Bool:
    return msg.find(needle) >= 0


def test_status_slots() raises:
    if first_status([0, 0, 642, 0, 641]) != 642 or first_status([0, 0]) != 0 or first_status(List[Int32]()) != 0:
        raise Error("first_status")
    # a network with more actions than the environment: the action chosen can be invalid, which the env clamps and flags
    var env = make_env("permutation", "line:3", 2)  # 2 actions
    var spec = make_spec(env, 8, [8])
    spec.num_actions = 5
    spec.obs_perms = List[List[Int]]()
    spec.act_perms = List[List[Int]]()
    var bad = BasicNet(spec, 3)
    var ob = bad.ps.offset("action.0.bias")
    bad.ps.data[ob + 4] = 40.0
    bad.ps.data[ob + 3] = 39.0
    var ref_ro = Rollout()
    var counts: List[Int] = [1, 2, 4, 8]
    for ci in range(len(counts)):
        var cfg = small_cfg(counts[ci], 24)
        var ro = collect(env, bad, cfg, 0)
        if ro.status != 642:
            raise Error("collect: expected status 642, got " + String(ro.status))
        if ci == 0:
            ref_ro = ro^
        elif not rollouts_equal(ro, ref_ro):
            raise Error("a rollout with an error status depends on the core count")
        var ev = evaluate(env, bad, cfg, 0, True, 1, 20)
        if ev.status != 642:
            raise Error("evaluate: expected status 642, got " + String(ev.status))
    # the trainer refuses a network that does not match the environment before running anything
    var tr = Trainer(env, bad^, small_cfg(1, 8))
    var msg = String()
    try:
        _ = tr.learn_step()
    except e:
        msg = String(e)
    if not has_text(msg, "QM-E642"):
        raise Error("learn_step with a mismatched network must raise QM-E642, got: " + msg)
    print("  status slots: QM-E642 flagged deterministically for any core count; mismatched network refused OK")


def test_nan_guard() raises:
    var env = make_env("permutation", "line:3", 2)
    var net = make_net(env, 12, 6)
    var vb = net.ps.offset("value.0.bias")
    var nan = Float32(0.0) / Float32(0.0)
    net.ps.data[vb] = nan
    var before = net.ps.data.copy()
    var tr = Trainer(env, net^, small_cfg(2, 24))
    var msg = String()
    try:
        _ = tr.learn_step()
    except e:
        msg = String(e)
    if not has_text(msg, "QM-E643"):
        raise Error("a non-finite loss must raise QM-E643, got: " + msg)
    if tr.iteration != 0:
        raise Error("the iteration counter advanced on a failed step")
    var nonfinite = 0
    for i in range(len(tr.net.ps.data)):
        if f32_bits(tr.net.ps.data[i]) != f32_bits(before[i]):
            raise Error("the weights were not restored to their pre-update values")
        if tr.net.ps.data[i] != tr.net.ps.data[i]:
            nonfinite += 1
    if nonfinite != 1:
        raise Error("expected exactly the planted NaN in the restored weights, found " + String(nonfinite))
    if tr.opt.steps_taken() != 0:
        raise Error("the Adam step counter was not restored")
    for i in range(len(tr.opt.m)):
        if tr.opt.m[i] != 0.0 or tr.opt.v[i] != 0.0:
            raise Error("the Adam moments were not restored")
    if len(tr.evaluated) != len(before):
        raise Error("Trainer.evaluated holds the evaluated weights")
    print("  NaN guard: QM-E643, weights and Adam state restored OK")


def test_evaluated_snapshot() raises:
    """`Trainer.evaluated` is the pre-update weights, so the `best_checkpoint = evaluated` alternative is available."""
    var env = make_env("permutation", "line:3", 2)
    var net = make_net(env, 12, 6)
    var before = net.ps.data.copy()
    var tr = Trainer(env, net^, small_cfg(2, 24))
    _ = tr.learn_step()
    if not bits_equal(tr.evaluated, before):
        raise Error("Trainer.evaluated is not the weights the iteration evaluated")
    if bits_equal(tr.net.ps.data, before):
        raise Error("the update did not change the weights")
    print("  Trainer.evaluated holds the pre-update weights OK")


def main() raises:
    test_update_golden()
    test_normalize_variant()
    test_update_thread_independence()
    test_sparse_adam_equals_dense()
    test_mask_new_logits()
    test_collector_consistency()
    test_stage_seeds()
    test_evaluate()
    test_promotion()
    test_status_slots()
    test_nan_guard()
    test_evaluated_snapshot()
    test_t_det()
    print("test_rl_ppo OK")
