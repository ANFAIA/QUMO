"""Rollout-statistics test for qrl.ppo (MOJO_PORT.md 13.1 and 13.2 `test_rl_ppo_slow`, part of acceptance A4).

`ppo/rollout_stats.json` was produced offline with the real qiskit-gym environments and the real twisterl collector and
evaluator on fixed explicit f32 weights, for `permutation line:4`, `linear_function ring:4`, `clifford line:3` and
`pauli_network line:3` at three difficulties each. This test loads the same weights, runs the Mojo `collect` (5,000
episodes) and `evaluate` (20,000 episodes, deterministic and sampled with 10 searches, as 20 blocks of 1,000; 200,000
for `pauli_network line:3` d=16, whose reference holds 200,000 deterministic and 1,000,000 sampled episodes), and
requires, at 4 sigma:

  * every collector mean (`rets`, `advs`, `values`, `rewards`, `old_lp`, fraction of all-masked rows): two-sample
    z = (m_mojo - m_ref) / sqrt(se_ref^2 + se_mojo^2) with the cluster-robust (per-episode) ratio estimator of the
    fixture on both sides;
  * the episode-length histogram and the recorded twist histogram: homogeneity chi-square at z <= 4 (bins pooled to an
    expected count of 5);
  * the fraction of solved episodes: binomial 4 sigma;
  * `evaluate` success (se = max(se_blocks, se_binomial)) and mean reward (se_blocks), both modes.

It pins what the unit goldens cannot see: reward alignment (row t carries the reward of state t), no bootstrap on the
terminal row, the all-masked terminal mask, the twist drawn per decision, and the reward accounting of `evaluate`. The
Mojo RNG streams differ from the reference by design (D3), so the comparison is statistical with fixed seeds.
"""

from std.math import sqrt
from qrl.json import JsonValue
from qrl.model import model_kind_code
from qrl.topology import parse_topology
from qrl.gateset import EnvConfig, build_gateset
from qrl.synth_env import TrainEnv
from qrl.nn import NetSpec, BasicNet
from qrl.ppo import PPOConfig, Rollout, collect, evaluate
from qrl.testutil import (
    load_fixture,
    flat_floats32,
    abs64,
    two_sample_z,
    cluster_mean_se,
    chi_square_homogeneity,
)

comptime Z_MAX = 4.0
comptime MAX_EVAL_BLOCKS = 200


def make_env(kind_name: String, topo: String, difficulty: Int) raises -> TrainEnv:
    var t = parse_topology(topo)
    var cfg = EnvConfig()
    cfg.kind = model_kind_code(kind_name)
    cfg.n = t.n
    cfg.actions = build_gateset(cfg.kind, t.n, t.edges)
    cfg.difficulty = difficulty
    return TrainEnv(cfg, 1)


def build_net(env: TrainEnv, c: JsonValue) raises -> BasicNet:
    """The fixture's explicit weights (torch layout, flat row-major) in the container layout; the symmetry tables come
    from the environment, exactly as the trainer builds them."""
    var tw = env.twists()
    var spec = NetSpec()
    spec.obs_size = env.obs_size()
    spec.num_actions = env.num_actions()
    spec.emb = c.get("emb").as_int()
    var common = c.get("common")
    for i in range(len(common)):
        spec.common.append(common.at(i).as_int())
    spec.obs_perms = tw.obs_perms.copy()
    spec.act_perms = tw.act_perms.copy()
    if spec.obs_size != c.get("obs_size").as_int() or spec.num_actions != c.get("num_actions").as_int():
        raise Error("fixture and environment disagree on obs_size / num_actions")
    if len(spec.obs_perms) != c.get("n_twists").as_int():
        raise Error("fixture and environment disagree on the number of twists: " + String(len(spec.obs_perms)))
    var net = BasicNet(spec, 0, False)
    var w = c.get("weights")
    for i in range(net.ps.count()):
        var name = net.ps.names[i]
        var vals = flat_floats32(w.get(name).get("data"))
        var off = net.ps.offsets[i]
        if len(vals) != net.ps.size_of(i):
            raise Error("tensor " + name + ": wrong size")
        if name == "embeddings.weight":  # torch [E][OBS] -> container [OBS][E]
            var obs = net.ps.rows_of(i)
            var e = net.ps.cols_of(i)
            for o in range(e):
                for k in range(obs):
                    net.ps.data[off + k * e + o] = vals[o * obs + k]
        else:
            for j in range(len(vals)):
                net.ps.data[off + j] = vals[j]
    return net^


def binomial_se(p: Float64, n: Int) -> Float64:
    var v = p * (1.0 - p)
    var floor = 0.25 / Float64(n)
    if v < floor:
        v = floor
    return sqrt(v / Float64(n))


def hist_arrays(ref_hist: JsonValue, mine: List[Int]) raises -> Tuple[List[Int], List[Int]]:
    """Aligns the reference histogram (a JSON object, key = episode length) with ours over 0..max."""
    var keys = ref_hist.keys()
    var hi = len(mine) - 1
    for i in range(len(keys)):
        var k = Int(keys[i])
        if k > hi:
            hi = k
    var a = List[Int](length=hi + 1, fill=0)
    var b = List[Int](length=hi + 1, fill=0)
    for i in range(len(keys)):
        a[Int(keys[i])] = ref_hist.get(keys[i]).as_int()
    for i in range(len(mine)):
        b[i] = mine[i]
    return (a^, b^)


struct Zlog(Movable):
    var worst: Float64
    var worst_what: String
    var count: Int

    def __init__(out self):
        self.worst = 0.0
        self.worst_what = String()
        self.count = 0

    def add(mut self, z: Float64, what: String) raises:
        self.count += 1
        if abs64(z) > self.worst:
            self.worst = abs64(z)
            self.worst_what = what
        if abs64(z) > Z_MAX:
            raise Error("rollout statistics: |z| = " + String(z) + " > 4 for " + what)


def check_collector(
    env: TrainEnv, net: BasicNet, d: JsonValue, tag: String, seed: UInt64, mut zl: Zlog
) raises:
    var col = d.get("collector")
    var episodes = col.get("episodes").as_int()
    var cfg = PPOConfig()
    cfg.num_episodes = episodes
    cfg.gamma = 0.995
    cfg.lam = 0.995
    cfg.seed = seed
    cfg.num_cores = 4
    var ro = collect(env, net, cfg, 0)
    if ro.status != 0:
        raise Error(tag + ": collector status " + String(ro.status))
    var names: List[String] = ["rets", "advs", "values", "rewards", "old_lp", "all_masked"]
    var sums = List[List[Float64]]()
    for _ in range(len(names)):
        sums.append(List[Float64](length=episodes, fill=0.0))
    var counts = List[Float64](length=episodes, fill=0.0)
    var len_hist = List[Int](length=1, fill=0)
    var solved = 0
    var row = 0
    for e in range(episodes):
        var L = Int(ro.ep_len[e])
        counts[e] = Float64(L)
        for _ in range(L):
            sums[0][e] += Float64(ro.rets[row])
            sums[1][e] += Float64(ro.advs[row])
            sums[2][e] += Float64(ro.values[row])
            sums[3][e] += Float64(ro.rewards[row])
            sums[4][e] += Float64(ro.old_lp[row])
            if ro.allowed[row] == 0:
                sums[5][e] += 1.0
            row += 1
        if ro.allowed[row - 1] == 0:
            solved += 1
        while len(len_hist) <= L:
            len_hist.append(0)
        len_hist[L] += 1
    var st = col.get("stats")
    for v in range(len(names)):
        var mse = cluster_mean_se(sums[v], counts)
        var ref_m = st.get(names[v]).get("mean").as_float()
        var ref_se = st.get(names[v]).get("se").as_float()
        zl.add(two_sample_z(mse[0], mse[1], ref_m, ref_se), tag + " collector " + names[v] + " (" + String(mse[0]) + " vs " + String(ref_m) + ")")
    # episode-length histogram
    var h = hist_arrays(col.get("len_hist"), len_hist)
    var c2 = chi_square_homogeneity(h[0], h[1])
    zl.add(c2.z, tag + " collector length histogram (chi2 " + String(c2.stat) + ", dof " + String(c2.dof) + ")")
    # solved-episode fraction
    var p = Float64(solved) / Float64(episodes)
    var sf = col.get("solved_episode_fraction")
    zl.add(
        two_sample_z(p, binomial_se(p, episodes), sf.get("mean").as_float(), sf.get("se").as_float()),
        tag + " collector solved-episode fraction (" + String(p) + " vs " + String(sf.get("mean").as_float()) + ")",
    )
    # recorded twist histogram (index -1 is "none"): shift by one
    var tc = col.get("twist_counts")
    var tkeys = tc.keys()
    var tmine = List[Int](length=1, fill=0)
    for r in range(ro.n):
        var t = Int(ro.sb.perm[r]) + 1
        while len(tmine) <= t:
            tmine.append(0)
        tmine[t] += 1
    var ta = List[Int](length=len(tmine), fill=0)
    for i in range(len(tkeys)):
        var k = Int(tkeys[i]) + 1
        while len(ta) <= k:
            ta.append(0)
        ta[k] = tc.get(tkeys[i]).as_int()
    while len(tmine) < len(ta):
        tmine.append(0)
    var c3 = chi_square_homogeneity(ta, tmine)
    zl.add(c3.z, tag + " collector twist histogram")


def check_eval(
    env: TrainEnv, net: BasicNet, rf: JsonValue, tag: String, deterministic: Bool, searches: Int, seed: UInt64, mut zl: Zlog
) raises:
    # The reference runs 20 blocks of 1,000 episodes, except pauli_network line:3 d=16 (200 deterministic, 1,000 sampled
    # blocks: its success rate is about 5.5e-4, so 20,000 episodes see only about 11 successes). Our side runs at most
    # 200 blocks (200,000 episodes, about 15 s sampled on 4 cores); the two-sample z uses each side's own error.
    var per = rf.get("episodes").as_int() // rf.get("blocks").as_int()
    var blocks = min(rf.get("blocks").as_int(), MAX_EVAL_BLOCKS)
    var cfg = PPOConfig()
    cfg.seed = seed
    cfg.num_cores = 4
    var sv = List[Float64]()
    var rv = List[Float64]()
    for b in range(blocks):
        var r = evaluate(env, net, cfg, b, deterministic, searches, per)
        if r.status != 0:
            raise Error(tag + ": evaluate status " + String(r.status))
        sv.append(Float64(r.success))
        rv.append(Float64(r.reward))
    var n_total = blocks * per
    var ms: Float64 = 0.0
    var mr: Float64 = 0.0
    for b in range(blocks):
        ms += sv[b]
        mr += rv[b]
    ms /= Float64(blocks)
    mr /= Float64(blocks)
    var vs: Float64 = 0.0
    var vr: Float64 = 0.0
    for b in range(blocks):
        vs += (sv[b] - ms) * (sv[b] - ms)
        vr += (rv[b] - mr) * (rv[b] - mr)
    var se_bs = sqrt(vs / Float64(blocks - 1)) / sqrt(Float64(blocks))
    var se_br = sqrt(vr / Float64(blocks - 1)) / sqrt(Float64(blocks))
    var se_s = max(se_bs, binomial_se(ms, n_total))
    var rs = rf.get("success")
    var ref_se_s = max(rs.get("se_blocks").as_float(), rs.get("se_binomial").as_float())
    zl.add(
        two_sample_z(ms, se_s, rs.get("mean").as_float(), ref_se_s),
        tag + " evaluate success (" + String(ms) + " vs " + String(rs.get("mean").as_float()) + ")",
    )
    var rr = rf.get("reward")
    # Floor on the standard errors: when every episode has the same total (a failing policy on a Pauli network) the
    # block spread is zero and the two means differ only by f32 summation rounding (relative 1e-6), not statistically.
    var ref_mr = rr.get("mean").as_float()
    var se_r_mine = max(se_br, 1e-6 * abs64(mr))
    var se_r_ref = max(rr.get("se_blocks").as_float(), 1e-6 * abs64(ref_mr))
    zl.add(
        two_sample_z(mr, se_r_mine, ref_mr, se_r_ref),
        tag + " evaluate reward (" + String(mr) + " vs " + String(rr.get("mean").as_float()) + ")",
    )


def main() raises:
    var fx = load_fixture("ppo/rollout_stats.json")
    var configs = fx.get("configs")
    var zl = Zlog()
    for ci in range(len(configs)):
        var c = configs.at(ci)
        var kind = c.get("kind").as_string()
        var topo = c.get("topology").as_string()
        var diffs = c.get("difficulties")
        for di in range(len(diffs)):
            var d = diffs.at(di)
            var difficulty = d.get("difficulty").as_int()
            var env = make_env(kind, topo, difficulty)
            var net = build_net(env, c)
            var tag = kind + " " + topo + " d=" + String(difficulty)
            var seed = UInt64(1000 + 10 * ci + di)
            check_collector(env, net, d, tag, seed, zl)
            check_eval(env, net, d.get("evaluate_deterministic"), tag + " det", True, 1, seed + 100, zl)
            check_eval(env, net, d.get("evaluate_sampled10"), tag + " sampled10", False, 10, seed + 200, zl)
            print("  " + tag + ": ok (worst |z| so far " + String(zl.worst) + ")")
    print("  " + String(zl.count) + " statistics compared, worst |z| = " + String(zl.worst) + " (" + zl.worst_what + ")")
    print("test_rl_ppo_slow OK")
