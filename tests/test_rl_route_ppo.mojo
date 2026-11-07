"""Tests for the router PPO fine-tune: qumo.rl.route_ppo and the `Episode` decision split of qumo.compiler.ai_route.

Pinned here (MOJO_PORT.md 13.2, `test_rl_route_ppo`):
  * `ppo_route_golden.json` through the integrated trainer code path (`ppo_minibatch_grad` / `ppo_apply_update`
    over the virtual workers): loss parts, ratios, raw gradients, the global clip norm and coefficient, the weights
    after three Adam steps, and the standardised advantages of the GAE case;
  * the inverse-CDF sampler (boundaries, frequencies);
  * the split `Episode.pre_decision / needs_policy / act` against the fused `run` (same actions, overhead, layout),
    and the independence of the overhead from `rec_ops`;
  * the rollout against an independent replay: observations, valid-slot counts, actions, per-decision rewards from
    overhead and 2q-gate counts at the decision boundaries, final overheads, f64 policy log-probabilities and values
    against the runtime `RouteModel.forward`, GAE and standardisation recomputed in closed form, and a z-test that the
    actions are drawn from the policy;
  * bit-identical rollouts and updates for 1, 2 and 4 workers;
  * a first-order sanity check (small steps reduce the PPO loss on a fixed rollout);
  * the trainer end to end: 2 iterations from the shipped router, worker independence, a run directory that loads
    (container, route model, `RouterNet.from_model`) and exports through `export_route_run`;
  * error paths: option validation (QM-E640), a non-router init (QM-E647), the divergence guard (QM-E643 stop at the
    last good weights).
The golden step uses explicit minibatches, so it pins the loss / gradient / Adam arithmetic, not the shuffling.
"""

from std.math import exp, log, sqrt
from std.memory import bitcast
from std.os import makedirs
from qumo.ai.json import JsonValue, parse_json
from qumo.compiler.policy import ModelFile, load_model_file
from qumo.compiler.ai_route import (
    RProblem,
    Episode,
    RouteModel,
    load_route_model,
    episode_max_steps,
    _random_layout,
    MODE_POLICY,
)
from qumo.sim.rng import Rng
from qumo.rl.optim import Adam
from qumo.rl.router_net import RouterNet, RouterCache
from qumo.rl.router_loss import gae_route
from qumo.rl.targets import RTopo, topology_by_name
from qumo.rl.jobs import build_jobs
from qumo.rl.export_route import export_route_run
from qumo.rl.route_ppo import (
    RoutePPOConfig,
    RoutePPOWorkspace,
    RouteRollout,
    RouteJobs,
    RouteStepStats,
    route_jobs_of,
    collect_route,
    ppo_minibatch_grad,
    ppo_apply_update,
    ppo_minibatch_step,
    route_ppo_update,
    train_route_ppo,
    write_ppo_run,
    validate_config,
    standardize,
    sample_cdf,
    episode_seed,
    actor_seed,
    curriculum_scale,
    job_build_seed,
    RPPO_VW,
    RPPO_CHUNK_ROWS,
)
from qumo.rl.testutil import (
    load_fixture,
    ints,
    floats,
    flat_floats,
    flat_floats32,
    assert_close,
    assert_list_close,
    assert_list_close32,
)

comptime OUT_DIR = "build/test_rl_route_ppo"
comptime SHIPPED = "models/ai_transpiler/route/route_mlp_h64_v1"


def _bits_equal(a: List[Float32], b: List[Float32]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            return False
    return True


def _flat_params(v: JsonValue, h: Int) raises -> List[Float32]:
    """The six route tensors (emb, b0, w1, b1, wv, bv) flattened row-major."""
    var sizes: List[Int] = [128 * h, h, 16 * h, 16, h, 1]
    var names: List[String] = ["emb", "b0", "w1", "b1", "wv", "bv"]
    var out = List[Float32]()
    for t in range(6):
        var f = flat_floats32(v.get(names[t]))
        if len(f) != sizes[t]:
            raise Error("tensor " + names[t] + " has " + String(len(f)) + " values, expected " + String(sizes[t]))
        for i in range(len(f)):
            out.append(f[i])
    return out^


def _net_with(h: Int, params: List[Float32]) -> RouterNet:
    var net = RouterNet(h, 0)
    net.params = params.copy()
    return net^


def _random_net(h: Int, seed: UInt64) -> RouterNet:
    """A random router with non-zero hidden biases (the init has none)."""
    var net = RouterNet(h, seed)
    var rng = Rng(seed + 1000)
    for j in range(h):
        net.params[net.off_b0() + j] = Float32(rng.uniform() - 0.5) * 0.2
    return net^


def _small_jobs(seed: Int, n: Int, scale: Float64 = 0.5) raises -> RouteJobs:
    var topos = List[RTopo]()
    topos.append(topology_by_name("grid3x3"))
    topos.append(topology_by_name("ring8"))
    topos.append(topology_by_name("line8"))
    topos.append(topology_by_name("heavy_hex19"))
    var js = build_jobs(seed, n, topos, List[String](), 9, scale, 0.4)
    return route_jobs_of(js)


def _rollout_equal(a: RouteRollout, b: RouteRollout) -> Bool:
    if a.n != b.n or len(a.sign) != len(b.sign):
        return False
    for i in range(len(a.sign)):
        if a.sign[i] != b.sign[i]:
            return False
    for i in range(a.n):
        if a.A[i] != b.A[i] or a.act[i] != b.act[i]:
            return False
    if not _bits_equal(a.logp_old, b.logp_old) or not _bits_equal(a.adv, b.adv) or not _bits_equal(a.ret, b.ret):
        return False
    for e in range(len(a.ep_overhead)):
        if a.ep_overhead[e] != b.ep_overhead[e] or a.ep_len[e] != b.ep_len[e]:
            return False
    return True


# ------------------------------------------------------------------------------------------------ golden


def test_golden() raises:
    var g = load_fixture("torch/ppo_route_golden.json")
    var h = g.get("H").as_int()
    var lr = g.get("lr").as_float()
    var cfg = RoutePPOConfig()
    cfg.iters = 1
    cfg.lr = lr
    cfg.clip = Float32(g.get("clip").as_float())
    cfg.ent_coef = Float32(g.get("ent").as_float())
    cfg.max_norm = g.get("max_norm").as_float()
    var steps = g.get("steps")
    var net = _net_with(h, _flat_params(g.get("init"), h))
    var opt = Adam(net.param_count(), lr)
    var ws = RoutePPOWorkspace(h, net.param_count(), 16)
    # the same steps through the one-call entry, on a second copy, must give the same bits
    var net2 = _net_with(h, _flat_params(g.get("init"), h))
    var opt2 = Adam(net2.param_count(), lr)
    var ws2 = RoutePPOWorkspace(h, net2.param_count(), 16)
    for s in range(len(steps)):
        var st = steps.at(s)
        var rows = len(st.get("A"))
        var ro = RouteRollout()
        ro.n = rows
        var sg = flat_floats(st.get("obs_sign"))
        for i in range(len(sg)):
            ro.sign.append(Int8(1) if sg[i] > 0.0 else (Int8(-1) if sg[i] < 0.0 else Int8(0)))
        var a_cnt = ints(st.get("A"))
        var act_i = ints(st.get("action"))
        for r in range(rows):
            ro.A.append(Int32(a_cnt[r]))
            ro.act.append(Int32(act_i[r]))
        ro.logp_old = flat_floats32(st.get("logp_old"))
        ro.adv = flat_floats32(st.get("adv"))
        ro.ret = flat_floats32(st.get("ret"))
        var order = List[Int32]()
        for r in range(rows):
            order.append(Int32(r))
        var parts = ppo_minibatch_grad(net, ws, ro, order, 0, rows, cfg, 4)
        var B = Float64(rows)
        var lt = st.get("loss")
        var pl = parts.policy / B
        var vl = parts.value / B
        var en = parts.entropy / B
        assert_close(pl, lt.get("policy").as_float(), 2e-5, 2e-5, "policy loss step " + String(s))
        assert_close(vl, lt.get("value").as_float(), 2e-5, 2e-5, "value loss step " + String(s))
        assert_close(en, lt.get("entropy").as_float(), 2e-5, 2e-5, "entropy step " + String(s))
        assert_close(
            pl + Float64(cfg.vf_coef) * vl - Float64(cfg.ent_coef) * en,
            lt.get("total").as_float(),
            2e-5,
            2e-5,
            "total loss step " + String(s),
        )
        var ratio = List[Float32]()
        for r in range(rows):
            ratio.append(ws.last_ratio[r])
        assert_list_close32(ratio, flat_floats32(st.get("ratio")), 2e-6, 2e-5, "ratios step " + String(s))
        assert_list_close32(ws.grad, _flat_params(st.get("grads_raw"), h), 3e-6, 2e-4, "raw grads step " + String(s))
        var norm = ppo_apply_update(net, opt, ws, lr, cfg.max_norm)
        assert_close(norm, st.get("grad_norm_before_clip").as_float(), 0.0, 1e-4, "grad norm step " + String(s))
        assert_list_close32(ws.grad, _flat_params(st.get("grads_clipped"), h), 3e-6, 2e-4, "clipped grads step " + String(s))
        assert_list_close32(net.params, _flat_params(st.get("weights_after"), h), 3e-5, 0.0, "weights step " + String(s))
        var stats = ppo_minibatch_step(net2, opt2, ws2, ro, order, 0, rows, cfg, lr, 1)
        if not _bits_equal(net.params, net2.params):
            raise Error("ppo_minibatch_step differs from grad + apply at step " + String(s))
        assert_close(stats.grad_norm, norm, 0.0, 1e-12, "step stats norm")
        assert_close(stats.policy, pl, 1e-12, 1e-12, "step stats policy")
        # the golden clip fires every step: coefficient 0.5 / (norm + 1e-6)
        assert_close(0.5 / (norm + 1e-6), st.get("clip_coef").as_float(), 0.0, 1e-4, "clip coef step " + String(s))
    # standardised advantages of the GAE case (population std + 1e-8), returns keep the raw advantage
    var ge = g.get("gae")
    var rew = floats(ge.get("reward"))
    var val = floats(ge.get("value"))
    var dn = List[Bool]()
    var dj = ge.get("done")
    for i in range(len(dj)):
        dn.append(dj.at(i).as_bool())
    var adv = List[Float64]()
    var ret = List[Float64]()
    gae_route(Span(rew), Span(val), Span(dn), ge.get("gamma").as_float(), ge.get("lam").as_float(), adv, ret)
    var norm_adv = standardize(adv)
    var want = floats(ge.get("adv_normalized"))
    for i in range(len(want)):
        assert_close(Float64(norm_adv[i]), want[i], 1e-6, 1e-6, "standardised advantage " + String(i))


# ------------------------------------------------------------------------------------------------ sampler


def test_sample_cdf() raises:
    var p: List[Float64] = [0.2, 0.3, 0.5, 0.0]
    # right-sided search on the cumulative sum [0.2, 0.5, 1.0]: u = 0.2 belongs to the second entry
    var cases: List[Float64] = [0.0, 0.1999999, 0.2, 0.4999999, 0.5, 0.9999999]
    var want: List[Int] = [0, 0, 1, 1, 2, 2]
    for i in range(len(cases)):
        if sample_cdf(p, 3, cases[i]) != want[i]:
            raise Error("sample_cdf(u=" + String(cases[i]) + ") = " + String(sample_cdf(p, 3, cases[i])) + ", want " + String(want[i]))
    # unnormalised probabilities are normalised by the last cumulative value
    var q: List[Float64] = [2.0, 2.0]
    if sample_cdf(q, 2, 0.49) != 0 or sample_cdf(q, 2, 0.5) != 1:
        raise Error("sample_cdf does not normalise by the cumulative sum")
    if sample_cdf(q, 1, 0.99) != 0:
        raise Error("sample_cdf with one entry")
    var total: List[Float64] = [0.1, 0.2, 0.3, 0.4]
    var rng = Rng(5)
    var cnt = List[Int](length=4, fill=0)
    var n = 40000
    for _ in range(n):
        cnt[sample_cdf(total, 4, rng.uniform())] += 1
    for k in range(4):
        var pk = total[k]
        var z = (Float64(cnt[k]) - Float64(n) * pk) / sqrt(Float64(n) * pk * (1.0 - pk))
        if z > 4.0 or z < -4.0:
            raise Error("sample_cdf frequency of " + String(k) + " off by z=" + String(z))


# ------------------------------------------------------------------------------------------------ episode split


def test_episode_split() raises:
    var jobs = _small_jobs(11, 24, 0.6)
    var model = RouteModel.random(16, 3)
    for j in range(jobs.num_jobs()):
        var circ = jobs.circuits[jobs.jc[j]].copy()
        var target = jobs.targets[jobs.jt[j]].copy()
        var prob = RProblem(circ.ops, circ.num_qubits)
        var N = target.num_qubits
        var r1 = Rng(UInt64(100 + j))
        var l2p = _random_layout(N, r1)
        var seed = UInt64(7000 + j)
        var a = Episode(prob, target, l2p, seed, True)
        a.run(prob, target, model, MODE_POLICY, 1.0, episode_max_steps(prob, target))
        # split loop with the same policy decision (first maximum of the logits among the active swaps)
        var b = Episode(prob, target, l2p, seed, True)
        b.rec_ops = False
        var limit = max(4, 2 * prob.n)
        var ms = episode_max_steps(prob, target)
        var hid = List[Float64](length=model.hidden, fill=0.0)
        var logits = List[Float64](length=16, fill=0.0)
        while True:
            var ex = b.pre_decision(prob, target, limit)
            if ex < 0:
                break
            var e: Int
            if b.needs_policy():
                model.forward(b.obs, hid, logits)
                var best = 0
                for k in range(1, len(b.active)):
                    if logits[k] > logits[best]:
                        best = k
                e = b.active[best]
            else:
                e = b.fallback_swap(prob, target)
                b.n_fallback += 1
                if e < 0:
                    b.failed = True
                    break
            b.act(prob, target, e)
            if b.steps > ms:
                b.failed = True
                break
        if len(a.actions) != len(b.actions):
            raise Error("job " + String(j) + ": split loop took " + String(len(b.actions)) + " actions, run " + String(len(a.actions)))
        for k in range(len(a.actions)):
            if a.actions[k] != b.actions[k]:
                raise Error("job " + String(j) + ": action " + String(k) + " differs")
        if a.overhead() != b.overhead() or a.steps != b.steps or a.failed != b.failed or a.n_fallback != b.n_fallback:
            raise Error("job " + String(j) + ": overhead / steps / fallback differ (rec_ops False vs True)")
        for w in range(N):
            if a.l2p[w] != b.l2p[w]:
                raise Error("job " + String(j) + ": final layout differs")


# ------------------------------------------------------------------------------------------------ rollout replay


def test_rollout_replay() raises:
    var h = 16
    var net = _random_net(h, 4)
    var cfg = RoutePPOConfig()
    cfg.iters = 1
    cfg.seed = 5
    cfg.workers = 4
    var jobs = _small_jobs(21, 30)
    var git = 3
    var ro = collect_route(net, jobs, cfg, git)
    var E = jobs.num_jobs()
    if len(ro.ep_len) != E or ro.n < 200:
        raise Error("rollout has " + String(ro.n) + " rows for " + String(E) + " episodes")
    # the runtime's float64 forward, with this network's weights, as the independent policy
    var model = RouteModel()
    model.ok = True
    model.hidden = h
    for i in range(128 * h):
        model.emb.append(net.params[net.off_emb() + i])
    for i in range(h):
        model.b0.append(net.params[net.off_b0() + i])
    for i in range(16 * h):
        model.w1.append(net.params[net.off_w1() + i])
    for i in range(16):
        model.b1.append(net.params[net.off_b1() + i])
    var obs = List[Int](length=128, fill=0)
    var hid = List[Float64](length=h, fill=0.0)
    var lg = List[Float64](length=16, fill=0.0)
    var row = 0
    var zsum = 0.0
    var zvar = 0.0
    var total_decisions = 0
    for e in range(E):
        var circ = jobs.circuits[jobs.jc[e]].copy()
        var target = jobs.targets[jobs.jt[e]].copy()
        var prob = RProblem(circ.ops, circ.num_qubits)
        var seed = episode_seed(cfg, git, e)
        var lrng = Rng(seed)
        var l2p = _random_layout(target.num_qubits, lrng)
        var ep = Episode(prob, target, l2p, seed, True)  # rec_ops stays True: the rollout used False
        var limit = max(4, 2 * prob.n)
        var ms = episode_max_steps(prob, target)
        var L = Int(ro.ep_len[e])
        var ov_at = List[Int]()
        var ex_at = List[Int]()
        var k = 0
        while True:
            var r = ep.pre_decision(prob, target, limit)
            if r < 0 or ep.steps > ms:
                break
            if ep.needs_policy():
                if k >= L:
                    raise Error("episode " + String(e) + " asks for more decisions than recorded")
                var A = len(ep.active)
                if Int(ro.A[row + k]) != A:
                    raise Error("episode " + String(e) + " decision " + String(k) + ": valid slots differ")
                for i in range(128):
                    var o = ep.obs[i]
                    var sv = 1 if o > 0 else (-1 if o < 0 else 0)
                    if Int(ro.sign[(row + k) * 128 + i]) != sv:
                        raise Error("episode " + String(e) + " decision " + String(k) + ": observation differs")
                    obs[i] = sv
                # independent policy: runtime float64 forward, softmax over the first A logits
                model.forward(obs, hid, lg)
                var mx = lg[0]
                for q in range(1, A):
                    mx = max(mx, lg[q])
                var sm = 0.0
                for q in range(A):
                    sm += exp(lg[q] - mx)
                var pa = exp(lg[Int(ro.act[row + k])] - mx) / sm
                assert_close(Float64(ro.logp_old[row + k]), log(pa + 1e-12), 1e-6, 1e-6, "logp_old")
                var v = Float64(net.params[net.off_bv()])
                for j in range(h):
                    if hid[j] > 0.0:  # RouteModel.forward leaves the pre-activation in `hid`
                        v += Float64(net.params[net.off_wv() + j]) * hid[j]
                assert_close(ro.value[row + k], v, 1e-9, 1e-9, "value")
                # sampler statistics: X = p[a] has mean sum p^2 and variance sum p^3 - (sum p^2)^2
                var s2 = 0.0
                var s3 = 0.0
                for q in range(A):
                    var pq = exp(lg[q] - mx) / sm
                    s2 += pq * pq
                    s3 += pq * pq * pq
                zsum += pa - s2
                zvar += s3 - s2 * s2
                total_decisions += 1
                ov_at.append(ep.overhead())
                ex_at.append(ep.exec2q)
                ep.act(prob, target, ep.active[Int(ro.act[row + k])])
                k += 1
            else:
                var fb = ep.fallback_swap(prob, target)
                ep.n_fallback += 1
                if fb < 0:
                    break
                ep.act(prob, target, fb)
        if k != L:
            raise Error("episode " + String(e) + ": replay made " + String(k) + " decisions, rollout has " + String(L))
        if ep.overhead() != Int(ro.ep_overhead[e]) or ep.steps != Int(ro.ep_steps[e]):
            raise Error("episode " + String(e) + ": final overhead / steps differ from the rollout")
        # rewards from the boundary states: the span of a decision ends at the next decision or at the end
        for q in range(L):
            var ov_end = ov_at[q + 1] if q + 1 < L else ep.overhead()
            var ex_end = ex_at[q + 1] if q + 1 < L else ep.exec2q
            var want = -Float64(ov_end - ov_at[q]) + 0.01 * Float64(ex_end - ex_at[q])
            assert_close(ro.reward[row + q], want, 1e-12, 1e-12, "reward of decision " + String(q) + " of episode " + String(e))
            if ro.done[row + q] != (q == L - 1):
                raise Error("done flag")
        row += L
    if row != ro.n:
        raise Error("rows " + String(row) + " vs " + String(ro.n))
    # the actions are draws from the policy: z-test on the picked probabilities
    var z = zsum / sqrt(zvar)
    if z > 4.0 or z < -4.0:
        raise Error("sampled actions do not follow the policy: z = " + String(z) + " over " + String(total_decisions) + " decisions")
    # GAE and standardisation in closed form (float64, reset at every episode end)
    var adv = List[Float64](length=ro.n, fill=0.0)
    var last = 0.0
    var i = ro.n - 1
    while i >= 0:
        var end = ro.done[i]
        var nv = 0.0 if end else ro.value[i + 1]
        var delta = ro.reward[i] + cfg.gamma * nv - ro.value[i]
        last = delta + cfg.gamma * cfg.lam * (0.0 if end else last)
        adv[i] = last
        i -= 1
    var m = 0.0
    for q in range(ro.n):
        m += adv[q]
    m /= Float64(ro.n)
    var vv = 0.0
    for q in range(ro.n):
        vv += (adv[q] - m) * (adv[q] - m)
    var sd = sqrt(vv / Float64(ro.n))
    for q in range(ro.n):
        assert_close(Float64(ro.adv[q]), (adv[q] - m) / (sd + 1e-8), 1e-5, 1e-5, "standardised advantage")
        assert_close(Float64(ro.ret[q]), adv[q] + ro.value[q], 1e-4, 1e-6, "return = raw advantage + value")


# ------------------------------------------------------------------------------------------------ determinism


def test_thread_independence() raises:
    var h = 24
    var base = _random_net(h, 8)
    var jobs = _small_jobs(31, 90, 0.8)
    var ref_ro = RouteRollout()
    var have_ref = False
    var cfg = RoutePPOConfig()
    cfg.iters = 1
    cfg.seed = 9
    # collection: identical rows for every worker count
    for w in range(3):
        cfg.workers = 1 if w == 0 else (2 if w == 1 else 4)
        var ro = collect_route(base, jobs, cfg, 2)
        if not have_ref:
            ref_ro = ro^
            have_ref = True
        elif not _rollout_equal(ref_ro, ro):
            raise Error("rollout differs with " + String(cfg.workers) + " workers")
    if ref_ro.n < 3 * 1024:
        raise Error("rollout of " + String(ref_ro.n) + " rows is too small to cover several minibatches")
    # update: several minibatches (1024 rows, last short) of several chunks, identical weights for 1, 2 and 4 workers
    var ref_params = List[Float32]()
    for w in range(3):
        cfg.workers = 1 if w == 0 else (2 if w == 1 else 4)
        var net = _net_with(h, base.params)
        var opt = Adam(net.param_count(), cfg.lr)
        var ws = RoutePPOWorkspace(h, net.param_count(), cfg.minibatch)
        var st = route_ppo_update(net, opt, ws, ref_ro, cfg, 2)
        if not (st.value == st.value) or st.grad_norm <= 0.0:
            raise Error("update statistics are not finite")
        if w == 0:
            ref_params = net.params.copy()
            if _bits_equal(ref_params, base.params):
                raise Error("the update did not change the weights")
        elif not _bits_equal(ref_params, net.params):
            raise Error("updated weights differ with " + String(cfg.workers) + " workers")
    # a rerun of the collection is identical too
    var again = collect_route(base, jobs, cfg, 2)
    if not _rollout_equal(ref_ro, again):
        raise Error("a rerun of the collection is not identical")
    # the iteration key changes the streams
    var other = collect_route(base, jobs, cfg, 3)
    if _rollout_equal(ref_ro, other):
        raise Error("global iteration 2 and 3 collected the same rollout")


def test_gather_order() raises:
    """A minibatch is `order[lo : lo + rows]` of the rollout: the gradient equals that of a rollout whose rows are the
    selected rows copied in that order (bit for bit: same chunks, same accumulation order)."""
    var h = 16
    var net = _random_net(h, 14)
    var jobs = _small_jobs(51, 30)
    var cfg = RoutePPOConfig()
    cfg.iters = 1
    cfg.seed = 6
    var ro = collect_route(net, jobs, cfg, 0)
    var n = ro.n
    var rng = Rng(77)
    var order = List[Int32]()
    var tmp = List[Int]()
    for i in range(n):
        tmp.append(i)
    var i = n - 1
    while i > 0:
        var j = rng.below(i + 1)
        var t = tmp[i]
        tmp[i] = tmp[j]
        tmp[j] = t
        i -= 1
    for k in range(n):
        order.append(Int32(tmp[k]))
    var lo = 137
    var rows = 421  # 4 chunks, the last of 37 rows
    var sub = RouteRollout()
    sub.n = rows
    var ident = List[Int32]()
    for r in range(rows):
        var src = Int(order[lo + r])
        for q in range(128):
            sub.sign.append(ro.sign[src * 128 + q])
        sub.A.append(ro.A[src])
        sub.act.append(ro.act[src])
        sub.logp_old.append(ro.logp_old[src])
        sub.adv.append(ro.adv[src])
        sub.ret.append(ro.ret[src])
        ident.append(Int32(r))
    var ws1 = RoutePPOWorkspace(h, net.param_count(), rows)
    var ws2 = RoutePPOWorkspace(h, net.param_count(), rows)
    var p1 = ppo_minibatch_grad(net, ws1, ro, order, lo, rows, cfg, 4)
    var p2 = ppo_minibatch_grad(net, ws2, sub, ident, 0, rows, cfg, 1)
    if not _bits_equal(ws1.grad, ws2.grad) or p1.policy != p2.policy or p1.value != p2.value or p1.entropy != p2.entropy:
        raise Error("the gathered minibatch differs from the explicitly copied rows")
    for r in range(rows):
        if ws1.last_ratio[r] != ws2.last_ratio[r]:
            raise Error("ratio order")
    var nz = False
    for k in range(len(ws1.grad)):
        if ws1.grad[k] != 0.0:
            nz = True
    if not nz:
        raise Error("zero gradient")


def test_first_order_decrease() raises:
    var h = 16
    var net = _random_net(h, 12)
    var jobs = _small_jobs(41, 24)
    var cfg = RoutePPOConfig()
    cfg.iters = 1
    cfg.seed = 3
    cfg.lr = 1e-3
    cfg.minibatch = 1 << 20
    var ro = collect_route(net, jobs, cfg, 0)
    var n = ro.n
    var order = List[Int32]()
    for i in range(n):
        order.append(Int32(i))
    var opt = Adam(net.param_count(), cfg.lr)
    var ws = RoutePPOWorkspace(h, net.param_count(), n)
    var first = 0.0
    var last = 0.0
    for s in range(30):
        var st = ppo_minibatch_step(net, opt, ws, ro, order, 0, n, cfg, cfg.lr, 4)
        var t = st.total(cfg)
        if s == 0:
            first = t
        last = t
    if not (last < first):
        raise Error("30 small steps did not reduce the PPO loss: " + String(first) + " -> " + String(last))


# ------------------------------------------------------------------------------------------------ the trainer


def test_train_smoke() raises:
    makedirs(OUT_DIR, exist_ok=True)
    var mf = load_model_file(SHIPPED)
    var start = RouterNet.from_model(mf)
    var cfg = RoutePPOConfig()
    cfg.iters = 2
    cfg.episodes = 12
    cfg.seed = 2
    cfg.workers = 4
    var a = train_route_ppo(cfg, mf)
    var res = a[1].copy()
    if res.iterations != 2 or len(res.log) != 2 or res.error.byte_length() != 0:
        raise Error("trainer ran " + String(res.iterations) + " iterations, log " + String(len(res.log)) + ", error '" + res.error + "'")
    if res.log[0].decisions <= 0 or res.log[0].iteration != 0 or res.log[1].iteration != 1:
        raise Error("log entries")
    assert_close(res.log[1].scale, curriculum_scale(1), 1e-15, 0.0, "curriculum scale")
    if _bits_equal(a[0].params, start.params):
        raise Error("training did not change the weights")
    var vh_changed = False
    for j in range(start.hidden):
        if a[0].params[a[0].off_wv() + j] != start.params[start.off_wv() + j]:
            vh_changed = True
    if not vh_changed:
        raise Error("the value head was not trained")
    for i in range(len(a[0].params)):
        var x = a[0].params[i]
        if not (x == x) or x > 1e30 or x < -1e30:
            raise Error("non-finite weight")
    # the result does not depend on the number of workers
    var cfg1 = cfg.copy()
    cfg1.workers = 1
    var b = train_route_ppo(cfg1, mf)
    if not _bits_equal(a[0].params, b[0].params):
        raise Error("trained weights differ between 4 workers and 1 worker")
    if b[1].log[0].decisions != res.log[0].decisions or b[1].log[1].mean_overhead != res.log[1].mean_overhead:
        raise Error("log statistics differ between 4 workers and 1 worker")
    # a stage that continues the global iteration counter uses other streams and a larger curriculum scale
    var cfg2 = cfg.copy()
    cfg2.start_iteration = 2
    var c = train_route_ppo(cfg2, mf)
    if c[1].log[0].iteration != 2 or c[1].log[0].decisions == res.log[0].decisions and c[1].log[0].mean_overhead == res.log[0].mean_overhead:
        raise Error("start_iteration did not change the rollouts")
    # run directory: loads as a container, as a runtime router model and as a warm start; exports with test vectors
    var run = OUT_DIR + "/run"
    for _ in range(2):
        write_ppo_run(run, a[0], res, cfg, SHIPPED)
    var back = load_model_file(run)
    if back.format != "qumo.rlroute.v0" or back.kind != "routing" or back.id != "run":
        raise Error("run manifest identity")
    var warm = RouterNet.from_model(back)
    if not _bits_equal(warm.params, a[0].params):
        raise Error("RouterNet.from_model(run) is not the trained net")
    var rm = load_route_model(run)
    if rm.hidden != 64:
        raise Error("runtime router model hidden size")
    var man = parse_json(back.manifest)
    var tr = man.get("training")
    if tr.get_string("algo", "") != "router PPO fine-tune" or tr.get_int("vw", 0) != RPPO_VW or tr.get_int("chunk_rows", 0) != RPPO_CHUNK_ROWS:
        raise Error("manifest training object")
    var lg = parse_json(_read(run + "/train_log.json"))
    if lg.get_int("iterations", 0) != 2 or len(lg.get("log")) != 2 or lg.get("log").at(1).get_int("decisions", -1) != res.log[1].decisions:
        raise Error("train_log.json")
    var stage = OUT_DIR + "/stage"
    var info = export_route_run(run, stage, "ppo_smoke", True, 1, "")
    if info.weights_bytes <= 0:
        raise Error("export wrote no weights")
    var exported = load_route_model(stage + "/ppo_smoke")
    if exported.hidden != 64:
        raise Error("exported router hidden size")
    # a second PPO stage warm-starts from the first run
    var d = train_route_ppo(cfg, back)
    if d[1].iterations != 2 or _bits_equal(d[0].params, a[0].params):
        raise Error("warm-started stage")


def _read(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


# ------------------------------------------------------------------------------------------------ errors and guards


def test_config_and_errors() raises:
    var cfg = RoutePPOConfig()
    var raised = False
    try:
        validate_config(cfg)  # neither --iters nor --budget-s
    except e:
        raised = String(e).startswith("QM-E640")
    if not raised:
        raise Error("neither iters nor budget must raise QM-E640")
    cfg.iters = 1
    cfg.budget_s = 5.0
    raised = False
    try:
        validate_config(cfg)  # both
    except e:
        raised = String(e).startswith("QM-E640")
    if not raised:
        raise Error("both iters and budget must raise QM-E640")
    cfg.budget_s = 0.0
    validate_config(cfg)
    cfg.episodes = 0
    raised = False
    try:
        validate_config(cfg)
    except e:
        raised = String(e).startswith("QM-E640")
    if not raised:
        raise Error("zero episodes must raise QM-E640")
    # a synthesis model is not a router
    var synth = load_model_file("models/ai_transpiler/synth/lf_grid2x3")
    var c2 = RoutePPOConfig()
    c2.iters = 1
    raised = False
    try:
        _ = train_route_ppo(c2, synth)
    except e:
        raised = String(e).startswith("QM-E647")
    if not raised:
        raise Error("a non-router init must raise QM-E647")
    # a time budget stops by the clock: zero-length budgets run no iteration, a tiny one at least stops
    var c3 = RoutePPOConfig()
    c3.budget_s = 0.0000001
    c3.episodes = 4
    var mf = load_model_file(SHIPPED)
    var r3 = train_route_ppo(c3, mf)
    if r3[1].iterations > 3:
        raise Error("the budget did not stop the run")


def test_divergence_guard() raises:
    var mf = load_model_file(SHIPPED)
    var start = RouterNet.from_model(mf)
    var cfg = RoutePPOConfig()
    cfg.iters = 3
    cfg.episodes = 8
    cfg.lr = 1e30  # the first Adam step throws the weights to +-1e30: forward passes overflow
    var out = train_route_ppo(cfg, mf)
    if not out[1].error.startswith("QM-E643"):
        raise Error("expected a QM-E643 stop, got '" + out[1].error + "'")
    if not _bits_equal(out[0].params, start.params):
        raise Error("the divergence guard must return the last good (initial) weights")
    if out[1].iterations != 0:
        raise Error("a diverged iteration must not be counted")


def main() raises:
    test_golden()
    test_sample_cdf()
    test_episode_split()
    test_rollout_replay()
    test_thread_independence()
    test_gather_order()
    test_first_order_decrease()
    test_train_smoke()
    test_config_and_errors()
    test_divergence_guard()
    print("test_rl_route_ppo OK")
