"""Tests for qrl.loss (synthesis PPO loss and gradient, log-softmax helpers, twisterl GAE) together with the full
batch path through `BasicNet` that the PPO update uses.

Pinned against data produced by the real packages (torch 2.14.1 / twisterl 0.5.1):
  * `ppo/gae.json`: four (gamma, lambda) settings of the compiled collector, 1e-6;
  * `ppo/ppo_update.json`: the 14-row batch (two twists, an all-masked row): `old_lp` including the all-masked row = 0.0
    exactly, the per-epoch loss dict for 10 epochs (2e-5), gradients and weights at epochs 0, 1, 9 (3e-6 / 3e-5), and the
    `normalize_advantage` variant;
independent checks that need no fixture:
  * the clip / tie rule and the entropy term against central finite differences of a Float64 reference loss;
  * thread-count independence of the virtual-worker gradient reduction on the 14-row batch replicated to 5,000 rows.
"""

from std.memory import bitcast
from std.math import exp, log
from max.algorithm import parallelize
from qrl.json import JsonValue
from qrl.rng import Rng
from qrl.tensor import FPtr, VW
from qrl.optim import Adam
from qrl.rlrng import uniform_f32
from qrl.nn import NetSpec, BasicNet, SparseBatch, ChunkCache, GradBufs
from qrl.loss import (
    PPOLossCfg,
    LossParts,
    IPtr,
    log_softmax_row,
    logsumexp_row,
    old_logp_row,
    normalize_advantages,
    ppo_chunk_loss,
    gae_twisterl,
)
from qrl.testutil import (
    load_fixture,
    flat_floats32,
    floats32,
    ints,
    ints2,
    assert_close,
    assert_list_close32,
    abs64,
    guard_alloc,
    guard_ptr,
    guard_set,
    guard_check,
)


def _bits_equal(a: List[Float32], b: List[Float32]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            return False
    return True


def _p(mut l: List[Float32]) -> FPtr:
    return l.unsafe_ptr().as_unsafe_any_origin()


def _ip(mut l: List[Int32]) -> IPtr:
    return l.unsafe_ptr().as_unsafe_any_origin()


def _cf(l: List[Float32]) -> FPtr:
    return l.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin()


def _ci(l: List[Int32]) -> IPtr:
    return l.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin()


# ------------------------------------------------------------------------------------------------ torch-layout weights


def tiny_spec() raises -> NetSpec:
    """The BasicPolicy of `policy_forward.json`: OBS 9, A 4, E 8, common [6], both twists."""
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
    """Writes tensor `i` (registration order = torch `state_dict` order) from torch-layout values: every tensor is
    `[out][in]` row-major, except `embeddings.weight`, which `BasicNet` stores transposed (`[OBS, E]`)."""
    var off = net.ps.offsets[i]
    if len(vals) != net.ps.size_of(i):
        raise Error("tensor " + String(i) + ": " + String(len(vals)) + " values, expected " + String(net.ps.size_of(i)))
    if i == 0:
        var obs = net.ps.rows_of(0)
        var e = net.ps.cols_of(0)
        for o in range(e):
            for k in range(obs):
                net.ps.data[off + k * e + o] = vals[o * obs + k]
    else:
        for j in range(len(vals)):
            net.ps.data[off + j] = vals[j]


def torch_layout(net: BasicNet, i: Int, flat: List[Float32]) -> List[Float32]:
    """Tensor `i` of a flat `ps`-layout buffer (weights or gradients) in torch layout."""
    var off = net.ps.offsets[i]
    var n = net.ps.size_of(i)
    var out = List[Float32](length=n, fill=0.0)
    if i == 0:
        var obs = net.ps.rows_of(0)
        var e = net.ps.cols_of(0)
        for o in range(e):
            for k in range(obs):
                out[o * obs + k] = flat[off + k * e + o]
    else:
        for j in range(n):
            out[j] = flat[off + j]
    return out^


def load_weights(mut net: BasicNet, w: JsonValue, keys: List[String]) raises:
    for i in range(len(keys)):
        set_param_torch(net, i, flat_floats32(w.get(keys[i]).get("data")))


def tiny_net() raises -> BasicNet:
    var fx = load_fixture("ppo/policy_forward.json")
    var net = BasicNet(tiny_spec(), 0, False)
    var keys = List[String]()
    var order = fx.get("tensor_order")
    for i in range(len(order)):
        keys.append(order.at(i).as_string())
    load_weights(net, fx.get("weights"), keys)
    return net^


# ------------------------------------------------------------------------------------------------ batches


struct Batch(Movable):
    var sb: SparseBatch
    var acts: List[Int32]
    var advs: List[Float32]
    var rets: List[Float32]
    var old_lp: List[Float32]
    var n: Int

    def __init__(out self):
        self.sb = SparseBatch()
        self.acts = List[Int32]()
        self.advs = List[Float32]()
        self.rets = List[Float32]()
        self.old_lp = List[Float32]()
        self.n = 0


def update_batch(reps: Int = 1) raises -> Batch:
    """The fixed 14-row batch of `ppo_update.json` (repeated `reps` times)."""
    var fx = load_fixture("ppo/ppo_update.json")
    var b = fx.get("batch")
    var obs = ints2(b.get("obs"))
    var perm = ints(b.get("perm"))
    var acts = ints(b.get("actions"))
    var advs = floats32(b.get("advs"))
    var rets = floats32(b.get("rets"))
    var olp = floats32(b.get("old_logp"))
    var out = Batch()
    for _ in range(reps):
        for r in range(len(obs)):
            out.sb.append_row_ints(obs[r], perm[r])
            out.acts.append(Int32(acts[r]))
            out.advs.append(advs[r])
            out.rets.append(rets[r])
            out.old_lp.append(olp[r])
    out.n = len(obs) * reps
    return out^


def batch_grad(net: BasicNet, mut b: Batch, cfg: PPOLossCfg, mut grad: List[Float32]) -> LossParts:
    """One full-batch loss + gradient in a single chunk; returns the partial sums."""
    var n = b.n
    var A = net.spec.num_actions
    var cc = ChunkCache(net.spec, n)
    net.forward_chunk(b.sb, 0, n, cc)
    var dz = List[Float32](length=n * A, fill=0.0)
    var dv = List[Float32](length=n, fill=0.0)
    var inv_n = Float32(1.0) / Float32(n)
    var parts = ppo_chunk_loss(
        _p(cc.logits), _p(cc.values), n, A, _ip(b.acts), _p(b.advs), _p(b.rets), _p(b.old_lp), cfg, inv_n, _p(dz), _p(dv)
    )
    for i in range(len(grad)):
        grad[i] = 0.0
    net.backward_chunk(b.sb, 0, n, cc, _p(dz), _p(dv), _p(grad))
    _ = len(cc.logits)
    _ = len(cc.values)
    _ = len(b.acts)
    _ = len(b.advs)
    _ = len(b.rets)
    _ = len(b.old_lp)
    _ = len(dz)
    _ = len(dv)
    _ = len(grad)
    return parts^


def total_loss(parts: LossParts, n: Int, cfg: PPOLossCfg) -> Float64:
    var nf = Float64(n)
    return parts.policy / nf + Float64(cfg.vf_coef) * (parts.value / nf) - Float64(cfg.ent_coef) * (parts.entropy / nf)


# ------------------------------------------------------------------------------------------------ GAE


def test_gae() raises:
    var fx = load_fixture("ppo/gae.json")
    var cases = fx.get("cases")
    assert_close(Float64(len(cases)), 4.0, 0.0, 0.0, "gae case count")
    for c in range(len(cases)):
        var cs = cases.at(c)
        var rew = floats32(cs.get("rewards"))
        var val = floats32(cs.get("values"))
        var want_a = floats32(cs.get("advs"))
        var want_r = floats32(cs.get("rets"))
        var gamma = Float32(cs.get("gamma").as_float())
        var lam = Float32(cs.get("lambda").as_float())
        var advs = List[Float32]()
        var rets = List[Float32]()
        gae_twisterl(Span(rew), Span(val), gamma, lam, advs, rets)
        assert_list_close32(advs, want_a, 1e-6, 1e-6, "gae advs case " + String(c))
        assert_list_close32(rets, want_r, 1e-6, 1e-6, "gae rets case " + String(c))
    # edge cases: one state, empty episode, the terminal row bootstraps from nothing
    var r1: List[Float32] = [1.0]
    var v1: List[Float32] = [0.25]
    var a1 = List[Float32]()
    var t1 = List[Float32]()
    gae_twisterl(Span(r1), Span(v1), 0.995, 0.995, a1, t1)
    assert_close(Float64(a1[0]), 0.75, 0.0, 0.0, "n = 1 advantage")
    assert_close(Float64(t1[0]), 1.0, 0.0, 0.0, "n = 1 return")
    var e0 = List[Float32]()
    var e1 = List[Float32]()
    gae_twisterl(Span(e0), Span(e1), 0.9, 0.9, a1, t1)
    assert_close(Float64(len(a1) + len(t1)), 0.0, 0.0, 0.0, "empty episode")
    # closed form of the recursion in f64 for a longer episode: advs[t] = delta_t + gamma lam advs[t+1]
    var rng = Rng(5)
    var n = 40
    var rr = List[Float32](length=n, fill=0.0)
    var vv = List[Float32](length=n, fill=0.0)
    for i in range(n):
        rr[i] = Float32(rng.uniform() - 0.3)
        vv[i] = Float32(rng.uniform() * 2.0 - 1.0)
    var ad = List[Float32]()
    var rt = List[Float32]()
    gae_twisterl(Span(rr), Span(vv), 0.97, 0.9, ad, rt)
    var nxt: Float64 = 0.0
    var t = n - 1
    while t >= 0:
        var delta = Float64(rr[t]) - Float64(vv[t])
        if t + 1 < n:
            delta = Float64(rr[t]) + 0.97 * Float64(vv[t + 1]) - Float64(vv[t])
        nxt = delta + 0.97 * 0.9 * nxt
        assert_close(Float64(ad[t]), nxt, 2e-5, 2e-5, "closed form advantage " + String(t))
        assert_close(Float64(rt[t]), nxt + Float64(vv[t]), 2e-5, 2e-5, "closed form return " + String(t))
        t -= 1
    print("  gae OK")


# ------------------------------------------------------------------------------------------------ log-softmax helpers


def test_log_softmax() raises:
    var z: List[Float32] = [1.0, 2.0, 3.0, -1.0]
    var lp = List[Float32](length=4, fill=0.0)
    log_softmax_row(_p(z), 4, _p(lp))
    var s: Float64 = 0.0
    for i in range(4):
        s += exp(Float64(z[i]))
    for i in range(4):
        assert_close(Float64(lp[i]), Float64(z[i]) - log(s), 1e-6, 1e-6, "log_softmax " + String(i))
    assert_close(Float64(logsumexp_row(_p(z), 4)), log(s), 1e-6, 1e-6, "logsumexp")
    # huge logits do not overflow (max shift)
    var big: List[Float32] = [1000.0, 1001.0, 999.0]
    var lb = List[Float32](length=3, fill=0.0)
    log_softmax_row(_p(big), 3, _p(lb))
    var sb = exp(-1.0) + 1.0 + exp(-2.0)
    assert_close(Float64(lb[1]), -log(sb), 1e-5, 1e-5, "log_softmax large logits")
    # A = 1: log-softmax is 0
    var one: List[Float32] = [7.5]
    var l1 = List[Float32](length=1, fill=1.0)
    log_softmax_row(_p(one), 1, _p(l1))
    assert_close(Float64(l1[0]), 0.0, 0.0, 0.0, "A = 1 log_softmax")
    # in-place (out aliases z)
    var zz: List[Float32] = [0.5, -0.5]
    var zc = zz.copy()
    var lz = List[Float32](length=2, fill=0.0)
    log_softmax_row(_p(zc), 2, _p(lz))
    log_softmax_row(_p(zz), 2, _p(zz))
    assert_list_close32(zz, lz, 0.0, 0.0, "in-place log_softmax")
    # all-masked row: old_lp is exactly 0.0, not -ln(A) (C-1)
    var masked: List[Float32] = [-1e10, -1e10, -1e10, -1e10]
    var olp = old_logp_row(_p(masked), 4, 2)
    assert_close(Float64(bitcast[DType.uint32](olp)), 0.0, 0.0, 0.0, "all-masked old_lp is +0.0")
    # one legal action: logp 0
    var one_legal: List[Float32] = [-1e10, 0.37, -1e10, -1e10]
    assert_close(Float64(old_logp_row(_p(one_legal), 4, 1)), 0.0, 1e-7, 0.0, "single legal action")
    print("  log-softmax helpers OK")


# ------------------------------------------------------------------------------------------------ ppo_update.json


def test_old_logp_golden() raises:
    var fx = load_fixture("ppo/ppo_update.json")
    var b = fx.get("batch")
    var masked = b.get("old_logits_masked")
    var acts = ints(b.get("actions"))
    var want = floats32(b.get("old_logp"))
    assert_close(Float64(len(want)), 14.0, 0.0, 0.0, "batch rows")
    for r in range(len(want)):
        var row = floats32(masked.at(r))
        var got = old_logp_row(_p(row), len(row), acts[r])
        assert_close(Float64(got), Float64(want[r]), 1e-6, 1e-6, "old_logp row " + String(r))
    var last = floats32(masked.at(13))
    var got13 = old_logp_row(_p(last), len(last), acts[13])
    if bitcast[DType.uint32](got13) != 0:
        raise Error("old_lp of the all-masked row is not exactly +0.0")
    print("  old_lp (incl. the all-masked row = 0.0) OK")


def test_update_epochs() raises:
    var fx = load_fixture("ppo/ppo_update.json")
    var cfgj = fx.get("config").get("training")
    var cfg = PPOLossCfg(
        Float32(cfgj.get("clip_ratio").as_float()),
        Float32(cfgj.get("vf_coef").as_float()),
        Float32(cfgj.get("ent_coef").as_float()),
    )
    var lr = fx.get("config").get("optimizer").get("lr").as_float()
    var epochs = fx.get("epochs")
    var net = tiny_net()
    var keys = List[String]()
    var order = load_fixture("ppo/policy_forward.json").get("tensor_order")
    for i in range(len(order)):
        keys.append(order.at(i).as_string())
    var batch = update_batch()
    var n = batch.n
    var opt = Adam(net.param_count(), lr)
    var grad = List[Float32](length=net.param_count(), fill=0.0)
    var checked = 0
    for e in range(len(epochs)):
        var ep = epochs.at(e)
        var parts = batch_grad(net, batch, cfg, grad)
        var nf = Float64(n)
        var pol = parts.policy / nf
        var val = parts.value / nf
        var ent = parts.entropy / nf
        var tot = total_loss(parts, n, cfg)
        var want = ep.get("losses")
        assert_close(val, want.get("value").as_float(), 2e-5, 2e-5, "epoch " + String(e) + " value loss")
        assert_close(pol, want.get("policy").as_float(), 2e-5, 2e-5, "epoch " + String(e) + " policy loss")
        assert_close(ent, want.get("entropy").as_float(), 2e-5, 2e-5, "epoch " + String(e) + " entropy")
        assert_close(tot, want.get("total").as_float(), 2e-5, 2e-5, "epoch " + String(e) + " total loss")
        if ep.has("grads"):
            for i in range(len(keys)):
                var wg = flat_floats32(ep.get("grads").get(keys[i]).get("data"))
                var gg = torch_layout(net, i, grad)
                assert_list_close32(gg, wg, 3e-6, 2e-4, "epoch " + String(e) + " grad " + keys[i])
            checked += 1
        opt.step(net.ps.data, grad, lr)
        if ep.has("weights_after"):
            for i in range(len(keys)):
                var ww = flat_floats32(ep.get("weights_after").get(keys[i]).get("data"))
                var got = torch_layout(net, i, net.ps.data)
                assert_list_close32(got, ww, 3e-5, 1e-4, "epoch " + String(e) + " weights " + keys[i])
            checked += 1
    assert_close(Float64(checked), 6.0, 0.0, 0.0, "epochs with gradients and weights (0, 1, 9)")
    print("  10-epoch update (losses, gradients and weights at epochs 0, 1, 9) OK")


def test_normalize_variant() raises:
    var fx = load_fixture("ppo/ppo_update.json")
    var cfgj = fx.get("config").get("training")
    var cfg = PPOLossCfg(
        Float32(cfgj.get("clip_ratio").as_float()),
        Float32(cfgj.get("vf_coef").as_float()),
        Float32(cfgj.get("ent_coef").as_float()),
    )
    var lr = fx.get("config").get("optimizer").get("lr").as_float()
    var net = tiny_net()
    var keys = List[String]()
    var order = load_fixture("ppo/policy_forward.json").get("tensor_order")
    for i in range(len(order)):
        keys.append(order.at(i).as_string())
    var batch = update_batch()
    normalize_advantages(_p(batch.advs), batch.n)
    # unbiased std: mean 0 and (sum x^2) = n - 1 up to the 1e-8 in the denominator
    var s: Float64 = 0.0
    var ss: Float64 = 0.0
    for i in range(batch.n):
        s += Float64(batch.advs[i])
        ss += Float64(batch.advs[i]) * Float64(batch.advs[i])
    assert_close(s, 0.0, 1e-5, 0.0, "normalised advantages have mean 0")
    assert_close(ss, Float64(batch.n - 1), 1e-4, 1e-5, "normalised advantages have unbiased std 1")
    var grad = List[Float32](length=net.param_count(), fill=0.0)
    var parts = batch_grad(net, batch, cfg, grad)
    var nf = Float64(batch.n)
    var want = fx.get("normalize_advantage_variant").get("loss")
    assert_close(parts.policy / nf, want.get("policy").as_float(), 2e-5, 2e-5, "normalised policy loss")
    assert_close(parts.value / nf, want.get("value").as_float(), 2e-5, 2e-5, "normalised value loss")
    assert_close(total_loss(parts, batch.n, cfg), want.get("total").as_float(), 2e-5, 2e-5, "normalised total loss")
    var opt = Adam(net.param_count(), lr)
    opt.step(net.ps.data, grad, lr)
    var wa = fx.get("normalize_advantage_variant").get("weights_after_step1")
    for i in range(len(keys)):
        var ww = flat_floats32(wa.get(keys[i]).get("data"))
        var got = torch_layout(net, i, net.ps.data)
        assert_list_close32(got, ww, 3e-5, 1e-4, "normalised weights " + keys[i])
    print("  normalize_advantage variant OK")


# ------------------------------------------------------------------------------------------------ finite differences


def _row_loss_f64(z: List[Float64], a: Int, adv: Float64, v: Float64, ret: Float64, old_lp: Float64, c: Float64, vf: Float64, ent: Float64) -> Float64:
    """Float64 reference of one row's contribution to `policy + vf * value - ent * entropy` (before dividing by N)."""
    var m = z[0]
    for i in range(len(z)):
        if z[i] > m:
            m = z[i]
    var s: Float64 = 0.0
    for i in range(len(z)):
        s += exp(z[i] - m)
    var lse = m + log(s)
    var h: Float64 = 0.0
    for i in range(len(z)):
        var lp = z[i] - lse
        h -= exp(lp) * lp
    var ratio = exp(z[a] - lse - old_lp)
    var s1 = ratio * adv
    var cl = ratio
    if cl < 1.0 - c:
        cl = 1.0 - c
    if cl > 1.0 + c:
        cl = 1.0 + c
    var s2 = cl * adv
    var mn = s1
    if s2 < s1:
        mn = s2
    return -mn + vf * (v - ret) * (v - ret) - ent * h


def test_clip_rule_and_entropy() raises:
    """The gradients dz / dv of `ppo_chunk_loss` against central differences of the Float64 reference, in every regime of the clip rule
    (ratio inside / above / below the band, advantage of either sign) and with the entropy and value terms active."""
    var cfg = PPOLossCfg(0.1, 0.8, 0.05)
    var A = 5
    var rng = Rng(11)
    var rows = 0
    var regimes = List[Int](length=6, fill=0)
    for trial in range(60):
        var z = List[Float64](length=A, fill=0.0)
        var zf = List[Float32](length=A, fill=0.0)
        for j in range(A):
            var x = Float32(rng.normal())
            zf[j] = x
            z[j] = Float64(x)
        var a = rng.below(A)
        var adv = Float32(rng.normal())
        var ret = Float32(rng.normal())
        var v = Float32(rng.normal())
        # choose old_lp so that the ratio lands in a chosen regime
        var m = zf[0]
        for j in range(A):
            if zf[j] > m:
                m = zf[j]
        var lse = Float64(m) + log(Float64(sum_exp(zf, A, m)))
        var lp = Float64(zf[a]) - lse
        var target = 1.0
        var regime = trial % 3
        if regime == 1:
            target = 1.6
        elif regime == 2:
            target = 0.55
        var old_lp = Float32(lp - log(target))
        var logits = zf.copy()
        var val = List[Float32](length=1, fill=v)
        var rets = List[Float32](length=1, fill=ret)
        var advs = List[Float32](length=1, fill=adv)
        var olp = List[Float32](length=1, fill=old_lp)
        var acts = List[Int32](length=1, fill=Int32(a))
        var dz = List[Float32](length=A, fill=0.0)
        var dv = List[Float32](length=1, fill=0.0)
        var parts = ppo_chunk_loss(_p(logits), _p(val), 1, A, _ip(acts), _p(advs), _p(rets), _p(olp), cfg, 1.0, _p(dz), _p(dv))
        var lref = _row_loss_f64(z, a, Float64(adv), Float64(v), Float64(ret), Float64(old_lp), 0.1, 0.8, 0.05)
        assert_close(total_loss(parts, 1, cfg), lref, 2e-5, 2e-5, "row loss " + String(trial))
        var eps = 1e-5
        for j in range(A):
            var zp = z.copy()
            var zm = z.copy()
            zp[j] += eps
            zm[j] -= eps
            var fd = (
                _row_loss_f64(zp, a, Float64(adv), Float64(v), Float64(ret), Float64(old_lp), 0.1, 0.8, 0.05)
                - _row_loss_f64(zm, a, Float64(adv), Float64(v), Float64(ret), Float64(old_lp), 0.1, 0.8, 0.05)
            ) / (2.0 * eps)
            assert_close(Float64(dz[j]), fd, 2e-4, 2e-3, "dz trial " + String(trial) + " j " + String(j))
        var fdv = (
            _row_loss_f64(z, a, Float64(adv), Float64(v) + eps, Float64(ret), Float64(old_lp), 0.1, 0.8, 0.05)
            - _row_loss_f64(z, a, Float64(adv), Float64(v) - eps, Float64(ret), Float64(old_lp), 0.1, 0.8, 0.05)
        ) / (2.0 * eps)
        assert_close(Float64(dv[0]), fdv, 2e-4, 2e-3, "dv trial " + String(trial))
        var r = exp(lp - Float64(old_lp))
        var kind = 0
        if r > 1.1:
            kind = 1
        elif r < 0.9:
            kind = 2
        regimes[kind * 2 + (0 if adv > 0 else 1)] += 1
        rows += 1
        _ = len(logits)
        _ = len(val)
    for k in range(6):
        if regimes[k] == 0:
            raise Error("clip-rule regime " + String(k) + " was never exercised")
    # exact tie s1 == s2 (ratio exactly 1 -> 1 inside the band): the full advantage passes through
    var z3: List[Float32] = [0.0, 0.0, 0.0]
    var a3 = List[Int32](length=1, fill=Int32(1))
    var adv3: List[Float32] = [0.5]
    var ret3: List[Float32] = [0.0]
    var v3: List[Float32] = [0.0]
    var olp3: List[Float32] = [Float32(-log(3.0))]
    var dz3 = List[Float32](length=3, fill=0.0)
    var dv3 = List[Float32](length=1, fill=0.0)
    var cfg0 = PPOLossCfg(0.1, 0.0, 0.0)
    _ = ppo_chunk_loss(_p(z3), _p(v3), 1, 3, _ip(a3), _p(adv3), _p(ret3), _p(olp3), cfg0, 1.0, _p(dz3), _p(dv3))
    # dL/dlp = -adv * ratio, dz = dL/dlp * (onehot - p): ratio = 1 up to f32 rounding
    assert_close(Float64(dz3[1]), -0.5 * (1.0 - 1.0 / 3.0), 1e-5, 1e-5, "tie: dz at the action")
    assert_close(Float64(dz3[0]), 0.5 / 3.0, 1e-5, 1e-5, "tie: dz at another action")
    # the f32-pointer overload (MOJO_PORT.md section 5 spelling) agrees with the Int32 form
    var acts_f: List[Float32] = [1.0]
    var dz4 = List[Float32](length=3, fill=0.0)
    var dv4 = List[Float32](length=1, fill=0.0)
    _ = ppo_chunk_loss(_p(z3), _p(v3), 1, 3, _p(acts_f), _p(adv3), _p(ret3), _p(olp3), cfg0, 1.0, _p(dz4), _p(dv4))
    assert_list_close32(dz4, dz3, 0.0, 0.0, "f32 action overload")
    print("  clip rule, tie, entropy and value gradients vs finite differences OK (" + String(rows) + " rows)")


def sum_exp(z: List[Float32], n: Int, m: Float32) -> Float64:
    var s: Float64 = 0.0
    for i in range(n):
        s += exp(Float64(z[i]) - Float64(m))
    return s


def test_chunk_guards() raises:
    """`ppo_chunk_loss` writes exactly dz[0 : rows * A] and dv[0 : rows]."""
    var A = 4
    var rows = 7
    var rng = Rng(3)
    var logits = List[Float32](length=rows * A, fill=0.0)
    var val = List[Float32](length=rows, fill=0.0)
    var advs = List[Float32](length=rows, fill=0.0)
    var rets = List[Float32](length=rows, fill=0.0)
    var olp = List[Float32](length=rows, fill=0.0)
    var acts = List[Int32](length=rows, fill=0)
    for r in range(rows):
        for a in range(A):
            logits[r * A + a] = Float32(rng.normal())
        val[r] = Float32(rng.normal())
        advs[r] = Float32(rng.normal())
        rets[r] = Float32(rng.normal())
        olp[r] = -1.3
        acts[r] = Int32(rng.below(A))
    var dzg = guard_alloc(rows * A)
    var dvg = guard_alloc(rows)
    var cfg = PPOLossCfg(0.1, 0.8, 0.01)
    _ = ppo_chunk_loss(_p(logits), _p(val), rows, A, _ip(acts), _p(advs), _p(rets), _p(olp), cfg, Float32(1.0 / 7.0), guard_ptr(dzg), guard_ptr(dvg))
    guard_check(dzg, rows * A, "ppo_chunk_loss dz")
    guard_check(dvg, rows, "ppo_chunk_loss dv")
    _ = len(dzg)
    _ = len(dvg)
    print("  loss kernel guard regions OK")


# ------------------------------------------------------------------------------------------------ thread-count independence


def grad_vw(net: BasicNet, mut b: Batch, cfg: PPOLossCfg, chunk_rows: Int, cores: Int, mut out: List[Float32]) -> List[Float64]:
    """The virtual-worker reduction of MOJO_PORT.md section 3 (rule 3) on one batch: rows are cut into chunks of
    `chunk_rows`, chunk `i` belongs to virtual worker `i % VW`, every worker has a private gradient buffer and processes
    its chunks in increasing order, the buffers are then summed in index order, and the loss partial sums are kept per
    chunk and summed in chunk order. `cores` only changes how many threads run the VW tasks. Returns [policy, value,
    entropy] sums."""
    var n = b.n
    var A = net.spec.num_actions
    var nchunks = (n + chunk_rows - 1) // chunk_rows
    var gb = GradBufs(net.param_count(), VW)
    var gp = gb.data.unsafe_ptr().as_unsafe_any_origin()
    var plen = net.param_count()
    var part = List[Float64](length=3 * nchunks, fill=0.0)
    var pp = part.unsafe_ptr().as_unsafe_any_origin()
    var inv_n = Float32(1.0) / Float32(n)

    def task(w: Int) {imm net, imm b, imm cfg, imm chunk_rows, imm n, imm A, imm nchunks, imm gp, imm plen, imm pp, imm inv_n}:
        var cc = ChunkCache(net.spec, chunk_rows)
        var dz = List[Float32](length=chunk_rows * A, fill=0.0)
        var dv = List[Float32](length=chunk_rows, fill=0.0)
        var c = w
        while c < nchunks:
            var lo = c * chunk_rows
            var hi = min(lo + chunk_rows, n)
            net.forward_chunk(b.sb, lo, hi, cc)
            var rows = hi - lo
            var bp = _cf(b.advs).unsafe_offset(lo)
            var rp = _cf(b.rets).unsafe_offset(lo)
            var op = _cf(b.old_lp).unsafe_offset(lo)
            var ap = _ci(b.acts).unsafe_offset(lo)
            var parts = ppo_chunk_loss(
                cc.logits.unsafe_ptr().as_unsafe_any_origin(), cc.values.unsafe_ptr().as_unsafe_any_origin(), rows, A,
                ap, bp, rp, op, cfg, inv_n, dz.unsafe_ptr().as_unsafe_any_origin(), dv.unsafe_ptr().as_unsafe_any_origin(),
            )
            pp[unsafe_offset = 3 * c] = parts.policy
            pp[unsafe_offset = 3 * c + 1] = parts.value
            pp[unsafe_offset = 3 * c + 2] = parts.entropy
            net.backward_chunk(
                b.sb, lo, hi, cc, dz.unsafe_ptr().as_unsafe_any_origin(), dv.unsafe_ptr().as_unsafe_any_origin(),
                gp.unsafe_offset(w * plen),
            )
            c += VW

    parallelize(task, VW, cores)
    _ = len(gb.data)
    _ = len(part)
    _ = len(b.acts)
    _ = len(b.advs)
    _ = len(b.rets)
    _ = len(b.old_lp)
    gb.reduce_into(out)
    var tot = List[Float64](length=3, fill=0.0)
    for c in range(nchunks):
        tot[0] += part[3 * c]
        tot[1] += part[3 * c + 1]
        tot[2] += part[3 * c + 2]
    return tot^


def test_thread_independence() raises:
    var fx = load_fixture("ppo/ppo_update.json")
    var cfgj = fx.get("config").get("training")
    var cfg = PPOLossCfg(
        Float32(cfgj.get("clip_ratio").as_float()),
        Float32(cfgj.get("vf_coef").as_float()),
        Float32(cfgj.get("ent_coef").as_float()),
    )
    var net = tiny_net()
    var reps = 358  # 14 * 358 = 5,012 rows
    var b = update_batch(reps)
    assert_close(Float64(b.n), 5012.0, 0.0, 0.0, "replicated rows")
    var chunk_rows = 64  # 79 chunks over 16 virtual workers: every worker handles several chunks in order
    var g1 = List[Float32]()
    var t1 = grad_vw(net, b, cfg, chunk_rows, 1, g1)
    var counts: List[Int] = [2, 4, 8, 16]
    for ci in range(len(counts)):
        var g = List[Float32]()
        var t = grad_vw(net, b, cfg, chunk_rows, counts[ci], g)
        if not _bits_equal(g, g1):
            raise Error("gradient differs between 1 and " + String(counts[ci]) + " cores")
        for k in range(3):
            if bitcast[DType.uint64](t[k]) != bitcast[DType.uint64](t1[k]):
                raise Error("loss partial sums differ between 1 and " + String(counts[ci]) + " cores")
    # the chunked sum equals the single-chunk full batch (different summation order: tolerance, not bits)
    var gref = List[Float32](length=net.param_count(), fill=0.0)
    var pref = batch_grad(net, b, cfg, gref)
    assert_list_close32(g1, gref, 2e-6, 2e-4, "chunked gradient vs one-chunk gradient")
    assert_close(t1[0], pref.policy, 1e-3, 1e-6, "chunked policy partial")
    assert_close(t1[1], pref.value, 1e-3, 1e-6, "chunked value partial")
    assert_close(t1[2], pref.entropy, 1e-3, 1e-6, "chunked entropy partial")
    # replicating the batch r times leaves the normalised gradient equal to the 14-row gradient (the batch is a pure
    # repetition, so the mean over 5,012 rows is the mean over 14)
    var b14 = update_batch(1)
    var g14 = List[Float32](length=net.param_count(), fill=0.0)
    _ = batch_grad(net, b14, cfg, g14)
    assert_list_close32(g1, g14, 3e-6, 2e-4, "replicated batch gradient vs the 14-row gradient")
    print("  gradient reduction identical for 1, 2, 4, 8, 16 cores (5,012 rows, 79 chunks) OK")


def main() raises:
    test_gae()
    test_log_softmax()
    test_old_logp_golden()
    test_update_epochs()
    test_normalize_variant()
    test_clip_rule_and_entropy()
    test_chunk_guards()
    test_thread_independence()
    print("test_rl_loss OK")
