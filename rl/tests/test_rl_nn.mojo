"""Tests for qrl.nn: the parameter store, the synthesis network forward (single state and batched) and the
hand-derived backward pass.

Pinned against data produced by the real packages (torch 2.14.1 / twisterl 0.5.1):
  * `ppo/policy_forward.json`: 15 cases (twist none / 0 / 1 over five observations incl. the empty one) of raw logits and
    value, plus 5 ensemble cases, 1e-5, through `forward_one` and through the batched `forward_chunk`;
  * `torch/synth_ppo_golden.json`: two architectures (one common layer; two common layers with hidden action and value
    layers), both with symmetries: forward loss, gradients (atol 3e-6, rtol 2e-4) and the weights after 3 Adam steps
    (atol 3e-5);
  * the 9 shipped synth manifests: `BasicNet.from_model(model).forward_one` on every manifest test vector (no twist, every
    twist, and the ensemble average) equals the stored `logits` / `logits_ensemble` / `value` at 1e-5. Those vectors were
    made independently (Python / numpy), so they bind the training layout to the inference layout;
independent checks that need no fixture: finite differences of a Float64-summed loss on a random net with every layer
kind, parameter counts of the shipped architectures, sentinel guard regions around the gradient / logit buffers and
unused cache space, out-of-range observation indices, and `from_model` error handling.
"""

from std.memory import bitcast
from qrl.json import JsonValue, parse_json
from qrl.rng import Rng
from qrl.model import ModelFile, load_model_file, read_model_index, MLPPolicy
from qrl.tensor import FPtr
from qrl.optim import Adam
from qrl.nn import (
    ParamStore,
    GradBufs,
    SparseBatch,
    NetSpec,
    ChunkCache,
    ActorScratch,
    BasicNet,
)
from qrl.loss import PPOLossCfg, LossParts, IPtr, ppo_chunk_loss, old_logp_row
from qrl.testutil import (
    load_fixture,
    flat_floats32,
    floats32,
    ints,
    ints2,
    assert_close,
    assert_list_close32,
    abs64,
    max_abs_diff32,
    guard_alloc,
    guard_ptr,
    guard_set,
    guard_data,
    guard_check,
    model_root,
    has_model_root,
    skip_without_models,
)



def _p(mut l: List[Float32]) -> FPtr:
    return l.unsafe_ptr().as_unsafe_any_origin()


def _ip(mut l: List[Int32]) -> IPtr:
    return l.unsafe_ptr().as_unsafe_any_origin()


def _bits_equal(a: List[Float32], b: List[Float32]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            return False
    return True


def _span_of(l: List[Int]) -> List[Int32]:
    var out = List[Int32](capacity=len(l))
    for i in range(len(l)):
        out.append(Int32(l[i]))
    return out^


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


# ------------------------------------------------------------------------------------------------ ParamStore, spec


def test_param_store() raises:
    var ps = ParamStore()
    var o0 = ps.add("a", 3, 4)
    var o1 = ps.add("b", 1, 5)
    var o2 = ps.add("c", 2, 2)
    assert_close(Float64(o0), 0.0, 0.0, 0.0, "offset a")
    assert_close(Float64(o1), 12.0, 0.0, 0.0, "offset b")
    assert_close(Float64(o2), 17.0, 0.0, 0.0, "offset c")
    assert_close(Float64(ps.total()), 21.0, 0.0, 0.0, "total")
    assert_close(Float64(ps.offset("b")), 12.0, 0.0, 0.0, "named offset")
    assert_close(Float64(ps.rows_of(2) * 10 + ps.cols_of(2)), 22.0, 0.0, 0.0, "shape of c")
    var raised = False
    try:
        _ = ps.offset("missing")
    except e:
        raised = String(e).startswith("QM-E640")
    if not raised:
        raise Error("ParamStore.offset of an unknown name must raise QM-E640")
    var cp = ps.copy()
    cp.data[0] = 5.0
    assert_close(Float64(ps.data[0]), 0.0, 0.0, 0.0, "ParamStore copy is deep")
    print("  ParamStore OK")


def spec_of(obs: Int, a: Int, e: Int, common: List[Int], ph: List[Int], vh: List[Int]) -> NetSpec:
    var s = NetSpec()
    s.obs_size = obs
    s.num_actions = a
    s.emb = e
    s.common = common.copy()
    s.policy_hidden = ph.copy()
    s.value_hidden = vh.copy()
    return s^


def test_param_counts() raises:
    var none = List[Int]()
    var c256: List[Int] = [256]
    # the formula of MOJO_PORT.md 4.4 for one common layer
    assert_close(Float64(spec_of(36, 10, 512, c256, none, none).params()), 153099.0, 0.0, 0.0, "OBS 36, A 10")
    assert_close(Float64(spec_of(64, 7, 512, c256, none, none).params()), 166664.0, 0.0, 0.0, "OBS 64, A 7")
    var net = BasicNet(spec_of(36, 10, 512, c256, none, none), 7)
    assert_close(Float64(net.param_count()), 153099.0, 0.0, 0.0, "net param_count")
    var two: List[Int] = [5, 4]
    var ph: List[Int] = [3]
    var vh: List[Int] = [2]
    var sp = spec_of(12, 4, 6, two, ph, vh)
    var n2 = BasicNet(sp, 1)
    assert_close(Float64(n2.param_count()), Float64(sp.params()), 0.0, 0.0, "deep net param_count")
    # consecutive layer numbering (D9) and registration order
    var want: List[String] = [
        "embeddings.weight", "embeddings.bias", "common.0.weight", "common.0.bias", "common.1.weight", "common.1.bias",
        "action.0.weight", "action.0.bias", "action.1.weight", "action.1.bias",
        "value.0.weight", "value.0.bias", "value.1.weight", "value.1.bias",
    ]
    assert_close(Float64(n2.ps.count()), Float64(len(want)), 0.0, 0.0, "tensor count")
    for i in range(len(want)):
        if n2.ps.names[i] != want[i]:
            raise Error("tensor " + String(i) + " is '" + n2.ps.names[i] + "', expected '" + want[i] + "'")
    # a spec that is not a valid network is rejected with QM-E642
    var bad = spec_of(0, 4, 6, two, ph, vh)
    var raised = False
    try:
        _ = BasicNet(bad, 1)
    except e:
        raised = String(e).startswith("QM-E642")
    if not raised:
        raise Error("an empty observation space must raise QM-E642")
    var bad2 = spec_of(3, 2, 4, none, none, none)
    bad2.obs_perms.append([0, 0, 1])  # not a permutation
    bad2.act_perms.append([1, 0])
    raised = False
    try:
        _ = BasicNet(bad2, 1)
    except e:
        raised = String(e).startswith("QM-E642")
    if not raised:
        raise Error("a symmetry table that is not a permutation must raise QM-E642")
    print("  parameter counts, layout names and spec validation OK")


def test_init() raises:
    """Init bound U(-1/sqrt(fan_in), 1/sqrt(fan_in)), reproducible per seed, different across seeds, spread."""
    var c256: List[Int] = [256]
    var none = List[Int]()
    var net = BasicNet(spec_of(36, 10, 512, c256, none, none), 1234)
    var again = BasicNet(spec_of(36, 10, 512, c256, none, none), 1234)
    var other = BasicNet(spec_of(36, 10, 512, c256, none, none), 1235)
    if not _bits_equal(net.ps.data, again.ps.data):
        raise Error("init is not reproducible for one seed")
    if _bits_equal(net.ps.data, other.ps.data):
        raise Error("different seeds gave the same init")
    var bounds: List[Float64] = [1.0 / 6.0, 1.0 / 6.0, 1.0 / 22.627416997969522, 1.0 / 22.627416997969522, 1.0 / 16.0, 1.0 / 16.0, 1.0 / 16.0, 1.0 / 16.0]
    # the tensors of this net are embeddings.{w,b}, common.0.{w,b}, action.0.{w,b}, value.0.{w,b}
    for i in range(net.ps.count()):
        var off = net.ps.offsets[i]
        var n = net.ps.size_of(i)
        var mx: Float64 = 0.0
        var s: Float64 = 0.0
        for j in range(n):
            var v = Float64(net.ps.data[off + j])
            mx = max(mx, abs(v))
            s += v
        if mx > bounds[i] * (1.0 + 1e-6):
            raise Error("tensor " + net.ps.names[i] + " exceeds its init bound")
        if n >= 256:
            if mx < 0.95 * bounds[i]:
                raise Error("tensor " + net.ps.names[i] + " does not use its init range")
            if abs(s / Float64(n)) > 0.1 * bounds[i]:
                raise Error("tensor " + net.ps.names[i] + " is not centred")
    print("  init bounds OK")


# ------------------------------------------------------------------------------------------------ policy_forward.json


def tiny_net() raises -> BasicNet:
    var fx = load_fixture("ppo/policy_forward.json")
    var s = NetSpec()
    s.obs_size = fx.get("obs_size").as_int()
    s.num_actions = fx.get("num_actions").as_int()
    s.emb = fx.get("emb").as_int()
    s.common = ints(fx.get("common"))
    s.obs_perms = ints2(fx.get("obs_perms"))
    s.act_perms = ints2(fx.get("act_perms"))
    var net = BasicNet(s, 0, False)
    var order = fx.get("tensor_order")
    for i in range(len(order)):
        set_param_torch(net, i, flat_floats32(fx.get("weights").get(order.at(i).as_string()).get("data")))
    return net^


def test_forward_golden() raises:
    var fx = load_fixture("ppo/policy_forward.json")
    var net = tiny_net()
    var A = net.spec.num_actions
    var sc = ActorScratch(net.spec)
    var cases = fx.get("cases")
    assert_close(Float64(len(cases)), 15.0, 0.0, 0.0, "forward cases")
    var logits = List[Float32](length=A, fill=0.0)
    var worst: Float64 = 0.0
    # one SparseBatch with every case for the batched path
    var sb = SparseBatch()
    var want_l = List[Float32]()
    var want_v = List[Float32]()
    var raw_want = List[List[Float32]]()
    for c in range(len(cases)):
        var cs = cases.at(c)
        var obs = _span_of(ints(cs.get("obs")))
        var twist = -1
        if not cs.get("twist").is_null():
            twist = cs.get("twist").as_int()
        var wl = floats32(cs.get("raw_logits"))
        var wv = Float32(cs.get("value").as_float())
        # `raw_logits` of the fixture are the post-twist logits of a policy with this one twist
        var v = net.forward_one(Span(obs), twist, sc, _p(logits))
        for a in range(A):
            worst = max(worst, abs(Float64(logits[a]) - Float64(wl[a])))
        worst = max(worst, abs(Float64(v) - Float64(wv)))
        sb.append_row(Span(obs), twist)
        for a in range(A):
            want_l.append(wl[a])
        want_v.append(wv)
    if worst > 1e-5:
        raise Error("forward_one worst error " + String(worst))
    # batched path on the same rows (also exercises a chunk smaller than the cache)
    var cc = ChunkCache(net.spec, 20)
    net.forward_chunk(sb, 0, sb.rows(), cc)
    var got_l = List[Float32](capacity=len(want_l))
    for i in range(len(want_l)):
        got_l.append(cc.logits[i])
    assert_list_close32(got_l, want_l, 1e-5, 0.0, "forward_chunk logits")
    for r in range(len(want_v)):
        assert_close(Float64(cc.values[r]), Float64(want_v[r]), 1e-5, 0.0, "forward_chunk value " + String(r))
    # a sub-range of the batch gives the same rows
    net.forward_chunk(sb, 4, 9, cc)
    for r in range(5):
        for a in range(A):
            assert_close(Float64(cc.logits[r * A + a]), Float64(want_l[(4 + r) * A + a]), 1e-5, 0.0, "sub-range logits")
    # ensemble cases: mean over every twist of the post-twist logits and values
    var ens = fx.get("ensemble_cases")
    var P = net.spec.num_twists()
    for c in range(len(ens)):
        var obs = _span_of(ints(ens.at(c).get("obs")))
        var wl = floats32(ens.at(c).get("mean_raw_logits"))
        var wv = ens.at(c).get("mean_value").as_float()
        var acc = List[Float64](length=A, fill=0.0)
        var vacc: Float64 = 0.0
        for p in range(P):
            var v = net.forward_one(Span(obs), p, sc, _p(logits))
            vacc += Float64(v) / Float64(P)
            for a in range(A):
                acc[a] += Float64(logits[a]) / Float64(P)
        for a in range(A):
            assert_close(acc[a], Float64(wl[a]), 1e-5, 0.0, "ensemble logits " + String(c))
        assert_close(vacc, wv, 1e-5, 0.0, "ensemble value " + String(c))
    print("  policy_forward.json: 15 cases + 5 ensembles through forward_one and forward_chunk OK (worst " + String(worst) + ")")


# ------------------------------------------------------------------------------------------------ synth_ppo_golden.json


def keys_of(c: JsonValue) raises -> List[String]:
    var out = List[String]()
    var k = c.get("state_dict_keys")
    for i in range(len(k)):
        out.append(k.at(i).as_string())
    return out^


def nested(v: JsonValue) raises -> List[Float32]:
    return flat_floats32(v)


def test_synth_golden() raises:
    var fx = load_fixture("torch/synth_ppo_golden.json")
    var cases = fx.get("cases")
    assert_close(Float64(len(cases)), 2.0, 0.0, 0.0, "synth golden architectures")
    for ci in range(len(cases)):
        var c = cases.at(ci)
        var arch = c.get("arch")
        var s = NetSpec()
        s.obs_size = c.get("obs_size").as_int()
        s.num_actions = c.get("num_actions").as_int()
        s.emb = arch.get("embedding_size").as_int()
        s.common = ints(arch.get("common_layers"))
        s.policy_hidden = ints(arch.get("policy_layers"))
        s.value_hidden = ints(arch.get("value_layers"))
        s.obs_perms = ints2(c.get("obs_perms"))
        s.act_perms = ints2(c.get("act_perms"))
        var name = c.get("name").as_string()
        var keys = keys_of(c)
        var net = BasicNet(s, 0, False)
        assert_close(Float64(net.ps.count()), Float64(len(keys)), 0.0, 0.0, name + " tensor count")
        assert_close(Float64(net.param_count()), Float64(s.params()), 0.0, 0.0, name + " param count")
        for i in range(len(keys)):
            set_param_torch(net, i, nested(c.get("init").get(keys[i])))
        var hy = c.get("hyper")
        var lr = hy.get("lr").as_float()
        var cfg = PPOLossCfg(Float32(hy.get("clip").as_float()), Float32(hy.get("vf_coef").as_float()), Float32(hy.get("ent_coef").as_float()))
        var b = c.get("batch")
        var oi = ints2(b.get("obs_indices"))
        var pidx = ints(b.get("perm_idx"))
        var acts = List[Int32]()
        var ai = ints(b.get("actions"))
        for i in range(len(ai)):
            acts.append(Int32(ai[i]))
        var advs = floats32(b.get("advs"))
        var rets = floats32(b.get("rets"))
        var lo = b.get("logits_old")
        var n = len(oi)
        var A = s.num_actions
        var olp = List[Float32](length=n, fill=0.0)
        for r in range(n):
            var row = floats32(lo.at(r))
            olp[r] = old_logp_row(_p(row), A, ai[r])
        var sb = SparseBatch()
        for r in range(n):
            sb.append_row_ints(oi[r], pidx[r])
        var opt = Adam(net.param_count(), lr)
        var grad = List[Float32](length=net.param_count(), fill=0.0)
        var steps = c.get("steps")
        for st in range(len(steps)):
            var cc = ChunkCache(s, n)
            net.forward_chunk(sb, 0, n, cc)
            var dz = List[Float32](length=n * A, fill=0.0)
            var dv = List[Float32](length=n, fill=0.0)
            var parts = ppo_chunk_loss(
                _p(cc.logits), _p(cc.values), n, A, _ip(acts), _p(advs), _p(rets), _p(olp), cfg, Float32(1.0) / Float32(n), _p(dz), _p(dv)
            )
            for i in range(len(grad)):
                grad[i] = 0.0
            net.backward_chunk(sb, 0, n, cc, _p(dz), _p(dv), _p(grad))
            var nf = Float64(n)
            var want = steps.at(st).get("metrics")
            var tag = name + " step " + String(st)
            assert_close(parts.value / nf, want.get("value").as_float(), 2e-5, 2e-5, tag + " value loss")
            assert_close(parts.policy / nf, want.get("policy").as_float(), 2e-5, 2e-5, tag + " policy loss")
            assert_close(parts.entropy / nf, want.get("entropy").as_float(), 2e-5, 2e-5, tag + " entropy")
            var tot = parts.policy / nf + Float64(cfg.vf_coef) * (parts.value / nf) - Float64(cfg.ent_coef) * (parts.entropy / nf)
            assert_close(tot, want.get("total").as_float(), 2e-5, 2e-5, tag + " total")
            for i in range(len(keys)):
                var wg = nested(steps.at(st).get("grads").get(keys[i]))
                assert_list_close32(torch_layout(net, i, grad), wg, 3e-6, 2e-4, tag + " grad " + keys[i])
            opt.step(net.ps.data, grad, lr)
            for i in range(len(keys)):
                var ww = nested(steps.at(st).get("weights_after").get(keys[i]))
                assert_list_close32(torch_layout(net, i, net.ps.data), ww, 3e-5, 1e-4, tag + " weights " + keys[i])
            _ = len(cc.logits)
            _ = len(cc.values)
        _ = len(acts)
        _ = len(advs)
        _ = len(rets)
        _ = len(olp)
        print("  synth_ppo_golden '" + name + "': forward, loss, gradients and 3 Adam steps OK")


# ------------------------------------------------------------------------------------------------ shipped manifests


def test_shipped_manifests() raises:
    if skip_without_models("test_shipped_manifests"):
        return
    var rows = read_model_index(model_root(), "synth")
    if len(rows) != 9:
        raise Error("expected 9 shipped synth models, found " + String(len(rows)))
    var worst: Float64 = 0.0
    var vectors = 0
    for r in range(len(rows)):
        var m = load_model_file(rows[r].path)
        var net = BasicNet.from_model(m)
        var pol = MLPPolicy.from_model(m)
        assert_close(Float64(net.spec.obs_size), Float64(m.obs_size), 0.0, 0.0, rows[r].id + " obs size")
        assert_close(Float64(net.spec.num_actions), Float64(m.num_actions()), 0.0, 0.0, rows[r].id + " actions")
        var total = 0
        for i in range(len(m.tensors)):
            total += len(m.tensors[i])
        assert_close(Float64(net.param_count()), Float64(total), 0.0, 0.0, rows[r].id + " param count")
        assert_close(Float64(net.spec.params()), Float64(total), 0.0, 0.0, rows[r].id + " spec.params")
        var man = parse_json(m.manifest)
        var tv = man.get("test_vectors")
        var A = net.spec.num_actions
        var P = net.spec.num_twists()
        var sc = ActorScratch(net.spec)
        var lg = List[Float32](length=A, fill=0.0)
        var sb = SparseBatch()
        var want = List[Float32]()
        var wantv = List[Float32]()
        for t in range(len(tv)):
            var obs = _span_of(ints(tv.at(t).get("obs")))
            var v = net.forward_one(Span(obs), -1, sc, _p(lg))
            var rl = floats32(tv.at(t).get("logits"))
            for a in range(A):
                worst = max(worst, abs(Float64(lg[a]) - Float64(rl[a])))
            worst = max(worst, abs(Float64(v) - tv.at(t).get("value").as_float()))
            sb.append_row(Span(obs), -1)
            for a in range(A):
                want.append(rl[a])
            wantv.append(Float32(tv.at(t).get("value").as_float()))
            # every twist, and the ensemble average (each term divided by P before accumulation, as the runtime does)
            var re = floats32(tv.at(t).get("logits_ensemble"))
            var acc = List[Float32](length=A, fill=0.0)
            var vacc: Float32 = 0.0
            var fp = Float32(P)
            for p in range(P):
                var vp = net.forward_one(Span(obs), p, sc, _p(lg))
                # the same twist through the independent inference implementation
                var ref_l = List[Float32]()
                var ref_v = pol.forward_sym(_ints_of(obs), p, ref_l)
                for a in range(A):
                    worst = max(worst, abs(Float64(lg[a]) - Float64(ref_l[a])))
                worst = max(worst, abs(Float64(vp) - Float64(ref_v)))
                vacc += vp / fp
                for a in range(A):
                    acc[a] += lg[a] / fp
            if P == 0:
                raise Error(rows[r].id + " has no symmetries")
            for a in range(A):
                worst = max(worst, abs(Float64(acc[a]) - Float64(re[a])))
            worst = max(worst, abs(Float64(vacc) - tv.at(t).get("value_ensemble").as_float()))
            vectors += 1
        var cc = ChunkCache(net.spec, max(len(tv), 1))
        net.forward_chunk(sb, 0, sb.rows(), cc)
        for t in range(len(tv)):
            for a in range(A):
                worst = max(worst, abs(Float64(cc.logits[t * A + a]) - Float64(want[t * A + a])))
            worst = max(worst, abs(Float64(cc.values[t]) - Float64(wantv[t])))
    if worst > 1e-5:
        raise Error("shipped manifest vectors: worst error " + String(worst))
    print("  9 shipped manifests, " + String(vectors) + " vectors (no twist, every twist, ensemble; forward_one and forward_chunk) OK, worst " + String(worst))


def _ints_of(a: List[Int32]) -> List[Int]:
    var out = List[Int](capacity=len(a))
    for i in range(len(a)):
        out.append(Int(a[i]))
    return out^


# ------------------------------------------------------------------------------------------------ finite differences


struct RandBatch(Movable):
    var net: BasicNet
    var sb: SparseBatch
    var acts: List[Int32]
    var advs: List[Float32]
    var rets: List[Float32]
    var olp: List[Float32]

    def __init__(out self, var net: BasicNet, var sb: SparseBatch, var acts: List[Int32], var advs: List[Float32], var rets: List[Float32], var olp: List[Float32]):
        self.net = net^
        self.sb = sb^
        self.acts = acts^
        self.advs = advs^
        self.rets = rets^
        self.olp = olp^


def random_net_batch(seed: UInt64, n: Int) raises -> RandBatch:
    """A random net with two common layers, a hidden action and a hidden value layer, two twists, and `n` random rows
    whose old_lp keeps every ratio inside the clip band (so the loss is smooth)."""
    var s = NetSpec()
    s.obs_size = 10
    s.num_actions = 4
    s.emb = 7
    s.common = [6, 5]
    s.policy_hidden = [4]
    s.value_hidden = [3]
    var rng = Rng(seed)
    s.obs_perms = [[1, 2, 0, 4, 5, 3, 7, 8, 9, 6], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
    s.act_perms = [[1, 2, 3, 0], [3, 2, 1, 0]]
    var net = BasicNet(s, seed)
    # make biases non-trivial so ReLU units are not dead at 0
    for i in range(net.param_count()):
        net.ps.data[i] = net.ps.data[i] * 2.0
    var sb = SparseBatch()
    var acts = List[Int32]()
    var advs = List[Float32]()
    var rets = List[Float32]()
    for _ in range(n):
        var k = 1 + rng.below(5)
        var idx = List[Int]()
        for _ in range(k):
            idx.append(rng.below(10))
        sb.append_row_ints(idx, rng.below(3) - 1)
        acts.append(Int32(rng.below(4)))
        advs.append(Float32(rng.normal()))
        rets.append(Float32(rng.normal()))
    var olp = List[Float32](length=n, fill=0.0)
    var cc = ChunkCache(s, n)
    net.forward_chunk(sb, 0, n, cc)
    for r in range(n):
        var row = List[Float32](length=4, fill=0.0)
        for a in range(4):
            row[a] = cc.logits[r * 4 + a]
        olp[r] = old_logp_row(_p(row), 4, Int(acts[r])) + Float32(0.01 * (rng.uniform() - 0.5))
    return RandBatch(net^, sb^, acts^, advs^, rets^, olp^)


def loss_of(net: BasicNet, sb: SparseBatch, mut acts: List[Int32], mut advs: List[Float32], mut rets: List[Float32], mut olp: List[Float32], cfg: PPOLossCfg, mut grad: List[Float32]) -> Float64:
    var n = sb.rows()
    var A = net.spec.num_actions
    var cc = ChunkCache(net.spec, n)
    net.forward_chunk(sb, 0, n, cc)
    var dz = List[Float32](length=n * A, fill=0.0)
    var dv = List[Float32](length=n, fill=0.0)
    var parts = ppo_chunk_loss(_p(cc.logits), _p(cc.values), n, A, _ip(acts), _p(advs), _p(rets), _p(olp), cfg, Float32(1.0) / Float32(n), _p(dz), _p(dv))
    for i in range(len(grad)):
        grad[i] = 0.0
    net.backward_chunk(sb, 0, n, cc, _p(dz), _p(dv), _p(grad))
    var nf = Float64(n)
    _ = len(cc.logits)
    _ = len(cc.values)
    _ = len(acts)
    _ = len(advs)
    _ = len(rets)
    _ = len(olp)
    _ = len(grad)
    return parts.policy / nf + Float64(cfg.vf_coef) * (parts.value / nf) - Float64(cfg.ent_coef) * (parts.entropy / nf)


def test_finite_difference() raises:
    var cfg = PPOLossCfg(0.1, 0.8, 0.05)
    var n = 9
    var rt = random_net_batch(77, n)
    ref net = rt.net
    ref sb = rt.sb
    ref acts = rt.acts
    ref advs = rt.advs
    ref rets = rt.rets
    ref olp = rt.olp
    var grad = List[Float32](length=net.param_count(), fill=0.0)
    _ = loss_of(net, sb, acts, advs, rets, olp, cfg, grad)
    var scratch = List[Float32](length=net.param_count(), fill=0.0)
    var eps: Float32 = 2e-3
    var worst: Float64 = 0.0
    var checked = 0
    for i in range(net.param_count()):
        var orig = net.ps.data[i]
        net.ps.data[i] = orig + eps
        var lp = loss_of(net, sb, acts, advs, rets, olp, cfg, scratch)
        net.ps.data[i] = orig - eps
        var lm = loss_of(net, sb, acts, advs, rets, olp, cfg, scratch)
        net.ps.data[i] = orig
        var fd = (lp - lm) / (2.0 * Float64(eps))
        var err = abs(fd - Float64(grad[i]))
        worst = max(worst, err)
        if err > 1e-3 + 2e-2 * abs(fd):
            raise Error("gradient of parameter " + String(i) + " analytic " + String(grad[i]) + " vs finite difference " + String(fd))
        checked += 1
    print("  finite-difference check of every parameter (" + String(checked) + " params) OK, worst " + String(worst))


# ------------------------------------------------------------------------------------------------ guards and edge cases


def test_guards_and_edges() raises:
    var cfg = PPOLossCfg(0.1, 0.8, 0.01)
    var n = 6
    var rt = random_net_batch(5, n)
    ref net = rt.net
    ref sb = rt.sb
    ref acts = rt.acts
    ref advs = rt.advs
    ref rets = rt.rets
    ref olp = rt.olp
    var A = net.spec.num_actions
    # a cache larger than the chunk: the unused space must stay untouched by forward_chunk
    var cc = ChunkCache(net.spec, n + 5)
    var sentinel = bitcast[DType.float32](UInt32(0xBEEFCAFE))
    for i in range(len(cc.logits)):
        cc.logits[i] = sentinel
    for i in range(len(cc.values)):
        cc.values[i] = sentinel
    net.forward_chunk(sb, 0, n, cc)
    for i in range(n * A, len(cc.logits)):
        if bitcast[DType.uint32](cc.logits[i]) != 0xBEEFCAFE:
            raise Error("forward_chunk wrote beyond its logits")
    for i in range(n, len(cc.values)):
        if bitcast[DType.uint32](cc.values[i]) != 0xBEEFCAFE:
            raise Error("forward_chunk wrote beyond its values")
    # backward_chunk: guards around grad, dz, dv
    var plen = net.param_count()
    var gg = guard_alloc(plen)
    var dzg = guard_alloc(n * A)
    var dvg = guard_alloc(n)
    var dz = List[Float32](length=n * A, fill=0.0)
    var dv = List[Float32](length=n, fill=0.0)
    var logits = List[Float32](length=n * A, fill=0.0)
    var values = List[Float32](length=n, fill=0.0)
    for i in range(n * A):
        logits[i] = cc.logits[i]
    for i in range(n):
        values[i] = cc.values[i]
    _ = ppo_chunk_loss(_p(logits), _p(values), n, A, _ip(acts), _p(advs), _p(rets), _p(olp), cfg, Float32(1.0) / Float32(n), _p(dz), _p(dv))
    guard_set(dzg, dz)
    guard_set(dvg, dv)
    net.backward_chunk(sb, 0, n, cc, guard_ptr(dzg), guard_ptr(dvg), guard_ptr(gg))
    guard_check(gg, plen, "backward_chunk grad")
    guard_check(dzg, n * A, "backward_chunk dz (read only)")
    guard_check(dvg, n, "backward_chunk dv (read only)")
    # the guarded run equals the plain run (grad accumulates: two calls give twice the gradient)
    var plain = List[Float32](length=plen, fill=0.0)
    net.backward_chunk(sb, 0, n, cc, _p(dz), _p(dv), _p(plain))
    assert_list_close32(guard_data(gg, plen), plain, 0.0, 0.0, "guarded vs plain gradient")
    net.backward_chunk(sb, 0, n, cc, _p(dz), _p(dv), _p(plain))
    for i in range(plen):
        var d = abs(Float64(plain[i]) - 2.0 * Float64(guard_data(gg, plen)[i]))
        if d > 1e-6 * (1.0 + abs(Float64(plain[i]))):
            raise Error("backward_chunk must accumulate (+=) into grad")
    # out-of-range observation indices are skipped (as MLPPolicy does) and never write outside grad
    var sb2 = SparseBatch()
    var junk: List[Int32] = [3, -1, 4000, 7, 10]
    sb2.append_row(Span(junk), -1)
    var good: List[Int32] = [3, 7]
    sb2.append_row(Span(good), -1)
    var cc2 = ChunkCache(net.spec, 2)
    net.forward_chunk(sb2, 0, 2, cc2)
    for a in range(A):
        if bitcast[DType.uint32](cc2.logits[a]) != bitcast[DType.uint32](cc2.logits[A + a]):
            raise Error("out-of-range indices must be skipped, not summed")
    var dz2 = List[Float32](length=2 * A, fill=0.5)
    var dv2 = List[Float32](length=2, fill=0.25)
    var gg2 = guard_alloc(plen)
    net.backward_chunk(sb2, 0, 2, cc2, _p(dz2), _p(dv2), guard_ptr(gg2))
    guard_check(gg2, plen, "backward_chunk with out-of-range indices")
    # forward_one: the logit buffer is written for exactly num_actions entries; junk indices are skipped
    var sc = ActorScratch(net.spec)
    var lgg = guard_alloc(A)
    var v1 = net.forward_one(Span(junk), -1, sc, guard_ptr(lgg))
    guard_check(lgg, A, "forward_one logits")
    var lgg2 = guard_alloc(A)
    var v2 = net.forward_one(Span(good), -1, sc, guard_ptr(lgg2))
    if bitcast[DType.uint32](v1) != bitcast[DType.uint32](v2):
        raise Error("forward_one: junk indices changed the value")
    assert_list_close32(guard_data(lgg, A), guard_data(lgg2, A), 0.0, 0.0, "forward_one junk indices")
    # an empty row is the bias path only; forward_one and forward_chunk agree on it
    var empty = List[Int32]()
    var sb3 = SparseBatch()
    sb3.append_row(Span(empty), 1)
    var cc3 = ChunkCache(net.spec, 1)
    net.forward_chunk(sb3, 0, 1, cc3)
    var l3 = List[Float32](length=A, fill=0.0)
    var v3 = net.forward_one(Span(empty), 1, sc, _p(l3))
    for a in range(A):
        assert_close(Float64(cc3.logits[a]), Float64(l3[a]), 1e-6, 1e-6, "empty row logits")
    assert_close(Float64(cc3.values[0]), Float64(v3), 1e-6, 1e-6, "empty row value")
    # forward_chunk / forward_one agreement on a random batch for every row
    var sc2 = ActorScratch(net.spec)
    var l4 = List[Float32](length=A, fill=0.0)
    for r in range(n):
        var idx = List[Int32]()
        for q in range(Int(sb.off[r]), Int(sb.off[r + 1])):
            idx.append(sb.idx[q])
        var v4 = net.forward_one(Span(idx), Int(sb.perm[r]), sc2, _p(l4))
        for a in range(A):
            assert_close(Float64(cc.logits[r * A + a]), Float64(l4[a]), 2e-6, 2e-6, "chunk vs one logits")
        assert_close(Float64(cc.values[r]), Float64(v4), 2e-6, 2e-6, "chunk vs one value")
    _ = len(acts)
    _ = len(advs)
    _ = len(rets)
    _ = len(olp)
    _ = len(logits)
    _ = len(values)
    _ = len(dz)
    _ = len(dv)
    _ = len(plain)
    print("  guard regions, junk indices, accumulation and forward_one / forward_chunk agreement OK")


def test_grad_bufs() raises:
    var g = GradBufs(10, 4)
    for b in range(4):
        var p = g.ptr(b)
        for i in range(10):
            p[unsafe_offset=i] = Float32(b * 100 + i)
    var out = List[Float32]()
    g.reduce_into(out)
    for i in range(10):
        assert_close(Float64(out[i]), Float64(0 + 100 + 200 + 300 + 4 * i), 0.0, 0.0, "reduce " + String(i))
    g.zero()
    g.reduce_into(out)
    for i in range(10):
        assert_close(Float64(out[i]), 0.0, 0.0, 0.0, "zeroed reduce")
    print("  GradBufs OK")


def test_from_model_errors() raises:
    # a container without a value head, a broken chain, and the shipped layout round trip
    if skip_without_models("test_from_model_errors"):
        return
    var rows = read_model_index(model_root(), "synth")
    var m = load_model_file(rows[0].path)
    var net = BasicNet.from_model(m)
    # re-created from the spec: same tensor order and sizes as the container
    var fresh = BasicNet(net.spec, 3)
    assert_close(Float64(fresh.param_count()), Float64(net.param_count()), 0.0, 0.0, "fresh layout size")
    for i in range(net.ps.count()):
        if fresh.ps.names[i] != net.ps.names[i] or fresh.ps.size_of(i) != net.ps.size_of(i):
            raise Error("layout differs from the container at tensor " + String(i))
        # the container holds the tensors in this very order and shape
        var ti = m.tensor_index(net.ps.names[i])
        if ti != i:
            raise Error("container tensor order differs for " + net.ps.names[i])
        var numel = 1
        for d in range(len(m.shapes[ti])):
            numel *= m.shapes[ti][d]
        if numel != net.ps.size_of(i):
            raise Error("container shape differs for " + net.ps.names[i])
    var broken = load_model_file(rows[0].path)
    var vi = broken.tensor_index("value.0.weight")
    broken.shapes[vi] = [2, broken.shapes[vi][1]]
    var raised = False
    try:
        _ = BasicNet.from_model(broken)
    except e:
        raised = String(e).startswith("QM-E647")
    if not raised:
        raise Error("a value head that does not end in one output must raise QM-E647")
    var missing = load_model_file(rows[0].path)
    missing.names[missing.tensor_index("action.0.weight")] = "act.0.weight"
    raised = False
    try:
        _ = BasicNet.from_model(missing)
    except e:
        raised = String(e).startswith("QM-E647")
    if not raised:
        raise Error("a container without an action head must raise QM-E647")
    print("  from_model layout and QM-E647 errors OK")


def main() raises:
    test_param_store()
    test_param_counts()
    test_init()
    test_grad_bufs()
    test_forward_golden()
    test_synth_golden()
    test_shipped_manifests()
    test_finite_difference()
    test_guards_and_edges()
    test_from_model_errors()
    print("test_rl_nn OK")
