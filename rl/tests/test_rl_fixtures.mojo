"""Tests for the RL golden fixtures (tests/fixtures): every file parses with qrl.json, stays within its
size budget, and satisfies a few structural invariants that the consuming test files rely on. Also exercises the
helpers of qrl.testutil that other tests import."""

from std.math import cos, pi, sqrt
from std.memory import bitcast
from qrl.json import JsonValue, parse_json
from qrl.testutil import (
    FPtr,
    GUARD,
    Chi2,
    load_fixture,
    load_fixture_text,
    load_fixture_bytes,
    fixture_size,
    ints,
    ints2,
    floats,
    floats32,
    flat_floats,
    flat_floats32,
    strings,
    close,
    abs64,
    max_abs_diff,
    max_abs_diff32,
    assert_close,
    assert_list_close,
    assert_ints_equal,
    two_sample_z,
    cluster_mean_se,
    chi_square_z,
    chi_square_gof,
    chi_square_homogeneity,
    ecdf_at,
    ks_against_knots,
    ks_critical,
    gemm_ref,
    rand_fill,
    guard_alloc,
    guard_ptr,
    guard_data,
    guard_set,
    guard_check,
)

# Per-file size ceilings (bytes). Targets of MOJO_PORT.md 13.1 where they were honoured; the few larger files are
# listed in PROVENANCE.md. The total budget is 2,000,000 bytes.
comptime TOTAL_BUDGET = 2000000


def expect(cond: Bool, msg: String) raises:
    if not cond:
        raise Error("test_rl_fixtures: " + msg)


struct Entry(Copyable):
    var rel: String
    var ceiling: Int

    def __init__(out self, rel: String, ceiling: Int):
        self.rel = rel
        self.ceiling = ceiling


def entries() -> List[Entry]:
    var e = List[Entry]()
    e.append(Entry("envs/gateset_twists.json", 56000))
    e.append(Entry("envs/steps_permutation.json", 100000))
    e.append(Entry("envs/steps_linear_function.json", 105000))
    e.append(Entry("envs/steps_clifford.json", 145000))
    e.append(Entry("envs/steps_pauli_network.json", 240000))
    e.append(Entry("envs/steps_custom_args.json", 75000))
    e.append(Entry("envs/inverts.json", 16000))
    e.append(Entry("envs/pauli_perm_cases.json", 150000))
    e.append(Entry("envs/reset_exact.json", 200000))
    e.append(Entry("envs/reset_empirical.json", 6000))
    e.append(Entry("envs/defaults.json", 2000))
    e.append(Entry("envs/pauli_twists.json", 40000))
    e.append(Entry("envs/reward_fma_cases.json", 8000))
    e.append(Entry("ppo/policy_forward.json", 20000))
    e.append(Entry("ppo/gae.json", 8000))
    e.append(Entry("ppo/ppo_update.json", 44000))
    e.append(Entry("ppo/safetensors_header.json", 2000))
    e.append(Entry("ppo/rollout_stats.json", 60000))
    e.append(Entry("ppo/py_curve.json", 8000))
    e.append(Entry("torch/adam_cosine_golden.json", 4000))
    e.append(Entry("torch/f16_golden.json", 12000))
    e.append(Entry("torch/misc_golden.json", 4000))
    e.append(Entry("torch/bc_golden.json", 170000))
    e.append(Entry("torch/ppo_route_golden.json", 180000))
    e.append(Entry("torch/synth_ppo_golden.json", 52000))
    e.append(Entry("topologies.json", 100000))
    e.append(Entry("route/jobs_stats.json", 40000))
    e.append(Entry("route/py_bc_reference.json", 10000))
    return e^


def test_sizes_and_parse() raises:
    var total = 0
    var es = entries()
    for i in range(len(es)):
        var text = load_fixture_text(es[i].rel)
        var n = text.byte_length()
        expect(n <= es[i].ceiling, es[i].rel + " is " + String(n) + " bytes, ceiling " + String(es[i].ceiling))
        var v = parse_json(text)
        expect(v.is_object() or v.is_array(), es[i].rel + " is not a JSON container")
        total += n
    # binary and markdown fixtures
    var st = len(load_fixture_bytes("ppo/tiny_policy.safetensors"))
    total += st
    total += load_fixture_text("PROVENANCE.md").byte_length()
    expect(total <= TOTAL_BUDGET, "fixtures total " + String(total) + " bytes exceeds " + String(TOTAL_BUDGET))
    expect(fixture_size("ppo/tiny_policy.safetensors") == st, "fixture_size")


def test_envs() raises:
    # gateset_twists: 44 configs, internally consistent
    var g = load_fixture("envs/gateset_twists.json")
    expect(len(g) == 44, "gateset_twists has 44 configs")
    var kinds = List[String]()
    for i in range(len(g)):
        var c = g.at(i)
        var k = c.get("kind").as_string()
        var seen = False
        for j in range(len(kinds)):
            if kinds[j] == k:
                seen = True
        if not seen:
            kinds.append(k)
        var a = c.get("num_actions").as_int()
        expect(a == len(c.get("gateset")), "num_actions == len(gateset) in config " + String(i))
        var is_pauli = k == "pauli_network"
        var perm_key = String("obs_perms")
        if is_pauli:
            perm_key = String("qubit_perms")  # the Pauli env permutes internally; no obs_perms are returned to the policy
        var np = len(c.get(perm_key))
        expect(np == len(c.get("act_perms")), "perm table lengths in config " + String(i))
        expect(len(c.get("edges")) > 0, "edges in config " + String(i))
        expect(np > 0, "at least the identity symmetry in config " + String(i))
        expect(len(c.get("act_perms").at(0)) == a, "act_perm width in config " + String(i))
        if is_pauli:
            expect(len(c.get(perm_key).at(0)) == c.get("num_qubits").as_int(), "qubit_perm width in config " + String(i))
        else:
            var shape = ints(c.get("obs_shape"))
            var prod = 1
            for j in range(len(shape)):
                prod *= shape[j]
            expect(len(c.get("obs_perms").at(0)) == prod, "obs_perm width in config " + String(i))
    expect(len(kinds) == 4, "gateset_twists covers the four env kinds")

    # step files
    var names = List[String]()
    names.append("envs/steps_permutation.json")
    names.append("envs/steps_linear_function.json")
    names.append("envs/steps_clifford.json")
    names.append("envs/steps_pauli_network.json")
    var counts = List[Int]()
    counts.append(3)
    counts.append(3)
    counts.append(3)
    counts.append(4)
    for f in range(len(names)):
        var s = load_fixture(names[f])
        expect(len(s) == counts[f], names[f] + ": config count")
        for i in range(len(s)):
            var cases = s.at(i).get("cases")
            expect(len(cases) == 24, names[f] + ": 24 start states per config")
            for j in range(len(cases)):
                var cs = cases.at(j)
                expect(cs.has("state") and cs.has("init") and cs.has("steps"), names[f] + ": case keys")
                expect(len(cs.get("steps")) > 0, names[f] + ": steps present")
                var st0 = cs.get("steps").at(0)
                expect(st0.has("obs") and st0.has("reward") and st0.has("is_final") and st0.has("action"), names[f] + ": step keys")
    var cu = load_fixture("envs/steps_custom_args.json")
    expect(len(cu) == 4, "steps_custom_args has 4 configs")
    expect(cu.at(3).get("constructor").get("max_rotations").as_int() == 2, "custom pauli constructor args")
    expect(close(cu.at(0).get("metrics_weights").get("n_cnots").as_float(), 0.02, 0.0, 1e-6), "custom weights")

    # inverts
    var iv = load_fixture("envs/inverts.json")
    expect(len(iv) == 3, "inverts has 3 configs")
    for i in range(len(iv)):
        var cs = iv.at(i).get("cases")
        expect(len(cs) == 20, "inverts: 20 cases per config")
        expect(cs.at(0).has("obs_plain") and cs.at(0).has("obs_inverted"), "inverts: obs keys")

    # pauli_perm_cases
    var pp = load_fixture("envs/pauli_perm_cases.json")
    expect(len(pp) == 4, "pauli_perm_cases has 4 topologies")
    for i in range(len(pp)):
        var t = pp.at(i)
        var n_perm = t.get("num_perms").as_int()
        expect(len(t.get("qubit_perms")) == n_perm and len(t.get("act_perms")) == n_perm, "pauli_perm_cases: perm tables")
        var cs = t.get("cases")
        expect(len(cs) == 10, "pauli_perm_cases: 10 cases")
        expect(len(cs.at(0).get("obs_by_perm")) == n_perm, "pauli_perm_cases: obs_by_perm per perm")

    # defaults
    var d = load_fixture("envs/defaults.json")
    expect(d.get("f32_thresholds_k_over_2pow24").get("cx_if_k_ge").as_int() == 5033166, "defaults: cx threshold")
    expect(d.get("gym_constructor_defaults").get("max_depth").as_int() == 128, "defaults: max_depth")


def _prob_sum(dist: JsonValue) raises -> Tuple[Float64, Float64]:
    """Sum of the exact fractions ('num/den', element 1) and of the float column (last element)."""
    var s_frac: Float64 = 0.0
    var s_float: Float64 = 0.0
    for i in range(len(dist)):
        var row = dist.at(i)
        var txt = row.at(1).as_string()
        var parts = txt.split("/")
        var num = Float64(parts[0])
        var den: Float64 = 1.0
        if len(parts) > 1:
            den = Float64(parts[1])
        s_frac += num / den
        s_float += row.at(2).as_float()
    return (s_frac, s_float)


def test_reset() raises:
    var r = load_fixture("envs/reset_exact.json")
    var sc = r.get("scramble")
    expect(len(sc) == 15, "reset_exact: 15 scramble cases")
    for i in range(len(sc)):
        var t = _prob_sum(sc.at(i).get("dist"))
        expect(abs64(t[0] - 1.0) < 1e-12, "scramble case " + String(i) + ": exact fractions sum to 1")
        expect(abs64(t[1] - 1.0) < 1e-5, "scramble case " + String(i) + ": float column sums to 1")
    var tb = r.get("pauli_tableau")
    expect(len(tb) == 3, "reset_exact: 3 tableau cases")
    for i in range(len(tb)):
        var t = _prob_sum(tb.at(i).get("dist"))
        expect(abs64(t[0] - 1.0) < 1e-12, "tableau case " + String(i) + ": exact fractions sum to 1")
        expect(tb.at(i).get("difficulty").as_int() <= 4, "tableau difficulty bound")
    var ro = r.get("pauli_rotations")
    expect(len(ro) == 4, "reset_exact: 4 rotation cases")
    for i in range(len(ro)):
        var t = _prob_sum(ro.at(i).get("dist"))
        expect(abs64(t[0] - 1.0) < 1e-12, "rotation case " + String(i) + ": exact fractions sum to 1")
    var e = load_fixture("envs/reset_empirical.json")
    var h = e.get("pauli_rotation_count_hist")
    expect(len(h) == 12, "reset_empirical: 12 rotation histograms")
    for i in range(len(h)):
        var total = 0
        var hist = h.at(i).get("visible_rotations_hist")
        var ks = hist.keys()
        for j in range(len(ks)):
            total += hist.get(ks[j]).as_int()
        expect(total == h.at(i).get("N").as_int(), "rotation histogram sums to N")
    expect(len(e.get("solved_at_reset_rate")) == 13, "reset_empirical: 13 solved-at-reset rates")


def test_pauli_twists() raises:
    var p = load_fixture("envs/pauli_twists.json")
    var cs = p.get("cases")
    expect(len(cs) == 6, "pauli_twists: 6 topologies")
    var want_autos = List[Int]()
    want_autos.append(10)  # ring:5
    want_autos.append(8)  # ring:4
    want_autos.append(6)  # star:4
    want_autos.append(4)  # grid:2x3
    want_autos.append(8)  # heavyhex_frag:H7
    want_autos.append(2)  # heavyhex_frag:T5
    for i in range(len(cs)):
        var c = cs.at(i)
        var n = c.get("num_qubits").as_int()
        var nact = len(c.get("actions"))
        var autos = c.get("automorphisms")
        expect(len(autos) == want_autos[i], c.get("topology").as_string() + ": automorphism count")
        expect(len(c.get("obs_perms")) == len(autos) and len(c.get("act_perms")) == len(autos), "one table per automorphism")
        var osz = c.get("obs_size").as_int()
        expect(osz == 2 * n * (2 * n + c.get("max_rotations").as_int()), "obs_size formula")
        for k in range(len(autos)):
            var op = ints(c.get("obs_perms").at(k))
            var ap = ints(c.get("act_perms").at(k))
            expect(len(op) == osz and len(ap) == nact, "perm widths")
            var seen_o = List[Bool](length=osz, fill=False)
            for j in range(osz):
                expect(op[j] >= 0 and op[j] < osz and not seen_o[op[j]], "obs perm is a permutation")
                seen_o[op[j]] = True
            var seen_a = List[Bool](length=nact, fill=False)
            for j in range(nact):
                expect(ap[j] >= 0 and ap[j] < nact and not seen_a[ap[j]], "act perm is a permutation")
                seen_a[ap[j]] = True
    # ring:5 has non-involutive automorphisms (they pin the direction of the relation)
    expect(cs.at(0).get("non_involutive_automorphisms").as_int() == 4, "ring:5 non-involutive count")
    expect(cs.at(1).get("non_involutive_automorphisms").as_int() == 2, "ring:4 non-involutive count")
    expect(cs.at(2).get("non_involutive_automorphisms").as_int() == 2, "star:4 non-involutive count")


@no_inline
def _mul32(a: Float32, b: Float32) -> Float32:
    return a * b


def test_reward_fma() raises:
    var f = load_fixture("envs/reward_fma_cases.json")
    var cs = f.get("cases")
    expect(len(cs) == 33, "reward_fma_cases: documented example plus 32 combinations")
    expect(cs.at(0).has("example"), "first case is the documented example")
    for i in range(len(cs)):
        var c = cs.at(i)
        var ach = Float32(c.get("achieved").as_float())
        var pen = Float32(c.get("penalty").as_float())
        var w = Float32(c.get("pauli_layer_reward").as_float())
        var cnt = Float32(c.get("count").as_int())
        var base = ach - pen
        var prod = _mul32(w, cnt)
        var unfused = base + prod
        expect(Int(bitcast[DType.uint32](unfused)) == c.get("unfused_bits").as_int(), "case " + String(i) + ": unfused reward bits")
        expect(c.get("unfused_bits").as_int() != c.get("fused_bits").as_int(), "case " + String(i) + ": fused differs")
        expect(Int(bitcast[DType.uint32](Float32(c.get("unfused").as_float()))) == c.get("unfused_bits").as_int(), "case " + String(i) + ": unfused value/bits agree")
        expect(Int(bitcast[DType.uint32](Float32(c.get("fused").as_float()))) == c.get("fused_bits").as_int(), "case " + String(i) + ": fused value/bits agree")


def test_ppo() raises:
    var pf = load_fixture("ppo/policy_forward.json")
    var obs = pf.get("obs_size").as_int()
    var emb = pf.get("emb").as_int()
    var a = pf.get("num_actions").as_int()
    var c = ints(pf.get("common"))
    expect(obs == 9 and emb == 8 and a == 4 and len(c) == 1 and c[0] == 6, "policy_forward architecture")
    var order = strings(pf.get("tensor_order"))
    expect(len(order) == 8, "8 tensors")
    var n_params = 0
    for i in range(len(order)):
        var t = pf.get("weights").get(order[i])
        var shape = ints(t.get("shape"))
        var cnt = 1
        for j in range(len(shape)):
            cnt *= shape[j]
        expect(len(t.get("data")) == cnt, order[i] + ": data length matches shape")
        n_params += cnt
    expect(n_params == obs * emb + emb + emb * c[0] + c[0] + c[0] * a + a + c[0] + 1, "parameter count formula")
    expect(len(pf.get("cases")) == 15, "policy_forward: 15 cases")
    expect(len(pf.get("ensemble_cases")) == 5, "policy_forward: 5 ensemble cases")

    # GAE (twisterl form, f32) reproduces the four golden settings
    var gae = load_fixture("ppo/gae.json").get("cases")
    expect(len(gae) == 4, "gae: 4 settings")
    for k in range(len(gae)):
        var gc = gae.at(k)
        var r = floats32(gc.get("rewards"))
        var v = floats32(gc.get("values"))
        var n = len(r)
        var gm = Float32(gc.get("gamma").as_float())
        var lm = Float32(gc.get("lambda").as_float())
        var adv = List[Float32](length=n, fill=0.0)
        var ret = List[Float32](length=n, fill=0.0)
        adv[n - 1] = r[n - 1] - v[n - 1]
        ret[n - 1] = r[n - 1]
        var t = n - 2
        while t >= 0:
            var inner = _mul32(lm, adv[t + 1])
            var boot = v[t + 1] + inner
            var scaled = _mul32(gm, boot)
            ret[t] = r[t] + scaled
            adv[t] = ret[t] - v[t]
            t -= 1
        expect(max_abs_diff32(adv, floats32(gc.get("advs"))) <= 1e-6, "gae advs, setting " + String(k))
        expect(max_abs_diff32(ret, floats32(gc.get("rets"))) <= 1e-6, "gae rets, setting " + String(k))

    # ppo_update: batch layout and the all-masked row (correction C-1: old_logp is exactly 0.0)
    var up = load_fixture("ppo/ppo_update.json")
    var b = up.get("batch")
    expect(len(b.get("obs")) == 14 and len(b.get("old_logp")) == 14, "ppo_update: 14 rows")
    var lm13 = floats(b.get("old_logits_masked").at(13))
    for i in range(len(lm13)):
        expect(lm13[i] == -1e10, "row 13 is all masked")
    expect(b.get("old_logp").at(13).as_float() == 0.0, "all-masked row has old_logp 0.0")
    expect(len(up.get("epochs")) == 10, "ppo_update: 10 epochs")
    expect(up.get("epochs").at(0).has("grads") and up.get("epochs").at(0).has("weights_after"), "epoch 0 has grads and weights")

    # safetensors: byte layout of the tiny policy checkpoint
    var raw = load_fixture_bytes("ppo/tiny_policy.safetensors")
    var hlen = 0
    for i in range(8):
        hlen += Int(raw[i]) << (8 * i)
    var hdr = load_fixture("ppo/safetensors_header.json")
    expect(hlen == hdr.get("header_len_bytes").as_int(), "safetensors header length")
    var hbytes = List[UInt8](capacity=hlen)
    for i in range(hlen):
        hbytes.append(raw[8 + i])
    var htext = String(unsafe_from_utf8=Span(hbytes))
    var hj = parse_json(htext)
    var last = 0
    var hk = hj.keys()
    for i in range(len(hk)):
        if hk[i] == "__metadata__":
            continue
        var off = ints(hj.get(hk[i]).get("data_offsets"))
        if off[1] > last:
            last = off[1]
        expect(hj.get(hk[i]).get("dtype").as_string() == "F32", "safetensors dtype F32")
    expect(len(raw) == 8 + hlen + last, "safetensors file size = 8 + header + data")


def test_rollout_stats() raises:
    var rs = load_fixture("ppo/rollout_stats.json")
    var cfgs = rs.get("configs")
    expect(len(cfgs) == 4, "rollout_stats: 4 configs")
    for i in range(len(cfgs)):
        var c = cfgs.at(i)
        var emb = c.get("emb").as_int()
        var com = ints(c.get("common"))
        var obs = c.get("obs_size").as_int()
        var a = c.get("num_actions").as_int()
        var wts = c.get("weights")
        var total = 0
        var ks = wts.keys()
        for j in range(len(ks)):
            total += len(wts.get(ks[j]).get("data"))
        expect(total == obs * emb + emb + emb * com[0] + com[0] + com[0] * a + a + com[0] + 1, "rollout_stats: parameter count of config " + String(i))
        var ds = c.get("difficulties")
        expect(len(ds) == 3, "rollout_stats: 3 difficulties per config")
        for j in range(len(ds)):
            var d = ds.at(j)
            var col = d.get("collector")
            expect(col.get("episodes").as_int() >= 5000, "collector episodes >= 5000")
            var hist = col.get("len_hist")
            var hk = hist.keys()
            var eps = 0
            var rows = 0
            for q in range(len(hk)):
                var cnt = hist.get(hk[q]).as_int()
                eps += cnt
                rows += cnt * Int(hk[q])
            expect(eps == col.get("episodes").as_int(), "len_hist sums to the episode count")
            expect(rows == col.get("rows").as_int(), "len_hist rows == total rows")
            expect(d.get("evaluate_deterministic").get("episodes").as_int() >= 20000, "evaluate episodes >= 20000")
            expect(d.get("evaluate_sampled10").get("episodes").as_int() >= 20000, "sampled evaluate episodes >= 20000")
            var stats = col.get("stats")
            expect(stats.has("rets") and stats.has("advs") and stats.has("values") and stats.has("rewards") and stats.has("old_lp") and stats.has("all_masked"), "collector stats keys")
            expect(stats.get("rets").get("se").as_float() > 0.0, "positive standard error")


def test_torch() raises:
    # cosine schedule closed form
    var ac = load_fixture("torch/adam_cosine_golden.json")
    var cosd = ac.get("cosine")
    var lrs = floats(cosd.get("lr_per_epoch_torch"))
    var base = cosd.get("base").as_float()
    var tmax = cosd.get("T_max").as_int()
    expect(len(lrs) == tmax + 1, "cosine: T_max + 1 entries")
    for e in range(len(lrs)):
        var want = base * (1.0 + cos(pi * Float64(e) / Float64(tmax))) / 2.0
        expect(close(lrs[e], want, 1e-15, 1e-9), "cosine lr at epoch " + String(e))
    expect(len(ac.get("adam_trace").get("trace")) == 4, "adam trace has 4 steps")

    # f16 golden: decode column is exact
    var fg = load_fixture("torch/f16_golden.json")
    var f32b = ints(fg.get("f32_bits"))
    var f16b = ints(fg.get("f16_bits"))
    var decb = ints(fg.get("f16_decoded_f32_bits"))
    expect(len(f32b) == 326 and len(f16b) == 326 and len(decb) == 326, "f16_golden: 326 values")
    for i in range(326):
        var h = bitcast[DType.float16](UInt16(f16b[i]))
        var dv = Float32(h)
        if dv == dv:
            expect(Int(bitcast[DType.uint32](dv)) == decb[i], "f16 decode of value " + String(i))
        var x = bitcast[DType.float32](UInt32(f32b[i]))
        if x == x:
            var enc = bitcast[DType.uint16](Float16(x))
            expect(Int(enc) == f16b[i], "f16 encode of value " + String(i))
    expect(fg.get("overflow_count").as_int() == 4, "f16 overflow count")

    var mg = load_fixture("torch/misc_golden.json")
    expect(mg.has("np_default_rng7_permutation10") and mg.has("safetensors") and mg.has("npz"), "misc_golden keys")

    # BC golden: value head frozen
    var bc = load_fixture("torch/bc_golden.json")
    expect(bc.get("H").as_int() == 8 and len(bc.get("steps")) == 3, "bc_golden: H and 3 steps")
    expect(bc.get("steps").at(0).get("grads").get("wv").is_null(), "bc_golden: frozen value head has no gradient")
    expect(len(bc.get("steps").at(0).get("obs_sign")) == 6, "bc_golden: batch of 6")

    # route PPO golden
    var rp = load_fixture("torch/ppo_route_golden.json")
    expect(len(rp.get("steps")) == 3, "ppo_route_golden: 3 steps")
    expect(len(rp.get("clip_rule_microtest")) == 24, "ppo_route_golden: 24 clip-rule cases")
    var gae = rp.get("gae")
    expect(len(gae.get("reward")) == len(gae.get("adv")) and len(gae.get("reward")) == 11, "ppo_route_golden: gae lengths")

    # synth PPO golden
    var sp = load_fixture("torch/synth_ppo_golden.json")
    var cases = sp.get("cases")
    expect(len(cases) == 2, "synth_ppo_golden: 2 architectures")
    for i in range(len(cases)):
        expect(len(cases.at(i).get("steps")) == 3, "synth_ppo_golden: 3 steps")
        expect(len(cases.at(i).get("act_perms")) > 0, "synth_ppo_golden: symmetry tables present")


def test_route() raises:
    # topologies
    var t = load_fixture("topologies.json")
    var tr = t.get("train")
    var ev = t.get("eval")
    expect(len(tr) == 25 and len(ev) == 10, "topologies: 25 train + 10 eval")
    for part in range(2):
        var arr = tr.copy()
        if part == 1:
            arr = ev.copy()
        for i in range(len(arr)):
            var n = arr.at(i).get("n").as_int()
            var edges = arr.at(i).get("edges")
            expect(len(edges) >= n - 1, "connected topology has >= n-1 edges")
            for j in range(len(edges)):
                var e = ints(edges.at(j))
                expect(e[0] != e[1] and e[0] >= 0 and e[1] >= 0 and e[0] < n and e[1] < n, "edge within range")
    expect(tr.at(0).get("name").as_string() == "grid3x3", "first train topology is grid3x3")
    expect(len(tr.at(0).get("edges")) == 12, "grid3x3 has 12 edges")

    # jobs_stats
    var js = load_fixture("route/jobs_stats.json")
    var scales = js.get("scales")
    expect(len(scales) == 2, "jobs_stats: 2 scales")
    for i in range(len(scales)):
        var s = scales.at(i)
        var kept = s.get("jobs_kept").as_int()
        var fc = s.get("family_counts")
        var fk = fc.keys()
        var tot = 0
        for j in range(len(fk)):
            tot += fc.get(fk[j]).as_int()
        expect(tot == kept, "family counts sum to jobs_kept")
        var tc = s.get("topology_counts")
        var tk = tc.keys()
        tot = 0
        for j in range(len(tk)):
            tot += tc.get(tk[j]).as_int()
        expect(tot == kept, "topology counts sum to jobs_kept")
        var feats = s.get("all")
        var names = List[String]()
        names.append("n")
        names.append("ops")
        names.append("twoq_frac")
        names.append("depth")
        names.append("n_over_topology_n")
        for q in range(len(names)):
            var f = feats.get(names[q])
            var xs = floats(f.get("x"))
            var fs = floats(f.get("F"))
            expect(len(xs) == len(fs) and len(xs) > 0, "ECDF knot lengths")
            for z in range(len(xs)):
                expect(fs[z] > 0.0 and fs[z] <= 1.0, "ECDF value in (0, 1]")
                if z > 0:
                    expect(xs[z] > xs[z - 1] and fs[z] >= fs[z - 1], "ECDF knots increase")
        expect(len(s.get("per_family")) == 6, "jobs_stats: 6 families")

    # BC reference
    var bc = load_fixture("route/py_bc_reference.json")
    expect(len(bc.get("runs")) == 3, "py_bc_reference: 3 seeds")
    expect(bc.get("d0").get("sha256").as_string().byte_length() == 64, "d0 sha256 recorded")
    for i in range(3):
        expect(len(bc.get("runs").at(i).get("log")) == 30, "py_bc_reference: 30 epochs")


def test_curve() raises:
    var pc = load_fixture("ppo/py_curve.json")
    var runs = pc.get("runs")
    expect(len(runs) == 2, "py_curve: 2 runs")
    for i in range(len(runs)):
        var r = runs.at(i)
        var it = ints(r.get("promotion_iterations"))
        expect(len(it) == len(r.get("promotion_times_s")), "promotion lists agree")
        expect(len(it) + 1 == r.get("final_difficulty").as_int(), "final difficulty = promotions + 1")
        for j in range(1, len(it)):
            expect(it[j] > it[j - 1], "promotion iterations increase")


def test_helpers() raises:
    # close / diffs
    expect(close(1.0, 1.0 + 1e-9, 0.0, 1e-8), "close rel")
    expect(not close(1.0, 1.1, 1e-3, 1e-3), "close fails")
    expect(not close(Float64(0.0) / Float64(0.0), 1.0, 1.0, 1.0), "NaN is never close")
    var a = List[Float64]()
    a.append(1.0)
    a.append(2.0)
    var b = List[Float64]()
    b.append(1.0)
    b.append(2.5)
    expect(max_abs_diff(a, b) == 0.5, "max_abs_diff")
    assert_list_close(a, a, 0.0, 0.0, "identical lists")

    # statistics
    var z = chi_square_z(10.0, 10)
    expect(abs64(z) < 0.5, "chi-square at its mean has z about 0")
    expect(chi_square_z(40.0, 10) > 4.0, "huge chi-square has z > 4")
    var obs = List[Int]()
    obs.append(25)
    obs.append(24)
    obs.append(26)
    obs.append(25)
    var pr = List[Float64]()
    for _ in range(4):
        pr.append(0.25)
    var c2 = chi_square_gof(obs, pr)
    expect(c2.dof == 3 and c2.stat < 1.0, "chi_square_gof uniform counts")
    var o2 = List[Int]()
    o2.append(100)
    o2.append(1)
    o2.append(0)
    var p2 = List[Float64]()
    p2.append(0.9)
    p2.append(0.05)
    p2.append(0.05)
    var c3 = chi_square_gof(o2, p2)
    expect(c3.dof >= 1, "small expected cells are pooled")
    var h1 = List[Int]()
    h1.append(50)
    h1.append(50)
    h1.append(100)
    var h2 = List[Int]()
    h2.append(52)
    h2.append(48)
    h2.append(99)
    var c4 = chi_square_homogeneity(h1, h2)
    expect(c4.dof == 2 and c4.stat < 1.0, "chi_square_homogeneity similar histograms")
    expect(abs64(two_sample_z(1.0, 0.1, 1.2, 0.1) + 1.4142135623730951) < 1e-9, "two_sample_z")
    var sums = List[Float64]()
    sums.append(4.0)
    sums.append(6.0)
    var cnts = List[Float64]()
    cnts.append(2.0)
    cnts.append(3.0)
    var ms = cluster_mean_se(sums, cnts)
    expect(abs64(ms[0] - 2.0) < 1e-12 and abs64(ms[1]) < 1e-12, "cluster_mean_se with identical cluster means")
    var sorted_v = List[Float64]()
    for i in range(10):
        sorted_v.append(Float64(i))
    expect(abs64(ecdf_at(sorted_v, 4.0) - 0.5) < 1e-12, "ecdf_at")
    var kx = List[Float64]()
    kx.append(4.0)
    kx.append(9.0)
    var kf = List[Float64]()
    kf.append(0.5)
    kf.append(1.0)
    expect(ks_against_knots(sorted_v, kx, kf) < 1e-12, "ks_against_knots identical")
    expect(ks_critical(2000, 2000) > 0.07 and ks_critical(2000, 2000) < 0.08, "ks_critical value")

    # reference GEMM: all four transpose combinations against a hand-computed 2x2x3
    var am = List[Float32]()
    for v in range(6):
        am.append(Float32(v + 1))  # A [2,3] = [[1,2,3],[4,5,6]]
    var bm = List[Float32]()
    for v in range(6):
        bm.append(Float32(v + 1))  # B [3,2] = [[1,2],[3,4],[5,6]]
    var cm = List[Float32](length=4, fill=0.0)
    gemm_ref(2, 2, 3, am.unsafe_ptr().as_unsafe_any_origin(), 3, False, bm.unsafe_ptr().as_unsafe_any_origin(), 2, False, cm.unsafe_ptr().as_unsafe_any_origin(), 2, False)
    expect(cm[0] == 22.0 and cm[1] == 28.0 and cm[2] == 49.0 and cm[3] == 64.0, "gemm_ref nn")
    var bt = List[Float32]()  # B^T [2,3] = [[1,3,5],[2,4,6]]
    bt.append(1.0)
    bt.append(3.0)
    bt.append(5.0)
    bt.append(2.0)
    bt.append(4.0)
    bt.append(6.0)
    var cm2 = List[Float32](length=4, fill=1.0)
    gemm_ref(2, 2, 3, am.unsafe_ptr().as_unsafe_any_origin(), 3, False, bt.unsafe_ptr().as_unsafe_any_origin(), 3, True, cm2.unsafe_ptr().as_unsafe_any_origin(), 2, True)
    expect(cm2[0] == 23.0 and cm2[3] == 65.0, "gemm_ref nt with accumulate")
    var at = List[Float32]()  # A^T [3,2] = [[1,4],[2,5],[3,6]]
    at.append(1.0)
    at.append(4.0)
    at.append(2.0)
    at.append(5.0)
    at.append(3.0)
    at.append(6.0)
    var cm3 = List[Float32](length=4, fill=0.0)
    gemm_ref(2, 2, 3, at.unsafe_ptr().as_unsafe_any_origin(), 2, True, bm.unsafe_ptr().as_unsafe_any_origin(), 2, False, cm3.unsafe_ptr().as_unsafe_any_origin(), 2, False)
    expect(cm3[1] == 28.0 and cm3[2] == 49.0, "gemm_ref tn")
    _ = len(am) + len(bm) + len(cm) + len(bt) + len(cm2) + len(at) + len(cm3)

    # guard regions: untouched passes, a write one past the end is caught
    var buf = guard_alloc(16, 1.0)
    var p = guard_ptr(buf)
    p[unsafe_offset=15] = 3.0
    guard_check(buf, 16, "in-bounds write")
    expect(guard_data(buf, 16)[15] == 3.0, "guard_data")
    p[unsafe_offset=16] = 9.0
    var caught = False
    try:
        guard_check(buf, 16, "one past the end")
    except e:
        caught = String(e).find("high guard") >= 0
    expect(caught, "guard_check detects an overrun")
    var buf2 = guard_alloc(4)
    var pg2 = guard_ptr(buf2)
    pg2[unsafe_offset=-1] = 1.0
    caught = False
    try:
        guard_check(buf2, 4, "one before the start")
    except e:
        caught = String(e).find("low guard") >= 0
    expect(caught, "guard_check detects an underrun")
    _ = len(buf) + len(buf2)

    # deterministic fill
    var r1 = List[Float32](length=8, fill=0.0)
    var r2 = List[Float32](length=8, fill=0.0)
    rand_fill(r1, 7)
    rand_fill(r2, 7)
    expect(max_abs_diff32(r1, r2) == 0.0, "rand_fill is deterministic")
    expect(abs64(Float64(r1[0])) <= 1.0, "rand_fill range")


def main() raises:
    test_sizes_and_parse()
    test_envs()
    test_reset()
    test_pauli_twists()
    test_reward_fma()
    test_ppo()
    test_rollout_stats()
    test_torch()
    test_route()
    test_curve()
    test_helpers()
    print("test_rl_fixtures OK")
