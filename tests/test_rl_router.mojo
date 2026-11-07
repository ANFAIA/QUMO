"""Tests for the router training stack: qumo.rl.{teacher_data, router_net, router_loss, bc}.

Pinned here (MOJO_PORT.md 13.2, `test_rl_router`):
  * teacher IO: a 328-byte record round trip with saturating int16, the BC row filter (`spread > 1e-6` in f32 drops
    A = 0 (NaN spread), A = 1 and ties, keeps rows whose spread exceeds 1e-6), the Q clip at 1e6, QM-E646 cases;
  * RouterNet: the torch init statistics, forward against the independent float64 `RouteModel.forward` of the
    runtime (random weights, hidden sizes that are not multiples of the SIMD width, and the shipped model),
    warm start from a container;
  * `bc_golden.json`: the explicit slot permutation (exact), then three steps of loss, gradients and weights after
    the Adam step with the frozen value head (equal to its init, no optimizer state);
  * a multi-chunk batch against a naive float64 reference (gradients, loss), identical bits for 1, 2 and 4 threads;
  * the validation metrics against a naive reference;
  * `ppo_route_golden.json`: route-PPO loss parts, ratios, raw gradients including the value head, the global clip
    and the Adam update for three steps, and the float64 GAE;
  * planted-optimum behaviour cloning (agreement > 0.9, regret < 0.1) and thread-count independence of `train_bc`;
  * the BC comparison of A6(b) against `route/py_bc_reference.json` when `QUMO_BC_REF_D0` names the reference
    teacher file (skipped otherwise).
"""

from std.math import exp, log, sqrt
from std.memory import bitcast
from std.os import getenv, makedirs
from std.time import perf_counter_ns
from qumo.ai.json import JsonValue, parse_json
from qumo.sim.rng import Rng
from qumo.compiler.policy import ModelFile, load_model_file
from qumo.compiler.ai_route import RouteModel, load_route_model
from qumo.rl.tensor import FPtr
from qumo.rl.optim import Adam, clip_grad_norm, cosine_epoch_lr
from qumo.rl.rlrng import seed_for, TAG_INIT
from qumo.rl.teacher_data import (
    TeacherData,
    REC_BYTES,
    read_teacher_files,
    parse_teacher_bytes,
    encode_teacher_record,
)
from qumo.rl.router_net import RouterNet, RouterCache, SPtr, ROUTER_OBS, ROUTER_SLOTS
from qumo.rl.router_loss import (
    gae_route,
    bc_soft_ce_chunk,
    route_ppo_chunk_loss,
    RoutePPOCfg,
    IPtr,
)
from qumo.rl.bc import (
    BCConfig,
    BCResult,
    BCWorkspace,
    BC_VW,
    BC_CHUNK_ROWS,
    bc_batch_step,
    bc_evaluate,
    permute_slots_row,
    train_bc,
    write_bc_run,
)
from qumo.rl.testutil import (
    load_fixture,
    flat_floats,
    flat_floats32,
    floats,
    ints,
    assert_close,
    assert_list_close,
    assert_list_close32,
    max_abs_diff32,
    read_bytes,
    load_fixture_text,
    guard_alloc,
    guard_ptr,
    guard_data,
    guard_check,
)

comptime OUT_DIR = "build/test_rl_router"


# ------------------------------------------------------------------------------------------------ helpers


def _bits_equal(a: List[Float32], b: List[Float32]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            return False
    return True


def _flat_params(v: JsonValue, h: Int) raises -> List[Float32]:
    """The six route tensors (emb, b0, w1, b1, wv, bv) flattened row-major; a JSON null becomes zeros."""
    var sizes: List[Int] = [128 * h, h, 16 * h, 16, h, 1]
    var names: List[String] = ["emb", "b0", "w1", "b1", "wv", "bv"]
    var out = List[Float32]()
    for t in range(6):
        var x = v.get(names[t])
        if x.is_null():
            for _ in range(sizes[t]):
                out.append(0.0)
        else:
            var f = flat_floats32(x)
            if len(f) != sizes[t]:
                raise Error("tensor " + names[t] + " has " + String(len(f)) + " values, expected " + String(sizes[t]))
            for i in range(len(f)):
                out.append(f[i])
    return out^


def _signs(v: JsonValue) raises -> List[Int8]:
    var f = flat_floats(v)
    var out = List[Int8](capacity=len(f))
    for i in range(len(f)):
        out.append(Int8(1) if f[i] > 0.0 else (Int8(-1) if f[i] < 0.0 else Int8(0)))
    return out^


def _net_with(h: Int, params: List[Float32]) -> RouterNet:
    var net = RouterNet(h, 0)
    net.params = params.copy()
    return net^


def _mutp[T: DType](mut l: List[Scalar[T]]) -> MutPointer[Scalar[T], MutAnyOrigin]:
    return l.unsafe_ptr().as_unsafe_any_origin()


def load_text(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def _write(path: String, buf: List[UInt8]) raises:
    with open(path, "w") as f:
        f.write_bytes(Span(buf))


# ------------------------------------------------------------------------------------------------ teacher data


def _q16(vals: List[Float32]) -> List[Float32]:
    return vals.copy()


def test_record_roundtrip() raises:
    var obs = List[Int]()
    for i in range(128):
        obs.append(i - 64)
    obs[0] = 40000  # saturates to 32767 in the record, 127 in the Int8 storage
    obs[1] = -40000  # -32768 -> -128
    obs[2] = 300
    obs[3] = -3
    var q: List[Float32] = [1.5, 2.5, 3.5, 0.25]
    var buf = List[UInt8]()
    encode_teacher_record(buf, obs, 4, 2, 70000, -5, q)
    if len(buf) != REC_BYTES:
        raise Error("record size " + String(len(buf)))
    # byte-level layout: obs[0] = 32767 little endian, A at 256, act at 258, job at 260 (saturated), tb at 262
    if buf[0] != 0xFF or buf[1] != 0x7F:
        raise Error("obs[0] not saturated to 32767")
    if buf[2] != 0x00 or buf[3] != 0x80:
        raise Error("obs[1] not saturated to -32768")
    if Int(buf[256]) != 4 or Int(buf[258]) != 2:
        raise Error("A / act bytes")
    if buf[260] != 0xFF or buf[261] != 0x7F:
        raise Error("job not saturated to 32767")
    if buf[262] != 0xFB or buf[263] != 0xFF:
        raise Error("tb = -5 bytes")
    var td = TeacherData()
    parse_teacher_bytes(buf, td, "roundtrip", False)
    if td.n != 1 or td.raw != 1:
        raise Error("expected one row")
    if td.obs[0] != 127 or td.obs[1] != -128 or td.obs[2] != 127 or td.obs[3] != -3 or td.obs[64] != 0:
        raise Error("obs decode / saturation")
    if td.obs[10] != -54:
        raise Error("obs[10]")
    if td.A[0] != 4 or td.act[0] != 2 or td.job[0] != 32767 or td.tb[0] != -5:
        raise Error("int16 fields decode")
    for k in range(4):
        if td.q[k] != q[k]:
            raise Error("Q decode slot " + String(k))
    for k in range(4, 16):
        if td.q[k] != Float32(1e6):  # the 1e9 invalid-slot value, clipped
            raise Error("invalid slot not clipped to 1e6")


def _rec(a: Int, q: List[Float32], act: Int = 0, job: Int = 0) -> List[UInt8]:
    var b = List[UInt8]()
    var obs = List[Int]()
    for i in range(128):
        obs.append((i + job) % 3 - 1)
    encode_teacher_record(b, obs, a, act, job, 0, q)
    return b^


def _cat(mut dst: List[UInt8], src: List[UInt8]):
    for i in range(len(src)):
        dst.append(src[i])


def test_filter_rule() raises:
    makedirs(OUT_DIR, exist_ok=True)
    var buf = List[UInt8]()
    # 0: A = 0 (NaN spread) dropped
    _cat(buf, _rec(0, List[Float32](), 0, 0))
    # 1: A = 1 dropped (spread 0)
    _cat(buf, _rec(1, [5.0], 0, 1))
    # 2: A = 3 all equal dropped
    _cat(buf, _rec(3, [2.0, 2.0, 2.0], 0, 2))
    # 3: spread 5e-7 <= 1e-6 dropped
    _cat(buf, _rec(2, [4.0, Float32(4.0) + Float32(5e-7)], 0, 3))
    # 4: spread 2e-6 > 1e-6 kept
    _cat(buf, _rec(2, [4.0, Float32(4.0) + Float32(2e-6)], 0, 4))
    # 5: valid costs above the clip: spread 1e6 kept, both clipped to 1e6
    _cat(buf, _rec(2, [2e6, 3e6], 0, 5))
    # 6: ordinary row, A = 16
    var q16 = List[Float32]()
    for k in range(16):
        q16.append(Float32(k) * 0.5)
    _cat(buf, _rec(16, q16, 3, 6))
    # 7: a spread hidden in an INVALID slot only (A = 2, equal valid costs, slot 5 differs) is dropped
    var qh: List[Float32] = [3.0, 3.0, 9.0, 9.0, 9.0, 1.0]
    _cat(buf, _rec(2, qh, 0, 7))
    var f1 = OUT_DIR + "/a.bin"
    var f2 = OUT_DIR + "/b.bin"
    _write(f1, buf)
    var buf2 = _rec(4, [1.0, 7.0, 3.0, 5.0], 1, 8)
    _write(f2, buf2)
    var paths = List[String]()
    paths.append(f1)
    paths.append(f2)
    var td = read_teacher_files(paths)
    if td.raw != 9:
        raise Error("raw records " + String(td.raw))
    # kept: rows 4, 5, 6 of the first file and the single row of the second file, in file order
    if td.n != 4:
        raise Error("kept rows " + String(td.n))
    if Int(td.job[0]) != 4 or Int(td.job[1]) != 5 or Int(td.job[2]) != 6 or Int(td.job[3]) != 8:
        raise Error("kept rows or order wrong")
    if td.q[1 * 16 + 0] != Float32(1e6) or td.q[1 * 16 + 1] != Float32(1e6):
        raise Error("Q not clipped at 1e6")
    if td.q[3 * 16 + 1] != 7.0:
        raise Error("second file order")
    # QM-E646 cases
    var bad = List[UInt8]()
    _cat(bad, _rec(2, [1.0, 2.0], 0, 0))
    bad.append(0)
    var fbad = OUT_DIR + "/bad_size.bin"
    _write(fbad, bad)
    var p1 = List[String]()
    p1.append(fbad)
    var raised = False
    try:
        _ = read_teacher_files(p1)
    except e:
        raised = String(e).startswith("QM-E646")
    if not raised:
        raise Error("size not a multiple of 328 must raise QM-E646")
    var nan_q = List[Float32]()
    nan_q.append(1.0)
    nan_q.append(Float32.MAX_FINITE * 2.0 - Float32.MAX_FINITE * 2.0)  # inf - inf = NaN
    var nan_rec = _rec(2, nan_q, 0, 0)
    var fnan = OUT_DIR + "/nan.bin"
    _write(fnan, nan_rec)
    var p2 = List[String]()
    p2.append(fnan)
    raised = False
    try:
        _ = read_teacher_files(p2)
    except e:
        raised = String(e).startswith("QM-E646")
    if not raised:
        raise Error("NaN cost in a valid slot must raise QM-E646")
    var a17 = _rec(17, q16, 0, 0)
    var f17 = OUT_DIR + "/a17.bin"
    _write(f17, a17)
    var p3 = List[String]()
    p3.append(f17)
    raised = False
    try:
        _ = read_teacher_files(p3)
    except e:
        raised = String(e).startswith("QM-E646")
    if not raised:
        raise Error("A = 17 must raise QM-E646")
    var missing = List[String]()
    missing.append(OUT_DIR + "/does_not_exist.bin")
    raised = False
    try:
        _ = read_teacher_files(missing)
    except e:
        raised = String(e).startswith("QM-E646")
    if not raised:
        raise Error("a missing file must raise QM-E646")
    # too few rows for the validation split
    var cfg = BCConfig()
    cfg.epochs = 1
    raised = False
    try:
        _ = train_bc(cfg, td, None)
    except e:
        raised = String(e).startswith("QM-E646")
    if not raised:
        raise Error("train_bc with 4 rows must raise QM-E646")


# ------------------------------------------------------------------------------------------------ network


def test_init_statistics() raises:
    var h = 64
    var a = RouterNet(h, 5)
    var b = RouterNet(h, 5)
    var c = RouterNet(h, 6)
    if not _bits_equal(a.params, b.params):
        raise Error("same seed must give the same init")
    if _bits_equal(a.params, c.params):
        raise Error("different seeds must differ")
    if a.param_count() != 128 * h + h + 16 * h + 16 + h + 1 or len(a.params) != a.param_count():
        raise Error("param count")
    # embeddings: N(0,1) / sqrt(128)
    var s = 0.0
    var s2 = 0.0
    var n = 128 * h
    for i in range(n):
        var x = Float64(a.params[i])
        s += x
        s2 += x * x
    var mean = s / Float64(n)
    var std = sqrt(s2 / Float64(n) - mean * mean)
    assert_close(mean, 0.0, 0.005, 0.0, "emb mean")
    assert_close(std, 1.0 / sqrt(128.0), 0.003, 0.0, "emb std")
    for i in range(h):
        if a.params[a.off_b0() + i] != 0.0:
            raise Error("b0 must be zero")
    var bound = Float32(1.0 / sqrt(Float64(h)))
    var lo: Float32 = 1.0
    var hi: Float32 = -1.0
    for i in range(a.off_w1(), a.param_count()):
        if i >= a.off_b0() and i < a.off_w1():
            continue
        var x = a.params[i]
        if x > bound or x < -bound:
            raise Error("head weight outside +-1/sqrt(H)")
        if x < lo:
            lo = x
        if x > hi:
            hi = x
    if hi < 0.8 * bound or lo > -0.8 * bound:
        raise Error("head init does not cover its interval")


def _random_net(h: Int, seed: UInt64) -> RouterNet:
    var net = RouterNet(h, seed)
    var r = Rng(seed + 99)
    for i in range(len(net.params)):
        net.params[i] = Float32(r.uniform() - 0.5)
    return net^


def _route_model_of(net: RouterNet) -> RouteModel:
    var m = RouteModel()
    m.ok = True
    m.hidden = net.hidden
    for i in range(128 * net.hidden):
        m.emb.append(net.params[net.off_emb() + i])
    for i in range(net.hidden):
        m.b0.append(net.params[net.off_b0() + i])
    for i in range(16 * net.hidden):
        m.w1.append(net.params[net.off_w1() + i])
    for i in range(16):
        m.b1.append(net.params[net.off_b1() + i])
    return m^


def test_forward_vs_runtime() raises:
    # the runtime RouteModel.forward is an independent float64 implementation of the same network
    for hi in range(3):
        var h = 64 if hi == 0 else (24 if hi == 1 else 5)
        var net = _random_net(h, UInt64(10 + hi))
        var model = _route_model_of(net)
        var rows = 37
        var rng = Rng(77)
        var sign = List[Int8](length=rows * 128, fill=0)
        var obs_all = List[List[Int]]()
        for r in range(rows):
            var obs = List[Int]()
            for i in range(128):
                var u = rng.below(5)
                var o = 0 if u < 2 else (u - 3)  # mostly 0, some -1/1
                obs.append(o * (1 + rng.below(3)))
                sign[r * 128 + i] = Int8(1 if o > 0 else (-1 if o < 0 else 0))
            obs_all.append(obs^)
        var cc = RouterCache(64, h)
        net.forward_chunk(_mutp(sign), rows, cc, True)
        var hid = List[Float64](length=h, fill=0.0)
        var logits = List[Float64](length=16, fill=0.0)
        for r in range(rows):
            model.forward(obs_all[r], hid, logits)
            for k in range(16):
                assert_close(Float64(cc.z[r * 16 + k]), logits[k], 2e-5, 2e-5, "logit h=" + String(h) + " row " + String(r))
            var v = Float64(net.params[net.off_bv()])
            for j in range(h):
                if hid[j] > 0.0:
                    v += Float64(net.params[net.off_wv() + j]) * hid[j]
            assert_close(Float64(cc.v[r]), v, 2e-5, 2e-5, "value h=" + String(h))


def test_shipped_model_forward_and_warm_start() raises:
    var dir = "models/ai_transpiler/route/route_mlp_h64_v1"
    var mf = load_model_file(dir)
    var net = RouterNet.from_model(mf)
    if net.hidden != 64:
        raise Error("hidden size of the shipped router")
    var emb = mf.tensor("embeddings.weight")
    var wv = mf.tensor("value.0.weight")
    for i in range(len(emb)):
        if net.params[i] != emb[i]:
            raise Error("embedding copy")
    for i in range(64):
        if net.params[net.off_wv() + i] != wv[i]:
            raise Error("value head copy")
    var rm = load_route_model(dir)
    var rows = 20
    var rng = Rng(5)
    var sign = List[Int8](length=rows * 128, fill=0)
    var obs_all = List[List[Int]]()
    for r in range(rows):
        var obs = List[Int]()
        for i in range(128):
            var o = rng.below(7) - 3
            if rng.below(3) != 0:
                o = 0
            obs.append(o)
            sign[r * 128 + i] = Int8(1 if o > 0 else (-1 if o < 0 else 0))
        obs_all.append(obs^)
    var cc = RouterCache(32, 64)
    net.forward_chunk(_mutp(sign), rows, cc)
    var hid = List[Float64](length=64, fill=0.0)
    var logits = List[Float64](length=16, fill=0.0)
    for r in range(rows):
        rm.forward(obs_all[r], hid, logits)
        for k in range(16):
            assert_close(Float64(cc.z[r * 16 + k]), logits[k], 1e-4, 1e-4, "shipped router logit")
    # the init must be rejected when it does not match: a synth model is not a router
    var synth = load_model_file("models/ai_transpiler/synth/perm_grid3x3")
    var raised = False
    try:
        _ = RouterNet.from_model(synth)
    except e:
        raised = String(e).startswith("QM-E647")
    if not raised:
        raise Error("a synthesis model must be rejected with QM-E647")


# ------------------------------------------------------------------------------------------------ BC golden


def test_bc_golden() raises:
    var g = load_fixture("torch/bc_golden.json")
    var h = g.get("H").as_int()
    var tau = Float32(g.get("tau").as_float())
    var steps = g.get("steps")
    var base = _flat_params(g.get("init"), h)
    var ref_final = List[Float32]()
    for threads in range(1, 5):
        if threads == 3:
            continue
        var net = _net_with(h, base)
        var opt = Adam(net.param_count(), 0.002)
        opt.set_frozen(net.off_wv(), h + 1)
        var ws = BCWorkspace(h, net.param_count(), 6)
        var value_init = List[Float32]()
        for i in range(net.off_wv(), net.param_count()):
            value_init.append(net.params[i])
        for s in range(len(steps)):
            var st = steps.at(s)
            var rows = len(st.get("A"))
            var obs_sign = _signs(st.get("obs_sign"))
            var q_raw = flat_floats32(st.get("Q_raw"))
            var a_cnt = ints(st.get("A"))
            var slot_perm = flat_floats(st.get("slot_perm"))
            var exp_obs = _signs(st.get("permuted_obs"))
            var exp_q = flat_floats32(st.get("permuted_Q"))
            var exp_valid = flat_floats(st.get("valid"))
            # the explicit slot permutation (exact), the Q clip first (Q_raw has 1e9 in the invalid slots)
            var sign = List[Int8](length=rows * 128, fill=0)
            var qq = List[Float32](length=rows * 16, fill=0.0)
            var valid = List[Int8](length=rows * 16, fill=0)
            var perm = List[Int8](length=16, fill=0)
            for r in range(rows):
                for j in range(16):
                    perm[j] = Int8(Int(slot_perm[r * 16 + j]))
                var qrow = List[Float32](length=16, fill=0.0)
                for k in range(16):
                    var qv = q_raw[r * 16 + k]
                    qrow[k] = qv if qv < 1e6 else Float32(1e6)
                var osrc = List[Int8](length=128, fill=0)
                for i in range(128):
                    osrc[i] = obs_sign[r * 128 + i]
                var sdst = List[Int8](length=128, fill=0)
                var qdst = List[Float32](length=16, fill=0.0)
                var vdst = List[Int8](length=16, fill=0)
                permute_slots_row(_mutp(osrc), _mutp(qrow), a_cnt[r], _mutp(perm), _mutp(sdst), _mutp(qdst), _mutp(vdst))
                for i in range(128):
                    sign[r * 128 + i] = sdst[i]
                    if sdst[i] != exp_obs[r * 128 + i]:
                        raise Error("permuted obs mismatch row " + String(r) + " col " + String(i))
                for k in range(16):
                    qq[r * 16 + k] = qdst[k]
                    valid[r * 16 + k] = vdst[k]
                    if Float64(vdst[k]) != exp_valid[r * 16 + k]:
                        raise Error("valid mask mismatch")
                    if vdst[k] != 0 and qdst[k] != exp_q[r * 16 + k]:
                        raise Error("permuted Q mismatch row " + String(r) + " slot " + String(k))
            var loss = bc_batch_step(net, opt, ws, sign, qq, valid, rows, tau, st.get("lr").as_float(), threads)
            assert_close(loss, st.get("loss").as_float(), 2e-5, 2e-5, "bc loss step " + String(s))
            assert_list_close32(ws.grad, _flat_params(st.get("grads"), h), 3e-6, 2e-4, "bc grads step " + String(s))
            assert_list_close32(net.params, _flat_params(st.get("weights_after"), h), 3e-5, 0.0, "bc weights step " + String(s))
        # the value head is frozen: equal to its init, no optimizer state
        for i in range(len(value_init)):
            if net.params[net.off_wv() + i] != value_init[i]:
                raise Error("value head moved in BC")
            if opt.m[net.off_wv() + i] != 0.0 or opt.v[net.off_wv() + i] != 0.0:
                raise Error("value head acquired optimizer state")
        if threads == 1:
            ref_final = net.params.copy()
        elif not _bits_equal(net.params, ref_final):
            raise Error("BC weights depend on the thread count")


# ------------------------------------------------------------------------------------------------ naive f64 reference


def _naive_bc(
    params: List[Float32], h: Int, sign: List[Int8], q: List[Float32], valid: List[Int8], rows: Int, tau: Float64
) -> Tuple[Float64, List[Float64]]:
    """Plain float64 loops: loss (mean) and gradient of the BC soft cross entropy."""
    var eoff = 0
    var b0off = 128 * h
    var w1off = b0off + h
    var b1off = w1off + 16 * h
    var grad = List[Float64](length=len(params), fill=0.0)
    var total = 0.0
    var B = Float64(rows)
    for r in range(rows):
        var pre = List[Float64](length=h, fill=0.0)
        for j in range(h):
            pre[j] = Float64(params[b0off + j])
        for i in range(128):
            var s = Float64(sign[r * 128 + i])
            if s != 0.0:
                for j in range(h):
                    pre[j] += s * Float64(params[eoff + i * h + j])
        var hid = List[Float64](length=h, fill=0.0)
        for j in range(h):
            hid[j] = pre[j] if pre[j] > 0.0 else 0.0
        var z = List[Float64](length=16, fill=0.0)
        var mx = -1e300
        var qmin = 1e300
        for k in range(16):
            if valid[r * 16 + k] == 0:
                continue
            var acc = Float64(params[b1off + k])
            for j in range(h):
                acc += Float64(params[w1off + k * h + j]) * hid[j]
            z[k] = acc
            mx = max(mx, acc)
            qmin = min(qmin, Float64(q[r * 16 + k]))
        var se = 0.0
        var tse = 0.0
        for k in range(16):
            if valid[r * 16 + k] != 0:
                se += exp(z[k] - mx)
                tse += exp(-(Float64(q[r * 16 + k]) - qmin) / tau)
        var dz = List[Float64](length=16, fill=0.0)
        for k in range(16):
            if valid[r * 16 + k] != 0:
                var lp = z[k] - mx - log(se)
                var t = exp(-(Float64(q[r * 16 + k]) - qmin) / tau) / tse
                total -= t * lp
                dz[k] = (exp(lp) - t) / B
        var dh = List[Float64](length=h, fill=0.0)
        for k in range(16):
            grad[b1off + k] += dz[k]
            for j in range(h):
                grad[w1off + k * h + j] += dz[k] * hid[j]
                dh[j] += dz[k] * Float64(params[w1off + k * h + j])
        for j in range(h):
            var dp = dh[j] if pre[j] > 0.0 else 0.0
            grad[b0off + j] += dp
            for i in range(128):
                var s = Float64(sign[r * 128 + i])
                if s != 0.0:
                    grad[eoff + i * h + j] += s * dp
    return (total / B, grad^)


def _random_batch(rows: Int, seed: UInt64, mut sign: List[Int8], mut q: List[Float32], mut valid: List[Int8]):
    var r = Rng(seed)
    sign = List[Int8](length=rows * 128, fill=0)
    q = List[Float32](length=rows * 16, fill=0.0)
    valid = List[Int8](length=rows * 16, fill=0)
    for i in range(rows):
        var a = 2 + r.below(15)
        for k in range(16):
            for d in range(8):
                var u = r.below(4)
                sign[i * 128 + 8 * k + d] = Int8(0 if u == 0 else (1 if u < 3 else -1))
            q[i * 16 + k] = Float32(r.uniform() * 6.0)
        # an arbitrary valid subset of the slots (as after a slot permutation), at least two slots
        var cnt = 0
        for k in range(16):
            var on = r.below(16) < a
            valid[i * 16 + k] = 1 if on else 0
            if on:
                cnt += 1
        if cnt < 2:
            valid[i * 16 + 0] = 1
            valid[i * 16 + 1] = 1


def test_multichunk_vs_naive() raises:
    var h = 24  # not a multiple of the SIMD width: exercises the tails
    var rows = 5 * BC_CHUNK_ROWS + 37  # six chunks: virtual workers 0 and 1 take two
    var sign = List[Int8]()
    var q = List[Float32]()
    var valid = List[Int8]()
    _random_batch(rows, 4242, sign, q, valid)
    var base = _random_net(h, 3)
    var ref_params = base.params.copy()
    var naive = _naive_bc(ref_params, h, sign, q, valid, rows, 0.5)
    var first_grad = List[Float32]()
    for threads in range(1, 5):
        var net = _net_with(h, base.params)
        var opt = Adam(net.param_count(), 1e-3)
        var ws = BCWorkspace(h, net.param_count(), rows)
        var loss = bc_batch_step(net, opt, ws, sign, q, valid, rows, 0.5, 1e-3, threads)
        assert_close(loss, naive[0], 1e-5, 1e-5, "multi-chunk loss")
        var g = naive[1].copy()
        for i in range(len(g)):
            assert_close(Float64(ws.grad[i]), g[i], 2e-6, 2e-4, "multi-chunk grad " + String(i))
        if threads == 1:
            first_grad = ws.grad.copy()
        elif not _bits_equal(ws.grad, first_grad):
            raise Error("reduced gradient depends on the thread count")
    # the value head receives no gradient in BC
    for i in range(base.off_wv(), base.param_count()):
        if first_grad[i] != 0.0:
            raise Error("BC wrote a value-head gradient")


def test_evaluate_vs_naive() raises:
    var h = 16
    var n = 300
    var net = _random_net(h, 9)
    var sign = List[Int8](length=n * 128, fill=0)
    var q = List[Float32](length=n * 16, fill=0.0)
    var a = List[Int32](length=n, fill=0)
    var r = Rng(31)
    for i in range(n):
        a[i] = Int32(2 + r.below(15))
        for j in range(128):
            var u = r.below(4)
            sign[i * 128 + j] = Int8(0 if u == 0 else (1 if u < 3 else -1))
        for k in range(16):
            q[i * 16 + k] = Float32(Int(r.uniform() * 8.0)) * 0.5  # ties in the costs
    var ws = BCWorkspace(h, net.param_count(), n)
    var res = bc_evaluate(net, ws, _mutp(sign), _mutp(q), _mutp(a), n, 4)
    # naive reference with float64 logits (random weights: logit ties are practically impossible)
    var agree = 0.0
    var regret = 0.0
    var model = _route_model_of(net)
    var hid = List[Float64](length=h, fill=0.0)
    var logits = List[Float64](length=16, fill=0.0)
    for i in range(n):
        var obs = List[Int]()
        for j in range(128):
            obs.append(Int(sign[i * 128 + j]))
        model.forward(obs, hid, logits)
        var pick = 0
        var qmin = q[i * 16]
        for k in range(1, Int(a[i])):
            if logits[k] > logits[pick]:
                pick = k
            if q[i * 16 + k] < qmin:
                qmin = q[i * 16 + k]
        var qp = q[i * 16 + pick]
        if qp <= qmin + Float32(1e-6):
            agree += 1.0
        regret += Float64(qp - qmin)
    assert_close(res[0], agree / Float64(n), 0.0, 1e-9, "validation agreement")
    assert_close(res[1], regret / Float64(n), 1e-9, 1e-9, "validation regret")


# ------------------------------------------------------------------------------------------------ route PPO


def test_route_ppo_golden() raises:
    var g = load_fixture("torch/ppo_route_golden.json")
    var h = g.get("H").as_int()
    var lr = g.get("lr").as_float()
    var max_norm = g.get("max_norm").as_float()
    var cfg = RoutePPOCfg()
    cfg.clip = Float32(g.get("clip").as_float())
    cfg.ent_coef = Float32(g.get("ent").as_float())
    var steps = g.get("steps")
    var net = _net_with(h, _flat_params(g.get("init"), h))
    var opt = Adam(net.param_count(), lr)
    var ranges = net.tensor_ranges()
    for s in range(len(steps)):
        var st = steps.at(s)
        var rows = len(st.get("A"))
        var sign = _signs(st.get("obs_sign"))
        var a_cnt = ints(st.get("A"))
        var valid = List[Int8](length=rows * 16, fill=0)
        for r in range(rows):
            for k in range(a_cnt[r]):
                valid[r * 16 + k] = 1
        var acts = List[Int32](length=rows, fill=0)
        var act_i = ints(st.get("action"))
        for r in range(rows):
            acts[r] = Int32(act_i[r])
        var logp_old = flat_floats32(st.get("logp_old"))
        var adv = flat_floats32(st.get("adv"))
        var ret = flat_floats32(st.get("ret"))
        var cc = RouterCache(rows, h)
        net.forward_chunk(_mutp(sign), rows, cc, True)
        var ratio = List[Float32](length=rows, fill=0.0)
        var inv_b = Float32(1.0) / Float32(rows)
        var parts = route_ppo_chunk_loss(
            _mutp(cc.z),
            _mutp(cc.v),
            _mutp(valid),
            rows,
            16,
            _mutp(acts),
            _mutp(logp_old),
            _mutp(adv),
            _mutp(ret),
            cfg,
            inv_b,
            _mutp(cc.dz),
            _mutp(cc.dv),
            _mutp(ratio),
        )
        var B = Float64(rows)
        var pl = parts.policy / B
        var vl = parts.value / B
        var en = parts.entropy / B
        var lt = st.get("loss")
        assert_close(pl, lt.get("policy").as_float(), 2e-5, 2e-5, "ppo policy loss step " + String(s))
        assert_close(vl, lt.get("value").as_float(), 2e-5, 2e-5, "ppo value loss step " + String(s))
        assert_close(en, lt.get("entropy").as_float(), 2e-5, 2e-5, "ppo entropy step " + String(s))
        assert_close(
            pl + Float64(cfg.vf_coef) * vl - Float64(cfg.ent_coef) * en,
            lt.get("total").as_float(),
            2e-5,
            2e-5,
            "ppo total loss step " + String(s),
        )
        assert_list_close32(ratio, flat_floats32(st.get("ratio")), 2e-6, 2e-5, "ratios step " + String(s))
        var grad = List[Float32](length=net.param_count(), fill=0.0)
        net.backward_chunk(_mutp(sign), rows, cc, _mutp(grad), True)
        assert_list_close32(grad, _flat_params(st.get("grads_raw"), h), 3e-6, 2e-4, "raw grads step " + String(s))
        var norm = clip_grad_norm(grad, ranges, max_norm)
        assert_close(norm, st.get("grad_norm_before_clip").as_float(), 0.0, 1e-4, "grad norm step " + String(s))
        opt.step(net.params, grad, lr)
        assert_list_close32(net.params, _flat_params(st.get("weights_after"), h), 3e-5, 0.0, "ppo weights step " + String(s))
    # clip rule micro test: dL/dratio per (ratio, adv) pair through a single-row call with one valid slot pair
    var rules = g.get("clip_rule_microtest")
    for i in range(len(rules)):
        var m = rules.at(i)
        var ratio_t = m.get("ratio").as_float()
        var adv_t = m.get("adv").as_float()
        if m.get("on_clip_boundary_informational_only").as_bool():
            continue
        # one row, two valid slots with equal logits: p = 0.5, so ratio = exp(lp - logp_old) with lp = ln 0.5
        var logits = List[Float32](length=16, fill=0.0)
        var valid1 = List[Int8](length=16, fill=0)
        valid1[0] = 1
        valid1[1] = 1
        var acts1: List[Int32] = [0]
        var lp_old: List[Float32] = [Float32(log(0.5) - log(ratio_t))]
        var adv1: List[Float32] = [Float32(adv_t)]
        var ret1: List[Float32] = [0.0]
        var val1: List[Float32] = [0.0]
        var dz = List[Float32](length=16, fill=0.0)
        var dv = List[Float32](length=1, fill=0.0)
        var rat = List[Float32](length=1, fill=0.0)
        var c1 = RoutePPOCfg()
        c1.clip = Float32(g.get("clip").as_float())
        c1.ent_coef = 0.0
        _ = route_ppo_chunk_loss(
            _mutp(logits),
            _mutp(val1),
            _mutp(valid1),
            1,
            16,
            _mutp(acts1),
            _mutp(lp_old),
            _mutp(adv1),
            _mutp(ret1),
            c1,
            1.0,
            _mutp(dz),
            _mutp(dv),
            _mutp(rat),
        )
        # d nlp / dz_0 = 1 - p = 0.5, so dz_0 = dratio * ratio * 0.5 with dratio = -adv when live, else 0
        var dnlp = Float64(dz[0]) / 0.5
        var got_dratio = dnlp / ratio_t
        var want = m.get("rule_dratio").as_float()  # d loss / d ratio with B = 1 (equal to torch_dratio off the boundary)
        assert_close(got_dratio, want, 1e-5, 1e-4, "clip rule ratio " + String(ratio_t) + " adv " + String(adv_t))


def test_route_gae_golden() raises:
    var g = load_fixture("torch/ppo_route_golden.json").get("gae")
    var gamma = g.get("gamma").as_float()
    var lam = g.get("lam").as_float()
    var rew = floats(g.get("reward"))
    var val = floats(g.get("value"))
    var dn = List[Bool]()
    var dj = g.get("done")
    for i in range(len(dj)):
        dn.append(dj.at(i).as_bool())
    var adv = List[Float64]()
    var ret = List[Float64]()
    gae_route(Span(rew), Span(val), Span(dn), gamma, lam, adv, ret)
    assert_list_close(adv, floats(g.get("adv")), 1e-9, 1e-9, "route GAE advantages")
    assert_list_close(ret, floats(g.get("ret")), 1e-9, 1e-9, "route GAE returns")
    # standardised advantages: population std + 1e-8 (the loss input), the returns keep the raw advantage
    var n = len(adv)
    var m = 0.0
    for i in range(n):
        m += adv[i]
    m /= Float64(n)
    var v = 0.0
    for i in range(n):
        v += (adv[i] - m) * (adv[i] - m)
    var sd = sqrt(v / Float64(n))
    var norm = List[Float64]()
    for i in range(n):
        norm.append((adv[i] - m) / (sd + 1e-8))
    assert_list_close(norm, floats(g.get("adv_normalized")), 1e-9, 1e-9, "standardised advantages")


# ------------------------------------------------------------------------------------------------ training


def _planted_file(rows: Int, seed: UInt64) -> List[UInt8]:
    """Teacher rows whose optimum is planted: slot `best` carries a +1 marker in its feature 0, every other valid slot
    a -1, the rest of the features are noise; the best slot costs 0, the others between 4 and 8."""
    var r = Rng(seed)
    var buf = List[UInt8]()
    for row in range(rows):
        var a = 2 + r.below(15)
        var best = r.below(a)
        var obs = List[Int](length=128, fill=0)
        var q = List[Float32]()
        for k in range(a):
            obs[8 * k] = (1 + r.below(3)) if k == best else -(1 + r.below(3))
            for d in range(1, 8):
                var u = r.below(3)
                obs[8 * k + d] = (u - 1) * (1 + r.below(2))
            q.append(0.0 if k == best else Float32(4.0 + 4.0 * r.uniform()))
        encode_teacher_record(buf, obs, a, best, row, best, q)
    return buf^


def test_planted_optimum() raises:
    makedirs(OUT_DIR, exist_ok=True)
    var path = OUT_DIR + "/planted.bin"
    _write(path, _planted_file(12000, 17))
    var paths = List[String]()
    paths.append(path)
    var td = read_teacher_files(paths)
    if td.n != 12000:
        raise Error("planted file rows " + String(td.n))
    var cfg = BCConfig()
    cfg.hidden = 32
    cfg.epochs = 15
    cfg.lr = 5e-3
    cfg.threads = 4
    cfg.seed = 3
    var t0 = perf_counter_ns()
    var out = train_bc(cfg, td, None)
    var secs = Float64(perf_counter_ns() - t0) / 1e9
    var res = out[1].copy()
    if len(res.log) != 15 or res.samples != 12000 - 1000 or res.val != 1000:
        raise Error("log length / sample counts: " + String(len(res.log)) + " " + String(res.samples) + " " + String(res.val))
    var last = res.log[len(res.log) - 1].copy()
    print("planted optimum: val_agree", last.val_agree, "val_regret", last.val_regret, "loss", last.loss, "train s", secs)
    if not (last.val_agree > 0.9):
        raise Error("planted optimum: agreement " + String(last.val_agree) + " must exceed 0.9")
    if not (last.val_regret < 0.1):
        raise Error("planted optimum: regret " + String(last.val_regret) + " must be below 0.1")
    # the loss goes down over the epochs
    if not (res.log[len(res.log) - 1].loss < res.log[0].loss):
        raise Error("loss did not decrease")
    # the value head stayed at its init
    var init_net = RouterNet(cfg.hidden, cfg.seed)
    for i in range(init_net.off_wv(), init_net.param_count()):
        if out[0].params[i] != init_net.params[i]:
            raise Error("BC moved the value head")


def test_train_thread_independence() raises:
    var path = OUT_DIR + "/planted_small.bin"
    makedirs(OUT_DIR, exist_ok=True)
    _write(path, _planted_file(2600, 5))
    var paths = List[String]()
    paths.append(path)
    var td = read_teacher_files(paths)
    var cfg = BCConfig()
    cfg.hidden = 16
    cfg.epochs = 3
    cfg.seed = 8
    cfg.threads = 1
    var o1 = train_bc(cfg, td, None)
    cfg.threads = 4
    var o4 = train_bc(cfg, td, None)
    cfg.threads = 2
    var o2 = train_bc(cfg, td, None)
    if not _bits_equal(o1[0].params, o4[0].params) or not _bits_equal(o1[0].params, o2[0].params):
        raise Error("train_bc weights depend on the thread count")
    var l1 = o1[1].copy()
    var l4 = o4[1].copy()
    for e in range(len(l1.log)):
        if l1.log[e].loss != l4.log[e].loss or l1.log[e].val_agree != l4.log[e].val_agree:
            raise Error("train_bc log depends on the thread count")
    # a different seed gives a different model; the same seed reproduces it
    cfg.threads = 4
    cfg.seed = 9
    var o9 = train_bc(cfg, td, None)
    if _bits_equal(o1[0].params, o9[0].params):
        raise Error("a different seed must change the result")
    cfg.seed = 8
    var again = train_bc(cfg, td, None)
    if not _bits_equal(o1[0].params, again[0].params):
        raise Error("rerun with the same seed must be bit identical")
    # warm start from a model: continuing from the trained weights with the same config keeps improving or at least runs
    # (the init path itself is exercised with the shipped container in test_shipped_model_forward_and_warm_start)


def test_run_dir_and_warm_start() raises:
    makedirs(OUT_DIR, exist_ok=True)
    var path = OUT_DIR + "/planted_run.bin"
    _write(path, _planted_file(3000, 11))
    var paths = List[String]()
    paths.append(path)
    var td = read_teacher_files(paths)
    var cfg = BCConfig()
    cfg.hidden = 16
    cfg.epochs = 4
    cfg.lr = 5e-3
    cfg.seed = 21
    var out = train_bc(cfg, td, None)
    var res = out[1].copy()
    var run = OUT_DIR + "/run_a"
    for _ in range(2):  # the second call replaces the first atomically
        write_bc_run(run, out[0], res, cfg, paths, "")
    var mf = load_model_file(run)
    if mf.format != "qumo.rlroute.v0" or mf.kind != "routing" or mf.id != "run_a":
        raise Error("run manifest identity: " + mf.format + " " + mf.kind + " " + mf.id)
    var want = out[0].to_tensors()
    if len(mf.names) != 6:
        raise Error("run directory has " + String(len(mf.names)) + " tensors")
    for t in range(6):
        if mf.names[t] != want[t].name:
            raise Error("tensor order / name: " + mf.names[t])
        if not _bits_equal(mf.tensors[t], want[t].data):
            raise Error("tensor " + mf.names[t] + " is not the f32 of the trained net")
    var rm = load_route_model(run)  # the runtime loader accepts the run directory as a router model
    if rm.hidden != 16:
        raise Error("route model hidden")
    var man = parse_json(mf.manifest)
    var tr = man.get("training")
    if tr.get_int("vw", 0) != BC_VW or tr.get_int("chunk_rows", 0) != BC_CHUNK_ROWS or tr.get_int("seed", -1) != 21:
        raise Error("manifest training.vw / chunk_rows / seed")
    if tr.get_string("engine", "") != "qumo.rl (Mojo)":
        raise Error("manifest training.engine")
    var lg = parse_json(load_text(run + "/train_log.json"))
    if lg.get_int("samples", 0) != res.samples or lg.get_int("val", 0) != res.val or lg.get_int("epochs_run", 0) != 4:
        raise Error("train_log.json counters")
    if len(lg.get("log")) != 4 or lg.get("log").at(3).get_float("loss", -1.0) != res.log[3].loss:
        raise Error("train_log.json epoch log")
    if lg.get("data").at(0).as_string() != "planted_run.bin":
        raise Error("train_log.json data basenames")
    # warm start: from_model reproduces the net exactly, and a warm-started run starts below a fresh run
    var back = RouterNet.from_model(mf)
    if not _bits_equal(back.params, out[0].params):
        raise Error("RouterNet.from_model(run) is not the trained net")
    var cold_cfg = BCConfig()
    cold_cfg.hidden = 16
    cold_cfg.epochs = 1
    cold_cfg.lr = 5e-3
    cold_cfg.seed = 22
    var cold = train_bc(cold_cfg, td, None)
    var init = Optional[ModelFile](mf^)
    var warm = train_bc(cold_cfg, td, init)
    if not (warm[1].log[0].loss < cold[1].log[0].loss):
        raise Error("warm-started epoch 0 loss " + String(warm[1].log[0].loss) + " is not below the fresh " + String(cold[1].log[0].loss))
    # a mismatching --hidden is rejected
    var bad_cfg = cold_cfg.copy()
    bad_cfg.hidden = 8
    var init2 = Optional[ModelFile](load_model_file(run))
    var raised = False
    try:
        _ = train_bc(bad_cfg, td, init2)
    except e:
        raised = String(e).startswith("QM-E647")
    if not raised:
        raise Error("init with a different hidden size must raise QM-E647")


def test_guard_regions() raises:
    # raw-pointer kernels write only inside their buffers: gradient and dz outputs sit between sentinel guards
    var h = 24
    var rows = 50
    var sign = List[Int8]()
    var q = List[Float32]()
    var valid = List[Int8]()
    _random_batch(rows, 77, sign, q, valid)
    var net = _random_net(h, 4)
    var cc = RouterCache(64, h)
    net.forward_chunk(_mutp(sign), rows, cc)
    var n = net.param_count()
    var g1 = guard_alloc(n)
    var dzg = guard_alloc(rows * 16)
    _ = bc_soft_ce_chunk(_mutp(cc.z), _mutp(q), _mutp(valid), rows, 16, 0.5, 0.02, guard_ptr(dzg))
    guard_check(dzg, rows * 16, "bc_soft_ce_chunk dz")
    var dz = guard_data(dzg, rows * 16)
    for i in range(rows * 16):
        cc.dz[i] = dz[i]
    net.backward_chunk(_mutp(sign), rows, cc, guard_ptr(g1), False)
    guard_check(g1, n, "backward_chunk grad")
    _ = len(g1)
    _ = len(dzg)
    # PPO flavour with the value head writes dz, dv and ratios inside their buffers too
    var acts = List[Int32](length=rows, fill=0)
    var lp_old = List[Float32](length=rows, fill=-2.0)
    var adv = List[Float32](length=rows, fill=0.5)
    var ret = List[Float32](length=rows, fill=1.0)
    var vals = List[Float32](length=rows, fill=0.0)
    for r in range(rows):
        for k in range(16):
            if valid[r * 16 + k] != 0:
                acts[r] = Int32(k)
                break
    var dz2 = guard_alloc(rows * 16)
    var dv2 = guard_alloc(rows)
    var rat2 = guard_alloc(rows)
    _ = route_ppo_chunk_loss(
        _mutp(cc.z), _mutp(vals), _mutp(valid), rows, 16, _mutp(acts), _mutp(lp_old), _mutp(adv), _mutp(ret),
        RoutePPOCfg(), 0.02, guard_ptr(dz2), guard_ptr(dv2), guard_ptr(rat2),
    )
    guard_check(dz2, rows * 16, "route_ppo dz")
    guard_check(dv2, rows, "route_ppo dv")
    guard_check(rat2, rows, "route_ppo ratio")
    _ = len(dz2)
    _ = len(dv2)
    _ = len(rat2)


def test_bc_reference() raises:
    var d0 = getenv("QUMO_BC_REF_D0")
    if d0.byte_length() == 0:
        print("test_bc_reference: QUMO_BC_REF_D0 not set, skipped (A6b is run by the lead / C13)")
        return
    var pyref = load_fixture("route/py_bc_reference.json")
    var summ = pyref.get("summary")
    var paths = List[String]()
    paths.append(d0)
    var td = read_teacher_files(paths)
    print("bc reference: records", td.raw, "kept", td.n)
    var ra = summ.get("val_agree")
    var rr = summ.get("val_regret")
    var ma = max(0.03, 3.0 * ra.get("std").as_float())
    var mr = max(0.03, 3.0 * rr.get("std").as_float())
    var a_lo = ra.get("min").as_float() - ma
    var a_hi = ra.get("max").as_float() + ma
    var r_lo = rr.get("min").as_float() - mr
    var r_hi = rr.get("max").as_float() + mr
    for seed in range(3):
        var cfg = BCConfig()
        cfg.seed = UInt64(seed)
        cfg.threads = 4
        var t0 = perf_counter_ns()
        var out = train_bc(cfg, td, None)
        var secs = Float64(perf_counter_ns() - t0) / 1e9
        var res = out[1].copy()
        var last = res.log[len(res.log) - 1].copy()
        print("seed", seed, "samples", res.samples, "val", res.val, "val_agree", last.val_agree, "val_regret", last.val_regret, "loss", last.loss, "wall_s", secs)
        if not (last.val_agree >= a_lo and last.val_agree <= a_hi):
            raise Error("A6b: val_agree " + String(last.val_agree) + " outside [" + String(a_lo) + ", " + String(a_hi) + "]")
        if not (last.val_regret >= r_lo and last.val_regret <= r_hi):
            raise Error("A6b: val_regret " + String(last.val_regret) + " outside [" + String(r_lo) + ", " + String(r_hi) + "]")


def main() raises:
    test_record_roundtrip()
    test_filter_rule()
    test_init_statistics()
    test_forward_vs_runtime()
    test_shipped_model_forward_and_warm_start()
    test_bc_golden()
    test_multichunk_vs_naive()
    test_evaluate_vs_naive()
    test_guard_regions()
    test_route_ppo_golden()
    test_route_gae_golden()
    test_planted_optimum()
    test_train_thread_independence()
    test_run_dir_and_warm_start()
    test_bc_reference()
    print("test_rl_router OK")
