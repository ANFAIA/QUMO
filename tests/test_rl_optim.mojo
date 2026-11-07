"""Tests for qumo.rl.optim (Adam, clip_grad_norm, cosine_epoch_lr) and qumo.rl.rlrng (seed tree, sampling, FMA barrier).

Adam is checked against the torch traces in `torch/adam_cosine_golden.json`, the three behaviour-cloning steps of
`torch/bc_golden.json` (frozen value head, lr schedule) and the three route-PPO steps of `torch/ppo_route_golden.json`
(global gradient clip, one optimizer across steps). The sparse-row skip is compared bit for bit with the dense update.
"""

from std.math import fma, log
from std.memory import bitcast
from qumo.ai.json import JsonValue
from qumo.sim.rng import Rng
from qumo.rl.tensor import FPtr
from qumo.rl.optim import Adam, clip_grad_norm, cosine_epoch_lr
from qumo.rl.rlrng import (
    TAG_INIT,
    TAG_COLLECT,
    TAG_EVAL_DET,
    TAG_EVAL_SAMPLED,
    TAG_BC_SPLIT,
    TAG_BC_EPOCH,
    TAG_BC_SLOT,
    TAG_JOBS,
    TAG_EXPORT_VEC,
    TAG_ROUTE_PPO,
    splitmix64,
    seed_for,
    uniform_f32,
    uniform_k24,
    uniform_sym_f32,
    normal_f32,
    randperm,
    gumbel_argmax,
    argmax_first,
    sample_softmax,
    sample_probs,
    mulf,
)
from qumo.rl.testutil import (
    load_fixture,
    flat_floats,
    flat_floats32,
    floats,
    floats32,
    ints,
    assert_close,
    assert_list_close,
    assert_list_close32,
    chi_square_gof,
    close,
    abs64,
    max_abs_diff32,
    f32_bits,
)


# ------------------------------------------------------------------------------------------------ helpers


def _bits_equal(a: List[Float32], b: List[Float32]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            return False
    return True


def _flat_params(v: JsonValue) raises -> List[Float32]:
    """The six route tensors in the order emb, b0, w1, b1, wv, bv, flattened row-major; a JSON null (a gradient that
    is None in torch) becomes zeros of the tensor's length."""
    var sizes: List[Int] = [1024, 8, 128, 16, 8, 1]
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


def _tensor_ranges() -> List[Int]:
    # (offset, length) pairs of emb, b0, w1, b1, wv, bv
    return [0, 1024, 1024, 8, 1032, 128, 1160, 16, 1176, 8, 1184, 1]


# ------------------------------------------------------------------------------------------------ Adam


def test_adam_trace() raises:
    var g = load_fixture("torch/adam_cosine_golden.json")
    var tr = g.get("adam_trace")
    var lr = tr.get("lr").as_float()
    var steps = tr.get("trace")
    var adam = Adam(3, lr)
    var x: List[Float32] = [1.0, -2.0, 0.5]
    for s in range(len(steps)):
        var st = steps.at(s)
        var grad = floats32(st.get("grad"))
        adam.step(x, grad, lr)
        assert_list_close32(x, floats32(st.get("x")), 2e-7, 2e-6, "adam trace x step " + String(s))
        assert_list_close32(adam.m, floats32(st.get("m")), 1e-8, 2e-6, "adam trace m step " + String(s))
        assert_list_close32(adam.v, floats32(st.get("v")), 1e-10, 2e-6, "adam trace v step " + String(s))
        if adam.steps_taken() != s + 1:
            raise Error("step counter")
    # The 4th step has a zero gradient yet the parameters still move on momentum (dense update).
    if x[0] == Float32(0.98109466):
        raise Error("zero-gradient step did not move the parameters")


def test_adam_bc_golden() raises:
    var g = load_fixture("torch/bc_golden.json")
    var steps = g.get("steps")
    for sparse in range(2):
        var w = _flat_params(g.get("init"))
        var adam = Adam(len(w), 0.002)
        adam.set_frozen(1176, 9)  # value head: grad is None in BC
        if sparse == 1:
            adam.mark_sparse_rows(0, 8, 128)  # embedding table
        var value_head_init = List[Float32]()
        for i in range(1176, 1185):
            value_head_init.append(w[i])
        for s in range(len(steps)):
            var st = steps.at(s)
            var grads = _flat_params(st.get("grads"))
            adam.step(w, grads, st.get("lr").as_float())
            var expect = _flat_params(st.get("weights_after"))
            # measured max difference to torch is 1.5e-8; the contract allows 3e-5 (Adam amplifies tiny gradient noise), 1e-6 here
            assert_list_close32(w, expect, 1e-6, 0.0, "bc weights after step " + String(s) + " sparse=" + String(sparse))
        # frozen value head never moved and has no optimizer state
        for i in range(9):
            if w[1176 + i] != value_head_init[i]:
                raise Error("frozen value head moved")
            if adam.m[1176 + i] != 0.0 or adam.v[1176 + i] != 0.0:
                raise Error("frozen value head acquired optimizer state")
        var a3 = g.get("adam_state_after_3")
        if adam.steps_taken() != Int(a3.get("step").as_float()):
            raise Error("step count after 3 steps")
        assert_list_close32(
            List[Float32](adam.m[0:1024]), flat_floats32(a3.get("emb_exp_avg")), 1e-6, 1e-4, "bc emb exp_avg"
        )
        assert_list_close32(
            List[Float32](adam.v[0:1024]), flat_floats32(a3.get("emb_exp_avg_sq")), 1e-9, 1e-4, "bc emb exp_avg_sq"
        )


def test_clip_and_route_ppo_golden() raises:
    var g = load_fixture("torch/ppo_route_golden.json")
    var steps = g.get("steps")
    var max_norm = g.get("max_norm").as_float()
    var lr = g.get("lr").as_float()
    var w = _flat_params(g.get("init"))
    var adam = Adam(len(w), lr)
    var ranges = _tensor_ranges()
    for s in range(len(steps)):
        var st = steps.at(s)
        var grads = _flat_params(st.get("grads_raw"))
        var norm = clip_grad_norm(grads, ranges, max_norm)
        assert_close(norm, st.get("grad_norm_before_clip").as_float(), 0.0, 1e-4, "grad norm step " + String(s))
        # clip coefficient: every gradient is multiplied by it
        var coef = st.get("clip_coef").as_float()
        assert_close(max_norm / (norm + 1e-6), coef, 0.0, 1e-4, "clip coef step " + String(s))
        assert_list_close32(grads, _flat_params(st.get("grads_clipped")), 1e-7, 2e-4, "clipped grads step " + String(s))
        adam.step(w, grads, lr)
        assert_list_close32(w, _flat_params(st.get("weights_after")), 1e-6, 0.0, "route ppo weights step " + String(s))

    # No clipping when the norm is below the maximum: coefficient exactly 1 (grads unchanged bit for bit).
    var small: List[Float32] = [0.1, -0.2, 0.05, 0.0, 0.3]
    var before = small.copy()
    var nrm = clip_grad_norm(small, [0, 3, 3, 2], 5.0)
    if not _bits_equal(small, before):
        raise Error("gradients below max_norm must be returned unchanged")
    assert_close(nrm, 0.37749172176353746, 0.0, 1e-6, "small norm")
    # A single-tensor call (empty ranges) clips the whole buffer.
    var big: List[Float32] = [3.0, 4.0]
    var n2 = clip_grad_norm(big, List[Int](), 1.0)
    assert_close(n2, 5.0, 0.0, 1e-6, "norm 5")
    assert_close(Float64(big[0]), 0.6, 1e-6, 0.0, "clipped x")
    assert_close(Float64(big[1]), 0.8, 1e-6, 0.0, "clipped y")


def test_frozen_matches_dense() raises:
    # Frozen slices are skipped; all other entries evolve exactly as in a dense optimizer (element-wise update).
    var n = 53
    var w1 = List[Float32](length=n, fill=0.0)
    var r = Rng(3)
    for i in range(n):
        w1[i] = Float32(r.uniform()) - 0.5
    var w2 = w1.copy()
    var dense = Adam(n, 1e-3)
    var frozen = Adam(n, 1e-3)
    frozen.set_frozen(10, 7)
    frozen.set_frozen(30, 1)
    frozen.set_frozen(40, 13)
    for s in range(12):
        var grads = List[Float32](length=n, fill=0.0)
        for i in range(n):
            grads[i] = Float32(r.uniform()) - 0.5
        dense.step(w1, grads, 1e-3 * Float64(s + 1))
        frozen.step(w2, grads, 1e-3 * Float64(s + 1))
    var init = List[Float32](length=n, fill=0.0)
    var r0 = Rng(3)
    for i in range(n):
        init[i] = Float32(r0.uniform()) - 0.5
    for i in range(n):
        var is_frozen = (i >= 10 and i < 17) or i == 30 or (i >= 40 and i < 53)
        if is_frozen:
            if w2[i] != init[i] or frozen.m[i] != 0.0 or frozen.v[i] != 0.0:
                raise Error("frozen element " + String(i) + " changed")
        else:
            if bitcast[DType.uint32](w2[i]) != bitcast[DType.uint32](w1[i]):
                raise Error("non-frozen element " + String(i) + " differs from the dense update")
            if bitcast[DType.uint32](frozen.m[i]) != bitcast[DType.uint32](dense.m[i]):
                raise Error("non-frozen m differs")
    if frozen.steps_taken() != 12:
        raise Error("step count")


def _sparse_grads(mut r: Rng, rows: Int, row_len: Int, step: Int, tail: Int) -> List[Float32]:
    """Gradient for rows * row_len sparse parameters followed by `tail` dense ones. Row `q` becomes active at step
    `q % 7` and is silent again for steps divisible by 4 (momentum keeps moving it); rows >= rows - 5 stay silent."""
    var g = List[Float32](length=rows * row_len + tail, fill=0.0)
    for q in range(rows - 5):
        if step >= (q % 7) and (step % 4 != 0 or q % 3 == 0):
            for j in range(row_len):
                g[q * row_len + j] = Float32(r.uniform()) - 0.5
    for j in range(tail):
        g[rows * row_len + j] = Float32(r.uniform()) - 0.5
    return g^


def test_sparse_rows_exact() raises:
    var rows = 40
    var row_len = 13
    var tail = 10
    var n = rows * row_len + tail
    var steps = 30
    var init = List[Float32](length=n, fill=0.0)
    var r0 = Rng(11)
    for i in range(n):
        init[i] = Float32(r0.uniform()) - 0.5
    # a frozen slice in the middle of the tail as well, to exercise region ordering
    var dense = Adam(n, 2e-3)
    var sparse = Adam(n, 2e-3)
    sparse.set_frozen(rows * row_len + 4, 3)
    dense.set_frozen(rows * row_len + 4, 3)
    sparse.mark_sparse_rows(0, row_len, rows)
    var wd = init.copy()
    var ws = init.copy()
    var rg_d = Rng(99)
    var rg_s = Rng(99)
    var ever = List[Bool](length=rows, fill=False)
    var snap_t = 0
    var snap_m = List[Float32]()
    var snap_v = List[Float32]()
    var snap_w = List[Float32]()
    for s in range(steps):
        var gd = _sparse_grads(rg_d, rows, row_len, s, tail)
        var gs = _sparse_grads(rg_s, rows, row_len, s, tail)
        for q in range(rows):
            for j in range(row_len):
                if gd[q * row_len + j] != 0.0:
                    ever[q] = True
        dense.step(wd, gd, 2e-3)
        sparse.step(ws, gs, 2e-3)
        if not _bits_equal(wd, ws) or not _bits_equal(dense.m, sparse.m) or not _bits_equal(dense.v, sparse.v):
            raise Error("sparse update differs from the dense update at step " + String(s))
        if s == 14:
            snap_t = sparse.steps_taken()
            snap_m = sparse.m.copy()
            snap_v = sparse.v.copy()
            snap_w = ws.copy()
    var n_ever = 0
    for q in range(rows):
        if ever[q]:
            n_ever += 1
    if sparse.touched_rows() != n_ever:
        raise Error("touched rows " + String(sparse.touched_rows()) + " vs expected " + String(n_ever))
    if n_ever != rows - 5:
        raise Error("test data should leave exactly 5 rows silent")

    # Resume: a fresh optimizer restored from the step-14 snapshot continues bit-identically.
    var resumed = Adam(n, 2e-3)
    resumed.set_frozen(rows * row_len + 4, 3)
    resumed.mark_sparse_rows(0, row_len, rows)
    resumed.load_state(snap_t, snap_m, snap_v)
    var wr = snap_w.copy()
    var rg = Rng(99)
    for s in range(steps):
        var gr = _sparse_grads(rg, rows, row_len, s, tail)
        if s > 14:
            resumed.step(wr, gr, 2e-3)
    if not _bits_equal(wr, ws) or not _bits_equal(resumed.m, sparse.m) or not _bits_equal(resumed.v, sparse.v):
        raise Error("resumed optimizer diverged from the uninterrupted run")
    if resumed.touched_rows() != sparse.touched_rows():
        raise Error("touched flags not restored by load_state")


def test_adam_misc() raises:
    # Constructor defaults and the step counter.
    var a = Adam(4, 0.1)
    if a.beta1 != 0.9 or a.beta2 != 0.999 or a.eps != 1e-8 or a.steps_taken() != 0:
        raise Error("defaults")
    # lr = 0 leaves parameters unchanged but still advances the moments and the counter.
    var w: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var g: List[Float32] = [0.5, -0.5, 0.25, 0.0]
    a.step(w, g, 0.0)
    if w[0] != 1.0 or w[3] != 4.0 or a.steps_taken() != 1 or a.m[0] == 0.0:
        raise Error("lr=0 step")
    # Bad restore arguments raise.
    var raised = False
    try:
        a.load_state(1, List[Float32](length=3, fill=0.0), List[Float32](length=4, fill=0.0))
    except e:
        raised = True
    if not raised:
        raise Error("load_state must reject a wrong length")


# ------------------------------------------------------------------------------------------------ cosine


def test_cosine() raises:
    var g = load_fixture("torch/adam_cosine_golden.json")
    var c = g.get("cosine")
    var base = c.get("base").as_float()
    var tmax = c.get("T_max").as_int()
    var lrs = floats(c.get("lr_per_epoch_torch"))
    for e in range(len(lrs)):
        var mine = cosine_epoch_lr(base, e, tmax)
        # relative 1e-12 (the last value is exactly 0.0 in torch: allow an absolute floor of 1e-18)
        assert_close(mine, lrs[e], 1e-18, 1e-12, "cosine lr epoch " + String(e))
    if cosine_epoch_lr(0.002, 0, 3) != 0.002:
        raise Error("epoch 0 is the base lr")
    assert_close(cosine_epoch_lr(0.002, 1, 3), 0.0015, 0.0, 1e-12, "bc golden lr 1")
    assert_close(cosine_epoch_lr(0.002, 2, 3), 0.0005, 0.0, 1e-12, "bc golden lr 2")
    if cosine_epoch_lr(0.002, 99, 3) != cosine_epoch_lr(0.002, 3, 3):
        raise Error("epoch clamp")
    if cosine_epoch_lr(0.002, 5, 0) != 0.002:
        raise Error("total_epochs <= 0 returns the base lr")


# ------------------------------------------------------------------------------------------------ rlrng


def test_splitmix_and_seed_for() raises:
    # Known SplitMix64 output for state 0 and agreement with the Rng seeding function.
    if splitmix64(0) != 0xE220A8397B1DCDAF:
        raise Error("splitmix64(0) wrong: " + String(splitmix64(0)))
    var xs: List[UInt64] = [0, 1, 12345, 0xFFFFFFFFFFFFFFFF, 0x9E3779B97F4A7C15]
    for i in range(len(xs)):
        var st = xs[i]
        var expect = Rng._splitmix(st)
        if splitmix64(xs[i]) != expect:
            raise Error("splitmix64 differs from Rng._splitmix")
    # distinct over a grid of (tag, a, b, c): 10 * 16 * 16 * 16 seeds
    var all = List[UInt64](capacity=10 * 16 * 16 * 16)
    for tag in range(1, 11):
        for a in range(16):
            for b in range(16):
                for c in range(16):
                    all.append(seed_for(2024, tag, a, b, c))
    sort(all)
    for i in range(1, len(all)):
        if all[i] == all[i - 1]:
            raise Error("seed_for collision")
    # deterministic, depends on every component and on the base seed, negative components are distinct values
    if seed_for(7, TAG_COLLECT, 1, 2, 3) != seed_for(7, TAG_COLLECT, 1, 2, 3):
        raise Error("seed_for not deterministic")
    var s0 = seed_for(7, TAG_COLLECT, 1, 2, 3)
    if s0 == seed_for(8, TAG_COLLECT, 1, 2, 3) or s0 == seed_for(7, TAG_EVAL_DET, 1, 2, 3):
        raise Error("base / tag ignored")
    if s0 == seed_for(7, TAG_COLLECT, 2, 2, 3) or s0 == seed_for(7, TAG_COLLECT, 1, 3, 3) or s0 == seed_for(7, TAG_COLLECT, 1, 2, 4):
        raise Error("component ignored")
    if seed_for(7, TAG_COLLECT, -1) == seed_for(7, TAG_COLLECT, 1):
        raise Error("negative component aliases the positive one")
    if seed_for(7, TAG_INIT) != seed_for(7, TAG_INIT, 0, 0, 0):
        raise Error("defaults are zeros")
    # the ten tags are distinct constants 1..10
    var tags: List[Int] = [
        TAG_INIT, TAG_COLLECT, TAG_EVAL_DET, TAG_EVAL_SAMPLED, TAG_BC_SPLIT,
        TAG_BC_EPOCH, TAG_BC_SLOT, TAG_JOBS, TAG_EXPORT_VEC, TAG_ROUTE_PPO,
    ]
    for i in range(10):
        if tags[i] != i + 1:
            raise Error("tag numbering")


def test_uniforms() raises:
    var r = Rng(5)
    var r2 = Rng(5)
    var sum: Float64 = 0.0
    var n = 100000
    for _ in range(n):
        var u = uniform_f32(r)
        var k = uniform_k24(r2)
        if u < 0.0 or u >= 1.0:
            raise Error("uniform_f32 out of [0, 1)")
        if k < 0 or k >= 16777216:
            raise Error("uniform_k24 out of range")
        if u != Float32(k) * Float32(5.9604644775390625e-08):
            raise Error("uniform_f32 is not k * 2^-24")
        sum += Float64(u)
    var mean = sum / Float64(n)
    if abs64(mean - 0.5) > 5.0 * 0.28868 / 316.2:  # 5 sigma
        raise Error("uniform mean " + String(mean))
    # symmetric uniform: bound respected, mean about zero
    var r3 = Rng(6)
    var ssum: Float64 = 0.0
    for _ in range(n):
        var x = uniform_sym_f32(r3, 0.25)
        if x < -0.25 or x >= 0.25:
            raise Error("uniform_sym_f32 out of range")
        ssum += Float64(x)
    if abs64(ssum / Float64(n)) > 5.0 * 0.1443 / 316.2:
        raise Error("uniform_sym mean")
    # normal: mean 0, variance 1
    var r4 = Rng(7)
    var s1: Float64 = 0.0
    var s2: Float64 = 0.0
    for _ in range(n):
        var z = Float64(normal_f32(r4))
        s1 += z
        s2 += z * z
    if abs64(s1 / Float64(n)) > 5.0 / 316.2:
        raise Error("normal mean")
    if abs64(s2 / Float64(n) - 1.0) > 5.0 * 1.4143 / 316.2:
        raise Error("normal variance")


def test_randperm() raises:
    var r = Rng(21)
    if len(randperm(r, 0)) != 0:
        raise Error("randperm(0)")
    if randperm(r, 1)[0] != 0:
        raise Error("randperm(1)")
    var sizes: List[Int] = [2, 3, 10, 100, 1000]
    for si in range(len(sizes)):
        var n = sizes[si]
        var p = randperm(r, n)
        var seen = List[Bool](length=n, fill=False)
        for i in range(n):
            if p[i] < 0 or p[i] >= n or seen[p[i]]:
                raise Error("randperm is not a permutation, n=" + String(n))
            seen[p[i]] = True
    # uniform over the 6 permutations of 3 elements
    var counts = List[Int](length=6, fill=0)
    var trials = 12000
    for _ in range(trials):
        var p = randperm(r, 3)
        counts[p[0] * 2 + (1 if p[1] > p[2] else 0)] += 1
    var probs = List[Float64](length=6, fill=1.0 / 6.0)
    var chi = chi_square_gof(counts, probs)
    if chi.z > 4.0:
        raise Error("randperm(3) not uniform: z = " + String(chi.z))
    # uniform position of element 0 in permutations of 7
    var pos = List[Int](length=7, fill=0)
    for _ in range(14000):
        var p = randperm(r, 7)
        for i in range(7):
            if p[i] == 0:
                pos[i] += 1
    var chi7 = chi_square_gof(pos, List[Float64](length=7, fill=1.0 / 7.0))
    if chi7.z > 4.0:
        raise Error("randperm(7) position not uniform: z = " + String(chi7.z))
    # deterministic given the seed
    var ra = Rng(33)
    var rb = Rng(33)
    var pa = randperm(ra, 50)
    var pb = randperm(rb, 50)
    for i in range(50):
        if pa[i] != pb[i]:
            raise Error("randperm not deterministic")


def _row(vals: List[Float32]) -> List[Float32]:
    return vals.copy()


def test_gumbel_argmax() raises:
    # All-masked row (-1e10): the noise is absorbed, every value ties and the first maximum, index 0, wins
    # for any generator state (MOJO_PORT.md 4.1, O-3 / C-7).
    var masked = List[Float32](length=10, fill=-1e10)
    var pm = masked.unsafe_ptr().as_unsafe_any_origin()
    for s in range(1000):
        var r = Rng(UInt64(s) * 7919 + 1)
        if gumbel_argmax(pm, 10, r) != 0:
            raise Error("all-masked row must return index 0 (state " + String(s) + ")")
    # Ties return the first index (large equal logits absorb the noise).
    var ties: List[Float32] = [-1e10, 1e10, 1e10, -1e10, 1e10]
    var pt = ties.unsafe_ptr().as_unsafe_any_origin()
    for s in range(1000):
        var r = Rng(UInt64(s) + 17)
        if gumbel_argmax(pt, 5, r) != 1:
            raise Error("tie must return the first maximum")
    # A NaN entry never wins unless it is element 0.
    var nan = Float32.MAX * 2.0 - Float32.MAX * 2.0
    var with_nan: List[Float32] = [0.0, nan, 0.5, nan, 1.0]
    var pn = with_nan.unsafe_ptr().as_unsafe_any_origin()
    for s in range(1000):
        var r = Rng(UInt64(s) + 3)
        var a = gumbel_argmax(pn, 5, r)
        if a == 1 or a == 3:
            raise Error("a NaN entry was chosen")
    var nan_first: List[Float32] = [nan, 5.0, 100.0]
    var pnf = nan_first.unsafe_ptr().as_unsafe_any_origin()
    var rr = Rng(1)
    if gumbel_argmax(pnf, 3, rr) != 0:
        raise Error("a NaN at element 0 cannot be beaten")
    # Logits of 100: the true maximum is picked (the Gumbel noise is far smaller than the gap).
    var big: List[Float32] = [0.0, 100.0, 3.0, 99.0 - 80.0]
    var pb = big.unsafe_ptr().as_unsafe_any_origin()
    for s in range(1000):
        var r = Rng(UInt64(s) + 11)
        if gumbel_argmax(pb, 4, r) != 1:
            raise Error("gumbel_argmax with logits of 100 did not pick the maximum")
    # Single element, and n <= 0 returns 0 without drawing.
    var one: List[Float32] = [3.0]
    var r1 = Rng(2)
    var before = r1.copy()
    if gumbel_argmax(one.unsafe_ptr().as_unsafe_any_origin(), 1, r1) != 0:
        raise Error("n = 1")
    if gumbel_argmax(one.unsafe_ptr().as_unsafe_any_origin(), 0, r1) != 0:
        raise Error("n = 0")
    # exactly one draw per element: after n = 1 and a no-op call the stream advanced by one value
    var adv = before.copy()
    _ = adv.next_u64()
    if r1.next_u64() != adv.next_u64():
        raise Error("gumbel_argmax must consume exactly one draw per element")
    # Distribution: argmax of Gumbel-perturbed logits is a softmax sample.
    var logits: List[Float32] = [0.0, 1.0, 2.0, 3.0, -1.0]
    var pl = logits.unsafe_ptr().as_unsafe_any_origin()
    var counts = List[Int](length=5, fill=0)
    var rg = Rng(2025)
    var trials = 60000
    for _ in range(trials):
        counts[gumbel_argmax(pl, 5, rg)] += 1
    var probs = _softmax64(logits)
    var chi = chi_square_gof(counts, probs)
    if chi.z > 4.0:
        raise Error("gumbel_argmax distribution: z = " + String(chi.z))
    _ = len(logits)


def _softmax64(logits: List[Float32]) -> List[Float64]:
    var mx = Float64(logits[0])
    for i in range(len(logits)):
        if Float64(logits[i]) > mx:
            mx = Float64(logits[i])
    var out = List[Float64](capacity=len(logits))
    var s: Float64 = 0.0
    for i in range(len(logits)):
        var e = exp64(Float64(logits[i]) - mx)
        out.append(e)
        s += e
    for i in range(len(out)):
        out[i] = out[i] / s
    return out^


def exp64(x: Float64) -> Float64:
    from std.math import exp

    return exp(x)


def test_argmax_first() raises:
    var nan = Float32.MAX * 2.0 - Float32.MAX * 2.0
    var v: List[Float32] = [1.0, 3.0, 3.0, 2.0]
    if argmax_first(v.unsafe_ptr().as_unsafe_any_origin(), 4) != 1:
        raise Error("first maximum")
    var w: List[Float32] = [0.0, nan, 5.0, nan]
    if argmax_first(w.unsafe_ptr().as_unsafe_any_origin(), 4) != 2:
        raise Error("NaN skipped")
    var x: List[Float32] = [nan, 1.0, 2.0]
    if argmax_first(x.unsafe_ptr().as_unsafe_any_origin(), 3) != 0:
        raise Error("NaN first element")
    var y: List[Float32] = [100.0, 99.0, 100.0]
    if argmax_first(y.unsafe_ptr().as_unsafe_any_origin(), 3) != 0:
        raise Error("logits of 100: first maximum")
    var z: List[Float32] = [-1e10, -1e10, -1e10]
    if argmax_first(z.unsafe_ptr().as_unsafe_any_origin(), 3) != 0:
        raise Error("all masked -> 0")
    if argmax_first(z.unsafe_ptr().as_unsafe_any_origin(), 0) != 0:
        raise Error("n = 0")
    _ = len(v) + len(w) + len(x) + len(y) + len(z)


def test_sample_softmax() raises:
    # Distribution on moderate logits.
    var logits: List[Float32] = [0.5, -1.0, 2.0, 0.0, 1.0, -3.0]
    var pl = logits.unsafe_ptr().as_unsafe_any_origin()
    var counts = List[Int](length=6, fill=0)
    var r = Rng(77)
    for _ in range(60000):
        counts[sample_softmax(pl, 6, r)] += 1
    var chi = chi_square_gof(counts, _softmax64(logits))
    if chi.z > 4.0:
        raise Error("sample_softmax distribution: z = " + String(chi.z))
    # Logits of 100: valid draws, no NaN, correct frequencies (p = 0.6225 / 0.3775 for a gap of 0.5).
    var big: List[Float32] = [100.0, 99.5, 0.0]
    var pbig = big.unsafe_ptr().as_unsafe_any_origin()
    var cb = List[Int](length=3, fill=0)
    for _ in range(40000):
        var a = sample_softmax(pbig, 3, r)
        if a < 0 or a > 2:
            raise Error("invalid draw")
        cb[a] += 1
    var chib = chi_square_gof(cb, _softmax64(big))
    if chib.z > 4.0 or cb[2] != 0:
        raise Error("sample_softmax with logits of 100: z = " + String(chib.z))
    # Very large logits still work (the max shift keeps the exponents <= 0).
    var huge: List[Float32] = [3e30, 3e30, -3e30]
    var phuge = huge.unsafe_ptr().as_unsafe_any_origin()
    var ch = List[Int](length=3, fill=0)
    for _ in range(4000):
        ch[sample_softmax(phuge, 3, r)] += 1
    if ch[2] != 0 or ch[0] < 1700 or ch[1] < 1700:
        raise Error("huge logits")
    # Masked entries are never drawn; an all-masked row yields a valid index.
    var part: List[Float32] = [-1e10, 0.0, -1e10, 0.3]
    var pp = part.unsafe_ptr().as_unsafe_any_origin()
    for _ in range(3000):
        var a = sample_softmax(pp, 4, r)
        if a != 1 and a != 3:
            raise Error("masked action drawn")
    var allm = List[Float32](length=5, fill=-1e10)
    var pam = allm.unsafe_ptr().as_unsafe_any_origin()
    for _ in range(200):
        var a = sample_softmax(pam, 5, r)
        if a < 0 or a > 4:
            raise Error("all-masked draw out of range")
    # Degenerate inputs consume one draw and return a valid index.
    var inf = Float32.MAX * 2.0
    var nan = inf - inf
    var bad: List[Float32] = [0.0, inf, 1.0]
    var bad2: List[Float32] = [nan, 1.0, 2.0]
    var a1 = sample_softmax(bad.unsafe_ptr().as_unsafe_any_origin(), 3, r)
    var a2 = sample_softmax(bad2.unsafe_ptr().as_unsafe_any_origin(), 3, r)
    if a1 != 1:
        raise Error("+inf logit: expected the infinite entry")
    if a2 < 0 or a2 > 2:
        raise Error("NaN logit: out of range")
    # One draw per call when the row is valid.
    var rr = Rng(5)
    var rc = Rng(5)
    _ = sample_softmax(pl, 6, rr)
    _ = uniform_f32(rc)
    if rr.next_u64() != rc.next_u64():
        raise Error("sample_softmax must consume exactly one uniform")
    _ = len(logits) + len(big) + len(huge) + len(part) + len(allm) + len(bad) + len(bad2)


def test_sample_probs() raises:
    var w: List[Float32] = [0.0, 0.5, 0.0, 1.5, 0.0]
    var pw = w.unsafe_ptr().as_unsafe_any_origin()
    var counts = List[Int](length=5, fill=0)
    var r = Rng(8)
    for _ in range(40000):
        counts[sample_probs(pw, 5, r)] += 1
    if counts[0] != 0 or counts[2] != 0 or counts[4] != 0:
        raise Error("zero-weight entry drawn")
    var probs: List[Float64] = [0.0, 0.25, 0.0, 0.75, 0.0]
    var chi = chi_square_gof(counts, probs)
    if chi.z > 4.0:
        raise Error("sample_probs distribution: z = " + String(chi.z))
    # Unnormalised weights are fine (u * sum).
    var big: List[Float32] = [1000.0, 3000.0]
    var cb = List[Int](length=2, fill=0)
    for _ in range(20000):
        cb[sample_probs(big.unsafe_ptr().as_unsafe_any_origin(), 2, r)] += 1
    var chib = chi_square_gof(cb, [0.25, 0.75])
    if chib.z > 4.0:
        raise Error("sample_probs unnormalised")
    # All-zero weights: no valid draw, returns 0.
    var zero = List[Float32](length=4, fill=0.0)
    if sample_probs(zero.unsafe_ptr().as_unsafe_any_origin(), 4, r) != 0:
        raise Error("all-zero weights")
    _ = len(w) + len(big) + len(zero)


def test_mulf() raises:
    var fx = load_fixture("envs/reward_fma_cases.json")
    var cases = fx.get("cases")
    var n_unfused_ok = 0
    var n_fused_differs = 0
    for i in range(len(cases)):
        var c = cases.at(i)
        var achieved = Float32(c.get("achieved").as_float())
        var penalty = Float32(c.get("penalty").as_float())
        var w = Float32(c.get("pauli_layer_reward").as_float())
        var count = Float32(c.get("count").as_int())
        # (achieved - penalty) + (w * count): the product goes through the barrier, the sum stays separate.
        var diff = achieved - penalty
        var unfused = diff + mulf(w, count)
        var unfused_bits = c.get("unfused_bits").as_int()
        if Int(f32_bits(unfused)) != unfused_bits:
            raise Error(
                "mulf reward case " + String(i) + ": got bits " + String(f32_bits(unfused)) + ", expected " + String(unfused_bits)
            )
        n_unfused_ok += 1
        # The explicitly fused form must give the other value, so the case does discriminate.
        var fused = fma(w, count, diff)
        if Int(f32_bits(fused)) != c.get("fused_bits").as_int():
            raise Error("fused reference bits differ for case " + String(i))
        if f32_bits(fused) != f32_bits(unfused):
            n_fused_differs += 1
    if n_unfused_ok != len(cases) or n_fused_differs != len(cases):
        raise Error("every fixture case must distinguish fused from unfused")
    # The documented example: achieved 0, penalty 1e-4, w 0.01, count 5.
    var d = Float32(0.0) - Float32(1e-4)
    var ex = d + mulf(Float32(0.01), Float32(5.0))
    if Int(f32_bits(ex)) != 1028416496:
        raise Error("documented example")
    if mulf(3.0, 4.0) != 12.0 or mulf(-0.5, 8.0) != -4.0:
        raise Error("mulf arithmetic")


def main() raises:
    test_adam_trace()
    test_adam_bc_golden()
    test_clip_and_route_ppo_golden()
    test_frozen_matches_dense()
    test_sparse_rows_exact()
    test_adam_misc()
    test_cosine()
    test_splitmix_and_seed_for()
    test_uniforms()
    test_randperm()
    test_gumbel_argmax()
    test_argmax_first()
    test_sample_softmax()
    test_sample_probs()
    test_mulf()
    print("test_rl_optim OK")
