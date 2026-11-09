"""Router network (sign-embedding MLP, `qumo.rlroute.v0`) with hand-derived forward and backward.

    pre = b0 + sum_i sign(obs_i) E[i, :]       (E is [128, H]; a sparse "embedding bag": no dense input is built)
    h   = relu(pre)
    z   = W1 h + b1                            logits, 16 active-swap slots
    v   = wv . h + bv                          value head (trained by route PPO only; frozen in behaviour cloning)

All parameters live in one flat f32 buffer so Adam, gradient clipping and the container writer see a single
array: `[E (128*H) | b0 (H) | W1 (16*H) | b1 (16) | wv (H) | bv (1)]`. The tensor shapes and names of the exported
container are those of the previous trainer (the reference framework's layout): `embeddings.weight` [128, H] (feature-major, no transpose),
`embeddings.bias` [H], `action.0.weight` [16, H], `action.0.bias` [16], `value.0.weight` [1, H], `value.0.bias` [1].

Single-threaded and re-entrant kernels over a chunk of rows (threading happens one level up in `bc.mojo`, with a
fixed reduction tree, so results do not depend on the thread count). The GEMMs are those of `qrl.tensor`.
"""

from std.math import sqrt
from std.os import abort
from qrl.model import ModelFile
from qrl.rng import Rng
from qrl.tensor import (
    FPtr,
    gemm_nt,
    gemm_nn,
    gemm_tn,
    add_bias_rows,
    relu_inplace,
    relu_backward,
    col_sum_acc,
    saxpy,
    add_inplace,
)
from qrl.rlrng import seed_for, TAG_INIT, normal_f32, uniform_sym_f32
from qrl.container import TensorOut

comptime ROUTER_OBS = 128
comptime ROUTER_SLOTS = 16
comptime SPtr = MutPointer[Int8, MutAnyOrigin]


def _e647(msg: String) -> Error:
    return Error("QM-E647: cannot warm-start the router from this model: " + msg + ". Hint: --init needs a qumo.rlroute.v0 directory (hidden size H, 128 observations, 16 slots).")


struct RouterCache(Movable):
    """Scratch of one virtual worker for up to `cap` rows: activations and gradients of the last chunk."""

    var cap: Int
    var hidden: Int
    var pre: List[Float32]  # [cap, H]
    var h: List[Float32]  # [cap, H]
    var z: List[Float32]  # [cap, 16] raw logits
    var v: List[Float32]  # [cap]
    var dz: List[Float32]  # [cap, 16]
    var dv: List[Float32]  # [cap]
    var dh: List[Float32]  # [cap, H]

    def __init__(out self, cap: Int, hidden: Int):
        self.cap = cap
        self.hidden = hidden
        self.pre = List[Float32](length=cap * hidden, fill=0.0)
        self.h = List[Float32](length=cap * hidden, fill=0.0)
        self.z = List[Float32](length=cap * ROUTER_SLOTS, fill=0.0)
        self.v = List[Float32](length=cap, fill=0.0)
        self.dz = List[Float32](length=cap * ROUTER_SLOTS, fill=0.0)
        self.dv = List[Float32](length=cap, fill=0.0)
        self.dh = List[Float32](length=cap * hidden, fill=0.0)


struct RouterNet(Movable):
    var hidden: Int
    var params: List[Float32]

    def __init__(out self, hidden: Int, seed: UInt64):
        """Fresh network in the reference framework's init: `E ~ N(0,1) / sqrt(128)`, `b0 = 0`, `W1, b1, wv, bv ~ U(+-1/sqrt(H))`,
        drawn in that order (weight before bias) from the stream `seed_for(seed, TAG_INIT)`."""
        self.hidden = hidden
        var h = hidden
        var n = ROUTER_OBS * h + h + ROUTER_SLOTS * h + ROUTER_SLOTS + h + 1
        self.params = List[Float32](length=n, fill=0.0)
        var rng = Rng(seed_for(seed, TAG_INIT))
        var emb_scale = Float32(1.0 / sqrt(Float64(ROUTER_OBS)))
        for i in range(ROUTER_OBS * h):
            self.params[i] = normal_f32(rng) * emb_scale
        # b0 stays zero
        var bound = Float32(1.0 / sqrt(Float64(max(h, 1))))
        var o = self.off_w1()
        for i in range(ROUTER_SLOTS * h):
            self.params[o + i] = uniform_sym_f32(rng, bound)
        o = self.off_b1()
        for i in range(ROUTER_SLOTS):
            self.params[o + i] = uniform_sym_f32(rng, bound)
        o = self.off_wv()
        for i in range(h):
            self.params[o + i] = uniform_sym_f32(rng, bound)
        self.params[self.off_bv()] = uniform_sym_f32(rng, bound)

    @staticmethod
    def from_model(m: ModelFile) raises -> RouterNet:
        """Warm start from a router container (an f16 shipped model decoded to f32, or a Mojo f32 run directory).
        The value head is optional: when the model has none it is initialised to zero. QM-E647 when the tensors are
        missing or have the wrong shapes."""
        if not m.has_tensor("embeddings.weight") or not m.has_tensor("embeddings.bias"):
            raise _e647("'" + m.id + "' has no embeddings.weight / embeddings.bias")
        if not m.has_tensor("action.0.weight") or not m.has_tensor("action.0.bias"):
            raise _e647("'" + m.id + "' has no action.0.weight / action.0.bias")
        var es = m.shape("embeddings.weight")
        if len(es) != 2 or es[0] != ROUTER_OBS or es[1] <= 0:
            raise _e647("embeddings.weight of '" + m.id + "' is not [128, H]")
        var h = es[1]
        var net = RouterNet(h, 0)
        net.params = List[Float32](length=len(net.params), fill=0.0)
        var emb = m.tensor("embeddings.weight")
        var b0 = m.tensor("embeddings.bias")
        var w1 = m.tensor("action.0.weight")
        var b1 = m.tensor("action.0.bias")
        if len(b0) != h:
            raise _e647("embeddings.bias of '" + m.id + "' has " + String(len(b0)) + " entries, expected " + String(h))
        if len(w1) != ROUTER_SLOTS * h or len(b1) != ROUTER_SLOTS:
            raise _e647("action.0.* of '" + m.id + "' is not [16, " + String(h) + "] / [16]")
        for i in range(len(emb)):
            net.params[i] = emb[i]
        for i in range(h):
            net.params[net.off_b0() + i] = b0[i]
        for i in range(len(w1)):
            net.params[net.off_w1() + i] = w1[i]
        for i in range(ROUTER_SLOTS):
            net.params[net.off_b1() + i] = b1[i]
        if m.has_tensor("value.0.weight") and m.has_tensor("value.0.bias"):
            var wv = m.tensor("value.0.weight")
            var bv = m.tensor("value.0.bias")
            if len(wv) != h or len(bv) != 1:
                raise _e647("value.0.* of '" + m.id + "' is not [1, " + String(h) + "] / [1]")
            for i in range(h):
                net.params[net.off_wv() + i] = wv[i]
            net.params[net.off_bv()] = bv[0]
        return net^

    # ---- layout of the flat parameter buffer
    def off_emb(self) -> Int:
        return 0

    def off_b0(self) -> Int:
        return ROUTER_OBS * self.hidden

    def off_w1(self) -> Int:
        return self.off_b0() + self.hidden

    def off_b1(self) -> Int:
        return self.off_w1() + ROUTER_SLOTS * self.hidden

    def off_wv(self) -> Int:
        return self.off_b1() + ROUTER_SLOTS

    def off_bv(self) -> Int:
        return self.off_wv() + self.hidden

    def param_count(self) -> Int:
        return self.off_bv() + 1

    def tensor_ranges(self) -> List[Int]:
        """(offset, length) pairs of the six parameter tensors, in the reference framework's parameter order (emb, b0, W1, b1,
        wv, bv): the `ranges` argument of `optim.clip_grad_norm`."""
        var r = List[Int]()
        r.append(self.off_emb())
        r.append(ROUTER_OBS * self.hidden)
        r.append(self.off_b0())
        r.append(self.hidden)
        r.append(self.off_w1())
        r.append(ROUTER_SLOTS * self.hidden)
        r.append(self.off_b1())
        r.append(ROUTER_SLOTS)
        r.append(self.off_wv())
        r.append(self.hidden)
        r.append(self.off_bv())
        r.append(1)
        return r^

    def to_tensors(self) -> List[TensorOut]:
        """The six tensors of the container in export order, all f32 (the exporter decides about f16)."""
        var h = self.hidden
        var out = List[TensorOut]()
        out.append(TensorOut("embeddings.weight", _shape2(ROUTER_OBS, h), self._slice(self.off_emb(), ROUTER_OBS * h), "f32"))
        out.append(TensorOut("embeddings.bias", _shape1(h), self._slice(self.off_b0(), h), "f32"))
        out.append(TensorOut("action.0.weight", _shape2(ROUTER_SLOTS, h), self._slice(self.off_w1(), ROUTER_SLOTS * h), "f32"))
        out.append(TensorOut("action.0.bias", _shape1(ROUTER_SLOTS), self._slice(self.off_b1(), ROUTER_SLOTS), "f32"))
        out.append(TensorOut("value.0.weight", _shape2(1, h), self._slice(self.off_wv(), h), "f32"))
        out.append(TensorOut("value.0.bias", _shape1(1), self._slice(self.off_bv(), 1), "f32"))
        return out^

    def _pp(self) -> FPtr:
        """Mutable-typed pointer to the parameters for read-only use by the (const-agnostic) kernels."""
        return self.params.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin()

    def _slice(self, off: Int, n: Int) -> List[Float32]:
        var r = List[Float32](capacity=n)
        for i in range(n):
            r.append(self.params[off + i])
        return r^

    # ---- chunk kernels

    def forward_chunk(self, sign: SPtr, rows: Int, mut cc: RouterCache, with_value: Bool = False):
        """Forward of `rows` rows (at most `cc.cap`). `sign` is [rows, 128] with entries in {-1, 0, +1}. Fills
        `cc.pre`, `cc.h`, `cc.z` (raw logits, all 16 slots) and, with `with_value`, `cc.v`."""
        if rows > cc.cap or rows < 0 or cc.hidden != self.hidden:
            abort("qrl.router_net.forward_chunk: " + String(rows) + " rows exceed the cache capacity " + String(cc.cap))
        var h = self.hidden
        var pp = self._pp()
        var emb = pp.unsafe_offset(self.off_emb())
        var b0 = pp.unsafe_offset(self.off_b0())
        var w1 = pp.unsafe_offset(self.off_w1())
        var b1 = pp.unsafe_offset(self.off_b1())
        var pre = cc.pre.unsafe_ptr().as_unsafe_any_origin()
        var hid = cc.h.unsafe_ptr().as_unsafe_any_origin()
        var z = cc.z.unsafe_ptr().as_unsafe_any_origin()
        for r in range(rows):
            var row = pre.unsafe_offset(r * h)
            for j in range(h):
                row[unsafe_offset=j] = b0[unsafe_offset=j]
            var sg = sign.unsafe_offset(r * ROUTER_OBS)
            for i in range(ROUTER_OBS):
                var s = sg[unsafe_offset=i]
                if s > 0:
                    add_inplace(row, emb.unsafe_offset(i * h), h)
                elif s < 0:
                    saxpy(row, emb.unsafe_offset(i * h), -1.0, h)
        for i in range(rows * h):
            hid[unsafe_offset=i] = pre[unsafe_offset=i]
        relu_inplace(hid, rows * h)
        gemm_nt(rows, ROUTER_SLOTS, h, hid, h, w1, h, z, ROUTER_SLOTS, False)
        add_bias_rows(z, rows, ROUTER_SLOTS, b1)
        if with_value:
            var wv = pp.unsafe_offset(self.off_wv())
            var v = cc.v.unsafe_ptr().as_unsafe_any_origin()
            gemm_nt(rows, 1, h, hid, h, wv, h, v, 1, False)
            var bv = pp[unsafe_offset=self.off_bv()]
            for r in range(rows):
                v[unsafe_offset=r] = v[unsafe_offset=r] + bv
        _ = len(self.params)
        _ = len(cc.pre)
        _ = len(cc.h)
        _ = len(cc.z)

    def backward_chunk(self, sign: SPtr, rows: Int, mut cc: RouterCache, grad: FPtr, with_value: Bool = False):
        """Backward of the rows of the last `forward_chunk`. `cc.dz` ([rows,16], zero at masked slots) and, with
        `with_value`, `cc.dv` ([rows]) hold the loss gradients w.r.t. the logits and values. Accumulates into `grad`
        (same layout as `params`): `dW1 += dz^T h`, `db1 += sum dz`, `dh = dz W1 (+ dv wv)`, ReLU mask `pre > 0`,
        `db0 += sum dpre`, `dE[i, :] += sign_i * dpre` (sparse scatter). The value head gradients are written only
        with `with_value`."""
        if rows > cc.cap or rows < 0 or cc.hidden != self.hidden:
            abort("qrl.router_net.backward_chunk: " + String(rows) + " rows exceed the cache capacity " + String(cc.cap))
        var h = self.hidden
        var pp = self._pp()
        var w1 = pp.unsafe_offset(self.off_w1())
        var pre = cc.pre.unsafe_ptr().as_unsafe_any_origin()
        var hid = cc.h.unsafe_ptr().as_unsafe_any_origin()
        var dz = cc.dz.unsafe_ptr().as_unsafe_any_origin()
        var dh = cc.dh.unsafe_ptr().as_unsafe_any_origin()
        gemm_tn(ROUTER_SLOTS, h, rows, dz, ROUTER_SLOTS, hid, h, grad.unsafe_offset(self.off_w1()), h, True)
        col_sum_acc(dz, rows, ROUTER_SLOTS, grad.unsafe_offset(self.off_b1()))
        gemm_nn(rows, h, ROUTER_SLOTS, dz, ROUTER_SLOTS, w1, h, dh, h, False)
        if with_value:
            var dv = cc.dv.unsafe_ptr().as_unsafe_any_origin()
            var wv = pp.unsafe_offset(self.off_wv())
            gemm_tn(1, h, rows, dv, 1, hid, h, grad.unsafe_offset(self.off_wv()), h, True)
            col_sum_acc(dv, rows, 1, grad.unsafe_offset(self.off_bv()))
            for r in range(rows):
                saxpy(dh.unsafe_offset(r * h), wv, dv[unsafe_offset=r], h)
        relu_backward(dh, pre, rows * h)
        col_sum_acc(dh, rows, h, grad.unsafe_offset(self.off_b0()))
        var gemb = grad.unsafe_offset(self.off_emb())
        for r in range(rows):
            var drow = dh.unsafe_offset(r * h)
            var sg = sign.unsafe_offset(r * ROUTER_OBS)
            for i in range(ROUTER_OBS):
                var s = sg[unsafe_offset=i]
                if s > 0:
                    add_inplace(gemb.unsafe_offset(i * h), drow, h)
                elif s < 0:
                    saxpy(gemb.unsafe_offset(i * h), drow, -1.0, h)
        _ = len(self.params)
        _ = len(cc.pre)
        _ = len(cc.h)
        _ = len(cc.dz)
        _ = len(cc.dh)
        _ = len(cc.dv)


def _shape1(a: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    return s^


def _shape2(a: Int, b: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    return s^
