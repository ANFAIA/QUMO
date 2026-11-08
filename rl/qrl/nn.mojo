"""Parameter store and the synthesis network (reference `BasicPolicy` shape) with a hand-derived backward pass.

MOJO_PORT.md sections 4.4 and 5. No autograd: `BasicNet.backward_chunk` is the explicit chain rule of the network

    h0 = relu(be + sum_{i in S'} Wt[i, :])                     sparse embedding bag, S' = obs_perm[p][S] for a twist p
    h_j = relu(W_j h_{j-1} + b_j)                              common layers (j = 1 .. k)
    raw = head_a(h_k)   value = head_v(h_k)                    heads: Linear chains, ReLU between hidden layers only
    logits[a] = raw[act_perm[p][a]]                            post-twist logits

Parameter layout (the model-container layout of `qumo.rlsynth.v0`, so a net and a container are a plain copy of each
other): one flat `List[Float32]` in registration order

    embeddings.weight  [OBS, E]   (transposed w.r.t. the reference [E, OBS])      embeddings.bias [E]
    common.0.weight    [C1, E]    common.0.bias [C1]    common.1.weight [C2, C1] ...   (reference layout [out, in])
    action.0.weight    ...        action.<last>.weight [A, previous]
    value.0.weight     ...        value.<last>.weight  [1, previous]

with *consecutive* layer numbering (`common.0`, `common.1`, ...) as `MLPPolicy` requires (D9). Biases are stored as
`1 x n` tensors.

Threading: `BasicNet` methods take `self` read-only and keep all scratch in the caller's `ChunkCache` / `ActorScratch`,
so any number of virtual workers can run them concurrently on one net. Gradients are written into a per-worker flat
buffer of `param_count()` floats (see `GradBufs`); the reduction of those buffers is `GradBufs.reduce_into`, a fixed
index-order sum, so the result never depends on the thread count (MOJO_PORT.md section 3, rule 3).

Raw-pointer code has no bounds checks. Observation indices outside `[0, OBS)` are skipped (as `MLPPolicy` does); a twist
index outside the symmetry table aborts (programming error).
"""

from std.math import sqrt
from std.os import abort
from qrl.model import ModelFile
from qrl.rng import Rng
from qrl.tensor import (
    FPtr,
    CHUNK_ROWS,
    VW,
    gemm_nt,
    gemm_nn,
    gemm_tn,
    add_bias_rows,
    relu_inplace,
    relu_backward,
    col_sum_acc,
    add_inplace,
)
from qrl.rlrng import TAG_INIT, seed_for, uniform_sym_f32


# ------------------------------------------------------------------------------------------------ small helpers


@always_inline
def _wptr(lst: List[Float32]) -> FPtr:
    """Mutable-typed raw pointer to a list that is only read through it (the GEMM API takes `FPtr`)."""
    return lst.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin()


@always_inline
def _copy_row(dst: FPtr, src: FPtr, n: Int):
    """dst[0:n] = src[0:n] (SIMD, exact)."""
    comptime W = 8
    var main = (n // W) * W
    var i = 0
    while i < main:
        dst.unsafe_store(i, src.unsafe_load[width=W](i))
        i += W
    while i < n:
        dst[unsafe_offset=i] = src[unsafe_offset=i]
        i += 1


def _qm_spec(msg: String) -> Error:
    return Error("QM-E642: network spec: " + msg + ". Hint: obs_size, num_actions, emb and every layer width must be positive and the symmetry tables must be permutations.")


def _qm_warm(msg: String) -> Error:
    return Error("QM-E647: warm start is incompatible: " + msg + ". Hint: the container must hold embeddings.*, consecutive common.k.*, action.k.* and value.k.* tensors that chain.")


# ------------------------------------------------------------------------------------------------ ParamStore


struct ParamStore(Movable, Copyable):
    """One flat parameter buffer plus named (rows x cols) tensors, laid out back to back in registration order."""

    var names: List[String]
    var shapes: List[Int]  # rows, cols per tensor
    var offsets: List[Int]
    var data: List[Float32]

    def __init__(out self):
        self.names = List[String]()
        self.shapes = List[Int]()
        self.offsets = List[Int]()
        self.data = List[Float32]()

    def add(mut self, name: String, rows: Int, cols: Int) -> Int:
        """Appends a zero-filled `rows x cols` tensor and returns its offset into `data`."""
        var off = len(self.data)
        self.names.append(name)
        self.shapes.append(rows)
        self.shapes.append(cols)
        self.offsets.append(off)
        var n = rows * cols
        self.data.reserve(off + n)
        for _ in range(n):
            self.data.append(0.0)
        return off

    def total(self) -> Int:
        return len(self.data)

    def count(self) -> Int:
        """Number of registered tensors."""
        return len(self.names)

    def index_of(self, name: String) raises -> Int:
        for i in range(len(self.names)):
            if self.names[i] == name:
                return i
        raise Error("QM-E640: ParamStore has no tensor '" + name + "'")

    def offset(self, name: String) raises -> Int:
        return self.offsets[self.index_of(name)]

    def rows_of(self, i: Int) -> Int:
        return self.shapes[2 * i]

    def cols_of(self, i: Int) -> Int:
        return self.shapes[2 * i + 1]

    def size_of(self, i: Int) -> Int:
        return self.shapes[2 * i] * self.shapes[2 * i + 1]

    def ptr(mut self) -> FPtr:
        return self.data.unsafe_ptr().as_unsafe_any_origin()


# ------------------------------------------------------------------------------------------------ gradient buffers


struct GradBufs(Movable):
    """`count` (default `VW`) private gradient buffers of `n_params` floats, one per virtual worker, plus the fixed-order
    reduction (MOJO_PORT.md 3, rule 3 and 4.4)."""

    var n_params: Int
    var count: Int
    var data: List[Float32]

    def __init__(out self, n_params: Int, count: Int = VW):
        self.n_params = n_params
        self.count = count
        self.data = List[Float32](length=n_params * count, fill=0.0)

    def ptr(mut self, i: Int) -> FPtr:
        """Pointer to buffer `i` (keep `self` alive while the pointer is used)."""
        return self.data.unsafe_ptr().as_unsafe_any_origin().unsafe_offset(i * self.n_params)

    def zero(mut self):
        for i in range(len(self.data)):
            self.data[i] = 0.0

    def reduce_into(self, mut out: List[Float32]):
        """`out = buf_0 + buf_1 + ... + buf_{count-1}`, summed in buffer-index order (exact adds; the result is
        independent of how the virtual workers were mapped to threads). `out` is resized to `n_params`."""
        if len(out) != self.n_params:
            out = List[Float32](length=self.n_params, fill=0.0)
        var po = out.unsafe_ptr().as_unsafe_any_origin()
        var pd = _wptr(self.data)
        _copy_row(po, pd, self.n_params)
        for b in range(1, self.count):
            add_inplace(po, pd.unsafe_offset(b * self.n_params), self.n_params)
        _ = len(out)
        _ = len(self.data)


# ------------------------------------------------------------------------------------------------ sparse batch


struct SparseBatch(Movable):
    """CSR rows of active observation indices: row `r` is `idx[off[r] : off[r + 1]]` with twist `perm[r]`
    (`-1` = no twist). `off` always has `rows() + 1` entries (it starts as `[0]`)."""

    var off: List[Int32]  # length N+1
    var idx: List[Int32]
    var perm: List[Int32]  # -1 = no twist

    def __init__(out self):
        self.off = List[Int32]()
        self.off.append(0)
        self.idx = List[Int32]()
        self.perm = List[Int32]()

    def rows(self) -> Int:
        return len(self.perm)

    def append_row(mut self, idx: Span[Int32, _], twist: Int):
        """Appends one row (indices in the given order, which is also the summation order of the embedding bag)."""
        for i in range(len(idx)):
            self.idx.append(idx[i])
        self.off.append(Int32(len(self.idx)))
        self.perm.append(Int32(twist))

    def append_row_ints(mut self, idx: List[Int], twist: Int):
        for i in range(len(idx)):
            self.idx.append(Int32(idx[i]))
        self.off.append(Int32(len(self.idx)))
        self.perm.append(Int32(twist))

    def clear(mut self):
        self.off.clear()
        self.off.append(0)
        self.idx.clear()
        self.perm.clear()

    def nnz(self) -> Int:
        return len(self.idx)


# ------------------------------------------------------------------------------------------------ spec


@fieldwise_init
struct NetSpec(Copyable):
    var obs_size: Int
    var num_actions: Int
    var emb: Int
    var common: List[Int]
    var policy_hidden: List[Int]
    var value_hidden: List[Int]
    var obs_perms: List[List[Int]]
    var act_perms: List[List[Int]]

    def __init__(out self):
        self.obs_size = 0
        self.num_actions = 0
        self.emb = 0
        self.common = List[Int]()
        self.policy_hidden = List[Int]()
        self.value_hidden = List[Int]()
        self.obs_perms = List[List[Int]]()
        self.act_perms = List[List[Int]]()

    def params(self) -> Int:
        """Parameter count (MOJO_PORT.md 4.4); for one common layer `OBS*E + E + E*C + C + C*A + A + C + 1`."""
        var total = self.obs_size * self.emb + self.emb
        var prev = self.emb
        for i in range(len(self.common)):
            total += prev * self.common[i] + self.common[i]
            prev = self.common[i]
        var pa = prev
        for i in range(len(self.policy_hidden)):
            total += pa * self.policy_hidden[i] + self.policy_hidden[i]
            pa = self.policy_hidden[i]
        total += pa * self.num_actions + self.num_actions
        var pv = prev
        for i in range(len(self.value_hidden)):
            total += pv * self.value_hidden[i] + self.value_hidden[i]
            pv = self.value_hidden[i]
        total += pv + 1
        return total

    def num_twists(self) -> Int:
        return len(self.obs_perms)

    def max_width(self) -> Int:
        """Widest activation matrix (embedding, any common or hidden layer, the logits)."""
        var m = max(self.emb, self.num_actions)
        for i in range(len(self.common)):
            m = max(m, self.common[i])
        for i in range(len(self.policy_hidden)):
            m = max(m, self.policy_hidden[i])
        for i in range(len(self.value_hidden)):
            m = max(m, self.value_hidden[i])
        return m

    def validate(self) raises:
        """QM-E642 on a malformed spec (non-positive sizes, symmetry tables that are not permutations)."""
        if self.obs_size <= 0:
            raise _qm_spec("obs_size " + String(self.obs_size))
        if self.num_actions <= 0:
            raise _qm_spec("num_actions " + String(self.num_actions))
        if self.emb <= 0:
            raise _qm_spec("emb " + String(self.emb))
        for i in range(len(self.common)):
            if self.common[i] <= 0:
                raise _qm_spec("common layer width " + String(self.common[i]))
        for i in range(len(self.policy_hidden)):
            if self.policy_hidden[i] <= 0:
                raise _qm_spec("policy hidden width " + String(self.policy_hidden[i]))
        for i in range(len(self.value_hidden)):
            if self.value_hidden[i] <= 0:
                raise _qm_spec("value hidden width " + String(self.value_hidden[i]))
        if len(self.obs_perms) != len(self.act_perms):
            raise _qm_spec("obs_perms / act_perms count mismatch")
        for p in range(len(self.obs_perms)):
            _check_perm(self.obs_perms[p], self.obs_size, "obs_perms[" + String(p) + "]")
            _check_perm(self.act_perms[p], self.num_actions, "act_perms[" + String(p) + "]")


def _check_perm(perm: List[Int], n: Int, what: String) raises:
    if len(perm) != n:
        raise _qm_spec(what + " has length " + String(len(perm)) + ", expected " + String(n))
    var seen = List[Bool](length=n, fill=False)
    for i in range(n):
        var v = perm[i]
        if v < 0 or v >= n or seen[v]:
            raise _qm_spec(what + " is not a permutation of 0.." + String(n - 1))
        seen[v] = True


# ------------------------------------------------------------------------------------------------ layout


@fieldwise_init
struct _Lin(Copyable):
    """One Linear layer inside the flat parameter buffer: weight `[n_out, n_in]` at `w`, bias `[n_out]` at `b`."""

    var w: Int
    var b: Int
    var n_in: Int
    var n_out: Int


struct _Layout(Copyable):
    var emb_w: Int
    var emb_b: Int
    var common: List[_Lin]
    var action: List[_Lin]
    var value: List[_Lin]

    def __init__(out self):
        self.emb_w = 0
        self.emb_b = 0
        self.common = List[_Lin]()
        self.action = List[_Lin]()
        self.value = List[_Lin]()


def _add_chain(
    mut ps: ParamStore, mut out: List[_Lin], prefix: String, n_in: Int, hidden: List[Int], last: Int
):
    var prev = n_in
    var n_layers = len(hidden) + 1
    for k in range(n_layers):
        var width = last
        if k < len(hidden):
            width = hidden[k]
        var w = ps.add(prefix + "." + String(k) + ".weight", width, prev)
        var b = ps.add(prefix + "." + String(k) + ".bias", 1, width)
        out.append(_Lin(w, b, prev, width))
        prev = width


def _build_layout(spec: NetSpec, mut ps: ParamStore) -> _Layout:
    var lay = _Layout()
    lay.emb_w = ps.add("embeddings.weight", spec.obs_size, spec.emb)
    lay.emb_b = ps.add("embeddings.bias", 1, spec.emb)
    var prev = spec.emb
    for k in range(len(spec.common)):
        var c = spec.common[k]
        var w = ps.add("common." + String(k) + ".weight", c, prev)
        var b = ps.add("common." + String(k) + ".bias", 1, c)
        lay.common.append(_Lin(w, b, prev, c))
        prev = c
    _add_chain(ps, lay.action, "action", prev, spec.policy_hidden, spec.num_actions)
    _add_chain(ps, lay.value, "value", prev, spec.value_hidden, 1)
    return lay^


def _trunk_width(spec: NetSpec) -> Int:
    if len(spec.common) > 0:
        return spec.common[len(spec.common) - 1]
    return spec.emb


# ------------------------------------------------------------------------------------------------ scratch


struct ChunkCache(Movable):
    """Per virtual worker scratch for up to `capacity` rows (default `CHUNK_ROWS`).

    After `forward_chunk(sb, lo, hi, cc)`: `cc.rows = hi - lo`, `cc.logits[r * A + a]` are the post-twist logits and
    `cc.values[r]` the values of row `lo + r`. The remaining fields are the activations that `backward_chunk` needs
    (post-ReLU matrices: the ReLU mask `pre > 0` equals `post > 0`) and gradient scratch."""

    var rows: Int
    var cap: Int
    var logits: List[Float32]  # post-twist logits of the chunk, rows x num_actions
    var values: List[Float32]
    var acts: List[Float32]  # h0, common outputs, policy hidden outputs, value hidden outputs (cap x width each)
    var act_off: List[Int]
    var act_w: List[Int]
    var raw: List[Float32]  # pre-twist action output, cap x A
    var rawv: List[Float32]
    var draw: List[Float32]  # un-twisted logit gradient, cap x A
    var gbuf: List[Float32]  # three gradient scratch matrices of cap x max_width
    var gw: Int  # cap * max_width: size of one gradient scratch matrix
    var n_common: Int
    var n_ph: Int
    var n_vh: Int
    var num_actions: Int

    def __init__(out self, spec: NetSpec, capacity: Int = CHUNK_ROWS):
        self.rows = 0
        self.cap = capacity
        var A = spec.num_actions
        self.num_actions = A
        self.logits = List[Float32](length=capacity * A, fill=0.0)
        self.values = List[Float32](length=capacity, fill=0.0)
        self.act_off = List[Int]()
        self.act_w = List[Int]()
        var total = 0
        self.act_off.append(total)
        self.act_w.append(spec.emb)
        total += capacity * spec.emb
        for i in range(len(spec.common)):
            self.act_off.append(total)
            self.act_w.append(spec.common[i])
            total += capacity * spec.common[i]
        for i in range(len(spec.policy_hidden)):
            self.act_off.append(total)
            self.act_w.append(spec.policy_hidden[i])
            total += capacity * spec.policy_hidden[i]
        for i in range(len(spec.value_hidden)):
            self.act_off.append(total)
            self.act_w.append(spec.value_hidden[i])
            total += capacity * spec.value_hidden[i]
        self.acts = List[Float32](length=total, fill=0.0)
        self.raw = List[Float32](length=capacity * A, fill=0.0)
        self.rawv = List[Float32](length=capacity, fill=0.0)
        self.draw = List[Float32](length=capacity * A, fill=0.0)
        self.gw = capacity * spec.max_width()
        self.gbuf = List[Float32](length=3 * self.gw, fill=0.0)
        self.n_common = len(spec.common)
        self.n_ph = len(spec.policy_hidden)
        self.n_vh = len(spec.value_hidden)


struct ActorScratch(Movable):
    """Per task scratch for `forward_one` (six vectors of the widest layer; no allocation per call)."""

    var buf: List[Float32]
    var w: Int  # width of one vector

    def __init__(out self, spec: NetSpec):
        self.w = spec.max_width()
        self.buf = List[Float32](length=6 * self.w, fill=0.0)


# ------------------------------------------------------------------------------------------------ layer kernels


@always_inline
def _linear_fwd(rows: Int, n_in: Int, n_out: Int, x: FPtr, w: FPtr, b: FPtr, y: FPtr, relu: Bool):
    """y[rows, n_out] = x[rows, n_in] W^T + b (ReLU optional); W is `[n_out, n_in]`."""
    gemm_nt(rows, n_out, n_in, x, n_in, w, n_in, y, n_out, False)
    add_bias_rows(y, rows, n_out, b)
    if relu:
        relu_inplace(y, rows * n_out)


@always_inline
def _linear_bwd(
    rows: Int, n_in: Int, n_out: Int, x: FPtr, dy: FPtr, w: FPtr, dw: FPtr, db: FPtr, dx: FPtr, acc_dx: Bool
):
    """Backward of `_linear_fwd` without the activation: `dW += dy^T x`, `db += colsum(dy)`, `dx (+)= dy W`."""
    gemm_tn(n_out, n_in, rows, dy, n_out, x, n_in, dw, n_in, True)
    col_sum_acc(dy, rows, n_out, db)
    gemm_nn(rows, n_in, n_out, dy, n_out, w, n_in, dx, n_in, acc_dx)


# ------------------------------------------------------------------------------------------------ the network


struct BasicNet(Movable):
    """The reference `BasicPolicy`: sparse embedding bag, ReLU common layers, Linear action / value heads."""

    var spec: NetSpec
    var ps: ParamStore
    var lay: _Layout

    def __init__(out self, spec: NetSpec, seed: UInt64, randomize: Bool = True) raises:
        """Registers the layout and (when `randomize`) draws the reference default init with the RNG stream
        `seed_for(seed, TAG_INIT)`: embeddings, common.*, action.*, value.*, each Linear weight then bias from
        `U(-1/sqrt(fan_in), 1/sqrt(fan_in))` with `fan_in` = in_features of the weight (embeddings: OBS), drawn in
        storage order. `randomize = False` leaves every parameter at zero (used by `from_model`)."""
        self.spec = spec.copy()
        self.ps = ParamStore()
        self.lay = _Layout()
        self.spec.validate()
        self.lay = _build_layout(self.spec, self.ps)
        if randomize:
            var r = Rng(seed_for(seed, TAG_INIT))
            self._init_linear(r, self.lay.emb_w, self.lay.emb_b, self.spec.obs_size, self.spec.obs_size * self.spec.emb, self.spec.emb)
            for i in range(len(self.lay.common)):
                var l = self.lay.common[i].copy()
                self._init_linear(r, l.w, l.b, l.n_in, l.n_in * l.n_out, l.n_out)
            for i in range(len(self.lay.action)):
                var l = self.lay.action[i].copy()
                self._init_linear(r, l.w, l.b, l.n_in, l.n_in * l.n_out, l.n_out)
            for i in range(len(self.lay.value)):
                var l = self.lay.value[i].copy()
                self._init_linear(r, l.w, l.b, l.n_in, l.n_in * l.n_out, l.n_out)

    def _init_linear(mut self, mut r: Rng, w_off: Int, b_off: Int, fan_in: Int, w_len: Int, b_len: Int):
        var bound = Float32(1.0 / sqrt(Float64(fan_in)))
        for i in range(w_len):
            self.ps.data[w_off + i] = uniform_sym_f32(r, bound)
        for i in range(b_len):
            self.ps.data[b_off + i] = uniform_sym_f32(r, bound)

    @staticmethod
    def from_model(m: ModelFile) raises -> BasicNet:
        """Warm start from any container (f16 already decoded to f32 by the loader). The architecture is read off the
        tensors: `embeddings.weight [OBS, E]`, consecutive `common.k`, `action.k`, `value.k` (the last action layer
        has A rows, the last value layer one row); symmetries come from the manifest. QM-E647 if anything does not
        chain."""
        if not m.has_tensor("embeddings.weight") or not m.has_tensor("embeddings.bias"):
            raise _qm_warm("missing embeddings.weight / embeddings.bias")
        var esh = m.shape("embeddings.weight")
        if len(esh) != 2:
            raise _qm_warm("embeddings.weight is not 2-D")
        var spec = NetSpec()
        spec.obs_size = esh[0]
        spec.emb = esh[1]
        var prev = spec.emb
        var k = 0
        while m.has_tensor("common." + String(k) + ".weight"):
            var sh = m.shape("common." + String(k) + ".weight")
            if len(sh) != 2 or sh[1] != prev:
                raise _qm_warm("common." + String(k) + ".weight does not chain (input " + String(prev) + ")")
            spec.common.append(sh[0])
            prev = sh[0]
            k += 1
        var trunk = prev
        var widths_a = List[Int]()
        k = 0
        var pa = trunk
        while m.has_tensor("action." + String(k) + ".weight"):
            var sh = m.shape("action." + String(k) + ".weight")
            if len(sh) != 2 or sh[1] != pa:
                raise _qm_warm("action." + String(k) + ".weight does not chain (input " + String(pa) + ")")
            widths_a.append(sh[0])
            pa = sh[0]
            k += 1
        if len(widths_a) == 0:
            raise _qm_warm("no action head")
        spec.num_actions = widths_a[len(widths_a) - 1]
        for i in range(len(widths_a) - 1):
            spec.policy_hidden.append(widths_a[i])
        var widths_v = List[Int]()
        k = 0
        var pv = trunk
        while m.has_tensor("value." + String(k) + ".weight"):
            var sh = m.shape("value." + String(k) + ".weight")
            if len(sh) != 2 or sh[1] != pv:
                raise _qm_warm("value." + String(k) + ".weight does not chain (input " + String(pv) + ")")
            widths_v.append(sh[0])
            pv = sh[0]
            k += 1
        if len(widths_v) == 0 or widths_v[len(widths_v) - 1] != 1:
            raise _qm_warm("the value head must end in one output")
        for i in range(len(widths_v) - 1):
            spec.value_hidden.append(widths_v[i])
        if len(m.obs_perms) != len(m.act_perms):
            raise _qm_warm("obs_perms / act_perms count mismatch")
        spec.obs_perms = m.obs_perms.copy()
        spec.act_perms = m.act_perms.copy()
        if m.num_actions() > 0 and m.num_actions() != spec.num_actions:
            raise _qm_warm("action head size " + String(spec.num_actions) + " != manifest actions " + String(m.num_actions()))
        if m.obs_size > 0 and m.obs_size != spec.obs_size:
            raise _qm_warm("obs size " + String(spec.obs_size) + " != manifest obs.size " + String(m.obs_size))
        try:
            spec.validate()
        except e:
            raise _qm_warm(String(e))
        var net = BasicNet(spec, 0, False)
        # Copy every tensor in registration order, checking its size.
        for i in range(net.ps.count()):
            var name = net.ps.names[i]
            var src = m.tensors[m.tensor_index(name)].copy()
            if len(src) != net.ps.size_of(i):
                raise _qm_warm("tensor '" + name + "' has " + String(len(src)) + " values, expected " + String(net.ps.size_of(i)))
            var off = net.ps.offsets[i]
            for j in range(len(src)):
                net.ps.data[off + j] = src[j]
        return net^

    def param_count(self) -> Int:
        return self.ps.total()

    # -------------------------------------------------------------------------------------- actor path

    def forward_one(self, idx: Span[Int32, _], twist: Int, mut sc: ActorScratch, logits: FPtr) -> Float32:
        """Single-state forward. `idx` are the active observation indices (relabelled with `obs_perms[twist]` when
        `twist >= 0`), `logits` receives the `num_actions` post-twist logits (`logits[a] = raw[act_perms[twist][a]]`)
        and the value is returned. No heap allocation: all scratch lives in `sc`."""
        var spec_p = len(self.spec.obs_perms)
        if twist >= spec_p:
            abort("qrl.nn.BasicNet.forward_one: twist " + String(twist) + " but the net has " + String(spec_p) + " symmetries")
        var E = self.spec.emb
        var A = self.spec.num_actions
        var OBS = self.spec.obs_size
        var wt = _wptr(self.ps.data)
        var s = sc.buf.unsafe_ptr().as_unsafe_any_origin()
        var mw = sc.w
        var t_a = s
        var t_b = s.unsafe_offset(mw)
        var h_a = s.unsafe_offset(2 * mw)
        var h_b = s.unsafe_offset(3 * mw)
        var raw = s.unsafe_offset(4 * mw)
        # embedding bag
        _copy_row(t_a, wt.unsafe_offset(self.lay.emb_b), E)
        for j in range(len(idx)):
            var i = Int(idx[j])
            if i < 0 or i >= OBS:
                continue
            if twist >= 0:
                i = self.spec.obs_perms[twist][i]
            add_inplace(t_a, wt.unsafe_offset(self.lay.emb_w + i * E), E)
        relu_inplace(t_a, E)
        var cur = t_a
        var other = t_b
        for k in range(len(self.lay.common)):
            var l = self.lay.common[k].copy()
            _linear_fwd(1, l.n_in, l.n_out, cur, wt.unsafe_offset(l.w), wt.unsafe_offset(l.b), other, True)
            var tmp = cur
            cur = other
            other = tmp
        # action head
        var na = len(self.lay.action)
        var x = cur
        for k in range(na):
            var l = self.lay.action[k].copy()
            var y = raw
            if k + 1 < na:
                y = h_a
                if k % 2 == 1:
                    y = h_b
            _linear_fwd(1, l.n_in, l.n_out, x, wt.unsafe_offset(l.w), wt.unsafe_offset(l.b), y, k + 1 < na)
            x = y
        # value head (ping-pong over h_a / h_b; the logits are already in `raw`)
        var nv = len(self.lay.value)
        x = cur
        var v: Float32 = 0.0
        for k in range(nv):
            var l = self.lay.value[k].copy()
            var y = h_a
            if k % 2 == 1:
                y = h_b
            _linear_fwd(1, l.n_in, l.n_out, x, wt.unsafe_offset(l.w), wt.unsafe_offset(l.b), y, k + 1 < nv)
            x = y
        v = x[unsafe_offset=0]
        if twist >= 0:
            for a in range(A):
                logits[unsafe_offset=a] = raw[unsafe_offset = self.spec.act_perms[twist][a]]
        else:
            _copy_row(logits, raw, A)
        _ = len(self.ps.data)
        _ = len(sc.buf)
        return v

    # -------------------------------------------------------------------------------------- batched path

    def _check_chunk(self, sb: SparseBatch, lo: Int, hi: Int, cc: ChunkCache, what: String):
        if lo < 0 or hi < lo or hi > sb.rows() or len(sb.off) != sb.rows() + 1:
            abort("qrl.nn.BasicNet." + what + ": bad row range [" + String(lo) + ", " + String(hi) + ") for a batch of " + String(sb.rows()) + " rows (off has " + String(len(sb.off)) + " entries)")
        if hi - lo > cc.cap:
            abort("qrl.nn.BasicNet." + what + ": " + String(hi - lo) + " rows exceed the cache capacity " + String(cc.cap))
        if (
            cc.n_common != len(self.spec.common)
            or cc.n_ph != len(self.spec.policy_hidden)
            or cc.n_vh != len(self.spec.value_hidden)
            or cc.num_actions != self.spec.num_actions
            or cc.act_w[0] != self.spec.emb
        ):
            abort("qrl.nn.BasicNet." + what + ": the ChunkCache was built for a different NetSpec")

    def forward_chunk(self, sb: SparseBatch, lo: Int, hi: Int, mut cc: ChunkCache):
        """Batched forward over rows `lo..hi` of `sb` (at most `cc.cap` rows). Logits (post-twist) and values land in
        `cc.logits` / `cc.values`; the activations stay in `cc` for `backward_chunk`."""
        self._check_chunk(sb, lo, hi, cc, "forward_chunk")
        var R = hi - lo
        cc.rows = R
        if R == 0:
            return
        var E = self.spec.emb
        var A = self.spec.num_actions
        var OBS = self.spec.obs_size
        var P = len(self.spec.obs_perms)
        var wt = _wptr(self.ps.data)
        var acts = cc.acts.unsafe_ptr().as_unsafe_any_origin()
        var h0 = acts.unsafe_offset(cc.act_off[0])
        var bias0 = wt.unsafe_offset(self.lay.emb_b)
        var emb_w = wt.unsafe_offset(self.lay.emb_w)
        for r in range(R):
            var row = lo + r
            var hr = h0.unsafe_offset(r * E)
            _copy_row(hr, bias0, E)
            var p = Int(sb.perm[row])
            if p >= P:
                abort("qrl.nn.BasicNet.forward_chunk: row " + String(row) + " has twist " + String(p) + " but the net has " + String(P) + " symmetries")
            var q = Int(sb.off[row])
            var qe = Int(sb.off[row + 1])
            while q < qe:
                var i = Int(sb.idx[q])
                q += 1
                if i < 0 or i >= OBS:
                    continue
                if p >= 0:
                    i = self.spec.obs_perms[p][i]
                add_inplace(hr, emb_w.unsafe_offset(i * E), E)
        relu_inplace(h0, R * E)
        # common layers: acts[j] = relu(acts[j-1] W_j^T + b_j)
        for k in range(len(self.lay.common)):
            var l = self.lay.common[k].copy()
            _linear_fwd(
                R, l.n_in, l.n_out, acts.unsafe_offset(cc.act_off[k]), wt.unsafe_offset(l.w), wt.unsafe_offset(l.b),
                acts.unsafe_offset(cc.act_off[k + 1]), True,
            )
        var trunk = acts.unsafe_offset(cc.act_off[cc.n_common])
        # action head -> raw (R x A)
        var raw = cc.raw.unsafe_ptr().as_unsafe_any_origin()
        var na = len(self.lay.action)
        var x = trunk
        for k in range(na):
            var l = self.lay.action[k].copy()
            var y = raw
            if k + 1 < na:
                y = acts.unsafe_offset(cc.act_off[1 + cc.n_common + k])
            _linear_fwd(R, l.n_in, l.n_out, x, wt.unsafe_offset(l.w), wt.unsafe_offset(l.b), y, k + 1 < na)
            x = y
        # value head -> rawv (R)
        var rawv = cc.rawv.unsafe_ptr().as_unsafe_any_origin()
        var nv = len(self.lay.value)
        x = trunk
        for k in range(nv):
            var l = self.lay.value[k].copy()
            var y = rawv
            if k + 1 < nv:
                y = acts.unsafe_offset(cc.act_off[1 + cc.n_common + cc.n_ph + k])
            _linear_fwd(R, l.n_in, l.n_out, x, wt.unsafe_offset(l.w), wt.unsafe_offset(l.b), y, k + 1 < nv)
            x = y
        # post-twist logits and values
        var lg = cc.logits.unsafe_ptr().as_unsafe_any_origin()
        for r in range(R):
            var p = Int(sb.perm[lo + r])
            var dst = lg.unsafe_offset(r * A)
            var src = raw.unsafe_offset(r * A)
            if p >= 0:
                for a in range(A):
                    dst[unsafe_offset=a] = src[unsafe_offset = self.spec.act_perms[p][a]]
            else:
                _copy_row(dst, src, A)
            cc.values[r] = cc.rawv[r]
        _ = len(self.ps.data)
        _ = len(cc.acts)
        _ = len(cc.raw)
        _ = len(cc.rawv)
        _ = len(cc.logits)

    def backward_chunk(
        self, sb: SparseBatch, lo: Int, hi: Int, mut cc: ChunkCache, dz: FPtr, dv: FPtr, grad: FPtr
    ):
        """Accumulates (`+=`) the gradient of rows `lo..hi` into `grad` (a flat buffer laid out like `ps.data`).

        Must follow `forward_chunk` on the same rows and cache. `dz[r * A + a]` is dLoss/d(post-twist logit) and `dv[r]`
        dLoss/d(value) of row `lo + r`; both already carry the batch normalisation 1/N. The heads' gradients are summed
        into the shared hidden gradient (action first, then value), the ReLU masks are `post > 0`, the logit gradient
        is un-twisted (`draw[act_perm[p][a]] = dz[a]`), and the embedding gradient is the sparse scatter
        `dWt[obs_perm[p][i], :] += dpre0[r, :]`, `dbe += dpre0[r, :]`."""
        self._check_chunk(sb, lo, hi, cc, "backward_chunk")
        var R = hi - lo
        if R == 0:
            return
        if cc.rows != R:
            abort("qrl.nn.BasicNet.backward_chunk: the cache holds " + String(cc.rows) + " rows, asked for " + String(R))
        var E = self.spec.emb
        var A = self.spec.num_actions
        var OBS = self.spec.obs_size
        var wt = _wptr(self.ps.data)
        var acts = cc.acts.unsafe_ptr().as_unsafe_any_origin()
        var gb = cc.gbuf.unsafe_ptr().as_unsafe_any_origin()
        var g_t = gb  # trunk gradient / common-chain ping
        var g_a = gb.unsafe_offset(cc.gw)  # head scratch / common-chain pong
        var g_b = gb.unsafe_offset(2 * cc.gw)  # head scratch
        # un-twist the logit gradient
        var draw = cc.draw.unsafe_ptr().as_unsafe_any_origin()
        for r in range(R):
            var p = Int(sb.perm[lo + r])
            var dst = draw.unsafe_offset(r * A)
            var src = dz.unsafe_offset(r * A)
            if p >= 0:
                for a in range(A):
                    dst[unsafe_offset = self.spec.act_perms[p][a]] = src[unsafe_offset=a]
            else:
                _copy_row(dst, src, A)
        var trunk = acts.unsafe_offset(cc.act_off[cc.n_common])
        # action head (writes the trunk gradient), then value head (adds to it)
        var heads = 2
        for head in range(heads):
            var n_l = len(self.lay.action)
            if head == 1:
                n_l = len(self.lay.value)
            var dy = draw
            if head == 1:
                dy = dv
            var hbase = 1 + cc.n_common
            if head == 1:
                hbase = 1 + cc.n_common + cc.n_ph
            var k = n_l - 1
            while k >= 0:
                var l = self.lay.action[k].copy()
                if head == 1:
                    l = self.lay.value[k].copy()
                var x = trunk
                if k > 0:
                    x = acts.unsafe_offset(cc.act_off[hbase + k - 1])
                var dx = g_t
                var acc_dx = head == 1
                if k > 0:
                    dx = g_a
                    if k % 2 == 0:
                        dx = g_b
                    acc_dx = False
                _linear_bwd(
                    R, l.n_in, l.n_out, x, dy, wt.unsafe_offset(l.w), grad.unsafe_offset(l.w), grad.unsafe_offset(l.b),
                    dx, acc_dx,
                )
                if k > 0:
                    relu_backward(dx, x, R * l.n_in)
                    dy = dx
                k -= 1
        # trunk ReLU, then the common layers in reverse (ping-pong g_t <-> g_a)
        relu_backward(g_t, trunk, R * _trunk_width(self.spec))
        var cur_is_t = True
        var j = len(self.lay.common)
        while j >= 1:
            var l = self.lay.common[j - 1].copy()
            var cur = g_t
            var nxt = g_a
            if not cur_is_t:
                cur = g_a
                nxt = g_t
            var x = acts.unsafe_offset(cc.act_off[j - 1])
            _linear_bwd(
                R, l.n_in, l.n_out, x, cur, wt.unsafe_offset(l.w), grad.unsafe_offset(l.w), grad.unsafe_offset(l.b),
                nxt, False,
            )
            relu_backward(nxt, x, R * l.n_in)
            cur_is_t = not cur_is_t
            j -= 1
        # embedding bag scatter
        var dpre0 = g_t
        if not cur_is_t:
            dpre0 = g_a
        var dwt = grad.unsafe_offset(self.lay.emb_w)
        var P = len(self.spec.obs_perms)
        for r in range(R):
            var row = lo + r
            var p = Int(sb.perm[row])
            if p >= P:
                abort("qrl.nn.BasicNet.backward_chunk: row " + String(row) + " has twist " + String(p) + " but the net has " + String(P) + " symmetries")
            var src = dpre0.unsafe_offset(r * E)
            var q = Int(sb.off[row])
            var qe = Int(sb.off[row + 1])
            while q < qe:
                var i = Int(sb.idx[q])
                q += 1
                if i < 0 or i >= OBS:
                    continue
                if p >= 0:
                    i = self.spec.obs_perms[p][i]
                add_inplace(dwt.unsafe_offset(i * E), src, E)
        col_sum_acc(dpre0, R, E, grad.unsafe_offset(self.lay.emb_b))
        _ = len(self.ps.data)
        _ = len(cc.acts)
        _ = len(cc.gbuf)
        _ = len(cc.draw)
