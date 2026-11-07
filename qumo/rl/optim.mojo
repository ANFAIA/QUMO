"""Adam, gradient-norm clipping and the cosine learning-rate schedule (MOJO_PORT.md section 4.3).

`Adam` is the torch single-tensor CPU path over one flat parameter buffer (torch.md 4):

    t += 1                                      # one counter per optimizer, never reset
    m  = m + (g - m) * (1 - beta1)              # lerp form, f32
    v  = v * beta2 + (1 - beta2) * g * g        # f32
    bc1 = 1 - beta1^t ; bc2 = 1 - beta2^t       # f64
    step_size = lr / bc1 ; denom = sqrt(v) / sqrt(bc2) + eps        # eps added AFTER the division
    p  = p - step_size * m / denom              # scalars computed in f64, cast to f32 per operation

The update is dense (momentum on rows with a zero gradient still moves them). Two exact refinements exist:
frozen slices (`set_frozen`: torch's `grad is None`, no step and no state) and sparse row regions
(`mark_sparse_rows`: rows that never received a non-zero gradient have m = v = 0 and update by exactly 0, so they are
skipped; the result is bit-identical to the dense update).
"""

from std.math import cos, fma, pi, pow, sqrt
from std.os import abort
from qumo.rl.tensor import FPtr, sumsq_f64

comptime _F32 = DType.float32
comptime _CPtr = Pointer[Float32, ImmutAnyOrigin]  # read-only gradient pointer
comptime _KIND_FROZEN = 1
comptime _KIND_SPARSE = 2


@always_inline
def _adam_run[
    W: Int
](
    p: FPtr,
    g: _CPtr,
    m: FPtr,
    v: FPtr,
    lo: Int,
    hi: Int,
    b1c: Float32,
    b2f: Float32,
    b2c: Float32,
    bc2s: Float32,
    epsf: Float32,
    neg_step: Float32,
) -> Int:
    """Updates elements `lo, lo + W, ...` while a full vector of W fits below `hi`; returns the first element not
    processed. W = 8 handles the bulk and W = 1 (a one-lane vector, the same operations) the tail, so every element is
    updated by the identical sequence of IEEE operations whichever path it takes."""
    var vb1c = SIMD[_F32, W](b1c)
    var vb2f = SIMD[_F32, W](b2f)
    var vb2c = SIMD[_F32, W](b2c)
    var vbc2s = SIMD[_F32, W](bc2s)
    var veps = SIMD[_F32, W](epsf)
    var vneg = SIMD[_F32, W](neg_step)
    var i = lo
    while i + W <= hi:
        var gv = g.unsafe_load[width=W](i)
        var mv = m.unsafe_load[width=W](i)
        var vv = v.unsafe_load[width=W](i)
        mv = fma(gv - mv, vb1c, mv)
        vv = fma(vb2c * gv, gv, vv * vb2f)
        var denom = sqrt(vv) / vbc2s + veps
        var q = mv / denom
        m.unsafe_store(i, mv)
        v.unsafe_store(i, vv)
        p.unsafe_store(i, fma(vneg, q, p.unsafe_load[width=W](i)))
        i += W
    return i


@always_inline
def _adam_range(
    p: FPtr,
    g: _CPtr,
    m: FPtr,
    v: FPtr,
    lo: Int,
    hi: Int,
    b1c: Float32,
    b2f: Float32,
    b2c: Float32,
    bc2s: Float32,
    epsf: Float32,
    neg_step: Float32,
):
    var i = _adam_run[8](p, g, m, v, lo, hi, b1c, b2f, b2c, bc2s, epsf, neg_step)
    _ = _adam_run[1](p, g, m, v, i, hi, b1c, b2f, b2c, bc2s, epsf, neg_step)


struct Adam(Movable):
    var n_params: Int
    var lr: Float64
    var beta1: Float64
    var beta2: Float64
    var eps: Float64
    var t: Int
    var m: List[Float32]
    var v: List[Float32]
    # Special ranges, sorted by offset and disjoint: frozen slices and sparse row regions.
    var _r_off: List[Int]
    var _r_len: List[Int]
    var _r_kind: List[Int]
    var _r_row: List[Int]  # row length (sparse regions)
    var _r_tbase: List[Int]  # index of the region's first row in `_touched`
    var _touched: List[Bool]

    def __init__(
        out self, n_params: Int, lr: Float64, beta1: Float64 = 0.9, beta2: Float64 = 0.999, eps: Float64 = 1e-8
    ):
        self.n_params = n_params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = List[Float32](length=n_params, fill=0.0)
        self.v = List[Float32](length=n_params, fill=0.0)
        self._r_off = List[Int]()
        self._r_len = List[Int]()
        self._r_kind = List[Int]()
        self._r_row = List[Int]()
        self._r_tbase = List[Int]()
        self._touched = List[Bool]()

    def _add_range(mut self, off: Int, length: Int, kind: Int, row: Int, rows: Int, what: String):
        """Inserts a disjoint range keeping the ranges sorted by offset; aborts on a malformed or overlapping one."""
        if off < 0 or length < 0 or off + length > self.n_params:
            abort("qumo.rl.optim.Adam." + what + ": range [" + String(off) + ", " + String(off + length) + ") outside the " + String(self.n_params) + " parameters")
        if length == 0:
            return
        var pos = 0
        while pos < len(self._r_off) and self._r_off[pos] < off:
            pos += 1
        if pos > 0 and self._r_off[pos - 1] + self._r_len[pos - 1] > off:
            abort("qumo.rl.optim.Adam." + what + ": range overlaps an earlier frozen / sparse range")
        if pos < len(self._r_off) and off + length > self._r_off[pos]:
            abort("qumo.rl.optim.Adam." + what + ": range overlaps a later frozen / sparse range")
        var tbase = len(self._touched)
        if kind == _KIND_SPARSE:
            for _ in range(rows):
                self._touched.append(False)
        self._r_off.insert(pos, off)
        self._r_len.insert(pos, length)
        self._r_kind.insert(pos, kind)
        self._r_row.insert(pos, row)
        self._r_tbase.insert(pos, tbase)

    def set_frozen(mut self, off: Int, length: Int):
        """Grad is None for the slice: no step and no state (m, v and the parameters stay untouched). Frozen slices
        must not overlap each other or a sparse region."""
        self._add_range(off, length, _KIND_FROZEN, 0, 0, "set_frozen")

    def mark_sparse_rows(mut self, off: Int, row_len: Int, rows: Int):
        """Declares `rows` rows of `row_len` parameters starting at `off` as a region whose never-touched rows may be
        skipped. A row counts as touched from the first step in which its gradient has a non-zero (or NaN) entry; before
        that m = v = 0 and the dense update is exactly 0, so skipping it is bit-exact."""
        if row_len <= 0 or rows < 0:
            abort("qumo.rl.optim.Adam.mark_sparse_rows: row_len must be positive and rows non-negative")
        self._add_range(off, row_len * rows, _KIND_SPARSE, row_len, rows, "mark_sparse_rows")

    def step(mut self, mut params: List[Float32], grads: List[Float32], lr: Float64):
        """One Adam step with learning rate `lr` (the scheduled value; `self.lr` is only the construction default)."""
        if len(params) != self.n_params or len(grads) != self.n_params:
            abort(
                "qumo.rl.optim.Adam.step: params / grads length " + String(len(params)) + " / " + String(len(grads))
                + " does not match the optimizer's " + String(self.n_params)
            )
        self.t += 1
        var tf = Float64(self.t)
        var bc1 = 1.0 - pow(self.beta1, tf)
        var bc2 = 1.0 - pow(self.beta2, tf)
        var b1c = Float32(1.0 - self.beta1)
        var b2f = Float32(self.beta2)
        var b2c = Float32(1.0 - self.beta2)
        var bc2s = Float32(sqrt(bc2))
        var epsf = Float32(self.eps)
        var neg_step = Float32(-(lr / bc1))
        var pp = params.unsafe_ptr().as_unsafe_any_origin()
        var pg = grads.unsafe_ptr().as_unsafe_any_origin()
        var pm = self.m.unsafe_ptr().as_unsafe_any_origin()
        var pv = self.v.unsafe_ptr().as_unsafe_any_origin()
        var pos = 0
        for r in range(len(self._r_off)):
            var off = self._r_off[r]
            _adam_range(pp, pg, pm, pv, pos, off, b1c, b2f, b2c, bc2s, epsf, neg_step)
            if self._r_kind[r] == _KIND_SPARSE:
                var rl = self._r_row[r]
                var nrows = self._r_len[r] // rl
                var tb = self._r_tbase[r]
                for row in range(nrows):
                    var lo = off + row * rl
                    if not self._touched[tb + row]:
                        var any = False
                        for q in range(rl):
                            if grads[lo + q] != 0.0:
                                any = True
                                break
                        if not any:
                            continue
                        self._touched[tb + row] = True
                    _adam_range(pp, pg, pm, pv, lo, lo + rl, b1c, b2f, b2c, bc2s, epsf, neg_step)
            pos = off + self._r_len[r]
        _adam_range(pp, pg, pm, pv, pos, self.n_params, b1c, b2f, b2c, bc2s, epsf, neg_step)
        # The raw pointers above must not outlive the lists (ASAP destruction).
        _ = len(params)
        _ = len(grads)
        _ = len(self.m)
        _ = len(self.v)

    def steps_taken(self) -> Int:
        return self.t

    def load_state(mut self, t: Int, m: List[Float32], v: List[Float32]) raises:
        """Restores the optimizer state of a checkpoint (step counter and both moments). The per-row "touched" flags of
        sparse regions are recomputed from the moments (a row is touched iff any m or v entry is non-zero), which
        keeps a resumed run bit-identical to an uninterrupted one."""
        if len(m) != self.n_params or len(v) != self.n_params:
            raise Error("Adam.load_state: moment length " + String(len(m)) + " / " + String(len(v)) + " does not match " + String(self.n_params))
        if t < 0:
            raise Error("Adam.load_state: negative step counter")
        self.t = t
        for i in range(self.n_params):
            self.m[i] = m[i]
            self.v[i] = v[i]
        for r in range(len(self._r_off)):
            if self._r_kind[r] != _KIND_SPARSE:
                continue
            var off = self._r_off[r]
            var rl = self._r_row[r]
            var nrows = self._r_len[r] // rl
            var tb = self._r_tbase[r]
            for row in range(nrows):
                var any = False
                for q in range(rl):
                    if self.m[off + row * rl + q] != 0.0 or self.v[off + row * rl + q] != 0.0:
                        any = True
                        break
                self._touched[tb + row] = any

    def touched_rows(self) -> Int:
        """Number of sparse-region rows that have received a non-zero gradient so far."""
        var c = 0
        for i in range(len(self._touched)):
            if self._touched[i]:
                c += 1
        return c


def clip_grad_norm(mut grads: List[Float32], ranges: List[Int], max_norm: Float64) -> Float64:
    """torch `clip_grad_norm_`: one global L2 norm over several tensors, then every gradient is multiplied by
    `coef = min(max_norm / (total + 1e-6), 1)` unconditionally (also when coef is 1.0). Returns the pre-clip norm.

    `ranges` lists the tensors as consecutive (offset, length) pairs: `[off0, len0, off1, len1, ...]`; an empty list
    means the whole buffer is one tensor. As in torch the per-tensor norms are f32 values and the total is the f32
    norm of that vector of norms (accumulated here in f64 before the final rounding)."""
    var p = grads.unsafe_ptr().as_unsafe_any_origin()
    var total_sq: Float64 = 0.0
    var n_ranges = len(ranges) // 2
    if len(ranges) == 0:
        var nrm = Float32(sqrt(sumsq_f64(p, len(grads))))
        total_sq = Float64(nrm) * Float64(nrm)
    else:
        for r in range(n_ranges):
            var off = ranges[2 * r]
            var ln = ranges[2 * r + 1]
            if off < 0 or ln < 0 or off + ln > len(grads):
                abort("qumo.rl.optim.clip_grad_norm: range outside the gradient buffer")
            var nrm = Float32(sqrt(sumsq_f64(p.unsafe_offset(off), ln)))
            total_sq += Float64(nrm) * Float64(nrm)
    var total = Float32(sqrt(total_sq))
    var coef64 = max_norm / (Float64(total) + 1e-6)
    if coef64 > 1.0:
        coef64 = 1.0
    var coef = Float32(coef64)
    if len(ranges) == 0:
        for i in range(len(grads)):
            grads[i] = grads[i] * coef
    else:
        for r in range(n_ranges):
            var off = ranges[2 * r]
            for i in range(off, off + ranges[2 * r + 1]):
                grads[i] = grads[i] * coef
    return Float64(total)


def cosine_epoch_lr(base: Float64, epoch: Int, total_epochs: Int) -> Float64:
    """`base * (1 + cos(pi * epoch / total_epochs)) / 2`: torch CosineAnnealingLR (eta_min 0, stepped once per epoch)
    in closed form; `epoch` is clamped to [0, total_epochs]. Same closed form as `qml_train.cosine_lr(epoch, total,
    base)`, kept here so the RL stack does not import the quantum-ML simulator stack."""
    if total_epochs <= 0:
        return base
    var e = min(max(epoch, 0), total_epochs)
    return base * (1.0 + cos(pi * Float64(e) / Float64(total_epochs))) / 2.0
