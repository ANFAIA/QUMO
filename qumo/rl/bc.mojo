"""Behaviour-cloning trainer for the router (MOJO_PORT.md 8.1 and the numerics spec in training/ai_transpiler/spec).

Recipe, per epoch `e` (30 epochs, lr `cosine_epoch_lr(2e-3, e, epochs)`): a fresh `randperm(N)` of the training rows;
batches of 512 (the last short batch is kept); per batch one random permutation of the 16 swap slots per sample
(observation columns `8 j + d`, the teacher costs Q and the validity mask move together; new slot j takes old slot
`perm[j]`), a soft target `softmax(-(Q - min Q) / tau)` over the valid slots, the loss
`-mean_b sum_k target * log_softmax(masked logits)`, a hand-derived backward and one Adam step. The value head is
frozen (grad is None in the reference: no step, no optimizer state). After every epoch the held-out rows are scored
without augmentation (`agree` = fraction of rows whose argmax slot has teacher cost within 1e-6 of the minimum,
`regret` = mean excess cost of the pick). A wall-clock budget stops after finishing an epoch; the schedule is not
rescaled.

Determinism (section 3): a batch is cut into fixed chunks of `BC_CHUNK_ROWS = 128` rows; chunk `c` belongs to the
virtual worker `c % BC_VW` (`BC_VW = 4`), each virtual worker owns a private gradient buffer and walks its chunks in
increasing order, the buffers are then summed in worker order. Thread count only changes the speed: results are
bit-identical for any `threads`. The permutations of a batch are drawn up front on the main thread from
`seed_for(seed, TAG_BC_SLOT, epoch, batch)`, the epoch shuffle from `seed_for(seed, TAG_BC_EPOCH, epoch)` and the
train / validation split from `seed_for(seed, TAG_BC_SPLIT)`. `BC_VW` and `BC_CHUNK_ROWS` are part of the run
configuration (recorded in the manifest as `training.vw` / `training.chunk_rows`).
"""

from std.math import sqrt
from std.time import perf_counter_ns
from std.runtime import parallelism_level
from max.algorithm import parallelize
from qumo.ai.json import json_quote, json_number, pretty_json
from qumo.compiler.policy import ModelFile
from qumo.rl.container import write_model_dir
from qumo.sim.rng import Rng
from qumo.rl.tensor import FPtr, add_inplace, fill_f32
from qumo.rl.optim import Adam, cosine_epoch_lr
from qumo.rl.rlrng import seed_for, randperm, TAG_BC_SPLIT, TAG_BC_EPOCH, TAG_BC_SLOT
from qumo.rl.teacher_data import TeacherData
from qumo.rl.router_net import RouterNet, RouterCache, SPtr, ROUTER_OBS, ROUTER_SLOTS
from qumo.rl.router_loss import bc_soft_ce_chunk, IPtr

comptime BC_VW = 4
comptime BC_CHUNK_ROWS = 128
comptime _AGREE_EPS: Float32 = 1e-6


@fieldwise_init
struct BCConfig(Copyable):
    var hidden: Int
    var tau: Float32
    var lr: Float64
    var epochs: Int
    var batch: Int
    var seed: UInt64
    var budget_s: Float64
    var threads: Int
    var verbose: Bool  # print one JSON line per epoch (the old trainer did)

    def __init__(out self):
        """The constants of MOJO_PORT.md 19.1: H 64, tau 0.5, lr 2e-3, 30 epochs, batch 512, seed 0, budget 1800 s."""
        self.hidden = 64
        self.tau = 0.5
        self.lr = 2e-3
        self.epochs = 30
        self.batch = 512
        self.seed = 0
        self.budget_s = 1800.0
        self.threads = 4
        self.verbose = False


@fieldwise_init
struct BCEpochLog(Copyable):
    var epoch: Int
    var loss: Float64
    var val_agree: Float64
    var val_regret: Float64
    var t: Float64


struct BCResult(Copyable):
    var log: List[BCEpochLog]
    var samples: Int  # training rows (after the filter and the validation split)
    var val: Int  # validation rows
    var wall_s: Float64

    def __init__(out self):
        self.log = List[BCEpochLog]()
        self.samples = 0
        self.val = 0
        self.wall_s = 0.0


def _e646(msg: String) -> Error:
    return Error("QM-E646: " + msg)


struct BCWorkspace(Movable):
    """Scratch of the BC trainer: one cache, one gradient buffer and one prepared-rows buffer per virtual worker."""

    var hidden: Int
    var params: Int
    var caches: List[RouterCache]
    var vw_grad: List[Float32]  # [BC_VW, params]
    var grad: List[Float32]  # [params] the reduced gradient of the last batch
    var p_sign: List[Int8]  # [BC_VW, CHUNK * 128] rows after the slot permutation (gather mode)
    var p_q: List[Float32]  # [BC_VW, CHUNK * 16]
    var p_valid: List[Int8]  # [BC_VW, CHUNK * 16]
    var chunk_loss: List[Float64]
    var chunk_a: List[Float64]
    var chunk_b: List[Float64]

    def __init__(out self, hidden: Int, params: Int, max_rows: Int):
        self.hidden = hidden
        self.params = params
        self.caches = List[RouterCache]()
        for _ in range(BC_VW):
            self.caches.append(RouterCache(BC_CHUNK_ROWS, hidden))
        self.vw_grad = List[Float32](length=BC_VW * params, fill=0.0)
        self.grad = List[Float32](length=params, fill=0.0)
        self.p_sign = List[Int8](length=BC_VW * BC_CHUNK_ROWS * ROUTER_OBS, fill=0)
        self.p_q = List[Float32](length=BC_VW * BC_CHUNK_ROWS * ROUTER_SLOTS, fill=0.0)
        self.p_valid = List[Int8](length=BC_VW * BC_CHUNK_ROWS * ROUTER_SLOTS, fill=0)
        var nch = (max_rows + BC_CHUNK_ROWS - 1) // BC_CHUNK_ROWS + 1
        self.chunk_loss = List[Float64](length=nch, fill=0.0)
        self.chunk_a = List[Float64](length=nch, fill=0.0)
        self.chunk_b = List[Float64](length=nch, fill=0.0)

    def ensure_chunks(mut self, rows: Int):
        var nch = (rows + BC_CHUNK_ROWS - 1) // BC_CHUNK_ROWS + 1
        if len(self.chunk_loss) < nch:
            self.chunk_loss = List[Float64](length=nch, fill=0.0)
            self.chunk_a = List[Float64](length=nch, fill=0.0)
            self.chunk_b = List[Float64](length=nch, fill=0.0)


def effective_threads(requested: Int) -> Int:
    """`min(requested, parallelism_level())`, at least 1. Speed only: results never depend on it."""
    return max(1, min(requested, parallelism_level()))


def _sign8(v: Int8) -> Int8:
    if v > 0:
        return 1
    if v < 0:
        return -1
    return 0


def permute_slots_row(
    sign_src: SPtr, q_src: FPtr, a: Int, perm: SPtr, sign_dst: SPtr, q_dst: FPtr, valid_dst: SPtr
):
    """Slot-permutation augmentation of one row: new slot `j` takes old slot `perm[j]`. The 8 observation columns of a
    slot (`8 j + d`), its cost and its validity (`perm[j] < a`) move together; `sign_src` holds signs in {-1, 0, 1}."""
    for j in range(ROUTER_SLOTS):
        var pj = Int(perm[unsafe_offset=j])
        for d in range(8):
            sign_dst[unsafe_offset = 8 * j + d] = sign_src[unsafe_offset = 8 * pj + d]
        q_dst[unsafe_offset=j] = q_src[unsafe_offset=pj]
        valid_dst[unsafe_offset=j] = 1 if pj < a else 0


def _run_batch(
    net: RouterNet,
    mut ws: BCWorkspace,
    rows: Int,
    tau: Float32,
    inv_b: Float32,
    threads: Int,
    gather: Bool,
    p_sign: SPtr,
    p_q: FPtr,
    p_valid: SPtr,
    t_sign: SPtr,
    t_q: FPtr,
    t_a: IPtr,
    order: IPtr,
    bstart: Int,
    perm: SPtr,
) -> Float64:
    """Forward, loss and backward of one batch of `rows` rows over the virtual workers; leaves the reduced gradient in
    `ws.grad` and returns the sum of the per-row losses. Prepared mode (`gather` False): `p_sign` / `p_q` / `p_valid`
    hold the whole batch after the slot permutation. Gather mode: the rows `order[bstart:bstart+rows]` of the training
    arrays `t_*` are gathered and slot-permuted by `perm` ([rows, 16], new slot j takes old slot perm[j]) inside the
    worker tasks."""
    ws.ensure_chunks(rows)
    var nchunks = (rows + BC_CHUNK_ROWS - 1) // BC_CHUNK_ROWS
    var P = ws.params
    var pg = ws.vw_grad.unsafe_ptr().as_unsafe_any_origin()
    var wsign = ws.p_sign.unsafe_ptr().as_unsafe_any_origin()
    var wq = ws.p_q.unsafe_ptr().as_unsafe_any_origin()
    var wvalid = ws.p_valid.unsafe_ptr().as_unsafe_any_origin()
    var pl = ws.chunk_loss.unsafe_ptr().as_unsafe_any_origin()

    def task(w: Int) {imm net, mut ws, imm nchunks, imm P, imm pg, imm wsign, imm wq, imm wvalid, imm pl, imm rows, imm tau, imm inv_b, imm gather, imm p_sign, imm p_q, imm p_valid, imm t_sign, imm t_q, imm t_a, imm order, imm bstart, imm perm}:
        var gw = pg.unsafe_offset(w * P)
        fill_f32(gw, P, 0.0)
        var c = w
        while c < nchunks:
            var lo = c * BC_CHUNK_ROWS
            var n = min(BC_CHUNK_ROWS, rows - lo)
            var sgn = p_sign.unsafe_offset(lo * ROUTER_OBS)
            var qq = p_q.unsafe_offset(lo * ROUTER_SLOTS)
            var vv = p_valid.unsafe_offset(lo * ROUTER_SLOTS)
            if gather:
                sgn = wsign.unsafe_offset(w * BC_CHUNK_ROWS * ROUTER_OBS)
                qq = wq.unsafe_offset(w * BC_CHUNK_ROWS * ROUTER_SLOTS)
                vv = wvalid.unsafe_offset(w * BC_CHUNK_ROWS * ROUTER_SLOTS)
                for r in range(n):
                    var src = Int(order[unsafe_offset = bstart + lo + r])
                    permute_slots_row(
                        t_sign.unsafe_offset(src * ROUTER_OBS),
                        t_q.unsafe_offset(src * ROUTER_SLOTS),
                        Int(t_a[unsafe_offset=src]),
                        perm.unsafe_offset((lo + r) * ROUTER_SLOTS),
                        sgn.unsafe_offset(r * ROUTER_OBS),
                        qq.unsafe_offset(r * ROUTER_SLOTS),
                        vv.unsafe_offset(r * ROUTER_SLOTS),
                    )
            net.forward_chunk(sgn, n, ws.caches[w])
            var zp = ws.caches[w].z.unsafe_ptr().as_unsafe_any_origin()
            var dzp = ws.caches[w].dz.unsafe_ptr().as_unsafe_any_origin()
            pl[unsafe_offset=c] = bc_soft_ce_chunk(zp, qq, vv, n, ROUTER_SLOTS, tau, inv_b, dzp)
            net.backward_chunk(sgn, n, ws.caches[w], gw)
            c += BC_VW

    parallelize(task, BC_VW, threads)
    _ = len(ws.vw_grad)
    _ = len(ws.p_sign)
    _ = len(ws.p_q)
    _ = len(ws.p_valid)
    _ = len(ws.chunk_loss)
    # reduction in fixed order: worker 0, 1, 2, 3
    var g = ws.grad.unsafe_ptr().as_unsafe_any_origin()
    fill_f32(g, P, 0.0)
    for w in range(BC_VW):
        add_inplace(g, pg.unsafe_offset(w * P), P)
    var total = 0.0
    for c in range(nchunks):
        total += ws.chunk_loss[c]
    _ = len(ws.grad)
    _ = len(ws.vw_grad)
    return total


def bc_batch_step(
    mut net: RouterNet,
    mut opt: Adam,
    mut ws: BCWorkspace,
    sign: List[Int8],
    q: List[Float32],
    valid: List[Int8],
    rows: Int,
    tau: Float32,
    lr: Float64,
    threads: Int,
) -> Float64:
    """One BC optimisation step on an already prepared batch: `sign` [rows,128] in {-1,0,1}, `q` [rows,16] (after
    slot permutation, any value in the masked slots), `valid` [rows,16] with 1 / 0. Forward, soft-CE, backward over the
    virtual workers, one Adam step with learning rate `lr`. Returns the mean loss; the gradient of the step stays in
    `ws.grad`. This is the entry the `bc_golden.json` test drives with the golden's explicit slot permutation."""
    var ps = sign.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin()
    var pq = q.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin()
    var pv = valid.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin()
    var inv_b = Float32(1.0) / Float32(rows)
    var dummy = List[Int32](length=1, fill=0)
    var pd = dummy.unsafe_ptr().as_unsafe_any_origin()
    var total = _run_batch(net, ws, rows, tau, inv_b, threads, False, ps, pq, pv, ps, pq, pd, pd, 0, pv)
    _ = len(dummy)
    _ = len(sign)
    _ = len(q)
    _ = len(valid)
    opt.step(net.params, ws.grad, lr)
    return total / Float64(rows)


def bc_evaluate(
    net: RouterNet,
    mut ws: BCWorkspace,
    sign: SPtr,
    q: FPtr,
    a: IPtr,
    n: Int,
    threads: Int,
) -> Tuple[Float64, Float64]:
    """Validation without augmentation: `pick = argmax(masked logits)` (first maximum among the valid slots `k < A`),
    `agree = mean(Q[pick] <= qmin + 1e-6f)` (the add in f32), `regret = mean(Q[pick] - qmin)`. `sign` [n,128] signs,
    `q` [n,16] clipped costs, `a` [n] valid-slot counts. Returns `(agree, regret)`."""
    ws.ensure_chunks(n)
    var nchunks = (n + BC_CHUNK_ROWS - 1) // BC_CHUNK_ROWS
    var pa = ws.chunk_a.unsafe_ptr().as_unsafe_any_origin()
    var pb = ws.chunk_b.unsafe_ptr().as_unsafe_any_origin()

    def task(w: Int) {imm net, mut ws, imm nchunks, imm pa, imm pb, imm n, imm sign, imm q, imm a}:
        var c = w
        while c < nchunks:
            var lo = c * BC_CHUNK_ROWS
            var m = min(BC_CHUNK_ROWS, n - lo)
            net.forward_chunk(sign.unsafe_offset(lo * ROUTER_OBS), m, ws.caches[w])
            var agree = 0.0
            var regret = 0.0
            for r in range(m):
                var row = lo + r
                var av = Int(a[unsafe_offset=row])
                if av <= 0:
                    continue
                var pick = 0
                var best = ws.caches[w].z[r * ROUTER_SLOTS]
                var qmin = q[unsafe_offset = row * ROUTER_SLOTS]
                for k in range(1, av):
                    var zk = ws.caches[w].z[r * ROUTER_SLOTS + k]
                    if zk > best:
                        best = zk
                        pick = k
                    var qk = q[unsafe_offset = row * ROUTER_SLOTS + k]
                    if qk < qmin:
                        qmin = qk
                var qp = q[unsafe_offset = row * ROUTER_SLOTS + pick]
                if qp <= qmin + _AGREE_EPS:
                    agree += 1.0
                regret += Float64(qp - qmin)
            pa[unsafe_offset=c] = agree
            pb[unsafe_offset=c] = regret
            c += BC_VW

    parallelize(task, BC_VW, threads)
    _ = len(ws.chunk_a)
    _ = len(ws.chunk_b)
    var sa = 0.0
    var sb = 0.0
    for c in range(nchunks):
        sa += ws.chunk_a[c]
        sb += ws.chunk_b[c]
    return (sa / Float64(max(n, 1)), sb / Float64(max(n, 1)))


def train_bc(cfg: BCConfig, data: TeacherData, init: Optional[ModelFile]) raises -> Tuple[RouterNet, BCResult]:
    """Trains a router by behaviour cloning on the (already filtered) teacher rows. `init` warm-starts every tensor
    (value head only when present). QM-E646 when there are not enough rows for the validation split
    (`nval = max(1000, N // 20)` rows are held out), QM-E647 when `init` does not match `cfg.hidden`, QM-E643 when the
    loss becomes non-finite."""
    var t0 = perf_counter_ns()
    var total = data.n
    var nval = max(1000, total // 20)
    if cfg.hidden <= 0 or cfg.hidden > 4096:
        raise Error("QM-E640: hidden must be in 1..4096, got " + String(cfg.hidden) + ". Hint: --hidden 64.")
    if cfg.epochs < 1 or cfg.batch < 1 or cfg.tau <= 0.0:
        raise Error("QM-E640: epochs and batch must be positive and tau > 0. Hint: --epochs 30 --batch 512 --tau 0.5.")
    if total <= nval:
        raise _e646(
            "only " + String(total) + " usable teacher rows after the filter; behaviour cloning holds out max(1000, N/20) = "
            + String(nval) + " rows for validation and needs more. Hint: generate more jobs with route_tool gen-jobs / gen."
        )
    var threads = effective_threads(cfg.threads)

    # ---- split: validation rows first (like the reference), then the training rows
    var split_rng = Rng(seed_for(cfg.seed, TAG_BC_SPLIT))
    var idx = randperm(split_rng, total)
    var ntr = total - nval
    var tr_sign = List[Int8](length=ntr * ROUTER_OBS, fill=0)
    var tr_q = List[Float32](length=ntr * ROUTER_SLOTS, fill=0.0)
    var tr_a = List[Int32](length=ntr, fill=0)
    var va_sign = List[Int8](length=nval * ROUTER_OBS, fill=0)
    var va_q = List[Float32](length=nval * ROUTER_SLOTS, fill=0.0)
    var va_a = List[Int32](length=nval, fill=0)
    for i in range(total):
        var src = idx[i]
        if i < nval:
            for j in range(ROUTER_OBS):
                va_sign[i * ROUTER_OBS + j] = _sign8(data.obs[src * ROUTER_OBS + j])
            for j in range(ROUTER_SLOTS):
                va_q[i * ROUTER_SLOTS + j] = data.q[src * ROUTER_SLOTS + j]
            va_a[i] = Int32(data.A[src])
        else:
            var t = i - nval
            for j in range(ROUTER_OBS):
                tr_sign[t * ROUTER_OBS + j] = _sign8(data.obs[src * ROUTER_OBS + j])
            for j in range(ROUTER_SLOTS):
                tr_q[t * ROUTER_SLOTS + j] = data.q[src * ROUTER_SLOTS + j]
            tr_a[t] = Int32(data.A[src])

    # ---- network, optimizer, workspace
    var net = RouterNet(cfg.hidden, cfg.seed)
    if init:
        var warm = RouterNet.from_model(init.value())
        if warm.hidden != cfg.hidden:
            raise Error(
                "QM-E647: cannot warm-start: the init router has hidden size " + String(warm.hidden) + " but --hidden is "
                + String(cfg.hidden) + ". Hint: pass --hidden " + String(warm.hidden) + "."
            )
        net = warm^
    var opt = Adam(net.param_count(), cfg.lr)
    opt.set_frozen(net.off_wv(), net.hidden + 1)  # the value head is not part of the BC loss: grad is None
    var ws = BCWorkspace(net.hidden, net.param_count(), max(cfg.batch, nval))

    var p_tr_sign = tr_sign.unsafe_ptr().as_unsafe_any_origin()
    var p_tr_q = tr_q.unsafe_ptr().as_unsafe_any_origin()
    var p_tr_a = tr_a.unsafe_ptr().as_unsafe_any_origin()
    var p_va_sign = va_sign.unsafe_ptr().as_unsafe_any_origin()
    var p_va_q = va_q.unsafe_ptr().as_unsafe_any_origin()
    var p_va_a = va_a.unsafe_ptr().as_unsafe_any_origin()
    var perm = List[Int8](length=cfg.batch * ROUTER_SLOTS, fill=0)
    var p_perm = perm.unsafe_ptr().as_unsafe_any_origin()
    var dummy_sign = List[Int8](length=1, fill=0)
    var dummy_q = List[Float32](length=1, fill=0.0)
    var d_sign = dummy_sign.unsafe_ptr().as_unsafe_any_origin()
    var d_q = dummy_q.unsafe_ptr().as_unsafe_any_origin()

    var res = BCResult()
    res.samples = ntr
    res.val = nval
    var nbatches = (ntr + cfg.batch - 1) // cfg.batch
    for epoch in range(cfg.epochs):
        var lr = cosine_epoch_lr(cfg.lr, epoch, cfg.epochs)
        var er = Rng(seed_for(cfg.seed, TAG_BC_EPOCH, epoch))
        var order_i = randperm(er, ntr)
        var order = List[Int32](capacity=ntr)
        for i in range(ntr):
            order.append(Int32(order_i[i]))
        var p_order = order.unsafe_ptr().as_unsafe_any_origin()
        var tot = 0.0
        for b in range(nbatches):
            var bstart = b * cfg.batch
            var rows = min(cfg.batch, ntr - bstart)
            # one uniform permutation of the 16 slots per sample, drawn up front on the main thread
            var sr = Rng(seed_for(cfg.seed, TAG_BC_SLOT, epoch, b))
            for r in range(rows):
                var base = r * ROUTER_SLOTS
                for j in range(ROUTER_SLOTS):
                    perm[base + j] = Int8(j)
                var i = ROUTER_SLOTS - 1
                while i >= 1:
                    var j = sr.below(i + 1)
                    var tmp = perm[base + i]
                    perm[base + i] = perm[base + j]
                    perm[base + j] = tmp
                    i -= 1
            var inv_b = Float32(1.0) / Float32(rows)
            var ls = _run_batch(
                net, ws, rows, cfg.tau, inv_b, threads, True, d_sign, d_q, d_sign, p_tr_sign, p_tr_q, p_tr_a, p_order, bstart, p_perm
            )
            opt.step(net.params, ws.grad, lr)
            tot += ls
        _ = len(order)
        var epoch_loss = tot / Float64(ntr)
        if not (epoch_loss == epoch_loss) or epoch_loss > 1e30:
            raise Error("QM-E643: behaviour cloning diverged (non-finite loss in epoch " + String(epoch) + "). Hint: lower --lr or check the teacher data.")
        var va = bc_evaluate(net, ws, p_va_sign, p_va_q, p_va_a, nval, threads)
        var el = Float64(perf_counter_ns() - t0) / 1e9
        res.log.append(BCEpochLog(epoch, epoch_loss, va[0], va[1], el))
        if cfg.verbose:
            print(
                '{"epoch": ' + String(epoch) + ', "loss": ' + String(epoch_loss) + ', "val_agree": ' + String(va[0])
                + ', "val_regret": ' + String(va[1]) + ', "t": ' + String(el) + "}"
            )
        if el > cfg.budget_s:
            break
    _ = len(tr_sign)
    _ = len(tr_q)
    _ = len(tr_a)
    _ = len(va_sign)
    _ = len(va_q)
    _ = len(va_a)
    _ = len(perm)
    _ = len(dummy_sign)
    _ = len(dummy_q)
    res.wall_s = Float64(perf_counter_ns() - t0) / 1e9
    return (net^, res^)


# ------------------------------------------------------------------------------------------------ run directory


def _basename(p: String) -> String:
    var s = p
    while s.byte_length() > 1 and s.endswith("/"):
        var t = String(s[byte = : s.byte_length() - 1])
        s = t^
    var bs = s.as_bytes()
    var cut = -1
    for i in range(len(bs)):
        if Int(bs[i]) == ord("/"):
            cut = i
    if cut < 0:
        return s
    return String(s[byte = cut + 1 :])


def _data_json(data: List[String]) -> String:
    var s = String("[")
    for i in range(len(data)):
        if i > 0:
            s += ","
        s += json_quote(_basename(data[i]))
    s += "]"
    return s^


def bc_train_log_json(res: BCResult, cfg: BCConfig, data: List[String], hidden: Int) -> String:
    """`train_log.json` body (compact): {samples, val, hidden, epochs_run, tau, lr, seed, data, wall_s, log[...]}, the
    schema of the old trainer (repo.md 4.3)."""
    var s = String("{")
    s += '"samples":' + String(res.samples)
    s += ',"val":' + String(res.val)
    s += ',"hidden":' + String(hidden)
    s += ',"epochs_run":' + String(len(res.log))
    s += ',"tau":' + json_number(Float64(cfg.tau))
    s += ',"lr":' + json_number(cfg.lr)
    s += ',"seed":' + String(cfg.seed)
    s += ',"data":' + _data_json(data)
    s += ',"wall_s":' + json_number(res.wall_s)
    s += ',"log":['
    for i in range(len(res.log)):
        if i > 0:
            s += ","
        s += (
            '{"epoch":' + String(res.log[i].epoch) + ',"loss":' + json_number(res.log[i].loss)
            + ',"val_agree":' + json_number(res.log[i].val_agree) + ',"val_regret":' + json_number(res.log[i].val_regret)
            + ',"t":' + json_number(res.log[i].t) + "}"
        )
    s += "]}"
    return s^


def bc_manifest_head(id: String, res: BCResult, cfg: BCConfig, data: List[String], hidden: Int, init: String) -> String:
    """The manifest members of a BC run directory that precede `tensors` (`qumo.rlroute.v0`, no test vectors)."""
    var h = String()
    h += '"format":"qumo.rlroute.v0","id":' + json_quote(id) + ',"kind":"routing","num_qubits":0'
    h += ',"obs":{"size":128,"layout":"route_sign_h8_k16","encoding":"sign(obs[d + 8k])"}'
    h += (
        ',"arch":{"type":"sign_embedding_mlp","horizon":8,"num_active_swaps":16,"hidden":' + String(hidden)
        + ',"activation":"relu","value_head":true,"first_layer_transposed":true,'
        + '"forward":"logits = W1 relu(b0 + sum_i sign(obs_i) E[i,:]) + b1; argmax, lowest index on ties, over valid active swaps"}'
    )
    h += ',"producer":"trained by the native QUMO stack (qumo.rl, Mojo): behaviour cloning on rollout-teacher soft labels"'
    h += ',"license":"Apache-2.0"'
    h += ',"training":{"engine":"qumo.rl (Mojo)","algo":"behaviour cloning + DAgger"'
    h += ',"samples":' + String(res.samples) + ',"val":' + String(res.val) + ',"hidden":' + String(hidden)
    h += ',"epochs_run":' + String(len(res.log)) + ',"epochs":' + String(cfg.epochs)
    h += ',"tau":' + json_number(Float64(cfg.tau)) + ',"lr":' + json_number(cfg.lr)
    h += ',"batch":' + String(cfg.batch) + ',"seed":' + String(cfg.seed)
    h += ',"chunk_rows":' + String(BC_CHUNK_ROWS) + ',"vw":' + String(BC_VW)
    h += ',"wall_s":' + json_number(res.wall_s)
    h += ',"init":' + json_quote(init)
    h += ',"data":' + _data_json(data) + "}"
    return h^


def write_bc_run(dir: String, net: RouterNet, res: BCResult, cfg: BCConfig, data: List[String], init: String) raises:
    """Writes a BC run directory: a valid f32 `qumo.rlroute.v0` container (`manifest.json` + `weights.bin`, atomic,
    read back and checked by `write_model_dir`) and `train_log.json`. The run id is the directory's base name."""
    _ = write_model_dir(
        dir,
        bc_manifest_head(_basename(dir), res, cfg, data, net.hidden, init),
        net.to_tensors(),
        "f32",
        List[String](),
        "",
    )
    with open(dir + "/train_log.json", "w") as f:
        f.write(pretty_json(bc_train_log_json(res, cfg, data, net.hidden), 1))
        f.write("\n")
