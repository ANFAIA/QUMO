"""Router losses with hand-derived gradients: behaviour-cloning soft cross-entropy, the route-PPO loss and the
float64 route GAE (MOJO_PORT.md 8.1 and 8.3 and the numerics spec in rl/spec).

Every kernel works on one chunk of rows (row stride `slots`, 16 for the router) and is single-threaded; callers
cut a batch into fixed chunks, so the sums do not depend on the thread count. Masked slots never enter a sum and
receive a gradient of exactly zero (the reference fills them with -1e9 before the softmax, which gives probability 0
and gradient 0; skipping them is the same and avoids multiplying 0 by -1e9).

`bc_soft_ce_chunk` (final signature; the W-1 stub lacked `inv_b`, the reciprocal of the batch size B, because the
gradient is divided by the whole batch while a chunk only sees some of its rows):

    target = softmax(-(Q - min_valid Q) / tau)         over the valid slots, constant (no gradient)
    loss_b = - sum_k target_k * log_softmax(z)_k        loss = mean_b loss_b
    dz_k   = (p_k * sum_k' target_k' - target_k) / B    for valid k, 0 otherwise
"""

from std.math import exp, log
from qrl.tensor import FPtr
from qrl.router_net import SPtr

comptime IPtr = MutPointer[Int32, MutAnyOrigin]


def gae_route(
    rewards: Span[Float64, _],
    values: Span[Float64, _],
    done: Span[Bool, _],
    gamma: Float64,
    lam: Float64,
    mut adv: List[Float64],
    mut ret: List[Float64],
):
    """Float64 router GAE over concatenated trajectories; `done[i]` marks the last step of a trajectory (8.3):
    `nv = 0 if done[i] else values[i+1]`, `delta = r[i] + gamma*nv - v[i]`, `last = delta + gamma*lam*(0 if done[i]
    else last)`, `adv[i] = last`, `ret = adv + v` (un-normalised advantage). The last element is always treated as
    the end of a trajectory (there is no `values[n]` to bootstrap from)."""
    var n = len(rewards)
    adv = List[Float64](length=n, fill=0.0)
    ret = List[Float64](length=n, fill=0.0)
    var last = 0.0
    var i = n - 1
    while i >= 0:
        var end = done[i] or i == n - 1
        var nv = 0.0 if end else values[i + 1]
        var delta = rewards[i] + gamma * nv - values[i]
        last = delta + gamma * lam * (0.0 if end else last)
        adv[i] = last
        ret[i] = last + values[i]
        i -= 1


def bc_soft_ce_chunk(
    logits: FPtr,
    q: FPtr,
    valid: SPtr,
    rows: Int,
    slots: Int,
    tau: Float32,
    inv_b: Float32,
    dz: FPtr,
) -> Float64:
    """Soft-target cross entropy over the valid slots of each row. `logits` and `q` are [rows, slots] (q already
    clipped and slot-permuted), `valid` is [rows, slots] with 1 / 0. Writes the gradient w.r.t. the logits (scaled by
    `inv_b`) into `dz` (masked slots 0) and returns the SUM of the per-row losses (the caller divides by the batch
    size for the loss and by N for the epoch mean). Arithmetic is f32 like the reference; the sum is f64."""
    var total = 0.0
    for r in range(rows):
        var base = r * slots
        var mx: Float32 = 0.0
        var qmin: Float32 = 0.0
        var nv = 0
        for k in range(slots):
            if valid[unsafe_offset = base + k] != 0:
                var zk = logits[unsafe_offset = base + k]
                var qk = q[unsafe_offset = base + k]
                if nv == 0:
                    mx = zk
                    qmin = qk
                else:
                    if zk > mx:
                        mx = zk
                    if qk < qmin:
                        qmin = qk
                nv += 1
        if nv == 0:
            for k in range(slots):
                dz[unsafe_offset = base + k] = 0.0
            continue
        var se: Float32 = 0.0
        var tsum: Float32 = 0.0
        for k in range(slots):
            if valid[unsafe_offset = base + k] != 0:
                se += exp(logits[unsafe_offset = base + k] - mx)
                tsum += exp(-(q[unsafe_offset = base + k] - qmin) / tau)
        var logse = log(se)
        var tt: Float32 = 0.0  # sum of the normalised targets (1 up to rounding; the reference autograd uses it as is)
        var row_loss: Float32 = 0.0
        for k in range(slots):
            if valid[unsafe_offset = base + k] != 0:
                var t = exp(-(q[unsafe_offset = base + k] - qmin) / tau) / tsum
                var lp = (logits[unsafe_offset = base + k] - mx) - logse
                tt += t
                row_loss -= t * lp
        for k in range(slots):
            if valid[unsafe_offset = base + k] != 0:
                var t = exp(-(q[unsafe_offset = base + k] - qmin) / tau) / tsum
                var lp = (logits[unsafe_offset = base + k] - mx) - logse
                var p = exp(lp)
                dz[unsafe_offset = base + k] = (p * tt - t) * inv_b
            else:
                dz[unsafe_offset = base + k] = 0.0
        total += Float64(row_loss)
    return total


@fieldwise_init
struct RoutePPOCfg(Copyable):
    var clip: Float32
    var vf_coef: Float32
    var ent_coef: Float32

    def __init__(out self):
        """The router PPO constants of MOJO_PORT.md 19.1 (clip 0.1, vf 0.5, ent 0.005)."""
        self.clip = 0.1
        self.vf_coef = 0.5
        self.ent_coef = 0.005


@fieldwise_init
struct RouteLossParts(Copyable):
    """Partial sums over a chunk, not yet divided by the batch size: `policy = -sum min(s1, s2)`,
    `value = sum (v - ret)^2`, `entropy = sum H_b`."""

    var policy: Float64
    var value: Float64
    var entropy: Float64


def route_ppo_chunk_loss(
    logits: FPtr,
    values: FPtr,
    valid: SPtr,
    rows: Int,
    slots: Int,
    acts: IPtr,
    logp_old: FPtr,
    adv: FPtr,
    ret: FPtr,
    cfg: RoutePPOCfg,
    inv_b: Float32,
    dz: FPtr,
    dv: FPtr,
    ratio_out: FPtr,
) -> RouteLossParts:
    """Route PPO loss of one chunk (numerics spec, route PPO). `logits` [rows, slots] raw, `values` [rows], `valid` [rows, slots],
    `acts` the sampled slot, `logp_old` the recorded log-probability (f32 of the f64 rollout value), `adv` the
    standardised advantage, `ret` the return. Writes `dz` ([rows, slots], masked slots 0) and `dv` ([rows]) scaled by
    `inv_b`, and the ratios into `ratio_out` (may point at a scratch of `rows` floats).

        lp = log_softmax(masked z) ; ratio = exp(lp[a] - logp_old) ; s1 = ratio adv ; s2 = clamp(ratio, 1-c, 1+c) adv
        pl = -mean min(s1, s2) ; vl = mean (v - ret)^2 ; ent = -mean sum p lp ; loss = pl + vf vl - ent_coef ent
        live = (adv >= 0) ? ratio <= 1+c : ratio >= 1-c                      (the min / clamp gradient rule)
        dnlp = live ? -adv ratio / B : 0
        dz_k = ([k == a] - p_k) dnlp + ent_coef / B p_k (lp_k + H_b) ;  dv = vf 2 (v - ret) / B
    """
    var pol = 0.0
    var val = 0.0
    var ent_sum = 0.0
    var lo = 1.0 - cfg.clip
    var hi = 1.0 + cfg.clip
    for r in range(rows):
        var base = r * slots
        var a = Int(acts[unsafe_offset=r])
        var vv = values[unsafe_offset=r]
        var rt = ret[unsafe_offset=r]
        var dvr = cfg.vf_coef * 2.0 * (vv - rt) * inv_b
        dv[unsafe_offset=r] = dvr
        val += Float64((vv - rt) * (vv - rt))
        var ok = a >= 0 and a < slots
        if ok:
            ok = valid[unsafe_offset = base + a] != 0
        if not ok:
            for k in range(slots):
                dz[unsafe_offset = base + k] = 0.0
            ratio_out[unsafe_offset=r] = 0.0
            continue
        var mx: Float32 = 0.0
        var first = True
        for k in range(slots):
            if valid[unsafe_offset = base + k] != 0:
                var zk = logits[unsafe_offset = base + k]
                if first or zk > mx:
                    mx = zk
                first = False
        var se: Float32 = 0.0
        for k in range(slots):
            if valid[unsafe_offset = base + k] != 0:
                se += exp(logits[unsafe_offset = base + k] - mx)
        var logse = log(se)
        var ent_b: Float32 = 0.0
        for k in range(slots):
            if valid[unsafe_offset = base + k] != 0:
                var lp = (logits[unsafe_offset = base + k] - mx) - logse
                ent_b -= exp(lp) * lp
        var nlp = (logits[unsafe_offset = base + a] - mx) - logse
        var ratio = exp(nlp - logp_old[unsafe_offset=r])
        ratio_out[unsafe_offset=r] = ratio
        var ad = adv[unsafe_offset=r]
        var s1 = ratio * ad
        var clamped = ratio
        if clamped < lo:
            clamped = lo
        if clamped > hi:
            clamped = hi
        var s2 = clamped * ad
        pol -= Float64(s1 if s1 < s2 else s2)
        ent_sum += Float64(ent_b)
        var live = (ratio <= hi) if ad >= 0.0 else (ratio >= lo)
        var dnlp: Float32 = 0.0
        if live:
            dnlp = -ad * inv_b * ratio
        for k in range(slots):
            if valid[unsafe_offset = base + k] != 0:
                var lp = (logits[unsafe_offset = base + k] - mx) - logse
                var p = exp(lp)
                var g = ((Float32(1.0) if k == a else Float32(0.0)) - p) * dnlp
                g += cfg.ent_coef * inv_b * p * (lp + ent_b)
                dz[unsafe_offset = base + k] = g
            else:
                dz[unsafe_offset = base + k] = 0.0
    return RouteLossParts(pol, val, ent_sum)
