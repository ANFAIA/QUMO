"""Synthesis PPO loss with explicit gradient, log-softmax helpers and reference-form GAE (MOJO_PORT.md 4.5).

Per row (post-twist logits `z` of length A, **unmasked**, value `v`, action `a`, advantage `adv`, return `ret`, recorded
`old_lp`), batch size N, constants `c = clip`, `vf`, `ent`:

    lp_all = log_softmax(z) ; p = exp(lp_all) ; lp = lp_all[a] ; H = -sum p * lp_all
    ratio  = exp(lp - old_lp) ; s1 = ratio * adv ; s2 = clamp(ratio, 1-c, 1+c) * adv
    policy = -(1/N) sum min(s1, s2) ; value = (1/N) sum (v - ret)^2 ; entropy = (1/N) sum H
    loss   = policy + vf * value - ent * entropy

    g_ratio = adv                       if s1 < s2
            = adv * [1-c <= ratio <= 1+c]   if s2 < s1
            = adv                       if s1 == s2          (the reference splits the tie 50/50: the total is adv)
    dL/dlp  = -(1/N) * g_ratio * ratio
    dz_j    = dL/dlp * ([j == a] - p_j) + (ent / N) * p_j * (lp_all_j + H)
    dv      = vf * 2 * (v - ret) / N

`old_lp` is `l_a - logsumexp(l)` of the recorded *masked* logits in f32 (`old_logp_row`); on an all-masked row
(`-1e10` everywhere) it is exactly 0.0, not `-ln A` (MOJO_PORT.md 18, C-1).
"""

from std.math import exp, log
from std.os import abort
from qrl.tensor import FPtr
from qrl.rlrng import mulf

comptime IPtr = MutPointer[Int32, MutAnyOrigin]  # action ids as stored in `Rollout.acts`


@fieldwise_init
struct PPOLossCfg(Copyable):
    var clip: Float32
    var vf_coef: Float32
    var ent_coef: Float32


@fieldwise_init
struct LossParts(Copyable):
    """Partial sums, not yet divided by N: `policy = -sum min(s1, s2)`, `value = sum (v - ret)^2`,
    `entropy = sum H` (each accumulated in f64 from the per-row f32 values)."""

    var policy: Float64
    var value: Float64
    var entropy: Float64


def _row_max(z: FPtr, n: Int) -> Float32:
    var m = z[unsafe_offset=0]
    for i in range(1, n):
        var v = z[unsafe_offset=i]
        if v > m:
            m = v
    return m


def log_softmax_row(z: FPtr, n: Int, out_lp: FPtr):
    """`out_lp[i] = (z[i] - max) - ln(sum exp(z - max))` in f32 (the reference `log_softmax` form). `out_lp` may alias `z`."""
    if n <= 0:
        return
    var m = _row_max(z, n)
    var s: Float32 = 0.0
    for i in range(n):
        s += exp(z[unsafe_offset=i] - m)
    var ls = log(s)
    for i in range(n):
        out_lp[unsafe_offset=i] = (z[unsafe_offset=i] - m) - ls


def logsumexp_row(z: FPtr, n: Int) -> Float32:
    """`max + ln(sum exp(z - max))` in f32 (the form `Categorical(logits=...)` normalises with)."""
    if n <= 0:
        return 0.0
    var m = _row_max(z, n)
    var s: Float32 = 0.0
    for i in range(n):
        s += exp(z[unsafe_offset=i] - m)
    return m + log(s)


def old_logp_row(z: FPtr, n: Int, a: Int) -> Float32:
    """Recorded log-probability of action `a` from the stored masked logits: `z[a] - logsumexp(z)`. Exactly 0.0 on
    an all-masked row (the f32 spacing at 1e10 absorbs `ln n`)."""
    return z[unsafe_offset=a] - logsumexp_row(z, n)


def normalize_advantages(advs: FPtr, n: Int):
    """`advs = (advs - mean) / (std + 1e-8)` with the unbiased (`n - 1`) standard deviation, as the reference `.std()`
    (`PPOConfig.normalize_advantage`, off in QUMO). A batch of fewer than two rows is only centred."""
    if n <= 0:
        return
    var s: Float64 = 0.0
    for i in range(n):
        s += Float64(advs[unsafe_offset=i])
    var mean = s / Float64(n)
    var ss: Float64 = 0.0
    for i in range(n):
        var d = Float64(advs[unsafe_offset=i]) - mean
        ss += d * d
    var denom: Float64 = 1.0
    if n > 1:
        denom = Float64(n - 1)
    var std = (ss / denom) ** 0.5
    var meanf = Float32(mean)
    var den = Float32(std + 1e-8)
    for i in range(n):
        advs[unsafe_offset=i] = (advs[unsafe_offset=i] - meanf) / den


def ppo_chunk_loss(
    logits: FPtr,
    value: FPtr,
    rows: Int,
    A: Int,
    acts: IPtr,
    advs: FPtr,
    rets: FPtr,
    old_lp: FPtr,
    cfg: PPOLossCfg,
    inv_n: Float32,
    dz: FPtr,
    dv: FPtr,
) -> LossParts:
    """Loss partial sums and gradients for `rows` consecutive rows of a batch of N rows (`inv_n = 1/N` as f32).

    `logits` is `rows x A` (post-twist, unmasked), `value`, `advs`, `rets`, `old_lp` have `rows` entries, `acts` the
    action ids. The gradients land in `dz` (`rows x A`, may not alias `logits`) and `dv` (`rows`), both already scaled
    by `1/N`, ready for `BasicNet.backward_chunk`. Nothing is written beyond `dz[0 : rows * A]` and `dv[0 : rows]`."""
    var lo_c = Float32(1.0) - cfg.clip
    var hi_c = Float32(1.0) + cfg.clip
    var pol: Float64 = 0.0
    var val: Float64 = 0.0
    var ent: Float64 = 0.0
    var ent_n = cfg.ent_coef * inv_n
    var vf2 = cfg.vf_coef * 2.0
    for r in range(rows):
        var z = logits.unsafe_offset(r * A)
        var g = dz.unsafe_offset(r * A)
        var a = Int(acts[unsafe_offset=r])
        if a < 0 or a >= A:
            abort("qrl.loss.ppo_chunk_loss: action " + String(a) + " of row " + String(r) + " is outside 0.." + String(A - 1))
        # lp_all lives in the dz row until the gradient overwrites it
        log_softmax_row(z, A, g)
        var h: Float32 = 0.0
        for j in range(A):
            var lj = g[unsafe_offset=j]
            h -= exp(lj) * lj
        var lp = g[unsafe_offset=a]
        var ratio = exp(lp - old_lp[unsafe_offset=r])
        var adv = advs[unsafe_offset=r]
        var s1 = ratio * adv
        var clamped = ratio
        if clamped < lo_c:
            clamped = lo_c
        if clamped > hi_c:
            clamped = hi_c
        var s2 = clamped * adv
        var g_ratio = adv
        var m = s1
        if s2 < s1:
            m = s2
            if not (ratio >= lo_c and ratio <= hi_c):
                g_ratio = 0.0
        pol -= Float64(m)
        ent += Float64(h)
        var d = value[unsafe_offset=r] - rets[unsafe_offset=r]
        val += Float64(d * d)
        dv[unsafe_offset=r] = vf2 * d * inv_n
        var dlp = -(inv_n * g_ratio * ratio)
        for j in range(A):
            var lj = g[unsafe_offset=j]
            var pj = exp(lj)
            var onehot: Float32 = 0.0
            if j == a:
                onehot = 1.0
            g[unsafe_offset=j] = dlp * (onehot - pj) + ent_n * pj * (lj + h)
    return LossParts(pol, val, ent)


def ppo_chunk_loss(
    logits: FPtr,
    value: FPtr,
    rows: Int,
    A: Int,
    acts: FPtr,
    advs: FPtr,
    rets: FPtr,
    old_lp: FPtr,
    cfg: PPOLossCfg,
    inv_n: Float32,
    dz: FPtr,
    dv: FPtr,
) -> LossParts:
    """Overload for the spelling of MOJO_PORT.md section 5, where the action ids are passed as f32 values (exact for
    ids below 2^24). Converts to an `Int32` copy and forwards."""
    var ids = List[Int32](length=max(rows, 1), fill=0)
    for r in range(rows):
        ids[r] = Int32(Int(acts[unsafe_offset=r]))
    var p = ids.unsafe_ptr().as_unsafe_any_origin()
    var out = ppo_chunk_loss(logits, value, rows, A, p, advs, rets, old_lp, cfg, inv_n, dz, dv)
    _ = len(ids)
    return out^


def gae_twisterl_ptr(n: Int, rewards: FPtr, values: FPtr, gamma: Float32, lam: Float32, advs: FPtr, rets: FPtr):
    """Raw-pointer form of `gae_twisterl` for an episode of `n` recorded states (writes `advs[0:n]`, `rets[0:n]`)."""
    if n <= 0:
        return
    var last = n - 1
    advs[unsafe_offset=last] = rewards[unsafe_offset=last] - values[unsafe_offset=last]
    rets[unsafe_offset=last] = rewards[unsafe_offset=last]
    var t = n - 2
    while t >= 0:
        # Every product goes through `mulf` and every sum is its own statement (no FMA contraction, rule 9).
        var inner = values[unsafe_offset = t + 1] + mulf(lam, advs[unsafe_offset = t + 1])
        var scaled = mulf(gamma, inner)
        var ret = rewards[unsafe_offset=t] + scaled
        rets[unsafe_offset=t] = ret
        advs[unsafe_offset=t] = ret - values[unsafe_offset=t]
        t -= 1


def gae_twisterl(
    rewards: Span[Float32, _],
    values: Span[Float32, _],
    gamma: Float32,
    lam: Float32,
    mut advs: List[Float32],
    mut rets: List[Float32],
):
    """Reference-trainer GAE of one episode (f32): `advs[n-1] = r[n-1] - v[n-1]; rets[n-1] = r[n-1]`, then for `t = n-2 .. 0`:
    `rets[t] = r[t] + gamma * (v[t+1] + lam * advs[t+1])`, `advs[t] = rets[t] - v[t]`. The reward of row `t` is the
    reward of state `t` itself and the terminal row bootstraps from nothing. `advs` and `rets` are replaced (resized
    to `n`)."""
    var n = len(rewards)
    advs = List[Float32](length=n, fill=0.0)
    rets = List[Float32](length=n, fill=0.0)
    if n == 0:
        return
    var rp = rewards.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin()
    var vp = values.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin()
    var ap = advs.unsafe_ptr().as_unsafe_any_origin()
    var tp = rets.unsafe_ptr().as_unsafe_any_origin()
    gae_twisterl_ptr(n, rp, vp, gamma, lam, ap, tp)
    _ = len(advs)
    _ = len(rets)
