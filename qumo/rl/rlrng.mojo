"""Seed derivation, sampling helpers and the FMA barrier for the native RL stack (MOJO_PORT.md sections 3 and 4.1).

Every random consumer of the training stack derives its own `Rng` (xoshiro256**, `qumo.sim.rng`) from
`seed_for(run_seed, TAG, a, b, c)`, so results never depend on thread count or task scheduling. The sampling helpers
below fix the exact rules used by the collector and the evaluators:

  * `uniform_f32` / `uniform_k24`: 24-bit uniforms (`k * 2^-24`), thresholds are compared on the integer `k`;
  * `gumbel_argmax`: the twisterl sampling rule (Gumbel noise in f32, strict `>` scan seeded with element 0);
  * `sample_softmax` / `sample_probs`: inverse CDF for the sampled evaluation;
  * `mulf`: the FMA barrier (rule 9 of section 3).
"""

from std.math import exp, log
from qumo.sim.rng import Rng
from qumo.rl.tensor import FPtr

comptime TAG_INIT = 1
comptime TAG_COLLECT = 2
comptime TAG_EVAL_DET = 3
comptime TAG_EVAL_SAMPLED = 4
comptime TAG_BC_SPLIT = 5
comptime TAG_BC_EPOCH = 6
comptime TAG_BC_SLOT = 7
comptime TAG_JOBS = 8
comptime TAG_EXPORT_VEC = 9
comptime TAG_ROUTE_PPO = 10

# Odd 64-bit multipliers that spread the (tag, a, b, c) tuple before each SplitMix64 round. Odd, so that
# `v -> v * K` is a bijection on 64-bit words and distinct components never collide inside one round.
comptime _K0: UInt64 = 0x9E3779B97F4A7C15
comptime _K1: UInt64 = 0xC2B2AE3D27D4EB4F
comptime _K2: UInt64 = 0x165667B19E3779F9
comptime _K3: UInt64 = 0x27D4EB2F165667C5

comptime _INV_2_24: Float32 = 5.9604644775390625e-08  # 2^-24


def splitmix64(x: UInt64) -> UInt64:
    """The mixing function of `Rng._splitmix` as a pure function: the output of one SplitMix64 step whose state
    is `x` (the state is advanced by the golden-ratio increment before mixing). A bijection on 64-bit words."""
    var z = x + 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def seed_for(base: UInt64, tag: Int, a: Int = 0, b: Int = 0, c: Int = 0) -> UInt64:
    """Derives the seed of one random consumer from the run seed: four chained SplitMix64 rounds over
    (tag, a, b, c). Negative components are taken modulo 2^64. Deterministic and platform independent."""
    var x = splitmix64(base ^ (UInt64(tag) * _K0))
    x = splitmix64(x ^ (UInt64(a) * _K1))
    x = splitmix64(x ^ (UInt64(b) * _K2))
    x = splitmix64(x ^ (UInt64(c) * _K3))
    return x


@always_inline
def uniform_k24(mut r: Rng) -> Int:
    """Integer k in [0, 2^24): the top 24 bits of one 64-bit draw."""
    return Int(r.next_u64() >> 40)


@always_inline
def uniform_f32(mut r: Rng) -> Float32:
    """`k * 2^-24` for the integer of `uniform_k24` (exact in f32), in [0, 1 - 2^-24]."""
    return Float32(r.next_u64() >> 40) * _INV_2_24


def uniform_sym_f32(mut r: Rng, bound: Float32) -> Float32:
    """torch-style `U(-bound, bound)`: `(2 u - 1) * bound` with a 24-bit u (one draw)."""
    var u = uniform_f32(r)
    return (2.0 * u - 1.0) * bound


def normal_f32(mut r: Rng) -> Float32:
    """Standard normal as f32 (`Rng.normal`, Box-Muller, two uniforms per call)."""
    return Float32(r.normal())


def randperm(mut r: Rng, n: Int) -> List[Int]:
    """Uniform random permutation of 0..n-1: Fisher-Yates from the top (`i = n-1 .. 1`, `j = below(i + 1)`)."""
    var p = List[Int](capacity=max(n, 0))
    for i in range(n):
        p.append(i)
    var i = n - 1
    while i >= 1:
        var j = r.below(i + 1)
        var t = p[i]
        p[i] = p[j]
        p[j] = t
        i -= 1
    return p^


def gumbel_argmax(logits: FPtr, n: Int, mut r: Rng) -> Int:
    """Samples an index with probability softmax(logits) by the Gumbel-max trick, exactly as twisterl does:
    value_i = l_i - ln|ln u_i| in f32 with `u_i = uniform_f32` clamped to [2^-24, 1 - 2^-24] (one draw per element,
    in index order, whether or not the element is masked), then a strict `>` scan seeded with element 0, so the
    first maximum wins and a NaN is never chosen (unless it is element 0, which nothing can beat).

    On an all-masked row (`-1e10`, f32 spacing 1024 there) the noise is absorbed, all values tie and the result
    is index 0 (MOJO_PORT.md 4.1, C-7). `n <= 0` returns 0 without drawing."""
    if n <= 0:
        return 0
    var lo: Float32 = _INV_2_24
    var hi: Float32 = 1.0 - _INV_2_24
    var best = 0
    var best_val: Float32 = 0.0
    for i in range(n):
        var u = uniform_f32(r)
        if u < lo:
            u = lo
        if u > hi:
            u = hi
        var g = logits[unsafe_offset=i] - log(abs(log(u)))
        if i == 0:
            best_val = g
        elif g > best_val:
            best_val = g
            best = i
    return best


def sample_probs(p: FPtr, n: Int, mut r: Rng) -> Int:
    """Inverse CDF over unnormalised non-negative weights `p`: draws `u` (24 bits), scans the running sum and returns
    the first index whose running sum exceeds `u * sum(p)`. Zero-weight entries are never returned; on round-off
    (the target reaches the total) the last index with positive weight is returned (0 if there is none)."""
    if n <= 0:
        return 0
    var total: Float32 = 0.0
    for i in range(n):
        total += p[unsafe_offset=i]
    var target = uniform_f32(r) * total
    var cum: Float32 = 0.0
    var last_pos = 0
    for i in range(n):
        var w = p[unsafe_offset=i]
        if w > 0.0:
            last_pos = i
        cum += w
        if target < cum:
            return i
    return last_pos


def sample_softmax(logits: FPtr, n: Int, mut r: Rng) -> Int:
    """Samples from softmax(logits): max-shifted exponentials, then the inverse CDF of `sample_probs` (first index
    whose running sum exceeds `u * sum`; zero-probability entries are never returned). Safe for large logits (the
    shift keeps every exponent <= 0). No heap use: the exponentials are recomputed in the second pass (same values).
    If the logits admit no valid distribution (a NaN, or an infinite maximum) the single draw is still consumed and
    the first maximum is returned."""
    if n <= 0:
        return 0
    var mx = logits[unsafe_offset=0]
    var arg = 0
    for i in range(1, n):
        var v = logits[unsafe_offset=i]
        if v > mx:
            mx = v
            arg = i
    var total: Float32 = 0.0
    for i in range(n):
        total += exp(logits[unsafe_offset=i] - mx)
    var u = uniform_f32(r)
    if not (total > 0.0) or not (total < Float32.MAX):
        return arg
    var target = u * total
    var cum: Float32 = 0.0
    var last_pos = arg
    for i in range(n):
        var wi = exp(logits[unsafe_offset=i] - mx)
        if wi > 0.0:
            last_pos = i
        cum += wi
        if target < cum:
            return i
    return last_pos


@no_inline
def mulf(a: Float32, b: Float32) -> Float32:
    """`a * b` behind a call barrier. Mojo contracts `(x - y) + (a * b)` into a fused multiply-add, whereas the
    Rust env and numpy round the product first; computing each product through `mulf` and keeping the sum in a
    separate statement reproduces the unfused value (MOJO_PORT.md section 3, rule 9)."""
    return a * b


def argmax_first(values: FPtr, n: Int) -> Int:
    """Deterministic action choice: index of the first maximum (strict `>` scan seeded with element 0, so a NaN is
    never chosen unless it is element 0). Shares the tie and NaN rule of `gumbel_argmax` without drawing."""
    if n <= 0:
        return 0
    var best = 0
    var best_val = values[unsafe_offset=0]
    for i in range(1, n):
        var v = values[unsafe_offset=i]
        if v > best_val:
            best_val = v
            best = i
    return best
