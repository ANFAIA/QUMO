"""Deterministic, splittable PRNG (xoshiro256** seeded by SplitMix64).

Reproducibility is an AI-native requirement: the same (circuit, seed) must give
bit-identical results on every run so agents can diff, cache and verify.
`split(k)` derives independent streams for parallel workers.
"""

from std.math import log, log1p, sqrt, cos, pi


struct Rng(Copyable):
    var s0: UInt64
    var s1: UInt64
    var s2: UInt64
    var s3: UInt64

    def __init__(out self, seed: UInt64):
        var sm = seed
        self.s0 = Rng._splitmix(sm)
        self.s1 = Rng._splitmix(sm)
        self.s2 = Rng._splitmix(sm)
        self.s3 = Rng._splitmix(sm)

    @staticmethod
    def _splitmix(mut x: UInt64) -> UInt64:
        x += 0x9E3779B97F4A7C15
        var z = x
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB
        return z ^ (z >> 31)

    @always_inline
    @staticmethod
    def _rotl(x: UInt64, k: UInt64) -> UInt64:
        return (x << k) | (x >> (64 - k))

    @always_inline
    def next_u64(mut self) -> UInt64:
        var result = Rng._rotl(self.s1 * 5, 7) * 9
        var t = self.s1 << 17
        self.s2 ^= self.s0
        self.s3 ^= self.s1
        self.s1 ^= self.s2
        self.s0 ^= self.s3
        self.s2 ^= t
        self.s3 = Rng._rotl(self.s3, 45)
        return result

    @always_inline
    def uniform(mut self) -> Float64:
        """Uniform in [0, 1) with 53 bits of entropy."""
        return Float64(self.next_u64() >> 11) * (1.0 / 9007199254740992.0)

    @always_inline
    def below(mut self, n: Int) -> Int:
        return Int(self.next_u64() % UInt64(n))

    def normal(mut self) -> Float64:
        """Standard normal N(0, 1) by Box-Muller from two uniform() draws.

        No spare value is cached: every call consumes exactly two uniforms, so
        a stream (and every `split()` child) stays reproducible however the
        calls are interleaved.
        """
        var u1 = 1.0 - self.uniform()  # (0, 1]: log(u1) is finite
        var u2 = self.uniform()
        return sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2)

    def normal_list(mut self, n: Int, scale: Float64 = 1.0) -> List[Float64]:
        """`n` independent N(0, scale^2) draws (e.g. PennyLane-style 0.01 * randn init)."""
        var out = List[Float64](capacity=max(n, 0))
        for _ in range(n):
            out.append(scale * self.normal())
        return out^

    def split(self, stream: Int) -> Rng:
        """Independent child stream (for parallel workers / per-shot seeds)."""
        return Rng(self.s0 ^ (UInt64(stream) * 0xD1B54A32D192ED03) ^ self.s3)

    def bernoulli_mask(mut self, p: Float64) -> UInt64:
        """64 independent Bernoulli(p) bits (one per shot lane)."""
        if p <= 0.0:
            return 0
        var m: UInt64 = 0
        if p >= 0.5:
            for b in range(64):
                if self.uniform() < p:
                    m |= UInt64(1) << UInt64(b)
            return m
        # geometric skipping: O(64 p) expected work
        var ln1p = log1p_neg(p)
        var pos = 0
        while True:
            var u = self.uniform()
            var skip = Int(_log(1.0 - u) / ln1p)
            pos += skip
            if pos >= 64:
                break
            m |= UInt64(1) << UInt64(pos)
            pos += 1
        return m



@always_inline
def _log(x: Float64) -> Float64:
    return log(x)


@always_inline
def log1p_neg(p: Float64) -> Float64:
    """log(1 - p)."""
    return log1p(-p)
