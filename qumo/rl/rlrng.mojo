"""Seed derivation, sampling helpers and the FMA barrier for the native RL stack.

STUB (W-1): the signatures of MOJO_PORT.md section 5 only. Component C4 replaces
this file; every behavioural body aborts with a "STUB" message.
"""

from std.os import abort
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


def splitmix64(x: UInt64) -> UInt64:
    abort("qumo.rl.rlrng.splitmix64: STUB (component C4)")


def seed_for(base: UInt64, tag: Int, a: Int = 0, b: Int = 0, c: Int = 0) -> UInt64:
    abort("qumo.rl.rlrng.seed_for: STUB (component C4)")


def uniform_f32(mut r: Rng) -> Float32:
    abort("qumo.rl.rlrng.uniform_f32: STUB (component C4)")


def uniform_k24(mut r: Rng) -> Int:
    """Integer k in [0, 2^24)."""
    abort("qumo.rl.rlrng.uniform_k24: STUB (component C4)")


def uniform_sym_f32(mut r: Rng, bound: Float32) -> Float32:
    abort("qumo.rl.rlrng.uniform_sym_f32: STUB (component C4)")


def normal_f32(mut r: Rng) -> Float32:
    abort("qumo.rl.rlrng.normal_f32: STUB (component C4)")


def randperm(mut r: Rng, n: Int) -> List[Int]:
    abort("qumo.rl.rlrng.randperm: STUB (component C4)")


def gumbel_argmax(logits: FPtr, n: Int, mut r: Rng) -> Int:
    """First maximum wins (strict >), NaN never wins."""
    abort("qumo.rl.rlrng.gumbel_argmax: STUB (component C4)")


def sample_softmax(logits: FPtr, n: Int, mut r: Rng) -> Int:
    """Max-shifted softmax, inverse CDF."""
    abort("qumo.rl.rlrng.sample_softmax: STUB (component C4)")


@no_inline
def mulf(a: Float32, b: Float32) -> Float32:
    """FMA barrier (MOJO_PORT.md section 3, rule 9)."""
    abort("qumo.rl.rlrng.mulf: STUB (component C4)")


# --- helper named in MOJO_PORT.md 4.1 / 11.3 (not part of the section 5 contract) ---
def sample_probs(p: FPtr, n: Int, mut r: Rng) -> Int:
    """Inverse CDF over unnormalised probabilities `p` (u * sum(p), last index on round-off)."""
    abort("qumo.rl.rlrng.sample_probs: STUB (component C4)")
