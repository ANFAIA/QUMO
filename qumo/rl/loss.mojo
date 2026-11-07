"""Synthesis PPO loss with explicit gradient, log-softmax helpers and twisterl-form GAE.

STUB (W-1): the signatures of MOJO_PORT.md section 5 only. Component C5 replaces
this file; every behavioural body aborts with a "STUB" message.
"""

from std.os import abort
from qumo.rl.tensor import FPtr


@fieldwise_init
struct PPOLossCfg(Copyable):
    var clip: Float32
    var vf_coef: Float32
    var ent_coef: Float32


@fieldwise_init
struct LossParts(Copyable):
    """Partial sums, not yet divided by N."""

    var policy: Float64
    var value: Float64
    var entropy: Float64


def log_softmax_row(z: FPtr, n: Int, out_lp: FPtr):
    abort("qumo.rl.loss.log_softmax_row: STUB (component C5)")


def logsumexp_row(z: FPtr, n: Int) -> Float32:
    abort("qumo.rl.loss.logsumexp_row: STUB (component C5)")


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
    abort("qumo.rl.loss.ppo_chunk_loss: STUB (component C5)")


def gae_twisterl(
    rewards: Span[Float32, _],
    values: Span[Float32, _],
    gamma: Float32,
    lam: Float32,
    mut advs: List[Float32],
    mut rets: List[Float32],
):
    abort("qumo.rl.loss.gae_twisterl: STUB (component C5)")
