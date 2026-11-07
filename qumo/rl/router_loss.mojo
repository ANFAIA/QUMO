"""Router losses: behaviour-cloning soft cross-entropy, route PPO loss and float64 GAE.

STUB (W-1): MOJO_PORT.md section 5 does not list this module; the signatures below are PROVISIONAL
(derived from 8.1 and 8.3, C8) and the owner finalises them through the lead. Bodies abort with a
"STUB" message.
"""

from std.os import abort
from qumo.rl.tensor import FPtr


def gae_route(
    rewards: Span[Float64, _],
    values: Span[Float64, _],
    done: Span[Bool, _],
    gamma: Float64,
    lam: Float64,
    mut adv: List[Float64],
    mut ret: List[Float64],
):
    """Float64 router GAE over concatenated trajectories; `done[i]` marks the last step of a trajectory (8.3)."""
    abort("qumo.rl.router_loss.gae_route: STUB (component C8)")


def bc_soft_ce_chunk(
    logits: FPtr,
    q: FPtr,
    valid: FPtr,
    rows: Int,
    slots: Int,
    tau: Float32,
    dz: FPtr,
) -> Float64:
    """Soft-target cross entropy over the valid slots of each row; writes `(p - target)/B` scaled gradient into
    `dz` and returns the summed loss (8.1)."""
    abort("qumo.rl.router_loss.bc_soft_ce_chunk: STUB (component C8)")
