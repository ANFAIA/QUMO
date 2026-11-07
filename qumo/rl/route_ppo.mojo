"""Optional router PPO fine-tune (outside `make train-ai`).

STUB (W-1): MOJO_PORT.md section 5 does not list this module; the signatures below are PROVISIONAL
(derived from 8.3 and 10.1, C12). Bodies raise with a "STUB" message.
"""

from qumo.compiler.policy import ModelFile
from qumo.rl.router_net import RouterNet


@fieldwise_init
struct RoutePPOConfig(Copyable):
    var budget_s: Float64
    var workers: Int
    var seed: UInt64

    def __init__(out self):
        self.budget_s = 0.0
        self.workers = 4
        self.seed = 0


def train_route_ppo(cfg: RoutePPOConfig, init: ModelFile) raises -> RouterNet:
    raise Error("QM-E640: qumo.rl.route_ppo.train_route_ppo is a STUB (component C12)")
