"""Behaviour-cloning trainer for the router.

STUB (W-1): the signatures of MOJO_PORT.md section 5 (`BCEpochLog` is added: section 5 names it without
spelling its fields). Component C8 replaces this file; every behavioural body raises with a "STUB" message.
"""

from qumo.compiler.policy import ModelFile
from qumo.rl.teacher_data import TeacherData
from qumo.rl.router_net import RouterNet


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


@fieldwise_init
struct BCEpochLog(Copyable):
    var epoch: Int
    var loss: Float64
    var val_agree: Float64
    var val_regret: Float64
    var t: Float64


struct BCResult(Copyable):
    var log: List[BCEpochLog]
    var samples: Int
    var val: Int
    var wall_s: Float64

    def __init__(out self):
        self.log = List[BCEpochLog]()
        self.samples = 0
        self.val = 0
        self.wall_s = 0.0


def train_bc(cfg: BCConfig, data: TeacherData, init: Optional[ModelFile]) raises -> Tuple[RouterNet, BCResult]:
    raise Error("QM-E640: qumo.rl.bc.train_bc is a STUB (component C8)")
