"""PPO for the synthesis environments: collector, evaluator, full-batch update and the trainer.

STUB (W-1): the signatures of MOJO_PORT.md section 5 only. Component C7 replaces
this file; every behavioural body aborts or raises with a "STUB" message.

Contract change versus section 5 (Mojo 1.2 compile fix): `Trainer.__init__` takes `var net` because
`BasicNet` is Movable but not Copyable, so it can only be moved into the trainer.
"""

from std.os import abort
from qumo.rl.tensor import CHUNK_ROWS
from qumo.rl.optim import Adam
from qumo.rl.nn import BasicNet, SparseBatch
from qumo.rl.synth_env import TrainEnv


@fieldwise_init
struct PPOConfig(Copyable):
    var num_cores: Int
    var num_episodes: Int
    var gamma: Float32
    var lam: Float32
    var num_epochs: Int
    var vf_coef: Float32
    var ent_coef: Float32
    var clip_ratio: Float32
    var normalize_advantage: Bool
    var lr: Float64
    var diff_threshold: Float32
    var diff_max: Int
    var eval_episodes: Int
    var eval_sampled_searches: Int
    var eval_sampled_every: Int
    var seed: UInt64
    var chunk_rows: Int
    var start_iteration: Int  # global iteration offset (section 3, rule 1)
    var mask_new_logits: Bool  # D6 switch, faithful default False
    var best_checkpoint: String  # D6 switch, faithful default "post_update"

    def __init__(out self):
        """The constants in effect for QUMO (MOJO_PORT.md 19.1)."""
        self.num_cores = 4
        self.num_episodes = 1024
        self.gamma = 0.995
        self.lam = 0.995
        self.num_epochs = 10
        self.vf_coef = 0.8
        self.ent_coef = 0.01
        self.clip_ratio = 0.1
        self.normalize_advantage = False
        self.lr = 3e-4
        self.diff_threshold = 0.85
        self.diff_max = 256
        self.eval_episodes = 200
        self.eval_sampled_searches = 10
        self.eval_sampled_every = 5
        self.seed = 1
        self.chunk_rows = CHUNK_ROWS
        self.start_iteration = 0
        self.mask_new_logits = False
        self.best_checkpoint = "post_update"


struct Rollout(Movable):
    var sb: SparseBatch
    var logits: List[Float32]
    var acts: List[Int32]
    var advs: List[Float32]
    var rets: List[Float32]
    var old_lp: List[Float32]
    var allowed: List[UInt8]  # per-row mask flag, only used when mask_new_logits (7.3); addition to section 5
    var n: Int
    var num_episodes: Int
    var status: Int  # first non-zero item status in episode order (section 3 rule 7)

    def __init__(out self):
        self.sb = SparseBatch()
        self.logits = List[Float32]()
        self.acts = List[Int32]()
        self.advs = List[Float32]()
        self.rets = List[Float32]()
        self.old_lp = List[Float32]()
        self.allowed = List[UInt8]()
        self.n = 0
        self.num_episodes = 0
        self.status = 0


@fieldwise_init
struct EvalResult(Copyable):
    var success: Float32
    var reward: Float32
    var status: Int


@fieldwise_init
struct LossStats(Copyable):
    var value: Float64
    var policy: Float64
    var entropy: Float64
    var total: Float64


@fieldwise_init
struct StepReport(Copyable):
    var det: EvalResult
    var sampled: EvalResult
    var have_sampled: Bool
    var difficulty: Int
    var loss: LossStats
    var rows: Int
    var step_s: Float64


def collect(env: TrainEnv, net: BasicNet, cfg: PPOConfig, iteration: Int) -> Rollout:
    abort("qumo.rl.ppo.collect: STUB (component C7)")


def evaluate(
    env: TrainEnv,
    net: BasicNet,
    cfg: PPOConfig,
    iteration: Int,
    deterministic: Bool,
    num_searches: Int,
    num_episodes: Int,
) -> EvalResult:
    abort("qumo.rl.ppo.evaluate: STUB (component C7)")


def ppo_update(mut net: BasicNet, mut opt: Adam, ro: Rollout, cfg: PPOConfig) -> LossStats:
    abort("qumo.rl.ppo.ppo_update: STUB (component C7)")


struct Trainer(Movable):
    var net: BasicNet
    var opt: Adam
    var env: TrainEnv
    var cfg: PPOConfig
    var iteration: Int

    def __init__(out self, env: TrainEnv, var net: BasicNet, cfg: PPOConfig):
        self.opt = Adam(net.param_count(), cfg.lr)
        self.net = net^
        self.env = env.copy()
        self.cfg = cfg.copy()
        self.iteration = 0

    def learn_step(mut self) raises -> StepReport:
        """evals -> collect -> update; does NOT promote difficulty; raises the first status-slot error
        (QM-E642) or QM-E643 on a non-finite loss."""
        raise Error("QM-E640: qumo.rl.ppo.Trainer.learn_step is a STUB (component C7)")

    def promote_if_ready(mut self, rep: StepReport) -> Bool:
        """success >= threshold and difficulty < diff_max."""
        abort("qumo.rl.ppo.Trainer.promote_if_ready: STUB (component C7)")
