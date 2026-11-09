"""PPO for the synthesis environments: collector, evaluator, full-batch update and the trainer (MOJO_PORT.md 7).

Algorithm (the reference trainer, constants of MOJO_PORT.md 19.1), one `Trainer.learn_step`:

    1. evals of the unchanged policy at the current difficulty: `ppo_deterministic` (argmax, 1 search) drives the
       curriculum, `ppo_10` (sampled, 10 searches) is logged (every `eval_sampled_every` iterations);
    2. `collect`: `num_episodes` full on-policy episodes, one seeded environment stream and one seeded actor stream per
       episode, generalised advantage estimates per episode in the reference form;
    3. `ppo_update`: `num_epochs` full-batch Adam steps over every recorded row.

Episode semantics (7.2): row `t` is the state `t` itself, with the reward earned by the previous action (`reward()`
read before acting); the terminal state is recorded too (its action is never executed) and carries all-masked logits
iff it is solved; the twist drawn for the decision is the one used again by the update (`Rollout.sb.perm`); the
recorded logits are the *masked* logits (`-1e10` everywhere on a solved state); the action is the Gumbel-max sample of
the masked logits (index 0 on an all-masked row, where the noise is absorbed). `old_lp` is `l_a - logsumexp(l)` of the
recorded masked logits; the update uses the *new, unmasked* logits (the reference quirk, switch `mask_new_logits`).

Determinism (section 3): everything random derives from `seed_for(cfg.seed, TAG, git, e, c)` with the global iteration
key `git = cfg.start_iteration + iteration`, so a stage of a warm-start chain never replays the previous stage. Episodes
and evaluation episodes are independent items: each item draws only from its own generators and writes into its own
slot; slots are merged in item-index order. The update cuts the rows into fixed chunks of `cfg.chunk_rows`, chunk `i`
belongs to virtual worker `i % VW`, every virtual worker owns a private gradient buffer, and the `VW` buffers are summed
in index order, so the number of cores changes the speed only. Tasks never raise: an environment error (QM-E642) is
recorded in a status slot per episode and raised by the caller after the parallel region, first non-zero in episode
order.

`iteration` arguments of `collect`, `evaluate` and the trainer are the *local* iteration count of this process; the
global key adds `cfg.start_iteration`.
"""

from std.math import isfinite
from std.os import abort
from std.runtime import parallelism_level
from std.time import perf_counter_ns
from max.algorithm import parallelize
from qrl.rng import Rng
from qrl.tensor import FPtr, CHUNK_ROWS, VW, fill_f32
from qrl.optim import Adam
from qrl.rlrng import (
    TAG_COLLECT,
    TAG_EVAL_DET,
    TAG_EVAL_SAMPLED,
    seed_for,
    gumbel_argmax,
    sample_softmax,
    argmax_first,
)
from qrl.nn import BasicNet, SparseBatch, ChunkCache, ActorScratch, GradBufs
from qrl.loss import PPOLossCfg, IPtr, ppo_chunk_loss, old_logp_row, gae_twisterl_ptr, normalize_advantages
from qrl.synth_env import TrainEnv

comptime MASKED_LOGIT: Float32 = -1e10  # the value the collector stores for an illegal action
comptime _MAX_BLOCKS = 64  # episode items are interleaved over this many tasks (speed only, never results)


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
    var best_checkpoint: String  # D6 switch, faithful default "post_update" (read by the driver)

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
    """All rows recorded by `collect`, episodes concatenated in episode-index order (row `r` of episode `e` follows
    row `r - 1`; `ep_len[e]` rows per episode)."""

    var sb: SparseBatch  # observation indices and the twist used for each row (-1 = none)
    var logits: List[Float32]  # N x A masked logits as recorded (-1e10 for an illegal action)
    var acts: List[Int32]
    var advs: List[Float32]
    var rets: List[Float32]
    var old_lp: List[Float32]
    var allowed: List[UInt8]  # per-row mask flag (1 = unsolved state); read by `mask_new_logits` (7.3)
    var values: List[Float32]  # value-head output of the recorded state (additive; used by tests and diagnostics)
    var rewards: List[Float32]  # `env.reward()` of the recorded state (additive)
    var ep_len: List[Int32]  # rows per episode (additive)
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
        self.values = List[Float32]()
        self.rewards = List[Float32]()
        self.ep_len = List[Int32]()
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


# ------------------------------------------------------------------------------------------------ seeds and helpers


def effective_cores(requested: Int) -> Int:
    """`min(requested, parallelism_level())`, at least 1 (MOJO_PORT.md section 3, rule 8). Speed only."""
    return max(1, min(requested, parallelism_level()))


def global_iteration(cfg: PPOConfig, iteration: Int) -> Int:
    """The iteration key of every seed: the local count plus the warm-start offset (section 3, rule 1)."""
    return cfg.start_iteration + iteration


def collect_env_seed(cfg: PPOConfig, iteration: Int, e: Int) -> UInt64:
    """Environment stream of collector episode `e` (reset, inversion bits, Pauli symmetry draws)."""
    return seed_for(cfg.seed, TAG_COLLECT, global_iteration(cfg, iteration), e, 0)


def collect_actor_seed(cfg: PPOConfig, iteration: Int, e: Int) -> UInt64:
    """Actor stream of collector episode `e` (twist draws and the Gumbel noise)."""
    return seed_for(cfg.seed, TAG_COLLECT, global_iteration(cfg, iteration), e, 1)


def eval_env_seed(cfg: PPOConfig, iteration: Int, e: Int) -> UInt64:
    """Environment stream of evaluation episode `e` (the reset); shared by the deterministic and the sampled eval, so
    both start from the same states."""
    return seed_for(cfg.seed, TAG_EVAL_DET, global_iteration(cfg, iteration), e, 0)


def eval_search_env_seed(cfg: PPOConfig, iteration: Int, e: Int, k: Int) -> UInt64:
    """Environment stream of search `k` of evaluation episode `e` (inversion bits, Pauli symmetry draws)."""
    return seed_for(cfg.seed, TAG_EVAL_DET, global_iteration(cfg, iteration), e, 2 + k)


def eval_actor_seed(cfg: PPOConfig, iteration: Int, e: Int, k: Int, deterministic: Bool) -> UInt64:
    """Actor stream (twist draws, and the sampling when not deterministic). The deterministic eval uses one stream per
    episode for all its searches; search `k` of the sampled eval has its own stream (`c = 2 + k`)."""
    if deterministic:
        return seed_for(cfg.seed, TAG_EVAL_DET, global_iteration(cfg, iteration), e, 1)
    return seed_for(cfg.seed, TAG_EVAL_SAMPLED, global_iteration(cfg, iteration), e, 2 + k)


def eval_better(s_new: Float32, r_new: Float32, s_old: Float32, r_old: Float32) -> Bool:
    """Lexicographic order of the search candidates: success first, then the summed reward; strictly greater wins
    (the first of equal candidates is kept)."""
    if s_new > s_old:
        return True
    if s_new == s_old and r_new > r_old:
        return True
    return False


def first_status(slots: List[Int32]) -> Int:
    """First non-zero slot in index order (section 3, rule 7), 0 when every slot is ok."""
    for i in range(len(slots)):
        if slots[i] != 0:
            return Int(slots[i])
    return 0


def _all_finite(xs: List[Float32]) -> Bool:
    for i in range(len(xs)):
        if not isfinite(xs[i]):
            return False
    return True


# ------------------------------------------------------------------------------------------------ collector


struct _Block(Movable):
    """Rows recorded by one collector task (its episodes back to back)."""

    var idx: List[Int32]
    var off: List[Int32]  # row r is idx[off[r] : off[r + 1]]; starts as [0]
    var perm: List[Int32]
    var logits: List[Float32]
    var acts: List[Int32]
    var advs: List[Float32]
    var rets: List[Float32]
    var old_lp: List[Float32]
    var values: List[Float32]
    var rewards: List[Float32]
    var allowed: List[UInt8]

    def __init__(out self):
        self.idx = List[Int32]()
        self.off = List[Int32]()
        self.off.append(0)
        self.perm = List[Int32]()
        self.logits = List[Float32]()
        self.acts = List[Int32]()
        self.advs = List[Float32]()
        self.rets = List[Float32]()
        self.old_lp = List[Float32]()
        self.values = List[Float32]()
        self.rewards = List[Float32]()
        self.allowed = List[UInt8]()

    def rows(self) -> Int:
        return len(self.perm)


def _collect_episode(
    env: TrainEnv,
    net: BasicNet,
    cfg: PPOConfig,
    iteration: Int,
    e: Int,
    mut blk: _Block,
    mut sc: ActorScratch,
    mut obs: List[Int],
    mut obs32: List[Int32],
    mut masked: List[Float32],
) -> Int:
    """One collector episode appended to `blk` (rows, GAE included). Returns the environment error code (0 = ok)."""
    var A = net.spec.num_actions
    var T = len(net.spec.obs_perms)
    var ev = env.copy()
    ev.reseed(collect_env_seed(cfg, iteration, e))
    ev.reset()
    var arng = Rng(collect_actor_seed(cfg, iteration, e))
    var start = blk.rows()
    var mp = masked.unsafe_ptr().as_unsafe_any_origin()
    while True:
        ev.observe(obs)
        var allowed = ev.all_allowed()
        var r = ev.reward()
        var t = -1
        if T > 0:
            t = arng.below(T)
        obs32.clear()
        for i in range(len(obs)):
            obs32.append(Int32(obs[i]))
        var v = net.forward_one(Span(obs32), t, sc, mp)
        if not allowed:
            for a in range(A):
                masked[a] = MASKED_LOGIT
        var act = gumbel_argmax(mp, A, arng)
        var lp = old_logp_row(mp, A, act)
        for i in range(len(obs32)):
            blk.idx.append(obs32[i])
        blk.off.append(Int32(len(blk.idx)))
        blk.perm.append(Int32(t))
        for a in range(A):
            blk.logits.append(masked[a])
        blk.acts.append(Int32(act))
        blk.advs.append(0.0)
        blk.rets.append(0.0)
        blk.old_lp.append(lp)
        blk.values.append(v)
        blk.rewards.append(r)
        var flag: UInt8 = 0
        if allowed:
            flag = 1
        blk.allowed.append(flag)
        if ev.is_final():
            break
        ev.step(act)
    var n = blk.rows() - start
    var rp = blk.rewards.unsafe_ptr().as_unsafe_any_origin().unsafe_offset(start)
    var vp = blk.values.unsafe_ptr().as_unsafe_any_origin().unsafe_offset(start)
    var ap = blk.advs.unsafe_ptr().as_unsafe_any_origin().unsafe_offset(start)
    var tp = blk.rets.unsafe_ptr().as_unsafe_any_origin().unsafe_offset(start)
    gae_twisterl_ptr(n, rp, vp, cfg.gamma, cfg.lam, ap, tp)
    _ = len(blk.rewards)
    _ = len(blk.values)
    _ = len(blk.advs)
    _ = len(blk.rets)
    _ = len(masked)
    return ev.error_code()


def collect(env: TrainEnv, net: BasicNet, cfg: PPOConfig, iteration: Int) -> Rollout:
    """`cfg.num_episodes` on-policy episodes at the environment's current difficulty (MOJO_PORT.md 7.2). Episode `e`
    copies `env`, reseeds it with `collect_env_seed`, resets, and acts with the generator `collect_actor_seed`. The
    result does not depend on `cfg.num_cores`. `Rollout.status` is the first non-zero environment error code in episode
    order (QM-E642 for an action outside the environment's gateset, e.g. a network with the wrong number of actions)."""
    var E = cfg.num_episodes
    var ro = Rollout()
    if E <= 0:
        return ro^
    var nblk = min(E, _MAX_BLOCKS)
    var blocks = List[_Block]()
    for _ in range(nblk):
        blocks.append(_Block())
    var ep_rows = List[Int32](length=E, fill=0)
    var ep_start = List[Int32](length=E, fill=0)
    var ep_status = List[Int32](length=E, fill=0)
    var p_rows = ep_rows.unsafe_ptr().as_unsafe_any_origin()
    var p_start = ep_start.unsafe_ptr().as_unsafe_any_origin()
    var p_status = ep_status.unsafe_ptr().as_unsafe_any_origin()

    def task(b: Int) {imm env, imm net, imm cfg, imm iteration, imm E, imm nblk, mut blocks, imm p_rows, imm p_start, imm p_status}:
        var sc = ActorScratch(net.spec)
        var obs = List[Int]()
        var obs32 = List[Int32]()
        var masked = List[Float32](length=net.spec.num_actions, fill=0.0)
        var e = b
        while e < E:
            var rs = blocks[b].rows()
            var code = _collect_episode(env, net, cfg, iteration, e, blocks[b], sc, obs, obs32, masked)
            p_start[unsafe_offset=e] = Int32(rs)
            p_rows[unsafe_offset=e] = Int32(blocks[b].rows() - rs)
            p_status[unsafe_offset=e] = Int32(code)
            e += nblk

    parallelize(task, nblk, effective_cores(cfg.num_cores))
    _ = len(ep_rows)
    _ = len(ep_start)
    _ = len(ep_status)
    # merge in episode-index order
    var total = 0
    for e in range(E):
        total += Int(ep_rows[e])
    var A = net.spec.num_actions
    ro.logits.reserve(total * A)
    ro.acts.reserve(total)
    ro.advs.reserve(total)
    ro.rets.reserve(total)
    ro.old_lp.reserve(total)
    ro.allowed.reserve(total)
    ro.values.reserve(total)
    ro.rewards.reserve(total)
    ro.sb.perm.reserve(total)
    ro.sb.off.reserve(total + 1)
    for e in range(E):
        var b = e % nblk
        var rs = Int(ep_start[e])
        var L = Int(ep_rows[e])
        ro.ep_len.append(Int32(L))
        for r in range(rs, rs + L):
            var q0 = Int(blocks[b].off[r])
            var q1 = Int(blocks[b].off[r + 1])
            for q in range(q0, q1):
                ro.sb.idx.append(blocks[b].idx[q])
            ro.sb.off.append(Int32(len(ro.sb.idx)))
            ro.sb.perm.append(blocks[b].perm[r])
            for a in range(A):
                ro.logits.append(blocks[b].logits[r * A + a])
            ro.acts.append(blocks[b].acts[r])
            ro.advs.append(blocks[b].advs[r])
            ro.rets.append(blocks[b].rets[r])
            ro.old_lp.append(blocks[b].old_lp[r])
            ro.allowed.append(blocks[b].allowed[r])
            ro.values.append(blocks[b].values[r])
            ro.rewards.append(blocks[b].rewards[r])
    ro.n = total
    ro.num_episodes = E
    ro.status = first_status(ep_status)
    return ro^


# ------------------------------------------------------------------------------------------------ evaluation


def _eval_episode(
    env: TrainEnv,
    net: BasicNet,
    cfg: PPOConfig,
    iteration: Int,
    e: Int,
    deterministic: Bool,
    searches: Int,
    mut sc: ActorScratch,
    mut obs: List[Int],
    mut obs32: List[Int32],
    mut logits: List[Float32],
    mut out_success: Float32,
    mut out_reward: Float32,
) -> Int:
    """One evaluation episode: reset once, then `searches` playouts from clones of that start state; the best
    candidate (success first, then the summed reward of every visited state, final one included) is kept. Returns the
    environment error code."""
    var A = net.spec.num_actions
    var T = len(net.spec.obs_perms)
    var base = env.copy()
    base.reseed(eval_env_seed(cfg, iteration, e))
    base.reset()
    var lp = logits.unsafe_ptr().as_unsafe_any_origin()
    var arng = Rng(eval_actor_seed(cfg, iteration, e, 0, deterministic))
    var have = False
    var best_s: Float32 = 0.0
    var best_r: Float32 = 0.0
    var status = 0
    for k in range(searches):
        var c = base.copy()
        c.reseed(eval_search_env_seed(cfg, iteration, e, k))
        if not deterministic:
            arng = Rng(eval_actor_seed(cfg, iteration, e, k, False))
        var total: Float32 = 0.0
        while not c.is_final():
            total += c.reward()
            c.observe(obs)
            var t = -1
            if T > 0:
                t = arng.below(T)
            obs32.clear()
            for i in range(len(obs)):
                obs32.append(Int32(obs[i]))
            _ = net.forward_one(Span(obs32), t, sc, lp)
            var act = 0
            if deterministic:
                act = argmax_first(lp, A)
            else:
                act = sample_softmax(lp, A, arng)
            c.step(act)
        total += c.reward()
        var s: Float32 = 0.0
        if c.success():
            s = 1.0
        if status == 0:
            status = c.error_code()
        if (not have) or eval_better(s, total, best_s, best_r):
            have = True
            best_s = s
            best_r = total
    _ = len(logits)
    out_success = best_s
    out_reward = best_r
    return status


def evaluate(
    env: TrainEnv,
    net: BasicNet,
    cfg: PPOConfig,
    iteration: Int,
    deterministic: Bool,
    num_searches: Int,
    num_episodes: Int,
) -> EvalResult:
    """Success rate and mean summed reward over `num_episodes` fresh episodes at the environment's difficulty
    (MOJO_PORT.md 7.4): the argmax policy (first maximum) when `deterministic`, else sampling from softmax(logits);
    every forward call uses a fresh random twist; masks never matter (states are unsolved until final). Success is the
    integer count divided by the episode count in f32; the reward is the mean of the per-episode f32 totals (summed in
    f64 in episode order, rounded once). Independent of `cfg.num_cores`."""
    var E = num_episodes
    if E <= 0:
        return EvalResult(0.0, 0.0, 0)
    var searches = max(num_searches, 1)
    var succ = List[Float32](length=E, fill=0.0)
    var rew = List[Float32](length=E, fill=0.0)
    var stat = List[Int32](length=E, fill=0)
    var ps = succ.unsafe_ptr().as_unsafe_any_origin()
    var pr = rew.unsafe_ptr().as_unsafe_any_origin()
    var pt = stat.unsafe_ptr().as_unsafe_any_origin()
    var nblk = min(E, _MAX_BLOCKS)

    def task(b: Int) {imm env, imm net, imm cfg, imm iteration, imm E, imm nblk, imm deterministic, imm searches, imm ps, imm pr, imm pt}:
        var sc = ActorScratch(net.spec)
        var obs = List[Int]()
        var obs32 = List[Int32]()
        var logits = List[Float32](length=net.spec.num_actions, fill=0.0)
        var e = b
        while e < E:
            var s: Float32 = 0.0
            var r: Float32 = 0.0
            var code = _eval_episode(env, net, cfg, iteration, e, deterministic, searches, sc, obs, obs32, logits, s, r)
            ps[unsafe_offset=e] = s
            pr[unsafe_offset=e] = r
            pt[unsafe_offset=e] = Int32(code)
            e += nblk

    parallelize(task, nblk, effective_cores(cfg.num_cores))
    _ = len(succ)
    _ = len(rew)
    _ = len(stat)
    # success: the integer count over E as one f32 division (so exactly 170 / 200 gives the f32 value of 0.85);
    # reward: the per-episode f32 totals summed in f64 in episode order and rounded once. The reference sums in f32,
    # whose rounding drift (about 1e-5 relative over 1,000 equal terms with a sequential sum) is an artefact of the
    # summation order, not of the statistic; it only matters for the lexicographic "best" tuple.
    var n_ok = 0
    var sum_r: Float64 = 0.0
    for e in range(E):
        if succ[e] > 0.5:
            n_ok += 1
        sum_r += Float64(rew[e])
    return EvalResult(Float32(n_ok) / Float32(E), Float32(sum_r / Float64(E)), first_status(stat))


# ------------------------------------------------------------------------------------------------ the update


def _epoch_pass(
    net: BasicNet,
    ro: Rollout,
    p_advs: FPtr,
    p_rets: FPtr,
    p_old: FPtr,
    p_acts: IPtr,
    lcfg: PPOLossCfg,
    chunk_rows: Int,
    mask_new: Bool,
    cores: Int,
    mut gb: GradBufs,
    mut caches: List[ChunkCache],
    mut dz_all: List[Float32],
    mut dv_all: List[Float32],
    mut part: List[Float64],
):
    """Forward, loss and backward of every chunk over the virtual workers: the private gradient buffers of `gb` hold the
    chunk sums of each worker, and `part` the (policy, value, entropy) partial sums per chunk."""
    var n = ro.n
    var A = net.spec.num_actions
    var P = net.param_count()
    var nchunks = (n + chunk_rows - 1) // chunk_rows
    var inv_n = Float32(1.0) / Float32(n)
    var gp = gb.data.unsafe_ptr().as_unsafe_any_origin()
    var pdz = dz_all.unsafe_ptr().as_unsafe_any_origin()
    var pdv = dv_all.unsafe_ptr().as_unsafe_any_origin()
    var pp = part.unsafe_ptr().as_unsafe_any_origin()

    def task(w: Int) {imm net, imm ro, imm p_advs, imm p_rets, imm p_old, imm p_acts, imm lcfg, imm chunk_rows, imm mask_new, imm n, imm A, imm P, imm nchunks, imm inv_n, imm gp, imm pdz, imm pdv, imm pp, mut caches}:
        var gw = gp.unsafe_offset(w * P)
        fill_f32(gw, P, 0.0)
        var dz = pdz.unsafe_offset(w * chunk_rows * A)
        var dv = pdv.unsafe_offset(w * chunk_rows)
        var c = w
        while c < nchunks:
            var lo = c * chunk_rows
            var hi = min(lo + chunk_rows, n)
            var rows = hi - lo
            net.forward_chunk(ro.sb, lo, hi, caches[w])
            var lg = caches[w].logits.unsafe_ptr().as_unsafe_any_origin()
            if mask_new:
                for r in range(rows):
                    if ro.allowed[lo + r] == 0:
                        for a in range(A):
                            lg[unsafe_offset = r * A + a] = MASKED_LOGIT
            var parts = ppo_chunk_loss(
                lg,
                caches[w].values.unsafe_ptr().as_unsafe_any_origin(),
                rows,
                A,
                p_acts.unsafe_offset(lo),
                p_advs.unsafe_offset(lo),
                p_rets.unsafe_offset(lo),
                p_old.unsafe_offset(lo),
                lcfg,
                inv_n,
                dz,
                dv,
            )
            if mask_new:
                for r in range(rows):
                    if ro.allowed[lo + r] == 0:
                        for a in range(A):
                            dz[unsafe_offset = r * A + a] = 0.0
            pp[unsafe_offset = 3 * c] = parts.policy
            pp[unsafe_offset = 3 * c + 1] = parts.value
            pp[unsafe_offset = 3 * c + 2] = parts.entropy
            net.backward_chunk(ro.sb, lo, hi, caches[w], dz, dv, gw)
            c += VW

    parallelize(task, VW, cores)
    _ = len(gb.data)
    _ = len(dz_all)
    _ = len(dv_all)
    _ = len(part)
    for w in range(VW):
        _ = len(caches[w].logits)
        _ = len(caches[w].values)


def ppo_update(mut net: BasicNet, mut opt: Adam, ro: Rollout, cfg: PPOConfig) -> LossStats:
    """`cfg.num_epochs` full-batch Adam steps over all `ro.n` rows (MOJO_PORT.md 7.3). Every epoch re-runs the forward
    pass with the row's recorded twist on the *unmasked* logits (unless `mask_new_logits`), the clipped surrogate loss
    with `old_lp` from the rollout, and the hand-derived backward. The returned losses are those of the last epoch's
    forward (as the reference returns them). Bit-identical for any `cfg.num_cores`."""
    if ro.n == 0 or cfg.num_epochs <= 0:
        return LossStats(0.0, 0.0, 0.0, 0.0)
    if cfg.mask_new_logits and len(ro.allowed) != ro.n:
        abort("qrl.ppo.ppo_update: mask_new_logits needs Rollout.allowed for every row")
    var n = ro.n
    var A = net.spec.num_actions
    var P = net.param_count()
    var chunk_rows = max(1, cfg.chunk_rows)
    var nchunks = (n + chunk_rows - 1) // chunk_rows
    var cores = effective_cores(cfg.num_cores)
    var lcfg = PPOLossCfg(cfg.clip_ratio, cfg.vf_coef, cfg.ent_coef)
    var advs_norm = List[Float32]()
    var src_advs = ro.advs.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin()
    if cfg.normalize_advantage:
        advs_norm = ro.advs.copy()
        src_advs = advs_norm.unsafe_ptr().as_unsafe_any_origin()
        normalize_advantages(src_advs, n)
    var p_rets = ro.rets.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin()
    var p_old = ro.old_lp.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin()
    var p_acts = ro.acts.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin()
    var gb = GradBufs(P, VW)
    var grad = List[Float32](length=P, fill=0.0)
    var caches = List[ChunkCache]()
    for _ in range(VW):
        caches.append(ChunkCache(net.spec, chunk_rows))
    var dz_all = List[Float32](length=VW * chunk_rows * A, fill=0.0)
    var dv_all = List[Float32](length=VW * chunk_rows, fill=0.0)
    var part = List[Float64](length=3 * nchunks, fill=0.0)
    var last_pol: Float64 = 0.0
    var last_val: Float64 = 0.0
    var last_ent: Float64 = 0.0
    for _epoch in range(cfg.num_epochs):
        _epoch_pass(net, ro, src_advs, p_rets, p_old, p_acts, lcfg, chunk_rows, cfg.mask_new_logits, cores, gb, caches, dz_all, dv_all, part)
        # chunk partial sums in chunk order, then the gradient buffers in worker order
        var sp: Float64 = 0.0
        var sv: Float64 = 0.0
        var se: Float64 = 0.0
        for c in range(nchunks):
            sp += part[3 * c]
            sv += part[3 * c + 1]
            se += part[3 * c + 2]
        last_pol = sp
        last_val = sv
        last_ent = se
        gb.reduce_into(grad)
        opt.step(net.ps.data, grad, cfg.lr)
    _ = len(ro.advs)
    _ = len(ro.rets)
    _ = len(ro.old_lp)
    _ = len(ro.acts)
    _ = len(advs_norm)
    var nf = Float64(n)
    var pol = last_pol / nf
    var val = last_val / nf
    var ent = last_ent / nf
    var total = pol + Float64(cfg.vf_coef) * val - Float64(cfg.ent_coef) * ent
    return LossStats(val, pol, ent, total)


# ------------------------------------------------------------------------------------------------ the trainer


def _e642(why: String) -> Error:
    return Error("QM-E642: invalid training environment: " + why + ". Hint: the network must match the environment (observation size and number of actions).")


struct Trainer(Movable):
    var net: BasicNet
    var opt: Adam
    var env: TrainEnv
    var cfg: PPOConfig
    var iteration: Int  # local iteration count of this process (the global key adds cfg.start_iteration)
    var evaluated: List[Float32]  # weights the last learn_step evaluated and collected with (pre-update); for the
    # D6 `best_checkpoint = "evaluated"` alternative

    def __init__(out self, env: TrainEnv, var net: BasicNet, cfg: PPOConfig):
        self.opt = Adam(net.param_count(), cfg.lr)
        # Rows of the embedding table that no observation ever activates stay exactly 0 in both moments, so skipping
        # them is bit-exact against the dense update (MOJO_PORT.md 4.3).
        self.opt.mark_sparse_rows(net.ps.offsets[0], net.ps.cols_of(0), net.ps.rows_of(0))
        self.net = net^
        self.env = env.copy()
        self.cfg = cfg.copy()
        self.iteration = 0
        self.evaluated = List[Float32]()

    def learn_step(mut self) raises -> StepReport:
        """evals -> collect -> update; does NOT promote difficulty; raises the first status-slot error (QM-E642) or
        QM-E643 on a non-finite loss or non-finite weights (the weights and the Adam state are then restored to their
        values before the update, so the caller can still checkpoint the last good state)."""
        var t0 = perf_counter_ns()
        if self.net.spec.obs_size != self.env.obs_size():
            raise _e642("network observation size " + String(self.net.spec.obs_size) + " but the environment has " + String(self.env.obs_size()))
        if self.net.spec.num_actions != self.env.num_actions():
            raise _e642("network has " + String(self.net.spec.num_actions) + " actions but the environment has " + String(self.env.num_actions()))
        var difficulty = self.env.difficulty
        var det = evaluate(self.env, self.net, self.cfg, self.iteration, True, 1, self.cfg.eval_episodes)
        var have = False
        var samp = EvalResult(0.0, 0.0, 0)
        var every = max(self.cfg.eval_sampled_every, 1)
        if self.iteration % every == 0:
            have = True
            samp = evaluate(
                self.env, self.net, self.cfg, self.iteration, False, self.cfg.eval_sampled_searches, self.cfg.eval_episodes
            )
        var ro = collect(self.env, self.net, self.cfg, self.iteration)
        if det.status != 0:
            raise _e642("environment error code " + String(det.status) + " during the deterministic evaluation (iteration " + String(global_iteration(self.cfg, self.iteration)) + ")")
        if samp.status != 0:
            raise _e642("environment error code " + String(samp.status) + " during the sampled evaluation (iteration " + String(global_iteration(self.cfg, self.iteration)) + ")")
        if ro.status != 0:
            raise _e642("environment error code " + String(ro.status) + " while collecting (iteration " + String(global_iteration(self.cfg, self.iteration)) + ")")
        self.evaluated = self.net.ps.data.copy()
        var m_snap = self.opt.m.copy()
        var v_snap = self.opt.v.copy()
        var t_snap = self.opt.t
        var loss = ppo_update(self.net, self.opt, ro, self.cfg)
        if not isfinite(loss.total) or not _all_finite(self.net.ps.data):
            self.net.ps.data = self.evaluated.copy()
            self.opt.load_state(t_snap, m_snap, v_snap)
            raise Error(
                "QM-E643: training diverged: non-finite loss or weights at iteration "
                + String(global_iteration(self.cfg, self.iteration))
                + " (loss "
                + String(loss.total)
                + "). Hint: lower the learning rate or restart from the last good checkpoint; the weights were restored to the state before this update."
            )
        self.iteration += 1
        var step_s = Float64(perf_counter_ns() - t0) * 1e-9
        return StepReport(det^, samp^, have, difficulty, loss^, ro.n, step_s)

    def promote_if_ready(mut self, rep: StepReport) -> Bool:
        """success >= threshold and difficulty < diff_max: at most +1 per iteration, never demoted. `rep.det.success`
        is the f32 quotient `n_ok / E` and the threshold the f32 value of 0.85, so exactly 170 / 200 promotes."""
        if Float64(rep.det.success) >= Float64(self.cfg.diff_threshold) and self.env.difficulty < self.cfg.diff_max:
            self.env.set_difficulty(self.env.difficulty + 1)
            return True
        return False
