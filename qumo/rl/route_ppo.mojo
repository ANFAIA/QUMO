"""Router PPO fine-tune (MOJO_PORT.md 8.3; optional, outside `make train-ai`).

A policy-gradient stage on top of a behaviour-cloned router (`qumo.rlroute.v0`, sign-embedding MLP with a value
head). It plays routing episodes of the circuit-family job generator in the routing environment (`Episode`, the
split `pre_decision` / `needs_policy` / `act` API) and updates the network with PPO.

Per iteration `git = start_iteration + it`:

  1. `build_jobs(700000 + 1000 * seed + git, episodes, max_n, scale)` with the curriculum
     `scale = min(1.5, 0.4 + 0.02 * git)`;
  2. rollouts: each job is one episode (an independent task: its generators are `seed_for(seed, TAG_ROUTE_PPO, git,
     e, 0 | 1)` for the environment / layout and for the action sampling, so the result does not depend on
     `workers`). The policy runs in Float64 on the weights of the iteration start: `h = relu(b0 + sum_i sign(obs_i)
     E[i,:])`, `p = softmax(W1 h + b1)` over the `A = len(active)` active swaps, the action is the inverse CDF of one
     uniform draw on the normalised cumulative sum (right-sided search), `logp = log(p[a] + 1e-12)`, `v = wv . h +
     bv`;
  3. reward of a decision: `-(overhead after the decision and the fallback swaps that follow it, until the next
     decision or the end) + (overhead before) + 0.01 * (2q gates executed in that span)`, i.e. `-delta(cx units) +
     0.01 * gates executed`;
  4. Float64 GAE over the concatenated trajectories (`gae_route`, gamma 0.99, lam 0.95; `ret = adv + V` with the raw
     advantage), advantages standardised with the population std + 1e-8;
  5. 4 epochs of shuffled minibatches of 1024 rows (`randperm`, last short batch kept): the clipped-ratio loss of
     `router_loss.route_ppo_chunk_loss` (clip 0.1, vf 0.5, ent 0.005), hand-derived backward over the value head
     too, global gradient-norm clip 0.5 over the six parameter tensors, one Adam step (lr 2e-4, one optimizer for the
     whole run, never reset).

Determinism (section 3): the gradient of a minibatch is accumulated over fixed chunks of `RPPO_CHUNK_ROWS = 128` rows
on `RPPO_VW = 4` virtual workers (private buffers, reduced in worker order), the loss partial sums are per chunk and
summed in chunk order, so the weights are bit-identical for any `workers`. Episodes write into per-item slots that
are merged in episode order. Wall-clock is read only by the budget check and the log.

Errors: QM-E640 (bad options), QM-E643 (non-finite policy in a rollout or non-finite loss / weights after an update:
the run stops at the last good weights), QM-E647 (the init model is not a router container).
"""

from std.math import exp, log, sqrt
from std.time import perf_counter_ns
from std.runtime import parallelism_level
from max.algorithm import parallelize
from qumo.ai.json import json_quote, json_number, pretty_json
from qumo.compiler.policy import ModelFile
from qumo.compiler.target import Target
from qumo.compiler.ai_route import RProblem, Episode, episode_max_steps, _random_layout
from qumo.ir.circuit import Circuit
from qumo.ir.qasm import load_qasm
from qumo.sim.rng import Rng
from qumo.rl.container import write_model_dir
from qumo.rl.tensor import FPtr, add_inplace, fill_f32
from qumo.rl.optim import Adam, clip_grad_norm
from qumo.rl.rlrng import seed_for, randperm, TAG_ROUTE_PPO
from qumo.rl.router_net import RouterNet, RouterCache, SPtr, ROUTER_OBS, ROUTER_SLOTS
from qumo.rl.router_loss import route_ppo_chunk_loss, gae_route, RoutePPOCfg, RouteLossParts, IPtr
from qumo.rl.targets import RTopo
from qumo.rl.jobs import JobSet, JobCircuit, build_jobs, OP_CX, OP_CZ, OP_SWAP, OP_U2, OP_U1, OP_MEASURE, OP_BARRIER

comptime RPPO_VW = 4
comptime RPPO_CHUNK_ROWS = 128
comptime RPPO_REWARD_GATE = 0.01  # reward per two-qubit gate executed
comptime RPPO_LOGP_EPS = 1e-12


# ------------------------------------------------------------------------------------------------ configuration


struct RoutePPOConfig(Copyable):
    """Options of the router PPO. Exactly one of `budget_s` (> 0) and `iters` (> 0) bounds a run (D7)."""

    var budget_s: Float64  # wall-clock budget, checked between iterations (0: not used)
    var iters: Int  # exact number of iterations (0: not used)
    var workers: Int  # threads (clamped to parallelism_level(); speed only)
    var seed: UInt64
    var episodes: Int  # episodes (jobs) per iteration
    var lr: Float64
    var clip: Float32
    var ent_coef: Float32
    var vf_coef: Float32
    var gamma: Float64
    var lam: Float64
    var epochs: Int
    var minibatch: Int
    var max_norm: Float64
    var max_n: Int  # largest circuit width of the job generator
    var start_iteration: Int  # global iteration offset (section 3 rule 1)
    var verbose: Bool  # print one JSON line per iteration

    def __init__(out self):
        """The constants of MOJO_PORT.md 19.1: 64 episodes, lr 2e-4, clip 0.1, vf 0.5, ent 0.005, gamma 0.99, lam
        0.95, 4 epochs, minibatch 1024, grad clip 0.5, circuits up to 24 qubits."""
        self.budget_s = 0.0
        self.iters = 0
        self.workers = 4
        self.seed = 0
        self.episodes = 64
        self.lr = 2e-4
        self.clip = 0.1
        self.ent_coef = 0.005
        self.vf_coef = 0.5
        self.gamma = 0.99
        self.lam = 0.95
        self.epochs = 4
        self.minibatch = 1024
        self.max_norm = 0.5
        self.max_n = 24
        self.start_iteration = 0
        self.verbose = False


def _e640(msg: String) -> Error:
    return Error("QM-E640: " + msg)


def _e643(msg: String) -> Error:
    return Error("QM-E643: " + msg)


def validate_config(cfg: RoutePPOConfig) raises:
    """QM-E640 unless exactly one of `budget_s` / `iters` is given and every constant is in range."""
    if (cfg.budget_s > 0.0) == (cfg.iters > 0):
        raise _e640("give exactly one of --budget-s and --iters. Hint: --iters N is exact and reproducible, --budget-s S stops by the clock.")
    if cfg.episodes < 1 or cfg.epochs < 1 or cfg.minibatch < 1:
        raise _e640("episodes, epochs and minibatch must be positive. Hint: defaults are 64, 4 and 1024.")
    if cfg.lr <= 0.0 or cfg.clip <= 0.0 or cfg.max_norm <= 0.0:
        raise _e640("lr, clip and the gradient-norm bound must be positive. Hint: defaults are 2e-4, 0.1 and 0.5.")
    if cfg.max_n < 3:
        raise _e640("max_n must be at least 3. Hint: the default is 24.")
    if cfg.gamma < 0.0 or cfg.gamma > 1.0 or cfg.lam < 0.0 or cfg.lam > 1.0:
        raise _e640("gamma and lam must be in [0, 1]. Hint: defaults are 0.99 and 0.95.")
    if cfg.workers < 1:
        raise _e640("workers must be at least 1.")
    if cfg.start_iteration < 0:
        raise _e640("start_iteration must be >= 0.")


def effective_workers(requested: Int) -> Int:
    """`min(requested, parallelism_level())`, at least 1. Speed only: results never depend on it."""
    return max(1, min(requested, parallelism_level()))


def curriculum_scale(git: Int) -> Float64:
    """Circuit-size scale of global iteration `git`: `min(1.5, 0.4 + 0.02 * git)`."""
    return min(1.5, 0.4 + 0.02 * Float64(git))


def job_build_seed(cfg: RoutePPOConfig, git: Int) -> Int:
    """Seed of the job set of global iteration `git`: `700000 + 1000 * seed + git`."""
    return 700000 + 1000 * Int(cfg.seed) + git


def episode_seed(cfg: RoutePPOConfig, git: Int, e: Int) -> UInt64:
    """Environment stream of episode `e`: seeds the random initial layout and the `Episode` itself."""
    return seed_for(cfg.seed, TAG_ROUTE_PPO, git, e, 0)


def actor_seed(cfg: RoutePPOConfig, git: Int, e: Int) -> UInt64:
    """Action-sampling stream of episode `e`."""
    return seed_for(cfg.seed, TAG_ROUTE_PPO, git, e, 1)


def shuffle_seed(cfg: RoutePPOConfig, git: Int, epoch: Int) -> UInt64:
    """Minibatch-shuffle stream of one epoch."""
    return seed_for(cfg.seed, TAG_ROUTE_PPO, git, epoch, 2)


# ------------------------------------------------------------------------------------------------ job conversion


def circuit_of_job(jc: JobCircuit) raises -> Circuit:
    """The routing circuit of a generated job circuit (op kinds as in `route_tool`'s jobs file: u2 is a two-qubit
    rotation, u1 a single-qubit gate); a `qasm` job circuit is loaded from its file."""
    if jc.qasm.byte_length() > 0:
        return load_qasm(jc.qasm)
    var nm = 0
    var k = 0
    while k + 2 < len(jc.ops):
        if jc.ops[k] == OP_MEASURE:
            nm += 1
        k += 3
    var c = Circuit(jc.n, nm)
    var m = 0
    k = 0
    while k + 2 < len(jc.ops):
        var kind = jc.ops[k]
        var a = jc.ops[k + 1]
        var b = jc.ops[k + 2]
        if kind == OP_CX:
            c.cx(a, b)
        elif kind == OP_CZ:
            c.cz(a, b)
        elif kind == OP_SWAP:
            c.swap(a, b)
        elif kind == OP_U2:
            c.rzz(0.3, a, b)
        elif kind == OP_U1:
            c.h(a)
        elif kind == OP_MEASURE:
            c.measure(a, m)
            m += 1
        elif kind == OP_BARRIER:
            c.barrier()
        else:
            raise Error("QM-E602: unknown op kind " + String(kind) + " in a job circuit. Hint: use cx cz swap u2 u1 measure barrier.")
        k += 3
    return c^


def target_of_topo(t: RTopo) raises -> Target:
    return Target.from_edges(t.n, t.edges, "any", t.name)


struct RouteJobs(Movable):
    """A job set converted to routing objects: one circuit per job circuit, the coupling targets that jobs use, and
    flat per-job (circuit index, target slot, router seed) triples."""

    var circuits: List[Circuit]
    var targets: List[Target]
    var jc: List[Int]
    var jt: List[Int]
    var js: List[Int]

    def __init__(out self):
        self.circuits = List[Circuit]()
        self.targets = List[Target]()
        self.jc = List[Int]()
        self.jt = List[Int]()
        self.js = List[Int]()

    def num_jobs(self) -> Int:
        return len(self.jc)


def route_jobs_of(js: JobSet) raises -> RouteJobs:
    var out = RouteJobs()
    for i in range(len(js.circuits)):
        out.circuits.append(circuit_of_job(js.circuits[i]))
    var slot = List[Int](length=len(js.targets), fill=-1)
    for j in range(js.num_jobs()):
        var ti = js.jobs[3 * j + 1]
        if slot[ti] < 0:
            slot[ti] = len(out.targets)
            out.targets.append(target_of_topo(js.targets[ti]))
        out.jc.append(js.jobs[3 * j])
        out.jt.append(slot[ti])
        out.js.append(js.jobs[3 * j + 2])
    return out^


# ------------------------------------------------------------------------------------------------ rollouts


struct _F64Net(Movable):
    """The router weights in Float64 for the rollout policy (the update uses the f32 network)."""

    var hidden: Int
    var emb: List[Float64]
    var b0: List[Float64]
    var w1: List[Float64]
    var b1: List[Float64]
    var wv: List[Float64]
    var bv: Float64

    def __init__(out self, net: RouterNet):
        var h = net.hidden
        self.hidden = h
        self.emb = List[Float64](capacity=ROUTER_OBS * h)
        self.b0 = List[Float64](capacity=h)
        self.w1 = List[Float64](capacity=ROUTER_SLOTS * h)
        self.b1 = List[Float64](capacity=ROUTER_SLOTS)
        self.wv = List[Float64](capacity=h)
        for i in range(ROUTER_OBS * h):
            self.emb.append(Float64(net.params[net.off_emb() + i]))
        for i in range(h):
            self.b0.append(Float64(net.params[net.off_b0() + i]))
        for i in range(ROUTER_SLOTS * h):
            self.w1.append(Float64(net.params[net.off_w1() + i]))
        for i in range(ROUTER_SLOTS):
            self.b1.append(Float64(net.params[net.off_b1() + i]))
        for i in range(h):
            self.wv.append(Float64(net.params[net.off_wv() + i]))
        self.bv = Float64(net.params[net.off_bv()])

    def forward(self, sg: List[Int8], A: Int, mut hid: List[Float64], mut p: List[Float64]) -> Float64:
        """Fills `hid` (post-ReLU hidden vector) and `p` (softmax over the first `A` slots); returns the value."""
        var h = self.hidden
        for j in range(h):
            hid[j] = self.b0[j]
        for i in range(ROUTER_OBS):
            var s = sg[i]
            if s > 0:
                for j in range(h):
                    hid[j] += self.emb[i * h + j]
            elif s < 0:
                for j in range(h):
                    hid[j] -= self.emb[i * h + j]
        for j in range(h):
            if hid[j] < 0.0:
                hid[j] = 0.0
        var mx = 0.0
        for k in range(A):
            var z = self.b1[k]
            for j in range(h):
                z += self.w1[k * h + j] * hid[j]
            p[k] = z
            if k == 0 or z > mx:
                mx = z
        var sum = 0.0
        for k in range(A):
            p[k] = exp(p[k] - mx)
            sum += p[k]
        for k in range(A):
            p[k] = p[k] / sum
        var v = self.bv
        for j in range(h):
            v += self.wv[j] * hid[j]
        return v


struct _Traj(Movable):
    """The decisions of one episode, in order."""

    var sign: List[Int8]  # [decisions, 128]
    var A: List[Int32]
    var act: List[Int32]
    var lp: List[Float64]
    var v: List[Float64]
    var r: List[Float64]
    var overhead: Int
    var steps: Int
    var status: Int  # 0 = ok, 643 = non-finite policy

    def __init__(out self):
        self.sign = List[Int8]()
        self.A = List[Int32]()
        self.act = List[Int32]()
        self.lp = List[Float64]()
        self.v = List[Float64]()
        self.r = List[Float64]()
        self.overhead = 0
        self.steps = 0
        self.status = 0


def sample_cdf(p: List[Float64], A: Int, u: Float64) -> Int:
    """Inverse-CDF draw over the first `A` probabilities: the cumulative sum is normalised by its last element and the
    result is the number of entries `<= u` (right-sided search), clamped to `A - 1`."""
    var cdf = List[Float64](length=A, fill=0.0)
    var run = 0.0
    for k in range(A):
        run += p[k]
        cdf[k] = run
    var last = cdf[A - 1]
    var idx = 0
    for k in range(A):
        if cdf[k] / last <= u:
            idx += 1
    return min(idx, A - 1)


def _run_route_episode(
    fw: _F64Net,
    circ: Circuit,
    target: Target,
    env_seed: UInt64,
    act_seed: UInt64,
    mut tr: _Traj,
):
    """One rollout episode with the sampling policy (see the module comment for the reward rule)."""
    var prob = RProblem(circ.ops, circ.num_qubits)
    var lay_rng = Rng(env_seed)
    var l2p = _random_layout(target.num_qubits, lay_rng)
    var ep = Episode(prob, target, l2p, env_seed, True)
    ep.rec_ops = False  # costs and decisions only
    var arng = Rng(act_seed)
    var limit = max(4, 2 * prob.n)
    var ms = episode_max_steps(prob, target)
    var hid = List[Float64](length=fw.hidden, fill=0.0)
    var pr = List[Float64](length=ROUTER_SLOTS, fill=0.0)
    var sg = List[Int8](length=ROUTER_OBS, fill=0)
    var pend = False
    var pend_ov = 0
    var pend_ex = 0
    while True:
        var r = ep.pre_decision(prob, target, limit)
        var done = r < 0 or ep.steps > ms
        if pend and (done or ep.needs_policy()):
            tr.r[len(tr.r) - 1] = -Float64(ep.overhead() - pend_ov) + RPPO_REWARD_GATE * Float64(ep.exec2q - pend_ex)
            pend = False
        if done:
            break
        if ep.needs_policy():
            var A = len(ep.active)
            for i in range(ROUTER_OBS):
                var o = ep.obs[i]
                var sv: Int8 = 0
                if o > 0:
                    sv = 1
                elif o < 0:
                    sv = -1
                sg[i] = sv
            var v = fw.forward(sg, A, hid, pr)
            var sm = 0.0
            for k in range(A):
                sm += pr[k]
            if not (sm == sm) or sm <= 0.0 or not (v == v) or v > 1e300 or v < -1e300:
                tr.status = 643
                break
            var a = sample_cdf(pr, A, arng.uniform())
            for i in range(ROUTER_OBS):
                tr.sign.append(sg[i])
            tr.A.append(Int32(A))
            tr.act.append(Int32(a))
            tr.lp.append(log(pr[a] + RPPO_LOGP_EPS))
            tr.v.append(v)
            tr.r.append(0.0)
            pend = True
            pend_ov = ep.overhead()
            pend_ex = ep.exec2q
            ep.act(prob, target, ep.active[a])
        else:
            var e = ep.fallback_swap(prob, target)
            ep.n_fallback += 1
            if e < 0:
                if pend:
                    tr.r[len(tr.r) - 1] = -Float64(ep.overhead() - pend_ov) + RPPO_REWARD_GATE * Float64(ep.exec2q - pend_ex)
                break
            ep.act(prob, target, e)
    tr.overhead = ep.overhead()
    tr.steps = ep.steps


struct RouteRollout(Movable):
    """One iteration's rollout, rows in episode order. `adv` is standardised (population std + 1e-8), `ret = raw adv + V`.
    The raw per-row reward / value / done arrays and the per-episode summaries are kept for logging and tests."""

    var n: Int
    var sign: List[Int8]  # [n, 128] in {-1, 0, 1}
    var A: List[Int32]  # valid active swaps per row
    var act: List[Int32]
    var logp_old: List[Float32]
    var adv: List[Float32]
    var ret: List[Float32]
    var reward: List[Float64]
    var value: List[Float64]
    var done: List[Bool]
    var ep_len: List[Int32]  # decisions per episode
    var ep_overhead: List[Int32]
    var ep_steps: List[Int32]
    var status: Int  # first non-zero episode status in episode order

    def __init__(out self):
        self.n = 0
        self.sign = List[Int8]()
        self.A = List[Int32]()
        self.act = List[Int32]()
        self.logp_old = List[Float32]()
        self.adv = List[Float32]()
        self.ret = List[Float32]()
        self.reward = List[Float64]()
        self.value = List[Float64]()
        self.done = List[Bool]()
        self.ep_len = List[Int32]()
        self.ep_overhead = List[Int32]()
        self.ep_steps = List[Int32]()
        self.status = 0

    def mean_reward(self) -> Float64:
        if self.n == 0:
            return 0.0
        var s = 0.0
        for i in range(self.n):
            s += self.reward[i]
        return s / Float64(self.n)

    def mean_overhead(self) -> Float64:
        var ne = len(self.ep_overhead)
        if ne == 0:
            return 0.0
        var s = 0.0
        for i in range(ne):
            s += Float64(self.ep_overhead[i])
        return s / Float64(ne)


def standardize(adv: List[Float64]) -> List[Float32]:
    """`(adv - mean) / (std + 1e-8)` with the population std, in Float64, rounded to f32."""
    var n = len(adv)
    var out = List[Float32](capacity=n)
    if n == 0:
        return out^
    var m = 0.0
    for i in range(n):
        m += adv[i]
    m /= Float64(n)
    var v = 0.0
    for i in range(n):
        v += (adv[i] - m) * (adv[i] - m)
    var sd = sqrt(v / Float64(n))
    for i in range(n):
        out.append(Float32((adv[i] - m) / (sd + 1e-8)))
    return out^


def collect_route(
    net: RouterNet, jobs: RouteJobs, cfg: RoutePPOConfig, git: Int
) raises -> RouteRollout:
    """Plays every job once with the sampling policy (Float64 weights of `net`) and returns the rollout with GAE and
    standardised advantages. Episode `e` uses `episode_seed` / `actor_seed` of `(cfg.seed, git, e)`; the result does
    not depend on `cfg.workers`. QM-E643 when a policy output is not finite."""
    var E = jobs.num_jobs()
    var ro = RouteRollout()
    if E == 0:
        return ro^
    var fw = _F64Net(net)
    var trajs = List[_Traj]()
    for _ in range(E):
        trajs.append(_Traj())

    def task(e: Int) {imm fw, imm jobs, imm cfg, imm git, mut trajs}:
        _run_route_episode(
            fw,
            jobs.circuits[jobs.jc[e]],
            jobs.targets[jobs.jt[e]],
            episode_seed(cfg, git, e),
            actor_seed(cfg, git, e),
            trajs[e],
        )

    if E == 1:
        task(0)
    else:
        parallelize(task, E, effective_workers(cfg.workers))
    var total = 0
    for e in range(E):
        total += len(trajs[e].act)
        if trajs[e].status != 0 and ro.status == 0:
            ro.status = trajs[e].status
    if ro.status != 0:
        raise _e643("the router policy produced a non-finite probability during rollouts. Hint: the init weights are corrupt or the learning rate diverged earlier.")
    ro.n = total
    ro.sign.reserve(total * ROUTER_OBS)
    for e in range(E):
        ref t = trajs[e]
        var L = len(t.act)
        ro.ep_len.append(Int32(L))
        ro.ep_overhead.append(Int32(t.overhead))
        ro.ep_steps.append(Int32(t.steps))
        for i in range(len(t.sign)):
            ro.sign.append(t.sign[i])
        for k in range(L):
            ro.A.append(t.A[k])
            ro.act.append(t.act[k])
            ro.logp_old.append(Float32(t.lp[k]))
            ro.reward.append(t.r[k])
            ro.value.append(t.v[k])
            ro.done.append(k == L - 1)
    var adv = List[Float64]()
    var ret = List[Float64]()
    if total > 0:
        gae_route(Span(ro.reward), Span(ro.value), Span(ro.done), cfg.gamma, cfg.lam, adv, ret)
    ro.adv = standardize(adv)
    for i in range(total):
        ro.ret.append(Float32(ret[i]))
    return ro^


# ------------------------------------------------------------------------------------------------ update


struct RoutePPOWorkspace(Movable):
    """Scratch of the update: one cache, one gradient buffer and one gathered-rows buffer per virtual worker, plus the
    reduced gradient of the last minibatch and the per-row ratios of the last minibatch."""

    var hidden: Int
    var params: Int
    var caches: List[RouterCache]
    var vw_grad: List[Float32]  # [RPPO_VW, params]
    var grad: List[Float32]  # [params]: the reduced gradient of the last minibatch
    var g_sign: List[Int8]  # [RPPO_VW, CHUNK * 128]
    var g_valid: List[Int8]  # [RPPO_VW, CHUNK * 16]
    var g_act: List[Int32]  # [RPPO_VW, CHUNK]
    var g_old: List[Float32]
    var g_adv: List[Float32]
    var g_ret: List[Float32]
    var last_ratio: List[Float32]  # [rows]: ratio of every row of the last minibatch, in batch order
    var part_pol: List[Float64]
    var part_val: List[Float64]
    var part_ent: List[Float64]

    def __init__(out self, hidden: Int, params: Int, max_rows: Int):
        self.hidden = hidden
        self.params = params
        self.caches = List[RouterCache]()
        for _ in range(RPPO_VW):
            self.caches.append(RouterCache(RPPO_CHUNK_ROWS, hidden))
        self.vw_grad = List[Float32](length=RPPO_VW * params, fill=0.0)
        self.grad = List[Float32](length=params, fill=0.0)
        self.g_sign = List[Int8](length=RPPO_VW * RPPO_CHUNK_ROWS * ROUTER_OBS, fill=0)
        self.g_valid = List[Int8](length=RPPO_VW * RPPO_CHUNK_ROWS * ROUTER_SLOTS, fill=0)
        self.g_act = List[Int32](length=RPPO_VW * RPPO_CHUNK_ROWS, fill=0)
        self.g_old = List[Float32](length=RPPO_VW * RPPO_CHUNK_ROWS, fill=0.0)
        self.g_adv = List[Float32](length=RPPO_VW * RPPO_CHUNK_ROWS, fill=0.0)
        self.g_ret = List[Float32](length=RPPO_VW * RPPO_CHUNK_ROWS, fill=0.0)
        var cap = max(max_rows, 1)
        self.last_ratio = List[Float32](length=cap, fill=0.0)
        var nch = (cap + RPPO_CHUNK_ROWS - 1) // RPPO_CHUNK_ROWS + 1
        self.part_pol = List[Float64](length=nch, fill=0.0)
        self.part_val = List[Float64](length=nch, fill=0.0)
        self.part_ent = List[Float64](length=nch, fill=0.0)

    def ensure(mut self, rows: Int):
        if rows > len(self.last_ratio):
            self.last_ratio = List[Float32](length=rows, fill=0.0)
        var nch = (rows + RPPO_CHUNK_ROWS - 1) // RPPO_CHUNK_ROWS + 1
        if nch > len(self.part_pol):
            self.part_pol = List[Float64](length=nch, fill=0.0)
            self.part_val = List[Float64](length=nch, fill=0.0)
            self.part_ent = List[Float64](length=nch, fill=0.0)


def _loss_cfg(cfg: RoutePPOConfig) -> RoutePPOCfg:
    return RoutePPOCfg(cfg.clip, cfg.vf_coef, cfg.ent_coef)


def ppo_minibatch_grad(
    net: RouterNet,
    mut ws: RoutePPOWorkspace,
    ro: RouteRollout,
    order: List[Int32],
    lo: Int,
    rows: Int,
    cfg: RoutePPOConfig,
    threads: Int,
) -> RouteLossParts:
    """Forward, loss and backward of the minibatch made of the rollout rows `order[lo : lo + rows]`, on the virtual
    workers. Leaves the reduced raw gradient (all six tensors, value head included) in `ws.grad` and the per-row ratios
    in `ws.last_ratio`; returns the loss sums over the minibatch (not yet divided by `rows`)."""
    ws.ensure(rows)
    var nchunks = (rows + RPPO_CHUNK_ROWS - 1) // RPPO_CHUNK_ROWS
    var P = ws.params
    var CH = RPPO_CHUNK_ROWS
    var pg = ws.vw_grad.unsafe_ptr().as_unsafe_any_origin()
    var gs = ws.g_sign.unsafe_ptr().as_unsafe_any_origin()
    var gv = ws.g_valid.unsafe_ptr().as_unsafe_any_origin()
    var ga = ws.g_act.unsafe_ptr().as_unsafe_any_origin()
    var go = ws.g_old.unsafe_ptr().as_unsafe_any_origin()
    var gd = ws.g_adv.unsafe_ptr().as_unsafe_any_origin()
    var gr = ws.g_ret.unsafe_ptr().as_unsafe_any_origin()
    var prat = ws.last_ratio.unsafe_ptr().as_unsafe_any_origin()
    var ppol = ws.part_pol.unsafe_ptr().as_unsafe_any_origin()
    var pval = ws.part_val.unsafe_ptr().as_unsafe_any_origin()
    var pent = ws.part_ent.unsafe_ptr().as_unsafe_any_origin()
    var rsign = ro.sign.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin()
    var rA = ro.A.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin()
    var ract = ro.act.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin()
    var rold = ro.logp_old.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin()
    var radv = ro.adv.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin()
    var rret = ro.ret.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin()
    var pord = order.unsafe_ptr().unsafe_mut_cast[True]().as_unsafe_any_origin()
    var lcfg = _loss_cfg(cfg)
    var inv_b = Float32(1.0) / Float32(rows)

    def task(w: Int) {imm net, mut ws, imm nchunks, imm P, imm CH, imm pg, imm gs, imm gv, imm ga, imm go, imm gd, imm gr, imm prat, imm ppol, imm pval, imm pent, imm rsign, imm rA, imm ract, imm rold, imm radv, imm rret, imm pord, imm lo, imm rows, imm lcfg, imm inv_b}:
        var gw = pg.unsafe_offset(w * P)
        fill_f32(gw, P, 0.0)
        var sgn = gs.unsafe_offset(w * CH * ROUTER_OBS)
        var vld = gv.unsafe_offset(w * CH * ROUTER_SLOTS)
        var acts = ga.unsafe_offset(w * CH)
        var old = go.unsafe_offset(w * CH)
        var adv = gd.unsafe_offset(w * CH)
        var ret = gr.unsafe_offset(w * CH)
        var c = w
        while c < nchunks:
            var clo = c * CH
            var n = min(CH, rows - clo)
            for r in range(n):
                var src = Int(pord[unsafe_offset = lo + clo + r])
                var srow = rsign.unsafe_offset(src * ROUTER_OBS)
                var drow = sgn.unsafe_offset(r * ROUTER_OBS)
                for i in range(ROUTER_OBS):
                    drow[unsafe_offset=i] = srow[unsafe_offset=i]
                var a_cnt = Int(rA[unsafe_offset=src])
                for k in range(ROUTER_SLOTS):
                    var vflag: Int8 = 0
                    if k < a_cnt:
                        vflag = 1
                    vld[unsafe_offset = r * ROUTER_SLOTS + k] = vflag
                acts[unsafe_offset=r] = ract[unsafe_offset=src]
                old[unsafe_offset=r] = rold[unsafe_offset=src]
                adv[unsafe_offset=r] = radv[unsafe_offset=src]
                ret[unsafe_offset=r] = rret[unsafe_offset=src]
            net.forward_chunk(sgn, n, ws.caches[w], True)
            var zp = ws.caches[w].z.unsafe_ptr().as_unsafe_any_origin()
            var vp = ws.caches[w].v.unsafe_ptr().as_unsafe_any_origin()
            var dzp = ws.caches[w].dz.unsafe_ptr().as_unsafe_any_origin()
            var dvp = ws.caches[w].dv.unsafe_ptr().as_unsafe_any_origin()
            var parts = route_ppo_chunk_loss(
                zp, vp, vld, n, ROUTER_SLOTS, acts, old, adv, ret, lcfg, inv_b, dzp, dvp, prat.unsafe_offset(clo)
            )
            ppol[unsafe_offset=c] = parts.policy
            pval[unsafe_offset=c] = parts.value
            pent[unsafe_offset=c] = parts.entropy
            net.backward_chunk(sgn, n, ws.caches[w], gw, True)
            c += RPPO_VW

    parallelize(task, RPPO_VW, threads)
    _ = len(ws.vw_grad)
    _ = len(ws.g_sign)
    _ = len(ws.g_valid)
    _ = len(ws.g_act)
    _ = len(ws.g_old)
    _ = len(ws.g_adv)
    _ = len(ws.g_ret)
    _ = len(ws.last_ratio)
    _ = len(ws.part_pol)
    _ = len(ws.part_val)
    _ = len(ws.part_ent)
    _ = len(ro.sign)
    _ = len(ro.A)
    _ = len(ro.act)
    _ = len(ro.logp_old)
    _ = len(ro.adv)
    _ = len(ro.ret)
    _ = len(order)
    # reduction in a fixed order: worker 0, 1, 2, 3; chunk partial sums in chunk order
    var g = ws.grad.unsafe_ptr().as_unsafe_any_origin()
    fill_f32(g, P, 0.0)
    for w in range(RPPO_VW):
        add_inplace(g, pg.unsafe_offset(w * P), P)
    var tp = 0.0
    var tv = 0.0
    var te = 0.0
    for c in range(nchunks):
        tp += ws.part_pol[c]
        tv += ws.part_val[c]
        te += ws.part_ent[c]
    _ = len(ws.grad)
    _ = len(ws.vw_grad)
    return RouteLossParts(tp, tv, te)


def ppo_apply_update(mut net: RouterNet, mut opt: Adam, mut ws: RoutePPOWorkspace, lr: Float64, max_norm: Float64) -> Float64:
    """Global gradient-norm clip over the six tensors (every gradient is multiplied by `min(max_norm / (norm + 1e-6),
    1)`, also when that is 1) and one Adam step with learning rate `lr` on the gradient in `ws.grad`. Returns the
    pre-clip norm."""
    var norm = clip_grad_norm(ws.grad, net.tensor_ranges(), max_norm)
    opt.step(net.params, ws.grad, lr)
    return norm


@fieldwise_init
struct RouteStepStats(Copyable):
    """Per-minibatch means: `policy = -mean min(s1, s2)`, `value = mean (v - ret)^2`, `entropy`, the pre-clip gradient
    norm and the fraction of rows whose ratio left `[1 - clip, 1 + clip]`."""

    var policy: Float64
    var value: Float64
    var entropy: Float64
    var grad_norm: Float64
    var clip_frac: Float64

    def total(self, cfg: RoutePPOConfig) -> Float64:
        return self.policy + Float64(cfg.vf_coef) * self.value - Float64(cfg.ent_coef) * self.entropy


def ppo_minibatch_step(
    mut net: RouterNet,
    mut opt: Adam,
    mut ws: RoutePPOWorkspace,
    ro: RouteRollout,
    order: List[Int32],
    lo: Int,
    rows: Int,
    cfg: RoutePPOConfig,
    lr: Float64,
    threads: Int,
) -> RouteStepStats:
    """One optimisation step on the minibatch `order[lo : lo + rows]`: `ppo_minibatch_grad`, then `ppo_apply_update`
    with `cfg.max_norm`. This is the entry the golden test drives with explicit minibatches."""
    var parts = ppo_minibatch_grad(net, ws, ro, order, lo, rows, cfg, threads)
    var norm = ppo_apply_update(net, opt, ws, lr, cfg.max_norm)
    var outside = 0
    var lo_r = 1.0 - Float64(cfg.clip)
    var hi_r = 1.0 + Float64(cfg.clip)
    for r in range(rows):
        var x = Float64(ws.last_ratio[r])
        if x < lo_r or x > hi_r:
            outside += 1
    var B = Float64(rows)
    return RouteStepStats(parts.policy / B, parts.value / B, parts.entropy / B, norm, Float64(outside) / B)


def route_ppo_update(
    mut net: RouterNet,
    mut opt: Adam,
    mut ws: RoutePPOWorkspace,
    ro: RouteRollout,
    cfg: RoutePPOConfig,
    git: Int,
) -> RouteStepStats:
    """`cfg.epochs` passes over the rollout; each pass shuffles the rows (`randperm` of `shuffle_seed`) and walks
    minibatches of `cfg.minibatch` rows (the last short one kept). Returns the row-weighted means of the last epoch."""
    var n = ro.n
    var threads = effective_workers(cfg.workers)
    var acc = RouteStepStats(0.0, 0.0, 0.0, 0.0, 0.0)
    if n == 0:
        return acc^
    for epoch in range(cfg.epochs):
        var rng = Rng(shuffle_seed(cfg, git, epoch))
        var perm = randperm(rng, n)
        var order = List[Int32](capacity=n)
        for i in range(n):
            order.append(Int32(perm[i]))
        acc = RouteStepStats(0.0, 0.0, 0.0, 0.0, 0.0)
        var b = 0
        while b < n:
            var rows = min(cfg.minibatch, n - b)
            var st = ppo_minibatch_step(net, opt, ws, ro, order, b, rows, cfg, cfg.lr, threads)
            var wgt = Float64(rows) / Float64(n)
            acc.policy += st.policy * wgt
            acc.value += st.value * wgt
            acc.entropy += st.entropy * wgt
            acc.grad_norm += st.grad_norm * wgt
            acc.clip_frac += st.clip_frac * wgt
            b += cfg.minibatch
    return acc^


# ------------------------------------------------------------------------------------------------ the trainer


@fieldwise_init
struct RoutePPOLog(Copyable):
    var iteration: Int  # global iteration
    var scale: Float64
    var decisions: Int
    var mean_overhead: Float64
    var mean_reward: Float64
    var policy_loss: Float64
    var value_loss: Float64
    var entropy: Float64
    var grad_norm: Float64
    var clip_frac: Float64
    var t: Float64


struct RoutePPOResult(Copyable):
    var log: List[RoutePPOLog]
    var iterations: Int
    var wall_s: Float64
    var error: String  # non-empty when the run stopped early (QM-E643); the returned weights are the last good ones

    def __init__(out self):
        self.log = List[RoutePPOLog]()
        self.iterations = 0
        self.wall_s = 0.0
        self.error = String()


def _all_finite(xs: List[Float32]) -> Bool:
    for i in range(len(xs)):
        var x = xs[i]
        if not (x == x) or x > 3.0e38 or x < -3.0e38:
            return False
    return True


def log_json(e: RoutePPOLog) -> String:
    """One iteration as a compact JSON object (the line `--verbose` prints and the entry of `train_log.json`)."""
    var s = String("{")
    s += '"iter":' + String(e.iteration)
    s += ',"scale":' + json_number(e.scale)
    s += ',"decisions":' + String(e.decisions)
    s += ',"mean_overhead":' + json_number(e.mean_overhead)
    s += ',"mean_reward":' + json_number(e.mean_reward)
    s += ',"policy_loss":' + json_number(e.policy_loss)
    s += ',"value_loss":' + json_number(e.value_loss)
    s += ',"entropy":' + json_number(e.entropy)
    s += ',"grad_norm":' + json_number(e.grad_norm)
    s += ',"clip_frac":' + json_number(e.clip_frac)
    s += ',"t":' + json_number(e.t)
    s += "}"
    return s^


def train_route_ppo(cfg: RoutePPOConfig, init: ModelFile) raises -> Tuple[RouterNet, RoutePPOResult]:
    """Fine-tunes the router `init` (a shipped f16 model decoded to f32 or a Mojo f32 run directory; a missing value
    head starts at zero) for `cfg.iters` iterations or until `cfg.budget_s` runs out (checked before each iteration).
    An iteration with no decision (every episode needs no policy) is logged and skipped. When a non-finite loss or
    weight appears the run stops at the weights of the iteration before and `result.error` carries the QM-E643 text."""
    validate_config(cfg)
    var t0 = perf_counter_ns()
    var net = RouterNet.from_model(init)
    var opt = Adam(net.param_count(), cfg.lr)
    var ws = RoutePPOWorkspace(net.hidden, net.param_count(), cfg.minibatch)
    var res = RoutePPOResult()
    var it = 0
    while True:
        var el = Float64(perf_counter_ns() - t0) / 1e9
        if cfg.iters > 0:
            if it >= cfg.iters:
                break
        elif el >= cfg.budget_s:
            break
        var git = cfg.start_iteration + it
        var scale = curriculum_scale(git)
        var js = build_jobs(job_build_seed(cfg, git), cfg.episodes, List[RTopo](), List[String](), cfg.max_n, scale, 0.4)
        var jobs = route_jobs_of(js)
        var ro = collect_route(net, jobs, cfg, git)
        var good = net.params.copy()
        var st = RouteStepStats(0.0, 0.0, 0.0, 0.0, 0.0)
        if ro.n > 0:
            st = route_ppo_update(net, opt, ws, ro, cfg, git)
            var finite = st.policy == st.policy and st.value == st.value and st.entropy == st.entropy and _all_finite(net.params)
            if not finite:
                net.params = good^
                res.error = (
                    "QM-E643: router PPO diverged (non-finite loss or weights in iteration " + String(git)
                    + "); stopped at the weights of the previous iteration. Hint: lower --lr or check the init model."
                )
                break
        var entry = RoutePPOLog(
            git,
            scale,
            ro.n,
            ro.mean_overhead(),
            ro.mean_reward(),
            st.policy,
            st.value,
            st.entropy,
            st.grad_norm,
            st.clip_frac,
            Float64(perf_counter_ns() - t0) / 1e9,
        )
        if cfg.verbose:
            print(log_json(entry))
        res.log.append(entry^)
        it += 1
    res.iterations = it
    res.wall_s = Float64(perf_counter_ns() - t0) / 1e9
    return (net^, res^)


# ------------------------------------------------------------------------------------------------ run directory


def _basename(p: String) -> String:
    var s = p
    while s.byte_length() > 1 and s.endswith("/"):
        var t = String(s[byte = : s.byte_length() - 1])
        s = t^
    var bs = s.as_bytes()
    var cut = -1
    for i in range(len(bs)):
        if Int(bs[i]) == ord("/"):
            cut = i
    if cut < 0:
        return s
    return String(s[byte = cut + 1 :])


def ppo_train_log_json(res: RoutePPOResult, cfg: RoutePPOConfig, init: String, hidden: Int) -> String:
    """`train_log.json` body (compact): {init, iterations, wall_s, workers, episodes_per_iter, hidden, seed,
    start_iteration, error, log[...]}."""
    var s = String("{")
    s += '"init":' + json_quote(_basename(init))
    s += ',"iterations":' + String(res.iterations)
    s += ',"wall_s":' + json_number(res.wall_s)
    s += ',"workers":' + String(cfg.workers)
    s += ',"episodes_per_iter":' + String(cfg.episodes)
    s += ',"hidden":' + String(hidden)
    s += ',"seed":' + String(cfg.seed)
    s += ',"start_iteration":' + String(cfg.start_iteration)
    s += ',"error":' + json_quote(res.error)
    s += ',"log":['
    for i in range(len(res.log)):
        if i > 0:
            s += ","
        s += log_json(res.log[i])
    s += "]}"
    return s^


def ppo_manifest_head(id: String, res: RoutePPOResult, cfg: RoutePPOConfig, init: String, hidden: Int) -> String:
    """The manifest members of a router-PPO run directory that precede `tensors` (`qumo.rlroute.v0`, no test vectors),
    in the layout of a behaviour-cloning run with `training.algo` set to the PPO stage."""
    var h = String()
    h += '"format":"qumo.rlroute.v0","id":' + json_quote(id) + ',"kind":"routing","num_qubits":0'
    h += ',"obs":{"size":128,"layout":"route_sign_h8_k16","encoding":"sign(obs[d + 8k])"}'
    h += (
        ',"arch":{"type":"sign_embedding_mlp","horizon":8,"num_active_swaps":16,"hidden":' + String(hidden)
        + ',"activation":"relu","value_head":true,"first_layer_transposed":true,'
        + '"forward":"logits = W1 relu(b0 + sum_i sign(obs_i) E[i,:]) + b1; argmax, lowest index on ties, over valid active swaps"}'
    )
    h += ',"producer":"trained by the native QUMO stack (qumo.rl, Mojo): PPO fine-tune of a behaviour-cloned router"'
    h += ',"license":"Apache-2.0"'
    h += ',"training":{"engine":"qumo.rl (Mojo)","algo":"router PPO fine-tune"'
    h += ',"iterations":' + String(res.iterations) + ',"episodes":' + String(cfg.episodes)
    h += ',"lr":' + json_number(cfg.lr) + ',"clip":' + String(cfg.clip)
    h += ',"vf_coef":' + String(cfg.vf_coef) + ',"ent_coef":' + String(cfg.ent_coef)
    h += ',"gamma":' + json_number(cfg.gamma) + ',"lam":' + json_number(cfg.lam)
    h += ',"epochs":' + String(cfg.epochs) + ',"minibatch":' + String(cfg.minibatch)
    h += ',"max_norm":' + json_number(cfg.max_norm) + ',"max_n":' + String(cfg.max_n)
    h += ',"seed":' + String(cfg.seed) + ',"start_iteration":' + String(cfg.start_iteration)
    h += ',"chunk_rows":' + String(RPPO_CHUNK_ROWS) + ',"vw":' + String(RPPO_VW)
    h += ',"wall_s":' + json_number(res.wall_s)
    h += ',"init":' + json_quote(init) + "}"
    return h^


def write_ppo_run(dir: String, net: RouterNet, res: RoutePPOResult, cfg: RoutePPOConfig, init: String) raises:
    """Writes a router-PPO run directory: a valid f32 `qumo.rlroute.v0` container (`manifest.json` + `weights.bin`,
    atomic, read back and checked by `write_model_dir`) and `train_log.json`. The run id is the directory's base
    name. `export_route` turns the run into the f16 model with test vectors."""
    _ = write_model_dir(
        dir,
        ppo_manifest_head(_basename(dir), res, cfg, init, net.hidden),
        net.to_tensors(),
        "f32",
        List[String](),
        "",
    )
    with open(dir + "/train_log.json", "w") as f:
        f.write(pretty_json(ppo_train_log_json(res, cfg, init, net.hidden), 1))
        f.write("\n")
