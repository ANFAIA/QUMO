"""Time- or iteration-bounded synthesis training driver: loop, checkpoints, resume, log and summary.

MOJO_PORT.md 7.5 and 10.1, component C11c. `run_synth_train(args)` is the whole of the `train_synth` binary:

    train_synth --kind perm|lf|clifford|pauli --topology SPEC --out RUN (--iters N | --budget-s S)
                [--cores 4] [--seed 1] [--emb 512] [--common 256[,..]] [--episodes 1024] [--diff-max 256]
                [--init DIR|auto] [--start-diff D] [--start-iteration N] [--resume] [--checkpoint-every 50]
                [--eval-episodes 200] [--sampled-eval-every 5] [--fidelity faithful|fixed] [--lr 3e-4] [--quiet]
                [--lane-log FILE] [--tag TEXT]

One iteration is `Trainer.learn_step` (evaluations of the unchanged policy, collection, ten full-batch Adam epochs),
then the curriculum promotion. The run directory is the layout of `qrl.checkpoint`:

    RUN/run.json      immutable configuration (kind, topology, gateset, architecture, seed, warm start, constants)
    RUN/best/         f32 container of the best weights: lexicographic maximum of (difficulty, deterministic success,
                      deterministic reward), `>=` so the latest of equal tuples wins; the faithful default stores the
                      weights AFTER the update of that iteration (`--fidelity faithful`), `--fidelity fixed` stores
                      the weights that were evaluated
    RUN/last/         latest weights, Adam moments and counters (the resume point), every `--checkpoint-every`
                      iterations and at the end
    RUN/log.jsonl     one JSON object per iteration
    RUN/summary.json  final record; `finished` is true only after the final checkpoint

Stop rule (D7): exactly one of `--iters` and `--budget-s`, else QM-E640. `--iters N` is a TOTAL for the run
directory: with `--resume` the iterations already done count, so a run stopped at 3 and resumed with `--iters 6`
ends bit-identical to an uninterrupted `--iters 6` run (every random stream derives from the seed, the global
iteration and the episode, nothing else is stateful). `--budget-s S` is cumulative through the elapsed time kept in
the checkpoint and is checked between iterations; the loop always runs at least one iteration. A budgeted run is a
deterministic prefix of the `--iters` run with the same seed, only the stopping point depends on the clock.

`--resume` continues RUN when it holds a `run.json` (starting from `last/`, or from the very beginning when the kill
came before the first checkpoint) and starts a fresh run otherwise. Result-determining options that are repeated on
the command line must equal the stored ones (QM-E640); `--cores`, `--budget-s`, `--iters`, `--checkpoint-every`,
`--quiet` and the lane options may change. A fresh run refuses a directory that already holds a run.

Warm start (`--init`): `auto` for a directory named `X__cont` (the sibling `X`), a run directory (its `best/`), or any
model container (a shipped f16 model works); see `checkpoint.resolve_init`. Without explicit `--emb` / `--common` the
architecture of the source is adopted; an incompatible source is QM-E647. The Adam state is fresh in a continuation.

Safety: a non-finite loss or weight (QM-E643) or an environment error (QM-E642) writes the last good state to `last/`
and an unfinished `summary.json`, then raises. Wall-clock time feeds the budget check and log fields only.
"""

from std.collections import List
from std.os import makedirs
from std.os.path import isfile
from std.time import perf_counter_ns
from qrl.json import JsonValue, parse_json, json_quote, json_number
from qrl.model import (
    MODEL_KIND_CLIFFORD,
    MODEL_KIND_LINEAR,
    MODEL_KIND_PAULI,
    MODEL_KIND_PERM,
    model_kind_name,
)
from qrl.checkpoint import (
    RunInfo,
    RunSummary,
    TrainState,
    WarmStart,
    new_run_info,
    net_spec_for,
    config_hash_of,
    write_run_json,
    read_run_json,
    write_best,
    write_last,
    read_checkpoint,
    has_checkpoint,
    read_train_state,
    load_adam_state,
    write_summary,
    read_summary,
    has_summary,
    append_log_line,
    truncate_log,
    resolve_init,
    BEST_DIR,
    LAST_DIR,
    RUN_FILE,
    LOG_FILE,
    STATE_FILE,
    SUMMARY_FILE,
)
from qrl.container import recover_dir, remove_tree
from qrl.f16 import max_abs_f32
from qrl.nn import BasicNet
from qrl.ppo import PPOConfig, Trainer, StepReport, effective_cores, global_iteration
from qrl.synth_env import TrainEnv
from qrl.tensor import CHUNK_ROWS, VW


# ------------------------------------------------------------------------------------------------ errors and paths


def _e640(msg: String) -> Error:
    return Error("QM-E640: " + msg + ". Hint: " + synth_train_usage())


def _e647(msg: String) -> Error:
    return Error("QM-E647: " + msg + ". Hint: the warm-start source must have the same gateset and architecture as the run.")


def synth_train_usage() -> String:
    return (
        "usage: train_synth --kind perm|lf|clifford|pauli --topology SPEC --out RUN (--iters N | --budget-s S) "
        "[--cores 4] [--seed 1] [--emb 512] [--common 256[,..]] [--episodes 1024] [--diff-max 256] "
        "[--init DIR|auto] [--start-diff D] [--start-iteration N] [--resume] [--checkpoint-every 50] "
        "[--eval-episodes 200] [--sampled-eval-every 5] [--fidelity faithful|fixed] [--lr 3e-4] [--quiet] "
        "[--lane-log FILE] [--tag TEXT]"
    )


def _strip(path: String) -> String:
    var p = path
    while p.byte_length() > 1 and p.endswith("/"):
        var q = String(unsafe_from_utf8=p.as_bytes()[0 : p.byte_length() - 1])
        p = q^
    return p^


def _basename(path: String) -> String:
    var p = _strip(path)
    var bs = p.as_bytes()
    var cut = -1
    for i in range(len(bs)):
        if Int(bs[i]) == ord("/"):
            cut = i
    if cut < 0:
        return p^
    return String(unsafe_from_utf8=bs[cut + 1 :])


def _parent(path: String) -> String:
    var p = _strip(path)
    var bs = p.as_bytes()
    var cut = -1
    for i in range(len(bs)):
        if Int(bs[i]) == ord("/"):
            cut = i
    if cut < 0:
        return String(".")
    if cut == 0:
        return String("/")
    return String(unsafe_from_utf8=bs[0:cut])


def _read_text(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def _round(x: Float64, digits: Int) -> Float64:
    var scale = 1.0
    for _ in range(digits):
        scale *= 10.0
    return Float64(Int(x * scale + 0.5)) / scale


# ------------------------------------------------------------------------------------------------ options


struct SynthArgs(Copyable):
    """The parsed command line. `given` lists the long options that were present (without the dashes), so a resume can
    tell an explicit value from a default."""

    var kind: String
    var topology: String
    var outdir: String
    var iters: Int  # 0 = not given
    var budget_s: Float64  # < 0 = not given
    var cores: Int
    var seed: Int
    var emb: Int
    var common: List[Int]
    var episodes: Int
    var diff_max: Int
    var init: String
    var start_diff: Int  # -1 = not given
    var start_iteration: Int  # -1 = not given
    var resume: Bool
    var checkpoint_every: Int
    var eval_episodes: Int
    var sampled_every: Int
    var fidelity: String
    var lr: Float64
    var quiet: Bool
    var lane_log: String
    var tag: String
    var given: List[String]

    def __init__(out self):
        self.kind = String()
        self.topology = String()
        self.outdir = String()
        self.iters = 0
        self.budget_s = -1.0
        self.cores = 4
        self.seed = 1
        self.emb = 512
        self.common = List[Int]()
        self.common.append(256)
        self.episodes = 1024
        self.diff_max = 256
        self.init = String()
        self.start_diff = -1
        self.start_iteration = -1
        self.resume = False
        self.checkpoint_every = 50
        self.eval_episodes = 200
        self.sampled_every = 5
        self.fidelity = String("faithful")
        self.lr = 3e-4
        self.quiet = False
        self.lane_log = String()
        self.tag = String()
        self.given = List[String]()

    def was_given(self, name: String) -> Bool:
        for i in range(len(self.given)):
            if self.given[i] == name:
                return True
        return False


def _int_opt(name: String, v: String, lo: Int) raises -> Int:
    var x: Int
    try:
        x = Int(v)
    except:
        raise _e640("option --" + name + " needs an integer, got '" + v + "'")
    if x < lo:
        raise _e640("option --" + name + " must be >= " + String(lo) + ", got " + String(x))
    return x


def _float_opt(name: String, v: String) raises -> Float64:
    var x: Float64
    try:
        x = Float64(v)
    except:
        raise _e640("option --" + name + " needs a number, got '" + v + "'")
    return x


def _parse_common(v: String) raises -> List[Int]:
    var res = List[Int]()
    var parts = v.split(",")
    for i in range(len(parts)):
        var w = _int_opt("common", String(parts[i]), 1)
        res.append(w)
    if len(res) == 0:
        raise _e640("option --common needs at least one layer width")
    return res^


def parse_synth_args(args: List[String]) raises -> SynthArgs:
    """Parses and validates the command line (QM-E640 on anything wrong, including both or neither of `--iters` and
    `--budget-s`)."""
    var a = SynthArgs()
    var i = 0
    while i < len(args):
        var o = args[i]
        if not o.startswith("--"):
            raise _e640("unexpected argument '" + o + "'")
        var name = String(unsafe_from_utf8=o.as_bytes()[2:])
        if name == "resume":
            a.resume = True
            a.given.append(name^)
            i += 1
            continue
        if name == "quiet":
            a.quiet = True
            a.given.append(name^)
            i += 1
            continue
        if i + 1 >= len(args):
            raise _e640("option " + o + " needs a value")
        var v = args[i + 1]
        if name == "kind":
            a.kind = v
        elif name == "topology":
            a.topology = v
        elif name == "out":
            a.outdir = v
        elif name == "iters":
            a.iters = _int_opt(name, v, 1)
        elif name == "budget-s":
            a.budget_s = _float_opt(name, v)
            if a.budget_s <= 0.0:
                raise _e640("option --budget-s must be positive, got " + v)
        elif name == "cores":
            a.cores = _int_opt(name, v, 1)
        elif name == "seed":
            a.seed = _int_opt(name, v, 0)
        elif name == "emb":
            a.emb = _int_opt(name, v, 1)
        elif name == "common":
            a.common = _parse_common(v)
        elif name == "episodes":
            a.episodes = _int_opt(name, v, 1)
        elif name == "diff-max":
            a.diff_max = _int_opt(name, v, 1)
        elif name == "init":
            a.init = v
        elif name == "start-diff":
            a.start_diff = _int_opt(name, v, 1)
        elif name == "start-iteration":
            a.start_iteration = _int_opt(name, v, 0)
        elif name == "checkpoint-every":
            a.checkpoint_every = _int_opt(name, v, 1)
        elif name == "eval-episodes":
            a.eval_episodes = _int_opt(name, v, 1)
        elif name == "sampled-eval-every":
            a.sampled_every = _int_opt(name, v, 1)
        elif name == "fidelity":
            if v != "faithful" and v != "fixed":
                raise _e640("option --fidelity must be 'faithful' or 'fixed', got '" + v + "'")
            a.fidelity = v
        elif name == "lr":
            a.lr = _float_opt(name, v)
            if not (a.lr > 0.0):
                raise _e640("option --lr must be positive, got " + v)
        elif name == "lane-log":
            a.lane_log = v
        elif name == "tag":
            a.tag = v
        else:
            raise _e640("unknown option '" + o + "'")
        a.given.append(name^)
        i += 2
    if a.outdir.byte_length() == 0:
        raise _e640("--out is required")
    if a.kind.byte_length() == 0 and not a.resume:
        raise _e640("--kind is required")
    if a.topology.byte_length() == 0 and not a.resume:
        raise _e640("--topology is required")
    if a.kind.byte_length() > 0:
        _ = kind_code_of(a.kind)
    var have_iters = a.iters > 0
    var have_budget = a.budget_s > 0.0
    if have_iters == have_budget:
        raise _e640("exactly one of --iters and --budget-s must be given")
    return a^


def kind_code_of(name: String) raises -> Int:
    if name == "perm" or name == "permutation":
        return MODEL_KIND_PERM
    if name == "lf" or name == "linear" or name == "linear_function":
        return MODEL_KIND_LINEAR
    if name == "clifford":
        return MODEL_KIND_CLIFFORD
    if name == "pauli" or name == "pauli_network":
        return MODEL_KIND_PAULI
    raise _e640("--kind must be perm, lf, clifford or pauli, got '" + name + "'")


# ------------------------------------------------------------------------------------------------ constants of the run


struct Hyper(Copyable):
    """The PPO and evaluation constants of a run, stored in `run.json` under `config` (so the resume hash covers them).
    The constants the CLI does not expose are the reference values of MOJO_PORT.md 19.1."""

    var episodes: Int
    var eval_episodes: Int
    var eval_searches: Int
    var sampled_every: Int
    var diff_max: Int
    var diff_threshold: Float32
    var gamma: Float32
    var lam: Float32
    var epochs: Int
    var vf_coef: Float32
    var ent_coef: Float32
    var clip: Float32
    var lr: Float64
    var fidelity: String

    def __init__(out self):
        var d = PPOConfig()
        self.episodes = d.num_episodes
        self.eval_episodes = d.eval_episodes
        self.eval_searches = d.eval_sampled_searches
        self.sampled_every = d.eval_sampled_every
        self.diff_max = d.diff_max
        self.diff_threshold = d.diff_threshold
        self.gamma = d.gamma
        self.lam = d.lam
        self.epochs = d.num_epochs
        self.vf_coef = d.vf_coef
        self.ent_coef = d.ent_coef
        self.clip = d.clip_ratio
        self.lr = d.lr
        self.fidelity = String("faithful")


def hyper_json(h: Hyper) -> String:
    var s = String("{")
    s += '"algo":"ppo"'
    s += ',"episodes":' + String(h.episodes)
    s += ',"eval_episodes":' + String(h.eval_episodes)
    s += ',"eval_sampled_searches":' + String(h.eval_searches)
    s += ',"eval_sampled_every":' + String(h.sampled_every)
    s += ',"diff_max":' + String(h.diff_max)
    s += ',"diff_threshold":' + json_number(Float64(h.diff_threshold))
    s += ',"gamma":' + json_number(Float64(h.gamma))
    s += ',"lam":' + json_number(Float64(h.lam))
    s += ',"epochs":' + String(h.epochs)
    s += ',"vf_coef":' + json_number(Float64(h.vf_coef))
    s += ',"ent_coef":' + json_number(Float64(h.ent_coef))
    s += ',"clip":' + json_number(Float64(h.clip))
    s += ',"lr":' + json_number(h.lr)
    s += ',"fidelity":' + json_quote(h.fidelity)
    s += "}"
    return s^


def parse_hyper(text: String) raises -> Hyper:
    var h = Hyper()
    if text.byte_length() == 0:
        return h^
    var j = parse_json(text)
    h.episodes = j.get_int("episodes", h.episodes)
    h.eval_episodes = j.get_int("eval_episodes", h.eval_episodes)
    h.eval_searches = j.get_int("eval_sampled_searches", h.eval_searches)
    h.sampled_every = j.get_int("eval_sampled_every", h.sampled_every)
    h.diff_max = j.get_int("diff_max", h.diff_max)
    h.diff_threshold = Float32(j.get_float("diff_threshold", Float64(h.diff_threshold)))
    h.gamma = Float32(j.get_float("gamma", Float64(h.gamma)))
    h.lam = Float32(j.get_float("lam", Float64(h.lam)))
    h.epochs = j.get_int("epochs", h.epochs)
    h.vf_coef = Float32(j.get_float("vf_coef", Float64(h.vf_coef)))
    h.ent_coef = Float32(j.get_float("ent_coef", Float64(h.ent_coef)))
    h.clip = Float32(j.get_float("clip", Float64(h.clip)))
    h.lr = j.get_float("lr", h.lr)
    h.fidelity = j.get_string("fidelity", "faithful")
    return h^


def ppo_config_for(h: Hyper, info: RunInfo) -> PPOConfig:
    """The `PPOConfig` of a run: the constants of `h`, the run's seed, chunking and global iteration offset, and the
    effective core count of this process (speed only; results never depend on it)."""
    var c = PPOConfig()
    c.num_cores = effective_cores(info.cores)
    c.num_episodes = h.episodes
    c.gamma = h.gamma
    c.lam = h.lam
    c.num_epochs = h.epochs
    c.vf_coef = h.vf_coef
    c.ent_coef = h.ent_coef
    c.clip_ratio = h.clip
    c.normalize_advantage = False
    c.lr = h.lr
    c.diff_threshold = h.diff_threshold
    c.diff_max = h.diff_max
    c.eval_episodes = h.eval_episodes
    c.eval_sampled_searches = h.eval_searches
    c.eval_sampled_every = h.sampled_every
    c.seed = UInt64(info.seed)
    c.chunk_rows = info.chunk_rows
    c.start_iteration = info.start_iteration
    if h.fidelity == "fixed":
        c.mask_new_logits = True
        c.best_checkpoint = String("evaluated")
    else:
        c.mask_new_logits = False
        c.best_checkpoint = String("post_update")
    return c^


# ------------------------------------------------------------------------------------------------ run setup


def _same_ints(a: List[Int], b: List[Int]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def _check_resume_flags(a: SynthArgs, info: RunInfo, h: Hyper) raises:
    """A repeated result-determining option must equal the stored one."""
    var bad = String()
    if a.was_given("kind") and kind_code_of(a.kind) != info.kind:
        bad = "--kind " + a.kind + " (the run is " + model_kind_name(info.kind) + ")"
    elif a.was_given("topology") and a.topology != info.topology:
        bad = "--topology " + a.topology + " (the run is " + info.topology + ")"
    elif a.was_given("seed") and a.seed != info.seed:
        bad = "--seed " + String(a.seed) + " (the run uses " + String(info.seed) + ")"
    elif a.was_given("emb") and a.emb != info.emb:
        bad = "--emb " + String(a.emb) + " (the run uses " + String(info.emb) + ")"
    elif a.was_given("common") and not _same_ints(a.common, info.common):
        bad = "--common (the run uses a different architecture)"
    elif a.was_given("episodes") and a.episodes != h.episodes:
        bad = "--episodes " + String(a.episodes) + " (the run uses " + String(h.episodes) + ")"
    elif a.was_given("diff-max") and a.diff_max != h.diff_max:
        bad = "--diff-max " + String(a.diff_max) + " (the run uses " + String(h.diff_max) + ")"
    elif a.was_given("eval-episodes") and a.eval_episodes != h.eval_episodes:
        bad = "--eval-episodes " + String(a.eval_episodes) + " (the run uses " + String(h.eval_episodes) + ")"
    elif a.was_given("sampled-eval-every") and a.sampled_every != h.sampled_every:
        bad = "--sampled-eval-every " + String(a.sampled_every) + " (the run uses " + String(h.sampled_every) + ")"
    elif a.was_given("fidelity") and a.fidelity != h.fidelity:
        bad = "--fidelity " + a.fidelity + " (the run uses " + h.fidelity + ")"
    elif a.was_given("lr") and a.lr != h.lr:
        bad = "--lr " + String(a.lr) + " (the run uses " + String(h.lr) + ")"
    if bad.byte_length() > 0:
        raise _e640("cannot resume '" + a.outdir + "' with " + bad + ": the options that determine the result must not change")


def _weights_of(m_dir: String, spec_like: BasicNet) raises -> List[Float32]:
    """Parameters of the model container `m_dir` in the layout of `spec_like` (QM-E647 when it does not fit)."""
    var m = read_checkpoint(m_dir)
    var src = BasicNet.from_model(m)
    if len(src.ps.names) != len(spec_like.ps.names) or len(src.ps.data) != len(spec_like.ps.data):
        raise _e647(
            "'" + m_dir + "' has a different architecture (" + String(len(src.ps.data)) + " parameters, the run needs "
            + String(len(spec_like.ps.data)) + ")"
        )
    for i in range(len(src.ps.names)):
        if src.ps.names[i] != spec_like.ps.names[i] or src.ps.rows_of(i) != spec_like.ps.rows_of(i) or src.ps.cols_of(i) != spec_like.ps.cols_of(i):
            raise _e647("tensor '" + src.ps.names[i] + "' of '" + m_dir + "' does not match the run's layout")
    return src.ps.data.copy()


def _discard_partial(rd: String) raises:
    """Resume found a run.json but no checkpoint: the run restarts from its first iteration."""
    remove_tree(rd + "/" + BEST_DIR)
    remove_tree(rd + "/" + LAST_DIR)
    remove_tree(rd + "/" + LOG_FILE)
    remove_tree(rd + "/" + STATE_FILE)
    remove_tree(rd + "/" + SUMMARY_FILE)


struct Driver(Movable):
    """Everything one training process owns."""

    var a: SynthArgs
    var rd: String
    var info: RunInfo
    var hy: Hyper
    var st: TrainState
    var tr: Trainer
    var base_elapsed: Float64
    var t0: Int
    var last_ck_it: Int  # st.it of the last written last/ checkpoint (-1 = none yet in this process)

    def __init__(
        out self,
        var a: SynthArgs,
        out_dir: String,
        var info: RunInfo,
        var hy: Hyper,
        var st: TrainState,
        var tr: Trainer,
    ):
        self.a = a^
        self.rd = out_dir
        self.info = info^
        self.hy = hy^
        self.st = st^
        self.tr = tr^
        self.base_elapsed = self.st.elapsed_s
        self.t0 = Int(perf_counter_ns())
        self.last_ck_it = -1

    def elapsed(self) -> Float64:
        return self.base_elapsed + Float64(Int(perf_counter_ns()) - self.t0) * 1e-9

    def reached(self) -> Bool:
        if self.a.iters > 0:
            return self.st.it >= self.a.iters
        return self.elapsed() >= self.a.budget_s

    def checkpoint(mut self) raises:
        self.st.elapsed_s = self.elapsed()
        self.st.difficulty = self.tr.env.difficulty
        _ = write_last(self.rd, self.tr.net, self.tr.opt, self.info, self.st)
        self.last_ck_it = self.st.it
        if self.a.lane_log.byte_length() > 0:
            var line = String("{")
            line += '"tag":' + json_quote(self.a.tag)
            line += ',"run":' + json_quote(self.rd)
            line += ',"it":' + String(self.st.it)
            line += ',"difficulty":' + String(self.st.difficulty)
            line += ',"t":' + json_number(_round(self.st.elapsed_s, 2))
            line += "}"
            with open(self.a.lane_log, "a") as f:
                f.write(line)
                f.write("\n")

    def log_line(self, rep: StepReport, step_index: Int) -> String:
        var s = String("{")
        s += '"it":' + String(step_index)
        s += ',"git":' + String(global_iteration(self.tr.cfg, step_index))
        s += ',"t":' + json_number(_round(self.elapsed(), 2))
        s += ',"step_s":' + json_number(_round(rep.step_s, 3))
        s += ',"difficulty":' + String(rep.difficulty)
        s += ',"succ":' + json_number(Float64(rep.det.success))
        s += ',"rew":' + json_number(Float64(rep.det.reward))
        if rep.have_sampled:
            s += ',"succ_sampled":' + json_number(Float64(rep.sampled.success))
            s += ',"rew_sampled":' + json_number(Float64(rep.sampled.reward))
        s += ',"loss":{"total":' + json_number(rep.loss.total)
        s += ',"policy":' + json_number(rep.loss.policy)
        s += ',"value":' + json_number(rep.loss.value)
        s += ',"entropy":' + json_number(rep.loss.entropy) + "}"
        s += ',"rows":' + String(rep.rows)
        s += "}"
        return s^

    def step(mut self) raises:
        """One iteration: learn_step, promotion, best checkpoint, log line and the periodic last/ checkpoint."""
        var rep = self.tr.learn_step()
        var step_index = self.st.it  # 0-based index of the iteration that just completed
        self.st.it += 1
        var improved = not self.st.have_best
        if not improved:
            if rep.difficulty != self.st.best_difficulty:
                improved = rep.difficulty > self.st.best_difficulty
            elif rep.det.success != self.st.best_success:
                improved = rep.det.success > self.st.best_success
            else:
                improved = rep.det.reward >= self.st.best_reward
        _ = self.tr.promote_if_ready(rep)
        self.st.difficulty = self.tr.env.difficulty
        self.st.elapsed_s = self.elapsed()
        if improved:
            self.st.have_best = True
            self.st.best_difficulty = rep.difficulty
            self.st.best_success = rep.det.success
            self.st.best_reward = rep.det.reward
            if self.tr.cfg.best_checkpoint == "evaluated":
                var keep = self.tr.net.ps.data.copy()
                self.tr.net.ps.data = self.tr.evaluated.copy()
                try:
                    _ = write_best(self.rd, self.tr.net, self.info, self.st)
                except e:
                    self.tr.net.ps.data = keep^
                    raise Error(String(e))
                self.tr.net.ps.data = keep^
            else:
                _ = write_best(self.rd, self.tr.net, self.info, self.st)
        append_log_line(self.rd, self.log_line(rep, step_index))
        if not self.a.quiet and step_index % 25 == 0:
            print(
                '{"it":' + String(step_index) + ',"t":' + json_number(_round(self.elapsed(), 2)) + ',"difficulty":'
                + String(rep.difficulty) + ',"succ":' + json_number(Float64(rep.det.success)) + "}",
                flush=True,
            )
        if self.st.it % self.a.checkpoint_every == 0:
            self.checkpoint()

    def summary(self, finished: Bool) -> RunSummary:
        var s = RunSummary()
        s.finished = finished
        s.iterations = self.st.it
        s.start_iteration = self.info.start_iteration
        s.final_difficulty = self.tr.env.difficulty
        s.best_difficulty = self.st.best_difficulty
        s.best_success = Float64(self.st.best_success)
        s.best_reward = Float64(self.st.best_reward)
        s.wall_s = _round(self.elapsed(), 1)
        s.budget_s = self.a.budget_s if self.a.budget_s > 0.0 else 0.0
        s.cores = effective_cores(self.info.cores)
        s.seed = self.info.seed
        s.topology = self.info.topology
        s.init = self.info.init
        return s^


def _open_driver(a: SynthArgs) raises -> Driver:
    """Builds the Driver for a fresh run, a resumed run or a warm-started run, and writes `run.json`."""
    var rd = _strip(a.outdir)
    makedirs(rd, exist_ok=True)
    recover_dir(rd + "/" + BEST_DIR)
    recover_dir(rd + "/" + LAST_DIR)
    var info: RunInfo
    var hy = Hyper()
    var st = TrainState()
    var resumed = False
    var have_run = isfile(rd + "/" + RUN_FILE)
    if a.resume and have_run:
        info = read_run_json(rd)
        hy = parse_hyper(info.config_json)
        _check_resume_flags(a, info, hy)
        if has_checkpoint(rd + "/" + LAST_DIR):
            var ck = read_checkpoint(rd + "/" + LAST_DIR)
            st = read_train_state(ck)
            if st.config_hash != config_hash_of(info):
                raise _e640(
                    "'" + rd + "/last' was written for a different configuration (hash " + st.config_hash
                    + ", the run is " + config_hash_of(info) + ")"
                )
            resumed = True
        else:
            _discard_partial(rd)
            st.start_iteration = info.start_iteration
            st.difficulty = info.start_difficulty
            st.seed = info.seed
            st.config_hash = config_hash_of(info)
    else:
        if have_run:
            raise _e640(
                "'" + rd + "' already holds a run (run.json). Use --resume to continue it or choose another --rd"
            )
        info = new_run_info(_basename(rd), kind_code_of(a.kind), a.topology)
        info.seed = a.seed
        info.chunk_rows = CHUNK_ROWS
        info.vw = VW
        info.emb = a.emb
        info.common = a.common.copy()
        hy.episodes = a.episodes
        hy.eval_episodes = a.eval_episodes
        hy.sampled_every = a.sampled_every
        hy.diff_max = a.diff_max
        hy.fidelity = a.fidelity
        hy.lr = a.lr
        if a.init.byte_length() > 0:
            var ws = resolve_init(a.init, rd, a.start_diff, a.start_iteration)
            if not a.was_given("emb") and not a.was_given("common"):
                var src = BasicNet.from_model(read_checkpoint(ws.model_dir))
                info.emb = src.spec.emb
                info.common = src.spec.common.copy()
            info.init = a.init
            info.start_difficulty = ws.start_difficulty
            info.start_iteration = ws.start_iteration
            if ws.is_run:
                var run_dir = _parent(ws.model_dir)
                if has_summary(run_dir):
                    info.init_final_difficulty = read_summary(run_dir).final_difficulty
        else:
            info.start_difficulty = 1
            info.start_iteration = 0
        info.config_json = hyper_json(hy)
        st.start_iteration = info.start_iteration
        st.difficulty = info.start_difficulty
        st.seed = info.seed
        st.config_hash = config_hash_of(info)
    # the fields that may change on a resume (they do not enter the hash)
    info.cores = effective_cores(a.cores)
    info.budget_s = a.budget_s if a.budget_s > 0.0 else 0.0
    info.iters = a.iters
    write_run_json(rd, info)

    var spec = net_spec_for(info)
    var net = BasicNet(spec, UInt64(info.seed))
    if resumed:
        net.ps.data = _weights_of(rd + "/" + LAST_DIR, net)
    elif info.init.byte_length() > 0:
        var ws2 = resolve_init(info.init, rd, info.start_difficulty, info.start_iteration)
        net.ps.data = _weights_of(ws2.model_dir, net)
    var env = TrainEnv(info.env_config(), UInt64(info.seed))
    env.set_difficulty(st.difficulty)
    var cfg = ppo_config_for(hy, info)
    var tr = Trainer(env, net^, cfg)
    if resumed:
        tr.iteration = st.it
        load_adam_state(tr.opt, read_checkpoint(rd + "/" + LAST_DIR))
        truncate_log(rd, st.it)
    var d = Driver(a.copy(), rd, info^, hy^, st^, tr^)
    return d^


def train_synth(a: SynthArgs) raises -> RunSummary:
    """Runs one training process as described in the module docstring and returns the final summary (also written to
    `RUN/summary.json`)."""
    var d = _open_driver(a)
    var resumed = d.st.it > 0
    if not a.quiet:
        print(
            '{"kind":' + json_quote(model_kind_name(d.info.kind)) + ',"topology":' + json_quote(d.info.topology)
            + ',"obs":' + String(d.tr.net.spec.obs_size) + ',"actions":' + String(d.tr.net.spec.num_actions)
            + ',"n_sym":' + String(d.tr.env.num_syms()) + ',"params":' + String(d.tr.net.param_count())
            + ',"cores":' + String(effective_cores(d.info.cores)) + ',"seed":' + String(d.info.seed)
            + ',"start_iteration":' + String(d.info.start_iteration) + ',"start_difficulty":' + String(d.st.difficulty)
            + ',"resumed_at":' + String(d.st.it) + "}",
            flush=True,
        )
    var stop = d.reached() if resumed else False
    try:
        while not stop:
            d.step()
            stop = d.reached()
    except e:
        # keep the last good state: after a QM-E643 the trainer has already restored weights and Adam state
        try:
            d.checkpoint()
            write_summary(d.rd, d.summary(False))
        except:
            pass
        raise Error(String(e))
    if d.last_ck_it != d.st.it:
        d.checkpoint()
    var worst = max_abs_f32(Span(d.tr.net.ps.data))
    if worst > 60000.0:
        print(
            "warning: max|w| = " + String(worst) + " exceeds 60000: an f16 export would overflow (QM-E645); export with --dtype f32",
            flush=True,
        )
    var summ = d.summary(True)
    write_summary(d.rd, summ)
    if not a.quiet:
        print("DONE " + String(_summary_line(summ)), flush=True)
    return summ^


def _summary_line(s: RunSummary) -> String:
    return (
        '{"iterations":' + String(s.iterations) + ',"final_difficulty":' + String(s.final_difficulty)
        + ',"best_difficulty":' + String(s.best_difficulty) + ',"best_success":' + json_number(s.best_success)
        + ',"wall_s":' + json_number(s.wall_s) + ',"seed":' + String(s.seed) + "}"
    )


def run_synth_train(args: List[String]) raises -> Int:
    """Entry of `train_synth` over its command-line arguments (10.1); returns the process exit code.
    Exactly one of --iters / --budget-s, else QM-E640."""
    var a = parse_synth_args(args)
    _ = train_synth(a)
    return 0
