"""Optional router PPO fine-tune (native Mojo; never part of `make train-ai`).

usage: train_route_ppo --init DIR --out RUN (--budget-s S | --iters N) [--workers 4] [--seed 0] [--episodes 64]
                       [--lr 2e-4] [--clip 0.1] [--ent 0.005] [--vf 0.5] [--gamma 0.99] [--lam 0.95] [--epochs 4]
                       [--minibatch 1024] [--max-norm 0.5] [--max-n 24] [--start-iteration 0] [--quiet]

`--init DIR` is a router container: a shipped model, a `train_route` run or an earlier PPO run (the value head is
used when present and starts at zero otherwise). Each iteration plays `--episodes` routing episodes of generated
circuits on the 25 training topologies (size curriculum `scale = min(1.5, 0.4 + 0.02 * iteration)`) with the sampling
policy and updates the network with PPO (see qumo_bridge/route_ppo.mojo for the exact recipe). Exactly one of `--budget-s`
(stops between iterations by the clock) and `--iters` (exact and reproducible) bounds the run. `--workers` only
changes the speed, never the result. `--start-iteration` continues the global iteration counter of an earlier stage
(seed streams and the curriculum key on it).

Output: RUN is a valid `qumo.rlroute.v0` model directory in f32 (`manifest.json`, `weights.bin`, no test vectors) plus
`train_log.json` (one entry per iteration). The directory is written atomically (a stale `RUN.tmp` / `RUN.old` from an
interrupted run is recovered first). `export_route RUN --root ROOT --id ID` turns it into the f16 model with test
vectors; this stage never touches `models/`. On a non-finite loss the run stops at the last good weights, writes them
and exits non-zero.
"""

from std.sys import argv
from qrl.model import load_model_file
from qrl.container import recover_dir
from qumo_bridge.route_ppo import RoutePPOConfig, train_route_ppo, write_ppo_run, validate_config


def _usage() -> String:
    return (
        "usage: train_route_ppo --init DIR --out RUN (--budget-s S | --iters N) [--workers 4] [--seed 0] [--episodes 64] "
        "[--lr 2e-4] [--clip 0.1] [--ent 0.005] [--vf 0.5] [--gamma 0.99] [--lam 0.95] [--epochs 4] [--minibatch 1024] "
        "[--max-norm 0.5] [--max-n 24] [--start-iteration 0] [--quiet]"
    )


def _e640(msg: String) -> Error:
    return Error("QM-E640: " + msg + ". Hint: " + _usage())


def _need(args: List[String], i: Int, flag: String) raises -> String:
    if i + 1 >= len(args):
        raise _e640("option " + flag + " needs a value")
    return args[i + 1]


def main() raises:
    var raw = argv()
    var args = List[String]()
    for i in range(1, len(raw)):
        args.append(String(raw[i]))
    var init_dir = String()
    var out = String()
    var cfg = RoutePPOConfig()
    var quiet = False
    var i = 0
    while i < len(args):
        var a = args[i]
        if a == "--quiet":
            quiet = True
            i += 1
            continue
        var known: List[String] = [
            "--init", "--out", "--budget-s", "--budget_s", "--iters", "--workers", "--seed", "--episodes", "--lr",
            "--clip", "--ent", "--vf", "--gamma", "--lam", "--epochs", "--minibatch", "--max-norm", "--max-n",
            "--start-iteration",
        ]
        var ok = False
        for kf in known:
            if kf == a:
                ok = True
        if not ok:
            raise _e640("unknown option '" + a + "'")
        var v = _need(args, i, a)
        if a == "--init":
            init_dir = v
        elif a == "--out":
            out = v
        elif a == "--budget-s" or a == "--budget_s":
            cfg.budget_s = Float64(v)
        elif a == "--iters":
            cfg.iters = Int(v)
        elif a == "--workers":
            cfg.workers = Int(v)
        elif a == "--seed":
            cfg.seed = UInt64(Int(v))
        elif a == "--episodes":
            cfg.episodes = Int(v)
        elif a == "--lr":
            cfg.lr = Float64(v)
        elif a == "--clip":
            cfg.clip = Float32(Float64(v))
        elif a == "--ent":
            cfg.ent_coef = Float32(Float64(v))
        elif a == "--vf":
            cfg.vf_coef = Float32(Float64(v))
        elif a == "--gamma":
            cfg.gamma = Float64(v)
        elif a == "--lam":
            cfg.lam = Float64(v)
        elif a == "--epochs":
            cfg.epochs = Int(v)
        elif a == "--minibatch":
            cfg.minibatch = Int(v)
        elif a == "--max-norm":
            cfg.max_norm = Float64(v)
        elif a == "--max-n":
            cfg.max_n = Int(v)
        elif a == "--start-iteration":
            cfg.start_iteration = Int(v)
        i += 2
    if init_dir.byte_length() == 0 or out.byte_length() == 0:
        raise _e640("--init and --out are required")
    cfg.verbose = not quiet
    validate_config(cfg)

    recover_dir(out)
    recover_dir(init_dir)
    var init = load_model_file(init_dir)
    var trained = train_route_ppo(cfg, init)
    var res = trained[1].copy()
    print(
        "train_route_ppo: " + String(res.iterations) + " iterations, " + String(res.wall_s) + " s, hidden "
        + String(trained[0].hidden)
    )
    write_ppo_run(out, trained[0], res, cfg, init_dir)
    print("train_route_ppo: wrote " + out)
    if res.error.byte_length() > 0:
        raise Error(res.error)
