"""Train QUMO's router policy by behaviour cloning / DAgger on rollout-teacher soft labels (native Mojo).

usage: train_route --data a.bin [b.bin ...] --out RUN [--hidden 64] [--epochs 30] [--tau 0.5] [--seed 0]
                   [--lr 2e-3] [--batch 512] [--init DIR] [--budget-s 1800] [--threads N] [--quiet]

Data: files written by `route_tool gen` (328-byte records; see qumo/rl/teacher_data.mojo). Rows whose teacher costs
have no spread (all equal, `A < 2`) are dropped, `max(1000, N // 20)` rows are held out for validation, the rest
trains a sign-embedding MLP (`acts = b0 + sign(obs) E`, `logits = W1 relu(acts) + b1`) with the soft cross entropy to
`softmax(-(Q - min Q) / tau)` over the valid swap slots and a random permutation of the 16 slots per sample.
`--init DIR` warm starts from a router container (a shipped f16 model or a previous run); without `--hidden` the
hidden size of the init model is used. `--budget-s` stops after finishing the epoch in which the wall-clock budget
runs out (the cosine schedule is not rescaled). `--threads` only changes the speed, never the result.

Output: RUN is a valid `qumo.rlroute.v0` model directory in f32 (`manifest.json`, `weights.bin`, no test vectors)
plus `train_log.json` ({samples, val, hidden, epochs_run, tau, lr, seed, data, wall_s, log}). The directory is
written atomically (a stale `RUN.tmp` / `RUN.old` from an interrupted run is recovered first). `export_route`
turns a run into the f16 model with test vectors.
"""

from std.sys import argv
from qumo.compiler.policy import ModelFile, load_model_file
from qumo.rl.container import recover_dir
from qumo.rl.teacher_data import read_teacher_files
from qumo.rl.router_net import RouterNet
from qumo.rl.bc import BCConfig, train_bc, write_bc_run


def _usage() -> String:
    return (
        "usage: train_route --data a.bin [b.bin ...] --out RUN [--hidden 64] [--epochs 30] [--tau 0.5] [--seed 0] "
        "[--lr 2e-3] [--batch 512] [--init DIR] [--budget-s 1800] [--threads N] [--quiet]"
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
    var data = List[String]()
    var out = String()
    var init_dir = String()
    var cfg = BCConfig()
    var hidden_given = False
    var quiet = False
    var i = 0
    while i < len(args):
        var a = args[i]
        if a == "--data":
            i += 1
            while i < len(args) and not args[i].startswith("--"):
                data.append(args[i])
                i += 1
        elif a == "--quiet":
            quiet = True
            i += 1
        else:
            var known: List[String] = ["--out", "--hidden", "--epochs", "--batch", "--lr", "--tau", "--seed", "--init", "--budget-s", "--budget_s", "--threads"]
            var ok = False
            for kf in known:
                if kf == a:
                    ok = True
            if not ok:
                raise _e640("unknown option '" + a + "'")
            var v = _need(args, i, a)
            if a == "--out":
                out = v
            elif a == "--hidden":
                cfg.hidden = Int(v)
                hidden_given = True
            elif a == "--epochs":
                cfg.epochs = Int(v)
            elif a == "--batch":
                cfg.batch = Int(v)
            elif a == "--lr":
                cfg.lr = Float64(v)
            elif a == "--tau":
                cfg.tau = Float32(Float64(v))
            elif a == "--seed":
                cfg.seed = UInt64(Int(v))
            elif a == "--init":
                init_dir = v
            elif a == "--budget-s" or a == "--budget_s":
                cfg.budget_s = Float64(v)
            elif a == "--threads":
                cfg.threads = Int(v)
            else:
                raise _e640("unknown option '" + a + "'")
            i += 2
    if len(data) == 0 or out.byte_length() == 0:
        raise _e640("--data and --out are required")
    cfg.verbose = not quiet

    recover_dir(out)
    var td = read_teacher_files(data)
    print("train_route: " + String(td.raw) + " records, " + String(td.n) + " rows after the filter")
    var init: Optional[ModelFile] = None
    if init_dir.byte_length() > 0:
        recover_dir(init_dir)
        var mf = load_model_file(init_dir)
        if not hidden_given:
            var warm = RouterNet.from_model(mf)
            cfg.hidden = warm.hidden
        init = mf^
    var trained = train_bc(cfg, td, init)
    var res = trained[1].copy()
    print(
        "train_route: " + String(res.samples) + " train / " + String(res.val) + " val rows, " + String(len(res.log))
        + " epochs, " + String(res.wall_s) + " s"
    )
    write_bc_run(out, trained[0], res, cfg, data, init_dir)
    print("train_route: wrote " + out)
