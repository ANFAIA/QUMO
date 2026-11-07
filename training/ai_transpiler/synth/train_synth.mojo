"""Train a synthesis policy (PPO) on one of the four synthesis environments, native Mojo.

usage: train_synth --kind perm|lf|clifford|pauli --topology SPEC --out RUN (--iters N | --budget-s S)
                   [--cores 4] [--seed 1] [--emb 512] [--common 256[,..]] [--episodes 1024] [--diff-max 256]
                   [--init DIR|auto] [--start-diff D] [--start-iteration N] [--resume] [--checkpoint-every 50]
                   [--eval-episodes 200] [--sampled-eval-every 5] [--fidelity faithful|fixed] [--lr 3e-4] [--quiet]

Thin main over qumo/rl/synth_train.mojo, which documents the run directory, the stop rule, resume and warm start.
Build: .venv/bin/mojo build -I . -o build/ai_train/bin/train_synth training/ai_transpiler/synth/train_synth.mojo
"""

from std.sys import argv, exit
from qumo.rl.synth_train import run_synth_train


def main() raises:
    var raw = argv()
    var args = List[String]()
    for i in range(1, len(raw)):
        args.append(String(raw[i]))
    var rc = run_synth_train(args)
    if rc != 0:
        exit(rc)
