"""Run the synthesis campaign: train every enabled model of a campaign file, evaluate it, export it when it is kept.

usage: train_all --campaign rl/drivers/synth/campaign.json [--lane K/L] [--only id,id]
                 [--root models/ai_transpiler/synth] [--work build/ai_train/synth] [--budget-cap-s 28800]
                 [--cores 4] [--bin DIR] [--no-eval] [--instances N] [--dry-run]

Thin main over qrl/campaign.mojo, which documents the file format, lanes, status files and idempotence. The sibling
tools train_synth, eval_synth (export_synth with --no-eval) are taken from --bin, default the directory of this binary.
Skips finished entries, resumes unfinished ones, and exits non-zero when any selected entry failed or missed the keep
rule. Build: make -C rl build/ai_train/bin/train_all (from rl/: mojo build -I . -o build/ai_train/bin/train_all drivers/synth/train_all.mojo)
"""

from std.sys import argv, exit
from qrl.campaign import parse_campaign_args, run_campaign


def _dir_of(path: String) -> String:
    var bs = path.as_bytes()
    var cut = -1
    for i in range(len(bs)):
        if Int(bs[i]) == ord("/"):
            cut = i
    if cut < 0:
        return String("build/ai_train/bin")
    if cut == 0:
        return String("/")
    return String(unsafe_from_utf8=bs[0:cut])


def main() raises:
    var raw = argv()
    var args = List[String]()
    for i in range(1, len(raw)):
        args.append(String(raw[i]))
    var o = parse_campaign_args(args, _dir_of(String(raw[0])))
    var failed = run_campaign(o)
    if failed > 0:
        exit(1)
