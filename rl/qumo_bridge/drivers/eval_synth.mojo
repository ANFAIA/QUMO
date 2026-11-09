"""Evaluate a trained synthesis run and publish it into the model root when it beats QUMO's greedy synthesizer.

usage: eval_synth RUN --id ID --root ROOT --work DIR [--instances 100] [--which best|last] [--dtype f16|f32]
                  [--keep-rule default] [--force-keep] [--default]

Steps:
 1. export RUN/<which> (default best) into the staging root DIR/stage/synth;
 2. run the block evaluation in-process (qumo_bridge/eval_blocks.mojo) on seeded random blocks of the model's own
    coupling graph: policy (greedy ensemble; beam 8 + 4 samples) vs the greedy / auto / exact synthesizers, every
    output verified exactly; write RUN/eval_blocks.json;
 3. keep the model when the best policy setting (beam 8) is at least as cheap as greedy on >= 60% of the blocks AND
    cheaper on average (cost ratio < 1) AND every verification passed; write RUN/eval_final.json (the evaluation, the
    rule text and the decision);
 4. when kept (or --force-keep), re-export into ROOT with the evaluation embedded under training.eval and rebuild
    ROOT/INDEX.json (--default marks ID the default model).
Exit status 0 whether or not the model is kept (the decision is printed as one JSON line); non-zero on an error.
"""

from std.sys import argv
from qumo_bridge.eval_blocks import eval_synth_run


def _usage() -> String:
    return (
        "usage: eval_synth RUN --id ID --root ROOT --work DIR [--instances 100] [--which best|last] "
        "[--dtype f16|f32] [--keep-rule default] [--force-keep] [--default]"
    )


def _e640(msg: String) -> Error:
    return Error("QM-E640: " + msg + ". Hint: " + _usage())


def main() raises:
    var raw = argv()
    var args = List[String]()
    for i in range(1, len(raw)):
        args.append(String(raw[i]))
    var run = String()
    var id = String()
    var root = String()
    var work = String()
    var which = String("best")
    var dtype = String("f16")
    var instances = 100
    var force_keep = False
    var make_default = False
    var i = 0
    while i < len(args):
        var a = args[i]
        if a == "--force-keep":
            force_keep = True
            i += 1
        elif a == "--default":
            make_default = True
            i += 1
        elif (
            a == "--id"
            or a == "--root"
            or a == "--work"
            or a == "--instances"
            or a == "--which"
            or a == "--dtype"
            or a == "--keep-rule"
        ):
            if i + 1 >= len(args):
                raise _e640("option " + a + " needs a value")
            var v = args[i + 1]
            if a == "--id":
                id = v
            elif a == "--root":
                root = v
            elif a == "--work":
                work = v
            elif a == "--which":
                which = v
            elif a == "--dtype":
                dtype = v
            elif a == "--keep-rule":
                if v != "default":
                    raise _e640("--keep-rule must be 'default', got '" + v + "'")
            else:
                try:
                    instances = Int(v)
                except:
                    raise _e640("--instances must be an integer, got '" + v + "'")
            i += 2
        elif a.startswith("--"):
            raise _e640("unknown option '" + a + "'")
        else:
            if run.byte_length() > 0:
                raise _e640("only one RUN directory is accepted")
            run = a
            i += 1
    if run.byte_length() == 0 or id.byte_length() == 0 or root.byte_length() == 0 or work.byte_length() == 0:
        raise _e640("RUN, --id, --root and --work are required")
    var res = eval_synth_run(run, which, id, root, work, instances, dtype, force_keep, make_default)
    print(res.summary_line)
