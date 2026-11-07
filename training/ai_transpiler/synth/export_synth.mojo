"""Export a trained synthesis run into QUMO's model container (qumo.rlsynth.v0), native Mojo.

usage: export_synth RUN [--which best|last] --id ID --out ROOT [--dtype f16|f32] [--eval-json F] [--default]

Reads RUN/best (or RUN/last), RUN/run.json and RUN/summary.json, runs the independent pre-write checks (training-layout
network versus inference policy versus a Float64 reference, and the Pauli symmetry check), writes
ROOT/ID/{manifest.json, weights.bin} atomically with test vectors and rollouts computed from the stored weights, and
rebuilds ROOT/INDEX.json (`--default` marks ID in a `default` column). Nothing under ROOT is touched when a check
fails. Prints one JSON line {id, kind, n, bytes, n_sym, rollout_lens}.
"""

from std.sys import argv
from qumo.ai.json import parse_json
from qumo.compiler.policy import load_model_file, model_kind_name
from qumo.rl.checkpoint import read_run_json, _read_text
from qumo.rl.export_synth import export_synth_run
from qumo.rl.container import recover_dir


def _usage() -> String:
    return "usage: export_synth RUN [--which best|last] --id ID --out ROOT [--dtype f16|f32] [--eval-json F] [--default]"


def _e640(msg: String) -> Error:
    return Error("QM-E640: " + msg + ". Hint: " + _usage())


def main() raises:
    var raw = argv()
    var args = List[String]()
    for i in range(1, len(raw)):
        args.append(String(raw[i]))
    var run = String()
    var which = String("best")
    var id = String()
    var out = String()
    var dtype = String("f16")
    var eval_json = String()
    var make_default = False
    var i = 0
    while i < len(args):
        var a = args[i]
        if a == "--default":
            make_default = True
            i += 1
        elif a == "--which" or a == "--id" or a == "--out" or a == "--dtype" or a == "--eval-json":
            if i + 1 >= len(args):
                raise _e640("option " + a + " needs a value")
            var v = args[i + 1]
            if a == "--which":
                which = v
            elif a == "--id":
                id = v
            elif a == "--out":
                out = v
            elif a == "--dtype":
                dtype = v
            else:
                eval_json = v
            i += 2
        elif a.startswith("--"):
            raise _e640("unknown option '" + a + "'")
        else:
            if run.byte_length() > 0:
                raise _e640("only one RUN directory is accepted")
            run = a
            i += 1
    if run.byte_length() == 0 or id.byte_length() == 0 or out.byte_length() == 0:
        raise _e640("RUN, --id and --out are required")
    recover_dir(run + "/" + which)
    var res = export_synth_run(run, which, id, out, dtype, make_default, eval_json)
    var m = load_model_file(out + "/" + id)
    var man = parse_json(m.manifest)
    var lens = String("[")
    var ro = man.get("rollouts")
    for r in range(len(ro)):
        if r > 0:
            lens += ","
        lens += String(len(ro.at(r).get("actions")))
    lens += "]"
    print(
        '{"id":"' + id + '","kind":"' + model_kind_name(m.kind_code) + '","n":' + String(m.num_qubits)
        + ',"bytes":' + String(res.weights_bytes) + ',"n_sym":' + String(len(m.obs_perms)) + ',"rollout_lens":' + lens + "}"
    )
