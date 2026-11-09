"""Export a router run (a `train_route` output directory) as a `qumo.rlroute.v0` model (native Mojo).

usage: export_route RUN --root ROOT --id ID [--default] [--episodes 2] [--meta FILE]

RUN is the directory `train_route` wrote (or any router container: a shipped model, a router PPO run). The model is
written to ROOT/ID (f16 weights, f32 biases, 64-byte aligned, fnv1a64 in the manifest) with
  * `training` merged from the run manifest, RUN/train_log.json (without its per-epoch log), RUN/training_meta.json
    and FILE (later wins),
  * 6 observation -> logits vectors computed in Float64 from the stored weights and re-derived independently,
  * `--episodes N` (default 2) routing episodes played in policy mode with the stored weights (the parity traces the
    tests replay bit for bit),
and ROOT/INDEX.json is rebuilt. `--default` makes ID the default router; without it the previous default is kept.
Nothing under ROOT/ID or in INDEX.json changes when a check fails.
"""

from std.sys import argv
from qumo_bridge.export_route import export_route_run, DEFAULT_EPISODES


def _usage() -> String:
    return "usage: export_route RUN --root ROOT --id ID [--default] [--episodes 2] [--meta FILE]"


def _e640(msg: String) -> Error:
    return Error("QM-E640: " + msg + ". Hint: " + _usage())


def main() raises:
    var raw = argv()
    var args = List[String]()
    for i in range(1, len(raw)):
        args.append(String(raw[i]))
    var run = String()
    var root = String()
    var id = String()
    var meta = String()
    var make_default = False
    var episodes = DEFAULT_EPISODES
    var i = 0
    while i < len(args):
        var a = args[i]
        if a == "--default":
            make_default = True
            i += 1
        elif a == "--root" or a == "--id" or a == "--episodes" or a == "--meta":
            if i + 1 >= len(args):
                raise _e640("option " + a + " needs a value")
            var v = args[i + 1]
            if a == "--root":
                root = v
            elif a == "--id":
                id = v
            elif a == "--meta":
                meta = v
            else:
                episodes = Int(v)
            i += 2
        elif a.startswith("--"):
            raise _e640("unknown option '" + a + "'")
        else:
            if run.byte_length() > 0:
                raise _e640("more than one run directory given ('" + run + "' and '" + a + "')")
            run = a
            i += 1
    if run.byte_length() == 0 or root.byte_length() == 0 or id.byte_length() == 0:
        raise _e640("RUN, --root and --id are required")
    var info = export_route_run(run, root, id, make_default, episodes, meta)
    print("exported " + id + " -> " + root + "/" + id + " bytes " + String(info.weights_bytes))
