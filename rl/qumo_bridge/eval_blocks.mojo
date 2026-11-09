"""Block-level evaluation of an exported synthesis model against QUMO's exact / greedy synthesizers.

This is the library behind `rl/drivers/synth/eval_blocks.mojo` (the thin tool) and
`rl/drivers/synth/eval_synth.mojo` (export, evaluate, apply the keep rule, re-export).

`eval_blocks_json(model_root, model_id, n)` draws `n` seeded random blocks on the complete graph of the model's
qubits (so the model's coupling graph has to route them) and hands every block to these proposers:

    policy_greedy   policy ensemble, beams = 1, no sampled searches
    policy_beam8    policy ensemble, beams = 8 plus 4 sampled searches
    greedy / auto   block_synth methods of the same name
    exact           block_synth exact search, only where `exact_supported(kind, n)`

Every result is verified exactly against its source block (Perm / GF2Mat / signed SympTab equality; Pauli networks
by dense statevector comparison, see `blockalg.dense_equiv`) and must only use edges of the model graph. Costs are
2-qubit costs in CX units (CX = CZ = 1, SWAP = 3) and the 2-qubit depth. The returned JSON is byte for byte what the
earlier standalone tool wrote, except that `ms_per_block` is a wall-clock measurement: `canonical_eval_json` drops it
for comparisons.

The keep rule (`decide_keep`) is the one of the shipped models: keep a model iff the best policy setting (beam 8 plus
4 samples) is at least as cheap as the greedy synthesizer on >= 60% of the blocks, cheaper on average (cost ratio
< 1) and every verification passed.

QM-E640 for a bad argument (root not named `synth`, no instances, unknown keep rule); a model that is missing or
corrupt raises its own load error.
"""

from std.os import makedirs
from std.os.path import isdir
from std.time import perf_counter_ns
from qumo.ir.circuit import Instruction
from qumo.ir.gates import Op
from qrl.rng import Rng
from qrl.json import JsonWriter, JsonValue, JsonKind, json_quote, parse_json
from qumo.compiler.blockalg import (
    SubGraph,
    BlockKind,
    Perm,
    GF2Mat,
    SympTab,
    PauliNet,
    dense_equiv,
    ops_depth2q,
)
from qumo.compiler.block_synth import (
    synth_perm,
    synth_linear,
    synth_clifford,
    synth_pauli_network,
    exact_supported,
)
from qrl.model import load_model_file
from qumo.compiler.policy_synth import PolicySession
from qrl.container import dump_indent1, remove_tree, write_text_atomic, recover_dir
from qrl.export_synth import export_synth_run

comptime EVAL_SEED = 20261014
"""Seed of the block generator (a fixed value: the instance set is a pure function of the model graph and `n`)."""

comptime EVAL_RULE = "keep iff policy (beam8+4 samples) <= greedy cost on >= 60% of instances, mean cost ratio < 1, 0 verify failures"
"""The text recorded next to the decision in `eval_final.json` (identical to the shipped manifests)."""

comptime KEEP_MIN_WIN_OR_TIE = 0.6
comptime KEEP_MAX_COST_RATIO = 1.0
comptime POLICY_BEAM_SEARCHES = 4
comptime POLICY_BEAMS = 8


def _e640(what: String) -> Error:
    return Error("QM-E640: " + what + ". Hint: see qumo_bridge/eval_blocks.mojo and `eval_synth` usage.")


# ------------------------------------------------------------------------------------------------ instances


def cost2(ops: List[Instruction]) -> Int:
    """2-qubit cost in CX units: CX = CZ = 1, SWAP = 3."""
    var c = 0
    for o in ops:
        if o.kind == Op.SWAP:
            c += 3
        elif o.kind == Op.CX or o.kind == Op.CZ:
            c += 1
    return c


def rand_2q(g: SubGraph, mut rng: Rng, kind: Int) -> Instruction:
    """A random directed 2-qubit gate on an edge of `g` (one edge draw, then one orientation draw)."""
    var e = rng.below(g.num_edges())
    var a = g.edges[2 * e]
    var b = g.edges[2 * e + 1]
    if rng.below(2) == 1:
        var t = a
        a = b
        b = t
    return Instruction.make(kind, a, b)


def make_instance(kind: Int, n: Int, mut rng: Rng) -> List[Instruction]:
    """Seeded random block on the complete graph of `n` qubits (so the model graph must route it).

    The draw order is part of the contract: the same `Rng(EVAL_SEED)` stream yields the instance sets the shipped
    evaluations were made on."""
    var full = SubGraph.complete(n)
    var ops = List[Instruction]()
    if kind == BlockKind.PERM:
        var p = List[Int]()
        for i in range(n):
            p.append(i)
        for i in range(n - 1, 0, -1):
            var j = rng.below(i + 1)
            var t = p[i]
            p[i] = p[j]
            p[j] = t
        # realise p as a SWAP circuit on the complete graph (cycle decomposition)
        var pos = List[Int]()  # pos[i] = where state i is
        for i in range(n):
            pos.append(i)
        var who = List[Int]()  # who[i] = the state at position i
        for i in range(n):
            who.append(i)
        for i in range(n):
            # bring the state s with p[s] == i to position i
            var s = -1
            for k in range(n):
                if p[k] == i:
                    s = k
            var at = pos[s]
            if at != i:
                ops.append(Instruction.make(Op.SWAP, min(at, i), max(at, i)))
                var other = who[i]
                who[i] = s
                who[at] = other
                pos[s] = i
                pos[other] = at
        return ops^
    if kind == BlockKind.LINEAR:
        for _ in range(3 * n * n):
            ops.append(rand_2q(full, rng, Op.CX))
        return ops^
    var depth = 6 * n * n if kind == BlockKind.CLIFFORD else 2 * n * n
    var nrot = 0
    var want_rot = 2 + rng.below(4) if kind == BlockKind.PAULI else 0
    for _ in range(depth):
        var r = rng.below(10)
        if r < 2:
            ops.append(Instruction.make(Op.H, rng.below(n)))
        elif r < 4:
            ops.append(Instruction.make(Op.S, rng.below(n)))
        else:
            ops.append(rand_2q(full, rng, Op.CX))
        if kind == BlockKind.PAULI and nrot < want_rot and rng.below(depth // want_rot + 1) == 0:
            var ax = rng.below(3)
            var k = Op.RZ if ax == 0 else (Op.RX if ax == 1 else Op.RY)
            ops.append(Instruction.make(k, rng.below(n), -1, -1, 0.1 + 0.7 * Float64(nrot)))
            nrot += 1
    while kind == BlockKind.PAULI and nrot < want_rot:
        ops.append(Instruction.make(Op.RZ, rng.below(n), -1, -1, 0.3 + Float64(nrot)))
        nrot += 1
        for _ in range(n):
            ops.append(rand_2q(full, rng, Op.CX))
    return ops^


def verify_block(
    kind: Int, n: Int, src: List[Instruction], res: List[Instruction], g: SubGraph
) raises -> Bool:
    """True iff `res` only uses edges of `g` and equals `src` exactly (see the module docstring)."""
    for o in res:
        if o.kind == Op.SWAP or o.kind == Op.CX or o.kind == Op.CZ:
            if not g.has_edge(min(o.q0, o.q1), max(o.q0, o.q1)):
                return False
    if kind == BlockKind.PERM:
        return Perm.from_ops(res, n) == Perm.from_ops(src, n)
    if kind == BlockKind.LINEAR:
        return GF2Mat.from_ops(res, n) == GF2Mat.from_ops(src, n)
    if kind == BlockKind.CLIFFORD:
        return SympTab.from_ops(res, n) == SympTab.from_ops(src, n)
    var ph: Float64 = 0.0
    return dense_equiv(src, res, n, ph)


# ------------------------------------------------------------------------------------------------ accumulation


struct Acc(Copyable):
    """Per-method results over the instance list (cost / depth -1 = no usable result)."""

    var name: String
    var cost: List[Int]
    var depth: List[Int]
    var ns: Int
    var bad: Int

    def __init__(out self, name: String):
        self.name = name
        self.cost = List[Int]()
        self.depth = List[Int]()
        self.ns = 0
        self.bad = 0


def run_one(
    name: String,
    kind: Int,
    n: Int,
    src: List[Instruction],
    g: SubGraph,
    mut acc: Acc,
    mut sess: PolicySession,
    model: String,
    seed: Int,
) raises:
    """Propose with method `name`, verify, and record cost and 2q depth in `acc`."""
    var t0 = perf_counter_ns()
    var res: List[Instruction]
    if name == "policy_greedy" or name == "policy_beam8":
        var beams = 1 if name == "policy_greedy" else POLICY_BEAMS
        var searches = 0 if name == "policy_greedy" else POLICY_BEAM_SEARCHES
        if kind == BlockKind.PERM:
            res = sess.synth_perm(Perm.from_ops(src, n), g, model, beams, searches, seed)
        elif kind == BlockKind.LINEAR:
            res = sess.synth_linear(GF2Mat.from_ops(src, n), g, model, beams, searches, seed)
        elif kind == BlockKind.CLIFFORD:
            res = sess.synth_clifford(SympTab.from_ops(src, n), g, model, beams, searches, seed)
        else:
            res = sess.synth_pauli_network(PauliNet.from_ops(src, n), g, model, beams, searches, seed, src)
    else:
        var method = String(name)
        if kind == BlockKind.PERM:
            res = synth_perm(Perm.from_ops(src, n), g, method, 0, seed)
        elif kind == BlockKind.LINEAR:
            res = synth_linear(GF2Mat.from_ops(src, n), g, method, 0, seed)
        elif kind == BlockKind.CLIFFORD:
            res = synth_clifford(SympTab.from_ops(src, n), g, method, 0, seed)
        else:
            res = synth_pauli_network(PauliNet.from_ops(src, n), g, method, 0, seed, src)
    acc.ns += Int(perf_counter_ns() - t0)
    if len(res) == 0:
        acc.cost.append(-1)
        acc.depth.append(-1)
        return
    if not verify_block(kind, n, src, res, g):
        acc.bad += 1
        acc.cost.append(-1)
        acc.depth.append(-1)
        return
    acc.cost.append(cost2(res))
    acc.depth.append(ops_depth2q(res, n))


def _strip_slashes(path: String) -> String:
    var end = path.byte_length()
    var bs = path.as_bytes()
    while end > 1 and bs[end - 1] == 47:
        end -= 1
    return String(path[byte=0:end])


def synth_parent(model_root: String) raises -> String:
    """The directory whose `synth/` child is `model_root`: `PolicySession` looks models up as `<dir>/synth/INDEX.json`,
    so a model root has to be named `synth`."""
    var r = _strip_slashes(model_root)
    if r == "synth":
        return String(".")
    if not r.endswith("/synth"):
        raise _e640("model root '" + model_root + "' must be a directory named 'synth' (models/ai_transpiler/synth)")
    var parent = String(r[byte=0 : r.byte_length() - 6])
    if parent.byte_length() == 0:
        return String("/")
    return parent^


def eval_blocks_json(model_root: String, model_id: String, num_instances: Int) raises -> String:
    """Evaluate model `model_id` of `model_root` (a directory with INDEX.json and `<id>/`, named `synth`) on
    `num_instances` seeded blocks and return the compact JSON summary the standalone tool wrote:
    `{model, num_qubits, instances, methods{...}, policy_vs_greedy, policy_vs_auto[, policy_vs_exact]}`."""
    if num_instances < 1:
        raise _e640("the number of instances must be at least 1, got " + String(num_instances))
    var parent = synth_parent(model_root)
    var m = load_model_file(_strip_slashes(model_root) + "/" + model_id)
    var n = m.num_qubits
    var kind = m.kind_code
    var g = SubGraph(n, m.edges)
    var dirs = List[String]()
    dirs.append(parent)
    var sess = PolicySession(dirs)
    var names: List[String] = ["policy_greedy", "policy_beam8", "greedy", "auto"]
    if exact_supported(kind, n):
        names.append("exact")
    var accs = List[Acc]()
    for nm in names:
        accs.append(Acc(nm))
    var rng = Rng(UInt64(EVAL_SEED))
    for i in range(num_instances):
        var src = make_instance(kind, n, rng)
        for k in range(len(names)):
            run_one(names[k], kind, n, src, g, accs[k], sess, model_id, 7 + i)
    var w = JsonWriter()
    w.begin_object()
    w.kv_string("model", model_id)
    w.kv_int("num_qubits", n)
    w.kv_int("instances", num_instances)
    w.key("methods")
    w.begin_object()
    for k in range(len(names)):
        ref a = accs[k]
        var solved = 0
        var csum = 0
        var dsum = 0
        for j in range(num_instances):
            if a.cost[j] >= 0:
                solved += 1
                csum += a.cost[j]
                dsum += a.depth[j]
        w.key(a.name)
        w.begin_object()
        w.kv_float("solved", Float64(solved) / Float64(num_instances))
        w.kv_float("mean_cost", Float64(csum) / Float64(max(solved, 1)))
        w.kv_float("mean_depth2q", Float64(dsum) / Float64(max(solved, 1)))
        w.kv_float("ms_per_block", Float64(a.ns) / 1e6 / Float64(num_instances))
        w.kv_int("verify_failures", a.bad)
        w.end_object()
    w.end_object()
    # head-to-head of the best policy setting (beam 8) against greedy / auto / exact, over the instances both solved
    var pi = 1
    for ref_name in ["greedy", "auto", "exact"]:
        var ri = -1
        for k in range(len(names)):
            if names[k] == ref_name:
                ri = k
        if ri < 0:
            continue
        var win = 0
        var tie = 0
        var both = 0
        var ps = 0
        var rs = 0
        for j in range(num_instances):
            var pc = accs[pi].cost[j]
            var rc = accs[ri].cost[j]
            if pc < 0 or rc < 0:
                continue
            both += 1
            ps += pc
            rs += rc
            if pc < rc:
                win += 1
            elif pc == rc:
                tie += 1
        w.key("policy_vs_" + String(ref_name))
        w.begin_object()
        w.kv_int("compared", both)
        w.kv_float("strict_win", Float64(win) / Float64(max(both, 1)))
        w.kv_float("win_or_tie", Float64(win + tie) / Float64(max(both, 1)))
        w.kv_float("cost_ratio", Float64(ps) / Float64(max(rs, 1)))
        w.end_object()
    w.end_object()
    return w.finish()


# ------------------------------------------------------------------------------------------------ JSON helpers


def _emit_without(v: JsonValue, skip_key: String, mut w: JsonWriter) raises:
    var k = v.kind()
    if k == JsonKind.OBJECT:
        w.begin_object()
        for i in range(len(v)):
            var key = v.key_at(i)
            if key == skip_key:
                continue
            w.key(key)
            _emit_without(v.get(key), skip_key, w)
        w.end_object()
    elif k == JsonKind.ARRAY:
        w.begin_array()
        for i in range(len(v)):
            _emit_without(v.at(i), skip_key, w)
        w.end_array()
    elif k == JsonKind.STRING:
        w.string(v.as_string())
    elif k == JsonKind.NUMBER:
        w.raw(v.number_text())
    elif k == JsonKind.BOOL:
        w.boolean(v.as_bool())
    else:
        w.null()


def canonical_eval_json(js: String) raises -> String:
    """`js` re-emitted compactly without the wall-clock member `ms_per_block`: the part of an evaluation that is a
    pure function of the model and the instance count. Two evaluations of the same model compare equal on it."""
    var w = JsonWriter()
    _emit_without(parse_json(js), "ms_per_block", w)
    return w.finish()


struct KeepDecision(Copyable):
    """Outcome of the keep rule. `compared`, `win_or_tie` and `cost_ratio` are those of policy_beam8 vs greedy;
    `fails` is the total number of verification failures over all methods."""

    var keep: Bool
    var fails: Int
    var compared: Int
    var win_or_tie: Float64
    var cost_ratio: Float64

    def __init__(out self):
        self.keep = False
        self.fails = 0
        self.compared = 0
        self.win_or_tie = 0.0
        self.cost_ratio = 9.0


def decide_keep(ev: JsonValue) raises -> KeepDecision:
    """The keep rule `win_or_tie >= 0.6 and cost_ratio < 1.0 and fails == 0` (policy_beam8 vs greedy) applied to an
    evaluation JSON. A missing `policy_vs_greedy` object counts as not kept (win_or_tie 0, cost_ratio 9)."""
    var d = KeepDecision()
    if ev.has("policy_vs_greedy"):
        var pv = ev.get("policy_vs_greedy")
        d.compared = pv.get_int("compared", 0)
        d.win_or_tie = pv.get_float("win_or_tie", 0.0)
        d.cost_ratio = pv.get_float("cost_ratio", 9.0)
    if ev.has("methods"):
        var meth = ev.get("methods")
        for i in range(len(meth)):
            d.fails += meth.at(i).get_int("verify_failures", 0)
    d.keep = d.win_or_tie >= KEEP_MIN_WIN_OR_TIE and d.cost_ratio < KEEP_MAX_COST_RATIO and d.fails == 0
    return d^


def eval_final_json(ev_json: String, kept: Bool) raises -> String:
    """The `eval_final.json` text: the evaluation with the rule text and the decision appended, laid out with one
    space of indent per level (as the shipped manifests embed it)."""
    var s = ev_json.strip()
    if not s.endswith("}"):
        raise _e640("the evaluation JSON is not an object")
    var body = String(s[byte=0 : s.byte_length() - 1])
    body += ',"rule":' + json_quote(EVAL_RULE)
    body += ',"kept":' + ("true" if kept else "false") + "}"
    return dump_indent1(parse_json(body))


def _round3(x: Float64) -> Float64:
    return Float64(Int(x * 1000.0 + (0.5 if x >= 0.0 else -0.5))) / 1000.0


def _summary_vs(mut w: JsonWriter, ev: JsonValue, key: String) raises:
    w.key(key)
    if ev.has(key):
        w.raw(ev.get(key).to_json())
    else:
        w.null()


def eval_summary_line(ev_json: String, kept: Bool) raises -> String:
    """One compact JSON line for the console: `{id, kept, policy_vs_greedy, policy_vs_auto, policy_vs_exact, costs,
    ms, verify_failures}` (costs and milliseconds rounded to 3 decimals; an absent comparison is null)."""
    var ev = parse_json(ev_json)
    var d = decide_keep(ev)
    var w = JsonWriter()
    w.begin_object()
    w.kv_string("id", ev.get_string("model", ""))
    w.kv_bool("kept", kept)
    _summary_vs(w, ev, "policy_vs_greedy")
    _summary_vs(w, ev, "policy_vs_auto")
    _summary_vs(w, ev, "policy_vs_exact")
    var meth = ev.get("methods")
    w.key("costs")
    w.begin_object()
    for i in range(len(meth)):
        w.kv_float(meth.key_at(i), _round3(meth.at(i).get_float("mean_cost", 0.0)))
    w.end_object()
    w.key("ms")
    w.begin_object()
    for i in range(len(meth)):
        w.kv_float(meth.key_at(i), _round3(meth.at(i).get_float("ms_per_block", 0.0)))
    w.end_object()
    w.kv_int("verify_failures", d.fails)
    w.end_object()
    return w.finish()


# ------------------------------------------------------------------------------------------------ orchestration


struct EvalOutcome(Copyable):
    """Result of `eval_synth_run`: the decision, the evaluation numbers and the files written."""

    var kept: Bool
    var rule_kept: Bool
    var decision: KeepDecision
    var summary_line: String
    var eval_blocks_path: String
    var eval_final_path: String

    def __init__(out self):
        self.kept = False
        self.rule_kept = False
        self.decision = KeepDecision()
        self.summary_line = String()
        self.eval_blocks_path = String()
        self.eval_final_path = String()


def eval_synth_run(
    run_dir: String,
    which: String,
    id: String,
    root: String,
    work: String,
    instances: Int = 100,
    dtype: String = "f16",
    force_keep: Bool = False,
    make_default: Bool = False,
) raises -> EvalOutcome:
    """Export `run_dir/<which>` into the staging root `<work>/stage/synth`, evaluate it in-process, write
    `<run_dir>/eval_blocks.json` and `<run_dir>/eval_final.json`, apply the keep rule and, when the model is kept
    (or `force_keep`), re-export it into `root` with the evaluation embedded under `training.eval` (INDEX.json is
    rebuilt; `make_default` marks `id` the default). Nothing under `root` changes when the model is not kept.
    The staging directory is removed on success. Raises like `export_synth_run` and `eval_blocks_json`."""
    if instances < 1:
        raise _e640("--instances must be at least 1, got " + String(instances))
    if which != "best" and which != "last":
        raise _e640("--which must be best or last, got '" + which + "'")
    var stage_base = _strip_slashes(work) + "/stage"
    var stage_root = stage_base + "/synth"
    remove_tree(stage_base)
    makedirs(stage_root, exist_ok=True)
    recover_dir(run_dir + "/" + which)
    _ = export_synth_run(run_dir, which, id, stage_root, dtype, False, String())
    var ev_json = eval_blocks_json(stage_root, id, instances)
    var out = EvalOutcome()
    out.eval_blocks_path = run_dir + "/eval_blocks.json"
    out.eval_final_path = run_dir + "/eval_final.json"
    write_text_atomic(out.eval_blocks_path, ev_json)
    out.decision = decide_keep(parse_json(ev_json))
    out.rule_kept = out.decision.keep
    out.kept = out.decision.keep or force_keep
    write_text_atomic(out.eval_final_path, eval_final_json(ev_json, out.kept))
    if out.kept:
        _ = export_synth_run(run_dir, which, id, root, dtype, make_default, out.eval_final_path)
    out.summary_line = eval_summary_line(ev_json, out.kept)
    remove_tree(stage_base)
    return out^
