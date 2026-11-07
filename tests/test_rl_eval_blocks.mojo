"""Tests of qumo/rl/eval_blocks.mojo: the block-level evaluation library and the `eval_synth` orchestration.

Run from the repository root: .venv/bin/mojo run -I . tests/test_rl_eval_blocks.mojo

* Regression against the earlier standalone tool: for each of the 9 shipped synthesis models, the evaluation JSON
  (wall-clock member `ms_per_block` dropped by `canonical_eval_json`) has the SHA-256 the earlier tool's output had on
  the same model and 100 instances, and its methods / policy_vs_* members equal the evaluation embedded in the shipped
  manifest.
* The keep rule on boundary cases and on the 9 shipped evaluations.
* eval_final.json layout, the verifier (a result that uses a non-edge, or computes another operator, is rejected).
* `eval_synth_run` end to end on a random-weight run: files, staging cleanup, a rejected model leaves the model root
  untouched, a forced one is exported with the evaluation embedded, argument errors.
"""

from std.os import makedirs
from std.os.path import isdir, isfile, exists
from qumo.ai.json import JsonValue, parse_json
from qumo.ir.circuit import Instruction
from qumo.ir.gates import Op
from qumo.sim.rng import Rng
from qumo.compiler.blockalg import SubGraph, BlockKind
from qumo.compiler.policy import MODEL_KIND_LINEAR, load_model_file
from qumo.rl.checkpoint import (
    RunSummary,
    TrainState,
    new_run_info,
    net_spec_for,
    write_run_json,
    config_hash_of,
    write_best,
    write_summary,
)
from qumo.rl.container import remove_tree, verify_models
from qumo.rl.digest import sha256_span_hex
from qumo.rl.eval_blocks import (
    EVAL_RULE,
    eval_blocks_json,
    canonical_eval_json,
    decide_keep,
    eval_final_json,
    eval_summary_line,
    eval_synth_run,
    make_instance,
    verify_block,
    cost2,
    synth_parent,
)
from qumo.rl.nn import BasicNet

comptime WORK = "build/test_rl_eval_blocks"
comptime SHIPPED = "models/ai_transpiler/synth"


def expect(cond: Bool, msg: String) raises:
    if not cond:
        raise Error("test_rl_eval_blocks: " + msg)


def read_text(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def sha_text(s: String) -> String:
    return sha256_span_hex(Span(s.as_bytes()))


# SHA-256 of the canonical JSON (ms_per_block dropped) that the earlier standalone tool wrote for each shipped model
# with 100 instances. They were obtained once by running that tool (built from the commit before this port) on the
# models in models/ai_transpiler/synth, so a change of the instance generator, of the proposers' seeds or of the cost /
# verification code shows here.
def reference_hash(id: String) raises -> String:
    if id == "lf_grid2x3":
        return "5c2f9e91faed3c1fea4c309208cae564f60749cc324997a334a3ea38a56ab220"
    if id == "lf_hhH7":
        return "f80fdffb3ad043c370743daea238ff07a60a7d6a0586f5699d5068770d11e3ab"
    if id == "lf_line6":
        return "44ec20bc8dfeb945132a53ad7e6696a7b760c848db71ce1ac1cd58d59d13f889"
    if id == "lf_ring6":
        return "01b6f1de610925dc647e8ad5e1afcee772aecc099fc0b639ee7a6cf666ba7e4e"
    if id == "pauli_line4":
        return "4ecd41d890024010feede1287b533a13201358c30e0c448c48c989736946faad"
    if id == "perm_grid3x3":
        return "af60c071169fcb6ccb1abd57bdc66bf54fefb6c8994d2890fcae02f3586ac49d"
    if id == "perm_grid3x4":
        return "a635088a1f58071237c5f3cbbb2e6c45874556682ff8a3de98758f4c4a93a354"
    if id == "perm_hhring10s2":
        return "b4e35cb867cdc1b9b1f56f473de97a9da86a4107804f61201aca740c7e867168"
    if id == "perm_king3x4":
        return "82d7244ff6aa93858d6ac4c96895e06307bdf08401f044065909ad4473fbd4f1"
    raise Error("no reference hash for " + id)


def shipped_ids() -> List[String]:
    var ids: List[String] = [
        "lf_grid2x3",
        "lf_hhH7",
        "lf_line6",
        "lf_ring6",
        "pauli_line4",
        "perm_grid3x3",
        "perm_grid3x4",
        "perm_hhring10s2",
        "perm_king3x4",
    ]
    return ids^


def json_eq(a: JsonValue, b: JsonValue) raises -> Bool:
    return canonical_eval_json(a.to_json()) == canonical_eval_json(b.to_json())


def test_shipped_regression() raises:
    for id in shipped_ids():
        var js = eval_blocks_json(SHIPPED, id, 100)
        var canon = canonical_eval_json(js)
        expect(sha_text(canon) == reference_hash(id), id + ": evaluation differs from the earlier tool's output")
        expect("ms_per_block" not in canon, "canonical form keeps ms_per_block")
        expect("ms_per_block" in js, "raw output has ms_per_block")
        # the evaluation embedded in the shipped manifest was made by the same procedure
        var man = parse_json(read_text(SHIPPED + "/" + id + "/manifest.json"))
        var shipped = man.get("training").get("eval")
        var mine = parse_json(js)
        expect(shipped.has("methods"), id + ": shipped manifest has no evaluation")
        expect(json_eq(mine.get("methods"), shipped.get("methods")), id + ": methods differ from the shipped manifest")
        for k in ["policy_vs_greedy", "policy_vs_auto", "policy_vs_exact"]:
            expect(mine.has(String(k)) == shipped.has(String(k)), id + ": " + String(k) + " presence")
            if mine.has(String(k)):
                expect(json_eq(mine.get(String(k)), shipped.get(String(k))), id + ": " + String(k))
        expect(mine.get_int("instances", 0) == 100, id + ": instance count")
        expect(mine.get_int("num_qubits", 0) == load_model_file(SHIPPED + "/" + id).num_qubits, id + ": n")
        # the shipped models were all kept by the rule
        var d = decide_keep(mine)
        expect(d.keep, id + ": the rule rejects a shipped model")
        expect(d.fails == 0, id + ": verification failures")
    print("  9 shipped models: JSON hash equals the earlier tool, equals the shipped evaluation, rule keeps them OK")


def test_deterministic_and_trailing_slash() raises:
    var a = canonical_eval_json(eval_blocks_json(SHIPPED, "perm_grid3x3", 20))
    var b = canonical_eval_json(eval_blocks_json(SHIPPED + "/", "perm_grid3x3", 20))
    expect(a == b, "evaluation is not a pure function of (model, n) or trailing slash changes it")
    var c = canonical_eval_json(eval_blocks_json(SHIPPED, "perm_grid3x3", 21))
    expect(a != c, "instance count has no effect")
    expect(parse_json(c).get_int("instances", 0) == 21, "instances member")
    # exact appears exactly where it is supported (perm on 9 qubits yes, 12 qubits no)
    expect(parse_json(a).get("methods").has("exact"), "perm_grid3x3 should include the exact method")
    var big = parse_json(eval_blocks_json(SHIPPED, "perm_grid3x4", 2))
    expect(not big.get("methods").has("exact"), "perm_grid3x4 should not include the exact method")
    expect(not big.has("policy_vs_exact"), "no exact comparison without exact")
    print("  determinism, trailing slash, exact only where supported OK")


def test_synth_parent_and_args() raises:
    expect(synth_parent("models/ai_transpiler/synth") == "models/ai_transpiler", "parent of the shipped root")
    expect(synth_parent("models/ai_transpiler/synth///") == "models/ai_transpiler", "trailing slashes")
    expect(synth_parent("synth") == ".", "bare synth")
    expect(synth_parent("/synth") == "/", "root level synth")
    var raised = False
    try:
        _ = synth_parent("models/ai_transpiler/other")
    except e:
        raised = String(e).startswith("QM-E640")
    expect(raised, "a root not named synth is a usage error")
    raised = False
    try:
        _ = eval_blocks_json(SHIPPED, "perm_grid3x3", 0)
    except e2:
        raised = String(e2).startswith("QM-E640")
    expect(raised, "zero instances is a usage error")
    raised = False
    try:
        _ = eval_blocks_json(SHIPPED, "no_such_model", 3)
    except e3:
        raised = True
    expect(raised, "an unknown model raises")
    raised = False
    try:
        _ = eval_blocks_json("build/does_not_exist/synth", "x", 3)
    except e4:
        raised = True
    expect(raised, "a missing root raises")
    print("  argument checks OK")


def keep_case(win_or_tie: String, ratio: String, fails: Int) raises -> Bool:
    var js = (
        '{"methods":{"a":{"verify_failures":0},"b":{"verify_failures":'
        + String(fails)
        + '}},"policy_vs_greedy":{"compared":100,"strict_win":0.1,"win_or_tie":'
        + win_or_tie
        + ',"cost_ratio":'
        + ratio
        + "}}"
    )
    return decide_keep(parse_json(js)).keep


def test_keep_rule() raises:
    expect(keep_case("0.6", "0.99", 0), "0.6 and 0.99 keep (>= 0.6 is inclusive)")
    expect(not keep_case("0.59", "0.5", 0), "win_or_tie 0.59 rejects")
    expect(not keep_case("0.9", "1.0", 0), "cost ratio 1.0 rejects (strictly below 1)")
    expect(not keep_case("0.9", "1.0000001", 0), "cost ratio above 1 rejects")
    expect(not keep_case("0.9", "0.9", 1), "a verification failure rejects")
    expect(keep_case("1.0", "0.0", 0), "best case keeps")
    # failures are summed over all methods, not just the policy
    var d = decide_keep(parse_json('{"methods":{"x":{"verify_failures":2},"y":{"verify_failures":3}},"policy_vs_greedy":{"win_or_tie":1.0,"cost_ratio":0.5}}'))
    expect(d.fails == 5 and not d.keep, "fails are summed over methods")
    # a missing head-to-head is not kept
    expect(not decide_keep(parse_json('{"methods":{"x":{"verify_failures":0}}}')).keep, "missing policy_vs_greedy")
    print("  keep rule boundaries OK")


def test_final_json_and_summary() raises:
    var js = eval_blocks_json(SHIPPED, "lf_ring6", 10)
    var fin = eval_final_json(js, True)
    expect(fin.startswith('{\n "model": "lf_ring6",\n "num_qubits": 6,'), "indent-1 layout")
    expect(not fin.endswith("\n"), "no trailing newline")
    var p = parse_json(fin)
    expect(p.get_string("rule", "") == EVAL_RULE, "rule text")
    expect(p.get_bool("kept", False), "kept flag")
    expect(not parse_json(eval_final_json(js, False)).get_bool("kept", True), "kept false")
    expect(canonical_eval_json(fin).startswith(String(canonical_eval_json(js)[byte=0 : canonical_eval_json(js).byte_length() - 1])), "evaluation members preserved in order")
    # the same layout as the evaluation embedded in a shipped manifest: member order model, num_qubits, instances, methods
    var keys = p.keys()
    expect(keys[0] == "model" and keys[1] == "num_qubits" and keys[2] == "instances" and keys[3] == "methods", "key order")
    expect(keys[len(keys) - 2] == "rule" and keys[len(keys) - 1] == "kept", "rule and kept are last")
    var line = parse_json(eval_summary_line(js, True))
    expect(line.get_string("id", "") == "lf_ring6" and line.get_bool("kept", False), "summary id and kept")
    expect(line.get("costs").has("policy_beam8") and line.get("ms").has("greedy"), "summary costs and ms")
    expect(line.get("policy_vs_exact").is_null(), "an absent comparison is null")
    expect(line.get_int("verify_failures", -1) == 0, "summary fails")
    var raised = False
    try:
        _ = eval_final_json("[1]", True)
    except e:
        raised = String(e).startswith("QM-E640")
    expect(raised, "non-object evaluation rejected")
    print("  eval_final.json layout and console summary OK")


def test_verifier() raises:
    # line graph 0-1-2; the block lives on the complete graph of 3 qubits
    var edges: List[Int] = [0, 1, 1, 2]
    var g = SubGraph(3, edges)
    var src = List[Instruction]()
    src.append(Instruction.make(Op.CX, 0, 2))
    # CX(0,2) = CX(1,2) CX(0,1) CX(1,2) CX(0,1) on a line
    var good = List[Instruction]()
    good.append(Instruction.make(Op.CX, 0, 1))
    good.append(Instruction.make(Op.CX, 1, 2))
    good.append(Instruction.make(Op.CX, 0, 1))
    good.append(Instruction.make(Op.CX, 1, 2))
    expect(verify_block(BlockKind.LINEAR, 3, src, good, g), "a correct line decomposition verifies")
    expect(cost2(good) == 4, "cost of four CX is 4")
    var off_graph = List[Instruction]()
    off_graph.append(Instruction.make(Op.CX, 0, 2))
    expect(not verify_block(BlockKind.LINEAR, 3, src, off_graph, g), "a gate on a non-edge is rejected")
    var wrong = List[Instruction]()
    wrong.append(Instruction.make(Op.CX, 0, 1))
    expect(not verify_block(BlockKind.LINEAR, 3, src, wrong, g), "another linear operator is rejected")
    # permutation: SWAP(0,2) as three SWAPs on the line is a different operator from SWAP(0,1)
    var psrc = List[Instruction]()
    psrc.append(Instruction.make(Op.SWAP, 0, 2))
    var pgood = List[Instruction]()
    pgood.append(Instruction.make(Op.SWAP, 0, 1))
    pgood.append(Instruction.make(Op.SWAP, 1, 2))
    pgood.append(Instruction.make(Op.SWAP, 0, 1))
    expect(verify_block(BlockKind.PERM, 3, psrc, pgood, g), "SWAP(0,2) = SWAP(0,1) SWAP(1,2) SWAP(0,1)")
    expect(cost2(pgood) == 9, "three SWAPs cost 9")
    var pbad = List[Instruction]()
    pbad.append(Instruction.make(Op.SWAP, 0, 1))
    expect(not verify_block(BlockKind.PERM, 3, psrc, pbad, g), "another permutation is rejected")
    # Clifford: H(0) vs H(1)
    var csrc = List[Instruction]()
    csrc.append(Instruction.make(Op.H, 0))
    var cgood = List[Instruction]()
    cgood.append(Instruction.make(Op.H, 0))
    var cbad = List[Instruction]()
    cbad.append(Instruction.make(Op.H, 1))
    expect(verify_block(BlockKind.CLIFFORD, 3, csrc, cgood, g), "H(0) verifies")
    expect(not verify_block(BlockKind.CLIFFORD, 3, csrc, cbad, g), "H(1) is another Clifford")
    # Pauli network: RZ(0, 0.4) vs RZ(0, 0.5)
    var rsrc = List[Instruction]()
    rsrc.append(Instruction.make(Op.RZ, 0, -1, -1, 0.4))
    var rbad = List[Instruction]()
    rbad.append(Instruction.make(Op.RZ, 0, -1, -1, 0.5))
    expect(verify_block(BlockKind.PAULI, 3, rsrc, rsrc, g), "RZ verifies against itself")
    expect(not verify_block(BlockKind.PAULI, 3, rsrc, rbad, g), "another angle is rejected")
    print("  verifier accepts correct and rejects wrong results OK")


def test_instances() raises:
    # the block generator is a pure function of the stream; each kind yields non-trivial, valid blocks
    var kinds: List[Int] = [BlockKind.PERM, BlockKind.LINEAR, BlockKind.CLIFFORD, BlockKind.PAULI]
    for k in kinds:
        var n = 5
        var r1 = Rng(20261014)
        var r2 = Rng(20261014)
        var a = make_instance(k, n, r1)
        var b = make_instance(k, n, r2)
        expect(len(a) == len(b) and len(a) > 0, "instance size, kind " + String(k))
        for i in range(len(a)):
            expect(a[i].kind == b[i].kind and a[i].q0 == b[i].q0 and a[i].q1 == b[i].q1, "instance equality")
            expect(a[i].q0 >= 0 and a[i].q0 < n, "qubit in range")
        # a block on the complete graph always verifies against itself
        var full = SubGraph.complete(n)
        expect(verify_block(k, n, a, a, full), "block verifies against itself, kind " + String(k))
        var c = make_instance(k, n, r1)
        var same = len(a) == len(c)
        if same:
            same = False
            for i in range(len(a)):
                if a[i].q0 != c[i].q0 or a[i].q1 != c[i].q1 or a[i].kind != c[i].kind:
                    same = True
                    break
            expect(same, "consecutive instances differ, kind " + String(k))
    print("  instance generator OK")


def make_run(run_dir: String, id: String, topo: String, seed: Int) raises:
    var info = new_run_info(id, MODEL_KIND_LINEAR, topo)
    info.emb = 24
    var common: List[Int] = [16]
    info.common = common.copy()
    info.seed = seed
    info.cores = 4
    info.budget_s = 12.5
    var net = BasicNet(net_spec_for(info), UInt64(seed))
    write_run_json(run_dir, info)
    var st = TrainState()
    st.it = 7
    st.difficulty = 5
    st.seed = seed
    st.config_hash = config_hash_of(info)
    _ = write_best(run_dir, net, info, st)
    var s = RunSummary()
    s.finished = True
    s.iterations = 7
    s.final_difficulty = 5
    s.wall_s = 3.5
    s.budget_s = 12.5
    s.cores = 4
    s.seed = seed
    s.topology = topo
    write_summary(run_dir, s)


def test_eval_synth_run_shipped_weights() raises:
    """A run holding the weights of the shipped lf_ring6 model: the whole pipeline (export to staging, evaluate, rule,
    publish) reproduces the shipped evaluation and keeps the model without --force-keep."""
    var work = String(WORK) + "/shipped"
    remove_tree(work)
    makedirs(work, exist_ok=True)
    var shipped = load_model_file(SHIPPED + "/lf_ring6")
    var info = new_run_info("lf_ring6", MODEL_KIND_LINEAR, "ring:6")
    info.seed = 1
    info.cores = 4
    var net = BasicNet.from_model(shipped)
    write_run_json(work + "/run", info)
    var st = TrainState()
    st.config_hash = config_hash_of(info)
    _ = write_best(work + "/run", net, info, st)
    var root = work + "/models/synth"
    var o = eval_synth_run(work + "/run", "best", "lf_ring6", root, work + "/w", 100)
    expect(o.rule_kept and o.kept, "the shipped weights satisfy the keep rule")
    expect(o.decision.compared == 100 and o.decision.fails == 0, "decision fields")
    expect(o.decision.win_or_tie == 0.87, "win_or_tie of the shipped evaluation")
    expect(isdir(root + "/lf_ring6") and isfile(root + "/INDEX.json"), "published")
    var man = parse_json(read_text(root + "/lf_ring6/manifest.json"))
    var emb = man.get("training").get("eval")
    expect(emb.get_bool("kept", False), "embedded kept")
    var ship_ev = parse_json(read_text(SHIPPED + "/lf_ring6/manifest.json")).get("training").get("eval")
    expect(json_eq(emb.get("methods"), ship_ev.get("methods")), "the published evaluation equals the shipped one")
    expect(json_eq(emb.get("policy_vs_greedy"), ship_ev.get("policy_vs_greedy")), "policy_vs_greedy")
    expect(not emb.has("qiskit_o3_unpaired"), "no foreign baseline member")
    # the published weights are the shipped weights (f16 round trip of already-f16 values)
    var ship_man = parse_json(read_text(SHIPPED + "/lf_ring6/manifest.json"))
    expect(man.get_string("sha256", "x") == ship_man.get_string("sha256", "y"), "published weights differ from the shipped ones")
    print("  eval_synth_run with the shipped lf_ring6 weights: kept by the rule, evaluation equals the shipped one OK")


def test_eval_synth_run() raises:
    var work = String(WORK) + "/run"
    remove_tree(work)
    makedirs(work, exist_ok=True)
    make_run(work + "/run", "tiny", "ring:4", 3)
    var root = work + "/models/synth"
    # 1. normal call: a random-weight model is evaluated; whatever the decision, the root is consistent with it
    var o = eval_synth_run(work + "/run", "best", "tiny", root, work + "/w", 12)
    expect(isfile(o.eval_blocks_path) and isfile(o.eval_final_path), "eval files written")
    expect(not isdir(work + "/w/stage"), "staging removed")
    var ev = parse_json(read_text(work + "/run/eval_blocks.json"))
    var fin = parse_json(read_text(work + "/run/eval_final.json"))
    expect(ev.get_string("model", "") == "tiny" and ev.get_int("instances", 0) == 12, "eval_blocks.json content")
    expect(fin.get_bool("kept", False) == o.kept, "eval_final.json kept matches the outcome")
    expect(decide_keep(ev).keep == o.rule_kept, "rule_kept is the rule's answer")
    expect(o.kept == o.rule_kept, "no force: kept equals the rule")
    expect(json_eq(ev.get("methods"), fin.get("methods")), "final repeats the evaluation")
    expect(read_text(work + "/run/eval_final.json") == eval_final_json(read_text(work + "/run/eval_blocks.json"), o.kept), "final text")
    expect(parse_json(o.summary_line).get_string("id", "") == "tiny", "summary line")
    var shipped_before = isfile(root + "/INDEX.json")
    expect(shipped_before == o.kept, "the model root changes exactly when the model is kept")
    # 2. an evaluation is a pure function of the run: a second call writes the same canonical JSON
    var o2 = eval_synth_run(work + "/run", "best", "tiny", root + "_b", work + "/w", 12)
    expect(
        canonical_eval_json(read_text(o2.eval_blocks_path)) == canonical_eval_json(read_text(o.eval_blocks_path)),
        "re-evaluation differs",
    )
    # 3. force_keep exports whatever the rule says, with the evaluation embedded and an INDEX row
    var forced_root = work + "/forced/synth"
    var o3 = eval_synth_run(work + "/run", "best", "tiny", forced_root, work + "/w", 12, "f16", True, True)
    expect(o3.kept, "force_keep keeps")
    expect(o3.rule_kept == o.rule_kept, "force_keep does not change the rule's answer")
    var man = parse_json(read_text(forced_root + "/tiny/manifest.json"))
    var emb = man.get("training").get("eval")
    expect(emb.get_bool("kept", False), "embedded evaluation says kept")
    expect(emb.get_string("rule", "") == EVAL_RULE, "embedded rule text")
    expect(json_eq(emb.get("methods"), parse_json(read_text(o3.eval_final_path)).get("methods")), "embedded methods")
    var idx = parse_json(read_text(forced_root + "/INDEX.json"))
    expect(len(idx) == 1 and idx.at(0).get_string("id", "") == "tiny", "INDEX row")
    expect(idx.at(0).get_bool("default", False), "make_default marks the model")
    var rep = verify_models(work + "/forced")
    expect(rep.n_problems == 0, "the exported tree verifies")
    # the exported model can itself be evaluated again by the library (same instances, same numbers)
    var again = canonical_eval_json(eval_blocks_json(forced_root, "tiny", 12))
    expect(again == canonical_eval_json(read_text(o3.eval_blocks_path)), "staged and published model evaluate identically")
    # 4. argument errors leave nothing behind
    var raised = False
    try:
        _ = eval_synth_run(work + "/run", "best", "tiny", work + "/x/synth", work + "/w", 0)
    except e:
        raised = String(e).startswith("QM-E640")
    expect(raised and not exists(work + "/x"), "zero instances is a usage error")
    raised = False
    try:
        _ = eval_synth_run(work + "/run", "middle", "tiny", work + "/x/synth", work + "/w", 5)
    except e2:
        raised = String(e2).startswith("QM-E640")
    expect(raised, "--which middle is a usage error")
    raised = False
    try:
        _ = eval_synth_run(work + "/missing_run", "best", "tiny", work + "/x/synth", work + "/w", 5)
    except e3:
        raised = True
    expect(raised and not exists(work + "/x"), "a missing run raises and publishes nothing")
    print("  eval_synth_run: files, rule, force-keep export, embedded evaluation, errors OK")


def main() raises:
    test_shipped_regression()
    test_deterministic_and_trailing_slash()
    test_synth_parent_and_args()
    test_keep_rule()
    test_final_json_and_summary()
    test_verifier()
    test_instances()
    test_eval_synth_run()
    test_eval_synth_run_shipped_weights()
    print("test_rl_eval_blocks OK")
