"""Tests for the synthesis training driver and the campaign orchestrator (component C11c; slow: not part of `make test`).

What is pinned:
  * the command line: exactly one of --iters / --budget-s, unknown options, bad numbers (QM-E640);
  * a real training run on `perm line:3` (12 iterations, tiny network): difficulty >= 3 is reached, the run directory
    is complete, the best checkpoint exports, the exported model loads and its greedy policy solves all 6
    permutations of 3 qubits;
  * T-RESUME: 6 iterations equal 3 iterations + stop + `--resume` + 3 more (identical `weights.bin` of `last/` and
    `best/`, identical numeric log fields), a resume from a run that holds only `run.json` (killed before the first
    checkpoint) restarts bit-identically, extra log lines after the last checkpoint are dropped, a changed
    result-determining option on resume is QM-E640, a fresh run refuses a directory that already holds a run;
  * T-DET: the result does not depend on `--cores` (1 against 4) and a budgeted run is a prefix of the `--iters` run;
  * warm start: `--init auto` on `X__cont` inherits the final difficulty and the global iteration offset (the log's
    `git` continues the counting, the run.json records it), an incompatible architecture or a wrong directory name is
    QM-E647;
  * the divergence guard (QM-E643): the last good state is checkpointed and the summary says unfinished;
  * the campaign: file format errors (QM-E648), the shipped campaign.json (9 enabled entries, 22,200 s of budget, 5
    disabled), lane selection, status files, chained stages (`--init <prev> --start-diff <final>`), skip of finished
    work, resume of an interrupted stage, failure / keep-rule rejection / missing decision exit with a non-zero count,
    the budget cap, all against fake sibling tools (shell scripts) so that only the orchestration is under test.
"""

from std.collections import List
from std.ffi import external_call, c_int
from std.os import makedirs
from std.os.path import isfile, isdir
from std.time import perf_counter_ns
from qumo.ai.json import JsonValue, parse_json
from qumo.compiler.policy import MLPPolicy, load_model_file
from qumo.compiler.policy_synth import PolicyEnv, greedy_trace
from qumo.rl.campaign import (
    CampaignOptions,
    Campaign,
    parse_campaign,
    read_campaign,
    select_entries,
    parse_campaign_args,
    run_campaign,
    kept_decision,
    shell_quote,
    run_command,
    model_ok,
    read_status,
    write_status,
    EntryStatus,
)
from qumo.rl.checkpoint import (
    read_run_json,
    read_summary,
    has_summary,
    has_checkpoint,
    read_checkpoint,
    read_train_state,
    log_line_count,
    RUN_FILE,
)
from qumo.rl.container import remove_tree, write_text_atomic
from qumo.rl.digest import sha256_file_hex
from qumo.rl.export_synth import export_synth_run, verify_exported_model
from qumo.rl.synth_train import (
    SynthArgs,
    Hyper,
    parse_synth_args,
    parse_hyper,
    hyper_json,
    train_synth,
)

comptime WORK = "build/test_rl_smoke"


def expect(cond: Bool, msg: String) raises:
    if not cond:
        raise Error("test_rl_smoke: " + msg)


def words(text: String) -> List[String]:
    var out_words = List[String]()
    var parts = text.split(" ")
    for i in range(len(parts)):
        if String(parts[i]).byte_length() > 0:
            out_words.append(String(parts[i]))
    return out_words^


def read_text(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


@fieldwise_init
struct RunSummaryLite(Copyable):
    var iterations: Int
    var final_difficulty: Int
    var best_difficulty: Int
    var finished: Bool


def run_args(line: String) raises -> RunSummaryLite:
    var a = parse_synth_args(words(line))
    var s = train_synth(a)
    return RunSummaryLite(s.iterations, s.final_difficulty, s.best_difficulty, s.finished)


def error_of(line: String) -> String:
    """The error text of running or parsing `line` ('' when it succeeds)."""
    try:
        var a = parse_synth_args(words(line))
        _ = train_synth(a)
    except e:
        return String(e)
    return String()


def parse_error_of(line: String) -> String:
    try:
        _ = parse_synth_args(words(line))
    except e:
        return String(e)
    return String()


def sha(path: String) raises -> String:
    return sha256_file_hex(path)


def log_numbers(path: String) raises -> List[Float64]:
    """The succ, rew and loss.total of every log line, in order (the clock fields are left out)."""
    var out_vals = List[Float64]()
    var parts = read_text(path).split("\n")
    for i in range(len(parts)):
        var l = String(parts[i])
        if l.byte_length() == 0:
            continue
        var j = parse_json(l)
        out_vals.append(Float64(j.get_int("difficulty", -1)))
        out_vals.append(j.get_float("succ", -1.0))
        out_vals.append(j.get_float("rew", -1.0))
        out_vals.append(j.get("loss").get_float("total", 0.0))
        out_vals.append(j.get_float("succ_sampled", -1.0))
    return out_vals^


def same_numbers(a: List[Float64], b: List[Float64]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def count_lines_with(text: String, needle: String) -> Int:
    var n = 0
    var parts = text.split("\n")
    for i in range(len(parts)):
        if String(parts[i]).find(needle) >= 0:
            n += 1
    return n


# ------------------------------------------------------------------------------------------------ driver tests


comptime TINY = " --kind perm --topology line:3 --emb 32 --common 32 --episodes 128 --eval-episodes 50 --quiet"


def test_cli() raises:
    expect(parse_error_of("--kind perm --topology line:3 --out x").startswith("QM-E640"), "neither --iters nor --budget-s")
    expect(parse_error_of("--kind perm --topology line:3 --out x --iters 3 --budget-s 5").startswith("QM-E640"), "both stop rules")
    expect(parse_error_of("--kind nope --topology line:3 --out x --iters 3").startswith("QM-E640"), "bad kind")
    expect(parse_error_of("--kind perm --topology line:3 --out x --iters three").startswith("QM-E640"), "bad integer")
    expect(parse_error_of("--kind perm --topology line:3 --out x --iters 3 --wat 1").startswith("QM-E640"), "unknown option")
    expect(parse_error_of("--kind perm --topology line:3 --iters 3").startswith("QM-E640"), "missing --out")
    expect(parse_error_of("--kind perm --topology line:3 --out x --iters 3 --fidelity maybe").startswith("QM-E640"), "bad fidelity")
    expect(parse_error_of("--kind perm --topology line:3 --out x --iters 3 --cores 0").startswith("QM-E640"), "cores 0")
    expect(parse_error_of("--kind perm --topology line:3 --out x --iters 3 --common 32,0").startswith("QM-E640"), "common 0")
    var a = parse_synth_args(words("--kind lf --topology ring:4 --out x --budget-s 2.5 --common 64,32 --cores 2"))
    expect(a.budget_s == 2.5 and a.iters == 0 and len(a.common) == 2 and a.common[1] == 32 and a.cores == 2, "parsed values")
    expect(a.was_given("common") and not a.was_given("emb"), "given tracking")
    # a topology that does not parse is QM-E641 when the run starts
    var e = error_of("--kind perm --topology blob:3 --out " + WORK + "/badtopo --iters 1 --quiet")
    expect(e.startswith("QM-E641"), "bad topology: " + e)
    var h = Hyper()
    h.lr = 1e-3
    h.fidelity = String("fixed")
    var h2 = parse_hyper(hyper_json(h))
    expect(h2.lr == 1e-3 and h2.fidelity == "fixed" and h2.episodes == 1024 and h2.diff_threshold == Float32(0.85), "hyper round trip")


def greedy_solves_all_perms(model_dir: String) raises:
    var m = load_model_file(model_dir)
    var pol = MLPPolicy.from_model(m)
    var env = PolicyEnv(m.kind_code, m.num_qubits, m)
    var starts = List[List[Int]]()
    for s in range(6):
        var p = List[Int]()
        if s == 0:
            p.append(0)
            p.append(1)
            p.append(2)
        elif s == 1:
            p.append(0)
            p.append(2)
            p.append(1)
        elif s == 2:
            p.append(1)
            p.append(0)
            p.append(2)
        elif s == 3:
            p.append(1)
            p.append(2)
            p.append(0)
        elif s == 4:
            p.append(2)
            p.append(0)
            p.append(1)
        else:
            p.append(2)
            p.append(1)
            p.append(0)
        starts.append(p^)
    for s in range(len(starts)):
        var st = List[UInt64]()
        for i in range(3):
            st.append(UInt64(starts[s][i]))
        var ok = False
        _ = greedy_trace(env, pol, m.max_depth, st, ok)
        expect(ok, "greedy policy failed to solve permutation " + String(s))


def test_train_export() raises:
    var run = WORK + "/run_a"
    var t0 = Int(perf_counter_ns())
    var s = run_args("--out " + run + " --iters 12 --cores 4" + TINY)
    var secs = Float64(Int(perf_counter_ns()) - t0) * 1e-9
    expect(secs < 15.0, "12 iterations took " + String(secs) + " s (budget 15 s)")
    expect(s.finished and s.iterations == 12, "summary")
    expect(s.best_difficulty >= 3, "difficulty >= 3 reached, best difficulty " + String(s.best_difficulty))
    expect(isfile(run + "/run.json") and isfile(run + "/log.jsonl") and isfile(run + "/state.json"), "run directory files")
    expect(has_checkpoint(run + "/best") and has_checkpoint(run + "/last"), "checkpoints")
    expect(log_line_count(run) == 12, "one log line per iteration")
    var info = read_run_json(run)
    expect(info.seed == 1 and info.iters == 12 and info.start_iteration == 0 and info.topology == "line:3", "run.json")
    var st = read_train_state(read_checkpoint(run + "/last"))
    expect(st.it == 12 and st.adam_t == 12 * 10, "train state of last/: it " + String(st.it) + " adam_t " + String(st.adam_t))
    var summ = read_summary(run)
    expect(summ.finished and summ.iterations == 12 and summ.final_difficulty >= s.best_difficulty, "summary.json")
    # export, load, greedy
    _ = export_synth_run(run, "best", "smoke_perm_line3", WORK + "/models", "f16", False, "")
    _ = verify_exported_model(WORK + "/models/smoke_perm_line3")
    greedy_solves_all_perms(WORK + "/models/smoke_perm_line3")


def test_resume() raises:
    var a = WORK + "/res_a"
    var b = WORK + "/res_b"
    _ = run_args("--out " + a + " --iters 6 --cores 4" + TINY)
    _ = run_args("--out " + b + " --iters 3 --cores 4" + TINY)
    expect(read_summary(b).iterations == 3, "first part")
    # lines logged after the last checkpoint must be dropped by the resume
    write_text_atomic(b + "/log.jsonl", read_text(b + "/log.jsonl") + '{"it":3,"stale":true}\n{"it":4,"stale":true}\n')
    var s = run_args("--out " + b + " --resume --iters 6 --cores 2 --quiet")
    expect(s.iterations == 6, "resumed total")
    expect(sha(a + "/last/weights.bin") == sha(b + "/last/weights.bin"), "T-RESUME: last/weights.bin differs")
    expect(sha(a + "/best/weights.bin") == sha(b + "/best/weights.bin"), "T-RESUME: best/weights.bin differs")
    expect(same_numbers(log_numbers(a + "/log.jsonl"), log_numbers(b + "/log.jsonl")), "T-RESUME: log numbers differ")
    expect(log_line_count(b) == 6, "stale log lines dropped")
    # resuming a finished run changes nothing
    var before = sha(b + "/last/weights.bin")
    _ = run_args("--out " + b + " --resume --iters 6 --quiet")
    expect(sha(b + "/last/weights.bin") == before, "resume of a finished run")
    # killed before the first checkpoint: only run.json exists
    var c = WORK + "/res_c"
    _ = run_args("--out " + c + " --iters 6 --cores 4" + TINY)
    remove_tree(c + "/best")
    remove_tree(c + "/last")
    remove_tree(c + "/summary.json")
    remove_tree(c + "/state.json")
    var s2 = run_args("--out " + c + " --resume --iters 6 --quiet")
    expect(s2.iterations == 6, "restart from run.json")
    expect(sha(a + "/last/weights.bin") == sha(c + "/last/weights.bin"), "restart from run.json is not bit-identical")
    expect(log_line_count(c) == 6, "log of the restarted run")
    # a periodic checkpoint in the middle, then a longer budget: iters is a total
    var d = WORK + "/res_d"
    var lane = WORK + "/res_lane.log"
    _ = run_args("--out " + d + " --iters 6 --checkpoint-every 4 --cores 1 --lane-log " + lane + " --tag e/s1" + TINY)
    expect(sha(a + "/last/weights.bin") == sha(d + "/last/weights.bin"), "checkpoint-every 4, cores 1")
    var ll = read_text(lane)
    expect(count_lines_with(ll, '"tag":"e/s1"') == 2 and ll.find('"it":4') >= 0 and ll.find('"it":6') >= 0, "lane log: a line at every checkpoint: " + ll)
    # option checks
    var e = error_of("--out " + b + " --resume --iters 8 --seed 2 --quiet")
    expect(e.startswith("QM-E640") and e.find("--seed") >= 0, "seed change on resume: " + e)
    e = error_of("--out " + b + " --resume --iters 8 --emb 64 --quiet")
    expect(e.startswith("QM-E640") and e.find("--emb") >= 0, "emb change on resume: " + e)
    e = error_of("--out " + b + " --iters 8" + TINY)
    expect(e.startswith("QM-E640") and e.find("already holds a run") >= 0, "fresh run over a run: " + e)
    # --resume without a run starts a fresh one
    var f = WORK + "/res_f"
    var s3 = run_args("--out " + f + " --resume --iters 2 --cores 4" + TINY)
    expect(s3.iterations == 2, "--resume on an empty directory starts fresh")


def test_det_and_budget() raises:
    var c1 = WORK + "/det_1"
    var c4 = WORK + "/det_4"
    _ = run_args("--out " + c1 + " --iters 4 --cores 1" + TINY)
    _ = run_args("--out " + c4 + " --iters 4 --cores 4" + TINY)
    expect(sha(c1 + "/last/weights.bin") == sha(c4 + "/last/weights.bin"), "T-DET: cores 1 vs 4 differ")
    expect(sha(c1 + "/best/weights.bin") == sha(c4 + "/best/weights.bin"), "T-DET: best cores 1 vs 4 differ")
    # a budgeted run is a prefix of the --iters run with the same seed
    var bd = WORK + "/budget"
    var s = run_args("--out " + bd + " --budget-s 0.05 --cores 4" + TINY)
    expect(s.finished and s.iterations >= 1, "a budgeted run runs at least one iteration")
    expect(read_summary(bd).budget_s == 0.05, "budget in the summary")
    var it = WORK + "/budget_iters"
    _ = run_args("--out " + it + " --iters " + String(s.iterations) + " --cores 4" + TINY)
    expect(sha(bd + "/last/weights.bin") == sha(it + "/last/weights.bin"), "budgeted run is not a prefix of the iters run")
    # the non-faithful switches (masked new logits, evaluated weights as best) run and change the result
    var fx = WORK + "/det_fixed"
    _ = run_args("--out " + fx + " --iters 4 --cores 2 --fidelity fixed" + TINY)
    expect(read_text(fx + "/run.json").find('"fidelity": "fixed"') >= 0, "fidelity recorded in run.json")
    expect(has_checkpoint(fx + "/best") and sha(c1 + "/last/weights.bin") != sha(fx + "/last/weights.bin"), "fixed fidelity differs from the faithful run")
    var e = error_of("--out " + fx + " --resume --iters 6 --fidelity faithful --quiet")
    expect(e.startswith("QM-E640") and e.find("--fidelity") >= 0, "fidelity change on resume: " + e)
    # different seeds differ
    var s2 = WORK + "/det_seed2"
    _ = run_args("--out " + s2 + " --iters 4 --cores 4 --seed 2" + TINY)
    expect(sha(c1 + "/last/weights.bin") != sha(s2 + "/last/weights.bin"), "seed has no effect")


def test_warm_start() raises:
    var x = WORK + "/chain/X"
    var xc = WORK + "/chain/X__cont"
    _ = run_args("--out " + x + " --iters 5 --cores 2" + TINY)
    var sx = read_summary(x)
    _ = run_args("--out " + xc + " --iters 3 --init auto --cores 2" + TINY)
    var info = read_run_json(xc)
    expect(info.start_iteration == 5 and info.start_difficulty == sx.final_difficulty, "auto inherits offset and difficulty")
    expect(info.init == "auto" and info.init_final_difficulty == sx.final_difficulty, "init recorded")
    var first = parse_json(String(read_text(xc + "/log.jsonl").split("\n")[0]))
    expect(first.get_int("it", -1) == 0 and first.get_int("git", -1) == 5, "global iteration key continues: " + String(first.get_int("git", -1)))
    expect(first.get_int("difficulty", -1) == sx.final_difficulty, "starts at the sibling's final difficulty")
    expect(read_summary(xc).start_iteration == 5, "summary offset")
    # the stage did not replay stage 1: its first iteration differs from stage 1's first at the same difficulty is not
    # testable, but the weights it starts from are the sibling's best, so a fresh-seed run differs
    var y = WORK + "/chain/Y"
    _ = run_args("--out " + y + " --iters 3 --init " + x + " --start-diff 2 --cores 2" + TINY)
    var iy = read_run_json(y)
    expect(iy.start_iteration == 5 and iy.start_difficulty == 2, "--init RUN: offset from the run, --start-diff honoured")
    var z = WORK + "/chain/Z"
    _ = run_args("--out " + z + " --iters 2 --init " + x + "/best --cores 2" + TINY)
    expect(read_run_json(z).start_iteration == 0, "--init <container> starts at iteration 0")
    var z2 = WORK + "/chain/Z2"
    _ = run_args("--out " + z2 + " --iters 2 --init " + x + "/best --start-iteration 40 --start-diff 3 --cores 2" + TINY)
    expect(read_run_json(z2).start_iteration == 40 and read_run_json(z2).start_difficulty == 3, "explicit start options")
    # architecture adoption and incompatibility
    var w = WORK + "/chain/W"
    _ = run_args("--out " + w + " --iters 1 --init " + x + " --kind perm --topology line:3 --cores 2 --quiet")
    expect(read_run_json(w).emb == 32 and read_run_json(w).common[0] == 32, "architecture adopted from the init model")
    var e = error_of("--out " + WORK + "/chain/V --iters 1 --init " + x + " --emb 64 --common 32 --kind perm --topology line:3 --quiet")
    expect(e.startswith("QM-E647"), "explicit incompatible --emb: " + e)
    e = error_of("--out " + WORK + "/chain/U --iters 1 --init " + x + " --kind lf --topology ring:4 --quiet")
    expect(e.startswith("QM-E647"), "init for another gateset: " + e)
    e = error_of("--out " + WORK + "/chain/NotCont --iters 1 --init auto" + TINY)
    expect(e.startswith("QM-E647"), "--init auto needs X__cont: " + e)
    e = error_of("--out " + WORK + "/chain/Q__cont --iters 1 --init auto" + TINY)
    expect(e.startswith("QM-E647"), "--init auto without a sibling summary: " + e)


def test_divergence_guard() raises:
    var run = WORK + "/nan"
    var e = error_of("--out " + run + " --iters 5 --lr 1e30 --cores 2" + TINY)
    expect(e.startswith("QM-E643"), "divergence raises QM-E643, got: " + e)
    expect(has_summary(run) and not read_summary(run).finished, "unfinished summary after a divergence")
    expect(has_checkpoint(run + "/last"), "last good checkpoint written")
    # the checkpoint holds finite weights and can be resumed with a sane configuration
    var m = read_checkpoint(run + "/last")
    var finite = True
    for t in range(len(m.tensors)):
        for k in range(len(m.tensors[t])):
            var v = m.tensors[t][k]
            if not (v == v) or v > 3.0e38 or v < -3.0e38:
                finite = False
    expect(finite, "the saved state is finite")


# ------------------------------------------------------------------------------------------------ campaign tests


def campaign_text(entries: String) -> String:
    return '{"format":"qumo.campaign.v0","defaults":{"cores":2},"entries":[' + entries + "]}"


def expect_e648(text: String, what: String) raises:
    try:
        _ = parse_campaign(text)
    except e:
        if String(e).startswith("QM-E648"):
            return
        raise Error("test_rl_smoke: " + what + ": expected QM-E648, got " + String(e))
    raise Error("test_rl_smoke: " + what + ": expected QM-E648, got success")


def test_campaign_format() raises:
    var ok = '{"id":"a","kind":"perm","topology":"line:3","stages":[{"budget_s":10}]}'
    var c = parse_campaign(campaign_text(ok))
    expect(len(c.entries) == 1 and c.entries[0].cores == 2 and c.entries[0].emb == 512 and c.entries[0].common == "256", "defaults")
    expect_e648("not json", "not JSON")
    expect_e648('{"entries":[' + ok + "]}", "missing format")
    expect_e648('{"format":"qumo.campaign.v0","entries":[]}', "no entries")
    expect_e648(campaign_text(ok + "," + ok), "duplicate id")
    expect_e648(campaign_text('{"id":"a b","kind":"perm","topology":"line:3","stages":[{"iters":1}]}'), "invalid id")
    expect_e648(campaign_text('{"id":"a","kind":"perm","topology":"line:3"}'), "enabled entry without stages")
    expect_e648(campaign_text('{"id":"a","kind":"perm","topology":"line:3","stages":[{}]}'), "stage without budget")
    expect_e648(campaign_text('{"id":"a","kind":"nope","topology":"line:3","stages":[{"iters":1}]}'), "bad kind")
    expect_e648(campaign_text('{"id":"a","kind":"perm","topology":"blob:3","stages":[{"iters":1}]}'), "bad topology")
    expect_e648(campaign_text('{"id":"a","kind":"perm","stages":[{"iters":1}]}'), "no topology")
    expect_e648(campaign_text('{"id":"a","kind":"perm","topology":"line:3","episodes":0,"stages":[{"iters":1}]}'), "episodes 0")
    var d = parse_campaign(campaign_text('{"id":"off","enabled":false}'))
    expect(not d.entries[0].enabled, "a disabled entry needs nothing else")
    # the shipped campaign
    var shipped = read_campaign("training/ai_transpiler/synth/campaign.json")
    var enabled = 0
    var disabled = 0
    var total = 0.0
    for i in range(len(shipped.entries)):
        if shipped.entries[i].enabled:
            enabled += 1
            total += shipped.entries[i].total_budget_s()
        else:
            disabled += 1
    expect(enabled == 9 and disabled == 5, "shipped campaign entries " + String(enabled) + "/" + String(disabled))
    expect(total == 22200.0, "shipped campaign budget " + String(total))
    var pl = Int(-1)
    for i in range(len(shipped.entries)):
        if shipped.entries[i].id == "pauli_line4":
            pl = i
    expect(pl >= 0 and len(shipped.entries[pl].stages) == 3 and shipped.entries[pl].stages[2].budget_s == 2100.0, "pauli_line4 has three 2100 s stages")


def ids_of(c: Campaign, idx: List[Int]) -> String:
    var s = String()
    for i in range(len(idx)):
        if i > 0:
            s += ","
        s += c.entries[idx[i]].id
    return s^


def test_lanes() raises:
    var c = read_campaign("training/ai_transpiler/synth/campaign.json")
    var none = List[String]()
    var all1 = select_entries(c, 1, 1, none)
    expect(len(all1) == 9, "one lane takes all nine")
    expect(c.entries[all1[0]].id == "lf_grid2x3" or c.entries[all1[0]].id == "perm_hhring10s2" or c.entries[all1[0]].id == "pauli_line4", "longest first")
    expect(c.entries[all1[0]].id == "pauli_line4", "the 6300 s entry is first")
    var l1 = select_entries(c, 1, 2, none)
    var l2 = select_entries(c, 2, 2, none)
    expect(len(l1) == 5 and len(l2) == 4, "lane sizes")
    var seen = List[String]()
    for i in range(len(l1)):
        seen.append(c.entries[l1[i]].id)
    for i in range(len(l2)):
        for k in range(len(seen)):
            expect(seen[k] != c.entries[l2[i]].id, "lanes are disjoint")
    var only = List[String]()
    only.append("lf_line6")
    only.append("perm_grid3x3")
    var sel = select_entries(c, 1, 1, only)
    expect(ids_of(c, sel) == "lf_line6,perm_grid3x3", "--only keeps budget order: " + ids_of(c, sel))
    var bad_only = List[String]()
    bad_only.append("lf_line7")
    var raised = False
    try:
        _ = select_entries(c, 1, 1, bad_only)
    except e:
        raised = String(e).startswith("QM-E640")
    expect(raised, "--only of a disabled entry is QM-E640")
    raised = False
    try:
        _ = select_entries(c, 3, 2, none)
    except e:
        raised = String(e).startswith("QM-E640")
    expect(raised, "bad lane")


def test_helpers() raises:
    expect(kept_decision('{"id":"x","kept":true,"policy_vs_greedy":{}}') == 1, "kept true")
    expect(kept_decision('noise\n{"id":"x", "kept": false}\n') == 0, "kept false with spaces")
    expect(kept_decision("nothing here") == -1, "no decision")
    expect(kept_decision('{"kept":true}\n{"kept":false}') == 0, "the last decision wins")
    expect(shell_quote("a b") == "'a b'" and shell_quote("it's") == "'it'\\''s'", "shell_quote")
    var o = parse_campaign_args(words("--campaign c.json --lane 2/3 --only a,b --cores 3 --no-eval --dry-run --instances 7"), "bindir")
    expect(o.lane_k == 2 and o.lane_l == 3 and len(o.only) == 2 and o.cores == 3 and o.no_eval and o.dry_run and o.instances == 7 and o.bin_dir == "bindir", "args")
    var raised = False
    try:
        _ = parse_campaign_args(words("--lane 1/2"), "b")
    except e:
        raised = String(e).startswith("QM-E640")
    expect(raised, "--campaign required")
    raised = False
    try:
        _ = parse_campaign_args(words("--campaign c --lane 3/2"), "b")
    except e:
        raised = String(e).startswith("QM-E640")
    expect(raised, "bad lane")
    var rc = run_command("exit 3", WORK + "/cmd.out")
    expect(rc == 3, "exit code of a command")
    rc = run_command("echo hi", WORK + "/cmd.out")
    expect(rc == 0 and read_text(WORK + "/cmd.out") == "hi\n", "output redirected")


def write_script(path: String, body: String) raises:
    with open(path, "w") as f:
        f.write("#!/bin/sh\n")
        f.write(body)
    var p = path
    _ = external_call["chmod", c_int](p.as_c_string_span().ptr(), c_int(493))


def test_campaign_run() raises:
    var fake = WORK + "/fake"
    makedirs(fake, exist_ok=True)
    var calls = fake + "/calls.log"
    remove_tree(calls)
    # a real exported model that the fake exporters copy into the root
    var real_root = WORK + "/models"
    expect(model_ok(real_root, "smoke_perm_line3"), "the real model of test_train_export is intact")
    var train_body = (
        'echo "train $*" >> ' + calls + '\n'
        'out=""\nwhile [ $# -gt 0 ]; do case "$1" in --out) out="$2"; shift;; esac; shift; done\n'
        'case "$out" in *fail_me*) echo "QM-E643: training diverged at iteration 3"; exit 1;; esac\n'
        'mkdir -p "$out"; echo "{}" > "$out/run.json"\n'
        'echo \'{"finished": true, "iterations": 7, "final_difficulty": 11, "start_iteration": 0}\' > "$out/summary.json"\n'
    )
    write_script(fake + "/train_synth", train_body)
    var eval_body = (
        'echo "eval $*" >> ' + calls + '\n'
        'run="$1"; shift; id=""; root=""\n'
        'while [ $# -gt 0 ]; do case "$1" in --id) id="$2"; shift;; --root) root="$2"; shift;; esac; shift; done\n'
        'case "$id" in *reject_me*) echo \'{"id":"\'$id\'","kept":false}\'; exit 0;; esac\n'
        'case "$id" in *nodecision*) echo "evaluated"; exit 0;; esac\n'
        'mkdir -p "$root"; rm -rf "$root/$id"; cp -R ' + real_root + '/smoke_perm_line3 "$root/$id"\n'
        'echo \'{"id":"\'$id\'","kept":true}\'\n'
    )
    write_script(fake + "/eval_synth", eval_body)
    var export_body = (
        'echo "export $*" >> ' + calls + '\n'
        'run="$1"; shift; id=""; root=""\n'
        'while [ $# -gt 0 ]; do case "$1" in --id) id="$2"; shift;; --out) root="$2"; shift;; esac; shift; done\n'
        'mkdir -p "$root"; rm -rf "$root/$id"; cp -R ' + real_root + '/smoke_perm_line3 "$root/$id"\n'
        'echo \'{"id":"\'$id\'"}\'\n'
    )
    write_script(fake + "/export_synth", export_body)

    var entries = (
        '{"id":"ok1","kind":"perm","topology":"line:3","stages":[{"iters":3},{"budget_s":50}]},'
        '{"id":"fail_me","kind":"perm","topology":"line:3","stages":[{"iters":2}]},'
        '{"id":"reject_me","kind":"lf","topology":"line:3","stages":[{"iters":2}]},'
        '{"id":"nodecision","kind":"perm","topology":"line:3","stages":[{"iters":2}]},'
        '{"id":"off","enabled":false}'
    )
    var camp = WORK + "/fake_campaign.json"
    write_text_atomic(camp, campaign_text(entries))
    var root = WORK + "/fake_root"
    var work = WORK + "/fake_work"
    remove_tree(root)
    remove_tree(work)
    var o = CampaignOptions()
    o.campaign = camp
    o.root = root
    o.work = work
    o.bin_dir = fake
    o.instances = 5
    # an interrupted stage: run.json present, no finished summary -> --resume
    makedirs(work + "/nodecision/s1", exist_ok=True)
    write_text_atomic(work + "/nodecision/s1/run.json", "{}")
    var failed = run_campaign(o)
    expect(failed == 3, "three entries fail (fail_me, reject_me, nodecision), got " + String(failed))
    var log = read_text(calls)
    expect(count_lines_with(log, "train") >= 5, "trainer calls")
    # chained stage: --init <prev> --start-diff <prev final_difficulty>
    expect(log.find("--init " + work + "/ok1/s1 --start-diff 11") >= 0, "stage 2 warm start: " + log)
    expect(log.find("--budget-s 50.0") >= 0 and log.find("--iters 3") >= 0, "stage stop rules")
    expect(log.find("--lane-log " + work + "/lane1.log --tag ok1/s2") >= 0, "lane log and tag")
    expect(log.find("nodecision/s1 --iters 2 --cores 2 --seed 1 --emb 512 --common 256 --episodes 1024 --diff-max 256 --lane-log " + work + "/lane1.log --tag nodecision/s1 --resume") >= 0, "interrupted stage is resumed")
    expect(log.find("--cores 2") >= 0, "defaults.cores")
    var s = read_status(work, "ok1")
    expect(s.state == "done" and s.kept and s.stage == 2 and s.stages == 2 and s.iteration == 7, "ok1 status " + s.state)
    expect(model_ok(root, "ok1"), "ok1 exported")
    s = read_status(work, "fail_me")
    expect(s.state == "failed" and s.error.find("QM-E643") >= 0, "fail_me status: " + s.error)
    expect(not isdir(root + "/fail_me"), "failed entry not exported")
    s = read_status(work, "reject_me")
    expect(s.state == "failed" and s.error.startswith("QM-E649") and not s.kept, "reject_me status: " + s.error)
    expect(not isdir(root + "/reject_me"), "rejected entry not exported")
    s = read_status(work, "nodecision")
    expect(s.state == "failed" and s.error.startswith("QM-E649"), "nodecision status: " + s.error)
    expect(read_status(work, "off").state == "pending", "disabled entry untouched")
    var lanelog = read_text(work + "/lane1.log")
    expect(lanelog.find('"event":"done"') >= 0 and lanelog.find('"event":"rejected"') >= 0 and lanelog.find('"event":"failed"') >= 0, "lane log events")
    # second pass: ok1 is skipped, the failures run again
    remove_tree(calls)
    failed = run_campaign(o)
    expect(failed == 3, "the same three fail again")
    log = read_text(calls)
    expect(log.find("ok1") < 0, "a finished entry is not trained or evaluated again: " + log)
    expect(count_lines_with(log, "fail_me") >= 1, "failed entries run again")
    # a finished stage is not trained again: a rerun of reject_me only evaluates
    expect(count_lines_with(log, "--tag reject_me/s1") == 0 and count_lines_with(log, "eval") >= 2, "finished stages are skipped, evaluation repeats: " + log)
    # a model that disappeared from the root is rebuilt although the status says done
    remove_tree(root + "/ok1")
    remove_tree(calls)
    failed = run_campaign(o)
    expect(model_ok(root, "ok1"), "ok1 rebuilt")
    expect(count_lines_with(read_text(calls), "train") == count_lines_with(read_text(calls), "fail_me"), "only the failing entry trains again; ok1's stages are finished")
    # --no-eval exports directly; lanes and --only
    var o2 = o.copy()
    o2.no_eval = True
    o2.root = WORK + "/fake_root2"
    o2.work = WORK + "/fake_work2"
    o2.only.append("ok1")
    remove_tree(o2.root)
    remove_tree(o2.work)
    remove_tree(calls)
    failed = run_campaign(o2)
    expect(failed == 0 and model_ok(o2.root, "ok1"), "--no-eval --only ok1")
    expect(read_text(calls).find("export ") >= 0 and read_text(calls).find("eval ") < 0, "export_synth instead of eval_synth")
    # the budget cap clips a stage and fails an entry that finds nothing left
    var o3 = o2.copy()
    o3.work = WORK + "/fake_work3"
    o3.root = WORK + "/fake_root3"
    o3.budget_cap_s = 30.0
    remove_tree(o3.work)
    remove_tree(o3.root)
    remove_tree(calls)
    failed = run_campaign(o3)
    expect(failed == 0 and read_text(calls).find("--budget-s 29.") >= 0, "budget clipped to the cap: " + read_text(calls))
    var o4 = o3.copy()
    o4.work = WORK + "/fake_work4"
    o4.budget_cap_s = 0.5
    remove_tree(o4.work)
    failed = run_campaign(o4)
    expect(failed == 1 and read_status(o4.work, "ok1").error.find("budget cap") >= 0, "an exhausted cap fails the entry")
    # a missing tool is reported before anything runs
    var o5 = o2.copy()
    o5.bin_dir = WORK + "/no_such_bin"
    var raised = False
    try:
        _ = run_campaign(o5)
    except e:
        raised = String(e).startswith("QM-E640")
    expect(raised, "missing tools")


def main() raises:
    remove_tree(WORK)
    makedirs(WORK, exist_ok=True)
    test_cli()
    test_train_export()
    test_resume()
    test_det_and_budget()
    test_warm_start()
    test_divergence_guard()
    test_campaign_format()
    test_lanes()
    test_helpers()
    test_campaign_run()
    print("test_rl_smoke OK")
