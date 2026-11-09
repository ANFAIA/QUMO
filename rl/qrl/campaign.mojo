"""Campaign orchestration for the synthesis models: parse `campaign.json`, pick a lane, train, evaluate, export.

MOJO_PORT.md 10.3, component C11c (a split of the `train_all` driver so tests can import it). The `train_all` binary
reads a campaign file (one entry per model, one or more training stages per entry) and, for every enabled entry of its
lane, runs the sibling tools of the same bin directory:

    train_synth   one run directory per stage, <work>/<id>/s<k>; a later stage warm-starts from the previous run
                  directory (`--init <prev> --start-diff <prev final_difficulty>`, which also hands over the global
                  iteration offset of section 3 rule 1); an interrupted stage is continued with `--resume`
    eval_synth    block evaluation of the last stage's best checkpoint, the keep rule, and the export into --root when
                  the model is kept (`--no-eval` replaces it by a plain `export_synth`, e.g. for smoke runs)

Idempotence. `<work>/<id>/status.json` records `state` (pending | running | done | failed), `stage`, `stages`,
`iteration`, `kept` and `error`. An entry that is `done` and whose exported model still loads and matches its sha256 is
skipped; a stage whose `summary.json` says `finished` is not trained again; a stage with a `run.json` but no finished
summary is resumed. The process exit status is non-zero when any selected entry failed, errored or missed the keep rule
(QM-E649), so `make` and CI see a failed lane. `<work>/lane<K>.log` collects one JSON line per event and per checkpoint
(the trainer appends the checkpoint lines itself through `--lane-log`).

Lanes. `--lane K/L` takes the selected entries at position `i % L == K - 1` after sorting by total stage budget,
longest first (stable), for balance. `--budget-cap-s` bounds the wall time this lane spends on budgeted stages: a stage's
budget is clipped to what is left, and an entry that finds nothing left fails; stages given as `iters` run to their
iteration count regardless.

File format (QM-E648 on anything wrong):

    {"format": "qumo.campaign.v0", "defaults": {"cores": 4, "seed": 1, "emb": 512, "common": "256", "episodes": 1024,
                                                  "diff_max": 256, "instances": 100},
     "entries": [{"id": "perm_grid3x3", "kind": "perm", "topology": "grid:3x3", "stages": [{"budget_s": 600}]},
                 {"id": "lf_line7", "enabled": false, "comment": "..."}]}

Every default can be overridden per entry; a stage has `budget_s`, `iters` or both (`iters` wins and then
`train_synth` runs with `--iters`, so the result no longer depends on the clock).
"""

from std.collections import List
from std.ffi import external_call, c_int
from std.os import makedirs
from std.os.path import isfile, isdir
from std.time import perf_counter_ns
from qrl.json import JsonValue, parse_json, json_quote, json_number, pretty_json
from qrl.model import load_model_file
from qrl.checkpoint import has_summary, read_summary, RUN_FILE
from qrl.container import write_text_atomic
from qrl.digest import sha256_file_hex
from qrl.synth_train import kind_code_of
from qrl.topology import parse_topology

comptime CAMPAIGN_FORMAT = "qumo.campaign.v0"
comptime STATUS_FILE = "status.json"


def _e640(msg: String) -> Error:
    return Error("QM-E640: " + msg + ". Hint: " + campaign_usage())


def _e648(msg: String) -> Error:
    return Error("QM-E648: campaign file: " + msg + ". Hint: see rl/drivers/synth/campaign.json for the format.")


def _e649(msg: String) -> Error:
    return Error("QM-E649: " + msg)


def campaign_usage() -> String:
    return (
        "usage: train_all --campaign FILE [--lane K/L] [--only id,id] [--root models/ai_transpiler/synth] "
        "[--work build/ai_train/synth] [--budget-cap-s 28800] [--cores 4] [--bin DIR] [--no-eval] [--instances N] "
        "[--dry-run]"
    )


# ------------------------------------------------------------------------------------------------ campaign file


struct Stage(Copyable):
    var budget_s: Float64  # 0 = not given
    var iters: Int  # 0 = not given

    def __init__(out self):
        self.budget_s = 0.0
        self.iters = 0


struct Entry(Copyable):
    var id: String
    var kind: String
    var topology: String
    var comment: String
    var enabled: Bool
    var stages: List[Stage]
    var cores: Int
    var seed: Int
    var emb: Int
    var common: String
    var episodes: Int
    var diff_max: Int
    var instances: Int

    def __init__(out self):
        self.id = String()
        self.kind = String()
        self.topology = String()
        self.comment = String()
        self.enabled = True
        self.stages = List[Stage]()
        self.cores = 4
        self.seed = 1
        self.emb = 512
        self.common = String("256")
        self.episodes = 1024
        self.diff_max = 256
        self.instances = 100

    def total_budget_s(self) -> Float64:
        var t = 0.0
        for i in range(len(self.stages)):
            if self.stages[i].iters == 0:
                t += self.stages[i].budget_s
        return t


struct Campaign(Copyable):
    var entries: List[Entry]

    def __init__(out self):
        self.entries = List[Entry]()


def _valid_id(id: String) -> Bool:
    var bs = id.as_bytes()
    if len(bs) == 0 or Int(bs[0]) == ord("."):
        return False
    for i in range(len(bs)):
        var c = Int(bs[i])
        var ok = (c >= ord("a") and c <= ord("z")) or (c >= ord("A") and c <= ord("Z")) or (c >= ord("0") and c <= ord("9"))
        if not (ok or c == ord("_") or c == ord("-") or c == ord(".")):
            return False
    return True


def _apply_overrides(mut e: Entry, j: JsonValue, where: String) raises:
    if j.has("cores"):
        e.cores = j.get("cores").as_int()
    if j.has("seed"):
        e.seed = j.get("seed").as_int()
    if j.has("emb"):
        e.emb = j.get("emb").as_int()
    if j.has("common"):
        var c = j.get("common")
        if c.is_null():
            raise _e648(where + ": common is null")
        try:
            e.common = c.as_string()
        except:
            e.common = String(c.as_int())
    if j.has("episodes"):
        e.episodes = j.get("episodes").as_int()
    if j.has("diff_max"):
        e.diff_max = j.get("diff_max").as_int()
    if j.has("instances"):
        e.instances = j.get("instances").as_int()
    if e.cores < 1 or e.emb < 1 or e.episodes < 1 or e.diff_max < 1 or e.instances < 1 or e.seed < 0:
        raise _e648(where + ": cores, emb, episodes, diff_max and instances must be >= 1 and seed >= 0")


def parse_campaign(text: String) raises -> Campaign:
    var j: JsonValue
    try:
        j = parse_json(text)
    except e:
        raise _e648("not valid JSON (" + String(e) + ")")
    if j.get_string("format", "") != CAMPAIGN_FORMAT:
        raise _e648("missing or unknown \"format\" (expected \"" + CAMPAIGN_FORMAT + "\")")
    if not j.has("entries") or len(j.get("entries")) == 0:
        raise _e648("no entries")
    var base = Entry()
    if j.has("defaults"):
        _apply_overrides(base, j.get("defaults"), "defaults")
    var c = Campaign()
    var ej = j.get("entries")
    for i in range(len(ej)):
        var x = ej.at(i)
        var e = base.copy()
        e.id = x.get_string("id", "")
        if not _valid_id(e.id):
            raise _e648("entry " + String(i) + " has an invalid id '" + e.id + "' (letters, digits, _ - . only)")
        for k in range(len(c.entries)):
            if c.entries[k].id == e.id:
                raise _e648("duplicate entry id '" + e.id + "'")
        e.enabled = x.get_bool("enabled", True)
        e.comment = x.get_string("comment", "")
        e.kind = x.get_string("kind", "")
        e.topology = x.get_string("topology", "")
        _apply_overrides(e, x, "entry " + e.id)
        if x.has("stages"):
            var sj = x.get("stages")
            for s in range(len(sj)):
                var st = Stage()
                st.budget_s = sj.at(s).get_float("budget_s", 0.0)
                st.iters = sj.at(s).get_int("iters", 0)
                if st.budget_s <= 0.0 and st.iters <= 0:
                    raise _e648("entry " + e.id + " stage " + String(s + 1) + " needs budget_s > 0 or iters > 0")
                e.stages.append(st^)
        if e.enabled:
            if e.kind.byte_length() == 0 or e.topology.byte_length() == 0:
                raise _e648("enabled entry " + e.id + " needs kind and topology")
            try:
                _ = kind_code_of(e.kind)
                _ = parse_topology(e.topology)
            except err:
                raise _e648("entry " + e.id + ": " + String(err))
            if len(e.stages) == 0:
                raise _e648("enabled entry " + e.id + " has no stages")
        c.entries.append(e^)
    return c^


def _read_all(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def read_campaign(path: String) raises -> Campaign:
    if not isfile(path):
        raise _e640("campaign file '" + path + "' does not exist")
    return parse_campaign(_read_all(path))


def select_entries(c: Campaign, lane_k: Int, lane_l: Int, only: List[String]) raises -> List[Int]:
    """Indices (into `c.entries`) of the enabled entries this lane runs, in the order it runs them. `only` restricts to
    those ids (QM-E640 for an id that is not an enabled entry); the lane split is applied after sorting the selection by
    total stage budget, longest first (stable)."""
    if lane_l < 1 or lane_k < 1 or lane_k > lane_l:
        raise _e640("--lane must be K/L with 1 <= K <= L, got " + String(lane_k) + "/" + String(lane_l))
    var sel = List[Int]()
    for i in range(len(c.entries)):
        if c.entries[i].enabled:
            sel.append(i)
    for k in range(len(only)):
        var found = False
        for i in range(len(sel)):
            if c.entries[sel[i]].id == only[k]:
                found = True
        if not found:
            raise _e640("--only names '" + only[k] + "', which is not an enabled entry")
    if len(only) > 0:
        var kept = List[Int]()
        for i in range(len(sel)):
            for k in range(len(only)):
                if c.entries[sel[i]].id == only[k]:
                    kept.append(sel[i])
                    break
        sel = kept^
    # stable insertion sort by total budget, longest first
    for i in range(1, len(sel)):
        var v = sel[i]
        var j = i - 1
        while j >= 0 and c.entries[sel[j]].total_budget_s() < c.entries[v].total_budget_s():
            sel[j + 1] = sel[j]
            j -= 1
        sel[j + 1] = v
    var mine = List[Int]()
    for i in range(len(sel)):
        if i % lane_l == lane_k - 1:
            mine.append(sel[i])
    return mine^


# ------------------------------------------------------------------------------------------------ status files


struct EntryStatus(Copyable):
    var state: String  # pending | running | done | failed
    var stage: Int
    var stages: Int
    var iteration: Int
    var kept: Bool
    var error: String

    def __init__(out self):
        self.state = String("pending")
        self.stage = 0
        self.stages = 0
        self.iteration = 0
        self.kept = False
        self.error = String()


def status_json(id: String, s: EntryStatus) -> String:
    var t = String("{")
    t += '"id":' + json_quote(id)
    t += ',"state":' + json_quote(s.state)
    t += ',"stage":' + String(s.stage)
    t += ',"stages":' + String(s.stages)
    t += ',"iteration":' + String(s.iteration)
    t += ',"kept":' + ("true" if s.kept else "false")
    t += ',"error":' + json_quote(s.error)
    t += "}"
    return t^


def write_status(work: String, id: String, s: EntryStatus) raises:
    makedirs(work + "/" + id, exist_ok=True)
    write_text_atomic(work + "/" + id + "/" + STATUS_FILE, pretty_json(status_json(id, s), 1) + "\n")


def read_status(work: String, id: String) -> EntryStatus:
    """The status of an entry; a missing or unreadable file is `pending`."""
    var s = EntryStatus()
    var path = work + "/" + id + "/" + STATUS_FILE
    if not isfile(path):
        return s^
    try:
        var j = parse_json(_read_all(path))
        s.state = j.get_string("state", "pending")
        s.stage = j.get_int("stage", 0)
        s.stages = j.get_int("stages", 0)
        s.iteration = j.get_int("iteration", 0)
        s.kept = j.get_bool("kept", False)
        s.error = j.get_string("error", "")
    except:
        return EntryStatus()
    return s^


# ------------------------------------------------------------------------------------------------ running commands


def shell_quote(s: String) -> String:
    """POSIX single-quote quoting."""
    return "'" + s.replace("'", "'\\''") + "'"


def run_command(cmd: String, out_file: String) -> Int:
    """Runs `cmd` through the shell with stdout and stderr redirected to `out_file`; returns the exit code (128 + signal
    when it was killed, -1 when the shell could not start)."""
    var full = cmd + " > " + shell_quote(out_file) + " 2>&1"
    var rc = Int(external_call["system", c_int](full.as_c_string_span().ptr()))
    if rc < 0:
        return -1
    if (rc & 0x7F) != 0:
        return 128 + (rc & 0x7F)
    return (rc >> 8) & 0xFF


def _read_file(path: String) -> String:
    if not isfile(path):
        return String()
    try:
        with open(path, "r") as f:
            return f.read()
    except:
        return String()


def _last_lines(text: String, n: Int) -> List[String]:
    var lines = List[String]()
    var parts = text.split("\n")
    for i in range(len(parts)):
        var l = String(String(parts[i]).strip())
        if l.byte_length() > 0:
            lines.append(l^)
    var out_lines = List[String]()
    var start = len(lines) - n if len(lines) > n else 0
    for i in range(start, len(lines)):
        out_lines.append(lines[i])
    return out_lines^


def _error_of(out_file: String, fallback: String) -> String:
    """The most informative line of a failed tool's output: the last line with a QM-E code, else the last line."""
    var lines = _last_lines(_read_file(out_file), 12)
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].find("QM-E") >= 0:
            return lines[i]
    if len(lines) > 0:
        return lines[len(lines) - 1]
    return fallback


def kept_decision(output: String) -> Int:
    """The `"kept"` member of the decision line `eval_synth` prints: 1 = kept, 0 = not kept, -1 = not found."""
    var lines = _last_lines(output, 6)
    for i in range(len(lines) - 1, -1, -1):
        var l = lines[i]
        var at = l.find('"kept"')
        if at < 0:
            continue
        var rest = String(unsafe_from_utf8=l.as_bytes()[at + 6 :])
        var k = String(rest.lstrip(": \t"))
        if k.startswith("true"):
            return 1
        if k.startswith("false"):
            return 0
    return -1


def model_ok(root: String, id: String) -> Bool:
    """True when `<root>/<id>` loads (FNV-1a checked by the loader) and its weights file matches the manifest sha256."""
    try:
        var m = load_model_file(root + "/" + id)
        var want = parse_json(m.manifest).get_string("sha256", "")
        if want.byte_length() == 0:
            return False
        return sha256_file_hex(root + "/" + id + "/weights.bin") == want
    except:
        return False


# ------------------------------------------------------------------------------------------------ orchestrator


struct CampaignOptions(Copyable):
    var campaign: String
    var lane_k: Int
    var lane_l: Int
    var only: List[String]
    var root: String
    var work: String
    var budget_cap_s: Float64
    var cores: Int  # 0 = use the entry's value
    var bin_dir: String
    var no_eval: Bool
    var instances: Int  # 0 = use the entry's value
    var dry_run: Bool

    def __init__(out self):
        self.campaign = String()
        self.lane_k = 1
        self.lane_l = 1
        self.only = List[String]()
        self.root = String("models/ai_transpiler/synth")
        self.work = String("build/ai_train/synth")
        self.budget_cap_s = 28800.0
        self.cores = 0
        self.bin_dir = String("build/ai_train/bin")
        self.no_eval = False
        self.instances = 0
        self.dry_run = False


def parse_campaign_args(args: List[String], default_bin: String) raises -> CampaignOptions:
    var o = CampaignOptions()
    o.bin_dir = default_bin
    var i = 0
    while i < len(args):
        var a = args[i]
        if a == "--no-eval":
            o.no_eval = True
            i += 1
            continue
        if a == "--dry-run":
            o.dry_run = True
            i += 1
            continue
        if not a.startswith("--"):
            raise _e640("unexpected argument '" + a + "'")
        if i + 1 >= len(args):
            raise _e640("option " + a + " needs a value")
        var v = args[i + 1]
        if a == "--campaign":
            o.campaign = v
        elif a == "--lane":
            var parts = v.split("/")
            if len(parts) != 2:
                raise _e640("--lane must be K/L, got '" + v + "'")
            try:
                o.lane_k = Int(String(parts[0]))
                o.lane_l = Int(String(parts[1]))
            except:
                raise _e640("--lane must be K/L with integers, got '" + v + "'")
            if o.lane_l < 1 or o.lane_k < 1 or o.lane_k > o.lane_l:
                raise _e640("--lane must be K/L with 1 <= K <= L, got '" + v + "'")
        elif a == "--only":
            var parts = v.split(",")
            for p in range(len(parts)):
                if String(parts[p]).byte_length() > 0:
                    o.only.append(String(parts[p]))
        elif a == "--root":
            o.root = v
        elif a == "--work":
            o.work = v
        elif a == "--budget-cap-s":
            try:
                o.budget_cap_s = Float64(v)
            except:
                raise _e640("--budget-cap-s needs a number, got '" + v + "'")
            if o.budget_cap_s <= 0.0:
                raise _e640("--budget-cap-s must be positive")
        elif a == "--cores":
            try:
                o.cores = Int(v)
            except:
                raise _e640("--cores needs an integer, got '" + v + "'")
            if o.cores < 1:
                raise _e640("--cores must be >= 1")
        elif a == "--bin":
            o.bin_dir = v
        elif a == "--instances":
            try:
                o.instances = Int(v)
            except:
                raise _e640("--instances needs an integer, got '" + v + "'")
            if o.instances < 1:
                raise _e640("--instances must be >= 1")
        else:
            raise _e640("unknown option '" + a + "'")
        i += 2
    if o.campaign.byte_length() == 0:
        raise _e640("--campaign is required")
    return o^


def _strip(path: String) -> String:
    var p = path
    while p.byte_length() > 1 and p.endswith("/"):
        var q = String(unsafe_from_utf8=p.as_bytes()[0 : p.byte_length() - 1])
        p = q^
    return p^


def _log_event(log: String, line: String):
    try:
        with open(log, "a") as f:
            f.write(line)
            f.write("\n")
    except:
        pass


def _say(msg: String):
    print(msg, flush=True)


def train_command(o: CampaignOptions, e: Entry, stage_index: Int, run_dir: String, prev_run: String, lane_log: String,
                  budget_s: Float64, dry: Bool = False) raises -> String:
    """The `train_synth` command line of stage `stage_index` (0-based) of entry `e`. With `dry` the previous stage's
    summary is not read (it does not exist yet in a `--dry-run`) and its final difficulty is shown symbolically."""
    var st = e.stages[stage_index].copy()
    var cores = o.cores if o.cores > 0 else e.cores
    var cmd = shell_quote(o.bin_dir + "/train_synth")
    cmd += " --kind " + shell_quote(e.kind) + " --topology " + shell_quote(e.topology)
    cmd += " --out " + shell_quote(run_dir)
    if st.iters > 0:
        cmd += " --iters " + String(st.iters)
    else:
        cmd += " --budget-s " + String(budget_s)
    cmd += " --cores " + String(cores) + " --seed " + String(e.seed) + " --emb " + String(e.emb)
    cmd += " --common " + shell_quote(e.common) + " --episodes " + String(e.episodes) + " --diff-max " + String(e.diff_max)
    cmd += " --lane-log " + shell_quote(lane_log) + " --tag " + shell_quote(e.id + "/s" + String(stage_index + 1))
    if isfile(run_dir + "/" + RUN_FILE):
        cmd += " --resume"
    elif stage_index > 0:
        if dry:
            cmd += " --init " + shell_quote(prev_run) + " --start-diff <final difficulty of " + prev_run + ">"
        else:
            var prev = read_summary(prev_run)
            cmd += " --init " + shell_quote(prev_run) + " --start-diff " + String(prev.final_difficulty)
    return cmd^


def _run_entry(o: CampaignOptions, e: Entry, lane_log: String, mut spent_s: Float64) raises -> Bool:
    """Runs one entry to completion. Returns True when its model is exported (done); False when it failed, errored or
    missed the keep rule (the status file says which)."""
    var dir = o.work + "/" + e.id
    var status = read_status(o.work, e.id)
    if status.state == "done" and model_ok(o.root, e.id):
        _say("entry " + e.id + ": done earlier, model verified, skipped")
        _log_event(lane_log, '{"entry":' + json_quote(e.id) + ',"event":"skip"}')
        return True
    makedirs(dir, exist_ok=True)
    status = EntryStatus()
    status.state = String("running")
    status.stages = len(e.stages)
    write_status(o.work, e.id, status)
    var prev_run = String()
    var last_run = String()
    for s in range(len(e.stages)):
        var run_dir = dir + "/s" + String(s + 1)
        last_run = run_dir
        status.stage = s + 1
        write_status(o.work, e.id, status)
        if has_summary(run_dir) and read_summary(run_dir).finished:
            _say("entry " + e.id + " stage " + String(s + 1) + ": finished earlier, skipped")
            prev_run = run_dir
            continue
        var budget = e.stages[s].budget_s
        if e.stages[s].iters == 0:
            var left = o.budget_cap_s - spent_s
            if left <= 1.0:
                status.state = String("failed")
                status.error = String("QM-E640: the lane budget cap of ") + String(o.budget_cap_s) + " s is used up before stage " + String(s + 1)
                write_status(o.work, e.id, status)
                _say("entry " + e.id + ": FAILED " + status.error)
                return False
            if budget > left:
                budget = left
        var cmd = train_command(o, e, s, run_dir, prev_run, lane_log, budget)
        _say("entry " + e.id + " stage " + String(s + 1) + "/" + String(len(e.stages)) + ": " + cmd)
        _log_event(lane_log, '{"entry":' + json_quote(e.id) + ',"stage":' + String(s + 1) + ',"event":"start"}')
        var out_file = dir + "/train_s" + String(s + 1) + ".out"
        var t0 = Int(perf_counter_ns())
        var rc = run_command(cmd, out_file)
        spent_s += Float64(Int(perf_counter_ns()) - t0) * 1e-9
        if rc != 0 or not (has_summary(run_dir) and read_summary(run_dir).finished):
            status.state = String("failed")
            status.error = _error_of(out_file, "train_synth exited with status " + String(rc))
            if has_summary(run_dir):
                status.iteration = read_summary(run_dir).iterations
            write_status(o.work, e.id, status)
            _log_event(lane_log, '{"entry":' + json_quote(e.id) + ',"stage":' + String(s + 1) + ',"event":"failed"}')
            _say("entry " + e.id + ": FAILED at stage " + String(s + 1) + " (" + status.error + ")")
            return False
        status.iteration = read_summary(run_dir).iterations
        write_status(o.work, e.id, status)
        _log_event(lane_log, '{"entry":' + json_quote(e.id) + ',"stage":' + String(s + 1) + ',"event":"end","iterations":' + String(status.iteration) + "}")
        prev_run = run_dir
    # evaluation and export
    var instances = o.instances if o.instances > 0 else e.instances
    var eval_cmd: String
    if o.no_eval:
        eval_cmd = (
            shell_quote(o.bin_dir + "/export_synth") + " " + shell_quote(last_run) + " --which best --id "
            + shell_quote(e.id) + " --out " + shell_quote(o.root)
        )
    else:
        eval_cmd = (
            shell_quote(o.bin_dir + "/eval_synth") + " " + shell_quote(last_run) + " --id " + shell_quote(e.id)
            + " --root " + shell_quote(o.root) + " --work " + shell_quote(dir + "/eval") + " --instances " + String(instances)
        )
    _say("entry " + e.id + ": " + eval_cmd)
    var eval_out = dir + "/eval.out"
    var erc = run_command(eval_cmd, eval_out)
    if erc != 0:
        status.state = String("failed")
        status.error = _error_of(eval_out, "evaluation exited with status " + String(erc))
        write_status(o.work, e.id, status)
        _log_event(lane_log, '{"entry":' + json_quote(e.id) + ',"event":"eval_failed"}')
        _say("entry " + e.id + ": FAILED during evaluation (" + status.error + ")")
        return False
    var kept = 1
    if not o.no_eval:
        kept = kept_decision(_read_file(eval_out))
        if kept < 0:
            status.state = String("failed")
            status.error = String("QM-E649: eval_synth printed no keep decision (see ") + eval_out + ")"
            write_status(o.work, e.id, status)
            _say("entry " + e.id + ": FAILED (" + status.error + ")")
            return False
    if kept == 0:
        status.state = String("failed")
        status.kept = False
        status.error = String("QM-E649: model " + e.id + " missed the keep rule (policy beam-8 vs greedy); not exported")
        write_status(o.work, e.id, status)
        _log_event(lane_log, '{"entry":' + json_quote(e.id) + ',"event":"rejected"}')
        _say("entry " + e.id + ": " + status.error)
        return False
    if not model_ok(o.root, e.id):
        status.state = String("failed")
        status.error = String("QM-E644: the exported model ") + o.root + "/" + e.id + " does not load or does not match its sha256"
        write_status(o.work, e.id, status)
        _say("entry " + e.id + ": FAILED (" + status.error + ")")
        return False
    status.state = String("done")
    status.kept = True
    status.error = String()
    write_status(o.work, e.id, status)
    _log_event(lane_log, '{"entry":' + json_quote(e.id) + ',"event":"done"}')
    _say("entry " + e.id + ": exported to " + o.root + "/" + e.id)
    return True


def run_campaign(o: CampaignOptions) raises -> Int:
    """Runs this lane of the campaign; returns the number of selected entries that did not end `done`."""
    var c = read_campaign(o.campaign)
    var sel = select_entries(c, o.lane_k, o.lane_l, o.only)
    var work = _strip(o.work)
    var opts = o.copy()
    opts.work = work
    opts.root = _strip(o.root)
    opts.bin_dir = _strip(o.bin_dir)
    if not o.dry_run:
        var tools = List[String]()
        tools.append("train_synth")
        if o.no_eval:
            tools.append("export_synth")
        else:
            tools.append("eval_synth")
        for i in range(len(tools)):
            if not isfile(opts.bin_dir + "/" + tools[i]):
                raise _e640("tool '" + opts.bin_dir + "/" + tools[i] + "' not found (build it first: make train-ai-synth builds all of them)")
        makedirs(work, exist_ok=True)
    var lane_log = work + "/lane" + String(o.lane_k) + ".log"
    if o.dry_run:
        _say("campaign " + o.campaign + ": lane " + String(o.lane_k) + "/" + String(o.lane_l) + ", " + String(len(sel)) + " entries")
        for k in range(len(sel)):
            var e = c.entries[sel[k]].copy()
            var prev = String()
            for s in range(len(e.stages)):
                var run_dir = work + "/" + e.id + "/s" + String(s + 1)
                _say(train_command(opts, e, s, run_dir, prev, lane_log, e.stages[s].budget_s, True))
                prev = run_dir
        return 0
    _log_event(lane_log, '{"event":"lane_start","lane":"' + String(o.lane_k) + "/" + String(o.lane_l) + '","entries":' + String(len(sel)) + "}")
    var failed = 0
    var spent_s = 0.0
    for k in range(len(sel)):
        var ok: Bool
        try:
            ok = _run_entry(opts, c.entries[sel[k]], lane_log, spent_s)
        except err:
            var st = EntryStatus()
            st.state = String("failed")
            st.stages = len(c.entries[sel[k]].stages)
            st.error = String(err)
            try:
                write_status(work, c.entries[sel[k]].id, st)
            except:
                pass
            _say("entry " + c.entries[sel[k]].id + ": ERROR " + String(err))
            ok = False
        if not ok:
            failed += 1
    _say("campaign lane " + String(o.lane_k) + "/" + String(o.lane_l) + ": " + String(len(sel) - failed) + " of " + String(len(sel)) + " entries done, " + String(failed) + " failed")
    _log_event(lane_log, '{"event":"lane_end","failed":' + String(failed) + "}")
    return failed
