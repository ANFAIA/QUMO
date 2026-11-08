"""End-to-end check of the built training tools (MOJO_PORT.md 13.2 `test_rl_e2e`, acceptance A7 and A10).

Slow: excluded from `make test`, run by `make test-rl-slow`. It drives the real binaries in `$QUMO_E2E_BUILD/bin`
(default build/ai_train/bin; a missing tool is built with `make`) through the shell:

  A10  determinism: `train_synth` on the smoke spec (lf on ring:4, seed 1) for QUMO_E2E_ITERS iterations (default 30; the
       full smoke is 150) with --cores 1, 2, 4, a requested 8 (clamped) and a rerun at 4: identical `best/weights.bin`
       and `last/weights.bin` SHA-256 for all of them; `train_route` with --threads 1, 2 and 4: identical weights.
  A7   staged-root replay: a synthesis model (eval_synth --force-keep, so this checks the export, not the quality) and
       a router (gen-jobs, gen, train_route, export_route) are written into a staging root; `models_verify` on that root
       prints 0 problems; `tests/test_policy.mojo` and `tests/test_ai_route.mojo` run with QUMO_TEST_MODEL_ROOT set to
       it and pass; an explicit pipeline names the staged synthesis model and the staged router and compiles a matching
       block without QM-E617 / QM-E620.

Everything is written under `$QUMO_E2E_BUILD/e2e`; models/ is never touched.
"""

from std.os import getenv, makedirs
from std.os.path import isfile
from qumo.ir.circuit import Circuit
from qumo.compiler.target import Target
from qumo.compiler.pass_manager import compile_spec
from qumo.compiler.policy import load_model_file
from qrl.campaign import run_command, shell_quote
from qrl.container import remove_tree
from qrl.digest import sha256_file_hex


def _env(name: String, default: String) -> String:
    var v = getenv(name)
    return v if v.byte_length() > 0 else default


def _read(path: String) -> String:
    try:
        with open(path, "r") as f:
            return f.read()
    except:
        return String()


def _tail(path: String, n: Int) -> String:
    var parts = _read(path).split("\n")
    var out = String()
    var start = max(0, len(parts) - n)
    for i in range(start, len(parts)):
        out += String(parts[i]) + "\n"
    return out


def sh(cmd: String, log: String, what: String) raises:
    var rc = run_command(cmd, log)
    if rc != 0:
        raise Error("test_bridge_e2e: " + what + " failed (exit " + String(rc) + "): " + cmd + "\n" + _tail(log, 25))


def ensure_tools(build: String, tools: List[String], log: String) raises:
    for t in tools:
        var path = build + "/bin/" + t
        if not isfile(path):
            print("  building", path)
            sh("make AI_BUILD=" + shell_quote(build) + " " + shell_quote(path), log, "build of " + t)
            if not isfile(path):
                raise Error("test_bridge_e2e: tool " + path + " was not built")


def expect(cond: Bool, msg: String) raises:
    if not cond:
        raise Error("test_bridge_e2e: " + msg)


def train_smoke(bin: String, run: String, cores: Int, iters: Int, log: String) raises:
    sh(
        bin + "/train_synth --kind lf --topology ring:4 --out " + shell_quote(run) + " --iters " + String(iters)
        + " --cores " + String(cores) + " --seed 1 --quiet",
        log,
        "train_synth --cores " + String(cores),
    )


def weights_sha(run: String, which: String) raises -> String:
    return sha256_file_hex(run + "/" + which + "/weights.bin")


def test_determinism(bin: String, work: String, iters: Int) raises:
    var log = work + "/log.txt"
    var cores: List[Int] = [1, 2, 4, 8, 4]
    var best_ref = String()
    var last_ref = String()
    for i in range(len(cores)):
        var run = work + "/det" + String(i) + "_c" + String(cores[i])
        train_smoke(bin, run, cores[i], iters, log)
        var b = weights_sha(run, "best")
        var l = weights_sha(run, "last")
        if i == 0:
            best_ref = b
            last_ref = l
        expect(b == best_ref, "best/weights.bin differs for --cores " + String(cores[i]) + " (run " + String(i) + ")")
        expect(l == last_ref, "last/weights.bin differs for --cores " + String(cores[i]) + " (run " + String(i) + ")")
    print("  A10 train_synth: cores 1, 2, 4, 8 (clamped) and a rerun give identical best/last weights.bin (", iters, "iterations )")


def make_teacher(bin: String, work: String) raises:
    var log = work + "/log.txt"
    var r = work + "/route"
    makedirs(r + "/stage", exist_ok=True)
    sh(bin + "/route_tool gen-jobs 31 100 30 0.5 " + shell_quote(r + "/jobs0.json"), log, "gen-jobs")
    sh(
        "ROUTE_TOOL_CAP=96 " + bin + "/route_tool gen " + shell_quote(r + "/jobs0.json") + " " + shell_quote(r + "/d0.bin")
        + " - 1.0 401 heuristic 16",
        log,
        "route_tool gen",
    )


def test_route_determinism(bin: String, work: String) raises:
    var log = work + "/log.txt"
    var r = work + "/route"
    var first = String()
    var threads: List[Int] = [1, 2, 4]
    for i in range(len(threads)):
        var run = r + "/bc_t" + String(threads[i])
        sh(
            bin + "/train_route --data " + shell_quote(r + "/d0.bin") + " --out " + shell_quote(run)
            + " --hidden 64 --tau 0.5 --epochs 4 --seed 0 --threads " + String(threads[i]) + " --quiet",
            log,
            "train_route --threads " + String(threads[i]),
        )
        var h = sha256_file_hex(run + "/weights.bin")
        if i == 0:
            first = h
        expect(h == first, "train_route weights differ for --threads " + String(threads[i]))
    print("  A10 train_route: threads 1, 2 and 4 give identical weights.bin")


def test_staged_root(bin: String, work: String, mojo: String) raises:
    var log = work + "/log.txt"
    var stage = work + "/stage"
    remove_tree(stage)
    makedirs(stage + "/synth", exist_ok=True)
    makedirs(stage + "/route", exist_ok=True)
    # synthesis model: staged through eval_synth (--force-keep: this test is about export and loading, not quality)
    var run = work + "/det0_c1"
    makedirs(work + "/synth_work", exist_ok=True)
    sh(
        bin + "/eval_synth " + shell_quote(run) + " --id e2e_lf_ring4 --root " + shell_quote(stage + "/synth") + " --work "
        + shell_quote(work + "/synth_work") + " --instances 20 --force-keep --default",
        log,
        "eval_synth",
    )
    expect(isfile(stage + "/synth/e2e_lf_ring4/manifest.json"), "the staged synthesis model is missing")
    # router: round 0 from the teacher data of make_teacher
    var r = work + "/route"
    sh(
        bin + "/train_route --data " + shell_quote(r + "/d0.bin") + " --out " + shell_quote(r + "/run0")
        + " --hidden 64 --tau 0.5 --epochs 4 --seed 0 --quiet",
        log,
        "train_route",
    )
    sh(
        bin + "/export_route " + shell_quote(r + "/run0") + " --root " + shell_quote(stage + "/route")
        + " --id e2e_route --default --episodes 1",
        log,
        "export_route",
    )
    expect(isfile(stage + "/route/e2e_route/manifest.json"), "the staged router is missing")

    # models_verify on the staging root
    var vlog = work + "/verify.txt"
    sh("QUMO_MODEL_DIR=" + shell_quote(stage) + " " + bin + "/models_verify", vlog, "models_verify on the staging root")
    expect(_read(vlog).find(" 0 problems") >= 0, "models_verify did not report 0 problems: " + _read(vlog))
    print("  A7 models_verify on the staging root: 0 problems")

    # both models load through the container loader
    var ms = load_model_file(stage + "/synth/e2e_lf_ring4")
    expect(ms.id == "e2e_lf_ring4", "synthesis model id")
    var mr = load_model_file(stage + "/route/e2e_route")
    expect(mr.id == "e2e_route", "router id")

    # an explicit pipeline names the staged models: QM-E617 / QM-E620 would be raised if they were not found or fit
    var c = Circuit(4, 0)
    for k in range(12):
        c.cx(k % 4, (k + 1) % 4)
    var ring: List[Int] = [0, 1, 1, 2, 2, 3, 0, 3]
    var t = Target.from_edges(4, ring)
    var spec = (
        "ai_linear(model=e2e_lf_ring4,model_dir=" + stage + ",method=policy,beams=4,searches=2),"
        + "route(sabre,policy=e2e_route,model_dir=" + stage + ")"
    )
    var res = compile_spec(c, t, spec, 7)
    expect(len(res.reports) >= 1, "the explicit pipeline ran no pass")
    print("  A7 explicit pipeline with the staged models compiled without QM-E617 / QM-E620")

    # the replay parts of the two inference tests against the staging root
    sh(
        "QUMO_TEST_MODEL_ROOT=" + shell_quote(stage) + " " + mojo + " run -I . tests/test_policy.mojo",
        work + "/replay_policy.txt",
        "tests/test_policy.mojo on the staging root",
    )
    expect(_tail(work + "/replay_policy.txt", 3).find("test_policy OK") >= 0, "test_policy did not print OK")
    sh(
        "QUMO_TEST_MODEL_ROOT=" + shell_quote(stage) + " " + mojo + " run -I . tests/test_ai_route.mojo",
        work + "/replay_route.txt",
        "tests/test_ai_route.mojo on the staging root",
    )
    expect(_tail(work + "/replay_route.txt", 3).find("test_ai_route OK") >= 0, "test_ai_route did not print OK")
    print("  A7 test_policy and test_ai_route pass with QUMO_TEST_MODEL_ROOT =", stage)


def main() raises:
    var build = _env("QUMO_E2E_BUILD", "build/ai_train")
    var bin = build + "/bin"
    var work = build + "/e2e"
    var mojo = _env("MOJO", ".venv/bin/mojo")
    var iters = Int(_env("QUMO_E2E_ITERS", "30"))
    makedirs(work, exist_ok=True)
    var tools: List[String] = ["train_synth", "eval_synth", "route_tool", "train_route", "export_route", "models_verify"]
    ensure_tools(build, tools, work + "/build_log.txt")
    # a fresh start: stale runs would make the sha comparison meaningless
    remove_tree(work + "/route")
    var cs: List[Int] = [1, 2, 4, 8, 4]
    for i in range(len(cs)):
        remove_tree(work + "/det" + String(i) + "_c" + String(cs[i]))
    test_determinism(bin, work, iters)
    make_teacher(bin, work)
    test_route_determinism(bin, work)
    test_staged_root(bin, work, mojo)
    print("test_bridge_e2e OK")
