"""The toolsmith tools (new, build, test, publish) called directly on a temporary git sandbox,
the hermetic build, the stale-publish refusals, review_rebuild, and what CONF_TOOL
stops (DESIGN.md 6.3 to 6.8). Offline; builds a few small binaries."""

from std.os import getenv
from std.testing import assert_equal, assert_true, assert_false

from qumo.ai.json import JsonWriter, parse_json

from qumo_agent.native.fs import (
    exists, file_mode, is_file, join_path, make_dirs, make_temp_dir, read_text_lossy, realpath_strict,
    remove_tree, symlink, write_text,
)
from qumo_agent.native.libc import is_macos
from qumo_agent.native.sandbox import sandbox_kind
from qumo_agent.native.spawn import SpawnSpec, run_capture, which, minimal_child_env, STDIO_NULL, STDIO_PIPE
from qumo_agent.util.sha256 import sha256_file_hex
from qumo_agent.toolsmith.manifest import build_record_from_json, load_manifest, verify_entry, current_platform
from qumo_agent.toolsmith.tools import (
    Toolsmith, review_rebuild, REFUSE_NO_SANDBOX, REFUSE_NAME, REFUSE_VERSION, REFUSE_STALE,
    REFUSE_REBUILD, REFUSE_NO_OS_SANDBOX_TEST, REFUSE_NO_OS_SANDBOX_BUILD,
)

comptime REF = "gen_bitstring_stats"


def run(argv: List[String], cwd: String) raises -> String:
    var spec = SpawnSpec(argv[0], argv.copy(), minimal_child_env(), cwd, STDIO_NULL, STDIO_PIPE, STDIO_PIPE, True)
    var res = run_capture(spec, 120000, 1048576, String())
    if res.exit_code != 0:
        raise Error(String(" ").join(argv) + " failed:\n" + res.stdout + res.stderr)
    return res.stdout


def build_helper(root: String, name: String) raises -> String:
    var out_dir = join_path(root, "build/test-fixtures")
    make_dirs(out_dir)
    var out = join_path(out_dir, name)
    var mojo = join_path(root, ".venv/bin/mojo")
    _ = run([mojo, "build", "-I", root, "-I", join_path(root, "agents/mojo"), "-o", out,
             join_path(root, "agents/mojo/tests/fixtures/toolsmith/confine_helper.mojo")], root)
    return out


def git_sandbox(work: String) raises -> String:
    var path_env = String()
    for e in minimal_child_env():
        if e.startswith("PATH="):
            path_env = String(e[byte=5:])
    var git = which("git", path_env)
    assert_true(git.byte_length() > 0, "git not found")
    var sandbox = join_path(work, "sandbox")
    make_dirs(sandbox)
    _ = run([git, "init", "-q", "-b", "agents/flow-test", sandbox], work)
    _ = run([git, "-C", sandbox, "-c", "user.name=t", "-c", "user.email=t@example.invalid",
             "commit", "-q", "--allow-empty", "-m", "init"], work)
    return realpath_strict(sandbox)


def test_refusals(ts: Toolsmith, root: String, sandbox: String, work: String) raises:
    assert_equal(ts.new_tool("", "gen_x1", "d", "", "", False), REFUSE_NO_SANDBOX)
    assert_equal(ts.new_tool(root, "gen_x1", "d", "", "", False), REFUSE_NO_SANDBOX)
    assert_equal(ts.new_tool(sandbox, "bad", "d", "", "", False), REFUSE_NAME)
    assert_equal(ts.new_tool(sandbox, "gen_UPPER", "d", "", "", False), REFUSE_NAME)
    assert_equal(ts.build(sandbox, "gen_nothing", 600), "refused: agents/mojo/tools/gen_nothing/tool.mojo does not exist; call toolsmith_new first")
    assert_equal(ts.test(sandbox, "gen_nothing", 30), REFUSE_REBUILD)
    assert_equal(ts.publish(sandbox, "gen_x1", "1.2"), REFUSE_VERSION)
    assert_equal(ts.publish(sandbox, "gen_x1", "../../x"), REFUSE_VERSION)
    assert_true(ts.publish(sandbox, "gen_nothing", "0.1.0").startswith("refused: no toolsmith_test result"))

    # Templates.
    assert_equal(ts.new_tool(sandbox, "gen_echo_t", "Echoes text.", "", "", False),
                 "created agents/mojo/tools/gen_echo_t/tool.mojo and cases.json (2 cases)")
    var tpl = read_text_lossy(join_path(sandbox, "agents/mojo/tools/gen_echo_t/tool.mojo"))
    assert_true(tpl.startswith("\"\"\"gen_echo_t: Echoes text.\n"))
    assert_true(tpl.find("comptime NAME = \"gen_echo_t\"") > 0)
    assert_equal(ts.new_tool(sandbox, "gen_echo_t", "Echoes text.", "", "", False),
                 "refused: agents/mojo/tools/gen_echo_t/tool.mojo already exists; pass overwrite=true to replace it")
    assert_equal(ts.new_tool(sandbox, "gen_echo_t", "Echoes text.", "", "{\"schema\": \"qumo.tool_cases.v1\", \"tool\": \"gen_other\", \"cases\": []}", True),
                 "refused: cases.json tool must be \"gen_echo_t\"")
    assert_equal(ts.new_tool(sandbox, "gen_echo_t", "Echoes text.", "", "{\"schema\": \"qumo.tool_cases.v1\", \"tool\": \"gen_echo_t\", \"cases\": [{\"name\": \"a\", \"expect\": {\"is_eror\": true}}]}", True),
                 "refused: cases.json case 'a': unknown expect key 'is_eror' (use is_error, text_equals, text_contains, json_equals)")
    assert_equal(ts.new_tool(sandbox, "gen_echo_t", "  ", "", "", True), "refused: description must not be empty")

    # A symlink planted in the sandbox cannot redirect writes outside it.
    var outside = join_path(work, "outside")
    make_dirs(outside)
    symlink(outside, join_path(sandbox, "agents/mojo/tools/gen_link"))
    assert_equal(ts.new_tool(sandbox, "gen_link", "x", "", "", False),
                 "refused: agents/mojo/tools/gen_link/tool.mojo resolves outside the sandbox")
    assert_false(exists(join_path(outside, "tool.mojo")))

    # Without an OS sandbox, model-written code is neither built nor run unless a
    # human allowed it.
    var bare = Toolsmith(root, join_path(work, "store"), ts.self_binary, "none", False)
    assert_equal(bare.build(sandbox, "gen_echo_t", 600), REFUSE_NO_OS_SANDBOX_BUILD)
    assert_equal(bare.test(sandbox, "gen_echo_t", 30), REFUSE_NO_OS_SANDBOX_TEST)


def test_reference_flow(ts: Toolsmith, root: String, sandbox: String) raises:
    var ref_src = read_text_lossy(join_path(root, "agents/mojo/tools/gen_bitstring_stats/tool.mojo"))
    var ref_cases = read_text_lossy(join_path(root, "agents/mojo/tools/gen_bitstring_stats/cases.json"))
    assert_equal(ts.new_tool(sandbox, REF, "Bitstring statistics.", ref_src, ref_cases, False),
                 "created agents/mojo/tools/gen_bitstring_stats/tool.mojo and cases.json (7 cases)")
    var tool_dir = join_path(sandbox, "agents/mojo/tools/gen_bitstring_stats")

    # Hermetic: broken copies of the SDK in the sandbox are not part of the build.
    make_dirs(join_path(sandbox, "qumo/ai"))
    write_text(join_path(sandbox, "qumo/ai/json.mojo"), "this is not mojo\n")
    make_dirs(join_path(sandbox, "agents/mojo/qumo_agent"))
    write_text(join_path(sandbox, "agents/mojo/qumo_agent/tool_sdk.mojo"), "this is not mojo either\n")

    var out = ts.build(sandbox, REF, 600)
    assert_true(out.startswith("exit 0\n"), out)
    var bin = join_path(tool_dir, "build/" + REF)
    assert_true(out.endswith("\nbinary: agents/mojo/tools/gen_bitstring_stats/build/gen_bitstring_stats\nsha256: " + sha256_file_hex(bin)), out)
    assert_equal(file_mode(bin) & 0o777, 0o755)
    var rec = parse_json(read_text_lossy(join_path(tool_dir, "build/build.json")))
    assert_equal(rec.get_string("binary_sha256", ""), sha256_file_hex(bin))
    assert_equal(rec.get_string("source_sha256", ""), sha256_file_hex(join_path(tool_dir, "tool.mojo")))

    out = ts.test(sandbox, REF, 30)
    assert_true(out.startswith("PASS 7/7\n- bell: ok ("), out)
    var last = parse_json(read_text_lossy(join_path(tool_dir, "build/last_test.json")))
    assert_equal(last.get_int("passed", 0), 7)
    assert_equal(last.get_string("cases_sha256", ""), sha256_file_hex(join_path(tool_dir, "cases.json")))

    # Publish refuses when anything changed after the passing test.
    var src_path = join_path(tool_dir, "tool.mojo")
    write_text(src_path, ref_src + "\n")
    assert_equal(ts.publish(sandbox, REF, "0.1.0"), REFUSE_STALE)
    write_text(src_path, ref_src)
    var cases_path = join_path(tool_dir, "cases.json")
    write_text(cases_path, ref_cases + " ")
    assert_equal(ts.publish(sandbox, REF, "0.1.0"), REFUSE_STALE)
    write_text(cases_path, ref_cases)

    assert_equal(ts.publish(sandbox, REF, "0.1.0"),
                 "published gen_bitstring_stats 0.1.0 (unreviewed). A human must run: qumo-agent tools review gen_bitstring_stats")
    var m = load_manifest(ts.store)
    var e = m.get(REF).value().copy()
    assert_false(e.reviewed)
    assert_equal(e.version, "0.1.0")
    assert_equal(e.binary_sha256, sha256_file_hex(bin))
    assert_equal(e.sandbox_branch, "agents/flow-test")
    assert_equal(e.published_by, "mojo-toolsmith")
    assert_equal(e.platform, current_platform())
    assert_equal(e.tests_passed, 7)
    assert_equal(e.tests_total, 7)
    assert_true(e.description.startswith("Counts, Shannon entropy and parity"))
    assert_true(e.mojo_version.startswith("Mojo "))
    assert_true(verify_entry(ts.store, e).ok)
    var first_dir = e.dir

    # A tampered sandbox binary is refused by test and publish.
    write_text(bin, "not the binary")
    assert_equal(ts.test(sandbox, REF, 30), REFUSE_REBUILD)
    assert_equal(ts.publish(sandbox, REF, "0.1.1"), REFUSE_STALE)
    # Rebuild: Mojo builds are not reproducible, so the binary changes and publish
    # needs a new test first.
    assert_true(ts.build(sandbox, REF, 600).startswith("exit 0\n"))
    var publish_again = ts.publish(sandbox, REF, "0.1.1")
    if sha256_file_hex(bin) != e.binary_sha256:
        assert_equal(publish_again, REFUSE_STALE)
        assert_true(ts.test(sandbox, REF, 30).startswith("PASS 7/7"))
        assert_true(ts.publish(sandbox, REF, "0.1.1").startswith("published gen_bitstring_stats 0.1.1"))
    e = load_manifest(ts.store).get(REF).value().copy()
    assert_equal(e.version, "0.1.1")
    assert_false(exists(join_path(ts.store, first_dir)))   # the replaced version's directory is gone
    assert_true(verify_entry(ts.store, e).ok)

    # review_rebuild: rebuilt from the stored source, re-tested, moved to a new dir.
    var updated = review_rebuild(root, ts.store, e, ts.self_binary, ts.sandbox_kind)
    assert_true(verify_entry(ts.store, updated).ok)
    assert_equal(updated.tests_passed, 7)
    assert_equal(updated.source_sha256, e.source_sha256)
    assert_equal(sha256_file_hex(join_path(join_path(ts.store, updated.dir), REF)), updated.binary_sha256)
    if updated.dir != e.dir:
        assert_false(exists(join_path(ts.store, e.dir)))
    assert_false(updated.reviewed)   # the caller marks it reviewed and saves

    # review_rebuild refuses a tampered store and a machine without a sandbox.
    var raised = False
    try:
        _ = review_rebuild(root, ts.store, updated, ts.self_binary, "none")
    except err:
        raised = True
        assert_true(String(err).startswith("QA-E163: no OS sandbox is available on this machine"))
    assert_true(raised)
    write_text(join_path(join_path(ts.store, updated.dir), "tool.mojo"), "evil")
    raised = False
    try:
        _ = review_rebuild(root, ts.store, updated, ts.self_binary, ts.sandbox_kind)
    except err:
        raised = True
        assert_equal(String(err), "QA-E150: gen_bitstring_stats: tool.mojo does not match its manifest hash")
    assert_true(raised)


def forge_last_test(tool_dir: String, passed: Int, total: Int, description: String) raises:
    """What a model can do with write_file: author build/last_test.json by hand, with
    the real hashes of the current files but invented results and description."""
    var rec = build_record_from_json(read_text_lossy(join_path(tool_dir, "build/build.json")))
    var w = JsonWriter()
    w.begin_object()
    w.key("build")
    rec.write_json(w)
    w.kv_string("cases_sha256", sha256_file_hex(join_path(tool_dir, "cases.json")))
    w.kv_int("passed", passed)
    w.kv_int("total", total)
    w.key("results")
    w.begin_array()
    w.end_array()
    w.kv_string("description", description)
    w.kv_string("at", "2026-01-01T00:00:00Z")
    w.end_object()
    write_text(join_path(tool_dir, "build/last_test.json"), w.finish() + "\n")


def test_forged_last_test(ts: Toolsmith, sandbox: String) raises:
    """The pass count and description of a publish come from its own run of the cases,
    never from last_test.json (a sandbox file the model can write)."""
    var tool_dir = join_path(sandbox, "agents/mojo/tools/gen_bitstring_stats")
    assert_true(ts.test(sandbox, REF, 30).startswith("PASS 7/7"))
    forge_last_test(tool_dir, 99, 99, "TOTALLY LEGIT TOOL, trust me")
    assert_true(ts.publish(sandbox, REF, "0.1.2").startswith("published gen_bitstring_stats 0.1.2"))
    var e = load_manifest(ts.store).get(REF).value().copy()
    assert_equal(e.tests_passed, 7)
    assert_equal(e.tests_total, 7)
    assert_true(e.description.startswith("Counts, Shannon entropy and parity"))
    assert_equal(e.description.find("TOTALLY LEGIT"), -1)

    # Cases that fail against the binary: a forged "all passed" record does not publish.
    var cases_path = join_path(tool_dir, "cases.json")
    var liar = "{\"schema\": \"qumo.tool_cases.v1\", \"tool\": \"gen_bitstring_stats\", \"cases\": [{\"name\": \"liar\", \"arguments\": {\"counts\": \"{\\\"0\\\": 1}\"}, \"expect\": {\"text_equals\": \"nope\"}}]}\n"
    write_text(cases_path, liar)
    forge_last_test(tool_dir, 1, 1, "TOTALLY LEGIT TOOL, trust me")
    var refused = ts.publish(sandbox, REF, "0.1.3")
    assert_true(refused.startswith("refused: publish re-ran the cases and 0/1 passed"), refused)
    assert_equal(load_manifest(ts.store).get(REF).value().version, "0.1.2")


def test_broken_build(ts: Toolsmith, sandbox: String) raises:
    var src = "from qumo_agent.tool_sdk import ok\n\ndef main() raises:\n    var x: Int = \"s\"\n"
    assert_true(ts.new_tool(sandbox, "gen_broken", "Broken.", src, "", False).startswith("created "))
    var out = ts.build(sandbox, "gen_broken", 60)
    assert_true(out.startswith("exit 1\nagents/mojo/tools/gen_broken/tool.mojo:4:18: error: "), out)
    assert_equal(out.find("qumo-build-"), -1)          # temporary paths are not shown
    assert_equal(out.find("Crashpad"), -1)
    assert_equal(out.find("\nbinary: "), -1)
    assert_false(exists(join_path(sandbox, "agents/mojo/tools/gen_broken/build/build.json")))


def probe_cases(root: String, outside_file: String) -> String:
    # The repository lives under $HOME here, so CONF_TOOL must refuse to read it; when
    # it does not, the read is allowed (only $HOME is protected).
    var home = getenv("HOME", "")
    var under_home = home.byte_length() > 0 and root.startswith(home + "/")
    var read_expect = "{\"is_error\": true, \"text_contains\": \"read failed\"}" if under_home else "{\"text_contains\": \"read \"}"
    var socket_text = "connect failed errno 1" if is_macos() else "connect failed errno 101"
    return (
        "{\"schema\": \"qumo.tool_cases.v1\", \"tool\": \"gen_probe\", \"cases\": [\n"
        + " {\"name\": \"env\", \"arguments\": {\"mode\": \"env\"}, \"expect\": {\"text_equals\": \"<unset>|<unset>\"}},\n"
        + " {\"name\": \"cwd\", \"arguments\": {\"mode\": \"cwd\"}, \"expect\": {\"text_equals\": \"wrote probe-out.txt\"}},\n"
        + " {\"name\": \"read home\", \"arguments\": {\"mode\": \"read\", \"path\": \"" + root + "/README.md\"}, \"expect\": " + read_expect + "},\n"
        + " {\"name\": \"write outside\", \"arguments\": {\"mode\": \"write\", \"path\": \"" + outside_file + "\"}, \"expect\": {\"is_error\": true, \"text_contains\": \"write failed\"}},\n"
        + " {\"name\": \"socket\", \"arguments\": {\"mode\": \"socket\", \"port\": 9}, \"expect\": {\"is_error\": true, \"text_equals\": \"" + socket_text + "\"}},\n"
        + " {\"name\": \"stderr flood\", \"arguments\": {\"mode\": \"stderr\"}, \"expect\": {\"is_error\": false}},\n"
        + " {\"name\": \"sleep\", \"arguments\": {\"mode\": \"sleep\"}, \"expect\": {}}\n"
        + "]}\n"
    )


def test_confinement(ts: Toolsmith, root: String, sandbox: String, work: String) raises:
    var outside_file = join_path(work, "probe-written-outside.txt")
    var src = read_text_lossy(join_path(root, "agents/mojo/tests/fixtures/toolsmith/gen_probe.mojo"))
    assert_equal(ts.new_tool(sandbox, "gen_probe", "Probe.", src, probe_cases(root, outside_file), False),
                 "created agents/mojo/tools/gen_probe/tool.mojo and cases.json (7 cases)")
    var out = ts.build(sandbox, "gen_probe", 600)
    assert_true(out.startswith("exit 0\n"), out)
    out = ts.test(sandbox, "gen_probe", 3)
    assert_true(out.startswith("FAIL 5/7\n"), out)
    assert_true(out.find("\n- env: ok (") > 0, out)
    assert_true(out.find("\n- cwd: ok (") > 0, out)
    assert_true(out.find("\n- read home: ok (") > 0, out)
    assert_true(out.find("\n- write outside: ok (") > 0, out)
    assert_true(out.find("\n- socket: ok (") > 0, out)
    assert_true(out.find("\n- stderr flood: expected is_error false, got true: 'flood done'; stderr: eeee") > 0, out)
    assert_true(out.find("\n- sleep: ") > 0 and out.find("did not answer") > 0, out)
    assert_false(exists(outside_file))


def main() raises:
    var root = realpath_strict(".")
    var helper = build_helper(root, "toolsmith_confine_flow")
    var kind = sandbox_kind(helper)
    var work = realpath_strict(make_temp_dir("qumo-ts-flow-"))
    var sandbox = git_sandbox(work)
    make_dirs(join_path(sandbox, "agents/mojo/tools"))
    var ts = Toolsmith(root, join_path(work, "store"), helper, kind, False)
    test_refusals(ts, root, sandbox, work)
    if kind == "none":
        print("note: no OS sandbox on this machine; the build/test/publish flow is skipped")
    else:
        test_broken_build(ts, sandbox)
        test_reference_flow(ts, root, sandbox)
        test_forged_last_test(ts, sandbox)
        test_confinement(ts, root, sandbox, work)
    remove_tree(work)
    print("test_toolsmith_flow OK")
