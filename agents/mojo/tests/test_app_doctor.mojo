"""Tests `qumo-agent doctor` (DESIGN.md 7.2): one `[ok]`/`[--]` line per check, names of
variables but never values, the OS sandbox kind, generated tools (reviewed, unreviewed,
refused with reasons), and the exit code (0 when a run can work). The network check runs
against a scripted local server standing in for the model endpoint."""

from std.os import stat
from std.os.path import exists
from std.testing import assert_equal, assert_true, assert_false

from qumo.ai.json import JsonValue, parse_json

from qumo_agent.app.testing import app_env, run_app, REPLAY_BELL
from qumo_agent.native.clock import monotonic_ms, sleep_ms
from qumo_agent.native.fs import make_dirs, make_temp_dir, read_text_lossy, remove_tree, write_text
from qumo_agent.native.libc import CStr, c_rename
from qumo_agent.native.rand import uuid4_hex
from qumo_agent.native.spawn import (
    Child, SpawnSpec, spawn, run_capture, minimal_child_env, environ_list, STDIO_NULL, STDIO_PIPE, STDIO_INHERIT,
)

comptime _FIXTURE_SRC = "agents/mojo/tests/support/http_fixture_server.mojo"
comptime _FIXTURE_BIN = "build/test-fixtures/http_fixture_server"
comptime SECRET = "sk-or-doctor-secret-4242"


def _mtime(path: String) raises -> Int:
    var st = stat(path)
    return Int(st.st_mtimespec.tv_sec) * 1000000000 + Int(st.st_mtimespec.tv_nsec)


def fixture_binary() raises -> String:
    """Builds build/test-fixtures/http_fixture_server unless it is newer than its source."""
    if exists(_FIXTURE_BIN) and _mtime(_FIXTURE_BIN) > _mtime(_FIXTURE_SRC):
        return _FIXTURE_BIN
    make_dirs("build/test-fixtures")
    var tmp = _FIXTURE_BIN + ".tmp-" + uuid4_hex()
    var argv: List[String] = [".venv/bin/mojo", "build", "-I", ".", "-I", "agents/mojo", "-o", tmp, _FIXTURE_SRC]
    var r = run_capture(SpawnSpec(".venv/bin/mojo", argv^, environ_list(), "", STDIO_NULL, STDIO_PIPE, STDIO_PIPE, False),
                        600000, 1 << 16, "")
    if r.exit_code != 0:
        raise Error("building the fixture server failed: " + r.stdout + r.stderr)
    var a = CStr(tmp)
    var b = CStr(_FIXTURE_BIN)
    _ = c_rename(a.addr(), b.addr())
    a.keep()
    b.keep()
    return _FIXTURE_BIN


def start_fixture(dir: String, script: String) raises -> Tuple[Child, Int]:
    var bin = fixture_binary()
    write_text(dir + "/script.json", script)
    var argv: List[String] = [bin, "--script", dir + "/script.json", "--port-file", dir + "/port", "--log", dir + "/log.jsonl"]
    var child = spawn(SpawnSpec(bin, argv^, minimal_child_env(), "", STDIO_NULL, STDIO_INHERIT, STDIO_INHERIT, True))
    var deadline = monotonic_ms() + 30000
    while not exists(dir + "/port"):
        if monotonic_ms() > deadline:
            raise Error("the fixture server did not write its port")
        sleep_ms(10)
    return (child^, Int(String(read_text_lossy(dir + "/port").strip())))


def agents_config(dir: String, base_url: String) raises -> String:
    """A copy of agents.json whose model endpoint is `base_url`."""
    var text = read_text_lossy("agents/agents.json")
    text = text.replace("https://openrouter.ai/api/v1", base_url)
    var path = dir + "/agents.json"
    write_text(path, text)
    return path


def test_offline() raises:
    var dir = make_temp_dir("qumo-app-doctor-")
    var env = app_env(["QUMO_AGENT_TOOLS_DIR=" + dir + "/store", "OPENROUTER_API_KEY=" + SECRET])
    var r = run_app(["doctor", "--offline", "--replay", REPLAY_BELL], env, "", 60000)
    assert_equal(r.exit_code, 0, r.stdout + r.stderr)
    var out = r.stdout
    assert_true(out.find("[ok] repository root") >= 0, out)
    assert_true(out.find("[ok] QUMO MCP server") >= 0, out)
    assert_true(out.find("answers initialize") >= 0, out)
    assert_true(out.find("[ok] exported prompts and specs are current") >= 0, out)
    assert_true(out.find("OS sandbox for model-written code: ") >= 0, out)
    assert_true(out.find("sandbox-exec") >= 0 or out.find("landlock") >= 0 or out.find("none") >= 0, out)
    assert_true(out.find("[ok] model key OPENROUTER_API_KEY is set") >= 0, out)
    assert_true(out.find("[--] alphaXiv papers (ALPHAXIV_API_KEY): optional") >= 0, out)
    assert_true(out.find("[--] GitHub token (GITHUB_TOKEN): optional") >= 0, out)
    assert_true(out.find("git in PATH") >= 0, out)
    assert_true(out.find("posix_spawn has a chdir action") >= 0, out)
    assert_true(out.find("generated tools in " + dir + "/store: 0 reviewed, 0 unreviewed") >= 0, out)
    assert_true(out.find("a run can work: yes") >= 0, out)
    assert_false(out.find("model endpoint") >= 0)            # --offline skips the network check
    assert_false(out.find(SECRET) >= 0)
    assert_false(r.stderr.find(SECRET) >= 0)

    # Without a key and without a replay, an offline run cannot work.
    var env2 = app_env(["QUMO_AGENT_TOOLS_DIR=" + dir + "/store"])
    r = run_app(["doctor", "--offline"], env2, "", 60000)
    assert_equal(r.exit_code, 1)
    assert_true(r.stdout.find("[--] model key OPENROUTER_API_KEY is set: export the key") >= 0, r.stdout)
    assert_true(r.stdout.find("a run can work: no") >= 0)

    # A replay is enough for an offline run.
    r = run_app(["doctor", "--offline", "--replay", REPLAY_BELL], env2, "", 60000)
    assert_equal(r.exit_code, 0, r.stdout)

    # A missing QUMO MCP server is a blocking problem.
    var env3 = app_env(["QUMO_BIN=/nonexistent/qumo", "QUMO_AGENT_TOOLS_DIR=" + dir + "/store"])
    r = run_app(["doctor", "--offline", "--replay", REPLAY_BELL], env3, "", 60000)
    assert_equal(r.exit_code, 1)
    assert_true(r.stdout.find("[--] QUMO MCP server /nonexistent/qumo answers initialize: /nonexistent/qumo not found") >= 0, r.stdout)
    remove_tree(dir)


def test_network_check() raises:
    var dir = make_temp_dir("qumo-app-doctor-net-")
    # A model endpoint that answers /models/count: the check passes and no key is sent.
    var script = '{"responses": [{"match": {"method": "GET", "path": "/api/v1/models/count"}, "status": 200, "reason": "OK", "headers": {"Content-Type": "application/json"}, "body": "{\\"data\\": {\\"count\\": 3}}"}]}'
    var fx = start_fixture(dir, script)
    var base = "http://127.0.0.1:" + String(fx[1]) + "/api/v1"
    var cfg = agents_config(dir, base)
    var env = app_env(["QUMO_AGENTS_CONFIG=" + cfg, "OPENROUTER_API_KEY=" + SECRET, "QUMO_AGENT_TOOLS_DIR=" + dir + "/store"])
    var r = run_app(["doctor"], env, "", 60000)
    assert_equal(r.exit_code, 0, r.stdout + r.stderr)
    assert_true(r.stdout.find("[ok] model endpoint " + base + " reachable") >= 0, r.stdout)
    assert_true(r.stdout.find("a run can work: yes") >= 0)
    var log = read_text_lossy(dir + "/log.jsonl")
    assert_true(log.find("/api/v1/models/count") >= 0, log)
    assert_false(log.find(SECRET) >= 0)                        # no key was sent
    assert_false(log.find("authorization") >= 0)
    _ = fx[0].terminate(200)

    # An endpoint that refuses the connection: the line says why, and the run cannot work.
    var cfg2 = agents_config(dir, "http://127.0.0.1:1/api/v1")
    var env2 = app_env(["QUMO_AGENTS_CONFIG=" + cfg2, "OPENROUTER_API_KEY=" + SECRET, "QUMO_AGENT_TOOLS_DIR=" + dir + "/store"])
    r = run_app(["doctor"], env2, "", 60000)
    assert_equal(r.exit_code, 1)
    assert_true(r.stdout.find("[--] model endpoint http://127.0.0.1:1/api/v1 reachable") >= 0, r.stdout)
    assert_false(r.stdout.find(SECRET) >= 0)
    remove_tree(dir)


def test_generated_tools_lines() raises:
    """Unreviewed and refused tools are listed with their reasons."""
    var dir = make_temp_dir("qumo-app-doctor-gen-")
    var store = dir + "/store"
    make_dirs(store + "/gen_demo/0.1.0-aaaaaaaaaaaa")
    write_text(store + "/gen_demo/0.1.0-aaaaaaaaaaaa/tool.mojo", "x")
    var manifest = ('{"schema": "qumo.agent_tools.v1", "tools": {"gen_demo": {"name": "gen_demo", "version": "0.1.0", '
                    + '"description": "d", "dir": "gen_demo/0.1.0-aaaaaaaaaaaa", "source_sha256": "' + ("0" * 64) + '", '
                    + '"cases_sha256": "' + ("0" * 64) + '", "binary_sha256": "' + ("a" * 64) + '", "sdk_sha256": "", '
                    + '"mojo_version": "", "platform": "darwin-arm64", "timeout_s": 30, "max_output_bytes": 1048576, '
                    + '"published_at": "", "published_by": "mojo-toolsmith", "sandbox_branch": "", "tests": {"passed": 1, "total": 1}, '
                    + '"reviewed": false, "reviewed_by": "", "reviewed_at": ""}}}')
    write_text(store + "/manifest.json", manifest)
    var env = app_env(["QUMO_AGENT_TOOLS_DIR=" + store, "OPENROUTER_API_KEY=" + SECRET])
    var r = run_app(["doctor", "--offline", "--replay", REPLAY_BELL], env, "", 60000)
    assert_equal(r.exit_code, 0, r.stdout)
    assert_true(r.stdout.find("[--] generated tools in " + store) >= 0, r.stdout)
    assert_true(r.stdout.find("refused: QA-E150: gen_demo:") >= 0, r.stdout)
    remove_tree(dir)


def main() raises:
    test_offline()
    test_network_check()
    test_generated_tools_lines()
    print("test_app_doctor OK")
