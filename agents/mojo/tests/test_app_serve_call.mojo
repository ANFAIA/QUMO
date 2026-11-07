"""Executable to executable: `qumo-agent serve` as an MCP server for every agent, and
`qumo-agent call`, which starts `serve` as a child over pipes and calls its `run` tool
(DESIGN.md 7.2). The model is replayed; QUMO tool calls reach the real `build/qumo mcp`."""

from std.testing import assert_equal, assert_true, assert_false

from qumo.ai.json import parse_json

from qumo_agent.app.testing import (
    app_binary, app_env, run_app, replay_line, write_replay, REPLAY_BELL, BELL_ANSWER,
)
from qumo_agent.mcp.stdio_client import McpStdioSession, StdioServerParams
from qumo_agent.native.fs import make_temp_dir, remove_tree
from qumo_agent.native.sandbox import Confinement
from qumo_agent.native.spawn import STDIO_INHERIT


def _session(args: List[String]) raises -> McpStdioSession:
    var bin = app_binary()
    var env = app_env(List[String]())
    return McpStdioSession(StdioServerParams(bin, args.copy(), env^, String(), 60000, 16 << 20,
                                             STDIO_INHERIT, Optional[Confinement](None), String()))


def test_serve() raises:
    # 1. The agent as a library: an MCP client talks to `serve`.
    var c = _session(["serve", "qumo-operator", "--replay", REPLAY_BELL, "--quiet"])
    c.initialize("test", "1")
    assert_equal(c.server_name, "qumo-agent-qumo-operator")
    var tools = c.list_tools()
    assert_equal(len(tools), 1)
    assert_equal(tools[0].name, "run")
    assert_true(tools[0].description.startswith("QUMO agent 'qumo-operator' (Mojo, model anthropic/claude-sonnet-5.5): "))
    assert_equal(parse_json(tools[0].input_schema_json).get("required").at(0).as_string(), "task")
    var r = c.call_tool("run", "{\"task\":\"Run a Bell state with seed 7\"}")
    assert_false(r.is_error, String("\n").join(r.parts))
    assert_equal(r.parts[0], BELL_ANSWER)
    var s = parse_json(r.structured_json)
    assert_equal(s.get_string("agent", ""), "qumo-operator")
    assert_equal(s.get_string("model", ""), "anthropic/claude-sonnet-5.5")
    assert_equal(s.get_string("answer", ""), BELL_ANSWER)
    assert_equal(s.get("stats").get_int("mcp_calls", -1), 2)
    assert_equal(s.get("stats").get_int("model_calls", -1), 3)
    # The replay is exhausted now: a second run is an isError result, not a dead server.
    var again = c.call_tool("run", "{\"task\":\"again\"}")
    assert_true(again.is_error)
    assert_true(String("\n").join(again.parts).find("replay file exhausted") >= 0, String("\n").join(again.parts))
    # An unknown tool and bad arguments are protocol errors.
    var bad = False
    try:
        _ = c.call_tool("nope", "{}")
    except e:
        bad = String(e).find("-32602") >= 0
    assert_true(bad)
    bad = False
    try:
        _ = c.call_tool("run", "{\"task\": 5}")
    except e:
        bad = String(e).find("-32602") >= 0
    assert_true(bad)
    assert_equal(c.close(), 0)

    # 2. `serve` works for the agents the old port refused, and refuses an unknown one.
    var dir = make_temp_dir("qumo-app-serve-")
    var lines = List[String]()
    lines.append(replay_line("", List[String](), "papers: none new"))
    write_replay(dir + "/watch.jsonl", lines)
    var w = _session(["serve", "research-watch", "--replay", dir + "/watch.jsonl", "--quiet"])
    w.initialize("test", "1")
    assert_equal(w.server_name, "qumo-agent-research-watch")
    var wr = w.call_tool("run", "{\"task\":\"anything new?\"}")
    assert_false(wr.is_error)
    assert_equal(wr.parts[0], "papers: none new")
    assert_equal(w.close(), 0)

    var res = run_app(["serve", "no-such-agent", "--replay", REPLAY_BELL], app_env(List[String]()), "", 20000)
    assert_equal(res.exit_code, 3)
    assert_true(res.stderr.find("unknown agent 'no-such-agent'") >= 0, res.stderr)
    remove_tree(dir)


def test_call() raises:
    # `call` spawns `serve` itself and prints the answer; the key is never printed.
    var secret = "sk-or-call-secret-777"
    var env = app_env(["OPENROUTER_API_KEY=" + secret])
    var res = run_app(["call", "qumo-operator", "Run a Bell state with seed 7", "--replay", REPLAY_BELL, "--quiet"], env, "", 60000)
    assert_equal(res.exit_code, 0, res.stderr)
    assert_equal(res.stdout, BELL_ANSWER + "\n")
    assert_true(res.stderr.find("[qumo-agent call] qumo-agent-qumo-operator exit=0 wall_s=") >= 0, res.stderr)
    assert_true(res.stderr.find("\"model_calls\":3") >= 0, res.stderr)
    assert_false(res.stderr.find(secret) >= 0)
    assert_false(res.stdout.find(secret) >= 0)

    # --module and a task from stdin work through `call` too.
    res = run_app(["call", "qumo-operator", "-", "--replay", REPLAY_BELL, "--quiet", "--module", "ir"], env, "check the IR\n", 60000)
    assert_equal(res.exit_code, 0, res.stderr)
    assert_equal(res.stdout, BELL_ANSWER + "\n")

    # A failing run in the child is exit code 1 with the child's message.
    var dir = make_temp_dir("qumo-app-call-")
    write_replay(dir + "/empty.jsonl", List[String]())
    res = run_app(["call", "qumo-operator", "x", "--replay", dir + "/empty.jsonl", "--quiet"], env, "", 60000)
    assert_equal(res.exit_code, 1)
    assert_true(res.stdout.find("qumo-agent qumo-operator failed: QA-E170: replay file exhausted") >= 0, res.stdout)

    # Another agent through `call`.
    var lines = List[String]()
    lines.append(replay_line("", List[String](), "no regressions found"))
    write_replay(dir + "/review.jsonl", lines)
    res = run_app(["call", "code-review", "review it", "--replay", dir + "/review.jsonl", "--quiet"], env, "", 60000)
    assert_equal(res.exit_code, 0, res.stderr)
    assert_equal(res.stdout, "no regressions found\n")
    remove_tree(dir)


def main() raises:
    test_serve()
    test_call()
    print("test_app_serve_call OK")
