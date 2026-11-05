"""Executable to executable: `qumo-agent serve` as an MCP server, and `qumo-agent call`,
which starts serve as a child over pipes and calls its run tool. The model is replayed;
the QUMO tool calls are real (build/qumo mcp). Needs build/qumo-agent (make agents-mojo)."""

from std.os.path import exists
from std.testing import assert_equal, assert_true, assert_false

from qumo.ai.json import parse_json
from qumo_agent.mcp_client import McpClient
from qumo_agent.proc import ChildProcess

comptime BIN = "build/qumo-agent"
comptime REPLAY = "agents/mojo/tests/fixtures/replay_bell.jsonl"
comptime ANSWER = "Bell state: only 00 and 11 were measured, as theory predicts."


def main() raises:
    assert_true(exists(BIN), "build/qumo-agent is missing: run `make agents-mojo`")

    # 1. The agent as a library: an MCP client talks to `serve`.
    var c = McpClient(BIN, ["serve", "qumo-operator", "--replay", REPLAY])
    c.initialize("test")
    assert_equal(c.server_name, "qumo-agent-qumo-operator")
    var tools = c.list_tools()
    assert_equal(len(tools), 1)
    assert_equal(tools[0].name, "run")
    assert_true(parse_json(tools[0].input_schema).get("required").at(0).as_string() == "task")
    var r = c.call_tool("run", "{\"task\":\"Run a Bell state with seed 7\"}")
    assert_false(r.is_error, r.text)
    assert_equal(r.text, ANSWER)
    var s = parse_json(r.structured)
    assert_equal(s.get_string("agent", ""), "qumo-operator")
    assert_equal(s.get("stats").get_int("mcp_calls", -1), 2)
    assert_equal(s.get("stats").get_int("turns", -1), 3)
    # The replay is exhausted now: a second run is an isError result, not a dead server.
    var again = c.call_tool("run", "{\"task\":\"again\"}")
    assert_true(again.is_error)
    assert_true(again.text.find("replay file exhausted") >= 0, again.text)
    var bad = False
    try:
        _ = c.call_tool("nope", "{}")
    except e:
        bad = String(e).find("-32602") >= 0
    assert_true(bad)
    assert_equal(c.close(), 0)

    # 2. The launcher: `call` spawns `serve` itself and prints the answer.
    var p = ChildProcess(BIN, ["call", "qumo-operator", "Run a Bell state with seed 7", "--replay", REPLAY])
    var line = String()
    assert_true(p.read_line(line))
    assert_equal(line, ANSWER)
    assert_equal(p.finish(30.0), 0)

    # 3. Python-only agents are refused with a clear message and exit code 1.
    var q = ChildProcess(BIN, ["call", "research-watch", "x"])
    assert_false(q.read_line(line))
    assert_equal(q.finish(30.0), 1)
    print("test_serve_call OK")
