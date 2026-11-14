"""Tests for core/replay.mojo (routing by delegation path, error lines, exhaustion,
request log directory) and core/stats.mojo, core/endpoint.mojo."""

from std.testing import assert_equal, assert_true, assert_false
from std.os import setenv, unsetenv, listdir, remove, rmdir
from std.os.path import exists
from std.tempfile import mkdtemp

from qumo.ai.json import parse_json
from qumo_agent.core.messages import ChatRequest, Message, ToolSpec, user_message
from qumo_agent.core.replay import ReplayModel
from qumo_agent.core.stats import RunStats, zero_stats
from qumo_agent.core.endpoint import ModelEndpoint

comptime FIXTURE = "agents/mojo/tests/fixtures/core/replay_paths.jsonl"


def _req(at: String, text: String) -> ChatRequest:
    var msgs = List[Message]()
    msgs.append(user_message(text))
    return ChatRequest("m", "SYS", msgs^, List[ToolSpec](), "agent", at)


def _error_of(mut model: ReplayModel, at: String) -> String:
    try:
        _ = model.complete(_req(at, "x"))
    except e:
        return String(e)
    return String()


def test_replay() raises:
    var log_dir = mkdtemp()
    _ = setenv("QUMO_AGENT_REPLAY_LOG", log_dir)
    var model = ReplayModel(FIXTURE)
    assert_equal(model.remaining(""), 2)
    assert_equal(model.remaining("call_a"), 2)
    assert_equal(model.remaining("call_a/call_b"), 2)
    var r1 = model.complete(_req("", "root task"))
    assert_equal(r1.message.tool_calls[0].id, "call_a")
    assert_equal(r1.total_tokens, 12)
    # The child path reads only its own lines, in order.
    var c1 = model.complete(_req("call_a", "child task"))
    assert_equal(c1.message.tool_calls[0].id, "call_b")
    # An error line is raised as is.
    assert_equal(_error_of(model, "call_a/call_b"), "QA-E102: context overflow: HTTP 400: prompt is too long")
    var g = model.complete(_req("call_a/call_b", "retry"))
    assert_equal(g.message.content, "grandchild done")
    assert_equal(model.complete(_req("call_a", "y")).message.content, "child done")
    assert_equal(model.complete(_req("", "z")).message.content, "root done")
    assert_equal(model.remaining(""), 0)
    # Exhaustion.
    assert_equal(_error_of(model, ""), "QA-E170: replay file exhausted at '' after 2 responses")
    assert_equal(_error_of(model, "call_a"), "QA-E170: replay file exhausted at 'call_a' after 2 responses")
    assert_equal(_error_of(model, "nobody"), "QA-E170: replay file exhausted at 'nobody' after 0 responses")
    # Every request is kept with its path, including the failed ones.
    assert_equal(len(model.requests), 9)
    assert_equal(model.request_paths[1], "call_a")
    assert_equal(parse_json(model.requests[0]).get("messages").at(1).get("content").as_string(), "root task")
    # The request log directory has one file per path.
    _ = unsetenv("QUMO_AGENT_REPLAY_LOG")
    var root_log = log_dir + "/root.jsonl"
    var child_log = log_dir + "/call_a.jsonl"
    var grand_log = log_dir + "/call_a__call_b.jsonl"
    assert_true(exists(root_log) and exists(child_log) and exists(grand_log))
    var lines = open(grand_log, "r").read().split("\n")
    assert_equal(len(lines), 3)   # two requests and the empty string after the last newline
    assert_equal(parse_json(String(lines[1])).get("messages").at(1).get("content").as_string(), "retry")
    for name in listdir(log_dir):
        remove(log_dir + "/" + name)
    rmdir(log_dir)
    # No log when the variable is unset.
    var m2 = ReplayModel(FIXTURE)
    _ = m2.complete(_req("", "q"))
    assert_false(exists(log_dir))


def test_replay_errors() raises:
    var raised = String()
    try:
        _ = ReplayModel("agents/mojo/tests/fixtures/core/does-not-exist.jsonl")
    except e:
        raised = String(e)
    assert_true(raised.startswith("QA-E170: cannot read replay file"), raised)
    var d = mkdtemp()
    var bad = d + "/bad.jsonl"
    with open(bad, "w") as f:
        f.write("// ok\n{not json}\n")
    raised = String()
    try:
        _ = ReplayModel(bad)
    except e:
        raised = String(e)
    assert_true(raised.startswith("QA-E170: replay file '" + bad + "' line 2 is not JSON"), raised)
    remove(bad)
    rmdir(d)


def test_stats_endpoint() raises:
    var a = zero_stats()
    assert_equal(a.to_json(), '{"model_calls":0,"tool_calls":0,"tool_errors":0,"mcp_calls":0,"http_requests":0,"http_retries":0,"prompt_tokens":0,"completion_tokens":0,"evictions":0,"summarizations":0,"subagent_runs":0,"wall_ms":0}')
    var b = RunStats(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 500)
    a.add(b)
    a.add(b)
    assert_equal(a.model_calls, 2)
    assert_equal(a.subagent_runs, 22)
    assert_equal(a.wall_ms, 500)
    var c = zero_stats()
    c.from_json(parse_json(a.to_json()))
    assert_equal(c.to_json(), a.to_json())
    var partial = zero_stats()
    partial.wall_ms = 7
    partial.from_json(parse_json('{"tool_calls": 4}'))
    assert_equal(partial.tool_calls, 4)
    assert_equal(partial.wall_ms, 7)
    var raised = False
    try:
        partial.from_json(parse_json("[1]"))
    except:
        raised = True
    assert_true(raised)
    assert_equal(String(b), "model_calls=1 tool_calls=2 tool_errors=3 mcp_calls=4 http_requests=5 http_retries=6 prompt_tokens=7 completion_tokens=8 evictions=9 summarizations=10 subagent_runs=11 wall_ms=500")
    var names: List[String] = ["X-Title"]
    var values: List[String] = ["QUMO agents"]
    var ep = ModelEndpoint("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", names^, values^, 300, 2)
    var ep2 = ep.copy()
    assert_equal(ep2.header_values[0], "QUMO agents")
    assert_equal(ep2.timeout_s, 300)


def main() raises:
    test_replay()
    test_replay_errors()
    test_stats_endpoint()
    print("test_core_replay OK")
