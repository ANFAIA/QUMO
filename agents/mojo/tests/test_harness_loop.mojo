"""Tests harness/loop.mojo (DESIGN.md 3.2): malformed calls (D2e), tool-node errors, the
mutation guard (D2d), `execute`, recursion limits including the after-agent step (R3),
and the answer / final-text rules. The wire parity scenarios S1..S6 are in
test_harness_wire.mojo."""

from std.os import makedirs
from std.testing import assert_equal, assert_true, assert_false

from qumo.ai.json import JsonValue, JsonWriter, parse_json

from qumo_agent.core.messages import Message, ROLE_ASSISTANT, ROLE_TOOL, ROLE_USER
from qumo_agent.harness.env import FileRoutes, AgentDef
from qumo_agent.harness.loop import Agent
from qumo_agent.harness.subagents import subagent_final_text
from qumo_agent.harness.testing import FakeEnv
from qumo_agent.harness.vfs import Vfs

comptime TMP = "build/test-fixtures/harness"


def call(id: String, name: String, raw_args: String) -> String:
    var w = JsonWriter()
    w.begin_object()
    w.kv_string("id", id)
    w.kv_string("type", "function")
    w.key("function")
    w.begin_object()
    w.kv_string("name", name)
    w.kv_string("arguments", raw_args)
    w.end_object()
    w.end_object()
    return w.finish()


def resp(calls: List[String], content: String) -> String:
    var w = JsonWriter()
    w.begin_object()
    w.key("choices")
    w.begin_array()
    w.begin_object()
    w.key("message")
    w.begin_object()
    w.kv_string("role", "assistant")
    w.kv_string("content", content)
    if len(calls) > 0:
        w.key("tool_calls")
        w.begin_array()
        for c in calls:
            w.raw(c)
        w.end_array()
    w.end_object()
    w.end_object()
    w.end_array()
    w.end_object()
    return w.finish()


def one(c: String) -> List[String]:
    var l = List[String]()
    l.append(c)
    return l^


def text(t: String) -> String:
    return resp(List[String](), t)


@fieldwise_init
struct Run(Movable):
    var requests: List[String]
    var answer: String
    var final_text: String
    var messages: List[Message]
    var error: String


def run(name: String, lines: List[String], limit: Int, general_purpose: Bool, routes: FileRoutes) raises -> Run:
    makedirs(TMP, exist_ok=True)
    var path = TMP + "/loop_" + name + ".replay.jsonl"
    with open(path, "w") as f:
        f.write(String("\n").join(lines) + "\n")
    var agents = "[{\"name\":\"a\",\"prompt\":\"P\",\"model\":\"m\",\"subagents\":[],\"recursion_limit\":" + String(limit) + "}]"
    var env = FakeEnv(path, agents, routes)
    var shared = env.shared
    var spec = env.agent_def("a", List[String]())
    var agent = Agent[FakeEnv](env^, spec, List[String](), "", general_purpose)
    try:
        var r = agent.run("TASK", Vfs())
        return Run(shared[].replay.requests.copy(), r.answer, r.final_text, r.messages.copy(), "")
    except e:
        return Run(shared[].replay.requests.copy(), "", "", List[Message](), String(e))


def test_malformed_calls() raises:
    var v0 = FileRoutes(False, False)
    # A turn with only malformed calls continues (D2e); then a mixed turn.
    var mixed = List[String]()
    mixed.append(call("v1", "echo", "{\"text\": \"ok\"}"))
    mixed.append(call("i2", "echo", "{\"text\": "))
    mixed.append(call("v3", "echo", "{\"text\": \"two\"}"))
    var lines: List[String] = [
        resp(one(call("i1", "echo", "{not json")), ""),
        resp(mixed, "thinking"),
        text("done"),
    ]
    var r = run("malformed", lines, 200, False, v0)
    assert_equal(r.error, "")
    assert_equal(r.answer, "done")
    var m = r.messages.copy()
    # user, ai(i1), tool(i1), ai(mixed), tool(i2), tool(v1), tool(v3), ai(done)
    assert_equal(len(m), 8)
    assert_equal(m[2].content, "Tool call echo with id i1 could not be executed - arguments were malformed or truncated.")
    assert_true(m[2].is_error)
    assert_equal(m[4].tool_call_id, "i2")
    assert_equal(m[5].tool_call_id, "v1")
    assert_equal(m[5].content, "ok")
    assert_equal(m[6].tool_call_id, "v3")
    # On the wire the invalid call keeps its raw arguments, after the valid ones.
    var body = parse_json(r.requests[2])
    var ai = body.get("messages").at(4)
    assert_equal(ai.get("tool_calls").at(2).get("id").as_string(), "i2")
    assert_equal(ai.get("tool_calls").at(2).get("function").get("arguments").as_string(), "{\"text\": ")


def test_tool_node_errors() raises:
    var v0 = FileRoutes(False, False)
    var calls = List[String]()
    calls.append(call("c1", "nope", "{}"))
    calls.append(call("c2", "execute", "{\"command\": \"ls\"}"))
    calls.append(call("c3", "echo", "{\"bad\": 1}"))
    calls.append(call("c4", "fail", "{}"))
    calls.append(call("c5", "write_file", "{\"file_path\": \"/x.md\", \"content\": \"1\"}"))
    calls.append(call("c6", "edit_file", "{\"path\": \"/x.md\", \"old\": \"1\", \"new\": \"2\"}"))
    calls.append(call("c7", "write_file", "{\"path\": \"/y.md\", \"content\": \"1\"}"))
    calls.append(call("c8", "delete", "{\"file_path\": \"/y.md\"}"))
    calls.append(call("c9", "crash", "{}"))
    var lines: List[String] = [resp(calls, ""), text("done")]
    var r = run("node_errors", lines, 200, False, v0)
    # `crash` is not among the agent's tools, so it is an unknown name.
    assert_equal(r.error, "")
    var m = r.messages.copy()
    assert_equal(m[2].content, "Error: nope is not a valid tool, try one of [ls, read_file, write_file, edit_file, delete, glob, grep, execute, task, echo, big, parts, fail].")
    assert_true(m[3].content.startswith("Error: Execution not available."))
    assert_equal(m[4].content, "Error invoking tool 'echo' with kwargs {'bad': 1} with error:\n text: Field required\n Please fix the error and try again.")
    assert_equal(m[5].content, "boom")
    assert_true(m[5].is_error)
    assert_equal(m[6].content, "Updated file /x.md")
    assert_equal(m[7].content, "Error: parallel file mutations to the same path are not allowed.")
    assert_equal(m[8].content, "Updated file /y.md")
    assert_equal(m[9].content, "Error: parallel file mutations to the same path are not allowed.")


def test_crash_propagates() raises:
    var lines: List[String] = [resp(one(call("c1", "crash", "{}")), ""), text("done")]
    makedirs(TMP, exist_ok=True)
    var path = TMP + "/loop_crash.replay.jsonl"
    with open(path, "w") as f:
        f.write(String("\n").join(lines) + "\n")
    var env = FakeEnv(path, "[{\"name\":\"a\",\"prompt\":\"P\",\"model\":\"m\",\"subagents\":[],\"tools\":[\"crash\"]}]", FileRoutes(False, False))
    var spec = env.agent_def("a", List[String]())
    var agent = Agent[FakeEnv](env^, spec, List[String](), "", False)
    var err = String()
    try:
        _ = agent.run("TASK", Vfs())
    except e:
        err = String(e)
    assert_equal(err, "QA-E130: crash")


def _looping(n_calls: Int, final_on: Int) -> List[String]:
    var lines = List[String]()
    for i in range(1, n_calls + 1):
        if i == final_on:
            lines.append(text("final " + String(i)))
        else:
            lines.append(resp(one(call("c" + String(i), "echo", "{\"text\": \"x\"}")), ""))
    return lines^


def test_recursion() raises:
    var v0 = FileRoutes(False, False)
    var r = run("rec25", _looping(30, 0), 25, False, v0)
    assert_equal(len(r.requests), 12)
    assert_equal(r.error, "QA-E110: Recursion limit of 25 reached without hitting a stop condition. You can increase the limit by setting the `recursion_limit` config key.. Hint: raise --recursion-limit")
    r = run("rec200", _looping(120, 0), 200, False, v0)
    assert_equal(len(r.requests), 100)
    assert_true(r.error.startswith("QA-E110: Recursion limit of 200 reached"))
    # R3: limit 24 with the answer on call 12 fails; on call 11 it completes.
    r = run("rec24_12", _looping(12, 12), 24, False, v0)
    assert_equal(len(r.requests), 12)
    assert_true(r.error.startswith("QA-E110: Recursion limit of 24 reached"))
    r = run("rec24_11", _looping(11, 11), 24, False, v0)
    assert_equal(r.error, "")
    assert_equal(r.answer, "final 11")
    # The general-purpose subagent has no after-agent step.
    r = run("rec24_gp", _looping(12, 12), 24, True, v0)
    assert_equal(r.error, "")
    assert_equal(r.answer, "final 12")


def test_answer_rules() raises:
    var v0 = FileRoutes(False, False)
    var lines: List[String] = [text("  answer text  \n")]
    var r = run("answer", lines, 200, False, v0)
    assert_equal(r.answer, "  answer text  \n")
    assert_equal(r.final_text, "  answer text")
    var msgs = List[Message]()
    msgs.append(Message(ROLE_USER, "q"))
    msgs.append(Message(ROLE_ASSISTANT, "real"))
    msgs.append(Message(ROLE_ASSISTANT, "  \n"))
    assert_equal(subagent_final_text(msgs), "real")
    var none = List[Message]()
    none.append(Message(ROLE_USER, "q"))
    assert_equal(subagent_final_text(none), "")


def test_tool_lists() raises:
    makedirs(TMP, exist_ok=True)
    var path = TMP + "/loop_tools.replay.jsonl"
    with open(path, "w") as f:
        f.write(text("x") + "\n")
    var agents = "[{\"name\":\"o\",\"prompt\":\"P\",\"model\":\"m\",\"subagents\":[\"b\",\"c\"]},{\"name\":\"b\",\"description\":\"B agent.\",\"prompt\":\"PB\",\"model\":\"mb\",\"subagents\":[\"o\",\"c\"]},{\"name\":\"c\",\"description\":\"C agent.\",\"prompt\":\"PC\",\"model\":\"mc\",\"subagents\":[]}]"
    var env = FakeEnv(path, agents, FileRoutes(True, False))
    var spec = env.agent_def("o", List[String]())
    var agent = Agent[FakeEnv](env^, spec, List[String](), "", False)
    var names = agent.tool_node_names()
    assert_equal(String(",").join(names), "ls,read_file,write_file,edit_file,delete,glob,grep,execute,task,echo,big,parts,fail")
    var tools = agent.wire_tools()
    assert_true("- general-purpose: General-purpose agent" in tools[7].description)
    assert_true("\n- b: B agent.\n- c: C agent.\n\nSpecify subagent_type" in tools[7].description)
    # A general-purpose agent has no task tool.
    var env2 = FakeEnv(path, agents, FileRoutes(True, False))
    var gp = Agent[FakeEnv](env2^, spec, List[String](), "", True)
    assert_equal(String(",").join(gp.tool_node_names()), "ls,read_file,write_file,edit_file,delete,glob,grep,execute,echo,big,parts,fail")


def main() raises:
    test_malformed_calls()
    test_tool_node_errors()
    test_crash_propagates()
    test_recursion()
    test_answer_rules()
    test_tool_lists()
    print("test_harness_loop OK")
