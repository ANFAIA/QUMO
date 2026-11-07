"""Tests harness/tokens.mojo against count_tokens_approximately (tests/golden/token_counts.json)."""

from std.testing import assert_equal

from qumo.ai.json import JsonValue, parse_json

from qumo_agent.core.messages import (
    Message, ToolCall, ROLE_SYSTEM, ROLE_USER, ROLE_ASSISTANT, ROLE_TOOL,
)
from qumo_agent.core.toolspec import ToolSpec, TOOL_HARNESS
from qumo_agent.harness.tokens import count_tokens_list, tool_calls_repr

comptime GOLDEN = "agents/mojo/tests/golden/token_counts.json"


def _read(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def message_from_golden(d: JsonValue) raises -> Message:
    """A message as gen_golden.py's msg_json wrote it."""
    var ty = d.get("type").as_string()
    var role = ROLE_USER
    if ty == "system":
        role = ROLE_SYSTEM
    elif ty == "ai":
        role = ROLE_ASSISTANT
    elif ty == "tool":
        role = ROLE_TOOL
    var m = Message(role, d.get_string("content", ""))
    if d.has("parts"):
        var p = d.get("parts")
        for i in range(len(p)):
            m.parts.append(p.at(i).as_string())
        m.has_parts = True
    m.name = d.get_string("name", "")
    if d.has("tool_calls"):
        var tc = d.get("tool_calls")
        for i in range(len(tc)):
            var c = tc.at(i)
            m.tool_calls.append(ToolCall(c.get("id").as_string(), c.get("name").as_string(), c.get("args")))
    m.tool_call_id = d.get_string("tool_call_id", "")
    m.usage_total_tokens = d.get_int("total_tokens", -1)
    return m^


def main() raises:
    var g = parse_json(_read(GOLDEN))
    var cj = g.get("count_json")
    var file_tools = List[ToolSpec]()
    var fake_tools = List[ToolSpec]()
    for i in range(len(cj)):
        var name = cj.key_at(i)
        var t = ToolSpec(name, "", "{}", TOOL_HARNESS, False, cj.at(i).to_json())
        if name == "echo" or name == "big" or name == "parts" or name == "fail":
            fake_tools.append(t^)
        else:
            file_tools.append(t^)
    var cases = g.get("cases")
    for i in range(len(cases)):
        var c = cases.at(i)
        var msgs = List[Message]()
        var ms = c.get("messages")
        for j in range(len(ms)):
            msgs.append(message_from_golden(ms.at(j)))
        var ts = c.get("tools").as_string()
        var tools = List[ToolSpec]()
        if ts == "fake":
            tools = fake_tools.copy()
        elif ts == "files":
            tools = file_tools.copy()
        var label = "case " + String(i) + " tools=" + ts
        assert_equal(count_tokens_list(msgs, tools, True), c.get("scaled").as_int(), label + " scaled")
        assert_equal(count_tokens_list(msgs, tools, False), c.get("unscaled").as_int(), label + " unscaled")
    var calls = List[ToolCall]()
    calls.append(ToolCall("call_1", "ls", parse_json("{\"path\": \"/é\", \"n\": 1.5, \"ok\": true, \"z\": null}")))
    assert_equal(tool_calls_repr(calls), "[{'name': 'ls', 'args': {'path': '/é', 'n': 1.5, 'ok': True, 'z': None}, 'id': 'call_1', 'type': 'tool_call'}]")
    print("test_harness_tokens OK (" + String(len(cases)) + " cases)")
