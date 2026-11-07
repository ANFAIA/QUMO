"""Tests harness/eviction.mojo: previews and stubs (tests/golden/previews.json), the
80,000 code-point threshold, exempt tools, id sanitizing, parts results becoming plain
strings, and human-message eviction through the loop."""

from std.os import makedirs
from std.testing import assert_equal, assert_true, assert_false

from qumo.ai.json import parse_json

from qumo_agent.core.toolspec import ToolOutcome, text_outcome, parts_outcome
from qumo_agent.harness.env import FileRoutes
from qumo_agent.harness.eviction import (
    content_preview, preview_note, large_tool_result_text, human_message_stub,
    sanitize_tool_call_id, display_tool_call_id, evict_tool_outcome, is_eviction_exempt,
)
from qumo_agent.harness.loop import Agent
from qumo_agent.harness.testing import FakeEnv
from qumo_agent.harness.vfs import Vfs, VfsTx

comptime GOLDEN = "agents/mojo/tests/golden/previews.json"
comptime TMP = "build/test-fixtures/harness"


def _read(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def _repeat(s: String, n: Int) -> String:
    var out = String()
    for _ in range(n):
        out += s
    return out^


def test_golden() raises:
    var g = parse_json(_read(GOLDEN))
    var cases = g.get("cases")
    for i in range(len(cases)):
        var c = cases.at(i)
        var text = c.get("input").as_string()
        var p = content_preview(text)
        var label = "preview " + String(i)
        assert_equal(p.text, c.get("text").as_string(), label)
        assert_equal(p.lines_omitted, c.get("lines_omitted").as_bool(), label)
        assert_equal(p.lines_clipped, c.get("lines_clipped").as_bool(), label)
        assert_equal(preview_note(p, "result"), c.get("note_result").as_string(), label)
        assert_equal(preview_note(p, "content"), c.get("note_content").as_string(), label)
        assert_equal(large_tool_result_text("call_" + _repeat("q", 40), "/large_tool_results/x", text),
                     c.get("tool_stub").as_string(), label)
        assert_equal(human_message_stub("/conversation_history/u.md", text), c.get("human_stub").as_string(), label)
    var san = g.get("sanitize")
    for i in range(len(san)):
        assert_equal(sanitize_tool_call_id(san.at(i).at(0).as_string()), san.at(i).at(1).as_string())
    var disp = g.get("display")
    for i in range(len(disp)):
        assert_equal(display_tool_call_id(disp.at(i).at(0).as_string()), disp.at(i).at(1).as_string())


def test_threshold() raises:
    var base = Vfs()
    # 80,000 code points of a 3-byte character: kept (bytes would be 240,000).
    var tx = VfsTx()
    var r = evict_tool_outcome("echo", "c1", text_outcome(_repeat("€", 80000)), base, tx, "t", "")
    assert_false(r[1])
    assert_equal(len(tx.writes), 0)
    r = evict_tool_outcome("echo", "c1", text_outcome(_repeat("€", 80001)), base, tx, "t", "")
    assert_true(r[1])
    assert_true(r[0].text.startswith("Tool result too large, the result of this tool call c1 was saved in the filesystem at this path: /large_tool_results/c1\n"))
    assert_equal(tx.get(base, "/large_tool_results/c1").value().content, _repeat("€", 80001))
    # exempt tools are never evicted
    var tx2 = VfsTx()
    for name in ["ls", "glob", "grep", "read_file", "edit_file", "write_file", "delete"]:
        assert_true(is_eviction_exempt(name))
        assert_false(evict_tool_outcome(name, "c", text_outcome(_repeat("x", 90000)), base, tx2, "t", "")[1])
    assert_false(is_eviction_exempt("task"))
    # a parts result is measured joined with "\n" and becomes a plain string
    var parts: List[String] = [_repeat("a", 40000), _repeat("b", 40000)]
    var p = evict_tool_outcome("parts", "c.2", parts_outcome(parts^, True), base, tx2, "t", "")
    assert_true(p[1])
    assert_false(p[0].is_parts)
    assert_equal(len(p[0].parts), 0)
    assert_true(p[0].is_error)
    assert_true(Bool(tx2.get(base, "/large_tool_results/c_2")))
    assert_equal(tx2.get(base, "/large_tool_results/c_2").value().content.byte_length(), 80001)
    # no id: unknown-<fallback>
    var tx3 = VfsTx()
    _ = evict_tool_outcome("echo", "", text_outcome(_repeat("x", 80001)), base, tx3, "t", "0123abcd")
    assert_true(Bool(tx3.get(base, "/large_tool_results/unknown-0123abcd")))
    # long ids are hashed
    assert_equal(sanitize_tool_call_id(_repeat("x", 129)).byte_length(), 5 + 64)
    assert_true(sanitize_tool_call_id(_repeat("x", 129)).startswith("call-"))


def test_human_eviction() raises:
    makedirs(TMP, exist_ok=True)
    var replay = TMP + "/human.replay.jsonl"
    with open(replay, "w") as f:
        f.write("{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"ok\"}}]}\n")
        f.write("{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"ok\"}}]}\n")
    var agents = "[{\"name\":\"a\",\"prompt\":\"P\",\"model\":\"m\",\"subagents\":[],\"tools\":[]}]"
    # 200,000 code points: kept.
    var env = FakeEnv(replay, agents, FileRoutes(False, False))
    var shared = env.shared
    var spec = env.agent_def("a", List[String]())
    var agent = Agent[FakeEnv](env^, spec, List[String](), "", False)
    var res = agent.run(_repeat("é", 200000), Vfs())
    assert_equal(len(res.files.files), 0)
    assert_true(_repeat("é", 200000) in shared[].replay.requests[0])
    # 200,001: saved to /conversation_history/<uuid>.md, the request carries the stub.
    var env2 = FakeEnv(replay, agents, FileRoutes(False, False))
    var shared2 = env2.shared
    var agent2 = Agent[FakeEnv](env2^, spec, List[String](), "", False)
    var task = _repeat("é", 200001)
    var res2 = agent2.run(task, Vfs())
    var paths = res2.files.paths()
    assert_equal(len(paths), 1)
    assert_equal(paths[0], "/conversation_history/00000000-0000-0000-0000-000000000001.md")
    assert_equal(res2.files.get(paths[0]).value().content, task)
    var body = parse_json(shared2[].replay.requests[0])
    var user = body.get("messages").at(1).get("content").as_string()
    assert_equal(user, human_message_stub(paths[0], task))
    assert_equal(res2.messages[0].content, task)        # state keeps the full text
    assert_equal(res2.messages[0].evicted_to, paths[0])


def main() raises:
    test_golden()
    test_threshold()
    test_human_eviction()
    print("test_harness_eviction OK")
