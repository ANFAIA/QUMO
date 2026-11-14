"""Tests harness/summarize.mojo and the summarization steps of the loop (DESIGN.md 3.3):
goldens from LangChain (tests/golden/summaries.json), cutoff arithmetic over two
summarizations, the context-overflow path (tail clip, exhausted recovery) and the
summary request's retry rules."""

from std.os import makedirs
from std.testing import assert_equal, assert_true, assert_false

from qumo.ai.json import JsonValue, JsonWriter, parse_json

from qumo_agent.core.messages import Message, ToolCall, ROLE_ASSISTANT, ROLE_TOOL, ROLE_USER, ROLE_SYSTEM, user_message
from qumo_agent.core.toolspec import ToolSpec
from qumo_agent.harness.tokens import count_tokens_list
from qumo_agent.util.pyfmt import cp_len, cp_head, cp_slice
from qumo_agent.harness.env import FileRoutes
from qumo_agent.harness.loop import Agent
from qumo_agent.harness.summarize import (
    xml_transcript, summary_prompt_text, safe_cutoff, overflow_retry_prompt, truncate_tool_args,
    apply_event, state_cutoff, summary_message, history_section, SummaryEvent,
)
from qumo_agent.harness.testing import FakeEnv
from qumo_agent.harness.vfs import Vfs

comptime GOLDEN = "agents/mojo/tests/golden/summaries.json"
comptime TMP = "build/test-fixtures/harness"
comptime AGENTS = "[{\"name\":\"a\",\"prompt\":\"P\",\"model\":\"m\",\"subagents\":[]}]"


def _read(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def message_from_golden(d: JsonValue) raises -> Message:
    """A message as gen_golden.py's msg_json wrote it (same as test_harness_tokens)."""
    var ty = d.get("type").as_string()
    var role = ROLE_USER
    if ty == "ai":
        role = ROLE_ASSISTANT
    elif ty == "tool":
        role = ROLE_TOOL
    elif ty == "system":
        role = ROLE_SYSTEM
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


def _msgs(v: JsonValue) raises -> List[Message]:
    var out = List[Message]()
    for i in range(len(v)):
        out.append(message_from_golden(v.at(i)))
    return out^


def test_golden() raises:
    var g = parse_json(_read(GOLDEN))
    var xml = g.get("xml")
    for i in range(len(xml)):
        assert_equal(xml_transcript(_msgs(xml.at(i).get("messages"))), xml.at(i).get("xml").as_string(), "xml " + String(i))
    assert_equal(summary_prompt_text(xml.at(0).get("xml").as_string()), g.get("summary_prompt_first").as_string())
    var cuts = g.get("cutoffs")
    for i in range(len(cuts)):
        assert_equal(safe_cutoff(_msgs(cuts.at(i).get("messages")), 6), cuts.at(i).get("cutoff").as_int(), "cutoff " + String(i))
    var ov = g.get("overflow_prompts")
    for i in range(len(ov)):
        var t = ov.at(i).get("transcript").as_string()
        var got = _overflow_for_transcript(t)
        if ov.at(i).get("prompt").is_null():
            assert_false(Bool(got), "overflow " + String(i))
        else:
            assert_equal(got.value(), summary_prompt_text(ov.at(i).get("prompt").as_string()), "overflow " + String(i))
    var ta = g.get("truncate_args")
    var all = _msgs(ta.get("input"))
    var results = ta.get("results")
    for i in range(len(results)):
        var r = results.at(i)
        var n = r.get("n").as_int()
        var sub = List[Message]()
        for j in range(n):
            sub.append(all[j].copy())
        var out = truncate_tool_args(sub)
        assert_equal(out[1], r.get("changed").as_bool(), "truncate " + String(n))
        var exp = _msgs(r.get("messages"))
        for j in range(n):
            assert_equal(xml_transcript(_one(out[0][j])), xml_transcript(_one(exp[j])), "truncate " + String(n) + " msg " + String(j))


def _one(m: Message) -> List[Message]:
    var l = List[Message]()
    l.append(m.copy())
    return l^


def _overflow_for_transcript(t: String) raises -> Optional[String]:
    """overflow_retry_prompt for a list whose transcript is exactly `t`: a single system
    message renders as `<message type="system">t</message>`, so build the prompt from the
    transcript through the same arithmetic (the golden stores transcripts directly)."""
    var one = List[Message]()
    one.append(user_message(t))
    var tokens = count_tokens_list(one, List[ToolSpec](), False)
    var budget = tokens // 2
    var keep = cp_len(t) * budget // (tokens if tokens > 1 else 1)
    if keep <= 0:
        return None
    var head = keep // 4
    var n = cp_len(t)
    return summary_prompt_text(cp_head(t, head) + "\n\n[... middle of conversation omitted ...]\n\n" + cp_slice(t, n - (keep - head), n))


def test_pure() raises:
    var msgs = List[Message]()
    for i in range(5):
        msgs.append(Message(ROLE_USER, "m" + String(i)))
    var s = summary_message("S", "/conversation_history/x.md")
    assert_equal(s.lc_source, "summarization")
    assert_true(s.content.startswith("You are in the middle of a conversation that has been summarized."))
    assert_equal(summary_message("S", "").content, "Here is a summary of the conversation to date:\n\nS")
    var ev = Optional[SummaryEvent](SummaryEvent(3, s.copy(), "/x", 5))
    var eff = apply_event(msgs, ev)
    assert_equal(len(eff), 3)
    assert_equal(eff[1].content, "m3")
    var ev2 = Optional[SummaryEvent](SummaryEvent(9, s.copy(), "/x", 5))
    assert_equal(len(apply_event(msgs, ev2)), 1)
    assert_equal(state_cutoff(None, 4), 4)
    assert_equal(state_cutoff(ev, 4), 6)
    var with_summary = List[Message]()
    with_summary.append(s.copy())
    with_summary.append(Message(ROLE_USER, "x"))
    assert_equal(history_section("T", with_summary), "## Summarized at T\n\n<message type=\"human\">x</message>\n\n")


# ---------------------------------------------------------------------------
# loop scenarios
# ---------------------------------------------------------------------------
def tool_line(id: String, name: String, args_json: String, total: Int) -> String:
    var w = JsonWriter()
    w.begin_object()
    w.key("choices")
    w.begin_array()
    w.begin_object()
    w.key("message")
    w.begin_object()
    w.kv_string("role", "assistant")
    w.kv_null("content")
    w.key("tool_calls")
    w.begin_array()
    w.begin_object()
    w.kv_string("id", id)
    w.kv_string("type", "function")
    w.key("function")
    w.begin_object()
    w.kv_string("name", name)
    w.kv_string("arguments", args_json)
    w.end_object()
    w.end_object()
    w.end_array()
    w.end_object()
    w.end_object()
    w.end_array()
    w.key("usage")
    w.begin_object()
    w.kv_int("prompt_tokens", 1)
    w.kv_int("completion_tokens", 1)
    w.kv_int("total_tokens", total)
    w.end_object()
    w.end_object()
    return w.finish()


def text_line(text: String) -> String:
    var w = JsonWriter()
    w.begin_object()
    w.key("choices")
    w.begin_array()
    w.begin_object()
    w.key("message")
    w.begin_object()
    w.kv_string("role", "assistant")
    w.kv_string("content", text)
    w.end_object()
    w.end_object()
    w.end_array()
    w.end_object()
    return w.finish()


def error_line(text: String) -> String:
    var w = JsonWriter()
    w.begin_object()
    w.kv_string("at", "")
    w.kv_string("error", text)
    w.end_object()
    return w.finish()


def run_lines(name: String, lines: List[String]) raises -> Tuple[List[String], String, List[Message], Vfs]:
    """Runs agent `a` over a replay made of `lines`; returns (request bodies, answer or
    error text, messages, files)."""
    makedirs(TMP, exist_ok=True)
    var path = TMP + "/" + name + ".replay.jsonl"
    with open(path, "w") as f:
        f.write(String("\n").join(lines) + "\n")
    var env = FakeEnv(path, AGENTS, FileRoutes(False, False))
    var shared = env.shared
    var spec = env.agent_def("a", List[String]())
    var agent = Agent[FakeEnv](env^, spec, List[String](), "", False)
    try:
        var res = agent.run("TASK", Vfs())
        return (shared[].replay.requests.copy(), res.answer, res.messages.copy(), res.files.copy())
    except e:
        return (shared[].replay.requests.copy(), "ERROR " + String(e), List[Message](), Vfs())


def test_two_summarizations() raises:
    var lines: List[String] = [
        tool_line("c1", "echo", "{\"text\":\"1\"}", 100),
        tool_line("c2", "echo", "{\"text\":\"2\"}", 100),
        tool_line("c3", "echo", "{\"text\":\"3\"}", 100),
        tool_line("c4", "echo", "{\"text\":\"4\"}", 170000),
        text_line("SUMMARY ONE"),
        tool_line("c5", "echo", "{\"text\":\"5\"}", 170000),
        text_line("SUMMARY TWO"),
        text_line("final"),
    ]
    var r = run_lines("two_summaries", lines)
    assert_equal(r[1], "final")
    var reqs = r[0].copy()
    assert_equal(len(reqs), 8)
    # The first summary request: one user message, no system message and no tools.
    var s1 = parse_json(reqs[4])
    assert_false(s1.has("tools"))
    assert_equal(len(s1.get("messages")), 1)
    assert_true("<message type=\"human\">TASK</message>" in s1.get("messages").at(0).get("content").as_string())
    # After it: system, summary, a2, t2, a3, t3, a4, t4.
    var m5 = parse_json(reqs[5]).get("messages")
    assert_equal(len(m5), 8)
    var summary1 = m5.at(1).get("content").as_string()
    assert_true(summary1.startswith("You are in the middle of a conversation that has been summarized.\n\nThe full conversation history has been saved to /conversation_history/session_"))
    assert_true(summary1.endswith("<summary>\nSUMMARY ONE\n</summary>"))
    assert_equal(m5.at(2).get("tool_calls").at(0).get("id").as_string(), "c2")
    # The second summary transcript starts with the first summary (a human message).
    var s2 = parse_json(reqs[6]).get("messages").at(0).get("content").as_string()
    assert_true("<message type=\"human\">You are in the middle of a conversation" in s2)
    # Final request: system, summary two, then state messages from index 5 (a3) on.
    var m7 = parse_json(reqs[7]).get("messages")
    assert_equal(len(m7), 8)
    assert_true(m7.at(1).get("content").as_string().endswith("<summary>\nSUMMARY TWO\n</summary>"))
    assert_equal(m7.at(2).get("tool_calls").at(0).get("id").as_string(), "c3")
    # The state keeps every message; the history file has two sections, no summary inside.
    assert_equal(len(r[2]), 12)
    var paths = r[3].paths()
    assert_equal(len(paths), 1)
    var hist = r[3].get(paths[0]).value().content
    assert_equal(hist.count("## Summarized at 2026-01-01T00:"), 2)
    assert_false("SUMMARY ONE" in hist)


def test_overflow_clip() raises:
    var big = String()
    for i in range(3000):
        big += "row " + String(i) + "\n"
    var w = JsonWriter()
    w.begin_object()
    w.kv_string("file_path", "/f.txt")
    w.kv_string("content", big)
    w.end_object()
    var write_args = w.finish()
    var parts = String("{\"items\":[\"") + big.replace("\n", "\\n") + "\",\"tail\"]}"
    var lines: List[String] = [
        tool_line("w1", "write_file", write_args, 10),
        tool_line("r1", "read_file", "{\"file_path\":\"/f.txt\",\"limit\":3000}", 10),
        tool_line("p1", "parts", parts, 10),
        error_line("QA-E102: context overflow: HTTP 400: prompt is too long"),
        text_line("THE SUMMARY"),    # 7 messages: the overflow summarizes (cutoff 1) first
        text_line("recovered"),
    ]
    # read_file and parts in separate turns: the tail run is only the parts result.
    var r = run_lines("overflow_clip", lines)
    assert_equal(r[1], "recovered")
    var last = parse_json(r[0][len(r[0]) - 1]).get("messages")
    var tail = last.at(len(last) - 1)
    assert_true(tail.get("content").is_string())   # the parts result became a plain string
    assert_true(tail.get("content").as_string().startswith("Tool result too large, the result of this tool call p1 was saved in the filesystem at this path: /large_tool_results/p1"))
    var msgs = r[2].copy()
    var found = False
    for m in msgs:
        if m.role == ROLE_TOOL and m.tool_call_id == "p1":
            found = True
            assert_false(m.has_parts)
            assert_true(m.content.startswith("Tool result too large"))
    assert_true(found)
    assert_true(Bool(r[3].get("/large_tool_results/p1")))
    # A read_file tail keeps 4,000 code points plus the notice.
    var lines2: List[String] = [
        tool_line("w1", "write_file", write_args, 10),
        tool_line("r1", "read_file", "{\"file_path\":\"/f.txt\",\"limit\":3000}", 10),
        error_line("QA-E102: context overflow: HTTP 413: maximum context length"),
        text_line("ok"),
    ]
    var r2 = run_lines("overflow_read", lines2)
    assert_equal(r2[1], "ok")
    var last2 = parse_json(r2[0][len(r2[0]) - 1]).get("messages")
    var rt = last2.at(len(last2) - 1).get("content").as_string()
    assert_true(rt.endswith("call read_file with file_path='/f.txt', offset=0, limit=100.]"))
    # A tiny tail cannot shrink: recovery is exhausted.
    var lines3: List[String] = [
        tool_line("e1", "echo", "{\"text\":\"x\"}", 10),
        error_line("QA-E102: context overflow: HTTP 400: prompt is too long"),
    ]
    var r3 = run_lines("overflow_exhausted", lines3)
    assert_equal(r3[1], "ERROR QA-E102: Model input still does not fit after recovery; reduce input, tools, or configured output tokens.")


def _summary_lines(failure: String) -> List[String]:
    var lines: List[String] = [
        tool_line("c1", "echo", "{\"text\":\"1\"}", 100),
        tool_line("c2", "echo", "{\"text\":\"2\"}", 100),
        tool_line("c3", "echo", "{\"text\":\"3\"}", 170000),
        error_line(failure),
        text_line("THE SUMMARY"),
        text_line("final"),
    ]
    return lines^


def test_summary_retries() raises:
    # Retried: 500, 429, a B5 response error, a network error.
    var retried: List[String] = [
        "QA-E101: model request failed: HTTP 500 Internal Server Error: boom",
        "QA-E101: model request failed: HTTP 429 Too Many Requests: slow down",
        "QA-E103: response has no choices",
    ]
    for f in retried:
        var r = run_lines("retry", _summary_lines(f))
        assert_equal(r[1], "final", f)
        assert_equal(len(r[0]), 6, f)    # 3 + failed summary + summary + final
    # Not retried: 401 and 400.
    var fatal: List[String] = [
        "QA-E101: model request failed: HTTP 401 Unauthorized: bad key",
        "QA-E101: model request failed: HTTP 400 Bad Request: nope",
    ]
    for f in fatal:
        var r = run_lines("fatal", _summary_lines(f))
        assert_equal(r[1], "ERROR " + f)
        assert_equal(len(r[0]), 4, f)


def main() raises:
    test_golden()
    test_pure()
    test_two_summarizations()
    test_overflow_clip()
    test_summary_retries()
    print("test_harness_summarize OK")
