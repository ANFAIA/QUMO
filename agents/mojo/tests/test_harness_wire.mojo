"""Tests the harness loop against deepagents' wire captures wire_S1..S6 (DESIGN.md 8.1).

Each scenario replays the same model responses Python got (wire_<S>.replay.jsonl) through
`Agent` + `FakeEnv` and compares every request body structurally (parsed JSON, ordered
keys and arrays) with Python's, after replacing uuids and timestamps by placeholders.

Known, intended differences that the comparison normalizes:
- read_file / write_file / edit_file are this port's merged tools (D2a): their
  description must start with deepagents' text followed by "\\n\\nQUMO notes:", and
  their parameters (Appendix B.1) are not compared.
- S3 runs its two subagents in parallel in Python, so the order of the two subagent
  requests is not fixed there; requests are compared after sorting by
  (system prompt, first user message, message count).
"""

from std.testing import assert_equal, assert_true

from qumo.ai.json import JsonValue, JsonWriter, parse_json

from qumo_agent.harness.env import FileRoutes, AgentDef
from qumo_agent.harness.loop import Agent
from qumo_agent.harness.testing import FakeEnv
from qumo_agent.harness.vfs import Vfs

comptime GOLDEN = "agents/mojo/tests/golden/"
comptime AGENTS = "[{\"name\":\"orchestrator\",\"description\":\"Orchestrates.\",\"prompt\":\"MAIN PROMPT\",\"model\":\"anthropic/claude-opus-5.5\",\"subagents\":[\"research-watch\"]},{\"name\":\"research-watch\",\"description\":\"Watches research.\",\"prompt\":\"SUB PROMPT\",\"model\":\"anthropic/claude-opus-5.5\",\"subagents\":[]}]"


def _read(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def _is_hex(c: Int) -> Bool:
    return (c >= 0x30 and c <= 0x39) or (c >= 0x61 and c <= 0x66)


def _is_digit(c: Int) -> Bool:
    return c >= 0x30 and c <= 0x39


def _hex_run(bs: Span[Byte, _], i: Int, n: Int) -> Bool:
    if i + n > len(bs):
        return False
    for j in range(i, i + n):
        if not _is_hex(Int(bs[j])):
            return False
    return True


def _match_uuid(bs: Span[Byte, _], i: Int) -> Int:
    """Length of a uuid (8-4-4-4-12) or `session_<32 hex>` at i, else 0."""
    if i + 40 <= len(bs) and String(unsafe_from_utf8=bs[i:i + 8]) == "session_" and _hex_run(bs, i + 8, 32):
        return 40
    var groups: List[Int] = [8, 4, 4, 4, 12]
    var j = i
    for g in range(5):
        if not _hex_run(bs, j, groups[g]):
            return 0
        j += groups[g]
        if g < 4:
            if j >= len(bs) or Int(bs[j]) != 0x2D:
                return 0
            j += 1
    return j - i


def _match_ts(bs: Span[Byte, _], i: Int) -> Int:
    """Length of YYYY-MM-DDTHH:MM:SS(.ffffff)?+00:00 at i, else 0."""
    var shape = "dddd-dd-ddTdd:dd:dd"
    var sb = shape.as_bytes()
    if i + len(sb) > len(bs):
        return 0
    for k in range(len(sb)):
        var c = Int(bs[i + k])
        if Int(sb[k]) == 0x64:
            if not _is_digit(c):
                return 0
        elif c != Int(sb[k]):
            return 0
    var j = i + len(sb)
    if j < len(bs) and Int(bs[j]) == 0x2E:
        j += 1
        for _ in range(6):
            if j >= len(bs) or not _is_digit(Int(bs[j])):
                return 0
            j += 1
    if j + 6 > len(bs) or String(unsafe_from_utf8=bs[j:j + 6]) != "+00:00":
        return 0
    return j + 6 - i


def placeholders(s: String) -> String:
    var bs = s.as_bytes()
    var out = String()
    var run = 0
    var i = 0
    while i < len(bs):
        var u = _match_uuid(bs, i)
        if u > 0:
            out += String(unsafe_from_utf8=bs[run:i]) + "<UUID>"
            i += u
            run = i
            continue
        var t = _match_ts(bs, i)
        if t > 0:
            out += String(unsafe_from_utf8=bs[run:i]) + "<TS>"
            i += t
            run = i
            continue
        i += 1
    out += String(unsafe_from_utf8=bs[run:len(bs)])
    return out^


def _sort_key(body: JsonValue) raises -> String:
    var msgs = body.get("messages")
    var sys = String()
    var first_user = String()
    for i in range(len(msgs)):
        var m = msgs.at(i)
        if m.get("role").as_string() == "system" and i == 0:
            sys = m.get("content").to_json()
        if m.get("role").as_string() == "user" and first_user.byte_length() == 0:
            first_user = m.get("content").to_json()
    return sys + "|" + first_user + "|" + String(len(msgs))


def _normalize(body: JsonValue) raises -> String:
    """The body as compact JSON with the merged file tools reduced to their deepagents
    description prefix (see the module docstring)."""
    var w = JsonWriter()
    w.begin_object()
    for i in range(len(body)):
        var k = body.key_at(i)
        w.key(k)
        if k != "tools":
            w.raw(body.at(i).to_json())
            continue
        var tools = body.at(i)
        w.begin_array()
        for j in range(len(tools)):
            var t = tools.at(j)
            var f = t.get("function")
            var name = f.get("name").as_string()
            if name != "read_file" and name != "write_file" and name != "edit_file":
                w.raw(t.to_json())
                continue
            var desc = f.get("description").as_string()
            var cut = desc.find("\n\nQUMO notes:")
            if cut >= 0:
                var short = String(desc[byte=0:cut])
                desc = short^
            w.begin_object()
            w.kv_string("type", "function")
            w.key("function")
            w.begin_object()
            w.kv_string("name", name)
            w.kv_string("description", desc)
            w.end_object()
            w.end_object()
        w.end_array()
    w.end_object()
    return w.finish()


def _sorted(bodies: List[String]) raises -> List[String]:
    var keyed = List[String]()
    for b in bodies:
        keyed.append(_sort_key(parse_json(b)) + "\u0000" + b)
    sort(keyed)
    var out = List[String]()
    for k in keyed:
        var at = k.find("\u0000")
        out.append(String(k[byte=at + 1:]))
    return out^


def run_scenario(name: String, sort_requests: Bool) raises:
    var golden = parse_json(_read(GOLDEN + "wire_" + name + ".json"))
    var env = FakeEnv(GOLDEN + "wire_" + name + ".replay.jsonl", AGENTS, FileRoutes(False, False))
    var shared = env.shared
    var spec = env.agent_def("orchestrator", List[String]())
    var agent = Agent[FakeEnv](env^, spec, List[String](), "", False)
    var result = agent.run("TASK", Vfs())
    var mine = shared[].replay.requests.copy()
    var expected = List[String]()
    var reqs = golden.get("requests")
    for i in range(len(reqs)):
        expected.append(reqs.at(i).to_json())
    assert_equal(len(mine), len(expected), name + ": number of requests")
    if sort_requests:
        mine = _sorted(mine)
        expected = _sorted(expected)
    for i in range(len(expected)):
        var a = _normalize(parse_json(placeholders(mine[i])))
        var b = _normalize(parse_json(expected[i]))
        if a != b:
            print("--- " + name + " request " + String(i) + " differs")
            print("mojo:   " + a)
            print("python: " + b)
        assert_equal(a, b, name + " request " + String(i))
    assert_equal(result.answer, golden.get("answer").as_string(), name + " answer")
    var files = golden.get("files")
    var mine_files = List[String]()
    for p in result.files.paths():
        mine_files.append(placeholders(p))
    sort(mine_files)
    assert_equal(len(mine_files), len(files), name + " files")
    for i in range(len(files)):
        assert_equal(mine_files[i], files.at(i).as_string(), name + " file")
    print("wire_" + name + " OK")


def main() raises:
    run_scenario("S1", False)
    run_scenario("S2", False)
    run_scenario("S3", True)
    run_scenario("S4", False)
    run_scenario("S5", False)
    run_scenario("S6", False)
    print("test_harness_wire OK")
