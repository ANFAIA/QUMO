"""Tests the virtual file tools and the merged read/write/edit tools (DESIGN.md 2.7.3, 2.7.4).

Part 1 replays every turn of tests/golden/fs_outputs.json (deepagents' StateBackend
driven by a scripted model) through `Agent` + `FakeEnv` (V0 routes, the deepagents
meaning of every path) and compares each tool result and its error status.
Part 2 checks the routing table of the merged tools for V0, V1 and V2 agents.
"""

from std.os import makedirs
from std.testing import assert_equal, assert_true, assert_false

from qumo.ai.json import JsonValue, JsonWriter, parse_json

from qumo_agent.core.messages import ROLE_TOOL
from qumo_agent.core.toolspec import ToolOutcome
from qumo_agent.harness.env import FileRoutes
from qumo_agent.harness.files import mutation_key, merged_file_tools, file_path_arg
from qumo_agent.harness.fs_tools import (
    format_with_line_numbers, truncate_list_repr, truncate_text, vfs_read, vfs_write, media_kind,
)
from qumo_agent.harness.loop import Agent
from qumo_agent.harness.testing import FakeEnv
from qumo_agent.harness.vfs import Vfs, VfsTx

comptime GOLDEN = "agents/mojo/tests/golden/fs_outputs.json"
comptime TMP = "build/test-fixtures/harness"


def _read(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def _write(path: String, text: String) raises:
    with open(path, "w") as f:
        f.write(text)


def _completion_line(at: String, calls: List[String], content: String) -> String:
    """A replay line; `calls` are pre-rendered tool_call objects."""
    var w = JsonWriter()
    w.begin_object()
    w.kv_string("at", at)
    w.key("response")
    w.begin_object()
    w.key("choices")
    w.begin_array()
    w.begin_object()
    w.key("message")
    w.begin_object()
    w.kv_string("role", "assistant")
    if len(calls) == 0:
        w.kv_string("content", content)
    else:
        w.kv_null("content")
        w.key("tool_calls")
        w.begin_array()
        for c in calls:
            w.raw(c)
        w.end_array()
    w.end_object()
    w.end_object()
    w.end_array()
    w.end_object()
    w.end_object()
    return w.finish()


def _call(id: String, name: String, args_json: String) -> String:
    var w = JsonWriter()
    w.begin_object()
    w.kv_string("id", id)
    w.kv_string("type", "function")
    w.key("function")
    w.begin_object()
    w.kv_string("name", name)
    w.kv_string("arguments", args_json)
    w.end_object()
    w.end_object()
    return w.finish()


def replay_fs_golden() raises:
    var g = parse_json(_read(GOLDEN))
    var steps = g.get("steps")
    var lines = String()
    var n = 0
    for i in range(len(steps)):
        var turn = steps.at(i)
        var calls = List[String]()
        for j in range(len(turn)):
            n += 1
            calls.append(_call("call_" + String(n), turn.at(j).at(0).as_string(), turn.at(j).at(1).to_json()))
        lines += _completion_line("", calls, "") + "\n"
    lines += _completion_line("", List[String](), "done") + "\n"
    makedirs(TMP, exist_ok=True)
    var replay = TMP + "/fs_outputs.replay.jsonl"
    _write(replay, lines)
    var env = FakeEnv(replay, "[{\"name\":\"probe\",\"prompt\":\"P\",\"model\":\"m\",\"subagents\":[],\"tools\":[],\"recursion_limit\":1000}]",
                      FileRoutes(False, False))
    var spec = env.agent_def("probe", List[String]())
    var agent = Agent[FakeEnv](env^, spec, List[String](), "", False)
    var result = agent.run("go", Vfs())
    var expected = g.get("results")
    var k = 0
    for m in result.messages:
        if m.role != ROLE_TOOL:
            continue
        var e = expected.at(k)
        var label = e.get("id").as_string() + " " + e.get("call").to_json()
        var exp_content = e.get("content").as_string()
        var got = m.text()
        if got != exp_content:
            print("--- " + label)
            print("mojo:   " + got)
            print("python: " + exp_content)
        assert_equal(got, exp_content, label)
        assert_equal(m.is_error, e.get("status").as_string() == "error", label + " status")
        assert_equal(m.tool_call_id, e.get("id").as_string())
        k += 1
    assert_equal(k, len(expected))
    var final_files = g.get("final_files")
    assert_equal(len(result.files.files), len(final_files))
    for i in range(len(final_files)):
        var fd = result.files.get(final_files.key_at(i))
        assert_true(Bool(fd), final_files.key_at(i))
        assert_equal(fd.value().content, final_files.at(i).as_string())
    print("fs_outputs: " + String(k) + " results equal")


def _agent_run(routes: FileRoutes, calls_json: List[String], mut env_out: List[String]) raises -> List[String]:
    """One turn with the given tool calls, then "done"; returns the tool results and the
    FakeEnv's recorded relative operations in env_out."""
    var calls = List[String]()
    for i in range(len(calls_json)):
        var c = parse_json(calls_json[i])
        calls.append(_call("c" + String(i + 1), c.get("name").as_string(), c.get("args").to_json()))
    var replay = TMP + "/merged.replay.jsonl"
    _write(replay, _completion_line("", calls, "") + "\n" + _completion_line("", List[String](), "done") + "\n")
    var env = FakeEnv(replay, "[{\"name\":\"a\",\"prompt\":\"P\",\"model\":\"m\",\"subagents\":[],\"tools\":[]}]", routes)
    env.shared[].rel_files["notes.md"] = "one\ntwo"
    env.shared[].sandbox_path = "/tmp/sb"
    var shared = env.shared
    var spec = env.agent_def("a", List[String]())
    var agent = Agent[FakeEnv](env^, spec, List[String](), "", False)
    var result = agent.run("go", Vfs())
    var out = List[String]()
    for m in result.messages:
        if m.role == ROLE_TOOL:
            out.append(m.text())
    for op in shared[].rel_ops:
        env_out.append(op)
    return out^


def test_merged_routing() raises:
    var v0 = FileRoutes(False, False)
    var v1 = FileRoutes(True, False)
    var v2 = FileRoutes(True, True)
    # V0: relative paths are virtual; aliases convert (start_line 1 = offset 0).
    var ops = List[String]()
    var calls: List[String] = [
        "{\"name\":\"write_file\",\"args\":{\"path\":\"notes.md\",\"content\":\"a\\nb\\nc\"}}",
    ]
    var r = _agent_run(v0, calls, ops)
    assert_equal(r[0], "Updated file /notes.md")
    calls = [
        "{\"name\":\"write_file\",\"args\":{\"file_path\":\"/n.md\",\"content\":\"a\\nb\\nc\"}}",
    ]
    r = _agent_run(v0, calls, ops)
    assert_equal(r[0], "Updated file /n.md")
    assert_equal(len(ops), 0)
    # V0: notes.md and /notes.md share one mutation key; the second is refused.
    calls = [
        "{\"name\":\"write_file\",\"args\":{\"path\":\"notes.md\",\"content\":\"x\"}}",
        "{\"name\":\"write_file\",\"args\":{\"file_path\":\"/notes.md\",\"content\":\"y\"}}",
        "{\"name\":\"edit_file\",\"args\":{\"path\":\"/other.md\",\"old\":\"a\",\"new\":\"b\"}}",
        "{\"name\":\"delete\",\"args\":{\"file_path\":\"/other.md\"}}",
    ]
    r = _agent_run(v0, calls, ops)
    assert_equal(r[0], "Updated file /notes.md")
    assert_equal(r[1], "Error: parallel file mutations to the same path are not allowed.")
    assert_equal(r[2], "Error: File '/other.md' not found")
    assert_equal(r[3], "Error: parallel file mutations to the same path are not allowed.")
    # Conflicting file_path and path; missing path.
    calls = [
        "{\"name\":\"read_file\",\"args\":{\"file_path\":\"/a\",\"path\":\"/b\"}}",
        "{\"name\":\"read_file\",\"args\":{\"file_path\":\"/a\",\"path\":\"/a\"}}",
        "{\"name\":\"read_file\",\"args\":{\"offset\":3}}",
        "{\"name\":\"edit_file\",\"args\":{\"path\":\"/a\"}}",
    ]
    r = _agent_run(v0, calls, ops)
    assert_equal(r[0], "Error: file_path and path name different files; give only one.")
    assert_equal(r[1], "Error: File '/a' not found")
    assert_equal(r[2], "Error invoking tool 'read_file' with kwargs {'offset': 3} with error:\n file_path: Field required\n Please fix the error and try again.")
    assert_equal(r[3], "Error invoking tool 'edit_file' with kwargs {'path': '/a'} with error:\n old_string: Field required\nnew_string: Field required\n Please fix the error and try again.")
    # V1: relative reads go to the env (start_line/max_lines from offset/limit), relative writes refused.
    ops = List[String]()
    calls = [
        "{\"name\":\"read_file\",\"args\":{\"path\":\"notes.md\"}}",
        "{\"name\":\"read_file\",\"args\":{\"file_path\":\"notes.md\",\"offset\":4,\"limit\":7}}",
        "{\"name\":\"read_file\",\"args\":{\"path\":\"notes.md\",\"start_line\":2,\"max_lines\":\"3\"}}",
        "{\"name\":\"write_file\",\"args\":{\"path\":\"notes.md\",\"content\":\"x\"}}",
        "{\"name\":\"edit_file\",\"args\":{\"path\":\"notes.md\",\"old\":\"one\",\"new\":\"1\"}}",
        "{\"name\":\"write_file\",\"args\":{\"path\":\"/v.md\",\"content\":\"l1\\nl2\\nl3\"}}",
    ]
    r = _agent_run(v1, calls, ops)
    assert_equal(r[0], "rel:one\ntwo")
    assert_equal(r[3], "Error: relative paths name files of the QUMO repository, which this agent cannot write. Write to the virtual file system with a path that starts with / (for example /notes.md), or save results with write_report.")
    assert_equal(r[4], r[3])
    assert_equal(r[5], "Updated file /v.md")
    assert_equal(len(ops), 3)
    assert_equal(ops[0], "read notes.md 1 400")
    assert_equal(ops[1], "read notes.md 5 7")
    assert_equal(ops[2], "read notes.md 2 3")
    # V1: absolute reads stay virtual, with start_line/max_lines converted.
    ops = List[String]()
    calls = [
        "{\"name\":\"write_file\",\"args\":{\"path\":\"/v.md\",\"content\":\"l1\\nl2\\nl3\"}}",
    ]
    r = _agent_run(v1, calls, ops)
    # V2: relative writes and edits go to the sandbox; mutation keys follow the route.
    ops = List[String]()
    calls = [
        "{\"name\":\"write_file\",\"args\":{\"path\":\"src/x.mojo\",\"content\":\"abc\"}}",
        "{\"name\":\"edit_file\",\"args\":{\"file_path\":\"./src/x.mojo\",\"old_string\":\"a\",\"new_string\":\"b\"}}",
        "{\"name\":\"edit_file\",\"args\":{\"path\":\"notes.md\",\"old\":\"one\",\"new\":\"1\",\"replace_all\":true}}",
        "{\"name\":\"write_file\",\"args\":{\"path\":\"/src/x.mojo\",\"content\":\"virtual\"}}",
    ]
    r = _agent_run(v2, calls, ops)
    assert_equal(r[0], "wrote src/x.mojo")
    assert_equal(r[1], "Error: parallel file mutations to the same path are not allowed.")
    assert_equal(r[2], "edited notes.md")
    assert_equal(r[3], "Updated file /src/x.mojo")
    assert_equal(len(ops), 2)
    # mutation_key
    var a = parse_json("{\"path\":\"notes.md\"}")
    assert_equal(mutation_key("write_file", a, v0), "vfs:/notes.md")
    assert_equal(mutation_key("write_file", a, v1), "")
    assert_equal(mutation_key("write_file", a, v2), "rel:notes.md")
    assert_equal(mutation_key("delete", parse_json("{\"file_path\":\"x/../y\"}"), v2), "")
    assert_equal(mutation_key("delete", parse_json("{\"file_path\":\"x\"}"), v2), "vfs:/x")
    assert_equal(mutation_key("read_file", a, v0), "")
    assert_equal(mutation_key("edit_file", parse_json("{\"file_path\":\"/a//b/\"}"), v1), "vfs:/a/b")
    # tool descriptions: deepagents text + "\n\n" + the variant's notes.
    var t0 = merged_file_tools(v0)
    var t2 = merged_file_tools(v2)
    assert_equal(t0[0].name, "read_file")
    assert_true("\n\nQUMO notes:\n- `path` is accepted as another name for `file_path`, `start_line`" in t0[0].description)
    assert_true("open sandbox worktree" in t2[1].description)
    print("merged routing OK")


def test_helpers() raises:
    var lines: List[String] = ["a", "b"]
    assert_equal(format_with_line_numbers(lines, 9), " 9  a\n10  b")
    var long_line = String()
    for _ in range(5001):
        long_line += "x"
    var one: List[String] = [long_line]
    var f = format_with_line_numbers(one, 1)
    assert_true(f.startswith("  1  xxx"))
    assert_true("\n1.1  x" in f)
    var items: List[String] = ["/a.txt", "/dir/"]
    assert_equal(truncate_list_repr(items), "['/a.txt', '/dir/']")
    var big = String()
    for _ in range(80001):
        big += "é"
    var t = truncate_text(big)
    assert_true(t.endswith("\n... [results truncated, try being more specific with your parameters]"))
    # X8: media placeholder; an empty media file gets the empty-file reminder.
    var base = Vfs()
    var tx = VfsTx()
    _ = vfs_write(base, tx, "/img/a.PNG", "binary?", "t")
    _ = vfs_write(base, tx, "/e.pdf", "", "t")
    assert_equal(vfs_read(base, tx, "/img/a.PNG", 0, 100).text,
                 "[read_file: /img/a.PNG was not attached because this model does not support image content (image/png).]")
    assert_equal(vfs_read(base, tx, "/e.pdf", 0, 100).text, "System reminder: File exists but has empty contents")
    assert_equal(media_kind("/v.mkv")[1], "video/x-matroska")
    assert_equal(media_kind("/.png")[0], "")
    print("helpers OK")


def main() raises:
    replay_fs_golden()
    test_merged_routing()
    test_helpers()
    print("test_harness_fs_tools OK")
