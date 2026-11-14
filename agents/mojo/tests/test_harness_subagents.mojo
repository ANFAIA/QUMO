"""Tests harness/subagents.mojo and the `task` steps of the loop (DESIGN.md 2.7.10, 3.2 T4):
listing, unknown types, isolation, final text, file merges, inline vs `run_subagents`,
replay paths, nesting and failures."""

from std.os import makedirs
from std.testing import assert_equal, assert_true, assert_false

from qumo.ai.json import JsonValue, JsonWriter, parse_json

from qumo_agent.core.messages import Message, ROLE_TOOL
from qumo_agent.core.stats import zero_stats
from qumo_agent.harness.env import FileRoutes, AgentDef, SubagentResult
from qumo_agent.harness.loop import Agent
from qumo_agent.harness.subagents import (
    task_tool_spec, general_purpose_def, unknown_subagent_text, child_replay_at, merge_subagent_files,
)
from qumo_agent.harness.testing import FakeEnv
from qumo_agent.harness.vfs import Vfs, VfsTx, new_file_data

comptime TMP = "build/test-fixtures/harness"
comptime AGENTS = "[{\"name\":\"orchestrator\",\"description\":\"Orchestrates.\",\"prompt\":\"MAIN\",\"model\":\"m-main\",\"subagents\":[\"watch\",\"review\"]},{\"name\":\"watch\",\"description\":\"Watches.\",\"prompt\":\"WATCH\",\"model\":\"m-watch\",\"subagents\":[\"orchestrator\",\"review\"]},{\"name\":\"review\",\"description\":\"Reviews.\",\"prompt\":\"REVIEW\",\"model\":\"m-review\",\"subagents\":[]}]"


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


def line(at: String, calls: List[String], content: String) -> String:
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
    w.end_object()
    return w.finish()


def one(c: String) -> List[String]:
    var l = List[String]()
    l.append(c)
    return l^


def task_call(id: String, sub: String, desc: String) -> String:
    return call(id, "task", "{\"description\": \"" + desc + "\", \"subagent_type\": \"" + sub + "\"}")


def test_pure() raises:
    var names: List[String] = ["general-purpose", "watch"]
    var descs: List[String] = ["GP.", "Watches."]
    var spec = task_tool_spec(names, descs)
    assert_true("\n- general-purpose: GP.\n- watch: Watches.\n" in spec.description)
    var allowed: List[String] = ["general-purpose", "watch"]
    assert_equal(unknown_subagent_text("x", allowed), "We cannot invoke subagent x because it does not exist, the only allowed types are `general-purpose`, `watch`")
    assert_equal(child_replay_at("", "c1"), "c1")
    assert_equal(child_replay_at("c1", "c9"), "c1/c9")
    var gp = general_purpose_def(AgentDef("o", "", "", "m-o", List[String](), 200))
    assert_equal(gp.name, "general-purpose")
    assert_equal(gp.model, "m-o")
    assert_equal(gp.recursion_limit, 9999)
    assert_true(gp.prompt.startswith("In order to complete the objective"))
    # merge: whole child dict, key by key; deletions not propagated.
    var child = Vfs()
    child.set("/a", new_file_data("child a", "t"))
    var res = SubagentResult(True, "", True, child.to_json(), "", zero_stats(), "")
    var tx = VfsTx()
    var base = Vfs()
    base.set("/a", new_file_data("parent a", "t0"))
    base.set("/gone", new_file_data("x", "t0"))
    merge_subagent_files(res, tx)
    var view = tx.view(base)
    assert_equal(view.get("/a").value().content, "child a")
    assert_true(Bool(view.get("/gone")))


def run(name: String, lines: List[String], parallel: Int) raises -> Tuple[List[String], List[String], String, List[Message], Vfs, List[Int], String]:
    makedirs(TMP, exist_ok=True)
    var path = TMP + "/sub_" + name + ".replay.jsonl"
    with open(path, "w") as f:
        f.write(String("\n").join(lines) + "\n")
    var env = FakeEnv(path, AGENTS, FileRoutes(False, False))
    env.shared[].parallel = parallel
    var shared = env.shared
    var spec = env.agent_def("orchestrator", List[String]())
    var agent = Agent[FakeEnv](env^, spec, List[String](), "", False)
    var start = Vfs()
    start.set("/shared.md", new_file_data("parent", "t0"))
    start.set("/del.md", new_file_data("parent", "t0"))
    try:
        var r = agent.run("TASK", start^)
        return (shared[].replay.requests.copy(), shared[].replay.request_paths.copy(), r.answer, r.messages.copy(),
                r.files.copy(), shared[].batches.copy(), String())
    except e:
        return (shared[].replay.requests.copy(), shared[].replay.request_paths.copy(), String(), List[Message](),
                Vfs(), shared[].batches.copy(), String(e))


def _two_tasks() -> List[String]:
    var calls = List[String]()
    calls.append(task_call("t1", "watch", "watch it"))
    calls.append(task_call("t2", "general-purpose", "gp it"))
    var lines: List[String] = [
        line("", calls, ""),
        # watch: writes /w.md and /shared.md, deletes /del.md, answers with trailing blanks
        line("t1", one(call("w1", "write_file", "{\"file_path\": \"/w.md\", \"content\": \"from watch\"}")), ""),
        line("t1", one(call("w2", "write_file", "{\"file_path\": \"/shared.md\", \"content\": \"watch\"}")), ""),
        line("t1", one(call("w3", "delete", "{\"file_path\": \"/del.md\"}")), ""),
        line("t1", List[String](), "watch answer  \n"),
        # general-purpose: writes /shared.md too (later call wins), answers
        line("t2", one(call("g1", "write_file", "{\"file_path\": \"/shared.md\", \"content\": \"gp\"}")), ""),
        line("t2", List[String](), "gp answer"),
        line("", List[String](), "final"),
    ]
    return lines^


def test_two_tasks(parallel: Int) raises:
    var r = run("two_" + String(parallel), _two_tasks(), parallel)
    assert_equal(r[6], "")
    assert_equal(r[2], "final")
    # isolation: the child request is its system prompt plus the description only.
    var paths = r[1].copy()
    var first_watch = -1
    for i in range(len(paths)):
        if paths[i] == "t1" and first_watch < 0:
            first_watch = i
    var wb = parse_json(r[0][first_watch])
    assert_equal(wb.get("model").as_string(), "m-watch")
    assert_equal(len(wb.get("messages")), 2)
    assert_equal(wb.get("messages").at(0).get("content").as_string(), "WATCH")
    assert_equal(wb.get("messages").at(1).get("content").as_string(), "watch it")
    # watch's own task tool lists general-purpose and review, not its ancestor orchestrator.
    var tools = wb.get("tools")
    var task_desc = String()
    for i in range(len(tools)):
        if tools.at(i).get("function").get("name").as_string() == "task":
            task_desc = tools.at(i).get("function").get("description").as_string()
    assert_true("- general-purpose: " in task_desc)
    assert_true("- review: Reviews." in task_desc)
    assert_false("- orchestrator:" in task_desc)
    # the general-purpose child: parent's model, DEFAULT_SUBAGENT_PROMPT, no task tool.
    for i in range(len(paths)):
        if paths[i] == "t2":
            var gb = parse_json(r[0][i])
            assert_equal(gb.get("model").as_string(), "m-main")
            var gt = gb.get("tools")
            for j in range(len(gt)):
                assert_true(gt.at(j).get("function").get("name").as_string() != "task")
            break
    # results in call order, without a tool name, rstripped final text.
    var m = r[3].copy()
    var results = List[Message]()
    for x in m:
        if x.role == ROLE_TOOL:
            results.append(x.copy())
    assert_equal(results[0].tool_call_id, "t1")
    assert_equal(results[0].content, "watch answer")
    assert_equal(results[0].name, "")
    assert_equal(results[1].content, "gp answer")
    # files: both children's dicts merged in call order; deletions not propagated.
    var files = r[4].copy()
    assert_equal(files.get("/w.md").value().content, "from watch")
    assert_equal(files.get("/shared.md").value().content, "gp")
    assert_true(Bool(files.get("/del.md")))
    if parallel > 1:
        assert_equal(len(r[5]), 1)
        assert_equal(r[5][0], 2)
    else:
        assert_equal(len(r[5]), 0)


def test_nested_and_failure() raises:
    # watch delegates to review (path t1/r1); review fails (replay exhausted) -> QA-E120.
    var lines: List[String] = [
        line("", one(task_call("t1", "watch", "w")), ""),
        line("t1", one(task_call("r1", "review", "r")), ""),
        line("t1/r1", List[String](), "review done"),
        line("t1", List[String](), "watch done"),
        line("", List[String](), "final"),
    ]
    var r = run("nested", lines, 1)
    assert_equal(r[6], "")
    assert_equal(String(",").join(r[1]), ",t1,t1/r1,t1,")
    var bad: List[String] = [
        line("", one(task_call("t1", "watch", "w")), ""),
    ]
    r = run("failing", bad, 1)
    assert_equal(r[6], "QA-E120: subagent watch failed: QA-E170: replay file exhausted at 't1' after 0 responses")
    # unknown type and an extra argument are tool results, not failures.
    var calls = List[String]()
    calls.append(task_call("t1", "nobody", "x"))
    calls.append(call("t2", "task", "{\"description\": \"x\", \"subagent_type\": \"watch\", \"prompt\": \"p\"}"))
    var lines3: List[String] = [line("", calls, ""), line("", List[String](), "final")]
    r = run("unknown", lines3, 1)
    var tools = List[Message]()
    for x in r[3]:
        if x.role == ROLE_TOOL:
            tools.append(x.copy())
    assert_equal(tools[0].content, "We cannot invoke subagent nobody because it does not exist, the only allowed types are `general-purpose`, `watch`, `review`")
    assert_equal(tools[1].content, "Error invoking tool 'task' with kwargs {'description': 'x', 'subagent_type': 'watch', 'prompt': 'p'} with error:\n prompt: Value error, Unexpected argument 'prompt'; put all instructions for the subagent in `description`.\n Please fix the error and try again.")


def main() raises:
    test_pure()
    test_two_tasks(1)
    test_two_tasks(4)
    test_nested_and_failure()
    print("test_harness_subagents OK")
