"""Tests AppEnv (DESIGN.md 2.10) in process: the agent loop over a replayed model with
QUMO's real MCP server, the per-agent tool lists and file routes, tool dispatch (local,
MCP), relative file operations, the sandbox shared by the agent tree, the deterministic
clock and ids, and `--model` applying to the top agent only (inline subagents)."""

from std.memory import ArcPointer
from std.testing import assert_equal, assert_true, assert_false

from qumo.ai.json import JsonValue, parse_json

from qumo_agent.app.env import (
    AppEnv, AppOptions, AppShared, make_shared, make_shared_offline, run_agent_once, shutdown_shared,
)
from qumo_agent.app.options import default_options
from qumo_agent.app.testing import (
    REPLAY_BELL, BELL_ANSWER, replay_line, tool_call_json, write_replay,
)
from qumo_agent.core.messages import Message, user_message, ChatRequest
from qumo_agent.core.toolspec import ToolSpec, TOOL_TOOLSMITH
from qumo_agent.harness.env import RelativeFileOp, ToolInvocation, FILE_READ, FILE_WRITE, FILE_EDIT
from qumo_agent.native.fs import make_dirs, make_temp_dir, read_text_lossy, realpath_strict, remove_tree, exists


def options(replay: String) -> AppOptions:
    var o = default_options(".", "build/qumo-agent")
    o.replay_file = replay
    o.verbose = False
    o.deterministic = True
    return o^


def names(env: AppEnv) -> List[String]:
    var out = List[String]()
    for t in env.user_tools():
        out.append(t.name)
    return out^


def has(items: List[String], x: String) -> Bool:
    for i in items:
        if i == x:
            return True
    return False


def invocation(env: AppEnv, name: String, args_json: String) raises -> ToolInvocation:
    return ToolInvocation(name, parse_json(args_json), "call_x", env.agent)


def test_tools_and_routes() raises:
    var shared = make_shared(options(REPLAY_BELL), "code-review")
    var review = AppEnv(shared, "code-review")
    var n = names(review)
    assert_true(has(n, "list_files") and has(n, "search_code") and has(n, "run_check") and has(n, "write_report"))
    assert_false(has(n, "read_file") or has(n, "write_file") or has(n, "edit_file"))   # deepagents' slots
    assert_false(has(n, "open_sandbox"))
    var routes = review.file_routes()
    assert_true(routes.repo_read)
    assert_false(routes.sandbox_write)

    var operator = AppEnv(shared, "qumo-operator")
    assert_false(operator.file_routes().repo_read)
    assert_false(operator.file_routes().sandbox_write)
    var on = names(operator)
    assert_true(has(on, "qumo_run_qasm") and has(on, "write_report"))
    assert_true(len(on) >= 25)                                          # write_report + the 24 MCP tools

    var auto = AppEnv(shared, "autoresearch")
    assert_true(auto.file_routes().repo_read and auto.file_routes().sandbox_write)
    assert_true(has(names(auto), "open_sandbox") and has(names(auto), "run_check"))

    var smith = AppEnv(shared, "mojo-toolsmith")
    assert_true(smith.file_routes().repo_read and smith.file_routes().sandbox_write)
    var sn = names(smith)
    assert_true(has(sn, "toolsmith_new") and has(sn, "toolsmith_publish") and has(sn, "qumo_describe"))
    assert_false(has(sn, "run_check"))
    for t in smith.user_tools():
        if t.name == "qumo_run_qasm":
            assert_true(False, "the toolsmith only gets qumo_describe from the MCP server")
        if t.name == "toolsmith_new":
            assert_equal(t.kind, TOOL_TOOLSMITH)

    # Subagent definitions: no ancestors, the 9999 limit, the agent's own model.
    var ad = review.agent_def("execution", List[String]())
    assert_equal(ad.recursion_limit, 9999)
    assert_equal(ad.name, "execution")
    var orch = review.agent_def("orchestrator", List[String]())
    var anc: List[String] = ["orchestrator"]
    var orch2 = review.agent_def("autoresearch", anc)
    assert_false(has(orch2.subagents, "orchestrator"))
    assert_true(has(orch.subagents, "autoresearch"))
    var bad = False
    try:
        _ = review.agent_def("no-such-agent", List[String]())
    except e:
        bad = String(e).startswith("QA-E140")
    assert_true(bad)
    shutdown_shared(shared)


def test_dispatch_and_files() raises:
    var shared = make_shared(options(REPLAY_BELL), "mojo-toolsmith")
    var env = AppEnv(shared, "mojo-toolsmith")
    # Local read-only tools.
    var r = env.call_tool(invocation(env, "list_files", "{\"path\": \"agents/prompts\", \"pattern\": \"*.md\"}"))
    assert_false(r.is_error)
    assert_true(r.text.find("agents/prompts/preamble.md") >= 0, r.text)
    r = env.call_tool(invocation(env, "search_code", "{\"regex\": \"class Foo[a-z]\", \"path\": \"agents/mojo/qumo_agent/app\", \"glob\": \"*.mojo\"}"))
    assert_equal(r.text, "no matches")
    r = env.call_tool(invocation(env, "search_code", "{\"regex\": \"struct AppEnv\", \"path\": \"agents/mojo/qumo_agent/app\", \"glob\": \"*.mojo\"}"))
    assert_true(r.text.find("app/env.mojo") >= 0, r.text)
    # QUMO MCP tool: a parts outcome from the real server.
    r = env.call_tool(invocation(env, "qumo_describe", "{\"topic\": \"capabilities\"}"))
    assert_false(r.is_error)
    assert_true(r.is_parts)
    assert_true(r.joined_text().byte_length() > 100)
    # A tool this agent does not have is an error outcome, not a crash.
    r = env.call_tool(invocation(env, "search_arxiv", "{\"query\": \"x\"}"))
    assert_true(r.is_error)

    # Relative file operations: reads follow the repository, writes need a sandbox.
    var read = env.relative_file_op(RelativeFileOp(FILE_READ, "agents/prompts/preamble.md", 1, 3, "", "", "", False))
    assert_true(read.text.startswith("     1\t") or read.text.startswith("1\t"), read.text)
    var write = env.relative_file_op(RelativeFileOp(FILE_WRITE, "notes.md", 1, 0, "hi", "", "", False))
    assert_equal(write.text, "refused: call open_sandbox first; the main checkout is read-only for agents")
    var edit = env.relative_file_op(RelativeFileOp(FILE_EDIT, "notes.md", 1, 0, "", "a", "b", False))
    assert_equal(edit.text, "refused: call open_sandbox first; the main checkout is read-only for agents")

    # With a sandbox (a temporary directory stands in for the worktree): writes land there,
    # and every agent of the tree sees it.
    var dir = realpath_strict(make_temp_dir("qumo-app-env-sb-"))
    assert_equal(env.sandbox(), "")
    env.adopt_sandbox(dir)
    var other = env.for_agent("autoresearch")
    assert_equal(other.sandbox(), dir)
    var w2 = other.relative_file_op(RelativeFileOp(FILE_WRITE, "notes.md", 1, 0, "hello", "", "", False))
    assert_equal(w2.text, "wrote notes.md (5 chars)")
    assert_true(exists(dir + "/notes.md"))
    var e2 = other.relative_file_op(RelativeFileOp(FILE_EDIT, "notes.md", 1, 0, "", "hello", "world", False))
    assert_equal(e2.text, "edited notes.md")
    var rd = env.relative_file_op(RelativeFileOp(FILE_READ, "notes.md", 1, 10, "", "", "", False))
    assert_true(rd.text.find("world") >= 0, rd.text)
    remove_tree(dir)
    shutdown_shared(shared)


def test_local_tools_dispatch() raises:
    """Every local tool name reaches its implementation with the validated arguments."""
    var shared = make_shared(options(REPLAY_BELL), "autoresearch")
    var env = AppEnv(shared, "autoresearch")
    # module_watchlist: offline, JSON with the module's data.
    var r = env.call_tool(invocation(env, "module_watchlist", "{\"module\": \"ir\"}"))
    assert_false(r.is_error)
    assert_equal(parse_json(r.text).get_string("module", ""), "ir")
    r = env.call_tool(invocation(env, "module_watchlist", "{\"module\": \"nope\"}"))
    assert_true(parse_json(r.text).get_string("error", "").startswith("unknown module 'nope'"), r.text)
    # run_check: the argv list is passed whole; refusals come back as text.
    r = env.call_tool(invocation(env, "run_check", "{\"argv\": [\"git\", \"status\", \"--short\"], \"timeout_s\": 60}"))
    assert_true(r.text.startswith("exit "), r.text)
    r = env.call_tool(invocation(env, "run_check", "{\"argv\": [\"rm\", \"-rf\", \"x\"]}"))
    assert_true(r.text.startswith("refused: not on the allow-list. Allowed: "), r.text)
    # write_report always writes into the main repository's agents/reports/.
    var unique = "app-env-test-" + env.new_uuid4_hex()
    r = env.call_tool(invocation(env, "write_report", "{\"name\": \"" + unique + "\", \"markdown\": \"# t\\nbody\"}"))
    assert_true(r.text.startswith("agents/reports/") and r.text.endswith(".md"), r.text)
    assert_equal(read_text_lossy(r.text), "# t\nbody")
    remove_tree(r.text)
    assert_equal(env.sandbox(), "")
    shutdown_shared(shared)


def test_clock_ids_and_model() raises:
    var shared = make_shared(options(REPLAY_BELL), "code-review")
    var env = AppEnv(shared, "code-review")
    assert_equal(env.now_utc_iso(), "2026-01-01T00:00:00+00:00")
    assert_equal(env.now_utc_iso(), "2026-01-01T00:00:01+00:00")
    assert_equal(env.new_uuid4_hex(), "00000000000000000000000000000001")
    assert_equal(env.new_uuid4_hex(), "00000000000000000000000000000002")
    assert_equal(env.parallel_limit(), 4)
    assert_equal(env.take_transport_stats().http_requests, 0)

    # Non-deterministic mode reads the real clock and random ids.
    var o = options(REPLAY_BELL)
    o.deterministic = False
    var shared2 = make_shared(o^, "code-review")
    var env2 = AppEnv(shared2, "code-review")
    assert_true(env2.now_utc_iso().startswith("20"))
    assert_equal(env2.new_uuid4_hex().byte_length(), 32)
    assert_true(env2.new_uuid4_hex() != env2.new_uuid4_hex())

    # The replayed model is the model.
    var msgs = List[Message]()
    msgs.append(user_message("hello"))
    var req = ChatRequest("m", "SYS", msgs^, List[ToolSpec](), "code-review", "")
    var resp = env.complete(req)
    assert_equal(resp.message.content, "Validating first.")
    assert_equal(resp.prompt_tokens, 100)
    shutdown_shared(shared)
    shutdown_shared(shared2)


def test_run_agent_once() raises:
    var shared = make_shared(options(REPLAY_BELL), "qumo-operator")
    var r = run_agent_once(shared, "qumo-operator", "Run a Bell state with seed 7")
    assert_equal(r.answer, BELL_ANSWER)
    assert_equal(r.stats.model_calls, 3)
    assert_equal(r.stats.tool_calls, 2)
    assert_equal(r.stats.mcp_calls, 2)
    assert_equal(r.stats.prompt_tokens, 600)
    assert_equal(r.stats.completion_tokens, 60)
    var requests = shared[].replay.value()[].requests.copy()
    assert_equal(len(requests), 3)
    var first = parse_json(requests[0])
    assert_true(first.get("messages").at(0).get_string("content", "").find("Role: QUMO operator.") >= 0)
    assert_equal(first.get_string("model", ""), "anthropic/claude-sonnet-5.5")
    shutdown_shared(shared)

    # The replay is consumed: a second run in the same process fails with QA-E170.
    var shared2 = make_shared(options(REPLAY_BELL), "execution")
    _ = run_agent_once(shared2, "execution", "first")
    var failed = False
    try:
        _ = run_agent_once(shared2, "execution", "again")
    except e:
        failed = String(e).startswith("QA-E170: replay file exhausted")
    assert_true(failed)
    shutdown_shared(shared2)


def test_model_override_is_top_only() raises:
    """--model applies to the top agent; a named subagent uses its own model."""
    var dir = make_temp_dir("qumo-app-env-model-")
    var lines = List[String]()
    var t = List[String]()
    t.append(tool_call_json("t1", "task", "{\"description\": \"size it\", \"subagent_type\": \"execution\"}"))
    t.append(tool_call_json("t2", "task", "{\"description\": \"anything\", \"subagent_type\": \"general-purpose\"}"))
    lines.append(replay_line("", t, ""))
    lines.append(replay_line("t1", List[String](), "sized"))
    lines.append(replay_line("t2", List[String](), "done"))
    lines.append(replay_line("", List[String](), "all done"))
    write_replay(dir + "/r.jsonl", lines)
    var o = options(dir + "/r.jsonl")
    o.model_override = "vendor/top-model"
    o.parallel = 1                                   # inline subagents
    var shared = make_shared(o^, "orchestrator")
    var r = run_agent_once(shared, "orchestrator", "go")
    assert_equal(r.answer, "all done")
    assert_equal(r.stats.subagent_runs, 2)
    var rp = shared[].replay.value().copy()
    var models = List[String]()
    for i in range(len(rp[].requests)):
        models.append(parse_json(rp[].requests[i]).get_string("model", ""))
    # Order: root, t1 (execution), t2 (general-purpose = the parent's model), root.
    assert_equal(len(models), 4)
    assert_equal(models[0], "vendor/top-model")
    assert_equal(models[1], "anthropic/claude-sonnet-5.5")
    assert_equal(models[2], "vendor/top-model")
    assert_equal(models[3], "vendor/top-model")
    assert_equal(rp[].request_paths[1], "t1")
    shutdown_shared(shared)
    remove_tree(dir)


def test_offline_listing_needs_no_model() raises:
    var o = default_options(".", "build/qumo-agent")
    o.verbose = False
    var shared = make_shared_offline(o^, "orchestrator")
    var env = AppEnv(shared, "orchestrator", False)          # no QUMO MCP session
    assert_true(has(names(env), "list_files"))
    assert_false(has(names(env), "qumo_describe"))
    assert_false(shared[].qumo_mcp.__bool__())
    var failed = False
    try:
        var req = ChatRequest("m", "S", List[Message](), List[ToolSpec](), "orchestrator", "")
        _ = env.complete(req)
    except e:
        failed = String(e).startswith("QA-E140")
    assert_true(failed)


def main() raises:
    test_tools_and_routes()
    test_dispatch_and_files()
    test_local_tools_dispatch()
    test_clock_ids_and_model()
    test_run_agent_once()
    test_model_override_is_top_only()
    test_offline_listing_needs_no_model()
    print("test_app_env OK")
