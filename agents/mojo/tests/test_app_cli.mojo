"""Tests the qumo-agent command line (DESIGN.md 7): usage errors and exit codes (0/1/2/3),
`list`, `--dry-run` JSON for all ten agents (wire lists equal the Python capture), and `run`
with a replayed model (answer, statistics, per-turn logs, --module, --quiet, --stats-json,
task from stdin, secrets never printed). Needs build/qumo (QUMO's MCP server). The test
binary is built on demand from agents/mojo/tests/fixtures/app/main_app.mojo."""

from std.testing import assert_equal, assert_true, assert_false

from qumo.ai.json import JsonValue, parse_json

from qumo_agent.app.testing import (
    app_binary, app_env, run_app, REPLAY_BELL, BELL_ANSWER,
)
from qumo_agent.native.fs import exists, make_dirs, make_temp_dir, read_text_lossy, realpath_strict, remove_tree
from qumo_agent.native.sandbox import CONF_CHECK, Confinement, sandbox_kind

comptime SECRET = "sk-test-secret-value-123456"


def args(items: List[String]) -> List[String]:
    return items.copy()


def env_with(extra: List[String]) -> List[String]:
    return app_env(extra)


def test_usage_and_exit_codes() raises:
    var none = List[String]()
    var r = run_app(none, env_with(none), "", 20000)
    assert_equal(r.exit_code, 2)
    assert_true(r.stderr.find("missing command") >= 0, r.stderr)
    assert_true(r.stderr.find("usage:") >= 0)

    r = run_app(["bogus"], env_with(none), "", 20000)
    assert_equal(r.exit_code, 2)
    assert_true(r.stderr.find("unknown command bogus") >= 0, r.stderr)

    r = run_app(["help"], env_with(none), "", 20000)
    assert_equal(r.exit_code, 0)
    assert_true(r.stdout.startswith("usage:"), r.stdout)
    assert_true(r.stdout.find("qumo-agent tools list | show NAME") >= 0)

    r = run_app(["run", "qumo-operator"], env_with(none), "", 20000)
    assert_equal(r.exit_code, 2)
    assert_true(r.stderr.find("run needs AGENT and TASK") >= 0, r.stderr)

    r = run_app(["run", "qumo-operator", "x", "--nope"], env_with(none), "", 20000)
    assert_equal(r.exit_code, 2)
    assert_true(r.stderr.find("unknown option --nope") >= 0, r.stderr)

    r = run_app(["run", "qumo-operator", "x", "--recursion-limit", "0"], env_with(none), "", 20000)
    assert_equal(r.exit_code, 2)
    r = run_app(["run", "qumo-operator", "x", "--parallel", "abc"], env_with(none), "", 20000)
    assert_equal(r.exit_code, 2)
    r = run_app(["run", "qumo-operator", "x", "--model"], env_with(none), "", 20000)
    assert_equal(r.exit_code, 2)
    assert_true(r.stderr.find("--model needs a value") >= 0, r.stderr)

    # An option of another command is a usage error, not silently ignored.
    r = run_app(["list", "--model", "x"], env_with(none), "", 20000)
    assert_equal(r.exit_code, 2)
    assert_true(r.stderr.find("does not apply to list") >= 0, r.stderr)
    r = run_app(["serve", "qumo-operator", "--dry-run"], env_with(none), "", 20000)
    assert_equal(r.exit_code, 2)
    r = run_app(["call", "qumo-operator", "x", "--stats-json", "f"], env_with(none), "", 20000)
    assert_equal(r.exit_code, 2)
    r = run_app(["reports", "--limit", "many"], env_with(none), "", 20000)
    assert_equal(r.exit_code, 2)

    # The task must not be empty; `-` reads stdin.
    r = run_app(["run", "qumo-operator", "   ", "--replay", REPLAY_BELL], env_with(none), "", 20000)
    assert_equal(r.exit_code, 2)
    assert_true(r.stderr.find("the task is empty") >= 0, r.stderr)
    r = run_app(["run", "qumo-operator", "-", "--replay", REPLAY_BELL], env_with(none), "  \n", 20000)
    assert_equal(r.exit_code, 2)
    r = run_app(["run", "qumo-operator", "x", "--module", "nope", "--replay", REPLAY_BELL], env_with(none), "", 20000)
    assert_equal(r.exit_code, 2)
    assert_true(r.stderr.find("unknown module 'nope'") >= 0, r.stderr)

    # Configuration errors are exit code 3.
    r = run_app(["run", "no-such-agent", "x", "--replay", REPLAY_BELL], env_with(none), "", 20000)
    assert_equal(r.exit_code, 3)
    assert_true(r.stderr.find("qumo-agent: error: QA-E140: unknown agent 'no-such-agent'") >= 0, r.stderr)
    r = run_app(["call", "no-such-agent", "x"], env_with(none), "", 20000)
    assert_equal(r.exit_code, 3)
    r = run_app(["run", "qumo-operator", "x"], env_with(none), "", 20000)   # no key, no replay
    assert_equal(r.exit_code, 3)
    assert_true(r.stderr.find("OPENROUTER_API_KEY") >= 0, r.stderr)
    assert_true(r.stderr.find("--replay FILE") >= 0, r.stderr)

    # A failing run is exit code 1: an exhausted replay.
    var dir = make_temp_dir("qumo-app-cli-")
    var empty = dir + "/empty.jsonl"
    with open(empty, "w") as f:
        f.write("// nothing\n")
    r = run_app(["run", "qumo-operator", "x", "--replay", empty], env_with(none), "", 20000)
    assert_equal(r.exit_code, 1)
    assert_true(r.stderr.find("QA-E170: replay file exhausted") >= 0, r.stderr)
    remove_tree(dir)


def test_list() raises:
    var dir = make_temp_dir("qumo-app-list-")
    var env = env_with(["QUMO_AGENT_TOOLS_DIR=" + dir])
    var r = run_app(["list"], env, "", 20000)
    assert_equal(r.exit_code, 0, r.stderr)
    var lines = r.stdout.split("\n")
    var agents = 0
    for l in lines:
        if l.startswith("research-watch ") or l.startswith("code-review ") or l.startswith("autoresearch ") \
                or l.startswith("qumo-operator ") or l.startswith("algorithm-research ") \
                or l.startswith("qec-research ") or l.startswith("compiler-research ") \
                or l.startswith("execution ") or l.startswith("orchestrator ") or l.startswith("mojo-toolsmith "):
            agents += 1
    assert_equal(agents, 10)
    assert_true(r.stdout.find("mojo-toolsmith") >= 0)
    assert_true(r.stdout.find("[mojo-only]") >= 0)
    assert_true(r.stdout.find("-> research-watch, code-review") >= 0)
    assert_true(r.stdout.find("\nmodules: ir, sim, gpu, compiler, algorithms, qec, ft, exec, ai\n") >= 0, r.stdout)
    assert_true(r.stdout.find("alphaXiv: off") >= 0)
    assert_true(r.stdout.find("generated tools: 0/0 reviewed") >= 0)
    # Name column is 20 wide, model column 32 wide (Python's format).
    var first = String(lines[0])
    assert_equal(String(first[byte=0:20]), "research-watch      ")

    r = run_app(["list", "--json"], env, "", 20000)
    assert_equal(r.exit_code, 0, r.stderr)
    var doc = parse_json(r.stdout)
    assert_equal(len(doc.get("agents")), 10)
    assert_equal(doc.get("agents").at(9).get_string("name", ""), "mojo-toolsmith")
    assert_true(doc.get("agents").at(9).get_bool("mojo_only", False))
    assert_equal(doc.get("generated_tools").get_int("total", -1), 0)

    # The model column honours QUMO_AGENTS_MODEL.
    var env2 = env_with(["QUMO_AGENT_TOOLS_DIR=" + dir, "QUMO_AGENTS_MODEL=vendor/model-x"])
    r = run_app(["list"], env2, "", 20000)
    assert_true(r.stdout.find("vendor/model-x") >= 0)
    remove_tree(dir)


def _names(v: JsonValue) raises -> List[String]:
    var out = List[String]()
    for i in range(len(v)):
        out.append(v.at(i).as_string())
    return out^


def test_dry_run_all_agents() raises:
    var golden = parse_json(read_text_lossy("agents/mojo/tests/fixtures/tools/wire_tool_names.json"))
    var agents = golden.get("agents")
    var checked = 0
    var dir = make_temp_dir("qumo-app-dry-")
    var env = env_with(["QUMO_AGENT_TOOLS_DIR=" + dir])
    for i in range(len(agents)):
        var name = agents.key_at(i)
        var r = run_app(["run", name, "task", "--dry-run"], env, "", 60000)
        assert_equal(r.exit_code, 0, name + ": " + r.stderr)
        var d = parse_json(r.stdout)
        var keys = d.keys()
        var expect: List[String] = ["agent", "model", "local_tools", "qumo_mcp", "alphaxiv", "subagents", "wire_tools", "generated_tools"]
        assert_equal(len(keys), len(expect))
        for k in range(len(expect)):
            assert_equal(keys[k], expect[k])
        assert_equal(d.get_string("agent", ""), name)
        var wire = _names(d.get("wire_tools"))
        var want = _names(agents.at(i))
        assert_equal(len(wire), len(want), name + " wire tools: " + String(", ").join(wire))
        for k in range(len(want)):
            assert_equal(wire[k], want[k], name + " wire tool " + String(k))
        checked += 1
    assert_equal(checked, 9)

    # The Mojo-only toolsmith: toolsmith tools, no run_check (K5).
    var r = run_app(["run", "mojo-toolsmith", "task", "--dry-run"], env, "", 60000)
    assert_equal(r.exit_code, 0, r.stderr)
    var d = parse_json(r.stdout)
    var wire = _names(d.get("wire_tools"))
    var has_new = False
    for n in wire:
        assert_true(n != "run_check")
        if n == "toolsmith_new":
            has_new = True
    assert_true(has_new)
    assert_true(len(_names(d.get("subagents"))) == 0)
    assert_equal(d.get("qumo_mcp").at(0).as_string(), "qumo_describe")
    assert_false(d.get_bool("alphaxiv", True))

    # --model overrides the top agent's model; QUMO's MCP is "all" for qumo-operator.
    r = run_app(["run", "qumo-operator", "task", "--dry-run", "--model", "vendor/x"], env, "", 60000)
    var d2 = parse_json(r.stdout)
    assert_equal(d2.get_string("model", ""), "vendor/x")
    assert_equal(d2.get_string("qumo_mcp", ""), "all")

    # --dry-run reads no model: no key and no replay needed, and the module prefix is validated.
    r = run_app(["run", "qumo-operator", "task", "--dry-run", "--module", "nope"], env, "", 60000)
    assert_equal(r.exit_code, 2)
    remove_tree(dir)


def test_run_with_replay() raises:
    var dir = make_temp_dir("qumo-app-run-")
    var log_dir = dir + "/log"
    make_dirs(log_dir)
    var env = env_with(["OPENROUTER_API_KEY=" + SECRET, "QUMO_AGENT_REPLAY_LOG=" + log_dir])
    var r = run_app(["run", "qumo-operator", "Run a Bell state with seed 7", "--replay", REPLAY_BELL], env, "", 60000)
    assert_equal(r.exit_code, 0, r.stderr)
    assert_equal(r.stdout, BELL_ANSWER + "\n")
    assert_true(r.stderr.find("[qumo-agent] qumo-operator turn 1: qumo_validate_qasm {\"qasm\":\"OPENQASM 2.0;") >= 0, r.stderr)
    assert_true(r.stderr.find("turn 2: qumo_run_qasm") >= 0, r.stderr)
    assert_true(r.stderr.find("[qumo-agent] qumo-operator model=anthropic/claude-sonnet-5.5 model_calls=3 tool_calls=2 tool_errors=0 mcp_calls=2") >= 0, r.stderr)
    assert_true(r.stderr.find("prompt_tokens=600 completion_tokens=60") >= 0, r.stderr)
    assert_false(r.stderr.find(SECRET) >= 0)
    assert_false(r.stdout.find(SECRET) >= 0)

    # What the model saw: the system prompt, guided tools, and the real tool result.
    var reqs = read_text_lossy(log_dir + "/root.jsonl").split("\n")
    var n = 0
    for l in reqs:
        if String(l.strip()).byte_length() > 0:
            n += 1
    assert_equal(n, 3)
    var third = parse_json(String(reqs[2]))
    var msgs = third.get("messages")
    assert_true(msgs.at(0).get_string("content", "").find("Role: QUMO operator.") >= 0)
    var last = msgs.at(len(msgs) - 1)
    assert_equal(last.get_string("role", ""), "tool")
    assert_equal(last.get_string("tool_call_id", ""), "call_2")
    var content = last.get("content")      # MCP results are text-part arrays (B6)
    var result_text = content.at(0).get_string("text", "") if content.is_array() else last.get_string("content", "")
    var counts = parse_json(result_text).get("counts")
    for k in counts.keys():
        assert_true(k == "00" or k == "11", "unexpected key " + k)
    var first_tools = parse_json(String(reqs[0])).get("tools")
    var guided = False
    for i in range(len(first_tools)):
        var f = first_tools.at(i).get("function")
        if f.get_string("name", "") == "qumo_validate_qasm":
            guided = f.get_string("description", "").startswith("Use when you wrote or received OpenQASM")
    assert_true(guided)

    # --quiet silences the per-turn lines but keeps the statistics line.
    r = run_app(["run", "qumo-operator", "Run a Bell state with seed 7", "--replay", REPLAY_BELL, "--quiet"], env, "", 60000)
    assert_equal(r.exit_code, 0, r.stderr)
    assert_false(r.stderr.find(" turn 1: ") >= 0)
    assert_true(r.stderr.find("model_calls=3") >= 0)

    # --stats-json writes {"agent", "model", "stats"}; the task can come from stdin.
    var stats_path = dir + "/stats.json"
    r = run_app(["run", "qumo-operator", "-", "--replay", REPLAY_BELL, "--quiet", "--stats-json", stats_path],
                env, "Run a Bell state with seed 7\n", 60000)
    assert_equal(r.exit_code, 0, r.stderr)
    var sj = parse_json(read_text_lossy(stats_path))
    assert_equal(sj.get_string("agent", ""), "qumo-operator")
    assert_equal(sj.get_string("model", ""), "anthropic/claude-sonnet-5.5")
    assert_equal(sj.get("stats").get_int("model_calls", -1), 3)
    assert_equal(sj.get("stats").get_int("mcp_calls", -1), 2)

    # --module prefixes the task (cli.py:_task).
    var log2 = dir + "/log2"
    make_dirs(log2)
    var env2 = env_with(["QUMO_AGENT_REPLAY_LOG=" + log2])
    r = run_app(["run", "qumo-operator", "check it", "--replay", REPLAY_BELL, "--quiet", "--module", "ir"], env2, "", 60000)
    assert_equal(r.exit_code, 0, r.stderr)
    var req0 = parse_json(String(read_text_lossy(log2 + "/root.jsonl").split("\n")[0]))
    var user = req0.get("messages").at(1)
    assert_equal(user.get_string("role", ""), "user")
    assert_true(user.get_string("content", "").startswith("Module: ir ("), user.get_string("content", ""))
    assert_true(user.get_string("content", "").endswith("\n\ncheck it"))
    remove_tree(dir)


def test_trampolines() raises:
    """`__exec-in` and `__confine` are dispatched before anything else (they are how the
    runtime starts children without a shell)."""
    var none = List[String]()
    var dir = make_temp_dir("qumo-app-tramp-")
    var r = run_app(["__exec-in", dir, "/bin/pwd"], env_with(none), "", 20000)
    assert_equal(r.exit_code, 0, r.stderr)
    assert_equal(String(r.stdout.strip()), realpath_strict(dir))
    r = run_app(["__exec-in", dir + "/missing", "/bin/pwd"], env_with(none), "", 20000)
    assert_equal(r.exit_code, 127)
    assert_true(r.stderr.find("QA-E160") >= 0, r.stderr)

    r = run_app(["__confine", "not json", "/usr/bin/true"], env_with(none), "", 20000)
    assert_equal(r.exit_code, 125)
    assert_true(r.stderr.find("QA-E163") >= 0, r.stderr)
    var bin = realpath_strict(app_binary())
    if sandbox_kind(bin) != "none":
        var conf = Confinement(CONF_CHECK, List[String](), List[String](), String(), 0, 0)
        r = run_app(["__confine", conf.to_json(), "/usr/bin/true"], env_with(none), "", 20000)
        assert_equal(r.exit_code, 0, r.stderr)
        # A write outside write_dirs is refused by the OS.
        r = run_app(["__confine", conf.to_json(), "/usr/bin/touch", dir + "/nope"], env_with(none), "", 20000)
        assert_true(r.exit_code != 0)
        assert_false(exists(dir + "/nope"))
        var conf2 = Confinement(CONF_CHECK, [realpath_strict(dir)], List[String](), String(), 0, 0)
        r = run_app(["__confine", conf2.to_json(), "/usr/bin/touch", realpath_strict(dir) + "/yes"], env_with(none), "", 20000)
        assert_equal(r.exit_code, 0, r.stderr)
        assert_true(exists(dir + "/yes"))
    remove_tree(dir)


def main() raises:
    test_usage_and_exit_codes()
    test_trampolines()
    test_list()
    test_dry_run_all_agents()
    test_run_with_replay()
    print("test_app_cli OK")
