"""Tests parallel subagents (DESIGN.md X2, 2.10 `run_subagent_children`, B.7).

1. `run_subagent_children` against stand-in children (shell scripts; tests may start
   `/bin/sh`, the runtime never does): results in request order, at most `parallel` children at
   once (wall time), the child environment is exactly `harness_child_env` (a secret of the
   parent's shell never reaches it), a 1 MiB file dict with 200 KiB of stderr logs completes,
   and failures (exit without a line, malformed line, reported error) become `ok = False`
   results with the texts of 2.10.
2. The real binary: an orchestrator replay with two `task` calls in one turn runs two
   `qumo-agent subagent` children concurrently (overlapping windows in
   `QUMO_AGENT_REPLAY_LOG`), their results come back in call order, their files are merged,
   and `--parallel 1` gives the same answer inline. A failing child fails the run (QA-E120).
"""

from std.os import setenv, unsetenv
from std.testing import assert_equal, assert_true, assert_false

from qumo.ai.json import JsonValue, parse_json

from qumo_agent.app.options import AppOptions, default_options
from qumo_agent.app.subproc import run_subagent_children
from qumo_agent.app.testing import (
    app_env, run_app, replay_line, tool_call_json, write_replay,
)
from qumo_agent.harness.env import SubagentRequest
from qumo_agent.native.clock import monotonic_ms
from qumo_agent.native.fs import chmod, make_dirs, make_temp_dir, read_text_lossy, remove_tree, write_text


def request(id: String, type: String) -> SubagentRequest:
    return SubagentRequest("orchestrator", "m", type, "do " + id, id, ["orchestrator"], "{}", "", id)


def script(dir: String, name: String, body: String) raises -> String:
    var path = dir + "/" + name
    write_text(path, "#!/bin/sh\n" + body)
    chmod(path, 0o755)
    return path


def options(bin: String) -> AppOptions:
    var o = default_options(".", bin)
    o.api_key_env = "OPENROUTER_API_KEY"
    return o^


comptime ID_OF_REQUEST = "req=$(cat); id=$(printf '%s' \"$req\" | sed -n 's/.*\"tool_call_id\":\"\\([^\"]*\\)\".*/\\1/p'); "


def result_cmd(content_arg: String) -> String:
    """A shell command printing one ok result line whose content is `content_arg` (a shell word)."""
    return "printf '{\"schema\":\"qumo.subagent_result.v1\",\"ok\":true,\"content\":\"%s\",\"files_changed\":false,\"files\":null,\"sandbox\":\"\",\"stats\":{\"model_calls\":2},\"error\":\"\"}\\n' " + content_arg + "\n"


def test_children_contract() raises:
    var dir = make_temp_dir("qumo-app-children-")
    # Results in request order, each child sees its own request.
    var echo = script(dir, "echo.sh", ID_OF_REQUEST + result_cmd("\"ran-$id\""))
    var reqs = List[SubagentRequest]()
    reqs.append(request("a", "execution"))
    reqs.append(request("b", "execution"))
    reqs.append(request("c", "execution"))
    var res = run_subagent_children(options(echo), reqs^, 3)
    assert_equal(len(res), 3)
    assert_true(res[0].ok and res[1].ok and res[2].ok)
    assert_equal(res[0].content, "ran-a")
    assert_equal(res[1].content, "ran-b")
    assert_equal(res[2].content, "ran-c")
    assert_equal(res[0].stats.model_calls, 2)

    # At most `parallel` children at once: 4 children of 0.6 s each.
    var slow = script(dir, "slow.sh", ID_OF_REQUEST + "sleep 0.6; " + result_cmd("\"slow-$id\""))
    var four = List[SubagentRequest]()
    for id in ["1", "2", "3", "4"]:
        four.append(request(String(id), "execution"))
    var t0 = monotonic_ms()
    var res4 = run_subagent_children(options(slow), four.copy(), 4)
    var wall4 = monotonic_ms() - t0
    assert_true(wall4 < 1600, "4 children in parallel took " + String(wall4) + " ms")
    assert_equal(res4[3].content, "slow-4")
    t0 = monotonic_ms()
    var res2 = run_subagent_children(options(slow), four.copy(), 2)
    var wall2 = monotonic_ms() - t0
    assert_true(wall2 >= 1150, "parallel 2 over 4 children took " + String(wall2) + " ms")
    assert_equal(res2[0].content, "slow-1")
    assert_equal(res2[3].content, "slow-4")
    t0 = monotonic_ms()
    var res1 = run_subagent_children(options(slow), four.copy(), 1)
    var wall1 = monotonic_ms() - t0
    assert_true(wall1 >= 2300, "parallel 1 took " + String(wall1) + " ms")
    assert_equal(res1[2].content, "slow-3")
    remove_tree(dir)


def test_child_environment() raises:
    """The child gets harness_child_env: no FOO_SECRET, no AWS_ACCESS_KEY_ID, but the
    allow-listed QUMO_* names and the model key."""
    var dir = make_temp_dir("qumo-app-childenv-")
    var env_dump = script(dir, "env.sh", "cat > /dev/null; " + result_cmd("\"$(env | sort | tr '\\n' ' ')\""))
    _ = setenv("FOO_SECRET", "must-not-leak")
    _ = setenv("AWS_ACCESS_KEY_ID", "AKIAMUSTNOTLEAK")
    _ = setenv("AWS_SECRET_ACCESS_KEY", "must-not-leak-either")
    _ = setenv("OPENROUTER_API_KEY", "sk-or-test-key-1")
    _ = setenv("QUMO_AGENT_DETERMINISTIC", "1")
    _ = setenv("MY_APP_TOKEN", "tok-1")
    var reqs = List[SubagentRequest]()
    reqs.append(request("e", "execution"))
    var res = run_subagent_children(options(env_dump), reqs^, 2)
    _ = unsetenv("FOO_SECRET")
    _ = unsetenv("AWS_ACCESS_KEY_ID")
    _ = unsetenv("AWS_SECRET_ACCESS_KEY")
    _ = unsetenv("OPENROUTER_API_KEY")
    _ = unsetenv("QUMO_AGENT_DETERMINISTIC")
    _ = unsetenv("MY_APP_TOKEN")
    assert_true(res[0].ok, res[0].error)
    var dump = res[0].content
    assert_false(dump.find("FOO_SECRET") >= 0, dump)
    assert_false(dump.find("AWS_") >= 0, dump)
    assert_false(dump.find("MY_APP_TOKEN") >= 0, dump)
    assert_true(dump.find("OPENROUTER_API_KEY=") >= 0, dump)       # the endpoint's key is passed (K4)
    assert_true(dump.find("QUMO_AGENT_DETERMINISTIC=1") >= 0, dump)
    remove_tree(dir)


def big_result_body(stderr_bytes: Int) -> String:
    """A stand-in child: reads its request, writes `stderr_bytes` of logs on stderr, then a
    result whose file dict holds one 1 MiB file."""
    var logs = String()
    if stderr_bytes > 0:
        logs = "head -c " + String(stderr_bytes) + " /dev/zero | tr '\\000' 'e' >&2; "
    return "cat > /dev/null; " + logs \
        + "printf '{\"schema\":\"qumo.subagent_result.v1\",\"ok\":true,\"content\":\"big\",\"files_changed\":true,\"files\":{\"/big.txt\":{\"content\":\"'; " \
        + "head -c 1048576 /dev/zero | tr '\\000' 'x'; " \
        + "printf '\",\"encoding\":\"utf-8\",\"created_at\":\"t\",\"modified_at\":\"t\"}},\"sandbox\":\"/tmp/sb\",\"stats\":{},\"error\":\"\"}\\n'\n"


def test_big_result_and_chatty_stderr() raises:
    var dir = make_temp_dir("qumo-app-childbig-")
    var big = script(dir, "big.sh", big_result_body(0))
    var reqs = List[SubagentRequest]()
    reqs.append(request("big", "execution"))
    var res = run_subagent_children(options(big), reqs^, 4)
    assert_true(res[0].ok, res[0].error)
    assert_true(res[0].files_changed)
    assert_true(res[0].files_json.byte_length() > 1048576)
    assert_equal(res[0].sandbox, "/tmp/sb")
    remove_tree(dir)


def test_child_failures() raises:
    var dir = make_temp_dir("qumo-app-childfail-")
    var no_line = script(dir, "noline.sh", "cat > /dev/null; exit 3\n")
    var reqs = List[SubagentRequest]()
    reqs.append(request("x", "execution"))
    var res = run_subagent_children(options(no_line), reqs.copy(), 2)
    assert_false(res[0].ok)
    assert_equal(res[0].error, "exit 3")

    var malformed = script(dir, "malformed.sh", "cat > /dev/null; echo 'this is not json'\n")
    res = run_subagent_children(options(malformed), reqs.copy(), 2)
    assert_false(res[0].ok)
    assert_true(res[0].error.startswith("malformed result: this is not json"), res[0].error)

    var reported = script(dir, "reported.sh", "cat > /dev/null; echo '{\"ok\": false, \"error\": \"QA-E110: boom\"}'; exit 1\n")
    res = run_subagent_children(options(reported), reqs.copy(), 2)
    assert_false(res[0].ok)
    assert_equal(res[0].error, "QA-E110: boom")

    # A secret that a failing child echoes is redacted from the error text.
    _ = setenv("OPENROUTER_API_KEY", "sk-or-very-secret-key-99")
    var leaky = script(dir, "leaky.sh", "cat > /dev/null; echo '{\"ok\": false, \"error\": \"auth failed for sk-or-very-secret-key-99\"}'; exit 1\n")
    res = run_subagent_children(options(leaky), reqs.copy(), 2)
    _ = unsetenv("OPENROUTER_API_KEY")
    assert_false(res[0].error.find("sk-or-very-secret-key-99") >= 0, res[0].error)

    # A program that cannot start fails only its own request.
    res = run_subagent_children(options(dir + "/missing.sh"), reqs^, 2)
    assert_false(res[0].ok)
    assert_true(res[0].error.find("No such file or directory") >= 0, res[0].error)
    remove_tree(dir)


def _message_texts(body: JsonValue) raises -> List[String]:
    var out = List[String]()
    var msgs = body.get("messages")
    for i in range(len(msgs)):
        var m = msgs.at(i)
        if m.get_string("role", "") == "tool":
            var c = m.get("content")
            if c.is_array():
                out.append(c.at(0).get_string("text", ""))
            else:
                out.append(m.get_string("content", ""))
    return out^


def fanout_replay(path: String) raises:
    var lines = List[String]()
    var tasks = List[String]()
    tasks.append(tool_call_json("t1", "task", "{\"description\": \"alpha job\", \"subagent_type\": \"general-purpose\"}"))
    tasks.append(tool_call_json("t2", "task", "{\"description\": \"beta job\", \"subagent_type\": \"general-purpose\"}"))
    lines.append(replay_line("", tasks, "delegating"))
    var a = List[String]()
    a.append(tool_call_json("a1", "write_file", "{\"file_path\": \"/a.md\", \"content\": \"alpha\"}"))
    a.append(tool_call_json("a2", "qumo_describe", "{\"topic\": \"gates\"}"))
    lines.append(replay_line("t1", a, ""))
    lines.append(replay_line("t1", List[String](), "alpha done"))
    var b = List[String]()
    b.append(tool_call_json("b1", "write_file", "{\"file_path\": \"/b.md\", \"content\": \"beta\"}"))
    b.append(tool_call_json("b2", "qumo_describe", "{\"topic\": \"errors\"}"))
    lines.append(replay_line("t2", b, ""))
    lines.append(replay_line("t2", List[String](), "beta done"))
    var reads = List[String]()
    reads.append(tool_call_json("r1", "read_file", "{\"file_path\": \"/a.md\"}"))
    reads.append(tool_call_json("r2", "read_file", "{\"file_path\": \"/b.md\"}"))
    lines.append(replay_line("", reads, ""))
    lines.append(replay_line("", List[String](), "Both done: alpha done and beta done."))
    write_replay(path, lines)


def _window(path: String) raises -> Tuple[Int, Int]:
    """(start, end) in microseconds from a child-*.window file."""
    var rows = read_text_lossy(path).split("\n")
    var s = String(rows[0]).split(" ")
    var e = String(rows[1]).split(" ")
    return (Int(String(s[0])) * 1000000 + Int(String(s[1])), Int(String(e[0])) * 1000000 + Int(String(e[1])))


def test_real_children() raises:
    var dir = make_temp_dir("qumo-app-fanout-")
    var replay = dir + "/fanout.jsonl"
    fanout_replay(replay)
    var log_dir = dir + "/log"
    make_dirs(log_dir)
    var env = app_env(["QUMO_AGENT_REPLAY_LOG=" + log_dir, "FOO_SECRET=must-not-leak", "AWS_ACCESS_KEY_ID=AKIAMUSTNOTLEAK"])
    var r = run_app(["run", "orchestrator", "split the work", "--replay", replay], env, "", 120000)
    assert_equal(r.exit_code, 0, r.stderr)
    assert_equal(r.stdout, "Both done: alpha done and beta done.\n")
    assert_true(r.stderr.find("subagent_runs=2") >= 0, r.stderr)
    assert_true(r.stderr.find("model_calls=7") >= 0, r.stderr)        # 3 root + 2 + 2
    assert_true(r.stderr.find("mcp_calls=2") >= 0, r.stderr)          # one qumo_describe per child

    # The two children ran at the same time.
    var w1 = _window(log_dir + "/child-t1.window")
    var w2 = _window(log_dir + "/child-t2.window")
    assert_true(w2[0] < w1[1] and w1[0] < w2[1],
                "windows do not overlap: " + String(w1[0]) + "-" + String(w1[1]) + " / " + String(w2[0]) + "-" + String(w2[1]))

    # Results come back in call order, and the children's files were merged into the parent's.
    var rows = read_text_lossy(log_dir + "/root.jsonl").split("\n")
    var after_tasks = parse_json(String(rows[1]))
    var texts = _message_texts(after_tasks)
    assert_equal(len(texts), 2)
    assert_equal(texts[0], "alpha done")
    assert_equal(texts[1], "beta done")
    var after_reads = parse_json(String(rows[2]))
    var read_texts = _message_texts(after_reads)
    assert_equal(len(read_texts), 4)
    assert_true(read_texts[2].find("alpha") >= 0, read_texts[2])
    assert_true(read_texts[3].find("beta") >= 0, read_texts[3])
    # The children's own request logs exist, each with its two model calls.
    assert_equal(len(read_text_lossy(log_dir + "/t1.jsonl").split("\n")), 3)
    assert_equal(len(read_text_lossy(log_dir + "/t2.jsonl").split("\n")), 3)
    remove_tree(dir)

    # --parallel 1 runs the same replay inline: same answer, no child windows.
    dir = make_temp_dir("qumo-app-fanout-")
    replay = dir + "/fanout.jsonl"
    fanout_replay(replay)
    log_dir = dir + "/log"
    make_dirs(log_dir)
    env = app_env(["QUMO_AGENT_REPLAY_LOG=" + log_dir])
    r = run_app(["run", "orchestrator", "split the work", "--replay", replay, "--parallel", "1", "--quiet"], env, "", 120000)
    assert_equal(r.exit_code, 0, r.stderr)
    assert_equal(r.stdout, "Both done: alpha done and beta done.\n")
    assert_true(r.stderr.find("subagent_runs=2") >= 0, r.stderr)
    assert_false(read_text_lossy(log_dir + "/root.jsonl").find("child") >= 0)
    var entries = 0
    for n in ["child-t1.window", "child-t2.window"]:
        if read_text_lossy_or_empty(log_dir + "/" + String(n)).byte_length() > 0:
            entries += 1
    assert_equal(entries, 0)
    remove_tree(dir)


def read_text_lossy_or_empty(path: String) -> String:
    try:
        return read_text_lossy(path)
    except:
        return String()


def test_chatty_children_through_the_binary() raises:
    """Two stand-in children that each return a 1 MiB file dict and write 200 KiB of logs on
    stderr: the run completes, the logs reach the user's stderr, the files are merged."""
    var dir = make_temp_dir("qumo-app-chatty-")
    var fake = script(dir, "fake-qumo-agent", big_result_body(204800))
    var lines = List[String]()
    var tasks = List[String]()
    tasks.append(tool_call_json("t1", "task", "{\"description\": \"alpha\", \"subagent_type\": \"general-purpose\"}"))
    tasks.append(tool_call_json("t2", "task", "{\"description\": \"beta\", \"subagent_type\": \"general-purpose\"}"))
    lines.append(replay_line("", tasks, ""))
    var reads = List[String]()
    reads.append(tool_call_json("r1", "ls", "{\"path\": \"/\"}"))
    lines.append(replay_line("", reads, ""))
    lines.append(replay_line("", List[String](), "merged"))
    write_replay(dir + "/chatty.jsonl", lines)
    var log_dir = dir + "/log"
    make_dirs(log_dir)
    var env = app_env(["QUMO_AGENT_BIN=" + fake, "QUMO_AGENT_REPLAY_LOG=" + log_dir])
    var r = run_app(["run", "orchestrator", "x", "--replay", dir + "/chatty.jsonl", "--quiet"], env, "", 120000)
    assert_equal(r.exit_code, 0, String(r.stderr[byte=0:min(2000, r.stderr.byte_length())]))
    assert_equal(r.stdout, "merged\n")
    assert_true(r.stderr.byte_length() >= 409600, String(r.stderr.byte_length()))
    var rows = read_text_lossy(log_dir + "/root.jsonl").split("\n")
    var third = String(rows[2])
    assert_true(third.find("/big.txt") >= 0)        # the merged file is visible to `ls`
    remove_tree(dir)


def test_child_failure_fails_the_run() raises:
    var dir = make_temp_dir("qumo-app-fanfail-")
    var lines = List[String]()
    var tasks = List[String]()
    tasks.append(tool_call_json("t1", "task", "{\"description\": \"alpha\", \"subagent_type\": \"general-purpose\"}"))
    tasks.append(tool_call_json("t2", "task", "{\"description\": \"beta\", \"subagent_type\": \"general-purpose\"}"))
    lines.append(replay_line("", tasks, ""))
    lines.append(replay_line("t1", List[String](), "fine"))
    # t2 has no scripted response: its child fails with an exhausted replay (QA-E170).
    write_replay(dir + "/fail.jsonl", lines)
    var r = run_app(["run", "orchestrator", "x", "--replay", dir + "/fail.jsonl", "--quiet"], app_env(List[String]()), "", 120000)
    assert_equal(r.exit_code, 1)
    assert_true(r.stderr.find("QA-E120: subagent general-purpose failed: QA-E170: replay file exhausted at 't2'") >= 0, r.stderr)
    remove_tree(dir)


def main() raises:
    test_children_contract()
    test_child_environment()
    test_big_result_and_chatty_stderr()
    test_chatty_children_through_the_binary()
    test_child_failures()
    test_real_children()
    test_child_failure_fails_the_run()
    print("test_app_subagents OK")
