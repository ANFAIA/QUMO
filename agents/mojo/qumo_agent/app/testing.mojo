"""Helpers of the application's tests (DESIGN.md 8.2, component A): the `qumo-agent` test
binary, running it with a controlled environment, and builders for replay files.

The test binary is `agents/mojo/tests/fixtures/app/main_app.mojo` (it calls
`app.commands.main_entry`, like `main.mojo`), built on demand into
`build/test-fixtures/qumo-agent-app` and rebuilt when any source is newer. Run the tests
from the repository root.
"""

from std.os import stat
from std.os.path import exists

from qumo.ai.json import JsonWriter

from qumo_agent.native.fs import make_dirs, walk, write_text
from qumo_agent.native.libc import CStr, c_rename
from qumo_agent.native.rand import uuid4_hex
from qumo_agent.native.spawn import (
    CaptureResult, SpawnSpec, run_capture, minimal_child_env, environ_list, STDIO_NULL, STDIO_PIPE,
)

comptime APP_MAIN = "agents/mojo/tests/fixtures/app/main_app.mojo"
comptime APP_BIN = "build/test-fixtures/qumo-agent-app"
comptime REPLAY_BELL = "agents/mojo/tests/fixtures/app/replay_bell.jsonl"
comptime BELL_ANSWER = "Bell state: only 00 and 11 were measured, as theory predicts."


def _mtime(path: String) raises -> Int:
    var st = stat(path)
    return Int(st.st_mtimespec.tv_sec) * 1000000000 + Int(st.st_mtimespec.tv_nsec)


def _newest_source() raises -> Int:
    var newest = _mtime(APP_MAIN)
    var dirs: List[String] = ["agents/mojo/qumo_agent", "qumo/ai"]
    for d in dirs:
        for p in walk(d, False):
            if p.endswith(".mojo"):
                newest = max(newest, _mtime(p))
    return newest


def app_binary() raises -> String:
    """Builds the test `qumo-agent` unless it is newer than every source; returns its path."""
    if exists(APP_BIN) and _mtime(APP_BIN) > _newest_source():
        return APP_BIN
    make_dirs("build/test-fixtures")
    var tmp = APP_BIN + ".tmp-" + uuid4_hex()
    var argv: List[String] = [".venv/bin/mojo", "build", "-I", ".", "-I", "agents/mojo", "-o", tmp, APP_MAIN]
    var r = run_capture(SpawnSpec(".venv/bin/mojo", argv^, environ_list(), "", STDIO_NULL, STDIO_PIPE, STDIO_PIPE, False),
                        900000, 1 << 16, "")
    if r.exit_code != 0:
        raise Error("building the test qumo-agent failed: " + r.stdout + r.stderr)
    var a = CStr(tmp)
    var b = CStr(APP_BIN)
    _ = c_rename(a.addr(), b.addr())
    a.keep()
    b.keep()
    return APP_BIN


def app_env(extra: List[String]) -> List[String]:
    """The environment tests give the binary: the minimal environment, never a model key
    or any other secret of the developer's shell, plus `extra` ("NAME=value")."""
    var env = minimal_child_env()
    for e in extra:
        env.append(e)
    return env^


def run_app(args: List[String], env: List[String], stdin_data: String, timeout_ms: Int) raises -> CaptureResult:
    """Runs the test binary with exactly `env`; stdout and stderr are captured."""
    var bin = app_binary()
    var argv = List[String]()
    argv.append(bin)
    for a in args:
        argv.append(a)
    return run_capture(SpawnSpec(bin, argv^, env.copy(), "", STDIO_PIPE, STDIO_PIPE, STDIO_PIPE, True),
                       timeout_ms, 8 << 20, stdin_data)


# ---------------------------------------------------------------------------
# replay file builders
# ---------------------------------------------------------------------------
def tool_call_json(id: String, name: String, raw_args: String) -> String:
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


def response_json(calls: List[String], content: String) -> String:
    """A chat-completions response with `calls` (tool call JSON texts) and `content`."""
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
    w.key("usage")
    w.begin_object()
    w.kv_int("prompt_tokens", 10)
    w.kv_int("completion_tokens", 2)
    w.end_object()
    w.end_object()
    return w.finish()


def replay_line(at: String, calls: List[String], content: String) -> String:
    """One replay line for the delegation path `at` ("" = the root agent)."""
    var w = JsonWriter()
    w.begin_object()
    w.kv_string("at", at)
    w.kv_raw("response", response_json(calls, content))
    w.end_object()
    return w.finish()


def write_replay(path: String, lines: List[String]) raises:
    write_text(path, String("\n").join(lines) + "\n")
