"""Tests for mcp/stdio_client.mojo against `build/qumo mcp` and small scripted servers:
handshake and tool calls, the server's environment (critic C2), message skipping and
server requests, JSON-RPC errors (redacted), timeouts, EOF, the close sequence
(critic C3) and stderr draining with STDIO_PIPE.

Needs `build/qumo` (`make cli`), or `QUMO_BIN`.
"""

from std.testing import assert_equal, assert_true, assert_false
from std.os import getenv, setenv, unsetenv, remove
from std.os.path import exists
from std.tempfile import mkdtemp
from std.time import perf_counter_ns

from qumo.ai.json import parse_json
from qumo_agent.native.libc import c_kill
from qumo_agent.native.spawn import STDIO_INHERIT, STDIO_PIPE, STDIO_NULL, mcp_server_env
from qumo_agent.native.sandbox import Confinement
from qumo_agent.mcp.protocol import McpTool
from qumo_agent.mcp.stdio_client import StdioServerParams, McpStdioSession, redact_mcp_text
from qumo_agent.mcp.results import call_result_to_outcome

comptime SCRIPTED = "agents/mojo/tests/fixtures/mcp/scripted_server.sh"
comptime MIB = 1024 * 1024
comptime SECRET_VALUE = "qa-m1-secret-value-1234567890"


def _qumo_bin() -> String:
    var b = getenv("QUMO_BIN", "")
    if b.byte_length() > 0:
        return b
    return "build/qumo"


def _params(program: String, var args: List[String], var env: List[String], timeout_ms: Int,
            max_bytes: Int, stderr_mode: Int) -> StdioServerParams:
    return StdioServerParams(program, args^, env^, "", timeout_ms, max_bytes, stderr_mode,
                             Optional[Confinement](None), "")


def _now_ms() -> Int:
    return Int(perf_counter_ns() // 1_000_000)


def _read(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def _alive(pid: Int) -> Bool:
    return c_kill(pid, 0) == 0


def test_qumo_server() raises:
    var s = McpStdioSession(_params(_qumo_bin(), ["mcp"], mcp_server_env(), 0, 16 * MIB, STDIO_INHERIT))
    assert_true(s.pid() > 0)
    s.initialize("mcp", "0.1.0")
    # QUMO's server answers 2025-06-18 to our 2025-11-25.
    assert_equal(s.protocol_version, "2025-06-18")
    assert_equal(s.server_name, "qumo")
    var tools = s.list_tools()
    assert_true(len(tools) >= 24, "expected at least 24 QUMO tools, got " + String(len(tools)))
    var found = False
    for t in tools:
        if t.name == "qumo_describe":
            found = True
            assert_true(t.description.byte_length() > 0)
            assert_true(parse_json(t.input_schema_json).is_object())
    assert_true(found)
    var r = s.call_tool("qumo_describe", "{}")
    assert_false(r.is_error)
    assert_equal(len(r.parts), 1)
    assert_true(r.parts[0].find("qumo.capabilities.v1") >= 0)
    assert_true(r.structured_json.byte_length() > 0)
    # A tool failure is a result, not a raise; the session keeps working.
    var bad = s.call_tool("qumo_run_qasm", "{\"qasm\": \"OPENQASM 3.0; qubit[1] q; nope q[0];\"}")
    assert_true(bad.is_error)
    var o = call_result_to_outcome(bad)
    assert_true(o.is_parts and o.is_error and len(o.parts) == 1)
    var again = s.call_tool("qumo_describe", "{}")
    assert_false(again.is_error)
    # Unknown tool: QUMO answers a JSON-RPC error.
    var raised = String()
    try:
        _ = s.call_tool("no_such_tool", "{}")
    except e:
        raised = String(e)
    assert_true(raised.startswith("QA-E130: MCP error -32602 on tools/call: Unknown tool: no_such_tool"), raised)
    # The session still works after a JSON-RPC error.
    assert_false(s.call_tool("qumo_describe", "{}").is_error)
    var pid = s.pid()
    var code = s.close()
    assert_equal(code, 0)
    assert_equal(s.close(), 0)
    assert_false(_alive(pid))


def test_server_env() raises:
    var dir = mkdtemp()
    var envfile = dir + "/env.txt"
    _ = setenv("QA_M1_TEST_API_KEY", SECRET_VALUE)
    _ = setenv("MY_SECRET", "x")
    _ = setenv("GITHUB_TOKEN", "ghp_test_m1")
    var args: List[String] = ["-c", "env > \"$0\"; exec \"$1\" mcp", envfile, _qumo_bin()]
    var s = McpStdioSession(_params("/bin/sh", args^, mcp_server_env(), 0, 16 * MIB, STDIO_INHERIT))
    s.initialize("mcp", "0.1.0")
    assert_true(len(s.list_tools()) >= 24)
    _ = s.close()
    var text = _read(envfile)
    var allowed: List[String] = ["HOME", "LOGNAME", "PATH", "SHELL", "TERM", "USER", "PWD", "SHLVL", "_", "OLDPWD"]
    for line in text.splitlines():
        var l = String(line)
        if l.byte_length() == 0:
            continue
        var eq = l.find("=")
        assert_true(eq > 0, "unexpected env line: " + l)
        var name = String(l[byte=0:eq])
        var ok = False
        for a in allowed:
            if a == name:
                ok = True
        assert_true(ok, "the MCP server saw " + name)
    assert_true(text.find(SECRET_VALUE) < 0)
    assert_true(text.find("ghp_test_m1") < 0)
    remove(envfile)
    _ = unsetenv("MY_SECRET")
    _ = unsetenv("GITHUB_TOKEN")


def test_scripted() raises:
    var dir = mkdtemp()
    var log = dir + "/log.jsonl"
    _ = setenv("QA_M1_TEST_API_KEY", SECRET_VALUE)
    var env = mcp_server_env()
    env.append("QA_M1_TEST_API_KEY=" + SECRET_VALUE)
    var args: List[String] = [SCRIPTED, log]
    var s = McpStdioSession(_params("/bin/sh", args^, env^, 10000, 16 * MIB, STDIO_INHERIT))
    s.initialize("mcp", "0.1.0")
    assert_equal(s.protocol_version, "2025-03-26")
    assert_equal(s.server_name, "scripted")
    var tools = s.list_tools()
    assert_equal(len(tools), 2)
    assert_equal(tools[0].name, "one")
    assert_equal(tools[1].name, "two")
    assert_equal(tools[1].description, "")
    var r = s.call_tool("one", "{}")
    assert_equal(len(r.parts), 2)
    var err = String()
    try:
        _ = s.call_tool("two", "{\"x\": \"y\"}")
    except e:
        err = String(e)
    assert_true(err.startswith("QA-E130: MCP error -32602 on tools/call: bad key "), err)
    assert_true(err.find(SECRET_VALUE) < 0, "secret not redacted: " + err)
    var eof = String()
    try:
        _ = s.call_tool("one", "{}")
    except e:
        eof = String(e)
    assert_equal(eof, "QA-E130: MCP server closed the connection while waiting for tools/call")
    # A broken session refuses further requests without touching the pipe.
    var after = String()
    try:
        _ = s.call_tool("one", "{}")
    except e:
        after = String(e)
    assert_equal(after, "QA-E130: MCP server closed the connection while waiting for tools/call")
    assert_equal(s.close(), 0)
    var lines = _read(log).splitlines()
    assert_equal(len(lines), 8)
    assert_equal(String(lines[0]), "{\"jsonrpc\":\"2.0\",\"id\":0,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2025-11-25\",\"capabilities\":{},\"clientInfo\":{\"name\":\"mcp\",\"version\":\"0.1.0\"}}}")
    assert_equal(String(lines[1]), "{\"jsonrpc\":\"2.0\",\"id\":\"srv-1\",\"error\":{\"code\":-32601,\"message\":\"Method not found\"}}")
    assert_equal(String(lines[2]), "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}")
    assert_equal(String(lines[3]), "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\"}")
    assert_equal(String(lines[4]), "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/list\",\"params\":{\"cursor\":\"p2\"}}")
    assert_equal(String(lines[5]), "{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"tools/call\",\"params\":{\"name\":\"one\",\"arguments\":{}}}")
    assert_equal(String(lines[6]), "{\"jsonrpc\":\"2.0\",\"id\":4,\"method\":\"tools/call\",\"params\":{\"name\":\"two\",\"arguments\":{\"x\": \"y\"}}}")
    assert_equal(String(lines[7]), "{\"jsonrpc\":\"2.0\",\"id\":5,\"method\":\"tools/call\",\"params\":{\"name\":\"one\",\"arguments\":{}}}")
    remove(log)
    # redact_mcp_text hides any secret-looking variable of this process.
    assert_true(redact_mcp_text("v=" + SECRET_VALUE).find(SECRET_VALUE) < 0)
    _ = unsetenv("QA_M1_TEST_API_KEY")


def test_unsupported_version() raises:
    var args: List[String] = ["-c", "read -r l; printf '%s\\n' '{\"jsonrpc\":\"2.0\",\"id\":0,\"result\":{\"protocolVersion\":\"1999-01-01\",\"capabilities\":{},\"serverInfo\":{\"name\":\"old\",\"version\":\"1\"}}}'; read -r l"]
    var s = McpStdioSession(_params("/bin/sh", args^, mcp_server_env(), 5000, 16 * MIB, STDIO_INHERIT))
    var err = String()
    try:
        s.initialize("mcp", "0.1.0")
    except e:
        err = String(e)
    assert_equal(err, "QA-E130: Unsupported protocol version from the server: 1999-01-01")
    _ = s.close()


def test_line_too_long() raises:
    var s = McpStdioSession(_params(_qumo_bin(), ["mcp"], mcp_server_env(), 0, 64, STDIO_INHERIT))
    var err = String()
    try:
        s.initialize("mcp", "0.1.0")
    except e:
        err = String(e)
    assert_true(err.startswith("QA-E161: "), err)
    var pid = s.pid()
    _ = s.close()
    assert_false(_alive(pid))


def test_timeout_kills() raises:
    var args: List[String] = ["30"]
    var s = McpStdioSession(_params("/bin/sleep", args^, mcp_server_env(), 1000, 16 * MIB, STDIO_INHERIT))
    var pid = s.pid()
    var t0 = _now_ms()
    var err = String()
    try:
        s.initialize("mcp", "0.1.0")
    except e:
        err = String(e)
    var dt = _now_ms() - t0
    assert_equal(err, "QA-E130: MCP server did not answer initialize within 1000 ms")
    assert_true(dt >= 900 and dt < 3000, "timeout took " + String(dt) + " ms")
    assert_false(_alive(pid))
    assert_equal(s.close(), -9)


def test_close_escalates() raises:
    # /bin/sleep ignores EOF on stdin: close() waits 2 s, then SIGTERM ends it.
    var args: List[String] = ["30"]
    var s = McpStdioSession(_params("/bin/sleep", args^, mcp_server_env(), 0, 16 * MIB, STDIO_INHERIT))
    var pid = s.pid()
    var t0 = _now_ms()
    var code = s.close()
    var dt = _now_ms() - t0
    assert_equal(code, -15)
    assert_true(dt >= 1900 and dt < 4000, "close took " + String(dt) + " ms")
    assert_false(_alive(pid))


def test_stderr_pipe() raises:
    # 1 MiB on stderr before the server starts answering: never blocks the session.
    var args: List[String] = ["-c", "head -c 1048576 /dev/zero | tr '\\000' x >&2; printf 'last-line\\n' >&2; exec \"$0\" mcp", _qumo_bin()]
    var s = McpStdioSession(_params("/bin/sh", args^, mcp_server_env(), 20000, 16 * MIB, STDIO_PIPE))
    s.initialize("mcp", "0.1.0")
    assert_equal(s.server_name, "qumo")
    var tail = s.stderr_tail()
    assert_true(tail.byte_length() > 0)
    assert_true(tail.byte_length() <= 64 * 1024)
    assert_true(tail.find("last-line") >= 0)
    assert_false(s.call_tool("qumo_describe", "{}").is_error)
    assert_equal(s.close(), 0)
    # Without STDIO_PIPE there is no tail.
    var q = McpStdioSession(_params(_qumo_bin(), ["mcp"], mcp_server_env(), 0, 16 * MIB, STDIO_NULL))
    q.initialize("mcp", "0.1.0")
    assert_equal(q.stderr_tail(), "")
    _ = q.close()


def main() raises:
    if not exists(_qumo_bin()):
        raise Error("test_mcp_stdio needs " + _qumo_bin() + ": run make cli, or set QUMO_BIN")
    test_qumo_server()
    test_server_env()
    test_scripted()
    test_unsupported_version()
    test_line_too_long()
    test_timeout_kills()
    test_close_escalates()
    test_stderr_pipe()
    print("test_mcp_stdio OK")
