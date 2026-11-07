"""Tests for mcp/http_client.mojo (streamable HTTP MCP) against tests/support/http_fixture_server
over plain HTTP: JSON and SSE answers (priming event, notification, server request, other
ids and event types before the response, events split across chunks, CRLF), the
`mcp-session-id` (taken from initialize only) and `mcp-protocol-version` headers checked in
the request log, 202 for `notifications/initialized`, 404 giving `Session terminated`,
unsupported version refused, `DELETE` on close (and none without a session id), JSON-RPC
errors, HTTP errors with the key echoed (redacted), 202/content-type/stream-end/id errors,
same-origin redirect followed and cross-origin one refused, SSE read timeout, message
caps, and construction checks (URL, key variable, plain http with a bearer token).

Needs `.venv/bin/mojo` to build the fixture server on first use.
"""

from std.os import stat, getenv, setenv, unsetenv
from std.os.path import exists
from std.testing import assert_equal, assert_true, assert_false

from qumo.ai.json import JsonValue, JsonWriter, parse_json
from qumo_agent.native.libc import CStr, c_rename
from qumo_agent.native.spawn import (
    SpawnSpec, Child, spawn, run_capture, minimal_child_env, environ_list, ignore_sigpipe,
    STDIO_NULL, STDIO_PIPE, STDIO_INHERIT,
)
from qumo_agent.native.fs import make_temp_dir, make_dirs, write_text, read_text_lossy, list_dir
from qumo_agent.native.clock import monotonic_ms, sleep_ms
from qumo_agent.native.rand import uuid4_hex
from qumo_agent.net.socket import TcpListener
from qumo_agent.net.http import HttpClient
from qumo_agent.mcp.protocol import initialize_params_json, rpc_request_json
from qumo_agent.mcp.results import call_result_to_outcome
from qumo_agent.mcp.http_client import McpHttpSession


# ---------------------------------------------------------------------------
# http_fixture_server helper (each test that needs the server carries a copy)
# ---------------------------------------------------------------------------
comptime _FIXTURE_SRC = "agents/mojo/tests/support/http_fixture_server.mojo"
comptime _FIXTURE_BIN = "build/test-fixtures/http_fixture_server"


def _mtime(path: String) raises -> Int:
    var st = stat(path)
    return Int(st.st_mtimespec.tv_sec) * 1000000000 + Int(st.st_mtimespec.tv_nsec)


def _newest_source() raises -> Int:
    var newest = _mtime(_FIXTURE_SRC)
    newest = max(newest, _mtime("qumo/ai/json.mojo"))
    var dirs: List[String] = ["agents/mojo/qumo_agent/net", "agents/mojo/qumo_agent/native", "agents/mojo/qumo_agent/core"]
    for d in dirs:
        for name in list_dir(d):
            if name.endswith(".mojo"):
                newest = max(newest, _mtime(d + "/" + name))
    return newest


def fixture_binary() raises -> String:
    """Builds build/test-fixtures/http_fixture_server unless it is newer than its sources."""
    if exists(_FIXTURE_BIN) and _mtime(_FIXTURE_BIN) > _newest_source():
        return _FIXTURE_BIN
    make_dirs("build/test-fixtures")
    var tmp = _FIXTURE_BIN + ".tmp-" + uuid4_hex()
    var argv: List[String] = [".venv/bin/mojo", "build", "-I", ".", "-I", "agents/mojo", "-o", tmp, _FIXTURE_SRC]
    var r = run_capture(SpawnSpec(".venv/bin/mojo", argv^, environ_list(), "", STDIO_NULL, STDIO_PIPE, STDIO_PIPE, False),
                        600000, 1 << 16, "")
    if r.exit_code != 0:
        raise Error("building the fixture server failed: " + r.stdout + r.stderr)
    var a = CStr(tmp)
    var b = CStr(_FIXTURE_BIN)
    _ = c_rename(a.addr(), b.addr())
    a.keep()
    b.keep()
    return _FIXTURE_BIN


struct Fixture(Movable):
    """One running http_fixture_server with its script, port file and request log."""

    var child: Child
    var dir: String
    var port: Int

    def __init__(out self, script: String) raises:
        var bin = fixture_binary()
        self.dir = make_temp_dir("qumo-mcp-http-fixture-")
        write_text(self.dir + "/script.json", script)
        var argv: List[String] = [bin, "--script", self.dir + "/script.json", "--port-file", self.dir + "/port",
                                  "--log", self.dir + "/log.jsonl"]
        self.child = spawn(SpawnSpec(bin, argv^, minimal_child_env(), "", STDIO_NULL, STDIO_INHERIT, STDIO_INHERIT, True))
        var deadline = monotonic_ms() + 30000
        while not exists(self.dir + "/port"):
            if monotonic_ms() > deadline:
                raise Error("the fixture server did not write its port")
            sleep_ms(10)
        self.port = Int(String(read_text_lossy(self.dir + "/port").strip()))

    def url(self, path: String) -> String:
        return "http://127.0.0.1:" + String(self.port) + path

    def log(self) raises -> List[JsonValue]:
        var out = List[JsonValue]()
        if not exists(self.dir + "/log.jsonl"):
            return out^
        for line in read_text_lossy(self.dir + "/log.jsonl").split("\n"):
            if String(line).byte_length() > 0:
                out.append(parse_json(String(line)))
        return out^

    def finish(mut self) raises -> Int:
        """Waits up to 5 s for the server to exit (script consumed), else stops it."""
        var deadline = monotonic_ms() + 5000
        while monotonic_ms() < deadline:
            var code = self.child.poll_exit()
            if code:
                return code.value()
            sleep_ms(10)
        return self.child.terminate(200)


# ---------------------------------------------------------------------------
# script building
# ---------------------------------------------------------------------------
comptime KEY_ENV = "QUMO_TEST_MCP_HTTP_KEY"
comptime KEY = "amx-0123456789abcdef-FAKE-KEY"
comptime SESSION = "sess-abc123"


def _entry(method: String, path: String, status: Int, reason: String, headers: List[String], body: String,
           framing: String, chunks: List[String], delay_ms: Int) -> String:
    var w = JsonWriter()
    w.begin_object()
    w.key("match")
    w.begin_object()
    w.kv_string("method", method)
    w.kv_string("path", path)
    w.end_object()
    w.kv_int("status", status)
    w.kv_string("reason", reason)
    w.key("headers")
    w.begin_object()
    var i = 0
    while i + 1 < len(headers):
        w.kv_string(headers[i], headers[i + 1])
        i += 2
    w.end_object()
    w.kv_string("body", body)
    w.kv_string("framing", framing)
    w.key("chunks")
    w.begin_array()
    for c in chunks:
        w.string(c)
    w.end_array()
    w.kv_int("delay_ms", delay_ms)
    w.end_object()
    return w.finish()


def _json(body: String, extra: List[String] = List[String]()) -> String:
    var h: List[String] = ["content-type", "application/json; charset=utf-8"]
    h.extend(extra.copy())
    return _entry("POST", "/mcp", 200, "OK", h, body, "content-length", List[String](), 0)


def _status(status: Int, reason: String, body: String, extra: List[String] = List[String]()) -> String:
    var h: List[String] = ["content-type", "text/plain"]
    h.extend(extra.copy())
    return _entry("POST", "/mcp", status, reason, h, body, "content-length", List[String](), 0)


def _sse(chunks: List[String], delay_ms: Int = 0) -> String:
    var h: List[String] = ["content-type", "text/event-stream"]
    return _entry("POST", "/mcp", 200, "OK", h, "", "chunked", chunks, delay_ms)


def _result(id: Int, result_json: String) -> String:
    return '{"jsonrpc":"2.0","id":' + String(id) + ',"result":' + result_json + "}"


def _text_result(id: Int, text: String) -> String:
    return _result(id, '{"content":[{"type":"text","text":"' + text + '"}],"isError":false}')


def _init_result(version: String) -> String:
    return _result(0, '{"protocolVersion":"' + version + '","capabilities":{"tools":{}},"serverInfo":{"name":"fake-alphaxiv","version":"1"}}')


def _script(entries: List[String]) -> String:
    var s = String('{"responses": [')
    for i in range(len(entries)):
        if i > 0:
            s += ",\n"
        s += entries[i]
    return s + "]}"


def _call_err(mut s: McpHttpSession) -> String:
    try:
        _ = s.call_tool("x", "{}")
        return "NO ERROR"
    except e:
        return String(e)


def _session(fx: Fixture, bearer: String = KEY_ENV) raises -> McpHttpSession:
    return McpHttpSession(HttpClient(None), fx.url("/mcp"), bearer)


def _hdr(entry: JsonValue, name: String) raises -> String:
    return entry.get("headers").get_string(name, "<absent>")


# ---------------------------------------------------------------------------
# tests that need no server
# ---------------------------------------------------------------------------
def _config_error(url: String, bearer: String) -> String:
    try:
        _ = McpHttpSession(HttpClient(None), url, bearer)
        return "NO ERROR"
    except e:
        return String(e)


def test_construction() raises:
    _ = setenv(KEY_ENV, KEY, True)
    assert_true(_config_error("not a url", KEY_ENV).startswith("QA-E140: invalid MCP URL: "))
    assert_equal(_config_error("https://api.example.test/mcp", "1BAD"),
                 "QA-E140: '1BAD' is not a valid environment variable name")
    _ = unsetenv("QUMO_TEST_MCP_HTTP_UNSET")
    assert_equal(_config_error("https://api.example.test/mcp", "QUMO_TEST_MCP_HTTP_UNSET"),
                 "QA-E140: set QUMO_TEST_MCP_HTTP_UNSET to use the MCP server https://api.example.test")
    # A bearer token is never sent in clear text beyond loopback.
    var refused = _config_error("http://api.example.test/mcp", KEY_ENV)
    assert_true(refused.startswith("QA-E140: refusing to send " + KEY_ENV + " to http://api.example.test over plain http"), refused)
    assert_true(refused.find(KEY) < 0)
    assert_equal(_config_error("http://api.example.test/mcp", ""), "NO ERROR")
    assert_equal(_config_error("http://127.0.0.1:9/mcp", KEY_ENV), "NO ERROR")
    assert_equal(_config_error("http://localhost/mcp", KEY_ENV), "NO ERROR")
    assert_equal(_config_error("http://[::1]:9/mcp", KEY_ENV), "NO ERROR")
    assert_equal(_config_error("https://api.example.test/mcp", KEY_ENV), "NO ERROR")


def test_refused_connection() raises:
    _ = setenv(KEY_ENV, KEY, True)
    var free = TcpListener("127.0.0.1", 0)
    var dead_port = free.port()
    _ = free^
    var s = McpHttpSession(HttpClient(None), "http://127.0.0.1:" + String(dead_port) + "/mcp", KEY_ENV)
    var err = _call_err(s)
    assert_true(err.startswith("QA-E104: 127.0.0.1:" + String(dead_port) + ": "), err)
    assert_true(err.find(KEY) < 0)
    s.close()      # no session id: nothing is sent, nothing raises
    var after = _call_err(s)
    assert_equal(after, "QA-E130: MCP session is closed (tools/call)")


# ---------------------------------------------------------------------------
# the main exchange: JSON, SSE, headers, pagination, errors in one session
# ---------------------------------------------------------------------------
def test_session_flow() raises:
    _ = setenv(KEY_ENV, KEY, True)
    var tools1 = '{"tools":[{"name":"search","title":"Search","description":"find papers","inputSchema":{"type":"object","properties":{"q":{"type":"string"}}}},{"name":"read","inputSchema":{"type":"object"}}],"nextCursor":"c2"}'
    var tools2 = '{"tools":[{"name":"full_text","description":"d"}]}'
    var sse_chunks: List[String] = [
        "id: 1\ndata:\n\nretry: 1000\n\n",                                                       # priming event, retry-only block
        'event: message\r\ndata: {"jsonrpc":"2.0","method":"notifications/progress","params":{"progress":1}}\r\n\r\n',
        'data: {"jsonrpc":"2.0","id":7,"method":"ping"}\n\nevent: x-other\ndata: ' + _result(3, '{"content":[]}') + "\n\n",
        "data: " + _result(99, "{}") + "\n\n: comment\n\nda",
        "ta: " + _result(3, '{"content":[{"type":"text","text":"paper A"},{"type":"text","text":"paper B"}],"isError":false}') + "\n\n",
    ]
    var entries: List[String] = [
        _json(_init_result("2025-06-18"), ["mcp-session-id", SESSION]),                          # 0 initialize
        _entry("POST", "/mcp", 202, "Accepted", List[String](), "", "content-length", List[String](), 0),   # 1 initialized
        _json(_result(1, tools1), ["mcp-session-id", "other-id"]),                               # 2 tools/list (id ignored)
        _json(_result(2, tools2)),                                                               # 3 tools/list page 2
        _sse(sse_chunks),                                                                        # 4 tools/call over SSE
        _json(_result(4, '{"content":[{"type":"text","text":"bad input"}],"isError":true}')),   # 5 tools/call isError
        _json('{"jsonrpc":"2.0","id":5,"error":{"code":-32602,"message":"Unknown tool: nope"}}'),   # 6 JSON-RPC error
        _status(404, "Not Found", "{}"),                                                         # 7 session gone
        _entry("DELETE", "/mcp", 200, "OK", List[String](), "", "content-length", List[String](), 0),       # 8 close
    ]
    var fx = Fixture(_script(entries))
    var s = _session(fx)
    assert_equal(s.session_id, "")
    s.initialize()
    assert_equal(s.session_id, SESSION)
    assert_equal(s.protocol_version, "2025-06-18")
    assert_equal(s.server_name, "fake-alphaxiv")

    var tools = s.list_tools()
    assert_equal(len(tools), 3)
    assert_equal(tools[0].name, "search")
    assert_equal(tools[0].title, "Search")
    assert_equal(tools[0].description, "find papers")
    assert_true(tools[0].input_schema_json.find('"q"') >= 0)
    assert_equal(tools[1].name, "read")
    assert_equal(tools[2].name, "full_text")
    assert_equal(tools[2].input_schema_json, '{"type":"object"}')
    assert_equal(s.session_id, SESSION)         # an mcp-session-id on a later answer is ignored

    var r = s.call_tool("search", '{"q":"qubits"}')
    assert_false(r.is_error)
    assert_equal(len(r.parts), 2)
    assert_equal(r.parts[0], "paper A")
    assert_equal(r.parts[1], "paper B")
    var o = call_result_to_outcome(r)
    assert_true(o.is_parts and not o.is_error and len(o.parts) == 2)

    var bad = s.call_tool("search", "{}")
    assert_true(bad.is_error)
    assert_equal(bad.parts[0], "bad input")

    var rpc = String()
    try:
        _ = s.call_tool("nope", "{}")
    except e:
        rpc = String(e)
    assert_equal(rpc, "QA-E130: MCP error -32602 on tools/call: Unknown tool: nope")

    var gone = _call_err(s)
    assert_equal(gone, "QA-E130: MCP error 32600 on tools/call: Session terminated")
    assert_equal(s.requests(), 8)
    s.close()
    s.close()           # a second close sends nothing
    assert_equal(s.requests(), 9)
    assert_equal(fx.finish(), 0)

    # What the server saw.
    var log = fx.log()
    assert_equal(len(log), 9)
    for i in range(8):
        assert_equal(log[i].get_string("path", ""), "/mcp")
        assert_equal(log[i].get_string("method", ""), "POST")
        var h = log[i].get("headers")
        assert_equal(_hdr(log[i], "accept"), "application/json, text/event-stream")
        assert_equal(_hdr(log[i], "content-type"), "application/json")
        assert_equal(_hdr(log[i], "authorization"), "Bearer " + KEY)
        assert_equal(_hdr(log[i], "user-agent"), "qumo-agent/0.2 (Mojo)")
        assert_equal(_hdr(log[i], "connection"), "close")
        if i == 0:
            assert_false(h.has("mcp-session-id"))
            assert_false(h.has("mcp-protocol-version"))
        else:
            assert_equal(_hdr(log[i], "mcp-session-id"), SESSION)
            assert_equal(_hdr(log[i], "mcp-protocol-version"), "2025-06-18")
    assert_equal(log[0].get_string("body", ""), rpc_request_json(0, "initialize", initialize_params_json("mcp", "0.1.0")))
    assert_equal(log[1].get_string("body", ""), '{"jsonrpc":"2.0","method":"notifications/initialized"}')
    assert_equal(log[2].get_string("body", ""), '{"jsonrpc":"2.0","id":1,"method":"tools/list"}')
    assert_equal(log[3].get_string("body", ""), '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{"cursor":"c2"}}')
    assert_equal(log[4].get_string("body", ""), '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"search","arguments":{"q":"qubits"}}}')
    assert_equal(log[7].get_string("body", ""), '{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"x","arguments":{}}}')
    # close: DELETE with the session id and the protocol version, no body.
    assert_equal(log[8].get_string("method", ""), "DELETE")
    assert_equal(_hdr(log[8], "mcp-session-id"), SESSION)
    assert_equal(_hdr(log[8], "mcp-protocol-version"), "2025-06-18")
    assert_equal(_hdr(log[8], "authorization"), "Bearer " + KEY)
    assert_equal(log[8].get_string("body", ""), "")
    assert_false(log[8].get("headers").has("content-length"))


# ---------------------------------------------------------------------------
# failures, redirects, timeouts, caps, no session id, no bearer
# ---------------------------------------------------------------------------
def test_failures() raises:
    _ = setenv(KEY_ENV, KEY, True)
    var long_text = String()
    for _ in range(100):
        long_text += "x"
    var oversize_json = _text_result(0, long_text)
    var oversize_sse = _sse(["data: " + _text_result(0, long_text) + "\n\n"])
    var entries: List[String] = [
        _json(_init_result("1999-01-01"), ["mcp-session-id", "sess-bad"]),                       # 0 unsupported version
        _entry("DELETE", "/mcp", 200, "OK", List[String](), "", "content-length", List[String](), 0),       # 1 its close
        _json('{"jsonrpc":"2.0","id":0,"error":{"code":-32602,"message":"Unknown tool: nope"}}'),    # 2
        _status(500, "Internal Server Error", '{"error":{"message":"boom with key ' + KEY + '"}}'),   # 3
        _status(401, "Unauthorized", "denied for " + KEY),                                           # 4
        _entry("POST", "/mcp", 202, "Accepted", List[String](), "", "content-length", List[String](), 0),   # 5
        _status(200, "OK", "plain text"),                                                            # 6
        _sse(['data: ' + _result(41, "{}") + "\n\n", "event: ping\ndata: {}\n\n"]),                  # 7 stream ends first
        _json(_result(41, "{}")),                                                                    # 8 wrong id
        _json("this is not json"),                                                                   # 9
        _entry("POST", "/mcp", 307, "Temporary Redirect", ["location", "/mcp-moved"], "", "content-length", List[String](), 0),  # 10
        _entry("POST", "/mcp-moved", 200, "OK", ["content-type", "application/json"], _text_result(0, "moved ok"), "content-length", List[String](), 0),  # 11
        _entry("POST", "/mcp", 307, "Temporary Redirect", ["location", "http://localhost:1/x"], "", "content-length", List[String](), 0),   # 12
        _sse(['data: ' + _text_result(0, "late") + "\n\n"], 800),                                    # 13 stalled stream
        _json(oversize_json),                                                                        # 14
        oversize_sse,                                                                                # 15
        _json(_init_result("2024-11-05")),                                                           # 16 no session id
        _entry("POST", "/mcp", 202, "Accepted", List[String](), "", "content-length", List[String](), 0),   # 17
    ]
    var fx = Fixture(_script(entries))

    # Unsupported protocol version; the session id is still released with DELETE on close.
    var s1 = _session(fx)
    var v = String()
    try:
        s1.initialize()
    except e:
        v = String(e)
    assert_equal(v, "QA-E130: Unsupported protocol version from the server: 1999-01-01")
    assert_equal(s1.session_id, "sess-bad")
    s1.close()

    var s2 = _session(fx)
    assert_equal(_call_err(s2), "QA-E130: MCP error -32602 on tools/call: Unknown tool: nope")

    # HTTP errors: the message of the body, the key echoed by the server redacted.
    var s3 = _session(fx)
    var e500 = _call_err(s3)
    assert_equal(e500, "QA-E130: MCP HTTP 500 Internal Server Error on tools/call: boom with key [REDACTED]")
    assert_true(e500.find(KEY) < 0)
    var s4 = _session(fx)
    var e401 = _call_err(s4)
    assert_equal(e401, "QA-E130: MCP HTTP 401 Unauthorized on tools/call: denied for [REDACTED]. Hint: check the value of " + KEY_ENV)
    assert_true(e401.find(KEY) < 0)

    var s5 = _session(fx)
    assert_equal(_call_err(s5), "QA-E130: MCP server accepted tools/call (HTTP 202) without answering it")
    var s6 = _session(fx)
    assert_equal(_call_err(s6), "QA-E130: MCP response to tools/call has an unexpected content type: text/plain")
    var s7 = _session(fx)
    assert_equal(_call_err(s7), "QA-E130: MCP server closed the event stream before answering tools/call")
    var s8 = _session(fx)
    assert_equal(_call_err(s8), "QA-E130: MCP response to tools/call has no JSON-RPC response for request id 0")
    var s9 = _session(fx)
    assert_true(_call_err(s9).startswith("QA-E130: MCP response to tools/call is not JSON: "))

    # A 307 to the same origin is followed (method, body and headers kept); to another origin it is an error.
    var s10 = _session(fx)
    var moved = s10.call_tool("x", "{}")
    assert_equal(moved.parts[0], "moved ok")
    var s11 = _session(fx)
    assert_equal(_call_err(s11), "QA-E130: MCP HTTP 307 Temporary Redirect on tools/call")

    # An SSE stream that stalls longer than the SSE read timeout.
    var s12 = _session(fx)
    s12.sse_read_timeout_ms = 250
    var t0 = monotonic_ms()
    assert_equal(_call_err(s12), "QA-E130: MCP server did not answer tools/call within 250 ms")
    var waited = monotonic_ms() - t0
    assert_true(waited >= 200 and waited < 3000, "timeout took " + String(waited))

    # Message caps (a JSON body and an SSE event).
    var s13 = _session(fx)
    s13.max_message_bytes = 60
    assert_equal(_call_err(s13), "QA-E130: MCP response to tools/call is larger than 60 bytes")
    var s14 = _session(fx)
    s14.max_message_bytes = 60
    assert_equal(_call_err(s14), "QA-E130: MCP event for tools/call is larger than 60 bytes")

    # No bearer variable: no Authorization header. No session id from the server: no DELETE on close.
    var s15 = _session(fx, "")
    s15.initialize()
    assert_equal(s15.session_id, "")
    assert_equal(s15.protocol_version, "2024-11-05")
    s15.close()
    assert_equal(fx.finish(), 0)

    var log = fx.log()
    assert_equal(len(log), 18)
    assert_equal(log[1].get_string("method", ""), "DELETE")
    assert_equal(_hdr(log[1], "mcp-session-id"), "sess-bad")
    # The redirected request kept method, body and Authorization.
    assert_equal(log[11].get_string("path", ""), "/mcp-moved")
    assert_equal(log[11].get_string("method", ""), "POST")
    assert_equal(log[11].get_string("body", ""), log[10].get_string("body", ""))
    assert_equal(_hdr(log[11], "authorization"), "Bearer " + KEY)
    # Sessions without a bearer variable send no Authorization.
    assert_equal(_hdr(log[16], "authorization"), "<absent>")
    assert_equal(_hdr(log[17], "authorization"), "<absent>")
    assert_equal(_hdr(log[17], "mcp-protocol-version"), "2024-11-05")
    assert_equal(_hdr(log[17], "mcp-session-id"), "<absent>")
    for i in range(len(log)):
        if i != 1:
            assert_equal(log[i].get_string("method", ""), "POST")


def main() raises:
    ignore_sigpipe()
    test_construction()
    test_refused_connection()
    test_session_flow()
    test_failures()
    _ = unsetenv(KEY_ENV)
    print("test_mcp_http OK")
