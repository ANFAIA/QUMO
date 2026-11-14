"""Tests for llm/client.mojo: the openai SDK retry policy (pure tables with fixed
randomness) and the client against tests/support/http_fixture_server over plain HTTP:
request headers and body, 429 with retry-after-ms retried, 500 twice then 200, 500 three
times gives QA-E101 after 3 attempts, x-should-retry, retry-after above 120 s not
retried, context overflow (QA-E102) at once and after retries for any status, 200 with
`error` (QA-E103), the key echoed by the server redacted, a truncated body retried, a
refused connection (QA-E104 after 3 attempts), and a plain-HTTP model URL working with
OpenSSL absent (QUMO_LIBSSL=/nonexistent)."""

from std.os import stat, getenv, setenv, unsetenv
from std.os.path import exists
from std.testing import assert_equal, assert_true, assert_false

from qumo.ai.json import JsonValue, parse_json
from qumo_agent.native.libc import CStr, c_rename
from qumo_agent.native.spawn import (
    SpawnSpec, Child, spawn, run_capture, minimal_child_env, environ_list, ignore_sigpipe,
    STDIO_NULL, STDIO_PIPE, STDIO_INHERIT,
)
from qumo_agent.native.fs import make_temp_dir, make_dirs, write_text, read_text_lossy, list_dir
from qumo_agent.native.clock import monotonic_ms, sleep_ms
from qumo_agent.native.rand import uuid4_hex
from qumo_agent.core.endpoint import ModelEndpoint
from qumo_agent.core.messages import ChatRequest, ChatResponse, Message, user_message
from qumo_agent.core.toolspec import ToolSpec
from qumo_agent.net.socket import TcpListener
from qumo_agent.net.tls import TlsContext
from qumo_agent.llm.client import LlmClient, should_retry, retry_delay_ms, parse_retry_after_s


# ---------------------------------------------------------------------------
# http_fixture_server helper (each N test that needs the server carries a copy)
# ---------------------------------------------------------------------------
comptime _FIXTURE_SRC = "agents/mojo/tests/support/http_fixture_server.mojo"
comptime _FIXTURE_BIN = "build/test-fixtures/http_fixture_server"


def _mtime(path: String) raises -> Int:
    var st = stat(path)
    return Int(st.st_mtimespec.tv_sec) * 1000000000 + Int(st.st_mtimespec.tv_nsec)


def _newest_source() raises -> Int:
    var newest = _mtime(_FIXTURE_SRC)
    newest = max(newest, _mtime("qumo/ai/json.mojo"))
    var dirs: List[String] = ["agents/mojo/qumo_agent/net", "agents/mojo/qumo_agent/native"]
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
        self.dir = make_temp_dir("qumo-http-fixture-")
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


comptime KEY_ENV = "QUMO_TEST_LLM_KEY"
comptime KEY = "sk-or-v1-0123456789abcdefFAKE"
comptime OK_BODY = ('{\\"id\\": \\"gen-1\\", \\"choices\\": [{\\"index\\": 0, \\"message\\": {\\"role\\": \\"assistant\\", '
                    + '\\"content\\": \\"hi there\\", \\"reasoning\\": \\"dropped\\"}, \\"finish_reason\\": \\"stop\\"}], '
                    + '\\"usage\\": {\\"prompt_tokens\\": 3, \\"completion_tokens\\": 2, \\"total_tokens\\": 5}}')


def _h(names: List[String], values: List[String]) -> Tuple[List[String], List[String]]:
    return (names.copy(), values.copy())


def test_policy() raises:
    var none = List[String]()
    # Backoff: min(0.5 * 2**n, 8) * (1 - 0.25 * u), in ms.
    var want0: List[Int] = [500, 1000, 2000, 4000, 8000, 8000]
    var want1: List[Int] = [375, 750, 1500, 3000, 6000, 6000]
    for n in range(6):
        assert_equal(retry_delay_ms(n, none, none, 0, 0.0), want0[n])
        assert_equal(retry_delay_ms(n, none, none, 0, 1.0), want1[n])
    assert_equal(retry_delay_ms(2, none, none, 0, 0.5), 1750)
    assert_equal(retry_delay_ms(5000, none, none, 0, 0.0), 8000)
    # Server-directed delays in (0, 120] s win; others fall back to the backoff.
    var n1: List[String] = ["Retry-After-Ms"]
    assert_equal(retry_delay_ms(0, n1, ["50"], 0, 0.0), 50)
    assert_equal(retry_delay_ms(0, n1, ["abc"], 0, 0.0), 500)
    var n2: List[String] = ["retry-after"]
    assert_equal(retry_delay_ms(0, n2, ["2.5"], 0, 0.0), 2500)
    assert_equal(retry_delay_ms(0, n2, ["0"], 0, 0.0), 500)
    assert_equal(retry_delay_ms(1, n2, ["121"], 0, 0.0), 1000)
    assert_equal(retry_delay_ms(0, n2, ["-1"], 0, 0.0), 500)
    var both: List[String] = ["retry-after", "retry-after-ms"]
    assert_equal(retry_delay_ms(0, both, ["9", "250"], 0, 0.0), 250)
    # HTTP date minus now.
    var date = "Sun, 06 Nov 1994 08:49:37 GMT"
    assert_equal(parse_retry_after_s(n2, [date], 784111777 - 30).value(), 30.0)
    assert_equal(retry_delay_ms(0, n2, [date], 784111777 - 30, 0.0), 30000)
    assert_false(parse_retry_after_s(n2, ["soon"], 0))
    assert_false(parse_retry_after_s(none, none, 0))
    assert_equal(parse_retry_after_s(n2, ["1e999"], 0).value(), 121.0)
    assert_equal(parse_retry_after_s(n2, [" 1_0 "], 0).value(), 10.0)
    assert_true(parse_retry_after_s(n2, ["-1e999"], 0).value() < 0.0)
    assert_true(should_retry(503, n2, ["inf"], 0))
    assert_false(should_retry(503, n2, ["1e999"], 0))
    # should_retry: retry-after > 120 s never; x-should-retry obeyed; 408/409/429/5xx.
    var codes: List[Int] = [400, 401, 403, 404, 408, 409, 413, 422, 429, 500, 502, 503, 599]
    var want: List[Bool] = [False, False, False, False, True, True, False, False, True, True, True, True, True]
    for i in range(len(codes)):
        assert_equal(should_retry(codes[i], none, none, 0), want[i], "status " + String(codes[i]))
    var xs: List[String] = ["x-should-retry"]
    assert_true(should_retry(400, xs, ["true"], 0))
    assert_false(should_retry(503, xs, ["false"], 0))
    assert_true(should_retry(503, xs, ["maybe"], 0))
    assert_false(should_retry(429, n2, ["200"], 0))
    assert_true(should_retry(429, n2, ["120"], 0))
    assert_false(should_retry(503, n1, ["120001"], 0))


def _request() -> ChatRequest:
    var msgs = List[Message]()
    msgs.append(user_message("hello é"))
    return ChatRequest("test/model", "SYS", msgs^, List[ToolSpec](), "qumo-operator", "")


def _endpoint(base: String, retries: Int) -> ModelEndpoint:
    var names: List[String] = ["X-Title"]
    var values: List[String] = ["QUMO agents"]
    return ModelEndpoint(base, KEY_ENV, names^, values^, 5, retries)


def _complete(mut c: LlmClient) -> String:
    try:
        var r = c.complete(_request())
        return "OK " + r.message.content + " " + r.finish_reason + " " + String(r.total_tokens)
    except e:
        return "ERR " + String(e)


def _resp(status: Int, reason: String, headers: String, body: String, extra: String) -> String:
    return ('{"match": {"method": "POST", "path": "/api/v1/chat/completions"}, "status": ' + String(status)
            + ', "reason": "' + reason + '", "headers": {' + headers + '}, "body": "' + body + '"' + extra + "}")


def _ok() -> String:
    return _resp(200, "OK", '"Content-Type": "application/json"', OK_BODY, "")


def main() raises:
    ignore_sigpipe()
    test_policy()
    # OpenSSL is never loaded for a plain-HTTP model URL.
    _ = setenv("QUMO_LIBSSL", "/nonexistent/libssl.3.dylib", True)
    _ = setenv(KEY_ENV, KEY, True)
    var boom = '{\\"error\\": {\\"message\\": \\"boom\\"}}'
    var entries: List[String] = [
        # A: plain success.
        _ok(),
        # B: 429 with retry-after-ms, then 200.
        _resp(429, "Too Many Requests", '"retry-after-ms": "50"', boom, ""), _ok(),
        # C: 500 (backoff), 500 (retry-after-ms), then 200.
        _resp(500, "Internal Server Error", "", boom, ""),
        _resp(500, "Internal Server Error", '"retry-after-ms": "20"', boom, ""), _ok(),
        # D: 500 three times.
        _resp(500, "Internal Server Error", '"retry-after-ms": "10"', boom, ""),
        _resp(500, "Internal Server Error", '"retry-after-ms": "10"', boom, ""),
        _resp(500, "Internal Server Error", '"retry-after-ms": "10"', boom, ""),
        # E: x-should-retry: false.
        _resp(503, "Service Unavailable", '"x-should-retry": "false"', "busy", ""),
        # F: retry-after above 120 s.
        _resp(429, "Too Many Requests", '"retry-after": "200"', boom, ""),
        # G: context overflow at once.
        _resp(400, "Bad Request", "", '{\\"error\\": {\\"message\\": \\"This model\'s maximum context length is 1000 tokens\\", \\"code\\": \\"context_length_exceeded\\"}}', ""),
        # H: "exceeds the context window" on a 503: retried, then QA-E102.
        _resp(503, "Service Unavailable", '"retry-after-ms": "10"', "input exceeds the context window", ""),
        _resp(503, "Service Unavailable", '"retry-after-ms": "10"', "input exceeds the context window", ""),
        _resp(503, "Service Unavailable", '"retry-after-ms": "10"', "input exceeds the context window", ""),
        # I: 200 with an error object (B5, not retried).
        _resp(200, "OK", "", '{\\"error\\": {\\"message\\": \\"upstream failed\\", \\"code\\": 502}}', ""),
        # J: the server echoes the key.
        _resp(401, "Unauthorized", "", '{\\"error\\": {\\"message\\": \\"bad key ' + KEY + '\\"}}', ""),
        # K: a truncated body is a connection error: retried.
        _resp(200, "OK", "", OK_BODY, ', "close_after_headers": true'), _ok(),
        # L: x-should-retry: true on a 400.
        _resp(400, "Bad Request", '"x-should-retry": "true", "retry-after-ms": "10"', boom, ""), _ok(),
    ]
    var script = String('{"responses": [')
    for i in range(len(entries)):
        if i > 0:
            script += ",\n"
        script += entries[i]
    script += "]}"
    var fx = Fixture(script)
    var c = LlmClient(_endpoint(fx.url("/api/v1/"), 2), None)

    assert_equal(_complete(c), "OK hi there stop 5")                                   # A
    assert_equal(c.stats.http_requests, 1)
    assert_equal(_complete(c), "OK hi there stop 5")                                   # B
    assert_equal(c.stats.http_retries, 1)
    var t0 = monotonic_ms()
    assert_equal(_complete(c), "OK hi there stop 5")                                   # C
    var waited = monotonic_ms() - t0
    assert_true(waited >= 375 and waited < 3000, "backoff then retry-after-ms: " + String(waited))
    assert_equal(c.stats.http_retries, 3)
    assert_equal(_complete(c), "ERR QA-E101: model request failed: HTTP 500 Internal Server Error: boom")  # D
    assert_equal(c.stats.http_retries, 5)
    assert_equal(c.stats.http_requests, 9)
    assert_equal(_complete(c), "ERR QA-E101: model request failed: HTTP 503 Service Unavailable: busy")    # E
    assert_equal(_complete(c), "ERR QA-E101: model request failed: HTTP 429 Too Many Requests: boom")     # F
    assert_equal(c.stats.http_requests, 11)
    assert_equal(_complete(c), "ERR QA-E102: context overflow: HTTP 400: This model's maximum context length is 1000 tokens")  # G
    assert_equal(c.stats.http_requests, 12)
    assert_equal(_complete(c), "ERR QA-E102: context overflow: HTTP 503: input exceeds the context window")  # H
    assert_equal(c.stats.http_requests, 15)
    var b5 = _complete(c)                                                                 # I
    assert_true(b5.startswith("ERR QA-E103: model returned an error: "), b5)
    assert_equal(c.stats.http_requests, 16)
    var echoed = _complete(c)                                                             # J
    assert_equal(echoed, "ERR QA-E101: model request failed: HTTP 401 Unauthorized: bad key [REDACTED]")
    assert_true(echoed.find(KEY) < 0)
    var retries_before = c.stats.http_retries
    assert_equal(_complete(c), "OK hi there stop 5")                                   # K
    assert_equal(c.stats.http_retries, retries_before + 1)
    assert_equal(_complete(c), "OK hi there stop 5")                                   # L
    assert_equal(c.stats.http_requests, 21)
    assert_equal(fx.finish(), 0)

    # What the server saw on the first request.
    var log = fx.log()
    assert_equal(len(log), 21)
    var first = log[0].copy()
    assert_equal(first.get_string("method", ""), "POST")
    assert_equal(first.get_string("path", ""), "/api/v1/chat/completions")
    var h = first.get("headers")
    var keys = h.keys()
    var want_keys: List[String] = ["host", "authorization", "content-type", "accept", "x-title", "user-agent",
                                   "accept-encoding", "connection", "content-length"]
    assert_equal(len(keys), len(want_keys))
    for i in range(len(keys)):
        assert_equal(keys[i], want_keys[i])
    assert_equal(h.get_string("authorization", ""), "Bearer " + KEY)
    assert_equal(h.get_string("content-type", ""), "application/json")
    assert_equal(h.get_string("accept", ""), "application/json")
    assert_equal(h.get_string("x-title", ""), "QUMO agents")
    assert_equal(h.get_string("user-agent", ""), "qumo-agent/0.2 (Mojo)")
    assert_equal(h.get_string("accept-encoding", ""), "identity")
    assert_equal(h.get_string("connection", ""), "close")
    assert_equal(first.get_string("body", ""), _request().body_json())

    # A refused connection: QA-E104 after 3 attempts (2 retries with backoff).
    var free = TcpListener("127.0.0.1", 0)
    var dead_port = free.port()
    _ = free^
    var dead = LlmClient(_endpoint("http://127.0.0.1:" + String(dead_port) + "/api/v1", 2), None)
    var refused = _complete(dead)
    assert_true(refused.startswith("ERR QA-E104: 127.0.0.1:" + String(dead_port) + ": "), refused)
    assert_equal(dead.stats.http_retries, 2)
    assert_equal(dead.stats.http_requests, 0)

    # The key variable: invalid name, unset.
    var bad_name = String()
    try:
        _ = LlmClient(ModelEndpoint("http://127.0.0.1:1/v1", "1BAD", List[String](), List[String](), 5, 2), None)
    except e:
        bad_name = String(e)
    assert_equal(bad_name, "QA-E140: '1BAD' is not a valid environment variable name")
    _ = unsetenv(KEY_ENV)
    var unset = String()
    try:
        _ = LlmClient(_endpoint("http://127.0.0.1:1/v1", 2), None)
    except e:
        unset = String(e)
    assert_equal(unset, "QA-E140: set QUMO_TEST_LLM_KEY to use the openrouter back end, or pass --replay FILE")
    _ = setenv(KEY_ENV, KEY, True)

    # https with OpenSSL absent: QA-E105 on the first request (OpenSSL is loaded lazily), not retried.
    var tls_less = LlmClient(_endpoint("https://127.0.0.1:1/api/v1", 2), None)
    var no_ssl = _complete(tls_less)
    assert_true(no_ssl.startswith("ERR QA-E105: OpenSSL 3 not found (tried: /nonexistent/libssl.3.dylib (QUMO_LIBSSL))"), no_ssl)
    assert_equal(tls_less.stats.http_retries, 0)
    _ = unsetenv("QUMO_LIBSSL")
    _ = unsetenv(KEY_ENV)
    print("test_llm_client OK")
