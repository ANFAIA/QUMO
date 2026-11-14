"""Tests for net/http.mojo over real sockets, through tests/support/http_fixture_server:
every framing for GET and POST, the request as the server saw it, redirect chains,
streamed SSE bodies, truncation and unmatched requests."""

from std.os import stat
from std.os.path import exists
from std.testing import assert_equal, assert_true, assert_false

from qumo.ai.json import JsonValue, parse_json, json_quote
from qumo_agent.native.libc import CStr, c_rename
from qumo_agent.native.spawn import (
    SpawnSpec, Child, spawn, run_capture, minimal_child_env, environ_list, ignore_sigpipe,
    STDIO_NULL, STDIO_PIPE, STDIO_INHERIT,
)
from qumo_agent.native.fs import make_temp_dir, make_dirs, write_text, read_text_lossy, list_dir
from qumo_agent.native.clock import monotonic_ms, sleep_ms
from qumo_agent.native.rand import uuid4_hex
from qumo_agent.net.http import (
    HttpClient, HttpRequest, HttpOptions, HttpResponse, REDIRECT_NONE, REDIRECT_URLLIB,
)
from qumo_agent.net.sse import SseParser, SseEvent


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


def _opts(mode: Int) -> HttpOptions:
    return HttpOptions(5000, 5000, 1 << 20, False, mode, 10, True, False)


def _no_headers() -> List[String]:
    return List[String]()


def main() raises:
    ignore_sigpipe()
    var script = """{"responses": [
      {"match": {"method": "GET", "path": "/cl"}, "status": 200, "reason": "OK",
       "headers": {"Content-Type": "text/plain", "X-Two": "2"}, "body": "content-length body"},
      {"match": {"method": "POST", "path": "/chunked"}, "status": 201, "reason": "Created",
       "framing": "chunked", "chunks": ["{\\"a\\": ", "1}", ""], "body": "ignored"},
      {"match": {"method": "GET", "path": "/close"}, "status": 200, "framing": "close", "body": "until close é"},
      {"match": {"path": "/start"}, "status": 302, "reason": "Found", "headers": {"Location": "/middle?x=1"}},
      {"match": {"path": "/middle?x=1"}, "status": 301, "headers": {"Location": "end"}},
      {"match": {"path": "/end"}, "status": 200, "body": "arrived"},
      {"match": {"path": "/sse"}, "status": 200, "headers": {"Content-Type": "text/event-stream"},
       "framing": "chunked", "delay_ms": 20,
       "chunks": [": priming\\n\\nid: 1\\ndata: {\\"jsonrpc\\":", "\\"2.0\\",\\"id\\":1}\\n", "\\nevent: done\\ndata: x\\n\\n"]},
      {"match": {"path": "/short"}, "status": 200, "body": "0123456789", "close_after_headers": true},
      {"match": {"path": "/after"}, "status": 204, "reason": "No Content"}
    ]}"""
    var fx = Fixture(script)
    var client = HttpClient(None)

    var r1 = client.request(HttpRequest("GET", fx.url("/cl"), _no_headers(), _no_headers(), ""), _opts(REDIRECT_NONE))
    assert_equal(r1.status, 200)
    assert_equal(r1.text(), "content-length body")
    assert_equal(r1.header("content-type"), "text/plain")
    assert_equal(r1.header("X-TWO"), "2")
    assert_equal(r1.final_url, fx.url("/cl"))

    var names: List[String] = ["Content-Type", "X-Title"]
    var values: List[String] = ["application/json", "QUMO agents"]
    var r2 = client.request(HttpRequest("POST", fx.url("/chunked"), names^, values^, '{"q": "é"}'), _opts(REDIRECT_NONE))
    assert_equal(r2.status, 201)
    assert_equal(r2.reason, "Created")
    assert_equal(r2.text(), '{"a": 1}')

    var r3 = client.request(HttpRequest("GET", fx.url("/close"), _no_headers(), _no_headers(), ""), _opts(REDIRECT_NONE))
    assert_equal(r3.text(), "until close é")

    # Redirect chain: relative Locations resolved, final URL reported.
    var r4 = client.request(HttpRequest("GET", fx.url("/start"), _no_headers(), _no_headers(), ""), _opts(REDIRECT_URLLIB))
    assert_equal(r4.status, 200)
    assert_equal(r4.text(), "arrived")
    assert_equal(r4.final_url, fx.url("/end"))

    # Streamed SSE body.
    var stream = client.open(HttpRequest("GET", fx.url("/sse"), _no_headers(), _no_headers(), ""), _opts(REDIRECT_NONE))
    assert_equal(stream.status, 200)
    assert_equal(stream.header("content-type"), "text/event-stream")
    var parser = SseParser()
    var events = List[String]()
    var chunks = 0
    while True:
        var piece = List[UInt8]()
        if not stream.read_chunk(piece, 5000):
            break
        chunks += 1
        parser.feed(piece)
        var ev = SseEvent(String(), String(), String(), -1)
        while parser.next_event(ev):
            events.append(ev.event + "|" + ev.data + "|" + ev.id)
    assert_true(chunks >= 2, "the body arrived in pieces")
    assert_equal(len(events), 2)
    assert_equal(events[0], 'message|{"jsonrpc":"2.0","id":1}|1')
    assert_equal(events[1], "done|x|1")
    stream.close()

    # Server closes right after the head of a Content-Length response.
    var err = String()
    try:
        _ = client.request(HttpRequest("GET", fx.url("/short"), _no_headers(), _no_headers(), ""), _opts(REDIRECT_NONE))
    except e:
        err = String(e)
    assert_equal(err, "QA-E104: body truncated (10 bytes missing)")

    # An unmatched request gets 500 and does not consume the scripted response.
    var miss = client.request(HttpRequest("GET", fx.url("/nope"), _no_headers(), _no_headers(), ""), _opts(REDIRECT_NONE))
    assert_equal(miss.status, 500)
    assert_equal(miss.text(), "no scripted response")
    var r5 = client.request(HttpRequest("GET", fx.url("/after"), _no_headers(), _no_headers(), ""), _opts(REDIRECT_NONE))
    assert_equal(r5.status, 204)
    assert_equal(len(r5.body), 0)
    assert_equal(client.requests, 10)

    assert_equal(fx.finish(), 0)
    # What the server saw.
    var log = fx.log()
    assert_equal(len(log), 10)
    assert_equal(log[0].get_string("method", ""), "GET")
    assert_equal(log[0].get_string("path", ""), "/cl")
    var h0 = log[0].get("headers")
    assert_equal(h0.get_string("host", ""), "127.0.0.1:" + String(fx.port))
    assert_equal(h0.get_string("accept-encoding", ""), "identity")
    assert_equal(h0.get_string("connection", ""), "close")
    assert_false(h0.has("content-length"))
    var h1 = log[1].get("headers")
    assert_equal(log[1].get_string("body", ""), '{"q": "é"}')
    assert_equal(h1.get_string("content-length", ""), "11")
    assert_equal(h1.get_string("content-type", ""), "application/json")
    assert_equal(h1.get_string("x-title", ""), "QUMO agents")
    assert_equal(h1.keys()[0], "host")
    assert_equal(log[4].get_string("path", ""), "/middle?x=1")
    assert_equal(log[5].get_string("path", ""), "/end")
    assert_equal(log[8].get_string("path", ""), "/nope")
    assert_equal(log[9].get_string("path", ""), "/after")
    print("test_net_http_fixture OK")
