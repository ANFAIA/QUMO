"""Tests for the redirect rules of net/http.mojo (D2c, [critic A8]): https to http
refused, Authorization and Cookie dropped on a scheme, host or port change, at most 10
hops and 4 visits of one URL, and each mode's method/body handling. The pure rules are
checked directly; the hop loop runs against tests/support/http_fixture_server."""

from std.os import stat
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
from qumo_agent.net.url import parse_url
from qumo_agent.net.http import (
    HttpClient, HttpRequest, HttpOptions, HttpResponse, redirect_target, keeps_authorization,
    REDIRECT_NONE, REDIRECT_URLLIB, REDIRECT_SAME_ORIGIN, REDIRECT_KEEP_METHOD,
)


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


def _err_of_target(cur: String, loc: String, status: Int, mode: Int) -> String:
    try:
        return redirect_target(parse_url(cur), loc, status, "GET", mode).to_string()
    except e:
        return "ERR " + String(e)


def test_pure() raises:
    # https -> http is always refused, whatever the mode.
    assert_equal(_err_of_target("https://a.example/x", "http://a.example/y", 302, REDIRECT_URLLIB),
                 "ERR QA-E104: refusing redirect from https to http")
    assert_equal(_err_of_target("https://a.example/x", "http://b.example/", 307, REDIRECT_KEEP_METHOD),
                 "ERR QA-E104: refusing redirect from https to http")
    assert_equal(_err_of_target("https://a.example/x/y", "../z?q=1", 301, REDIRECT_URLLIB), "https://a.example/z?q=1")
    assert_equal(_err_of_target("http://a.example/x", "https://a.example/y", 308, REDIRECT_URLLIB), "https://a.example/y")
    assert_equal(_err_of_target("https://a.example/x", "//cdn.example/f", 302, REDIRECT_URLLIB), "https://cdn.example/f")
    assert_true(_err_of_target("https://a.example/x", "/y", 302, REDIRECT_NONE).startswith("ERR QA-E104"))
    assert_true(_err_of_target("https://a.example/x", "/y", 200, REDIRECT_URLLIB).startswith("ERR QA-E104"))
    # Authorization kept only when scheme, host and port are unchanged.
    var base = parse_url("https://api.github.com/repos/a/b")
    assert_true(keeps_authorization(base, parse_url("https://api.github.com/other")))
    assert_true(keeps_authorization(base, parse_url("https://API.github.com:443/x")))
    assert_false(keeps_authorization(base, parse_url("https://objects.github.com/x")))
    assert_false(keeps_authorization(base, parse_url("https://api.github.com:8443/x")))
    assert_false(keeps_authorization(parse_url("http://h/x"), parse_url("https://h/x")))


def _opts(mode: Int) -> HttpOptions:
    return HttpOptions(5000, 5000, 1 << 20, False, mode, 10, True, False)


def _redirect(path: String, status: Int, location: String) -> String:
    return ('{"match": {"path": "' + path + '"}, "status": ' + String(status)
            + ', "reason": "Moved", "headers": {"Location": "' + location + '"}}')


def _ok(path: String, body: String) -> String:
    return '{"match": {"path": "' + path + '"}, "status": 200, "body": "' + body + '"}'


def _get(mut c: HttpClient, url: String, mode: Int) -> String:
    try:
        var r = c.request(HttpRequest("GET", url, List[String](), List[String](), ""), _opts(mode))
        return String(r.status) + " " + r.text()
    except e:
        return "ERR " + String(e)


def test_hops_and_visits() raises:
    # Exactly 10 hops are followed; the 11th redirect is a loop.
    var entries = List[String]()
    for i in range(10):
        entries.append(_redirect("/h" + String(i), 302, "/h" + String(i + 1)))
    entries.append(_ok("/h10", "ten hops"))
    for i in range(11):
        entries.append(_redirect("/k" + String(i), 307, "/k" + String(i + 1)))
    # /a <-> /b: the 5th visit of /b is refused (4 visits of one URL at most).
    for _ in range(4):
        entries.append(_redirect("/a", 302, "/b"))
        entries.append(_redirect("/b", 302, "/a"))
    entries.append(_redirect("/a", 302, "/b"))
    var script = String('{"responses": [')
    for i in range(len(entries)):
        if i > 0:
            script += ",\n"
        script += entries[i]
    script += "]}"
    var fx = Fixture(script)
    var c = HttpClient(None)
    assert_equal(_get(c, fx.url("/h0"), REDIRECT_URLLIB), "200 ten hops")
    assert_equal(_get(c, fx.url("/k0"), REDIRECT_URLLIB), "ERR QA-E104: redirect loop")
    assert_equal(_get(c, fx.url("/a"), REDIRECT_URLLIB), "ERR QA-E104: redirect loop")
    assert_equal(fx.finish(), 0)
    assert_equal(len(fx.log()), 11 + 11 + 9)


def test_modes_and_headers() raises:
    # A second server: a redirect to it changes the port (Authorization and Cookie dropped).
    var other = Fixture('{"responses": [' + _ok("/landing", "other port") + "]}")
    var script = ('{"responses": ['
        + _redirect("/same", 302, "/same2") + ",\n"
        + _ok("/same2", "same origin") + ",\n"
        + _redirect("/cross", 302, other.url("/landing")) + ",\n"
        # URLLIB: POST + 303 becomes GET without a body or content headers.
        + _redirect("/post303", 303, "/got") + ",\n"
        + '{"match": {"method": "GET", "path": "/got"}, "status": 200, "body": "as GET"}' + ",\n"
        # URLLIB: POST + 307 is not followed (urllib raises HTTPError): the 307 comes back.
        + _redirect("/post307", 307, "/never") + ",\n"
        # KEEP_METHOD: 307 keeps POST and its body; 302 is not followed.
        + _redirect("/keep", 307, "/kept") + ",\n"
        + '{"match": {"method": "POST", "path": "/kept"}, "status": 200, "body": "kept"}' + ",\n"
        + _redirect("/keep302", 302, "/never") + ",\n"
        # SAME_ORIGIN: 308 to the same origin followed, to another origin returned.
        + _redirect("/so", 308, "/so2") + ",\n"
        + _ok("/so2", "same origin ok") + ",\n"
        + _redirect("/so-cross", 307, other.url("/landing")) + ",\n"
        # NONE: nothing followed.
        + _redirect("/none", 301, "/never")
        + "]}")
    var fx = Fixture(script)
    var c = HttpClient(None)
    var names: List[String] = ["Authorization", "Cookie", "X-Keep"]
    var values: List[String] = ["Bearer secret-token", "a=b", "1"]
    var r1 = c.request(HttpRequest("GET", fx.url("/same"), names.copy(), values.copy(), ""), _opts(REDIRECT_URLLIB))
    assert_equal(r1.text(), "same origin")
    var r2 = c.request(HttpRequest("GET", fx.url("/cross"), names.copy(), values.copy(), ""), _opts(REDIRECT_URLLIB))
    assert_equal(r2.text(), "other port")
    assert_equal(r2.final_url, other.url("/landing"))
    var pn: List[String] = ["Content-Type", "Authorization"]
    var pv: List[String] = ["application/json", "Bearer k"]
    var r3 = c.request(HttpRequest("POST", fx.url("/post303"), pn.copy(), pv.copy(), '{"x": 1}'), _opts(REDIRECT_URLLIB))
    assert_equal(r3.text(), "as GET")
    var r4 = c.request(HttpRequest("POST", fx.url("/post307"), pn.copy(), pv.copy(), "{}"), _opts(REDIRECT_URLLIB))
    assert_equal(r4.status, 307)
    var r5 = c.request(HttpRequest("POST", fx.url("/keep"), pn.copy(), pv.copy(), '{"keep": true}'), _opts(REDIRECT_KEEP_METHOD))
    assert_equal(r5.text(), "kept")
    var r6 = c.request(HttpRequest("POST", fx.url("/keep302"), pn.copy(), pv.copy(), "{}"), _opts(REDIRECT_KEEP_METHOD))
    assert_equal(r6.status, 302)
    assert_equal(_get(c, fx.url("/so"), REDIRECT_SAME_ORIGIN), "200 same origin ok")
    var r7 = c.request(HttpRequest("GET", fx.url("/so-cross"), List[String](), List[String](), ""), _opts(REDIRECT_SAME_ORIGIN))
    assert_equal(r7.status, 307)
    assert_equal(r7.final_url, fx.url("/so-cross"))
    var r8 = c.request(HttpRequest("GET", fx.url("/none"), List[String](), List[String](), ""), _opts(REDIRECT_NONE))
    assert_equal(r8.status, 301)
    assert_equal(r8.header("location"), "/never")
    assert_equal(fx.finish(), 0)
    assert_equal(other.finish(), 0)

    var log = fx.log()
    # Same host and port: Authorization and Cookie kept.
    var same2 = log[1].get("headers")
    assert_equal(log[1].get_string("path", ""), "/same2")
    assert_equal(same2.get_string("authorization", ""), "Bearer secret-token")
    assert_equal(same2.get_string("cookie", ""), "a=b")
    # Port change: dropped, other headers kept.
    var landing = other.log()[0].get("headers")
    assert_false(landing.has("authorization"))
    assert_false(landing.has("cookie"))
    assert_equal(landing.get_string("x-keep", ""), "1")
    # POST 303 -> GET without body and content headers, Authorization kept (same origin).
    var got = log[4].copy()
    assert_equal(got.get_string("method", ""), "GET")
    assert_equal(got.get_string("body", ""), "")
    assert_false(got.get("headers").has("content-type"))
    assert_false(got.get("headers").has("content-length"))
    assert_equal(got.get("headers").get_string("authorization", ""), "Bearer k")
    # 307 with KEEP_METHOD: POST and body kept.
    var kept = log[7].copy()
    assert_equal(kept.get_string("method", ""), "POST")
    assert_equal(kept.get_string("body", ""), '{"keep": true}')
    assert_equal(kept.get("headers").get_string("content-type", ""), "application/json")


def main() raises:
    ignore_sigpipe()
    test_pure()
    test_hops_and_visits()
    test_modes_and_headers()
    print("test_net_redirects OK")
