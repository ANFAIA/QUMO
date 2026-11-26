"""Tests for qumo/net (the copy the hardware clients import) against the local fixture
server and a local `openssl s_server`: the calls of qumo/exec/*_api.mojo, which are
`HttpClient(None)` or a TLS context, REDIRECT_NONE, bounded timeouts, QM-E959 errors.

Cases: a JSON POST with headers arrives as sent; a 302 is returned as the response and the
Location is not requested; a body delayed past the I/O timeout gives `read timed out`; TLS to `localhost` with a test CA succeeds, an
untrusted CA and a wrong host name fail the handshake with QM-E959.

Nothing leaves 127.0.0.1. The TLS part needs an OpenSSL 3 command-line tool
($QUMO_TEST_OPENSSL, Homebrew's openssl@3, or /usr/bin/openssl when it is OpenSSL 3).
"""

from std.memory import ArcPointer
from std.os import getenv, stat
from std.os.path import exists
from std.testing import assert_equal, assert_true, assert_false

from qumo.core.json import JsonValue, parse_json
from qumo.net.libc import CStr, c_rename, c_close, SIGKILL
from qumo.net.http import HttpClient, HttpRequest, HttpOptions, HttpResponse, REDIRECT_NONE
from qumo.net.socket import TcpListener, tcp_connect
from qumo.net.tls import TlsContext
from qumo_agent.native.spawn import (
    SpawnSpec, Child, spawn, run_capture, minimal_child_env, environ_list, ignore_sigpipe,
    STDIO_NULL, STDIO_PIPE, STDIO_INHERIT,
)
from qumo_agent.native.fs import make_temp_dir, make_dirs, write_text, read_text_lossy, list_dir, remove_tree
from qumo_agent.native.clock import monotonic_ms, sleep_ms
from qumo_agent.native.rand import uuid4_hex


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
    newest = max(newest, _mtime("qumo/core/json.mojo"))
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


def _openssl() raises -> String:
    var cands = List[String]()
    var env = getenv("QUMO_TEST_OPENSSL", "")
    if env.byte_length() > 0:
        cands.append(env)
    cands.append("/opt/homebrew/opt/openssl@3/bin/openssl")
    cands.append("/usr/local/opt/openssl@3/bin/openssl")
    cands.append("/usr/bin/openssl")
    for c in cands:
        if not exists(c):
            continue
        var r = run_capture(SpawnSpec(c, [c, "version"], minimal_child_env(), "", STDIO_NULL, STDIO_PIPE, STDIO_PIPE, False),
                            10000, 4096, "")
        if r.exit_code == 0 and r.stdout.startswith("OpenSSL 3"):
            return c
    raise Error("no OpenSSL 3 command-line tool found; set QUMO_TEST_OPENSSL")


def _run(openssl: String, args: List[String], dir: String) raises:
    var argv: List[String] = [openssl]
    for a in args:
        argv.append(a)
    var r = run_capture(SpawnSpec(openssl, argv^, minimal_child_env(), dir, STDIO_NULL, STDIO_PIPE, STDIO_PIPE, False),
                        60000, 1 << 16, "")
    if r.exit_code != 0:
        raise Error("openssl " + args[0] + " failed: " + r.stderr)


def _ext(san: String) -> String:
    return ("basicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature\nextendedKeyUsage=serverAuth\n"
            + "subjectAltName=" + san + "\nsubjectKeyIdentifier=hash\nauthorityKeyIdentifier=keyid\n")


def _make_ca(o: String, dir: String, name: String) raises:
    _run(o, ["req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:P-256", "-nodes",
             "-keyout", name + ".key", "-out", name + ".pem", "-days", "2", "-subj", "/CN=QUMO Test " + name,
             "-addext", "basicConstraints=critical,CA:TRUE", "-addext", "keyUsage=critical,keyCertSign,cRLSign"], dir)


def _make_leaf(o: String, dir: String, name: String, ca: String, san: String, expired: Bool) raises:
    write_text(dir + "/" + name + ".ext", _ext(san))
    _run(o, ["req", "-new", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:P-256", "-nodes",
             "-keyout", name + ".key", "-out", name + ".csr", "-subj", "/CN=" + name], dir)
    var args: List[String] = ["x509", "-req", "-in", name + ".csr", "-CA", ca + ".pem", "-CAkey", ca + ".key",
                              "-set_serial", "7", "-out", name + ".pem", "-extfile", name + ".ext"]
    if expired:
        args.append("-not_before")
        args.append("20200101000000Z")
        args.append("-not_after")
        args.append("20200102000000Z")
    else:
        args.append("-days")
        args.append("2")
    _run(o, args, dir)


def _free_port() raises -> Int:
    var l = TcpListener("127.0.0.1", 0)
    return l.port()


struct Server(Movable):
    var child: Child
    var port: Int

    def __init__(out self, o: String, dir: String, cert: String, chain: String, mode: String) raises:
        """Starts `openssl s_server` on 127.0.0.1:<free port> in `dir`; `mode` is -WWW or -HTTP."""
        self.port = _free_port()
        var argv: List[String] = [o, "s_server", "-accept", "127.0.0.1:" + String(self.port), "-cert", cert + ".pem",
                                  "-key", cert + ".key", mode, "-quiet"]
        if chain.byte_length() > 0:
            argv.append("-cert_chain")
            argv.append(chain + ".pem")
        self.child = spawn(SpawnSpec(o, argv^, minimal_child_env(), dir, STDIO_NULL, STDIO_NULL, STDIO_NULL, True))
        var deadline = monotonic_ms() + 10000
        while True:
            try:
                var fd = tcp_connect("127.0.0.1", self.port, 500)
                _ = c_close(fd)
                break
            except:
                if monotonic_ms() > deadline:
                    raise Error("s_server did not start")
                sleep_ms(50)

    def kill(mut self):
        self.child.signal_group(SIGKILL)

    def stop(mut self) raises:
        _ = self.child.terminate(500)


def _hw_opts(io_ms: Int) -> HttpOptions:
    """The options of the hardware clients, with a short I/O timeout."""
    return HttpOptions(3000, io_ms, 1 << 20, False, REDIRECT_NONE, 0, True, False)


def _call(mut c: HttpClient, method: String, url: String, body: String, io_ms: Int) -> String:
    """`<status> <body>` or `ERR <error>`."""
    var names: List[String] = ["Authorization", "Content-Type"]
    var values: List[String] = ["apiKey test-key", "application/json"]
    try:
        var r = c.request(HttpRequest(method, url, names^, values^, body), _hw_opts(io_ms))
        return String(r.status) + " " + r.text()
    except e:
        return "ERR " + String(e)


def test_fixture_server() raises:
    var script = ('{"responses": ['
                  + '{"match": {"method": "POST", "path": "/v0.3/jobs"}, "status": 200, "headers": {"Content-Type": "application/json"}, "body": "{\\"id\\": \\"j1\\"}"},\n'
                  + '{"match": {"path": "/moved"}, "status": 302, "reason": "Found", "headers": {"Location": "/v0.3/jobs"}, "body": "see other"},\n'
                  + '{"match": {"path": "/chunked"}, "status": 200, "framing": "chunked", "chunks": ["ab", "cd"], "body": "abcd"},\n'
                  + '{"match": {"path": "/slow"}, "status": 200, "body": "late", "delay_ms": 1500}'
                  + ']}')
    var fx = Fixture(script)
    var c = HttpClient(None)
    assert_equal(_call(c, "POST", fx.url("/v0.3/jobs"), '{"shots": 100}', 3000), '200 {"id": "j1"}')
    # REDIRECT_NONE: the 302 is the answer.
    assert_equal(_call(c, "GET", fx.url("/moved"), "", 3000), "302 see other")
    assert_equal(_call(c, "GET", fx.url("/chunked"), "", 3000), "200 abcd")
    var t0 = monotonic_ms()
    assert_equal(_call(c, "GET", fx.url("/slow"), "", 400), "ERR QM-E959: read timed out")
    var ms = monotonic_ms() - t0
    assert_true(ms >= 300 and ms <= 1400, "read timeout took " + String(ms) + " ms")
    _ = fx.finish()
    var log = fx.log()
    assert_equal(len(log), 4)   # the Location of /moved was not requested
    assert_equal(log[0].get_string("method", ""), "POST")
    assert_equal(log[0].get_string("body", ""), '{"shots": 100}')
    assert_equal(log[0].get("headers").get_string("authorization", ""), "apiKey test-key")
    assert_equal(log[1].get_string("path", ""), "/moved")
    assert_equal(log[2].get_string("path", ""), "/chunked")
    assert_equal(log[3].get_string("path", ""), "/slow")


def test_tls() raises:
    var o = _openssl()
    var dir = make_temp_dir("qumo-net-tls-")
    _make_ca(o, dir, "ca")
    _make_ca(o, dir, "ca2")
    _make_leaf(o, dir, "leaf", "ca", "DNS:localhost,IP:127.0.0.1", False)
    _make_leaf(o, dir, "other", "ca", "DNS:other.test", False)
    write_text(dir + "/hello.txt", "hello over TLS\n")
    var good = Server(o, dir, "leaf", "", "-WWW")
    var base = "https://localhost:" + String(good.port)
    var c = HttpClient(ArcPointer(TlsContext(dir + "/ca.pem")))
    assert_equal(_call(c, "GET", base + "/hello.txt", "", 3000), "200 hello over TLS\n")
    assert_equal(_call(c, "GET", "https://127.0.0.1:" + String(good.port) + "/hello.txt", "", 3000), "200 hello over TLS\n")
    # A CA that did not sign the leaf: verify error 20, as QM-E959.
    var stranger = HttpClient(ArcPointer(TlsContext(dir + "/ca2.pem")))
    var untrusted = _call(stranger, "GET", base + "/hello.txt", "", 3000)
    assert_true(untrusted.startswith("ERR QM-E959: TLS handshake with localhost failed") and untrusted.find("20") >= 0, untrusted)
    good.stop()
    # A leaf for another name: host name mismatch (62).
    var wrong = Server(o, dir, "other", "", "-WWW")
    var mismatch = _call(c, "GET", "https://localhost:" + String(wrong.port) + "/hello.txt", "", 3000)
    assert_true(mismatch.startswith("ERR QM-E959: TLS handshake with localhost failed") and mismatch.find("62") >= 0, mismatch)
    wrong.stop()
    remove_tree(dir)


def main() raises:
    ignore_sigpipe()
    test_fixture_server()
    test_tls()
    print("test_net_qumo OK")
