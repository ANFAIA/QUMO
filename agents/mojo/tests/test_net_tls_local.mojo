"""Tests for net/tls.mojo and net/socket.mojo against a local `openssl s_server`, with a
per-run CA and leaf certificates made by `openssl` 3 (no network).

Cases: GET with `TlsContext(ca)`; read timeouts after the handshake and on plain TCP; the default store gives verify error 20;
`SSL_CERT_FILE=ca.pem` with the production constructor succeeds; a leaf for
`other.test` gives 62; an IP literal with an IP SAN succeeds and without one gives 64;
an expired leaf gives 10, a self-signed leaf 18, an untrusted root in the chain 19;
killing the server mid-body gives a truncation error for a Content-Length body and,
without `accept_ragged_eof`, for a close-delimited body; a listener that never accepts
makes the handshake time out in 1.5 s +- 0.5.

Needs an OpenSSL 3 command-line tool: $QUMO_TEST_OPENSSL, Homebrew's openssl@3, or
/usr/bin/openssl when it is OpenSSL 3 (macOS's /usr/bin/openssl is LibreSSL).
"""

from std.memory import ArcPointer
from std.os import getenv, setenv, unsetenv
from std.os.path import exists
from std.testing import assert_equal, assert_true, assert_false

from qumo_agent.native.spawn import (
    SpawnSpec, Child, spawn, run_capture, minimal_child_env, ignore_sigpipe,
    STDIO_NULL, STDIO_PIPE,
)
from qumo_agent.native.fs import make_temp_dir, write_text, remove_tree
from qumo_agent.native.clock import monotonic_ms, sleep_ms
from qumo_agent.native.libc import SIGKILL
from qumo_agent.net.tls import TlsContext
from qumo_agent.net.socket import TcpListener, tcp_connect, connect_stream
from qumo_agent.net.url import parse_url
from qumo_agent.net.http import HttpClient, HttpRequest, HttpOptions, REDIRECT_NONE
from qumo_agent.native.libc import c_close


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


def _opts(ragged_ok: Bool) -> HttpOptions:
    return HttpOptions(3000, 3000, 1 << 26, False, REDIRECT_NONE, 10, False, ragged_ok)


def _get(mut client: HttpClient, url: String) -> String:
    """The body, or "ERR <error>"."""
    try:
        var r = client.request(HttpRequest("GET", url, List[String](), List[String](), ""), _opts(False))
        return String(r.status) + " " + r.text()
    except e:
        return "ERR " + String(e)


def _expect_err(got: String, part: String) raises:
    assert_true(got.startswith("ERR QA-E104: TLS handshake with ") and got.find(part) >= 0,
                "expected '" + part + "', got: " + got)


def main() raises:
    ignore_sigpipe()
    var o = _openssl()
    var dir = make_temp_dir("qumo-tls-")
    _make_ca(o, dir, "ca")
    _make_ca(o, dir, "ca2")
    _make_leaf(o, dir, "leaf", "ca", "DNS:localhost,IP:127.0.0.1", False)
    _make_leaf(o, dir, "other", "ca", "DNS:other.test", False)
    _make_leaf(o, dir, "expired", "ca", "DNS:localhost,IP:127.0.0.1", True)
    _make_leaf(o, dir, "untrusted", "ca2", "DNS:localhost", False)
    _run(o, ["req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:P-256", "-nodes", "-keyout", "self.key",
             "-out", "self.pem", "-days", "2", "-subj", "/CN=localhost", "-addext", "subjectAltName=DNS:localhost"], dir)
    write_text(dir + "/hello.txt", "hello over TLS\n")
    var ca = dir + "/ca.pem"
    var test_ctx = ArcPointer(TlsContext(ca))
    assert_equal(test_ctx[].trust_source, "ca-file:" + ca)

    # 1. Good leaf: host name and IP literal (IP SAN).
    var good = Server(o, dir, "leaf", "", "-WWW")
    var base = "https://localhost:" + String(good.port)
    var client = HttpClient(test_ctx)
    assert_equal(_get(client, base + "/hello.txt"), "200 hello over TLS\n")
    assert_equal(_get(client, "https://127.0.0.1:" + String(good.port) + "/hello.txt"), "200 hello over TLS\n")
    var s = connect_stream(parse_url(base + "/"), test_ctx, 3000)
    assert_true(s.tls_version().startswith("TLSv1."))
    # Read timeout after the handshake: the server waits for a request that never comes.
    var idle_buf = List[UInt8]()
    var t_idle = monotonic_ms()
    var idle_err = String()
    try:
        _ = s.read_some(idle_buf, 500)
    except e:
        idle_err = String(e)
    var idle_ms = monotonic_ms() - t_idle
    assert_equal(idle_err, "QA-E104: read timed out")
    assert_true(idle_ms >= 400 and idle_ms <= 1500, "TLS read timeout took " + String(idle_ms) + " ms")
    s.close()
    # The same on plain TCP.
    var quiet = TcpListener("127.0.0.1", 0)
    var plain = connect_stream(parse_url("http://127.0.0.1:" + String(quiet.port()) + "/"), None, 3000)
    var peer = quiet.accept(1000)
    var plain_err = String()
    try:
        _ = plain.read_some(idle_buf, 300)
    except e:
        plain_err = String(e)
    assert_equal(plain_err, "QA-E104: read timed out")
    peer.close()
    assert_equal(plain.read_some(idle_buf, 300), 0)

    # 2. The production store without SSL_CERT_FILE does not trust the test CA: 20.
    var saved_file = getenv("SSL_CERT_FILE", "")
    var saved_dir = getenv("SSL_CERT_DIR", "")
    _ = unsetenv("SSL_CERT_FILE")
    _ = unsetenv("SSL_CERT_DIR")
    var prod = HttpClient(ArcPointer(TlsContext()))
    _expect_err(_get(prod, base + "/hello.txt"), "unable to get local issuer certificate (20)")
    # 3. SSL_CERT_FILE=ca.pem with the production constructor.
    _ = setenv("SSL_CERT_FILE", ca, True)
    var env_ctx = ArcPointer(TlsContext())
    assert_equal(env_ctx[].trust_source, "SSL_CERT_FILE=" + ca)
    var env_client = HttpClient(env_ctx)
    assert_equal(_get(env_client, base + "/hello.txt"), "200 hello over TLS\n")
    _ = unsetenv("SSL_CERT_FILE")
    if saved_file.byte_length() > 0:
        _ = setenv("SSL_CERT_FILE", saved_file, True)
    if saved_dir.byte_length() > 0:
        _ = setenv("SSL_CERT_DIR", saved_dir, True)
    good.stop()

    # 4. Host name mismatch (62) and IP mismatch (64).
    var other = Server(o, dir, "other", "", "-WWW")
    _expect_err(_get(client, "https://localhost:" + String(other.port) + "/hello.txt"), "hostname mismatch (62)")
    _expect_err(_get(client, "https://127.0.0.1:" + String(other.port) + "/hello.txt"), "IP address mismatch (64)")
    other.stop()

    # 5. Expired (10), self-signed leaf (18), untrusted root in the chain (19).
    var expired = Server(o, dir, "expired", "", "-WWW")
    _expect_err(_get(client, "https://localhost:" + String(expired.port) + "/hello.txt"), "certificate has expired (10)")
    expired.stop()
    var selfs = Server(o, dir, "self", "", "-WWW")
    _expect_err(_get(client, "https://localhost:" + String(selfs.port) + "/hello.txt"), "self-signed certificate (18)")
    selfs.stop()
    var chain = Server(o, dir, "untrusted", "ca2", "-WWW")
    _expect_err(_get(client, "https://localhost:" + String(chain.port) + "/hello.txt"),
                "self-signed certificate in certificate chain (19)")
    chain.stop()

    # 6. Truncation: the server is killed in the middle of a large body.
    var blob = String()
    var line = String("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcde\n")
    for _ in range(4096):
        blob += line
    var big = String()
    for _ in range(32):
        big += blob   # 8 MiB
    write_text(dir + "/big.txt", big)
    write_text(dir + "/big.http", "HTTP/1.0 200 ok\r\nContent-Type: text/plain\r\nContent-Length: "
               + String(big.byte_length()) + "\r\n\r\n" + big)
    var cases: List[String] = ["-HTTP big.http", "-WWW big.txt", "-WWW big.txt ragged-ok"]
    for k in range(len(cases)):
        var parts = cases[k].split(" ")
        var srv = Server(o, dir, "leaf", "", String(parts[0]))
        var ragged_ok = len(parts) == 3
        var stream = client.open(HttpRequest("GET", "https://localhost:" + String(srv.port) + "/" + String(parts[1]),
                                             List[String](), List[String](), ""), _opts(ragged_ok))
        assert_equal(stream.status, 200)
        var got = List[UInt8]()
        _ = stream.read_chunk(got, 3000)
        srv.kill()
        var err = String()
        try:
            while stream.read_chunk(got, 3000):
                pass
        except e:
            err = String(e)
        assert_true(len(got) < big.byte_length(), "the body must be cut: " + cases[k])
        if k == 0:
            assert_true(err.startswith("QA-E104: body truncated (") and err.endswith(" bytes missing)"),
                        "Content-Length body: " + err)
        elif k == 1:
            assert_equal(err, "QA-E104: body truncated (TLS connection closed without close_notify)")
        else:
            assert_equal(err, "", "accept_ragged_eof keeps the short body")
        _ = srv.child.wait()

    # 7. Handshake timeout against a listener that never accepts.
    var silent = TcpListener("127.0.0.1", 0)
    var t0 = monotonic_ms()
    var terr = String()
    try:
        _ = connect_stream(parse_url("https://127.0.0.1:" + String(silent.port()) + "/"), test_ctx, 1500)
    except e:
        terr = String(e)
    var elapsed = monotonic_ms() - t0
    assert_true(terr.startswith("QA-E104: TLS handshake with 127.0.0.1 failed: timed out"), terr)
    assert_true(elapsed >= 1000 and elapsed <= 2000, "handshake timeout took " + String(elapsed) + " ms")
    _ = silent.port()

    remove_tree(dir)
    print("test_net_tls_local OK")
