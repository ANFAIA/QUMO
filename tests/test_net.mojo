"""Tests for qumo/net as the hardware clients use it (`HttpClient(None)`, REDIRECT_NONE,
QM-E959 / QM-E951), with no process spawned and nothing outside 127.0.0.1.

Framing runs over a scripted stream; the socket cases use an in-process `TcpListener`:
one request and response over a real connection, a redirect that REDIRECT_NONE returns
as it is, a read timeout, a refused connection, a TLS handshake that times out, and the
bound on interim (1xx) responses. The fixture-server and `openssl s_server` cases of the
same path are in agents/mojo/tests/test_net_qumo.mojo; the full suite of the stack runs on
the generated agent copy (agents/mojo/tests/test_net_*.mojo, tests/test_net_mirror.mojo).
"""

from std.memory import ArcPointer
from std.testing import assert_equal, assert_true, assert_false
from std.time import perf_counter_ns

from qumo.net.errors import E_NETWORK, E_TLS_SETUP
from qumo.net.guard import UrlGuard, no_guard, literal_is_global
from qumo.net.http import (
    HttpClient, HttpOptions, HttpRequest, HttpResponse, request_head, read_response,
    MAX_INTERIM_RESPONSES, REDIRECT_NONE,
)
from qumo.net.socket import ByteStream, NetStream, TcpListener, connect_stream, tcp_connect
from qumo.net.tls import TlsContext
from qumo.net.url import parse_url, is_loopback_host


struct CannedStream(ByteStream):
    """Serves `text` in pieces of `piece` bytes, then end of stream."""

    var data: List[UInt8]
    var pos: Int
    var piece: Int

    def __init__(out self, text: String, piece: Int):
        self.data = List[UInt8]()
        for b in text.as_bytes():
            self.data.append(b)
        self.pos = 0
        self.piece = piece

    def write_all(mut self, data: String, timeout_ms: Int) raises:
        pass

    def read_some(mut self, mut buf: List[UInt8], timeout_ms: Int) raises -> Int:
        var n = min(self.piece, len(self.data) - self.pos)
        for i in range(n):
            buf.append(self.data[self.pos + i])
        self.pos += n
        return n

    def ended_ragged(self) -> Bool:
        return False

    def close(mut self):
        pass


def _hw_opts(io_ms: Int) -> HttpOptions:
    """The options of the hardware clients (qumo/exec/*_api.mojo) with a short I/O timeout."""
    return HttpOptions(2000, io_ms, 1 << 20, False, REDIRECT_NONE, 0, True, False)


def _canned(text: String, piece: Int) -> String:
    """`<status> <body>` or `ERR <error>`."""
    var s = CannedStream(text, piece)
    try:
        var r = read_response(s, "GET", _hw_opts(1000))
        return String(r.status) + " " + r.text()
    except e:
        return "ERR " + String(e)


def test_codes() raises:
    assert_equal(E_NETWORK, "QM-E959")
    assert_equal(E_TLS_SETUP, "QM-E951")
    var bad = String()
    try:
        _ = parse_url("ftp://example.org/")
    except e:
        bad = String(e)
    assert_true(bad.startswith("QM-E959: invalid URL"), bad)
    assert_equal(_canned("HTTP/1.1 200 OK\r\nContent-Length: 9\r\n\r\nab", 1), "ERR QM-E959: body truncated (7 bytes missing)")
    assert_equal(_canned("HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nab", 1), "200 ab")


def test_interim_responses_are_bounded() raises:
    """1xx responses are skipped up to MAX_INTERIM_RESPONSES; a server cannot hold the client
    in the head loop with an endless run of them."""
    assert_equal(MAX_INTERIM_RESPONSES, 16)
    var final = "HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok"
    var sizes: List[Int] = [1 << 30, 7, 1]
    for k in range(len(sizes)):
        assert_equal(_canned("HTTP/1.1 100 Continue\r\n\r\nHTTP/1.1 103 Early Hints\r\nLink: </a>\r\n\r\n" + final, sizes[k]), "200 ok")
        assert_equal(_canned(String("HTTP/1.1 100 Continue\r\n\r\n") * 16 + final, sizes[k]), "200 ok")
        assert_equal(_canned(String("HTTP/1.1 100 Continue\r\n\r\n") * 17 + final, sizes[k]),
                     "ERR QM-E959: more than 16 interim (1xx) responses")
    # The reported case: 200 000 interim responses. The 17th is refused; the rest is never read.
    var flood = CannedStream(String("HTTP/1.1 102 Processing\r\n\r\n") * 200000 + final, 4096)
    var got = String()
    try:
        _ = read_response(flood, "GET", _hw_opts(1000))
    except e:
        got = String(e)
    assert_equal(got, "QM-E959: more than 16 interim (1xx) responses")
    assert_true(flood.pos <= 2 * 4096, "read " + String(flood.pos) + " bytes of the flood")


def test_guard_and_loopback() raises:
    assert_true(is_loopback_host("localhost") and is_loopback_host("127.0.0.1") and is_loopback_host("::1"))
    assert_false(is_loopback_host("127.0.0.1.attacker.example"))
    assert_false(no_guard().enabled)
    # NAT64-embedded loopback, IPv4-compatible and multicast are refused in the core guard too
    for host in ["64:ff9b::7f00:1", "::127.0.0.1", "224.0.0.1", "ff02::1"]:
        assert_equal(literal_is_global(host), 0, host)
    assert_equal(literal_is_global("64:ff9b::808:808"), 1, "NAT64 of a public address")
    # public_only: a loopback address is refused before a socket is opened.
    var got = String()
    try:
        _ = tcp_connect("127.0.0.1", 9, 500, True)
    except e:
        got = String(e)
    assert_equal(got, "QM-E959: refused URL: 127.0.0.1 is not a public address")


def _read_request(mut peer: NetStream) raises -> String:
    var buf = List[UInt8]()
    while True:
        var text = String(from_utf8_lossy=Span(buf))
        if text.find("\r\n\r\n") >= 0:
            return text
        if peer.read_some(buf, 2000) == 0:
            return text


def test_exchange_over_a_socket() raises:
    """Request head and response over a real 127.0.0.1 connection, both ends in this process."""
    var listener = TcpListener("127.0.0.1", 0)
    var base = "http://127.0.0.1:" + String(listener.port())
    var url = parse_url(base + "/v1/jobs?x=1")
    var req = HttpRequest("GET", base + "/v1/jobs?x=1", ["Authorization"], ["apiKey k"], "")
    var stream = connect_stream(url, None, 2000)
    var peer = listener.accept(2000)
    stream.write_all(request_head(req, url), 2000)
    var seen = _read_request(peer)
    assert_true(seen.startswith("GET /v1/jobs?x=1 HTTP/1.1\r\n"), seen)
    assert_true(seen.find("Authorization: apiKey k\r\n") >= 0, seen)
    # A redirect is the response under REDIRECT_NONE: status and Location, no second request.
    peer.write_all("HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1:1/elsewhere\r\nContent-Length: 4\r\n\r\nmove", 2000)
    var r = read_response(stream, "GET", _hw_opts(2000))
    assert_equal(r.status, 302)
    assert_equal(r.header("location"), "http://127.0.0.1:1/elsewhere")
    assert_equal(r.text(), "move")
    peer.close()
    stream.close()


def _now_ms() -> Int:
    return Int(perf_counter_ns()) // 1000000


def _released_port() raises -> Int:
    var gone = TcpListener("127.0.0.1", 0)
    return gone.port()


def test_timeouts_and_refusals() raises:
    var client = HttpClient(None)
    # Read timeout: the connection is accepted by the kernel, nobody answers.
    var quiet = TcpListener("127.0.0.1", 0)
    var t0 = _now_ms()
    var got = String()
    try:
        _ = client.request(HttpRequest("GET", "http://127.0.0.1:" + String(quiet.port()) + "/", List[String](), List[String](), ""),
                           _hw_opts(300))
    except e:
        got = String(e)
    assert_equal(got, "QM-E959: read timed out")
    var ms = _now_ms() - t0
    assert_true(ms >= 200 and ms <= 2000, "read timeout took " + String(ms) + " ms")
    # Refused connection: a port that was bound and released.
    var dead_port = _released_port()
    got = String()
    try:
        _ = client.request(HttpRequest("GET", "http://127.0.0.1:" + String(dead_port) + "/", List[String](), List[String](), ""),
                           _hw_opts(300))
    except e:
        got = String(e)
    assert_true(got.startswith("QM-E959: 127.0.0.1:" + String(dead_port) + ": "), got)
    # Plain http is refused unless the caller allows it (the hardware clients allow it for loopback only).
    var strict = HttpOptions(2000, 300, 1 << 20, False, REDIRECT_NONE, 0, False, False)
    got = String()
    try:
        _ = client.request(HttpRequest("GET", "http://127.0.0.1:" + String(quiet.port()) + "/", List[String](), List[String](), ""), strict)
    except e:
        got = String(e)
    assert_true(got.startswith("QM-E959: "), got)
    _ = quiet.port()


def test_tls_handshake_timeout() raises:
    """TLS to a host that never answers: QM-E959 after the connect timeout. Without an
    OpenSSL 3 library on this machine the context cannot be made (QM-E951) and the case is skipped."""
    var made = List[ArcPointer[TlsContext]]()
    try:
        made.append(ArcPointer(TlsContext()))
    except e:
        assert_true(String(e).startswith("QM-E951: "), String(e))
        print("test_net: TLS case skipped (" + String(e) + ")")
        return
    var ctx: Optional[ArcPointer[TlsContext]] = made[0]
    var silent = TcpListener("127.0.0.1", 0)
    var t0 = _now_ms()
    var got = String()
    try:
        _ = connect_stream(parse_url("https://127.0.0.1:" + String(silent.port()) + "/"), ctx, 800)
    except e:
        got = String(e)
    var ms = _now_ms() - t0
    assert_true(got.startswith("QM-E959: TLS handshake with 127.0.0.1 failed: timed out"), got)
    assert_true(ms >= 500 and ms <= 2500, "handshake timeout took " + String(ms) + " ms")
    _ = silent.port()


def main() raises:
    test_codes()
    test_interim_responses_are_bounded()
    test_guard_and_loopback()
    test_exchange_over_a_socket()
    test_timeouts_and_refusals()
    test_tls_handshake_timeout()
    print("test_net: OK")
