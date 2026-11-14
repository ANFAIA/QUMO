"""Tests for net/http.mojo framing and request heads over a scripted `CannedStream`
(no sockets): chunked with extensions and trailers, Content-Length, close-delimited,
1xx, HEAD/204/304, truncation, ragged TLS EOF, header caps, injection refused, body cap."""

from std.testing import assert_equal, assert_true, assert_false

from qumo_agent.net.socket import ByteStream
from qumo_agent.net.url import parse_url
from qumo_agent.net.http import (
    HttpRequest, HttpOptions, HttpResponse, request_head, read_response,
    REDIRECT_NONE,
)


struct CannedStream(ByteStream):
    """Serves fixed pieces, then end of stream; `ragged` scripts a TLS EOF without close_notify."""

    var pieces: List[List[UInt8]]
    var next: Int
    var ragged: Bool
    var reads: Int

    def __init__(out self, text: String, piece_size: Int, ragged: Bool):
        self.pieces = List[List[UInt8]]()
        var bs = text.as_bytes()
        var i = 0
        while i < len(bs):
            var p = List[UInt8]()
            var end = min(len(bs), i + piece_size)
            for k in range(i, end):
                p.append(bs[k])
            self.pieces.append(p^)
            i = end
        self.next = 0
        self.ragged = ragged
        self.reads = 0

    def write_all(mut self, data: String, timeout_ms: Int) raises:
        pass

    def read_some(mut self, mut buf: List[UInt8], timeout_ms: Int) raises -> Int:
        self.reads += 1
        if self.next >= len(self.pieces):
            return 0
        var p = self.pieces[self.next].copy()
        self.next += 1
        var n = len(p)
        buf.extend(p^)
        return n

    def ended_ragged(self) -> Bool:
        return self.ragged and self.next >= len(self.pieces)

    def close(mut self):
        pass


def _opts(max_body: Int, truncate: Bool, ragged_ok: Bool) -> HttpOptions:
    return HttpOptions(1000, 1000, max_body, truncate, REDIRECT_NONE, 10, True, ragged_ok)


def _read(text: String, piece: Int, method: String, opts: HttpOptions, ragged: Bool) raises -> HttpResponse:
    var s = CannedStream(text, piece, ragged)
    return read_response(s, method, opts)


def _body_all_splits(text: String, method: String, want_status: Int, want_body: String) raises:
    """Same result whole, in 7-byte pieces and byte by byte."""
    var sizes: List[Int] = [1 << 30, 7, 1]
    for k in range(len(sizes)):
        var r = _read(text, sizes[k], method, _opts(1 << 20, False, False), False)
        assert_equal(r.status, want_status)
        assert_equal(r.text(), want_body, "piece size " + String(sizes[k]))
        assert_false(r.truncated)


def _error_all_splits(text: String, method: String, opts: HttpOptions, ragged: Bool, want: String) raises:
    var sizes: List[Int] = [1 << 30, 5, 1]
    for k in range(len(sizes)):
        var got = String()
        try:
            _ = _read(text, sizes[k], method, opts, ragged)
        except e:
            got = String(e)
        assert_equal(got, want, "piece size " + String(sizes[k]))


def test_framing() raises:
    # Chunked with an extension and a trailer wins over Content-Length; bytes after the
    # terminating chunk are never read as body.
    var chunked = ("HTTP/1.1 200 OK\r\nTransfer-Encoding: gzip, Chunked\r\nContent-Length: 999\r\n\r\n"
                   + "4;ext=1\r\nWiki\r\n5\r\npedia\r\nE\r\n in\r\n\r\nchunks.\r\n0;last\r\nX-Trailer: a\r\n\r\nGARBAGE")
    _body_all_splits(chunked, "GET", 200, "Wikipedia in\r\n\r\nchunks.")
    var s = CannedStream(chunked + "MORE", 1, False)
    _ = read_response(s, "GET", _opts(1 << 20, False, False))
    assert_true(s.next < len(s.pieces), "the stream must not be read to its end after the last chunk")
    # Upper-case hex sizes, no trailers.
    _body_all_splits("HTTP/1.1 200 OK\r\ntransfer-encoding: chunked\r\n\r\n1A\r\nabcdefghijklmnopqrstuvwxyz\r\n0\r\n\r\n",
                     "POST", 200, "abcdefghijklmnopqrstuvwxyz")
    # Content-Length: exactly that many bytes, the rest ignored.
    _body_all_splits("HTTP/1.1 201 Created\r\nContent-Length: 5\r\n\r\nhello world", "POST", 201, "hello")
    _body_all_splits("HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n", "GET", 200, "")
    # Close-delimited (HTTP/1.0 style), LF-only line endings.
    _body_all_splits("HTTP/1.0 200 OK\nServer: x\n\nline 1\nline 2\n", "GET", 200, "line 1\nline 2\n")
    # 1xx skipped.
    _body_all_splits("HTTP/1.1 100 Continue\r\n\r\nHTTP/1.1 103 Early Hints\r\nLink: </a>\r\n\r\n"
                     + "HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok", "POST", 200, "ok")
    # No body for HEAD, 204 and 304, whatever the headers say.
    _body_all_splits("HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\n", "HEAD", 200, "")
    _body_all_splits("HTTP/1.1 204 No Content\r\nContent-Length: 10\r\n\r\nxxxxxxxxxx", "GET", 204, "")
    _body_all_splits("HTTP/1.1 304 Not Modified\r\nTransfer-Encoding: chunked\r\n\r\n", "GET", 304, "")
    # Transfer-Encoding without chunked last: read to close.
    _body_all_splits("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked, gzip\r\n\r\nraw", "GET", 200, "raw")


def test_headers() raises:
    var r = _read("HTTP/1.1 429 Too Many Requests\r\nRetry-After-Ms:  50 \r\nX-Multi: a\r\nx-multi: b\r\n"
                  + "X-Folded: one\r\n  two\r\nContent-Length: 2\r\n\r\n{}", 3, "POST", _opts(100, False, False), False)
    assert_equal(r.status, 429)
    assert_equal(r.reason, "Too Many Requests")
    assert_equal(r.header("retry-after-ms"), "50")
    assert_equal(r.header("RETRY-AFTER-MS"), "50")
    assert_equal(r.header("x-multi"), "a")
    assert_equal(r.header("x-folded"), "one two")
    assert_equal(r.header("absent"), "")
    assert_equal(r.header_names[0], "retry-after-ms")
    var noreason = _read("HTTP/1.1 200\r\nContent-Length: 0\r\n\r\n", 100, "GET", _opts(100, False, False), False)
    assert_equal(noreason.status, 200)
    assert_equal(noreason.reason, "")


def test_errors() raises:
    var strict = _opts(1 << 20, False, False)
    _error_all_splits("HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\nabc", "GET", strict, False,
                      "QA-E104: body truncated (7 bytes missing)")
    _error_all_splits("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nab", "GET", strict, False,
                      "QA-E104: body truncated (3 bytes missing)")
    _error_all_splits("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nabcde\r\n", "GET", strict, False,
                      "QA-E104: body truncated (chunked body ended before its last chunk)")
    _error_all_splits("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\nzz\r\n", "GET", strict, False,
                      "QA-E104: invalid chunk size")
    _error_all_splits("", "GET", strict, False, "QA-E104: connection closed before the response was received")
    _error_all_splits("HTTP/1.1 200 OK\r\nContent-Le", "GET", strict, False,
                      "QA-E104: connection closed inside the response headers")
    _error_all_splits("SSH-2.0-OpenSSH\r\n\r\n", "GET", strict, False, "QA-E104: invalid HTTP status line: SSH-2.0-OpenSSH")
    # A malformed status line with multi-byte characters is cut at a code point boundary
    # (a cut inside one used to abort the process).
    var accents = String()
    for _ in range(45):
        accents += "é"
    var want_accents = String()
    for _ in range(35):
        want_accents += "é"
    _error_all_splits("XTTP/1.1 " + accents + "\r\n\r\n", "GET", strict, False,
                      "QA-E104: invalid HTTP status line: XTTP/1.1 " + want_accents)
    _error_all_splits("HTTP/1.1 20" + accents + "\r\n\r\n", "GET", strict, False,
                      "QA-E104: invalid HTTP status line: HTTP/1.1 20" + String("é") * 34)
    var multibyte_reason = _read("HTTP/1.1 200é ok\r\nContent-Length: 0\r\n\r\n", 100, "GET", strict, False)
    assert_equal(multibyte_reason.status, 200)
    assert_equal(multibyte_reason.reason, "é ok")
    _error_all_splits("HTTP/1.1 200 OK\r\nContent-Length: 3\r\nContent-Length: 4\r\n\r\nabcd", "GET", strict, False,
                      "QA-E104: conflicting Content-Length headers")
    # Close-delimited body on a TLS stream that ended without close_notify (X15).
    var close_body = "HTTP/1.1 200 OK\r\n\r\npartial json"
    _error_all_splits(close_body, "GET", strict, True,
                      "QA-E104: body truncated (TLS connection closed without close_notify)")
    var lenient = _read(close_body, 4, "GET", _opts(1 << 20, False, True), True)
    assert_equal(lenient.text(), "partial json")
    # A complete Content-Length or chunked body is fine even if the TLS EOF is ragged afterwards.
    var ok = _read("HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok", 4, "GET", strict, True)
    assert_equal(ok.text(), "ok")
    # Header caps: 100 lines allowed, 101 refused; 64 KiB block.
    var head100 = String("HTTP/1.1 200 OK\r\n")
    for i in range(99):
        head100 += "X-H" + String(i) + ": v\r\n"
    var r100 = _read(head100 + "Content-Length: 1\r\n\r\nz", 1000, "GET", strict, False)
    assert_equal(r100.text(), "z")
    _error_all_splits(head100 + "X-Extra: 1\r\nContent-Length: 1\r\n\r\nz", "GET", strict, False,
                      "QA-E104: more than 100 response header lines")
    var big = String("HTTP/1.1 200 OK\r\nX-Big: ")
    for _ in range(70000):
        big += "a"
    _error_all_splits(big + "\r\n\r\n", "GET", strict, False, "QA-E104: response header line longer than 65536 bytes")
    var many = String("HTTP/1.1 200 OK\r\n")
    for i in range(20):
        many += "X-" + String(i) + ": "
        for _ in range(4000):
            many += "b"
        many += "\r\n"
    _error_all_splits(many + "\r\n", "GET", strict, False, "QA-E104: response header block larger than 65536 bytes")


def test_body_cap() raises:
    var text = "HTTP/1.1 200 OK\r\nContent-Length: 12\r\n\r\nhello world!"
    var cut = _read(text, 3, "GET", _opts(5, True, False), False)
    assert_equal(cut.text(), "hello")
    assert_true(cut.truncated)
    var exact = _read(text, 3, "GET", _opts(12, False, False), False)
    assert_equal(exact.text(), "hello world!")
    assert_false(exact.truncated)
    _error_all_splits(text, "GET", _opts(5, False, False), False, "QA-E104: response body larger than 5 bytes")
    _error_all_splits("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\nc\r\nhello world!\r\n0\r\n\r\n", "GET",
                      _opts(11, False, False), False, "QA-E104: response body larger than 11 bytes")


def test_request_head() raises:
    var names: List[String] = ["Authorization", "Content-Type", "X-Title"]
    var values: List[String] = ["Bearer k", "application/json", "QUMO agents"]
    var req = HttpRequest("POST", "https://openrouter.ai/api/v1/chat/completions", names^, values^, '{"a": "é"}')
    var head = request_head(req, parse_url(req.url))
    assert_equal(head, "POST /api/v1/chat/completions HTTP/1.1\r\nHost: openrouter.ai\r\n"
                 + "Authorization: Bearer k\r\nContent-Type: application/json\r\nX-Title: QUMO agents\r\n"
                 + "Accept-Encoding: identity\r\nConnection: close\r\nContent-Length: 11\r\n\r\n")
    # GET without a body: no Content-Length; caller's Connection/Accept-Encoding kept, Host replaced.
    var n2: List[String] = ["connection", "Accept-Encoding", "Host"]
    var v2: List[String] = ["keep-alive", "gzip", "evil.example"]
    var get = HttpRequest("GET", "http://127.0.0.1:8080/x?y=1", n2^, v2^, "")
    assert_equal(request_head(get, parse_url(get.url)),
                 "GET /x?y=1 HTTP/1.1\r\nHost: 127.0.0.1:8080\r\nconnection: keep-alive\r\nAccept-Encoding: gzip\r\n\r\n")
    # An empty POST still sends Content-Length: 0.
    var post0 = HttpRequest("POST", "http://h/", List[String](), List[String](), "")
    assert_true(request_head(post0, parse_url(post0.url)).endswith("Content-Length: 0\r\n\r\n"))
    # Injection refused.
    var bad_values: List[String] = ["x\r\nX-Injected: 1"]
    var bad_names: List[String] = ["X-Ok"]
    var colon_names: List[String] = ["X-Bad:"]
    var plain: List[String] = ["v"]
    var cases = List[HttpRequest]()
    cases.append(HttpRequest("GET", "http://h/", bad_names^, bad_values^, ""))
    cases.append(HttpRequest("GET", "http://h/", colon_names^, plain^, ""))
    cases.append(HttpRequest("GET\r\nX: y", "http://h/", List[String](), List[String](), ""))
    for i in range(len(cases)):
        var got = String()
        try:
            _ = request_head(cases[i], parse_url("http://h/"))
        except e:
            got = String(e)
        assert_equal(got, "QA-E104: header injection refused", "case " + String(i))


def main() raises:
    test_framing()
    test_headers()
    test_errors()
    test_body_cap()
    test_request_head()
    print("test_net_http_framing OK")
