"""HTTP/1.1 client: request heads, response framing, redirects (DESIGN.md 2.4.5).

Every request is sent with `Connection: close` (one request per connection) and
`Accept-Encoding: identity` unless the caller set them. Response framing follows
RFC 9112: 1xx responses are skipped; HEAD, 204 and 304 have no body; a
`Transfer-Encoding` ending in `chunked` wins over `Content-Length` (sizes in hex,
`;` extensions ignored, trailers read up to the empty line, then the body ends without
waiting for the connection to close); a `Content-Length` body is read exactly (an
earlier end of stream is a truncation error); otherwise the body runs to the end of
the stream, and a TLS stream that ended without `close_notify` is a truncation error
unless the request accepts it (`accept_ragged_eof`, research GETs only, X15).

Redirects (D2c): at most `max_redirects` hops and 4 visits of one URL, https to http
is always refused, and `Authorization` and `Cookie` are dropped when the scheme, host
or port change.
"""

from std.memory import ArcPointer

from qumo_agent.core.errors import E_NETWORK
from qumo_agent.net.url import Url, parse_url, resolve_reference
from qumo_agent.net.socket import ByteStream, NetStream, connect_stream
from qumo_agent.net.tls import TlsContext

comptime REDIRECT_NONE = 0
comptime REDIRECT_URLLIB = 1        # research GETs: 301/302/303/307/308, D2c rules
comptime REDIRECT_SAME_ORIGIN = 2   # MCP streamable HTTP: 307/308 to the same origin only
comptime REDIRECT_KEEP_METHOD = 3   # model POST: 307/308, method and body kept, D2c rules

comptime MAX_HEADER_BYTES = 65536
comptime MAX_HEADER_LINES = 100
comptime MAX_VISITS_PER_URL = 4
comptime _MAX_CHUNK_LINE = 4096

comptime _FRAME_NONE = 0
comptime _FRAME_LENGTH = 1
comptime _FRAME_CHUNKED = 2
comptime _FRAME_CLOSE = 3

comptime _CH_SIZE = 0
comptime _CH_DATA = 1
comptime _CH_DATA_END = 2
comptime _CH_TRAILERS = 3


@fieldwise_init
struct HttpRequest(Copyable):
    var method: String
    var url: String
    var header_names: List[String]
    var header_values: List[String]
    var body: String


@fieldwise_init
struct HttpOptions(Copyable):
    var connect_timeout_ms: Int
    var io_timeout_ms: Int
    var max_body_bytes: Int
    var truncate_at_max: Bool      # True: stop silently (urllib read(n)); False: raise QA-E104 "response body larger than N bytes"
    var redirect_mode: Int
    var max_redirects: Int         # 10
    var allow_plain_http: Bool
    var accept_ragged_eof: Bool    # True only for research GETs (urllib parity); False for the model client and MCP HTTP (X15)


def _lower(s: String) -> String:
    var bs = s.as_bytes()
    var out = List[UInt8](capacity=len(bs))
    for i in range(len(bs)):
        var c = Int(bs[i])
        if c >= 0x41 and c <= 0x5A:
            c += 0x20
        out.append(UInt8(c))
    return String(from_utf8_lossy=Span(out))


def _find_header(names: List[String], values: List[String], name: String) -> String:
    var want = _lower(name)
    for i in range(min(len(names), len(values))):
        if _lower(names[i]) == want:
            return values[i].copy()
    return String()


@fieldwise_init
struct HttpResponse(Copyable):
    var status: Int
    var reason: String
    var header_names: List[String]     # lower-case
    var header_values: List[String]
    var body: List[UInt8]
    var final_url: String
    var truncated: Bool

    def header(self, name: String) -> String:
        """First value of header `name` (case-insensitive), "" when absent."""
        return _find_header(self.header_names, self.header_values, name)

    def text(self) -> String:
        """The body as lossy UTF-8."""
        return String(from_utf8_lossy=Span(self.body))


# ---------------------------------------------------------------------------
# Request head
# ---------------------------------------------------------------------------
def _has_ctl(s: String) -> Bool:
    var bs = s.as_bytes()
    for i in range(len(bs)):
        var c = Int(bs[i])
        if c == 0x0D or c == 0x0A or c == 0:
            return True
    return False


def _valid_name(s: String) -> Bool:
    var bs = s.as_bytes()
    if len(bs) == 0:
        return False
    for i in range(len(bs)):
        var c = Int(bs[i])
        if c <= 0x20 or c >= 0x7F or c == 0x3A:
            return False
    return True


def request_head(req: HttpRequest, url: Url) raises -> String:
    """`METHOD path_query HTTP/1.1`, `Host`, the caller's headers, then
    `Accept-Encoding: identity` and `Connection: close` unless given, and
    `Content-Length` when there is a body or the method is POST/PUT/PATCH.
    CR or LF in the method, the URL or any header raises QA-E104."""
    if _has_ctl(req.method) or _has_ctl(url.path_query) or _has_ctl(url.host) or not _valid_name(req.method):
        raise Error(E_NETWORK + ": header injection refused")
    var head = req.method + " " + url.path_query + " HTTP/1.1\r\nHost: " + url.host_header() + "\r\n"
    var has_ae = False
    var has_conn = False
    for i in range(min(len(req.header_names), len(req.header_values))):
        var n = req.header_names[i]
        var v = req.header_values[i]
        if not _valid_name(n) or _has_ctl(v):
            raise Error(E_NETWORK + ": header injection refused")
        var ln = _lower(n)
        if ln == "host" or ln == "content-length":
            continue
        if ln == "accept-encoding":
            has_ae = True
        if ln == "connection":
            has_conn = True
        head += n + ": " + v + "\r\n"
    if not has_ae:
        head += "Accept-Encoding: identity\r\n"
    if not has_conn:
        head += "Connection: close\r\n"
    if req.body.byte_length() > 0 or req.method == "POST" or req.method == "PUT" or req.method == "PATCH":
        head += "Content-Length: " + String(req.body.byte_length()) + "\r\n"
    head += "\r\n"
    return head^


# ---------------------------------------------------------------------------
# Response framing
# ---------------------------------------------------------------------------
struct _Reader(Movable):
    """Bytes read from the stream and not consumed yet: buf[pos:]."""

    var buf: List[UInt8]
    var pos: Int
    var eof: Bool

    def __init__(out self):
        self.buf = List[UInt8]()
        self.pos = 0
        self.eof = False

    def available(self) -> Int:
        return len(self.buf) - self.pos

    def compact(mut self):
        if self.pos > 0 and self.pos >= len(self.buf) // 2:
            var rest = List[UInt8](capacity=len(self.buf) - self.pos + 65536)
            for i in range(self.pos, len(self.buf)):
                rest.append(self.buf[i])
            self.buf = rest^
            self.pos = 0


def _fill[S: ByteStream](mut s: S, mut r: _Reader, timeout_ms: Int) raises -> Int:
    """Reads more bytes into `r`; 0 at the end of the stream."""
    if r.eof:
        return 0
    r.compact()
    var n = s.read_some(r.buf, timeout_ms)
    if n == 0:
        r.eof = True
    return n


def _read_line[S: ByteStream](mut s: S, mut r: _Reader, timeout_ms: Int, limit: Int, mut line: String) raises -> Bool:
    """Next line without its CRLF/LF into `line`; False when the stream ends before the
    line is complete. A line longer than `limit` raises QA-E104."""
    var scan = r.pos
    while True:
        var found = -1
        for i in range(scan, len(r.buf)):
            if Int(r.buf[i]) == 0x0A:
                found = i
                break
        if found >= 0:
            var end = found
            if end > r.pos and Int(r.buf[end - 1]) == 0x0D:
                end -= 1
            if end - r.pos > limit:
                raise Error(E_NETWORK + ": response header line longer than " + String(limit) + " bytes")
            line = String(from_utf8_lossy=Span(r.buf)[r.pos:end])
            r.pos = found + 1
            return True
        if r.available() > limit:
            raise Error(E_NETWORK + ": response header line longer than " + String(limit) + " bytes")
        var scanned = len(r.buf) - r.pos   # compaction in _fill may move the unread bytes
        var n = _fill(s, r, timeout_ms)
        scan = r.pos + scanned
        if n == 0:
            return False   # end of stream before a complete line


@fieldwise_init
struct _Head(Copyable):
    var status: Int
    var reason: String
    var names: List[String]
    var values: List[String]


def _boundary_at_or_before(s: String, n: Int) -> Int:
    """The largest code point boundary <= n (a server may send anything, and a byte
    cut inside a multi-byte character aborts the process)."""
    var bs = s.as_bytes()
    var i = min(n, len(bs))
    while i > 0 and i < len(bs) and (Int(bs[i]) & 0xC0) == 0x80:
        i -= 1
    return i


def _status_excerpt(line: String) -> String:
    """At most the first 80 bytes of `line`, cut at a code point boundary."""
    return String(line[byte=0:_boundary_at_or_before(line, 80)])


def _parse_status(line: String) raises -> Tuple[Int, String]:
    if not line.startswith("HTTP/1.") or line.byte_length() < 12:
        raise Error(E_NETWORK + ": invalid HTTP status line: " + _status_excerpt(line))
    var bs = line.as_bytes()
    if Int(bs[8]) != 0x20:
        raise Error(E_NETWORK + ": invalid HTTP status line: " + _status_excerpt(line))
    var code = 0
    for i in range(9, 12):
        var c = Int(bs[i])
        if c < 0x30 or c > 0x39:
            raise Error(E_NETWORK + ": invalid HTTP status line: " + _status_excerpt(line))
        code = code * 10 + (c - 0x30)
    var reason = String()
    if line.byte_length() > 12:
        var from_byte = _boundary_at_or_before(line, 12)   # byte 12 is not checked: it may start a multi-byte character
        reason = String(String(line[byte=from_byte:]).strip())
    return (code, reason^)


def _read_head[S: ByteStream](mut s: S, mut r: _Reader, timeout_ms: Int) raises -> _Head:
    """Status line and headers of the final (non-1xx) response."""
    while True:
        var line = String()
        var used = 0
        if not _read_line(s, r, timeout_ms, MAX_HEADER_BYTES, line):
            raise Error(E_NETWORK + ": connection closed before the response was received")
        used += line.byte_length() + 2
        var st = _parse_status(line)
        var names = List[String]()
        var values = List[String]()
        var count = 0
        while True:
            var h = String()
            if not _read_line(s, r, timeout_ms, MAX_HEADER_BYTES, h):
                raise Error(E_NETWORK + ": connection closed inside the response headers")
            used += h.byte_length() + 2
            if used > MAX_HEADER_BYTES:
                raise Error(E_NETWORK + ": response header block larger than " + String(MAX_HEADER_BYTES) + " bytes")
            if h.byte_length() == 0:
                break
            count += 1
            if count > MAX_HEADER_LINES:
                raise Error(E_NETWORK + ": more than " + String(MAX_HEADER_LINES) + " response header lines")
            if (h.startswith(" ") or h.startswith("\t")) and len(values) > 0:
                # Obsolete line folding: continue the previous value.
                values[len(values) - 1] = values[len(values) - 1] + " " + String(h.strip())
                continue
            var colon = h.find(":")
            if colon <= 0:
                raise Error(E_NETWORK + ": invalid response header line")
            names.append(_lower(String(String(h[byte=0:colon]).strip())))
            values.append(String(String(h[byte=colon + 1:]).strip()))
        if st[0] >= 100 and st[0] < 200:
            continue
        return _Head(st[0], st[1], names^, values^)


struct _Body(Copyable):
    """Framing state of a response body being read."""

    var mode: Int
    var remaining: Int          # _FRAME_LENGTH: bytes still expected
    var phase: Int              # _FRAME_CHUNKED: _CH_*
    var chunk_left: Int
    var done: Bool
    var total: Int              # body bytes delivered so far
    var ragged_ok: Bool

    def __init__(out self, mode: Int, length: Int, ragged_ok: Bool):
        self.mode = mode
        self.remaining = length
        self.phase = _CH_SIZE
        self.chunk_left = 0
        self.done = mode == _FRAME_NONE or (mode == _FRAME_LENGTH and length == 0)
        self.total = 0
        self.ragged_ok = ragged_ok


def _framing(head: _Head, method: String, ragged_ok: Bool) raises -> _Body:
    if method == "HEAD" or head.status == 204 or head.status == 304 or (head.status >= 100 and head.status < 200):
        return _Body(_FRAME_NONE, 0, ragged_ok)
    var te = String()
    var has_te = False
    var cl = String()
    var has_cl = False
    for i in range(len(head.names)):
        if head.names[i] == "transfer-encoding":
            if has_te:
                te += ","
            te += head.values[i]
            has_te = True
        elif head.names[i] == "content-length":
            if has_cl and _lower(head.values[i]) != cl:
                raise Error(E_NETWORK + ": conflicting Content-Length headers")
            cl = _lower(head.values[i])
            has_cl = True
    if has_te:
        var parts = te.split(",")
        var last = _lower(String(String(parts[len(parts) - 1]).strip()))
        if last == "chunked":
            return _Body(_FRAME_CHUNKED, 0, ragged_ok)
        return _Body(_FRAME_CLOSE, 0, ragged_ok)
    if has_cl:
        var bs = cl.as_bytes()
        if len(bs) == 0 or len(bs) > 18:
            raise Error(E_NETWORK + ": invalid Content-Length")
        var n = 0
        for i in range(len(bs)):
            var c = Int(bs[i])
            if c < 0x30 or c > 0x39:
                raise Error(E_NETWORK + ": invalid Content-Length")
            n = n * 10 + (c - 0x30)
        return _Body(_FRAME_LENGTH, n, ragged_ok)
    return _Body(_FRAME_CLOSE, 0, ragged_ok)


def _hex_value(c: Int) -> Int:
    if c >= 0x30 and c <= 0x39:
        return c - 0x30
    if c >= 0x61 and c <= 0x66:
        return c - 0x61 + 10
    if c >= 0x41 and c <= 0x46:
        return c - 0x41 + 10
    return -1


def _parse_chunk_size(line: String) raises -> Int:
    var text = line.copy()
    var semi = text.find(";")
    if semi >= 0:
        text = String(line[byte=0:semi])
    var t = String(text.strip())
    var bs = t.as_bytes()
    if len(bs) == 0 or len(bs) > 15:
        raise Error(E_NETWORK + ": invalid chunk size")
    var n = 0
    for i in range(len(bs)):
        var v = _hex_value(Int(bs[i]))
        if v < 0:
            raise Error(E_NETWORK + ": invalid chunk size")
        n = n * 16 + v
    return n


def _take(mut r: _Reader, mut out: List[UInt8], n: Int) -> Int:
    """Moves up to `n` buffered bytes to `out`; returns how many."""
    var k = min(n, r.available())
    for i in range(k):
        out.append(r.buf[r.pos + i])
    r.pos += k
    return k


def _body_next[S: ByteStream](mut s: S, mut r: _Reader, mut st: _Body, mut out: List[UInt8], timeout_ms: Int) raises -> Bool:
    """Appends the next body bytes to `out` (at least one); False at the end of the body."""
    while not st.done:
        if st.mode == _FRAME_LENGTH:
            if r.available() == 0 and _fill(s, r, timeout_ms) == 0:
                raise Error(E_NETWORK + ": body truncated (" + String(st.remaining) + " bytes missing)")
            var k = _take(r, out, st.remaining)
            st.remaining -= k
            st.total += k
            if st.remaining == 0:
                st.done = True
            return True
        if st.mode == _FRAME_CLOSE:
            if r.available() == 0 and _fill(s, r, timeout_ms) == 0:
                st.done = True
                if s.ended_ragged() and not st.ragged_ok:
                    raise Error(E_NETWORK + ": body truncated (TLS connection closed without close_notify)")
                return False
            var k = _take(r, out, r.available())
            st.total += k
            return True
        # Chunked.
        if st.phase == _CH_SIZE:
            var line = String()
            if not _read_line(s, r, timeout_ms, _MAX_CHUNK_LINE, line):
                raise Error(E_NETWORK + ": body truncated (chunked body ended before its last chunk)")
            st.chunk_left = _parse_chunk_size(line)
            st.phase = _CH_TRAILERS if st.chunk_left == 0 else _CH_DATA
            continue
        if st.phase == _CH_DATA:
            if r.available() == 0 and _fill(s, r, timeout_ms) == 0:
                raise Error(E_NETWORK + ": body truncated (" + String(st.chunk_left) + " bytes missing)")
            var k = _take(r, out, st.chunk_left)
            st.chunk_left -= k
            st.total += k
            if st.chunk_left == 0:
                st.phase = _CH_DATA_END
            return True
        if st.phase == _CH_DATA_END:
            var line = String()
            if not _read_line(s, r, timeout_ms, _MAX_CHUNK_LINE, line):
                raise Error(E_NETWORK + ": body truncated (chunked body ended before its last chunk)")
            if line.byte_length() != 0:
                raise Error(E_NETWORK + ": invalid chunked body (missing CRLF after chunk data)")
            st.phase = _CH_SIZE
            continue
        # Trailers, up to the empty line.
        var used = 0
        while True:
            var t = String()
            if not _read_line(s, r, timeout_ms, MAX_HEADER_BYTES, t):
                raise Error(E_NETWORK + ": body truncated (chunked body ended inside its trailers)")
            used += t.byte_length() + 2
            if used > MAX_HEADER_BYTES:
                raise Error(E_NETWORK + ": response trailers larger than " + String(MAX_HEADER_BYTES) + " bytes")
            if t.byte_length() == 0:
                break
        st.done = True
    return False


def _read_body[S: ByteStream](mut s: S, mut r: _Reader, mut st: _Body, opts: HttpOptions, mut body: List[UInt8]) raises -> Bool:
    """Reads the whole body into `body` under the size cap; returns `truncated`."""
    while True:
        var piece = List[UInt8]()
        if not _body_next(s, r, st, piece, opts.io_timeout_ms):
            return False
        if len(body) + len(piece) > opts.max_body_bytes:
            if not opts.truncate_at_max:
                raise Error(E_NETWORK + ": response body larger than " + String(opts.max_body_bytes) + " bytes")
            var room = opts.max_body_bytes - len(body)
            for i in range(room):
                body.append(piece[i])
            return True
        body.extend(piece^)


def read_response[S: ByteStream](mut s: S, method: String, opts: HttpOptions) raises -> HttpResponse:
    """Reads one response (head and framed body) from `s`. `final_url` is left empty."""
    var r = _Reader()
    var head = _read_head(s, r, opts.io_timeout_ms)
    var st = _framing(head, method, opts.accept_ragged_eof)
    var body = List[UInt8]()
    var truncated = _read_body(s, r, st, opts, body)
    return HttpResponse(head.status, head.reason.copy(), head.names.copy(), head.values.copy(), body^, String(), truncated)


# ---------------------------------------------------------------------------
# Redirects
# ---------------------------------------------------------------------------
def keeps_authorization(current: Url, target: Url) -> Bool:
    """True when scheme, host and port are unchanged (Authorization and Cookie kept)."""
    return current.scheme == target.scheme and current.host == target.host and current.port == target.port


def _is_redirect_status(status: Int) -> Bool:
    return status == 301 or status == 302 or status == 303 or status == 307 or status == 308


def _follows(status: Int, method: String, mode: Int) -> Bool:
    """Whether `mode` follows a `status` response to `method` (before the origin check)."""
    if mode == REDIRECT_URLLIB:
        if _is_redirect_status(status) and (method == "GET" or method == "HEAD"):
            return True
        return (status == 301 or status == 302 or status == 303) and method == "POST"
    if mode == REDIRECT_SAME_ORIGIN or mode == REDIRECT_KEEP_METHOD:
        return status == 307 or status == 308
    return False


def redirect_target(current: Url, location: String, status: Int, method: String, mode: Int) raises -> Url:
    """The URL a redirect leads to. https to http is always refused (QA-E104)."""
    if mode == REDIRECT_NONE or not _is_redirect_status(status):
        raise Error(E_NETWORK + ": not a followed redirect: HTTP " + String(status))
    var loc = String(location.strip())
    if loc.byte_length() == 0:
        raise Error(E_NETWORK + ": redirect without a Location header")
    var target = resolve_reference(current, loc)
    if current.scheme == "https" and target.scheme == "http":
        raise Error(E_NETWORK + ": refusing redirect from https to http")
    return target^


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
struct HttpBodyStream(Movable):
    """A response whose body is read incrementally (SSE). Redirects were already
    resolved; `read_chunk` applies the request's body cap and framing rules."""

    var status: Int
    var reason: String
    var header_names: List[String]
    var header_values: List[String]
    var final_url: String
    var _stream: NetStream
    var _reader: _Reader
    var _body: _Body
    var _max_body: Int
    var _truncate: Bool

    def __init__(out self, head: _Head, final_url: String, var stream: NetStream, var reader: _Reader,
                 var body: _Body, opts: HttpOptions):
        self.status = head.status
        self.reason = head.reason.copy()
        self.header_names = head.names.copy()
        self.header_values = head.values.copy()
        self.final_url = final_url
        self._stream = stream^
        self._reader = reader^
        self._body = body^
        self._max_body = opts.max_body_bytes
        self._truncate = opts.truncate_at_max

    def header(self, name: String) -> String:
        return _find_header(self.header_names, self.header_values, name)

    def read_chunk(mut self, mut out: List[UInt8], timeout_ms: Int) raises -> Bool:
        """Appends the next body bytes to `out`; False at the end of the body."""
        if self._body.done:
            return False
        var before = self._body.total
        var piece = List[UInt8]()
        if not _body_next(self._stream, self._reader, self._body, piece, timeout_ms):
            return False
        if before + len(piece) > self._max_body:
            if not self._truncate:
                raise Error(E_NETWORK + ": response body larger than " + String(self._max_body) + " bytes")
            for i in range(self._max_body - before):
                out.append(piece[i])
            self._body.done = True
            self.close()
            return True
        out.extend(piece^)
        return True

    def close(mut self):
        self._body.done = True
        self._stream.close()


struct _Exchange(Movable):
    var head: _Head
    var url: Url
    var method: String
    var stream: NetStream
    var reader: _Reader

    def __init__(out self, var head: _Head, var url: Url, method: String, var stream: NetStream, var reader: _Reader):
        self.head = head^
        self.url = url^
        self.method = method
        self.stream = stream^
        self.reader = reader^

    def into_body_stream(deinit self, var body: _Body, opts: HttpOptions) -> HttpBodyStream:
        var final_url = self.url.to_string()
        return HttpBodyStream(self.head, final_url, self.stream^, self.reader^, body^, opts)


def _without(names: List[String], values: List[String], drop: List[String],
             mut out_names: List[String], mut out_values: List[String]):
    out_names.clear()
    out_values.clear()
    for i in range(min(len(names), len(values))):
        var ln = _lower(names[i])
        var skip = False
        for d in drop:
            if ln == d:
                skip = True
                break
        if not skip:
            out_names.append(names[i].copy())
            out_values.append(values[i].copy())


struct HttpClient(Movable):
    var tls: Optional[ArcPointer[TlsContext]]
    var requests: Int                            # counters for RunStats

    def __init__(out self, tls: Optional[ArcPointer[TlsContext]]):
        """`tls` may be None: the production context is then created (and OpenSSL
        loaded) on the first https request only."""
        self.tls = tls
        self.requests = 0

    def _tls_for(mut self, url: Url) raises -> Optional[ArcPointer[TlsContext]]:
        if url.scheme != "https":
            return None
        if not self.tls:
            self.tls = ArcPointer(TlsContext())
        return self.tls

    def _exchange(mut self, req: HttpRequest, opts: HttpOptions) raises -> _Exchange:
        """Sends `req`, following redirects per `opts`, and returns the final response
        head with its unread body on the open connection."""
        var url = parse_url(req.url)
        var method = req.method.copy()
        var body = req.body.copy()
        var names = req.header_names.copy()
        var values = req.header_values.copy()
        var visits = Dict[String, Int]()
        var hops = 0
        while True:
            if url.scheme == "http" and not opts.allow_plain_http:
                raise Error(E_NETWORK + ": refusing plain http URL " + url.to_string())
            var hop_req = HttpRequest(method, url.to_string(), names.copy(), values.copy(), body)
            var head_text = request_head(hop_req, url)
            var tls = self._tls_for(url)
            var stream = connect_stream(url, tls, opts.connect_timeout_ms)
            self.requests += 1
            stream.write_all(head_text + body, opts.io_timeout_ms)
            var reader = _Reader()
            var head = _read_head(stream, reader, opts.io_timeout_ms)
            var location = _find_header(head.names, head.values, "location")
            if not _follows(head.status, method, opts.redirect_mode) or location.byte_length() == 0:
                return _Exchange(head^, url^, method, stream^, reader^)
            var target = redirect_target(url, location, head.status, method, opts.redirect_mode)
            if opts.redirect_mode == REDIRECT_SAME_ORIGIN and target.origin() != url.origin():
                return _Exchange(head^, url^, method, stream^, reader^)
            stream.close()
            hops += 1
            var key = target.to_string()
            var seen = visits.get(key, 0)
            if hops > opts.max_redirects or seen >= MAX_VISITS_PER_URL:
                raise Error(E_NETWORK + ": redirect loop")
            visits[key] = seen + 1
            var drop = List[String]()
            if not keeps_authorization(url, target):
                drop.append("authorization")
                drop.append("cookie")
            if opts.redirect_mode == REDIRECT_URLLIB and head.status != 307 and head.status != 308:
                # urllib: 301/302/303 become GET (HEAD stays HEAD) without a body.
                method = "HEAD" if method == "HEAD" else "GET"
                body = String()
                drop.append("content-length")
                drop.append("content-type")
            var nn = List[String]()
            var nv = List[String]()
            _without(names, values, drop, nn, nv)
            names = nn^
            values = nv^
            url = target^

    def request(mut self, req: HttpRequest, opts: HttpOptions) raises -> HttpResponse:
        """One request with redirects; the whole body is read under `opts`' cap."""
        var ex = self._exchange(req, opts)
        var st = _framing(ex.head, ex.method, opts.accept_ragged_eof)
        var body = List[UInt8]()
        var truncated: Bool
        try:
            truncated = _read_body(ex.stream, ex.reader, st, opts, body)
        except e:
            ex.stream.close()
            raise e^
        ex.stream.close()
        return HttpResponse(ex.head.status, ex.head.reason.copy(), ex.head.names.copy(), ex.head.values.copy(),
                            body^, ex.url.to_string(), truncated)

    def open(mut self, req: HttpRequest, opts: HttpOptions) raises -> HttpBodyStream:
        """Like `request`, but the body is streamed (redirects resolved before)."""
        var ex = self._exchange(req, opts)
        var st = _framing(ex.head, ex.method, opts.accept_ragged_eof)
        return ex^.into_body_stream(st^, opts)
