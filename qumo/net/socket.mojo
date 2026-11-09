"""TCP and TLS byte streams over libc sockets (agents/mojo/DESIGN.md 2.4.2).

`tcp_connect` resolves with `getaddrinfo` and tries the addresses in order with a
non-blocking `connect` + `poll`; every attempt but the last gets at most 10 s of the
connect budget (openrouter.ai resolves to IPv6 first). DNS itself has no timeout.

`read_some`/`write_all` set `SO_RCVTIMEO`/`SO_SNDTIMEO` before each call: the timeout
is per operation (httpx's semantics), not a total deadline. An expiry raises
`QM-E959: read timed out` / `QM-E959: write timed out`.

TLS reads: a `close_notify` ends the stream (0); an unexpected EOF (no `close_notify`,
OpenSSL reason 294, or a bare EOF reported as `SSL_ERROR_SYSCALL`) also returns 0 and
sets `ragged_eof`, which HTTP framing turns into a truncation error unless the caller
accepts it (X15). After a fatal TLS error no `SSL_shutdown` is sent.
"""

from std.memory import ArcPointer

from qumo.net.errors import E_NETWORK, E_TLS_SETUP
from qumo.net.libc import (
    CStr, buf_addr, load_i32, load_i64, load_u16, load_u32, store_u32, c_string_at, errno, strerror_text,
    c_getaddrinfo, c_freeaddrinfo, c_gai_strerror, c_socket, c_connect, c_close, c_fcntl,
    c_getsockopt, c_setsockopt, c_read, c_write, c_bind, c_listen, c_accept, c_getsockname,
    AF_UNSPEC, SOCK_STREAM, SOL_SOCKET, SO_ERROR, SO_REUSEADDR, F_GETFL, F_SETFL, O_NONBLOCK,
    POLLIN, POLLOUT, EINTR, EAGAIN, EINPROGRESS, ECONNRESET, EPIPE,
    SIZEOF_ADDRINFO, AI_FAMILY_OFFSET, AI_SOCKTYPE_OFFSET, AI_PROTOCOL_OFFSET, AI_ADDRLEN_OFFSET,
    AI_ADDR_OFFSET, AI_NEXT_OFFSET,
)
from qumo.net.ossl import (
    SSL_ERROR_ZERO_RETURN, SSL_ERROR_WANT_READ, SSL_ERROR_WANT_WRITE, SSL_ERROR_SYSCALL,
    SSL_R_UNEXPECTED_EOF_WHILE_READING, SSL_R_CERTIFICATE_VERIFY_FAILED,
)
from qumo.net.tls import TlsContext, tls_handshake
from qumo.net.url import Url
from qumo.net.fdutil import now_ms, set_io_timeouts, set_cloexec, set_nosigpipe, poll_one, ip_literal_family

comptime READ_CHUNK = 65536
comptime _MAX_ATTEMPT_MS = 10000


trait ByteStream(Movable, Deinitable):
    def write_all(mut self, data: String, timeout_ms: Int) raises:
        ...

    def read_some(mut self, mut buf: List[UInt8], timeout_ms: Int) raises -> Int:
        """Appends the bytes read to `buf` and returns their count; 0 = end of stream."""
        ...

    def ended_ragged(self) -> Bool:
        """The end of stream was a TLS EOF without close_notify (always False for plain
        TCP and test streams unless scripted)."""
        ...

    def close(mut self):
        ...


def _close_stream(fd: Int, ssl: Int, tls: Optional[ArcPointer[TlsContext]], fatal: Bool):
    if ssl != 0 and tls:
        try:
            ref lib = tls.value()[].lib
            if not fatal:
                lib.ERR_clear_error()
                _ = lib.SSL_shutdown(ssl)
                lib.ERR_clear_error()
            lib.SSL_free(ssl)
        except:
            pass
    if fd >= 0:
        _ = c_close(fd)


struct NetStream(ByteStream):
    var fd: Int
    var ssl: Int                                  # 0 for plain TCP
    var tls: Optional[ArcPointer[TlsContext]]
    var fatal: Bool                               # no SSL_shutdown after a fatal error [tls §8]
    var ragged_eof: Bool                          # peer closed without close_notify

    def __init__(out self, fd: Int, ssl: Int, tls: Optional[ArcPointer[TlsContext]]):
        self.fd = fd
        self.ssl = ssl
        self.tls = tls
        self.fatal = False
        self.ragged_eof = False

    def write_all(mut self, data: String, timeout_ms: Int) raises:
        if self.fd < 0:
            raise Error(E_NETWORK + ": write on a closed connection")
        _ = set_io_timeouts(self.fd, timeout_ms)
        var bs = data.as_bytes()
        var base = Int(bs.unsafe_ptr())
        var total = len(bs)
        var off = 0
        if self.ssl == 0:
            while off < total:
                var n = c_write(self.fd, base + off, total - off)
                if n > 0:
                    off += n
                    continue
                var e = errno()
                if n < 0 and e == EINTR:
                    continue
                if n < 0 and e == EAGAIN:
                    raise Error(E_NETWORK + ": write timed out")
                raise Error(E_NETWORK + ": write failed: " + strerror_text(e))
            return
        ref lib = self.tls.value()[].lib
        var written = List[UInt8](length=8, fill=0)
        while off < total:
            lib.ERR_clear_error()
            var r = lib.SSL_write_ex(self.ssl, base + off, total - off, buf_addr(written))
            if r == 1:
                off += load_i64(buf_addr(written))
                _ = len(written)   # keep `written` alive until the load (ASAP destruction)
                continue
            var se = lib.SSL_get_error(self.ssl, r)
            var e = errno()
            var q = lib.errors()
            if se == SSL_ERROR_WANT_READ or se == SSL_ERROR_WANT_WRITE:
                raise Error(E_NETWORK + ": write timed out")
            self.fatal = True
            var detail = q if q.byte_length() > 0 else strerror_text(e)
            raise Error(E_NETWORK + ": TLS write failed: " + detail)
        _ = len(written)

    def read_some(mut self, mut buf: List[UInt8], timeout_ms: Int) raises -> Int:
        if self.fd < 0:
            return 0
        _ = set_io_timeouts(self.fd, timeout_ms)
        var old = len(buf)
        buf.resize(old + READ_CHUNK, 0)
        var dst = buf_addr(buf) + old
        if self.ssl == 0:
            while True:
                var n = c_read(self.fd, dst, READ_CHUNK)
                if n >= 0:
                    buf.resize(old + n, 0)
                    return n
                var e = errno()
                if e == EINTR:
                    continue
                buf.resize(old, 0)
                if e == EAGAIN:
                    raise Error(E_NETWORK + ": read timed out")
                if e == ECONNRESET:
                    raise Error(E_NETWORK + ": connection reset by peer")
                raise Error(E_NETWORK + ": read failed: " + strerror_text(e))
        ref lib = self.tls.value()[].lib
        var got = List[UInt8](length=8, fill=0)
        lib.ERR_clear_error()
        var r = lib.SSL_read_ex(self.ssl, dst, READ_CHUNK, buf_addr(got))
        if r == 1:
            var n = load_i64(buf_addr(got))
            _ = len(got)   # keep `got` alive until the load (ASAP destruction)
            buf.resize(old + n, 0)
            return n
        buf.resize(old, 0)
        _ = len(got)
        var se = lib.SSL_get_error(self.ssl, r)
        var e = errno()
        var reasons = List[Int]()
        var q = lib.drain_errors(reasons)
        if se == SSL_ERROR_ZERO_RETURN:
            return 0
        if se == SSL_ERROR_WANT_READ or se == SSL_ERROR_WANT_WRITE:
            raise Error(E_NETWORK + ": read timed out")
        self.fatal = True
        for i in range(len(reasons)):
            if reasons[i] == SSL_R_UNEXPECTED_EOF_WHILE_READING:
                self.ragged_eof = True
                return 0
        if se == SSL_ERROR_SYSCALL and q.byte_length() == 0 and e == 0:
            self.ragged_eof = True
            return 0
        if se == SSL_ERROR_SYSCALL and q.byte_length() == 0:
            if e == ECONNRESET:
                raise Error(E_NETWORK + ": connection reset by peer")
            raise Error(E_NETWORK + ": TLS read failed: " + strerror_text(e))
        var detail = q
        for i in range(len(reasons)):
            if reasons[i] == SSL_R_CERTIFICATE_VERIFY_FAILED:
                var vr = lib.SSL_get_verify_result(self.ssl)
                detail += " (" + c_string_at(lib.X509_verify_cert_error_string(vr)) + ")"
                break
        raise Error(E_NETWORK + ": TLS read failed: " + detail)

    def ended_ragged(self) -> Bool:
        return self.ragged_eof

    def close(mut self):
        _close_stream(self.fd, self.ssl, self.tls, self.fatal)
        self.fd = -1
        self.ssl = 0

    def tls_version(self) -> String:
        if self.ssl == 0 or not self.tls:
            return String()
        try:
            return c_string_at(self.tls.value()[].lib.SSL_get_version(self.ssl))
        except:
            return String()

    def __deinit__(deinit self):
        _close_stream(self.fd, self.ssl, self.tls, self.fatal)


def is_ip_literal(host: String) -> Bool:
    return ip_literal_family(host) != 0


def _resolve(host: String, port: Int, passive: Bool) raises -> Int:
    """getaddrinfo(host, port, {SOCK_STREAM}); returns the list head (free with c_freeaddrinfo)."""
    var hints = List[UInt8](length=SIZEOF_ADDRINFO, fill=0)
    var ha = buf_addr(hints)
    if passive:
        store_u32(ha, 1)   # AI_PASSIVE
    store_u32(ha + AI_FAMILY_OFFSET, AF_UNSPEC)
    store_u32(ha + AI_SOCKTYPE_OFFSET, SOCK_STREAM)
    var res = List[UInt8](length=8, fill=0)
    var hc = CStr(host)
    var pc = CStr(String(port))
    var rc = c_getaddrinfo(hc.addr(), pc.addr(), ha, buf_addr(res))
    hc.keep()
    pc.keep()
    _ = len(hints)
    if rc != 0:
        raise Error(E_NETWORK + ": " + host + ": " + c_string_at(c_gai_strerror(rc)))
    var head = load_i64(buf_addr(res))
    _ = len(res)
    return head


def _connect_one(fd: Int, addr: Int, addrlen: Int, budget_ms: Int) -> String:
    """Non-blocking connect with a poll budget; "" on success, else the reason."""
    var fl = c_fcntl(fd, F_GETFL, 0)
    _ = c_fcntl(fd, F_SETFL, fl | O_NONBLOCK)
    var reason = String()
    var cr = c_connect(fd, addr, addrlen)
    if cr != 0:
        var e = errno()
        if e == EINPROGRESS or e == EINTR:
            var pr = poll_one(fd, POLLOUT, budget_ms)
            if pr == 0:
                reason = "connect timed out after " + String(budget_ms) + " ms"
            elif pr < 0:
                reason = "poll failed: " + strerror_text(errno())
            else:
                var err = List[UInt8](length=4, fill=0)
                var elen = List[UInt8](length=4, fill=0)
                store_u32(buf_addr(elen), 4)
                _ = c_getsockopt(fd, SOL_SOCKET, SO_ERROR, buf_addr(err), buf_addr(elen))
                var soerr = load_i32(buf_addr(err))
                _ = len(err)
                _ = len(elen)
                if soerr != 0:
                    reason = strerror_text(soerr)
        else:
            reason = strerror_text(e)
    if reason.byte_length() == 0:
        _ = c_fcntl(fd, F_SETFL, fl)
    return reason^


def tcp_connect(host: String, port: Int, timeout_ms: Int) raises -> Int:
    """A connected TCP socket to `host:port` (FD_CLOEXEC, SO_NOSIGPIPE on macOS)."""
    var head = _resolve(host, port, False)
    var deadline = now_ms() + timeout_ms
    var last = String("no addresses")
    var node = head
    var fd = -1
    while node != 0:
        var nxt = load_i64(node + AI_NEXT_OFFSET)
        var remaining = deadline - now_ms()
        if remaining <= 0:
            last = "connect timed out after " + String(timeout_ms) + " ms"
            break
        var budget = remaining if nxt == 0 else min(remaining, _MAX_ATTEMPT_MS)
        var s = c_socket(load_i32(node + AI_FAMILY_OFFSET), load_i32(node + AI_SOCKTYPE_OFFSET),
                         load_i32(node + AI_PROTOCOL_OFFSET))
        if s < 0:
            last = "socket: " + strerror_text(errno())
            node = nxt
            continue
        set_cloexec(s)
        set_nosigpipe(s)
        var why = _connect_one(s, load_i64(node + AI_ADDR_OFFSET), load_u32(node + AI_ADDRLEN_OFFSET), budget)
        if why.byte_length() == 0:
            fd = s
            break
        _ = c_close(s)
        last = why^
        node = nxt
    c_freeaddrinfo(head)
    if fd < 0:
        raise Error(E_NETWORK + ": " + host + ":" + String(port) + ": " + last)
    return fd


def connect_stream(url: Url, tls: Optional[ArcPointer[TlsContext]], connect_timeout_ms: Int) raises -> NetStream:
    """TCP connection to `url`'s host and port, with the TLS handshake for https."""
    if url.scheme == "https" and not tls:
        raise Error(E_TLS_SETUP + ": https://" + url.host_header() + " needs a TLS context")
    var fd = tcp_connect(url.host, url.port, connect_timeout_ms)
    if url.scheme != "https":
        return NetStream(fd, 0, None)
    var ssl: Int
    try:
        ssl = tls_handshake(tls.value(), fd, url.host, connect_timeout_ms)
    except e:
        _ = c_close(fd)
        raise e^
    return NetStream(fd, ssl, tls)


struct TcpListener(Movable):
    """A listening TCP socket (tests and the fixture server only)."""

    var fd: Int

    def __init__(out self, host: String, port: Int) raises:
        """Binds `host:port` (port 0 = ephemeral) and listens."""
        var head = _resolve(host, port, True)
        var s = c_socket(load_i32(head + AI_FAMILY_OFFSET), load_i32(head + AI_SOCKTYPE_OFFSET),
                         load_i32(head + AI_PROTOCOL_OFFSET))
        if s < 0:
            var e = errno()
            c_freeaddrinfo(head)
            raise Error(E_NETWORK + ": socket: " + strerror_text(e))
        set_cloexec(s)
        var one = List[UInt8](length=4, fill=0)
        store_u32(buf_addr(one), 1)
        _ = c_setsockopt(s, SOL_SOCKET, SO_REUSEADDR, buf_addr(one), 4)
        _ = len(one)
        var br = c_bind(s, load_i64(head + AI_ADDR_OFFSET), load_u32(head + AI_ADDRLEN_OFFSET))
        var be = errno()
        c_freeaddrinfo(head)
        if br != 0:
            _ = c_close(s)
            raise Error(E_NETWORK + ": bind " + host + ":" + String(port) + ": " + strerror_text(be))
        if c_listen(s, 64) != 0:
            var le = errno()
            _ = c_close(s)
            raise Error(E_NETWORK + ": listen: " + strerror_text(le))
        self.fd = s

    def port(self) -> Int:
        """The bound port (sin_port / sin6_port: big-endian at offset 2 on both platforms)."""
        var sa = List[UInt8](length=128, fill=0)
        var slen = List[UInt8](length=4, fill=0)
        store_u32(buf_addr(slen), 128)
        if c_getsockname(self.fd, buf_addr(sa), buf_addr(slen)) != 0:
            return -1
        var p = (Int(sa[2]) << 8) | Int(sa[3])
        _ = len(slen)
        return p

    def accept(mut self, timeout_ms: Int) raises -> NetStream:
        """The next connection (plain), or QM-E959 after `timeout_ms`."""
        var pr = poll_one(self.fd, POLLIN, timeout_ms)
        if pr == 0:
            raise Error(E_NETWORK + ": accept timed out")
        if pr < 0:
            raise Error(E_NETWORK + ": poll failed: " + strerror_text(errno()))
        while True:
            var c = c_accept(self.fd, 0, 0)
            if c >= 0:
                set_cloexec(c)
                set_nosigpipe(c)
                # An accepted socket may inherit O_NONBLOCK on some systems: clear it.
                var fl = c_fcntl(c, F_GETFL, 0)
                if fl >= 0 and (fl & O_NONBLOCK) != 0:
                    _ = c_fcntl(c, F_SETFL, fl & ~O_NONBLOCK)
                return NetStream(c, 0, None)
            var e = errno()
            if e != EINTR:
                raise Error(E_NETWORK + ": accept failed: " + strerror_text(e))

    def __deinit__(deinit self):
        if self.fd >= 0:
            _ = c_close(self.fd)
