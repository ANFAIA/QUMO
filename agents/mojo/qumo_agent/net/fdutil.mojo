"""Small socket helpers shared by net/socket.mojo and net/tls.mojo (all libc calls go
through native/libc.mojo)."""

from std.time import perf_counter_ns

from qumo_agent.native.libc import (
    CStr, buf_addr, store_u32, store_u16, store_i64, load_u16,
    c_setsockopt, c_fcntl, c_poll, c_inet_pton, errno,
    SOL_SOCKET, SO_RCVTIMEO, SO_SNDTIMEO, SO_NOSIGPIPE, F_GETFD, F_SETFD, FD_CLOEXEC,
    AF_INET, AF_INET6, EINTR, SIZEOF_TIMEVAL, TIMEVAL_USEC_OFFSET, TIMEVAL_USEC_BYTES, is_macos,
)


def now_ms() -> Int:
    """Monotonic milliseconds (perf_counter_ns() // 1_000_000)."""
    return Int(perf_counter_ns() // 1_000_000)


def set_io_timeouts(fd: Int, timeout_ms: Int) -> Int:
    """SO_RCVTIMEO and SO_SNDTIMEO = `timeout_ms` (at least 1 ms; 0 would mean "never")."""
    var ms = timeout_ms if timeout_ms > 0 else 1
    var tv = List[UInt8](length=SIZEOF_TIMEVAL, fill=0)
    var a = buf_addr(tv)
    store_i64(a, ms // 1000)
    var usec = (ms % 1000) * 1000
    if TIMEVAL_USEC_BYTES == 4:
        store_u32(a + TIMEVAL_USEC_OFFSET, usec)
    else:
        store_i64(a + TIMEVAL_USEC_OFFSET, usec)
    var r1 = c_setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, a, SIZEOF_TIMEVAL)
    var r2 = c_setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, a, SIZEOF_TIMEVAL)
    _ = len(tv)
    return -1 if (r1 != 0 or r2 != 0) else 0


def set_cloexec(fd: Int):
    var fl = c_fcntl(fd, F_GETFD, 0)
    if fl >= 0:
        _ = c_fcntl(fd, F_SETFD, fl | FD_CLOEXEC)


def set_nosigpipe(fd: Int):
    """macOS: SO_NOSIGPIPE, so a write to a closed peer gives EPIPE instead of SIGPIPE."""
    if is_macos():
        var one = List[UInt8](length=4, fill=0)
        store_u32(buf_addr(one), 1)
        _ = c_setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, buf_addr(one), 4)
        _ = len(one)


def poll_one(fd: Int, events: Int, timeout_ms: Int) -> Int:
    """poll() on one descriptor, retrying EINTR within the budget.
    Returns the revents (> 0), 0 on timeout, -1 on error."""
    var deadline = now_ms() + timeout_ms
    var pfd = List[UInt8](length=8, fill=0)
    while True:
        var a = buf_addr(pfd)
        store_u32(a, fd)
        store_u16(a + 4, events)
        store_u16(a + 6, 0)
        var left = deadline - now_ms()
        if left < 0:
            left = 0
        var r = c_poll(a, 1, left)
        if r > 0:
            var rev = load_u16(a + 6)
            _ = len(pfd)
            return rev if rev > 0 else events
        if r == 0:
            _ = len(pfd)
            return 0
        if errno() != EINTR:
            _ = len(pfd)
            return -1


def ip_literal_family(host: String) -> Int:
    """AF_INET or AF_INET6 when `host` is an IP literal (inet_pton accepts it), else 0."""
    var c = CStr(host)
    var dst = List[UInt8](length=16, fill=0)
    var fam = 0
    if c_inet_pton(AF_INET, c.addr(), buf_addr(dst)) == 1:
        fam = AF_INET
    elif c_inet_pton(AF_INET6, c.addr(), buf_addr(dst)) == 1:
        fam = AF_INET6
    c.keep()
    _ = len(dst)
    return fam
