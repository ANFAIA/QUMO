"""Child processes through posix_spawn with an explicit environment (DESIGN.md 2.1.2).

No shell is involved: the program is resolved in the parent (`which` over the PATH
of the child's own environment), argv and envp are NUL-terminated buffers, and the
environment is exactly `SpawnSpec.env` (D2b: children never inherit secrets).

File actions: pipe ends are dup2'd onto 0/1/2 (`STDIO_NULL` opens /dev/null), the
parent's pipe ends are closed, then `posix_spawn_file_actions_addchdir_np(cwd)`.
When the libc lacks the chdir action, `spawn` raises QA-E162 unless a chdir
trampoline was registered with `set_chdir_trampoline(binary)`: the child is then
`binary __exec-in <cwd> <path> <argv[1:]...>` (`exec_in_main`). On macOS
`POSIX_SPAWN_CLOEXEC_DEFAULT` keeps only 0/1/2 open in the child; on Linux every
descriptor we create is `FD_CLOEXEC` and `addclosefrom_np(3)` is added when glibc has
it. SIGPIPE is reset to its default in the child, and the signal mask is cleared.

`Child.read_line` and `Child.read_available` poll a piped stderr together with the
descriptor they wait on and keep its last 64 KiB (`stderr_tail`), so a child that
writes a lot to stderr never blocks while the parent waits for stdout.
"""

from std.os import getenv, setenv
from std.pathlib import cwd

from qumo_agent.native.shutdown import ignore_sigpipe_signal, register_child, unregister_child
from qumo_agent.native.libc import (
    CStr,
    CStrArray,
    buf_addr,
    load_i32,
    load_i64,
    load_u16,
    store_u16,
    store_u32,
    c_string_at,
    errno,
    strerror_text,
    has_symbol,
    is_macos,
    c_posix_spawn,
    c_fa_init,
    c_fa_destroy,
    c_fa_adddup2,
    c_fa_addclose,
    c_fa_addopen,
    fa_addchdir,
    fa_addclosefrom,
    c_attr_init,
    c_attr_destroy,
    c_attr_setflags,
    c_attr_setpgroup,
    c_attr_setsigdefault,
    c_attr_setsigmask,
    c_waitpid,
    c_kill,
    c_execve,
    c_chdir,
    c_pipe,
    c_fcntl,
    c_close,
    c_read,
    c_write,
    c_poll,
    c_ns_get_environ,
    c_access,
    POSIX_SPAWN_SETPGROUP,
    POSIX_SPAWN_SETSIGDEF,
    POSIX_SPAWN_SETSIGMASK,
    POSIX_SPAWN_CLOEXEC_DEFAULT,
    SIGPIPE,
    SIGTERM,
    SIGKILL,
    WNOHANG,
    O_RDONLY,
    O_WRONLY,
    O_NONBLOCK,
    F_GETFD,
    F_SETFD,
    F_GETFL,
    F_SETFL,
    FD_CLOEXEC,
    POLLIN,
    POLLOUT,
    POLLERR,
    POLLHUP,
    POLLNVAL,
    X_OK,
    EINTR,
    EAGAIN,
    EPIPE,
    ENOENT,
    ECHILD,
)
from qumo_agent.native.clock import monotonic_ms, sleep_ms
from qumo_agent.native.fs import py_oserror, is_dir


comptime STDIO_INHERIT = 0
comptime STDIO_PIPE = 1
comptime STDIO_NULL = 2     # /dev/null
comptime STDIO_MERGE = 3    # stderr only: into the stdout pipe

comptime READ_LINE = 1
comptime READ_EOF = 0
comptime READ_TIMEOUT = -1

comptime _RING_MAX = 65536          # stderr_tail keeps the last 64 KiB
comptime _READ_CHUNK = 65536
comptime _DEFAULT_PATH = "/bin:/usr/bin"   # os.defpath, when the child env has no PATH
comptime _TRAMPOLINE_ENV = "QUMO_AGENT_CHDIR_TRAMPOLINE"


@fieldwise_init
struct SpawnSpec(Copyable):
    var path: String            # absolute; relative with "/" (resolved against cwd); or a bare name (searched in PATH from `env`)
    var argv: List[String]      # argv[0] included
    var env: List[String]       # "NAME=value", exactly the child environment
    var cwd: String             # "" = inherit
    var stdin_mode: Int
    var stdout_mode: Int
    var stderr_mode: Int
    var new_process_group: Bool # child leads its own process group (kill(-pid, sig) reaches grandchildren)


@fieldwise_init
struct CaptureResult(Copyable):
    var exit_code: Int          # >= 0 exit status, -N terminated by signal N
    var timed_out: Bool
    var stdout: String          # lossy UTF-8; the last `max_bytes` bytes when longer
    var stderr: String
    var stdout_truncated: Bool
    var stderr_truncated: Bool


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _decode_status(s: Int) -> Int:
    """waitpid status: exit code (>= 0) or -signal."""
    if (s & 0x7F) == 0:
        return (s >> 8) & 0xFF
    if (s & 0xFF) == 0x7F:   # stopped (never requested); report as signal 0x7f
        return -0x7F
    return -(s & 0x7F)


def _write_fd_all(fd: Int, text: String):
    """Best-effort write of `text` to `fd` (used for stderr messages of trampolines)."""
    var bs = text.as_bytes()
    var p = Int(bs.unsafe_ptr())
    var n = len(bs)
    var off = 0
    while off < n:
        var w = c_write(fd, p + off, n - off)
        if w < 0:
            if errno() == EINTR:
                continue
            break
        off += w
    _ = text.byte_length()


def _env_value(env: List[String], name: String) -> Optional[String]:
    var prefix = name + "="
    for e in env:
        if e.startswith(prefix):
            return String(e[byte = prefix.byte_length() :])
    return None


def _make_pipe() raises -> Tuple[Int, Int]:
    var fds = List[UInt8](length=8, fill=0)
    var rc = c_pipe(buf_addr(fds))
    var e = errno()
    var r = load_i32(buf_addr(fds))
    var w = load_i32(buf_addr(fds) + 4)
    _ = len(fds)
    if rc != 0:
        raise Error("QA-E160: pipe failed: [Errno " + String(e) + "] " + strerror_text(e))
    _ = c_fcntl(r, F_SETFD, FD_CLOEXEC)
    _ = c_fcntl(w, F_SETFD, FD_CLOEXEC)
    return (r, w)


def _close_fd(fd: Int):
    if fd >= 0:
        _ = c_close(fd)


def _abs_cwd() -> String:
    try:
        return String(cwd())
    except:
        return String("/")


def which(name: String, path_env: String) -> String:
    """`shutil.which`-like lookup: the first executable regular file named `name` in
    the ':'-separated `path_env` (an empty entry means "."); "" when not found. A name
    containing "/" is checked as is."""
    if name.byte_length() == 0:
        return String()
    if name.find("/") >= 0:
        var c = CStr(name)
        var ok = c_access(c.addr(), X_OK) == 0
        c.keep()
        if ok and not is_dir(name):
            return name
        return String()
    for d in path_env.split(":"):
        var dir = String(d)
        if dir.byte_length() == 0:
            dir = "."
        var cand = dir + "/" + name if not dir.endswith("/") else dir + name
        var c = CStr(cand)
        var ok = c_access(c.addr(), X_OK) == 0
        c.keep()
        if ok and not is_dir(cand):
            return cand
    return String()


def resolve_program(path: String, env: List[String], cwd: String) raises -> String:
    """The program `spawn` would execute for `path`: a bare name through `which`
    with the PATH of `env` (os.defpath when unset); a relative path containing "/"
    joined to `cwd` (or the current directory); an absolute path unchanged.
    Raises `QA-E160: [Errno 2] No such file or directory: '<path>'` when a bare
    name is not found."""
    if path.byte_length() == 0:
        raise Error("QA-E160: " + py_oserror(ENOENT, path))
    if path.startswith("/"):
        return path
    if path.find("/") >= 0:
        var base = cwd
        if base.byte_length() == 0:
            base = _abs_cwd()
        elif not base.startswith("/"):
            base = _abs_cwd() + "/" + base
        return base + "/" + path if not base.endswith("/") else base + path
    var p = _env_value(env, "PATH")
    var path_env = p.value() if p else String(_DEFAULT_PATH)
    var found = which(path, path_env)
    if found.byte_length() == 0:
        raise Error("QA-E160: " + py_oserror(ENOENT, path))
    if not found.startswith("/"):
        var base = cwd
        if base.byte_length() == 0:
            base = _abs_cwd()
        found = base + "/" + found
    return found^


def set_chdir_trampoline(binary: String):
    """Registers `binary` (this executable, absolute) as the chdir trampoline used
    when the libc has no posix_spawn chdir action (glibc < 2.29). Stored in this
    process's environment (never passed to children: their env is explicit)."""
    _ = setenv(_TRAMPOLINE_ENV, binary)


# ---------------------------------------------------------------------------
# Child
# ---------------------------------------------------------------------------
struct Child(Movable):
    var pid: Int
    var stdin_fd: Int           # -1 when not piped
    var stdout_fd: Int
    var stderr_fd: Int
    var _group: Bool
    var _reaped: Bool
    var _code: Int
    var _pending: List[UInt8]   # stdout bytes read but not yet returned by read_line
    var _out_eof: Bool
    var _err_eof: Bool
    var _ring: List[UInt8]      # tail of a piped stderr

    def __init__(out self, pid: Int, stdin_fd: Int, stdout_fd: Int, stderr_fd: Int, group: Bool):
        self.pid = pid
        self.stdin_fd = stdin_fd
        self.stdout_fd = stdout_fd
        self.stderr_fd = stderr_fd
        self._group = group
        self._reaped = False
        self._code = 0
        self._pending = List[UInt8]()
        self._out_eof = False
        self._err_eof = False
        self._ring = List[UInt8]()

    # -- writing ------------------------------------------------------------
    def write_all(mut self, data: Span[Byte, _]) raises:
        """Writes every byte to the child's stdin; a closed pipe (EPIPE, SIGPIPE being
        ignored) raises `QA-E160: child closed its input`."""
        if self.stdin_fd < 0:
            raise Error("QA-E160: child closed its input")
        var p = Int(data.unsafe_ptr())
        var n = len(data)
        var off = 0
        while off < n:
            var w = c_write(self.stdin_fd, p + off, n - off)
            if w < 0:
                var e = errno()
                if e == EINTR:
                    continue
                if e == EAGAIN:
                    _ = self._poll_one(self.stdin_fd, POLLOUT, 100)
                    continue
                if e == EPIPE:
                    raise Error("QA-E160: child closed its input")
                raise Error("QA-E160: write to child failed: [Errno " + String(e) + "] " + strerror_text(e))
            off += w

    def close_stdin(mut self):
        if self.stdin_fd >= 0:
            _ = c_close(self.stdin_fd)
            self.stdin_fd = -1

    # -- reading ------------------------------------------------------------
    def _poll_one(mut self, fd: Int, events: Int, timeout_ms: Int) -> Int:
        var pfd = List[UInt8](length=8, fill=0)
        store_u32(buf_addr(pfd), fd)
        store_u16(buf_addr(pfd) + 4, events)
        var r = c_poll(buf_addr(pfd), 1, timeout_ms)
        _ = len(pfd)
        return r

    def _append_ring(mut self, data: List[UInt8], n: Int):
        for i in range(n):
            self._ring.append(data[i])
        if len(self._ring) > 2 * _RING_MAX:
            var keep = List[UInt8](capacity=_RING_MAX)
            var start = len(self._ring) - _RING_MAX
            for i in range(start, len(self._ring)):
                keep.append(self._ring[i])
            self._ring = keep^

    def _drain_stderr_once(mut self):
        """Reads what a piped stderr has (one read) into the ring; marks EOF."""
        if self.stderr_fd < 0 or self._err_eof:
            return
        var buf = List[UInt8](length=_READ_CHUNK, fill=0)
        var n = c_read(self.stderr_fd, buf_addr(buf), _READ_CHUNK)
        if n > 0:
            self._append_ring(buf, n)
        elif n == 0:
            self._err_eof = True
        elif errno() != EINTR and errno() != EAGAIN:
            self._err_eof = True
        _ = len(buf)

    def _wait_readable(mut self, fd: Int, timeout_ms: Int) -> Bool:
        """True when `fd` is readable (or at EOF/error), False at the timeout
        (`timeout_ms < 0` waits forever). Drains a piped stderr meanwhile."""
        var deadline = monotonic_ms() + timeout_ms
        var pfd = List[UInt8](length=16, fill=0)
        while True:
            var watch_err = self.stderr_fd >= 0 and not self._err_eof and fd != self.stderr_fd
            var a = buf_addr(pfd)
            store_u32(a, fd)
            store_u16(a + 4, POLLIN)
            store_u16(a + 6, 0)
            var nfds = 1
            if watch_err:
                store_u32(a + 8, self.stderr_fd)
                store_u16(a + 12, POLLIN)
                store_u16(a + 14, 0)
                nfds = 2
            var wait = -1
            if timeout_ms >= 0:
                wait = deadline - monotonic_ms()
                if wait < 0:
                    wait = 0
            var r = c_poll(a, nfds, wait)
            if r < 0:
                if errno() == EINTR:
                    continue
                _ = len(pfd)
                return True   # let the read report the error
            if r == 0:
                _ = len(pfd)
                if timeout_ms >= 0 and monotonic_ms() >= deadline:
                    return False
                continue
            var rev = load_u16(a + 6)
            if watch_err and load_u16(a + 14) != 0:
                self._drain_stderr_once()
            if rev != 0:
                _ = len(pfd)
                return True
            if timeout_ms >= 0 and monotonic_ms() >= deadline:
                _ = len(pfd)
                return False

    def read_line(mut self, mut line: String, timeout_ms: Int, max_bytes: Int) raises -> Int:
        """Next stdout line without its "\\n" (lossy UTF-8): READ_LINE, READ_EOF (no
        more data; a last unterminated line is returned first as READ_LINE) or
        READ_TIMEOUT (`timeout_ms < 0` waits forever). A line longer than
        `max_bytes` raises `QA-E161: line longer than <max_bytes> bytes`."""
        var deadline = monotonic_ms() + timeout_ms
        while True:
            var nl = -1
            for i in range(len(self._pending)):
                if Int(self._pending[i]) == 0x0A:
                    nl = i
                    break
            if nl >= 0:
                if nl > max_bytes:
                    raise Error("QA-E161: line longer than " + String(max_bytes) + " bytes")
                line = String(from_utf8_lossy=Span(self._pending)[0:nl])
                var rest = List[UInt8](capacity=len(self._pending) - nl - 1)
                for i in range(nl + 1, len(self._pending)):
                    rest.append(self._pending[i])
                self._pending = rest^
                return READ_LINE
            if len(self._pending) > max_bytes:
                raise Error("QA-E161: line longer than " + String(max_bytes) + " bytes")
            if self._out_eof or self.stdout_fd < 0:
                if len(self._pending) > 0:
                    line = String(from_utf8_lossy=Span(self._pending))
                    self._pending = List[UInt8]()
                    return READ_LINE
                return READ_EOF
            var remaining = -1
            if timeout_ms >= 0:
                remaining = deadline - monotonic_ms()
                if remaining <= 0:
                    return READ_TIMEOUT
            if not self._wait_readable(self.stdout_fd, remaining):
                return READ_TIMEOUT
            var buf = List[UInt8](length=_READ_CHUNK, fill=0)
            var n = c_read(self.stdout_fd, buf_addr(buf), _READ_CHUNK)
            if n > 0:
                for i in range(n):
                    self._pending.append(buf[i])
            elif n == 0:
                self._out_eof = True
            else:
                var e = errno()
                if e != EINTR and e != EAGAIN:
                    self._out_eof = True
            _ = len(buf)

    def read_available(mut self, fd: Int, mut out: List[UInt8], timeout_ms: Int) raises -> Int:
        """Appends what `fd` (stdout_fd or stderr_fd) has to `out`: the byte count,
        0 at EOF, -1 at the timeout. Bytes read from stderr also go to the tail."""
        if fd < 0:
            return 0
        if fd == self.stdout_fd and len(self._pending) > 0:
            var k = len(self._pending)
            for i in range(k):
                out.append(self._pending[i])
            self._pending = List[UInt8]()
            return k
        if fd == self.stdout_fd and self._out_eof:
            return 0
        if fd == self.stderr_fd and self._err_eof:
            return 0
        while True:
            if not self._wait_readable(fd, timeout_ms):
                return -1
            var buf = List[UInt8](length=_READ_CHUNK, fill=0)
            var n = c_read(fd, buf_addr(buf), _READ_CHUNK)
            if n < 0:
                var e = errno()
                _ = len(buf)
                if e == EINTR or e == EAGAIN:
                    continue
                raise Error("QA-E160: read from child failed: [Errno " + String(e) + "] " + strerror_text(e))
            for i in range(n):
                out.append(buf[i])
            if fd == self.stderr_fd:
                if n == 0:
                    self._err_eof = True
                else:
                    self._append_ring(buf, n)
            elif fd == self.stdout_fd and n == 0:
                self._out_eof = True
            _ = len(buf)
            return n

    def stderr_tail(self) -> String:
        """Lossy UTF-8 of the last 64 KiB drained from a piped stderr ("" otherwise)."""
        var n = len(self._ring)
        var start = n - _RING_MAX if n > _RING_MAX else 0
        return String(from_utf8_lossy=Span(self._ring)[start:n])

    # -- exit -------------------------------------------------------------
    def poll_exit(mut self) raises -> Optional[Int]:
        """waitpid(WNOHANG): the exit code (>= 0) or -signal once the child ended."""
        if self._reaped:
            return self._code
        var st = List[UInt8](length=8, fill=0)
        while True:
            var r = c_waitpid(self.pid, buf_addr(st), WNOHANG)
            if r == self.pid:
                self._code = _decode_status(load_i32(buf_addr(st)))
                self._reaped = True
                unregister_child(self.pid, self._group)
                _ = len(st)
                return self._code
            if r == 0:
                _ = len(st)
                return None
            var e = errno()
            if e == EINTR:
                continue
            _ = len(st)
            raise Error("QA-E160: waitpid failed: [Errno " + String(e) + "] " + strerror_text(e))

    def wait(mut self) raises -> Int:
        if self._reaped:
            return self._code
        var st = List[UInt8](length=8, fill=0)
        while True:
            var r = c_waitpid(self.pid, buf_addr(st), 0)
            if r == self.pid:
                self._code = _decode_status(load_i32(buf_addr(st)))
                self._reaped = True
                unregister_child(self.pid, self._group)
                _ = len(st)
                return self._code
            var e = errno()
            if e == EINTR:
                continue
            _ = len(st)
            raise Error("QA-E160: waitpid failed: [Errno " + String(e) + "] " + strerror_text(e))

    def signal_group(mut self, sig: Int):
        """kill(-pid, sig) when the child leads its own group, else kill(pid, sig).
        Does nothing once the child was reaped (its pid could be reused)."""
        if self._reaped or self.pid <= 0:
            return
        if self._group:
            _ = c_kill(-self.pid, sig)
        else:
            _ = c_kill(self.pid, sig)

    def _wait_for(mut self, grace_ms: Int) raises -> Bool:
        var deadline = monotonic_ms() + grace_ms
        while True:
            if self.poll_exit():
                return True
            if monotonic_ms() >= deadline:
                return False
            sleep_ms(10)

    def terminate(mut self, grace_ms: Int) raises -> Int:
        """Close stdin, wait `grace_ms`, SIGTERM the group, wait `grace_ms`, SIGKILL the
        group, reap. Returns the exit code (>= 0) or -signal."""
        self.close_stdin()
        if self._wait_for(grace_ms):
            return self._code
        self.signal_group(SIGTERM)
        if self._wait_for(grace_ms):
            return self._code
        self.signal_group(SIGKILL)
        return self.wait()

    def __deinit__(deinit self):
        """Closes the descriptors; kills (SIGKILL to the group) and reaps a child that
        is still running."""
        _close_fd(self.stdin_fd)
        _close_fd(self.stdout_fd)
        _close_fd(self.stderr_fd)
        if not self._reaped and self.pid > 0:
            if self._group:
                _ = c_kill(-self.pid, SIGKILL)
            else:
                _ = c_kill(self.pid, SIGKILL)
            var st = List[UInt8](length=8, fill=0)
            while True:
                var r = c_waitpid(self.pid, buf_addr(st), 0)
                if r >= 0 or errno() != EINTR:
                    break
            _ = len(st)
            unregister_child(self.pid, self._group)


# ---------------------------------------------------------------------------
# spawn
# ---------------------------------------------------------------------------
def _fd_valid(fd: Int) -> Bool:
    return c_fcntl(fd, F_GETFD, 0) >= 0


def spawn(spec: SpawnSpec) raises -> Child:
    """Starts `spec` with posix_spawn (see the module docstring)."""
    var path = resolve_program(spec.path, spec.env, spec.cwd)
    var argv = spec.argv.copy()
    if len(argv) == 0:
        argv.append(spec.path)
    var use_chdir_action = False
    if spec.cwd.byte_length() > 0:
        if has_symbol("posix_spawn_file_actions_addchdir_np"):
            use_chdir_action = True
        else:
            var tramp = getenv(_TRAMPOLINE_ENV, "")
            if tramp.byte_length() == 0:
                raise Error("QA-E162: this libc has no posix_spawn chdir action. Hint: use glibc 2.29 or newer")
            var targv = List[String]()
            targv.append(tramp)
            targv.append("__exec-in")
            targv.append(spec.cwd)
            targv.append(path)
            for i in range(1, len(argv)):
                targv.append(argv[i])
            path = tramp
            argv = targv^

    var in_r = -1
    var in_w = -1
    var out_r = -1
    var out_w = -1
    var err_r = -1
    var err_w = -1
    try:
        if spec.stdin_mode == STDIO_PIPE:
            var p = _make_pipe()
            in_r = p[0]
            in_w = p[1]
        if spec.stdout_mode == STDIO_PIPE:
            var p = _make_pipe()
            out_r = p[0]
            out_w = p[1]
        if spec.stderr_mode == STDIO_PIPE:
            var p = _make_pipe()
            err_r = p[0]
            err_w = p[1]
    except e:
        _close_fd(in_r)
        _close_fd(in_w)
        _close_fd(out_r)
        _close_fd(out_w)
        _close_fd(err_r)
        _close_fd(err_w)
        raise e^

    var fa = List[UInt64](length=64, fill=0)
    var attr = List[UInt64](length=64, fill=0)
    var fa_a = Int(fa.unsafe_ptr())
    var attr_a = Int(attr.unsafe_ptr())
    _ = c_fa_init(fa_a)
    _ = c_attr_init(attr_a)
    var devnull = CStr("/dev/null")
    var cwd_c = CStr(spec.cwd)
    var rc = 0

    # stdin
    if spec.stdin_mode == STDIO_PIPE:
        rc |= c_fa_adddup2(fa_a, in_r, 0)
    elif spec.stdin_mode == STDIO_NULL:
        rc |= c_fa_addopen(fa_a, 0, devnull.addr(), O_RDONLY, 0)
    elif is_macos() and _fd_valid(0):
        rc |= c_fa_adddup2(fa_a, 0, 0)    # keep it across POSIX_SPAWN_CLOEXEC_DEFAULT
    # stdout
    if spec.stdout_mode == STDIO_PIPE:
        rc |= c_fa_adddup2(fa_a, out_w, 1)
    elif spec.stdout_mode == STDIO_NULL:
        rc |= c_fa_addopen(fa_a, 1, devnull.addr(), O_WRONLY, 0)
    elif is_macos() and _fd_valid(1):
        rc |= c_fa_adddup2(fa_a, 1, 1)
    # stderr
    if spec.stderr_mode == STDIO_PIPE:
        rc |= c_fa_adddup2(fa_a, err_w, 2)
    elif spec.stderr_mode == STDIO_NULL:
        rc |= c_fa_addopen(fa_a, 2, devnull.addr(), O_WRONLY, 0)
    elif spec.stderr_mode == STDIO_MERGE:
        rc |= c_fa_adddup2(fa_a, 1, 2)
    elif is_macos() and _fd_valid(2):
        rc |= c_fa_adddup2(fa_a, 2, 2)
    # the parent's pipe ends
    var mine: List[Int] = [in_r, in_w, out_r, out_w, err_r, err_w]
    for fd in mine:
        if fd > 2:
            rc |= c_fa_addclose(fa_a, fd)
    if use_chdir_action:
        rc |= fa_addchdir(fa_a, cwd_c.addr())
    if not is_macos():
        _ = fa_addclosefrom(fa_a, 3)

    var flags = POSIX_SPAWN_SETSIGDEF | POSIX_SPAWN_SETSIGMASK
    if spec.new_process_group:
        flags |= POSIX_SPAWN_SETPGROUP
        rc |= c_attr_setpgroup(attr_a, 0)
    if is_macos():
        flags |= POSIX_SPAWN_CLOEXEC_DEFAULT
    var sigdef = List[UInt8](length=128, fill=0)
    sigdef[(SIGPIPE - 1) // 8] = UInt8(1 << ((SIGPIPE - 1) % 8))
    var sigmask = List[UInt8](length=128, fill=0)
    rc |= c_attr_setsigdefault(attr_a, buf_addr(sigdef))
    rc |= c_attr_setsigmask(attr_a, buf_addr(sigmask))
    rc |= c_attr_setflags(attr_a, flags)

    var path_c = CStr(path)
    var argv_c = CStrArray(argv)
    var envp_c = CStrArray(spec.env)
    var pid_buf = List[UInt8](length=8, fill=0)
    var err: Int
    if rc != 0:
        err = -1
    else:
        err = c_posix_spawn(buf_addr(pid_buf), path_c.addr(), fa_a, attr_a, argv_c.addr(), envp_c.addr())
    var pid = load_i32(buf_addr(pid_buf))
    _ = c_fa_destroy(fa_a)
    _ = c_attr_destroy(attr_a)
    _ = len(fa)
    _ = len(attr)
    _ = len(sigdef)
    _ = len(sigmask)
    _ = len(pid_buf)
    devnull.keep()
    cwd_c.keep()
    path_c.keep()
    argv_c.keep()
    envp_c.keep()

    # The child's ends belong to the child now.
    _close_fd(in_r)
    _close_fd(out_w)
    _close_fd(err_w)
    if err != 0:
        _close_fd(in_w)
        _close_fd(out_r)
        _close_fd(err_r)
        if err < 0:
            raise Error("QA-E160: posix_spawn file actions could not be set up for " + path)
        var shown = spec.path
        if use_chdir_action and err == ENOENT and not is_dir(spec.cwd):
            shown = spec.cwd
        raise Error("QA-E160: " + py_oserror(err, shown))
    register_child(pid, spec.new_process_group)
    return Child(pid, in_w, out_r, err_r, spec.new_process_group)


# ---------------------------------------------------------------------------
# run_capture
# ---------------------------------------------------------------------------
def _keep_tail(mut buf: List[UInt8], max_bytes: Int):
    if max_bytes <= 0:
        return
    if len(buf) > 2 * max_bytes + _READ_CHUNK:
        var keep = List[UInt8](capacity=max_bytes)
        for i in range(len(buf) - max_bytes, len(buf)):
            keep.append(buf[i])
        buf = keep^


def _tail_text(buf: List[UInt8], max_bytes: Int) -> String:
    var n = len(buf)
    var start = 0
    if max_bytes > 0 and n > max_bytes:
        start = n - max_bytes
    return String(from_utf8_lossy=Span(buf)[start:n])


def run_capture(spec: SpawnSpec, timeout_ms: Int, max_bytes: Int, stdin_data: String) raises -> CaptureResult:
    """Runs `spec` to completion. Piped stdout/stderr (per the spec's modes) are
    drained with poll (no deadlock) and only their last `max_bytes` bytes are kept;
    `stdin_data` is written (the stdin mode becomes a pipe when it is not empty) and
    stdin is closed. At the timeout the process group (or the child) gets SIGKILL
    and is reaped, and `timed_out` is True."""
    var s = spec.copy()
    if stdin_data.byte_length() > 0:
        s.stdin_mode = STDIO_PIPE
    var child = spawn(s)
    var deadline = monotonic_ms() + timeout_ms
    var in_data = stdin_data.as_bytes()
    var in_off = 0
    if child.stdin_fd >= 0:
        if len(in_data) == 0:
            child.close_stdin()
        else:
            var fl = c_fcntl(child.stdin_fd, F_GETFL, 0)
            _ = c_fcntl(child.stdin_fd, F_SETFL, fl | O_NONBLOCK)
    var out_buf = List[UInt8]()
    var err_buf = List[UInt8]()
    var out_total = 0
    var err_total = 0
    var out_open = child.stdout_fd >= 0
    var err_open = child.stderr_fd >= 0
    var timed_out = False
    var pfd = List[UInt8](length=24, fill=0)
    var chunk = List[UInt8](length=_READ_CHUNK, fill=0)
    while out_open or err_open or child.stdin_fd >= 0:
        var remaining = deadline - monotonic_ms()
        if remaining <= 0:
            timed_out = True
            break
        var a = buf_addr(pfd)
        var n = 0
        var slot_out = -1
        var slot_err = -1
        var slot_in = -1
        if out_open:
            store_u32(a + 8 * n, child.stdout_fd)
            store_u16(a + 8 * n + 4, POLLIN)
            store_u16(a + 8 * n + 6, 0)
            slot_out = n
            n += 1
        if err_open:
            store_u32(a + 8 * n, child.stderr_fd)
            store_u16(a + 8 * n + 4, POLLIN)
            store_u16(a + 8 * n + 6, 0)
            slot_err = n
            n += 1
        if child.stdin_fd >= 0:
            store_u32(a + 8 * n, child.stdin_fd)
            store_u16(a + 8 * n + 4, POLLOUT)
            store_u16(a + 8 * n + 6, 0)
            slot_in = n
            n += 1
        var r = c_poll(a, n, remaining)
        if r < 0:
            if errno() == EINTR:
                continue
            raise Error("QA-E160: poll failed: [Errno " + String(errno()) + "] " + strerror_text(errno()))
        if r == 0:
            continue
        if slot_out >= 0 and load_u16(a + 8 * slot_out + 6) != 0:
            var k = c_read(child.stdout_fd, buf_addr(chunk), _READ_CHUNK)
            if k > 0:
                for i in range(k):
                    out_buf.append(chunk[i])
                out_total += k
                _keep_tail(out_buf, max_bytes)
            elif k == 0 or (errno() != EINTR and errno() != EAGAIN):
                out_open = False
        if slot_err >= 0 and load_u16(a + 8 * slot_err + 6) != 0:
            var k = c_read(child.stderr_fd, buf_addr(chunk), _READ_CHUNK)
            if k > 0:
                for i in range(k):
                    err_buf.append(chunk[i])
                err_total += k
                _keep_tail(err_buf, max_bytes)
            elif k == 0 or (errno() != EINTR and errno() != EAGAIN):
                err_open = False
        if slot_in >= 0:
            var rev = load_u16(a + 8 * slot_in + 6)
            if (rev & (POLLERR | POLLHUP | POLLNVAL)) != 0:
                child.close_stdin()   # the child closed its stdin: drop the rest (as communicate())
            elif (rev & POLLOUT) != 0:
                var w = c_write(child.stdin_fd, Int(in_data.unsafe_ptr()) + in_off, len(in_data) - in_off)
                if w > 0:
                    in_off += w
                    if in_off >= len(in_data):
                        child.close_stdin()
                elif w < 0 and errno() != EINTR and errno() != EAGAIN:
                    child.close_stdin()
    _ = len(pfd)
    _ = len(chunk)
    _ = stdin_data.byte_length()
    if not timed_out:
        while not child.poll_exit():
            if monotonic_ms() >= deadline:
                timed_out = True
                break
            sleep_ms(5)
    if timed_out:
        child.signal_group(SIGKILL)
    var code = child.wait()
    return CaptureResult(
        code,
        timed_out,
        _tail_text(out_buf, max_bytes),
        _tail_text(err_buf, max_bytes),
        max_bytes > 0 and out_total > max_bytes,
        max_bytes > 0 and err_total > max_bytes,
    )


# ---------------------------------------------------------------------------
# Environments
# ---------------------------------------------------------------------------
def environ_list() -> List[String]:
    """This process's full environment ("NAME=value"): macOS `_NSGetEnviron()`,
    Linux `/proc/self/environ` (the environment the process started with)."""
    var out = List[String]()
    if is_macos():
        var pp = c_ns_get_environ()
        if pp == 0:
            return out^
        var envp = load_i64(pp)
        if envp == 0:
            return out^
        var i = 0
        while True:
            var s = load_i64(envp + 8 * i)
            if s == 0:
                break
            out.append(c_string_at(s))
            i += 1
        return out^
    try:
        with open("/proc/self/environ", "r") as f:
            var data = f.read_bytes()
            var start = 0
            for i in range(len(data)):
                if data[i] == 0:
                    if i > start:
                        out.append(String(from_utf8_lossy=Span(data)[start:i]))
                    start = i + 1
            if start < len(data):
                out.append(String(from_utf8_lossy=Span(data)[start : len(data)]))
    except:
        pass
    return out^


comptime _UNSET = "\x00unset"


def env_subset(names: List[String]) -> List[String]:
    """"NAME=value" for each name in `names` that is set, in `names` order
    (duplicates skipped)."""
    var out = List[String]()
    var seen = List[String]()
    for n in names:
        var dup = False
        for s in seen:
            if s == n:
                dup = True
                break
        if dup or n.byte_length() == 0:
            continue
        seen.append(n)
        var v = getenv(n, _UNSET)
        if v == _UNSET:
            continue
        out.append(n + "=" + v)
    return out^


def is_secret_env_name(name: String) -> Bool:
    """Upper-cased name ends with _API_KEY, _TOKEN, _SECRET or _PASSWORD, or contains
    SECRET or PASSWORD."""
    var u = name.upper()
    return (
        u.endswith("_API_KEY") or u.endswith("_TOKEN") or u.endswith("_SECRET") or u.endswith("_PASSWORD")
        or u.find("SECRET") >= 0 or u.find("PASSWORD") >= 0
    )


def minimal_child_env() -> List[String]:
    var names: List[String] = ["HOME", "LOGNAME", "PATH", "SHELL", "TERM", "USER", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR"]
    var out = List[String]()
    for e in env_subset(names):
        var eq = e.find("=")
        if not is_secret_env_name(String(e[byte=0:eq])):
            out.append(e)
    return out^


def mcp_server_env() -> List[String]:
    """The names Python's MCP stdio client passes on POSIX (mcp/client/stdio)."""
    var names: List[String] = ["HOME", "LOGNAME", "PATH", "SHELL", "TERM", "USER"]
    return env_subset(names)


def _valid_env_name(s: String) -> Bool:
    var bs = s.as_bytes()
    if len(bs) == 0:
        return False
    for i in range(len(bs)):
        var c = Int(bs[i])
        var ok = (c >= 0x41 and c <= 0x5A) or (c >= 0x61 and c <= 0x7A) or c == 0x5F or (i > 0 and c >= 0x30 and c <= 0x39)
        if not ok:
            return False
    return True


def harness_child_env(api_key_env: String) -> List[String]:
    """K4: `minimal_child_env()` plus the endpoint's key variable and the listed QUMO,
    alphaXiv, GitHub and TLS variables, for the children that run this binary
    (parallel subagents, the serve child of `call`). Nothing else is passed."""
    var out = minimal_child_env()
    var names = List[String]()
    if _valid_env_name(api_key_env):
        names.append(api_key_env)
    var fixed: List[String] = [
        "ALPHAXIV_API_KEY", "ALPHAXIV_MCP_URL", "GITHUB_TOKEN", "QUMO_ROOT", "QUMO_BIN",
        "QUMO_AGENT_BIN", "QUMO_AGENTS_MODEL", "QUMO_AGENTS_CONFIG", "QUMO_AGENT_TOOLS_DIR",
        "QUMO_AGENT_REPLAY_LOG", "QUMO_AGENT_DETERMINISTIC", "QUMO_LIBSSL", "QUMO_LIBCRYPTO",
        "SSL_CERT_FILE", "SSL_CERT_DIR",
    ]
    for n in fixed:
        names.append(n)
    for e in env_subset(names):
        var eq = e.find("=")
        var name = String(e[byte=0:eq])
        var dup = False
        for o in out:
            if o.startswith(name + "="):
                dup = True
                break
        if not dup:
            out.append(e)
    return out^


def ignore_sigpipe():
    """Writes to a closed pipe or socket return EPIPE instead of raising SIGPIPE."""
    ignore_sigpipe_signal()


# ---------------------------------------------------------------------------
# __exec-in trampoline
# ---------------------------------------------------------------------------
def exec_in_main(args: List[String]) -> Int:
    """Body of `qumo-agent __exec-in DIR PATH ARG...` (the chdir trampoline for libcs
    without posix_spawn's chdir action): chdir(DIR), then execve(PATH, [PATH, ARG...],
    this process's environment). `args` is the full argv (`args[1] == "__exec-in"`)
    or starts at "__exec-in". Returns only on failure (127, message on stderr)."""
    var k = -1
    if len(args) > 1 and args[1] == "__exec-in":
        k = 1
    elif len(args) > 0 and args[0] == "__exec-in":
        k = 0
    if k < 0 or len(args) < k + 3:
        _write_fd_all(2, "QA-E160: usage: __exec-in DIR PATH [ARG...]\n")
        return 127
    var dir = args[k + 1]
    var path = args[k + 2]
    var d = CStr(dir)
    var rc = c_chdir(d.addr())
    var e = errno()
    d.keep()
    if rc != 0:
        _write_fd_all(2, "QA-E160: " + py_oserror(e, dir) + "\n")
        return 127
    var argv = List[String]()
    argv.append(path)
    for i in range(k + 3, len(args)):
        argv.append(args[i])
    var p = CStr(path)
    var av = CStrArray(argv)
    var ev = CStrArray(environ_list())
    _ = c_execve(p.addr(), av.addr(), ev.addr())
    var e2 = errno()
    p.keep()
    av.keep()
    ev.keep()
    _write_fd_all(2, "QA-E160: " + py_oserror(e2, path) + "\n")
    return 127
