"""libc bindings: the only module of qumo_agent that contains `external_call`.

One wrapper per C symbol and one Mojo signature per symbol for the whole binary
(Mojo 1.2 rejects two different signatures for one symbol). Pointers cross the
boundary as `Int` addresses (NULL is `Int(0)`); strings go in as NUL-terminated
`CStr` buffers that the caller keeps alive until the call returns; C-returned
`char *` are copied with `c_string_at`.

Every wrapper returns `Int`. C `int` results are read back as `c_int` and widened,
so a `-1` stays `-1`; C `int` arguments are narrowed to `c_int` before the call.

Weak or optional symbols (`posix_spawn_file_actions_addchdir_np`,
`posix_spawn_file_actions_addclosefrom_np`) are resolved at run time through
`dlopen(NULL)` (`OwnedDLHandle(RTLD.NOW)`), never with `external_call`, so a libc
without them is not a link error. Linux-only calls (`unshare`, `prctl`, `syscall`)
and the macOS-only `_NSGetEnviron` are compiled for their platform only and return
-1 (or 0 for `_NSGetEnviron`) elsewhere.

Constants that differ between macOS and Linux (glibc) are `comptime` values chosen
with `CompilationTarget.is_macos()` (DESIGN.md section 4.2). A Linux value marked
"n/a" in the design is `-1` here.

Lifetimes: Mojo destroys a value right after its last use, so a `CStr`, a
`CStrArray` or a `List` whose address is passed to C must be used again after the
call (`s.keep()`, `_ = len(buf)`); otherwise it can be freed before C reads it.

Rule for buffers that C fills: read them back by the length the C function
returns (`String(from_utf8_lossy=Span(buf)[0:n])`) and keep the buffer alive past
the call (`_ = len(buf)` after the last use of an address taken from it).
"""

from std.ffi import external_call, c_int, get_errno, OwnedDLHandle, RTLD
from std.sys import CompilationTarget
from std.memory import OpaquePointer


# ---------------------------------------------------------------------------
# Platform constants (DESIGN.md 4.2)
# ---------------------------------------------------------------------------
comptime _MAC = CompilationTarget.is_macos()

comptime POSIX_SPAWN_SETPGROUP = 0x02
comptime POSIX_SPAWN_SETSIGDEF = 0x04
comptime POSIX_SPAWN_SETSIGMASK = 0x08
comptime POSIX_SPAWN_CLOEXEC_DEFAULT = 0x4000 if _MAC else -1   # macOS only

comptime SIZEOF_POSIX_SPAWN_FILE_ACTIONS = 8 if _MAC else 80
comptime SIZEOF_POSIX_SPAWNATTR = 8 if _MAC else 336
comptime SIZEOF_SIGSET = 4 if _MAC else 128

comptime SIGPIPE = 13
comptime SIGTERM = 15
comptime SIGKILL = 9
comptime SIG_IGN = 1
comptime SIG_DFL = 0
comptime WNOHANG = 1

comptime O_RDONLY = 0
comptime O_WRONLY = 1
comptime O_RDWR = 2
comptime O_CREAT = 0x200 if _MAC else 0x40
comptime O_TRUNC = 0x400 if _MAC else 0x200
comptime O_NONBLOCK = 0x4 if _MAC else 0x800
comptime O_CLOEXEC = 0x1000000 if _MAC else 0x80000
comptime O_PATH = -1 if _MAC else 0x200000        # Linux only (Landlock rules)

comptime F_GETFD = 1
comptime F_SETFD = 2
comptime F_GETFL = 3
comptime F_SETFL = 4
comptime FD_CLOEXEC = 1

comptime POLLIN = 0x1
comptime POLLOUT = 0x4
comptime POLLERR = 0x8
comptime POLLHUP = 0x10
comptime POLLNVAL = 0x20

comptime F_OK = 0
comptime X_OK = 1
comptime W_OK = 2
comptime R_OK = 4

comptime CLOCK_REALTIME = 0

comptime REG_EXTENDED = 1
comptime REG_ICASE = 2
comptime REG_NOSUB = 4 if _MAC else 8
comptime REG_NEWLINE = 8 if _MAC else 4
comptime REG_NOMATCH = 1

comptime FNM_NOMATCH = 1
comptime FNM_NOESCAPE = 0x1 if _MAC else 0x2
comptime FNM_PATHNAME = 0x2 if _MAC else 0x1
comptime FNM_PERIOD = 0x4

comptime AF_INET = 2
comptime AF_INET6 = 30 if _MAC else 10
comptime AF_UNSPEC = 0
comptime SOCK_STREAM = 1
comptime IPPROTO_TCP = 6
comptime SOL_SOCKET = 0xFFFF if _MAC else 1
comptime SO_RCVTIMEO = 0x1006 if _MAC else 20
comptime SO_SNDTIMEO = 0x1005 if _MAC else 21
comptime SO_ERROR = 0x1007 if _MAC else 4
comptime SO_REUSEADDR = 0x4 if _MAC else 2
comptime SO_NOSIGPIPE = 0x1022 if _MAC else -1   # macOS only

comptime EPERM = 1
comptime ENOENT = 2
comptime ESRCH = 3
comptime EINTR = 4
comptime EBADF = 9
comptime ECHILD = 10
comptime EACCES = 13
comptime EEXIST = 17
comptime ENOTDIR = 20
comptime EISDIR = 21
comptime EINVAL = 22
comptime EPIPE = 32
comptime EAGAIN = 35 if _MAC else 11
comptime EINPROGRESS = 36 if _MAC else 115
comptime ECONNRESET = 54 if _MAC else 104
comptime ETIMEDOUT = 60 if _MAC else 110
comptime ECONNREFUSED = 61 if _MAC else 111

# struct timeval: i64 tv_sec, then tv_usec (i32 + 4 bytes padding on macOS, i64 on Linux).
comptime SIZEOF_TIMEVAL = 16
comptime TIMEVAL_USEC_OFFSET = 8
comptime TIMEVAL_USEC_BYTES = 4 if _MAC else 8
comptime SIZEOF_TIMESPEC = 16

# struct addrinfo (48 bytes on both).
comptime SIZEOF_ADDRINFO = 48
comptime AI_FLAGS_OFFSET = 0
comptime AI_FAMILY_OFFSET = 4
comptime AI_SOCKTYPE_OFFSET = 8
comptime AI_PROTOCOL_OFFSET = 12
comptime AI_ADDRLEN_OFFSET = 16
comptime AI_CANONNAME_OFFSET = 24 if _MAC else 32
comptime AI_ADDR_OFFSET = 32 if _MAC else 24
comptime AI_NEXT_OFFSET = 40

# struct tm: 9 ints from offset 0 (sec, min, hour, mday, mon, year, wday, yday, isdst).
comptime SIZEOF_TM = 64
comptime TM_SEC_OFFSET = 0
comptime TM_MIN_OFFSET = 4
comptime TM_HOUR_OFFSET = 8
comptime TM_MDAY_OFFSET = 12
comptime TM_MON_OFFSET = 16
comptime TM_YEAR_OFFSET = 20
comptime TM_WDAY_OFFSET = 24
comptime TM_YDAY_OFFSET = 28
comptime TM_ISDST_OFFSET = 32

# setrlimit (struct rlimit = two 64-bit values).
comptime RLIMIT_CPU = 0
comptime RLIMIT_FSIZE = 1

# Linux confinement (unshare, prctl, Landlock syscalls; same numbers on x86_64 and aarch64).
comptime CLONE_NEWUSER = 0x10000000
comptime CLONE_NEWNET = 0x40000000
comptime PR_SET_NO_NEW_PRIVS = 38
comptime SYS_LANDLOCK_CREATE_RULESET = 444
comptime SYS_LANDLOCK_ADD_RULE = 445
comptime SYS_LANDLOCK_RESTRICT_SELF = 446
comptime SIGXCPU = 24   # RLIMIT_CPU exceeded (macOS, Linux x86_64 and aarch64)

comptime PATH_BUF_SIZE = 4096   # realpath output buffer (PATH_MAX 1024 macOS, 4096 Linux)


# ---------------------------------------------------------------------------
# Strings and raw memory
# ---------------------------------------------------------------------------
# Pointer arguments and pointer results cross `external_call` as untyped
# pointers (`OpaquePointer`), the form the standard library uses for the libc
# symbols it calls itself (write, strerror, ...): Mojo refuses two different
# signatures for one symbol in one binary.
comptime _VP = OpaquePointer[MutAnyOrigin]


def _p(addr: Int) -> _VP:
    return _VP(unsafe_from_address=addr)


struct CStr(Movable):
    """A NUL-terminated copy of a String; keep it alive until the C call returns."""

    var buf: List[UInt8]

    def __init__(out self, s: String):
        var bs = s.as_bytes()
        self.buf = List[UInt8](capacity=len(bs) + 1)
        for i in range(len(bs)):
            self.buf.append(bs[i])
        self.buf.append(0)

    def addr(self) -> Int:
        return Int(self.buf.unsafe_ptr())

    def keep(self):
        """No-op use that keeps this buffer alive up to this point (ASAP destruction
        frees a value right after its last use, even inside the expression that
        passes `addr()` to C: call `keep()` after the C call)."""
        pass


struct CStrArray(Movable):
    """argv/envp: NUL-terminated copies plus a NULL-terminated pointer array."""

    var bufs: List[List[UInt8]]
    var ptrs: List[Int]

    def __init__(out self, items: List[String]):
        self.bufs = List[List[UInt8]](capacity=len(items))
        for item in items:
            var bs = item.as_bytes()
            var b = List[UInt8](capacity=len(bs) + 1)
            for i in range(len(bs)):
                b.append(bs[i])
            b.append(0)
            self.bufs.append(b^)
        # Heap buffers do not move when the outer list grows, so take the
        # addresses only once every buffer is in place.
        self.ptrs = List[Int](capacity=len(items) + 1)
        for i in range(len(self.bufs)):
            self.ptrs.append(Int(self.bufs[i].unsafe_ptr()))
        self.ptrs.append(0)

    def addr(self) -> Int:
        return Int(self.ptrs.unsafe_ptr())

    def keep(self):
        """No-op use that keeps the buffers alive up to this point (see `CStr.keep`)."""
        pass


def buf_addr(mut buf: List[UInt8]) -> Int:
    """Address of the first byte of `buf` (keep `buf` alive past the C call)."""
    return Int(buf.unsafe_ptr())


def _bytes_at(addr: Int) -> Pointer[UInt8, MutUntrackedOrigin]:
    return Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=addr)


def load_u8(addr: Int) -> Int:
    return Int(_bytes_at(addr).unsafe_load[alignment=1]())


def store_u8(addr: Int, v: Int):
    _bytes_at(addr).unsafe_store[alignment=1](UInt8(v & 0xFF))


def load_u16(addr: Int) -> Int:
    return Int(_bytes_at(addr).unsafe_bitcast[UInt16]().unsafe_load[alignment=1]())


def store_u16(addr: Int, v: Int):
    _bytes_at(addr).unsafe_bitcast[UInt16]().unsafe_store[alignment=1](UInt16(v & 0xFFFF))


def load_i32(addr: Int) -> Int:
    return Int(_bytes_at(addr).unsafe_bitcast[Int32]().unsafe_load[alignment=1]())


def load_u32(addr: Int) -> Int:
    return Int(_bytes_at(addr).unsafe_bitcast[UInt32]().unsafe_load[alignment=1]())


def store_u32(addr: Int, v: Int):
    _bytes_at(addr).unsafe_bitcast[UInt32]().unsafe_store[alignment=1](UInt32(v & 0xFFFFFFFF))


def load_i64(addr: Int) -> Int:
    return Int(_bytes_at(addr).unsafe_bitcast[Int64]().unsafe_load[alignment=1]())


def store_i64(addr: Int, v: Int):
    _bytes_at(addr).unsafe_bitcast[Int64]().unsafe_store[alignment=1](Int64(v))


def c_string_at(addr: Int) -> String:
    """Copy of the C string at `addr` (lossy UTF-8); "" for NULL."""
    if addr == 0:
        return String()
    var n = 0
    while load_u8(addr + n) != 0:
        n += 1
    var buf = List[UInt8](capacity=n)
    for i in range(n):
        buf.append(UInt8(load_u8(addr + i)))
    return String(from_utf8_lossy=Span(buf))


def errno() -> Int:
    """The calling thread's errno (`std.ffi.get_errno().value`)."""
    return Int(get_errno().value)


def strerror_text(code: Int) -> String:
    """strerror(), e.g. "No such file or directory"."""
    return c_string_at(c_strerror(code))


def is_macos() -> Bool:
    comptime if _MAC:
        return True
    else:
        return False


def has_symbol(name: String) -> Bool:
    """True when the running process can resolve `name` (dlopen(NULL) + dlsym)."""
    try:
        var h = OwnedDLHandle(RTLD.NOW)
        return h.check_symbol(name)
    except:
        return False


# ---------------------------------------------------------------------------
# Processes: posix_spawn and friends
# ---------------------------------------------------------------------------
def c_posix_spawn(pid_addr: Int, path: Int, fa: Int, attr: Int, argv: Int, envp: Int) -> Int:
    """Returns an errno value (0 = ok); the pid is stored at `pid_addr` (a 4-byte pid_t)."""
    return Int(external_call["posix_spawn", c_int](_p(pid_addr), _p(path), _p(fa), _p(attr), _p(argv), _p(envp)))


def c_fa_init(fa: Int) -> Int:
    return Int(external_call["posix_spawn_file_actions_init", c_int](_p(fa)))


def c_fa_destroy(fa: Int) -> Int:
    return Int(external_call["posix_spawn_file_actions_destroy", c_int](_p(fa)))


def c_fa_adddup2(fa: Int, fd: Int, newfd: Int) -> Int:
    return Int(external_call["posix_spawn_file_actions_adddup2", c_int](_p(fa), c_int(fd), c_int(newfd)))


def c_fa_addclose(fa: Int, fd: Int) -> Int:
    return Int(external_call["posix_spawn_file_actions_addclose", c_int](_p(fa), c_int(fd)))


def c_fa_addopen(fa: Int, fd: Int, path: Int, oflag: Int, mode: Int) -> Int:
    comptime if _MAC:
        return Int(external_call["posix_spawn_file_actions_addopen", c_int](
            _p(fa), c_int(fd), _p(path), c_int(oflag), UInt16(mode)))
    else:
        return Int(external_call["posix_spawn_file_actions_addopen", c_int](
            _p(fa), c_int(fd), _p(path), c_int(oflag), UInt32(mode)))


def fa_addchdir(fa: Int, path: Int) raises -> Int:
    """posix_spawn_file_actions_addchdir_np through dlsym (macOS 10.15+, glibc 2.29+).

    Raises QA-E162 when this libc does not have it."""
    var h = OwnedDLHandle(RTLD.NOW)
    if not h.check_symbol("posix_spawn_file_actions_addchdir_np"):
        raise Error("QA-E162: this libc has no posix_spawn chdir action. Hint: use glibc 2.29 or newer")
    return Int(h.get_function[c_int]("posix_spawn_file_actions_addchdir_np")(_p(fa), _p(path)))


def fa_addclosefrom(fa: Int, lowfd: Int) -> Int:
    """posix_spawn_file_actions_addclosefrom_np through dlsym (glibc 2.34+); -1 when absent."""
    try:
        var h = OwnedDLHandle(RTLD.NOW)
        if not h.check_symbol("posix_spawn_file_actions_addclosefrom_np"):
            return -1
        return Int(h.get_function[c_int]("posix_spawn_file_actions_addclosefrom_np")(_p(fa), c_int(lowfd)))
    except:
        return -1


def c_attr_init(a: Int) -> Int:
    return Int(external_call["posix_spawnattr_init", c_int](_p(a)))


def c_attr_destroy(a: Int) -> Int:
    return Int(external_call["posix_spawnattr_destroy", c_int](_p(a)))


def c_attr_setflags(a: Int, flags: Int) -> Int:
    return Int(external_call["posix_spawnattr_setflags", c_int](_p(a), Int16(flags)))


def c_attr_setpgroup(a: Int, pgid: Int) -> Int:
    return Int(external_call["posix_spawnattr_setpgroup", c_int](_p(a), c_int(pgid)))


def c_attr_setsigdefault(a: Int, set: Int) -> Int:
    return Int(external_call["posix_spawnattr_setsigdefault", c_int](_p(a), _p(set)))


def c_attr_setsigmask(a: Int, set: Int) -> Int:
    return Int(external_call["posix_spawnattr_setsigmask", c_int](_p(a), _p(set)))


def c_waitpid(pid: Int, status_addr: Int, options: Int) -> Int:
    """`WIFEXITED`: `(s & 0x7f) == 0`, code `(s >> 8) & 0xff`; signalled: `s & 0x7f` not 0 or 0x7f."""
    return Int(external_call["waitpid", c_int](c_int(pid), _p(status_addr), c_int(options)))


def c_kill(pid: Int, sig: Int) -> Int:
    """`pid < 0` signals the process group `-pid`."""
    return Int(external_call["kill", c_int](c_int(pid), c_int(sig)))


def c_execve(path: Int, argv: Int, envp: Int) -> Int:
    return Int(external_call["execve", c_int](_p(path), _p(argv), _p(envp)))


def c_chdir(path: Int) -> Int:
    return Int(external_call["chdir", c_int](_p(path)))


def c_setrlimit(resource: Int, rl: Int) -> Int:
    """`rl` points to two 64-bit values (soft, hard)."""
    return Int(external_call["setrlimit", c_int](c_int(resource), _p(rl)))


def c_isatty(fd: Int) -> Int:
    return Int(external_call["isatty", c_int](c_int(fd)))


def c_unshare(flags: Int) -> Int:
    """Linux only; -1 elsewhere."""
    comptime if _MAC:
        return -1
    else:
        return Int(external_call["unshare", c_int](c_int(flags)))


def c_prctl(option: Int, arg2: Int, arg3: Int, arg4: Int, arg5: Int) -> Int:
    """Linux only (variadic, one fixed argument); -1 elsewhere."""
    comptime if _MAC:
        return -1
    else:
        return Int(external_call["prctl", c_int, num_fixed_args=1](c_int(option), arg2, arg3, arg4, arg5))


def c_syscall4(nr: Int, a: Int, b: Int, c: Int) -> Int:
    """Linux only: `syscall(nr, a, b, c, 0)` (Landlock); -1 elsewhere.

    A fourth system-call argument, always 0, is passed so that
    `landlock_add_rule(fd, type, attr, flags)` gets `flags == 0`; the 2- and
    3-argument calls ignore it. One call shape keeps one signature per symbol."""
    comptime if _MAC:
        return -1
    else:
        return external_call["syscall", Int, num_fixed_args=1](nr, a, b, c, 0)


# ---------------------------------------------------------------------------
# File descriptors, pipes, poll, signals, environment
# ---------------------------------------------------------------------------
def c_pipe(fds_addr: Int) -> Int:
    """`fds_addr` points to two 4-byte ints."""
    return Int(external_call["pipe", c_int](_p(fds_addr)))


def c_fcntl(fd: Int, cmd: Int, arg: Int) -> Int:
    return Int(external_call["fcntl", c_int, num_fixed_args=2](c_int(fd), c_int(cmd), c_int(arg)))


def c_close(fd: Int) -> Int:
    return Int(external_call["close", c_int](c_int(fd)))


def c_read(fd: Int, buf: Int, n: Int) -> Int:
    return external_call["read", Int](c_int(fd), _p(buf), UInt(n))


def c_write(fd: Int, buf: Int, n: Int) -> Int:
    return external_call["write", Int](c_int(fd), _p(buf), UInt(n))


def c_poll(fds: Int, n: Int, timeout_ms: Int) -> Int:
    """`pollfd` is 8 bytes: {int fd; short events; short revents}."""
    return Int(external_call["poll", c_int](_p(fds), UInt(n), c_int(timeout_ms)))


def c_signal(sig: Int, handler: Int) -> Int:
    """Only used with SIG_IGN (1) for SIGPIPE; returns the previous handler."""
    return Int(external_call["signal", _VP](c_int(sig), _p(handler)))


def c_ns_get_environ() -> Int:
    """macOS `_NSGetEnviron()` (a `char ***`); 0 elsewhere."""
    comptime if _MAC:
        return Int(external_call["_NSGetEnviron", _VP]())
    else:
        return 0


def c_open(path: Int, flags: Int, mode: Int) -> Int:
    """`open(path, flags, mode)`; the same call shape as the standard library's own
    `open` (variadic, two fixed arguments), so both can live in one binary. Used for
    Landlock's `O_PATH` descriptors only; regular files go through std `open()`."""
    return Int(external_call["open", c_int, num_fixed_args=2](_p(path), c_int(flags), c_int(mode)))


def c_readlink(path: Int, buf: Int, n: Int) -> Int:
    """`readlink`: bytes written (no NUL terminator), or -1."""
    return external_call["readlink", Int](_p(path), _p(buf), UInt(n))


def c_access(path: Int, mode: Int) -> Int:
    return Int(external_call["access", c_int](_p(path), c_int(mode)))


def c_realpath(path: Int, out_buf: Int) -> Int:
    """`out_buf` must hold PATH_BUF_SIZE bytes; returns it, or 0 on error."""
    # Same signature as the standard library's realpath call (std.sys._libc).
    var r = external_call["realpath", Optional[_VP]](_p(path), _p(out_buf))
    if r:
        return Int(r.value())
    return 0


def c_mkdtemp(template: Int) -> Int:
    return Int(external_call["mkdtemp", _VP](_p(template)))


def c_chmod(path: Int, mode: Int) -> Int:
    comptime if _MAC:
        return Int(external_call["chmod", c_int](_p(path), UInt16(mode)))
    else:
        return Int(external_call["chmod", c_int](_p(path), UInt32(mode)))


def c_rename(a: Int, b: Int) -> Int:
    return Int(external_call["rename", c_int](_p(a), _p(b)))


def c_strerror(code: Int) -> Int:
    # Same signature as the standard library's own strerror call (std.sys._libc_errno).
    return Int(external_call["strerror", _VP](c_int(code)))


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------
def c_time() -> Int:
    """`time(NULL)`."""
    return external_call["time", Int](_p(0))


def c_clock_gettime(clock: Int, ts: Int) -> Int:
    """`ts` points to a 16-byte timespec {i64 sec, i64 nsec}."""
    return Int(external_call["clock_gettime", c_int](c_int(clock), _p(ts)))


def c_gmtime_r(t_addr: Int, tm: Int) -> Int:
    return Int(external_call["gmtime_r", _VP](_p(t_addr), _p(tm)))


def c_localtime_r(t_addr: Int, tm: Int) -> Int:
    return Int(external_call["localtime_r", _VP](_p(t_addr), _p(tm)))


def c_strftime(buf: Int, n: Int, fmt: Int, tm: Int) -> Int:
    return Int(external_call["strftime", UInt](_p(buf), UInt(n), _p(fmt), _p(tm)))


def c_timegm(tm: Int) -> Int:
    return external_call["timegm", Int](_p(tm))


# ---------------------------------------------------------------------------
# Sockets and name resolution
# ---------------------------------------------------------------------------
def c_getaddrinfo(host: Int, port: Int, hints: Int, res_addr: Int) -> Int:
    return Int(external_call["getaddrinfo", c_int](_p(host), _p(port), _p(hints), _p(res_addr)))


def c_freeaddrinfo(ai: Int):
    external_call["freeaddrinfo", NoneType](_p(ai))


def c_gai_strerror(code: Int) -> Int:
    return Int(external_call["gai_strerror", _VP](c_int(code)))


def c_socket(family: Int, type: Int, proto: Int) -> Int:
    return Int(external_call["socket", c_int](c_int(family), c_int(type), c_int(proto)))


def c_connect(fd: Int, addr: Int, len: Int) -> Int:
    return Int(external_call["connect", c_int](c_int(fd), _p(addr), UInt32(len)))


def c_setsockopt(fd: Int, level: Int, name: Int, val: Int, len: Int) -> Int:
    return Int(external_call["setsockopt", c_int](c_int(fd), c_int(level), c_int(name), _p(val), UInt32(len)))


def c_getsockopt(fd: Int, level: Int, name: Int, val: Int, len_addr: Int) -> Int:
    return Int(external_call["getsockopt", c_int](c_int(fd), c_int(level), c_int(name), _p(val), _p(len_addr)))


def c_inet_pton(family: Int, src: Int, dst: Int) -> Int:
    return Int(external_call["inet_pton", c_int](c_int(family), _p(src), _p(dst)))


def c_bind(fd: Int, addr: Int, len: Int) -> Int:
    return Int(external_call["bind", c_int](c_int(fd), _p(addr), UInt32(len)))


def c_listen(fd: Int, backlog: Int) -> Int:
    return Int(external_call["listen", c_int](c_int(fd), c_int(backlog)))


def c_accept(fd: Int, addr: Int, len_addr: Int) -> Int:
    return Int(external_call["accept", c_int](c_int(fd), _p(addr), _p(len_addr)))


def c_getsockname(fd: Int, addr: Int, len_addr: Int) -> Int:
    return Int(external_call["getsockname", c_int](c_int(fd), _p(addr), _p(len_addr)))


# ---------------------------------------------------------------------------
# POSIX regex and fnmatch
# ---------------------------------------------------------------------------
def c_regcomp(re: Int, pattern: Int, flags: Int) -> Int:
    return Int(external_call["regcomp", c_int](_p(re), _p(pattern), c_int(flags)))


def c_regexec(re: Int, s: Int, nmatch: Int, pmatch: Int, flags: Int) -> Int:
    return Int(external_call["regexec", c_int](_p(re), _p(s), UInt(nmatch), _p(pmatch), c_int(flags)))


def c_regerror(code: Int, re: Int, buf: Int, n: Int) -> Int:
    return Int(external_call["regerror", UInt](c_int(code), _p(re), _p(buf), UInt(n)))


def c_regfree(re: Int):
    external_call["regfree", NoneType](_p(re))


def c_fnmatch(pattern: Int, s: Int, flags: Int) -> Int:
    """0 = match, FNM_NOMATCH (1) = no match."""
    return Int(external_call["fnmatch", c_int](_p(pattern), _p(s), c_int(flags)))
