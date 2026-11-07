"""Tests for native/libc.mojo: every binding of DESIGN.md 4.2 called once on this platform."""

from std.testing import assert_equal, assert_true, assert_false
from std.os import rmdir, getenv

from qumo_agent.native.libc import *


def _wait_status(pid: Int) raises -> Int:
    var st = List[UInt8](length=8, fill=0)
    var r = c_waitpid(pid, buf_addr(st), 0)
    assert_equal(r, pid)
    var s = load_i32(buf_addr(st))
    _ = len(st)
    return s


def _exit_code(s: Int) -> Int:
    if (s & 0x7F) == 0:
        return (s >> 8) & 0xFF
    return -(s & 0x7F)


def _spawn(path: String, argv: List[String], env: List[String], cwd: String, out_fd: Int, pgroup: Bool) raises -> Int:
    """posix_spawn with stdout on `out_fd` (or inherited when -1)."""
    var fa = List[UInt64](length=64, fill=0)
    var attr = List[UInt64](length=64, fill=0)
    var fa_a = Int(fa.unsafe_ptr())
    var attr_a = Int(attr.unsafe_ptr())
    assert_equal(c_fa_init(fa_a), 0)
    assert_equal(c_attr_init(attr_a), 0)
    if out_fd >= 0:
        assert_equal(c_fa_adddup2(fa_a, out_fd, 1), 0)
        assert_equal(c_fa_addclose(fa_a, out_fd), 0)
    var devnull = CStr("/dev/null")
    assert_equal(c_fa_addopen(fa_a, 0, devnull.addr(), O_RDONLY, 0), 0)
    var cwd_c = CStr(cwd)
    if cwd.byte_length() > 0:
        assert_equal(fa_addchdir(fa_a, cwd_c.addr()), 0)
    # SIGPIPE back to default in the child, empty signal mask.
    var sigs = List[UInt8](length=128, fill=0)
    sigs[(SIGPIPE - 1) // 8] = UInt8(1 << ((SIGPIPE - 1) % 8))
    var empty = List[UInt8](length=128, fill=0)
    assert_equal(c_attr_setsigdefault(attr_a, buf_addr(sigs)), 0)
    assert_equal(c_attr_setsigmask(attr_a, buf_addr(empty)), 0)
    var flags = POSIX_SPAWN_SETSIGDEF | POSIX_SPAWN_SETSIGMASK
    if pgroup:
        flags |= POSIX_SPAWN_SETPGROUP
        assert_equal(c_attr_setpgroup(attr_a, 0), 0)
    if is_macos():
        flags |= POSIX_SPAWN_CLOEXEC_DEFAULT
    else:
        _ = fa_addclosefrom(fa_a, 3)
    assert_equal(c_attr_setflags(attr_a, flags), 0)
    var path_c = CStr(path)
    var argv_c = CStrArray(argv)
    var env_c = CStrArray(env)
    var pid_buf = List[UInt8](length=8, fill=0)
    var rc = c_posix_spawn(buf_addr(pid_buf), path_c.addr(), fa_a, attr_a, argv_c.addr(), env_c.addr())
    assert_equal(rc, 0)
    var pid = load_i32(buf_addr(pid_buf))
    _ = c_fa_destroy(fa_a)
    _ = c_attr_destroy(attr_a)
    _ = len(fa)
    _ = len(attr)
    _ = len(sigs)
    _ = len(empty)
    _ = len(pid_buf)
    devnull.keep()
    cwd_c.keep()
    path_c.keep()
    argv_c.keep()
    env_c.keep()
    return pid


def _read_all(fd: Int) -> String:
    var out = List[UInt8]()
    var buf = List[UInt8](length=4096, fill=0)
    while True:
        var n = c_read(fd, buf_addr(buf), 4096)
        if n <= 0:
            break
        for i in range(n):
            out.append(buf[i])
    _ = len(buf)
    return String(from_utf8_lossy=Span(out))


def _pipe() raises -> Tuple[Int, Int]:
    var fds = List[UInt8](length=8, fill=0)
    assert_equal(c_pipe(buf_addr(fds)), 0)
    var r = load_i32(buf_addr(fds))
    var w = load_i32(buf_addr(fds) + 4)
    assert_equal(c_fcntl(r, F_SETFD, FD_CLOEXEC), 0)
    assert_equal(c_fcntl(w, F_SETFD, FD_CLOEXEC), 0)
    assert_equal(c_fcntl(r, F_GETFD, 0) & FD_CLOEXEC, FD_CLOEXEC)
    _ = len(fds)
    return (r, w)


def _run(path: String, argv: List[String], env: List[String], cwd: String) raises -> Tuple[Int, String]:
    var p = _pipe()
    var pid = _spawn(path, argv, env, cwd, p[1], False)
    _ = c_close(p[1])
    var text = _read_all(p[0])
    _ = c_close(p[0])
    return (_exit_code(_wait_status(pid)), text)


def test_strings_and_memory() raises:
    var c = CStr("héllo")
    assert_equal(len(c.buf), 7)
    assert_equal(c_string_at(c.addr()), "héllo")
    assert_equal(c_string_at(0), "")
    var items: List[String] = ["a", "bc", ""]
    var arr = CStrArray(items)
    assert_equal(len(arr.ptrs), 4)
    assert_equal(load_i64(arr.addr() + 8 * 3), 0)
    assert_equal(c_string_at(load_i64(arr.addr() + 8)), "bc")
    assert_equal(c_string_at(load_i64(arr.addr() + 16)), "")
    var b = List[UInt8](length=16, fill=0)
    var a = buf_addr(b)
    store_u32(a, 0xDEADBEEF)
    assert_equal(load_u32(a), 0xDEADBEEF)
    store_i64(a + 8, -5)
    assert_equal(load_i64(a + 8), -5)
    store_u16(a + 4, 0xABCD)
    assert_equal(load_u16(a + 4), 0xABCD)
    store_u32(a, 0xFFFFFFFF)
    assert_equal(load_i32(a), -1)
    store_u8(a, 7)
    assert_equal(load_u8(a), 7)
    _ = len(b)
    c.keep()
    arr.keep()


def test_errno_and_symbols() raises:
    assert_equal(c_close(-1), -1)
    assert_equal(errno(), EBADF)
    assert_equal(strerror_text(ENOENT), "No such file or directory")
    assert_true(has_symbol("posix_spawn"))
    assert_false(has_symbol("qumo_no_such_symbol_xyz"))
    assert_equal(is_macos(), has_symbol("_NSGetEnviron"))
    assert_equal(c_isatty(-1), 0)
    var missing = CStr("/nonexistent/qumo-agent-x")
    assert_equal(c_access(missing.addr(), X_OK), -1)
    assert_equal(errno(), ENOENT)
    var sh = CStr("/bin/sh")
    assert_equal(c_access(sh.addr(), X_OK), 0)
    # execve of a missing program returns instead of replacing the process.
    var argv: List[String] = ["x"]
    var a = CStrArray(argv)
    var env = CStrArray(List[String]())
    assert_equal(c_execve(missing.addr(), a.addr(), env.addr()), -1)
    assert_equal(errno(), ENOENT)
    # Bad resource numbers are refused without touching the real limits.
    var rl = List[UInt64](length=2, fill=0)
    assert_equal(c_setrlimit(9999, Int(rl.unsafe_ptr())), -1)
    if is_macos():
        assert_true(c_ns_get_environ() != 0)
        assert_equal(c_unshare(CLONE_NEWNET), -1)
        assert_equal(c_prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0), -1)
        assert_equal(c_syscall4(SYS_LANDLOCK_CREATE_RULESET, 0, 0, 1), -1)
        var fa = List[UInt64](length=64, fill=0)
        assert_equal(c_fa_init(Int(fa.unsafe_ptr())), 0)
        assert_equal(fa_addclosefrom(Int(fa.unsafe_ptr()), 3), -1)
        _ = c_fa_destroy(Int(fa.unsafe_ptr()))
        _ = len(fa)
    _ = len(rl)
    missing.keep()
    sh.keep()
    a.keep()
    env.keep()


def test_pipes_poll_sigpipe() raises:
    var p = _pipe()
    var pfd = List[UInt8](length=8, fill=0)
    store_u32(buf_addr(pfd), p[0])
    store_u16(buf_addr(pfd) + 4, POLLIN)
    assert_equal(c_poll(buf_addr(pfd), 1, 10), 0)
    var msg = CStr("ping")
    assert_equal(c_write(p[1], msg.addr(), 4), 4)
    assert_equal(c_poll(buf_addr(pfd), 1, 1000), 1)
    assert_true((load_u16(buf_addr(pfd) + 6) & POLLIN) != 0)
    var buf = List[UInt8](length=16, fill=0)
    var n = c_read(p[0], buf_addr(buf), 16)
    assert_equal(n, 4)
    assert_equal(String(from_utf8_lossy=Span(buf)[0:n]), "ping")
    # Non-blocking read of an empty pipe gives EAGAIN.
    var fl = c_fcntl(p[0], F_GETFL, 0)
    assert_equal(c_fcntl(p[0], F_SETFL, fl | O_NONBLOCK), 0)
    assert_equal(c_read(p[0], buf_addr(buf), 16), -1)
    assert_equal(errno(), EAGAIN)
    # With SIGPIPE ignored, writing to a pipe without readers fails with EPIPE.
    _ = c_signal(SIGPIPE, SIG_IGN)
    _ = c_close(p[0])
    assert_equal(c_write(p[1], msg.addr(), 4), -1)
    assert_equal(errno(), EPIPE)
    _ = c_close(p[1])
    _ = len(pfd)
    _ = len(buf)
    msg.keep()


def test_spawn() raises:
    # argv is passed verbatim (the Mojo 1.2 Process.run argv bug cannot occur).
    var argv: List[String] = ["echo", "ab lit", "x"]
    var r = _run("/bin/echo", argv, List[String](), "")
    assert_equal(r[0], 0)
    assert_equal(r[1], "ab lit x\n")
    # The environment is exactly envp.
    var env: List[String] = ["A=1", "B=two words"]
    var e = _run("/usr/bin/env", ["env"], env, "")
    assert_equal(e[1], "A=1\nB=two words\n")
    var e0 = _run("/usr/bin/env", ["env"], List[String](), "")
    assert_equal(e0[1], "")
    # chdir file action.
    var out_buf = List[UInt8](length=PATH_BUF_SIZE, fill=0)
    var tmp = CStr("/tmp")
    var rp = c_realpath(tmp.addr(), buf_addr(out_buf))
    assert_true(rp != 0)
    var real_tmp = c_string_at(rp)
    var w = _run("/bin/pwd", ["pwd"], List[String](), "/tmp")
    assert_equal(w[1], real_tmp + "\n")
    # Exit codes.
    var f = _run("/bin/sh", ["sh", "-c", "exit 3"], List[String](), "")
    assert_equal(f[0], 3)
    # Process group kill.
    var pid = _spawn("/bin/sleep", ["sleep", "30"], List[String](), "", -1, True)
    assert_equal(c_kill(-pid, SIGKILL), 0)
    var st = _wait_status(pid)
    assert_equal(_exit_code(st), -SIGKILL)
    # WNOHANG on a running child returns 0.
    var pid2 = _spawn("/bin/sleep", ["sleep", "30"], List[String](), "", -1, False)
    var sb = List[UInt8](length=8, fill=0)
    assert_equal(c_waitpid(pid2, buf_addr(sb), WNOHANG), 0)
    assert_equal(c_kill(pid2, SIGTERM), 0)
    assert_equal(_exit_code(_wait_status(pid2)), -SIGTERM)
    _ = len(out_buf)
    _ = len(sb)
    tmp.keep()


def test_files() raises:
    var base = getenv("TMPDIR", "/tmp")
    if not base.endswith("/"):
        base += "/"
    var tmpl = CStr(base + "qumo-libc-XXXXXX")
    var d = c_mkdtemp(tmpl.addr())
    assert_true(d != 0)
    var dir = c_string_at(d)
    assert_true(dir.startswith(base + "qumo-libc-"))
    var dc = CStr(dir)
    assert_equal(c_chmod(dc.addr(), 0o700), 0)
    var renamed = dir + "-moved"
    var rc = CStr(renamed)
    assert_equal(c_rename(dc.addr(), rc.addr()), 0)
    # chdir there and back.
    var cwd_buf = List[UInt8](length=PATH_BUF_SIZE, fill=0)
    var dot = CStr(".")
    var cwd = c_string_at(c_realpath(dot.addr(), buf_addr(cwd_buf)))
    assert_equal(c_chdir(rc.addr()), 0)
    var back = CStr(cwd)
    assert_equal(c_chdir(back.addr()), 0)
    rmdir(renamed)
    assert_equal(c_chdir(rc.addr()), -1)
    assert_equal(errno(), ENOENT)
    _ = len(cwd_buf)
    tmpl.keep()
    dc.keep()
    rc.keep()
    dot.keep()
    back.keep()


def test_time() raises:
    var t = c_time()
    assert_true(t > 1700000000)
    var ts = List[Int64](length=2, fill=0)
    assert_equal(c_clock_gettime(CLOCK_REALTIME, Int(ts.unsafe_ptr())), 0)
    assert_true(Int(ts[0]) >= t - 2 and Int(ts[1]) >= 0 and Int(ts[1]) < 1000000000)
    var tt = List[Int64](length=1, fill=0)
    tt[0] = 1700000000
    var tm = List[UInt8](length=SIZEOF_TM, fill=0)
    assert_true(c_gmtime_r(Int(tt.unsafe_ptr()), buf_addr(tm)) != 0)
    assert_equal(load_i32(buf_addr(tm) + TM_YEAR_OFFSET), 123)
    assert_equal(load_i32(buf_addr(tm) + TM_MON_OFFSET), 10)
    assert_equal(load_i32(buf_addr(tm) + TM_MDAY_OFFSET), 14)
    assert_equal(load_i32(buf_addr(tm) + TM_HOUR_OFFSET), 22)
    var out = List[UInt8](length=64, fill=0)
    var fmt = CStr("%Y-%m-%dT%H:%M:%S")
    var n = c_strftime(buf_addr(out), 64, fmt.addr(), buf_addr(tm))
    assert_equal(String(from_utf8_lossy=Span(out)[0:n]), "2023-11-14T22:13:20")
    assert_equal(c_timegm(buf_addr(tm)), 1700000000)
    var lt = List[UInt8](length=SIZEOF_TM, fill=0)
    assert_true(c_localtime_r(Int(tt.unsafe_ptr()), buf_addr(lt)) != 0)
    assert_equal(load_i32(buf_addr(lt) + TM_YEAR_OFFSET), 123)
    _ = len(ts)
    _ = len(tt)
    _ = len(tm)
    _ = len(out)
    _ = len(lt)
    fmt.keep()


def test_sockets() raises:
    # inet_pton detects IP literals.
    var dst = List[UInt8](length=16, fill=0)
    var v4 = CStr("127.0.0.1")
    var v6 = CStr("::1")
    var name = CStr("localhost")
    assert_equal(c_inet_pton(AF_INET, v4.addr(), buf_addr(dst)), 1)
    assert_equal(c_inet_pton(AF_INET6, v6.addr(), buf_addr(dst)), 1)
    assert_equal(c_inet_pton(AF_INET, name.addr(), buf_addr(dst)), 0)
    # A listener on 127.0.0.1 with a kernel-chosen port.
    var lfd = c_socket(AF_INET, SOCK_STREAM, 0)
    assert_true(lfd >= 0)
    var one = List[UInt8](length=4, fill=0)
    store_u32(buf_addr(one), 1)
    assert_equal(c_setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, buf_addr(one), 4), 0)
    var sa = List[UInt8](length=16, fill=0)
    if is_macos():
        sa[0] = 16
        sa[1] = UInt8(AF_INET)
    else:
        sa[0] = UInt8(AF_INET)
    sa[4] = 127
    sa[7] = 1
    assert_equal(c_bind(lfd, buf_addr(sa), 16), 0)
    assert_equal(c_listen(lfd, 4), 0)
    var slen = List[UInt8](length=4, fill=0)
    store_u32(buf_addr(slen), 16)
    assert_equal(c_getsockname(lfd, buf_addr(sa), buf_addr(slen)), 0)
    var port = Int(sa[2]) * 256 + Int(sa[3])
    assert_true(port > 0)
    # Resolve it with getaddrinfo (numeric host and service) and connect.
    var hints = List[UInt8](length=SIZEOF_ADDRINFO, fill=0)
    store_u32(buf_addr(hints) + AI_FAMILY_OFFSET, AF_INET)
    store_u32(buf_addr(hints) + AI_SOCKTYPE_OFFSET, SOCK_STREAM)
    var res = List[UInt8](length=8, fill=0)
    var port_c = CStr(String(port))
    assert_equal(c_getaddrinfo(v4.addr(), port_c.addr(), buf_addr(hints), buf_addr(res)), 0)
    var ai = load_i64(buf_addr(res))
    assert_true(ai != 0)
    assert_equal(load_i32(ai + AI_FAMILY_OFFSET), AF_INET)
    assert_equal(load_u32(ai + AI_ADDRLEN_OFFSET), 16)
    var cfd = c_socket(load_i32(ai + AI_FAMILY_OFFSET), load_i32(ai + AI_SOCKTYPE_OFFSET), load_i32(ai + AI_PROTOCOL_OFFSET))
    assert_true(cfd >= 0)
    if is_macos():
        assert_equal(c_setsockopt(cfd, SOL_SOCKET, SO_NOSIGPIPE, buf_addr(one), 4), 0)
    var tv = List[UInt8](length=SIZEOF_TIMEVAL, fill=0)
    store_i64(buf_addr(tv), 2)
    assert_equal(c_setsockopt(cfd, SOL_SOCKET, SO_RCVTIMEO, buf_addr(tv), SIZEOF_TIMEVAL), 0)
    assert_equal(c_setsockopt(cfd, SOL_SOCKET, SO_SNDTIMEO, buf_addr(tv), SIZEOF_TIMEVAL), 0)
    assert_equal(c_connect(cfd, load_i64(ai + AI_ADDR_OFFSET), load_u32(ai + AI_ADDRLEN_OFFSET)), 0)
    c_freeaddrinfo(ai)
    var err = List[UInt8](length=4, fill=0)
    var elen = List[UInt8](length=4, fill=0)
    store_u32(buf_addr(elen), 4)
    assert_equal(c_getsockopt(cfd, SOL_SOCKET, SO_ERROR, buf_addr(err), buf_addr(elen)), 0)
    assert_equal(load_i32(buf_addr(err)), 0)
    var afd = c_accept(lfd, 0, 0)
    assert_true(afd >= 0)
    var msg = CStr("hi")
    assert_equal(c_write(cfd, msg.addr(), 2), 2)
    var buf = List[UInt8](length=8, fill=0)
    assert_equal(c_read(afd, buf_addr(buf), 8), 2)
    _ = c_close(afd)
    _ = c_close(cfd)
    _ = c_close(lfd)
    # gai_strerror for an unknown host.
    var bad = CStr("no-such-host.invalid")
    var rc = c_getaddrinfo(bad.addr(), port_c.addr(), buf_addr(hints), buf_addr(res))
    assert_true(rc != 0)
    assert_true(c_string_at(c_gai_strerror(rc)).byte_length() > 0)
    _ = len(dst)
    _ = len(one)
    _ = len(sa)
    _ = len(slen)
    _ = len(hints)
    _ = len(res)
    _ = len(tv)
    _ = len(err)
    _ = len(elen)
    _ = len(buf)
    v4.keep()
    v6.keep()
    name.keep()
    port_c.keep()
    msg.keep()
    bad.keep()


def test_regex_fnmatch() raises:
    var re = List[UInt64](length=64, fill=0)
    var ra = Int(re.unsafe_ptr())
    var pat = CStr("ab+c")
    assert_equal(c_regcomp(ra, pat.addr(), REG_EXTENDED | REG_NOSUB), 0)
    var s1 = CStr("xxabbbcx")
    var s2 = CStr("xxacx")
    assert_equal(c_regexec(ra, s1.addr(), 0, 0, 0), 0)
    assert_equal(c_regexec(ra, s2.addr(), 0, 0, 0), REG_NOMATCH)
    c_regfree(ra)
    var ipat = CStr("HELLO")
    var s3 = CStr("say hello")
    assert_equal(c_regcomp(ra, ipat.addr(), REG_EXTENDED | REG_NOSUB | REG_ICASE), 0)
    assert_equal(c_regexec(ra, s3.addr(), 0, 0, 0), 0)
    c_regfree(ra)
    # REG_NEWLINE: "^b" matches after a newline only with the flag.
    var nl = CStr("^b")
    var s4 = CStr("a\nb")
    assert_equal(c_regcomp(ra, nl.addr(), REG_EXTENDED | REG_NOSUB), 0)
    assert_equal(c_regexec(ra, s4.addr(), 0, 0, 0), REG_NOMATCH)
    c_regfree(ra)
    assert_equal(c_regcomp(ra, nl.addr(), REG_EXTENDED | REG_NOSUB | REG_NEWLINE), 0)
    assert_equal(c_regexec(ra, s4.addr(), 0, 0, 0), 0)
    c_regfree(ra)
    var badp = CStr("a(b")
    var code = c_regcomp(ra, badp.addr(), REG_EXTENDED)
    assert_true(code != 0)
    var eb = List[UInt8](length=256, fill=0)
    var n = c_regerror(code, ra, buf_addr(eb), 256)
    assert_true(n > 1)
    assert_true(String(from_utf8_lossy=Span(eb)[0:n - 1]).byte_length() > 0)
    # fnmatch flags.
    var star = CStr("*.py")
    var deep = CStr("src/app/main.py")
    var hidden = CStr(".x.py")
    assert_equal(c_fnmatch(star.addr(), deep.addr(), 0), 0)
    assert_equal(c_fnmatch(star.addr(), deep.addr(), FNM_PATHNAME), FNM_NOMATCH)
    assert_equal(c_fnmatch(star.addr(), hidden.addr(), 0), 0)
    assert_equal(c_fnmatch(star.addr(), hidden.addr(), FNM_PERIOD), FNM_NOMATCH)
    var esc = CStr("a\\*")
    var lit = CStr("a\\b")
    var star_name = CStr("a*")
    assert_equal(c_fnmatch(esc.addr(), star_name.addr(), 0), 0)
    assert_equal(c_fnmatch(esc.addr(), lit.addr(), 0), FNM_NOMATCH)
    assert_equal(c_fnmatch(esc.addr(), lit.addr(), FNM_NOESCAPE), 0)
    _ = len(re)
    _ = len(eb)
    pat.keep()
    s1.keep()
    s2.keep()
    ipat.keep()
    s3.keep()
    nl.keep()
    s4.keep()
    badp.keep()
    star.keep()
    deep.keep()
    hidden.keep()
    esc.keep()
    lit.keep()
    star_name.keep()


def main() raises:
    test_strings_and_memory()
    test_errno_and_symbols()
    test_pipes_poll_sigpipe()
    test_spawn()
    test_files()
    test_time()
    test_sockets()
    test_regex_fnmatch()
    print("test_native_libc OK")
