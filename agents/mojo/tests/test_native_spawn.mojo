"""Tests for qumo_agent.native.spawn (DESIGN.md 2.1.2, 8.2).

System programs (/bin/echo, /usr/bin/env, /bin/sh, ...) are started as fixtures; the
code under test never starts a shell itself.
"""

from std.testing import assert_equal, assert_true, assert_false
from std.os import setenv, unsetenv

from qumo_agent.native.spawn import *
from qumo_agent.native.libc import c_kill, errno, ESRCH
from qumo_agent.native.clock import monotonic_ms, sleep_ms
from qumo_agent.native.fs import make_temp_dir, realpath_strict, remove_tree, write_text, chmod


def _spec(path: String, argv: List[String], env: List[String], cwd: String, out_mode: Int, err_mode: Int) -> SpawnSpec:
    return SpawnSpec(path, argv.copy(), env.copy(), cwd, STDIO_NULL, out_mode, err_mode, False)


def _base_env() -> List[String]:
    var e: List[String] = ["PATH=/usr/bin:/bin"]
    return e^


def test_argv_and_env() raises:
    var r = run_capture(_spec("/bin/echo", ["echo", "ab lit", "x"], _base_env(), "", STDIO_PIPE, STDIO_PIPE), 10000, 1 << 20, "")
    assert_equal(r.exit_code, 0)
    assert_equal(r.stdout, "ab lit x\n")
    assert_false(r.timed_out)
    var env: List[String] = ["A=1", "B=two words", "C="]
    r = run_capture(_spec("/usr/bin/env", ["env"], env, "", STDIO_PIPE, STDIO_PIPE), 10000, 1 << 20, "")
    assert_equal(r.stdout, "A=1\nB=two words\nC=\n")
    r = run_capture(_spec("/usr/bin/env", ["env"], List[String](), "", STDIO_PIPE, STDIO_PIPE), 10000, 1 << 20, "")
    assert_equal(r.stdout, "")
    assert_equal(r.exit_code, 0)
    # bare name resolved through the child's PATH
    r = run_capture(_spec("echo", ["echo", "found"], _base_env(), "", STDIO_PIPE, STDIO_PIPE), 10000, 1 << 20, "")
    assert_equal(r.stdout, "found\n")
    try:
        _ = run_capture(_spec("no-such-program-qumo", ["x"], _base_env(), "", STDIO_PIPE, STDIO_PIPE), 1000, 100, "")
        assert_true(False, "an unknown program must raise")
    except e:
        assert_equal(String(e), "QA-E160: [Errno 2] No such file or directory: 'no-such-program-qumo'")
    try:
        _ = run_capture(_spec("/nonexistent/prog", ["x"], _base_env(), "", STDIO_PIPE, STDIO_PIPE), 1000, 100, "")
        assert_true(False, "a missing path must raise")
    except e:
        assert_equal(String(e), "QA-E160: [Errno 2] No such file or directory: '/nonexistent/prog'")
    # exit codes and signals
    r = run_capture(_spec("/bin/sh", ["sh", "-c", "exit 3"], _base_env(), "", STDIO_PIPE, STDIO_PIPE), 10000, 100, "")
    assert_equal(r.exit_code, 3)
    r = run_capture(_spec("/bin/sh", ["sh", "-c", "kill -TERM $$"], _base_env(), "", STDIO_PIPE, STDIO_PIPE), 10000, 100, "")
    assert_equal(r.exit_code, -15)


def test_cwd() raises:
    var t = make_temp_dir("qumo-spawn-cwd-")
    var r = run_capture(_spec("/bin/pwd", ["pwd", "-P"], _base_env(), t, STDIO_PIPE, STDIO_PIPE), 10000, 4096, "")
    assert_equal(r.stdout, realpath_strict(t) + "\n")
    # a relative program path is resolved against cwd
    write_text(t + "/x.sh", "#!/bin/sh\necho rel $1\n")
    chmod(t + "/x.sh", 0o755)
    var r2 = run_capture(_spec("./x.sh", ["x.sh", "arg"], _base_env(), t, STDIO_PIPE, STDIO_PIPE), 10000, 4096, "")
    assert_equal(r2.stdout, "rel arg\n")
    remove_tree(t)
    try:
        _ = run_capture(_spec("/bin/pwd", ["pwd"], _base_env(), t + "/gone", STDIO_PIPE, STDIO_PIPE), 1000, 100, "")
        assert_true(False, "a missing cwd must raise")
    except e:
        assert_true(String(e).startswith("QA-E160: [Errno 2] No such file or directory"), String(e))


def test_pipes_and_lines() raises:
    var spec = SpawnSpec("/bin/cat", ["cat"], _base_env(), "", STDIO_PIPE, STDIO_PIPE, STDIO_PIPE, False)
    var c = spawn(spec)
    var line = String()
    c.write_all("hello\nwor".as_bytes())
    assert_equal(c.read_line(line, 5000, 1000), READ_LINE)
    assert_equal(line, "hello")
    assert_equal(c.read_line(line, 200, 1000), READ_TIMEOUT)
    c.write_all("ld\n".as_bytes())
    assert_equal(c.read_line(line, 5000, 1000), READ_LINE)
    assert_equal(line, "world")
    c.write_all("tail".as_bytes())
    c.close_stdin()
    assert_equal(c.read_line(line, 5000, 1000), READ_LINE)
    assert_equal(line, "tail")
    assert_equal(c.read_line(line, 5000, 1000), READ_EOF)
    assert_equal(c.wait(), 0)
    assert_equal(c.poll_exit().value(), 0)
    # too long a line
    var c2 = spawn(spec)
    var big = String()
    for _ in range(100):
        big += "0123456789"
    c2.write_all(big.as_bytes())
    try:
        _ = c2.read_line(line, 5000, 500)
        assert_true(False, "a long line must raise")
    except e:
        assert_equal(String(e), "QA-E161: line longer than 500 bytes")
    _ = c2.terminate(500)
    # read_available
    var c3 = spawn(spec)
    c3.write_all("abc".as_bytes())
    var got = List[UInt8]()
    var n = c3.read_available(c3.stdout_fd, got, 5000)
    assert_equal(n, 3)
    assert_equal(String(from_utf8_lossy=Span(got)), "abc")
    assert_equal(c3.read_available(c3.stdout_fd, got, 100), -1)
    c3.close_stdin()
    assert_equal(c3.read_available(c3.stdout_fd, got, 5000), 0)
    assert_equal(c3.wait(), 0)


def test_timeout_and_group_kill() raises:
    var t0 = monotonic_ms()
    var r = run_capture(_spec("/bin/sleep", ["sleep", "30"], _base_env(), "", STDIO_PIPE, STDIO_PIPE), 300, 100, "")
    var dt = monotonic_ms() - t0
    assert_true(r.timed_out)
    assert_equal(r.exit_code, -9)
    assert_true(dt < 1300, "timeout took " + String(dt) + " ms")
    # The group kill reaches a grandchild.
    var spec = SpawnSpec("/bin/sh", ["sh", "-c", "sleep 30 & echo $!; wait"], _base_env(), "", STDIO_NULL, STDIO_PIPE, STDIO_PIPE, True)
    var c = spawn(spec)
    var line = String()
    assert_equal(c.read_line(line, 5000, 100), READ_LINE)
    var gpid = Int(line)
    assert_true(gpid > 0)
    assert_equal(c_kill(gpid, 0), 0)
    var code = c.terminate(200)
    assert_equal(code, -15)
    var gone = False
    var deadline = monotonic_ms() + 3000
    while monotonic_ms() < deadline:
        if c_kill(gpid, 0) != 0:
            gone = True
            break
        sleep_ms(20)
    assert_true(gone, "grandchild " + String(gpid) + " survived the group kill")
    # A timeout in run_capture also kills the group.
    var spec2 = SpawnSpec("/bin/sh", ["sh", "-c", "sleep 30 & echo $!; wait"], _base_env(), "", STDIO_NULL, STDIO_PIPE, STDIO_PIPE, True)
    var r2 = run_capture(spec2, 500, 1000, "")
    assert_true(r2.timed_out)
    var gpid2 = Int(String(r2.stdout.strip()))
    deadline = monotonic_ms() + 3000
    gone = False
    while monotonic_ms() < deadline:
        if c_kill(gpid2, 0) != 0:
            gone = True
            break
        sleep_ms(20)
    assert_true(gone)


def test_dead_child_write() raises:
    ignore_sigpipe()
    var spec = SpawnSpec("/usr/bin/true", ["true"], _base_env(), "", STDIO_PIPE, STDIO_NULL, STDIO_NULL, False)
    var c = spawn(spec)
    assert_equal(c.wait(), 0)
    var data = List[UInt8](length=1 << 20, fill=65)
    try:
        c.write_all(Span(data))
        assert_true(False, "writing to a dead child must raise")
    except e:
        assert_equal(String(e), "QA-E160: child closed its input")


def test_stdio_modes() raises:
    var r = run_capture(_spec("/bin/sh", ["sh", "-c", "echo out; echo err >&2"], _base_env(), "", STDIO_PIPE, STDIO_MERGE), 10000, 1000, "")
    assert_equal(r.stdout, "out\nerr\n")
    assert_equal(r.stderr, "")
    r = run_capture(_spec("/bin/sh", ["sh", "-c", "echo out; echo err >&2"], _base_env(), "", STDIO_PIPE, STDIO_PIPE), 10000, 1000, "")
    assert_equal(r.stdout, "out\n")
    assert_equal(r.stderr, "err\n")
    r = run_capture(_spec("/bin/sh", ["sh", "-c", "echo out; echo err >&2"], _base_env(), "", STDIO_NULL, STDIO_NULL), 10000, 1000, "")
    assert_equal(r.stdout, "")
    assert_equal(r.exit_code, 0)
    # inherited stderr stays open in the child (POSIX_SPAWN_CLOEXEC_DEFAULT on macOS)
    r = run_capture(_spec("/bin/sh", ["sh", "-c", ": >&2"], _base_env(), "", STDIO_PIPE, STDIO_INHERIT), 10000, 1000, "")
    assert_equal(r.exit_code, 0)
    # stdin data and output caps
    var cat = SpawnSpec("/bin/cat", ["cat"], _base_env(), "", STDIO_NULL, STDIO_PIPE, STDIO_PIPE, False)
    r = run_capture(cat, 10000, 1000, "abc")
    assert_equal(r.stdout, "abc")
    assert_false(r.stdout_truncated)
    var big = String()
    for i in range(100000):
        big += String(i % 10)
    big += "END"
    r = run_capture(cat, 20000, 1000, big)
    assert_true(r.stdout_truncated)
    assert_equal(r.stdout.byte_length(), 1000)
    assert_true(r.stdout.endswith("789END"))
    assert_equal(r.exit_code, 0)


def test_stderr_flood() raises:
    var script = "head -c 1048576 /dev/zero | tr '\\000' x >&2; echo done; sleep 30"
    var spec = SpawnSpec("/bin/sh", ["sh", "-c", script], _base_env(), "", STDIO_NULL, STDIO_PIPE, STDIO_PIPE, True)
    var c = spawn(spec)
    var line = String()
    assert_equal(c.read_line(line, 20000, 1000), READ_LINE)
    assert_equal(line, "done")
    var tail = c.stderr_tail()
    assert_equal(tail.byte_length(), 65536)
    assert_true(tail.startswith("xxxx") and tail.endswith("xxxx"))
    _ = c.terminate(100)


def test_envs() raises:
    _ = setenv("X_API_KEY", "k1")
    _ = setenv("GITHUB_TOKEN", "t1")
    _ = setenv("MY_SECRET", "s1")
    _ = setenv("AWS_SECRET_ACCESS_KEY", "a1")
    _ = setenv("OPENROUTER_API_KEY", "or1")
    _ = setenv("QUMO_ROOT", "/r")
    _ = setenv("QUMO_AGENT_DETERMINISTIC", "1")
    _ = setenv("SSL_CERT_FILE", "/c.pem")
    _ = setenv("TERM", "dumb")
    var m = minimal_child_env()
    for e in m:
        assert_false(e.startswith("X_API_KEY=") or e.startswith("GITHUB_TOKEN=") or e.startswith("MY_SECRET="))
    var has_path = False
    for e in m:
        if e.startswith("PATH="):
            has_path = True
    assert_true(has_path)
    var mcp = mcp_server_env()
    var allowed: List[String] = ["HOME", "LOGNAME", "PATH", "SHELL", "TERM", "USER"]
    for e in mcp:
        var name = String(e[byte=0 : e.find("=")])
        var ok = False
        for a in allowed:
            if a == name:
                ok = True
        assert_true(ok, "unexpected " + name)
    var h = harness_child_env("OPENROUTER_API_KEY")
    var joined = String("\n").join(h)
    assert_true(joined.find("OPENROUTER_API_KEY=or1") >= 0)
    assert_true(joined.find("QUMO_ROOT=/r") >= 0)
    assert_true(joined.find("QUMO_AGENT_DETERMINISTIC=1") >= 0)
    assert_true(joined.find("GITHUB_TOKEN=t1") >= 0)
    assert_true(joined.find("SSL_CERT_FILE=/c.pem") >= 0)
    assert_true(joined.find("AWS_SECRET_ACCESS_KEY") < 0)
    assert_true(joined.find("X_API_KEY") < 0)
    assert_true(joined.find("MY_SECRET") < 0)
    var bad = harness_child_env("BAD NAME=1")
    assert_true(String("\n").join(bad).find("BAD") < 0)
    # no duplicate names
    var dup = harness_child_env("PATH")
    var n_path = 0
    for e in dup:
        if e.startswith("PATH="):
            n_path += 1
    assert_equal(n_path, 1)
    assert_true(is_secret_env_name("db_password") and is_secret_env_name("A_SECRET_X") and is_secret_env_name("x_token"))
    assert_false(is_secret_env_name("TOKENS") or is_secret_env_name("PATH"))
    var sub = env_subset(["QUMO_ROOT", "NOT_SET_QUMO_X", "QUMO_ROOT"])
    assert_equal(len(sub), 1)
    assert_equal(sub[0], "QUMO_ROOT=/r")
    _ = setenv("QUMO_EMPTY_X", "")
    assert_equal(len(env_subset(["QUMO_EMPTY_X"])), 1)
    var full = environ_list()
    var saw = False
    for e in full:
        if e == "QUMO_ROOT=/r":
            saw = True
    assert_true(saw)
    for n in ["X_API_KEY", "GITHUB_TOKEN", "MY_SECRET", "AWS_SECRET_ACCESS_KEY", "OPENROUTER_API_KEY", "QUMO_ROOT", "QUMO_AGENT_DETERMINISTIC", "SSL_CERT_FILE", "QUMO_EMPTY_X"]:
        _ = unsetenv(n)


def test_which() raises:
    var sh = which("sh", "/nonexistent:/usr/bin:/bin")
    assert_true(sh.endswith("/sh"), sh)
    assert_equal(which("definitely-not-here-qumo", "/usr/bin:/bin"), "")
    assert_equal(which("/bin/sh", ""), "/bin/sh")
    assert_equal(which("/bin", "/usr/bin"), "")
    assert_equal(which("", "/bin"), "")
    assert_equal(which("bin", "/"), "")   # a directory is not a program


def main() raises:
    ignore_sigpipe()
    test_argv_and_env()
    test_cwd()
    test_pipes_and_lines()
    test_timeout_and_group_kill()
    test_dead_child_write()
    test_stdio_modes()
    test_stderr_flood()
    test_envs()
    test_which()
    print("test_native_spawn OK")
