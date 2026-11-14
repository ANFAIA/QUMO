"""Tests for qumo_agent.native.sandbox (K5; DESIGN.md 2.1.8, 6.8, 8.2).

The confinement tests run for real on a machine with an OS sandbox (macOS
sandbox-exec here; Linux Landlock when run there). They build the test program
tests/fixtures/native/trampoline.mojo (it dispatches `__confine` and `__exec-in` as
main.mojo does) into a fresh temp directory, so a stale binary is never used.
Run from the repository root.
"""

from std.testing import assert_equal, assert_true, assert_false
from std.os import getenv, mkdir, symlink as std_symlink

from qumo_agent.native.sandbox import *
from qumo_agent.native.spawn import (
    SpawnSpec,
    STDIO_NULL,
    STDIO_PIPE,
    run_capture,
    environ_list,
    minimal_child_env,
)
from qumo_agent.native.fs import (
    make_temp_dir,
    realpath_strict,
    remove_tree,
    exists,
    write_text,
    copy_file,
    make_dirs,
)
from qumo_agent.native.libc import (
    is_macos,
    buf_addr,
    load_u16,
    store_u16,
    c_socket,
    c_bind,
    c_listen,
    c_getsockname,
    c_close,
    AF_INET,
    SOCK_STREAM,
    SIGXCPU,
    SIGKILL,
)
from qumo_agent.native.clock import monotonic_ms


def _spec(path: String, argv: List[String], env: List[String]) -> SpawnSpec:
    return SpawnSpec(path, argv.copy(), env.copy(), String(), STDIO_NULL, STDIO_PIPE, STDIO_PIPE, True)


def _q(s: String) -> String:
    return '"' + s + '"'


def _has(items: List[String], p: String) -> Bool:
    for i in items:
        if i == p:
            return True
    return False


def test_json_round_trip() raises:
    var c = Confinement(CONF_TOOL, ["/tmp/a b", "/x\"y"], ["/lib\\z"], "/Users/u", 14, 67108864)
    var j = c.to_json()
    assert_equal(j, '{"profile":3,"write_dirs":["/tmp/a b","/x\\"y"],"read_allow":["/lib\\\\z"],"home":"/Users/u","cpu_s":14,"fsize_bytes":67108864}')
    var back = confinement_from_json(j)
    assert_equal(back.profile, CONF_TOOL)
    assert_equal(len(back.write_dirs), 2)
    assert_equal(back.write_dirs[1], "/x\"y")
    assert_equal(back.read_allow[0], "/lib\\z")
    assert_equal(back.home, "/Users/u")
    assert_equal(back.cpu_s, 14)
    assert_equal(back.fsize_bytes, 67108864)
    for bad in ['[]', '{"profile":9,"write_dirs":[],"read_allow":[],"home":"","cpu_s":0,"fsize_bytes":0}', '{"profile":1}', '{"profile":1,"write_dirs":[1],"read_allow":[],"home":"","cpu_s":0,"fsize_bytes":0}']:
        try:
            _ = confinement_from_json(bad)
            assert_true(False, "accepted " + bad)
        except e:
            assert_true(String(e).startswith("QA-E163: OS confinement failed: malformed confinement: "), String(e))


def test_profiles() raises:
    var home = realpath_strict(make_temp_dir("qumo-home-"))
    mkdir(home + "/.ssh")
    mkdir(home + "/.some-new-cli")
    mkdir(home + "/.local")
    mkdir(home + "/visible")
    mkdir(home + "/real_config")
    std_symlink(home + "/real_config", home + "/.config")
    var tool = Confinement(CONF_TOOL, ["/w/one", "/w/q\"uote\\d"], ["/r/lib"], home, 0, 0)
    assert_equal(
        macos_profile(tool),
        '(version 1)(allow default)(deny network*)(deny file-write*)'
        + '(allow file-write* (literal "/dev/null") (subpath "/w/one") (subpath "/w/q\\"uote\\\\d"))'
        + '(deny process-fork)(deny file-read* (subpath ' + _q(home) + '))(allow file-read* (subpath "/r/lib"))',
    )
    var no_allow = Confinement(CONF_TOOL, List[String](), List[String](), home, 0, 0)
    assert_equal(
        macos_profile(no_allow),
        '(version 1)(allow default)(deny network*)(deny file-write*)(allow file-write* (literal "/dev/null"))'
        + '(deny process-fork)(deny file-read* (subpath ' + _q(home) + '))',
    )
    var sens = sensitive_home_paths(home)
    var want: List[String] = [
        home + "/.ssh", home + "/.aws", home + "/.azure", home + "/.gnupg", home + "/real_config", home + "/.config",
        home + "/.docker", home + "/.kube", home + "/.netrc", home + "/.git-credentials", home + "/.pypirc",
        home + "/.npmrc", home + "/.zsh_history", home + "/.bash_history", home + "/Library/Keychains",
    ]
    # The fixed names come first (existing ones as realpaths), then the other dot entries.
    for i in range(len(want)):
        assert_equal(sens[i], want[i])
    for extra in [".codex", ".claude", ".claude.json", ".gemini", ".hf-cli", ".qiskit", ".cache", ".gitconfig",
                  ".zshrc", "Library/Application Support"]:
        assert_true(_has(sens, home + "/" + extra), "not denied: " + extra)
    assert_true(_has(sens, home + "/.some-new-cli"), "an unlisted dot directory must be denied")
    assert_false(_has(sens, home + "/.local"))
    assert_false(_has(sens, home + "/visible"))
    var deny = String("(deny file-read*")
    for p in sens:
        deny += " (subpath " + _q(p) + ")"
    deny += ")"
    for prof in [CONF_CHECK, CONF_BUILD]:
        var c = Confinement(prof, ["/sandbox"], List[String](), home, 0, 0)
        assert_equal(
            macos_profile(c),
            '(version 1)(allow default)(deny network*)(deny file-write*)(allow file-write* (literal "/dev/null") (subpath "/sandbox"))' + deny,
        )
    # What the caller names below $HOME stays readable even inside a denied directory.
    var inner = Confinement(CONF_BUILD, [home + "/.cache/work"], [home + "/.local/lib", "/elsewhere"], home, 0, 0)
    assert_equal(
        macos_profile(inner),
        '(version 1)(allow default)(deny network*)(deny file-write*)(allow file-write* (literal "/dev/null") (subpath ' + _q(home + "/.cache/work") + '))'
        + deny + '(allow file-read* (subpath ' + _q(home + "/.cache/work") + ') (subpath ' + _q(home + "/.local/lib") + '))',
    )
    assert_equal(len(sensitive_home_paths("")), 0)
    try:
        _ = macos_profile(Confinement(CONF_TOOL, ["/a\nb"], List[String](), "", 0, 0))
        assert_true(False, "a control character must be refused")
    except e:
        assert_true(String(e).startswith("QA-E163: OS confinement failed: path with a control character"), String(e))
    try:
        _ = macos_profile(Confinement(7, List[String](), List[String](), "", 0, 0))
        assert_true(False)
    except e:
        assert_true(String(e).startswith("QA-E163"))
    remove_tree(home)


def test_confine_shape(self_bin: String) raises:
    var c = Confinement(CONF_CHECK, ["/tmp"], List[String](), "", 0, 0)
    var env: List[String] = ["PATH=/usr/bin:/bin", "A=1"]
    var spec = SpawnSpec("echo", ["echo", "x y", "z"], env.copy(), "/tmp", STDIO_NULL, STDIO_PIPE, STDIO_PIPE, True)
    var out = confine(spec, c, self_bin)
    assert_equal(out.path, self_bin)
    assert_equal(len(out.argv), 6)
    assert_equal(out.argv[0], self_bin)
    assert_equal(out.argv[1], "__confine")
    assert_equal(out.argv[2], c.to_json())
    assert_true(out.argv[3] == "/usr/bin/echo" or out.argv[3] == "/bin/echo", out.argv[3])
    assert_equal(out.argv[4], "x y")
    assert_equal(out.argv[5], "z")
    assert_equal(len(out.env), 2)
    assert_equal(out.env[1], "A=1")
    assert_equal(out.cwd, "/tmp")
    assert_equal(out.stdin_mode, STDIO_NULL)
    assert_equal(out.stdout_mode, STDIO_PIPE)
    assert_true(out.new_process_group)
    try:
        _ = confine(SpawnSpec("no-such-prog-qumo", ["x"], env.copy(), "", STDIO_NULL, STDIO_NULL, STDIO_NULL, False), c, self_bin)
        assert_true(False)
    except e:
        assert_equal(String(e), "QA-E160: [Errno 2] No such file or directory: 'no-such-prog-qumo'")


def _build_fixture(dir: String) raises -> String:
    var out = dir + "/native_trampoline"
    var argv: List[String] = [
        ".venv/bin/mojo", "build", "-I", ".", "-I", "agents/mojo", "-o", out,
        "agents/mojo/tests/fixtures/native/trampoline.mojo",
    ]
    var r = run_capture(_spec(".venv/bin/mojo", argv, environ_list()), 600000, 20000, "")
    assert_equal(r.exit_code, 0, "fixture build failed: " + r.stdout + r.stderr)
    return out^


def _listener() raises -> Tuple[Int, Int]:
    var fd = c_socket(AF_INET, SOCK_STREAM, 0)
    assert_true(fd >= 0)
    var sa = List[UInt8](length=16, fill=0)
    if is_macos():
        sa[0] = 16
        sa[1] = UInt8(AF_INET)
    else:
        store_u16(buf_addr(sa), AF_INET)
    sa[4] = 127
    sa[7] = 1
    assert_equal(c_bind(fd, buf_addr(sa), 16), 0)
    assert_equal(c_listen(fd, 4), 0)
    var ln = List[UInt8](length=4, fill=0)
    ln[0] = 16
    assert_equal(c_getsockname(fd, buf_addr(sa), buf_addr(ln)), 0)
    var port = (Int(sa[2]) << 8) | Int(sa[3])
    _ = len(sa)
    _ = len(ln)
    return (fd, port)


def test_confined_runs(tramp: String, work: String) raises:
    var home = realpath_strict(getenv("HOME"))
    var repo = realpath_strict(".")
    var inside = realpath_strict(make_temp_dir("qumo-conf-in-"))
    var outside = realpath_strict(make_temp_dir("qumo-conf-out-"))
    var env = minimal_child_env()

    # Writes: only below write_dirs.
    var check = Confinement(CONF_CHECK, [inside], List[String](), home, 0, 0)
    var r = run_capture(confine(_spec("/usr/bin/touch", ["touch", outside + "/x"], env), check, tramp), 20000, 4000, "")
    assert_true(r.exit_code != 0, "touch outside write_dirs succeeded")
    assert_false(exists(outside + "/x"))
    r = run_capture(confine(_spec("/usr/bin/touch", ["touch", inside + "/x"], env), check, tramp), 20000, 4000, "")
    assert_equal(r.exit_code, 0, r.stderr)
    assert_true(exists(inside + "/x"))

    # CONF_TOOL: no reads under $HOME outside read_allow (the repository is under $HOME).
    var probe_dir = repo + "/build/test-fixtures"
    make_dirs(probe_dir)
    var probe = probe_dir + "/native_sandbox_read_probe.txt"
    write_text(probe, "readable")
    if probe.startswith(home + "/"):
        var tool = Confinement(CONF_TOOL, [inside], List[String](), home, 0, 0)
        r = run_capture(_spec("/bin/cat", ["cat", probe], env), 20000, 4000, "")
        assert_equal(r.stdout, "readable")
        r = run_capture(confine(_spec("/bin/cat", ["cat", probe], env), tool, tramp), 20000, 4000, "")
        assert_true(r.exit_code != 0, "CONF_TOOL read a file under $HOME")
        assert_equal(r.stdout, "")
    else:
        print("note: repository not under $HOME; the $HOME read check was skipped")

    # CONF_CHECK/CONF_BUILD: credential files other CLIs keep under $HOME are not readable
    # (a fake home, so no real secret is involved); the toolchain dirs and the caller's
    # allowed paths stay readable.
    if is_macos():
        var fake = realpath_strict(make_temp_dir("qumo-conf-home-"))
        for d in [".codex", ".some-new-cli", ".cache/huggingface", ".local/share", "plain"]:
            make_dirs(fake + "/" + d)
        write_text(fake + "/.codex/auth.json", "secret")
        write_text(fake + "/.some-new-cli/token", "secret")
        write_text(fake + "/.cache/huggingface/token", "secret")
        write_text(fake + "/.claude.json", "secret")
        write_text(fake + "/.local/share/ok", "fine")
        write_text(fake + "/plain/ok", "fine")
        for prof in [CONF_CHECK, CONF_BUILD]:
            var fc = Confinement(prof, [inside], List[String](), fake, 0, 0)
            for f in [".codex/auth.json", ".some-new-cli/token", ".cache/huggingface/token", ".claude.json"]:
                r = run_capture(confine(_spec("/bin/cat", ["cat", fake + "/" + f], env), fc, tramp), 20000, 4000, "")
                assert_true(r.exit_code != 0 and r.stdout.find("secret") < 0, "profile " + String(prof) + " read " + f)
            for f in [".local/share/ok", "plain/ok"]:
                r = run_capture(confine(_spec("/bin/cat", ["cat", fake + "/" + f], env), fc, tramp), 20000, 4000, "")
                assert_equal(r.stdout, "fine")
        var fa = Confinement(CONF_CHECK, [inside], [fake + "/.some-new-cli"], fake, 0, 0)
        r = run_capture(confine(_spec("/bin/cat", ["cat", fake + "/.some-new-cli/token"], env), fa, tramp), 20000, 4000, "")
        assert_equal(r.stdout, "secret")
        remove_tree(fake)

    # A reference Mojo binary copied to a temp dir runs under CONF_TOOL with an empty
    # environment, and cannot connect to a local listener.
    var tool_dir = realpath_strict(make_temp_dir("qumo-conf-tool-"))
    copy_file(tramp, tool_dir + "/tool", 0o755)
    var tool = Confinement(CONF_TOOL, [tool_dir], [modular_lib_dir(repo)], home, 30, 64 * 1024 * 1024)
    r = run_capture(confine(_spec(tool_dir + "/tool", ["tool", "hello"], List[String]()), tool, tramp), 60000, 4000, "")
    assert_equal(r.exit_code, 0, r.stderr)
    assert_equal(r.stdout, "hello from fixture\n")
    var lst = _listener()
    var port = String(lst[1])
    r = run_capture(_spec(tool_dir + "/tool", ["tool", "connect", port], List[String]()), 20000, 4000, "")
    assert_equal(r.stdout, "connected\n")
    r = run_capture(confine(_spec(tool_dir + "/tool", ["tool", "connect", port], List[String]()), tool, tramp), 20000, 4000, "")
    assert_true(r.stdout.startswith("connect failed"), r.stdout + r.stderr)
    _ = c_close(lst[0])
    # The tool cannot read its own home files either, but can read its temp dir.
    write_text(tool_dir + "/data.txt", "abc")
    r = run_capture(confine(_spec(tool_dir + "/tool", ["tool", "read", tool_dir + "/data.txt"], List[String]()), tool, tramp), 20000, 4000, "")
    assert_equal(r.stdout, "read ok 3\n")
    if probe.startswith(home + "/"):
        r = run_capture(confine(_spec(tool_dir + "/tool", ["tool", "read", probe], List[String]()), tool, tramp), 20000, 4000, "")
        assert_equal(r.stdout, "read failed\n")
    if is_macos():
        # no fork under CONF_TOOL
        # (sh runs the first command in a forked child; the last one it may exec directly)
        var fork_cmd = "/usr/bin/true && /bin/echo forked"
        r = run_capture(_spec("/bin/sh", ["sh", "-c", fork_cmd], env), 20000, 4000, "")
        assert_equal(r.stdout, "forked\n")
        r = run_capture(confine(_spec("/bin/sh", ["sh", "-c", fork_cmd], env), tool, tramp), 20000, 4000, "")
        assert_true(r.exit_code != 0 and r.stdout.find("forked") < 0, "fork allowed: " + r.stdout)

    # RLIMIT_CPU kills a busy loop (SIGXCPU; SIGKILL on Linux when soft == hard).
    # A shell loop is used: a Mojo-built binary installs a SIGXCPU handler (the
    # runtime's crash reporter prints a stack dump and the program keeps running),
    # and macOS sends nothing more at the hard limit, so for Mojo tools the caller's
    # timeout and process-group kill are what stop a busy loop.
    var cpu = Confinement(CONF_CHECK, [inside], List[String](), home, 1, 0)
    var t0 = monotonic_ms()
    r = run_capture(confine(_spec("/bin/sh", ["sh", "-c", "while :; do :; done"], env), cpu, tramp), 30000, 4000, "")
    var dt = monotonic_ms() - t0
    assert_false(r.timed_out)
    assert_true(r.exit_code == -SIGXCPU or r.exit_code == -SIGKILL, "exit " + String(r.exit_code))
    assert_true(dt < 15000)

    # The Linux trampoline forks and mirrors the child's status (an exit code, or the
    # signal that killed it); the same helper is exercised here on every platform.
    r = run_capture(_spec(tramp, [tramp, "fork-mirror", "exit 7"], env), 20000, 4000, "")
    assert_equal(r.exit_code, 7)
    r = run_capture(_spec(tramp, [tramp, "fork-mirror", "kill -9 $$"], env), 20000, 4000, "")
    assert_equal(r.exit_code, -SIGKILL)

    # __confine failures exit 125 with QA-E163 on stderr.
    r = run_capture(_spec(tramp, [tramp, "__confine", "not json", "/bin/true"], env), 20000, 4000, "")
    assert_equal(r.exit_code, 125)
    assert_true(r.stderr.startswith("QA-E163: OS confinement failed: "), r.stderr)
    r = run_capture(_spec(tramp, [tramp, "__confine"], env), 20000, 4000, "")
    assert_equal(r.exit_code, 125)
    assert_true(r.stderr.startswith("QA-E163: OS confinement failed: usage"), r.stderr)
    var bad = Confinement(CONF_TOOL, ["/a\x01b"], List[String](), home, 0, 0)
    r = run_capture(_spec(tramp, [tramp, "__confine", bad.to_json(), "/bin/true"], env), 20000, 4000, "")
    assert_equal(r.exit_code, 125)
    assert_true(r.stderr.startswith("QA-E163: OS confinement failed: path with a control character"), r.stderr)

    # __exec-in (the chdir trampoline for old glibc)
    r = run_capture(_spec(tramp, [tramp, "__exec-in", inside, "/bin/pwd", "-P"], env), 20000, 4000, "")
    assert_equal(r.exit_code, 0, r.stderr)
    assert_equal(r.stdout, inside + "\n")
    r = run_capture(_spec(tramp, [tramp, "__exec-in", inside + "/missing", "/bin/pwd"], env), 20000, 4000, "")
    assert_equal(r.exit_code, 127)
    assert_equal(r.stderr, "QA-E160: [Errno 2] No such file or directory: '" + inside + "/missing'\n")

    remove_tree(inside)
    remove_tree(outside)
    remove_tree(tool_dir)
    remove_tree(probe)


def main() raises:
    test_json_round_trip()
    test_profiles()
    var work = make_temp_dir("qumo-sandbox-test-")
    var tramp = _build_fixture(work)
    test_confine_shape(tramp)
    var kind = sandbox_kind(tramp)
    if is_macos():
        assert_equal(kind, "sandbox-exec")
    if kind == "none":
        print("note: no OS sandbox on this machine; confined runs were not exercised")
        try:
            _ = confine(_spec("/bin/true", ["true"], List[String]()), Confinement(CONF_TOOL, List[String](), List[String](), "", 0, 0), tramp)
            assert_true(False, "confine must refuse without an OS sandbox")
        except e:
            assert_equal(String(e), "QA-E163: no OS sandbox is available on this machine")
    else:
        test_confined_runs(tramp, work)
    remove_tree(work)
    print("test_native_sandbox OK")
