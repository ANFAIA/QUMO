"""Compile check of the Linux branches of native/libc.mojo (DESIGN.md 9.3, F).

Cross-compiles tests/test_native_libc_std.mojo (which calls every wrapper) to object
files for x86_64 and aarch64 Linux with `.venv/bin/mojo build --emit object`, so the
`comptime` Linux constants and the Linux-only calls (`unshare`, `prctl`, `syscall`)
are type-checked and lowered even on a macOS host. Nothing is linked or run.
The other native modules are compile-checked the same way through their tests and
the sandbox test program (its `__confine` path is the Linux Landlock code).
On a Linux host the native tests cover the same code; this test still runs.
"""

from std.testing import assert_equal, assert_true
from std.os import remove, rmdir, listdir
from std.os.path import exists
from std.tempfile import mkdtemp

from qumo_agent.native.libc import *


def _environ() raises -> List[String]:
    """This process's environment (the compiler needs PATH, HOME, ...)."""
    var out = List[String]()
    if is_macos():
        var envp = load_i64(c_ns_get_environ())
        var i = 0
        while True:
            var s = load_i64(envp + 8 * i)
            if s == 0:
                break
            out.append(c_string_at(s))
            i += 1
    else:
        var data = open("/proc/self/environ", "r").read()
        for item in data.split("\x00"):
            if item.byte_length() > 0:
                out.append(String(item))
    return out^


def _run(argv: List[String]) raises -> Int:
    """posix_spawn argv[0] with inherited stdio; returns the exit code."""
    var path = CStr(argv[0])
    var argv_c = CStrArray(argv)
    var envp = CStrArray(_environ())
    var pid_buf = List[UInt8](length=8, fill=0)
    var rc = c_posix_spawn(buf_addr(pid_buf), path.addr(), 0, 0, argv_c.addr(), envp.addr())
    assert_equal(rc, 0, "posix_spawn " + argv[0] + ": " + strerror_text(rc))
    var pid = load_i32(buf_addr(pid_buf))
    var st = List[UInt8](length=8, fill=0)
    while c_waitpid(pid, buf_addr(st), 0) < 0:
        assert_equal(errno(), EINTR)
    var s = load_i32(buf_addr(st))
    _ = len(pid_buf)
    _ = len(st)
    path.keep()
    argv_c.keep()
    envp.keep()
    if (s & 0x7F) == 0:
        return (s >> 8) & 0xFF
    return -(s & 0x7F)


def main() raises:
    var mojo = String(".venv/bin/mojo")
    assert_true(exists(mojo), "run from the repository root")
    var out_dir = mkdtemp()
    var targets: List[String] = ["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"]
    var cpus: List[String] = ["x86-64", "generic"]
    var sources: List[String] = [
        "agents/mojo/tests/test_native_libc_std.mojo",
        "agents/mojo/tests/fixtures/native/trampoline.mojo",
        "agents/mojo/tests/test_native_sandbox.mojo",
        "agents/mojo/tests/test_native_spawn.mojo",
        "agents/mojo/tests/test_native_fs.mojo",
        "agents/mojo/tests/test_native_clock.mojo",
        "agents/mojo/tests/test_native_rand.mojo",
        "agents/mojo/tests/test_native_regex.mojo",
        "agents/mojo/tests/test_native_fnmatch.mojo",
        "agents/mojo/tests/test_util_sha256.mojo",
        "agents/mojo/tests/test_util_redact.mojo",
    ]
    var both_targets = 3   # the first three sources are built for both targets
    for k in range(len(sources)):
        for i in range(len(targets)):
            if i > 0 and k >= both_targets:
                continue
            var obj = out_dir + "/obj_" + String(k) + "_" + targets[i] + ".o"
            var argv: List[String] = [
                mojo, "build", "--target-triple", targets[i], "--target-cpu", cpus[i],
                "--emit", "object", "-I", ".", "-I", "agents/mojo", sources[k], "-o", obj,
            ]
            assert_equal(_run(argv), 0, "cross-compile " + sources[k] + " for " + targets[i])
            assert_true(exists(obj))
    for name in listdir(out_dir):
        remove(out_dir + "/" + name)
    rmdir(out_dir)
    print("test_native_libc_linux OK")
