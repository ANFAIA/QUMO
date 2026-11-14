"""SIGINT and SIGTERM end every child process group (native/shutdown.mojo).

Regression test for orphaned children: the program used to install no handler, so Ctrl-C
or `kill` left a `qumo mcp` computation or a subagent (each in its own process group)
running with PPID 1. The probe program starts a child that starts `sleep`; after the signal
the probe exits with 128 + signal and the `sleep` must be gone too (the nested handler ran).
Run from the repository root."""

from std.testing import assert_equal, assert_true, assert_false

from qumo_agent.native.clock import monotonic_ms, sleep_ms
from qumo_agent.native.fs import exists, is_file, make_temp_dir, read_text_lossy, realpath_strict, remove_tree
from qumo_agent.native.libc import c_kill
from qumo_agent.native.shutdown import (
    SIGINT, SIGTERM, live_child_count, register_child, unregister_child,
)
from qumo_agent.native.spawn import (
    SpawnSpec, STDIO_NULL, STDIO_PIPE, run_capture, spawn, minimal_child_env, environ_list,
)


def _build(dir: String) raises -> String:
    var out = dir + "/shutdown_probe"
    var argv: List[String] = [
        ".venv/bin/mojo", "build", "-I", ".", "-I", "agents/mojo", "-o", out,
        "agents/mojo/tests/fixtures/native/shutdown_probe.mojo",
    ]
    var spec = SpawnSpec(".venv/bin/mojo", argv^, environ_list(), String(), STDIO_NULL, STDIO_PIPE, STDIO_PIPE, True)
    var r = run_capture(spec, 600000, 20000, "")
    assert_equal(r.exit_code, 0, "fixture build failed: " + r.stdout + r.stderr)
    return out^


def _alive(pid: Int) -> Bool:
    return c_kill(pid, 0) == 0


def _wait_until_gone(pid: Int, timeout_ms: Int) -> Bool:
    var deadline = monotonic_ms() + timeout_ms
    while monotonic_ms() < deadline:
        if not _alive(pid):
            return True
        sleep_ms(20)
    return not _alive(pid)


def _run_signal(probe: String, dir: String, sig: Int, expect_exit: Int) raises:
    var file = dir + "/sleep-" + String(sig) + ".pid"
    var spec = SpawnSpec(probe, [probe, "parent", file], minimal_child_env(), String(), STDIO_NULL, STDIO_NULL, STDIO_NULL, True)
    var parent = spawn(spec)
    var deadline = monotonic_ms() + 30000
    while not is_file(file) and monotonic_ms() < deadline:
        sleep_ms(20)
    assert_true(is_file(file), "the probe did not start its grandchild")
    sleep_ms(100)
    var sleep_pid = Int(read_text_lossy(file).strip())
    assert_true(_alive(sleep_pid))
    _ = c_kill(parent.pid, sig)
    var code = parent.wait()
    assert_equal(code, expect_exit)
    assert_true(_wait_until_gone(sleep_pid, 5000), "the grandchild survived the signal")


def test_registry() raises:
    var before = live_child_count()
    register_child(999999, True)
    register_child(999998, False)
    assert_equal(live_child_count(), before + 2)
    unregister_child(999999, True)
    unregister_child(999998, False)
    assert_equal(live_child_count(), before)
    # A reaped child is gone from the registry.
    var c = spawn(SpawnSpec("/usr/bin/true", ["true"], minimal_child_env(), String(), STDIO_NULL, STDIO_NULL, STDIO_NULL, True))
    assert_equal(live_child_count(), before + 1)
    _ = c.wait()
    assert_equal(live_child_count(), before)


def main() raises:
    test_registry()
    var dir = realpath_strict(make_temp_dir("qumo-shutdown-"))
    var probe = _build(dir)
    _run_signal(probe, dir, SIGTERM, 143)
    _run_signal(probe, dir, SIGINT, 130)
    remove_tree(dir)
    print("test_native_shutdown OK")
