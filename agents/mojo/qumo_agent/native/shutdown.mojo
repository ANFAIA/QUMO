"""SIGINT, SIGTERM and SIGHUP end the whole process tree (no orphaned children).

Every child is started in its own process group (so a timeout can kill its
grandchildren), which means the terminal's Ctrl-C never reaches it, and a plain
`kill` of this process skips `Child.__deinit__`. Without a handler a `qumo mcp`
computation, a `qumo-agent subagent` (which keeps making paid model calls) or a
`mojo build` would run on with PPID 1.

`spawn` records each child in a registry (`register_child`, `unregister_child` when it
is reaped). `install_shutdown_handlers()` installs `_on_shutdown_signal`, which sends
SIGTERM to every registered group (a child `qumo-agent` runs the same handler and ends
its own children), waits up to `GRACE_MS` reaping them, sends SIGKILL to what is left
and calls `_exit(128 + signal)`.

A handler may only use async-signal-safe calls and cannot reach Mojo globals, so the
registry is a block of `SLOTS` 64-bit slots from `calloc` whose address is kept (as
decimal text, set once before the handler is installed) in the environment variable
`QUMO_AGENT_CHILD_REGISTRY`. A slot holds `pid` for a child leading its own group, `-pid`
for a plain child and 0 when free. The extra slot after the last is `waitpid` scratch.
"""

from std.os import getenv, setenv
from std.ffi import external_call, c_int

from qumo_agent.native.libc import (
    SIGKILL,
    SIGPIPE,
    SIGTERM,
    c_calloc,
    c_exit_now,
    c_getenv_addr,
    c_kill,
    c_strtoll,
    c_usleep,
    c_waitpid,
    load_i64,
    store_i64,
    WNOHANG,
)

comptime REGISTRY_ENV = "QUMO_AGENT_CHILD_REGISTRY"
comptime SLOTS = 512
comptime GRACE_MS = 2000
comptime SIGHUP = 1
comptime SIGINT = 2


def _registry_addr() -> Int:
    var p = c_getenv_addr(Int(REGISTRY_ENV.unsafe_ptr()))   # a literal: no allocation (signal handler)
    if p == 0:
        return 0
    return c_strtoll(p)


def _ensure_registry() -> Int:
    var a = _registry_addr()
    if a != 0:
        return a
    a = c_calloc(SLOTS + 1, 8)
    if a == 0:
        return 0
    _ = setenv(REGISTRY_ENV, String(a))
    return a


def register_child(pid: Int, group: Bool):
    """Records a child so a shutdown signal reaches it. Silent when the registry is full."""
    if pid <= 0:
        return
    var base = _ensure_registry()
    if base == 0:
        return
    var v = pid if group else -pid
    for i in range(SLOTS):
        if load_i64(base + 8 * i) == 0:
            store_i64(base + 8 * i, v)
            return


def unregister_child(pid: Int, group: Bool):
    var base = _registry_addr()
    if base == 0 or pid <= 0:
        return
    var v = pid if group else -pid
    for i in range(SLOTS):
        if load_i64(base + 8 * i) == v:
            store_i64(base + 8 * i, 0)
            return


def live_child_count() -> Int:
    """Registered children (for tests)."""
    var base = _registry_addr()
    if base == 0:
        return 0
    var n = 0
    for i in range(SLOTS):
        if load_i64(base + 8 * i) != 0:
            n += 1
    return n


def _on_shutdown_signal(sig: c_int) abi("C"):
    # Async-signal-safe calls only: no allocation, no String, no raising.
    var base = _registry_addr()
    if base != 0:
        for i in range(SLOTS):
            var v = load_i64(base + 8 * i)
            if v != 0:
                _ = c_kill(-v, SIGTERM)
        var waited = 0
        while waited < GRACE_MS:
            var alive = 0
            for i in range(SLOTS):
                var v = load_i64(base + 8 * i)
                if v == 0:
                    continue
                var pid = v if v > 0 else -v
                var r = c_waitpid(pid, base + 8 * SLOTS, WNOHANG)
                if r == 0:
                    alive += 1
                else:
                    store_i64(base + 8 * i, 0)
            if alive == 0:
                break
            _ = c_usleep(10000)
            waited += 10
        for i in range(SLOTS):
            var v = load_i64(base + 8 * i)
            if v != 0:
                _ = c_kill(-v, SIGKILL)
    c_exit_now(128 + Int(sig))


def _noop_signal(sig: c_int) abi("C"):
    pass


def ignore_sigpipe_signal():
    """A write to a closed pipe or socket returns EPIPE instead of killing the process.
    A no-op handler stands in for SIG_IGN (a handler is reset by exec, SIG_IGN is not)."""
    _ = external_call["signal", Int](c_int(SIGPIPE), _noop_signal)


def install_shutdown_handlers():
    """Ctrl-C, `kill` and a closed terminal end the registered children, then this
    process (exit code 128 + signal)."""
    if _ensure_registry() == 0:
        return
    _ = external_call["signal", Int](c_int(SIGINT), _on_shutdown_signal)
    _ = external_call["signal", Int](c_int(SIGTERM), _on_shutdown_signal)
    _ = external_call["signal", Int](c_int(SIGHUP), _on_shutdown_signal)
