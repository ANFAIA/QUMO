"""Helpers of the live checks (DESIGN.md 8.3): `make agents-mojo-live`.

A live test runs only with QUMO_LIVE=1 and the keys it needs; otherwise it prints
`test_live_<x> SKIP: <reason>` and exits normally, so `make test-agents-mojo` (which skips
test_live_*) and a plain `mojo run` of a live file never need a network or a key. Live tests
spend real model tokens: they run the built `build/qumo-agent` (make agents-mojo) with the
developer's own environment and never print a key.

Import with `-I agents/mojo/tests/support`.
"""

from std.os import getenv
from std.os.path import exists

from qumo_agent.native.spawn import CaptureResult, SpawnSpec, run_capture, environ_list, STDIO_PIPE

comptime BIN = "build/qumo-agent"


def skip_reason(test: String, keys: List[String]) -> String:
    """Why `test` cannot run here ("" when it can): QUMO_LIVE is not 1, a key is not set,
    or build/qumo-agent / build/qumo is missing."""
    if getenv("QUMO_LIVE", "") != "1":
        return "QUMO_LIVE=1 is not set"
    for k in keys:
        if getenv(k, "").byte_length() == 0:
            return k + " is not set"
    if not exists(BIN):
        return BIN + " is missing: run `make agents-mojo`"
    if not exists("build/qumo"):
        return "build/qumo is missing: run `make cli`"
    return String()


def report_skip(test: String, reason: String):
    print(test + " SKIP: " + reason)


def run_live(args: List[String], stdin_data: String, timeout_ms: Int) raises -> CaptureResult:
    """Runs build/qumo-agent with this process's full environment (the model key included,
    on purpose) from the repository root."""
    var argv = List[String]()
    argv.append(BIN)
    for a in args:
        argv.append(a)
    return run_capture(SpawnSpec(BIN, argv^, environ_list(), "", STDIO_PIPE, STDIO_PIPE, STDIO_PIPE, True),
                       timeout_ms, 8 << 20, stdin_data)


def fail_text(r: CaptureResult) -> String:
    """A short failure description: exit code and the tail of stderr (the binary redacts keys)."""
    var full = r.stderr.copy()
    var tail = full.copy()
    if full.byte_length() > 1500:
        tail = String(full[byte = full.byte_length() - 1500 :])
    return "exit " + String(r.exit_code) + (" (timed out)" if r.timed_out else "") + ": " + tail
