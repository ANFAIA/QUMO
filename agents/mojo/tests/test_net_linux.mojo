"""Compile check of net/* and llm/* for Linux (DESIGN.md 9.3, N).

Cross-compiles to object files for x86_64 and aarch64 Linux the programs that use
every net/llm module: tests/test_net_tls_local.mojo (OpenSSL, TLS, sockets, HTTP
streaming), tests/test_llm_client.mojo (LLM client, HTTP, redirects, fixture helper)
and tests/support/http_fixture_server.mojo (listener). The Linux `comptime` branches
(socket constants, `timeval` and `addrinfo` layouts, library names) are type-checked
and lowered on a macOS host. Nothing is linked or run.
"""

from std.os.path import exists
from std.testing import assert_equal, assert_true

from qumo_agent.native.spawn import SpawnSpec, run_capture, environ_list, STDIO_NULL, STDIO_PIPE
from qumo_agent.native.fs import make_temp_dir, remove_tree


def main() raises:
    var mojo = String(".venv/bin/mojo")
    assert_true(exists(mojo), "run from the repository root")
    var out_dir = make_temp_dir("qumo-net-linux-")
    var targets: List[String] = ["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"]
    var cpus: List[String] = ["x86-64", "generic"]
    var sources: List[String] = [
        "agents/mojo/tests/test_net_tls_local.mojo",
        "agents/mojo/tests/test_llm_client.mojo",
        "agents/mojo/tests/support/http_fixture_server.mojo",
    ]
    for i in range(len(targets)):
        for k in range(len(sources)):
            var obj = out_dir + "/" + String(k) + "_" + targets[i] + ".o"
            var argv: List[String] = [
                mojo, "build", "--target-triple", targets[i], "--target-cpu", cpus[i],
                "--emit", "object", "-I", ".", "-I", "agents/mojo", sources[k], "-o", obj,
            ]
            var r = run_capture(SpawnSpec(mojo, argv^, environ_list(), "", STDIO_NULL, STDIO_PIPE, STDIO_PIPE, False),
                                600000, 1 << 16, "")
            assert_equal(r.exit_code, 0, "cross-compile " + sources[k] + " for " + targets[i] + ": " + r.stderr)
            assert_true(exists(obj))
    remove_tree(out_dir)
    print("test_net_linux OK")
