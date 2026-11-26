"""The agent harness's copies of the coded network modules are generated from qumo/net.

`qumo/net` is the one implementation of the HTTP/1.1 + TLS stack. The agent harness
(agents/mojo) reports network failures with its own codes (QA-E104, QA-E105), which its retry
rules, exit codes and docs/ERRORS.md depend on, so the modules that raise coded errors are
mirrored into `agents/mojo/qumo_agent/net/` by a fixed text substitution (imports, the two
codes, the DESIGN.md path) instead of being imported. This test fails when a mirror differs
from what qumo/net generates;

    QUMO_UPDATE_NET=1 .venv/bin/mojo run -I . tests/test_net_mirror.mojo

rewrites the mirrors. `libc` and `fdutil` raise no network code and are re-exported, which
the test also checks.
"""

from std.os import getenv
from std.os.path import exists
from std.testing import assert_true

comptime SOURCE_DIR = "qumo/net"
comptime MIRROR_DIR = "agents/mojo/qumo_agent/net"
comptime UPDATE = "QUMO_UPDATE_NET=1 .venv/bin/mojo run -I . tests/test_net_mirror.mojo"


def _read(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def mirrored_modules() -> List[String]:
    return ["url", "ossl", "tls", "socket", "http", "guard", "sse"]


def mirror_text(name: String, source: String) -> String:
    """The agent copy of qumo/net/<name>.mojo."""
    var t = source
    t = t.replace("qumo.net.errors", "qumo_agent.core.errors")
    t = t.replace("qumo.net.libc", "qumo_agent.native.libc")
    t = t.replace("qumo.net.", "qumo_agent.net.")
    t = t.replace("QM-E959", "QA-E104")
    t = t.replace("QM-E951", "QA-E105")
    t = t.replace("agents/mojo/DESIGN.md", "DESIGN.md")
    return ("# GENERATED from " + SOURCE_DIR + "/" + name + ".mojo by tests/test_net_mirror.mojo. Do not edit:\n"
            + "# change that file, then run " + UPDATE + "\n" + t)


def test_mirrors_in_sync() raises:
    var update = getenv("QUMO_UPDATE_NET") == "1"
    for name in mirrored_modules():
        var want = mirror_text(name, _read(SOURCE_DIR + "/" + name + ".mojo"))
        var path = MIRROR_DIR + "/" + name + ".mojo"
        if update:
            with open(path, "w") as f:
                f.write(want)
            print(path + " regenerated from " + SOURCE_DIR + "/" + name + ".mojo")
        assert_true(exists(path), path + " is missing: run " + UPDATE)
        assert_true(_read(path) == want, path + " differs from " + SOURCE_DIR + "/" + name + ".mojo: edit the qumo/net file, then run " + UPDATE)


def test_sources_carry_no_agent_names() raises:
    """A source that named the agent package or its codes would not survive the substitution."""
    for name in mirrored_modules():
        var src = _read(SOURCE_DIR + "/" + name + ".mojo")
        assert_true(src.find("qumo_agent.") < 0, name + ": imports the agent package")
        assert_true(src.find("QA-E") < 0, name + ": names an agent error code")
        assert_true(src.find("\"QM-E") < 0, name + ": a literal code instead of qumo.net.errors")


def test_shared_modules_are_reexports() raises:
    """Module libc and module fdutil exist once: the agent paths only re-export them."""
    var libc = _read("agents/mojo/qumo_agent/native/libc.mojo")
    assert_true(libc.find("from qumo.net.libc import *") >= 0)
    assert_true(libc.find("external_call[") < 0, "a libc binding was added to the agent re-export; add it to qumo/net/libc.mojo")
    var fd = _read(MIRROR_DIR + "/fdutil.mojo")
    assert_true(fd.find("from qumo.net.fdutil import *") >= 0)
    assert_true(fd.find("def ") < 0, "fdutil.mojo of the agent must stay a re-export")
    assert_true(not exists(MIRROR_DIR + "/libc.mojo") and not exists(MIRROR_DIR + "/errors.mojo"))


def main() raises:
    test_mirrors_in_sync()
    test_sources_carry_no_agent_names()
    test_shared_modules_are_reexports()
    print("test_net_mirror: OK")
