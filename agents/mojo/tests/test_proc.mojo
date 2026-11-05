"""ChildProcess: a line round trip through /bin/cat, arguments that are never
shell-parsed, EOF propagation and exit codes."""

from std.testing import assert_equal, assert_true, assert_false

from qumo_agent.proc import ChildProcess


def main() raises:
    var cat = ChildProcess("/bin/cat", List[String]())
    var line = String()
    cat.send_line("hello")
    assert_true(cat.read_line(line))
    assert_equal(line, "hello")
    cat.send_line("{\"jsonrpc\":\"2.0\",\"id\":1} ünïcode")
    assert_true(cat.read_line(line))
    assert_equal(line, "{\"jsonrpc\":\"2.0\",\"id\":1} ünïcode")
    var big = String()
    for _ in range(5000):
        big += "abcdefghij"
    cat.send_line(big)
    assert_true(cat.read_line(line))
    assert_equal(line.byte_length(), 50000)
    cat.close_input()
    assert_false(cat.read_line(line))  # EOF once the child exits
    assert_equal(cat.finish(), 0)

    # (cat streams: a line above the 64 KiB pipe buffer would deadlock a writer that never reads.)
    # Shell metacharacters in arguments stay literal.
    var echo = ChildProcess("/bin/echo", ["$(touch /tmp/qumo-agent-pwned) ; `id` | x", "*"])
    assert_true(echo.read_line(line))
    assert_equal(line, "$(touch /tmp/qumo-agent-pwned) ; `id` | x *")
    assert_equal(echo.finish(), 0)

    # Short, computed strings as program and arguments (Mojo 1.2's Process.run
    # aliases their argv pointers unless proc.spawn moves them to the heap).
    var prog = String("/bin") + "/echo"
    var e2 = ChildProcess(prog, [String("a") + "b", String("c") + "d", "lit"])
    assert_true(e2.read_line(line))
    assert_equal(line, "ab cd lit")
    assert_equal(e2.finish(), 0)

    var f = ChildProcess("/bin/sh", ["-c", "exit 3"])
    assert_equal(f.finish(), 3)
    print("test_proc OK")
