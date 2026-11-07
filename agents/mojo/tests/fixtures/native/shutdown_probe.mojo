"""Test-only program for test_native_shutdown (built on demand into a temp directory).

`shutdown_probe parent FILE`: installs the shutdown handlers and starts `shutdown_probe child FILE`
in its own process group. `shutdown_probe child FILE`: installs the handlers, starts
`/bin/sleep 120` in its own process group and writes that pid to FILE. Both then wait forever,
so the test can send them SIGINT or SIGTERM and look for orphans.
"""

from std.sys import argv

from qumo_agent.native.clock import sleep_ms
from qumo_agent.native.fs import write_text
from qumo_agent.native.shutdown import install_shutdown_handlers
from qumo_agent.native.spawn import SpawnSpec, STDIO_NULL, spawn, minimal_child_env


def main() raises:
    var args = List[String]()
    for a in argv():
        args.append(String(a))
    install_shutdown_handlers()
    var mode = args[1]
    var file = args[2]
    var spec: SpawnSpec
    if mode == "parent":
        spec = SpawnSpec(args[0], [args[0], "child", file], minimal_child_env(), String(), STDIO_NULL, STDIO_NULL, STDIO_NULL, True)
    else:
        spec = SpawnSpec("/bin/sleep", ["sleep", "120"], minimal_child_env(), String(), STDIO_NULL, STDIO_NULL, STDIO_NULL, True)
    var child = spawn(spec)
    if mode == "child":
        write_text(file, String(child.pid))
    while child.pid > 0:   # uses `child` in the loop: it must stay alive (its destructor kills the group)
        sleep_ms(50)
