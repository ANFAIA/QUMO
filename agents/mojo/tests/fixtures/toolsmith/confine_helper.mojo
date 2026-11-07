"""Test trampoline: a binary that dispatches `__confine` like qumo-agent does
(DESIGN.md 2.1.8), so the toolsmith tests can confine builds and tool runs before
the real main.mojo exists. Built on demand into build/test-fixtures/."""

from std.sys import argv, exit

from qumo_agent.native.sandbox import confine_main


def main():
    var args = List[String]()
    for a in argv():
        args.append(String(a))
    if len(args) >= 2 and args[1] == "__confine":
        exit(confine_main(args))
    print("usage: confine_helper __confine CONF_JSON PATH ARGV...")
    exit(2)
