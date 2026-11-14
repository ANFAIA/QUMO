"""Test trampoline for test_tools_repo: a binary that dispatches `__confine` as qumo-agent
does (DESIGN.md 2.1.8), so run_check inside a sandbox can be confined before main.mojo
exists. Built on demand into build/test-fixtures/tools_confine_trampoline."""

from std.sys import argv, exit

from qumo_agent.native.sandbox import confine_main


def main():
    var args = List[String]()
    for a in argv():
        args.append(String(a))
    if len(args) >= 2 and args[1] == "__confine":
        exit(confine_main(args))
    print("usage: tools_confine_trampoline __confine CONF_JSON PATH ARGV...")
    exit(2)
