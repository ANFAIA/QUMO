"""Test-only `qumo-agent` for the app tests (built on demand into
build/test-fixtures/qumo-agent-app): the same entry point main.mojo will use, so the tests
do not depend on Integration's main.mojo."""

from std.sys import argv, exit

from qumo_agent.app.commands import main_entry


def main():
    var args = List[String]()
    for a in argv():
        args.append(String(a))
    exit(main_entry(args))
