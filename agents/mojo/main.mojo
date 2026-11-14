"""The qumo-agent launcher: QUMO agents as one native executable (no Python at run time).

Build: `make agents-mojo` (= mojo build -I . -I agents/mojo -o build/qumo-agent agents/mojo/main.mojo).
The command line is documented in qumo_agent/app/commands.mojo and agents/mojo/README.md.
"""

from std.sys import argv, exit

from qumo_agent.app.commands import main_entry


def main():
    var args = List[String]()
    for a in argv():
        args.append(String(a))
    exit(main_entry(args))
