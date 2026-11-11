"""The QUMO command-line interface (`qumo`): the executable.

Build: `mojo build -I . -o build/qumo cli/main.mojo` (`make cli`). (The entry
file must not be called `qumo.mojo`: a main module named `qumo` shadows the
`qumo` package and every `from qumo...` import fails.)

    qumo <command> [args] [options] [--json] [--pretty]

All of the interface lives in `cli/app.mojo` (`run_cli`), which tests call in
process: commands, their options and the generated help are declared in its
`command_table()`. Exit code 0 = success, 1 = the operation failed (diagnostic
printed: human on stderr, JSON on stdout), 2 = usage error.
"""

from std.sys import argv, exit, stderr

from cli.app import run_cli


def main():
    var raw = argv()
    var args = List[String]()
    for i in range(1, len(raw)):
        args.append(String(raw[i]))
    var r = run_cli(args)
    if r.out.byte_length() > 0:
        print(r.out, end="")
    if r.err.byte_length() > 0:
        print(r.err, end="", file=stderr)
    exit(r.code)
