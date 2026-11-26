#!/usr/bin/env python3
"""Write agents/mojo/qumo_agent/core/guard_rules.mojo from guard_rules.json.

The Mojo harness reads the lists of the specification through that module (the low layers,
native/ and net/, have no JSON reader at hand, and a list read at start-up could be changed
by whoever can write the file next to the binary). Run this after every change of `lists`:

    python3 agents/spec/gen_mojo_rules.py          # rewrite the module
    python3 agents/spec/gen_mojo_rules.py --check  # exit 1 when it is stale

A stale module fails agents/tests/test_guard_spec.py (text comparison) and
agents/mojo/tests/test_guard_spec.mojo (value by value against the JSON). Standard library only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = HERE / "guard_rules.json"
TARGET = HERE.parent / "mojo" / "qumo_agent" / "core" / "guard_rules.mojo"


def _s(text: str) -> str:
    """A Mojo string literal (ASCII only: the lists hold nothing else)."""
    assert text.isascii() and text.isprintable(), text
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _list_fn(name: str, items: list[str], doc: str) -> str:
    lines = [f"def {name}() -> List[String]:", f'    """{doc}"""', "    var out = List[String]()"]
    lines += [f"    out.append({_s(x)})" for x in items]
    lines += ["    return out^", "", ""]
    return "\n".join(lines)


def render(spec: dict) -> str:
    L = spec["lists"]
    repo, env, research = L["repo"], L["env"], L["research"]
    out = ['"""GENERATED from agents/spec/guard_rules.json by agents/spec/gen_mojo_rules.py. Do not edit:',
           "change the JSON and run the script. The lists of the security rules both agent harnesses",
           'share (agents/spec/README.md)."""', "", ""]
    out.append(f"comptime GUARD_RULES_SCHEMA = {_s(spec['schema'])}")
    out.append(f"comptime MAX_REDIRECTS = {int(research['max_redirects'])}")
    out.append(f"comptime RESEARCH_SCHEME = {_s(research['scheme'])}")
    out.append(f"comptime RESEARCH_PORT = {int(research['port'])}")
    out.append(f"comptime ARG_OTHER_CHARS = {_s(repo['arg_chars']['others'])}")
    out.append(f"comptime GIT_FLAGS_TEXT = {_s(' '.join(sorted(repo['git_flags'])))}")
    out.append(f"comptime QUMO_SUBCOMMANDS_TEXT = {_s(' '.join(sorted(repo['qumo_subcommands'])))}")
    out.append(f"comptime UNTRUSTED_MARKER = {_s(L['untrusted']['marker'])}")
    out.append(f"comptime MAX_ANSWER_CHARS = {int(L['untrusted']['max_answer_chars'])}")
    out += ["", ""]
    body = [
        _list_fn("skip_dirs", repo["skip_dirs"], "Directories no agent tool reads (compared without regard to ASCII case)."),
        _list_fn("secret_name_prefixes", repo["secret_names"]["prefix"], "A file name starting with one of these is secret-looking."),
        _list_fn("secret_name_contains", repo["secret_names"]["contains"], "A file name holding one of these is secret-looking."),
        _list_fn("secret_name_suffixes", repo["secret_names"]["suffix"], "A file name ending in one of these is secret-looking."),
        _list_fn("arg_ranges", repo["arg_chars"]["ranges"], "Character ranges of a run_check argument, each as `a-z`."),
        _list_fn("git_flags", repo["git_flags"], "Options run_check accepts after `git diff` / `git status --short`."),
        _list_fn("qumo_subcommands", repo["qumo_subcommands"], "`build/qumo` subcommands run_check accepts."),
        _list_fn("qumo_output_options", repo["qumo_output_options"], "`build/qumo` options that write a file."),
        _list_fn("qumo_options", repo["qumo_options"], "Other `build/qumo` options run_check accepts."),
        _list_fn("secret_env_substrings", env["secret_name_substrings"], "An environment name holding one of these (any case) is a secret."),
        _list_fn("check_env_extra_substrings", env["check_env_extra_substrings"], "Further substrings the Python check environment drops."),
        _list_fn("git_injected_exact", env["git_injected"]["exact"], "Git configuration passed through the environment: whole names."),
        _list_fn("git_injected_numbered", env["git_injected"]["numbered"], "Git configuration passed through the environment: prefixes of numbered names."),
        _list_fn("minimal_child_env_names", env["minimal_child_env"], "The only variables a child of the Mojo harness inherits."),
        _list_fn("api_hosts", research["api_hosts"], "Hosts the research API tools may call."),
        _list_fn("web_groups", L["delegation"]["web_groups"], "Tool groups that fetch from the web."),
        _list_fn("exec_groups", L["delegation"]["exec_groups"], "Tool groups that run code."),
    ]
    lines = ["def run_check_allowed() -> List[List[String]]:", '    """The argv prefixes run_check accepts, in order."""',
             "    var out = List[List[String]]()"]
    for _, prefix in repo["run_check_allowed"]:
        lines.append("    out.append([" + ", ".join(_s(x) for x in prefix) + "])")
    lines += ["    return out^", ""]
    return "\n".join(out) + "".join(body) + "\n".join(lines)


def main(argv: list[str]) -> int:
    text = render(json.loads(SPEC.read_text(encoding="utf-8")))
    if "--check" in argv:
        if not TARGET.is_file() or TARGET.read_text() != text:
            print(f"{TARGET} is stale: run python3 agents/spec/gen_mojo_rules.py", file=sys.stderr)
            return 1
        return 0
    TARGET.write_text(text)
    print(f"wrote {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
