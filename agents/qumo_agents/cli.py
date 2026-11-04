"""Command line for the QUMO agent layer.

    agents/.venv/bin/python -m qumo_agents list
    agents/.venv/bin/python -m qumo_agents run research-watch "What is new for routing?" --module compiler
    agents/.venv/bin/python -m qumo_agents run qumo-operator "Run a 3-qubit GHZ and check it" --backend claude-code
    agents/.venv/bin/python -m qumo_agents run code-review "Review the QASM parser" --module ir --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from qumo_agents import modules, registry
from qumo_agents.config import BACKENDS, Settings
from qumo_agents.tools import alphaxiv


def _task(args) -> str:
    task = args.task
    if args.module:
        m = modules.get(args.module)
        task = f"Module: {m.name} ({m.summary}) Code: {', '.join(m.paths)}.\n\n{task}"
    return task


def cmd_list(_args, settings: Settings) -> int:
    for spec in registry.AGENTS.values():
        subs = f" -> {', '.join(spec.subagents)}" if spec.subagents else ""
        print(f"{spec.name:20s} {settings.model_for(spec.name):32s} {spec.description}{subs}")
    print(f"\nmodules: {', '.join(modules.MODULES)}")
    print(f"backends: {', '.join(BACKENDS)} (default {settings.default_backend})")
    print(f"alphaXiv: {'on' if alphaxiv.available() else 'off (set ALPHAXIV_API_KEY; arXiv fallback)'}")
    return 0


def cmd_run(args, settings: Settings) -> int:
    spec = registry.get(args.agent)
    backend = args.backend or settings.default_backend
    task = _task(args)
    if backend == "openrouter":
        if args.dry_run:
            print(json.dumps({"agent": spec.name, "model": settings.model_for(spec.name, args.model),
                              "local_tools": [t.name for t in registry.local_tools(spec)],
                              "qumo_mcp": "all" if spec.mcp_tools is None else list(spec.mcp_tools),
                              "alphaxiv": "research" in spec.tool_groups and alphaxiv.available(),
                              "subagents": list(spec.subagents)}, indent=1))
            return 0
        from qumo_agents.graph import run_agent
        print(asyncio.run(run_agent(spec, task, settings, args.model)))
        return 0
    from qumo_agents import backends
    if args.dry_run:
        cmd = (backends.claude_code_command(spec, task, settings, None, args.model) if backend == "claude-code"
               else backends.codex_command(spec, task, settings, registry_root(), args.model))
        print(json.dumps(cmd[:2] + ["<prompt>" if len(c) > 200 else c for c in cmd[2:]], indent=1))
        return 0
    print(backends.run_cli_backend(backend, spec, task, settings, args.model))
    return 0


def registry_root():
    from qumo_agents.config import REPO_ROOT
    return REPO_ROOT


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="qumo_agents", description="QUMO agent layer (deepagents + MCP)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list agents, models, modules and back ends")
    r = sub.add_parser("run", help="run one agent on a task")
    r.add_argument("agent", choices=list(registry.AGENTS))
    r.add_argument("task")
    r.add_argument("--module", choices=list(modules.MODULES))
    r.add_argument("--backend", choices=BACKENDS)
    r.add_argument("--model", help="model id (OpenRouter id, or the CLI's own model name)")
    r.add_argument("--dry-run", action="store_true", help="show what would run, call nothing")
    args = ap.parse_args(argv)
    settings = Settings.load()
    try:
        return {"list": cmd_list, "run": cmd_run}[args.cmd](args, settings)
    except (ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
