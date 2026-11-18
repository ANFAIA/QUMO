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
from qumo_agents.config import API_BACKENDS, BACKENDS, ConfigError, Settings
from qumo_agents.tools import alphaxiv


def _task(args) -> str:
    task = sys.stdin.read() if args.task == "-" else args.task
    if args.module:
        m = modules.get(args.module)
        task = f"Module: {m.name} ({m.summary}) Code: {', '.join(m.paths)}.\n\n{task}"
    return task


def _resolved(settings: Settings, agent: str, backend: str | None = None, model: str | None = None) -> tuple[str, str]:
    """(back end, model) an agent would run on; the model is a readable note when unset."""
    b = settings.backend_for(agent, backend)
    try:
        m = settings.model_for(agent, model, b) or "(the CLI's default)"
    except ConfigError:
        m = "(not set: --model)"
    return b, m


def cmd_list(args, settings: Settings) -> int:
    print(f"{'agent':20s} {'back end':12s} {'model':32s} does")
    for spec in registry.AGENTS.values():
        subs = f" -> {', '.join(spec.subagents)}" if spec.subagents else ""
        b, m = _resolved(settings, spec.name, args.backend, args.model)
        print(f"{spec.name:20s} {b:12s} {m:32s} {spec.description}{subs}")
    print(f"\nmodules: {', '.join(modules.MODULES)}")
    print(f"backends: {', '.join(BACKENDS)} (default {settings.default_backend})")
    print("choose per run with --backend / --model, per shell with QUMO_AGENTS_BACKEND / QUMO_AGENTS_MODEL,")
    print('or per agent in agents/agents.json ("backends", and "models" inside each back end)')
    print(f"alphaXiv: {'on' if alphaxiv.available() else 'off (set ALPHAXIV_API_KEY; arXiv fallback)'}")
    return 0


def cmd_run(args, settings: Settings) -> int:
    spec = registry.get(args.agent)
    backend = settings.backend_for(spec.name, args.backend)
    task = _task(args)
    if backend in API_BACKENDS:
        model = settings.model_for(spec.name, args.model, backend)
        if args.dry_run:
            print(json.dumps({"agent": spec.name, "backend": backend, "model": model,
                              "local_tools": [t.name for t in registry.local_tools(spec)],
                              "qumo_mcp": "all" if spec.mcp_tools is None else list(spec.mcp_tools),
                              "alphaxiv": "research" in spec.tool_groups and alphaxiv.available(),
                              "subagents": {s: settings.model_for(s, None, backend, model if backend == "local" else None)
                                            for s in spec.subagents}}, indent=1))
            return 0
        from qumo_agents.graph import run_agent
        print(asyncio.run(run_agent(spec, task, settings, args.model, backend=backend)))
        return 0
    from qumo_agents import backends
    if args.dry_run:
        cmd = (backends.claude_code_command(spec, task, settings, None, args.model) if backend == "claude-code"
               else backends.codex_command(spec, task, settings, registry_root(), args.model))
        print(json.dumps(cmd[:2] + ["<prompt>" if len(c) > 200 else c for c in cmd[2:]], indent=1))
        return 0
    print(backends.run_cli_backend(backend, spec, task, settings, args.model))
    return 0


def _local_reachable(settings: Settings) -> bool:
    """True if the local OpenAI-compatible server answers GET <base_url>/models within 2 s."""
    import urllib.request

    from qumo_agents.config import local_base_url
    url = local_base_url(settings).rstrip("/") + "/models"
    if not url.startswith(("http://", "https://")):
        return False
    try:
        with urllib.request.urlopen(url, timeout=2) as r:  # noqa: S310 (scheme checked above)
            return r.status == 200
    except Exception:
        return False


def cmd_doctor(_args, settings: Settings) -> int:
    """Check what each back end needs; prints names of variables, never their values."""
    import os
    import shutil

    from qumo_agents.tools import qumo_mcp

    or_env = settings.section("openrouter").get("api_key_env", "OPENROUTER_API_KEY")
    checks = [
        ("QUMO MCP server", qumo_mcp.QUMO_BIN.exists(), f"{qumo_mcp.QUMO_BIN} (build it with `make cli`)"),
        (f"openrouter back end ({or_env})", bool(os.environ.get(or_env)), "export the key to use the default back end"),
        ("local back end (OpenAI-compatible server)", _local_reachable(settings),
         "optional; start Ollama / LM Studio / llama.cpp / vLLM, set QUMO_AGENTS_LOCAL_URL if it is not localhost:11434"),
        ("claude-code back end", shutil.which(settings.section("claude_code").get("command", "claude")) is not None,
         "install Claude Code: https://claude.com/claude-code"),
        ("codex back end", shutil.which(settings.section("codex").get("command", "codex")) is not None,
         "install the Codex CLI"),
        (f"alphaXiv papers ({alphaxiv.KEY_ENV})", alphaxiv.available(), "optional; without it agents use the arXiv API"),
        ("GitHub token (GITHUB_TOKEN)", bool(os.environ.get("GITHUB_TOKEN")), "optional; raises the release-notes rate limit"),
    ]
    for name, ok, hint in checks:
        print(f"[{'ok' if ok else '--'}] {name}" + ("" if ok else f": {hint}"))
    usable = [b for b, ok in (("openrouter", checks[1][1]), ("local", checks[2][1]),
                              ("claude-code", checks[3][1]), ("codex", checks[4][1])) if ok]
    print(f"\nusable back ends: {', '.join(usable) or 'none'}")
    return 0 if checks[0][1] and usable else 1


def cmd_export_prompts(_args, _settings: Settings) -> int:
    from qumo_agents import export

    changed = export.write()
    for path in changed:
        print(f"wrote {path.relative_to(export.AGENTS_DIR.parent)}")
    print(f"{len(changed)} file(s) changed in {export.PROMPTS_DIR.relative_to(export.AGENTS_DIR.parent)}")
    return 0


def cmd_reports(args, _settings: Settings) -> int:
    from qumo_agents.config import REPORTS_DIR

    files = sorted(REPORTS_DIR.glob("*.md"), reverse=True)
    if args.show:
        match = [f for f in files if args.show in f.name]
        if not match:
            print(f"no report matches {args.show!r}", file=sys.stderr)
            return 1
        print(match[0].read_text())
        return 0
    for f in files[: args.limit]:
        print(f"{f.stat().st_size:>7}  {f.relative_to(REPORTS_DIR.parent.parent)}")
    return 0


def registry_root():
    from qumo_agents.config import REPO_ROOT
    return REPO_ROOT


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="qumo_agents", description="QUMO agent layer (deepagents + MCP)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list", help="list agents with the back end and model each would run on")
    ls.add_argument("--backend", choices=BACKENDS, help="show the models for this back end")
    ls.add_argument("--model", help="show the list as if --model were given")
    r = sub.add_parser("run", help="run one agent on a task")
    r.add_argument("agent", choices=list(registry.AGENTS))
    r.add_argument("task", help="the task in plain words; use - to read it from stdin")
    r.add_argument("--module", choices=list(modules.MODULES))
    r.add_argument("--backend", choices=BACKENDS,
                   help="harness: openrouter (API), local (your own OpenAI-compatible server), claude-code or codex (CLIs)")
    r.add_argument("--model", help="model for this run: an OpenRouter id, a local model name, or the CLI's own model name")
    r.add_argument("--dry-run", action="store_true", help="show what would run, call nothing")
    sub.add_parser("doctor", help="check keys, the MCP server and the back-end CLIs")
    rp = sub.add_parser("reports", help="list saved reports, or print one")
    rp.add_argument("--show", metavar="TEXT", help="print the newest report whose name contains TEXT")
    rp.add_argument("--limit", type=int, default=20)
    sub.add_parser("export-prompts", help="write prompts, specs and MCP tool guides to agents/prompts/ (read by the Mojo agents)")
    args = ap.parse_args(argv)
    settings = Settings.load()
    try:
        return {"list": cmd_list, "run": cmd_run, "doctor": cmd_doctor, "reports": cmd_reports,
                "export-prompts": cmd_export_prompts}[args.cmd](args, settings)
    except (ValueError, RuntimeError, ConfigError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
