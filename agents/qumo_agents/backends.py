"""Back ends that run an agent spec.

openrouter   deepagents graph on an OpenRouter model (default; see graph.py).
claude-code  the Claude Code CLI in print mode, with the agent prompt appended to its system
             prompt and QUMO's MCP server attached.
codex        the Codex CLI (`codex exec`), with the prompt in front of the task and QUMO's
             MCP server set through -c overrides.

The CLI back ends keep the same safety rules: research agents run read-only; an agent that
writes code runs inside a fresh sandbox worktree and may edit only there. Nothing commits
or pushes. The command line is built by a pure function so it can be checked without
running the CLI.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from qumo_agents.config import REPO_ROOT, Settings
from qumo_agents.registry import AgentSpec
from qumo_agents.tools import alphaxiv, qumo_mcp, repo

CLI_NOTE = """
Harness note: you run inside {harness}, not the deepagents runtime. Tool names differ from the
ones above: search papers and SDK releases with web search / fetch (arxiv.org, pypi.org,
github.com releases), read code with the file tools, call QUMO through the MCP server named
'qumo'. Save the report as agents/reports/<YYYY-MM-DD>-<name>.md {where}.
"""


def _wants_alphaxiv(spec: AgentSpec) -> bool:
    return "research" in spec.tool_groups and alphaxiv.available()


def _mcp_config_file(spec: AgentSpec) -> Path:
    servers = {}
    if spec.uses_mcp:
        servers["qumo"] = {"command": str(qumo_mcp.QUMO_BIN), "args": ["mcp"]}
    if _wants_alphaxiv(spec):
        servers["alphaxiv"] = alphaxiv.claude_code_server()
    cfg = {"mcpServers": servers}
    fh = tempfile.NamedTemporaryFile("w", suffix=".json", prefix="qumo-mcp-", delete=False)
    json.dump(cfg, fh)
    fh.close()
    return Path(fh.name)


def claude_code_command(spec: AgentSpec, task: str, settings: Settings, mcp_config: Path | None,
                        model: str | None = None) -> list[str]:
    cc = settings.section("claude_code")
    allowed = ["Read", "Grep", "Glob"]
    if "research" in spec.tool_groups:
        allowed += ["WebSearch", "WebFetch"]
    if "check" in spec.tool_groups or spec.writes_code:
        allowed += ["Bash(.venv/bin/mojo run:*)", "Bash(bench/.venv/bin/python examples/run_examples.py:*)",
                    "Bash(git diff:*)", "Bash(git status:*)", "Bash(build/qumo:*)"]
    allowed += ["Write(agents/reports/**)"]
    if spec.writes_code:
        allowed += ["Edit", "Write"]
    if spec.uses_mcp:
        allowed.append("mcp__qumo")
    if _wants_alphaxiv(spec):
        allowed += [f"mcp__alphaxiv__{t}" for t in alphaxiv.READ_TOOLS]
    where = "in this sandbox worktree" if spec.writes_code else "in the repository"
    cmd = [cc.get("command", "claude"), "-p", task,
           "--append-system-prompt", spec.prompt + CLI_NOTE.format(harness="Claude Code", where=where),
           "--allowedTools", ",".join(allowed),
           "--disallowedTools", "Bash(git commit:*),Bash(git push:*),Bash(gh:*)",
           "--output-format", "text",
           "--permission-mode", cc.get("permission_mode", "default")]
    if mcp_config:
        cmd += ["--mcp-config", str(mcp_config), "--strict-mcp-config"]
    if model or cc.get("model"):
        cmd += ["--model", model or cc["model"]]
    if cc.get("max_budget_usd"):
        cmd += ["--max-budget-usd", str(cc["max_budget_usd"])]
    return cmd


def codex_command(spec: AgentSpec, task: str, settings: Settings, workdir: Path, model: str | None = None) -> list[str]:
    cx = settings.section("codex")
    sandbox = "workspace-write" if spec.writes_code else cx.get("sandbox", "read-only")
    where = "in this sandbox worktree" if spec.writes_code else "(print it if you cannot write)"
    prompt = spec.prompt + CLI_NOTE.format(harness="Codex", where=where) + "\n\nTask:\n" + task
    cmd = [cx.get("command", "codex"), "exec", "--sandbox", sandbox, "--cd", str(workdir)]
    if spec.uses_mcp:
        cmd += ["-c", f"mcp_servers.qumo.command={json.dumps(str(qumo_mcp.QUMO_BIN))}",
                "-c", 'mcp_servers.qumo.args=["mcp"]']
    if _wants_alphaxiv(spec):
        cmd += alphaxiv.codex_overrides()
    if model or cx.get("model"):
        cmd += ["-m", model or cx["model"]]
    return cmd + [prompt]


def _workdir(spec: AgentSpec, task: str) -> Path:
    if not spec.writes_code:
        return REPO_ROOT
    opened = json.loads(repo.open_sandbox.invoke({"topic": f"{spec.name}-{task[:30]}"}))
    return Path(opened["sandbox"])


def run_cli_backend(backend: str, spec: AgentSpec, task: str, settings: Settings,
                    model: str | None = None, timeout_s: int = 3600) -> str:
    exe = settings.section("claude_code" if backend == "claude-code" else "codex").get(
        "command", "claude" if backend == "claude-code" else "codex")
    if shutil.which(exe) is None:
        raise RuntimeError(f"{exe!r} is not on PATH: install it or choose another --backend")
    workdir = _workdir(spec, task)
    if backend == "claude-code":
        mcp = _mcp_config_file(spec) if spec.uses_mcp or _wants_alphaxiv(spec) else None
        try:
            cmd = claude_code_command(spec, task, settings, mcp, model)
            r = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=timeout_s)
        finally:
            if mcp:
                mcp.unlink(missing_ok=True)
    else:
        cmd = codex_command(spec, task, settings, workdir, model)
        r = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=timeout_s)
    if r.returncode != 0:
        raise RuntimeError(f"{exe} exited {r.returncode}: {r.stderr.strip()[-2000:]}")
    return r.stdout
