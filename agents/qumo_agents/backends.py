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
ones above: read code with the file tools. {research}{mcp}{checks}
Save the report as agents/reports/<YYYY-MM-DD>-<name>.md {where}.
"""

CHECK_NOTE = """
Run a Mojo file as `.venv/bin/mojo run -I . <path>.mojo` from the working directory: one
command, no `cd`, no pipe, no `make`.{refused} Look at changes with `git diff`, `git diff --stat`
or `git status --short`. Compile and run every file you change before you report; code you
could not run must be reported as not run."""


def _cli_note(spec: AgentSpec, harness: str, where: str) -> str:
    """What this agent really has in this harness: an agent told about a tool it lacks
    wastes its run asking for it."""
    research = ""
    if "research" in spec.tool_groups:
        papers = ("alphaXiv's read tools are attached." if _wants_alphaxiv(spec) else
                  "alphaXiv is not attached: say in the report that paper claims come from abstracts.")
        research = ("Search papers and SDK releases with web search / fetch (arxiv.org, pypi.org, "
                    f"github.com releases). {papers} ")
    mcp = ("Call QUMO through the MCP server named 'qumo'." if spec.uses_mcp else
           "No QUMO MCP server is attached to this agent: read the code and docs instead.")
    checks = ""
    if "check" in spec.tool_groups or spec.writes_code:
        # only Claude Code enforces an allow-list of commands; Codex relies on its sandbox
        checks = CHECK_NOTE.format(refused=" Any other form is refused." if harness == "Claude Code" else "")
    return CLI_NOTE.format(harness=harness, where=where, research=research, mcp=mcp, checks=checks)


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
        # git diff is allowed in fixed forms only: `git diff --output=<path>` writes a file
        # anywhere, so a prefix rule would hand a read-only agent a write.
        allowed += ["Bash(.venv/bin/mojo run:*)", "Bash(bench/.venv/bin/python examples/run_examples.py:*)",
                    "Bash(git diff)", "Bash(git diff --stat)", "Bash(git diff HEAD)", "Bash(git diff --stat HEAD)",
                    "Bash(git status)", "Bash(git status --short)", "Bash(build/qumo:*)"]
    # Claude Code grants file writes through Edit rules; a Write(path) rule is not honoured
    # in print mode, and the report write is then denied. Markdown only: an agent that may
    # also run `mojo run` must not be able to write a program into the report folder.
    allowed += ["Edit(agents/reports/*.md)"]
    if spec.writes_code:
        allowed += ["Edit", "Write"]
    if spec.uses_mcp:
        allowed.append("mcp__qumo")
    if _wants_alphaxiv(spec):
        allowed += [f"mcp__alphaxiv__{t}" for t in alphaxiv.READ_TOOLS]
    where = "in this sandbox worktree" if spec.writes_code else "in the repository"
    cmd = [cc.get("command", "claude"), "-p", task,
           "--append-system-prompt", spec.prompt + _cli_note(spec, "Claude Code", where),
           "--allowedTools", ",".join(allowed),
           "--disallowedTools", "Bash(git commit:*),Bash(git push:*),Bash(gh:*)",
           "--output-format", "text",
           "--permission-mode", cc.get("permission_mode", "default")]
    if mcp_config:
        cmd += ["--mcp-config", str(mcp_config), "--strict-mcp-config"]
    chosen = settings.model_for(spec.name, model, "claude-code")
    if chosen:
        cmd += ["--model", chosen]
    if cc.get("max_budget_usd"):
        cmd += ["--max-budget-usd", str(cc["max_budget_usd"])]
    return cmd


def codex_command(spec: AgentSpec, task: str, settings: Settings, workdir: Path, model: str | None = None) -> list[str]:
    cx = settings.section("codex")
    sandbox = "workspace-write" if spec.writes_code else cx.get("sandbox", "read-only")
    where = "in this sandbox worktree" if spec.writes_code else "(print it if you cannot write)"
    prompt = spec.prompt + _cli_note(spec, "Codex", where) + "\n\nTask:\n" + task
    cmd = [cx.get("command", "codex"), "exec", "--sandbox", sandbox, "--cd", str(workdir)]
    if spec.uses_mcp:
        cmd += ["-c", f"mcp_servers.qumo.command={json.dumps(str(qumo_mcp.QUMO_BIN))}",
                "-c", 'mcp_servers.qumo.args=["mcp"]']
    if _wants_alphaxiv(spec):
        cmd += alphaxiv.codex_overrides()
    chosen = settings.model_for(spec.name, model, "codex")
    if chosen:
        cmd += ["-m", chosen]
    return cmd + [prompt]


def mojo_command(spec: AgentSpec, task: str, settings: Settings, model: str | None = None,
                 module: str | None = None) -> list[str]:
    """The native Mojo harness (`build/qumo-agent`): the same agent on QUMO's own harness, with
    its LLM on the OpenAI-compatible endpoint of agents.json (OpenRouter by default).

    `task` is the plain task: the harness adds the module context itself from --module."""
    exe = settings.section("mojo").get("command", "build/qumo-agent")
    path = Path(exe) if Path(exe).is_absolute() else REPO_ROOT / exe
    cmd = [str(path), "run", spec.name, task]
    if module:
        cmd += ["--module", module]
    chosen = settings.model_for(spec.name, model, "mojo")
    if chosen:
        cmd += ["--model", chosen]
    return cmd


def run_mojo_backend(spec: AgentSpec, task: str, settings: Settings, model: str | None = None,
                     module: str | None = None, timeout_s: int = 3600) -> str:
    cmd = mojo_command(spec, task, settings, model, module)
    if not Path(cmd[0]).exists():
        raise RuntimeError(f"{cmd[0]} not found: build the native harness with `make agents-mojo`, "
                           "or choose another --backend")
    r = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout_s)
    if r.returncode != 0:
        raise RuntimeError(f"qumo-agent exited {r.returncode}: {r.stderr.strip()[-2000:]}")
    return r.stdout


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
