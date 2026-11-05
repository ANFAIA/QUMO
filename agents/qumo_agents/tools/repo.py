"""Repository tools with hard limits.

* Reading is confined to the repository (no .venv, build, .git or secrets files).
* Writing goes to agents/reports/, or into a sandbox git worktree that the agent opens
  with open_sandbox on a fresh branch. The main checkout is never written.
* Commands come from an allow-list and run without a shell. Nothing commits or pushes:
  a human reviews the sandbox branch.
"""

from __future__ import annotations

import datetime as _dt
import fnmatch
import json
import os
import re
import subprocess
from pathlib import Path

from langchain_core.tools import tool

from qumo_agents.config import REPO_ROOT, REPORTS_DIR

SKIP_DIRS = {".git", ".venv", "build", "node_modules", "__pycache__", ".worktrees", ".pixi", "models"}
SECRET_NAMES = re.compile(r"(^\.env|secret|credential|\.pem$|\.key$|id_rsa)", re.I)
MAX_READ = 60_000
MAX_OUTPUT = 12_000
SANDBOX_ROOT = REPO_ROOT / ".worktrees"

_state = {"sandbox": None}  # Path of the sandbox worktree opened in this process


def _resolve(path: str, root: Path) -> Path:
    # The path as written is checked first, so a forbidden directory is refused by its own
    # rule even when it is a symlink that resolves outside the repository (.venv often is).
    lexical = Path(os.path.normpath(path))
    if set(lexical.parts) & SKIP_DIRS or SECRET_NAMES.search(lexical.name):
        raise ValueError(f"{path} is not readable by agents")
    p = (root / path).resolve()
    if p != root and root not in p.parents:
        raise ValueError(f"{path} is outside {root}")
    parts = set(p.relative_to(root).parts)
    if parts & SKIP_DIRS or SECRET_NAMES.search(p.name):
        raise ValueError(f"{path} is not readable by agents")
    return p


def _root() -> Path:
    return _state["sandbox"] or REPO_ROOT


@tool
def read_file(path: str, start_line: int = 1, max_lines: int = 400) -> str:
    """Read a text file of the QUMO repository (or of the open sandbox), with line
    numbers. Paths are relative to the repository root."""
    try:
        p = _resolve(path, _root())
        lines = p.read_text(errors="replace").splitlines()
    except (OSError, ValueError) as e:
        return f"error: {e}"
    chunk = lines[max(0, start_line - 1): start_line - 1 + max(1, min(max_lines, 2000))]
    out = "\n".join(f"{start_line + i}\t{line}" for i, line in enumerate(chunk))
    return out[:MAX_READ] + (f"\n... ({len(lines)} lines total)" if len(lines) > start_line - 1 + len(chunk) else "")


@tool
def list_files(path: str = ".", pattern: str = "*") -> str:
    """List files under a repository directory whose names match a glob pattern
    (e.g. path='qumo/compiler', pattern='*.mojo'). At most 300 entries."""
    try:
        base = _resolve(path, _root())
    except ValueError as e:
        return f"error: {e}"
    out = []
    for p in sorted(base.rglob("*")):
        rel = p.relative_to(_root())
        if set(rel.parts) & SKIP_DIRS or not p.is_file() or not fnmatch.fnmatch(p.name, pattern):
            continue
        out.append(str(rel))
        if len(out) >= 300:
            out.append("... truncated")
            break
    return "\n".join(out)


@tool
def search_code(regex: str, path: str = ".", glob: str = "*.mojo") -> str:
    """Search the repository with a regular expression; returns file:line: text,
    at most 200 matches. Default glob is Mojo sources."""
    try:
        rx = re.compile(regex)
        base = _resolve(path, _root())
    except (re.error, ValueError) as e:
        return f"error: {e}"
    hits = []
    for p in sorted(base.rglob(glob)):
        rel = p.relative_to(_root())
        if set(rel.parts) & SKIP_DIRS or not p.is_file():
            continue
        for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
            if rx.search(line):
                hits.append(f"{rel}:{i}: {line.strip()[:200]}")
                if len(hits) >= 200:
                    return "\n".join(hits + ["... truncated"])
    return "\n".join(hits) or "no matches"


# ----------------------------------------------------------------------------- commands
ALLOWED = [
    # (description, argv prefix pattern); every argument after the prefix is checked below.
    ("run one Mojo test or example", [".venv/bin/mojo", "run", "-I", "."]),
    ("run one example with its checks", ["bench/.venv/bin/python", "examples/run_examples.py", "--only"]),
    ("run the hardware plugin tests", ["bench/.venv/bin/python", "-m", "pytest", "-q", "plugins/hardware/tests"]),
    ("QUMO CLI", ["build/qumo"]),
    ("git status or diff of the sandbox", ["git", "diff"]),
    ("git status of the sandbox", ["git", "status", "--short"]),
]
ARG_OK = re.compile(r"^[\w./:=,@+-]+$")


@tool
def run_check(argv: list[str], timeout_s: int = 600) -> str:
    """Run one allow-listed command (no shell) in the sandbox if open, else in the
    repository: `.venv/bin/mojo run -I . tests/test_x.mojo`, `bench/.venv/bin/python
    examples/run_examples.py --only <file>`, the plugin pytest, `build/qumo ...`,
    `git diff`, `git status --short`. Returns exit code and the output tail."""
    if not any(argv[:len(prefix)] == prefix for _, prefix in ALLOWED):
        return "refused: not on the allow-list. Allowed: " + "; ".join(" ".join(p) for _, p in ALLOWED)
    if any(not ARG_OK.match(a) or ".." in a for a in argv):
        return "refused: arguments may only hold path-like characters"
    try:
        r = subprocess.run(argv, cwd=_root(), capture_output=True, text=True, timeout=max(10, min(timeout_s, 1800)))
    except subprocess.TimeoutExpired:
        return f"timeout after {timeout_s}s"
    except OSError as e:
        return f"error: {e}"
    out = (r.stdout + r.stderr)[-MAX_OUTPUT:]
    return f"exit {r.returncode}\n{out}"


# ----------------------------------------------------------------------------- writing
@tool
def write_report(name: str, markdown: str) -> str:
    """Save a markdown report under agents/reports/ as <date>-<name>.md and return
    its path. Use it for every finding list, research digest or plan."""
    slug = re.sub(r"[^\w-]+", "-", name).strip("-")[:80] or "report"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"{_dt.date.today().isoformat()}-{slug}.md"
    path.write_text(markdown)
    return str(path.relative_to(REPO_ROOT))


@tool
def open_sandbox(topic: str) -> str:
    """Create an isolated git worktree on a new branch agents/<topic>-<time> from the
    current HEAD, and route read/write/run tools into it. Call it once before any
    code change. Nothing is committed or pushed; a human reviews the branch."""
    if _state["sandbox"]:
        return f"sandbox already open: {_state['sandbox']}"
    slug = re.sub(r"[^\w-]+", "-", topic).strip("-")[:40] or "change"
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    branch = f"agents/{slug}-{stamp}"
    path = SANDBOX_ROOT / f"agent-{slug}-{stamp}"
    r = subprocess.run(["git", "worktree", "add", "-b", branch, str(path), "HEAD"],
                       cwd=REPO_ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        return f"error: {r.stderr.strip()[-500:]}"
    for link in (".venv", "build", "bench/.venv"):
        src = REPO_ROOT / link
        if src.exists() and not (path / link).exists():
            (path / link).symlink_to(src)
    _state["sandbox"] = path
    return json.dumps({"sandbox": str(path), "branch": branch})


@tool
def write_file(path: str, content: str) -> str:
    """Write a whole file inside the open sandbox (open_sandbox first). Paths are
    relative to the sandbox root. Refused when no sandbox is open."""
    root = _state["sandbox"]
    if root is None:
        return "refused: call open_sandbox first; the main checkout is read-only for agents"
    try:
        p = _resolve(path, root)
    except ValueError as e:
        return f"error: {e}"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"wrote {path} ({len(content)} chars)"


@tool
def edit_file(path: str, old: str, new: str) -> str:
    """Replace one exact, unique occurrence of `old` with `new` in a sandbox file."""
    root = _state["sandbox"]
    if root is None:
        return "refused: call open_sandbox first; the main checkout is read-only for agents"
    try:
        p = _resolve(path, root)
        text = p.read_text()
    except (OSError, ValueError) as e:
        return f"error: {e}"
    n = text.count(old)
    if n != 1:
        return f"refused: `old` occurs {n} times; it must occur exactly once"
    p.write_text(text.replace(old, new))
    return f"edited {path}"


READ_TOOLS = [read_file, list_files, search_code]
CHECK_TOOLS = [run_check]
REPORT_TOOLS = [write_report]
SANDBOX_TOOLS = [open_sandbox, write_file, edit_file]
