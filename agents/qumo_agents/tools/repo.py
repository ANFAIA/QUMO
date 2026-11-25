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
import shutil
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


def _forbidden(parts) -> bool:
    """True when any path component is a skipped directory or a secret-looking name.
    The one rule behind _resolve, list_files and search_code, so they cannot drift."""
    return any(part in SKIP_DIRS or SECRET_NAMES.search(part) for part in parts)


def _walkable(p: Path, root: Path) -> bool:
    """A file found while walking: regular, with nothing forbidden in its root-relative
    path or in where it really points (a symlink must not leave the root or reach a secret)."""
    try:
        if _forbidden(p.relative_to(root).parts) or not p.is_file():
            return False
        real = p.resolve()
        return real == root.resolve() or not _forbidden(real.relative_to(root.resolve()).parts)
    except (OSError, ValueError):
        return False


def _resolve(path: str, root: Path) -> Path:
    # The path as written is checked first, so a forbidden directory is refused by its own
    # rule even when it is a symlink that resolves outside the repository (.venv often is).
    lexical = Path(os.path.normpath(path))
    if _forbidden(lexical.parts):
        raise ValueError(f"{path} is not readable by agents")
    p = (root / path).resolve()
    if p != root and root not in p.parents:
        raise ValueError(f"{path} is outside {root}")
    if _forbidden(p.relative_to(root).parts):
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
        if not fnmatch.fnmatch(p.name, pattern) or not _walkable(p, _root()):
            continue
        rel = p.relative_to(_root())
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
        if not _walkable(p, _root()):
            continue
        rel = p.relative_to(_root())
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
GIT_FLAGS = frozenset({"--stat", "--name-only", "--name-status", "--cached", "--staged", "--"})
# `build/qumo` takes subcommands and options from these lists only. `tool` and `mcp` are left
# out (they call any registered tool). Output options write a file, so they are refused unless
# a sandbox is open and the path resolves inside it.
QUMO_SUBCOMMANDS = frozenset({"run", "statevector", "expect", "transpile", "emulate", "validate", "info",
                              "draw", "convert", "vqe", "describe", "version", "help"})
QUMO_OUTPUT_OPTIONS = frozenset({"-o", "--output"})
QUMO_OPTIONS = frozenset({
    "--shots", "--seed", "--params", "--max-qubits", "--max-iterations", "--device", "--precision",
    "--json", "--pretty", "--max-amplitudes", "--observable", "--target", "--level", "--basis",
    "--pipeline", "--qasm", "--compile", "--no-compile", "--width", "--unicode", "--to",
    "--instructions", "--layers", "--entanglement", "--optimizer", "--gradient", "--maxiter",
    "--qubits", "-h", "--help"})
# Checks run code the agent may have written, so they do not inherit credentials.
SECRET_ENV = re.compile(r"TOKEN|SECRET|KEY|PASSWORD|CREDENTIAL|AUTH|COOKIE|SESSION", re.I)


def _check_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if not SECRET_ENV.search(k)}


def _qumo_refusal(args: list[str]) -> str:
    """Why `build/qumo <args>` is refused, or "" when it may run.

    Every file argument passes the same rule as read_file (`_resolve`: inside the root, no
    skipped directory, no secret-looking name), so `validate .env` is refused, and an output
    option is accepted only with an open sandbox and a path inside it."""
    if not args or args[0] not in QUMO_SUBCOMMANDS:
        return "refused: build/qumo subcommands allowed here: " + " ".join(sorted(QUMO_SUBCOMMANDS))
    sandbox = _state["sandbox"]
    i = 1
    while i < len(args):
        a = args[i]
        i += 1
        if a.startswith("-") and a != "-":
            name, eq, value = a.partition("=")
            if name in QUMO_OUTPUT_OPTIONS:
                if not eq:
                    value = args[i] if i < len(args) else ""
                    i += 1
                if sandbox is None:
                    return ("refused: build/qumo output options write a file; open_sandbox first "
                            "(the main checkout is read-only for agents)")
                if not value:
                    return f"refused: {name} needs a path"
                try:
                    _resolve(value, sandbox)
                except ValueError as e:
                    return f"refused: {e}"
                continue
            if name not in QUMO_OPTIONS:
                return f"refused: build/qumo option {name} is not allowed here"
            if not eq:
                continue
            a = value  # an `--opt=value` value is checked like a positional argument
        if a == "-":
            continue
        try:
            _resolve(a, _root())
        except ValueError as e:
            return f"refused: {e}"
    return ""


@tool
def run_check(argv: list[str], timeout_s: int = 600) -> str:
    """Run one allow-listed command (no shell) in the sandbox if open, else in the
    repository: `.venv/bin/mojo run -I . tests/test_x.mojo`, `bench/.venv/bin/python
    examples/run_examples.py --only <file>`, the plugin pytest, `build/qumo ...`,
    `git diff`, `git status --short`. `build/qumo` takes the run, statevector, expect, transpile,
    emulate, validate, info, draw, convert, vqe and describe subcommands; its file arguments follow
    the read_file rules, and -o/--output needs an open sandbox and a path inside it.
    Returns exit code and the output tail."""
    if not any(argv[:len(prefix)] == prefix for _, prefix in ALLOWED):
        return "refused: not on the allow-list. Allowed: " + "; ".join(" ".join(p) for _, p in ALLOWED)
    if any(not ARG_OK.match(a) or ".." in a for a in argv):
        return "refused: arguments may only hold path-like characters"
    # Paths stay inside the working tree: no absolute path, bare or as an option value.
    if any(a.startswith("/") or "=/" in a for a in argv):
        return "refused: absolute paths are not allowed"
    if argv[0] == "build/qumo":
        why = _qumo_refusal(argv[1:])
        if why:
            return why
    if argv[0] == "git":
        # `git diff` reads any two files with --no-index and writes one with --output.
        extra = argv[2:] if argv[1] == "diff" else argv[3:]
        if any(a.startswith("-") and a not in GIT_FLAGS for a in extra):
            return "refused: git options allowed here: " + " ".join(sorted(GIT_FLAGS))
    try:
        r = subprocess.run(argv, cwd=_root(), env=_check_env(), capture_output=True, text=True,
                           timeout=max(10, min(timeout_s, 1800)))
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
    # Reports are git-ignored, so the new worktree has none: copy them, or an agent told to
    # implement a proposal cannot read the report it comes from.
    for src in sorted((REPO_ROOT / "agents" / "reports").glob("*.md")):
        dst = path / "agents" / "reports" / src.name
        # regular files of report size only: a symlink would copy whatever it points to
        if src.is_symlink() or not src.is_file() or src.stat().st_size > 1_000_000:
            continue
        if not dst.exists() and not dst.is_symlink():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
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
