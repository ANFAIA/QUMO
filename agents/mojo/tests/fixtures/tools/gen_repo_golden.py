"""Write repo_golden.json: Python's repository tools on a fixed file tree.

Dev-time only (owner T, DESIGN.md 8.2). The tree (files, raw bytes, symlinks, plus a directory
outside the root) and the calls are defined here and copied into the JSON, so
test_tools_repo builds the same tree and makes the same calls with tools/repo.mojo.
The Python tools run with repo._state["sandbox"] pointing at the tree, so every read tool and
the sandbox write/edit tools use it as their root. Paths in the outputs are written with
<ROOT> and <OUT> in place of the temporary directories.

Calls where the Mojo port deliberately differs (X5, X14 in DESIGN.md 0.4) carry "mojo": the
expected Mojo text, and "why".

    agents/.venv/bin/python -I agents/mojo/tests/fixtures/tools/gen_repo_golden.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "agents"))

from qumo_agents.tools import repo  # noqa: E402

FILES = {
    "README.md": "line1\nline2\n",
    "src/a.mojo": "def main():\n    print('hello')  \n# TODO fix\n",
    "src/b.py": "x = 1\r\ny = 2\rz = 3\n",
    "src/c.txt": "a\fb c\x1cd\x85e\n",
    "a/b": "ab\n",
    "a.b": "dot\n",
    "a/c/d.mojo": "  TODO deep éé  \n",
    "a-b/e.mojo": "TODO dash\n",
    "big.txt": "".join(f"line {i}\n" for i in range(1, 2501)),
    "wide.txt": "é" * 70000 + "\nsecond\n",
    "long_line.mojo": "TODO " + "x" * 300 + "\n",
    ".env": "KEY=1 TODO\n",
    "config/secret.mojo": "TODO secret\n",
    "config/Credentials.txt": "TODO\n",
    "keys/id_rsa": "TODO\n",
    "server.PEM": "TODO\n",
    ".venv/x": "v\n",
    "build/out.mojo": "TODO build\n",
    ".git/config": "[core]\n",
    "models/m.mojo": "TODO model\n",
    "node_modules/p/index.mojo": "TODO node\n",
    "deep/__pycache__/x.mojo": "TODO cache\n",
    "tests/test_x.mojo": "def test(): # TODO\n",
    "Upper/MiXed.MOJO": "TODO upper\n",
}
FILES.update({f"many/f{i:03d}.txt": f"{i}\n" for i in range(310)})
BYTES = {"bad_utf8.mojo": "544f444f20ff0a"}   # b"TODO \xff\n"
OUTSIDE = {"o.mojo": "TODO outside\n"}
# symlink path -> target; "<OUT>" is the outside directory
LINKS = {
    "link_out.mojo": "<OUT>/o.mojo",
    "src/link_in.mojo": "a.mojo",
    "linkdir": "<OUT>",
    "a/c/up": "..",
}

X5_SECRET = "search_code skips secret-looking names (X5)"
X5_LINK = "search_code skips symlinks resolving outside the root (X5)"
X14 = "skip directories compared case-insensitively (X14)"

CALLS = [
    ("read_file", {"path": "README.md"}),
    ("read_file", {"path": "README.md", "start_line": 2, "max_lines": 1}),
    ("read_file", {"path": "src/b.py"}),
    ("read_file", {"path": "src/c.txt"}),
    ("read_file", {"path": "big.txt", "start_line": 10, "max_lines": 3}),
    ("read_file", {"path": "big.txt", "max_lines": 5000}),
    ("read_file", {"path": "big.txt", "start_line": 0, "max_lines": 2}),
    ("read_file", {"path": "big.txt", "start_line": -5, "max_lines": 3}),
    ("read_file", {"path": "big.txt", "start_line": 3000}),
    ("read_file", {"path": "big.txt", "start_line": 2499, "max_lines": 0}),
    ("read_file", {"path": "wide.txt"}),
    ("read_file", {"path": "bad_utf8.mojo"}),
    ("read_file", {"path": "../x"}),
    ("read_file", {"path": "/etc/passwd"}),
    ("read_file", {"path": ".env"}),
    ("read_file", {"path": ".venv/x"}),
    ("read_file", {"path": "build/out.mojo"}),
    ("read_file", {"path": ".git/config"}),
    ("read_file", {"path": "src/../.git/config"}),
    ("read_file", {"path": "link_out.mojo"}),
    ("read_file", {"path": "linkdir/o.mojo"}),
    ("read_file", {"path": "missing.txt"}),
    ("read_file", {"path": "src"}),
    ("read_file", {"path": "server.PEM"}),
    ("read_file", {"path": "config/secret.mojo"}),
    ("read_file", {"path": "config/Credentials.txt"}),
    ("read_file", {"path": "keys/id_rsa"}),
    ("read_file", {"path": "src/link_in.mojo"}),
    ("read_file", {"path": "a/c/up/b"}),
    ("read_file", {"path": "<ROOT>/README.md"}),
    ("read_file", {"path": ".GIT/config"}, "error: .GIT/config is not readable by agents", X14),
    ("read_file", {"path": "Build/out.mojo"}, "error: Build/out.mojo is not readable by agents", X14),
    ("read_file", {"path": ".Venv/x"}, "error: .Venv/x is not readable by agents", X14),
    ("read_file", {"path": "MODELS/m.mojo"}, "error: MODELS/m.mojo is not readable by agents", X14),
    ("list_files", {}),
    ("list_files", {"path": "a"}),
    ("list_files", {"path": ".", "pattern": "*.mojo"}),
    ("list_files", {"path": "src", "pattern": "[!a]*"}),
    ("list_files", {"path": "Upper", "pattern": "*.mojo"}),
    ("list_files", {"path": "nonexistent"}),
    ("list_files", {"path": "README.md"}),
    ("list_files", {"path": ".git"}),
    ("list_files", {"path": "linkdir"}),
    ("list_files", {"path": "../"}),
    ("search_code", {"regex": "TODO"}),
    ("search_code", {"regex": "print\\(", "path": "src"}),
    ("search_code", {"regex": "^x", "glob": "*.py"}),
    ("search_code", {"regex": "(?i)todo", "glob": "*"}),
    ("search_code", {"regex": "TODO", "glob": "c/*.mojo"}),
    ("search_code", {"regex": "TODO", "path": "a", "glob": "**/*.mojo"}),
    ("search_code", {"regex": "a", "path": "../"}),
    ("search_code", {"regex": "zzz_no_match"}),
    ("search_code", {"regex": "line \\d+", "glob": "big.txt"}),
    ("search_code", {"regex": "TODO", "path": ".git"}),
    ("search_code", {"regex": "TODO", "glob": "*.txt"}),
    ("write_file", {"path": "new/dir/f.txt", "content": "héllo"}),
    ("edit_file", {"path": "new/dir/f.txt", "old": "l", "new": "L"}),
    ("edit_file", {"path": "new/dir/f.txt", "old": "zz", "new": "L"}),
    ("edit_file", {"path": "new/dir/f.txt", "old": "hé", "new": "HE"}),
    ("read_file", {"path": "new/dir/f.txt"}),
    ("edit_file", {"path": "src/b.py", "old": "y = 2", "new": "y = 3"}),
    ("read_file", {"path": "src/b.py"}),
    ("edit_file", {"path": "missing.txt", "old": "a", "new": "b"}),
    ("edit_file", {"path": "bad_utf8.mojo", "old": "a", "new": "b"}),
    ("write_file", {"path": ".env", "content": "x"}),
    ("write_file", {"path": "../escape.txt", "content": "x"}),
    ("run_check", {"argv": ["rm", "-rf", "/"]}),
    ("run_check", {"argv": []}),
    ("run_check", {"argv": ["git", "diff", "$(id)"]}),
    ("run_check", {"argv": [".venv/bin/mojo", "run", "-I", ".", "../x.mojo"]}),
    ("run_check", {"argv": ["build/qumo", "a b"]}),
    ("run_check", {"argv": ["build/qumo", "é"]}, "refused: arguments may only hold path-like characters",
     "arguments are ASCII only (X5; Python's \\w is Unicode)"),
    ("run_check", {"argv": ["build/qumo", "x\n"]}, "refused: arguments may only hold path-like characters",
     "a trailing newline is refused (X5; Python's $ accepts it)"),
    ("run_check", {"argv": ["git", "diff", "--output=x"]}, "refused: arguments may only hold path-like characters",
     "git diff --output is refused (X5)"),
    ("run_check", {"argv": ["git", "diff", "--no-index", "a", "b"]}, "refused: arguments may only hold path-like characters",
     "git diff --no-index is refused (X5)"),
]
# Results that depend on the tree's search order and the X5 skips are recomputed below.
X5_SEARCHES = {("search_code", "TODO", "*.mojo"), ("search_code", "(?i)todo", "*"), ("search_code", "TODO", "*.txt")}


def build(root: Path, out: Path) -> None:
    for rel, text in FILES.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(text.encode())
    for rel, hx in BYTES.items():
        (root / rel).write_bytes(bytes.fromhex(hx))
    for rel, text in OUTSIDE.items():
        (out / rel).write_bytes(text.encode())
    for rel, target in LINKS.items():
        (root / rel).symlink_to(target.replace("<OUT>", str(out)))


def call(tool: str, args: dict, root: Path) -> str:
    fn = {"read_file": repo.read_file, "list_files": repo.list_files, "search_code": repo.search_code,
          "run_check": repo.run_check, "write_file": repo.write_file, "edit_file": repo.edit_file}[tool]
    args = {k: (v.replace("<ROOT>", str(root)) if isinstance(v, str) else v) for k, v in args.items()}
    return fn.invoke(args)


def main() -> None:
    base = Path(os.path.realpath(tempfile.mkdtemp(prefix="qumo-repo-golden-")))
    root, out = base / "root", base / "outside"
    root.mkdir()
    out.mkdir()
    build(root, out)
    repo._state["sandbox"] = root
    results = []
    for c in CALLS:
        tool, args = c[0], c[1]
        text = call(tool, args, root).replace(str(root), "<ROOT>").replace(str(out), "<OUT>")
        entry = {"tool": tool, "args": args, "python": text}
        if len(c) > 2:
            entry["mojo"], entry["why"] = c[2], c[3]
        key = (tool, args.get("regex"), args.get("glob", "*.mojo"))
        if key in X5_SEARCHES and "path" not in args:
            kept = [ln for ln in text.split("\n")
                    if not repo.SECRET_NAMES.search(ln.split(":", 1)[0].rsplit("/", 1)[-1])
                    and not ln.startswith("link_out.mojo:")]
            entry["mojo"] = "\n".join(kept) or "no matches"
            entry["why"] = X5_SECRET + "; " + X5_LINK
        results.append(entry)
    doc = {"schema": "qumo.test_repo_golden.v1",
           "files": FILES, "bytes": BYTES, "outside": OUTSIDE, "links": LINKS,
           "skip_dirs": sorted(repo.SKIP_DIRS), "calls": results}
    (HERE / "repo_golden.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")


main()
