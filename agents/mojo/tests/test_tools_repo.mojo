"""Tests for tools/repo.mojo (DESIGN.md 2.8.3, 5.4, 8.2).

1. Golden: tests/fixtures/tools/repo_golden.json holds a file tree (with CRLF, form feeds,
   U+2028, invalid UTF-8, secret names, skip directories, symlinks inside and outside the
   root) and Python's repo.py outputs for read_file, list_files, search_code, write_file,
   edit_file and run_check refusals on it (gen_repo_golden.py). The test builds the same
   tree and compares every text; calls where the port deliberately differs (X5, X14) carry
   the expected Mojo text.
2. Mojo-only behaviour: run_check with injected allow-lists (minimal environment without
   secrets, timeout, universal newlines per stream, a spawn error, git status in a temp
   repository, the CONF_CHECK confinement of a sandbox), secret names, report slugs,
   write_report, open_sandbox in a temp git repository, replace_all edits.

The confinement check needs a binary that runs `__confine` (tests/fixtures/tools/
confine_trampoline.mojo); it is built on demand into build/test-fixtures/."""

from std.os import getenv, setenv, unsetenv, stat
from std.testing import assert_equal, assert_true, assert_false

from qumo.ai.json import JsonValue, parse_json
from qumo_agent.native.clock import local_date, monotonic_ms
from qumo_agent.native.fs import (
    exists,
    file_mode,
    is_symlink,
    make_dirs,
    make_temp_dir,
    read_text_lossy,
    realpath_strict,
    remove_tree,
    symlink,
    write_text,
)
from qumo_agent.native.sandbox import sandbox_kind
from qumo_agent.native.spawn import STDIO_NULL, STDIO_PIPE, SpawnSpec, minimal_child_env, run_capture
from qumo_agent.tools.repo import (
    RepoState,
    REFUSED_NO_SANDBOX,
    REFUSED_UNSANDBOXED,
    is_secret_name,
    report_slug,
    resolve_repo_path,
    tool_list_files,
    tool_open_sandbox,
    tool_read_file,
    tool_run_check,
    tool_sandbox_edit,
    tool_sandbox_write,
    tool_search_code,
    tool_write_report,
)

comptime FX = "agents/mojo/tests/fixtures/tools"
comptime TRAMPOLINE_SRC = "agents/mojo/tests/fixtures/tools/confine_trampoline.mojo"
comptime TRAMPOLINE_BIN = "build/test-fixtures/tools_confine_trampoline"


def write_bytes(path: String, data: List[UInt8]) raises:
    with open(path, "w") as f:
        f.write_bytes(Span(data))


def _hexval(c: Int) -> Int:
    if c >= 0x30 and c <= 0x39:
        return c - 0x30
    return (c | 0x20) - 0x61 + 10


def hex_bytes(h: String) raises -> List[UInt8]:
    var out = List[UInt8]()
    var bs = h.as_bytes()
    for i in range(0, len(bs), 2):
        out.append(UInt8(_hexval(Int(bs[i])) * 16 + _hexval(Int(bs[i + 1]))))
    return out^


def parent_of(p: String) -> String:
    var k = p.rfind("/")
    return String(p[byte=0:k])


def string_list(v: JsonValue) raises -> List[String]:
    var out = List[String]()
    for i in range(len(v)):
        out.append(v.at(i).as_string())
    return out^


def state(root: String, skip: List[String], allowed: List[List[String]]) -> RepoState:
    return RepoState(root, String(), skip.copy(), 60000, 12000, allowed.copy(), String(), String("none"), False)


def default_allowed() -> List[List[String]]:
    var a = List[List[String]]()
    a.append([".venv/bin/mojo", "run", "-I", "."])
    a.append(["bench/.venv/bin/python", "examples/run_examples.py", "--only"])
    a.append(["bench/.venv/bin/python", "-m", "pytest", "-q", "plugins/hardware/tests"])
    a.append(["build/qumo"])
    a.append(["git", "diff"])
    a.append(["git", "status", "--short"])
    return a^


def run(argv: List[String], cwd: String) raises -> Int:
    var r = run_capture(SpawnSpec(argv[0], argv.copy(), minimal_child_env(), cwd, STDIO_NULL, STDIO_PIPE, STDIO_PIPE, True), 120000, 1 << 20, "")
    if r.exit_code != 0:
        print("command failed:", String(" ").join(argv), r.stdout, r.stderr)
    return r.exit_code


# --------------------------------------------------------------------------- golden
def build_tree(g: JsonValue, root: String, outside_dir: String) raises:
    var files = g.get("files")
    for i in range(len(files)):
        var p = root + "/" + files.key_at(i)
        make_dirs(parent_of(p))
        write_text(p, files.at(i).as_string())
    var raw = g.get("bytes")
    for i in range(len(raw)):
        write_bytes(root + "/" + raw.key_at(i), hex_bytes(raw.at(i).as_string()))
    var outside = g.get("outside")
    for i in range(len(outside)):
        write_text(outside_dir + "/" + outside.key_at(i), outside.at(i).as_string())
    var links = g.get("links")
    for i in range(len(links)):
        symlink(links.at(i).as_string().replace("<OUT>", outside_dir), root + "/" + links.key_at(i))


def arg_s(args: JsonValue, key: String, default: String, root: String) raises -> String:
    return args.get_string(key, default).replace("<ROOT>", root)


def call(mut st: RepoState, tool: String, args: JsonValue, root: String) raises -> String:
    if tool == "read_file":
        return tool_read_file(st, arg_s(args, "path", "", root), args.get_int("start_line", 1), args.get_int("max_lines", 400))
    if tool == "list_files":
        return tool_list_files(st, arg_s(args, "path", ".", root), arg_s(args, "pattern", "*", root))
    if tool == "search_code":
        return tool_search_code(st, arg_s(args, "regex", "", root), arg_s(args, "path", ".", root), arg_s(args, "glob", "*.mojo", root))
    if tool == "write_file":
        return tool_sandbox_write(st, arg_s(args, "path", "", root), arg_s(args, "content", "", root))
    if tool == "edit_file":
        return tool_sandbox_edit(st, arg_s(args, "path", "", root), arg_s(args, "old", "", root), arg_s(args, "new", "", root), False)
    if tool == "run_check":
        return tool_run_check(st, string_list(args.get("argv")), args.get_int("timeout_s", 600))
    raise Error("unknown tool " + tool)


def test_golden(base: String) raises:
    var g = parse_json(read_text_lossy(FX + "/repo_golden.json"))
    var root = base + "/root"
    var out = base + "/outside"
    make_dirs(root)
    make_dirs(out)
    build_tree(g, root, out)
    var st = state(root, string_list(g.get("skip_dirs")), default_allowed())
    st.sandbox = root     # as the generator does: _state["sandbox"] = root
    var calls = g.get("calls")
    var checked = 0
    for i in range(len(calls)):
        var c = calls.at(i)
        var tool = c.get_string("tool", "")
        var want = c.get_string("mojo", c.get_string("python", ""))
        var got = call(st, tool, c.get("args"), root).replace(root, "<ROOT>").replace(out, "<OUT>")
        assert_equal(got, want, "call " + String(i) + " " + tool + " " + c.get("args").to_json())
        checked += 1
    assert_true(checked >= 70)


# --------------------------------------------------------------------------- resolve
def expect_error(path: String, root: String, skip: List[String], want: String) raises:
    try:
        _ = resolve_repo_path(path, root, skip)
        assert_true(False, "accepted " + path)
    except e:
        assert_equal(String(e), want)


def test_resolve(base: String) raises:
    var root = base + "/resolve"
    make_dirs(root + "/sub/.GIT")
    make_dirs(root + "/BUILD")
    var skip: List[String] = [".git", ".venv", "build", "models"]
    expect_error("../x", root, skip, "../x is outside " + root)
    expect_error(".env", root, skip, ".env is not readable by agents")
    expect_error(".ENV.local", root, skip, ".ENV.local is not readable by agents")
    expect_error(".git/HEAD", root, skip, ".git/HEAD is not readable by agents")
    expect_error(".GIT/config", root, skip, ".GIT/config is not readable by agents")
    expect_error("sub/.Git/HEAD", root, skip, "sub/.Git/HEAD is not readable by agents")
    expect_error("Build/x", root, skip, "Build/x is not readable by agents")
    expect_error(".Venv/x", root, skip, ".Venv/x is not readable by agents")
    expect_error("/etc/passwd", root, skip, "/etc/passwd is outside " + root)
    # A symlink inside the root that leads to a skip directory is refused by its real path.
    symlink(root + "/BUILD", root + "/innocent")
    expect_error("innocent", root, skip, "innocent is not readable by agents")
    assert_equal(resolve_repo_path("sub", root, skip), root + "/sub")
    assert_equal(resolve_repo_path(".", root, skip), root)
    assert_equal(resolve_repo_path("", root, skip), root)
    assert_equal(resolve_repo_path("sub/../new.txt", root, skip), root + "/new.txt")


def test_search_errors(base: String) raises:
    """Patterns POSIX ERE cannot express are refused with the translator's text."""
    var root = base + "/search"
    make_dirs(root)
    write_text(root + "/a.mojo", "alpha\n")
    var st = state(root, [".git"], default_allowed())
    var look = tool_search_code(st, "(?=x)alpha", ".", "*.mojo")
    assert_true(look.startswith("error: unsupported in POSIX ERE: "), look)
    var bad = tool_search_code(st, "a(", "../nowhere", "*.mojo")
    assert_true(bad.startswith("error: ") and bad.find("outside") < 0, bad)
    assert_equal(tool_search_code(st, "\\balpha\\b", ".", "*.mojo"), "a.mojo:1: alpha")
    assert_equal(tool_search_code(st, "ALPHA", ".", "*.mojo"), "no matches")


def test_secret_names() raises:
    var yes: List[String] = [".env", ".env.local", ".ENVIRONMENT", "my_secret.txt", "SECRET", "aws-credentials",
                             "server.pem", "x.PEM", "x.pem\n", "a.key", "id_rsa.pub", "\u017fecret", "x.\u212aey",
                             "credent\u0130al"]
    var no: List[String] = ["my.env", "x.pem.txt", "keys", "README.md", "secre", "x.pem\n\n", "idrsa"]
    for n in yes:
        assert_true(is_secret_name(n), "should be secret: " + n)
    for n in no:
        assert_false(is_secret_name(n), "should not be secret: " + n)


def test_report_slug() raises:
    # Python: re.sub(r"[^\w-]+", "-", name).strip("-")[:n] or fallback (ASCII inputs).
    assert_equal(report_slug("Hello World!", 80, "report"), "Hello-World")
    assert_equal(report_slug("--a--b--", 80, "report"), "a--b")
    assert_equal(report_slug("", 80, "report"), "report")
    assert_equal(report_slug("!!!", 80, "report"), "report")
    assert_equal(report_slug("a/b\\c..d", 80, "report"), "a-b-c-d")
    var long = String()
    for _ in range(100):
        long += "x"
    assert_equal(report_slug(long, 80, "report").byte_length(), 80)
    assert_equal(report_slug("abc def ghi", 5, "change"), "abc-d")
    assert_equal(report_slug("abcd ef", 5, "change"), "abcd-")
    # X10: non-ASCII letters become '-'
    assert_equal(report_slug("ñandú report", 80, "report"), "and-report")
    assert_equal(report_slug("éé", 40, "change"), "change")


# --------------------------------------------------------------------------- run_check
def test_run_check(base: String) raises:
    var root = base + "/check"
    make_dirs(root)
    var skip: List[String] = [".git"]
    var allowed = List[List[String]]()
    allowed.append(["/usr/bin/env"])
    allowed.append(["/bin/sleep"])
    allowed.append(["/bin/sh", "crlf.sh"])
    allowed.append(["./nope"])
    var st = state(root, skip, allowed)

    # minimal environment: no secrets reach the child (D2b)
    _ = setenv("X_API_KEY", "k-123")
    _ = setenv("GITHUB_TOKEN", "t-123")
    _ = setenv("MY_SECRET", "s-123")
    var env_out = tool_run_check(st, ["/usr/bin/env"], 30)
    assert_true(env_out.startswith("exit 0\n"), env_out)
    assert_true(env_out.find("PATH=") >= 0, env_out)
    assert_true(env_out.find("X_API_KEY") < 0 and env_out.find("GITHUB_TOKEN") < 0 and env_out.find("MY_SECRET") < 0, env_out)
    assert_true(env_out.find("k-123") < 0 and env_out.find("t-123") < 0 and env_out.find("s-123") < 0, env_out)
    _ = unsetenv("X_API_KEY")
    _ = unsetenv("GITHUB_TOKEN")
    _ = unsetenv("MY_SECRET")

    # universal newlines per stream, stdout then stderr
    write_text(root + "/crlf.sh", "printf 'a\\r\\nb\\rc'\nprintf 'x\\r\\ny' >&2\nexit 3\n")
    assert_equal(tool_run_check(st, ["/bin/sh", "crlf.sh"], 30), "exit 3\na\nb\ncx\ny")
    st.max_output = 4
    assert_equal(tool_run_check(st, ["/bin/sh", "crlf.sh"], 30), "exit 3\ncx\ny")
    st.max_output = 12000

    # spawn failure: Python's OSError text
    assert_equal(tool_run_check(st, ["./nope"], 30), "error: [Errno 2] No such file or directory: './nope'")

    # timeout: clamped to at least 10 s, reported with the requested value
    var t0 = monotonic_ms()
    assert_equal(tool_run_check(st, ["/bin/sleep", "60"], 1), "timeout after 1s")
    var took = monotonic_ms() - t0
    assert_true(took >= 9000 and took < 20000, "timeout took " + String(took) + " ms")

    # the sandbox needs an OS sandbox unless a human allowed running without one
    st.sandbox = root
    st.sandbox_kind = "none"
    assert_equal(tool_run_check(st, ["/usr/bin/env"], 30), REFUSED_UNSANDBOXED)
    st.allow_unsandboxed = True
    var unconfined = tool_run_check(st, ["/usr/bin/env"], 30)
    assert_true(unconfined.startswith("exit 0\n") and unconfined.find("TMPDIR=") >= 0 and unconfined.find("MODULAR_CACHE_DIR=" + root + "/build/agent-sandbox-cache") >= 0, unconfined)


def test_git_status(base: String) raises:
    var g = base + "/gitrepo"
    make_dirs(g)
    assert_equal(run(["git", "init", "-q", "."], g), 0)
    write_text(g + "/f.txt", "x\n")
    var st = state(realpath_strict(g), [".git"], default_allowed())
    assert_equal(tool_run_check(st, ["git", "status", "--short"], 60), "exit 0\n?? f.txt\n")


def mtime(path: String) raises -> Int:
    return Int(stat(path).st_mtimespec.tv_sec)


def ensure_trampoline() raises -> String:
    """The binary at TRAMPOLINE_BIN, built when missing or older than its sources."""
    if exists(TRAMPOLINE_BIN):
        var built = mtime(TRAMPOLINE_BIN)
        var fresh = True
        var sources: List[String] = [TRAMPOLINE_SRC, "agents/mojo/qumo_agent/native/sandbox.mojo",
                                     "agents/mojo/qumo_agent/native/spawn.mojo", "agents/mojo/qumo_agent/native/libc.mojo"]
        for src in sources:
            if mtime(src) > built:
                fresh = False
        if fresh:
            return realpath_strict(TRAMPOLINE_BIN)
    make_dirs("build/test-fixtures")
    var argv: List[String] = [".venv/bin/mojo", "build", "-I", ".", "-I", "agents/mojo", "-o", TRAMPOLINE_BIN, TRAMPOLINE_SRC]
    var env = minimal_child_env()
    var r = run_capture(SpawnSpec(argv[0], argv.copy(), env^, String(), STDIO_NULL, STDIO_PIPE, STDIO_PIPE, True), 600000, 1 << 20, "")
    if r.exit_code != 0:
        raise Error("building the confinement trampoline failed: " + r.stdout + r.stderr)
    return realpath_strict(TRAMPOLINE_BIN)


def test_confined_check(base: String) raises:
    var tramp = ensure_trampoline()
    var kind = sandbox_kind(tramp)
    if kind == "none":
        print("test_tools_repo: no OS sandbox on this machine; confined run_check not checked")
        return
    var repo_root = base + "/confined_repo"
    var sandbox = repo_root + "/.worktrees/agent-x"
    make_dirs(sandbox)
    var outside = base + "/confined_outside.txt"
    var allowed = List[List[String]]()
    allowed.append(["/usr/bin/touch"])
    allowed.append(["/usr/bin/env"])
    var st = RepoState(realpath_strict(repo_root), realpath_strict(sandbox), [".git"], 60000, 12000, allowed^, tramp, kind, False)
    var inside = tool_run_check(st, ["/usr/bin/touch", "inside.txt"], 30)
    assert_equal(inside, "exit 0\n")
    assert_true(exists(sandbox + "/inside.txt"))
    var denied = tool_run_check(st, ["/usr/bin/touch", outside], 30)
    assert_true(denied.startswith("exit 1\n") and denied.find("Operation not permitted") >= 0, denied)
    assert_false(exists(outside))
    var env_out = tool_run_check(st, ["/usr/bin/env"], 30)
    assert_true(env_out.find("MODULAR_CACHE_DIR=" + st.repo_root + "/build/agent-sandbox-cache") >= 0, env_out)
    assert_true(env_out.find("TMPDIR=") >= 0, env_out)


# --------------------------------------------------------------------------- writing
def test_write_report(base: String) raises:
    var root = base + "/reports_repo"
    make_dirs(root)
    var st = state(root, [".git"], default_allowed())
    st.sandbox = root + "/elsewhere"   # reports always go to the main repository
    var rel = tool_write_report(st, "Weekly digest: routing!", "# Report\n")
    assert_equal(rel, "agents/reports/" + local_date() + "-Weekly-digest-routing.md")
    assert_equal(read_text_lossy(root + "/" + rel), "# Report\n")
    assert_equal(tool_write_report(st, "Weekly digest: routing!", "# Second\n"), rel)
    assert_equal(read_text_lossy(root + "/" + rel), "# Second\n")
    write_text(root + "/blocker", "file, not a directory")
    var st2 = state(root + "/blocker", [".git"], default_allowed())
    assert_true(tool_write_report(st2, "x", "y").startswith("error: [Errno "))


def test_sandbox_edit_and_refusals(base: String) raises:
    var root = base + "/edit"
    make_dirs(root)
    var st = state(root, [".git"], default_allowed())
    assert_equal(tool_sandbox_write(st, "a.txt", "x"), REFUSED_NO_SANDBOX)
    assert_equal(tool_sandbox_edit(st, "a.txt", "x", "y", False), REFUSED_NO_SANDBOX)
    assert_equal(REFUSED_NO_SANDBOX, "refused: call open_sandbox first; the main checkout is read-only for agents")
    st.sandbox = root
    assert_equal(tool_sandbox_write(st, "a.txt", "one two one"), "wrote a.txt (11 chars)")
    assert_equal(tool_sandbox_edit(st, "a.txt", "one", "1", False), "refused: `old` occurs 2 times; it must occur exactly once")
    assert_equal(tool_sandbox_edit(st, "a.txt", "zzz", "1", True), "refused: `old` occurs 0 times; it must occur at least once")
    assert_equal(tool_sandbox_edit(st, "a.txt", "one", "1", True), "edited a.txt")
    assert_equal(read_text_lossy(root + "/a.txt"), "1 two 1")
    assert_equal(tool_sandbox_edit(st, "a.txt", "two", "2", False), "edited a.txt")
    assert_equal(read_text_lossy(root + "/a.txt"), "1 2 1")


def test_open_sandbox(base: String) raises:
    var g = base + "/sandbox_repo"
    make_dirs(g + "/bench")
    write_text(g + "/README.md", "hello\n")
    write_text(g + "/bench/README.md", "bench\n")
    write_text(g + "/.gitignore", ".venv\nbuild\nbench/.venv\n.worktrees\n")
    assert_equal(run(["git", "init", "-q", "."], g), 0)
    assert_equal(run(["git", "add", "-A"], g), 0)
    assert_equal(run(["git", "-c", "user.name=t", "-c", "user.email=t@example.org", "-c", "commit.gpgsign=false", "commit", "-q", "-m", "init"], g), 0)
    make_dirs(g + "/.venv")
    make_dirs(g + "/build")
    make_dirs(g + "/bench/.venv")
    var real = realpath_strict(g)
    var st = state(real, [".git"], default_allowed())
    var res = parse_json(tool_open_sandbox(st, "Speed up routing!"))
    var path = res.get_string("sandbox", "")
    var branch = res.get_string("branch", "")
    assert_true(path.startswith(real + "/.worktrees/agent-Speed-up-routing-"), path)
    assert_true(branch.startswith("agents/Speed-up-routing-"), branch)
    # the stamp is %Y%m%d-%H%M%S
    assert_equal(String(path[byte=path.byte_length() - 15:]), String(branch[byte=branch.byte_length() - 15:]))
    assert_equal(st.sandbox, path)
    assert_equal(st.root(), path)
    assert_true(exists(path + "/README.md"))
    assert_true(is_symlink(path + "/.venv") and is_symlink(path + "/build") and is_symlink(path + "/bench/.venv"))
    assert_equal(realpath_strict(path + "/build"), real + "/build")
    assert_equal(tool_open_sandbox(st, "again"), "sandbox already open: " + path)
    var r = run_capture(SpawnSpec("git", ["git", "branch", "--list", branch], minimal_child_env(), real, STDIO_NULL, STDIO_PIPE, STDIO_PIPE, True), 60000, 1 << 20, "")
    assert_true(r.stdout.find(branch) >= 0, r.stdout)
    # read and write now go to the sandbox
    assert_equal(tool_sandbox_write(st, "new.txt", "n"), "wrote new.txt (1 chars)")
    assert_true(exists(path + "/new.txt") and not exists(real + "/new.txt"))
    assert_equal(tool_read_file(st, "new.txt", 1, 400), "1\tn")
    # not a git repository: git's own message
    var plain = base + "/not_git"
    make_dirs(plain)
    var st2 = state(realpath_strict(plain), [".git"], default_allowed())
    var err = tool_open_sandbox(st2, "x")
    assert_true(err.startswith("error: fatal: not a git repository"), err)
    assert_equal(st2.sandbox, "")


def main() raises:
    var base = realpath_strict(make_temp_dir("qumo-tools-repo-"))
    test_golden(base)
    test_resolve(base)
    test_search_errors(base)
    test_secret_names()
    test_report_slug()
    test_run_check(base)
    test_git_status(base)
    test_confined_check(base)
    test_write_report(base)
    test_sandbox_edit_and_refusals(base)
    test_open_sandbox(base)
    remove_tree(base)
    print("test_tools_repo OK")
