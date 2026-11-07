"""Repository tools with hard limits (DESIGN.md 2.8.3), ported from
`agents/qumo_agents/tools/repo.py` with the same texts.

* Reading is confined to the repository, or the open sandbox (no .git, .venv, build, other
  skip directories, nor secret-looking names), after symlinks are resolved. Skip directories
  are compared case-insensitively (X14), so `.GIT/config` cannot reach `.git` on macOS.
* Writing goes to agents/reports/ of the main repository, or into a sandbox git worktree that
  the agent opens with open_sandbox on a fresh branch. The main checkout is never written.
* Commands come from an allow-list and run without a shell, with a minimal environment
  without secrets, in their own process group (killed on timeout). Inside a sandbox they run
  under the OS confinement CONF_CHECK (writes only to the sandbox and temp dirs, no network).
  Nothing commits or pushes: a human reviews the sandbox branch.

Every tool returns text; failures are `error: ...` or `refused: ...` results, never
exceptions (X4).
"""

from std.os import getenv

from qumo_agent.native.clock import local_date, local_stamp
from qumo_agent.native.fnmatch import fnmatch_py, pathlib_glob_match
from qumo_agent.native.fs import (
    exists,
    is_dir,
    is_file,
    is_symlink,
    join_path,
    list_dir,
    make_dirs,
    make_temp_dir,
    normpath,
    read_bytes,
    read_text_lossy,
    realpath_strict,
    remove_tree,
    resolve_nonstrict,
    symlink,
    write_text,
)
from qumo_agent.native.regex import PosixRegex, py_regex_to_ere
from qumo_agent.native.sandbox import CONF_CHECK, Confinement, confine
from qumo_agent.native.spawn import STDIO_NULL, STDIO_PIPE, SpawnSpec, minimal_child_env, run_capture
from qumo_agent.util.pyfmt import cp_head, cp_len, cp_tail, py_json_dumps_string, py_splitlines, py_strip, universal_newlines

comptime MAX_LIST_ENTRIES = 300
comptime MAX_SEARCH_HITS = 200
comptime SEARCH_LINE_CHARS = 200
comptime CHECK_MAX_BYTES = 4 * 1024 * 1024
comptime CHECK_FSIZE_BYTES = 1024 * 1024 * 1024
comptime OPEN_SANDBOX_TIMEOUT_MS = 120000
comptime REFUSED_NO_SANDBOX = "refused: call open_sandbox first; the main checkout is read-only for agents"
comptime REFUSED_ARGS = "refused: arguments may only hold path-like characters"
comptime REFUSED_UNSANDBOXED = "refused: no OS sandbox on this machine; run_check inside a sandbox needs one. Hint: pass --allow-unsandboxed-tool-tests to run it unconfined"


@fieldwise_init
struct RepoState(Copyable):
    var repo_root: String             # realpath of the repository root
    var sandbox: String               # "" until open_sandbox
    var skip_dirs: List[String]
    var max_read: Int                 # 60000
    var max_output: Int               # 12000
    var allowed: List[List[String]]
    var self_binary: String           # trampoline for confine (2.1.8)
    var sandbox_kind: String          # cached sandbox_kind(self_binary)
    var allow_unsandboxed: Bool       # --allow-unsandboxed-tool-tests (a human flag, 7.1)

    def root(self) -> String:
        """The open sandbox, else the repository (repo.py _root)."""
        if self.sandbox.byte_length() > 0:
            return self.sandbox
        return self.repo_root


# --------------------------------------------------------------------------- small helpers
def _strip_code(msg: String) -> String:
    """Python's own text of an error raised by native/ (`QA-E160: [Errno 2] ...` gives
    `[Errno 2] ...`)."""
    if msg.startswith("QA-E") and msg.byte_length() > 9 and String(msg[byte=7:9]) == ": ":
        return String(msg[byte=9:])
    return msg


def _ascii_lower(s: String) -> String:
    var bs = s.as_bytes()
    var out = List[UInt8](capacity=len(bs))
    for b in bs:
        var c = Int(b)
        if c >= 0x41 and c <= 0x5A:
            out.append(UInt8(c + 32))
        else:
            out.append(b)
    return String(unsafe_from_utf8=Span(out))


def _lower_set(items: List[String]) -> List[String]:
    var out = List[String]()
    for s in items:
        out.append(_ascii_lower(s))
    return out^


def _in(items: List[String], x: String) -> Bool:
    for s in items:
        if s == x:
            return True
    return False


def _parts(path: String) -> List[String]:
    """Path components without the root and empty or "." parts (pathlib parts, minus "/")."""
    var out = List[String]()
    for c in path.split("/"):
        var comp = String(c)
        if comp.byte_length() > 0 and comp != ".":
            out.append(comp)
    return out^


def _name(path: String) -> String:
    """pathlib's name: the last part ("" for "." and "/")."""
    var ps = _parts(path)
    if len(ps) == 0:
        return String()
    return ps[len(ps) - 1]


def _any_skipped(parts: List[String], skip_lower: List[String]) -> Bool:
    """set(parts) & SKIP_DIRS, both sides ASCII lower-cased (X14)."""
    for p in parts:
        if _in(skip_lower, _ascii_lower(p)):
            return True
    return False


def _relative_parts(p: String, root: String) -> Optional[List[String]]:
    """parts of p.relative_to(root), or None when p is neither root nor below it."""
    if p == root:
        return List[String]()
    var prefix = root if root.endswith("/") else root + "/"
    if not p.startswith(prefix):
        return None
    return _parts(String(p[byte=prefix.byte_length():]))


def _cp_at(bs: Span[Byte, _], i: Int) -> Tuple[Int, Int]:
    var b0 = Int(bs[i])
    if b0 < 0x80:
        return (b0, 1)
    if b0 < 0xE0 and i + 1 < len(bs):
        return (((b0 & 0x1F) << 6) | (Int(bs[i + 1]) & 0x3F), 2)
    if b0 < 0xF0 and i + 2 < len(bs):
        return (((b0 & 0x0F) << 12) | ((Int(bs[i + 1]) & 0x3F) << 6) | (Int(bs[i + 2]) & 0x3F), 3)
    if i + 3 < len(bs):
        return (((b0 & 0x07) << 18) | ((Int(bs[i + 1]) & 0x3F) << 12) | ((Int(bs[i + 2]) & 0x3F) << 6) | (Int(bs[i + 3]) & 0x3F), 4)
    return (b0, 1)


def _letter_matches(letter: Int, cp: Int) -> Bool:
    """`re.I` match of an ASCII lower-case letter (Python 3.12 sre: the letter, its upper case,
    and U+017F for s, U+0130/U+0131 for i, U+212A for k)."""
    if cp == letter:
        return True
    if letter >= 0x61 and letter <= 0x7A and cp == letter - 32:
        return True
    if letter == 0x73 and cp == 0x17F:
        return True
    if letter == 0x69 and (cp == 0x130 or cp == 0x131):
        return True
    if letter == 0x6B and cp == 0x212A:
        return True
    return False


def _ci_match_at(bs: Span[Byte, _], at: Int, word: String) -> Int:
    """End offset when `word` (ASCII lower case and punctuation) matches at `at` under re.I."""
    var w = word.as_bytes()
    var i = at
    for k in range(len(w)):
        if i >= len(bs):
            return -1
        var c = _cp_at(bs, i)
        if not _letter_matches(Int(w[k]), c[0]):
            return -1
        i += c[1]
    return i


def is_secret_name(name: String) -> Bool:
    """SECRET_NAMES.search(name): `(^\\.env|secret|credential|\\.pem$|\\.key$|id_rsa)` with
    re.I ($ also matches before a final newline)."""
    var bs = name.as_bytes()
    if _ci_match_at(bs, 0, ".env") >= 0:
        return True
    var exts: List[String] = [".pem", ".key"]
    var i = 0
    while i < len(bs):
        if _ci_match_at(bs, i, "secret") >= 0 or _ci_match_at(bs, i, "credential") >= 0 or _ci_match_at(bs, i, "id_rsa") >= 0:
            return True
        for ext in exts:
            var e = _ci_match_at(bs, i, ext)
            if e >= 0 and (e == len(bs) or (e == len(bs) - 1 and Int(bs[e]) == 0x0A)):
                return True
        i += _cp_at(bs, i)[1]
    return False


# --------------------------------------------------------------------------- paths
def resolve_repo_path(path: String, root: String, skip_dirs: List[String]) raises -> String:
    """repo.py _resolve, with skip directories compared case-insensitively (X14) and the
    real path of an existing target checked against the real root. Raises the plain
    message (`<path> is not readable by agents` / `<path> is outside <root>`); callers
    return `error: <message>`."""
    var skip = _lower_set(skip_dirs)
    var refused = path + " is not readable by agents"
    # The path as written first, so a forbidden directory is refused by its own rule even
    # when it is a symlink that resolves outside the repository (.venv often is).
    var lexical = normpath(path)
    if _any_skipped(_parts(lexical), skip) or is_secret_name(_name(lexical)):
        raise Error(refused)
    var p = resolve_nonstrict(join_path(root, path))
    var rel = _relative_parts(p, root)
    if not rel:
        raise Error(path + " is outside " + root)
    if _any_skipped(rel.value(), skip) or is_secret_name(_name(p)):
        raise Error(refused)
    if exists(p):
        var rp = realpath_strict(p)
        var rr = realpath_strict(root)
        var rrel = _relative_parts(rp, rr)
        if not rrel:
            raise Error(path + " is outside " + root)
        if _any_skipped(rrel.value(), skip) or is_secret_name(_name(rp)):
            raise Error(refused)
    return p^


def _inside_real_root(p: String, real_root: String) -> Bool:
    try:
        if _relative_parts(realpath_strict(p), real_root):
            return True
        return False
    except:
        return False


# --------------------------------------------------------------------------- read_file
def _py_slice_bounds(n: Int, start: Int, stop: Int) -> Tuple[Int, Int]:
    """Python's list[start:stop] bounds for a list of length n (step 1)."""
    var a = start
    var b = stop
    if a < 0:
        a = max(0, a + n)
    if b < 0:
        b = max(0, b + n)
    a = min(a, n)
    b = min(b, n)
    if b < a:
        b = a
    return (a, b)


def _read_lines(p: String) raises -> List[String]:
    """p.read_text(errors="replace").splitlines()."""
    return py_splitlines(universal_newlines(read_text_lossy(p)))


def tool_read_file(st: RepoState, path: String, start_line: Int, max_lines: Int) -> String:
    var lines: List[String]
    try:
        var p = resolve_repo_path(path, st.root(), st.skip_dirs)
        lines = _read_lines(p)
    except e:
        return "error: " + _strip_code(String(e))
    var b = _py_slice_bounds(len(lines), max(0, start_line - 1), start_line - 1 + max(1, min(max_lines, 2000)))
    var out = String()
    for i in range(b[0], b[1]):
        if i > b[0]:
            out += "\n"
        out += String(start_line + (i - b[0])) + "\t" + lines[i]
    var text = cp_head(out, st.max_read)
    if len(lines) > start_line - 1 + (b[1] - b[0]):
        text += "\n... (" + String(len(lines)) + " lines total)"
    return text^


# --------------------------------------------------------------------------- walking
def _parts_less(a: String, b: String) -> Bool:
    """pathlib order: compare the lists of parts (a '/' sorts before every character)."""
    var x = a.as_bytes()
    var y = b.as_bytes()
    var n = min(len(x), len(y))
    for i in range(n):
        var cx = Int(x[i])
        var cy = Int(y[i])
        if cx == cy:
            continue
        if cx == 0x2F:
            return True
        if cy == 0x2F:
            return False
        return cx < cy
    return len(x) < len(y)


def _sort_paths(mut items: List[String]):
    """Merge sort in pathlib order (stable)."""
    var n = len(items)
    if n < 2:
        return
    var src = items.copy()
    var dst = items.copy()
    var width = 1
    while width < n:
        var lo = 0
        while lo < n:
            var mid = min(lo + width, n)
            var hi = min(lo + 2 * width, n)
            var i = lo
            var j = mid
            var k = lo
            while i < mid and j < hi:
                if _parts_less(src[j], src[i]):
                    dst[k] = src[j]
                    j += 1
                else:
                    dst[k] = src[i]
                    i += 1
                k += 1
            while i < mid:
                dst[k] = src[i]
                i += 1
                k += 1
            while j < hi:
                dst[k] = src[j]
                j += 1
                k += 1
            lo += 2 * width
        var tmp = src^
        src = dst^
        dst = tmp^
        width *= 2
    items = src^


def _collect_files(dir: String, rel: String, skip_lower: List[String], mut out: List[String]):
    """Root-relative paths of the regular files (following file symlinks) below `dir`,
    without entering symlinked directories or skip directories (their files would be
    filtered out anyway). Unreadable directories are skipped, as rglob does."""
    var names: List[String]
    try:
        names = list_dir(dir)
    except:
        return
    for name in names:
        var p = dir + "/" + name
        var r = name if rel.byte_length() == 0 else rel + "/" + name
        if is_dir(p):
            if is_symlink(p) or _in(skip_lower, _ascii_lower(name)):
                continue
            _collect_files(p, r, skip_lower, out)
        elif is_file(p):
            out.append(r)


def _walk_files(base: String, root: String, skip_lower: List[String]) -> List[String]:
    """Root-relative paths of the files below `base` (resolved, inside root), pathlib order."""
    var out = List[String]()
    var rel = _relative_parts(base, root)
    if not rel:
        return out^
    var prefix = String("/").join(rel.value())
    if is_dir(base):
        _collect_files(base, prefix, skip_lower, out)
    _sort_paths(out)
    return out^


# --------------------------------------------------------------------------- list_files
def tool_list_files(st: RepoState, path: String, pattern: String) -> String:
    var root = st.root()
    var base: String
    try:
        base = resolve_repo_path(path, root, st.skip_dirs)
    except e:
        return "error: " + _strip_code(String(e))
    var skip = _lower_set(st.skip_dirs)
    var out = List[String]()
    for rel in _walk_files(base, root, skip):
        var parts = _parts(rel)
        if _any_skipped(parts, skip) or not fnmatch_py(parts[len(parts) - 1], pattern):
            continue
        out.append(rel)
        if len(out) >= MAX_LIST_ENTRIES:
            out.append("... truncated")
            break
    return String("\n").join(out)


# --------------------------------------------------------------------------- search_code
def tool_search_code(st: RepoState, regex: String, path: String, glob: String) -> String:
    var root = st.root()
    var ere: Tuple[String, Bool]
    try:
        ere = py_regex_to_ere(regex)
    except e:
        return "error: " + _strip_code(String(e))
    var rx: PosixRegex
    try:
        rx = PosixRegex(ere[0], ere[1])
    except e:
        return "error: " + _strip_code(String(e))
    var base: String
    try:
        base = resolve_repo_path(path, root, st.skip_dirs)
    except e:
        return "error: " + _strip_code(String(e))
    var real_root: String
    try:
        real_root = realpath_strict(root)
    except:
        real_root = root
    var skip = _lower_set(st.skip_dirs)
    var base_rel = _relative_parts(base, root)
    var below = 0
    if base_rel:
        below = len(base_rel.value())
    var hits = List[String]()
    for rel in _walk_files(base, root, skip):
        var parts = _parts(rel)
        if _any_skipped(parts, skip):
            continue
        var under_base = List[String]()
        for k in range(below, len(parts)):
            under_base.append(parts[k])
        if not pathlib_glob_match(under_base, glob):
            continue
        var name = parts[len(parts) - 1]
        var p = root + "/" + rel
        # X5: secret-looking files and symlinks leaving the root are never searched.
        if is_secret_name(name):
            continue
        if is_symlink(p) and not _inside_real_root(p, real_root):
            continue
        var lines: List[String]
        try:
            lines = _read_lines(p)
        except:
            continue   # unreadable file (X4)
        for i in range(len(lines)):
            if rx.search(lines[i]):
                hits.append(rel + ":" + String(i + 1) + ": " + cp_head(py_strip(lines[i]), SEARCH_LINE_CHARS))
                if len(hits) >= MAX_SEARCH_HITS:
                    hits.append("... truncated")
                    return String("\n").join(hits)
    if len(hits) == 0:
        return "no matches"
    return String("\n").join(hits)


# --------------------------------------------------------------------------- run_check
def _arg_ok(a: String) -> Bool:
    """ARG_OK = [A-Za-z0-9_./:=,@+-]+ (ASCII), no "..", no newline (X5)."""
    var bs = a.as_bytes()
    if len(bs) == 0:
        return False
    for b in bs:
        var c = Int(b)
        var ok = (c >= 0x30 and c <= 0x39) or (c >= 0x41 and c <= 0x5A) or (c >= 0x61 and c <= 0x7A)
        ok = ok or c == 0x5F or c == 0x2E or c == 0x2F or c == 0x3A or c == 0x3D or c == 0x2C or c == 0x40 or c == 0x2B or c == 0x2D
        if not ok:
            return False
    return a.find("..") < 0


def _allowed(argv: List[String], allowed: List[List[String]]) -> Bool:
    for prefix in allowed:
        if len(argv) < len(prefix):
            continue
        var same = True
        for i in range(len(prefix)):
            if argv[i] != prefix[i]:
                same = False
                break
        if same:
            return True
    return False


def run_check_refusal(argv: List[String], allowed: List[List[String]]) -> String:
    """The refusal text for argv, or "" when it may run (repo.py run_check + X5)."""
    if not _allowed(argv, allowed):
        var listed = List[String]()
        for p in allowed:
            listed.append(String(" ").join(p))
        return "refused: not on the allow-list. Allowed: " + String("; ").join(listed)
    for a in argv:
        if not _arg_ok(a):
            return REFUSED_ARGS
    if len(argv) >= 2 and argv[0] == "git" and argv[1] == "diff":
        for i in range(2, len(argv)):
            if argv[i].startswith("--output") or argv[i] == "--no-index":
                return REFUSED_ARGS
    return String()


def _env_with(var env: List[String], name: String, value: String) -> List[String]:
    var out = List[String]()
    var prefix = name + "="
    for e in env:
        if not e.startswith(prefix):
            out.append(e)
    out.append(prefix + value)
    return out^


def tool_run_check(st: RepoState, argv: List[String], timeout_s: Int) -> String:
    var refusal = run_check_refusal(argv, st.allowed)
    if refusal.byte_length() > 0:
        return refusal
    var confined = st.sandbox.byte_length() > 0
    if confined and st.sandbox_kind == "none" and not st.allow_unsandboxed:
        return REFUSED_UNSANDBOXED
    var env = minimal_child_env()
    var spec = SpawnSpec(argv[0], argv.copy(), env.copy(), st.root(), STDIO_NULL, STDIO_PIPE, STDIO_PIPE, True)
    var tmp = String()
    if confined:
        # The sandbox's content was written by agents: confine it (CONF_CHECK, 6.8).
        try:
            tmp = make_temp_dir("qumo-check-")
            var cache = st.repo_root + "/build/agent-sandbox-cache"
            make_dirs(cache)
            env = _env_with(env^, "TMPDIR", tmp + "/")
            env = _env_with(env^, "MODULAR_CACHE_DIR", cache)
            spec.env = env.copy()
            if st.sandbox_kind != "none":
                var write_dirs: List[String] = [realpath_strict(st.sandbox), realpath_strict(tmp), realpath_strict(cache)]
                var home = String()
                var h = getenv("HOME", "")
                if h.byte_length() > 0:
                    try:
                        home = realpath_strict(h)
                    except:
                        home = h
                var c = Confinement(CONF_CHECK, write_dirs^, List[String](), home, 0, CHECK_FSIZE_BYTES)
                spec = confine(spec, c, st.self_binary)
        except e:
            if tmp.byte_length() > 0:
                try:
                    remove_tree(tmp)
                except:
                    pass
            return "error: " + _strip_code(String(e))
    var timeout = max(10, min(timeout_s, 1800))
    var result: String
    try:
        var r = run_capture(spec, timeout * 1000, CHECK_MAX_BYTES, "")
        if r.timed_out:
            result = "timeout after " + String(timeout_s) + "s"
        else:
            var out = universal_newlines(r.stdout) + universal_newlines(r.stderr)
            result = "exit " + String(r.exit_code) + "\n" + cp_tail(out, st.max_output)
    except e:
        result = "error: " + _strip_code(String(e))
    if tmp.byte_length() > 0:
        try:
            remove_tree(tmp)
        except:
            pass
    return result^


# --------------------------------------------------------------------------- writing
def report_slug(name: String, max_len: Int, fallback: String) -> String:
    """re.sub(r"[^\\w-]+", "-", name).strip("-")[:max_len] or fallback, with ASCII word
    characters only (X10: non-ASCII letters become '-', so file names stay ASCII)."""
    var bs = name.as_bytes()
    var out = List[UInt8]()
    var in_run = False
    for b in bs:
        var c = Int(b)
        var word = (c >= 0x30 and c <= 0x39) or (c >= 0x41 and c <= 0x5A) or (c >= 0x61 and c <= 0x7A) or c == 0x5F or c == 0x2D
        if word:
            out.append(b)
            in_run = False
        elif not in_run:
            out.append(0x2D)
            in_run = True
    var a = 0
    var e = len(out)
    while a < e and Int(out[a]) == 0x2D:
        a += 1
    while e > a and Int(out[e - 1]) == 0x2D:
        e -= 1
    e = min(e, a + max(0, max_len))
    if e <= a:
        return fallback
    return String(unsafe_from_utf8=Span(out)[a:e])


def tool_write_report(st: RepoState, name: String, markdown: String) -> String:
    """<repo_root>/agents/reports/<date>-<slug>.md (always the main repository); returns the
    repository-relative path."""
    var rel = "agents/reports/" + local_date() + "-" + report_slug(name, 80, "report") + ".md"
    try:
        make_dirs(st.repo_root + "/agents/reports")
        write_text(st.repo_root + "/" + rel, markdown)
    except e:
        return "error: " + _strip_code(String(e))
    return rel^


def tool_open_sandbox(mut st: RepoState, topic: String) -> String:
    """A git worktree on a new branch agents/<topic>-<time> from HEAD, shared by every agent
    of the process tree; .venv, build and bench/.venv are linked to the repository's."""
    if st.sandbox.byte_length() > 0:
        return "sandbox already open: " + st.sandbox
    var slug = report_slug(topic, 40, "change")
    var stamp = local_stamp()
    var branch = "agents/" + slug + "-" + stamp
    var path = st.repo_root + "/.worktrees/agent-" + slug + "-" + stamp
    var argv: List[String] = ["git", "worktree", "add", "-b", branch, path, "HEAD"]
    var spec = SpawnSpec("git", argv^, minimal_child_env(), st.repo_root, STDIO_NULL, STDIO_PIPE, STDIO_PIPE, True)
    try:
        var r = run_capture(spec, OPEN_SANDBOX_TIMEOUT_MS, CHECK_MAX_BYTES, "")
        if r.timed_out:
            return "error: git worktree add timed out after " + String(OPEN_SANDBOX_TIMEOUT_MS // 1000) + "s"
        if r.exit_code != 0:
            return "error: " + cp_tail(py_strip(universal_newlines(r.stderr)), 500)
    except e:
        return "error: " + _strip_code(String(e))
    var links: List[String] = [".venv", "build", "bench/.venv"]
    for link in links:
        var src = st.repo_root + "/" + link
        var dst = path + "/" + link
        if exists(src) and not exists(dst):
            try:
                symlink(src, dst)
            except:
                pass   # e.g. bench/ missing in the worktree: the link is a convenience
    st.sandbox = path
    return "{\"sandbox\": " + py_json_dumps_string(path, True) + ", \"branch\": " + py_json_dumps_string(branch, True) + "}"


def tool_sandbox_write(st: RepoState, path: String, content: String) -> String:
    if st.sandbox.byte_length() == 0:
        return REFUSED_NO_SANDBOX
    try:
        var p = resolve_repo_path(path, st.sandbox, st.skip_dirs)
        var slash = p.rfind("/")
        if slash > 0:
            make_dirs(String(p[byte=0:slash]))
        write_text(p, content)
    except e:
        return "error: " + _strip_code(String(e))
    return "wrote " + path + " (" + String(cp_len(content)) + " chars)"


def _utf8_error(bs: Span[Byte, _]) -> String:
    """Python's UnicodeDecodeError text for invalid UTF-8, "" when valid."""
    var i = 0
    var n = len(bs)
    while i < n:
        var b0 = Int(bs[i])
        if b0 < 0x80:
            i += 1
            continue
        var need: Int
        var lo = 0x80
        var hi = 0xBF
        if b0 >= 0xC2 and b0 <= 0xDF:
            need = 1
        elif b0 >= 0xE0 and b0 <= 0xEF:
            need = 2
            if b0 == 0xE0:
                lo = 0xA0
            elif b0 == 0xED:
                hi = 0x9F
        elif b0 >= 0xF0 and b0 <= 0xF4:
            need = 3
            if b0 == 0xF0:
                lo = 0x90
            elif b0 == 0xF4:
                hi = 0x8F
        else:
            return "'utf-8' codec can't decode byte 0x" + _hex2(b0) + " in position " + String(i) + ": invalid start byte"
        for k in range(1, need + 1):
            if i + k >= n:
                if i + k - 1 == i:
                    return "'utf-8' codec can't decode byte 0x" + _hex2(b0) + " in position " + String(i) + ": unexpected end of data"
                return "'utf-8' codec can't decode bytes in position " + String(i) + "-" + String(n - 1) + ": unexpected end of data"
            var c = Int(bs[i + k])
            var l = lo if k == 1 else 0x80
            var h = hi if k == 1 else 0xBF
            if c < l or c > h:
                return "'utf-8' codec can't decode byte 0x" + _hex2(b0) + " in position " + String(i) + ": invalid continuation byte"
        i += need + 1
    return String()


def _hex2(v: Int) -> String:
    var digits = "0123456789abcdef".as_bytes()
    var out = List[UInt8]()
    out.append(digits[(v >> 4) & 0xF])
    out.append(digits[v & 0xF])
    return String(unsafe_from_utf8=Span(out))


def _count(text: String, old: String) -> Int:
    """str.count: non-overlapping occurrences; "" occurs len(text) + 1 times."""
    if old.byte_length() == 0:
        return cp_len(text) + 1
    var n = 0
    var at = 0
    while True:
        var k = text.find(old, at)
        if k < 0:
            return n
        n += 1
        at = k + old.byte_length()


def _replace_all(text: String, old: String, new: String) -> String:
    """str.replace(old, new) (every occurrence; "" inserts `new` around every character)."""
    if old.byte_length() == 0:
        var bs = text.as_bytes()
        var out = new
        var i = 0
        while i < len(bs):
            var w = _cp_at(bs, i)[1]
            out += String(unsafe_from_utf8=bs[i:i + w]) + new
            i += w
        return out^
    var out = String()
    var at = 0
    while True:
        var k = text.find(old, at)
        if k < 0:
            out += String(text[byte=at:])
            return out^
        out += String(text[byte=at:k]) + new
        at = k + old.byte_length()


def tool_sandbox_edit(st: RepoState, path: String, old: String, new: String, replace_all: Bool) -> String:
    if st.sandbox.byte_length() == 0:
        return REFUSED_NO_SANDBOX
    var p: String
    var text: String
    try:
        p = resolve_repo_path(path, st.sandbox, st.skip_dirs)
        var raw = read_bytes(p)
        var bad = _utf8_error(Span(raw))
        if bad.byte_length() > 0:
            return "error: " + bad
        text = universal_newlines(String(unsafe_from_utf8=Span(raw)))
    except e:
        return "error: " + _strip_code(String(e))
    var n = _count(text, old)
    if not replace_all and n != 1:
        return "refused: `old` occurs " + String(n) + " times; it must occur exactly once"
    if replace_all and n == 0:
        return "refused: `old` occurs 0 times; it must occur at least once"
    try:
        write_text(p, _replace_all(text, old, new))
    except e:
        return "error: " + _strip_code(String(e))
    return "edited " + path
