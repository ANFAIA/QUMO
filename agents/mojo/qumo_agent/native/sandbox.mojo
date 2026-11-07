"""OS confinement of model-written code (K5; DESIGN.md 2.1.8 and 6.8).

A confined child is spawned through this binary's `__confine` trampoline (no
shell): `confine(spec, c, self)` rewrites a `SpawnSpec` into
`self __confine CONF_JSON PATH ARGV[1:]...`, and `confine_main` (dispatched by
`main.mojo` before any other work) applies the limits and execs the program:

- macOS: `setrlimit`, then `execve("/usr/bin/sandbox-exec", ["sandbox-exec", "-p",
  macos_profile(c), PATH, ARGV...], environ)`.
- Linux: `fork`; in the child (the only thread: `unshare(CLONE_NEWUSER)` fails with
  EINVAL in a multithreaded process and a Mojo binary starts runtime threads before
  `main`) `setrlimit`, `unshare(CLONE_NEWUSER | CLONE_NEWNET)` (no network: the new
  namespace has only a down loopback), `prctl(PR_SET_NO_NEW_PRIVS)`, a Landlock
  ruleset over every file-system right of the running ABI, `landlock_restrict_self`,
  then `execve(PATH, [PATH, ARGV...], environ)`. The trampoline waits and ends with the
  child's exit code, or with the signal that killed it (`wait_and_mirror`).

Profiles (6.8): `CONF_TOOL` (generated tool binaries: writes only to `write_dirs`,
no reads under `$HOME` except `read_allow`, no network, no fork on macOS),
`CONF_BUILD` and `CONF_CHECK` (writes only to `write_dirs`, no network, no reads of
`sensitive_home_paths` on macOS: named credential stores plus every other top-level
dot entry of `$HOME` except a few toolchain locations). Any failure in the trampoline prints
`QA-E163: OS confinement failed: <detail>` on stderr and exits 125.

RLIMIT_CPU caveat (measured on macOS 26 with Mojo 1.2): a Mojo-built program
installs a SIGXCPU handler (the runtime's crash reporter prints a stack dump and the
program keeps running) and macOS sends nothing more at the hard limit, so the CPU
limit stops shells and C programs but not a busy Mojo tool on macOS; there the
caller's timeout and process-group kill are what end it. On Linux soft == hard
makes the kernel send SIGKILL, which cannot be caught.

The Linux branch is compile-checked only (no Linux host was available while it was
written); `sandbox_kind` probes it for real before anything relies on it.
"""

from std.os import getenv, listdir
from std.os.path import isdir as _isdir

from qumo.ai.json import JsonWriter, JsonKind, parse_json
from qumo_agent.native.libc import (
    CStr,
    CStrArray,
    buf_addr,
    store_u32,
    store_i64,
    errno,
    strerror_text,
    is_macos,
    c_access,
    c_close,
    c_execve,
    c_exit_now,
    c_fork,
    c_getpid,
    c_kill,
    c_open,
    c_prctl,
    c_setrlimit,
    c_syscall4,
    c_unshare,
    c_waitpid,
    c_write,
    X_OK,
    EINTR,
    ENOENT,
    O_CLOEXEC,
    O_PATH,
    RLIMIT_CPU,
    RLIMIT_FSIZE,
    CLONE_NEWUSER,
    CLONE_NEWNET,
    PR_SET_NO_NEW_PRIVS,
    load_i32,
    SYS_LANDLOCK_CREATE_RULESET,
    SYS_LANDLOCK_ADD_RULE,
    SYS_LANDLOCK_RESTRICT_SELF,
)
from qumo_agent.native.spawn import (
    SpawnSpec,
    STDIO_NULL,
    resolve_program,
    run_capture,
    environ_list,
    which,
)
from qumo_agent.native.fs import exists, realpath_strict


comptime CONF_CHECK = 1     # run_check whose cwd is the sandbox
comptime CONF_BUILD = 2     # mojo build of a generated tool (toolsmith_build, tools review)
comptime CONF_TOOL = 3      # a generated tool binary (toolsmith_test, calls, review re-test)

comptime _SANDBOX_EXEC = "/usr/bin/sandbox-exec"
comptime _CONFINE_FAILED = "QA-E163: OS confinement failed: "
comptime _NO_SANDBOX = "QA-E163: no OS sandbox is available on this machine"

# Landlock (linux/landlock.h)
comptime _LL_EXECUTE = 1 << 0
comptime _LL_WRITE_FILE = 1 << 1
comptime _LL_READ_FILE = 1 << 2
comptime _LL_READ_DIR = 1 << 3
comptime _LL_REFER = 1 << 13
comptime _LL_TRUNCATE = 1 << 14
comptime _LL_IOCTL_DEV = 1 << 15
comptime _LL_FILE_RIGHTS = _LL_EXECUTE | _LL_WRITE_FILE | _LL_READ_FILE | _LL_TRUNCATE | _LL_IOCTL_DEV
comptime _LL_CREATE_RULESET_VERSION = 1
comptime _LL_RULE_PATH_BENEATH = 1


@fieldwise_init
struct Confinement(Copyable):
    var profile: Int               # CONF_*
    var write_dirs: List[String]   # realpaths of the only subtrees the child may write (plus /dev/null)
    var read_allow: List[String]   # CONF_TOOL: the only subtrees readable below `home`; CONF_CHECK/CONF_BUILD: kept readable inside a denied dir
    var home: String               # realpath of $HOME of this process ("" when unset)
    var cpu_s: Int                 # RLIMIT_CPU soft and hard, 0 = unchanged
    var fsize_bytes: Int           # RLIMIT_FSIZE soft and hard, 0 = unchanged

    def to_json(self) -> String:
        """`{"profile":..,"write_dirs":[..],"read_allow":[..],"home":"..","cpu_s":..,"fsize_bytes":..}`."""
        var w = JsonWriter()
        w.begin_object()
        w.kv_int("profile", self.profile)
        w.kv_string_list("write_dirs", self.write_dirs)
        w.kv_string_list("read_allow", self.read_allow)
        w.kv_string("home", self.home)
        w.kv_int("cpu_s", self.cpu_s)
        w.kv_int("fsize_bytes", self.fsize_bytes)
        w.end_object()
        return w.finish()


def _bad_conf(detail: String) -> Error:
    return Error(_CONFINE_FAILED + "malformed confinement: " + detail)


def confinement_from_json(text: String) raises -> Confinement:
    """Parses `Confinement.to_json()` output; raises QA-E163 on any other shape."""
    var v = parse_json(text)
    if not v.is_object():
        raise _bad_conf("not a JSON object")
    var profile = v.get("profile")
    if profile.kind() != JsonKind.NUMBER:
        raise _bad_conf("profile")
    var p = profile.as_int()
    if p != CONF_CHECK and p != CONF_BUILD and p != CONF_TOOL:
        raise _bad_conf("unknown profile " + String(p))
    var lists = List[List[String]]()
    for key in ["write_dirs", "read_allow"]:
        var arr = v.get(key)
        if not arr.is_array():
            raise _bad_conf(key)
        var items = List[String]()
        for i in range(len(arr)):
            var item = arr.at(i)
            if not item.is_string():
                raise _bad_conf(key)
            items.append(item.as_string())
        lists.append(items^)
    var home = v.get("home")
    if not home.is_string():
        raise _bad_conf("home")
    var cpu = v.get("cpu_s")
    var fsize = v.get("fsize_bytes")
    if cpu.kind() != JsonKind.NUMBER or fsize.kind() != JsonKind.NUMBER:
        raise _bad_conf("limits")
    var cpu_s = cpu.as_int()
    var fsize_bytes = fsize.as_int()
    if cpu_s < 0 or fsize_bytes < 0:
        raise _bad_conf("negative limit")
    return Confinement(p, lists[0].copy(), lists[1].copy(), home.as_string(), cpu_s, fsize_bytes)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
comptime _SENSITIVE_HOME_NAMES: List[String] = [
    ".ssh", ".aws", ".azure", ".gnupg", ".config", ".docker", ".kube", ".netrc",
    ".git-credentials", ".pypirc", ".npmrc", ".zsh_history", ".bash_history", "Library/Keychains",
    # Credential stores of other CLIs and shells (found readable on a typical developer Mac).
    ".codex", ".claude", ".claude.json", ".claude.json.backup", ".gemini", ".hf-cli", ".huggingface",
    ".qiskit", ".cache", ".cargo", ".gitconfig", ".zshrc", ".zprofile", ".zshenv", ".profile",
    ".bashrc", ".bash_profile",
    "Library/Application Support", "Library/Cookies", "Library/Safari", "Library/Mail",
    "Library/Messages", "Library/Containers", "Library/Group Containers",
]

# Top-level entries of $HOME that start with "." and are not denied: language and
# toolchain locations a confined build or check has to read (uv-managed Python lives in
# ~/.local/share/uv) and a file CoreFoundation reads.
comptime _HOME_DOT_ALLOWED: List[String] = [".local", ".modular", ".pixi", ".pyenv", ".rustup", ".CFUserTextEncoding"]


def _home_deny_entry(base: String, rel: String, mut out: List[String]):
    var p = base + "/" + rel
    var shown = p
    if exists(p):
        try:
            shown = realpath_strict(p)
        except:
            pass
    out.append(shown)
    if shown != p:
        out.append(p)


def sensitive_home_paths(home: String) -> List[String]:
    """Paths under `home` denied for reads under CONF_CHECK and CONF_BUILD (macOS):
    the named credential stores and shell files in `_SENSITIVE_HOME_NAMES` (.ssh .aws
    .config .codex .claude .gemini .cache ... Library/Keychains ...), then every other
    existing top-level dotfile or dot directory of `home` except `_HOME_DOT_ALLOWED`
    (deny by default for the places CLIs keep tokens). An existing path is given as
    its realpath (the sandbox matches resolved paths); a missing one as `home/name`."""
    var out = List[String]()
    if home.byte_length() == 0:
        return out^
    var base = home if not home.endswith("/") else String(home[byte = 0 : home.byte_length() - 1])
    for n in materialize[_SENSITIVE_HOME_NAMES]():
        _home_deny_entry(base, n, out)
    var names: List[String]
    try:
        names = listdir(base)
    except:
        names = List[String]()
    sort(names)
    for n in names:
        if not n.startswith("."):
            continue
        var known = False
        for k in materialize[_SENSITIVE_HOME_NAMES]():
            if k == n:
                known = True
        for k in materialize[_HOME_DOT_ALLOWED]():
            if k == n:
                known = True
        if not known:
            _home_deny_entry(base, n, out)
    return out^


def modular_lib_dir(repo_root: String) raises -> String:
    """realpath of `<repo_root>/.venv/lib/python3.*/site-packages/modular/lib` (the
    runtime libraries a Mojo-built binary loads through its LC_RPATH/RUNPATH)."""
    var lib = repo_root + "/.venv/lib"
    var names: List[String]
    try:
        names = listdir(lib)
    except:
        names = List[String]()
    sort(names)
    for n in names:
        if not n.startswith("python3."):
            continue
        var cand = lib + "/" + n + "/site-packages/modular/lib"
        if _isdir(cand):
            return realpath_strict(cand)
    raise Error(
        _CONFINE_FAILED + "Mojo runtime libraries not found in " + lib
        + "/python3.*/site-packages/modular/lib. Hint: run make setup"
    )


# ---------------------------------------------------------------------------
# macOS profile (SBPL)
# ---------------------------------------------------------------------------
def _sbpl_string(path: String) raises -> String:
    """`"path"` with `\\` and `"` escaped; a control character raises QA-E163."""
    var out = String('"')
    var bs = path.as_bytes()
    for i in range(len(bs)):
        var c = Int(bs[i])
        if c < 0x20 or c == 0x7F:
            raise Error(_CONFINE_FAILED + "path with a control character: " + path.replace("\n", "\\n").replace("\r", "\\r"))
    for ch in path.codepoint_slices():
        var s = String(ch)
        if s == "\\" or s == '"':
            out += "\\"
        out += s
    out += '"'
    return out^


def macos_profile(c: Confinement) raises -> String:
    """The sandbox-exec (SBPL) profile for `c` (DESIGN.md 2.1.8):
    `(version 1)(allow default)(deny network*)(deny file-write*)(allow file-write*
    (literal "/dev/null") (subpath "<w>")...)`, then for CONF_TOOL `(deny
    process-fork)(deny file-read* (subpath "<home>"))(allow file-read* (subpath
    "<r>")...)`, for CONF_CHECK and CONF_BUILD `(deny file-read* (subpath "<p>")...)`
    over `sensitive_home_paths(home)`, then `(allow file-read* ...)` for the `write_dirs` and
    `read_allow` entries below `home`."""
    if c.profile != CONF_CHECK and c.profile != CONF_BUILD and c.profile != CONF_TOOL:
        raise Error(_CONFINE_FAILED + "unknown profile " + String(c.profile))
    var s = String("(version 1)(allow default)(deny network*)(deny file-write*)")
    s += '(allow file-write* (literal "/dev/null")'
    for w in c.write_dirs:
        s += " (subpath " + _sbpl_string(w) + ")"
    s += ")"
    if c.profile == CONF_TOOL:
        s += "(deny process-fork)"
        if c.home.byte_length() > 0:
            s += "(deny file-read* (subpath " + _sbpl_string(c.home) + "))"
            if len(c.read_allow) > 0:
                s += "(allow file-read*"
                for r in c.read_allow:
                    s += " (subpath " + _sbpl_string(r) + ")"
                s += ")"
    else:
        var paths = sensitive_home_paths(c.home)
        if len(paths) > 0:
            s += "(deny file-read*"
            for p in paths:
                s += " (subpath " + _sbpl_string(p) + ")"
            s += ")"
            # What the caller names (the sandbox, temp and cache dirs, the toolchain) stays
            # readable even when it lies inside a denied directory; the last rule wins.
            var reallow = List[String]()
            var prefix = c.home if c.home.endswith("/") else c.home + "/"
            for w in c.write_dirs:
                if w.startswith(prefix):
                    reallow.append(w)
            for r in c.read_allow:
                if r.startswith(prefix):
                    reallow.append(r)
            if len(reallow) > 0:
                s += "(allow file-read*"
                for r in reallow:
                    s += " (subpath " + _sbpl_string(r) + ")"
                s += ")"
    return s^


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------
def _sandbox_exec_present() -> Bool:
    var p = CStr(_SANDBOX_EXEC)
    var ok = c_access(p.addr(), X_OK) == 0
    p.keep()
    return ok


def _linux_quick_check() -> Bool:
    """Landlock ABI >= 1 and user namespaces not disabled (cheap, in-process)."""
    if c_syscall4(SYS_LANDLOCK_CREATE_RULESET, 0, 0, _LL_CREATE_RULESET_VERSION) < 1:
        return False
    try:
        with open("/proc/sys/user/max_user_namespaces", "r") as f:
            if String(f.read().strip()) == "0":
                return False
    except:
        pass
    return True


def sandbox_kind(self_binary: String) -> String:
    """"sandbox-exec" (macOS, /usr/bin/sandbox-exec is executable), "landlock" (Linux:
    `self_binary __confine <probe> /bin/true` exits 0, so user namespaces and Landlock
    both work) or "none". Each call probes; callers cache the answer."""
    if is_macos():
        return String("sandbox-exec") if _sandbox_exec_present() else String("none")
    if not _linux_quick_check():
        return String("none")
    var true_bin = which("true", "/usr/bin:/bin")
    if true_bin.byte_length() == 0:
        return String("none")
    var probe = Confinement(CONF_TOOL, List[String](), List[String](), String(), 0, 0)
    var argv: List[String] = [self_binary, "__confine", probe.to_json(), true_bin]
    var spec = SpawnSpec(self_binary, argv^, List[String](), String(), STDIO_NULL, STDIO_NULL, STDIO_NULL, True)
    try:
        var r = run_capture(spec, 10000, 4096, "")
        if r.exit_code == 0 and not r.timed_out:
            return String("landlock")
    except:
        pass
    return String("none")


def confine(spec: SpawnSpec, c: Confinement, self_binary: String) raises -> SpawnSpec:
    """`spec` wrapped in the trampoline: path = `self_binary`, argv = [self_binary,
    "__confine", c.to_json(), <spec.path resolved as spawn would>, spec.argv[1:]...];
    env, cwd, stdio modes and process group unchanged. Raises `QA-E163: no OS
    sandbox is available on this machine` when the cheap availability check fails
    (macOS: no sandbox-exec; Linux: no Landlock or user namespaces disabled), and the
    profile errors of `macos_profile` early on macOS."""
    if is_macos():
        if not _sandbox_exec_present():
            raise Error(_NO_SANDBOX)
        _ = macos_profile(c)
    elif not _linux_quick_check():
        raise Error(_NO_SANDBOX)
    var target = resolve_program(spec.path, spec.env, spec.cwd)
    var out = spec.copy()
    var argv = List[String]()
    argv.append(self_binary)
    argv.append("__confine")
    argv.append(c.to_json())
    argv.append(target)
    for i in range(1, len(spec.argv)):
        argv.append(spec.argv[i])
    out.path = self_binary
    out.argv = argv^
    return out^


# ---------------------------------------------------------------------------
# Trampoline
# ---------------------------------------------------------------------------
def _write_stderr(text: String):
    var bs = text.as_bytes()
    var p = Int(bs.unsafe_ptr())
    var n = len(bs)
    var off = 0
    while off < n:
        var w = c_write(2, p + off, n - off)
        if w < 0:
            if errno() == EINTR:
                continue
            break
        off += w
    _ = text.byte_length()


def _fail(detail: String) -> Int:
    _write_stderr(_CONFINE_FAILED + detail + "\n")
    return 125


def _errno_text(what: String, e: Int) -> String:
    return what + ": [Errno " + String(e) + "] " + strerror_text(e)


def _set_limit(resource: Int, value: Int) raises:
    var rl = List[UInt64](length=2, fill=0)
    rl[0] = UInt64(value)
    rl[1] = UInt64(value)
    var rc = c_setrlimit(resource, Int(rl.unsafe_ptr()))
    var e = errno()
    _ = len(rl)
    if rc != 0:
        raise Error(_errno_text("setrlimit(" + String(resource) + ", " + String(value) + ")", e))


def _landlock_rule(ruleset: Int, path: String, access: Int, handled: Int, optional: Bool) raises:
    var c = CStr(path)
    var fd = c_open(c.addr(), O_PATH | O_CLOEXEC, 0)
    var e = errno()
    c.keep()
    if fd < 0:
        if optional and e == ENOENT:
            return
        raise Error(_errno_text("open(" + path + ", O_PATH)", e))
    var allowed = access & handled
    if not _isdir(path):
        allowed &= _LL_FILE_RIGHTS
    # struct landlock_path_beneath_attr (packed): u64 allowed_access; s32 parent_fd
    var attr = List[UInt64](length=2, fill=0)
    var a = Int(attr.unsafe_ptr())
    store_i64(a, allowed)
    store_u32(a + 8, fd)
    var rc = c_syscall4(SYS_LANDLOCK_ADD_RULE, ruleset, _LL_RULE_PATH_BENEATH, a)
    var e2 = errno()
    _ = len(attr)
    _ = c_close(fd)
    if rc != 0:
        raise Error(_errno_text("landlock_add_rule(" + path + ")", e2))


def _linux_confine(c: Confinement) raises:
    if c_unshare(CLONE_NEWUSER | CLONE_NEWNET) != 0:
        raise Error(_errno_text("unshare(CLONE_NEWUSER | CLONE_NEWNET)", errno()))
    if c_prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise Error(_errno_text("prctl(PR_SET_NO_NEW_PRIVS)", errno()))
    var abi = c_syscall4(SYS_LANDLOCK_CREATE_RULESET, 0, 0, _LL_CREATE_RULESET_VERSION)
    if abi < 1:
        raise Error("Landlock is not available on this kernel")
    var handled = (1 << 13) - 1   # ABI 1: EXECUTE .. MAKE_SYM
    if abi >= 2:
        handled |= _LL_REFER
    if abi >= 3:
        handled |= _LL_TRUNCATE
    if abi >= 5:
        handled |= _LL_IOCTL_DEV
    # struct landlock_ruleset_attr: only handled_access_fs (size 8); the network is
    # cut by the new network namespace instead.
    var attr = List[UInt64](length=3, fill=0)
    attr[0] = UInt64(handled)
    var ruleset = c_syscall4(SYS_LANDLOCK_CREATE_RULESET, Int(attr.unsafe_ptr()), 8, 0)
    var e = errno()
    _ = len(attr)
    if ruleset < 0:
        raise Error(_errno_text("landlock_create_ruleset", e))
    var rx = _LL_EXECUTE | _LL_READ_FILE | _LL_READ_DIR
    var reads = List[String]()
    if c.profile == CONF_TOOL:
        var base: List[String] = ["/usr", "/lib", "/lib64", "/bin", "/etc", "/proc", "/sys/devices/system/cpu", "/dev/urandom", "/dev/random"]
        for b in base:
            reads.append(b)
        for r in c.read_allow:
            reads.append(r)
    else:
        reads.append("/")
    for r in reads:
        _landlock_rule(ruleset, r, rx, handled, True)
    for w in c.write_dirs:
        _landlock_rule(ruleset, w, handled, handled, False)
    _landlock_rule(ruleset, "/dev/null", handled, handled, False)
    if c_syscall4(SYS_LANDLOCK_RESTRICT_SELF, ruleset, 0, 0) != 0:
        var e3 = errno()
        _ = c_close(ruleset)
        raise Error(_errno_text("landlock_restrict_self", e3))
    _ = c_close(ruleset)


def wait_and_mirror(pid: Int) -> Int:
    """Waits for the forked child `pid` and ends like it did: returns its exit code, or
    when it was killed by a signal sends that signal to this process (then, should it
    survive, returns 128 + signal), so the caller of the trampoline sees the same wait
    status as with a direct exec (an RLIMIT_CPU kill is `-SIGKILL`, not 125)."""
    var st = List[UInt8](length=8, fill=0)
    while True:
        var r = c_waitpid(pid, buf_addr(st), 0)
        if r == pid:
            break
        if r < 0 and errno() != EINTR:
            _ = len(st)
            return _fail(_errno_text("waitpid", errno()))
    var status = load_i32(buf_addr(st))
    _ = len(st)
    var sig = status & 0x7F
    if sig == 0:
        return (status >> 8) & 0xFF
    _ = c_kill(c_getpid(), sig)
    return 128 + sig


def _linux_exec_confined(conf: Confinement, path: String, rest: List[String]) -> Int:
    """Linux: runs in a freshly forked, single-threaded child (see `confine_main`):
    rlimits, user and network namespaces, Landlock, then execve. Never returns on
    success; returns 125 after printing QA-E163."""
    try:
        if conf.cpu_s > 0:
            _set_limit(RLIMIT_CPU, conf.cpu_s)
        if conf.fsize_bytes > 0:
            _set_limit(RLIMIT_FSIZE, conf.fsize_bytes)
        _linux_confine(conf)
    except e:
        return _fail(String(e))
    var env = CStrArray(environ_list())
    var argv = List[String]()
    argv.append(path)
    for r in rest:
        argv.append(r)
    var p = CStr(path)
    var av = CStrArray(argv)
    _ = c_execve(p.addr(), av.addr(), env.addr())
    var e = errno()
    p.keep()
    av.keep()
    env.keep()
    return _fail(_errno_text("execve " + path, e))


def confine_main(args: List[String]) -> Int:
    """Body of `qumo-agent __confine CONF_JSON PATH ARGV...`. `args` is the full argv
    (`args[1] == "__confine"`) or starts at "__confine". Applies the rlimits and the
    OS sandbox, then execs PATH with argv [PATH, ARGV...] and this process's
    environment. Returns only on failure: 125, after printing
    `QA-E163: OS confinement failed: <detail>` on stderr. On Linux the work happens in
    a forked child and the return value mirrors the child's exit status."""
    var k = -1
    if len(args) > 1 and args[1] == "__confine":
        k = 1
    elif len(args) > 0 and args[0] == "__confine":
        k = 0
    if k < 0 or len(args) < k + 3:
        return _fail("usage: __confine CONF_JSON PATH [ARG...]")
    var conf: Confinement
    try:
        conf = confinement_from_json(args[k + 1])
    except e:
        var t = String(e)
        if t.startswith(_CONFINE_FAILED):
            return _fail(String(t[byte = _CONFINE_FAILED.byte_length() :]))
        return _fail("malformed confinement: " + t)
    var path = args[k + 2]
    var rest = List[String]()
    for i in range(k + 3, len(args)):
        rest.append(args[i])
    if not is_macos():
        # unshare(CLONE_NEWUSER) fails with EINVAL in a multithreaded process, and a Mojo
        # binary starts runtime threads before main. A forked child has only the calling
        # thread, so the namespaces, Landlock and the exec happen there; this process
        # waits and mirrors the child's status (the harness kills the whole process group).
        var child = c_fork()
        if child < 0:
            return _fail(_errno_text("fork", errno()))
        if child == 0:
            c_exit_now(_linux_exec_confined(conf, path, rest))
        return wait_and_mirror(child)
    try:
        if conf.cpu_s > 0:
            _set_limit(RLIMIT_CPU, conf.cpu_s)
        if conf.fsize_bytes > 0:
            _set_limit(RLIMIT_FSIZE, conf.fsize_bytes)
    except e:
        return _fail(String(e))
    var env = CStrArray(environ_list())
    var profile: String
    try:
        profile = macos_profile(conf)
    except e:
        var t = String(e)
        if t.startswith(_CONFINE_FAILED):
            return _fail(String(t[byte = _CONFINE_FAILED.byte_length() :]))
        return _fail(t)
    var argv: List[String] = ["sandbox-exec", "-p", profile, path]
    for r in rest:
        argv.append(r)
    var p = CStr(_SANDBOX_EXEC)
    var av = CStrArray(argv)
    _ = c_execve(p.addr(), av.addr(), env.addr())
    var e = errno()
    p.keep()
    av.keep()
    env.keep()
    return _fail(_errno_text("execve " + String(_SANDBOX_EXEC), e))
