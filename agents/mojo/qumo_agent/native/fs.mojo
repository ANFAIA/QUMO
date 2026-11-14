"""File system helpers with Python's semantics and error texts (DESIGN.md 2.1.4).

Errors raise `QA-E160: [Errno N] <strerror>: '<path>'`, the text Python's OSError
prints (`py_oserror`); callers that show Python's text drop the `QA-E160: ` prefix.
Path functions (`normpath`, `join_path`, `resolve_nonstrict`) follow `os.path` and
`pathlib` on POSIX.

Regular files are read and written through the standard library's `open()`
(it honours the umask); directories, links and stat go through `std.os`; the libc
calls the standard library does not offer (realpath, mkdtemp, readlink, chmod,
rename) go through `native/libc.mojo`.
"""

from std.os import listdir, mkdir, makedirs, rmdir, unlink, stat, lstat
from std.os import symlink as _std_symlink
from std.os.path import isdir as _isdir, isfile as _isfile, islink as _islink, lexists as _lexists
from std.os import getenv
from std.pathlib import cwd

from qumo_agent.native.libc import (
    CStr,
    PATH_BUF_SIZE,
    ENOENT,
    EISDIR,
    ENOTDIR,
    EEXIST,
    buf_addr,
    errno,
    strerror_text,
    c_string_at,
    c_realpath,
    c_mkdtemp,
    c_chmod,
    c_rename,
    c_readlink,
)
from qumo_agent.native.rand import random_bytes
from qumo_agent.util.pyfmt import py_str_repr


comptime _S_IFMT = 0o170000
comptime _S_IFDIR = 0o040000
comptime _S_IFLNK = 0o120000


def py_oserror(code: Int, path: String) -> String:
    """Python's OSError text: `[Errno 2] No such file or directory: 'x'`."""
    return "[Errno " + String(code) + "] " + strerror_text(code) + ": " + py_str_repr(path)


def _err(code: Int, path: String) -> Error:
    return Error("QA-E160: " + py_oserror(code, path))


def _errno_or(default: Int) -> Int:
    var e = errno()
    if e <= 0:
        return default
    return e


# ---------------------------------------------------------------------------
# Reading and writing
# ---------------------------------------------------------------------------
def read_bytes(path: String) raises -> List[UInt8]:
    if _isdir(path):
        raise _err(EISDIR, path)
    try:
        with open(path, "r") as f:
            return f.read_bytes()
    except:
        raise _err(_errno_or(ENOENT), path)


def read_text_lossy(path: String) raises -> String:
    """Text with invalid UTF-8 replaced by U+FFFD (Python `errors="replace"`)."""
    var b = read_bytes(path)
    return String(from_utf8_lossy=Span(b))


def _write_bytes(path: String, data: Span[Byte, _]) raises:
    if _isdir(path):
        raise _err(EISDIR, path)
    try:
        with open(path, "w") as f:
            f.write_bytes(data)
    except:
        raise _err(_errno_or(ENOENT), path)


def write_text(path: String, text: String) raises:
    _write_bytes(path, text.as_bytes())


def _dirname(path: String) -> String:
    """os.path.dirname (POSIX)."""
    var i = path.rfind("/")
    if i < 0:
        return String()
    var head = String(path[byte=0 : i + 1])
    # Strip trailing slashes unless head is all slashes.
    var hb = head.as_bytes()
    var all_slash = True
    for k in range(len(hb)):
        if Int(hb[k]) != 0x2F:
            all_slash = False
    if all_slash:
        return head^
    var end = len(hb)
    while end > 0 and Int(hb[end - 1]) == 0x2F:
        end -= 1
    return String(head[byte=0:end])


def _basename(path: String) -> String:
    var i = path.rfind("/")
    if i < 0:
        return path
    return String(path[byte = i + 1 :])


def _rand_suffix() -> String:
    try:
        var b = random_bytes(6)
        var out = String()
        var hexd = String("0123456789abcdef")
        for i in range(len(b)):
            out += String(hexd[byte = Int(b[i]) >> 4 : (Int(b[i]) >> 4) + 1])
            out += String(hexd[byte = Int(b[i]) & 15 : (Int(b[i]) & 15) + 1])
        return out^
    except:
        return String("tmp")


def write_text_atomic(path: String, text: String) raises:
    """Writes a temporary file in the same directory, then renames it over `path`."""
    var d = _dirname(path)
    var tmp = (d + "/" if d.byte_length() > 0 else String()) + "." + _basename(path) + ".tmp-" + _rand_suffix()
    write_text(tmp, text)
    var a = CStr(tmp)
    var b = CStr(path)
    var rc = c_rename(a.addr(), b.addr())
    var e = errno()
    a.keep()
    b.keep()
    if rc != 0:
        try:
            unlink(tmp)
        except:
            pass
        raise _err(e, path)


def copy_file(src: String, dst: String, mode: Int) raises:
    """Copies the bytes of `src` to `dst`, then sets `dst`'s permission bits to `mode`."""
    var data = read_bytes(src)
    _write_bytes(dst, Span(data))
    chmod(dst, mode)


# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
def make_dirs(path: String) raises:
    """`os.makedirs(path, exist_ok=True)`."""
    if _isdir(path):
        return
    try:
        makedirs(path, exist_ok=True)
    except:
        if _isdir(path):
            return
        raise _err(_errno_or(EEXIST), path)


def remove_tree(path: String) raises:
    """`shutil.rmtree(path)`, never following symlinks: a symlink (to a file or a
    directory) is unlinked, not traversed. Raises when `path` does not exist."""
    if not _lexists(path):
        raise _err(ENOENT, path)
    if _islink(path) or not _isdir(path):
        try:
            unlink(path)
        except:
            raise _err(_errno_or(ENOENT), path)
        return
    var names: List[String]
    try:
        names = listdir(path)
    except:
        raise _err(_errno_or(ENOENT), path)
    for name in names:
        remove_tree(path + "/" + name)
    try:
        rmdir(path)
    except:
        raise _err(_errno_or(ENOENT), path)


def make_temp_dir(prefix: String) raises -> String:
    """`mkdtemp` under `$TMPDIR` (or `/tmp`): `<dir>/<prefix>XXXXXX`, mode 0700."""
    var base = getenv("TMPDIR", "")
    if base.byte_length() == 0:
        base = "/tmp"
    while base.byte_length() > 1 and base.endswith("/"):
        var base_cut = String(base[byte = 0 : base.byte_length() - 1])
        base = base_cut^
    var tmpl = CStr(base + "/" + prefix + "XXXXXX")
    var r = c_mkdtemp(tmpl.addr())
    var e = errno()
    var out = c_string_at(r)
    tmpl.keep()
    if r == 0:
        raise _err(e, base + "/" + prefix + "XXXXXX")
    return out^


def list_dir(path: String) raises -> List[String]:
    """Entry names, unsorted, without "." and ".."."""
    if not _lexists(path):
        raise _err(ENOENT, path)
    if not _isdir(path):
        raise _err(ENOTDIR, path)
    try:
        var names = listdir(path)
        var out = List[String](capacity=len(names))
        for n in names:
            if n != "." and n != "..":
                out.append(n)
        return out^
    except:
        raise _err(_errno_or(ENOENT), path)


def _abs(path: String) raises -> String:
    if path.startswith("/"):
        return path
    var here = String(cwd())
    if path.byte_length() == 0:
        return here^
    return here + "/" + path


def walk(base: String, follow_dir_symlinks: Bool) raises -> List[String]:
    """Every entry below `base` as an absolute path, depth first, each directory
    before its children. Symlinked directories are listed but not entered unless
    `follow_dir_symlinks`; a directory (by realpath) is never entered twice, so
    symlink loops end. Unreadable subdirectories are skipped, as `os.walk` does."""
    var root = _abs(base)
    while root.byte_length() > 1 and root.endswith("/"):
        var root_cut = String(root[byte = 0 : root.byte_length() - 1])
        root = root_cut^
    var out = List[String]()
    var visited = List[String]()
    try:
        visited.append(realpath_strict(root))
    except:
        pass
    # Stack of directories whose entries are still to be emitted: (dir, names, next index).
    var dirs = List[String]()
    var listings = List[List[String]]()
    var next = List[Int]()
    dirs.append(root)
    listings.append(list_dir(root))
    next.append(0)
    while len(dirs) > 0:
        var top = len(dirs) - 1
        if next[top] >= len(listings[top]):
            _ = dirs.pop()
            _ = listings.pop()
            _ = next.pop()
            continue
        var name = listings[top][next[top]]
        next[top] += 1
        var d = dirs[top]
        var p = d + "/" + name if d != "/" else "/" + name
        out.append(p)
        if not _isdir(p):
            continue
        if _islink(p) and not follow_dir_symlinks:
            continue
        var real: String
        try:
            real = realpath_strict(p)
        except:
            continue
        var seen = False
        for v in visited:
            if v == real:
                seen = True
                break
        if seen:
            continue
        var sub: List[String]
        try:
            sub = list_dir(p)
        except:
            continue
        visited.append(real)
        dirs.append(p)
        listings.append(sub^)
        next.append(0)
    return out^


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def realpath_strict(path: String) raises -> String:
    """libc realpath(3): every component must exist."""
    var c = CStr(path)
    var buf = List[UInt8](length=PATH_BUF_SIZE, fill=0)
    var r = c_realpath(c.addr(), buf_addr(buf))
    var e = errno()
    c.keep()
    if r == 0:
        _ = len(buf)
        raise _err(_errno_or(ENOENT) if e <= 0 else e, path)
    var out = c_string_at(buf_addr(buf))
    _ = len(buf)
    return out^


def _readlink(path: String) -> Optional[String]:
    var c = CStr(path)
    var buf = List[UInt8](length=PATH_BUF_SIZE, fill=0)
    var n = c_readlink(c.addr(), buf_addr(buf), PATH_BUF_SIZE)
    c.keep()
    if n < 0:
        _ = len(buf)
        return None
    var out = String(from_utf8_lossy=Span(buf)[0:n])
    _ = len(buf)
    return out^


def _py_join(a: String, b: String) -> String:
    """posixpath.join(a, b)."""
    if b.startswith("/"):
        return b
    if a.byte_length() == 0 or a.endswith("/"):
        return a + b
    return a + "/" + b


def _py_split(path: String) -> Tuple[String, String]:
    """posixpath.split."""
    var i = path.rfind("/") + 1
    var head = String(path[byte=0:i])
    var tail = String(path[byte=i:])
    var hb = head.as_bytes()
    var all_slash = True
    for k in range(len(hb)):
        if Int(hb[k]) != 0x2F:
            all_slash = False
    if head.byte_length() > 0 and not all_slash:
        var end = len(hb)
        while end > 0 and Int(hb[end - 1]) == 0x2F:
            end -= 1
        var head_cut = String(head[byte=0:end])
        head = head_cut^
    return (head^, tail^)


@fieldwise_init
struct _Seen(Copyable):
    var key: String
    var resolved: Optional[String]


def _seen_find(seen: List[_Seen], key: String) -> Int:
    for i in range(len(seen)):
        if seen[i].key == key:
            return i
    return -1


def _join_realpath(var path: String, var rest: String, mut seen: List[_Seen]) -> Tuple[String, Bool]:
    """posixpath._joinrealpath(path, rest, strict=False, seen) of Python 3.12."""
    if rest.startswith("/"):
        var rest_cut = String(rest[byte=1:])
        rest = rest_cut^
        path = "/"
    while rest.byte_length() > 0:
        var i = rest.find("/")
        var name: String
        if i < 0:
            name = rest
            rest = String()
        else:
            name = String(rest[byte=0:i])
            var r2 = String(rest[byte = i + 1 :])
            rest = r2^
        if name.byte_length() == 0 or name == ".":
            continue
        if name == "..":
            if path.byte_length() > 0:
                var sp = _py_split(path)
                path = sp[0]
                if sp[1] == "..":
                    path = _py_join(_py_join(path, ".."), "..")
            else:
                path = ".."
            continue
        var newpath = _py_join(path, name)
        if not _islink(newpath):
            path = newpath
            continue
        var idx = _seen_find(seen, newpath)
        if idx >= 0:
            if seen[idx].resolved:
                path = seen[idx].resolved.value()
                continue
            return (_py_join(newpath, rest), False)
        seen.append(_Seen(newpath, None))
        var target = _readlink(newpath)
        if not target:
            path = newpath
            continue
        var res = _join_realpath(path, target.value(), seen)
        path = res[0]
        if not res[1]:
            return (_py_join(path, rest), False)
        var j = _seen_find(seen, newpath)
        if j >= 0:
            seen[j].resolved = path
    return (path^, True)


def resolve_nonstrict(path: String) -> String:
    """`str(pathlib.Path(path).resolve(strict=False))` (Python 3.12): symlinks are
    resolved component by component; missing components are kept lexically."""
    var absolute: String
    try:
        absolute = _abs(path)
    except:
        absolute = path
    var seen = List[_Seen]()
    var res = _join_realpath(String(), absolute, seen)
    return _pathlib_str(normpath(res[0]))


def normpath(path: String) -> String:
    """os.path.normpath (POSIX): keeps a leading "//" (but not "///")."""
    if path.byte_length() == 0:
        return String(".")
    var initial = 0
    if path.startswith("/"):
        initial = 1
        if path.startswith("//") and not path.startswith("///"):
            initial = 2
    var comps = List[String]()
    for c in path.split("/"):
        var comp = String(c)
        if comp.byte_length() == 0 or comp == ".":
            continue
        if comp != ".." or (initial == 0 and len(comps) == 0) or (len(comps) > 0 and comps[len(comps) - 1] == ".."):
            comps.append(comp)
        elif len(comps) > 0:
            _ = comps.pop()
    var out = String()
    for _ in range(initial):
        out += "/"
    for i in range(len(comps)):
        if i > 0:
            out += "/"
        out += comps[i]
    if out.byte_length() == 0:
        return String(".")
    return out^


def _pathlib_str(path: String) -> String:
    """str(PurePosixPath(path)): collapses repeated "/" and "." parts, drops a
    trailing "/", keeps exactly two leading slashes ("//a"), "" becomes "."."""
    var root = String()
    if path.startswith("//") and not path.startswith("///"):
        root = "//"
    elif path.startswith("/"):
        root = "/"
    var parts = List[String]()
    for c in path.split("/"):
        var comp = String(c)
        if comp.byte_length() == 0 or comp == ".":
            continue
        parts.append(comp)
    var out = root
    for i in range(len(parts)):
        if i > 0:
            out += "/"
        out += parts[i]
    if out.byte_length() == 0:
        return String(".")
    return out^


def join_path(a: String, b: String) -> String:
    """`str(pathlib.PurePosixPath(a) / b)`: an absolute `b` replaces `a`."""
    if b.startswith("/"):
        return _pathlib_str(b)
    if a.byte_length() == 0:
        return _pathlib_str(b)
    if a.endswith("/"):
        return _pathlib_str(a + b)
    return _pathlib_str(a + "/" + b)


def exists(path: String) -> Bool:
    """Follows symlinks (a dangling link does not exist), as os.path.exists."""
    try:
        _ = stat(path)
        return True
    except:
        return False


def is_dir(path: String) -> Bool:
    return _isdir(path)


def is_file(path: String) -> Bool:
    return _isfile(path)


def is_symlink(path: String) -> Bool:
    return _islink(path)


def symlink(target: String, link: String) raises:
    """`os.symlink(target, link)`; errors read `[Errno 17] File exists: 'target' -> 'link'`."""
    try:
        _std_symlink(target, link)
    except:
        var code = _errno_or(EEXIST)
        raise Error(
            "QA-E160: [Errno " + String(code) + "] " + strerror_text(code) + ": "
            + py_str_repr(target) + " -> " + py_str_repr(link)
        )


def file_mode(path: String) raises -> Int:
    """Permission bits (`st_mode & 0o7777`), following symlinks."""
    try:
        return Int(stat(path).st_mode) & 0o7777
    except:
        raise _err(_errno_or(ENOENT), path)


def chmod(path: String, mode: Int) raises:
    var c = CStr(path)
    var rc = c_chmod(c.addr(), mode)
    var e = errno()
    c.keep()
    if rc != 0:
        raise _err(e, path)
