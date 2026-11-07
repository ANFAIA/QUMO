"""Model container writer (qumo.rlsynth.v0 / qumo.rlroute.v0), atomic directory swap, INDEX writer, verifier.

Implements MOJO_PORT.md 9.1 (container, atomic writes), 9.4 (verification) and 9.5 (INDEX writer).

Container layout (byte exact, shared with the legacy exporter so the 10 shipped models re-encode identically):

* `<dir>/weights.bin` is the concatenation of the tensors in manifest order. The buffer is zero padded to a multiple
  of 64 bytes before every tensor and after the last one; `offset` is the padded start; data is little endian,
  row-major. A tensor is stored as f16 unless the global dtype is `f32`, the tensor is marked `f32`, its name is in
  `f32_names` or its name ends in `.bias` (biases are always f32). The writer raises QM-E645 when a tensor does not
  survive the f16 conversion as a finite number.
* `<dir>/manifest.json` is compact JSON: the caller's `manifest_head` members, then `tensors` (name, shape, dtype,
  offset), `weights_file`, `weights_bytes`, `fnv1a64` (FNV-1a 64 over the whole file including pad bytes, 16 lowercase
  hex digits), `sha256` (64 hex digits, same bytes), then the caller's `manifest_tail` members.

Crash safety: a writer never touches the final directory in place. It writes `<dir>.tmp`, reads it back with
`load_model_file` and compares every decoded tensor with what it encoded (QM-E644 on any mismatch), and only then
swaps: `<dir>` is renamed to `<dir>.old`, `<dir>.tmp` is renamed to `<dir>`, `<dir>.old` is removed. `std.os` has no
`rename` in Mojo 1.2, so the swap uses the POSIX libc symbols `rename`, `unlink`, `fsync` and `sync` through
`external_call` (the only libc calls of the RL stack). `recover_dir` runs at the start of every tool that reads or
writes a model or run directory: it restores `<dir>.old` when `<dir>` is missing and deletes stale `.tmp` / `.old`
leftovers, so a kill or a power loss leaves either the complete old or the complete new directory.

Error codes: QM-E640 usage, QM-E644 container write / read-back mismatch, QM-E645 f16 overflow.
"""

from std.collections import List, Dict
from std.ffi import external_call, c_int
from std.memory import bitcast
from std.os import listdir, makedirs, remove, rmdir
from std.os.path import exists, isdir, isfile, getsize
from qumo.ai.json import JsonValue, JsonKind, parse_json, json_quote
from qumo.compiler.policy import (
    load_model_file,
    fnv1a64_bytes,
    hex16,
    wl_hash,
    f16_to_f32,
)
from qumo.rl.digest import sha256_hex, sha256_file_hex
from qumo.rl.f16 import encode_f16_span, f32_to_f16_bits

comptime ALIGN = 64
comptime WEIGHTS_FILE = "weights.bin"
comptime MANIFEST_FILE = "manifest.json"
comptime INDEX_FILE = "INDEX.json"


@fieldwise_init
struct TensorOut(Copyable):
    var name: String
    var shape: List[Int]
    var data: List[Float32]
    var dtype: String  # "f16" | "f32"; "f32" forces f32 storage for this tensor, "f16" leaves the default rules


@fieldwise_init
struct ModelDirInfo(Copyable):
    var weights_bytes: Int
    var fnv1a64: String
    var sha256: String
    var manifest_bytes: Int


struct VerifyReport(Copyable):
    """Outcome of `verify_models`: the counters of the old make target plus its text lines (`FAIL ...` lines in the
    order found, then the final `models-verify: ...` summary line, which is always the last element)."""

    var n_index: Int  # INDEX.json files found
    var n_files: Int  # files whose sha256 matched
    var n_problems: Int
    var total_bytes: Int  # sum of every file size under the root
    var lines: List[String]

    def __init__(out self):
        self.n_index = 0
        self.n_files = 0
        self.n_problems = 0
        self.total_bytes = 0
        self.lines = List[String]()


@fieldwise_init
struct ManifestParts(Copyable):
    """A manifest split around the members the writer owns (see `split_manifest`)."""

    var head: String  # members before "tensors", verbatim, comma separated, no braces
    var tail: String  # members after "tensors" other than the writer's own, verbatim ("" if none)


def _e644(what: String) -> Error:
    return Error("QM-E644: " + what + ". Hint: re-run the export; the previous model directory was left untouched.")


def _e640(what: String) -> Error:
    return Error("QM-E640: " + what)


# --------------------------------------------------------------------------------------------- libc and paths


def _rename(src: String, dst: String) -> Int:
    var s = src.copy()
    var d = dst.copy()
    var rc = external_call["rename", c_int](s.as_c_string_span().ptr(), d.as_c_string_span().ptr())
    return Int(rc)


def _unlink(path: String) -> Int:
    var s = path.copy()
    var rc = external_call["unlink", c_int](s.as_c_string_span().ptr())
    return Int(rc)


def _fsync_file(path: String):
    """Best-effort fsync of a regular file (flushes the file's pages whichever descriptor is used)."""
    try:
        with open(path, "r") as f:
            _ = external_call["fsync", c_int](f._get_raw_fd())
    except:
        pass


def _sync_all():
    """POSIX `sync()`: makes the directory entries of a rename durable (no portable per-directory fsync without
    libc `open`, which the Mojo runtime already declares with another signature)."""
    external_call["sync", NoneType]()


def _strip_slashes(path: String) -> String:
    var bs = path.as_bytes()
    var n = len(bs)
    while n > 1 and Int(bs[n - 1]) == ord("/"):
        n -= 1
    return String(unsafe_from_utf8=bs[0:n])


def _parent_dir(path: String) -> String:
    """Directory part of `path` ("." when there is none)."""
    var bs = path.as_bytes()
    var i = len(bs) - 1
    while i >= 0 and Int(bs[i]) != ord("/"):
        i -= 1
    if i < 0:
        return String(".")
    if i == 0:
        return String("/")
    return String(unsafe_from_utf8=bs[0:i])


def _join(a: String, b: String) -> String:
    """`os.path.join` semantics for two components."""
    if b.byte_length() > 0 and Int(b.as_bytes()[0]) == ord("/"):
        return b.copy()
    if a.byte_length() == 0:
        return b.copy()
    if Int(a.as_bytes()[a.byte_length() - 1]) == ord("/"):
        return a + b
    return a + "/" + b


def _normpath(path: String) -> String:
    """`os.path.normpath` for POSIX paths (collapses `//`, `.` and `..`)."""
    if path.byte_length() == 0:
        return String(".")
    var absolute = Int(path.as_bytes()[0]) == ord("/")
    var out = List[String]()
    for part in path.split("/"):
        var p = String(part)
        if p.byte_length() == 0 or p == ".":
            continue
        if p == "..":
            if len(out) > 0 and out[len(out) - 1] != "..":
                _ = out.pop()
                continue
            if absolute:
                continue
        out.append(p)
    var s = String()
    for i in range(len(out)):
        if i > 0:
            s += "/"
        s += out[i]
    if absolute:
        return "/" + s
    if s.byte_length() == 0:
        return String(".")
    return s^


def _rmtree(path: String) raises:
    """Removes a file, a symlink (never followed) or a directory tree. Only called on `.tmp` / `.old` leftovers."""
    if not exists(path):
        return
    if _unlink(path) == 0:
        return  # a file or a symlink; unlink fails on a directory
    if isdir(path):
        for n in listdir(path):
            _rmtree(path + "/" + n)
        rmdir(path)
    else:
        remove(path)


def remove_tree(path: String) raises:
    """Deletes a file, a symlink (not followed) or a whole directory tree; a missing path is fine. Training tools
    use it for staging directories and tests for scratch space; callers must pass a path they own."""
    _rmtree(path)


def _write_durable(path: String, data: Span[UInt8, _]) raises:
    with open(path, "w") as f:
        f.write_bytes(data)
    _fsync_file(path)


def write_text_atomic(path: String, text: String) raises:
    """Writes `text` to `<path>.tmp` and renames it over `path` (rename is atomic on POSIX)."""
    var tmp = path + ".tmp"
    _write_durable(tmp, text.as_bytes())
    if _rename(tmp, path) != 0:
        raise _e644("cannot rename '" + tmp + "' to '" + path + "'")


def _read_text(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


# ------------------------------------------------------------------------------------------- atomic directories


def atomic_replace_dir_stop(tmp: String, final: String, stop_after: Int) raises:
    """`atomic_replace_dir` that halts after step `stop_after` (1: `<final>` renamed to `<final>.old`; 2: `<tmp>`
    renamed to `<final>`; 3 or more: complete). It exists so tests can simulate a kill between the rename steps."""
    var f = _strip_slashes(final)
    var t = _strip_slashes(tmp)
    var old = f + ".old"
    if not isdir(t):
        raise _e644("staging directory '" + t + "' does not exist")
    if exists(old):
        _rmtree(old)
    var had_final = exists(f)
    if had_final:
        if _rename(f, old) != 0:
            raise _e644("cannot move '" + f + "' to '" + old + "'")
    if stop_after == 1:
        return
    if _rename(t, f) != 0:
        if had_final:
            _ = _rename(old, f)
        raise _e644("cannot move '" + t + "' to '" + f + "'")
    if stop_after == 2:
        return
    _sync_all()
    if had_final:
        _rmtree(old)


def atomic_replace_dir(tmp: String, final: String) raises:
    """Swaps the complete directory `tmp` into `final`: `final` -> `final.old`, `tmp` -> `final`, drop `final.old`.
    A stop between the renames is repaired by `recover_dir`."""
    atomic_replace_dir_stop(tmp, final, 3)


def recover_dir(final: String) raises:
    """Run at startup by every tool that reads or writes a model or run directory. Restores `<final>.old` when
    `<final>` is missing, drops a stale `<final>.old` when `<final>` exists, and deletes a stale `<final>.tmp`."""
    var f = _strip_slashes(final)
    var old = f + ".old"
    var tmp = f + ".tmp"
    if exists(old):
        if exists(f):
            _rmtree(old)
        elif _rename(old, f) != 0:
            raise _e644("cannot restore '" + old + "' to '" + f + "'")
    if exists(tmp):
        _rmtree(tmp)


# --------------------------------------------------------------------------------------------- manifest splitting


def _skip_ws(bs: Span[Byte, _], mut pos: Int):
    while pos < len(bs):
        var c = Int(bs[pos])
        if c == 0x20 or c == 0x09 or c == 0x0A or c == 0x0D:
            pos += 1
        else:
            break


def _scan_string(bs: Span[Byte, _], mut pos: Int) raises:
    """`pos` is at the opening quote; leaves it after the closing quote."""
    pos += 1
    while pos < len(bs):
        var c = Int(bs[pos])
        if c == 0x5C:
            pos += 2
            continue
        pos += 1
        if c == 0x22:
            return
    raise _e644("manifest has an unterminated string")


def _scan_value(bs: Span[Byte, _], mut pos: Int) raises:
    """Skips one JSON value starting at `pos` (string, container or literal)."""
    if pos >= len(bs):
        raise _e644("manifest ends where a value was expected")
    var c = Int(bs[pos])
    if c == 0x22:
        _scan_string(bs, pos)
        return
    if c == ord("{") or c == ord("["):
        var depth = 0
        while pos < len(bs):
            var d = Int(bs[pos])
            if d == 0x22:
                _scan_string(bs, pos)
                continue
            if d == ord("{") or d == ord("["):
                depth += 1
            elif d == ord("}") or d == ord("]"):
                depth -= 1
                if depth == 0:
                    pos += 1
                    return
            pos += 1
        raise _e644("manifest has an unterminated container")
    while pos < len(bs):
        var d = Int(bs[pos])
        if d == ord(",") or d == ord("}") or d == ord("]") or d == 0x20 or d == 0x0A or d == 0x0D or d == 0x09:
            return
        pos += 1


def _is_writer_member(key: String) -> Bool:
    return key == "tensors" or key == "weights_file" or key == "weights_bytes" or key == "fnv1a64" or key == "sha256"


def split_manifest(text: String) raises -> ManifestParts:
    """Splits a compact manifest into the `manifest_head` / `manifest_tail` the writer takes: the top-level members
    before `tensors` and the members after it, excluding the writer's own (`tensors`, `weights_file`,
    `weights_bytes`, `fnv1a64`, `sha256`). Member text is kept byte for byte, so
    `write_model_dir(split_manifest(manifest))` reproduces the manifest exactly. Used to rewrite the tail (test
    vectors, rollouts) of an exported model."""
    var bs = text.as_bytes()
    var pos = 0
    _skip_ws(bs, pos)
    if pos >= len(bs) or Int(bs[pos]) != ord("{"):
        raise _e644("manifest is not a JSON object")
    pos += 1
    var head = String()
    var tail = String()
    var seen_tensors = False
    while True:
        _skip_ws(bs, pos)
        if pos >= len(bs):
            raise _e644("manifest object is not closed")
        if Int(bs[pos]) == ord("}"):
            break
        if Int(bs[pos]) == ord(","):
            pos += 1
            continue
        if Int(bs[pos]) != 0x22:
            raise _e644("manifest member does not start with a key")
        var start = pos
        _scan_string(bs, pos)
        var key = String(unsafe_from_utf8=bs[start + 1 : pos - 1])
        _skip_ws(bs, pos)
        if pos >= len(bs) or Int(bs[pos]) != ord(":"):
            raise _e644("manifest member '" + key + "' has no ':'")
        pos += 1
        _skip_ws(bs, pos)
        _scan_value(bs, pos)
        var member = String(unsafe_from_utf8=bs[start:pos])
        if key == "tensors":
            seen_tensors = True
        if _is_writer_member(key):
            continue
        if seen_tensors:
            if tail.byte_length() > 0:
                tail += ","
            tail += member
        else:
            if head.byte_length() > 0:
                head += ","
            head += member
    return ManifestParts(head^, tail^)


# ----------------------------------------------------------------------------------------------------- writer


def _is_f32_tensor(t: TensorOut, dtype: String, f32_names: List[String]) -> Bool:
    if dtype == "f32" or t.dtype == "f32" or t.name.endswith(".bias"):
        return True
    for n in f32_names:
        if n == t.name:
            return True
    return False


def _pad_to_align(mut blob: List[UInt8]):
    var rem = len(blob) % ALIGN
    if rem != 0:
        for _ in range(ALIGN - rem):
            blob.append(UInt8(0))


def _append_f32_le(mut blob: List[UInt8], src: List[Float32]):
    for i in range(len(src)):
        var b = bitcast[DType.uint32](src[i])
        blob.append(UInt8(b & 0xFF))
        blob.append(UInt8((b >> 8) & 0xFF))
        blob.append(UInt8((b >> 16) & 0xFF))
        blob.append(UInt8((b >> 24) & 0xFF))


def _shape_text(shape: List[Int]) -> String:
    var s = String("[")
    for i in range(len(shape)):
        if i > 0:
            s += ","
        s += String(shape[i])
    s += "]"
    return s^


def _bits32(x: Float32) -> UInt32:
    return bitcast[DType.uint32](x)


def _read_back_check(tmp: String, tensors: List[TensorOut], is_f32: List[Bool]) raises:
    """Loads `tmp` with the runtime loader and compares every decoded tensor with the encoder's intent."""
    try:
        var lm = load_model_file(tmp)
        if len(lm.names) != len(tensors):
            raise Error("tensor count " + String(len(lm.names)) + " != " + String(len(tensors)))
        for i in range(len(tensors)):
            if lm.names[i] != tensors[i].name:
                raise Error("tensor " + String(i) + " is '" + lm.names[i] + "', expected '" + tensors[i].name + "'")
            if lm.shapes[i] != tensors[i].shape:
                raise Error("shape of '" + tensors[i].name + "' differs")
            if len(lm.tensors[i]) != len(tensors[i].data):
                raise Error("size of '" + tensors[i].name + "' differs")
            for k in range(len(tensors[i].data)):
                var want = tensors[i].data[k]
                if not is_f32[i]:
                    want = f16_to_f32(f32_to_f16_bits(want))
                if _bits32(lm.tensors[i][k]) != _bits32(want):
                    raise Error("tensor '" + tensors[i].name + "' element " + String(k) + " reads back differently")
    except e:
        raise _e644("read-back of '" + tmp + "' failed (" + String(e) + ")")


def write_model_dir(
    dir: String,
    manifest_head: String,
    tensors: List[TensorOut],
    dtype: String,
    f32_names: List[String],
    manifest_tail: String,
) raises -> ModelDirInfo:
    """manifest_head: the manifest's JSON members that precede "tensors" (text, no trailing comma); the writer
    emits the "tensors" array and the weights_file / weights_bytes / fnv1a64 / sha256 members itself;
    manifest_tail: the members that follow, without the closing brace ("" if none). Atomic: writes <dir>.tmp, reads
    it back, then swaps it into <dir> (9.1). Nothing is touched when encoding fails (QM-E645)."""
    if dtype != "f16" and dtype != "f32":
        raise _e640("dtype must be f16 or f32, got '" + dtype + "'")
    var d = _strip_slashes(dir)
    if d.byte_length() == 0:
        raise _e640("empty model directory")
    var head = String(manifest_head.strip())
    var tail = String(manifest_tail.strip())
    if head.endswith(",") or head.startswith(",") or tail.endswith(",") or tail.startswith(","):
        raise _e640("manifest_head / manifest_tail must not start or end with a comma")
    for i in range(len(tensors)):
        var cnt = 1
        for s in tensors[i].shape:
            cnt *= s
        if cnt != len(tensors[i].data):
            raise _e640(
                "tensor '" + tensors[i].name + "' has " + String(len(tensors[i].data)) + " values for shape " + _shape_text(tensors[i].shape)
            )
        if tensors[i].dtype != "f16" and tensors[i].dtype != "f32":
            raise _e640("tensor '" + tensors[i].name + "' has dtype '" + tensors[i].dtype + "'")

    # encode everything first: an f16 overflow must not disturb the disk
    var blob = List[UInt8]()
    var is_f32 = List[Bool]()
    var entries = String()
    for i in range(len(tensors)):
        var f32t = _is_f32_tensor(tensors[i], dtype, f32_names)
        is_f32.append(f32t)
        _pad_to_align(blob)
        var off = len(blob)
        if f32t:
            _append_f32_le(blob, tensors[i].data)
        else:
            var enc = encode_f16_span(Span(tensors[i].data), tensors[i].name)
            blob.extend(enc^)
        if i > 0:
            entries += ","
        entries += (
            '{"name":'
            + json_quote(tensors[i].name)
            + ',"shape":'
            + _shape_text(tensors[i].shape)
            + ',"dtype":"'
            + ("f32" if f32t else "f16")
            + '","offset":'
            + String(off)
            + "}"
        )
    _pad_to_align(blob)
    var fnv = hex16(fnv1a64_bytes(blob))
    var sha = sha256_hex(blob)

    var manifest = String("{")
    if head.byte_length() > 0:
        manifest += head + ","
    manifest += (
        '"tensors":['
        + entries
        + '],"weights_file":"'
        + WEIGHTS_FILE
        + '","weights_bytes":'
        + String(len(blob))
        + ',"fnv1a64":"'
        + fnv
        + '","sha256":"'
        + sha
        + '"'
    )
    if tail.byte_length() > 0:
        manifest += "," + tail
    manifest += "}"

    # stage
    recover_dir(d)
    var tmp = d + ".tmp"
    var parent = _parent_dir(d)
    if not exists(parent):
        makedirs(parent, exist_ok=True)
    makedirs(tmp, exist_ok=True)
    try:
        _write_durable(tmp + "/" + WEIGHTS_FILE, Span(blob))
        _write_durable(tmp + "/" + MANIFEST_FILE, manifest.as_bytes())
        _read_back_check(tmp, tensors, is_f32)
        if sha256_file_hex(tmp + "/" + WEIGHTS_FILE) != sha:
            raise _e644("weights.bin of '" + tmp + "' does not hash to the encoded bytes")
    except e:
        _rmtree(tmp)
        raise Error(String(e))
    atomic_replace_dir(tmp, d)
    return ModelDirInfo(len(blob), fnv^, sha^, manifest.byte_length())


# ------------------------------------------------------------------------------------------- INDEX formatting


def _hex4(v: Int) -> String:
    var digits = "0123456789abcdef"
    var bs = digits.as_bytes()
    var s = String("\\u")
    s += String(unsafe_from_utf8=bs[(v >> 12) & 15 : ((v >> 12) & 15) + 1])
    s += String(unsafe_from_utf8=bs[(v >> 8) & 15 : ((v >> 8) & 15) + 1])
    s += String(unsafe_from_utf8=bs[(v >> 4) & 15 : ((v >> 4) & 15) + 1])
    s += String(unsafe_from_utf8=bs[v & 15 : (v & 15) + 1])
    return s^


def _quote_ascii(s: String) -> String:
    """JSON string literal with every character outside printable ASCII written as \\uXXXX (surrogate pairs above
    U+FFFF), the layout of the legacy `ensure_ascii` encoder."""
    var out = String('"')
    for cp in s.codepoints():
        var c = Int(cp)
        if c == 0x22:
            out += '\\"'
        elif c == 0x5C:
            out += "\\\\"
        elif c == 0x0A:
            out += "\\n"
        elif c == 0x0D:
            out += "\\r"
        elif c == 0x09:
            out += "\\t"
        elif c == 0x08:
            out += "\\b"
        elif c == 0x0C:
            out += "\\f"
        elif c >= 0x20 and c <= 0x7E:
            out += String(cp)
        elif c > 0xFFFF:
            var v = c - 0x10000
            out += _hex4(0xD800 + (v >> 10)) + _hex4(0xDC00 + (v & 0x3FF))
        else:
            out += _hex4(c)
    out += '"'
    return out^


def _indent(mut out: String, depth: Int):
    for _ in range(depth):
        out += " "


def _dump_node(v: JsonValue, depth: Int, mut out: String) raises:
    var k = v.kind()
    if k == JsonKind.ARRAY:
        var n = len(v)
        if n == 0:
            out += "[]"
            return
        out += "[\n"
        for i in range(n):
            _indent(out, depth + 1)
            _dump_node(v.at(i), depth + 1, out)
            out += ",\n" if i + 1 < n else "\n"
        _indent(out, depth)
        out += "]"
    elif k == JsonKind.OBJECT:
        var n = len(v)
        if n == 0:
            out += "{}"
            return
        out += "{\n"
        for i in range(n):
            var key = v.key_at(i)
            _indent(out, depth + 1)
            out += _quote_ascii(key) + ": "
            _dump_node(v.get(key), depth + 1, out)
            out += ",\n" if i + 1 < n else "\n"
        _indent(out, depth)
        out += "}"
    elif k == JsonKind.STRING:
        out += _quote_ascii(v.as_string())
    elif k == JsonKind.NUMBER:
        out += v.number_text()
    elif k == JsonKind.BOOL:
        out += "true" if v.as_bool() else "false"
    else:
        out += "null"


def dump_indent1(v: JsonValue) raises -> String:
    """Indent-1 JSON as the shipped INDEX.json files are laid out: every array element and object member on its own
    line, one space of indent per level, `": "` after keys, `[]` / `{}` for empty containers, ASCII only, no
    trailing newline. (`pretty_json` keeps short arrays on one line, so it differs.)"""
    var out = String()
    _dump_node(v, 0, out)
    return out^


# ----------------------------------------------------------------------------------------------- INDEX rows


def _edges_text(man: JsonValue, mut flat: List[Int]) raises -> String:
    var s = String("[")
    if man.has("edges"):
        var ej = man.get("edges")
        for i in range(len(ej)):
            var a = ej.at(i).at(0).as_int()
            var b = ej.at(i).at(1).as_int()
            flat.append(a)
            flat.append(b)
            if i > 0:
                s += ","
            s += "[" + String(a) + "," + String(b) + "]"
    s += "]"
    return s^


def _index_row(dir: String, default_flag: Int) raises -> String:
    """Compact JSON of the INDEX row of the model directory `dir`; `default_flag` < 0 omits the `default` member."""
    var d = _strip_slashes(dir)
    var mpath = d + "/" + MANIFEST_FILE
    if not isfile(mpath):
        raise _e644("'" + d + "' has no manifest.json")
    var text = _read_text(mpath)
    var man = parse_json(text)
    var wfile = man.get_string("weights_file", WEIGHTS_FILE)
    var wpath = d + "/" + wfile
    if not isfile(wpath):
        raise _e644("'" + wpath + "' is missing")
    var flat = List[Int]()
    var etxt = _edges_text(man, flat)
    var n = man.get_int("num_qubits", 0)
    var wl = wl_hash(n, flat) if n > 0 else String()
    var bs = d.as_bytes()
    var i = len(bs) - 1
    while i >= 0 and Int(bs[i]) != ord("/"):
        i -= 1
    var name = String(unsafe_from_utf8=bs[i + 1 : len(bs)])
    var row = (
        '{"id":'
        + json_quote(man.get("id").as_string())
        + ',"kind":'
        + json_quote(man.get("kind").as_string())
        + ',"num_qubits":'
        + String(n)
        + ',"edges":'
        + etxt
        + ',"wl_hash":'
        + json_quote(wl)
        + ',"dir":'
        + json_quote(name)
        + ',"sha256":'
        + json_quote(sha256_file_hex(wpath))
        + ',"bytes":'
        + String(getsize(wpath) + text.byte_length())
    )
    if default_flag >= 0:
        row += ',"default":' + ("true" if default_flag == 1 else "false")
    row += "}"
    return row^


def index_row_from_dir(dir: String) raises -> String:
    """JSON text (compact) of the INDEX row `{id, kind, num_qubits, edges, wl_hash, dir, sha256, bytes}` of a model
    directory; `sha256` is of weights.bin and `bytes` is weights.bin plus manifest.json."""
    return _index_row(dir, -1)


def _previous_defaults(path: String, mut any_default_key: Bool) -> List[String]:
    """ids of the rows of an existing INDEX.json marked `"default": true`; sets `any_default_key` when any row
    has a `default` member. An unreadable or absent file gives an empty list."""
    var ids = List[String]()
    if not isfile(path):
        return ids^
    try:
        var j = parse_json(_read_text(path))
        for i in range(len(j)):
            var r = j.at(i)
            if r.has("default"):
                any_default_key = True
                if r.get_bool("default", False):
                    ids.append(r.get_string("id", ""))
    except:
        pass
    return ids^


def rebuild_index(root: String, sub: String, default_id: String) raises:
    """Rebuilds `<root>/<sub>/INDEX.json` from every `<root>/<sub>/<id>/manifest.json`, rows sorted by directory
    name, written atomically in the indent-1 layout of 9.5. Synthesis registry: `default_id == ""` and no `default`
    column. Router registry (rows of kind `routing`, or an existing INDEX with a `default` column, or a non-empty
    `default_id`): every row gets `"default"`; a non-empty `default_id` marks that row, and an empty one preserves
    the previous default (the legacy exporter lost it, defect 7.4). QM-E640 when `default_id` names no row."""
    var dir = _strip_slashes(root)
    if sub.byte_length() > 0:
        dir = dir + "/" + _strip_slashes(sub)
    var ipath = dir + "/" + INDEX_FILE
    var any_default_key = False
    var prev = _previous_defaults(ipath, any_default_key)
    var names = List[String]()
    for n in listdir(dir):
        if n.endswith(".tmp") or n.endswith(".old"):
            continue
        if isfile(dir + "/" + n + "/" + MANIFEST_FILE):
            names.append(n)
    sort(names)
    var texts = List[String]()
    var ids = List[String]()
    var routing = False
    for n in names:
        var man = parse_json(_read_text(dir + "/" + n + "/" + MANIFEST_FILE))
        ids.append(man.get("id").as_string())
        if man.get_string("kind", "") == "routing":
            routing = True
    var with_default = routing or any_default_key or default_id.byte_length() > 0
    var found = default_id.byte_length() == 0
    for i in range(len(names)):
        var flag = -1
        if with_default:
            flag = 0
            if default_id.byte_length() > 0:
                if ids[i] == default_id:
                    flag = 1
                    found = True
            else:
                for p in prev:
                    if p == ids[i]:
                        flag = 1
        texts.append(_index_row(dir + "/" + names[i], flag))
    if not found:
        raise _e640("default model id '" + default_id + "' is not among the models under '" + dir + "'")
    var arr = String("[")
    for i in range(len(texts)):
        if i > 0:
            arr += ","
        arr += texts[i]
    arr += "]"
    write_text_atomic(ipath, dump_indent1(parse_json(arr)))


# --------------------------------------------------------------------------------------------------- verifier


def _is_str(v: JsonValue) -> Bool:
    return v.kind() == JsonKind.STRING


def _truthy_str(v: JsonValue) raises -> Bool:
    return v.kind() == JsonKind.STRING and v.as_string().byte_length() > 0


struct _Pairs(Movable):
    var paths: List[String]
    var hashes: List[String]
    var problems: List[String]  # malformed entries (reported as FAIL lines by the caller)

    def __init__(out self):
        self.paths = List[String]()
        self.hashes = List[String]()
        self.problems = List[String]()

    def add(mut self, path: String, h: String):
        self.paths.append(path.copy())
        self.hashes.append(h.copy())


def _collect_pairs(node: JsonValue, base: String, mut out: _Pairs) raises:
    """Collects (path, sha256) pairs from any INDEX.json layout, with the recursion of the legacy checker:
    {"dir": d, "sha256": h} (h of d/weights.bin, the registry layout), {"path"|"file": p, "sha256": h},
    {"dir": d, "sha256": {name: h}}, {"files": {p: h}} and {"files": [{"path": p, "sha256": h}]}."""
    if node.kind() == JsonKind.ARRAY:
        for i in range(len(node)):
            _collect_pairs(node.at(i), base, out)
        return
    if node.kind() != JsonKind.OBJECT:
        return
    var d = String()
    if _truthy_str(node.get("dir")):
        d = node.get("dir").as_string()
    elif _truthy_str(node.get("path_dir")):
        d = node.get("path_dir").as_string()
    var h = node.get("sha256")
    var p = node.get("path")
    if not _truthy_str(p):
        p = node.get("file")
    if _is_str(h) and _is_str(p):
        out.add(_join(base, p.as_string()), h.as_string())
    elif _is_str(h) and d.byte_length() > 0:
        out.add(_join(_join(base, d), node.get_string("weights_file", WEIGHTS_FILE)), h.as_string())
    elif h.kind() == JsonKind.OBJECT:
        for k in h.keys():
            var hv = h.get(k)
            if _is_str(hv):
                out.add(_join(_join(base, d), k), hv.as_string())
            else:
                out.problems.append(_join(_join(base, d), k) + ": malformed sha256 entry")
    var f = node.get("files")
    if f.kind() == JsonKind.OBJECT:
        for k in f.keys():
            var v = f.get(k)
            if _is_str(v):
                out.add(_join(_join(base, d), k), v.as_string())
            elif v.kind() == JsonKind.OBJECT and _is_str(v.get("sha256")):
                var fp = v.get_string("path", k) if _is_str(v.get("path")) else k
                out.add(_join(_join(base, d), fp), v.get("sha256").as_string())
    elif f.kind() == JsonKind.ARRAY:
        for i in range(len(f)):
            var x = f.at(i)
            if x.kind() == JsonKind.OBJECT and _is_str(x.get("sha256")):
                var xp = x.get("path")
                if not _truthy_str(xp):
                    xp = x.get("file")
                if _is_str(xp):
                    out.add(_join(_join(base, d), xp.as_string()), x.get("sha256").as_string())
                else:
                    out.problems.append(_join(base, d) + ": files entry without path")
    for k in node.keys():
        if k == "sha256" or k == "files":
            continue
        var c = node.get(k)
        if c.kind() == JsonKind.OBJECT or c.kind() == JsonKind.ARRAY:
            _collect_pairs(c, base, out)


struct _Walk(Movable):
    """Directory tree in a deterministic order: directories top-down, names sorted; `files[i]` are the non-directory
    entries of `dirs[i]`."""

    var dirs: List[String]
    var files: List[List[String]]

    def __init__(out self):
        self.dirs = List[String]()
        self.files = List[List[String]]()


def _walk(path: String, mut w: _Walk) raises:
    var names = listdir(path)
    sort(names)
    var fs = List[String]()
    var subs = List[String]()
    for n in names:
        var full = _join(path, n)
        if isdir(full):
            subs.append(full)
        else:
            fs.append(n)
    w.dirs.append(path.copy())
    w.files.append(fs^)
    for s in subs:
        _walk(s, w)


def _fixed2(size: Int) -> String:
    """`size / 1e6` formatted like `%.2f`, round half to even on the exact value of the double."""
    var centi = (size * 100) // 1000000
    var rem = (size * 100) % 1000000
    if rem > 500000:
        centi += 1
    elif rem == 500000:
        var x = Float64(size) / 1.0e6
        var resid = (x * 200.0) - Float64(2 * centi + 1)  # sign tells on which side of the decimal tie the double lies
        if resid > 0.0:
            centi += 1
        elif resid == 0.0 and centi % 2 == 1:
            centi += 1
    var whole = centi // 100
    var frac = centi % 100
    return String(whole) + "." + ("0" if frac < 10 else "") + String(frac)


def _fail(mut rep: VerifyReport, line: String):
    rep.lines.append(line.copy())
    rep.n_problems += 1


def verify_models(root: String) -> VerifyReport:
    """Same text output as the old `make models-verify`: `FAIL ...` lines, then
    `models-verify: N INDEX.json, M files ok, K problems, S MB total` (the last element of `lines`). Never raises:
    every internal error becomes a FAIL line. A missing root is not a problem."""
    var rep = VerifyReport()
    if not isdir(root):
        rep.lines.append("models-verify: no model directory '" + root + "' (nothing to verify)")
        return rep^
    var listed = Dict[String, Bool]()
    var w = _Walk()
    try:
        _walk(root, w)
    except e:
        _fail(rep, "FAIL " + root + ": cannot walk (" + String(e) + ")")
        rep.lines.append("models-verify: 0 INDEX.json, 0 files ok, " + String(rep.n_problems) + " problems, 0.00 MB total")
        return rep^
    for i in range(len(w.dirs)):
        var has_index = False
        for f in w.files[i]:
            if f == INDEX_FILE:
                has_index = True
        if not has_index:
            continue
        rep.n_index += 1
        var ip = _join(w.dirs[i], INDEX_FILE)
        var pairs = _Pairs()
        try:
            var index = parse_json(_read_text(ip))
            _collect_pairs(index, w.dirs[i], pairs)
        except e:
            _fail(rep, "FAIL " + ip + ": unreadable (" + String(e) + ")")
            continue
        for q in pairs.problems:
            _fail(rep, "FAIL " + _normpath(q) + " (listed in " + ip + ")")
        for k in range(len(pairs.paths)):
            var p = _normpath(pairs.paths[k])
            listed[p] = True
            if not isfile(p):
                _fail(rep, "FAIL " + p + ": missing (listed in " + ip + ")")
                continue
            var got: String
            try:
                got = sha256_file_hex(p)
            except e:
                _fail(rep, "FAIL " + p + ": unreadable (" + String(e) + ")")
                continue
            if got != pairs.hashes[k].lower():
                _fail(rep, "FAIL " + p + ": sha256 mismatch (QM-E617)")
            else:
                rep.n_files += 1
    for i in range(len(w.dirs)):
        for f in w.files[i]:
            var p = _normpath(_join(w.dirs[i], f))
            if f.endswith(".bin") and p not in listed:
                _fail(rep, "FAIL " + p + ": weights not listed in any INDEX.json")
            try:
                rep.total_bytes += getsize(_join(w.dirs[i], f))
            except:
                pass
    rep.lines.append(
        "models-verify: "
        + String(rep.n_index)
        + " INDEX.json, "
        + String(rep.n_files)
        + " files ok, "
        + String(rep.n_problems)
        + " problems, "
        + _fixed2(rep.total_bytes)
        + " MB total"
    )
    return rep^
