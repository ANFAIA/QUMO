"""The generated-tool store and its manifest (DESIGN.md 6.3 to 6.6).

`<store>/manifest.json` lists every published tool with the sha256 of its stored
source, cases and binary, the SDK it was built against, the platform, the test
summary and the review state. Each tool lives in `<store>/<name>/<version>-<sha12>/`
(`entry_dir`), holding `tool.mojo`, `cases.json` and the binary `<name>`.

Nothing here trusts the manifest: `check_entry_shape` validates the name, version and
directory of an entry before any path is built from it, and `verify_entry` re-hashes
the stored files. A failing entry is reported with
`QA-E150: <name>: malformed manifest entry (<rule>)`.
"""

from std.os import getenv
from std.sys import CompilationTarget

from qumo.ai.json import JsonValue, JsonWriter, parse_json

from qumo_agent.core.errors import E_TOOL_REFUSED
from qumo_agent.native.fs import exists, join_path, make_dirs, read_text_lossy, realpath_strict, write_text_atomic
from qumo_agent.util.pyfmt import py_json_dumps, py_str_repr
from qumo_agent.util.sha256 import sha256_file_hex, sha256_hex_text

comptime MANIFEST_SCHEMA = "qumo.agent_tools.v1"
comptime MANIFEST_FILE = "manifest.json"
comptime DEFAULT_TOOL_TIMEOUT_S = 30
comptime DEFAULT_MAX_OUTPUT_BYTES = 1048576


@fieldwise_init
struct ToolEntry(Copyable):
    var name: String
    var version: String
    var description: String
    var dir: String                   # relative to the store, "<name>/<version>-<binary_sha256[:12]>"
    var source_sha256: String
    var cases_sha256: String
    var binary_sha256: String
    var sdk_sha256: String
    var mojo_version: String
    var platform: String              # "darwin-arm64", "linux-x86_64", "linux-aarch64"
    var timeout_s: Int
    var max_output_bytes: Int
    var published_at: String
    var published_by: String
    var sandbox_branch: String
    var tests_passed: Int
    var tests_total: Int
    var reviewed: Bool
    var reviewed_by: String
    var reviewed_at: String

    def write_json(self, mut w: JsonWriter):
        """The entry as an object, keys in the order of DESIGN.md 6.5."""
        w.begin_object()
        w.kv_string("name", self.name)
        w.kv_string("version", self.version)
        w.kv_string("description", self.description)
        w.kv_string("dir", self.dir)
        w.kv_string("source_sha256", self.source_sha256)
        w.kv_string("cases_sha256", self.cases_sha256)
        w.kv_string("binary_sha256", self.binary_sha256)
        w.kv_string("sdk_sha256", self.sdk_sha256)
        w.kv_string("mojo_version", self.mojo_version)
        w.kv_string("platform", self.platform)
        w.kv_int("timeout_s", self.timeout_s)
        w.kv_int("max_output_bytes", self.max_output_bytes)
        w.kv_string("published_at", self.published_at)
        w.kv_string("published_by", self.published_by)
        w.kv_string("sandbox_branch", self.sandbox_branch)
        w.key("tests")
        w.begin_object()
        w.kv_int("passed", self.tests_passed)
        w.kv_int("total", self.tests_total)
        w.end_object()
        w.kv_bool("reviewed", self.reviewed)
        w.kv_string("reviewed_by", self.reviewed_by)
        w.kv_string("reviewed_at", self.reviewed_at)
        w.end_object()

    def to_json(self) -> String:
        var w = JsonWriter()
        self.write_json(w)
        return w.finish()


def _str_field(v: JsonValue, key: String) -> String:
    """A string member, or "" when it is missing or not a string (the shape and hash
    checks then refuse the entry)."""
    var f = v.get(key)
    if not f.is_string():
        return String()
    try:
        return f.as_string()
    except:
        return String()


def _int_field(v: JsonValue, key: String, default: Int) -> Int:
    var f = v.get(key)
    if not f.is_number():
        return default
    try:
        return f.as_int()
    except:
        return default


def _bool_field(v: JsonValue, key: String) -> Bool:
    """True only for a JSON `true` (anything else is not reviewed)."""
    var f = v.get(key)
    if not f.is_bool():
        return False
    try:
        return f.as_bool()
    except:
        return False


def tool_entry_from_json(key: String, v: JsonValue) -> ToolEntry:
    """An entry read from the manifest; `key` is its name in `tools` (used when the
    entry has no `name` string). Wrong or missing fields become empty values."""
    var name = _str_field(v, "name")
    if name.byte_length() == 0:
        name = key
    var tests = v.get("tests")
    return ToolEntry(
        name,
        _str_field(v, "version"),
        _str_field(v, "description"),
        _str_field(v, "dir"),
        _str_field(v, "source_sha256"),
        _str_field(v, "cases_sha256"),
        _str_field(v, "binary_sha256"),
        _str_field(v, "sdk_sha256"),
        _str_field(v, "mojo_version"),
        _str_field(v, "platform"),
        _int_field(v, "timeout_s", DEFAULT_TOOL_TIMEOUT_S),
        _int_field(v, "max_output_bytes", DEFAULT_MAX_OUTPUT_BYTES),
        _str_field(v, "published_at"),
        _str_field(v, "published_by"),
        _str_field(v, "sandbox_branch"),
        _int_field(tests, "passed", 0),
        _int_field(tests, "total", 0),
        _bool_field(v, "reviewed"),
        _str_field(v, "reviewed_by"),
        _str_field(v, "reviewed_at"),
    )


struct Manifest(Copyable):
    var path: String
    var store: String
    var entries: List[ToolEntry]

    def __init__(out self, path: String, store: String, var entries: List[ToolEntry]):
        self.path = path
        self.store = store
        self.entries = entries^

    def _index(self, name: String) -> Int:
        for i in range(len(self.entries)):
            if self.entries[i].name == name:
                return i
        return -1

    def get(self, name: String) -> Optional[ToolEntry]:
        var i = self._index(name)
        if i < 0:
            return None
        return self.entries[i].copy()

    def put(mut self, e: ToolEntry):
        """Add the entry, or replace the one with the same name."""
        var i = self._index(e.name)
        if i < 0:
            self.entries.append(e.copy())
        else:
            self.entries[i] = e.copy()

    def remove(mut self, name: String) -> Bool:
        var i = self._index(name)
        if i < 0:
            return False
        _ = self.entries.pop(i)
        return True

    def to_json(self) -> String:
        """`{"schema": ..., "tools": {...}}` with entries sorted by name, indent 1."""
        var order = _sorted_indices(self.entries)
        var w = JsonWriter()
        w.begin_object()
        w.kv_string("schema", MANIFEST_SCHEMA)
        w.key("tools")
        w.begin_object()
        for i in order:
            w.key(self.entries[i].name)
            self.entries[i].write_json(w)
        w.end_object()
        w.end_object()
        var compact = w.finish()
        try:
            return py_json_dumps(parse_json(compact), False, 1)
        except:
            return compact   # unreachable: the writer produced valid JSON

    def save(self) raises:
        """Write the manifest atomically (temp file in the store, then rename)."""
        make_dirs(self.store)
        write_text_atomic(self.path, self.to_json() + "\n")


def _sorted_indices(entries: List[ToolEntry]) -> List[Int]:
    var order = List[Int](capacity=len(entries))
    for i in range(len(entries)):
        var j = len(order)
        order.append(i)
        while j > 0 and entries[order[j - 1]].name > entries[i].name:
            order[j] = order[j - 1]
            j -= 1
        order[j] = i
    return order^


@fieldwise_init
struct HashCheck(Copyable):
    var ok: Bool
    var problems: List[String]


@fieldwise_init
struct BuildRecord(Copyable):
    """`<tool dir>/build/build.json`, written by toolsmith_build."""
    var source_sha256: String     # of the tool.mojo that was compiled (the copy in the hermetic dir)
    var sdk_sha256: String        # sdk_sha256(repo_root) at build time
    var binary_sha256: String
    var mojo_version: String
    var built_at: String

    def write_json(self, mut w: JsonWriter):
        w.begin_object()
        w.kv_string("source_sha256", self.source_sha256)
        w.kv_string("sdk_sha256", self.sdk_sha256)
        w.kv_string("binary_sha256", self.binary_sha256)
        w.kv_string("mojo_version", self.mojo_version)
        w.kv_string("built_at", self.built_at)
        w.end_object()

    def to_json(self) -> String:
        var w = JsonWriter()
        self.write_json(w)
        return w.finish()


def build_record_from_value(v: JsonValue) raises -> BuildRecord:
    if not v.is_object():
        raise Error(String(E_TOOL_REFUSED) + ": build record: expected a JSON object")
    return BuildRecord(
        _str_field(v, "source_sha256"),
        _str_field(v, "sdk_sha256"),
        _str_field(v, "binary_sha256"),
        _str_field(v, "mojo_version"),
        _str_field(v, "built_at"),
    )


def build_record_from_json(text: String) raises -> BuildRecord:
    var v: JsonValue
    try:
        v = parse_json(text)
    except e:
        raise Error(String(E_TOOL_REFUSED) + ": build record is not valid JSON: " + String(e))
    return build_record_from_value(v)


# ---------------------------------------------------------------------------
# SDK identity, store location, platform
# ---------------------------------------------------------------------------
def sdk_files() -> List[String]:
    return ["agents/mojo/qumo_agent/tool_sdk.mojo", "qumo/ai/json.mojo", "qumo/ai/mcp.mojo"]


def sdk_sha256(repo_root: String) raises -> String:
    """sha256 of the three SDK file hashes (sdk_files() order, from the main checkout)
    joined with "\\n"."""
    var hashes = List[String]()
    for f in sdk_files():
        hashes.append(sha256_file_hex(join_path(repo_root, f)))
    return sha256_hex_text(String("\n").join(hashes))


def tool_store_dir(root: String) -> String:
    """$QUMO_AGENT_TOOLS_DIR, or <root>/build/agent-tools."""
    var env = getenv("QUMO_AGENT_TOOLS_DIR", "")
    if env.byte_length() > 0:
        return env
    return join_path(root, "build/agent-tools")


def current_platform() -> String:
    var os_name: String
    comptime if CompilationTarget.is_macos():
        os_name = "darwin"
    else:
        os_name = "linux"
    comptime if CompilationTarget.is_x86():
        return os_name + "-x86_64"
    else:
        return os_name + ("-arm64" if os_name == "darwin" else "-aarch64")


# ---------------------------------------------------------------------------
# Names, versions, directories
# ---------------------------------------------------------------------------
def _is_lower_digit_underscore(c: Int) -> Bool:
    return (c >= 0x61 and c <= 0x7A) or (c >= 0x30 and c <= 0x39) or c == 0x5F


def _is_digit(c: Int) -> Bool:
    return c >= 0x30 and c <= 0x39


def _is_alnum(c: Int) -> Bool:
    return _is_digit(c) or (c >= 0x41 and c <= 0x5A) or (c >= 0x61 and c <= 0x7A)


def valid_tool_name(name: String) -> Bool:
    """^gen_[a-z0-9_]{2,40}$"""
    var bs = name.as_bytes()
    if len(bs) < 6 or len(bs) > 44 or not name.startswith("gen_"):
        return False
    for i in range(4, len(bs)):
        if not _is_lower_digit_underscore(Int(bs[i])):
            return False
    return True


def valid_tool_version(version: String) -> Bool:
    """^[0-9]{1,4}\\.[0-9]{1,4}\\.[0-9]{1,4}([.-][A-Za-z0-9]{1,16})?$, checked by hand."""
    var bs = version.as_bytes()
    var n = len(bs)
    var i = 0
    for part in range(3):
        var start = i
        while i < n and _is_digit(Int(bs[i])):
            i += 1
        var digits = i - start
        if digits < 1 or digits > 4:
            return False
        if part < 2:
            if i >= n or Int(bs[i]) != 0x2E:
                return False
            i += 1
    if i == n:
        return True
    var sep = Int(bs[i])
    if sep != 0x2E and sep != 0x2D:
        return False
    i += 1
    var suffix = n - i
    if suffix < 1 or suffix > 16:
        return False
    while i < n:
        if not _is_alnum(Int(bs[i])):
            return False
        i += 1
    return True


def _is_sha256_hex(s: String) -> Bool:
    var bs = s.as_bytes()
    if len(bs) != 64:
        return False
    for b in bs:
        var c = Int(b)
        if not (_is_digit(c) or (c >= 0x61 and c <= 0x66)):
            return False
    return True


def entry_dir(name: String, version: String, binary_sha256: String) -> String:
    """name + "/" + version + "-" + binary_sha256[:12]."""
    var bs = binary_sha256.as_bytes()
    var n = 12 if len(bs) >= 12 else len(bs)
    return name + "/" + version + "-" + String(unsafe_from_utf8=bs[0:n])


def display_name(name: String) -> String:
    """The name as written in error texts: a valid name as is, anything else as a
    Python repr (so control characters never reach a log line raw)."""
    if valid_tool_name(name):
        return name
    return py_str_repr(name)


def malformed(name: String, rule: String) -> String:
    return String(E_TOOL_REFUSED) + ": " + display_name(name) + ": malformed manifest entry (" + rule + ")"


def refused(name: String, reason: String) -> String:
    return String(E_TOOL_REFUSED) + ": " + display_name(name) + ": " + reason


def _dir_is_plain_relative(d: String) -> Bool:
    if d.byte_length() == 0 or d.startswith("/"):
        return False
    for part in d.split("/"):
        if part == ".." or part == "." or part.byte_length() == 0:
            return False
    return True


def is_inside(path: String, root: String) -> Bool:
    """True when `path` is `root` or below it (both already resolved)."""
    if path == root:
        return True
    var prefix = root if root.endswith("/") else root + "/"
    return path.startswith(prefix)


def check_entry_shape(store: String, e: ToolEntry) -> HashCheck:
    """Name, version and directory rules (DESIGN.md 6.4), checked before any path is
    built from the entry."""
    var problems = List[String]()
    if not valid_tool_name(e.name):
        problems.append(malformed(e.name, "name must match gen_[a-z0-9_]{2,40}"))
    if not valid_tool_version(e.version):
        problems.append(malformed(e.name, "version must look like 1.2.3 or 1.2.3-rc1"))
    if not _is_sha256_hex(e.binary_sha256):
        problems.append(malformed(e.name, "binary_sha256 must be 64 lower-case hex digits"))
    var dir_ok = _dir_is_plain_relative(e.dir)
    if not dir_ok:
        problems.append(malformed(e.name, "dir must be a relative path without '..'"))
    elif e.dir != entry_dir(e.name, e.version, e.binary_sha256):
        problems.append(malformed(e.name, "dir must equal <name>/<version>-<binary_sha256[:12]>"))
        dir_ok = False
    if dir_ok and len(problems) == 0:
        var full = join_path(store, e.dir)
        if exists(full):
            try:
                if not is_inside(realpath_strict(full), realpath_strict(store)):
                    problems.append(malformed(e.name, "dir resolves outside the store"))
            except err:
                problems.append(malformed(e.name, "dir cannot be resolved: " + String(err)))
    return HashCheck(len(problems) == 0, problems^)


def _check_file_hash(mut problems: List[String], name: String, path: String, label: String, expected: String):
    if not exists(path):
        problems.append(refused(name, label + " is missing from the store"))
        return
    try:
        if sha256_file_hex(path) != expected:
            problems.append(refused(name, label + " does not match its manifest hash"))
    except err:
        problems.append(refused(name, label + " cannot be read: " + String(err)))


def verify_entry(store: String, e: ToolEntry) raises -> HashCheck:
    """check_entry_shape, then the platform and the hashes of the stored tool.mojo,
    cases.json and binary."""
    var shape = check_entry_shape(store, e)
    if not shape.ok:
        return shape^
    var problems = List[String]()
    var here = current_platform()
    if e.platform != here:
        problems.append(refused(e.name, "built for " + py_str_repr(e.platform) + ", this machine is " + here))
    var d = join_path(store, e.dir)
    _check_file_hash(problems, e.name, join_path(d, "tool.mojo"), "tool.mojo", e.source_sha256)
    _check_file_hash(problems, e.name, join_path(d, "cases.json"), "cases.json", e.cases_sha256)
    _check_file_hash(problems, e.name, join_path(d, e.name), "the binary", e.binary_sha256)
    return HashCheck(len(problems) == 0, problems^)


def empty_manifest(store: String) -> Manifest:
    return Manifest(join_path(store, MANIFEST_FILE), store, List[ToolEntry]())


def load_manifest(store: String) raises -> Manifest:
    """The store's manifest; a missing file gives an empty manifest. Entries failing
    check_entry_shape are kept (verify_entry refuses them, `tools remove` can drop
    them). A file that is not a manifest raises QA-E150."""
    var m = empty_manifest(store)
    if not exists(m.path):
        return m^
    var label = m.path
    var v: JsonValue
    try:
        v = parse_json(read_text_lossy(m.path))
    except e:
        raise Error(String(E_TOOL_REFUSED) + ": " + label + ": not valid JSON (" + String(e) + ")")
    if not v.is_object() or _str_field(v, "schema") != MANIFEST_SCHEMA:
        raise Error(String(E_TOOL_REFUSED) + ": " + label + ": schema must be " + String(MANIFEST_SCHEMA))
    var tools = v.get("tools")
    if tools.is_defined() and not tools.is_object():
        raise Error(String(E_TOOL_REFUSED) + ": " + label + ": tools must be an object")
    for i in range(len(tools)):
        m.entries.append(tool_entry_from_json(tools.key_at(i), tools.at(i)))
    return m^
