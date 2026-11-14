"""Portions ported from deepagents 0.7.32 and langchain 1.4.8 / langchain-core 1.7.4
(Copyright (c) LangChain, Inc., MIT License; see agents/mojo/LICENSES/langchain-MIT.txt).

The virtual file system of a run (deepagents' StateBackend `files` channel) and the
per-call write transactions that reproduce LangGraph's concurrent super-step semantics
(DESIGN.md 1.4, 2.7.2, X1).

`Vfs.files` maps a path to a `FileData` record; its insertion order is Python's dict
order. A tool call writes into its own `VfsTx` (`None` = delete); the loop applies the
transactions of one turn in call order with the reducer "right side wins, None deletes".
"""

from std.collections import Dict, Optional

from qumo.ai.json import JsonValue, JsonWriter, parse_json

from qumo_agent.native.fs import normpath
from qumo_agent.harness._fmt import fmt1, fmt2
from qumo_agent.harness.texts import (
    ERR_PATH_TRAVERSAL, ERR_WINDOWS_PATH, ERR_PATH_TRAVERSAL_AFTER_NORMALIZE,
)


@fieldwise_init
struct FileData(Copyable):
    var content: String
    var encoding: String            # always "utf-8" in this port
    var created_at: String
    var modified_at: String

    def same_as(self, other: FileData) -> Bool:
        return (self.content == other.content and self.encoding == other.encoding
                and self.created_at == other.created_at and self.modified_at == other.modified_at)


def _write_file_data(mut w: JsonWriter, fd: FileData):
    w.begin_object()
    w.kv_string("content", fd.content)
    w.kv_string("encoding", fd.encoding)
    w.kv_string("created_at", fd.created_at)
    w.kv_string("modified_at", fd.modified_at)
    w.end_object()


struct Vfs(Copyable):
    var files: Dict[String, FileData]

    def __init__(out self):
        self.files = Dict[String, FileData]()

    def get(self, path: String) -> Optional[FileData]:
        var v = self.files.get(path)
        if v:
            return v.value().copy()
        return None

    def set(mut self, path: String, fd: FileData):
        """Inserts or replaces; an existing key keeps its position (Python dict)."""
        self.files[path] = fd.copy()

    def remove(mut self, path: String):
        try:
            _ = self.files.pop(path)
        except:
            pass

    def equals(self, other: Vfs) -> Bool:
        """Same keys (order-insensitive) and the same four fields for each."""
        if len(self.files) != len(other.files):
            return False
        for item in self.files.items():
            var o = other.files.get(item.key)
            if not o:
                return False
            if not item.value.same_as(o.value()):
                return False
        return True

    def to_json(self) -> String:
        """`{"<path>":{"content","encoding","created_at","modified_at"},...}`."""
        var w = JsonWriter()
        w.begin_object()
        for item in self.files.items():
            w.key(item.key)
            _write_file_data(w, item.value)
        w.end_object()
        return w.finish()

    def paths(self) -> List[String]:
        """The keys in insertion order."""
        var out = List[String]()
        for k in self.files.keys():
            out.append(k)
        return out^


def _field(v: JsonValue, key: String, default: String) raises -> String:
    var f = v.get(key)
    if not f.is_defined() or f.is_null():
        return default
    if not f.is_string():
        raise Error("QA-E120: file field '" + key + "' must be a string")
    return f.as_string()


def vfs_from_json(text: String) raises -> Vfs:
    """The inverse of `Vfs.to_json`; "" gives an empty file system."""
    var out = Vfs()
    if text.byte_length() == 0:
        return out^
    var doc = parse_json(text)
    if doc.is_null():
        return out^
    if not doc.is_object():
        raise Error("QA-E120: a file dict must be a JSON object, got " + doc.kind_name())
    for i in range(len(doc)):
        var v = doc.at(i)
        if not v.is_object():
            raise Error("QA-E120: file '" + doc.key_at(i) + "' must be an object")
        var c = v.get("content")
        if not c.is_string():
            raise Error("QA-E120: file '" + doc.key_at(i) + "' has no string content")
        out.set(doc.key_at(i), FileData(c.as_string(), _field(v, "encoding", "utf-8"),
                                        _field(v, "created_at", ""), _field(v, "modified_at", "")))
    return out^


struct VfsTx(Copyable):
    """The writes of one tool call; `None` deletes. Insertion order = first write order."""
    var writes: Dict[String, Optional[FileData]]

    def __init__(out self):
        self.writes = Dict[String, Optional[FileData]]()

    def get(self, base: Vfs, path: String) -> Optional[FileData]:
        """The call's view: its own writes over the turn-start snapshot."""
        var w = self.writes.get(path)
        if w:
            var v = w.value().copy()
            if v:
                return v.value().copy()
            return None
        return base.get(path)

    def set(mut self, path: String, fd: FileData):
        self.writes[path] = Optional[FileData](fd.copy())

    def delete(mut self, path: String):
        self.writes[path] = Optional[FileData](None)

    def view(self, base: Vfs) -> Vfs:
        """Base plus writes: base order (deleted keys removed), then new keys."""
        var out = base.copy()
        apply_tx(out, self)
        return out^


def apply_tx(mut vfs: Vfs, tx: VfsTx):
    """Reducer [fs §1]: right side wins, None deletes."""
    for item in tx.writes.items():
        if item.value:
            vfs.set(item.key, item.value.value())
        else:
            vfs.remove(item.key)


def new_file_data(content: String, now: String) -> FileData:
    """deepagents create_file_data."""
    return FileData(content, "utf-8", now, now)


def updated_file_data(old: FileData, content: String, now: String) -> FileData:
    """deepagents update_file_data: keeps encoding and created_at."""
    return FileData(content, old.encoding, old.created_at, now)


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
def _has_dotdot_component(path: String) -> Bool:
    for part in path.replace("\\", "/").split("/"):
        if part == "..":
            return True
    return False


def _is_windows_drive(path: String) -> Bool:
    var bs = path.as_bytes()
    if len(bs) < 2 or Int(bs[1]) != 0x3A:
        return False
    var c = Int(bs[0])
    return (c >= 0x41 and c <= 0x5A) or (c >= 0x61 and c <= 0x7A)


def validate_vfs_path(path: String) raises -> String:
    """deepagents validate_path; raises with the message only (callers add "Error: ").

    `..` as a component or a leading `~` is traversal; `X:` is a Windows path; then
    normpath, backslashes to `/`, a leading `/` added, and a second traversal check.
    `foo/bar` -> `/foo/bar`, `/./x//y/` -> `/x/y`, `""` and `.` -> `/.`."""
    if _has_dotdot_component(path) or path.startswith("~"):
        raise Error(fmt1(ERR_PATH_TRAVERSAL, "path", path))
    if _is_windows_drive(path):
        raise Error(fmt1(ERR_WINDOWS_PATH, "path", path))
    var normalized = normpath(path).replace("\\", "/")
    if not normalized.startswith("/"):
        normalized = "/" + normalized
    for part in normalized.split("/"):
        if part == "..":
            raise Error(fmt2(ERR_PATH_TRAVERSAL_AFTER_NORMALIZE, "path", path, "normalized", normalized))
    return normalized^
