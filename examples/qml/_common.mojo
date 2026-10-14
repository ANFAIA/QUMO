"""Shared helpers for the examples/qml suite (not an example itself).

The examples run from the repository root (`.venv/bin/mojo run -I . examples/qml/<file>.mojo`)
and import this module as a sibling: `from _common import Fixture, ...`.

* `Fixture(section)` reads one section of the committed PennyLane reference
  file examples/qml/reference/pennylane_reference.json (parsed once) and
  returns numbers, vectors (nested arrays are flattened row-major) and
  matrices by key; a key may be a '/'-separated path ("init/clayer_1.weight").
* `load_table(path)` reads a whitespace- or comma-separated numeric table
  (PennyLane-style .txt data files).
* `seconds_since(t0)` and `vmax_abs_diff` are small reporting helpers.

Errors reuse QM-E834 (dataset or fixture file missing or malformed).
"""

from std.os.path import exists
from std.time import perf_counter_ns

from qumo.ai.json import parse_json, JsonValue


comptime FIXTURE_PATH = "examples/qml/reference/pennylane_reference.json"
comptime DATA_DIR = "examples/qml/data/"


def read_text(path: String) raises -> String:
    if not exists(path):
        raise Error("QM-E834: file '" + path + "' not found. Hint: run the examples from the repository root.")
    with open(path, "r") as f:
        return f.read()


def _flatten_into(v: JsonValue, mut out: List[Float64]) raises:
    if v.is_number():
        out.append(v.as_float())
        return
    if not v.is_array():
        raise Error("QM-E834: expected a number or an array of numbers in the fixture, got " + v.kind_name() + ".")
    for i in range(len(v)):
        _flatten_into(v.at(i), out)


struct Fixture(Movable):
    """One section of examples/qml/reference/pennylane_reference.json."""

    var section: String
    var root: JsonValue

    def __init__(out self, section: String) raises:
        self.section = section
        var doc = parse_json(read_text(FIXTURE_PATH))
        self.root = doc.get(section)
        if not self.root.is_defined():
            raise Error("QM-E834: section '" + section + "' missing from " + FIXTURE_PATH + ". Hint: regenerate it with examples/qml/reference/pennylane_reference.py --only " + section + ".")

    def node(self, key: String) raises -> JsonValue:
        var v = self.root.copy()
        for part in key.split("/"):
            var k = String(part)
            if v.is_object() and v.has(k):
                v = v.get(k)
            elif v.is_array():
                v = v.at(Int(k))
            else:
                raise Error("QM-E834: key '" + key + "' missing from fixture section '" + self.section + "'.")
            if not v.is_defined():
                raise Error("QM-E834: key '" + key + "' missing from fixture section '" + self.section + "'.")
        return v^

    def has(self, key: String) -> Bool:
        return self.root.has(key)

    def num(self, key: String) raises -> Float64:
        return self.node(key).as_float()

    def int(self, key: String) raises -> Int:
        return self.node(key).as_int()

    def string(self, key: String) raises -> String:
        return self.node(key).as_string()

    def vec(self, key: String) raises -> List[Float64]:
        """All numbers under `key`, flattened row-major."""
        var out = List[Float64]()
        _flatten_into(self.node(key), out)
        return out^

    def ivec(self, key: String) raises -> List[Int]:
        var out = List[Int]()
        for v in self.vec(key):
            out.append(Int(v))
        return out^

    def mat(self, key: String) raises -> List[List[Float64]]:
        """A 2-D array (rows may differ in length)."""
        var v = self.node(key)
        var out = List[List[Float64]]()
        for i in range(len(v)):
            var row = List[Float64]()
            _flatten_into(v.at(i), row)
            out.append(row^)
        return out^

    def imat(self, key: String) raises -> List[List[Int]]:
        var m = self.mat(key)
        var out = List[List[Int]]()
        for row in m:
            var r = List[Int]()
            for v in row:
                r.append(Int(v))
            out.append(r^)
        return out^


def load_table(path: String, skip_header: Bool = False) raises -> List[List[Float64]]:
    """Rows of a whitespace- or comma-separated numeric text file."""
    var text = read_text(path)
    var out = List[List[Float64]]()
    var first = True
    var width = -1
    for raw in text.split("\n"):
        var line = String(String(raw).strip())
        if line.byte_length() == 0 or line.startswith("#"):
            continue
        if first and skip_header:
            first = False
            continue
        first = False
        var row = List[Float64]()
        for tok in line.replace(",", " ").split(" "):
            var s = String(String(tok).strip())
            if s.byte_length() == 0:
                continue
            try:
                row.append(Float64(s))
            except:
                raise Error("QM-E834: '" + path + "': '" + s + "' is not a number.")
        if width < 0:
            width = len(row)
        elif len(row) != width:
            raise Error("QM-E834: '" + path + "': rows have " + String(width) + " and " + String(len(row)) + " fields.")
        out.append(row^)
    if len(out) == 0:
        raise Error("QM-E834: '" + path + "' contains no data rows.")
    return out^


def seconds_since(t0: Int) -> Float64:
    return Float64(perf_counter_ns() - t0) / 1e9


def vmax_abs_diff(a: List[Float64], b: List[Float64]) raises -> Float64:
    """max_i |a_i - b_i| (QM-E831 on a length mismatch)."""
    if len(a) != len(b):
        raise Error("QM-E831: compared vectors have lengths " + String(len(a)) + " and " + String(len(b)) + ".")
    var m = 0.0
    for i in range(len(a)):
        m = max(m, abs(a[i] - b[i]))
    return m


def fmt_s(x: Float64) -> String:
    """Seconds with three decimals for the timing lines."""
    var ms = Int(x * 1000.0 + 0.5)
    var frac = String(ms % 1000)
    while frac.byte_length() < 3:
        frac = "0" + frac
    return String(ms // 1000) + "." + frac
