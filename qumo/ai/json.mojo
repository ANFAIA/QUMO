"""Tiny, fast, dependency-free JSON for QUMO's AI-facing surfaces.

Two halves:

* `JsonWriter` - a streaming builder (begin_object / key / value / end) that
  writes straight into one `String`: correct escaping, shortest round-trip
  float formatting (full precision, `NaN`/`Inf` -> `null` because JSON has no
  such literals), no trailing commas, deterministic key order (= call order).
* `parse_json` - a byte-level recursive-descent parser into an **arena**
  (`JsonArena`: flat `List` of POD nodes + a flat child-index list + a string
  table). `JsonValue` is a cheap handle (`ArcPointer[JsonArena]`, node index),
  so `v.get("params").get("shots").as_int()` never copies the tree.

Design notes / inspiration: simdjson's "tape" layout and Rust serde_json's
`Value` ergonomics, minus everything QUMO does not need. Strings without
escapes are sliced directly (one allocation); numbers keep both the Float64 and
an exact Int when the literal is integral (JSON-RPC ids, shot counts).

Errors carry the stable code `QM-E900` plus line/col/offset so agents can fix
malformed tool arguments mechanically.
"""

from std.memory import ArcPointer
from std.math import isnan, isinf


# ---------------------------------------------------------------------------
# Escaping / scalar formatting
# ---------------------------------------------------------------------------
comptime _HEX = "0123456789abcdef"


def json_escape_into(mut out: String, s: StringSlice):
    """Append `s` as a quoted JSON string (RFC 8259 escaping) to `out`."""
    out += "\""
    var bs = s.as_bytes()
    var n = len(bs)
    var run = 0
    var hexb = _HEX.as_bytes()
    for i in range(n):
        var c = Int(bs[i])
        if c >= 0x20 and c != 0x22 and c != 0x5C:
            continue
        if i > run:
            out.write_string(StringSlice(unsafe_from_utf8=bs[run:i]))
        run = i + 1
        if c == 0x22:
            out += "\\\""
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
        else:
            out += "\\u00"
            out.write_string(StringSlice(unsafe_from_utf8=hexb[c >> 4 : (c >> 4) + 1]))
            out.write_string(StringSlice(unsafe_from_utf8=hexb[c & 15 : (c & 15) + 1]))
    if n > run:
        out.write_string(StringSlice(unsafe_from_utf8=bs[run:n]))
    out += "\""


def json_quote(s: String) -> String:
    """`s` as a quoted, escaped JSON string literal."""
    var out = String()
    json_escape_into(out, s)
    return out^


def json_number(v: Float64) -> String:
    """Shortest round-trip decimal for `v`; non-finite values become `null`."""
    if isnan(v) or isinf(v):
        return "null"
    return String(v)


# ---------------------------------------------------------------------------
# Streaming writer
# ---------------------------------------------------------------------------
struct JsonWriter(Movable):
    """Streaming JSON builder.

    ```mojo
    var w = JsonWriter()
    w.begin_object()
    w.kv_string("name", "bell")
    w.key("counts"); w.begin_object(); w.kv_int("00", 512); w.end_object()
    w.end_object()
    var text = w.finish()
    ```
    Commas are inserted automatically; `key()` must be followed by exactly
    one value (scalar, `raw`, or a nested container).
    """

    var out: String
    var _first: List[Bool]
    var _after_key: Bool

    def __init__(out self):
        self.out = String()
        self._first = List[Bool]()
        self._after_key = False

    def _sep(mut self):
        if self._after_key:
            self._after_key = False
            return
        var n = len(self._first)
        if n > 0:
            if self._first[n - 1]:
                self._first[n - 1] = False
            else:
                self.out += ","

    # containers
    def begin_object(mut self):
        self._sep()
        self.out += "{"
        self._first.append(True)

    def end_object(mut self):
        _ = self._first.pop()
        self.out += "}"

    def begin_array(mut self):
        self._sep()
        self.out += "["
        self._first.append(True)

    def end_array(mut self):
        _ = self._first.pop()
        self.out += "]"

    def key(mut self, k: StringSlice):
        self._sep()
        json_escape_into(self.out, k)
        self.out += ":"
        self._after_key = True

    # scalar values
    def string(mut self, s: StringSlice):
        self._sep()
        json_escape_into(self.out, s)

    def integer(mut self, v: Int):
        self._sep()
        self.out += String(v)

    def number(mut self, v: Float64):
        self._sep()
        self.out += json_number(v)

    def boolean(mut self, v: Bool):
        self._sep()
        self.out += "true" if v else "false"

    def null(mut self):
        self._sep()
        self.out += "null"

    def raw(mut self, json_text: StringSlice):
        """Insert already-serialized JSON (e.g. a nested document)."""
        self._sep()
        self.out += json_text

    # key/value shortcuts
    def kv_string(mut self, k: StringSlice, v: StringSlice):
        self.key(k)
        self.string(v)

    def kv_int(mut self, k: StringSlice, v: Int):
        self.key(k)
        self.integer(v)

    def kv_float(mut self, k: StringSlice, v: Float64):
        self.key(k)
        self.number(v)

    def kv_bool(mut self, k: StringSlice, v: Bool):
        self.key(k)
        self.boolean(v)

    def kv_null(mut self, k: StringSlice):
        self.key(k)
        self.null()

    def kv_raw(mut self, k: StringSlice, json_text: StringSlice):
        self.key(k)
        self.raw(json_text)

    def kv_string_list(mut self, k: StringSlice, items: List[String]):
        self.key(k)
        self.begin_array()
        for s in items:
            self.string(s)
        self.end_array()

    def kv_int_list(mut self, k: StringSlice, items: List[Int]):
        self.key(k)
        self.begin_array()
        for v in items:
            self.integer(v)
        self.end_array()

    def finish(mut self) -> String:
        """Return the document and reset the writer."""
        var s = self.out^
        self.out = String()
        self._first = List[Bool]()
        self._after_key = False
        return s^


# ---------------------------------------------------------------------------
# Arena tree
# ---------------------------------------------------------------------------
struct JsonKind:
    comptime UNDEFINED = -1  # missing key / out-of-range index
    comptime NULL = 0
    comptime BOOL = 1
    comptime NUMBER = 2
    comptime STRING = 3
    comptime ARRAY = 4
    comptime OBJECT = 5


def json_kind_name(k: Int) -> String:
    if k == JsonKind.NULL: return "null"
    if k == JsonKind.BOOL: return "boolean"
    if k == JsonKind.NUMBER: return "number"
    if k == JsonKind.STRING: return "string"
    if k == JsonKind.ARRAY: return "array"
    if k == JsonKind.OBJECT: return "object"
    return "undefined"


@fieldwise_init
struct JsonNode(ImplicitlyCopyable):
    var kind: Int
    var num: Float64  # NUMBER value; BOOL: 1.0 / 0.0
    var ival: Int  # exact integer when `is_int`
    var is_int: Bool
    var str_idx: Int  # STRING payload (index into strs)
    var key_idx: Int  # key of this node inside its parent object, -1 otherwise
    var first: Int  # ARRAY/OBJECT: first child slot in `kids`
    var count: Int  # ARRAY/OBJECT: number of children


struct JsonArena(Movable):
    var nodes: List[JsonNode]
    var kids: List[Int]
    var strs: List[String]

    def __init__(out self):
        self.nodes = List[JsonNode]()
        self.kids = List[Int]()
        self.strs = List[String]()


struct JsonValue(Copyable, Sized, Writable):
    """Handle to one node of a parsed JSON document (cheap to copy)."""

    var arena: ArcPointer[JsonArena]
    var idx: Int  # -1 => undefined

    def __init__(out self, arena: ArcPointer[JsonArena], idx: Int):
        self.arena = arena
        self.idx = idx

    def _node(self) -> JsonNode:
        return self.arena[].nodes[self.idx]

    def kind(self) -> Int:
        if self.idx < 0:
            return JsonKind.UNDEFINED
        return self.arena[].nodes[self.idx].kind

    def kind_name(self) -> String:
        return json_kind_name(self.kind())

    def is_defined(self) -> Bool:
        return self.idx >= 0

    def is_null(self) -> Bool:
        return self.kind() == JsonKind.NULL

    def is_bool(self) -> Bool:
        return self.kind() == JsonKind.BOOL

    def is_number(self) -> Bool:
        return self.kind() == JsonKind.NUMBER

    def is_string(self) -> Bool:
        return self.kind() == JsonKind.STRING

    def is_array(self) -> Bool:
        return self.kind() == JsonKind.ARRAY

    def is_object(self) -> Bool:
        return self.kind() == JsonKind.OBJECT

    def __len__(self) -> Int:
        var k = self.kind()
        if k == JsonKind.ARRAY or k == JsonKind.OBJECT:
            return self.arena[].nodes[self.idx].count
        return 0

    def at(self, i: Int) -> JsonValue:
        """i-th element of an array (or i-th member value of an object)."""
        var k = self.kind()
        if k != JsonKind.ARRAY and k != JsonKind.OBJECT:
            return JsonValue(self.arena, -1)
        var nd = self._node()
        if i < 0 or i >= nd.count:
            return JsonValue(self.arena, -1)
        return JsonValue(self.arena, self.arena[].kids[nd.first + i])

    def key_at(self, i: Int) -> String:
        """Key of the i-th member of an object ("" otherwise)."""
        var v = self.at(i)
        if v.idx < 0:
            return String()
        var ki = self.arena[].nodes[v.idx].key_idx
        if ki < 0:
            return String()
        return self.arena[].strs[ki]

    def get(self, key: StringSlice) -> JsonValue:
        """Member `key` of an object (last one wins); undefined when missing."""
        if self.kind() != JsonKind.OBJECT:
            return JsonValue(self.arena, -1)
        var nd = self._node()
        var found = -1
        for j in range(nd.count):
            var c = self.arena[].kids[nd.first + j]
            var ki = self.arena[].nodes[c].key_idx
            if ki >= 0 and self.arena[].strs[ki] == key:
                found = c
        return JsonValue(self.arena, found)

    def has(self, key: StringSlice) -> Bool:
        return self.get(key).idx >= 0

    def keys(self) -> List[String]:
        var out = List[String]()
        if self.kind() != JsonKind.OBJECT:
            return out^
        for i in range(len(self)):
            out.append(self.key_at(i))
        return out^

    # scalar accessors (raise QM-E911 on type mismatch)
    def as_string(self) raises -> String:
        if self.kind() != JsonKind.STRING:
            raise Error("QM-E911: expected a JSON string, got " + self.kind_name() + ". Hint: quote the value.")
        return self.arena[].strs[self._node().str_idx]

    def as_float(self) raises -> Float64:
        if self.kind() != JsonKind.NUMBER:
            raise Error("QM-E911: expected a JSON number, got " + self.kind_name() + ". Hint: pass a numeric literal such as 0.5.")
        return self._node().num

    def as_int(self) raises -> Int:
        if self.kind() != JsonKind.NUMBER:
            raise Error("QM-E911: expected a JSON integer, got " + self.kind_name() + ". Hint: pass an integer literal such as 1024.")
        var nd = self._node()
        if nd.is_int:
            return nd.ival
        if nd.num == Float64(Int(nd.num)):
            return Int(nd.num)
        raise Error("QM-E911: expected a JSON integer, got " + String(nd.num) + ". Hint: drop the fractional part.")

    def as_bool(self) raises -> Bool:
        if self.kind() != JsonKind.BOOL:
            raise Error("QM-E911: expected a JSON boolean, got " + self.kind_name() + ". Hint: use true or false.")
        return self._node().num != 0.0

    # defaulted member accessors (raise only on wrong type, not on absence)
    def get_string(self, key: StringSlice, default: String) raises -> String:
        var v = self.get(key)
        if v.idx < 0 or v.is_null():
            return default
        try:
            return v.as_string()
        except e:
            raise Error(_field_err(String(e), key))

    def get_int(self, key: StringSlice, default: Int) raises -> Int:
        var v = self.get(key)
        if v.idx < 0 or v.is_null():
            return default
        try:
            return v.as_int()
        except e:
            raise Error(_field_err(String(e), key))

    def get_float(self, key: StringSlice, default: Float64) raises -> Float64:
        var v = self.get(key)
        if v.idx < 0 or v.is_null():
            return default
        try:
            return v.as_float()
        except e:
            raise Error(_field_err(String(e), key))

    def get_bool(self, key: StringSlice, default: Bool) raises -> Bool:
        var v = self.get(key)
        if v.idx < 0 or v.is_null():
            return default
        try:
            return v.as_bool()
        except e:
            raise Error(_field_err(String(e), key))

    # serialization
    def to_json(self) -> String:
        """Compact serialization of this subtree (undefined -> `null`)."""
        var w = JsonWriter()
        if self.idx < 0:
            w.null()
        else:
            _write_node(self.arena[], self.idx, w)
        return w.finish()

    def to_pretty_json(self, indent: Int = 2) -> String:
        var out = String()
        if self.idx < 0:
            return "null"
        _pretty_node(self.arena[], self.idx, 0, indent, out)
        return out^

    def write_to(self, mut w: Some[Writer]):
        w.write(self.to_json())


def _field_err(err: String, key: StringSlice) -> String:
    """Prefix a QM-E911 message with the offending field name."""
    if err.startswith("QM-E911: "):
        return "QM-E911: field '" + String(key) + "': " + String(err[byte=9:])
    return err


def _write_node(arena: JsonArena, idx: Int, mut w: JsonWriter):
    var nd = arena.nodes[idx]
    var k = nd.kind
    if k == JsonKind.NULL:
        w.null()
    elif k == JsonKind.BOOL:
        w.boolean(nd.num != 0.0)
    elif k == JsonKind.NUMBER:
        if nd.is_int:
            w.integer(nd.ival)
        else:
            w.number(nd.num)
    elif k == JsonKind.STRING:
        w.string(arena.strs[nd.str_idx])
    elif k == JsonKind.ARRAY:
        w.begin_array()
        for j in range(nd.count):
            _write_node(arena, arena.kids[nd.first + j], w)
        w.end_array()
    else:
        w.begin_object()
        for j in range(nd.count):
            var c = arena.kids[nd.first + j]
            w.key(arena.strs[arena.nodes[c].key_idx])
            _write_node(arena, c, w)
        w.end_object()


def _pad(mut out: String, n: Int):
    for _ in range(n):
        out += " "


def _pretty_node(arena: JsonArena, idx: Int, depth: Int, indent: Int, mut out: String):
    var nd = arena.nodes[idx]
    var k = nd.kind
    if k == JsonKind.ARRAY or k == JsonKind.OBJECT:
        var open = "[" if k == JsonKind.ARRAY else "{"
        var close = "]" if k == JsonKind.ARRAY else "}"
        if nd.count == 0:
            out += open
            out += close
            return
        # short scalar arrays stay on one line
        var flat = k == JsonKind.ARRAY and nd.count <= 8
        if flat:
            for j in range(nd.count):
                var ck = arena.nodes[arena.kids[nd.first + j]].kind
                if ck == JsonKind.ARRAY or ck == JsonKind.OBJECT:
                    flat = False
        if flat:
            var w = JsonWriter()
            _write_node(arena, idx, w)
            out += w.finish()
            return
        out += open
        out += "\n"
        for j in range(nd.count):
            var c = arena.kids[nd.first + j]
            _pad(out, (depth + 1) * indent)
            if k == JsonKind.OBJECT:
                json_escape_into(out, arena.strs[arena.nodes[c].key_idx])
                out += ": "
            _pretty_node(arena, c, depth + 1, indent, out)
            if j + 1 < nd.count:
                out += ","
            out += "\n"
        _pad(out, depth * indent)
        out += close
        return
    var w = JsonWriter()
    _write_node(arena, idx, w)
    out += w.finish()


def pretty_json(text: String, indent: Int = 2) raises -> String:
    """Re-indent a compact JSON document for humans."""
    return parse_json(text).to_pretty_json(indent)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
comptime _MAX_DEPTH = 512


def _json_error(text: String, pos: Int, msg: String, hint: String) -> Error:
    var bs = text.as_bytes()
    var line = 1
    var col = 1
    var n = min(pos, len(bs))
    for i in range(n):
        if Int(bs[i]) == 0x0A:
            line += 1
            col = 1
        elif (Int(bs[i]) & 0xC0) != 0x80:
            col += 1
    return Error(
        "QM-E900: invalid JSON: " + msg + " (line " + String(line) + ", col "
        + String(col) + ", offset " + String(pos) + "). Hint: " + hint
    )


def _skip_ws(bs: Span[Byte, _], mut pos: Int):
    var n = len(bs)
    while pos < n:
        var c = Int(bs[pos])
        if c == 0x20 or c == 0x0A or c == 0x0D or c == 0x09:
            pos += 1
        else:
            return


def _new_node(mut arena: JsonArena, kind: Int) -> Int:
    arena.nodes.append(JsonNode(kind, 0.0, 0, False, -1, -1, 0, 0))
    return len(arena.nodes) - 1


def _hexval(c: Int) -> Int:
    if c >= 0x30 and c <= 0x39: return c - 0x30
    if c >= 0x61 and c <= 0x66: return c - 0x61 + 10
    if c >= 0x41 and c <= 0x46: return c - 0x41 + 10
    return -1


def _utf8_append(mut buf: List[UInt8], cp: Int):
    if cp < 0x80:
        buf.append(UInt8(cp))
    elif cp < 0x800:
        buf.append(UInt8(0xC0 | (cp >> 6)))
        buf.append(UInt8(0x80 | (cp & 0x3F)))
    elif cp < 0x10000:
        buf.append(UInt8(0xE0 | (cp >> 12)))
        buf.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
        buf.append(UInt8(0x80 | (cp & 0x3F)))
    else:
        buf.append(UInt8(0xF0 | (cp >> 18)))
        buf.append(UInt8(0x80 | ((cp >> 12) & 0x3F)))
        buf.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
        buf.append(UInt8(0x80 | (cp & 0x3F)))


def _parse_string(text: String, bs: Span[Byte, _], mut pos: Int) raises -> String:
    """`pos` at the opening quote; returns decoded payload, pos after closing quote."""
    var n = len(bs)
    var start = pos + 1
    var i = start
    # fast path: no escapes
    while i < n:
        var c = Int(bs[i])
        if c == 0x22:
            pos = i + 1
            return String(unsafe_from_utf8=bs[start:i])
        if c == 0x5C:
            break
        if c < 0x20:
            raise _json_error(text, i, "control character in string", "escape newlines as \\n and tabs as \\t")
        i += 1
    if i >= n:
        raise _json_error(text, pos, "unterminated string", "close the string with a double quote")
    var buf = List[UInt8](capacity=(i - start) + 16)
    for j in range(start, i):
        buf.append(bs[j])
    while i < n:
        var c = Int(bs[i])
        if c == 0x22:
            pos = i + 1
            return String(unsafe_from_utf8=Span(buf))
        if c < 0x20:
            raise _json_error(text, i, "control character in string", "escape newlines as \\n and tabs as \\t")
        if c != 0x5C:
            buf.append(bs[i])
            i += 1
            continue
        if i + 1 >= n:
            break
        var e = Int(bs[i + 1])
        i += 2
        if e == 0x22: buf.append(0x22)
        elif e == 0x5C: buf.append(0x5C)
        elif e == 0x2F: buf.append(0x2F)
        elif e == 0x62: buf.append(0x08)
        elif e == 0x66: buf.append(0x0C)
        elif e == 0x6E: buf.append(0x0A)
        elif e == 0x72: buf.append(0x0D)
        elif e == 0x74: buf.append(0x09)
        elif e == 0x75:
            if i + 4 > n:
                raise _json_error(text, i, "truncated \\u escape", "use exactly four hex digits, e.g. \\u00e9")
            var cp = 0
            for j in range(4):
                var h = _hexval(Int(bs[i + j]))
                if h < 0:
                    raise _json_error(text, i + j, "bad hex digit in \\u escape", "use exactly four hex digits, e.g. \\u00e9")
                cp = cp * 16 + h
            i += 4
            # surrogate pair
            if cp >= 0xD800 and cp <= 0xDBFF and i + 6 <= n and Int(bs[i]) == 0x5C and Int(bs[i + 1]) == 0x75:
                var lo = 0
                var ok = True
                for j in range(4):
                    var h = _hexval(Int(bs[i + 2 + j]))
                    if h < 0:
                        ok = False
                    lo = lo * 16 + h
                if ok and lo >= 0xDC00 and lo <= 0xDFFF:
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00)
                    i += 6
            _utf8_append(buf, cp)
        else:
            raise _json_error(text, i - 1, "invalid escape sequence", "valid escapes are \\\" \\\\ \\/ \\b \\f \\n \\r \\t \\uXXXX")
    raise _json_error(text, pos, "unterminated string", "close the string with a double quote")


def _is_digit(c: Int) -> Bool:
    return c >= 0x30 and c <= 0x39


def _parse_value(text: String, bs: Span[Byte, _], mut pos: Int, mut arena: JsonArena, depth: Int) raises -> Int:
    var n = len(bs)
    _skip_ws(bs, pos)
    if pos >= n:
        raise _json_error(text, pos, "unexpected end of input", "the document is truncated or empty")
    if depth > _MAX_DEPTH:
        raise _json_error(text, pos, "nesting deeper than 512", "flatten the document")
    var c = Int(bs[pos])
    if c == 0x7B:  # {
        var node = _new_node(arena, JsonKind.OBJECT)
        pos += 1
        var members = List[Int]()
        _skip_ws(bs, pos)
        if pos < n and Int(bs[pos]) == 0x7D:
            pos += 1
            arena.nodes[node].first = len(arena.kids)
            return node
        while True:
            _skip_ws(bs, pos)
            if pos >= n or Int(bs[pos]) != 0x22:
                raise _json_error(text, pos, "expected a quoted object key", "object keys must be double-quoted strings; no trailing commas")
            var k = _parse_string(text, bs, pos)
            arena.strs.append(k^)
            var key_idx = len(arena.strs) - 1
            _skip_ws(bs, pos)
            if pos >= n or Int(bs[pos]) != 0x3A:
                raise _json_error(text, pos, "expected ':' after object key", "write \"key\": value")
            pos += 1
            var child = _parse_value(text, bs, pos, arena, depth + 1)
            arena.nodes[child].key_idx = key_idx
            members.append(child)
            _skip_ws(bs, pos)
            if pos < n and Int(bs[pos]) == 0x2C:
                pos += 1
                continue
            if pos < n and Int(bs[pos]) == 0x7D:
                pos += 1
                break
            raise _json_error(text, pos, "expected ',' or '}' in object", "separate members with commas and close with }")
        arena.nodes[node].first = len(arena.kids)
        arena.nodes[node].count = len(members)
        arena.kids.extend(members^)
        return node
    if c == 0x5B:  # [
        var node = _new_node(arena, JsonKind.ARRAY)
        pos += 1
        var items = List[Int]()
        _skip_ws(bs, pos)
        if pos < n and Int(bs[pos]) == 0x5D:
            pos += 1
            arena.nodes[node].first = len(arena.kids)
            return node
        while True:
            items.append(_parse_value(text, bs, pos, arena, depth + 1))
            _skip_ws(bs, pos)
            if pos < n and Int(bs[pos]) == 0x2C:
                pos += 1
                continue
            if pos < n and Int(bs[pos]) == 0x5D:
                pos += 1
                break
            raise _json_error(text, pos, "expected ',' or ']' in array", "separate items with commas and close with ]")
        arena.nodes[node].first = len(arena.kids)
        arena.nodes[node].count = len(items)
        arena.kids.extend(items^)
        return node
    if c == 0x22:
        var s = _parse_string(text, bs, pos)
        arena.strs.append(s^)
        var node = _new_node(arena, JsonKind.STRING)
        arena.nodes[node].str_idx = len(arena.strs) - 1
        return node
    if c == 0x74 or c == 0x66 or c == 0x6E:  # true / false / null
        var lit = String("true") if c == 0x74 else (String("false") if c == 0x66 else String("null"))
        var lb = lit.as_bytes()
        var m = len(lb)
        if pos + m > n:
            raise _json_error(text, pos, "invalid literal", "use true, false or null (lowercase)")
        for j in range(m):
            if bs[pos + j] != lb[j]:
                raise _json_error(text, pos, "invalid literal", "use true, false or null (lowercase); strings need double quotes")
        pos += m
        if c == 0x6E:
            return _new_node(arena, JsonKind.NULL)
        var node = _new_node(arena, JsonKind.BOOL)
        arena.nodes[node].num = 1.0 if c == 0x74 else 0.0
        return node
    if c == 0x2D or _is_digit(c):
        var start = pos
        var neg = c == 0x2D
        if neg:
            pos += 1
        if pos >= n or not _is_digit(Int(bs[pos])):
            raise _json_error(text, pos, "invalid number", "numbers look like -12, 0.5 or 1e-3")
        var iv = 0
        var digits = 0
        if Int(bs[pos]) == 0x30:
            pos += 1
            digits = 1
        else:
            while pos < n and _is_digit(Int(bs[pos])):
                iv = iv * 10 + (Int(bs[pos]) - 0x30)
                digits += 1
                pos += 1
        var is_int = True
        if pos < n and Int(bs[pos]) == 0x2E:
            is_int = False
            pos += 1
            if pos >= n or not _is_digit(Int(bs[pos])):
                raise _json_error(text, pos, "digit expected after decimal point", "write 1.0 rather than 1.")
            while pos < n and _is_digit(Int(bs[pos])):
                pos += 1
        if pos < n and (Int(bs[pos]) == 0x65 or Int(bs[pos]) == 0x45):
            is_int = False
            pos += 1
            if pos < n and (Int(bs[pos]) == 0x2B or Int(bs[pos]) == 0x2D):
                pos += 1
            if pos >= n or not _is_digit(Int(bs[pos])):
                raise _json_error(text, pos, "digit expected in exponent", "write 1e-3")
            while pos < n and _is_digit(Int(bs[pos])):
                pos += 1
        var node = _new_node(arena, JsonKind.NUMBER)
        if is_int and digits <= 18:
            var v = -iv if neg else iv
            arena.nodes[node].ival = v
            arena.nodes[node].is_int = True
            arena.nodes[node].num = Float64(v)
        else:
            arena.nodes[node].num = Float64(StringSlice(unsafe_from_utf8=bs[start:pos]))
        return node
    raise _json_error(text, pos, "unexpected character '" + String(unsafe_from_utf8=bs[pos:pos + 1]) + "'", "values are objects, arrays, strings, numbers, true, false or null")


def parse_json(text: String) raises -> JsonValue:
    """Parse a complete JSON document; raises `QM-E900` with line/col on error."""
    var arena = JsonArena()
    var bs = text.as_bytes()
    var pos = 0
    var root = _parse_value(text, bs, pos, arena, 0)
    _skip_ws(bs, pos)
    if pos < len(bs):
        raise _json_error(text, pos, "trailing characters after the document", "send exactly one JSON value per message/line")
    _ = root
    return JsonValue(ArcPointer(arena^), 0)
