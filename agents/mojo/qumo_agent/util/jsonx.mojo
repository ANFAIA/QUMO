"""Lenient JSON and small JSON edits on top of qumo.ai.json.

`qumo/ai/json.mojo` belongs to QUMO core and is not changed; Python's
`json.loads(text, strict=False)` (which models' tool-call arguments go through)
also accepts raw control characters inside strings, so leniency is a pre-pass
that escapes them (DESIGN.md 2.2.2).

`parse_tool_arguments` applies the B2 rules of DESIGN.md 1.4 to the `arguments`
field of one tool call.
"""

from std.memory import ArcPointer

from qumo.ai.json import JsonValue, JsonKind, JsonWriter, JsonArena, JsonNode, parse_json

comptime _HEX = "0123456789abcdef"

comptime NOT_AN_OBJECT = "arguments must be a JSON object"


def escape_raw_controls(text: String) -> String:
    """Raw bytes below 0x20 inside JSON strings become \\u00XX; the rest is unchanged."""
    var bs = text.as_bytes()
    var n = len(bs)
    var needed = False
    var in_str = False
    var i = 0
    while i < n:
        var c = Int(bs[i])
        if in_str:
            if c == 0x5C:
                i += 2
                continue
            if c == 0x22:
                in_str = False
            elif c < 0x20:
                needed = True
                break
        elif c == 0x22:
            in_str = True
        i += 1
    if not needed:
        return text
    var hx = _HEX.as_bytes()
    var out = List[UInt8](capacity=n + 16)
    in_str = False
    i = 0
    while i < n:
        var c = Int(bs[i])
        if in_str:
            if c == 0x5C:
                out.append(bs[i])
                if i + 1 < n:
                    out.append(bs[i + 1])
                i += 2
                continue
            if c == 0x22:
                in_str = False
                out.append(bs[i])
            elif c < 0x20:
                out.append(0x5C)
                out.append(0x75)
                out.append(0x30)
                out.append(0x30)
                out.append(hx[c >> 4])
                out.append(hx[c & 15])
            else:
                out.append(bs[i])
        else:
            if c == 0x22:
                in_str = True
            out.append(bs[i])
        i += 1
    return String(unsafe_from_utf8=Span(out))


def parse_json_lenient(text: String) raises -> JsonValue:
    """json.loads(text, strict=False): raw control characters allowed inside strings."""
    return parse_json(escape_raw_controls(text))


@fieldwise_init
struct ParsedArgs(Copyable):
    var ok: Bool
    var args: JsonValue          # an object when ok
    var error: String            # parse error text when not ok


def empty_object() -> JsonValue:
    """A fresh `{}`."""
    var arena = JsonArena()
    arena.nodes.append(JsonNode(JsonKind.OBJECT, 0.0, 0, False, False, -1, -1, 0, 0))
    return JsonValue(ArcPointer(arena^), 0)


def _is_falsy(v: JsonValue) -> Bool:
    """Python truthiness of a json.loads() result is false."""
    var k = v.kind()
    if k == JsonKind.UNDEFINED or k == JsonKind.NULL:
        return True
    if k == JsonKind.BOOL:
        try:
            return not v.as_bool()
        except:
            return True
    if k == JsonKind.NUMBER:
        try:
            return v.as_float() == 0.0
        except:
            return False
    if k == JsonKind.STRING:
        try:
            return v.as_string().byte_length() == 0
        except:
            return True
    return len(v) == 0


def _classify(v: JsonValue) -> ParsedArgs:
    if v.is_object():
        return ParsedArgs(True, v.copy(), String())
    if _is_falsy(v):
        return ParsedArgs(True, empty_object(), String())
    return ParsedArgs(False, empty_object(), NOT_AN_OBJECT)


def parse_tool_arguments(raw: JsonValue) -> ParsedArgs:
    """B2 rules (DESIGN.md 1.4): `raw` is the `arguments` field of one tool call.

    Absent, null or "" give {}; a string is parsed leniently; an object (given
    directly or parsed) is used as is; a falsy non-object (null, false, 0, "", [])
    gives {}; a truthy non-object ([1], 5, "x", true) is invalid with
    "arguments must be a JSON object" (X13); a parse failure is invalid with the
    parse error text."""
    if raw.is_string():
        var text = String()
        try:
            text = raw.as_string()
        except:
            pass
        if text.byte_length() == 0:
            return ParsedArgs(True, empty_object(), String())
        try:
            return _classify(parse_json_lenient(text))
        except e:
            return ParsedArgs(False, empty_object(), String(e))
    return _classify(raw)


def _members(obj: JsonValue) -> List[Tuple[String, Int]]:
    """(key, member index) as Python's dict holds them: first position, last value."""
    var out = List[Tuple[String, Int]]()
    for i in range(len(obj)):
        var k = obj.key_at(i)
        var found = -1
        for j in range(len(out)):
            if out[j][0] == k:
                found = j
                break
        if found >= 0:
            out[found] = (k, i)
        else:
            out.append((k, i))
    return out^


def json_with(obj: JsonValue, key: String, value_json: String) raises -> JsonValue:
    """Copy of `obj` with `key` set to the JSON text `value_json` (an existing key
    keeps its position; a new key is appended)."""
    if not obj.is_object():
        raise Error("QA-E140: json_with needs a JSON object, got " + obj.kind_name())
    _ = parse_json(value_json)   # refuse invalid JSON before splicing it in
    var w = JsonWriter()
    w.begin_object()
    var replaced = False
    for m in _members(obj):
        w.key(m[0])
        if m[0] == key:
            w.raw(value_json)
            replaced = True
        else:
            w.raw(obj.at(m[1]).to_json())
    if not replaced:
        w.key(key)
        w.raw(value_json)
    w.end_object()
    return parse_json(w.finish())


def json_without(obj: JsonValue, key: String) raises -> JsonValue:
    """Copy of `obj` without `key`."""
    if not obj.is_object():
        raise Error("QA-E140: json_without needs a JSON object, got " + obj.kind_name())
    var w = JsonWriter()
    w.begin_object()
    for m in _members(obj):
        if m[0] == key:
            continue
        w.key(m[0])
        w.raw(obj.at(m[1]).to_json())
    w.end_object()
    return parse_json(w.finish())
