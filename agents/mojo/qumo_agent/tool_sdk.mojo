"""Tool SDK compiled into every generated QUMO tool (DESIGN.md 6.2).

A generated tool is one program with one tool. It describes itself with a
`ToolInfo`, implements `handle(args: JsonValue) raises -> ToolOutput`, and calls
`serve_tool(info, handle)` from `main`:

```mojo
from qumo.ai.json import JsonValue
from qumo_agent.tool_sdk import ToolInfo, ToolOutput, ok, fail, arg_string, serve_tool

def handle(args: JsonValue) raises -> ToolOutput:
    var text = arg_string(args, "text", "")
    if text.byte_length() == 0:
        return fail("text is required")
    return ok(text)

def main() raises:
    serve_tool(ToolInfo("gen_echo", "Echo", "Returns its input.",
                        "{\"type\":\"object\",\"properties\":{\"text\":{\"type\":\"string\"}}}",
                        "0.1.0"), handle)
```

`serve_tool` speaks MCP over stdio, one JSON object per line:

- `initialize` answers the requested protocol version when it is one of
  2024-11-05, 2025-03-26, 2025-06-18, 2025-11-25, else 2025-06-18; `serverInfo` is
  the tool's name, title and version; capabilities `{"tools": {"listChanged": false}}`.
- `ping` answers `{}`; `tools/list` lists exactly this tool; `tools/call` runs the
  handler (another tool name is `-32602`). An exception raised by the handler
  becomes the result `{"content":[{"type":"text","text":"error: <msg>"}],"isError":true}`.
- Notifications (messages without an `id`) are ignored; other methods are `-32601`;
  unparsable lines `-32700`; lines longer than 4 MiB `-32600`.
- Result text longer than 1,000,000 bytes is cut at a code-point boundary and ends
  with `\\n... [output truncated]`.

The SDK never reads the environment and never writes files; it exits at the end of
standard input. It imports only `qumo.ai.json` and `qumo.ai.mcp`, so the binary
depends on nothing else of the harness (`sdk_sha256` in the manifest hashes this
file together with those two).
"""

from std.ffi import c_int
from std.io.file_descriptor import RawFileDescriptor
from std.memory import ArcPointer

from qumo.ai.json import JsonValue, JsonWriter, JsonArena, parse_json
from qumo.ai.mcp import rpc_result, rpc_error

comptime SDK_FALLBACK_PROTOCOL = "2025-06-18"
comptime SDK_MAX_LINE_BYTES = 4194304          # 4 MiB, one JSON-RPC message
comptime SDK_MAX_TEXT_BYTES = 1000000          # result text cap
comptime SDK_TRUNCATION_NOTE = "\n... [output truncated]"

comptime _LINE_OK = 1
comptime _LINE_EOF = 0
comptime _LINE_TOO_LONG = 2


@fieldwise_init
struct ToolInfo(Copyable):
    var name: String
    var title: String
    var description: String
    var input_schema_json: String
    var version: String


@fieldwise_init
struct ToolOutput(Copyable):
    var text: String
    var is_error: Bool
    var structured_json: String       # "" or a JSON object


def ok(text: String) -> ToolOutput:
    """A successful text result."""
    return ToolOutput(text, False, String())


def fail(text: String) -> ToolOutput:
    """An error result (`isError: true`) with this text; use it for bad input."""
    return ToolOutput(text, True, String())


def ok_json(object_json: String) -> ToolOutput:
    """A successful result whose text and `structuredContent` both carry this JSON
    object. A text that is not a JSON object becomes an error result when sent."""
    return ToolOutput(object_json, False, object_json)


# ---------------------------------------------------------------------------
# Argument helpers: a missing or null argument gives the default; a value of the
# wrong type raises, and serve_tool turns that into an "error: ..." result.
# ---------------------------------------------------------------------------
def _arg(args: JsonValue, key: String) -> JsonValue:
    return args.get(key)


def _wrong_type(key: String, expected: String, v: JsonValue) -> Error:
    return Error("argument '" + key + "' must be " + expected + ", got " + v.kind_name())


def arg_string(args: JsonValue, key: String, default: String) raises -> String:
    var v = _arg(args, key)
    if not v.is_defined() or v.is_null():
        return default
    if not v.is_string():
        raise _wrong_type(key, "a string", v)
    return v.as_string()


def arg_int(args: JsonValue, key: String, default: Int) raises -> Int:
    var v = _arg(args, key)
    if not v.is_defined() or v.is_null():
        return default
    if not v.is_number():
        raise _wrong_type(key, "an integer", v)
    try:
        return v.as_int()
    except:
        raise Error("argument '" + key + "' must be an integer, got " + v.to_json())


def arg_float(args: JsonValue, key: String, default: Float64) raises -> Float64:
    var v = _arg(args, key)
    if not v.is_defined() or v.is_null():
        return default
    if not v.is_number():
        raise _wrong_type(key, "a number", v)
    return v.as_float()


def arg_bool(args: JsonValue, key: String, default: Bool) raises -> Bool:
    var v = _arg(args, key)
    if not v.is_defined() or v.is_null():
        return default
    if not v.is_bool():
        raise _wrong_type(key, "a boolean", v)
    return v.as_bool()


# ---------------------------------------------------------------------------
# Protocol (pure functions, tested without a process)
# ---------------------------------------------------------------------------
def _supported_version(v: String) -> Bool:
    return v == "2024-11-05" or v == "2025-03-26" or v == "2025-06-18" or v == "2025-11-25"


def _null_id() -> JsonValue:
    """An undefined value: `rpc_error` writes it as `"id": null`."""
    return JsonValue(ArcPointer(JsonArena()), -1)


def _cut_text(text: String) -> String:
    """`text`, or its first bytes up to a code-point boundary at or below the cap
    followed by the truncation note."""
    var bs = text.as_bytes()
    if len(bs) <= SDK_MAX_TEXT_BYTES:
        return text
    var end = SDK_MAX_TEXT_BYTES
    # Step back over UTF-8 continuation bytes (10xxxxxx) to a code-point start.
    while end > 0 and (Int(bs[end]) & 0xC0) == 0x80:
        end -= 1
    return String(unsafe_from_utf8=bs[0:end]) + SDK_TRUNCATION_NOTE


def call_result_json(result: ToolOutput) -> String:
    """MCP CallToolResult for one handler output: one text block, `structuredContent`
    when the output carries a JSON object, and `isError`."""
    var text = result.text
    var is_error = result.is_error
    var structured = String()
    if result.structured_json.byte_length() > 0:
        # Re-serialized compactly: the raw text may hold newlines, which would split
        # the one-message-per-line protocol.
        var compact = String()
        try:
            var parsed = parse_json(result.structured_json)
            if parsed.is_object():
                compact = parsed.to_json()
        except:
            compact = String()
        if compact.byte_length() > 0:
            if compact.byte_length() <= SDK_MAX_TEXT_BYTES:
                structured = compact^
        else:
            text = "error: ok_json needs a JSON object, got: " + _cut_text(result.structured_json)
            is_error = True
    var w = JsonWriter()
    w.begin_object()
    w.key("content")
    w.begin_array()
    w.begin_object()
    w.kv_string("type", "text")
    w.kv_string("text", _cut_text(text))
    w.end_object()
    w.end_array()
    if structured.byte_length() > 0:
        w.kv_raw("structuredContent", structured)
    w.kv_bool("isError", is_error)
    w.end_object()
    return w.finish()


def tools_list_json(info: ToolInfo) -> String:
    """`{"tools":[{"name","title","description","inputSchema"}]}`."""
    var w = JsonWriter()
    w.begin_object()
    w.key("tools")
    w.begin_array()
    w.begin_object()
    w.kv_string("name", info.name)
    w.kv_string("title", info.title)
    w.kv_string("description", info.description)
    # Compact form: a schema written over several lines must not split the reply line.
    var schema = String("{\"type\":\"object\"}")
    try:
        var parsed = parse_json(info.input_schema_json)
        if parsed.is_object():
            schema = parsed.to_json()
    except:
        pass
    w.kv_raw("inputSchema", schema)
    w.end_object()
    w.end_array()
    w.end_object()
    return w.finish()


def initialize_result_json(info: ToolInfo, requested: String) -> String:
    var version = requested if _supported_version(requested) else String(SDK_FALLBACK_PROTOCOL)
    var w = JsonWriter()
    w.begin_object()
    w.kv_string("protocolVersion", version)
    w.key("capabilities")
    w.begin_object()
    w.key("tools")
    w.begin_object()
    w.kv_bool("listChanged", False)
    w.end_object()
    w.end_object()
    w.key("serverInfo")
    w.begin_object()
    w.kv_string("name", info.name)
    w.kv_string("title", info.title)
    w.kv_string("version", info.version)
    w.end_object()
    w.end_object()
    return w.finish()


def too_long_reply() -> String:
    """The answer to an input line longer than 4 MiB (its id is unknown)."""
    return rpc_error(_null_id(), -32600, "Invalid Request: message longer than " + String(SDK_MAX_LINE_BYTES) + " bytes")


def _is_blank(line: String) -> Bool:
    for b in line.as_bytes():
        var c = Int(b)
        if c != 0x20 and c != 0x09 and c != 0x0D and c != 0x0A:
            return False
    return True


def handle_line[H: def(JsonValue) raises -> ToolOutput](info: ToolInfo, handler: H, line: String) -> String:
    """One JSON-RPC line in, the reply line out ("" for notifications and blank lines)."""
    if _is_blank(line):
        return String()
    var msg: JsonValue
    try:
        msg = parse_json(line)
    except e:
        return rpc_error(_null_id(), -32700, "Parse error: " + String(e))
    if not msg.is_object():
        return rpc_error(_null_id(), -32600, "Invalid Request: a JSON-RPC message must be an object")
    var id = msg.get("id")
    if not id.is_defined():
        return String()   # notification (or a response to nothing we sent): ignored
    var mv = msg.get("method")
    if not mv.is_string():
        return rpc_error(id, -32600, "Invalid Request: missing method")
    var method: String
    try:
        method = mv.as_string()
    except:
        return rpc_error(id, -32600, "Invalid Request: missing method")
    var params = msg.get("params")
    if method == "initialize":
        var requested = String()
        if params.is_object():
            var pv = params.get("protocolVersion")
            if pv.is_string():
                try:
                    requested = pv.as_string()
                except:
                    requested = String()
        return rpc_result(id, initialize_result_json(info, requested))
    if method == "ping":
        return rpc_result(id, "{}")
    if method == "tools/list":
        return rpc_result(id, tools_list_json(info))
    if method == "tools/call":
        if not params.is_object():
            return rpc_error(id, -32602, "Invalid params: tools/call needs {\"name\": ..., \"arguments\": {...}}")
        var nv = params.get("name")
        var name = String()
        if nv.is_string():
            try:
                name = nv.as_string()
            except:
                name = String()
        if name != info.name:
            return rpc_error(id, -32602, "Unknown tool: " + name)
        var args = params.get("arguments")
        if not args.is_defined() or args.is_null():
            try:
                args = parse_json("{}")
            except:
                pass
        elif not args.is_object():
            return rpc_error(id, -32602, "Invalid params: arguments must be an object")
        var output: ToolOutput
        try:
            output = handler(args)
        except e:
            output = ToolOutput("error: " + String(e), True, String())
        return rpc_result(id, call_result_json(output))
    return rpc_error(id, -32601, "Method not found: " + method)


struct _StdinLines(Movable):
    """Newline-delimited reader over fd 0 that never holds more than one 4 MiB line
    plus one read chunk: a longer line is skipped and reported as too long."""

    var fd: RawFileDescriptor
    var buf: List[UInt8]
    var start: Int
    var eof: Bool

    def __init__(out self):
        self.fd = RawFileDescriptor(c_int(0))
        self.buf = List[UInt8]()
        self.start = 0
        self.eof = False

    def _compact(mut self):
        if self.start == 0:
            return
        var rest = List[UInt8](capacity=len(self.buf) - self.start + 65536)
        for i in range(self.start, len(self.buf)):
            rest.append(self.buf[i])
        self.buf = rest^
        self.start = 0

    def _fill(mut self) raises -> Bool:
        if self.eof:
            return False
        var chunk = List[UInt8](length=65536, fill=0)
        var got = self.fd.read_bytes(Span(chunk))
        if got <= 0:
            self.eof = True
            return False
        for i in range(got):
            self.buf.append(chunk[i])
        return True

    def next_line(mut self, mut line: String) raises -> Int:
        """_LINE_OK with `line` set (without "\\n" or "\\r\\n"), _LINE_TOO_LONG, or _LINE_EOF."""
        var too_long = False
        var scan = self.start
        while True:
            var n = len(self.buf)
            var i = scan
            while i < n:
                if Int(self.buf[i]) == 0x0A:
                    var s = self.start
                    self.start = i + 1
                    if too_long or i - s > SDK_MAX_LINE_BYTES:
                        return _LINE_TOO_LONG
                    var e = i
                    if e > s and Int(self.buf[e - 1]) == 0x0D:
                        e -= 1
                    line = String(from_utf8_lossy=Span(self.buf)[s:e])
                    return _LINE_OK
                i += 1
            if n - self.start > SDK_MAX_LINE_BYTES:
                # Drop what is buffered of the oversized line and keep skipping.
                too_long = True
                self.buf = List[UInt8]()
                self.start = 0
            else:
                self._compact()
            scan = len(self.buf)
            if not self._fill():
                if too_long:
                    return _LINE_TOO_LONG
                if len(self.buf) > self.start:
                    line = String(from_utf8_lossy=Span(self.buf)[self.start:len(self.buf)])
                    self.start = len(self.buf)
                    return _LINE_OK
                return _LINE_EOF


def serve_tool[H: def(JsonValue) raises -> ToolOutput](info: ToolInfo, handler: H) raises:
    """Serve one tool over MCP stdio until standard input ends."""
    var schema = parse_json(info.input_schema_json)
    if not schema.is_object():
        raise Error("tool " + info.name + ": input_schema_json must be a JSON object")
    var reader = _StdinLines()
    var line = String()
    while True:
        var r = reader.next_line(line)
        if r == _LINE_EOF:
            break
        var reply: String
        if r == _LINE_TOO_LONG:
            reply = too_long_reply()
        else:
            reply = handle_line(info, handler, line)
        if reply.byte_length() > 0:
            print(reply, flush=True)
