"""Model Context Protocol server (MCP 2025-06-18, JSON-RPC 2.0 over stdio).

Transport: newline-delimited JSON messages on stdin/stdout; nothing but
protocol messages is ever written to stdout (logs go to stderr).

Supported methods
  initialize                -> protocol negotiation, serverInfo, capabilities.tools
  notifications/initialized -> (notification, no reply); every id-less message
                               is a notification and never gets a reply
  ping                      -> {}
  tools/list                -> every tool of the `ToolRegistry` (name, title,
                               description, inputSchema, annotations)
  tools/call                -> {content:[{type:"text",text}], structuredContent, isError}

Tool failures are *results* with `isError: true` and a structured
`Diagnostic` (stable QM-Exxx code + hint) so the model can self-correct;
protocol failures are JSON-RPC errors (-32700 parse, -32600 invalid request,
-32601 unknown method, -32602 invalid params / unknown tool).

`McpServer.handle(line) -> String` is pure (no I/O) and fully testable;
`serve_stdio()` is the thin I/O loop. Other modules add tools by registering
them in `qumo.ai.tools.default_registry()` - no protocol code changes.
"""

from std.io.file_descriptor import FileDescriptor

from qumo.ai.json import JsonWriter, JsonValue, JsonKind, parse_json
from qumo.ai.tools import ToolRegistry, ToolResult
from qumo.ai.registry import full_registry
from qumo.ai.diagnostics import diagnostic_from_error
from qumo.ai.describe import QUMO_VERSION

comptime MCP_PROTOCOL_VERSION = "2025-06-18"

comptime _INSTRUCTIONS = "QUMO quantum toolkit. Circuits are OpenQASM 3.0 (2.0 accepted) strings: 'OPENQASM 3.0; include \"stdgates.inc\"; qubit[2] q; bit[2] c; h q[0]; cx q[0], q[1]; c = measure q;'. Use qumo_validate_qasm to check code (errors carry a stable QM-Exxx code, line/col and a Hint), qumo_run_qasm for counts (seeded, deterministic), qumo_statevector / qumo_expectation for exact results, qumo_circuit_info for analysis and qumo_describe for gates/errors/capabilities. Bitstrings are Qiskit-style (leftmost = highest bit). Symbolic angles: 'input float theta;' + params."


def _supported_version(v: String) -> Bool:
    return v == "2025-06-18" or v == "2025-03-26" or v == "2024-11-05"


def _write_id(mut w: JsonWriter, id: JsonValue):
    var k = id.kind()
    if k == JsonKind.STRING:
        try:
            w.string(id.as_string())
        except:
            w.null()
    elif k == JsonKind.NUMBER:
        try:
            w.integer(id.as_int())
        except:
            try:
                w.number(id.as_float())
            except:
                w.null()
    else:
        w.null()


def rpc_error(id: JsonValue, code: Int, message: String, data_json: String = "") -> String:
    var w = JsonWriter()
    w.begin_object()
    w.kv_string("jsonrpc", "2.0")
    w.key("id")
    _write_id(w, id)
    w.key("error")
    w.begin_object()
    w.kv_int("code", code)
    w.kv_string("message", message)
    if data_json.byte_length() > 0:
        w.kv_raw("data", data_json)
    w.end_object()
    w.end_object()
    return w.finish()


def rpc_result(id: JsonValue, result_json: String) -> String:
    var w = JsonWriter()
    w.begin_object()
    w.kv_string("jsonrpc", "2.0")
    w.key("id")
    _write_id(w, id)
    w.kv_raw("result", result_json)
    w.end_object()
    return w.finish()


def tool_result_json(r: ToolResult) -> String:
    """MCP CallToolResult."""
    var w = JsonWriter()
    w.begin_object()
    w.key("content")
    w.begin_array()
    w.begin_object()
    w.kv_string("type", "text")
    w.kv_string("text", r.content_text())
    w.end_object()
    w.end_array()
    w.kv_raw("structuredContent", r.structured)
    w.kv_bool("isError", r.is_error)
    w.end_object()
    return w.finish()


def _null_id() raises -> JsonValue:
    return parse_json("null")


struct McpServer(Movable):
    var registry: ToolRegistry
    var initialized: Bool
    var protocol_version: String

    def __init__(out self):
        self.registry = full_registry()
        self.initialized = False
        self.protocol_version = MCP_PROTOCOL_VERSION

    def __init__(out self, var registry: ToolRegistry):
        self.registry = registry^
        self.initialized = False
        self.protocol_version = MCP_PROTOCOL_VERSION

    def handle(mut self, line: String) -> String:
        """Process one JSON-RPC message; returns the reply line ("" = none)."""
        var bs = line.as_bytes()
        var blank = True
        for i in range(len(bs)):
            var c = Int(bs[i])
            if c != 0x20 and c != 0x09 and c != 0x0D and c != 0x0A:
                blank = False
                break
        if blank:
            return String()
        try:
            return self._handle(line)
        except e:
            # last-resort guard: never crash the server loop
            try:
                return rpc_error(_null_id(), -32603, "Internal error: " + String(e))
            except:
                return String()

    def _handle(mut self, line: String) raises -> String:
        var msg: JsonValue
        try:
            msg = parse_json(line)
        except e:
            var d = diagnostic_from_error(String(e))
            return rpc_error(_null_id(), -32700, "Parse error: " + d.message, d.to_json())
        if not msg.is_object():
            var why = String("batch requests are not supported by MCP 2025-06-18; send one message per line") if msg.is_array() else String("a JSON-RPC message must be an object")
            return rpc_error(_null_id(), -32600, "Invalid Request: " + why)
        var id = msg.get("id")
        var is_notification = not id.is_defined()
        var jv = msg.get("jsonrpc")
        if jv.is_defined() and (not jv.is_string() or jv.as_string() != "2.0"):
            if is_notification:
                return String()
            return rpc_error(id, -32600, "Invalid Request: jsonrpc must be \"2.0\"")
        var mv = msg.get("method")
        if not mv.is_string():
            if is_notification:
                return String()  # a response from the client or junk: ignore
            return rpc_error(id, -32600, "Invalid Request: missing method")
        var method = mv.as_string()
        var params = msg.get("params")
        if is_notification:
            if method == "notifications/initialized":
                self.initialized = True
            return String()
        if method == "initialize":
            return rpc_result(id, self._initialize(params))
        if method == "ping":
            return rpc_result(id, "{}")
        if method == "tools/list":
            var w = JsonWriter()
            w.begin_object()
            w.key("tools")
            self.registry.write_list_json(w)
            w.end_object()
            return rpc_result(id, w.finish())
        if method == "tools/call":
            if not params.is_object():
                return rpc_error(id, -32602, "Invalid params: tools/call needs {\"name\": ..., \"arguments\": {...}}")
            var nv = params.get("name")
            if not nv.is_string():
                return rpc_error(id, -32602, "Invalid params: missing tool name")
            var name = nv.as_string()
            if self.registry.find(name) < 0:
                var w = JsonWriter()
                w.begin_object()
                w.kv_string("code", "QM-E912")
                w.kv_string_list("available", self.registry.names())
                w.end_object()
                return rpc_error(id, -32602, "Unknown tool: " + name, w.finish())
            var args = params.get("arguments")
            if not args.is_defined() or args.is_null():
                args = parse_json("{}")
            elif not args.is_object():
                return rpc_error(id, -32602, "Invalid params: arguments must be an object")
            var r = self.registry.call(name, args)
            return rpc_result(id, tool_result_json(r))
        return rpc_error(id, -32601, "Method not found: " + method)

    def _initialize(mut self, params: JsonValue) raises -> String:
        var requested = String()
        if params.is_object():
            requested = params.get_string("protocolVersion", "")
        self.protocol_version = requested if _supported_version(requested) else String(MCP_PROTOCOL_VERSION)
        var w = JsonWriter()
        w.begin_object()
        w.kv_string("protocolVersion", self.protocol_version)
        w.key("capabilities")
        w.begin_object()
        w.key("tools")
        w.begin_object()
        w.kv_bool("listChanged", False)
        w.end_object()
        w.end_object()
        w.key("serverInfo")
        w.begin_object()
        w.kv_string("name", "qumo")
        w.kv_string("title", "QUMO quantum toolkit")
        w.kv_string("version", QUMO_VERSION)
        w.end_object()
        w.kv_string("instructions", _INSTRUCTIONS)
        w.end_object()
        return w.finish()


def handle_request(line: String) -> String:
    """Stateless convenience wrapper (fresh server with the default registry)."""
    var s = McpServer()
    return s.handle(line)


# ---------------------------------------------------------------------------
# stdio transport
# ---------------------------------------------------------------------------
struct LineReader(Movable):
    """Buffered newline splitter over a raw file descriptor (handles lines of
    any length and a final line without trailing newline)."""

    var fd: FileDescriptor
    var buf: List[UInt8]
    var start: Int
    var end: Int
    var eof: Bool

    def __init__(out self, fd: Int = 0):
        self.fd = FileDescriptor(fd)
        self.buf = List[UInt8](length=1 << 16, fill=0)
        self.start = 0
        self.end = 0
        self.eof = False

    def _fill(mut self) raises -> Bool:
        if self.eof:
            return False
        if self.start > 0:
            var n = self.end - self.start
            for i in range(n):
                self.buf[i] = self.buf[self.start + i]
            self.start = 0
            self.end = n
        if self.end == len(self.buf):
            self.buf.resize(len(self.buf) * 2, 0)
        var got = self.fd.read_bytes(Span(self.buf)[self.end:])
        if got <= 0:
            self.eof = True
            return False
        self.end += got
        return True

    def read_line(mut self, mut line: String) raises -> Bool:
        """Next line without the newline; False at end of input."""
        var scan = self.start
        while True:
            var i = scan
            while i < self.end:
                if Int(self.buf[i]) == 0x0A:
                    var e = i
                    if e > self.start and Int(self.buf[e - 1]) == 0x0D:
                        e -= 1
                    line = String(unsafe_from_utf8=Span(self.buf)[self.start:e])
                    self.start = i + 1
                    return True
                i += 1
            scan = self.end - self.start
            if not self._fill():
                if self.end > self.start:
                    line = String(unsafe_from_utf8=Span(self.buf)[self.start:self.end])
                    self.start = self.end
                    return True
                return False
            scan = self.start + scan

    def read_all(mut self) raises -> String:
        while self._fill():
            pass
        var s = String(unsafe_from_utf8=Span(self.buf)[self.start:self.end])
        self.start = self.end
        return s^


def serve_stdio(var server: McpServer) raises:
    var reader = LineReader(0)
    var line = String()
    while reader.read_line(line):
        var reply = server.handle(line)
        if reply.byte_length() > 0:
            print(reply, flush=True)


def serve_stdio() raises:
    """Run the default QUMO MCP server on stdin/stdout until EOF."""
    serve_stdio(McpServer())
