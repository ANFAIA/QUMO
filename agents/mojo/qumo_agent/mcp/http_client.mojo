"""MCP client over streamable HTTP (DESIGN.md 2.6.3), used for alphaXiv.

One `McpHttpSession` is one MCP session on one URL. Every JSON-RPC request is a POST
on its own connection (the net layer sends `Connection: close`); the session itself
lives in the `mcp-session-id` header, which is taken from the response to `initialize`
only. Every POST sends

    accept: application/json, text/event-stream
    content-type: application/json
    authorization: Bearer <value of the bearer_env variable>      (when bearer_env is set)
    mcp-session-id: <id>                                           (after initialize)
    mcp-protocol-version: <negotiated version>                     (after initialize)

The key is read with `getenv` each time a request is built; it is never stored, logged
or returned, and every error text built from a response or from the transport goes
through `redact_all` (the bearer variable plus every secret-looking variable).

Responses to a request: `application/json` is one JSON-RPC message (an array is also
searched for the matching id); `text/event-stream` is read with `SseParser` until the
response with the request's id arrives (priming events, notifications, server requests
and events of another type are skipped; a stream that ends first is an error); 202 with
no answer, an unknown content type and any status outside 2xx are errors; 404 is the
synthetic JSON-RPC error `Session terminated` (code 32600, as the Python SDK).
`notifications/initialized` accepts any 2xx (a 404 on a notification is ignored, as the
SDK does). No GET stream is opened, server requests are not answered, and a broken
stream is not resumed with `last-event-id`.

Timeouts (httpx's, `SP/mcp/client/_httpx_utils.py:14-15`): 30 s to connect and for the
exchanges that answer at once (initialize, tools/list, notifications), 300 s for the
head of a `tools/call` answer and for every read of an SSE stream. The cap is 16 MiB
per message (a JSON body or one SSE event) and 64 MiB per stream.

Errors: protocol and HTTP failures are `QA-E130`; a read that times out is
`QA-E130: MCP server did not answer <method> within <n> ms`; connect, DNS and TLS
failures keep the net layer's `QA-E104` text. A session is not broken by a failed
request (each request has its own connection).
"""

from std.os import getenv

from qumo.ai.json import JsonValue, parse_json

from qumo_agent.core.errors import E_MCP, E_CONFIG, E_NETWORK, qa_error, error_code
from qumo_agent.net.url import Url, parse_url
from qumo_agent.net.http import (
    HttpClient, HttpRequest, HttpOptions, HttpBodyStream, REDIRECT_SAME_ORIGIN,
)
from qumo_agent.net.sse import SseParser, SseEvent
from qumo_agent.util.redact import redact_all
from qumo_agent.mcp.protocol import (
    McpTool, McpCallResult, mcp_supported_version, rpc_request_json, rpc_notification_json,
    initialize_params_json, tools_list_params_json, call_params_json,
    parse_tools_page, parse_call_result, MCP_CLIENT_NAME, MCP_CLIENT_VERSION,
)
from qumo_agent.mcp.stdio_client import redact_mcp_text, MAX_TOOL_PAGES

comptime MCP_CONNECT_TIMEOUT_MS = 30000
comptime MCP_IO_TIMEOUT_MS = 30000
comptime MCP_SSE_READ_TIMEOUT_MS = 300000
comptime MCP_MAX_MESSAGE_BYTES = 16 * 1024 * 1024
comptime MCP_MAX_STREAM_BYTES = 64 * 1024 * 1024
comptime MCP_CLOSE_TIMEOUT_MS = 5000
comptime MCP_REDIRECTS = 10
comptime ERROR_BODY_BYTES = 64 * 1024
comptime ERROR_MESSAGE_CODE_POINTS = 600
comptime SESSION_TERMINATED_CODE = 32600       # positive in the SDK source (streamable_http.py:372)
comptime USER_AGENT = "qumo-agent/0.2 (Mojo)"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _is_valid_env_name(s: String) -> Bool:
    var bs = s.as_bytes()
    if len(bs) == 0:
        return False
    for i in range(len(bs)):
        var c = Int(bs[i])
        var alpha = (c >= 0x41 and c <= 0x5A) or (c >= 0x61 and c <= 0x7A) or c == 0x5F
        var digit = c >= 0x30 and c <= 0x39
        if not alpha and not (digit and i > 0):
            return False
    return True


def _is_loopback_host(host: String) -> Bool:
    """localhost, 127.0.0.0/8 and ::1 (the URL parser lower-cases and unbrackets hosts)."""
    return host == "localhost" or host == "::1" or host.startswith("127.")


def _first_code_points(s: String, n: Int) -> String:
    var bs = s.as_bytes()
    var count = 0
    var i = 0
    while i < len(bs) and count < n:
        i += 1
        while i < len(bs) and (Int(bs[i]) & 0xC0) == 0x80:
            i += 1
        count += 1
    return String(s[byte=0:i])


def _error_message(body: String) -> String:
    """`error.message` / `message` / a string `error` of a JSON body, else the first
    600 code points of the body."""
    try:
        var doc = parse_json(body)
        if doc.is_object():
            var e = doc.get("error")
            if e.is_object():
                var m = e.get("message")
                if m.is_string():
                    return m.as_string()
            elif e.is_string():
                return e.as_string()
            var m2 = doc.get("message")
            if m2.is_string():
                return m2.as_string()
    except:
        pass
    return _first_code_points(body, ERROR_MESSAGE_CODE_POINTS)


def _lower(s: String) -> String:
    return s.lower()


def _rpc_error_text(err: JsonValue, method: String) -> String:
    var code = String("?")
    var cv = err.get("code")
    try:
        if cv.is_number():
            code = cv.number_text()
        elif cv.is_defined():
            code = cv.to_json()
    except:
        pass
    var message = String()
    var mt = err.get("message")
    try:
        if mt.is_string():
            message = mt.as_string()
        elif mt.is_defined():
            message = mt.to_json()
    except:
        pass
    return "MCP error " + code + " on " + method + ": " + message


def _match_response(msg: JsonValue, id: Int) -> Bool:
    """True when `msg` is a JSON-RPC response (result or error, no `method`) to `id`."""
    if not msg.is_object():
        return False
    if msg.get("method").is_string():
        return False        # a notification or a request from the server
    var rid = msg.get("id")
    if not rid.is_number():
        return False
    try:
        return rid.as_int() == id
    except:
        return False


struct McpHttpSession(Movable):
    var session_id: String           # "" until initialize returns an mcp-session-id
    var protocol_version: String     # negotiated; "" before initialize
    var server_name: String          # serverInfo.name from initialize
    # Limits and timeouts: public so that tests (and callers with a slow server) can change them.
    var max_message_bytes: Int
    var connect_timeout_ms: Int
    var io_timeout_ms: Int           # initialize, tools/list, notifications
    var sse_read_timeout_ms: Int     # head of tools/call answers and every SSE read
    var _http: HttpClient
    var _url: String
    var _bearer_env: String
    var _plain: Bool
    var _next_id: Int
    var _closed: Bool

    def __init__(out self, var http: HttpClient, url: String, bearer_env: String) raises:
        """Checks the URL and the key variable; nothing is sent. `bearer_env` is the NAME of
        the variable holding the bearer token ("" = no Authorization header). Plain `http`
        is accepted when the URL itself is `http://`, but a bearer token is never sent in
        clear text to a host other than loopback."""
        var parsed: Url
        try:
            parsed = parse_url(url)
        except e:
            raise qa_error(E_CONFIG, "invalid MCP URL: " + String(e), "")
        if bearer_env.byte_length() > 0:
            if not _is_valid_env_name(bearer_env):
                raise qa_error(E_CONFIG, "'" + bearer_env + "' is not a valid environment variable name", "")
            if getenv(bearer_env, "").byte_length() == 0:
                raise qa_error(E_CONFIG, "set " + bearer_env + " to use the MCP server " + parsed.origin(), "")
            if parsed.scheme == "http" and not _is_loopback_host(parsed.host):
                raise qa_error(E_CONFIG, "refusing to send " + bearer_env + " to " + parsed.origin()
                               + " over plain http", "use an https URL")
        self.session_id = String()
        self.protocol_version = String()
        self.server_name = String()
        self.max_message_bytes = MCP_MAX_MESSAGE_BYTES
        self.connect_timeout_ms = MCP_CONNECT_TIMEOUT_MS
        self.io_timeout_ms = MCP_IO_TIMEOUT_MS
        self.sse_read_timeout_ms = MCP_SSE_READ_TIMEOUT_MS
        self._http = http^
        self._url = url
        self._bearer_env = bearer_env
        self._plain = parsed.scheme == "http"
        self._next_id = 0
        self._closed = False

    def requests(self) -> Int:
        """HTTP requests sent so far (for RunStats)."""
        return self._http.requests

    # ------------------------------------------------------------------
    # transport
    # ------------------------------------------------------------------
    def _redact(self, text: String) -> String:
        var names = List[String]()
        names.append(self._bearer_env)
        return redact_all(redact_mcp_text(text), names)

    def _fail(self, code: String, message: String, hint: String = "") -> Error:
        return qa_error(code, self._redact(message), hint)

    def _options(self, io_ms: Int) -> HttpOptions:
        return HttpOptions(self.connect_timeout_ms, io_ms, MCP_MAX_STREAM_BYTES, False, REDIRECT_SAME_ORIGIN,
                           MCP_REDIRECTS, self._plain, False)

    def _build_request(self, http_method: String, body: String) raises -> HttpRequest:
        var names: List[String] = ["Accept", "Content-Type"]
        var values: List[String] = ["application/json, text/event-stream", "application/json"]
        if self._bearer_env.byte_length() > 0:
            var key = getenv(self._bearer_env, "")
            if key.byte_length() == 0:
                raise qa_error(E_CONFIG, "set " + self._bearer_env + " to use the MCP server", "")
            names.append("Authorization")
            values.append("Bearer " + key)
        if self.session_id.byte_length() > 0:
            names.append("Mcp-Session-Id")
            values.append(self.session_id)
        if self.protocol_version.byte_length() > 0:
            names.append("Mcp-Protocol-Version")
            values.append(self.protocol_version)
        names.append("User-Agent")
        values.append(USER_AGENT)
        return HttpRequest(http_method, self._url, names^, values^, body)

    def _transport_error(self, e: Error, method: String, wait_ms: Int) -> Error:
        """A read/write timeout becomes the MCP "did not answer" error; anything else keeps
        its text (redacted)."""
        var text = String(e)
        if error_code(e) == E_NETWORK and (text.find("read timed out") >= 0 or text.find("write timed out") >= 0):
            return qa_error(E_MCP, "MCP server did not answer " + method + " within " + String(wait_ms) + " ms", "")
        return Error(self._redact(text))

    def _open(mut self, http_method: String, body: String, method: String, io_ms: Int) raises -> HttpBodyStream:
        var req = self._build_request(http_method, body)
        var opts = self._options(io_ms)
        try:
            return self._http.open(req, opts)
        except e:
            raise self._transport_error(e, method, io_ms)

    def _read_all(self, mut stream: HttpBodyStream, limit: Int, timeout_ms: Int, method: String,
                  clip: Bool) raises -> String:
        """The rest of the body. Over `limit` bytes: an error, or (with `clip`) the first
        `limit` bytes."""
        var body = List[UInt8]()
        while True:
            var chunk = List[UInt8]()
            var more: Bool
            try:
                more = stream.read_chunk(chunk, timeout_ms)
            except e:
                raise self._transport_error(e, method, timeout_ms)
            if len(body) + len(chunk) > limit:
                if not clip:
                    raise self._fail(E_MCP, "MCP response to " + method + " is larger than " + String(limit) + " bytes")
                var room = limit - len(body)
                for i in range(room):
                    body.append(chunk[i])
                return String(from_utf8_lossy=Span(body))
            body.extend(chunk^)
            if not more:
                break
        return String(from_utf8_lossy=Span(body))

    def _status_error(self, mut stream: HttpBodyStream, method: String) -> Error:
        """The error for a non-2xx answer (the stream is read for its message and closed)."""
        var text = String()
        try:
            text = self._read_all(stream, ERROR_BODY_BYTES, self.io_timeout_ms, method, True)
        except:
            pass
        var status = stream.status
        var reason = stream.reason.copy()
        stream.close()
        var message = _error_message(text)
        var hint = String()
        if (status == 401 or status == 403) and self._bearer_env.byte_length() > 0:
            hint = "check the value of " + self._bearer_env
        var line = "MCP HTTP " + String(status)
        if reason.byte_length() > 0:
            line += " " + reason
        line += " on " + method
        if message.byte_length() > 0:
            line += ": " + message
        return self._fail(E_MCP, line, hint)

    def _take_message(self, msg: JsonValue, id: Int, method: String, mut found: Bool) raises -> JsonValue:
        """If `msg` answers request `id`, sets `found` and returns its `result`; a JSON-RPC
        `error` raises."""
        if not _match_response(msg, id):
            return JsonValue(msg.arena, -1)
        var err = msg.get("error")
        if err.is_defined() and not err.is_null():
            raise self._fail(E_MCP, _rpc_error_text(err, method))
        found = True
        return msg.get("result")

    def _json_answer(self, body: String, id: Int, method: String) raises -> JsonValue:
        var doc: JsonValue
        try:
            doc = parse_json(body)
        except e:
            raise self._fail(E_MCP, "MCP response to " + method + " is not JSON: " + _first_code_points(String(e), 200))
        var found = False
        var result: JsonValue
        if doc.is_array():
            result = JsonValue(doc.arena, -1)
            for i in range(len(doc)):
                var r = self._take_message(doc.at(i), id, method, found)
                if found:
                    result = r^
                    break
        else:
            result = self._take_message(doc, id, method, found)
        if not found:
            raise self._fail(E_MCP, "MCP response to " + method + " has no JSON-RPC response for request id " + String(id))
        return result^

    def _sse_answer(self, mut stream: HttpBodyStream, id: Int, method: String) raises -> JsonValue:
        var parser = SseParser()
        var ev = SseEvent(String(), String(), String(), -1)
        while True:
            var chunk = List[UInt8]()
            var more: Bool
            try:
                more = stream.read_chunk(chunk, self.sse_read_timeout_ms)
                parser.feed(chunk)
            except e:
                raise self._transport_error(e, method, self.sse_read_timeout_ms)
            while parser.next_event(ev):
                if ev.event != "message" or ev.data.byte_length() == 0:
                    continue        # priming events and other event types
                if ev.data.byte_length() > self.max_message_bytes:
                    raise self._fail(E_MCP, "MCP event for " + method + " is larger than " + String(self.max_message_bytes) + " bytes")
                var msg: JsonValue
                try:
                    msg = parse_json(ev.data)
                except e:
                    raise self._fail(E_MCP, "MCP event for " + method + " is not JSON: " + _first_code_points(String(e), 200))
                var found = False
                var r = self._take_message(msg, id, method, found)
                if found:
                    return r^
            if not more:
                raise self._fail(E_MCP, "MCP server closed the event stream before answering " + method)

    def _request(mut self, method: String, params_json: String) raises -> JsonValue:
        """POST one request and return the `result` of its response."""
        if self._closed:
            raise self._fail(E_MCP, "MCP session is closed (" + method + ")")
        var id = self._next_id
        self._next_id += 1
        var wait_ms = self.sse_read_timeout_ms if method == "tools/call" else self.io_timeout_ms
        var stream = self._open("POST", rpc_request_json(id, method, params_json), method, wait_ms)
        var status = stream.status
        if status == 404:
            stream.close()
            raise self._fail(E_MCP, "MCP error " + String(SESSION_TERMINATED_CODE) + " on " + method + ": Session terminated")
        if status == 202:
            stream.close()
            raise self._fail(E_MCP, "MCP server accepted " + method + " (HTTP 202) without answering it")
        if status < 200 or status >= 300:
            raise self._status_error(stream, method)
        if method == "initialize":
            self.session_id = stream.header("mcp-session-id")
        var ctype = _lower(stream.header("content-type"))
        var result: JsonValue
        try:
            if ctype.startswith("application/json"):
                var text = self._read_all(stream, self.max_message_bytes, wait_ms, method, False)
                result = self._json_answer(text, id, method)
            elif ctype.startswith("text/event-stream"):
                result = self._sse_answer(stream, id, method)
            else:
                var shown = ctype if ctype.byte_length() > 0 else String("none")
                raise self._fail(E_MCP, "MCP response to " + method + " has an unexpected content type: " + _first_code_points(shown, 100))
        except e:
            stream.close()
            raise e^
        stream.close()
        return result^

    def _notify(mut self, method: String) raises:
        """POST a notification; any 2xx is accepted (404 is ignored, as the SDK does)."""
        if self._closed:
            raise self._fail(E_MCP, "MCP session is closed (" + method + ")")
        var stream = self._open("POST", rpc_notification_json(method), method, self.io_timeout_ms)
        var status = stream.status
        if status == 404 or (status >= 200 and status < 300):
            stream.close()
            return
        raise self._status_error(stream, method)

    # ------------------------------------------------------------------
    # MCP methods
    # ------------------------------------------------------------------
    def initialize(mut self) raises:
        """`initialize` (protocol 2025-11-25), check the answered version, remember the
        `mcp-session-id` of the answer, then `notifications/initialized`."""
        var result = self._request("initialize", initialize_params_json(MCP_CLIENT_NAME, MCP_CLIENT_VERSION))
        if not result.is_object():
            raise self._fail(E_MCP, "initialize result is not an object (" + result.kind_name() + ")")
        var pv = result.get("protocolVersion")
        var version = pv.as_string() if pv.is_string() else pv.to_json()
        if not mcp_supported_version(version):
            raise self._fail(E_MCP, "Unsupported protocol version from the server: " + _first_code_points(version, 100))
        self.protocol_version = version
        var info = result.get("serverInfo")
        if info.is_object():
            var nv = info.get("name")
            if nv.is_string():
                self.server_name = nv.as_string()
        self._notify("notifications/initialized")

    def list_tools(mut self) raises -> List[McpTool]:
        """All tools in server order, following `nextCursor` up to 1000 pages."""
        var out = List[McpTool]()
        var cursor = String()
        var pages = 0
        while True:
            pages += 1
            if pages > MAX_TOOL_PAGES:
                raise qa_error(E_MCP, "Reached max of 1000 iterations while listing tools.", "")
            var page = parse_tools_page(self._request("tools/list", tools_list_params_json(cursor)))
            for t in page[0]:
                out.append(t.copy())
            if page[1].byte_length() == 0:
                break
            cursor = page[1]
        return out^

    def call_tool(mut self, name: String, args_json: String) raises -> McpCallResult:
        """`tools/call {name, arguments}`; a tool failure is `is_error`, not a raise."""
        return parse_call_result(self._request("tools/call", call_params_json(name, args_json)))

    def close(mut self):
        """`DELETE` with the session id (5 s at most); every failure is ignored. Without a
        session id nothing is sent. A second call does nothing."""
        if self._closed:
            return
        self._closed = True
        if self.session_id.byte_length() == 0:
            return
        try:
            var req = self._build_request("DELETE", String())
            var opts = self._options(MCP_CLOSE_TIMEOUT_MS)
            opts.connect_timeout_ms = min(self.connect_timeout_ms, MCP_CLOSE_TIMEOUT_MS)
            var stream = self._http.open(req, opts)
            stream.close()
        except:
            pass
