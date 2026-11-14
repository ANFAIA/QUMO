"""Scripted plain-HTTP server for tests (DESIGN.md 2.4.7). Test-only program.

Usage:
    http_fixture_server --script SCRIPT.json --port-file PORT_FILE --log LOG.jsonl

Binds 127.0.0.1:0, writes the port (atomically: a temporary file renamed onto
PORT_FILE) and serves the script's responses in order, one connection per request:

    {"responses": [
      {"match": {"method": "POST", "path": "/api/v1/chat/completions"},
       "status": 429, "reason": "Too Many Requests", "headers": {"retry-after-ms": "50"},
       "body": "...", "framing": "content-length", "chunks": [], "delay_ms": 0,
       "close_after_headers": false}
    ]}

- `match` (optional): `method` and/or `path`. A `path` without '?' is compared with
  the request path without its query. A request that does not match the next
  scripted response gets `500 no scripted response` and does not consume it.
- `framing`: `content-length` (default), `chunked` (the body split into `chunks` when
  given, else one chunk) or `close` (no length; the connection is closed after the
  body). `delay_ms` pauses before each piece of the body (each chunk, or the whole
  body), after the head was sent. `close_after_headers` sends the head (with its
  framing headers) and closes at once.
- Each request is appended to LOG.jsonl as
  {"method", "path", "headers": {lower-case name: value}, "body"}.

The server exits after the script is consumed, after 60 s without a connection, or
on SIGTERM.
"""

from std.sys import argv
from std.os.path import exists

from qumo.ai.json import JsonValue, JsonWriter, parse_json
from qumo_agent.native.libc import CStr, c_rename
from qumo_agent.native.clock import sleep_ms
from qumo_agent.net.socket import TcpListener, NetStream

comptime ACCEPT_TIMEOUT_MS = 60000
comptime IO_TIMEOUT_MS = 30000


def _arg(args: List[String], name: String) raises -> String:
    for i in range(len(args) - 1):
        if args[i] == name:
            return args[i + 1].copy()
    raise Error("usage: http_fixture_server --script SCRIPT.json --port-file PORT_FILE --log LOG.jsonl (missing " + name + ")")


def _read_text(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def _lower(s: String) -> String:
    return s.lower()


@fieldwise_init
struct Request(Copyable):
    var method: String
    var path: String
    var names: List[String]
    var values: List[String]
    var body: String


def _find(buf: List[UInt8], pat: String, start: Int) -> Int:
    var p = pat.as_bytes()
    var n = len(p)
    var i = start
    while i + n <= len(buf):
        var ok = True
        for k in range(n):
            if buf[i + k] != p[k]:
                ok = False
                break
        if ok:
            return i
        i += 1
    return -1


def _read_request(mut s: NetStream) raises -> Optional[Request]:
    var buf = List[UInt8]()
    var end = -1
    while end < 0:
        var n = s.read_some(buf, IO_TIMEOUT_MS)
        if n == 0:
            if len(buf) == 0:
                return None
            raise Error("connection closed inside the request head")
        end = _find(buf, "\r\n\r\n", 0)
    var head = String(from_utf8_lossy=Span(buf)[0:end])
    var lines = head.split("\r\n")
    var first = String(lines[0]).split(" ")
    if len(first) < 3:
        raise Error("bad request line: " + String(lines[0]))
    var names = List[String]()
    var values = List[String]()
    var length = 0
    for i in range(1, len(lines)):
        var line = String(lines[i])
        var c = line.find(":")
        if c <= 0:
            continue
        var name = _lower(String(String(line[byte=0:c]).strip()))
        var value = String(String(line[byte=c + 1:]).strip())
        if name == "content-length":
            length = Int(value)
        names.append(name^)
        values.append(value^)
    var body_start = end + 4
    while len(buf) - body_start < length:
        if s.read_some(buf, IO_TIMEOUT_MS) == 0:
            raise Error("connection closed inside the request body")
    var body = String(from_utf8_lossy=Span(buf)[body_start:body_start + length])
    return Request(String(first[0]), String(first[1]), names^, values^, body^)


def _log(path: String, req: Request) raises:
    var w = JsonWriter()
    w.begin_object()
    w.kv_string("method", req.method)
    w.kv_string("path", req.path)
    w.key("headers")
    w.begin_object()
    for i in range(len(req.names)):
        w.kv_string(req.names[i], req.values[i])
    w.end_object()
    w.kv_string("body", req.body)
    w.end_object()
    var line = w.finish() + "\n"
    var old = String()
    if exists(path):
        old = _read_text(path)
    with open(path, "w") as f:
        f.write(old + line)


def _matches(spec: JsonValue, req: Request) raises -> Bool:
    var m = spec.get("match")
    if not m.is_defined() or not m.is_object():
        return True
    var method = m.get("method")
    if method.is_defined() and method.as_string() != req.method:
        return False
    var path = m.get("path")
    if path.is_defined():
        var want = path.as_string()
        var got = req.path.copy()
        if want.find("?") < 0:
            var q = got.find("?")
            if q >= 0:
                got = String(req.path[byte=0:q])
        if want != got:
            return False
    return True


def _write_response(mut s: NetStream, spec: JsonValue) raises:
    var status = spec.get_int("status", 200)
    var reason = spec.get_string("reason", "OK")
    var body = spec.get_string("body", "")
    var framing = spec.get_string("framing", "content-length")
    var delay = spec.get_int("delay_ms", 0)
    var close_early = spec.get_bool("close_after_headers", False)
    var chunks = List[String]()
    var cv = spec.get("chunks")
    if cv.is_defined() and cv.is_array():
        for i in range(len(cv)):
            chunks.append(cv.at(i).as_string())
    if len(chunks) == 0 and body.byte_length() > 0:
        chunks.append(body)
    var head = "HTTP/1.1 " + String(status) + " " + reason + "\r\n"
    var hv = spec.get("headers")
    if hv.is_defined() and hv.is_object():
        for i in range(len(hv)):
            head += hv.key_at(i) + ": " + hv.at(i).as_string() + "\r\n"
    if framing == "chunked":
        head += "Transfer-Encoding: chunked\r\n"
    elif framing == "content-length":
        var total = 0
        for c in chunks:
            total += c.byte_length()
        head += "Content-Length: " + String(total) + "\r\n"
    head += "Connection: close\r\n\r\n"
    s.write_all(head, IO_TIMEOUT_MS)
    if close_early:
        return
    if framing == "content-length" and delay > 0:
        sleep_ms(delay)
        var all = String()
        for c in chunks:
            all += c
        s.write_all(all, IO_TIMEOUT_MS)
        return
    for c in chunks:
        if delay > 0:
            sleep_ms(delay)
        if framing == "chunked":
            s.write_all(String(hex(c.byte_length())[byte=2:]) + "\r\n" + c + "\r\n", IO_TIMEOUT_MS)
        else:
            s.write_all(c, IO_TIMEOUT_MS)
    if framing == "chunked":
        s.write_all("0\r\n\r\n", IO_TIMEOUT_MS)


def _write_port(path: String, port: Int) raises:
    var tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(String(port) + "\n")
    var a = CStr(tmp)
    var b = CStr(path)
    var r = c_rename(a.addr(), b.addr())
    a.keep()
    b.keep()
    if r != 0:
        raise Error("cannot write the port file " + path)


def main() raises:
    var args = List[String]()
    for a in argv():
        args.append(String(a))
    var script = parse_json(_read_text(_arg(args, "--script")))
    var port_file = _arg(args, "--port-file")
    var log_path = _arg(args, "--log")
    var responses = script.get("responses")
    var listener = TcpListener("127.0.0.1", 0)
    _write_port(port_file, listener.port())
    var next = 0
    while next < len(responses):
        var conn: NetStream
        try:
            conn = listener.accept(ACCEPT_TIMEOUT_MS)
        except:
            return
        try:
            var req = _read_request(conn)
            if not req:
                conn.close()
                continue
            _log(log_path, req.value())
            var spec = responses.at(next)
            if _matches(spec, req.value()):
                next += 1
                _write_response(conn, spec)
            else:
                var msg = "no scripted response"
                conn.write_all("HTTP/1.1 500 Internal Server Error\r\nContent-Length: " + String(msg.byte_length())
                               + "\r\nConnection: close\r\n\r\n" + msg, IO_TIMEOUT_MS)
        except e:
            print("http_fixture_server:", e)
        conn.close()
