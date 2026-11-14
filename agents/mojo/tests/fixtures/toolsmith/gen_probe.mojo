"""Probe tool for test_toolsmith_flow (not a real generated tool).

Each `mode` tries one thing that the CONF_TOOL confinement or the empty environment
must stop: reading the environment, reading or writing a path, opening a TCP
connection, running forever, flooding stderr. `cwd` writes a file in the working
directory, which must work.
"""

from std.ffi import external_call, c_int, get_errno
from std.io.file_descriptor import RawFileDescriptor
from std.memory import OpaquePointer
from std.os import getenv
from std.sys import CompilationTarget
from std.time import sleep

from qumo.ai.json import JsonValue
from qumo_agent.tool_sdk import ToolInfo, ToolOutput, ok, fail, arg_string, arg_int, serve_tool

comptime NAME = "gen_probe"
comptime SCHEMA = "{\"type\":\"object\",\"properties\":{\"mode\":{\"type\":\"string\"},\"path\":{\"type\":\"string\"},\"port\":{\"type\":\"integer\"}},\"required\":[\"mode\"]}"


def _errno() -> Int:
    return Int(get_errno().value)


def try_connect(port: Int) -> ToolOutput:
    var fd = external_call["socket", c_int](c_int(2), c_int(1), c_int(0))
    if Int(fd) < 0:
        return fail("socket failed errno " + String(_errno()))
    var addr = List[UInt8](length=16, fill=0)
    comptime if CompilationTarget.is_macos():
        addr[0] = 16      # sin_len
        addr[1] = 2       # AF_INET
    else:
        addr[0] = 2       # sin_family (little-endian u16)
    addr[2] = UInt8((port >> 8) & 0xFF)
    addr[3] = UInt8(port & 0xFF)
    addr[4] = 127
    addr[7] = 1
    var p = OpaquePointer[MutAnyOrigin](unsafe_from_address=Int(addr.unsafe_ptr()))
    var rc = external_call["connect", c_int](fd, p, c_int(16))
    var err = _errno()
    _ = len(addr)
    if Int(rc) != 0:
        return fail("connect failed errno " + String(err))
    return ok("connected")


def handle(args: JsonValue) raises -> ToolOutput:
    var mode = arg_string(args, "mode", "")
    var path = arg_string(args, "path", "")
    if mode == "env":
        return ok(getenv("HOME", "<unset>") + "|" + getenv("PATH", "<unset>"))
    if mode == "read":
        try:
            with open(path, "r") as f:
                var text = f.read()
                return ok("read " + String(text.byte_length()) + " bytes")
        except e:
            return fail("read failed: " + String(e))
    if mode == "write" or mode == "cwd":
        var target = path if mode == "write" else String("probe-out.txt")
        try:
            with open(target, "w") as f:
                f.write("probe")
            return ok("wrote " + target)
        except e:
            return fail("write failed: " + String(e))
    if mode == "socket":
        return try_connect(arg_int(args, "port", 9))
    if mode == "sleep":
        sleep(600.0)
        return ok("woke up")
    if mode == "stderr":
        var chunk = List[UInt8](length=65536, fill=0x65)   # "e"
        var fd = RawFileDescriptor(c_int(2))
        for _ in range(16):
            fd.write_bytes(Span(chunk))
        return fail("flood done")
    return fail("unknown mode " + mode)


def main() raises:
    serve_tool(ToolInfo(NAME, "Probe", "Test probe for the toolsmith confinement.", SCHEMA, "0.1.0"), handle)
