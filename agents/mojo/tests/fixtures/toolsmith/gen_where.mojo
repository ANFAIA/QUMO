"""Test tool for test_toolsmith_generated (not a real generated tool).

`mode` picks what it reports: `exe` its own program path (argv[0], to show which
bytes were started), `echo` the `text` argument, `big` a text of `n` bytes, `fail`
an error result. The schema carries a top-level `title` and a property `title` so the
tests can check the schema transform and `count_json`.
"""

from std.sys import argv

from qumo.ai.json import JsonValue
from qumo_agent.tool_sdk import ToolInfo, ToolOutput, ok, fail, arg_string, arg_int, serve_tool

comptime NAME = "gen_where"
comptime SCHEMA = "{\"type\":\"object\",\"title\":\"Where args\",\"description\":\"top\",\"properties\":{\"mode\":{\"type\":\"string\",\"title\":\"Mode\"},\"text\":{\"type\":\"string\"},\"n\":{\"type\":\"integer\"}},\"required\":[\"mode\"]}"


def handle(args: JsonValue) raises -> ToolOutput:
    var mode = arg_string(args, "mode", "")
    if mode == "exe":
        return ok(String(argv()[0]))
    if mode == "echo":
        return ok(arg_string(args, "text", ""))
    if mode == "big":
        var n = arg_int(args, "n", 0)
        return ok(String("x") * n)
    if mode == "fail":
        return fail("deliberate failure")
    return fail("unknown mode " + mode)


def main() raises:
    serve_tool(ToolInfo(NAME, "Where", "Reports where it runs.", SCHEMA, "1.0.0"), handle)
