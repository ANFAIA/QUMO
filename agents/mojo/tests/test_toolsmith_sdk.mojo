"""The tool SDK (DESIGN.md 6.2): the protocol handled in-process, then the reference
tool built hermetically and called over MCP with an empty environment, directly and
through run_cases. Offline; builds two small binaries with .venv/bin/mojo."""

from std.testing import assert_equal, assert_true, assert_false

from qumo.ai.json import JsonValue, parse_json

from qumo_agent.native.fs import (
    exists, file_mode, is_file, join_path, make_dirs, read_text_lossy, realpath_strict, remove_tree,
)
from qumo_agent.native.sandbox import sandbox_kind
from qumo_agent.native.spawn import (
    SpawnSpec, Child, spawn, run_capture, READ_LINE, READ_EOF, STDIO_NULL, STDIO_PIPE, minimal_child_env,
)
from qumo_agent.util.sha256 import sha256_file_hex
from qumo_agent.tool_sdk import (
    ToolInfo, ToolOutput, ok, fail, ok_json, arg_string, arg_int, arg_float, arg_bool,
    handle_line, call_result_json, too_long_reply, SDK_MAX_LINE_BYTES,
)
from qumo_agent.toolsmith.tools import hermetic_build, run_cases

comptime SCHEMA = "{\"type\":\"object\",\"properties\":{\"mode\":{\"type\":\"string\"}}}"


def handler(args: JsonValue) raises -> ToolOutput:
    var mode = arg_string(args, "mode", "ok")
    if mode == "ok":
        return ok(arg_string(args, "text", "default text"))
    if mode == "fail":
        return fail("bad input")
    if mode == "json":
        return ok_json("{\"a\": 1, \"b\": [true]}")
    if mode == "pretty":
        return ok_json("{\n  \"a\": 1\n}")
    if mode == "notobject":
        return ok_json("[1, 2]")
    if mode == "raise":
        raise Error("boom")
    if mode == "types":
        var s = arg_string(args, "s", "d")
        var i = arg_int(args, "i", 7)
        var f = arg_float(args, "f", 0.5)
        var b = arg_bool(args, "b", True)
        return ok(s + "|" + String(i) + "|" + String(f) + "|" + String(b))
    return fail("unknown mode")


def info() -> ToolInfo:
    return ToolInfo("gen_sdk_test", "SDK test", "Exercises the SDK.", SCHEMA, "1.2.3")


def call_line(id: Int, name: String, args_json: String) -> String:
    return "{\"jsonrpc\":\"2.0\",\"id\":" + String(id) + ",\"method\":\"tools/call\",\"params\":{\"name\":\"" + name + "\",\"arguments\":" + args_json + "}}"


def reply(line: String) raises -> JsonValue:
    return parse_json(handle_line(info(), handler, line))


def result_text(r: JsonValue) raises -> String:
    return r.get("result").get("content").at(0).get("text").as_string()


def test_protocol() raises:
    var r = reply("{\"jsonrpc\":\"2.0\",\"id\":0,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2025-11-25\",\"capabilities\":{},\"clientInfo\":{\"name\":\"t\",\"version\":\"1\"}}}")
    assert_equal(r.get("id").as_int(), 0)
    assert_equal(r.get("result").get("protocolVersion").as_string(), "2025-11-25")
    assert_equal(r.get("result").get("serverInfo").to_json(), "{\"name\":\"gen_sdk_test\",\"title\":\"SDK test\",\"version\":\"1.2.3\"}")
    assert_equal(r.get("result").get("capabilities").to_json(), "{\"tools\":{\"listChanged\":false}}")
    r = reply("{\"jsonrpc\":\"2.0\",\"id\":\"x\",\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"1999-01-01\"}}")
    assert_equal(r.get("id").as_string(), "x")
    assert_equal(r.get("result").get("protocolVersion").as_string(), "2025-06-18")

    assert_equal(handle_line(info(), handler, "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}"), "")
    assert_equal(handle_line(info(), handler, "   "), "")
    assert_equal(reply("{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}").get("result").to_json(), "{}")

    r = reply("{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/list\"}")
    assert_equal(r.get("result").to_json(), "{\"tools\":[{\"name\":\"gen_sdk_test\",\"title\":\"SDK test\",\"description\":\"Exercises the SDK.\",\"inputSchema\":" + String(SCHEMA) + "}]}")

    r = reply(call_line(3, "gen_sdk_test", "{\"text\":\"héllo\"}"))
    assert_equal(r.get("result").to_json(), "{\"content\":[{\"type\":\"text\",\"text\":\"héllo\"}],\"isError\":false}")
    r = reply(call_line(4, "gen_sdk_test", "{\"mode\":\"fail\"}"))
    assert_equal(r.get("result").to_json(), "{\"content\":[{\"type\":\"text\",\"text\":\"bad input\"}],\"isError\":true}")
    r = reply(call_line(5, "gen_sdk_test", "{\"mode\":\"json\"}"))
    assert_equal(r.get("result").get("structuredContent").to_json(), "{\"a\":1,\"b\":[true]}")
    assert_equal(result_text(r), "{\"a\": 1, \"b\": [true]}")
    assert_false(r.get("result").get("isError").as_bool())
    var pretty = handle_line(info(), handler, call_line(51, "gen_sdk_test", "{\"mode\":\"pretty\"}"))
    assert_equal(pretty.find("\n"), -1)   # one message per line, whatever the tool returns
    r = parse_json(pretty)
    assert_equal(r.get("result").get("structuredContent").to_json(), "{\"a\":1}")
    assert_equal(result_text(r), "{\n  \"a\": 1\n}")
    r = reply(call_line(6, "gen_sdk_test", "{\"mode\":\"notobject\"}"))
    assert_true(r.get("result").get("isError").as_bool())
    assert_false(r.get("result").has("structuredContent"))
    assert_true(result_text(r).startswith("error: ok_json needs a JSON object"))
    r = reply(call_line(7, "gen_sdk_test", "{\"mode\":\"raise\"}"))
    assert_equal(r.get("result").to_json(), "{\"content\":[{\"type\":\"text\",\"text\":\"error: boom\"}],\"isError\":true}")

    # Argument helpers: defaults for missing and null, errors for wrong types.
    assert_equal(result_text(reply(call_line(8, "gen_sdk_test", "{\"mode\":\"types\"}"))), "d|7|0.5|True")
    assert_equal(result_text(reply(call_line(9, "gen_sdk_test", "{\"mode\":\"types\",\"s\":null,\"i\":3,\"f\":2,\"b\":false}"))), "d|3|2.0|False")
    assert_equal(result_text(reply(call_line(10, "gen_sdk_test", "{\"mode\":\"types\",\"i\":\"3\"}"))), "error: argument 'i' must be an integer, got string")
    assert_equal(result_text(reply(call_line(11, "gen_sdk_test", "{\"mode\":\"types\",\"i\":1.5}"))), "error: argument 'i' must be an integer, got 1.5")
    assert_equal(result_text(reply(call_line(12, "gen_sdk_test", "{\"mode\":\"types\",\"b\":1}"))), "error: argument 'b' must be a boolean, got number")
    assert_equal(result_text(reply(call_line(13, "gen_sdk_test", "{\"mode\":7}"))), "error: argument 'mode' must be a string, got number")
    # Missing or null arguments are {}.
    assert_equal(result_text(reply("{\"jsonrpc\":\"2.0\",\"id\":14,\"method\":\"tools/call\",\"params\":{\"name\":\"gen_sdk_test\"}}")), "default text")
    assert_equal(result_text(reply(call_line(15, "gen_sdk_test", "null"))), "default text")

    # Protocol errors.
    r = reply(call_line(16, "gen_other", "{}"))
    assert_equal(r.get("error").get("code").as_int(), -32602)
    assert_equal(r.get("error").get("message").as_string(), "Unknown tool: gen_other")
    assert_equal(reply(call_line(17, "gen_sdk_test", "[1]")).get("error").get("code").as_int(), -32602)
    assert_equal(reply("{\"jsonrpc\":\"2.0\",\"id\":18,\"method\":\"tools/call\",\"params\":5}").get("error").get("code").as_int(), -32602)
    r = reply("{\"jsonrpc\":\"2.0\",\"id\":19,\"method\":\"resources/list\"}")
    assert_equal(r.get("error").get("code").as_int(), -32601)
    r = reply("not json")
    assert_equal(r.get("error").get("code").as_int(), -32700)
    assert_true(r.get("id").is_null())
    assert_equal(reply("[1]").get("error").get("code").as_int(), -32600)
    assert_equal(reply("{\"jsonrpc\":\"2.0\",\"id\":20}").get("error").get("code").as_int(), -32600)
    r = parse_json(too_long_reply())
    assert_equal(r.get("error").get("code").as_int(), -32600)
    assert_true(r.get("id").is_null())


def test_truncation() raises:
    # 600,000 two-byte characters: cut at a code-point boundary below 1,000,000 bytes.
    var big = String("é") * 600000
    var r = parse_json(call_result_json(ok(big)))
    var t = r.get("content").at(0).get("text").as_string()
    assert_true(t.endswith("\n... [output truncated]"))
    var kept = String(t[byte=0:t.byte_length() - 23])
    assert_equal(kept.byte_length(), 1000000)
    assert_equal(kept, String("é") * 500000)
    # Three-byte characters: 1,000,000 is not a boundary, so it steps back.
    var big3 = String("€") * 400000
    t = parse_json(call_result_json(ok(big3))).get("content").at(0).get("text").as_string()
    assert_equal(String(t[byte=0:t.byte_length() - 23]), String("€") * 333333)
    # Exactly at the cap: unchanged.
    var exact = String("a") * 1000000
    assert_equal(parse_json(call_result_json(ok(exact))).get("content").at(0).get("text").as_string(), exact)


def build_helper(root: String, name: String) raises -> String:
    """Build tests/fixtures/toolsmith/confine_helper.mojo (the `__confine` trampoline
    the toolsmith needs) into build/test-fixtures/<name>."""
    var out_dir = join_path(root, "build/test-fixtures")
    make_dirs(out_dir)
    var out = join_path(out_dir, name)
    var mojo = join_path(root, ".venv/bin/mojo")
    var spec = SpawnSpec(mojo, [mojo, "build", "-I", root, "-I", join_path(root, "agents/mojo"), "-o", out,
                                join_path(root, "agents/mojo/tests/fixtures/toolsmith/confine_helper.mojo")],
                         minimal_child_env(), root, STDIO_NULL, STDIO_PIPE, STDIO_PIPE, True)
    var res = run_capture(spec, 600000, 1048576, String())
    if res.exit_code != 0:
        raise Error("building the confine helper failed:\n" + res.stdout + res.stderr)
    return out


def send(mut child: Child, line: String) raises:
    var data = line + "\n"
    child.write_all(data.as_bytes())


def recv(mut child: Child) raises -> JsonValue:
    var line = String()
    var rc = child.read_line(line, 30000, 8388608)
    assert_equal(rc, READ_LINE)
    return parse_json(line)


def test_reference_binary(root: String, helper: String) raises:
    var kind = sandbox_kind(helper)
    var confined = kind != "none"
    var src = join_path(root, "agents/mojo/tools/gen_bitstring_stats/tool.mojo")
    var res = hermetic_build(root, join_path(root, ".venv/bin/mojo"), src, "gen_bitstring_stats", 600,
                             helper if confined else String())
    var bin = res[2]
    var tmp = String(bin[byte=0:bin.rfind("/")])
    assert_equal(res[0], 0, res[1])
    assert_true(is_file(bin))
    var rec = res[3].copy()
    assert_equal(rec.source_sha256, sha256_file_hex(src))
    assert_true(rec.mojo_version.startswith("Mojo "))
    assert_equal(rec.binary_sha256, "")

    # Called directly over MCP with an empty environment.
    var spec = SpawnSpec(bin, [bin], List[String](), tmp, STDIO_PIPE, STDIO_PIPE, STDIO_PIPE, True)
    var child = spawn(spec)
    send(child, "{\"jsonrpc\":\"2.0\",\"id\":0,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2025-11-25\",\"capabilities\":{},\"clientInfo\":{\"name\":\"t\",\"version\":\"1\"}}}")
    assert_equal(recv(child).get("result").get("serverInfo").get("name").as_string(), "gen_bitstring_stats")
    send(child, "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}")
    send(child, "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\"}")
    var tools = recv(child).get("result").get("tools")
    assert_equal(len(tools), 1)
    assert_equal(tools.at(0).get("name").as_string(), "gen_bitstring_stats")
    send(child, call_line(2, "gen_bitstring_stats", "{\"counts\":\"{\\\"00\\\": 500, \\\"11\\\": 500}\"}"))
    var r = recv(child).get("result")
    assert_false(r.get("isError").as_bool())
    assert_equal(r.get("structuredContent").to_json(), "{\"shots\":1000,\"outcomes\":2,\"width\":2,\"entropy_bits\":1.0,\"parity_even\":1.0,\"parity_odd\":0.0,\"most_frequent\":\"00\"}")
    send(child, call_line(3, "gen_bitstring_stats", "{\"counts\":\"nope\"}"))
    assert_true(recv(child).get("result").get("isError").as_bool())
    send(child, call_line(4, "gen_wrong_name", "{}"))
    assert_equal(recv(child).get("error").get("code").as_int(), -32602)
    # A line above 4 MiB is refused, and the tool keeps serving.
    var huge = String("{\"jsonrpc\":\"2.0\",\"id\":5,\"method\":\"ping\",\"pad\":\"") + String("x") * (SDK_MAX_LINE_BYTES + 10) + "\"}"
    send(child, huge)
    var too_long = recv(child)
    assert_equal(too_long.get("error").get("code").as_int(), -32600)
    send(child, "{\"jsonrpc\":\"2.0\",\"id\":6,\"method\":\"ping\"}")
    assert_equal(recv(child).get("id").as_int(), 6)
    child.close_stdin()
    var line = String()
    assert_equal(child.read_line(line, 10000, 1024), READ_EOF)
    assert_equal(child.wait(), 0)
    assert_equal(child.stderr_tail(), "")

    # The same binary through run_cases (confined when an OS sandbox exists).
    var cases = read_text_lossy(join_path(root, "agents/mojo/tools/gen_bitstring_stats/cases.json"))
    var rc = run_cases(bin, "gen_bitstring_stats", cases, 30, helper, confined)
    assert_equal(rc[0], rc[1], rc[2])
    assert_equal(rc[1], 7)
    assert_true(rc[2].startswith("- bell: ok ("), rc[2])
    remove_tree(tmp)
    if not confined:
        print("note: no OS sandbox on this machine; the reference tool ran unconfined")


def main() raises:
    test_protocol()
    test_truncation()
    var root = realpath_strict(".")
    var helper = build_helper(root, "toolsmith_confine_sdk")
    test_reference_binary(root, helper)
    print("test_toolsmith_sdk OK")
