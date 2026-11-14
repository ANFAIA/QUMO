"""Tests for mcp/protocol.mojo, mcp/results.mojo and mcp/schema.mojo: request JSON,
tools pages with nextCursor, call results to outcomes (critic B6) and the inputSchema
transform (critic B7)."""

from std.testing import assert_equal, assert_true, assert_false

from qumo.ai.json import parse_json
from qumo_agent.mcp.protocol import (
    MCP_CLIENT_PROTOCOL, mcp_supported_version, rpc_request_json, rpc_notification_json,
    initialize_params_json, tools_list_params_json, call_params_json,
    McpTool, McpCallResult, parse_tools_page, parse_call_result,
)
from qumo_agent.mcp.results import call_result_to_outcome, content_block_text, MCP_EMPTY_ERROR_TEXT
from qumo_agent.mcp.schema import transform_input_schema


def _raises(text: String) -> String:
    try:
        _ = transform_input_schema(text, "")
    except e:
        return String(e)
    return String()


def _params(schema: String) raises -> String:
    return transform_input_schema(schema, "d")[0]


def test_messages() raises:
    assert_equal(MCP_CLIENT_PROTOCOL, "2025-11-25")
    for v in ["2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"]:
        assert_true(mcp_supported_version(v))
    assert_false(mcp_supported_version("2025-01-01"))
    assert_false(mcp_supported_version(""))
    # The exact initialize request Python sends (mojo_gaps 4).
    assert_equal(
        rpc_request_json(0, "initialize", initialize_params_json("mcp", "0.1.0")),
        "{\"jsonrpc\":\"2.0\",\"id\":0,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2025-11-25\",\"capabilities\":{},\"clientInfo\":{\"name\":\"mcp\",\"version\":\"0.1.0\"}}}",
    )
    assert_equal(rpc_notification_json("notifications/initialized"),
                 "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}")
    assert_equal(rpc_request_json(1, "tools/list", tools_list_params_json("")),
                 "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\"}")
    assert_equal(rpc_request_json(2, "tools/list", tools_list_params_json("c\"2")),
                 "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/list\",\"params\":{\"cursor\":\"c\\\"2\"}}")
    assert_equal(call_params_json("qumo_describe", "{\"topic\": \"gates\"}"),
                 "{\"name\":\"qumo_describe\",\"arguments\":{\"topic\": \"gates\"}}")
    assert_equal(call_params_json("x", ""), "{\"name\":\"x\",\"arguments\":{}}")


def test_tools_page() raises:
    var page = parse_tools_page(parse_json(
        "{\"tools\":[{\"name\":\"a\",\"title\":\"A\",\"description\":\"does a\",\"inputSchema\":{\"type\":\"object\",\"properties\":{\"x\":{\"type\":\"integer\"}}}},"
        + "{\"name\":\"b\",\"description\":null}],\"nextCursor\":\"page2\"}"
    ))
    var tools = page[0].copy()
    assert_equal(len(tools), 2)
    assert_equal(tools[0].name, "a")
    assert_equal(tools[0].title, "A")
    assert_equal(tools[0].description, "does a")
    assert_equal(tools[0].input_schema_json, "{\"type\":\"object\",\"properties\":{\"x\":{\"type\":\"integer\"}}}")
    assert_equal(tools[1].name, "b")
    assert_equal(tools[1].title, "")
    assert_equal(tools[1].description, "")
    assert_equal(tools[1].input_schema_json, "{\"type\":\"object\"}")
    assert_equal(page[1], "page2")
    var last = parse_tools_page(parse_json("{\"tools\":[]}"))
    assert_equal(len(last[0]), 0)
    assert_equal(last[1], "")
    var empty_cursor = parse_tools_page(parse_json("{\"tools\":[],\"nextCursor\":\"\"}"))
    assert_equal(empty_cursor[1], "")
    var raised = False
    try:
        _ = parse_tools_page(parse_json("{\"tools\":[{\"description\":\"no name\"}]}"))
    except e:
        raised = String(e).startswith("QA-E130: ")
    assert_true(raised)


def test_call_results() raises:
    # B6: two text parts stay two parts.
    var r = parse_call_result(parse_json(
        "{\"content\":[{\"type\":\"text\",\"text\":\"hello\"},{\"type\":\"text\",\"text\":\"world\"}],\"structuredContent\":{\"k\":1},\"isError\":false}"
    ))
    assert_false(r.is_error)
    assert_equal(len(r.parts), 2)
    assert_equal(r.structured_json, "{\"k\":1}")
    var o = call_result_to_outcome(r)
    assert_true(o.is_parts)
    assert_false(o.is_error)
    assert_equal(len(o.parts), 2)
    assert_equal(o.parts[0], "hello")
    assert_equal(o.parts[1], "world")
    assert_equal(o.joined_text(), "hello\nworld")
    # Error with empty content: the adapter's single placeholder part.
    var err = call_result_to_outcome(parse_call_result(parse_json("{\"content\":[],\"isError\":true}")))
    assert_true(err.is_error)
    assert_true(err.is_parts)
    assert_equal(len(err.parts), 1)
    assert_equal(err.parts[0], "MCP tool returned an error with empty content.")
    assert_equal(err.parts[0], MCP_EMPTY_ERROR_TEXT)
    # Error with content keeps it.
    var err2 = call_result_to_outcome(parse_call_result(parse_json(
        "{\"content\":[{\"type\":\"text\",\"text\":\"QM-E101: bad\"}],\"isError\":true}")))
    assert_equal(len(err2.parts), 1)
    assert_equal(err2.parts[0], "QM-E101: bad")
    # Success with no content: an empty part list ("content": [] on the wire).
    var ok = call_result_to_outcome(parse_call_result(parse_json("{\"content\":[]}")))
    assert_true(ok.is_parts)
    assert_false(ok.is_error)
    assert_equal(len(ok.parts), 0)
    var no_content = parse_call_result(parse_json("{\"structuredContent\":null}"))
    assert_equal(len(no_content.parts), 0)
    assert_equal(no_content.structured_json, "")
    # Non-text blocks become placeholders (X9); embedded text resources give their text.
    var mixed = parse_call_result(parse_json(
        "{\"content\":[{\"type\":\"image\",\"data\":\"AAAA\",\"mimeType\":\"image/png\"},"
        + "{\"type\":\"resource\",\"resource\":{\"uri\":\"file:///a\",\"text\":\"inside\"}},"
        + "{\"type\":\"resource\",\"resource\":{\"uri\":\"file:///b\",\"blob\":\"AAAA\"}},"
        + "{\"type\":\"audio\",\"data\":\"AAAA\",\"mimeType\":\"audio/wav\"},"
        + "{\"type\":\"resource_link\",\"uri\":\"file:///c\",\"name\":\"c\"}]}"
    ))
    assert_equal(len(mixed.parts), 5)
    assert_equal(mixed.parts[0], "[image content omitted: qumo-agent passes text only]")
    assert_equal(mixed.parts[1], "inside")
    assert_equal(mixed.parts[2], "[resource content omitted: qumo-agent passes text only]")
    assert_equal(mixed.parts[3], "[audio content omitted: qumo-agent passes text only]")
    assert_equal(mixed.parts[4], "[resource_link content omitted: qumo-agent passes text only]")
    assert_equal(content_block_text(parse_json("{\"type\":\"text\",\"text\":\"t\"}")), "t")


def test_schema_identity() raises:
    # QUMO's schemas have no titles: the transform is the identity (key order kept).
    var qumo = "{\"type\":\"object\",\"properties\":{\"qasm\":{\"type\":\"string\",\"description\":\"OpenQASM\"},\"shots\":{\"type\":\"integer\",\"minimum\":1}},\"required\":[\"qasm\"],\"additionalProperties\":false}"
    var t = transform_input_schema(qumo, "Runs a circuit.")
    assert_equal(t[0], qumo)
    assert_equal(t[1], "Runs a circuit.")


def test_schema_b7() raises:
    var schema = (
        "{\"$schema\":\"http://json-schema.org/draft-07/schema#\",\"title\":\"Args\",\"description\":\"Schema text\","
        + "\"type\":\"object\",\"properties\":{"
        + "\"title\":{\"type\":\"string\",\"title\":\"Title\"},"
        + "\"item\":{\"$ref\":\"#/$defs/Item\",\"description\":\"an item\"},"
        + "\"tags\":{\"type\":\"array\",\"items\":{\"title\":\"Tag\",\"type\":\"string\"}},"
        + "\"choice\":{\"oneOf\":[{\"title\":\"A\",\"type\":\"string\"},{\"type\":\"null\"}]}},"
        + "\"required\":[\"title\"],\"additionalProperties\":false,"
        + "\"$defs\":{\"Item\":{\"title\":\"Item\",\"type\":\"object\",\"properties\":{\"n\":{\"title\":\"N\",\"type\":\"integer\"}}}}}"
    )
    var t = transform_input_schema(schema, "")
    var expected = (
        "{\"$schema\":\"http://json-schema.org/draft-07/schema#\",\"type\":\"object\",\"properties\":{"
        + "\"title\":{\"type\":\"string\"},"
        + "\"item\":{\"type\":\"object\",\"properties\":{\"n\":{\"type\":\"integer\"}},\"description\":\"an item\"},"
        + "\"tags\":{\"type\":\"array\",\"items\":{\"type\":\"string\"}},"
        + "\"choice\":{\"oneOf\":[{\"title\":\"A\",\"type\":\"string\"},{\"type\":\"null\"}]}},"
        + "\"required\":[\"title\"],\"additionalProperties\":false}"
    )
    assert_equal(t[0], expected)
    # The popped top-level description is used only when the tool has none.
    assert_equal(t[1], "Schema text")
    assert_equal(transform_input_schema(schema, "Tool text")[1], "Tool text")


def test_schema_refs() raises:
    # `definitions` is dropped too; an extra key next to $ref overrides in place.
    assert_equal(
        _params("{\"type\":\"object\",\"properties\":{\"a\":{\"$ref\":\"#/definitions/A\",\"type\":\"number\"}},\"definitions\":{\"A\":{\"type\":\"string\",\"minLength\":1}}}"),
        "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"number\",\"minLength\":1}}}",
    )
    # A recursive definition is cut where it repeats (only the extra keys stay).
    assert_equal(
        _params("{\"type\":\"object\",\"properties\":{\"node\":{\"$ref\":\"#/$defs/N\"}},\"$defs\":{\"N\":{\"type\":\"object\",\"properties\":{\"next\":{\"$ref\":\"#/$defs/N\",\"description\":\"tail\"}}}}}"),
        "{\"type\":\"object\",\"properties\":{\"node\":{\"type\":\"object\",\"properties\":{\"next\":{\"description\":\"tail\"}}}}}",
    )
    # Refs inside arrays are resolved; array indices in a path work.
    assert_equal(
        _params("{\"type\":\"object\",\"properties\":{\"u\":{\"anyOf\":[{\"$ref\":\"#/$defs/L/1\"}]}},\"$defs\":{\"L\":[{\"type\":\"string\"},{\"type\":\"integer\"}]}}"),
        "{\"type\":\"object\",\"properties\":{\"u\":{\"anyOf\":[{\"type\":\"integer\"}]}}}",
    )
    # A title key whose value is not an object is removed even under properties.
    assert_equal(
        _params("{\"type\":\"object\",\"properties\":{\"title\":\"x\",\"t\":{\"type\":\"string\"}}}"),
        "{\"type\":\"object\",\"properties\":{\"t\":{\"type\":\"string\"}}}",
    )
    # A repeated key keeps its first position and its last value (Python dict).
    assert_equal(_params("{\"type\":\"object\",\"a\":1,\"type\":\"string\"}"), "{\"type\":\"string\",\"a\":1}")
    # Errors.
    assert_true(_raises("{\"properties\":{\"a\":{\"$ref\":\"#/$defs/Missing\"}}}").startswith(
        "QA-E130: invalid inputSchema: Reference '#/$defs/Missing' not found."))
    assert_true(_raises("{\"properties\":{\"a\":{\"$ref\":\"other.json#/x\"}}}").startswith(
        "QA-E130: invalid inputSchema: ref paths are expected to be URI fragments"))
    assert_true(_raises("[1]").startswith("QA-E130: invalid inputSchema: expected a JSON object"))
    assert_true(_raises("{").startswith("QA-E130: invalid inputSchema: "))


def main() raises:
    test_messages()
    test_tools_page()
    test_call_results()
    test_schema_identity()
    test_schema_b7()
    test_schema_refs()
    print("test_mcp_protocol OK")
