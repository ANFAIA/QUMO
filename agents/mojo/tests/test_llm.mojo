"""Chat-completions request building and response parsing, redaction, truncation."""

from std.testing import assert_equal, assert_true, assert_false, assert_raises

from qumo.ai.json import parse_json
from qumo_agent.llm import (build_request, parse_response, message_json, tool_message_json,
                            function_tool_json, redact, truncate_utf8, ChatResponse, ToolCall)

comptime CANNED = """{"id":"gen-1","object":"chat.completion","model":"anthropic/claude-sonnet-5.5","choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":"Let me validate it.","tool_calls":[{"id":"toolu_01","type":"function","function":{"name":"qumo_validate_qasm","arguments":"{\\"qasm\\": \\"OPENQASM 2.0; qreg q[1];\\"}"}},{"id":"toolu_02","type":"function","function":{"name":"qumo_describe","arguments":"{\\"topic\\":\\"error\\",\\"name\\":\\"QM-E521\\"}"}}],"reasoning_details":[{"type":"reasoning.text","text":"hm"}]}}],"usage":{"prompt_tokens":1234,"completion_tokens":56,"total_tokens":1290}}"""


def main() raises:
    var r = parse_response(CANNED)
    assert_equal(r.content, "Let me validate it.")
    assert_equal(r.finish_reason, "tool_calls")
    assert_equal(len(r.tool_calls), 2)
    assert_equal(r.tool_calls[0].id, "toolu_01")
    assert_equal(r.tool_calls[0].name, "qumo_validate_qasm")
    assert_equal(parse_json(r.tool_calls[0].arguments).get_string("qasm", ""), "OPENQASM 2.0; qreg q[1];")
    assert_equal(r.tool_calls[1].name, "qumo_describe")
    assert_equal(r.prompt_tokens, 1234)
    assert_equal(r.completion_tokens, 56)
    assert_true(r.reasoning_details.find("reasoning.text") >= 0)

    # The assistant turn we send back keeps the calls and arguments verbatim.
    var am = parse_json(r.assistant_message_json())
    assert_equal(am.get_string("role", ""), "assistant")
    assert_equal(am.get("tool_calls").at(1).get("function").get_string("arguments", ""), r.tool_calls[1].arguments)

    # Final answer with null content and no tool calls.
    var fin = parse_response("""{"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":null}}]}""")
    assert_false(fin.has_content)
    assert_equal(len(fin.tool_calls), 0)

    # An error body raises, with key-like text redacted.
    with assert_raises(contains="REDACTED"):
        _ = parse_response("""{"error":{"code":401,"message":"bad key sk-or-v1-0123456789abcdef"}}""")

    # Build a request and parse it back.
    var msgs = List[String]()
    msgs.append(message_json("system", "You are QUMO.\nLine 2 \"quoted\" ü"))
    msgs.append(message_json("user", "Run a Bell state"))
    msgs.append(r.assistant_message_json())
    msgs.append(tool_message_json("toolu_01", "{\"ok\":true}"))
    var tools = List[String]()
    tools.append(function_tool_json("qumo_run_qasm", "Use when ...\n\nServer text", "{\"type\":\"object\",\"properties\":{\"qasm\":{\"type\":\"string\"}},\"required\":[\"qasm\"]}"))
    var req = parse_json(build_request("anthropic/claude-sonnet-5.5", msgs, tools, 4096))
    assert_equal(req.get_string("model", ""), "anthropic/claude-sonnet-5.5")
    assert_equal(req.get_int("max_tokens", 0), 4096)
    assert_equal(len(req.get("messages")), 4)
    assert_equal(req.get("messages").at(0).get_string("content", ""), "You are QUMO.\nLine 2 \"quoted\" ü")
    assert_equal(req.get("messages").at(3).get_string("tool_call_id", ""), "toolu_01")
    var f = req.get("tools").at(0).get("function")
    assert_equal(f.get_string("name", ""), "qumo_run_qasm")
    assert_equal(f.get("parameters").get("required").at(0).as_string(), "qasm")

    # Redaction and truncation.
    assert_equal(redact("Authorization: Bearer abcdefghijkl, ok"), "Authorization: Bearer [REDACTED], ok")
    assert_equal(redact("my secret-value-123 here", "secret-value-123"), "my [REDACTED] here")
    assert_equal(redact("task-runner sk-short"), "task-runner sk-short")
    var t = truncate_utf8("ééé", 3)  # 6 bytes; byte 3 is inside the second é
    assert_true(t.startswith("é\n... [truncated: 6 bytes"))
    print("test_llm OK")
