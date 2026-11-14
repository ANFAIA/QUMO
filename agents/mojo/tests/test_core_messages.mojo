"""Tests for core/messages.mojo, core/toolspec.mojo and core/errors.mojo."""

from std.testing import assert_equal, assert_true, assert_false

from qumo.ai.json import JsonValue, parse_json
from qumo_agent.core.errors import *
from qumo_agent.core.toolspec import *
from qumo_agent.core.messages import *
from qumo_agent.util.pyfmt import py_json_dumps


def _wire(m: Message) raises -> String:
    """The wire JSON re-serialized with Python's separators, for readable comparisons."""
    return py_json_dumps(parse_json(m.wire_json()), False, -1)


def _raises_with(body: String, prefix: String) raises:
    var got = String()
    try:
        _ = parse_chat_response(body)
    except e:
        got = String(e)
    assert_true(got.startswith(prefix), "expected '" + prefix + "', got '" + got + "'")


def test_wire_roles() raises:
    assert_equal(_wire(system_message("SYS")), '{"content": "SYS", "role": "system"}')
    assert_equal(_wire(user_message("hi é")), '{"content": "hi é", "role": "user"}')
    # Assistant with calls and empty content: content null, name, valid calls first.
    var a = Message(ROLE_ASSISTANT, "")
    a.name = "orchestrator"
    a.tool_calls.append(ToolCall("call_1", "ls", parse_json('{"path":"/é","n":[1,2.5]}')))
    a.invalid_tool_calls.append(InvalidToolCall("call_2", "grep", '{"pattern": ', "bad json"))
    a.tool_calls.append(ToolCall("call_3", "read_file", parse_json("{}")))
    var w = parse_json(a.wire_json())
    assert_equal(w.keys()[0], "content")
    assert_true(w.get("content").is_null())
    assert_equal(w.get("name").as_string(), "orchestrator")
    assert_equal(w.get("role").as_string(), "assistant")
    var calls = w.get("tool_calls")
    assert_equal(len(calls), 3)
    assert_equal(calls.at(0).keys()[0], "type")
    assert_equal(calls.at(0).keys()[1], "id")
    assert_equal(calls.at(0).get("id").as_string(), "call_1")
    # arguments re-serialized with ", " and ": ", é kept raw.
    assert_equal(calls.at(0).get("function").get("arguments").as_string(), '{"path": "/é", "n": [1, 2.5]}')
    assert_equal(calls.at(1).get("id").as_string(), "call_3")
    assert_equal(calls.at(1).get("function").get("arguments").as_string(), "{}")
    # The invalid call comes last with its raw text.
    assert_equal(calls.at(2).get("id").as_string(), "call_2")
    assert_equal(calls.at(2).get("function").get("arguments").as_string(), '{"pattern": ')
    # Assistant with text and no calls: string content, no tool_calls key.
    var b = Message(ROLE_ASSISTANT, "done")
    b.name = "a"
    assert_equal(_wire(b), '{"content": "done", "name": "a", "role": "assistant"}')
    var b2 = Message(ROLE_ASSISTANT, "")
    assert_equal(_wire(b2), '{"content": "", "role": "assistant"}')
    # Tool messages: string and text-part arrays; is_error and name are not sent.
    var t = tool_message("call_1", "ls", error_outcome("Error: x"))
    assert_true(t.is_error)
    assert_equal(t.name, "ls")
    assert_equal(_wire(t), '{"content": "Error: x", "role": "tool", "tool_call_id": "call_1"}')
    var parts: List[String] = ["one", "two"]
    var tp = tool_message("c9", "qumo_x", parts_outcome(parts^, False))
    assert_true(tp.has_parts)
    assert_equal(tp.text(), "onetwo")
    assert_equal(_wire(tp), '{"content": [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}], "role": "tool", "tool_call_id": "c9"}')
    assert_equal(tool_call_args_json(a.tool_calls[0]), '{"path": "/é", "n": [1, 2.5]}')
    assert_equal(a.role_name(), "assistant")
    assert_equal(t.role_name(), "tool")
    assert_equal(a.usage_total_tokens, -1)


def test_toolspec() raises:
    var s = ToolSpec("ls", "List files.", '{"type":"object","properties":{}}', TOOL_HARNESS, False, "")
    var w = parse_json(s.wire_json())
    assert_equal(w.get("type").as_string(), "function")
    assert_equal(w.get("function").keys()[0], "name")
    assert_equal(w.get("function").keys()[2], "parameters")
    assert_equal(w.get("function").get("parameters").get("type").as_string(), "object")
    assert_equal(text_outcome("x").joined_text(), "x")
    assert_false(text_outcome("x").is_error)
    var parts: List[String] = ["a", "b"]
    var po = parts_outcome(parts^, True)
    assert_equal(po.joined_text(), "a\nb")
    assert_true(po.is_error and po.is_parts)


def test_request_body() raises:
    var msgs = List[Message]()
    msgs.append(user_message("task"))
    var tools = List[ToolSpec]()
    tools.append(ToolSpec("ls", "d", '{"type":"object"}', TOOL_HARNESS, False, ""))
    var req = ChatRequest("m/x", "SYS", msgs.copy(), tools^, "orchestrator", "")
    var body = parse_json(req.body_json())
    var keys = body.keys()
    assert_equal(len(keys), 4)
    assert_equal(keys[0], "model")
    assert_equal(keys[1], "stream")
    assert_equal(keys[2], "messages")
    assert_equal(keys[3], "tools")
    assert_false(body.get("stream").as_bool())
    assert_equal(len(body.get("messages")), 2)
    assert_equal(body.get("messages").at(0).get("role").as_string(), "system")
    # Summary call: no system message, no tools key.
    var req2 = ChatRequest("m/x", "", msgs^, List[ToolSpec](), "a", "x/y")
    var body2 = parse_json(req2.body_json())
    assert_equal(len(body2.keys()), 3)
    assert_false(body2.has("tools"))
    assert_equal(len(body2.get("messages")), 1)
    assert_equal(body2.get("messages").at(0).get("role").as_string(), "user")


comptime FULL = """{"id":"gen-1","choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":"Let me look.","reasoning":"secret thoughts","reasoning_details":[{"type":"reasoning.text","text":"hm"}],"refusal":null,"tool_calls":[{"id":"toolu_01","type":"function","function":{"name":"ls","arguments":"{\\"path\\": \\"/\\"}"}},{"type":"function","function":{"name":"read_file","arguments":""}},{"id":"t3","type":"function","function":{"name":"grep","arguments":"{\\"pattern\\": "}},{"id":"t4","type":"function","function":{"name":"x","arguments":"[1]"}},{"id":"t5","type":"function","function":{"name":"edit_file","arguments":"{\\"code\\": \\"a\\nb\\"}"}}]}}],"usage":{"prompt_tokens":1234,"completion_tokens":56,"total_tokens":1290}}"""


def test_parse_response() raises:
    var r = parse_chat_response(FULL)
    assert_equal(r.finish_reason, "tool_calls")
    assert_equal(r.prompt_tokens, 1234)
    assert_equal(r.completion_tokens, 56)
    assert_equal(r.total_tokens, 1290)
    var m = r.message.copy()
    assert_equal(m.role, ROLE_ASSISTANT)
    assert_equal(m.content, "Let me look.")
    assert_equal(m.usage_total_tokens, 1290)
    # B4: reasoning is dropped from what we send back.
    assert_true(m.wire_json().find("reasoning") < 0)
    assert_true(m.wire_json().find("secret") < 0)
    assert_equal(len(m.tool_calls), 3)
    assert_equal(m.tool_calls[0].id, "toolu_01")
    assert_equal(m.tool_calls[0].args.get("path").as_string(), "/")
    assert_equal(m.tool_calls[1].id, "call_1")               # missing id: call_<index>
    assert_equal(m.tool_calls[1].name, "read_file")
    assert_equal(len(m.tool_calls[1].args), 0)               # "" gives {}
    assert_equal(m.tool_calls[2].args.get("code").as_string(), "a\nb")   # lenient
    assert_equal(len(m.invalid_tool_calls), 2)
    assert_equal(m.invalid_tool_calls[0].id, "t3")
    assert_equal(m.invalid_tool_calls[0].raw_args, '{"pattern": ')
    assert_true(m.invalid_tool_calls[0].error.startswith("QM-E900"))
    assert_equal(m.invalid_tool_calls[1].error, "arguments must be a JSON object")
    # null content, list content, no usage.
    var r2 = parse_chat_response('{"choices":[{"message":{"role":"assistant","content":null}}]}')
    assert_equal(r2.message.content, "")
    assert_equal(r2.total_tokens, -1)
    assert_equal(r2.prompt_tokens, 0)
    assert_equal(r2.finish_reason, "")
    var r3 = parse_chat_response('{"choices":[{"finish_reason":"stop","message":{"content":[{"type":"text","text":"a"},{"type":"image_url"},{"type":"text","text":"b"}]}}],"usage":{"prompt_tokens":3}}')
    assert_equal(r3.message.content, "ab")
    assert_equal(r3.total_tokens, -1)
    assert_equal(r3.prompt_tokens, 3)
    # B5 errors.
    _raises_with('{"error":{"message":"rate limited","code":429},"choices":null}', 'QA-E103: model returned an error: {"message": "rate limited", "code": 429}')
    _raises_with('{"id":"x","object":"chat.completion"}', "QA-E103: Response missing 'choices' key: dict_keys(['id', 'object'])")
    _raises_with('{"id":"x","choices":null}', "QA-E103: Received response with null value for 'choices'. This can happen when using OpenAI-compatible APIs (e.g., vLLM) that return a response in an unexpected format. Full response keys: ['id', 'choices']")
    _raises_with('{"choices":[]}', "QA-E103: response has no choices")
    _raises_with('not json', "QA-E103: response is not valid JSON")
    # A falsy error is ignored.
    var r4 = parse_chat_response('{"error":null,"choices":[{"message":{"content":"ok"}}]}')
    assert_equal(r4.message.content, "ok")


def test_overflow() raises:
    var markers: List[String] = [
        "context_length_exceeded", "ContextWindowExceededError", "maximum context length",
        "exceeds the context window", "exceeds the available context size", "context window exceeded",
        "context limit exceeded", "input tokens exceed the configured limit", "prompt is too long",
    ]
    for m in markers:
        for status in [400, 413, 422]:
            assert_true(is_context_overflow(status, '{"error": {"message": "This ' + m + '!"}}'), m)
        assert_false(is_context_overflow(401, '{"error": "' + m + '"}') and m != "exceeds the context window")
    assert_true(is_context_overflow(503, "Input EXCEEDS THE CONTEXT WINDOW of this model"))
    assert_false(is_context_overflow(503, "prompt is too long"))
    assert_false(is_context_overflow(400, "bad request"))
    assert_false(is_context_overflow(200, "exceeds the context window"))


def test_errors() raises:
    var e = qa_error(E_MODEL_HTTP, "model request failed: HTTP 429 Too Many Requests: slow down", "")
    assert_equal(String(e), "QA-E101: model request failed: HTTP 429 Too Many Requests: slow down")
    assert_equal(error_code(e), "QA-E101")
    assert_equal(error_http_status(e), 429)
    var o = qa_error(E_CONTEXT_OVERFLOW, "context overflow: HTTP 413: too big", "")
    assert_equal(error_http_status(o), 413)
    var h = qa_error(E_CONFIG, "unknown agent 'x'", "choose one of a, b")
    assert_equal(String(h), "QA-E140: unknown agent 'x'. Hint: choose one of a, b")
    assert_equal(error_code(h), "QA-E140")
    assert_equal(error_http_status(h), -1)
    assert_equal(error_text(h), String(h))
    assert_equal(error_code(Error("QM-E900: other")), "")
    assert_equal(error_code(Error("QA-E1x0: no")), "")
    assert_equal(error_code(Error("QA-E")), "")
    assert_equal(error_http_status(Error("QA-E102: Model input still does not fit after recovery")), -1)
    assert_equal(error_http_status(Error("QA-E101: model request failed: HTTP x")), -1)


def main() raises:
    test_wire_roles()
    test_toolspec()
    test_request_body()
    test_parse_response()
    test_overflow()
    test_errors()
    print("test_core_messages OK")
