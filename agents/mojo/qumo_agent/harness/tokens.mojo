"""Portions ported from deepagents 0.7.32 and langchain 1.4.8 / langchain-core 1.7.4
(Copyright (c) LangChain, Inc., MIT License; see agents/mojo/LICENSES/langchain-MIT.txt).

langchain-core `count_tokens_approximately` (CU:2257-2431) with chars_per_token 4.0,
extra_tokens_per_message 3.0 and count_name True, as SummarizationMiddleware calls it
(DESIGN.md 2.7.6). Every length is in code points.
"""

from std.math import ceil

from qumo.ai.json import JsonValue, parse_json

from qumo_agent.core.messages import Message, ToolCall, ROLE_ASSISTANT, ROLE_TOOL, system_message
from qumo_agent.core.toolspec import ToolSpec
from qumo_agent.util.jsonx import json_without
from qumo_agent.util.pyfmt import cp_len, py_str_repr, py_repr_json, py_json_dumps, py_json_dumps_string

comptime _CHARS_PER_TOKEN = 4.0
comptime _EXTRA_PER_MESSAGE = 3.0


def tool_calls_repr(calls: List[ToolCall]) -> String:
    """repr(AIMessage.tool_calls): `[{'name': 'ls', 'args': {...}, 'id': 'call_1', 'type': 'tool_call'}]`."""
    var out = String("[")
    for i in range(len(calls)):
        if i > 0:
            out += ", "
        out += "{'name': " + py_str_repr(calls[i].name) + ", 'args': " + py_repr_json(calls[i].args)
        out += ", 'id': " + py_str_repr(calls[i].id) + ", 'type': 'tool_call'}"
    out += "]"
    return out^


def _role_len(m: Message) -> Int:
    return cp_len(m.role_name())


def message_chars(m: Message) -> Int:
    """Characters counted for one message (content, tool calls, tool_call_id, role, name)."""
    var n = 0
    if m.has_parts:
        for p in m.parts:
            n += cp_len(p)
    else:
        n += cp_len(m.content)
        if m.role == ROLE_ASSISTANT and len(m.tool_calls) > 0:
            n += cp_len(tool_calls_repr(m.tool_calls))
    if m.role == ROLE_TOOL:
        n += cp_len(m.tool_call_id)
    n += _role_len(m)
    n += cp_len(m.name)
    return n


def _count_dict_json(t: ToolSpec) -> String:
    """`json.dumps(tool_dict)` (ensure_ascii, default separators) as Python counts the tool."""
    try:
        if t.count_json.byte_length() > 0:
            return py_json_dumps(parse_json(t.count_json), True, -1)
        var params = parse_json(t.parameters_json if t.parameters_json.byte_length() > 0 else "{}")
        if params.is_object():
            params = json_without(json_without(params, "title"), "description")
        var out = String("{\"name\": ") + py_json_dumps_string_ascii(t.name)
        out += ", \"description\": " + py_json_dumps_string_ascii(t.description)
        out += ", \"parameters\": " + py_json_dumps(params, True, -1) + "}"
        return out^
    except:
        return t.wire_json()


def py_json_dumps_string_ascii(s: String) -> String:
    return py_json_dumps_string(s, True)


def tools_chars(tools: List[ToolSpec]) -> Int:
    var n = 0
    for t in tools:
        n += cp_len(_count_dict_json(t))
    return n


def _ceil_div4(chars: Int) -> Float64:
    return Float64(ceil(Float64(chars) / _CHARS_PER_TOKEN))


def count_tokens_list(messages: List[Message], tools: List[ToolSpec], use_scaling: Bool) -> Int:
    """count_tokens_approximately(messages, tools=tools) over exactly `messages`."""
    var token_count = 0.0
    if len(tools) > 0:
        token_count += _ceil_div4(tools_chars(tools))
    var saw_ai = False
    var last_total = -1
    var approx_at_last = 0.0
    for m in messages:
        token_count += _ceil_div4(message_chars(m)) + _EXTRA_PER_MESSAGE
        if m.role == ROLE_ASSISTANT:
            saw_ai = True
            if m.usage_total_tokens >= 0:
                last_total = m.usage_total_tokens
                approx_at_last = token_count
    if use_scaling and len(messages) > 1 and saw_ai and last_total >= 0 and approx_at_last > 0.0:
        var scale = Float64(last_total) / approx_at_last
        if scale < 1.0:
            scale = 1.0
        if scale > 1.25:
            scale = 1.25
        token_count *= scale
    return Int(ceil(token_count))


def count_tokens_approx(system: String, messages: List[Message], tools: List[ToolSpec], use_scaling: Bool) -> Int:
    """count_tokens_approximately([system, *messages], tools=tools); the system prompt counts
    as the first message (role "system"). With `use_scaling`, the last assistant message
    reporting `usage_total_tokens` scales the count by min(1.25, max(1.0, total / approx))."""
    var all = List[Message]()
    all.append(system_message(system))
    for m in messages:
        all.append(m.copy())
    return count_tokens_list(all, tools, use_scaling)
