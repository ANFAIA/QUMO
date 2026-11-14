"""Portions ported from deepagents 0.7.32 and langchain 1.4.8 / langchain-core 1.7.4
(Copyright (c) LangChain, Inc., MIT License; see agents/mojo/LICENSES/langchain-MIT.txt).
The response checks and their texts follow langchain-openai 1.7.3 (same copyright
and license).

Chat messages, the chat-completions request body and response parsing.

The wire format is DESIGN.md section 1.5: the request body has the keys `model`,
`stream`, `messages`, `tools` (in that order, `tools` omitted when empty); message
keys follow the order of the dicts Python sends. Response parsing follows
langchain-openai (critic B2, B4, B5): reasoning fields are dropped, list content is
joined, an `error` body, missing, null or empty `choices` raise QA-E103.
"""

from qumo.ai.json import JsonValue, JsonKind, JsonWriter, parse_json

from qumo_agent.core.toolspec import ToolSpec, ToolOutcome
from qumo_agent.util.jsonx import parse_tool_arguments
from qumo_agent.util.pyfmt import py_json_dumps, py_str_repr

comptime ROLE_SYSTEM = 0
comptime ROLE_USER = 1
comptime ROLE_ASSISTANT = 2
comptime ROLE_TOOL = 3

# Appendix A.12: markers of a context-window overflow in an error body (lowercased).
comptime OVERFLOW_MARKERS: List[String] = [
    "context_length_exceeded",
    "contextwindowexceedederror",
    "maximum context length",
    "exceeds the context window",
    "exceeds the available context size",
    "context window exceeded",
    "context limit exceeded",
    "input tokens exceed the configured limit",
    "prompt is too long",
]


@fieldwise_init
struct ToolCall(Copyable):
    var id: String
    var name: String
    var args: JsonValue              # always an object


@fieldwise_init
struct InvalidToolCall(Copyable):
    var id: String
    var name: String
    var raw_args: String
    var error: String


struct Message(Copyable):
    var role: Int
    var content: String
    var parts: List[String]
    var has_parts: Bool
    var name: String
    var tool_calls: List[ToolCall]
    var invalid_tool_calls: List[InvalidToolCall]
    var tool_call_id: String
    var is_error: Bool
    var lc_source: String
    var evicted_to: String
    var usage_total_tokens: Int

    def __init__(out self, role: Int, content: String):
        self.role = role
        self.content = content
        self.parts = List[String]()
        self.has_parts = False
        self.name = String()
        self.tool_calls = List[ToolCall]()
        self.invalid_tool_calls = List[InvalidToolCall]()
        self.tool_call_id = String()
        self.is_error = False
        self.lc_source = String()
        self.evicted_to = String()
        self.usage_total_tokens = -1

    def text(self) -> String:
        """AIMessage.text: the string content, or the text parts joined with ""."""
        if self.has_parts:
            return String("").join(self.parts)
        return self.content

    def role_name(self) -> String:
        if self.role == ROLE_SYSTEM:
            return "system"
        if self.role == ROLE_USER:
            return "user"
        if self.role == ROLE_ASSISTANT:
            return "assistant"
        return "tool"

    def _write_content(self, mut w: JsonWriter):
        w.key("content")
        if self.has_parts:
            w.begin_array()
            for p in self.parts:
                w.begin_object()
                w.kv_string("type", "text")
                w.kv_string("text", p)
                w.end_object()
            w.end_array()
        elif (self.role == ROLE_ASSISTANT and self.content.byte_length() == 0
              and (len(self.tool_calls) > 0 or len(self.invalid_tool_calls) > 0)):
            w.null()
        else:
            w.string(self.content)

    def wire_json(self) -> String:
        """The message as sent to the model (DESIGN.md 1.5)."""
        var w = JsonWriter()
        w.begin_object()
        self._write_content(w)
        if self.role == ROLE_ASSISTANT and self.name.byte_length() > 0:
            w.kv_string("name", self.name)
        w.kv_string("role", self.role_name())
        if self.role == ROLE_ASSISTANT and (len(self.tool_calls) > 0 or len(self.invalid_tool_calls) > 0):
            w.key("tool_calls")
            w.begin_array()
            for c in self.tool_calls:
                w.begin_object()
                w.kv_string("type", "function")
                w.kv_string("id", c.id)
                w.key("function")
                w.begin_object()
                w.kv_string("name", c.name)
                w.kv_string("arguments", tool_call_args_json(c))
                w.end_object()
                w.end_object()
            for c in self.invalid_tool_calls:
                w.begin_object()
                w.kv_string("type", "function")
                w.kv_string("id", c.id)
                w.key("function")
                w.begin_object()
                w.kv_string("name", c.name)
                w.kv_string("arguments", c.raw_args)
                w.end_object()
                w.end_object()
            w.end_array()
        if self.role == ROLE_TOOL:
            w.kv_string("tool_call_id", self.tool_call_id)
        w.end_object()
        return w.finish()


def system_message(text: String) -> Message:
    return Message(ROLE_SYSTEM, text)


def user_message(text: String) -> Message:
    return Message(ROLE_USER, text)


def tool_message(call_id: String, tool_name: String, outcome: ToolOutcome) -> Message:
    """A tool result; a parts outcome keeps its parts (content stays "")."""
    var m = Message(ROLE_TOOL, String() if outcome.is_parts else outcome.text)
    if outcome.is_parts:
        m.parts = outcome.parts.copy()
        m.has_parts = True
    m.name = tool_name
    m.tool_call_id = call_id
    m.is_error = outcome.is_error
    return m^


def tool_call_args_json(call: ToolCall) -> String:
    """json.dumps(args, ensure_ascii=False) with ", " and ": " separators."""
    return py_json_dumps(call.args, False, -1)


struct ChatRequest(Copyable):
    var model: String
    var system: String
    var messages: List[Message]       # without the system message
    var tools: List[ToolSpec]         # empty for the summary call
    var agent: String                 # logging and replay routing
    var replay_at: String

    def __init__(out self, model: String, system: String, var messages: List[Message], var tools: List[ToolSpec], agent: String, replay_at: String):
        self.model = model
        self.system = system
        self.messages = messages^
        self.tools = tools^
        self.agent = agent
        self.replay_at = replay_at

    def body_json(self) -> String:
        """`{"model", "stream": false, "messages", "tools"}`; the system message is
        omitted when `system` is empty and `tools` when there are none."""
        var w = JsonWriter()
        w.begin_object()
        w.kv_string("model", self.model)
        w.kv_bool("stream", False)
        w.key("messages")
        w.begin_array()
        if self.system.byte_length() > 0:
            w.raw(system_message(self.system).wire_json())
        for m in self.messages:
            w.raw(m.wire_json())
        w.end_array()
        if len(self.tools) > 0:
            w.key("tools")
            w.begin_array()
            for t in self.tools:
                w.raw(t.wire_json())
            w.end_array()
        w.end_object()
        return w.finish()


@fieldwise_init
struct ChatResponse(Copyable):
    var message: Message              # ROLE_ASSISTANT; the loop sets `name`
    var finish_reason: String
    var prompt_tokens: Int
    var completion_tokens: Int
    var total_tokens: Int             # -1 when absent


def _truthy(v: JsonValue) -> Bool:
    var k = v.kind()
    if k == JsonKind.UNDEFINED or k == JsonKind.NULL:
        return False
    try:
        if k == JsonKind.BOOL:
            return v.as_bool()
        if k == JsonKind.NUMBER:
            return v.as_float() != 0.0
        if k == JsonKind.STRING:
            return v.as_string().byte_length() > 0
    except:
        return False
    return len(v) > 0


def _keys_repr(doc: JsonValue) -> String:
    """The keys of an object as Python lists them: 'a', 'b'."""
    var out = String()
    var keys = doc.keys()
    var seen = List[String]()
    for k in keys:
        var dup = False
        for s in seen:
            if s == k:
                dup = True
                break
        if dup:
            continue
        if len(seen) > 0:
            out += ", "
        out += py_str_repr(k)
        seen.append(k)
    return out^


def _str_or(v: JsonValue, default: String) -> String:
    if v.is_string():
        try:
            return v.as_string()
        except:
            pass
    return default


def _content_text(c: JsonValue) -> String:
    """String content as is; null or absent gives ""; a list gives its text parts joined."""
    if c.is_string():
        return _str_or(c, "")
    if c.is_array():
        var out = String()
        for i in range(len(c)):
            var p = c.at(i)
            if p.is_string():
                out += _str_or(p, "")
            elif p.is_object() and _str_or(p.get("type"), "") == "text" and p.get("text").is_string():
                out += _str_or(p.get("text"), "")
        return out^
    return String()


def _int_or(v: JsonValue, default: Int) -> Int:
    if not v.is_number():
        return default
    try:
        return v.as_int()
    except:
        return default


def parse_chat_response(body: String) raises -> ChatResponse:
    """A chat-completions response body as an assistant message (critic B2, B4, B5)."""
    var doc: JsonValue
    try:
        doc = parse_json(body)
    except e:
        raise Error("QA-E103: response is not valid JSON: " + String(e))
    if not doc.is_object():
        raise Error("QA-E103: response is not a JSON object, got " + doc.kind_name())
    var err = doc.get("error")
    if _truthy(err):
        raise Error("QA-E103: model returned an error: " + py_json_dumps(err, False, -1))
    var choices = doc.get("choices")
    if not choices.is_defined():
        raise Error("QA-E103: Response missing 'choices' key: dict_keys([" + _keys_repr(doc) + "])")
    if choices.is_null():
        raise Error(
            "QA-E103: Received response with null value for 'choices'. "
            + "This can happen when using OpenAI-compatible APIs (e.g., vLLM) "
            + "that return a response in an unexpected format. "
            + "Full response keys: [" + _keys_repr(doc) + "]"
        )
    if not choices.is_array() or len(choices) == 0:
        raise Error("QA-E103: response has no choices")
    var ch = choices.at(0)
    var m = ch.get("message")
    var msg = Message(ROLE_ASSISTANT, _content_text(m.get("content")))
    var tcs = m.get("tool_calls")
    if tcs.is_array():
        for i in range(len(tcs)):
            var tc = tcs.at(i)
            var id = _str_or(tc.get("id"), "")
            if id.byte_length() == 0:
                id = "call_" + String(i)
            var f = tc.get("function")
            var name = _str_or(f.get("name"), "")
            var raw = f.get("arguments")
            var parsed = parse_tool_arguments(raw)
            if parsed.ok:
                msg.tool_calls.append(ToolCall(id, name, parsed.args.copy()))
            else:
                var raw_text = String()
                if raw.is_string():
                    raw_text = _str_or(raw, "")
                elif raw.is_defined():
                    raw_text = raw.to_json()
                msg.invalid_tool_calls.append(InvalidToolCall(id, name, raw_text, parsed.error))
    var usage = doc.get("usage")
    var prompt = 0
    var completion = 0
    var total = -1
    if usage.is_object():
        prompt = _int_or(usage.get("prompt_tokens"), 0)
        completion = _int_or(usage.get("completion_tokens"), 0)
        total = _int_or(usage.get("total_tokens"), -1)
    msg.usage_total_tokens = total
    return ChatResponse(msg^, _str_or(ch.get("finish_reason"), ""), prompt, completion, total)


def is_context_overflow(status: Int, body: String) -> Bool:
    """True for 400/413/422 with an A.12 marker in the body, or any other non-2xx
    status whose body says "exceeds the context window" (langchain-openai)."""
    if status >= 200 and status < 300:
        return False
    var lower = body.lower()
    if status == 400 or status == 413 or status == 422:
        for marker in materialize[OVERFLOW_MARKERS]():
            if marker in lower:
                return True
        return False
    return "exceeds the context window" in lower
