"""OpenAI-compatible chat completions: OpenRouter through curl, and an offline replay.

Mojo's standard library has no TLS, so HTTPS goes through /usr/bin/curl, spawned
without a shell. The API key never reaches argv, logs or files: curl reads it from
the environment (`--variable %NAME`) and expands it into the header itself. The
request body and the response travel through files in a private temporary directory
(mktemp -d, mode 0700) that is removed after every call. Error text is redacted
before it is shown.

`ReplayProvider` returns canned chat-completions responses, one JSON object per line
of a .jsonl file, so the agent loop runs offline and deterministically.
"""

from std.os import getenv, remove, rmdir
from std.os.path import exists
from std.subprocess import run

from qumo.ai.json import JsonWriter, JsonValue, parse_json, json_quote

from qumo_agent.config import read_text
from qumo_agent.proc import spawn


@fieldwise_init
struct ToolCall(Copyable):
    var id: String
    var name: String
    var arguments: String  # JSON text, as the model wrote it


struct ChatResponse(Copyable):
    var content: String
    var has_content: Bool
    var tool_calls: List[ToolCall]
    var finish_reason: String
    var prompt_tokens: Int
    var completion_tokens: Int
    var reasoning_details: String  # JSON text ("" when absent); OpenRouter wants it sent back

    def __init__(out self):
        self.content = String()
        self.has_content = False
        self.tool_calls = List[ToolCall]()
        self.finish_reason = String()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.reasoning_details = String()

    def assistant_message_json(self) -> String:
        """The assistant turn to append to the conversation."""
        var w = JsonWriter()
        w.begin_object()
        w.kv_string("role", "assistant")
        w.kv_string("content", self.content)
        if len(self.tool_calls) > 0:
            w.key("tool_calls")
            w.begin_array()
            for tc in self.tool_calls:
                w.begin_object()
                w.kv_string("id", tc.id)
                w.kv_string("type", "function")
                w.key("function")
                w.begin_object()
                w.kv_string("name", tc.name)
                w.kv_string("arguments", tc.arguments)
                w.end_object()
                w.end_object()
            w.end_array()
        if self.reasoning_details.byte_length() > 0:
            w.kv_raw("reasoning_details", self.reasoning_details)
        w.end_object()
        return w.finish()


def message_json(role: String, content: String) -> String:
    var w = JsonWriter()
    w.begin_object()
    w.kv_string("role", role)
    w.kv_string("content", content)
    w.end_object()
    return w.finish()


def tool_message_json(tool_call_id: String, content: String) -> String:
    var w = JsonWriter()
    w.begin_object()
    w.kv_string("role", "tool")
    w.kv_string("tool_call_id", tool_call_id)
    w.kv_string("content", content)
    w.end_object()
    return w.finish()


def function_tool_json(name: String, description: String, parameters_json: String) -> String:
    """A tool in OpenAI function format."""
    var w = JsonWriter()
    w.begin_object()
    w.kv_string("type", "function")
    w.key("function")
    w.begin_object()
    w.kv_string("name", name)
    w.kv_string("description", description)
    w.kv_raw("parameters", parameters_json)
    w.end_object()
    w.end_object()
    return w.finish()


def build_request(model: String, messages: List[String], tools: List[String], max_tokens: Int) -> String:
    """Chat-completions request body; `messages` and `tools` are JSON texts."""
    var w = JsonWriter()
    w.begin_object()
    w.kv_string("model", model)
    w.key("messages")
    w.begin_array()
    for m in messages:
        w.raw(m)
    w.end_array()
    if len(tools) > 0:
        w.key("tools")
        w.begin_array()
        for t in tools:
            w.raw(t)
        w.end_array()
    if max_tokens > 0:
        w.kv_int("max_tokens", max_tokens)
    w.end_object()
    return w.finish()


def _str_or_empty(v: JsonValue) raises -> String:
    if v.is_string():
        return v.as_string()
    if not v.is_defined() or v.is_null():
        return String()
    if v.is_array():
        # content as a list of parts: keep the text parts
        var out = String()
        for i in range(len(v)):
            var p = v.at(i)
            if p.get_string("type", "") == "text":
                out += p.get_string("text", "")
        return out^
    return v.to_json()


def parse_response(text: String) raises -> ChatResponse:
    """Parse a chat-completions response; an `error` object raises."""
    var doc = parse_json(text)
    if not doc.is_object():
        raise Error("LLM response is not a JSON object")
    var err = doc.get("error")
    if err.is_defined() and not err.is_null():
        var msg = err.get_string("message", "") if err.is_object() else err.to_json()
        raise Error("LLM error: " + redact(msg))
    var choices = doc.get("choices")
    if len(choices) == 0:
        raise Error("LLM response has no choices")
    var ch = choices.at(0)
    var r = ChatResponse()
    r.finish_reason = _str_or_empty(ch.get("finish_reason"))
    var m = ch.get("message")
    var c = m.get("content")
    r.has_content = c.is_defined() and not c.is_null()
    r.content = _str_or_empty(c)
    var tcs = m.get("tool_calls")
    for i in range(len(tcs)):
        var tc = tcs.at(i)
        var f = tc.get("function")
        var args = f.get("arguments")
        var args_text = String("{}")
        if args.is_string():
            args_text = args.as_string()
        elif args.is_object():
            args_text = args.to_json()
        var id = tc.get_string("id", "")
        if id.byte_length() == 0:
            id = "call_" + String(i)
        r.tool_calls.append(ToolCall(id, f.get_string("name", ""), args_text))
    var rd = m.get("reasoning_details")
    if rd.is_array() and len(rd) > 0:
        r.reasoning_details = rd.to_json()
    var u = doc.get("usage")
    if u.is_object():
        r.prompt_tokens = u.get_int("prompt_tokens", 0)
        r.completion_tokens = u.get_int("completion_tokens", 0)
    return r^


def truncate_utf8(s: String, max_bytes: Int) -> String:
    """`s` cut to at most `max_bytes` bytes on a character boundary, with a marker."""
    if s.byte_length() <= max_bytes:
        return s
    var bs = s.as_bytes()
    var n = max_bytes
    while n > 0 and (Int(bs[n]) & 0xC0) == 0x80:
        n -= 1
    return String(unsafe_from_utf8=bs[0:n]) + "\n... [truncated: " + String(s.byte_length()) + " bytes in total]"


def _is_key_char(c: Int) -> Bool:
    return (c >= 0x30 and c <= 0x39) or (c >= 0x41 and c <= 0x5A) or (c >= 0x61 and c <= 0x7A) or c == 0x2D or c == 0x5F


def redact(s: String, secret: String = "") -> String:
    """Hide the secret itself and anything shaped like an API key (sk-..., Bearer ...)."""
    var t = s
    if secret.byte_length() >= 8:
        t = t.replace(secret, "[REDACTED]")
    var bs = t.as_bytes()
    var n = len(bs)
    var out = String()
    var i = 0
    while i < n:
        var hit = 0
        if i + 3 <= n and Int(bs[i]) == ord("s") and Int(bs[i + 1]) == ord("k") and Int(bs[i + 2]) == ord("-"):
            hit = 3
        elif i + 7 <= n and String(unsafe_from_utf8=bs[i:i + 7]) == "Bearer ":
            hit = 7
        if hit > 0 and (i == 0 or not _is_key_char(Int(bs[i - 1]))):
            var j = i + hit
            while j < n and _is_key_char(Int(bs[j])):
                j += 1
            if j - i - hit >= 8:
                out += String(unsafe_from_utf8=bs[i:i + hit]) + "[REDACTED]"
                i = j
                continue
        var k = i + 1
        while k < n and (Int(bs[k]) & 0xC0) == 0x80:
            k += 1
        out += String(unsafe_from_utf8=bs[i:k])
        i = k
    return out^


def _valid_env_name(s: String) -> Bool:
    var bs = s.as_bytes()
    if len(bs) == 0:
        return False
    for i in range(len(bs)):
        var c = Int(bs[i])
        if not ((c >= 0x41 and c <= 0x5A) or (c >= 0x30 and c <= 0x39) or c == 0x5F):
            return False
    return True


struct LlmClient(Movable):
    """`complete(request_json)` against OpenRouter, or against a replay file."""

    var replay: Bool
    var replay_lines: List[String]
    var replay_pos: Int
    var base_url: String
    var api_key_env: String
    var header_names: List[String]
    var header_values: List[String]
    var timeout_s: Int
    var max_retries: Int
    var curl: String
    var requests: List[String]  # replay mode keeps every request for tests

    def __init__(out self, base_url: String, api_key_env: String, header_names: List[String],
                 header_values: List[String], timeout_s: Int, max_retries: Int) raises:
        if not _valid_env_name(api_key_env):
            raise Error("api_key_env must be an environment variable name")
        if getenv(api_key_env, "").byte_length() == 0:
            raise Error("set " + api_key_env + " to use the openrouter back end, or pass --replay FILE")
        self.replay = False
        self.replay_lines = List[String]()
        self.replay_pos = 0
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.header_names = header_names.copy()
        self.header_values = header_values.copy()
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.curl = getenv("QUMO_CURL", "/usr/bin/curl")
        self.requests = List[String]()

    def __init__(out self, *, replay_file: String) raises:
        self.replay = True
        self.replay_lines = List[String]()
        for line in read_text(replay_file).split("\n"):
            var s = String(line.strip())
            if s.byte_length() > 0 and not s.startswith("//"):
                self.replay_lines.append(s)
        self.replay_pos = 0
        self.base_url = String()
        self.api_key_env = String()
        self.header_names = List[String]()
        self.header_values = List[String]()
        self.timeout_s = 0
        self.max_retries = 0
        self.curl = String()
        self.requests = List[String]()

    def complete(mut self, request_json: String) raises -> ChatResponse:
        if self.replay:
            self.requests.append(request_json)
            if self.replay_pos >= len(self.replay_lines):
                raise Error("replay file exhausted after " + String(self.replay_pos) + " responses")
            var line = self.replay_lines[self.replay_pos]
            self.replay_pos += 1
            return parse_response(line)
        return parse_response(self._post(request_json))

    def _post(self, body: String) raises -> String:
        var dir = String(run("mktemp -d -t qumo-agent").strip())
        if dir.byte_length() == 0 or not exists(dir):
            raise Error("mktemp -d failed")
        var req = dir + "/request.json"
        var resp = dir + "/response.json"
        var result = String()
        var code: Int
        try:
            with open(req, "w") as f:
                f.write(body)
            var args: List[String] = [
                "-sS", "--fail-with-body", "--retry", String(self.max_retries), "--max-time", String(self.timeout_s),
                "--variable", "%" + self.api_key_env,
                "--expand-header", "Authorization: Bearer {{" + self.api_key_env + "}}",
                "-H", "Content-Type: application/json",
            ]
            for i in range(len(self.header_names)):
                args.append("-H")
                args.append(self.header_names[i] + ": " + self.header_values[i])
            args.append("--data-binary")
            args.append("@" + req)
            args.append("-o")
            args.append(resp)
            args.append(self.base_url + "/chat/completions")
            var p = spawn(self.curl, args)
            var st = p.wait()
            code = st.exit_code.value() if st.exit_code else -1
            if exists(resp):
                result = read_text(resp)
        finally:
            if exists(req):
                remove(req)
            if exists(resp):
                remove(resp)
            rmdir(dir)
        if code != 0:
            var detail = result
            try:
                var doc = parse_json(result)
                var e = doc.get("error")
                if e.is_object():
                    detail = e.get_string("message", result)
            except:
                pass
            detail = truncate_utf8(detail, 600)
            raise Error("OpenRouter request failed (curl exit " + String(code) + "): " + redact(detail, getenv(self.api_key_env, "")))
        return result^
