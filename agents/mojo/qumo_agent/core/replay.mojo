"""Offline model: scripted chat-completions responses read from a JSONL file.

A line is a bare chat-completions response (for the root agent, `at = ""`), or a
wrapper `{"at": "<path>", "response": {...}}` or `{"at": "<path>", "error":
"QA-E102: ..."}` (the error is raised). Blank lines and lines starting with `//`
are skipped. `at` is the delegation path: the root is ""; the child started by the
`task` call with id X from path P is X when P is "", else P + "/" + X. Each request
consumes the next unused line whose `at` equals `req.replay_at` (one cursor per
path), so a child process given the same file reads only its own lines.

When `$QUMO_AGENT_REPLAY_LOG` names a directory, every request body is appended to
`<dir>/<at with "/" replaced by "__", or "root">.jsonl`.
"""

from std.os import getenv
from std.os.path import isdir

from qumo.ai.json import JsonValue, parse_json

from qumo_agent.core.messages import ChatRequest, ChatResponse, parse_chat_response


@fieldwise_init
struct _ReplayLine(Copyable):
    var at: String
    var response: String     # response JSON text ("" for an error line)
    var error: String        # error text to raise ("" for a response line)
    var used: Bool


def _read_text(path: String) raises -> String:
    try:
        with open(path, "r") as f:
            return f.read()
    except e:
        raise Error("QA-E170: cannot read replay file '" + path + "': " + String(e))


def _log_name(at: String) -> String:
    if at.byte_length() == 0:
        return "root"
    return at.replace("/", "__")


struct ReplayModel(Movable):
    var requests: List[String]        # request bodies in call order, for assertions
    var request_paths: List[String]   # the replay path of each request
    var _lines: List[_ReplayLine]
    var _consumed_paths: List[String]
    var _consumed_counts: List[Int]

    def __init__(out self, path: String) raises:
        self.requests = List[String]()
        self.request_paths = List[String]()
        self._lines = List[_ReplayLine]()
        self._consumed_paths = List[String]()
        self._consumed_counts = List[Int]()
        var n = 0
        for raw in _read_text(path).split("\n"):
            n += 1
            var line = String(raw.strip())
            if line.byte_length() == 0 or line.startswith("//"):
                continue
            var doc: JsonValue
            try:
                doc = parse_json(line)
            except e:
                raise Error("QA-E170: replay file '" + path + "' line " + String(n) + " is not JSON: " + String(e))
            if doc.is_object() and doc.has("at") and (doc.has("response") or doc.has("error")):
                var at_v = doc.get("at")
                if not at_v.is_string():
                    raise Error("QA-E170: replay file '" + path + "' line " + String(n) + ": \"at\" must be a string")
                var at = at_v.as_string()
                if doc.has("error"):
                    var err = doc.get("error")
                    var text = err.as_string() if err.is_string() else err.to_json()
                    self._lines.append(_ReplayLine(at, String(), text, False))
                else:
                    self._lines.append(_ReplayLine(at, doc.get("response").to_json(), String(), False))
            else:
                self._lines.append(_ReplayLine(String(), line, String(), False))

    def _count_consumed(mut self, at: String):
        for i in range(len(self._consumed_paths)):
            if self._consumed_paths[i] == at:
                self._consumed_counts[i] += 1
                return
        self._consumed_paths.append(at)
        self._consumed_counts.append(1)

    def _consumed(self, at: String) -> Int:
        for i in range(len(self._consumed_paths)):
            if self._consumed_paths[i] == at:
                return self._consumed_counts[i]
        return 0

    def complete(mut self, req: ChatRequest) raises -> ChatResponse:
        """The next scripted response for `req.replay_at` (an error line raises)."""
        var body = req.body_json()
        self.requests.append(body)
        self.request_paths.append(req.replay_at)
        var log_dir = getenv("QUMO_AGENT_REPLAY_LOG", "")
        if log_dir.byte_length() > 0 and isdir(log_dir):
            var log_path = log_dir + "/" + _log_name(req.replay_at) + ".jsonl"
            with open(log_path, "a") as f:
                f.write(body, "\n")
        for i in range(len(self._lines)):
            if self._lines[i].used or self._lines[i].at != req.replay_at:
                continue
            self._lines[i].used = True
            self._count_consumed(req.replay_at)
            if self._lines[i].error.byte_length() > 0:
                raise Error(self._lines[i].error)
            return parse_chat_response(self._lines[i].response)
        raise Error("QA-E170: replay file exhausted at '" + req.replay_at + "' after "
                    + String(self._consumed(req.replay_at)) + " responses")

    def remaining(self, at: String) -> Int:
        """Unused lines for the path `at`."""
        var n = 0
        for i in range(len(self._lines)):
            if not self._lines[i].used and self._lines[i].at == at:
                n += 1
        return n
