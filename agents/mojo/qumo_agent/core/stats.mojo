"""Run statistics, summed over subagents (inline and child processes)."""

from qumo.ai.json import JsonValue, JsonWriter


@fieldwise_init
struct RunStats(Copyable, Writable):
    var model_calls: Int
    var tool_calls: Int
    var tool_errors: Int
    var mcp_calls: Int
    var http_requests: Int
    var http_retries: Int
    var prompt_tokens: Int
    var completion_tokens: Int
    var evictions: Int
    var summarizations: Int
    var subagent_runs: Int
    var wall_ms: Int

    def add(mut self, other: RunStats):
        """Adds every counter of `other`. `wall_ms` keeps the larger value: a subagent's
        wall time is already inside its parent's (and parallel children overlap)."""
        self.model_calls += other.model_calls
        self.tool_calls += other.tool_calls
        self.tool_errors += other.tool_errors
        self.mcp_calls += other.mcp_calls
        self.http_requests += other.http_requests
        self.http_retries += other.http_retries
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.evictions += other.evictions
        self.summarizations += other.summarizations
        self.subagent_runs += other.subagent_runs
        if other.wall_ms > self.wall_ms:
            self.wall_ms = other.wall_ms

    def to_json(self) -> String:
        """One JSON object, keys in field order."""
        var w = JsonWriter()
        w.begin_object()
        w.kv_int("model_calls", self.model_calls)
        w.kv_int("tool_calls", self.tool_calls)
        w.kv_int("tool_errors", self.tool_errors)
        w.kv_int("mcp_calls", self.mcp_calls)
        w.kv_int("http_requests", self.http_requests)
        w.kv_int("http_retries", self.http_retries)
        w.kv_int("prompt_tokens", self.prompt_tokens)
        w.kv_int("completion_tokens", self.completion_tokens)
        w.kv_int("evictions", self.evictions)
        w.kv_int("summarizations", self.summarizations)
        w.kv_int("subagent_runs", self.subagent_runs)
        w.kv_int("wall_ms", self.wall_ms)
        w.end_object()
        return w.finish()

    def from_json(mut self, v: JsonValue) raises:
        """Sets every field present in `v` (an object as written by `to_json`);
        missing fields keep their value. A non-integer field raises."""
        if not v.is_object():
            raise Error("QA-E120: statistics must be a JSON object, got " + v.kind_name())
        self.model_calls = v.get_int("model_calls", self.model_calls)
        self.tool_calls = v.get_int("tool_calls", self.tool_calls)
        self.tool_errors = v.get_int("tool_errors", self.tool_errors)
        self.mcp_calls = v.get_int("mcp_calls", self.mcp_calls)
        self.http_requests = v.get_int("http_requests", self.http_requests)
        self.http_retries = v.get_int("http_retries", self.http_retries)
        self.prompt_tokens = v.get_int("prompt_tokens", self.prompt_tokens)
        self.completion_tokens = v.get_int("completion_tokens", self.completion_tokens)
        self.evictions = v.get_int("evictions", self.evictions)
        self.summarizations = v.get_int("summarizations", self.summarizations)
        self.subagent_runs = v.get_int("subagent_runs", self.subagent_runs)
        self.wall_ms = v.get_int("wall_ms", self.wall_ms)

    def write_to(self, mut w: Some[Writer]):
        """One line for stderr: "model_calls=3 tool_calls=5 ... wall_ms=1234"."""
        w.write("model_calls=", self.model_calls, " tool_calls=", self.tool_calls,
                " tool_errors=", self.tool_errors, " mcp_calls=", self.mcp_calls,
                " http_requests=", self.http_requests, " http_retries=", self.http_retries,
                " prompt_tokens=", self.prompt_tokens, " completion_tokens=", self.completion_tokens,
                " evictions=", self.evictions, " summarizations=", self.summarizations,
                " subagent_runs=", self.subagent_runs, " wall_ms=", self.wall_ms)


def zero_stats() -> RunStats:
    return RunStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
