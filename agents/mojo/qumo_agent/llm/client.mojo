"""OpenAI-compatible chat-completions client over native HTTP(S) (DESIGN.md 2.5).

One POST to `base_url + "/chat/completions"` per attempt, with the openai Python SDK's
retry policy (3.45.0, behaviour only, Apache-2.0; no code copied): `1 + max_retries`
attempts; retry on connection errors and timeouts, and on responses for which
`should_retry` holds (retry-after above 120 s never; `x-should-retry: true|false`
obeyed; else 408, 409, 429 or >= 500). The wait is the server's `retry-after-ms` /
`retry-after` when it is in (0, 120] seconds, else `min(0.5 * 2**n, 8) * (1 - 0.25 * u)`
seconds. Timeouts apply to each connect/read/write operation (httpx's semantics).

The API key is read from the environment when each request is built, and is never
stored, logged or returned; every error text produced from the server's answer or the
transport goes through `redact_all`.
"""

from std.memory import ArcPointer
from std.os import getenv
from std.math import isfinite

from qumo.ai.json import JsonValue, parse_json
from qumo_agent.core.errors import E_MODEL_HTTP, E_CONTEXT_OVERFLOW, E_NETWORK, E_CONFIG, error_code
from qumo_agent.core.endpoint import ModelEndpoint
from qumo_agent.core.messages import ChatRequest, ChatResponse, parse_chat_response, is_context_overflow
from qumo_agent.core.stats import RunStats, zero_stats
from qumo_agent.native.clock import sleep_ms, http_date_to_unix, unix_now
from qumo_agent.native.rand import random_unit
from qumo_agent.util.redact import redact_all, secret_env_names
from qumo_agent.net.url import parse_url
from qumo_agent.net.tls import TlsContext
from qumo_agent.net.http import HttpClient, HttpRequest, HttpOptions, HttpResponse, REDIRECT_KEEP_METHOD

comptime MAX_RETRY_AFTER_S = 120.0       # openai MAX_RETRY_AFTER_DELAY
comptime INITIAL_RETRY_DELAY_S = 0.5
comptime MAX_RETRY_DELAY_S = 8.0
comptime MAX_MODEL_BODY_BYTES = 64 * 1024 * 1024
comptime USER_AGENT = "qumo-agent/0.2 (Mojo)"
comptime ERROR_MESSAGE_CODE_POINTS = 600


def _is_valid_env_name(s: String) -> Bool:
    var bs = s.as_bytes()
    if len(bs) == 0:
        return False
    for i in range(len(bs)):
        var c = Int(bs[i])
        var alpha = (c >= 0x41 and c <= 0x5A) or (c >= 0x61 and c <= 0x7A) or c == 0x5F
        var digit = c >= 0x30 and c <= 0x39
        if not alpha and not (digit and i > 0):
            return False
    return True


def _missing_key_error(env: String) -> Error:
    return Error(E_CONFIG + ": set " + env + " to use the openrouter back end, or pass --replay FILE")


def _header(names: List[String], values: List[String], name: String) -> Optional[String]:
    var want = name.lower()
    for i in range(min(len(names), len(values))):
        if names[i].lower() == want:
            return values[i].copy()
    return None


def _is_ascii_space(c: Int) -> Bool:
    return c == 0x20 or (c >= 0x09 and c <= 0x0D)


def _digits(bs: Span[Byte, _], mut i: Int, mut clean: List[UInt8]) -> Int:
    """Consumes digits (single underscores between digits allowed) from bs[i:], appending
    the digits to `clean`; returns how many digits were read."""
    var count = 0
    while i < len(bs):
        var c = Int(bs[i])
        if c >= 0x30 and c <= 0x39:
            clean.append(UInt8(c))
            count += 1
            i += 1
        elif c == 0x5F and count > 0 and i + 1 < len(bs) and Int(bs[i + 1]) >= 0x30 and Int(bs[i + 1]) <= 0x39:
            i += 1
        else:
            break
    return count


def _py_float(text: String) -> Optional[Float64]:
    """Python's float(str) for ASCII text: surrounding whitespace, sign, digits with
    single underscores between them, fraction, exponent, inf/infinity/nan."""
    var t = String(text.strip()).lower()
    var bs = t.as_bytes()
    var n = len(bs)
    if n == 0:
        return None
    var i = 0
    var neg = False
    if Int(bs[0]) == 0x2B or Int(bs[0]) == 0x2D:
        neg = Int(bs[0]) == 0x2D
        i = 1
    var rest = String(t[byte=i:])
    if rest == "inf" or rest == "infinity":
        var inf = Float64.MAX * 2.0
        return -inf if neg else inf
    if rest == "nan":
        return Float64.MAX * 2.0 - Float64.MAX * 2.0
    # digitpart ("." digitpart?)? | "." digitpart, then optional exponent.
    var clean = List[UInt8]()
    if neg:
        clean.append(0x2D)

    var intd = _digits(bs, i, clean)
    var fracd = 0
    if i < n and Int(bs[i]) == 0x2E:
        clean.append(0x2E)
        i += 1
        fracd = _digits(bs, i, clean)
    if intd == 0 and fracd == 0:
        return None
    if i < n and Int(bs[i]) == 0x65:
        clean.append(0x65)
        i += 1
        if i < n and (Int(bs[i]) == 0x2B or Int(bs[i]) == 0x2D):
            clean.append(bs[i])
            i += 1
        if _digits(bs, i, clean) == 0:
            return None
    if i != n:
        return None
    if intd == 0:
        # Mojo's parser wants a digit before the dot.
        var fixed = List[UInt8]()
        for k in range(len(clean)):
            if Int(clean[k]) == 0x2E and (k == 0 or Int(clean[k - 1]) == 0x2D):
                fixed.append(0x30)
            fixed.append(clean[k])
        clean = fixed^
    try:
        return Float64(String(from_utf8_lossy=Span(clean)))
    except:
        return None


def parse_retry_after_s(header_names: List[String], header_values: List[String], now_unix: Int) -> Optional[Float64]:
    """Seconds to wait as the server asks: `retry-after-ms` / 1000, else `retry-after` as
    float seconds, else `retry-after` as an HTTP date minus now; None when unspecified.
    An overflowing number (not an infinity literal) gives 121 (more than the 120 s cap)."""
    var specs: List[String] = ["retry-after-ms", "retry-after"]
    for k in range(2):
        var v = _header(header_names, header_values, specs[k])
        if not v:
            continue
        var f = _py_float(v.value())
        if not f:
            continue
        var delay = f.value()
        if not isfinite(delay) and delay > 0.0:   # +inf from an overflowing number
            var lit = String(v.value().strip()).lower()
            if lit != "inf" and lit != "+inf" and lit != "infinity" and lit != "+infinity":
                return MAX_RETRY_AFTER_S + 1.0
        return delay / (1000.0 if k == 0 else 1.0)
    var ra = _header(header_names, header_values, "retry-after")
    if not ra:
        return None
    var when = http_date_to_unix(ra.value())
    if not when:
        return None
    return Float64(when.value() - now_unix)


def should_retry(status: Int, headers_names: List[String], headers_values: List[String], now_unix: Int) -> Bool:
    """openai SDK `_should_retry` for a non-2xx response."""
    var ra = parse_retry_after_s(headers_names, headers_values, now_unix)
    if ra and isfinite(ra.value()) and ra.value() > MAX_RETRY_AFTER_S:
        return False
    var hdr = _header(headers_names, headers_values, "x-should-retry")
    if hdr:
        if hdr.value() == "true":
            return True
        if hdr.value() == "false":
            return False
    return status == 408 or status == 409 or status == 429 or status >= 500


def retry_delay_ms(retries_done: Int, header_names: List[String], header_values: List[String], now_unix: Int,
                   unit_random: Float64) -> Int:
    """openai SDK `_calculate_retry_timeout`, in milliseconds (rounded)."""
    var ra = parse_retry_after_s(header_names, header_values, now_unix)
    if ra and isfinite(ra.value()) and ra.value() > 0.0 and ra.value() <= MAX_RETRY_AFTER_S:
        return Int(ra.value() * 1000.0 + 0.5)
    var n = min(retries_done, 1000)
    var sleep = INITIAL_RETRY_DELAY_S
    for _ in range(n):
        sleep *= 2.0
        if sleep >= MAX_RETRY_DELAY_S:
            break
    sleep = min(sleep, MAX_RETRY_DELAY_S)
    var timeout = sleep * (1.0 - 0.25 * unit_random)
    if timeout < 0.0:
        timeout = 0.0
    return Int(timeout * 1000.0 + 0.5)


def _first_code_points(s: String, n: Int) -> String:
    var bs = s.as_bytes()
    var count = 0
    var i = 0
    while i < len(bs) and count < n:
        i += 1
        while i < len(bs) and (Int(bs[i]) & 0xC0) == 0x80:
            i += 1
        count += 1
    return String(s[byte=0:i])


def _error_message(body: String) -> String:
    """`error.message` of a JSON body, else the first 600 code points of the body."""
    try:
        var doc = parse_json(body)
        if doc.is_object():
            var e = doc.get("error")
            if e.is_defined():
                if e.is_object():
                    var m = e.get("message")
                    if m.is_defined() and m.is_string():
                        return m.as_string()
                elif e.is_string():
                    return e.as_string()
    except:
        pass
    return _first_code_points(body, ERROR_MESSAGE_CODE_POINTS)


def _retryable_network_error(e: Error) -> Bool:
    """Connection errors and timeouts (QA-E104) are retried; request-side refusals are not."""
    var text = String(e)
    if error_code(e) != E_NETWORK:
        return False
    var permanent: List[String] = ["header injection refused", "invalid URL", "refusing", "redirect loop",
                                   "response body larger than"]
    for p in permanent:
        if text.find(p) >= 0:
            return False
    return True


struct LlmClient(Movable):
    var endpoint: ModelEndpoint
    var http: HttpClient
    var stats: RunStats                     # http_requests, http_retries

    def __init__(out self, endpoint: ModelEndpoint, tls: Optional[ArcPointer[TlsContext]]) raises:
        """Checks the key variable (QA-E140) and the base URL. Plain `http` is used only
        when `base_url` is itself `http://` (local model servers); for `https` without
        `tls` the production TLS context is created on the first request."""
        if not _is_valid_env_name(endpoint.api_key_env):
            raise Error(E_CONFIG + ": '" + endpoint.api_key_env + "' is not a valid environment variable name")
        if getenv(endpoint.api_key_env, "").byte_length() == 0:
            raise _missing_key_error(endpoint.api_key_env)
        try:
            _ = parse_url(endpoint.base_url)
        except e:
            raise Error(E_CONFIG + ": invalid model base_url: " + String(e))
        self.endpoint = endpoint.copy()
        self.http = HttpClient(tls)
        self.stats = zero_stats()

    def _url(self) -> String:
        var base = self.endpoint.base_url.copy()
        while base.endswith("/"):
            var cut = String(base[byte=0:base.byte_length() - 1])
            base = cut^
        return base + "/chat/completions"

    def _options(self) -> HttpOptions:
        var ms = self.endpoint.timeout_s * 1000
        var plain = self.endpoint.base_url.lower().startswith("http://")
        return HttpOptions(ms, ms, MAX_MODEL_BODY_BYTES, False, REDIRECT_KEEP_METHOD, 10, plain, False)

    def _request(self, body: String) raises -> HttpRequest:
        var key = getenv(self.endpoint.api_key_env, "")
        if key.byte_length() == 0:
            raise _missing_key_error(self.endpoint.api_key_env)
        var names: List[String] = ["Authorization", "Content-Type", "Accept"]
        var values: List[String] = ["Bearer " + key, "application/json", "application/json"]
        var has_ua = False
        for i in range(min(len(self.endpoint.header_names), len(self.endpoint.header_values))):
            if self.endpoint.header_names[i].lower() == "user-agent":
                has_ua = True
            names.append(self.endpoint.header_names[i].copy())
            values.append(self.endpoint.header_values[i].copy())
        if not has_ua:
            names.append("User-Agent")
            values.append(USER_AGENT)
        return HttpRequest("POST", self._url(), names^, values^, body)

    def _redact(self, text: String) -> String:
        return redact_all(text, secret_env_names(self.endpoint.api_key_env))

    def complete(mut self, req: ChatRequest) raises -> ChatResponse:
        """One chat completion with retries; raises QA-E101/E102/E103/E104 (redacted)."""
        var body = req.body_json()
        var opts = self._options()
        var attempts = 1 + max(self.endpoint.max_retries, 0)
        for attempt in range(attempts):
            var remaining = attempts - 1 - attempt
            var before = self.http.requests
            var resp: HttpResponse
            try:
                resp = self.http.request(self._request(body), opts)
            except e:
                self.stats.http_requests += self.http.requests - before
                if remaining > 0 and _retryable_network_error(e):
                    self.stats.http_retries += 1
                    sleep_ms(retry_delay_ms(attempt, List[String](), List[String](), unix_now()[0], random_unit()))
                    continue
                raise Error(self._redact(String(e)))
            self.stats.http_requests += self.http.requests - before
            if resp.status >= 200 and resp.status < 300:
                return parse_chat_response(resp.text())
            var now = unix_now()[0]
            if remaining > 0 and should_retry(resp.status, resp.header_names, resp.header_values, now):
                self.stats.http_retries += 1
                sleep_ms(retry_delay_ms(attempt, resp.header_names, resp.header_values, now, random_unit()))
                continue
            var text = resp.text()
            var message = self._redact(_error_message(text))
            if is_context_overflow(resp.status, text):
                raise Error(E_CONTEXT_OVERFLOW + ": context overflow: HTTP " + String(resp.status) + ": " + message)
            var reason = self._redact(resp.reason)
            raise Error(E_MODEL_HTTP + ": model request failed: HTTP " + String(resp.status) + " " + reason + ": " + message)
        raise Error(E_NETWORK + ": model request failed after " + String(attempts) + " attempts")
