"""Stable error codes of the agent harness (`QA-Ennn`) and helpers to build and read them.

Every error the harness and the application raise starts with one of these codes and
ends with `Hint: ...` when a fix exists (QUMO convention; texts in DESIGN.md
Appendix B.2). The codes are listed in docs/ERRORS.md.
"""

comptime E_MODEL_HTTP = "QA-E101"       # model endpoint answered non-2xx after retries
comptime E_CONTEXT_OVERFLOW = "QA-E102" # 400/413/422 with an overflow marker (2.5)
comptime E_MODEL_RESPONSE = "QA-E103"   # 200 with "error", missing/null/empty choices (B5)
comptime E_NETWORK = "QA-E104"          # DNS, connect, TLS, timeout (after retries for the model)
comptime E_TLS_SETUP = "QA-E105"        # OpenSSL not found/too old, trust store unusable
comptime E_RECURSION = "QA-E110"        # recursion limit reached
comptime E_SUBAGENT = "QA-E120"         # subagent child failed
comptime E_MCP = "QA-E130"              # MCP transport or protocol error
comptime E_CONFIG = "QA-E140"           # unknown agent, missing key or file, stale export
comptime E_TOOL_REFUSED = "QA-E150"     # generated tool refused (hash, review)
comptime E_SPAWN = "QA-E160"            # process start/IO errors (E161 line too long, E162 no chdir action, E163 OS confinement)
comptime E_REPLAY = "QA-E170"           # replay file exhausted or malformed

# Fixed prefixes of the E101/E102 texts (B.2), followed by the HTTP status.
comptime _E101_PREFIX = "QA-E101: model request failed: HTTP "
comptime _E102_PREFIX = "QA-E102: context overflow: HTTP "


def qa_error(code: String, message: String, hint: String) -> Error:
    """`<code>: <message>`, plus `. Hint: <hint>` when `hint` is not empty."""
    var text = code + ": " + message
    if hint.byte_length() > 0:
        text += ". Hint: " + hint
    return Error(text)


def _is_digit(c: Int) -> Bool:
    return c >= 0x30 and c <= 0x39


def _code_of(text: String) -> String:
    var bs = text.as_bytes()
    if len(bs) < 8:
        return String()
    if not text.startswith("QA-E"):
        return String()
    for i in range(4, 7):
        if not _is_digit(Int(bs[i])):
            return String()
    if Int(bs[7]) != 0x3A:   # ':'
        return String()
    return String(unsafe_from_utf8=bs[0:7])


def error_code(e: Error) -> String:
    """The "QA-Ennn" prefix of the error text, or ""."""
    return _code_of(String(e))


def error_text(e: Error) -> String:
    return String(e)


def error_http_status(e: Error) -> Int:
    """For E101/E102 errors, the <status> written after "HTTP " (B.2); -1 otherwise."""
    var text = String(e)
    var prefix_len: Int
    if text.startswith(_E101_PREFIX):
        prefix_len = _E101_PREFIX.byte_length()
    elif text.startswith(_E102_PREFIX):
        prefix_len = _E102_PREFIX.byte_length()
    else:
        return -1
    var bs = text.as_bytes()
    var i = prefix_len
    var v = 0
    var digits = 0
    while i < len(bs) and _is_digit(Int(bs[i])) and digits < 4:
        v = v * 10 + (Int(bs[i]) - 0x30)
        digits += 1
        i += 1
    if digits == 0:
        return -1
    return v
