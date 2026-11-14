"""Secret redaction for every error text and log line built from remote or child
data (DESIGN.md 2.2.4)."""

from std.os import getenv


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


def secret_env_names(api_key_env: String) -> List[String]:
    """[api_key_env, "ALPHAXIV_API_KEY", "GITHUB_TOKEN"], empty names and duplicates removed."""
    var out = List[String]()
    var cands: List[String] = [api_key_env, "ALPHAXIV_API_KEY", "GITHUB_TOKEN"]
    for c in cands:
        if c.byte_length() == 0:
            continue
        var dup = False
        for o in out:
            if o == c:
                dup = True
                break
        if not dup:
            out.append(c)
    return out^


def redact_all(s: String, env_names: List[String]) -> String:
    """`redact(s, value)` for the current value of every variable in `env_names` that
    is set and non-empty (read with getenv at each call, never cached), then the
    `sk-`/`Bearer ` patterns."""
    var t = s
    for name in env_names:
        if name.byte_length() == 0:
            continue
        var v = getenv(name, "")
        if v.byte_length() == 0:
            continue
        # Only the value replacement here; the key-shape pass runs once below.
        if v.byte_length() >= 8:
            t = t.replace(v, "[REDACTED]")
    return redact(t, "")
