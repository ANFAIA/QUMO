"""Secret redaction for every error text and log line built from remote or child
data (DESIGN.md 2.2.4)."""

from std.os import getenv

from qumo.core.json import json_quote
from qumo_agent.native.spawn import environ_list, is_secret_env_name


comptime REDACTED = "[REDACTED]"
# Length rule for a known value. 8 bytes or more: hidden wherever it occurs. 4 to 7: only as
# a whole word (not next to a letter, digit, `_` or `-`), so a short value does not blank out
# pieces of ordinary words. Under 4: never (it would mark every occurrence of two or three
# common characters, which hides nothing and ruins the text).
comptime MIN_SUBSTRING_SECRET = 8
comptime MIN_WORD_SECRET = 4


def _is_key_char(c: Int) -> Bool:
    return (c >= 0x30 and c <= 0x39) or (c >= 0x41 and c <= 0x5A) or (c >= 0x61 and c <= 0x7A) or c == 0x2D or c == 0x5F


def _is_header_value_char(c: Int) -> Bool:
    """Key characters plus `. ~ + / =` (JSON Web Tokens, base64)."""
    return _is_key_char(c) or c == 0x2E or c == 0x7E or c == 0x2B or c == 0x2F or c == 0x3D


def _scheme_at(bs: Span[UInt8, _], i: Int, word: StringSlice) -> Bool:
    """True when `word` (lower case, ASCII) is at byte `i`, in any letter case."""
    var ws = word.as_bytes()
    var m = len(ws)
    if i + m > len(bs):
        return False
    for k in range(m):
        var c = Int(bs[i + k])
        if c >= 0x41 and c <= 0x5A:
            c = c | 0x20
        if c != Int(ws[k]):
            return False
    return True


def _looks_opaque(bs: Span[UInt8, _], lo: Int, hi: Int) -> Bool:
    """A word that reads like a key, not like prose: 20 bytes or more, or 8 or more with a
    digit, one of `+ / =`, or a capital after the first byte next to a lower-case letter."""
    var n = hi - lo
    if n >= 20:
        return True
    if n < 8:
        return False
    var upper_inside = False
    var lower = False
    for k in range(lo, hi):
        var c = Int(bs[k])
        if (c >= 0x30 and c <= 0x39) or c == 0x2B or c == 0x2F or c == 0x3D:
            return True
        if c >= 0x41 and c <= 0x5A and k > lo:
            upper_inside = True
        if c >= 0x61 and c <= 0x7A:
            lower = True
    return upper_inside and lower


def _replace_word(s: String, word: String) -> String:
    """Every occurrence of `word` that is not next to a key character."""
    var bs = s.as_bytes()
    var ws = word.as_bytes()
    var n = len(bs)
    var m = len(ws)
    var out = String()
    var i = 0
    var run = 0
    while i + m <= n:
        var same = True
        for k in range(m):
            if bs[i + k] != ws[k]:
                same = False
                break
        if same and (i == 0 or not _is_key_char(Int(bs[i - 1]))) and (i + m == n or not _is_key_char(Int(bs[i + m]))):
            out += String(unsafe_from_utf8=bs[run:i]) + REDACTED
            i += m
            run = i
        else:
            i += 1
    out += String(unsafe_from_utf8=bs[run:n])
    return out^


def _hide_form(s: String, form: String) -> String:
    if form.byte_length() >= MIN_SUBSTRING_SECRET:
        return s.replace(form, REDACTED)
    if form.byte_length() >= MIN_WORD_SECRET:
        return _replace_word(s, form)
    return s


def _hide_value(s: String, value: String) -> String:
    """`value` and its JSON-escaped form (what a server shows when it echoes the value
    inside a JSON string), by the length rule above."""
    if value.byte_length() < MIN_WORD_SECRET:
        return s
    var t = s
    var quoted = json_quote(value)
    var escaped = String(quoted[byte=1:quoted.byte_length() - 1])
    if escaped != value:
        t = _hide_form(t, escaped)
    return _hide_form(t, value)


def redact(s: String, secret: String = "") -> String:
    """Hide the secret itself (also JSON-escaped) and anything shaped like a credential:
    `sk-...`, and what follows `Bearer ` or `Basic ` in any letter case."""
    var t = _hide_value(s, secret)
    var bs = t.as_bytes()
    var n = len(bs)
    var out = String()
    var i = 0
    while i < n:
        var hit = 0
        var header = False
        var basic = False
        if i + 3 <= n and Int(bs[i]) == ord("s") and Int(bs[i + 1]) == ord("k") and Int(bs[i + 2]) == ord("-"):
            hit = 3
        elif _scheme_at(bs, i, "bearer "):
            hit = 7
            header = True
        elif _scheme_at(bs, i, "basic "):
            hit = 6
            header = True
            basic = True
        if hit > 0 and (i == 0 or not _is_key_char(Int(bs[i - 1]))):
            var j = i + hit
            while j < n and (_is_header_value_char(Int(bs[j])) if header else _is_key_char(Int(bs[j]))):
                j += 1
            # `Basic` is also an ordinary word: its value must look like base64, not prose
            if j - i - hit >= 8 and (not basic or _looks_opaque(bs, i + hit, j)):
                out += String(unsafe_from_utf8=bs[i:i + hit]) + REDACTED
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
    """Hides the current value (read with getenv at each call, never cached) of every
    variable in `env_names` and of every variable of this process whose name looks secret
    (`is_secret_env_name`: `*_API_KEY`, `*_TOKEN`, `*SECRET*`, `*PASSWORD*`), longest value
    first, each also in its JSON-escaped form; then the `sk-`/`Bearer `/`Basic ` patterns."""
    var names = List[String]()
    for name in env_names:
        names.append(name)
    for entry in environ_list():
        var eq = entry.find("=")
        if eq > 0:
            var name = String(entry[byte=0:eq])
            if is_secret_env_name(name):
                names.append(name^)
    var values = List[String]()
    for name in names:
        if name.byte_length() == 0:
            continue
        var v = getenv(name, "")
        if v.byte_length() < MIN_WORD_SECRET:
            continue
        # insertion by decreasing length; duplicates skipped
        var at = len(values)
        var dup = False
        for i in range(len(values)):
            if values[i] == v:
                dup = True
                break
            if values[i].byte_length() < v.byte_length():
                at = i
                break
        if not dup:
            values.insert(at, v)
    var t = s
    for v in values:
        # only the value replacement here; the key-shape pass runs once below
        t = _hide_value(t, v)
    return redact(t, "")
