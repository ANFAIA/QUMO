"""SpeQtrum (Qilimanjaro) client in native Mojo: login, device list, execute, job status, results.

Native counterpart of plugins/hardware/qumo_hardware/speqtrum.py (the Python fallback). Same
wire protocol, same security rules, no Python on the request path:

    POST {base}/authorisation-tokens   {grantType, assertion, scope} -> {accessToken, ...}
    GET  {base}/devices                -> {"items": [{code, nqubits, name, description, type, status}]}
    POST {base}/execute                {device_code, payload, job_type, meta} -> {"id", "message"}
    GET  {base}/jobs/{id}?payload=false&result=true&error=true

* Credentials are arguments (the adapter reads them from the environment or its options).
  The access token is created and used inside one call and never leaves this module: every
  call logs in first (one extra request per call, no token cache to leak).
* A 401 on an authenticated request means it was rejected before acting: log in again and
  retry once, so a retried POST /execute cannot create a second job.
* https only; plain http only to a loopback host (tests). No user information in the URL.
  Redirects are not followed, so the bearer token never reaches a Location host.
* Responses are data. Result texts pass the refusal rule (`unsafe_construct`): no YAML tag,
  anchor, alias, block style or second document. Refusals raise QM-E955 "... was not loaded".
* Server text in messages is cleaned: secrets redacted, whitespace collapsed, 160 bytes at most.
* The qumo.payload.v1 document and the execute payload are byte-identical to Python's
  json.dumps(..., sort_keys=True) output (ensure_ascii escaping, ", " and ": " separators).
"""

from std.collections import Dict

from qumo.core.json import JsonValue, parse_json
from qumo.net.http import HttpClient, HttpOptions, HttpRequest, HttpResponse, REDIRECT_NONE
from qumo.net.libc import c_time
from qumo.net.url import Url, parse_url

comptime GRANT_TYPE = "urn:ietf:params:oauth:grant-type:jwt-bearer"
comptime PAYLOAD_TYPE = "digital_propagation"
comptime PAYLOAD_FORMAT_V1 = "qumo.payload.v1"
comptime ENV_URL = "QUMO_SPEQTRUM_URL"
comptime ENV_USERNAME = "QUMO_SPEQTRUM_USERNAME"
comptime ENV_APIKEY = "QUMO_SPEQTRUM_APIKEY"
comptime E_CREDENTIALS = "QM-E952"
comptime E_INVALID_PAYLOAD = "QM-E955"
comptime E_UNKNOWN_JOB = "QM-E958"
comptime E_JOB = "QM-E959"
comptime TIMEOUT_MS = 30000
comptime MAX_RESPONSE_BYTES = 64 * 1024 * 1024


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def _hex4(v: Int) -> String:
    var out = List[UInt8](capacity=4)
    var shift = 12
    while shift >= 0:
        var d = (v >> shift) & 15
        out.append(UInt8(48 + d) if d < 10 else UInt8(87 + d))  # lower-case hex, as Python writes it
        shift -= 4
    return String(from_utf8_lossy=Span(out))


def py_json_string(s: String) -> String:
    """Python json.dumps(str) with ensure_ascii=True: the exact same bytes."""
    var bs = s.as_bytes()
    var n = len(bs)
    var out = String("\"")
    var i = 0
    while i < n:
        var b = Int(bs[i])
        var cp = b
        var width = 1
        if b >= 0xF0 and i + 3 < n:
            cp = ((b & 0x07) << 18) | ((Int(bs[i + 1]) & 0x3F) << 12) | ((Int(bs[i + 2]) & 0x3F) << 6) | (Int(bs[i + 3]) & 0x3F)
            width = 4
        elif b >= 0xE0 and i + 2 < n:
            cp = ((b & 0x0F) << 12) | ((Int(bs[i + 1]) & 0x3F) << 6) | (Int(bs[i + 2]) & 0x3F)
            width = 3
        elif b >= 0xC0 and i + 1 < n:
            cp = ((b & 0x1F) << 6) | (Int(bs[i + 1]) & 0x3F)
            width = 2
        i += width
        if cp == 0x22:
            out += "\\\""
        elif cp == 0x5C:
            out += "\\\\"
        elif cp == 0x0A:
            out += "\\n"
        elif cp == 0x0D:
            out += "\\r"
        elif cp == 0x09:
            out += "\\t"
        elif cp == 0x08:
            out += "\\b"
        elif cp == 0x0C:
            out += "\\f"
        elif cp < 0x20 or cp > 0x7E:
            if cp > 0xFFFF:
                var v = cp - 0x10000
                out += "\\u" + _hex4(0xD800 | (v >> 10)) + "\\u" + _hex4(0xDC00 | (v & 0x3FF))
            else:
                out += "\\u" + _hex4(cp)
        else:
            out += chr(cp)
    return out + "\""


def _cut(s: String, limit: Int) -> String:
    """At most `limit` bytes of `s`, never splitting a UTF-8 sequence."""
    var bs = s.as_bytes()
    if len(bs) <= limit:
        return s
    var end = limit
    while end > 0 and (Int(bs[end]) & 0xC0) == 0x80:
        end -= 1
    return String(unsafe_from_utf8=bs[0:end])


def _clean(value: String, secrets: List[String]) -> String:
    """Server text for a message: secrets redacted, whitespace collapsed, 160 bytes at most."""
    var text = value
    for sec in secrets:
        if sec.byte_length() > 0:
            text = text.replace(sec, "<redacted>")
    var out = List[UInt8]()
    var pending_space = False
    for b in text.as_bytes():
        if b == 32 or b == 9 or b == 10 or b == 13 or b == 11 or b == 12:
            pending_space = len(out) > 0
            continue
        if pending_space:
            out.append(32)
            pending_space = False
        out.append(b)
    return _cut(String(from_utf8_lossy=Span(out)), 160)


# ---------------------------------------------------------------------------
# base64
# ---------------------------------------------------------------------------
comptime _B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"


def b64url_nopad(data: String) -> String:
    """base64url without '=' padding (Python urlsafe_b64encode(...).rstrip(b'='))."""
    var bs = data.as_bytes()
    var n = len(bs)
    var out = List[UInt8](capacity=(n + 2) // 3 * 4)
    var alpha = _B64.as_bytes()
    var i = 0
    while i < n:
        var b0 = Int(bs[i])
        var b1 = Int(bs[i + 1]) if i + 1 < n else 0
        var b2 = Int(bs[i + 2]) if i + 2 < n else 0
        var v = (b0 << 16) | (b1 << 8) | b2
        var chars = 4 if i + 2 < n else (3 if i + 1 < n else 2)
        for k in range(chars):
            var idx = (v >> (18 - 6 * k)) & 63
            var c = alpha[idx]
            if c == 43:
                c = 45  # '+' -> '-'
            elif c == 47:
                c = 95  # '/' -> '_'
            out.append(c)
        i += 3
    return String(from_utf8_lossy=Span(out))


def _b64val(c: Int) -> Int:
    if c >= 65 and c <= 90:
        return c - 65
    if c >= 97 and c <= 122:
        return c - 71
    if c >= 48 and c <= 57:
        return c + 4
    if c == 43:
        return 62
    if c == 47:
        return 63
    return -1


def b64decode_strict(text: String) raises -> List[UInt8]:
    """Standard base64 with padding, validated (Python b64decode(..., validate=True))."""
    var bs = text.as_bytes()
    var n = len(bs)
    if n % 4 != 0:
        raise Error("bad base64 length")
    var out = List[UInt8](capacity=n // 4 * 3)
    var i = 0
    while i < n:
        var pad = 0
        var v = 0
        for k in range(4):
            var c = Int(bs[i + k])
            if c == 61:  # '='
                if i + 4 != n or k < 2:
                    raise Error("bad base64 padding")
                pad += 1
                v = v << 6
                continue
            if pad > 0:
                raise Error("bad base64 padding")
            var d = _b64val(c)
            if d < 0:
                raise Error("bad base64 character")
            v = (v << 6) | d
        out.append(UInt8((v >> 16) & 255))
        if pad < 2:
            out.append(UInt8((v >> 8) & 255))
        if pad < 1:
            out.append(UInt8(v & 255))
        i += 4
    return out^


# ---------------------------------------------------------------------------
# Refusal rule (port of speqtrum.unsafe_construct) and strict JSON loading
# ---------------------------------------------------------------------------
def _token_at(bs: Span[Byte, _], i: Int) -> String:
    var j = i
    var n = len(bs)
    while j < n and j - i < 48:
        var c = Int(bs[j])
        if c == 32 or c == 9 or c == 13 or c == 10 or c == 44 or c == 91 or c == 93 or c == 123 or c == 125 or c == 34:
            break
        j += 1
    if j == i:
        j = i + 1
    var raw = List[UInt8]()
    for k in range(i, j):
        raw.append(bs[k])
    return String(from_utf8_lossy=Span(raw))


def unsafe_construct(text: String) -> String:
    """The first construct QUMO refuses to load in `text`, or "" when it is acceptable."""
    var bs = text.as_bytes()
    var n = len(bs)
    var blank = True
    for b in bs:
        if not (b == 32 or b == 9 or b == 10 or b == 13 or b == 11 or b == 12):
            blank = False
            break
    if blank:
        return ""
    var in_string = False
    var escaped = False
    for i in range(n):
        var ch = Int(bs[i])
        if in_string:
            if escaped:
                escaped = False
            elif ch == 92:
                escaped = True
            elif ch == 34:
                in_string = False
            continue
        var col0 = i == 0 or Int(bs[i - 1]) == 10
        if col0 and i + 3 <= n:
            var m0 = Int(bs[i])
            if (m0 == 45 or m0 == 46) and Int(bs[i + 1]) == m0 and Int(bs[i + 2]) == m0:
                var nxt = Int(bs[i + 3]) if i + 3 < n else -1
                if nxt == -1 or nxt == 32 or nxt == 9 or nxt == 13 or nxt == 10:
                    return "several YAML documents (a '---' or '...' line)"
        if ch == 34:
            in_string = True
        elif ch == 33:
            return "a YAML tag `" + _token_at(bs, i) + "`"
        elif ch == 38:
            return "a YAML anchor `" + _token_at(bs, i) + "`"
        elif ch == 42:
            return "a YAML alias `" + _token_at(bs, i) + "`"
    var k = 0
    while k < n and (bs[k] == 32 or bs[k] == 9 or bs[k] == 10 or bs[k] == 13 or bs[k] == 11 or bs[k] == 12):
        k += 1
    if k >= n or bs[k] != 123:
        return "block-style YAML (the document must be one flow object that starts with '{')"
    return ""


def load_document(text: String, what: String, job_id: String) raises -> JsonValue:
    """One JSON object, reached only through the refusal rule."""
    var construct = unsafe_construct(text)
    if construct.byte_length() > 0:
        raise Error(E_INVALID_PAYLOAD + ": SpeQtrum " + what + " of job " + job_id + " was not loaded: it contains "
                    + construct + ". Hint: QUMO loads only plain JSON results; report the job id to SpeQtrum")
    var bs = text.as_bytes()
    var n = len(bs)
    var start = 0
    while start < n and (bs[start] == 32 or bs[start] == 9 or bs[start] == 10 or bs[start] == 13 or bs[start] == 11
                         or bs[start] == 12):
        start += 1
    if start >= n:
        raise Error(E_INVALID_PAYLOAD + ": SpeQtrum " + what + " of job " + job_id + " is not valid JSON")
    # end of the first top-level value: brackets balanced outside strings
    var depth = 0
    var in_string = False
    var escaped = False
    var end = -1
    for i in range(start, n):
        var c = Int(bs[i])
        if in_string:
            if escaped:
                escaped = False
            elif c == 92:
                escaped = True
            elif c == 34:
                in_string = False
            continue
        if c == 34:
            in_string = True
        elif c == 123 or c == 91:
            depth += 1
        elif c == 125 or c == 93:
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        raise Error(E_INVALID_PAYLOAD + ": SpeQtrum " + what + " of job " + job_id + " is not valid JSON")
    var rest = end
    while rest < n and (bs[rest] == 32 or bs[rest] == 9 or bs[rest] == 10 or bs[rest] == 13 or bs[rest] == 11
                        or bs[rest] == 12):
        rest += 1
    if rest < n:
        if bs[rest] == 123 or bs[rest] == 91:
            raise Error(E_INVALID_PAYLOAD + ": SpeQtrum " + what + " of job " + job_id + " was not loaded: it contains "
                        + "several documents (a second JSON value follows the first). Hint: QUMO loads only plain "
                        + "JSON results; report the job id to SpeQtrum")
        raise Error(E_INVALID_PAYLOAD + ": SpeQtrum " + what + " of job " + job_id + " is not valid JSON")
    try:
        return parse_json(String(unsafe_from_utf8=bs[start:end]))
    except e:
        raise Error(E_INVALID_PAYLOAD + ": SpeQtrum " + what + " of job " + job_id + " is not valid JSON")


def _valid_utf8(bs: List[UInt8]) -> Bool:
    """Strict UTF-8 (no overlong forms, no surrogates, at most U+10FFFF), as Python's decode('utf-8')."""
    var n = len(bs)
    var i = 0
    while i < n:
        var b = Int(bs[i])
        if b < 0x80:
            i += 1
            continue
        var need: Int
        var cp: Int
        if b >= 0xC2 and b <= 0xDF:
            need = 1
            cp = b & 0x1F
        elif b >= 0xE0 and b <= 0xEF:
            need = 2
            cp = b & 0x0F
        elif b >= 0xF0 and b <= 0xF4:
            need = 3
            cp = b & 0x07
        else:
            return False
        if i + need >= n:  # the continuation bytes are at i+1 .. i+need
            return False
        for k in range(1, need + 1):
            var c = Int(bs[i + k])
            if (c & 0xC0) != 0x80:
                return False
            cp = (cp << 6) | (c & 0x3F)
        if (need == 2 and cp < 0x800) or (need == 3 and (cp < 0x10000 or cp > 0x10FFFF)) or (cp >= 0xD800 and cp <= 0xDFFF):
            return False
        i += need + 1
    return True


def _is_count(v: JsonValue) raises -> Bool:
    if not v.is_number():
        return False
    var t = v.number_text()
    for b in t.as_bytes():
        if b < 48 or b > 57:
            return False  # sign, fraction or exponent
    return True


def decode_result(encoded: String, job_id: String) raises -> Dict[String, Int]:
    """Base64 result envelope -> counts. Each level passes the refusal rule before parsing."""
    var raw: List[UInt8]
    try:
        raw = b64decode_strict(encoded)
    except e:
        raise Error(E_INVALID_PAYLOAD + ": SpeQtrum result of job " + job_id + " is not base64 text")
    if not _valid_utf8(raw):
        raise Error(E_INVALID_PAYLOAD + ": SpeQtrum result of job " + job_id + " is not base64 text")
    var text = String(from_utf8_lossy=Span(raw))
    var envelope = load_document(text, "result", job_id)
    var fr = envelope.get("functional_result")
    if not envelope.is_object() or not fr.is_string():
        raise Error(E_INVALID_PAYLOAD + ": SpeQtrum result of job " + job_id + " has no functional_result text")
    var inner = load_document(fr.as_string(), "functional result", job_id)
    var counts = inner.get("counts")
    if not inner.is_object() or not counts.is_object() or len(counts) == 0:
        raise Error(E_INVALID_PAYLOAD + ": functional result of job " + job_id + " has no counts object")
    var out = Dict[String, Int]()
    for i in range(len(counts)):
        var key = counts.key_at(i)
        var ok = key.byte_length() > 0
        for b in key.as_bytes():
            if b != 48 and b != 49:
                ok = False
        var v = counts.at(i)
        if not ok or not _is_count(v):
            raise Error(E_INVALID_PAYLOAD + ": counts of job " + job_id
                        + " contain an entry that is not a bitstring with a count")
        out[key] = v.as_int()
    return out^


# ---------------------------------------------------------------------------
# Payload documents (byte-identical to the Python fallback)
# ---------------------------------------------------------------------------
def _int_array(vals: List[Int]) -> String:
    var out = String("[")
    for i in range(len(vals)):
        if i > 0:
            out += ", "
        out += String(vals[i])
    return out + "]"


def payload_document(backend: String, shots: Int, qasm: String, num_qubits: Int, num_clbits: Int,
                     initial_layout: List[Int], final_layout: List[Int], measure_clbits: List[Int],
                     measure_qubits: List[Int], auto_measured: Bool) -> String:
    """qumo.payload.v1, as base.qumo_payload_yaml writes it (json.dumps, sort_keys=True)."""
    # measure_map keys are clbit numbers as strings, sorted as strings
    var keys = List[String]()
    var vals = List[Int]()
    for i in range(len(measure_clbits)):
        var k = String(measure_clbits[i])
        var pos = len(keys)
        keys.append(k)
        vals.append(measure_qubits[i])
        while pos > 0 and keys[pos] < keys[pos - 1]:
            var tk = keys[pos]
            keys[pos] = keys[pos - 1]
            keys[pos - 1] = tk
            var tv = vals[pos]
            vals[pos] = vals[pos - 1]
            vals[pos - 1] = tv
            pos -= 1
    var mm = String("{")
    for i in range(len(keys)):
        if i > 0:
            mm += ", "
        mm += py_json_string(keys[i]) + ": " + String(vals[i])
    mm += "}"
    return (
        "{\"auto_measured\": " + (String("true") if auto_measured else String("false"))
        + ", \"backend\": " + py_json_string(backend)
        + ", \"final_layout\": " + _int_array(final_layout)
        + ", \"format\": " + py_json_string(PAYLOAD_FORMAT_V1)
        + ", \"initial_layout\": " + _int_array(initial_layout)
        + ", \"measure_map\": " + mm
        + ", \"num_clbits\": " + String(num_clbits)
        + ", \"num_qubits\": " + String(num_qubits)
        + ", \"qasm\": " + py_json_string(qasm)
        + ", \"shots\": " + String(shots) + "}"
    )


def execute_payload(document: String, shots: Int) raises -> String:
    """The `payload` string of POST /execute (json.dumps(..., sort_keys=True) of the envelope)."""
    var construct = unsafe_construct(document)
    if construct.byte_length() > 0:
        raise Error(E_INVALID_PAYLOAD + ": payload refused before sending: it contains " + construct)
    var readout = "{\"sampling\": {\"nshots\": " + String(shots) + "}}"
    return ("{\"digital_propagation_payload\": {\"digital_propagation\": " + py_json_string(document)
            + ", \"readout\": " + py_json_string(readout) + "}, \"type\": " + py_json_string(PAYLOAD_TYPE) + "}")


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
struct SpeQtrumAuth(Copyable):
    var base: String
    var username: String
    var api_key: String
    var audience: String
    var token: String  # the access token of the current call; never returned

    def __init__(out self, var base: String, var username: String, var api_key: String, var audience: String):
        self.base = base^
        self.username = username^
        self.api_key = api_key^
        self.audience = audience^
        self.token = String()

    def secrets(self) -> List[String]:
        var out = List[String]()
        out.append(self.api_key.copy())
        out.append(self.username.copy())
        if self.token.byte_length() > 0:
            out.append(self.token.copy())
        return out^


def _loopback(host: String) -> Bool:
    return host == "127.0.0.1" or host == "localhost" or host == "::1"


def checked_base(url: String) raises -> String:
    """https, or http only for a loopback host. No user information. Trailing '/' removed."""
    var text = String(url.strip())
    var at = text.find("@")
    var authority_end = text.find("/", text.find("//") + 2) if text.find("//") >= 0 else -1
    if at >= 0 and (authority_end < 0 or at < authority_end):
        raise Error(E_CREDENTIALS + ": " + ENV_URL + " must not contain user information. Hint: put the username and "
                    + "API key in " + ENV_USERNAME + " and " + ENV_APIKEY)
    var u: Url
    try:
        u = parse_url(text)
    except e:
        raise Error(E_CREDENTIALS + ": " + ENV_URL + " is not a valid URL")
    if not (u.scheme == "https" or (u.scheme == "http" and _loopback(u.host))):
        raise Error(E_CREDENTIALS + ": " + ENV_URL + " must be an https URL (plain http is accepted only for "
                    + "localhost, 127.0.0.1 or [::1], for tests)")
    while text.endswith("/"):
        var t = String(text[byte=0:text.byte_length() - 1])
        text = t^
    return text^


def _send(auth: SpeQtrumAuth, method: String, url: String, body: String, token: String) raises -> HttpResponse:
    var names = List[String]()
    var values = List[String]()
    names.append("Accept")
    values.append("application/json")
    names.append("User-Agent")
    values.append("qumo-hardware/1.0 (native Mojo client)")
    if body.byte_length() > 0:
        names.append("Content-Type")
        values.append("application/json")
    if token.byte_length() > 0:
        names.append("Authorization")
        values.append("Bearer " + token)
    var u = parse_url(url)
    var opts = HttpOptions(TIMEOUT_MS, TIMEOUT_MS, MAX_RESPONSE_BYTES, False, REDIRECT_NONE, 0, _loopback(u.host), False)
    var client = HttpClient(None)
    try:
        return client.request(HttpRequest(method, url, names^, values^, body), opts)
    except e:
        var hint = String(" Hint: check ") + ENV_URL + ", the network and proxy settings, then retry"
        if method == "POST":
            hint = " Hint: the request may have reached SpeQtrum; check the SpeQtrum web UI for a new job before retrying"
        raise Error(E_JOB + ": cannot reach SpeQtrum (" + _clean(String(e), auth.secrets()) + ")." + hint)


def _object(r: HttpResponse, what: String) raises -> JsonValue:
    var v: JsonValue
    try:
        v = parse_json(r.text())
    except e:
        raise Error(E_JOB + ": SpeQtrum " + what + " returned no JSON. Hint: retry later; report it if it persists")
    if not v.is_object():
        raise Error(E_JOB + ": SpeQtrum " + what + " returned JSON that is not an object. Hint: retry later; "
                    + "report it if it persists")
    return v^


def _failure(r: HttpResponse, what: String, job_lookup: Bool, secrets: List[String]) -> Error:
    if r.status >= 300 and r.status < 400:
        return Error(E_JOB + ": SpeQtrum " + what + " answered with a redirect (HTTP " + String(r.status)
                     + "); QUMO does not follow redirects because they could forward the access token. Hint: check "
                     + ENV_URL + " for a moved endpoint")
    if r.status == 404 and job_lookup:
        return Error(E_UNKNOWN_JOB + ": SpeQtrum has no job with this id (HTTP 404)")
    var message = String()
    try:
        var data = parse_json(r.text())
        if data.is_object():
            for key in ["message", "error", "detail"]:
                if data.has(key):
                    var v = data.get(key)
                    if v.is_string():
                        message = _clean(v.as_string(), secrets)
                    break
    except e:
        pass
    return Error(E_JOB + ": SpeQtrum " + what + " failed (HTTP " + String(r.status)
                 + (": " + message if message.byte_length() > 0 else String()) + "). Hint: retry later; check the "
                 + "device status in the SpeQtrum web UI")


def login_assertion(username: String, api_key: String, audience: String, iat: Int) -> String:
    """base64url(JSON claims, sorted keys). Not a signed JWT: no header and no signature."""
    var claims = ("{\"api_key\": " + py_json_string(api_key) + ", \"audience\": " + py_json_string(audience)
                  + ", \"iat\": " + String(iat) + ", \"username\": " + py_json_string(username) + "}")
    return b64url_nopad(claims)


def _login(auth: SpeQtrumAuth) raises -> String:
    var body = ("{\"grantType\": " + py_json_string(GRANT_TYPE) + ", \"assertion\": "
                + py_json_string(login_assertion(auth.username, auth.api_key, auth.audience, c_time()))
                + ", \"scope\": \"user profile\"}")
    var r = _send(auth, "POST", auth.base + "/authorisation-tokens", body, String())
    if r.status == 401 or r.status == 403:
        raise Error(E_CREDENTIALS + ": SpeQtrum rejected the login (HTTP " + String(r.status) + "). Hint: check "
                    + ENV_USERNAME + " and " + ENV_APIKEY + ", and that the API key has not expired")
    if r.status < 200 or r.status >= 300:
        raise _failure(r, "login", False, auth.secrets())
    var data = _object(r, "login")
    var tok = data.get("accessToken")
    if not tok.is_string() or tok.as_string().byte_length() == 0:
        raise Error(E_CREDENTIALS + ": SpeQtrum login response has no access token. Hint: retry; if it persists, "
                    + "check the account in the SpeQtrum web UI")
    return tok.as_string()


def _call(mut auth: SpeQtrumAuth, method: String, path_query: String, body: String, what: String,
          job_lookup: Bool = False) raises -> JsonValue:
    var url = auth.base + path_query
    var token = _login(auth)
    auth.token = token
    var r = _send(auth, method, url, body, token)
    if r.status == 401:  # rejected before acting on the request: log in again, retry once
        token = _login(auth)
        auth.token = token
        r = _send(auth, method, url, body, token)
        if r.status == 401:
            raise Error(E_CREDENTIALS + ": SpeQtrum rejected the access token after re-authentication. Hint: check "
                        + ENV_USERNAME + " and " + ENV_APIKEY)
    var secrets = auth.secrets()
    secrets.append(token)
    if r.status < 200 or r.status >= 300:
        raise _failure(r, what, job_lookup, secrets)
    return _object(r, what)


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------
struct SpeQtrumDevice(Copyable):
    var code: String
    var nqubits: Int  # -1 when not published
    var status: String
    var notes: String

    def __init__(out self, var code: String, nqubits: Int, var status: String, var notes: String):
        self.code = code^
        self.nqubits = nqubits
        self.status = status^
        self.notes = notes^


def speqtrum_devices(mut auth: SpeQtrumAuth) raises -> List[SpeQtrumDevice]:
    var data = _call(auth, "GET", "/devices", String(), "device list")
    var items = data.get("items")
    if not items.is_array():
        raise Error(E_JOB + ": SpeQtrum device list has no 'items' array. Hint: retry later")
    var out = List[SpeQtrumDevice]()
    var secrets = auth.secrets()
    for i in range(len(items)):
        var item = items.at(i)
        if not item.is_object():
            continue
        var code = item.get("code")
        if not code.is_string() or code.as_string().byte_length() == 0:
            continue
        var nq = item.get("nqubits")
        var n = -1
        if nq.is_number() and _is_count(nq):
            n = nq.as_int()
        var st = item.get("status")
        var status = st.as_string() if st.is_string() else String()
        var notes = String()
        for key in ["name", "description"]:
            var v = item.get(key)
            if v.is_string() and v.as_string().byte_length() > 0:
                var c = _clean(v.as_string(), secrets)
                notes = (notes + " - " + c) if notes.byte_length() > 0 else c
        if notes.byte_length() == 0:
            notes = "from SpeQtrum"
        out.append(SpeQtrumDevice(code.as_string(), n, status^, notes^))
    return out^


def speqtrum_execute(mut auth: SpeQtrumAuth, device_code: String, document: String, shots: Int) raises -> String:
    """POST /execute; the new job id (decimal digits)."""
    var payload = execute_payload(document, shots)
    var body = ("{\"device_code\": " + py_json_string(device_code) + ", \"payload\": " + py_json_string(payload)
                + ", \"job_type\": \"digital\", \"meta\": {}}")
    var data = _call(auth, "POST", "/execute", body, "execute")
    var jid = data.get("id")
    var text = String()
    if jid.is_number() and _is_count(jid):
        text = String(jid.as_int())
    elif jid.is_string():
        text = jid.as_string()
    var ok = text.byte_length() > 0 and text.byte_length() <= 18
    for b in text.as_bytes():
        if b < 48 or b > 57:
            ok = False
    if not ok:
        raise Error(E_JOB + ": SpeQtrum accepted the request but returned no job id. Hint: check the SpeQtrum web UI "
                    + "for a new job before resubmitting")
    return String(Int(text))


def _job(mut auth: SpeQtrumAuth, job_id: String) raises -> JsonValue:
    var ok = job_id.byte_length() > 0 and job_id.byte_length() <= 18
    for b in job_id.as_bytes():
        if b < 48 or b > 57:
            ok = False
    if not ok:
        raise Error(E_UNKNOWN_JOB + ": '" + job_id + "' is not a SpeQtrum job id")
    return _call(auth, "GET", "/jobs/" + job_id + "?payload=false&result=true&error=true", String(), "job status",
                 job_lookup=True)


def map_status(raw: String) -> String:
    var st = String(raw.strip()).lower()
    if st == "pending" or st == "validating" or st == "queued":
        return "QUEUED"
    if st == "running":
        return "RUNNING"
    if st == "completed":
        return "DONE"
    if st == "error" or st == "timeout":
        return "ERROR"
    if st == "cancelled":
        return "CANCELLED"
    return "UNKNOWN"


def _status_of(job: JsonValue) raises -> String:
    var st = job.get("status")
    if st.is_string():
        return map_status(st.as_string())
    if st.is_number():
        return map_status(st.number_text())
    return "UNKNOWN"


def speqtrum_status(mut auth: SpeQtrumAuth, job_id: String) raises -> String:
    return _status_of(_job(auth, job_id))


def speqtrum_result(mut auth: SpeQtrumAuth, job_id: String) raises -> Dict[String, Int]:
    var data = _job(auth, job_id)
    var state = _status_of(data)
    if state != "DONE":
        var detail = String()
        if state == "ERROR" or state == "CANCELLED":
            var err = data.get("error")
            if err.is_string():
                detail = _clean(err.as_string(), auth.secrets())
        raise Error(E_JOB + ": SpeQtrum job " + job_id + " is " + state
                    + (": " + detail if detail.byte_length() > 0 else String()))
    var enc = data.get("result")
    if not enc.is_string() or enc.as_string().byte_length() == 0:
        raise Error(E_JOB + ": SpeQtrum job " + job_id + " is DONE but has no result yet. Hint: retry shortly with "
                    + "`python -m qumo_hardware result " + job_id + " --wait`")
    return decode_result(enc.as_string(), job_id)
