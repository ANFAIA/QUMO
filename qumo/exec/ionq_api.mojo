"""IonQ cloud client in native Mojo: request bodies, HTTPS calls, result sampling.

No vendor SDK and no Python on this path. The pieces are all QUMO's own:
OpenQASM parsing (qumo.ir.qasm), JSON (qumo.ai.json), HTTP/1.1 + TLS over libc sockets
with OpenSSL 3 loaded at run time (qumo.net), and the seeded RNG (qumo.sim.rng).
plugins/hardware/qumo_hardware/ionq.py only adapts these functions to the plugin API.

Wire format: IonQ API v0.4, `ionq.circuit.v1`, gateset `qis`. The circuit list is the one
qiskit-ionq 1.1.1 sends for the same circuit (alias table: sdg -> si, tdg -> ti, sx -> v,
sxdg -> vi, rxx/ryy/rzz -> xx/yy/zz; controlled gates send the base gate with `controls`;
rotations in radians under `rotation`). Measurements are not gates: they become the
clbit -> qubit map. A gate on a measured qubit (mid-circuit measurement) is refused,
because the v1 circuit format cannot express it.

Credentials: IONQ_API_KEY is read here, from the environment, at call time. It goes only
into the `Authorization: apiKey ...` header of requests to the IonQ API URL; it is never
returned, printed or put in an error message. Redirects are not followed, so the header
never reaches another host. IONQ_API_URL (default https://api.ionq.co/v0.4) must be https;
plain http is accepted only for a loopback host (offline tests).

Results: IonQ returns a probability histogram (outcome integer -> probability, bit q =
qubit q). Shots are drawn from it with QUMO's seeded RNG; the same seed gives the same
counts here (not the same counts as qiskit-ionq's numpy draws).
"""

from std.collections import Dict
from std.math import isinf, isnan
from std.os import getenv
from std.time import sleep

from qumo.ai.json import JsonValue, json_quote, parse_json
from qumo.ir.circuit import Instruction
from qumo.ir.gates import Op, op_arity
from qumo.ir.qasm import parse_qasm
from qumo.net.http import HttpClient, HttpOptions, HttpRequest, HttpResponse, REDIRECT_NONE
from qumo.net.url import Url, parse_url
from qumo.sim.rng import Rng

comptime DEFAULT_API_URL = "https://api.ionq.co/v0.4"
comptime API_KEY_VAR = "IONQ_API_KEY"
comptime API_URL_VAR = "IONQ_API_URL"
comptime USER_AGENT = "qumo-native-ionq/1"
comptime E_CREDENTIALS = "QM-E952"
comptime E_CONFIG = "QM-E953"
comptime E_INVALID_PAYLOAD = "QM-E955"
comptime E_LIMIT = "QM-E956"
comptime E_JOB = "QM-E959"
comptime MAX_RESPONSE_BYTES = 64 * 1024 * 1024  # a 2^22-entry histogram is about 60 MB of JSON


# ---------------------------------------------------------------------------
# Request body
# ---------------------------------------------------------------------------
struct IonqRequest(Copyable):
    var body: String  # JSON text for POST /jobs
    var clbit_qubit: List[Int]  # per clbit: measured qubit, -1 if the clbit is never written
    var num_clbits: Int
    var dense_qubits: Int  # IonQ "qubits": highest referenced qubit + 1

    def __init__(out self, var body: String, var clbit_qubit: List[Int], num_clbits: Int, dense_qubits: Int):
        self.body = body^
        self.clbit_qubit = clbit_qubit^
        self.num_clbits = num_clbits
        self.dense_qubits = dense_qubits


def _angle(g: Instruction) raises -> String:
    if g.p0.is_symbolic():
        raise Error(E_INVALID_PAYLOAD + ": '" + String(g) + "' has an unbound parameter; bind it before submitting")
    var v = g.p0.value()
    if isnan(v) or isinf(v):
        raise Error(E_INVALID_PAYLOAD + ": '" + String(g) + "' has a rotation angle that is not finite")
    return String(v)


def _one(name: String, q: Int) -> String:
    return "{\"gate\":\"" + name + "\",\"targets\":[" + String(q) + "]}"


def _gate_object(g: Instruction) raises -> String:
    """One IonQ circuit entry for a unitary instruction (identity and barrier are skipped by the caller)."""
    var k = g.kind
    if k == Op.H:
        return _one("h", g.q0)
    if k == Op.X:
        return _one("x", g.q0)
    if k == Op.Y:
        return _one("y", g.q0)
    if k == Op.Z:
        return _one("z", g.q0)
    if k == Op.S:
        return _one("s", g.q0)
    if k == Op.SDG:
        return _one("si", g.q0)
    if k == Op.T:
        return _one("t", g.q0)
    if k == Op.TDG:
        return _one("ti", g.q0)
    if k == Op.SX:
        return _one("v", g.q0)
    if k == Op.SXDG:
        return _one("vi", g.q0)
    if k == Op.RX or k == Op.RY or k == Op.RZ:
        var name = String("rx") if k == Op.RX else (String("ry") if k == Op.RY else String("rz"))
        return "{\"gate\":\"" + name + "\",\"targets\":[" + String(g.q0) + "],\"rotation\":" + _angle(g) + "}"
    if k == Op.CX or k == Op.CY or k == Op.CZ:
        var name = String("x") if k == Op.CX else (String("y") if k == Op.CY else String("z"))
        return ("{\"gate\":\"" + name + "\",\"controls\":[" + String(g.q0) + "],\"targets\":["
                + String(g.q1) + "]}")
    if k == Op.SWAP:
        return "{\"gate\":\"swap\",\"targets\":[" + String(g.q0) + "," + String(g.q1) + "]}"
    if k == Op.RXX or k == Op.RYY or k == Op.RZZ:
        var name = String("xx") if k == Op.RXX else (String("yy") if k == Op.RYY else String("zz"))
        return ("{\"gate\":\"" + name + "\",\"targets\":[" + String(g.q0) + "," + String(g.q1)
                + "],\"rotation\":" + _angle(g) + "}")
    raise Error(E_INVALID_PAYLOAD + ": gate '" + String(g) + "' is not in the IonQ QIS gateset; decompose it into "
                + "h, x, y, z, s, sdg, t, tdg, sx, rx, ry, rz, cx, cy, cz, swap, rxx, ryy, rzz")


def ionq_request(source: String, shots: Int, noise_model: String, sampler_seed: Int,
                 name: String) raises -> IonqRequest:
    """OpenQASM 2 text -> the `ionq.circuit.v1` request body and the measurement map.

    `noise_model` is an IonQ simulator noise model (e.g. "aria-1"); empty means none.
    `sampler_seed` is recorded in the job metadata; counts are drawn client-side (ionq_counts)."""
    if shots < 1:
        raise Error(E_LIMIT + ": shots must be at least 1")
    var c = parse_qasm(source, name)
    if c.num_clbits == 0 or c.num_measurements == 0:
        raise Error(E_INVALID_PAYLOAD + ": the circuit has no measurements; IonQ returns no counts without them")
    var clbit_qubit = List[Int](length=c.num_clbits, fill=-1)
    var measured = List[Bool](length=c.num_qubits, fill=False)
    var dense = 0
    var gates = String()
    var count = 0
    for g in c.ops:
        var k = g.kind
        if k == Op.BARRIER:
            continue
        if k == Op.I:  # no-op on the wire, but qiskit-ionq counts its qubit in the width
            dense = max(dense, g.q0 + 1)
            continue
        if g.cond >= 0:
            raise Error(E_INVALID_PAYLOAD + ": '" + String(g) + "' is classically conditioned; the IonQ circuit "
                        + "format takes unitary gates and final measurements only")
        if k == Op.MEASURE:
            clbit_qubit[g.clbit] = g.q0
            measured[g.q0] = True
            dense = max(dense, g.q0 + 1)
            continue
        for j in range(op_arity(k)):
            var q = g.qubit(j)
            if measured[q]:
                raise Error(E_INVALID_PAYLOAD + ": '" + String(g) + "' acts on a measured qubit; mid-circuit "
                            + "measurement is not supported by the IonQ v1 circuit format")
            dense = max(dense, q + 1)
        if count > 0:
            gates += ","
        gates += _gate_object(g)  # raises for reset, noise channels and gates outside the QIS gateset
        count += 1
    var body = (
        "{\"type\":\"ionq.circuit.v1\",\"backend\":\"simulator\",\"shots\":" + String(shots)
        + ",\"name\":" + json_quote(name)
        + ",\"input\":{\"gateset\":\"qis\",\"qubits\":" + String(dense) + ",\"circuit\":[" + gates + "]}"
        + ",\"metadata\":{\"shots\":\"" + String(shots) + "\",\"sampler_seed\":\"" + String(sampler_seed)
        + "\",\"client\":\"" + USER_AGENT + "\"}"
    )
    if noise_model != "":
        body += ",\"noise\":{\"model\":" + json_quote(noise_model) + ",\"seed\":" + String(sampler_seed) + "}"
    body += "}"
    return IonqRequest(body^, clbit_qubit^, c.num_clbits, dense)


# ---------------------------------------------------------------------------
# Counts
# ---------------------------------------------------------------------------
def ionq_counts(keys: List[Int], probs: List[Float64], clbit_qubit: List[Int], shots: Int,
                seed: UInt64) raises -> Dict[String, Int]:
    """IonQ histogram -> counts over the clbits (leftmost character = highest clbit).

    Clbits that are never written read "0". Shots are drawn from the normalised histogram."""
    var n = len(clbit_qubit)
    if n == 0:
        raise Error(E_INVALID_PAYLOAD + ": no measured classical bits to build counts from")
    if shots < 1:
        raise Error(E_LIMIT + ": shots must be at least 1")
    if len(keys) != len(probs) or len(keys) == 0:
        raise Error(E_JOB + ": IonQ returned an empty or malformed probability histogram")
    var total = 0.0
    for i in range(len(probs)):
        var p = probs[i]
        if isnan(p) or isinf(p) or p < 0.0:
            raise Error(E_JOB + ": IonQ returned an invalid probability")
        if keys[i] < 0:
            raise Error(E_JOB + ": IonQ returned a negative outcome index")
        total += p
    if total <= 0.0:
        raise Error(E_JOB + ": IonQ returned a histogram with zero total probability")
    var cdf = List[Float64](capacity=len(probs))
    var acc = 0.0
    for p in probs:
        acc += p / total
        cdf.append(acc)
    var hits = List[Int](length=len(probs), fill=0)
    var rng = Rng(seed)
    for _ in range(shots):
        var u = rng.uniform()
        var lo = 0  # first index whose cumulative probability exceeds u
        var hi = len(cdf) - 1
        while lo < hi:
            var mid = (lo + hi) // 2
            if cdf[mid] > u:
                hi = mid
            else:
                lo = mid + 1
        hits[lo] += 1
    var out = Dict[String, Int]()
    for i in range(len(keys)):
        if hits[i] == 0:
            continue
        var bits = String()
        for k in range(n):
            var q = clbit_qubit[n - 1 - k]
            var bit = 0
            if q >= 0 and q < 63:
                bit = (keys[i] >> q) & 1
            bits += "1" if bit == 1 else "0"
        out[bits] = out.get(bits, 0) + hits[i]
    return out^


# ---------------------------------------------------------------------------
# HTTPS client
# ---------------------------------------------------------------------------
def ionq_api_url() raises -> String:
    """IONQ_API_URL or the default, without a trailing '/'. https only, except loopback http for tests."""
    var base = String(getenv(API_URL_VAR, DEFAULT_API_URL).strip())
    if base.byte_length() == 0:
        base = DEFAULT_API_URL
    while base.endswith("/"):
        var trimmed = String(base[byte=0:base.byte_length() - 1])
        base = trimmed^
    var u: Url
    try:
        u = parse_url(base + "/")
    except e:
        raise Error(E_CONFIG + ": " + API_URL_VAR + " is not a valid http(s) URL: " + _printable(String(e), 200))
    if u.scheme != "https" and not _loopback(u.host):
        raise Error(E_CONFIG + ": " + API_URL_VAR + " must be an https URL (plain http only for a loopback host)")
    return base^


def _loopback(host: String) -> Bool:
    return host == "127.0.0.1" or host == "localhost" or host == "::1"


def _api_key() raises -> String:
    var key = String(getenv(API_KEY_VAR, "").strip())
    if key.byte_length() == 0:
        raise Error(E_CREDENTIALS + ": IonQ API key not found: the " + API_KEY_VAR
                    + " environment variable is not set. Hint: export " + API_KEY_VAR
                    + "=<your IonQ API key> (create one at https://cloud.ionq.com/settings/keys)")
    for b in key.as_bytes():
        if b < 33 or b > 126:
            raise Error(E_CREDENTIALS + ": " + API_KEY_VAR + " contains a character that cannot go in an HTTP header")
    return key^


def _check_job_id(job_id: String) raises:
    """IonQ job ids are UUIDs; anything else could change the request path."""
    if job_id.byte_length() == 0 or job_id.byte_length() > 64:
        raise Error(E_JOB + ": invalid IonQ job id")
    for b in job_id.as_bytes():
        var ok = (b >= 48 and b <= 57) or (b >= 65 and b <= 90) or (b >= 97 and b <= 122) or b == 45
        if not ok:
            raise Error(E_JOB + ": invalid IonQ job id '" + job_id + "'")


def _printable(s: String, limit: Int) -> String:
    """At most `limit` bytes of printable ASCII; anything else becomes '?'. Safe to slice and to show."""
    var out = List[UInt8]()
    for b in s.as_bytes():
        if len(out) >= limit:
            break
        out.append(b if (b >= 32 and b < 127) else UInt8(63))
    return String(from_utf8_lossy=Span(out))


def _error_detail(text: String) -> String:
    """The useful part of an IonQ error body. Shapes seen in the API (qiskit-ionq exceptions.py):
    {"statusCode", "error": "Bad Request", "message": ...}, {"error": {"type", "message"}},
    {"code", "message"}."""
    var detail = String()
    try:
        var err = parse_json(text)
        if err.is_object():
            var msg = err.get("message")
            var inner = err.get("error")
            if msg.is_string():
                detail = msg.as_string()
            elif inner.is_object() and inner.get("message").is_string():
                detail = inner.get("message").as_string()
            elif inner.is_string():
                detail = inner.as_string()
    except e:
        pass
    return _printable(detail, 200)


def _retryable(status: Int) -> Bool:
    return status == 429 or status == 502 or status == 503 or status == 504


def _call(method: String, path: String, body: String, what: String, authenticated: Bool = True) raises -> JsonValue:
    """One request to the IonQ API; the JSON response body, or a coded error.

    GETs are retried (3 attempts, 1 s then 2 s apart) on transport errors and on 429/502/503/504.
    POSTs are never retried: after a transport error the job may already exist on IonQ."""
    var base = ionq_api_url()
    var url = base + "/" + path
    var u = parse_url(url)
    var opts = HttpOptions(15000, 60000, MAX_RESPONSE_BYTES, False, REDIRECT_NONE, 0, _loopback(u.host), False)
    var attempts = 3 if method == "GET" else 1
    var attempt = 0
    while True:
        attempt += 1
        var names = List[String]()
        var values = List[String]()
        names.append("Accept")
        values.append("application/json")
        names.append("User-Agent")
        values.append(USER_AGENT)
        if authenticated:
            names.append("Authorization")
            values.append("apiKey " + _api_key())
        if body.byte_length() > 0:
            names.append("Content-Type")
            values.append("application/json")
        var client = HttpClient(None)
        var r: HttpResponse
        try:
            r = client.request(HttpRequest(method, url, names^, values^, body), opts)
        except e:
            if attempt < attempts:
                sleep(Float64(attempt))
                continue
            var hint = String(" Hint: check ") + API_URL_VAR + ", the network and proxy settings, then retry"
            if method == "POST":
                hint = " Hint: the request may have reached IonQ and created a job; check the IonQ console before retrying"
            raise Error(E_JOB + ": IonQ " + what + " failed: " + _printable(String(e), 300) + "." + hint)
        if _retryable(r.status) and attempt < attempts:
            sleep(Float64(attempt))
            continue
        if r.status == 401 or r.status == 403:
            raise Error(E_CREDENTIALS + ": IonQ " + what + " failed (HTTP " + String(r.status) + "). Hint: check "
                        + API_KEY_VAR + " and the IonQ account")
        if r.status < 200 or r.status >= 300:
            var detail = _error_detail(r.text())
            var where = String(" (redirects are not followed)") if r.status >= 300 and r.status < 400 else String()
            raise Error(E_JOB + ": IonQ " + what + " failed (HTTP " + String(r.status)
                        + (": " + detail if detail.byte_length() > 0 else String("")) + ")" + where)
        if len(r.body) == 0:
            return parse_json("{}")
        try:
            return parse_json(r.text())
        except e:
            raise Error(E_JOB + ": IonQ " + what + " returned a body that is not JSON")


struct IonqSubmitted(Copyable):
    var job_id: String
    var request: IonqRequest

    def __init__(out self, var job_id: String, var request: IonqRequest):
        self.job_id = job_id^
        self.request = request^


def ionq_submit(source: String, shots: Int, noise_model: String, sampler_seed: Int,
                name: String) raises -> IonqSubmitted:
    """POST /jobs with the circuit; the new job id and the measurement map."""
    var req = ionq_request(source, shots, noise_model, sampler_seed, name)
    var resp = _call("POST", "jobs", req.body, "job submission")
    var job_id = resp.get_string("id", "") if resp.is_object() else String()
    if job_id.byte_length() == 0:
        raise Error(E_JOB + ": IonQ accepted the request but returned no job id. Hint: check the IonQ console "
                    + "for a new job before retrying")
    _check_job_id(job_id)
    return IonqSubmitted(job_id^, req^)


def _state(api_status: String) -> String:
    var st = api_status.lower()
    if st == "submitted" or st == "ready":
        return "QUEUED"
    if st == "started" or st == "running":
        return "RUNNING"
    if st == "completed":
        return "DONE"
    if st == "failed":
        return "ERROR"
    if st == "canceled" or st == "cancelled":
        return "CANCELLED"
    return "UNKNOWN"


def _job(job_id: String) raises -> JsonValue:
    _check_job_id(job_id)
    var resp = _call("GET", "jobs/" + job_id, String(), "status of job '" + job_id + "'")
    if not resp.is_object():
        raise Error(E_JOB + ": IonQ status of job '" + job_id + "' is not a JSON object")
    return resp^


def _failure(job: JsonValue) -> String:
    """`failure.code` / `failure.error` of a failed job, printable, or ""."""
    var f = job.get("failure")
    var out = String()
    try:
        if f.is_object():
            var code = f.get("code")
            var err = f.get("error")
            if code.is_string():
                out = code.as_string()
            if err.is_string():
                out = (out + ": " if out.byte_length() > 0 else String()) + err.as_string()
        elif f.is_string():
            out = f.as_string()
    except e:
        pass
    return _printable(out, 300)


def ionq_status(job_id: String) raises -> String:
    """GET /jobs/<id> -> QUMO state: QUEUED, RUNNING, DONE, ERROR, CANCELLED or UNKNOWN."""
    var job = _job(job_id)
    return _state(job.get_string("status", ""))


def ionq_result(job_id: String, clbit_qubit: List[Int], shots: Int, seed: UInt64) raises -> Dict[String, Int]:
    """GET /jobs/<id>/results/probabilities -> counts (the job must be DONE)."""
    var job = _job(job_id)
    var st = _state(job.get_string("status", ""))
    if st != "DONE":
        var why = _failure(job)
        raise Error(E_JOB + ": IonQ job '" + job_id + "' is " + st + (" (" + why + ")" if why.byte_length() > 0 else String()))
    var hist = _call("GET", "jobs/" + job_id + "/results/probabilities", String(),
                     "result of job '" + job_id + "'")
    if not hist.is_object() or len(hist) == 0:
        raise Error(E_JOB + ": IonQ job '" + job_id + "' returned an empty probability histogram")
    var keys = List[Int]()
    var probs = List[Float64]()
    for i in range(len(hist)):
        var key = hist.key_at(i)
        var v = 0
        if key.byte_length() == 0 or key.byte_length() > 18:
            raise Error(E_JOB + ": IonQ job '" + job_id + "' returned an outcome key that is not an integer")
        for b in key.as_bytes():
            if b < 48 or b > 57:
                raise Error(E_JOB + ": IonQ job '" + job_id + "' returned an outcome key that is not an integer")
            v = v * 10 + Int(b) - 48
        keys.append(v)
        var p = hist.at(i)
        if not p.is_number():
            raise Error(E_JOB + ": IonQ job '" + job_id + "' returned a probability that is not a number")
        probs.append(p.as_float())
    return ionq_counts(keys, probs, clbit_qubit, shots, seed)


struct IonqBackendRow(Copyable):
    var name: String
    var status: String
    var qubits: Int

    def __init__(out self, var name: String, var status: String, qubits: Int):
        self.name = name^
        self.status = status^
        self.qubits = qubits


def ionq_backends() raises -> List[IonqBackendRow]:
    """GET /backends (read-only; the listing is public, the key is not sent)."""
    var rows = _call("GET", "backends", String(), "backend listing", authenticated=False)
    var out = List[IonqBackendRow]()
    if not rows.is_array():
        raise Error(E_JOB + ": IonQ backend listing is not a JSON array")
    for i in range(len(rows)):
        var row = rows.at(i)
        if not row.is_object():
            continue
        out.append(IonqBackendRow(row.get_string("backend", ""), row.get_string("status", ""),
                                  Int(row.get_float("qubits", 0.0)) if row.get("qubits").is_number() else 0))
    return out^
