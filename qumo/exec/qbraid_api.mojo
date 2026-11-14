"""qBraid client in native Mojo: devices, cost estimates, jobs, results (qBraid Runtime API v2).

qBraid is an aggregator: one API in front of many vendors' simulators and QPUs (AWS, Azure, IBM,
IonQ, qBraid's own simulators; providers AQT, IQM, OQC, Pasqal, QuEra, Rigetti, ...). Devices are
named by QRN, vendor:provider:type:name (e.g. qbraid:qbraid:sim:qir-sv).

    GET  {api}/devices[?page&limit&...]           data: [RuntimeDevice]
    GET  {api}/devices/{qrn}                      data: RuntimeDevice
    GET  {api}/jobs/cost-estimate?deviceQrn&shots data: {pricingAvailable, estimatedCost | reason}
    POST {api}/jobs  {name, shots, deviceQrn, program: {format, data}}   data: {jobQrn, status}
    GET  {api}/jobs/{qrn}                         data: RuntimeJob (status, statusMsg, cost)
    GET  {api}/jobs/{qrn}/result                  data: {status, cost, resultData}
    POST {api}/jobs/{qrn}/cancel

{api} = https://api-v2.qbraid.com/api/v1 (OpenAPI: https://docs.qbraid.com/openapi-v2.json, read
2026-11-09). Envelope {"success": true, "data": ...}; errors {"success": false, "message",
"error": {"message", "code"}}. Costs are in qBraid credits: 1 credit = 0.01 USD.

The API key is read here from QBRAID_API_KEY at call time and sent only in the X-API-KEY header
to the qBraid API host; it is never returned, printed or put in an error. https only (plain
http only for a loopback QUMO_QBRAID_URL in tests), no redirects.
"""

from std.collections import Dict
from std.os import getenv
from std.time import sleep

from qumo.core.json import JsonValue, json_quote, parse_json
from qumo.net.http import HttpClient, HttpOptions, HttpRequest, HttpResponse, REDIRECT_NONE
from qumo.net.url import Url, parse_url, quote

comptime DEFAULT_API = "https://api-v2.qbraid.com/api/v1"
comptime USER_AGENT = "qumo-hardware/1.0 (native Mojo client; +https://github.com/ANFAIA/QUMO)"
comptime KEY_VAR = "QBRAID_API_KEY"
comptime E_CREDENTIALS = "QM-E952"
comptime E_UNKNOWN_BACKEND = "QM-E953"
comptime E_INVALID_PAYLOAD = "QM-E955"
comptime E_LIMIT = "QM-E956"
comptime E_UNKNOWN_JOB = "QM-E958"
comptime E_JOB = "QM-E959"
comptime MAX_RESPONSE_BYTES = 64 * 1024 * 1024


def _printable(s: String, limit: Int) -> String:
    var out = List[UInt8]()
    for b in s.as_bytes():
        if len(out) >= limit:
            break
        out.append(b if (b >= 32 and b < 127) else UInt8(63))
    return String(from_utf8_lossy=Span(out))


def _loopback(host: String) -> Bool:
    return host == "127.0.0.1" or host == "localhost" or host == "::1"


def check_qrn(qrn: String, what: String, code: String) raises:
    """QRNs are vendor:provider:type:name; job QRNs add -qjob-<hex>. Anything else could change the path."""
    if qrn.byte_length() == 0 or qrn.byte_length() > 200:
        raise Error(code + ": invalid qBraid " + what + " '" + _printable(qrn, 80) + "'")
    for b in qrn.as_bytes():
        var ok = ((b >= 48 and b <= 57) or (b >= 65 and b <= 90) or (b >= 97 and b <= 122) or b == 58 or b == 45
                  or b == 95 or b == 46)
        if not ok:
            raise Error(code + ": invalid qBraid " + what + " '" + _printable(qrn, 80) + "'")


struct QbraidSession(Copyable):
    var base: String
    var key: String

    def __init__(out self) raises:
        var base = String(getenv("QUMO_QBRAID_URL", DEFAULT_API).strip())
        while base.endswith("/"):
            var t = String(base[byte=0:base.byte_length() - 1])
            base = t^
        var u: Url
        try:
            u = parse_url(base + "/")
        except e:
            raise Error(E_CREDENTIALS + ": QUMO_QBRAID_URL is not a valid URL")
        if u.scheme != "https" and not _loopback(u.host):
            raise Error(E_CREDENTIALS + ": the qBraid API URL must be https (plain http only for a loopback host)")
        self.base = base^
        var key = String(getenv(KEY_VAR, "").strip())
        if key.byte_length() == 0:
            raise Error(E_CREDENTIALS + ": qBraid API key not found: " + KEY_VAR + " is not set. Hint: create a key at "
                        + "https://account.qbraid.com and export " + KEY_VAR + "=<key> in the shell that runs QUMO")
        for b in key.as_bytes():
            if b < 33 or b > 126:
                raise Error(E_CREDENTIALS + ": " + KEY_VAR + " contains a character that cannot go in an HTTP header")
        self.key = key^

    def call(self, method: String, path: String, body: String, what: String, not_found: String = "") raises -> JsonValue:
        var attempts = 3 if method == "GET" else 1
        var attempt = 0
        while True:
            attempt += 1
            var names = List[String]()
            var values = List[String]()
            names.append("X-API-KEY")
            values.append(self.key)
            names.append("Accept")
            values.append("application/json")
            names.append("User-Agent")
            values.append(USER_AGENT)
            if body.byte_length() > 0:
                names.append("Content-Type")
                values.append("application/json")
            var url = self.base + path
            var u = parse_url(url)
            var opts = HttpOptions(15000, 120000, MAX_RESPONSE_BYTES, False, REDIRECT_NONE, 0, _loopback(u.host), False)
            var client = HttpClient(None)
            var r: HttpResponse
            try:
                r = client.request(HttpRequest(method, url, names^, values^, body), opts)
            except e:
                if attempt < attempts:
                    sleep(Float64(attempt))
                    continue
                var hint = String(" Hint: check network access to qBraid, then retry")
                if method == "POST" and what == "job submission":
                    hint = " Hint: the request may have reached qBraid and created a job; check account.qbraid.com before retrying"
                raise Error(E_JOB + ": qBraid " + what + " failed: " + _printable(String(e), 300) + "." + hint)
            if (r.status == 429 or r.status == 502 or r.status == 503 or r.status == 504) and attempt < attempts:
                sleep(Float64(attempt))
                continue
            if r.status == 401:
                raise Error(E_CREDENTIALS + ": qBraid rejected the API key (HTTP 401). Hint: check " + KEY_VAR)
            if r.status == 404 and not_found.byte_length() > 0:
                raise Error(not_found)
            if r.status >= 300 and r.status < 400:
                raise Error(E_JOB + ": qBraid " + what + " answered with a redirect (HTTP " + String(r.status)
                            + "); redirects are not followed")
            if r.status < 200 or r.status >= 300:
                var detail = String()
                try:
                    var v = parse_json(r.text())
                    var err = v.get("error")
                    if err.is_object() and err.get("message").is_string():
                        detail = err.get_string("message", "")
                    elif v.get("message").is_string():
                        detail = v.get_string("message", "")
                except e:
                    pass
                detail = detail.replace(self.key, "<redacted>")
                var code = E_CREDENTIALS if r.status == 403 else E_JOB
                raise Error(code + ": qBraid " + what + " failed (HTTP " + String(r.status)
                            + (": " + _printable(detail, 300) if detail.byte_length() > 0 else String()) + ")")
            var v: JsonValue
            try:
                v = parse_json(r.text())
            except e:
                raise Error(E_JOB + ": qBraid " + what + " returned a body that is not JSON")
            if v.is_object() and v.has("data"):
                return v.get("data")
            return v^


# ---------------------------------------------------------------------------
# Devices
# ---------------------------------------------------------------------------
struct QbraidDevice(Copyable):
    var qrn: String
    var name: String
    var vendor: String
    var device_type: String  # SIMULATOR | QPU
    var status: String  # ONLINE | UNAVAILABLE | OFFLINE | RETIRED
    var qubits: Int  # -1 when not given
    var paradigm: String  # gate-model, analog, annealing, ...
    var run_input_types: List[String]
    var per_task: Float64  # credits, -1 when not given
    var per_shot: Float64
    var per_minute: Float64
    var pricing_model: String
    var queue_depth: Int

    def __init__(out self):
        self.qrn = String()
        self.name = String()
        self.vendor = String()
        self.device_type = String()
        self.status = String()
        self.qubits = -1
        self.paradigm = String()
        self.run_input_types = List[String]()
        self.per_task = -1.0
        self.per_shot = -1.0
        self.per_minute = -1.0
        self.pricing_model = String()
        self.queue_depth = -1


def _device_of(d: JsonValue) raises -> QbraidDevice:
    var out = QbraidDevice()
    out.qrn = d.get_string("qrn", "")
    out.name = d.get_string("name", "")
    out.vendor = d.get_string("vendor", "")
    out.device_type = d.get_string("deviceType", "")
    out.status = d.get_string("status", "")
    if d.get("numberQubits").is_number():
        out.qubits = Int(d.get("numberQubits").as_float())
    var p = d.get("paradigm")
    out.paradigm = p.as_string() if p.is_string() else String()
    var ins = d.get("runInputTypes")
    for i in range(len(ins)):
        if ins.at(i).is_string():
            out.run_input_types.append(ins.at(i).as_string())
    var pr = d.get("pricing")
    if pr.is_object():
        if pr.get("perTask").is_number():
            out.per_task = pr.get("perTask").as_float()
        if pr.get("perShot").is_number():
            out.per_shot = pr.get("perShot").as_float()
        if pr.get("perMinute").is_number():
            out.per_minute = pr.get("perMinute").as_float()
    if d.get("pricingModel").is_string():
        out.pricing_model = d.get_string("pricingModel", "")
    if d.get("queueDepth").is_number():
        out.queue_depth = Int(d.get("queueDepth").as_float())
    return out^


def qbraid_devices() raises -> List[QbraidDevice]:
    """Every device the account can see (pages of 100)."""
    var s = QbraidSession()
    var out = List[QbraidDevice]()
    var page = 1
    while page <= 20:
        var data = s.call("GET", "/devices?page=" + String(page) + "&limit=100", String(), "device listing")
        if not data.is_array():
            raise Error(E_JOB + ": qBraid device listing is not a list")
        for i in range(len(data)):
            out.append(_device_of(data.at(i)))
        if len(data) < 100:
            break
        page += 1
    return out^


def qbraid_device(qrn: String) raises -> QbraidDevice:
    check_qrn(qrn, "device QRN", E_UNKNOWN_BACKEND)
    var s = QbraidSession()
    var nf = E_UNKNOWN_BACKEND + ": qBraid has no device '" + qrn + "' for this account. Hint: list them with "
    nf += "`python -m qumo_hardware backends --provider qbraid --online`"
    return _device_of(s.call("GET", "/devices/" + qrn, String(), "device '" + qrn + "'", nf))


struct QbraidQuote(Copyable):
    var available: Bool
    var credits: Float64  # estimated cost in credits (1 credit = 0.01 USD) when available
    var reason: String  # no_pricing_configured | dynamic_pricing_unavailable when not available

    def __init__(out self, available: Bool, credits: Float64, var reason: String):
        self.available = available
        self.credits = credits
        self.reason = reason^


def qbraid_cost_estimate(qrn: String, shots: Int) raises -> QbraidQuote:
    check_qrn(qrn, "device QRN", E_UNKNOWN_BACKEND)
    var s = QbraidSession()
    var d = s.call("GET", "/jobs/cost-estimate?deviceQrn=" + quote(qrn, "") + "&shots=" + String(shots), String(),
                   "cost estimate")
    var avail = d.get("pricingAvailable")
    if avail.is_bool() and avail.as_bool() and d.get("estimatedCost").is_number():
        return QbraidQuote(True, d.get("estimatedCost").as_float(), String())
    return QbraidQuote(False, -1.0, d.get_string("reason", "unknown"))


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------
struct QbraidJob(Copyable):
    var qrn: String
    var status: String  # QUMO state
    var raw_status: String
    var message: String
    var cost: Float64  # credits, -1 when not given

    def __init__(out self, var qrn: String, var status: String, var raw_status: String, var message: String, cost: Float64):
        self.qrn = qrn^
        self.status = status^
        self.raw_status = raw_status^
        self.message = message^
        self.cost = cost


def _state(st: String) -> String:
    var u = st.upper()
    if u == "INITIALIZING" or u == "QUEUED" or u == "VALIDATING" or u == "HOLD":
        return "QUEUED"
    if u == "RUNNING":
        return "RUNNING"
    if u == "COMPLETED":
        return "DONE"
    if u == "FAILED":
        return "ERROR"
    if u == "CANCELLING" or u == "CANCELLED":
        return "CANCELLED"
    return "UNKNOWN"


def qbraid_submit(device_qrn: String, program_format: String, program: String, shots: Int, name: String) raises -> String:
    """POST /jobs with one program; the job QRN."""
    check_qrn(device_qrn, "device QRN", E_UNKNOWN_BACKEND)
    if shots < 1:
        raise Error(E_LIMIT + ": shots must be at least 1")
    var s = QbraidSession()
    var body = ("{\"name\":" + json_quote(name) + ",\"shots\":" + String(shots) + ",\"deviceQrn\":" + json_quote(device_qrn)
                + ",\"program\":{\"format\":" + json_quote(program_format) + ",\"data\":" + json_quote(program) + "}"
                + ",\"tags\":{\"client\":\"qumo\"}}")
    var d = s.call("POST", "/jobs", body, "job submission")
    var qrn = d.get_string("jobQrn", "")
    if qrn.byte_length() == 0:
        raise Error(E_JOB + ": qBraid accepted the job but returned no job QRN. Hint: check account.qbraid.com before retrying")
    check_qrn(qrn, "job QRN", E_JOB)
    return qrn^


def qbraid_job(job_qrn: String) raises -> QbraidJob:
    check_qrn(job_qrn, "job QRN", E_UNKNOWN_JOB)
    var s = QbraidSession()
    var nf = E_UNKNOWN_JOB + ": qBraid job '" + job_qrn + "' not found for this account"
    var d = s.call("GET", "/jobs/" + job_qrn, String(), "status of job '" + job_qrn + "'", nf)
    var raw = d.get_string("status", "")
    var msg = d.get_string("statusMsg", "") if d.get("statusMsg").is_string() else String()
    var cost = d.get("cost").as_float() if d.get("cost").is_number() else -1.0
    return QbraidJob(job_qrn, _state(raw), raw^, _printable(msg, 300), cost)


def qbraid_result_counts(job_qrn: String) raises -> Dict[String, Int]:
    """measurementCounts of a completed gate-model job, keys as qBraid returns them."""
    check_qrn(job_qrn, "job QRN", E_UNKNOWN_JOB)
    var s = QbraidSession()
    var nf = E_UNKNOWN_JOB + ": qBraid job '" + job_qrn + "' not found for this account"
    var d = s.call("GET", "/jobs/" + job_qrn + "/result", String(), "result of job '" + job_qrn + "'", nf)
    var st = _state(d.get_string("status", ""))
    if st != "DONE":
        raise Error(E_JOB + ": qBraid job '" + job_qrn + "' is " + st)
    var rd = d.get("resultData")
    var counts = rd.get("measurementCounts")
    if not counts.is_object():
        raise Error(E_JOB + ": qBraid result of job '" + job_qrn + "' has no measurementCounts (not a gate-model result?)")
    var out = Dict[String, Int]()
    for i in range(len(counts)):
        var key = counts.key_at(i)
        var ok = key.byte_length() > 0
        for b in key.as_bytes():
            if b != 48 and b != 49:
                ok = False
        var v = counts.at(i)
        if not ok or not v.is_number() or v.as_float() < 0:
            raise Error(E_JOB + ": qBraid result of job '" + job_qrn + "' has an entry that is not a bitstring with a count")
        out[key] = out.get(key, 0) + Int(v.as_float())
    return out^


def qbraid_cancel(job_qrn: String) raises:
    check_qrn(job_qrn, "job QRN", E_UNKNOWN_JOB)
    var s = QbraidSession()
    _ = s.call("POST", "/jobs/" + job_qrn + "/cancel", "{}", "cancel of job '" + job_qrn + "'")
