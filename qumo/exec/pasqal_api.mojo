"""Pasqal Cloud client in native Mojo: login, device specs, batches, job status, results.

Wire protocol of pasqal-cloud (pasqal-io/pasqal-cloud, read 2026-11-09), reimplemented here:

    POST https://pasqal.eu.auth0.com/oauth/token     form: password-realm grant, realm pcs-users -> access_token
    GET  {core}/api/v1/devices/specs                 data: {device_type: "<specs JSON string>"}
    POST {core}/api/v1/batches                       {project_id, sequence_builder: "<Pulser JSON string>",
                                                      device_type, open: false, jobs: [{runs}]}
    GET  {core}/api/v1/batches/{id}/status | {core}/api/v1/jobs/{id}/status       data.status
    GET  {core}/api/v1/jobs/{id}/results_link        data.results_link (a presigned https URL)
    GET  <results_link>                              {"counter": {bitstring: count}, "raw": [...], ...}

core = https://apis.pasqal.cloud/core-fast. Responses use the JSend envelope
{"status": "success", "data": ...}.

Credentials are read here from the environment at call time: PASQAL_USERNAME / PASQAL_PASSWORD
(or PASQAL_user / PASQAL_pass), and the project id from the argument or PASQAL_PROJECT_ID. The
password goes only into the login form to the Auth0 host; the access token only into the
Authorization header to the Pasqal API host; neither is returned, printed or put in an error.
https only (plain http only for loopback test overrides), no redirects.
"""

from std.collections import Dict
from std.os import getenv
from std.time import sleep

from qumo.core.json import JsonValue, json_quote, parse_json
from qumo.ir.analog import AnalogDevice, AnalogProgram, check_against_device
from qumo.net.http import HttpClient, HttpOptions, HttpRequest, HttpResponse, REDIRECT_NONE
from qumo.net.url import Url, parse_url, quote

comptime CORE_URL = "https://apis.pasqal.cloud/core-fast"
comptime AUTH_URL = "https://pasqal.eu.auth0.com/oauth/token"
comptime CLIENT_ID = "PeZvo7Atx7IVv3iel59asJSb4Ig7vuSB"
comptime AUDIENCE = "https://apis.pasqal.cloud/account/api/v1"
comptime USER_AGENT = "qumo-hardware/1.0 (native Mojo client; +https://github.com/ANFAIA/QUMO)"
comptime E_CREDENTIALS = "QM-E952"
comptime E_UNKNOWN_BACKEND = "QM-E953"
comptime E_INVALID_PAYLOAD = "QM-E955"
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


def _checked(url: String, what: String) raises -> Url:
    var u: Url
    try:
        u = parse_url(url)
    except e:
        raise Error(E_CREDENTIALS + ": " + what + " is not a valid URL")
    if u.scheme != "https" and not _loopback(u.host):
        raise Error(E_CREDENTIALS + ": " + what + " must be https (plain http only for a loopback host)")
    return u^


def _env(primary: String, alias: String) -> String:
    var v = String(getenv(primary, "").strip())
    if v.byte_length() == 0:
        v = String(getenv(alias, "").strip())
    return v^


def _check_id(s: String, what: String, code: String) raises:
    if s.byte_length() == 0 or s.byte_length() > 64:
        raise Error(code + ": invalid Pasqal " + what + " '" + _printable(s, 64) + "'")
    for b in s.as_bytes():
        if not ((b >= 48 and b <= 57) or (b >= 65 and b <= 90) or (b >= 97 and b <= 122) or b == 45):
            raise Error(code + ": invalid Pasqal " + what + " '" + _printable(s, 64) + "'")


def _request(method: String, url: String, var names: List[String], var values: List[String], body: String,
             what: String) raises -> HttpResponse:
    names.append("User-Agent")
    values.append(USER_AGENT)
    var u = _checked(url, what + " URL")
    var opts = HttpOptions(15000, 120000, MAX_RESPONSE_BYTES, False, REDIRECT_NONE, 0, _loopback(u.host), False)
    var client = HttpClient(None)
    try:
        return client.request(HttpRequest(method, url, names^, values^, body), opts)
    except e:
        var hint = String(" Hint: check network access to Pasqal Cloud, then retry")
        if method == "POST" and what == "batch submission":
            hint = " Hint: the request may have reached Pasqal and created a batch; check the Pasqal portal before retrying"
        raise Error(E_JOB + ": Pasqal " + what + " failed: " + _printable(String(e), 300) + "." + hint)


struct PasqalSession(Copyable):
    var core: String
    var token: String
    var project: String

    def __init__(out self, project_id: String) raises:
        var core = String(getenv("QUMO_PASQAL_URL", CORE_URL).strip())
        while core.endswith("/"):
            var t = String(core[byte=0:core.byte_length() - 1])
            core = t^
        _ = _checked(core, "the Pasqal API URL")
        self.core = core^
        var project = project_id if project_id.byte_length() > 0 else _env("PASQAL_PROJECT_ID", "PASQAL_project")
        if project.byte_length() == 0:
            raise Error(E_CREDENTIALS + ": no Pasqal project id. Hint: pass it, or export PASQAL_PROJECT_ID")
        _check_id(project, "project id", E_CREDENTIALS)
        self.project = project^
        var user = _env("PASQAL_USERNAME", "PASQAL_user")
        var password = _env("PASQAL_PASSWORD", "PASQAL_pass")
        if user.byte_length() == 0 or password.byte_length() == 0:
            raise Error(E_CREDENTIALS + ": Pasqal credentials not found. Hint: export PASQAL_USERNAME and PASQAL_PASSWORD "
                        + "(or PASQAL_user and PASQAL_pass) in the shell that runs QUMO")
        var auth = String(getenv("QUMO_PASQAL_AUTH_URL", AUTH_URL).strip())
        var au = _checked(auth, "the Pasqal login URL")
        if not (_loopback(au.host) or au.host == "pasqal.eu.auth0.com"):
            raise Error(E_CREDENTIALS + ": the Pasqal login host '" + au.host + "' is not Pasqal's; the password is sent "
                        + "only to pasqal.eu.auth0.com")
        var body = ("client_id=" + quote(CLIENT_ID, "") + "&grant_type=" + quote("http://auth0.com/oauth/grant-type/password-realm", "")
                    + "&realm=pcs-users&audience=" + quote(AUDIENCE, "") + "&username=" + quote(user, "")
                    + "&password=" + quote(password, ""))
        var names = List[String]()
        var values = List[String]()
        names.append("Content-Type")
        values.append("application/x-www-form-urlencoded")
        names.append("Accept")
        values.append("application/json")
        var r = _request("POST", auth, names^, values^, body, "login")
        if r.status == 401 or r.status == 403:
            raise Error(E_CREDENTIALS + ": Pasqal rejected the login (HTTP " + String(r.status) + "). Hint: check "
                        + "PASQAL_USERNAME / PASQAL_PASSWORD")
        if r.status < 200 or r.status >= 300:
            var detail = String()
            try:
                var v = parse_json(r.text())
                detail = v.get_string("error_description", v.get_string("error", ""))
            except e:
                pass
            detail = detail.replace(password, "<redacted>")
            raise Error(E_CREDENTIALS + ": Pasqal login failed (HTTP " + String(r.status)
                        + (": " + _printable(detail, 200) if detail.byte_length() > 0 else String()) + ")")
        var tok = String()
        try:
            tok = parse_json(r.text()).get_string("access_token", "")
        except e:
            pass
        if tok.byte_length() == 0:
            raise Error(E_CREDENTIALS + ": Pasqal login returned no access token")
        self.token = tok^

    def call(self, method: String, path: String, body: String, what: String, not_found: String = "") raises -> JsonValue:
        var attempts = 3 if method == "GET" else 1
        var attempt = 0
        while True:
            attempt += 1
            var names = List[String]()
            var values = List[String]()
            names.append("Authorization")
            values.append("Bearer " + self.token)
            names.append("Accept")
            values.append("application/json")
            if body.byte_length() > 0:
                names.append("Content-Type")
                values.append("application/json")
            var r = _request(method, self.core + path, names^, values^, body, what)
            if (r.status == 408 or r.status == 429 or r.status == 500 or r.status == 502 or r.status == 503
                    or r.status == 504) and attempt < attempts:
                sleep(Float64(attempt))
                continue
            if r.status == 401 or r.status == 403:
                raise Error(E_CREDENTIALS + ": Pasqal refused " + what + " (HTTP " + String(r.status) + "). Hint: check "
                            + "that the account is a member of project " + self.project)
            if r.status == 404 and not_found.byte_length() > 0:
                raise Error(not_found)
            if r.status >= 300 and r.status < 400:
                raise Error(E_JOB + ": Pasqal " + what + " answered with a redirect (HTTP " + String(r.status)
                            + "); redirects are not followed")
            if r.status < 200 or r.status >= 300:
                var detail = String()
                try:
                    var v = parse_json(r.text())
                    var d = v.get("data")
                    var det = v.get("detail")
                    if not det.is_defined() and d.is_object():
                        det = d.get("detail")
                    if det.is_array() and len(det) > 0:
                        # FastAPI validation errors: [{"loc": [...], "msg": ..., "type": ...}]
                        for k in range(min(len(det), 3)):
                            var e0 = det.at(k)
                            var loc = String()
                            var l = e0.get("loc")
                            for m in range(len(l)):
                                loc += ("." if m > 0 else "") + (l.at(m).as_string() if l.at(m).is_string() else l.at(m).number_text())
                            detail += ("; " if k > 0 else "") + loc + ": " + e0.get_string("msg", "")
                    elif det.is_string():
                        detail = det.as_string()
                    elif d.is_object() and d.get("description").is_string():
                        detail = d.get_string("description", "")
                    elif v.get("message").is_string():
                        detail = v.get_string("message", "")
                except e:
                    pass
                detail = detail.replace(self.token, "<redacted>")
                raise Error(E_JOB + ": Pasqal " + what + " failed (HTTP " + String(r.status)
                            + (": " + _printable(detail, 300) if detail.byte_length() > 0 else String()) + ")")
            var v: JsonValue
            try:
                v = parse_json(r.text())
            except e:
                raise Error(E_JOB + ": Pasqal " + what + " returned a body that is not JSON")
            if v.is_object() and v.has("data"):
                return v.get("data")
            return v^


def pasqal_device(project_id: String, device: String) raises -> AnalogDevice:
    """Specs of `device` (e.g. FRESNEL) as the cloud gives them to this account."""
    var s = PasqalSession(project_id)
    var specs = s.call("GET", "/api/v1/devices/specs", String(), "device specs")
    var raw = specs.get(device)
    if not raw.is_string():
        var names = String()
        for i in range(len(specs)):
            names += (", " if i > 0 else "") + specs.key_at(i)
        raise Error(E_UNKNOWN_BACKEND + ": Pasqal has no device '" + device + "' for this account (available: " + names + ")")
    return AnalogDevice(device, raw.as_string())


struct PasqalBatch(Copyable):
    var batch_id: String
    var job_id: String

    def __init__(out self, var batch_id: String, var job_id: String):
        self.batch_id = batch_id^
        self.job_id = job_id^


def pasqal_submit(project_id: String, prog: AnalogProgram, spec_device: String, device_type: String,
                  runs: Int) raises -> PasqalBatch:
    """Validate `prog` against `spec_device`'s specs, then run it on `device_type` (a QPU such as FRESNEL,
    or an emulator such as EMU_FREE, EMU_MPS, EMU_SV) in a closed batch with one job of `runs` shots.
    Emulators get the virtual form of the device (Pulser Device.to_virtual()): same channels and
    interaction, no layout, so small registers are allowed."""
    var s = PasqalSession(project_id)
    var specs = s.call("GET", "/api/v1/devices/specs", String(), "device specs")
    var raw = specs.get(spec_device)
    if not raw.is_string():
        raise Error(E_UNKNOWN_BACKEND + ": Pasqal has no device specs for '" + spec_device + "'")
    var dev = AnalogDevice(spec_device, raw.as_string())
    # emulators take the device's virtual form (no layout, any positions); QPUs the physical device + layout
    var virtual = device_type.startswith("EMU")
    check_against_device(prog, dev, runs, virtual)
    var seq = (prog.to_pulser_json(dev.virtual_json, "", "") if virtual
               else prog.to_pulser_json(dev.json, dev.layout_json, dev.layout_slug))
    var body = ("{\"project_id\":" + json_quote(s.project) + ",\"sequence_builder\":" + json_quote(seq)
                + ",\"device_type\":" + json_quote(device_type) + ",\"open\":false,\"jobs\":[{\"runs\":" + String(runs)
                + "}]}")
    var b = s.call("POST", "/api/v1/batches", body, "batch submission")
    var bid = b.get_string("id", "")
    _check_id(bid, "batch id", E_JOB)
    var jobs = b.get("jobs")
    if not jobs.is_array() or len(jobs) == 0:
        raise Error(E_JOB + ": Pasqal created batch " + bid + " but returned no job")
    var jid = jobs.at(0).get_string("id", "")
    _check_id(jid, "job id", E_JOB)
    return PasqalBatch(bid^, jid^)


def _state(st: String) -> String:
    var u = st.upper()
    if u == "PENDING" or u == "PAUSED":
        return "QUEUED"
    if u == "RUNNING" or u == "CANCELING":
        return "RUNNING"
    if u == "DONE":
        return "DONE"
    if u == "ERROR" or u == "TIMED_OUT":
        return "ERROR"
    if u == "CANCELED":
        return "CANCELLED"
    return "UNKNOWN"


def pasqal_job_status(project_id: String, job_id: String) raises -> String:
    _check_id(job_id, "job id", E_UNKNOWN_JOB)
    var s = PasqalSession(project_id)
    var nf = E_UNKNOWN_JOB + ": Pasqal job '" + job_id + "' not found for this account"
    var d = s.call("GET", "/api/v1/jobs/" + job_id + "/status", String(), "status of job '" + job_id + "'", nf)
    return _state(d.get_string("status", ""))


def pasqal_job_counts(project_id: String, job_id: String, num_atoms: Int) raises -> Dict[String, Int]:
    """Counter of a finished job; bitstring keys list atoms in register order (atom 0 leftmost)."""
    _check_id(job_id, "job id", E_UNKNOWN_JOB)
    var s = PasqalSession(project_id)
    var nf = E_UNKNOWN_JOB + ": Pasqal job '" + job_id + "' not found for this account"
    var st = s.call("GET", "/api/v1/jobs/" + job_id + "/status", String(), "status of job '" + job_id + "'", nf)
    var state = _state(st.get_string("status", ""))
    if state != "DONE":
        raise Error(E_JOB + ": Pasqal job '" + job_id + "' is " + state)
    var link = s.call("GET", "/api/v1/jobs/" + job_id + "/results_link", String(), "results link of job '" + job_id + "'")
    var url = link.get_string("results_link", "")
    if url.byte_length() == 0:
        raise Error(E_JOB + ": Pasqal job '" + job_id + "' is DONE but has no results link yet. Hint: retry shortly")
    var u = _checked(url, "the Pasqal results link")
    var names = List[String]()
    var values = List[String]()
    names.append("Accept")
    values.append("application/json")
    var r = _request("GET", url, names^, values^, String(), "result download")  # presigned link: no Authorization header
    if r.status < 200 or r.status >= 300:
        raise Error(E_JOB + ": Pasqal result download failed (HTTP " + String(r.status) + ")")
    var res: JsonValue
    try:
        res = parse_json(r.text())
    except e:
        raise Error(E_JOB + ": Pasqal result of job '" + job_id + "' is not JSON")
    var counter = res.get("counter")
    if not counter.is_object():
        raise Error(E_JOB + ": Pasqal result of job '" + job_id + "' has no counter")
    var out = Dict[String, Int]()
    for i in range(len(counter)):
        var key = counter.key_at(i)
        var ok = key.byte_length() == num_atoms
        for b in key.as_bytes():
            if b != 48 and b != 49:
                ok = False
        var v = counter.at(i)
        if not ok or not v.is_number() or v.as_float() < 0:
            raise Error(E_JOB + ": Pasqal result of job '" + job_id + "' has an entry that is not a "
                        + String(num_atoms) + "-bit string with a count")
        out[key] = out.get(key, 0) + Int(v.as_float())
    _ = u.host
    return out^
