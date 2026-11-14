"""IBM Quantum Platform client in native Mojo: account, IAM, targets, ISA payloads, Sampler jobs.

Native counterpart of plugins/hardware/qumo_hardware/ibm.py (the Python fallback, built on
qiskit-ibm-runtime). REST API as documented by IBM ("Sampler with the REST API"):

    POST https://iam.cloud.ibm.com/identity/token     grant_type=...:apikey&apikey=... -> access_token
    GET  {api}/backends                               -> {"devices": [...]}
    GET  {api}/backends/{name}/configuration | /properties
    POST {api}/jobs     {program_id: "sampler", backend, params: {pubs: [[qasm3, null, shots]], ...}, cost}
    GET  {api}/jobs/{id}           -> {"state": {"status", "reason"}, ...}
    GET  {api}/jobs/{id}/results   -> {"results": [{"data": {"c": {"samples": ["0x3", ...]}}}]}

{api} = https://quantum.cloud.ibm.com/api/v1, or https://<region>.quantum.cloud.ibm.com/api/v1
for a CRN outside us-east (same rule as qiskit-ibm-runtime). Headers: Authorization Bearer,
Service-CRN, IBM-API-Version.

Credentials, read here at call time, never returned: QISKIT_IBM_TOKEN + QISKIT_IBM_INSTANCE
(a CRN), else the account saved by qiskit-ibm-runtime in ~/.qiskit/qiskit-ibm.json (named,
default-flagged, or the only one). The IAM token lives inside one call. Redirects are not
followed. https only, except loopback http for the test overrides QUMO_IBM_API_URL and IAM_URL.

ISA payload: the compiled OpenQASM 2 is checked against the target (basis gates, coupling map
and direction, width, bound angles) and written as OpenQASM 3 on physical qubits `$n` with
gates from stdgates.inc. Gates outside stdgates.inc (ecr, rzz) and dynamic circuits are refused
here; the Python fallback client handles them.
"""

from std.collections import Dict
from std.math import isinf, isnan
from std.os import getenv
from std.time import sleep

from qumo.core.json import JsonValue, json_quote, parse_json
from qumo.ir.circuit import Instruction
from qumo.ir.gates import Op, op_arity
from qumo.ir.qasm import parse_qasm
from qumo.net.http import HttpClient, HttpOptions, HttpRequest, HttpResponse, REDIRECT_NONE
from qumo.net.url import Url, parse_url, quote

comptime USER_AGENT = "qumo-hardware/1.0 (native Mojo client; +https://github.com/ANFAIA/QUMO)"
comptime API_VERSION = "2026-04-15"  # IBM-API-Version sent by qiskit-ibm-runtime 0.45-0.50
comptime DEFAULT_URL = "https://cloud.ibm.com"
comptime DEFAULT_IAM = "https://iam.cloud.ibm.com"
comptime E_CREDENTIALS = "QM-E952"
comptime E_UNKNOWN_BACKEND = "QM-E953"
comptime E_INVALID_PAYLOAD = "QM-E955"
comptime E_LIMIT = "QM-E956"
comptime E_UNKNOWN_JOB = "QM-E958"
comptime E_JOB = "QM-E959"
comptime MAX_RESPONSE_BYTES = 256 * 1024 * 1024  # results carry one hex string per shot
comptime HINT_PY = "use the Python fallback client: --client python (QUMO_HW_CLIENT_IBM=python)"


def _printable(s: String, limit: Int) -> String:
    var out = List[UInt8]()
    for b in s.as_bytes():
        if len(out) >= limit:
            break
        out.append(b if (b >= 32 and b < 127) else UInt8(63))
    return String(from_utf8_lossy=Span(out))


def _loopback(host: String) -> Bool:
    return host == "127.0.0.1" or host == "localhost" or host == "::1"


def _trim_slash(s: String) -> String:
    var t = String(s.strip())
    while t.endswith("/"):
        var u = String(t[byte=0:t.byte_length() - 1])
        t = u^
    return t^


def _checked(url: String, what: String) raises -> Url:
    var u: Url
    try:
        u = parse_url(url)
    except e:
        raise Error(E_CREDENTIALS + ": " + what + " is not a valid URL")
    if u.scheme != "https" and not _loopback(u.host):
        raise Error(E_CREDENTIALS + ": " + what + " must be https (plain http only for a loopback host)")
    return u^


# ---------------------------------------------------------------------------
# Account
# ---------------------------------------------------------------------------
struct IbmAccount(Copyable):
    var token: String
    var crn: String
    var url: String  # raw service URL, e.g. https://cloud.ibm.com
    var source: String  # "environment" or "saved account '<name>'" (never the token)

    def __init__(out self, var token: String, var crn: String, var url: String, var source: String):
        self.token = token^
        self.crn = crn^
        self.url = url^
        self.source = source^

    def region(self) -> String:
        """Location field of the CRN (crn:v1:bluemix:public:quantum-computing:<location>:...)."""
        var parts = self.crn.split(":")
        if len(parts) > 5:
            return String(parts[5])
        return "us-east"

    def api_base(self) raises -> String:
        var override = String(getenv("QUMO_IBM_API_URL", "").strip())
        if override.byte_length() > 0:
            _ = _checked(override, "QUMO_IBM_API_URL")
            return _trim_slash(override)
        var u = _checked(self.url, "the IBM account URL")
        var region = self.region()
        var prefix = String("") if region == "us-east" else region + "."
        return u.scheme + "://" + prefix + "quantum." + u.host_part() + "/api/v1"


def _no_account() -> Error:
    return Error(E_CREDENTIALS + ": no IBM Quantum account found (QISKIT_IBM_TOKEN is unset and no saved account in "
                 + "~/.qiskit/qiskit-ibm.json). Hint: create an API key at https://quantum.cloud.ibm.com, then "
                 + "export QISKIT_IBM_TOKEN=... and QISKIT_IBM_INSTANCE=<instance CRN>")


def _need_crn(crn: String, source: String) raises:
    if not crn.startswith("crn:"):
        raise Error(E_CREDENTIALS + ": the IBM instance from " + source + " is not a CRN; the native client needs "
                    + "the instance CRN. Hint: set QISKIT_IBM_INSTANCE=crn:v1:..., or " + HINT_PY)


def ibm_account(name: String, instance: String = "") raises -> IbmAccount:
    """Environment first (unless an account name is given), then the saved-account file.
    `instance` (--instance) fills in or overrides the instance CRN."""
    if name.byte_length() == 0:
        var tok = String(getenv("QISKIT_IBM_TOKEN", "").strip())
        if tok.byte_length() > 0:
            var inst = String(getenv("QISKIT_IBM_INSTANCE", "").strip())
            if instance.byte_length() > 0:
                inst = instance
            if inst.byte_length() == 0:
                raise Error(E_CREDENTIALS + ": QISKIT_IBM_TOKEN is set but QISKIT_IBM_INSTANCE is not. Hint: export "
                            + "QISKIT_IBM_INSTANCE=<instance CRN>")
            _need_crn(inst, "QISKIT_IBM_INSTANCE")
            var url = String(getenv("QISKIT_IBM_URL", DEFAULT_URL).strip())
            return IbmAccount(tok^, inst^, url^, "environment")
    var home = String(getenv("QUMO_IBM_ACCOUNT_FILE", "").strip())
    if home.byte_length() == 0:
        home = String(getenv("HOME", "")) + "/.qiskit/qiskit-ibm.json"
    var text: String
    try:
        with open(home, "r") as f:
            text = f.read()
    except e:
        raise _no_account()
    var accounts: JsonValue
    try:
        accounts = parse_json(text)
    except e:
        raise Error(E_CREDENTIALS + ": the saved IBM account file is not valid JSON")
    if not accounts.is_object() or len(accounts) == 0:
        raise _no_account()
    var pick = -1
    if name.byte_length() > 0:
        for i in range(len(accounts)):
            if accounts.key_at(i) == name:
                pick = i
        if pick < 0:
            raise Error(E_CREDENTIALS + ": no saved IBM account named '" + name + "'")
    else:
        for i in range(len(accounts)):
            var flag = accounts.at(i).get("is_default_account")
            if flag.is_bool() and flag.as_bool():
                pick = i
        if pick < 0:
            for i in range(len(accounts)):
                var k = accounts.key_at(i)
                if k == "default-ibm-quantum-platform" or k == "default-ibm-cloud":
                    pick = i
                    break
        if pick < 0 and len(accounts) == 1:
            pick = 0
        if pick < 0:
            raise Error(E_CREDENTIALS + ": several saved IBM accounts and none is the default. Hint: pass --account")
    var acc = accounts.at(pick)
    var source = "saved account '" + accounts.key_at(pick) + "'"
    var channel = acc.get_string("channel", "ibm_quantum_platform")
    if channel != "ibm_quantum_platform" and channel != "ibm_cloud":
        raise Error(E_CREDENTIALS + ": " + source + " uses channel '" + channel + "', which the native client does "
                    + "not support. Hint: " + HINT_PY)
    var pe = acc.get("private_endpoint")
    if pe.is_bool() and pe.as_bool():
        raise Error(E_CREDENTIALS + ": " + source + " uses a private endpoint, which the native client does not "
                    + "support. Hint: " + HINT_PY)
    var tok = acc.get_string("token", "")
    if tok.byte_length() == 0:
        raise Error(E_CREDENTIALS + ": " + source + " has no API key")
    var inst = acc.get_string("instance", "")
    _need_crn(inst, source)
    return IbmAccount(tok^, inst^, acc.get_string("url", DEFAULT_URL), source^)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
def _form_escape(s: String) -> String:
    return quote(s, "")


def _request(method: String, url: String, var names: List[String], var values: List[String], body: String,
             what: String) raises -> HttpResponse:
    # IBM's edge (Cloudflare) refuses requests without a client signature (error 1010)
    names.append("User-Agent")
    values.append(USER_AGENT)
    var u = parse_url(url)
    var opts = HttpOptions(15000, 120000, MAX_RESPONSE_BYTES, False, REDIRECT_NONE, 0, _loopback(u.host), False)
    var client = HttpClient(None)
    try:
        return client.request(HttpRequest(method, url, names^, values^, body), opts)
    except e:
        var hint = String(" Hint: check network access and https://quantum.cloud.ibm.com status, then retry")
        if method == "POST" and what == "job submission":
            hint = " Hint: the request may have reached IBM and created a job; check the IBM Quantum dashboard before retrying"
        raise Error(E_JOB + ": IBM " + what + " failed: " + _printable(String(e), 300) + "." + hint)


def _iam_token(acc: IbmAccount) raises -> String:
    var iam = _trim_slash(String(getenv("IAM_URL", DEFAULT_IAM)))
    var iu = _checked(iam, "IAM_URL")
    if not (_loopback(iu.host) or iu.host == "iam.cloud.ibm.com" or iu.host.endswith(".cloud.ibm.com")):
        raise Error(E_CREDENTIALS + ": IAM_URL host '" + iu.host + "' is not an IBM Cloud host; QUMO sends the API "
                    + "key only to *.cloud.ibm.com")
    var names = List[String]()
    var values = List[String]()
    names.append("Content-Type")
    values.append("application/x-www-form-urlencoded")
    names.append("Accept")
    values.append("application/json")
    var body = "grant_type=urn%3Aibm%3Aparams%3Aoauth%3Agrant-type%3Aapikey&apikey=" + _form_escape(acc.token)
    var r = _request("POST", iam + "/identity/token", names^, values^, body, "IAM login")
    if r.status == 400 or r.status == 401 or r.status == 403:
        raise Error(E_CREDENTIALS + ": IBM Cloud IAM rejected the API key from " + acc.source + " (HTTP "
                    + String(r.status) + "). Hint: create a new API key at https://quantum.cloud.ibm.com")
    if r.status < 200 or r.status >= 300:
        raise Error(E_JOB + ": IBM Cloud IAM login failed (HTTP " + String(r.status) + ")")
    try:
        var tok = parse_json(r.text()).get("access_token")
        if tok.is_string() and tok.as_string().byte_length() > 0:
            return tok.as_string()
    except e:
        pass
    raise Error(E_CREDENTIALS + ": IBM Cloud IAM returned no access token")


def _detail(r: HttpResponse) -> String:
    """IBM error bodies: {"errors": [{"message", "code"}], "trace"} or {"message"} or {"error"}."""
    var out = String()
    try:
        var v = parse_json(r.text())
        if v.is_object():
            var errs = v.get("errors")
            if errs.is_array() and len(errs) > 0 and errs.at(0).get("message").is_string():
                out = errs.at(0).get("message").as_string()
            elif v.get("message").is_string():
                out = v.get("message").as_string()
            elif v.get("error").is_string():
                out = v.get("error").as_string()
    except e:
        pass
    return _printable(out, 240)


struct IbmSession(Copyable):
    """Account + one IAM token, inside one native call only."""
    var acc: IbmAccount
    var token: String
    var base: String

    def __init__(out self, name: String, instance: String) raises:
        var acc = ibm_account(name, instance)
        if instance.byte_length() > 0:  # --instance overrides the account's instance
            _need_crn(instance, "--instance")
            acc.crn = instance
        self.acc = acc^
        self.base = self.acc.api_base()
        self.token = _iam_token(self.acc)

    def call(self, method: String, path: String, body: String, what: String, not_found: String = "") raises -> JsonValue:
        var attempts = 3 if method == "GET" else 1
        var attempt = 0
        while True:
            attempt += 1
            var names = List[String]()
            var values = List[String]()
            names.append("Authorization")
            values.append("Bearer " + self.token)
            names.append("Service-CRN")
            values.append(self.acc.crn)
            names.append("IBM-API-Version")
            values.append(API_VERSION)
            names.append("Accept")
            values.append("application/json")
            if body.byte_length() > 0:
                names.append("Content-Type")
                values.append("application/json")
            var r = _request(method, self.base + path, names^, values^, body, what)
            if (r.status == 429 or r.status == 502 or r.status == 503 or r.status == 504) and attempt < attempts:
                sleep(Float64(attempt))
                continue
            if r.status == 401 or r.status == 403:
                raise Error(E_CREDENTIALS + ": IBM Quantum rejected the request for " + what + " (HTTP "
                            + String(r.status) + ", " + self.acc.source + "). Hint: check the API key and that the "
                            + "instance CRN can use this backend")
            if r.status == 404 and not_found.byte_length() > 0:
                raise Error(not_found)
            if r.status >= 300 and r.status < 400:
                raise Error(E_JOB + ": IBM " + what + " answered with a redirect (HTTP " + String(r.status)
                            + "); redirects are not followed")
            if r.status < 200 or r.status >= 300:
                var d = _detail(r)
                if self.token.byte_length() > 0:
                    d = d.replace(self.token, "<redacted>")
                if self.acc.token.byte_length() > 0:
                    d = d.replace(self.acc.token, "<redacted>")
                raise Error(E_JOB + ": IBM " + what + " failed (HTTP " + String(r.status)
                            + (": " + d if d.byte_length() > 0 else String()) + ")")
            if len(r.body) == 0:
                return parse_json("{}")
            try:
                return parse_json(r.text())
            except e:
                raise Error(E_JOB + ": IBM " + what + " returned a body that is not JSON")


def _check_id(s: String, what: String, code: String) raises:
    if s.byte_length() == 0 or s.byte_length() > 80:
        raise Error(code + ": invalid IBM " + what + " '" + _printable(s, 80) + "'")
    for b in s.as_bytes():
        var ok = (b >= 48 and b <= 57) or (b >= 65 and b <= 90) or (b >= 97 and b <= 122) or b == 45 or b == 95
        if not ok:
            raise Error(code + ": invalid IBM " + what + " '" + _printable(s, 80) + "'")


# ---------------------------------------------------------------------------
# Backends and targets
# ---------------------------------------------------------------------------
struct IbmBackendRow(Copyable):
    var name: String
    var qubits: Int
    var status: String
    var pending: Int

    def __init__(out self, var name: String, qubits: Int, var status: String, pending: Int):
        self.name = name^
        self.qubits = qubits
        self.status = status^
        self.pending = pending


def _int_of(v: JsonValue, default: Int) raises -> Int:
    if v.is_number():
        return Int(v.as_float())
    return default


def ibm_backends(account: String, instance: String) raises -> List[IbmBackendRow]:
    var s = IbmSession(account, instance)
    var data = s.call("GET", "/backends", String(), "backend listing")
    var devices = data.get("devices") if data.is_object() else data.copy()
    var out = List[IbmBackendRow]()
    if not devices.is_array():
        raise Error(E_JOB + ": IBM backend listing has no 'devices' array")
    for i in range(len(devices)):
        var d = devices.at(i)
        if d.is_string():
            out.append(IbmBackendRow(d.as_string(), -1, String(), -1))
            continue
        if not d.is_object():
            continue
        var name = d.get_string("name", d.get_string("backend_name", ""))
        if name.byte_length() == 0:
            continue
        var st = d.get("status")
        var status = st.as_string() if st.is_string() else (st.get_string("name", "") if st.is_object() else String())
        var q = _int_of(d.get("qubits"), _int_of(d.get("n_qubits"), -1))
        var pending = _int_of(d.get("queue_length"), _int_of(d.get("pending_jobs"), -1))
        out.append(IbmBackendRow(name^, q, status^, pending))
    return out^


struct IbmTarget(Copyable):
    var name: String
    var num_qubits: Int
    var basis: List[String]
    var edges: List[Int]  # directed pairs, flattened a0, b0, a1, b1, ...
    var edge_errors: List[Float64]  # per directed pair, -1 when unknown
    var symmetric: Bool
    var err_1q: Float64  # medians, -1 when unknown
    var err_2q: Float64
    var err_ro: Float64
    var dur_1q: Float64  # seconds, -1 when unknown
    var dur_2q: Float64
    var dur_ro: Float64
    var readout: List[Float64]  # per qubit, -1 when unknown
    var max_shots: Int
    var simulator: Bool
    var processor: String
    var dt: Float64  # seconds, -1 when unknown
    var rep_delay: Float64  # seconds, -1 when unknown
    var fractional: Bool  # rzz offered by the backend and requested
    var native_2q: String
    var faulty_qubits: List[Int]  # operational = 0 in the properties: left out of the target
    var faulty_edges: Int  # couplings left out (faulty gate or qubit)

    def __init__(out self):
        self.name = String()
        self.num_qubits = 0
        self.basis = List[String]()
        self.edges = List[Int]()
        self.edge_errors = List[Float64]()
        self.symmetric = True
        self.err_1q = -1.0
        self.err_2q = -1.0
        self.err_ro = -1.0
        self.dur_1q = -1.0
        self.dur_2q = -1.0
        self.dur_ro = -1.0
        self.readout = List[Float64]()
        self.max_shots = 100000
        self.simulator = False
        self.processor = String()
        self.dt = -1.0
        self.rep_delay = -1.0
        self.fractional = False
        self.native_2q = String()
        self.faulty_qubits = List[Int]()
        self.faulty_edges = 0


def _median(var xs: List[Float64]) -> Float64:
    if len(xs) == 0:
        return -1.0
    sort(xs)
    var n = len(xs)
    if n % 2 == 1:
        return xs[n // 2]
    return (xs[n // 2 - 1] + xs[n // 2]) / 2.0


def _seconds(value: Float64, unit: String) -> Float64:
    if unit == "ns":
        return value * 1e-9
    if unit == "us" or unit == "µs":
        return value * 1e-6
    if unit == "ms":
        return value * 1e-3
    return value


def _param(params: JsonValue, name: String, mut unit: String) raises -> Float64:
    """Value of the parameter `name` in an IBM properties parameter list, -1 when absent."""
    for i in range(len(params)):
        var p = params.at(i)
        if p.get_string("name", "") == name and p.get("value").is_number():
            unit = p.get_string("unit", "")
            return p.get("value").as_float()
    return -1.0


def _has(xs: List[String], s: String) -> Bool:
    for x in xs:
        if x == s:
            return True
    return False


def ibm_target(account: String, instance: String, backend: String, use_fractional: Bool) raises -> IbmTarget:
    _check_id(backend, "backend name", E_UNKNOWN_BACKEND)
    var s = IbmSession(account, instance)
    var nf = E_UNKNOWN_BACKEND + ": IBM backend '" + backend + "' is not available to this account. Hint: list what "
    nf += "it sees with `python -m qumo_hardware backends --provider ibm --online`"
    var conf = s.call("GET", "/backends/" + backend + "/configuration", String(), "configuration of '" + backend + "'", nf)
    var props = s.call("GET", "/backends/" + backend + "/properties", String(), "properties of '" + backend + "'", nf)
    var t = IbmTarget()
    t.name = backend
    t.num_qubits = _int_of(conf.get("n_qubits"), 0)
    if t.num_qubits <= 0:
        raise Error(E_UNKNOWN_BACKEND + ": IBM backend '" + backend + "' has no qubit count in its configuration")
    var names = List[String]()
    var bg = conf.get("basis_gates")
    for i in range(len(bg)):
        if bg.at(i).is_string():
            names.append(bg.at(i).as_string())
    for g in ["rz", "sx", "x"]:
        if _has(names, g):
            t.basis.append(g)
    var twoq = List[String]()
    for g in ["cz", "ecr", "cx"]:
        if _has(names, g):
            twoq.append(g)
            t.basis.append(g)
    if len(twoq) == 0:
        raise Error(E_UNKNOWN_BACKEND + ": backend '" + backend + "' exposes no cz/ecr/cx gate")
    t.native_2q = twoq[0]
    var has_rzz = _has(names, "rzz")
    t.fractional = use_fractional and has_rzz
    if t.fractional:
        t.basis.append("rzz")
        if _has(names, "rx"):
            t.basis.append("rx")
    t.max_shots = _int_of(conf.get("max_shots"), 100000)
    var sim = conf.get("simulator")
    t.simulator = sim.is_bool() and sim.as_bool()
    var pt = conf.get("processor_type")
    if pt.is_object():
        t.processor = pt.get_string("family", "")
        var rev = pt.get("revision")
        if rev.is_number() or rev.is_string():
            t.processor += " r" + (rev.as_string() if rev.is_string() else rev.number_text())
    if conf.get("dt").is_number():
        t.dt = conf.get("dt").as_float() * 1e-9  # raw configuration: nanoseconds
    if conf.get("default_rep_delay").is_number():
        t.rep_delay = conf.get("default_rep_delay").as_float() * 1e-6  # raw configuration: microseconds
    # faulty qubits (operational = 0) are left out, as qiskit-ibm-runtime does (filter_faulty)
    var faulty = List[Bool](length=t.num_qubits, fill=False)
    var ro = List[Float64](length=t.num_qubits, fill=-1.0)
    var ro_errs = List[Float64]()
    var ro_durs = List[Float64]()
    var qprops = props.get("qubits")
    for q in range(min(len(qprops), t.num_qubits)):
        var unit = String()
        if _param(qprops.at(q), "operational", unit) == 0.0:
            faulty[q] = True
            t.faulty_qubits.append(q)
            continue
        var e = _param(qprops.at(q), "readout_error", unit)
        if e >= 0:
            ro[q] = e
            ro_errs.append(e)
        var d = _param(qprops.at(q), "readout_length", unit)
        if d >= 0:
            ro_durs.append(_seconds(d, unit))
    # two-qubit pairs and errors from the properties; coupling map when a gate has none
    var seen = Dict[String, Int]()
    var errs2 = List[Float64]()
    var durs2 = List[Float64]()
    var errs1 = List[Float64]()
    var durs1 = List[Float64]()
    var any_2q_props = False
    var gates = props.get("gates")
    for i in range(len(gates)):
        var g = gates.at(i)
        var gname = g.get_string("gate", "")
        var qs = g.get("qubits")
        var params = g.get("parameters")
        var unit = String()
        var err = _param(params, "gate_error", unit)
        var dur = _param(params, "gate_length", unit)
        if dur >= 0:
            dur = _seconds(dur, unit)
        var broken = _param(params, "operational", unit) == 0.0
        if _has(twoq, gname) and qs.is_array() and len(qs) == 2:
            any_2q_props = True
            var a = _int_of(qs.at(0), -1)
            var b = _int_of(qs.at(1), -1)
            var key = String(a) + "-" + String(b)
            if a < 0 or b < 0 or a >= t.num_qubits or b >= t.num_qubits or key in seen:
                continue
            if broken or faulty[a] or faulty[b]:
                t.faulty_edges += 1
                continue
            seen[key] = len(t.edge_errors)
            t.edges.append(a)
            t.edges.append(b)
            t.edge_errors.append(err)
            if err >= 0:
                errs2.append(err)
            if dur >= 0:
                durs2.append(dur)
        elif gname == "sx" and qs.is_array() and len(qs) == 1:
            var q = _int_of(qs.at(0), -1)
            if broken or q < 0 or q >= t.num_qubits or faulty[q]:
                continue
            if err >= 0:
                errs1.append(err)
            if dur >= 0:
                durs1.append(dur)
    if not any_2q_props:
        var cm = conf.get("coupling_map")
        for i in range(len(cm)):
            var pair = cm.at(i)
            if pair.is_array() and len(pair) == 2:
                var a = _int_of(pair.at(0), -1)
                var b = _int_of(pair.at(1), -1)
                if a < 0 or b < 0 or a >= t.num_qubits or b >= t.num_qubits or faulty[a] or faulty[b]:
                    continue
                seen[String(a) + "-" + String(b)] = len(t.edge_errors)
                t.edges.append(a)
                t.edges.append(b)
                t.edge_errors.append(-1.0)
    for i in range(0, len(t.edges), 2):
        if not ((String(t.edges[i + 1]) + "-" + String(t.edges[i])) in seen):
            t.symmetric = False
    t.readout = ro^
    t.err_1q = _median(errs1^)
    t.err_2q = _median(errs2^)
    t.err_ro = _median(ro_errs^)
    t.dur_1q = _median(durs1^)
    t.dur_2q = _median(durs2^)
    t.dur_ro = _median(ro_durs^)
    return t^


# ---------------------------------------------------------------------------
# ISA check and OpenQASM 3
# ---------------------------------------------------------------------------
struct IbmPayload(Copyable):
    var qasm3: String
    var duration_s: Float64  # ASAP estimate from the target's median durations
    var two_qubit_gates: Int
    var active_qubits: Int

    def __init__(out self, var qasm3: String, duration_s: Float64, two_qubit_gates: Int, active_qubits: Int):
        self.qasm3 = qasm3^
        self.duration_s = duration_s
        self.two_qubit_gates = two_qubit_gates
        self.active_qubits = active_qubits


def _angle(g: Instruction) raises -> String:
    if g.p0.is_symbolic():
        raise Error(E_INVALID_PAYLOAD + ": '" + String(g) + "' has an unbound parameter; bind it with --params")
    var v = g.p0.value()
    if isnan(v) or isinf(v):
        raise Error(E_INVALID_PAYLOAD + ": '" + String(g) + "' has an angle that is not finite")
    return String(v)


def ibm_isa_payload(source: String, num_qubits: Int, basis: List[String], edges: List[Int], symmetric: Bool,
                    dur_1q: Float64, dur_2q: Float64, dur_ro: Float64) raises -> IbmPayload:
    """Compiled OpenQASM 2 -> ISA OpenQASM 3 on physical qubits, after checking it against the target."""
    var c = parse_qasm(source, "isa")
    if c.num_qubits > num_qubits:
        raise Error(E_INVALID_PAYLOAD + ": the circuit has " + String(c.num_qubits) + " qubits; the backend has "
                    + String(num_qubits))
    if c.num_measurements == 0:
        raise Error(E_INVALID_PAYLOAD + ": the circuit has no measurements; the Sampler returns no counts without them")
    var edge_ok = Dict[String, Bool]()
    for i in range(0, len(edges), 2):
        edge_ok[String(edges[i]) + "-" + String(edges[i + 1])] = True
        if symmetric:
            edge_ok[String(edges[i + 1]) + "-" + String(edges[i])] = True
    var clock = List[Float64](length=num_qubits, fill=0.0)
    var used = List[Bool](length=num_qubits, fill=False)
    var d1 = dur_1q if dur_1q > 0 else 0.0
    var d2 = dur_2q if dur_2q > 0 else 0.0
    var dr = dur_ro if dur_ro > 0 else 0.0
    var body = String()
    var twoq = 0
    for g in c.ops:
        var k = g.kind
        if g.cond >= 0:
            raise Error(E_INVALID_PAYLOAD + ": '" + String(g) + "' is classically conditioned; the native IBM client "
                        + "sends static circuits only. Hint: " + HINT_PY)
        if k == Op.RESET:
            body += "reset $" + String(g.q0) + ";\n"
            clock[g.q0] += dr
            used[g.q0] = True
            continue
        if k == Op.BARRIER:
            if g.q0 < 0:  # the parser keeps a barrier as one instruction over every qubit
                body += "barrier;\n"
                continue
            var line = String("barrier ")
            var first = True
            for j in range(op_arity(k)):
                var q = g.qubit(j)
                if q < 0:
                    continue
                line += ("" if first else ", ") + "$" + String(q)
                first = False
            if not first:
                body += line + ";\n"
            continue
        if k == Op.I:
            continue
        if k == Op.MEASURE:
            body += "c[" + String(g.clbit) + "] = measure $" + String(g.q0) + ";\n"
            clock[g.q0] += dr
            used[g.q0] = True
            continue
        var name: String
        var params = String()
        if k == Op.RZ:
            name = "rz"
            params = "(" + _angle(g) + ")"
        elif k == Op.SX:
            name = "sx"
        elif k == Op.X:
            name = "x"
        elif k == Op.RX:
            name = "rx"
            params = "(" + _angle(g) + ")"
        elif k == Op.CZ:
            name = "cz"
        elif k == Op.CX:
            name = "cx"
        elif k == Op.ECR or k == Op.RZZ:
            raise Error(E_INVALID_PAYLOAD + ": '" + String(g) + "' is not in OpenQASM 3 stdgates.inc, so the native "
                        + "IBM client cannot send it. Hint: " + HINT_PY)
        else:
            raise Error(E_INVALID_PAYLOAD + ": '" + String(g) + "' is not a basis gate of the backend")
        if not _has(basis, name):
            raise Error(E_INVALID_PAYLOAD + ": gate '" + name + "' is not in the backend's basis ("
                        + String(", ").join(basis) + ")")
        if op_arity(k) == 2:
            var key = String(g.q0) + "-" + String(g.q1)
            if not (key in edge_ok):
                raise Error(E_INVALID_PAYLOAD + ": '" + String(g) + "' acts on (" + String(g.q0) + ", " + String(g.q1)
                            + "), which is not a native coupling (or not in its native direction). Hint: "
                            + "compile for this backend's target; for direction finishing, " + HINT_PY)
            var start = max(clock[g.q0], clock[g.q1])
            clock[g.q0] = start + d2
            clock[g.q1] = start + d2
            used[g.q0] = True
            used[g.q1] = True
            twoq += 1
            body += name + params + " $" + String(g.q0) + ", $" + String(g.q1) + ";\n"
        else:
            if name == "rz":
                pass  # virtual Z: no duration
            else:
                clock[g.q0] += d1
            used[g.q0] = True
            body += name + params + " $" + String(g.q0) + ";\n"
    var dur = 0.0
    var active = 0
    for q in range(num_qubits):
        dur = max(dur, clock[q])
        if used[q]:
            active += 1
    var text = "OPENQASM 3.0;\ninclude \"stdgates.inc\";\nbit[" + String(c.num_clbits) + "] c;\n" + body
    return IbmPayload(text^, dur, twoq, active)


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------
def ibm_submit(account: String, instance: String, backend: String, qasm3: String, shots: Int,
               cap_seconds: Int) raises -> String:
    """POST /jobs with one Sampler PUB; twirling and dynamical decoupling off; `cost` = max_execution_time."""
    _check_id(backend, "backend name", E_UNKNOWN_BACKEND)
    if shots < 1:
        raise Error(E_LIMIT + ": shots must be at least 1")
    var s = IbmSession(account, instance)
    var body = ("{\"program_id\":\"sampler\",\"backend\":" + json_quote(backend)
                + ",\"params\":{\"pubs\":[[" + json_quote(qasm3) + ",null," + String(shots) + "]]"
                + ",\"options\":{\"dynamical_decoupling\":{\"enable\":false},"
                + "\"twirling\":{\"enable_gates\":false,\"enable_measure\":false}},\"version\":2}")
    if cap_seconds > 0:
        body += ",\"cost\":" + String(cap_seconds)
    body += "}"
    var resp = s.call("POST", "/jobs", body, "job submission")
    var jid = resp.get_string("id", "") if resp.is_object() else String()
    if jid.byte_length() == 0:
        raise Error(E_JOB + ": IBM accepted the request but returned no job id. Hint: check the IBM Quantum dashboard "
                    + "for a new job before retrying")
    _check_id(jid, "job id", E_JOB)
    return jid^


struct IbmJobState(Copyable):
    var state: String  # QUEUED RUNNING DONE ERROR CANCELLED UNKNOWN
    var reason: String

    def __init__(out self, var state: String, var reason: String):
        self.state = state^
        self.reason = reason^


def _job_state(s: IbmSession, job_id: String) raises -> IbmJobState:
    _check_id(job_id, "job id", E_UNKNOWN_JOB)
    var nf = E_UNKNOWN_JOB + ": IBM job '" + job_id + "' not found for this account"
    var j = s.call("GET", "/jobs/" + job_id, String(), "status of job '" + job_id + "'", nf)
    var st = j.get("state")
    var raw = String()
    var reason = String()
    if st.is_object():
        raw = st.get_string("status", "")
        reason = st.get_string("reason", "")
        var code = st.get("reasonCode") if st.has("reasonCode") else st.get("reason_code")
        if code.is_number() and reason.byte_length() > 0:
            reason = "Error code " + code.number_text() + "; " + reason
    elif j.get("status").is_string():
        raw = j.get_string("status", "")
    var up = raw.upper()
    var mapped = String("UNKNOWN")
    if up == "QUEUED" or up == "INITIALIZING" or up == "VALIDATING":
        mapped = "QUEUED"
    elif up == "RUNNING":
        mapped = "RUNNING"
    elif up == "COMPLETED":
        mapped = "DONE"
    elif up == "FAILED":
        mapped = "ERROR"
    elif up.startswith("CANCELLED"):
        mapped = "CANCELLED"
        if up.find("TOO LONG") >= 0 and reason.byte_length() == 0:
            reason = "ran longer than max_execution_time (the cap QUMO sent as `cost`)"
    return IbmJobState(mapped^, _printable(reason, 300))


def ibm_status(account: String, instance: String, job_id: String) raises -> IbmJobState:
    var s = IbmSession(account, instance)
    return _job_state(s, job_id)


def _hex_value(t: String) raises -> Int:
    var bs = t.as_bytes()
    var i = 0
    if len(bs) >= 2 and bs[0] == 48 and (bs[1] == 120 or bs[1] == 88):
        i = 2
    if i >= len(bs) or len(bs) - i > 16:
        raise Error("bad sample")
    var v = 0
    while i < len(bs):
        var c = Int(bs[i])
        var d = -1
        if c >= 48 and c <= 57:
            d = c - 48
        elif c >= 97 and c <= 102:
            d = c - 87
        elif c >= 65 and c <= 70:
            d = c - 55
        if d < 0 or v >= (1 << 58):
            raise Error("bad sample")
        v = v * 16 + d
        i += 1
    return v


def ibm_result(account: String, instance: String, job_id: String, num_clbits: Int) raises -> Dict[String, Int]:
    """GET /jobs/<id>/results -> counts over the clbits (leftmost character = highest clbit)."""
    var s = IbmSession(account, instance)
    var st = _job_state(s, job_id)
    if st.state != "DONE":
        raise Error(E_JOB + ": IBM job '" + job_id + "' is " + st.state
                    + (" (" + st.reason + ")" if st.reason.byte_length() > 0 else String()))
    var res = s.call("GET", "/jobs/" + job_id + "/results", String(), "result of job '" + job_id + "'")
    var results = res.get("results")
    if not results.is_array() or len(results) == 0:
        raise Error(E_JOB + ": IBM job '" + job_id + "' returned no results")
    var data = results.at(0).get("data")
    if not data.is_object() or len(data) == 0:
        raise Error(E_JOB + ": IBM job '" + job_id + "' has no measurement data")
    var reg = data.get("c") if data.has("c") else data.at(0)
    var samples = reg.get("samples")
    if not samples.is_array():
        raise Error(E_JOB + ": IBM job '" + job_id + "' result has no samples (it may be a server dry run or an "
                    + "encoded result). Hint: " + HINT_PY)
    if num_clbits < 1 or num_clbits > 62:
        raise Error(E_JOB + ": cannot build counts for " + String(num_clbits) + " classical bits")
    var out = Dict[String, Int]()
    for i in range(len(samples)):
        var sv = samples.at(i)
        if not sv.is_string():
            raise Error(E_JOB + ": IBM job '" + job_id + "' returned a sample that is not a hex string")
        var v: Int
        try:
            v = _hex_value(sv.as_string())
        except e:
            raise Error(E_JOB + ": IBM job '" + job_id + "' returned a sample that is not a hex string")
        if (v >> num_clbits) != 0:
            raise Error(E_JOB + ": IBM job '" + job_id + "' returned a sample wider than " + String(num_clbits) + " bits")
        var bits = String()
        for k in range(num_clbits - 1, -1, -1):
            bits += "1" if ((v >> k) & 1) == 1 else "0"
        out[bits] = out.get(bits, 0) + 1
    return out^
