"""Quantinuum Nexus client in native Mojo for System Model H2 devices (pytket circuit programs).

Native counterpart of plugins/hardware/qumo_hardware/quantinuum.py (the Python fallback, built
on qnexus + pytket). The wire protocol is the one qnexus 0.51 uses; Nexus has no published
REST reference, so the request bodies were taken from qnexus itself (captured with an httpx
mock transport) and the offline tests compare against them:

    POST {nexus}/auth/tokens/refresh            Cookie myqos_oat=<refresh> -> Set-Cookie myqos_id=<access>
    GET  {nexus}/api/projects/v1beta2?filter[name_exact]=...&filter[archived]=false&scope=user&page[number]=0
    POST {nexus}/api/projects/v1beta2           JSON:API project (when it does not exist yet)
    POST {nexus}/api/circuits/v1beta2           JSON:API circuit: pytket Circuit.to_dict() + name/properties
    POST {nexus}/api/jobs/v1beta3               execute job: QuantinuumConfig + items [{program_id, n_shots, max_cost}]
    GET  {nexus}/api/jobs/v1beta3/{id}?scope=user        attributes.status.status, definition.items[].result_id
    GET  {nexus}/api/results/v1beta3/{rid}?scope=user    attributes = pytket BackendResult dict
    GET  {nexus}/api/v5/available_devices?backend=QuantinuumConfig

Login stays with Quantinuum's own tooling (`qnx login`, browser + MFA): this client reads the
tokens it stores in ~/.qnx/auth (token.json, id.json), sends them only as cookies to the Nexus
host, refreshes the access token in memory when it expired, and never writes, returns or
prints them. https only, except a loopback QUMO_NEXUS_URL for tests. No redirects.

Circuits: the compiled OpenQASM 2 (QUMO basis rz/rx/ry/rzz) becomes a native pytket circuit
1:1 (Rz(t), PhasedX(t, 0), PhasedX(t, 1/2), ZZPhase(t), angles in half-turns; an identity
PhasedX is dropped as pytket does), plus Measure, Reset, Barrier and `if(c==k)` as pytket
Conditional. More than 63 classical bits are split into c_0, c_1, ... like the Python client;
a condition over a split register is refused. Helios (HUGR) devices and the local emulators
stay with the Python fallback.

Configuration: QUMO_NEXUS_URL, NEXUS_DOMAIN / NEXUS_HOST and QUMO_NEXUS_AUTH_DIR. qnexus
settings this client cannot honour (NEXUS_PORT, NEXUS_PROTOCOL, NEXUS_TOKEN_PATH, a qnexus
config file) make it refuse to run, so a token is never sent to a host it was not issued for.
"""

from std.collections import Dict
from std.math import isinf, isnan, pi, floor
from std.os import getenv
from std.time import sleep

from qumo.core.json import JsonValue, json_quote, parse_json
from qumo.ir.gates import Op, op_arity
from qumo.ir.qasm import parse_qasm
from qumo.net.http import HttpClient, HttpOptions, HttpRequest, HttpResponse, REDIRECT_NONE
from qumo.net.url import Url, parse_url, quote

comptime DEFAULT_NEXUS = "https://nexus.quantinuum.com"
comptime USER_AGENT = "qumo-hardware/1.0 (native Mojo client; +https://github.com/ANFAIA/QUMO)"
comptime E_CREDENTIALS = "QM-E952"
comptime E_UNKNOWN_BACKEND = "QM-E953"
comptime E_INVALID_PAYLOAD = "QM-E955"
comptime E_LIMIT = "QM-E956"
comptime E_UNKNOWN_JOB = "QM-E958"
comptime E_JOB = "QM-E959"
comptime H2_MAX_CREG_WIDTH = 63
comptime MAX_RESPONSE_BYTES = 256 * 1024 * 1024
comptime HINT_PY = "use the Python fallback client: --client python (QUMO_HW_CLIENT_QUANTINUUM=python)"
comptime HINT_LOGIN = "log in once with Quantinuum's tooling: `qnx login` (pip install qnexus), which stores tokens in ~/.qnx/auth"


def _printable(s: String, limit: Int) -> String:
    var out = List[UInt8]()
    for b in s.as_bytes():
        if len(out) >= limit:
            break
        out.append(b if (b >= 32 and b < 127) else UInt8(63))
    return String(from_utf8_lossy=Span(out))


def _loopback(host: String) -> Bool:
    return host == "127.0.0.1" or host == "localhost" or host == "::1"


# ---------------------------------------------------------------------------
# Native pytket circuit
# ---------------------------------------------------------------------------
def _half_turns(theta: Float64, period: Float64) raises -> Float64:
    """Radians -> half-turns, reduced to [0, period) as pytket stores them."""
    if isnan(theta) or isinf(theta):
        raise Error(E_INVALID_PAYLOAD + ": rotation angle is not finite")
    var t = theta / pi
    t = t - period * floor(t / period)
    if t >= period or abs(t - period) < 1e-12:
        t = 0.0
    if abs(t) < 1e-15:
        t = 0.0
    return t


def _num(v: Float64) -> String:
    var s = String(v)
    if s.find(".") < 0 and s.find("e") < 0 and s.find("n") < 0:
        s += ".0"
    return s^


def _q(i: Int) -> String:
    return "[\"q\",[" + String(i) + "]]"


def _cbit(k: Int, n_clbits: Int) -> String:
    """Flat clbit k -> its pytket Bit: c[k] up to 63 bits, else c_{k // 63}[k % 63] (the H2 register split)."""
    if n_clbits <= H2_MAX_CREG_WIDTH:
        return "[\"c\",[" + String(k) + "]]"
    return "[\"c_" + String(k // H2_MAX_CREG_WIDTH) + "\",[" + String(k % H2_MAX_CREG_WIDTH) + "]]"


def _c(i: Int) -> String:
    return "[\"c\",[" + String(i) + "]]"


struct QtmCircuit(Copyable):
    var json: String  # pytket Circuit.to_dict() as JSON (without name/properties)
    var n_qubits: Int
    var n_clbits: Int
    var n1q: Int  # PhasedX gates (Rz is virtual), conditional ones included
    var n2q: Int  # ZZPhase gates
    var n_measure_reset: Int
    var registers: List[String]  # classical register names in flat clbit order: "c" or "c_0", "c_1", ...

    def __init__(out self, var json: String, n_qubits: Int, n_clbits: Int, n1q: Int, n2q: Int, n_measure_reset: Int,
                 var registers: List[String]):
        self.json = json^
        self.n_qubits = n_qubits
        self.n_clbits = n_clbits
        self.n1q = n1q
        self.n2q = n2q
        self.n_measure_reset = n_measure_reset
        self.registers = registers^


def qtm_native_circuit(source: String) raises -> QtmCircuit:
    """Compiled OpenQASM 2 (rz/rx/ry/rzz + measure/reset/barrier, optional classical conditions)
    -> native pytket circuit JSON, with the H2 register layout."""
    var c = parse_qasm(source, "qumo-payload")
    var nc = c.num_clbits
    if c.num_measurements == 0:
        raise Error(E_INVALID_PAYLOAD + ": the circuit has no measurements; Nexus returns no counts without them")
    var cmds = List[String]()
    var n1 = 0
    var n2 = 0
    var nm = 0
    for g in c.ops:
        var k = g.kind
        if k == Op.I:
            continue
        var op = String()
        var args = List[String]()
        if k == Op.BARRIER:
            if g.cond >= 0:
                continue
            if g.q0 < 0:
                for q in range(c.num_qubits):
                    args.append(_q(q))
            else:
                for j in range(op_arity(k)):
                    if g.qubit(j) >= 0:
                        args.append(_q(g.qubit(j)))
            if len(args) == 0:
                continue
            var sig = String()
            for i in range(len(args)):
                sig += ("," if i > 0 else "") + "\"Q\""
            cmds.append("{\"args\":[" + String(",").join(args) + "],\"op\":{\"data\":\"\",\"signature\":[" + sig
                        + "],\"type\":\"Barrier\"}}")
            continue
        if k == Op.RZ or k == Op.RX or k == Op.RY or k == Op.RZZ:
            if g.p0.is_symbolic():
                raise Error(E_INVALID_PAYLOAD + ": '" + String(g) + "' has an unbound parameter; bind it with --params")
        if k == Op.MEASURE:
            op = "{\"type\":\"Measure\"}"
            args.append(_q(g.q0))
            args.append(_cbit(g.clbit, nc))
            nm += 1
        elif k == Op.RESET:
            op = "{\"type\":\"Reset\"}"
            args.append(_q(g.q0))
            nm += 1
        elif k == Op.RZ:
            op = "{\"params\":[\"" + _num(_half_turns(g.p0.value(), 4.0)) + "\"],\"type\":\"Rz\"}"
            args.append(_q(g.q0))
        elif k == Op.RX or k == Op.RY:
            var t = _half_turns(g.p0.value(), 4.0)
            var t2 = t - 2.0 * floor(t / 2.0)
            if g.cond < 0 and (t2 < 1e-11 or 2.0 - t2 < 1e-11):
                continue  # identity up to a global phase: pytket drops it, so it is neither sent nor billed
            var phase = String("0.0") if k == Op.RX else String("0.5")
            op = "{\"params\":[\"" + _num(t) + "\",\"" + phase + "\"],\"type\":\"PhasedX\"}"
            args.append(_q(g.q0))
            n1 += 1
        elif k == Op.RZZ:
            op = "{\"params\":[\"" + _num(_half_turns(g.p0.value(), 4.0)) + "\"],\"type\":\"ZZPhase\"}"
            args.append(_q(g.q0))
            args.append(_q(g.q1))
            n2 += 1
        else:
            raise Error(E_INVALID_PAYLOAD + ": '" + String(g) + "' is not in the Quantinuum native basis (rz, rx, ry, "
                        + "rzz); compile for the Quantinuum target")
        if g.cond >= 0:
            # pytket Conditional: condition bits first (bit i of `value` is the i-th listed bit)
            var cbits = List[Int]()
            var cvals = List[Int]()
            if g.cond_reg >= 0:
                var n = c.cond_table[g.cond_reg]
                for i in range(n):
                    cbits.append(c.cond_table[g.cond_reg + 1 + 2 * i])
                    cvals.append(c.cond_table[g.cond_reg + 2 + 2 * i])
            else:
                cbits.append(g.cond)
                cvals.append(g.cond_val)
            # list the bits in clbit order, as pytket's QASM reader does for `if(c==k)`
            for i in range(1, len(cbits)):
                var j = i
                while j > 0 and cbits[j] < cbits[j - 1]:
                    var tb = cbits[j]
                    cbits[j] = cbits[j - 1]
                    cbits[j - 1] = tb
                    var tv = cvals[j]
                    cvals[j] = cvals[j - 1]
                    cvals[j - 1] = tv
                    j -= 1
            if len(cbits) > H2_MAX_CREG_WIDTH or nc > H2_MAX_CREG_WIDTH:
                raise Error(E_LIMIT + ": '" + String(g) + "' is conditioned on more than "
                            + String(H2_MAX_CREG_WIDTH) + " classical bits or on a split register; H2 conditions read "
                            + "one register of at most 63 bits")
            var value = 0
            var cargs = List[String]()
            for i in range(len(cbits)):
                cargs.append(_cbit(cbits[i], nc))
                if cvals[i] == 1:
                    value |= 1 << i
            for x in args:
                cargs.append(x)
            cmds.append("{\"args\":[" + String(",").join(cargs) + "],\"op\":{\"conditional\":{\"op\":" + op
                        + ",\"value\":" + String(value) + ",\"width\":" + String(len(cbits))
                        + "},\"type\":\"Conditional\"}}")
        else:
            cmds.append("{\"args\":[" + String(",").join(args) + "],\"op\":" + op + "}")
    var qubits = List[String]()
    var perm = List[String]()
    for q in range(c.num_qubits):
        qubits.append(_q(q))
        perm.append("[" + _q(q) + "," + _q(q) + "]")
    var bits = List[String]()
    var regs = List[String]()
    for b in range(nc):
        bits.append(_cbit(b, nc))
        var name = String("c") if nc <= H2_MAX_CREG_WIDTH else "c_" + String(b // H2_MAX_CREG_WIDTH)
        if len(regs) == 0 or regs[len(regs) - 1] != name:
            regs.append(name)
    var text = ("{\"bits\":[" + String(",").join(bits) + "],\"commands\":[" + String(",").join(cmds)
                + "],\"created_qubits\":[],\"discarded_qubits\":[],\"implicit_permutation\":["
                + String(",").join(perm) + "],\"phase\":\"0.0\",\"qubits\":[" + String(",").join(qubits) + "]}")
    return QtmCircuit(text^, c.num_qubits, nc, n1, n2, nm, regs^)


def qtm_hqc(n1q: Int, n2q: Int, n_spam: Int, shots: Int) -> Float64:
    """Quantinuum HQC formula: 5 + C (N1q + 10 N2q + 5 Nm) / 5000."""
    return 5.0 + Float64(shots) * (Float64(n1q) + 10.0 * Float64(n2q) + 5.0 * Float64(n_spam)) / 5000.0


# ---------------------------------------------------------------------------
# Session: tokens, cookies, requests
# ---------------------------------------------------------------------------
def _read_token(path: String, field: String) raises -> String:
    var text: String
    try:
        with open(path, "r") as f:
            text = f.read()
    except e:
        return String()
    try:
        var v = parse_json(text)
        var tok = v.get("data").get(field)
        if tok.is_string():
            return tok.as_string()
    except e:
        pass
    return String()


def _check_cookie_value(v: String) raises:
    for b in v.as_bytes():
        if b < 33 or b > 126 or b == 59 or b == 44:
            raise Error(E_CREDENTIALS + ": the stored Nexus token has characters that cannot go in a cookie. Hint: "
                        + HINT_LOGIN)


struct NexusSession(Copyable):
    var base: String
    var refresh: String
    var access: String

    def __init__(out self) raises:
        for var_name in ["NEXUS_PORT", "NEXUS_PROTOCOL", "NEXUS_TOKEN_PATH", "NEXUS_CONFIG_FILE"]:
            if String(getenv(var_name, "").strip()).byte_length() > 0:
                raise Error(E_CREDENTIALS + ": " + var_name + " is set for qnexus; the native Quantinuum client does not "
                            + "read it, so it will not send your Nexus token anywhere. Hint: " + HINT_PY)
        var cfg = String(getenv("HOME", "")) + "/.qnexus/config"
        var has_cfg = False
        try:
            with open(cfg, "r") as f:
                has_cfg = f.read().byte_length() > 0
        except e:
            pass
        if has_cfg:
            raise Error(E_CREDENTIALS + ": a qnexus config file (~/.qnexus/config) is present; the native Quantinuum "
                        + "client does not read it, so it will not send your Nexus token anywhere. Hint: " + HINT_PY)
        var base = String(getenv("QUMO_NEXUS_URL", "").strip())
        if base.byte_length() == 0:
            var dom = String(getenv("NEXUS_DOMAIN", "").strip())
            if dom.byte_length() == 0:
                dom = String(getenv("NEXUS_HOST", "").strip())
            base = ("https://" + dom) if dom.byte_length() > 0 else String(DEFAULT_NEXUS)
        while base.endswith("/"):
            var t = String(base[byte=0:base.byte_length() - 1])
            base = t^
        var u: Url
        try:
            u = parse_url(base + "/")
        except e:
            raise Error(E_CREDENTIALS + ": the Nexus URL is not valid")
        if u.scheme != "https" and not _loopback(u.host):
            raise Error(E_CREDENTIALS + ": the Nexus URL must be https (plain http only for a loopback host)")
        self.base = base^
        var dir = String(getenv("QUMO_NEXUS_AUTH_DIR", "").strip())
        if dir.byte_length() == 0:
            dir = String(getenv("HOME", "")) + "/.qnx/auth"
        self.refresh = _read_token(dir + "/token.json", "refresh_token")
        self.access = _read_token(dir + "/id.json", "access_token")
        if self.refresh.byte_length() == 0:
            raise Error(E_CREDENTIALS + ": not logged in to Quantinuum Nexus (no refresh token in " + dir + "). Hint: "
                        + HINT_LOGIN)
        _check_cookie_value(self.refresh)
        if self.access.byte_length() > 0:
            _check_cookie_value(self.access)

    def _send(self, method: String, path: String, body: String, cookie: String, what: String) raises -> HttpResponse:
        var names = List[String]()
        var values = List[String]()
        names.append("Accept")
        values.append("application/json")
        names.append("User-Agent")
        values.append(USER_AGENT)
        names.append("Cookie")
        values.append(cookie)
        if body.byte_length() > 0:
            names.append("Content-Type")
            values.append("application/json")
        var url = self.base + path
        var u = parse_url(url)
        var opts = HttpOptions(15000, 120000, MAX_RESPONSE_BYTES, False, REDIRECT_NONE, 0, _loopback(u.host), False)
        var client = HttpClient(None)
        try:
            return client.request(HttpRequest(method, url, names^, values^, body), opts)
        except e:
            var hint = String(" Hint: check network access to Nexus, then retry")
            if method == "POST" and what == "job submission":
                hint = " Hint: the request may have reached Nexus and created a job; check the Nexus web UI before retrying"
            raise Error(E_JOB + ": Nexus " + what + " failed: " + _printable(String(e), 300) + "." + hint)

    def _cookie(self) -> String:
        var c = "myqos_oat=" + self.refresh
        if self.access.byte_length() > 0:
            c += "; myqos_id=" + self.access
        return c^

    def _refresh_access(mut self) raises:
        var r = self._send("POST", "/auth/tokens/refresh", String(), "myqos_oat=" + self.refresh, "login refresh")
        if r.status == 429 or r.status == 502 or r.status == 503 or r.status == 504:
            sleep(1.0)
            r = self._send("POST", "/auth/tokens/refresh", String(), "myqos_oat=" + self.refresh, "login refresh")
        if r.status == 401 or r.status == 403:
            raise Error(E_CREDENTIALS + ": the Nexus login has expired. Hint: " + HINT_LOGIN)
        if r.status < 200 or r.status >= 300:
            raise Error(E_JOB + ": Nexus login refresh failed (HTTP " + String(r.status) + ")")
        var found = String()
        for i in range(min(len(r.header_names), len(r.header_values))):
            if r.header_names[i] != "set-cookie":
                continue
            var v = r.header_values[i]
            if v.startswith("myqos_id="):
                var end = v.find(";")
                var raw = String(v[byte=9:end]) if end > 0 else String(v[byte=9:v.byte_length()])
                found = raw^
        if found.byte_length() == 0:
            raise Error(E_CREDENTIALS + ": Nexus refreshed the login without an access token. Hint: " + HINT_LOGIN)
        _check_cookie_value(found)
        self.access = found^

    def call(mut self, method: String, path: String, body: String, what: String,
             not_found: String = "") raises -> JsonValue:
        var attempts = 3 if method == "GET" else 1
        var attempt = 0
        var refreshed = False
        while True:
            attempt += 1
            var r = self._send(method, path, body, self._cookie(), what)
            if r.status == 401 and not refreshed:  # expired access token: refresh once, resend
                self._refresh_access()
                refreshed = True
                attempt -= 1
                continue
            if (r.status == 429 or r.status == 502 or r.status == 503 or r.status == 504) and attempt < attempts:
                sleep(Float64(attempt))
                continue
            if r.status == 401 or r.status == 403:
                raise Error(E_CREDENTIALS + ": Nexus refused " + what + " (HTTP " + String(r.status) + "). Hint: "
                            + HINT_LOGIN)
            if r.status == 404 and not_found.byte_length() > 0:
                raise Error(not_found)
            if r.status >= 300 and r.status < 400:
                raise Error(E_JOB + ": Nexus " + what + " answered with a redirect (HTTP " + String(r.status)
                            + "); redirects are not followed")
            if r.status < 200 or r.status >= 300:
                var detail = String()
                try:
                    var v = parse_json(r.text())
                    if v.is_object():
                        var errs = v.get("errors")
                        if errs.is_array() and len(errs) > 0:
                            var e0 = errs.at(0)
                            detail = e0.get_string("detail", e0.get_string("title", ""))
                        elif v.get("detail").is_string():
                            detail = v.get_string("detail", "")
                        elif v.get("message").is_string():
                            detail = v.get_string("message", "")
                except e:
                    pass
                detail = detail.replace(self.refresh, "<redacted>")
                if self.access.byte_length() > 0:
                    detail = detail.replace(self.access, "<redacted>")
                raise Error(E_JOB + ": Nexus " + what + " failed (HTTP " + String(r.status)
                            + (": " + _printable(detail, 240) if detail.byte_length() > 0 else String()) + ")")
            if len(r.body) == 0:
                return parse_json("{}")
            try:
                return parse_json(r.text())
            except e:
                raise Error(E_JOB + ": Nexus " + what + " returned a body that is not JSON")


def _check_id(s: String, what: String, code: String) raises:
    if s.byte_length() == 0 or s.byte_length() > 64:
        raise Error(code + ": invalid Nexus " + what + " '" + _printable(s, 64) + "'")
    for b in s.as_bytes():
        if not ((b >= 48 and b <= 57) or (b >= 65 and b <= 90) or (b >= 97 and b <= 122) or b == 45):
            raise Error(code + ": invalid Nexus " + what + " '" + _printable(s, 64) + "'")


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------
def _project_id(mut s: NexusSession, name: String) raises -> String:
    var q = ("/api/projects/v1beta2?filter%5Bname_exact%5D=" + quote(name, "") + "&filter%5Barchived%5D=false"
             + "&scope=user&page%5Bnumber%5D=0")
    var found = s.call("GET", q, String(), "project lookup")
    var data = found.get("data")
    if data.is_array() and len(data) > 0:
        var pid = data.at(0).get_string("id", "")
        _check_id(pid, "project id", E_JOB)
        return pid^
    var body = ("{\"data\":{\"attributes\":{\"name\":" + json_quote(name) + ",\"properties\":{}},"
                + "\"relationships\":{},\"type\":\"project\"}}")
    var made = s.call("POST", "/api/projects/v1beta2", body, "project creation")
    var pid = made.get("data").get_string("id", "")
    _check_id(pid, "project id", E_JOB)
    return pid^


def _dash_digits_before(name: String, suffix_len: Int) -> Bool:
    """True when `name` without its last `suffix_len` bytes ends in '-' followed by one or more digits."""
    var bs = name.as_bytes()
    var i = len(bs) - suffix_len - 1
    var digits = 0
    while i >= 0 and bs[i] >= 48 and bs[i] <= 57:
        digits += 1
        i -= 1
    return digits > 0 and i >= 0 and bs[i] == 45


def _targets_emulator(name: String) -> Bool:
    if name.endswith("-Emulator") or name.endswith("-simulator") or name.endswith("-E"):
        return True
    if name.endswith("E-lite") and _dash_digits_before(name, 6):
        return True
    if name.endswith("LE") and _dash_digits_before(name, 2):
        return True
    return name.endswith("E") and _dash_digits_before(name, 1)


def _backend_config(device: String) -> String:
    """QuantinuumConfig(device_name=...).model_dump(exclude_none=True, mode="json") as qnexus 0.51 sends it."""
    # emulators and syntax checkers carry the simulator fields; hardware does not (quantinuum_schemas:
    # re.search(r"(?:-Emulator|-simulator|-E|-\d+E|-\d+LE|-\d+E-lite)$", name) or name.endswith("SC"))
    var sim = _targets_emulator(device) or device.endswith("SC")
    return ("{\"type\":\"QuantinuumConfig\",\"device_name\":" + json_quote(device)
            + (",\"simulator\":\"state-vector\"" if sim else "")
            + ",\"machine_debug\":false,\"attempt_batching\":false,\"allow_implicit_swaps\":true,"
            + "\"postprocess\":false" + (",\"noisy_simulation\":true" if sim else "")
            + ",\"no_opt\":true,\"allow_2q_gate_rebase\":false,"
            + "\"leakage_detection\":false,\"simplify_initial\":false}")


struct QtmSubmitted(Copyable):
    var job_id: String
    var project_id: String
    var circuit_id: String

    def __init__(out self, var job_id: String, var project_id: String, var circuit_id: String):
        self.job_id = job_id^
        self.project_id = project_id^
        self.circuit_id = circuit_id^


def qtm_submit(device: String, project: String, name: String, source: String, shots: Int,
               max_cost: Float64) raises -> QtmSubmitted:
    """Project (get or create), circuit upload, execute job. `max_cost` < 0 sends no server-side cap."""
    if shots < 1:
        raise Error(E_LIMIT + ": shots must be at least 1")
    var circ = qtm_native_circuit(source)
    var s = NexusSession()
    var pid = _project_id(s, project)
    var rel = "\"relationships\":{\"project\":{\"data\":{\"id\":" + json_quote(pid) + ",\"type\":\"project\"}}}"
    var attrs = String(circ.json[byte=0:circ.json.byte_length() - 1]) + ",\"name\":" + json_quote(name) + ",\"properties\":{}}"
    var cbody = "{\"data\":{\"attributes\":" + attrs + "," + rel + ",\"type\":\"circuit\"}}"
    var cref = s.call("POST", "/api/circuits/v1beta2", cbody, "circuit upload")
    var cid = cref.get("data").get_string("id", "")
    _check_id(cid, "circuit id", E_JOB)
    var item = "{\"program_id\":" + json_quote(cid) + ",\"n_shots\":" + String(shots)
    if max_cost >= 0:
        item += ",\"max_cost\":" + String(max_cost)
    item += "}"
    var jbody = ("{\"data\":{\"attributes\":{\"name\":" + json_quote(name) + ",\"description\":\"\",\"properties\":{},"
                 + "\"job_type\":\"execute\",\"definition\":{\"job_definition_type\":\"execute_job_definition\","
                 + "\"backend_config\":" + _backend_config(device) + ",\"user_group\":null,\"target_region\":null,"
                 + "\"valid_check\":true,\"language\":\"AUTO\",\"wasm_module_id\":null,\"gpu_decoder_config_id\":null,"
                 + "\"credential_name\":null,\"compiler_version\":null,\"items\":[" + item + "]}},"
                 + rel + ",\"type\":\"job\"}}")
    var jref = s.call("POST", "/api/jobs/v1beta3", jbody, "job submission")
    var jid = jref.get("data").get_string("id", "")
    if jid.byte_length() == 0:
        raise Error(E_JOB + ": Nexus accepted the job but returned no id. Hint: check the Nexus web UI before retrying")
    _check_id(jid, "job id", E_JOB)
    return QtmSubmitted(jid^, pid^, cid^)


struct QtmState(Copyable):
    var state: String  # QUEUED RUNNING DONE ERROR CANCELLED UNKNOWN
    var detail: String
    var result_id: String

    def __init__(out self, var state: String, var detail: String, var result_id: String):
        self.state = state^
        self.detail = detail^
        self.result_id = result_id^


def _map(st: String) -> String:
    if st == "COMPLETED":
        return "DONE"
    if st == "QUEUED" or st == "SUBMITTED" or st == "RETRYING":
        return "QUEUED"
    if st == "RUNNING":
        return "RUNNING"
    if st == "CANCELLED" or st == "CANCELLING":
        return "CANCELLED"
    if st == "ERROR" or st == "TERMINATED" or st == "DEPLETED":
        return "ERROR"
    return "UNKNOWN"


def _state(mut s: NexusSession, job_id: String) raises -> QtmState:
    _check_id(job_id, "job id", E_UNKNOWN_JOB)
    var nf = E_UNKNOWN_JOB + ": Nexus job '" + job_id + "' not found for this account"
    var j = s.call("GET", "/api/jobs/v1beta3/" + job_id + "?scope=user", String(), "status of job '" + job_id + "'", nf)
    var attrs = j.get("data").get("attributes")
    var st = attrs.get("status")
    var raw = st.get_string("status", "") if st.is_object() else String()
    var detail = String()
    if st.is_object():
        detail = st.get_string("error_detail", "") if st.get("error_detail").is_string() else st.get_string("message", "")
    var rid = String()
    var items = attrs.get("definition").get("items")
    if items.is_array() and len(items) > 0 and items.at(0).get("result_id").is_string():
        rid = items.at(0).get_string("result_id", "")
    return QtmState(_map(raw.upper()), _printable(detail, 300), rid^)


def qtm_status(job_id: String) raises -> QtmState:
    var s = NexusSession()
    return _state(s, job_id)


def _outcome_bits(row: JsonValue, width: Int) raises -> List[Int]:
    """One pytket OutcomeArray row (bytes, big-endian bit order) -> `width` bits."""
    var out = List[Int](capacity=width)
    if not row.is_array() or len(row) < (width + 7) // 8:
        raise Error(E_JOB + ": Nexus result row is shorter than its declared width of " + String(width) + " bits")
    for j in range(width):
        var byte = row.at(j // 8)
        if not byte.is_number():
            raise Error("bad outcome")
        var v = Int(byte.as_float())
        out.append((v >> (7 - j % 8)) & 1)
    return out^


def qtm_result(job_id: String, num_clbits: Int) raises -> Dict[String, Int]:
    """Counts of a completed H2 job (leftmost character = highest clbit)."""
    var s = NexusSession()
    var st = _state(s, job_id)
    if st.state != "DONE":
        raise Error(E_JOB + ": Nexus job '" + job_id + "' is " + st.state
                    + (" (" + st.detail + ")" if st.detail.byte_length() > 0 else String()))
    if st.result_id.byte_length() == 0:
        raise Error(E_JOB + ": Nexus job '" + job_id + "' is completed but lists no result yet. Hint: retry shortly")
    _check_id(st.result_id, "result id", E_JOB)
    var res = s.call("GET", "/api/results/v1beta3/" + st.result_id + "?scope=user", String(), "result of job '" + job_id + "'")
    var attrs = res.get("data").get("attributes")
    var bits = attrs.get("bits")
    if not bits.is_array():
        raise Error(E_JOB + ": Nexus result of job '" + job_id + "' has no bit list (Helios results need " + HINT_PY + ")")
    # position in the outcome rows -> user clbit (register c only)
    var pos_clbit = List[Int]()
    for i in range(len(bits)):
        var b = bits.at(i)
        var reg = b.at(0).as_string()
        var idx = Int(b.at(1).at(0).as_float())
        if reg == "c":
            pos_clbit.append(idx)
        elif reg.startswith("c_"):
            var part = 0
            var digits = String(reg[byte=2:reg.byte_length()])
            if digits.byte_length() == 0:
                raise Error(E_JOB + ": Nexus result uses register '" + reg + "'")
            for d in digits.as_bytes():
                if d < 48 or d > 57:
                    raise Error(E_JOB + ": Nexus result uses register '" + reg + "'")
                part = part * 10 + Int(d) - 48
            pos_clbit.append(part * H2_MAX_CREG_WIDTH + idx)
        else:
            pos_clbit.append(-1)  # a register the payload did not declare: ignored, as get_counts(cbits=...) does
    var out = Dict[String, Int]()
    var shots = attrs.get("shots")
    var counts = attrs.get("counts")
    if shots.is_object() and shots.get("array").is_array():
        var width = Int(shots.get("width").as_float())
        var arr = shots.get("array")
        for i in range(len(arr)):
            _add(out, _outcome_bits(arr.at(i), width), pos_clbit, num_clbits, 1)
    elif counts.is_array():
        for i in range(len(counts)):
            var entry = counts.at(i)
            var oc = entry.get("outcome")
            var width = Int(oc.get("width").as_float())
            var n = Int(entry.get("count").as_float())
            _add(out, _outcome_bits(oc.get("array").at(0), width), pos_clbit, num_clbits, n)
    else:
        raise Error(E_JOB + ": Nexus result of job '" + job_id + "' has neither shots nor counts")
    return out^


def _add(mut out: Dict[String, Int], bits: List[Int], pos_clbit: List[Int], num_clbits: Int, n: Int) raises:
    var flat = List[Int](length=num_clbits, fill=0)
    for p in range(min(len(bits), len(pos_clbit))):
        var k = pos_clbit[p]
        if k == -1:
            continue
        if k < 0 or k >= num_clbits:
            raise Error(E_JOB + ": Nexus result bit c[" + String(k) + "] is outside the circuit's " + String(num_clbits)
                        + " classical bits")
        flat[k] = bits[p]
    var key = String()
    for k in range(num_clbits - 1, -1, -1):
        key += "1" if flat[k] == 1 else "0"
    out[key] = out.get(key, 0) + n


struct QtmDevice(Copyable):
    var name: String
    var qubits: Int

    def __init__(out self, var name: String, qubits: Int):
        self.name = name^
        self.qubits = qubits


def qtm_devices() raises -> List[QtmDevice]:
    var s = NexusSession()
    var lists = s.call("GET", "/api/v5/available_devices?backend=QuantinuumConfig", String(), "device listing")
    var out = List[QtmDevice]()
    if not lists.is_array():
        raise Error(E_JOB + ": Nexus device listing is not a JSON array")
    for i in range(len(lists)):
        var infos = lists.at(i).get("backend_info_list")
        for j in range(len(infos)):
            var info = infos.at(j)
            var name = info.get_string("device_name", "")
            if name.byte_length() == 0:
                continue
            var n = -1
            var nodes = info.get("architecture").get("n_nodes")
            if nodes.is_number():
                n = Int(nodes.as_float())
            out.append(QtmDevice(name^, n))
    return out^
