"""SpeQtrum (Qilimanjaro) provider: digital propagation over HTTPS.

Wire protocol, with endpoint and field names as specified for QUMO (they mirror the
public qilisdk client, Apache-2.0; no code is taken from it):

    POST {base}/authorisation-tokens          {grantType, assertion, scope} -> {accessToken, ...}
    GET  {base}/devices                       -> {"items": [{code, nqubits, name, description, type, status}]}
    POST {base}/execute                       {device_code, payload, job_type, meta} -> {"id", "message"}
    GET  {base}/jobs/{id}?payload=false&result=true&error=true

Security model
* Credentials come only from the environment (QUMO_SPEQTRUM_USERNAME, QUMO_SPEQTRUM_APIKEY,
  QUMO_SPEQTRUM_URL, QUMO_SPEQTRUM_AUDIENCE); constructor arguments may override them. They
  are never written to disk, printed, logged, or put in an exception or repr. The access
  token lives in memory only.
* The login assertion is NOT a signed JWT. It is base64url(JSON{username, api_key, audience,
  iat}), so it carries the API key encoded, not encrypted. It is sent only over https (plain
  http only to a loopback address, for tests) and only to the URL the user configured. QUMO
  has no default SpeQtrum host.
* Redirects are not followed, so the bearer token never travels to a Location header's host.
* A 401 means the server rejected the request before acting on it. QUMO re-authenticates once
  and retries once, so the retry cannot create a second job.
* Responses are data. They are read with json.loads only, and only after `unsafe_construct`
  has checked them. A YAML tag, anchor, alias, block-style document or second document is
  refused: UnsafeFormatWarning is emitted at that moment and QM-E955 is raised.
* submit() runs check_confirmed() before it builds or sends anything.

SpeQtrum publishes no topology, basis or price for its devices. The caller therefore supplies
each device's TargetSpec as `targets={code: TargetSpec}`, and the cost is reported as unknown
(CostEstimate.unit == "unknown"), never as a quote.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from .base import (
    CANCELLED,
    DONE,
    ERROR,
    QUEUED,
    RUNNING,
    UNKNOWN,
    BackendInfo,
    CostEstimate,
    JobHandle,
    TargetSpec,
    check_confirmed,
    qumo_payload_yaml,
)
from .errors import (
    E_CREDENTIALS,
    E_INVALID_PAYLOAD,
    E_JOB,
    E_LIMIT,
    E_UNKNOWN_BACKEND,
    E_UNKNOWN_JOB,
    HINT_SHOTS,
    HardwareError,
    safe_exc_text,
)

if TYPE_CHECKING:
    from .compile import CompiledCircuit

ENV_USERNAME = "QUMO_SPEQTRUM_USERNAME"
ENV_APIKEY = "QUMO_SPEQTRUM_APIKEY"
ENV_URL = "QUMO_SPEQTRUM_URL"
ENV_AUDIENCE = "QUMO_SPEQTRUM_AUDIENCE"
DEFAULT_AUDIENCE = "urn:qilimanjaro.tech:public-api:beren"
GRANT_TYPE = "urn:ietf:params:oauth:grant-type:jwt-bearer"
TIMEOUT_S = 30.0
LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "::1")
PAYLOAD_TYPE = "digital_propagation"

Transport = Callable[[str, str, Mapping[str, str], bytes | None, float], tuple[int, bytes]]


# --------------------------------------------------------------------------- refusal rule
class UnsafeFormatWarning(UserWarning):
    """A SpeQtrum response carried YAML syntax that QUMO refuses to load (see `unsafe_construct`)."""


_DOC_MARKERS = ("---", "...")


def _token_at(text: str, i: int) -> str:
    j = i
    while j < len(text) and j - i < 48 and text[j] not in " \t\r\n,[]{}\"":
        j += 1
    return text[i:j] or text[i]


def unsafe_construct(text: str) -> str | None:
    """Name the first construct QUMO refuses to load in `text`, or None when it is acceptable.

    QUMO reads one plain JSON document, which is also one YAML 1.2 flow mapping. Refused:
    a tag (any '!' outside a double-quoted string), an anchor ('&') or an alias ('*') outside
    a double-quoted string, block-style YAML (the text must open with '{'), and several
    documents (a '---' or '...' line). A '!', '&' or '*' inside a double-quoted string is data.
    The scan is one linear pass and evaluates nothing.
    """
    if not isinstance(text, str):
        return "non-text content"
    if not text.strip():
        return None  # nothing to refuse; the JSON parser reports the empty document itself
    in_string = False
    escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        at_column0 = i == 0 or text[i - 1] == "\n"
        if at_column0 and text.startswith(_DOC_MARKERS, i) and text[i + 3:i + 4] in ("", " ", "\t", "\r", "\n"):
            return "several YAML documents (a '---' or '...' line)"
        if ch == '"':
            in_string = True
        elif ch == "!":
            return f"a YAML tag `{_token_at(text, i)}`"
        elif ch == "&":
            return f"a YAML anchor `{_token_at(text, i)}`"
        elif ch == "*":
            return f"a YAML alias `{_token_at(text, i)}`"
    if not text.lstrip().startswith("{"):
        return "block-style YAML (the document must be one flow object that starts with '{')"
    return None


_JSON_START = ("{", "[")


def _refuse(construct: str, what: str, job_id: str) -> None:
    """Warn at the moment of loading, then raise QM-E955 naming the construct. Never returns."""
    warnings.warn(f"SpeQtrum {what} of job {job_id} contains {construct}; QUMO did not load it",
                  UnsafeFormatWarning, stacklevel=2)
    raise HardwareError(E_INVALID_PAYLOAD,
                        f"SpeQtrum {what} of job {job_id} was not loaded: it contains {construct}",
                        "QUMO loads only plain JSON results; report the job id to SpeQtrum")


def _not_json(what: str, job_id: str) -> HardwareError:
    return HardwareError(E_INVALID_PAYLOAD, f"SpeQtrum {what} of job {job_id} is not valid JSON",
                         "report the job id; the provider returned a result QUMO cannot read")


def _load_response_text(text: str, job_id: str, what: str) -> Any:
    """json.loads, reached only through the refusal rule. A refusal warns and raises QM-E955."""
    construct = unsafe_construct(text)
    if construct is not None:
        _refuse(construct, what, job_id)
    body = text.lstrip()
    try:
        value, end = json.JSONDecoder().raw_decode(body)
    except (ValueError, RecursionError):
        raise _not_json(what, job_id) from None
    rest = body[end:].lstrip()
    if rest.startswith(_JSON_START):  # a second JSON value with no '---' marker is a second document
        _refuse("several documents (a second JSON value follows the first)", what, job_id)
    if rest:
        raise _not_json(what, job_id)
    return value


def decode_result(encoded: str, job_id: str) -> dict[str, int]:
    """Base64 result envelope -> counts. Each level is parsed with json.loads after the refusal rule."""
    try:
        raw = base64.b64decode(encoded, validate=True).decode("utf-8")
    except (binascii.Error, ValueError, UnicodeDecodeError):
        raise HardwareError(E_INVALID_PAYLOAD, f"SpeQtrum result of job {job_id} is not base64 text",
                            "report the job id to SpeQtrum") from None
    envelope = _load_response_text(raw, job_id, "result")
    if not isinstance(envelope, dict) or not isinstance(envelope.get("functional_result"), str):
        raise HardwareError(E_INVALID_PAYLOAD, f"SpeQtrum result of job {job_id} has no functional_result text",
                            "report the job id to SpeQtrum")
    inner = _load_response_text(envelope["functional_result"], job_id, "functional result")
    counts = inner.get("counts") if isinstance(inner, dict) else None
    if not isinstance(counts, dict) or not counts:
        raise HardwareError(E_INVALID_PAYLOAD, f"functional result of job {job_id} has no counts object",
                            "report the job id to SpeQtrum")
    out: dict[str, int] = {}
    for key, n in counts.items():
        if not (isinstance(key, str) and key and set(key) <= {"0", "1"}) or isinstance(n, bool) \
                or not isinstance(n, int) or n < 0:
            raise HardwareError(E_INVALID_PAYLOAD,
                                f"counts of job {job_id} contain an entry that is not a bitstring with a count",
                                "report the job id to SpeQtrum")
        out[key] = n
    return out


# --------------------------------------------------------------------------- status and auth
_STATUS = {
    "pending": QUEUED,
    "validating": QUEUED,
    "queued": QUEUED,
    "running": RUNNING,
    "completed": DONE,
    "error": ERROR,
    "timeout": ERROR,
    "cancelled": CANCELLED,
}


def map_status(raw: Any) -> str:
    return _STATUS.get(str(raw).strip().lower(), UNKNOWN)


def login_assertion(username: str, api_key: str, audience: str) -> str:
    """base64url(JSON claims). Not a signed JWT: no header and no signature."""
    claims = {"username": username, "api_key": api_key, "audience": audience, "iat": int(time.time())}
    return base64.urlsafe_b64encode(json.dumps(claims, sort_keys=True).encode("utf-8")).rstrip(b"=").decode("ascii")


def _checked_base_url(url: str) -> str:
    """https, or http only for a loopback host (tests). No user information in the URL."""
    text = url.strip()
    try:
        parts = urllib.parse.urlsplit(text)
        _ = parts.port  # raises ValueError for a malformed port
    except ValueError:
        raise HardwareError(E_CREDENTIALS, f"{ENV_URL} is not a valid URL",
                            f"set {ENV_URL} to the https base URL of the SpeQtrum API") from None
    if "@" in parts.netloc:
        raise HardwareError(E_CREDENTIALS, f"{ENV_URL} must not contain user information",
                            f"put the username and API key in {ENV_USERNAME} and {ENV_APIKEY}")
    host = parts.hostname or ""
    if not host or not (parts.scheme == "https" or (parts.scheme == "http" and host in LOOPBACK_HOSTS)):
        raise HardwareError(E_CREDENTIALS,
                            f"{ENV_URL} must be an https URL (plain http is accepted only for localhost, "
                            "127.0.0.1 or [::1], for tests)",
                            f"set {ENV_URL} to the https base URL of your SpeQtrum API; QUMO has no default host")
    return text.rstrip("/")


# --------------------------------------------------------------------------- transport
class _NoRedirects(urllib.request.HTTPRedirectHandler):
    """A redirect could forward the bearer token elsewhere, so it is reported, not followed."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        return None  # urllib then raises HTTPError with the 3xx status


def urllib_transport(method: str, url: str, headers: Mapping[str, str], body: bytes | None,
                     timeout: float) -> tuple[int, bytes]:
    """Default transport: urllib.request, standard library only. Returns (HTTP status, body bytes).

    The opener is built per request so proxy settings are read when the request is made.
    """
    request = urllib.request.Request(url, data=body, headers=dict(headers), method=method)
    opener = urllib.request.build_opener(_NoRedirects)
    try:
        with opener.open(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, exc.read()
        except OSError:
            return exc.code, b""
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise HardwareError(E_JOB, f"cannot reach SpeQtrum ({safe_exc_text(exc)})",
                            "check QUMO_SPEQTRUM_URL, the network and proxy settings, then retry") from None


# --------------------------------------------------------------------------- provider
class SpeQtrumProvider:
    """SpeQtrum digital-propagation backend.

    Options (each defaults to its environment variable):
      username, api_key, url, audience   QUMO_SPEQTRUM_USERNAME / _APIKEY / _URL / _AUDIENCE
      targets      {device code: TargetSpec}, the topology and basis QUMO compiles for
      transport    HTTP function (injected by tests); default: urllib.request, 30 s timeout
    """

    name = "speqtrum"
    payload_format = "yaml"

    def __init__(self, username: str | None = None, api_key: str | None = None, url: str | None = None,
                 audience: str | None = None, targets: Mapping[str, TargetSpec] | None = None,
                 transport: Transport | None = None):
        env = os.environ
        self._username = username or env.get(ENV_USERNAME) or None
        self._api_key = api_key or env.get(ENV_APIKEY) or None
        self._url = url or env.get(ENV_URL) or None
        self._audience = audience or env.get(ENV_AUDIENCE) or DEFAULT_AUDIENCE
        self._targets: dict[str, TargetSpec] = {}
        for code, spec in (targets or {}).items():
            if not isinstance(spec, TargetSpec):
                raise TypeError(f"targets[{code!r}] must be a qumo_hardware.base.TargetSpec")
            self._targets[str(code)] = spec
        self._transport: Transport = transport or urllib_transport
        self._token: str | None = None

    def __repr__(self) -> str:  # never shows the username, the API key or the token
        return f"SpeQtrumProvider(url={self._url!r}, targets={sorted(self._targets)!r})"

    def is_local(self, backend: str) -> bool:
        return False

    # ------------------------------------------------------------ configuration and transport
    def _config(self) -> tuple[str, str, str]:
        """(base URL, username, API key). Checked before any request; names only in messages."""
        missing = [var for var, value in ((ENV_URL, self._url), (ENV_USERNAME, self._username),
                                          (ENV_APIKEY, self._api_key)) if not value]
        if missing:
            raise HardwareError(E_CREDENTIALS, f"SpeQtrum settings missing: {', '.join(missing)}",
                                f"export {', '.join(missing)} (or pass them to SpeQtrumProvider); QUMO has no "
                                "default SpeQtrum host")
        return _checked_base_url(self._url), self._username, self._api_key

    def _secrets(self) -> tuple[str, ...]:
        return tuple(s for s in (self._api_key, self._username, self._token) if s)

    def _clean(self, value: Any) -> str:
        """Server text for a message: secrets redacted, whitespace collapsed, 160 characters at most."""
        if not isinstance(value, str):
            return ""
        for secret in self._secrets():
            value = value.replace(secret, "<redacted>")
        return " ".join(value.split())[:160]

    def _send(self, method: str, url: str, payload: Any, token: str | None) -> tuple[int, bytes]:
        headers = {"Accept": "application/json"}
        body = None
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return self._transport(method, url, headers, body, TIMEOUT_S)

    @staticmethod
    def _object(body: bytes, what: str) -> dict:
        try:
            data = json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            raise HardwareError(E_JOB, f"SpeQtrum {what} returned no JSON", "retry later; report it if it persists") \
                from None
        if not isinstance(data, dict):
            raise HardwareError(E_JOB, f"SpeQtrum {what} returned JSON that is not an object",
                                "retry later; report it if it persists")
        return data

    def _failure(self, status: int, body: bytes, what: str, job_lookup: bool = False) -> HardwareError:
        if 300 <= status < 400:
            return HardwareError(E_JOB, f"SpeQtrum {what} answered with a redirect (HTTP {status}); QUMO does not "
                                 "follow redirects because they could forward the access token",
                                 f"check {ENV_URL} for a moved endpoint")
        if status == 404 and job_lookup:
            return HardwareError(E_UNKNOWN_JOB, "SpeQtrum has no job with this id (HTTP 404)")
        try:
            data = json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            data = {}
        message = ""
        if isinstance(data, dict):
            message = next((self._clean(data[k]) for k in ("message", "error", "detail") if k in data), "")
        suffix = f": {message}" if message else ""
        return HardwareError(E_JOB, f"SpeQtrum {what} failed (HTTP {status}{suffix})",
                             "retry later; check the device status in the SpeQtrum web UI")

    def _login(self) -> None:
        base, username, api_key = self._config()
        payload = {"grantType": GRANT_TYPE, "assertion": login_assertion(username, api_key, self._audience),
                   "scope": "user profile"}
        status, body = self._send("POST", f"{base}/authorisation-tokens", payload, None)
        if status in (401, 403):
            raise HardwareError(E_CREDENTIALS, f"SpeQtrum rejected the login (HTTP {status})",
                                f"check {ENV_USERNAME} and {ENV_APIKEY}, and that the API key has not expired")
        if not 200 <= status < 300:
            raise self._failure(status, body, "login")
        token = self._object(body, "login").get("accessToken")
        if not isinstance(token, str) or not token:
            raise HardwareError(E_CREDENTIALS, "SpeQtrum login response has no access token",
                                "retry; if it persists, check the account in the SpeQtrum web UI")
        self._token = token

    def _call(self, method: str, path: str, payload: Any = None, query: Mapping[str, str] | None = None,
              what: str = "request", job_lookup: bool = False) -> dict:
        base, _, _ = self._config()
        url = base + path + ("?" + urllib.parse.urlencode(query) if query else "")
        if self._token is None:
            self._login()
        status, body = self._send(method, url, payload, self._token)
        if status == 401:  # rejected before acting on the request: re-authenticate once, retry once
            self._token = None
            self._login()
            status, body = self._send(method, url, payload, self._token)
            if status == 401:
                self._token = None
                raise HardwareError(E_CREDENTIALS, "SpeQtrum rejected the access token after re-authentication",
                                    f"check {ENV_USERNAME} and {ENV_APIKEY}")
        if not 200 <= status < 300:
            raise self._failure(status, body, what, job_lookup)
        return self._object(body, what)

    # ------------------------------------------------------------ Provider API
    def list_backends(self, online: bool = False) -> list[BackendInfo]:
        rows: dict[str, BackendInfo] = {
            code: BackendInfo(code, self.name, "hardware", spec.num_qubits, None, None, True,
                              "explicit target (targets=); SpeQtrum publishes no topology")
            for code, spec in self._targets.items()
        }
        if not online:
            return list(rows.values())
        items = self._call("GET", "/devices", what="device list").get("items")
        if not isinstance(items, list):
            raise HardwareError(E_JOB, "SpeQtrum device list has no 'items' array", "retry later")
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("code"), str) or not item["code"]:
                continue
            code = item["code"]
            nq = item.get("nqubits")
            nq = nq if isinstance(nq, int) and not isinstance(nq, bool) else None
            status = item.get("status") if isinstance(item.get("status"), str) else None
            notes = " - ".join(self._clean(item[k]) for k in ("name", "description") if item.get(k)) or "from SpeQtrum"
            row = rows.get(code)
            if row is None:
                rows[code] = BackendInfo(code, self.name, "hardware", nq, status, None, True, notes)
            else:
                row.status = status
                row.num_qubits = row.num_qubits if row.num_qubits is not None else nq
        return list(rows.values())

    def target(self, backend: str) -> TargetSpec:
        spec = self._targets.get(backend)
        if spec is None:
            raise HardwareError(
                E_UNKNOWN_BACKEND, f"no target for SpeQtrum device '{backend}'",
                f"SpeQtrum publishes no topology or basis gates, so pass them once: SpeQtrumProvider(targets="
                f"{{'{backend}': TargetSpec(name='{backend}', provider='speqtrum', num_qubits=N, "
                f"edges=[(0, 1), ...], basis=['rz', 'sx', 'cz'])}}); use all_to_all=True for a fully connected device")
        return replace(spec, name=backend, provider=self.name)

    def prepare(self, compiled: CompiledCircuit, backend: str):
        """CompiledCircuit -> (OpenQASM 2 text, report). The shot count is known only at submit,
        so the qumo.payload.v1 document itself is built by `encode` there."""
        self.target(backend)
        return compiled.qasm, {"payload_format": self.payload_format, "encoded_at": "submit"}

    def encode(self, compiled: CompiledCircuit, shots: int, backend: str) -> str:
        """QASM-to-payload conversion: the qumo.payload.v1 YAML document for this circuit.

        The document is produced by `qumo_payload_yaml` and checked with the same refusal rule
        used for responses, so QUMO never ships a document it would refuse to read.
        """
        document = qumo_payload_yaml(compiled, shots, backend)
        construct = unsafe_construct(document)
        if construct is not None:  # cannot happen for json.dumps output; kept as a guard
            raise HardwareError(E_INVALID_PAYLOAD, f"payload refused before sending: it contains {construct}",
                                "report this circuit with --show-qasm")
        return document

    def validate(self, payload: Any, backend: str) -> dict:
        self.target(backend)
        if not isinstance(payload, str) or not payload.lstrip().startswith("OPENQASM 2"):
            raise HardwareError(E_INVALID_PAYLOAD, "SpeQtrum needs QUMO's compiled OpenQASM 2 text",
                                "pass the output of prepare(), the CompiledCircuit.qasm text")
        return {"checks": ["explicit target for this device (targets=)",
                           "compiled OpenQASM 2 text; encode() wraps it as qumo.payload.v1 YAML at submit"]}

    def estimate_cost(self, compiled: CompiledCircuit, shots: int, backend: str, payload=None) -> CostEstimate:
        return CostEstimate(
            self.name, backend, shots, "unknown", 0.0, "unknown: SpeQtrum publishes no price per shot or device",
            breakdown={"num_qubits": compiled.num_qubits, "two_qubit_gates": compiled.two_qubit_gates,
                       "depth": compiled.depth},
            notes=["SpeQtrum publishes no price for its devices",
                   "the job is billable; --max-cost cannot be enforced for this provider"],
        )

    def submit(self, compiled: CompiledCircuit, shots: int, backend: str, seed: int | None = None,
               payload=None, report=None, max_cost: float | None = None, confirm: bool = False,
               **_: Any) -> JobHandle:
        check_confirmed(self, backend, confirm)  # first: without confirm=True nothing is built or sent
        spec = self.target(backend)
        if shots < 1 or shots > spec.max_shots:
            raise HardwareError(E_LIMIT, f"shots must be in 1..{spec.max_shots} for {backend}", HINT_SHOTS)
        if max_cost is not None:
            raise HardwareError(E_LIMIT, "max_cost cannot be enforced for SpeQtrum: its devices publish no price",
                                "remove the cost cap; the SpeQtrum cost stays unknown for this provider")
        self._config()  # credentials and URL are checked before any request
        body = {"device_code": backend, "payload": self._execute_payload(compiled, shots, backend),
                "job_type": "digital", "meta": {}}
        data = self._call("POST", "/execute", payload=body, what="execute")
        job_id = data.get("id")
        if isinstance(job_id, str) and job_id.isdigit():
            job_id = int(job_id)
        if isinstance(job_id, bool) or not isinstance(job_id, int):
            raise HardwareError(E_JOB, "SpeQtrum accepted the request but returned no job id",
                                "check the SpeQtrum web UI for a new job before resubmitting")
        meta = compiled.result_meta()
        meta.update({"device_code": backend, "payload_format": self.payload_format})
        return JobHandle(str(job_id), self.name, backend, shots, False, QUEUED, meta=meta)

    def _execute_payload(self, compiled: CompiledCircuit, shots: int, backend: str) -> str:
        document = self.encode(compiled, shots, backend)
        readout = json.dumps({"sampling": {"nshots": int(shots)}}, sort_keys=True)
        return json.dumps({"type": PAYLOAD_TYPE,
                           "digital_propagation_payload": {"digital_propagation": document, "readout": readout}},
                          sort_keys=True)

    def _job(self, job: JobHandle) -> dict:
        path = f"/jobs/{urllib.parse.quote(str(job.job_id), safe='')}"
        return self._call("GET", path, query={"payload": "false", "result": "true", "error": "true"},
                          what="job status", job_lookup=True)

    def status(self, job: JobHandle) -> str:
        if job.local:
            return job.status
        return map_status(self._job(job).get("status"))

    def result(self, job: JobHandle) -> dict[str, int]:
        if job.local:
            return dict(job.counts or {})
        data = self._job(job)
        state = map_status(data.get("status"))
        if state != DONE:
            detail = self._clean(data.get("error")) if state in (ERROR, CANCELLED) else ""
            raise HardwareError(E_JOB, f"SpeQtrum job {job.job_id} is {state}" + (f": {detail}" if detail else ""))
        encoded = data.get("result")
        if not isinstance(encoded, str) or not encoded:
            raise HardwareError(E_JOB, f"SpeQtrum job {job.job_id} is DONE but has no result yet",
                                f"retry shortly with `python -m qumo_hardware result {job.job_id} --wait`")
        counts = decode_result(encoded, str(job.job_id))
        width = job.meta.get("num_clbits")
        if isinstance(width, int) and any(len(key) != width for key in counts):
            widths = sorted({len(key) for key in counts})
            raise HardwareError(E_INVALID_PAYLOAD,
                                f"counts of job {job.job_id} have {widths} bits per shot, expected {width}",
                                "the classical width of the circuit and of the result disagree; report the job id")
        return counts
