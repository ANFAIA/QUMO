"""SpeQtrum provider: interface conformance, the YAML payload, the refusal rule, and a local fake server.

The fake server listens on 127.0.0.1 only, and a guard refuses every other outbound connection, so
nothing here can reach the real SpeQtrum service. The compiled circuit is built by hand, so the
tests do not need the qumo_py extension.
"""

from __future__ import annotations

import base64
import json
import socket
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from conftest import BELL

from qumo_hardware.base import (
    CANCELLED,
    DONE,
    ERROR,
    QUEUED,
    RUNNING,
    UNKNOWN,
    JobHandle,
    Provider,
    TargetSpec,
    qumo_payload_yaml,
)
from qumo_hardware.cli import EXIT_OK, main
from qumo_hardware.compile import CompiledCircuit
from qumo_hardware.errors import (
    E_CONFIRM,
    E_CREDENTIALS,
    E_INVALID_PAYLOAD,
    E_JOB,
    E_LIMIT,
    E_UNKNOWN_BACKEND,
    E_UNKNOWN_JOB,
    HardwareError,
)
from qumo_hardware.ibm import IBMProvider
from qumo_hardware.local import LocalEmulatorProvider
from qumo_hardware.quantinuum import QuantinuumProvider
from qumo_hardware.registry import get_provider, provider_names
from qumo_hardware.speqtrum import (
    ENV_APIKEY,
    ENV_AUDIENCE,
    ENV_URL,
    ENV_USERNAME,
    GRANT_TYPE,
    SpeQtrumProvider,
    UnsafeFormatWarning,
    decode_result,
    map_status,
    unsafe_construct,
)

DEVICE = "SPQ-TEST-2"
USERNAME = "qumo-tester"
API_KEY = "spq-key-7f3c2a9e41b8d0c6"  # a fake key; the fake server accepts only this one
AUDIENCE = "urn:qilimanjaro.tech:public-api:beren"
DEVICES = [
    {"code": DEVICE, "nqubits": 2, "name": "Test 2", "description": "two qubits", "type": "digital",
     "status": "online"},
    {"code": "SPQ-OTHER", "nqubits": 5, "name": "Other", "description": "", "type": "digital",
     "status": "maintenance"},
]


def device_target() -> TargetSpec:
    return TargetSpec(name=DEVICE, provider="speqtrum", num_qubits=2, edges=[(0, 1)],
                      basis=["rz", "sx", "cz"], max_shots=5000)


def make_compiled(qasm: str = BELL) -> CompiledCircuit:
    return CompiledCircuit(
        qasm=qasm, source_qasm=qasm, num_qubits=2, num_logical=2, num_clbits=2,
        initial_layout=[0, 1], final_layout=[0, 1], measure_map={0: 0, 1: 1}, logical_measure_map={0: 0, 1: 1},
        auto_measured=False, ops={"h": 1, "cx": 1}, two_qubit_gates=1, depth=3, level=2, seed=1,
        target=device_target(), compile_ms=0.1, qumo_version="test", active_qubits=[0, 1],
    )


def make_provider(**kwargs) -> SpeQtrumProvider:
    kwargs.setdefault("targets", {DEVICE: device_target()})
    return SpeQtrumProvider(**kwargs)


def b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def envelope(functional_result: str) -> str:
    return b64(json.dumps({"type": "digital_propagation", "functional_result": functional_result}))


def recorder(calls: list):
    def transport(method, url, headers, body, timeout):
        calls.append((method, url))
        return 200, b"{}"
    return transport


# --------------------------------------------------------------------------- guards and fixtures
@pytest.fixture(autouse=True)
def no_speqtrum_env(monkeypatch):
    for var in (ENV_USERNAME, ENV_APIKEY, ENV_URL, ENV_AUDIENCE):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def loopback_only(monkeypatch):
    """Any outbound connection other than to the local fake server fails the test."""
    real_connect = socket.socket.connect

    def guarded(self, address):
        host = address[0] if isinstance(address, tuple) else address
        if host != "127.0.0.1":
            raise AssertionError(f"test tried to connect to {host!r}; only the local fake server is allowed")
        return real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded)


class FakeSpeQtrum:
    """State and assertions for the fake server. It logs requests in memory, never to stderr."""

    def __init__(self):
        self.base_url = ""
        self.token: str | None = None
        self.logins = 0
        self.requests: list[tuple[str, str]] = []
        self.job_queries: list[str] = []
        self.log_lines: list[str] = []
        self.executes: list[dict] = []
        self.reject_next = 0  # authenticated requests to answer with 401 (an expired token)
        self.redirect_login_to: str | None = None
        self.execute_http: tuple[int, dict] | None = None
        self.job = {"id": 42, "status": "completed", "result": None, "error": None}


def _handler_for(fake: FakeSpeQtrum):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # the access log goes to memory, so tests can check it
            fake.log_lines.append(fmt % args)

        def _reply(self, status: int, payload=None, headers=None):
            body = b"" if payload is None else json.dumps(payload).encode("utf-8")
            self.send_response(status)
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _login(self, body: dict):
            assertion = str(body.get("assertion", ""))
            try:
                claims = json.loads(base64.urlsafe_b64decode(assertion + "=" * (-len(assertion) % 4)))
            except ValueError:
                return self._reply(400, {"message": "malformed assertion"})
            ok = (body.get("grantType") == GRANT_TYPE and body.get("scope") == "user profile"
                  and claims.get("api_key") == API_KEY and claims.get("username") == USERNAME
                  and claims.get("audience") == AUDIENCE)
            if not ok:
                return self._reply(401, {"message": "invalid credentials"})
            fake.logins += 1
            fake.token = f"token-{fake.logins}"
            return self._reply(200, {"accessToken": fake.token, "refreshToken": "refresh-x",
                                     "expiresIn": 3600, "tokenType": "Bearer"})

        def _dispatch(self, method: str):
            url = urllib.parse.urlsplit(self.path)
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            fake.requests.append((method, self.path))
            if method == "POST" and url.path == "/authorisation-tokens":
                if fake.redirect_login_to:
                    return self._reply(302, {"message": "moved"}, {"Location": fake.redirect_login_to})
                return self._login(json.loads(raw or b"{}"))
            if fake.token is None or self.headers.get("Authorization") != f"Bearer {fake.token}":
                return self._reply(401, {"message": "missing or expired token"})
            if fake.reject_next:
                fake.reject_next -= 1
                fake.token = None
                return self._reply(401, {"message": "token expired"})
            if method == "GET" and url.path == "/devices":
                return self._reply(200, {"items": DEVICES})
            if method == "POST" and url.path == "/execute":
                fake.executes.append(json.loads(raw))
                if fake.execute_http:
                    return self._reply(*fake.execute_http)
                return self._reply(200, {"id": 42, "message": "queued"})
            if method == "GET" and url.path.startswith("/jobs/"):
                fake.job_queries.append(url.query)
                if url.path != "/jobs/42":
                    return self._reply(404, {"message": "no such job"})
                return self._reply(200, fake.job)
            return self._reply(404, {"message": "unknown route"})

        def do_GET(self):
            self._dispatch("GET")

        def do_POST(self):
            self._dispatch("POST")

    return Handler


@pytest.fixture
def fake():
    state = FakeSpeQtrum()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler_for(state))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state.base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def configured(monkeypatch, fake):
    """Credentials and the fake server's URL come from the environment, as they do for users."""
    monkeypatch.setenv(ENV_USERNAME, USERNAME)
    monkeypatch.setenv(ENV_APIKEY, API_KEY)
    monkeypatch.setenv(ENV_URL, fake.base_url)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    return fake


# --------------------------------------------------------------------------- provider protocol
def test_builtin_providers_satisfy_the_interface():
    providers = [IBMProvider(), QuantinuumProvider(), LocalEmulatorProvider(), make_provider()]
    for p in providers:
        assert isinstance(p, Provider)
        assert callable(p.encode)
    assert [p.payload_format for p in providers] == ["openqasm2", "openqasm2", "openqasm2", "yaml"]


def test_openqasm_providers_ship_the_compiled_text_unchanged():
    compiled = make_compiled()
    for p in (IBMProvider(), QuantinuumProvider(), LocalEmulatorProvider()):
        assert p.encode(compiled, 100, "any-backend") == BELL


def test_speqtrum_encode_is_a_flow_document_carrying_the_qasm_exactly():
    compiled = make_compiled()
    text = make_provider().encode(compiled, 123, DEVICE)
    doc = json.loads(text)
    assert doc["format"] == "qumo.payload.v1" and doc["backend"] == DEVICE and doc["shots"] == 123
    assert doc["qasm"] == BELL
    assert doc["num_qubits"] == 2 and doc["num_clbits"] == 2 and doc["auto_measured"] is False
    assert doc["initial_layout"] == [0, 1] and doc["final_layout"] == [0, 1]
    assert doc["measure_map"] == {"0": 0, "1": 1}
    assert text == qumo_payload_yaml(compiled, 123, DEVICE)  # deterministic
    assert unsafe_construct(text) is None


def test_yaml_reader_agrees_with_json_when_pyyaml_is_available():
    yaml = pytest.importorskip("yaml")
    text = make_provider().encode(make_compiled(), 7, DEVICE)
    assert yaml.safe_load(text) == json.loads(text)


def test_registry_lists_and_builds_speqtrum():
    assert "speqtrum" in provider_names()
    prov = get_provider("speqtrum", targets={DEVICE: device_target()}, channel="ignored-by-speqtrum")
    assert isinstance(prov, SpeQtrumProvider) and prov.name == "speqtrum" and prov.payload_format == "yaml"


def test_cli_backends_for_speqtrum_is_offline_and_empty_without_targets(capsys):
    assert main(["backends", "--provider", "speqtrum", "--json"]) == EXIT_OK
    assert json.loads(capsys.readouterr().out)["backends"] == []


# --------------------------------------------------------------------------- refusal rule
@pytest.mark.parametrize("text, expected", [
    ('!!python/object/apply:os.system ["id"]', "a YAML tag"),
    ('{"a": !!str 1}', "a YAML tag"),
    ('{"a": &anchor 1}', "a YAML anchor"),
    ('{"a": *anchor}', "a YAML alias"),
    ("a: 1\nb: 2\n", "block-style"),
    ("- 1\n- 2\n", "block-style"),
    ('{"a": 1}\n---\n{"b": 2}\n', "several YAML documents"),
    ('{"a": 1}\n...\n', "several YAML documents"),
])
def test_refusal_rule_names_the_first_construct(text, expected):
    found = unsafe_construct(text)
    assert found is not None and expected in found


@pytest.mark.parametrize("text", [
    '{"note": "!!python/object/apply:os.system &x *y"}',  # tag-like text inside a quoted string
    '{"qasm": "a \\"!tag\\" &x *y"}',  # escaped quotes inside a string
    '{"counts": {"00": 3, "11": 4}}',
    '{\n  "a": 1\n}',  # multi-line flow JSON
])
def test_refusal_rule_accepts_plain_json_and_quoted_text(text):
    assert unsafe_construct(text) is None
    json.loads(text)  # and it is valid JSON


# --------------------------------------------------------------------------- submission and results
def test_submit_sends_the_documented_request_and_reads_counts_back(configured):
    prov = make_provider()
    job = prov.submit(make_compiled(), 200, DEVICE, confirm=True)
    assert job.job_id == "42" and job.provider == "speqtrum" and job.status == QUEUED
    assert job.meta["num_clbits"] == 2 and job.meta["payload_format"] == "yaml"
    json.dumps(job.to_dict())  # the job store keeps it as JSON, without secrets

    (body,) = configured.executes
    assert set(body) == {"device_code", "payload", "job_type", "meta"}
    assert body["device_code"] == DEVICE and body["job_type"] == "digital" and body["meta"] == {}
    payload = json.loads(body["payload"])
    assert payload["type"] == "digital_propagation"
    inner = payload["digital_propagation_payload"]
    assert set(inner) == {"digital_propagation", "readout"}
    assert json.loads(inner["readout"]) == {"sampling": {"nshots": 200}}
    doc = json.loads(inner["digital_propagation"])
    assert doc["qasm"] == BELL and doc["shots"] == 200 and doc["format"] == "qumo.payload.v1"

    configured.job.update(status="completed",
                          result=envelope(json.dumps({"counts": {"00": 120, "11": 80}})))
    assert prov.status(job) == DONE
    assert prov.result(job) == {"00": 120, "11": 80}
    assert configured.job_queries == ["payload=false&result=true&error=true"] * 2


def test_login_devices_and_list_backends_online(configured):
    rows = {b.name: b for b in make_provider().list_backends(online=True)}
    assert configured.logins == 1
    assert rows[DEVICE].num_qubits == 2 and rows[DEVICE].status == "online" and rows[DEVICE].billable
    assert rows["SPQ-OTHER"].num_qubits == 5 and rows["SPQ-OTHER"].status == "maintenance"


def test_offline_listing_makes_no_request(configured):
    rows = make_provider().list_backends()
    assert [r.name for r in rows] == [DEVICE]
    assert configured.requests == []


def test_api_key_stays_out_of_logs_output_and_errors(configured, capsys):
    configured.execute_http = (500, {"message": f"internal error while handling key {API_KEY} for {USERNAME}"})
    prov = make_provider()
    with pytest.raises(HardwareError) as exc:
        prov.submit(make_compiled(), 10, DEVICE, confirm=True)
    text = str(exc.value) + repr(exc.value.to_dict()) + repr(prov)
    assert exc.value.code == E_JOB and "<redacted>" in str(exc.value)
    for secret in (API_KEY, USERNAME):
        assert secret not in text
        assert all(secret not in line for line in configured.log_lines)
    captured = capsys.readouterr()
    assert API_KEY not in captured.out + captured.err


def test_wrong_api_key_is_a_credentials_error(configured, monkeypatch):
    monkeypatch.setenv(ENV_APIKEY, "spq-key-wrong-0000000000")
    with pytest.raises(HardwareError) as exc:
        make_provider().list_backends(online=True)
    assert exc.value.code == E_CREDENTIALS
    assert "spq-key" not in str(exc.value)


def test_401_once_reauthenticates_and_retries(configured):
    configured.reject_next = 1
    rows = make_provider().list_backends(online=True)
    assert configured.logins == 2
    assert [method for method, _ in configured.requests] == ["POST", "GET", "POST", "GET"]
    assert {r.name for r in rows} == {DEVICE, "SPQ-OTHER"}


def test_401_after_reauthentication_is_a_credentials_error(configured):
    configured.reject_next = 2
    with pytest.raises(HardwareError) as exc:
        make_provider().list_backends(online=True)
    assert exc.value.code == E_CREDENTIALS and configured.logins == 2


@pytest.mark.parametrize("shape", ["envelope", "bare"])
def test_tagged_result_warns_and_is_not_loaded(configured, tmp_path, shape):
    marker = tmp_path / "pwned"
    tagged = f'!!python/object/apply:os.system ["touch {marker}"]'
    configured.job.update(status="completed", result=envelope(tagged) if shape == "envelope" else b64(tagged))
    job = JobHandle("42", "speqtrum", DEVICE, 100, False, DONE, meta={"num_clbits": 2})
    with pytest.warns(UnsafeFormatWarning, match="YAML tag") as record:
        with pytest.raises(HardwareError) as exc:
            make_provider().result(job)
    assert exc.value.code == E_INVALID_PAYLOAD and "was not loaded" in exc.value.message
    assert any(w.category is UnsafeFormatWarning for w in record)
    assert not marker.exists()


def test_result_is_checked_against_the_classical_width(configured):
    configured.job.update(status="completed", result=envelope(json.dumps({"counts": {"0": 5}})))
    job = JobHandle("42", "speqtrum", DEVICE, 5, False, DONE, meta={"num_clbits": 2})
    with pytest.raises(HardwareError) as exc:
        make_provider().result(job)
    assert exc.value.code == E_INVALID_PAYLOAD


@pytest.mark.parametrize("inner", ['{"counts": {"0x": 1}}', '{"counts": {"00": -1}}', '{"counts": ',
                                   '{"other": 1}', '{"counts": {"0": 1}} junk'])
def test_malformed_counts_are_invalid_payload_not_loaded(inner):
    with pytest.raises(HardwareError) as exc:
        decode_result(envelope(inner), "42")
    assert exc.value.code == E_INVALID_PAYLOAD


def test_second_json_value_without_marker_is_refused_with_warning():
    with pytest.warns(UnsafeFormatWarning, match="several documents"):
        with pytest.raises(HardwareError) as exc:
            decode_result(envelope('{"counts": {"0": 1}} {"counts": {"1": 2}}'), "42")
    assert exc.value.code == E_INVALID_PAYLOAD and "was not loaded" in exc.value.message


def test_submit_without_confirm_sends_nothing(configured):
    prov = make_provider()
    with pytest.raises(HardwareError) as exc:
        prov.submit(make_compiled(), 10, DEVICE)
    assert exc.value.code == E_CONFIRM
    with pytest.raises(HardwareError) as exc:
        prov.submit(make_compiled(), 10, DEVICE, confirm=False)
    assert exc.value.code == E_CONFIRM
    assert configured.requests == []


@pytest.mark.parametrize("missing", [ENV_USERNAME, ENV_APIKEY, ENV_URL])
def test_missing_setting_is_a_credentials_error_before_any_request(monkeypatch, configured, missing):
    monkeypatch.delenv(missing)
    with pytest.raises(HardwareError) as exc:
        make_provider().submit(make_compiled(), 10, DEVICE, confirm=True)
    assert exc.value.code == E_CREDENTIALS and missing in exc.value.message
    assert configured.requests == []


def test_no_default_url_is_used(monkeypatch):
    monkeypatch.setenv(ENV_USERNAME, USERNAME)
    monkeypatch.setenv(ENV_APIKEY, API_KEY)
    calls: list = []
    prov = make_provider(transport=recorder(calls))
    with pytest.raises(HardwareError) as exc:
        prov.submit(make_compiled(), 10, DEVICE, confirm=True)
    assert exc.value.code == E_CREDENTIALS and ENV_URL in exc.value.message
    assert calls == []


@pytest.mark.parametrize("url", [
    "http://speqtrum.example.com/api",
    "https://user:not-a-real-pw@speqtrum.example.com",
    "ftp://speqtrum.example.com",
    "http://10.0.0.5:8080",
    "https://",
])
def test_url_must_be_https_or_loopback(url):
    calls: list = []
    prov = SpeQtrumProvider(username=USERNAME, api_key=API_KEY, url=url, targets={DEVICE: device_target()},
                            transport=recorder(calls))
    with pytest.raises(HardwareError) as exc:
        prov.submit(make_compiled(), 10, DEVICE, confirm=True)
    assert exc.value.code == E_CREDENTIALS and calls == []
    assert "not-a-real-pw" not in str(exc.value) and API_KEY not in str(exc.value)


def test_device_without_target_is_unknown_backend_with_hint(configured):
    with pytest.raises(HardwareError) as exc:
        SpeQtrumProvider().submit(make_compiled(), 10, "SPQ-UNCONFIGURED", confirm=True)
    assert exc.value.code == E_UNKNOWN_BACKEND and "targets=" in exc.value.hint
    assert configured.requests == []


def test_max_cost_cannot_be_enforced_so_it_is_refused(configured):
    with pytest.raises(HardwareError) as exc:
        make_provider().submit(make_compiled(), 10, DEVICE, max_cost=5.0, confirm=True)
    assert exc.value.code == E_LIMIT
    assert configured.requests == []


def test_shot_limit_is_checked_before_sending(configured):
    with pytest.raises(HardwareError) as exc:
        make_provider().submit(make_compiled(), 6000, DEVICE, confirm=True)
    assert exc.value.code == E_LIMIT and configured.requests == []


def test_cost_is_unknown_but_billable():
    est = make_provider().estimate_cost(make_compiled(), 100, DEVICE)
    assert est.unit == "unknown" and est.billable and est.amount == 0.0
    assert est.to_dict()["billable"] is True
    assert est.summary().startswith("unknown, billable")


def test_redirect_is_reported_and_not_followed(configured):
    configured.redirect_login_to = configured.base_url + "/devices"
    with pytest.raises(HardwareError) as exc:
        make_provider().list_backends(online=True)
    assert exc.value.code == E_JOB and "redirect" in exc.value.message
    assert configured.requests == [("POST", "/authorisation-tokens")]


def test_unknown_job_id_is_reported_as_unknown_job(configured):
    with pytest.raises(HardwareError) as exc:
        make_provider().status(JobHandle("999", "speqtrum", DEVICE, 10, False))
    assert exc.value.code == E_UNKNOWN_JOB


def test_non_json_result_is_invalid_payload_not_a_refusal(configured):
    configured.job.update(status="completed", result=b64('{"counts": '))
    job = JobHandle("42", "speqtrum", DEVICE, 10, False, DONE, meta={"num_clbits": 2})
    with pytest.raises(HardwareError) as exc:
        make_provider().result(job)
    assert exc.value.code == E_INVALID_PAYLOAD and "not valid JSON" in exc.value.message


@pytest.mark.parametrize("raw, expected", [
    ("pending", QUEUED), ("validating", QUEUED), ("queued", QUEUED), ("running", RUNNING),
    ("completed", DONE), ("error", ERROR), ("timeout", ERROR), ("cancelled", CANCELLED),
    ("exploded", UNKNOWN), (None, UNKNOWN),
])
def test_status_mapping(raw, expected):
    assert map_status(raw) == expected


def test_status_over_http_maps_the_provider_state(configured):
    configured.job["status"] = "running"
    job = JobHandle("42", "speqtrum", DEVICE, 10, False, QUEUED, meta={"num_clbits": 2})
    assert make_provider().status(job) == RUNNING


def test_repr_never_shows_credentials():
    prov = SpeQtrumProvider(username=USERNAME, api_key=API_KEY, url="https://speqtrum.example.com")
    assert API_KEY not in repr(prov) and USERNAME not in repr(prov)


def test_cli_targets_file_reaches_the_speqtrum_provider(tmp_path):
    """--targets FILE loads JSON target specs, so `-p speqtrum` works from the CLI."""
    import argparse

    from qumo_hardware.cli import _load_targets, _provider_options
    from qumo_hardware.errors import E_INVALID_PAYLOAD, HardwareError

    spec = {"name": "dev_line", "provider": "speqtrum", "num_qubits": 3,
            "basis": ["rz", "sx", "x", "cz"], "edges": [[0, 1], [1, 2]]}
    f = tmp_path / "targets.json"
    f.write_text(json.dumps({"dev_line": spec}), encoding="utf-8")
    loaded = _load_targets(str(f))
    assert loaded["dev_line"].num_qubits == 3
    opts = _provider_options(argparse.Namespace(targets=str(f)))
    assert set(opts["targets"]) == {"dev_line"}
    assert _provider_options(argparse.Namespace())["targets"] is None
    bad = tmp_path / "bad.json"
    bad.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(HardwareError) as e:
        _load_targets(str(bad))
    assert e.value.code == E_INVALID_PAYLOAD
