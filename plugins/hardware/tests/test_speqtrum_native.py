"""SpeQtrum native client (qumo/exec/speqtrum_api.mojo): protocol parity with the Python fallback.

The server is tests/speqtrum_fake_server.py on 127.0.0.1 in its own process. Documents the
native client sends are compared byte for byte with what the Python fallback would send.
"""

from __future__ import annotations

import json
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
from test_speqtrum import (
    API_KEY,
    AUDIENCE,
    DEVICE,
    DEVICES,
    USERNAME,
    b64,
    device_target,
    envelope,
    make_compiled,
)

from qumo_hardware.base import CANCELLED, DONE, ERROR, QUEUED, qumo_payload_yaml
from qumo_hardware.compile import load_qumo_py
from qumo_hardware.errors import (
    E_CONFIRM,
    E_CREDENTIALS,
    E_INVALID_PAYLOAD,
    E_JOB,
    E_UNKNOWN_JOB,
    HardwareError,
)
from qumo_hardware.registry import get_provider
from qumo_hardware.speqtrum import (
    ENV_APIKEY,
    ENV_URL,
    ENV_USERNAME,
    SpeQtrumProvider,
    UnsafeFormatWarning,
    decode_result,
    unsafe_construct,
)
from qumo_hardware.speqtrum_native import SpeQtrumNativeProvider

SERVER = Path(__file__).with_name("speqtrum_fake_server.py")


@pytest.fixture(scope="module")
def qumo():
    return load_qumo_py()


class Fake:
    def __init__(self, tmp: Path):
        self.log, self.state = tmp / "log.jsonl", tmp / "state.json"
        self.log.write_text("")
        self.put({"credentials": {"username": USERNAME, "api_key": API_KEY, "audience": AUDIENCE},
                  "devices": DEVICES, "job": {"id": 42, "status": "completed", "result": None, "error": None},
                  "logins": 0})
        self.proc = subprocess.Popen([sys.executable, "-I", str(SERVER), str(self.log), str(self.state)],
                                     stdout=subprocess.PIPE, text=True)
        self.base_url = f"http://127.0.0.1:{int(self.proc.stdout.readline())}"

    def get(self) -> dict:
        return json.loads(self.state.read_text())

    def put(self, state: dict) -> None:
        self.state.write_text(json.dumps(state))

    def update(self, **kw) -> None:
        st = self.get()
        st.update(kw)
        self.put(st)

    def requests(self) -> list[dict]:
        return [json.loads(x) for x in self.log.read_text().splitlines() if x.strip()]

    def stop(self) -> None:
        self.proc.kill()
        self.proc.wait()


@pytest.fixture
def fake(tmp_path, monkeypatch, qumo):
    f = Fake(tmp_path)
    monkeypatch.setenv(ENV_USERNAME, USERNAME)
    monkeypatch.setenv(ENV_APIKEY, API_KEY)
    monkeypatch.setenv(ENV_URL, f.base_url)
    yield f
    f.stop()


def native(**kw) -> SpeQtrumNativeProvider:
    kw.setdefault("targets", {DEVICE: device_target()})
    return SpeQtrumNativeProvider(**kw)


# ----------------------------------------------------------------- selection
def test_native_is_the_default_client(monkeypatch):
    monkeypatch.delenv("QUMO_HW_CLIENT", raising=False)
    monkeypatch.delenv("QUMO_HW_CLIENT_SPEQTRUM", raising=False)
    assert type(get_provider("speqtrum")) is SpeQtrumNativeProvider
    assert type(get_provider("speqtrum", client="python")) is SpeQtrumProvider


# ----------------------------------------------------------------- byte parity (no network)
@pytest.mark.parametrize("qasm", [
    make_compiled().qasm,
    'OPENQASM 2.0;\ninclude "qelib1.inc";\n// café \U0001F600 "quoted" \\ tab\t\nqreg q[2];\n',
])
def test_payload_documents_are_byte_identical_to_the_python_fallback(qumo, qasm):
    compiled = make_compiled(qasm)
    compiled.measure_map = {0: 1, 1: 0, 10: 0, 2: 1}
    compiled.num_clbits = 11
    assert native().encode(compiled, 321, DEVICE) == qumo_payload_yaml(compiled, 321, DEVICE)
    assert native().encode(compiled, 321, DEVICE) == SpeQtrumProvider(targets={DEVICE: device_target()}).encode(
        compiled, 321, DEVICE)


@pytest.mark.parametrize("text", [
    '{"a": 1}', '{"a": "!&*"}', '{"a": !tag 1}', '{"a": &x 1}', '{"a": *x}', "a: 1", "---\n{}", "{}\n...\n",
    "", "   ", '{"a": "esc \\" ! &"}', "[1]",
])
def test_refusal_rule_matches_the_python_fallback(qumo, text):
    assert str(qumo.speqtrum_unsafe_construct(text)) == (unsafe_construct(text) or "")


@pytest.mark.parametrize("encoded", [
    envelope(json.dumps({"counts": {"00": 3, "11": 5}})),
    envelope(json.dumps({"counts": {"0": 1}})),
    envelope('{"counts": &a {"00": 1}}'),
    envelope('{"counts": {"00": 1}}\n{"counts": {}}'),
    envelope(json.dumps({"counts": {"02": 1}})),
    envelope(json.dumps({"counts": {"00": -1}})),
    envelope(json.dumps({"counts": {"00": 1.5}})),
    envelope(json.dumps({"counts": {}})),
    b64(json.dumps({"no": "functional_result"})),
    "not base64!",
    b64("---\n{}"),
])
def test_result_decoding_matches_the_python_fallback(qumo, encoded):
    try:
        expected = decode_result(encoded, "7")
    except HardwareError as exc:
        expected = exc.code
    try:
        got = {str(k): int(v) for k, v in dict(qumo.speqtrum_decode_result(encoded, "7")).items()}
    except Exception as exc:
        got = str(exc).split(":", 1)[0]
    assert got == expected


# ----------------------------------------------------------------- protocol against the fake server
def test_online_device_list(fake):
    rows = {r.name: r for r in native().list_backends(online=True)}
    assert rows[DEVICE].status == "online" and rows["SPQ-OTHER"].num_qubits == 5
    reqs = fake.requests()
    assert [(r["method"], r["path"]) for r in reqs] == [("POST", "/authorisation-tokens"), ("GET", "/devices")]
    assert reqs[1]["authorization"] == "Bearer token-1"
    assert API_KEY not in reqs[1]["path"]


def test_submit_requires_confirmation_and_sends_nothing(fake):
    with pytest.raises(HardwareError) as exc:
        native().submit(make_compiled(), 100, DEVICE)
    assert exc.value.code == E_CONFIRM and fake.requests() == []


def test_execute_body_is_the_python_fallback_body(fake):
    compiled = make_compiled()
    job = native().submit(compiled, 100, DEVICE, confirm=True)
    assert job.job_id == "42" and job.status == QUEUED
    post = [r for r in fake.requests() if r["path"] == "/execute"][0]
    body = json.loads(post["body"])
    expected = SpeQtrumProvider(targets={DEVICE: device_target()})._execute_payload(compiled, 100, DEVICE)
    assert body == {"device_code": DEVICE, "payload": expected, "job_type": "digital", "meta": {}}
    assert API_KEY not in json.dumps(job.to_dict(), default=str)


def test_result_status_and_failure_detail(fake):
    prov = native()
    job = prov.submit(make_compiled(), 100, DEVICE, confirm=True)
    fake.update(job={"id": 42, "status": "completed", "result": envelope(json.dumps({"counts": {"00": 60, "11": 40}})),
                     "error": None})
    assert prov.status(job) == DONE
    assert prov.result(job) == {"00": 60, "11": 40}
    fake.update(job={"id": 42, "status": "error", "result": None, "error": f"failed for key {API_KEY}"})
    assert prov.status(job) == ERROR
    with pytest.raises(HardwareError) as exc:
        prov.result(job)
    assert exc.value.code == E_JOB and API_KEY not in str(exc.value) and "<redacted>" in str(exc.value)
    fake.update(job={"id": 42, "status": "cancelled"})
    assert prov.status(job) == CANCELLED


def test_expired_token_is_renewed_once(fake):
    prov = native()
    job = prov.submit(make_compiled(), 10, DEVICE, confirm=True)
    fake.update(reject_next=1, job={"id": 42, "status": "queued"})
    assert prov.status(job) == QUEUED
    logins = [r for r in fake.requests() if r["path"] == "/authorisation-tokens"]
    assert len(logins) == 3  # submit: 1; status: 1, then 1 more after the 401
    fake.update(reject_next=5)
    with pytest.raises(HardwareError) as exc:
        prov.status(job)
    assert exc.value.code == E_CREDENTIALS


def test_redirect_is_not_followed(fake):
    fake.update(redirect_login_to="http://127.0.0.1:9/elsewhere")
    with pytest.raises(HardwareError) as exc:
        native().list_backends(online=True)
    assert exc.value.code == E_JOB and "redirect" in str(exc.value)
    assert len(fake.requests()) == 1


def test_rejected_login(fake, monkeypatch):
    monkeypatch.setenv(ENV_APIKEY, "wrong-key")
    with pytest.raises(HardwareError) as exc:
        native().list_backends(online=True)
    assert exc.value.code == E_CREDENTIALS and "wrong-key" not in str(exc.value)


def test_unknown_job_and_bad_job_id(fake):
    from qumo_hardware.base import JobHandle

    prov = native()
    with pytest.raises(HardwareError) as exc:
        prov.status(JobHandle("43", "speqtrum", DEVICE, 10, False, QUEUED))
    assert exc.value.code == E_UNKNOWN_JOB
    with pytest.raises(HardwareError) as exc:
        prov.status(JobHandle("../devices", "speqtrum", DEVICE, 10, False, QUEUED))
    assert exc.value.code == E_UNKNOWN_JOB


def test_refused_result_warns_and_raises(fake):
    prov = native()
    job = prov.submit(make_compiled(), 10, DEVICE, confirm=True)
    fake.update(job={"id": 42, "status": "completed", "result": envelope('{"counts": !!python/object {"00": 1}}')})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(HardwareError) as exc:
            prov.result(job)
    assert exc.value.code == E_INVALID_PAYLOAD and "YAML tag" in str(exc.value)
    assert any(issubclass(w.category, UnsafeFormatWarning) for w in caught)


@pytest.mark.parametrize("url", ["http://speqtrum.example.com", "https://user:pw@speqtrum.example.com", "ftp://x"])
def test_unsafe_urls_are_refused_before_any_request(fake, monkeypatch, url):
    monkeypatch.setenv(ENV_URL, url)
    with pytest.raises(HardwareError) as exc:
        native().list_backends(online=True)
    assert exc.value.code == E_CREDENTIALS
    assert fake.requests() == []


# ----------------------------------------------------------------- review regressions
def test_del_and_non_ascii_payload_bytes_match_python(qumo):
    compiled = make_compiled('OPENQASM 2.0;\n// del \x7f, café, \U0001F600\nqreg q[2];\n')
    assert native().encode(compiled, 3, DEVICE) == qumo_payload_yaml(compiled, 3, DEVICE)
    assert str(qumo.speqtrum_payload_document("a\x7fb", 1, "", 0, 0, [], [], [], [], False)) == \
        json.dumps({"auto_measured": False, "backend": "a\x7fb", "final_layout": [], "format": "qumo.payload.v1",
                    "initial_layout": [], "measure_map": {}, "num_clbits": 0, "num_qubits": 0, "qasm": "", "shots": 1},
                   sort_keys=True)


@pytest.mark.parametrize("raw", [b"\xf0\x9f\x98", b"\xc0\xaf", b"\xed\xa0\x80", b'{"x": "\xff"}'])
def test_invalid_utf8_results_are_refused_like_python(qumo, raw):
    import base64

    encoded = base64.b64encode(raw).decode()
    with pytest.raises(HardwareError) as py_exc:
        decode_result(encoded, "1")
    with pytest.raises(Exception) as native_exc:
        qumo.speqtrum_decode_result(encoded, "1")
    assert str(native_exc.value).startswith(py_exc.value.code)


def test_the_access_token_is_redacted_from_job_errors(fake):
    prov = native()
    job = prov.submit(make_compiled(), 10, DEVICE, confirm=True)
    fake.update(job={"id": 42, "status": "error", "error": "token token-2 rejected"})
    with pytest.raises(HardwareError) as exc:
        prov.result(job)
    assert "token-2" not in str(exc.value) and "<redacted>" in str(exc.value)
