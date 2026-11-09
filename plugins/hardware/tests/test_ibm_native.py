"""IBM native client (qumo/exec/ibm_api.mojo) against a fake IAM + Runtime API on 127.0.0.1.

No qiskit is used on this path. The fake backend is a 5-qubit Heron-like line (cz, rz, sx, x).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import BELL, have

from qumo_hardware import runner
from qumo_hardware.base import CANCELLED, DONE, ERROR, QUEUED, RUNNING
from qumo_hardware.compile import compile_for_target, load_qumo_py
from qumo_hardware.errors import (
    E_CONFIRM,
    E_CREDENTIALS,
    E_INVALID_PAYLOAD,
    E_JOB,
    E_UNKNOWN_BACKEND,
    E_UNKNOWN_JOB,
    HardwareError,
)
from qumo_hardware.ibm_native import IBMNativeProvider
from qumo_hardware.registry import get_provider

SERVER = Path(__file__).with_name("ibm_fake_server.py")
BACKEND = "ibm_testline_5"
KEY = "ibm-test-key-not-real-42"
CRN = "crn:v1:bluemix:public:quantum-computing:us-east:a/acct:inst::"

LINE = [(0, 1), (1, 2), (2, 3), (3, 4)]


def configuration(directed: bool = False) -> dict:
    cmap = [[a, b] for a, b in LINE] + ([] if directed else [[b, a] for a, b in LINE])
    return {"backend_name": BACKEND, "n_qubits": 5, "basis_gates": ["cz", "id", "rz", "sx", "x", "measure", "delay"],
            "coupling_map": cmap, "max_shots": 100000, "simulator": False,
            "processor_type": {"family": "Heron", "revision": 2}, "dt": 4.0, "default_rep_delay": 250.0}


def properties(directed: bool = False) -> dict:
    gates = []
    for a, b in LINE:
        for qa, qb in ([(a, b)] if directed else [(a, b), (b, a)]):
            gates.append({"gate": "cz", "qubits": [qa, qb],
                          "parameters": [{"name": "gate_error", "value": 0.002 + 0.001 * a, "unit": ""},
                                         {"name": "gate_length", "value": 68, "unit": "ns"}]})
    for q in range(5):
        gates.append({"gate": "sx", "qubits": [q], "parameters": [{"name": "gate_error", "value": 0.0004, "unit": ""},
                                                                  {"name": "gate_length", "value": 36, "unit": "ns"}]})
    qubits = [[{"name": "T1", "value": 100, "unit": "us"}, {"name": "readout_error", "value": 0.01 * (q + 1), "unit": ""},
               {"name": "readout_length", "value": 2684, "unit": "ns"}] for q in range(5)]
    return {"backend_name": BACKEND, "gates": gates, "qubits": qubits}


@pytest.fixture(scope="module")
def qumo():
    return load_qumo_py()


class Fake:
    def __init__(self, tmp: Path):
        self.log, self.state = tmp / "log.jsonl", tmp / "state.json"
        self.log.write_text("")
        self.put({"api_key": KEY, "configuration": configuration(), "properties": properties(),
                  "backends": [{"name": BACKEND, "status": {"name": "online"}, "qubits": 5, "queue_length": 3}],
                  "samples": ["0x0", "0x3", "0x3", "0x0", "0x3", "0x1"], "require_user_agent": True})
        self.proc = subprocess.Popen([sys.executable, "-I", str(SERVER), str(self.log), str(self.state)],
                                     stdout=subprocess.PIPE, text=True)
        self.base = f"http://127.0.0.1:{int(self.proc.stdout.readline())}"

    def put(self, st: dict) -> None:
        self.state.write_text(json.dumps(st))

    def update(self, **kw) -> None:
        st = json.loads(self.state.read_text())
        st.update(kw)
        self.put(st)

    def requests(self, path_prefix: str = "") -> list[dict]:
        rows = [json.loads(x) for x in self.log.read_text().splitlines() if x.strip()]
        return [r for r in rows if r["path"].startswith(path_prefix)]

    def stop(self) -> None:
        self.proc.kill()
        self.proc.wait()


@pytest.fixture
def ibm(tmp_path, monkeypatch, qumo):
    f = Fake(tmp_path)
    monkeypatch.setenv("IAM_URL", f.base)
    monkeypatch.setenv("QUMO_IBM_API_URL", f.base + "/api/v1")
    monkeypatch.setenv("QISKIT_IBM_TOKEN", KEY)
    monkeypatch.setenv("QISKIT_IBM_INSTANCE", CRN)
    monkeypatch.delenv("QUMO_HW_CLIENT", raising=False)
    monkeypatch.delenv("QUMO_HW_CLIENT_IBM", raising=False)
    yield f
    f.stop()


@pytest.fixture
def compiled(ibm):
    return compile_for_target(BELL, IBMNativeProvider().target(BACKEND), level=2, seed=1)


# ----------------------------------------------------------------- selection and independence
def test_native_is_the_default_client(ibm):
    assert type(get_provider("ibm")) is IBMNativeProvider
    assert type(get_provider("ibm", client="python")).__name__ == "IBMProvider"


def test_native_module_imports_no_qiskit():
    src = (Path(__file__).resolve().parents[1] / "qumo_hardware" / "ibm_native.py").read_text()
    assert "import qiskit" not in src and "from qiskit" not in src


# ----------------------------------------------------------------- target
def test_target_from_configuration_and_properties(ibm):
    spec = IBMNativeProvider().target(BACKEND)
    assert spec.num_qubits == 5 and spec.basis == ["rz", "sx", "x", "cz"]
    assert spec.edges == LINE and spec.directed_edges is None
    assert spec.errors["1q"] == pytest.approx(0.0004) and spec.errors["readout"] == pytest.approx(0.03)
    assert spec.durations == pytest.approx({"1q": 36e-9, "2q": 68e-9, "readout": 2.684e-6})
    assert spec.readout_errors == pytest.approx([0.01, 0.02, 0.03, 0.04, 0.05])
    assert spec.edge_errors["0-1"] == pytest.approx(0.002)
    assert spec.extras["dt"] == pytest.approx(4e-9) and spec.extras["default_rep_delay"] == pytest.approx(250e-6)
    assert spec.extras["processor_type"] == "Heron r2" and spec.extras["native_2q"] == "cz"
    for r in ibm.requests("/api/v1"):
        h = r["headers"]
        assert h["Authorization"] == "Bearer iam-token-1" and h["Service-CRN"] == CRN
        assert h["IBM-API-Version"] and h["User-Agent"].startswith("qumo-hardware")
    (iam,) = ibm.requests("/identity/token")
    assert "apikey=" + KEY in iam["body"] and iam["headers"]["Content-Type"] == "application/x-www-form-urlencoded"


def test_directed_coupling_map_is_reported(ibm):
    ibm.update(configuration=configuration(directed=True), properties=properties(directed=True))
    spec = IBMNativeProvider().target(BACKEND)
    assert spec.directed_edges == LINE


def test_unknown_backend(ibm):
    with pytest.raises(HardwareError) as exc:
        IBMNativeProvider().target("ibm_nowhere")
    assert exc.value.code == E_UNKNOWN_BACKEND
    with pytest.raises(HardwareError) as exc:
        IBMNativeProvider().target("../jobs")
    assert exc.value.code == E_UNKNOWN_BACKEND


def test_online_listing(ibm):
    rows = [r for r in IBMNativeProvider().list_backends(online=True) if r.kind == "hardware"]
    assert [(r.name, r.num_qubits, r.status, r.pending_jobs) for r in rows] == [(BACKEND, 5, "online", 3)]


# ----------------------------------------------------------------- ISA payload
def test_isa_payload_is_openqasm3_on_physical_qubits(ibm, compiled):
    payload, report = IBMNativeProvider().prepare(compiled, BACKEND)
    lines = payload.splitlines()
    assert lines[:3] == ["OPENQASM 3.0;", 'include "stdgates.inc";', "bit[2] c;"]
    assert any(line.startswith("cz $") for line in lines)
    assert all("q[" not in line for line in lines[3:])
    assert sum(line.endswith(";") and "measure" in line for line in lines) == 2
    assert report["two_qubit_gates"] == 1 and report["duration_s"] > 0


@pytest.mark.parametrize("body, what", [
    ("cz q[0],q[2];\nmeasure q[0] -> c[0];\n", "not a native coupling"),
    ("h q[0];\nmeasure q[0] -> c[0];\n", "not a basis gate"),
    ("measure q[0] -> c[0];\nif(c==1) x q[1];\nmeasure q[1] -> c[1];\n", "classically conditioned"),
    ("ecr q[0],q[1];\nmeasure q[0] -> c[0];\n", "stdgates.inc"),
    ("x q[0];\n", "no measurements"),
])
def test_isa_check_refuses(ibm, qumo, body, what):
    src = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[5];\ncreg c[2];\n' + body
    with pytest.raises(Exception) as exc:
        qumo.ibm_isa_payload(src, 5, ["rz", "sx", "x", "cz"], [0, 1, 1, 2, 2, 3, 3, 4], True, 3.6e-8, 6.8e-8, 2.7e-6)
    assert "QM-E955" in str(exc.value) and what in str(exc.value)


def test_direction_is_enforced_for_a_directed_backend(qumo):
    src = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\ncz q[1],q[0];\nmeasure q[0] -> c[0];\n'
    with pytest.raises(Exception, match="native direction"):
        qumo.ibm_isa_payload(src, 2, ["rz", "sx", "x", "cz"], [0, 1], False, -1.0, -1.0, -1.0)
    ok = qumo.ibm_isa_payload(src.replace("cz q[1],q[0]", "cz q[0],q[1]"), 2, ["rz", "sx", "x", "cz"], [0, 1], False,
                              -1.0, -1.0, -1.0)
    assert "cz $0, $1;" in ok[0]


# ----------------------------------------------------------------- submission and results
def test_submit_requires_confirmation(ibm, compiled):
    with pytest.raises(HardwareError) as exc:
        IBMNativeProvider().submit(compiled, 100, BACKEND)
    assert exc.value.code == E_CONFIRM
    assert ibm.requests("/api/v1/jobs") == []


def test_submit_sends_one_sampler_pub_with_the_cost_cap(ibm, compiled):
    prov = IBMNativeProvider()
    job = prov.submit(compiled, 1000, BACKEND, confirm=True, max_cost=5)
    assert job.job_id == "d3fakejob0001" and job.status == QUEUED and job.meta["max_execution_time"] == 5
    (post,) = [r for r in ibm.requests("/api/v1/jobs") if r["method"] == "POST"]
    body = json.loads(post["body"])
    assert body["program_id"] == "sampler" and body["backend"] == BACKEND and body["cost"] == 5
    (pub,) = body["params"]["pubs"]
    assert pub[0].startswith("OPENQASM 3.0;") and pub[1] is None and pub[2] == 1000
    assert body["params"]["options"]["twirling"] == {"enable_gates": False, "enable_measure": False}
    assert body["params"]["options"]["dynamical_decoupling"] == {"enable": False}
    assert KEY not in json.dumps(job.to_dict(), default=str) and "iam-token" not in json.dumps(job.to_dict(), default=str)


def test_default_cap_comes_from_the_estimate(ibm, compiled):
    prov = IBMNativeProvider()
    est = prov.estimate_cost(compiled, 1000, BACKEND)
    assert est.unit == "QPU-seconds" and est.amount > 2.0
    job = prov.submit(compiled, 1000, BACKEND, confirm=True)
    assert job.meta["max_execution_time"] == max(30, -(-int(2 * est.amount * 1000) // 1000))


@pytest.mark.parametrize("api, expected", [
    ("Queued", QUEUED), ("Running", RUNNING), ("Completed", DONE), ("Failed", ERROR), ("Cancelled", CANCELLED),
    ("Cancelled - Ran too long", CANCELLED),
])
def test_status_mapping(ibm, compiled, api, expected):
    job = IBMNativeProvider().submit(compiled, 10, BACKEND, confirm=True)
    ibm.update(job_state={"status": api})
    assert IBMNativeProvider().status(job) == expected


def test_end_to_end_run_and_counts(ibm):
    out = runner.run(BELL, get_provider("ibm"), BACKEND, shots=6, confirm=True, wait=False)
    assert out["submitted"] and out["job_id"] == "d3fakejob0001"
    res = runner.job_result("d3fakejob0001")
    assert res["counts"] == {"00": 2, "11": 3, "01": 1}


def test_failed_job_reports_the_reason(ibm, compiled):
    job = IBMNativeProvider().submit(compiled, 10, BACKEND, confirm=True)
    ibm.update(job_state={"status": "Failed", "reason": "Instruction not supported", "reasonCode": 1517})
    with pytest.raises(HardwareError) as exc:
        IBMNativeProvider().result(job)
    assert exc.value.code == E_JOB and "Error code 1517; Instruction not supported" in str(exc.value)


def test_samples_wider_than_the_register_are_refused(ibm, compiled):
    job = IBMNativeProvider().submit(compiled, 10, BACKEND, confirm=True)
    ibm.update(samples=["0x4"])
    with pytest.raises(HardwareError) as exc:
        IBMNativeProvider().result(job)
    assert exc.value.code == E_JOB


def test_unknown_and_malformed_job_ids(ibm, compiled):
    from qumo_hardware.base import JobHandle

    with pytest.raises(HardwareError) as exc:
        IBMNativeProvider().status(JobHandle("d3other", "ibm", BACKEND, 1, False, QUEUED))
    assert exc.value.code == E_UNKNOWN_JOB
    before = len(ibm.requests("/api/v1/jobs"))
    with pytest.raises(HardwareError) as exc:
        IBMNativeProvider().status(JobHandle("../backends", "ibm", BACKEND, 1, False, QUEUED))
    assert exc.value.code == E_UNKNOWN_JOB and len(ibm.requests("/api/v1/jobs")) == before


def test_server_dry_run_and_sampler_options_need_the_fallback(ibm, compiled):
    for prov in (IBMNativeProvider(server_dry_run=True), IBMNativeProvider(sampler_options={"twirling.enable_gates": 1})):
        with pytest.raises(HardwareError) as exc:
            prov.submit(compiled, 10, BACKEND, confirm=True)
        assert exc.value.code == E_INVALID_PAYLOAD and "--client python" in exc.value.hint


# ----------------------------------------------------------------- credentials
def test_rejected_api_key(ibm, monkeypatch):
    monkeypatch.setenv("QISKIT_IBM_TOKEN", "wrong-key")
    with pytest.raises(HardwareError) as exc:
        IBMNativeProvider().target(BACKEND)
    assert exc.value.code == E_CREDENTIALS and "wrong-key" not in str(exc.value)


def test_instance_must_be_a_crn(ibm, monkeypatch):
    monkeypatch.setenv("QISKIT_IBM_INSTANCE", "my-instance-name")
    with pytest.raises(HardwareError) as exc:
        IBMNativeProvider().target(BACKEND)
    assert exc.value.code == E_CREDENTIALS and "CRN" in str(exc.value)


def test_saved_account_file(ibm, monkeypatch, tmp_path):
    monkeypatch.delenv("QISKIT_IBM_TOKEN")
    monkeypatch.delenv("QISKIT_IBM_INSTANCE")
    with pytest.raises(HardwareError) as exc:
        IBMNativeProvider().target(BACKEND)
    assert exc.value.code == E_CREDENTIALS and "no IBM Quantum account" in str(exc.value)
    accounts = tmp_path / "qiskit-ibm.json"
    accounts.write_text(json.dumps({
        "other": {"channel": "ibm_quantum_platform", "token": "x", "instance": CRN, "url": "https://cloud.ibm.com"},
        "default-ibm-quantum-platform": {"channel": "ibm_quantum_platform", "token": KEY, "instance": CRN,
                                         "url": "https://cloud.ibm.com"},
    }))
    monkeypatch.setenv("QUMO_IBM_ACCOUNT_FILE", str(accounts))
    assert IBMNativeProvider().target(BACKEND).num_qubits == 5
    with pytest.raises(HardwareError) as exc:
        IBMNativeProvider(account_name="other").target(BACKEND)
    assert exc.value.code == E_CREDENTIALS  # that account's key is wrong


def test_plain_http_api_url_to_a_remote_host_is_refused(ibm, monkeypatch):
    monkeypatch.setenv("QUMO_IBM_API_URL", "http://quantum.example.com/api/v1")
    with pytest.raises(HardwareError) as exc:
        IBMNativeProvider().target(BACKEND)
    assert exc.value.code == E_CREDENTIALS and "https" in str(exc.value)


@pytest.mark.skipif(not have("qiskit_ibm_runtime"), reason="fake_* backends need qiskit-ibm-runtime + Aer")
def test_fake_backends_are_delegated_to_the_python_client():
    spec = IBMNativeProvider().target("fake_torino")
    assert spec.local and spec.num_qubits == 133


# ----------------------------------------------------------------- review regressions
def test_faulty_qubits_and_gates_are_left_out(ibm):
    props = properties()
    props["qubits"][4].append({"name": "operational", "value": 0, "unit": ""})
    for g in props["gates"]:
        if g["gate"] == "cz" and sorted(g["qubits"]) == [1, 2]:
            g["parameters"].append({"name": "operational", "value": 0, "unit": ""})
    ibm.update(properties=props)
    spec = IBMNativeProvider().target(BACKEND)
    assert spec.edges == [(0, 1), (2, 3)]
    assert spec.readout_errors[4] is None
    assert any("not operational" in w for w in spec.extras["warnings"])


def test_barrier_and_reset_are_sent(qumo):
    src = ('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\nreset q[0];\nx q[0];\nbarrier q[0],q[1];\n'
           'cz q[0],q[1];\nmeasure q[0] -> c[0];\n')
    qasm3 = qumo.ibm_isa_payload(src, 2, ["rz", "sx", "x", "cz"], [0, 1], True, -1.0, -1.0, -1.0)[0]
    assert "reset $0;" in qasm3 and ("barrier;" in qasm3 or "barrier $0, $1;" in qasm3)


def test_instance_flag_completes_an_environment_token(ibm, monkeypatch):
    monkeypatch.delenv("QISKIT_IBM_INSTANCE")
    assert IBMNativeProvider(instance=CRN).target(BACKEND).num_qubits == 5
    (r,) = ibm.requests("/api/v1/backends/" + BACKEND + "/configuration")
    assert r["headers"]["Service-CRN"] == CRN


def test_api_key_goes_only_to_an_ibm_cloud_iam_host(ibm, monkeypatch):
    monkeypatch.setenv("IAM_URL", "https://iam.example.com")
    with pytest.raises(HardwareError) as exc:
        IBMNativeProvider().target(BACKEND)
    assert exc.value.code == E_CREDENTIALS and "cloud.ibm.com" in str(exc.value)
    assert ibm.requests() == []


def test_private_endpoint_accounts_need_the_fallback(ibm, monkeypatch, tmp_path):
    monkeypatch.delenv("QISKIT_IBM_TOKEN")
    f = tmp_path / "acc.json"
    f.write_text(json.dumps({"default-ibm-quantum-platform": {"channel": "ibm_quantum_platform", "token": KEY,
                                                                "instance": CRN, "private_endpoint": True}}))
    monkeypatch.setenv("QUMO_IBM_ACCOUNT_FILE", str(f))
    with pytest.raises(HardwareError) as exc:
        IBMNativeProvider().target(BACKEND)
    assert "private endpoint" in str(exc.value) and "--client python" in exc.value.hint


def test_62_bit_samples(ibm, compiled, qumo):
    IBMNativeProvider().submit(compiled, 10, BACKEND, confirm=True)
    ibm.update(samples=["0x3fffffffffffffff", "0x1"])
    counts = dict(qumo.ibm_result("", "", "d3fakejob0001", 62))
    assert counts == {"1" * 62: 1, "0" * 61 + "1": 1}


def test_errors_never_carry_the_key_or_the_iam_token(ibm, compiled, monkeypatch):
    ibm.update(submit_status=400, submit_body={"errors": [{"message": f"bad request iam-token-1 {KEY}"}]})
    with pytest.raises(HardwareError) as exc:
        IBMNativeProvider().submit(compiled, 10, BACKEND, confirm=True)
    # the server echoed them on purpose; the client must not pass a bearer or key through
    text = str(exc.value) + exc.value.hint
    assert exc.value.code == E_JOB and "<redacted>" in text
    assert KEY not in text and "iam-token-1" not in text


def test_jobs_without_a_recorded_client_are_read_with_the_python_client():
    from qumo_hardware.base import JobHandle

    job = JobHandle("old-sdk-job", "ibm", "ibm_marrakesh", 10, False, QUEUED, meta={"provider_options": {}})
    assert type(runner._provider_for(job)).__name__ == "IBMProvider"
