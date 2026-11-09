"""IonQ plugin over the native Mojo client (qumo/exec/ionq_api.mojo), fully offline.

Every request goes through QUMO's own HTTP/TLS stack to a fake IonQ API
(tests/ionq_fake_server.py) on 127.0.0.1, so the real request bodies, headers, status
handling, JSON parsing and counts sampling are exercised. Nothing reaches IonQ and no
real key is used. When qiskit-ionq is installed, the circuit JSON is also compared with
the one that SDK builds (it is not needed by the plugin).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import BELL, have

from qumo_hardware import runner
from qumo_hardware.base import CANCELLED, DONE, ERROR, QUEUED, RUNNING, UNKNOWN
from qumo_hardware.compile import compile_for_target, load_qumo_py
from qumo_hardware.errors import (
    E_CONFIRM,
    E_CREDENTIALS,
    E_INVALID_PAYLOAD,
    E_JOB,
    E_LIMIT,
    E_UNKNOWN_BACKEND,
    HardwareError,
)
from qumo_hardware.ionq import MAX_SHOTS, NOISE_MODELS, SIMULATOR_QUBITS, IonQProvider
from qumo_hardware.registry import get_provider, provider_names

ARIA1 = "ionq_simulator:aria-1"
KEY = "test-key-not-real-0123"  # a placeholder, never a real credential
JOB_ID = "0a1b2c3d-0000-4000-8000-000000000042"
SERVER = Path(__file__).with_name("ionq_fake_server.py")

RICH = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
rx(0.5) q[0];
ry(0.25) q[1];
rz(1.5) q[2];
cz q[0],q[2];
cy q[1],q[0];
swap q[1],q[2];
sdg q[0];
tdg q[1];
sx q[1];
rxx(0.1) q[0],q[2];
ryy(0.2) q[1],q[2];
rzz(0.3) q[0],q[1];
measure q[2] -> c[0];
measure q[0] -> c[2];
"""


@pytest.fixture(scope="module")
def qumo():
    return load_qumo_py()


class FakeIonQ:
    def __init__(self, tmp: Path):
        self.log = tmp / "requests.jsonl"
        self.conf = tmp / "conf.json"
        self.log.write_text("")
        self.set()
        self.proc = subprocess.Popen([sys.executable, "-I", str(SERVER), str(self.log), str(self.conf)],
                                     stdout=subprocess.PIPE, text=True)
        self.port = int(self.proc.stdout.readline().strip())
        self.url = f"http://127.0.0.1:{self.port}/v0.4"

    def set(self, **conf):
        base = {"job_status": "completed", "histogram": {"0": 0.5, "3": 0.5},
                "backends": [{"backend": "simulator", "status": "available", "qubits": 29},
                             {"backend": "qpu.aria-1", "status": "retired", "qubits": 25},
                             {"backend": "qpu.forte-1", "status": "available", "qubits": 36}]}
        base.update(conf)
        self.conf.write_text(json.dumps(base))

    def requests(self) -> list[dict]:
        return [json.loads(line) for line in self.log.read_text().splitlines() if line.strip()]

    def stop(self):
        self.proc.kill()
        self.proc.wait()


@pytest.fixture
def ionq(tmp_path, monkeypatch, qumo):
    srv = FakeIonQ(tmp_path)
    monkeypatch.setenv("IONQ_API_URL", srv.url)
    monkeypatch.setenv("IONQ_API_KEY", KEY)
    yield srv
    srv.stop()


@pytest.fixture
def compiled_bell(ionq):
    return compile_for_target(BELL, IonQProvider().target(ARIA1), level=2, seed=1)


# ----------------------------------------------------------------- independence
def test_plugin_imports_no_sdk_and_no_python_http_or_json():
    src = (Path(__file__).resolve().parents[1] / "qumo_hardware" / "ionq.py").read_text()
    for banned in ("qiskit", "requests", "urllib", "import json", "http.client"):
        assert f"import {banned}" not in src and f"from {banned}" not in src, banned


# ----------------------------------------------------------------- backends and targets
def test_plugin_lists_the_simulator_noise_models():
    rows = IonQProvider().list_backends()
    assert [r.name for r in rows] == [f"ionq_simulator:{m}" for m in NOISE_MODELS]
    assert all(r.kind == "cloud-emulator" and r.billable for r in rows)


def test_target_is_the_remote_all_to_all_simulator():
    spec = IonQProvider().target(ARIA1)
    assert spec.all_to_all and spec.simulator and not spec.local
    assert spec.extras["noise_model"] == "aria-1" and spec.extras["native_2q_gate"] == "ms"
    assert spec.num_qubits == SIMULATOR_QUBITS and spec.max_shots == MAX_SHOTS
    assert {"h", "cx", "rz", "rzz"} <= set(spec.basis)


def test_is_local_is_false_for_every_backend():
    prov = IonQProvider()
    assert prov.is_local(ARIA1) is False and prov.is_local("ionq_simulator:forte-1") is False


@pytest.mark.parametrize("name", ["ionq_qpu.aria-1", "ionq_qpu.forte-1", "ionq_simulator:ideal",
                                  "ionq_simulator:no-such-model", "aria-1"])
def test_unsupported_backend_names_are_refused(name):
    with pytest.raises(HardwareError) as exc:
        IonQProvider().target(name)
    assert exc.value.code == E_UNKNOWN_BACKEND


def test_registry_knows_ionq():
    assert "ionq" in provider_names()
    assert isinstance(get_provider("ionq"), IonQProvider)
    # the CLI passes its options through the registry; IonQ ignores the ones it does not use
    assert isinstance(get_provider("ionq", seed=3, use_fractional_gates=False), IonQProvider)


def test_cli_dry_run_plans_without_network(ionq, capsys):
    from qumo_hardware import cli

    rc = cli.main(["run", str(Path(__file__).resolve().parents[3] / "examples" / "foundations" / "bell.qasm"),
                   "-p", "ionq", "-b", ARIA1, "-s", "100", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0 and "ionq_simulator:aria-1" in out and "unknown" in out
    assert ionq.requests() == []


def test_online_listing_is_public_and_flags_qpus(ionq):
    rows = IonQProvider().list_backends(online=True)
    qpus = [r for r in rows if r.kind == "hardware"]
    assert {r.name for r in qpus} == {"ionq_qpu.aria-1", "ionq_qpu.forte-1"}
    assert all("not enabled" in r.notes for r in qpus)
    reqs = ionq.requests()
    assert [(r["method"], r["path"]) for r in reqs] == [("GET", "/v0.4/backends")]
    assert "authorization" not in reqs[0]["headers"]  # the listing is public: the key is not sent


# ----------------------------------------------------------------- request body (no network)
def test_bell_body_matches_the_ionq_v1_format(qumo, compiled_bell):
    body, qmap, nclbits, dense = qumo.ionq_request(compiled_bell.qasm, 100, "aria-1", 7, "bell")
    obj = json.loads(body)
    assert obj["type"] == "ionq.circuit.v1" and obj["backend"] == "simulator" and obj["shots"] == 100
    assert obj["noise"] == {"model": "aria-1", "seed": 7}
    assert obj["input"] == {"gateset": "qis", "qubits": 2,
                            "circuit": [{"gate": "h", "targets": [0]},
                                        {"gate": "x", "controls": [0], "targets": [1]}]}
    assert obj["metadata"]["sampler_seed"] == "7"
    assert list(qmap) == [0, 1] and nclbits == 2 and dense == 2


def test_gate_aliases_and_measurement_map(qumo):
    body, qmap, nclbits, dense = qumo.ionq_request(RICH, 10, "", 0, "rich")
    obj = json.loads(body)
    assert "noise" not in obj
    gates = [(g["gate"], g.get("controls"), g["targets"], g.get("rotation")) for g in obj["input"]["circuit"]]
    assert gates == [("rx", None, [0], 0.5), ("ry", None, [1], 0.25), ("rz", None, [2], 1.5),
                     ("z", [0], [2], None), ("y", [1], [0], None), ("swap", None, [1, 2], None),
                     ("si", None, [0], None), ("ti", None, [1], None), ("v", None, [1], None),
                     ("xx", None, [0, 2], 0.1), ("yy", None, [1, 2], 0.2), ("zz", None, [0, 1], 0.3)]
    assert list(qmap) == [2, -1, 0] and nclbits == 3 and dense == 3


@pytest.mark.skipif(not have("qiskit_ionq"), reason="qiskit-ionq not installed (only used as a reference)")
@pytest.mark.parametrize("src", [BELL, RICH])
def test_circuit_json_equals_qiskit_ionq(qumo, src, monkeypatch):
    import warnings

    import qiskit_ionq
    import qiskit_ionq.ionq_backend as ib
    from qiskit_ionq.helpers import qiskit_to_ionq

    from qumo_hardware.qasm_bridge import to_qiskit

    monkeypatch.setattr(ib, "get_n_qubits", lambda backend, fallback=4: SIMULATOR_QUBITS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        backend = qiskit_ionq.IonQProvider(token=KEY).get_backend("ionq_simulator", gateset="qis")
    theirs = json.loads(qiskit_to_ionq(to_qiskit(src), backend, {"shots": 10, "sampler_seed": None}))
    ours = json.loads(qumo.ionq_request(src, 10, "aria-1", 0, "x")[0])
    assert ours["input"]["circuit"] == theirs["input"]["circuit"]
    assert ours["input"]["qubits"] == theirs["input"]["qubits"]
    assert ours["input"]["gateset"] == theirs["input"]["gateset"] == "qis"
    assert ours["type"] == theirs["type"] and ours["backend"] == theirs["backend"]


@pytest.mark.parametrize("src, what", [
    ('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\ncreg c[3];\nccx q[0],q[1],q[2];\nmeasure q[0] -> c[0];\n',
     "not in the IonQ QIS gateset"),
    ('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\ncreg c[1];\nu3(1,2,3) q[0];\nmeasure q[0] -> c[0];\n',
     "not in the IonQ QIS gateset"),
    ('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\nmeasure q[0] -> c[0];\nh q[0];\n'
     'measure q[0] -> c[1];\n', "mid-circuit"),
    ('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\nh q[0];\n', "no measurements"),
    ('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\ncreg c[1];\nreset q[0];\nmeasure q[0] -> c[0];\n',
     "not in the IonQ QIS gateset"),
])
def test_validate_refuses_what_the_v1_format_cannot_carry(src, what):
    with pytest.raises(HardwareError) as exc:
        IonQProvider().validate(src, ARIA1)
    assert exc.value.code == E_INVALID_PAYLOAD
    assert what in str(exc.value)


# ----------------------------------------------------------------- counts sampling (no network)
def test_counts_bit_order_leftmost_is_highest_clbit(qumo):
    # Qubit 0 reads 1 (key 1). clbit 1 measures qubit 0 and clbit 0 measures qubit 1 -> "10".
    assert dict(qumo.ionq_counts([1], [1.0], [1, 0], 50, 3)) == {"10": 50}
    # an unwritten clbit reads 0
    assert dict(qumo.ionq_counts([1], [1.0], [0, -1], 5, 3)) == {"01": 5}


def test_counts_follow_the_histogram_and_are_seeded(qumo):
    keys, probs = [0, 1, 2, 3], [0.1, 0.2, 0.3, 0.4]
    a = dict(qumo.ionq_counts(keys, probs, [0, 1], 20000, 11))
    b = dict(qumo.ionq_counts(keys, probs, [0, 1], 20000, 11))
    assert a == b and sum(a.values()) == 20000
    for key, p in (("00", 0.1), ("01", 0.2), ("10", 0.3), ("11", 0.4)):
        assert abs(a[key] / 20000 - p) < 0.015, (key, a)
    assert dict(qumo.ionq_counts(keys, probs, [0, 1], 20000, 12)) != a


@pytest.mark.parametrize("keys, probs", [([], []), ([0], [-0.1]), ([0, 1], [0.0, 0.0]), ([0], [float("nan")])])
def test_malformed_histograms_are_refused(qumo, keys, probs):
    with pytest.raises(Exception, match="QM-E959"):
        qumo.ionq_counts(keys, probs, [0], 10, 1)


# ----------------------------------------------------------------- credentials and endpoint safety
def test_missing_key_is_refused_before_any_request(ionq, compiled_bell, monkeypatch):
    monkeypatch.delenv("IONQ_API_KEY", raising=False)
    with pytest.raises(HardwareError) as exc:
        IonQProvider().submit(compiled_bell, 100, ARIA1, confirm=True)
    assert exc.value.code == E_CREDENTIALS and "IONQ_API_KEY" in str(exc.value)
    assert ionq.requests() == []


def test_blank_key_counts_as_missing(ionq, compiled_bell, monkeypatch):
    monkeypatch.setenv("IONQ_API_KEY", "   ")
    with pytest.raises(HardwareError) as exc:
        IonQProvider().submit(compiled_bell, 100, ARIA1, confirm=True)
    assert exc.value.code == E_CREDENTIALS
    assert ionq.requests() == []


def test_plain_http_to_a_remote_host_is_refused(ionq, compiled_bell, monkeypatch):
    monkeypatch.setenv("IONQ_API_URL", "http://api.example.com/v0.4")
    with pytest.raises(HardwareError) as exc:
        IonQProvider().submit(compiled_bell, 100, ARIA1, confirm=True)
    assert exc.value.code == E_UNKNOWN_BACKEND and "https" in str(exc.value)
    assert ionq.requests() == []


def test_key_goes_only_to_the_authorization_header(ionq, compiled_bell):
    job = IonQProvider().submit(compiled_bell, 100, ARIA1, confirm=True)
    (post,) = ionq.requests()
    assert post["headers"]["authorization"] == f"apiKey {KEY}"
    assert KEY not in post["body"] and KEY not in post["path"]
    text = json.dumps(job.to_dict(), default=str)
    assert KEY not in text and "IONQ_API_KEY" not in text


def test_rejected_key_maps_to_the_credentials_error(ionq, compiled_bell):
    ionq.set(submit_status=401, submit_body={"error": "Unauthorized"})
    with pytest.raises(HardwareError) as exc:
        IonQProvider().submit(compiled_bell, 100, ARIA1, confirm=True)
    assert exc.value.code == E_CREDENTIALS


@pytest.mark.parametrize("body", [
    {"error": "noise model unavailable"},
    {"statusCode": 400, "error": "Bad Request", "message": "noise model unavailable"},
    {"error": {"type": "InvalidRequest", "message": "noise model unavailable"}},
    {"code": "bad_request", "message": "noise model unavailable"},
])
def test_server_error_carries_the_api_message(ionq, compiled_bell, body):
    ionq.set(submit_status=400, submit_body=body)
    with pytest.raises(HardwareError) as exc:
        IonQProvider().submit(compiled_bell, 100, ARIA1, confirm=True)
    assert exc.value.code == E_JOB and "noise model unavailable" in str(exc.value)


def test_long_non_ascii_error_body_does_not_crash(ionq, compiled_bell):
    # regression: the detail was cut at byte 200 inside a UTF-8 sequence and Mojo aborted the process
    ionq.set(submit_status=400, submit_body={"message": "a" * 199 + "\u00e9" * 300})
    with pytest.raises(HardwareError) as exc:
        IonQProvider().submit(compiled_bell, 100, ARIA1, confirm=True)
    assert exc.value.code == E_JOB and "aaaa" in str(exc.value)


def test_redirect_is_not_followed_and_the_key_stays_put(ionq, compiled_bell):
    ionq.set(submit_status=302, submit_body={}, submit_headers={"Location": "http://127.0.0.1:9/steal"})
    with pytest.raises(HardwareError) as exc:
        IonQProvider().submit(compiled_bell, 100, ARIA1, confirm=True)
    assert exc.value.code == E_JOB and "redirects are not followed" in str(exc.value)
    assert [r["path"] for r in ionq.requests()] == ["/v0.4/jobs"]


def test_failed_submission_transport_error_warns_about_a_possible_job(ionq, compiled_bell):
    ionq.set(submit_raw="{not json")  # body arrives but is not JSON: the POST reached the server
    with pytest.raises(HardwareError) as exc:
        IonQProvider().submit(compiled_bell, 100, ARIA1, confirm=True)
    assert exc.value.code == E_JOB
    ionq.stop()  # nothing listening any more: a transport error after the POST was attempted
    with pytest.raises(HardwareError) as exc:
        IonQProvider().submit(compiled_bell, 100, ARIA1, confirm=True)
    assert exc.value.code == E_JOB and "IonQ console" in exc.value.hint


def test_status_retries_transient_errors(ionq, compiled_bell):
    job = IonQProvider().submit(compiled_bell, 10, ARIA1, confirm=True)
    ionq.set(status_fail_first=2, job_status="completed")
    assert IonQProvider().status(job) == DONE
    gets = [r for r in ionq.requests() if r["method"] == "GET"]
    assert len(gets) == 3


def test_failed_job_reports_the_failure_reason(ionq, compiled_bell):
    job = IonQProvider().submit(compiled_bell, 10, ARIA1, confirm=True)
    ionq.set(job_status="failed", job_extra={"failure": {"code": "CompilationError", "error": "too many gates"}})
    with pytest.raises(HardwareError) as exc:
        IonQProvider().result(job)
    assert "ERROR" in str(exc.value) and "too many gates" in str(exc.value)


def test_noise_seed_and_identity_width_match_qiskit_ionq(qumo):
    src = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[4];\ncreg c[2];\nh q[0];\nid q[3];\nmeasure q[0] -> c[0];\n'
    obj = json.loads(qumo.ionq_request(src, 10, "aria-1", 9, "x")[0])
    assert obj["noise"] == {"model": "aria-1", "seed": 9}
    assert obj["input"]["qubits"] == 4 and obj["input"]["circuit"] == [{"gate": "h", "targets": [0]}]


def test_invalid_api_url_is_a_configuration_error(ionq, compiled_bell, monkeypatch):
    monkeypatch.setenv("IONQ_API_URL", "ftp://api.ionq.co/v0.4")
    with pytest.raises(HardwareError) as exc:
        IonQProvider().submit(compiled_bell, 100, ARIA1, confirm=True)
    assert exc.value.code == E_UNKNOWN_BACKEND


@pytest.mark.parametrize("job_id", ["../backends", "a/b", "x?y=1", "", "a" * 65])
def test_job_ids_that_could_change_the_path_are_refused(ionq, qumo, job_id):
    with pytest.raises(Exception, match="QM-E959"):
        qumo.ionq_status(job_id)
    assert ionq.requests() == []


# ----------------------------------------------------------------- safety gate and dry run
def test_submit_without_confirmation_is_refused(ionq, compiled_bell):
    with pytest.raises(HardwareError) as exc:
        IonQProvider().submit(compiled_bell, 100, ARIA1)
    assert exc.value.code == E_CONFIRM
    assert ionq.requests() == []


def test_plan_and_dry_run_never_submit(ionq):
    prov = IonQProvider()
    p = runner.plan(BELL, prov, ARIA1, shots=100)
    assert p.requires_confirmation and p.cost.unit == "unknown"
    out = runner.run(BELL, prov, ARIA1, shots=100, dry_run=True)
    assert out["dry_run"] is True and out["submitted"] is False
    with pytest.raises(HardwareError) as exc:
        runner.run(BELL, prov, ARIA1, shots=100)  # no confirm: the runner's gate refuses
    assert exc.value.code == E_CONFIRM
    assert ionq.requests() == []


def test_submit_refuses_qpu_and_out_of_range_shots(ionq, compiled_bell):
    with pytest.raises(HardwareError) as exc:
        IonQProvider().submit(compiled_bell, 10, "ionq_qpu.aria-1", confirm=True)
    assert exc.value.code == E_UNKNOWN_BACKEND
    with pytest.raises(HardwareError) as exc:
        IonQProvider().submit(compiled_bell, MAX_SHOTS + 1, ARIA1, confirm=True)
    assert exc.value.code == E_LIMIT
    assert ionq.requests() == []


# ----------------------------------------------------------------- submit, status, result
def test_submit_posts_the_bell_job(ionq, compiled_bell):
    job = IonQProvider().submit(compiled_bell, 500, ARIA1, seed=7, confirm=True)
    assert job.job_id == JOB_ID and job.status == QUEUED and job.local is False
    (post,) = ionq.requests()
    assert post["method"] == "POST" and post["path"] == "/v0.4/jobs"
    assert post["headers"]["content-type"] == "application/json"
    body = json.loads(post["body"])
    assert body["shots"] == 500 and body["noise"] == {"model": "aria-1", "seed": 7}
    assert [g["gate"] for g in body["input"]["circuit"]] == ["h", "x"]
    assert job.meta["clbit_qubit"] == [0, 1] and job.meta["num_clbits"] == 2 and job.meta["sampler_seed"] == 7


def test_end_to_end_run_then_result(ionq):
    out = runner.run(BELL, IonQProvider(), ARIA1, shots=400, seed=5, confirm=True, wait=False)
    assert out["submitted"] is True and out["status"] == QUEUED and out["job_id"] == JOB_ID
    res = runner.job_result(JOB_ID)
    counts = res["counts"]
    assert set(counts) <= {"00", "11"} and sum(counts.values()) == 400
    assert 150 < counts.get("00", 0) < 250
    assert res["bit_order"].startswith("leftmost")
    paths = [(r["method"], r["path"]) for r in ionq.requests()]
    assert paths[0] == ("POST", "/v0.4/jobs")
    assert ("GET", f"/v0.4/jobs/{JOB_ID}/results/probabilities") in paths


@pytest.mark.parametrize("api_status, expected", [
    ("submitted", QUEUED), ("ready", QUEUED), ("started", RUNNING), ("running", RUNNING),
    ("completed", DONE), ("failed", ERROR), ("canceled", CANCELLED), ("something-new", UNKNOWN),
])
def test_status_mapping(ionq, compiled_bell, api_status, expected):
    job = IonQProvider().submit(compiled_bell, 10, ARIA1, confirm=True)
    ionq.set(job_status=api_status)
    assert IonQProvider().status(job) == expected


def test_result_of_unfinished_job_is_refused(ionq, compiled_bell):
    job = IonQProvider().submit(compiled_bell, 10, ARIA1, confirm=True)
    ionq.set(job_status="started")
    with pytest.raises(HardwareError) as exc:
        IonQProvider().result(job)
    assert exc.value.code == E_JOB and "RUNNING" in str(exc.value)


def test_result_with_a_bad_histogram_is_refused(ionq, compiled_bell):
    job = IonQProvider().submit(compiled_bell, 10, ARIA1, confirm=True)
    ionq.set(histogram={"zero": 1.0})
    with pytest.raises(HardwareError) as exc:
        IonQProvider().result(job)
    assert exc.value.code == E_JOB


# ----------------------------------------------------------------- costs
def test_cost_is_unknown_not_a_number():
    est = IonQProvider().estimate_cost(None, 1000, ARIA1)
    assert est.unit == "unknown" and est.amount == 0.0 and est.usd is None
    assert est.billable and est.summary().startswith("unknown")
