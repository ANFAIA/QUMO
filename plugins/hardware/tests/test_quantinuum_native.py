"""Quantinuum native client (qumo/exec/quantinuum_api.mojo) against a fake Nexus on 127.0.0.1.

Request bodies are compared with what qnexus 0.51 itself builds, and circuits / results with
pytket, when those packages are installed (they are references only; the native path does not
import them).
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
    E_UNKNOWN_JOB,
    HardwareError,
)
from qumo_hardware.quantinuum_native import QuantinuumNativeProvider
from qumo_hardware.registry import get_provider

SERVER = Path(__file__).with_name("nexus_fake_server.py")
JOB = "7a1e2b3c-0000-4000-8000-000000000b10"
REFRESH, ACCESS = "refresh-tok-not-real", "access-tok-1"


@pytest.fixture(scope="module")
def qumo():
    return load_qumo_py()


def backend_result(readouts: list[list[int]]) -> dict:
    """A pytket BackendResult dict (OutcomeArray rows packed big-endian, bits c[0], c[1], ...)."""
    width = len(readouts[0])
    rows = []
    for r in readouts:
        row = [0] * ((width + 7) // 8)
        for j, b in enumerate(r):
            if b:
                row[j // 8] |= 1 << (7 - j % 8)
        rows.append(row)
    return {"bits": [["c", [j]] for j in range(width)], "qubits": [["q", [j]] for j in range(width)],
            "shots": {"width": width, "array": rows}}


class Fake:
    def __init__(self, tmp: Path, logged_in: bool = True):
        self.log, self.state = tmp / "log.jsonl", tmp / "state.json"
        self.log.write_text("")
        self.auth = tmp / "auth"
        self.auth.mkdir()
        if logged_in:
            (self.auth / "token.json").write_text(json.dumps({"data": {"refresh_token": REFRESH,
                                                                       "delete_version_after": None}}))
            (self.auth / "id.json").write_text(json.dumps({"data": {"access_token": ACCESS}}))
        self.put({"refresh_token": REFRESH, "access_token": ACCESS, "refreshed_access": "access-tok-2",
                  "projects": ["99999999-aaaa-4bbb-8ccc-dddddddddddd"], "result_id": "r1r1r1r1-0000-4000-8000-00000000r001".replace("r", "e"),
                  "backend_result": backend_result([[0, 0], [1, 1], [1, 1], [0, 1]]),
                  "devices": [{"is_local": False, "backend_info_list": [
                      {"name": "QuantinuumBackend", "device_name": "H2-1", "architecture": {"n_nodes": 56}},
                      {"name": "QuantinuumBackend", "device_name": "H2-9", "architecture": {"n_nodes": 64}}]}]})
        self.proc = subprocess.Popen([sys.executable, "-I", str(SERVER), str(self.log), str(self.state)],
                                     stdout=subprocess.PIPE, text=True)
        self.base = f"http://127.0.0.1:{int(self.proc.stdout.readline())}"

    def put(self, st: dict) -> None:
        self.state.write_text(json.dumps(st))

    def update(self, **kw) -> None:
        st = json.loads(self.state.read_text())
        st.update(kw)
        self.put(st)

    def requests(self, prefix: str = "") -> list[dict]:
        rows = [json.loads(x) for x in self.log.read_text().splitlines() if x.strip()]
        return [r for r in rows if r["path"].startswith(prefix)]

    def stop(self) -> None:
        self.proc.kill()
        self.proc.wait()


@pytest.fixture
def nexus(tmp_path, monkeypatch, qumo):
    f = Fake(tmp_path)
    monkeypatch.setenv("QUMO_NEXUS_URL", f.base)
    monkeypatch.setenv("QUMO_NEXUS_AUTH_DIR", str(f.auth))
    monkeypatch.delenv("QUMO_HW_CLIENT", raising=False)
    monkeypatch.delenv("QUMO_HW_CLIENT_QUANTINUUM", raising=False)
    yield f
    f.stop()


@pytest.fixture
def compiled(nexus):
    return compile_for_target(BELL, QuantinuumNativeProvider().target("H2-1E"), level=2, seed=1)


# ----------------------------------------------------------------- selection, independence
def test_native_is_the_default_client(nexus):
    assert type(get_provider("quantinuum")) is QuantinuumNativeProvider
    assert type(get_provider("quantinuum", client="python")).__name__ == "QuantinuumProvider"


def test_native_module_imports_neither_qnexus_nor_pytket():
    src = (Path(__file__).resolve().parents[1] / "qumo_hardware" / "quantinuum_native.py").read_text()
    for banned in ("qnexus", "pytket", "guppy"):
        assert f"import {banned}" not in src and f"from {banned}" not in src


# ----------------------------------------------------------------- circuit and cost (no network)
@pytest.mark.skipif(not have("pytket"), reason="pytket is only the reference here")
def test_native_circuit_matches_pytket(qumo):
    from qumo_hardware.qasm_bridge import to_pytket
    from qumo_hardware.quantinuum import native_gate_counts, to_native

    src = ('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\ncreg c[3];\nrz(-0.7) q[0];\nrx(1.25) q[1];\n'
           'ry(-3.5) q[2];\nrzz(-1.5707963267948966) q[0],q[2];\nreset q[1];\nmeasure q[0] -> c[0];\n'
           'measure q[1] -> c[2];\nmeasure q[2] -> c[1];\n')
    ref = to_native(to_pytket(src))
    js, nq, nc, n1, n2, nm, _ = qumo.quantinuum_native_circuit(src)
    ours, theirs = json.loads(js), ref.to_dict()

    def per_wire(d):
        out = {}
        for c in d["commands"]:
            ps = tuple(round(float(x) % (4 if i == 0 else 2), 9) for i, x in enumerate(c["op"].get("params", [])))
            for a in c["args"]:
                out.setdefault((a[0], a[1][0]), []).append((c["op"]["type"], ps, tuple((x[0], x[1][0]) for x in c["args"])))
        return out

    assert per_wire(ours) == per_wire(theirs)
    assert ours["qubits"] == theirs["qubits"] and ours["bits"] == theirs["bits"]
    counts = native_gate_counts(ref)
    assert (n1, n2, nm) == (counts["n1q"], counts["n2q"], counts["n_measure_reset"])


@pytest.mark.parametrize("body, what", [
    ("h q[0];\nmeasure q[0] -> c[0];\n", "not in the Quantinuum native basis"),
    ("rx(0.1) q[0];\n", "no measurements"),
])
def test_payloads_the_native_client_refuses(qumo, body, what):
    src = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\n' + body
    with pytest.raises(Exception) as exc:
        qumo.quantinuum_native_circuit(src)
    assert "QM-E955" in str(exc.value) and what in str(exc.value)


@pytest.mark.skipif(not have("pytket"), reason="pytket is only the reference here")
@pytest.mark.parametrize("n_cl, cond", [(2, True), (5, True), (33, False), (63, False), (64, False), (98, False)])
def test_registers_and_conditions_match_the_python_client(qumo, n_cl, cond):
    from qumo_hardware.qasm_bridge import to_pytket
    from qumo_hardware.quantinuum import clbit_registers, native_gate_counts, split_classical_register, to_native

    nq = min(n_cl, 6)
    lines = ['OPENQASM 2.0;', 'include "qelib1.inc";', f"qreg q[{nq}];", f"creg c[{n_cl}];", "rx(1.0) q[0];"]
    lines += [f"measure q[{k % nq}] -> c[{k}];" for k in range(n_cl)]
    if cond:
        lines += ["if(c==3) rx(0.5) q[1];", "if(c==2) rzz(0.25) q[0],q[1];", "if(c==1) measure q[1] -> c[1];"]
    src = "\n".join(lines) + "\n"
    ref = split_classical_register(to_native(to_pytket(src)), clbit_registers(n_cl, "H2"))
    js, nq2, nc, n1, n2, nm, regs = qumo.quantinuum_native_circuit(src)
    ours, theirs = json.loads(js), ref.to_dict()

    def norm(cmds):
        out = []
        for c in cmds:
            op = c["op"]
            if op["type"] == "Conditional":
                inner = op["conditional"]["op"]
                ps = tuple(round(float(x) % 4, 9) for x in inner.get("params", []))
                out.append(("Conditional", inner["type"], ps, op["conditional"]["value"], op["conditional"]["width"],
                            tuple((a[0], a[1][0]) for a in c["args"])))
            elif op["type"] != "Barrier":
                ps = tuple(round(float(x) % (4 if i == 0 else 2), 9) for i, x in enumerate(op.get("params", [])))
                out.append((op["type"], ps, tuple((a[0], a[1][0]) for a in c["args"])))
        return sorted(out)

    assert norm(ours["commands"]) == norm(theirs["commands"])
    assert sorted(map(tuple, (map(str, b) for b in ours["bits"]))) == sorted(map(tuple, (map(str, b) for b in theirs["bits"])))
    assert list(regs) == [name for name, _ in clbit_registers(n_cl, "H2")]
    counts = native_gate_counts(ref)
    assert (n1, n2, nm) == (counts["n1q"], counts["n2q"], counts["n_measure_reset"])


def test_condition_over_a_split_register_is_refused(qumo):
    src = ('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[64];\nmeasure q[0] -> c[0];\n'
           'if(c==1) rx(0.5) q[1];\nmeasure q[1] -> c[63];\n')
    with pytest.raises(Exception, match="one register of at most 63 bits"):
        qumo.quantinuum_native_circuit(src)


@pytest.mark.parametrize("n_cl", [64, 98])
def test_wide_results_round_trip(nexus, qumo, n_cl):
    ones = {0, 1, 62, 63, n_cl - 1}
    readout = [1 if k in ones else 0 for k in range(n_cl)]
    d = backend_result([readout])
    # split layout, as the uploaded circuit declares it: c_0[0..62], c_1[...]
    d["bits"] = [[f"c_{k // 63}", [k % 63]] for k in range(n_cl)]
    nexus.update(backend_result=d)
    counts = dict(qumo.quantinuum_result(JOB, n_cl))
    (key,) = counts
    assert {n_cl - 1 - i for i, ch in enumerate(key) if ch == "1"} == ones


def test_cost_formula_and_free_syntax_checker(nexus, compiled):
    prov = QuantinuumNativeProvider()
    est = prov.estimate_cost(compiled, 100, "H2-1E")
    b = est.breakdown
    assert est.unit == "eHQC" and est.amount == pytest.approx(5 + 100 * (b["n1q"] + 10 * b["n2q"] + 5 * b["n_spam"]) / 5000)
    assert prov.estimate_cost(compiled, 100, "H2-1SC").unit == "free"


# ----------------------------------------------------------------- protocol
def test_submit_requires_confirmation(nexus, compiled):
    with pytest.raises(HardwareError) as exc:
        QuantinuumNativeProvider().submit(compiled, 100, "H2-1E")
    assert exc.value.code == E_CONFIRM and nexus.requests() == []


def test_submit_flow_and_bodies(nexus, compiled):
    prov = QuantinuumNativeProvider(project="qumo-hardware")
    job = prov.submit(compiled, 100, "H2-1E", confirm=True, name="qumo-test")
    assert job.job_id == JOB and job.status == QUEUED
    paths = [(r["method"], r["path"].split("?")[0]) for r in nexus.requests()]
    assert paths == [("GET", "/api/projects/v1beta2"), ("POST", "/api/circuits/v1beta2"), ("POST", "/api/jobs/v1beta3")]
    lookup = nexus.requests("/api/projects")[0]["path"]
    assert "filter%5Bname_exact%5D=qumo-hardware" in lookup and "filter%5Barchived%5D=false" in lookup
    for r in nexus.requests():
        assert r["cookie"] == f"myqos_oat={REFRESH}; myqos_id={ACCESS}" and r["user_agent"].startswith("qumo-hardware")
    circ = json.loads(nexus.requests("/api/circuits")[0]["body"])["data"]
    assert circ["type"] == "circuit" and circ["attributes"]["name"] == "qumo-test"
    assert circ["relationships"]["project"]["data"] == {"id": "99999999-aaaa-4bbb-8ccc-dddddddddddd", "type": "project"}
    assert {c["op"]["type"] for c in circ["attributes"]["commands"]} <= {"Rz", "PhasedX", "ZZPhase", "Measure"}
    job_body = json.loads(nexus.requests("/api/jobs")[0]["body"])["data"]
    d = job_body["attributes"]["definition"]
    assert job_body["attributes"]["job_type"] == "execute" and d["job_definition_type"] == "execute_job_definition"
    est = prov.estimate_cost(compiled, 100, "H2-1E")
    assert d["items"] == [{"program_id": "c1c1c1c1-0000-4000-8000-000000000001", "n_shots": 100,
                           "max_cost": round(est.amount * 1.25 + 1.0, 2)}]
    assert d["backend_config"]["type"] == "QuantinuumConfig" and d["backend_config"]["device_name"] == "H2-1E"
    text = json.dumps(job.to_dict(), default=str)
    assert REFRESH not in text and ACCESS not in text


@pytest.mark.skipif(not have("qnexus"), reason="qnexus is only the reference here")
@pytest.mark.parametrize("device", ["H2-1", "H2-2", "H2-1E", "H2-2E", "H2-Emulator", "H2-1SC", "H2-1-simulator",
                                    "H2E"])
def test_backend_config_is_what_qnexus_sends(nexus, compiled, device):
    import qnexus as qnx

    QuantinuumNativeProvider().submit(compiled, 10, device, confirm=True, max_cost=50)
    d = json.loads(nexus.requests("/api/jobs")[0]["body"])["data"]["attributes"]["definition"]
    assert d["backend_config"] == qnx.QuantinuumConfig(device_name=device).model_dump(exclude_none=True, mode="json")
    assert d["items"][0]["max_cost"] == 50


def test_project_is_created_when_missing(nexus, compiled):
    nexus.update(projects=[])
    QuantinuumNativeProvider().submit(compiled, 10, "H2-1E", confirm=True)
    (create,) = [r for r in nexus.requests("/api/projects") if r["method"] == "POST"]
    assert json.loads(create["body"]) == {"data": {"attributes": {"name": "qumo-hardware", "properties": {}},
                                                   "relationships": {}, "type": "project"}}


def test_expired_access_token_is_refreshed_once(nexus, compiled):
    nexus.update(access_token="access-tok-2")  # the stored access token no longer works
    job = QuantinuumNativeProvider().submit(compiled, 10, "H2-1E", confirm=True)
    assert job.job_id == JOB
    refreshes = nexus.requests("/auth/tokens/refresh")
    assert len(refreshes) == 1 and refreshes[0]["cookie"] == f"myqos_oat={REFRESH}"
    assert nexus.requests("/api/jobs")[0]["cookie"].endswith("myqos_id=access-tok-2")


def test_expired_login(nexus, compiled):
    nexus.update(access_token="nobody", refreshed_access="")
    with pytest.raises(HardwareError) as exc:
        QuantinuumNativeProvider().submit(compiled, 10, "H2-1E", confirm=True)
    assert exc.value.code == E_CREDENTIALS and "qnx login" in exc.value.hint


def test_not_logged_in(nexus, compiled, monkeypatch, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("QUMO_NEXUS_AUTH_DIR", str(empty))
    with pytest.raises(HardwareError) as exc:
        QuantinuumNativeProvider().submit(compiled, 10, "H2-1E", confirm=True)
    assert exc.value.code == E_CREDENTIALS and "qnx login" in exc.value.hint
    assert nexus.requests() == []


@pytest.mark.parametrize("api, expected", [
    ("SUBMITTED", QUEUED), ("QUEUED", QUEUED), ("RETRYING", QUEUED), ("RUNNING", RUNNING), ("COMPLETED", DONE),
    ("CANCELLED", CANCELLED), ("ERROR", ERROR), ("DEPLETED", ERROR),
])
def test_status_mapping(nexus, compiled, api, expected):
    job = QuantinuumNativeProvider().submit(compiled, 10, "H2-1E", confirm=True)
    nexus.update(job_status=api)
    assert QuantinuumNativeProvider().status(job) == expected


def test_end_to_end_run_and_counts(nexus):
    out = runner.run(BELL, get_provider("quantinuum"), "H2-1E", shots=4, confirm=True, wait=False)
    assert out["job_id"] == JOB
    res = runner.job_result(JOB)
    assert res["counts"] == {"00": 1, "11": 2, "10": 1}  # readout [0, 1] = c[1] set -> "10"


@pytest.mark.skipif(not have("pytket"), reason="pytket is only the reference here")
def test_counts_match_pytket_backend_result(nexus, compiled, qumo):
    from pytket.backends.backendresult import BackendResult

    from qumo_hardware.quantinuum import counts_from_backend_result

    readouts = [[1, 0, 1], [0, 0, 1], [1, 1, 1], [1, 0, 1]]
    d = backend_result(readouts)
    nexus.update(backend_result=d)
    QuantinuumNativeProvider().submit(compiled, 10, "H2-1E", confirm=True)
    ours = dict(qumo.quantinuum_result(JOB, 3))
    assert ours == counts_from_backend_result(BackendResult.from_dict(d), 3)


def test_counts_form_of_backend_result(nexus, compiled, qumo):
    nexus.update(backend_result={"bits": [["c", [0]], ["c", [1]]],
                                 "counts": [{"outcome": {"width": 2, "array": [[192]]}, "count": 7},
                                            {"outcome": {"width": 2, "array": [[0]]}, "count": 3}]})
    assert dict(qumo.quantinuum_result(JOB, 2)) == {"11": 7, "00": 3}


def test_failed_job_reports_the_detail(nexus, compiled):
    job = QuantinuumNativeProvider().submit(compiled, 10, "H2-1E", confirm=True)
    nexus.update(job_status="ERROR", error_detail="Circuit has too many qubits")
    with pytest.raises(HardwareError) as exc:
        QuantinuumNativeProvider().result(job)
    assert exc.value.code == E_JOB and "too many qubits" in str(exc.value)


def test_unknown_and_malformed_job_ids(nexus, qumo):
    with pytest.raises(Exception, match="QM-E958"):
        qumo.quantinuum_status("0000aaaa-0000-4000-8000-000000000000")
    before = len(nexus.requests())
    with pytest.raises(Exception, match="QM-E958"):
        qumo.quantinuum_status("../results/v1beta3/x")
    assert len(nexus.requests()) == before


def test_online_device_listing(nexus):
    rows = {r.name: r for r in QuantinuumNativeProvider().list_backends(online=True)}
    assert rows["H2-9"].num_qubits == 64 and rows["H2-9"].status == "online-listed"


def test_helios_and_local_emulators_use_the_python_client():
    prov = QuantinuumNativeProvider()
    assert not prov._native_kind("Helios-1") and not prov._native_kind("H2-1LE") and not prov._native_kind("selene")
    assert prov._native_kind("H2-1") and prov._native_kind("H2-1E") and prov._native_kind("H2-1SC")


def test_plain_http_nexus_url_to_a_remote_host_is_refused(nexus, compiled, monkeypatch):
    monkeypatch.setenv("QUMO_NEXUS_URL", "http://nexus.example.com")
    with pytest.raises(HardwareError) as exc:
        QuantinuumNativeProvider().submit(compiled, 10, "H2-1E", confirm=True)
    assert exc.value.code == E_CREDENTIALS
    assert nexus.requests() == []


# ----------------------------------------------------------------- review regressions
@pytest.mark.skipif(not have("pytket"), reason="pytket is only the reference here")
@pytest.mark.parametrize("gate", ["rx(0)", "rx(6.283185307179586)", "rx(12.566370614359172)", "ry(1e-13)",
                                  "rx(3.141592653589793)", "rz(0)", "rzz(6.283185307179586)"])
def test_identity_rotations_are_counted_like_pytket(qumo, gate):
    from qumo_hardware.qasm_bridge import to_pytket
    from qumo_hardware.quantinuum import native_gate_counts, to_native

    args = "q[0],q[1]" if gate.startswith("rzz") else "q[0]"
    src = f'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\n{gate} {args};\nmeasure q[0] -> c[0];\n'
    ref = native_gate_counts(to_native(to_pytket(src)))
    _, _, _, n1, n2, nm, _ = qumo.quantinuum_native_circuit(src)
    assert (n1, n2, nm) == (ref["n1q"], ref["n2q"], ref["n_measure_reset"])


@pytest.mark.skipif(not have("pytket"), reason="TKET comparison mode needs pytket")
def test_tket_comparison_mode_costs_the_tket_circuit(nexus):
    from conftest import ghz

    native = runner.plan(ghz(5), get_provider("quantinuum"), "H2-1E", shots=100, provider_transpile=True)
    python = runner.plan(ghz(5), get_provider("quantinuum", client="python"), "H2-1E", shots=100,
                         provider_transpile=True)
    assert native.cost.amount == python.cost.amount
    assert nexus.requests() == []


@pytest.mark.parametrize("var", ["NEXUS_PORT", "NEXUS_PROTOCOL", "NEXUS_TOKEN_PATH", "NEXUS_CONFIG_FILE"])
def test_qnexus_settings_it_cannot_honour_stop_it(nexus, compiled, monkeypatch, var):
    monkeypatch.setenv(var, "x")
    with pytest.raises(HardwareError) as exc:
        QuantinuumNativeProvider().submit(compiled, 10, "H2-1E", confirm=True)
    assert exc.value.code == E_CREDENTIALS and var in str(exc.value) and "--client python" in exc.value.hint
    assert nexus.requests() == []


def test_a_qnexus_config_file_stops_it(nexus, compiled, monkeypatch, tmp_path):
    home = tmp_path / "home2"
    (home / ".qnexus").mkdir(parents=True)
    (home / ".qnexus" / "config").write_text("NEXUS_DOMAIN=staging.nexus.example\n")
    monkeypatch.setenv("HOME", str(home))
    with pytest.raises(HardwareError) as exc:
        QuantinuumNativeProvider().submit(compiled, 10, "H2-1E", confirm=True)
    assert exc.value.code == E_CREDENTIALS and "qnexus config" in str(exc.value)
    assert nexus.requests() == []


def test_results_ignore_undeclared_registers_and_check_rows(nexus, qumo):
    d = backend_result([[1, 0]])
    d["bits"].append(["extra", [0]])
    d["shots"]["width"] = 3
    d["shots"]["array"] = [[0b10100000]]
    nexus.update(backend_result=d)
    assert dict(qumo.quantinuum_result(JOB, 2)) == {"01": 1}
    nexus.update(backend_result={"bits": [["c", [j]] for j in range(9)], "shots": {"width": 9, "array": [[255]]}})
    with pytest.raises(Exception, match="QM-E959"):
        qumo.quantinuum_result(JOB, 9)


def test_conditional_circuit_end_to_end(nexus):
    src = ('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\nrx(1.5707963267948966) q[0];\n'
           'measure q[0] -> c[0];\nif(c==1) rx(3.141592653589793) q[1];\nmeasure q[1] -> c[1];\n')
    out = runner.run(src, get_provider("quantinuum"), "H2-1E", shots=4, confirm=True, wait=False)
    assert out["job_id"] == JOB
    body = json.loads(nexus.requests("/api/circuits")[0]["body"])["data"]["attributes"]
    conds = [c for c in body["commands"] if c["op"]["type"] == "Conditional"]
    assert len(conds) == 1 and conds[0]["op"]["conditional"]["value"] == 1
    assert conds[0]["op"]["conditional"]["op"]["type"] == "PhasedX"


def test_redirects_are_refused_and_tokens_redacted(nexus, compiled):
    nexus.update(job_http=[302, {}])
    with pytest.raises(HardwareError) as exc:
        QuantinuumNativeProvider().submit(compiled, 10, "H2-1E", confirm=True)
    assert "redirect" in str(exc.value)
    nexus.update(job_http=[400, {"errors": [{"detail": f"bad cookie {REFRESH} {ACCESS}"}]}])
    with pytest.raises(HardwareError) as exc:
        QuantinuumNativeProvider().submit(compiled, 10, "H2-1E", confirm=True)
    text = str(exc.value)
    assert REFRESH not in text and ACCESS not in text and "<redacted>" in text


def test_missing_access_token_is_refreshed_on_the_first_call(nexus, compiled):
    (nexus.auth / "id.json").unlink()
    nexus.update(access_token="access-tok-2")
    assert QuantinuumNativeProvider().submit(compiled, 10, "H2-1E", confirm=True).job_id == JOB
    assert len(nexus.requests("/auth/tokens/refresh")) == 1


def test_non_finite_cost_cap_is_refused(nexus, compiled):
    with pytest.raises(HardwareError) as exc:
        QuantinuumNativeProvider().submit(compiled, 10, "H2-1E", confirm=True, max_cost=float("inf"))
    assert exc.value.code in ("QM-E956",)
    assert nexus.requests() == []
