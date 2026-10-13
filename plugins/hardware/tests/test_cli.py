"""CLI: dry-run JSON, refusal without --yes, coded errors without stack traces."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
from conftest import BELL, ROOT, ghz, needs_ibm, needs_pytket, needs_qnexus, x_on

from qumo_hardware.cli import EXIT_CONFIRM, EXIT_ERR, EXIT_INTERRUPTED, EXIT_OK, EXIT_USAGE, main


@pytest.fixture
def qasm(tmp_path):
    def write(text, name="c.qasm"):
        p = tmp_path / name
        p.write_text(text)
        return str(p)
    return write


def run_cli(capsys, *argv):
    code = main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_local_run_json_and_job_store(capsys, qasm):
    code, out, err = run_cli(capsys, "run", qasm(x_on(3, 0)), "--backend", "qumo-sim:line:3/sc_cz", "--shots", "50", "--json")
    assert code == EXIT_OK, err
    d = json.loads(out)
    assert d["counts"] == {"001": 50} and d["submitted"] and not d["dry_run"]
    assert d["bit_order"].startswith("leftmost character = highest classical bit")
    job_id = d["job_id"]
    code, out, _ = run_cli(capsys, "status", job_id, "--json")
    assert code == EXIT_OK and json.loads(out)["status"] == "DONE"
    code, out, _ = run_cli(capsys, "result", job_id, "--json")
    assert code == EXIT_OK and json.loads(out)["counts"] == {"001": 50}
    code, out, _ = run_cli(capsys, "jobs", "--json")
    assert job_id in [j["job_id"] for j in json.loads(out)["jobs"]]


@needs_pytket
def test_dry_run_json_for_hardware(capsys, qasm):
    code, out, err = run_cli(capsys, "run", qasm(ghz(4)), "--provider", "quantinuum", "--backend", "H2-1",
                             "--shots", "200", "--dry-run", "--json")
    assert code == EXIT_OK, err
    d = json.loads(out)
    assert d["dry_run"] is True and d["submitted"] is False
    assert d["provider"] == "quantinuum" and d["backend"] == "H2-1" and d["shots"] == 200
    assert d["requires_confirmation"] and not d["local"]
    assert d["cost"]["unit"] == "HQC" and d["cost"]["amount"] > 5
    assert d["compiled"]["num_logical"] == 4 and d["validation"]["native"]
    assert "job_id" not in d


@needs_pytket
def test_refuses_to_submit_without_yes(capsys, qasm, monkeypatch):
    from qumo_hardware.quantinuum import QuantinuumProvider

    def boom(*a, **k):
        raise AssertionError("submit must not be called without --yes")

    monkeypatch.setattr(QuantinuumProvider, "submit", boom)
    code, out, err = run_cli(capsys, "run", qasm(BELL), "--provider", "quantinuum", "--backend", "Helios-1",
                             "--shots", "100", "--json")
    assert code == EXIT_CONFIRM
    e = json.loads(out)["error"]
    assert e["code"] == "QM-E957" and "--yes" in e["hint"]
    plan = e["details"]["plan"]
    assert plan["backend"] == "Helios-1" and plan["shots"] == 100 and plan["cost"]["billable"]
    # Human mode prints the plan (backend, shots, cost) before refusing.
    code, out, err = run_cli(capsys, "run", qasm(BELL), "--provider", "quantinuum", "--backend", "Helios-1",
                             "--shots", "100")
    assert code == EXIT_CONFIRM
    assert "Helios-1" in out and "shots" in out and "cost estimate" in out
    assert "QM-E957" in err


@needs_ibm
def test_missing_ibm_credentials_no_traceback(capsys, qasm):
    code, out, err = run_cli(capsys, "run", qasm(BELL), "--provider", "ibm", "--backend", "ibm_torino", "--json")
    assert code == EXIT_ERR
    e = json.loads(out)["error"]
    assert e["code"] == "QM-E952" and "QISKIT_IBM_TOKEN" in e["hint"]
    code, out, err = run_cli(capsys, "run", qasm(BELL), "--provider", "ibm", "--backend", "ibm_torino", "--yes")
    assert code == EXIT_ERR and err.startswith("error: QM-E952") and "Traceback" not in err + out


@needs_pytket
def test_estimate_never_submits(capsys, qasm, monkeypatch):
    from qumo_hardware.quantinuum import QuantinuumProvider

    monkeypatch.setattr(QuantinuumProvider, "submit", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    code, out, err = run_cli(capsys, "estimate", qasm(ghz(3)), "--provider", "quantinuum", "--backend", "H2-1",
                             "--shots", "1000", "--json")
    assert code == EXIT_OK, err
    assert json.loads(out)["cost"]["formula"].startswith("HQC = 5 + C")


def test_unknown_job_and_usage(capsys):
    code, out, _ = run_cli(capsys, "result", "nope-123", "--json")
    assert code == EXIT_ERR and json.loads(out)["error"]["code"] == "QM-E958"
    assert main(["run"]) == EXIT_USAGE
    capsys.readouterr()


def test_missing_file(capsys):
    code, out, _ = run_cli(capsys, "run", "/nonexistent/x.qasm", "--json")
    assert code == EXIT_ERR and json.loads(out)["error"]["code"] == "QM-E954"


def test_errors_catalogue(capsys):
    code, out, _ = run_cli(capsys, "errors", "--json")
    codes = [e["code"] for e in json.loads(out)["errors"]]
    assert codes == [f"QM-E95{i}" for i in range(10)]


@needs_pytket
@needs_ibm
def test_backends_and_target(capsys):
    code, out, _ = run_cli(capsys, "backends", "--provider", "quantinuum", "--json")
    names = {b["name"] for b in json.loads(out)["backends"]}
    assert {"H2-1", "H2-1E", "H2-1SC", "Helios-1", "H2-1LE", "selene"} <= names
    code, out, _ = run_cli(capsys, "backends", "--provider", "ibm", "--json")
    assert "fake_torino" in {b["name"] for b in json.loads(out)["backends"]}
    code, out, _ = run_cli(capsys, "target", "--provider", "ibm", "--backend", "fake_torino", "--json")
    d = json.loads(out)
    assert d["num_qubits"] == 133 and "edges" not in d and d["num_edges"] == 150
    code, out, _ = run_cli(capsys, "target", "--provider", "ibm", "--backend", "fake_torino", "--json", "--full")
    assert len(json.loads(out)["edges"]) == 150


def test_module_entry_point():
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    r = subprocess.run([sys.executable, "-m", "qumo_hardware", "errors", "--json"], capture_output=True, text=True,
                       env=env, timeout=120)
    assert r.returncode == 0 and "QM-E957" in r.stdout


@needs_qnexus
@pytest.mark.parametrize("final", ["ERROR", "CANCELLED"])
def test_run_wait_exits_nonzero_when_job_fails(capsys, qasm, nexus, final):
    nexus(status=final)
    code, out, err = run_cli(capsys, "run", qasm(BELL), "-p", "quantinuum", "-b", "H2-1", "--shots", "10",
                             "--yes", "--json", "--timeout", "5")
    assert code == EXIT_ERR
    d = json.loads(out)
    assert d["submitted"] and d["job_id"] == "nexus-job-123" and d["status"] == final
    assert final in d["warning"] and "result nexus-job-123" in d["next"]


@needs_qnexus
def test_ctrl_c_while_waiting_prints_job_id(capsys, qasm, nexus):
    nexus(status=KeyboardInterrupt)
    code, out, err = run_cli(capsys, "run", qasm(BELL), "-p", "quantinuum", "-b", "H2-1", "--shots", "10", "--yes")
    assert code == EXIT_INTERRUPTED
    assert "submitted job nexus-job-123" in err
    assert "interrupted: job nexus-job-123" in err and "result nexus-job-123 --wait" in err
    from qumo_hardware import jobs

    assert jobs.load("nexus-job-123").job_id == "nexus-job-123"  # saved before waiting


@needs_ibm
def test_fractional_note_and_fake_fractional_listed(capsys, qasm):
    code, out, err = run_cli(capsys, "estimate", qasm(BELL), "-p", "ibm", "-b", "fake_torino", "--fractional")
    assert code == EXIT_OK, err
    assert "note" in out and "--fractional ignored" in out
    code, out, _ = run_cli(capsys, "backends", "--provider", "ibm", "--json")
    rows = {b["name"]: b for b in json.loads(out)["backends"]}
    assert "fake_fractional" in rows and "--fractional" in rows["fake_fractional"]["notes"]


@needs_ibm
def test_no_fake_nighthawk_warning_on_ibm_commands():
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    r = subprocess.run([sys.executable, "-m", "qumo_hardware", "target", "-p", "ibm", "-b", "fake_torino", "--json"],
                       capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0, r.stderr
    assert "nighthawk" not in r.stderr and "UserWarning" not in r.stderr
