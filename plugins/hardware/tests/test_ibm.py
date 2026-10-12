"""IBM plugin: cost estimate, mocked remote submission, credential errors."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from conftest import BELL, ghz, needs_ibm

from qumo_hardware import runner
from qumo_hardware.errors import E_CONFIRM, E_CREDENTIALS, E_JOB, HardwareError

pytestmark = needs_ibm


def _provider_with_mock_remote(name="ibm_mock_torino", service=None):
    """IBMProvider whose 'remote' backend is a FakeTorino snapshot (no network)."""
    from qiskit_ibm_runtime.fake_provider import FakeTorino

    from qumo_hardware.ibm import IBMProvider

    prov = IBMProvider(service=service, instance="crn:v1:mock-instance")
    prov._backends[name] = FakeTorino()
    return prov


def test_qpu_time_cost_estimate(monkeypatch):
    prov = _provider_with_mock_remote()
    plan = runner.plan(ghz(4), prov, "ibm_mock_torino", shots=4000)
    cost = plan.cost
    assert cost.unit == "QPU-seconds" and cost.billable
    b = cost.breakdown
    assert b["rep_delay_s"] == pytest.approx(250e-6)
    assert 0 < b["circuit_duration_s"] < 1e-3
    expected = 2.0 + 4000 * (b["rep_delay_s"] + b["circuit_duration_s"])
    assert cost.amount == pytest.approx(expected, abs=1e-3)
    assert cost.usd == pytest.approx(expected * 1.60, abs=0.01)
    monkeypatch.setenv("QUMO_IBM_USD_PER_SECOND", "2.0")
    assert prov.estimate_cost(plan.compiled, 4000, "ibm_mock_torino").usd == pytest.approx(expected * 2.0, abs=0.01)
    assert plan.requires_confirmation


def test_fake_backend_cost_is_free():
    from qumo_hardware.ibm import IBMProvider

    plan = runner.plan(BELL, IBMProvider(), "fake_torino", shots=100)
    assert plan.cost.unit == "free" and not plan.requires_confirmation


class StubSampler:
    instances: list["StubSampler"] = []

    def __init__(self, mode=None, options=None):
        from qiskit_ibm_runtime.options_models.sampler import SamplerOptions

        self.mode = mode
        self.options = SamplerOptions()
        self.runs = []
        StubSampler.instances.append(self)

    def run(self, pubs, shots=None, dry_run=False):
        self.runs.append((list(pubs), shots))
        return SimpleNamespace(job_id=lambda: "ibm-remote-job-1")


def _primitive_result(counts, nbits):
    from qiskit.primitives.containers import BitArray, DataBin, PrimitiveResult, SamplerPubResult

    ba = BitArray.from_counts(counts, num_bits=nbits)
    return PrimitiveResult([SamplerPubResult(DataBin(c=ba, shape=ba.shape))])


def test_mocked_remote_submission_and_result(monkeypatch):
    import qiskit_ibm_runtime.executor_sampler as es
    from qiskit_ibm_runtime.utils.utils import is_isa_circuit

    StubSampler.instances.clear()
    monkeypatch.setattr(es, "Sampler", StubSampler)
    remote_job = SimpleNamespace(status=lambda: "DONE", result=lambda: _primitive_result({"11": 480, "00": 520}, 2))
    service = SimpleNamespace(job=lambda job_id: remote_job)
    prov = _provider_with_mock_remote(service=service)

    with pytest.raises(HardwareError) as ei:  # safety gate first
        runner.run(BELL, prov, "ibm_mock_torino", shots=1000)
    assert ei.value.code == E_CONFIRM and not StubSampler.instances

    out = runner.run(BELL, prov, "ibm_mock_torino", shots=1000, confirm=True, wait=False)
    assert out["submitted"] and out["job_id"] == "ibm-remote-job-1"
    (s,) = StubSampler.instances
    (pubs, shots), = s.runs
    assert shots == 1000 and len(pubs) == 1
    assert is_isa_circuit(pubs[0], prov.backend("ibm_mock_torino").target) == ""
    assert s.options.twirling.enable_gates is False and s.options.dynamical_decoupling.enable is False

    st = runner.job_status("ibm-remote-job-1", provider=prov)
    assert st["status"] == "DONE"
    res = runner.job_result("ibm-remote-job-1", provider=prov)
    assert res["counts"] == {"11": 480, "00": 520}
    from qumo_hardware import jobs

    rec = jobs.load("ibm-remote-job-1")
    assert rec.meta["provider_options"]["instance"] == "crn:v1:mock-instance"
    assert rec.meta["sampler"].endswith("executor_sampler.Sampler")


def test_result_width_mismatch_is_reported():
    from qumo_hardware.ibm import counts_from_primitive_result

    with pytest.raises(HardwareError) as ei:
        counts_from_primitive_result(_primitive_result({"101": 3}, 3), 2)
    assert ei.value.code == E_JOB


def test_missing_credentials_are_a_clear_error():
    from qumo_hardware.ibm import IBMProvider

    with pytest.raises(HardwareError) as ei:
        IBMProvider().service()
    e = ei.value
    assert e.code == E_CREDENTIALS
    assert "QISKIT_IBM_TOKEN" in e.hint and "save_account" in e.hint
    with pytest.raises(HardwareError) as ei:
        runner.run(BELL, "ibm", "ibm_torino", shots=100, confirm=True)
    assert ei.value.code == E_CREDENTIALS


def test_provider_transpile_comparison():
    from qumo_hardware.ibm import IBMProvider

    plan = runner.plan(ghz(5), IBMProvider(), "fake_torino", shots=100, provider_transpile=True)
    q, theirs = plan.comparison["qumo"], plan.comparison["provider"]
    assert q["two_qubit_gates"] == 4 and theirs["two_qubit_gates"] >= 4
    assert q["duration_s"] > 0 and theirs["duration_s"] > 0


def test_server_dry_run_flag_is_forwarded(monkeypatch):
    import qiskit_ibm_runtime.executor_sampler as es

    seen = {}

    class DrySampler(StubSampler):
        def run(self, pubs, shots=None, dry_run=False):
            seen["dry_run"] = dry_run
            return SimpleNamespace(job_id=lambda: "ibm-dry-1")

    monkeypatch.setattr(es, "Sampler", DrySampler)
    prov = _provider_with_mock_remote()
    prov.server_dry_run = True
    out = runner.run(BELL, prov, "ibm_mock_torino", shots=100, confirm=True, wait=False)
    assert seen["dry_run"] is True and out["job_id"] == "ibm-dry-1"
    from qumo_hardware import jobs

    assert jobs.load("ibm-dry-1").meta["server_dry_run"] is True


def test_wait_timeout_returns_submitted_job(monkeypatch):
    import qiskit_ibm_runtime.executor_sampler as es

    monkeypatch.setattr(es, "Sampler", StubSampler)
    queued = SimpleNamespace(status=lambda: "QUEUED", result=lambda: None)
    prov = _provider_with_mock_remote(service=SimpleNamespace(job=lambda job_id: queued))
    out = runner.run(BELL, prov, "ibm_mock_torino", shots=10, confirm=True, wait=True, timeout=0.0, poll=0.0)
    assert out["submitted"] and out["status"] == "QUEUED" and "result ibm-remote-job-1" in out["next"]
    assert out.get("counts") is None
