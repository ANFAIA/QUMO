"""IBM plugin: cost estimate, mocked remote submission, credential errors."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from conftest import BELL, ghz, needs_ibm

from qumo_hardware import runner
from qumo_hardware.errors import (
    E_CONFIRM,
    E_CREDENTIALS,
    E_INVALID_PAYLOAD,
    E_JOB,
    E_LIMIT,
    E_UNKNOWN_BACKEND,
    HardwareError,
)

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
        self.max_execution_time = self.options.max_execution_time
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
    assert out["server_dry_run"] is True and out["mock_results"] is True and "mock" in out["warning"]
    from qumo_hardware import jobs

    assert jobs.load("ibm-dry-1").meta["server_dry_run"] is True
    # Results of a server-side dry run are flagged as mock data wherever they are read.
    prov._service = SimpleNamespace(job=lambda job_id: SimpleNamespace(
        status=lambda: "DONE", result=lambda: _primitive_result({"10": 100}, 2)))
    res = runner.job_result("ibm-dry-1", provider=prov)
    assert res["mock_results"] is True and "not measurements" in res["warning"]


def test_wait_timeout_returns_submitted_job(monkeypatch):
    import qiskit_ibm_runtime.executor_sampler as es

    monkeypatch.setattr(es, "Sampler", StubSampler)
    queued = SimpleNamespace(status=lambda: "QUEUED", result=lambda: None)
    prov = _provider_with_mock_remote(service=SimpleNamespace(job=lambda job_id: queued))
    out = runner.run(BELL, prov, "ibm_mock_torino", shots=10, confirm=True, wait=True, timeout=0.0, poll=0.0)
    assert out["submitted"] and out["status"] == "QUEUED" and "result ibm-remote-job-1" in out["next"]
    assert out.get("counts") is None


def test_provider_submit_refuses_remote_without_confirm(monkeypatch):
    import qiskit_ibm_runtime.executor_sampler as es

    StubSampler.instances.clear()
    monkeypatch.setattr(es, "Sampler", StubSampler)
    prov = _provider_with_mock_remote()
    plan = runner.plan(BELL, prov, "ibm_mock_torino", shots=10)
    with pytest.raises(HardwareError) as ei:
        prov.submit(plan.compiled, 10, "ibm_mock_torino", payload=plan.payload)
    assert ei.value.code == E_CONFIRM and not StubSampler.instances


def test_server_side_cap_max_execution_time(monkeypatch):
    import qiskit_ibm_runtime.executor_sampler as es

    from qumo_hardware import jobs
    from qumo_hardware.ibm import SERVER_CAP_FACTOR, SERVER_CAP_MIN_S

    StubSampler.instances.clear()
    monkeypatch.setattr(es, "Sampler", StubSampler)
    prov = _provider_with_mock_remote()
    out = runner.run(ghz(3), prov, "ibm_mock_torino", shots=50_000, confirm=True, wait=False)
    est = out["cost"]["amount"]
    cap = StubSampler.instances[-1].max_execution_time
    assert cap == max(SERVER_CAP_MIN_S, -(-SERVER_CAP_FACTOR * est // 1)) and cap >= est
    assert jobs.load("ibm-remote-job-1").meta["max_execution_time"] == cap
    runner.run(BELL, prov, "ibm_mock_torino", shots=100, confirm=True, wait=False, max_cost=12.5)
    assert StubSampler.instances[-1].max_execution_time == 13  # --max-cost (QPU seconds), rounded up


def test_fake_backend_refuses_too_many_active_qubits(monkeypatch):
    import qiskit_ibm_runtime.executor_sampler as es

    from qumo_hardware.ibm import LOCAL_MAX_ACTIVE_QUBITS

    def boom(*a, **k):
        raise AssertionError("Aer must not be started")

    monkeypatch.setattr(es, "Sampler", boom)
    n = LOCAL_MAX_ACTIVE_QUBITS + 2
    wide = ghz(n).replace("h q[0];", "h q[0];\nt q[1];")  # non-Clifford
    with pytest.raises(HardwareError) as ei:
        runner.run(wide, "ibm", "fake_torino", shots=10)
    e = ei.value
    assert e.code == E_LIMIT and f"{n} active qubits" in e.message
    assert "--shots" not in e.hint and "emulator" in e.hint


def test_fractional_flag_on_backend_without_fractional_gates_is_reported():
    from qumo_hardware.ibm import IBMProvider

    plan = runner.plan(BELL, IBMProvider(use_fractional_gates=True), "fake_torino", shots=10)
    assert any("--fractional ignored" in n and "fake_fractional" in n for n in plan.notes)
    assert "rzz" not in plan.compiled.target.basis
    ok = runner.plan(BELL, IBMProvider(use_fractional_gates=True), "fake_fractional", shots=10)
    assert not ok.notes and "rzz" in ok.compiled.target.basis


def test_fake_fractional_is_listed():
    from qumo_hardware.ibm import IBMProvider

    rows = {b.name: b for b in IBMProvider().list_backends()}
    assert "fake_fractional" in rows and "--fractional" in rows["fake_fractional"].notes
    assert "fake_torino" in rows


class _Boom(Exception):
    pass


def test_sdk_errors_become_coded_errors(monkeypatch):
    """IBM SDK failures other than missing credentials map onto QM-E95x, secrets redacted."""
    import qiskit_ibm_runtime.executor_sampler as es

    from qumo_hardware.ibm import IBMProvider

    secret = "Bearer " + "s3cr3tT0k3n" * 5
    def failing_backend(name, use_fractional_gates=False):
        raise _Boom(f"HTTP 500 {secret}")

    svc = SimpleNamespace(backend=failing_backend)
    with pytest.raises(HardwareError) as ei:
        IBMProvider(service=svc).backend("ibm_fez")
    assert ei.value.code == E_UNKNOWN_BACKEND and "s3cr3t" not in str(ei.value)

    class RejectingSampler(StubSampler):
        def run(self, pubs, shots=None, dry_run=False):
            raise _Boom("quota exhausted")

    monkeypatch.setattr(es, "Sampler", RejectingSampler)
    prov = _provider_with_mock_remote()
    with pytest.raises(HardwareError) as ei:
        runner.run(BELL, prov, "ibm_mock_torino", shots=10, confirm=True, wait=False)
    assert ei.value.code == E_JOB and "quota exhausted" in ei.value.message

    class IBMInputValueError(Exception):
        pass

    class InvalidSampler(StubSampler):
        def run(self, pubs, shots=None, dry_run=False):
            raise IBMInputValueError("bad circuit")

    monkeypatch.setattr(es, "Sampler", InvalidSampler)
    with pytest.raises(HardwareError) as ei:
        runner.run(BELL, prov, "ibm_mock_torino", shots=10, confirm=True, wait=False)
    assert ei.value.code == E_INVALID_PAYLOAD

    monkeypatch.setattr(es, "Sampler", StubSampler)
    flaky = SimpleNamespace(status=lambda: (_ for _ in ()).throw(_Boom("timeout")), result=lambda: None)
    prov._service = SimpleNamespace(job=lambda job_id: flaky)
    out = runner.run(BELL, prov, "ibm_mock_torino", shots=10, confirm=True, wait=False)
    with pytest.raises(HardwareError) as ei:
        runner.job_status(out["job_id"], provider=prov)
    assert ei.value.code == E_JOB and "timeout" in ei.value.message
    prov._service = SimpleNamespace(job=lambda job_id: (_ for _ in ()).throw(_Boom("dns")))
    with pytest.raises(HardwareError) as ei:
        runner.job_status(out["job_id"], provider=prov)
    assert ei.value.code == E_JOB


def test_job_ending_in_error_is_reported(monkeypatch):
    import qiskit_ibm_runtime.executor_sampler as es

    monkeypatch.setattr(es, "Sampler", StubSampler)
    failed = SimpleNamespace(status=lambda: "ERROR", result=lambda: None)
    prov = _provider_with_mock_remote(service=SimpleNamespace(job=lambda job_id: failed))
    out = runner.run(BELL, prov, "ibm_mock_torino", shots=10, confirm=True, wait=True, timeout=5, poll=0.0)
    assert out["submitted"] and out["status"] == "ERROR" and "ERROR" in out["warning"]
    assert out.get("counts") is None
