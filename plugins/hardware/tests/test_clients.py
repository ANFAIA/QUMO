"""Client selection: native Mojo by default, vendor Python SDK only as an explicit fallback."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from conftest import BELL, have

from qumo_hardware import runner
from qumo_hardware.errors import E_CONFIRM, E_UNKNOWN_BACKEND, HardwareError
from qumo_hardware.ionq import IonQProvider
from qumo_hardware.registry import NATIVE, PYTHON, get_provider, requested_client, resolve_client

ARIA1 = "ionq_simulator:aria-1"
KEY = "test-key-not-real-0123"


@pytest.fixture(autouse=True)
def no_client_env(monkeypatch):
    for var in ("QUMO_HW_CLIENT", "QUMO_HW_CLIENT_IONQ", "QUMO_HW_CLIENT_IBM", "QUMO_HW_CLIENT_SPEQTRUM"):
        monkeypatch.delenv(var, raising=False)


def test_native_is_the_default():
    assert requested_client("ionq") == "native"
    prov = get_provider("ionq")
    assert type(prov) is IonQProvider and prov.client == "native"


def test_python_fallback_by_argument_and_by_environment(monkeypatch):
    from qumo_hardware.ionq_sdk import IonQSdkProvider

    assert type(get_provider("ionq", client="python")) is IonQSdkProvider
    monkeypatch.setenv("QUMO_HW_CLIENT", "python")
    assert type(get_provider("ionq")) is IonQSdkProvider
    monkeypatch.setenv("QUMO_HW_CLIENT_IONQ", "native")  # per-provider variable wins over the global one
    assert type(get_provider("ionq")) is IonQProvider
    assert get_provider("ionq", client="python").client == "python"  # the argument wins over both


def test_unknown_client_is_refused():
    with pytest.raises(HardwareError) as exc:
        requested_client("ionq", "rust")
    assert exc.value.code == E_UNKNOWN_BACKEND


def test_local_has_no_python_fallback():
    with pytest.raises(HardwareError) as exc:
        get_provider("local", client="python")
    assert exc.value.code == E_UNKNOWN_BACKEND
    assert get_provider("local").client == "native"


@pytest.mark.parametrize("name", sorted(set(PYTHON) - set(NATIVE)))
def test_providers_not_ported_yet_use_their_python_client(name):
    _, used = resolve_client(name, "native")
    assert used == "python"


def test_job_store_keeps_the_client_used(monkeypatch):
    """A job submitted with the fallback is read back with the fallback."""
    from qumo_hardware.ionq_sdk import IonQSdkProvider

    seen = {}

    def fake_submit(self, compiled, shots, backend, **kw):
        from qumo_hardware.base import QUEUED, JobHandle

        seen["client"] = self.client
        return JobHandle("j-1", "ionq", backend, shots, False, QUEUED, meta=compiled.result_meta())

    monkeypatch.setattr(IonQSdkProvider, "submit", fake_submit)
    monkeypatch.setattr(IonQSdkProvider, "prepare", lambda self, c, b: (c.qasm, {}))
    monkeypatch.setattr(IonQSdkProvider, "validate", lambda self, p, b: {"checks": []})
    out = runner.run(BELL, get_provider("ionq", client="python"), ARIA1, shots=10, confirm=True, wait=False)
    assert seen["client"] == "python" and out["job_id"] == "j-1"
    job = runner.job_store.load("j-1")
    assert job.meta["provider_options"]["client"] == "python"
    assert type(runner._provider_for(job)) is IonQSdkProvider


def test_cli_accepts_the_client_flag(monkeypatch, capsys):
    from qumo_hardware import cli

    rc = cli.main(["backends", "-p", "ionq", "--client", "native"])
    assert rc == 0 and "ionq_simulator:aria-1" in capsys.readouterr().out


# ----------------------------------------------------------------- the IonQ Python fallback itself
@pytest.mark.skipif(not have("qiskit_ionq"), reason="qiskit-ionq not installed (Python fallback only)")
class TestIonQSdkFallback:
    @pytest.fixture
    def sdk(self, monkeypatch):
        from qiskit.providers import JobStatus

        from qumo_hardware.ionq_sdk import IonQSdkProvider

        monkeypatch.setenv("IONQ_API_KEY", KEY)

        class FakeJob:
            def __init__(self, jid, status=JobStatus.DONE, counts=None):
                self._id, self._status, self._counts = jid, status, counts or {}

            def job_id(self):
                return self._id

            def status(self):
                return self._status

            def get_counts(self):
                return self._counts

        fake = SimpleNamespace(basis_gates=["h", "cx", "rz", "rzz", "x"], run_calls=[], jobs={})

        def run(circuit, **options):
            fake.run_calls.append((circuit, options))
            job = FakeJob("sdk-job-1", JobStatus.QUEUED)
            fake.jobs[job.job_id()] = job
            return job

        fake.run = run
        fake.retrieve_job = lambda jid: fake.jobs[jid]
        fake.FakeJob, fake.JobStatus = FakeJob, JobStatus
        monkeypatch.setattr(IonQSdkProvider, "_sdk_backend", lambda self: fake)
        return fake

    def test_submit_status_result_through_the_sdk(self, sdk):
        from qumo_hardware.base import DONE, QUEUED
        from qumo_hardware.compile import compile_for_target

        prov = get_provider("ionq", client="python")
        compiled = compile_for_target(BELL, prov.target(ARIA1), level=2, seed=1)
        with pytest.raises(HardwareError) as exc:
            prov.submit(compiled, 100, ARIA1)
        assert exc.value.code == E_CONFIRM and sdk.run_calls == []
        job = prov.submit(compiled, 100, ARIA1, seed=3, confirm=True)
        assert job.status == QUEUED and job.meta["client"].startswith("python")
        circuit, options = sdk.run_calls[0]
        assert options == {"shots": 100, "noise_model": "aria-1", "sampler_seed": 3}
        assert dict(circuit.count_ops()) == {"h": 1, "cx": 1, "measure": 2}
        assert KEY not in json.dumps(job.to_dict(), default=str)
        sdk.jobs["sdk-job-1"] = sdk.FakeJob("sdk-job-1", sdk.JobStatus.DONE, {"00": 60, "11": 40})
        assert prov.status(job) == DONE
        assert prov.result(job) == {"00": 60, "11": 40}


def test_a_global_python_choice_leaves_native_only_providers_native(monkeypatch):
    monkeypatch.setenv("QUMO_HW_CLIENT", "python")
    assert get_provider("local").client == "native"
    monkeypatch.setenv("QUMO_HW_CLIENT_LOCAL", "python")
    with pytest.raises(HardwareError):
        get_provider("local")
