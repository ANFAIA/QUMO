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


def test_job_ids_never_share_a_file():
    """`a/b` and `a_b` are two jobs: one used to overwrite the other."""
    from qumo_hardware import jobs as job_store
    from qumo_hardware.base import QUEUED, JobHandle

    for jid, shots in (("a/b", 1), ("a_b", 2), ("a%2Fb", 3), ("a b", 4)):
        job_store.save(JobHandle(jid, "local", "qumo-sim", shots, True, QUEUED))
    assert [job_store.load(j).shots for j in ("a/b", "a_b", "a%2Fb", "a b")] == [1, 2, 3, 4]
    assert all(p.parent == job_store.store_dir() for p in map(job_store._path, ("../x", "a/b", "..")))


def test_job_stored_under_the_old_file_name_is_still_found():
    import json

    from qumo_hardware import jobs as job_store
    from qumo_hardware.base import QUEUED, JobHandle
    from qumo_hardware.errors import E_UNKNOWN_JOB, HardwareError

    old = job_store._legacy_path("x:1")
    old.write_text(json.dumps(JobHandle("x:1", "local", "qumo-sim", 7, True, QUEUED).to_dict()))
    assert job_store.load("x:1").shots == 7
    with pytest.raises(HardwareError) as exc:  # same old file name, another job
        job_store.load("x/1")
    assert exc.value.code == E_UNKNOWN_JOB


def test_job_file_options_are_limited_to_the_declared_ones():
    from qumo_hardware.base import QUEUED, JobHandle
    from qumo_hardware.errors import E_JOB, HardwareError

    def opts(provider, stored, local=False):
        return runner._job_file_options(JobHandle("j", provider, "b", 1, local, QUEUED, meta={"provider_options": stored}))

    assert opts("quantinuum", {"project": "p", "client": "python", "max_cost": 1e9, "seed": 3}) == {
        "project": "p", "client": "python"}
    assert opts("ibm", {"instance": "crn:v1:x", "account_name": "work", "channel": None, "use_fractional_gates": True,
                        "server_dry_run": True, "sampler_options": {"a": 1}, "client": "native"}) == {
        "instance": "crn:v1:x", "account_name": "work", "channel": None, "use_fractional_gates": True,
        "client": "native"}
    assert opts("ionq", {}) == {"client": "python"} and opts("local", {}, local=True) == {}
    for provider, stored in (("ibm", {"service": {}}), ("ibm", {"url": "https://other.example"}),
                             ("ionq", {"api_key": "k"}), ("qbraid", {"Token": "t", "client": "native"}), ("ibm", {"instance": ["x"]}),
                             ("ionq", {"client": 3}), ("speqtrum", ["url"])):
        with pytest.raises(HardwareError) as exc:
            opts(provider, stored)
        assert exc.value.code == E_JOB
