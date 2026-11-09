"""IonQ plugin, Python fallback (qiskit-ionq >= 1.1.1, IonQ cloud API v0.4).

Not the default. The default IonQ client is native Mojo (qumo_hardware/ionq.py over
qumo/exec/ionq_api.mojo). Select this one with `--client python`, QUMO_HW_CLIENT=python
or QUMO_HW_CLIENT_IONQ=python; it needs `pip install qiskit-ionq`.

* Credentials: only the IONQ_API_KEY environment variable. The key is passed to
  qiskit-ionq explicitly (`IonQProvider(token=...)`), so the SDK's fallback
  lookup, which also reads a `.env` file in the working directory, is never
  used. This module never prints, logs or stores the key: it is not in
  JobHandle.meta, not in the job store and not in error messages.
* Backends: `ionq_simulator:<noise model>` as in the native client; QPU submission
  is not enabled.
* Circuit: QUMO's compiled OpenQASM 2 -> qiskit.QuantumCircuit (qasm_bridge, 1:1).
  No client-side transpile; IonQ compiles the QIS gates to its native set.
* Results: `job.get_counts()` of qiskit-ionq is already in Qiskit / QUMO bit order
  (the SDK maps the histogram through qiskit_ionq.ionq_job.map_output). Counts are
  sampled by the SDK with numpy, so they differ from the native client's draws.
* Cost: unknown (IonQ publishes no simulator price).
"""

from __future__ import annotations

import os
import warnings
from typing import Any

from .base import (
    CANCELLED,
    DONE,
    ERROR,
    QUEUED,
    RUNNING,
    UNKNOWN,
    JobHandle,
    check_confirmed,
    new_local_job_id,
)
from .compile import CompiledCircuit
from .errors import (
    E_CREDENTIALS,
    E_INVALID_PAYLOAD,
    E_JOB,
    E_LIMIT,
    E_SDK_MISSING,
    E_UNKNOWN_BACKEND,
    HINT_SHOTS,
    HardwareError,
    safe_exc_text,
)
from .ionq import (
    API_KEY_VAR,
    API_SIMULATOR,
    API_URL_VAR,
    MAX_SHOTS,
    SDK_BACKEND,
    SIMULATOR_QUBITS,
    IonQProvider,
    _parse_backend,
)
from .qasm_bridge import to_qiskit

KEY_PAGE = "https://cloud.ionq.com/settings/keys"

# qiskit-ionq job status names (qiskit JobStatus) -> QUMO states.
_STATUS = {
    "INITIALIZING": QUEUED,
    "QUEUED": QUEUED,
    "VALIDATING": QUEUED,
    "RUNNING": RUNNING,
    "DONE": DONE,
    "ERROR": ERROR,
    "CANCELLED": CANCELLED,
}


def _sdk():
    try:
        import qiskit_ionq
    except ImportError:
        raise HardwareError(E_SDK_MISSING, "qiskit-ionq is not installed (Python fallback client for IonQ)",
                            "pip install 'qiskit-ionq>=1.1.1', or use the default native client "
                            "(drop --client python / QUMO_HW_CLIENT=python)") from None
    return qiskit_ionq


def _sdk_version() -> str:
    from importlib import metadata

    try:
        return metadata.version("qiskit-ionq")
    except metadata.PackageNotFoundError:
        return "unknown"


def _api_key() -> str:
    """IONQ_API_KEY from the environment. The value is only ever passed to the SDK."""
    key = os.environ.get(API_KEY_VAR, "").strip()
    if not key:
        raise HardwareError(
            E_CREDENTIALS,
            f"IonQ API key not found: the {API_KEY_VAR} environment variable is not set",
            f"export {API_KEY_VAR}=<your IonQ API key> in the shell that runs QUMO (create one at {KEY_PAGE})",
        )
    return key


def counts_from_sdk(counts: dict, num_clbits: int) -> dict[str, int]:
    """qiskit-ionq counts (keys already in Qiskit bit order) -> QUMO counts over the user's clbits."""
    out: dict[str, int] = {}
    for key, value in dict(counts).items():
        bits = str(key).replace(" ", "")
        if len(bits) != num_clbits or set(bits) - {"0", "1"}:
            raise HardwareError(E_JOB, f"IonQ result key '{key}' has {len(bits)} bits, expected {num_clbits}",
                                "report the circuit with --show-qasm; the result layout does not match the payload")
        out[bits] = out.get(bits, 0) + int(value)
    return out


class IonQSdkProvider(IonQProvider):
    """Same backends, target and cost as the native client; submission through qiskit-ionq."""

    client = "python"

    def __init__(self, **_: Any):
        self._sdk_obj: Any = None  # qiskit-ionq backend object, built on first use

    # ------------------------------------------------------------ SDK access
    def _sdk_backend(self):
        """The qiskit-ionq simulator backend (QIS gateset), authenticated with IONQ_API_KEY."""
        if self._sdk_obj is None:
            sdk = _sdk()
            key = _api_key()
            from qiskit_ionq.exceptions import IonQTranspileLevelWarning

            try:
                with warnings.catch_warnings():
                    # QUMO never calls transpile(); the SDK's transpile-level warning does not apply.
                    warnings.filterwarnings("ignore", category=IonQTranspileLevelWarning)
                    provider = sdk.IonQProvider(token=key, url=os.environ.get(API_URL_VAR) or None)
                    self._sdk_obj = provider.get_backend(SDK_BACKEND, gateset="qis")
            except Exception as exc:
                raise HardwareError(E_UNKNOWN_BACKEND,
                                    f"IonQ backend '{SDK_BACKEND}' is not available ({safe_exc_text(exc)})",
                                    "check IONQ_API_KEY and the network access to the IonQ API") from None
        return self._sdk_obj

    def _retrieve(self, job: JobHandle):
        return self._sdk_backend().retrieve_job(job.job_id)

    # ------------------------------------------------------------ Provider API
    def _online_backends(self):
        """The fallback lists the same public endpoint through the native client (no key sent)."""
        return super()._online_backends()

    def prepare(self, compiled: CompiledCircuit, backend: str):
        """CompiledCircuit -> qiskit.QuantumCircuit (1:1 translation, no client-side transpile)."""
        _parse_backend(backend)
        qc = to_qiskit(compiled.qasm)
        report = {"passes": [], "client": "python (qiskit-ionq)",
                  "translation": "1:1 OpenQASM 2 -> qiskit.QuantumCircuit; IonQ compiles the QIS gates to its "
                                 "native set (gpi, gpi2, ms) on its side"}
        self.validate(qc, backend)
        return qc, report

    def validate(self, payload, backend: str) -> dict:
        _parse_backend(backend)
        if isinstance(payload, str):
            payload = to_qiskit(payload)
        accepted = set(self._sdk_backend().basis_gates) | {"measure", "reset", "barrier"}
        names = {inst.operation.name for inst in payload.data}
        bad = sorted(names - accepted)
        if bad:
            raise HardwareError(E_INVALID_PAYLOAD, f"the IonQ QIS gateset does not accept {bad}",
                                "decompose these gates in the circuit, or use a basis of IONQ_BASIS gates")
        if payload.num_qubits > SIMULATOR_QUBITS:
            raise HardwareError(E_LIMIT, f"the circuit has {payload.num_qubits} qubits; the IonQ simulator "
                                f"has {SIMULATOR_QUBITS}", "reduce the circuit width")
        return {"native": False, "checks": ["gate names in the IonQ QIS gateset (qiskit_ionq backend.basis_gates)",
                                            f"qubits <= {SIMULATOR_QUBITS}"]}

    def submit(self, compiled: CompiledCircuit, shots: int, backend: str, seed: int | None = None,
               payload=None, report: dict | None = None, max_cost: float | None = None, confirm: bool = False,
               **_: Any) -> JobHandle:
        check_confirmed(self, backend, confirm)
        noise = _parse_backend(backend)
        if shots < 1 or shots > MAX_SHOTS:
            raise HardwareError(E_LIMIT, f"shots must be in 1..{MAX_SHOTS} for {backend}", HINT_SHOTS)
        if payload is None or isinstance(payload, str):
            payload, report = self.prepare(compiled, backend)
        else:
            self.validate(payload, backend)
        options: dict[str, Any] = {"shots": int(shots), "noise_model": noise}
        if seed is not None:
            options["sampler_seed"] = int(seed)
        meta = compiled.result_meta()
        meta.update({"sdk_backend": SDK_BACKEND, "api_backend": API_SIMULATOR, "noise_model": noise,
                     "sampler_seed": seed, "finishing": report or {}, "max_cost": max_cost,
                     "client": "python (qiskit-ionq)", "qiskit_ionq": _sdk_version()})
        try:
            job = self._sdk_backend().run(payload, **options)
        except HardwareError:
            raise
        except Exception as exc:
            raise HardwareError(E_JOB, f"IonQ refused the job ({safe_exc_text(exc)})",
                                "nothing was run; check IONQ_API_KEY, the noise model and the shot count") from None
        job_id = str(job.job_id())
        return JobHandle(job_id or new_local_job_id("ionq"), self.name, backend, shots, False, QUEUED, meta=meta)

    def status(self, job: JobHandle) -> str:
        if job.local:
            return job.status
        _sdk()
        from qiskit_ionq.exceptions import IonQJobFailureError, IonQJobStateError

        try:
            st = self._retrieve(job).status()
        except IonQJobFailureError:
            return ERROR
        except IonQJobStateError:
            return CANCELLED
        except HardwareError:
            raise
        except Exception as exc:
            raise HardwareError(E_JOB, f"cannot read the status of IonQ job '{job.job_id}' ({safe_exc_text(exc)})",
                                f"check {API_KEY_VAR} and the network access, then retry") from None
        name = getattr(st, "name", None) or str(st)
        return _STATUS.get(str(name).upper(), UNKNOWN)

    def result(self, job: JobHandle) -> dict[str, int]:
        if job.local:
            return dict(job.counts or {})
        st = self.status(job)
        if st != DONE:
            raise HardwareError(E_JOB, f"IonQ job '{job.job_id}' is {st}")
        try:
            raw = self._retrieve(job).get_counts()
        except HardwareError:
            raise
        except Exception as exc:
            raise HardwareError(E_JOB, f"cannot fetch the result of IonQ job '{job.job_id}' "
                                f"({safe_exc_text(exc)})", "retry later; IonQ keeps results of finished jobs") from None
        return counts_from_sdk(raw, int(job.meta["num_clbits"]))
