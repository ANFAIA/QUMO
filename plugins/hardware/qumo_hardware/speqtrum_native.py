"""SpeQtrum provider, native client (default): the protocol runs in Mojo (qumo/exec/speqtrum_api.mojo).

Login, device list, execute, job status and result decoding (base64, refusal rule, strict JSON)
all run in the qumo_py extension over QUMO's own HTTP/TLS stack (qumo/net). This adapter keeps
only configuration (credentials and URL from the environment or options, checked before any
request), the explicit targets and the plugin API. The access token never leaves Mojo.

The Python implementation in speqtrum.py stays as the fallback: `--client python`,
QUMO_HW_CLIENT=python or QUMO_HW_CLIENT_SPEQTRUM=python.
"""

from __future__ import annotations

import warnings
from typing import Any

from .base import QUEUED, BackendInfo, JobHandle, check_confirmed
from .compile import CompiledCircuit, load_qumo_py
from .errors import E_INVALID_PAYLOAD, E_JOB, E_LIMIT, HINT_SHOTS, HardwareError
from .speqtrum import SpeQtrumProvider, UnsafeFormatWarning

NATIVE = "qumo/exec/speqtrum_api.mojo"


def _native_error(exc: Exception) -> HardwareError:
    """Mojo 'QM-E9xx: message[ Hint: ...]' -> HardwareError; a refused document also warns."""
    text = str(exc)
    code, _, msg = text.partition(": ")
    if not (code.startswith("QM-E") and len(code) == 7):
        code, msg = E_JOB, text
    msg, _, hint = msg.partition(" Hint: ")
    if code == E_INVALID_PAYLOAD and "was not loaded" in msg:
        warnings.warn(msg.replace(" was not loaded: it contains ", " contains ") + "; QUMO did not load it",
                      UnsafeFormatWarning, stacklevel=3)
    return HardwareError(code, msg, hint)


class SpeQtrumNativeProvider(SpeQtrumProvider):
    """Same options as SpeQtrumProvider except `transport` (the native client has its own)."""

    client = "native"

    def __init__(self, username: str | None = None, api_key: str | None = None, url: str | None = None,
                 audience: str | None = None, targets=None, **_: Any):
        super().__init__(username=username, api_key=api_key, url=url, audience=audience, targets=targets)

    def __repr__(self) -> str:  # never shows the username or the API key
        return f"SpeQtrumNativeProvider(url={self._url!r}, targets={sorted(self._targets)!r})"

    def _auth(self) -> tuple[str, str, str, str]:
        base, username, api_key = self._config()  # Python checks first: names only in messages
        return base, username, api_key, self._audience

    @staticmethod
    def _q():
        return load_qumo_py()

    # ------------------------------------------------------------ Provider API
    def list_backends(self, online: bool = False) -> list[BackendInfo]:
        rows = {r.name: r for r in super().list_backends(online=False)}
        if not online:
            return list(rows.values())
        try:
            items = self._q().speqtrum_devices(*self._auth())
        except Exception as exc:
            raise _native_error(exc) from None
        for code, nq, status, notes in items:
            code, nq = str(code), (int(nq) if int(nq) >= 0 else None)
            status = str(status) or None
            row = rows.get(code)
            if row is None:
                rows[code] = BackendInfo(code, self.name, "hardware", nq, status, None, True, str(notes))
            else:
                row.status = status
                row.num_qubits = row.num_qubits if row.num_qubits is not None else nq
        return list(rows.values())

    def encode(self, compiled: CompiledCircuit, shots: int, backend: str) -> str:
        """qumo.payload.v1 document built natively (byte-identical to qumo_payload_yaml)."""
        mm = sorted((int(c), int(q)) for c, q in compiled.measure_map.items())
        q = self._q()
        doc = str(q.speqtrum_payload_document(
            str(backend), int(shots), compiled.qasm, int(compiled.num_qubits), int(compiled.num_clbits),
            [int(v) for v in compiled.initial_layout], [int(v) for v in compiled.final_layout],
            [c for c, _ in mm], [qq for _, qq in mm], bool(compiled.auto_measured)))
        construct = str(q.speqtrum_unsafe_construct(doc))
        if construct:  # cannot happen for this writer; kept as a guard
            raise HardwareError(E_INVALID_PAYLOAD, f"payload refused before sending: it contains {construct}",
                                "report this circuit with --show-qasm")
        return doc

    def submit(self, compiled: CompiledCircuit, shots: int, backend: str, seed: int | None = None,
               payload=None, report=None, max_cost: float | None = None, confirm: bool = False,
               **_: Any) -> JobHandle:
        check_confirmed(self, backend, confirm)  # first: without confirm=True nothing is built or sent
        spec = self.target(backend)
        if shots < 1 or shots > spec.max_shots:
            raise HardwareError(E_LIMIT, f"shots must be in 1..{spec.max_shots} for {backend}", HINT_SHOTS)
        if max_cost is not None:
            raise HardwareError(E_LIMIT, "max_cost cannot be enforced for SpeQtrum: its devices publish no price",
                                "remove the cost cap; the SpeQtrum cost stays unknown for this provider")
        auth = self._auth()
        document = self.encode(compiled, shots, backend)
        try:
            job_id = str(self._q().speqtrum_execute(*auth, str(backend), document, int(shots)))
        except Exception as exc:
            raise _native_error(exc) from None
        meta = compiled.result_meta()
        meta.update({"device_code": backend, "payload_format": self.payload_format, "client": NATIVE})
        return JobHandle(job_id, self.name, backend, shots, False, QUEUED, meta=meta)

    def status(self, job: JobHandle) -> str:
        if job.local:
            return job.status
        try:
            return str(self._q().speqtrum_status(*self._auth(), str(job.job_id)))
        except Exception as exc:
            raise _native_error(exc) from None

    def result(self, job: JobHandle) -> dict[str, int]:
        if job.local:
            return dict(job.counts or {})
        try:
            counts = {str(k): int(v) for k, v in dict(self._q().speqtrum_result(*self._auth(),
                                                                                  str(job.job_id))).items()}
        except Exception as exc:
            raise _native_error(exc) from None
        width = job.meta.get("num_clbits")
        if isinstance(width, int) and any(len(key) != width for key in counts):
            widths = sorted({len(key) for key in counts})
            raise HardwareError(E_INVALID_PAYLOAD,
                                f"counts of job {job.job_id} have {widths} bits per shot, expected {width}",
                                "the classical width of the circuit and of the result disagree; report the job id")
        return counts
