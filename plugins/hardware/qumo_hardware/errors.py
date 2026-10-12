"""Stable error codes for the hardware plugins (range QM-E950..QM-E959).

Every failure the plugins report to a user is a `HardwareError` whose text form
follows QUMO's convention (docs/ERRORS.md):

    QM-E952: no IBM Quantum credentials found. Hint: set QISKIT_IBM_TOKEN ...

and whose structured form (`to_dict()`) matches QUMO's diagnostic JSON
(`code`, `severity`, `message`, `hint`, `docs`). Agents should branch on the
code. Messages never contain credentials.
"""

from __future__ import annotations

import re

E_QUMO_PY = "QM-E950"
E_SDK_MISSING = "QM-E951"
E_CREDENTIALS = "QM-E952"
E_UNKNOWN_BACKEND = "QM-E953"
E_COMPILE = "QM-E954"
E_INVALID_PAYLOAD = "QM-E955"
E_LIMIT = "QM-E956"
E_CONFIRM = "QM-E957"
E_UNKNOWN_JOB = "QM-E958"
E_JOB = "QM-E959"

DOCS = "docs/HARDWARE_PLUGINS.md#error-codes"

# code -> (summary, default hint). Mirrored in docs/HARDWARE_PLUGINS.md.
CATALOG: dict[str, tuple[str, str]] = {
    E_QUMO_PY: (
        "The QUMO Python extension (build/qumo_py.so) could not be loaded.",
        "Run `make py` at the repository root, or set QUMO_PY_PATH to the directory containing qumo_py.so.",
    ),
    E_SDK_MISSING: (
        "A provider SDK needed for this operation is not installed.",
        "Install the extra: pip install 'qumo-hardware[ibm]' or 'qumo-hardware[quantinuum]'.",
    ),
    E_CREDENTIALS: (
        "Provider credentials are missing, expired or rejected.",
        "IBM: export QISKIT_IBM_TOKEN (and QISKIT_IBM_INSTANCE) or QiskitRuntimeService.save_account(); "
        "Quantinuum: run qnexus.login() once.",
    ),
    E_UNKNOWN_BACKEND: (
        "Unknown provider or backend name, or the backend is not available to this account.",
        "List backends with `python -m qumo_hardware backends --provider <name>`.",
    ),
    E_COMPILE: (
        "The circuit cannot be compiled for the backend (too many qubits, unsupported operation, parse error).",
        "Reduce the circuit width or check the circuit with `build/qumo validate`.",
    ),
    E_INVALID_PAYLOAD: (
        "The compiled payload failed the provider's own validation (not ISA / not native).",
        "Report the circuit; retry with --provider-transpile to compare with the provider's compiler.",
    ),
    E_LIMIT: (
        "A shot, size or cost limit would be exceeded.",
        "The message names the limit: lower --shots for a shot limit, shrink the circuit or pick a larger "
        "backend for a size limit, raise --max-cost only if a higher cost is intended.",
    ),
    E_CONFIRM: (
        "Submission to a remote (billable) backend needs explicit confirmation.",
        "Review backend, shots and cost estimate, then re-run with --yes (CLI) or confirm=True (API).",
    ),
    E_UNKNOWN_JOB: (
        "Unknown job id (not in the local job store and not found at the provider).",
        "List known jobs with `python -m qumo_hardware jobs`.",
    ),
    E_JOB: (
        "The job failed, was cancelled, or its result is not available yet.",
        "Check `python -m qumo_hardware status <job>`; results exist only for completed jobs.",
    ),
}


# Cause-specific hints for QM-E956 (one code, several limits).
HINT_SHOTS = "lower --shots, or split the run into several jobs and add the counts"
HINT_COST = "raise --max-cost if this cost is intended, or lower --shots / simplify the circuit"
HINT_LOCAL_QUBITS = ("local emulation needs memory exponential in the number of active qubits; reduce the circuit, "
                     "or use a provider's cloud emulator or hardware (e.g. -p quantinuum -b H2-1E, -p ibm -b <device>)")


class HardwareError(Exception):
    """User-facing error with a stable QM-E95x code and a hint."""

    def __init__(self, code: str, message: str, hint: str | None = None, details: dict | None = None):
        self.code = code
        self.message = message.rstrip(".")
        self.hint = hint if hint is not None else CATALOG.get(code, ("", ""))[1]
        self.details = details or {}
        super().__init__(str(self))

    def __str__(self) -> str:
        text = f"{self.code}: {self.message}."
        if self.hint:
            text += f" Hint: {self.hint}"
        return text

    def to_dict(self) -> dict:
        out = {
            "code": self.code,
            "severity": "error",
            "message": self.message,
            "hint": self.hint,
            "docs": DOCS,
        }
        if self.details:
            out["details"] = self.details
        return out


_SECRETISH = re.compile(r"(?i)(bearer\s+|token[\"'=:\s]+|apikey[\"'=:\s]+|api_key[\"'=:\s]+)\S+|[A-Za-z0-9_\-]{40,}")


def safe_exc_text(exc: BaseException, limit: int = 240) -> str:
    """`Type: first line of the message`, shortened, with anything that looks like
    a credential (bearer/token/apikey values, long opaque strings) redacted.
    Provider SDK exceptions may echo request details; never print them raw."""
    first = (str(exc).strip().splitlines() or [""])[0]
    first = _SECRETISH.sub("<redacted>", first)
    if len(first) > limit:
        first = first[: limit - 3] + "..."
    return f"{type(exc).__name__}: {first}" if first else type(exc).__name__


def catalog() -> list[dict]:
    """Machine-readable error catalogue (for `python -m qumo_hardware errors --json`)."""
    return [{"code": c, "summary": s, "hint": h} for c, (s, h) in sorted(CATALOG.items())]
