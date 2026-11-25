"""Stable error codes for the hardware plugins (range QM-E950..QM-E959).

Every failure the plugins report to a user is a `HardwareError` whose text form
follows QUMO's convention (docs/ERRORS.md):

    QM-E952: no IBM Quantum credentials found. Hint: set QISKIT_IBM_TOKEN ...

and whose structured form (`to_dict()`) matches QUMO's diagnostic JSON
(`code`, `severity`, `message`, `hint`, `docs`). Agents should branch on the
code. Messages never contain credentials.
"""

from __future__ import annotations

import json
import os
import re
import urllib.parse

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
        "Quantinuum: run qnexus.login() once; IonQ: export IONQ_API_KEY.",
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
        "The compiled payload failed the provider's own validation (not ISA / not native) or cannot be "
        "translated into the provider's program format.",
        "The message names the failed check. If it is a QUMO bug, report the circuit and its payload "
        "(--show-qasm); for ISA/native-gate failures, --provider-transpile submits the provider compiler's "
        "output instead.",
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


# --------------------------------------------------------------------------- redaction
REDACTED = "<redacted>"

# Credential variables the plugin (or the client it drives) reads. Their current values are
# redacted literally, together with those of any other variable whose name looks secret.
CREDENTIAL_ENV_VARS = ("QISKIT_IBM_TOKEN", "IONQ_API_KEY", "QBRAID_API_KEY", "QUMO_SPEQTRUM_APIKEY")
_SECRET_ENV_NAME = re.compile(r"(?i)(?:_API_?KEY|_TOKEN|_SECRET|_PASSWORD)$|SECRET|PASSWORD")

# Length rule for literal values. 8 characters or more: redacted wherever they occur.
# 4 to 7: only as a whole word (not next to a letter, digit, `_` or `-`), so that a short
# value does not blank out pieces of ordinary words. Under 4: never (it would mark every
# occurrence of two or three common characters, which hides nothing and ruins the message).
MIN_SUBSTRING_SECRET = 8
MIN_WORD_SECRET = 4

_WORD = r"A-Za-z0-9_\-"
_VALUE = r"[A-Za-z0-9._~+/=\-]{8,}"
_AUTH_HEADER = re.compile(r"(?i)((?:proxy-)?authorization[\"']?\s*[:=]\s*[\"']?)((?:bearer|basic|token|apikey|digest)\s+)?"
                          r"[^\s\"',;]+")
_SCHEME = re.compile(rf"(?i)(?<![{_WORD}])(bearer|basic)(\s+)({_VALUE})")
_KEYED = re.compile(
    rf"(?i)(?<![A-Za-z0-9])((?:[{_WORD}]*[_\-])?(?:api[_\-]?key|key|token|secret|password|passwd|pwd))(?![{_WORD}])"
    rf"(?:([\"']?\s*[=:]\s*)(\"[^\"]*\"|'[^']*'|[^\s\"'&,;]+)|(\s+)({_VALUE}))")
_JWT = re.compile(rf"(?<![{_WORD}])eyJ[A-Za-z0-9_\-]{{5,}}\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]*")
_LONG_OPAQUE = re.compile(rf"[{_WORD}]{{40,}}")
_HEX_DIGEST = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_UUID_ID = re.compile(r"(?i)(?:[a-z]{1,12}[_\-])?[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def _looks_opaque(value: str) -> bool:
    """True for a word that reads like a key, not like prose: 20 characters or more, or 8 or
    more with a digit, one of `+/=`, or a capital after the first character."""
    if len(value) >= 20:
        return True
    if len(value) < 8:
        return False
    if any(c.isdigit() or c in "+/=" for c in value):
        return True
    return any(c.isupper() for c in value[1:]) and any(c.islower() for c in value)


def _secret_forms() -> list[str]:
    """Current values of the credential variables (read at each call, never cached), each also
    URL-encoded and JSON-escaped, longest first."""
    forms: set[str] = set()
    for name, value in os.environ.items():
        if name not in CREDENTIAL_ENV_VARS and not _SECRET_ENV_NAME.search(name):
            continue
        for v in {value, value.strip()}:
            if len(v) < MIN_WORD_SECRET:
                continue
            forms.update((v, urllib.parse.quote(v, safe=""), urllib.parse.quote_plus(v),
                          json.dumps(v)[1:-1], json.dumps(v, ensure_ascii=False)[1:-1]))
    return sorted(forms, key=len, reverse=True)


def _keyed(m: re.Match) -> str:
    name, sep, value, space, word = m.groups()
    if sep is not None:
        if value.startswith("<") or value in ('""', "''"):
            return m.group(0)
        return f"{name}{sep}{REDACTED}"
    return f"{name}{space}{REDACTED}" if _looks_opaque(word) else m.group(0)


def redact_text(text: str) -> str:
    """`text` with credentials replaced by `<redacted>`:

    * the current value of every credential environment variable (CREDENTIAL_ENV_VARS and any
      variable named `*_API_KEY`, `*_APIKEY`, `*_TOKEN`, `*SECRET*`, `*PASSWORD*`), also in
      its URL-encoded and JSON-escaped forms (length rule: MIN_SUBSTRING_SECRET);
    * the value of an `Authorization` header, and what follows `Bearer` / `Basic`;
    * the value after a credential name (`x-api-key: v`, `access_token=v`, `password=v`,
      `"secret": "v"`, `key v`); the name stays. After a plain space only a word that looks
      like a key is redacted, so `the token expired` is kept;
    * JSON Web Tokens and words of 40 or more key characters, except lower-case hexadecimal
      digests of 40 or 64 digits and UUID job ids, which are kept.
    """
    for form in _secret_forms():
        if len(form) >= MIN_SUBSTRING_SECRET:
            text = text.replace(form, REDACTED)
        else:
            text = re.sub(rf"(?<![{_WORD}]){re.escape(form)}(?![{_WORD}])", REDACTED, text)
    text = _AUTH_HEADER.sub(lambda m: f"{m.group(1)}{m.group(2) or ''}{REDACTED}", text)
    text = _SCHEME.sub(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}"
                       if m.group(1).lower() == "bearer" or _looks_opaque(m.group(3)) else m.group(0), text)
    text = _KEYED.sub(_keyed, text)
    text = _JWT.sub(REDACTED, text)
    return _LONG_OPAQUE.sub(
        lambda m: m.group(0) if _HEX_DIGEST.fullmatch(m.group(0)) or _UUID_ID.fullmatch(m.group(0)) else REDACTED, text)


def safe_exc_text(exc: BaseException, limit: int = 240) -> str:
    """`Type: first line of the message`, shortened, with credentials redacted by
    `redact_text`. Provider SDK exceptions may echo request details; never print them raw."""
    first = (str(exc).strip().splitlines() or [""])[0]
    first = redact_text(first)
    if len(first) > limit:
        first = first[: limit - 3] + "..."
    return f"{type(exc).__name__}: {first}" if first else type(exc).__name__


def catalog() -> list[dict]:
    """Machine-readable error catalogue (for `python -m qumo_hardware errors --json`)."""
    return [{"code": c, "summary": s, "hint": h} for c, (s, h) in sorted(CATALOG.items())]
