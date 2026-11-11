"""Errors raised by the `qumo` package.

Every failure of the native library arrives as a `QumoError` with the QUMO
error code (`QM-Exxx`, catalogued in docs/ERRORS.md) and a hint written for
Python callers.
"""

from __future__ import annotations

import re

__all__ = ["QumoError", "QumoTypeError", "QumoNativeMissing"]

_CODE = re.compile(r"^(QM-[EW]\d{3}):\s*")

# Hints of the native library name Mojo functions; these replace them with what
# a Python caller can type.
_PYTHON_HINTS = {
    "QM-E402": "call circuit.measure_all() (or circuit.measure(q, c)) before sampling, or use "
               "qumo.statevector(circuit) / qumo.expectation(circuit, observable).",
    "QM-E404": "Clifford circuits of any width run with qumo.sample() (automatic dispatch picks the "
               "stabilizer simulator); otherwise reduce the number of qubits.",
    "QM-E101": "pass params={name: value, ...} or bind first with circuit.bind({name: value, ...}); "
               "circuit.parameter_names lists the names.",
    "QM-E410": "qumo.sample() takes Pauli noise on more qubits (trajectories), or reduce the number of qubits.",
    "QM-E411": "use qumo.sample(), which runs per-shot trajectories.",
    "QM-E409": "use qumo.sample(circuit, method='auto'), which falls back to per-shot trajectories.",
}

_REWRITES = (
    ("evolve()/expectation()", "qumo.statevector() / qumo.expectation()"),
    ("density_evolve()", "qumo.expectation() / qumo.probabilities()"),
    ("sample_ptsbe()", "qumo.sample(circuit, method='ptsbe')"),
    ("sample()", "qumo.sample()"),
    ("measure_all()", "circuit.measure_all()"),
)


class QumoError(Exception):
    """An error reported by QUMO.

    Attributes:
        code: the QUMO error code, e.g. ``"QM-E402"`` (``""`` when the native
            library gave none). ``docs/ERRORS.md`` has one entry per code.
        hint: what to do about it, or ``""``.
        message: the problem, without code and hint.

    Example:
        >>> import qumo
        >>> try:
        ...     qumo.sample(qumo.Circuit(1).h(0))
        ... except qumo.QumoError as e:
        ...     print(e.code)
        QM-E402
    """

    def __init__(self, message: str, code: str = "", hint: str = ""):
        self.message = message
        self.code = code
        self.hint = hint
        text = f"{code}: {message}" if code else message
        if hint:
            text += f"\n  Hint: {hint}"
        super().__init__(text)

    def __reduce__(self):
        return (type(self), (self.message, self.code, self.hint))

    @property
    def docs(self) -> str:
        """Anchor of this code in the error catalogue."""
        return f"docs/ERRORS.md#{self.code.lower()}" if self.code else ""


class QumoTypeError(QumoError, TypeError):
    """An argument has the wrong type (QM-E911). Also a `TypeError`."""


class QumoNativeMissing(QumoError, ImportError):
    """The compiled extension `qumo_py` was not found or could not be loaded."""


def from_native(exc: BaseException) -> QumoError:
    """Turn the bare `Exception` of the extension into a `QumoError`."""
    text = str(exc)
    code = ""
    m = _CODE.match(text)
    if m:
        code = m.group(1)
        text = text[m.end():]
    # the parser appends a source excerpt after the first line: keep it in the message
    first, sep, excerpt = text.partition("\n")
    hint = ""
    at = first.find(" Hint: ")
    if at >= 0:
        hint = first[at + 7:].strip()
        first = first[:at].rstrip()
    if code in _PYTHON_HINTS:
        hint = _PYTHON_HINTS[code]
    elif hint:
        for old, new in _REWRITES:
            if old in hint and new not in hint:
                hint = hint.replace(old, new)
                break
    message = first + (sep + excerpt if excerpt else "")
    cls = QumoTypeError if code == "QM-E911" else QumoError
    return cls(message, code, hint)
