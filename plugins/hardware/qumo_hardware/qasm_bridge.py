"""1:1 translation of QUMO's OpenQASM 2 export into provider SDK objects.

No optimisation happens here: every QUMO gate becomes exactly one SDK gate.

QUMO's exporter (`qumo/ir/qasm2.mojo`) writes a `gate` definition for every
gate outside the original `qelib1.inc` (ecr, iswap, iswapdg, ryy, ccz and, in
newer versions, also sx, sxdg, swap, cswap, crx, cry, rxx, rzz). A parser that
inlines such a definition would turn one hardware gate into several (an `rzz`
would become cx-rz-cx, i.e. two ZZPhase gates on Quantinuum), and the payload
would no longer be QUMO's circuit. So, before parsing, the bridges remove the
definitions of gates that the SDK already knows natively with the *same*
matrix, and the SDK's own gate class is used instead:

* Qiskit: `ECRGate`, `iSwapGate`, `RYYGate`, `RZXGate`, `CCZGate` are added
  as built-in custom instructions (IBM's ISA check requires the real gate
  classes); the other names are covered by `qiskit.qasm2`'s
  `LEGACY_CUSTOM_INSTRUCTIONS`, which substitute the standard classes.
* pytket: `sx, sxdg, swap, cswap, crx, cry, rxx, rzz, ecr` are parsed into
  the native OpTypes (pytket's `ecr` is the same matrix as Qiskit's ECR in
  pytket's big-endian convention). Gates that pytket names differently or
  parametrises differently (`iswap` has one parameter in pytket) keep their
  definition and become a `CustomGate`, which the provider rebase decomposes
  exactly.

Origins of the workaround: an earlier QUMO exporter defined `ecr` through
`rzx`, which no include file declares, so every strict parser rejected it.
Stripping the `ecr` definition made both the old and the fixed exporter
output load; the strip lists above are harmless when a definition is absent
or already correct. `rzx` stays registered for the same reason.
"""

from __future__ import annotations

import re

from .errors import E_INVALID_PAYLOAD, E_SDK_MISSING, HardwareError

# QUMO gate name -> (qiskit.circuit.library class name, num_params, num_qubits)
_QISKIT_EXTRA = {
    "ecr": ("ECRGate", 0, 2),
    "iswap": ("iSwapGate", 0, 2),
    "iswapdg": (None, 0, 2),  # iSwapGate().inverse()
    "ryy": ("RYYGate", 1, 2),
    "rzx": ("RZXGate", 1, 2),
    "ccz": ("CCZGate", 0, 3),
}

# Definitions pytket must not inline: these names are native pytket OpTypes
# with exactly QUMO's semantics (see the module docstring).
PYTKET_NATIVE = ("sx", "sxdg", "swap", "cswap", "crx", "cry", "rxx", "rzz", "ecr")

_RE_GATE_DEF = re.compile(r"^\s*gate\s+(\w+)\b[^{]*\{[^}]*\}\s*$", re.M)
_RE_CREG = re.compile(r"^\s*creg\s+\w+\s*\[\s*(\d+)\s*\]\s*;", re.M)

_BRIDGE_HINT = ("this is a bug in the QUMO -> {sdk} bridge, not in your circuit; report it together with the "
                "compiled payload (`python -m qumo_hardware estimate ... --show-qasm`)")


def strip_gate_definitions(qasm2: str, names) -> str:
    names = set(names)
    return _RE_GATE_DEF.sub(lambda m: "" if m.group(1) in names else m.group(0), qasm2)


def max_creg_width(qasm2: str) -> int:
    """Width of the widest classical register declared in `qasm2` (0 if none)."""
    return max((int(w) for w in _RE_CREG.findall(qasm2)), default=0)


def to_qiskit(qasm2: str):
    """QUMO OpenQASM 2 -> qiskit.QuantumCircuit with standard gate classes."""
    try:
        from qiskit import qasm2 as q2
        from qiskit.circuit import library as lib
    except ImportError:
        raise HardwareError(E_SDK_MISSING, "qiskit is not installed",
                            "pip install 'qumo-hardware[ibm]'") from None

    def _iswapdg():
        return lib.iSwapGate().inverse()

    custom = list(q2.LEGACY_CUSTOM_INSTRUCTIONS)
    for name, (cls, npar, nq) in _QISKIT_EXTRA.items():
        ctor = getattr(lib, cls) if cls else _iswapdg
        custom.append(q2.CustomInstruction(name, npar, nq, ctor, builtin=True))
    text = strip_gate_definitions(qasm2, _QISKIT_EXTRA.keys())
    try:
        return q2.loads(text, custom_instructions=custom)
    except Exception as exc:
        raise HardwareError(E_INVALID_PAYLOAD, f"Qiskit cannot load the compiled circuit: {exc}",
                            _BRIDGE_HINT.format(sdk="Qiskit")) from None


def to_pytket(qasm2: str):
    """QUMO OpenQASM 2 -> pytket.Circuit (gates translated 1:1).

    pytket refuses classical registers wider than its `maxwidth` argument
    (default 32). QUMO payloads have one register `c` with one bit per user
    clbit, so the limit is raised to the widest register in the text; device
    limits on register width are applied later by the provider
    (`quantinuum.split_classical_register`)."""
    try:
        from pytket.qasm import circuit_from_qasm_str
    except ImportError:
        raise HardwareError(E_SDK_MISSING, "pytket is not installed",
                            "pip install 'qumo-hardware[quantinuum]'") from None
    text = strip_gate_definitions(qasm2, PYTKET_NATIVE)
    try:
        return circuit_from_qasm_str(text, maxwidth=max(64, max_creg_width(text)))
    except Exception as exc:
        raise HardwareError(E_INVALID_PAYLOAD, f"pytket cannot load the compiled circuit: {exc}",
                            _BRIDGE_HINT.format(sdk="pytket")) from None
