"""Vendor-neutral provider interface.

QUMO stays hardware neutral: a provider only has to describe its devices as a
`TargetSpec` (plain data: qubits, couplings, native basis, error rates) and
know how to ship an already-compiled OpenQASM 2 payload and fetch counts back.
Compilation is done by QUMO itself (`qumo_hardware.compile`) for every
provider; providers never re-optimise the circuit, they only translate it 1:1
into their SDK's object (Qiskit `QuantumCircuit`, pytket `Circuit`, HUGR, ...).

Bit order everywhere: QUMO / Qiskit style, i.e. count keys are bitstrings over
the user's classical bits with the leftmost character = highest clbit.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .compile import CompiledCircuit

# Normalised job states (provider states are mapped onto these).
QUEUED = "QUEUED"
RUNNING = "RUNNING"
DONE = "DONE"
ERROR = "ERROR"
CANCELLED = "CANCELLED"
UNKNOWN = "UNKNOWN"
FINAL_STATES = (DONE, ERROR, CANCELLED)

BIT_ORDER = "leftmost character = highest classical bit c[n-1] (QUMO/Qiskit convention)"


@dataclass
class TargetSpec:
    """Neutral description of a device, consumed by QUMO's compiler.

    `edges` are undirected physical couplings (a < b); empty with
    `all_to_all=True` for fully connected devices (trapped ions). `basis` uses
    QUMO gate names (rz, sx, x, cz, ecr, cx, rx, ry, rzz, ...). `errors` holds
    median error rates (keys "1q", "2q", "readout"), `durations` median
    durations in seconds (same keys). `directed_edges` lists the native
    direction of directional two-qubit gates (ECR/CX on some IBM devices) and is
    None when the native 2q gate is symmetric.
    """

    name: str
    provider: str
    num_qubits: int
    edges: list[tuple[int, int]] = field(default_factory=list)
    basis: list[str] = field(default_factory=list)
    all_to_all: bool = False
    errors: dict[str, float] = field(default_factory=dict)
    durations: dict[str, float] = field(default_factory=dict)
    max_shots: int = 100_000
    directed_edges: list[tuple[int, int]] | None = None
    edge_errors: dict[str, float] = field(default_factory=dict)
    readout_errors: list[float] | None = None
    simulator: bool = False
    local: bool = False
    extras: dict[str, Any] = field(default_factory=dict)

    def qumo_edges(self) -> list[list[int]]:
        """Edges in the form `qumo_py.transpile` expects ([] = all-to-all)."""
        return [] if self.all_to_all else [[int(a), int(b)] for a, b in self.edges]

    def two_qubit_gates(self) -> list[str]:
        return [g for g in self.basis if g in TWO_QUBIT_GATES]

    def to_dict(self, compact: bool = False) -> dict:
        d = asdict(self)
        d["edges"] = [list(e) for e in self.edges]
        if self.directed_edges is not None:
            d["directed_edges"] = [list(e) for e in self.directed_edges]
        d["num_edges"] = len(self.edges)
        if compact:
            for k in ("edges", "directed_edges", "edge_errors", "readout_errors"):
                d.pop(k, None)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TargetSpec":
        d = dict(d)
        d.pop("num_edges", None)
        d["edges"] = [tuple(e) for e in d.get("edges", [])]
        if d.get("directed_edges") is not None:
            d["directed_edges"] = [tuple(e) for e in d["directed_edges"]]
        return cls(**d)


TWO_QUBIT_GATES = ("cx", "cz", "ecr", "iswap", "rxx", "ryy", "rzz", "rzx", "cy", "ch", "swap", "cp", "crz", "zz")


@dataclass
class BackendInfo:
    """One row of `backends` output."""

    name: str
    provider: str
    kind: str  # "hardware" | "cloud-emulator" | "syntax-checker" | "local-emulator"
    num_qubits: int | None = None
    status: str | None = None
    pending_jobs: int | None = None
    billable: bool = True
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CostEstimate:
    """Pre-submission cost estimate. `amount` is in `unit`.

    Units: "HQC" (Quantinuum hardware credits), "eHQC" (Quantinuum emulator
    credits), "QPU-seconds" (IBM billed QPU time), "free" (local / syntax check).
    `usd` is only filled when a public list price is known and is indicative.
    """

    provider: str
    backend: str
    shots: int
    unit: str
    amount: float
    formula: str
    usd: float | None = None
    breakdown: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def billable(self) -> bool:
        return self.unit != "free" and self.amount > 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["billable"] = self.billable
        return d

    def summary(self) -> str:
        if self.unit == "free":
            return "free (" + (self.notes[0] if self.notes else "no billing") + ")"
        s = f"{self.amount:.4g} {self.unit}"
        if self.usd is not None:
            s += f" (~${self.usd:,.2f} at list price)"
        return s


@dataclass
class JobHandle:
    """What `submit` returns. Persisted in the local job store (no secrets)."""

    job_id: str
    provider: str
    backend: str
    shots: int
    local: bool
    status: str = QUEUED
    submitted_at: float = field(default_factory=time.time)
    meta: dict[str, Any] = field(default_factory=dict)
    counts: dict[str, int] | None = None  # filled immediately by synchronous local runs

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "JobHandle":
        return cls(**d)


def new_local_job_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


@runtime_checkable
class Provider(Protocol):
    """Interface every hardware plugin implements.

    `compiled` is a `qumo_hardware.compile.CompiledCircuit` produced by QUMO
    for `target(backend)`. `submit` must not change the circuit's semantics:
    only 1:1 gate-object translation and provider-mandated, semantics-preserving
    finishing (e.g. gate direction, angle wrapping) are allowed, and they must be
    reported in the job metadata.

    Safety contract: `submit(..., confirm=False)` must refuse (QM-E957) to send
    anything to a remote backend unless the caller passes `confirm=True`
    (`check_confirmed` does it). `runner.run` passes `confirm=True` only after
    its own gate, so calling a provider directly is never less safe than the CLI.
    """

    name: str

    def list_backends(self, online: bool = False) -> list[BackendInfo]: ...

    def target(self, backend: str) -> TargetSpec: ...

    def is_local(self, backend: str) -> bool: ...

    def prepare(self, compiled: "CompiledCircuit", backend: str) -> Any: ...

    def validate(self, payload: Any, backend: str) -> dict: ...

    def estimate_cost(self, compiled: "CompiledCircuit", shots: int, backend: str) -> CostEstimate: ...

    def submit(self, compiled: "CompiledCircuit", shots: int, backend: str, confirm: bool = False,
               **options: Any) -> JobHandle: ...

    def status(self, job: JobHandle) -> str: ...

    def result(self, job: JobHandle) -> dict[str, int]: ...


def requires_confirmation(provider: Provider, backend: str) -> bool:
    """Anything that leaves this machine needs explicit confirmation."""
    return not provider.is_local(backend)


def check_confirmed(provider: Provider, backend: str, confirm: bool) -> None:
    """Provider-side safety gate (defence in depth behind `runner.run`)."""
    if requires_confirmation(provider, backend) and not confirm:
        from .errors import E_CONFIRM, HardwareError

        raise HardwareError(
            E_CONFIRM,
            f"not submitted: {provider.name}.submit() was called for remote backend '{backend}' without confirm=True",
            "review the plan (runner.plan) and call submit(..., confirm=True), or use runner.run(..., confirm=True)",
        )
