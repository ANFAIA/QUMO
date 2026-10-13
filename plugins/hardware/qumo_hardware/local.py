"""Local emulation provider: no account, no network, no cost.

Backends:

* `qumo-sim` - QUMO's own state-vector simulator, all-to-all, any gate.
* `qumo-sim:<topology>[/<basis>]` - QUMO simulator behind a constrained target,
  so layout/routing/bit-order handling is exercised exactly as for hardware:
  `line:N`, `ring:N`, `grid:RxC`, `all:N`, `ibm:<fake_backend>` (the exact
  IBM TargetSpec, e.g. `qumo-sim:ibm:fake_torino`) or `quantinuum:<device>`.
  Basis: a QUMO preset (`sc_cz`, `sc_ecr`, `ion_zz`, ...) or a gate list.
* `fake_<name>` - IBM calibration snapshot + Qiskit Aer noise model, executed
  through the same client-side Sampler code path as real IBM hardware.
* `H2-1LE` / `selene` - Quantinuum local emulators (pytket-pecos / Guppy+Selene).

The compiled physical circuit is restricted to its active qubits before QUMO
simulates it (a 133-qubit ISA circuit touching 5 qubits simulates 5 qubits).
"""

from __future__ import annotations

from typing import Any

from .base import DONE, BackendInfo, CostEstimate, JobHandle, TargetSpec, new_local_job_id
from .compile import CompiledCircuit, compress_qasm, expand_basis, load_qumo_py
from .errors import E_LIMIT, E_UNKNOWN_BACKEND, HINT_LOCAL_QUBITS, HardwareError

SIM_MAX_QUBITS = 26


def _line(n: int) -> list[tuple[int, int]]:
    return [(i, i + 1) for i in range(n - 1)]


def _ring(n: int) -> list[tuple[int, int]]:
    return _line(n) + ([(0, n - 1)] if n > 2 else [])


def _grid(r: int, c: int) -> list[tuple[int, int]]:
    e = []
    for i in range(r):
        for j in range(c):
            k = i * c + j
            if j + 1 < c:
                e.append((k, k + 1))
            if i + 1 < r:
                e.append((k, k + c))
    return e


def parse_sim_backend(name: str) -> TargetSpec:
    """'qumo-sim[:topology[/basis]]' -> TargetSpec."""
    if name == "qumo-sim":
        return TargetSpec("qumo-sim", "local", SIM_MAX_QUBITS, [], [], True, simulator=True, local=True,
                          extras={"kind": "ideal state vector"})
    if not name.startswith("qumo-sim:"):
        raise HardwareError(E_UNKNOWN_BACKEND, f"unknown local backend '{name}'")
    spec_text = name[len("qumo-sim:"):]
    basis_text = ""
    if "/" in spec_text:
        spec_text, basis_text = spec_text.split("/", 1)
    kind, _, size = spec_text.partition(":")
    try:
        if kind == "ibm":
            from .ibm import IBMProvider

            spec = IBMProvider().target(size)
        elif kind == "quantinuum":
            from .quantinuum import QuantinuumProvider

            spec = QuantinuumProvider().target(size)
        elif kind == "line":
            n = int(size)
            spec = TargetSpec(name, "local", n, _line(n))
        elif kind == "ring":
            n = int(size)
            spec = TargetSpec(name, "local", n, _ring(n))
        elif kind == "grid":
            r, _, c = size.lower().partition("x")
            r, c = int(r), int(c or r)
            spec = TargetSpec(name, "local", r * c, _grid(r, c))
        elif kind in ("all", "all_to_all"):
            spec = TargetSpec(name, "local", int(size), [], all_to_all=True)
        else:
            raise ValueError(kind)
    except HardwareError:
        raise
    except Exception:
        raise HardwareError(E_UNKNOWN_BACKEND, f"cannot parse local backend '{name}'",
                            "use qumo-sim, qumo-sim:line:5/sc_cz, qumo-sim:grid:3x3, qumo-sim:ibm:fake_torino") from None
    spec.name = name
    spec.provider = "local"
    spec.simulator = True
    spec.local = True
    if basis_text:
        spec.basis = expand_basis(basis_text)
    spec.directed_edges = None  # QUMO simulation does not care about gate direction
    return spec


class LocalEmulatorProvider:
    name = "local"

    def __init__(self, seed: int | None = None, use_fractional_gates: bool = False, **_: Any):
        self.seed = seed
        self.use_fractional_gates = use_fractional_gates  # forwarded to IBM fake backends

    def public_options(self) -> dict:
        return {"use_fractional_gates": self.use_fractional_gates} if self.use_fractional_gates else {}

    def is_local(self, backend: str) -> bool:
        return True

    def _delegate(self, backend: str):
        if backend.startswith("fake_"):
            from .ibm import IBMProvider

            return IBMProvider(use_fractional_gates=self.use_fractional_gates)
        if backend in ("H2-1LE", "selene") or backend.endswith("LE"):
            from .quantinuum import QuantinuumProvider

            return QuantinuumProvider()
        return None

    def list_backends(self, online: bool = False) -> list[BackendInfo]:
        out = [
            BackendInfo("qumo-sim", self.name, "local-emulator", SIM_MAX_QUBITS, "local", 0, False,
                        "QUMO state vector, all-to-all, ideal"),
            BackendInfo("qumo-sim:<topology>[/<basis>]", self.name, "local-emulator", None, "local", 0, False,
                        "e.g. qumo-sim:line:5/sc_cz, qumo-sim:grid:3x3/sc_cz, qumo-sim:ibm:fake_torino"),
            BackendInfo("fake_<ibm_backend>", self.name, "local-emulator", None, "local", 0, False,
                        "IBM snapshot + Aer noise (see `backends --provider ibm`)"),
            BackendInfo("H2-1LE", self.name, "local-emulator", 56, "local", 0, False,
                        "Quantinuum H2 local emulator (pytket-quantinuum[pecos])"),
            BackendInfo("selene", self.name, "local-emulator", 98, "local", 0, False,
                        "Guppy + Selene emulator (guppylang)"),
        ]
        return out

    def target(self, backend: str) -> TargetSpec:
        d = self._delegate(backend)
        return d.target(backend) if d else parse_sim_backend(backend)

    def prepare(self, compiled: CompiledCircuit, backend: str):
        d = self._delegate(backend)
        if d:
            return d.prepare(compiled, backend)
        text, kept = compress_qasm(compiled)
        if len(kept) > SIM_MAX_QUBITS:
            raise HardwareError(E_LIMIT, f"{len(kept)} active qubits exceed the local simulator limit {SIM_MAX_QUBITS}",
                                HINT_LOCAL_QUBITS)
        return text, {"active_qubits": kept}

    def validate(self, payload, backend: str) -> dict:
        d = self._delegate(backend)
        if d:
            return d.validate(payload, backend)
        return {"checks": ["qumo conformance (native gates, couplings)"]}

    def estimate_cost(self, compiled: CompiledCircuit, shots: int, backend: str, payload=None) -> CostEstimate:
        return CostEstimate(self.name, backend, shots, "free", 0.0, "local simulation",
                            notes=["runs on this machine; nothing is billed"])

    def submit(self, compiled: CompiledCircuit, shots: int, backend: str, seed: int | None = None,
               payload=None, report=None, **opts: Any) -> JobHandle:
        d = self._delegate(backend)
        seed = self.seed if seed is None else seed
        if d:
            job = d.submit(compiled, shots, backend, seed=seed, payload=payload, report=report, **opts)
            job.provider = self.name
            return job
        if payload is None:
            payload, report = self.prepare(compiled, backend)
        q = load_qumo_py()
        circ = q.load_qasm(payload)
        counts = {str(k): int(v) for k, v in dict(q.sample(circ, int(shots), int(seed or 0))).items()}
        meta = compiled.result_meta()
        meta["simulated_qubits"] = (report or {}).get("active_qubits")
        return JobHandle(new_local_job_id("local"), self.name, backend, shots, True, DONE, meta=meta, counts=counts)

    def status(self, job: JobHandle) -> str:
        return job.status

    def result(self, job: JobHandle) -> dict[str, int]:
        return dict(job.counts or {})
