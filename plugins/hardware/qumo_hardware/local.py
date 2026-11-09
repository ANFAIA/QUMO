"""Local emulation provider: no account, no network, no cost.

Backends:

* `qumo-sim` - QUMO's own state-vector simulator, all-to-all, any gate.
* `qumo-sim:<topology>[/<basis>]` - QUMO simulator behind a constrained target,
  so layout/routing/bit-order handling is exercised exactly as for hardware:
  `line:N`, `ring:N`, `grid:RxC`, `all:N`, `ibm:<fake_backend>` (the exact
  IBM TargetSpec, e.g. `qumo-sim:ibm:fake_torino`) or `quantinuum:<device>`.
  Basis: a QUMO preset (`sc_cz`, `sc_ecr`, `ion_zz`, ...) or a gate list.
* `qumo-sim[:<topology>[/<basis>]]+noise:<modality>` - the same, plus a
  vendor-neutral noise model named after a hardware *modality*, not a vendor:
  `superconducting`, `trapped_ion`, `neutral_atom` (rates of QUMO's
  HardwareProfile presets, qumo/exec/resources.mojo) or `custom(p1,p2,pm)`.
  After QUMO compiles the circuit for the target, a depolarizing channel is
  inserted after every gate (`depolarize1(p1)` / `depolarize2(p2)`) and a bit
  flip `x_error(pm)` before every measurement, as `#pragma qumo` noise that
  QUMO simulates as per-shot trajectories (seeded, deterministic). This is a
  depolarizing approximation, not a device calibration; still free and local.
* `fake_<name>` - IBM calibration snapshot + Qiskit Aer noise model, executed
  through the same client-side Sampler code path as real IBM hardware.
* `H2-1LE` / `selene` - Quantinuum local emulators (pytket-pecos / Guppy+Selene).

The compiled physical circuit is restricted to its active qubits before QUMO
simulates it (a 133-qubit ISA circuit touching 5 qubits simulates 5 qubits).
"""

from __future__ import annotations

import re
from typing import Any

from .base import DONE, BackendInfo, CostEstimate, JobHandle, TargetSpec, new_local_job_id
from .compile import CompiledCircuit, compress_qasm, expand_basis, load_qumo_py
from .errors import E_LIMIT, E_UNKNOWN_BACKEND, HINT_LOCAL_QUBITS, HardwareError

SIM_MAX_QUBITS = 26

# Modality noise presets: (1q gate error, 2q gate error, readout error). Same
# numbers as QUMO's HardwareProfile presets (qumo/exec/resources.mojo), so the
# emulator and `qumo_estimate_resources` agree on what a modality means.
NOISE_PRESETS: dict[str, tuple[float, float, float]] = {
    "superconducting": (2.5e-4, 3e-3, 1e-2),
    "trapped_ion": (3e-5, 1e-3, 3e-3),
    "neutral_atom": (5e-4, 5e-3, 1e-2),
}
_RE_CUSTOM = re.compile(r"^custom\(\s*([^,()]+)\s*,\s*([^,()]+)\s*,\s*([^,()]+)\s*\)$")
NOISE_NOTE = ("noise: depolarizing approximation of the '{m}' modality preset (1q {p1:g}, 2q {p2:g}, "
              "readout {pm:g}); not a device calibration")


def parse_noise(text: str) -> dict:
    """'superconducting' | 'trapped_ion' | 'neutral_atom' | 'custom(p1,p2,pm)' -> noise model dict."""
    key = text.strip().lower()
    if key in NOISE_PRESETS:
        p1, p2, pm = NOISE_PRESETS[key]
        modality = key
    else:
        m = _RE_CUSTOM.match(key)
        if not m:
            raise HardwareError(E_UNKNOWN_BACKEND, f"unknown noise modality '{text}'",
                                "use +noise:superconducting, +noise:trapped_ion, +noise:neutral_atom or "
                                "+noise:custom(p1,p2,pm)")
        try:
            p1, p2, pm = (float(v) for v in m.groups())
        except ValueError:
            raise HardwareError(E_UNKNOWN_BACKEND, f"cannot parse noise rates in '{text}'",
                                "custom(p1,p2,pm) takes three probabilities, e.g. custom(1e-4,1e-3,1e-2)") from None
        for v in (p1, p2, pm):
            if not 0.0 <= v <= 0.75:
                raise HardwareError(E_UNKNOWN_BACKEND, f"noise rate {v} in '{text}' is outside [0, 0.75]",
                                    "probabilities are per gate / per measurement")
        modality = "custom"
    return {"modality": modality, "p1": p1, "p2": p2, "readout": pm, "channels": {
        "after_1q_gate": "depolarize1(p1)", "after_2q_gate": "depolarize2(p2)",
        "before_measure": "x_error(readout)"}, "simulation": "statevector trajectories (#pragma qumo noise)"}


def inject_noise(qasm2: str, noise: dict, qumo: Any = None) -> tuple[str, dict]:
    """Insert `#pragma qumo` noise after every gate and before every measurement (native, qumo/ir/noise_inject.mojo).

    Works on QUMO-exported OpenQASM 2 (one statement per line). Gates of three
    or more qubits get depolarize1(p2) on each qubit. Returns (text, counts of
    inserted channels)."""
    q = qumo if qumo is not None else load_qumo_py()
    text, d1, d2, xe = q.inject_noise(qasm2, float(noise["p1"]), float(noise["p2"]), float(noise["readout"]))
    return str(text), {"depolarize1": int(d1), "depolarize2": int(d2), "x_error": int(xe)}


# --------------------------------------------------------------------------- noisy sampling
def sample_noisy(q, text: str, shots: int, seed: int) -> tuple[dict[str, int], dict]:
    """Sample a pragma-noise payload with QUMO's Mojo pre-trajectory sampler (PTSBE).

    Every shot draws its Pauli errors (gate channels and readout flips) up front;
    identical error patterns are simulated once, with one state per prefix of the
    pattern trie. Same distribution as per-shot trajectories, seeded and exact.
    Dynamic payloads (conditions, resets, gates after a measurement) are outside
    PTSBE (QM-E409) and fall back to per-shot trajectories."""
    circ = q.load_qasm(text)
    try:
        distinct, sites, error_free = q.plan_trajectories(circ, int(shots), int(seed))
    except Exception as exc:
        if "QM-E409" not in str(exc):
            raise
        counts = {str(k): int(v) for k, v in dict(q.sample(circ, int(shots), int(seed))).items()}
        return counts, {"method": "statevector:trajectories (QUMO pragma noise)"}
    if sites == 0:  # zero-rate channels: same draws as the noise-free backend
        counts = {str(k): int(v) for k, v in dict(q.sample(circ, int(shots), int(seed))).items()}
        return counts, {"method": "statevector (no noise sites drawn)", "noise_sites": 0}
    counts = {str(k): int(v) for k, v in dict(q.sample_ptsbe(circ, int(shots), int(seed))).items()}
    return counts, {"method": "statevector:ptsbe (QUMO Mojo, one simulation per distinct error pattern)",
                    "error_free_shots": int(error_free),
                    "distinct_error_patterns": int(distinct) - (1 if error_free > 0 else 0),
                    "noise_sites": int(sites)}


def split_noise_suffix(name: str) -> tuple[str, str | None]:
    """'qumo-sim:grid:4x4/sc_cz+noise:superconducting' -> ('qumo-sim:grid:4x4/sc_cz', 'superconducting')."""
    if "+noise:" in name:
        base, _, noise = name.partition("+noise:")
        return base, noise
    return name, None


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
    """'qumo-sim[:topology[/basis]][+noise:modality]' -> TargetSpec."""
    base, noise_text = split_noise_suffix(name)
    if noise_text is not None:
        noise = parse_noise(noise_text)
        spec = parse_sim_backend(base)
        spec.name = name
        spec.errors = {"1q": noise["p1"], "2q": noise["p2"], "readout": noise["readout"]}
        spec.extras = dict(spec.extras or {})
        spec.extras["noise"] = noise
        spec.extras["kind"] = f"noisy state vector ({noise['modality']} modality preset)"
        spec.extras.setdefault("warnings", [])
        spec.extras["warnings"] = list(spec.extras["warnings"]) + [
            NOISE_NOTE.format(m=noise["modality"], p1=noise["p1"], p2=noise["p2"], pm=noise["readout"])]
        return spec
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
                            "use qumo-sim, qumo-sim:line:5/sc_cz, qumo-sim:grid:3x3, qumo-sim:ibm:fake_torino, "
                            "qumo-sim:grid:4x4/sc_cz+noise:superconducting") from None
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
    payload_format = "openqasm2"

    def __init__(self, seed: int | None = None, use_fractional_gates: bool = False, **_: Any):
        self.seed = seed
        self.use_fractional_gates = use_fractional_gates  # forwarded to IBM fake backends

    def public_options(self) -> dict:
        return {"use_fractional_gates": self.use_fractional_gates} if self.use_fractional_gates else {}

    def is_local(self, backend: str) -> bool:
        return True

    def _delegate(self, backend: str):
        if backend.startswith("qumo-sim"):
            return None  # QUMO's own simulator (ideal or with +noise:<modality>)
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
            BackendInfo("qumo-sim:<topology>[/<basis>]+noise:<modality>", self.name, "local-emulator", SIM_MAX_QUBITS,
                        "local", 0, False,
                        "vendor-neutral noisy emulator: modality superconducting | trapped_ion | neutral_atom | "
                        "custom(p1,p2,pm), e.g. qumo-sim:grid:4x4/sc_cz+noise:superconducting"),
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

    def encode(self, compiled: CompiledCircuit, shots: int, backend: str) -> str:
        """QASM-to-payload conversion: the compiled OpenQASM 2 text, unchanged."""
        return compiled.qasm

    def prepare(self, compiled: CompiledCircuit, backend: str):
        d = self._delegate(backend)
        if d:
            return d.prepare(compiled, backend)
        text, kept = compress_qasm(compiled)
        if len(kept) > SIM_MAX_QUBITS:
            raise HardwareError(E_LIMIT, f"{len(kept)} active qubits exceed the local simulator limit {SIM_MAX_QUBITS}",
                                HINT_LOCAL_QUBITS)
        report: dict = {"active_qubits": kept}
        _, noise_text = split_noise_suffix(backend)
        if noise_text is not None:
            noise = parse_noise(noise_text)
            text, inserted = inject_noise(text, noise)
            report.update({"noisy": True, "noise_model": noise, "noise_channels_inserted": inserted})
        return text, report

    def validate(self, payload, backend: str) -> dict:
        d = self._delegate(backend)
        if d:
            return d.validate(payload, backend)
        checks = ["qumo conformance (native gates, couplings)"]
        if split_noise_suffix(backend)[1] is not None:
            checks.append("noise channels inserted after compilation (depolarizing + readout)")
        return {"checks": checks}

    def estimate_cost(self, compiled: CompiledCircuit, shots: int, backend: str, payload=None) -> CostEstimate:
        how = "local noisy simulation" if split_noise_suffix(backend)[1] is not None else "local simulation"
        return CostEstimate(self.name, backend, shots, "free", 0.0, how,
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
        meta = compiled.result_meta()
        meta["simulated_qubits"] = (report or {}).get("active_qubits")
        if (report or {}).get("noisy"):
            counts, info = sample_noisy(q, payload, int(shots), int(seed or 0))
            meta["noisy"] = True
            meta["noise_model"] = report["noise_model"]
            meta["noise_sampling"] = info
        else:
            circ = q.load_qasm(payload)
            counts = {str(k): int(v) for k, v in dict(q.sample(circ, int(shots), int(seed or 0))).items()}
        return JobHandle(new_local_job_id("local"), self.name, backend, shots, True, DONE, meta=meta, counts=counts)

    def status(self, job: JobHandle) -> str:
        return job.status

    def result(self, job: JobHandle) -> dict[str, int]:
        return dict(job.counts or {})
