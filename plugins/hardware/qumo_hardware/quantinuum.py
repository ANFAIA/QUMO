"""Quantinuum plugin (Quantinuum Nexus via `qnexus` >= 0.51, pytket-quantinuum >= 0.59).

* Credentials: only qnexus' own login (`qnexus.login()` in a browser, or
  `qnexus.login_with_credentials()`); qnexus keeps its tokens in ~/.qnx/auth.
  This module never reads, stores or prints them.
* Target: all-to-all trapped-ion device, QUMO basis rz/rx/ry/rzz, translated
  1:1 to the native gate set Rz / PhasedX / ZZPhase (Rx(t) = PhasedX(t, 0),
  Ry(t) = PhasedX(t, 1/2) in half-turns, RZZ(t) = ZZPhase(t/pi)).
* Program formats: System Model H2 devices (H2-1, H2-2, emulators H2-1E/H2-2E/
  H2-Emulator, syntax checkers H2-xSC) take pytket circuits
  (`QuantinuumConfig(device_name=...)`); Helios devices (Helios-1, Helios-1E,
  Helios-1SC, ...) take HUGR programs (`HeliosConfig(system_name=...)`), built
  here by wrapping the pytket circuit with `guppy.load_pytket`.
* Local, offline, no account: `H2-1LE` (pytket-quantinuum local emulator,
  extra `pecos`) and `selene` (Guppy + Selene statevector emulator, the
  Helios-generation runtime).
* Cost: HQC = 5 + C (N1q + 10 N2q + 5 Nm) / 5000 computed on the native
  circuit (N1q = PhasedX gates, Rz is virtual; N2q = ZZPhase/ZZMax/TK2;
  Nm = measurements + resets + initial state preparations; C = shots).
* Classical registers: QUMO payloads carry one register `c` with one bit per
  user clbit. H2 programs use integer registers of at most 63 bits
  (`max_classical_register_width` in pytket-quantinuum's device data), so a
  wider `c` is split into `c_0, c_1, ...` of 63 bits (flat clbit k ->
  c_{k // 63}[k % 63]); the layout is stored with the job and undone when
  results are read, so counts keep the user's bit order. Helios/Selene
  programs return bool arrays and keep the single register `c`.
* Classical conditions: H2 runs `if(c==k)` conditions (registers <= 63 bits).
  Helios programs are HUGR built by guppylang's pytket loader, which cannot
  lower conditional gates yet (the Selene HUGR-QIS compiler reports "Pytket op
  'Conditional' is not currently supported"), so they are refused before
  upload (QM-E955).
"""

from __future__ import annotations

import os
import time
from typing import Any

from .base import (
    CANCELLED,
    DONE,
    ERROR,
    QUEUED,
    RUNNING,
    UNKNOWN,
    BackendInfo,
    CostEstimate,
    JobHandle,
    TargetSpec,
    check_confirmed,
    new_local_job_id,
)
from .compile import CompiledCircuit, reorder_counts
from .errors import (
    E_CREDENTIALS,
    E_INVALID_PAYLOAD,
    E_JOB,
    E_LIMIT,
    E_SDK_MISSING,
    E_UNKNOWN_BACKEND,
    E_UNKNOWN_JOB,
    HINT_COST,
    HINT_LOCAL_QUBITS,
    HINT_SHOTS,
    HardwareError,
    safe_exc_text,
)
from .qasm_bridge import to_pytket

BASIS = ["rz", "rx", "ry", "rzz"]
MAX_SHOTS = 10_000  # per-job shot cap on System Model H2 (Quantinuum H2 operation guide)
LOCAL_MAX_QUBITS = 26  # state-vector memory guard for the local emulators
# Widest classical register an H2 program may use: pytket-quantinuum's device data reports
# max_classical_register_width = 63 for every H-series device (checked 2026-10-12).
H2_MAX_CREG_WIDTH = 63
# Native op types the Helios (HUGR) path is known to lower (checked with the Selene HUGR-QIS
# compiler, guppylang 1.0.0a4); anything else (conditional gates,
# classical expressions, WASM calls, ...) is refused before upload.
HELIOS_LOWERABLE = ("Rz", "PhasedX", "ZZPhase", "Measure", "Reset", "Barrier")  # verified with Selene

# Nominal error rates from public product data sheets; used only for neutral
# success-probability estimates, never for billing.
_NOMINAL = {
    "H2": {"1q": 3e-5, "2q": 1e-3, "readout": 1e-3},
    "Helios": {"1q": 2.5e-5, "2q": 7.9e-4, "readout": 5e-4},
}

# name -> (family, kind, num_qubits, note)
DEVICES: dict[str, tuple[str, str, int, str]] = {
    "H2-1": ("H2", "hardware", 56, "System Model H2 trapped-ion QPU"),
    "H2-2": ("H2", "hardware", 56, "System Model H2 trapped-ion QPU"),
    "H2-1E": ("H2", "cloud-emulator", 56, "noisy emulator of H2-1 (eHQC); 32 qubits for state vector"),
    "H2-2E": ("H2", "cloud-emulator", 56, "noisy emulator of H2-2 (eHQC)"),
    "H2-Emulator": ("H2", "cloud-emulator", 56, "H2 emulator (eHQC)"),
    "H2-1SC": ("H2", "syntax-checker", 56, "free syntax checker: compiles, returns all-zero results"),
    "H2-2SC": ("H2", "syntax-checker", 56, "free syntax checker"),
    "Helios-1": ("Helios", "hardware", 98, "Helios trapped-ion QPU (HUGR programs)"),
    "Helios-1E": ("Helios", "cloud-emulator", 98, "Helios emulator in Nexus (eHQC); set n_qubits"),
    "Helios-1E-lite": ("Helios", "cloud-emulator", 98, "lightweight Helios emulator in Nexus"),
    "Helios-1SC": ("Helios", "syntax-checker", 98, "Helios syntax checker / cost estimator"),
    "H2-1LE": ("H2", "local-emulator", 56, "pytket-quantinuum local emulator (pip extra 'pecos'), noiseless"),
    "selene": ("Helios", "local-emulator", 98, "Guppy + Selene local emulator (guppylang), noiseless state vector"),
}

_STATUS = {
    "COMPLETED": DONE,
    "QUEUED": QUEUED,
    "SUBMITTED": QUEUED,
    "RETRYING": QUEUED,
    "RUNNING": RUNNING,
    "CANCELLED": CANCELLED,
    "CANCELLING": CANCELLED,
    "ERROR": ERROR,
    "TERMINATED": ERROR,
    "DEPLETED": ERROR,
}


def _device(name: str) -> tuple[str, str, int, str]:
    if name in DEVICES:
        return DEVICES[name]
    # Unknown but well-formed names (new devices): infer from Quantinuum's naming scheme.
    family = "Helios" if name.startswith("Helios") else "H2" if name.startswith("H") else None
    if family is None:
        raise HardwareError(E_UNKNOWN_BACKEND, f"unknown Quantinuum device '{name}'")
    if name.endswith("SC"):
        kind = "syntax-checker"
    elif name.endswith("LE"):
        kind = "local-emulator"
    elif name.endswith(("E", "Emulator", "E-lite")):
        kind = "cloud-emulator"
    else:
        kind = "hardware"
    return family, kind, 98 if family == "Helios" else 56, "not in QUMO's device table; inferred"


def _pytket():
    try:
        import pytket  # noqa: F401
        from pytket.circuit import OpType  # noqa: F401
    except ImportError:
        raise HardwareError(E_SDK_MISSING, "pytket is not installed", "pip install 'qumo-hardware[quantinuum]'") from None


def _qnx():
    try:
        import qnexus as qnx
    except ImportError:
        raise HardwareError(E_SDK_MISSING, "qnexus is not installed", "pip install 'qumo-hardware[quantinuum]'") from None
    return qnx


def native_gate_counts(circ) -> dict[str, int]:
    """Counts used by the HQC formula, on a native (Rz/PhasedX/ZZPhase) pytket circuit."""
    from pytket.circuit import OpType

    n1 = n2 = nm = 0
    for cmd in circ.get_commands():
        op = cmd.op
        t = op.type
        if t == OpType.Conditional:
            t = op.op.type
        if t in (OpType.Measure, OpType.Reset):
            nm += 1
        elif t in (OpType.ZZPhase, OpType.ZZMax, OpType.TK2):
            n2 += 1
        elif t in (OpType.Barrier, OpType.Rz, OpType.Phase) or not op.is_gate():
            continue
        else:
            n1 += 1
    return {"n1q": n1, "n2q": n2, "n_spam": nm + circ.n_qubits, "n_measure_reset": nm, "n_init": circ.n_qubits}


def hqc_cost(n1q: int, n2q: int, n_spam: int, shots: int) -> float:
    """Quantinuum HQC formula: 5 + C (N1q + 10 N2q + 5 Nm) / 5000."""
    return 5.0 + shots * (n1q + 10.0 * n2q + 5.0 * n_spam) / 5000.0


def to_native(circ):
    """pytket circuit with rz/rx/ry/rzz -> Rz/PhasedX/ZZPhase.

    For QUMO payloads this is a 1:1 rename (Rx -> PhasedX(t, 0), Ry -> PhasedX(t, 1/2),
    ZZPhase kept, Rz kept); `AutoRebase` is exact for any other input as well."""
    _pytket()
    from pytket.circuit import OpType
    from pytket.passes import AutoRebase

    AutoRebase({OpType.Rz, OpType.PhasedX, OpType.ZZPhase}).apply(circ)
    return circ


def clbit_registers(num_clbits: int, family: str) -> list[tuple[str, int]]:
    """Classical register layout of a payload, as [(name, width), ...] in flat clbit order.

    H2 keeps one register `c` up to 63 bits and otherwise uses `c_0, c_1, ...`
    of 63 bits each (flat clbit k -> c_{k // 63}[k % 63]). Helios / Selene
    (bool-array results) always use one register `c`."""
    w = H2_MAX_CREG_WIDTH
    if family != "H2" or num_clbits <= w:
        return [("c", num_clbits)]
    return [(f"c_{i}", min(w, num_clbits - i * w)) for i in range((num_clbits + w - 1) // w)]


def layout_bits(layout) -> list:
    """pytket Bits of a register layout, in flat (user) clbit order."""
    from pytket.circuit import Bit

    return [Bit(str(name), i) for name, width in layout for i in range(int(width))]


def _conditional_commands(circ) -> list:
    from pytket.circuit import OpType

    return [cmd for cmd in circ.get_commands() if cmd.op.type == OpType.Conditional]


def split_classical_register(circ, layout: list[tuple[str, int]]):
    """Rename the payload's bits c[k] to the register layout (no-op for a single `c`).

    Conditions compare whole registers. QUMO's OpenQASM 2 conditions always read
    the whole register `c`, so a condition on a register that has to be split
    (> 63 bits on H2) cannot be expressed and is refused (QM-E956)."""
    n = sum(w for _, w in layout)
    if [tuple(x) for x in layout] == [("c", n)]:
        return circ
    if _conditional_commands(circ):
        raise HardwareError(
            E_LIMIT,
            f"the circuit has classical conditions on a {n}-bit register, but H2 registers hold at most "
            f"{H2_MAX_CREG_WIDTH} bits",
            f"condition on a register of at most {H2_MAX_CREG_WIDTH} bits (measure fewer clbits), or run it on "
            "a local backend (-p local)",
        )
    from pytket.circuit import Bit

    circ.rename_units({Bit("c", k): b for k, b in enumerate(layout_bits(layout))})
    return circ


def check_register_layout(circ, layout: list[tuple[str, int]]) -> None:
    """The payload's classical registers must be exactly `layout` (QM-E955 otherwise)."""
    have = sorted((r.name, r.size) for r in circ.c_registers)
    want = sorted((str(nm), int(w)) for nm, w in layout)
    if have != want:
        raise HardwareError(E_INVALID_PAYLOAD, f"payload classical registers {have} do not match the expected layout {want}",
                            "build the payload with QuantinuumProvider.prepare (or runner.plan)")


def offline_backend(device: str, n_qubits: int, local_emulator: bool = False):
    """pytket QuantinuumBackend built from static data (no network access)."""
    from pytket.circuit import OpType
    from pytket.extensions.quantinuum import QuantinuumBackend
    from pytket.extensions.quantinuum.backends.data import QuantinuumBackendData

    gate_set = frozenset({OpType.Rz, OpType.PhasedX, OpType.ZZPhase, OpType.ZZMax, OpType.TK2,
                          OpType.Measure, OpType.Reset, OpType.Barrier})
    data = QuantinuumBackendData(n_qubits=n_qubits, n_cl_reg=4000, gate_set=gate_set,
                                 local_emulator=local_emulator)
    return QuantinuumBackend(device, data=data)


def hugr_program(native_circ, n_qubits: int):
    """Wrap a native pytket circuit as a Guppy `main` (HUGR) for Helios / Selene.

    Results are emitted with tag "c" as an array of bools, index = clbit."""
    try:
        from guppylang import guppy
        from guppylang.std.builtins import array, comptime, result
        from guppylang.std.quantum import discard_array, qubit
    except Exception:
        raise HardwareError(E_SDK_MISSING, "guppylang is not installed (needed for Helios / selene)",
                            "pip install 'qumo-hardware[helios]'") from None

    payload = guppy.load_pytket("payload", native_circ)

    @guppy
    def main() -> None:
        qs = array(qubit() for _ in range(comptime(n_qubits)))
        cs = payload(qs)
        result("c", cs)
        discard_array(qs)

    return main


class QuantinuumProvider:
    name = "quantinuum"

    def __init__(self, project: str = "qumo-hardware", max_cost: float | None = None, seed: int | None = None):
        self.project = project
        self.max_cost = max_cost
        self.seed = seed

    def public_options(self) -> dict:
        return {"project": self.project}

    def is_local(self, backend: str) -> bool:
        return _device(backend)[1] == "local-emulator"

    # ------------------------------------------------------------ Provider API
    def list_backends(self, online: bool = False) -> list[BackendInfo]:
        out = [
            BackendInfo(name, self.name, kind, nq, "local" if kind == "local-emulator" else None, None,
                        kind in ("hardware", "cloud-emulator"), note)
            for name, (fam, kind, nq, note) in DEVICES.items()
        ]
        if online:
            qnx = _qnx()
            self._ensure_login(qnx)
            from qnexus.models import IssuerEnum

            known = {b.name for b in out}
            try:
                devices = qnx.devices.get_all(issuers=[IssuerEnum.QUANTINUUM])
            except Exception as exc:
                raise HardwareError(E_CREDENTIALS, f"Nexus device listing failed ({type(exc).__name__})") from None
            for d in devices:
                info = getattr(d, "backend_info", None)
                name = getattr(info, "device_name", None) or getattr(d, "device_name", None) or str(d)
                if name in known:
                    continue
                n = getattr(getattr(info, "architecture", None), "n_nodes", None) if info else None
                out.append(BackendInfo(name, self.name, _device(name)[1], n, "online-listed", None, True, "from Nexus"))
        return out

    def target(self, backend: str) -> TargetSpec:
        family, kind, nq, note = _device(backend)
        return TargetSpec(
            name=backend,
            provider=self.name,
            num_qubits=nq,
            edges=[],
            basis=list(BASIS),
            all_to_all=True,
            errors=dict(_NOMINAL[family]),
            durations={},
            max_shots=MAX_SHOTS,
            simulator=kind != "hardware",
            local=kind == "local-emulator",
            extras={"family": family, "kind": kind, "note": note,
                    "native_gates": ["Rz", "PhasedX", "ZZPhase"],
                    "errors_source": "nominal public data-sheet values, not live calibration",
                    "program_format": "HUGR (Guppy)" if family == "Helios" else "pytket circuit"},
        )

    def prepare(self, compiled: CompiledCircuit, backend: str):
        """CompiledCircuit -> native pytket circuit (validated offline)."""
        family = _device(backend)[0]
        layout = clbit_registers(compiled.num_clbits, family)
        circ = split_classical_register(to_native(to_pytket(compiled.qasm)), layout)
        circ.name = "qumo-payload"
        self.validate(circ, backend)
        report = {"rebase": "pytket AutoRebase {Rz, PhasedX, ZZPhase} (1:1 for rz/rx/ry/rzz)",
                  "clbit_registers": [list(x) for x in layout]}
        if len(layout) > 1:
            report["passes"] = [{"name": f"split creg c[{compiled.num_clbits}] into "
                                 + ", ".join(f"{nm}[{w}]" for nm, w in layout),
                                 "reason": f"H2 registers hold at most {H2_MAX_CREG_WIDTH} bits"}]
        return circ, report

    def validate(self, payload, backend: str) -> dict:
        family, kind, nq, _ = _device(backend)
        b = offline_backend(backend, nq)
        if not b.valid_circuit(payload):
            bad = [str(p) for p in b.required_predicates if not p.verify(payload)]
            raise HardwareError(E_INVALID_PAYLOAD, f"pytket QuantinuumBackend rejects the payload: {bad}",
                                "the compiled circuit is not native for this device; report it with --show-qasm")
        checks = ["pytket QuantinuumBackend.required_predicates"]
        if family == "H2":
            wide = [(r.name, r.size) for r in payload.c_registers if r.size > H2_MAX_CREG_WIDTH]
            if wide:
                raise HardwareError(E_INVALID_PAYLOAD, f"classical registers {wide} exceed the H2 limit of "
                                    f"{H2_MAX_CREG_WIDTH} bits", "build the payload with QuantinuumProvider.prepare, "
                                    "which splits wide registers")
            checks.append(f"classical registers <= {H2_MAX_CREG_WIDTH} bits")
        else:
            bad = sorted({cmd.op.type.name for cmd in payload.get_commands()
                          if cmd.op.type.name not in HELIOS_LOWERABLE})
            if bad:
                raise HardwareError(
                    E_INVALID_PAYLOAD,
                    f"'{backend}' runs HUGR programs and the pytket-to-HUGR loader cannot lower {bad} "
                    "(classically conditioned gates are not supported there yet)",
                    "run circuits with if(c==k) conditions on an H2 backend (H2-1, H2-1E, H2-1SC, H2-1LE), or write "
                    "the feed-forward directly in Guppy",
                )
            checks.append("Helios lowering: only " + "/".join(HELIOS_LOWERABLE))
        return {"native": True, "checks": checks}

    def estimate_cost(self, compiled: CompiledCircuit, shots: int, backend: str, payload=None) -> CostEstimate:
        family, kind, nq, _ = _device(backend)
        circ = payload if payload is not None else self.prepare(compiled, backend)[0]
        c = native_gate_counts(circ)
        hqc = hqc_cost(c["n1q"], c["n2q"], c["n_spam"], shots)
        formula = "HQC = 5 + C * (N1q + 10*N2q + 5*Nm) / 5000"
        notes = ["Quantinuum HQC formula (Helios: maxima over a 10-shot syntax-checker simulation)",
                 "exact quote: qnexus.circuits.cost(...) / qnexus.hugr.cost(...) on the syntax checker"]
        usd = None
        rate = os.environ.get("QUMO_QUANTINUUM_USD_PER_HQC")
        if kind in ("local-emulator", "syntax-checker"):
            return CostEstimate(self.name, backend, shots, "free", 0.0, formula, breakdown={**c, "hqc_if_hardware": round(hqc, 3)},
                                notes=["local emulator" if kind == "local-emulator" else "syntax checkers are free"])
        unit = "eHQC" if kind == "cloud-emulator" else "HQC"
        if rate:
            usd = round(hqc * float(rate), 2)
        return CostEstimate(self.name, backend, shots, unit, round(hqc, 4), formula, usd=usd,
                            breakdown={**c, "shots": shots}, notes=notes)

    def provider_compile(self, compiled: CompiledCircuit, backend: str):
        """TKET's own Quantinuum compilation of the same logical circuit (comparison mode)."""
        _pytket()
        family, kind, nq, _ = _device(backend)
        b = offline_backend(backend, nq)
        circ = to_pytket(compiled.source_qasm)
        out = b.get_compiled_circuit(circ, optimisation_level=min(max(compiled.level, 0), 3))
        return split_classical_register(out, clbit_registers(compiled.num_clbits, family))

    def circuit_stats(self, circ, backend: str) -> dict:
        c = native_gate_counts(circ)
        return {"two_qubit_gates": c["n2q"], "phased_x": c["n1q"], "depth": circ.depth(),
                "hqc_per_1000_shots": round(hqc_cost(c["n1q"], c["n2q"], c["n_spam"], 1000), 3)}

    # ------------------------------------------------------------ submission
    def _ensure_login(self, qnx) -> None:
        try:
            ok = qnx.auth.is_logged_in()
        except Exception:
            ok = False
        if not ok:
            raise HardwareError(
                E_CREDENTIALS,
                "not logged in to Quantinuum Nexus",
                "run `python -c \"import qnexus as qnx; qnx.login()\"` once (browser), or "
                "qnx.login_with_credentials(); qnexus stores its tokens in ~/.qnx/auth",
            )

    def submit(self, compiled: CompiledCircuit, shots: int, backend: str, seed: int | None = None,
               payload=None, max_cost: float | None = None, name: str | None = None, confirm: bool = False,
               **_: Any) -> JobHandle:
        family, kind, nq, _ = _device(backend)
        check_confirmed(self, backend, confirm)
        if shots < 1 or shots > MAX_SHOTS:
            raise HardwareError(E_LIMIT, f"shots must be in 1..{MAX_SHOTS} for {backend}",
                                HINT_SHOTS + f" (Quantinuum devices accept at most {MAX_SHOTS} shots per job)")
        circ = payload if payload is not None else self.prepare(compiled, backend)[0]
        layout = clbit_registers(compiled.num_clbits, family)
        check_register_layout(circ, layout)
        meta = compiled.result_meta()
        meta.update({"family": family, "kind": kind, "project": self.project,
                     "clbit_registers": [list(x) for x in layout]})
        seed = self.seed if seed is None else seed
        if kind == "local-emulator":
            if compiled.num_logical > LOCAL_MAX_QUBITS:
                raise HardwareError(E_LIMIT, f"the circuit has {compiled.num_logical} qubits; local emulation is "
                                    f"limited to {LOCAL_MAX_QUBITS}", HINT_LOCAL_QUBITS)
            counts = self._run_local(circ, shots, backend, layout, seed)
            return JobHandle(new_local_job_id("qtm-local"), self.name, backend, shots, True, DONE, meta=meta,
                             counts=counts)

        qnx = _qnx()
        self._ensure_login(qnx)
        est = self.estimate_cost(compiled, shots, backend, payload=circ)
        cap = max_cost if max_cost is not None else self.max_cost
        if cap is not None and est.amount > cap:
            raise HardwareError(E_LIMIT, f"estimated {est.amount:.3f} {est.unit} exceeds max_cost {cap}", HINT_COST)
        server_cap = cap if cap is not None else (round(est.amount * 1.25 + 1.0, 2) if est.billable else None)
        job_name = name or f"qumo-{int(time.time())}"
        try:
            project = qnx.projects.get_or_create(name=self.project)
            if family == "Helios":
                from qnexus.models import HeliosEmulatorConfig

                program = hugr_program(circ, compiled.num_qubits).compile()
                ref = qnx.hugr.upload(hugr_package=program, name=job_name, project=project)
                # n_qubits goes with the job item (start_execute_job), not into the deprecated
                # HeliosEmulatorConfig.n_qubits field (quantinuum-schemas 7.8).
                emu = HeliosEmulatorConfig() if kind == "cloud-emulator" else None
                config = qnx.HeliosConfig(system_name=backend, emulator_config=emu)
            else:
                ref = qnx.circuits.upload(circuit=circ, name=job_name, project=project)
                config = qnx.QuantinuumConfig(device_name=backend)  # no_opt=True by default: run QUMO's circuit as is
            job = qnx.start_execute_job(
                programs=[ref], n_shots=[int(shots)], backend_config=config, name=job_name, project=project,
                max_cost=[server_cap] if server_cap is not None else [],
                n_qubits=[compiled.num_qubits] if family == "Helios" else [],
            )
        except HardwareError:
            raise
        except Exception as exc:
            auth = "Authentication" in type(exc).__name__ or "Unauthori" in type(exc).__name__
            raise HardwareError(
                E_CREDENTIALS if auth else E_JOB, f"Nexus submission failed ({safe_exc_text(exc)})",
                "log in again with qnx.login()" if auth else
                "check the project, the organisation's HQC quota and the device status in the Nexus web UI; "
                "nothing was submitted if no job id was printed",
            ) from None
        meta.update({"max_cost": server_cap, "estimate": est.to_dict(), "job_name": job_name})
        return JobHandle(str(job.id), self.name, backend, shots, False, QUEUED, meta=meta)

    def _run_local(self, circ, shots: int, backend: str, layout, seed: int | None) -> dict[str, int]:
        num_clbits = sum(int(w) for _, w in layout)
        if backend == "selene":
            prog = hugr_program(circ, circ.n_qubits)
            inst = prog.emulator(n_qubits=circ.n_qubits).with_shots(int(shots))
            if seed is not None:
                inst = inst.with_seed(int(seed))
            res = inst.run()
            return counts_from_qsys(res, num_clbits)
        try:
            import pytket_pecos  # noqa: F401
        except ImportError:
            raise HardwareError(E_SDK_MISSING, "the H2 local emulator needs pytket-quantinuum[pecos]",
                                "pip install --pre 'pytket-quantinuum[pecos]', or use backend 'selene'") from None
        b = offline_backend(backend, max(circ.n_qubits, 1), local_emulator=True)
        h = b.process_circuit(circ, n_shots=int(shots), seed=seed)
        return counts_from_backend_result(b.get_result(h), num_clbits, layout)

    def _job_ref(self, job: JobHandle):
        qnx = _qnx()
        self._ensure_login(qnx)
        try:
            return qnx, qnx.jobs.get(id=job.job_id)
        except Exception:
            raise HardwareError(E_UNKNOWN_JOB, f"Nexus job '{job.job_id}' not found for this account") from None

    def _remote_status(self, qnx, ref, job: JobHandle) -> str:
        try:
            st = qnx.jobs.status(ref).status
        except Exception as exc:
            raise HardwareError(E_JOB, f"cannot read the status of Nexus job '{job.job_id}' ({safe_exc_text(exc)})",
                                "check network access and the Nexus login, then retry "
                                f"`python -m qumo_hardware status {job.job_id}`") from None
        return _STATUS.get(str(getattr(st, "value", st)).upper(), UNKNOWN)

    def status(self, job: JobHandle) -> str:
        if job.local:
            return job.status
        qnx, ref = self._job_ref(job)
        return self._remote_status(qnx, ref, job)

    def result(self, job: JobHandle) -> dict[str, int]:
        if job.local:
            return dict(job.counts or {})
        qnx, ref = self._job_ref(job)
        st = self._remote_status(qnx, ref, job)
        if st != DONE:
            raise HardwareError(E_JOB, f"Nexus job '{job.job_id}' is {st}")
        try:
            res = qnx.jobs.results(ref)[0].download_result()
        except Exception as exc:
            raise HardwareError(E_JOB, f"cannot download the result of Nexus job '{job.job_id}' ({safe_exc_text(exc)})",
                                "retry later; the result stays available in Nexus") from None
        n = int(job.meta["num_clbits"])
        if hasattr(res, "register_counts"):
            return counts_from_qsys(res, n)
        return counts_from_backend_result(res, n, job.meta.get("clbit_registers"))


def counts_from_backend_result(res, num_clbits: int, registers=None) -> dict[str, int]:
    """pytket BackendResult -> QUMO counts.

    `registers` is the payload's register layout ([(name, width), ...] in flat
    clbit order, default one register `c`); result tuples are read over those
    bits, so flat clbit k is the k-th bit of the layout."""
    layout = registers or [("c", num_clbits)]
    cbits = layout_bits(layout)
    if len(cbits) != num_clbits:
        raise HardwareError(E_JOB, f"register layout {layout} has {len(cbits)} bits, expected {num_clbits}")
    counts = res.get_counts(cbits=cbits)
    return reorder_counts(counts, list(range(num_clbits)), num_clbits, leftmost_is_first=True)


def counts_from_qsys(res, num_clbits: int, register: str = "c") -> dict[str, int]:
    """QsysResult / Selene EmulatorResult (register strings, c[0] first) -> QUMO counts."""
    regs = res.register_counts()
    if register not in regs:
        raise HardwareError(E_JOB, f"result has no register '{register}' (got {sorted(regs)})")
    return reorder_counts(regs[register], list(range(num_clbits)), num_clbits, leftmost_is_first=True)
