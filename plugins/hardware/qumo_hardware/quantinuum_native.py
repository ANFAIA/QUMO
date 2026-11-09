"""Quantinuum plugin, native client (default) for System Model H2 devices in Nexus.

The Nexus protocol runs in Mojo (qumo/exec/quantinuum_api.mojo):
* the session reuses the tokens that Quantinuum's own `qnx login` stores in ~/.qnx/auth;
* the compiled circuit is converted to a native pytket circuit (Rz / PhasedX / ZZPhase);
* the client finds or creates the project, uploads the circuit, starts the execute job
  with the HQC cap, and reads status and results.
Neither qnexus nor pytket is needed for H2 hardware, H2 cloud emulators or syntax checkers.

Handled by the Python fallback (quantinuum.py, qnexus + pytket + guppylang) instead:
* automatically: Helios devices (HUGR programs), the local emulators (H2-1LE, selene), pytket
  payloads from TKET comparison mode, and jobs submitted by that client (the runner records
  each job's client and reads it back with the same one);
* registers wider than 63 bits are split natively (c_0, c_1, ...), and `if(c==k)` conditions
  become pytket Conditional ops, as in the Python client; a condition over a split register is
  refused (H2 conditions read one register).
"""

from __future__ import annotations

import math
import time
from typing import Any

from .base import QUEUED, BackendInfo, CostEstimate, JobHandle, check_confirmed
from .compile import CompiledCircuit, load_qumo_py
from .errors import E_JOB, E_LIMIT, HINT_COST, HINT_SHOTS, HardwareError
from .quantinuum import DEVICES, MAX_SHOTS, QuantinuumProvider, _device, clbit_registers, hqc_cost

NATIVE = "qumo/exec/quantinuum_api.mojo"


def _native_error(exc: Exception) -> HardwareError:
    text = str(exc)
    code, _, msg = text.partition(": ")
    if not (code.startswith("QM-E") and len(code) == 7):
        code, msg = E_JOB, text
    msg, _, hint = msg.partition(" Hint: ")
    return HardwareError(code, msg, hint)


class QuantinuumNativeProvider(QuantinuumProvider):
    client = "native"

    @staticmethod
    def _q():
        return load_qumo_py()

    @staticmethod
    def _native_kind(backend: str) -> bool:
        family, kind, _, _ = _device(backend)
        return family == "H2" and kind in ("hardware", "cloud-emulator", "syntax-checker")

    def _counts(self, qasm: str) -> dict:
        try:
            _, nq, nc, n1, n2, nm, _regs = self._q().quantinuum_native_circuit(qasm)
        except Exception as exc:
            raise _native_error(exc) from None
        nq, nm = int(nq), int(nm)
        return {"n1q": int(n1), "n2q": int(n2), "n_spam": nm + nq, "n_measure_reset": nm, "n_init": nq}

    # ------------------------------------------------------------ Provider API
    def list_backends(self, online: bool = False) -> list[BackendInfo]:
        out = super().list_backends(online=False)
        if online:
            known = {b.name for b in out}
            try:
                devices = self._q().quantinuum_devices()
            except Exception as exc:
                raise _native_error(exc) from None
            for name, qubits in devices:
                name = str(name)
                if name not in known:
                    out.append(BackendInfo(name, self.name, _device(name)[1], int(qubits) if int(qubits) >= 0 else None,
                                           "online-listed", None, True, "from Nexus"))
        return out

    def prepare(self, compiled: CompiledCircuit, backend: str):
        if not self._native_kind(backend):
            return super().prepare(compiled, backend)
        counts = self._counts(compiled.qasm)
        return compiled.qasm, {"rebase": f"native pytket circuit Rz/PhasedX/ZZPhase ({NATIVE})",
                               "clbit_registers": [list(x) for x in clbit_registers(compiled.num_clbits, "H2")],
                               "native_counts": counts,
                               "client": NATIVE}

    def validate(self, payload, backend: str) -> dict:
        if not self._native_kind(backend) or not isinstance(payload, str):
            return super().validate(payload, backend)
        self._counts(payload)
        return {"native": True, "checks": [f"native gate set Rz/PhasedX/ZZPhase ({NATIVE})",
                                           "H2 registers of at most 63 bits; conditions on one register"]}

    def estimate_cost(self, compiled: CompiledCircuit, shots: int, backend: str, payload=None) -> CostEstimate:
        if not self._native_kind(backend) or (payload is not None and not isinstance(payload, str)):
            # a pytket payload (TKET comparison mode) is costed by the Python client
            return super().estimate_cost(compiled, shots, backend, payload=payload)
        family, kind, _, _ = _device(backend)
        c = self._counts(payload if isinstance(payload, str) else compiled.qasm)
        hqc = hqc_cost(c["n1q"], c["n2q"], c["n_spam"], shots)
        formula = "HQC = 5 + C * (N1q + 10*N2q + 5*Nm) / 5000"
        if kind == "syntax-checker":
            return CostEstimate(self.name, backend, shots, "free", 0.0, formula,
                                breakdown={**c, "hqc_if_hardware": round(hqc, 3)}, notes=["syntax checkers are free"])
        unit = "eHQC" if kind == "cloud-emulator" else "HQC"
        import os

        rate = os.environ.get("QUMO_QUANTINUUM_USD_PER_HQC")
        return CostEstimate(self.name, backend, shots, unit, round(hqc, 4), formula,
                            usd=round(hqc * float(rate), 2) if rate else None, breakdown={**c, "shots": shots},
                            notes=["Quantinuum HQC formula on the native circuit (counted in Mojo)"])

    def circuit_stats(self, circ, backend: str) -> dict:
        if not isinstance(circ, str):
            return super().circuit_stats(circ, backend)  # pytket circuit (TKET comparison mode)
        c = self._counts(circ)
        return {"two_qubit_gates": c["n2q"], "phased_x": c["n1q"], "depth": None,
                "hqc_per_1000_shots": round(hqc_cost(c["n1q"], c["n2q"], c["n_spam"], 1000), 3)}

    def submit(self, compiled: CompiledCircuit, shots: int, backend: str, seed: int | None = None,
               payload=None, max_cost: float | None = None, name: str | None = None, confirm: bool = False,
               **kw: Any) -> JobHandle:
        if not self._native_kind(backend) or (payload is not None and not isinstance(payload, str)):
            # Helios / local emulators, or a pytket payload from TKET comparison mode: the Python client
            return super().submit(compiled, shots, backend, seed=seed, payload=payload, max_cost=max_cost,
                                  name=name, confirm=confirm, **kw)
        family, kind, _, _ = _device(backend)
        check_confirmed(self, backend, confirm)
        if shots < 1 or shots > MAX_SHOTS:
            raise HardwareError(E_LIMIT, f"shots must be in 1..{MAX_SHOTS} for {backend}",
                                HINT_SHOTS + f" (Quantinuum devices accept at most {MAX_SHOTS} shots per job)")
        qasm = payload if isinstance(payload, str) else compiled.qasm
        est = self.estimate_cost(compiled, shots, backend, payload=qasm)
        cap = max_cost if max_cost is not None else self.max_cost
        if cap is not None and est.amount > cap:
            raise HardwareError(E_LIMIT, f"estimated {est.amount:.3f} {est.unit} exceeds max_cost {cap}", HINT_COST)
        server_cap = cap if cap is not None else (round(est.amount * 1.25 + 1.0, 2) if est.billable else None)
        if server_cap is not None and not math.isfinite(float(server_cap)):
            raise HardwareError(E_LIMIT, f"max_cost must be a finite number, got {server_cap}", HINT_COST)
        job_name = name or f"qumo-{int(time.time())}"
        try:
            job_id, project_id, circuit_id = self._q().quantinuum_submit(
                backend, self.project, job_name, qasm, int(shots), float(server_cap) if server_cap is not None else -1.0)
        except Exception as exc:
            raise _native_error(exc) from None
        meta = compiled.result_meta()
        meta.update({"family": family, "kind": kind, "project": self.project,
                     "clbit_registers": [list(x) for x in clbit_registers(compiled.num_clbits, "H2")],
                     "max_cost": server_cap,
                     "estimate": est.to_dict(), "job_name": job_name, "client": NATIVE,
                     "nexus_project_id": str(project_id), "nexus_circuit_id": str(circuit_id)})
        return JobHandle(str(job_id), self.name, backend, shots, False, QUEUED, meta=meta)

    def _native_job(self, job: JobHandle) -> bool:
        return not job.local and job.meta.get("family", "H2") == "H2"

    def status(self, job: JobHandle) -> str:
        if not self._native_job(job):
            return super().status(job)
        try:
            state, _ = self._q().quantinuum_status(job.job_id)
        except Exception as exc:
            raise _native_error(exc) from None
        return str(state)

    def result(self, job: JobHandle) -> dict[str, int]:
        if not self._native_job(job):
            return super().result(job)
        try:
            counts = self._q().quantinuum_result(job.job_id, int(job.meta["num_clbits"]))
        except Exception as exc:
            raise _native_error(exc) from None
        return {str(k): int(v) for k, v in dict(counts).items()}


assert "H2-1" in DEVICES  # the native client relies on the shared device table
