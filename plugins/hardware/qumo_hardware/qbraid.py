"""qBraid plugin: native Mojo client for the qBraid Runtime API v2 (qumo/exec/qbraid_api.mojo).

qBraid is an aggregator: one API key reaches simulators and QPUs of many vendors (qBraid's own
simulators, AWS Braket, Azure, IBM, IonQ; providers AQT, IQM, OQC, Pasqal, QuEra, Rigetti, ...).
Backends are qBraid device QRNs, vendor:provider:type:name, for example:

    qbraid:qbraid:sim:qir-sv                 free state-vector simulator (0 credits)
    rigetti:rigetti:qpu:cepheus-1-108q       Rigetti QPU
    `python -m qumo_hardware backends --provider qbraid --online` lists what the account sees.

* Credentials: only QBRAID_API_KEY (create one at https://account.qbraid.com), read by the
  Mojo client at call time and sent only in the X-API-KEY header to the qBraid API.
* Payload: QUMO compiles the circuit, then sends OpenQASM 2 when the device accepts `qasm2`,
  else OpenQASM 3 (converted natively) when it accepts `qasm3`. Devices that take only other
  formats (qiskit, pyquil, pulser, braket_ahs) and non-gate-model devices are refused.
* Target: qBraid's device record gives the qubit count and the accepted formats, not the
  coupling map or native gates, so QUMO compiles all-to-all to a generic basis and qBraid /
  the vendor compiles for the device. A warning in the target says so for QPUs.
* Cost: qBraid's own quote (GET /jobs/cost-estimate), in credits (1 credit = 0.01 USD).
  --max-cost is in credits and is checked against the quote before submission.
* Results: measurementCounts come back with the leftmost character = highest clbit (verified
  with an asymmetric circuit on qir-sv, 2026-11-09), QUMO's order; widths are checked.
"""

from __future__ import annotations

import time
from typing import Any

from .base import QUEUED, BackendInfo, CostEstimate, JobHandle, TargetSpec, check_confirmed
from .compile import CompiledCircuit, load_qumo_py
from .errors import E_INVALID_PAYLOAD, E_JOB, E_LIMIT, E_UNKNOWN_BACKEND, HINT_COST, HINT_SHOTS, HardwareError

NATIVE = "qumo/exec/qbraid_api.mojo"
MAX_SHOTS = 100_000  # qBraid does not publish a global cap; vendors enforce their own
GENERIC_BASIS = ["h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx", "rx", "ry", "rz", "cx", "cz", "swap"]
FREE_SIM = "qbraid:qbraid:sim:qir-sv"


def _native_error(exc: Exception) -> HardwareError:
    text = str(exc)
    code, _, msg = text.partition(": ")
    if not (code.startswith("QM-E") and len(code) == 7):
        code, msg = E_JOB, text
    msg, _, hint = msg.partition(" Hint: ")
    return HardwareError(code, msg, hint)


def _format_for(device: dict) -> str | None:
    ins = list(device.get("run_input_types") or [])
    if "qasm2" in ins:
        return "qasm2"
    if "qasm3" in ins:
        return "qasm3"
    return None


class QbraidProvider:
    name = "qbraid"
    payload_format = "openqasm2"
    client = "native"

    def __init__(self, **_: Any):
        self._devices: dict[str, dict] = {}

    def public_options(self) -> dict:
        return {}

    def is_local(self, backend: str) -> bool:
        return False  # every qBraid device runs remotely; submission needs confirmation (--yes)

    @staticmethod
    def _q():
        return load_qumo_py()

    def _call(self, fn: str, *args):
        try:
            return getattr(self._q(), fn)(*args)
        except Exception as exc:
            raise _native_error(exc) from None

    def _device(self, qrn: str) -> dict:
        if qrn not in self._devices:
            self._devices[qrn] = dict(self._call("qbraid_device", qrn))
        return self._devices[qrn]

    # ------------------------------------------------------------ Provider API
    def list_backends(self, online: bool = False) -> list[BackendInfo]:
        if not online:
            return [BackendInfo(FREE_SIM, self.name, "cloud-emulator", 30, None, None, False,
                                "qBraid free state-vector simulator; `--online` lists every device of the account")]
        out = []
        for d in self._call("qbraid_devices"):
            d = dict(d)
            if d["status"] == "RETIRED":
                continue
            fmt = _format_for(d)
            kind = "cloud-emulator" if d["device_type"] == "SIMULATOR" else "hardware"
            notes = f"{d['paradigm']}; inputs {', '.join(d['run_input_types'])}"
            if d["paradigm"] != "gate_model" or fmt is None:
                notes += " (not runnable from QUMO: needs a gate-model device that takes qasm2/qasm3)"
            billable = not (d["per_task"] == 0 and d["per_shot"] == 0 and d["per_minute"] == 0)
            out.append(BackendInfo(d["qrn"], self.name, kind, d["qubits"] if d["qubits"] >= 0 else None, d["status"],
                                   d["queue_depth"] if d["queue_depth"] >= 0 else None, billable, notes))
        return out

    def target(self, backend: str) -> TargetSpec:
        d = self._device(backend)
        if d["paradigm"] != "gate_model":
            raise HardwareError(E_UNKNOWN_BACKEND, f"qBraid device '{backend}' is {d['paradigm']}, not gate-model",
                                "QUMO submits gate circuits; analog Pasqal programs use examples/hardware/pasqal_mis.mojo")
        fmt = _format_for(d)
        if fmt is None:
            raise HardwareError(E_UNKNOWN_BACKEND,
                                f"qBraid device '{backend}' accepts {d['run_input_types']}, not qasm2/qasm3",
                                "pick a device that lists qasm2 or qasm3 (`backends --provider qbraid --online`)")
        sim = d["device_type"] == "SIMULATOR"
        warnings = [] if sim else [
            "qBraid publishes no coupling map or native gate set for this device: QUMO compiles all-to-all to a "
            "generic basis and qBraid / the vendor compile it for the hardware"]
        if d["status"] != "ONLINE":
            warnings.append(f"device status is {d['status']}: jobs may wait or be refused")
        nq = d["qubits"] if d["qubits"] > 0 else 30
        return TargetSpec(
            name=backend, provider=self.name, num_qubits=nq, edges=[], basis=list(GENERIC_BASIS), all_to_all=True,
            errors={}, durations={}, max_shots=MAX_SHOTS, simulator=sim, local=False,
            extras={"vendor": d["vendor"], "device_type": d["device_type"], "status": d["status"],
                    "run_input_types": d["run_input_types"], "payload": fmt, "client": NATIVE,
                    "pricing_credits": {"per_task": d["per_task"], "per_shot": d["per_shot"],
                                        "per_minute": d["per_minute"], "model": d["pricing_model"]},
                    **({"warnings": warnings} if warnings else {})},
        )

    def _payload(self, compiled: CompiledCircuit, backend: str) -> tuple[str, str]:
        fmt = compiled.target.extras.get("payload") or _format_for(self._device(backend))
        if fmt == "qasm2":
            return "qasm2", compiled.qasm
        try:
            q = self._q()
            return "qasm3", str(q.load_qasm(compiled.qasm).to_qasm3())
        except Exception as exc:
            raise _native_error(exc) from None

    def encode(self, compiled: CompiledCircuit, shots: int, backend: str) -> str:
        return self._payload(compiled, backend)[1]

    def prepare(self, compiled: CompiledCircuit, backend: str):
        fmt, text = self._payload(compiled, backend)
        return text, {"passes": [], "client": NATIVE, "program_format": fmt}

    def validate(self, payload, backend: str) -> dict:
        if not isinstance(payload, str) or not payload.startswith("OPENQASM"):
            raise HardwareError(E_INVALID_PAYLOAD, "the qBraid payload must be OpenQASM text from prepare()",
                                "let the plugin prepare the payload")
        return {"native": False, "checks": ["gate-model device that accepts the payload format",
                                            "qBraid / the vendor compile for the hardware"]}

    def estimate_cost(self, compiled: CompiledCircuit, shots: int, backend: str, payload=None) -> CostEstimate:
        available, credits, reason = self._call("qbraid_cost_estimate", backend, int(shots))
        if not available:
            return CostEstimate(self.name, backend, shots, "unknown", 0.0, f"qBraid gives no quote ({reason})",
                                notes=["treated as billable: submission needs confirmation (--yes)"])
        credits = float(credits)
        return CostEstimate(self.name, backend, shots, "credits", round(credits, 4), "qBraid quote (GET /jobs/cost-estimate)",
                            usd=round(credits / 100.0, 4),
                            notes=["1 qBraid credit = 0.01 USD", "free simulators quote 0 credits"])

    def submit(self, compiled: CompiledCircuit, shots: int, backend: str, seed: int | None = None,
               payload=None, report: dict | None = None, max_cost: float | None = None, confirm: bool = False,
               **_: Any) -> JobHandle:
        check_confirmed(self, backend, confirm)
        if shots < 1 or shots > MAX_SHOTS:
            raise HardwareError(E_LIMIT, f"shots must be in 1..{MAX_SHOTS} for {backend}", HINT_SHOTS)
        fmt, text = self._payload(compiled, backend)
        est = self.estimate_cost(compiled, shots, backend)
        if max_cost is not None and est.unit == "credits" and est.amount > max_cost:
            raise HardwareError(E_LIMIT, f"qBraid quotes {est.amount} credits, above --max-cost {max_cost}", HINT_COST)
        name = f"qumo-{int(time.time())}"
        job_qrn = str(self._call("qbraid_submit", backend, fmt, text, int(shots), name))
        meta = compiled.result_meta()
        meta.update({"device_qrn": backend, "program_format": fmt, "estimate": est.to_dict(), "job_name": name,
                     "finishing": report or {}, "client": NATIVE})
        return JobHandle(job_qrn, self.name, backend, shots, False, QUEUED, meta=meta)

    def status(self, job: JobHandle) -> str:
        if job.local:
            return job.status
        state, _raw, _msg, _cost = self._call("qbraid_job", job.job_id)
        return str(state)

    def result(self, job: JobHandle) -> dict[str, int]:
        if job.local:
            return dict(job.counts or {})
        state, raw, msg, _cost = self._call("qbraid_job", job.job_id)
        if state != "DONE":
            raise HardwareError(E_JOB, f"qBraid job '{job.job_id}' is {state} ({raw}{': ' + msg if msg else ''})")
        counts = {str(k): int(v) for k, v in dict(self._call("qbraid_result_counts", job.job_id)).items()}
        width = job.meta.get("num_clbits")
        if isinstance(width, int) and any(len(k) != width for k in counts):
            raise HardwareError(E_JOB, f"qBraid result keys have {sorted({len(k) for k in counts})} bits, "
                                f"expected {width}", "report the job QRN; the result layout does not match the circuit")
        return counts
