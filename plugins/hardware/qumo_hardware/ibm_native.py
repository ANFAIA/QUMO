"""IBM Quantum plugin, native client (default): the REST protocol runs in Mojo (qumo/exec/ibm_api.mojo).

Account (QISKIT_IBM_TOKEN + QISKIT_IBM_INSTANCE, or the account saved by qiskit-ibm-runtime in
~/.qiskit/qiskit-ibm.json), IAM login, backend listing, the target (from /configuration and
/properties), the ISA check, the OpenQASM 3 payload on physical qubits, Sampler submission with
the `cost` cap, status and results all run in the qumo_py extension over QUMO's own HTTP/TLS
stack. qiskit and qiskit-ibm-runtime are not needed for real backends.

Handled by the Python fallback (ibm.py, qiskit-ibm-runtime) instead:
* `fake_*` backends (calibration snapshots + Aer): delegated automatically, they need the SDK;
* gates outside OpenQASM 3 stdgates.inc (ecr on Eagle, fractional rzz), native-direction
  finishing, dynamic circuits, the server-side dry run and extra sampler options: refused here
  with a hint to `--client python` (QUMO_HW_CLIENT_IBM=python).
"""

from __future__ import annotations

import math
import os
from typing import Any

from .base import QUEUED, BackendInfo, CostEstimate, JobHandle, TargetSpec, check_confirmed
from .compile import CompiledCircuit, load_qumo_py
from .errors import E_INVALID_PAYLOAD, E_JOB, E_LIMIT, HINT_SHOTS, HardwareError
from .ibm import DEFAULT_USD_PER_SECOND, JOB_OVERHEAD_S, IBMProvider

NATIVE = "qumo/exec/ibm_api.mojo"
HINT_PY = "use the Python fallback client: --client python (QUMO_HW_CLIENT_IBM=python)"


def _native_error(exc: Exception) -> HardwareError:
    text = str(exc)
    code, _, msg = text.partition(": ")
    if not (code.startswith("QM-E") and len(code) == 7):
        code, msg = E_JOB, text
    msg, _, hint = msg.partition(" Hint: ")
    return HardwareError(code, msg, hint)


def _opt(v: float) -> float | None:
    return None if v is None or v < 0 else float(v)


class IBMNativeProvider(IBMProvider):
    """Same options as IBMProvider; `channel` is not needed (IBM Quantum Platform only)."""

    client = "native"

    # ------------------------------------------------------------ native calls
    @staticmethod
    def _q():
        return load_qumo_py()

    def _ids(self) -> tuple[str, str]:
        """(saved-account name, instance CRN override): identifiers only, never the key."""
        return self.account_name or "", self.instance or ""

    def _call(self, fn: str, *args):
        try:
            return getattr(self._q(), fn)(*self._ids(), *args)
        except Exception as exc:
            raise _native_error(exc) from None

    # ------------------------------------------------------------ Provider API
    def list_backends(self, online: bool = False) -> list[BackendInfo]:
        try:
            out = super().list_backends(online=False)  # offline fake snapshots, when the SDK is installed
        except HardwareError:
            out = []
        if online:
            for name, qubits, status, pending in self._call("ibm_backends"):
                out.append(BackendInfo(str(name), self.name, "hardware", int(qubits) if int(qubits) >= 0 else None,
                                       str(status) or None, int(pending) if int(pending) >= 0 else None, True,
                                       "billed by QPU time"))
        return out

    def target(self, backend: str) -> TargetSpec:
        if self.is_local(backend):
            return super().target(backend)
        t = dict(self._call("ibm_target", backend, bool(self.use_fractional_gates)))
        edges = list(t["edges"])
        directed = [(int(edges[i]), int(edges[i + 1])) for i in range(0, len(edges), 2)]
        errs = list(t["edge_errors"])
        undirected = sorted({(min(a, b), max(a, b)) for a, b in directed})
        edge_err: dict[str, float] = {}
        for (a, b), e in zip(directed, errs):
            if e >= 0:
                key = f"{min(a, b)}-{max(a, b)}"
                edge_err[key] = min(edge_err.get(key, 1.0), float(e))
        basis = [str(g) for g in t["basis"]]
        notes = []
        if self.use_fractional_gates and not t["fractional"]:
            notes.append(f"--fractional ignored: backend '{backend}' exposes no fractional gates (rzz, rx)")
        if list(t["faulty_qubits"]) or int(t["faulty_edges"]):
            notes.append(f"left out as not operational in today's calibration: qubits {list(t['faulty_qubits'])}, "
                         f"{int(t['faulty_edges'])} couplings")
        if t["fractional"]:
            notes.append("fractional gates: rzz is not in OpenQASM 3 stdgates.inc, so circuits that use it need "
                         + HINT_PY)
        errors = {k: v for k, v in (("1q", _opt(t["err_1q"])), ("2q", _opt(t["err_2q"])),
                                    ("readout", _opt(t["err_ro"]))) if v is not None}
        durations = {k: v for k, v in (("1q", _opt(t["dur_1q"])), ("2q", _opt(t["dur_2q"])),
                                       ("readout", _opt(t["dur_ro"]))) if v is not None}
        return TargetSpec(
            name=backend, provider=self.name, num_qubits=int(t["num_qubits"]), edges=undirected, basis=basis,
            all_to_all=False, errors=errors, durations=durations, max_shots=int(t["max_shots"]),
            directed_edges=None if t["symmetric"] else sorted(directed), edge_errors=edge_err,
            readout_errors=[_opt(v) for v in t["readout"]], simulator=bool(t["simulator"]), local=False,
            angle_bounds={"rzz": (0.0, math.pi / 2)} if "rzz" in basis else {},
            extras={"dt": _opt(t["dt"]), "default_rep_delay": _opt(t["rep_delay"]),
                    "processor_type": str(t["processor"]) or None, "fractional_gates_available": bool(t["fractional"]),
                    "use_fractional_gates": bool(self.use_fractional_gates), "native_2q": str(t["native_2q"]),
                    "client": NATIVE, **({"warnings": notes} if notes else {})},
        )

    def _isa(self, compiled: CompiledCircuit) -> tuple[str, float, int, int]:
        spec = compiled.target
        pairs = spec.directed_edges if spec.directed_edges is not None else spec.edges
        flat = [int(x) for e in pairs for x in e]
        d = spec.durations or {}
        try:
            qasm3, dur, twoq, active = self._q().ibm_isa_payload(
                compiled.qasm, int(spec.num_qubits), list(spec.basis), flat, spec.directed_edges is None,
                float(d.get("1q", -1.0)), float(d.get("2q", -1.0)), float(d.get("readout", -1.0)))
        except Exception as exc:
            raise _native_error(exc) from None
        return str(qasm3), float(dur), int(twoq), int(active)

    def prepare(self, compiled: CompiledCircuit, backend: str):
        """CompiledCircuit -> (ISA OpenQASM 3 on physical qubits, report). No finishing passes run here."""
        if self.is_local(backend):
            return super().prepare(compiled, backend)
        qasm3, dur, twoq, active = self._isa(compiled)
        return qasm3, {"passes": [], "client": NATIVE, "payload": "OpenQASM 3 (stdgates.inc, physical qubits $n)",
                       "duration_s": dur, "two_qubit_gates": twoq, "active_qubits": active}

    def validate(self, payload, backend: str) -> dict:
        if self.is_local(backend):
            return super().validate(payload, backend)
        if not isinstance(payload, str) or not payload.startswith("OPENQASM 3"):
            raise HardwareError(E_INVALID_PAYLOAD, "the native IBM client sends the ISA OpenQASM 3 text from prepare()",
                                "let the plugin prepare the payload")
        return {"isa": True, "checks": [f"basis gates, native couplings and width ({NATIVE})",
                                        "bound, finite angles; static circuit"]}

    def provider_compile(self, compiled: CompiledCircuit, backend: str):
        raise HardwareError(E_INVALID_PAYLOAD, "--provider-transpile runs Qiskit's transpiler", HINT_PY)

    def estimate_cost(self, compiled: CompiledCircuit, shots: int, backend: str, payload=None) -> CostEstimate:
        if self.is_local(backend):
            return super().estimate_cost(compiled, shots, backend, payload=payload)
        _, duration, _, _ = self._isa(compiled)
        rep = compiled.target.extras.get("default_rep_delay") or 250e-6
        seconds = JOB_OVERHEAD_S + shots * (rep + duration)
        rate = float(os.environ.get("QUMO_IBM_USD_PER_SECOND", DEFAULT_USD_PER_SECOND))
        return CostEstimate(
            self.name, backend, shots, "QPU-seconds", round(seconds, 3),
            "2 s + shots * (rep_delay + circuit_duration)", usd=round(seconds * rate, 2),
            breakdown={"overhead_s": JOB_OVERHEAD_S, "rep_delay_s": rep, "circuit_duration_s": duration,
                       "usd_per_second": rate, "duration_model": "ASAP with the target's median gate durations"},
            notes=["heuristic from IBM's 'Estimate job run time' guide; billed usage is QPU time",
                   "Open Plan: up to 10 free QPU minutes per month; Pay-As-You-Go starts at $96/min billed per "
                   "second (ibm.com/quantum/pricing, 2026-10-12)",
                   "override the USD rate with QUMO_IBM_USD_PER_SECOND"],
        )

    def submit(self, compiled: CompiledCircuit, shots: int, backend: str, seed: int | None = None,
               payload=None, report: dict | None = None, max_cost: float | None = None, confirm: bool = False,
               **kw: Any) -> JobHandle:
        if self.is_local(backend):
            return super().submit(compiled, shots, backend, seed=seed, payload=payload, report=report,
                                  max_cost=max_cost, confirm=confirm, **kw)
        check_confirmed(self, backend, confirm)
        if shots < 1 or shots > compiled.target.max_shots:
            raise HardwareError(E_LIMIT, f"shots must be in 1..{compiled.target.max_shots} for {backend}", HINT_SHOTS)
        if self.server_dry_run:
            raise HardwareError(E_INVALID_PAYLOAD, "the IBM server-side dry run is not available in the native client",
                                HINT_PY)
        if self.sampler_options:
            raise HardwareError(E_INVALID_PAYLOAD, "extra sampler options are not available in the native client",
                                HINT_PY)
        if not (isinstance(payload, str) and payload.startswith("OPENQASM 3")):
            payload, report = self.prepare(compiled, backend)
        cap = self.server_cap_seconds(compiled, shots, backend, payload=payload, max_cost=max_cost)
        job_id = str(self._call("ibm_submit", backend, payload, int(shots), int(cap)))
        meta = compiled.result_meta()
        meta.update({"finishing": report, "instance": self.instance, "account_name": self.account_name,
                     "sampler": "IBM REST Sampler (program_id sampler, version 2)", "client": NATIVE,
                     "server_dry_run": False, "max_execution_time": cap})
        return JobHandle(job_id, self.name, backend, shots, False, QUEUED, meta=meta)

    def status(self, job: JobHandle) -> str:
        if job.local:
            return job.status
        state, _ = self._call("ibm_status", job.job_id)
        return str(state)

    def result(self, job: JobHandle) -> dict[str, int]:
        if job.local:
            return dict(job.counts or {})
        counts = self._call("ibm_result", job.job_id, int(job.meta["num_clbits"]))
        return {str(k): int(v) for k, v in dict(counts).items()}
