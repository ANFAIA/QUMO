"""IonQ plugin: thin adapter over QUMO's native Mojo IonQ client (qumo/exec/ionq_api.mojo).

No vendor SDK, no Python HTTP and no Python JSON on the IonQ path. Everything that talks
to IonQ runs in Mojo through the qumo_py extension:

* request body: OpenQASM 2 parsed by QUMO -> `ionq.circuit.v1` (gateset `qis`), the
  same circuit list qiskit-ionq 1.1.1 sends (differential test in tests/test_ionq.py);
* transport: QUMO's HTTP/1.1 + TLS client (qumo/net, OpenSSL 3 loaded at run time),
  redirects refused, https only (plain http only to a loopback host, for tests);
* results: the probability histogram is parsed by QUMO's JSON parser and sampled into
  counts with QUMO's seeded RNG (`--seed`, 0 when not given). Counts are reproducible
  here, but not equal to qiskit-ionq's numpy draws.

Credentials: only IONQ_API_KEY, read by the Mojo client from the environment at call
time. This module never reads, prints, logs or stores it; it is not in JobHandle.meta,
not in the job store and not in error messages. `IONQ_API_URL` overrides the API base.

Backends: the IonQ simulator (API name `simulator`, 29 qubits) with a noise model:
`aria-1`, `aria-2`, `forte-1`, `forte-enterprise-1` (QUMO names: `ionq_simulator:<model>`).
`ideal` is refused. QPUs are listed by `backends --online` but not enabled for submission.

Cost: IonQ publishes no simulator price and its estimate endpoint refuses simulator
backends, so the cost unit is "unknown" and the amount is 0.
"""

from __future__ import annotations

from typing import Any

from .base import (
    QUEUED,
    BackendInfo,
    CostEstimate,
    JobHandle,
    TargetSpec,
    check_confirmed,
)
from .compile import CompiledCircuit, load_qumo_py
from .errors import (
    E_INVALID_PAYLOAD,
    E_JOB,
    E_LIMIT,
    E_UNKNOWN_BACKEND,
    HINT_SHOTS,
    HardwareError,
)

API_KEY_VAR = "IONQ_API_KEY"
API_URL_VAR = "IONQ_API_URL"

SDK_BACKEND = "ionq_simulator"  # QUMO backend prefix (kept from the qiskit-ionq naming)
API_SIMULATOR = "simulator"
SIMULATOR_QUBITS = 29  # GET /backends on the IonQ API (read 2026-11-09)
NOISE_MODELS = ("aria-1", "aria-2", "forte-1", "forte-enterprise-1")
# The noisy-simulator shot limit is not published; the QPU cap (10,000) is used.
MAX_SHOTS = 10_000

# QIS gates (qiskit names) the compiler may keep; qumo/exec/ionq_api.mojo maps each one.
IONQ_BASIS = ["h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx", "rx", "ry", "rz",
              "cx", "cy", "cz", "swap", "rxx", "ryy", "rzz"]

NATIVE = "qumo/exec/ionq_api.mojo"


def _native_error(exc: Exception, hint: str = "") -> HardwareError:
    """Mojo error text 'QM-E9xx: message[ Hint: ...]' -> HardwareError with that code."""
    text = str(exc)
    code, _, msg = text.partition(": ")
    if not (code.startswith("QM-E") and len(code) == 7):
        code, msg = E_JOB, text
    msg, _, native_hint = msg.partition(" Hint: ")
    return HardwareError(code, msg, native_hint or hint)


def _parse_backend(name: str) -> str:
    """'ionq_simulator:<noise model>' -> noise model. Anything else is refused (QM-E953)."""
    prefix = f"{SDK_BACKEND}:"
    if name.startswith("ionq_qpu"):
        raise HardwareError(E_UNKNOWN_BACKEND, f"IonQ QPU backend '{name}' is not enabled in the QUMO plugin",
                            "use the simulator, e.g. --backend ionq_simulator:aria-1; QPU submission is not "
                            "available in this plugin version")
    if not name.startswith(prefix):
        raise HardwareError(E_UNKNOWN_BACKEND, f"unknown IonQ backend '{name}'",
                            f"use {prefix}<noise model>, one of: {', '.join(prefix + m for m in NOISE_MODELS)}")
    noise = name[len(prefix):]
    if noise == "ideal":
        raise HardwareError(E_UNKNOWN_BACKEND,
                            "the 'ideal' noise model is not supported (it returns exact probabilities, "
                            "not shot-sampled results)",
                            f"use one of: {', '.join(prefix + m for m in NOISE_MODELS)}")
    if noise not in NOISE_MODELS:
        raise HardwareError(E_UNKNOWN_BACKEND, f"IonQ simulator noise model '{noise}' is not supported",
                            f"use one of: {', '.join(NOISE_MODELS)}")
    return noise


def _native_2q(noise: str) -> str | None:
    """Native two-qubit gate of the noise model's device family (aria: ms, forte: zz)."""
    if noise.startswith("aria"):
        return "ms"
    if noise.startswith("forte"):
        return "zz"
    return None


class IonQProvider:
    name = "ionq"
    payload_format = "openqasm2"
    client = "native"

    def __init__(self, **_: Any):
        """No options: the registry may pass CLI options (seed, ...) that IonQ does not use here."""

    def public_options(self) -> dict:
        """Nothing to persist: the API key is never stored with a job."""
        return {}

    def is_local(self, backend: str) -> bool:
        return False  # every IonQ backend runs in the IonQ cloud and needs confirmation

    # ------------------------------------------------------------ native calls
    @staticmethod
    def _q():
        return load_qumo_py()

    def _check(self, qasm: str, noise: str) -> tuple[list[int], int, int]:
        """Build the request natively (no network): gateset, measurements and width are checked."""
        try:
            _, qmap, nclbits, dense = self._q().ionq_request(qasm, 1, noise, 0, "qumo")
        except Exception as exc:
            raise _native_error(exc, "decompose the gates into IONQ_BASIS and measure only at the end") from None
        if int(dense) > SIMULATOR_QUBITS:
            raise HardwareError(E_LIMIT, f"the circuit uses {int(dense)} qubits; the IonQ simulator has "
                                f"{SIMULATOR_QUBITS}", "reduce the circuit width")
        return [int(v) for v in qmap], int(nclbits), int(dense)

    # ------------------------------------------------------------ Provider API
    def list_backends(self, online: bool = False) -> list[BackendInfo]:
        out = [
            BackendInfo(f"{SDK_BACKEND}:{noise}", self.name, "cloud-emulator", SIMULATOR_QUBITS, None, None, True,
                        f"IonQ simulator with the {noise} noise model")
            for noise in NOISE_MODELS
        ]
        if online:
            out += self._online_backends()
        return out

    def _online_backends(self) -> list[BackendInfo]:
        """Native GET /backends (public, read-only; the key is not sent). QPUs are listed, not enabled."""
        try:
            rows = self._q().ionq_backends()
        except Exception as exc:
            raise _native_error(exc, f"check {API_URL_VAR} and the network access to the IonQ API") from None
        out = []
        for api_name, status, qubits in rows:
            api_name = str(api_name)
            if api_name == API_SIMULATOR:
                continue  # listed statically above with its noise models
            out.append(BackendInfo(api_name.replace("qpu.", "ionq_qpu."), self.name, "hardware",
                                   int(qubits) or None, str(status) or None, None, True,
                                   "QPU: listed only, not enabled for submission in the QUMO plugin"))
        return out

    def target(self, backend: str) -> TargetSpec:
        noise = _parse_backend(backend)
        return TargetSpec(
            name=backend,
            provider=self.name,
            num_qubits=SIMULATOR_QUBITS,
            edges=[],
            basis=list(IONQ_BASIS),
            all_to_all=True,
            errors={},  # no calibration data is published for the simulator
            durations={},
            max_shots=MAX_SHOTS,
            simulator=True,
            local=False,
            extras={
                "api_backend": API_SIMULATOR,
                "noise_model": noise,
                "noise_models": list(NOISE_MODELS),
                "native_2q_gate": _native_2q(noise),
                "client": NATIVE,
                "warnings": [f"max shots {MAX_SHOTS} is the QPU cap: the noisy-simulator limit is not published"],
            },
        )

    def prepare(self, compiled: CompiledCircuit, backend: str):
        """CompiledCircuit -> OpenQASM 2 payload, checked by building the IonQ request natively."""
        noise = _parse_backend(backend)
        qmap, _, dense = self._check(compiled.qasm, noise)
        report = {"passes": [], "translation": f"OpenQASM 2 -> IonQ ionq.circuit.v1 (qis) in Mojo ({NATIVE}); "
                                               "IonQ compiles to its native gates (gpi, gpi2, ms) on its side",
                  "ionq_qubits": dense, "clbit_qubit": qmap}
        return compiled.qasm, report

    def validate(self, payload, backend: str) -> dict:
        noise = _parse_backend(backend)
        if not isinstance(payload, str):
            raise HardwareError(E_INVALID_PAYLOAD, "the IonQ payload must be OpenQASM 2 text",
                                "let the plugin prepare the payload")
        self._check(payload, noise)
        return {"native": False, "checks": [f"gates in the IonQ QIS gateset ({NATIVE})",
                                            "measurements only at the end",
                                            f"qubits <= {SIMULATOR_QUBITS}"]}

    def estimate_cost(self, compiled: CompiledCircuit, shots: int, backend: str, payload=None) -> CostEstimate:
        _parse_backend(backend)
        return CostEstimate(
            self.name, backend, shots, "unknown", 0.0,
            "no public price for the IonQ simulator",
            notes=["IonQ publishes no simulator price; GET /jobs/estimate refuses simulator backends "
                   "('Cost estimation is only available for QPU backends')",
                   "treated as remote and billable: submission needs confirmation (--yes)"],
        )

    def submit(self, compiled: CompiledCircuit, shots: int, backend: str, seed: int | None = None,
               payload=None, report: dict | None = None, max_cost: float | None = None, confirm: bool = False,
               **_: Any) -> JobHandle:
        check_confirmed(self, backend, confirm)
        noise = _parse_backend(backend)
        if shots < 1 or shots > MAX_SHOTS:
            raise HardwareError(E_LIMIT, f"shots must be in 1..{MAX_SHOTS} for {backend}", HINT_SHOTS)
        qasm = payload if isinstance(payload, str) else compiled.qasm
        self._check(qasm, noise)  # width and gateset, before anything leaves the machine
        sampler_seed = int(seed or 0)
        try:
            job_id, qmap, nclbits, dense = self._q().ionq_submit(qasm, int(shots), noise, sampler_seed, "qumo")
        except Exception as exc:
            raise _native_error(exc, "nothing was run if this was refused; check IONQ_API_KEY, the noise "
                                     "model and the shot count") from None
        meta = compiled.result_meta()
        meta.update({"api_backend": API_SIMULATOR, "noise_model": noise, "sampler_seed": sampler_seed,
                     "clbit_qubit": [int(v) for v in qmap], "ionq_qubits": int(dense),
                     "finishing": report or {}, "max_cost": max_cost, "client": NATIVE})
        meta["num_clbits"] = int(nclbits)
        return JobHandle(str(job_id), self.name, backend, shots, False, QUEUED, meta=meta)

    def status(self, job: JobHandle) -> str:
        if job.local:
            return job.status
        try:
            return str(self._q().ionq_status(job.job_id))
        except Exception as exc:
            raise _native_error(exc, f"check {API_KEY_VAR} and the network access, then retry") from None

    def result(self, job: JobHandle) -> dict[str, int]:
        if job.local:
            return dict(job.counts or {})
        qmap = job.meta.get("clbit_qubit")
        if not isinstance(qmap, list):
            raise HardwareError(E_JOB, f"IonQ job '{job.job_id}' has no measurement map in the job store",
                                "this job was submitted by an older plugin version; read it in the IonQ console")
        try:
            counts = self._q().ionq_result(job.job_id, [int(v) for v in qmap], int(job.shots),
                                           int(job.meta.get("sampler_seed") or 0))
        except Exception as exc:
            raise _native_error(exc, "retry later; IonQ keeps results of finished jobs") from None
        return {str(k): int(v) for k, v in dict(counts).items()}
