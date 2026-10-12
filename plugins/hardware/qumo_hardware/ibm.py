"""IBM Quantum plugin (qiskit-ibm-runtime >= 0.50, IBM Quantum Platform).

* Credentials: only the SDK's standard mechanisms (QISKIT_IBM_TOKEN /
  QISKIT_IBM_INSTANCE / QISKIT_IBM_CHANNEL environment variables, or an account
  saved with `QiskitRuntimeService.save_account(...)` in ~/.qiskit/qiskit-ibm.json).
  This module never reads, stores or prints a token.
* Target: built from `backend.target` (basis rz/sx/x + cz (Heron) or ecr
  (Eagle); fractional rzz/rx on Heron when `use_fractional_gates=True`).
* Execution: the non-deprecated client-side `Sampler`
  (`qiskit_ibm_runtime.executor_sampler.Sampler`, built on `Executor`) in job
  mode. The server-side `SamplerV2` is deprecated since 0.50.
* Payload: QUMO's compiled circuit, translated 1:1 to an ISA `QuantumCircuit`.
  Only semantics-preserving finishing is applied when the device demands it
  (native gate direction for ECR/CX, rzz angle folding for fractional gates) and
  it is reported in the job metadata. Validation uses qiskit-ibm-runtime's own
  `is_isa_circuit` plus Qiskit's `GatesInBasis`.
* Local mode: backend names starting with `fake_` (FakeTorino, FakeFez, ...)
  run on Qiskit Aer with the snapshot's noise model; no account needed. Aer
  simulates only the active qubits, with memory exponential in their number,
  so more than LOCAL_MAX_ACTIVE_QUBITS active qubits are refused (QM-E956).
* Server-side cap: every remote job carries `max_execution_time` (IBM cancels
  a job whose QPU usage exceeds it): --max-cost if given (QPU-seconds), else
  max(SERVER_CAP_MIN_S, SERVER_CAP_FACTOR x estimate).
* Errors from the IBM SDK are reported as QM-E95x with hints; their text is
  shortened and anything that looks like a credential is redacted.
"""

from __future__ import annotations

import math
import os
import statistics
import warnings
from functools import lru_cache
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
from .compile import CompiledCircuit, compile_for_target
from .errors import (
    E_CREDENTIALS,
    E_INVALID_PAYLOAD,
    E_JOB,
    E_LIMIT,
    E_SDK_MISSING,
    E_UNKNOWN_BACKEND,
    E_UNKNOWN_JOB,
    HINT_LOCAL_QUBITS,
    HINT_SHOTS,
    HardwareError,
    safe_exc_text,
)
from .qasm_bridge import to_qiskit

ONE_Q = ("rz", "sx", "x")
TWO_Q = ("cz", "ecr", "cx")
FRACTIONAL = ("rzz", "rx")
DEFAULT_USD_PER_SECOND = 1.60  # Pay-As-You-Go list price: "starts at $96 USD / minute" (ibm.com/quantum/pricing)
JOB_OVERHEAD_S = 2.0  # per sub-job overhead in IBM's run-time estimate guide
LOCAL_MAX_ACTIVE_QUBITS = 26  # Aer noisy state vector on fake backends (memory guard, same as other local emulators)
SERVER_CAP_FACTOR = 2.0  # default max_execution_time = factor x estimated QPU seconds ...
SERVER_CAP_MIN_S = 30  # ... but at least this many seconds (the estimate is a heuristic)
FRACTIONAL_FAKE = "fake_fractional"  # offline snapshot with fractional gates (rzz, rx)

# Exceptions raised inside the IBM SDK (other than the ones handled explicitly).
_IBM_HINT_NET = "check network access, the account's channel/instance and https://quantum.cloud.ibm.com status"

_STATUS = {
    "INITIALIZING": QUEUED,
    "QUEUED": QUEUED,
    "VALIDATING": QUEUED,
    "RUNNING": RUNNING,
    "DONE": DONE,
    "ERROR": ERROR,
    "CANCELLED": CANCELLED,
}


def _sdk():
    try:
        import qiskit_ibm_runtime  # noqa: F401
    except ImportError:
        raise HardwareError(E_SDK_MISSING, "qiskit-ibm-runtime is not installed",
                            "pip install 'qumo-hardware[ibm]'") from None
    return qiskit_ibm_runtime


@lru_cache(maxsize=1)
def fake_provider():
    """qiskit-ibm-runtime's offline snapshots (built once per process).

    Building the provider instantiates every snapshot; fake_nighthawk then warns
    that its properties are not representative. That warning concerns only
    fake_nighthawk, so it is silenced here and attached to that backend's
    TargetSpec instead (see target_from_qiskit)."""
    _sdk()
    from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"Properties of fake_\w+ are not intended", category=UserWarning)
        return FakeProviderForBackendV2()


def _median(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return float(statistics.median(xs)) if xs else None


def _cfg(backend, attr, default=None):
    try:
        conf = backend.configuration()
        return getattr(conf, attr, default)
    except Exception:
        return getattr(backend, attr, default)


def target_from_qiskit(backend, use_fractional_gates: bool = False, provider: str = "ibm") -> TargetSpec:
    """Neutral TargetSpec from a Qiskit BackendV2 (real or fake)."""
    t = backend.target
    names = set(t.operation_names)
    twoq = [g for g in TWO_Q if g in names]
    if not twoq:
        raise HardwareError(E_UNKNOWN_BACKEND, f"backend '{backend.name}' exposes no cz/ecr/cx gate")
    basis = [g for g in ONE_Q if g in names] + twoq
    notes: list[str] = []
    if use_fractional_gates:
        basis += [g for g in FRACTIONAL if g in names]
        if "rzz" not in names:
            notes.append(f"--fractional ignored: backend '{backend.name}' exposes no fractional gates (rzz, rx); "
                         f"Heron devices fetched with use_fractional_gates=True do, offline use '{FRACTIONAL_FAKE}'")
    if backend.name == "fake_nighthawk":
        notes.append("fake_nighthawk: qiskit-ibm-runtime warns that its properties are not typical Nighthawk "
                     "error values")

    directed: set[tuple[int, int]] = set()
    undirected: set[tuple[int, int]] = set()
    edge_err: dict[str, float] = {}
    two_errs: list[float] = []
    two_durs: list[float] = []
    for g in twoq:
        for qargs, props in t[g].items():
            if qargs is None or len(qargs) != 2:
                continue
            a, b = int(qargs[0]), int(qargs[1])
            directed.add((a, b))
            undirected.add((min(a, b), max(a, b)))
            if props is not None:
                if props.error is not None:
                    two_errs.append(props.error)
                    key = f"{min(a, b)}-{max(a, b)}"
                    edge_err[key] = min(edge_err.get(key, 1.0), float(props.error))
                if props.duration is not None:
                    two_durs.append(props.duration)
    symmetric = all((b, a) in directed for a, b in directed)

    def _props(gate):
        if gate not in names:
            return [], []
        errs, durs = [], []
        for props in t[gate].values():
            if props is None:
                continue
            errs.append(props.error)
            durs.append(props.duration)
        return errs, durs

    sx_err, sx_dur = _props("sx")
    ro_err, ro_dur = _props("measure")
    readout: list[float | None] = [None] * t.num_qubits
    if "measure" in names:
        for qargs, props in t["measure"].items():
            if qargs is not None and props is not None:
                readout[qargs[0]] = props.error

    errors = {k: v for k, v in (("1q", _median(sx_err)), ("2q", _median(two_errs)), ("readout", _median(ro_err))) if v is not None}
    durations = {k: v for k, v in (("1q", _median(sx_dur)), ("2q", _median(two_durs)), ("readout", _median(ro_dur))) if v is not None}
    name = backend.name
    rep_delay = _cfg(backend, "default_rep_delay", None)
    if rep_delay is None:
        rep_delay = getattr(backend, "default_rep_delay", None)
    return TargetSpec(
        name=name,
        provider=provider,
        num_qubits=int(t.num_qubits),
        edges=sorted(undirected),
        basis=basis,
        all_to_all=False,
        errors=errors,
        durations=durations,
        max_shots=int(_cfg(backend, "max_shots", 100_000) or 100_000),
        directed_edges=None if symmetric else sorted(directed),
        edge_errors=edge_err,
        readout_errors=readout,
        simulator=bool(_cfg(backend, "simulator", False)),
        local=name.startswith("fake_"),
        extras={
            "dt": getattr(t, "dt", None),
            "default_rep_delay": rep_delay,
            "processor_type": _cfg(backend, "processor_type", None),
            "fractional_gates_available": "rzz" in names,
            "use_fractional_gates": bool(use_fractional_gates),
            "native_2q": twoq[0],
            **({"warnings": notes} if notes else {}),
        },
    )


def _absorb_global_phase_gates(qc):
    """Fold zero-qubit GlobalPhaseGate instructions into circuit.global_phase
    (FoldRzzAngle can emit them; they are not target instructions)."""
    if "global_phase" not in qc.count_ops():
        return qc
    out = qc.copy_empty_like()
    phase = qc.global_phase
    for inst in qc.data:
        if inst.operation.name == "global_phase":
            phase += inst.operation.params[0]
        else:
            out.append(inst)
    out.global_phase = phase
    return out


def counts_from_primitive_result(result, num_clbits: int) -> dict[str, int]:
    """PrimitiveResult (one PUB) -> QUMO counts over the user's clbits."""
    pub = result[0]
    counts = pub.join_data().get_counts()
    out: dict[str, int] = {}
    for k, v in counts.items():
        key = k.replace(" ", "")
        if len(key) != num_clbits:
            raise HardwareError(E_JOB, f"result has {len(key)} bits per shot, expected {num_clbits}")
        out[key] = out.get(key, 0) + int(v)
    return out


class IBMProvider:
    name = "ibm"

    def __init__(
        self,
        channel: str | None = None,
        instance: str | None = None,
        account_name: str | None = None,
        use_fractional_gates: bool = False,
        service: Any = None,
        sampler_options: dict | None = None,
        server_dry_run: bool = False,
    ):
        self.channel = channel
        self.instance = instance
        self.account_name = account_name
        self.use_fractional_gates = use_fractional_gates
        self._service = service
        self._backends: dict[str, Any] = {}
        self.sampler_options = sampler_options or {}
        # IBM's server-side dry run (qiskit-ibm-runtime 0.50): validates the job and estimates
        # usage without using the QPU; results are mock data and access may be restricted.
        self.server_dry_run = server_dry_run

    # ------------------------------------------------------------ plumbing
    def public_options(self) -> dict:
        """Constructor options worth persisting with a job (identifiers only, never tokens)."""
        return {"channel": self.channel, "instance": self.instance, "account_name": self.account_name,
                "use_fractional_gates": self.use_fractional_gates}

    def is_local(self, backend: str) -> bool:
        return backend.startswith("fake_")

    def service(self):
        if self._service is not None:
            return self._service
        _sdk()
        from qiskit_ibm_runtime import QiskitRuntimeService
        from qiskit_ibm_runtime.accounts.exceptions import AccountNotFoundError, AccountsError
        from qiskit_ibm_runtime.exceptions import IBMError

        kwargs = {k: v for k, v in (("channel", self.channel), ("instance", self.instance), ("name", self.account_name)) if v}
        try:
            self._service = QiskitRuntimeService(**kwargs)
        except AccountNotFoundError:
            raise HardwareError(
                E_CREDENTIALS,
                "no IBM Quantum account found (QISKIT_IBM_TOKEN is unset and no saved account in ~/.qiskit/qiskit-ibm.json)",
                "create an API key at https://quantum.cloud.ibm.com, then `export QISKIT_IBM_TOKEN=...` and "
                "`export QISKIT_IBM_INSTANCE=<CRN or instance name>` (or QiskitRuntimeService.save_account(token=..., instance=...))",
            ) from None
        except (AccountsError, IBMError) as exc:
            raise HardwareError(E_CREDENTIALS, f"IBM Quantum rejected the account ({type(exc).__name__})") from None
        except Exception as exc:  # network errors etc.; the message may echo request details, keep only the type
            raise HardwareError(E_CREDENTIALS, f"cannot connect to IBM Quantum ({type(exc).__name__})",
                                "check network access and the account's channel/instance") from None
        return self._service

    def backend(self, name: str):
        if name in self._backends:
            return self._backends[name]
        _sdk()
        from qiskit.providers.exceptions import QiskitBackendNotFoundError

        if self.is_local(name):
            try:
                b = fake_provider().backend(name)
            except QiskitBackendNotFoundError:
                raise HardwareError(E_UNKNOWN_BACKEND, f"unknown IBM fake backend '{name}'") from None
        else:
            svc = self.service()
            try:
                b = svc.backend(name, use_fractional_gates=self.use_fractional_gates)
            except QiskitBackendNotFoundError:
                raise HardwareError(E_UNKNOWN_BACKEND, f"IBM backend '{name}' is not available to this account") from None
            except HardwareError:
                raise
            except Exception as exc:
                raise HardwareError(_classify(exc, E_UNKNOWN_BACKEND),
                                    f"IBM Quantum could not return backend '{name}' ({safe_exc_text(exc)})",
                                    _IBM_HINT_NET + "; list what the account sees with "
                                    "`python -m qumo_hardware backends --provider ibm --online`") from None
        self._backends[name] = b
        return b

    # ------------------------------------------------------------ Provider API
    def list_backends(self, online: bool = False) -> list[BackendInfo]:
        out = []
        for b in fake_provider().backends():
            if b.name == FRACTIONAL_FAKE:
                out.append(BackendInfo(b.name, self.name, "local-emulator", b.num_qubits, "local", 0, False,
                                       "snapshot with fractional gates rzz/rx: use with --fractional"))
            elif b.num_qubits >= 100:  # current-generation snapshots (Eagle / Heron / Nighthawk)
                out.append(BackendInfo(b.name, self.name, "local-emulator", b.num_qubits, "local", 0, False,
                                       "calibration snapshot + Aer noise model"))
        if online:
            try:
                remote = self.service().backends()
            except HardwareError:
                raise
            except Exception as exc:
                raise HardwareError(_classify(exc, E_CREDENTIALS), f"IBM backend listing failed ({safe_exc_text(exc)})",
                                    _IBM_HINT_NET) from None
            for b in remote:
                try:
                    st = b.status()
                    status, pending = st.status_msg, st.pending_jobs
                except Exception:
                    status, pending = None, None
                out.append(BackendInfo(b.name, self.name, "hardware", b.num_qubits, status, pending, True,
                                       "billed by QPU time"))
        return out

    def target(self, backend: str) -> TargetSpec:
        return target_from_qiskit(self.backend(backend), self.use_fractional_gates, self.name)

    def prepare(self, compiled: CompiledCircuit, backend: str):
        """CompiledCircuit -> (ISA QuantumCircuit, finishing report)."""
        b = self.backend(backend)
        qc = to_qiskit(compiled.qasm)
        report: dict[str, Any] = {"passes": []}
        t = b.target
        spec = compiled.target
        if spec.directed_edges is not None:
            from qiskit.transpiler import PassManager
            from qiskit.transpiler.passes import GateDirection, Optimize1qGatesDecomposition

            native = set(map(tuple, spec.directed_edges))
            flips = sum(
                1 for inst in qc.data
                if inst.operation.num_qubits == 2 and inst.operation.name in TWO_Q
                and (qc.find_bit(inst.qubits[0]).index, qc.find_bit(inst.qubits[1]).index) not in native
            )
            if flips:
                qc = PassManager([GateDirection(None, target=t), Optimize1qGatesDecomposition(target=t)]).run(qc)
                report["passes"].append({"name": "GateDirection+Optimize1qGatesDecomposition", "flipped": flips})
        if "rzz" in qc.count_ops():
            from qiskit.transpiler import PassManager
            from qiskit.transpiler.passes import Optimize1qGatesDecomposition
            from qiskit_ibm_runtime.transpiler.passes import FoldRzzAngle

            qc = PassManager([FoldRzzAngle(), Optimize1qGatesDecomposition(target=t)]).run(qc)
            qc = _absorb_global_phase_gates(qc)
            report["passes"].append({"name": "FoldRzzAngle+Optimize1qGatesDecomposition"})
        self.validate(qc, backend)
        return qc, report

    def validate(self, payload, backend: str) -> dict:
        from qiskit.transpiler import PassManager
        from qiskit.transpiler.passes import GatesInBasis
        from qiskit_ibm_runtime.utils.utils import is_isa_circuit

        t = self.backend(backend).target
        msg = is_isa_circuit(payload, t)
        if msg:
            raise HardwareError(E_INVALID_PAYLOAD, f"not an ISA circuit for {backend}: {msg}")
        pm = PassManager([GatesInBasis(target=t)])
        pm.run(payload)
        if not pm.property_set.get("all_gates_in_basis", False):
            raise HardwareError(E_INVALID_PAYLOAD, f"Qiskit GatesInBasis rejects the circuit for {backend}")
        if self.use_fractional_gates and "rzz" in payload.count_ops():
            from qiskit_ibm_runtime.utils.utils import _is_valid_rzz_pub_helper  # noqa: PLC0415

            bad = _is_valid_rzz_pub_helper(payload)
            if isinstance(bad, str) and bad:
                raise HardwareError(E_INVALID_PAYLOAD, bad)
        return {"isa": True, "checks": ["qiskit_ibm_runtime.is_isa_circuit", "qiskit.GatesInBasis"]}

    def provider_compile(self, compiled: CompiledCircuit, backend: str):
        """Qiskit's own preset pass manager on the same logical circuit (comparison)."""
        from qiskit.transpiler import generate_preset_pass_manager

        b = self.backend(backend)
        logical = to_qiskit(compiled.source_qasm)
        pm = generate_preset_pass_manager(optimization_level=min(max(compiled.level, 0), 3), backend=b,
                                          seed_transpiler=compiled.seed)
        return pm.run(logical)

    def circuit_stats(self, qc, backend: str) -> dict:
        t = self.backend(backend).target
        ops = dict(qc.count_ops())
        two = sum(1 for inst in qc.data if inst.operation.num_qubits == 2 and inst.operation.name not in ("barrier",))
        try:
            dur = float(qc.estimate_duration(t, unit="s"))
        except Exception:
            dur = None
        return {"two_qubit_gates": two, "depth": qc.depth(), "ops": {k: int(v) for k, v in ops.items()},
                "duration_s": dur}

    def estimate_cost(self, compiled: CompiledCircuit, shots: int, backend: str, payload=None) -> CostEstimate:
        if self.is_local(backend):
            return CostEstimate(self.name, backend, shots, "free", 0.0, "local Aer simulation",
                                notes=["local simulation of a fake backend; no IBM account used"])
        qc = payload if payload is not None else self.prepare(compiled, backend)[0]
        t = self.backend(backend).target
        try:
            duration = float(qc.estimate_duration(t, unit="s"))
        except Exception:
            d = compiled.target.durations
            duration = compiled.depth * max(d.get("2q", 1e-7), d.get("1q", 3e-8)) + d.get("readout", 1e-6)
        rep = compiled.target.extras.get("default_rep_delay") or 250e-6
        seconds = JOB_OVERHEAD_S + shots * (rep + duration)
        rate = float(os.environ.get("QUMO_IBM_USD_PER_SECOND", DEFAULT_USD_PER_SECOND))
        return CostEstimate(
            self.name, backend, shots, "QPU-seconds", round(seconds, 3),
            "2 s + shots * (rep_delay + circuit_duration)",
            usd=round(seconds * rate, 2),
            breakdown={"overhead_s": JOB_OVERHEAD_S, "rep_delay_s": rep, "circuit_duration_s": duration,
                       "usd_per_second": rate},
            notes=[
                "heuristic from IBM's 'Estimate job run time' guide; billed usage is QPU time",
                "Open Plan: up to 10 free QPU minutes per month (28-day rolling window); Pay-As-You-Go starts at "
                "$96/min billed per second (ibm.com/quantum/pricing, 2026-10-12)",
                "override the USD rate with QUMO_IBM_USD_PER_SECOND",
            ],
        )

    def server_cap_seconds(self, compiled: CompiledCircuit, shots: int, backend: str, payload=None,
                           max_cost: float | None = None) -> int:
        """`max_execution_time` sent with a remote job (QPU seconds)."""
        if max_cost is not None:
            return max(1, int(math.ceil(max_cost)))
        est = self.estimate_cost(compiled, shots, backend, payload=payload)
        return int(max(SERVER_CAP_MIN_S, math.ceil(SERVER_CAP_FACTOR * est.amount)))

    def submit(self, compiled: CompiledCircuit, shots: int, backend: str, seed: int | None = None,
               payload=None, report: dict | None = None, max_cost: float | None = None, confirm: bool = False,
               **_: Any) -> JobHandle:
        check_confirmed(self, backend, confirm)
        local = self.is_local(backend)
        if shots < 1 or shots > compiled.target.max_shots:
            raise HardwareError(E_LIMIT, f"shots must be in 1..{compiled.target.max_shots} for {backend}", HINT_SHOTS)
        if payload is None:
            payload, report = self.prepare(compiled, backend)
        if local:
            active = active_qubits(payload)
            if active > LOCAL_MAX_ACTIVE_QUBITS:
                raise HardwareError(E_LIMIT, f"{active} active qubits exceed the local Aer limit of "
                                    f"{LOCAL_MAX_ACTIVE_QUBITS} for fake backends", HINT_LOCAL_QUBITS)
        _sdk()
        from qiskit_ibm_runtime.executor_sampler import Sampler

        b = self.backend(backend)
        sampler = Sampler(mode=b)
        sampler.options.twirling.enable_gates = False
        sampler.options.twirling.enable_measure = False
        sampler.options.dynamical_decoupling.enable = False
        cap = None
        if local and seed is not None:
            sampler.options.simulator.seed_simulator = int(seed)
        if not local:
            cap = self.server_cap_seconds(compiled, shots, backend, payload=payload, max_cost=max_cost)
            sampler.options.max_execution_time = cap
        for key, value in self.sampler_options.items():
            obj = sampler.options
            *path, last = key.split(".")
            for p in path:
                obj = getattr(obj, p)
            setattr(obj, last, value)
        if not local:
            cap = sampler.options.max_execution_time  # may be overridden through sampler_options
        dry = bool(self.server_dry_run and not local)
        try:
            job = sampler.run([payload], shots=int(shots), dry_run=True) if dry else sampler.run([payload], shots=int(shots))
        except HardwareError:
            raise
        except Exception as exc:
            raise HardwareError(_classify(exc, E_JOB), f"IBM Quantum refused the job ({safe_exc_text(exc)})",
                                "nothing was submitted; check the instance's remaining QPU time, the backend status "
                                "(`backends --provider ibm --online`) and the job limits") from None
        meta = compiled.result_meta()
        meta.update({"finishing": report, "instance": self.instance, "channel": self.channel,
                     "sampler": "qiskit_ibm_runtime.executor_sampler.Sampler",
                     "server_dry_run": dry, "max_execution_time": cap})
        if dry:
            meta["mock_results"] = True
        if local:
            try:
                res = job.result()
            except Exception as exc:
                raise HardwareError(E_JOB, f"local Aer simulation of '{backend}' failed ({safe_exc_text(exc)})",
                                    HINT_LOCAL_QUBITS) from None
            counts = counts_from_primitive_result(res, compiled.num_clbits)
            return JobHandle(new_local_job_id("ibm-local"), self.name, backend, shots, True, DONE, meta=meta,
                             counts=counts)
        return JobHandle(str(job.job_id()), self.name, backend, shots, False, QUEUED, meta=meta)

    def _remote_job(self, job: JobHandle):
        from qiskit_ibm_runtime.exceptions import RuntimeJobNotFound

        svc = self.service()
        try:
            return svc.job(job.job_id)
        except RuntimeJobNotFound:
            raise HardwareError(E_UNKNOWN_JOB, f"IBM job '{job.job_id}' not found for this account") from None
        except HardwareError:
            raise
        except Exception as exc:
            raise HardwareError(_classify(exc, E_JOB), f"cannot look up IBM job '{job.job_id}' ({safe_exc_text(exc)})",
                                _IBM_HINT_NET) from None

    def status(self, job: JobHandle) -> str:
        if job.local:
            return job.status
        rj = self._remote_job(job)
        try:
            st = rj.status()
        except Exception as exc:
            raise HardwareError(_classify(exc, E_JOB), f"cannot read the status of IBM job '{job.job_id}' "
                                f"({safe_exc_text(exc)})", _IBM_HINT_NET) from None
        name = getattr(st, "name", None) or str(st)
        return _STATUS.get(str(name).upper(), UNKNOWN)

    def result(self, job: JobHandle) -> dict[str, int]:
        if job.local:
            return dict(job.counts or {})
        rj = self._remote_job(job)
        st = self.status(job)
        if st != DONE:
            raise HardwareError(E_JOB, f"IBM job '{job.job_id}' is {st}")
        try:
            res = rj.result()
        except Exception as exc:
            raise HardwareError(_classify(exc, E_JOB), f"cannot fetch the result of IBM job '{job.job_id}' "
                                f"({safe_exc_text(exc)})", "retry later; IBM keeps results of finished jobs") from None
        return counts_from_primitive_result(res, int(job.meta["num_clbits"]))


def active_qubits(qc) -> int:
    """Qubits touched by any instruction other than a barrier (what Aer simulates)."""
    used = set()
    for inst in qc.data:
        if inst.operation.name != "barrier":
            used.update(qc.find_bit(q).index for q in inst.qubits)
    return len(used)


def _classify(exc: BaseException, default: str) -> str:
    """Map an IBM SDK exception onto a QM-E95x code."""
    name = type(exc).__name__
    if "NotAuthorized" in name or "Unauthorized" in name or "Forbidden" in name or "AccountNotFound" in name:
        return E_CREDENTIALS
    if "InputValue" in name or "Validation" in name:
        return E_INVALID_PAYLOAD
    return default


def compile_for_ibm(source, provider: IBMProvider, backend: str, level: int = 2, seed: int = 1) -> CompiledCircuit:
    return compile_for_target(source, provider.target(backend), level=level, seed=seed)
