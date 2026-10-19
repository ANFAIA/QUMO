"""Shared helpers for the four heterogeneous pipelines in this folder.

    sqd/sqd_hubbard.py                 QPU sampling -> CPU configuration recovery -> HPC subspace diagonalisation
    cutting/cut_two_modalities.py      CPU cut finder -> ion + superconducting QPUs -> GPU/CPU reconstruction
    realtime_qec/stream_decode.py      QPU syndrome stream -> CPU decoder pool under a real-time budget
    qaoa_escalation/ws_qaoa_escalation.py  CPU warm start -> GPU adjoint training -> SC QPU -> ion QPU -> CPU

Nothing here decides *what* to run. It provides: loading qumo_py (CPU or GPU
build), GPU placement rules (the same as stages.Runtime.place), a stage log
that records which device ran each stage and whether that device was
emulated, a multi-QPU gate with the same safety rules as the main example
(local emulators run; remote backends only dry-run unless --allow-remote
--submit --yes --max-cost; qumo_hardware.runner re-checks QM-E956/QM-E957),
HEFT schedules from QUMO's orchestrator (qumo_py.schedule), and reports.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(HERE), str(HERE.parent)):  # examples/heterogeneous, examples (_checks.py)
    if _p not in sys.path:
        sys.path.insert(0, _p)


# --------------------------------------------------------------------------- qumo_py
def load_qumo(prefer: str = "auto"):
    """Import qumo_py: PYTHONPATH / QUMO_PY_PATH first, else build/gpu (prefer gpu/auto), else build."""
    try:
        import qumo_py  # noqa: F401
        return _sanitize_env(sys.modules["qumo_py"])
    except ImportError:
        pass
    cands = []
    env = os.environ.get("QUMO_PY_PATH")
    if env:
        cands.append(Path(env).parent if env.endswith(".so") else Path(env))
    if prefer in ("auto", "gpu"):
        cands.append(REPO / "build" / "gpu")
    if prefer in ("auto", "cpu"):
        cands.append(REPO / "build")
    for d in cands:
        if (d / "qumo_py.so").is_file():
            sys.path.insert(0, str(d))
            import qumo_py  # noqa: F401
            return _sanitize_env(sys.modules["qumo_py"])
    raise SystemExit("qumo_py.so not found: run `make py` (CPU) or `make py-gpu` (GPU) first")


_ENV_AT_START = dict(os.environ)


def _sanitize_env(q):
    """Undo process-level environment changes made while the Mojo runtime starts (it sets
    PYTHONEXECUTABLE to another interpreter and prefixes PYTHONPATH in the C environment, which
    spawned worker processes would inherit and warn about). os.environ is unaffected."""
    for k in ("PYTHONEXECUTABLE", "PYTHONPATH"):
        if k in _ENV_AT_START:
            os.putenv(k, _ENV_AT_START[k])
        else:
            os.unsetenv(k)
    return q


def require_bindings(q, names: list[str]) -> None:
    missing = [n for n in names if not hasattr(q, n)]
    if missing:
        raise SystemExit(f"this qumo_py.so predates {missing}: rebuild it with `make py` (or `make py-gpu`)")


def ensure_hardware_plugin() -> None:
    try:
        import qumo_hardware  # noqa: F401
    except ImportError:
        sys.path.insert(0, str(REPO / "plugins" / "hardware"))


def gpu_status(q) -> dict:
    g = dict(q.gpu_info())
    g["usable"] = bool(g.get("compiled")) and bool(g.get("available"))
    return g


def place_gpu(gpu: dict, num_qubits: int, want: str = "auto") -> tuple[str, str]:
    """('gpu' | 'cpu', note). want: 'auto' (threshold rule), 'gpu' (force when usable), 'cpu'."""
    if want == "cpu":
        return "cpu", "--gpu-device cpu"
    if not gpu.get("compiled"):
        return "cpu", "GPU support not compiled into this qumo_py (make py-gpu; PYTHONPATH=build/gpu)"
    if not gpu.get("available"):
        return "cpu", f"no usable GPU ({gpu.get('error') or 'gpu_info().available is false'})"
    minq = int(gpu.get("min_qubits_auto") or 20)
    if want == "gpu" or num_qubits >= minq:
        note = "" if num_qubits >= minq else f"forced: {num_qubits} < {minq} qubits (the CPU would be faster here)"
        return "gpu", note
    return "cpu", f"below the GPU threshold ({num_qubits} < {minq} qubits): the CPU is faster here"


# --------------------------------------------------------------------------- stages
@dataclass
class StageRecord:
    stage: str  # short name, e.g. "qpu-sampling"
    device: str  # "cpu" | "gpu" | "qpu:<target>" | "fpga (emulated)" ...
    role: str  # why this device: what it does that the others cannot
    seconds: float = 0.0
    method: str = ""
    emulated: bool = False  # the device is emulated on this machine (QPU emulator, FPGA budget model)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["seconds"] = round(self.seconds, 4)
        return d


class StageLog:
    def __init__(self, say: Callable[[str], None] = print):
        self.records: list[StageRecord] = []
        self.say = say

    def add(self, rec: StageRecord) -> StageRecord:
        self.records.append(rec)
        tag = " [emulated]" if rec.emulated else ""
        self.say(f"[{rec.stage}] {rec.device}{tag}: {rec.method or '-'} {rec.seconds:.2f} s"
                 + (f" - {'; '.join(rec.notes)}" if rec.notes else ""))
        return rec

    def table(self) -> str:
        rows = ["| Stage | Device | Why this device | Method | Seconds | Notes |", "|---|---|---|---|---|---|"]
        for r in self.records:
            dev = r.device + (" (emulated)" if r.emulated else "")
            rows.append(f"| {r.stage} | {dev} | {r.role} | {r.method} | {r.seconds:.3f} | {'; '.join(r.notes)[:200]} |")
        return "\n".join(rows)


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0
        return False


# --------------------------------------------------------------------------- QPUs
def is_remote(target: str) -> bool:
    prov, _, be = target.partition(":")
    if prov == "local":
        return False
    if prov == "ibm":
        return not be.startswith("fake_")
    if prov == "quantinuum":
        return not be.endswith("LE")
    return True


def modality_of(target: str) -> str:
    s = target.lower()
    if "trapped_ion" in s or "ion_" in s or s.startswith("quantinuum"):
        return "trapped_ion"
    if "neutral_atom" in s:
        return "neutral_atom"
    return "superconducting"


class QpuGate:
    """Runs QPU jobs on several targets with one safety policy (human flags only)."""

    def __init__(self, allow_remote: bool = False, submit: bool = False, yes: bool = False,
                 max_cost: float | None = None, runner=None, say: Callable[[str], None] = print):
        ensure_hardware_plugin()
        if runner is None:
            from qumo_hardware import runner as _runner
            runner = _runner
        self.runner = runner
        self.allow_remote = allow_remote
        self.submit = submit
        self.yes = yes
        self.max_cost = max_cost
        self.say = say
        self.jobs: list[dict] = []

    def run(self, qasm: str, target: str, shots: int, seed: int, label: str, level: int = 2) -> dict:
        """One job -> record with counts (None on dry runs / refusals), cost and compile summary."""
        from qumo_hardware.errors import HardwareError

        prov, _, backend = target.partition(":")
        remote = is_remote(target)
        rec: dict[str, Any] = {"label": label, "target": target, "shots": int(shots), "local": not remote,
                               "submitted": False, "dry_run": False, "counts": None,
                               "emulated": not remote, "modality": modality_of(target)}
        t0 = time.perf_counter()
        try:
            if not remote:
                out = self.runner.run(qasm, prov, backend, shots=int(shots), seed=int(seed), level=level)
            elif not self.allow_remote:
                raise HardwareError("QM-E957", f"remote backend '{target}' is not allowed without --allow-remote")
            elif not self.submit:
                out = self.runner.run(qasm, prov, backend, shots=int(shots), seed=int(seed), level=level,
                                      dry_run=True)
                rec["dry_run"] = True
            elif self.max_cost is None:
                raise HardwareError("QM-E956", "remote submission needs an explicit --max-cost",
                                    "pass --max-cost in the provider's cost unit after reviewing the dry run")
            else:
                out = self.runner.run(qasm, prov, backend, shots=int(shots), seed=int(seed), level=level,
                                      confirm=bool(self.yes), max_cost=self.max_cost)
        except HardwareError as exc:
            rec.update({"error": exc.to_dict(), "seconds": round(time.perf_counter() - t0, 3)})
            plan = exc.details.get("plan") if exc.details else None
            if plan:
                rec["cost_summary"] = plan.get("cost_summary")
            self.jobs.append(rec)
            self.say(f"[qpu] {label} on {target}: not run ({exc.code}: {str(exc)[:160]})")
            return rec
        comp = out.get("compiled") or {}
        rec.update({
            "seconds": round(time.perf_counter() - t0, 3),
            "counts": out.get("counts"),
            "submitted": bool(out.get("submitted")),
            "job_id": out.get("job_id"),
            "cost_summary": out.get("cost_summary"),
            "compiled": {k: comp.get(k) for k in ("num_qubits", "two_qubit_gates", "depth", "ops") if k in comp},
            "noise_model": (out.get("finishing") or {}).get("noise_model"),
            "noisy": bool((out.get("finishing") or {}).get("noisy")) or prov == "ibm" or backend.endswith("LE"),
        })
        self.jobs.append(rec)
        return rec

    def summary(self) -> list[dict]:
        return [{k: v for k, v in j.items() if k != "counts"} for j in self.jobs]

    def nothing_remote_submitted(self) -> bool:
        return all(j["local"] or not j["submitted"] for j in self.jobs)


def counts_to_ints(counts: dict) -> dict[int, int]:
    """Qiskit-style keys (leftmost = highest clbit) -> {basis index: count}."""
    out: dict[int, int] = {}
    for k, v in counts.items():
        x = int(str(k).replace(" ", ""), 2)
        out[x] = out.get(x, 0) + int(v)
    return out


def add_safety_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--allow-remote", action="store_true", help="let a remote (billable) QPU target be used")
    ap.add_argument("--submit", action="store_true", help="submit to a remote QPU (else dry run with cost)")
    ap.add_argument("--yes", action="store_true", help="confirm remote submission after reviewing the dry run")
    ap.add_argument("--max-cost", type=float, default=None, help="cost cap in the provider's unit (required)")


def check_targets(targets: list[str], allow_remote: bool) -> None:
    for t in targets:
        if is_remote(t) and not allow_remote:
            raise SystemExit(f"{t} is a remote (billable) backend: add --allow-remote to plan for it "
                             "(a dry run shows the cost; --submit --yes --max-cost submits)")


# --------------------------------------------------------------------------- orchestrator
def heft(q, jobs: list[dict], devices: list[dict], say: Callable[[str], None] = print) -> dict:
    """QUMO's HEFT plan (qumo.exec.orchestrator via qumo_py.schedule); prints its notes."""
    sch = dict(q.schedule(jobs, devices))
    for line in sch["notes"]:
        say(f"[heft] {line}")
    return sch


def qpu_device(name: str, target: str, max_qubits: int) -> dict:
    """Orchestrator Device for a QPU target (queue / shot-time model per modality)."""
    if modality_of(target) == "trapped_ion":  # slow shots, all-to-all, high fidelity
        return {"name": name, "kind": "qpu", "max_qubits": max_qubits, "queue_s": 2.0, "shot_time_s": 2e-2,
                "cost_per_task": 0.0, "cost_per_shot": 0.0}
    return {"name": name, "kind": "qpu", "max_qubits": max_qubits, "queue_s": 1.0, "shot_time_s": 3e-4,
            "cost_per_task": 0.0, "cost_per_shot": 0.0}


# --------------------------------------------------------------------------- reports
def environment(q) -> dict:
    return {"qumo_version": q.version(), "qumo_py": getattr(q, "__file__", ""), "python": sys.version.split()[0],
            "platform": platform.platform(), "machine": platform.machine(), "cpus": os.cpu_count()}


def write_report(out_dir: Path, report: dict, stages: StageLog, title: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    report["stages"] = [r.to_dict() for r in stages.records]  # also returned to the caller
    (out_dir / "report.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")
    lines = [f"# {title}", "", stages.table(), "", "## Results", "", "```json",
             json.dumps(report.get("results", {}), indent=1, default=str)[:8000], "```", "", "## Checks", ""]
    lines += [f"- `{c['label']}`: {c['status']} {c['detail']}" for c in report.get("checks", [])]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    return out_dir / "report.json"
