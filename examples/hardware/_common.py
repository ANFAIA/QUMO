"""Shared helpers for the hardware examples (path setup + interactive confirmation)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "plugins" / "hardware"))

from qumo_hardware import runner  # noqa: E402,F401  (re-exported for the examples)
from qumo_hardware.errors import HardwareError  # noqa: E402


def print_plan(plan: dict) -> None:
    c = plan["compiled"]
    print(f"backend      : {plan['provider']} / {plan['backend']} ({'local' if plan['local'] else 'REMOTE'})")
    print(f"compiled     : {c['num_logical']} logical qubits, {c['two_qubit_gates']} two-qubit gates, depth {c['depth']}")
    print(f"layout       : {c['initial_layout']} -> {c['final_layout']}")
    print(f"shots        : {plan['shots']}")
    print(f"cost         : {plan['cost_summary']}")
    for note in plan.get("notes", []):
        print(f"note         : {note}")


def print_counts(counts: dict, limit: int = 8) -> None:
    total = sum(counts.values())
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1])[:limit]:
        print(f"  {k}  {v:6d}  {v / total:.3f}")


def confirm(plan: dict, assume_yes: bool) -> bool:
    """Ask a human before anything billable is submitted."""
    print_plan(plan)
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        print("not a terminal: refusing to submit without --yes")
        return False
    answer = input("Submit this job to real hardware / a billable service? Type 'yes' to continue: ")
    return answer.strip().lower() == "yes"


def fail(exc: HardwareError) -> int:
    print(f"error: {exc}", file=sys.stderr)
    return 1


def submit_and_report(**run_kwargs) -> int:
    """runner.run(confirm=True, wait=True) for a remote backend, with honest reporting:
    exit 1 if the job ended in ERROR/CANCELLED, mock results flagged, and the job id
    printed if the wait is interrupted (the job keeps running at the provider)."""
    try:
        out = runner.run(confirm=True, wait=True, on_submitted=lambda job: print(f"submitted job {job.job_id}"),
                         **run_kwargs)
    except KeyboardInterrupt as exc:
        job_id = getattr(exc, "qumo_job_id", None)
        if job_id:
            print(f"\ninterrupted: job {job_id} keeps running; fetch it later with "
                  f"`python -m qumo_hardware result {job_id} --wait`")
        return 130
    print(f"job {out['job_id']}: {out['status']}")
    if out.get("warning"):
        print(f"warning: {out['warning']}")
    if out.get("counts"):
        print_counts(out["counts"])
    return 1 if out["status"] in ("ERROR", "CANCELLED") else 0


def dump(obj) -> None:
    print(json.dumps(obj, indent=1, default=str))
