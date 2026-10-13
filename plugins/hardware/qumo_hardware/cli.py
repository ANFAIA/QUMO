"""`python -m qumo_hardware` - run QUMO circuits on quantum hardware.

    python -m qumo_hardware run circuit.qasm --provider ibm --backend ibm_torino --shots 1000 [--yes]
    python -m qumo_hardware estimate circuit.qasm --provider quantinuum --backend H2-1 --shots 100
    python -m qumo_hardware backends [--provider ibm] [--online]
    python -m qumo_hardware target --provider ibm --backend fake_torino
    python -m qumo_hardware status JOB      |  result JOB [--wait]  |  jobs  |  errors

Every command accepts --json (machine output on stdout, errors as
{"error": {...}} like the qumo CLI). Exit codes: 0 ok, 1 error (also: the
job `run` waited for ended in ERROR or CANCELLED), 2 usage, 3 not submitted
because confirmation (--yes) is required, 130 interrupted (Ctrl-C) while
waiting for a submitted job (its id is printed; the job keeps running).
Remote submission is opt-in: without --yes, `run` prints the plan (backend,
shots, cost estimate) and exits with code 3; --dry-run never submits.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
import warnings
from pathlib import Path

from . import jobs as job_store
from . import runner
from .base import CANCELLED, ERROR
from .errors import E_COMPILE, E_CONFIRM, HardwareError, catalog
from .registry import get_provider, provider_names

EXIT_OK, EXIT_ERR, EXIT_USAGE, EXIT_CONFIRM, EXIT_INTERRUPTED = 0, 1, 2, 3, 130


def _read_circuit(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    p = Path(path)
    if not p.exists():
        raise HardwareError(E_COMPILE, f"cannot read file '{path}'", "check the path, or pass '-' to read from stdin")
    return p.read_text()


def _provider_options(args) -> dict:
    return {
        "channel": getattr(args, "channel", None),
        "instance": getattr(args, "instance", None),
        "account_name": getattr(args, "account", None),
        "use_fractional_gates": True if getattr(args, "fractional", False) else None,
        "server_dry_run": True if getattr(args, "ibm_dry_run", False) else None,
        "project": getattr(args, "project", None),
        "seed": getattr(args, "seed", None),
    }


def _add_common(p: argparse.ArgumentParser, circuit: bool = True, backend: bool = True) -> None:
    if circuit:
        p.add_argument("circuit", help="OpenQASM 2/3 file ('-' = stdin)")
    p.add_argument("--provider", "-p", choices=None, default="local",
                   help="provider: " + ", ".join(provider_names()) + " (default: local)")
    if backend:
        p.add_argument("--backend", "-b", default=None, help="backend / device name (default: qumo-sim for local)")
    p.add_argument("--shots", "-s", type=int, default=1000)
    p.add_argument("--level", type=int, default=2, choices=[0, 1, 2, 3], help="QUMO optimisation level")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--json", action="store_true", help="machine-readable JSON on stdout")
    p.add_argument("--fractional", action="store_true", help="IBM: use Heron fractional gates (rzz, rx)")
    p.add_argument("--provider-transpile", action="store_true",
                   help="compare with (and submit) the provider's own compiler output")
    p.add_argument("--channel", help="IBM channel (default: SDK default / QISKIT_IBM_CHANNEL)")
    p.add_argument("--instance", help="IBM instance CRN or name (default: QISKIT_IBM_INSTANCE / saved account)")
    p.add_argument("--account", help="IBM saved-account name")
    p.add_argument("--project", help="Quantinuum Nexus project (default: qumo-hardware)")
    p.add_argument("--show-qasm", action="store_true", help="include the compiled OpenQASM 2 payload")
    p.add_argument("--params", help="bind symbolic parameters: JSON list (declaration order) or {name: value}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m qumo_hardware", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="compile with QUMO and run (remote needs --yes)")
    _add_common(r)
    r.add_argument("--dry-run", action="store_true", help="compile, validate and estimate only")
    r.add_argument("--yes", "-y", action="store_true", help="confirm submission to a remote/billable backend")
    r.add_argument("--max-cost", type=float,
                   help="refuse if the estimate exceeds this (Quantinuum: HQC/eHQC, IBM: QPU seconds); "
                        "also sent as the server-side cap (max_cost / max_execution_time)")
    r.add_argument("--no-wait", action="store_true", help="return right after submission")
    r.add_argument("--timeout", type=float, default=600.0, help="seconds to wait for a remote result")
    r.add_argument("--ibm-dry-run", action="store_true",
                   help="IBM: server-side dry run (validates + estimates usage; counts are mock data and are "
                        "flagged mock_results; needs --yes)")

    e = sub.add_parser("estimate", help="compile, validate and estimate cost (never submits)")
    _add_common(e)

    b = sub.add_parser("backends", help="list backends")
    b.add_argument("--provider", "-p", default=None, help="one provider (default: all built-in)")
    b.add_argument("--online", action="store_true", help="also query the provider (needs credentials)")
    b.add_argument("--json", action="store_true")
    b.add_argument("--channel")
    b.add_argument("--instance")
    b.add_argument("--account")

    t = sub.add_parser("target", help="show the neutral TargetSpec of a backend")
    _add_common(t, circuit=False)
    t.add_argument("--full", action="store_true", help="include edges and per-qubit data")

    s = sub.add_parser("status", help="status of a submitted job")
    s.add_argument("job_id")
    s.add_argument("--json", action="store_true")

    res = sub.add_parser("result", help="counts of a finished job (user's bit order)")
    res.add_argument("job_id")
    res.add_argument("--wait", action="store_true")
    res.add_argument("--timeout", type=float, default=600.0)
    res.add_argument("--json", action="store_true")

    j = sub.add_parser("jobs", help="jobs in the local job store")
    j.add_argument("--limit", type=int, default=20)
    j.add_argument("--json", action="store_true")

    er = sub.add_parser("errors", help="QM-E95x error catalogue")
    er.add_argument("--json", action="store_true")
    return ap


def _default_backend(args) -> str:
    if args.backend:
        return args.backend
    if args.provider == "local":
        return "qumo-sim"
    raise HardwareError("QM-E953", f"--backend is required for provider '{args.provider}'",
                        f"list them with `python -m qumo_hardware backends --provider {args.provider}`")


def _emit(obj, as_json: bool, human) -> None:
    if as_json:
        print(json.dumps(obj, indent=1, sort_keys=False, default=str))
    else:
        human(obj)


def _human_plan(d: dict) -> None:
    c = d["compiled"]
    where = "local" if d["local"] else "REMOTE"
    print(f"provider/backend : {d['provider']} / {d['backend']} ({where})")
    print(f"circuit          : {c['num_logical']} logical -> {c['num_physical']} physical qubits "
          f"({c['active_qubits']} active), {c['num_clbits']} clbits"
          + (" (measure-all added)" if c["auto_measured"] else ""))
    print(f"compiled by QUMO : {c['two_qubit_gates']} two-qubit gates, depth {c['depth']}, "
          f"level {c['level']}, {c['compile_ms']} ms")
    print(f"layout           : initial {c['initial_layout']} -> final {c['final_layout']}")
    checks = ", ".join(d["validation"].get("checks", [])) or "-"
    print(f"validation       : OK ({checks})")
    if d.get("finishing", {}).get("passes"):
        print(f"finishing        : {d['finishing']['passes']}")
    if "comparison" in d:
        print(f"comparison       : QUMO {d['comparison']['qumo']}")
        print(f"                   provider {d['comparison']['provider']}")
    print(f"shots            : {d['shots']}")
    print(f"cost estimate    : {d['cost_summary']}  [{d['cost']['formula']}]")
    if d["cost"]["unit"] != "free":
        for n in d["cost"].get("notes", [])[:2]:
            print(f"                   - {n}")
    if c.get("region"):
        print(f"region           : {len(c['region'])} best-calibrated connected qubits {c['region']}")
    for n in d.get("notes", []):
        print(f"note             : {n}")
    if "qasm" in d:
        print("payload (OpenQASM 2):")
        print(d["qasm"])


def _human_counts(counts: dict, limit: int = 16) -> None:
    total = sum(counts.values()) or 1
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1])[:limit]:
        print(f"  {k}  {v:6d}  {v / total:6.3f}")
    if len(counts) > limit:
        print(f"  ... {len(counts) - limit} more outcomes")


def cmd_run(args, estimate_only: bool = False) -> int:
    backend = _default_backend(args)
    prov = get_provider(args.provider, **_provider_options(args))
    src = _read_circuit(args.circuit)
    params = None
    if args.params:
        try:
            params = json.loads(args.params)
        except ValueError:
            raise HardwareError(E_COMPILE, "--params must be JSON (a list or an object)") from None
    if estimate_only:
        p = runner.plan(src, prov, backend, args.shots, args.level, args.seed, args.provider_transpile, params)
        d = p.to_dict(include_qasm=args.show_qasm)
        _emit(d, args.json, _human_plan)
        return EXIT_OK
    wait = not args.no_wait

    def submitted(job):
        if not job.local and wait:
            print(f"submitted job {job.job_id} ({job.provider} / {job.backend}); waiting up to {args.timeout:g} s "
                  "(Ctrl-C stops waiting, the job keeps running)", file=sys.stderr, flush=True)

    try:
        out = runner.run(src, prov, backend, args.shots, args.level, args.seed, dry_run=args.dry_run,
                         confirm=args.yes, max_cost=args.max_cost, wait=wait, timeout=args.timeout,
                         provider_transpile=args.provider_transpile, include_qasm=args.show_qasm, params=params,
                         on_submitted=submitted)
    except KeyboardInterrupt as exc:
        job_id = getattr(exc, "qumo_job_id", None)
        if job_id:
            print(f"interrupted: job {job_id} was submitted and keeps running; fetch it later with "
                  f"`python -m qumo_hardware result {job_id} --wait`", file=sys.stderr)
        raise

    def human(d):
        _human_plan(d)
        if d.get("dry_run"):
            print("dry run: nothing submitted")
        elif d.get("counts") is not None:
            print(f"job {d['job_id']}: {d['status']}; counts ({d['bit_order']}):")
            _human_counts(d["counts"])
        else:
            print(f"submitted job {d['job_id']} ({d['status']}); "
                  f"fetch with: python -m qumo_hardware result {d['job_id']} --wait")
        if d.get("warning"):
            print(f"warning: {d['warning']}", file=sys.stderr)

    _emit(out, args.json, human)
    if out.get("submitted") and out.get("status") in (ERROR, CANCELLED):
        return EXIT_ERR
    return EXIT_OK


def cmd_backends(args) -> int:
    names = [args.provider] if args.provider else ["local", "ibm", "quantinuum"]
    rows, warnings = [], []
    for n in names:
        try:
            rows += runner.list_backends(n, online=args.online, channel=args.channel, instance=args.instance,
                                         account_name=args.account)
        except HardwareError as exc:
            if args.provider:
                raise
            warnings.append(exc.to_dict())

    def human(d):
        for r in d["backends"]:
            nq = r["num_qubits"] if r["num_qubits"] is not None else "-"
            bill = "billable" if r["billable"] else "free"
            print(f"{r['provider']:<11} {r['name']:<32} {r['kind']:<15} {nq!s:>4}q  {bill:<8} {r['notes']}")
        for w in d["warnings"]:
            print(f"warning: {w['code']}: {w['message']}", file=sys.stderr)

    _emit({"backends": rows, "warnings": warnings}, args.json, human)
    return EXIT_OK


def cmd_target(args) -> int:
    backend = _default_backend(args)
    spec = get_provider(args.provider, **_provider_options(args)).target(backend)
    d = spec.to_dict(compact=not args.full)
    _emit(d, args.json, lambda x: print(json.dumps(x, indent=1, default=str)))
    return EXIT_OK


def cmd_status(args) -> int:
    d = runner.job_status(args.job_id)
    _emit(d, args.json, lambda x: print(f"{x['job_id']}: {x['status']} ({x['provider']} / {x['backend']}, {x['shots']} shots)"))
    return EXIT_OK


def cmd_result(args) -> int:
    d = runner.job_result(args.job_id, wait=args.wait, timeout=args.timeout)

    def human(x):
        print(f"{x['job_id']}: {x['status']} ({x['provider']} / {x['backend']}); counts ({x['bit_order']}):")
        _human_counts(x["counts"])
        if x.get("warning"):
            print(f"warning: {x['warning']}", file=sys.stderr)

    _emit(d, args.json, human)
    return EXIT_OK


def cmd_jobs(args) -> int:
    rows = [{"job_id": j.job_id, "provider": j.provider, "backend": j.backend, "shots": j.shots,
             "status": j.status, "local": j.local, "submitted_at": j.submitted_at}
            for j in job_store.list_jobs(args.limit)]
    _emit({"jobs": rows, "store": str(job_store.store_dir())}, args.json,
          lambda d: [print(f"{r['job_id']:<40} {r['provider']:<11} {r['backend']:<24} {r['status']}") for r in d["jobs"]])
    return EXIT_OK


def cmd_errors(args) -> int:
    _emit({"errors": catalog()}, args.json,
          lambda d: [print(f"{e['code']}  {e['summary']}\n         Hint: {e['hint']}") for e in d["errors"]])
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    # qiskit-ibm-runtime warns about fake_nighthawk's properties whenever its snapshots are
    # loaded, even for unrelated backends; the plugin reports it as a note on that backend only.
    warnings.filterwarnings("ignore", message=r"Properties of fake_\w+ are not intended", category=UserWarning)
    ap = build_parser()
    try:
        args = ap.parse_args(argv)
    except SystemExit as exc:  # argparse usage error / --help
        return EXIT_USAGE if exc.code not in (0, None) else EXIT_OK
    as_json = getattr(args, "json", False)
    handlers = {"run": cmd_run, "estimate": lambda a: cmd_run(a, estimate_only=True), "backends": cmd_backends,
                "target": cmd_target, "status": cmd_status, "result": cmd_result, "jobs": cmd_jobs,
                "errors": cmd_errors}
    try:
        return handlers[args.cmd](args)
    except HardwareError as exc:
        if as_json:
            print(json.dumps({"error": exc.to_dict()}, indent=1, default=str))
        else:
            if exc.code == E_CONFIRM and "plan" in exc.details:
                _human_plan(exc.details["plan"])
            print(f"error: {exc}", file=sys.stderr)
        return EXIT_CONFIRM if exc.code == E_CONFIRM else EXIT_ERR
    except KeyboardInterrupt:
        return EXIT_INTERRUPTED
    except Exception as exc:  # no stack traces for users; QUMO_HW_DEBUG=1 shows them
        if os.environ.get("QUMO_HW_DEBUG"):
            traceback.print_exc()
        err = {"code": "QM-E999", "severity": "error", "message": f"unexpected {type(exc).__name__}: {exc}",
               "hint": "re-run with QUMO_HW_DEBUG=1 and report the traceback", "docs": "docs/HARDWARE_PLUGINS.md"}
        if as_json:
            print(json.dumps({"error": err}, indent=1))
        else:
            print(f"error: {err['code']}: {err['message']}. Hint: {err['hint']}", file=sys.stderr)
        return EXIT_ERR


if __name__ == "__main__":
    sys.exit(main())
