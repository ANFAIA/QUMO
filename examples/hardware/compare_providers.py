"""Compile one circuit for several real devices with QUMO and compare them side by side.

    python examples/hardware/compare_providers.py                       # GHZ-6, plans only (no account, no cost)
    python examples/hardware/compare_providers.py --circuit examples/ghz.qasm
    python examples/hardware/compare_providers.py --qv 5                # quantum-volume style circuit (QUMO generator)
    python examples/hardware/compare_providers.py --provider-compilers   # also Qiskit's / TKET's own compilers
    python examples/hardware/compare_providers.py --run-local           # + noisy IBM snapshot and Quantinuum local emulator

IBM rows use the public calibration snapshots shipped with qiskit-ibm-runtime
(fake_torino ~ ibm_torino, ...); their cost column is what the same job would
cost on the real device (QPU-time heuristic). Quantinuum rows use the HQC
formula. "ESP" is the estimated success probability from median error rates.
Nothing is ever submitted by this script.

--circuit takes any OpenQASM 2/3 file the hardware plugins accept: bound
parameters, no simulation-only noise pragmas, and classical conditions only on
whole registers (`if(c==k)`). Circuits with single-bit conditions, such as
examples/teleport.qasm, are refused with QM-E954 (see the limitations in
docs/HARDWARE_PLUGINS.md); a refused device is reported in its row.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _common import REPO, HardwareError, fail, runner

IBM_DEVICES = [("ibm_torino", "fake_torino"), ("ibm_kingston", "fake_kingston"), ("ibm_sherbrooke", "fake_sherbrooke")]
QTM_DEVICES = ["H2-1", "Helios-1"]


def ghz(n: int) -> str:
    body = "".join(f"cx q[{i}],q[{i + 1}];\n" for i in range(n - 1))
    meas = "".join(f"measure q[{i}] -> c[{i}];\n" for i in range(n))
    return f'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[{n}];\ncreg c[{n}];\nh q[0];\n{body}{meas}'


def source(args) -> tuple[str, str]:
    if args.circuit:
        return Path(args.circuit).read_text(), Path(args.circuit).name
    if args.qv:
        from qumo_hardware.compile import load_qumo_py

        q = load_qumo_py()
        return q.quantum_volume(args.qv, args.qv, 11).to_qasm2(), f"qv{args.qv}"
    return ghz(args.n), f"ghz{args.n}"


def ibm_snapshot_provider():
    """IBMProvider whose real device names resolve to the bundled snapshots (offline)."""
    from qumo_hardware.ibm import IBMProvider, fake_provider

    prov = IBMProvider()
    fp = fake_provider()
    for real, fake in IBM_DEVICES:
        prov._backends[real] = fp.backend(fake)
    return prov


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--circuit", help="OpenQASM 2/3 file")
    ap.add_argument("-n", type=int, default=6, help="GHZ size when no circuit is given")
    ap.add_argument("--qv", type=int, default=0, help="use a quantum-volume circuit of this width")
    ap.add_argument("--shots", type=int, default=1000)
    ap.add_argument("--provider-compilers", action="store_true", help="compare with Qiskit / TKET compilers")
    ap.add_argument("--run-local", action="store_true", help="also run on local noisy/ideal emulators")
    args = ap.parse_args()

    src, label = source(args)
    rows = []
    try:
        from qumo_hardware.quantinuum import QuantinuumProvider

        ibm = ibm_snapshot_provider()
        qtm = QuantinuumProvider()
        targets = [(ibm, real, f"{real} (snapshot)") for real, _ in IBM_DEVICES] + [(qtm, d, d) for d in QTM_DEVICES]
        for prov, backend, name in targets:
            try:
                p = runner.plan(src, prov, backend, shots=args.shots, provider_transpile=args.provider_compilers)
            except HardwareError as exc:
                if exc.code == "QM-E954":  # the circuit itself cannot be compiled: same for every device
                    raise
                rows.append({"target": name, "refused": f"{exc.code}: {exc.message}"})
                continue
            c = p.compiled
            row = {
                "target": name,
                "basis": "/".join(c.target.basis),
                "2q": c.two_qubit_gates,
                "depth": c.depth,
                "esp": c.estimated_success(),
                "cost": p.cost.summary(),
            }
            if p.comparison:
                row["provider_2q"] = p.comparison["provider"]["two_qubit_gates"]
            rows.append(row)
    except HardwareError as exc:
        return fail(exc)

    print(f"circuit: {label}, shots: {args.shots}  (compiled by QUMO; nothing submitted)\n")
    hdr = f"{'target':<28} {'native basis':<18} {'2q':>4} {'depth':>6} {'ESP':>6}  cost estimate"
    if args.provider_compilers:
        hdr = hdr.replace("  cost", " vendor-2q  cost")
    print(hdr)
    for r in rows:
        if "refused" in r:
            print(f"{r['target']:<28} refused: {r['refused']}")
            continue
        esp = f"{r['esp']:.3f}" if r["esp"] is not None else "-"
        extra = f" {r['provider_2q']:>9}" if "provider_2q" in r else ""
        print(f"{r['target']:<28} {r['basis']:<18} {r['2q']:>4} {r['depth']:>6} {esp:>6}{extra}  {r['cost']}")

    if args.run_local:
        print("\nlocal runs:")
        for prov_name, backend in (("ibm", "fake_torino"), ("quantinuum", "H2-1LE"), ("local", "qumo-sim")):
            try:
                out = runner.run(src, prov_name, backend, shots=min(args.shots, 1000), seed=3)
            except HardwareError as exc:
                print(f"  {backend:<12} skipped: {exc.code} {exc.message}")
                continue
            counts = out["counts"]
            top = sorted(counts.items(), key=lambda kv: -kv[1])[:3]
            print(f"  {backend:<12} top outcomes {top}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(REPO))
    sys.exit(main())
