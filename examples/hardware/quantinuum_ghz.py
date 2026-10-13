"""GHZ state on Quantinuum, compiled by QUMO into the native Rz / PhasedX / ZZPhase gate set.

    python examples/hardware/quantinuum_ghz.py                         # dry run for H2-1: payload + HQC estimate (no account)
    python examples/hardware/quantinuum_ghz.py --backend H2-1LE --run  # offline H2 emulator (pytket-quantinuum[pecos])
    python examples/hardware/quantinuum_ghz.py --backend selene --run  # offline Helios-style emulator (guppylang)
    python examples/hardware/quantinuum_ghz.py --backend H2-1E --submit   # Nexus emulator (eHQC), asks first
    python examples/hardware/quantinuum_ghz.py --backend Helios-1 --submit --max-cost 20

Remote devices need a Nexus login once: python -c "import qnexus as qnx; qnx.login()".
"""

from __future__ import annotations

import argparse
import sys

from _common import HardwareError, confirm, fail, print_counts, print_plan, runner, submit_and_report


def ghz(n: int) -> str:
    body = "".join(f"cx q[{i}], q[{i + 1}];\n" for i in range(n - 1))
    return f'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[{n}] q;\nbit[{n}] c;\nh q[0];\n{body}c = measure q;\n'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="H2-1")
    ap.add_argument("-n", type=int, default=6, help="number of qubits")
    ap.add_argument("--shots", type=int, default=100)
    ap.add_argument("--run", action="store_true", help="run on a local emulator (H2-1LE / selene)")
    ap.add_argument("--submit", action="store_true", help="submit to Nexus (asks first)")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--max-cost", type=float, default=None, help="refuse above this many HQC")
    ap.add_argument("--show-payload", action="store_true", help="print the native pytket commands")
    args = ap.parse_args()

    from qumo_hardware.quantinuum import QuantinuumProvider

    provider = QuantinuumProvider()
    src = ghz(args.n)
    try:
        p = runner.plan(src, provider, args.backend, shots=args.shots)
        plan = p.to_dict()
        if args.show_payload:
            for cmd in p.payload.get_commands():
                print("  ", cmd)
        if provider.is_local(args.backend) and args.run:
            out = runner.run(src, provider, args.backend, shots=args.shots, seed=1)
            print_plan(out)
            print("counts:")
            print_counts(out["counts"])
            return 0
        if not args.submit:
            print_plan(plan)
            b = plan["cost"]["breakdown"]
            print(f"HQC inputs   : N1q={b.get('n1q')} N2q={b.get('n2q')} Nm={b.get('n_spam')} (formula {plan['cost']['formula']})")
            print("dry run: nothing submitted (--run for local emulators, --submit for Nexus)")
            return 0
        if not confirm(plan, args.yes):
            print("not submitted")
            return 3
        return submit_and_report(source=src, provider=provider, backend=args.backend, shots=args.shots,
                                 max_cost=args.max_cost, timeout=3600)
    except HardwareError as exc:
        return fail(exc)


if __name__ == "__main__":
    sys.exit(main())
