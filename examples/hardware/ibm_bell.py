"""Bell pair on IBM Quantum, compiled by QUMO.

    python examples/hardware/ibm_bell.py                          # local: fake_torino + Aer noise (no account)
    python examples/hardware/ibm_bell.py --backend ibm_fez        # dry run on a real QPU: plan + QPU-time estimate
    python examples/hardware/ibm_bell.py --backend ibm_fez --submit   # asks you to type 'yes' before submitting

Real backends need IBM Quantum credentials (QISKIT_IBM_TOKEN + QISKIT_IBM_INSTANCE,
or QiskitRuntimeService.save_account); see docs/HARDWARE_PLUGINS.md.
"""

from __future__ import annotations

import argparse
import sys

from _common import HardwareError, confirm, fail, print_counts, print_plan, runner, submit_and_report

BELL = """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c = measure q;
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="fake_torino")
    ap.add_argument("--shots", type=int, default=1000)
    ap.add_argument("--submit", action="store_true", help="really submit to a remote backend (asks first)")
    ap.add_argument("--yes", action="store_true", help="skip the interactive question (use with care)")
    ap.add_argument("--fractional", action="store_true", help="use Heron fractional gates (rzz, rx)")
    ap.add_argument("--max-cost", type=float, default=None,
                    help="refuse above this many QPU seconds; also sent to IBM as max_execution_time")
    args = ap.parse_args()

    from qumo_hardware.ibm import IBMProvider

    provider = IBMProvider(use_fractional_gates=args.fractional)
    try:
        if provider.is_local(args.backend):
            out = runner.run(BELL, provider, args.backend, shots=args.shots, seed=7)
            print_plan(out)
            print("counts (leftmost = c[1]):")
            print_counts(out["counts"])
            return 0
        plan = runner.plan(BELL, provider, args.backend, shots=args.shots).to_dict()
        if not args.submit:
            print_plan(plan)
            print("dry run: nothing submitted (add --submit to run on the QPU)")
            return 0
        if not confirm(plan, args.yes):
            print("not submitted")
            return 3
        return submit_and_report(source=BELL, provider=provider, backend=args.backend, shots=args.shots,
                                 max_cost=args.max_cost, timeout=3600)
    except HardwareError as exc:
        return fail(exc)


if __name__ == "__main__":
    sys.exit(main())
