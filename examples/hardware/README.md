# Hardware examples

Run QUMO-compiled circuits on IBM Quantum and Quantinuum hardware with the
`qumo_hardware` plugins ([docs/HARDWARE_PLUGINS.md](../../docs/HARDWARE_PLUGINS.md)).
Every script is **dry-run by default**: without flags it either runs on a local
emulator or only prints the compiled plan and cost estimate. Submitting to a
remote device needs `--submit` and an interactive `yes` (or `--yes`).

Prerequisites: `make py` (builds `build/qumo_py.so`) and the Python packages in
`bench/.venv` (qiskit-ibm-runtime for IBM; qnexus + pytket-quantinuum for
Quantinuum). Run from the repository root:

| Script | Default (no account) | Real hardware |
|---|---|---|
| `ibm_bell.py` | Bell pair on `fake_torino` (IBM snapshot + Aer noise) | `--backend ibm_fez` shows plan + QPU-time estimate; `--submit [--max-cost SECONDS]` runs it (the cap is also sent to IBM as `max_execution_time`) |
| `quantinuum_ghz.py` | GHZ-6 plan for `H2-1`: native Rz/PhasedX/ZZPhase payload + HQC estimate | `--backend H2-1E --submit` (emulator, eHQC), `--backend Helios-1 --submit --max-cost 20` |
| CLI only (no script): `python -m qumo_hardware run examples/foundations/bell.qasm -p ionq -b ionq_simulator:aria-1` | Bell pair plan for the IonQ Aria-1 simulator (`--dry-run`: no submission; cost reported as unknown) | `--yes` submits to the IonQ simulator (needs `IONQ_API_KEY`); the simulator price is not published |
| `compare_providers.py` | same circuit compiled for ibm_torino / ibm_kingston / ibm_sherbrooke (snapshots), H2-1, Helios-1: 2q gates, depth, ESP, cost | never submits; `--run-local` adds local emulator runs, `--provider-compilers` compares with Qiskit / TKET |

```bash
bench/.venv/bin/python examples/hardware/ibm_bell.py
bench/.venv/bin/python examples/hardware/quantinuum_ghz.py --backend H2-1LE --run   # offline H2 emulator
bench/.venv/bin/python examples/hardware/quantinuum_ghz.py --backend selene --run   # offline Helios-style emulator
bench/.venv/bin/python examples/hardware/compare_providers.py --qv 5 --provider-compilers
```

When a job is submitted the scripts print its id first. They exit with 1 if
the job ends in `ERROR`/`CANCELLED`; Ctrl-C while waiting prints the job id
(the job keeps running; fetch it with `python -m qumo_hardware result <id> --wait`).

Credentials (only needed for real devices):

* IBM: `export QISKIT_IBM_TOKEN=...` and `export QISKIT_IBM_INSTANCE=<CRN>` (or
  `QiskitRuntimeService.save_account(...)` once).
* Quantinuum: `python -c "import qnexus as qnx; qnx.login()"` once.
* IonQ: `export IONQ_API_KEY=...` (read only from the environment; never stored).

The same flows are available from the CLI:
`python -m qumo_hardware run examples/foundations/ghz.qasm -p quantinuum -b H2-1 --dry-run`
(with `PYTHONPATH=plugins/hardware`).
