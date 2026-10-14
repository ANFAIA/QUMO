# Hardware plugins: running QUMO circuits on quantum hardware

`plugins/hardware` (Python package `qumo_hardware`, distribution `qumo-hardware`)
runs circuits compiled by QUMO on real quantum computers. The first two
providers are **IBM Quantum** (Qiskit Runtime / IBM Quantum Compute API) and
**Quantinuum** (Quantinuum Nexus: System Model H2 and Helios). A `local`
provider runs the same payloads offline, so you can test everything without an
account.

```bash
make py                                                    # build/qumo_py.so: the QUMO compiler used by the plugins
export PYTHONPATH=plugins/hardware                         # or: pip install -e 'plugins/hardware[ibm,quantinuum]'
python -m qumo_hardware run examples/bell.qasm             # QUMO simulator, all-to-all
python -m qumo_hardware run examples/ghz.qasm -p ibm -b fake_torino            # IBM snapshot + Aer noise, no account
python -m qumo_hardware run examples/ghz.qasm -p quantinuum -b H2-1 --dry-run  # native payload + HQC estimate
python -m qumo_hardware run examples/ghz.qasm -p ibm -b ibm_fez                # prints the plan, does NOT submit (exit 3)
python -m qumo_hardware run examples/ghz.qasm -p ibm -b ibm_fez --yes          # submits (credentials needed)
python -m qumo_hardware result <job-id> --wait
```

## 1. Design

QUMO stays hardware neutral: **QUMO compiles for every provider**, and a
provider only describes its devices as data and ships an already-compiled
circuit.

```
 OpenQASM 2/3 ──► qumo_py.load_qasm ──► measure-all if needed (logical order)
                                │
 provider.target(backend) ──► TargetSpec {num_qubits, edges, basis, errors, durations, max_shots, all_to_all}
                                │
                                ▼
           select_region (best-calibrated connected qubits, optional)
                                │
            qumo_py.transpile(circ, n, edges, basis, level, seed)  ──► physical circuit + initial/final layout
                                │
                                ▼
           CompiledCircuit (OpenQASM 2, measure_map clbit→physical, layouts)
                                │
 provider.prepare ──► 1:1 SDK object (Qiskit ISA QuantumCircuit │ pytket native Circuit │ HUGR)
 provider.validate ─► the provider's own checks (is_isa_circuit / GatesInBasis │ QuantinuumBackend predicates)
 provider.estimate_cost ─► CostEstimate (QPU-seconds │ HQC │ eHQC │ free)
                                │   safety gate: remote ⇒ confirm=True / --yes
 provider.submit ──► JobHandle ──► local job store ($QUMO_HARDWARE_HOME/jobs)
 provider.status / provider.result ──► counts in the user's clbit order
```

Rules every provider follows:

* **No re-optimisation.** `prepare` translates gate-for-gate. The only allowed
  changes are semantics-preserving fixes a device requires, and they are
  reported in the plan (`finishing`): IBM ECR/CX direction (`GateDirection` +
  1q resynthesis) and IBM fractional `rzz` angle folding (`FoldRzzAngle`).
  `--provider-transpile` instead submits the vendor compiler's output (Qiskit
  preset pass manager / TKET `default_compilation_pass`) and prints a side-by-side
  comparison with QUMO.
* **Bit order.** Results are counts keyed by bitstrings over the user's
  classical bits, leftmost character = highest clbit (QUMO/Qiskit convention,
  [ARCHITECTURE.md](ARCHITECTURE.md#7-conventions)). Registers are flattened
  in declaration order. QUMO's compiled circuit measures the physical qubit that
  holds logical qubit *l* into the user's original clbit, so payloads already
  carry the user's classical layout; provider result formats (Qiskit
  `BitArray`, pytket tuples ordered `c[0], c[1], ...`, Helios/Selene register
  strings with `c[0]` first) are normalised by `reorder_counts`. Providers that
  return per-*physical*-qubit data (Braket's `measured_qubits`, IonQ) use
  `physical_counts_to_clbits(counts, measured_qubits, compiled)`.
* **Circuits without measurements** get `measure q[l] -> c[l]` for every
  logical qubit before compilation (`auto_measured: true`), like `measure_all`.
* **Parameters** are bound before compilation (`--params '[0.1, 0.2]'` in
  declaration order, or `{"t0": 0.1, ...}` by OpenQASM 3 `input` name).
  Simulation-only instructions (`#pragma qumo` noise channels, detectors) are
  refused with QM-E954: hardware brings its own noise.
* **Calibration-aware region.** On large coupling-map devices with error data,
  `select_region` picks `n + max(3, n/2)` connected qubits with the lowest
  readout + two-qubit error and QUMO places and routes inside that region
  (`region="none"` disables it, a list pins it). It only uses `TargetSpec` data,
  so any provider that reports errors benefits. On FakeTorino (Aer noise) a GHZ-8
  went from ~6 % to ~76 % correct outcomes with it (Qiskit's noise-aware layout: ~80 %).

## 2. Provider API

`qumo_hardware.base`:

| Type | Content |
|---|---|
| `TargetSpec` | `name, provider, num_qubits, edges` (undirected, a<b; `[]` if `all_to_all`), `basis` (QUMO gate names), `errors` (median `1q`/`2q`/`readout`), `durations` (s), `max_shots`, `directed_edges` (native direction of directional 2q gates, else `None`), `edge_errors`, `readout_errors`, `simulator`, `local`, `extras` |
| `CompiledCircuit` | `qasm` (physical, OpenQASM 2), `source_qasm` (logical), `num_qubits/num_logical/num_clbits`, `initial_layout/final_layout` (physical qubit of logical *l*), `measure_map` (clbit → physical), `ops`, `two_qubit_gates`, `depth`, `region`, `estimated_success()` |
| `CostEstimate` | `unit` (`QPU-seconds`, `HQC`, `eHQC`, `free`), `amount`, `formula`, `usd` (only with a known list price), `breakdown`, `notes` |
| `JobHandle` | `job_id, provider, backend, shots, local, status, meta` (layouts, measure map, provider options; never tokens), `counts` (local runs) |
| `Provider` (protocol) | `name`, `list_backends(online)`, `target(backend)`, `is_local(backend)`, `prepare(compiled, backend) -> (payload, report)`, `validate(payload, backend)`, `estimate_cost(compiled, shots, backend, payload=None)`, `submit(compiled, shots, backend, seed=None, payload=None, report=None, confirm=False, **opts) -> JobHandle` (refuses remote backends with QM-E957 unless `confirm=True`), `status(job)`, `result(job) -> counts`; optional `provider_compile` / `circuit_stats` (comparison), `public_options()` |

High-level workflow (`qumo_hardware.runner`), used by the CLI, examples and agents.
This snippet runs from the repository root without any account (`make py` first):

```python
import sys; sys.path.insert(0, "plugins/hardware")
from qumo_hardware import runner, HardwareError

qasm = open("examples/ghz.qasm").read()
plan = runner.plan(qasm, "quantinuum", "H2-1", shots=200)            # never submits, needs no login
print(plan.to_dict()["cost_summary"], plan.compiled.two_qubit_gates)
out = runner.run(qasm, "local", "qumo-sim:ibm:fake_torino", shots=1000)   # local: runs now
print(out["counts"])
try:
    runner.run(qasm, "quantinuum", "H2-1", shots=200)                 # remote: refused with QM-E957
except HardwareError as e:
    if e.code != "QM-E957":                                             # e.g. QM-E952 (credentials)
        raise
    print(e.details["plan"]["cost_summary"])                            # review, then confirm=True
# After a confirmed remote run (out = runner.run(..., confirm=True, wait=False)):
#   runner.job_status(out["job_id"]); runner.job_result(out["job_id"], wait=True)
```

`runner.run` returns a JSON-ready dict: the plan plus `submitted`, `job_id`,
`status`, `counts`; `warning` and `next` when a waited-for job has no result
(still queued at `timeout`, or ended in `ERROR`/`CANCELLED`);
`server_dry_run`/`mock_results` when the counts are mock data (IBM
`--ibm-dry-run`). `on_submitted=callback` is called with the `JobHandle` right
after submission; if the wait is interrupted (Ctrl-C), the `KeyboardInterrupt`
carries the job id as `exc.qumo_job_id`.

## 3. CLI

`python -m qumo_hardware <command>` (or the `qumo-hardware` script when installed).
Every command takes `--json` (machine output on stdout; failures as
`{"error": {"code", "severity", "message", "hint", "docs"}}` like `build/qumo`).

| Command | Purpose |
|---|---|
| `run CIRCUIT -p PROVIDER -b BACKEND [--shots N] [--level 0-3] [--seed S] [--dry-run] [--yes] [--max-cost X] [--no-wait] [--timeout S] [--provider-transpile] [--fractional] [--ibm-dry-run] [--params JSON] [--show-qasm]` | compile with QUMO, validate, estimate, and run (remote only with `--yes`). `--max-cost` is in the provider's unit (IBM: QPU seconds; Quantinuum: HQC/eHQC) and is also sent as the server-side cap |
| `estimate CIRCUIT -p P -b B --shots N` | plan only, never submits |
| `backends [-p P] [--online]` | static/offline list; `--online` also asks the provider (credentials) |
| `target -p P -b B [--full]` | the neutral `TargetSpec` |
| `status JOB` / `result JOB [--wait]` / `jobs` | job store and provider queries |
| `errors` | the QM-E95x catalogue |

Exit codes: `0` ok, `1` error (QM-E95x; also when `run` waited for a job that
ended in `ERROR` or `CANCELLED`), `2` usage, `3` **not submitted: confirmation
required**, `130` interrupted (Ctrl-C) while waiting: the job id is printed
and the job keeps running (`result JOB --wait` fetches it later). While `run`
waits for a remote job it prints `submitted job <id>` to stderr first.
Plan notes (for example `--fractional ignored: ...`) are printed as `note`
lines and returned in `notes`.
`CIRCUIT` may be `-` (stdin). IBM options: `--channel`, `--instance`, `--account`;
Quantinuum: `--project` (Nexus project, default `qumo-hardware`).

## 4. IBM Quantum

**Access.** Create an IBM Cloud account at <https://quantum.cloud.ibm.com>, an
instance and an API key. Plans (IBM pricing page, checked 2026-10-12): Open
Plan, free, "up to 10 minutes of runtime per month" (IBM's documentation
describes it as 10 QPU minutes per 28-day rolling window); Pay-As-You-Go
"starts at $96 USD / minute", billed per second of QPU usage; Flex and
Premium are contracts with lower per-minute prices. Credentials are read **only**
through qiskit-ibm-runtime's standard mechanisms; the plugin never reads, stores
or prints a token:

```bash
export QISKIT_IBM_TOKEN=<api key>              # environment (CI, containers)
export QISKIT_IBM_INSTANCE=<instance CRN or name>
# or once, saved to ~/.qiskit/qiskit-ibm.json:
python -c "from qiskit_ibm_runtime import QiskitRuntimeService as S; S.save_account(token='...', instance='...', set_as_default=True)"
```

`QiskitRuntimeService()` resolves the account (env vars first, then the saved
default; channel `ibm_quantum_platform` unless `QISKIT_IBM_CHANNEL`/`--channel`
says `ibm_cloud`). Missing or rejected credentials raise **QM-E952** with this
hint, never a stack trace. Other IBM SDK failures are mapped too: backend
lookup → QM-E953, job submission → QM-E959 (QM-E955 for input validation
errors, QM-E952 for authorisation errors), job lookup/status/result →
QM-E959. Their message keeps the exception type and a shortened first line in
which anything that looks like a credential is replaced by `<redacted>`.

**Target.** `target_from_qiskit(backend)` reads `backend.target`: basis
`rz, sx, x` + `cz` (Heron: ibm_torino, ibm_fez, ibm_marrakesh, ibm_kingston, ...)
or `ecr` (Eagle); undirected couplings, median errors/durations, per-qubit
readout errors, per-edge 2q errors, `max_shots`, `default_rep_delay`. Directional
gates set `directed_edges`. `--fractional` fetches the backend with
`use_fractional_gates=True` and adds `rzz` and `rx` to the basis (Heron); QUMO
then emits arbitrary-angle `rzz`, folded into IBM's allowed range [0, π/2] by
`FoldRzzAngle` and checked with qiskit-ibm-runtime's rzz validator. If the
backend has no fractional gates, `--fractional` has no effect and the plan says
so in a note (`--fractional ignored: ...`). Offline, the snapshot
`fake_fractional` (5 qubits) has them; `backends --provider ibm` lists it.

**Payload.** QUMO's compiled circuit is an ISA circuit over all device qubits:
`to_qiskit` maps every gate to the Qiskit standard gate class (`ECRGate`,
`RZZGate`, ...); it is validated with `qiskit_ibm_runtime.utils.is_isa_circuit`
and Qiskit's `GatesInBasis(target)`.

**Execution.** qiskit-ibm-runtime 0.50 deprecates the server-side `SamplerV2`;
the plugin uses the replacement client-side
`qiskit_ibm_runtime.executor_sampler.Sampler` (built on `Executor`) in job mode,
with gate/measure twirling and dynamical decoupling **off** (QUMO's circuit runs
as compiled). Every remote job carries a server-side cap,
`SamplerOptions.max_execution_time` (IBM cancels a job whose QPU usage exceeds
it): `--max-cost` (QPU seconds, rounded up) when given, otherwise
`max(30 s, 2 × estimate)`, recorded in the job record as
`max_execution_time`. `--ibm-dry-run` forwards IBM's server-side
`dry_run=True` (job validation and usage estimate; access may be restricted);
its counts are **mock data**, so `run` and `result` mark them with
`"server_dry_run": true, "mock_results": true` and a `warning`.
Results: `PrimitiveResult[0].join_data().get_counts()`.

**Local mode.** Backends named `fake_*` (FakeTorino, FakeFez, FakeKingston,
FakeSherbrooke, ...; calibration snapshots in qiskit-ibm-runtime) run through
the *same* Sampler code path on Qiskit Aer with the snapshot's noise model, no
account. Aer simulates only the active qubits, but with memory exponential in
their number, so payloads with more than 26 active qubits are refused with
QM-E956 before Aer starts. `qumo-sim:ibm:fake_torino` simulates the exact ISA
payload ideally with QUMO (only active qubits are simulated). Loading the
snapshots makes qiskit-ibm-runtime warn that `fake_nighthawk`'s properties are
not typical; the plugin silences that warning and shows it as a note on
`fake_nighthawk` only.

**Cost.** IBM bills QPU time. Estimate (IBM's run-time guide):
`2 s + shots × (rep_delay + circuit_duration)`, with `rep_delay =
backend.default_rep_delay` (250 µs) and `circuit_duration =
QuantumCircuit.estimate_duration(target)`. The USD figure uses the
Pay-As-You-Go list price USD 1.60/s; override with `QUMO_IBM_USD_PER_SECOND`.

## 5. Quantinuum

**Access.** A Quantinuum Nexus account (<https://nexus.quantinuum.com>) with an
organisation that has H-series/Helios quota (HQCs). Log in once; qnexus stores
its refresh/access tokens in `~/.qnx/auth` (about 30 days). The plugin only
calls `qnexus.auth.is_logged_in()`; it never touches the tokens:

```bash
python -c "import qnexus as qnx; qnx.login()"                    # browser flow
python -c "import qnexus as qnx; qnx.login_with_credentials()"   # prompt (email/password)
```

**Devices** (static table in `quantinuum.py`; `backends --online` lists what your account sees):

| Name | Kind | Program format |
|---|---|---|
| `H2-1`, `H2-2` | hardware, 56 qubits | pytket circuit, `QuantinuumConfig(device_name=...)` |
| `H2-1E`, `H2-2E`, `H2-Emulator` | Nexus emulators (eHQC) | pytket circuit |
| `H2-1SC`, `H2-2SC` | syntax checkers (free) | pytket circuit |
| `Helios-1` | hardware, 98 qubits | HUGR, `HeliosConfig(system_name=...)` |
| `Helios-1E`, `Helios-1E-lite` | Nexus emulators (eHQC) | HUGR + `HeliosEmulatorConfig()`; the qubit count goes with the job (`start_execute_job(n_qubits=[...])`; the `HeliosEmulatorConfig.n_qubits` field is deprecated) |
| `Helios-1SC` | syntax checker / cost estimator | HUGR |
| `H2-1LE` | **local**, offline | pytket circuit, pytket-quantinuum local emulator (`[pecos]` extra) |
| `selene` | **local**, offline | HUGR, Guppy + Selene state-vector emulator |

Unknown names that follow Quantinuum's scheme (`…E`, `…SC`, `…LE`) are accepted
and classified accordingly.

**Target and payload.** All-to-all, QUMO basis `rz, rx, ry, rzz`, translated 1:1
to the native gates: `Rx(θ) → PhasedX(θ/π, 0)`, `Ry(θ) → PhasedX(θ/π, ½)`,
`Rz → Rz` (virtual), `rzz(θ) → ZZPhase(θ/π)` (pytket `AutoRebase`). The payload
is validated offline with `QuantinuumBackend(..., data=QuantinuumBackendData(...))`
predicates. H2 jobs upload the circuit (`qnx.circuits.upload`) and run with
`QuantinuumConfig(device_name=..., no_opt=True)` (the default: the server does
not re-optimise QUMO's circuit). Helios jobs wrap the circuit with
`guppy.load_pytket` into a Guppy `main` that allocates the qubits, runs the
circuit, emits `result("c", bits)` and discards the qubits; the HUGR package is
uploaded with `qnx.hugr.upload` and run with `HeliosConfig`. Submission:
`qnx.start_execute_job(programs, n_shots, backend_config, name, project, max_cost, n_qubits)`;
status: `qnx.jobs.status`; results: `qnx.jobs.results(job)[0].download_result()`.

**Classical registers.** QUMO payloads have one register `c` with one bit per
user clbit. pytket's OpenQASM reader limits registers to 32 bits by default;
the bridge raises that limit to the register width. H2 programs allow at most
63 bits per register (`max_classical_register_width` in pytket-quantinuum's
device data), so for H2 a wider `c` is split into `c_0, c_1, ...` of 63 bits
(clbit *k* → `c_{k // 63}[k % 63]`). The plan shows the layout
(`finishing.clbit_registers`), the job record stores it, and results are read
back through it, so counts stay in the user's bit order. Helios and Selene
return bool arrays and keep one register `c` (up to 98 bits on Helios-1).
Conditions (`if(c==k)`) compare a whole register, so a condition on a register
wider than 63 bits cannot run on H2 and is refused with QM-E956.

**Classical conditions on Helios.** Helios programs are HUGR built by
guppylang's `load_pytket`, which cannot lower conditional gates yet (the Selene
HUGR-QIS compiler reports "Pytket op 'Conditional' is not currently
supported"). Such circuits are refused with QM-E955 before anything is
uploaded; run them on an H2 backend (`H2-1`, `H2-1E`, `H2-1SC`, `H2-1LE`).

**Cost.** `HQC = 5 + C (N1q + 10 N2q + 5 Nm) / 5000`, C = shots, computed on the
native payload: N1q = PhasedX gates (Rz is virtual), N2q = ZZPhase/ZZMax/TK2,
Nm = measurements + resets + initial state preparations (one per qubit).
Emulators consume eHQC with the same formula; syntax checkers and local
emulators are free. For Helios the official figure uses per-shot maxima from a
10-shot syntax-checker simulation (identical for static circuits). Every
billable submission carries a server-side `max_cost` cap: `--max-cost` if given,
otherwise `1.25 × estimate + 1`. H2 caps jobs at 10,000 shots. An exact quote
is available from Nexus (`qnx.circuits.cost` / `qnx.hugr.cost`, which run on
the syntax checker). Set `QUMO_QUANTINUUM_USD_PER_HQC` to get a USD figure
(HQC prices depend on your contract).

## 6. Local emulators (`-p local`)

| Backend | What runs |
|---|---|
| `qumo-sim` | QUMO state vector, all-to-all, any gate (≤ 26 active qubits) |
| `qumo-sim:line:N`, `ring:N`, `grid:RxC`, `all:N` + optional `/basis` (`sc_cz`, `sc_ecr`, `sc_iswap`, `ion_ms`, `ion_zz`, `atom_cz`, or `rz,ry,cz`) | QUMO compile for that topology, ideal QUMO simulation |
| `qumo-sim:ibm:<fake>` / `qumo-sim:quantinuum:<device>` | exact provider TargetSpec, ideal QUMO simulation |
| `fake_<ibm>` | IBM snapshot + Aer noise via the IBM plugin's Sampler path |
| `H2-1LE`, `selene` | Quantinuum local emulators via the Quantinuum plugin |
| any `qumo-sim...` name + `+noise:<modality>` (`superconducting`, `trapped_ion`, `neutral_atom`, `custom(p1,p2,pm)`) | QUMO compile for the target, then a vendor-neutral depolarizing + readout noise model, sampled locally (free) |

### Vendor-neutral noisy emulator (`+noise:<modality>`)

A noise model named after a hardware *modality*, not a vendor, so examples can
show "what a noisy QPU does to this circuit" without picking a company:

```bash
python -m qumo_hardware run examples/ghz.qasm -p local -b 'qumo-sim:grid:4x4/sc_cz+noise:superconducting' --shots 2000
python -m qumo_hardware run examples/ghz.qasm -p local -b 'qumo-sim:all:16/ion_zz+noise:trapped_ion'
python -m qumo_hardware run examples/ghz.qasm -p local -b 'qumo-sim+noise:custom(1e-4,1e-3,2e-2)'
```

| Modality | 1q gate error | 2q gate error | readout error |
|---|---|---|---|
| `superconducting` | 2.5e-4 | 3e-3 | 1e-2 |
| `trapped_ion` | 3e-5 | 1e-3 | 3e-3 |
| `neutral_atom` | 5e-4 | 5e-3 | 1e-2 |
| `custom(p1,p2,pm)` | p1 | p2 | pm (each in [0, 0.75]) |

The rates are QUMO's `HardwareProfile` presets (qumo/exec/resources.mojo), so
the emulator and `qumo_estimate_resources` agree on what a modality means.
After QUMO compiles the circuit for the target topology and basis, the plugin
inserts `#pragma qumo depolarize1(p1)` after every 1-qubit gate,
`depolarize2(p2)` after every 2-qubit gate (`depolarize1(p2)` per qubit for
wider gates) and `x_error(pm)` before every measurement
(`inject_noise`, plugins/hardware/qumo_hardware/local.py).

Sampling (`sample_noisy`) is exact in distribution and seeded: each shot draws
its Pauli error pattern; shots without gate errors share one static
simulation and every distinct error pattern is simulated once with explicit
Pauli gates; readout flips are applied classically. Circuits that are dynamic
after compilation (mid-circuit measurement followed by more gates on the same
qubit, conditions, resets) fall back to QUMO's per-shot trajectories, which
are slower (a routed 16-qubit QAOA p=2 on `grid:4x4` with 2000 shots takes
about 2 minutes on an M5 CPU). `QUMO_HW_NOISE_WORKERS=N` opts into process
parallelism for the pattern route.

The result carries `noisy: true`, `noise_model` and `noise_sampling` (method,
error-free shots, distinct patterns); the cost estimate stays "free, local".
This is a depolarizing approximation, not a device calibration: use
`ibm:fake_<backend>` (calibration snapshot + Aer) or `quantinuum:H2-1LE` for
vendor-specific noise. Tests: plugins/hardware/tests/test_local_noise.py
(Bell fidelity drop within 3 sigma of the analytic value, determinism per
seed, noise-free unchanged, unknown modality refused, plan says local and
free).

## 7. Safety model

* Default is safe: `run` submits to a **remote** backend (anything not local)
  only with `--yes` / `confirm=True`. Without it, it compiles, validates,
  estimates, prints backend, shots and cost, and exits with code 3 / raises
  QM-E957 whose `details.plan` holds the full plan for review. The providers
  enforce the same rule again: `Provider.submit(..., confirm=False)` refuses a
  remote backend with QM-E957, so calling a provider directly is not a way
  around the gate.
* `--dry-run` and `estimate` never submit. `--max-cost` refuses (QM-E956)
  when the estimate is above the limit. Every remote job also carries a
  server-side cap: Quantinuum `max_cost` (`--max-cost`, else
  `1.25 × estimate + 1` HQC) and IBM `max_execution_time` (`--max-cost`, else
  `max(30 s, 2 × estimate)` QPU seconds).
* `qumo_py.so` (the QUMO compiler) is loaded from an explicit path only:
  `QUMO_PY_PATH` (a `.so` file or a directory that holds `qumo_py.so`), else
  `build/qumo_py.so` of the checkout that contains the package. The current
  directory and `sys.path` are never searched, so running the CLI inside an
  untrusted directory cannot load a planted extension. A non-editable
  `pip install` of the plugin therefore needs `QUMO_PY_PATH`.
* Credentials only through the providers' standard mechanisms (IBM env vars or
  saved account; qnexus login store). No token is read, written or printed by
  the plugins, and none is stored in the repository or in job records
  (`$QUMO_HARDWARE_HOME/jobs`, default `~/.qumo/hardware/jobs`, holds job ids,
  backend names, layouts and counts).
* Errors are coded (QM-E95x) with hints; no stack traces (set
  `QUMO_HW_DEBUG=1` to see one for an unexpected `QM-E999`).
* The test suite redirects `HOME`, the IBM account file and the job store and
  unsets `QISKIT_IBM_*`, so it can never pick up real credentials or submit.

## Error codes

New range **QM-E950–QM-E959** (inside QUMO's `QM-E9xx` interface range;
`python -m qumo_hardware errors --json`):

| Code | Meaning | Typical fix |
|---|---|---|
| QM-E950 | `build/qumo_py.so` cannot be imported | `make py`, or `QUMO_PY_PATH=<dir>` |
| QM-E951 | provider SDK missing | `pip install 'qumo-hardware[ibm]'` / `[quantinuum]` / `[helios]` / `[local-quantinuum]` |
| QM-E952 | credentials missing, expired or rejected | IBM: `QISKIT_IBM_TOKEN` + `QISKIT_IBM_INSTANCE` or `save_account`; Quantinuum: `qnx.login()` |
| QM-E953 | unknown provider/backend, or not available to the account | `backends --provider P [--online]` |
| QM-E954 | cannot compile (parse error, too many qubits, unbound parameters, missing file) | fix the circuit / pick a larger device |
| QM-E955 | payload rejected by the provider's own validation, or it cannot be lowered to the provider's program format (e.g. conditions on Helios) | follow the hint; for a bridge failure report the payload (`--show-qasm`) |
| QM-E956 | shot, size or cost limit exceeded; the hint is specific to the limit | shots: lower `--shots` or split the run; local emulator size: shrink the circuit or use a cloud emulator; cost: raise `--max-cost` only if intended |
| QM-E957 | not submitted: confirmation required | review the plan, add `--yes` |
| QM-E958 | unknown job id | `jobs` |
| QM-E959 | job failed, cancelled or not finished | `status JOB`, `result JOB --wait` |

These codes live in `qumo_hardware/errors.py`; they are not (yet) part of the
Mojo `error_catalog()` that generates [ERRORS.md](ERRORS.md).

## 8. Installation and tests

Versions verified on 2026-10-12: qiskit 2.5.2, qiskit-ibm-runtime 0.50.0,
qiskit-aer 0.17.2, qnexus 0.51.1, pytket 2.18.5, pytket-quantinuum 0.59.3,
pytket-pecos 0.3.3 / quantum-pecos 0.8.0.dev8, guppylang 1.0.0a4 (with
guppylang-internals 1.0.0a4), selene-sim 0.2.18.

```bash
uv pip install --python bench/.venv/bin/python qnexus pytket-quantinuum          # IBM deps already in bench/.venv
uv pip install --prerelease=allow --python bench/.venv/bin/python \
    "pytket-quantinuum[pecos]==0.59.3" "pytket==2.18.5" "guppylang==1.0.0a4" "guppylang-internals==1.0.0a4"
bench/.venv/bin/python -m pytest plugins/hardware/tests -q                        # 160 tests (156 run, 4 slow skipped), no accounts
QUMO_HW_SLOW=1 bench/.venv/bin/python -m pytest plugins/hardware/tests -q -m slow # + the 4 slow H2-1LE / Selene emulator tests
```

Pin `pytket` when adding the `pecos` extra: an unpinned resolve can downgrade
pytket to 1.x. The tests cover: TargetSpec extraction from five IBM snapshots
(+ fractional, directionality), Quantinuum and local targets; QUMO compilation
→ valid ISA (plugin and qiskit-ibm-runtime checks) and statevector
equivalence up to layout on IBM, Quantinuum and five local topologies; bit
order end to end with non-palindromic expected keys (single X on each qubit,
routed circuits, permuted measurements, auto measure-all, GHZ, noisy
FakeTorino, H2-1LE, Selene); wide classical registers on Quantinuum (33, 63,
64 and 98 clbits: plan, payload registers, and a round trip through a mocked
Nexus whose fake device evaluates the uploaded payload; register split on the
real H2-1LE emulator); conditions refused on Helios and on > 63-bit H2
registers; both OpenQASM 2 exporter styles loading 1:1 into Qiskit and pytket;
native Quantinuum payloads (1:1 gate counts, operator equivalence); HQC and
QPU-time formulas; IBM server-side cap, mock-result flags, SDK errors mapped to
QM-E95x with redaction, the fake-backend size guard and `--fractional` notes;
mocked qnexus and IBM submissions/status/results; provider-level confirmation;
`qumo_py.so` never loaded from the working directory; CLI JSON, exit codes
(including failed jobs and Ctrl-C), refusal without `--yes`, credential errors
without tracebacks, no `fake_nighthawk` warning.

## 9. Adding a provider (Braket, IonQ, IQM, ...)

1. Write a class with the `Provider` methods (copy `quantinuum.py` for an
   all-to-all device or `ibm.py` for a coupling map). `target()` returns a
   `TargetSpec` in QUMO gate names; `prepare()` translates QUMO's OpenQASM 2
   1:1 (`qasm_bridge.to_qiskit` / `to_pytket` or the vendor's OpenQASM 3 import).
2. Map results to the user's clbits: if the service returns per-clbit data use
   `reorder_counts`; if it returns per-physical-qubit data use
   `physical_counts_to_clbits(counts, measured_qubits, compiled)`.
3. `is_local()` must be `False` for anything that leaves the machine (the
   safety gate depends on it); `estimate_cost()` must return the provider's
   billing unit.
4. Register it, either in `registry.BUILTIN` or from a separate package:
   `[project.entry-points."qumo_hardware.providers"] braket = "qumo_braket:BraketProvider"`.
5. Add tests with mocked SDK calls (see `tests/test_quantinuum.py`).

Notes for the next ones:

* **Amazon Braket** (`amazon-braket-sdk`): `AwsDevice(arn).properties` gives
  the native gate set and connectivity (`paradigm.connectivity`, empty = all-to-all);
  submit OpenQASM 3 `Program` with `disable_qubit_rewiring=True` (verbatim box)
  so QUMO's layout is kept; `result.measurement_counts` are keyed by
  `measured_qubits` order → `physical_counts_to_clbits`. Cost: per task + per
  shot (device pricing table).
* **IonQ** (direct API or Braket): all-to-all, native `gpi, gpi2, ms/zz`;
  extend QUMO's basis translation or rebase `rx/ry/rz/rxx` 1:1 (`rx = gpi2`
  sequences are not 1:1, so translate at the gate level and report it).
  Results are probabilities/histograms over qubit indices.
* **IQM** (`iqm-client`): coupling map from `DynamicQuantumArchitecture`,
  native `prx` + `cz`; QUMO's `rz, rx, ry, cz` maps with `rz` absorbed into
  `prx` phases.

## 10. Limitations

* Single circuit per job (no PUB broadcasting / parameter sweeps / sessions or
  batches yet); parameters are bound once before submission.
* Dynamic circuits: mid-circuit measurements and conditions on whole classical
  registers (`if(c==k)`) compile and run locally and on H2 (registers up to 63
  bits); single-bit conditions (e.g. `examples/teleport.qasm`) are refused
  with QM-E954 because the bridges use OpenQASM 2 (an OpenQASM 3 bridge is the
  fix); Helios refuses conditions (QM-E955, see section 5). Dynamic circuits
  are untested on IBM and Quantinuum hardware.
* Region selection is a greedy heuristic on per-qubit readout and per-edge
  two-qubit errors; QUMO's own layout does not yet consume error rates.
* Cost figures are estimates (IBM: QPU-time heuristic; Quantinuum: HQC formula
  on the static circuit). Use the providers' quotes (`--ibm-dry-run`,
  `qnx.circuits.cost`) for billing-grade numbers.
* Quantinuum device table and nominal error rates are static (verified
  2026-10-12); `backends --online` shows what Nexus offers your account.
* Helios submission depends on guppylang's `load_pytket` (pre-release 1.0
  line) and on HUGR acceptance by Helios; it is exercised locally with Selene
  and with mocked Nexus calls, not on Helios hardware.
* The IBM ECR direction fix and fractional-angle folding use Qiskit passes
  (semantics-preserving and reported); everything else is QUMO's output.
* `qasm_bridge` removes the `gate` definitions of gates the SDK knows natively
  with the same matrix (Qiskit: `ecr`, `iswap`, `iswapdg`, `ryy`, `rzx`, `ccz`
  as built-in classes, the rest via `LEGACY_CUSTOM_INSTRUCTIONS`; pytket:
  `sx, sxdg, swap, cswap, crx, cry, rxx, rzz, ecr`), so each hardware gate
  stays one SDK gate. This started as a workaround for an older exporter that
  defined `ecr` through the undeclared `rzx`; it is also what keeps `rzz` a
  single `ZZPhase` now that the fixed exporter defines every non-qelib1 gate.
  Both exporter styles are covered by tests.
* The IBM fake-backend guard (26 active qubits) and the local emulator limits
  are memory guards, not properties of the devices.
* The IBM server-side cap uses a heuristic estimate; if IBM cancels a job for
  exceeding `max_execution_time`, rerun with a larger `--max-cost`.

## Sources

* qiskit-ibm-runtime release notes (0.47–0.50: Executor, client-side Sampler,
  SamplerV2 deprecation, `dry_run`, `instance="auto"`, fractional-gate fixes):
  <https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/release-notes>
* IBM, "Estimate job run time" (overhead + (rep_delay + circuit length) × executions):
  <https://quantum.cloud.ibm.com/docs/en/guides/estimate-job-run-time>
* IBM Quantum pricing (Open Plan "up to 10 minutes of runtime per month"; Pay-As-You-Go
  "starts at $96 USD / minute", billed per second; checked 2026-10-12): <https://www.ibm.com/quantum/pricing>
* IBM, "Maximum execution time" (`max_execution_time` is QPU usage, jobs over it are cancelled,
  three-hour service maximum; Open Plan 10 minutes per 28-day rolling window):
  <https://quantum.cloud.ibm.com/docs/en/guides/max-execution-time>
* qiskit-ibm-runtime 0.50 source: `accounts/management.py` (QISKIT_IBM_* variables,
  `~/.qiskit/qiskit-ibm.json`), `executor_sampler/sampler.py`, `utils/utils.py`
  (`is_isa_circuit`, rzz angle check), `transpiler/passes/basis/fold_rzz_angle.py`.
* Quantinuum Nexus, getting started (login, projects, `start_execute_job`, results):
  <https://docs.quantinuum.com/nexus/trainings/notebooks/basics/getting_started.html>
* Quantinuum Helios getting started and job costing (HeliosConfig, Helios-1E,
  Helios-1SC, HQC formula with per-shot maxima):
  <https://docs.quantinuum.com/systems/trainings/helios/getting_started/>,
  <https://docs.quantinuum.com/systems/trainings/helios/getting_started/costing.html>
* Quantinuum H2 operation (native gates Rz/PhasedX/ZZMax/ZZPhase/TK2, 10,000 shots per job):
  <https://docs.quantinuum.com/systems/user_guide/hardware_user_guide/h2.html>
* Azure Quantum, Quantinuum provider (HQC/eHQC formula, H2 emulators and syntax checkers):
  <https://learn.microsoft.com/azure/quantum/provider-quantinuum>
* pytket-quantinuum changelog (remote submission removed in 0.56, local emulator kept):
  <https://docs.quantinuum.com/tket/extensions/pytket-quantinuum/changelog.html>
* qnexus 0.51.1 and quantinuum-schemas 7.8.2 source (`QuantinuumConfig`,
  `HeliosConfig`, `HeliosEmulatorConfig.n_qubits` deprecation,
  `KNOWN_NEXUS_HELIOS_EMULATORS`, token store `~/.qnx/auth`).
* pytket 2.18.5 `pytket.qasm.circuit_from_qasm_str(..., maxwidth=32)` and
  pytket-quantinuum 0.59.3 `backends/api_wrappers.py`
  (`max_classical_register_width: 63`).
