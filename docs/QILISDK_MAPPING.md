# qilisdk → QUMO mapping

This document says which parts of [qilisdk](https://github.com/qilimanjaro-tech/qilisdk)
(Apache-2.0, Qilimanjaro) QUMO implements, under which QUMO names, which QUMO
already had, and which it leaves out, with the reason for each. The rules:

- QUMO names stay. A qilisdk name appears only in this table.
- Features are written in Mojo. No code is copied from qilisdk.
- Anything that cannot be made safe is left out, and this document says so.

The qilisdk source was read on `main` on 2026-11-02 (CHANGELOG at v0.3.1).
Numbers from qilisdk were not compared: qilisdk is not installed in this repository.

## Security: qilisdk's YAML and dill

qilisdk reads YAML with an unsafe loader and restores callables with dill. Both let
input name Python objects and run code while it loads. Checked in the qilisdk source:

- `src/qilisdk/yaml.py:296`: `yaml = QiliYAML(typ="unsafe")`. `serialize` and
  `deserialize` use this loader.
- `src/qilisdk/utils/serialization.py`: `deserialize` and `deserialize_from` call
  `yaml.load` on their input and check only the resulting type, after the load.
- `src/qilisdk/speqtrum/speqtrum_models.py`: the payload and result strings that the
  SpeQtrum server returns are passed to `deserialize`. The JSON envelope is read with
  `json.loads` (`speqtrum.py`, `_safe_json_loads`), but the YAML inside it is not.
- `src/qilisdk/yaml.py:148` and `:167`: `dill.loads` restores serialized functions and
  lambdas.

What this means: a compromised or spoofed SpeQtrum response, a man-in-the-middle on a
connection that does not verify the server, or a YAML file from anyone else can run
code in the process that loads it. `deserialize_from` on any file has the same risk.

What QUMO does instead: it has no unsafe loader and no dill. The Mojo side
(`qumo/ai/yaml_io.mojo`) reads flow-style YAML only, and refuses tags, anchors, aliases,
block style and several documents with QM-E915. The Python plugin
(`plugins/hardware/qumo_hardware/speqtrum.py`) parses responses with `json` only. When a
SpeQtrum result carries a YAML tag, it emits `UnsafeFormatWarning` at that moment, raises
QM-E955, and loads nothing. A test checks that a tag running `os.system` does not run. `tests/test_qilisdk_yaml.mojo` checks
those refusals, including a qilisdk-style tag (`!DigitalPropagation`) and a Python
object tag (`!!python/object/apply:...`), and checks that a tag-like text inside a
quoted string is still accepted.

## Names

### Already in QUMO (unchanged)

| qilisdk | QUMO | Where | Note |
|---|---|---|---|
| `Hamiltonian` (Pauli terms) | `PauliSum` | `qumo/ir/pauli.mojo` | No separate Hamiltonian type. |
| `Hamiltonian.transverse_field`, `ising`, `heisenberg` | `transverse_field_ising`, `heisenberg_xxz` | `qumo/algorithms/hamiltonians.mojo` | qilisdk takes scalars or per-term lists; QUMO takes scalars. |
| `AnalogEvolution` with the direct method | `exact_evolve` | `qumo/algorithms/hamiltonians.mojo` | Matrix exponential; the reference for small systems. |
| `TrotterizedSchedule`, `trotter_evolution` | `trotter` | `qumo/algorithms/library.mojo` | |
| `QAOA` | `qaoa_ansatz`, `qaoa_maxcut_ansatz` | `qumo/algorithms/library.mojo` | |
| `HardwareEfficientAnsatz` | `hardware_efficient_ansatz` | `qumo/algorithms/library.mojo` | Parameters and entanglement names differ. |
| `CircuitTranspiler` (SABRE layout and swap) | routing | `qumo/compiler/routing.mojo` | The algorithm was not compared with qilisdk's. |
| OpenQASM 2 and 3 | `parse_qasm`, `to_qasm`, `to_qasm2` | `qumo/ir/qasm.mojo`, `qasm2.mojo` | |
| `Readout` sampling and expectation | `sample`, expectation values | `qumo/sim/runner.mojo`, `hamiltonians.mojo` | |
| `KrausChannel`, `AmplitudeDamping`, `Dephasing` | `KrausChannel`, `amplitude_damping`, `phase_damping`, `thermal_relaxation` | `qumo/ir/kraus.mojo` | |
| `ReadoutAssignment` | `ReadoutCalibration`, `mitigate_counts` | `qumo/qec/characterization.mojo` | Calibration circuits and mitigation of counts. |
| `NoiseModel` | `NoiseModel`, `apply_noise_model` | `qumo/qec/characterization.mojo` | Pauli channels per qubit, not qilisdk's scopes. |

### Added in this change

| qilisdk | QUMO | Where | What is different |
|---|---|---|---|
| `QUBO` | `Qubo` | `qumo/algorithms/qubo.mojo` | Plain quadratic model over bits, with `offset`, `linear`, and `(i, j, coeff)` pairs with `i < j`. Energy: `offset + sum l_i x_i + sum Q_ij x_i x_j`. At most 64 variables. |
| `QUBO.to_hamiltonian` | `qubo_to_ising` | `qumo/algorithms/qubo.mojo` | Uses `x_i = (1 - z_i) / 2`. The result is a `PauliSum` with one term per variable, so the width survives. Not compared numerically with qilisdk. |
| Ising to QUBO (no qilisdk name) | `ising_to_qubo` | `qumo/algorithms/qubo.mojo` | Accepts identity, Z and ZZ terms only; anything else is QM-E812. The width comes from the sum, so trailing variables with no term are lost. |
| `BruteForceSolver` | `qubo_brute_force` | `qumo/algorithms/qubo.mojo` | Exact enumeration. QUMO caps it at 24 variables (QM-E811). Ties go to the smallest mask. |
| `SimulatedAnnealingSolver` | `qubo_simulated_annealing` | `qumo/algorithms/qubo.mojo` | QUMO's own single-bit Metropolis, not qilisdk's C++ code. Parameters `num_reads`, `num_sweeps`, `beta_min`, `beta_max`, `seed`. Deterministic for a fixed seed. Invalid values are QM-E822. |
| `Schedule` | `AnalogSchedule` | `qumo/algorithms/schedule.mojo` | Built with `total_time` and `num_qubits`. Each term has knots with `INTERP_LINEAR` or `INTERP_STEP`; the value is held at the ends. Symbolic coefficients are not supported. A term may act on fewer qubits than the schedule and is padded with identities. |
| `AnalogEvolution` of a schedule | `evolve_schedule` | `qumo/algorithms/schedule.mojo` | Midpoint rule with `steps` piecewise-constant steps, each applied with `exact_evolve`. Second order in the step for smooth coefficients. No RK4, Arnoldi or Lindblad solvers. |
| `GaussianPerturbation` | `gaussian_angle_perturbation` | `qumo/sim/perturbation.mojo` | Adds a normal sample to each rotation angle, per gate instance, seeded. Gate angles only. |
| `OffsetPerturbation` | `offset_angle_perturbation` | `qumo/sim/perturbation.mojo` | Adds a constant to each rotation angle. Symbolic angles keep their coefficient and move their offset. |
| `SpeQtrumProvider` (service client) | `SpeQtrumProvider`, `-p speqtrum` | `plugins/hardware/qumo_hardware/speqtrum.py` | Login, devices, execute, job status and result over HTTP with `urllib`; no new dependency. Credentials only from `QUMO_SPEQTRUM_USERNAME`, `QUMO_SPEQTRUM_APIKEY`, `QUMO_SPEQTRUM_URL` (required; https, or localhost for tests) and `QUMO_SPEQTRUM_AUDIENCE`. Nothing is sent before `confirm=True`. Targets from `--targets FILE` (the service publishes no topology). The payload is QUMO's YAML, inside qilisdk's field names, so the live service may refuse it: unverified. |
| `serialize`, `deserialize` (flow form) | `pauli_sum_to_yaml`, `pauli_sum_from_yaml`, `yaml_load_flow` | `qumo/ai/yaml_io.mojo` | Writes and reads YAML 1.2 flow style (JSON). No tags, anchors, aliases, block style or several documents. |

Errors: QM-E415 (perturbation), QM-E811 to QM-E814 (QUBO, schedule), QM-E822
(annealing parameters), QM-E915 (YAML). All are in `docs/ERRORS.md`.

### Not added, and why

- **QiliSim**, the C++ simulator. QUMO already has state-vector, tableau and branching
  simulators. A second native simulator would add a build dependency (pybind11, OpenMP,
  and cuBLAS loaded at run time) with no feature gap for the examples here.
- **CUDA-Q and QuTiP backends.** Vendor SDKs. The core stays hardware-neutral; the
  CUDA-Q examples run on QUMO's own simulators.
- **MPS simulator, non-Markovian `EnvironmentNoise`, `QTensor.permute`, seed on
  `expectation_value`.** These are only `changes/*.md` fragments on qilisdk's `main`,
  not in its 0.3.1 release. They are not stable yet.
- **Quantum reservoir computing** (`QuantumReservoir`, `ReservoirLayer`,
  `ReservoirInput`) and its datasets (Henon, logistic map, Lorenz, Mackey-Glass,
  NARMA, Santa Fe). Deferred to a later change. qilisdk also leaves readout training
  to the user.
- **Constrained models** (`Model`, `add_constraint`, Rosenberg linearization, slack
  and unbalanced penalties, variable encodings). Deferred. `Qubo` here is a plain
  quadratic model.
- **SCIP and SciPy solvers** (`ScipSolver`, `ScipySolver`). SCIP would add a
  dependency. QUMO's optimizers are its own.
- **Experiments** (`ExperimentFunctional`, S21 sweeps). Pulse-level; QUMO has no pulse
  layer.
- **Visualization styles, logging configuration, settings, OpenFermion interop.** Out
  of scope; no dependency added.

## Deliberate differences

- **Qubit width.** `PauliSum` has no width field. `AnalogSchedule` and the YAML reader
  therefore take the count explicitly. `pauli_sum_to_yaml` writes the highest qubit
  that carries a term, so a sum whose top qubits hold only identities loses its
  declared width in the document. qilisdk's behaviour here was not verified.
- **Perturbations** act on gate angles only. qilisdk also applies them to schedule
  parameters and scopes them by parameter name. Neither is implemented.
- **Time evolution** is the midpoint rule, not an integrator with an error control.
  Its accuracy is checked in `tests/test_qilisdk_schedule.mojo`: halving the step
  divides the error by about four.
- **Annealing** is QUMO's own algorithm. Its results match brute force in the tests
  but its schedule of inverse temperatures is a geometric one, not qilisdk's.

## Verification

- `tests/test_qilisdk_qubo.mojo`, `tests/test_qilisdk_schedule.mojo`,
  `tests/test_qilisdk_perturbation.mojo`, `tests/test_qilisdk_yaml.mojo`: each prints
  its `OK` line.
- `tests/test_mcp.mojo` checks the error catalogue; `docs/ERRORS.md` was regenerated
  with `QUMO_UPDATE_DOCS=1`.
- Not verified: numerical parity with qilisdk, and any behaviour of the SpeQtrum
  service (no account, no network test).

## Base

This branch starts from `2d3768d`, the size-cap commit, which is part of PR #1 and is
not merged yet. Merge PR #1 first, or rebase this branch onto `main` after it lands.
