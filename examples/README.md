# QUMO examples

Run everything from the repository root after `make setup` (Mojo toolchain in `.venv`). Mojo examples:
`.venv/bin/mojo run -I . examples/<category>/<file>.mojo`. OpenQASM files:
`build/qumo run examples/<category>/<file>.qasm` (after `make cli`).

## Start here

Five examples in order of difficulty. Each runs in seconds; the Mojo ones spend 3 to 5 s compiling first.
New to QUMO? Do the [tutorial](../docs/TUTORIAL.md) first: it explains every line of the first one.

| # | Example | What you learn | Run |
|---|---|---|---|
| 1 | [`foundations/bell.qasm`](foundations/bell.qasm) | A two-qubit circuit in OpenQASM and its histogram: about half `00`, half `11` | `build/qumo run examples/foundations/bell.qasm --shots 1000 --seed 7` |
| 2 | [`foundations/teleport.qasm`](foundations/teleport.qasm) | Mid-circuit measurement and `if`: a state is teleported to the third qubit, which reads 1 in about 27% of the shots | `build/qumo run examples/foundations/teleport.qasm --shots 1000 --seed 7` |
| 3 | [`algorithms/grover.mojo`](algorithms/grover.mojo) | A first Mojo program (40 lines): Grover search from 3 to 8 qubits, exact and sampled success probability | `.venv/bin/mojo run -I . examples/algorithms/grover.mojo` |
| 4 | [`chemistry_and_physics/vqe_h2.mojo`](chemistry_and_physics/vqe_h2.mojo) | A variational algorithm: the ground energy of H2 with four optimisers, all within chemical accuracy | `.venv/bin/mojo run -I . examples/chemistry_and_physics/vqe_h2.mojo` |
| 5 | [`error_correction/qec_memory_experiment.mojo`](error_correction/qec_memory_experiment.mojo) | Error correction end to end: surface-code circuits, noise, decoding, logical error rate against distance and noise | `.venv/bin/mojo run -I . examples/error_correction/qec_memory_experiment.mojo` |

After these, browse the categories below or [`CATALOG.md`](CATALOG.md).

## How the examples are organised

Examples are organised **by category**, by what they show, not by the SDK they come from. Every
example says where it comes from in its header:

```
Reproduces: Qiskit - CHSH inequality                         what it reproduces
Source: https://quantum.cloud.ibm.com/docs/tutorials/...      the exact original page or notebook
Framework: Qiskit (IBM) - https://github.com/Qiskit/qiskit    the SDK or framework it comes from
```

Examples that are QUMO's own say `Framework: QUMO`. [`CATALOG.md`](CATALOG.md) lists every example with
its origin and original. To run the ones that come from one SDK, use `--framework qiskit` (also `braket`,
`cirq`, `classiq`, `cudaq`, `guppyft`, `pennylane`, `qibo`, `qsharp`, `tket`, `qumo`). Each SDK's notes,
reference numbers and optional cross-check scripts are in [`origins/<framework>/`](origins/).

## Categories

| Folder | What it holds |
|---|---|
| [`foundations/`](foundations/) | Bell pair, GHZ and teleportation as OpenQASM inputs; CHSH, Deutsch-Jozsa, Bernstein-Vazirani, Simon, the swap and Hadamard tests, textbook algorithms |
| [`algorithms/`](algorithms/) | Grover, phase estimation, QFT, Shor and order finding, discrete logarithms, HHL, hidden shift, amplitude amplification, quantum walks |
| [`optimization/`](optimization/) | QAOA and combinatorial problems: MaxCut, vertex cover, independent set, colouring, set cover |
| [`chemistry_and_physics/`](chemistry_and_physics/) | VQE for H2, Hamiltonian simulation and product formulas, adiabatic evolution, lattice models, sample-based Krylov and SQD |
| [`qml/`](qml/) | Quantum machine learning, following PennyLane's demos (embeddings, variational classifiers, data re-uploading, kernels, QNG, barren plateaus, QCNN, QCBM), verified against pinned PennyLane reference numbers ([docs/QML.md](../docs/QML.md)) |
| [`error_correction/`](error_correction/) | Surface-code memory experiment end to end, plugging in your own decoder, Stim interoperability |
| [`ft/`](ft/) | Fault-tolerant programming on stabilizer codes: Steane, Iceberg, Helix, magic-state T gates, logical-error sweeps ([docs/FT.md](../docs/FT.md)) |
| [`noise_and_mitigation/`](noise_and_mitigation/) | Pauli-channel noise, trajectories, zero-noise extrapolation, readout mitigation, XEB, randomized benchmarking, quantum volume, calibration to noise model |
| [`compilation/`](compilation/) | Compiling for several hardware modalities, noise-aware layout, Pauli-based computation, measurement reduction, resource estimation |
| [`simulation/`](simulation/) | CPU vs GPU sampling, batched expectation values and parallel gradients, stabilizer branching beyond the state vector, device emulation |
| [`control_flow/`](control_flow/) | Programs with `while` and classical variables: repeat-until-success, and iterative phase estimation with a gate angle computed during the shot ([docs/CONTROL_FLOW.md](../docs/CONTROL_FLOW.md)) |
| [`heterogeneous/`](heterogeneous/) | A model (OpenRouter, a local Ollama / LM Studio server, or offline rules) chooses the algorithm; QUMO runs it partly on a GPU, partly on a CPU and partly on a QPU (local noisy emulator by default) ([README](heterogeneous/README.md)) |
| [`hardware/`](hardware/) | Running QUMO-compiled circuits on quantum hardware providers through the `qumo_hardware` plugins (dry-run by default) |
| [`colab/`](colab/) | Benchmark QUMO on cloud GPUs: Google Colab notebook (NVIDIA), ROCm notebook (AMD), headless script for any GPU host or a Mac, result comparison |
| [`origins/`](origins/) | Not examples: per-framework notes (original, URL, what is checked, reference numbers and their provenance) and the optional `reference/` scripts that rerun an original with its own SDK |

Verify any of them: `bench/.venv/bin/python examples/run_examples.py --only examples/<category>`
(`--tier fast|slow|all`, `--framework NAME`, `--gpu`, `--list`; see [`run_examples.py`](run_examples.py) `--help`).

## Index

<!-- BEGIN EXAMPLES INDEX (generated by examples/run_examples.py --write-index; do not edit) -->

Every example below is run and verified by `bench/.venv/bin/python examples/run_examples.py`
(`--tier fast` by default; `--tier all` adds the slow tier). Status lines are `CHECK <label>: PASS|FAIL ...`.

| Folder | Examples | Origins | Fast / slow |
|---|---|---|---|
| [`foundations/`](foundations/) | 15 | Amazon Braket, Cirq (Google), Classiq, QUMO, Qiskit (IBM), TKET (Quantinuum) | 15 / 0 |
| [`algorithms/`](algorithms/) | 24 | Cirq (Google), Classiq, Q# / Azure Quantum, QUMO, Qibo, Qiskit (IBM), TKET (Quantinuum) | 24 / 0 |
| [`optimization/`](optimization/) | 9 | CUDA-Q (NVIDIA), Classiq, QUMO, Qiskit (IBM) | 9 / 0 |
| [`chemistry_and_physics/`](chemistry_and_physics/) | 17 | CUDA-Q (NVIDIA), Cirq (Google), Classiq, QUMO, Qibo, Qiskit (IBM), TKET (Quantinuum) | 16 / 1 |
| [`qml/`](qml/) | 14 | PennyLane (Xanadu), QUMO, Qibo | 14 / 0 |
| [`error_correction/`](error_correction/) | 3 | Cirq (Google), QUMO | 3 / 0 |
| [`ft/`](ft/) | 6 | Guppy FT (Quantinuum) | 5 / 1 |
| [`noise_and_mitigation/`](noise_and_mitigation/) | 8 | Amazon Braket, CUDA-Q (NVIDIA), Cirq (Google), Classiq, QUMO, Qiskit (IBM) | 7 / 1 |
| [`compilation/`](compilation/) | 6 | Q# / Azure Quantum, QUMO, Qiskit (IBM), TKET (Quantinuum) | 6 / 0 |
| [`simulation/`](simulation/) | 4 | Amazon Braket, CUDA-Q (NVIDIA), Q# / Azure Quantum | 3 / 1 |
| [`control_flow/`](control_flow/) | 2 | QUMO | 2 / 0 |
| [`heterogeneous/`](heterogeneous/) | 6 | QUMO | 6 / 0 |

Every example with its origin, its original and what it reproduces: [`CATALOG.md`](CATALOG.md).

<!-- END EXAMPLES INDEX -->
