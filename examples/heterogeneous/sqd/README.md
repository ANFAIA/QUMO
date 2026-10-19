# SQD / SKQD: QPU sampling, CPU configuration recovery, HPC diagonalisation

`sqd_hubbard.py` runs sample-based Krylov quantum diagonalisation of an
8-site Fermi-Hubbard chain (16 qubits) the way IBM and RIKEN split SQD between
a QPU and a supercomputer, with every device emulated locally.

```bash
make py                                     # or make py-gpu for the GPU twin
PYTHONPATH=build:plugins/hardware bench/.venv/bin/python examples/heterogeneous/sqd/sqd_hubbard.py
bench/.venv/bin/python examples/heterogeneous/ai_gpu_qpu.py --offline --problem examples/heterogeneous/problems/hubbard_sqd.json
```

## Pipeline: stage, device, why

| Stage | Device | Why this device (what the others cannot do) |
|---|---|---|
| Krylov sampling | QPU: `local:qumo-sim:grid:2x8/sc_cz+noise:superconducting` (emulated) | Prepares and measures U(dt)^k \|Neel>, the only stage that touches the quantum state. The hopping and on-site terms form a 2x8 ladder, so the grid needs no SWAPs (44 / 88 / 132 CZ for k = 1, 2, 3). Noise breaks particle number and spin. |
| Configuration recovery | CPU (Python, one core) | Per-sample symmetry repair (Robledo-Moreno et al. 2024): flips bits of wrong-sector samples with probabilities from the previous eigenvector's orbital occupations. Branchy, sequential, tiny arithmetic. |
| Projected-H build + eigensolver | CPU, all cores (`qumo_py.subspace_ground_state`, Mojo) | Product (CI-string) basis of the sampled alpha/beta strings, ELL matrix built in parallel, restarted Lanczos; returns the occupations for the next round. IBM + RIKEN ran this stage on Fugaku's CPU nodes. QUMO has no GPU eigensolver yet. |
| Ideal twin (optional) | GPU state vector (`sample_device(..., "gpu")`, fp32), CPU fallback | Noiseless samples of the same circuits, to separate the cost of noise from the cost of the subspace cap. Verification only: at scale no classical twin exists. |
| Reference | CPU (QUMO + SciPy) | Exact sector energy for the checks (possible because the sector has 4900 determinants). |

QUMO's orchestrator plans the circuit jobs (`qumo_py.schedule`): the Krylov
jobs are pinned to the QPU device, the twin jobs are placed by HEFT
(GPU in its cost model).

## Measured (Apple M5, 10 cores, 2026-10-17, `build/gpu/qumo_py.so`, seed 7)

| Quantity | Value |
|---|---|
| Exact sector energy (QUMO / SciPy) | -12.235806999 (agree to 1e-14) |
| Raw QPU samples in the (4, 4) sector | 67.95 % |
| Subspace per batch | 1681 - 2401 determinants (<= 49 % of the sector) |
| Final SQD energy, relative error | -11.9004, 2.74e-2 (band 5e-2) |
| Postselection only, same budget | 0.4 % lower energy than recovery here (within the 1 % slack): at this noise level enough in-sector samples survive, so recovery does not help yet |
| Ideal twin (GPU, fp32) | -11.9505 |
| Wall time | 22 s (20 s of it in the noisy QPU emulator); 30-38 s with the CPU build |

## Checks

HPC solver = SciPy on the full sector (1e-8); QPU samples are noisy
(in-sector fraction <= 0.95); recovery puts every shot in the sector; every
SQD energy is variational; final energy within `--band` (5 %); largest
subspace below `--max-subspace-fraction` (0.6) of the sector; recovery not
worse than postselection (1 % slack); twin energy variational; no remote
submission.

## Limits (honest)

* The QPU is QUMO's vendor-neutral depolarizing emulator (superconducting
  preset), not a device calibration. `--qpu ibm:fake_torino` uses an IBM
  snapshot with Aer noise; remote backends need `--allow-remote` and only
  dry-run unless `--submit --yes --max-cost`.
* The sector is small enough to diagonalise exactly; the subspace cap
  (`--samples-per-batch`) emulates the regime where it is not.
* The recovery weighting follows the paper's modified ReLU; the subsampling
  and spin symmetrisation follow `qiskit-addon-sqd`'s documented behaviour,
  re-implemented (the package is not a dependency).

Sources: [qiskit-addon-sqd](https://github.com/Qiskit/qiskit-addon-sqd);
Robledo-Moreno et al., [arXiv:2405.05068](https://arxiv.org/abs/2405.05068)
(IBM + RIKEN); Yu et al., SKQD, [arXiv:2501.09702](https://arxiv.org/abs/2501.09702).
