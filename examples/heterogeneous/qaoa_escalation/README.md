# Warm-started QAOA with QPU escalation

`ws_qaoa_escalation.py` solves a 12-node weighted MaxCut with warm-started
QAOA (Egger, Marecek, Woerner 2021): a classical warm start on the CPU, exact
adjoint training on the GPU, many cheap shots on a superconducting emulator,
then a few high-fidelity shots on a trapped-ion emulator to verify the best
candidates, and a final local search.

```bash
PYTHONPATH=build/gpu:plugins/hardware bench/.venv/bin/python examples/heterogeneous/qaoa_escalation/ws_qaoa_escalation.py
bench/.venv/bin/python examples/heterogeneous/ai_gpu_qpu.py --offline --problem examples/heterogeneous/problems/maxcut_ws_escalation.json
```

## Pipeline: stage, device, why

| Stage | Device | Why this device |
|---|---|---|
| Goemans-Williamson warm start | CPU (NumPy) | MaxCut SDP by a Burer-Monteiro factorisation (rank 6), best of 10 random hyperplanes: an upper bound and a good cut (21.201 of 21.739 here). |
| Exact training | GPU: `qumo_py.value_and_gradient(device="gpu")`, fp32 (CPU fp64 fallback) | Exact energy and adjoint gradient of the full state per optimiser step: no shot noise, no QPU time. GPU adjoint via a doubled register (see docs/GPU.md). |
| Sampling | QPU: `local:qumo-sim:grid:3x4/sc_cz+noise:superconducting` (emulated) | 4000 cheap shots of the trained circuit; the grid needs SWAPs (85 CZ). |
| Verification | QPU: `local:qumo-sim:all:16/ion_zz+noise:trapped_ion` (emulated) | 400 high-fidelity shots (29 ZZ, no SWAPs): confirms the superconducting top-k candidates and estimates <cut> closer to the ideal value. |
| Local search | CPU | One-flip polish of the confirmed candidates. |

## Measured (Apple M5, 2026-10-17, seed 7)

| Quantity | Value |
|---|---|
| Max cut (brute force) / SDP bound / GW cut | 21.739 / 22.308 / 21.201 |
| Ideal approximation ratio, p = 1 | WS-QAOA 0.878, standard QAOA 0.810, warm-start state 0.742 |
| <cut>: ideal / superconducting / ion | 19.090 / 18.517 +- 0.037 / 18.998 +- 0.103 |
| Candidates | 6 of the 8 best superconducting samples confirmed by the ion shots; final cut 21.739 = optimum |
| Training, 12 qubits | CPU fp64 0.21 s (96 calls); GPU fp32 1.03 s (105 calls) - below the GPU threshold the CPU wins |
| One value + gradient (p = 2, 3-regular-like graphs) | 16 qubits: CPU 33 ms, GPU 19 ms; 20 qubits: 224 vs 73 ms; 22 qubits: 1217 vs 232 ms |
| Wall time | 4 s (GPU training) |

## Checks

SDP bound >= max cut; GW rounding >= 0.878 x SDP; GPU gradient = CPU fp64
gradient (1e-4); WS-QAOA ratio >= standard QAOA and >= the warm-start
state; best superconducting sample >= GW cut; ion <cut> no further from the
ideal than the superconducting one (3 sigma slack); final cut = brute-force
optimum; no remote submission.

## Limits

The graph is small enough to brute-force (the check needs it); both QPUs
are QUMO's modality-preset emulators; "verification" is statistical (the
ion shots re-sample the same circuit), MaxCut values themselves are checked
classically. `--gpu-device gpu` (default) forces the GPU below its threshold
to exercise it; `auto` applies the threshold.

Sources: Egger, Marecek, Woerner, [Quantum 5, 479 (2021)](https://quantum-journal.org/papers/q-2021-06-17-479/)
([arXiv:2009.10095](https://arxiv.org/abs/2009.10095)); Goemans & Williamson,
[J. ACM 42 (1995)](https://dl.acm.org/doi/10.1145/227683.227684).
