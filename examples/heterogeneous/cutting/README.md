# Circuit cutting across two QPU modalities

`cut_two_modalities.py` cuts an 18-qubit circuit that no local QPU holds
into two fragments, runs the dense one on a trapped-ion emulator and the wide
one on a superconducting-grid emulator, each compiled by QUMO for its own
target, and reconstructs the expectation value by quasi-probability
contraction.

```bash
PYTHONPATH=build:plugins/hardware bench/.venv/bin/python examples/heterogeneous/cutting/cut_two_modalities.py
bench/.venv/bin/python examples/heterogeneous/ai_gpu_qpu.py --offline --problem examples/heterogeneous/problems/cut_two_modalities.json
```

The circuit: a 6-qubit all-to-all block (15 RZZ) followed by a 13-qubit
two-layer brickwork that shares one qubit with it. The observable is a sum of
six Pauli correlators, most of them across both parts.

## Pipeline: stage, device, why

| Stage | Device | Why this device |
|---|---|---|
| Cut finding | CPU | Enumerates sets of <= 2 wire cuts (87 examined), keeps one-directional two-fragment splits, compiles every fragment for every target with QUMO (`compile_for_target`) and scores assignments by estimated success (compiled gate counts x modality error rates). Picks the fewest cuts, then the best assignment. |
| Dense fragment | QPU: `local:qumo-sim:all:8/ion_zz+noise:trapped_ion` (emulated) | All-to-all native ZZ: the dense block plus two brickwork qubits in 17 ZZ, no SWAPs. On the grid the same fragment would need about 40 CZ. |
| Wide fragment | QPU: `local:qumo-sim:grid:4x4/sc_cz+noise:superconducting` (emulated) | 11 qubits: more than the 8-qubit ion register, nearest-neighbour on the grid (10 CZ). |
| Uncut reference + exact fragment tensors | GPU state vector (fp32; CPU fallback) | The full 18-qubit noiseless state, which no QPU here can run, and all fragment observables in one batched reduction per fragment circuit. |
| fp64 identity check | CPU | The 1e-9 reconstruction identity needs fp64 (the Apple GPU is fp32-only). |
| Reconstruction | CPU (NumPy einsum) | 4^k-term contraction; at k = 1 it is tiny. CutQC's GPU-heavy contraction at many cuts is not implemented on QUMO's GPU. |

With the default seed the finder cuts qubit 7: the wire leaves the grid
fragment (measured in Z / X / Y) and enters the ion fragment (prepared in
0, 1, +, +i). The reverse assignment does not fit (11 > 8 ion qubits).
Fragment jobs are pinned to their QPUs in QUMO's orchestrator; HEFT places
the exact jobs.

## Measured (Apple M5, 2026-10-17, GPU build, seed 7)

| Quantity | Value |
|---|---|
| Exact value (CPU fp64) | 0.728059 |
| GPU fp32 uncut reference, difference | 7.7e-9 |
| Noiseless reconstruction, difference | 3.0e-15 |
| Noisy reconstruction (20 000 shots per circuit, 11 circuits) | 0.7355 +- 0.0078 (error 0.0074) |
| Seeds 7-10: noisy error | 0.006 - 0.043 for values 0.5 - 1.3 |
| Wall time | 2 s |

## Checks

One wire cut; dense fragment on the ion QPU; noiseless reconstruction =
uncut exact value (1e-9); GPU reference and GPU fragment tensors = CPU fp64
(1e-4); noisy reconstruction within 5 sigma + 0.03 x sum|c_t|; no remote
submission.

## Limits

Wire cuts only (no gate cuts); the decomposition is the basic Pauli
identity (Peng et al.), not the optimal-overhead one; both QPUs are QUMO's
modality-preset emulators. `--ion-qpu quantinuum:H2-1LE` and
`--sc-qpu ibm:fake_torino` select vendor emulators; remote targets follow
the usual `--allow-remote --submit --yes --max-cost` gate.

Sources: [qiskit-addon-cutting](https://github.com/Qiskit/qiskit-addon-cutting);
Tang et al., CutQC, [arXiv:2012.02333](https://arxiv.org/abs/2012.02333);
Peng et al., [arXiv:1904.00102](https://arxiv.org/abs/1904.00102).
