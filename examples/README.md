# QUMO examples

Run everything from the repository root after `make setup` (Mojo toolchain in `.venv`). Mojo examples:
`.venv/bin/mojo run -I . examples/<file>.mojo`. OpenQASM files: `build/qumo run examples/<file>.qasm`
(after `make cli`).

## Algorithms and compilation (Mojo)

| File | What it shows |
|---|---|
| [`compile_demo.mojo`](compile_demo.mojo) | QFT and Clifford+T circuits compiled for several hardware modalities |
| [`grover.mojo`](grover.mojo) | Grover search: success probability vs problem size, exact and sampled |
| [`qpe.mojo`](qpe.mojo) | Quantum phase estimation built from the IR functors `power()` and `controlled()` |
| [`qaoa_maxcut.mojo`](qaoa_maxcut.mojo) | QAOA for MaxCut: approximation ratio vs depth, with a brute-force reference |
| [`vqe_h2.mojo`](vqe_h2.mojo) | VQE for H2 (STO-3G): optimisers and gradient methods vs exact diagonalisation |
| [`gradient_bench.mojo`](gradient_bench.mojo) | Adjoint differentiation vs parameter shift: time for one full gradient |

## Error correction and characterisation (Mojo)

| File | What it shows |
|---|---|
| [`qec_memory_experiment.mojo`](qec_memory_experiment.mojo) | Surface-code memory experiment end to end (sampling, detector error model, matching) |
| [`qec_custom_decoder.mojo`](qec_custom_decoder.mojo) | Plugging your own decoder into the parallel sample+decode loop |
| [`qec_characterization.mojo`](qec_characterization.mojo) | Calibration -> noise model -> RB, T1/T2, readout mitigation, QEC under that noise |

## OpenQASM inputs

| File | What it shows |
|---|---|
| [`bell.qasm`](bell.qasm), [`bell_qasm2.qasm`](bell_qasm2.qasm) | Bell pair in OpenQASM 3 and 2 |
| [`ghz.qasm`](ghz.qasm) | n-qubit GHZ state with a constant `for` loop |
| [`teleport.qasm`](teleport.qasm) | Teleportation with mid-circuit measurement and feed-forward |
| [`noisy_bell.qasm`](noisy_bell.qasm) | QUMO noise / detector extensions carried in pragmas |
| [`vqe_ansatz.qasm`](vqe_ansatz.qasm) | Hardware-efficient ansatz with symbolic parameters (`qumo expect ... --params`) |

## Folders

| Folder | What it holds |
|---|---|
| [`colab/`](colab/) | Benchmark QUMO on cloud GPUs: Google Colab notebook (NVIDIA), ROCm notebook (AMD), headless script for any GPU host or a Mac, result comparison |
| [`hardware/`](hardware/) | Running QUMO-compiled circuits on quantum hardware providers through the `qumo_hardware` plugins (dry-run by default) |
