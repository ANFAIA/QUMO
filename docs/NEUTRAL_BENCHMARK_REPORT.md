# Hardware-neutral compiler benchmark — QUMO, Qiskit, TKET, Cirq

> Run date: 2026-10-12, one machine (Apple Silicon, 10 cores, 32 GB), runs sequential per SDK.
> SDKs: QUMO 0.1.0 · Qiskit 2.5.2 · pytket 2.18.5 · Cirq 1.7.0.
> Harness: `bench/neutral/` (methodology from [BENCHPRESS_BIAS_REPORT.md](BENCHPRESS_BIAS_REPORT.md) §3). Raw records: `bench/results/neutral.jsonl` (5632 records); aggregates: `bench/results/neutral_{default,max,pairwise,third_party}.md`.
> Reproduce: `make py && bench/.venv/bin/python bench/neutral/run_neutral.py --circuits bench/neutral/circuits --out results.jsonl && bench/.venv/bin/python bench/neutral/report.py results.jsonl`.

## 1. Design

| Principle | Implementation |
|---|---|
| Every modality weighs the same | 11 targets: 4 superconducting (heavy-hex, square grid, **diagonal grid** à la Sycamore/Willow, **octagonal** à la Aspen; CZ), 2 trapped-ion (**all-to-all with RXX/MS**, **QCCD zone ring with RZZ**, shuttling = swaps), 2 neutral-atom (**grid with interaction radius 2**, **reconfigurable/all-to-all**; CZ), 3 abstract (linear, binary tree, torus; CX). Bold = not covered by Benchpress. |
| Same input for everyone | 102 community circuits (QASMBench small/medium + Feynman) normalised once to the same OpenQASM 2 file per circuit (no measurements, common qelib1 gate set); each SDK parses that file. This report uses the 64 circuits with ≤ 14 qubits so every output can be verified. |
| Correctness is mandatory | Each output is simulated from a random product input state with the SDK's reported initial/final layouts (implicit permutations included) and compared to the ideal state: **4970 / 4970 successful compilations verified** (fidelity ≥ 1 − 1e-6). |
| Native-gate validation | 2-qubit gates must be the target's native gate on a coupling edge; 1-qubit gates may be spelled any way (counted as *runs*, see metrics). |
| Several metrics | 2Q count, 2Q depth, depth, 1Q runs, estimated success probability (ESP: per-modality gate errors, durations, T1/T2; virtual Z on superconducting targets), wall and CPU time. |
| Comparable effort | Each SDK at its documented default ("default") and its strongest general setting ("max"): Qiskit O2/O3; TKET `FullPeepholeOptimise` + `DefaultMappingPass` + `SynthesiseTket` (+ `CliffordSimp`, `KAKDecomposition`, post-routing KAK for max; TK2/ZZPhase path for Ising-type gates); Cirq `optimize_for_target_gateset(CZTargetGateset)` + `RouteCQC` (+ `eject_z`/`eject_phased_paulis` for max); QUMO O2/O3. |
| Honest SKIPs | "unsupported" only when the SDK has no built-in path (Cirq has no CX, RXX or RZZ target gateset). Errors and timeouts (120 s) are reported, not hidden. |
| Fair process model | Persistent per-SDK worker processes started with `spawn` and warmed up once (Qiskit's Rust thread pool deadlocks in forked children; first-call plugin loading would otherwise be timed). |

## 2. Status (64 circuits × 11 targets × 2 levels = 1408 jobs per SDK)

| SDK | ok (verified) | unsupported | timeout > 120 s | error | incorrect | invalid |
|---|---:|---:|---:|---:|---:|---:|
| QUMO | 1408 | 0 | 0 | 0 | 0 | 0 |
| Qiskit | 1408 | 0 | 0 | 0 | 0 | 0 |
| TKET | 1396 | 0 | 12 | 0 | 0 | 0 |
| Cirq | 758 | 640 (no CX/RXX/RZZ gateset) | 0 | 10 (`RouteCQC` IndexError) | 0 | 0 |

## 3. Results without QUMO (is Qiskit's Benchpress lead real?)

Pairwise on jobs both SDKs completed (ratio X/Qiskit, > 1 ⇒ Qiskit better):

| Target | TKET 2Q ratio | TKET time ratio | Cirq 2Q ratio | Cirq time ratio |
|---|---:|---:|---:|---:|
| superconducting heavy-hex | 1.043 | 47.5 | 1.153 | 12.3 |
| superconducting grid | 1.026 | 51.7 | 1.117 | 12.5 |
| superconducting diagonal grid | 1.012 | 48.7 | 1.203 | 12.3 |
| superconducting octagonal | 1.036 | 48.4 | 1.141 | 12.7 |
| trapped ion all-to-all (RXX) | **0.947** | 43.5 | – | – |
| trapped ion zone ring (RZZ) | 1.033 | 45.0 | – | – |
| neutral atom radius-2 grid | **0.951** | 35.9 | 1.189 | 10.7 |
| neutral atom reconfigurable | **0.937** | 45.2 | 1.144 | 14.2 |
| linear chain | 1.078 | 44.9 | – | – |
| binary tree | 1.061 | 49.0 | – | – |
| torus | 1.001 | 53.5 | – | – |
| **all** | **1.010** | 46.4 | **1.157** | 12.4 |

* With identical inputs and a hardware-neutral target mix, **TKET and Qiskit are essentially tied on circuit quality** (TKET better on all-to-all ion and atom targets, Qiskit better by 3–8 % on routed sparse topologies). In Benchpress' published data the same two SDKs differ by 1.30× overall and up to 1.77× on HamLib-on-device — most of that gap comes from the benchmark's design (heavy-hex focus, different inputs per SDK, TKET run through its IBM adapter), not from compiler quality.
* **Qiskit's speed lead is real and large** (≈ 46× vs TKET, ≈ 12× vs Cirq here).
* Cirq trails on quality (+16 %) and lacks built-in targets for CX and Ising-type gates.

## 4. Results with QUMO

### 4.1 Pairwise (ratio QUMO/X; < 1 ⇒ QUMO better)

| vs | jobs | 2Q ratio | time ratio | QUMO fewer 2Q | tie | X fewer 2Q |
|---|---:|---:|---:|---:|---:|---:|
| Qiskit | 1408 | **0.928** | 0.201 | 606 | 731 | 71 |
| TKET | 1396 | **0.919** | 0.004 | 666 | 550 | 180 |
| Cirq | 758 | **0.803** | 0.016 | 496 | 255 | 7 |

By modality vs Qiskit (default level): superconducting 0.926, trapped ion 0.924, neutral atom 0.944, abstract 0.939 (QUMO 5–11× faster).
By modality vs TKET: TKET ties QUMO on neutral-atom targets (2Q ratio 1.000 default / 0.998 max) and produces fewer 2Q gates than QUMO in 14–34 jobs per modality and level; QUMO leads on the other modalities (0.87–0.94).

### 4.2 Versus the best SDK per (circuit, target, level)

| SDK | 2Q ratio to best (default) | 2Q ratio to best (max) | strict wins (default / max) | median compile time (default / max) |
|---|---:|---:|---:|---|
| QUMO | **1.047** | **1.047** | **268 / 269** | 0.8 / 1.9 ms |
| Qiskit | 1.123 | 1.133 | 38 / 24 | 5.8 / 5.9 ms |
| TKET | 1.161 | 1.120 | 84 / 94 | 208 / 398 ms |
| Cirq | 1.299 | 1.308 | 3 / 3 | 97 / 109 ms |

Per-target tables (2Q, ESP and time ratios, wins, ties, statuses): `bench/results/neutral_default.md`, `bench/results/neutral_max.md`.

### 4.3 ESP
QUMO has the highest estimated success probability on every target at both levels (geometric-mean ESP ratio to best 0.955 default / 0.963 max vs Qiskit 0.517 / 0.487, TKET 0.413 / 0.487). ESP compounds 2Q count, 1Q runs and duration exponentially, so its ratios amplify small gate-count differences; on the "abstract" targets the pessimistic generic noise model drives ESP towards 0 and those ratios are indicative only.

## 5. Caveats (read before quoting any number)

1. **Conflict of interest.** This harness and QUMO were written by the same project. The methodology is designed to be neutral and is fully open, but it has not been reviewed by the other SDKs' authors.
2. **Tuning on the benchmark.** QUMO's compiler was improved during the project using `qb_small` circuits on three of these targets (linear, heavy-hex, ion all-to-all) and Benchpress' HamLib set as development cases. Its advantage on these exact circuits likely overstates its advantage on unseen workloads.
3. **Third-party configurations are ours.** TKET and Cirq pipelines were configured from their documentation to be strong and fair (and two harness bugs that penalised TKET were found and fixed: implicit wire permutations and an invalid post-routing pass). Experts in those SDKs may obtain better results.
4. **Scale.** Verification limits this report to ≤ 14 logical qubits; 100+-qubit behaviour is covered by Benchpress ([BENCHPRESS_RESULTS.md](BENCHPRESS_RESULTS.md)), where Qiskit remains faster than QUMO on the largest dense circuits.
5. **Timing.** QUMO is called in-process through a CPython extension; Qiskit uses its default Rust parallelism; TKET and Cirq timings were re-measured while unrelated unit tests ran on the machine (their gaps of 10–300× dwarf that noise). QUMO and Qiskit timings come from quiet runs.
6. **ESP model** is a simple, uniform per-modality model (no crosstalk, no calibration variance).
