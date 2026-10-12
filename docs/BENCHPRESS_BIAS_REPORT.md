# Is Benchpress biased toward Qiskit/IBM? A technical audit

> Audited object: [Qiskit/benchpress](https://github.com/Qiskit/benchpress) @ `b695f30` (2026-07-29) and its published results (`previous_results` branch).
> Associated paper: Nation, Saki, Brandhofer, Bello, Garion, Treinish, Javadi-Abhari, *"Benchmarking the performance of quantum computing software for quantum circuit creation, manipulation, and compilation"*, Nat. Comput. Sci. (2025).
> Method: full reading of the harness code (workouts, utilities, gyms), per-SDK test counts, re-analysis of the published JSON results, and local execution.
> Date: 2026-10-12.

## Short verdict

Benchpress is **open, reproducible and technically serious**, and many of its circuits come from community collections (QASMBench, Feynman, HamLib). But **its design is structurally tilted toward Qiskit and IBM hardware**: the default target is an IBM device; the gate set and the topology set match IBM's; the harness itself depends on Qiskit; several inputs are handed to Qiskit as abstract objects that only Qiskit knows how to synthesise; and the metrics (count of IBM-type two-qubit gates and wall-clock time) leave out correctness, fidelity and non-superconducting modalities. None of these is an isolated "trick"; each is a reasonable decision from IBM's point of view. Together they turn the benchmark into a measure of **"how well an SDK compiles for IBM hardware through Qiskit's interfaces"**, not of quantum software quality in general.

The published data, re-analysed, and a neutral re-run (§2b) confirm it: on an **all-to-all topology, Qiskit 2.5 and TKET 2.18 tie on two-qubit gate count** (geometric mean TKET/Qiskit = 0.98–1.03, most tests tied). Qiskit's advantage concentrates on **heavy-hex** (IBM's proprietary topology: 1.23–1.54×) and on the **Hamiltonian tests**, where each SDK receives a different input circuit (1.21–1.77×).

## 1. Findings (with evidence)

| # | Finding | Evidence in the code | Effect |
|---|---|---|---|
| 1 | **Default target = an IBM device** | `default.conf`: `backend_name = 'fake_torino'` (IBM Heron, 133 q, heavy-hex) | All "Device" tests (~150) measure compilation for an IBM chip with IBM calibration data |
| 2 | **IBM gate set** | `default.conf`: `basis_gates = ['id','sx','x','rz','cz']` | Native bases of trapped ions (RXX/MS, RZZ), Google (√iSWAP, CZ+PhasedXZ) or neutral atoms are never evaluated |
| 3 | **Only IBM two-qubit gates are valid** | `config.py`: `POSSIBLE_2Q_GATES = set(["cx","cz","ecr"])`; `FlexibleBackend.two_q_gate_type` raises if none is present | A trapped-ion (MS/RXX/RZZ) or iSWAP target cannot be configured without modifying the harness |
| 4 | **Proprietary topology included, others absent** | `abstract_topologies = ['all-to-all','square','heavy-hex','linear']` | heavy-hex is IBM-only; no diagonal grid (Sycamore/Willow), no neutral-atom connectivity (radius > 1, reconfigurable), no ion ring with shuttling |
| 5 | **The harness depends on Qiskit** | `workouts/abstract_transpile/hamlib_hamiltonians.py` imports `qiskit.quantum_info.SparsePauliOp`; `FlexibleBackend(GenericBackendV2)` and `QasmBackendConfiguration` from `qiskit_ibm_runtime` | Every gym needs Qiskit installed; other SDKs' backends are derived from Qiskit objects |
| 6 | **Other SDKs compile "through" IBM** | `tket_gym/utils/tket_backend_utils.py`: `TketFakeIBMQBackend(IBMQBackend)`, `_extend_ibm_fake_backend` | TKET runs its IBM-specific pass (`IBMQBackend.default_compilation_pass`) instead of its best generic pipeline |
| 7 | **Different inputs per SDK** | Hamiltonians: Qiskit receives an abstract `PauliEvolutionGate` (`qiskit_gym/utils/io.py`); TKET receives `gen_term_sequence_circuit`. QV: Qiskit receives `QuantumVolume` with unsynthesised `UnitaryGate`s; TKET receives `tket_QV` | Compares Qiskit's "own synthesis + compilation" with everyone else's "compilation of an already-synthesised circuit"; quality differences mix two things |
| 8 | **"Construction" measures lazy objects** | `test_QV100_build` (`quantum_volume` creates 4×4 `UnitaryGate`s without decomposing them, 1.3 ms) and `test_multi_control_circuit` (lazy `.control()`, 0.36 ms) | An SDK that synthesises at build time (real work) looks "slower" |
| 9 | **Tests named after Qiskit APIs** | `test_DTC100_twirling` (= `pauli_twirl_2q_gates`), `test_bigint_qasm2_import` (a Qiskit parser feature), `test_random_clifford_decompose` (`Clifford(...).to_circuit()`) | Measures feature parity with Qiskit, not general functionality |
| 10 | **SKIPPED = "lacks the functionality"** — but the harness never implemented it | **Cirq and Braket gyms: 0 of ~1000 transpilation tests implemented** (`abstract_transpile/test_qasmbench.py` = `pass`); `Configuration.backend()` raises for them | Cirq has `optimize_for_target_gateset` + `RouteCQC`, yet appears unable to compile |
| 11 | **Metric = IBM-type 2Q count + wall time** | `output_circuit_properties` records `gate_count_2q`, `depth_2q` | No estimated fidelity, no duration, no 1Q count, no memory by default |
| 12 | **No correctness verification** | `qiskit_circuit_validation` only checks that gates are in the basis and edges in the coupling map | An incorrect compiler that respects basis and topology "passes" |
| 13 | **Non-equivalent optimisation levels** | Qiskit O2, TKET O2, BQSKit O1, Staq O2 | "Level 2" means different effort; the chosen point is Qiskit's default |
| 14 | **Wall time, not CPU time** | pytest-benchmark (own fork), `--benchmark-min-rounds=1` | Multithreaded Rust passes in Qiskit vs. single-threaded SDKs: valid as latency, not as efficiency/energy |
| 15 | **Authorship and publication** | All paper authors are IBM. Commits: 244 of ~323 by `nonhermitian` (IBM); top contributors IBM. `previous_results`: 10 Qiskit versions, 2 TKET, 1 pyqpanda | Evaluation by an interested party, no independent replication published |

### What Benchpress does well
- Open source (Apache-2.0), accepts PRs, publishes raw JSON.
- Community circuits (QASMBench, Feynman, HamLib) and generic topologies (all-to-all, square, linear).
- Validates basis and topology; reports failures and timeouts explicitly.
- Runs everything on one controlled machine.

## 2. Quantification with the published data (Qiskit 2.5.0rc1 vs. TKET 2.18.0, same AMD Ryzen 9 7900 machine)

Re-analysis of `previous_results` (560 common tests with a 2Q count). Ratio TKET/Qiskit (> 1 ⇒ Qiskit better):

| Category | n | Time TKET/Qiskit | 2Q TKET/Qiskit | Qiskit better | TKET better | Tie |
|---|---:|---:|---:|---:|---:|---:|
| QASMBench small, all-to-all | 42 | 55× | **0.990** | 5 | 6 | 31 |
| QASMBench medium, all-to-all | 21 | 79× | **0.984** | 1 | 5 | 15 |
| QASMBench large, all-to-all | 53 | 63× | **1.033** | 4 | 5 | 44 |
| QASMBench small, heavy-hex | 42 | 47× | 1.037 | 12 | 8 | 22 |
| QASMBench medium, heavy-hex | 21 | 114× | 1.234 | 14 | 7 | 0 |
| QASMBench large, heavy-hex | 51 | 97× | **1.537** | 34 | 17 | 0 |
| Hamiltonians, all-to-all | 100 | 163× | 1.212 | 57 | 8 | 35 |
| Hamiltonians, heavy-hex | 97 | 246× | **1.643** | 81 | 10 | 6 |
| HamLib on device (fake_torino) | 78 | 193× | **1.767** | 67 | 1 | 10 |
| Feynman on device | 48 | 71× | 1.100 | 27 | 11 | 10 |

Reading:
1. **Quality**: where the topology is neutral (all-to-all) and the input is the same QASM file, both compilers are equivalent. The gap appears on heavy-hex and on non-equivalent inputs (Hamiltonians).
2. **Speed**: Qiskit's time advantage (Rust core, multithreaded passes) is real and large; here Benchpress measures something legitimate. Still, TKET runs through the IBM adapter and its timing includes a Python-level `circuit.copy()`.
3. TKET records 26 failures (timeouts on `cc_n301`, `multiplier_n400`, HamLib JW-22…): a binary penalty that does not weigh the quality of what does compile.

Raw numbers: `bench/results/benchpress_published_qiskit_vs_tket.txt` (regenerable with `bench/analysis/published_results.py`).

## 2b. Cross-check: the same two compilers under neutral conditions

The neutral harness (§3) re-ran Qiskit 2.5.2 and TKET 2.18.5 on 64 community circuits × 11 targets covering superconducting, trapped-ion, neutral-atom and abstract topologies, with identical QASM inputs, TKET configured with its best generic pipeline (not the IBM adapter), and every output verified for correctness (results in [NEUTRAL_BENCHMARK_REPORT.md](NEUTRAL_BENCHMARK_REPORT.md) §3):

| Metric | Benchpress (published) | Neutral harness |
|---|---:|---:|
| 2Q ratio TKET/Qiskit, all common tests | 1.301 | **1.010** |
| 2Q ratio TKET/Qiskit, heavy-hex | 1.04–1.64 | 1.043 |
| 2Q ratio TKET/Qiskit, all-to-all (ions / atoms) | 0.98–1.03 (all-to-all) | **0.947 / 0.937** |
| Time ratio TKET/Qiskit | 116× | 46× |

Interpretation: the **quality** gap Benchpress reports between Qiskit and TKET mostly disappears when the benchmark is neutral (TKET is ahead on ion and neutral-atom targets, Qiskit ahead by 3–8 % on sparse routed topologies). The **speed** advantage of Qiskit is genuine. Benchpress therefore does not invent a Qiskit win, but its design choices magnify it and present an IBM-specific measurement as a general one.

## 3. What an impartial benchmark needs (implemented in `bench/neutral/`)

1. **Multi-modality targets with their native gates**, equally weighted:
   superconducting heavy-hex (CZ), superconducting grid (CZ), trapped-ion all-to-all (RXX/MS + RZ/RY), neutral-atom grid with interaction radius 2 (CZ), and abstract linear chain (CX).
2. **Same input for everyone**: the same OpenQASM 2 file, already synthesised, read by each SDK's own parser. No SDK-specific lazy objects.
3. **Mandatory correctness verification**: state equivalence (with each SDK's reported layout permutations) for circuits up to 16 qubits.
4. **Multiple metrics**: 2Q count, 2Q depth, total depth, 1Q count, **estimated success probability (ESP)** with a per-modality noise model, wall time **and** CPU time.
5. **Comparable levels**: each SDK at its default *and* its maximum setting; report the time/quality Pareto front.
6. **Honest aggregation**: geometric means per modality, win/tie/loss counts, no single headline number.
7. **SKIPPED only when the SDK really lacks the capability** (documented), never because the harness did not implement it.

Results of this methodology (QUMO, Qiskit, TKET, Cirq) are in [`NEUTRAL_BENCHMARK_REPORT.md`](NEUTRAL_BENCHMARK_REPORT.md); QUMO inside Benchpress (original configuration) is in [`BENCHPRESS_RESULTS.md`](BENCHPRESS_RESULTS.md).

## 4. Recommendations to the Benchpress maintainers

1. Make the device and topology set modality-neutral by default (add ion, neutral-atom and diagonal-grid targets; allow RXX/RZZ/iSWAP families in `POSSIBLE_2Q_GATES`).
2. Feed every SDK the same pre-synthesised input in transpilation tests; benchmark synthesis (e.g. Pauli evolution) separately.
3. Add a correctness check (state or unitary equivalence with layouts) to `circuit_validator`.
4. Report ESP/duration from the backend's own error data next to 2Q counts, and CPU time next to wall time.
5. Implement the missing Cirq/Braket transpilation gyms or mark those tests "not implemented by the harness" instead of SKIPPED.
6. Remove the Qiskit import from the workouts and the Qiskit-derived backends from other gyms (plain coupling-map/basis data suffices, as the QUMO gym shows).
7. Invite each SDK's maintainers to supply their recommended pipeline per optimisation level.
