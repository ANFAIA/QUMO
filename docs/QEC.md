# QUMO QEC + experimentalist stack

Pure-Mojo replacement for the Stim → PyMatching → Sinter pipeline, plus the
hardware-characterization tooling an experimentalist needs (noise models from
calibration data, randomized benchmarking, T1/Ramsey, readout mitigation).
Everything shares the one QUMO `Circuit` IR (noise channels, `DETECTOR`,
`OBSERVABLE`, measurement records), so there is no Cirq↔Stim↔Qiskit
conversion step anywhere.

```
qumo.qec.codes ──► Circuit (+coords) ──► qumo.qec.dem ──► DetectorErrorModel ──► qumo.qec.decoders
     │                 │                                   (Stim-exact)            MatchingGraph
     │                 ├──► qumo.sim.stabilizer  (tableau: reference sample, exact trajectories)
     │                 └──► qumo.sim.frame       (bit-packed Pauli frames: 512 shots / SIMD op)
     │                                │
     └──────────► qumo.qec.experiments (fused parallel sample+decode, Sinter-style stopping)
qumo.qec.characterization: NoiseModel(T1,T2,gates,readout) → apply_noise_model → any of the above
```

Run the tests / examples / benchmarks:

```bash
.venv/bin/mojo run -I . tests/test_stabilizer.mojo      # tableau vs state vector
.venv/bin/mojo run -I . tests/test_qec.mojo             # whole QEC stack
.venv/bin/mojo run -I . examples/qec_memory_experiment.mojo
.venv/bin/mojo run -I . examples/qec_characterization.mojo
.venv/bin/mojo run -I . examples/qec_custom_decoder.mojo
.venv/bin/mojo run -I . bench/bench_qec.mojo 200000     # throughput + LER grid
bench/.venv/bin/python bench/bench_qec.py 1000000       # cross-check vs Stim + PyMatching
```

## Where each idea comes from

| Piece | Inspiration | QUMO module |
|---|---|---|
| Inverse-tableau stabilizer sim, O(n/64) deterministic measurement, collapse by gates "at the beginning of time" | Stim `TableauSimulator`, Aaronson–Gottesman CHP | `qumo/sim/stabilizer.mojo` |
| Reference sample + bit-packed Pauli frames, Z-gauge randomisation after M/R | Stim `FrameSimulator` | `qumo/sim/frame.mojo` |
| Circuit-level noise parameters (`after_clifford_depolarization`, …) and generated memory circuits, hook-safe N/Z CX order | Stim `Circuit.generated` | `qumo/qec/codes.mojo` |
| Detector error model, backward sensitivity propagation, disjoint→independent channel conversion, `^` decomposition | Stim `ErrorAnalyzer` | `qumo/qec/dem.mojo` |
| Matching graph from DEM, log((1-p)/p) weights, virtual boundary node | PyMatching | `qumo/qec/decoders.mojo` |
| Union-Find decoder with weighted growth + peeling | Delfosse–Nickerson; Huang–Newman–Brown | `UnionFindDecoder` |
| Exact MWPM (APSP table, Edmonds blossom) | PyMatching v0, van Rantwijk's mwmatching | `MwpmDecoder`, `BlossomMatcher` |
| Decoder plug-in interface, batch decode | CUDA-Q QEC `decoder` API | `trait Decoder` |
| Fused sample+decode, `max_errors` early stop, Wilson intervals, CSV/JSON tables | Sinter | `qumo/qec/experiments.mojo` |
| Noise model from calibration (T1/T2/gate/readout), RB, T1/Ramsey, readout mitigation | Qibocal, Cirq experiments, Qiskit Experiments / Aer `NoiseModel.from_backend` | `qumo/qec/characterization.mojo` |
| Stim text export (`QUBIT_COORDS`, `DETECTOR(x,y,t)`, `rec[-k]`) for Crumble/Sinter cross-checks | Stim | `to_stim()` |

## Simulators

### `qumo.sim.stabilizer` — inverse tableau

We store the inverse tableau of the Clifford `C` that prepared the state:
row `X_q` = C† X_q C and row `Z_q` = C† Z_q C, each a bit-packed Pauli string
(`UInt64` words) with a sign bit.

* **Gates** are row operations: T'⁻¹(P) = T⁻¹(G† P G), i.e. at most two
  Pauli-string products per gate (O(n/64), phases tracked with popcounts of the
  XY/YZ/ZX "plus" and "minus" masks). Primitives: X, Y, Z, H, S, S†, √X, √X†,
  CX, CZ, SWAP; CY, iSWAP(†), ECR and RX/RY/RZ/P/RXX/RYY/RZZ at multiples of
  π/2 are lowered to them.
* **Measurement of Z_q** is deterministic iff row Z_q has no X support (an
  O(n/64) scan) and the outcome is its sign bit — after the first QEC round
  every stabilizer measurement takes this fast path.
* **Random outcomes** are collapsed Stim-style: CX gates from a pivot input
  qubit (a no-op on |0⟩) clear the other X support, then H (or H_YZ) on the
  pivot and an optional X fix the outcome; these are column operations.
* `run(circuit)` supports mid-circuit measurement, reset, single-bit and
  register feed-forward, and Pauli noise (per-shot trajectories).
  `reference_sample()` forces random outcomes to 0 (Stim's convention).
* Validated against the state-vector simulator on 300 random Clifford
  circuits (all gates, every Pauli expectation) and 200 post-measurement
  collapse checks; GHZ-1200 with 1200 mid-circuit measurements runs in ~0.5 ms.

### `qumo.sim.frame` — bit-packed Pauli frames

One shot = one bit; a batch is `FW = 8` UInt64 words = **512 shots** and every
frame gate is one `SIMD[uint64, 8]` load/xor/store per operand. Noise lanes
are drawn by geometric skipping over the whole 512-shot batch (O(512·p) work
per channel). Measurement flips are the X component of the frame; Z is
re-randomised after every measurement/reset and at start (Z on a Z eigenstate
is a gauge), which makes non-deterministic measurements correctly random.
Detectors and observables are XORs of record flips (Stim's detection events).
Batches are seeded by `(seed, batch index)` and run in parallel → output is
bit-identical for any thread count. Only Pauli feedback conditioned on one
measured bit is frame-simulable (QM-E704 otherwise).

Outputs: `DetectorSamples` (bit-packed detection events + observable flips,
`fired(shot)`, `detector_fraction(d)`, …) and `MeasurementSamples`.

## Codes (`qumo.qec.codes`)

* `repetition_code_memory(d, rounds, noise)` — Stim layout (data = even qubits).
* `surface_code_memory(d, rounds, noise, basis="Z"|"X")` — rotated surface
  code, d² data + d²−1 measure qubits, X checks in "N" order
  (NE, SE, NW, SW), Z checks in "Z" order (NE, NW, SE, SW): hook errors are
  perpendicular to the logical they could shorten, so the circuit distance
  equals d (`MatchingGraph.graphlike_distance()` checks this in the tests and
  Stim's `shortest_graphlike_error` agrees).
* `NoiseParams(after_clifford_depolarization, before_measure_flip_probability,
  after_reset_flip_probability, before_round_data_depolarization)`;
  `NoiseParams.uniform(p)` = Stim/Sinter "uniform circuit noise".
* `CodeCircuit` carries qubit/detector coordinates, data/X/Z qubit lists;
  `to_stim()` exports Stim text (verified: Stim parses it, finds every
  detector deterministic and the graph-like distance = d).

## Detector error models (`qumo.qec.dem`)

`detector_error_model(circuit, decompose=True)` walks the frame program
**backwards** keeping, per qubit, the sorted set of detectors/observables an
X or Z error at that moment would flip (CX: `sx[c] ^= sx[t]`,
`sz[t] ^= sz[c]`; M adds its record's detectors to `sx`; R clears). Every
noise channel met contributes one mechanism per Pauli component; identical
mechanisms merge with p = p₁(1−p₂) + p₂(1−p₁). DEPOLARIZE1/2 and
PAULI_CHANNEL_1 are converted to independent mechanisms exactly like Stim.
A Z-sensitivity reaching a measurement/reset raises **QM-E703**
(non-deterministic detector).

`decompose=True` splits each mechanism along its channel's single-qubit
X/Z basis errors into graph-like pieces (singles first, else an irreducible
pair + singles), reproducing Stim's `decompose_errors`. Result on the
d = 3, 5 surface codes and the repetition code: **the same mechanisms with
the same probabilities as Stim (max relative difference ~1e-15)**, and the
`MatchingGraph` built from it is edge-for-edge identical to PyMatching's
graph from Stim's DEM. Extraction cost: d=11, 11 rounds (1320 detectors,
24k mechanisms) in ~40 ms.

`to_text()` prints Stim DEM syntax (`error(p) D0 D1 L0`, `^` separators,
`detector(x, y, t) Dk`).

## Decoders (`qumo.qec.decoders`)

```mojo
trait Decoder(Copyable, Deinitable):
    def decode(mut self, fired: List[Int]) -> UInt64: ...   # sorted detector ids -> observable mask
    def name(self) -> String: ...
```

* `MatchingGraph(dem)`: nodes = detectors + boundary; edges from graph-like
  errors and decomposition pieces, weight log((1−p)/p); undecomposed
  hyperedges are decomposed against existing edges or dropped (counted).
  `graphlike_distance()` = fewest graph-like errors forming a logical.
* `UnionFindDecoder`: weighted cluster growth on integer edge lengths (32 per
  nat), event-driven growth step, union by size, boundary vertex grounds
  clusters, peeling on a BFS spanning forest rooted at the boundary.
* `MwpmDecoder`: exact. All-pairs shortest paths (distance + observable
  parity) precomputed once per graph (≤ 3000 nodes, ≤ 8 observables, shared
  by thread copies through an `ArcPointer`), otherwise per-defect Dijkstra.
  Defects are split into independent components (i, j interact only if
  d_ij < b_i + b_j); each component is solved by subset DP (≤ 10 defects) or
  Edmonds' blossom (port of van Rantwijk's O(n³) algorithm) on the reduced
  graph w'(i,j) = min(d_ij, b_i + b_j) plus one boundary vertex for odd
  counts. Limits: per-shot cost is O(k³) in the largest component, so it is
  the right tool for d ≤ 7 at p ≲ 0.5 %; beyond that use UF (PyMatching's
  sparse blossom would be the next step).
* `LookupTableDecoder(dem, max_weight=2)`: most likely observable per
  syndrome over all ≤ max_weight error combinations; ≤ 64 detectors.
* Batch helpers: `decode_samples`, `count_logical_errors` (parallel over
  512-shot batches, one decoder copy per worker).

All three correct every single fault of the d = 3, 5, 7 surface codes
(Z and X memory) — checked in `tests/test_qec.mojo`.

## Experiments (`qumo.qec.experiments`)

```mojo
var r = memory_experiment("surface_z", distance=5, rounds=-1, p=0.002,
                          shots=2_000_000, decoder="mwpm", seed=7, max_errors=200)
print(r)              # p_L, Wilson 95% interval, per-round rate, throughput
print(r.to_json())    # machine-readable for agents
print(threshold_sweep("surface_z", [3, 5, 7], [0.003, 0.006, 0.01], 200_000, "uf"))
```

`sample_and_decode(prog, decoder, shots, seed, max_errors)` is the fused
loop: each worker propagates a 512-shot frame batch, extracts fired detectors
(CSR), decodes, and compares with the sampled observables — detection events
never hit memory in bulk. Shots are processed in waves of 64 batches; the run
stops after the first wave reaching `max_errors` (deterministic). Results:
`ExperimentResult` with `logical_error_rate()`, `std_error()`,
`wilson_interval()`, `per_round_error_rate()`, `csv_row()`, `to_json()`.

## Characterization (`qumo.qec.characterization`)

* `NoiseModel` (per-qubit T1, T2, readout P(1|0)/P(0|1), 1q gate error; 2q
  error; gate/measure/reset durations). `apply_noise_model(circuit, model)`:
  * relaxation during gates and idling (ASAP schedule) as the Pauli twirl of
    amplitude+phase damping: px = py = (1−e^{−t/T1})/4, pz = (1−e^{−t/T2})/2 − px;
  * the remaining gate infidelity e as DEPOLARIZE1(3e/2) / DEPOLARIZE2(5e/4)
    (thermal part subtracted, like Aer's `from_backend`);
  * readout as X_ERROR((p01+p10)/2) before measurement (the asymmetric part
    is handled by mitigation).
* `randomized_benchmarking_1q(lengths, num_sequences, shots, seed,
  depolarizing_per_clifford, model)` — 24-element Clifford group from H/S
  BFS, inverting Clifford, frame-simulated survival, `fit_decay` →
  EPC = (1−p)/2. Test: injected 1 % depolarizing per Clifford → EPC 0.00668
  vs. 0.00667 expected.
* `t1_experiment` / `ramsey_experiment` (+ `t1_circuit`, `ramsey_circuit`)
  and `fit_exponential_decay` (variable projection: log grid + golden section
  on the rate, linear LS for amplitude/offset). Test: T1=50 → 51.4,
  T2=30 → 30.3.
* `ReadoutCalibration.from_counts(...)` from `readout_calibration_circuits(n)`
  and `mitigate_counts(counts, cal)` (tensored inverse, quasi-probabilities),
  `expectation_z(probs, qubits)`.

## Plugging in an AI decoder (AlphaQubit-style)

A neural decoder is just another `Decoder`:

```mojo
struct NeuralDecoder(Decoder):
    var model: MyCompiledModel        # e.g. a MAX graph / ONNX session handle (ArcPointer-shared)
    var coords: List[Float64]         # CodeCircuit.detector_coords: (x, y, t) per detector
    def name(self) -> String: return "neural"
    def decode(mut self, fired: List[Int]) -> UInt64:
        # build the (rounds x stabilizers) event tensor from `fired` + coords,
        # run the network, threshold the logit(s) -> observable mask
        ...
```

* **Training data**: `FrameSimulator.sample_detectors()` produces labelled
  data (detection events + true observable flips) at >10⁷ shots/s with
  deterministic seeds; `CodeCircuit.detector_coords` gives the spatial/
  temporal layout a transformer/recurrent model needs (AlphaQubit consumes
  per-round stabilizer events in exactly this form). Calibrated noise
  (`apply_noise_model`) gives the realistic, non-uniform training
  distribution; `detector_error_model` gives per-edge priors that can be fed
  as extra inputs (as AlphaQubit does with PyMatching-style weights).
* **Inference** runs inside `sample_and_decode` / `decode_samples`, one copy
  per worker thread; share heavy weights through an `ArcPointer` as
  `MwpmDecoder` does with its distance table. For GPU batch inference,
  implement batching in your own loop over `DetectorSamples` (bit-packed,
  detector-major per 512-shot batch) and keep `decode()` for the single-shot
  fallback.
* **Evaluation** is apples-to-apples with MWPM/UF: same circuits, seeds,
  `ExperimentResult` statistics and Sinter-style stopping.
* Soft readout information (analog IQ) is not modelled yet; the natural
  extension is a per-measurement confidence array next to `MeasurementSamples`.

## Error codes (QM-E7xx)

| Code | Meaning | Hint |
|---|---|---|
| QM-E701 | Non-Clifford gate/rotation in the stabilizer, frame, DEM or Stim-export path | use `qumo.sim.runner` (state vector), or angles that are multiples of π/2 |
| QM-E702 | Unbound symbolic parameter in a Clifford-only path | `circuit.bind(values)` first |
| QM-E703 | Non-deterministic detector/observable (anticommutes with a measurement, reset or the initial state) | fix first-round detectors / logical basis |
| QM-E704 | Unsupported feed-forward for frame simulation / Stim export (only X/Y/Z conditioned on one measured clbit) | use `sample_stabilizer()` |
| QM-E705 | Size limit (clbits > 62 for integer counts, > 64 observables, > 64 detectors for lookup, > 24 qubits for dense mitigation) | use bit-packed outputs / another decoder |
| QM-E706 | Invalid code parameters (name, distance, rounds, basis) | `surface_z`, `surface_x`, `repetition`; d ≥ 2, rounds ≥ 1 |
| QM-E707 | Unknown decoder name | `uf`, `mwpm`, `lookup`, or pass a `Decoder` to `sample_and_decode` |
| QM-E708 | Reserved: decoding failure (decoders currently count isolated defects in `failures` instead of raising) | check DEM connectivity |
| QM-E709 | Invalid calibration / noise model (T2 > 2·T1, probabilities out of range, singular confusion matrix, empty counts) | check units and calibration data |
| QM-E710 | Fit failure (fewer than 3 points, RB inverse not found) | more lengths / delays |
| QM-E711 | Noise probability outside [0, 1] | e.g. `NoiseParams.uniform(0.001)` |

## Benchmarks

`bench/bench_qec.py` (Apple M-series laptop, 1 M shots per point, rotated
surface code memory-Z, rounds = d, uniform circuit noise p). Stim and
PyMatching run on the *identical* circuit exported by `to_stim()`.

| d | p | DEM == Stim | p_L PyMatching | p_L QUMO MWPM | z | p_L QUMO UF | Stim sample shots/s | QUMO sample shots/s (1 thread) | QUMO sample shots/s (all threads) | PyMatching decode/s (1 thread) | QUMO MWPM sample+decode/s | QUMO UF sample+decode/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 0.001 | yes | 7.72e-04 | 8.15e-04 | +1.08 | 8.42e-04 | 1.23e7 | 5.39e7 | 2.03e8 | 1.37e7 | 1.33e7 | 7.71e6 |
| 3 | 0.002 | yes | 3.09e-03 | 3.07e-03 | -0.27 | 3.17e-03 | 1.06e7 | 3.63e7 | 1.36e8 | 9.53e6 | 6.39e6 | 3.16e6 |
| 3 | 0.003 | yes | 6.74e-03 | 6.82e-03 | +0.75 | 6.99e-03 | 6.74e6 | 3.18e7 | 1.24e8 | 5.91e6 | 4.36e6 | 2.91e6 |
| 3 | 0.005 | yes | 1.78e-02 | 1.79e-02 | +0.18 | 1.83e-02 | 5.87e6 | 2.48e7 | 6.88e7 | 3.62e6 | 2.70e6 | 1.55e6 |
| 5 | 0.001 | yes | 1.34e-04 | 1.25e-04 | -0.56 | 1.36e-04 | 2.96e6 | 9.75e6 | 2.76e7 | 2.51e6 | 2.03e6 | 1.29e6 |
| 5 | 0.002 | yes | 1.00e-03 | 1.07e-03 | +1.45 | 1.14e-03 | 2.62e6 | 7.92e6 | 3.22e7 | 1.26e6 | 1.22e6 | 8.73e5 |
| 5 | 0.003 | yes | 3.29e-03 | 3.39e-03 | +1.16 | 3.65e-03 | 2.13e6 | 6.63e6 | 2.68e7 | 8.00e5 | 6.06e5 | 6.33e5 |
| 5 | 0.005 | yes | 1.42e-02 | 1.42e-02 | -0.15 | 1.52e-02 | 1.37e6 | 5.11e6 | 1.68e7 | 4.22e5 | 1.81e5 | 3.41e5 |

* **DEM**: identical mechanism sets, max relative probability difference ≤ 7e-13.
* **MWPM vs PyMatching**: max |z| = 1.45 over 8 points (1 M shots each) —
  statistically indistinguishable. UF is 0–7 % worse (expected).
* **Sampling**: 3.3–4.3× Stim single-threaded (512-shot SIMD batches,
  geometric noise skipping, no output transposition), 12–17× with all cores
  (Stim timed with `bit_packed=True`, single-threaded as Stim is).
* **Decoding**: QUMO numbers are fused sample+decode on all threads;
  PyMatching is decode-only on one thread. At low p the exact MWPM keeps up;
  at d=5, p=0.5 % PyMatching's sparse blossom is ~2.3× faster per shot (per
  core: ~20×) — the dense O(k³) blossom is the known bottleneck.

`bench/bench_qec.mojo 200000` (QUMO only, same machine):

| d | qubits | ops | detectors | DEM mechanisms | DEM extraction | sample shots/s (all thr) | shot·ops/s | UF p_L @ p=0.3 % | MWPM p_L @ p=0.3 % |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 17 | 414 | 24 | 285 | 2.3 ms | 1.0e8 | 4.3e10 | 6.95e-3 | 6.80e-3 |
| 5 | 49 | 1950 | 120 | 1945 | 15 ms | 2.2e7 | 4.3e10 | 3.55e-3 | 3.30e-3 |
| 7 | 97 | 5390 | 336 | 6077 | 40 ms | 9.2e6 | 5.0e10 | 1.54e-3 | 1.46e-3 |

Below threshold the logical error rate falls with distance (p = 0.1 %:
d=3 8e-4 → d=5 1.3e-4 → d=7 ~2e-5; UF crossing point ≈ 0.6–0.8 % from the
`threshold_sweep` example).

## Limitations / next steps

* MWPM is exact but uses a dense O(k³) blossom per component — for d ≥ 9 or
  p ≥ 0.5 % prefer UF (or add a sparse-blossom implementation).
* Lookup decoder ≤ 64 detectors; DEM ≤ 64 observables; APSP table ≤ 3000 nodes.
* Frame simulation supports only single-clbit Pauli feedback; general
  feed-forward uses tableau trajectories (`sample_stabilizer`).
* Noise channels are Pauli only (no leakage, no coherent errors, no
  correlated crosstalk beyond DEPOLARIZE2); asymmetric readout is handled by
  mitigation, not by the circuit noise model.
* Codes: repetition + rotated surface (Z/X memory). Color codes, lattice
  surgery, stability experiments are natural additions to `codes.mojo`.
