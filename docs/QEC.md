# QUMO QEC + experimentalist stack

Pure-Mojo replacement for the Stim → PyMatching → Sinter pipeline, plus the
hardware-characterization tooling an experimentalist needs (noise models from
calibration data, randomized benchmarking, T1/Ramsey, readout mitigation).
Everything shares the one QUMO `Circuit` IR (noise channels, `DETECTOR`,
`OBSERVABLE`, measurement records), so there is no Cirq↔Stim↔Qiskit
conversion step anywhere.

```
qumo.qec.codes ──► Circuit (+coords) ──► qumo.qec.dem ──► DetectorErrorModel ──► qumo.qec.decoders
     │                 │                                   (Stim-exact)            MatchingGraph ──► qumo.qec.sparse_blossom
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
.venv/bin/mojo run -I . tests/test_qec_sparse_blossom.mojo  # sparse blossom exactness suite
.venv/bin/mojo run -I . examples/qec_memory_experiment.mojo
.venv/bin/mojo run -I . examples/qec_characterization.mojo
.venv/bin/mojo run -I . examples/qec_custom_decoder.mojo
.venv/bin/mojo run -I . bench/bench_qec.mojo 200000     # throughput + LER grid
bench/.venv/bin/python bench/bench_qec.py --shots 1000000   # cross-check vs Stim + PyMatching (parts A + B)
bench/.venv/bin/python bench/bench_qec.py --part B --tp-shots 30000 --reps 5   # identical-syndrome decoder table
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
| Exact MWPM on the detector graph: graph fill regions, integer event queue, alternating trees, blossoms, compressed edges | Sparse blossom (Higgott & Gidney 2023, PyMatching 2) | `qumo/qec/sparse_blossom.mojo` |
| Dense exact MWPM (APSP table, Edmonds blossom) | PyMatching v0, van Rantwijk's mwmatching | `MwpmDecoder`, `BlossomMatcher` |
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
* `SparseBlossomDecoder` (`qumo.qec.sparse_blossom`, decoder name `mwpm` /
  `sparse_blossom`): exact MWPM by sparse blossom — the default matcher, see
  the next section.
* `MwpmDecoder` (decoder name `mwpm_dense`): exact, dense. All-pairs shortest
  paths (distance + observable parity) precomputed once per graph (≤ 3000
  nodes, ≤ 8 observables, shared by thread copies through an `ArcPointer`),
  otherwise per-defect Dijkstra. Defects are split into independent
  components (i, j interact only if d_ij < b_i + b_j); each component is
  solved by subset DP (≤ 10 defects) or Edmonds' blossom (port of van
  Rantwijk's O(n³) algorithm) on the reduced graph
  w'(i,j) = min(d_ij, b_i + b_j) plus one boundary vertex for odd counts.
  Per-shot cost is O(k³) in the largest component: fine for d ≤ 5, slow
  beyond (kept as an independent reference implementation).
* `LookupTableDecoder(dem, max_weight=2)`: most likely observable per
  syndrome over all ≤ max_weight error combinations; ≤ 64 detectors.
* Batch helpers: `decode_samples`, `count_logical_errors` (parallel over
  512-shot batches, one decoder copy per worker).

All of them correct every single fault of the d = 3, 5, 7 surface codes
(Z and X memory) — checked in `tests/test_qec.mojo` and
`tests/test_qec_sparse_blossom.mojo`.

### Sparse blossom (`qumo.qec.sparse_blossom`)

Port of the algorithm behind PyMatching 2 — O. Higgott & C. Gidney,
*"Sparse Blossom: correcting a million errors per core second with
minimum-weight matching"*, Quantum 9, 1600 (2025), arXiv:2303.15933. Instead
of building shortest paths and a complete graph of defects (the dense
decoder), Edmonds' primal-dual blossom algorithm runs **directly on the
detector graph**; dual variables are regions of the graph, so per-shot work
is proportional to the area the regions sweep (local at low p).

```mojo
from qumo.qec.sparse_blossom import SparseBlossomDecoder, MatchingOracle, CompressedEdge
var dec = SparseBlossomDecoder(MatchingGraph(dem))   # or SparseBlossomDecoder.from_dem(dem)
var obs  = dec.decode(fired)               # UInt64 observable mask (trait Decoder)
var es   = dec.decode_to_edges(fired)      # List[CompressedEdge(a, b, obs)], b = -1: boundary
var prs  = dec.decode_to_matches(fired)    # [a0, b0, a1, b1, ...] (PyMatching's decode_to_matched_dets_array)
var w    = dec.last_weight()               # matching weight (integer units; / (2 * dec.weight_scale()) = nats)
# counters: dec.failures (odd component without boundary), dec.ignored (bad / repeated ids), dec.anomalies (QM-E712, must be 0)
```

* **Integer weights** (PyMatching's discretisation): w_int = 2·round(w·c)
  with c = (2²⁴ − 1)/max|w|. Even weights make every collision happen at an
  integer time (two growing fronts close an even gap at rate 2), so all
  arithmetic is exact `Int` (`discretize_weights`, `weight_scale()`).
* **Graph fill regions**: one per detection event plus one per blossom. A
  region's radius is `y + slope·t` with slope +1 (outer node of an
  alternating tree), −1 (inner) or 0 (matched, or a blossom child). Each
  detector node stores the region that reached it, its top-level blossom, the
  source detection event, the arrival radius, a cached "wrapped radius"
  (frozen radii of intermediate blossom children) and the observables
  crossed from the source. Node data are graph-sized arrays allocated once
  and reset lazily through a touched-node list; regions and tree nodes come
  from pools that keep their capacity across shots (no per-shot allocation).
* **Event queue (flooder)**: a binary min-heap of (time, node | region)
  with PyMatching's queued/desired-time trackers, so stale entries are
  dropped lazily. Node events: a growing region reaches an empty neighbour,
  hits another region, or hits the boundary; the next event time at a node
  is one branch-free scan of its CSR neighbours (empty nodes belong to a
  frozen zero-radius sentinel region). Region events: a shrinking region gives
  back its last node (LIFO shell), a trivial inner region implodes (its two
  tree neighbours touch → blossom), an inner blossom reaches radius 0 and
  shatters.
* **Matcher**: alternating trees of (inner, outer) region pairs. A collision
  either grows a tree by a matched pair, augments (two trees meet, or a tree
  meets the boundary or a boundary-matched region: re-root at the collision
  and dissolve the tree into matches), or forms a blossom (a tree hits
  itself: the odd cycle through the lowest common ancestor becomes a new
  growing region). Shattering puts the even path between the blossom's two
  tree edges back into the tree and matches the rest of the cycle in pairs.
* **Compressed edges**: every match / tree / blossom edge stores only its two
  endpoint detection events and the XOR of the observables along the region
  path between them; the prediction is the XOR over the final matching's
  compressed edges (blossoms expanded recursively), so no path is traced.
* **Thread safety**: the immutable integer graph is shared by decoder copies
  through an `ArcPointer`; each copy owns its mutable state, so
  `sample_and_decode` / `decode_samples` run one copy per worker.

**Verification.** `tests/test_qec_sparse_blossom.mojo` compares every result
against `MatchingOracle` — an independent reference on the *same* integer
weights (Dijkstra from every node on the product graph node × observable
mask, plus the dense exact matcher): primal weight of the returned pairs ==
dual objective == oracle optimum, every detection event matched exactly once,
and each pair's observable mask is that of some shortest path between its
endpoints (so the prediction is the observable of a minimum-weight
correction). Suite: surface Z/X and repetition codes, d = 3, 5, 7,
p ∈ {0.001, 0.005, 0.01}, sampled + dense random syndromes (10k), and 300
random graphs with heterogeneous / tied weights and 1–3 observables (7.5k
syndromes): **0 mismatches**. Offline we ran the same check on 198k
syndromes (d ≤ 7) and 4.8k syndromes at d = 9, 11 (up to 245 detection
events per syndrome): 0 mismatches, 0 anomalies. Because the discretisation
is PyMatching's, QUMO and PyMatching 2.4 return **the same prediction on
299 999 of 300 000** identical Stim-sampled syndromes in the benchmark below;
the single difference is a degenerate tie (two minimum-weight matchings of
exactly equal integer weight with different observable parity).

## Experiments (`qumo.qec.experiments`)

```mojo
var r = memory_experiment("surface_z", distance=5, rounds=-1, p=0.002,
                          shots=2_000_000, decoder="mwpm", seed=7, max_errors=200)
print(r)              # p_L, Wilson 95% interval, per-round rate, throughput
print(r.to_json())    # machine-readable for agents
print(threshold_sweep("surface_z", [3, 5, 7], [0.003, 0.006, 0.01], 200_000, "mwpm"))
```

Decoder names (`canonical_decoder_name`): `mwpm` (= sparse blossom, the
default; aliases `matching`, `pymatching`), `sparse_blossom`, `mwpm_dense`
(APSP + dense blossom `MwpmDecoder`), `uf`, `lookup`.

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
| QM-E707 | Unknown decoder name | `mwpm` (sparse blossom), `sparse_blossom`, `mwpm_dense`, `uf`, `lookup`, or pass a `Decoder` to `sample_and_decode` |
| QM-E708 | Reserved: decoding failure (decoders currently count isolated defects in `failures` instead of raising) | check DEM connectivity |
| QM-E709 | Invalid calibration / noise model (T2 > 2·T1, probabilities out of range, singular confusion matrix, empty counts) | check units and calibration data |
| QM-E710 | Fit failure (fewer than 3 points, RB inverse not found) | more lengths / delays |
| QM-E711 | Noise probability outside [0, 1] | e.g. `NoiseParams.uniform(0.001)` |
| QM-E712 | Internal invariant violation in sparse blossom (never raised per shot; counted in `SparseBlossomDecoder.anomalies`, must stay 0) | report the syndrome; decode it with `mwpm_dense` meanwhile |

## Benchmarks

All numbers: Apple M5 laptop (4 performance + 6 efficiency cores;
`parallelism_level()` = 4), shared with other jobs while measuring, so
absolute throughputs carry ±30 % noise; ratios come from interleaved A/B
runs. Rotated surface code memory-Z (rounds = d) unless stated, uniform
circuit noise p; Stim 1.16 and PyMatching 2.4.0 run on the *identical*
circuit exported by `to_stim()`.

### Part A — logical error rates (`bench/bench_qec.py --part A --shots 500000 --dense`)

QUMO samples with its own frame simulator and decodes in the fused parallel
loop; PyMatching decodes Stim samples (decode-only, one thread). 500 k shots
per point.

| d | p | DEM == Stim | p_L PyMatching | p_L QUMO `mwpm` (sparse blossom) | z | p_L QUMO `mwpm_dense` | p_L QUMO UF | Stim sample/s | QUMO sample/s (1 thr) | QUMO sample/s (all thr) | PyMatching dec/s (1 thr) | QUMO `mwpm` sample+dec/s (all thr) | QUMO UF sample+dec/s (all thr) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 0.001 | yes | 8.12e-04 | 8.46e-04 | +0.59 | 8.46e-04 | 8.74e-04 | 1.21e7 | 4.58e7 | 1.07e8 | 1.84e7 | 2.25e7 | 6.12e6 |
| 3 | 0.002 | yes | 3.10e-03 | 3.11e-03 | +0.05 | 3.11e-03 | 3.19e-03 | 6.5e6 | 2.93e7 | 6.28e7 | 4.47e6 | 4.23e6 | 2.15e6 |
| 3 | 0.003 | yes | 6.94e-03 | 6.86e-03 | -0.52 | 6.86e-03 | 7.04e-03 | 7.61e6 | 1.98e7 | 6.77e7 | 5.29e6 | 9.85e6 | 1.74e6 |
| 3 | 0.005 | yes | 1.76e-02 | 1.77e-02 | +0.61 | 1.77e-02 | 1.82e-02 | 4.99e6 | 1.86e7 | 4.68e7 | 3.15e6 | 8.24e6 | 1.02e6 |
| 5 | 0.001 | yes | 1.30e-04 | 1.22e-04 | -0.36 | 1.22e-04 | 1.34e-04 | 2.29e6 | 9.32e6 | 3.65e7 | 2.11e6 | 4.17e6 | 9.57e5 |
| 5 | 0.002 | yes | 9.96e-04 | 1.06e-03 | +1.03 | 1.06e-03 | 1.15e-03 | 1.72e6 | 7.48e6 | 2.82e7 | 7.87e5 | 3.97e6 | 5.81e5 |
| 5 | 0.003 | yes | 3.43e-03 | 3.44e-03 | +0.03 | 3.44e-03 | 3.67e-03 | 1.72e6 | 7.73e6 | 2.38e7 | 7.88e5 | 3.35e6 | 5.06e5 |
| 5 | 0.005 | yes | 1.40e-02 | 1.43e-02 | +1.28 | 1.43e-02 | 1.54e-02 | 1.15e6 | 6.01e6 | 1.04e7 | 3.04e5 | 1.39e6 | 2.0e5 |
| 7 | 0.001 | yes | 2.40e-05 | 1.40e-05 | -1.15 | 1.40e-05 | 1.60e-05 | 9.74e5 | 3.58e6 | 1.22e7 | 8.1e5 | 2.35e6 | 3.74e5 |
| 7 | 0.002 | yes | 2.84e-04 | 2.82e-04 | -0.06 | 2.82e-04 | 3.16e-04 | 6.87e5 | 1.92e6 | 5.25e6 | 3.27e5 | 1.26e6 | 1.66e5 |
| 7 | 0.003 | yes | 1.40e-03 | 1.41e-03 | +0.16 | 1.41e-03 | 1.52e-03 | 7.65e5 | 2.03e6 | 6.9e6 | 2.35e5 | 7.36e5 | 1.15e5 |
| 7 | 0.005 | yes | 1.03e-02 | 1.00e-02 | -1.36 | 1.00e-02 | 1.11e-02 | 3.97e5 | 2.48e6 | 7.4e6 | 8.02e4 | 3.36e5 | 5.91e4 |

* **DEM**: identical mechanism sets, max relative probability difference ≤ 7e-13.
* **MWPM vs PyMatching**: max |z| = 1.36 over 12 points — statistically
  indistinguishable. Sparse and dense MWPM report the *same* error count on
  all 12 points (same QUMO samples). UF is 3–14 % worse (expected), and
  with sparse blossom in place the exact matcher is now also 2–7× *faster*
  than QUMO's UF in the fused loop (UF still allocates per shot).
* **Sampling**: 2.6–6× Stim single-threaded, 8–19× with all cores (Stim
  timed with `bit_packed=True`, single-threaded as Stim is).

### Part B — identical syndromes (`bench/bench_qec.py --part B --tp-shots 30000 --reps 5`)

Stim samples detection events once (b8 file); PyMatching (`decode_batch`)
and QUMO's sparse blossom (`bench/bench_qec_decoders.mojo --decode`) decode
the *same* shots. Decode-only throughput, medians of 5 interleaved A/B
repetitions; QUMO "all thr" = one decoder copy per worker (4 workers).

| code | d | p | shots | events/shot | predictions differ | errors PyMatching | errors QUMO | PyMatching dec/s (1 thr) | QUMO dec/s (1 thr) | QUMO / PyMatching (1 thr) | QUMO dec/s (all thr) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| surface_z | 5 | 0.001 | 30000 | 1.7 | 0 | 4 | 4 | 2.73e6 | 3.88e6 | 1.42 | 9.24e6 |
| surface_z | 5 | 0.005 | 30000 | 8.3 | 0 | 444 | 444 | 5.12e5 | 7.26e5 | 1.42 | 2.16e6 |
| surface_z | 7 | 0.001 | 30000 | 5.4 | 0 | 0 | 0 | 1.08e6 | 1.70e6 | 1.57 | 4.12e6 |
| surface_z | 7 | 0.005 | 30000 | 25.0 | 0 | 310 | 310 | 1.72e5 | 2.32e5 | 1.34 | 6.67e5 |
| surface_z | 9 | 0.001 | 30000 | 12.0 | 0 | 0 | 0 | 5.34e5 | 7.40e5 | 1.38 | 1.77e6 |
| surface_z | 9 | 0.005 | 10000 | 55.7 | 0 | 68 | 68 | 7.44e4 | 1.04e5 | 1.39 | 2.91e5 |
| surface_z | 11 | 0.001 | 30000 | 22.6 | 0 | 0 | 0 | 2.19e5 | 3.56e5 | 1.63 | 9.61e5 |
| surface_z | 11 | 0.005 | 10000 | 105.2 | 0 | 49 | 49 | 4.14e4 | 5.56e4 | 1.34 | 1.50e5 |
| surface_x | 5 | 0.005 | 30000 | 8.4 | 0 | 486 | 486 | - | - | - | - |
| surface_x | 7 | 0.005 | 30000 | 25.3 | 1 | 365 | 366 | - | - | - | - |
| repetition | 7 | 0.01 | 30000 | 3.5 | 0 | 9 | 9 | - | - | - | - |
| repetition | 11 | 0.005 | 10000 | 4.7 | 0 | 0 | 0 | - | - | - | - |

* **Agreement**: identical predictions on 299 999 of 300 000 shots. The one
  difference (surface_x d=7, shot 26723, 54 detection events) is a genuine
  degenerate tie: QUMO's and PyMatching's matchings have the same integer
  weight 640 799 958 (= the oracle optimum; 126.3746815 nats) but pair six
  events differently, with opposite observable parity.
* **Speed**: QUMO's sparse blossom decodes 1.34–1.63× more shots per second
  than PyMatching 2 on one thread for d = 5–11 at p = 0.1 % and 0.5 % (the
  DoD target was "within 2×"). Hardware instruction counts (`/usr/bin/time
  -l`, decode-only deltas, single thread, same syndrome distribution) agree:
  at p = 0.5 % QUMO retires 0.81–0.96× PyMatching's instructions per shot
  (d = 7: 47.8 k vs 50.8 k; d = 11: 236 k vs 246 k), at p = 0.1 % and
  d ≥ 7 ~0.7×. PyMatching's figures include its batch-API overhead (one
  dense uint8 detector row per shot), so per-algorithm the two are close
  to parity rather than QUMO being fundamentally faster.
* **Profile** (d = 7, p = 0.5 %, 25 events/shot): ~93 node scans, 64 heap
  push/pop pairs, ~14 region collisions, 3.4 boundary hits and 0.4 blossoms
  per shot; neighbour scans are ~42 % of the time, the binary-heap event
  queue ~22 %, the rest is the matcher, region pools and extraction. About
  half the node events are lazy re-checks of nodes whose region froze (the
  same lazy-invalidation strategy as PyMatching).

### Sparse vs dense MWPM (`bench_qec_decoders --self ... 3`, single thread)

| d | p | events/shot | `mwpm_dense` µs/shot | `mwpm` (sparse blossom) µs/shot | speed-up |
|---|---|---|---|---|---|
| 3 | 0.001 | 0.3 | 0.13 | 0.024 | 5.4× |
| 3 | 0.005 | 1.4 | 0.66 | 0.17 | 3.7× |
| 5 | 0.001 | 1.8 | 0.69 | 0.17 | 4.0× |
| 5 | 0.005 | 8.3 | 16.3 | 1.09 | 15× |
| 7 | 0.001 | 5.4 | 5.4 | 0.82 | 6.6× |
| 7 | 0.005 | 25.0 | 223 | 4.3 | 52× |

(Measured back-to-back on the loaded machine; an earlier, quieter run gave
~64 µs/shot for the dense decoder at d = 7, p = 0.5 %.)

### QUMO only (`bench/bench_qec.mojo 200000`, earlier run, same machine; the MWPM column was measured with the dense decoder, which gives the same predictions)

| d | qubits | ops | detectors | DEM mechanisms | DEM extraction | sample shots/s (all thr) | shot·ops/s | UF p_L @ p=0.3 % | MWPM p_L @ p=0.3 % |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 17 | 414 | 24 | 285 | 2.3 ms | 1.0e8 | 4.3e10 | 6.95e-3 | 6.80e-3 |
| 5 | 49 | 1950 | 120 | 1945 | 15 ms | 2.2e7 | 4.3e10 | 3.55e-3 | 3.30e-3 |
| 7 | 97 | 5390 | 336 | 6077 | 40 ms | 9.2e6 | 5.0e10 | 1.54e-3 | 1.46e-3 |

Below threshold the logical error rate falls with distance (p = 0.1 %:
d=3 8e-4 → d=5 1.3e-4 → d=7 ~2e-5; UF crossing point ≈ 0.6–0.8 % from the
`threshold_sweep` example).

## Limitations / next steps

* Sparse blossom (`mwpm`) is exact w.r.t. the discretised weights
  (2²⁴ levels, PyMatching's choice): two matchings whose float weights differ
  by less than the rounding step can be ranked differently than with exact
  real weights. Edge weights come from `MatchingGraph`, i.e. log((1−p)/p)
  clamped to [1e−3, 50] nats — p > 0.5 edges are *not* turned into negative
  weights as PyMatching does. Not ported from PyMatching: correlated
  (two-pass) matching, per-shot edge reweighting, and `decode_to_edges_array`
  (the explicit edge set of the correction — QUMO returns the matched
  detection-event pairs with their observable masks). Events in an odd
  component without a boundary are counted in `failures`.
* Single-thread throughput is on par with PyMatching 2 (see Benchmarks); the
  remaining cost is dominated by neighbour scans (~45 % of instructions) and
  the binary-heap event queue (~20 %); PyMatching's radix heap and 32-bit
  cyclic times are not ported.
* The dense `mwpm_dense` decoder is O(k³) per component — only for small codes.
* Lookup decoder ≤ 64 detectors; DEM ≤ 64 observables; APSP table ≤ 3000 nodes.
* Frame simulation supports only single-clbit Pauli feedback; general
  feed-forward uses tableau trajectories (`sample_stabilizer`).
* Noise channels are Pauli only (no leakage, no coherent errors, no
  correlated crosstalk beyond DEPOLARIZE2); asymmetric readout is handled by
  mitigation, not by the circuit noise model.
* Codes: repetition + rotated surface (Z/X memory). Color codes, lattice
  surgery, stability experiments are natural additions to `codes.mojo`.
