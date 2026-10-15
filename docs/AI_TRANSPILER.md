# QUMO AI transpiler

QUMO's AI transpiler is a native, hardware-neutral re-implementation of the
techniques of [qiskit-ibm-transpiler](https://github.com/Qiskit/qiskit-ibm-transpiler)
0.18.0 (Apache-2.0): block collection, topology-aware resynthesis of Clifford,
linear-function, permutation and Pauli-network blocks, "smart replacement",
learned-policy layout and routing, and a hybrid heuristic + AI pass manager.
Everything runs in Mojo inside the QUMO compiler. There is no IBM dependency
at runtime, no IBM model weight and no cloud service: the models are trained
locally with [qiskit-gym](https://github.com/AI4quantum/qiskit-gym) 0.4.1
(Apache-2.0) on QUMO's own topologies (grid, heavy-hex, line, ion, neutral
atom), and every learned proposal is verified exactly before it is used.

```mojo
from qumo.compiler.target import parse_target
from qumo.compiler.pass_manager import transpile_ai, compile_spec

var t = parse_target("heavy_hex:5/sc_cz")
var r = transpile_ai(circuit, t, 2)          # AI level 1..3 (default 2)
print(r.table())                             # per-pass report incl. replacement statistics

var r2 = compile_spec(circuit, t,
    "route(sabre,policy=auto,ai_level=2),kak,linear_synth(direction=both),perm_synth,basis,check")
```

Python (`build/qumo_py.so`): `qumo_py.transpile_ai(circ, n, edges, basis, level, seed)`
-> `(circuit, initial, final)`; `qumo_py.compile_spec_full(circ, n, edges, basis, spec, seed)`
-> `(circuit, initial, final, report_json)`. MCP / CLI: `qumo_transpile` takes
`"ai": true, "ai_level": 1..3`; `qumo describe passes` and `qumo describe models`
list the passes and the installed models.

<!-- SECTION:technique-map -->
## Technique map (IBM -> QUMO)

| qiskit-ibm-transpiler 0.18.0 | QUMO | Notes |
|---|---|---|
| `CollectCliffords` (h s sdg x y z cx cy cz swap, <= 9 qubits, >= 2 gates) | `collect.collect_blocks(kind=CLIFFORD)` | greedy growth on the commutation-aware dependency order, forward and backward, split into connected components, convex blocks only |
| `CollectLinearFunctions` (cx swap, <= 9, >= 4) | `collect_blocks(kind=LINEAR)` | |
| `CollectPermutations` (swap, <= 12, >= 4) | `collect_blocks(kind=PERM)` | |
| `CollectPauliNetworks` (Clifford + rx ry rz rzz rxx ryy ..., <= 6, >= 4) | `collect_blocks(kind=PAULI)` | |
| `AICliffordSynthesis` | `clifford_synth` pass -> `block_synth.synth_clifford` / `policy_synth.policy_synth_clifford` | exact search for small blocks, greedy otherwise, learned policy when a model matches |
| `AILinearFunctionSynthesis` | `linear_synth` -> `synth_linear` / `policy_synth_linear` | |
| `AIPermutationSynthesis` | `perm_synth` -> `synth_perm` / `policy_synth_perm` | exact A* token swapping for small blocks |
| `AIPauliNetworkSynthesis` | `pauli_network` -> `synth_pauli_network` / `policy_synth_pauli_network` | |
| "replace only if better" | `resynth.resynthesize_blocks` | plus exact verification of every proposal (IBM does not verify) and a vendor-neutral cost |
| `AIRouting` (layout_mode optimize / improve / keep, optimization_level 1-3) | `route(..., policy=auto, ai_level=, layout_mode=)` -> `ai_route.ai_route` | portfolio with SABRE: kept only when verified and cheaper |
| `generate_ai_pass_manager` | `preset_pipeline_ai(level, target)` / `transpile_ai` | ai3 is additionally a portfolio with O3 |
| model registry keyed by WL graph hash + networkx isomorphism | `policy.find_model` (WL hash prefilter + VF2 confirmation, same-size monomorphism) | |
| thread pool over blocks | `parallelize` over blocks with per-block seeded RNG splits | results do not depend on the thread count |
<!-- /SECTION:technique-map -->

<!-- SECTION:passes -->
## Pass specs and presets

The four synthesis passes share one argument set:

| key | values (default) | meaning |
|---|---|---|
| `method` (or a bare word) | `auto` \| `exact` \| `astar` \| `greedy` \| `policy` (`auto`) | `auto` = exact for small blocks, policy when a model matches, greedy otherwise |
| `max_qubits` | int (Clifford 9, linear 9, permutation 12, Pauli 6) | block width cap (IBM collector defaults) |
| `min_block` | int (Clifford 2, others 4) | minimum gates per block |
| `direction` | `forward` \| `backward` \| `both` (`both`) | collection direction (IBM `collect_from_back`) |
| `commute` | `0` \| `1` (`1`) | commutation-aware collection |
| `reps` | int (10) | collect-and-collapse repetitions |
| `model` | `auto` \| `off` \| `<model id>` (`auto`) | learned model selection; an explicit id that does not exist / match raises QM-E616 / QM-E620 |
| `beams`, `searches` | int (8, 4) | policy beam width and seeded sampled rollouts |
| `metric` | `cost_depth` \| `cost` (`cost_depth`) | replacement metric (below) |
| `strict` | `0` \| `1` (`0`) | raise QM-E619 when a proposal fails verification instead of keeping the original |
| `budget` | int (0 = default) | node-expansion cap of exact / A* / beam searches |
| `model_dir` | path | model directory (else `$QUMO_MODEL_DIR`, then `./models/ai_transpiler`) |
| `seed` | int (pipeline seed) | |

Before routing (or on an all-to-all target) blocks are synthesised for an
all-to-all subgraph; after routing on the subgraph induced by the block's
physical qubits, so every emitted 2q gate sits on a coupling edge.

Route keys: `policy=off|auto|<id>` (default `off`: plain SABRE, byte-identical
to earlier releases), `ai_level=1..3` (IBM effort table: 8/2/2, 16/16/8,
32/16/8 parallel runs / iterations / repetitions), `layout_mode=optimize|improve|keep`
(a user initial layout turns `optimize` into `improve`, as IBM does),
`model_dir=`. With a policy the pass first runs SABRE as before, then
`ai_route` on the same pre-route circuit seeded with SABRE's layout; the
learned result replaces SABRE's only when it is verified and strictly cheaper
in 2q cost (target CX units), then 2q depth.

Presets (`preset_pipeline_ai(level, target)`):

* **ai1** = O1 with `route(..., policy=auto, ai_level=1)`.
* **ai2** = O2 with `route(..., policy=auto, ai_level=2)`, then
  `linear_synth(direction=forward,commute=0)`,
  `linear_synth(direction=backward,commute=0)` (IBM's level-2 post-routing
  stage), `perm_synth` on residual SWAP networks, then kak, basis, cancel,
  commute, fuse1q, check. The synthesis passes run before `kak` (unlike the
  first draft, which ran them after it): KAK then consolidates the
  resynthesised CX networks into the target entangler, which measured better
  on the neutral set (2q geometric mean vs O2 0.994 with 31 wins / 1 loss,
  against 0.998 with 19 / 0 after `kak`, at the same compile time).
* **ai3** = O3 with `clifford_synth` before routing (all-to-all, after
  `pauli_simp`), `route(..., policy=auto, ai_level=3, layout_mode=optimize)`,
  `kak`, then `fixpoint(linear_synth(direction=both,commute=1), clifford_synth,
  perm_synth, pauli_network, max=2)` (IBM's level-3 list run twice, plus the
  opt-in Clifford / permutation / Pauli-network stages), basis, the O3 cleanup
  fixpoint and check. `transpile_ai(..., 3)` runs it as a portfolio with
  `transpile(..., 3)` (itself O3 + O2) and returns the better result (2q
  count, then depth, then gate count): AI level 3 is never worse than O3.

Without models (or on all-to-all targets, where nothing is routed) the
presets degrade gracefully: exact / greedy synthesis still runs and routing is
SABRE's.
<!-- /SECTION:passes -->

<!-- SECTION:replacement -->
## Smart replacement and the vendor-neutral metric

IBM keeps the original block when `original.num_nonlocal_gates() <=
synthesized.num_nonlocal_gates()` (SWAP = 3 CX), and compares (2q count, 2q
depth) for Pauli networks. QUMO uses one rule for every block kind: a
proposal replaces the block only if it is verified equal (below) and strictly
better in (2q cost, 2q depth) lexicographically (`metric=cost_depth`; `cost`
ignores depth). The 2q cost is counted in the target's own CX units
(`kak.cx_cost`: a native entangler counts 1 whatever its name, iSWAP-family 2
per CX, SWAP 3 unless native), so CZ, iSWAP, RZZ / RXX (ions) and CX targets
are all priced in their own currency rather than in IBM's CX count. Ties keep
the original.
<!-- /SECTION:replacement -->

<!-- SECTION:verification -->
## Verification guarantees

A learned policy (or a search) only proposes; QUMO verifies:

* permutation blocks: permutation equality;
* linear-function blocks: GF(2) matrix equality;
* Clifford blocks: Aaronson-Gottesman tableau equality including signs; the
  global phase is fixed explicitly so the statevector-exact checks of the test
  suites hold;
* Pauli-network blocks: dense unitary equality (<= 6 qubits) including phase;
* topology: every 2q gate of a physical-qubit proposal must sit on a coupling
  edge of the induced subgraph;
* learned routing: the routed circuit is replayed structurally and must use
  coupling edges only, with layouts composed exactly as SABRE's.

A failed check silently keeps the original block (counted as `rejected` in the
pass note) or, with `strict=1`, raises QM-E619. All searches use seeded
`Rng.split(k)` streams with an explicit tie-break (highest logit, then the
lowest action index), so results are deterministic and independent of the
thread count; upstream twisterl samples with an unseeded `thread_rng`.

Test coverage: tests/test_block_synth.mojo, tests/test_collect.mojo,
tests/test_policy.mojo, tests/test_ai_route.mojo and
tests/test_ai_transpiler.mojo (statevector-verified `transpile_ai` levels 1-3
on grid, heavy-hex, line, ion and neutral-atom targets; ai3 never worse than
O3; determinism; O0-O3 snapshot unchanged).
<!-- /SECTION:verification -->

<!-- SECTION:models -->
## Model format and registry

Two container formats, both a directory with `manifest.json` + `weights.bin`
(little-endian f16 or f32, 64-byte aligned tensors, FNV-1a64 checksum in the
manifest, embedded test vectors that the loader and `make test` replay):

* `qumo.rlsynth.v0` - synthesis policy (twisterl `BasicPolicy`: sparse
  embedding bag 512 -> ReLU -> 256 -> ReLU -> action / value heads), with the
  coupling graph, gate set, observation encoding and symmetry maps.
* `qumo.rlroute.v0` - routing policy (128 -> 64 -> 16 sign-embedding MLP).

`models/ai_transpiler/{synth,route}/INDEX.json` lists every model (id, kind,
qubits, edges, WL hash, sha256, bytes). Lookup order: `model_dir=`,
`$QUMO_MODEL_DIR`, `./models/ai_transpiler`. A synthesis model fires on a block
only when the induced block graph is isomorphic (or a same-size monomorphic
host) to the model graph (WL hash prefilter, exact matcher). `make
models-verify` checks sha256 against INDEX.json; the loader raises QM-E617 on
a missing / corrupt / mismatching file and QM-E620 on an incompatible model.

Shipped models (3.3 MB in total, trained locally by QUMO):

| model | kind | qubits | topology |
|---|---|---:|---|
| perm_grid3x3, perm_grid3x4, perm_king3x4, perm_hhring10s2 | permutation | 9 / 12 | grid, king grid, heavy-hex ring |
| lf_grid2x3, lf_ring6, lf_line6, lf_hhH7 | linear function | 6 / 7 | grid, ring, line, heavy-hex H |
| pauli_line4 (continued) | Pauli network | 4 | line |
| route_mlp_h64_v1 | routing | any | topology-independent |

`qumo describe models` lists what is installed.
<!-- /SECTION:models -->

<!-- SECTION:training -->
## Training budgets and retraining

`make train-ai` (not part of `make test`) runs `training/ai_transpiler/synth`
(qiskit-gym 0.4.1 PPO, CPU, Apple M5) and `training/ai_transpiler/route`
(behaviour cloning + one DAgger round on a rollout teacher). Requirements:
qiskit-gym, twisterl and torch in a scratch venv (training only, never at
runtime). See `training/ai_transpiler/README.md`.

Budget ledger of the shipped campaign (2 parallel lanes, about 9.4 h CPU,
about 4.7 h wall for synthesis; about 76 min for the router). Costs are mean 2q
cost in CX units over 100 seeded random blocks of the model graph.

| model | qubits | budget | greedy | policy beam 8 | ratio vs greedy | kept |
|---|---:|---:|---:|---:|---:|---|
| perm_grid3x3 | 9 | 10 min | 28.65 | 27.03 | 0.943 | yes |
| perm_grid3x4 | 12 | 15 min | 47.61 | 43.59 | 0.916 | yes |
| perm_king3x4 | 12 | 20 min | 37.17 | 33.69 | 0.906 | yes |
| perm_hhring10s2 | 12 | 45 min | 68.01 | 60.33 | 0.887 | yes |
| lf_grid2x3 | 6 | 45 min | 19.35 | 16.77 | 0.867 | yes |
| lf_ring6 | 6 | 40 min | 21.61 | 18.93 | 0.876 | yes |
| lf_line6 | 6 | 45 min | 26.10 | 25.77 | 0.987 | yes |
| lf_hhH7 | 7 | 45 min | 33.41 | 32.10 | 0.961 | yes |
| pauli_line4 (cont.) | 4 | 70 min | 18.75 | 17.02 | 0.908 | yes |
| clifford_* (5 models, 4q) | 4 | 20-50 min | 7.3-9.9 | 8.6-11.2 | 1.02-1.21 | dropped |
| lf_line7, pauli_complete5, pauli_line6, perm_line8 | 5-8 | 10-40 min | | | >= 1.0 | dropped |

Drop rule: a model that is not better than the greedy synthesizer (mean cost
ratio < 1) is not shipped. On this budget the Clifford models did not beat the
exact / greedy Clifford synthesizers (exact search is optimal up to 3 qubits),
so Clifford synthesis ships without a model. Honest summary: learned synthesis
gains are limited to permutations (6-13% vs greedy) and linear functions up to
7 qubits; everything else is carried by exact search and greedy methods.
<!-- /SECTION:training -->

<!-- SECTION:results -->
## Results

All numbers below come from `make bench-ai` style runs on one Apple M5 (10
cores, other jobs running, so times are noisy), every row verified:
statevector (<= 14 qubits) for the neutral harness, exact Pauli-sum
propagation for the Clifford stress set. Ratios < 1 mean qumo-ai is better.
Raw data and full tables: `bench/results/ai_neutral.jsonl`,
`ai_neutral_{default,max}.md`, `ai_stress.{jsonl,md}`, `ai_benchpress.md`.
The IBM-trained AI transpiler was NOT benchmarked (it requires downloading an
unlicensed package and IBM weights; it stays an optional external baseline,
`qiskit+ibm-ai` in `bench/neutral/sdks.py`).

### Neutral harness: 64 circuits x 11 targets x 2 levels (1408 jobs, 0 failures)

| level (qumo-ai) | vs | 2q ratio | 2q-depth ratio | W/T/L | median time ratio |
|---|---|---:|---:|---:|---:|
| default (ai level 2) | qumo | 0.994 | 0.994 | 88/603/13 | 3.2x |
| default | qiskit | 0.884 | 0.883 | 339/352/13 | 0.45x |
| default | tket | 0.856 | 0.861 | 361/264/78 | 0.01x |
| max (ai level 3) | qumo | 0.954 | 0.945 | 188/516/0 | 10.8x |
| max | qiskit | 0.840 | 0.831 | 394/306/4 | 3.4x |
| max | tket | 0.852 | 0.851 | 377/262/54 | 0.06x |

* Level 3 is never worse than O3 (0 losses, portfolio); level 2 loses 13 of
  704 jobs (1.9%) to the plain pipeline, in KAK interplay cases.
* Gains concentrate on superconducting / abstract targets (0.991 default,
  0.955 max); neutral-atom and ion targets move 0.2-5%.
* Median compile time: 2.0 ms (default) and 31 ms (max) against 0.9 / 2.0 ms
  for qumo. The default ratio meets the 10x target; max (10.8x) narrowly
  misses it.
* Technique contributions at max (replaced blocks): clifford_synth 24.8k
  (saved 14.2k CX units), linear_synth 13.0k (9.3k), pauli_network 11.9k
  (1.3k), perm_synth 27 (0). Learned routing beat SABRE in 14% (default) and
  20% (max) of the jobs where it ran.

### Clifford / linear / permutation / Pauli stress set (208 jobs per SDK, up to 60 qubits)

| family | qumo-ai/qumo (default / max) | vs qiskit (max) | vs tket (max) |
|---|---|---:|---:|
| random Clifford | 0.934 / 0.869 | 0.819 | 0.716 |
| CNOT-only linear functions | 0.788 / 0.663 | 0.607 | 0.551 |
| Trotter Pauli networks | 0.918 / 0.879 | 0.789 | 0.702 |
| UCC-like Pauli networks | 0.914 / 0.888 | 0.710 | 0.619 |
| SWAP networks / permutations | 1.000 / 1.000 | 1.000 | 1.000 |

Permutation and SWAP-network circuits tie everywhere: all compilers absorb a
pure SWAP network into the layout, so there is nothing left to synthesize.
Median compile time on these large circuits is 8-26x qumo (above the 10x
goal), i.e. synthesis is paid for in time on 40-60 qubit Cliffords.
Tket timed out on 5 jobs.

### Component notes

* Synthesis: `method=auto` is optimal on all 140 random permutations up to 10
  qubits; greedy LF / Clifford are within 10-20% of optimal at 4-5 qubits.
* Collect driver overhead on tiny circuits is not negligible: clifford /
  pauli passes cost 2-5x O2 on a 1.6 ms circuit (documented limitation).
* Router: policy alone is 1.07x SABRE's 2q cost; only the portfolio is
  never-worse (0.995-0.997x), at 6-7x SABRE's route time.
* Paper reference points (quoted, not claimed): IBM's cloud service reached
  0.96 geometric-mean 2q count relative to Qiskit at about 18.7x the runtime
  (arXiv:2409.08844); the RL routing paper reports about 10% fewer 2q gates on
  133-qubit QV (arXiv:2405.13196).

### Benchpress device suite

Benchpress `device_transpile` (133-qubit heavy-hex, cz basis, level 2), qumo vs
qumo-ai (`[qumo] ai = true`, `transpile_ai` level 2) in one session: 140 tests
passed in both runs (Benchpress's own checks). Geometric-mean 2q ratio 0.955
(depth 0.965; W/T/L 87/44/9): feynman 0.958, hamiltonians 0.950, summit-100
0.983. The median transpile time is 17x qumo's (above the 10x goal on these
large circuits). Details: `bench/results/ai_benchpress.md`. The 3 h cap was
not needed (22 min for the AI run).
<!-- /SECTION:results -->

<!-- SECTION:licensing -->
## Licensing

* The algorithms follow qiskit-ibm-transpiler 0.18.0 (Apache-2.0), qiskit-gym
  0.4.1 and twisterl 0.5.1 (Apache-2.0), credited in NOTICE as algorithmic
  references. No file of those projects is copied (several IBM files carry an
  "All Rights Reserved" header, so QUMO re-implements from specifications).
* No IBM-trained weights (Hugging Face `qiskit/ai-transpiler_*`) and no code
  from `qiskit-ibm-ai-local-transpiler` (which declares no license) are used or
  redistributed. QUMO's models are trained locally and ship under Apache-2.0.
* qiskit-gym / twisterl / torch are needed only to retrain
  (`make train-ai`), never at runtime.
<!-- /SECTION:licensing -->
