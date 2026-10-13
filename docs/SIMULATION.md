# State-vector simulation (CPU)

`qumo.sim.statevector` (kernels) and `qumo.sim.runner` (execution) implement
QUMO's exact state-vector simulator: double precision, split re/im storage,
SIMD + multi-threaded kernels, gate fusion, and cache blocking with **qubit
reordering** (index-bit swaps). Everything here is plain Mojo on the CPU and
vendor-neutral: it relies only on "there is a cache hierarchy and SIMD
registers", and the two tunables (`W`, `BLOCK_BITS`) are portable constants.

```mojo
from qumo.ir.pauli import parse_pauli_sum
from qumo.sim.runner import evolve, sample, expectation
var sv = evolve(circuit)                  # StateVector, canonical qubit order
var res = sample(circuit, 1000, seed=7)   # res.counts, res.method
var e = expectation(circuit, parse_pauli_sum("Z0 Z1"))
```

## 1. Layers

| Layer | Where | What |
|---|---|---|
| Kernels | `qumo.sim.statevector` (`k1_range`, `kcx_range`, `kswap_range`, ...) | One gate on an index range of any amplitude array. Pair/quad index enumeration with `insert_zero`, `W = 8` SIMD lanes; gates on qubits 0..2 use in-vector lane shuffles / masks (qsim's low-qubit kernels). |
| Drivers | `StateVector.apply_1q`, `apply_cx`, `apply_swap`, ... | One parallel pass (`run_chunks`) per gate over the whole state. Public, stable. |
| Bit swaps | `StateVector.apply_swap_layer(qa, qb)` | Exchanges several pairs of amplitude-index bits in one in-place pass. |
| Plan | `runner.build_plan` -> `List[KOp]` | Unitary part of a circuit; runs of 1-qubit gates are fused per wire into one 2x2 matrix. Also executed by the GPU backend. |
| Executor | `runner.execute_plan` | n < 19: one parallel pass per op. n >= 19: `execute_plan_reordered` (below). |
| Entry points | `evolve`, `evolve_into`, `sample`, `expectation` | Static circuits evolve once and sample all shots; dynamic or noisy circuits run per-shot trajectories (gate by gate, `apply_instruction`). |

### 1.1 Classical bits written more than once

`sample` packs measurement results into a classical register (bit c =
clbit c; keys are Qiskit-style, leftmost = highest clbit). When several
MEASUREs write the same clbit, **the last one in program order wins**, as in
Qiskit/Aer: `x q[0]; measure q[0]->c[1]; measure q[1]->c[1]; measure q[0]->c[0]`
gives `01` on every shot. All samplers agree:

| Path | `SampleResult.method` | How the last write wins |
|---|---|---|
| static (terminal measurements only) | `statevector:static` | `terminal_measure_map(circuit)` keeps one source qubit per clbit (the last MEASURE into it); `TerminalMeasureMap.pack(index)` ORs those bits (`outcome & mask` when every pair is qubit k -> clbit k) |
| per-shot trajectories | `statevector:trajectories` | each MEASURE sets or clears its clbit |
| stabilizer / Pauli frames | `stabilizer`, `pauli_frame` | per-clbit store / records replayed in order (`qumo.sim.stabilizer`, `qumo.exec.dispatch.counts_from_measurements`) |

`terminal_measure_map` is public so that other static samplers (the GPU
state vector) can share the exact semantics. Before 2026-10-12 the static
path ORed every MEASURE into the register, so an earlier 1 survived a later
0 (the example above gave `11`); `tests/test_sim.mojo`
(`test_repeated_clbit_last_write_wins`) covers the reported cases and 150
random circuits against the trajectory path and the stabilizer sampler.

## 2. Cache blocking with qubit reordering

Above 2^19 amplitudes (8 MiB) the state no longer fits in cache and a pass
per gate is memory-bound. The executor therefore splits the amplitude index
into **local** bits `[0, nb)` and **global** bits `[nb, n)`, with
`nb = min(BLOCK_BITS = 15, n - 5)`: a block is 2^15 amplitudes = 512 KiB
(fits L2) and there are at least 32 blocks so every core gets work. A
*block pass* loads each block once and applies a whole run of ops to it while
it is resident (`_apply_run`, one parallel task per block).

### 2.1 The logical -> physical invariant (`QubitMap`)

During execution the state is stored under a permutation of the index bits:
logical qubit `q` lives at physical bit `l2p[q]` (`p2l` is the inverse).
Every op is translated to physical bits right before it runs (`_phys`).
Invariant: *amplitude `i` of the physical array is the amplitude of the
logical basis state obtained by moving bit `l2p[q]` of `i` to position `q`*.

### 2.2 Which ops can run inside a block

An op runs in a block pass when every qubit it acts on **non-diagonally** is
local (`_op_need`). Diagonal factors and controls on global bits are
constant inside a block (read from the block's base index), so they never
force a data movement:

| Op | Must be local |
|---|---|
| fused 1q matrix | its qubit, unless the matrix is diagonal |
| CX / controlled-1q | the target (control may be global) |
| CZ / CP / RZZ | nothing (diagonal) |
| diagonal controlled-1q (CRZ, CP-like) | nothing |
| dense 2q (RXX, iSWAP, ECR, ...) | both qubits |
| SWAP | nothing: it is a **relabel** of the map, no data moves |
| 3q gates (CCX, CSWAP, CCZ) | applied as one full-state pass (physical qubits) |

`apply_kop_block` contains the global-bit variants (e.g. CX with a global
control = X on the target in blocks whose control bit is 1; RZZ with one
global bit = a diagonal 1q phase on the local bit).

### 2.3 Scheduling

* Greedy, in plan order, with commutation: pending ops are scanned (up to
  `REORDER_WINDOW = 4096`); an op that cannot run blocks its wires, later ops
  on other wires may still run (gates on disjoint wires commute).
  `_collect_run` gathers every op that can run now into one block pass.
* When the head op needs a global bit, `_select_local` scans ahead and builds
  the set S of <= nb physical bits that lets the longest dependency-respecting
  stream of upcoming ops run (lookahead). If that saves at least
  `SWAP_GAIN_FIRST = 4` (first remap, which must also pay for the final
  restore) or `SWAP_GAIN = 2` (later remaps) full-state passes, ONE
  `apply_swap_layer` pass exchanges the global bits of S with local bits.
  Otherwise the head op runs as one full-state pass.
* Victims (local bits that become global): bits >= 3 first (an in-vector
  bit would cost an extra pass in the swap layer), then the bit whose next
  use is farthest away (Belady), ties -> the higher bit (longer contiguous
  runs in the swap pass).
* The "bits >= 3 first" rule is only a preference: when S fills the whole
  block there is no spare local bit >= 3 and in-vector bits 0..2 must go
  (one extra vectorized full-state pass each at swap-in, and usually again
  in the restore). In that case (`nb >= 6`) the selection is recomputed
  with bits 0..2 pinned local (they occupy 3 of the nb slots) and the pinned
  set is used when it loses at most as many upcoming ops as lane victims it
  avoids and still meets the gain threshold. Lane pairs remain possible
  (`ReorderStats.vector_pairs` counts them).

### 2.4 Restore at the end

When the plan is done, the permutation is undone so callers always see the
canonical little-endian order (qubit k = index bit k). Any permutation is
the product of two involutions: each cycle `c_0 -> c_1 -> ... -> c_{m-1}`
(data at `c_i` must move to `c_{i+1}`) is written as `tau2 o tau1` with
`tau1: c_i <-> c_{-i}` and `tau2: c_i <-> c_{1-i}`, so the restore is at most
two swap layers (a 2-cycle needs only one; the identity needs none, e.g. when
SWAP gates cancel). Cycles start at their lowest bit, which `tau1` fixes, so
qubits 0..2 mostly appear in a single layer.

### 2.5 Which APIs see which order

| API | Qubit order |
|---|---|
| `evolve`, `evolve_into`, `expectation`, `sample` (static and trajectories) | canonical (logical) |
| `execute_plan`, `execute_plan_reordered[_stats]` | canonical on return |
| `StateVector` methods (`probabilities`, `sample`, `expectation`, `measure`, `reset_qubit`, `apply_*`) | whatever the array holds; always canonical outside the executor |
| `apply_kop_block`, `_apply_run`, `QubitMap` | physical (executor internals) |

Nothing outside `execute_plan_reordered_stats` ever observes a permuted
state, so measurement, reset, feed-forward, noise, adjoint gradients
(`qumo.algorithms`), the Python extension and the GPU backend are unaffected.

### 2.6 Introspection

`execute_plan_reordered_stats(sv, plan, nb) -> ReorderStats` runs the
executor and reports `block_passes`, `block_ops`, `global_ops`,
`swap_layers`, `swapped_bits`, `relabels`, `restore_layers`, `restore_pairs`,
`restore_min_bit` and `vector_pairs` (pairs with an in-vector bit 0..2, at
swap-in or restore: each is one extra vectorized full-state pass). Example
(QFT, 24 qubits, nb = 15): `block_passes=2, block_ops=300, global_ops=0,
swap_layers=2, swapped_bits=18, relabels=12, restore_layers=2,
restore_pairs=20, vector_pairs=3` - the 312-gate circuit costs 2 block
passes + 2 swap-in passes + 2 restore passes + 3 vectorized lane passes
instead of ~300 passes (the 3 lane pairs come from QFT's final SWAPs of
qubits 0..2, which are relabels).

In-vector pairs before / after the pinned re-selection of §2.3
(deterministic counts, nb = 15, `vector_pairs` / swap-in layers / restore
layers):

| circuit | before | after |
|---|---|---|
| layered 20 | 4 / 1 / 1 | 0 / 2 / 2 (and one more block pass) |
| layered 22 | 6 / 2 / 2 | 0 / 2 / 2 |
| layered 24 | 6 / 2 / 2 | 0 / 2 / 1 |
| qft 20 / 22 / 24 | 5 / 6 / 6 | 3 / 5 / 3 (swap-in and restore layers unchanged) |
| random 20 / 22 / 24 | 7 / 10 / 11 | 5 / 7 / 7 (swap-in layers unchanged) |

Wall time of the two variants was within noise on the loaded development
machine (interleaved, order flipped every round): 22 qubits, 15 rounds at
load 44-69: the new variant won 28 of 45 paired runs, medians 0.99-1.07x;
24 qubits, 6 rounds at load 37-48: 9 of 18, medians 0.96-1.03x; 22 + 24
qubits, 7 rounds at load 15-48: 26 of 42, medians 1.00-1.15x. The change is
kept because it removes passes by count and, by construction, gives up at
most one op per avoided lane pass; no wall-time gain is claimed.

## 3. `apply_swap_layer`: index-bit swaps

`apply_swap_layer(qa, qb)` applies the disjoint SWAPs `(qa[i], qb[i])`
together, i.e. exchanges amplitude-index bits (cuStateVec `swapIndexBits`,
Aer chunk swap). All 2k positions must be distinct and in `[0, n)`
(QM-E407).

* Pairs whose low bit is an in-vector bit (0..2) run as one vectorized SWAP
  pass each (lane shuffles / masked selects in `kswap_range`). A combined
  pass for all such pairs was implemented and measured no faster (per-group
  overhead; vectors 2^y apart alias the same cache set), so it was dropped.
* All other pairs run in ONE pass. Writing an index as `(rest, H, P, low)`
  with `H` = the high bits of the pairs, `P` = the low bits and `low` = the
  `2^r` contiguous amplitudes below the lowest swapped bit `r`, the
  permutation exchanges the run at `(H=h, P=p)` with the run at `(H=p, P=h)`.
  Runs with `h == p` are fixed points and are never touched, so a k-pair
  layer reads and writes a fraction `1 - 2^-k` of the state exactly once.
* Runs are swapped directly with vector loads/stores (`_swap_runs`); when a
  run is shorter than a 128-byte line (`r = 3`) the lowest pair is folded
  into a 2x2 tile so both halves of each line are used while resident. Runs
  longer than `2^SWAP_RUN_BITS = 4096` amplitudes are cut into work items so
  that every core gets work even when only one or two runs move.
* **Memory: none.** No scratch buffer, no allocation proportional to the
  state or to the run length; the only allocations are O(2^k) bit-deposit
  tables.

## 4. Numbers

All numbers: Apple M5 (10 cores, 32 GB), fp64, `evolve()` wall time
(plan + execution), 2026-10-12, on a **shared, heavily loaded machine**
(1-minute load average 22-44 during the QUMO runs). Baseline = the
pre-reorder simulator (cache blocking only for runs of ops on qubits < 15),
built from the same tree with the old `runner.mojo` / `statevector.mojo`.
Baseline and new binaries were alternated back-to-back (order flipped every
round), 3 rounds; the table shows the median and the three samples.

**layered** (10 x [ry+rz on every qubit, CX brick]; the original benchmark)

| qubits | baseline s | reordered s | speedup |
|---|---|---|---|
| 20 | 0.087 (0.087, 0.097, 0.087) | 0.044 (0.090, 0.038, 0.044) | 1.99x |
| 22 | 0.577 (0.470, 0.593, 0.577) | 0.364 (0.302, 0.364, 0.420) | 1.58x |
| 24 | 2.460 (2.460, 3.495, 2.245) | 1.840 (1.840, 2.710, 1.793) | 1.34x |
| 26 | 12.66 (12.62, 12.66, 13.82) | 8.18 (7.54, 8.18, 11.05) | 1.55x |

**qft** (h + cp on every pair + final swaps; most cp gates touch high qubits)

| qubits | baseline s | reordered s | speedup |
|---|---|---|---|
| 20 | 0.028 (0.028, 0.026, 0.052) | 0.022 (0.025, 0.020, 0.022) | 1.30x |
| 22 | 0.268 (0.256, 0.268, 0.337) | 0.100 (0.100, 0.091, 0.108) | 2.68x |
| 24 | 1.566 (1.511, 1.566, 1.680) | 0.823 (0.823, 0.857, 0.625) | 1.90x |
| 26 | 7.51 (8.65, 6.65, 7.51) | 4.21 (4.21, 4.77, 4.17) | 1.78x |

**random** (10 x [u3 on every qubit, CX on a random perfect matching])

| qubits | baseline s | reordered s | speedup |
|---|---|---|---|
| 20 | 0.120 (0.110, 0.141, 0.120) | 0.057 (0.057, 0.055, 0.062) | 2.09x |
| 22 | 0.834 (0.689, 0.836, 0.834) | 0.342 (0.303, 0.342, 0.399) | 2.44x |
| 24 | 4.114 (4.672, 4.114, 3.806) | 2.415 (2.783, 2.120, 2.415) | 1.70x |
| 26 | 16.50 (15.09, 16.50, 16.59) | 11.42 (11.42, 8.24, 11.43) | 1.45x |

Below 19 qubits the executor is unchanged (one parallel pass per op); A/B
medians at 14/16/18 qubits were within the run-to-run spread (2-5x between
repetitions of the same binary under this load).

Absolute times are 2-3x above the quiet-machine figures quoted in the
README (e.g. layered 26: 5.65 s baseline) because of the load; the ratios are
what the interleaving protects.

The tables above rest on 3 samples per configuration and the spread is wide
(e.g. layered 26 reordered: 7.54-11.05 s). An independent re-run on the same
machine under heavier load confirms the direction but with smaller ratios
(baseline / reordered, medians):

| run | layered | qft | random | reordered faster |
|---|---|---|---|---|
| 22 q, 5 rounds, load 141-276 | 1.19x | 1.56x | 1.53x | 15/15 rounds |
| 24 q, 5 rounds, load 141-276 | 1.26x | 1.53x | 1.14x | 13/15 rounds |
| 26 q, 3 rounds, load 44-83 | 1.22x | 1.57x | 1.27x | 9/9 rounds |

Treat the speedups as "1.1-2.7x depending on load and family"; a quiet
machine is needed for tighter numbers.

**Where the time goes** (24 qubits, instrumented copy): block passes
dominate (L2-resident compute); each swap-in layer costs ~0.02-0.04 s
(about half a streaming pass); the restore 0.04-0.09 s (more when it must
move in-vector bits 0..2, e.g. after QFT's final reversal). A swap layer with
3 in-vector pairs measured 0.033-0.058 s against 0.014-0.017 s for a 4-pair
layer on bits >= 3 (24 qubits, load ~87), which is why lane victims are
avoided (§2.3).

**Swap-layer memory/time fix** (`apply_swap_layer` with the lowest swapped
bit r >= 10 used to allocate and zero-fill a 4 x 2^r-double staging buffer
per task and copy fixed-point tiles out and back; reached via the final
restore whenever the permutation cycles only among high bits):

| case | before (buffered tiles) | after (in-place runs) |
|---|---|---|
| peak RSS, 24 q, `evolve` with a final SWAP of the top two qubits | 781 MB | 269 MB (= the state) |
| peak RSS, 26 q, same | 2057-2723 MB | 1037 MB (= the state) |
| `apply_swap_layer([22],[23])`, 24 q (median of 3x5) | 0.035 s | 0.0049 s (apply_swap: 0.0039 s) |
| `apply_swap_layer([14],[23])`, 24 q | 0.0117 s | 0.0050 s (apply_swap: 0.0040 s) |
| 4 pairs `[3..6] <-> [23..20]`, 24 q | 0.0146 s | 0.0119 s (4 x apply_swap: 0.025 s) |

**QUMO vs Qiskit Aer** (qiskit-aer, `AerSimulator(method="statevector",
precision="double")`, default fusion and threads; `bench/bench_sv_aer.py`
builds the identical circuits). QUMO (reordered) and Aer were alternated
back-to-back, 3 rounds, load average 27-45. Aer wall time includes
`run().result()`; its own `time_taken` (one extra run, `--aer-internal`) is
within noise of the wall time, so harness overhead is not what separates them.

| family | qubits | QUMO s | Aer wall s | Aer internal s | Aer / QUMO |
|---|---|---|---|---|---|
| layered | 24 | 1.50 (1.50, 1.29, 1.63) | 3.94 (3.74, 3.94, 4.53) | 4.35 | 2.63x |
| layered | 26 | 7.07 (7.07, 4.61, 7.12) | 15.68 (17.40, 13.89, 15.68) | 15.22 | 2.22x |
| qft | 24 | 0.73 (0.68, 0.77, 0.73) | 2.95 (2.95, 3.08, 2.87) | 3.43 | 4.05x |
| qft | 26 | 4.71 (6.55, 3.44, 4.71) | 13.95 (15.72, 13.95, 13.80) | 15.28 | 2.97x |
| random | 24 | 1.65 (1.95, 1.65, 1.25) | 3.99 (4.42, 3.16, 3.99) | 4.02 | 2.41x |
| random | 26 | 10.45 (7.50, 10.45, 12.28) | 14.39 (14.39, 13.28, 16.36) | 14.71 | 1.38x |

The `random` family at 26 qubits has the smallest margin (1.4x, and the
widest QUMO spread, 7.5-12.3 s). Every layer puts a dense u3 on all qubits and
CX on random pairs, so the executor needs a swap-in layer per circuit layer
(6 at 24 qubits) and each block pass applies one dense 1q matrix per qubit.
Aer fuses several gates into one dense multi-qubit matrix, which plausibly
helps most on this family; multi-qubit fusion is not implemented in QUMO yet
(see Limitations), so that explanation is not measured.

## 5. Reproducing

```bash
make bench-sv                                    # layered, QUMO 16..26 + Aer
.venv/bin/mojo build -I . -o build/bench_sv bench/bench_sv.mojo
./build/bench_sv 26 qft 20 3                     # nmax family nmin reps (median)
bench/.venv/bin/python bench/bench_sv_aer.py 26 qft 20 3   # same circuits on Aer
bench/.venv/bin/python bench/bench_sv_aer.py 26 qft 20 3 --aer-internal
```

`bench/bench_sv_aer.py` rebuilds the `layered`, `qft` and `random` circuits
gate for gate (the `random` family uses the same 64-bit LCG); the final
amplitudes of both sides were checked to agree to 1e-15 at 5 and 8 qubits.
For A/B comparisons on a shared machine, alternate the two binaries
back-to-back, repeat >= 3 times, report medians and the load average.

## 6. Limitations

* 3-qubit gates (CCX, CSWAP, CCZ) are not blocked: each is one full-state
  pass. Transpile them to 1q/2q gates for large states.
* No multi-qubit gate fusion: blocks receive one op per fused 1q run / 2q
  gate. Inside a block the work is L2-bandwidth/compute bound, which now
  dominates (fusing 1q gates into neighbouring 2q gates, as qsim does, is the
  next step).
* The scheduler is greedy (lookahead window 4096, fixed gain thresholds);
  it is not an optimal swap schedule.
* In-vector bits 0..2 are still moved when SWAP gates relabel qubits that
  sit on them (e.g. QFT's final reversal: the restore must undo it) and when
  a full selection is worth more than the lane passes it costs: one extra
  vectorized pass per such pair (`ReorderStats.vector_pairs`).
* `BLOCK_BITS = 15` was kept: a sweep of nb = 12..16 at 24 qubits was within
  noise on the (heavily loaded) development machine.
* fp64, shared memory, one node. The GPU backend consumes the same fused plan
  (`build_plan`) but not this executor or its index-bit swaps: it has its own
  shared-memory tile scheduler (`qumo/gpu/tiles.mojo`, docs/GPU.md).

## 7. Origins

* **qsim** (Google): gate fusion and SIMD kernels with special in-register
  paths for low qubits.
* **Qiskit Aer**: cache blocking (`blocking_enable` / `blocking_qubits`), whose
  transpiler pass inserts chunk swaps that bring the qubits of upcoming gates
  into a chunk.
* **cuStateVec** (`custatevecSwapIndexBits`): index-bit swaps as a primitive,
  used here as `apply_swap_layer`.
* Belady's farthest-next-use rule (cache replacement) for choosing which
  qubits leave the block.

Errors: `QM-E407` (invalid swap-layer bits), `QM-E408` (invalid block size);
see docs/ERRORS.md.
