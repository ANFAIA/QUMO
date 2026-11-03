# QUMO compiler

`qumo.compiler` turns a `Circuit` into one that runs on a `Target`: only basis
gates, two-qubit gates only on coupling edges, every rewrite exact (the unitary
is preserved, global phase tracked in `Circuit.global_phase`). It is vendor
neutral: targets are described by topology and gate family, not by vendor.

```mojo
from qumo.compiler.target import Target, parse_target
from qumo.compiler.pass_manager import transpile, compile_spec

var t = Target.superconducting_heavy_hex(7)          # or parse_target("heavy_hex:7/sc_cz")
var r = transpile(circuit, t, 2)                      # preset level 0..3
print(r.summary())                                    # one line
print(r.table())                                      # per-pass report
var out = r.circuit                                   # physical-qubit circuit
# r.initial_layout[l] / r.final_layout[l]: physical qubit of logical l before / after

var r2 = compile_spec(circuit, t, "remove_identities,cancel,route(sabre,trials=20),kak,basis,fuse1q,check")
```

Run `.venv/bin/mojo run -I . examples/compile_demo.mojo` for a tour and
`.venv/bin/mojo run -I . tests/test_compiler.mojo` for the statevector-verified tests.

## Targets (`target.mojo`)

A `Target` is plain data (Qiskit `Target` / Braket device properties /
PennyLane TOML capabilities): `num_qubits`, undirected coupling `edges`,
`basis` (opcode list, empty = any gate), per-qubit `error_1q`, `readout_error`,
`t1_us`, `t2_us`, per-edge `error_2q`, per-opcode `duration_ns`, and a
precomputed all-pairs hop-distance matrix.

| Topology presets | Modality presets (default basis) |
|---|---|
| `all_to_all(n)` (default target, basis `any`) | `superconducting_heavy_hex(d)` (`sc_cz`) |
| `line(n)`, `ring(n)` | `superconducting_grid(r, c)` (`sc_cz`) |
| `grid(r, c)`, `torus(r, c)` | `superconducting_grid_iswap(r, c)` (`sc_iswap`) |
| `tree(n, branching)` | `trapped_ion(n)` (`ion_ms`, all-to-all) |
| `heavy_hex(d)` (= rustworkx `heavy_hex_graph(d)`, (5d²-2d-1)/2 qubits) | `neutral_atom(r, c, radius)` (`atom_cz`, king graph by default) |
| `from_edges(n, flat_edges, basis)` (any coupling map) | |

Basis strings: a gate list (`"rz,sx,x,cz"`, `"rx,ry,rz,cz"`, `"rz,rx,rxx"`,
`"u,cx"`, `"rz,ry,rzz"`, ...) or a preset: `any`, `generic` {u,cx}, `sc_cx`,
`sc_cz` {rz,sx,x,cz}, `sc_ecr`, `sc_iswap` {rz,rx,ry,iswap}, `ion_ms`
{rz,rx,ry,rxx}, `ion_zz` {rz,ry,rzz}, `atom_cz` {rz,rx,ry,cz}, `clifford_t`.
One-line target specs: `parse_target("grid:4x4/sc_cz")`, `"heavy_hex:5/rz,sx,x,ecr"`,
`"trapped_ion:11"`, `"neutral_atom:3x3"`, `"line:20/generic"`.
`describe()` returns compact JSON; `estimate_fidelity(c)` and
`estimate_duration_ns(c)` use the error / duration data.

## Pipeline language (`pass_manager.mojo`)

```
pipeline := item ((","|";") item)*
item     := name [ "(" arg ("," arg)* ")" ]
arg      := word | key "=" value | item          # fixpoint(...) nests pipelines
```

Example: `remove_identities,elide_swaps,fixpoint(cancel,commute,fuse1q,max=4),route(sabre,layout=auto,trials=20,swap_trials=20),kak,basis,cancel,commute,fuse1q,check`.
`PassManager(spec).canonical()` prints the normalised spec; `describe_passes()`
is a JSON catalog. Each pass yields a `PassReport` (gates / 2q / depth before
and after, wall time in µs, note); `CompileResult.to_json()` is the
machine-readable summary.

| Pass | What it does | Inspired by |
|---|---|---|
| `remove_identities` | drop gates within 1e-12 process infidelity of the identity (rz(0), rx(4π), cp(1e-9), ...); phase kept | Qiskit `RemoveIdentityEquivalent` |
| `cancel(window)` | commutation-aware inverse-pair cancellation over a per-wire linked list (O(n·window)) | Qiskit `InverseCancellation` / `CommutativeInverseCancellation`, TKET `RemoveRedundancies` |
| `merge` | adjacent same-axis rotations (rz rz, t t→s, rzz rzz, cp cp, cz·cp), symbolic params merge by `Param.try_add` | Qiskit `Optimize1qGates`, TKET |
| `commute(window)` | merge/cancel rotations *through* commuting gates (rz through cx controls / cz / diagonals, rx through cx targets) | Qiskit `CommutativeCancellation` |
| `fuse1q(basis)` | fuse 1q runs, resynthesise with the shortest exact Euler form of the basis (U, ZYZ, ZXZ, XYX, ZSX), memoised | Qiskit `Optimize1qGatesDecomposition` |
| `eject_z` | push Z rotations through Z-type wires, swaps, Pauli-Z frames through CX targets, into U gates | Cirq `eject_z` (virtual Z) |
| `phase_fold` | phase-polynomial folding of diagonal rotations on equal parities (T-count), RZZ folds too | TKET / PyZX phase folding, T-par |
| `wrap_angles(gate[min:max])` | bring constant angles into the target's bounds with Pauli corrections; not in presets unless the target has bounds | Qiskit `WrapAngles`, IBM `FoldRzzAngle` |
| `post_layout(call_limit)` | rename physical qubits to the best embedding under the calibration; not in presets unless the target is calibrated | Qiskit `VF2PostLayout` |
| `pauli_simp` | Pauli-gadget resynthesis: a bit-packed stabiliser tableau pulls every rotation back to the input frame (`U = C_total . prod exp(-i t P_k)`), identical commuting Paulis merge, gadgets are re-emitted as basis change + CX chain + rz with each chain reusing the previous chain's prefix (ladders cancel between consecutive terms), commuting gadgets are reordered greedily; kept only when it beats the original after cancel/commute/KAK | TKET `PauliSimp`, Qiskit Pauli-network synthesis |
| `elide_swaps` | delete SWAPs by relabelling (folded into `final_layout`); only trailing swaps once routed | Qiskit `ElidePermutations` |
| `unroll3q` | ccx / ccz / cswap → 1q + cx | Qiskit `Unroll3qOrMore` |
| `route(sabre, ...)` | VF2 exact layout → else SabreLayout trials (greedy weighted embeddings, relaxed-VF2 spanning-forest seed, dense, random; forward/backward refinement; heuristic portfolio) → swap trials (on small circuits half of them favour swaps KAK can absorb, ranked by 3·swaps − 2·absorbable) → LightSABRE routing → swap sliding (each SWAP is moved through relabelled gates next to a 2q gate on the same pair, staying on coupling edges); star pre-routing variant evaluated as a portfolio | Qiskit `VF2Layout`, `SabreLayout`/LightSABRE, `StarPreRouting`; TKET `GraphPlacement` |
| `kak` | maximal 2q-block consolidation (repeated until stable, so blocks freed by an identity block merge), exact Weyl/KAK resynthesis with 0-3 CX (one rxx/ryy/rzz per Weyl coordinate on ion/ZZ targets, 0-2 iSWAPs for c=0 classes on iSWAP targets) when cheaper; absorbs SWAPs into neighbouring gates | Qiskit `ConsolidateBlocks`+`UnitarySynthesis`, TKET `KAKDecomposition` |
| `basis(gates)` | exact rule-based translation to any gate family (cx, cz, ecr, iswap, iswapdg, rzz, rxx, ryy, cp entanglers) | Qiskit `BasisTranslator`, TKET `rebase` |
| `fixpoint(p1,...,max=N)` | repeat until 2q / gates / depth stop improving | TKET `RepeatPass` |
| `check` | predicate: fail with QM-E615 unless the circuit conforms to the target | TKET predicates |
| `clifford_synth` | collect Clifford blocks (h, s, sdg, sx, x, y, z, cx, cy, cz, swap; <= 9 qubits, >= 2 gates) and resynthesise them on the block's coupling subgraph (exact search for small blocks, greedy, or a learned policy); replaced only when verified equal by tableau and cheaper | qiskit-ibm-transpiler `CollectCliffords` + `AICliffordSynthesis` |
| `linear_synth` | the same for CNOT/SWAP linear functions (<= 9 qubits, >= 4 gates), verified by GF(2) matrix | `CollectLinearFunctions` + `AILinearFunctionSynthesis` |
| `perm_synth` | the same for SWAP networks (<= 12 qubits) by token swapping (exact A* / greedy / policy), verified by permutation | `CollectPermutations` + `AIPermutationSynthesis` |
| `pauli_network` | the same for Pauli-rotation networks (Clifford + rx / ry / rz / rzz / rxx / ryy; <= 6 qubits), verified by dense unitary | `CollectPauliNetworks` + `AIPauliNetworkSynthesis` |

`route` keys: `layout=auto|sabre|dense|trivial|random|vf2`, `seed`, `trials`,
`iterations`, `swap_trials`, `lookahead` (extended-set size), `weight`,
`decay`, `heuristic=auto|extended|layer`, `vf2_steps`, `valve`, `star=on|off`,
`absorb` / `absorb_alt` (score factor for absorbable swaps; 1.0 / 0.9).
AI routing keys (docs/AI_TRANSPILER.md): `policy=off|auto|<model id>` (off by
default; otherwise the learned-policy layout + routing runs after SABRE on the
same circuit, seeded with SABRE's layout, and replaces it only when its
verified result is strictly cheaper in 2q cost (target CX units), then 2q
depth), `ai_level=1..3` (effort), `layout_mode=optimize|improve|keep`,
`model_dir=<dir>`.

Synthesis pass keys (`clifford_synth`, `linear_synth`, `perm_synth`,
`pauli_network`): `method=auto|exact|astar|greedy|policy` (a bare word is the
method), `max_qubits`, `min_block`, `direction=forward|backward|both`,
`commute=0|1`, `reps`, `model=auto|off|<id>`, `beams`, `searches`,
`metric=cost_depth|cost`, `strict=0|1` (raise QM-E619 instead of silently
keeping a block whose proposal fails verification), `budget` (search node
cap), `model_dir`, `seed`. Before routing (or on all-to-all targets) blocks are
synthesised all-to-all; after routing on the induced coupling subgraph of the
block's physical qubits.

Passes that only optimise (cancel, merge, commute, fuse1q, eject_z,
phase_fold) emit gates of the *working gate set*: any gate before basis
translation, the target basis once the circuit is expressed in it, so passes
placed after `basis` never leave the basis.

### Preset levels (`preset_pipeline(level, target)`)

* **O0** trivial layout routing (if needed) + basis.
* **O1** identities, cancel, merge, route (5 trials), basis, cancel, merge, fuse1q, check.
* **O2** identities, elide_swaps, pauli_simp, fixpoint(cancel, commute, fuse1q),
  route (20 layout trials, 2 iterations, 20 swap trials), kak, basis, cancel,
  commute, fuse1q, check. (Pre-routing KAK and phase folding were measured at O2
  and left out: they made routed results slightly worse on the Benchpress set.)
* **O3** adds phase folding and pre-routing KAK, 40 trials / 3 iterations, a
  post-basis fixpoint with eject_z; `transpile(..., 3)` also runs O2 and returns
  the better of the two (so O3 is never worse than O2).

### AI presets (`preset_pipeline_ai(level, target)`, `transpile_ai`)

The hybrid heuristic + AI presets mirror qiskit-ibm-transpiler's
`generate_ai_pass_manager` while staying vendor neutral (docs/AI_TRANSPILER.md):

* **ai1** O1 with `route(..., policy=auto, ai_level=1)`.
* **ai2** O2 with policy routing (`ai_level=2`), then after `route`:
  `linear_synth(direction=forward,commute=0)`, `linear_synth(direction=backward,commute=0)`,
  `perm_synth`, then kak, basis, cancel, commute, fuse1q, check.
* **ai3** O3 with `clifford_synth` before routing (all-to-all), policy routing
  (`ai_level=3, layout_mode=optimize`), `kak`, then
  `fixpoint(linear_synth(direction=both,commute=1), clifford_synth, perm_synth, pauli_network, max=2)`,
  basis, the O3 cleanup fixpoint and check; `transpile_ai(..., 3)` runs it as a
  portfolio with `transpile(..., 3)`, so it is never worse than O3 in 2q count.

`transpile` / `preset_pipeline` / `compile_spec` and the O0-O3 outputs are
unchanged (a snapshot test in tests/test_ai_transpiler.mojo pins them).

### Calibration and angle bounds

A `Target` holds per-qubit and per-edge errors. `set_calibration(error_1q,
edge_errors, readout, t1_us, t2_us)` and `set_edge_error(a, b, e)` fill them, and
`is_calibrated()` says whether they differ between qubits or edges. Presets only
change for such a target, so uniform targets keep their output byte for byte:

| Target has | Preset adds, before `check` | Pass |
|---|---|---|
| non-uniform errors, level >= 1 | `post_layout` | renames physical qubits to the embedding of the interaction graph with the best `estimate_fidelity`; branch and bound with a `call_limit` (default 200000), starting from the identity, so it never lowers the estimate |
| `gate[min:max]` in the basis, e.g. `heavy_hex:3/rz,sx,x,cz,rzz[0:pi/2]` | `wrap_angles,fuse1q` | brings constant angles into the interval with Pauli corrections, exact with the tracked global phase |

O3's portfolio compares by estimated fidelity on a calibrated target. `check`
reports an angle outside its interval (QM-E615). JSON: `calibration` and
`angle_bounds` in `qumo_transpile`; Python: `qumo_py.transpile_calibrated`; the
hardware plugin passes `TargetSpec.calibration()` by default.

`post_layout` moves the circuit, it does not route for noise: SABRE still minimises
swaps. On IBM fake backends (Torino, Fez, Marrakesh; GHZ 5 to 16, QFT 6 and 8, BV
12; `bench/crosscheck/tier1.py layout`, success estimated from the backend's real
calibration) it gains 0 to 4% over the plugin's `select_region` (geomean 1.009).
Without any calibration a compile on the whole device can land on a broken coupler
(estimates of 1e-19 and below in that table); `select_region` already avoids those.
Qiskit O3 is slightly ahead on some GHZ cases on Torino and behind on QFT 8.

`wrap_angles` matched `FoldRzzAngle` from qiskit-ibm-runtime on 60 random angles:
the same rzz angle in all 60, the same number of Pauli corrections, operators equal.

### Pauli-based computation (`qumo.compiler.pbc`)

`to_pbc(c, fix_clifford=True)` returns a `PauliProgram`: time-ordered Pauli-product
rotations and measurements in the input frame, plus one final Clifford. A product
can touch any number of qubits, so this is a side structure and not an opcode;
`to_circuit()` gives gates back. Angles follow `exp(-i angle/2 P)`, so Litinski's
pi/8 rotation (T) is `angle = pi/4`. `counts()` gives rotations, pi8, arbitrary,
measurements, max_weight and layers. Same rotations as Qiskit's
`LitinskiTransformation` on 40 of 40 random Clifford+T circuits. Conditions, resets
and noise raise QM-E623.

### T-count objective

`pauli_simp(objective=t)` keeps the resynthesis when it has fewer T / non-Clifford
rotations after phase folding, with two-qubit gates as the tie break and no
two-qubit budget, so it can never lose to folding the original. It can cost
two-qubit gates, so it stays out of O0-O3. On the Feynman circuits against
`unroll3q` + phase folding:

| circuit | T in | folding | + objective=t | tket `PauliGraphResynthesis` folding |
|---|---|---|---|---|
| mod5_4 | 28 | 16 | **8** | 8 |
| adder_8 | 399 | 215 | **201** | 173 |
| tof_3, barenco_tof_3, rc_adder_6, gf2p4_mult, csum_mux_9, qcla_com_7, mod_mult_55, ham15-low | | same | same | same |

## Correctness model

* Every pass is exact; phases are tracked (`global_phase`). The tests check
  equality *including the global phase* on random input states, and routed
  circuits through `initial_layout` / `final_layout`.
* Classically conditioned gates are never merged / cancelled / commuted; when a
  conditioned gate is decomposed every piece keeps `cond` / `cond_val` /
  `cond_reg`; routing treats conditioned 2q gates like any other gate (routing
  SWAPs are unconditioned). Measurements, resets, detectors and conditioned
  gates keep their program order (classical chain in the routing DAG).
* Symbolic parameters: affine `Param` arithmetic keeps circuits parametric; the
  parameter-dependent part of a global phase (rz ↔ p) cannot be stored in
  `Circuit.global_phase` and is reported as `symbolic_phase_dropped`.

* Constant angles of magnitude 1e4 or more are reduced through sin and cos before
  any pass (`reduce_large_angles` in `PassManager.run`). Merging sums angles, and at
  2^63 the sum drops the smaller term and `wrap_angle` has no bits left: `transpile`
  at level 2 used to change `sx rz(1) p(2^63-1) sx rz(1) p(2^63-1)`. The verifier of
  programs found it. Output changes only for such angles. The offset of a symbolic
  angle reduces the same way; a huge coefficient on a symbol stays a float limit.
* **`transpile` drops small rotations:** a rotation below about 2e-6 is within the
  1e-12 infidelity of `remove_identities` and goes, so many such gates can add up to
  more than each one. Program blocks compile in exact mode instead, below.
* Past 2^22 quarter turns an angle is not Clifford (`QUARTER_TURN_LIMIT`), in
  `Instruction.is_clifford` and in the helpers that read a count of quarter turns.

## Programs

`compile_program(p, target, level, seed)` and `compile_spec_program(p, target, spec,
seed)` compile a `Program` block by block on one layout, in
`qumo/compiler/program_compile.mojo`. A program without runtime regions goes through
`transpile`.

* Each block compiles in exact mode (`PassManager.exact`, `GateSet.exact`): angle and
  matrix tolerances are 1e-14 (`EXACT_TOL`) and the KAK check 1e-13, far below the 1e-7
  of `verify_program`, so nothing it can see is dropped.
* The pipeline splits at its `route`. Passes around it run on each block, without
  `elide_swaps`, `post_layout` and `star`, which relabel qubits going forward only.
  A second `route`, a relabeling pass inside `fixpoint`, synthesis passes and the AI
  presets are QM-E630.
* One layout comes from a summary circuit that weighs loop bodies, and each block
  routes from it with `RouteOptions.initial_layout`.
* A region exits with the layout it entered with: each branch, `break` and `continue`
  restores the header's layout with a SWAP network, chosen inside each connected
  component. If none exists, QM-E632.
* A runtime angle on a gate with target angle bounds is QM-E633.
* `verify_program` runs on every result (QM-E631): skeleton, layouts on every edge,
  `l_out == perm(l_in)`, a compiled block at the place of each reachable source block,
  no reset, noise or conditional gate that the source lacks, and each block against its
  source on two seeded random states, with measurements deferred onto ancillas. Blocks
  whose source has a reset, noise or a conditional gate, and blocks over the budget
  (gates x 2^wires x 2 up to 2^27 per block and 2^29 per compile, 22 wires at most), are
  not checked, and the result counts them: `blocks_checked`, `blocks_unchecked`.
* `ProgramMetrics` gives base gates plus a cost per loop iteration. There is no global
  depth, which would be a misleading number.

The rules, the numbers and what stays out are in [CONTROL_FLOW.md](CONTROL_FLOW.md).

## Error codes

| Code | Meaning |
|---|---|
| QM-E600 | unknown pass name (did-you-mean + list of valid passes) |
| QM-E601 | malformed pipeline (unbalanced parentheses, text after `)`) |
| QM-E602 | bad pass argument (unknown key, non-integer, unknown layout / heuristic) |
| QM-E603 | basis has no usable two-qubit entangler / no decomposition rule |
| QM-E604 | a 1-qubit gate cannot be expressed in the basis |
| QM-E605 | circuit has more qubits than the target |
| QM-E606 | interacting qubits in disconnected parts of the coupling map |
| QM-E607 | a 3-qubit gate reached routing (run `unroll3q` / `basis` first) |
| QM-E609 | optimization level outside 0..3 |
| QM-E610 | unknown gate name in a basis spec |
| QM-E611 | malformed target spec |
| QM-E612 | invalid target construction (bad edge, even heavy-hex distance, ...) |
| QM-E615 | `check` predicate failed (gate outside basis or off the coupling map) |
| QM-E616 | unknown synthesis method or model name (`method=auto|exact|astar|greedy|policy`) |
| QM-E617 | AI model file missing, corrupt, or failed its checksum / format check |
| QM-E618 | block too wide for the requested synthesis method |
| QM-E619 | synthesized block not equivalent to the original (`strict=1` only; otherwise the original is kept) |
| QM-E620 | AI model incompatible with the request (kind, size, topology or gate set) |
| QM-E630 | a `Program` cannot use the AI presets, synthesis passes, a second `route` or a relabeling pass inside `fixpoint` |
| QM-E631 | a compiled `Program` failed verification |
| QM-E632 | no SWAP network brings a region back to its entry layout |
| QM-E633 | a gate with target angle bounds takes an angle known only at run time |

## Benchmarks

Benchpress device suite (133-qubit heavy-hex, basis cz,id,rz,sx,x, O2):

| circuit | QUMO 2q | Qiskit 2.5 O2 2q | QUMO s | Qiskit s |
|---|---|---|---|---|
| QFT_100 | 8272 | 9971 | 0.27 | 0.25 |
| QV_100 | 98397 | 96366 | 3.0 | 4.1 |
| circSU2_89 | 300 | 345 | 0.05 | 2.9 |
| BV_100 | 197 | 196 | 0.02 | 0.04 |
| square_heisenberg_100 | 1083 | 1572 | 0.05 | 0.05 |
| QAOA_100 | 8576 | 8660 | 0.19 | 0.19 |
| clifford_100 | 58442 | 58252 | 0.89 | 0.76 |

HamLib Hamiltonians (Benchpress, 2q-count geometric mean QUMO / Qiskit; lower is
better): all-to-all 0.92 (100 instances), heavy-hex 0.82 (100), device 0.91 (81).
QASMBench small circuits on linear / heavy-hex / all-to-all ion targets (neutral
harness): 2q geometric mean QUMO / Qiskit 0.92, QUMO fewer 2q gates on 37 of 114
(circuit, target) pairs, more on 5.

## Known limitations

* iSWAP-family targets: KAK emits 0 / 1 / 2 iSWAPs whenever a Weyl coordinate
  vanishes (CX, CP, RZZ, SWAP·CX, ...), but a generic SU(4) block costs 4
  iSWAPs (Ud(a,b,0)·Ud(0,0,c)); the optimal 3-iSWAP template is future work.
* `phase_fold` treats measurement / reset / barriers as hard region boundaries.
* KAK skips blocks that contain symbolic parameters or classically conditioned gates.
* SABRE is heuristic: results depend on `seed` (deterministic for a given seed).
* SABRE ignores calibration: the noise-aware step is `post_layout` after routing.
  KAK and `fuse1q` still price native gates, not per-edge errors.
* A `clifford_t` target needs angles it can express exactly (QM-E604 names the first
  input gate that it cannot); there is no gridsynth yet.
