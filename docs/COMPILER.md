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
| `pauli_simp` | Pauli-gadget resynthesis: a bit-packed stabiliser tableau pulls every rotation back to the input frame (`U = C_total . prod exp(-i t P_k)`), identical commuting Paulis merge, gadgets are re-emitted as basis change + CX chain + rz with each chain reusing the previous chain's prefix (ladders cancel between consecutive terms), commuting gadgets are reordered greedily; kept only when it beats the original after cancel/commute/KAK | TKET `PauliSimp`, Qiskit Pauli-network synthesis |
| `elide_swaps` | delete SWAPs by relabelling (folded into `final_layout`); only trailing swaps once routed | Qiskit `ElidePermutations` |
| `unroll3q` | ccx / ccz / cswap → 1q + cx | Qiskit `Unroll3qOrMore` |
| `route(sabre, ...)` | VF2 exact layout → else SabreLayout trials (greedy weighted embeddings, relaxed-VF2 spanning-forest seed, dense, random; forward/backward refinement; heuristic portfolio) → swap trials (on small circuits half of them favour swaps KAK can absorb, ranked by 3·swaps − 2·absorbable) → LightSABRE routing → swap sliding (each SWAP is moved through relabelled gates next to a 2q gate on the same pair, staying on coupling edges); star pre-routing variant evaluated as a portfolio | Qiskit `VF2Layout`, `SabreLayout`/LightSABRE, `StarPreRouting`; TKET `GraphPlacement` |
| `kak` | maximal 2q-block consolidation (repeated until stable, so blocks freed by an identity block merge), exact Weyl/KAK resynthesis with 0-3 CX (one rxx/ryy/rzz per Weyl coordinate on ion/ZZ targets, 0-2 iSWAPs for c=0 classes on iSWAP targets) when cheaper; absorbs SWAPs into neighbouring gates | Qiskit `ConsolidateBlocks`+`UnitarySynthesis`, TKET `KAKDecomposition` |
| `basis(gates)` | exact rule-based translation to any gate family (cx, cz, ecr, iswap, iswapdg, rzz, rxx, ryy, cp entanglers) | Qiskit `BasisTranslator`, TKET `rebase` |
| `fixpoint(p1,...,max=N)` | repeat until 2q / gates / depth stop improving | TKET `RepeatPass` |
| `check` | predicate: fail with QM-E615 unless the circuit conforms to the target | TKET predicates |

`route` keys: `layout=auto|sabre|dense|trivial|random|vf2`, `seed`, `trials`,
`iterations`, `swap_trials`, `lookahead` (extended-set size), `weight`,
`decay`, `heuristic=auto|extended|layer`, `vf2_steps`, `valve`, `star=on|off`,
`absorb` / `absorb_alt` (score factor for absorbable swaps; 1.0 / 0.9).

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
