# QUMO pipeline passes: names, aliases and arguments

<!-- GENERATED FILE, do not edit by hand. Source of truth: pass_decl() in
     qumo/compiler/pass_manager.mojo. Regenerate with
     QUMO_UPDATE_DOCS=1 .venv/bin/mojo run -I . tests/test_mcp.mojo -->

What each pass does and where it comes from is in [COMPILER.md](COMPILER.md). This page is the reference of
what a pipeline string may hold. An alias is another name for the same pass; `canonical()` prints the
canonical one. An argument is written `key=value`; where the table says so the value alone is enough.
When an argument is written twice the last one counts. A flag takes `0|1|true|false|on|off`.

| Pass | Aliases | Arguments |
|---|---|---|
| `remove_identities` | `remove_identity`, `rmid`, `identities` | none |
| `cancel` | `cancellation`, `inverse_cancel`, `cancel_inverses` | `window` |
| `merge` | `merge_rotations` | none |
| `commute` | `commutation`, `commute_rotations` | `window` |
| `fuse1q` | `fuse`, `fuse_1q`, `optimize1q` | `gates` |
| `phase_fold` | `phase_folding`, `phasefold` | none |
| `eject_z` | `ejectz`, `virtual_z` | none |
| `unroll3q` | `unroll_3q`, `decompose3q` | none |
| `route` | `routing`, `sabre` | `method`, `layout`, `seed`, `trials`, `iterations`, `lookahead`, `weight`, `decay`, `vf2_steps`, `valve`, `star`, `heuristic`, `swap_trials`, `absorb`, `absorb_alt`, `mirror`, `mirror_gain`, `commute`, `policy`, `ai_level`, `layout_mode`, `model_dir` |
| `basis` | `translate`, `basis_translate`, `rebase` | `gates` |
| `fixpoint` | `fix`, `repeat`, `loop` | `passes`, `max` |
| `check` | `verify`, `predicate` | none |
| `kak` | `consolidate`, `consolidate_blocks`, `2q_synth` | none |
| `elide_swaps` | `elide`, `elide_permutations` | none |
| `pauli_simp` | `paulisimp`, `pauli`, `gadgets` | `objective` |
| `clifford_synth` | `ai_clifford`, `clifford_synthesis` | `method`, `max_qubits`, `min_block`, `direction`, `commute`, `reps`, `model`, `beams`, `searches`, `metric`, `strict`, `budget`, `model_dir`, `seed` |
| `linear_synth` | `ai_linear`, `ai_linear_function`, `linear_function_synth` | `method`, `max_qubits`, `min_block`, `direction`, `commute`, `reps`, `model`, `beams`, `searches`, `metric`, `strict`, `budget`, `model_dir`, `seed` |
| `perm_synth` | `ai_permutation`, `permutation_synth` | `method`, `max_qubits`, `min_block`, `direction`, `commute`, `reps`, `model`, `beams`, `searches`, `metric`, `strict`, `budget`, `model_dir`, `seed` |
| `pauli_network` | `ai_pauli_network`, `pauli_network_synth` | `method`, `max_qubits`, `min_block`, `direction`, `commute`, `reps`, `model`, `beams`, `searches`, `metric`, `strict`, `budget`, `model_dir`, `seed` |
| `wrap_angles` | `wrap`, `fold_angles` | `bounds` |
| `post_layout` | `vf2_post_layout`, `noise_layout` | `call_limit` |

## cancel

commutation-aware inverse-pair cancellation.

| Argument | Written | Values | Default | Notes |
|---|---|---|---|---|
| `window` | `window=...` or the value alone | `N` | `32` | gates looked back along a wire |

## commute

merge/cancel rotations through commuting gates.

| Argument | Written | Values | Default | Notes |
|---|---|---|---|---|
| `window` | `window=...` or the value alone | `N` | `32` | gates looked back along a wire |

## fuse1q

fuse 1q runs and resynthesise over the basis.

| Argument | Written | Values | Default | Notes |
|---|---|---|---|---|
| `gates` | the value alone | `gate list or basis preset` |  | default: the target basis |

## route

VF2 exact layout, else SabreLayout trials + LightSABRE routing (star pre-routing portfolio).

| Argument | Written | Values | Default | Notes |
|---|---|---|---|---|
| `method` | the value alone | `sabre\|lightsabre` |  | the routing method (one, under two names) |
| `layout` | `layout=...` | `auto\|sabre\|dense\|trivial\|random\|vf2` | `auto` | initial layout |
| `seed` | `seed=...` | `N` |  | default: the compile seed |
| `trials` | `trials=...` | `N` | `4` | layout trials |
| `iterations` | `iterations=...` | `N` | `2` | forward/backward refinements of a layout trial |
| `lookahead` | `lookahead=...` | `N` | `20` | extended-set size |
| `weight` | `weight=...` | `X` | `0.5` | weight of the extended set |
| `decay` | `decay=...` | `X` | `0.001` | decay added to a qubit each time it is swapped |
| `vf2_steps` | `vf2_steps=...` | `N` | `20000` | step budget of the exact layout search |
| `valve` | `valve=...` | `N` | `0` | swaps without progress before the release valve (0 = 10 * num_qubits) |
| `star` | `star=...` | `on\|off` | `on` | star pre-routing as a candidate |
| `heuristic` | `heuristic=...` | `auto\|extended\|layer` | `auto` | swap score (auto: trials alternate) |
| `swap_trials` | `swap_trials=...` | `N` | `1` | routing seeds tried on the chosen layout (best kept) |
| `absorb` | `absorb=...` | `X` | `1.0` | score factor for swaps KAK can absorb (< 1 favours them) |
| `absorb_alt` | `absorb_alt=...` | `X` | `0.9` | that factor on every other swap trial |
| `mirror` | `mirror=...` | `on\|off\|free\|auto` | `auto` | mirror-gate routing: apply swaps absorbed by the gate just executed on the pair (free: also the ones that cost nothing; auto: decided per circuit) |
| `mirror_gain` | `mirror_gain=...` | `X` | `1.0` | heuristic gain a 1-CX mirror must reach |
| `commute` | `commute=...` | `on\|off\|auto` | `auto` | commutation-aware routing DAG (auto: decided per circuit) |
| `policy` | `policy=...` | `<policy>` | `off` | off\|auto\|<model id>: learned-policy layout + routing after SABRE, kept only when verified and cheaper |
| `ai_level` | `ai_level=...` | `1..3` | `2` | effort of the learned router |
| `layout_mode` | `layout_mode=...` | `optimize\|improve\|keep` | `optimize` | what the learned router does with the SABRE layout |
| `model_dir` | `model_dir=...` | `<model_dir>` |  | model directory, searched first |

## basis

translate into a basis.

| Argument | Written | Values | Default | Notes |
|---|---|---|---|---|
| `gates` | the value alone | `gate list or basis preset` |  | default: the target basis |

## fixpoint

repeat the inner pipeline until gates/2q/depth stop improving.

| Argument | Written | Values | Default | Notes |
|---|---|---|---|---|
| `passes` | the value alone | `pass` |  | the inner pipeline |
| `max` | `max=...` | `N` | `10` | iterations at most |

## pauli_simp

Pauli-gadget resynthesis (Clifford tableau extraction, identical-Pauli merging, ladder-sharing chains); kept only when it saves 2q gates.

| Argument | Written | Values | Default | Notes |
|---|---|---|---|---|
| `objective` | `objective=...` | `2q\|t` | `2q` | t keeps the result on fewer T / non-Clifford rotations instead (FT track, may cost 2q gates, not in O0-O3) |

## clifford_synth

collect Clifford blocks (h,s,sdg,sx,x,y,z,cx,cy,cz,swap; <= 9 qubits) and resynthesise them topology-aware (exact search, greedy or learned policy); a block is replaced only when verified equal by tableau and cheaper (2q cost, then 2q depth).

| Argument | Written | Values | Default | Notes |
|---|---|---|---|---|
| `method` | `method=...` or the value alone | `auto\|exact\|astar\|greedy\|policy` | `auto` | how a block is resynthesised; list policy models with describe |
| `max_qubits` | `max_qubits=...` | `1..64` | `9` | widest block collected |
| `min_block` | `min_block=...` | `N` | `2` | fewest gates of a collected block |
| `direction` | `direction=...` | `forward\|backward\|both` | `both` | collection sweeps |
| `commute` | `commute=...` | `on\|off` | `1` | commutation-aware collection |
| `reps` | `reps=...` | `N` | `10` | sweeps at most |
| `model` | `model=...` | `<model>` | `auto` | auto\|off\|<model id>; list models with describe |
| `beams` | `beams=...` | `N` | `8` | beam width of the policy search |
| `searches` | `searches=...` | `N` | `4` | policy searches per block |
| `metric` | `metric=...` | `cost_depth\|cost` | `cost_depth` | what makes a replacement cheaper: 2q cost then 2q depth, or 2q cost only |
| `strict` | `strict=...` | `on\|off` | `0` | raise QM-E619 instead of keeping a block whose proposal fails verification |
| `budget` | `budget=...` | `N` | `0` | search node cap (0 = the method's own) |
| `model_dir` | `model_dir=...` | `<model_dir>` |  | model directory, searched first |
| `seed` | `seed=...` | `N` |  | default: the compile seed |

## linear_synth

collect linear-function blocks (cx,swap; <= 9 qubits) and resynthesise them on the coupling subgraph; replaced only when verified equal by GF(2) matrix and cheaper.

| Argument | Written | Values | Default | Notes |
|---|---|---|---|---|
| `method` | `method=...` or the value alone | `auto\|exact\|astar\|greedy\|policy` | `auto` | how a block is resynthesised; list policy models with describe |
| `max_qubits` | `max_qubits=...` | `1..64` | `9` | widest block collected |
| `min_block` | `min_block=...` | `N` | `4` | fewest gates of a collected block |
| `direction` | `direction=...` | `forward\|backward\|both` | `both` | collection sweeps |
| `commute` | `commute=...` | `on\|off` | `1` | commutation-aware collection |
| `reps` | `reps=...` | `N` | `10` | sweeps at most |
| `model` | `model=...` | `<model>` | `auto` | auto\|off\|<model id>; list models with describe |
| `beams` | `beams=...` | `N` | `8` | beam width of the policy search |
| `searches` | `searches=...` | `N` | `4` | policy searches per block |
| `metric` | `metric=...` | `cost_depth\|cost` | `cost_depth` | what makes a replacement cheaper: 2q cost then 2q depth, or 2q cost only |
| `strict` | `strict=...` | `on\|off` | `0` | raise QM-E619 instead of keeping a block whose proposal fails verification |
| `budget` | `budget=...` | `N` | `0` | search node cap (0 = the method's own) |
| `model_dir` | `model_dir=...` | `<model_dir>` |  | model directory, searched first |
| `seed` | `seed=...` | `N` |  | default: the compile seed |

## perm_synth

collect SWAP networks (<= 12 qubits) and resynthesise the permutation by token swapping (exact A* for small blocks, greedy or learned policy); replaced only when verified equal and cheaper.

| Argument | Written | Values | Default | Notes |
|---|---|---|---|---|
| `method` | `method=...` or the value alone | `auto\|exact\|astar\|greedy\|policy` | `auto` | how a block is resynthesised; list policy models with describe |
| `max_qubits` | `max_qubits=...` | `1..64` | `12` | widest block collected |
| `min_block` | `min_block=...` | `N` | `4` | fewest gates of a collected block |
| `direction` | `direction=...` | `forward\|backward\|both` | `both` | collection sweeps |
| `commute` | `commute=...` | `on\|off` | `1` | commutation-aware collection |
| `reps` | `reps=...` | `N` | `10` | sweeps at most |
| `model` | `model=...` | `<model>` | `auto` | auto\|off\|<model id>; list models with describe |
| `beams` | `beams=...` | `N` | `8` | beam width of the policy search |
| `searches` | `searches=...` | `N` | `4` | policy searches per block |
| `metric` | `metric=...` | `cost_depth\|cost` | `cost_depth` | what makes a replacement cheaper: 2q cost then 2q depth, or 2q cost only |
| `strict` | `strict=...` | `on\|off` | `0` | raise QM-E619 instead of keeping a block whose proposal fails verification |
| `budget` | `budget=...` | `N` | `0` | search node cap (0 = the method's own) |
| `model_dir` | `model_dir=...` | `<model_dir>` |  | model directory, searched first |
| `seed` | `seed=...` | `N` |  | default: the compile seed |

## pauli_network

collect Pauli-rotation networks (Clifford + rx/ry/rz/rzz/rxx/ryy; <= 6 qubits) and resynthesise rotations + Clifford tail topology-aware; replaced only when verified equal (unitary) and cheaper.

| Argument | Written | Values | Default | Notes |
|---|---|---|---|---|
| `method` | `method=...` or the value alone | `auto\|exact\|astar\|greedy\|policy` | `auto` | how a block is resynthesised; list policy models with describe |
| `max_qubits` | `max_qubits=...` | `1..64` | `6` | widest block collected |
| `min_block` | `min_block=...` | `N` | `4` | fewest gates of a collected block |
| `direction` | `direction=...` | `forward\|backward\|both` | `both` | collection sweeps |
| `commute` | `commute=...` | `on\|off` | `1` | commutation-aware collection |
| `reps` | `reps=...` | `N` | `10` | sweeps at most |
| `model` | `model=...` | `<model>` | `auto` | auto\|off\|<model id>; list models with describe |
| `beams` | `beams=...` | `N` | `8` | beam width of the policy search |
| `searches` | `searches=...` | `N` | `4` | policy searches per block |
| `metric` | `metric=...` | `cost_depth\|cost` | `cost_depth` | what makes a replacement cheaper: 2q cost then 2q depth, or 2q cost only |
| `strict` | `strict=...` | `on\|off` | `0` | raise QM-E619 instead of keeping a block whose proposal fails verification |
| `budget` | `budget=...` | `N` | `0` | search node cap (0 = the method's own) |
| `model_dir` | `model_dir=...` | `<model_dir>` |  | model directory, searched first |
| `seed` | `seed=...` | `N` |  | default: the compile seed |

## wrap_angles

bring constant angles into the target's angle bounds (spec gate[min:max]) with Pauli corrections, exact up to the tracked global phase.

| Argument | Written | Values | Default | Notes |
|---|---|---|---|---|
| `bounds` | the value alone | `gate[min:max]` |  | a bound added for this pass |

## post_layout

rename physical qubits to the embedding with the best estimated fidelity under the target calibration (never lowers it; no effect on uniform targets).

| Argument | Written | Values | Default | Notes |
|---|---|---|---|---|
| `call_limit` | `call_limit=...` or the value alone | `N` | `200000` | search steps at most |
