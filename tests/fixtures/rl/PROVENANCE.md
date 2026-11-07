# Provenance of the RL golden fixtures

Produced once, offline, by component C0 of `training/ai_transpiler/MOJO_PORT.md` (section 13.1, 16). Nothing in this directory is generated at test
time and no Mojo test starts Python. The generator scripts are not committed (design decision D11: they are line-level transcriptions); the data here is the arbiter.

## Environment of the one-off generation

A scratch virtual environment built from the documented training requirements (`training/ai_transpiler/requirements.txt` at the commit below):

| package | version |
|---|---|
| qiskit-gym | 0.4.1 |
| twisterl | 0.5.1 |
| torch | 2.14.1 |
| numpy | 2.5.4 |
| qiskit | 2.5.2 |
| safetensors | 0.8.0 |
| Python | 3.12 |

Repository state used for the Python oracles: branch `mojo-rl-training`, commit `780d5da` (the Python files of `training/ai_transpiler/` were run from a `git archive` snapshot of that commit).
Research goldens come from the scratch directory `rlport/{envs_golden,ppo_golden,torch_golden}` produced earlier in the session by the research scripts `envs_ref.py`, `ref_ppo.py`, `ref_np.py` and the `gen_golden.py` files (not committed).

## Compaction rule

JSON, compact separators. A float that is exactly representable in f32 is re-printed with its shortest f32 round-trip decimal (at most 9 significant digits), so a consumer that casts the parsed
double to `Float32` recovers the original f32 bit for bit; any other float (a genuine f64 such as a loss, a learning rate or a probability) keeps its full repr. Integers and strings are untouched.
`verify_base` (scratch) re-parsed every trimmed research golden against its source: all values equal under this rule, 0 mismatches.

Trims: `envs/reset_exact.json` keeps every scramble case, `pauli_tableau` cases with difficulty <= 4 and `pauli_rotations` cases of difficulty 16 and 32 (the d=48 rotation list and the d=6 / d=11 tableaus were dropped,
847,594 -> 151,038 bytes). Every other research golden is kept whole.

## Files

| fixture | source / method | source sha256 (first 16) | source bytes | bytes | note | sha256 of the fixture |
|---|---|---|---|---|---|---|
| `envs/defaults.json` | trimmed copy of `rlport/envs_golden/defaults.json` | 954dd8555d5ec4e7 | 943 | 944 |  | `67fc7c03c70211f8` |
| `envs/gateset_twists.json` | trimmed copy of `rlport/envs_golden/gateset_twists.json` | ea03dd834db84408 | 50145 | 50146 |  | `2e6c01172fefc75f` |
| `envs/inverts.json` | trimmed copy of `rlport/envs_golden/inverts.json` | a19e9d6085c495c9 | 13061 | 12300 |  | `c6f3f094aa0fe38b` |
| `envs/pauli_perm_cases.json` | trimmed copy of `rlport/envs_golden/pauli_perm_cases.json` | fd5613d943143293 | 144967 | 134640 |  | `75b11b9b86938684` |
| `envs/pauli_twists.json` | Python common/symmetry.py pauli_twists(n, actions, 5, edges) with actions from the real qiskit-gym env gateset (train.make_env('pauli', topo)); six topologies | - | - | 26216 | oracle (a) | `e0b6fe464892e1e5` |
| `envs/reset_empirical.json` | trimmed copy of `rlport/envs_golden/reset_empirical.json` | 574ff3aaaa346596 | 4074 | 4075 |  | `1b5d26806c158220` |
| `envs/reset_exact.json` | trimmed copy of `rlport/envs_golden/reset_exact.json` | d6e491ca671e3fc2 | 847594 | 151038 | kept: all scramble cases; pauli_tableau difficulty<=4; pauli_rotations difficulty 16 and 32 | `02a591ca84eca65c` |
| `envs/reward_fma_cases.json` | brute force over (achieved, penalty, count, pauli_layer_reward) with numpy f32 unfused arithmetic versus a float64-exact fused multiply-add; 85 differing combinations found, 32 diverse ones kept plus the documented example | - | - | 7021 | oracle (d) | `1d46557059b4b881` |
| `envs/steps_clifford.json` | trimmed copy of `rlport/envs_golden/steps_clifford.json` | 967bbf159681bde5 | 146248 | 132667 |  | `3f8cf7190e86fc5c` |
| `envs/steps_custom_args.json` | trimmed copy of `rlport/envs_golden/steps_custom_args.json` | 5de8328100c6f5a1 | 71192 | 64907 |  | `f2718f584e71edb7` |
| `envs/steps_linear_function.json` | trimmed copy of `rlport/envs_golden/steps_linear_function.json` | 3210a2102039aa3e | 108066 | 95049 |  | `377fd43b4cc3aee4` |
| `envs/steps_pauli_network.json` | trimmed copy of `rlport/envs_golden/steps_pauli_network.json` | d6336ec0e2e16149 | 240838 | 223238 |  | `36ad9d574d41bbce` |
| `envs/steps_permutation.json` | trimmed copy of `rlport/envs_golden/steps_permutation.json` | df3c80195b66ae17 | 100343 | 91683 |  | `9ade289b3c7db55f` |
| `ppo/gae.json` | trimmed copy of `rlport/ppo_golden/gae.json` | 6ceb05135bce96f1 | 5292 | 2391 |  | `f419b9066e70cb15` |
| `ppo/policy_forward.json` | trimmed copy of `rlport/ppo_golden/policy_forward.json` | d9f99464827ff074 | 18372 | 9727 |  | `8000446664ea61ac` |
| `ppo/ppo_update.json` | trimmed copy of `rlport/ppo_golden/ppo_update.json` | cac98d8d28f02b2d | 42116 | 19078 |  | `03b8c79ef9d04163` |
| `ppo/py_curve.json` | Python synth/train.py 400 s runs (--seed 1 --cores 4, defaults otherwise) of perm grid:3x3 and lf grid:2x3, promotions read from log.jsonl | - | - | 6143 | oracle (f) | `2aa3dd03d15ddbf7` |
| `ppo/rollout_stats.json` | real qiskit-gym 0.4.1 env + real twisterl 0.5.1 PPOCollector / evaluate on fixed explicit f32 weights (E=12, common [8], torch default init x3, torch.manual_seed 101..104); 5,000 single-episode collect() calls and 20 x 1000 evaluate() episodes per (config, difficulty), deterministic and sampled x10 | - | - | 46078 | oracle (b) | `a61797e49959f8f5` |
| `ppo/safetensors_header.json` | trimmed copy of `rlport/ppo_golden/safetensors_header.json` | 1c83d48bc157f0ee | 1130 | 739 |  | `08330bcb836c028b` |
| `ppo/tiny_policy.safetensors` | verbatim copy of `rlport/ppo_golden/tiny_policy.safetensors` | 9d66dc76f30f14ae | 1252 | 1252 | verbatim | `9d66dc76f30f14ae` |
| `route/jobs_stats.json` | Python route/circuits.py build_jobs(31, 2000, max_n=30, scale=S) for S = 0.5 and 1.0, summarised as ECDF knots and exact histograms | - | - | 32578 | oracle (c) | `7b26e2245aa16107` |
| `route/py_bc_reference.json` | Python route/train_route.py (torch) on one fixed d0.bin, seeds 0, 1, 2, --hidden 64 --tau 0.5 --epochs 30; d0.bin built from Python build_jobs(31, 1600, max_n=30, scale=0.5) by the existing route_tool gen (repo HEAD 780d5da, ROUTE_TOOL_CAP=96, '- 1.0 401 heuristic 16') | - | - | 8044 | oracle (e) | `7c8021fb91388e55` |
| `topologies.json` | bench/neutral/targets.py edge builders through route/circuits.py topology() and route/eval_sets.py EVAL_TOPOLOGIES; edge order verbatim | - | - | 10448 | oracle | `465c4f9f9647c372` |
| `torch/adam_cosine_golden.json` | trimmed copy of `rlport/torch_golden/adam_cosine_golden.json` | a2f1e42d87043e5e | 1698 | 1468 |  | `38533f77c376f0d6` |
| `torch/bc_golden.json` | trimmed copy of `rlport/torch_golden/bc_golden.json` | 81e0cd2d27d7e829 | 251358 | 159095 |  | `15a276afa9a61b45` |
| `torch/f16_golden.json` | trimmed copy of `rlport/torch_golden/f16_golden.json` | ca3d35c132d8f7c6 | 9189 | 9190 |  | `7ef44210b9bd9017` |
| `torch/misc_golden.json` | trimmed copy of `rlport/torch_golden/misc_golden.json` | a76679442c19efd3 | 1775 | 1652 |  | `b68271c34fc5ddcd` |
| `torch/ppo_route_golden.json` | trimmed copy of `rlport/torch_golden/ppo_route_golden.json` | aee551081c2cce5d | 268562 | 166746 |  | `02372b954a557394` |
| `torch/synth_ppo_golden.json` | trimmed copy of `rlport/torch_golden/synth_ppo_golden.json` | d184a63f4f4d340b | 50670 | 32536 |  | `337351afbf5ef064` |

Total fixture payload: 1501089 bytes (budget 2,000,000).

Spec archive: `training/ai_transpiler/spec/{ENVS,PPO,TORCH,REPO}.md` are verbatim copies of the research reports `envs.md`, `ppo.md`, `torch.md`, `repo.md`.

## Deviations from the size targets of section 13.1

The total is far below the 2 MB budget, so the per-file size targets were treated as soft where trimming would have cut test power: `envs/steps_*.json` (92-223 KB, target 60 KB; all 24 start states per config kept),
`envs/steps_custom_args.json` (65 KB, target 40 KB), `torch/bc_golden.json` (159 KB, target 120 KB), `torch/ppo_route_golden.json` (167 KB, target 140 KB), `route/py_bc_reference.json` (full per-epoch logs, target 4 KB),
`ppo/py_curve.json` (all promotions, target 4 KB). `tests/test_rl_fixtures.mojo` pins the ceilings actually shipped.

## Custody items that are not committed

* `d0.bin` of the BC reference (50,595 records, 16,595,160 bytes, sha256 `7ff924e3e13515fe9f81ebcb7983d5c6b6dfd0c34221b808e8f24502f178a58f`), built from `jobs0.json` (sha256 `d4d753d4e5223c864c267ee6228a5b728791f11ad81bb89ce28d05e3fdb2acc5`). It is kept in the lead's scratch directory (`.../scratchpad/c0/d0.bin`) and is the input of acceptance check A6(b) (`QUMO_BC_REF_D0`).
* The Python generator scripts and the scratch virtual environment.

## Reading notes for consumers

* Each oracle fixture documents its own schema in a `doc` field (what is recorded, units, and the intended test statistic).
* `ppo/rollout_stats.json` is a statistical oracle: the Rust collector draws its twist and its Gumbel noise from an unseeded thread RNG, so no seed reproduces it; compare at 4 sigma with the standard errors recorded.
* `envs/reward_fma_cases.json` discriminates fused from unfused arithmetic; the Rust env and numpy f32 give the `unfused` value.
