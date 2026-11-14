# QUMO RL-training port to native Mojo: repo porting map

Worktree: /Users/ismaelsertage/code/QUMO/.worktrees/mojo-rl-training (branch mojo-rl-training).
Scope of this document: what in `training/ai_transpiler`, the Makefile training targets and
`docs/AI_TRANSPILER.md` depends on Python / Rust / numpy / torch / safetensors / qiskit /
gymnasium / qiskit-gym / twisterl, what exactly each call must do, which Mojo module replaces it,
the byte-level data formats, and the size of the native tensor / NN / optimizer library.
Package-internal semantics of qiskit-gym's Rust environments and twisterl's collector are
summarised only where the repo code exposes them; the scratch dirs `twisterl_src/` and the venv
were used to confirm PPO/BasicPolicy facts (marked "pkg"). Anything marked **[GAP]** must be
settled from the qiskit-gym Rust source (other workflow agents).

All line numbers refer to the files as of this worktree (HEAD e0ad469).

---------------------------------------------------------------------------------------------

## 0. Executive summary

* Inference is already 100 % Mojo (policy.mojo, policy_synth.mojo, ai_route.mojo). What needs
  Python is TRAINING + EXPORT + EVAL ORCHESTRATION: 17 Python files, ~3200 lines
  (`common` 1028, `route` 1707, `synth` 466), plus an inline `python3 -c` in the Makefile
  (`models-verify`) and two `python -c` / `python script` calls inside `train-ai-route`.
* Two training algorithms exist:
  1. **Router** (`route/`): behaviour cloning + one DAgger round on a rollout teacher. The
     expensive part (teacher rollouts, env) is ALREADY Mojo (`route_tool.mojo gen`). Remaining
     Python: circuit/topology job generator, the BC trainer (torch, a 128->64->16 MLP), the
     exporter, parity check. This is small: a hand-written forward/backward fits in ~300 lines.
  2. **Synthesis** (`synth/`): qiskit-gym 0.4.1 Rust environments + twisterl 0.5.1 Rust PPO
     collector + torch `BasicPolicy` + Adam. This is the large item: needs a native PPO
     (collector, GAE, curriculum, evals) around the EXISTING Mojo `PolicyEnv`/`MLPPolicy`, plus
     the env training extras that `PolicyEnv` does not have (scrambling by difficulty, reward,
     masks, random inversions / symmetries) **[GAP: exact qiskit-gym semantics]**.
* The NN needed is tiny and fixed-shape: sparse embedding bag (obs 36..144 -> 512), ReLU, one
  or more dense ReLU layers (512->256), linear action head (A = 10..29), linear value head (1);
  router: sign-embedding 128->H(64), ReLU, linear 16 + unused value head. A purpose-written
  native library (no autograd): `EmbeddingBag`, `Linear`, `relu`, `(log_)softmax`, masked
  cross-entropy / PPO loss with hand-derived gradients, torch-style Adam, cosine LR, torch-style
  init, a seeded RNG. ~900-1100 lines of Mojo plus ~500 of tests.
* Things with NO Mojo equivalent today and that must be written: SHA-256, f32->f16 encoding
  (decode exists), Python-`random`-style circuit/job generation (or a re-seeded equivalent),
  topology generators of `bench/neutral/targets.py` (heavy-hex already exists in `Target`),
  automorphism enumeration / symmetry maps, container + INDEX.json writers, an inline
  `models-verify`, a Mojo orchestrator replacing `train_all.sh` (which does not exist, see 7).
* Reproducibility note (important for the port design): the existing training is explicitly
  NOT bit-reproducible (twisterl uses an unseeded `thread_rng`; README line "Training is not
  bit-reproducible"). Models are fixed hashed artefacts. So the native port does not need to
  reproduce Python RNG streams (Mersenne Twister, PCG64, torch Philox/MT) bit-for-bit; it should
  use the seeded `qumo.sim.rng.Rng` everywhere and become deterministic per (seed, threads).
  Exception worth keeping: golden comparison tests against the COMMITTED models/manifests
  (those hold Python-derived expectations and are the independent oracle).

---------------------------------------------------------------------------------------------

## 1. File inventory and disposition

Legend: PORT = needs a native replacement; RETIRE = delete (a Mojo equivalent already exists or
its only purpose is Python<->Mojo parity); KEEP = already Mojo.

### 1.1 Already Mojo (no port needed)

| file | lines | role | note |
|---|---:|---|---|
| `qumo/compiler/policy.mojo` | 946 | container loader (`load_model_file`, f16 decode, FNV-1a64, `wl_hash`, `fnv_words`), `MLPPolicy` forward (embedding bag, SIMD dense), symmetry ensemble, INDEX reader, graph embedding | exports `fnv_words`, `wl_hash`, `hex16`, `f16_to_f32` that the exporter can reuse |
| `qumo/compiler/policy_synth.mojo` | 1326 | `PolicyEnv` (all 4 kinds: step/observe/solved/inverse/key/init_pauli), beam / greedy / sampled search, Pauli algebra | is the Mojo twin of `refenv.py` |
| `qumo/compiler/ai_route.mojo` | 1921 | `RProblem`, `Episode` (full env), `RouteModel`, `load_route_model`, SABRE-style heuristic, verify | Mojo twin of `route_env.py`; `Episode.rec_ops=False` already exists for training rollouts |
| `training/ai_transpiler/route/route_tool.mojo` | 462 | `gen` (teacher/DAgger data), `eval`, `episode` | consumes jobs.json produced by Python `circuits.py` |
| `training/ai_transpiler/synth/eval_blocks.mojo` | 280 | policy vs greedy/auto/exact on 100 seeded blocks, verification, JSON summary | called as a subprocess by `eval.py` |
| `tests/test_policy.mojo` | 344 | loader/parity/determinism, replays embedded test vectors of fixtures AND `models/ai_transpiler/synth` | |
| `tests/test_ai_route.mojo` | 433 | replays `test_vectors.obs_logits` and `test_vectors.episodes` of the committed router, statevector equivalence | |
| `qumo/sim/rng.mojo` | | xoshiro256** + SplitMix64, `uniform`, `below`, `normal` (Box-Muller, 2 uniforms/call), `split` | |
| `qumo/algorithms/optimizers.mojo` | | `Adam` (Float64, whole-problem `minimize`, not stepwise) | not usable as-is for NN training |
| `qumo/algorithms/qml_train.mojo` | | `StepOptimizer.adam(... form="optax")` (stepwise, Float64), `cosine_lr(step,total,lr0)`, `Batcher`, `cross_entropy` (takes probabilities, hard labels) | formulas reusable; need f32 tensor-level versions + soft-target CE on logits |
| `qumo/core/json.mojo` | | `parse_json`, `JsonValue`, `JsonWriter` (compact), `pretty_json` | exporter can build manifests with `JsonWriter` |
| `qumo/compiler/target.mojo` | | `Target.heavy_hex_edges(d)` == rustworkx `heavy_hex_graph(d)` edge-for-edge; `from_edges` (dedup a<b insertion order, CSR, BFS dist) | `bench/neutral/targets.py` heavy_hex_edges(min_qubits) = smallest odd d with (5d^2-2d-1)/2 >= min |

### 1.2 Python files (all 17): disposition

| file | lines | purpose | disposition | Mojo target module (proposed) |
|---|---:|---|---|---|
| `common/export.py` | 166 | container writer/reader, INDEX.json, state_dict -> tensor list | PORT | `qumo/rl/container.mojo` (+ `sha256.mojo`) |
| `common/topology.py` | 155 | topology spec parser, slug, connectivity, `wl_hash`, FNV | PORT parse/slug/connected; wl_hash+FNV DUPLICATE of policy.mojo (reuse, add golden test) | `qumo/rl/topology.mojo` |
| `common/symmetry.py` | 99 | graph automorphisms -> (obs_perms, act_perms) incl. Pauli ensemble maps | PORT | `qumo/rl/symmetry.mojo` |
| `common/refenv.py` | 450 | numpy reference of policy forward, 4 envs, beam, greedy trace, emit, cost | RETIRE (Mojo `MLPPolicy`/`PolicyEnv`/`_beam`/`greedy_trace` are the same algorithms); the exporter calls the Mojo ones | - |
| `common/make_fixtures.py` | 158 | tiny random-weight fixture models for tests | PORT (or freeze: fixtures are committed) | `training/ai_transpiler/common/make_fixtures.mojo` |
| `synth/train.py` | 146 | time-bounded PPO with qiskit-gym + twisterl + torch | PORT (biggest) | `qumo/rl/ppo.mojo`, `qumo/rl/synth_env.mojo`, `training/ai_transpiler/synth/train_synth.mojo` |
| `synth/export_model.py` | 199 | run dir -> `qumo.rlsynth.v0` model, test vectors, rollouts, INDEX | PORT | `training/ai_transpiler/synth/export_synth.mojo` (library part in `container.mojo`) |
| `synth/eval.py` | 121 | export to staging, run `eval_blocks`, keep/drop rule, optional Qiskit O3 baseline | PORT orchestration; DROP Qiskit baseline from the training path (see 3.3) | `training/ai_transpiler/synth/eval_synth.mojo` (in-process call of the eval_blocks logic) |
| `route/route_env.py` | 720 | bit-exact Python mirror of the Mojo router env | RETIRE | - |
| `route/parity.py` | 83 | Python<->Mojo episode parity | RETIRE (tests/test_ai_route.mojo + committed golden vectors cover it) | - |
| `route/circuits.py` | 248 | seeded circuit families + 10 topology families + `build_jobs` | PORT | `qumo/rl/jobs.mojo` + `route_tool.mojo gen-jobs` |
| `route/eval_sets.py` | 74 | held-out eval job sets (synth seeds >= 910000, bench/neutral/circuits) | PORT | `route_tool.mojo eval-sets` |
| `route/train_route.py` | 179 | BC / DAgger trainer (torch) | PORT | `training/ai_transpiler/route/train_route.mojo` |
| `route/train_ppo.py` | 192 | optional PPO fine-tune of the router; README says it made things worse and was dropped | PORT as phase-3 optional, or drop (decision needed, 6) | reuse `ppo.mojo` |
| `route/export_npz.py` | 53 | run dir -> model dir with parity episodes | PORT | `training/ai_transpiler/route/export_route.mojo` |
| `route/export_route.py` | 99 | writer for `qumo.rlroute.v0`, obs->logits vectors, episode vectors | PORT | same |
| `route/summarize_eval.py` | 59 | geometric-mean tables over `eval` JSONL | PORT (pure json/math) or fold into `route_tool eval --summary` | `route_tool.mojo summarize` |
| `requirements.txt` | 10 | qiskit-gym, twisterl, torch, safetensors, numpy, qiskit, gymnasium | DELETE at the end; note it does not list rustworkx although `circuits.py` -> `bench/neutral/targets.py` imports it (README says "torch + numpy + rustworkx") |
| `README.md`, `route/README.md` | | docs | REWRITE |

Also outside `training/`: `Makefile` lines 66-189 (`AI_TRAIN_PY`, `models-verify`, `train-ai*`),
`docs/AI_TRANSPILER.md` sections `models` (lines ~158-186) and `training` (188-221) plus
the line-440-448 "qiskit-gym / twisterl / torch are needed only to retrain" note, `.gitignore`
lines 51-60 (venv/safetensors patterns; harmless), `NOTICE` line 27 (credits stay: algorithmic
derivation from qiskit-gym/twisterl remains true).
`bench/neutral/targets.py` is imported by `circuits.py` (needs `rustworkx` for heavy-hex only; the
other generators are pure python/math). That file is a benchmark input shared with other code;
do NOT delete it, port the generators to Mojo and (optionally) golden-test against it.

---------------------------------------------------------------------------------------------

## 2. Third-party calls, file by file (exact semantics to reproduce)

### 2.1 common/export.py

numpy:
* `np.asarray(x, dtype=np.float32)`, `np.ascontiguousarray` - row-major f32.
* `arr.astype("<f2")` - IEEE binary16, round-to-nearest-even, overflow -> inf (then
  `np.isfinite` check raises `"tensor X overflows f16"`), subnormals kept. Mojo: `Float16(x)`
  conversion (hardware RNE) or hand-rolled; verify with an exhaustive 65536-value round trip vs
  the existing `f16_to_f32`.
* `.astype("<f4")`, `.tobytes(order="C")` - little-endian raw bytes.
* `np.array_equal`, `np.frombuffer(raw, dtype, count, offset)` - self-check read-back.
* `np.prod(shape)` - element count.
stdlib: `hashlib.sha256` (over the whole `weights.bin`), `json.dump(separators=(",",":"))` for
manifests, `json.dump(indent=1)` for INDEX.json, `os.listdir` (sorted), `os.path.getsize`.
torch (only in `basic_mlp_tensors`): `v.detach().cpu().float().numpy()`; `.T` + `ascontiguousarray`
to store `embeddings.weight` as [OBS, EMB].
**Quirk to fix, not copy:** `basic_mlp_tensors` renames `common.<j>` with j = the torch
`Sequential` index. `make_sequential` interleaves ReLU modules, so for >1 common layer (or non-empty
policy/value layer lists) the indices are 0, 2, 4, ... while `MLPPolicy._seq` and `ModelFile`
require consecutive k = 0, 1, 2... All shipped models have exactly one Linear per sequence, so the
bug is dormant. Native writer must number layers consecutively.

### 2.2 common/topology.py, symmetry.py
Pure Python/stdlib; `hashlib.sha1` only in `topology_slug("edges:..")` (file-name slug - can use
FNV instead). `automorphisms` = backtracking over degree/adjacency-consistent images, result
`sorted(out)` (lexicographic over permutation lists). Verified now: re-running `twists()` on the
9 shipped manifests reproduces `symmetries.obs_perms/act_perms` **exactly in order** for all 9
(perm/lf/pauli, grid/ring/line/king/heavyhex_frag). That is the golden test for the Mojo port.
Action maps: SWAP keys normalised (min,max); CX keys directional; `None` (-> skip that
automorphism) if the image of an action is not an action. Pauli: `pauli_twists` returns the
inverse-style ensemble maps (`obs_perm` maps actual dense index to permuted-frame index; rows and
tableau columns permuted, rotation columns not; `act_perm` = inverse of the action perm).
Convention (twisterl): `obs' = obs_perms[p][i]`, `logits_out[a] = raw[act_perms[p][a]]`.

### 2.3 common/refenv.py (RETIRE; used only to produce test vectors)
numpy used: int64 index arrays, `h += WeT[i]` (f32), `np.maximum`, `W @ h` (f32 matvec through
BLAS sgemv: summation order differs from Mojo's 8-lane SIMD - tests allow 1e-5), log-softmax in
f64, `np.exp`. Semantics = `MLPPolicy._raw/forward_ensemble` + `_beam`/`greedy_trace` already in
Mojo. After retirement the exporter's `test_vectors` and `rollouts` come from Mojo code; to keep an
independent oracle the committed models' embedded vectors (made by the Python reference) stay as
golden tests (`tests/test_policy.mojo::test_trained_models`).

### 2.4 common/make_fixtures.py
numpy: `np.random.default_rng(1000+k)` (PCG64) `.normal(0, sigma, shape)`, `.integers`,
`r2.integers(len(actions))`. Writes `tests/fixtures/ai/policy/synth/<id>/{manifest.json,weights.bin}`,
`synth/INDEX.json`, plus `bad_checksum` (byte 100 of weights.bin XOR 1) and `bad_format`
(`format = "qumo.rlsynth.v9"`). The fixtures are committed; tests only replay embedded vectors, so
a Mojo regenerator may use its own RNG (the files change once, then stay frozen). FIXTURES list:
`fx_perm_line3` (f32), `fx_perm_star4`, `fx_lf_line3`, `fx_cliff_line2`, `fx_pauli_line3`
(max_rot 2); E = 16, common [16]; `max_depth` 40.

### 2.5 synth/train.py  (the heavy one)
qiskit-gym 0.4.1 (pkg-confirmed in venv):
* `qiskit_gym.envs.synthesis.{PermutationGym, LinearFunctionGym, CliffordGym, PauliGym}
  .from_coupling_map(cm, basis_gates=...)`: builds `gateset` = for each basis gate in order: 1q
  gates for q in 0..n-1, or 2q gates for each edge of the SORTED coupling list (SWAP: one
  direction a<b only; CX/CZ: both directions, whole list sorted). Config defaults:
  `difficulty=1, depth_slope=2, max_depth=128, metrics_weights=None, add_inverts=True,
  add_perms=True`. `num_qubits = max qubit + 1` (checked against the spec).
* `env.config["num_qubits"]`, `env.obs_shape()`, `env.num_actions()`, `env.twists()` -> (obs_perms,
  act_perms) (len = number of automorphisms of the action set), `env.difficulty` (settable int;
  curriculum counter). `PauliGym` has `max_rotations` default 5 (train.py's `max_rotations=5`
  argument of `make_env` is unused).
* The Rust envs (`qiskit_gym_rs.{Permutation,LinearFunction,Clifford,PauliNetwork}Env`) provide the
  training dynamics: `reset()` (scramble of `difficulty` random actions, inversions, random
  symmetry), `step`, `observe` (sparse index list, identical to refenv.observe), `masks`,
  `reward`, `is_final`, `success`, `set_state`. **[GAP]** the exact scramble length rule
  (depth_slope * difficulty?), reward values, final conditions, add_inverts / add_perms
  behaviour and Pauli rotation sampling must be taken from the Rust source. What the repo already
  fixes: observation/state/step/solved (policy_synth.mojo header documents them: they are exact
  mirrors of qiskit-gym 0.4.1).
* `RLSynthesis(env, ppo, pol)`, `rls.save(config.json)` (JSON: env_cls, env{...}, policy_cls,
  policy{embedding_size, common_layers, policy_layers, value_layers}, algorithm_cls,
  algorithm{collecting{num_cores,num_episodes,lambda,gamma}, training{num_epochs,vf_coef,ent_coef,
  clip_ratio,normalize_advantage}, learning{diff_threshold,diff_max,diff_metric}, optimizer{lr},
  evals{...}, logging}); `rls.algorithm` = twisterl `PPO`.
* `PPOConfig(num_cores=cores, num_episodes=--episodes(1024), evals=..., diff_max=--diff-max(256))`;
  everything else default: `gae_lambda=0.995, gamma=0.995, num_epochs=10, vf_coef=0.8,
  ent_coef=0.01, clip_ratio=0.1, normalize_advantage=False, lr=3e-4, diff_threshold=0.85,
  diff_metric="ppo_deterministic"`. Evals: `ppo_deterministic` (200 episodes, argmax, 1 search),
  `ppo_10` (200 episodes, sampled, 10 searches, keep best).
* `BasicPolicyConfig(embedding_size=--emb(512), common_layers=[256], policy_layers=[], value_layers=[])`.
* `algo.learn_step()` -> `(times, bench, train), total_time` where per step: (1) sync policy to
  Rust, (2) run every eval (`evaluate(env, rs_pol, **cfg)` -> (success_rate, mean_reward)), bench =
  {successes{name}, rewards{name}, difficulty, success, reward} with `success/reward` taken from
  `diff_metric`, (3) PPO `collect`, (4) tensor conversion, (5) `train`: `num_epochs=10`
  **full-batch** gradient steps (no minibatches) of Adam on the whole collected batch, returns the
  last step's {value, policy, entropy, total} losses.
* Driver loop (train.py): while elapsed < `--budget-s`: learn_step; `cur=(difficulty, success,
  reward)`; checkpoint `model.safetensors` when `cur >= best` (tuple compare); if `success >=
  diff_threshold(0.85)` and `difficulty < diff_max` -> `env.difficulty += 1`; write `log.jsonl`
  record {it, t, step_s, difficulty, succ (dict), rew (dict), loss}; print every 25 its. After the
  loop `model_last.safetensors` and `summary.json`.
* Continuation convention: output dir named `*__cont` warm-starts from sibling dir without the
  suffix: loads its `model.safetensors`, `start_diff = summary.json["final_difficulty"]`.
  `--init` explicit path likewise. `at most 8 cores` guard.
* `torch.manual_seed`, `torch.set_num_threads`: only relevant for init + the training matmuls.
twisterl 0.5.1 (pkg-confirmed, `twisterl_src/`):
* `BasicPolicy.__init__`: `embeddings = nn.Linear(obs_size, emb)`; `common = make_sequential(emb,
  [256])` = Linear+ReLU; `action = make_sequential(256, policy_layers + (A,), final_relu=False)`;
  `value = make_sequential(256, value_layers + (1,), final_relu=False)`. torch default init for
  every Linear: W, b ~ U(-1/sqrt(fan_in), 1/sqrt(fan_in)) (kaiming_uniform a=sqrt(5)).
* Forward: `relu(embeddings(x))`, `common(.)`, then heads. With symmetries `forward(x, perm_idx)`:
  gather obs columns by the inverse perm (`argsort(obs_perms[p])`), forward, then
  `logits.index_select(1, act_perms[p])`; **without** `perm_idx`: random perm per sample
  (`randint`). During PPO the Rust collector records `perm_idx` per step so training reuses the
  same symmetry the actor used (`forward_with_perm`).
* PPO collector (Rust, `collector/ppo.rs`): per episode: `env.reset()`; loop {obs, masks, reward =
  `env.reward()` of the CURRENT state, `(logits, value, perm_idx) = policy.forward_with_perm`,
  `action = sample_from_logits`, push; break if `is_final`; `env.step(action)`}; then GAE:
  `adv[n-1] = r[n-1]-v[n-1]; ret[n-1]=r[n-1]; for t=n-2..0: ret[t]=r[t]+gamma*(v[t+1]+lambda*adv[t+1]);
  adv[t]=ret[t]-v[t]`. `rayon` pool with `num_cores`; RNG unseeded.
* Training step (python `ppo.py`): dense one-hot obs `[B, obs_size]`; `pt_log_probs =
  Categorical(logits=stored_logits).log_prob(acts)` (old logits recorded at collection time);
  new `pred_logits, pred_vals = policy(obs, perm_indices)`; ratio = exp(new_lp - old_lp);
  `policy_loss = -mean(min(ratio*adv, clamp(ratio, 0.9, 1.1)*adv))`;
  `value_loss = mse(pred_vals[B,1], rets[B,1])` (mean); `entropy = Categorical(logits).entropy().mean()`;
  `loss = policy_loss + 0.8*value_loss - 0.01*entropy`; `Adam(lr=3e-4)` step. NOTE: action
  masks are NOT applied in the training loss (only the Rust sampler gets `masks`).
* Evaluate / solve (Rust `rl/evaluate.rs`): argmax or sampled episodes, `num_searches` repeats
  keep best; returns (success_rate, mean_reward). **[GAP]** (mask handling, tie-breaking).
safetensors: `save_file(state_dict, path)` / `load_file`. State-dict keys: `embeddings.weight`
[EMB, OBS], `embeddings.bias`, `common.0.weight/bias` [256,EMB]/[256], `action.<i>.weight/bias`
[A,256], `value.<i>.weight/bias` [1,256] (f32 little-endian). Non-persistent buffers (perm tensors)
are not saved. Format in 4.6.

### 2.6 synth/export_model.py
* stdlib json; `safetensors.numpy.load_file(model.safetensors)`; `numpy` (tensors, `default_rng(20261014)`
  `.integers`, `np.zeros/flatnonzero` in the Pauli self-check).
* qiskit-gym: `SYNTH_ENVS[cls].from_json(envcfg)` + `genv.twists()` to cross-check `symmetry.py`
  (`"symmetry.py disagrees with qiskit-gym twists"`). Not reproducible without qiskit-gym; replace
  by the golden test of 2.2 (the 9 shipped manifests) - the cross-check already proved equal.
* From `config.json`: `cfg["env_cls"]` -> kind map {PermutationEnv: permutation, LinearFunctionEnv:
  linear_function, CliffordEnv: clifford, PauliNetworkEnv: pauli_network}; `envcfg["num_qubits"]`,
  `envcfg["gateset"]` -> actions list `[GATE_UPPER, q0, q1 or -1]`; `edges = sorted({(min,max)})`
  of the 2q gates; `max_rot = envcfg.get("max_rotations", 5)` for Pauli.
* Manifest fields written (order, compact JSON): format, id, kind, num_qubits, edges, actions, obs
  {encoding:"sparse_index", size, layout, max_rotations}, max_depth (--max-depth 128), arch
  {type:"basic_mlp", emb, common[], act:"relu", first_layer_transposed:true}, symmetries{obs_perms,
  act_perms}, training{producer, algo:"PPO (twisterl)", seed, budget_s, wall_s, cores, iterations,
  final_difficulty, topology_spec, eval{}, [continued_from, total_budget_s, total_wall_s]}, license;
  then writer: tensors, weights_file, weights_bytes, fnv1a64, sha256; then test_vectors (6),
  rollouts (2).
* Layout strings: perm/lf `"{n}x{n} row-major: perm i*n+state[i] / matrix bits"`, obs size n*n;
  clifford size 4n^2 `"{2n}x{2n} symplectic of C^-1, rows = coords (x..., z...), cols = generators"`;
  pauli size 2n*(2n+max_rot) `"{2n}x{2n+max_rot}: tableau of C^-1 + first {max_rot} active rotation columns"`.
* Test vectors: 6 `random_start` states (random walk of 3+4i actions from identity; Pauli: random
  rotations count 1..max_rot+2, weights as masks, random Clifford walk, `pauli_start`, `clean`), each
  {obs (sparse idx list), logits (raw), value, logits_ensemble, value_ensemble}; rollouts: 2 starts
  (depth 6+6i) -> `greedy_trace(env, pol, st, max_depth)` = {input (packed state ints), R, actions,
  solved}. All from the STORED (f16-rounded) weights.

### 2.7 synth/eval.py
* subprocess: `export_model.py` (stage), `eval_blocks` binary (Mojo), `mojo build`.
* `qiskit_baseline` (optional, wrapped in try/except, only n <= 9 and not Pauli): Qiskit 2.5.2
  `transpile(level 3, basis [cx,u], trivial layout, seed 7)` of `LinearFunction(random_invertible_binary_matrix)`,
  `PermutationGate`, `random_clifford(...).to_circuit()` on the model graph, final-layout fix
  cost by BFS (3 CX per SWAP); mean cost over <= 50 instances; recorded in `eval_final.json` as
  `qiskit_o3_unpaired` and stored in the manifest `training.eval`. Not on the training critical
  path; recommendation: drop it from the native pipeline and document the Qiskit comparison as
  living in `bench/neutral` (still Python, a benchmark not training). The rule itself:
  `keep = win_or_tie >= 0.6 and cost_ratio < 1.0 and fails == 0` using
  `policy_vs_greedy` from eval_blocks JSON (`policy_beam8` vs `greedy`, beams 8 + 4 samples).
* Run flow: export f16 to staging -> eval_blocks -> JSON `eval_blocks.json` -> `eval_final.json`
  (+ `kept`, `rule`) -> if kept re-export to `models/ai_transpiler/synth` with `--eval-json`.

### 2.8 route/circuits.py
* stdlib `random.Random(seed)` (Mersenne Twister): `random()`, `shuffle`, `randrange`, `randint`,
  `choice`; `math.isqrt`. `build_jobs(seed, n_jobs, topologies, families, max_n, scale, min_frac)`
  generates jobs.json (format in 4.4). With the CPython MT algorithm replaced by `qumo Rng`, the
  job set differs from the shipped campaign's, which is fine (see section 0).
* `bench/neutral/targets.py` generators (pure python except `rustworkx.generators.heavy_hex_graph`):
  `heavy_hex_edges`, `grid_edges(min_qubits, radius)` (hypot <= radius + 1e-9, i-major pairs),
  `diagonal_grid_edges`, `octagonal_edges`, `ion_zone_ring_edges(ions_per_zone=2)`, `tree_edges`
  (complete binary tree, parent (i-1)//2), `torus_edges`. Edge EMISSION ORDER matters (it defines
  edge indices = active-swap order in the env) - port literally. In circuits.py `line`, `ring`,
  `grid` are built locally. Training topologies (25): grid 3x3,4x4,5x5,6x6,3x5; heavy_hex 19,57;
  diag 16,25; octagonal 16,32; line 8,12,20; ring 8,14,20; tree 15,31; torus 16,25; atom2 16,25;
  ionring 12,16. Eval topologies (10): grid 4x4, heavy_hex 19, diag 16, octagonal 16, line 16,
  ring 16, tree 15, torus 16, atom2 16, ionring 16.
* Mojo already has `Target.line/ring/grid/torus/tree/heavy_hex`; check `Target.tree/torus` differ
  from the neutral-benchmark variants (tree: levels so that 2^(l+1)-1 >= min; torus: side >= 3) and
  port the neutral ones verbatim rather than reuse blindly.
* Families: rand2q, qv, qft, trotter, qaoa, sparse (parameters in file); ops are
  `(kind, a, b)` with kinds `cx cz swap u2 u1 measure barrier`; Mojo maps `u2 -> rzz(0.3)`,
  `u1 -> h` (route_tool.mojo `circuit_from_json`).
* Makefile uses `C.build_jobs(31, 1000, max_n=30, scale=0.5)` (round 0) and `(32, 1000, ...)` (DAgger).

### 2.9 route/eval_sets.py
`zlib.crc32(filename)` (-> need CRC-32/IEEE, 15 lines), `re.findall(r"qreg\s+\w+\[(\d+)\]")` to
count qubits of `bench/neutral/circuits/*.qasm` (Mojo: `load_qasm(path).num_qubits`; note the
regex SUMS all qregs, equal to total qubits), `os.listdir` sorted, `json.dump`. synth_set: per
eval topology 8 jobs at seed `910000 + ti`, `max_n=min(20, N)`; bench_set: circuits with 3 <= n
<= 14 on 3 topologies `(crc32(name) + 3k) % 10`, seed 7.

### 2.10 route/train_route.py  (torch BC)
See 5.1 for the full numeric spec. Third-party calls: `np.fromfile` structured dtype (4.2),
`np.concatenate`, `np.where/nanmax/nanmin`, `np.sign`, `np.minimum`, `np.random.default_rng(seed)
.permutation`, `torch.randn/zeros/nn.Linear/relu/matmul`, `torch.argsort(torch.rand(B,16, generator=g))`,
`torch.gather`, `masked_fill`, `softmax/log_softmax`, `torch.optim.Adam(lr 2e-3)`,
`CosineAnnealingLR(T_max=epochs)`, `torch.randperm(N, generator=g)`, `loss.backward()`, `np.savez`.
Also `export.read_model` for `--init`.

### 2.11 route/export_route.py, export_npz.py
`np.load(weights.npz)` (keys emb,b0,w1,b1,wv,bv), `np.random.default_rng(0)`
`.integers(-3, 4, size=128)` and `.random(128) < 0.5` for 6 obs vectors; `random.Random(7)`
`.shuffle(layout)`, `.randrange(1<<40)`; `C.build_jobs(424242, neps, topologies=[("grid",(3,3)),
("heavy_hex",19)], max_n=9)`; `E.Episode(...).run(model, MODE_POLICY)` (Python mirror) for the
parity episodes. `neps=2` default (`--episodes N`); committed model has 6 obs vectors + 2 episodes.
`RouteMLP.logits` is float64: `h = b0 + sign(obs) @ emb; logits = w1 @ max(h,0) + b1` using the
stored f16-decoded weights. INDEX update then `default` flag fix-up (see quirk 7.4).

### 2.12 route/parity.py, route_env.py (RETIRE)
numpy: `RTarget.D` int64 matrix BFS, `default_rng(3)` normal weights. Mirror of `ai_route.mojo`.
All needed semantics live in the Mojo `Episode`; one thing is MISSING in Mojo and used by PPO only:
the split `pre_decision(limit) / needs_policy() / act(e)` (Mojo has only the fused `step`).

### 2.13 route/train_ppo.py (optional)
numpy: `default_rng(seed).choice(A, p=p)` (inverse-CDF sampling), softmax, GAE in numpy, `np.sign`;
torch: Adam(lr 2e-4), `clip_grad_norm_(0.5)`, clamp/min/exp, `torch.randperm`;
`multiprocessing` fork Pool with `--workers 8`; reward per policy decision =
`-(overhead increase incl. fallback swaps until next decision) + 0.01*(2q gates executed)`; GAE
gamma 0.99, lam 0.95 with episode-final `D`; advantages normalised; 4 epochs of minibatch 1024;
`vl = mse(v, ret)`, loss `pl + 0.5 vl - ent_coef(0.005)*ent`; curriculum `scale = min(1.5, 0.4+0.02*it)`;
jobs `build_jobs(700000+1000*seed+it, episodes, max_n=24, scale)`.

### 2.14 route/summarize_eval.py
stdlib only (json, math, collections). Port is mechanical.

### 2.15 Makefile (lines 66-189)
* `AI_TRAIN_PY ?= training/ai_transpiler/.venv/bin/python` (every train target uses it).
* `models-verify`: `python3 -c "$$MODELS_VERIFY_PY"`, stdlib-only (hashlib, json, os). Walks
  `$QUMO_MODEL_DIR` or `models/ai_transpiler`, supports 5 INDEX layouts, flags `.bin` files not
  listed in any INDEX, prints `models-verify: N INDEX.json, M files ok, K problems, S MB total`.
  Needs a native SHA-256 to leave Python; recommended as `qumo models verify` / a Mojo tool.
* `train-ai-synth` -> `training/ai_transpiler/synth/train_all.sh` (NOT PRESENT, see 7.1).
* `train-ai-route`: `MOJO build route_tool`; `parity.py`; inline `python -c` for jobs0/jobs1;
  `route_tool gen` x2 with `ROUTE_TOOL_CAP=96`; `train_route.py` x2; `export_npz.py` x2.
  Parameters: gen args `<jobs> <out.bin> <model|-> <beta> <seed> <rollout> <G>`: round 0 `- 1.0 401
  heuristic 16`, DAgger `stage/round0 0.3 402 heuristic 16`; train `--hidden 64 --tau 0.5
  --epochs 30 --seed 0` (round 1 trains on d0.bin + d1.bin); export `--default`, id
  `route_mlp_h64_v1` (`AI_ROUTE_ID`). `AI_ROUTE_BUDGET_S` is declared but never used by the recipe.

---------------------------------------------------------------------------------------------

## 3. numpy / torch feature matrix (library sizing)

### 3.1 Needed operations

| feature | used by | semantics |
|---|---|---|
| f32 row-major matrices, vectors | all | `List[Float32]`/`UnsafePointer`, contiguous |
| sparse EmbeddingBag fwd | PPO policy (obs idx list, B variable nnz), router (sign in {-1,0,+1}) | `h[b,:] = bias + sum_{i in obs_b} We_T[i,:]`; router: `+/-` row add; invalid idx ignored in inference |
| sparse EmbeddingBag bwd | both | `dWe_T[i,:] += g_h[b,:]` (scatter-add, sign for router); `dbias += g_h` |
| dense Linear fwd/bwd | common (512x256), heads (256xA, 256x1), router out (64x16), value | `y = x W^T + b`; `dW = g^T x`, `dx = g W`, `db = sum g`; row-major SIMD, f32 accumulate |
| ReLU fwd/bwd | all | `y = max(x,0)`; grad passes where `x > 0` (torch: grad 0 at x==0) |
| softmax / log_softmax / logsumexp | BC, PPO | max-subtract, f32 (torch) |
| masked fill | BC (-1e9 on invalid slots) | logits and Q masked before softmax |
| soft-target cross-entropy | BC | `L = -mean_b sum_k t_bk log p_bk`; grad logits `(p - t)/B` (since sum t = 1) |
| Categorical log_prob / entropy | PPO | `lp = logits - lse`; `H = -sum p lp` |
| PPO clipped loss + value MSE + entropy | PPO | analytic gradients: d/dlogits of `-min(r A, clip(r) A)` = `-A r (1[unclipped]) (onehot - p)`; value: `2(v-ret)/B`; entropy: `-p (lp + H)` form |
| Adam | BC (lr 2e-3), PPO (lr 3e-4), PPO-router (2e-4) | torch semantics: `m=b1 m+(1-b1)g; v=b2 v+(1-b2)g^2; step=lr/(1-b1^t); denom=sqrt(v)/sqrt(1-b2^t)+eps; p-=step*m/denom`, betas (0.9,0.999), eps 1e-8, no weight decay, f32 state. A param whose grad is None is SKIPPED (router BC: value head gets no gradient and keeps its init) |
| cosine LR | BC | `CosineAnnealingLR(T_max=epochs, eta_min=0)` stepped once PER EPOCH after the epoch: epoch e uses `lr0*(1+cos(pi*e/T))/2` (recursive torch form is numerically equal up to rounding) |
| grad-norm clipping | PPO-router only | global L2, max 0.5 |
| init: torch `nn.Linear` | policy, router out/value | U(-1/sqrt(fan_in), 1/sqrt(fan_in)) for W and b |
| init: normal | router `emb ~ N(0,1)/sqrt(128)`, `b0 = 0` | use `Rng.normal()` (Box-Muller, 2 uniforms) |
| permutation / argsort / gather / index_select | slot-permutation augmentation (BC), symmetry perms (PPO) | per-sample random key sort; gather index maps |
| seeded shuffles (randperm) | BC epochs; PPO none (full batch) | Fisher-Yates with `Rng.below` |
| categorical sampling | PPO actor, eval sampled | inverse-CDF over softmax(logits) |
| structured binary IO | teacher data | little-endian i16/f32, 328 B records |
| f16 encode/decode, FNV-1a64, SHA-256, CRC-32 | export / eval sets | see 4.1, 2.9 |
| matrix sizes | | PPO: obs 36..144 x 512, 512 x 256, 256 x A(10..29), 256 x 1 (~154k-213k params); router: 128x64, 64x16, 64x1 (~9.4k params) |

### 3.2 Sizing numbers
* PPO shipped models (from manifests): params 153k-213k; obs 36-144 (Pauli 104); A 10-29; |sym|
  2-12; budgets 600-2700 s with 4 cores, 335-1508 PPO iterations (1.4-4.5 s each).
* PPO batch per iteration: `num_episodes=1024` episodes x steps (up to `max_depth` 128 hence up to
  ~130k samples at high difficulty). Full-batch forward+backward per epoch ~ `6 * B * (obs_nnz*512 +
  512*256 + 256*A)` flops; with B~1e5 about 0.8-1 GFLOP x B/1e3... i.e. order 0.1-0.15 TFLOP per
  epoch, 10 epochs per iteration: needs a blocked SIMD GEMM (parallelised over batch rows with
  `parallelize`) to stay within the 1.4-4.5 s the Python stack achieves (torch dense one-hot x
  weight on 10 cores). Memory: activations `B x 512` + `B x 256` f32 = 0.4 GB at B=1e5; process in
  chunks and accumulate gradients (identical result to full batch).
* Router BC: ~168k samples x 30 epochs x (batch 512) ~ 10k Adam steps, each ~1 MFLOP: seconds.
* Total native library estimate: tensors/NN/loss/Adam/init ~ 700-900 LOC; PPO collector + eval +
  curriculum + drivers ~ 600-800 LOC (plus env-extension per **[GAP]**); container/INDEX/SHA-256/
  f16 ~ 450 LOC; topology/symmetry/jobs/targets ~ 700 LOC; BC trainer + exporters + summarizer
  ~ 500 LOC; tests ~ 500 LOC. Total ~ 3.5-4.5 k LOC vs 3.2 k Python lines being replaced.

### 3.3 Not needed (confirmed absent from training paths)
conv layers (`Conv1dPolicy` not used by train.py), AlphaZero/MCTS (`AZ`, `num_mcts_searches=0`),
TensorBoard (`SummaryWriter` only if `run_path`; train.py passes None), `gymnasium` (installed by
requirements but never imported by any QUMO file), qiskit (only `eval.py` optional baseline),
GPU/CUDA (device cpu).

---------------------------------------------------------------------------------------------

## 4. Data formats (exact byte layouts)

### 4.1 Model container (shared by `qumo.rlsynth.v0` and `qumo.rlroute.v0`)

Directory `<root>/<id>/` containing `manifest.json` and `weights.bin`.

`weights.bin`: concatenation of tensors in manifest order. Before each tensor, zero-pad the buffer
to a multiple of 64 bytes (`ALIGN=64`); after the last tensor pad again to a multiple of 64.
`offset` in the manifest is the padded start. Data row-major little-endian:
* dtype `f16` (IEEE binary16) for matrices and `f32` for every tensor whose name ends in `.bias`
  (also for all tensors when `dtype="f32"` or the name is in `f32_names`). Writer asserts finite after
  f16 rounding.
* `fnv1a64` = FNV-1a 64-bit (offset 0xCBF29CE484222325, prime 0x100000001B3) over the ENTIRE file
  including pad bytes, 16 lowercase hex digits; `sha256` over the entire file, 64 hex digits.
* Example (lf_line6, 307840 B): `embeddings.weight` [36,512] f16 @0 (36864 B), `embeddings.bias`
  [512] f32 @36864, pad to 38912, `common.0.weight` [256,512] f16 @38912, `common.0.bias` @301056,
  `action.0.weight` [10,256] f16 @302080, `action.0.bias` @307200, `value.0.weight` [1,256] f16 @307264,
  `value.0.bias` [1] f32 @307776, pad to 307840.

Tensor names / shapes:
* synth `basic_mlp`: `embeddings.weight` [OBS, EMB] (TRANSPOSED vs torch's [EMB, OBS]),
  `embeddings.bias` [EMB], `common.<k>.weight` [out,in], `common.<k>.bias`, `action.<k>.*`,
  `value.<k>.*` (k consecutive from 0; see quirk in 2.1).
* route: `embeddings.weight` [128,H] (feature-major = the torch Parameter `emb` directly),
  `embeddings.bias` [H], `action.0.weight` [16,H], `action.0.bias` [16], `value.0.weight` [1,H],
  `value.0.bias` [1] (order exactly this; value head included when present).

`manifest.json` (compact JSON `separators=(",",":")`, key order = insertion order; loader is
order-agnostic; fields read by `load_model_file`: format, id, kind, num_qubits, edges, actions, obs.size/
layout/max_rotations, max_depth, symmetries.obs_perms/act_perms, weights_file, weights_bytes, fnv1a64,
tensors[{name,shape,dtype,offset}]):
* synth: format `qumo.rlsynth.v0`, id, kind (`permutation|linear_function|clifford|pauli_network`),
  num_qubits, edges [[a,b] a<b sorted], actions [[GATE,q0,q1|-1]...] (exact model action order),
  obs{encoding:"sparse_index",size,layout,max_rotations}, max_depth, arch{type:"basic_mlp",emb,common[],
  act:"relu",first_layer_transposed:true}, symmetries{obs_perms[[...]],act_perms[[...]]}, training{...},
  license "Apache-2.0", tensors, weights_file "weights.bin", weights_bytes, fnv1a64, sha256,
  test_vectors[6 x {obs,logits,value,logits_ensemble,value_ensemble}], rollouts[2 x {input,R,actions,solved}].
* route: format `qumo.rlroute.v0`, id, kind "routing", num_qubits 0, obs{size:128,layout:"route_sign_h8_k16",
  encoding:"sign(obs[d + 8k])"}, arch{type:"sign_embedding_mlp",horizon:8,num_active_swaps:16,hidden:H,
  activation:"relu",value_head:bool,first_layer_transposed:true,forward:"..."}, derived_from, producer,
  license, training{...}, tensors..., test_vectors{obs_logits[6 x {obs[128],logits[16] (float64)}],
  episodes[N x {circuit{n,ops}, target{n,edges}, layout[], seed, actions[], overhead, final[],
  fallback, failed}]}.
  `load_route_model` requires obs.size==128, horizon==8, num_active_swaps==16, 0<hidden<=4096.
* Loader integrity: QM-E617 on missing file, bad JSON, unsupported format, size/offset error,
  `weights_bytes` mismatch, fnv mismatch; QM-E620 incompatible router.

`INDEX.json` per sub-root (`models/ai_transpiler/synth/INDEX.json`, `.../route/INDEX.json`):
`json.dump(indent=1)` of a list sorted by directory name: `{id, kind, num_qubits, edges, wl_hash,
dir, sha256 (of weights.bin), bytes (weights.bin + manifest.json sizes)}` + for the route registry
an extra `"default": true|false`. `wl_hash` = Weisfeiler-Lehman hash (rule documented in
`topology.py::wl_hash` and implemented identically by `policy.mojo::wl_hash`; "" for n == 0).
`read_model_index` needs id, kind, num_qubits, edges, wl_hash, dir. `find_route_model("auto")` uses
the row with `"default": true`, else the first routing row.
NOTE `bytes` must be computed AFTER the final manifest (with test vectors) is written.

### 4.2 Router teacher data (`route_tool gen` output, `*.bin`)
No header. Concatenation of fixed 328-byte little-endian records in job order (each job
contributes up to `ROUTE_TOOL_CAP` (env, default and maximum 256; Makefile uses 96) records):
```
offset  size  field
0       256   int16 obs[128]      obs[d + 8*k]: layer d in 0..7, active-swap slot k in 0..15
                                    (value = distance change of swap k on layer d; clamped to int16)
256     2     int16 A             number of valid active swaps (1..16)
258     2     int16 act           BEHAVIOUR action slot (teacher argmin with prob beta else model/heuristic)
260     2     int16 job           job index (clamped to int16 range)
262     2     int16 tb            teacher argmin slot
264     64    float32 Q[16]       teacher cost per slot (CX units +0.01*SABRE score +1.5 if it undoes the previous swap);
                                    1e9 for invalid slots (>= A)
```
`numpy` dtype: `[("obs","<i2",(128,)),("A","<i2"),("act","<i2"),("job","<i2"),("tb","<i2"),("Q","<f4",(16,))]`
(itemsize asserted 328). The Mojo writer is `put_i16` (saturating) / `put_f32` (bitcast). Trainer
uses `sign(obs)` as input, `Q` clipped to 1e6, filters rows whose max-min Q over valid slots <= 1e-6.

### 4.3 Router training run dir (`train_route.py --out DIR`)
* `weights.npz` (numpy `savez`, a ZIP of `.npy` v1.0 files): `emb` f32 [128,H], `b0` f32 [H],
  `w1` f32 [16,H], `b1` f32 [16], `wv` f32 [1,H], `bv` f32 [1]. (`export_npz` reads exactly these keys.)
* `train_log.json` (indent=1): {samples, val, hidden, epochs_run, tau, lr, seed, data[basenames],
  wall_s, log[{epoch,loss,val_agree,val_regret,t}]}; the exporter pops `log` and embeds the rest in
  `manifest.training`. Optional `training_meta.json` merged into `training`.
* Makefile layout under `build/ai_train/route/`: `parity/`, `jobs0.json`, `jobs1.json`, `d0.bin`, `d1.bin`,
  `run0/`, `stage/` (a model root: `stage/round0/` + `stage/INDEX.json`), `run1/`.
* Native replacement: skip the npz; write a model-directory-like run (container with all-f32 tensors,
  `qumo.rlroute.v0`) or two raw f32 files + JSON meta. If the format is the container with `dtype="f32"`,
  `--init` works with the existing loader (note: the Python `--init` path read a shipped f16 model; a
  native `--init` should accept both).

### 4.4 Router jobs / eval / episode files (JSON)
* `jobs.json`: `{"targets":[{"name","n","edges":[[a,b]...],"family"}], "circuits":[{"name","n","ops":
  [[kind,a,b]...],"family"} | {"name","qasm":path,"n"}], "jobs":[[circuit_idx,target_idx,seed(,layout[])]...]}`.
  op kinds: `cx cz swap u2 (rzz 0.3) u1 (h) measure barrier` (b = -1 for 1q ops).
* `route_tool eval` JSONL, one object per job: `{"circuit","target","n","seed","sabre_o2":[cx,depth,ms],
  "sabre_o3":[...],"policy":[cx,depth,ms,won,rejected]|null,"portfolio":[...],"heuristic":[...],"scored":[...]}`;
  `ROUTE_EVAL_CONFIGS` env selects configs. `summarize_eval.py` computes geometric means of
  `(v+1)/(base+1)` for cost/depth and `v_ms/base_ms` for time.
* `route_tool episode` output: `[{"actions":[edge idx...],"overhead":int,"final":[l2p...],"fallback":int}]`.

### 4.5 Synth run dir (`train.py --out RUN`)
`config.json` (indent=2, see 2.5), `model.safetensors` (best), `model_last.safetensors`, `log.jsonl`
(one JSON per iteration: {it,t,step_s,difficulty,succ{ppo_deterministic,ppo_10},rew{...},loss{value,policy,
entropy,total}}), `summary.json` (indent=1: kind, topology, num_qubits, iterations, wall_s, budget_s, seed,
cores, emb, common (string), best [difficulty,success,reward], final_difficulty, params, init,
start_difficulty), then `eval_blocks.json`, `eval_final.json`. `.gitignore` already excludes
`training/ai_transpiler/**/runs/`, `*.safetensors`, `*.pt`, `*.ckpt`, `logs/`, `.venv/`.

### 4.6 safetensors (only if backward-compatible reading of old runs is wanted)
`u64 LE N` | `N bytes UTF-8 JSON {"name":{"dtype":"F32","shape":[..],"data_offsets":[b,e]},...,
"__metadata__":{..}}` (padded with spaces to 8-byte alignment) | raw little-endian tensor data;
offsets relative to the start of the data region. Keys per 2.5. Recommendation: do not keep
safetensors; use the container with `dtype="f32"` for checkpoints (`model` best / `model_last`).

### 4.7 Fixtures / tests
`tests/fixtures/ai/policy/{synth/<id>/,synth/INDEX.json,bad_checksum/,bad_format/}`, root
`FX="tests/fixtures/ai/policy"`. `tests/test_policy.mojo` also reads `models/ai_transpiler/synth` and
requires worst-parity < ~1e-5 and exact greedy rollouts; `tests/test_ai_route.mojo::test_committed_model`
requires `obs_logits` error < 1e-6 and bit-exact episodes (actions, overhead, final layout) for the
COMMITTED router. These committed artifacts are Python-derived oracles; keep them untouched.

---------------------------------------------------------------------------------------------

## 5. Algorithm specs the Mojo port must reproduce

### 5.1 Router BC (train_route.py)
1. Load records (4.2); drop rows with `spread = nanmax(Q valid) - nanmin(Q valid) <= 1e-6`.
2. Shuffle with `default_rng(seed).permutation`; `nval = max(1000, N//20)`; first nval -> val, rest -> train.
3. Net: `emb ~ N(0,1) * (1/sqrt(128))` shape [128,H]; `b0 = 0`; `out = Linear(H,16)` (torch init);
   `value = Linear(H,1)` (torch init, never trained in BC; stays at init and is exported as is).
   Forward: `h = relu(b0 + s @ emb)`, `logits = out(h)`; `s = sign(obs)` in {-1,0,1}.
4. Epoch loop (epochs 30, batch 512, last partial batch kept, lr 2e-3, tau 0.5): per epoch fresh
   `randperm(N)`; per batch: random slot permutation (new slot j takes old slot perm[j];
   `perm = argsort(rand(B,16))`; obs viewed as [B,16,8] and gathered on dim 1; Q and valid gathered
   likewise); `Qm = Q masked 1e9`; `target = softmax(-(Qm - min_k Qm)/tau)`; `logits masked -1e9`;
   `loss = -mean_b sum_k target * log_softmax(logits)`; Adam step; running `tot += loss*len(batch)`;
   after the epoch `sched.step()`; validation: argmax of masked logits, `agree = mean(Q[pick] <= qmin+1e-6)`,
   `regret = mean(Q[pick]-qmin)` (no augmentation), log `{epoch,loss=tot/N,val_agree,val_regret,t}`;
   stops early when elapsed > `--budget_s` (1800 default; unrelated to AI_ROUTE_BUDGET_S).
5. Save emb,b0,w1,b1 and the (untrained) value head.
Hand gradient: `g_logits[b,k] = (softmax(logits)[b,k] - target[b,k]) / B` on valid slots (masked slots
carry `target = 0`, their softmax underflows to 0 with -1e9); `dW1 = g^T h`, `db1 = sum g`,
`dh = g W1`, `dh *= (h_pre > 0)`, `db0 = sum dh`, `demb[i,:] += sign_i * dh` (scatter-add).

### 5.2 Router teacher / DAgger (already in route_tool.mojo; keep)
Per active swap k: 2 seeded rollouts (heuristic or model) until G=16 more 2q gates execute, Q =
overhead delta (CX units) (+ `1000 + front distance` if the rollout fails to reach the goal), min over
the 2; `+0.01 * heuristic_score` tie-break; `+1.5` for undoing `last_e`; behaviour action = teacher argmin
with prob beta else the model's argmax (DAgger beta 0.3), common random numbers across k. Odd job seeds
start from a reverse-pass placement. Seeds: `Rng(seed*1000003 + job_seed)`, `mixr = Rng(seed*31 + job)`.
Parallel over jobs with `parallelize` (batches of 160 jobs).

### 5.3 Synthesis PPO
See 2.5. Native pieces: (a) env wrapper over `PolicyEnv` with reset-by-difficulty and reward/mask/final
**[GAP]**; (b) actor: sparse forward with a uniformly random symmetry per step (records perm), sampling
from softmax(logits) (twisterl samples from logits without masks - confirm); (c) GAE exactly as above;
(d) 10 full-batch Adam epochs with the loss of 2.5; (e) eval pass with two configs; (f) curriculum
+ best checkpoint + budget loop + continuation. Threading: episodes are independent -> `parallelize`;
determinism by giving each episode `Rng.split(episode_index)`.

### 5.4 Export (synth)
Edges, actions, obs layout strings, symmetries (via `symmetry` port), manifest assembly (2.6), test
vectors / rollouts from the stored weights with the Mojo `MLPPolicy` and `PolicyEnv.greedy_trace`, INDEX
rebuild. Pauli self-check against qiskit-gym's dense permutation (`gym_dense_perm`) is replaced by the
golden-manifest test (pauli_line4: obs_perms/act_perms reproduced exactly).

---------------------------------------------------------------------------------------------

## 6. Proposed Mojo module layout and ordering

New package `qumo/rl/` (library; training only, not imported by the runtime compiler modules):
`tensor.mojo` (aligned f32 buffers, GEMM, SIMD helpers), `nn.mojo` (EmbeddingBag, Linear, MLP with
caches + backward), `loss.mojo` (log_softmax, soft-CE, PPO terms), `optim.mojo` (Adam f32, cosine,
grad clip), `init.mojo` (torch-style init on `Rng`), `container.mojo` (+ `sha256.mojo`, f16 encode, FNV
reuse from policy.mojo, manifest/INDEX writers), `topology.mojo`, `symmetry.mojo`, `jobs.mojo`
(targets + circuit families + `build_jobs`), `ppo.mojo` + `synth_env.mojo`, `teacher_data.mojo` (record IO).
Tools under `training/ai_transpiler/`: `route/route_tool.mojo` (+ `gen-jobs`, `eval-sets`, `summarize`),
`route/train_route.mojo`, `route/export_route.mojo`, `synth/train_synth.mojo`, `synth/export_synth.mojo`,
`synth/eval_synth.mojo`, `common/models_verify.mojo`, `common/make_fixtures.mojo`, a single
`train_all.mojo` orchestrator (replaces the missing `train_all.sh`).
Order of work (each step independently testable): (1) sha256 + f16 encode + container writer +
models_verify (golden: re-write shipped models from their decoded tensors -> byte-identical weights.bin,
identical fnv/sha, identical INDEX rows); (2) topology + symmetry + targets (golden: 9 manifests, 25+10
topologies vs bench/neutral/targets.py edge lists); (3) tensor/nn/loss/optim with gradient checks vs
finite differences and (optional, one-off, scratch venv) vs torch on random tensors; (4) router BC
(golden: train on a small d0.bin, val agreement, parity with committed regime; export -> loads in
`tests/test_ai_route.mojo`); (5) jobs generator + `train-ai-route` without Python; (6) PPO + synth env
(needs the **[GAP]** facts); (7) orchestrator, Makefile, docs, delete Python + requirements.

Decisions the lead needs to take:
* D1 keep `train_ppo.py` (router PPO)? Recommended: port last or drop, README says it degraded the model.
* D2 Qiskit O3 baseline inside `eval.py`: drop (recommended) or keep as optional external step.
* D3 Python MT / PCG64 stream compatibility: not required (see section 0).
* D4 checkpoints: container-f32 instead of safetensors (recommended).
* D5 `bench/neutral/targets.py`: keep (benchmark input) and golden-test the Mojo copy against it.

---------------------------------------------------------------------------------------------

## 7. Defects, quirks and gaps found in the existing training tree

7.1 `training/ai_transpiler/synth/train_all.sh` does not exist (not in git, not on disk), yet
`make train-ai-synth` requires it (otherwise prints "not found" and exits 1). The campaign actually
run (docs ledger: 9 shipped models + dropped ones, budgets 10-70 min, 2 lanes) is documented only in
`docs/AI_TRANSPILER.md` lines ~188-221. Port task: define the model campaign as data (model id,
kind, topology spec, budget, continuation) in the new orchestrator. Shipped specs (from manifests):
perm_grid3x3 `grid:3x3` 600 s; perm_grid3x4 `grid:3x4` 900 s; perm_king3x4 `king:3x4` 1200 s;
perm_hhring10s2 `heavyhex_frag:ring10s2` 2700 s; lf_grid2x3 `grid:2x3` 2700 s; lf_ring6 `ring:6` 2400 s;
lf_line6 `line:6` 2700 s; lf_hhH7 `heavyhex_frag:H7` 2700 s; pauli_line4 `line:4` 2100 s continued from
2 earlier stages (total 6300 s). All with 4 cores, seed 1, emb 512, common [256], episodes 1024,
diff-max 256. Dropped (docs): clifford_* (5 models, 4q), lf_line7, pauli_complete5, pauli_line6, perm_line8
(topology specs of these are not in the repo; ledger gives sizes only).
7.2 `AI_ROUTE_BUDGET_S` unused; `train_route.py --budget_s 1800` default silently truncates epochs.
7.3 `requirements.txt` omits rustworkx (needed by `bench/neutral/targets.py` -> `circuits.py`).
7.4 `export_route.export_router`: `update_index` rebuilds INDEX.json from manifests without the
`default` key, then sets `default` only for the model being exported when `--default`; with
`default=False` the previous default is lost (`bool(e.get("default"))` is always False on fresh
entries). The native INDEX writer must preserve/take an explicit default id.
7.5 `basic_mlp_tensors` layer-index quirk (2.1).
7.6 `train.py` `make_env(..., max_rotations=5)` argument unused; Pauli rotations come from the env
default (5), also hard-coded fallback `envcfg.get("max_rotations", 5)` in the exporter.
7.7 `eval.py` hard-codes `.venv/bin/mojo`; `Makefile` `MOJO` default is the same.
7.8 `docs/AI_TRANSPILER.md` still says "twisterl `BasicPolicy`" for the model format - the format stays
(the runtime re-implements that forward) but wording on training must change.
7.9 Python `route_env.py` defines the golden episodes that the committed router manifest carries; after
retiring it, new exports embed Mojo-generated episodes (self-consistency only). Keep the committed
model in tests so the independent Python-derived oracle survives.
7.10 `parity.py` is run inside `make train-ai-route` as a gate; after the port the gate becomes
`tests/test_ai_route.mojo` (+ new tests for the trainer/export).

---------------------------------------------------------------------------------------------

## 8. Acceptance tests to add (golden)

1. `weights.bin` writer: re-encode each shipped model from its decoded tensors -> identical bytes,
   `fnv1a64`, `sha256` and INDEX rows (all 10 models).
2. SHA-256 known-answer tests (FIPS vectors) + `models verify` on `models/ai_transpiler` prints 0 problems.
3. f16 encode exhaustive round trip vs `f16_to_f32`; overflow raises.
4. `symmetry` port reproduces `symmetries` of all 9 shipped synth manifests exactly (order included).
5. Topology / target generators reproduce edge lists of `bench/neutral/targets.py` for all 25 training +
   10 eval topologies (generate once with scratch python, commit as a fixture JSON).
6. Gradient checks (finite differences) for EmbeddingBag/Linear/ReLU/soft-CE/PPO loss; Adam against a
   hand-computed reference (and against torch once, offline, scratch venv only).
7. Router BC: train on a synthetic teacher file with a planted optimum, reach agreement > 0.9;
   export loads with `load_route_model`; `tests/test_ai_route.mojo` passes with the new model.
8. Synth PPO smoke: `perm:line:3` (or `line:4`) reaches difficulty > 3 within seconds; exported model
   loads, test vectors replay, greedy rollout solves.
9. `make train-ai` end-to-end with no `python`/`uv`/`torch` on PATH (CI check: `env -i` PATH with only mojo).
