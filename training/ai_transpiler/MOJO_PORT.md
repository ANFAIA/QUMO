# Native Mojo training stack for the AI transpiler: design and implementation plan

Status: plan of record, revision r2, 2026-11-07. Branch `mojo-rl-training`. Owner of this file: the lead;
component agents read it, they do not edit it (a signature change is requested from the lead, who edits section 5
and bumps the revision line at the bottom). Revision r2 applies the review of r1; each objection and its
disposition are listed in section 20.

Request that triggered this work (user, verbatim): "convierte todas las dependencias de RL para los AI transpiler
passes to Mojo (desde Python o Rust)."

Goal: `make train-ai` (and `make models-verify`) run with **no Python, no Rust, no torch, no numpy, no
subprocess to Python** anywhere in the training path. The runtime inference stack (`qumo/compiler/policy.mojo`,
`policy_synth.mojo`, `ai_route.mojo`) is already native and stays untouched except for the few additive edits
listed in section 12.

Contents

1. Scope, decisions and what is reused
2. Architecture and module layout
3. Determinism and threading model
4. Design of each library module (what it does, exact math)
5. Normative signatures (the contract between components)
6. Environments (qiskit-gym port)
7. PPO trainer (twisterl port)
8. Router training (behaviour cloning, DAgger, optional PPO)
9. Export, container, INDEX, models-verify
10. Drivers, CLI, Makefile, campaign
11. Feature-by-feature mapping (every Python / Rust / third-party feature to Mojo)
12. Edits to existing files, legacy Python removal, docs, NOTICE
13. Test plan (golden data, fixtures, tolerances)
14. Acceptance criteria
15. Performance targets
16. Implementation plan: components, disjoint file ownership, dependency order
17. Risks
18. Open questions and corrections to the research reports
19. Appendix: constants, probes, golden-to-fixture map
20. Review log (r1 to r2)

Inputs this plan is built on (data, copied into the repo by component C0 under `training/ai_transpiler/spec/`
because the originals live in a scratch directory): `envs.md` (qiskit-gym 0.4.1 environments), `ppo.md` (twisterl
0.5.1 BasicPolicy + PPO), `torch.md` (torch / numpy / safetensors operations), `repo.md` (porting map of the
17 Python files and the Makefile). Section references like "envs.md 9.7" point there. When this plan and a spec
disagree, section 18 lists the known corrections; otherwise the spec wins and the golden data is the arbiter.
The scripts that produced the goldens (`envs_ref.py`, `ref_ppo.py`, `ref_np.py`) are not committed (D11, they are
line-level transcriptions); the golden data arbitrates, and for a case it does not cover the lead (who keeps the
scripts in scratch) answers and, if needed, has C0 add a fixture (16, rule 6).

---------------------------------------------------------------------------------------------------------

## 1. Scope, decisions and what is reused

### 1.1 What must stop depending on Python / Rust

| dependency today | used for | replaced by |
|---|---|---|
| qiskit-gym 0.4.1 (Rust env + Python wrappers) | the 4 synthesis environments, gateset construction, scramble generator, reward, symmetries (`twists`) | `qumo/rl/gateset.mojo`, `qumo/rl/synth_env.mojo`, `qumo/rl/symmetry.mojo` |
| twisterl 0.5.1 (Rust collector, evaluate; Python PPO, BasicPolicy) | rollouts, GAE, evals, PPO update, policy network | `qumo/rl/ppo.mojo`, `qumo/rl/nn.mojo`, `qumo/rl/loss.mojo` |
| torch 2.14.1 | init, forward/backward, Adam, cosine LR, randperm, softmax/CE, grad clipping | `qumo/rl/{tensor,nn,loss,optim,rlrng}.mojo` (hand-derived backward, no autograd) |
| numpy 2.5.4 | f16 conversion, structured binary IO, `.npz`, RNG, test vectors | `qumo/rl/f16.mojo`, `teacher_data.mojo`, Mojo `Rng`; `.npz` is dropped |
| safetensors 0.8.0 | PPO checkpoints and warm start | f32 model-container checkpoints (`checkpoint.mojo`); optional read-only safetensors importer |
| qiskit 2.5.2 | only the optional Qiskit-O3 cost baseline in `synth/eval.py` | dropped from the training path (stays in `bench/neutral`) |
| gymnasium 1.4.0 | installed by requirements, never imported by QUMO | nothing |
| rustworkx (undeclared) | heavy-hex edges in `bench/neutral/targets.py` imported by `route/circuits.py` | existing `Target.heavy_hex_edges(d)` |
| hashlib / zlib / random / json / os / subprocess (stdlib) | SHA-256, CRC-32, MT19937 job generator, JSON, filesystem, calling `mojo` | `digest.mojo`, `Rng`, `qumo.ai.json`, `std.os`, direct function calls |
| inline `python3 -c` in `make models-verify` | sha256 check of all models | `training/ai_transpiler/common/models_verify.mojo` |

### 1.2 Decisions (D1-D5 of repo.md resolved, plus new ones)

* **D1 router PPO (`train_ppo.py`)**: ported, as the lowest-priority component (C12), outside `make train-ai`.
  The user asked for all RL dependencies; the README says it degraded the model, so it is available as
  `make train-ai-route-ppo` and never part of the default recipe.
* **D2 Qiskit O3 baseline in `eval.py`**: dropped from the training path. The "qiskit_o3_unpaired" key disappears
  from newly exported manifests (the loader never reads it). The Qiskit comparison lives in `bench/neutral`
  (a benchmark, not training) and is untouched.
* **D3 Python RNG compatibility**: not required (MT19937, PCG64, torch Philox are not reproduced). All randomness
  is the seeded `qumo.sim.rng.Rng` (xoshiro256**). The old pipeline was not reproducible anyway.
* **D4 checkpoints**: f32 *model-container directories* (loadable by `load_model_file`) replace `.safetensors`.
  A read-only safetensors importer is provided only so an old Python run can still warm-start a Mojo run
  (component C11a, optional, low priority).
* **D5 `bench/neutral/targets.py`**: stays (benchmark input). The Mojo topology generators are golden-tested against
  edge lists produced once from it (fixture `tests/fixtures/rl/topologies.json`).
* **D6 numerical fidelity**: default is *twisterl-faithful* (including the "masked old logits, unmasked new
  logits" ratio and "save post-update weights as best"). `PPOConfig` carries two switches, both defaulting to the
  faithful behaviour: `mask_new_logits: Bool = False`, `best_checkpoint: String = "post_update"` (alternative
  `"evaluated"`). Fixing either is a later, measured decision, not part of this port.
* **D7 time budget**: exactly one of `--iters N` (exact, reproducible) and `--budget-s S` (checked between
  iterations) must be given; both or neither is QM-E640. A budgeted run is a deterministic prefix of the `--iters`
  run with the same seed; only the stopping point depends on the clock.
* **D8 scope exclusions**: AlphaZero / MCTS (`num_mcts_searches`), `Conv1dPolicy`, TensorBoard, GPU, multi-agent.
  QUMO sets `num_mcts_searches = 0` everywhere. The architecture leaves a hook (`Trainer` takes an `Algorithm`
  mode enum with only `PPO`).
* **D9 generalisation kept small**: `BasicNet` supports any list of common layers and optional hidden layers in the
  action / value heads (twisterl allows them), but the layer-numbering quirk is fixed: tensors are named with
  *consecutive* indices (`common.0`, `common.1`, ...), as `MLPPolicy._seq` requires (repo.md 2.1, defect 7.5).
* **D10 campaign as data**: the missing `train_all.sh` is replaced by `training/ai_transpiler/synth/campaign.json`
  plus a Mojo orchestrator (section 10.3).
* **D11 clean-room**: implementers work from `training/ai_transpiler/spec/*.md` and the golden fixtures. They do
  not read, copy or transliterate qiskit-gym, twisterl or torch source. Section 12.5 has the NOTICE text.
* **D12 hardware neutrality**: all kernels are portable SIMD Mojo (no Metal, Accelerate, CUDA, AMX, NEON
  intrinsics). An optional `-D QUMO_RL_BLAS` build may bind the standard CBLAS `sgemm` symbol; it is off by
  default, documented as non-deterministic across thread counts, and not used by any acceptance test.

### 1.3 Reused as-is from the existing tree (do not re-port)

| need | existing symbol |
|---|---|
| seeded RNG, `split`, `normal`, `below` | `qumo/sim/rng.mojo::Rng` |
| model loader, f16 decode, FNV-1a64, WL hash, INDEX reader | `qumo/compiler/policy.mojo::{load_model_file, ModelFile, f16_to_f32, fnv1a64_bytes, hex16, wl_hash, read_model_index, ModelIndexEntry, MODEL_FORMAT_*, model_kind_code, gate_code}` |
| packed-state environment dynamics (all four kinds), `inverse`, `solved`, `observe`, `apply`, Pauli DAG cleaning | `qumo/compiler/policy_synth.mojo::PolicyEnv` (wrapped, not copied, by `TrainEnv`) |
| inference forward, ensemble, greedy / beam / sampled search, rollouts | `MLPPolicy`, `greedy_trace`, `search_best` |
| routing environment (full episode), problem builder, model, verification | `qumo/compiler/ai_route.mojo::{Episode, RProblem, RouteModel, load_route_model, run_batch}` |
| teacher / DAgger data generation, eval, episode traces | `training/ai_transpiler/route/route_tool.mojo` (`gen`, `eval`, `episode`; helpers move to `qumo/rl/jobs.mojo`) |
| block-level evaluation vs exact / greedy synthesizers | `training/ai_transpiler/synth/eval_blocks.mojo` (body moves to `qumo/rl/eval_blocks.mojo`) |
| heavy-hex edges, BFS distances, `Target.from_edges` | `qumo/compiler/target.mojo::Target` |
| JSON parse / compact writer | `qumo/ai/json.mojo::{parse_json, JsonValue, JsonWriter, pretty_json}` |
| cosine schedule (Float64) | `qumo/algorithms/qml_train.mojo::cosine_lr` (same closed form `lr0 (1+cos(pi e/T))/2`) |
| `parallelize`, `run_chunks` | `max.algorithm`, `qumo/sim/statevector.mojo::run_chunks` |

The runtime modules are never imported *by* `qumo/rl/` in the other direction: `qumo/compiler/*` must not import
`qumo/rl/*` (training code stays out of the compiler binary).

### 1.4 New things that have no Mojo equivalent today

SHA-256 (verified absent from the Mojo 1.2 stdlib: `std.hashlib` does not exist and `std.hash` has no `sha256`),
CRC-32, f32 to f16 *encode*, aligned/padded container writer and INDEX writer, f32 GEMM family with backward,
Adam / grad clipping / init, topology spec parser, graph automorphisms, the qiskit-gym training environment
wrapper (reset, reward, metrics, random inversion, Pauli generator and hidden symmetry), the PPO collector /
evaluator / trainer, the router BC trainer, the circuit-family job generator, the neutral topology generators.

Probe results obtained for this plan (scratch Mojo 1.2.0 program, run with the repo's `.venv/bin/mojo`; numpy
used only as the comparison oracle): `Float16(Float32(x))` is round-to-nearest-even and matches numpy bit for
bit on 65519.99 (31743), 65520 (inf), 2^-25 (0, tie to even), 2^-24 (1), 0.1 (11878) and the 1+2^-11 tie (15360).
So the encoder is `bitcast[DType.uint16](Float16(x))` plus a finiteness check; the exhaustive golden test in
section 13 still gates it.

---------------------------------------------------------------------------------------------------------

## 2. Architecture and module layout

```
qumo/rl/                       (library, training only; Apache-2.0; tests are tests/test_rl_*.mojo)
  __init__.mojo                package docstring                                  [C0]
  testutil.mojo                fixture loader, assert_close helpers               [C0]
  digest.mojo                  sha256, crc32                                      [C1]
  f16.mojo                     f32 -> f16 encode, overflow check, bulk helpers    [C1]
  rlrng.mojo                   seed derivation, sampling, shuffles, init draws    [C4]
  tensor.mojo                  Mat, f32 GEMM (nt / nn / tn), vector kernels       [C4]
  optim.mojo                   Adam, grad-norm clip, cosine LR                    [C4]
  nn.mojo                      ParamStore, GradBufs, BasicNet (+ caches, fwd/bwd) [C5]
  loss.mojo                    log-softmax, synth PPO loss+grad, GAE (twisterl)   [C5]
  topology.mojo                spec parser, slug, connectivity                    [C3]
  symmetry.mojo                automorphisms, twists (plain + Pauli)              [C3]
  gateset.mojo                 gateset + shell ModelFile from (kind, n, edges)    [C6]
  synth_env.mojo               TrainEnv: reset / step / reward / metrics / masks  [C6]
  ppo.mojo                     collector, evaluate, ppo_update, Trainer           [C7]
  container.mojo               model-dir writer, manifest + INDEX writers, verify [C2]
  checkpoint.mojo              run-dir layout, f32 checkpoints, safetensors read  [C11a]
  export_synth.mojo            run -> qumo.rlsynth.v0 (f16, vectors, rollouts)    [C11a]
  eval_blocks.mojo             block-level eval library (moved from the tool)     [C11b]
  synth_train.mojo             time/iter-bounded driver loop, log, summary        [C11c]
  teacher_data.mojo            328-byte teacher records IO                        [C8]
  router_net.mojo              RouterNet (sign-embedding MLP) fwd/bwd             [C8]
  router_loss.mojo             BC soft-CE, route-PPO loss, GAE (float64)          [C8]
  bc.mojo                      behaviour-cloning trainer                          [C8]
  targets.mojo                 25 + 10 neutral topologies (ordered edge lists)    [C9]
  jobs.mojo                    circuit families, build_jobs, eval sets, JSON glue [C9]
  export_route.mojo            run -> qumo.rlroute.v0, INDEX default handling     [C10]
  route_ppo.mojo               optional router PPO                                [C12]

training/ai_transpiler/        (drivers: thin mains; each builds to build/<name>)
  synth/train_synth.mojo       [C11c]    synth/export_synth.mojo  [C11a]   synth/eval_synth.mojo [C11b]
  synth/eval_blocks.mojo       thin main over qumo/rl/eval_blocks.mojo  [C11b]
  synth/train_all.mojo + campaign.json                                    [C11c]
  route/route_tool.mojo        + gen-jobs, eval-sets, summarize           [C9]
  route/train_route.mojo       [C8]     route/export_route.mojo [C10]    route/train_route_ppo.mojo [C12]
  common/models_verify.mojo    [C2]     common/make_fixtures.mojo [C11a]
  spec/*.md                    [C0]     README.md, MOJO_PORT.md (this file) [lead / C13]
tests/fixtures/rl/**           golden data in small form                  [C0]
tests/test_rl_*.mojo           one per library module, plus fixtures, smoke, e2e [owner as listed in 13.2; fixtures = C0, smoke = C11c, e2e = C13]
```

Dependency graph of the library (arrows = imports; nothing imports upwards):

```
rlrng, tensor ---> optim
tensor, rlrng ---> nn ---> loss
topology ---> symmetry ---> synth_env <--- gateset <--- (policy_synth.PolicyEnv, policy.ModelFile)
nn, loss, synth_env, optim, rlrng ---> ppo ---> synth_train
digest, f16 ---> container ---> {checkpoint, export_synth, export_route, models_verify}
checkpoint, export_synth, eval_blocks ---> synth_train, train_all
tensor, optim, rlrng ---> router_net ---> router_loss ---> bc ---> train_route ; route_ppo
targets, jobs ---> route_tool, export_route
```

`qumo/rl` files stay below ~1200 lines each (compile time and agent context); a module that grows past that is
split by its owner without changing the public names.

---------------------------------------------------------------------------------------------------------

## 3. Determinism and threading model

Goal: for a given `(seed, config)` the result is **independent of `--cores`** and bit-identical across reruns on
the same machine and binary. This improves on twisterl, whose Rust collector used `rand::thread_rng()` and a
fresh rayon pool per call (not reproducible).

Rules every component follows:

1. **Seed tree.** One `run_seed: UInt64`. Every random consumer derives its own `Rng` with
   `seed_for(run_seed, TAG, a, b, c)` (SplitMix64 mixing of the tuple; `rlrng.mojo`). Tags are an enum in
   `rlrng.mojo`: `TAG_INIT`, `TAG_COLLECT`, `TAG_EVAL_DET`, `TAG_EVAL_SAMPLED`, `TAG_BC_SPLIT`, `TAG_BC_EPOCH`,
   `TAG_BC_SLOT`, `TAG_JOBS`, `TAG_EXPORT_VEC`, `TAG_ROUTE_PPO`. Examples: episode `e` of iteration `git` in the
   collector uses two generators, `seed_for(s, TAG_COLLECT, git, e, 0)` for the environment and
   `seed_for(s, TAG_COLLECT, git, e, 1)` for the actor (action sampling and symmetry draws). Evaluation episode `e`
   of iteration `git` uses `TAG_EVAL_DET` (env stream shared with the sampled eval of the same episode) and search
   `k` of the sampled eval uses `c = 2 + k` for the actor stream.
   **The iteration key is global**: `git = start_iteration + it`, where `it` counts the iterations of this process
   and `start_iteration` is 0 for a fresh run and, for a warm start, the cumulative iteration count of the init run
   (its `start_iteration + iterations`, read from `run.json` / `summary.json`; 0 when the init is a bare model
   container, or the explicit `--start-iteration N`); `run.json` records it. Without the offset every stage of a
   `--init auto` chain (the three `pauli_line4` stages all use `--seed 1`) would replay the scramble prefixes and the
   Gumbel noise of the previous stage at the same local iteration, silently correlating warm-started stages, whereas
   the old unseeded Rust RNG gave independent rollouts per stage. A rerun of one stage with a different budget still
   replays the same prefix, which is the D7 property. T-DET (13.2) asserts that stage 2 iteration 0 uses different
   stream seeds, and collects different start states, than stage 1 iteration 0.
2. **No shared mutable RNG across tasks.** `parallelize` tasks are *items* (episode, virtual worker, job); each owns
   its generators. Results are written to preallocated per-item slots and merged in item-index order.
3. **Fixed reduction tree.** Gradient accumulation does not depend on the thread count: rows are cut into fixed
   chunks of `CHUNK_ROWS = 512`; chunk `i` belongs to *virtual worker* `i % VW` with the constant `VW = 16`; each
   virtual worker owns a private gradient buffer and processes its chunks in increasing order; the 16 buffers are
   then summed in index order. `parallelize(body, VW, cores)` maps virtual workers to however many threads exist.
   Loss partial sums are per chunk (Float64), summed in chunk order. The same `VW` rule is used by the router BC
   (with `VW = 4`, `CHUNK_ROWS = 128`, part of the config hash).
4. **Constants that fix results are part of the run config** and are written to `run.json` / the manifest
   (`chunk_rows`, `vw`), so a run that changes them is visibly different.
5. **Where determinism cannot be promised** (documented in the README): (a) a wall-clock budget decides how many
   iterations run (use `--iters`); (b) a different CPU architecture or a different Mojo version may change
   floating-point contraction (FMA) and `exp`/`log` rounding, after which a run diverges chaotically (the expressions
   that must stay bit-exact against the Rust env are protected by rule 9); (c) the optional BLAS build; (d) hardware
   faults. Same binary + same machine + any `--cores` must give identical bytes; this is a test (section 13, T-DET).
6. Wall-clock is read only by the budget check and by logging fields (`t`, `step_s`); it never feeds the
   numerics.
7. **Tasks never raise.** `parallelize` bodies cannot raise in Mojo 1.2 (probe: a `raises` task passed to `parallelize`
   is rejected, "cannot be converted from `def task(i: Int) raises` to `def(Int) -> None`"). Every parallel region
   therefore preallocates one status slot per item (`List[Int32]`, 0 = ok, otherwise the QM-E64x number). After
   `parallelize` returns, the caller scans the slots in item-index order and raises the first non-zero one, so the
   reported error is deterministic. Conditions the Rust code treats as panics (`TrainEnv.step` with `a >= A` under
   `add_perms`, a non-finite logit in a task) are clamped (action 0) and flagged in the slot, never aborted inside a
   task. The NaN guard (QM-E643) runs in the driver on the reduced loss.
8. **Cores.** `effective_cores = min(requested, parallelism_level())`, printed in the first log line and stored in
   `run.json`. Measured on the reference machine (10-core M5 with 4 performance and 6 efficiency cores):
   `parallelism_level()` is 4, and asking `parallelize` for more workers adds no threads (16 equal items took 160 ms
   with 4, 8 and 16 workers; `MAX_NUM_THREADS`, `MODULAR_NUM_THREADS` and `KGEN_NUM_THREADS` did not change it). The
   6 E-cores are therefore unused (recorded in section 15). Because of the virtual workers of rule 3 the clamp changes
   speed only, never results. Two processes that each ask for 4 workers share the same 4 pool threads (measured: about
   0.69x GEMM throughput per process), which is why `AI_LANES` defaults to 1 (10.2).
9. **No FMA contraction where exactness is claimed.** Mojo 1.2 contracts `(a - b) + (w * c)` into a fused
   multiply-add by default; the Rust env does not. Probe: `achieved = 0, penalty = 1e-4f, w = 0.01f, count = 5` gives
   0.04989999905 fused and 0.04989999533 unfused; numpy f32 and Rust give the unfused value, and the same expression
   with the product behind a `@no_inline` function returns it too. Every f32 expression for which this plan claims
   bit-exact parity with the Rust env or twisterl (the metrics penalty chain, the reward
   `(achieved - penalty) + (pauli_layer_reward * f32(removed))`, the GAE recursion) therefore computes each product
   through `mulf(a, b)` (`@no_inline`, in `rlrng.mojo`) and keeps the sum or difference in a separate statement.
   Kernels (GEMM, Adam, loss) may fuse; their goldens are tolerance-based. The shipped step goldens cannot detect a
   missing barrier (every golden Pauli reward lies in both the fused and the unfused value sets), so
   `envs/reward_fma_cases.json` (the example above and the 32 (achieved, penalty, count) combinations found by brute
   force to differ) is tested bit-exactly in `test_rl_env`. Bit-exact reward parity holds only with this barrier.

The existing teacher generator (`route_tool gen`) is already seed-deterministic per job (`Rng(seed*1000003 +
job_seed)`), batched in groups of 160 jobs; it is unchanged.

---------------------------------------------------------------------------------------------------------

## 4. Design of each library module

All numbers are IEEE f32 unless a quantity is marked f64. "Chunk" means a block of at most `CHUNK_ROWS` consecutive
rows of the batch.

### 4.1 `rlrng.mojo`

* `splitmix64(x)`: the exact mixing function of `Rng._splitmix` as a pure function.
* `seed_for(base, tag, a, b, c)`: `x = splitmix64(base ^ (tag * K0)); x = splitmix64(x ^ (a * K1)); ... ` (K0..K3 odd
  64-bit constants documented in the file).
* `uniform_f32(r) = Float32(r.next_u64() >> 40) * 2^-24` (24-bit, matches the `k * 2^-24` model of the Pauli
  generator in envs.md 9.7: `k` is the integer, thresholds are compared on `k`, never on the float).
* `uniform_sym_f32(r, bound) = (2 u - 1) * bound` for torch-style `U(-b, b)` init.
* `normal_f32(r)`: `Float32(r.normal())` (Box-Muller, 2 uniforms per call).
* `randperm(r, n)`: Fisher-Yates, `i = n-1 .. 1`, `j = r.below(i + 1)`.
* `gumbel_argmax(logits, n, r)`: `argmax_i (l_i - ln|ln u_i|)` computed in f32, `u_i = uniform_f32 clamped to
  [2^-24, 1 - 2^-24]`, **first maximum wins**: a strict `>` scan seeded with element 0, NaN never wins (this is
  twisterl's `argmax`; its source reads `if val > max_val` starting from `values[0]`). This is the exact twisterl
  sampling rule. With the f32 mask value `-1e10` the Gumbel term is absorbed (spacing at 1e10 is 1024), so on an
  all-masked row the result is **index 0**. This settles O-3 (section 18.2) without a probe, and the sentence in
  `ppo.md` section 6 that the action on such a row is "a random one" is wrong (C-7). Unit tests: an all-`-1e10`
  row returns index 0 for 1000 generator states; ties return the first index; a NaN entry is never chosen unless it
  is element 0.
* `sample_probs(p, n, r)` / `sample_softmax`: inverse CDF for the sampled evaluation (`u * sum(p)`, linear scan, last
  index on round-off) over a softmax computed with max-subtraction; the deterministic evaluation takes the argmax of
  the logits (first maximum). **Deliberate deviation from Rust**: Rust takes the argmax of
  `probs = exp(logit) / (sum + 1e-6)` with no max shift, and a failing `WeightedIndex` silently returns action 0; for
  logits above about 88.7 (f32 `exp` overflows) its probabilities become inf/NaN and it returns index 0 when element 0
  is NaN. The max-shifted version differs only for such a diverged policy and is numerically safe; a unit test with
  logits of 100 checks that the argmax picks the true maximum and `sample_softmax` yields a valid draw without NaN.
* `mulf(a: Float32, b: Float32) -> Float32` (`@no_inline`): the FMA barrier of section 3, rule 9.

### 4.2 `tensor.mojo`

* `comptime FPtr = MutPointer[Float32, MutAnyOrigin]` (the one pointer alias of the contract; every other module
  imports it from `tensor.mojo`), `comptime LANES = 8` (matches `policy.mojo::_dense`), `Mat` = row-major
  `List[Float32]` + `rows`, `cols`; buffers are used with unaligned vector loads (no alignment requirement).
* Three GEMM variants, all `C (+)= ...`, f32 accumulate; the scalar tail loops are exact. **Simple kernel first**
  (measured for review r2 on the reference machine): a plain 4-row x 16-column register tile (two SIMD8 vectors,
  fused multiply-add, no packing, no K-blocking) reached 63 GFLOP/s on one core, and forward + `dW` + `dx` for
  B = 512, E = 512, C = 256 with `W^T` transposed on every call reached 117-122k row-epochs/s per core (448-470k on
  four cores), which is the best case: no ReLU, loss, embedding scatter, Adam or gradient reduction. The
  implementation therefore is:
  * `gemm_nn(m, n, k, A, lda, B, ldb, C, ldc, acc)`: `C[m,n] = A[m,k] . B[k,n]` with the 4x16 tile (backward
    `dx = dz W`, and the forward when the weight is held transposed).
  * `gemm_nt(...)`: `C[m,n] = A[m,k] . B[n,k]^T` (forward `x W^T`): `gemm_nn` on a transposed copy of the weight that
    `ppo_update` refreshes **once per epoch** (the weights change once per epoch); a one-off call transposes into
    scratch.
  * `gemm_tn(...)`: `C[m,n] = A[k,m]^T . B[k,n]` (backward `dW = dz^T x`): outer-product accumulation with the same
    4x16 tile.
  Panel packing and `KC`/`MC` blocking are added only if G2 still fails on the integrated `ppo_update` (17, R1,
  fallback 1); they are not part of the first implementation.
* Vector kernels on raw pointers: `add_bias_rows`, `relu_inplace`, `relu_backward(dx, pre, n)` (`dx *= (pre > 0)`,
  torch rule: gradient 0 at exactly 0), `col_sum_acc(g, m, n, out)` (`db += sum_rows g`), `saxpy`, `sumsq_f64`.
* Single-threaded, re-entrant kernels; threading happens one level up (virtual workers). The API takes
  `MutPointer[Float32, MutAnyOrigin]` obtained with `lst.unsafe_ptr().as_unsafe_any_origin()`; callers keep the
  owning lists alive with `_ = len(lst)` after the parallel region (ASAP destruction, MOJO_GUIDE).
* Performance gate G1 (section 15) is measured by `bench/bench_rl_gemm.mojo` (C4) before C5 starts. All kernels are
  raw-pointer code with no bounds checks (bounds checking covers `List[]` indexing only), so `test_rl_tensor` allocates
  every buffer with sentinel guard regions (13.2) and runs once with `-D ASSERT=all`.

### 4.3 `optim.mojo`

`Adam` over one flat parameter buffer, exactly the torch single-tensor CPU path (torch.md 4):

```
t += 1                                        # one counter per optimizer, never reset (also across learn_steps)
m   = m + (g - m) * (1 - beta1)               # lerp form, f32
v   = v * beta2 + (1 - beta2) * g * g         # f32
bc1 = 1 - beta1^t ; bc2 = 1 - beta2^t         # f64
step_size = lr / bc1 ; denom = sqrt(v) / sqrt(bc2) + eps     # eps added AFTER the division
p   = p - step_size * m / denom               # scalars computed in f64, cast to f32 per operation
```
* Defaults beta (0.9, 0.999), eps 1e-8, no weight decay, no amsgrad.
* **Dense update on every element** (momentum on rows with zero gradient still moves them). An exact optimisation is
  optional and not needed for any gate; add it only if profiling shows the embedding update matters (the table is
  `OBS x E`, 36 x 512 for the shipped models): keep a per-row `touched: List[Bool]`; rows never touched have
  `m = v = 0` and update exactly 0, so they may be skipped (`mark_sparse_rows`). The BC value head is *excluded*, not
  zero-gradded: `Adam.set_frozen(off, len)` marks a slice as "grad is None" (no step, no state), reproducing the torch
  behaviour in torch.md 3.2.
* `clip_grad_norm(grads, ranges, max_norm)`: `total = sqrt(sum_over_all_tensors sumsq)` (per-tensor f32 norms
  accumulated like torch), `coef = min(max_norm / (total + 1e-6), 1.0)`, multiply every gradient by `coef`
  unconditionally. Router PPO only.
* `cosine_epoch_lr(base, epoch, total) = base * (1 + cos(pi * epoch / total)) / 2` (calls `qml_train.cosine_lr`;
  test against `adam_cosine_golden.json`).

### 4.4 `nn.mojo`: parameter store and the synthesis network

* `ParamStore`: one flat `List[Float32]`; `add(name, rows, cols) -> offset`; named lookups; `GradBufs` = `VW`
  flat buffers of the same length plus `reduce_into(out)` (sum in buffer order). Adam, clipping, checkpointing and
  the container writer all operate on this single flat buffer; layers hold offsets.
* `BasicNet` mirrors twisterl `BasicPolicy` (ppo.md 2):
  `embeddings`: `Wt [OBS, E]` (the **container layout, transposed vs torch**) and `be [E]`;
  `common`: Linear(E->C1) ReLU ... ;
  `action`: Linear chain ending in Linear(Ck->A) with ReLU between hidden layers and none after the last;
  `value`: same ending in Linear(Ck->1).
  Initialisation order (the RNG draw order): embeddings, common.*, action.*, value.*; each Linear draws weight
  then bias from `U(-1/sqrt(fan_in), 1/sqrt(fan_in))`, `fan_in = in_features` of the weight (embeddings: `OBS`).
* Sparse embedding bag: `h0 = relu(be + sum_{i in S'} Wt[i, :])` with `S' = {obs_perm[p][i]}` when the row has a
  twist `p >= 0`; no dense one-hot is ever built. Duplicate indices add twice (qiskit-gym never produces them).
* `forward_one(idx, twist, scratch, logits_out) -> Float32 value`: single-state actor path (heap-free scratch),
  logits are *post-act-perm*: `logits[a] = raw[act_perm[p][a]]`.
* `forward_chunk(sb, lo, hi, cache)` and `backward_chunk(sb, lo, hi, cache, dz, dv, grad_ptr)`: batched path.
  Backward order: heads (`dWa = dz^T h`, `dba`, `dh += dz Wa`; same for value; **the two head gradients are summed
  into the shared hidden gradient**), ReLU masks `(pre > 0)`, common layers in reverse, embedding bag scatter
  `dWt[obs_perm[p][i], :] += dpre0[b, :]`, `dbe += dpre0[b, :]`. The un-twist of the logit gradient is
  `draw[act_perm[p][a]] = dz[a]`.
* `SparseBatch`: CSR (`off: List[Int32]` length N+1, `idx: List[Int32]`, `perm: List[Int32]` with -1 for none).
* `NetSpec.params()` count equals `OBS*E + E + E*C + C + C*A + A + C + 1` for one common layer (check against the 9
  shipped manifests: 153,099 for OBS 36, A 10).

### 4.5 `loss.mojo`: synthesis PPO loss, exact formulas

Per row (post-twist logits `z` of length A, **unmasked**, value `v`, action `a`, advantage `adv`, return `ret`,
recorded `old_lp`), batch size `N`, constants `c = clip_ratio`, `vf`, `ent`:

```
lp_all = log_softmax(z) (max-shifted, f32) ; p = exp(lp_all) ; lp = lp_all[a] ; H = -sum p * lp_all
ratio  = exp(lp - old_lp) ; s1 = ratio * adv ; s2 = clamp(ratio, 1-c, 1+c) * adv
policy_loss = -(1/N) sum min(s1, s2) ; value_loss = (1/N) sum (v - ret)^2 ; entropy = (1/N) sum H
loss = policy_loss + vf * value_loss - ent * entropy

g_ratio = adv                          if s1 < s2
        = adv * [1-c <= ratio <= 1+c]  if s2 < s1
        = adv                          if s1 == s2                       (torch splits the tie 50/50: total adv)
dL/dlp  = -(1/N) * g_ratio * ratio
dz_j    = dL/dlp * ([j == a] - p_j) + (ent / N) * p_j * (lp_all_j + H)
dv      = vf * 2 * (v - ret) / N
```
`old_lp` is computed once per iteration from the recorded *masked* logits as `l_a - logsumexp(l)` in f32 (max-shift
form). For an all-masked row (`-1e10` everywhere) this gives exactly 0.0, **not** `-ln A`; the golden batch row 13
in `ppo_update.json` confirms 0.0 (section 18, C-1). `PPOConfig.normalize_advantage` (off in QUMO) uses the
unbiased std (`N-1`) plus 1e-8.

GAE (twisterl form, f32, ppo.md 4): `advs[n-1] = r[n-1] - v[n-1]; rets[n-1] = r[n-1]; for t = n-2..0:
rets[t] = r[t] + gamma * (v[t+1] + lam * advs[t+1]); advs[t] = rets[t] - v[t]`. The reward of row `t` is the reward
of the state `t` itself (the reward earned by the previous action). The terminal row bootstraps from nothing (there
is no `v[n]` term). Both products (`lam * advs[t+1]` and `gamma * (...)`) go through `mulf` (section 3, rule 9) so the
recursion rounds like the reference; the golden tolerance is 1e-6 either way. The wiring of GAE into the collector
(reward alignment, terminal row, per-row twist and mask) is pinned by the rollout-statistics fixture (13.1, 13.2).

### 4.6 `topology.mojo`, `symmetry.mojo`

* Spec grammar of `common/topology.py`: `line:N`, `ring:N`, `grid:RxC`, `king:RxC`, `complete:N`, `star:N`,
  `heavyhex_frag:{H7,cell12,ring10s2,T5}`, `edges:a-b,c-d`; result `(n, edges)` with edges sorted `(a<b)`.
  `topology_slug(spec)`: for `edges:` specs use `fnv` hex of the normalised list (replaces sha1; only used for
  directory names); `is_connected`. WL hash is **not** re-implemented: `policy.wl_hash(n, flat_edges)`; the test
  asserts the 9 manifests' `wl_hash` values in INDEX.json.
* `automorphisms(n, adj) -> List[List[Int]]`: backtracking over degree / adjacency-consistent images, result sorted
  lexicographically (identical order to `sorted(out)` of the Python version and to the Rust enumeration after
  `sort + dedup`; if the graph has no edge, all `n!` permutations).
* `twists(kind, n, actions) -> Twists{obs_perms, act_perms}` for permutation / linear / Clifford: adjacency from
  the 2-qubit gates, for each automorphism `perm` build `act_perm` (relabel every gate, SWAP keys unordered, CX/CZ
  ordered, 1q keyed by qubit; skip the automorphism if any image is missing), `obs_perm` for squares
  `[r*n+c] = perm[r]*n+perm[c]` and Clifford `m(i)*2n+m(j)` with `m(i) = perm[i]` for `i<n` else `n+perm[i-n]`;
  if none kept use identity when its action perm exists.
* `pauli_twists(...)` for the Pauli *export* ensemble (inference side, reproduces `symmetries` of `pauli_line4`
  exactly) and `pauli_qubit_perms_and_act_perms(...)` for the *training* env (envs.md 9.4 / 9.5; the table in
  `gateset_twists.json` for Pauli kinds). **The direction (inverse or direct relation to the env's internal
  `qubit_perms` / `act_perms`) cannot be settled by the `pauli_line4` manifest**: `line:4` has only two automorphisms,
  identity and reversal, both involutions (verified on the manifest: both `act_perms` square to the identity), so the
  direct and the inverse table coincide there, and a wrong direction on `ring:5`, a grid or a heavy-hex would load,
  pass every shipped-model test and quietly average the wrong views at inference. It is therefore pinned by two
  independent checks that must both pass (O-5 is resolved by method, not by prose):
  1. a fixture `envs/pauli_twists.json`, dumped by C0 from the Python `pauli_twists` before `symmetry.py` is deleted,
     for `ring:5`, `ring:4`, `star:4`, `grid:2x3`, `heavyhex_frag:H7` and `heavyhex_frag:T5` (`ring:5`, `ring:4` and
     `star:4` have non-involutive automorphisms, which is what distinguishes the two directions), asserted element-wise
     equal (`obs_perms`, `act_perms`, order included);
  2. the behavioural self-check the Python exporter had (`gym_dense_perm` versus `pauli_twists`), ported as
     `check_pauli_ensemble(...)` in `export_synth.mojo` (C11a; it needs `TrainEnv`, which `symmetry.mojo` must not
     import): for a random-weight policy, a random Pauli state and every symmetry `p`, the logits of the ensemble view
     (`obs_perms[p]`, `act_perms[p]`) must equal the logits of `TrainEnv`'s hidden-permutation view (`pauli_obs_map[p]`
     observation, `act_perms[p]` un-permutation). It runs as a test on `ring:5` (and the other fixture topologies) and as
     a runtime assertion in `export_synth` (9.2).
* Golden: re-running `twists` on the 9 shipped manifests' `training.topology_spec` reproduces
  `symmetries.obs_perms/act_perms` element for element (repo.md 2.2 verified this with the Python code).

### 4.7 `container.mojo`, `digest.mojo`, `f16.mojo`

Byte-level format and checks are in section 9.1. `digest.mojo`: `sha256(bytes) -> String(64 hex)` streaming over
64-byte blocks (`Sha256` struct with `update`/`finalize` for 1 MiB file chunks), FIPS 180-4 known-answer tests;
`crc32(bytes) -> UInt32` (IEEE, reflected 0xEDB88320, table-driven) for `eval_sets` (`zlib.crc32(filename)`).
`f16.mojo`: `f32_to_f16_bits(x) -> UInt16` (via `Float16`), `encode_f16(src: List[Float32], tensor_name) raises ->
List[UInt8]` raising `tensor 'X' overflows f16` when any result is non-finite.

---------------------------------------------------------------------------------------------------------

## 5. Normative signatures (contract between components)

Names and semantics are normative; pointer / ownership spellings follow `docs/MOJO_GUIDE.md` (1.2 dialect:
`def`, `raises`, `comptime`, `mut`, `out self`). All code in this section is meant to compile: before wave 1 the
lead compiles the whole block as one stub file (not committed); probes in r2 showed that `InlineArray` does not exist
in Mojo 1.2, so fixed-size fields are spelled as named fields; `FPtr` is the pointer alias of 4.2; fields written on
one line with `;` are documentation shorthand, one field per line in the stubs. A component may add helpers but may not change or remove
anything listed here without a lead-approved revision. Errors use `QM-E640`..`QM-E649` (reserved here, registered by
C13 in `qumo/ai/diagnostics.mojo` and `docs/ERRORS.md`; E640 usage, E641 bad topology spec, E642 environment
config, E643 training diverged (NaN/inf), E644 container write / read-back mismatch, E645 f16 overflow, E646
teacher data malformed, E647 warm-start incompatible, E648 campaign file, E649 eval/keep-rule failure).

```mojo
# ------------------------------------------------------------------ rlrng.mojo   [C4]   (imports FPtr from tensor.mojo)
comptime TAG_INIT = 1  # ... TAG_COLLECT = 2, TAG_EVAL_DET = 3, TAG_EVAL_SAMPLED = 4, TAG_BC_SPLIT = 5,
                       # TAG_BC_EPOCH = 6, TAG_BC_SLOT = 7, TAG_JOBS = 8, TAG_EXPORT_VEC = 9, TAG_ROUTE_PPO = 10
def splitmix64(x: UInt64) -> UInt64
def seed_for(base: UInt64, tag: Int, a: Int = 0, b: Int = 0, c: Int = 0) -> UInt64
def uniform_f32(mut r: Rng) -> Float32
def uniform_k24(mut r: Rng) -> Int                       # integer k in [0, 2^24)
def uniform_sym_f32(mut r: Rng, bound: Float32) -> Float32
def normal_f32(mut r: Rng) -> Float32
def randperm(mut r: Rng, n: Int) -> List[Int]
def gumbel_argmax(logits: FPtr, n: Int, mut r: Rng) -> Int         # first maximum wins (strict >), NaN never wins
def sample_softmax(logits: FPtr, n: Int, mut r: Rng) -> Int        # max-shifted softmax, inverse CDF
@no_inline
def mulf(a: Float32, b: Float32) -> Float32                          # FMA barrier (section 3, rule 9)

# ------------------------------------------------------------------ tensor.mojo  [C4]
comptime FPtr = MutPointer[Float32, MutAnyOrigin]   # the one pointer alias of the contract
comptime LANES = 8
comptime CHUNK_ROWS = 512
comptime VW = 16
struct Mat(Movable, Copyable):
    var rows: Int
    var cols: Int
    var data: List[Float32]
    def __init__(out self, rows: Int, cols: Int)
    def ptr(mut self) -> FPtr
def gemm_nt(m: Int, n: Int, k: Int, a: FPtr, lda: Int, b: FPtr, ldb: Int, c: FPtr, ldc: Int, acc: Bool)
def gemm_nn(m: Int, n: Int, k: Int, a: FPtr, lda: Int, b: FPtr, ldb: Int, c: FPtr, ldc: Int, acc: Bool)
def gemm_tn(m: Int, n: Int, k: Int, a: FPtr, lda: Int, b: FPtr, ldb: Int, c: FPtr, ldc: Int, acc: Bool)
def relu_backward(dx: FPtr, pre: FPtr, n: Int)
def col_sum_acc(g: FPtr, m: Int, n: Int, out: FPtr)
def sumsq_f64(p: FPtr, n: Int) -> Float64

# ------------------------------------------------------------------ optim.mojo   [C4]
struct Adam(Movable):
    def __init__(out self, n_params: Int, lr: Float64, beta1: Float64 = 0.9, beta2: Float64 = 0.999, eps: Float64 = 1e-8)
    def set_frozen(mut self, off: Int, length: Int)       # grad is None: no step, no state
    def mark_sparse_rows(mut self, off: Int, row_len: Int, rows: Int)   # skip never-touched rows (exact)
    def step(mut self, mut params: List[Float32], grads: List[Float32], lr: Float64)
    def steps_taken(self) -> Int
def clip_grad_norm(mut grads: List[Float32], ranges: List[Int], max_norm: Float64) -> Float64  # returns pre-clip norm
def cosine_epoch_lr(base: Float64, epoch: Int, total_epochs: Int) -> Float64

# ------------------------------------------------------------------ nn.mojo      [C5]
struct ParamStore(Movable, Copyable):
    def add(mut self, name: String, rows: Int, cols: Int) -> Int
    def total(self) -> Int
    def offset(self, name: String) raises -> Int
    var data: List[Float32]
struct SparseBatch(Movable):
    var off: List[Int32]
    var idx: List[Int32]
    var perm: List[Int32]                 # -1 = no twist
    def rows(self) -> Int
struct NetSpec(Copyable):
    var obs_size: Int
    var num_actions: Int
    var emb: Int
    var common: List[Int]
    var policy_hidden: List[Int]
    var value_hidden: List[Int]
    var obs_perms: List[List[Int]]
    var act_perms: List[List[Int]]
struct ChunkCache(Movable):               # per virtual worker scratch for up to CHUNK_ROWS rows
    def __init__(out self, spec: NetSpec)
struct ActorScratch(Movable):             # per task scratch for forward_one
    def __init__(out self, spec: NetSpec)
struct BasicNet(Movable):
    var spec: NetSpec
    var ps: ParamStore
    def __init__(out self, spec: NetSpec, seed: UInt64) raises
    @staticmethod
    def from_model(m: ModelFile) raises -> BasicNet          # warm start from any container (f16 decoded)
    def forward_one(self, idx: Span[Int32], twist: Int, mut sc: ActorScratch, logits: FPtr) -> Float32
    def forward_chunk(self, sb: SparseBatch, lo: Int, hi: Int, mut cc: ChunkCache)       # logits/values in cc
    def backward_chunk(self, sb: SparseBatch, lo: Int, hi: Int, mut cc: ChunkCache, dz: FPtr, dv: FPtr, grad: FPtr)
    def param_count(self) -> Int

# ------------------------------------------------------------------ loss.mojo    [C5]
struct PPOLossCfg(Copyable):  var clip: Float32;  var vf_coef: Float32;  var ent_coef: Float32
struct LossParts(Copyable):   var policy: Float64; var value: Float64; var entropy: Float64   # partial sums, not yet /N
def log_softmax_row(z: FPtr, n: Int, out_lp: FPtr)
def logsumexp_row(z: FPtr, n: Int) -> Float32
def ppo_chunk_loss(logits: FPtr, value: FPtr, rows: Int, A: Int, acts: FPtr, advs: FPtr, rets: FPtr, old_lp: FPtr,
                   cfg: PPOLossCfg, inv_n: Float32, dz: FPtr, dv: FPtr) -> LossParts
def gae_twisterl(rewards: Span[Float32], values: Span[Float32], gamma: Float32, lam: Float32,
                 mut advs: List[Float32], mut rets: List[Float32])

# ------------------------------------------------------------------ gateset.mojo / synth_env.mojo   [C6]
struct EnvConfig(Copyable):
    var kind: Int                         # MODEL_KIND_PERM | LINEAR | CLIFFORD | PAULI
    var n: Int
    var actions: List[Int]                # flat (opcode, q0, q1|-1) in gateset order
    var difficulty: Int                   # 1
    var depth_slope: Int                  # 2
    var max_depth: Int                    # 128
    var w_cnots: Float32                  # metrics weights: n_cnots 0.01
    var w_layers_cnots: Float32           # 0.0
    var w_layers: Float32                 # 0.0
    var w_gates: Float32                  # 0.0001
    var add_inverts: Bool                 # True (plain kinds)
    var add_perms: Bool                   # True
    var max_rotations: Int                # 5 (Pauli)
    var pauli_diff_scale: Int             # 16
    var num_qubits_decay: Float32         # 0.5
    var final_pauli_layers: Int           # 7
    var pauli_layer_reward: Float32       # 0.01
def build_gateset(kind: Int, topo_n: Int, edges: List[Int]) raises -> List[Int]   # flat triples, qiskit-gym order
# the environment's n is (max qubit over the edges) + 1, as in qiskit-gym; QM-E641 if it differs from topo_n (a trailing
# isolated qubit); the emitted gate names are only {H, S, CX, SWAP} (6.1)
def validate_actions(cfg: EnvConfig) raises                 # QM-E642 for any gate outside {H, S, CX, SWAP} or bad qubit ids
def shell_model(cfg: EnvConfig) -> ModelFile                                    # metadata-only ModelFile for PolicyEnv
struct TrainEnv(Copyable):
    def __init__(out self, cfg: EnvConfig, seed: UInt64) raises
    def num_actions(self) -> Int
    def obs_size(self) -> Int
    def reseed(mut self, seed: UInt64)
    def set_difficulty(mut self, d: Int)
    def reset(mut self)
    def set_state_perm(mut self, s: List[Int])
    def set_state_bits(mut self, v: List[Int])               # LF / Clifford flat 0/1 matrix
    def set_state_pauli(mut self, v: List[Int]) raises       # envs.md 9.6 vector
    def observe(mut self, mut obs: List[Int])                # sorted ascending; Pauli draws + stores the hidden symmetry
    def all_allowed(self) -> Bool                            # masks() == [not success] * A
    def is_final(self) -> Bool
    def success(self) -> Bool
    def reward(self) -> Float32
    def step(mut self, a: Int)                               # a >= A with add_perms: clamped to 0, error_code() set, never raised
    def twists(self) -> Twists                               # plain kinds: symmetry.twists; Pauli: empty
    def error_code(self) -> Int                              # 0 = ok; set (never raised) inside tasks, section 3 rule 7

# ------------------------------------------------------------------ ppo.mojo     [C7]
struct PPOConfig(Copyable):
    var num_cores: Int; var num_episodes: Int; var gamma: Float32; var lam: Float32
    var num_epochs: Int; var vf_coef: Float32; var ent_coef: Float32; var clip_ratio: Float32
    var normalize_advantage: Bool; var lr: Float64
    var diff_threshold: Float32; var diff_max: Int
    var eval_episodes: Int; var eval_sampled_searches: Int; var eval_sampled_every: Int
    var seed: UInt64; var chunk_rows: Int; var start_iteration: Int      # global iteration offset (section 3, rule 1)
    var mask_new_logits: Bool; var best_checkpoint: String         # D6 switches, faithful defaults
struct Rollout(Movable):
    var sb: SparseBatch; var logits: List[Float32]; var acts: List[Int32]; var advs: List[Float32]
    var rets: List[Float32]; var old_lp: List[Float32]; var n: Int; var num_episodes: Int
    var status: Int                                                # first non-zero item status in episode order (section 3 rule 7)
struct EvalResult(Copyable):  var success: Float32; var reward: Float32; var status: Int
struct LossStats(Copyable):   var value: Float64; var policy: Float64; var entropy: Float64; var total: Float64
struct StepReport(Copyable):
    var det: EvalResult; var sampled: EvalResult; var have_sampled: Bool
    var difficulty: Int; var loss: LossStats; var rows: Int; var step_s: Float64
def collect(env: TrainEnv, net: BasicNet, cfg: PPOConfig, iteration: Int) -> Rollout
def evaluate(env: TrainEnv, net: BasicNet, cfg: PPOConfig, iteration: Int, deterministic: Bool,
             num_searches: Int, num_episodes: Int) -> EvalResult
def ppo_update(mut net: BasicNet, mut opt: Adam, ro: Rollout, cfg: PPOConfig) -> LossStats
struct Trainer(Movable):
    var net: BasicNet; var opt: Adam; var env: TrainEnv; var cfg: PPOConfig; var iteration: Int
    def __init__(out self, env: TrainEnv, net: BasicNet, cfg: PPOConfig)
    def learn_step(mut self) raises -> StepReport           # evals -> collect -> update; does NOT promote difficulty; raises the
                                                            # first status-slot error (QM-E642) or QM-E643 on a non-finite loss
    def promote_if_ready(mut self, rep: StepReport) -> Bool # success >= threshold and difficulty < diff_max

# ------------------------------------------------------------------ container.mojo / digest.mojo / f16.mojo   [C2, C1]
def sha256_hex(data: List[UInt8]) -> String
def crc32(data: List[UInt8]) -> UInt32
def encode_f16(src: List[Float32], tensor_name: String) raises -> List[UInt8]
struct TensorOut(Copyable):
    var name: String; var shape: List[Int]; var data: List[Float32]; var dtype: String   # "f16" | "f32"
struct ModelDirInfo(Copyable):  var weights_bytes: Int; var fnv1a64: String; var sha256: String; var manifest_bytes: Int
# manifest_head: the manifest's JSON object members that precede "tensors" (text, no trailing comma); the writer emits
# the "tensors" array (name, dtype, shape, offset, bytes) itself; manifest_tail: the members that follow, as JSON text
# without the closing brace ("" if none). Atomic: writes <dir>.tmp, reads it back, then swaps it into <dir> (9.1).
def write_model_dir(dir: String, manifest_head: String, tensors: List[TensorOut], dtype: String,
                    f32_names: List[String], manifest_tail: String) raises -> ModelDirInfo
def atomic_replace_dir(tmp: String, final: String) raises   # libc rename swap, 9.1
def recover_dir(final: String) raises                       # restores <final>.old if <final> is missing, drops stale .tmp
def rebuild_index(root: String, sub: String, default_id: String) raises            # synth: default_id == ""
def verify_models(root: String) -> VerifyReport                                     # same text output as the old make target

# ------------------------------------------------------------------ router   [C8]
struct TeacherData(Movable):  var n: Int; var obs: List[Int8]; var A: List[Int16]; var act: List[Int16]
                              var job: List[Int16]; var tb: List[Int16]; var q: List[Float32]
def read_teacher_files(paths: List[String]) raises -> TeacherData
struct RouterNet(Movable):    def __init__(out self, hidden: Int, seed: UInt64);  @staticmethod def from_model(m: ModelFile) raises
struct BCConfig(Copyable):    var hidden: Int; var tau: Float32; var lr: Float64; var epochs: Int; var batch: Int
                              var seed: UInt64; var budget_s: Float64; var threads: Int
struct BCResult(Copyable):    var log: List[BCEpochLog]; var samples: Int; var val: Int; var wall_s: Float64
def train_bc(cfg: BCConfig, data: TeacherData, init: Optional[ModelFile]) raises -> Tuple[RouterNet, BCResult]
```
The signatures of `route_ppo.mojo`, `checkpoint.mojo`, `export_synth.mojo`, `export_route.mojo`, `jobs.mojo`,
`targets.mojo` are fixed in their component sections (16.x) because only drivers consume them.

---------------------------------------------------------------------------------------------------------

## 6. Environments (qiskit-gym port)

Spec of record: `envs.md` (sections 2-11) plus the golden files. This section fixes how it maps to code.

### 6.1 Layering: reuse `PolicyEnv`, add the training shell

For all four kinds and for the gate subset covered by the goldens (see the scope note below), `PolicyEnv` already
implements on the packed `List[UInt64]` state exactly what the Rust envs do for state transitions: `apply` (all gate
actions, the mirrored Pauli CX convention, `clean()` with petgraph swap-remove node order), `solved`, `observe`
(sparse indices, first `max_rotations` active rotations), `inverse` (GF(2) Gauss-Jordan; permutation inverse).
`TrainEnv` wraps one `PolicyEnv` plus what `refenv.py` / `PolicyEnv` deliberately lack (envs.md 15):

**Scope of the reuse claim (r2).** The step goldens (`gateset_twists.json`, `steps_*.json`) contain only the gate names
{H, S, CX, SWAP}, and `build_gateset` emits only those. Reading the code against the reference shows that
`PolicyEnv`'s Pauli `CZ` branch calls `_pauli_cx(st, a, b)` (`row[b] ^= row[a]`), whereas the reference implements
`CZ` as `H(q1); CNOT(q0, q1); H(q1)` with the mirrored CNOT (`row[q0] ^= row[q1]`), which is the pattern of
`PolicyEnv`'s own CX branch, `_pauli_cx(st, b, a)`; `Sdg`, `SX` and `SXdg` handling is unverified for Pauli. This is
latent for QUMO today. `TrainEnv` therefore accepts only {H, S, CX, SWAP} in `EnvConfig.actions` (`validate_actions`,
QM-E642 for anything else), R16's "the golden tests exercise `PolicyEnv`" holds for that subset only, and the CZ
discrepancy is recorded as C-8 (18.1) with a deferred fix: add a Pauli/Clifford golden with CZ, SX and Sdg, then
correct `policy_synth.mojo` additively together with its inference tests.

1. `build_gateset` and `shell_model`: `PolicyEnv.__init__` wants a `ModelFile`; `shell_model(cfg)` returns a
   default `ModelFile()` with `num_qubits`, `actions`, `max_rotations` set. No change to `policy_synth.mojo`.
   `build_gateset(kind, topo_n, edges)` derives the environment's qubit count as `max qubit + 1` over the edges (as
   qiskit-gym does) and raises QM-E641 if it differs from the topology's `topo_n`: a trailing isolated qubit would
   silently change `obs_size` and the symmetries relative to what qiskit-gym trains, and the old trainer aborted on it.
2. **Reset generators**: plain kinds apply `difficulty` uniform random actions to the identity (no inversion, no
   rejection); `depth = min(depth_slope * difficulty, max_depth)`. Pauli: `pauli_difficulty = difficulty // 16`,
   rotation list from `generate_paulis` (envs.md 9.7, including the label-mirroring quirk: position `q` of the
   generated label becomes qubit `n-1-q`), tableau from `random_clifford_tableau(difficulty)` with the integer
   thresholds `k >= 5033166` (CX), `2516583 <= k <= 5033165` (H), `k <= 2516582` (S), `k` uniform on `[0, 2^24)`;
   decay stop `k <= 2^23`. Pauli state is assembled in the `PolicyEnv` packed layout with `R = number of
   generated rotations` (<= `final_pauli_layers`), then `init_pauli(R, st)` builds the anticommutation table.
3. **Metrics and reward**: `Metrics` tracker (n_cnots, n_gates, `last_gates[]`, `last_cxs[]`, distinct-layer sets
   as growable byte arrays), `snapshot()`, saturating integer deltas converted to f32, penalty
   `((w0*d0 + w1*d1) + w2*d2) + w3*d3` in f32, `reward_value = (achieved - penalty) + (pauli_layer_reward *
   f32(removed))`. `removed` for Pauli = node count before minus after the step (the count word of the packed state).
   Every product (`w_i * d_i`, `pauli_layer_reward * f32(removed)`) goes through `mulf` (section 3, rule 9): without
   the barrier Mojo fuses `(achieved - penalty) + (w * c)` and the reward differs by 1 ulp for some counts (for
   example achieved 0, penalty 1e-4, w 0.01, count 5), which the shipped step goldens cannot detect;
   `envs/reward_fma_cases.json` pins it.
4. **Random inversion** (plain kinds, `add_inverts`): after the gate and before `solved()`, with probability 1/2,
   `state = inverse(state)` via `PolicyEnv.inverse`. Never at reset / set_state.
5. **Pauli hidden symmetry**: `observe()` draws `p` uniformly in `[0, n_sym)` (when `add_perms`), maps the sparse
   indices through a precomputed `pauli_obs_map[p][flat]` (envs.md 9.4 `apply_perm`: rows permuted, tableau columns
   permuted, rotation columns fixed), stores `cur_perm = p` (initial 0); `step(a)` maps `a -> act_perms[cur_perm][a]`.
   `step(a >= A)` with `add_perms` is the Rust panic; since tasks never raise (section 3, rule 7) the port clamps the
   action to 0, sets `error_code() = 642` and the caller raises QM-E642 after the parallel region; with
   `add_perms == False` it is a silent no-op.
6. `masks`: `all_allowed() == (not success)`; `is_final() == depth == 0 or success`; `step` always decrements depth
   (saturating) even when final; construction state is identity, `depth = 1`, `success = True`, `reward = 1.0`.
7. `set_state_*` converters (needed by the golden tests only): `depth = max_depth`, `reward = 1.0 if solved else 0.0`,
   metrics reset, Pauli `set_state` keeps only the first `max_rotations` labels and does **not** clean.
8. `twists()`: plain kinds -> `symmetry.twists`, used by the trainer to build `NetSpec.obs_perms / act_perms`;
   Pauli -> empty (the policy has no twists while the env permutes internally), exactly as `PauliGym`.

Per-episode cost: one `TrainEnv` copy (the packed state, the small tables). `TrainEnv` is `Copyable`; the heavy
read-only tables (distance pairs, `pauli_obs_map`, `act_perms`) live behind an `ArcPointer` shared by copies.

### 6.2 Random numbers

`TrainEnv` owns an `Rng`; `reseed` installs the per-episode generator. The draw order inside one episode is part
of the determinism contract: `reset()` draws, then per step the inversion bit (plain) or symmetry index (Pauli).
Because the Rust draw order cannot be reproduced, reset and inversion are tested *distributionally*; set_state /
step / observe / reward / is_final are tested *exactly* (envs.md "Gaps").

### 6.3 Not modelled (not needed for training)

`track_solution`, `solution()` reconstruction, rotation phases (`base_phase`), `get_state` circuit conversion
(`get_state.json` is only a fixture for a possible later "solve a real circuit with the add_inverts env" feature,
which would need `solution ++ reversed(solution_inv)`; the exported-model path uses `PolicyEnv` with the inversion
trick that already exists in `search_best`).

---------------------------------------------------------------------------------------------------------

## 7. PPO trainer (twisterl port)

Spec of record: `ppo.md` and `torch.md` 3.5. Constants in effect for QUMO: gamma 0.995, lam 0.995, 10 epochs of
full-batch Adam (lr 3e-4, one optimizer for the whole run), clip 0.1, vf 0.8, ent 0.01, no advantage
normalisation, 1024 episodes per iteration, evals `ppo_deterministic` (200 episodes, argmax, 1 search) and
`ppo_10` (200 episodes, sampled, 10 searches), diff_threshold 0.85, diff_max 256, start difficulty 1.

### 7.1 `learn_step` (one iteration)

```
1. evals (net unchanged, env at current difficulty d):
     det = evaluate(deterministic, searches=1, episodes=200)        # drives the curriculum
     samp = evaluate(sampled, searches=10, episodes=200)            # logged; computed every eval_sampled_every iterations (default 5; 1 = faithful)
2. ro = collect(env, net, 1024 episodes)                            # parallel over episodes, seeded per episode
3. old_lp from the recorded masked logits; advantages are already in ro (GAE per episode)
4. for epoch in 0..10: one full-batch gradient step over ALL rows of ro       # ppo_update
5. return StepReport (loss of the LAST epoch, evaluation of the pre-update policy)
```
`Trainer.promote_if_ready(rep)`: `Float64(rep.det.success) >= Float64(diff_threshold)` where `success` is the f32
quotient `n_ok / 200` and `diff_threshold` is the f32 value of 0.85 (so exactly 170/200 promotes, ppo.md 8);
at most +1 per iteration, never demoted.

### 7.2 Collector (episode task)

```
env' = copy of env (seeded: env stream); env'.reset()
loop:
    obs   = env'.observe()                         # exactly one observe per decision (Pauli relies on it)
    allowed = env'.all_allowed(); r = env'.reward()
    t = actor stream: uniform twist index in [0, T) if T > 0 else -1
    (logits, v) = net.forward_one(obs, t)          # post act-perm
    masked = logits if allowed else all -1e10 (f32)
    a = gumbel_argmax(masked)                      # actor stream
    record(obs, masked, a, v, r, t)                # the terminal state is recorded too
    if env'.is_final(): break
    env'.step(a)
GAE on the episode, then append in episode-index order
```
Rows are appended to preallocated per-episode buffers by the parallel task and concatenated in index order after
`parallelize` (the merge order of twisterl, "last episode first", is irrelevant to a full-batch loss).

The collector wiring that the unit goldens cannot see (row `t` carries the reward of state `t`, the terminal row is
recorded with all-masked logits and no bootstrap, the twist drawn for the decision is the one used in the update,
`evaluate` reward accounting) is pinned externally by the rollout-statistics fixture `ppo/rollout_stats.json` (13.1:
produced with the real qiskit-gym env and the real twisterl collector/evaluate on fixed explicit weights) in
`test_rl_ppo_slow` with 4-sigma bounds, plus the collector self-consistency test of `test_rl_ppo`. A learning smoke
test would pass with an off-by-one reward alignment, so it is not the equivalence criterion for these.

### 7.3 `ppo_update`

Per epoch: refresh the transposed weight copies (4.2); `parallelize(vw_body, VW, cores)`; virtual worker `w` processes chunks `w, w+VW, ...`: `forward_chunk`,
`ppo_chunk_loss` (uses the **new** unmasked logits; `old_lp` from the batch), `backward_chunk` into `grads[w]`;
then `reduce_into` -> `Adam.step`. Loss values reported are those of the last epoch's forward (as torch returns
them). `mask_new_logits = True` (non-faithful) would mask new logits with the row's `allowed` flag before the
softmax; it needs the per-row flag stored in the rollout (`allowed: List[UInt8]`), cheap, off by default.

### 7.4 `evaluate`

Per episode (task): env copy, `reset()`, repeat `num_searches` from the *same* start state (a clone): loop
`while not is_final { total += reward; obs; (logits) with a fresh random twist; a = argmax(logits) (first max wins)
or sampled from softmax(logits); step }`, `total += reward`; candidate `(1.0 if success else 0.0, total)`; keep the
best by lexicographic compare (strict `>`, first wins). Results: `(sum success / E, sum reward / E)` accumulated in
f32 in episode order. Masks never matter inside `evaluate` (states are unsolved until final). The argmax is the first maximum of the
logits; Rust's argmax over un-shifted `exp(logit) / (sum + 1e-6)` differs only for logits above about 88 (inf/NaN
probabilities, index 0), a deliberate deviation recorded in 4.1 and covered by a unit test with logits of 100. The
success and reward accounting of `evaluate` is pinned by the rollout-statistics fixture (13.1).

### 7.5 Driver loop (`synth_train.mojo`)

```
stop rule: exactly one of --iters N (stop after N iterations of this process) and --budget-s S (stop once
           elapsed >= S, checked between iterations; D7); both or neither is QM-E640
env.difficulty = start_diff ; it = 0 ; stop = False        # global iteration git = start_iteration + it (section 3)
while not stop:
    rep = trainer.learn_step()
    cur = (env.difficulty, rep.det.success, rep.det.reward)    # lexicographic
    if best is None or cur >= best:                            # D6: weights AFTER this update (faithful)
        best = cur ; write best/ checkpoint (atomic, 9.1)
    trainer.promote_if_ready(rep)
    append log.jsonl ; every 25 its print
    it += 1
    if it % checkpoint_every == 0: write last/ + state.json + adam.bin (atomic)
    stop = (it >= iters) if iters mode else (elapsed >= budget_s)
write last/ ; summary.json (finished = true)
```
Guards kept from `train.py`: `--cores` is clamped to `parallelism_level()` (section 3, rule 8; the old cap of 8
had no effect on the reference machine), NaN/inf in the loss raises QM-E643 and writes the last good checkpoint
before exiting.

Warm start and starting difficulty (the old trainer's behaviour, now stated; the r1 text said "1, or the final
difficulty of the warm-start run" for all cases):
* `--init auto`: valid only for a run directory named `X__cont`; the warm-start run is the sibling `X`. The new run
  starts at the sibling's `summary.final_difficulty` with `start_iteration = sibling.start_iteration +
  sibling.iterations`. A missing or unreadable sibling `summary.json` is QM-E647 (the old trainer silently fell back
  to difficulty 1).
* `--init DIR`: DIR may be a run dir (its `best/` is used) or any model container (a shipped f16 model works). The run
  starts at difficulty 1 unless `--start-diff D` is given (the old trainer inherited a difficulty only for `__cont`
  siblings); `start_iteration` is DIR's `run.json` value plus its iterations when DIR is a run dir, else 0 unless
  `--start-iteration N`. `train_all` chains campaign stages with an explicit `--init <prev>/best --start-diff
  <prev final_difficulty>`.
* neither: fresh run, difficulty 1, `start_iteration = 0`.
An incompatible architecture raises QM-E647. The Adam state is fresh in a continuation (as in the old per-process
behaviour).

Crash safety and resume (a lane runs for hours; the old pipeline had the same weakness but r1 raised the stakes by
putting everything in one process). Checkpoints (`best/`, `last/`) are written by the atomic directory writer of 9.1,
so a reader never sees a half-written checkpoint. Every `--checkpoint-every N` iterations (default 50) and at the end
the driver writes `last/` together with `state.json` (`it`, `start_iteration`, difficulty, best tuple, `elapsed_s`,
seed, config hash) and `adam.bin` (the Adam `m` and `v` as raw f32; `t` is in `state.json`), also atomically.
`train_synth --resume RUN` restores all of it and continues. Because every random stream derives from `(seed, global
iteration, episode)` and nothing else is stateful, a resumed `--iters` run is bit-identical to the uninterrupted run
(tested in `test_rl_smoke`, T-RESUME); `--budget-s` is cumulative through `elapsed_s`. `summary.json` carries
`"finished": true` only after the final checkpoint; the campaign (10.3) uses it to skip finished entries and resume
unfinished ones.

---------------------------------------------------------------------------------------------------------

## 8. Router training

### 8.1 Behaviour cloning (`bc.mojo`, `train_route.mojo`), spec: torch.md 3.2-3.3, 5.1; repo.md 5.1

Data: `teacher_data.mojo` reads the 328-byte records (obs i16[128] @0, A @256, act @258, job @260, tb @262,
Q f32[16] @264; little-endian), concatenates the files in the given order, keeps exactly the rows
with `spread > 1e-6`, `spread = nanmax(Q[:A]) - nanmin(Q[:A])` (written as keep-if-greater: a NaN spread, i.e. `A = 0`,
is dropped, and so is `A = 1` with spread 0; a literal "drop if `<=`" test would keep the NaN rows and produce NaN
targets), clips `Q = min(Q, 1e6)` before both the loss and the validation `qmin` / `agree` comparison, shuffles with `randperm(seed_for(seed, TAG_BC_SPLIT))`, `nval =
max(1000, N // 20)`. Net: `emb = N(0,1) * (1/sqrt(128))` [128,H], `b0 = 0`, `out = Linear(H,16)` torch init,
`value = Linear(H,1)` torch init (never trained in BC, **frozen** in Adam, exported as is). Per epoch `e` (30, lr
`cosine_epoch_lr(2e-3, e, epochs)`): fresh `randperm(N)`; batches of 512 (last short batch kept); per batch the
slot permutation (Fisher-Yates per sample over 16 slots, drawn up-front on the main thread from
`seed_for(seed, TAG_BC_SLOT, epoch, batch)`), `target = softmax(-(Q - min Q)/tau)` over valid slots only,
`loss = -mean_b sum_k target * log_softmax(masked logits)`, gradient `(p - target)/B` on valid slots, backward
`dW1 = dz^T h; db1; dh = dz W1; dh *= (pre > 0); db0; demb[i,:] += sign_i * dh` (scatter, signs from
`sign(obs)` in {-1,0,+1}), Adam step. Chunks of 128 rows over `VW = 4` virtual workers. After each epoch:
validation without augmentation (`pick = argmax(masked logits)`, first max; `agree = mean(Q[pick] <= qmin + 1e-6f)`
with the add in f32; `regret = mean(Q[pick] - qmin)`), log `{epoch, loss = sum/N, val_agree, val_regret, t}`.
`--budget-s` (default 1800) breaks after finishing the epoch; the cosine schedule is not rescaled (faithful).
`--init DIR` accepts a shipped f16 router (decoded) or a Mojo run dir.

Output: a run dir that is a valid `qumo.rlroute.v0` container in f32 (`weights.bin` + `manifest.json` without
test vectors) plus `train_log.json` (schema of repo.md 4.3 minus the npz). There is no `.npz`.

### 8.2 DAgger round

Unchanged: `route_tool gen` already does teacher rollouts and the beta-mixture with the previous model; the
Makefile recipe sequence stays (section 10.2), only the Python steps are replaced.

### 8.3 Optional router PPO (`route_ppo.mojo`), spec: torch.md 3.4, 5.2; ppo.md 12

Needs the split `Episode` API (`pre_decision(limit)`, `needs_policy()`, `act(e)`, `overhead()`, `exec2q`) that the
Mojo environment currently only offers fused as `step`; C12 adds it to `ai_route.mojo` additively (behaviour
unchanged, `tests/test_ai_route.mojo` must still pass bit-exactly). Per-decision reward `-(overhead delta incl.
fallback swaps until the next decision) + 0.01 * (2q gates executed)`; float64 forward, `rng.choice` as inverse CDF
on the cumulative sum; GAE gamma 0.99, lam 0.95 float64 (router form: `delta + gamma*lam*adv_next`, `ret = adv +
V`); advantages standardised with population std + 1e-8; 4 epochs, shuffled minibatches of 1024, clip 0.1, vf 0.5,
ent 0.005, Adam lr 2e-4, `clip_grad_norm(0.5)`; curriculum `scale = min(1.5, 0.4 + 0.02 * it)`, jobs
`build_jobs(700000 + 1000*seed + it, episodes, max_n=24, scale)`. Rollouts: `parallelize` over episodes with the
per-episode seed rule of section 3 (replaces the fork Pool).

`torch.md` 5.2 is normative for the details this summary compresses, and the `gae`, loss and update cases of
`ppo_route_golden.json` must pass (GAE at 1e-6): GAE is float64 over the concatenated trajectories with `D[i]` marking
the last step of a trajectory, `nv = 0 if D[i] else V[i+1]`, `delta = R[i] + gamma*nv - V[i]`, `last = delta +
gamma*lam*(0 if D[i] else last)`, `adv[i] = last` (bootstrap and carry-over reset at every episode end); `ret = adv +
V` uses the un-normalised advantage while the loss uses the standardised one (population std + 1e-8); the sampled
action is an inverse CDF on the cumulative sum normalised by its last element (`searchsorted(side='right')`); the
recorded old log-prob is `log(p[a] + 1e-12)` evaluated in float64 and then cast to f32; tensors are cast to f32 before
the epochs.

---------------------------------------------------------------------------------------------------------

## 9. Export, container, INDEX, models-verify

### 9.1 Container (shared by `qumo.rlsynth.v0` and `qumo.rlroute.v0`), byte-exact

Directory `<root>/<id>/` with `manifest.json` and `weights.bin`. `weights.bin` = concatenation of tensors in
manifest order; zero padding to a multiple of **64 bytes before each tensor and after the last**; `offset` is the
padded start; data little-endian row-major; `f16` for every tensor not named `*.bias` unless `dtype = f32` or the
name is in `f32_names`; `*.bias` always f32; the writer asserts finite after f16 rounding (QM-E645). Manifest
fields: repo.md 4.1 (unchanged). `fnv1a64` (16 lowercase hex) and `sha256` (64 hex) over the *entire* file
including pad bytes. After writing, the writer reads the directory back with `load_model_file` and compares decoded
tensors with what it encoded (QM-E644 on mismatch), exactly like the Python writer's self-check.

**Atomic writes (every writer, run checkpoints included).** A writer never touches the final directory in place. It
writes `<dir>.tmp` (removing a stale one first), performs the read-back check on `<dir>.tmp`, and only then swaps:
`<dir>` is renamed to `<dir>.old` (if it exists), `<dir>.tmp` is renamed to `<dir>`, and `<dir>.old` is removed.
`std.os` has no `rename` in Mojo 1.2 (probe: "package 'os' does not contain 'rename'"), so C2 provides
`atomic_replace_dir` through the libc symbol with `external_call["rename", Int32]` on C strings (probed: rc 0, file
moved); it is the only libc call of the stack and is POSIX, not vendor-specific. `recover_dir`, run at startup by every
tool that reads or writes a model or run directory, renames `<dir>.old` back if `<dir>` is missing and deletes a stale
`<dir>.tmp`. A kill or a power loss therefore leaves either the complete old or the complete new directory. C2 probes
whether `std.os` offers directory removal and otherwise uses libc `unlink` / `rmdir` through the same mechanism.

Tensor names: synth `embeddings.weight` [OBS,E] (transposed), `embeddings.bias`, `common.<k>.*`, `action.<k>.*`,
`value.<k>.*` with **consecutive** `k`; router `embeddings.weight` [128,H], `embeddings.bias`, `action.0.*`,
`value.0.*`. Manifest additions for Mojo-trained models (the loader ignores unknown keys): `training.engine =
"qumo.rl (Mojo)"`, `training.algo = "PPO"` / `"behaviour cloning + DAgger"`, `training.chunk_rows`, `training.vw`,
`training.seed`; `producer` string says trained by the native QUMO stack.

### 9.2 Synthesis export (`export_synth.mojo`), spec: repo.md 2.6

Input: a run dir (`best/` or `last/`, f32 container + `run.json`). Steps: read `kind`, `num_qubits`, `actions`
(gateset), `edges = sorted undirected pairs of the 2q actions`, `max_rotations` (default 5 for Pauli); symmetries
from `symmetry.twists` / `pauli_twists`; layout strings per kind (repo.md 2.6); write with `dtype = f16`; compute
`test_vectors` (6) and `rollouts` (2) from the **stored f16-rounded weights** by loading the written directory with
`load_model_file` + `MLPPolicy` and `PolicyEnv` (`random_start`: random walk of `3 + 4i` actions from identity; Pauli:
random rotation count `1..max_rot+2`, weights as masks, random Clifford walk, `pauli_start`, clean; seeds from
`TAG_EXPORT_VEC`), then rewrite the manifest tail with the vectors and update `INDEX.json`. `bytes` in the INDEX row
is computed after the final manifest is written. The Python-side cross-check against qiskit-gym `twists()` is
replaced by the shipped-manifest golden test (plain kinds) and, for Pauli models, by the fixture
`envs/pauli_twists.json` plus the runtime assertion `check_pauli_ensemble` (4.6).

**Independent checks before anything is written.** The test vectors would otherwise be tautological, being computed by
the same Mojo `MLPPolicy` that the replay tests later check them against. (1) On the six vector observations,
`BasicNet.forward_one` (training layout, from the f32 run checkpoint rounded to f16) and `MLPPolicy` (inference layout,
from the written directory) must agree to 1e-5 in logits and value, for every twist and for the ensemble average, and
both must agree with a naive independent Float64 forward over the decoded tensors (plain loops, no shared kernel);
(2) for Pauli models `check_pauli_ensemble` runs on the exported symmetries with the exported weights. Any failure
aborts with QM-E644 before `INDEX.json` is touched. The shipped manifests' independent (Python/numpy-made) vectors are
additionally replayed through `BasicNet.from_model` in `test_rl_nn`, which binds the training-side layout (`Wt [OBS,E]`,
consecutive naming, act_perm direction) to the inference side on real models.

### 9.3 Router export (`export_route.mojo`), spec: repo.md 2.11, torch.md 7

Writes `qumo.rlroute.v0` (all fields of repo.md 4.1), merges `train_log.json` (minus `log`) and an optional
`training_meta.json` into `training`, creates 6 `obs_logits` test vectors (float64 forward of the stored f16
weights; observations from a seeded `Rng`: 3 random integer vectors in [-3,3] and 3 Bernoulli(0.5) vectors) and
`--episodes N` (default 2) parity episodes with `jobs.build_jobs(424242, N, topologies = [grid 3x3, heavy_hex
19], max_n = 9)` run through `Episode` in policy mode. **INDEX default handling is fixed** (defect 7.4): the writer
takes an explicit `default_id`; `--default` sets it to the exported id, otherwise the previous default is
*preserved* (read from the existing INDEX) instead of being lost.

### 9.4 `models_verify.mojo` (replaces the inline Python of `make models-verify`)

Same behaviour and output line `models-verify: N INDEX.json, M files ok, K problems, S MB total`, same five INDEX
layouts, same `FAIL ...` lines, exit code 1 on any problem, `QUMO_MODEL_DIR` honoured. The five layouts are
implemented over `JsonValue` with the same recursion as the Python `pairs()`. SHA-256 streams the file in 1 MiB
blocks. "Files not listed in any INDEX.json" check kept.

### 9.5 INDEX.json writer

Rows sorted by directory name: `{id, kind, num_qubits, edges, wl_hash, dir, sha256, bytes}` (+ `default` for the
router registry). Output format is Python's `json.dump(indent=1)` (every array element on its own line, as in the
committed files); `qumo.ai.json::pretty_json` is checked first and a dedicated `dump_indent1` is added to
`container.mojo` only if it differs. Gate: regenerating the INDEX of the shipped `models/ai_transpiler/synth` from
its manifests gives a semantically identical JSON (all fields equal); byte equality is a target, not a gate.

---------------------------------------------------------------------------------------------------------

## 10. Drivers, CLI, Makefile, campaign

### 10.1 Binaries (all `mojo build -I . -o build/<name> training/ai_transpiler/...`)

| binary | usage |
|---|---|
| `train_synth` | `--kind perm\|lf\|clifford\|pauli --topology SPEC --out RUN (--budget-s S \| --iters N) [--cores C=4 (clamped to parallelism_level())] [--seed 1] [--emb 512] [--common 256[,..]] [--episodes 1024] [--diff-max 256] [--init DIR\|auto] [--start-diff D] [--start-iteration N] [--resume] [--checkpoint-every 50] [--eval-episodes 200] [--sampled-eval-every 5] [--fidelity twisterl\|fixed] [--quiet]`; exactly one of `--budget-s` / `--iters` |
| `export_synth` | `RUN [--which best\|last] --id ID --out ROOT [--dtype f16\|f32] [--eval-json F] [--default]` |
| `eval_synth` | `RUN --id ID --root ROOT --work DIR [--instances 100] [--keep-rule default]`: export to staging, run the block evaluation in-process, write `eval_blocks.json` / `eval_final.json`, apply `keep = win_or_tie >= 0.6 and cost_ratio < 1.0 and fails == 0` (policy_beam8 vs greedy), re-export into ROOT with the eval embedded when kept |
| `train_all` | `--campaign training/ai_transpiler/synth/campaign.json [--lane K/L] [--only id,id] [--root models/ai_transpiler/synth] [--work build/ai_train/synth] [--budget-cap-s 28800] [--cores 4]`; skips finished entries, resumes unfinished ones, exits non-zero if any enabled entry failed (10.3) |
| `route_tool` | existing `gen`, `eval`, `episode`; new `gen-jobs SEED N MAX_N SCALE OUT [--topologies train\|eval\|name,..]`, `eval-sets OUT_DIR`, `summarize EVAL.jsonl` |
| `train_route` | `--data d0.bin [d1.bin ...] --out RUN --hidden 64 --tau 0.5 --epochs 30 --seed 0 [--lr 2e-3] [--batch 512] [--init DIR] [--budget-s 1800] [--threads N]` |
| `export_route` | `RUN --root ROOT --id ID [--default] [--episodes 2] [--meta F]` |
| `train_route_ppo` | `--init DIR --out RUN --budget-s S [--workers N] [--seed S]` (optional) |
| `models_verify` | no arguments; `QUMO_MODEL_DIR` |
| `make_fixtures` | regenerates `tests/fixtures/ai/policy/**` with random weights from the Mojo `Rng` (the committed fixtures stay frozen; this exists only so the generator is not Python) |

### 10.2 Makefile (owned by C13; replaces lines 66-189 region)

```make
AI_BUILD ?= build/ai_train
AI_CORES ?= 4
AI_LANES ?= 1            # two lanes share the 4 pool threads and run about 0.69x slower each (section 3, rule 8)
AI_SYNTH_BUDGET_S ?= 28800
AI_ROUTE_WORK ?= $(AI_BUILD)/route
AI_ROUTE_ID ?= route_mlp_h64_v1
AI_TOOLS := route_tool train_route export_route train_synth export_synth eval_synth train_all models_verify train_route_ppo

vpath %.mojo training/ai_transpiler/synth training/ai_transpiler/route training/ai_transpiler/common
AI_RL_SRC := $(wildcard qumo/rl/*.mojo) $(wildcard qumo/compiler/*.mojo)

$(AI_BUILD)/bin/%: %.mojo $(AI_RL_SRC)       # one tool per main file; vpath finds it in synth/, route/ or common/
	@mkdir -p $(dir $@)
	$(MOJO) build -I . -o $@ $<

models-verify: $(AI_BUILD)/bin/models_verify          ## sha256 of every model file against its INDEX.json (no Python)
	@$<

train-ai: train-ai-synth train-ai-route               ## retrain every AI-transpiler model (hours; no Python, no Rust, no torch)
	@$(MAKE) --no-print-directory models-verify

train-ai-synth: $(AI_BUILD)/bin/train_all $(AI_BUILD)/bin/train_synth $(AI_BUILD)/bin/export_synth $(AI_BUILD)/bin/eval_synth
	@pids=; for k in $$(seq 1 $(AI_LANES)); do \
	  $(AI_BUILD)/bin/train_all --campaign training/ai_transpiler/synth/campaign.json --lane $$k/$(AI_LANES) \
	      --cores $(AI_CORES) --budget-cap-s $(AI_SYNTH_BUDGET_S) & pids="$$pids $$!"; done; \
	  rc=0; for p in $$pids; do wait $$p || rc=1; done; exit $$rc     # a bare `wait` always returns 0: wait on every PID

train-ai-route: $(addprefix $(AI_BUILD)/bin/,route_tool train_route export_route)
	@mkdir -p $(AI_ROUTE_WORK)/stage
	$(AI_BUILD)/bin/route_tool gen-jobs 31 1000 30 0.5 $(AI_ROUTE_WORK)/jobs0.json
	$(AI_BUILD)/bin/route_tool gen-jobs 32 1000 30 0.5 $(AI_ROUTE_WORK)/jobs1.json
	ROUTE_TOOL_CAP=96 $(AI_BUILD)/bin/route_tool gen $(AI_ROUTE_WORK)/jobs0.json $(AI_ROUTE_WORK)/d0.bin - 1.0 401 heuristic 16
	$(AI_BUILD)/bin/train_route --data $(AI_ROUTE_WORK)/d0.bin --out $(AI_ROUTE_WORK)/run0 --hidden 64 --tau 0.5 --epochs 30 --seed 0
	$(AI_BUILD)/bin/export_route $(AI_ROUTE_WORK)/run0 --root $(AI_ROUTE_WORK)/stage --id round0 --default
	ROUTE_TOOL_CAP=96 $(AI_BUILD)/bin/route_tool gen $(AI_ROUTE_WORK)/jobs1.json $(AI_ROUTE_WORK)/d1.bin $(AI_ROUTE_WORK)/stage/round0 0.3 402 heuristic 16
	$(AI_BUILD)/bin/train_route --data $(AI_ROUTE_WORK)/d0.bin $(AI_ROUTE_WORK)/d1.bin --out $(AI_ROUTE_WORK)/run1 --hidden 64 --tau 0.5 --epochs 30 --seed 0
	$(AI_BUILD)/bin/export_route $(AI_ROUTE_WORK)/run1 --root models/ai_transpiler/route --id $(AI_ROUTE_ID) --default

train-ai-smoke: $(addprefix $(AI_BUILD)/bin/,train_synth export_synth eval_synth route_tool train_route export_route models_verify)
	# ~5 min end-to-end into $(AI_BUILD)/smoke (A5 synth smoke, A6 router smoke, A7 export + verify); never writes models/
	# smoke.sh and tests/test_rl_smoke.mojo are owned by C11c; C13 only wires the target
	training/ai_transpiler/smoke.sh $(AI_BUILD)

train-ai-route-ppo: $(addprefix $(AI_BUILD)/bin/,train_route_ppo export_route)   ## optional router PPO fine-tune (D1); never part of train-ai
	@test -n "$(INIT)" -a -n "$(BUDGET_S)" || { echo "usage: make train-ai-route-ppo INIT=<router dir> BUDGET_S=<seconds>"; exit 2; }
	$(AI_BUILD)/bin/train_route_ppo --init $(INIT) --out $(AI_ROUTE_WORK)/ppo_run --budget-s $(BUDGET_S)

test-rl:         ## fast RL tests only (tests/test_rl_*.mojo except *_slow, smoke, e2e); prints wall time per file, each must stay under 30 s
	@fail=0; for t in $(filter-out %_slow.mojo tests/test_rl_smoke.mojo tests/test_rl_e2e.mojo,$(wildcard tests/test_rl_*.mojo)); do \
	  s=$$(date +%s); out=$$($(MOJO) run -I . $$t 2>&1 | grep -v Crashpad | tail -1); e=$$(( $$(date +%s) - s )); \
	  case "$$out" in *OK*) echo "PASS $$t ($${e}s)";; *) echo "FAIL $$t: $$out"; fail=1;; esac; \
	done; exit $$fail

test-rl-slow:    ## statistical and end-to-end RL tests (*_slow, smoke, e2e) and test_rl_tensor with -D ASSERT=all
	@fail=0; for t in $(wildcard tests/test_rl_*_slow.mojo) tests/test_rl_smoke.mojo tests/test_rl_e2e.mojo; do \
	  out=$$($(MOJO) run -I . $$t 2>&1 | grep -v Crashpad | tail -1); \
	  case "$$out" in *OK*) echo "PASS $$t";; *) echo "FAIL $$t: $$out"; fail=1;; esac; \
	done; \
	out=$$($(MOJO) run -D ASSERT=all -I . tests/test_rl_tensor.mojo 2>&1 | grep -v Crashpad | tail -1); \
	case "$$out" in *OK*) echo "PASS tests/test_rl_tensor.mojo (-D ASSERT=all)";; *) echo "FAIL tests/test_rl_tensor.mojo: $$out"; fail=1;; esac; exit $$fail

check-pure-mojo:         ## fails if the training path invokes or names python/pip/uv/torch/numpy/qiskit/twisterl/rustworkx/safetensors (section 14, A8)
	training/ai_transpiler/check_pure_mojo.sh
```
`TESTS` (line 4 of the Makefile) becomes `$(filter-out %_slow.mojo tests/test_rl_smoke.mojo tests/test_rl_e2e.mojo,
$(wildcard tests/test_*.mojo))` so `make test` stays fast; `train-ai-route-ppo`, `test-rl` and `test-rl-slow` are part of
the C13-owned recipe above (r1 listed neither).
The old `AI_TRAIN_PY`, `AI_ROUTE_BUDGET_S` (it was unused), the inline `MODELS_VERIFY_PY` and the `parity.py` gate
(replaced by `tests/test_ai_route.mojo`) are deleted. The route `--budget-s` default of `train_route` stays 1800
and is surfaced in the recipe as `AI_ROUTE_BC_BUDGET_S` so it no longer silently truncates.

### 10.3 Campaign (`synth/campaign.json`, replaces the missing `train_all.sh`)

Data, one entry per model; defaults from the shipped manifests (4 cores, seed 1, emb 512, common [256], episodes
1024, diff-max 256):

| id | kind | topology | stages (budget_s) |
|---|---|---|---|
| perm_grid3x3 | perm | grid:3x3 | 600 |
| perm_grid3x4 | perm | grid:3x4 | 900 |
| perm_king3x4 | perm | king:3x4 | 1200 |
| perm_hhring10s2 | perm | heavyhex_frag:ring10s2 | 2700 |
| lf_grid2x3 | lf | grid:2x3 | 2700 |
| lf_ring6 | lf | ring:6 | 2400 |
| lf_line6 | lf | line:6 | 2700 |
| lf_hhH7 | lf | heavyhex_frag:H7 | 2700 |
| pauli_line4 | pauli | line:4 | 2100, 2100, 2100 (each stage `--init auto`, total 6300 s as in the manifest) |

Dropped models (clifford_*, lf_line7, pauli_complete5, pauli_line6, perm_line8) have no recorded spec; they are
listed in the campaign file as `"enabled": false` entries with a comment, to be filled if someone decides to
retrain them. Each entry: train -> `eval_synth` -> export if kept. `--lane K/L` takes the entries with index `i % L ==
K-1` (sorted by budget, longest first, for balance). Total budget cap and per-entry budget are honoured.

* **Budgets and lanes.** An entry's stage may give `iters` instead of (or besides) `budget_s`; when `iters` is present
  `train_synth` runs with `--iters`, so the model's training no longer depends on the clock or on the other lane. The
  default file keeps the shipped `budget_s` values for parity with the Python campaign; with `AI_LANES=1` the nine
  enabled entries need 22,200 s of training plus evaluation, inside the 28,800 s cap. With two lanes each lane runs about
  0.69x slower (section 3, rule 8) and a time-boxed entry gets fewer iterations than in the Python campaign.
* **Chained stages** (`pauli_line4`) use `--init <prev>/best --start-diff <prev final_difficulty>` and inherit the
  global iteration offset (section 3, rule 1).
* **Idempotence and failure reporting.** Each entry has `<work>/<id>/status.json` (`state` = `pending | running | done |
  failed`, stage, iteration, error code). `train_all` skips entries that are `done` and whose exported model passes
  `models_verify`, resumes `running` ones with `train_synth --resume` (7.5), appends `entry id, stage, iteration` lines
  to `<work>/lane<k>.log` at every checkpoint, and exits non-zero if any enabled entry failed, errored or missed the keep
  rule (QM-E649). `make train-ai-synth` waits on every lane PID, so a failed lane fails the target and `train-ai` does
  not go on to the router and a passing `models-verify` over stale models.

---------------------------------------------------------------------------------------------------------

## 11. Feature-by-feature mapping

### 11.1 Files (all 17 Python files, the Makefile, the requirements)

| existing | disposition | Mojo replacement |
|---|---|---|
| `common/export.py` | port | `qumo/rl/{container,digest,f16}.mojo` (+ numbering fix) |
| `common/topology.py` | port parse/slug/connected; WL hash reused | `qumo/rl/topology.mojo`, `policy.wl_hash` |
| `common/symmetry.py` | port | `qumo/rl/symmetry.mojo` |
| `common/refenv.py` | retire | `PolicyEnv`, `MLPPolicy`, `greedy_trace`, `search_best` |
| `common/make_fixtures.py` | port | `common/make_fixtures.mojo` (C11a) |
| `synth/train.py` | port | `ppo.mojo`, `synth_env.mojo`, `nn.mojo`, `synth_train.mojo`, `train_synth.mojo` |
| `synth/export_model.py` | port | `export_synth.mojo` |
| `synth/eval.py` | port orchestration, drop Qiskit baseline | `eval_synth.mojo`, `qumo/rl/eval_blocks.mojo` |
| `synth/train_all.sh` (missing) | define | `campaign.json` + `train_all.mojo` |
| `route/route_env.py`, `parity.py` | retire | `ai_route.mojo::Episode`, `tests/test_ai_route.mojo` |
| `route/circuits.py` | port | `qumo/rl/{jobs,targets}.mojo`, `route_tool gen-jobs` |
| `route/eval_sets.py` | port | `route_tool eval-sets` (CRC-32 in `digest.mojo`) |
| `route/train_route.py` | port | `bc.mojo`, `train_route.mojo` |
| `route/train_ppo.py` | port (optional) | `route_ppo.mojo`, `train_route_ppo.mojo` |
| `route/export_npz.py`, `export_route.py` | port, npz dropped | `export_route.mojo` |
| `route/summarize_eval.py` | port | `route_tool summarize` |
| Makefile `models-verify`, `train-ai*` | rewrite | section 10.2 |
| `requirements.txt` | delete | none |

### 11.2 qiskit-gym / twisterl (Rust) features

| feature | Mojo location |
|---|---|
| `from_coupling_map` gateset order, `num_qubits = max qubit + 1` | `gateset.build_gateset` |
| `obs_shape`, `num_actions`, flat sparse observation | `TrainEnv.obs_size/num_actions/observe` |
| `reset` (scramble, depth) for Perm / LF / Clifford | `TrainEnv.reset` |
| Pauli `reset` (rotations, tableau, mirrored label quirk) | `TrainEnv.reset` Pauli branch |
| `step`, metrics tracker, reward, `maybe_random_invert` | `TrainEnv.step`, `Metrics` |
| `masks`, `is_final`, `success`, `reward` | `all_allowed`, `is_final`, `success`, `reward` |
| `twists()` plain / Pauli internal | `symmetry.twists`, `pauli_qubit_perms_and_act_perms` |
| `set_state` (perm / bits / Pauli vector) | `set_state_*` (tests only) |
| `difficulty` get/set | `set_difficulty` |
| `dyn_clone` per episode | `TrainEnv` copy |
| `PPOCollector.single_collect`, Gumbel-max sampling, GAE, merge | `ppo.collect`, `rlrng.gumbel_argmax`, `loss.gae_twisterl` |
| `evaluate` / `solve` (num_searches, deterministic) | `ppo.evaluate` |
| `Policy::forward_with_perm`, random twist per call | `BasicNet.forward_one` + actor stream |
| `BasicPolicy` (nn.Linear stack, init, symmetry gather) | `BasicNet` |
| `PPO.train_step` (Categorical, clamp/min, mse, entropy) | `loss.ppo_chunk_loss` + `ppo_update` |
| `Algorithm.learn_step` curriculum / promotion | `Trainer.learn_step`, `promote_if_ready` |
| `RLSynthesis.save` (`config.json`) | `run.json` |
| rayon pools | `parallelize(body, items, cores)` |

### 11.3 torch / numpy / stdlib calls

| call | Mojo |
|---|---|
| `nn.Linear` init, `manual_seed` | `uniform_sym_f32`, `seed_for(TAG_INIT)` |
| `relu`, `@`, `x @ W.T + b` | `relu_inplace`, `gemm_nt`, `add_bias_rows` |
| `softmax`, `log_softmax`, `masked_fill`, `gather`, `argsort(rand)` | `loss.log_softmax_row`, loops over valid slots, per-sample Fisher-Yates |
| `Categorical.log_prob/entropy` | `ppo_chunk_loss` |
| `clamp`, `min`, `exp`, `mse_loss` | inside `ppo_chunk_loss` with the explicit gradient rule |
| `loss.backward()` | hand-derived `backward_chunk` (section 4.4) |
| `optim.Adam`, `CosineAnnealingLR`, `clip_grad_norm_` | `optim.Adam`, `cosine_epoch_lr`, `clip_grad_norm` |
| `randperm`, `default_rng(seed).permutation`, `Generator.choice(p=)` | `randperm`, `sample_probs` |
| `np.fromfile(structured dtype)`, `np.sign`, `nanmax/nanmin` | `teacher_data.read_teacher_files`, loops |
| `np.savez` / `np.load` | dropped (container f32 run dirs) |
| `astype("<f2")`, `isfinite`, `tobytes`, `frombuffer` | `f16.encode_f16`, `container` |
| `safetensors.save_file/load_file` | `checkpoint` (container) + optional importer |
| `hashlib.sha256`, `zlib.crc32` | `digest.mojo` |
| `random.Random` (shuffle, randrange, choice, random) | `Rng` (`jobs.mojo`) |
| `json.dump` compact / indent=1 | `JsonWriter` / `container.dump_indent1` |
| `os.walk/listdir/getsize`, `subprocess` | `std.os`, direct calls |
| `multiprocessing.Pool(fork)` | `parallelize` |
| `rustworkx.heavy_hex_graph` | `Target.heavy_hex_edges` |
| `qiskit.transpile` baseline | dropped from training (bench keeps it) |

---------------------------------------------------------------------------------------------------------

## 12. Edits to existing files, legacy removal, docs, NOTICE

### 12.1 Additive edits to runtime files (kept minimal; each owned by exactly one component)

| file | edit | owner |
|---|---|---|
| `qumo/compiler/ai_route.mojo` | add `Episode.pre_decision/needs_policy/act` split (behaviour-preserving); update the error hint that names `export_route.py` | C12 |
| `qumo/compiler/policy.mojo` | `_e617` hint: "re-export with `build/export_synth` / `make train-ai`" instead of `export.py` | C13 |
| `qumo/ai/diagnostics.mojo`, `docs/ERRORS.md` | register QM-E640..E649 | C13 |
| `training/ai_transpiler/synth/eval_blocks.mojo` | becomes a thin main over `qumo/rl/eval_blocks.mojo` | C11b |
| `training/ai_transpiler/route/route_tool.mojo` | helpers moved to `qumo/rl/jobs.mojo`; new subcommands | C9 |
| `tests/test_policy.mojo`, `tests/test_ai_route.mojo` | honour `QUMO_TEST_MODEL_ROOT` (default `models/ai_transpiler`) so a Mojo-trained model in a staging root can be run through the same replay tests | C13 |

### 12.2 Legacy Python

Waves 1-4 keep the Python files untouched (they are the oracle while the port is validated; everything only they can produce is dumped into fixtures by C0 in wave 0, so the wave-5 deletion loses no oracle). In wave 5 (C13), after
acceptance A1-A8 pass, **all 17 Python files and `requirements.txt` are deleted** (`git rm`); the lead tags the
last commit that contains them (`legacy-python-training`) and `README.md` names the tag. Nothing is kept as
"legacy" inside the tree. The golden data derived from them lives in `tests/fixtures/rl/`. `bench/neutral/targets.py`,
`bench/ai_vs_ibm/*` and the other Python benchmarks are not training and stay.

### 12.3 `.gitignore`

Keep the run / checkpoint patterns (`training/ai_transpiler/**/runs/`, `build/ai_train/`); drop `*.safetensors`,
`.venv` patterns that only existed for the training venv (harmless to keep one release).

### 12.4 Documentation (owner C13)

* `training/ai_transpiler/README.md`: rewritten (no Setup / venv; `make train-ai`, binaries, run-dir layout,
  determinism statement replacing "Training is not bit-reproducible", campaign file, how to add a model, resume after a crash, the `--cores` clamp and the 4-thread limit of the reference machine, and the plain statement that toolchain installation (`make setup`, the pinned Mojo launcher is a Python script) still uses uv and pip).
* `training/ai_transpiler/route/README.md`: file table updated (no `.py`), "Reproduce" block becomes the make recipe.
* `docs/AI_TRANSPILER.md`: the model-format sentence ("twisterl `BasicPolicy`" -> "a `BasicPolicy`-shaped MLP"),
  the "Training budgets and retraining" section (requirements sentence), the licensing bullet "qiskit-gym / twisterl
  / torch are needed only to retrain" (no longer true), verification section (new tests).
* `README.md` (root) mentions of training requirements; `docs/ARCHITECTURE.md` module map gains `qumo/rl`; `llms.txt` if
  it lists training; `bench/ai_vs_ibm/README.md` stays (IBM baseline venv is a benchmark).
* `docs/ERRORS.md` new codes. `docs/MOJO_GUIDE.md` unchanged.

### 12.5 NOTICE and licensing

Append to the "AI transpiler" block of `NOTICE` (C13):

> Training stack (`qumo/rl`, `training/ai_transpiler`): a native Mojo re-implementation, from written
> specifications, of the training *algorithms* of qiskit-gym 0.4.1 (AI4quantum, Apache-2.0: environments,
> metrics-based rewards, curriculum) and twisterl 0.5.1 (IBM, Apache-2.0: PPO with generalised advantage
> estimation, `BasicPolicy` architecture and symmetry augmentation). No source code of qiskit-gym, twisterl, torch,
> numpy or safetensors is copied or transliterated. Numerical reference data under `tests/fixtures/rl` are outputs
> of those packages obtained offline; they carry no code. None of these packages is required to train, run or test
> QUMO.

and change the existing sentence "qiskit-gym and twisterl are needed only at training time" to say they are no
longer needed. The shipped model weights remain QUMO's (Apache-2.0); models retrained natively are likewise QUMO's.
Process rules for agents (D11): no opening of the downloaded packages' source, no pasting of the research scripts
(`envs_ref.py`, `ref_ppo.py`, `ref_np.py` are *not* committed because they are line-level transcriptions); fixtures
are data only.

---------------------------------------------------------------------------------------------------------

## 13. Test plan

Every test file is `tests/test_rl_<x>.mojo`, `def main() raises:`, prints `test_rl_<x> OK` as its last line. `make test`
runs each file with `mojo run`, so every file pays its own compile time (cold, measured by the reviewer on the
reference machine: `tests/test_policy.mojo` 5.8 s, `tests/test_ai_route.mojo` 10.4 s, building `route_tool` 12.8 s; a
warm compile cache is much faster, 1.5 s for `test_policy` when re-measured). With 16 fast files at 6-13 s cold the RL
tests add roughly 100-200 s to `make test`; the r1 claim "<= 120 s in total" had no mechanism and is withdrawn. The
enforceable rules are: (1) **each fast file stays under 30 s wall including compile**; (2) files whose name ends in
`_slow` (the 40,000-reset chi-square, the rollout statistics), `test_rl_smoke.mojo` and `test_rl_e2e.mojo` are excluded
from `make test` and run by `make test-rl-slow`; (3) `make test-rl` runs only the fast `tests/test_rl_*.mojo` and prints
the wall time per file so a regression is visible; (4) `make test-rl-slow` also runs `test_rl_tensor` once with
`-D ASSERT=all`.

### 13.1 Fixtures (`tests/fixtures/rl/`, <= 2 MB total, JSON, produced once by C0)

C0 trims the research golden data with a one-off scratch script (not committed): floats re-printed compactly with
9 significant digits (f32 round trip), compact separators, big case lists cut to the smallest representative
subset. `PROVENANCE.md` records the source file, its sha256, the package versions and the size before / after.

| fixture (small form) | from | size target | used by |
|---|---|---|---|
| `envs/gateset_twists.json` | `gateset_twists.json` (44 configs) | 52 KB | gateset + symmetry + Pauli perms tests |
| `envs/steps_{permutation,linear_function,clifford,pauli_network}.json` | `steps_*.json` | <= 60 KB each (25 starts per config) | exact step / observe / reward / is_final |
| `envs/steps_custom_args.json` | same | <= 40 KB | non-default weights, Pauli args |
| `envs/inverts.json` | same | 16 KB | add_inverts outcome set |
| `envs/pauli_perm_cases.json` | same | <= 60 KB | hidden symmetry recovery |
| `envs/reset_exact.json` | same (828 KB) | <= 200 KB (d <= 3 scramble cases, Pauli tableau d=1..4, rotation lists d=16 and 32 on two topologies) | chi-square on reset |
| `envs/reset_empirical.json`, `envs/defaults.json` | same | small | solved-at-reset rates, constants |
| `envs/get_state.json` | same | optional (not committed unless a circuit converter is added) | - |
| `ppo/policy_forward.json`, `ppo/gae.json`, `ppo/ppo_update.json` | `ppo_golden/` | 20 + 8 + 44 KB | forward (twist 0/1/none), ensemble, GAE, 10-epoch update |
| `ppo/tiny_policy.safetensors`, `ppo/safetensors_header.json` | same | 2 KB | safetensors importer |
| `torch/adam_cosine_golden.json`, `torch/f16_golden.json`, `torch/misc_golden.json` | `torch_golden/` | 4 + 12 + 4 KB | Adam trace, cosine lrs, exhaustive-ish f16 |
| `torch/bc_golden.json` | same (260 KB) | <= 120 KB after re-printing | BC steps incl. frozen value head |
| `torch/ppo_route_golden.json` | same (324 KB) | <= 140 KB | route PPO steps, clip rule, grad clip, GAE |
| `torch/synth_ppo_golden.json` | same | 52 KB | two BasicNet architectures with symmetry, 3 steps |
| `topologies.json` | generated once from `bench/neutral/targets.py` and `circuits.py` | <= 100 KB | 25 train + 10 eval topologies, ordered edge lists |
| `envs/pauli_twists.json` | `pauli_twists` of the Python `symmetry.py`, dumped by C0 before deletion, for `ring:5`, `ring:4`, `star:4`, `grid:2x3`, `heavyhex_frag:H7`, `heavyhex_frag:T5` | <= 40 KB | Pauli export ensemble, direction (4.6) |
| `envs/reward_fma_cases.json` | brute force over (achieved, penalty, count) with numpy f32 unfused arithmetic: the 32 combinations that differ when fused, plus (0, 1e-4, 0.01, 5) | 2 KB | FMA barrier (3, rule 9) |
| `ppo/rollout_stats.json` | offline, real qiskit-gym env + real twisterl collector/evaluate on fixed explicit f32 weights; `perm line:4`, `lf ring:4`, `clifford line:3`, `pauli line:3`, 2-3 difficulties each | <= 60 KB | rollout wiring (`test_rl_ppo_slow`) |
| `route/jobs_stats.json` | Python `build_jobs`, 2000 jobs at scale 0.5 and at 1.0: histograms of n, op count, 2q fraction and depth per family, topology mix | <= 40 KB | job-distribution check (`test_rl_jobs`) |
| `ppo/py_curve.json` | 400 s Python runs of `perm_grid3x3` and `lf_grid2x3`: iteration of every promotion | 4 KB | A12 learning-speed reference |
| `route/py_bc_reference.json` | Python BC on one fixed `d0.bin` (>= 300 jobs built from Python-built jobs by the existing `route_tool gen`; its sha256 recorded, the 16 MB file kept in the lead's custody, not committed), 3 seeds, 30 epochs | 4 KB | BC comparison (A6b) |

Existing independent oracles that stay untouched: `models/ai_transpiler/**` (10 models with Python-made embedded
vectors), `tests/fixtures/ai/policy/**`.

**Rollout-statistics fixture (`ppo/rollout_stats.json`, part of A4).** The unit goldens cover forward, loss, GAE on a
toy episode, Adam and the env transitions separately, but nothing external pinned the assembled `collect()` /
`evaluate()` pipeline on a real env, and a learning smoke test passes with an off-by-one reward alignment, a bootstrap
on the terminal row, a wrong twist reused in the update, a wrong terminal-row mask or wrong `evaluate` reward
accounting. C0 therefore generates it offline, before the Python code is deleted, with the real qiskit-gym env and the
real twisterl collector/evaluate (the scratch venv already drives them) on **fixed explicit f32 weights** (as
`policy_forward.json`) for four small configs: `perm line:4`, `lf ring:4`, `clifford line:3`, `pauli line:3`, each at 2-3
fixed difficulties. For every (config, difficulty) it records: (a) `evaluate()` success and mean reward, for the
deterministic mode and for the sampled mode with 10 searches, over at least 20,000 episodes each; (b) collector
aggregates over at least 5,000 episodes: the episode-length histogram, mean and std of `rets`, `advs`, `v` and
`rewards`, the fraction of rows whose logits are all masked, and the mean of `old_lp`; plus the standard deviations and
sample sizes needed to form standard errors. `test_rl_ppo_slow` loads the same weights (as an f32 container /
`ParamStore`), runs the Mojo collector and evaluate with fixed seeds, and requires `|z| <= 4` for every mean (two-sample,
`z = (m_mojo - m_ref) / sqrt(s_ref^2/n_ref + s_mojo^2/n_mojo)`), a homogeneity chi-square at z <= 4 for the length
histograms (bins pooled to an expected count >= 5), and a binomial 4-sigma bound on the all-masked fraction and the
success rates. The generator script is not committed (D11); `PROVENANCE.md` records versions and sha256.

### 13.2 Test files and what they pin

| test (owner) | cases | tolerance |
|---|---|---|
| `test_rl_digest` (C1; includes f16) | FIPS 180-4 vectors (empty, "abc", 448-bit, 1,000,000 x "a"), a 3.3 MB file hashed in 1 MiB chunks equals the INDEX sha256 of every shipped model; CRC-32 check values (`"123456789"` -> 0xCBF43926) and `zlib` values of three file names; all 65536 f16 patterns: `encode(decode(h)) == h` (NaN payload ignored); the 326-value `f16_golden.json`; overflow raises | exact |
| `test_rl_container` (C2) | re-encode each of the 10 shipped models from decoded tensors -> `weights.bin` byte-identical, fnv/sha identical; INDEX regeneration semantically equal; `models_verify` on `models/ai_transpiler` prints 0 problems; corrupt a byte -> FAIL; consecutive layer numbering for 2 common layers loads in `MLPPolicy`; atomic swap: a simulated stop between the rename steps leaves a complete old or new directory and `recover_dir` restores `<dir>.old` | exact |
| `test_rl_topology` (C3) | the 9 manifests: `twists` from `training.topology_spec` equals stored `symmetries` (order included); Pauli export ensemble for `pauli_line4` **and** `envs/pauli_twists.json` element-wise for `ring:5`, `ring:4`, `star:4`, `grid:2x3`, `heavyhex_frag:H7`, `heavyhex_frag:T5` (non-involutive automorphisms included, so a wrong direction fails); automorphism counts (line:n 2, ring:5 10, star:5 24, grid:2x3 4, H7 8); WL hash equals INDEX; the 35 `topologies.json` edge lists | exact |
| `test_rl_tensor` (C4) | each GEMM variant vs `gemm_ref` (naive triple loop in `testutil`) on random shapes incl. non-multiples of the tile, accumulate flag both ways, 1-row and 1-col cases; **every buffer is allocated with 64-float sentinel guard regions before and after and asserted unchanged after each call** (all three variants, odd shapes, `acc` both ways, the embedding scatter and CSR index paths); `make test-rl-slow` also runs this file once with `-D ASSERT=all` | rel 1e-5; guards exact |
| `test_rl_optim` (C4; also covers `rlrng`) | Adam vs `adam_trace` (incl. the zero-gradient step with momentum), frozen slice stays unchanged, (optional) sparse-row skip equals dense bit for bit, cosine lrs vs `lr_per_epoch_torch` (rel 1e-12), clip coefficient vs `ppo_route_golden` (norm rel 1e-4); `rlrng`: `seed_for` distinct over a (tag, a, b, c) grid, `randperm` is a permutation and uniform on 3 elements, `gumbel_argmax` returns index 0 on an all-`-1e10` row for 1000 generator states, ties return the first index, NaN never wins, logits of 100: argmax picks the true maximum and `sample_softmax` gives a valid draw without NaN, `mulf` returns the unfused product | as listed |
| `test_rl_nn` (C5) | `policy_forward.json` (15 cases: raw logits, value, masked logits, 5 ensemble cases) 1e-5; `synth_ppo_golden.json` both architectures: forward, loss, gradients (atol 3e-6, rtol 2e-4), weights after 3 steps (atol 3e-5); finite-difference sanity (eps 1e-2) on a random tiny net; param counts of the shipped architectures; **the 9 shipped synth manifests: `BasicNet.from_model(model).forward_one` on each manifest `test_vectors` observation (no twist, every twist `p`, and the ensemble average) equals the stored `logits` / `logits_ensemble` and value at 1e-5** (independent Python/numpy-made vectors that bind the training layout to the inference layout) | as listed |
| `test_rl_loss` (C5) | `gae.json` (4 settings) 1e-6; `ppo_update.json` batch: `old_lp` incl. the all-masked row = 0.0; per-epoch loss dict for 10 epochs (2e-5), gradients and weights at epochs 0, 1, 9 (3e-6 / 3e-5), `normalize_advantage` variant | as listed |
| `test_rl_env` (C6) | gateset of the 44 configs; `set_state` + action sequences: obs / is_final exact, reward exact f32 for the 4 kinds, custom args file, Pauli hidden symmetry (`pauli_perm_cases`: observations are members of `obs_by_perm`, recovered `p`, mapped actions, next obs); **`reward_fma_cases.json` bit-exact (the `mulf` barrier)**; `inverts.json` outcome set and P(invert) in [0.45, 0.55] over 4000 trials; horizon `min(2d,128)`; exactly-one-observe protocol; gates outside {H, S, CX, SWAP} raise QM-E642; `step(a >= A)` clamps and sets `error_code()` | as listed |
| `test_rl_env_slow` (C6) | **reset distributions**: draw 40,000 resets per case with fixed seeds, chi-square z <= 4.0 against `reset_exact.json` (scramble, Pauli tableau, Pauli rotation lists); empirical rotation-count histogram and solved-at-reset rates within 3 sigma of `reset_empirical.json` | z <= 4.0 |
| `test_rl_ppo` (C7) | **T-DET**: same seed, `cores` 1, 2 and 4 (and a requested 8, clamped to `parallelism_level()`), 3 iterations on `perm:line:3` -> identical `weights` bytes and identical log lines; rerun identical; **stage seeds**: stage 2 iteration 0 of an `--init` chain uses different stream seeds and collects different start states than stage 1 iteration 0; thread-count independence of `ppo_update` on the `ppo_update.json` batch replicated to 5,000 rows; GAE on a collected episode equals the closed form; **collector self-consistency**: for every collected row the recorded logits equal `forward_one(obs, twist)` bit for bit where allowed, the update's forward with the row's twist reproduces them to 1e-5 (a wrong twist reused in the update fails), the terminal row has no bootstrap and its mask is all `-1e10` iff solved; promotion rule (170/200 promotes, 169 does not); eval ordering; NaN guard; status slots: an invalid action raises QM-E642 deterministically after the parallel region | exact for T-DET |
| `test_rl_ppo_slow` (C7) | `ppo/rollout_stats.json` (13.1): collector aggregates (>= 5,000 episodes) and `evaluate` success / reward (>= 20,000 episodes, deterministic and sampled x10) on the 4 configs | 4 sigma |
| `test_rl_router` (C8) | `bc_golden.json` (3 steps, explicit slot perm, value head frozen and equal to init), planted-optimum BC (synthetic teacher file, agreement > 0.9, regret < 0.1), teacher IO round trip of a 328-byte record with saturating i16, split / filter rule: `keep = spread > 1e-6` drops rows with `A = 0` (NaN spread) and `A = 1`, `Q` is clipped at 1e6; BC reference numbers of A6(b) when `QUMO_BC_REF_D0` is set | as listed |
| `test_rl_route_ppo` (C12) | `ppo_route_golden.json` (3 steps incl. clipping every step, the `gae` case at 1e-6, clip rule, grad clip), 2-iteration smoke | as listed |
| `test_rl_jobs` (C9) | determinism of `build_jobs`, op kinds, size bounds, connectivity of every topology, `topologies.json` match, eval-set rules (seeds >= 910000, CRC-32 placement), JSON loads in `route_tool` helpers; **distribution drift**: family mix, n, op count, 2q fraction, depth and topology mix of 2000 Mojo jobs per scale against `route/jobs_stats.json` (chi-square for categorical, two-sample KS for continuous, 4 sigma) | exact / 4 sigma |
| `test_rl_export` (C11a, synthesis) | export a tiny trained synth run -> directory loads, `load_model_file` checksum ok, embedded vectors replay at 1e-5 through `MLPPolicy`, rollouts replay exactly; the pre-write independent checks of 9.2 (BasicNet vs MLPPolicy vs Float64) abort on a deliberately transposed tensor; `check_pauli_ensemble` on `ring:5` and the other fixture topologies | as listed |
| `test_rl_export_route` (C10, router) | export a tiny BC run -> loads through `load_route_model`, embedded vectors replay at 1e-5 through `RouteModel`, parity episodes replay exactly, INDEX default preserved when exporting without `--default` | as listed |
| `test_rl_fixtures` (C0) | every fixture parses with `qumo.ai.json`; sizes within budget; a few invariants per file | exact |
| `test_rl_smoke` (C11c, slow) | train `perm:line:3` for 12 iterations (<= 15 s), difficulty >= 3 reached, export, load, greedy solves all 6 permutations; **T-RESUME**: 6 iterations equal 3 iterations, stop, `--resume`, 3 more (identical `weights.bin` sha256) | exact |
| `test_rl_e2e` (C13, slow) | the staged-root replay of A7 and the determinism rerun of A10 | exact |

### 13.3 Golden philosophy

* Deterministic quantities (set_state / step / observe / reward / is_final / forward / loss / grad / Adam / GAE /
  f16 / sha / container bytes / symmetry tables) are tested **exactly or to the stated tolerance** against data
  produced by the real packages.
* Random quantities (reset, inversions, rollouts) are tested **distributionally** with fixed seeds (so the tests are
  deterministic): exact rational tables from `reset_exact.json`, empirical statistics otherwise.
* End-to-end equivalence is judged in two layers. The **assembled rollout pipeline** (collector wiring: reward
  alignment, bootstrap, terminal mask, twist; `evaluate` reward accounting) is pinned by numbers: the
  rollout-statistics fixture (13.1), generated offline with the real env and the real twisterl collector/evaluate,
  tested at 4 sigma. **Learning quality** on the same data (A6, A12) is judged by quality, not by weights, because the
  RNG streams cannot match; r1 judged everything by quality, which would have passed with an off-by-one reward
  alignment.
* Nothing in the Mojo tests starts Python. The fixtures were produced offline once; regenerating them needs the old
  packages and the git tag of section 12.2.

---------------------------------------------------------------------------------------------------------

## 14. Acceptance criteria

All must hold on the reference machine (Apple silicon, 4+ cores) and on one Linux x86-64 host with no Python on
`PATH` (portability of the kernels; performance gates are measured on the reference machine only).

* **A1 parity tests**: `make test`, `make test-rl` and `make test-rl-slow` pass, including every `test_rl_*`; the
  golden comparisons of section 13.2 meet their tolerances.
* **A2 container**: all 10 shipped models re-encode byte-identically; `make models-verify` (Mojo) prints
  `0 problems`; the old Python verifier output on the same tree is identical (one-off comparison before deletion).
* **A3 environments**: all exact checks pass, including the discriminating reward cases of `reward_fma_cases.json` (bit-exact reward needs the `mulf` barrier, section 3 rule 9) and the QM-E642 rejection of gates outside {H, S, CX, SWAP}; reset chi-square z <= 4.0 on all cases (fixed seeds).
* **A4 PPO update and rollout pipeline**: 10-epoch golden within tolerance for both the toy batch and the two
  architectures; T-DET passes (cores 1, 2 and 4 identical); the collector self-consistency test passes; **the
  rollout-statistics fixture passes at 4 sigma** (collector aggregates over >= 5,000 episodes and `evaluate` over
  >= 20,000 episodes on the four configs of 13.1).
* **A5 synthesis smoke that learns and beats greedy**: `make train-ai-smoke` trains a 4-qubit model in at most 150 s
  on 4 cores (`--iters` bound, fixed seed) and then, with `eval_synth` on 100 seeded blocks, **the beam-8
  policy has cost_ratio < 1.0 against QUMO's greedy synthesizer, win_or_tie >= 0.6, 0 verification failures** (the
  existing keep rule). Candidates in order: `lf` on `ring:4`, `lf` on `line:4`, `perm` on `ring:5`; the first that
  satisfies the rule is the smoke spec and is then frozen in the Makefile. Because exact search is optimal for such
  small blocks, "beats greedy" can only be measured against the greedy synthesizer, not the exact one; if none of
  the candidates beats greedy, that is a failing acceptance and a finding (O-7), not something to tune the rule
  for. Independently of the cost rule the smoke must show learning: curriculum difficulty >= 12 with deterministic
  success >= 0.85 and greedy rollouts solving >= 95 of 100 random blocks within `max_depth`.
* **A6 router BC**: (a) hard gate: synthetic planted-optimum data, agreement > 0.9 and regret < 0.1; (b) comparison
  with the Python trainer (data for the reference produced by C0 in wave 0, run by C8 as part of its DoD and again by
  C13 in wave 5 with `QUMO_BC_REF_D0` pointing at the lead's copy): both trainers train on the **same** `d0.bin`, built
  from at least 300 jobs (about 50k rows before the tie filter; its sha256 and the Python results for 3 seeds are in
  `route/py_bc_reference.json`), and the Mojo run uses the same 3 seeds. The r1 gate (40 jobs, a fixed 0.03 / 10 % band)
  was underpowered: after `nval = max(1000, N // 20)` the sampling std of one validation agreement is already about
  0.014 (binomial, p = 0.7, n = 1000) before any training variance. Acceptance: each Mojo `val_agree` and `val_regret`
  lies within `[min_py - m, max_py + m]`, where `min_py` and `max_py` are the extremes over the Python seeds and
  `m = max(0.03, 3 * std_py)`. This is a smoke check, not the quality judge; (c) quality, the real judge: the full round
  on the real recipe produces an exported model that beats the heuristic on the `route_tool eval` `scored` metric by the
  same geometric-mean margin as the committed router within 5 % (the shipped model's `val_agree` is not available, the
  manifest drops the log). Job-set drift of the Mojo generator is bounded separately by the statistics test of
  `test_rl_jobs` (13.2).
* **A7 export and loading**: a Mojo-trained synth model and router, exported into a staging root, pass
  `models_verify`, load through `load_model_file` / `load_route_model`, and the replay parts of
  `tests/test_policy.mojo` and `tests/test_ai_route.mojo` pass with `QUMO_TEST_MODEL_ROOT=<staging>`; the model is
  found and used by `transpile_ai` on a matching block (QM-E617/E620 not raised).
* **A8 no Python / Rust / torch in QUMO's training path**: `make check-pure-mojo` passes. What is claimed, precisely:
  *no Python or Rust is invoked or named by QUMO sources, by the Makefile recipes of the training targets, or by the
  built training binaries.* What is **not** claimed: the pinned Mojo toolchain itself. The repo's `.venv/bin/mojo` is a
  Python entry-point script with an absolute shebang (it supplies the module and library paths to the real compiler,
  which cannot be called directly: "unable to locate module 'std'"), and `make setup` installs the toolchain with uv and
  pip; the README says so plainly. The check has four parts:
  1. **source scan** (`check_pure_mojo.sh`): over every non-Markdown file under `training/` and `qumo/rl/` and over the
     expanded recipes (`make -n train-ai train-ai-smoke models-verify`), word-boundary regexes find none of
     `\bpython3?\b`, `\bpip3?\b`, `\buv\b`, `\btorch\b`, `\bnumpy\b`, `\bqiskit\b`, `\btwisterl\b`, `\brustworkx\b`,
     `\bsafetensors\b`, `\bcargo\b`, `\brustc\b`. Because `_` is a word character, `pipefail`, `pipeline` and the symbol
     `safetensors_import` (the optional importer in `checkpoint.mojo`, also named in tests) do not match. The exclusions
     are an explicit list: the checker script itself, `*.md`, `NOTICE`, `PROVENANCE.md`. C9 rewrites the
     `route_tool.mojo` docstring that says "Python/Mojo parity" (line 26), and C13 dry-runs the checker against the
     plan's own recipes before relying on it;
  2. **environment scan**: `env -i HOME=$HOME PATH=<tmp dir with symlinks to mojo, make, sh, the coreutils, cc and
     clang>` runs `make -n train-ai` and `make train-ai-smoke` to completion. `cc`/`clang` are needed because `mojo
     build` fails at the link step with "unable to find suitable c compiler for linking" otherwise, and `HOME` avoids
     Crashpad errors. No `python`, `python3`, `uv`, `pip`, `cargo` or `rustc` is on that PATH; the launcher's own
     absolute-shebang Python is the toolchain and is accepted explicitly;
  3. **binary scan**: for every `build/ai_train/bin/*`, `otool -L` (macOS) / `ldd` (Linux) lists no libpython and no
     libtorch, and a PATH shim that records calls to `python*`, `uv` and `pip` stays unused during the smoke run;
  4. no `.py` under `training/ai_transpiler/`; `requirements.txt` does not exist.
* **A9 performance**: gates of section 15.
* **A10 determinism**: T-DET and T-RESUME plus a full `train-ai-smoke` twice with `--cores 1`, `--cores 2` and
  `--cores 4` (a requested 8 clamps to the same effective value on the reference machine): identical `weights.bin`
  sha256. (G11 scaling is measured, no longer a gate.)
* **A11 documentation**: section 12.4 list done; `docs/AI_TRANSPILER.md` has no statement that training needs
  Python / torch / qiskit-gym / twisterl.
* **A12 (not merge-gating; run once before the campaign is trusted)**: full campaign on one lane subset (two models:
  `perm_grid3x3`, `lf_grid2x3`) at the shipped budgets. r1 required reaching "the shipped final difficulty (+/- 20 %)";
  that has no power: 8 of the 9 shipped models ended at the cap of 256 and `depth = min(2*difficulty, 128)` saturates at
  difficulty 64, so beyond 64 the number only measures how many iterations the clock allowed. The criteria are: (i) the
  **iteration at which difficulty first reaches 64** (a learning-speed measure; target within 1.5x of the Python curve
  in `ppo/py_curve.json`, which C0 records because the shipped run logs are not in the repository); (ii) the final
  `cost_ratio` and `win_or_tie` of the keep rule within 0.03 of the shipped values; (iii) iterations per second,
  reported separately. The result is recorded in the `docs/AI_TRANSPILER.md` ledger as "native retrain" rows.

---------------------------------------------------------------------------------------------------------

## 15. Performance targets

Python-stack reference numbers (from the shipped manifests, Apple M5, 4 cores for training; the runs shared the machine
between two lanes, as the docs ledger says, so they are somewhat pessimistic). Synthesis iteration wall = `budget_s /
iterations`: perm_grid3x3 0.61 s (600 s / 976), perm_king3x4 1.69 s, perm_hhring10s2 1.79 s, perm_grid3x4 1.89 s,
lf_grid2x3 1.94 s, lf_hhH7 2.90 s, lf_ring6 2.90 s, lf_line6 4.12 s (2700 s / 656) and **pauli_line4 6.27 s** (2100 s /
335 for its last stage; its R x R anticommutation table and DAG cleaning make it the model least likely to fit a
plain-SIMD budget): range 0.61-6.27 s over the 9 models, evals included (r1 said 1.4-4.5 s and 1.4 s for perm_grid3x3;
both were wrong, C-9). Router BC wall_s = 35.4 s for 168,176 samples, 30 epochs, hidden 64 (torch, 10 threads).

Machine facts that shape the targets (measured for r2): `parallelism_level()` is 4 on the reference machine (4
performance cores of 10; the 6 efficiency cores are unused by `parallelize`), so scaling past 4 cores cannot be measured
there. One row-epoch of the 512-256 net costs about 0.79 MFLOP (three GEMMs of 2 x 512 x 256), so 400k row-epochs/s
would need about 315 GFLOP/s and 1M about 790 GFLOP/s, unreachable without a matrix engine. Measured: a plain 4x16 FMA
tile reaches 63 GFLOP/s on one core; forward + `dW` + `dx` of the 512-256 net reach 117-122k row-epochs/s per core and
448-470k on four cores (the best case, without ReLU, loss, scatter, Adam and gradient reduction), and about 325k each
when two processes run it at once; a batch-1 matvec 512x256 takes 4.4-6.1 us (about 160-230k decisions/s per core).

| id | metric | gate (merge) | stretch |
|---|---|---|---|
| G1 | single-core GEMM (512 x 512 x 256) all three variants (plain 4x16 tile) | >= 60 GFLOP/s on the reference machine; kernels pass `test_rl_tensor` on x86-64 | >= 90 |
| G2 | synthesis iteration, config of lf_line6 (E=512, C=256, A=10, 1024 episodes, 4 cores, difficulty about 100, both evals every iteration) | <= 1.5 x the Python reference of that model (about 6 s) | <= 1.0 x |
| G3 | the same with `--sampled-eval-every 5` (default) | <= 1.0 x the Python reference | <= 0.6 x |
| G2p | synthesis iteration on the `pauli_line4` configuration (difficulty about 200, 1024 episodes, 4 cores, both evals every iteration) | <= 1.5 x the Python reference 6.3 s (about 9.4 s) | <= 1.0 x |
| G4 | collector throughput (4 cores) | >= 400,000 decisions / s on the lf_line6 net (r1's 150k was trivially met by the measured matvec cost) | >= 800,000 |
| G5 | `ppo_update` rows/s (10 epochs counted as 10 passes), **measured on the integrated `ppo_update`** (forward, loss, backward, reduction, Adam), diagnostic for G2 / G3 | >= 250,000 row-epochs / s at 4 cores for the 512-256 net | >= 400,000 |
| G6 | router BC, 168k samples x 30 epochs, hidden 64 | <= 53 s (1.5 x torch) with `--threads 4` | <= 35 s |
| G7 | `models_verify` on the 3.3 MB tree | <= 1 s (SHA-256 >= 100 MB/s) | - |
| G8 | `gen-jobs 31 1000 30 0.5` | <= 3 s | - |
| G9 | export of one synth model incl. vectors | <= 3 s | - |
| G10 | peak RSS at N = 130,000 rows | <= 2 GB | <= 1 GB |
| G11 | scaling: iteration time at 2 vs 4 cores (8 cores clamps to 4, section 3 rule 8) | measured and reported, **not a merge gate** | - |

`bench/bench_rl_gemm.mojo` (C4) measures G1 *before* C5 / C7 start (wave 1 exit criterion); `bench/bench_rl.mojo` (C13)
prints one JSON line per gate on the integrated code. **G2 / G3 / G2p are the real merge gates** because they bound the
cost of a whole iteration (about 1.3e5 rows x 10 epochs x 0.79 MFLOP is about 1 TFLOP per iteration, and G2 allows about
6 s); G5 is diagnostic. The GEMM work is still the dominant risk (R1): G1 below 40 GFLOP/s or G5 below 150k row-epochs/s
triggers the fallback list in 17. A time-boxed `--budget-s` campaign gives each model fewer iterations than the Python
campaign whenever the machine is slower or shared (see 10.3).

---------------------------------------------------------------------------------------------------------

## 16. Implementation plan: components, disjoint ownership, dependency order

Rule: **a file has exactly one owner**; an agent edits only files it owns plus its own `tests/test_rl_*.mojo`.
Anything a component needs from another is consumed through the signatures of section 5 (stubs that compile are
the first commit of each component). Shared files (`Makefile`, `docs/**`, `NOTICE`, `README.md`,
`qumo/ai/diagnostics.mojo`, `.gitignore`) belong to C13 alone, which only wires targets whose content other components own (`smoke.sh`, the
`test-rl*` targets). Every test file named in 13.2 and every Makefile target has exactly one owner.

Waves (a component starts when everything it depends on has passed its own tests):

```
W-1: lead compiles all of section 5 as one stub file (not committed); signatures are fixed before any component starts
W0:  C0
W1:  C1   C3   C4   C9            (parallel; C9 needs only existing Target + C0 fixtures)
W2:  C2 (C1,C0)   C5 (C4)   C6 (C3)   C8 (C4,C2)
W3:  C7 (C5,C6)   C10 (C2,C9)   C11a (C2,C3,C5,C6)
W4:  C11b (C11a)   C11c (C7,C11a,C11b)   C12 (C8,C9,C5,C4)
W5:  C13 (everything)
```
Critical path: C0 -> C4 -> C5 -> C7 -> C11c -> C13. Peak parallelism 4 agents (W1, W2).

### C0 - fixtures, spec archive, test utilities (W0)
Owns: `tests/fixtures/rl/**`, `training/ai_transpiler/spec/**`, `qumo/rl/__init__.mojo`, `qumo/rl/testutil.mojo`,
`tests/test_rl_fixtures.mojo`.
Does: copy the four research reports to `spec/` (renamed `ENVS.md`, `PPO.md`, `TORCH.md`, `REPO.md`); trim and
compact the golden JSON per 13.1; generate `topologies.json` from `bench/neutral/targets.py` + `circuits.py` in a
scratch venv (one-off, not committed); write `PROVENANCE.md`; `testutil.mojo`: `load_fixture(rel) raises ->
JsonValue`, `close(a, b, atol, rtol)`, `max_abs_diff`, `chi_square_z(...)`, `ints(v)`, `floats(v)`, `gemm_ref` (naive
triple loop) and guard-region helpers (`guard_alloc`, `guard_check`). While the Python code still exists, and in a
scratch venv built from the documented requirements (the one-off offline work that is the exception to rule 3 below),
C0 also produces every oracle that only that code can produce: (a) `envs/pauli_twists.json` from `symmetry.py`;
(b) `ppo/rollout_stats.json` with the real qiskit-gym env and the real twisterl collector/evaluate on fixed explicit
weights (13.1); (c) `route/jobs_stats.json` from the Python `build_jobs`; (d) `envs/reward_fma_cases.json`;
(e) `route/py_bc_reference.json`: Python BC on a fixed `d0.bin` (13.1, A6); (f) `ppo/py_curve.json`: 400 s Python runs
of `perm_grid3x3` and `lf_grid2x3` recording every promotion. (The r1 task "probe O-3" is dropped: the twisterl source
answers it, 4.1.) DoD: every fixture parses with `qumo.ai.json`; sizes within budget; `tests/test_rl_fixtures.mojo`
loads each fixture and checks a few invariants. About 400 LOC Mojo + data.

### C1 - digest and f16 (W1)
Owns: `qumo/rl/digest.mojo`, `qumo/rl/f16.mojo`, `tests/test_rl_digest.mojo`. Signatures: `Sha256` (update /
finalize), `sha256_hex`, `crc32`, `f32_to_f16_bits`, `encode_f16`. DoD: FIPS vectors, 65536-pattern round trip,
`f16_golden.json`, G7 throughput. About 250 LOC.

### C3 - topology and symmetry (W1)
Owns: `qumo/rl/topology.mojo`, `qumo/rl/symmetry.mojo`, `tests/test_rl_topology.mojo`. Signatures:
`parse_topology(spec) raises -> Topology{n, edges}`, `topology_slug`, `is_connected`, `automorphisms`,
`twists(kind, n, actions) -> Twists`, `pauli_twists(n, actions, ...)`, `pauli_qubit_perms_and_act_perms(n, actions)`,
`struct Twists{obs_perms, act_perms}`. DoD: 9 manifest tables identical, Pauli cases of `gateset_twists.json`, the `envs/pauli_twists.json` fixture (six
topologies, direction pinned), counts table. About 450 LOC.

### C4 - kernels, RNG helpers, optimizer (W1)
Owns: `qumo/rl/{tensor,rlrng,optim}.mojo`, `tests/test_rl_tensor.mojo`, `tests/test_rl_optim.mojo`,
`bench/bench_rl_gemm.mojo`. DoD: tests (including the guard-region cases and the `-D ASSERT=all` run), the `rlrng` unit tests (gumbel
tie rule, logits of 100, `mulf`), G1 (>= 60 GFLOP/s) measured and written to the PR description. If G1 < 40 GFLOP/s
the agent must try the fallbacks of 17-R1 before declaring done. About 900 LOC.

### C9 - topologies, circuit families, jobs (W1)
Owns: `qumo/rl/{targets,jobs}.mojo`, `training/ai_transpiler/route/route_tool.mojo`, `tests/test_rl_jobs.mojo`.
Signatures: `neutral_topology(name: String, param: Int/Tuple) raises -> RTopo{name, n, edges}` for line, ring, grid,
heavy_hex (via `Target.heavy_hex_edges`), diag grid, octagonal, tree (complete binary, parent `(i-1)//2`,
smallest level count with `2^(l+1)-1 >= min`), torus (side >= 3), atom2 (radius 2), ion zone ring
(`ions_per_zone=2`); **edge emission order verbatim from `bench/neutral/targets.py`** (it defines the active-swap
order); do not reuse `Target.tree/torus` unless the golden shows they are identical. `train_topologies() ->
List[RTopo]` (25), `eval_topologies()` (10); circuit families rand2q, qv, qft, trotter, qaoa, sparse with the
parameters of `circuits.py`; `build_jobs(seed, n_jobs, topologies, families, max_n, scale, min_frac) ->
JobSet`, `JobSet.to_json() -> String`; `eval_sets(out_dir)` (uses C1's `crc32` through its stub until C1 lands; synth seeds >= 910000, bench circuits 3 <= n <= 14 on
3 topologies `(crc32(name) + 3k) % 10`, seed 7); `summarize(path)`; move `circuit_from_json`, `target_from_json`,
`put_i16`, `put_f32` out of `route_tool.mojo` into `jobs.mojo`; add `gen-jobs`, `eval-sets`, `summarize` commands;
existing commands behave byte-identically (regression: `route_tool gen` output on a fixed jobs file before/after
the refactor has the same sha256). The job set differs from the shipped campaign (different RNG), but **how different is bounded**: `test_rl_jobs`
compares family mix, size, 2q fraction, depth and topology mix with `route/jobs_stats.json` (C0, from the Python
`build_jobs`, 2000 jobs at scales 0.5 and 1.0) by chi-square / two-sample KS at 4 sigma. DoD also includes rewording the
`route_tool.mojo` docstring that says "Python/Mojo parity" (so `check-pure-mojo` passes).
About 800 LOC.

### C2 - container, INDEX, verify (W2)
Owns: `qumo/rl/container.mojo`, `training/ai_transpiler/common/models_verify.mojo`, `tests/test_rl_container.mojo`.
Signatures as section 5 plus `dump_indent1`, `index_row_from_dir`, `atomic_replace_dir`, `recover_dir` (9.1: write
`<dir>.tmp`, read back, libc `rename` swap). DoD: A2 and a test that a simulated stop between the rename steps leaves a
complete old or new directory. About 500 LOC.

### C5 - network and synthesis loss (W2)
Owns: `qumo/rl/{nn,loss}.mojo`, `tests/test_rl_nn.mojo`, `tests/test_rl_loss.mojo`. DoD: goldens of 13.2 (including the 9 shipped manifests' independent test vectors replayed through
`BasicNet.from_model`), T-DET
fragments (thread-count independence of `backward` over virtual workers). About 800 LOC.

### C6 - gateset and training environment (W2)
Owns: `qumo/rl/{gateset,synth_env}.mojo`, `tests/test_rl_env.mojo`, `tests/test_rl_env_slow.mojo`. Stub-first: expose the `TrainEnv` interface with
the plain kinds, then Pauli. DoD: A3 (including the `mulf` reward cases and the QM-E642 rejection of non-golden gates). About 1,100 LOC (largest component; the Pauli generator and hidden symmetry
are about a third).

### C8 - router data, net, loss, BC (W2)
Owns: `qumo/rl/{teacher_data,router_net,router_loss,bc}.mojo`, `training/ai_transpiler/route/train_route.mojo`,
`tests/test_rl_router.mojo`. DoD: goldens, planted optimum, the NaN-safe filter and `Q` clip cases, BC reference numbers of A6(b) (against C0's
`py_bc_reference.json` and the lead's `d0.bin`), G6. About 800 LOC.

### C7 - PPO (W3)
Owns: `qumo/rl/ppo.mojo`, `tests/test_rl_ppo.mojo`, `tests/test_rl_ppo_slow.mojo`. DoD: A4 (including the
rollout-statistics fixture and the collector self-consistency test), G2-G5 and G2p measured. About 800 LOC.

### C10 - router export (W3)
Owns: `qumo/rl/export_route.mojo`, `training/ai_transpiler/route/export_route.mojo`, `tests/test_rl_export_route.mojo`
(the router half of the export tests of 13.2; the synthesis half is `tests/test_rl_export.mojo`, owned by C11a). About
350 LOC.

### C11a - checkpoint, synth export, fixtures regenerator (W3)
Owns: `qumo/rl/{checkpoint,export_synth}.mojo`, `training/ai_transpiler/synth/export_synth.mojo`,
`training/ai_transpiler/common/make_fixtures.mojo`, `tests/test_rl_export.mojo`. Contains: run-dir layout
(`run.json`, `best/`, `last/`, `state.json`, `adam.bin`, `log.jsonl`, `summary.json`), atomic checkpoint read/write
(f32 container, through C2's `atomic_replace_dir`), `check_pauli_ensemble` and the independent pre-write checks of 9.2, optional
`safetensors_import(path) -> List[(name, shape, data)]` (read-only, for old runs and for `tiny_policy.safetensors`
test), the exporter of 9.2. About 700 LOC.

### C11b - block evaluation (W4)
Owns: `qumo/rl/eval_blocks.mojo`, `training/ai_transpiler/synth/{eval_blocks,eval_synth}.mojo`. The body of the
existing `eval_blocks.mojo` becomes a library returning the same JSON; `eval_synth` applies the keep rule and
re-exports. DoD: JSON of the library equals the old tool's output on the same model (sha256 of the JSON). About 450 LOC.

### C11c - synth driver, campaign, smoke (W4)
Owns: `qumo/rl/synth_train.mojo`, `training/ai_transpiler/synth/{train_synth,train_all}.mojo`,
`training/ai_transpiler/synth/campaign.json`, `training/ai_transpiler/smoke.sh`, `tests/test_rl_smoke.mojo`. DoD: A5
(smoke) and A10, run by invoking `smoke.sh` directly (C13 later only wires the `train-ai-smoke` target); resume
(T-RESUME); idempotent `train_all` (status files, lane logs, non-zero exit on failure). About 700 LOC.

### C12 - router PPO (W4, lowest priority; depends on C8, C9, C5, C4)
Owns: `qumo/rl/route_ppo.mojo`, `training/ai_transpiler/route/train_route_ppo.mojo`, `qumo/compiler/ai_route.mojo`
(additive edits only), `tests/test_rl_route_ppo.mojo`. DoD: `ppo_route_golden.json` in the integrated trainer,
`test_ai_route.mojo` still bit-exact, a 2-iteration smoke. About 500 LOC.

### C13 - integration, docs, deletion, performance (W5)
Owns: `Makefile` (every target, including `train-ai-route-ppo`, `test-rl`, `test-rl-slow` and the `TESTS` filter),
`docs/**`, `README.md`, `NOTICE`, `.gitignore`, `qumo/ai/diagnostics.mojo`, `qumo/compiler/policy.mojo` (hint string
only), `tests/test_policy.mojo`, `tests/test_ai_route.mojo` (env override only), `tests/test_rl_e2e.mojo`,
`bench/bench_rl.mojo`, `training/ai_transpiler/check_pure_mojo.sh`, `training/ai_transpiler/README.md`,
`training/ai_transpiler/route/README.md`, deletion of the 17 Python files and `requirements.txt`. Runs A1-A11, records
G1-G10 (G11 is measured), re-runs the A6(b) comparison before the deletion, runs A12 if time permits. About 400 LOC plus
docs. (`smoke.sh` and `test_rl_smoke.mojo` belong to C11c.)

Total estimate: the per-component figures above sum to about 9,600 LOC Mojo (r1 said 8,600; its own per-component
numbers did not add up to that), of which about 2,700 are tests, versus about 3,200 Python lines replaced (repo.md
estimated 3.5-4.5 k; this plan is larger because it also ships the environments, which repo.md could not size, because
it adds the rollout-statistics test, resume and the atomic writer, and because tests are counted).

### Per-component rules
1. First commit: compiling stubs of the section 5 signatures, so dependants can import.
2. Every component states in its PR description: files touched (must be a subset of its ownership), tests run,
   numbers for any gate it owns.
3. No component runs Python or opens the source of qiskit-gym / twisterl / torch. The only exceptions are C0's
   one-off offline fixture generation in a scratch venv built from the documented requirements and C13's wave-5 re-run
   of the A6(b) comparison before the deletion; nothing of that is committed except data.
4. If a golden test fails and the fixture looks wrong, the component reports it to the lead; it does not edit
   `tests/fixtures/rl` (C0 owns it).
5. The lead integrates waves in order and updates section 5 if a signature moved.
6. The research scripts that produced the goldens (`envs_ref.py`, `ref_ppo.py`, `ref_np.py`) are not committed (D11:
   they are line-level transcriptions). The golden data arbitrates, and where a spec text and a golden disagree the golden
   wins. For a case no golden covers, the component asks the lead, who keeps the scripts in scratch and can have C0 add a
   fixture; implementers do not lose the tie-breaker, they reach it through the lead.

---------------------------------------------------------------------------------------------------------

## 17. Risks

| id | risk | mitigation |
|---|---|---|
| R1 | Portable SIMD GEMM too slow; iterations 2-3x slower than the Python stack, which uses vendor BLAS (Accelerate / AMX on Apple) | measure G1 first (wave 1 gate). Fallbacks in order: (1) panel packing + K-blocking on top of the baseline 4x16 register tile (which already measured 63 GFLOP/s single-core); (2) exploit ReLU sparsity in the backward of the 512x256 layer (skip rows where `dpre0 == 0`); (3) reduce evaluation cost (`sampled-eval-every`); (4) optional CBLAS build `-D QUMO_RL_BLAS` binding the standard `cblas_sgemm` (vendor-neutral symbol, off by default, loses the thread-count determinism guarantee, not used by acceptance); (5) Mojo GPU path is out of scope here |
| R2 | f32 divergence vs torch in goldens (Adam amplifies tiny gradient noise: update is about `lr * sign(g)`) | tolerances already scale with lr (torch.md 9); run the kernels in Float64 once to confirm the 1e-10 level against the f64 reference when a mismatch appears; fixed reduction order |
| R3 | Pauli env complexity (generator, mirrored CX, hidden symmetry, petgraph order) | `PolicyEnv` already implements the transition core; exact-rational reset tests; `pauli_perm_cases`; behaviourally validated spec (envs.md 13); C6 builds plain kinds first |
| R4 | Spec drift between the research reports and the Rust (the Rust source was read, but the compiled binary was not diffed line by line) | the goldens were produced from the real binary; where spec text and goldens disagree the goldens win (C-1 below is one such case) |
| R5 | Mojo compiler limits: `parallelize` closure captures, ASAP destruction of lists behind raw pointers, long compile times of big modules, `Layout` alignment API drift | modules <= 1200 lines; `_ = len(x)` after parallel regions; owners compile early and often; no `fn`; follow `docs/MOJO_GUIDE.md` |
| R6 | Retrained models are not the shipped ones (different RNG) and may land slightly better or worse | acceptance is quality-based (A5, A12); shipped models stay committed and are never overwritten by CI; `make train-ai` writes new models in place only when run by a human |
| R7 | Cost of the sampled evaluation (`ppo_10` = 2,000 episodes per iteration, more than the collector) | default every 5 iterations, faithful mode available; it influences nothing |
| R8 | Time-bounded training makes runs differ in length | `--iters` for exact runs; documented in 3 and 7.5 |
| R9 | Memory: N up to 1.3e5 rows with `E = 512` activations if run unchunked | chunked processing (`CHUNK_ROWS`), per-worker caches only; G10 |
| R10 | f16 overflow at export aborts a long training | export is a separate step; `train_synth` also writes a finiteness warning when `max|w|` > 60000 |
| R11 | Licensing perception: a transliteration of Rust/Python logic | clean-room rules D11, no research scripts committed, NOTICE text 12.5 |
| R12 | Defects of the old pipeline silently reproduced | list in 18 (C-2..C-6) each fixed deliberately: layer numbering, INDEX default, unused budget variable, unused `max_rotations`, undeclared rustworkx |
| R13 | Smoke cannot beat greedy on tiny blocks because greedy is already optimal | candidate list in A5; failure is reported, not hidden |
| R14 | Cross-platform FMA differences make A10 fail on another CPU | A10 is per machine; tests with fixed seeds use tolerances where they compare to goldens and exact equality only within one binary; bit-exact reward / GAE parity with the Rust env additionally needs the `mulf` barrier (section 3, rule 9) |
| R15 | Pauli training policy has no symmetry ensemble but exported model gets one at inference | same as the shipped `pauli_line4`; A5 does not include Pauli; Pauli quality is checked in A12-style run by the owner of the Pauli model |
| R16 | A bug in `PolicyEnv` (inference env) would now also corrupt training | the env golden tests exercise `PolicyEnv.apply/observe/solved` through `TrainEnv` against the real Rust envs, which also strengthens the inference side |
| R17 | `parallelize` has only 4 pool threads on the reference machine and two lanes slow each other (about 0.69x each) | `--cores` clamped to `parallelism_level()` (3, rule 8); `AI_LANES=1` default; G11 is no longer a gate; campaign `iters` option |
| R18 | A kill, power loss or segfault during an 8 h lane corrupts a checkpoint or loses the lane | atomic directory writes (libc `rename`), `state.json` + `adam.bin` for bit-exact resume, idempotent `train_all` with status files (7.5, 9.1, 10.3) |
| R19 | Raw-pointer kernels silently corrupt memory on an off-by-one (no bounds checks) | sentinel guard regions in every tensor test, `-D ASSERT=all` run, `gemm_ref` |
| R20 | `PolicyEnv`'s Pauli `CZ` argument order differs from the reference (latent: no golden gateset contains CZ, SX or Sdg) | `TrainEnv` accepts only {H, S, CX, SWAP} (QM-E642 otherwise); fix later together with a CZ/SX/Sdg golden (C-8, O-12) |
| R21 | Pauli export ensemble direction wrong on non-involutive automorphisms: loads, passes the shipped `line:4` test, quietly degrades inference | `envs/pauli_twists.json` on six topologies plus `check_pauli_ensemble` as test and export assertion (4.6, 9.2) |

---------------------------------------------------------------------------------------------------------

## 18. Open questions and corrections to the research reports

### 18.1 Corrections found while writing this plan

* **C-1** `ppo.md` section 6 says an all-masked row has `old_logp = -ln(A)`. The golden (`ppo_update.json`, batch
  row 13, all four actions masked) has `old_logp = 0.0`, which is what f32 `logits - logsumexp(logits)` gives when
  every logit is `-1e10` (`ln A` is absorbed by f32 spacing 1024 at 1e10). The port computes it in f32 with the
  max-shift logsumexp and the test pins 0.0. Consequence: such rows have ratio `p_new(a)` (about `1/A`), far below
  `1 - c`, so they push `p(a)` up when `adv > 0` and are clipped when `adv < 0`.
* **C-2** repo.md 3.5 says the `float16` encoder and SHA-256 "may" be missing; probed: encoder exists via
  `Float16`, SHA-256 does not exist (section 1.4).
* **C-3** repo.md reports 17 Python files; the tree has 17 plus the missing `train_all.sh`; the Makefile also has
  an unused `AI_ROUTE_BUDGET_S`; `train_route.py --budget_s 1800` truncates silently (fixed by surfacing the budget).
* **C-4** the Pauli `max_rotations` argument of `train.make_env` is unused (qiskit-gym default 5); the Mojo config
  makes it explicit in `run.json` and `EnvConfig.max_rotations`.
* **C-5** `basic_mlp_tensors` layer numbering (dormant: all shipped models have one Linear per sequence) is fixed.
* **C-6** `export_route.update_index` loses the previous default flag when exporting without `--default`; fixed in
  9.3.
* **C-7** `ppo.md` section 6 says the action recorded on an all-masked row is "a random one". The twisterl source
  disagrees: its `argmax` is a strict `>` scan seeded with element 0, and in f32 the Gumbel term is absorbed at
  `-1e10`, so the action is index 0 (4.1). The evaluate path also differs from the plan's max-shifted softmax for
  logits above about 88 (4.1, 7.4).
* **C-8** `PolicyEnv`'s Pauli `CZ` branch calls `_pauli_cx(st, a, b)`, while the reference does `H(q1); CNOT(q0, q1);
  H(q1)` with the mirrored CNOT (`row[q0] ^= row[q1]`), the same pattern as `PolicyEnv`'s own CX branch
  (`_pauli_cx(st, b, a)`). Latent: `build_gateset` never emits CZ, SX or Sdg and the step goldens contain only H, S, CX,
  SWAP. Handled by restricting `TrainEnv` (6.1); the runtime fix is deferred (O-12).
* **C-9** Section 15's r1 Python reference timings (1.4-4.5 s, perm_grid3x3 1.4 s) were wrong; the manifests give
  0.61-6.27 s.

### 18.2 Open questions (with recommended default)

| id | question | recommendation |
|---|---|---|
| O-1 | Keep the faithful quirks (masked-old / unmasked-new ratio; post-update weights as "best")? | keep (D6); decide after the first native campaign by an ablation |
| O-2 | Evaluate `ppo_10` every iteration (log parity) or every 5? | every 5 by default; `--sampled-eval-every 1` for parity |
| O-3 | ~~Which action does twisterl record on an all-masked terminal row?~~ | **Resolved**: index 0. The twisterl `argmax` is a strict `>` scan from element 0 and the f32 Gumbel term is absorbed at `-1e10` (4.1, C-7); no probe is needed |
| O-4 | Is the `ppo_deterministic` policy evaluated with one random twist per step (faithful) or the ensemble? | faithful (one random twist); the ensemble is an inference-time feature of the Mojo runtime |
| O-5 | ~~Exact relation (direct vs inverse) between the Pauli training env's internal `qubit_perms / act_perms` and the manifest's export ensemble~~ | **Resolved by method** (4.6): the `envs/pauli_twists.json` fixture on six topologies including non-involutive ones, plus the `check_pauli_ensemble` behavioural check; the `pauli_line4` manifest alone cannot decide because both of its automorphisms are involutions |
| O-6 | Should the dropped model specs (clifford_*, lf_line7, pauli_complete5, pauli_line6, perm_line8) be re-created? | no; keep disabled entries in the campaign |
| O-7 | If no 4-qubit smoke candidate beats greedy at 150 s, relax the smoke time or the candidate list? | extend the time budget first (up to 300 s), then widen candidates; never relax the keep rule |
| O-8 | Should `Target.tree` / `Target.torus` be unified with the neutral-benchmark variants? | no; separate generators in `targets.mojo`, golden-tested; unification is a later cleanup |
| O-9 | Ship a safetensors *writer*? | no; read-only importer only |
| O-10 | A Mojo GPU (Metal / CUDA) training backend? | out of scope; the kernels are isolated in `tensor.mojo` so a backend can be added without touching callers |
| O-11 | Regenerate `models/ai_transpiler` with the native stack and replace the shipped weights? | separate decision after A12; never part of this port's merge |
| O-12 | When to fix `PolicyEnv`'s Pauli `CZ` argument order (C-8)? | together with a Pauli / Clifford golden that contains CZ, SX and Sdg, as an additive runtime change with the inference tests re-run; not part of this port's training path |

---------------------------------------------------------------------------------------------------------

## 19. Appendix

### 19.1 Constants (in effect for QUMO)

| constant | value | source |
|---|---|---|
| gamma, lambda | 0.995, 0.995 | PPOConfig |
| epochs per iteration | 10 full-batch | PPOConfig |
| clip, vf, ent | 0.1, 0.8, 0.01 | PPOConfig |
| lr (PPO), betas, eps | 3e-4, (0.9, 0.999), 1e-8 | torch defaults |
| episodes / iteration | 1024 | train.py |
| evals | `ppo_deterministic` 200 ep argmax x1; `ppo_10` 200 ep sampled x10 | train.py |
| diff threshold / max / start | 0.85 / 256 / 1 | PPOConfig, train.py |
| env | `depth_slope` 2, `max_depth` 128, `add_inverts`, `add_perms` True; metrics weights (0.01, 0, 0, 0.0001) | qiskit-gym defaults |
| Pauli | `max_rotations` 5, `pauli_diff_scale` 16, `num_qubits_decay` 0.5, `final_pauli_layers` 7, `pauli_layer_reward` 0.01 | PauliGym defaults |
| net | E 512, common [256], heads single Linear | train.py |
| BC | H 64, tau 0.5, lr 2e-3 cosine per epoch, batch 512, 30 epochs, seed 0, budget 1800 s | train_route.py |
| router PPO | gamma 0.99, lam 0.95, clip 0.1, vf 0.5, ent 0.005, lr 2e-4, clip-grad 0.5, 4 epochs, mb 1024 | train_ppo.py |
| teacher | records 328 B, cap 96 (Makefile), G 16, beta 1.0 / 0.3, seeds 401 / 402 | route_tool |
| container | align 64, f16 matrices, f32 biases, FNV-1a64 + SHA-256 over the whole file | export.py |
| error codes reserved | QM-E640 .. QM-E649 | section 5 |

### 19.2 Golden-to-component map

| golden | component that must pass it |
|---|---|
| `gateset_twists.json` | C3, C6 |
| `steps_*.json`, `steps_custom_args.json`, `inverts.json`, `pauli_perm_cases.json`, `reset_*.json`, `defaults.json` | C6 |
| `policy_forward.json`, `synth_ppo_golden.json` | C5 |
| `gae.json`, `ppo_update.json` | C5 (loss, GAE), C7 (end-to-end update, determinism) |
| `adam_cosine_golden.json` | C4 |
| `bc_golden.json`, `ppo_route_golden.json` | C8, C12 |
| `f16_golden.json` | C1 |
| `tiny_policy.safetensors`, `safetensors_header.json`, `misc_golden.json` | C11a (importer), C0 (provenance) |
| shipped `models/ai_transpiler/**` | C2, C3, C5 (vectors through `BasicNet.from_model`), C11a, C10 |
| `envs/pauli_twists.json` | C3, C11a (`check_pauli_ensemble`) |
| `envs/reward_fma_cases.json` | C6 |
| `ppo/rollout_stats.json` | C7 (`test_rl_ppo_slow`) |
| `route/jobs_stats.json` | C9 |
| `route/py_bc_reference.json` | C8, C13 |
| `ppo/py_curve.json` | A12 (reference for the learning-speed criterion) |

### 19.3 Reading order for an implementing agent

1. This file sections 1-5 and the section of your component. 2. The matching spec in `training/ai_transpiler/spec/`.
3. `docs/MOJO_GUIDE.md`. 4. The existing module you reuse (`policy.mojo`, `policy_synth.mojo`, `rng.mojo`).
5. The fixtures you must pass. Do not read the source of qiskit-gym, twisterl or torch.

---------------------------------------------------------------------------------------------------------

## 20. Review log (r1 to r2)

Each review objection, verified against the repository, the research reports and the review probes where that was
cheap, and its disposition. "Accept" means the plan text changed as described; section numbers are the places changed.

* **L-1** [major, 13.2 / 13.3 / 14 A4-A5, rollout pipeline unpinned]: ACCEPT. Confirmed that `ppo.md` and `torch.md` list
  no golden rollout from a real env and that A5 / A12 judged only learning quality. Added `ppo/rollout_stats.json` (four
  configs, fixed explicit weights, `evaluate` over >= 20,000 and collector over >= 5,000 episodes) as 4-sigma
  `test_rl_ppo_slow`, plus a collector self-consistency test; A4 extended; 13.3 rewritten (7.2, 13.1-13.3, 14, C0, C7).
* **L-2** [major, Pauli export ensemble direction]: ACCEPT. Verified on the manifest that both `pauli_line4` `act_perms`
  are involutions, so it cannot pin the direction. Added `envs/pauli_twists.json` (ring:5, ring:4, star:4, grid:2x3, H7,
  T5), dumped before `symmetry.py` is deleted, and `check_pauli_ensemble` as a test and an `export_synth` assertion; O-5
  resolved by method (4.6, 9.2, 13.1-13.2, 18.2, C0, C3, C11a).
* **L-3** [minor, seed derivation across continuation stages]: ACCEPT. Verified in `train.py` that the iteration counter
  restarts per run and that all three `pauli_line4` stages use seed 1. The seed key is now the global iteration
  `start_iteration + it`, stored in `run.json`; T-DET checks that stage 2 iteration 0 differs (3 rule 1, 7.5, 13.2).
* **L-4** [minor, FMA contraction]: ACCEPT. Confirmed with the review probes (fused 0.04989999905 vs unfused
  0.04989999533) that the shipped goldens cannot discriminate. New rule 9 (`mulf`, `@no_inline`) for the penalty chain,
  reward and GAE, a bit-exact `reward_fma_cases.json`, and R14 / A3 reworded (3, 4.1, 4.5, 6.1, 13, 14, 17).
* **L-5** [minor, gumbel_argmax / O-3 / evaluate]: ACCEPT. Verified in twisterl 0.5.1 `policy.rs`: `argmax` is a strict
  `>` scan from `values[0]`. O-3 resolved (index 0) and removed from C0; correction C-7; the max-shift deviation for
  |logit| > 88 is stated and tested with logits of 100 (4.1, 7.4, 18).
* **L-6** [minor, PolicyEnv Pauli CZ]: ACCEPT, restriction option chosen. Verified against `envs_ref.py` (CZ is `H(b);
  CNOT(a, b); H(b)` with the mirrored CNOT) and `policy_synth.mojo` lines 383-396 that the argument order differs.
  `TrainEnv` accepts only {H, S, CX, SWAP} (QM-E642), the 6.1 and R16 claims are scoped, and the runtime fix is deferred
  (C-8, O-12, R20) so that `policy_synth.mojo` stays untouched in this port (5, 6.1, 17, 18).
* **L-7** [minor, setup behaviours dropped]: ACCEPT. Verified in `train.py`: the `num_qubits` abort, the `__cont`-only
  inheritance with the silent `except Exception` fallback, and explicit `--init` starting at difficulty 1.
  `build_gateset` derives n and raises QM-E641; warm-start cases stated one by one (`auto` errors on a missing
  summary; explicit `--init` needs `--start-diff`); exactly one of `--iters` / `--budget-s`; the loop is `while not stop`
  (D7, 5, 7.5, 10.1).
* **L-8** [minor, BC filter]: ACCEPT. Verified `spread > 1e-6` and `np.minimum(Q, 1e6)` in `train_route.py`. Keep-if-greater
  form, the `Q` clip and the A = 0 / A = 1 test cases (8.1, 13.2).
* **L-9** [minor, self-referential export vectors]: ACCEPT. Independent pre-write checks (BasicNet vs MLPPolicy vs a naive
  Float64 forward) in `export_synth`, and the shipped manifests' independent vectors replayed through
  `BasicNet.from_model` in `test_rl_nn` (9.2, 13.2).
* **L-10** [minor, route PPO details]: ACCEPT. `torch.md` 5.2 declared normative and its GAE / old-logprob / `ret` /
  `rng.choice` rules copied into 8.3.
* **L-11** [major, threading: cores, lanes, G11]: ACCEPT. Re-verified `sysctl hw.perflevel0.logicalcpu` = 4 and the review
  scaling probe. `--cores` is clamped to `parallelism_level()`, G11 becomes a measurement, T-DET and A10 use cores 1 / 2 /
  4, `AI_LANES` defaults to 1, campaign entries may carry `iters`, the unused E-cores are recorded (3 rule 8, 7.5,
  10.1-10.3, 14, 15, R17).
* **L-12** [major, miscalibrated performance gates]: ACCEPT. G1 >= 60 GFLOP/s, G4 >= 400k decisions/s, G5 >= 250k
  row-epochs/s (stretch 400k) on the integrated update; G2 / G3 are the merge gates. The first kernel is the plain 4x16
  tile with a once-per-epoch transposed weight copy; packing, blocking and the sparse-Adam rows are optional (4.2, 4.3,
  15, 17 R1, C4).
* **L-13** [major, A8(2) cannot pass]: ACCEPT. Verified that `.venv/bin/mojo` is a script with a Python shebang and
  `from mojo._entrypoints import exec_mojo`. A8 now claims only "no Python or Rust invoked or named by QUMO sources,
  recipes or built binaries", allows `cc`/`clang`/`HOME`, adds a binary scan and says plainly that `make setup` and the
  launcher use uv, pip and Python (14 A8, 12.4).
* **L-14** [major, A8(1) grep false positives]: ACCEPT. Verified `route_tool.mojo` line 26 ("Python/Mojo parity") and that
  `pip` matches `pipefail`. Word-boundary regexes, an explicit exclusion list, `safetensors_import` allowed, C9 rewords the
  docstring, C13 dry-runs the checker (14 A8, C9, C13).
* **L-15** [major, `wait` hides lane failures]: ACCEPT. Per-PID wait recipe; `train_all` exits non-zero on any failed
  enabled entry (10.2, 10.3).
* **L-16** [major, no crash safety or resume]: ACCEPT, extended. Atomic directory writes via libc `rename` through
  `external_call` (`std.os.rename` does not exist, re-verified), plus `state.json` + `adam.bin` so a resumed `--iters` run
  is bit-identical (T-RESUME), idempotent `train_all` with status files and lane logs (7.5, 9.1, 10.3, C2, C11a, C11c, R18).
* **L-17** [minor, tasks cannot raise]: ACCEPT. Re-ran the probe: `parallelize` rejects a `raises` task. Rule 7 (status
  slots scanned in index order), `Rollout.status`, `EvalResult.status`, clamped `step` with `error_code()` (3 rule 7, 5, 6.1).
* **L-18** [minor, non-compiling contracts]: ACCEPT. Re-ran the probe: `InlineArray` is absent from `std.collections`. Named
  weight fields, the `FPtr` alias, a concrete `write_model_dir` signature, the `;` shorthand declared, and the lead
  compiles section 5 as one stub in W-1. I did not use `Array`, which I did not verify (5, 4.2, 16).
* **L-19** [major, A6(b) underpowered]: ACCEPT WITH CHANGES. >= 300 jobs, acceptance by the Python seed spread (`m =
  max(0.03, 3 std)`), planted-optimum as the hard gate, real quality judged by A6(c). Change: both trainers use the same
  `d0.bin` built from Python-built jobs in wave 0, not Mojo-generated jobs, which removes the dataset confound; job-set drift
  is bounded separately by L-20 (14 A6, 13.1, C0, C8).
* **L-20** [major, job generator drift untested]: ACCEPT. `route/jobs_stats.json` from the Python `build_jobs` and a
  chi-square / KS test in `test_rl_jobs` (13.1, 13.2, C0, C9).
* **L-21** [minor, A12 has no power]: ACCEPT. Verified in the manifests: 8 of 9 models end at difficulty 256 and
  `pauli_line4` at 208. A12 now uses the iteration at which difficulty first reaches 64 (Python curve recorded by C0 with
  400 s runs, as the shipped run logs are not in the repo), the keep-rule numbers, and iterations per second separately
  (14 A12, 13.1).
* **L-22** [minor, wrong Python timings, no Pauli gate]: ACCEPT. Recomputed from the manifests: 0.61 s to 6.27 s. Section 15
  corrected, G2p added for `pauli_line4`, two-lane contention noted (15, 18.1 C-9).
* **L-23** [minor, ownership and dependency gaps (a)-(g)]: ACCEPT (a)-(f), REJECT (g) as stated. Every test file and
  Makefile target now has one owner, C12 depends on C9, rule 3 has the C0 / C13 exception, `crc32` stub for C9. For (g):
  committing `envs_ref.py` / `ref_ppo.py` contradicts D11 (they are line-level transcriptions); instead the golden data
  arbitrates and the lead keeps the scripts as the tie-breaker (16 rule 6, preface).
* **L-24** [minor, test-time claim and LOC total]: ACCEPT. Withdrew the "<= 120 s" claim; per-file 30 s cap, `_slow` files,
  `make test-rl` and `test-rl-slow`; LOC total corrected to about 9,600 (re-added the component figures: 350 + 250 + 450 +
  900 + 800 + 450 + 800 + 1,100 + 800 + 700 + 350 + 700 + 450 + 600 + 500 + 400). The reviewer's cold timings were not
  reproduced exactly: `test_policy` ran in 1.5 s with a warm cache, so the text cites cold figures as the reviewer's (13
  intro, 10.2, 16).
* **L-25** [minor, continuation-run seeds, duplicate of L-3]: ACCEPT as L-3. The "rerun with a changed budget replays the
  same prefix" remark is intended behaviour (D7) and is documented, not changed.
* **L-26** [minor, unchecked raw-pointer kernels]: ACCEPT. Sentinel guard regions in every tensor test, `gemm_ref` in
  `testutil`, one `-D ASSERT=all` run in `test-rl-slow` (4.2, 13.2, C0, C4, R19).

Revision: r2, 2026-11-07 (r1 plus the review log above; r1 was the initial plan).
