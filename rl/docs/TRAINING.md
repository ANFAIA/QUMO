# Training QUMO's AI-transpiler models

QUMO's AI transpiler (docs/AI_TRANSPILER.md) runs natively in Mojo, and so does its training, in the independent
package `rl/` (see [rl/README.md](../README.md) for the package layout, the standalone use and the QUMO bridge). The
tool mains are in `rl/drivers/` (QUMO-free) and `rl/qumo_bridge/drivers/` (the tools that need QUMO's compiler); the
libraries behind them are `rl/qrl/` and `rl/qumo_bridge/`. `make train-ai` and `make models-verify` run **no Python,
no Rust, no torch and no numpy**: every step is a Mojo binary. The design of record, with the exact math, the
determinism rules and the acceptance criteria, is [MOJO_PORT.md](MOJO_PORT.md). The make targets below exist in the
QUMO root Makefile and in `rl/Makefile` (the root delegates).

| directory | produces | runtime consumer |
|---|---|---|
| `rl/drivers/synth/` (+ `eval_synth`, `eval_blocks` in `rl/qumo_bridge/drivers/`) | synthesis policies (`qumo.rlsynth.v0`) for permutation, linear-function, Clifford and Pauli-network blocks, trained by PPO in native ports of the four synthesis environments | `qumo/compiler/policy.mojo`, `policy_synth.mojo` (passes `perm_synth`, `linear_synth`, `clifford_synth`, `pauli_network`) |
| `rl/drivers/route/` (+ `route_tool`, `export_route`, `train_route_ppo` in `rl/qumo_bridge/drivers/`) | the learned router (`qumo.rlroute.v0`), trained by behaviour cloning / DAgger in QUMO's own routing environment | `qumo/compiler/ai_route.mojo` (`route(..., policy=auto)`) |
| `rl/drivers/common/` | `models_verify` (sha256 of every model file against its INDEX.json) and `make_fixtures` (random-weight policy fixtures for `tests/fixtures/ai/policy`) | |
| `rl/spec/` | the written specifications the port was built from (Markdown, data only) | |

Everything is Apache-2.0. The training algorithms (environments, metrics-based rewards, curriculum, PPO with
generalised advantage estimation, a `BasicPolicy`-shaped network with symmetry augmentation) are a clean-room
re-implementation from written specifications of the algorithms of qiskit-gym 0.4.1 and twisterl 0.5.1 (both
Apache-2.0); see `NOTICE`. None of those packages, nor torch, numpy or safetensors, is needed to train, run or test
QUMO. No IBM model weights are used; the trained weights are QUMO's.

## What is and is not Python-free

The training sources, the recipes of the training targets and the built training binaries invoke and name no Python,
Rust, torch or numpy (`make check-pure-mojo` enforces it, see below). What this does **not** claim: the pinned Mojo
toolchain itself. `.venv/bin/mojo` is an entry-point script that runs under the virtual environment's interpreter
(it supplies the module and library paths to the real compiler), and `make setup` installs the toolchain with `uv` and
`pip`. So installing and launching the compiler still touches Python; nothing QUMO builds or runs for training does.

## Make targets

| target | what it does | budget |
|---|---|---|
| `make ai-tools` | builds every tool into `build/ai_train/bin` | a few minutes |
| `make train-ai` | `train-ai-synth` + `train-ai-route`, then `make models-verify` | synthesis cap `AI_SYNTH_BUDGET_S` (28,800 s), the router a few minutes |
| `make train-ai-synth` | `train_all` over `rl/drivers/synth/campaign.json`: trains, evaluates and exports every enabled model; a model is exported only if it beats the greedy synthesizer (keep rule) | the shipped `budget_s` values total 22,200 s |
| `make train-ai-route` | teacher data, behaviour cloning, one DAgger round, export of `route_mlp_h64_v1` into `models/ai_transpiler/route` | `AI_ROUTE_BC_BUDGET_S` (1,800 s) per round |
| `make train-ai-smoke` | about 5 minutes into `build/ai_train/smoke`: a 4-qubit synthesis model that learns and beats greedy, router BC, export and `models_verify`; never writes `models/` | |
| `make train-ai-route-ppo INIT=<router dir> BUDGET_S=<s>` | optional router PPO fine-tune (it degraded the shipped router and is not part of `train-ai`) | `BUDGET_S` |
| `make models-verify` | sha256 of every `weights.bin` against its INDEX.json (`QUMO_MODEL_DIR` overrides `models/ai_transpiler`); flags weights not listed in any registry | milliseconds |
| `make test-rl`, `make test-rl-slow` | the fast and the slow RL test files: core `rl/tests/test_rl_*.mojo` (with `make check-independence`) and bridge `rl/qumo_bridge/tests/test_bridge_*.mojo` | see "Tests" |
| `make check-pure-mojo` | the source, environment, binary and tree scans of acceptance A8 (`CHECK_PURE_SMOKE=1` also runs the smoke in a clean environment) | seconds |
| `make bench-rl` | the performance gates G1-G10 of MOJO_PORT.md section 15, one JSON line each | a few minutes, idle machine |

Make variables: `AI_BUILD` (`build/ai_train`), `AI_CORES` (4), `AI_LANES` (1: lanes share the same pool threads and each
runs about 0.69x slower), `AI_SYNTH_BUDGET_S`, `AI_ROUTE_BC_BUDGET_S`, `AI_ROUTE_WORK` (`$(AI_BUILD)/route`),
`AI_ROUTE_ID`. `train-ai` writes into `models/ai_transpiler` only when you run it; CI never does.

## Binaries

All are built by the pattern rule `$(AI_BUILD)/bin/%`: from the QUMO root `mojo build -I . -I rl -o build/ai_train/bin/<name> <main>.mojo`; in `rl/Makefile` the core tools build with `-I .` alone and the bridge tools with `-I $(QUMO_ROOT) -I .`.

| binary | usage |
|---|---|
| `train_synth` | `--kind perm\|lf\|clifford\|pauli --topology SPEC --out RUN (--iters N \| --budget-s S) [--cores 4] [--seed 1] [--emb 512] [--common 256[,..]] [--episodes 1024] [--diff-max 256] [--init DIR\|auto] [--start-diff D] [--start-iteration N] [--resume] [--checkpoint-every 50] [--eval-episodes 200] [--sampled-eval-every 5] [--fidelity faithful\|fixed] [--lr 3e-4] [--quiet]`; exactly one of `--iters` and `--budget-s` |
| `export_synth` | `RUN [--which best\|last] --id ID --out ROOT [--dtype f16\|f32] [--eval-json F] [--default]` |
| `eval_synth` | `RUN --id ID --root ROOT --work DIR [--instances 100] [--which best\|last] [--dtype f16\|f32] [--keep-rule default] [--force-keep] [--default]`: exports to staging, runs the block evaluation, writes `eval_blocks.json` / `eval_final.json` into RUN, applies the keep rule (`win_or_tie >= 0.6 and cost_ratio < 1.0 and 0 verification failures`, beam-8 policy against QUMO's greedy synthesizer) and publishes into ROOT only when kept |
| `eval_blocks` | the block-level evaluation alone, JSON on stdout |
| `train_all` | `--campaign FILE [--lane K/L] [--only id,id] [--root ROOT] [--work DIR] [--budget-cap-s 28800] [--cores 4] [--bin DIR] [--no-eval] [--instances N] [--dry-run]` |
| `route_tool` | `gen` (teacher / DAgger data), `eval`, `episode`, and `gen-jobs SEED N MAX_N SCALE OUT [--topologies train\|eval\|name,..] [--families ...] [--min-frac F]`, `eval-sets OUT_DIR`, `summarize EVAL.jsonl [JOBS.json] [--base sabre_o2] [--md]` |
| `train_route` | `--data d0.bin [d1.bin ...] --out RUN [--hidden 64] [--tau 0.5] [--epochs 30] [--seed 0] [--lr 2e-3] [--batch 512] [--init DIR] [--budget-s 1800] [--threads N] [--quiet]` |
| `export_route` | `RUN --root ROOT --id ID [--default] [--episodes 2] [--meta FILE]` |
| `train_route_ppo` | `--init DIR --out RUN (--budget-s S \| --iters N) [--workers 4] [--seed 0] [...]` (optional) |
| `models_verify` | no arguments; `QUMO_MODEL_DIR`; exits 1 when a file fails |
| `make_fixtures` | regenerates `tests/fixtures/ai/policy/**` from the Mojo `Rng` (the committed fixtures stay frozen) |

Errors use the codes `QM-E640` to `QM-E649` (docs/ERRORS.md).

## Synthesis models (`synth/`)

`train_synth` runs PPO with a curriculum on scramble difficulty (promotion at deterministic success >= 0.85, up to
`--diff-max`) on a `BasicPolicy`-shaped network (sparse embedding 512, one common layer of 256, action and value
heads) with symmetry augmentation. One iteration collects 1,024 episodes, evaluates (deterministic every iteration,
10-search sampled every `--sampled-eval-every`, default 5) and updates for 10 full-batch epochs.

**Run directory.** `RUN/run.json` (immutable configuration), `RUN/best/` and `RUN/last/` (f32 model containers that
load with `load_model_file`), `RUN/state.json` (a human-readable mirror), `RUN/log.jsonl` (one object per
iteration), `RUN/summary.json` (`finished` is true only after the last checkpoint). Unlike the first design
(`state.json` + `adam.bin` beside the checkpoint), the Adam moments and the training counters live *inside* `last/`
(tensors `adam.m`, `adam.v` and the manifest member `train_state`), so weights, optimizer and counters commit as one
atomic directory swap and a resumed run is bit-identical to an uninterrupted one. `state.json` is never read back.

**Resume after a crash.** Every checkpoint directory is staged in `<dir>.tmp`, read back, then swapped in with two
renames; `recover_dir` repairs a kill between the renames. Re-run the same command with `--resume` (a run directory
that has no `run.json` starts fresh; `--iters N` is the total for the run directory, so the iterations already done
count). Result-determining options repeated on `--resume` must equal the stored ones (QM-E640).

**Export and evaluation.** `export_synth` stages the model in `<root>/<id>.tmp`, runs three independent checks before
anything under the root changes (the training network on the stored tensors, the inference policy on the written
directory and a naive Float64 forward must agree for every symmetry and for the ensemble; Pauli models also run
`check_pauli_ensemble`), writes the f16 container with six test vectors and two rollouts taken from the stored weights,
replays them, and rebuilds `INDEX.json` last. A weight outside f16 range is QM-E645 (export with `--dtype f32`).

**Campaign.** `rl/drivers/synth/campaign.json` (format `qumo.campaign.v0`) has one entry per model with its stages
(`budget_s` or `iters`), `cores`, `seed`, architecture and `instances`; the five models that were dropped from the
shipped set (clifford, `lf_line7`, `pauli_complete5`, `pauli_line6`, `perm_line8`) are disabled entries whose
topologies are inferred from their names. `train_all` skips entries that are `done` and verified, resumes `running`
ones, chains stages (`pauli_line4` has three, each warm-started from the previous run and continuing its global
iteration counter), writes `<work>/<id>/status.json` and `<work>/lane<K>.log`, and exits non-zero when an enabled entry
failed, errored or missed the keep rule (QM-E649). Two lanes writing the same `--root` can race on `INDEX.json`, so
`AI_LANES` defaults to 1.

The ledger (budgets, wall times, final curriculum difficulty, evaluation and keep / drop decision per model) of the
shipped models is in docs/AI_TRANSPILER.md ("Training budgets and retraining").

## Router (`route/`)

See `rl/drivers/route/README.md`: QUMO's routing environment (`qumo/compiler/ai_route.mojo`), seeded teacher rollouts
(`route_tool gen`), behaviour cloning on soft teacher labels (`train_route`), DAgger, export with test vectors
(`export_route`), optional PPO fine-tuning (`train_route_ppo`). Held out of all training: every circuit of
`bench/neutral/circuits` and every synthetic seed >= 800000.

## Determinism and the thread count

For a given seed and configuration, a run is **independent of `--cores` and bit-identical across reruns** on the same
machine and binary. All randomness is the seeded `qumo.sim.rng.Rng`, derived per consumer from a seed tree
(`seed_for(run_seed, tag, a, b, c)`); parallel work is cut into fixed items (episodes, virtual workers of fixed chunks)
whose results are merged in index order, and gradient buffers are reduced in a fixed tree. `--cores` is clamped to
`parallelism_level()` and changes only the speed. This replaces the statement of the Python era that training was not
bit-reproducible.

What is not promised: (a) a wall-clock budget (`--budget-s`) decides how many iterations run, so use `--iters` when you
need exact reruns; a budgeted run is a deterministic prefix of the `--iters` run; (b) a different CPU architecture or
Mojo version can change floating-point contraction and the rounding of `exp` / `log`, after which a run diverges; (c)
the optional BLAS build; (d) hardware faults.

**Cores on the reference machine.** `parallelize` has as many pool threads as `parallelism_level()`, which is 4 on
the reference machine (4 performance cores of a 10-core M5; the 6 efficiency cores are unused). Asking for more adds
no threads, so scaling beyond 4 cores cannot be measured there, and two processes that each ask for 4 share the same 4
threads (about 0.69x GEMM throughput each), which is why `AI_LANES` is 1 by default.

## Tests

`make test-rl` runs `make check-independence`, the fast core files `rl/tests/test_rl_*.mojo` (from `rl/`, with
`-I .` only) and the fast bridge files `rl/qumo_bridge/tests/test_bridge_*.mojo` (from the QUMO root, `-I . -I rl`),
and prints the wall time of each (each stays under 30 s including compile). `make test-rl-slow` runs the statistical
and end-to-end files (`*_slow`, `test_rl_smoke`, `test_bridge_e2e`) and `test_rl_tensor` once with `-D ASSERT=all`.
`make test` runs every QUMO test and then, once, `make test-rl`.
Golden data from the original packages, produced offline once, is in `rl/tests/fixtures/` (see its `PROVENANCE.md`);
regenerating it needs the git tag `legacy-python-training`, which holds the retired Python trainers.

## Adding a model

1. Add an entry to `rl/drivers/synth/campaign.json` (id, kind, topology, stages) and run `make train-ai-synth`, or train one model
   by hand with `train_synth --out RUN` into a scratch directory (never commit raw runs; `.gitignore` excludes
   `rl/**/runs/` and checkpoints).
2. `eval_synth RUN --id <id> --root models/ai_transpiler/synth --work build/ai_train/work` evaluates against the greedy
   synthesizer, applies the keep rule and exports into `models/ai_transpiler/synth/<id>/` (rebuilding `INDEX.json`).
3. `make models-verify` and `make test` (tests/test_policy.mojo and tests/test_ai_route.mojo replay each model's
   embedded test vectors; set `QUMO_TEST_MODEL_ROOT` to replay a staging root instead).
4. Keep the whole `models/ai_transpiler` tree under 10 MB.
