# qrl: native Mojo RL for quantum-circuit synthesis and routing policies

`rl/` is an independent package that lives in the QUMO repository. It trains the learned policies of QUMO's AI
transpiler (synthesis of permutation, linear-function, Clifford and Pauli-network blocks; qubit routing), but its core
does not need QUMO: nothing under `qrl/` imports `qumo.*`, it builds with `-I rl` alone, and it runs no Python, Rust,
torch or numpy. QUMO is one consumer of the models it writes. Whether `rl/` later becomes its own repository is open;
the layout below already allows it.

```
rl/
  qrl/                 the core package (import qrl.*): imports nothing from QUMO
  drivers/             QUMO-free tool mains: synth/ (train_synth, export_synth, train_all, campaign.json),
                       route/ (train_route), common/ (models_verify, make_fixtures)
  tests/               core tests test_rl_*.mojo and their golden fixtures (tests/fixtures, see PROVENANCE.md)
  bench/               performance gates (bench_rl.mojo, bench_rl_gemm.mojo)
  qumo_bridge/         the ONLY code that imports qumo.*: the QUMO glue, its tool mains (drivers/) and tests (tests/)
  docs/                MOJO_PORT.md (design of record), TRAINING.md (targets, binaries, run layout, budgets)
  spec/                the written specifications the port was built from (archive, Markdown)
  Makefile             build / test / train entry points (run from rl/)
  check_independence.sh, check_pure_mojo.sh, smoke.sh
```

## Using it standalone

Requirements: the Mojo toolchain QUMO pins (Mojo 1.2.0, MAX 26.7.0; `parallelize` comes from `max.algorithm`). From
`rl/`:

```bash
make MOJO=mojo test               # check-independence + the fast core tests (mojo run -I . tests/test_rl_*.mojo)
make MOJO=mojo test-slow          # statistical tests, the driver smoke test, test_rl_tensor with -D ASSERT=all
make MOJO=mojo core-tools         # build/ai_train/bin/{train_synth,export_synth,train_all,train_route,models_verify,make_fixtures}
build/ai_train/bin/train_synth --kind lf --topology ring:4 --out runs/lf_ring4 --iters 150 --cores 4 --seed 1
build/ai_train/bin/export_synth runs/lf_ring4 --which best --id lf_ring4 --out models/synth --default
QUMO_MODEL_DIR=models build/ai_train/bin/models_verify
```

In code, `-I rl` (or `-I .` inside `rl/`) and `from qrl.synth_train import run_synth_train`,
`from qrl.ppo import Trainer`, `from qrl.model import load_model_file, MLPPolicy`, and so on (`qrl/__init__.mojo`
lists the modules). Tests read their fixtures from `tests/fixtures` (`QRL_FIXTURES` overrides it). A few core tests
also check a root of exported models, by default the one QUMO ships at `../models/ai_transpiler`
(`QRL_MODEL_ROOT` overrides it); without one they print `SKIP ...` and pass.

What needs QUMO (the bridge, below): teacher data for the router (`route_tool`), the router export and PPO (their
episodes run in QUMO's routing environment), and the keep-rule evaluation of synthesis models against QUMO's
synthesizers (`eval_synth`, `eval_blocks`, used by `train_all` unless `--no-eval`).

## Model formats (owned by qrl)

A model is a directory with `manifest.json` and `weights.bin`; a model root holds `synth/` and `route/`, each with an
`INDEX.json` (sha256 per file). Writer: `qrl.container` (`write_model_dir`, atomic directory swap, INDEX rebuild,
`verify_models`); reader: `qrl.model` (`load_model_file`, FNV-1a 64 check, `MLPPolicy` reference forward).

* `qumo.rlsynth.v0`: kind (`permutation`, `linear_function`, `clifford`, `pauli_network`), num_qubits, edges,
  actions (gate, q0, q1), obs (size, layout, max_rotations), arch (sparse embedding, common layers, action and value
  heads), symmetries (obs / action permutations), tensors (f32 or f16, 64-byte aligned, little endian), fnv1a64,
  sha256, test_vectors (logits and values, identity and ensemble), rollouts (greedy traces), training, license.
* `qumo.rlroute.v0`: kind `routing`, obs 128 sign features (8 layers x 16 active swaps), 128 -> H -> 16 MLP (optional
  value head), test vectors and parity episodes.

The format names keep their historical `qumo.` prefix: the shipped models carry it and QUMO's runtime checks it.
Field-level details are in docs/MOJO_PORT.md (sections 4 and 9).

## The QUMO bridge (`qumo_bridge/`)

One narrow, optional interface. qrl hands QUMO plain data (flat edge lists, flat op triples, model directories); the
bridge turns them into QUMO objects and back:

| module | what it does with QUMO |
|---|---|
| `circuits` | a jobs-file circuit / target to a QUMO `Circuit` / `Target` |
| `export_route` | router export: parity episodes in `qumo.compiler.ai_route.Episode`, vectors cross-checked with `RouteModel` |
| `route_ppo` | router PPO fine-tune with episodes in QUMO's routing environment |
| `eval_blocks` | block evaluation against `qumo.compiler.block_synth` (exact, greedy, auto) and `PolicySession`, the keep rule |
| `drivers/` | `route_tool`, `export_route`, `train_route_ppo`, `eval_synth`, `eval_blocks` |

Bridge code builds with `-I <QUMO root> -I rl` and its tests run from the QUMO root. They also pin qrl's copies of
QUMO code against the originals: `test_bridge_parity` (the Rng stream, the opcodes, FNV / WL hash / f16, the
`PolicyEnv` dynamics on random walks of all four kinds, JSON formatting) and `test_bridge_policy` (qrl-exported
models, QUMO's committed fixtures and every shipped model load in `qumo.compiler.policy` with identical decoded
tensors, bit-identical logits for every symmetry and the ensemble, and identical greedy rollouts).

### Copies of QUMO code inside qrl

To import nothing from QUMO, qrl carries copies (Apache-2.0, same authors; each file says where it comes from):
`qrl/json.mojo` (qumo/ai/json.mojo), `qrl/rng.mojo` (qumo/sim/rng.mojo, bit-exact stream), `qrl/gates.mojo` (the
opcode subset of qumo/ir/gates.mojo, same numbers), `qrl/model.mojo` (the reader half of qumo/compiler/policy.mojo),
`qrl/env_core.mojo` (the environment dynamics and greedy trace of qumo/compiler/policy_synth.mojo) and
`heavy_hex_lattice_edges` in `qrl/targets.mojo` (Target.heavy_hex_edges). A change to one of the originals that
the bridge parity tests catch must be mirrored here.

## Inside QUMO

The QUMO root Makefile delegates: `make test-rl` (= `make -C rl test test-bridge`), `make test-rl-slow`
(`test-slow test-bridge-slow`), `make train-ai`, `train-ai-synth`, `train-ai-route`, `train-ai-smoke`,
`train-ai-route-ppo`, `models-verify`, `check-independence`, `bench-rl`; `make check-pure-mojo` runs
`rl/check_pure_mojo.sh`. `make test` runs QUMO's tests and then `make test-rl` once. Training writes into QUMO's
`models/ai_transpiler` only through `make train-ai*` (variable `MODELS`); the smoke and the tests never do.

## Independence check

`make check-independence` (part of `make test` here and of QUMO's `make test-rl`) runs `check_independence.sh`:
an import scan that fails on any `from qumo...` / `import qumo...` / `qumo_bridge` import under `qrl/`, `drivers/`
or `tests/`, and a compile check that builds every core tool from a scratch copy of `qrl/` and `drivers/` alone.
