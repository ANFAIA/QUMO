# Learned router training (`qumo.rlroute.v0`)

QUMO's hardware-neutral counterpart of qiskit-ibm-transpiler's AIRouting. The
algorithm (active swaps over 8 topological layers, sign-encoded 128-int
observation, 128 -> H -> 16 MLP, stochastic episodes, bidirectional layout
search) is derived from qiskit-ibm-transpiler 0.18.0 `rust/src/routing`
(Apache-2.0) and re-implemented from a written spec; the weights are trained
here, on QUMO's own topologies. No IBM code or weights are used. Inference is
`qumo/compiler/ai_route.mojo`; models live in `models/ai_transpiler/route/`.

The whole pipeline is native Mojo (`make train-ai-route`): no Python, torch or
numpy is involved in generating jobs, teacher data, training, evaluation or export.

| File | Role |
|---|---|
| `route_tool.mojo` | driver: `gen` (teacher data / DAgger), `eval` (held-out comparison with SABRE), `episode` (parity traces), `gen-jobs` (seeded job sets), `eval-sets` (held-out evaluation sets), `summarize` (geometric-mean tables) |
| `train_route.mojo` | behaviour cloning on soft teacher labels (slot-permutation augmentation), `qumo/rl/bc.mojo` |
| `export_route.mojo` | writes the model directory (shared container writer `qumo/rl/container.mojo`) with test vectors and parity episodes, rebuilds `INDEX.json` (the previous default is kept unless `--default` is given) |
| `train_route_ppo.mojo` | optional PPO fine-tune in the native routing environment (reward = -delta(cx_units) + 0.01 * gates executed); it degraded the shipped router and is not part of `make train-ai` |

The library behind these files is `qumo/rl/{targets,jobs,teacher_data,router_net,router_loss,bc,route_ppo,export_route}.mojo`:
`targets.mojo` has the 25 training and 10 evaluation topologies (grid, heavy-hex, diagonal grid, octagonal, line, ring,
tree, torus, neutral-atom r=2 grid, ion zone ring, with the edge order of the benchmark generators) and `jobs.mojo`
the seeded circuit families (rand2q, qv, qft, trotter, qaoa, sparse).

Reproduce (repo root; `make train-ai-route` runs exactly this, with the binaries in `build/ai_train/bin`):

```bash
make ai-tools
B=build/ai_train/bin; W=build/ai_train/route; mkdir -p $W/stage
$B/route_tool gen-jobs 31 1000 30 0.5 $W/jobs0.json       # round-0 jobs (seed 31); round 1 uses seed 32
ROUTE_TOOL_CAP=96 $B/route_tool gen $W/jobs0.json $W/d0.bin - 1.0 401 heuristic 16   # teacher episodes
$B/train_route --data $W/d0.bin --out $W/run0 --hidden 64 --tau 0.5 --epochs 30 --seed 0
$B/export_route $W/run0 --root $W/stage --id round0 --default
# DAgger round: gen with the exported model and beta=0.3, then retrain on all data
ROUTE_TOOL_CAP=96 $B/route_tool gen $W/jobs1.json $W/d1.bin $W/stage/round0 0.3 402 heuristic 16
$B/train_route --data $W/d0.bin $W/d1.bin --out $W/run1 --hidden 64 --tau 0.5 --epochs 30 --seed 0
$B/export_route $W/run1 --root models/ai_transpiler/route --id route_mlp_h64_v1 --default
```

`train_route` writes an f32 run directory (a valid `qumo.rlroute.v0` container plus `train_log.json`);
`export_route` turns it into the shipped f16 model with six observation test vectors and parity episodes, after
re-deriving the vectors with an independent Float64 forward pass. `--threads` changes only the speed: the weights are
bit-identical for any thread count and any rerun with the same seed. Because the random streams are QUMO's own
(xoshiro256**, not the earlier trainer's), the job sets and the trained weights differ from the shipped model;
the distribution of the job sets is checked against the earlier generator's statistics by `tests/test_rl_jobs.mojo`,
and training quality is judged by the `route_tool eval` `scored` metric and by validation agreement / regret.

The teacher scores each active swap by two seeded heuristic rollouts (common
random numbers) until 16 more two-qubit gates execute, in routing overhead CX
units (3 per SWAP, 1 per SWAP fused with a CX, minus the credit of SWAPs
merged into an adjacent 2q block, which KAK resynthesises with at most 3 CX).

Held out of all training: every circuit in `bench/neutral/circuits` (the
benchmark test set) and every synthetic seed >= 800000 (`route_tool eval-sets`
writes the held-out sets, with build seeds >= 910000).
