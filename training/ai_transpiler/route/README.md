# Learned router training (`qumo.rlroute.v0`)

QUMO's hardware-neutral counterpart of qiskit-ibm-transpiler's AIRouting. The
algorithm (active swaps over 8 topological layers, sign-encoded 128-int
observation, 128 -> H -> 16 MLP, stochastic episodes, bidirectional layout
search) is derived from qiskit-ibm-transpiler 0.18.0 `rust/src/routing`
(Apache-2.0) and re-implemented from a written spec; the weights are trained
here, on QUMO's own topologies. No IBM code or weights are used. Inference is
`qumo/compiler/ai_route.mojo`; models live in `models/ai_transpiler/route/`.

| File | Role |
|---|---|
| `route_env.py` | bit-exact Python mirror of the Mojo environment (xoshiro `Rng`, node grouping, execution, layers + shuffles, active swaps, observation, fallback, SWAP simplification and the overhead in CX units) |
| `route_tool.mojo` | Mojo driver: `gen` (teacher data / DAgger), `eval` (held-out comparison with SABRE), `episode` (parity traces) |
| `circuits.py` | seeded circuit families (rand2q, qv, qft, trotter, qaoa, sparse) and topologies (grid, heavy-hex, diagonal grid, octagonal, line, ring, tree, torus, neutral-atom r=2 grid, ion zone ring) |
| `train_route.py` | behaviour cloning on soft teacher labels (slot-permutation augmentation) |
| `train_ppo.py` | optional PPO fine-tune in the Python mirror (reward = -delta(cx_units) + 0.01 * gates executed) |
| `export_route.py`, `export_npz.py` | write the model directory (shared container of `../common/export.py`) with test vectors and rebuild `INDEX.json` |
| `parity.py` | Python <-> Mojo episode parity on random weights |
| `eval_sets.py`, `summarize_eval.py` | held-out evaluation sets and geometric-mean tables |

Reproduce (repo root; `PY` = a venv with torch + numpy + rustworkx, e.g. the
qiskit-gym probe venv):

```bash
.venv/bin/mojo build -I . -o build/route_tool training/ai_transpiler/route/route_tool.mojo
$PY training/ai_transpiler/route/parity.py /tmp/route_parity build/route_tool        # PARITY OK
$PY -c "import sys,json; sys.path.insert(0,'training/ai_transpiler/route'); import circuits as C; \
  json.dump(C.build_jobs(31, 1000, max_n=30, scale=0.5), open('/tmp/a.json','w'))"
ROUTE_TOOL_CAP=96 build/route_tool gen /tmp/a.json /tmp/a.bin - 1.0 401 heuristic 16   # teacher episodes
$PY training/ai_transpiler/route/train_route.py --data /tmp/a.bin --out /tmp/run --hidden 64 --tau 0.5 --epochs 30
$PY training/ai_transpiler/route/export_npz.py /tmp/run models/ai_transpiler/route route_mlp_h64_v1 --default
# DAgger round: gen with the exported model and beta=0.3, then retrain on all data
```

The teacher scores each active swap by two seeded heuristic rollouts (common
random numbers) until 16 more two-qubit gates execute, in routing overhead CX
units (3 per SWAP, 1 per SWAP fused with a CX, minus the credit of SWAPs
merged into an adjacent 2q block, which KAK resynthesises with at most 3 CX).

Held out of all training: every circuit in `bench/neutral/circuits` (the
benchmark test set) and every synthetic seed >= 800000 (`eval_sets.py`).
