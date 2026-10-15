"""Export a trained router policy as a `qumo.rlroute.v0` model directory.

Container (shared with front 'policy', training/ai_transpiler/common/export.py):
manifest.json + weights.bin (f16 matrices, f32 biases, 64-byte aligned,
FNV-1a 64 in the manifest, sha256 in models/ai_transpiler/route/INDEX.json).

Tensors: embeddings.weight [128, H] (row i = embedding of observation
feature i, i.e. the transposed first Linear layer), embeddings.bias [H],
action.0.weight [16, H], action.0.bias [16], value.0.weight [1, H],
value.0.bias [1] (PPO critic; unused at inference).

Test vectors embedded in the manifest (computed from the *stored* weights):
  obs_logits: observation -> 16 logits (float64 reference)
  episodes:   full routed episodes (circuit ops, coupling edges, initial
              layout, seed) -> action trace, overhead, final layout; the
              Mojo test suite replays them for bit-exact parity.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "common"))
from export import write_model, update_index  # noqa: E402
import route_env as E  # noqa: E402


def tensors_from_numpy(emb, b0, w1, b1, wv=None, bv=None):
    """emb [128, H] (feature-major), b0 [H], w1 [16, H], b1 [16]."""
    H = emb.shape[1]
    t = [("embeddings.weight", np.asarray(emb, np.float32)),
         ("embeddings.bias", np.asarray(b0, np.float32)),
         ("action.0.weight", np.asarray(w1, np.float32)),
         ("action.0.bias", np.asarray(b1, np.float32))]
    if wv is not None:
        t.append(("value.0.weight", np.asarray(wv, np.float32).reshape(1, H)))
        t.append(("value.0.bias", np.asarray(bv, np.float32).reshape(1)))
    return t


def episode_vector(model, ops, n, tgt_n, edges, layout, seed):
    tgt = E.RTarget(tgt_n, edges)
    prob = E.RProblem([tuple(o) for o in ops], n)
    ep = E.Episode(prob, tgt, layout, seed, True)
    ep.run(model, E.MODE_POLICY)
    return {"circuit": {"n": n, "ops": [list(o) for o in ops]},
            "target": {"n": tgt_n, "edges": [list(e) for e in edges]},
            "layout": list(layout), "seed": int(seed),
            "actions": list(ep.actions), "overhead": ep.overhead(),
            "final": list(ep.l2p), "fallback": ep.n_fallback, "failed": ep.failed}


def export_router(out_root, model_id, emb, b0, w1, b1, wv=None, bv=None, training=None,
                  episodes=None, n_obs_vectors=6, default=True, seed=0):
    """Write <out_root>/<model_id>/ and rebuild <out_root>/INDEX.json.

    episodes: list of (ops, n, tgt_n, edges, layout, seed) to embed as parity
    vectors (computed with the stored, f16-rounded weights)."""
    H = int(np.asarray(emb).shape[1])
    manifest = {
        "format": "qumo.rlroute.v0",
        "id": model_id,
        "kind": "routing",
        "num_qubits": 0,  # topology-agnostic
        "obs": {"size": E.OBS_SIZE, "layout": "route_sign_h8_k16", "encoding": "sign(obs[d + 8k])"},
        "arch": {"type": "sign_embedding_mlp", "horizon": E.HORIZON, "num_active_swaps": E.NUM_ACTIVE_SWAPS,
                 "hidden": H, "activation": "relu", "value_head": wv is not None,
                 "first_layer_transposed": True,
                 "forward": "logits = W1 relu(b0 + sum_i sign(obs_i) E[i,:]) + b1; argmax, lowest index on ties, over valid active swaps"},
        "derived_from": "qiskit-ibm-transpiler 0.18.0 rust/src/routing (Apache-2.0): algorithm and network shape only; weights trained by QUMO",
        "producer": "QUMO training/ai_transpiler/route (behaviour cloning / DAgger from a rollout teacher, optional PPO)",
        "license": "Apache-2.0",
        "training": training or {},
    }
    out_dir = os.path.join(out_root, model_id)
    tens = tensors_from_numpy(emb, b0, w1, b1, wv, bv)
    m, dec = write_model(out_dir, manifest, tens, dtype="f16")
    model = E.RouteMLP(dec["embeddings.weight"], dec["embeddings.bias"], dec["action.0.weight"], dec["action.0.bias"])
    rng = np.random.default_rng(seed)
    vecs = []
    for _ in range(n_obs_vectors):
        obs = rng.integers(-3, 4, size=E.OBS_SIZE)
        obs[rng.random(E.OBS_SIZE) < 0.5] = 0
        vecs.append({"obs": [int(x) for x in obs], "logits": [float(x) for x in model.logits(obs)]})
    eps = [episode_vector(model, *e) for e in (episodes or [])]
    m["test_vectors"] = {"obs_logits": vecs, "episodes": eps}
    m2, _ = write_model(out_dir, {k: v for k, v in m.items() if k not in ("tensors", "weights_file", "weights_bytes", "fnv1a64", "sha256")},
                        tens, dtype="f16")
    entries = update_index(out_root)
    for e in entries:
        e["default"] = (e["id"] == model_id) if default else bool(e.get("default", False))
    with open(os.path.join(out_root, "INDEX.json"), "w") as f:
        json.dump(entries, f, indent=1)
    return m2, model
