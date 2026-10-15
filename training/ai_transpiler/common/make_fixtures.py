"""Generate the tiny deterministic fixture models used by tests/test_policy.mojo.

    python make_fixtures.py [--out ../../../tests/fixtures/ai/policy]

Random (seeded) weights, small layers (emb 16, common [16]), so the Mojo tests
do not depend on trained models. Layout: <out>/synth/<id>/{manifest.json,
weights.bin} + <out>/synth/INDEX.json (a model root usable as model_dir=).
Total size stays well under 50 KB. Only numpy is required.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import refenv  # noqa: E402
from export import write_model, update_index  # noqa: E402
from symmetry import twists  # noqa: E402
from topology import parse_topology  # noqa: E402

FIXTURES = [
    # id, kind, topology, basis, dtype, max_rot
    ("fx_perm_line3", "permutation", "line:3", ("SWAP",), "f32", 0),
    ("fx_perm_star4", "permutation", "star:4", ("SWAP",), "f16", 0),
    ("fx_lf_line3", "linear_function", "line:3", ("CX",), "f16", 0),
    ("fx_cliff_line2", "clifford", "line:2", ("H", "S", "CX"), "f16", 0),
    ("fx_pauli_line3", "pauli_network", "line:3", ("H", "S", "CX"), "f16", 2),
]


def actions_for(kind, edges, n, basis):
    acts = []
    for g in basis:
        if g in ("H", "S", "SDG", "SX", "SXDG"):
            acts += [(g, q, -1) for q in range(n)]
        elif g == "SWAP":
            acts += [(g, a, b) for a, b in edges]
        else:  # CX / CZ on both directions, sorted like qiskit-gym
            acts += [(g, a, b) for a, b in sorted(edges + [(b, a) for a, b in edges])]
    return acts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "..", "..", "tests", "fixtures", "ai", "policy"))
    a = ap.parse_args()
    root = os.path.join(a.out, "synth")
    os.makedirs(root, exist_ok=True)
    for k, (mid, kind, topo, basis, dtype, max_rot) in enumerate(FIXTURES):
        n, edges = parse_topology(topo)
        acts = actions_for(kind, edges, n, basis)
        if kind in ("permutation", "linear_function"):
            obs = n * n
        elif kind == "clifford":
            obs = 4 * n * n
        else:
            obs = 2 * n * (2 * n + max_rot)
        rng = np.random.default_rng(1000 + k)
        E, Cm, A = 16, 16, len(acts)
        tensors = [
            ("embeddings.weight", rng.normal(0, 0.5, (obs, E))),
            ("embeddings.bias", rng.normal(0, 0.1, E)),
            ("common.0.weight", rng.normal(0, 0.4, (Cm, E))),
            ("common.0.bias", rng.normal(0, 0.1, Cm)),
            ("action.0.weight", rng.normal(0, 0.6, (A, Cm))),
            ("action.0.bias", rng.normal(0, 0.1, A)),
            ("value.0.weight", rng.normal(0, 0.3, (1, Cm))),
            ("value.0.bias", rng.normal(0, 0.1, 1)),
        ]
        op, apm = twists(kind, n, acts, edges, max_rot)
        manifest = {
            "format": "qumo.rlsynth.v0", "id": mid, "kind": kind, "num_qubits": n,
            "edges": [list(e) for e in edges], "actions": [list(x) for x in acts],
            "obs": {"encoding": "sparse_index", "size": obs, "layout": "fixture", "max_rotations": max_rot},
            "max_depth": 40,
            "arch": {"type": "basic_mlp", "emb": E, "common": [Cm], "act": "relu", "first_layer_transposed": True},
            "symmetries": {"obs_perms": op, "act_perms": apm},
            "training": {"producer": "random fixture (tests only)", "algo": "none", "seed": 1000 + k,
                         "budget_s": 0, "wall_s": 0, "cores": 0, "final_difficulty": 0, "eval": {}},
            "license": "Apache-2.0",
        }
        mdir = os.path.join(root, mid)
        m, dec = write_model(mdir, manifest, tensors, dtype=dtype)
        pol = refenv.Policy(dec, op, apm)
        env = refenv.Env(kind, n, acts, max_rot)
        r2 = np.random.default_rng(7 + k)
        tvs, rolls = [], []
        for i in range(4):
            st = _start(kind, env, n, r2, 2 + 3 * i)
            o = env.observe(st)
            l, v = pol.raw(o)
            le, ve = pol.ensemble(o)
            tvs.append({"obs": o, "logits": [float(x) for x in l], "value": v,
                        "logits_ensemble": [float(x) for x in le], "value_ensemble": ve})
        for i in range(2):
            st = _start(kind, env, n, r2, 3 + 2 * i)
            acts_r, ok = refenv.greedy_trace(env, pol, st, 40)
            rolls.append({"input": st, "R": env.R if kind == "pauli_network" else 0,
                          "actions": acts_r, "solved": ok})
        m["test_vectors"] = tvs
        m["rollouts"] = rolls
        with open(os.path.join(mdir, "manifest.json"), "w") as f:
            json.dump(m, f, separators=(",", ":"))
    # corrupted copies (outside synth/, never indexed): flipped weight byte, bad format
    import shutil
    src = os.path.join(root, "fx_perm_line3")
    for name in ("bad_checksum", "bad_format"):
        d = os.path.join(a.out, name)
        shutil.rmtree(d, ignore_errors=True)
        shutil.copytree(src, d)
    wp = os.path.join(a.out, "bad_checksum", "weights.bin")
    raw = bytearray(open(wp, "rb").read())
    raw[100] ^= 0x01
    open(wp, "wb").write(bytes(raw))
    mp = os.path.join(a.out, "bad_format", "manifest.json")
    mm = json.load(open(mp))
    mm["format"] = "qumo.rlsynth.v9"
    json.dump(mm, open(mp, "w"), separators=(",", ":"))
    ents = update_index(root)
    tot = sum(e["bytes"] for e in ents)
    print("fixtures:", [e["id"] for e in ents], "bytes", tot)


def _start(kind, env, n, rng, depth):
    if kind == "permutation":
        st = list(range(n))
    elif kind == "linear_function":
        st = [1 << i for i in range(n)]
    elif kind == "clifford":
        st = [1 << i for i in range(2 * n)]
    else:
        R = int(rng.integers(1, 4))
        rots = []
        for _ in range(R):
            x = z = 0
            while (x | z) == 0:
                x = int(rng.integers(0, 1 << n))
                z = int(rng.integers(0, 1 << n))
            rots.append((x, z, int(rng.integers(0, 2))))
        tab = [1 << i for i in range(2 * n)]
        cenv = refenv.Env("clifford", n, env.actions)
        for _ in range(depth):
            tab = cenv.step(tab, int(rng.integers(len(env.actions))))
        st = env.pauli_start(tab, rots)
        env.clean(st, None)
        return st
    for _ in range(depth):
        st = env.step(st, int(rng.integers(len(env.actions))))
    return st


if __name__ == "__main__":
    main()
