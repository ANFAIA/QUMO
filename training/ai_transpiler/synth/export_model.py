"""Export a trained qiskit-gym run into QUMO's model container (qumo.rlsynth.v0).

    python export_model.py RUN_DIR --id perm_line8 --out ../../../models/ai_transpiler/synth \
        [--dtype f16] [--weights model.safetensors] [--eval-json EVAL.json]

Writes <out>/<id>/{manifest.json, weights.bin} and rebuilds <out>/INDEX.json.
The manifest carries the exact action order, the observation layout, the
symmetry ensemble (obs_perms / act_perms in twisterl's convention:
obs' = obs_perms[p][i], logits_out[a] = raw[act_perms[p][a]]), test vectors
and two deterministic greedy rollouts computed with the *stored* (f16-rounded)
weights by refenv.py, plus the training record (summary.json) and evaluation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import refenv  # noqa: E402
from export import write_model, update_index, basic_mlp_tensors  # noqa: E402
from topology import parse_topology  # noqa: E402
from symmetry import automorphisms, pauli_twists, twists  # noqa: E402

KIND_OF_ENV = {"PermutationEnv": "permutation", "LinearFunctionEnv": "linear_function",
               "CliffordEnv": "clifford", "PauliNetworkEnv": "pauli_network"}
PRODUCER = "qiskit-gym 0.4.1 + twisterl 0.5.1 (Apache-2.0), trained locally by QUMO"


def gym_dense_perm(dense, perm, n, C):
    """qiskit-gym PauliEnv.apply_perm_to_obs (for the self-check)."""
    rows = 2 * n
    temp = np.zeros_like(dense)
    for i in range(n):
        temp[i] = dense[perm[i]]
        temp[n + i] = dense[n + perm[i]]
    res = temp.copy()
    for i in range(n):
        res[:, i] = temp[:, perm[i]]
        res[:, n + i] = temp[:, n + perm[i]]
    return res


def random_start(kind, env, n, rng, depth):
    """A reachable start state: a random walk of `depth` actions from the identity."""
    if kind == "permutation":
        st = list(range(n))
    elif kind == "linear_function":
        st = [1 << i for i in range(n)]
    elif kind == "clifford":
        st = [1 << i for i in range(2 * n)]
    else:
        # random Clifford walk + random rotations (weights 1..n) in the input frame
        R = int(rng.integers(1, env.max_rot + 3))
        rots = []
        for _ in range(R):
            while True:
                x = int(rng.integers(0, 1 << n))
                z = int(rng.integers(0, 1 << n))
                if (x | z) != 0:
                    break
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--id", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="f16", choices=("f16", "f32"))
    ap.add_argument("--weights", default="model.safetensors")
    ap.add_argument("--eval-json", default="")
    ap.add_argument("--max-depth", type=int, default=128)
    a = ap.parse_args()

    from safetensors.numpy import load_file

    cfg = json.load(open(os.path.join(a.run, "config.json")))
    summ = json.load(open(os.path.join(a.run, "summary.json"))) if os.path.exists(os.path.join(a.run, "summary.json")) else {}
    envcfg = cfg["env"]
    kind = KIND_OF_ENV[cfg["env_cls"].split(".")[-1]]
    n = envcfg["num_qubits"]
    actions = [(g.upper(), int(q[0]), int(q[1]) if len(q) > 1 else -1) for g, q in envcfg["gateset"]]
    edges = sorted({(min(q[0], q[1]), max(q[0], q[1])) for g, q in envcfg["gateset"] if len(q) == 2})
    max_rot = int(envcfg.get("max_rotations", 5)) if kind == "pauli_network" else 0
    if kind == "permutation" or kind == "linear_function":
        obs_size, layout = n * n, f"{n}x{n} row-major: perm i*n+state[i] / matrix bits"
    elif kind == "clifford":
        obs_size, layout = 4 * n * n, f"{2*n}x{2*n} symplectic of C^-1, rows = coords (x..., z...), cols = generators"
    else:
        obs_size, layout = 2 * n * (2 * n + max_rot), f"{2*n}x{2*n + max_rot}: tableau of C^-1 + first {max_rot} active rotation columns"

    # symmetries exactly as used in training
    if kind == "pauli_network":
        obs_perms, act_perms = pauli_twists(n, actions, max_rot, edges)
    else:
        from qiskit_gym.envs.synthesis import SYNTH_ENVS
        genv = SYNTH_ENVS[cfg["env_cls"].split(".")[-1]].from_json(envcfg)
        op, apm = genv.twists()
        obs_perms = [list(map(int, p)) for p in op]
        act_perms = [list(map(int, p)) for p in apm]
        o2, a2 = twists(kind, n, actions, edges)
        if sorted(zip(map(tuple, o2), map(tuple, a2))) != sorted(zip(map(tuple, obs_perms), map(tuple, act_perms))):
            raise SystemExit("symmetry.py disagrees with qiskit-gym twists")

    sd = load_file(os.path.join(a.run, a.weights))
    tensors = basic_mlp_tensors(sd)
    emb = tensors[0][1].shape[1]
    common = [t.shape[0] for name, t in tensors if name.startswith("common.") and name.endswith(".weight")]
    if tensors[0][1].shape[0] != obs_size:
        raise SystemExit(f"obs size mismatch {tensors[0][1].shape[0]} vs {obs_size}")

    manifest = {
        "format": "qumo.rlsynth.v0", "id": a.id, "kind": kind, "num_qubits": n,
        "edges": [list(e) for e in edges],
        "actions": [list(x) for x in actions],
        "obs": {"encoding": "sparse_index", "size": obs_size, "layout": layout, "max_rotations": max_rot},
        "max_depth": a.max_depth,
        "arch": {"type": "basic_mlp", "emb": emb, "common": common, "act": "relu", "first_layer_transposed": True},
        "symmetries": {"obs_perms": obs_perms, "act_perms": act_perms},
        "training": {"producer": PRODUCER, "algo": "PPO (twisterl)", "seed": summ.get("seed"),
                     "budget_s": summ.get("budget_s"), "wall_s": summ.get("wall_s"), "cores": summ.get("cores"),
                     "iterations": summ.get("iterations"), "final_difficulty": summ.get("final_difficulty"),
                     "topology_spec": summ.get("topology"), "eval": {}},
        "license": "Apache-2.0",
    }
    if summ.get("init"):
        bdir = os.path.dirname(summ["init"])
        try:
            bs = json.load(open(os.path.join(bdir, "summary.json")))
            manifest["training"]["continued_from"] = {"run": os.path.basename(bdir), "budget_s": bs.get("budget_s"),
                                                       "wall_s": bs.get("wall_s"), "final_difficulty": bs.get("final_difficulty")}
            manifest["training"]["total_budget_s"] = (bs.get("budget_s") or 0) + (summ.get("budget_s") or 0)
            manifest["training"]["total_wall_s"] = (bs.get("wall_s") or 0) + (summ.get("wall_s") or 0)
        except OSError:
            pass
    tmp_out = os.path.join(a.out, a.id)
    m, dec = write_model(tmp_out, manifest, tensors, dtype=a.dtype)

    # test vectors + rollouts from the stored weights
    pol = refenv.Policy(dec, obs_perms, act_perms)
    env = refenv.Env(kind, n, actions, max_rot)
    rng = np.random.default_rng(20261014)
    tvs = []
    for i in range(6):
        st = random_start(kind, env, n, rng, 3 + 4 * i)
        obs = env.observe(st)
        l, v = pol.raw(obs)
        le, ve = pol.ensemble(obs)
        tvs.append({"obs": obs, "logits": [float(x) for x in l], "value": v,
                    "logits_ensemble": [float(x) for x in le], "value_ensemble": ve})
    # Pauli ensemble self-check against qiskit-gym's internal permutation logic
    if kind == "pauli_network" and obs_perms:
        C = 2 * n + max_rot
        autos = [p for p in automorphisms(n, edges)]
        st = random_start(kind, env, n, rng, 10)
        dense = np.zeros((2 * n, C), dtype=np.uint8)
        for idx in env.observe(st):
            dense[idx // C, idx % C] = 1
        for p in range(len(obs_perms)):
            dp = gym_dense_perm(dense, autos[p], n, C)
            raw_gym, _ = pol.raw(list(np.flatnonzero(dp.reshape(-1))))
            ours, _ = pol.raw(env.observe(st), p)
            # gym: actual action A[i] gets raw[i]  <=> ours[a] = raw[act_perm[a]]
            if not np.allclose(raw_gym[act_perms[p]], ours, atol=1e-5):
                raise SystemExit(f"Pauli symmetry {p} self-check failed")
    rollouts = []
    for i in range(2):
        st = random_start(kind, env, n, rng, 6 + 6 * i)
        acts, ok = refenv.greedy_trace(env, pol, st, a.max_depth)
        rollouts.append({"input": st, "R": env.R if kind == "pauli_network" else 0,
                         "actions": acts, "solved": ok})
    m["test_vectors"] = tvs
    m["rollouts"] = rollouts
    if a.eval_json and os.path.exists(a.eval_json):
        m["training"]["eval"] = json.load(open(a.eval_json))
    with open(os.path.join(tmp_out, "manifest.json"), "w") as f:
        json.dump(m, f, separators=(",", ":"))
    update_index(a.out)
    print(json.dumps({"id": a.id, "kind": kind, "n": n, "bytes": m["weights_bytes"], "n_sym": len(obs_perms),
                      "rollout_lens": [len(r["actions"]) for r in rollouts]}))


if __name__ == "__main__":
    main()
