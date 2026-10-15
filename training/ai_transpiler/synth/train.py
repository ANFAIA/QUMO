"""Time-bounded PPO training of a QUMO synthesis policy with qiskit-gym.

    python train.py --kind {perm,lf,clifford,pauli} --topology line:6 \
        --budget-s 2700 --seed 1 --cores 8 --emb 512 --common 256 --out runs/lf_line6

Environments come from qiskit-gym 0.4.1 (Apache-2.0) and the PPO loop from
twisterl 0.5.1 (Apache-2.0); the trained weights are QUMO's own. No IBM model
or model package is used. Writes into --out:

* config.json        qiskit-gym RLSynthesis config (env, policy, algorithm)
* model.safetensors  best checkpoint (highest (difficulty, success, reward))
* model_last.safetensors
* log.jsonl          one record per PPO iteration
* summary.json       iterations, wall_s, final difficulty, budget, seed, cores

Determinism: torch is seeded, but twisterl's rollout collection uses an
unseeded thread RNG, so two trainings are not bit-identical. Trained models
are therefore treated as fixed, hashed artefacts; QUMO's inference is seeded.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
from topology import parse_topology  # noqa: E402

GATESETS = {
    # kind -> (qiskit-gym env class name, basis gates, edges bidirectional?)
    "perm": ("PermutationGym", ("SWAP",), False),
    "lf": ("LinearFunctionGym", ("CX",), True),
    "clifford": ("CliffordGym", ("H", "S", "CX"), True),
    "pauli": ("PauliGym", ("H", "S", "CX"), True),
}


def coupling_list(edges, bidirectional):
    out = []
    for a, b in edges:
        out.append((a, b))
        if bidirectional:
            out.append((b, a))
    return sorted(out)


def make_env(kind, topology, max_rotations=5):
    from qiskit_gym.envs import synthesis as E
    n, edges = parse_topology(topology)
    cls_name, basis, bidir = GATESETS[kind]
    cls = getattr(E, cls_name)
    cm = coupling_list(edges, bidir)
    env = cls.from_coupling_map(cm, basis_gates=basis)
    if env.config["num_qubits"] != n:
        raise SystemExit(f"topology {topology}: env has {env.config['num_qubits']} qubits, expected {n}")
    return env, n, edges


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", required=True, choices=sorted(GATESETS))
    ap.add_argument("--topology", required=True)
    ap.add_argument("--budget-s", type=float, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cores", type=int, default=8)
    ap.add_argument("--emb", type=int, default=512)
    ap.add_argument("--common", type=str, default="256")
    ap.add_argument("--episodes", type=int, default=1024)
    ap.add_argument("--diff-max", type=int, default=256)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default="", help="optional safetensors to warm-start from (same architecture)")
    a = ap.parse_args()
    if a.cores > 8:
        raise SystemExit("at most 8 cores per training job on the M5")

    import torch
    from qiskit_gym.rl import RLSynthesis, PPOConfig, BasicPolicyConfig
    from qiskit_gym.rl.configs import EvalConfig
    from safetensors.torch import save_file, load_file

    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.cores)

    env, n, edges = make_env(a.kind, a.topology)
    evals = {
        "ppo_deterministic": EvalConfig(num_episodes=200, num_cores=a.cores),
        "ppo_10": EvalConfig(num_episodes=200, deterministic=False, num_searches=10, num_cores=a.cores),
    }
    ppo = PPOConfig(num_cores=a.cores, num_episodes=a.episodes, evals=evals, diff_max=a.diff_max)
    pol = BasicPolicyConfig(embedding_size=a.emb, common_layers=[int(x) for x in a.common.split(",")])
    rls = RLSynthesis(env, ppo, pol)
    rls.save(os.path.join(a.out, "config.json"))
    algo = rls.algorithm
    start_diff = 1
    base_name = os.path.basename(os.path.normpath(a.out))
    if not a.init and base_name.endswith("__cont"):
        # continuation run: warm-start from the sibling run without the suffix
        base = os.path.join(os.path.dirname(os.path.normpath(a.out)), base_name[: -len("__cont")])
        a.init = os.path.join(base, "model.safetensors")
        try:
            start_diff = int(json.load(open(os.path.join(base, "summary.json")))["final_difficulty"])
        except Exception:
            start_diff = 1
    if a.init:
        algo.policy.load_state_dict(load_file(a.init))
    nparams = sum(p.numel() for p in algo.policy.parameters())
    print(json.dumps({"kind": a.kind, "topology": a.topology, "obs": list(env.obs_shape()),
                      "actions": env.num_actions(), "n_sym": len(env.twists()[0]), "params": nparams}), flush=True)

    t0 = time.time()
    best = None
    it = 0
    env.difficulty = start_diff
    with open(os.path.join(a.out, "log.jsonl"), "w") as log:
        while time.time() - t0 < a.budget_s:
            (times, bench, train), tot = algo.learn_step()
            cur = (bench["difficulty"], bench["success"], bench["reward"])
            if best is None or cur >= best:
                best = cur
                save_file(algo.policy.state_dict(), os.path.join(a.out, "model.safetensors"))
            if bench["success"] >= algo.config["learning"]["diff_threshold"] and env.difficulty < a.diff_max:
                env.difficulty += 1
            rec = {"it": it, "t": round(time.time() - t0, 2), "step_s": round(tot, 3),
                   "difficulty": bench["difficulty"], "succ": bench["successes"],
                   "rew": bench["rewards"], "loss": train}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if it % 25 == 0:
                print(json.dumps({k: rec[k] for k in ("it", "t", "difficulty", "succ")}), flush=True)
            it += 1
    save_file(algo.policy.state_dict(), os.path.join(a.out, "model_last.safetensors"))
    summ = {"kind": a.kind, "topology": a.topology, "num_qubits": n, "iterations": it,
            "wall_s": round(time.time() - t0, 1), "budget_s": a.budget_s, "seed": a.seed,
            "cores": a.cores, "emb": a.emb, "common": a.common, "best": best,
            "final_difficulty": env.difficulty, "params": nparams, "init": a.init, "start_difficulty": start_diff}
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    print("DONE", json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
