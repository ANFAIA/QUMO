"""PPO fine-tuning of the router policy in the Python mirror environment.

Reward per policy decision (accumulated over the fallback steps that follow
it): -(routing overhead increase in CX units) + 0.01 * (2q gates executed),
i.e. reward = -delta(cx_units) + 0.01 * gates executed. Episodes are sampled
from circuits.py with a size curriculum (scale grows with the iteration).
Rollouts run in `--workers` processes; each episode is seeded, so a run is
reproducible for a given (seed, workers) pair.

usage: python train_ppo.py --init <model_dir> --out <run_dir> --budget_s 1500
Writes <run_dir>/weights.npz, <run_dir>/ppo_log.json.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import route_env as E  # noqa: E402
import circuits as C  # noqa: E402

_W = None


def _set_weights(w):
    global _W
    _W = w


def _policy(obs, A, rng):
    emb, b0, w1, b1, wv, bv = _W
    s = np.sign(np.asarray(obs, dtype=np.float64))
    h = np.maximum(b0 + s @ emb, 0.0)
    lg = w1 @ h + b1
    lg = lg[:A]
    p = np.exp(lg - lg.max())
    p /= p.sum()
    a = int(rng.choice(A, p=p))
    v = float(wv.reshape(-1) @ h + bv.reshape(-1)[0])
    return a, float(np.log(p[a] + 1e-12)), v


def _episode(args):
    w, job, seed = args
    _set_weights(w)
    rng = np.random.default_rng(seed)
    c, t = job
    tgt = E.RTarget(t["n"], [tuple(e) for e in t["edges"]])
    prob = E.RProblem([tuple(o) for o in c["ops"]], c["n"])
    lay = E.random_layout(tgt.N, E.Rng(seed))
    ep = E.Episode(prob, tgt, lay, seed, True)
    limit = max(4, 2 * prob.n)
    ms = E.max_steps(prob, tgt)
    traj = []  # (sign obs, A, a, logp, v, reward)
    pend = None  # (overhead, exec2q) when the last policy action was taken
    while True:
        r = ep.pre_decision(limit)
        done = r < 0 or ep.steps > ms
        if pend is not None and (done or ep.needs_policy()):
            # reward of a decision: its swap plus any fallback swaps until the next decision
            traj[-1][5] = -(ep.overhead() - pend[0]) + 0.01 * (ep.exec2q - pend[1])
            pend = None
        if done:
            break
        if ep.needs_policy():
            A = len(ep.active)
            a, lp, v = _policy(ep.obs, A, rng)
            traj.append([np.sign(np.asarray(ep.obs)).astype(np.int8), A, a, lp, v, 0.0])
            pend = (ep.overhead(), ep.exec2q)
            ep.act(ep.active[a])
        else:
            e = ep.fallback_swap()
            ep.n_fallback += 1
            if e < 0:
                if pend is not None:
                    traj[-1][5] = -(ep.overhead() - pend[0]) + 0.01 * (ep.exec2q - pend[1])
                break
            ep.act(e)
    return traj, ep.overhead(), ep.steps


def main():
    import torch
    import torch.nn as nn
    from train_route import RouterNet
    from export import read_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget_s", type=float, default=1500)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--clip", type=float, default=0.1)
    ap.add_argument("--ent", type=float, default=0.005)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    m, t = read_model(args.init)
    H = t["embeddings.weight"].shape[1]
    net = RouterNet(H)
    net.load_numpy(t["embeddings.weight"], t["embeddings.bias"], t["action.0.weight"], t["action.0.bias"],
                   t.get("value.0.weight"), t.get("value.0.bias"))
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    t0 = time.time()
    log = []
    pool = mp.get_context("fork").Pool(args.workers)
    it = 0
    rng = random.Random(args.seed)
    while time.time() - t0 < args.budget_s:
        scale = min(1.5, 0.4 + 0.02 * it)
        jobs = C.build_jobs(700000 + 1000 * args.seed + it, args.episodes, max_n=24, scale=scale)
        w = tuple(x.astype(np.float64) for x in net.numpy())
        tasks = [(w, (jobs["circuits"][j[0]], jobs["targets"][j[1]]), rng.randrange(1 << 40)) for j in jobs["jobs"]]
        res = pool.map(_episode, tasks)
        S, A_, act, lp, V, R, D = [], [], [], [], [], [], []
        ovh = []
        for traj, ov, _ in res:
            ovh.append(ov)
            for k, tr in enumerate(traj):
                S.append(tr[0]); A_.append(tr[1]); act.append(tr[2]); lp.append(tr[3]); V.append(tr[4]); R.append(tr[5])
                D.append(k == len(traj) - 1)
        if not S:
            continue
        R = np.array(R, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        D = np.array(D)
        adv = np.zeros_like(R)
        last = 0.0
        for i in range(len(R) - 1, -1, -1):
            nv = 0.0 if D[i] else V[i + 1]
            delta = R[i] + args.gamma * nv - V[i]
            last = delta + args.gamma * args.lam * (0.0 if D[i] else last)
            adv[i] = last
        ret = adv + V
        st = torch.tensor(np.array(S, dtype=np.float32))
        At = torch.tensor(np.array(A_, dtype=np.int64))
        at = torch.tensor(np.array(act, dtype=np.int64))
        lpt = torch.tensor(np.array(lp, dtype=np.float32))
        advt = torch.tensor(((adv - adv.mean()) / (adv.std() + 1e-8)).astype(np.float32))
        rett = torch.tensor(ret.astype(np.float32))
        valid = torch.arange(16)[None, :] < At[:, None]
        n = len(st)
        for _ in range(4):
            perm = torch.randperm(n)
            for b in range(0, n, 1024):
                bi = perm[b:b + 1024]
                logits, v = net(st[bi])
                logits = logits.masked_fill(~valid[bi], -1e9)
                logp = torch.log_softmax(logits, 1)
                nlp = logp.gather(1, at[bi, None]).squeeze(1)
                ratio = torch.exp(nlp - lpt[bi])
                s1 = ratio * advt[bi]
                s2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * advt[bi]
                pl = -torch.min(s1, s2).mean()
                ent = -(torch.exp(logp) * logp.masked_fill(~valid[bi], 0.0)).sum(1).mean()
                vl = ((v - rett[bi]) ** 2).mean()
                loss = pl + 0.5 * vl - args.ent * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
        log.append({"iter": it, "scale": scale, "decisions": n, "mean_overhead": float(np.mean(ovh)),
                    "mean_reward": float(R.mean()), "t": time.time() - t0})
        print(json.dumps(log[-1]), flush=True)
        it += 1
    pool.close()
    pool.join()
    os.makedirs(args.out, exist_ok=True)
    emb, b0, w1, b1, wv, bv = net.numpy()
    np.savez(os.path.join(args.out, "weights.npz"), emb=emb, b0=b0, w1=w1, b1=b1, wv=wv, bv=bv)
    with open(os.path.join(args.out, "ppo_log.json"), "w") as f:
        json.dump({"init": os.path.basename(args.init), "iterations": it, "wall_s": time.time() - t0,
                   "workers": args.workers, "episodes_per_iter": args.episodes, "log": log}, f, indent=1)


if __name__ == "__main__":
    main()
