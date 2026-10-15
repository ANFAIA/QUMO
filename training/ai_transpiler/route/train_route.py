"""Train QUMO's router policy (behaviour cloning / DAgger on teacher Q values).

Data: records written by `route_tool.mojo gen` (328 bytes each): int16
obs[128], int16 A (number of active swaps), int16 behaviour action, int16 job,
int16 teacher argmin, float32 Q[16] (teacher rollout cost per active swap,
CX units; 1e9 for invalid slots).

Model (qumo.rlroute.v0): acts = b0 + sign(obs) @ E (E: [128, H]);
logits = W1 relu(acts) + b1; optional value head. Loss: cross-entropy to the
soft teacher distribution p_k ∝ exp(-(Q_k - min Q) / tau) over valid slots,
with random permutations of the 16 swap slots (obs columns d + 8k and Q
permuted together) as augmentation, since the environment shuffles the
active-swap order anyway.

usage: python train_route.py --data a.bin b.bin ... --out <dir> --hidden 64
       --epochs 30 --tau 0.5 --seed 0 [--init model_dir]
Writes <out>/weights.npz and <out>/train_log.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import route_env as E  # noqa: E402

REC = np.dtype([("obs", "<i2", (128,)), ("A", "<i2"), ("act", "<i2"), ("job", "<i2"), ("tb", "<i2"), ("Q", "<f4", (16,))])
assert REC.itemsize == 328


def load(paths):
    arrs = [np.fromfile(p, dtype=REC) for p in paths]
    return np.concatenate(arrs)


class RouterNet(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.emb = nn.Parameter(torch.randn(E.OBS_SIZE, H) * (1.0 / np.sqrt(E.OBS_SIZE)))
        self.b0 = nn.Parameter(torch.zeros(H))
        self.out = nn.Linear(H, E.NUM_ACTIVE_SWAPS)
        self.value = nn.Linear(H, 1)

    def forward(self, s):
        h = torch.relu(self.b0 + s @ self.emb)
        return self.out(h), self.value(h).squeeze(-1)

    def numpy(self):
        return (self.emb.detach().numpy().copy(), self.b0.detach().numpy().copy(),
                self.out.weight.detach().numpy().copy(), self.out.bias.detach().numpy().copy(),
                self.value.weight.detach().numpy().copy(), self.value.bias.detach().numpy().copy())

    def load_numpy(self, emb, b0, w1, b1, wv=None, bv=None):
        with torch.no_grad():
            self.emb.copy_(torch.tensor(emb, dtype=torch.float32))
            self.b0.copy_(torch.tensor(b0, dtype=torch.float32))
            self.out.weight.copy_(torch.tensor(w1, dtype=torch.float32))
            self.out.bias.copy_(torch.tensor(b1, dtype=torch.float32))
            if wv is not None:
                self.value.weight.copy_(torch.tensor(wv, dtype=torch.float32).reshape(1, -1))
                self.value.bias.copy_(torch.tensor(bv, dtype=torch.float32).reshape(1))


def tensors(d):
    s = torch.tensor(np.sign(d["obs"]).astype(np.float32))
    A = torch.tensor(d["A"].astype(np.int64))
    Q = torch.tensor(np.minimum(d["Q"], 1e6).astype(np.float32))
    return s, A, Q


def permute_slots(s, Q, valid, g):
    """Random permutation of the 16 swap slots per sample (obs columns d+8k move with k)."""
    B = s.shape[0]
    perm = torch.argsort(torch.rand(B, 16, generator=g), dim=1)  # new slot j takes old slot perm[j]
    s3 = s.view(B, 16, 8)
    s2 = torch.gather(s3, 1, perm.unsqueeze(-1).expand(B, 16, 8)).reshape(B, 128)
    Q2 = torch.gather(Q, 1, perm)
    v2 = torch.gather(valid, 1, perm)
    return s2, Q2, v2


def evaluate(net, s, A, Q):
    with torch.no_grad():
        logits, _ = net(s)
        valid = torch.arange(16).unsqueeze(0) < A.unsqueeze(1)
        logits = logits.masked_fill(~valid, -1e9)
        pick = logits.argmax(1)
        Qm = Q.masked_fill(~valid, 1e9)
        qmin = Qm.min(1).values
        regret = (Qm.gather(1, pick.unsqueeze(1)).squeeze(1) - qmin)
        agree = (Qm.gather(1, pick.unsqueeze(1)).squeeze(1) <= qmin + 1e-6).float().mean().item()
        return agree, regret.mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default="")
    ap.add_argument("--budget_s", type=float, default=1800.0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(8)
    t0 = time.time()
    d = load(args.data)
    # drop degenerate states (all-equal Q: nothing to learn)
    Qv = np.where(np.arange(16)[None, :] < d["A"][:, None], d["Q"], np.nan)
    spread = np.nanmax(Qv, axis=1) - np.nanmin(Qv, axis=1)
    d = d[spread > 1e-6]
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(d))
    nval = max(1000, len(d) // 20)
    val, tr = d[idx[:nval]], d[idx[nval:]]
    s, A, Q = tensors(tr)
    vs, vA, vQ = tensors(val)
    net = RouterNet(args.hidden)
    if args.init:
        sys.path.insert(0, os.path.join(HERE, "..", "common"))
        from export import read_model
        _, t = read_model(args.init)
        net.load_numpy(t["embeddings.weight"], t["embeddings.bias"], t["action.0.weight"], t["action.0.bias"],
                       t.get("value.0.weight"), t.get("value.0.bias"))
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    g = torch.Generator().manual_seed(args.seed)
    log = []
    N = s.shape[0]
    for ep in range(args.epochs):
        net.train()
        order = torch.randperm(N, generator=g)
        tot = 0.0
        for b in range(0, N, args.batch):
            bi = order[b:b + args.batch]
            sb, Ab, Qb = s[bi], A[bi], Q[bi]
            valid = torch.arange(16).unsqueeze(0) < Ab.unsqueeze(1)
            sb, Qb, valid = permute_slots(sb, Qb, valid, g)
            Qm = Qb.masked_fill(~valid, 1e9)
            target = torch.softmax(-(Qm - Qm.min(1, keepdim=True).values) / args.tau, dim=1)
            logits, _ = net(sb)
            logits = logits.masked_fill(~valid, -1e9)
            loss = -(target * torch.log_softmax(logits, dim=1)).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(bi)
        sched.step()
        net.eval()
        agree, regret = evaluate(net, vs, vA, vQ)
        log.append({"epoch": ep, "loss": tot / N, "val_agree": agree, "val_regret": regret, "t": time.time() - t0})
        print(json.dumps(log[-1]), flush=True)
        if time.time() - t0 > args.budget_s:
            break
    os.makedirs(args.out, exist_ok=True)
    emb, b0, w1, b1, wv, bv = net.numpy()
    np.savez(os.path.join(args.out, "weights.npz"), emb=emb, b0=b0, w1=w1, b1=b1, wv=wv, bv=bv)
    meta = {"samples": int(N), "val": int(nval), "hidden": args.hidden, "epochs_run": len(log), "tau": args.tau,
            "lr": args.lr, "seed": args.seed, "data": [os.path.basename(p) for p in args.data],
            "wall_s": time.time() - t0, "log": log}
    with open(os.path.join(args.out, "train_log.json"), "w") as f:
        json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main()
