"""Export a train_route.py run (weights.npz) as a qumo.rlroute.v0 model.

usage: python export_npz.py <run_dir> <models_root> <model_id> [--default] [--episodes N]
Embeds obs->logits vectors and N full parity episodes (held-out seeds)."""
from __future__ import annotations

import json
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import circuits as C  # noqa: E402
from export_route import export_router  # noqa: E402


def main():
    run, root, mid = sys.argv[1], sys.argv[2], sys.argv[3]
    default = "--default" in sys.argv
    neps = 2
    if "--episodes" in sys.argv:
        neps = int(sys.argv[sys.argv.index("--episodes") + 1])
    w = np.load(os.path.join(run, "weights.npz"))
    training = {}
    lp = os.path.join(run, "train_log.json")
    if os.path.exists(lp):
        with open(lp) as f:
            tl = json.load(f)
        tl.pop("log", None)
        training = tl
    extra = os.path.join(run, "training_meta.json")
    if os.path.exists(extra):
        with open(extra) as f:
            training.update(json.load(f))
    jobs = C.build_jobs(424242, neps, topologies=[("grid", (3, 3)), ("heavy_hex", 19)], max_n=9)
    r = random.Random(7)
    eps = []
    for j in jobs["jobs"]:
        c = jobs["circuits"][j[0]]
        t = jobs["targets"][j[1]]
        lay = list(range(t["n"]))
        r.shuffle(lay)
        eps.append((c["ops"], c["n"], t["n"], [tuple(e) for e in t["edges"]], lay, r.randrange(1 << 40)))
    m, _ = export_router(root, mid, w["emb"], w["b0"], w["w1"], w["b1"], w["wv"], w["bv"], training=training,
                         episodes=eps, default=default)
    print("exported", mid, "->", os.path.join(root, mid), "bytes", m["weights_bytes"])


if __name__ == "__main__":
    main()
