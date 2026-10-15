"""Python <-> Mojo parity check of the routing environment.

Exports a random-weight router model to a scratch directory, builds a few
jobs (mixed circuit families, measurements and barriers, several
topologies), replays one episode per job in route_env.py and in Mojo
(`route_tool.mojo episode`), and compares action traces, overheads and final
layouts (policy and heuristic modes).

usage: python parity.py <scratch_dir> <route_tool_binary>
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import route_env as E  # noqa: E402
import circuits as C  # noqa: E402
from export_route import export_router  # noqa: E402


def main():
    scratch, tool = sys.argv[1], sys.argv[2]
    os.makedirs(scratch, exist_ok=True)
    rng = np.random.default_rng(3)
    H = 32
    emb = rng.normal(0, 0.5, (E.OBS_SIZE, H))
    b0 = rng.normal(0, 0.1, H)
    w1 = rng.normal(0, 0.5, (E.NUM_ACTIVE_SWAPS, H))
    b1 = rng.normal(0, 0.1, E.NUM_ACTIVE_SWAPS)
    root = os.path.join(scratch, "parity_models")
    _, model = export_router(root, "parity_random_h32", emb, b0, w1, b1, training={"note": "random weights (parity test)"})
    mdir = os.path.join(root, "parity_random_h32")
    r = random.Random(5)
    jobs = C.build_jobs(9, 14, max_n=16)
    # add measure/barrier variety
    for c in jobs["circuits"][:4]:
        ops = c["ops"]
        ops.insert(len(ops) // 2, ["barrier", -1, -1])
        ops.append(["measure", 0, -1])
    for j in jobs["jobs"]:
        N = jobs["targets"][j[1]]["n"]
        lay = list(range(N))
        r.shuffle(lay)
        j[2] = r.randrange(1 << 40)
        j.append(lay)
    jp = os.path.join(scratch, "parity_jobs.json")
    with open(jp, "w") as f:
        json.dump(jobs, f)
    ok = True
    for mode, marg in (("policy", mdir), ("heuristic", "-")):
        outp = os.path.join(scratch, f"parity_{mode}.json")
        subprocess.run([tool, "episode", jp, marg, outp], check=True, capture_output=True)
        with open(outp) as f:
            mojo = json.load(f)
        for k, j in enumerate(jobs["jobs"]):
            c = jobs["circuits"][j[0]]
            t = jobs["targets"][j[1]]
            tgt = E.RTarget(t["n"], [tuple(e) for e in t["edges"]])
            prob = E.RProblem([tuple(o) for o in c["ops"]], c["n"])
            ep = E.Episode(prob, tgt, j[3], j[2], True)
            ep.run(model if mode == "policy" else None, E.MODE_POLICY if mode == "policy" else E.MODE_HEURISTIC)
            m = mojo[k]
            same = m["actions"] == ep.actions and m["overhead"] == ep.overhead() and m["final"] == ep.l2p
            if not same:
                ok = False
                first = next((i for i, (a, b) in enumerate(zip(m["actions"], ep.actions)) if a != b), None)
                print(f"MISMATCH {mode} job {k} {c['name']} on {t['name']}: mojo {len(m['actions'])} acts ovh {m['overhead']} "
                      f"py {len(ep.actions)} acts ovh {ep.overhead()} first diff at {first}")
            else:
                print(f"ok {mode} job {k} {c['name']} on {t['name']}: {len(ep.actions)} swaps, overhead {ep.overhead()}, fallback {ep.n_fallback}")
    print("PARITY", "OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
