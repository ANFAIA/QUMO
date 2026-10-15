"""Held-out evaluation job sets for the router (never used in training).

  synth   per topology family, 8 jobs of each circuit family mix at seeds >= 910000
          (training uses seeds < 800000), <= 20 qubits
  bench   bench/neutral/circuits (QASMBench / Feynman / generated benchmark circuits,
          fully held out of training) with <= 14 qubits, each on 3 of the 10
          topology families (deterministic choice by name)

usage: python eval_sets.py <out_dir>
"""
from __future__ import annotations

import json
import os
import re
import sys
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import circuits as C  # noqa: E402

EVAL_TOPOLOGIES = [("grid", (4, 4)), ("heavy_hex", 19), ("diag", 16), ("octagonal", 16), ("line", 16),
                   ("ring", 16), ("tree", 15), ("torus", 16), ("atom2", 16), ("ionring", 16)]


def targets():
    out = []
    for k, s in EVAL_TOPOLOGIES:
        nm, n, e = C.topology(k, s)
        out.append({"name": nm, "n": n, "edges": [list(x) for x in e], "family": C.FAMILY_OF[k]})
    return out


def synth_set():
    tg = targets()
    jobs = {"targets": tg, "circuits": [], "jobs": []}
    for ti, t in enumerate(tg):
        sub = C.build_jobs(910000 + ti, 8, topologies=[EVAL_TOPOLOGIES[ti]], max_n=min(20, t["n"]), scale=1.0)
        for j in sub["jobs"]:
            c = sub["circuits"][j[0]]
            jobs["circuits"].append(c)
            jobs["jobs"].append([len(jobs["circuits"]) - 1, ti, j[2]])
    return jobs


def bench_set(repo):
    tg = targets()
    d = os.path.join(repo, "bench", "neutral", "circuits")
    jobs = {"targets": tg, "circuits": [], "jobs": []}
    for f in sorted(os.listdir(d)):
        if not f.endswith(".qasm"):
            continue
        txt = open(os.path.join(d, f)).read()
        n = sum(int(m) for m in re.findall(r"qreg\s+\w+\[(\d+)\]", txt))
        if n < 3 or n > 14:
            continue
        jobs["circuits"].append({"name": f[:-5], "qasm": os.path.join("bench", "neutral", "circuits", f), "n": n})
        ci = len(jobs["circuits"]) - 1
        h = zlib.crc32(f.encode())
        for k in range(3):
            ti = (h + 3 * k) % len(tg)
            jobs["jobs"].append([ci, ti, 7])
    return jobs


if __name__ == "__main__":
    out = sys.argv[1]
    repo = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
    s = synth_set()
    b = bench_set(repo)
    json.dump(s, open(os.path.join(out, "eval_synth.json"), "w"))
    json.dump(b, open(os.path.join(out, "eval_bench.json"), "w"))
    print("synth jobs", len(s["jobs"]), "bench circuits", len(b["circuits"]), "bench jobs", len(b["jobs"]))
