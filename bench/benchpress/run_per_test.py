"""Run one Benchpress gym with a hard time limit per test.

Benchpress's own --timeout-skip-list runs each test in a helper process that can hang (it does for the TKET gym), so this
runs every test in its own pytest process and kills it after --timeout seconds. The benchmarks of all the processes are
merged into one pytest-benchmark JSON; the ids that hit the limit are listed in <out>.timeouts.txt. Each test leaves its
result in <out>.parts/, so a stopped run resumes where it was.

    bench/.venv/bin/python bench/benchpress/run_per_test.py --checkout .ref/benchpress --gym tket_gym \
        --timeout 60 --out bench/results/benchpress/tket_full.json
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkout", required=True)
    ap.add_argument("--gym", required=True)
    ap.add_argument("--timeout", type=float, default=60.0, help="seconds per test, not counting pytest start-up")
    ap.add_argument("--out", required=True)
    ap.add_argument("--startup", type=float, default=30.0, help="allowance for importing the gym, added to --timeout")
    ap.add_argument("--pytest-args", default="-p no:cacheprovider")
    a = ap.parse_args()
    base = [sys.executable, "-m", "pytest", "-q"] + a.pytest_args.split()
    cwd = os.path.join(a.checkout, "benchpress")  # the pytest.ini of Benchpress is here, so ids are relative to it
    listing = subprocess.run(base + [a.gym, "--collect-only", "-o", "addopts="], cwd=cwd, capture_output=True, text=True)
    ids = [l.strip() for l in listing.stdout.splitlines() if l.startswith(a.gym + "/") and "::" in l]
    print(f"{len(ids)} tests in {a.gym}", flush=True)
    parts = a.out + ".parts"
    os.makedirs(parts, exist_ok=True)
    t_start = time.time()
    for i, nid in enumerate(ids, 1):
        stem = os.path.join(parts, hashlib.sha1(nid.encode()).hexdigest()[:16])
        if any(os.path.exists(stem + e) for e in (".json", ".timeout", ".none")):
            continue
        tmp = stem + ".tmp"
        try:
            subprocess.run(base + [nid, f"--benchmark-json={tmp}"], cwd=cwd, capture_output=True, text=True, timeout=a.timeout + a.startup)
            if os.path.exists(tmp) and os.path.getsize(tmp) > 0:
                os.replace(tmp, stem + ".json")
            else:
                open(stem + ".none", "w").write(nid)  # skipped or failed inside pytest: no benchmark
        except subprocess.TimeoutExpired:
            open(stem + ".timeout", "w").write(nid)
        if os.path.exists(tmp):
            os.remove(tmp)
        if i % 25 == 0:
            print(f"{i}/{len(ids)}  {time.time() - t_start:.0f}s", flush=True)
    merged, head, timeouts, none = [], None, [], 0
    for f in sorted(os.listdir(parts)):
        p = os.path.join(parts, f)
        if f.endswith(".json"):
            try:
                data = json.load(open(p))
            except ValueError:
                continue
            head = head or data
            merged.extend(data.get("benchmarks", []))
        elif f.endswith(".timeout"):
            timeouts.append(open(p).read())
        elif f.endswith(".none"):
            none += 1
    head = head or {"machine_info": {}, "commit_info": {}, "version": "?"}
    head["benchmarks"] = merged
    with open(a.out, "w") as fh:
        json.dump(head, fh)
    with open(a.out + ".timeouts.txt", "w") as fh:
        fh.write("\n".join(timeouts) + ("\n" if timeouts else ""))
    print(f"done: {len(merged)} benchmarks, {len(timeouts)} timed out, {none} without a result of {len(ids)}", flush=True)


if __name__ == "__main__":
    main()
