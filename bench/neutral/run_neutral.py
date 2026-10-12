"""Hardware-neutral compiler benchmark runner.

    python bench/neutral/run_neutral.py --circuits bench/neutral/circuits \
        --sdks qumo,qiskit,tket,cirq --targets all --levels default,max \
        --out bench/results/neutral.jsonl [--max-qubits 14] [--timeout 300]

Each (circuit, target, sdk, level) record contains: status (ok / unsupported /
invalid / incorrect / error / timeout), metrics (n2q, depth2q, depth, runs1q,
duration_us, esp), fidelity (when verified), wall/cpu compile time and parse
time. Runs are sequential so timings do not interfere.
"""
import argparse
import json
import multiprocessing as mp  # noqa: F401
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "build"))

from qiskit import qasm2  # noqa: E402

from targets import ALL_TARGETS, make_target  # noqa: E402
from sdks import SDKS, Unsupported, warm_up  # noqa: E402
from verify import check_native, metrics, verify  # noqa: E402


def one_job(args):
    sdk, qasm_text, tkind, level, seed = args
    try:
        original = qasm2.loads(qasm_text, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
        target = make_target(tkind, original.num_qubits)
        out = SDKS[sdk](qasm_text, target, level, seed)
        rec = {"status": "ok", "wall_s": out.wall_s, "cpu_s": out.cpu_s, "parse_s": out.parse_s,
               "version": out.sdk_version, "width": out.circuit.num_qubits}
        bad = check_native(out.circuit, target)
        if bad:
            rec["status"] = "invalid"
            rec["problems"] = bad
        rec.update(metrics(out.circuit, target))
        fid = verify(original, out)
        rec["fidelity"] = fid
        if fid is not None and fid < 1 - 1e-6 and rec["status"] == "ok":
            rec["status"] = "incorrect"
        return rec
    except Unsupported as e:
        return {"status": "unsupported", "reason": str(e)}
    except Exception as e:
        return {"status": "error", "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc()[-1500:]}


def worker_loop(sdk, inq, outq):
    """Persistent per-SDK worker: warm up once (imports, plugin discovery,
    JIT/first-call costs), then serve jobs. Started with 'spawn' so SDKs with
    native thread pools (e.g. Qiskit's Rust/rayon) never run in a forked child."""
    warm_up(sdk)
    outq.put("ready")
    while True:
        job = inq.get()
        if job is None:
            return
        outq.put(one_job(job))


class Worker:
    def __init__(self, sdk):
        self.sdk = sdk
        self.ctx = mp.get_context("spawn")
        self._start()

    def _start(self):
        self.inq = self.ctx.Queue()
        self.outq = self.ctx.Queue()
        self.proc = self.ctx.Process(target=worker_loop, args=(self.sdk, self.inq, self.outq), daemon=True)
        self.proc.start()
        self.outq.get(timeout=600)

    def run(self, job, timeout):
        self.inq.put(job)
        try:
            return self.outq.get(timeout=timeout)
        except Exception:
            self.proc.kill()
            self.proc.join()
            self._start()
            return {"status": "timeout", "timeout_s": timeout}

    def close(self):
        try:
            self.inq.put(None)
            self.proc.join(5)
        finally:
            if self.proc.is_alive():
                self.proc.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuits", required=True)
    ap.add_argument("--sdks", default="qumo,qiskit,tket,cirq")
    ap.add_argument("--targets", default="all")
    ap.add_argument("--levels", default="default,max")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-qubits", type=int, default=14)
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--filter", default="")
    a = ap.parse_args()
    targets = ALL_TARGETS if a.targets == "all" else a.targets.split(",")
    files = sorted(f for f in os.listdir(a.circuits) if f.endswith(".qasm") and a.filter in f)
    workers = {sdk: Worker(sdk) for sdk in a.sdks.split(",")}
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line)
            done.add((r["circuit"], r["target"], r["sdk"], r["level"]))
    with open(a.out, "a") as fh:
        for f in files:
            text = open(os.path.join(a.circuits, f)).read()
            nq = qasm2.loads(text, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS).num_qubits
            if nq > a.max_qubits:
                continue
            for t in targets:
                for level in a.levels.split(","):
                    for sdk in a.sdks.split(","):
                        key = (f[:-5], t, sdk, level)
                        if key in done:
                            continue
                        rec = workers[sdk].run((sdk, text, t, level, a.seed), a.timeout)
                        rec.update({"circuit": f[:-5], "qubits": nq, "target": t, "sdk": sdk, "level": level})
                        fh.write(json.dumps(rec) + "\n")
                        fh.flush()
                        tag = rec["status"]
                        extra = f"n2q={rec.get('n2q')} esp={rec.get('esp', 0):.3f} t={rec.get('wall_s', 0):.3f}s" if tag in ("ok", "invalid", "incorrect") else rec.get("reason", rec.get("error", ""))[:80]
                        print(f"{f[:-5]:34s} {t:28s} {level:7s} {sdk:7s} {tag:11s} {extra}", flush=True)


if __name__ == "__main__":
    main()

