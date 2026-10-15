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
import math
import multiprocessing as mp  # noqa: F401
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "build"))

from qiskit import qasm2  # noqa: E402

from targets import ALL_TARGETS, make_target  # noqa: E402
from sdks import SDKS, Unsupported, warm_up  # noqa: E402
from verify import check_native, metrics, verify, verify_clifford  # noqa: E402


def make_target_ext(kind, min_qubits):
    """targets.make_target plus the stress-set king grid (square lattice with
    diagonals, cz; the interaction graph of short-range neutral-atom arrays)."""
    if kind == "king_grid":
        import targets as T
        n, e = T.grid_edges(min_qubits, math.sqrt(2.0))
        return T.Target(kind, "neutral_atom", n, e, ["rz", "rx", "ry"], "cz", T.ATOM,
                        "king graph (square lattice + diagonals), rz/rx/ry + cz")
    return make_target(kind, min_qubits)


# --------------------------------------------------------------------------- stress set
def stress_circuits(seed=2026, max_qubits=60):
    """Clifford / linear / permutation / Pauli-network stress circuits
    (deterministic). Returns [(name, qasm2_text)]. Clifford-only circuits are
    verified by stim above 14 qubits; Pauli networks stay <= 14 qubits so the
    statevector check applies."""
    import random
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import LinearFunction, PermutationGate
    from qiskit.quantum_info import random_clifford
    from qiskit.synthesis.linear import random_invertible_binary_matrix

    rng = random.Random(seed)
    out = []

    def add(name, qc):
        if qc.num_qubits <= max_qubits:
            from qiskit import transpile as qtranspile
            # expand library objects into h/s/sdg/cx/swap/rz... (no optimisation, no routing)
            flat = qtranspile(qc, basis_gates=["h", "s", "sdg", "x", "y", "z", "sx", "sxdg", "cx", "swap", "rz", "rx"],
                              optimization_level=0)
            out.append((name, qasm2.dumps(flat)))

    for n in (10, 20, 40, 60):
        add(f"stress__clifford_n{n}", random_clifford(n, seed=seed + n).to_circuit())
    for n in (10, 20, 40, 60):
        mat = random_invertible_binary_matrix(n, seed=seed + 100 + n)
        add(f"stress__linear_n{n}", LinearFunction(mat).definition)
        qc = QuantumCircuit(n)
        for _ in range(3 * n):
            a, b = rng.sample(range(n), 2)
            qc.cx(a, b)
        add(f"stress__cxrand_n{n}", qc)
    for n in (10, 20, 40, 60):
        perm = list(range(n))
        rng.shuffle(perm)
        qc = QuantumCircuit(n)
        qc.append(PermutationGate(perm), range(n))
        add(f"stress__perm_n{n}", qc)
        qc = QuantumCircuit(n)
        for _ in range(2 * n):
            a, b = rng.sample(range(n), 2)
            qc.swap(a, b)
        add(f"stress__swapnet_n{n}", qc)
    for n in (6, 10, 14):
        # Trotter step of a random 2-4-local Pauli Hamiltonian (basis change + CX ladder + rz)
        qc = QuantumCircuit(n)
        for _ in range(3 * n):
            w = rng.randint(2, 4)
            qs = rng.sample(range(n), w)
            axes = [rng.choice("XYZ") for _ in qs]
            for q, a in zip(qs, axes):
                if a == "X":
                    qc.h(q)
                elif a == "Y":
                    qc.sdg(q)
                    qc.h(q)
            for i in range(w - 1):
                qc.cx(qs[i], qs[i + 1])
            qc.rz(rng.uniform(-math.pi, math.pi), qs[-1])
            for i in reversed(range(w - 1)):
                qc.cx(qs[i], qs[i + 1])
            for q, a in zip(qs, axes):
                if a == "X":
                    qc.h(q)
                elif a == "Y":
                    qc.h(q)
                    qc.s(q)
        add(f"stress__trotter_n{n}", qc)
        # UCC-like: double excitations as 4-qubit Pauli-rotation ladders
        qc = QuantumCircuit(n)
        for i in range(n // 2):
            qc.x(i)
        for _ in range(n):
            i, j, a, b = sorted(rng.sample(range(n), 4))
            for pat in ("XXXY", "YXXX"):
                qs = [i, j, a, b]
                for q, ax in zip(qs, pat):
                    if ax == "X":
                        qc.h(q)
                    else:
                        qc.rx(math.pi / 2, q)
                for k in range(3):
                    qc.cx(qs[k], qs[k + 1])
                qc.rz(rng.uniform(-0.5, 0.5), b)
                for k in reversed(range(3)):
                    qc.cx(qs[k], qs[k + 1])
                for q, ax in zip(qs, pat):
                    if ax == "X":
                        qc.h(q)
                    else:
                        qc.rx(-math.pi / 2, q)
        add(f"stress__ucc_n{n}", qc)
    return out


def one_job(args):
    sdk, qasm_text, tkind, level, seed = args
    try:
        original = qasm2.loads(qasm_text, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
        target = make_target_ext(tkind, original.num_qubits)
        out = SDKS[sdk](qasm_text, target, level, seed)
        rec = {"status": "ok", "wall_s": out.wall_s, "cpu_s": out.cpu_s, "parse_s": out.parse_s,
               "version": out.sdk_version, "width": out.circuit.num_qubits}
        rec.update(out.extra or {})
        bad = check_native(out.circuit, target)
        if bad:
            rec["status"] = "invalid"
            rec["problems"] = bad
        rec.update(metrics(out.circuit, target))
        fid = verify(original, out)
        if fid is None:
            # above the statevector size: exact Pauli-propagation check for Clifford circuits
            fid = verify_clifford(original, out)
            if fid is not None:
                rec["verified_by"] = "clifford"
        else:
            rec["verified_by"] = "statevector"
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
    ap.add_argument("--circuits", default=os.path.join(os.path.dirname(__file__), "circuits"))
    ap.add_argument("--sdks", default="qumo,qiskit,tket,cirq")
    ap.add_argument("--targets", default="all")
    ap.add_argument("--levels", default="default,max")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-qubits", type=int, default=14)
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--filter", default="")
    ap.add_argument("--stress", action="store_true",
                    help="run the generated Clifford / linear / permutation / Pauli-network stress set "
                         "(targets default to sc_heavy_hex,sc_grid,king_grid,linear_cx; --max-qubits default 60)")
    a = ap.parse_args()
    if a.stress:
        if a.targets == "all":
            a.targets = "sc_heavy_hex,sc_grid,king_grid,linear_cx"
        if a.max_qubits == 14:
            a.max_qubits = 60
    targets = ALL_TARGETS if a.targets == "all" else a.targets.split(",")
    if a.stress:
        texts = [(nm, txt) for nm, txt in stress_circuits(max_qubits=a.max_qubits) if a.filter in nm]
    else:
        files = sorted(f for f in os.listdir(a.circuits) if f.endswith(".qasm") and a.filter in f)
        texts = [(f[:-5], open(os.path.join(a.circuits, f)).read()) for f in files]
    workers = {sdk: Worker(sdk) for sdk in a.sdks.split(",")}
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line)
            done.add((r["circuit"], r["target"], r["sdk"], r["level"]))
    with open(a.out, "a") as fh:
        for name, text in texts:
            f = name + ".qasm"
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

