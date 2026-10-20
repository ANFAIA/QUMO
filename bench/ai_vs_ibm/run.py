"""QUMO-AI vs IBM's AI transpiler (qiskit-ibm-transpiler, local mode): head-to-head.

    <venv>/bin/python bench/ai_vs_ibm/run.py --groups families,summit \
        --devices fake_torino --out bench/results/ai_vs_ibm.jsonl

Run it from the scratch venv described in bench/ai_vs_ibm/README.md (Qiskit,
qiskit-ibm-transpiler, qiskit-ibm-runtime, qiskit-aer; never the repository's
bench/.venv). Every contender compiles the same normalised OpenQASM 2 text
(circuits.py) for the same coupling map and basis (an IBM device's coupling map
from qiskit-ibm-runtime's fake backends, basis rz,sx,x,cz; no calibration data
is given to anyone), with the same seeds. Contenders run interleaved in
persistent per-family worker processes ('spawn', warmed up once), circuit by
circuit and repetition by repetition with a rotated contender order; every
record stores the 1-minute load average.

Every output is checked in the parent process:
  * ISA validity (bench/neutral/verify.check_native: cz only, on coupling edges);
  * equivalence up to the reported initial / final layouts:
      - statevector on a random product input (<= 14 logical qubits, numpy;
        <= 20 logical / 24 active qubits with Qiskit Aer),
      - Clifford originals of any width: stabiliser-tableau conjugation of every
        logical X / Z generator (Qiskit Clifford), falling back to exact
        Pauli-sum propagation (bench/neutral/verify.verify_clifford) when the
        compiled 1q gates are not individually Clifford;
  * otherwise the record is ISA-valid only ("verified_by": null) and the report
    keeps it out of the win / loss counts.

Contenders (`--contenders`, comma list or 'main' / 'all'):
  qiskit-o2, qiskit-o3          Qiskit transpile (coupling map + basis)
  ibm-ai2, ibm-ai3              generate_ai_pass_manager(level, ai level, optimize)
  ibm-route3                    the same at level 3 without AI synthesis
  ibm-full3                     AIRouting + every documented Collect*/AI*Synthesis pass
  ibm-synth3                    Qiskit O3 Sabre routing + that synthesis list
  qumo-o2, qumo-o3              qumo_py.transpile (O3 = portfolio with O2)
  qumo-ai2, qumo-ai3            qumo_py.transpile_ai (ai3 = portfolio with O3)
  qumo-o3s, qumo-ai3s           the single O3 / ai3 pipelines (no portfolio)
  qumo-route3                   O3 pipeline with learned-policy routing only
  qumo-synth3                   ai3 pipeline with plain SABRE routing (synthesis only)
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "bench", "neutral"))

MAIN = ["qiskit-o2", "qiskit-o3", "ibm-ai2", "ibm-ai3", "qumo-o2", "qumo-o3", "qumo-ai2", "qumo-ai3"]
ATTRIB = ["ibm-route3", "ibm-synth3", "ibm-full3", "qumo-o3s", "qumo-ai3s", "qumo-route3", "qumo-synth3"]
FAMILY = {c: ("qumo" if c.startswith("qumo") else "qiskit") for c in MAIN + ATTRIB}

# QUMO pipelines for a routed target with a basis (qumo/compiler/pass_manager.mojo
# preset_pipeline(3) / preset_pipeline_ai(3)); used for the attribution runs.
_R3 = "route(sabre,layout=auto,trials=40,iterations=3,swap_trials=40{})"
_AI = ",policy=auto,ai_level=3,layout_mode=optimize"
_PRE = "remove_identities,elide_swaps,pauli_simp,"
_POST = "basis,fixpoint(cancel,commute,phase_fold,eject_z,fuse1q,max=10),check"
_SYN = "fixpoint(linear_synth(direction=both,commute=1),clifford_synth,perm_synth,pauli_network,max=2),"
QUMO_SPECS = {
    "qumo-o3s": _PRE + "fixpoint(cancel,commute,phase_fold,fuse1q,max=4),kak," + _R3.format("") + ",kak," + _POST,
    "qumo-route3": _PRE + "fixpoint(cancel,commute,phase_fold,fuse1q,max=4),kak," + _R3.format(_AI) + ",kak," + _POST,
    "qumo-ai3s": _PRE + "clifford_synth,fixpoint(cancel,commute,phase_fold,fuse1q,max=4),kak," + _R3.format(_AI)
                 + ",kak," + _SYN + _POST,
    "qumo-synth3": _PRE + "clifford_synth,fixpoint(cancel,commute,phase_fold,fuse1q,max=4),kak," + _R3.format("")
                   + ",kak," + _SYN + _POST,
}


# --------------------------------------------------------------------------- targets
def device_target(name, n_circuit=None):
    """bench/neutral Target for an IBM device (coupling map of the
    qiskit-ibm-runtime fake backend, basis rz,sx,x + cz) or, for 'hh', the
    neutral harness' smallest heavy-hex lattice that fits the circuit."""
    import targets as T
    if name == "hh":
        return T.make_target("sc_heavy_hex", n_circuit)
    from qiskit_ibm_runtime import fake_provider
    cls = {"fake_torino": "FakeTorino", "fake_fez": "FakeFez", "fake_marrakesh": "FakeMarrakesh",
           "fake_sherbrooke": "FakeSherbrooke", "fake_kyiv": "FakeKyiv"}[name]
    b = getattr(fake_provider, cls)()
    edges = sorted({tuple(sorted(e)) for e in b.coupling_map.get_edges()})
    two = [g for g in ("cz", "ecr", "cx") if g in b.operation_names][0]
    return T.Target(name, "superconducting", b.num_qubits, edges, ["rz", "sx", "x"], two, T.SC,
                    f"{name} coupling map ({b.num_qubits} q), rz/sx/x + {two}")


# --------------------------------------------------------------------------- workers
def _guard_network():
    """Refuse every non-local socket connection inside the workers: IBM's local
    mode must not reach any service (models come from the local HF cache)."""
    import socket
    orig = socket.socket.connect

    def guarded(self, addr):
        host = addr[0] if isinstance(addr, tuple) else str(addr)
        if host not in ("127.0.0.1", "::1", "localhost") and not str(host).startswith("/"):
            raise RuntimeError(f"network access blocked by the benchmark harness: {addr}")
        return orig(self, addr)
    socket.socket.connect = guarded


def compile_job(contender, text, target, seed):
    import sdks
    if contender in ("qiskit-o2", "qiskit-o3"):
        return sdks.compile_qiskit(text, target, "default" if contender.endswith("2") else "max", seed)
    if contender in ("ibm-ai2", "ibm-ai3"):
        return sdks.compile_qiskit_ibm_ai(text, target, int(contender[-1]), seed, "default")
    if contender in ("ibm-route3", "ibm-synth3", "ibm-full3"):
        return sdks.compile_qiskit_ibm_ai(text, target, 3, seed, contender[4:-1])
    if contender in ("qumo-o2", "qumo-o3"):
        return sdks.compile_qumo(text, target, "default" if contender.endswith("2") else "max", seed)
    if contender in ("qumo-ai2", "qumo-ai3"):
        return sdks.compile_qumo_ai(text, target, "default" if contender.endswith("2") else "max", seed)
    if contender in QUMO_SPECS:
        import qumo_py
        from qiskit import qasm2
        os.environ.setdefault("QUMO_MODEL_DIR", os.path.join(REPO, "models", "ai_transpiler"))
        circ = qumo_py.load_qasm(text)
        nlog = circ.num_qubits()
        edges = [list(e) for e in target.edges]
        w0, c0 = time.perf_counter(), time.process_time()
        out, init, final, report = qumo_py.compile_spec_full(circ, target.num_qubits, edges, target.basis,
                                                             QUMO_SPECS[contender], seed)
        wall, cpu = time.perf_counter() - w0, time.process_time() - c0
        qc = qasm2.loads(out.to_qasm2(), custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
        return sdks.CompileOutput(qc, list(init)[:nlog], list(final)[:nlog], wall, cpu, 0.0, qumo_py.version(),
                                  {"ai": sdks.ai_stats(report)})
    raise ValueError(contender)


def env_info(family):
    import platform
    info = {"python": platform.python_version()}
    if family == "qiskit":
        import importlib.metadata as md
        for p in ("qiskit", "qiskit-ibm-transpiler", "qiskit-ibm-runtime", "qiskit-gym", "twisterl", "torch",
                  "networkx", "rustworkx", "numpy", "qiskit-aer", "huggingface-hub"):
            try:
                info[p] = md.version(p)
            except Exception:
                info[p] = None
        try:
            info["qiskit-ibm-ai-local-transpiler"] = md.version("qiskit-ibm-ai-local-transpiler")
        except Exception:
            info["qiskit-ibm-ai-local-transpiler"] = "not installed (not used by 0.18.x)"
    else:
        import qumo_py
        info["qumo"] = qumo_py.version()
        info["qumo_py"] = os.path.realpath(qumo_py.__file__)
        info["QUMO_MODEL_DIR"] = os.environ.get("QUMO_MODEL_DIR")
    return info


def worker_loop(family, contenders, warm_target, inq, outq):
    sys.path.insert(0, os.environ.get("QUMO_PY_DIR", os.path.join(REPO, "build")))
    os.environ.setdefault("QUMO_MODEL_DIR", os.path.join(REPO, "models", "ai_transpiler"))
    _guard_network()
    import warnings
    warnings.filterwarnings("ignore")
    text = ('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[4];\nh q[0];\ncx q[0],q[1];\nccx q[0],q[1],q[2];\n'
            'rz(0.3) q[2];\ncx q[2],q[3];\nswap q[0],q[3];\ncx q[1],q[3];\n')
    for c in contenders:  # pay import / model-loading / first-call costs once
        try:
            compile_job(c, text, warm_target, 1)
        except Exception as e:
            outq.put({"warmup_error": f"{c}: {type(e).__name__}: {e}"})
    outq.put({"ready": env_info(family)})
    while True:
        job = inq.get()
        if job is None:
            return
        contender, qtext, target, seed = job
        try:
            outq.put({"out": compile_job(contender, qtext, target, seed)})
        except Exception as e:
            outq.put({"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]})


class Worker:
    def __init__(self, family, contenders, warm_target):
        self.family, self.contenders, self.warm_target = family, contenders, warm_target
        self.ctx = mp.get_context("spawn")
        self.info = None
        self._start()

    def _start(self):
        self.inq, self.outq = self.ctx.Queue(), self.ctx.Queue()
        self.proc = self.ctx.Process(target=worker_loop, daemon=True,
                                     args=(self.family, self.contenders, self.warm_target, self.inq, self.outq))
        self.proc.start()
        while True:
            msg = self.outq.get(timeout=1800)
            if "warmup_error" in msg:
                print("  warm-up:", msg["warmup_error"], flush=True)
            if "ready" in msg:
                self.info = msg["ready"]
                return

    def run(self, job, timeout):
        self.inq.put(job)
        try:
            return self.outq.get(timeout=timeout)
        except Exception:
            self.proc.kill()
            self.proc.join()
            self._start()
            return {"timeout": timeout}

    def close(self):
        try:
            self.inq.put(None)
            self.proc.join(10)
        finally:
            if self.proc.is_alive():
                self.proc.kill()


# --------------------------------------------------------------------------- verification
def verify_aer(original, out, max_n=20, max_m=24, seed=7):
    """verify.verify with Qiskit Aer for 15-20 logical qubits (<= 24 active)."""
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    import verify as V
    nlog = original.num_qubits
    if nlog > max_n:
        return None
    body, initial, final = V._compress(out)
    m = body.num_qubits
    if m > max_m:
        return None
    a, b = V._prep(nlog, seed)
    sim = AerSimulator(method="statevector", precision="double")
    ref = QuantumCircuit(nlog)
    for i in range(nlog):
        ref.ry(a[i], i)
        ref.rz(b[i], i)
    ref.compose(original.remove_final_measurements(inplace=False), inplace=True)
    ref.save_statevector()
    psi = np.asarray(sim.run(ref).result().get_statevector())
    phys = QuantumCircuit(m)
    for i in range(nlog):
        phys.ry(a[i], initial[i])
        phys.rz(b[i], initial[i])
    phys.compose(body, qubits=list(range(m)), inplace=True)
    phys.save_statevector()
    phi = np.asarray(sim.run(phys).result().get_statevector())
    idx = np.arange(2 ** nlog, dtype=np.int64)
    pos = np.zeros_like(idx)
    for i in range(nlog):
        pos |= ((idx >> i) & 1) << final[i]
    exp = np.zeros(2 ** m, dtype=complex)
    exp[pos] = psi
    return float(abs(np.vdot(exp, phi)) ** 2)


def verify_relabelled(original, out, max_n=20, max_m=24, seed=7):
    """Equivalence up to an input relabelling the compiler misreported.

    Runs the compiled circuit backwards from the expected output (original
    applied to a random product input, placed by the reported FINAL layout,
    ancillas |0>), reads every physical qubit's reduced state, assigns each
    logical input state to the physical qubit that holds it, and checks the
    whole back-propagated state equals that placement exactly. Returns
    (fidelity, initial_layout_mismatch) or (None, None) when too large."""
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    import verify as V
    nlog = original.num_qubits
    if nlog > max_n:
        return None, None
    body, initial, final = V._compress(out)
    m = body.num_qubits
    if m > max_m:
        return None, None
    a, b = V._prep(nlog, seed)
    sim = AerSimulator(method="statevector", precision="double")
    ref = QuantumCircuit(nlog)
    for i in range(nlog):
        ref.ry(a[i], i)
        ref.rz(b[i], i)
    ref.compose(original.remove_final_measurements(inplace=False), inplace=True)
    ref.save_statevector()
    psi = np.asarray(sim.run(ref).result().get_statevector())
    idx = np.arange(2 ** nlog, dtype=np.int64)
    pos = np.zeros_like(idx)
    for i in range(nlog):
        pos |= ((idx >> i) & 1) << final[i]
    exp = np.zeros(2 ** m, dtype=complex)
    exp[pos] = psi
    back = QuantumCircuit(m)
    back.set_statevector(exp)
    back.compose(body.inverse(), inplace=True)
    back.save_statevector()
    chi = np.asarray(sim.run(back).result().get_statevector())
    t = chi.reshape([2] * m)  # axis k <-> qubit m-1-k
    states = []
    for i in range(nlog):  # ry(a) then rz(b) on |0>
        c, s = math.cos(a[i] / 2), math.sin(a[i] / 2)
        states.append(np.array([c * np.exp(-0.5j * b[i]), s * np.exp(0.5j * b[i])]))
    rho = []
    for q in range(m):
        ax = m - 1 - q
        v = np.moveaxis(t, ax, 0).reshape(2, -1)
        rho.append(v @ v.conj().T)
    inferred = []
    for i in range(nlog):
        scores = [float(np.real(states[i].conj() @ rho[q] @ states[i])) for q in range(m)]
        inferred.append(int(np.argmax(scores)))
    mismatch = inferred != [int(x) for x in initial]
    if len(set(inferred)) != nlog:
        return 0.0, mismatch
    want = QuantumCircuit(m)
    for i in range(nlog):
        want.ry(a[i], inferred[i])
        want.rz(b[i], inferred[i])
    want.save_statevector()
    w = np.asarray(sim.run(want).result().get_statevector())
    return float(abs(np.vdot(w, chi)) ** 2), mismatch


def verify_clifford_tableau(original, out, relabel=True):
    """Exact Clifford check by tableau conjugation: for every logical X_l / Z_l,
    C_out P(initial[l]) C_out^dag must equal C_orig P_l C_orig^dag placed by the
    final layout (signs included). None when a circuit is not built from
    Clifford gates (then verify.verify_clifford's Pauli-sum propagation runs).
    Returns (fidelity, initial_layout_mismatch); with `relabel` a failed check
    is retried with the input placement inferred from the tableau."""
    import numpy as np
    from qiskit.quantum_info import Clifford, Pauli
    try:
        co = Clifford(original.remove_final_measurements(inplace=False))
        body = out.circuit.copy()
        body.data = [i for i in body.data if i.operation.name not in ("measure", "barrier", "reset", "delay")]
        cc = Clifford(body)
    except Exception:
        return None, None
    n, w = co.num_qubits, cc.num_qubits

    def want(l, kind):
        lab = ["I"] * n
        lab[n - 1 - l] = kind
        ws = Pauli("".join(lab)).evolve(co, frame="s")
        z = np.zeros(w, dtype=bool)
        x = np.zeros(w, dtype=bool)
        for i in range(n):
            z[out.final[i]] = ws.z[i]
            x[out.final[i]] = ws.x[i]
        return Pauli((z, x, ws.phase))

    def got(p, kind):
        src = ["I"] * w
        src[w - 1 - p] = kind
        return Pauli("".join(src)).evolve(cc, frame="s")

    wants = [(want(l, "X"), want(l, "Z")) for l in range(n)]
    if all(got(out.initial[l], "X") == wx and got(out.initial[l], "Z") == wz for l, (wx, wz) in enumerate(wants)):
        return 1.0, False
    if not relabel:
        return 0.0, None
    # misreported initial layout: find the physical qubit that carries each logical generator
    images = {}
    for p in range(w):
        images[p] = (got(p, "X"), got(p, "Z"))
    inferred = []
    for wx, wz in wants:
        hit = [p for p, (gx, gz) in images.items() if gx == wx and gz == wz]
        if len(hit) != 1:
            return 0.0, True
        inferred.append(hit[0])
    return (1.0 if len(set(inferred)) == n else 0.0), True


def check(original, out, target):
    import verify as V
    rec = {"width": out.circuit.num_qubits}
    bad = V.check_native(out.circuit, target)
    rec["status"] = "invalid" if bad else "ok"
    if bad:
        rec["problems"] = bad
    rec.update(V.metrics(out.circuit, target))
    rec.pop("esp", None)
    rec.pop("duration_us", None)
    t0 = time.perf_counter()
    fid, how = None, None
    mismatch = False
    try:
        fid = V.verify(original, out)
        how = "statevector" if fid is not None else None
        if fid is None:
            fid = verify_aer(original, out)
            how = "statevector-aer" if fid is not None else None
        if fid is not None and fid < 1 - 1e-6:
            # same rule for every contender: retry with the input placement
            # inferred from the circuit (catches misreported initial layouts)
            f2, mm = verify_relabelled(original, out)
            if f2 is not None and f2 >= 1 - 1e-6:
                fid, how, mismatch = f2, how + "-relabelled", mm
        if fid is None and V.is_clifford_circuit(original.remove_final_measurements(inplace=False)):
            fid, mm = verify_clifford_tableau(original, out)
            how = "clifford-tableau" if fid is not None else None
            if mm:
                mismatch = True
                how = "clifford-tableau-relabelled"
            if fid is None:
                fid = V.verify_clifford(original, out)
                how = "clifford-pauli" if fid is not None else None
    except MemoryError:
        fid, how = None, None
    rec["initial_layout_mismatch"] = bool(mismatch)
    rec["verify_s"] = round(time.perf_counter() - t0, 3)
    rec["fidelity"] = fid
    rec["verified_by"] = how
    if fid is not None and fid < 1 - 1e-6 and rec["status"] == "ok":
        rec["status"] = "incorrect"
    return rec


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="families")
    ap.add_argument("--devices", default="fake_torino")
    ap.add_argument("--contenders", default="main")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=3, help="repetitions for circuits below --large qubits")
    ap.add_argument("--reps-large", type=int, default=1)
    ap.add_argument("--large", type=int, default=30)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--deadline-min", type=float, default=0, help="stop starting new circuits after this many minutes")
    ap.add_argument("--filter", default="")
    ap.add_argument("--max-qubits", type=int, default=1000)
    ap.add_argument("--benchpress", default=os.path.join(REPO, ".ref", "benchpress"))
    a = ap.parse_args()

    import circuits as C
    from qiskit import qasm2

    contenders = {"main": MAIN, "all": MAIN + ATTRIB}.get(a.contenders) or a.contenders.split(",")
    groups = a.groups.split(",")
    t_start = time.time()
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line)
            if not r.get("meta"):
                done.add((r["group"], r["circuit"], r["device"], r["contender"], r["rep"]))
    for device in a.devices.split(","):
        probe = device_target(device, 5)
        cap = min(a.max_qubits, probe.num_qubits if device != "hh" else 19)
        cache = os.path.join(os.path.dirname(os.path.abspath(a.out)), f".suite_{'-'.join(groups)}_{cap}.json")
        if os.path.exists(cache):  # the suite is deterministic; cache the built QASM texts
            suite = [tuple(s) for s in json.load(open(cache))]
        else:
            suite = C.build(groups, cap, benchpress=a.benchpress)
            json.dump(suite, open(cache, "w"))
        suite = [s for s in suite if a.filter in s[1]]
        suite.sort(key=lambda s: (groups.index(s[0]), qasm2.loads(s[2], custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS).num_qubits, s[1]))
        print(f"== {device}: {len(suite)} circuits, contenders {contenders}", flush=True)
        fams = {}
        for c in contenders:
            fams.setdefault(FAMILY[c], []).append(c)
        workers = {f: Worker(f, cs, probe) for f, cs in fams.items()}
        meta = {"meta": True, "device": device, "env": {f: w.info for f, w in workers.items()},
                "contenders": contenders, "groups": groups, "start": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(a.out, "a") as fh:
            fh.write(json.dumps(meta) + "\n")
            dropped = []
            for group, name, text in suite:
                original = qasm2.loads(text, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
                nq = original.num_qubits
                if a.deadline_min and (time.time() - t_start) / 60 > a.deadline_min:
                    dropped.append(name)
                    continue
                target = device_target(device, nq) if device == "hh" else probe
                reps = a.reps if nq < a.large else a.reps_large
                for rep in range(reps):
                    seed = a.seed + rep
                    k = rep % len(contenders)
                    order = contenders[k:] + contenders[:k]
                    for c in order:
                        key = (group, name, device, c, rep)
                        if key in done:
                            continue
                        load1 = os.getloadavg()[0]
                        msg = workers[FAMILY[c]].run((c, text, target, seed), a.timeout)
                        rec = {"group": group, "circuit": name, "qubits": nq, "device": device, "contender": c,
                               "rep": rep, "seed": seed, "load1": round(load1, 2)}
                        if "out" in msg:
                            out = msg["out"]
                            rec.update({"wall_s": out.wall_s, "cpu_s": out.cpu_s, "version": out.sdk_version})
                            if out.extra:
                                rec.update(out.extra)
                            try:
                                rec.update(check(original, out, target))
                            except Exception as e:
                                rec.update({"status": "check_error", "error": f"{type(e).__name__}: {e}"})
                        elif "timeout" in msg:
                            rec.update({"status": "timeout", "timeout_s": msg["timeout"]})
                        else:
                            rec.update({"status": "error", "error": msg.get("error"), "trace": msg.get("trace")})
                        fh.write(json.dumps(rec) + "\n")
                        fh.flush()
                        print(f"{group:8s} {name:34s} {nq:4d} r{rep} {c:12s} {rec['status']:9s} "
                              f"2q={rec.get('n2q')} d2q={rec.get('depth2q')} t={rec.get('wall_s', 0):.2f}s "
                              f"{rec.get('verified_by') or '-'} load={load1:.1f}", flush=True)
            if dropped:
                fh.write(json.dumps({"meta": True, "device": device, "dropped_by_deadline": dropped}) + "\n")
                print(f"dropped by deadline on {device}: {dropped}", flush=True)
        for w in workers.values():
            w.close()


if __name__ == "__main__":
    main()
