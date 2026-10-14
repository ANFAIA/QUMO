"""GPU, CPU and QPU stages of the heterogeneous example, plus metrics.

Placement is decided here by rules, never by the model:

* GPU stage: `gpu` when this build of qumo_py has GPU support, a usable device,
  and the circuit has at least `min_qubits_auto` qubits (16 on Apple, 20 on
  NVIDIA/AMD), or with --force-gpu; otherwise the CPU, with a note saying why.
  The run never fails over placement.
* CPU stage: exact state-vector expectations (fp64) and classical solvers
  (scipy optimisers, brute-force MaxCut, SVM dual).
* QPU stage: `qumo_hardware.runner.run` on the chosen backend. The default is
  a vendor-neutral noisy local emulator; remote backends only run with
  --allow-remote --submit --yes and a cost within --max-cost, and the runner's
  own QM-E956 / QM-E957 gate stays authoritative.

Phase 1 notes (stated in every report): qumo_py exposes exact expectations on
the CPU only and no gradients yet, so GPU work uses `sample_device` (sampled
energies) and gradients are finite differences; phase 2 adds
`expectation_device` / `value_and_gradient` (TODO hooks: `exact_energy`,
`gradient`).
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np
import scipy.optimize  # noqa: F401  (imported up front so stage timings exclude the import)

import kernels as K

# --------------------------------------------------------------------------- runtime
@dataclass
class Stage:
    name: str  # gpu | cpu | qpu
    task: str
    device_requested: str
    device_used: str = ""
    method: str = ""
    precision: str = ""
    seconds: float = 0.0
    evaluations: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["seconds"] = round(self.seconds, 4)
        return d


class Runtime:
    """qumo_py module + GPU status + QPU access, shared by every stage."""

    def __init__(self, q, gpu_info: dict, qpu: str, shots_cap: int = 100_000, force_gpu: bool = False,
                 allow_remote: bool = False, submit: bool = False, yes: bool = False, max_cost: float | None = None,
                 runner=None, seed: int = 7, log: Callable[[str], None] = print):
        self.q = q
        self.gpu_info = gpu_info
        self.qpu = qpu
        self.provider_name, _, self.backend = qpu.partition(":")
        self.shots_cap = shots_cap
        self.force_gpu = force_gpu
        self.allow_remote = allow_remote
        self.submit = submit
        self.yes = yes
        self.max_cost = max_cost
        self.seed = seed
        self.log = log
        if runner is None:
            from qumo_hardware import runner as _runner
            runner = _runner
        self.runner = runner
        self.qpu_jobs: list[dict] = []
        self.payloads: list[str] = []

    # -- placement -------------------------------------------------------------------
    def place(self, num_qubits: int) -> tuple[str, str]:
        g = self.gpu_info or {}
        usable = bool(g.get("compiled")) and bool(g.get("available"))
        minq = int(g.get("min_qubits_auto") or 20)
        if usable and (num_qubits >= minq or self.force_gpu):
            note = "" if num_qubits >= minq else f"--force-gpu: {num_qubits} < {minq} qubits (CPU would be faster)"
            return "gpu", note
        if not g.get("compiled"):
            return "cpu", "GPU support not compiled into this qumo_py (make py-gpu, then PYTHONPATH=build/gpu)"
        if not g.get("available"):
            return "cpu", f"no usable GPU ({g.get('error') or 'gpu_info().available is false'})"
        return "cpu", f"below the GPU threshold ({num_qubits} < {minq} qubits): the CPU is faster here"

    # -- simulation helpers ---------------------------------------------------------------
    def sample(self, circ, shots: int, seed: int, device: str, precision: str = "auto") -> tuple[dict, str, str]:
        """qumo_py.sample_device -> (counts, method, note); device is 'cpu' or 'gpu'."""
        kwargs = {}
        if device == "gpu" and precision in ("fp32", "fp64"):
            kwargs["precision"] = precision
        try:
            counts, method, note = self.q.sample_device(circ, int(shots), int(seed), device, **kwargs)
        except TypeError:  # older binding without the precision keyword
            counts, method, note = self.q.sample_device(circ, int(shots), int(seed), device)
        return {str(k): int(v) for k, v in dict(counts).items()}, str(method), str(note)

    def exact_energy(self, circ_unbound, values, observable: str) -> float:
        """Exact <observable> on the CPU (fp64). TODO(phase 2): qumo_py.expectation_device on the GPU."""
        return float(self.q.expectation(circ_unbound.bind([float(v) for v in values]), observable))

    # -- QPU ---------------------------------------------------------------------------------
    def qpu_run(self, qasm_bound: str, shots: int, seed: int, label: str) -> dict:
        """Run (or dry-run) one QPU job. Returns {counts|None, local, dry_run, cost, plan, ...}."""
        from qumo_hardware.errors import HardwareError
        from qumo_hardware.registry import get_provider

        shots = int(min(shots, self.shots_cap))
        prov = get_provider(self.provider_name)
        local = bool(prov.is_local(self.backend))
        rec: dict[str, Any] = {"label": label, "provider": self.provider_name, "backend": self.backend,
                               "shots": shots, "local": local, "submitted": False, "dry_run": False}
        self.payloads.append(qasm_bound)
        t0 = time.perf_counter()
        try:
            if local:
                out = self.runner.run(qasm_bound, self.provider_name, self.backend, shots=shots, seed=seed)
            elif not self.allow_remote:
                raise HardwareError("QM-E957", f"remote backend '{self.backend}' is not allowed without --allow-remote")
            elif not self.submit:
                out = self.runner.run(qasm_bound, self.provider_name, self.backend, shots=shots, seed=seed,
                                      dry_run=True)
                rec["dry_run"] = True
            elif self.max_cost is None:
                raise HardwareError("QM-E956", "remote submission needs an explicit --max-cost",
                                    "pass --max-cost in the provider's cost unit after reviewing the dry run")
            else:
                out = self.runner.run(qasm_bound, self.provider_name, self.backend, shots=shots, seed=seed,
                                      confirm=bool(self.yes), max_cost=self.max_cost)
        except HardwareError as exc:
            rec.update({"error": exc.to_dict(), "seconds": round(time.perf_counter() - t0, 3), "counts": None})
            plan = exc.details.get("plan") if exc.details else None
            if plan:
                rec["cost"] = plan.get("cost")
                rec["cost_summary"] = plan.get("cost_summary")
            self.qpu_jobs.append(rec)
            return rec
        rec.update({
            "seconds": round(time.perf_counter() - t0, 3),
            "counts": out.get("counts"),
            "submitted": bool(out.get("submitted")),
            "job_id": out.get("job_id"),
            "cost": out.get("cost"),
            "cost_summary": out.get("cost_summary"),
            "requires_confirmation": out.get("requires_confirmation"),
            "compiled": {k: out.get("compiled", {}).get(k) for k in ("num_qubits", "two_qubit_gates", "depth", "ops")
                         if k in (out.get("compiled") or {})},
            "noisy": bool((out.get("finishing") or {}).get("noisy")) or self.provider_name == "ibm",
            "noise_model": (out.get("finishing") or {}).get("noise_model"),
            "notes": out.get("notes", []),
        })
        self.qpu_jobs.append(rec)
        return rec


# --------------------------------------------------------------------------- optimisers
def minimize(f: Callable[[np.ndarray], float], x0, method: str, maxiter: int, seed: int = 7,
             fd_step: float = 1e-4) -> dict:
    """COBYLA / Nelder-Mead / BFGS (central finite differences) via scipy; Adam / SPSA built in.

    TODO(phase 2): `gradient` from qumo_py.value_and_gradient (adjoint) + L-BFGS-B."""
    from scipy.optimize import minimize as sp_minimize

    x0 = np.asarray(x0, dtype=float)
    nfev = [0]
    hist: list[float] = []

    def fun(x):
        nfev[0] += 1
        v = float(f(np.asarray(x, dtype=float)))
        hist.append(v)
        return v

    def grad(x):
        g = np.zeros_like(x)
        for i in range(len(x)):
            e = np.zeros_like(x)
            e[i] = fd_step
            g[i] = (fun(x + e) - fun(x - e)) / (2 * fd_step)
        return g

    m = method.lower()
    if m == "cobyla":
        r = sp_minimize(fun, x0, method="COBYLA", options={"maxiter": int(maxiter), "rhobeg": 0.2, "tol": 1e-8})
        x, fx = r.x, r.fun
    elif m == "nelder-mead":
        r = sp_minimize(fun, x0, method="Nelder-Mead", options={"maxiter": int(maxiter), "xatol": 1e-6,
                                                                "fatol": 1e-9})
        x, fx = r.x, r.fun
    elif m == "bfgs":
        r = sp_minimize(fun, x0, jac=grad, method="BFGS", options={"maxiter": int(maxiter), "gtol": 1e-7})
        x, fx = r.x, r.fun
    elif m == "adam":
        x = x0.copy()
        mom, vel = np.zeros_like(x), np.zeros_like(x)
        lr, b1, b2, eps = 0.05, 0.9, 0.999, 1e-8
        for t in range(1, int(maxiter) + 1):
            g = grad(x)
            mom = b1 * mom + (1 - b1) * g
            vel = b2 * vel + (1 - b2) * g * g
            x = x - lr * (mom / (1 - b1 ** t)) / (np.sqrt(vel / (1 - b2 ** t)) + eps)
        fx = fun(x)
    elif m == "spsa":
        rng = np.random.default_rng(seed)
        x = x0.copy()
        a, c, alpha, gamma, big_a = 0.2, 0.1, 0.602, 0.101, max(1.0, 0.1 * maxiter)
        best_x, best_f = x.copy(), fun(x)
        for k in range(int(maxiter)):
            ak = a / (k + 1 + big_a) ** alpha
            ck = c / (k + 1) ** gamma
            delta = rng.choice([-1.0, 1.0], size=len(x))
            gk = (fun(x + ck * delta) - fun(x - ck * delta)) / (2 * ck) * delta
            x = x - ak * gk
            if (k + 1) % 10 == 0 or k == maxiter - 1:
                fk = fun(x)
                if fk < best_f:
                    best_x, best_f = x.copy(), fk
        x, fx = best_x, best_f
    else:
        raise ValueError(f"unknown optimizer {method}")
    return {"x": np.asarray(x, dtype=float), "fun": float(fx), "nfev": nfev[0], "history_first_last":
            [hist[0], hist[-1]] if hist else []}


# --------------------------------------------------------------------------- metrics
def tvd(a: dict, b: dict) -> float:
    na, nb = sum(a.values()), sum(b.values())
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0) / na - b.get(k, 0) / nb) for k in keys)


def pauli_expectations_from_counts(counts: dict, terms: list[tuple[float, dict[int, str]]],
                                   group: list[int]) -> tuple[float, float]:
    """(sum_i c_i <P_i>, standard error) for one QWC group measured in its rotated basis."""
    shots = sum(counts.values())
    keys = list(counts)
    w = np.array([counts[k] for k in keys], dtype=float)
    vals = np.zeros(len(keys))
    for i in group:
        c, ops = terms[i]
        sign = np.array([(-1) ** sum(int(k[::-1][qb]) for qb in ops) for k in keys], dtype=float)
        vals += c * sign
    mean = float((w * vals).sum() / shots)
    var = float((w * (vals - mean) ** 2).sum() / max(shots - 1, 1))
    return mean, math.sqrt(var / shots)


def interp_angles(gammas: list[float], betas: list[float]) -> tuple[list[float], list[float]]:
    """INTERP warm start (Zhou et al. 2020): depth p -> p+1."""
    p = len(gammas)

    def up(v):
        out = []
        for i in range(p + 1):
            left = v[i - 1] if i - 1 >= 0 else 0.0
            right = v[i] if i < p else 0.0
            out.append((i / p) * left + ((p - i) / p) * right)
        return out

    return up(gammas), up(betas)


# --------------------------------------------------------------------------- QAOA MaxCut
def run_qaoa(problem: dict, plan: dict, rt: Runtime, prefer_tool: bool = True) -> dict:
    q = rt.q
    edges = K.parse_edges(problem["data"]["graph"])
    n = K.graph_nodes(edges)
    p = int(plan["parameters"]["depth"])
    shots = int(plan["shots"])
    seed = rt.seed
    stages: list[Stage] = []
    out: dict[str, Any] = {"num_qubits": n, "num_edges": len(edges), "depth": p}

    t0 = time.perf_counter()
    max_cut, optimal = K.brute_force_maxcut(edges, n)
    total_w = sum(w for _, _, w in edges)
    cut_table = K.cut_values(edges, n)
    t_bf = time.perf_counter() - t0

    built1 = K.build_qaoa(edges, n, 1, prefer_tool)
    built = K.build_qaoa(edges, n, p, prefer_tool) if p > 1 else built1
    out["circuit"] = built.summary()
    c1 = q.load_qasm(built1.qasm_measured)
    cp_plain = q.load_qasm(built.qasm)
    cp_meas = q.load_qasm(built.qasm_measured)

    def cut_mean(counts):
        tot = sum(counts.values())
        cuts = np.array([cut_table[int(k, 2)] for k in counts])
        w = np.array([counts[k] for k in counts], dtype=float)
        mean = float((w * cuts).sum() / tot)
        sd = float(math.sqrt(max((w * (cuts - mean) ** 2).sum() / max(tot - 1, 1), 0.0)))
        return mean, sd / math.sqrt(tot), float(cuts.max()) if len(cuts) else 0.0, \
            float(w[np.abs(cuts - max_cut) < 1e-9].sum() / tot)

    # ---- GPU stage: p=1 landscape scan (sampled energies) --------------------------------
    gtask = plan["partition"]["gpu"]
    dev, why = rt.place(n)
    g_stage = Stage("gpu", gtask, "gpu", precision=plan["precision"])
    if why:
        g_stage.notes.append(why)
    gammas, betas = [0.6], [0.4]  # Farhi-style default when no scan runs
    t0 = time.perf_counter()
    if gtask == "landscape_scan":
        grid = 16
        best = None
        method = ""
        for i in range(grid):
            for j in range(grid):
                gm = math.pi * (i + 0.5) / grid
                bt = (math.pi / 2) * (j + 0.5) / grid
                counts, method, note = rt.sample(c1.bind([gm, bt]), 4096, seed + i * grid + j, dev, plan["precision"])
                if note and note not in g_stage.notes:
                    g_stage.notes.append(note)
                m, se, _, _ = cut_mean(counts)
                g_stage.evaluations += 1
                if best is None or m > best[0]:
                    best = (m, se, gm, bt)
        gammas, betas = [best[2]], [best[3]]
        g_stage.method = method
        out["scan"] = {"grid": f"{grid}x{grid}", "shots_per_point": 4096, "best_expected_cut": best[0],
                       "best_stderr": best[1], "gamma": best[2], "beta": best[3],
                       "estimator": "sample mean of the cut (diagonal cost: unbiased), standard error reported"}
    else:
        g_stage.notes.append("no landscape scan requested; initial angles gamma=0.6, beta=0.4")
    g_stage.device_used = dev
    while len(gammas) < p:
        gammas, betas = interp_angles(gammas, betas)
    x0 = np.array(gammas + betas)

    # ---- CPU stage: refine on exact expectations ----------------------------------------
    c_stage = Stage("cpu", plan["partition"]["cpu"], "cpu", device_used="cpu", method="statevector:exact (fp64)",
                    precision="fp64")
    t1 = time.perf_counter()
    obs = built.observable

    def energy(x):
        return rt.exact_energy(cp_plain, x, obs)

    method = plan["parameters"]["optimizer"]
    res = minimize(energy, x0, method, int(plan["parameters"]["maxiter"]), seed)
    c_stage.evaluations = res["nfev"]
    c_stage.notes.append(f"{method} on exact energies from p=1 scan warm start (INTERP to p={p})")
    c_stage.notes.append("QAOA angles are shared across gates: gradient-free or adjoint, never per-gate shift")
    e_opt = res["fun"]
    x_opt = res["x"]
    ideal_cut = -e_opt
    c_stage.seconds = time.perf_counter() - t1 + t_bf
    out["angles"] = {"gamma": [float(v) for v in x_opt[:p]], "beta": [float(v) for v in x_opt[p:]]}

    # GPU (or CPU) ideal distribution at the final angles: TVD reference
    t2 = time.perf_counter()
    bound_meas = cp_meas.bind([float(v) for v in x_opt])
    ideal_shots = 100_000 if n <= 20 else 20_000
    ideal_counts, method2, note2 = rt.sample(bound_meas, ideal_shots, seed + 99991, dev, plan["precision"])
    if note2 and note2 not in g_stage.notes:
        g_stage.notes.append(note2)
    g_stage.method = g_stage.method or method2
    g_stage.evaluations += 1
    g_stage.seconds = (t1 - t0) + (time.perf_counter() - t2)
    im, ise, ibest, ipopt = cut_mean(ideal_counts)
    out["ideal"] = {"expected_cut_exact": ideal_cut, "approximation_ratio": ideal_cut / max_cut,
                    "expected_cut_sampled": im, "stderr_sampled": ise, "p_optimal": ipopt,
                    "shots": ideal_shots, "method": method2}

    # ---- QPU stage: final sampling ----------------------------------------------------------
    q_stage = Stage("qpu", plan["partition"]["qpu"], rt.qpu)
    t3 = time.perf_counter()
    payload = bound_meas.to_qasm3()
    job = rt.qpu_run(payload, shots, seed, "final_sampling")
    q_stage.seconds = time.perf_counter() - t3
    q_stage.evaluations = 1
    q_stage.device_used = rt.qpu
    q_stage.method = "noisy emulator" if job.get("noisy") else ("dry run" if job.get("dry_run") else "qpu/emulator")
    out["qpu"] = {k: v for k, v in job.items() if k != "counts"}
    counts = job.get("counts")
    if counts:
        qm, qse, qbest, qpopt = cut_mean(counts)
        out["qpu"].update({"expected_cut": qm, "stderr": qse, "approximation_ratio": qm / max_cut,
                           "best_cut": qbest, "p_optimal": qpopt, "tvd_vs_ideal": tvd(counts, ideal_counts)})
        best_key = max(counts, key=lambda k: (cut_table[int(k, 2)], counts[k]))
        out["qpu"]["best_bitstring"] = best_key
    else:
        q_stage.notes.append("no counts (dry run or gate refusal); see qpu.error / qpu.cost")
    out["reference"] = {"max_cut": max_cut, "num_optimal_assignments": len(optimal), "total_weight": total_w,
                        "random_expected_cut": total_w / 2, "random_ratio": total_w / 2 / max_cut}
    return {"stages": [g_stage, c_stage, q_stage], "results": out}


# --------------------------------------------------------------------------- VQE (UCCSD / HEA)
def run_vqe(problem: dict, plan: dict, rt: Runtime, prefer_tool: bool = True) -> dict:
    q = rt.q
    d = problem["data"]
    alg = plan["algorithm"]
    depth = int(plan["parameters"]["depth"])
    seed = rt.seed
    if alg == "vqe_uccsd":
        built = K.build_uccsd(problem, prefer_tool)
        if depth > 1:
            built.notes.append(f"depth {depth} requested; UCCSD runs one Trotter step in this example")
    else:
        built = K.build_hea(problem, depth, prefer_tool)
    n = built.num_qubits
    obs = d["hamiltonian"]
    terms = K.parse_pauli_terms(obs)
    const = sum(c for c, ops in terms if not ops)
    groups = K.qwc_groups(terms)
    circ = q.load_qasm(built.qasm)
    npar = len(built.parameters)
    rng = np.random.default_rng(seed)
    x0 = np.zeros(npar) if alg == "vqe_uccsd" else rng.normal(0.0, 0.1, npar)
    out: dict[str, Any] = {"num_qubits": n, "circuit": built.summary(), "num_terms": len(terms),
                           "qwc_groups": len(groups)}
    e_exact = float(problem.get("reference", {}).get("ground_energy"))

    # ---- CPU (or GPU) training on exact energies -------------------------------------------
    gtask = plan["partition"]["gpu"]
    dev, why = rt.place(n)
    train_on_gpu_requested = gtask == "train_exact"
    c_stage = Stage("cpu", plan["partition"]["cpu"], "cpu", device_used="cpu", method="statevector:exact (fp64)",
                    precision="fp64")
    if train_on_gpu_requested:
        c_stage.notes.append("train_exact was requested on the GPU; phase 1 has no exact GPU expectation in "
                             "qumo_py (TODO phase 2: expectation_device), so training ran on the CPU")
    t0 = time.perf_counter()
    e_init = rt.exact_energy(circ, x0, obs)
    res = minimize(lambda x: rt.exact_energy(circ, x, obs), x0, plan["parameters"]["optimizer"],
                   int(plan["parameters"]["maxiter"]), seed)
    c_stage.seconds = time.perf_counter() - t0
    c_stage.evaluations = res["nfev"]
    x_opt = res["x"]
    e_opt = res["fun"]
    c_stage.notes.append(f"{plan['parameters']['optimizer']} from "
                         + ("Hartree-Fock (all zeros)" if alg == "vqe_uccsd" else "N(0, 0.1) seeded") + " start")
    out["ideal"] = {"energy_initial": e_init, "energy": e_opt, "energy_error": e_opt - e_exact,
                    "parameters": [float(v) for v in x_opt]}

    # ---- GPU stage: sampled verification of the energy (one sampling run per QWC group) ------
    g_stage = Stage("gpu", gtask if gtask != "train_exact" else "verify_exact", "gpu",
                    precision=plan["precision"])
    if why:
        g_stage.notes.append(why)
    t1 = time.perf_counter()
    bound = circ.bind([float(v) for v in x_opt])
    bound_qasm = bound.to_qasm3()
    group_circs = []
    for gi, grp in enumerate(groups):
        bases: dict[int, str] = {}
        for i in grp:
            bases.update(terms[i][1])
        group_circs.append((grp, _measure_in_basis(bound_qasm, n, bases)))
    est, var = const, 0.0
    method = ""
    if gtask != "none":
        for gi, (grp, qasm_g) in enumerate(group_circs):
            counts, method, note = rt.sample(q.load_qasm(qasm_g), 50_000, seed + 7 * gi, dev, plan["precision"])
            if note and note not in g_stage.notes:
                g_stage.notes.append(note)
            m, se = pauli_expectations_from_counts(counts, terms, grp)
            est += m
            var += se * se
            g_stage.evaluations += 1
        out["verify"] = {"energy_sampled": est, "stderr": math.sqrt(var), "shots_per_group": 50_000,
                         "method": method}
    g_stage.device_used = dev
    g_stage.method = method
    g_stage.seconds = time.perf_counter() - t1

    # ---- QPU stage: one estimation job per QWC group -----------------------------------------
    q_stage = Stage("qpu", plan["partition"]["qpu"], rt.qpu, device_used=rt.qpu)
    t2 = time.perf_counter()
    e_qpu, v_qpu, ok = const, 0.0, True
    for gi, (grp, qasm_g) in enumerate(group_circs):
        job = rt.qpu_run(qasm_g, int(plan["shots"]), seed + 1000 + gi, f"group {gi}")
        q_stage.evaluations += 1
        if not job.get("counts"):
            ok = False
            continue
        m, se = pauli_expectations_from_counts(job["counts"], terms, grp)
        e_qpu += m
        v_qpu += se * se
    q_stage.seconds = time.perf_counter() - t2
    q_stage.method = "noisy emulator" if rt.qpu_jobs and rt.qpu_jobs[-1].get("noisy") else "qpu/emulator"
    q_stage.notes.append(f"{len(groups)} job(s), one per qubit-wise-commuting group (the plugin has no batch API: "
                         "per-job overhead applies)")
    out["qpu"] = {"jobs": len(groups), "shots_per_job": int(plan["shots"])}
    if ok:
        out["qpu"].update({"energy": e_qpu, "stderr": math.sqrt(v_qpu), "energy_error": e_qpu - e_exact})
    out["reference"] = {"ground_energy": e_exact, "maximally_mixed_energy": const,
                        "source": problem.get("reference", {}).get("source", "")}
    return {"stages": [g_stage, c_stage, q_stage], "results": out}


def _measure_in_basis(qasm3: str, n: int, bases: dict[int, str]) -> str:
    """Bound OpenQASM 3 (no measurements) + basis rotations + measure all."""
    lines = qasm3.rstrip().splitlines()
    out = []
    for ln in lines:
        out.append(ln)
        if ln.strip().startswith("qubit["):
            out.append(f"bit[{n}] c;")
    out += K.basis_rotation_lines(bases)
    out.append("c = measure q;")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- Grover
def run_grover(problem: dict, plan: dict, rt: Runtime, prefer_tool: bool = True) -> dict:
    q = rt.q
    marked = list(problem["data"]["marked"])
    built = K.build_grover(problem, prefer_tool)
    ns = built.extra["search_qubits"]
    k = built.extra["iterations"]
    seed = rt.seed
    circ = q.load_qasm(built.qasm_measured)
    p_ideal = K.grover_success_probability(ns, len(marked), k)
    out: dict[str, Any] = {"num_qubits": built.num_qubits, "search_qubits": ns, "iterations": k,
                           "circuit": built.summary(), "marked": marked}
    dev, why = rt.place(built.num_qubits)
    g_stage = Stage("gpu", plan["partition"]["gpu"], "gpu", precision=plan["precision"], device_used=dev)
    if why:
        g_stage.notes.append(why)
    t0 = time.perf_counter()
    if plan["partition"]["gpu"] != "none":
        counts, method, note = rt.sample(circ, 20_000, seed, dev, plan["precision"])
        if note:
            g_stage.notes.append(note)
        g_stage.method = method
        g_stage.evaluations = 1
        succ = sum(v for kk, v in counts.items() if kk in marked) / 20_000
        out["ideal"] = {"success_probability_sampled": succ, "shots": 20_000,
                        "success_probability_exact": p_ideal}
    else:
        out["ideal"] = {"success_probability_exact": p_ideal}
    g_stage.seconds = time.perf_counter() - t0
    c_stage = Stage("cpu", plan["partition"]["cpu"], "cpu", device_used="cpu", method="post-processing",
                    precision="fp64")
    q_stage = Stage("qpu", plan["partition"]["qpu"], rt.qpu, device_used=rt.qpu)
    t1 = time.perf_counter()
    job = rt.qpu_run(circ.to_qasm3(), int(plan["shots"]), seed, "final_sampling")
    q_stage.seconds = time.perf_counter() - t1
    q_stage.evaluations = 1
    q_stage.method = "noisy emulator" if job.get("noisy") else "qpu/emulator"
    out["qpu"] = {kk: v for kk, v in job.items() if kk != "counts"}
    t2 = time.perf_counter()
    if job.get("counts"):
        cnt = job["counts"]
        tot = sum(cnt.values())
        ps = sum(v for kk, v in cnt.items() if kk in marked) / tot
        out["qpu"].update({"success_probability": ps, "stderr": math.sqrt(max(ps * (1 - ps), 1e-12) / tot),
                           "most_frequent": max(cnt, key=cnt.get)})
    out["reference"] = {"random_success_probability": len(marked) / (1 << ns), "ideal": p_ideal}
    c_stage.seconds = time.perf_counter() - t2
    return {"stages": [g_stage, c_stage, q_stage], "results": out}


# --------------------------------------------------------------------------- classification
def _scale(X_train: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Min-max scale features to [0, pi] using the training range."""
    lo, hi = X_train.min(axis=0), X_train.max(axis=0)
    return (X - lo) / np.where(hi > lo, hi - lo, 1.0) * math.pi


def svc_fit(Kmat: np.ndarray, y: np.ndarray, C: float = 1.0) -> tuple[np.ndarray, float]:
    """Soft-margin SVM dual (SLSQP): max sum a - 1/2 a^T (yy^T * K) a, 0 <= a <= C, y.a = 0."""
    from scipy.optimize import minimize as sp_minimize

    n = len(y)
    Q = (y[:, None] * y[None, :]) * Kmat

    res = sp_minimize(lambda a: 0.5 * a @ Q @ a - a.sum(), np.zeros(n), jac=lambda a: Q @ a - np.ones(n),
                      bounds=[(0.0, C)] * n, constraints=[{"type": "eq", "fun": lambda a: a @ y,
                                                           "jac": lambda a: y}],
                      method="SLSQP", options={"maxiter": 500, "ftol": 1e-12})
    a = np.clip(res.x, 0.0, C)
    free = (a > 1e-6) & (a < C - 1e-6)
    f_no_b = Kmat @ (a * y)
    b = float(np.mean(y[free] - f_no_b[free])) if free.any() else float(np.mean(y - f_no_b))
    return a, b


def run_kernel_svm(problem: dict, plan: dict, rt: Runtime, prefer_tool: bool = True) -> dict:
    q = rt.q
    d = problem["data"]
    Xtr, ytr = np.array(d["X_train"], float), np.array(d["y_train"], float)
    Xte, yte = np.array(d["X_test"], float), np.array(d["y_test"], float)
    n = Xtr.shape[1]
    reps = int(plan["parameters"]["depth"])
    built = K.kernel_circuit(n, reps)
    kc = q.load_qasm(built.qasm)
    kc_meas = q.load_qasm(built.qasm_measured)
    Str, Ste = _scale(Xtr, Xtr), _scale(Xtr, Xte)
    ftr = [K.kernel_feature_values(list(x)) for x in Str]
    fte = [K.kernel_feature_values(list(x)) for x in Ste]
    out: dict[str, Any] = {"num_qubits": n, "train": len(ytr), "test": len(yte), "circuit": built.summary()}
    # ---- GPU stage: Gram matrices (exact) ------------------------------------------------------
    dev, why = rt.place(n)
    g_stage = Stage("gpu", plan["partition"]["gpu"], "gpu", device_used="cpu", method="statevector:exact (fp64)",
                    precision="fp64")
    if why:
        g_stage.notes.append(why)
    if dev == "gpu":
        g_stage.notes.append("phase 1: exact kernel entries need expectation_device (TODO phase 2); ran on the CPU")
    t0 = time.perf_counter()

    def kval(fa, fb):
        return rt.exact_energy(kc, fa + fb, built.observable)

    N, M = len(ftr), len(fte)
    Ktr = np.eye(N)
    for i in range(N):
        for j in range(i + 1, N):
            Ktr[i, j] = Ktr[j, i] = kval(ftr[i], ftr[j])
    Kte = np.array([[kval(fte[i], ftr[j]) for j in range(N)] for i in range(M)])
    g_stage.evaluations = N * (N - 1) // 2 + M * N
    g_stage.seconds = time.perf_counter() - t0
    diag_err = max(abs(kval(ftr[i], ftr[i]) - 1.0) for i in range(min(N, 5)))
    # ---- CPU stage: SVM solve ---------------------------------------------------------------------
    c_stage = Stage("cpu", plan["partition"]["cpu"], "cpu", device_used="cpu", method="SVM dual (SLSQP)",
                    precision="fp64")
    t1 = time.perf_counter()
    alpha, b = svc_fit(Ktr, ytr, C=1.0)
    pred_tr = np.sign(Ktr @ (alpha * ytr) + b)
    pred_te = np.sign(Kte @ (alpha * ytr) + b)
    c_stage.seconds = time.perf_counter() - t1
    c_stage.evaluations = 1
    out["ideal"] = {"train_accuracy": float((pred_tr == ytr).mean()), "test_accuracy": float((pred_te == yte).mean()),
                    "support_vectors": int((alpha > 1e-6).sum()), "gram_min_eigenvalue": float(
                        np.linalg.eigvalsh(Ktr).min()), "gram_diag_max_error": float(diag_err)}
    # ---- QPU stage: a capped number of kernel entries -----------------------------------------
    q_stage = Stage("qpu", plan["partition"]["qpu"], rt.qpu, device_used=rt.qpu)
    t2 = time.perf_counter()
    cap = 20
    pairs = [(i, j) for i in range(M) for j in range(N)][:: max(1, (M * N) // cap)][:cap]
    errs = []
    for (i, j) in pairs:
        bound = kc_meas.bind(fte[i] + ftr[j]).to_qasm3()
        job = rt.qpu_run(bound, int(plan["shots"]), rt.seed + 17 * i + j, f"K[{i},{j}]")
        q_stage.evaluations += 1
        if job.get("counts"):
            cnt = job["counts"]
            p0 = cnt.get("0" * n, 0) / sum(cnt.values())
            errs.append(p0 - Kte[i, j])
    q_stage.seconds = time.perf_counter() - t2
    q_stage.method = "noisy emulator" if rt.qpu_jobs and rt.qpu_jobs[-1].get("noisy") else "qpu/emulator"
    q_stage.notes.append(f"{len(pairs)} test kernel entries (capped at {cap}); each is one job")
    out["qpu"] = {"entries": len(pairs), "shots_per_entry": int(plan["shots"])}
    if errs:
        out["qpu"].update({"mean_abs_error": float(np.mean(np.abs(errs))), "mean_error": float(np.mean(errs))})
    return {"stages": [g_stage, c_stage, q_stage], "results": out}


def run_vqc(problem: dict, plan: dict, rt: Runtime, prefer_tool: bool = True) -> dict:
    q = rt.q
    d = problem["data"]
    Xtr, ytr = np.array(d["X_train"], float), np.array(d["y_train"], float)
    Xte, yte = np.array(d["X_test"], float), np.array(d["y_test"], float)
    nf = Xtr.shape[1]
    layers = int(plan["parameters"]["depth"])
    built = K.reuploading_circuit(max(2, nf), layers, nf)
    circ = q.load_qasm(built.qasm)
    circ_meas = q.load_qasm(built.qasm_measured)
    nw = built.extra["num_weights"]
    Str, Ste = _scale(Xtr, Xtr) - math.pi / 2, _scale(Xtr, Xte) - math.pi / 2
    out: dict[str, Any] = {"num_qubits": built.num_qubits, "train": len(ytr), "test": len(yte),
                           "circuit": built.summary()}
    dev, why = rt.place(built.num_qubits)
    g_stage = Stage("gpu", plan["partition"]["gpu"], "gpu", device_used="cpu", method="statevector:exact (fp64)",
                    precision="fp64")
    if why:
        g_stage.notes.append(why)
    g_stage.notes.append("phase 1: exact training expectations run on the CPU (TODO phase 2: expectation_device)")
    t0 = time.perf_counter()

    def predict(w, X):
        return np.array([rt.exact_energy(circ, list(w) + list(x), "Z0") for x in X])

    def loss(w):
        return float(np.mean((predict(w, Str) - ytr) ** 2))

    rng = np.random.default_rng(rt.seed)
    res = minimize(loss, rng.normal(0, 0.5, nw), plan["parameters"]["optimizer"], int(plan["parameters"]["maxiter"]),
                   rt.seed)
    w = res["x"]
    g_stage.evaluations = res["nfev"] * len(ytr)
    g_stage.seconds = time.perf_counter() - t0
    f_te = predict(w, Ste)
    out["ideal"] = {"train_loss": res["fun"], "train_accuracy": float((np.sign(predict(w, Str)) == ytr).mean()),
                    "test_accuracy": float((np.sign(f_te) == yte).mean())}
    c_stage = Stage("cpu", plan["partition"]["cpu"], "cpu", device_used="cpu", method="post-processing",
                    precision="fp64")
    q_stage = Stage("qpu", plan["partition"]["qpu"], rt.qpu, device_used=rt.qpu)
    t1 = time.perf_counter()
    cap = 50
    agree, correct, done = 0, 0, 0
    for i in range(min(cap, len(yte))):
        job = rt.qpu_run(circ_meas.bind(list(w) + list(Ste[i])).to_qasm3(), int(plan["shots"]), rt.seed + i,
                         f"x_test[{i}]")
        q_stage.evaluations += 1
        if job.get("counts"):
            cnt = job["counts"]
            tot = sum(cnt.values())
            z = (cnt.get("0", 0) - cnt.get("1", 0)) / tot
            done += 1
            agree += int(np.sign(z) == np.sign(f_te[i]))
            correct += int(np.sign(z) == yte[i])
    q_stage.seconds = time.perf_counter() - t1
    q_stage.method = "noisy emulator" if rt.qpu_jobs and rt.qpu_jobs[-1].get("noisy") else "qpu/emulator"
    q_stage.notes.append(f"inference on {min(cap, len(yte))} test points (capped at {cap}); one job each")
    out["qpu"] = {"points": done}
    if done:
        out["qpu"].update({"agreement_with_exact": agree / done, "accuracy": correct / done})
    return {"stages": [g_stage, c_stage, q_stage], "results": out}


PIPELINES: dict[str, Callable[..., dict]] = {
    "qaoa_maxcut": run_qaoa,
    "vqe_uccsd": run_vqe,
    "vqe_hea": run_vqe,
    "grover_search": run_grover,
    "quantum_kernel_svm": run_kernel_svm,
    "vqc_reuploading": run_vqc,
}
