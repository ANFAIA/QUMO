#!/usr/bin/env python3
"""AI-selected algorithm, run partly on a GPU and partly on a QPU (with an offline selector).

Reproduces: heterogeneous CPU+GPU+QPU workflow pattern - CUDA-Q multi-QPU, Qiskit execution modes, Braket Hybrid Jobs, PennyLane device swaps
Source: https://nvidia.github.io/cuda-quantum/latest/using/backends/sims/mqpusims.html
Source: https://quantum.cloud.ibm.com/docs/en/guides/execution-modes
Source: https://docs.aws.amazon.com/braket/latest/developerguide/braket-jobs.html
Source: https://docs.pennylane.ai/projects/lightning/en/stable/lightning_gpu/device.html

A model picks *what* to run (algorithm, hyperparameters, stage tasks) from
QUMO's catalogue; QUMO decides *where* each stage runs and whether anything
billable may run. Pipeline: context pack -> AI (or offline rules) plan ->
schema + allow-list validation -> circuit from QUMO's library
(qumo_build_circuit) -> GPU stage -> CPU stage -> QPU stage -> report with
CHECK lines.

Models: any OpenAI-compatible endpoint. Default: OpenRouter (key from
OPENROUTER_API_KEY only; default model anthropic/claude-opus-5.5, any other
OpenRouter model with --model). Local: --provider ollama | lmstudio | llamacpp
| vllm (or --provider local --base-url URL). Without a key or a reachable
server, or with --offline, the deterministic rules choose (logged).

QPU: default local:qumo-sim:grid:4x4/sc_cz+noise:superconducting, a
vendor-neutral noisy emulator. IBM (ibm:fake_torino), Quantinuum
(quantinuum:H2-1LE) and any qumo_hardware entry-point provider are selectable
peers. Real hardware needs --allow-remote --submit --yes --max-cost.

Checks: QAOA ideal approximation ratio above the problem's bound (Wurtz-Love
p=2 bound 0.7559 for 3-regular graphs; 5/6 for the 8-ring at p=2), sampled vs
exact energies, best sampled cut = brute-force max cut, QPU ratio between the
random baseline and ideal + 3 sigma, placement, nothing submitted remotely.

Run:  bench/.venv/bin/python examples/heterogeneous/ai_gpu_qpu.py --offline --small
      (PYTHONPATH=build:plugins/hardware, or build/gpu for the GPU build; see README.md)
example-meta: {"framework": "qumo", "tier": "fast", "timeout": 120, "requires": ["qumo_py", "cli", "scipy"], "args": ["--offline", "--small", "--seed", "7"]}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))  # examples/_checks.py

import kernels as K  # noqa: E402
import planner as P  # noqa: E402

DEFAULT_PROBLEM = "problems/maxcut_16_3reg.json"
SMALL_PROBLEM = "problems/maxcut_8_ring.json"
LARGE_PROBLEM = "problems/maxcut_20_3reg.json"


def load_qumo_py(prefer_gpu: str = "auto"):
    """Import qumo_py: PYTHONPATH first, else build/gpu (when --qumo-build gpu/auto and present), else build."""
    try:
        import qumo_py  # noqa: F401
        return sys.modules["qumo_py"]
    except ImportError:
        pass
    cands = []
    if prefer_gpu in ("auto", "gpu"):
        cands.append(REPO / "build" / "gpu")
    if prefer_gpu in ("auto", "cpu"):
        cands.append(REPO / "build")
    for d in cands:
        if (d / "qumo_py.so").is_file():
            sys.path.insert(0, str(d))
            import qumo_py  # noqa: F401
            return sys.modules["qumo_py"]
    raise SystemExit("qumo_py.so not found: run `make py` (CPU) or `make py-gpu` (GPU) first")


def ensure_hardware_plugin() -> None:
    try:
        import qumo_hardware  # noqa: F401
    except ImportError:
        sys.path.insert(0, str(REPO / "plugins" / "hardware"))


def modality_of(qpu: str) -> str:
    s = qpu.lower()
    if "trapped_ion" in s or s.startswith("quantinuum"):
        return "trapped_ion"
    if "neutral_atom" in s:
        return "neutral_atom"
    return "superconducting"


def is_remote(qpu: str) -> bool:
    prov, _, be = qpu.partition(":")
    if prov == "local":
        return False
    if prov == "ibm":
        return not be.startswith("fake_")
    if prov == "quantinuum":
        return not be.endswith("LE")
    return True  # third-party providers: assume billable until the provider says otherwise


def candidates(problem: dict, qpu: str, prefer_tool: bool) -> list[dict]:
    """Per compatible algorithm: built circuit summary + qumo_estimate_resources for the QPU modality."""
    out = []
    kind = problem["kind"]
    for alg, spec in sorted(P.ALGORITHMS.items()):
        if kind not in spec["kinds"]:
            continue
        rules_plan, _ = P.offline_plan(problem, qpu)
        depth = rules_plan["parameters"]["depth"] if rules_plan["algorithm"] == alg else 2
        try:
            b = _build_for(alg, problem, depth, prefer_tool)
        except Exception as exc:  # noqa: BLE001 - a candidate that cannot be built is reported, not fatal
            out.append({"algorithm": alg, "error": f"{type(exc).__name__}: {exc}"[:200]})
            continue
        row = {"algorithm": alg, "depth_estimated": depth, **{k: v for k, v in b.summary().items()
                                                               if k in ("num_qubits", "num_parameters", "source",
                                                                        "two_qubit_gates", "depth")}}
        try:
            est = K.cli_tool("qumo_estimate_resources", {"qasm": b.qasm, "profile": modality_of(qpu)})
        except RuntimeError as exc:
            est = {"error": str(exc)[:200]}
        if est:
            lg, hw = est.get("logical", {}), est.get("hardware", {})
            row["resources"] = {"gates_2q": lg.get("gates_2q"), "depth": lg.get("depth"),
                                "statevector_memory_bytes": lg.get("statevector_memory_bytes"),
                                "nisq_profile": hw.get("profile"),
                                "nisq_success_probability": hw.get("success_probability"),
                                "duration_us": hw.get("duration_us")}
        row["within_local_limit"] = b.num_qubits <= K.LOCAL_QUBIT_CAP
        out.append(row)
    return out


def _build_for(alg: str, problem: dict, depth: int, prefer_tool: bool):
    d = problem["data"]
    if alg == "qaoa_maxcut":
        e = K.parse_edges(d["graph"])
        return K.build_qaoa(e, K.graph_nodes(e), depth, prefer_tool)
    if alg == "vqe_uccsd":
        return K.build_uccsd(problem, prefer_tool)
    if alg == "vqe_hea":
        return K.build_hea(problem, depth, prefer_tool)
    if alg == "grover_search":
        return K.build_grover(problem, prefer_tool)
    if alg == "quantum_kernel_svm":
        return K.kernel_circuit(len(d["X_train"][0]), depth)
    if alg == "vqc_reuploading":
        nf = len(d["X_train"][0])
        return K.reuploading_circuit(max(2, nf), depth, nf)
    raise ValueError(alg)


def backends_info(allowed: list[str]) -> list[dict]:
    rows = []
    for b in allowed:
        remote = is_remote(b)
        rows.append({"qpu": b, "modality": modality_of(b), "local": not remote,
                     "cost": "billable: dry run shows the provider's estimate; submission needs a human's "
                             "--submit --yes --max-cost" if remote else "free (runs on this machine)",
                     "noise": "modality depolarizing model" if "+noise:" in b else (
                         "IBM calibration snapshot + Aer noise" if b.startswith("ibm:fake_") else
                         ("noiseless emulator" if not remote else "real device"))})
    return rows


# --------------------------------------------------------------------------- checks
STOCHASTIC_OPTIMIZERS = ("spsa",)  # small budgets do not reach optimal angles: checked against the p=1 bound
RATIO_SLACK = 5e-3  # optimiser tolerance under the analytic guarantee (bounds are for optimal angles)


def qaoa_ratio_bound(chk: dict, depth: int) -> tuple:
    """Lower bound on the ideal QAOA approximation ratio for the depth the plan chose.

    ``chk["ideal_ratio_min_by_depth"]`` tabulates the known guarantees (rings: (2p+1)/(2p+2)
    while the ring is longer than 2p+1; 3-regular graphs: 0.6924 at p=1, 0.7559 at p=2).
    Deeper QAOA at optimal angles is never worse than shallower QAOA, so a depth without a
    tabulated value uses the guarantee of the deepest tabulated depth below it; every
    schema-valid depth (1-6) is therefore checked against a bound that holds for it.
    Returns (bound, description)."""
    table = {int(k): float(v) for k, v in (chk.get("ideal_ratio_min_by_depth") or {}).items()}
    below = [d for d in table if d <= depth]
    if below:
        d = max(below)
        how = f"p={d} guarantee {table[d]:g}" if d == depth else f"p={depth} not tabulated, p={d} guarantee {table[d]:g}"
        return max(0.0, table[d] - RATIO_SLACK), how
    legacy = float(chk.get("ideal_ratio_min", 0.0))  # old problem files: single bound
    return legacy, f"fixed {legacy:g}"


def add_checks(log, problem: dict, plan: dict, res: dict, stages: list, rt, args, wall: float) -> None:
    r = res
    alg = plan["algorithm"]
    chk = problem.get("checks") or {}
    # placement follows the rules
    g = next(s for s in stages if s.name == "gpu")
    expect_dev, _ = rt.place(r.get("num_qubits", 0))
    if alg in ("quantum_kernel_svm", "vqc_reuploading"):
        expect_dev = "cpu"  # phase 1: exact expectations only on the CPU
    log.check("gpu stage placement follows the rules", g.device_used == expect_dev,
              f"device={g.device_used} expected={expect_dev} method={g.method or '-'}")
    log.check("no remote submission", all(j.get("local") or not j.get("submitted") for j in rt.qpu_jobs),
              f"{len(rt.qpu_jobs)} QPU job(s) on {rt.qpu}")
    qpu_ran = all(j.get("counts") for j in rt.qpu_jobs) and rt.qpu_jobs
    if "+noise:" in rt.qpu and qpu_ran:
        log.check("qpu emulator is noisy and local", all(j.get("noisy") and j.get("local") for j in rt.qpu_jobs),
                  f"noise={rt.qpu.split('+noise:')[1]}")
    if alg == "qaoa_maxcut":
        ideal = r["ideal"]
        ref = r["reference"]
        depth = int(plan["parameters"].get("depth", 1))
        stochastic = plan["parameters"].get("optimizer") in STOCHASTIC_OPTIMIZERS
        lo, basis = qaoa_ratio_bound(chk, 1 if stochastic else depth)
        if stochastic:
            basis = f"stochastic optimiser, {basis}"
        # hi: 1 plus float rounding (a perfect cut can evaluate to 1 + 1e-15)
        log.within(f"qaoa ideal approximation ratio (p={depth}, bound {basis})", ideal["approximation_ratio"],
                   lo, 1.0 + 1e-9)
        # 5 sigma of the sampled mean, plus the "rule of three" floor: outcomes with probability below
        # 3/shots may never be sampled (stderr 0 for a near-deterministic state), each costing <= max_cut
        log.close("ideal sampled cut matches exact expectation", ideal["expected_cut_sampled"],
                  ideal["expected_cut_exact"],
                  5 * ideal["stderr_sampled"] + 3.0 * ref["max_cut"] / max(1, int(ideal.get("shots", 1))) + 1e-9)
        if "scan" in r:
            log.within("p=1 scan found a cut above random", r["scan"]["best_expected_cut"],
                       ref["random_expected_cut"], ref["max_cut"])
        if qpu_ran:
            qp = r["qpu"]
            q_shots = int(qp.get("shots", 0))
            expected_opt = ideal["p_optimal"] * q_shots  # optimal samples expected without noise
            if expected_opt >= 20:  # missing them all is improbable even with a few % of noise
                log.check("qpu best sampled cut equals brute-force max cut", qp["best_cut"] == ref["max_cut"],
                          f"best={qp['best_cut']} max={ref['max_cut']} ({qp['best_bitstring']})")
            else:  # few shots or a shallow plan: the optimum may legitimately go unsampled
                log.within("qpu best sampled cut reaches the ideal mean cut (optimum rarely sampled)",
                           qp["best_cut"], math.floor(ideal["expected_cut_exact"]), ref["max_cut"])
            hi = ideal["approximation_ratio"] + 3 * qp["stderr"] / ref["max_cut"]
            log.within("qpu ratio between random baseline and ideal + 3 sigma", qp["approximation_ratio"],
                       ref["random_ratio"] + 1e-12, hi)
    elif alg in ("vqe_uccsd", "vqe_hea"):
        ideal = r["ideal"]
        e0 = r["reference"]["ground_energy"]
        log.within("vqe ideal energy is variational", ideal["energy"], e0 - 1e-9, ideal["energy_initial"] + 1e-12)
        if "energy_error_max" in chk:
            log.within("vqe ideal energy error", ideal["energy_error"], -1e-9, float(chk["energy_error_max"]))
        if "verify" in r:
            v = r["verify"]
            log.close("sampled energy matches exact", v["energy_sampled"], ideal["energy"], 5 * v["stderr"] + 1e-9)
        if qpu_ran and "energy" in r["qpu"]:
            qe, qs = r["qpu"]["energy"], r["qpu"]["stderr"]
            log.within("qpu energy between ground and maximally mixed (5 sigma)", qe, e0 - 5 * qs,
                       r["reference"]["maximally_mixed_energy"] + 5 * qs)
    elif alg == "grover_search":
        ideal = r["ideal"]
        if "success_probability_sampled" in ideal:
            ps, pe = ideal["success_probability_sampled"], ideal["success_probability_exact"]
            log.close("sampled success probability matches theory", ps, pe,
                      5 * (pe * (1 - pe) / ideal["shots"]) ** 0.5 + 1e-9)
        if qpu_ran:
            qp = r["qpu"]
            log.within("qpu success probability above random", qp["success_probability"],
                       r["reference"]["random_success_probability"] + 1e-12,
                       r["reference"]["ideal"] + 5 * qp["stderr"])
    elif alg == "quantum_kernel_svm":
        ideal = r["ideal"]
        log.close("kernel diagonal is 1", ideal["gram_diag_max_error"], 0.0, 1e-9)
        log.within("gram matrix is positive semidefinite", ideal["gram_min_eigenvalue"], -1e-9, 1e9)
        log.within("svm test accuracy", ideal["test_accuracy"], float(chk.get("test_accuracy_min", 0.7)), 1.0)
        if qpu_ran and "mean_abs_error" in r["qpu"]:
            log.within("qpu kernel entries mean abs error", r["qpu"]["mean_abs_error"], 0.0,
                       float(chk.get("qpu_kernel_mae_max", 0.15)))
    elif alg == "vqc_reuploading":
        ideal = r["ideal"]
        log.within("vqc test accuracy", ideal["test_accuracy"], float(chk.get("vqc_test_accuracy_min", 0.6)), 1.0)
        if qpu_ran and "agreement_with_exact" in r["qpu"]:
            log.within("qpu inference agrees with exact predictions", r["qpu"]["agreement_with_exact"], 0.7, 1.0)
    if args.small or args.size == "small":
        # value kept out of the CHECK line so repeated runs print identical CHECK lines
        log.check("small run wall time under 60 s", wall <= 60.0, "")
        print(f"[time] small run wall time {wall:.2f} s")


# --------------------------------------------------------------------------- report
def write_report(out_dir: Path, report: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")
    (out_dir / "plan.json").write_text(json.dumps({
        "schema": "qumo.hetero_plan.v1", "plan": report["plan"], "selector": report["decision_log"]["selector"],
        "problem": report["problem"]["name"], "problem_sha256": report["problem"]["sha256"]}, indent=1) + "\n")
    lines = [f"# Heterogeneous run: {report['problem']['name']} (seed {report['seed']})", "",
             f"- Selector: `{report['decision_log']['selector']}`"
             + (f" (fallback: {report['decision_log']['fallback_reason']})"
                if report['decision_log'].get('fallback_reason') else ""),
             f"- Algorithm: `{report['plan']['algorithm']}`, depth {report['plan']['parameters']['depth']}, "
             f"optimizer {report['plan']['parameters']['optimizer']}, shots {report['plan']['shots']}",
             f"- QPU: `{report['qpu']}`", f"- Rationale: {report['plan']['rationale'][:600]}", "",
             "| Stage | Task | Device | Method | Seconds | Evaluations | Notes |", "|---|---|---|---|---|---|---|"]
    for s in report["stages"]:
        lines.append(f"| {s['name']} | {s['task']} | {s['device_used']} | {s['method']} | {s['seconds']:.3f} | "
                     f"{s['evaluations']} | {'; '.join(s['notes'])[:300]} |")
    lines += ["", "## Results", "", "```json", json.dumps(report["results"], indent=1, default=str)[:6000], "```",
              "", "## Overrides", ""] + [f"- {o}" for o in report["decision_log"].get("overrides", [])] + [
        "", "## Checks", ""] + [f"- `{c['label']}`: {c['status']} {c['detail']}" for c in report["checks"]] + [""]
    (out_dir / "report.md").write_text("\n".join(lines))
    if report.get("payloads"):
        (out_dir / "payload.qasm").write_text(report["payloads"][0])


def environment(q) -> dict:
    try:
        from qumo_hardware.registry import provider_names
        provs = provider_names()
    except Exception:  # noqa: BLE001
        provs = []
    return {"qumo_version": q.version(), "qumo_py": getattr(q, "__file__", ""), "python": sys.version.split()[0],
            "platform": platform.platform(), "machine": platform.machine(), "qpu_providers": provs,
            "cli": str(K.cli_path() or "")}


# --------------------------------------------------------------------------- main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--problem", default=None, help=f"problem JSON (default {DEFAULT_PROBLEM})")
    ap.add_argument("--goal", default="", help="natural-language goal attached to the problem (data, not code)")
    ap.add_argument("--offline", action="store_true", help="never call a model: deterministic rules")
    ap.add_argument("--replay", default=None, help="re-run a stored plan.json (no model call)")
    ap.add_argument("--provider", default=None,
                    help="openrouter (default) | ollama | lmstudio | llamacpp | vllm | local | rules")
    ap.add_argument("--model", default=None, help="model id, e.g. anthropic/claude-opus-5.5 or qwen3:8b")
    ap.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint (local servers)")
    ap.add_argument("--profiles", default=None, help="LLM profiles file (default llm_profiles.json)")
    ap.add_argument("--llm-retries", type=int, default=2, help="corrective retries after an invalid answer")
    ap.add_argument("--llm-timeout", type=float, default=None)
    ap.add_argument("--transport", choices=["urllib", "openai"], default=None,
                    help="HTTP client: urllib (default) or the optional openai package")
    ap.add_argument("--qpu", default=None, help=f"PROVIDER:BACKEND (default {P.DEFAULT_QPU})")
    ap.add_argument("--allow-remote", action="store_true", help="let remote (billable) backends be chosen")
    ap.add_argument("--submit", action="store_true", help="submit to a remote backend (else dry run)")
    ap.add_argument("--yes", action="store_true", help="confirm remote submission after reviewing the dry run")
    ap.add_argument("--max-cost", type=float, default=None, help="cost cap in the provider's unit (required to submit)")
    ap.add_argument("--size", choices=["small", "default", "large"], default="default")
    ap.add_argument("--small", action="store_true", help="alias of --size small")
    ap.add_argument("--shots", type=int, default=None, help="cap on QPU shots per job")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--force-gpu", action="store_true", help="run the GPU stage on the GPU below the threshold")
    ap.add_argument("--qumo-build", choices=["auto", "cpu", "gpu"], default="auto")
    ap.add_argument("--no-tool", action="store_true", help="use the Python circuit builders, not qumo_build_circuit")
    ap.add_argument("--out", default=str(REPO / "build" / "hetero"))
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    if a.small:
        a.size = "small"
    return a


def resolve_problem_path(a) -> str:
    if a.problem:
        return a.problem
    return {"small": SMALL_PROBLEM, "large": LARGE_PROBLEM}.get(a.size, DEFAULT_PROBLEM)


def main(argv=None, runner=None, client_factory=None, transport=None) -> dict:
    t_start = time.perf_counter()
    a = parse_args(argv)
    say = (lambda *_: None) if a.quiet else (lambda m: print(m, flush=True))
    from _checks import CheckLog

    ensure_hardware_plugin()
    q = load_qumo_py(a.qumo_build)
    problem = K.load_problem(resolve_problem_path(a), a.size if a.problem else "default")
    n_need = P.required_qubits(problem)
    qpu_pinned = a.qpu is not None
    qpu = a.qpu or (P.DEFAULT_QPU if n_need <= 16 else P.DEFAULT_QPU_LARGE)
    if is_remote(qpu) and not a.allow_remote:
        raise SystemExit(f"--qpu {qpu} is a remote (billable) backend: add --allow-remote to plan for it "
                         "(a dry run shows the cost; --submit --yes --max-cost submits)")
    allowed = list(P.LOCAL_QPUS) + (list(P.REMOTE_QPUS) if a.allow_remote else [])
    if qpu not in allowed:
        allowed.append(qpu)
    allowed = [b for b in allowed if (P.backend_capacity(b) or 10 ** 6) >= n_need]
    gpu_info = dict(q.gpu_info())
    say(f"[context] problem {problem['name']} ({problem['kind']}, {K.problem_qubits(problem)} qubits), "
        f"qumo_py {q.version()}, GPU compiled={gpu_info.get('compiled')} available={gpu_info.get('available')} "
        f"min_qubits={gpu_info.get('min_qubits_auto')}")

    # Stage 1: context pack (deterministic)
    prefer_tool = not a.no_tool
    t0 = time.perf_counter()
    caps = K.cli_describe_capabilities()
    cands = candidates(problem, qpu, prefer_tool)
    ctx = P.context_pack(problem, cands, gpu_info, caps, backends_info(allowed), allowed)
    t_ctx = time.perf_counter() - t0

    # Stage 2: selection
    replay = json.loads(Path(a.replay).read_text()) if a.replay else None
    plan, decision = P.select_plan(
        problem, provider=a.provider, model=a.model, base_url=a.base_url, offline=a.offline, goal=a.goal,
        qpu_backends=allowed, default_qpu=qpu, allow_remote=a.allow_remote, gpu_info=gpu_info, context=ctx,
        replay=replay, llm_retries=a.llm_retries, profiles_path=a.profiles, transport=transport,
        client_factory=client_factory, qpu_pinned=qpu_pinned, log=say, timeout=a.llm_timeout,
        transport_name=a.transport)
    say(f"[plan] {decision['selector']}: {plan['algorithm']} depth={plan['parameters']['depth']} "
        f"opt={plan['parameters']['optimizer']} gpu={plan['partition']['gpu']} cpu={plan['partition']['cpu']} "
        f"qpu={plan['partition']['qpu']} -> {plan['qpu_backend']} shots={plan['shots']}")
    for o in decision["overrides"]:
        say(f"[override] {o}")

    # Stages 3-7: build, GPU, CPU, QPU, post-process
    from stages import PIPELINES, Runtime

    rt = Runtime(q, gpu_info, plan["qpu_backend"], shots_cap=a.shots or 100_000, force_gpu=a.force_gpu,
                 allow_remote=a.allow_remote, submit=a.submit, yes=a.yes, max_cost=a.max_cost, runner=runner,
                 seed=a.seed, log=say)
    pipe = PIPELINES[plan["algorithm"]]  # allow-listed dispatch: nothing from the model is evaluated
    res = pipe(problem, plan, rt, prefer_tool)
    stages = res["stages"]
    for s in stages:
        say(f"[{s.name}] {s.task} on {s.device_used} ({s.method or '-'}) {s.seconds:.2f} s, "
            f"{s.evaluations} evaluation(s)" + (f" - {'; '.join(s.notes)}" if s.notes else ""))
    wall = time.perf_counter() - t_start

    log = CheckLog(f"heterogeneous/{problem['name']}")
    add_checks(log, problem, plan, res["results"], stages, rt, a, wall)
    report = {
        "schema": "qumo.hetero_report.v1",
        "problem": {"name": problem["name"], "kind": problem["kind"], "sha256": problem["sha256"],
                    "size": problem["size"], "qubits": K.problem_qubits(problem)},
        "seed": a.seed,
        "qpu": rt.qpu,
        "environment": environment(q),
        "gpu_info": gpu_info,
        "context": {"seconds": round(t_ctx, 3), "sha256": decision["context_sha256"],
                    "bytes": decision["context_bytes"], "candidates": cands},
        "decision_log": decision,
        "plan": plan,
        "stages": [s.to_dict() for s in stages],
        "results": res["results"],
        "qpu_jobs": [{k: v for k, v in j.items() if k != "counts"} for j in rt.qpu_jobs],
        "cost": {"estimated": [j.get("cost_summary") for j in rt.qpu_jobs],
                 "billed": "nothing billed" if all(j.get("local") or not j.get("submitted") for j in rt.qpu_jobs)
                 else "see provider", "llm_usd": sum((x.get("cost_usd") or 0.0) for x in decision["attempts"])},
        "checks": log.records,
        "payloads": rt.payloads[:1],
        "wall_seconds": round(wall, 3),
        "phase": "phase 1 (exact expectations on the CPU; GPU via sample_device; finite-difference gradients)",
    }
    out_dir = Path(a.out) / f"{problem['name']}-{a.seed}"
    write_report(out_dir, report)
    say(f"[report] {out_dir / 'report.json'}")
    report["_out_dir"] = str(out_dir)
    report["_all_passed"] = log.all_passed()
    log.finish(exit_on_failure=(__name__ == "__main__"))
    return report


if __name__ == "__main__":
    main()
