"""The four multi-device pipelines as planner targets (allow-listed dispatch for ai_gpu_qpu.py).

    sqd_krylov            -> sqd/sqd_hubbard.py                 (problem kind lattice_fermions)
    cut_two_modalities    -> cutting/cut_two_modalities.py      (problem kind wide_circuit)
    stream_decode_qec     -> realtime_qec/stream_decode.py      (problem kind qec_memory)
    ws_qaoa_escalation    -> qaoa_escalation/ws_qaoa_escalation.py (maxcut with a warm-start / escalation goal)

The plan's generic fields map to pipeline options by fixed rules (nothing the
model writes is passed through as text): depth = Krylov dimension (sqd), cut
budget (cutting), code distance (qec) or QAOA p (ws-qaoa); shots = QPU shots
per circuit, or shots per streamed batch (qec); maxiter = optimiser
iterations (ws-qaoa); partition.gpu = "none" skips the optional GPU stage.
Problem data supplies the physics (sites, t, U, graph, p, ...). QPU targets
per stage are the pipelines' own defaults (an ion and a superconducting
emulator where two modalities are needed); a human's --qpu replaces the
superconducting stage's target. The plan's qpu_backend is recorded only.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import kernels as K

HERE = Path(__file__).resolve().parent

PIPELINES: dict[str, dict] = {
    "sqd_krylov": {"path": "sqd/sqd_hubbard.py", "kinds": ["lattice_fermions"]},
    "cut_two_modalities": {"path": "cutting/cut_two_modalities.py", "kinds": ["wide_circuit"]},
    "stream_decode_qec": {"path": "realtime_qec/stream_decode.py", "kinds": ["qec_memory"]},
    "ws_qaoa_escalation": {"path": "qaoa_escalation/ws_qaoa_escalation.py", "kinds": ["maxcut"]},
}


def load(alg: str):
    """Import a pipeline module by its allow-listed algorithm name."""
    rel = PIPELINES[alg]["path"]
    name = "hetero_pipeline_" + alg
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _num(v) -> str:
    return repr(float(v)) if isinstance(v, float) else str(int(v))


def argv_for(alg: str, problem: dict, plan: dict, seed: int, out_dir: str, sc_qpu: str | None,
             safety: list[str]) -> tuple[list[str], dict]:
    """Pipeline argv from (problem data, validated plan, human flags) -> (argv, run kwargs)."""
    d = problem.get("data") or {}
    p = plan["parameters"]
    gpu_none = plan["partition"]["gpu"] == "none"
    argv = ["--seed", str(int(seed)), "--out", out_dir, "--quiet"]
    kw: dict = {}
    if alg == "sqd_krylov":
        argv += ["--sites", _num(d.get("sites", 8)), "--t", _num(float(d.get("t", 1.0))), "--U", _num(float(d.get("U", 4.0))),
                 "--krylov", str(int(p["depth"])), "--shots", str(int(plan["shots"]))]
        argv += ["--gpu-device", "none" if gpu_none else "auto"]
        if sc_qpu:
            argv += ["--qpu", sc_qpu]
    elif alg == "cut_two_modalities":
        argv += ["--max-cuts", str(min(2, int(p["depth"]))), "--sc-shots", str(int(plan["shots"])),
                 "--ion-shots", str(int(plan["shots"]))]
        argv += ["--gpu-device", "cpu" if gpu_none else "auto"]
        if sc_qpu:
            argv += ["--sc-qpu", sc_qpu]
    elif alg == "stream_decode_qec":
        argv += ["--code", str(d.get("code", "surface_z")), "--distance", str(int(p["depth"])),
                 "--p", _num(float(d.get("p", 0.004))), "--shots", str(int(d.get("shots", 200_000))),
                 "--batch", str(int(plan["shots"]))]
        argv += ["--gpu-device", "cpu" if gpu_none else "auto"]
        return argv, kw  # no QPU safety flags: the stand-in is local
    elif alg == "ws_qaoa_escalation":
        edges = K.parse_edges(d["graph"])
        kw["edges_override"] = edges
        argv += ["--depth", str(int(p["depth"])), "--maxiter", str(int(p["maxiter"])),
                 "--sc-shots", str(int(plan["shots"])), "--nodes", str(K.graph_nodes(edges))]
        argv += ["--gpu-device", "gpu"]
        if sc_qpu:
            argv += ["--sc-qpu", sc_qpu]
    else:
        raise KeyError(alg)
    return argv + safety, kw


def run(alg: str, problem: dict, plan: dict, seed: int, out_dir: str, sc_qpu: str | None, safety: list[str],
        runner=None) -> dict:
    mod = load(alg)
    argv, kw = argv_for(alg, problem, plan, seed, out_dir, sc_qpu, safety)
    if alg != "stream_decode_qec":
        kw["runner"] = runner
    return mod.run(argv, **kw)
