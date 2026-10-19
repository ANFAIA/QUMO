"""Planner tie-in for the four multi-device pipelines, the QPU safety gate and HEFT pinning."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(REPO / "plugins" / "hardware"))

import hetero_common as H  # noqa: E402
import kernels as K  # noqa: E402
import pipelines as HP  # noqa: E402
import planner as P  # noqa: E402

HAVE = (REPO / "build" / "qumo_py.so").is_file() or (REPO / "build" / "gpu" / "qumo_py.so").is_file()
needs_qumo = pytest.mark.skipif(not HAVE, reason="build/qumo_py.so missing (make py)")
PROBLEMS = {"hubbard_sqd": "sqd_krylov", "cut_two_modalities": "cut_two_modalities",
            "surface_code_stream": "stream_decode_qec", "maxcut_ws_escalation": "ws_qaoa_escalation"}


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("QUMO_HARDWARE_HOME", str(tmp_path / "hw"))
    for var in ("OPENROUTER_API_KEY", "QUMO_LLM_PROVIDER", "QUMO_LLM_MODEL", "QUMO_LLM_BASE_URL"):
        monkeypatch.delenv(var, raising=False)


def load(name):
    return K.load_problem(HERE / "problems" / f"{name}.json")


def test_rules_pick_each_pipeline_and_plans_validate():
    for name, alg in PROBLEMS.items():
        prob = load(name)
        plan, _ = P.offline_plan(prob, P.DEFAULT_QPU)
        assert plan["algorithm"] == alg
        assert P.validate_plan(plan, prob, P.LOCAL_QPUS) == [], name
        assert alg in HP.PIPELINES and prob["kind"] in HP.PIPELINES[alg]["kinds"]
    plan, notes = P.offline_plan(load("maxcut_8_ring"), P.DEFAULT_QPU, goal="warm start it, then escalate")
    assert plan["algorithm"] == "ws_qaoa_escalation" and any("warm start" in n for n in notes)


def test_cross_field_rules_for_pipelines():
    prob = load("surface_code_stream")
    plan, _ = P.offline_plan(prob, P.DEFAULT_QPU)
    bad = copy.deepcopy(plan)
    bad["parameters"]["depth"] = 4
    assert any("code distance" in e for e in P.validate_plan(bad, prob, P.LOCAL_QPUS))
    bad = copy.deepcopy(plan)
    bad["algorithm"] = "sqd_krylov"
    assert any("does not solve" in e for e in P.validate_plan(bad, prob, P.LOCAL_QPUS))
    ws = load("maxcut_ws_escalation")
    plan, _ = P.offline_plan(ws, P.DEFAULT_QPU)
    bad = copy.deepcopy(plan)
    bad["parameters"]["gradient"] = "none"
    assert any("adjoint" in e for e in P.validate_plan(bad, ws, P.LOCAL_QPUS))
    bad = copy.deepcopy(plan)
    bad["partition"]["qpu"] = "final_sampling"
    assert any("partition.qpu" in e for e in P.validate_plan(bad, ws, P.LOCAL_QPUS))
    cut = load("cut_two_modalities")
    plan, _ = P.offline_plan(cut, P.DEFAULT_QPU)
    bad = copy.deepcopy(plan)
    bad["parameters"]["depth"] = 3
    assert any("wire-cut budget" in e for e in P.validate_plan(bad, cut, P.LOCAL_QPUS))


def test_argv_mapping_is_fixed_by_rules():
    prob = load("hubbard_sqd")
    plan, _ = P.offline_plan(prob, P.DEFAULT_QPU)
    plan["rationale"] = "--qpu ibm:ibm_fez; rm -rf ~"  # model text never reaches argv
    argv, _ = HP.argv_for("sqd_krylov", prob, plan, 7, "/tmp/x", None, [])
    assert "ibm:ibm_fez" not in " ".join(argv) and "--krylov" in argv
    argv, kw = HP.argv_for("ws_qaoa_escalation", load("maxcut_ws_escalation"),
                           P.offline_plan(load("maxcut_ws_escalation"), P.DEFAULT_QPU)[0], 7, "/tmp/x",
                           "local:qumo-sim:grid:4x4/sc_cz+noise:superconducting", ["--allow-remote"])
    assert argv[argv.index("--sc-qpu") + 1].startswith("local:") and argv[-1] == "--allow-remote"
    assert len(kw["edges_override"]) > 0


class FakeRunner:
    def __init__(self):
        self.calls = []

    def run(self, source, provider, backend, **kw):
        self.calls.append({"provider": provider, "backend": backend, **kw})
        return {"counts": None if kw.get("dry_run") else {"0": 1}, "submitted": not kw.get("dry_run"),
                "cost_summary": "1 HQC", "compiled": {}}


def test_qpu_gate_safety_policy():
    r = FakeRunner()
    g = H.QpuGate(runner=r, say=lambda *_: None)
    rec = g.run("qasm", "quantinuum:H2-1", 10, 1, "x")
    assert rec["counts"] is None and rec["error"]["code"] == "QM-E957" and not r.calls
    g = H.QpuGate(allow_remote=True, runner=r, say=lambda *_: None)
    rec = g.run("qasm", "quantinuum:H2-1", 10, 1, "x")
    assert rec["dry_run"] and r.calls[-1]["dry_run"] is True
    g = H.QpuGate(allow_remote=True, submit=True, yes=True, runner=r, say=lambda *_: None)
    n = len(r.calls)
    rec = g.run("qasm", "ibm:ibm_fez", 10, 1, "x")
    assert rec["error"]["code"] == "QM-E956" and len(r.calls) == n  # no --max-cost: nothing sent
    g = H.QpuGate(allow_remote=True, submit=True, yes=True, max_cost=5.0, runner=r, say=lambda *_: None)
    g.run("qasm", "ibm:ibm_fez", 10, 1, "x")
    assert r.calls[-1]["confirm"] is True and r.calls[-1]["max_cost"] == 5.0
    assert H.is_remote("ibm:ibm_fez") and not H.is_remote("ibm:fake_torino") and not H.is_remote("quantinuum:H2-1LE")


@needs_qumo
def test_heft_honours_pins():
    q = H.load_qumo()
    c = q.load_qasm('OPENQASM 3.0; include "stdgates.inc"; qubit[3] q; bit[3] c; h q[0]; cx q[0], q[1]; '
                    'c = measure q;')
    devs = [{"name": "cpu0", "kind": "cpu"}, H.qpu_device("ion-qpu", "local:qumo-sim:all:8/ion_zz", 8),
            H.qpu_device("sc-qpu", "local:qumo-sim:grid:4x4/sc_cz", 16)]
    s = q.schedule([{"name": "a", "circuit": c, "shots": 100, "device": "ion-qpu"},
                    {"name": "b", "circuit": c, "shots": 100}], devs)
    assert s["device_of"][0] == "ion-qpu" and any("(pinned)" in n for n in s["notes"])
    with pytest.raises(Exception) as exc:
        q.schedule([{"name": "a", "circuit": c, "device": "nope"}], devs)
    assert "QM-E463" in str(exc.value)


@needs_qumo
@pytest.mark.parametrize("name", ["cut_two_modalities", "maxcut_ws_escalation"])
def test_ai_gpu_qpu_dispatches_pipeline_offline(tmp_path, name):
    import ai_gpu_qpu as A

    rep = A.main(["--offline", "--problem", str(HERE / "problems" / f"{name}.json"), "--seed", "7",
                  "--out", str(tmp_path), "--quiet"])
    assert rep["_all_passed"], rep["checks"]
    assert rep["plan"]["algorithm"] == PROBLEMS[name] and rep["pipeline"] == HP.PIPELINES[PROBLEMS[name]]["path"]
    assert rep["cost"]["billed"] == "nothing billed"
    assert (Path(rep["_out_dir"]) / "plan.json").is_file()
    r2 = A.main(["--replay", str(Path(rep["_out_dir"]) / "plan.json"), "--problem",
                 str(HERE / "problems" / f"{name}.json"), "--seed", "7", "--out", str(tmp_path / "r"), "--quiet"])
    assert r2["decision_log"]["selector"] == "replay" and r2["_all_passed"]
