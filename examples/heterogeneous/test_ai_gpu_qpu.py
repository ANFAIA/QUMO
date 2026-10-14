"""Tests for the heterogeneous example (pytest, offline: no network, no paid call).

Model access is exercised against a fake OpenAI-compatible server
(stdlib http.server on 127.0.0.1) that returns scripted answers: valid plans,
invalid JSON, schema violations, non-allow-listed algorithms, refusals,
truncation, timeouts and HTTP 401 / 429 / 500.

Run: bench/.venv/bin/python -m pytest -q examples/heterogeneous/test_ai_gpu_qpu.py
(needs build/qumo_py.so: `make py`; build/qumo for the context pack: `make cli`).
"""

from __future__ import annotations

import builtins
import copy
import json
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(REPO / "plugins" / "hardware"))

import kernels as K  # noqa: E402
import llm_client as L  # noqa: E402
import planner as P  # noqa: E402

HAVE_QUMO_PY = (REPO / "build" / "qumo_py.so").is_file() or (REPO / "build" / "gpu" / "qumo_py.so").is_file()
needs_qumo = pytest.mark.skipif(not HAVE_QUMO_PY, reason="build/qumo_py.so missing (make py)")
SECRET = "sk-or-v1-TESTSECRET0123456789abcdef"


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    for var in ("OPENROUTER_API_KEY", "QUMO_LLM_PROVIDER", "QUMO_LLM_MODEL", "QUMO_LLM_BASE_URL",
                "ANTHROPIC_API_KEY", "LOCAL_LLM_API_KEY", "QISKIT_IBM_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("QUMO_HARDWARE_HOME", str(tmp_path / "hw"))
    yield


# --------------------------------------------------------------------------- fake server
class FakeServer:
    """Scripted OpenAI-compatible endpoint. `script` is a list of responses
    consumed in order (the last one repeats): dict(status, body|raw, delay, headers)."""

    def __init__(self):
        self.script: list[dict] = []
        self.requests: list[dict] = []
        self.models = ["fake-model"]
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass

            def _send(self, status, body: bytes, headers=None):
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_GET(self):
                server.requests.append({"method": "GET", "path": self.path, "headers": dict(self.headers)})
                if self.path.endswith("/models"):
                    self._send(200, json.dumps({"data": [{"id": m} for m in server.models]}).encode())
                else:
                    self._send(404, b'{"error": {"message": "not found"}}')

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                server.requests.append({"method": "POST", "path": self.path, "headers": dict(self.headers),
                                        "json": body})
                step = server.script[min(len(server.chat_requests()) - 1, len(server.script) - 1)]
                if callable(step):
                    step = step(body)
                if step.get("delay"):
                    time.sleep(step["delay"])
                raw = step["raw"] if "raw" in step else json.dumps(step.get("body", {})).encode()
                self._send(step.get("status", 200), raw, step.get("headers"))

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"

    def chat_requests(self):
        return [r for r in self.requests if r["method"] == "POST"]

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    s = FakeServer()
    yield s
    s.close()


def completion(content: str, finish="stop", refusal=None, usage=None, model="fake-model") -> dict:
    msg = {"role": "assistant", "content": content}
    if refusal:
        msg["refusal"] = refusal
    return {"status": 200, "body": {"id": "gen-123", "model": model, "object": "chat.completion",
                                    "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                                    "usage": usage or {"prompt_tokens": 1200, "completion_tokens": 300,
                                                       "total_tokens": 1500}}}


def maxcut_problem(name="maxcut_8_ring.json"):
    return K.load_problem(HERE / "problems" / name)


def good_plan(problem=None, **changes) -> dict:
    plan, _ = P.offline_plan(problem or maxcut_problem(), P.DEFAULT_QPU)
    plan = copy.deepcopy(plan)
    plan["rationale"] = "model: QAOA fits MaxCut"
    plan["confidence"] = "medium"
    plan.update(changes)
    return plan


def select(problem, server_url=None, provider="local", **kw):
    args = dict(provider=provider, model=None, base_url=server_url, offline=False, goal="",
                qpu_backends=list(P.LOCAL_QPUS), default_qpu=P.DEFAULT_QPU, allow_remote=False,
                gpu_info={"compiled": False, "available": False}, context="{}", llm_retries=2,
                sleep=lambda s: None, log=lambda m: None)
    args.update(kw)
    return P.select_plan(problem, **args)


# --------------------------------------------------------------------------- planner unit tests
def test_offline_selection_deterministic_for_every_problem():
    for f in sorted((HERE / "problems").glob("*.json")):
        prob = K.load_problem(f)
        a, _ = P.offline_plan(prob, P.DEFAULT_QPU)
        b, _ = P.offline_plan(prob, P.DEFAULT_QPU)
        assert a == b, f.name
        qpu = P.DEFAULT_QPU if P.required_qubits(prob) <= 16 else P.DEFAULT_QPU_LARGE
        a, _ = P.offline_plan(prob, qpu)
        assert P.validate_plan(a, prob, P.LOCAL_QPUS) == [], f.name
    kinds = {K.load_problem(f)["kind"]: P.offline_plan(K.load_problem(f), P.DEFAULT_QPU)[0]["algorithm"]
             for f in (HERE / "problems").glob("*.json")}
    assert kinds == {"maxcut": "qaoa_maxcut", "molecule": "vqe_uccsd", "spin_model": "vqe_hea",
                     "search": "grover_search", "classification": "quantum_kernel_svm"}


def test_goal_keywords():
    prob = K.load_problem(HERE / "problems" / "moons_classification.json")
    plan, notes = P.offline_plan(prob, P.DEFAULT_QPU, goal="train a variational re-uploading model")
    assert plan["algorithm"] == "vqc_reuploading" and notes
    plan, _ = P.offline_plan(maxcut_problem("maxcut_16_3reg.json"), P.DEFAULT_QPU, goal="be accurate")
    assert plan["shots"] == 8000 and plan["parameters"]["maxiter"] == 400


def test_schema_and_cross_field_validation():
    prob = maxcut_problem()
    assert P.validate_plan(good_plan(prob), prob, P.LOCAL_QPUS) == []
    bad = good_plan(prob)
    bad["confirm"] = True
    assert any("unexpected property 'confirm'" in e for e in P.validate_plan(bad, prob, P.LOCAL_QPUS))
    bad = good_plan(prob)
    bad["parameters"]["max_cost"] = 1e9
    assert any("max_cost" in e for e in P.validate_plan(bad, prob, P.LOCAL_QPUS))
    for field, value in (("algorithm", "shor"), ("shots", 1_000_000), ("precision", "fp16"),
                         ("qpu_backend", "ibm:ibm_fez")):
        bad = good_plan(prob)
        bad[field] = value
        assert P.validate_plan(bad, prob, P.LOCAL_QPUS), field
    bad = good_plan(prob, algorithm="vqe_uccsd")
    assert any("does not solve" in e for e in P.validate_plan(bad, prob, P.LOCAL_QPUS))
    bad = good_plan(prob)
    bad["parameters"]["gradient"] = "parameter-shift"
    assert any("parameter shift" in e for e in P.validate_plan(bad, prob, P.LOCAL_QPUS))
    bad = good_plan(prob)
    bad["parameters"]["depth"] = "2"
    assert any("expected an integer" in e for e in P.validate_plan(bad, prob, P.LOCAL_QPUS))
    bad = good_plan(prob, precision="fp64")
    apple = {"compiled": True, "available": True, "vendor": "apple"}
    assert any("QM-E472" in e for e in P.validate_plan(bad, prob, P.LOCAL_QPUS, apple))
    big = maxcut_problem("maxcut_20_3reg.json")
    bad = good_plan(big, qpu_backend=P.DEFAULT_QPU)
    bad["parameters"]["depth"] = 1
    assert any("needs 20" in e for e in P.validate_plan(bad, big, P.LOCAL_QPUS))


def test_remote_backend_is_downgraded_without_allow_remote():
    plan = good_plan(qpu_backend="ibm:ibm_fez")
    notes = P.normalize_backend(plan, False, P.LOCAL_QPUS, P.DEFAULT_QPU)
    assert plan["qpu_backend"] == P.DEFAULT_QPU and "downgraded" in notes[0]
    plan = good_plan(qpu_backend="ibm:ibm_fez")
    assert P.normalize_backend(plan, True, P.LOCAL_QPUS + P.REMOTE_QPUS, P.DEFAULT_QPU) == []
    assert plan["qpu_backend"] == "ibm:ibm_fez"


def test_extract_json_and_redact():
    assert L.extract_json('{"a": 1}') == {"a": 1}
    assert L.extract_json('Here you go:\n```json\n{"a": {"b": "}"}}\n```\nthanks') == {"a": {"b": "}"}}
    assert L.extract_json('prefix {"a": [1, 2]} suffix {"c": 3}') == {"a": [1, 2]}
    with pytest.raises(ValueError):
        L.extract_json("no json here")
    text = f"401 for key {SECRET} with header Authorization: Bearer {SECRET} and api_key={SECRET}"
    red = L.redact(text, [SECRET])
    assert SECRET not in red and "TESTSECRET" not in red
    assert "sk-" not in L.redact("token sk-abcdefghijklmnop leaked")


def test_profiles_and_config(monkeypatch):
    cfg = L.resolve_config()
    assert cfg.provider == "openrouter" and cfg.base_url == "https://openrouter.ai/api/v1"
    assert cfg.model.startswith("anthropic/") and cfg.api_key_env == "OPENROUTER_API_KEY"
    assert L.resolve_config("ollama").base_url == "http://localhost:11434/v1"
    assert L.resolve_config("lmstudio").base_url == "http://localhost:1234/v1"
    assert L.resolve_config("local").preset == "ollama"
    c = L.resolve_config("local", model="my-model", base_url="http://127.0.0.1:9999/v1/")
    assert c.base_url == "http://127.0.0.1:9999/v1" and c.model == "my-model"
    monkeypatch.setenv("OPENROUTER_HTTP_REFERER", "https://example.org/qumo")
    assert L.resolve_config("openrouter").headers["HTTP-Referer"] == "https://example.org/qumo"
    monkeypatch.setenv("OPENROUTER_API_KEY", SECRET)
    pub = json.dumps(L.resolve_config("openrouter").public())
    assert SECRET not in pub and '"api_key_present": true' in pub
    with pytest.raises(L.LLMError):
        L.resolve_config("nonexistent")


# --------------------------------------------------------------------------- model calls (fake server)
def test_valid_plan_accepted_with_structured_output(server):
    prob = maxcut_problem()
    server.script = [completion(json.dumps(good_plan(prob)))]
    plan, dec = select(prob, server.url, model="fake-model")
    assert dec["selector"] == "ollama:fake-model" and dec["fallback_reason"] is None
    assert plan["rationale"] == "model: QAOA fits MaxCut"
    req = server.chat_requests()[0]["json"]
    assert req["response_format"]["type"] == "json_schema"
    assert req["response_format"]["json_schema"]["strict"] is True
    assert "<problem>" in req["messages"][1]["content"]
    att = dec["attempts"][0]
    assert att["usage"]["total_tokens"] == 1500 and att["latency_s"] >= 0 and att["structured_mode"] == "json_schema"


def test_model_auto_resolves_from_models_list(server):
    prob = maxcut_problem()
    server.models = ["qwen3:8b", "other"]
    server.script = [completion(json.dumps(good_plan(prob)), model="qwen3:8b")]
    plan, dec = select(prob, server.url, provider="lmstudio", model="auto")
    assert server.chat_requests()[0]["json"]["model"] == "qwen3:8b"
    assert dec["selector"] == "lmstudio:qwen3:8b"


def test_invalid_json_then_valid_retry(server):
    prob = maxcut_problem()
    server.script = [completion("I think QAOA is best!"), completion(json.dumps(good_plan(prob)))]
    plan, dec = select(prob, server.url, model="fake-model")
    assert dec["fallback_reason"] is None and len(dec["attempts"]) == 2
    assert "not JSON" in dec["attempts"][0]["validation_errors"][0]
    msgs = server.chat_requests()[1]["json"]["messages"]
    assert msgs[-1]["role"] == "user" and "rejected by QUMO's validator" in msgs[-1]["content"]


@pytest.mark.parametrize("mutate,needle", [
    (lambda p: p.update(confirm=True), "unexpected property 'confirm'"),
    (lambda p: p.update(algorithm="shor"), "is not one of"),
    (lambda p: p["parameters"].update(depth=9), "is not one of"),
    (lambda p: p.update(shell="rm -rf ~"), "unexpected property 'shell'"),
])
def test_schema_violations_fall_back_to_rules(server, mutate, needle):
    prob = maxcut_problem()
    bad = good_plan(prob)
    mutate(bad)
    server.script = [completion(json.dumps(bad))]
    plan, dec = select(prob, server.url, model="fake-model", llm_retries=1)
    assert dec["selector"] == "offline-rules" and "no valid plan after 2 attempt(s)" in dec["fallback_reason"]
    assert needle in dec["fallback_reason"]
    assert len(server.chat_requests()) == 2
    assert plan == P.offline_plan(prob, P.DEFAULT_QPU)[0] | {"qpu_backend": P.DEFAULT_QPU}


def test_refusal_and_length_fall_back(server):
    prob = maxcut_problem()
    server.script = [completion("", refusal="I can't help with that")]
    _, dec = select(prob, server.url, model="fake-model")
    assert dec["selector"] == "offline-rules" and "refused" in dec["fallback_reason"]
    assert len(server.chat_requests()) == 1
    server.script = [completion('{"algorithm": "qaoa_', finish="length")]
    _, dec = select(prob, server.url, model="fake-model")
    assert "max_tokens" in dec["fallback_reason"]


def test_timeout_falls_back(server):
    prob = maxcut_problem()
    server.script = [{"delay": 1.5, **completion(json.dumps(good_plan(prob)))}]
    t0 = time.perf_counter()
    _, dec = select(prob, server.url, model="fake-model", timeout=0.3)
    assert dec["selector"] == "offline-rules" and "timeout" in dec["fallback_reason"]
    assert time.perf_counter() - t0 < 6.0


def test_http_401_no_retry_and_key_redacted(server, monkeypatch, capsys):
    prob = maxcut_problem()
    monkeypatch.setenv("OPENROUTER_API_KEY", SECRET)
    server.script = [{"status": 401, "body": {"error": {"message": f"invalid key {SECRET}", "code": 401}}}]
    plan, dec = select(prob, server.url, provider="openrouter", model="anthropic/claude-opus-5.5",
                       log=print)
    assert dec["selector"] == "offline-rules" and "auth" in dec["fallback_reason"]
    assert len(server.chat_requests()) == 1  # 401 is not retried
    hdr = server.chat_requests()[0]["headers"]
    assert hdr.get("Authorization") == f"Bearer {SECRET}"  # sent to the endpoint...
    assert hdr.get("X-Title") == "QUMO heterogeneous example"
    body = server.chat_requests()[0]["json"]
    assert body["usage"] == {"include": True} and body["provider"] == {"require_parameters": True}
    dump = json.dumps(dec, default=str) + capsys.readouterr().out
    assert SECRET not in dump and "TESTSECRET" not in dump  # ...but never logged


def test_http_429_retried_then_accepted(server):
    prob = maxcut_problem()
    sleeps = []
    server.script = [{"status": 429, "headers": {"Retry-After": "2"}, "body": {"error": {"message": "slow down"}}},
                     completion(json.dumps(good_plan(prob)),
                                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
                                       "cost": 0.0012})]
    plan, dec = select(prob, server.url, model="fake-model", sleep=sleeps.append)
    assert dec["fallback_reason"] is None and sleeps == [2.0]
    assert dec["attempts"][0]["attempts"] == 2 and dec["attempts"][0]["cost_usd"] == 0.0012


def test_http_500_exhausts_retries(server):
    prob = maxcut_problem()
    server.script = [{"status": 500, "body": {"error": {"message": "boom"}}}]
    _, dec = select(prob, server.url, model="fake-model")
    assert dec["selector"] == "offline-rules" and "server" in dec["fallback_reason"]
    assert len(server.chat_requests()) == 2  # ollama profile: max_retries 1 -> 2 requests


def test_structured_output_unsupported_downgrades_to_prompt(server):
    prob = maxcut_problem()

    def step(body):
        if "response_format" in body:
            return {"status": 400, "body": {"error": {"message": "response_format is not supported"}}}
        return completion("Sure:\n```json\n" + json.dumps(good_plan(prob)) + "\n```")

    server.script = [step]
    plan, dec = select(prob, server.url, model="fake-model")
    assert dec["fallback_reason"] is None and dec["attempts"][0]["structured_mode"] == "prompt"
    second = server.chat_requests()[1]["json"]
    assert "response_format" not in second and "JSON Schema" in second["messages"][0]["content"]


def test_remote_choice_downgraded_and_logged(server):
    prob = maxcut_problem()
    server.script = [completion(json.dumps(good_plan(prob, qpu_backend="ibm:ibm_fez")))]
    plan, dec = select(prob, server.url, model="fake-model")
    assert plan["qpu_backend"] == P.DEFAULT_QPU and dec["fallback_reason"] is None
    assert any("downgraded" in o for o in dec["overrides"])


def test_no_key_means_no_request(server):
    prob = maxcut_problem()
    _, dec = select(prob, server.url, provider="openrouter")
    assert dec["selector"] == "offline-rules" and "OPENROUTER_API_KEY is not set" in dec["fallback_reason"]
    assert server.requests == []


def test_unreachable_local_server_falls_back():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    _, dec = select(maxcut_problem(), f"http://127.0.0.1:{port}/v1")
    assert dec["selector"] == "offline-rules" and "local server" in dec["fallback_reason"]


def test_openai_transport_requires_package(server, monkeypatch):
    real_import = builtins.__import__

    def blocked(name, *a, **kw):
        if name == "openai":
            raise ImportError("blocked for the test")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", blocked)
    prob = maxcut_problem()
    server.script = [completion(json.dumps(good_plan(prob)))]
    _, dec = select(prob, server.url, model="fake-model", transport_name="openai")
    assert dec["selector"] == "offline-rules" and "openai package is not installed" in dec["fallback_reason"]


# --------------------------------------------------------------------------- end to end
def _run(args, **kw):
    import ai_gpu_qpu

    return ai_gpu_qpu.main(args + ["--quiet"], **kw)


def _stable(report: dict) -> dict:
    """Report without wall-clock fields (for run-to-run comparisons)."""
    r = copy.deepcopy(report["results"])

    def strip(x):
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items() if k not in ("seconds", "latency_s", "job_id")}
        if isinstance(x, list):
            return [strip(v) for v in x]
        return x

    return {"plan": report["plan"], "results": strip(r), "checks": [
        c for c in report["checks"] if "wall time" not in c["label"]]}


@needs_qumo
def test_end_to_end_offline_small_twice_and_replay(tmp_path):
    out = str(tmp_path / "out")
    r1 = _run(["--offline", "--small", "--seed", "7", "--out", out])
    assert r1["_all_passed"], r1["checks"]
    assert r1["schema"] == "qumo.hetero_report.v1"
    for key in ("environment", "decision_log", "stages", "results", "cost", "plan", "context"):
        assert key in r1
    devs = {s["name"]: s for s in r1["stages"]}
    assert devs["cpu"]["device_used"] == "cpu"
    assert devs["gpu"]["device_used"] in ("cpu", "gpu") and devs["gpu"]["evaluations"] == 257
    assert devs["qpu"]["device_used"] == P.DEFAULT_QPU
    assert r1["results"]["ideal"]["approximation_ratio"] >= 0.83
    assert r1["cost"]["billed"] == "nothing billed"
    od = Path(r1["_out_dir"])
    for f in ("report.json", "report.md", "plan.json", "payload.qasm"):
        assert (od / f).is_file(), f
    r2 = _run(["--offline", "--small", "--seed", "7", "--out", str(tmp_path / "out2")])
    assert _stable(r1) == _stable(r2)
    r3 = _run(["--replay", str(od / "plan.json"), "--small", "--seed", "7", "--out", str(tmp_path / "out3")])
    assert r3["decision_log"]["selector"] == "replay"
    assert _stable(r1)["results"] == _stable(r3)["results"]


@needs_qumo
def test_runs_with_openai_and_anthropic_imports_blocked(tmp_path, monkeypatch):
    real_import = builtins.__import__

    def blocked(name, *a, **kw):
        if name.split(".")[0] in ("openai", "anthropic"):
            raise ImportError(f"{name} blocked for the test")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", blocked)
    r = _run(["--small", "--seed", "7", "--out", str(tmp_path)])  # no --offline: no key -> rules
    assert r["_all_passed"]
    assert "OPENROUTER_API_KEY is not set" in r["decision_log"]["fallback_reason"]


class RecordingRunner:
    """Wraps qumo_hardware.runner: records every call; remote calls never reach a provider."""

    def __init__(self):
        from qumo_hardware import runner

        self.real = runner
        self.calls = []

    def run(self, source, provider, backend, **kw):
        self.calls.append({"provider": provider, "backend": backend, **{k: v for k, v in kw.items()
                                                                         if k != "params"}})
        if provider != "local" and not kw.get("dry_run"):
            raise AssertionError(f"submission attempted to {provider}:{backend}")
        return self.real.run(source, provider, backend, **kw)


@needs_qumo
def test_injection_problem_never_submits(server, tmp_path):
    prob = K.load_problem(HERE / "problems" / "injection_test.json")
    evil = good_plan(prob, qpu_backend="ibm:ibm_fez")
    evil_with_confirm = dict(evil, confirm=True, max_cost=1e9)
    server.script = [completion(json.dumps(evil_with_confirm)), completion(json.dumps(evil))]
    rec = RecordingRunner()
    r = _run(["--problem", str(HERE / "problems" / "injection_test.json"), "--provider", "local", "--base-url",
              server.url, "--model", "fake-model", "--out", str(tmp_path)], runner=rec)
    assert r["_all_passed"], r["checks"]
    assert r["plan"]["qpu_backend"] == P.DEFAULT_QPU
    assert any("downgraded" in o for o in r["decision_log"]["overrides"])
    assert rec.calls and all(c["provider"] == "local" for c in rec.calls)
    assert all(not c.get("confirm") for c in rec.calls)
    user = server.chat_requests()[0]["json"]["messages"][1]["content"]
    assert user.index("<problem>") < user.index("SYSTEM OVERRIDE") < user.index("</problem>")


@needs_qumo
def test_remote_qpu_flag_requires_allow_remote(tmp_path):
    with pytest.raises(SystemExit) as exc:
        _run(["--offline", "--small", "--qpu", "ibm:ibm_fez", "--out", str(tmp_path)])
    assert "--allow-remote" in str(exc.value)


@needs_qumo
@pytest.mark.parametrize("problem,extra", [
    ("h2_molecule.json", []),
    ("grover_5.json", []),
    ("moons_classification.json", []),
    ("moons_classification.json", ["--goal", "train a variational re-uploading classifier, quick"]),
    ("heisenberg_16.json", ["--size", "small"]),
])
def test_other_problem_kinds_offline(tmp_path, problem, extra):
    r = _run(["--offline", "--problem", str(HERE / "problems" / problem), "--seed", "7", "--out", str(tmp_path)]
             + extra)
    assert r["_all_passed"], json.dumps(r["checks"], indent=1)
    assert [s["name"] for s in r["stages"]] == ["gpu", "cpu", "qpu"]


@needs_qumo
def test_python_builders_match_qumo_build_circuit():
    """The fallback builders give the same expectations as the library tool (when the CLI has it)."""
    from ai_gpu_qpu import load_qumo_py

    if K.cli_tool("qumo_build_circuit", {"kind": "ghz", "args": {"n": 2}}) is None:
        pytest.skip("build/qumo predates qumo_build_circuit (make cli)")
    q = load_qumo_py("cpu")
    edges = K.parse_edges("0-1 1-2 2-3 3-0 0-2:0.5")
    pairs = [(K.build_qaoa(edges, 4, 2, True), K.qaoa_maxcut_python(edges, 4, 2), [0.3, 0.2, 0.5, 0.4])]
    h2 = K.load_problem(HERE / "problems" / "h2_molecule.json")
    pairs.append((K.build_uccsd(h2, True), K.build_uccsd(h2, False), [0.01, -0.02, 0.11]))
    heis = K.load_problem(HERE / "problems" / "heisenberg_16.json", "small")
    heis["data"].pop("initial_x")
    pairs.append((K.build_hea(heis, 1, True), K.build_hea(heis, 1, False), [0.1 * i for i in range(32)]))
    for tool_b, py_b, vals in pairs:
        assert tool_b.source == "qumo_build_circuit" and py_b.source == "python-builder"
        obs = py_b.observable or tool_b.observable
        a = q.expectation(q.load_qasm(tool_b.qasm).bind(vals), obs)
        b = q.expectation(q.load_qasm(py_b.qasm).bind(vals), obs)
        assert abs(a - b) < 1e-12, (tool_b.algorithm, a, b)
    g_tool, g_py = K.build_grover(K.load_problem(HERE / "problems" / "grover_5.json"), True), \
        K.grover_python(["10110"])
    ct = q.sample(q.load_qasm(g_tool.qasm_measured), 2000, 3)
    cp = q.sample(q.load_qasm(g_py.qasm_measured), 2000, 3)
    assert max(ct, key=ct.get) == max(cp, key=cp.get) == "10110"


# --------------------------------------------------------------------------- depth-aware QAOA bound
def test_qaoa_ratio_bound_follows_plan_depth():
    import ai_gpu_qpu as A

    ring = maxcut_problem()["checks"]
    reg = maxcut_problem("maxcut_16_3reg.json")["checks"]
    s = A.RATIO_SLACK
    assert A.qaoa_ratio_bound(ring, 1)[0] == pytest.approx(0.75 - s)
    assert A.qaoa_ratio_bound(ring, 2)[0] == pytest.approx(0.8333 - s)
    assert A.qaoa_ratio_bound(ring, 3)[0] == pytest.approx(0.8333 - s)  # deeper: p=2 guarantee
    assert A.qaoa_ratio_bound(ring, 6)[0] == pytest.approx(0.8333 - s)
    assert A.qaoa_ratio_bound(reg, 1)[0] == pytest.approx(0.6924 - s)
    assert A.qaoa_ratio_bound(reg, 2)[0] == pytest.approx(0.7559 - s)
    assert A.qaoa_ratio_bound(reg, 5)[0] == pytest.approx(0.7559 - s)
    assert "not tabulated" in A.qaoa_ratio_bound(reg, 5)[1]
    assert A.qaoa_ratio_bound({"ideal_ratio_min": 0.5}, 2)[0] == 0.5  # legacy single bound


@needs_qumo
@pytest.mark.parametrize("depth", [1, 3])
def test_model_chosen_depth_passes_small_run(server, tmp_path, depth):
    """A schema-valid model plan with a depth other than the rules' default must not FAIL the ratio check."""
    prob = maxcut_problem()
    plan = good_plan(prob)
    plan["parameters"]["depth"] = depth
    server.script = [completion(json.dumps(plan))]
    r = _run(["--small", "--seed", "7", "--provider", "local", "--base-url", server.url, "--model", "fake-model",
              "--out", str(tmp_path)])
    assert r["decision_log"]["selector"] != "offline-rules", r["decision_log"]
    assert r["plan"]["parameters"]["depth"] == depth
    assert r["_all_passed"], [c for c in r["checks"] if c["status"] != "PASS"]
    assert any(f"(p={depth}," in c["label"] for c in r["checks"])
