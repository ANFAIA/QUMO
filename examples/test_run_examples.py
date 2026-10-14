"""Tests for examples/run_examples.py and examples/_checks.py (pytest, offline).

Run: bench/.venv/bin/python -m pytest -q examples/test_run_examples.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_examples as rx  # noqa: E402
from _checks import CheckLog, fmt  # noqa: E402

META_OK = '{"framework": "qumo", "tier": "fast", "timeout": 30, "requires": [], "args": []}'


def _example(body: str, meta: str | None = META_OK, title: str = "Demo example",
             source: str = "https://example.org/demo", repro: str = "Demo - original") -> str:
    hdr = [f'"""{title}.', ""]
    if repro:
        hdr.append(f"Reproduces: {repro}")
    if source:
        hdr.append(f"Source: {source}")
    if meta is not None:
        hdr.append(f"example-meta: {meta}")
    hdr.append('"""')
    imports = "import os, sys\nsys.path.insert(0, os.environ['CHECKS_DIR'])\nfrom _checks import CheckLog\n"
    return "\n".join(hdr) + "\n" + imports + textwrap.dedent(body)


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A fake repository: <tmp>/examples/... with a few runnable Python examples."""
    monkeypatch.setenv("CHECKS_DIR", str(HERE))
    ex = tmp_path / "examples"
    (ex / "demo").mkdir(parents=True)
    (ex / "demo" / "good.py").write_text(_example("""
        log = CheckLog("good")
        log.close("value", 1.0, 1.0 + 1e-13, 1e-12)
        log.within("ratio", 0.8, 0.7559, 1.0)
        log.check("flag", True, "detail text")
        log.finish()
        if __name__ == "__main__":
            pass
    """))
    (ex / "demo" / "bad.py").write_text(_example("""
        log = CheckLog("bad")
        log.close("value", 1.0, 2.0, 1e-3)
        log.check("flag", True)
        log.finish()
    """))
    (ex / "demo" / "lies.py").write_text(_example("""
        print("CHECK sneaky: FAIL value=1")
        print("ALL CHECKS PASSED (1)")
    """))
    (ex / "demo" / "nochecks.py").write_text(_example("""
        print("done")
    """))
    (ex / "demo" / "slow.py").write_text(_example("""
        import time
        time.sleep(30)
    """, meta='{"framework": "qumo", "tier": "slow", "timeout": 1, "requires": [], "args": []}'))
    (ex / "demo" / "needs_gpu.py").write_text(_example("""
        raise SystemExit(3)
    """, meta='{"framework": "qumo", "tier": "fast", "timeout": 30, "requires": ["gpu"], "args": []}'))
    (ex / "demo" / "needs_mod.py").write_text(_example("""
        raise SystemExit(3)
    """, meta='{"framework": "qumo", "tier": "fast", "timeout": 30, '
              '"requires": ["surely_not_a_module_xyz"], "args": []}'))
    (ex / "demo" / "args.py").write_text(_example("""
        log = CheckLog("args")
        log.check("argv", sys.argv[1:] == ["--small", "--seed", "7"], " ".join(sys.argv[1:]))
        log.finish()
    """, meta='{"framework": "qumo", "tier": "fast", "timeout": 30, "requires": [], '
              '"args": ["--small", "--seed", "7"]}'))
    (ex / "demo" / "helper.py").write_text("def f():\n    return 1\n")  # imported module: not an example
    (ex / "demo" / "_private.py").write_text("raise SystemExit(1)\n")
    (ex / "demo" / "test_something.py").write_text("raise SystemExit(1)\n")
    (ex / "demo" / "reference").mkdir()
    (ex / "demo" / "reference" / "ref.py").write_text("raise SystemExit(1)\n")
    (ex / "colab").mkdir()
    (ex / "colab" / "nb.py").write_text("raise SystemExit(1)\n")
    (ex / "hardware").mkdir()
    (ex / "hardware" / "hw.py").write_text("if __name__ == '__main__':\n    raise SystemExit(1)\n")
    (ex / "legacy.py").write_text("print('legacy ok')\n")
    (ex / "legacy_broken.py").write_text("raise SystemExit(2)\n")
    (ex / "badmeta.py").write_text('"""Bad.\nexample-meta: {"framework": "qumo", tier: fast}\n"""\n')
    return tmp_path


def _by_path(examples):
    return {Path(e.path).name: e for e in examples}


def _run(tree, name, **kw):
    exs = _by_path(rx.discover(tree / "examples", repo=tree))
    return rx.run_one(exs[name], python=sys.executable, repo=tree, **kw)


# ----------------------------------------------------------------------------- _checks.py
def test_checklog_lines(capsys):
    log = CheckLog("unit")
    assert log.close("energy", 0.5, 0.25, 0.5)
    assert not log.close("tight", 1.0, 0.0, 1e-12)
    assert log.within("ratio", 0.8, 0.7559, 1.0)
    assert not log.within("nan", float("nan"), 0.0, 1.0)
    assert log.check("flag", True, "best=5")
    out = capsys.readouterr().out.splitlines()
    assert out == [
        "CHECK energy: PASS value=0.5 ref=0.25 diff=0.25 tol=0.5",
        "CHECK tight: FAIL value=1.0 ref=0.0 diff=1.0 tol=1e-12",
        "CHECK ratio: PASS value=0.8 range=[0.7559, 1.0]",
        "CHECK nan: FAIL value=nan range=[0.0, 1.0]",
        "CHECK flag: PASS best=5",
    ]
    for line in out:
        assert rx._RE_CHECK.match(line), line
    assert log.passed == 3 and log.failed == 2
    assert log.summary() == "unit: 2 of 5 checks failed (tight, nan)"
    with pytest.raises(SystemExit) as exc:
        log.finish()
    assert exc.value.code == 1


def test_checklog_finish_ok(capsys):
    log = CheckLog("ok")
    log.check("a", True)
    assert log.finish() is True
    assert capsys.readouterr().out.strip().splitlines()[-1] == "ALL CHECKS PASSED (1)"


def test_fmt_matches_mojo_style():
    # Mojo's String(Float64) prints the shortest round-trip form, like repr()
    assert fmt(0.1) == "0.1" and fmt(1e-12) == "1e-12" and fmt(1.0) == "1.0" and fmt(1e20) == "1e+20"


# ----------------------------------------------------------------------------- discovery
def test_discovery_rules(tree):
    exs = rx.discover(tree / "examples", repo=tree)
    names = {Path(e.path).name for e in exs}
    assert "good.py" in names and "legacy.py" in names
    for excluded in ("_private.py", "test_something.py", "ref.py", "nb.py", "hw.py"):
        assert excluded not in names, excluded
    by = _by_path(exs)
    assert by["helper.py"].helper and not by["helper.py"].legacy
    assert by["legacy.py"].legacy
    assert by["good.py"].meta["tier"] == "fast"
    assert by["good.py"].sources == ["https://example.org/demo"]
    assert by["good.py"].reproduces == "Demo - original"
    assert by["good.py"].title == "Demo example"
    assert "invalid example-meta" in by["badmeta.py"].meta_error
    with_hw = rx.discover(tree / "examples", include_hardware=True, include_references=True, repo=tree)
    names_hw = {Path(e.path).name for e in with_hw}
    assert "hw.py" in names_hw and "ref.py" in names_hw


def test_header_parsing_multiple_sources_and_qasm():
    text = textwrap.dedent('''\
        """Title here.

        Reproduces: PennyLane - Demo
        Source: https://pennylane.ai/qml/demos/x
        Source: https://github.com/PennyLaneAI/qml/blob/master/demonstrations_v2/x/demo.py (Apache-2.0)
        example-meta: {"framework": "pennylane", "tier": "slow", "timeout": 200, "requires": [], "args": []}
        """
    ''')
    h = rx.parse_header(text, "py")
    assert h["sources"] == ["https://pennylane.ai/qml/demos/x",
                            "https://github.com/PennyLaneAI/qml/blob/master/demonstrations_v2/x/demo.py"]
    assert h["meta"]["tier"] == "slow"
    q = rx.parse_header("// GHZ state.\n// Source: https://a.b/c\nOPENQASM 3.0;\n", "qasm")
    assert q["title"] == "GHZ state" and q["sources"] == ["https://a.b/c"]
    two = rx.parse_header('example-meta: {"a": 1}\nexample-meta: {"b": 2}\n', "py")
    assert "exactly one" in two["meta_error"]


def test_validate_meta():
    assert rx.validate_meta(json.loads(META_OK)) == []
    errs = rx.validate_meta({"framework": "x", "tier": "medium", "timeout": -1, "requires": "gpu", "args": [],
                             "extra": 1})
    text = " ".join(errs)
    assert "tier" in text and "timeout" in text and "requires" in text and "unknown keys" in text
    assert any("lacks" in e for e in rx.validate_meta({"framework": "x"}))


def test_select_tiers_and_only(tree):
    exs = rx.discover(tree / "examples", repo=tree)
    fast = {Path(e.path).name for e in rx.select(exs, "fast", None)}
    slow = {Path(e.path).name for e in rx.select(exs, "slow", None)}
    every = {Path(e.path).name for e in rx.select(exs, "all", None)}
    assert "slow.py" in slow and "slow.py" not in fast and "helper.py" not in every
    assert "legacy.py" in fast  # legacy smoke tests are fast tier
    only = rx.select(exs, "fast", ["examples/demo/slow.py"])
    assert [Path(e.path).name for e in only] == ["slow.py"]  # --only ignores the tier
    folder = rx.select(exs, "fast", ["examples/demo"])
    assert all(e.path.startswith("examples/demo/") for e in folder) and len(folder) >= 5


# ----------------------------------------------------------------------------- running
def test_pass(tree):
    r = _run(tree, "good.py")
    assert r.status == "PASS", r
    assert r.checks_passed == 3 and r.checks_failed == 0
    assert [c["label"] for c in r.checks] == ["value", "ratio", "flag"]


def test_fail_on_failed_check(tree):
    r = _run(tree, "bad.py")
    assert r.status == "FAIL" and "exit status 1" in r.reason and "value" in r.reason
    assert r.checks_failed == 1


def test_fail_on_fail_line_even_with_exit_zero(tree):
    r = _run(tree, "lies.py")
    assert r.status == "FAIL" and "sneaky" in r.reason


def test_fail_without_all_checks_passed(tree):
    r = _run(tree, "nochecks.py")
    assert r.status == "FAIL" and "ALL CHECKS PASSED" in r.reason


def test_timeout(tree):
    r = _run(tree, "slow.py")
    assert r.status == "TIMEOUT" and r.seconds < 10


def test_skip_requirements(tree):
    r = _run(tree, "needs_gpu.py")
    assert r.status == "SKIP" and "requires gpu" in r.reason
    r = _run(tree, "needs_mod.py")
    assert r.status == "SKIP" and "surely_not_a_module_xyz" in r.reason


def test_args_forwarded(tree):
    r = _run(tree, "args.py")
    assert r.status == "PASS", r.tail


def test_legacy_smoke(tree):
    assert _run(tree, "legacy.py").status == "PASS"
    r = _run(tree, "legacy_broken.py")
    assert r.status == "FAIL" and "exit status 2" in r.reason


def test_bad_meta_fails_without_running(tree):
    r = _run(tree, "badmeta.py")
    assert r.status == "FAIL" and r.command == []


def test_cli_and_qumo_py_requirements(tree):
    qasm = tree / "examples" / "demo" / "x.qasm"
    qasm.write_text("// X.\nOPENQASM 3.0;\nqubit[1] q;\nx q[0];\n")
    r = _run(tree, "x.qasm")
    assert r.status == "SKIP" and "requires cli" in r.reason
    (tree / "examples" / "demo" / "py_needs.py").write_text(_example(
        "pass\n", meta='{"framework": "qumo", "tier": "fast", "timeout": 5, "requires": ["qumo_py"], "args": []}'))
    r = _run(tree, "py_needs.py")
    assert r.status == "SKIP" and "make py" in r.reason


def test_run_all_report(tree, tmp_path):
    exs = rx.select(rx.discover(tree / "examples", repo=tree), "fast", ["examples/demo/good.py",
                                                                      "examples/demo/bad.py"])
    res = rx.run_all(exs, jobs=2, python=sys.executable, repo=tree)
    assert [r.status for r in res] == ["FAIL", "PASS"]  # discovery order (sorted paths)
    out = tmp_path / "report.json"
    rx.write_report(res, out, {"tier": "fast"})
    doc = json.loads(out.read_text())
    assert doc["schema"] == "qumo.examples_report.v1"
    assert doc["summary"]["PASS"] == 1 and doc["summary"]["FAIL"] == 1
    assert doc["results"][1]["checks"][0]["label"] == "value"
    table = rx.format_table(res)
    assert "1 passed, 1 failed" in table


# ----------------------------------------------------------------------------- index
def test_index_idempotent(tree):
    exs = rx.discover(tree / "examples", repo=tree)
    block = rx.readme_table(exs)
    readme = "# QUMO examples\n\nIntro text.\n"
    once = rx.splice_readme(readme, block)
    twice = rx.splice_readme(once, block)
    assert once == twice
    assert once.count(rx.INDEX_BEGIN) == 1 and once.startswith("# QUMO examples\n\nIntro text.\n")
    fw1 = rx.frameworks_index(exs)
    assert fw1 == rx.frameworks_index(rx.discover(tree / "examples", repo=tree))


def test_header_problems(tree):
    exs = rx.discover(tree / "examples", repo=tree)
    errs = "\n".join(rx.header_problems(exs))
    assert "badmeta.py" in errs
    assert "good.py" not in errs
    assert "legacy.py" not in errs  # top-level legacy examples are grandfathered
    (tree / "examples" / "demo" / "nosrc.py").write_text(_example("pass\n", source="", repro=""))
    errs = "\n".join(rx.header_problems(rx.discover(tree / "examples", repo=tree)))
    assert "nosrc.py: no 'Source: <url>' line" in errs and "nosrc.py: no 'Reproduces:' line" in errs


def test_repo_examples_have_valid_headers():
    """Every example in this repository parses (metadata JSON is valid)."""
    exs = rx.discover()
    bad = [f"{e.path}: {e.meta_error}" for e in exs if e.meta_error]
    bad += [f"{e.path}: {m}" for e in exs if e.meta for m in rx.validate_meta(e.meta)]
    assert not bad, "\n".join(bad)


def test_cli_list_runs():
    out = subprocess.run([sys.executable, str(HERE / "run_examples.py"), "--list", "--tier", "all"],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "example(s)" in out.stdout
