"""QumoError and the loader of the extension."""

import pickle
import subprocess
import sys

import pytest

import qumo
from qumo import _native
from qumo.errors import from_native


def test_error_carries_code_hint_and_message():
    with pytest.raises(qumo.QumoError) as e:
        qumo.sample(qumo.Circuit(40).h(0).t(0).measure_all(), method="statevector")
    err = e.value
    assert isinstance(err, Exception) and err.code == "QM-E404"
    assert err.message.startswith("state vector for 40 qubits") and "Hint" not in err.message
    assert err.hint and str(err).startswith("QM-E404: state vector for 40 qubits")
    assert "\n  Hint: " in str(err) and err.docs == "docs/ERRORS.md#qm-e404"
    back = pickle.loads(pickle.dumps(err))
    assert (back.code, back.hint, back.message) == (err.code, err.hint, err.message)


def test_hints_are_written_for_python():
    samples = {
        "QM-E402: circuit has no measurements. Hint: call measure_all() or use evolve()/expectation().":
            ("QM-E402", "circuit.measure_all()"),
        "QM-E410: a density matrix on 16 qubits needs 4^16 x 16 bytes; the limit is 15 qubits (16 GiB). Hint: sample_ptsbe() or sample() take Pauli noise on more qubits.":
            ("QM-E410", "qumo.sample()"),
        "QM-E101: circuit has 1 parameters [t] but 0 values were given.": ("QM-E101", "params="),
    }
    for text, (code, needle) in samples.items():
        err = from_native(Exception(text))
        assert err.code == code and needle in err.hint, err.hint
        assert "evolve()" not in err.hint and "sample_ptsbe()" not in err.hint
    plain = from_native(Exception("something odd"))
    assert plain.code == "" and plain.hint == "" and str(plain) == "something odd"


def test_parser_errors_keep_the_source_excerpt():
    with pytest.raises(qumo.QumoError) as e:
        qumo.load("OPENQASM 3.0;\nqubit[1] q;\nfoo q[0];\n")
    assert e.value.code == "QM-E510" and "foo q[0];" in e.value.message and "^" in e.value.message
    assert "define it" in e.value.hint


def test_type_errors_are_both_typeerror_and_qumoerror(bell):
    with pytest.raises(qumo.QumoTypeError) as e:
        qumo.sample(bell, shots="100")
    assert isinstance(e.value, TypeError) and isinstance(e.value, qumo.QumoError) and e.value.code == "QM-E911"


def test_missing_extension_says_make_py(monkeypatch, tmp_path):
    saved = sys.modules.pop("qumo_py")
    try:
        monkeypatch.setattr(_native, "candidate_paths", lambda: [tmp_path / "qumo_py.so"])
        with pytest.raises(qumo.QumoNativeMissing) as e:
            _native.load()
    finally:
        sys.modules["qumo_py"] = saved
    assert isinstance(e.value, ImportError) and e.value.code == "QM-E950"
    assert "make py" in e.value.hint and "QUMO_PY_PATH" in e.value.hint and str(tmp_path) in str(e.value)


def test_candidate_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("QUMO_PY_PATH", str(tmp_path))
    paths = _native.candidate_paths()
    assert paths[0] == tmp_path / "qumo_py.so"
    assert paths[-1].parts[-2:] == ("build", "qumo_py.so")
    monkeypatch.setenv("QUMO_PY_PATH", str(tmp_path / "x" / "custom.so"))
    assert _native.candidate_paths()[0].name == "custom.so"


def test_import_without_path_hacks_in_a_fresh_process():
    """`import qumo` finds <checkout>/build/qumo_py.so by itself: no sys.path entry for build/."""
    from conftest import ROOT

    code = ("import sys, qumo; assert not any(p.rstrip('/').endswith('build') for p in sys.path); "
            "print(qumo.__version__, qumo.sample(qumo.ghz(2, measure=True), shots=8).shots, "
            "sys.modules['qumo_py'].__file__)")
    env = {"PYTHONPATH": str(ROOT / "python"), "PATH": "/usr/bin:/bin"}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd="/")
    assert out.returncode == 0, out.stderr
    version, shots, path = out.stdout.split()
    assert version == qumo.__version__ and shots == "8" and path.endswith("build/qumo_py.so")


def test_public_api_is_documented():
    for name in qumo.__all__:
        obj = getattr(qumo, name)
        if callable(obj):
            assert obj.__doc__ and obj.__doc__.strip(), name
    for name in ("sample", "run", "statevector", "expectation", "transpile", "vqe", "qaoa_maxcut", "load",
                 "memory_experiment", "threshold_sweep", "probabilities", "bind"):
        assert ">>> " in getattr(qumo, name).__doc__, f"{name} has no runnable example"
