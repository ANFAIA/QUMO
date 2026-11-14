"""`import qumo` at the repository root, with nothing installed and no PYTHONPATH.

Python finds the Mojo source directory ./qumo first there. qumo/__init__.py
redirects it to python/qumo, so a student who runs `python` at the root after
`make py` gets the real package and not an empty namespace package.
"""

import os
import subprocess
import sys

from conftest import ROOT

CODE = (
    "import qumo, pickle; "
    "c = qumo.Circuit(2).h(0).cx(0, 1).measure_all(); "
    "assert sum(qumo.sample(c, shots=100).values()) == 100; "
    "assert pickle.loads(pickle.dumps(c)) == c; "
    "import qumo.circuit; "
    "print(qumo.__file__)"
)


def _run(cwd, pythonpath=None):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    if pythonpath:
        env["PYTHONPATH"] = pythonpath
    return subprocess.run([sys.executable, "-c", CODE], cwd=cwd, env=env, capture_output=True, text=True)


def test_bare_import_at_the_repository_root():
    out = _run(ROOT)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == str(ROOT / "python" / "qumo" / "__init__.py")


def test_import_with_pythonpath_from_another_directory(tmp_path):
    out = _run(tmp_path, str(ROOT / "python"))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == str(ROOT / "python" / "qumo" / "__init__.py")


def test_the_mojo_directory_alone_says_what_to_do(tmp_path):
    # a copy of the redirect without python/qumo next to it: a clear ImportError, not an empty module
    (tmp_path / "qumo").mkdir()
    (tmp_path / "qumo" / "__init__.py").write_text((ROOT / "qumo" / "__init__.py").read_text())
    out = subprocess.run([sys.executable, "-c", "import qumo"], cwd=tmp_path, capture_output=True, text=True,
                         env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"})
    assert out.returncode != 0
    assert "pip install -e ." in out.stderr
