"""Tests of the `qumo` Python package and of the fixes in the `qumo_py` extension.

Run from the repository root after `make py`:

    PYTHONPATH=python bench/.venv/bin/python -m pytest -q python/tests

Without PYTHONPATH (and without `pip install -e .`) this file puts python/ on
sys.path, so plain `pytest python/tests` works too.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _import_package():
    """`import qumo`, adding python/ to sys.path when the package is not installed.

    At the repository root a bare `import qumo` finds the Mojo source directory
    ./qumo, whose __init__.py redirects to python/qumo (test_import_at_root.py).
    Should that redirect be missing, ./qumo is an empty namespace package, so the
    test for the real package is `hasattr(qumo, "Circuit")`."""
    try:
        import qumo
    except ImportError as exc:
        if type(exc).__name__ == "QumoNativeMissing":
            raise
        qumo = None
    if qumo is None or not hasattr(qumo, "Circuit"):
        for name in [m for m in sys.modules if m == "qumo" or m.startswith("qumo.")]:
            del sys.modules[name]
        sys.path.insert(0, str(ROOT / "python"))
        import qumo  # noqa: F401


_import_package()


@pytest.fixture(scope="session")
def q():
    """The low-level extension module."""
    import qumo

    return qumo.native


@pytest.fixture()
def bell():
    import qumo

    return qumo.Circuit(2).h(0).cx(0, 1).measure_all()
