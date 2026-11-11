"""Locate and load the compiled extension `qumo_py` (built by `make py`).

Search order, first hit wins:

1. a module named ``qumo_py`` that the process already imported;
2. ``$QUMO_PY_PATH``: the ``.so`` file, or a directory holding ``qumo_py.so``;
3. next to this package (``qumo/qumo_py.so``) or beside it in the same
   directory (an installed copy);
4. ``<checkout>/build/qumo_py.so`` when the package runs from a QUMO checkout
   (an editable install, or ``PYTHONPATH=python``).

The current directory and the rest of ``sys.path`` are not searched, so
running inside an untrusted directory cannot load a planted library (the same
rule as plugins/hardware).
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import sys
from pathlib import Path

from .errors import QumoNativeMissing

_NAME = "qumo_py"


def candidate_paths() -> list[Path]:
    out: list[Path] = []
    env = os.environ.get("QUMO_PY_PATH")
    if env:
        p = Path(env).expanduser()
        out.append(p if p.suffix in (".so", ".pyd", ".dylib") else p / "qumo_py.so")
    here = Path(__file__).resolve().parent
    out.append(here / "qumo_py.so")
    out.append(here.parent / "qumo_py.so")
    # python/qumo/_native.py -> the checkout is two levels above the package
    root = here.parent.parent
    if (root / "python" / "qumo_py.mojo").is_file():
        out.append(root / "build" / "qumo_py.so")
    return out


def load():
    """The `qumo_py` module, loading it on first use."""
    if _NAME in sys.modules:
        return sys.modules[_NAME]
    tried = candidate_paths()
    path = next((p for p in tried if p.is_file()), None)
    if path is None:
        where = "\n".join(f"    {p}" for p in tried)
        raise QumoNativeMissing(
            "the compiled extension qumo_py.so was not found. Looked at:\n" + where,
            "QM-E950",
            "run `make py` at the root of the QUMO checkout (about a minute; needs the Mojo toolchain "
            "from `make setup`), or point QUMO_PY_PATH at an existing qumo_py.so.",
        )
    try:
        loader = importlib.machinery.ExtensionFileLoader(_NAME, str(path))
        spec = importlib.util.spec_from_file_location(_NAME, str(path), loader=loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[_NAME] = module
        loader.exec_module(module)
    except Exception as exc:  # wrong Python ABI, missing Modular runtime, stale build
        sys.modules.pop(_NAME, None)
        raise QumoNativeMissing(
            f"cannot load {path} ({type(exc).__name__}: {exc})",
            "QM-E950",
            "rebuild it with `make py` using the same Python that imports it; the library links the "
            "Modular runtime of the checkout's .venv, so that environment must still exist.",
        ) from None
    if not hasattr(module, "op_table"):
        sys.modules.pop(_NAME, None)
        raise QumoNativeMissing(
            f"{path} is older than this qumo package (it has no op_table())",
            "QM-E950",
            "rebuild it with `make py`.",
        )
    return module


native = load()
