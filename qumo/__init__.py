"""Redirect to the Python package in python/qumo.

This directory holds the Mojo sources of QUMO, and the Mojo compiler reads only
its .mojo files. Python, started at the repository root, finds this directory
first for `import qumo`, ahead of an installed package. Without this file it
would be an empty namespace package and `qumo.Circuit` would not exist. With
it, `import qumo` at the root gives the real package (python/qumo, see
docs/PYTHON.md), the same one `pip install -e .` and `PYTHONPATH=python` give.
"""

import os as _os

_real = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "python", "qumo")
_init = _os.path.join(_real, "__init__.py")
if not _os.path.isfile(_init):
    raise ImportError(
        "this is the Mojo source directory of QUMO, not the Python package, and python/qumo was not found "
        "next to it (looked for " + _init + "). Install the package from a full checkout: "
        "`make py && pip install -e .` (docs/PYTHON.md)."
    )
__path__[:] = [_real]  # submodules (qumo.circuit, qumo.sim, ...) load from python/qumo
__file__ = _init
with open(_init, encoding="utf-8") as _f:
    exec(compile(_f.read(), _init, "exec"))
del _f
