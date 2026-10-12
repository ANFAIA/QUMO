"""Benchpress gym for QUMO (Mojo quantum stack).

Install into a Benchpress checkout with `bench/benchpress/install.sh <benchpress_dir>`.
QUMO is called through its CPython extension (build/qumo_py.so). Targets are
passed as plain data (coupling edges + basis names) extracted from the same
Benchpress backends every other gym uses, so QUMO compiles for exactly the
same devices/topologies.
"""
import os
import sys

from benchpress.config import Configuration

_BUILD = os.environ.get("QUMO_BUILD_DIR", os.path.join(os.path.dirname(__file__), "_build"))
if _BUILD not in sys.path:
    sys.path.insert(0, _BUILD)

Configuration.gym_name = "qumo"
