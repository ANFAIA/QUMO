"""The security rules shared with the Mojo harness: agents/spec/guard_rules.json.

That file is the one specification of what the agent tools refuse (paths, commands, URLs,
environment names, delegation). This module loads its `lists` at import; tools/repo.py,
tools/research.py, registry.py, backends.py, config.py and untrusted.py build their tables
from them instead of writing the values again. The Mojo harness reads the same lists through
a generated module (agents/spec/gen_mojo_rules.py). tests/test_guard_spec.py runs every
vector of the file through the real functions; agents/mojo/tests/test_guard_spec.mojo does
the same on the Mojo side. See agents/spec/README.md.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

SPEC_PATH = Path(__file__).resolve().parent.parent / "spec" / "guard_rules.json"
SCHEMA = "qumo.guard_rules.v1"

with open(SPEC_PATH, encoding="utf-8") as _fh:
    SPEC = json.load(_fh)
if SPEC.get("schema") != SCHEMA:
    raise RuntimeError(f"{SPEC_PATH}: expected schema {SCHEMA}")

LISTS = SPEC["lists"]
REPO = LISTS["repo"]
ENV = LISTS["env"]
RESEARCH = LISTS["research"]
MODEL_KEY = LISTS["model_key"]
DELEGATION = LISTS["delegation"]
UNTRUSTED = LISTS["untrusted"]


def secret_names_pattern() -> str:
    """The regular expression of a secret-looking file name (searched, case-insensitive)."""
    s = REPO["secret_names"]
    parts = (["^" + re.escape(p) for p in s["prefix"]] + [re.escape(p) for p in s["contains"]]
             + [re.escape(p) + "$" for p in s["suffix"]])
    return "(" + "|".join(parts) + ")"


def substrings_pattern(words: list[str]) -> str:
    return "|".join(re.escape(w) for w in words)


def git_injected_pattern() -> str:
    """Git configuration passed through the environment, matched from the start of the name."""
    g = ENV["git_injected"]
    parts = [re.escape(n) for n in g["exact"]] + [re.escape(p) + r"\d+" for p in g["numbered"]]
    return "(?:" + "|".join(parts) + r")\Z"
