"""The default install and the default (native) path import no third-party Python package.

The vendor SDKs (qiskit, qiskit-ibm-runtime, qiskit-aer, qiskit-ionq, qnexus, pytket, guppylang,
pytket-pecos) are optional extras used only by the `--client python` fallbacks and by features that
are those SDKs' own simulators or compilers (IBM fake_*, Helios, H2-1LE, selene, comparison modes).
This test runs a child interpreter whose imports of those packages raise, then plans circuits and
lists backends on every built-in provider's default client.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]

CHILD = textwrap.dedent('''
    import importlib.abc, sys
    BANNED = {"qiskit", "qiskit_ibm_runtime", "qiskit_aer", "qiskit_ionq", "qnexus", "pytket", "pytket_pecos",
              "guppylang", "selene_sim", "requests", "httpx", "numpy", "scipy"}
    hits = []
    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in BANNED:
                hits.append(name)
                raise ImportError("blocked: " + name)
    sys.meta_path.insert(0, Block())
    sys.path.insert(0, sys.argv[1])
    from qumo_hardware import cli, runner
    from qumo_hardware.registry import BUILTIN, get_provider
    import qumo_hardware.ibm_native, qumo_hardware.quantinuum_native, qumo_hardware.speqtrum_native
    import qumo_hardware.qbraid, qumo_hardware.ionq
    bell = open(sys.argv[2]).read()
    for prov, backend in [("local", "qumo-sim"), ("local", "qumo-sim:line:3/sc_cz+noise:trapped_ion"),
                          ("ionq", "ionq_simulator:aria-1"), ("quantinuum", "H2-1E"), ("quantinuum", "H2-1SC")]:
        runner.plan(bell, get_provider(prov), backend, shots=10)
    for prov in BUILTIN:
        get_provider(prov).list_backends(online=False)
        assert get_provider(prov).client in ("native", None), prov
    print("HITS", sorted(set(hits)))
''')


def test_default_install_declares_no_dependencies():
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert meta["project"].get("dependencies", []) == []


def test_default_path_imports_no_third_party_package(tmp_path):
    out = subprocess.run([sys.executable, "-I", "-c", CHILD, str(ROOT), str(REPO / "examples/foundations/bell.qasm")],
                         capture_output=True, text=True, cwd=REPO, timeout=300,
                         env={"QUMO_PY_PATH": str(REPO / "build/qumo_py.so"), "HOME": str(tmp_path),
                              "PATH": "/usr/bin:/bin"})
    assert out.returncode == 0, out.stderr[-2000:]
    assert "HITS []" in out.stdout, out.stdout
