"""Test fixtures. Tests never need (or use) real provider accounts:

* QUMO_HARDWARE_HOME points to a temporary job store;
* HOME, the IBM account file and the QISKIT_IBM_* variables are redirected or
  removed, so no saved credential can be picked up and nothing can be submitted
  to a real backend from the test suite.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

EXAMPLES = REPO / "examples"


def have(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


needs_ibm = pytest.mark.skipif(not have("qiskit_ibm_runtime"), reason="qiskit-ibm-runtime not installed")
needs_pytket = pytest.mark.skipif(not (have("pytket") and have("pytket.extensions.quantinuum")),
                                  reason="pytket-quantinuum not installed")
needs_qnexus = pytest.mark.skipif(not have("qnexus"), reason="qnexus not installed")
needs_pecos = pytest.mark.skipif(not have("pytket_pecos"), reason="pytket-quantinuum[pecos] not installed")
needs_guppy = pytest.mark.skipif(not have("guppylang"), reason="guppylang not installed")


def pytest_collection_modifyitems(config, items):
    if os.environ.get("QUMO_HW_SLOW"):
        return
    skip = pytest.mark.skip(reason="slow local emulator test (set QUMO_HW_SLOW=1)")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    monkeypatch.setenv("QUMO_HARDWARE_HOME", str(tmp_path / "hw"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for var in ("QISKIT_IBM_TOKEN", "QISKIT_IBM_INSTANCE", "QISKIT_IBM_CHANNEL", "QISKIT_IBM_URL"):
        monkeypatch.delenv(var, raising=False)
    if have("qiskit_ibm_runtime"):
        import qiskit_ibm_runtime.accounts.management as mgmt

        monkeypatch.setattr(mgmt, "_DEFAULT_ACCOUNT_CONFIG_JSON_FILE", str(tmp_path / "home" / "qiskit-ibm.json"))
    yield


@pytest.fixture(scope="session")
def qumo():
    from qumo_hardware.compile import load_qumo_py

    return load_qumo_py()


BELL = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
"""


def x_on(n: int, k: int, measure: bool = True) -> str:
    lines = ['OPENQASM 2.0;', 'include "qelib1.inc";', f"qreg q[{n}];"]
    if measure:
        lines.append(f"creg c[{n}];")
    lines.append(f"x q[{k}];")
    if measure:
        lines += [f"measure q[{i}] -> c[{i}];" for i in range(n)]
    return "\n".join(lines) + "\n"


def ghz(n: int) -> str:
    body = "".join(f"cx q[{i}],q[{i + 1}];\n" for i in range(n - 1))
    meas = "".join(f"measure q[{i}] -> c[{i}];\n" for i in range(n))
    return f'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[{n}];\ncreg c[{n}];\nh q[0];\n{body}{meas}'


# Routing-heavy: interactions between distant qubits force SWAPs / permutations on a line.
ROUTED = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
x q[0];
cx q[0],q[4];
cx q[4],q[1];
cx q[0],q[3];
cx q[3],q[2];
cx q[1],q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
"""
# x0=1; q4^=1 ->1; q1^=q4 ->1; q3^=q0 ->1; q2^=q3 ->1; q2^=q1 ->0   => c = [1,1,0,1,1] -> key "11011"
ROUTED_EXPECTED = "11011"

# Measurements into permuted classical bits.
PERMUTED_MEASURE = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
x q[0];
cx q[0],q[1];
measure q[0] -> c[2];
measure q[1] -> c[0];
measure q[2] -> c[1];
"""
# c2 = 1, c0 = 1, c1 = 0 -> key "101"
PERMUTED_EXPECTED = "101"

RANDOMISH = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
h q[0];
ry(0.7) q[1];
rz(1.3) q[2];
cx q[0],q[3];
ccx q[0],q[1],q[2];
rx(0.4) q[3];
cz q[1],q[3];
swap q[0],q[2];
u(0.3,0.2,0.1) q[1];
cx q[2],q[1];
measure q -> c;
"""
