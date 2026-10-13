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
from types import SimpleNamespace

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
x q[4];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
"""
# x0=1; q4^=1 ->1; q1^=q4 ->1; q3^=q0 ->1; q2^=q3 ->1; q2^=q1 ->0; x4 -> 0  => c = [1,1,0,1,0] -> key "01011"
# (not a palindrome, so a reversed bit order cannot pass unnoticed)
ROUTED_EXPECTED = "01011"

# Measurements into permuted classical bits.
PERMUTED_MEASURE = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
x q[0];
cx q[0],q[1];
measure q[0] -> c[1];
measure q[1] -> c[2];
measure q[2] -> c[0];
"""
# q = (1, 1, 0); c1 = q0 = 1, c2 = q1 = 1, c0 = q2 = 0 -> key "110" (identity mapping would give "011")
PERMUTED_EXPECTED = "110"

def not_palindrome(key: str) -> str:
    assert key != key[::-1], f"bit-order test key {key!r} is a palindrome"
    return key


def x_pattern(n_qubits: int, ones, extra_measures: int = 0) -> tuple[str, str]:
    """X on `ones`, measure q[i] -> c[i], then q[j % n] -> c[n + j] for j < extra_measures.

    Returns (OpenQASM 2, expected QUMO key). Re-measuring qubits gives circuits
    with more clbits than qubits (wide classical registers on small devices)."""
    n_cl = n_qubits + extra_measures
    lines = ['OPENQASM 2.0;', 'include "qelib1.inc";', f"qreg q[{n_qubits}];", f"creg c[{n_cl}];"]
    lines += [f"x q[{k}];" for k in sorted(set(ones))]
    lines += [f"measure q[{i}] -> c[{i}];" for i in range(n_qubits)]
    lines += [f"measure q[{j % n_qubits}] -> c[{n_qubits + j}];" for j in range(extra_measures)]
    bits = [1 if i in ones else 0 for i in range(n_qubits)] + [1 if (j % n_qubits) in ones else 0
                                                              for j in range(extra_measures)]
    return "\n".join(lines) + "\n", not_palindrome("".join(str(b) for b in reversed(bits)))


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


# ----------------------------------------------------------------- mocked Quantinuum Nexus
class FakeNexus:
    """Stands in for the `qnexus` module: records every call, never touches the network.

    `device` (optional) computes the job result from the uploaded payload, so a
    test can check bit order end to end (QUMO -> payload -> "device" -> counts)."""

    def __init__(self, logged_in=True, result=None, device=None, status="COMPLETED"):
        import qnexus as real

        self.calls: list[tuple[str, dict]] = []
        self.result = result
        self.device = device
        self.state = status
        self.uploaded = None
        self.QuantinuumConfig = real.QuantinuumConfig
        self.HeliosConfig = real.HeliosConfig
        self.auth = SimpleNamespace(is_logged_in=lambda: logged_in)
        self.projects = SimpleNamespace(get_or_create=self._rec("projects.get_or_create", "PROJECT"))
        self.circuits = SimpleNamespace(upload=self._upload("circuits.upload", "CIRCREF", "circuit"))
        self.hugr = SimpleNamespace(upload=self._upload("hugr.upload", "HUGRREF", "hugr_package"))
        self.start_execute_job = self._rec("start_execute_job", SimpleNamespace(id="nexus-job-123"))
        self.jobs = SimpleNamespace(
            get=self._rec("jobs.get", SimpleNamespace(id="nexus-job-123")),
            status=self._status,
            results=lambda ref: [SimpleNamespace(download_result=self._download)],
        )

    def _status(self, ref):
        st = self.state
        if isinstance(st, BaseException) or (isinstance(st, type) and issubclass(st, BaseException)):
            raise self.state
        return SimpleNamespace(status=SimpleNamespace(value=self.state))

    def _download(self):
        if self.device is not None:
            return self.device(self.uploaded, self.called("start_execute_job")[-1]["n_shots"][0])
        return self.result

    def _rec(self, name, ret):
        def f(*args, **kwargs):
            self.calls.append((name, kwargs))
            return ret
        return f

    def _upload(self, name, ret, key):
        def f(*args, **kwargs):
            self.calls.append((name, kwargs))
            self.uploaded = kwargs.get(key)
            return ret
        return f

    def called(self, name):
        return [kw for n, kw in self.calls if n == name]


@pytest.fixture
def nexus(monkeypatch):
    def make(**kw):
        fake = FakeNexus(**kw)
        monkeypatch.setattr("qumo_hardware.quantinuum._qnx", lambda: fake)
        return fake
    return make


def classical_outcome(circ) -> dict:
    """Deterministic 'device' for X-only native pytket payloads: {Bit: value}.

    PhasedX(1, phi) flips a basis state (up to phase); Rz and ZZPhase only add
    phases; Measure copies; Reset clears. Any other rotation makes the outcome
    random, which these tests never use."""
    from pytket.circuit import OpType

    state = {q: 0 for q in circ.qubits}
    out = {b: 0 for b in circ.bits}
    for cmd in circ.get_commands():
        t = cmd.op.type
        if t == OpType.PhasedX:
            a = float(cmd.op.params[0]) % 2
            if abs(a) < 1e-9:
                continue
            assert abs(a - 1) < 1e-9, f"non-deterministic rotation {cmd}"
            state[cmd.qubits[0]] ^= 1
        elif t == OpType.Measure:
            out[cmd.bits[0]] = state[cmd.qubits[0]]
        elif t == OpType.Reset:
            state[cmd.qubits[0]] = 0
        elif t in (OpType.Rz, OpType.ZZPhase, OpType.Barrier):
            continue
        else:
            raise AssertionError(f"unexpected op in payload: {cmd}")
    return out


def h2_device(circ, shots):
    """Fake H2: BackendResult over the payload's bits, listed in a scrambled order
    (registers reversed) so only a correct `cbits` selection gives the right key."""
    from pytket.backends.backendresult import BackendResult
    from pytket.utils.outcomearray import OutcomeArray

    out = classical_outcome(circ)
    bits = sorted(out, key=lambda b: (b.reg_name, b.index[0]), reverse=True)
    rows = [[out[b] for b in bits]] * shots
    return BackendResult(shots=OutcomeArray.from_readouts(rows), c_bits=bits)


def helios_device_for(native_circ):
    """Fake Helios: QsysResult with tag "c" listing c[0], c[1], ... (guppy result order)."""
    def device(_hugr_package, shots):
        from hugr.qsystem.result import QsysResult

        out = classical_outcome(native_circ)
        bits = sorted(out, key=lambda b: b.index[0])
        return QsysResult([[("c", [out[b] for b in bits])]] * shots)
    return device
