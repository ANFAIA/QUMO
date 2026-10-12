"""Quantinuum plugin: native payload conversion, HQC estimates, mocked Nexus calls."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from conftest import BELL, RANDOMISH, ghz, needs_pytket, needs_qnexus

from qumo_hardware import jobs, runner
from qumo_hardware.compile import compile_for_target
from qumo_hardware.errors import E_CONFIRM, E_CREDENTIALS, E_LIMIT, HardwareError
from qumo_hardware.quantinuum import QuantinuumProvider, hqc_cost, native_gate_counts

pytestmark = needs_pytket


def _compiled(src, device="H2-1"):
    p = QuantinuumProvider()
    c = compile_for_target(src, p.target(device), level=2, seed=1)
    return p, c


def test_payload_is_native_and_one_to_one():
    from pytket.circuit import OpType

    p, c = _compiled(RANDOMISH)
    circ, report = p.prepare(c, "H2-1")
    types = {cmd.op.type for cmd in circ.get_commands()}
    assert types <= {OpType.Rz, OpType.PhasedX, OpType.ZZPhase, OpType.Measure, OpType.Barrier}
    assert circ.n_gates_of_type(OpType.PhasedX) == c.ops.get("rx", 0) + c.ops.get("ry", 0)
    assert circ.n_gates_of_type(OpType.ZZPhase) == c.ops.get("rzz", 0)
    assert circ.n_gates_of_type(OpType.Rz) == c.ops.get("rz", 0)
    assert circ.n_gates_of_type(OpType.Measure) == c.ops.get("measure", 0)
    assert p.validate(circ, "H2-1")["native"]


def test_payload_semantics_match_qumo_circuit():
    pytest.importorskip("pytket.extensions.qiskit")
    from pytket.extensions.qiskit import tk_to_qiskit
    from qiskit.quantum_info import Operator

    from qumo_hardware.qasm_bridge import to_qiskit

    p, c = _compiled(RANDOMISH)
    circ, _ = p.prepare(c, "H2-1")
    native = tk_to_qiskit(circ).remove_final_measurements(inplace=False)
    qumo = to_qiskit(c.qasm).remove_final_measurements(inplace=False)
    assert Operator(native).equiv(Operator(qumo))


def test_hqc_formula():
    assert hqc_cost(0, 0, 0, 1000) == 5.0
    assert hqc_cost(15, 7, 16, 100) == pytest.approx(8.3)
    assert hqc_cost(10, 5, 5000, 1) == pytest.approx(10.012)


def test_cost_estimates_by_device_kind():
    p, c = _compiled(ghz(8))
    circ, _ = p.prepare(c, "H2-1")
    n = native_gate_counts(circ)
    assert n["n_spam"] == 8 + 8 and n["n2q"] == c.ops["rzz"]
    hw = p.estimate_cost(c, 100, "H2-1", payload=circ)
    assert hw.unit == "HQC" and hw.amount == pytest.approx(hqc_cost(n["n1q"], n["n2q"], n["n_spam"], 100), abs=1e-3)
    assert hw.billable
    emu = p.estimate_cost(c, 100, "H2-1E", payload=circ)
    assert emu.unit == "eHQC" and emu.amount == hw.amount
    sc = p.estimate_cost(c, 100, "H2-1SC", payload=circ)
    assert sc.unit == "free" and not sc.billable and sc.breakdown["hqc_if_hardware"] > 5
    loc = p.estimate_cost(c, 100, "H2-1LE", payload=circ)
    assert loc.unit == "free"


# ----------------------------------------------------------------- mocked Nexus
class FakeNexus:
    def __init__(self, logged_in=True, result=None):
        import qnexus as real

        self.calls: list[tuple[str, dict]] = []
        self.result = result
        self.QuantinuumConfig = real.QuantinuumConfig
        self.HeliosConfig = real.HeliosConfig
        self.auth = SimpleNamespace(is_logged_in=lambda: logged_in)
        self.projects = SimpleNamespace(get_or_create=self._rec("projects.get_or_create", "PROJECT"))
        self.circuits = SimpleNamespace(upload=self._rec("circuits.upload", "CIRCREF"))
        self.hugr = SimpleNamespace(upload=self._rec("hugr.upload", "HUGRREF"))
        self.start_execute_job = self._rec("start_execute_job", SimpleNamespace(id="nexus-job-123"))
        self.jobs = SimpleNamespace(
            get=self._rec("jobs.get", SimpleNamespace(id="nexus-job-123")),
            status=lambda ref: SimpleNamespace(status=SimpleNamespace(value="COMPLETED")),
            results=lambda ref: [SimpleNamespace(download_result=lambda: self.result)],
        )

    def _rec(self, name, ret):
        def f(*args, **kwargs):
            self.calls.append((name, kwargs))
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


@needs_qnexus
def test_submit_h2_mocked_end_to_end(nexus):
    from pytket.backends.backendresult import BackendResult
    from pytket.circuit import Bit, OpType
    from pytket.utils.outcomearray import OutcomeArray

    # Device returns c[0..2] readouts: 70 x (c0=1,c1=0,c2=1) and 30 x (c0=0,c1=0,c2=0)
    rows = [[1, 0, 1]] * 70 + [[0, 0, 0]] * 30
    res = BackendResult(shots=OutcomeArray.from_readouts(rows), c_bits=[Bit("c", i) for i in range(3)])
    fake = nexus(result=res)
    src = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\ncreg c[3];\nh q[0];\ncx q[0],q[2];\nmeasure q -> c;\n'
    out = runner.run(src, "quantinuum", "H2-1", shots=100, confirm=True, wait=False)
    assert out["submitted"] and out["job_id"] == "nexus-job-123" and out["status"] == "QUEUED"
    (job_kw,) = fake.called("start_execute_job")
    assert job_kw["backend_config"].device_name == "H2-1"
    assert job_kw["n_shots"] == [100]
    est = out["cost"]["amount"]
    assert job_kw["max_cost"] == [round(est * 1.25 + 1.0, 2)]  # server-side guard
    (up,) = fake.called("circuits.upload")
    types = {cmd.op.type for cmd in up["circuit"].get_commands()}
    assert types <= {OpType.Rz, OpType.PhasedX, OpType.ZZPhase, OpType.Measure, OpType.Barrier}
    rec = jobs.load("nexus-job-123")
    assert rec.meta["provider_options"] == {"project": "qumo-hardware"}
    assert "token" not in str(rec.to_dict()).lower()
    st = runner.job_status("nexus-job-123")
    assert st["status"] == "DONE"
    r = runner.job_result("nexus-job-123")
    assert r["counts"] == {"101": 70, "000": 30}


@needs_qnexus
def test_submit_helios_mocked(nexus, monkeypatch):
    from hugr.qsystem.result import QsysResult

    res = QsysResult([[("c", [1, 1])]] * 4 + [[("c", [0, 0])]] * 6)
    fake = nexus(result=res)
    monkeypatch.setattr("qumo_hardware.quantinuum.hugr_program",
                        lambda circ, n: SimpleNamespace(compile=lambda: "HUGR-PACKAGE"))
    out = runner.run(BELL, "quantinuum", "Helios-1E", shots=10, confirm=True, wait=False)
    assert out["submitted"]
    (up,) = fake.called("hugr.upload")
    assert up["hugr_package"] == "HUGR-PACKAGE"
    (job_kw,) = fake.called("start_execute_job")
    cfg = job_kw["backend_config"]
    assert cfg.system_name == "Helios-1E" and cfg.emulator_config.n_qubits == 2
    assert job_kw["n_qubits"] == [2]
    assert out["cost"]["unit"] == "eHQC"
    r = runner.job_result(out["job_id"])
    assert r["counts"] == {"11": 4, "00": 6}


@needs_qnexus
def test_not_logged_in_is_a_clear_error(nexus):
    fake = nexus(logged_in=False)
    with pytest.raises(HardwareError) as ei:
        runner.run(BELL, "quantinuum", "H2-1", shots=10, confirm=True)
    assert ei.value.code == E_CREDENTIALS
    assert "qnx.login()" in ei.value.hint
    assert not fake.called("circuits.upload") and not fake.called("start_execute_job")


@needs_qnexus
def test_max_cost_refusal(nexus):
    fake = nexus()
    with pytest.raises(HardwareError) as ei:
        runner.run(ghz(8), "quantinuum", "H2-1", shots=1000, confirm=True, max_cost=10.0)
    assert ei.value.code == E_LIMIT
    assert not fake.calls


@needs_qnexus
def test_refuses_without_confirmation(nexus):
    fake = nexus()
    with pytest.raises(HardwareError) as ei:
        runner.run(BELL, "quantinuum", "H2-1", shots=10)
    assert ei.value.code == E_CONFIRM
    plan = ei.value.details["plan"]
    assert plan["backend"] == "H2-1" and plan["shots"] == 10 and plan["cost"]["unit"] == "HQC"
    assert not fake.calls


def test_shot_limit():
    with pytest.raises(HardwareError) as ei:
        runner.plan(BELL, "quantinuum", "H2-1", shots=20_000)
    assert ei.value.code == E_LIMIT


def test_tket_comparison_mode():
    p = runner.plan(ghz(5), "quantinuum", "H2-1", shots=100, provider_transpile=True)
    cmp = p.comparison
    assert cmp["qumo"]["two_qubit_gates"] == 4 and cmp["provider"]["two_qubit_gates"] >= 4
