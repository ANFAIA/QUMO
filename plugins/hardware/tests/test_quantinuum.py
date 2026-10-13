"""Quantinuum plugin: native payload conversion, HQC estimates, mocked Nexus calls."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from conftest import BELL, RANDOMISH, ghz, needs_pytket, needs_qnexus, not_palindrome

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


# ----------------------------------------------------------------- mocked Nexus (conftest.FakeNexus)
@needs_qnexus
def test_submit_h2_mocked_end_to_end(nexus):
    from pytket.backends.backendresult import BackendResult
    from pytket.circuit import Bit, OpType
    from pytket.utils.outcomearray import OutcomeArray

    # Device returns c[0..2] readouts: 70 x (c0=1,c1=1,c2=0) and 30 x (c0=0,c1=0,c2=0)
    rows = [[1, 1, 0]] * 70 + [[0, 0, 0]] * 30
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
    assert r["counts"] == {not_palindrome("011"): 70, "000": 30}  # leftmost = c[2]


@needs_qnexus
def test_submit_helios_mocked(nexus, monkeypatch):
    from hugr.qsystem.result import QsysResult

    # Helios result arrays list c[0] first: [1, 0] means c0=1, c1=0 -> key "01".
    res = QsysResult([[("c", [1, 0])]] * 4 + [[("c", [0, 1])]] * 6)
    fake = nexus(result=res)
    monkeypatch.setattr("qumo_hardware.quantinuum.hugr_program",
                        lambda circ, n: SimpleNamespace(compile=lambda: "HUGR-PACKAGE"))
    out = runner.run(BELL, "quantinuum", "Helios-1E", shots=10, confirm=True, wait=False)
    assert out["submitted"]
    (up,) = fake.called("hugr.upload")
    assert up["hugr_package"] == "HUGR-PACKAGE"
    (job_kw,) = fake.called("start_execute_job")
    cfg = job_kw["backend_config"]
    assert cfg.system_name == "Helios-1E"
    assert cfg.emulator_config is not None and cfg.emulator_config.n_qubits is None  # deprecated field unused
    assert job_kw["n_qubits"] == [2]  # the per-job-item replacement
    assert out["cost"]["unit"] == "eHQC"
    r = runner.job_result(out["job_id"])
    assert r["counts"] == {not_palindrome("01"): 4, "10": 6}


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


@needs_qnexus
def test_provider_submit_refuses_remote_without_confirm(nexus):
    """Defence in depth: calling the provider directly is gated too (QM-E957)."""
    fake = nexus()
    p, c = _compiled(BELL)
    with pytest.raises(HardwareError) as ei:
        p.submit(c, 10, "H2-1")
    assert ei.value.code == E_CONFIRM and "confirm=True" in ei.value.hint
    assert not fake.calls
    job = p.submit(c, 10, "H2-1", confirm=True)
    assert job.job_id == "nexus-job-123" and fake.called("start_execute_job")


@needs_qnexus
def test_status_and_download_errors_are_coded(nexus):
    from qumo_hardware.errors import E_JOB

    fake = nexus()
    out = runner.run(BELL, "quantinuum", "H2-1", shots=10, confirm=True, wait=False)
    fake.state = RuntimeError("upstream 502 Bearer abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJ")
    with pytest.raises(HardwareError) as ei:
        runner.job_status(out["job_id"])
    assert ei.value.code == E_JOB and "abcdefghijklmnopqrstuvwxyz0123456789" not in str(ei.value)
    assert "<redacted>" in str(ei.value)
