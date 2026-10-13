"""Wide classical registers on Quantinuum (regression: > 32 clbits used to fail with QM-E955).

pytket's OpenQASM parser defaults to registers of at most 32 bits and H2
programs allow at most 63 bits per register. QUMO payloads have one register
`c` with one bit per user clbit, so the plugin raises the parser limit and
splits `c` into 63-bit registers for H2. These tests cover 33, 63, 64 and 98
bits: dry run (plan), payload conversion, and a bit-order round trip through a
mocked Nexus whose fake device evaluates the *uploaded* payload.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from conftest import h2_device, helios_device_for, needs_pytket, needs_qnexus, x_pattern

from qumo_hardware import runner
from qumo_hardware.errors import E_INVALID_PAYLOAD, E_LIMIT, HardwareError
from qumo_hardware.quantinuum import H2_MAX_CREG_WIDTH, clbit_registers

pytestmark = needs_pytket

# (device, n_qubits, extra re-measurements) -> total clbits = n_qubits + extra
CASES = {
    "33": ("H2-1", 33, 0),
    "63": ("H2-1", 40, 23),
    "64": ("H2-1", 40, 24),
    "98-h2": ("H2-1", 49, 49),
    "98-helios": ("Helios-1", 98, 0),
}


def _case(name):
    device, nq, extra = CASES[name]
    ones = {0, 1, 4, nq - 2}  # asymmetric pattern: a reversed key cannot match
    src, expected = x_pattern(nq, ones, extra)
    return device, nq + extra, src, expected


def test_register_layout_rule():
    assert clbit_registers(33, "H2") == [("c", 33)]
    assert clbit_registers(63, "H2") == [("c", 63)]
    assert clbit_registers(64, "H2") == [("c_0", 63), ("c_1", 1)]
    assert clbit_registers(98, "H2") == [("c_0", 63), ("c_1", 35)]
    assert clbit_registers(127, "H2") == [("c_0", 63), ("c_1", 63), ("c_2", 1)]
    assert clbit_registers(98, "Helios") == [("c", 98)]  # bool arrays, no integer-register limit


@pytest.mark.parametrize("name", sorted(CASES))
def test_dry_run_and_payload(name):
    device, n_cl, src, _ = _case(name)
    p = runner.plan(src, "quantinuum", device, shots=100)
    assert p.compiled.num_clbits == n_cl
    layout = [tuple(x) for x in p.report["clbit_registers"]]
    regs = sorted((r.name, r.size) for r in p.payload.c_registers)
    assert regs == sorted(layout) and sum(w for _, w in layout) == n_cl
    if device.startswith("H2"):
        assert all(w <= H2_MAX_CREG_WIDTH for _, w in regs)
        assert (len(regs) > 1) == (n_cl > H2_MAX_CREG_WIDTH)
    else:
        assert regs == [("c", n_cl)]
    from pytket.circuit import OpType

    assert p.payload.n_gates_of_type(OpType.Measure) == n_cl
    d = p.to_dict()
    assert d["validation"]["native"] and d["cost"]["unit"] == "HQC"


@pytest.mark.parametrize("name", ["33", "64"])
def test_cli_dry_run_json(name, tmp_path, capsys):
    from qumo_hardware.cli import EXIT_OK, main

    device, n_cl, src, _ = _case(name)
    f = tmp_path / "wide.qasm"
    f.write_text(src)
    code = main(["run", str(f), "-p", "quantinuum", "-b", device, "--shots", "10", "--dry-run", "--json"])
    out = capsys.readouterr()
    assert code == EXIT_OK, out.err
    d = json.loads(out.out)
    assert d["compiled"]["num_clbits"] == n_cl and d["dry_run"] and not d["submitted"]


@needs_qnexus
@pytest.mark.parametrize("name", sorted(CASES))
def test_bit_order_round_trip_mocked_nexus(name, nexus, monkeypatch):
    device, n_cl, src, expected = _case(name)
    if device.startswith("Helios"):
        native = {}

        def fake_hugr(circ, n):
            native["circ"] = circ
            return SimpleNamespace(compile=lambda: "HUGR-PACKAGE")

        monkeypatch.setattr("qumo_hardware.quantinuum.hugr_program", fake_hugr)
        fake = nexus(device=lambda pkg, shots: helios_device_for(native["circ"])(pkg, shots))
    else:
        fake = nexus(device=h2_device)
    out = runner.run(src, "quantinuum", device, shots=7, confirm=True, wait=False)
    assert out["submitted"]
    if device.startswith("H2"):
        regs = sorted((r.name, r.size) for r in fake.uploaded.c_registers)
        assert all(w <= H2_MAX_CREG_WIDTH for _, w in regs)
    res = runner.job_result(out["job_id"])
    assert res["counts"] == {expected: 7}
    assert res["num_clbits"] == n_cl


def test_condition_on_register_wider_than_h2_limit_is_refused():
    src, _ = x_pattern(40, {0}, 24)  # 64 clbits
    src = src.replace("measure q[0] -> c[0];", "measure q[0] -> c[0];\nif(c==1) x q[1];")
    with pytest.raises(HardwareError) as ei:
        runner.plan(src, "quantinuum", "H2-1", shots=10)
    assert ei.value.code == E_LIMIT and "63" in ei.value.message
    assert "--shots" not in ei.value.hint  # cause-specific hint, not the shot/cost one


def test_condition_within_h2_limit_is_accepted():
    from pytket.circuit import OpType

    dyn = ('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\nx q[0];\nmeasure q[0] -> c[0];\n'
           'if(c==1) x q[1];\nmeasure q[1] -> c[1];\n')
    p = runner.plan(dyn, "quantinuum", "H2-1", shots=10)
    assert p.payload.n_gates_of_type(OpType.Conditional) == 1


@pytest.mark.parametrize("device", ["Helios-1", "Helios-1E", "selene"])
def test_helios_refuses_conditions_before_upload(device):
    dyn = ('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\nx q[0];\nmeasure q[0] -> c[0];\n'
           'if(c==1) x q[1];\nmeasure q[1] -> c[1];\n')
    with pytest.raises(HardwareError) as ei:
        runner.plan(dyn, "quantinuum", device, shots=10)
    assert ei.value.code == E_INVALID_PAYLOAD
    assert "Conditional" in ei.value.message and "H2" in ei.value.hint


def test_pytket_bridge_error_hint_is_not_provider_transpile():
    from qumo_hardware.qasm_bridge import to_pytket

    with pytest.raises(HardwareError) as ei:
        to_pytket('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nfoo q[0];\n')
    assert ei.value.code == E_INVALID_PAYLOAD
    assert "--provider-transpile" not in ei.value.hint and "--show-qasm" in ei.value.hint


@pytest.mark.slow
def test_h2_local_emulator_with_split_registers():
    """End to end through the real pytket-quantinuum local emulator (70 clbits -> c_0[63], c_1[7])."""
    pytest.importorskip("pytket_pecos")
    src, expected = x_pattern(10, {0, 3, 4}, 60)
    out = runner.run(src, "quantinuum", "H2-1LE", shots=5, seed=2)
    assert out["finishing"]["clbit_registers"] == [["c_0", 63], ["c_1", 7]]
    assert out["counts"] == {expected: 5}


@pytest.mark.slow
def test_selene_with_more_than_32_clbits():
    pytest.importorskip("guppylang")
    src, expected = x_pattern(8, {1, 2, 6}, 30)  # 38 clbits, single register c
    out = runner.run(src, "local", "selene", shots=4, seed=2)
    assert out["counts"] == {expected: 4}
