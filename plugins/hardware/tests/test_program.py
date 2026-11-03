"""Programs from Python, and the hardware plugins refusing them.

A program with classical variables or control flow at run time simulates through
qumo_py. Nothing of the kind reaches a provider yet: the plugins refuse it by name.
"""

from __future__ import annotations

import json

import pytest

from qumo_hardware.base import TargetSpec
from qumo_hardware.compile import compile_for_target, load_qumo_py
from qumo_hardware.errors import E_COMPILE, HardwareError

RUS = """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
output uint[4] tries;
uint[4] k = 0;
while (!c[0] && k < 7) {
  reset q[0];
  h q[0];
  c[0] = measure q[0];
  k = k + 1;
}
tries = k;
"""
BELL = 'OPENQASM 3.0; include "stdgates.inc"; qubit[2] q; bit[2] c; h q[0]; cx q[0], q[1]; c = measure q;'


@pytest.fixture
def q():
    return load_qumo_py()


def _spec() -> TargetSpec:
    return TargetSpec(name="fake_line", provider="local", num_qubits=4, basis=["rz", "sx", "x", "cx"],
                      edges=[(0, 1), (1, 2), (2, 3)])


def test_load_program(q):
    p = q.load_program(RUS)
    assert (p.num_qubits(), p.num_clbits(), p.num_parameters()) == (2, 2, 0)
    assert p.has_runtime_regions()
    assert "while (" in p.to_qasm3()
    assert not q.load_program(BELL).has_runtime_regions()


def test_to_circuit(q):
    flat = q.load_program(BELL).to_circuit()
    assert flat.num_qubits() == 2
    with pytest.raises(Exception, match="QM-E585"):
        q.load_program(RUS).to_circuit()


def test_sample_program(q):
    r = q.sample_program(q.load_program(RUS), 4000, 5)
    assert sum(r["counts"].values()) == 4000
    assert len(r["iterations"]) == 4000
    assert r["outputs"]["tries"] == r["iterations"]
    assert 1.9 < sum(r["iterations"]) / 4000 < 2.1  # capped Geom(1/2)
    assert r["warnings"] == []
    # the seed fixes the run
    assert q.sample_program(q.load_program(RUS), 4000, 5)["counts"] == r["counts"]


def test_sample_program_inputs_and_iterations(q):
    p = q.load_program('OPENQASM 3.0; input float t; output float r; float a = 0.0; a = t; r = a * 2;')
    assert q.sample_program(p, 2, 1, params=[1.5])["outputs"]["r"] == [3.0, 3.0]
    stuck = q.load_program("OPENQASM 3.0; int k = 0; while (k < 100) { k += 1; }")
    with pytest.raises(Exception, match="QM-E480"):
        q.sample_program(stuck, 1, 1, max_iterations=10)
    assert q.sample_program(stuck, 1, 1, max_iterations=100)["iterations"] == [100]


def test_sample_program_methods(q):
    p = q.load_program(RUS)
    assert q.sample_program(p, 10, 1, method="tableau")["method"].startswith("stabilizer")
    r = q.sample_program(p, 10, 1, method="gpu")
    assert r["warnings"] and r["warnings"][0].startswith("QM-W020")
    with pytest.raises(Exception, match="QM-E483"):
        q.sample_program(p, 10, 1, method="density")
    with pytest.raises(Exception, match="QM-E913"):
        q.sample_program(p, 10, 1, shots=3)


def test_transpile_program(q):
    p = q.load_program(RUS)
    out, initial, final, report = q.transpile_program(p, 4, [(0, 1), (1, 2), (2, 3)], ["rz", "sx", "x", "cx"], 2, 7)
    assert out.num_qubits() == 4
    assert len(initial) == len(final) >= 2
    assert report["blocks_checked"] + report["blocks_unchecked"] >= 1
    assert "while (" in out.to_qasm3()
    a = q.sample_program(p, 4000, 5)["counts"]
    b = q.sample_program(out, 4000, 5)["counts"]
    tv = sum(abs(a.get(k, 0) - b.get(k, 0)) for k in set(a) | set(b)) / 8000
    assert tv < 0.03


def test_program_json(q):
    p = q.load_program(RUS)
    text = p.to_json()
    assert json.loads(text)["format"] == "qumo.program.v1"
    back = q.program_from_json(text)
    assert back.to_qasm3() == p.to_qasm3()
    assert q.sample_program(back, 100, 2)["counts"] == q.sample_program(p, 100, 2)["counts"]
    with pytest.raises(Exception, match="QM-E587"):
        q.program_from_json('{"format": "nope"}')


def test_draw_and_lint(q):
    p = q.load_program(RUS)
    assert "while (" in q.draw_program(p, 80, "ascii")
    assert p.lint() == []
    stuck = q.load_program("OPENQASM 3.0; int k = 0; while (k < 3) { }")
    assert [code for code, _ in stuck.lint()] == ["QM-W054"]


def test_lint_has_the_warnings_of_the_parser(q):
    flat = q.load_program('OPENQASM 3.0; include "stdgates.inc"; qubit q; float t = 3 / 4 * pi; rz(t) q;')
    assert [code for code, _ in flat.lint()] == ["QM-W055"]
    loop = q.load_program('OPENQASM 3.0; include "stdgates.inc"; qubit q; rz(1 / 2) q; int k = 0; while (k < 3) { }')
    assert [code for code, _ in loop.lint()] == ["QM-W055", "QM-W054"]


def test_plugin_refuses_program_text():
    with pytest.raises(HardwareError) as e:
        compile_for_target(RUS, _spec())
    assert e.value.code == E_COMPILE
    assert "run time" in str(e.value) and "hardware" in str(e.value)


def test_plugin_refuses_program_file(tmp_path):
    f = tmp_path / "rus.qasm"
    f.write_text(RUS)
    with pytest.raises(HardwareError) as e:
        compile_for_target(str(f), _spec())
    assert e.value.code == E_COMPILE and "hardware" in str(e.value)


def test_plugin_refuses_program_object(q):
    with pytest.raises(HardwareError) as e:
        compile_for_target(q.load_program(RUS), _spec())
    assert e.value.code == E_COMPILE and "hardware" in str(e.value)


def test_plugin_takes_a_program_that_is_a_circuit(q):
    assert compile_for_target(q.load_program(BELL), _spec()).num_logical == 2
    # program syntax, no runtime regions: flat parser refuses, the program parser folds it
    text = 'OPENQASM 3.0; include "stdgates.inc"; qubit[2] q; bit[2] c; rz(int[8](3)) q[0]; c = measure q;'
    assert compile_for_target(text, _spec()).num_logical == 2


def test_plugin_still_reports_a_parse_error():
    with pytest.raises(HardwareError) as e:
        compile_for_target("OPENQASM 3.0; qubit[1] q; hh q[0];", _spec())
    assert e.value.code == E_COMPILE and "QM-E510" in str(e.value)
