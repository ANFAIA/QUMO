"""The Python gaps of docs/PYTHON.md: qumo.emulate, complex coefficients in expectation,
Kraus channels in the Circuit builder."""

import math
import subprocess
import json
from pathlib import Path

import numpy as np
import pytest

import qumo
from conftest import ROOT

LINE3 = "line:3/u,cx"  # 0-1-2: cx(0, 2) is not an edge, so it must be routed


# ----------------------------------------------------------------------------- qumo.emulate

def test_emulate_runs_as_written_when_the_circuit_fits():
    fits = qumo.Circuit(2).u(0.1, 0.2, 0.3, 0).cx(0, 1).measure_all()
    out = qumo.emulate(fits, "line:2/u,cx", shots=400, seed=5)
    assert isinstance(out, qumo.Emulation)
    assert out.compiled is False and out.num_qubits == 2
    assert out.target == qumo.target_info("line:2/u,cx")["name"]
    assert out.counts.shots == 400 and out.counts.seed == 5 and sum(out.counts.values()) == 400
    assert out.counts.method == out.method and 0.0 < out.estimated_fidelity <= 1.0
    assert "estimated_fidelity" in repr(out)


def test_emulate_compiles_when_needed_and_says_so():
    far = qumo.Circuit(3).h(0).cx(0, 2).measure_all()
    out = qumo.emulate(far, LINE3, shots=200, seed=3)
    assert out.compiled is True and out.num_qubits == 3
    assert sum(out.counts.values()) == 200


def test_emulate_compile_false_refuses_a_circuit_that_does_not_fit():
    far = qumo.Circuit(3).h(0).cx(0, 2).measure_all()
    with pytest.raises(qumo.QumoError) as e:
        qumo.emulate(far, LINE3, compile=False)
    assert e.value.code == "QM-E615"
    assert qumo.emulate(far, LINE3, compile=True, shots=50).compiled is True


def test_emulate_same_seed_same_counts():
    far = qumo.Circuit(3).h(0).cx(0, 2).measure_all()
    a = qumo.emulate(far, LINE3, shots=300, seed=11)
    b = qumo.emulate(far, LINE3, shots=300, seed=11)
    assert a.counts == b.counts and a.method == b.method
    assert qumo.emulate(far, LINE3, shots=300).counts.seed == 1234  # the CLI default


def test_emulate_defaults_equal_the_cli():
    import inspect

    sig = inspect.signature(qumo.emulate)
    assert sig.parameters["shots"].default == 1024
    assert sig.parameters["seed"].default == 1234
    assert sig.parameters["level"].default == 2


def test_emulate_argument_errors():
    bell = qumo.Circuit(2).h(0).cx(0, 1).measure_all()
    with pytest.raises(TypeError):
        qumo.emulate(bell, 5)
    with pytest.raises(TypeError):
        qumo.emulate(bell, "line:2/u,cx", shots=10.5)
    with pytest.raises(qumo.QumoError) as e:
        qumo.emulate(bell, "line:2/u,cx", shots=0)
    assert e.value.code == "QM-E913"
    with pytest.raises(qumo.QumoError) as e:
        qumo.emulate(qumo.Circuit(1).rx("theta", 0).measure_all(), "line:2/u,cx")
    assert e.value.code == "QM-E101"


def test_emulate_matches_the_cli_for_the_same_seed(tmp_path):
    cli = ROOT / "build" / "qumo"
    if not cli.exists():
        pytest.skip("build/qumo is not built (make cli)")
    far = qumo.Circuit(3).h(0).cx(0, 2).measure_all()
    src = tmp_path / "far.qasm"
    src.write_text(far.to_qasm(3))
    run = subprocess.run([str(cli), "emulate", str(src), "--target", LINE3, "--shots", "300", "--seed", "11",
                          "--json"], capture_output=True, text=True, check=True)
    cli_counts = json.loads(run.stdout)["counts"]
    py = qumo.emulate(far, LINE3, shots=300, seed=11)
    assert dict(py.counts) == cli_counts


# ----------------------------------------------------- complex coefficients (Qiskit to_list)

def test_expectation_accepts_qiskit_to_list_coefficients():
    bell = qumo.Circuit(2).h(0).cx(0, 1)
    real = qumo.expectation(bell, [("ZZ", 1.0), ("XX", 0.5)])
    assert real == pytest.approx(1.5)
    # SparsePauliOp.to_list() gives complex coefficients: zero imaginary part matches the real result
    assert qumo.expectation(bell, [("ZZ", 1.0 + 0j), ("XX", 0.5 + 0j)]) == pytest.approx(real, abs=1e-12)
    assert qumo.expectation(bell, [("ZZ", np.complex128(1.0)), ("XX", np.complex128(0.5))]) == pytest.approx(
        real, abs=1e-12)
    assert qumo.expectation(bell, {"ZZ": 1.0 + 0j, "XX": 0.5 + 0j}) == pytest.approx(real, abs=1e-12)
    _, grad = qumo.value_and_gradient(qumo.Circuit(1).ry("theta", 0), [("Z", 1.0 + 0j)], {"theta": 0.5})
    assert grad.shape == (1,)


def test_expectation_complex_coefficient_with_imaginary_part_raises():
    bell = qumo.Circuit(2).h(0).cx(0, 1)
    with pytest.raises(qumo.QumoError) as e:
        qumo.expectation(bell, [("ZZ", 1.0 + 0.5j)])
    assert e.value.code == "QM-E913"
    assert "ZZ" in str(e.value) and "real" in str(e.value)
    with pytest.raises(qumo.QumoError):
        qumo.expectation(bell, [("ZZ", 1.0), ("XX", 0.25j)])
    with pytest.raises(qumo.QumoError):
        qumo.batch_value_and_gradient(qumo.Circuit(1).ry("t", 0), [("Z", 1j)], [[0.1]])


# ----------------------------------------------------------------- Kraus channels in Circuit

AMP_DAMP = [[[1, 0], [0, math.sqrt(1 - 0.3)]], [[0, math.sqrt(0.3)], [0, 0]]]
FLIP2 = [[[math.sqrt(0.75), 0, 0, 0], [0, math.sqrt(0.75), 0, 0], [0, 0, math.sqrt(0.75), 0],
          [0, 0, 0, math.sqrt(0.75)]],
         [[0, 0, 0, math.sqrt(0.25)], [0, 0, math.sqrt(0.25), 0], [0, math.sqrt(0.25), 0, 0],
          [math.sqrt(0.25), 0, 0, 0]]]


def test_kraus1_population_matches_the_channel():
    p = qumo.probabilities(qumo.Circuit(1).x(0).kraus1("damp", AMP_DAMP, 0))
    assert p == pytest.approx([0.3, 0.7], abs=1e-12)


def test_kraus2_flip_channel():
    p = qumo.probabilities(qumo.Circuit(2).kraus2("flip", FLIP2, 0, 1))
    assert p == pytest.approx([0.75, 0.0, 0.0, 0.25], abs=1e-12)


def test_kraus_channel_round_trips_to_qasm():
    c = qumo.Circuit(2).h(0).x(1).kraus1("damp", AMP_DAMP, 0).cx(0, 1).measure_all()
    text = c.to_qasm(3)
    assert '#pragma qumo kraus1 "damp"' in text
    assert text.index("h q[0]") < text.index("kraus1") < text.index("cx q[0], q[1]")
    back = qumo.load(text)
    assert back.to_qasm(3) == text
    assert np.allclose(qumo.probabilities(back), qumo.probabilities(c), atol=1e-9)


def test_kraus_builder_errors():
    c = qumo.Circuit(2)
    with pytest.raises(qumo.QumoError) as e:  # not trace preserving
        c.kraus1("bad", [[[0.5, 0], [0, 0.5]]], 0)
    assert e.value.code == "QM-E414"
    with pytest.raises(qumo.QumoError) as e:  # a 2x2 operator on two qubits
        c.kraus2("bad", AMP_DAMP, 0, 1)
    assert e.value.code == "QM-E414"
    with pytest.raises(qumo.QumoError) as e:  # qubit outside the circuit
        c.kraus1("damp", AMP_DAMP, 5)
    assert e.value.code == "QM-E001"
    with pytest.raises(qumo.QumoError) as e:
        c.kraus2("flip", FLIP2, 1, 1)
    assert e.value.code == "QM-E002"
    with pytest.raises(qumo.QumoError) as e:  # a quote would break the OpenQASM pragma
        c.kraus1('a"b', AMP_DAMP, 0)
    assert e.value.code == "QM-E913"
    with pytest.raises(qumo.QumoError):
        c.kraus1("empty", [], 0)
    with pytest.raises(TypeError):
        c.kraus1("damp", 5, 0)
