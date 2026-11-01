"""Tests for qumo.ir.draw: golden drawings."""

from std.math import pi
from std.testing import assert_equal, assert_true
from qumo.ir.circuit import Circuit
from qumo.ir.kraus import amplitude_damping
from qumo.ir.qasm import parse_qasm


def test_bell() raises:
    var c = Circuit(2, 2)
    c.h(0)
    c.cx(0, 1)
    c.measure(0, 0)
    c.measure(1, 1)
    assert_equal(c.draw(), "q0: --H--@--M>c0--\n         |\nq1: -----+--M>c1--\n")


def test_labels() raises:
    var c = Circuit(3, 1)
    var th = c.parameter("theta")
    c.rz(pi / 2, 0)
    c.ry(0.1234, 1)
    c.rx(th * 2.0, 2)
    c.cp(pi / 4, 0, 2)
    c.swap(1, 2)
    c.h(0)
    c.c_if(0, 1)
    assert_equal(c.draw(), "q0: ---RZ(pi/2)-------@-----H?c0=1--\n                      |\nq1: ---RY(0.123)------|-------x-----\n                      |       |\nq2: --RX(2*theta)--P(pi/4)----x-----\n")


def test_spans_do_not_overlap() raises:
    var c = Circuit(3)
    c.cx(0, 2)
    c.h(1)
    var s = c.draw()
    # h must sit right of the vertical span of the cx
    assert_equal(s, "q0: --@-----\n      |\nq1: --|--H--\n      |\nq2: --+-----\n")


def test_noise_detectors_and_fold() raises:
    var c = Circuit(2, 1)
    c.depolarize1(0.01, 0)
    c.measure(0, 0)
    c.detector([0])
    c.barrier()
    var noisy = c.draw()
    assert_true(noisy.find("DEPOLARIZE1(0.01)") >= 0 and noisy.find("DET[0]") >= 0, noisy)
    var long = Circuit(1)
    for _ in range(50):
        long.h(0)
    var s = long.draw(width=40)
    var lines = s.split("\n")
    for l in lines:
        assert_true(l.byte_length() <= 40, "fold keeps lines short: " + String(l))
    assert_true(len(lines) > 4, "fold happened")
    assert_equal(long.draw(width=0).count("\n"), 1, "width 0 never folds")


def test_side_table_noise() raises:
    var probs = List[Float64](length=15, fill=0.01)
    var c = Circuit(3)
    c.correlated_error(0.25, [0, 2], [1, 3])
    c.pauli_channel_2(probs, 0, 1)
    c.apply_channel(amplitude_damping(0.3), 2)
    var s = c.draw()
    assert_true(s.find("E(0.25):X") >= 0 and s.find("E(0.25):Z") >= 0, s)
    assert_true(s.find("PC2") >= 0 and s.find(c.kraus[0].name) >= 0, s)
    # the product spans its qubits, so an E never shares a column with a gate between them
    var d = Circuit(3)
    d.correlated_error(0.25, [0, 2], [1, 3])
    d.h(1)
    assert_true(d.draw().find("E(0.25):X") >= 0, "column layered over the span")
    print("  E, PAULI_CHANNEL_2, Kraus labels")


def test_unicode() raises:
    var c = parse_qasm("OPENQASM 3.0; include \"stdgates.inc\"; qubit[2] q; cx q[0], q[1]; swap q[0], q[1];")
    var s = c.draw(style="unicode")
    assert_true(s.find("●") >= 0 and s.find("⊕") >= 0 and s.find("×") >= 0, "glyphs")
    var raised = False
    try:
        _ = c.draw(style="bogus")
    except e:
        raised = String(e).startswith("QM-E005")
    assert_true(raised, "QM-E005")


def main() raises:
    test_bell()
    test_labels()
    test_spans_do_not_overlap()
    test_noise_detectors_and_fold()
    test_side_table_noise()
    test_unicode()
    print("test_draw OK")
