"""Regressions for classical conditions: each case gave a wrong answer or lost
information before; see the commit that added this file."""

from std.testing import assert_equal, assert_true, assert_raises

from qumo.ir.qasm import parse_qasm, to_qasm
from qumo.ir.qasm2 import to_qasm2
from qumo.ir.circuit import Circuit
from qumo.ir.pauli import parse_pauli_sum
from qumo.sim.runner import sample
from qumo.sim.branching import sample_branching, BranchingSimulator
from qumo.sim.rng import Rng
from qumo.sim.stabilizer import TableauSimulator
from qumo.ai.describe import circuit_to_json, circuit_from_json
from qumo.ai.json import parse_json
from qumo.exec.primitives import EstimatorPub, run_estimator


def _error_of(src: String) -> String:
    try:
        _ = parse_qasm(src)
    except e:
        return String(e)
    return ""


def test_if_body_writing_its_condition() raises:
    # x q[2] must run: the condition holds when the if is entered. The flat form
    # would test the new c[0], so the parser refuses instead of answering 000.
    var bad = String("qubit[3] q; bit[3] c; x q[0]; c[0] = measure q[0]; if (c[0]==1) { c[0] = measure q[1]; x q[2]; } c[2] = measure q[2];")
    assert_true(_error_of(bad).startswith("QM-E541"))
    # one measurement into several bits, where the later ones test the first
    assert_true(_error_of("qubit[2] q; bit[2] c; if (c[0]==1) c = measure q;").startswith("QM-E541"))
    # the else branch would run after the if branch cleared the bit
    assert_true(_error_of("qubit[3] q; bit[3] c; x q[0]; c[0] = measure q[0]; if (c[0]==1) { c[0] = measure q[1]; } else { x q[2]; }").startswith("QM-E541"))
    # the bits of a register condition count as well
    assert_true(_error_of("OPENQASM 2.0; include \"qelib1.inc\"; qreg q[2]; creg c[2]; if(c==1) { measure q[0] -> c[1]; x q[1]; }").startswith("QM-E541"))
    # a write as the last conditioned operation changes nothing that follows
    var last = parse_qasm("qubit[2] q; bit[2] c; x q[0]; c[0] = measure q[0]; if (c[0]==1) { x q[1]; c[0] = measure q[1]; } c[1] = measure q[1];")
    assert_equal(sample(last, 20, 3).counts.get("11", 0), 20)
    # writing another bit is unrelated
    var other = parse_qasm("qubit[3] q; bit[3] c; x q[0]; c[0] = measure q[0]; if (c[0]==1) { c[1] = measure q[1]; x q[2]; } c[2] = measure q[2];")
    assert_equal(sample(other, 20, 3).counts.get("101", 0), 20)


def test_int_mirror_is_a_copy() raises:
    var bad = String("qubit[2] q; bit[2] c; int d; bit[2] r; qubit[1] t; x q[0]; x q[1]; c[0] = measure q[0]; d = c; c[1] = measure q[1]; switch (d) { case 1 { x t[0]; } default { } } r[0] = measure t[0];")
    assert_true(_error_of(bad).startswith("QM-E541"))
    assert_true(_error_of("qubit[1] q; bit[1] c; int d; if (c[0]) d = c;").startswith("QM-E541"))
    var fresh = parse_qasm("qubit[2] q; bit[2] c; int d; bit[2] r; x q[0]; x q[1]; c[0] = measure q[0]; c[1] = measure q[1]; d = c; switch (d) { case 3 { r[0] = measure q[0]; } default { } }")
    assert_equal(sample(fresh, 20, 3).counts.get("0111", 0), 20)


def test_partial_register_condition_round_trip() raises:
    var c = parse_qasm("qubit[2] q; bit[2] c; bit[3] r; x q[0]; c[0] = measure q[0]; switch (c) { case 1 { x q[1]; } default { } } r[0] = measure q[1];")
    var text = to_qasm(c)
    assert_true(text.find("QM-E552") < 0)
    assert_true(text.find("if (c[0] == 1 && c[1] == 0) x q[1];") >= 0)
    assert_equal(to_qasm(parse_qasm(text)), text)
    assert_equal(sample(c, 20, 3).counts.get("00101", 0), 20)
    assert_equal(sample(parse_qasm(text), 20, 3).counts.get("00101", 0), 20)
    # the spellings a test of one bit can take
    var d = parse_qasm("qubit[2] q; bit[3] c; x q[0]; c[0] = measure q[0]; if (c[0] && !c[1] && c[2] != 1) x q[1]; c[1] = measure q[1];")
    assert_equal(sample(d, 20, 3).counts.get("011", 0), 20)
    assert_true(_error_of("qubit[1] q; bit[2] c; if (c[0] || c[1]) x q[0];").startswith("QM-E541"))
    assert_true(_error_of("qubit[1] q; bit[2] c; if (c == 1 && c[1]) x q[0];").startswith("QM-E541"))
    assert_true(_error_of("qubit[1] q; bit[2] c; bit[2] d; if (c[0] && d) x q[0];").startswith("QM-E541"))
    assert_true(_error_of("qubit[1] q; bit[2] c; if (c[0] && c[1]) x q[0]; else x q[0];").startswith("QM-E541"))
    # OpenQASM 2 has no such condition: still refused rather than dropped
    with assert_raises(contains="QM-E552"):
        _ = to_qasm2(c)


def test_json_keeps_register_conditions() raises:
    var src = String("OPENQASM 2.0; include \"qelib1.inc\"; qreg q[3]; creg c[2]; x q[0]; measure q[0] -> c[0]; if(c==3) x q[1]; measure q[1] -> c[1];")
    var c = parse_qasm(src)
    var back = circuit_from_json(parse_json(circuit_to_json(c)))
    assert_equal(sample(c, 20, 3).counts.get("01", 0), 20)
    assert_equal(sample(back, 20, 3).counts.get("01", 0), 20)
    assert_equal(to_qasm(back), to_qasm(c))
    # a one-bit condition keeps its old JSON shape
    var one = parse_qasm("qubit[2] q; bit[1] c; x q[0]; c[0] = measure q[0]; if (c[0]) x q[1];")
    assert_true(circuit_to_json(one).find("\"condition\":{\"clbit\":0,\"value\":1}") >= 0)


def test_noise_pragmas_keep_their_condition() raises:
    var pc2 = parse_qasm("qubit[2] q; bit[2] c; c[0] = measure q[0]; if (c[0]==1) { #pragma qumo pauli_channel_2(0,0,0,0,0,0,0,0,0,0,0,0,1,0,0) q[1], q[0];\n x q[0]; } c[1] = measure q[1];")
    assert_equal(sample(pc2, 20, 3).counts.get("00", 0), 20)
    var ce = parse_qasm("qubit[2] q; bit[2] c; c[0] = measure q[0]; if (c[0]==1) { #pragma qumo correlated_error(1) x q[1];\n } c[1] = measure q[1];")
    assert_equal(sample(ce, 20, 3).counts.get("00", 0), 20)
    var kr = parse_qasm("qubit[2] q; bit[2] c; c[0] = measure q[0]; if (c[0]==1) { #pragma qumo kraus1 \"bf\" (0,0,1,0,1,0,0,0) q[1];\n } c[1] = measure q[1];")
    assert_equal(sample(kr, 20, 3).counts.get("00", 0), 20)
    # the channels fire when the condition holds, also after a round trip
    var fires = String("qubit[2] q; bit[2] c; x q[0]; c[0] = measure q[0]; if (c[0]==1) { #pragma qumo correlated_error(1) x q[1];\n#pragma qumo else_correlated_error(1) z q[1];\n#pragma qumo kraus1 \"id\" (1,0,0,0,0,0,1,0) q[1];\n#pragma qumo pauli_channel_2(0,0,0,0,0,0,0,0,0,0,0,0,0,0,1e-300) q[1], q[0];\n } c[1] = measure q[1];")
    var fc = parse_qasm(fires)
    assert_equal(sample(fc, 20, 3).counts.get("11", 0), 20)
    var text = to_qasm(fc)
    assert_equal(to_qasm(parse_qasm(text)), text)
    assert_equal(sample(parse_qasm(text), 20, 3).counts.get("11", 0), 20)
    var js = circuit_from_json(parse_json(circuit_to_json(fc)))
    assert_equal(to_qasm(js), text)
    # an ELSE must not read a stale chain flag
    assert_true(_error_of("qubit[1] q; bit[1] c; #pragma qumo correlated_error(0.5) x q[0];\nif (c[0]) { #pragma qumo else_correlated_error(1) z q[0];\n }").startswith("QM-E541"))


def test_controlled_refuses_conditions() raises:
    var c = Circuit(3, 1)
    c.x(1)
    c.c_if(0, 1)
    with assert_raises(contains="QM-E202"):
        _ = c.controlled(2)


def test_skipped_measure_keeps_record_indices() raises:
    # c0 = 0, so measurement site 1 is skipped; site 2 reads the X
    var c = Circuit(2, 2)
    c.measure(0, 0)
    c.measure(1, 1)
    c.c_if(0, 1)
    c.x(1)
    var recs: List[Int] = [2]
    c.measure(1, 1)
    c.detector(recs)
    var sim = TableauSimulator(2, 1)
    sim.run(c)
    assert_equal(len(sim.records), c.num_measurements)
    assert_equal(Int(sim.records[2]), 1)
    assert_equal(Int(sim.detector_values(c)[0]), 1)
    var br = BranchingSimulator(2, 1)
    var rng = Rng(1)
    br.run(c, rng)
    assert_equal(len(br.frame.records), c.num_measurements)
    assert_equal(Int(br.frame.records[2]), 1)


def test_branching_validates() raises:
    var c = Circuit(2, 1)
    c.cx(0, 0)
    c.measure(0, 0)
    with assert_raises(contains="QM-E002"):
        _ = sample_branching(c, 5, 1)


def test_estimator_reports_the_nested_error() raises:
    var c = Circuit(1, 1)
    c.measure(0, 0)
    c.x(0)
    c.c_if(0, 1)
    var r = EstimatorPub.single(c, parse_pauli_sum("Z0"))
    var msg = String()
    try:
        _ = run_estimator(r)
    except e:
        msg = String(e)
    assert_true(msg.startswith("QM-E403"), msg)
    assert_true(msg.find("QM-E401") > 0, msg)


def main() raises:
    test_if_body_writing_its_condition()
    test_int_mirror_is_a_copy()
    test_partial_register_condition_round_trip()
    test_json_keeps_register_conditions()
    test_noise_pragmas_keep_their_condition()
    test_controlled_refuses_conditions()
    test_skipped_measure_keeps_record_indices()
    test_branching_validates()
    test_estimator_reports_the_nested_error()
    print("test_conditions OK")
