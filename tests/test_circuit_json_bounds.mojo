"""Hostile values inside the instructions of a qumo.circuit.v1 document.

circuit_from_json bounds the sizes a document declares (QM-E911); this file holds
the values an instruction carries: every index, record, count and probability is
checked against the document or a stated limit, and a refusal is quick and has a
code. Valid documents still load (the round trips in test_mcp.mojo and
test_conditions.mojo, and the cases at the end of this file).
"""

from std.testing import assert_equal, assert_true

from qumo.ai.describe import circuit_from_json, circuit_to_json
from qumo.core.json import parse_json
from qumo.ir.circuit import Circuit

comptime INT_MAX = "9223372036854775807"
comptime HEAD = "{\"num_qubits\":2,\"num_clbits\":2,\"instructions\":["


def _doc(instructions: String) -> String:
    return HEAD + instructions + "]}"


def _said(text: String) -> String:
    """The error loading says, or "" when the document loads."""
    try:
        _ = circuit_from_json(parse_json(text))
    except e:
        return String(e)
    return ""


def _refused(text: String, why: String) raises:
    var said = _said(text)
    assert_true(said.startswith("QM-E911"), why + ": " + said)


def _refused_qm(text: String, why: String) raises:
    """Refused with a code of the Circuit builder it reaches (QM-E732, QM-E414)."""
    var said = _said(text)
    assert_true(said.startswith("QM-E"), why + ": " + said)


def test_the_two_reported_documents() raises:
    # a negative clbit used to load and return counts {"": 1024}
    _refused("{\"num_qubits\":1,\"num_clbits\":1,\"instructions\":[{\"op\":\"measure\",\"qubits\":[0],\"clbit\":-3}]}", "negative clbit")
    # an observable index of 1e11 made num_observables 1e11 and hung the simulator
    _refused("{\"num_qubits\":1,\"num_clbits\":1,\"instructions\":[{\"op\":\"x_error\",\"qubits\":[0],\"params\":[0.1]},{\"op\":\"measure\",\"qubits\":[0],\"clbit\":0},{\"op\":\"observable\",\"index\":99999999999,\"records\":[-1]}]}", "huge observable")


def test_measure_clbit() raises:
    _refused(_doc("{\"op\":\"measure\",\"qubits\":[0],\"clbit\":" + INT_MAX + "}"), "INT_MAX clbit")
    _refused(_doc("{\"op\":\"measure\",\"qubits\":[0],\"clbit\":-9223372036854775807}"), "most negative clbit")
    _refused(_doc("{\"op\":\"measure\",\"qubits\":[0],\"clbit\":100000}"), "clbit past the room")
    assert_equal(_said(_doc("{\"op\":\"measure\",\"qubits\":[0],\"clbit\":5}")), "")  # the register still grows


def test_observable_index() raises:
    var m = "{\"op\":\"measure\",\"qubits\":[0],\"clbit\":0},"
    _refused(_doc(m + "{\"op\":\"observable\",\"index\":-1,\"records\":[0]}"), "negative index")
    _refused(_doc(m + "{\"op\":\"observable\",\"index\":64,\"records\":[0]}"), "index 64: the masks hold 64")
    _refused(_doc(m + "{\"op\":\"observable\",\"index\":" + INT_MAX + ",\"records\":[0]}"), "INT_MAX index")
    assert_equal(_said(_doc(m + "{\"op\":\"observable\",\"index\":63,\"records\":[0]}")), "")


def test_record_indices() raises:
    var m = "{\"op\":\"measure\",\"qubits\":[0],\"clbit\":0},"
    # absolute: 0 .. measurements so far; negative counts back
    _refused(_doc(m + "{\"op\":\"detector\",\"records\":[1]}"), "a record that does not exist yet")
    _refused(_doc("{\"op\":\"detector\",\"records\":[0]}"), "a record before any measurement")
    _refused(_doc(m + "{\"op\":\"detector\",\"records\":[-2]}"), "a look-back past the first measurement")
    _refused(_doc(m + "{\"op\":\"detector\",\"records\":[" + INT_MAX + "]}"), "INT_MAX record")
    _refused(_doc(m + "{\"op\":\"detector\",\"records\":[-9223372036854775807]}"), "most negative record")
    _refused(_doc(m + "{\"op\":\"observable\",\"index\":0,\"records\":[7]}"), "an observable record past the end")
    _refused(_doc(m + "{\"op\":\"detector\",\"records\":3}"), "records that is not an array")
    var back = circuit_from_json(parse_json(_doc(m + "{\"op\":\"measure\",\"qubits\":[1],\"clbit\":1},{\"op\":\"detector\",\"records\":[-1,0]}")))
    assert_equal(back.records[0], 1)  # rec[-1] is the second measurement
    assert_equal(back.records[1], 0)


def test_qubit_indices() raises:
    _refused(_doc("{\"op\":\"h\",\"qubits\":[-1]}"), "negative qubit")
    _refused(_doc("{\"op\":\"h\",\"qubits\":[2]}"), "qubit past the register")
    _refused(_doc("{\"op\":\"cx\",\"qubits\":[0," + INT_MAX + "]}"), "INT_MAX qubit")
    _refused(_doc("{\"op\":\"measure\",\"qubits\":[-1],\"clbit\":0}"), "measure of a negative qubit")
    _refused(_doc("{\"op\":\"correlated_error\",\"probability\":0.1,\"paulis\":[[5,1]]}"), "correlated error on a missing qubit")
    _refused(_doc("{\"op\":\"correlated_error\",\"probability\":0.1,\"paulis\":[[-1,1]]}"), "correlated error on a negative qubit")


def test_conditions() raises:
    var x = "{\"op\":\"x\",\"qubits\":[0],\"condition\":"
    _refused(_doc(x + "{\"clbit\":-1,\"value\":1}}"), "negative condition bit: it used to mean no condition")
    var said = _said(_doc(x + "{\"clbit\":2,\"value\":1}}"))  # Circuit.validate's own code
    assert_true(said.startswith("QM-E003"), "condition bit past the register: " + said)
    _refused(_doc(x + "{\"clbit\":" + INT_MAX + ",\"value\":1}}"), "INT_MAX condition bit")
    _refused(_doc(x + "{\"clbit\":0,\"value\":2}}"), "condition value 2 can never hold")
    _refused(_doc(x + "{\"clbit\":0,\"value\":-1}}"), "negative condition value")
    _refused(_doc(x + "{\"bits\":[0,-1],\"values\":[1,1]}}"), "negative bit in a register condition")
    _refused(_doc(x + "{\"bits\":[0,1],\"values\":[1,5]}}"), "register value 5")
    _refused(_doc(x + "{\"bits\":[0," + INT_MAX + "],\"values\":[1,1]}}"), "INT_MAX register bit")
    _refused(_doc(x + "{\"bits\":[0,1],\"values\":[1]}}"), "bits and values differ in length")
    _refused(_doc(x + "7}"), "a condition that is not an object")
    assert_equal(_said(_doc(x + "{\"clbit\":1,\"value\":0}}")), "")
    assert_equal(_said(_doc(x + "{\"bits\":[0,1],\"values\":[1,0]}}")), "")


def test_noise_probabilities() raises:
    _refused(_doc("{\"op\":\"x_error\",\"qubits\":[0],\"params\":[1.5]}"), "p above 1")
    _refused(_doc("{\"op\":\"z_error\",\"qubits\":[0],\"params\":[-0.1]}"), "negative p")
    _refused(_doc("{\"op\":\"depolarize1\",\"qubits\":[0],\"params\":[2]}"), "depolarize1 p = 2")
    _refused(_doc("{\"op\":\"depolarize2\",\"qubits\":[0,1],\"params\":[-1]}"), "depolarize2 p < 0")
    _refused(_doc("{\"op\":\"pauli_channel_1\",\"qubits\":[0],\"params\":[0.5,0.5,0.5]}"), "pauli_channel_1 adds past 1")
    _refused(_doc("{\"op\":\"pauli_channel_1\",\"qubits\":[0],\"params\":[0.1,-0.1,0.1]}"), "pauli_channel_1 negative")
    _refused(_doc("{\"op\":\"correlated_error\",\"probability\":1.5,\"paulis\":[[0,1]]}"), "correlated p above 1")
    _refused(_doc("{\"op\":\"correlated_error\",\"probability\":-0.5,\"paulis\":[[0,1]]}"), "correlated p below 0")
    var zeros = "0,0,0,0,0,0,0,0,0,0,0,0,0,0"
    _refused(_doc("{\"op\":\"pauli_channel_2\",\"qubits\":[0,1],\"probabilities\":[1.5," + zeros + "]}"), "pauli_channel_2 entry above 1")
    _refused(_doc("{\"op\":\"pauli_channel_2\",\"qubits\":[0,1],\"probabilities\":[-0.5," + zeros + "]}"), "pauli_channel_2 negative")
    _refused(_doc("{\"op\":\"pauli_channel_2\",\"qubits\":[0,1],\"probabilities\":[0.6,0.6,0,0,0,0,0,0,0,0,0,0,0,0,0]}"), "pauli_channel_2 adds past 1")
    assert_equal(_said(_doc("{\"op\":\"x_error\",\"qubits\":[0],\"params\":[1]}")), "")
    assert_equal(_said(_doc("{\"op\":\"pauli_channel_1\",\"qubits\":[0],\"params\":[0.2,0.3,0.5]}")), "")


def test_tables() raises:
    _refused(_doc("{\"op\":\"correlated_error\",\"probability\":0.1,\"paulis\":[[0,0]]}"), "Pauli letter 0")
    _refused(_doc("{\"op\":\"correlated_error\",\"probability\":0.1,\"paulis\":[[0,4]]}"), "Pauli letter 4")
    _refused(_doc("{\"op\":\"correlated_error\",\"probability\":0.1,\"paulis\":[[0," + INT_MAX + "]]}"), "INT_MAX Pauli letter")
    _refused_qm(_doc("{\"op\":\"correlated_error\",\"probability\":0.1,\"paulis\":[]}"), "no Paulis")
    _refused(_doc("{\"op\":\"correlated_error\",\"probability\":0.1,\"paulis\":[[0]]}"), "a Pauli with no letter")
    _refused_qm(_doc("{\"op\":\"pauli_channel_2\",\"qubits\":[0,1],\"probabilities\":[0,0]}"), "pauli_channel_2 table of 2")
    # a 1-qubit Kraus matrix is 2 x 2 complex = 8 numbers, a 2-qubit one 32
    _refused(_doc("{\"op\":\"kraus1\",\"qubits\":[0],\"matrices\":[[1,0,0,0]]}"), "a short Kraus matrix")
    _refused(_doc("{\"op\":\"kraus1\",\"qubits\":[0],\"matrices\":[[1,0,0,0,0,0,1,0,9,9]]}"), "a long Kraus matrix")
    _refused(_doc("{\"op\":\"kraus2\",\"qubits\":[0,1],\"matrices\":[[1,0,0,0,0,0,1,0]]}"), "a 1-qubit matrix on 2 qubits")
    _refused_qm(_doc("{\"op\":\"kraus1\",\"qubits\":[0],\"matrices\":[]}"), "no Kraus operators")
    _refused(_doc("{\"op\":\"kraus1\",\"qubits\":[0],\"matrices\":[[1,0,0,0,0,0,1,0],3]}"), "a matrix that is not an array")


def test_valid_documents_are_unchanged() raises:
    var c = Circuit(3, 0, "ok")
    c.h(0)
    c.cx(0, 1)
    c.x_error(0.1, 2)
    c.measure(0, 0)
    c.measure(1, 1)
    c.measure(2, 2)
    c.x(2)
    c.c_if(1, 1)
    c.detector([0, 1])
    c.detector([1, 2])
    c.observable(0, [2])
    c.observable(63, [0])
    var js = circuit_to_json(c)
    var back = circuit_from_json(parse_json(js))
    assert_equal(circuit_to_json(back), js)
    assert_equal(back.num_observables, 64)


def main() raises:
    test_the_two_reported_documents()
    test_measure_clbit()
    test_observable_index()
    test_record_indices()
    test_qubit_indices()
    test_conditions()
    test_noise_probabilities()
    test_tables()
    test_valid_documents_are_unchanged()
    print("test_circuit_json_bounds OK")
