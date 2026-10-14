"""Tests for qumo.ai.algorithm_tools (`qumo_build_circuit`).

For every kind, the OpenQASM the tool returns parses back (parse_qasm) to a
circuit whose state vector equals the library builder's at random parameter
values (1e-12), with the parameters in the same order.
"""

from std.testing import assert_equal, assert_true, assert_false
from std.math import sqrt, sin, cos

from qumo.ai.json import JsonValue, parse_json
from qumo.ai.registry import full_registry
from qumo.ai.tools import ToolRegistry
from qumo.ai.algorithm_tools import build_algorithm_circuit
from qumo.ir.circuit import Circuit
from qumo.ir.qasm import parse_qasm
from qumo.ir.pauli import parse_pauli_sum
from qumo.sim.runner import evolve
from qumo.algorithms.hamiltonians import Graph, maxcut, h2_sto3g, heisenberg_xxz
from qumo.algorithms.library import (
    ghz, qft, iqft, grover_bitstrings, bernstein_vazirani, deutsch_jozsa, qaoa_maxcut_ansatz,
    hardware_efficient_ansatz, uccsd_ansatz, h2_ansatz, trotter,
)
from qumo.algorithms.qml import append_rot, append_angle_embedding
from qumo.algorithms.qkernels import kernel_circuit
from qumo.sim.runner import expectation


def _values(n: Int, seed: Int) -> List[Float64]:
    var out = List[Float64]()
    for k in range(n):
        out.append(0.3 + 0.7 * sin(Float64(k * 7 + seed) * 1.3))
    return out^


def _call(reg: ToolRegistry, args_json: String) raises -> JsonValue:
    var r = reg.call("qumo_build_circuit", parse_json(args_json))
    assert_false(r.is_error, "tool error for " + args_json + ": " + r.content_text())
    return parse_json(r.structured)


def _err(reg: ToolRegistry, args_json: String, code: String) raises:
    var r = reg.call("qumo_build_circuit", parse_json(args_json))
    assert_true(r.is_error, "expected an error for " + args_json)
    assert_true(r.structured.find(code) >= 0, "expected " + code + " for " + args_json + ", got " + r.structured)


def _same_state(a: Circuit, b: Circuit, params: List[Float64], what: String) raises:
    assert_equal(a.num_qubits, b.num_qubits, what + ": width")
    var sa = evolve(a, params)
    var sb = evolve(b, params)
    var worst = 0.0
    for i in range(sa.size()):
        worst = max(worst, abs(sa.re[i] - sb.re[i]) + abs(sa.im[i] - sb.im[i]))
    assert_true(worst <= 1e-12, what + ": max amplitude difference " + String(worst))


def _check_kind(reg: ToolRegistry, args_json: String, ref_circ: Circuit, what: String) raises -> JsonValue:
    var out = _call(reg, args_json)
    var parsed = parse_qasm(out.get("qasm").as_string())
    var n = ref_circ.num_params()
    assert_equal(out.get("num_parameters").as_int(), n, what + ": parameter count")
    assert_equal(parsed.num_params(), n, what + ": parsed parameter count")
    var names = out.get("parameters")
    for i in range(n):
        assert_equal(names.at(i).as_string(), ref_circ.param_names[i], what + ": parameter order")
        assert_equal(parsed.param_names[i], ref_circ.param_names[i], what + ": parsed parameter name")
    assert_equal(out.get("num_qubits").as_int(), ref_circ.num_qubits)
    assert_equal(out.get("two_qubit_gates").as_int(), ref_circ.num_2q_gates(), what + ": 2q gates")
    assert_equal(out.get("depth").as_int(), ref_circ.depth(), what + ": depth")
    _same_state(parsed, ref_circ, _values(n, 3), what)
    return out^


def test_every_kind_round_trips() raises:
    var reg = full_registry()
    _ = _check_kind(reg, "{\"kind\":\"ghz\",\"args\":{\"n\":5}}", ghz(5), "ghz")
    _ = _check_kind(reg, "{\"kind\":\"qft\",\"args\":{\"n\":4}}", qft(4), "qft")
    _ = _check_kind(reg, "{\"kind\":\"qft\",\"args\":{\"n\":5,\"do_swaps\":false,\"approximation_degree\":1}}", qft(5, False, 1), "aqft")
    _ = _check_kind(reg, "{\"kind\":\"iqft\",\"args\":{\"n\":4}}", iqft(4), "iqft")
    var keys: List[String] = ["1011", "0110"]
    var gout = _check_kind(reg, "{\"kind\":\"grover\",\"args\":{\"marked\":[\"1011\",\"0110\"]}}", grover_bitstrings(keys, -1, False), "grover")
    assert_true(gout.get("notes").at(0).as_string().find("iterations") >= 0)
    _ = _check_kind(reg, "{\"kind\":\"bernstein_vazirani\",\"args\":{\"secret\":\"1011\"}}", bernstein_vazirani(4, 11), "bv")
    _ = _check_kind(reg, "{\"kind\":\"bernstein_vazirani\",\"args\":{\"secret\":5,\"n\":3}}", bernstein_vazirani(3, 5), "bv int")
    _ = _check_kind(reg, "{\"kind\":\"deutsch_jozsa\",\"args\":{\"n\":3,\"balanced_mask\":5}}", deutsch_jozsa(3, 5, 0), "dj")
    var g = Graph.parse("0-1 1-2 2-3 3-0 0-2:0.5")
    var qout = _check_kind(reg, "{\"kind\":\"qaoa_maxcut\",\"args\":{\"graph\":\"0-1 1-2 2-3 3-0 0-2:0.5\",\"p\":2}}", qaoa_maxcut_ansatz(g, 2), "qaoa")
    # the observable string parses back to the MaxCut cost: same expectation value
    var obs = parse_pauli_sum(qout.get("observable").as_string())
    var c = qaoa_maxcut_ansatz(g, 2)
    var st = evolve(c, _values(4, 1))
    assert_true(abs(st.expectation(obs) - st.expectation(maxcut(g, True))) <= 1e-12, "qaoa observable")
    _ = _check_kind(reg, "{\"kind\":\"hea\",\"args\":{\"n\":3,\"layers\":2,\"entanglement\":\"circular\",\"rotations\":\"rx,ry\"}}", hardware_efficient_ansatz(3, 2, "circular", "rx,ry"), "hea")
    _ = _check_kind(reg, "{\"kind\":\"uccsd\",\"args\":{\"num_qubits\":6,\"num_electrons\":2}}", uccsd_ansatz(6, 2), "uccsd")
    var hout = _check_kind(reg, "{\"kind\":\"h2_ansatz\"}", h2_ansatz(), "h2")
    var h2 = parse_pauli_sum(hout.get("observable").as_string())
    var hp: List[Float64] = [0.0, 0.0, 0.11177]
    var e = evolve(h2_ansatz(), hp).expectation(h2)
    assert_true(abs(e - (-1.1373060357533993)) < 1e-4, "h2 energy " + String(e))
    var hz = heisenberg_xxz(4)
    var tq = String("{\"kind\":\"trotter\",\"args\":{\"hamiltonian\":\"") + String(hz) + "\",\"time\":0.7,\"steps\":3,\"order\":2}}"
    _ = _check_kind(reg, tq, trotter(hz, 0.7, 3, 2), "trotter")
    var tp = String("{\"kind\":\"trotter\",\"args\":{\"hamiltonian\":\"") + String(hz) + "\",\"steps\":2,\"param_name\":\"t\"}}"
    _ = _check_kind(reg, tp, trotter(hz, 1.0, 2, 1, -1, "t"), "trotter symbolic")


def test_measure_and_qasm2() raises:
    var reg = full_registry()
    var out = _call(reg, "{\"kind\":\"ghz\",\"args\":{\"n\":3},\"measure\":true,\"format\":\"qasm2\"}")
    assert_equal(out.get("format").as_string(), "qasm2")
    assert_true(out.get("qasm").as_string().startswith("OPENQASM 2.0;"))
    assert_equal(out.get("num_clbits").as_int(), 3)
    var c = parse_qasm(out.get("qasm").as_string())
    assert_equal(c.num_measurements, 3)
    # grover measures only the search qubits (5 search + 2 ancillas)
    var gout = _call(reg, "{\"kind\":\"grover\",\"args\":{\"marked\":[\"10110\"]},\"measure\":true}")
    assert_equal(gout.get("num_qubits").as_int(), 7)
    assert_equal(gout.get("num_clbits").as_int(), 5)
    # parametric circuits cannot be exported to OpenQASM 2
    _err(reg, "{\"kind\":\"qaoa_maxcut\",\"args\":{\"graph\":\"0-1 1-2\"},\"format\":\"qasm2\"}", "QM-E102")


def test_rejections() raises:
    var reg = full_registry()
    _err(reg, "{\"kind\":\"shor\"}", "QM-E801")
    _err(reg, "{\"args\":{}}", "QM-E801")
    _err(reg, "{\"kind\":\"ghz\",\"args\":{\"n\":3},\"confirm\":true}", "QM-E801")
    _err(reg, "{\"kind\":\"ghz\",\"args\":{\"n\":3,\"shell\":\"rm -rf /\"}}", "QM-E801")
    _err(reg, "{\"kind\":\"ghz\",\"args\":{\"n\":65}}", "QM-E802")
    _err(reg, "{\"kind\":\"ghz\",\"args\":{\"n\":2.5}}", "QM-E801")
    _err(reg, "{\"kind\":\"ghz\"}", "QM-E801")
    _err(reg, "{\"kind\":\"qft\",\"args\":{\"n\":33}}", "QM-E802")
    _err(reg, "{\"kind\":\"grover\",\"args\":{\"marked\":[\"10101010101010101\"]}}", "QM-E802")
    _err(reg, "{\"kind\":\"grover\",\"args\":{\"marked\":[\"10a\"]}}", "QM-E802")
    _err(reg, "{\"kind\":\"grover\",\"args\":{\"marked\":[]}}", "QM-E801")
    _err(reg, "{\"kind\":\"qaoa_maxcut\",\"args\":{\"graph\":\"0-1\",\"p\":11}}", "QM-E802")
    _err(reg, "{\"kind\":\"qaoa_maxcut\",\"args\":{\"graph\":\"0-70\"}}", "QM-E802")
    _err(reg, "{\"kind\":\"uccsd\",\"args\":{\"num_qubits\":14,\"num_electrons\":2}}", "QM-E802")
    _err(reg, "{\"kind\":\"hea\",\"args\":{\"n\":2,\"hamiltonian\":\"Z0 Z5\"}}", "QM-E802")
    _err(reg, "{\"kind\":\"trotter\",\"args\":{\"hamiltonian\":\"Z0\",\"steps\":0}}", "QM-E802")
    _err(reg, "{\"kind\":\"trotter\",\"args\":{}}", "QM-E801")
    _err(reg, "{\"kind\":\"ghz\",\"args\":{\"n\":3},\"format\":\"quil\"}", "QM-E801")


def _reuploading_ref(n: Int, layers: Int, nf: Int) -> Circuit:
    var c = Circuit(n, 0, "reuploading_ref")
    var w = c.parameters("w", 3 * n * layers)
    var x = c.parameters("x", nf)
    for l in range(layers):
        for q in range(n):
            append_rot(c, x[(3 * q) % nf], x[(3 * q + 1) % nf], x[(3 * q + 2) % nf], q)
            var b = 3 * (l * n + q)
            append_rot(c, w[b], w[b + 1], w[b + 2], q)
        for q in range(n - 1):
            c.cz(q, q + 1)
    return c^


def test_qml_kinds() raises:
    var reg = full_registry()
    # PennyLane data re-uploading demo circuit (1 qubit, 3 layers, 3 features) and a 2-qubit variant
    var r1 = _check_kind(reg, "{\"kind\":\"reuploading_classifier\"}", _reuploading_ref(1, 3, 3), "reuploading 1q")
    assert_equal(r1.get("observable").as_string().find("Z0") >= 0, True)
    _ = _check_kind(reg, "{\"kind\":\"reuploading_classifier\",\"args\":{\"n\":2,\"layers\":2,\"num_features\":2}}", _reuploading_ref(2, 2, 2), "reuploading 2q")
    # angle kernel: state equals the qkernels route; <|0..0><0..0|> = prod cos^2((a_i - b_i)/2) for RX
    var embed = Circuit(3, 0, "angle_embedding")
    var f = embed.parameters("x", 3)
    append_angle_embedding(embed, f, [0, 1, 2], "x")
    var kref = kernel_circuit(embed, 3)
    var kout = _check_kind(reg, "{\"kind\":\"angle_kernel\",\"args\":{\"n\":3}}", kref, "angle_kernel")
    var parsed = parse_qasm(kout.get("qasm").as_string())
    var obs = parse_pauli_sum(kout.get("observable").as_string())
    var p: List[Float64] = [0.3, -1.1, 2.0, 0.9, 0.4, -0.5]
    var want = 1.0
    for i in range(3):
        var c = cos((p[i] - p[3 + i]) / 2.0)
        want *= c * c
    var got = expectation(parsed, obs, p)
    assert_true(abs(got - want) <= 1e-12, "angle kernel value " + String(got) + " vs " + String(want))
    _err(reg, "{\"kind\":\"angle_kernel\",\"args\":{\"n\":13}}", "QM-E802")
    _err(reg, "{\"kind\":\"angle_kernel\"}", "QM-E801")
    _err(reg, "{\"kind\":\"reuploading_classifier\",\"args\":{\"num_features\":4}}", "QM-E802")
    _err(reg, "{\"kind\":\"reuploading_classifier\",\"args\":{\"weights\":[1]}}", "QM-E801")


def test_listed_in_registry() raises:
    var reg = full_registry()
    var found = False
    for n in reg.names():
        if n == "qumo_build_circuit":
            found = True
    assert_true(found, "qumo_build_circuit missing from full_registry()")
    var i = reg.find("qumo_build_circuit")
    var sch = parse_json(reg.tools[i].input_schema)
    assert_equal(sch.get("type").as_string(), "object")
    assert_equal(len(sch.get("properties").get("kind").get("enum")), 13)
    assert_false(sch.get("additionalProperties").as_bool())


def main() raises:
    test_every_kind_round_trips()
    test_measure_and_qasm2()
    test_rejections()
    test_qml_kinds()
    test_listed_in_registry()
    print("test_algorithm_tools OK")
