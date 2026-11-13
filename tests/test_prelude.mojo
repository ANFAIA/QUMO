"""Tests for the Mojo front door: `from qumo.prelude import *` is enough for a
first program, the state vector prints bitstrings as the CLI does, and the
result types have `to_json()`."""

from std.testing import assert_equal, assert_true, assert_false, assert_almost_equal

from qumo.prelude import *
from qumo.core.json import parse_json
from qumo.algorithms.variational import vqe, VQEResult
from qumo.algorithms.library import hardware_efficient_ansatz
from qumo.sim.runner import evolve as deep_evolve, sample as deep_sample

comptime BELL = "OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit[2] q;\nbit[2] c;\nh q[0];\ncx q[0], q[1];\nc = measure q;\n"


def test_first_program() raises:
    # parse or build
    var c = parse_qasm(BELL)
    var b = Circuit(2, 2)
    b.h(0)
    b.cx(0, 1)
    b.measure(0, 0)
    b.measure(1, 1)
    assert_equal(to_qasm(b), to_qasm(c))
    assert_true(c.draw().byte_length() > 0)
    # sample
    var r: SampleResult = sample_auto(c, 1000, seed=7)
    assert_equal(r.shots, 1000)
    assert_equal(r.counts.get("00", 0) + r.counts.get("11", 0), 1000)
    var r2 = sample(c, 1000, 7)
    assert_equal(r2.shots, 1000)
    # exact state and expectation values
    var sv: StateVector = statevector(c)
    assert_almost_equal(sv.re[0], 0.7071067811865476, atol=1e-12)
    assert_almost_equal(sv.re[3], 0.7071067811865476, atol=1e-12)
    assert_almost_equal(evolve(c).fidelity(sv), 1.0, atol=1e-12)
    var h: PauliSum = parse_pauli_sum("Z0 Z1")
    assert_almost_equal(expectation(c, h), 1.0, atol=1e-12)
    # compile for a target
    var t: Target = parse_target("grid:2x2/sc_cz")
    var cr: CompileResult = transpile(c, t)
    assert_equal(cr.circuit.num_qubits, 4)
    assert_equal(cr.circuit.num_2q_gates(), 1)
    assert_true(cr.to_json().byte_length() > 2)
    assert_equal(Op.H, Op.H)


def test_prelude_is_the_same_code() raises:
    """A prelude name is the defining module's function: same results, bit for bit."""
    var c = parse_qasm(BELL)
    var a = sample(c, 500, 11)
    var d = deep_sample(c, 500, 11)
    assert_equal(String(a), String(d))
    assert_equal(String(statevector(c)), String(deep_evolve(c)))
    # default seeds are the documented ones (1234 for sampling)
    assert_equal(String(sample_auto(c, 300)), String(sample_auto(c, 300, seed=1234)))
    assert_equal(to_qasm(transpile(c, parse_target("line:3/rz,sx,x,cz")).circuit),
                 to_qasm(transpile(c, parse_target("line:3/rz,sx,x,cz"), 2, 7).circuit))


def test_statevector_prints_bitstrings() raises:
    var text = String(statevector(parse_qasm(BELL)))
    assert_true(text.startswith("StateVector(2 qubits)"), text)
    assert_true(text.find("|00>: ") >= 0 and text.find("|11>: ") >= 0, text)
    assert_false(text.find("|0>") >= 0 or text.find("|3>") >= 0, text)
    # leftmost character = highest qubit, as in the counts and the CLI: X on qubit 0 of 3 is |001>
    var c = Circuit(3, 0)
    c.x(0)
    assert_true(String(statevector(c)).find("|001>: 1.0") >= 0, String(statevector(c)))
    var d = Circuit(3, 0)
    d.x(2)
    assert_true(String(statevector(d)).find("|100>: 1.0") >= 0, String(statevector(d)))
    # long states are cut after 16 amplitudes, and say so
    var w = Circuit(5, 0)
    for q in range(5):
        w.h(q)
    var wt = String(statevector(w))
    assert_true(wt.find("|01111>") >= 0 and wt.find("|10000>") < 0, wt)
    assert_true(wt.endswith("... (first 16 nonzero amplitudes shown)"), wt)
    var g = Circuit(4, 0)
    for q in range(4):
        g.h(q)
    assert_true(String(statevector(g)).endswith("j"), String(statevector(g)))  # exactly 16: no note


def test_sample_result_to_json() raises:
    var r = sample_auto(parse_qasm(BELL), 400, seed=3)
    var v = parse_json(r.to_json())
    assert_equal(v.get("shots").as_int(), 400)
    assert_equal(v.get("num_clbits").as_int(), 2)
    assert_equal(v.get("method").as_string(), r.method)
    var counts = v.get("counts")
    assert_equal(len(counts), len(r.counts))
    assert_equal(counts.get("00").as_int(), r.counts.get("00", 0))
    assert_equal(counts.get("11").as_int(), r.counts.get("11", 0))
    assert_equal(counts.key_at(0), "00")  # sorted keys


def test_vqe_result_to_json() raises:
    var h = parse_pauli_sum("-1.05*II + 0.39*ZI - 0.39*IZ - 0.01*ZZ + 0.18*XX")
    var r = vqe(hardware_efficient_ansatz(2, 1, "linear"), h)
    var v = parse_json(r.to_json())
    assert_almost_equal(v.get("energy").as_float(), r.energy, atol=1e-12)
    assert_equal(len(v.get("params")), len(r.params))
    assert_equal(len(v.get("param_names")), len(r.param_names))
    assert_equal(v.get("iterations").as_int(), r.nit)
    assert_equal(v.get("circuit_evaluations").as_int(), r.circuit_evals)
    assert_equal(v.get("converged").as_bool(), r.converged)
    assert_equal(v.get("optimizer").as_string(), r.optimizer)
    assert_equal(v.get("gradient").as_string(), r.grad_method)
    assert_equal(len(v.get("history")), len(r.history))
    # text fields are escaped, non-finite numbers become null
    var e = VQEResult()
    e.message = "line \"one\"\nand a \\ backslash"
    var zero = 0.0
    e.energy = 1.0 / zero
    var ev = parse_json(e.to_json())
    assert_equal(ev.get("message").as_string(), e.message)
    assert_true(ev.get("energy").is_null())


def main() raises:
    test_first_program()
    test_prelude_is_the_same_code()
    test_statevector_prints_bitstrings()
    test_sample_result_to_json()
    test_vqe_result_to_json()
    print("test_prelude OK")
