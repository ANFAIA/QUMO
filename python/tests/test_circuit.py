"""qumo.Circuit: builder, inspection, text forms, Python protocol, loading."""

import copy
import math
import pickle
import re
from pathlib import Path

import numpy as np
import pytest

import qumo
from conftest import ROOT

BELL_QASM3 = 'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[2] q;\nbit[2] c;\nh q[0];\ncx q[0], q[1];\nc[0] = measure q[0];\nc[1] = measure q[1];\n'


def everything() -> qumo.Circuit:
    """One of every instruction the builder offers."""
    c = qumo.Circuit(4)
    c.id(0).x(0).y(1).z(2).h(3).s(0).sdg(1).t(2).tdg(3).sx(0).sxdg(1)
    c.rx(0.1, 0).ry("a", 1).rz(2 * qumo.Parameter("b") + 0.5, 2).p(-0.3, 3).u(0.1, "a", 0.3, 0)
    c.cx(0, 1).cy(1, 2).cz(2, 3).ch(3, 0).swap(0, 2).iswap(1, 3).ecr(0, 3)
    c.cp(0.2, 0, 1).crx("a", 1, 2).cry(0.4, 2, 3).crz(0.5, 3, 0).rxx(0.6, 0, 1).ryy(0.7, 1, 2).rzz("b", 2, 3)
    c.ccx(0, 1, 2).cswap(1, 2, 3).ccz(0, 2, 3)
    c.barrier()
    c.depolarize1(0.01, 0).depolarize2(0.02, 0, 1).x_error(0.03, 2).y_error(0.04, 3).z_error(0.05, 0)
    c.pauli_channel_1(0.01, 0.02, 0.03, 1)
    c.measure(0, 0).x(1).c_if(0, 1).reset(2).measure_all()
    return c


def test_bell_builder_chains_and_counts(bell):
    assert bell.num_qubits == 2 and bell.num_clbits == 2
    assert len(bell) == 4 and bell.size() == 2 and bell.depth() == 2
    assert bell.count_ops() == {"h": 1, "cx": 1, "measure": 2}
    assert bell.to_qasm() == BELL_QASM3
    assert bell.to_qasm(2).startswith("OPENQASM 2.0;")
    assert bell.has_measurements() and not bell.has_noise() and bell.is_clifford()


def test_gate_methods_mirror_the_mojo_builder():
    """Every simple gate method of qumo/ir/circuit.mojo exists on qumo.Circuit."""
    src = (ROOT / "qumo" / "ir" / "circuit.mojo").read_text()
    mojo = set(re.findall(r"^    def (\w+)\(mut self[^)]*\): self\.add\(Op\.", src, flags=re.M))
    assert {"h", "x", "y", "z", "s", "t", "rx", "ry", "rz", "cx", "cz", "swap", "ccx"} <= mojo
    mojo -= {"tick"}  # QEC timing annotation: not part of the Python builder
    missing = sorted(name for name in mojo if not callable(getattr(qumo.Circuit, name, None)))
    assert missing == [], f"Mojo builder methods without a Python twin: {missing}"
    for name in ("measure", "measure_all", "c_if", "u", "parameter", "parameters", "bind", "compose", "inverse",
                 "draw", "depth", "count_ops", "copy"):
        assert callable(getattr(qumo.Circuit, name)), name


def test_every_instruction_reaches_the_native_circuit():
    c = everything()
    ops = c.count_ops()
    for name in ("id", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx", "sxdg", "rx", "ry", "rz", "p", "u", "cx",
                 "cy", "cz", "ch", "swap", "iswap", "ecr", "cp", "crx", "cry", "crz", "rxx", "ryy", "rzz", "ccx",
                 "cswap", "ccz", "barrier", "depolarize1", "depolarize2", "x_error", "y_error", "z_error",
                 "pauli_channel_1", "measure", "reset"):
        assert ops.get(name, 0) >= 1, name
    assert c.parameter_names == ["a", "b"]
    assert c.has_noise() and c.is_dynamic()
    assert "if(c0==1) x q1" in c.text()
    assert "rz(2.0*$1+0.5) q2" in c.text()
    assert c.num_clbits == 5  # measure(0, 0), then measure_all on fresh bits 1..4


def test_matches_circuit_loaded_from_qasm():
    built = qumo.Circuit(3).h(0).cx(0, 1).rz("t", 2).ccx(0, 1, 2).measure_all()
    loaded = qumo.load(built.to_qasm())
    assert loaded == built and loaded.parameter_names == ["t"]
    assert loaded.to_qasm() == built.to_qasm()


def test_parameters_by_name_and_bind():
    c = qumo.Circuit(2)
    theta = c.parameter("theta")
    c.rx(theta, 0).ry(2 * theta + 0.5, 1).rz("phi", 0).rzz(-theta / 2, 0, 1)
    assert c.parameter_names == ["theta", "phi"] and c.num_parameters == 2
    bound = c.bind({"theta": 0.25, "phi": 1.0})
    assert bound.num_parameters == 0
    assert bound == c.bind([0.25, 1.0]) == c.bind(theta=0.25, phi=1.0) == qumo.bind(c, {"phi": 1.0, "theta": 0.25})
    assert "ry(1.0) q1" in bound.text() and "rzz(-0.125) q0, q1" in bound.text()
    assert c.num_parameters == 2  # bind returns a new circuit
    names = [p.name for p in qumo.Circuit(1).parameters("w", 3)]
    assert names == ["w[0]", "w[1]", "w[2]"]


def test_bind_errors_name_the_parameters():
    c = qumo.Circuit(1).rx("theta", 0).rz("phi", 0)
    with pytest.raises(qumo.QumoError) as e:
        c.bind({"theta": 1.0})
    assert e.value.code == "QM-E101" and "'phi'" in str(e.value)
    with pytest.raises(qumo.QumoError) as e:
        c.bind({"theta": 1.0, "phi": 2.0, "omega": 3.0})
    assert "'omega'" in str(e.value)
    with pytest.raises(qumo.QumoError) as e:
        c.bind([1.0])
    assert e.value.code == "QM-E101"
    with pytest.raises(TypeError):
        c.bind(0.5)


def test_parameter_arithmetic():
    t = qumo.Parameter("t")
    assert (2 * t + 1) == qumo.Parameter("t", 2.0, 1.0)
    assert (t - 1) * 2 == qumo.Parameter("t", 2.0, -2.0)
    assert (1 - t) == qumo.Parameter("t", -1.0, 1.0)
    assert (t + t) == qumo.Parameter("t", 2.0, 0.0)
    assert -t / 4 == qumo.Parameter("t", -0.25, 0.0)
    with pytest.raises(qumo.QumoError):
        t + qumo.Parameter("s")
    assert pickle.loads(pickle.dumps(2 * t + 1)) == 2 * t + 1


def test_mutation_after_use_extends_the_native_circuit():
    c = qumo.Circuit(2).h(0)
    assert c.depth() == 1  # materialises
    native_before = c.native
    c.cx(0, 1).measure_all()
    assert c.native is native_before  # same object, extended in place
    assert c.count_ops() == {"h": 1, "cx": 1, "measure": 2}
    assert sorted(qumo.sample(c, shots=200)) == ["00", "11"]
    c.x(0).c_if(0, 1)
    assert len(c) == 5


def test_copy_is_independent(bell):
    for other in (bell.copy(), copy.copy(bell), copy.deepcopy(bell)):
        assert other == bell and other is not bell
        other.x(0)
        assert other != bell and len(bell) == 4


def test_equality_is_structural():
    a = qumo.Circuit(2).h(0).cx(0, 1)
    assert a == qumo.Circuit(2).h(0).cx(0, 1)
    assert a != qumo.Circuit(2).cx(0, 1).h(0)
    assert a != qumo.Circuit(3).h(0).cx(0, 1)
    assert a != "h q[0]"
    with pytest.raises(TypeError):
        hash(a)


@pytest.mark.parametrize("make", [
    lambda: qumo.Circuit(2).h(0).cx(0, 1).measure_all(),
    everything,
    lambda: qumo.ghz(5, measure=True),
    lambda: qumo.hardware_efficient_ansatz(3, 2),
])
def test_pickle_round_trip(make):
    c = make()
    for protocol in (2, pickle.HIGHEST_PROTOCOL):
        back = pickle.loads(pickle.dumps(c, protocol))
        assert isinstance(back, qumo.Circuit)
        assert back == c
        assert back.parameter_names == c.parameter_names
        assert back.num_qubits == c.num_qubits and back.num_clbits == c.num_clbits
    back.h(0)  # a loaded circuit is still a builder
    assert len(back) == len(c) + 1


def test_pickled_circuit_samples_identically(bell):
    back = pickle.loads(pickle.dumps(bell))
    assert qumo.sample(back, 500, 3) == qumo.sample(bell, 500, 3)


def test_compose_and_inverse():
    a = qumo.Circuit(2).h(0).cx(0, 1)
    ident = a.compose(a.inverse())
    psi = qumo.statevector(ident)
    assert np.allclose(psi, [1, 0, 0, 0])
    assert len(a) == 2


def test_draw_and_reprs(bell):
    art = bell.draw()
    assert "q0" in art and "q1" in art and "H" in art
    assert bell.draw(style="ascii").isascii()
    assert str(bell) == art
    assert repr(bell) == "<qumo.Circuit 2 qubits, 2 clbits, 2 gates, depth 2>"
    html = bell._repr_html_()
    assert html.startswith("<div><code>&lt;qumo.Circuit 2 qubits") and "<pre" in html and "q0" in html

    class Printer:
        text_out = ""

        def text(self, s):
            self.text_out += s

    p = Printer()
    bell._repr_pretty_(p, False)
    assert p.text_out.startswith(repr(bell)) and "q1" in p.text_out
    big = qumo.Circuit(2)
    for _ in range(300):
        big.h(0).cx(0, 1)
    assert str(big) == repr(big) and "<pre" not in big._repr_html_()
    assert "parameters theta" in repr(qumo.Circuit(1).rx("theta", 0))


def test_depth_of_and_gate_statistics():
    c = qumo.Circuit(3).cz(0, 1).h(1).cz(1, 2).cx(0, 1).t(0)
    assert c.depth_of("cz") == 2
    assert c.depth_of(["cz"]) == 2
    assert c.depth_of(["cz", "cx"]) == 3
    assert c.count_gate("cz") == 2 and c.num_2q_gates() == 3 and c.t_count() == 1
    assert c.two_qubit_edges() == [(0, 1), (1, 2), (0, 1)]
    with pytest.raises(qumo.QumoError) as e:
        c.depth_of("czz")
    assert e.value.code == "QM-E913"
    with pytest.raises(TypeError):
        c.depth_of(3)
    with pytest.raises(qumo.QumoError):
        c.count_gate("nope")


def test_qubit_errors_are_raised_at_the_call_and_leave_the_circuit_intact():
    c = qumo.Circuit(2).h(0)
    with pytest.raises(qumo.QumoError) as e:
        c.h(2)
    assert e.value.code == "QM-E001" and "0..1" in e.value.hint
    with pytest.raises(qumo.QumoError) as e:
        c.cx(0, 0)
    assert e.value.code == "QM-E002"
    with pytest.raises(qumo.QumoError):
        c.cx(0, 5)
    with pytest.raises(TypeError) as e:
        c.rx(0, 0.5)  # angle and qubit swapped
    assert "angles come before qubits" in str(e.value)
    with pytest.raises(TypeError):
        c.h("0")
    with pytest.raises(TypeError):
        c.h(True)
    with pytest.raises(TypeError):
        c.rx([0.1], 0)
    with pytest.raises(TypeError):
        c.u(0.1, object(), 0.3, 0)
    with pytest.raises(qumo.QumoError):
        c.depolarize1(1.5, 0)
    with pytest.raises(qumo.QumoError):
        c.c_if(0, 1)  # no classical bit yet
    with pytest.raises(qumo.QumoError):
        c.measure(0, -1)
    # nothing half-appended: the circuit is still H followed by what we add now
    c.rx(0.5, 1).cx(0, 1).measure_all()
    assert c.count_ops() == {"h": 1, "rx": 1, "cx": 1, "measure": 2}
    assert "rx(0.5) q1" in c.text()


def test_numpy_integers_are_qubits():
    c = qumo.Circuit(3)
    for q in np.arange(2):
        c.cx(q, q + 1)
    c.rx(np.float64(0.5), np.int64(2)).measure(np.int32(0), np.int64(1))
    assert c.count_ops() == {"cx": 2, "rx": 1, "measure": 1}


def test_constructor_errors():
    with pytest.raises(TypeError) as e:
        qumo.Circuit("bell.qasm")
    assert "qumo.load" in str(e.value)
    with pytest.raises(qumo.QumoError):
        qumo.Circuit(0)
    with pytest.raises(TypeError):
        qumo.Circuit(2.0)


def test_load_detects_text_and_paths(tmp_path, bell):
    assert qumo.load(BELL_QASM3) == bell
    assert qumo.load("OPENQASM 3.0; qubit[2] q; bit[2] c; h q[0]; cx q[0], q[1]; c = measure q;") == bell
    path = tmp_path / "bell.qasm"
    path.write_text(BELL_QASM3)
    assert qumo.load(path) == bell  # pathlib.Path
    assert qumo.load(str(path)) == bell  # str path
    assert qumo.load_qasm(str(path)) == bell  # load_qasm given a path just works
    qasm2 = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\nmeasure q -> c;\n'
    assert qumo.load(qasm2) == bell


def test_load_missing_file_and_bad_text():
    with pytest.raises(FileNotFoundError) as e:
        qumo.load_qasm("file.qasm")
    assert "file.qasm" in str(e.value) and "OpenQASM text" in str(e.value)
    with pytest.raises(FileNotFoundError):
        qumo.load(Path("nope") / "x.qasm")
    with pytest.raises(qumo.QumoError) as e:
        qumo.load("OPENQASM 3.0; qubit[1] q; foo q[0];")
    assert e.value.code == "QM-E510" and "foo" in e.value.message
    with pytest.raises(TypeError):
        qumo.load(42)


def test_example_circuits_load():
    files = sorted((ROOT / "examples").rglob("*.qasm"))[:5]
    for f in files:
        try:
            c = qumo.load(f)
        except qumo.QumoError as e:  # programs with run-time control flow are not one Circuit
            assert e.code.startswith("QM-E5"), (f, e)
            continue
        assert c.num_qubits >= 1


def test_from_native_and_native_round_trip(q, bell):
    raw = q.load_qasm(BELL_QASM3)
    wrapped = qumo.Circuit.from_native(raw)
    assert wrapped == bell and wrapped.native is raw
    assert q.sample(bell.native, 100, 1) == q.sample(raw, 100, 1)
    with pytest.raises(TypeError):
        qumo.Circuit.from_native("x")


def test_stim_round_trip():
    c = qumo.Circuit(2).h(0).cx(0, 1).depolarize2(0.01, 0, 1).measure_all()
    back = qumo.from_stim(c.to_stim())
    assert back.count_ops() == c.count_ops()


def test_large_build_is_bulk():
    n, layers = 20, 200
    c = qumo.Circuit(n)
    for layer in range(layers):
        for i in range(n):
            c.ry(0.1 * i, i)
        for i in range(layer % 2, n - 1, 2):
            c.cx(i, i + 1)
    assert c._nat.num_ops() == 0  # nothing crossed into the extension yet
    assert len(c) == layers * n + sum(len(range(layer % 2, n - 1, 2)) for layer in range(layers))
    assert math.isclose(float(np.abs(qumo.statevector(qumo.Circuit(1).ry(math.pi, 0))[1])), 1.0)
