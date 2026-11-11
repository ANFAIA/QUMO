"""Fixes in the qumo_py extension itself: docstrings, argument checks, inject_noise, extend_packed."""

from array import array

import pytest

import qumo

BELL2 = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\n'


# ------------------------------------------------------------------ docstrings
def test_help_is_useful(q):
    assert q.sample.__doc__.startswith("sample(circuit, shots, seed) -> dict")
    assert "QM-E404" in q.sample.__doc__ and "sample_device" in q.sample.__doc__
    assert q.Circuit.depth_of.__doc__.startswith("depth_of(name) -> int")
    assert "sc_cz" in q.transpile.__doc__


def test_every_function_and_method_has_a_docstring(q):
    missing = [name for name in dir(q) if not name.startswith("_") and callable(getattr(q, name))
               and not isinstance(getattr(q, name), type) and not getattr(q, name).__doc__]
    assert missing == []
    for cls in (q.Circuit, q.Program, q.QecMemory):
        methods = [m for m in dir(cls) if not m.startswith("_")]
        assert methods
        assert [m for m in methods if not getattr(cls, m).__doc__] == []
    signature_first = [name for name in dir(q) if not name.startswith("_") and callable(getattr(q, name))
                       and not isinstance(getattr(q, name), type) and not getattr(q, name).__doc__.startswith(name + "(")]
    assert signature_first == [], "the first docstring line is the call signature"


# ------------------------------------------------------------------ depth_of
def test_depth_of_type_checks(q):
    """Regression: depth_of(['cz']) stringified the list and returned 0."""
    c = q.load_qasm("OPENQASM 3.0; qubit[3] q; cz q[0], q[1]; h q[1]; cz q[1], q[2]; cx q[0], q[1];")
    assert c.depth_of("cz") == 2
    assert c.depth_of(["cz"]) == 2
    assert c.depth_of(("cz", "cx")) == 3
    assert c.depth_of("ecr") == 0  # a known gate that is absent
    for bad in (3, None, [], ["cz", 3], 1.5):
        with pytest.raises(Exception, match="QM-E911"):
            c.depth_of(bad)
    with pytest.raises(Exception, match="QM-E913.*'czz'"):
        c.depth_of("czz")


# ------------------------------------------------------------------ inject_noise
def test_inject_noise_canonical_text_is_unchanged_in_shape(q):
    text, d1, d2, xe = q.inject_noise(BELL2, 0.001, 0.01, 0.02)
    assert (d1, d2, xe) == (1, 1, 2)
    assert text == ('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\nh q[0];\n'
                    "#pragma qumo depolarize1(0.001) q[0]\ncx q[0],q[1];\n#pragma qumo depolarize2(0.01) q[0], q[1]\n"
                    "#pragma qumo x_error(0.02) q[0]\nmeasure q[0] -> c[0];\n#pragma qumo x_error(0.02) q[1]\n"
                    "measure q[1] -> c[1];\n")


def test_inject_noise_single_line_and_register_measure(q):
    """Regression: one-line QASM came back unchanged with (0, 0, 0), and `measure q -> c` got no readout noise."""
    one_line = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg c[2]; h q[0]; cx q[0],q[1]; measure q -> c;'
    text, d1, d2, xe = q.inject_noise(one_line, 0.001, 0.01, 0.02)
    assert (d1, d2, xe) == (1, 1, 2)
    assert text.count("#pragma qumo x_error(0.02)") == 2 and "x_error(0.02) q[1]\nmeasure q -> c;" in text
    noisy = q.load_qasm(text)
    assert noisy.count_ops() == {"h": 1, "depolarize1": 1, "cx": 1, "depolarize2": 1, "x_error": 2, "measure": 2}
    # the same circuit written canonically gets the same channels
    reference = q.load_qasm(q.inject_noise(BELL2, 0.001, 0.01, 0.02)[0])
    assert reference.count_ops() == noisy.count_ops()
    exact, expected = q.density_probabilities(noisy), q.density_probabilities(reference)
    assert max(abs(a - b) for a, b in zip(exact, expected)) < 1e-12 and exact[1] > 0.005  # readout flips present


def test_inject_noise_any_layout(q):
    spread = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\ncreg c[3];\n  h q[0]; h q[1];\ncx q[0],\n   q[2];  // long\nh q;\nccx q[0],q[1],q[2];\nbarrier q;\nif(c==1) x q[1];\nmeasure q -> c;\n'
    text, d1, d2, xe = q.inject_noise(spread, 0.1, 0.2, 0.3)
    assert (d1, d2, xe) == (2 + 3 + 3 + 1, 1, 3)  # h, h, h on 3 qubits, ccx as three 1q channels, conditional x
    assert "cx q[0], q[2];\n#pragma qumo depolarize2(0.2) q[0], q[2]\n" in text
    assert q.load_qasm(text).count_ops()["depolarize1"] == 9


def test_inject_noise_keeps_gate_definitions(q):
    text = ('OPENQASM 2.0;\ninclude "qelib1.inc";\ngate g2 a,b\n{\n  cx a,b;\n}\nqreg q[3];\ncreg c[3];\n'
            "barrier q[0],q[1];\nccx q[0],q[1],q[2];\ng2 q[0],q[2];\nmeasure q[2] -> c[0];\n")
    out, d1, d2, xe = q.inject_noise(text, 0.001, 0.01, 0.02)
    assert (d1, d2, xe) == (3, 1, 1)
    assert "  cx a,b;\n#pragma" not in out and "gate g2 a,b\n{\n  cx a,b;\n}\n" in out
    assert q.inject_noise("", 0.1, 0.1, 0.1) == ("\n", 0, 0, 0)
    assert q.inject_noise(BELL2, 0.0, 0.0, 0.0) == (BELL2, 0, 0, 0)


@pytest.mark.parametrize("text, what", [
    ('OPENQASM 2.0;\nqreg r[2];\nh r[0];\n', "register named q"),
    ('OPENQASM 2.0;\nqreg q[2];\nqreg a[1];\nh q[0];\n', "register named q"),
    ('OPENQASM 2.0;\nqreg q[2];\nh q[0]\n', "without its ';'"),
    ('OPENQASM 2.0;\nqreg q[2];\ncreg c[2];\ncx q, q[1];\n', "cannot place noise"),
    ('OPENQASM 2.0;\nqreg q[2];\ncreg c[2];\nh a[0];\n', "cannot place noise"),
])
def test_inject_noise_raises_instead_of_skipping(q, text, what):
    with pytest.raises(Exception) as e:
        q.inject_noise(text, 0.1, 0.1, 0.1)
    assert str(e.value).startswith("QM-E913") and what in str(e.value)


def test_inject_noise_matches_the_plugin_reference_on_compiled_text(q):
    """Text exported by QUMO (one statement per line) is transformed exactly as before."""
    c = qumo.transpile(qumo.ghz(4, measure=True), "line:4/sc_cz").circuit
    exported = c.to_qasm(2)
    out, d1, d2, xe = q.inject_noise(exported, 0.001, 0.01, 0.02)
    lines = [line for line in out.split("\n") if not line.startswith("#pragma")]
    assert "\n".join(lines) == exported  # every original line kept, in order
    ops = c.count_ops()
    assert d2 == ops["cz"] and xe == 4 and d1 == sum(v for k, v in ops.items() if k in ("rz", "sx", "x"))


# ------------------------------------------------------------------ extend_packed / op_table
def test_op_table(q):
    table = q.op_table()
    assert table["cx"][1:] == (2, 0) and table["rx"][1:] == (1, 1) and table["u"][1:] == (1, 3)
    assert table["ccx"][1] == 3 and "measure" in table
    kinds = [v[0] for v in table.values()]
    assert len(kinds) == len(set(kinds))


def test_extend_packed_validates(q):
    table = q.op_table()

    def extend(circuit, ops, params=(), names=()):
        o, p = array("q", ops), array("d", params)
        return circuit.extend_packed(o.buffer_info()[0], len(o) // 5, p.buffer_info()[0], len(p), list(names))

    c = q.circuit(2)
    assert extend(c, [table["h"][0], 0, -1, -1, -1, table["rx"][0], 1, -1, -1, -1], [0.5, 0, 2.0], ["t"]) == 2
    assert c.parameter_names() == ["t"] and "rx(2.0*$0+0.5) q1" in c.text()
    with pytest.raises(Exception, match="QM-E001"):
        extend(q.circuit(2), [table["h"][0], 2, -1, -1, -1])
    with pytest.raises(Exception, match="QM-E913"):
        extend(q.circuit(2), [999, 0, -1, -1, -1])
    with pytest.raises(Exception, match="QM-E913"):
        extend(q.circuit(2), [table["rx"][0], 0, -1, -1, -1], [])  # angle missing
    with pytest.raises(Exception, match="QM-E913"):
        extend(q.circuit(2), [table["rx"][0], 0, -1, -1, -1], [0.0, 3, 1.0])  # undeclared parameter
    with pytest.raises(Exception, match="QM-E913"):
        extend(q.circuit(2), [-1, 0, 1, -1, -1])  # condition with nothing before it
    with pytest.raises(Exception, match="QM-E913"):
        extend(q.circuit(2), [table["kraus1"][0], 0, -1, -1, -1], [0.0, -1, 0.0])
    assert extend(q.circuit(2), []) == 0


# ------------------------------------------------------------------ existing calls keep working
def test_existing_positional_calls(q):
    c = q.load_qasm(BELL2)
    assert set(q.sample(c, 100, 1)) == {"00", "11"}
    counts, method, note = q.sample_device(c, 100, 1, "auto")
    assert set(counts) == {"00", "11"} and isinstance(method, str) and note == ""
    assert q.sample_device(c, 100, 1, "cpu", precision="auto")[0] == counts
    assert abs(q.expectation(q.ghz(2), "Z0 Z1") - 1.0) < 1e-12
    assert abs(q.density_expectation(q.ghz(2), "Z0 Z1") - 1.0) < 1e-12
    out, initial, final = q.transpile(c, 2, [(0, 1)], ["rz", "sx", "x", "cz"], 2, 7)
    assert set(out.count_ops()) <= {"rz", "sx", "x", "cz", "measure"} and len(initial) == len(final) == 2
    assert q.circuit(3).num_qubits() == 3 and q.version() == qumo.__version__
    p = q.load_qasm("OPENQASM 3.0; input float t; qubit[1] q; bit[1] c; rx(t) q[0]; c[0] = measure q[0];")
    assert q.sample(p.bind([3.141592653589793]), 10, 1) == {"1": 10}
    with pytest.raises(Exception, match="QM-E913"):
        q.sample_device(c, 100, 1, "auto", shots_per_job=5)
