"""`from qumo import ...` gives the names of `qumo.prelude` (the package root re-exports them)."""

from std.testing import assert_equal, assert_true

from qumo import Circuit, sample_auto, statevector, expectation, parse_pauli_sum, parse_target, transpile, parse_qasm, to_qasm
from qumo.prelude import sample_auto as prelude_sample_auto


def test_root_names() raises:
    var c = Circuit(2)
    c.h(0)
    c.cx(0, 1)
    c.measure_all()
    var r = sample_auto(c, 1000, seed=7)
    assert_equal(r.counts.get("00", 0) + r.counts.get("11", 0), 1000)
    # the same function as through the prelude
    assert_equal(r.counts.get("00", 0), prelude_sample_auto(c, 1000, seed=7).counts.get("00", 0))
    assert_equal(statevector(c).num_qubits, 2)
    assert_true(abs(expectation(c, parse_pauli_sum("Z0 Z1")) - 1.0) < 1e-12)
    var out = transpile(c, parse_target("grid:2x2/sc_cz"))
    assert_true(len(out.circuit.ops) > 0)
    assert_equal(to_qasm(parse_qasm(to_qasm(c))), to_qasm(c))


def main() raises:
    test_root_names()
    print("test_root_import OK")
