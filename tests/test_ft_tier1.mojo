"""Tests for the reichardt QEC style, Iceberg Pauli exponentials and the Helix code."""

from std.math import pi
from std.testing import assert_equal, assert_true
from qumo.ir.circuit import Circuit
from qumo.ir.gates import Op
from qumo.sim.stabilizer import TableauSimulator
from qumo.ft.pauli import SymPauli
from qumo.ft.code import helix_code, helix_10_2_3_code, code_by_name, iceberg_code, steane_code
from qumo.ft.gadgets import (
    iceberg_rotation, iceberg_exp_kind, steane_reichardt_cycle, steane_syndrome_supports, steane_encoder_zero,
)
from qumo.ft.verify import valid_clifford_implementation
from qumo.ft.arch import Architecture, SteaneBuilder, QECPolicy
from qumo.ft.encode import encode
from qumo.ft.logical import LOp
from qumo.ft.emulate import run_encoded, reference_counts, total_variation
from qumo.qec.codes import NoiseParams


def test_helix() raises:
    var small = helix_10_2_3_code()
    small.validate()
    assert_equal(small.css_distance(), 3, "[[10,2,3]]")
    var h = helix_code()
    h.validate()
    assert_equal(h.n, 20, "n")
    assert_equal(h.k, 2, "k")
    assert_equal(len(h.generators), 18, "rank 18")
    assert_equal(h.css_distance(), 6, "[[20,2,6]]: no logical below weight 6")
    assert_equal(code_by_name("helix").name, "helix", "by name")
    print("  Helix [[20,2,6]] and [[10,2,3]]: commuting, rank, k, d")


def sem_exp(k: Int, paulis: String, theta: Float64) -> Circuit:
    """Rotation exp(-i theta/2 P) by a ladder; paulis[i] acts on logical qubit i."""
    var c = Circuit(k)
    var qs = List[Int]()
    var bs = paulis.as_bytes()
    for i in range(k):
        var l = Int(bs[i])
        if l == ord("I"):
            continue
        qs.append(i)
        if l == ord("X"):
            c.h(i)
        elif l == ord("Y"):
            c.rx(pi / 2, i)
    var last = qs[len(qs) - 1]
    for i in range(len(qs) - 1):
        c.cx(qs[i], last)
    c.rz(theta, last)
    var j = len(qs) - 2
    while j >= 0:
        c.cx(qs[j], last)
        j -= 1
    for i in range(k):
        var l = Int(bs[i])
        if l == ord("X"):
            c.h(i)
        elif l == ord("Y"):
            c.rx(-pi / 2, i)
    return c^


def test_iceberg_exponentials() raises:
    # theta = pi/2 is Clifford: the verifier checks each gadget on the code exactly
    for k in [2, 4, 6]:
        var code = iceberg_code(k)
        var words = List[String]()
        words.append(String("X") * k)
        words.append(String("Y") * k)
        words.append(String("Z") * k)
        for i in range(k):
            var xi = String("")
            var zi = String("")
            var xz = String("")
            var zx = String("")
            for j in range(k):
                xi += "I" if j == i else "X"
                zi += "I" if j == i else "Z"
                xz += "Z" if j == i else "X"
                zx += "X" if j == i else "Z"
            words.append(xi)
            words.append(zi)
            words.append(xz)
            words.append(zx)
        for w in words:
            var kind = iceberg_exp_kind(w)
            var report = valid_clifford_implementation(sem_exp(k, w, pi / 2), iceberg_rotation(code.n, kind[0], kind[1], -1, pi / 2), code)
            assert_true(report.ok, "gadget for " + w + " on k = " + String(k))
    print("  seven Iceberg Pauli exponentials verified for k = 2, 4, 6")


def expectation(mut sim: TableauSimulator, p: SymPauli) raises -> Int:
    var qs = List[Int]()
    var ls = List[Int]()
    for q in range(p.n):
        var x = p.x(q)
        var z = p.z(q)
        if x == 1 or z == 1:
            qs.append(q)
            ls.append(2 if x == 1 and z == 1 else (1 if x == 1 else 3))
    return sim.expectation_sparse(qs, ls)


def test_reichardt_corrects_every_single_error() raises:
    var sup = steane_syndrome_supports()
    var code = steane_code()
    for q in range(7):
        for letter in range(2):
            var c = Circuit(10, 6)
            c.compose(steane_encoder_zero())
            if letter == 0:
                c.x(q)
            else:
                c.z(q)
            c.compose(steane_reichardt_cycle())
            for d in range(7):
                var vals = List[Int]()
                for s in range(3):
                    var inside = 0
                    for x in sup[s]:
                        if x == d:
                            inside = 1
                    vals.append(inside)
                var zbits: List[Int] = [0, 4, 5]
                var xbits: List[Int] = [3, 1, 2]
                c.z(d)
                c.c_if_register(zbits, vals)
                c.x(d)
                c.c_if_register(xbits, vals)
            var sim = TableauSimulator(10, 3)
            sim.run(c)
            # every stabilizer and Z_L back to +1
            for g in code.generators:
                assert_equal(expectation(sim, g), 1, "stabilizer " + g.label() + " after error on " + String(q))
            assert_equal(expectation(sim, code.z_logicals[0]), 1, "logical state kept")
    print("  every single X and Z data error is corrected by the 3-bit table")


def test_reichardt_style_encoded() raises:
    var c = Circuit(1, 1)
    c.h(0)
    c.s(0)
    c.h(0)
    c.measure(0, 0)
    var plain = SteaneBuilder().with_qec_policy(QECPolicy("steane", 0.5)).build(2)
    var dense = SteaneBuilder().with_qec_policy(QECPolicy("reichardt", 0.5)).build(2)
    plain.qec.costs[LOp.H] = 1.0
    dense.qec.costs[LOp.H] = 1.0
    var p_plain = encode(plain, c)
    var p_dense = encode(dense, c)
    assert_true(p_dense.circuit.num_qubits < p_plain.circuit.num_qubits, "3 ancillas instead of a logical block: " + String(p_dense.circuit.num_qubits) + " vs " + String(p_plain.circuit.num_qubits))
    var refc = reference_counts(c, 4000, 5)
    var enc = run_encoded(p_dense, 4000, 6)
    assert_true(total_variation(refc, enc) < 0.04, "noiseless logical action")
    print("  reichardt style:", p_dense.circuit.num_qubits, "qubits against", p_plain.circuit.num_qubits, "for steane")


def test_branching_past_the_statevector_limit() raises:
    # four Steane blocks and a T: wider than 26 qubits with the magic block
    var c = Circuit(4, 4)
    c.h(0)
    c.t(0)
    c.h(0)
    c.cx(0, 1)
    c.cx(1, 2)
    c.cx(2, 3)
    for q in range(4):
        c.measure(q, q)
    var prog = encode(SteaneBuilder().build(5), c)
    assert_true(prog.circuit.num_qubits > 26, "wider than the trajectories: " + String(prog.circuit.num_qubits))
    var r = run_encoded(prog, 400, 7)
    assert_equal(r.method, "branching", "auto picks branching")
    var refc = reference_counts(c, 8000, 5)
    assert_true(total_variation(refc, r) < 0.1, "matches the unencoded circuit")
    print("  encoded T program on", prog.circuit.num_qubits, "qubits runs on branching")


def main() raises:
    test_helix()
    test_iceberg_exponentials()
    test_reichardt_corrects_every_single_error()
    test_reichardt_style_encoded()
    test_branching_past_the_statevector_limit()
    print("test_ft_tier1 OK")
