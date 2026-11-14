"""Regressions from the independent review of the Program compiler.

Each test names its finding. Every one of them failed before its fix: run against
commit a55628b they raise, or pass a wrong value.
"""

from std.testing import assert_equal, assert_true, assert_raises

from qumo.compiler.blockalg import dense_equiv
from qumo.compiler.pass_manager import conformance_error, transpile
from qumo.compiler.program_compile import compact_qubits, compile_program, compile_spec_program, verify_program
from qumo.compiler.target import Target
from qumo.ir.gates import Op
from qumo.ir.program import Program, SK
from qumo.ir.qasm import parse_qasm, parse_qasm_program
from qumo.sim.program_runner import sample_program


def test_b1_three_qubit_gates_compile_and_are_checked() raises:
    var names: List[String] = ["ccx q[0], q[1], q[2];", "cswap q[0], q[1], q[2];", "ccz q[0], q[1], q[2];"]
    for g in names:
        var p = parse_qasm_program("qubit[3] q; bit[1] c; if (c[0]) { " + g + " } h q[0]; " + g + " c[0] = measure q[0];")
        var a = compile_program(p, Target.all_to_all(3), 0)
        assert_true(a.blocks_checked > 0, String(a))
        var b = compile_program(p, Target.line(3), 1)
        assert_true(b.blocks_checked > 0, String(b))
        var wide = compile_program(p, Target.line(3, "rz,sx,x,cz"), 2)
        assert_equal(conformance_error(wide.program.body, Target.line(3, "rz,sx,x,cz")), String(""))


def test_b2_an_emptied_block_keeps_its_layout() raises:
    var p = parse_qasm_program("qubit[3] q; bit[1] c; if (c[0]) { cx q[0], q[1]; cx q[1], q[2]; cx q[2], q[0]; c[0] = 0; swap q[1], q[2]; }")
    var r = compile_program(p, Target.line(3), 1)
    _ = verify_program(p, r)
    var q = parse_qasm_program("qubit[3] q; bit[3] c; bit[3] f; int a = 0; a = 0; int b = 1; b = 1; switch (a) { case 0 { cx q[1], q[2]; } } if (c[0]) { cx q[0], q[1]; cx q[2], q[0]; for int i5 in [1:b + 1] { } swap q[1], q[0]; }")
    _ = compile_program(q, Target.ring(6), 1)


def test_b3_restore_on_a_disconnected_target() raises:
    var srcs: List[String] = [
        "qubit[4] q; bit[4] c; x q[0]; x q[1]; int k = 0; while (k < 2) { k += 1; cx q[0], q[2]; cx q[1], q[3]; } c[0] = measure q[0]; c[1] = measure q[1]; c[2] = measure q[2]; c[3] = measure q[3];",
        "qubit[4] q; bit[4] c; x q[0]; x q[1]; int k = 0; while (k < 2) { k += 1; cx q[2], q[0]; cx q[3], q[1]; } c[0] = measure q[0]; c[1] = measure q[1]; c[2] = measure q[2]; c[3] = measure q[3];",
    ]
    var e1: List[Int] = [0, 5, 5, 2, 1, 4, 4, 3]
    var e2: List[Int] = [0, 4, 4, 2, 1, 5, 5, 3]
    for s in srcs:
        var p = parse_qasm_program(s)
        var want = sample_program(p, 200, 3)
        for seed in range(12):
            for t in [Target.from_edges(6, e1), Target.from_edges(6, e2)]:
                var r = compile_program(p, t, 0, seed)
                assert_equal(String(sample_program(compact_qubits(r.program), 200, 3).sample), String(want.sample))


def _swap_to_identity(mut p: Program) -> Bool:
    """Turn the last SWAP of the first block that measures into I."""
    for s in p.tape:
        if s.kind != SK.OPS:
            continue
        var has_measure = False
        for i in range(s.a, s.b):
            has_measure = has_measure or p.body.ops[i].kind == Op.MEASURE
        if not has_measure:
            continue
        var i = s.b - 1
        while i >= s.a:
            if p.body.ops[i].kind == Op.SWAP:
                p.body.ops[i].kind = Op.I
                p.body.ops[i].q1 = -1
                return True
            i -= 1
    return False


def test_c1_verifier_covers_measuring_blocks() raises:
    var src = String("OPENQASM 3.0; qubit[3] q; bit[3] c; x q[0]; int k = 0; while (k < 2) { k += 1; cx q[0], q[2]; c[1] = measure q[1]; cx q[1], q[2]; cx q[0], q[1]; } c[0] = measure q[0]; c[1] = measure q[1]; c[2] = measure q[2];")
    var p = parse_qasm_program(src)
    var r = compile_program(p, Target.line(3), 0, 7)
    assert_equal(String(sample_program(compact_qubits(r.program), 200, 3).sample), "{\"101\": 200}")
    assert_true(r.blocks_unchecked == 0, String(r))
    var m = r.copy()
    assert_true(_swap_to_identity(m.program))
    assert_equal(String(sample_program(compact_qubits(m.program), 200, 3).sample), "{\"001\": 200}")
    with assert_raises(contains="QM-E631"):
        _ = verify_program(p, m)
    # l_out must follow from l_in and the block's permutation
    var bad = r.copy()
    var k = 0
    while bad.blocks[k].perm[0] == 0 and k + 1 < len(bad.blocks):
        k += 1
    var a = bad.blocks[k].perm[0]
    bad.blocks[k].perm[0] = bad.blocks[k].perm[1]
    bad.blocks[k].perm[1] = a
    with assert_raises(contains="QM-E631"):
        _ = verify_program(p, bad)
    # a wrong measurement outcome wiring is found: the clbit written changes
    var wrong = r.copy()
    for ref g in wrong.program.body.ops:
        if g.kind == Op.MEASURE and g.clbit == 1:
            g.clbit = 2
            break
    with assert_raises(contains="QM-E631"):
        _ = verify_program(p, wrong)
    # reset, noise and conditions stay out of the check
    var q = parse_qasm_program("qubit[2] q; bit[1] c; int k = 0; while (k < 1) { k += 1; h q[0]; reset q[0]; cx q[0], q[1]; c[0] = measure q[0]; }")
    var rq = compile_program(q, Target.line(2), 1)
    assert_true(rq.blocks_unchecked >= 1, String(rq))


def test_c1_huge_angles_compile_correctly() raises:
    # the new check found this: merge lost the smaller term of 1 + 2^63 and wrapped
    # what was left with no bits to spare, and transpile() did it too
    var c = parse_qasm("qubit[1] q; sx q[0]; rz(1) q[0]; p(9223372036854775807) q[0]; sx q[0]; rz(1) q[0]; p(9223372036854775807) q[0];")
    var r = transpile(c, Target.all_to_all(1), 2, 7)
    var ph = 0.0
    assert_true(dense_equiv(c.ops, r.circuit.ops, 1, ph))
    var p = parse_qasm_program("qubit[1] q; bit[1] c; for int i in [9223372036854775806:9223372036854775807] { if (c[0]) { break; } sx q[0]; rz(1) q[0]; p(i) q[0]; c[0] = measure q[0]; } const int k = 9223372036854775807; sx q[0]; rz(1) q[0]; p(k) q[0]; sx q[0]; rz(1) q[0]; p(k) q[0]; c[0] = measure q[0];")
    _ = compile_program(p, Target.heavy_hex(3), 2)


def test_open_second_route_in_a_pipeline() raises:
    var p = parse_qasm_program("qubit[3] q; bit[1] c; int k = 0; while (k < 1) { k += 1; cx q[0], q[2]; } c[0] = measure q[0];")
    with assert_raises(contains="QM-E630"):
        _ = compile_spec_program(p, Target.line(3), "route(sabre),route(sabre)")
    with assert_raises(contains="QM-E630"):
        _ = compile_spec_program(p, Target.line(3), "route(sabre),fixpoint(elide_swaps,cancel)")


def main() raises:
    test_b1_three_qubit_gates_compile_and_are_checked()
    test_b2_an_emptied_block_keeps_its_layout()
    test_b3_restore_on_a_disconnected_target()
    test_c1_verifier_covers_measuring_blocks()
    test_c1_huge_angles_compile_correctly()
    test_open_second_route_in_a_pipeline()
    print("test_program_compile_regressions_layout OK")
