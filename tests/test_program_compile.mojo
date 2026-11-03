"""The program compiler: the compiled distribution equals the source on real topologies."""

from std.math import pi
from std.os import listdir
from std.testing import assert_equal, assert_true, assert_raises

from qumo.compiler.pass_manager import conformance_error
from qumo.compiler.program_compile import compile_program, compile_spec_program, compact_qubits, program_metrics, verify_program
from qumo.compiler.target import Target
from qumo.ir.circuit import Circuit
from qumo.ir.gates import Op, op_arity, op_is_unitary
from qumo.ir.program import Program, SK
from qumo.ir.qasm import parse_qasm_program, to_qasm, load_any
from qumo.sim.program_runner import sample_program
from qumo.sim.runner import SampleResult


def _read(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def _tv(a: SampleResult, b: SampleResult) -> Float64:
    var total = 0.0
    for k in a.sorted_keys():
        total += abs(a.probability(k) - b.probability(k))
    for k in b.sorted_keys():
        if a.counts.get(k, 0) == 0:
            total += b.probability(k)
    return total / 2.0


def _conforms(p: Program, t: Target) raises:
    assert_equal(conformance_error(p.body, t), String(""))


def _same_distribution(src: String, t: Target, level: Int, seed: Int = 7) raises:
    """Exit test. Threshold fixed before the first run: TV < 0.04 on 12000 shots
    (two samples of up to 16 outcomes differ by about 0.02)."""
    var p = parse_qasm_program(src)
    var r = compile_program(p, t, level, seed)
    _conforms(r.program, t)
    var a = sample_program(p, 12000, 31)
    var b = sample_program(compact_qubits(r.program), 12000, 37)
    var tv = _tv(a.sample, b.sample)
    if tv >= 0.04:
        print("TV", tv, t.name, "level", level, src)
        print(a.sample)
        print(b.sample)
    assert_true(tv < 0.04)
    assert_true(abs(a.mean_iterations() - b.mean_iterations()) < 0.15)
    # the text of a compiled program is a program
    var again = parse_qasm_program(to_qasm(r.program))
    assert_equal(len(again.tape), len(r.program.tape))


def _corpus() -> List[String]:
    var s = List[String]()
    # repeat until success on entangled pairs; the body needs SWAPs
    s.append("qubit[4] q; bit[4] c; h q[0]; cx q[0], q[3]; cx q[1], q[2]; while (!c[0]) { reset q[3]; h q[3]; cx q[3], q[1]; cx q[1], q[0]; c[0] = measure q[3]; cx q[0], q[2]; } c[1] = measure q[0]; c[2] = measure q[1]; c[3] = measure q[2];")
    # break and continue from nested ifs, layouts must match on those edges
    s.append("qubit[4] q; bit[4] c; int i; h q[0]; cx q[0], q[1]; cx q[1], q[2]; cx q[2], q[3]; for int i in [0:5] { cx q[0], q[3]; c[0] = measure q[0]; if (c[0]) { cx q[1], q[3]; break; } cx q[2], q[0]; c[1] = measure q[1]; if (c[1]) { cx q[3], q[1]; continue; } cx q[0], q[1]; } c[2] = measure q[2]; c[3] = measure q[3];")
    # if and else moving qubits differently, then gates that see the layout
    s.append("qubit[5] q; bit[3] c; h q[0]; c[0] = measure q[0]; if (c[0]) { cx q[0], q[4]; cx q[1], q[3]; } else { cx q[2], q[4]; cx q[0], q[1]; h q[3]; } cx q[4], q[2]; cx q[3], q[0]; c[1] = measure q[2]; c[2] = measure q[0];")
    # switch, one case empty, a loop inside a case
    s.append("qubit[4] q; bit[4] c; h q[0]; h q[1]; c[0:1] = measure q[0:1]; switch (c[0:1]) { case 0 { cx q[0], q[3]; } case 1 { int k = 0; while (k < 2) { cx q[1], q[3]; cx q[0], q[2]; k = k + 1; } } case 2 { } default { x q[2]; cx q[2], q[3]; } } c[2] = measure q[2]; c[3] = measure q[3];")
    # runtime angles through routing
    s.append("qubit[3] q; bit[3] c; float a; a = pi / 4; int i; i = 0; while (i < 3) { rx(a) q[0]; cx q[0], q[2]; cx q[2], q[1]; rz(2 * a) q[1]; a = a + pi / 8; i = i + 1; } c[0] = measure q[0]; c[1] = measure q[1]; c[2] = measure q[2];")
    return s^


def test_exit_distribution() raises:
    var src = _corpus()
    var targets = List[Target]()
    targets.append(Target.line(6))
    targets.append(Target.grid(2, 3))
    targets.append(Target.heavy_hex(3))
    for ref t in targets:
        for s in src:
            for level in [1, 2]:
                _same_distribution(s, t, level)
    # basis translation with routing, and the heaviest level
    var sc = Target.superconducting_heavy_hex(3)
    var g = Target.superconducting_grid(2, 3)
    for s in src:
        _same_distribution(s, sc, 2)
        _same_distribution(s, g, 3)
        _same_distribution(s, Target.line(6), 0)


def test_qiskit_fixtures_compile() raises:
    for nm in listdir("tests/fixtures/control_flow"):
        if nm.endswith(".qasm"):
            var p = load_any(_read("tests/fixtures/control_flow/" + nm))
            var t = Target.line(p.num_qubits() + 2)
            var r = compile_program(p, t, 2)
            _conforms(r.program, t)


def test_vf2_embedding_needs_no_swaps() raises:
    # a line program on a line: VF2 embeds the summary, so no block moves a qubit
    var p = parse_qasm_program("qubit[4] q; bit[1] c; while (!c[0]) { h q[0]; cx q[0], q[1]; cx q[1], q[2]; cx q[2], q[3]; c[0] = measure q[3]; }")
    var r = compile_program(p, Target.line(4), 2)
    assert_equal(r.restored_swaps, 0)
    for g in r.program.body.ops:
        assert_true(g.kind != Op.SWAP)


def test_restore_needs_swaps_and_is_counted() raises:
    # a triangle in a loop on a line: the body must move a qubit and put it back
    var p = parse_qasm_program("qubit[3] q; bit[1] c; int i; i = 0; while (i < 2) { cx q[0], q[1]; cx q[1], q[2]; cx q[2], q[0]; i = i + 1; } c[0] = measure q[0];")
    var r = compile_program(p, Target.line(3), 1)
    assert_true(r.restored_swaps > 0)
    assert_true(r.blocks_checked > 0, String(r))


def test_verifier_catches_a_wrong_program() raises:
    var p = parse_qasm_program("qubit[3] q; bit[1] c; int i; i = 0; while (i < 2) { cx q[0], q[1]; cx q[1], q[2]; cx q[2], q[0]; i = i + 1; } if (c[0]) { cx q[0], q[2]; } c[0] = measure q[0];")
    var r = compile_program(p, Target.line(3), 0)
    _ = verify_program(p, r)
    # a block that does something else
    var bad = r.copy()
    var at = -1
    for i in range(len(bad.program.body.ops)):
        if op_is_unitary(bad.program.body.ops[i].kind) and op_arity(bad.program.body.ops[i].kind) == 2:
            at = i
    bad.program.body.ops[at].kind = Op.CZ
    with assert_raises(contains="QM-E631"):
        _ = verify_program(p, bad)
    # an edge that joins two layouts
    var moved = r.copy()
    assert_true(len(moved.blocks) > 2)
    var a = moved.blocks[0].l_out[0]
    moved.blocks[0].l_out[0] = moved.blocks[0].l_out[1]
    moved.blocks[0].l_out[1] = a
    with assert_raises(contains="QM-E631"):
        _ = verify_program(p, moved)
    # a different structure
    var other = r.copy()
    var k = 0
    while other.program.tape[k].kind != SK.WHILE:
        k += 1
    other.program.tape[k].kind = SK.IF
    with assert_raises(contains="QM-E631"):
        _ = verify_program(p, other)


def test_rejections() raises:
    var p = parse_qasm_program("qubit[2] q; bit[1] c; while (!c[0]) { h q[0]; c[0] = measure q[0]; }")
    with assert_raises(contains="QM-E630"):
        _ = compile_program(p, Target.line(3), 2, ai=True)
    with assert_raises(contains="QM-E630"):
        _ = compile_spec_program(p, Target.line(3), "clifford_synth,route(sabre)")
    with assert_raises(contains="QM-E605"):
        _ = compile_program(p, Target.line(1), 2)
    # a runtime angle cannot be wrapped into bounds
    var t = Target.line(3, "rz,sx,x,cz,rzz")
    t.set_angle_bounds(Op.RZZ, 0.0, pi / 2)
    var b = parse_qasm_program("qubit[2] q; bit[1] c; float a; a = 3.0; rzz(a) q[0], q[1]; c[0] = measure q[0];")
    with assert_raises(contains="QM-E633"):
        _ = compile_program(b, t, 2)


def test_passes_excluded() raises:
    # elide_swaps would relabel across blocks: a SWAP in a body must stay physical
    var p = parse_qasm_program("qubit[2] q; bit[2] c; int i; i = 0; x q[0]; while (i < 1) { swap q[0], q[1]; i = i + 1; } c[0] = measure q[0]; c[1] = measure q[1];")
    var r = compile_program(p, Target.all_to_all(2), 2)
    assert_equal(sample_program(r.program, 50, 1).sample.counts.get("10", 0), 50)


def test_metrics() raises:
    var p = parse_qasm_program("qubit[2] q; bit[1] c; h q[0]; cx q[0], q[1]; c[0] = measure q[0]; if (c[0]) { x q[0]; x q[1]; } else { x q[1]; } for int i in [0:2] { c[0] = measure q[0]; if (c[0]) break; h q[0]; cx q[0], q[1]; h q[1]; } int k; k = 0; while (k < 4) { x q[0]; k += 1; }")
    var m = program_metrics(p)
    assert_equal(m.base_gates, 4)  # h, cx, and the costlier branch
    assert_equal(m.base_2q, 1)
    assert_equal(len(m.loops), 2)
    assert_equal(m.loops[0].gates, 3)
    assert_equal(m.loops[0].two_q, 1)
    assert_equal(m.loops[0].iterations, 3)
    assert_equal(m.loops[1].gates, 1)
    assert_equal(m.loops[1].iterations, -1)
    var n: List[Int] = [3, 4]
    assert_equal(m.gates_for(n), 4 + 9 + 4)
    var r = compile_program(p, Target.line(2), 2)
    assert_true(r.metrics.base_gates >= 1)


def test_flat_programs_use_transpile() raises:
    var p = load_any("qubit[3] q; bit[3] c; h q[0]; cx q[0], q[2]; c = measure q;")
    var r = compile_program(p, Target.line(3), 2)
    assert_true(not r.program.has_runtime_region())


def main() raises:
    test_rejections()
    test_metrics()
    test_flat_programs_use_transpile()
    test_vf2_embedding_needs_no_swaps()
    test_restore_needs_swaps_and_is_counted()
    test_passes_excluded()
    test_verifier_catches_a_wrong_program()
    test_qiskit_fixtures_compile()
    test_exit_distribution()
    print("test_program_compile OK")
