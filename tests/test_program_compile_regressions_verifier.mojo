"""Regressions from the second review of the compiler for programs: findings 6, 8,
9, 10 and 11. Every test but the valid-reorder pin raises or fails on commit 5e1944f.
"""

from std.math import pi
from std.testing import assert_equal, assert_true

from qumo.compiler.common import reduce_angle
from qumo.compiler.pass_manager import transpile
from qumo.compiler.program_compile import ProgramCompileResult, compact_qubits, compile_program, verify_program
from qumo.compiler.target import Target
from qumo.ir.classical import CN
from qumo.ir.gates import Op
from qumo.ir.param import Param
from qumo.ir.program import Program, SK, Stmt
from qumo.ir.qasm import parse_qasm, parse_qasm_program
from qumo.sim.program_runner import sample_program
from qumo.sim.runner import SampleResult, sample

comptime TAMPER = "OPENQASM 3.0; qubit[3] q; bit[3] c; x q[0]; int k = 0; while (k < 2) { k += 1; cx q[0], q[2]; c[1] = measure q[1]; cx q[1], q[2]; cx q[0], q[1]; c[2] = measure q[2]; } if (c[2]) { x q[1]; cx q[1], q[0]; } else { h q[1]; } c[0] = measure q[0]; c[1] = measure q[1];"


def _tv(a: SampleResult, b: SampleResult) -> Float64:
    var total = 0.0
    for k in a.sorted_keys():
        total += abs(a.probability(k) - b.probability(k))
    for k in b.sorted_keys():
        if a.counts.get(k, 0) == 0:
            total += b.probability(k)
    return total / 2.0


def _caught(p: Program, m: ProgramCompileResult, code: String = "QM-E631") raises:
    var message = String()
    try:
        _ = verify_program(p, m)
    except e:
        message = String(e)
    assert_true(message.startswith(code), "expected " + code + ", got: " + message)


def _measure(r: ProgramCompileResult, nth: Int) -> Int:
    var n = 0
    for s in r.program.tape:
        if s.kind == SK.OPS:
            for i in range(s.a, s.b):
                if r.program.body.ops[i].kind == Op.MEASURE:
                    if n == nth:
                        return i
                    n += 1
    return -1


def _base() raises -> Tuple[Program, ProgramCompileResult]:
    var p = parse_qasm_program(TAMPER)
    return (p.copy(), compile_program(p, Target.line(4), 0, 7))


def test_tampering_still_fires() raises:
    var pair = _base()
    ref p = pair[0]
    ref r = pair[1]
    _ = verify_program(p, r)
    # two final measures of different clbits on different qubits: the same meaning
    var m1 = r.copy()
    var a = _measure(m1, 2)
    var b = _measure(m1, 3)
    var tmp = m1.program.body.ops[a]
    m1.program.body.ops[a] = m1.program.body.ops[b]
    m1.program.body.ops[b] = tmp
    _ = verify_program(p, m1)
    var m2 = r.copy()
    m2.program.body.ops[_measure(m2, 0)].clbit = 0
    _caught(p, m2)
    var m3 = r.copy()
    var i3 = _measure(m3, 2)
    m3.program.body.ops[i3].q0 = (m3.program.body.ops[i3].q0 + 1) % 3
    _caught(p, m3)
    var m4 = r.copy()
    m4.program.body.ops[_measure(m4, 0)].kind = Op.BARRIER
    _caught(p, m4)
    var m9 = r.copy()
    m9.program.body.ops[_measure(m9, 3)].q0 = 3
    _caught(p, m9)
    var m10 = r.copy()
    m10.blocks[1].src_lo = m10.blocks[0].src_lo
    _caught(p, m10)


def test_a_small_angle_does_not_hide() raises:
    var src = "OPENQASM 3.0; qubit[3] q; bit[3] c; int k = 0; while (k < 2) { k += 1; h q[0]; rz(0.3) q[1]; cx q[0], q[1]; rz(0.7) q[2]; c[0] = measure q[2]; }"
    var p = parse_qasm_program(src)
    var r = compile_program(p, Target.line(4), 0, 7)
    _ = verify_program(p, r)
    var m = r.copy()
    for ref g in m.program.body.ops:
        if g.kind == Op.RZ:
            g.p0 = Param(g.p0.value() + 1e-5)
            break
    _caught(p, m)


def test_8_every_source_block_has_its_compiled_block() raises:
    var pair = _base()
    ref p = pair[0]
    ref r = pair[1]
    # a whole branch dropped, with its record
    var m6 = r.copy()
    var k = 0
    var at = -1
    var blk = -1
    var seen_if = False
    for i in range(len(m6.program.tape)):
        var s = m6.program.tape[i]
        if s.kind == SK.IF:
            seen_if = True
        if s.kind == SK.OPS:
            if seen_if and at < 0:
                at = i
                blk = k
            k += 1
    var out = List[Stmt]()
    for i in range(len(m6.program.tape)):
        if i == at:
            continue
        var s = m6.program.tape[i]
        if s.end > at:
            s.end -= 1
        if s.alt > at:
            s.alt -= 1
        out.append(s)
    m6.program.tape = out^
    _ = m6.blocks.pop(blk)
    _caught(p, m6)
    # the block emptied, with a record that says it has no source
    var m7 = r.copy()
    var j = 0
    for i in range(len(m7.program.tape)):
        ref s = m7.program.tape[i]
        if s.kind == SK.OPS:
            if j == 2:
                m7.blocks[j].src_lo = -1
                m7.blocks[j].src_hi = -1
                s.b = s.a
            j += 1
    _caught(p, m7)
    # a block that optimises to nothing keeps its place
    var q = parse_qasm_program("OPENQASM 3.0; qubit[2] q; bit[1] c; int k = 0; while (k < 2) { k += 1; h q[0]; h q[0]; } c[0] = measure q[0];")
    var r2 = compile_program(q, Target.line(2), 2, 7)
    _ = verify_program(q, r2)
    assert_equal(sample_program(compact_qubits(r2.program), 20, 1).sample.counts.get("0", 0), 20)
    # code after a break is out of reach, and needs no block
    var d = parse_qasm_program("OPENQASM 3.0; qubit[2] q; bit[1] c; int k = 0; while (k < 2) { k += 1; h q[0]; break; x q[1]; }")
    _ = verify_program(d, compile_program(d, Target.line(2), 1, 7))


def test_8_classical_tables_are_compared() raises:
    var pair = _base()
    var m = pair[1].copy()
    for ref n in m.program.nodes:
        if n.op == CN.CONST and n.i == 2:
            n.i = 3
    _caught(pair[0], m)


def test_9_a_compiled_block_cannot_gain_a_kind_of_op() raises:
    var pair = _base()
    ref p = pair[0]
    ref r = pair[1]
    var m5 = r.copy()
    m5.program.body.ops[_measure(m5, 1)].kind = Op.RESET
    _caught(p, m5)
    var m = r.copy()
    m.program.body.ops[_measure(m, 1)].cond = 0
    _caught(p, m)
    var n = r.copy()
    ref g = n.program.body.ops[_measure(n, 1)]
    g.kind = Op.DEPOLARIZE1
    g.p0 = Param(0.1)
    _caught(p, n)
    # a block whose source has a reset stays unchecked
    var q = parse_qasm_program("OPENQASM 3.0; qubit[2] q; bit[1] c; int k = 0; while (k < 2) { k += 1; reset q[0]; h q[0]; c[0] = measure q[0]; }")
    var c = compile_program(q, Target.line(2), 1, 7)
    assert_true(c.blocks_unchecked >= 1, String(c))


def test_10_huge_constant_phases() raises:
    var hd = String("OPENQASM 3.0; qubit[2] q; bit[2] c; int k = 0; k = 1; ")
    var rows: List[String] = ["U(0.3, 1e12, 1.1) q[0];", "U(0.3, 1.1, 1e12) q[0];", "U(1e12, 1.1, 0.2) q[0];", "U(0.3, 1e12, 1e12) q[0];", "h q[0]; rz(1e12) q[0]; cp(1e12) q[0], q[1];"]
    for row in rows:
        for level in range(4):
            _ = compile_program(parse_qasm_program(hd + row), Target.line(2), level, 7)


def test_6_symbolic_offsets_reduce() raises:
    var src = "OPENQASM 3.0; input float t; qubit[1] q; bit[1] c; h q[0]; sx q[0]; rz(1) q[0]; p(t + 9223372036854775807.0) q[0]; sx q[0]; rz(1) q[0]; p(t + 9223372036854775807.0) q[0]; h q[0]; c[0] = measure q[0];"
    var values: List[Float64] = [0.4]
    var c = parse_qasm(src)
    # the source itself cannot be the reference: at 2^63 the sum t + offset drops t, a
    # float limit of binding. The same gates with the offset reduced by hand are.
    var small = String(reduce_angle(9223372036854775807.0))
    var base = sample(parse_qasm(src.replace("9223372036854775807.0", small)), 40000, 3, values)
    for level in range(4):
        var r = transpile(c, Target.all_to_all(1), level, 7)
        assert_true(_tv(base, sample(r.circuit, 40000, 5, values)) < 0.02)
    var p = parse_qasm_program(src.replace("h q[0]; sx q[0];", "int k = 0; k = 1; h q[0]; sx q[0];"))
    for level in range(4):
        _ = compile_program(p, Target.line(1), level, 7)
    # the offset goes, the symbol stays
    var d = c.copy()
    var r = transpile(d, Target.all_to_all(1), 0, 7)
    for g in r.circuit.ops:
        if g.kind == Op.P:
            assert_true(g.p0.is_symbolic() and abs(g.p0.offset) < 7.0)


def test_11_blocks_past_ten_wires_are_checked() raises:
    var body = String()
    for i in range(60):
        body += "cx q[" + String(i % 13) + "], q[" + String((i + 1) % 13) + "]; rz(0.1) q[" + String(i % 13) + "]; "
    for j in range(3):
        body += "c[" + String(j) + "] = measure q[" + String(j) + "]; "
    var src = "OPENQASM 3.0; qubit[13] q; bit[3] c; int k = 0; while (k < 2) { k += 1; " + body + " }"
    var p = parse_qasm_program(src)
    var r = compile_program(p, Target.line(13), 1, 7)
    assert_equal(r.blocks_unchecked, 0)
    var m = r.copy()
    for ref g in m.program.body.ops:
        if g.kind == Op.RZ:
            g.p0 = Param(g.p0.value() + 1e-4)
            break
    _caught(p, m)


def test_11_over_budget_is_counted_and_not_an_error() raises:
    var body = String()
    for i in range(400):
        body += "cx q[" + String(i % 21) + "], q[" + String((i + 1) % 21) + "]; "
    var src = "OPENQASM 3.0; qubit[21] q; bit[1] c; int k = 0; while (k < 2) { k += 1; " + body + " }"
    var r = compile_program(parse_qasm_program(src), Target.all_to_all(21), 0, 7)
    assert_equal(r.blocks_checked, 0)
    assert_true(r.blocks_unchecked >= 1)
    var wide = String("OPENQASM 3.0; qubit[23] q; bit[1] c; int k = 0; while (k < 2) { k += 1; h q[22]; h q[0]; cx q[0], q[1]; cx q[1], q[2]; cx q[2], q[3]; cx q[3], q[4]; cx q[4], q[5]; cx q[5], q[6]; cx q[6], q[7]; cx q[7], q[8]; cx q[8], q[9]; cx q[9], q[10]; cx q[10], q[11]; cx q[11], q[12]; cx q[12], q[13]; cx q[13], q[14]; cx q[14], q[15]; cx q[15], q[16]; cx q[16], q[17]; cx q[17], q[18]; cx q[18], q[19]; cx q[19], q[20]; cx q[20], q[21]; cx q[21], q[22]; }")
    var w = compile_program(parse_qasm_program(wide), Target.all_to_all(23), 0, 7)
    assert_equal(w.blocks_checked, 0)


def main() raises:
    test_tampering_still_fires()
    test_a_small_angle_does_not_hide()
    test_8_every_source_block_has_its_compiled_block()
    test_8_classical_tables_are_compared()
    test_9_a_compiled_block_cannot_gain_a_kind_of_op()
    test_10_huge_constant_phases()
    test_6_symbolic_offsets_reduce()
    test_11_blocks_past_ten_wires_are_checked()
    test_11_over_budget_is_counted_and_not_an_error()
    print("test_program_compile_regressions_verifier OK")
