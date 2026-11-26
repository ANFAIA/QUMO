"""Clifford conjugation table (qumo/ir/clifford_table.mojo).

1. Exhaustive against matrices: for every opcode of the table and every
   A in (X_a, Z_a, X_b, Z_b), G^dag A G and G A G^dag computed from
   matrix_1q / matrix_2q equal the table's i^k X^x Z^z entry exactly (1e-12),
   phase included.
2. The specialised loops agree with the table, sign included: the inverse
   tableau of TableauSimulator (rows G^dag P G) and the forward tableau
   SympTab of qumo.compiler.blockalg (rows G P G^dag), one gate from identity.
Prints `test_clifford_table OK`."""

from std.testing import assert_true

from qumo.ir.gates import Op, op_arity, op_name, matrix_1q, matrix_2q, M2, M4
from qumo.ir.circuit import Instruction
from qumo.ir.clifford_table import clifford_heisenberg, clifford_image, clifford_sign
from qumo.sim.stabilizer import TableauSimulator
from qumo.compiler.blockalg import SympTab


def _kinds() -> List[Int]:
    return [Op.I, Op.X, Op.Y, Op.Z, Op.H, Op.S, Op.SDG, Op.SX, Op.SXDG,
            Op.CX, Op.CY, Op.CZ, Op.SWAP, Op.ISWAP, Op.ISWAPDG, Op.ECR]


def _dag(m: M4) -> M4:
    var o = M4()
    for r in range(4):
        for c in range(4):
            o.set(r, c, m.re[c * 4 + r], -m.im[c * 4 + r])
    return o^


def _gate(kind: Int) -> M4:
    if op_arity(kind) == 1:
        return M4.from_1q(matrix_1q(kind), False)
    return matrix_2q(kind)


def _xz(x: Int, z: Int, on_q1: Bool) -> M4:
    var m = M4.identity()
    if x == 1:
        m = m @ M4.from_1q(matrix_1q(Op.X), on_q1)
    if z == 1:
        m = m @ M4.from_1q(matrix_1q(Op.Z), on_q1)
    return m^


def _entry(xa: Int, za: Int, xb: Int, zb: Int, k: Int) -> M4:
    var m = _xz(xa, za, False) @ _xz(xb, zb, True)
    # times i^k
    var o = M4()
    for i in range(16):
        var re = m.re[i]
        var im = m.im[i]
        for _ in range(k & 3):
            var t = re
            re = -im
            im = t
        o.re[i] = re
        o.im[i] = im
    return o^


def _close(a: M4, b: M4) -> Bool:
    for i in range(16):
        if abs(a.re[i] - b.re[i]) > 1e-12 or abs(a.im[i] - b.im[i]) > 1e-12:
            return False
    return True


def _a(idx: Int) -> M4:
    return _xz(1 if idx % 2 == 0 else 0, 1 if idx % 2 == 1 else 0, idx >= 2)


def test_against_matrices() raises:
    for kind in _kinds():
        var g = _gate(kind)
        var gd = _dag(g)
        var nrows = 2 if op_arity(kind) == 1 else 4
        for idx in range(nrows):
            var xa = 0; var za = 0; var xb = 0; var zb = 0; var k = 0
            assert_true(clifford_heisenberg(kind, idx, xa, za, xb, zb, k), op_name(kind) + " is in the table")
            assert_true(_close(gd @ _a(idx) @ g, _entry(xa, za, xb, zb, k)), op_name(kind) + " G^dag A G, A index " + String(idx))
            assert_true(clifford_image(kind, idx, xa, za, xb, zb, k), op_name(kind))
            assert_true(_close(g @ _a(idx) @ gd, _entry(xa, za, xb, zb, k)), op_name(kind) + " G A G^dag, A index " + String(idx))
    var xa = 0; var za = 0; var xb = 0; var zb = 0; var k = 0
    assert_true(not clifford_heisenberg(Op.T, 0, xa, za, xb, zb, k), "T is not Clifford")


def _inst(kind: Int) -> Instruction:
    if op_arity(kind) == 1:
        return Instruction.make(kind, 0)
    return Instruction.make(kind, 0, 1)


def test_tableau_rows() raises:
    """TableauSimulator rows are C^dag P C: after one gate, G^dag A G."""
    for kind in _kinds():
        var t = TableauSimulator(2)
        t.apply_gate(_inst(kind))
        var nrows = 2 if op_arity(kind) == 1 else 4
        for idx in range(nrows):
            var q = 0 if idx < 2 else 1
            var row = q if idx % 2 == 0 else 2 + q
            var xa = 0; var za = 0; var xb = 0; var zb = 0; var k = 0
            _ = clifford_heisenberg(kind, idx, xa, za, xb, zb, k)
            var x = Int(t.xs[row * t.nw])
            var z = Int(t.zs[row * t.nw])
            assert_true(x == (xa | (xb << 1)) and z == (za | (zb << 1)), op_name(kind) + " tableau row bits, A index " + String(idx))
            assert_true(Int(t.signs[row]) == clifford_sign(xa, za, xb, zb, k), op_name(kind) + " tableau row sign, A index " + String(idx))


def test_symptab_rows() raises:
    """SympTab rows are C P C^dag: after one gate, G A G^dag."""
    for kind in _kinds():
        var t = SympTab(2)
        t.apply(_inst(kind))
        var nrows = 2 if op_arity(kind) == 1 else 4
        for idx in range(nrows):
            var q = 0 if idx < 2 else 1
            var row = q if idx % 2 == 0 else 2 + q
            var xa = 0; var za = 0; var xb = 0; var zb = 0; var k = 0
            _ = clifford_image(kind, idx, xa, za, xb, zb, k)
            var x = Int(t.x[row])
            var z = Int(t.z[row])
            assert_true(x == (xa | (xb << 1)) and z == (za | (zb << 1)), op_name(kind) + " SympTab row bits, A index " + String(idx))
            assert_true(Int(t.r[row]) == clifford_sign(xa, za, xb, zb, k), op_name(kind) + " SympTab row sign, A index " + String(idx))


def main() raises:
    test_against_matrices()
    test_tableau_rows()
    test_symptab_rows()
    print("test_clifford_table OK")
