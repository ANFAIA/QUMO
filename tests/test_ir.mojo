from qumo.ir.circuit import Circuit, Instruction
from qumo.ir.gates import Op, matrix_1q, zyz_decompose, u3_matrix, m2_approx_eq_up_to_phase, op_from_name
from qumo.ir.param import Param
from std.testing import assert_equal, assert_true, assert_false


def main() raises:
    var c = Circuit(3)
    var theta = c.parameter("theta")
    c.h(0)
    c.cx(0, 1)
    c.rz(theta * 2.0, 2)
    c.t(2)
    c.measure_all()
    print(c)
    assert_equal(c.depth(), 2)
    assert_equal(c.t_count(), 2)
    assert_true(c.is_parametric())
    var b = c.bind([0.25])
    assert_false(b.is_parametric())
    assert_equal(b.ops[2].p0.value(), 0.5)
    c.validate()
    # ZYZ round trip on every 1q gate
    for k in range(Op.U + 1):
        var m = matrix_1q(k, 0.3, 1.1, -0.7)
        var z = zyz_decompose(m)
        var r = u3_matrix(z.theta, z.phi, z.lam)
        assert_true(m2_approx_eq_up_to_phase(m, r), "zyz roundtrip")
    assert_equal(op_from_name("cnot"), Op.CX)
    var u = Circuit(2)
    u.h(0)
    u.s(1)
    u.cx(0, 1)
    var inv = u.inverse()
    assert_equal(inv.ops[0].kind, Op.CX)
    assert_equal(inv.ops[1].kind, Op.SDG)
    var bad = Circuit(2)
    bad.cx(0, 5)
    try:
        bad.validate()
        raise Error("should fail")
    except e:
        assert_true(String(e).startswith("QM-E001"))
    print("test_ir OK")
