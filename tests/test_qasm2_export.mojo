from qumo.ir.circuit import Circuit
from qumo.ir.gates import Op
from qumo.ir.qasm2 import to_qasm2
from qumo.ir.qasm import parse_qasm
from qumo.sim.runner import evolve
from std.testing import assert_true, assert_almost_equal


def main() raises:
    var c = Circuit(3)
    for k in range(Op.CCZ + 1):
        c.add(k, 0, 1, 2, 0.3, -0.7, 1.1)
        c.add(k, 2, 0, 1, -0.2, 0.4, 0.9)
    var text = to_qasm2(c)
    var back = parse_qasm(text)
    assert_almost_equal(evolve(c).fidelity(evolve(back)), 1.0)
    print("test_qasm2_export OK")
