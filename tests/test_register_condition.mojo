from qumo.ir.qasm import parse_qasm, to_qasm, decimal_to_bits, bits_to_decimal
from qumo.ir.qasm2 import to_qasm2
from qumo.sim.runner import sample
from std.testing import assert_equal, assert_true


def main() raises:
    var b = decimal_to_bits("2037035976334486086268445688409378161051468393665936250636140449354381299763336706183397376", 301)
    assert_equal(len(b), 301)
    assert_equal(b[300], 1)
    var ones = 0
    for v in b:
        ones += v
    assert_equal(ones, 1)
    assert_equal(bits_to_decimal(b), "2037035976334486086268445688409378161051468393665936250636140449354381299763336706183397376")
    var src = String("OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[3];\ncreg c[2];\nx q[0];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\nif(c==1) x q[2];\nif(c==3) x q[1];\nmeasure q[2] -> c[1];\n")
    var circ = parse_qasm(src)
    var r = sample(circ, 50, 3)
    # c0=1 (x q0), c1 after reset by measuring q2: if(c==1) fired -> q2=1 -> c1=1 => "11"; if(c==3) did not fire
    assert_equal(r.counts.get("11", 0), 50)
    var back = parse_qasm(to_qasm(circ))
    assert_equal(sample(back, 50, 3).counts.get("11", 0), 50)
    var back2 = parse_qasm(to_qasm2(circ))
    assert_equal(sample(back2, 50, 3).counts.get("11", 0), 50)
    print("test_register_condition OK")
