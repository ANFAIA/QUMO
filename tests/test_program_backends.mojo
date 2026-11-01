"""Program backends: the same answers from each simulator and any number of workers."""

from std.math import pi
from std.testing import assert_equal, assert_true, assert_raises

from qumo.ir.circuit import Circuit
from qumo.ir.qasm import parse_qasm, parse_qasm_program
from qumo.ir.program import Program, ProgramBuilder
from qumo.sim.branching import sample_branching
from qumo.sim.program_runner import sample_program, sample_interpreted, is_clifford_program, ProgramResult
from qumo.sim.runner import sample, SampleResult
from qumo.sim.stabilizer import sample_stabilizer

comptime RUS = "qubit[2] q; bit[2] c; output int rounds; rounds = 0; while (!c[0]) { reset q[0]; h q[0]; c[0] = measure q[0]; rounds += 1; if (rounds == 3) { x q[1]; } } c[1] = measure q[1];"
comptime NEAR = "qubit[2] q; bit[2] c; output int rounds; rounds = 0; h q[0]; t q[0]; h q[0]; c[0] = measure q[0]; while (!c[0] && rounds < 4) { reset q[0]; h q[0]; t q[0]; h q[0]; c[0] = measure q[0]; rounds += 1; } cx q[0], q[1]; c[1] = measure q[1];"


def _tv(a: SampleResult, b: SampleResult) -> Float64:
    var total = 0.0
    for k in a.sorted_keys():
        total += abs(a.probability(k) - b.probability(k))
    for k in b.sorted_keys():
        if a.counts.get(k, 0) == 0:
            total += b.probability(k)
    return total / 2.0


def _same(a: ProgramResult, b: ProgramResult) raises:
    assert_equal(len(a.sample.memory), len(b.sample.memory))
    for s in range(len(a.sample.memory)):
        assert_equal(a.sample.memory[s], b.sample.memory[s])
        assert_equal(a.iterations[s], b.iterations[s])
    assert_equal(len(a.out_int), len(b.out_int))
    for s in range(len(a.out_int)):
        assert_equal(a.out_int[s], b.out_int[s])
    for s in range(len(a.out_real)):
        assert_equal(a.out_real[s], b.out_real[s])


def test_rng_order_matches_the_flat_samplers() raises:
    """The interpreter reproduces the flat samplers: run_range keeps the draws of run."""
    var src = String("qubit[3] q; bit[3] c; h q[0]; cx q[0], q[1]; c[0] = measure q[0]; if (c[0] == 1) x q[2]; h q[1]; c[1] = measure q[1]; reset q[0]; h q[0]; c[2] = measure q[0];")
    var c = parse_qasm(src)
    var p = Program.from_circuit(c)
    var a = sample_stabilizer(c, 2000, 7)
    var b = sample_interpreted(p, 2000, 7, method="tableau", workers=1)
    assert_equal(String(a), String(b.sample))
    assert_equal(b.sample.method, a.method)
    var d = sample_branching(c, 500, 7)
    var e = sample_interpreted(p, 500, 7, method="branching", workers=1)
    assert_equal(String(d), String(e.sample))
    assert_equal(e.sample.method, d.method)
    # and with Pauli noise, which draws from the same stream
    var nc = parse_qasm("qubit[2] q; bit[2] c; h q[0]; #pragma qumo depolarize1(0.2) q[0];\n#pragma qumo x_error(0.1) q[1];\ncx q[0], q[1]; c = measure q;")
    var np = Program.from_circuit(nc)
    assert_equal(String(sample_stabilizer(nc, 2000, 3)), String(sample_interpreted(np, 2000, 3, method="tableau", workers=1).sample))


def test_workers_do_not_change_the_answer() raises:
    """Exit test: the same counts with 1 and N workers."""
    var p = parse_qasm_program(RUS)
    var n = parse_qasm_program(NEAR)
    var f = parse_qasm_program("qubit[2] q; bit[2] c; output float x; float a; a = 0.3; int i; i = 0; while (i < 3) { rx(a) q[0]; a = a * 2; c[0] = measure q[0]; if (c[0]) { i = i + 1; } else { i = i + 2; } } x = a; c[1] = measure q[1];")
    _same(sample_interpreted(p, 3000, 11, workers=1), sample_interpreted(p, 3000, 11, workers=8))
    _same(sample_interpreted(p, 3000, 11, method="tableau", workers=1), sample_interpreted(p, 3000, 11, method="tableau", workers=8))
    _same(sample_interpreted(n, 400, 11, method="branching", workers=1), sample_interpreted(n, 400, 11, method="branching", workers=8))
    _same(sample_interpreted(f, 3000, 11, workers=1), sample_interpreted(f, 3000, 11, workers=8))
    _same(sample_interpreted(f, 3000, 11, workers=1), sample_interpreted(f, 3000, 11, workers=3))
    # the default picks parallel shots here, with the same result
    _same(sample_interpreted(f, 3000, 11, workers=1), sample_interpreted(f, 3000, 11))


def test_backends_agree() raises:
    """Exit test. Threshold fixed before the first run: TV < 0.03 on 20000 shots
    (two independent samples of at most 8 outcomes differ by about 0.011)."""
    var p = parse_qasm_program(RUS)
    var a = sample_interpreted(p, 20000, 21, method="statevector")
    var b = sample_interpreted(p, 20000, 22, method="tableau")
    var c = sample_interpreted(p, 5000, 23, method="branching")
    assert_true(_tv(a.sample, b.sample) < 0.03)
    assert_true(_tv(a.sample, c.sample) < 0.05)
    assert_true(abs(a.mean_iterations() - b.mean_iterations()) < 0.1)
    var n = parse_qasm_program(NEAR)
    var d = sample_interpreted(n, 20000, 24, method="statevector")
    var e = sample_interpreted(n, 5000, 25, method="branching")
    assert_true(_tv(d.sample, e.sample) < 0.05)
    assert_true(abs(d.mean_iterations() - e.mean_iterations()) < 0.1)


def test_reset_on_an_entangled_qubit() raises:
    """A reset measures: half of the time the partner ends in 1."""
    var c = parse_qasm("qubit[2] q; bit[1] c; h q[0]; cx q[0], q[1]; reset q[0]; c[0] = measure q[1];")
    var a = sample_stabilizer(c, 20000, 1)
    assert_true(abs(a.probability("1") - 0.5) < 0.02, String(a))
    var p = parse_qasm_program("qubit[2] q; bit[1] c; int i; i = 0; while (i < 1) { h q[0]; cx q[0], q[1]; reset q[0]; i += 1; } c[0] = measure q[1];")
    var r = sample_program(p, 20000, 2, method="tableau")
    assert_true(abs(r.sample.probability("1") - 0.5) < 0.02)


def test_auto_method() raises:
    var p = parse_qasm_program(RUS)
    assert_true(is_clifford_program(p))
    assert_equal(sample_program(p, 50, 1).sample.method, "stabilizer:tableau")
    assert_equal(sample_program(parse_qasm_program(NEAR), 50, 1).sample.method, "statevector:trajectories")
    # a runtime angle is not known to be Clifford
    var v = parse_qasm_program("qubit[1] q; bit[1] c; float a; a = pi / 2; rx(a) q[0]; c[0] = measure q[0];")
    assert_true(not is_clifford_program(v))
    assert_equal(sample_program(v, 10, 1).sample.method, "statevector:trajectories")
    # wide Clifford programs run where no state vector could
    var wide = parse_qasm_program("qubit[60] q; bit[2] c; h q[0]; for int i in [0:58] { cx q[i], q[i + 1]; } c[0] = measure q[0]; while (false) { } c[1] = measure q[59];")
    var w = sample_program(wide, 200, 1)
    assert_equal(w.sample.counts.get("00", 0) + w.sample.counts.get("11", 0), 200)
    # plain circuits keep the path of sample()
    var flat = Program.from_circuit(parse_qasm("qubit[2] q; bit[2] c; h q[0]; cx q[0], q[1]; c = measure q;"))
    assert_equal(sample_program(flat, 10, 1).sample.method, "statevector:static")


def test_explicit_methods() raises:
    var p = parse_qasm_program(RUS)
    for m in ["density", "ptsbe", "frame"]:
        with assert_raises(contains="QM-E483"):
            _ = sample_program(p, 10, 1, method=String(m))
    with assert_raises(contains="QM-E483"):
        _ = sample_program(p, 10, 1, method="nosuch")
    # gpu runs on the CPU and says so
    var g = sample_program(p, 200, 5, method="gpu")
    assert_equal(len(g.warnings), 1)
    assert_true(g.warnings[0].startswith("QM-W020"))
    assert_equal(String(g.sample), String(sample_program(p, 200, 5).sample))
    # the tableau binds a runtime angle as the gate runs
    var v = parse_qasm_program("qubit[1] q; bit[1] c; float a; a = pi / 2; a = a * 2; rx(a) q[0]; c[0] = measure q[0];")
    assert_equal(sample_program(v, 20, 1, method="tableau").sample.counts.get("1", 0), 20)
    var bad = parse_qasm_program("qubit[1] q; bit[1] c; float a; a = 0.3; rx(a) q[0]; c[0] = measure q[0];")
    with assert_raises(contains="QM-E701"):
        _ = sample_program(bad, 20, 1, method="tableau")
    with assert_raises(contains="QM-E701"):
        _ = sample_program(parse_qasm_program("qubit[1] q; bit[1] c; while (!c[0]) { t q[0]; c[0] = measure q[0]; x q[0]; c[0] = measure q[0]; }"), 20, 1, method="tableau")
    # branching takes it, and runtime angles of any value
    var r = sample_program(bad, 2000, 1, method="branching")
    var want = sample_program(bad, 2000, 1, method="statevector")
    assert_true(_tv(r.sample, want.sample) < 0.05)
    # Kraus noise is not for the tableau: auto stays on the state vector
    var k = parse_qasm_program("qubit[1] q; bit[1] c; int i; i = 0; while (i < 1) { #pragma qumo kraus1 \"bf\" (0,0,1,0,1,0,0,0) q[0];\n i += 1; } c[0] = measure q[0];")
    var kr = sample_program(k, 10, 1)
    assert_equal(kr.sample.method, "statevector:trajectories")
    assert_equal(kr.sample.counts.get("1", 0), 10)
    with assert_raises(contains="QM-E414"):
        _ = sample_program(k, 10, 1, method="tableau")


def test_block_plans() raises:
    """A long static block runs from a plan: same distribution, and gates with
    runtime angles or conditions split it correctly."""
    var c = Circuit(6, 6)
    for d in range(12):
        for q in range(6):
            c.ry(0.3 + 0.1 * Float64(q + d), q)
        for q in range(5):
            c.cx(q, q + 1)
    c.measure(0, 0)
    c.reset(0)
    c.h(0)
    for q in range(6):
        c.measure(q, q)
    var a = sample(c, 20000, 3)
    var b = sample_interpreted(Program.from_circuit(c), 20000, 3)
    assert_true(_tv(a, b.sample) < 0.03)
    # a long block in a loop, with a runtime angle in the middle of it
    var head = String("qubit[3] q; bit[1] c; float a; a = 0.0; int i; i = 0; while (i < 2) { a = a + pi / 2; ")
    var gates = String()
    for _ in range(20):
        gates += "h q[1]; x q[2]; "
    var src = head + gates + "rx(a) q[0]; " + gates + "i += 1; } c[0] = measure q[0];"
    var r = sample_program(parse_qasm_program(src), 2000, 2)
    # rx(pi/2) then rx(pi): 3 pi / 2 in all, an even split
    assert_true(abs(r.sample.probability("1") - 0.5) < 0.05)
    var src2 = head + gates + "rx(a) q[0]; rx(a) q[0]; " + gates + "i += 1; } c[0] = measure q[0];"
    var r2 = sample_program(parse_qasm_program(src2), 200, 2)  # pi, then 2 pi: 3 pi in all
    assert_equal(r2.sample.counts.get("1", 0), 200)


def test_errors_in_parallel() raises:
    var p = parse_qasm_program("qubit[1] q; bit[1] c; h q[0]; c[0] = measure q[0]; int d; d = 0; if (c[0]) d = 1; int e; e = 1 / d;")
    with assert_raises(contains="QM-E481"):
        _ = sample_program(p, 100, 1)
    var q = parse_qasm_program("qubit[1] q; bit[1] c; while (!c[0]) { }")
    with assert_raises(contains="QM-E480"):
        _ = sample_program(q, 10, 1, max_iterations=50)


def main() raises:
    test_rng_order_matches_the_flat_samplers()
    test_workers_do_not_change_the_answer()
    test_backends_agree()
    test_reset_on_an_entangled_qubit()
    test_auto_method()
    test_explicit_methods()
    test_block_plans()
    test_errors_in_parallel()
    print("test_program_backends OK")
