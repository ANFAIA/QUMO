"""Size limits of OpenQASM source text.

A short source could ask for any amount of memory: `pow(2^53 - 1) @ x` built 2^53
gates, nested gate definitions doubled at each level, and `qubit[2^40]` was taken at
its word. Before these limits, each QM-E506 case below and each huge pow took more
than 3 GB or ended the process. A source now has a budget of MAX_SOURCE_ITEMS plus its own length (QM-E506),
declares at most UNUSED_WIRES wires beyond 3 per operation (QM-E507), and the tableau
refuses a width it cannot hold (QM-E713). A pow of a gate of known period folds, so
it needs no budget at all.
"""

from std.testing import assert_equal, assert_true

from qumo.ai.diagnostics import lint, lint_program
from qumo.ir.circuit import UNUSED_WIRES, Circuit
from qumo.ir.qasm import MAX_SOURCE_ITEMS, load_any, load_any_with_warnings, parse_qasm, parse_qasm_program
from qumo.sim.program_runner import sample_program
from qumo.sim.runner import evolve
from qumo.sim.stabilizer import MAX_TABLEAU_QUBITS, TableauSimulator

comptime H = 'OPENQASM 3.0; include "stdgates.inc"; '
comptime PREP = "qubit[3] q; ry(0.3) q[0]; rz(0.7) q[0]; ry(1.1) q[1]; rz(0.4) q[1]; ry(0.9) q[2]; rz(1.3) q[2]; cx q[0], q[2]; "
comptime PREP_OPS = 7


def _code(src: String) -> String:
    """What the flat parser says: the code, or nothing when it parses."""
    try:
        _ = parse_qasm(src)
    except e:
        return String(String(e)[byte=:7])
    return ""


def _program_code(src: String) -> String:
    try:
        _ = parse_qasm_program(src)
    except e:
        return String(String(e)[byte=:7])
    return ""


def _any_code(src: String) -> String:
    try:
        _ = load_any(src)
    except e:
        return String(String(e)[byte=:7])
    return ""


def _refused(src: String, code: String, what: String) raises:
    """Refused by all three entry points, with a location and a hint."""
    assert_equal(_code(src), code, what)
    assert_equal(_program_code(src), code, what + " (program parser)")
    assert_equal(_any_code(src), code, what + " (load_any)")
    try:
        _ = parse_qasm(src)
    except e:
        var s = String(e)
        assert_true(s.find("(line ") > 0 and s.find("Hint: ") > 0, what + ": " + s)


def _operands(name: String) -> String:
    if name == "ccx" or name == "cswap" or name == "ccz":
        return " q[0], q[1], q[2];"
    if name == "cx" or name == "cy" or name == "cz" or name == "ch" or name == "swap" or name == "ecr" or name == "iswap" or name == "iswapdg":
        return " q[0], q[1];"
    return " q[1];"


def test_pow_folds_gates_of_known_period() raises:
    """U^k, global phase included, equals the gates written out, and an exponent
    past the period p leaves |k| mod p gates."""
    var names: List[String] = ["x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx", "sxdg", "cx", "cy", "cz", "ch", "swap", "ecr", "iswap", "iswapdg", "ccx", "cswap", "ccz"]
    var periods: List[Int] = [2, 2, 2, 2, 4, 4, 8, 8, 4, 4, 2, 2, 2, 2, 2, 2, 4, 4, 2, 2, 2]
    for gi in range(len(names)):
        var g = names[gi]
        var p = periods[gi]
        for k in range(-19, 20):
            var power = parse_qasm(H + PREP + "pow(" + String(k) + ") @ " + g + _operands(g))
            var spelled = String(H + PREP)
            for _ in range(abs(k)):
                spelled += ("" if k > 0 else "inv @ ") + g + _operands(g)
            var a = evolve(power)
            var b = evolve(parse_qasm(spelled))
            for i in range(len(a.re)):
                assert_true(abs(a.re[i] - b.re[i]) < 1e-12 and abs(a.im[i] - b.im[i]) < 1e-12, "pow(" + String(k) + ") @ " + g)
            var left = abs(k) % p if abs(k) > p else abs(k)
            assert_equal(len(power.ops) - PREP_OPS, left, "gates left by pow(" + String(k) + ") @ " + g)


def test_pow_with_a_huge_exponent() raises:
    var c = parse_qasm(H + "qubit q; pow(9007199254740991) @ x q;")
    assert_equal(len(c.ops), 1)
    c = parse_qasm(H + "qubit q; pow(-9007199254740991) @ t q;")
    assert_equal(len(c.ops), 7)  # 2^53 - 1 is 7 mod 8
    assert_equal(c.ops[0].kind, parse_qasm(H + "qubit q; tdg q;").ops[0].kind)
    # nested: the inner pow folds to no gate, so the outer one has nothing to repeat
    assert_equal(len(parse_qasm(H + "qubit q; pow(100000) @ pow(100000) @ x q;").ops), 0)
    # at most one period, so what parsed before keeps its gates
    assert_equal(len(parse_qasm(H + "qubit q; pow(2) @ h q;").ops), 2)
    assert_equal(len(parse_qasm(H + "qubit q; pow(4) @ h q;").ops), 0)
    # past 2^53 a float exponent is no longer exact, and its parity is lost
    assert_equal(_code(H + "qubit q; pow(9007199254740993) @ x q;"), "QM-E550")
    assert_equal(_code(H + "qubit q; pow(1e300) @ x q;"), "QM-E550")


def test_pow_of_a_phase_or_of_nothing() raises:
    """No gate to repeat: no loop, whatever the exponent."""
    var c = parse_qasm(H + "gate e a { } qubit q; pow(9007199254740991) @ e q;")
    assert_equal(len(c.ops), 0)
    c = parse_qasm(H + "gate ph a { gphase(0.5); } qubit q; pow(3) @ ph q;")
    assert_true(abs(c.global_phase - 1.5) < 1e-12, String(c.global_phase))
    c = parse_qasm(H + "gate ph a { gphase(0.5); } qubit q; pow(-2) @ ph q;")
    assert_true(abs(c.global_phase + 1.0) < 1e-12, String(c.global_phase))
    _ = parse_qasm(H + "gate ph a { gphase(0.5); } qubit q; pow(9007199254740991) @ ph q;")


def test_repeats_past_the_budget() raises:
    """Each of these took more than 3 GB, or ended the process."""
    _refused(H + "qubit q; pow(9000000) @ rz(0.1) q;", "QM-E506", "pow of a gate with no period")
    _refused(H + "qubit q; pow(3000) @ pow(3000) @ rz(0.1) q;", "QM-E506", "pow of a pow")
    var laughs = String(H + "qubit q; gate g0 a { } ")
    for i in range(1, 31):
        laughs += "gate g" + String(i) + " a { g" + String(i - 1) + " a; g" + String(i - 1) + " a; } "
    _refused(laughs + "g30 q;", "QM-E506", "gates that call the one before twice")
    _refused(H + "qubit q; for int i in [0:999999] { ; ; ; ; ; ; ; ; ; }", "QM-E506", "statements of an unrolled loop")


def test_sizes_past_the_budget() raises:
    _refused(H + "qubit[1099511627776] q; h q[0];", "QM-E506", "a huge register")
    _refused(H + "qubit[4611686018427387904] a; qubit[4611686018427387904] b; h a[0];", "QM-E506", "two registers whose sizes overflow")
    _refused(H + "qubit q; bit[1099511627776] c; c[0] = measure q;", "QM-E506", "a huge bit register")
    _refused(H + "h $4611686018427387904;", "QM-E506", "a huge hardware qubit")
    _refused(H + "h $99999999999999999999999999;", "QM-E506", "a hardware qubit past 2^64")
    _refused(H + "qubit[2] q; ctrl(9223372036854775807) @ ctrl(9223372036854775807) @ ctrl(3) @ x q[0], q[1];", "QM-E506", "control counts whose sum overflows")
    var big = String(MAX_SOURCE_ITEMS + 1000)
    _refused(H + "qubit[" + big + "] q; h q;", "QM-E506", "a register past the budget, every wire used")


def test_wires_need_operations() raises:
    """The rule of circuit_from_json: UNUSED_WIRES free, then 3 per operation."""
    assert_equal(UNUSED_WIRES, 4096)
    _refused(H + "qubit[5000] q; h q[0];", "QM-E507", "idle qubits")
    _refused(H + "bit[5000] c; qubit q; h q;", "QM-E507", "idle bits")
    _refused(H + "h $5000;", "QM-E507", "an idle hardware qubit range")
    var idle = H + "qubit[5000] q; int i = 0; while (i < 1) { i += 1; h q[0]; }"
    assert_equal(_program_code(idle), "QM-E507")
    assert_equal(_any_code(idle), "QM-E507")
    assert_equal(_code(H + "qubit[4096] q; h q[0];"), "")
    assert_equal(_code(H + "qubit[5000] q; h q;"), "")
    assert_equal(_program_code(H + "qubit[5000] q; int i = 0; while (i < 1) { i += 1; h q; }"), "")


def test_load_any_reports_a_size_once() raises:
    var src = H + "qubit q; pow(9000000) @ rz(0.1) q;"
    var warnings = List[String]()
    try:
        _ = load_any_with_warnings(src, warnings)
        raise Error("loaded")
    except e:
        assert_true(String(e).startswith("QM-E506"), String(e))


def test_large_circuits_still_load() raises:
    """A wide Clifford circuit on the tableau, and the old loop limit with a small body."""
    var src = String(H + "qubit[150] q; bit[2] c; h q[0]; for int i in [0:148] { cx q[i], q[i + 1]; } c[0] = measure q[0]; c[1] = measure q[149];")
    var r = sample_program(load_any(src), 200, 3, method="tableau")
    assert_equal(r.sample.counts.get("00", 0) + r.sample.counts.get("11", 0), 200)
    var c = parse_qasm(H + "qubit[2] q; for int i in [0:999999] { x q[i % 2]; }")
    assert_equal(len(c.ops), 1000000)


def test_tableau_refuses_a_width_it_cannot_hold() raises:
    try:
        _ = TableauSimulator(MAX_TABLEAU_QUBITS + 1)
        raise Error("built")
    except e:
        assert_true(String(e).startswith("QM-E713"), String(e))
    var p = load_any(H + "qubit[300000] q; bit[1] c; h q; c[0] = measure q[0];")
    try:
        _ = sample_program(p, 2, 1, method="tableau")
        raise Error("sampled")
    except e:
        assert_true(String(e).startswith("QM-E713"), String(e))


def test_lint_lists_100_findings_per_code() raises:
    var src = String(H + "qubit q; ")
    for _ in range(1000):
        src += "x q; "
    var found = 0
    var last = String()
    var ds = lint(parse_qasm(src))
    for d in ds:
        if d.code == "QM-W010":
            found += 1
            last = d.message
    assert_equal(found, 101)
    assert_true(last.startswith("899 more"), last)
    var loops = String(H + "int k = 0; ")
    for _ in range(150):
        loops += "while (k < 3) { } "
    var stuck = lint_program(parse_qasm_program(loops))
    assert_equal(len(stuck), 101)
    assert_true(stuck[100].message.startswith("50 more"), stuck[100].message)


def main() raises:
    test_pow_folds_gates_of_known_period()
    test_pow_with_a_huge_exponent()
    test_pow_of_a_phase_or_of_nothing()
    test_repeats_past_the_budget()
    test_sizes_past_the_budget()
    test_wires_need_operations()
    test_load_any_reports_a_size_once()
    test_large_circuits_still_load()
    test_tableau_refuses_a_width_it_cannot_hold()
    test_lint_lists_100_findings_per_code()
    print("test_qasm_limits OK")
