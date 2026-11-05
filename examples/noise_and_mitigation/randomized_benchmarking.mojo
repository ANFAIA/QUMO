"""Randomized benchmarking: random Clifford sequences followed by their inverse return to |0>.

Reproduces: Classiq library - Randomized Benchmarking
Source: https://github.com/Classiq/classiq-library/blob/main/applications/benchmarking/randomized_benchmarking/randomized_benchmarking.ipynb (Classiq Library, MIT)
Framework: Classiq - https://github.com/Classiq/classiq-library

The notebook builds one 1-qubit model per sequence length (5, 10, 15, 20, 25 Cliffords). Each draws random gates from the hardware basis {id, rz, sx, x, cx} restricted to its Clifford map (id, x, rz(pi), sx), applies them, then applies the inverse of the whole sequence. It runs each program on two noiseless Classiq simulators and the analysis reports the trivial result: survival probability 1 for every length.
Here the same five models are built with a QUMO Circuit and Circuit.inverse(); the random choices come from a small LCG (seed 0), not Python's random, so the gate lists differ from the notebook's. Checks: the survival probability of every length is 1 on the exact statevector (the notebook's 100% fidelity), 1024 sampled shots per length are all "0", the circuit has exactly 2L gates, and, as an independent reference, the forward half alone matches a hand-written 2x2 complex matrix product (so the sequence is not trivially the identity and the inverse is what restores |0>).
Deviations: no Classiq synthesis, transpilation option or backend batching (QUMO simulates the circuit directly); the two Classiq simulators (statevector and sampling) become QUMO's exact statevector and 1024-shot sampling; the RBAnalysis plot is replaced by a printed table; cx is in the basis list but not in the Clifford map, so it is never drawn (as in the notebook).

Run: .venv/bin/mojo run -I . examples/noise_and_mitigation/randomized_benchmarking.mojo
example-meta: {"framework": "classiq", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from std.math import pi
from qumo.ir.circuit import Circuit
from qumo.sim.runner import evolve, sample
from qumo.testing.checks import CheckLog

comptime GATE_ID = 0
comptime GATE_RZ = 1
comptime GATE_SX = 2
comptime GATE_X = 3


struct Lcg(Movable):
    var state: UInt64

    def __init__(out self, seed: UInt64):
        self.state = seed * 6364136223846793005 + 1442695040888963407

    def below(mut self, n: Int) -> Int:
        self.state = self.state * 6364136223846793005 + 1442695040888963407
        return Int((self.state >> 33) % UInt64(n))


def random_cliffords(mut rng: Lcg, count: Int) -> List[Int]:
    """Gates drawn uniformly from the notebook's supported set {id, rz(pi), sx, x}."""
    var g = List[Int]()
    for _ in range(count):
        g.append(rng.below(4))
    return g^


def apply_gates(mut c: Circuit, gates: List[Int]):
    for g in gates:
        if g == GATE_ID:
            c.id(0)
        elif g == GATE_RZ:
            c.rz(pi, 0)
        elif g == GATE_SX:
            c.sx(0)
        else:
            c.x(0)


def rb_circuit(gates: List[Int], measure: Bool) raises -> Circuit:
    """The notebook's main: apply the sequence, then invert(apply the sequence)."""
    var fwd = Circuit(1, name="forward")
    apply_gates(fwd, gates)
    var c = Circuit(1, 1, name="rb")
    c.compose(fwd)
    c.compose(fwd.inverse())
    if measure:
        c.measure(0, 0)
    return c^


# 2x2 complex matrix as [re00, im00, re01, im01, re10, im10, re11, im11].
def mat_mul(a: List[Float64], b: List[Float64]) -> List[Float64]:
    var out = List[Float64](length=8, fill=0.0)
    for i in range(2):
        for j in range(2):
            var re = 0.0
            var im = 0.0
            for k in range(2):
                var ar = a[2 * (2 * i + k)]
                var ai = a[2 * (2 * i + k) + 1]
                var br = b[2 * (2 * k + j)]
                var bi = b[2 * (2 * k + j) + 1]
                re += ar * br - ai * bi
                im += ar * bi + ai * br
            out[2 * (2 * i + j)] = re
            out[2 * (2 * i + j) + 1] = im
    return out^


def gate_matrix(g: Int) -> List[Float64]:
    var m = List[Float64](length=8, fill=0.0)
    if g == GATE_ID:
        m[0] = 1.0
        m[6] = 1.0
    elif g == GATE_RZ:  # diag(exp(-i pi/2), exp(i pi/2)) = diag(-i, i)
        m[1] = -1.0
        m[7] = 1.0
    elif g == GATE_SX:  # (1/2) [[1+i, 1-i], [1-i, 1+i]]
        m[0] = 0.5
        m[1] = 0.5
        m[2] = 0.5
        m[3] = -0.5
        m[4] = 0.5
        m[5] = -0.5
        m[6] = 0.5
        m[7] = 0.5
    else:  # X
        m[2] = 1.0
        m[4] = 1.0
    return m^


def forward_survival(gates: List[Int]) -> Float64:
    """|<0| U_L ... U_1 |0>|^2 from the matrix product."""
    var u = gate_matrix(GATE_ID)
    for g in gates:
        u = mat_mul(gate_matrix(g), u)
    return u[0] * u[0] + u[1] * u[1]


def main() raises:
    var log = CheckLog("classiq/randomized_benchmarking")
    var lengths = List[Int]()
    lengths.append(5)
    lengths.append(10)
    lengths.append(15)
    lengths.append(20)
    lengths.append(25)
    var rng = Lcg(0)
    var shots = 1024
    print("Cliffords | survival (exact) | shots at '0' | forward-only survival")
    for idx in range(len(lengths)):
        var L = lengths[idx]
        var gates = random_cliffords(rng, L)
        var circ = rb_circuit(gates, False)
        var p0 = evolve(circ).probabilities()[0]
        var res = sample(rb_circuit(gates, True), shots, seed=UInt64(100 + idx))
        var frac = res.probability("0")
        var fwd = Circuit(1, name="fwd_only")
        apply_gates(fwd, gates)
        var p_fwd_qumo = evolve(fwd).probabilities()[0]
        var p_fwd_ref = forward_survival(gates)
        print(L, "|", p0, "|", frac, "|", p_fwd_qumo)
        var tag = "L=" + String(L) + " "
        _ = log.close(tag + "survival probability with the inverse (notebook: 100% fidelity)", p0, 1.0, 1e-12)
        _ = log.close(tag + "sampled fraction of '0' over 1024 shots", frac, 1.0, 0.0)
        _ = log.check(tag + "circuit holds 2L gates", len(circ) == 2 * L, "gates=" + String(len(circ)))
        _ = log.close(tag + "forward half alone matches the 2x2 matrix product", p_fwd_qumo, p_fwd_ref, 1e-12)
    log.finish()
