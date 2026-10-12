"""Quantum phase estimation built from the IR functors power() + controlled().

Run: .venv/bin/mojo run -I . examples/qpe.mojo
"""

from std.math import pi
from qumo.ir.circuit import Circuit
from qumo.sim.runner import evolve, sample
from qumo.algorithms.library import qpe, qpe_phase, bitstring_to_int


def main() raises:
    print("QPE of P(2 pi k / 16) on eigenstate |1>, 4 evaluation qubits (exact):")
    for k in range(0, 16, 3):
        var u = Circuit(1)
        u.p(2 * pi * Float64(k) / 16.0, 0)
        var prep = Circuit(1)
        prep.x(0)
        var c = qpe(u, 4, prep.copy(), measure=True)
        var r = sample(c, 256, seed=1)
        var v = bitstring_to_int(r.most_frequent())
        print("  k =", k, " measured", r.most_frequent(), "-> phase", qpe_phase(v, 4), "(true", Float64(k) / 16.0, ")",
              " P =", r.probability(r.most_frequent()))

    print("QPE of a 2-qubit unitary U = CP(c) . (RZ(a) (x) RZ(b)) on |11>, phase = (a/2 + b/2 + c) / 2pi mod 1:")
    var a = 2 * pi * 0.125
    var b = 2 * pi * 0.25
    var cc = 2 * pi * 0.375
    var u2 = Circuit(2)
    u2.rz(a, 0)
    u2.rz(b, 1)
    u2.cp(cc, 0, 1)
    var p2 = Circuit(2)
    p2.x(0)
    p2.x(1)
    # |11>: RZ contributes e^{+i a/2} e^{+i b/2}, CP contributes e^{i c}
    var true_phase = (a / 2 + b / 2 + cc) / (2 * pi)
    true_phase = true_phase - Float64(Int(true_phase))
    for n_eval in range(3, 7):
        var c = qpe(u2, n_eval, p2.copy(), measure=True)
        var r = sample(c, 512, seed=2)
        var v = bitstring_to_int(r.most_frequent())
        print("  n_eval =", n_eval, " estimate =", qpe_phase(v, n_eval), " true =", true_phase,
              " P(top) =", r.probability(r.most_frequent()), " gates =", len(c))

    print("Non-dyadic phase 1/3: resolution improves with n_eval")
    for n_eval in range(3, 9):
        var u = Circuit(1)
        u.p(2 * pi / 3.0, 0)
        var prep = Circuit(1)
        prep.x(0)
        var c = qpe(u, n_eval, prep.copy(), measure=True)
        var r = sample(c, 2000, seed=3)
        var v = bitstring_to_int(r.most_frequent())
        print("  n_eval =", n_eval, " estimate =", qpe_phase(v, n_eval), " error =", abs(qpe_phase(v, n_eval) - 1.0 / 3.0),
              " P(top) =", r.probability(r.most_frequent()))
