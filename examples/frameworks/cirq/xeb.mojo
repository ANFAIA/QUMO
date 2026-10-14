"""Cross-entropy benchmarking of sqrt(iSWAP) under depolarizing noise.

Reproduces: Cirq - Cross Entropy Benchmarking Theory
Source: https://quantumai.google/cirq/noise/qcvv/xeb_theory (Cirq docs, CC BY 4.0 text, Apache-2.0 code)

The Cirq page benchmarks a 2-qubit SQRT_ISWAP layer with random single-qubit
gates: 64 PhasedXZ gates (x = 1/2, z and axis phase a in {0, 1/4, ..., 7/4}),
cycle depths 1, 10, ..., 100 (12 values), 10 random circuits, depolarizing
noise E_PAULI = 5e-3 on every qubit after every moment (density-matrix
simulator, 10,000 repetitions per circuit). For each depth it estimates the
linear XEB fidelity f = sum(x y) / sum(x^2) over the circuits, with
x = sum_s p(s)^2 - 1/4 and y = sum_s p_est(s) p(s) - 1/4 (ideal p, sampled
p_est), and fits a f_layer^depth. Page: "XEB layer fidelity: 9.787e-01 +-
1.79e-04" against "Noise model fidelity: 9.788e-01", the model being
(1 - e_depol)^4 with e_depol = E_PAULI / (1 - 1/D^2), D = 4 (four noisy
qubit-moments per cycle).

QUMO, same placement: sqrt(iSWAP) = RXX(-pi/4) RYY(-pi/4); PhasedXZ(x, z, a) =
Z^z Z^a X^x Z^-a = RZ(pi (a + z)) RX(pi x) RZ(-pi a) up to a global phase;
`depolarize1(5e-3)` on both qubits after every moment; noisy circuits are
sampled with per-shot Pauli trajectories (`sample`), ideal probabilities come
from `evolve`, and `fit_decay(..., fixed_b=0)` (qumo/qec/characterization.mojo)
fits A r^depth. The 120 (depth, circuit) pairs run in parallel threads.

Checks:
- the fitted layer fidelity is within 3e-3 of the model (1 - 16 p / 15)^4 =
  0.97885 for this placement (page: 0.9787 vs 0.9788);
- XEB fidelity decreases with depth (first > last) and f(depth 1) > 0.9;
- every ideal distribution is normalised (1e-12).

Run: .venv/bin/mojo run -I . examples/frameworks/cirq/xeb.mojo
example-meta: {"framework": "cirq", "tier": "slow", "timeout": 300, "requires": [], "args": []}
"""

from std.math import pi
from std.time import perf_counter_ns
from max.algorithm import parallelize
from qumo.ir.circuit import Circuit
from qumo.sim.runner import evolve, sample
from qumo.sim.rng import Rng
from qumo.qec.characterization import fit_decay
from qumo.testing.checks import CheckLog

comptime E_PAULI = 5e-3
comptime N_CIRCUITS = 10
comptime MAX_DEPTH = 100
comptime SHOTS = 4000


def phased_xz(mut c: Circuit, q: Int, idx: Int):
    """PhasedXZ(x=1/2, z, a) with (a, z) = (idx // 8, idx % 8) quarter turns."""
    var a = Float64(idx // 8) * 0.25
    var z = Float64(idx % 8) * 0.25
    c.rz(-pi * a, q)
    c.rx(pi / 2, q)
    c.rz(pi * (a + z), q)


def xeb_circuit(gates: List[Int], depth: Int, noisy: Bool) -> Circuit:
    """`depth` cycles of [1q layer, sqrt(iSWAP)] plus a final 1q layer (Cirq truncation)."""
    var c = Circuit(2, name="xeb")
    for cyc in range(depth + 1):
        phased_xz(c, 0, gates[2 * cyc])
        phased_xz(c, 1, gates[2 * cyc + 1])
        if noisy:
            c.depolarize1(E_PAULI, 0)
            c.depolarize1(E_PAULI, 1)
        if cyc == depth:
            break
        c.rxx(-pi / 4, 0, 1)
        c.ryy(-pi / 4, 0, 1)
        if noisy:
            c.depolarize1(E_PAULI, 0)
            c.depolarize1(E_PAULI, 1)
    if noisy:
        c.measure_all()
    return c^


def main() raises:
    var log = CheckLog("cirq/xeb")
    var depths = List[Int]()
    var d = 1
    while d <= MAX_DEPTH:
        depths.append(d)
        d += 9
    var rng = Rng(2024)
    var gate_seqs = List[List[Int]]()
    for _ in range(N_CIRCUITS):
        var g = List[Int]()
        for _ in range(2 * (MAX_DEPTH + 1)):
            g.append(rng.below(64))
        gate_seqs.append(g^)

    # ideal distributions and noisy circuits for every (depth, circuit) pair
    var nd = len(depths)
    var ideal = List[List[Float64]]()
    var noisy = List[Circuit]()
    var worst_norm = 0.0
    for di in range(nd):
        for ci in range(N_CIRCUITS):
            var p = evolve(xeb_circuit(gate_seqs[ci], depths[di], False)).probabilities()
            worst_norm = max(worst_norm, abs(p[0] + p[1] + p[2] + p[3] - 1.0))
            ideal.append(p^)
            noisy.append(xeb_circuit(gate_seqs[ci], depths[di], True))
    _ = log.close("ideal distributions normalised", worst_norm, 0.0, 1e-12)

    var npairs = len(noisy)
    var m_u = List[Float64](length=npairs, fill=0.0)
    var failed = List[Int](length=1, fill=0)
    var mp = m_u.unsafe_ptr().as_unsafe_any_origin()
    var fp = failed.unsafe_ptr().as_unsafe_any_origin()
    var cp = Pointer(to=noisy).as_unsafe_any_origin()
    var ip = Pointer(to=ideal).as_unsafe_any_origin()

    def task(k: Int) {imm mp, imm fp, imm cp, imm ip}:
        try:
            var r = sample(cp[][k], SHOTS, seed=UInt64(7919 * k + 1))
            var m = 0.0
            for s in r.memory:
                m += ip[][k][s]
            mp[unsafe_offset=k] = m / Float64(SHOTS)
        except:
            fp[unsafe_offset=0] = 1

    var t0 = perf_counter_ns()
    parallelize(task, npairs)
    var secs = Float64(perf_counter_ns() - t0) / 1e9
    _ = len(noisy)
    _ = len(ideal)
    if failed[0] != 0:
        raise Error("sampling failed for at least one XEB circuit")
    print("sampled", npairs, "noisy circuits x", SHOTS, "trajectory shots in", secs, "s")

    var xs = List[Float64]()
    var fs = List[Float64]()
    print("  depth   XEB fidelity")
    for di in range(nd):
        var sxy = 0.0
        var sxx = 0.0
        for ci in range(N_CIRCUITS):
            var k = di * N_CIRCUITS + ci
            var e_u = 0.0
            for s in range(4):
                e_u += ideal[k][s] * ideal[k][s]
            var x = e_u - 0.25
            var y = m_u[k] - 0.25
            sxy += x * y
            sxx += x * x
        var f = sxy / sxx
        xs.append(Float64(depths[di]))
        fs.append(f)
        print("  ", depths[di], "   ", f)
    var fit = fit_decay(xs, fs, 0.0)
    var e_depol = E_PAULI / (1.0 - 1.0 / 16.0)
    var model = (1.0 - e_depol) ** 4
    print("fit:", fit)
    print("XEB layer fidelity:", fit.rate, " noise model fidelity:", model, "(page: 0.9787 +- 1.79e-4 vs 0.9788)")
    _ = log.close("layer fidelity vs noise model", fit.rate, model, 3e-3)
    _ = log.check("XEB fidelity decays with depth", fs[0] > fs[len(fs) - 1],
                  "f(1)=" + String(fs[0]) + " f(100)=" + String(fs[len(fs) - 1]))
    _ = log.within("XEB fidelity at depth 1", fs[0], 0.9, 1.1)
    log.finish()
