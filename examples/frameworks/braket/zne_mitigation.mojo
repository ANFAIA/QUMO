"""Zero-noise extrapolation with unitary folding (Amazon Braket + Mitiq notebook).

Reproduces: Amazon Braket - Zero noise extrapolation with Mitiq (error_mitigation/on_mitiq)
Source: https://github.com/amazon-braket/amazon-braket-examples/tree/main/examples/error_mitigation/on_mitiq (2_Zero_noise_extrapolation_with_mitiq.ipynb; Apache-2.0)

The notebook mitigates local <Z_i> of a 1D transverse-field Ising time
evolution (RX "self-interaction" layers and CNOT-RZ-CNOT ZZ gadgets on even
then odd bonds) under a device noise model, with Mitiq's unitary folding and
the Linear, Exponential and Richardson factories. QUMO reproduces the
workflow with `qumo.algorithms.mitigation`:
- the circuit: 5 qubits, 4 Trotter steps of RX(0.3) on every qubit and
  ZZ gadgets CX . RZ(0.4) . CX on bonds (0,1) (2,3) then (1,2) (3,4);
- the observable: the mean magnetisation M = (1/5) sum_i <Z_i>;
- the noise: a vendor-neutral `NoiseModel` with the superconducting modality
  preset's gate errors (e1 = 2.5e-4, e2 = 3e-3 from
  `HardwareProfile.superconducting()`), i.e. DEPOLARIZE1 after 1q and
  DEPOLARIZE2 after 2q gates, applied AFTER folding (the measurement-basis
  change and readout are part of the noisy circuit);
- noise scales [1, 3, 5], global folding (Mitiq fold_global) and random local
  folding (fold_gates_at_random), 20,000 trajectory shots per scale;
- Richardson, linear and exponential (asymptote 0) extrapolation.
The notebook uses a vendor's emulator properties, 8 qubits and 10 steps,
and prints no mitigated numbers, so the instance and the noise differ; the
reference is the exact ideal value from the state vector.

Checks:
- every folded circuit is unitarily equivalent to the original (statevector, 1e-12);
- global folding + Richardson: |ZNE - ideal| < 0.5 |raw - ideal| (fixed seed);
- the raw (scale 1) value is biased by the noise (|raw - ideal| > 3 sigma).

Run: .venv/bin/mojo run -I . examples/frameworks/braket/zne_mitigation.mojo
example-meta: {"framework": "braket", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from qumo.ir.circuit import Circuit
from qumo.ir.pauli import PauliSum
from qumo.sim.runner import evolve
from qumo.sim.statevector import StateVector
from qumo.exec.resources import HardwareProfile
from qumo.qec.characterization import NoiseModel
from qumo.algorithms.mitigation import (
    fold_gates_global, fold_gates_at_random, folded_scale, zne_expectation,
    richardson_extrapolate, linear_extrapolate, exp_extrapolate, ZNEResult,
)
from qumo.ai.checks import CheckLog

comptime NQ = 5
comptime STEPS = 4
comptime SHOTS = 20000


def ising_steps(n: Int, steps: Int, theta_x: Float64, theta_zz: Float64) -> Circuit:
    var c = Circuit(n, name="tfim_trotter")
    for _ in range(steps):
        for q in range(n):
            c.rx(theta_x, q)
        for parity in range(2):
            var q = parity
            while q + 1 < n:
                c.cx(q, q + 1)
                c.rz(theta_zz, q + 1)
                c.cx(q, q + 1)
                q += 2
    return c^


def max_diff(a: StateVector, b: StateVector) -> Float64:
    var d = 0.0
    for i in range(a.size()):
        d = max(d, max(abs(a.re[i] - b.re[i]), abs(a.im[i] - b.im[i])))
    return d


def main() raises:
    var log = CheckLog("braket/zne_mitigation")
    var c = ising_steps(NQ, STEPS, 0.3, 0.4)
    var m = PauliSum()
    for q in range(NQ):
        m.add(1.0 / Float64(NQ), 0, UInt64(1) << UInt64(q))
    var ideal_sv = evolve(c)
    var ideal = ideal_sv.expectation(m)
    print("circuit:", c.num_qubits, "qubits,", c.num_unitary_gates(), "gates,", c.num_2q_gates(), "CX; ideal M =", ideal)

    var sc = HardwareProfile.superconducting()
    var nm = NoiseModel(NQ)
    for q in range(NQ):
        nm.gate_error_1q[q] = sc.err_1q
    nm.gate_error_2q = sc.err_2q
    print("noise: superconducting preset e1 =", sc.err_1q, " e2 =", sc.err_2q,
          "-> DEPOLARIZE1(", nm.depolarizing_1q(0), "), DEPOLARIZE2(", nm.depolarizing_2q(0, 1), ")")

    # folding equivalence
    var scales: List[Float64] = [1.0, 3.0, 5.0]
    var fold_err = 0.0
    for s in scales:
        var g = fold_gates_global(c, s)
        var r = fold_gates_at_random(c, s, 99)
        fold_err = max(fold_err, max(max_diff(evolve(g), ideal_sv), max_diff(evolve(r), ideal_sv)))
        print("  scale", s, ": global", g.num_unitary_gates(), "gates (x", folded_scale(c, g), "), random",
              r.num_unitary_gates(), "gates (x", folded_scale(c, r), ")")
    _ = log.close("folded circuits equal the original statevector", fold_err, 0.0, 1e-12)

    var zg = zne_expectation(c, m, nm, scales, SHOTS, 2024, "richardson", "global")
    var zr = zne_expectation(c, m, nm, scales, SHOTS, 2025, "richardson", "random")
    print("global folding:", zg)
    print("random folding:", zr)
    var raw = zg.raw()
    var sigma = zg.stds[0]
    print("  method        global folding        random folding    (ideal", ideal, ", raw", raw, "+-", sigma, ")")
    var g_lin = linear_extrapolate(zg.achieved_scales, zg.values)
    var r_lin = linear_extrapolate(zr.achieved_scales, zr.values)
    var g_exp = exp_extrapolate(zg.achieved_scales, zg.values, 0.0)
    var r_exp = exp_extrapolate(zr.achieved_scales, zr.values, 0.0)
    print("  richardson   ", zg.estimate, "  ", zr.estimate)
    print("  linear       ", g_lin, "  ", r_lin)
    print("  exponential  ", g_exp, "  ", r_exp)
    print("  |error|: raw", abs(raw - ideal), " richardson", abs(zg.estimate - ideal), " linear", abs(g_lin - ideal),
          " exp", abs(g_exp - ideal))
    _ = log.within("raw bias in standard errors", abs(raw - ideal) / sigma, 3.0, 1000.0)
    _ = log.within("global+Richardson error / raw error", abs(zg.estimate - ideal) / abs(raw - ideal), 0.0, 0.5)
    log.finish()
