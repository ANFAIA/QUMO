"""Hadamard test for the QFT unitary: Re<0000|QFT|0000> = 1/4, read from P(0) = 5/8 on the control.

Reproduces: Classiq library - Hadamard Test
Source: https://github.com/Classiq/classiq-library/blob/main/algorithms/quantum_primitives/hadamard_test/hadamard_test.ipynb (Classiq Library, MIT)
Framework: Classiq - https://github.com/Classiq/classiq-library

The notebook puts a control qubit in |+>, applies the QFT on a 4-qubit target psi conditioned on the control being |1>, applies H to the control again and measures it. Then P(0) = (1 + Re<psi|U|psi>) / 2, so Re<psi|U|psi> = 2 P(0) - 1. For psi = |0000> the QFT gives |++++>, so Re<0000|U|0000> = 1/4, P(0) = 0.625 and P(1) = 0.375. The checks compare the exact statevector with these values, read the amplitude <0000|QFT|0000> off the unitary, sweep the Hadamard test over all 16 basis inputs against Re<j|QFT|j> = cos(2 pi j^2 / 16) / 4 (from the QFT definition), and compare 100000 sampled shots with the analytic values and with the notebook's printed P_0 = 0.62788 and Re(<U>) = 0.25576, at 4 standard errors of shot noise.

Deviations from the notebook:
* Classiq's control(), within_apply() and qft() on a QArray are replaced by QUMO's controlled_by(qft(4), [control]) between an explicit H and H on the control. This is the same V^dag U V shape, since QUMO has no within_apply.
* Target psi is qubits 0..3 and the control is qubit 4. The measured control goes to clbit 0.
* The notebook samples on Classiq's simulator as a platform job. QUMO samples its own statevector with a fixed seed, so its shot counts differ from the notebook's printed P_0 by shot noise only, and that comparison uses a statistical tolerance.
* The 16-input sweep and the amplitude check go beyond the notebook, which treats only |0000>. The sweep uses the convention of qumo.algorithms.library.qft() (QFT|j> = 2^-2 sum_k exp(2 pi i jk/16)|k>, qubit k holds bit k). The notebook's Classiq QFT may order its qubits differently, but that does not change the |0000> result.
* No Classiq synthesis, show(), platform links or cloud job.

Run: .venv/bin/mojo run -I . examples/foundations/hadamard_test.mojo
example-meta: {"framework": "classiq", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from std.math import cos, pi, sqrt
from qumo.ir.circuit import Circuit
from qumo.ir.mc import controlled_by
from qumo.algorithms.library import qft
from qumo.sim.runner import evolve, sample
from qumo.testing.checks import CheckLog


comptime NT = 4  # target psi on qubits 0..3, little-endian
comptime CTRL = 4  # control qubit (the notebook's expectation_value)
comptime NUM_SHOTS = 100_000  # the notebook's NUM_SHOTS
comptime NB_P0 = 0.62788  # the notebook's printed P_0
comptime NB_RE = 0.25576  # the notebook's printed Re(<U>) = 2 P_0 - 1


def hadamard_test(prep: Int, measure: Bool) raises -> Circuit:
    """Hadamard test with psi = |prep> (bit k on qubit k): H on the control, QFT on psi if the control is 1, H on the control, optional measurement."""
    var c = Circuit(NT + 1, 1 if measure else 0, "hadamard_test")
    for q in range(NT):
        if ((prep >> q) & 1) == 1:
            c.x(q)
    var controls = List[Int]()
    controls.append(CTRL)
    c.h(CTRL)
    c.compose(controlled_by(qft(NT), controls))
    c.h(CTRL)
    if measure:
        c.measure(CTRL, 0)
    return c^


def p_control_zero(probs: List[Float64]) -> Float64:
    """P(control = 0). The control is the top qubit, so it is 0 on the basis states below 2^CTRL."""
    var total = 0.0
    for i in range(1 << CTRL):
        total += probs[i]
    return total


def re_qft_analytic(j: Int) -> Float64:
    """Re<j|QFT|j> = cos(2 pi j^2 / 2^NT) / 2^(NT/2), from QFT|j> = 2^(-NT/2) sum_k exp(2 pi i j k / 2^NT)|k>."""
    var big_n = 1 << NT
    var angle = 2.0 * pi * Float64(j * j) / Float64(big_n)
    return cos(angle) / sqrt(Float64(big_n))


def binom_sigma(p: Float64, shots: Int) -> Float64:
    """Standard error of a frequency p estimated from `shots` samples."""
    return sqrt(p * (1.0 - p) / Float64(shots))


def main() raises:
    var log = CheckLog("classiq/hadamard_test")

    # ---- exact: the QFT unitary on |0000>, then the notebook's Hadamard test ----
    var qft_state = evolve(qft(NT))
    _ = log.close("QFT amplitude <0000|U|0000> (real part)", qft_state.re[0], 0.25, 1e-12)
    _ = log.close("QFT amplitude <0000|U|0000> (imaginary part)", qft_state.im[0], 0.0, 1e-12)

    var p0_exact = p_control_zero(evolve(hadamard_test(0, False)).probabilities())
    _ = log.close("exact P(0) for psi = |0000> is 5/8", p0_exact, 0.625, 1e-12)
    _ = log.close("exact Re<U> = 2 P(0) - 1 is 1/4", 2.0 * p0_exact - 1.0, 0.25, 1e-12)

    var worst = 0.0
    for j in range(1 << NT):
        var p0_j = p_control_zero(evolve(hadamard_test(j, False)).probabilities())
        worst = max(worst, abs(p0_j - 0.5 * (1.0 + re_qft_analytic(j))))
    _ = log.close("exact P(0) matches (1 + Re<j|U|j>) / 2 on all 16 basis inputs (max deviation)", worst, 0.0, 1e-12)

    # ---- sampled: 100000 shots of the measured control, as in the notebook ----
    var res = sample(hadamard_test(0, True), NUM_SHOTS, seed=UInt64(20261103))
    var zeros = 0
    for b in res.memory:
        if b == 0:
            zeros += 1
    var p0_s = Float64(zeros) / Float64(NUM_SHOTS)
    var re_s = 2.0 * p0_s - 1.0
    var sig_q = binom_sigma(0.625, NUM_SHOTS)
    var sig_nb = binom_sigma(NB_P0, NUM_SHOTS)
    print("exact: P(0) =", p0_exact, "Re(<U>) =", 2.0 * p0_exact - 1.0)
    print("sampled from", NUM_SHOTS, "shots: P_0 =", p0_s, "Re(<U>) =", re_s, "| notebook: P_0 =", NB_P0, "Re(<U>) =", NB_RE)
    _ = log.close("sampled P(0) agrees with exact 5/8 within 4 standard errors", p0_s, 0.625, 4.0 * sig_q)
    _ = log.close("sampled Re(<U>) agrees with exact 1/4 within 4 standard errors", re_s, 0.25, 8.0 * sig_q)
    _ = log.close("sampled P(0) agrees with the notebook's printed 0.62788 within shot noise", p0_s, NB_P0, 4.0 * sqrt(sig_q * sig_q + sig_nb * sig_nb))
    _ = log.close("sampled Re(<U>) agrees with the notebook's printed 0.25576 within shot noise", re_s, NB_RE, 8.0 * sqrt(sig_q * sig_q + sig_nb * sig_nb))

    log.finish()
