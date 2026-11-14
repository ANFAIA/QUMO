"""QMCI with a user-defined Grover operator: amplitude estimation of a Monte Carlo sum.

Reproduces: Classiq library - Quantum Monte Carlo Integration (QMCI)
Framework: QUMO - https://github.com/ANFAIA/QUMO
Source: https://github.com/Classiq/classiq-library/blob/main/algorithms/amplitude_amplification_and_estimation/qmc_user_defined/qmc_user_defined.ipynb (Classiq Library, MIT)

The notebook estimates a = sum_i p_i f_i with p_i = i/28 on 3 qubits and f_i = sin^2(0.25 i + 0.2), exact value 0.83384. It loads A|0> = sum_i sqrt(p_i)|i>(sqrt(f_i)|1> + sqrt(1 - f_i)|0>), builds the Grover operator from A, the zero reflection and the good-state reflection, and runs 3-qubit QPE on it. QUMO builds the same construction on 7 qubits. Checks: the joint law of (x, ind) after A against p_i f_i, P(ind = 1) against the notebook's exact amplitude, the exact QPE phase distribution against the ideal QPE kernel of the eigenphases +-arcsin(sqrt(a))/pi, every one of the 8 seeded sampled histogram bins against that distribution (3 sigma each, sigma = sqrt(p (1 - p) / shots)), and the notebook's printed measured amplitude 0.8535533905932737 with its |measured - exact| < 0.1 assertion.

Deviations from the notebook:
* sqrt(p_i) is loaded with QUMO's amplitude embedding (a Mottonen RY tree), not inplace_prepare_state. The controlled rotations of linear_pauli_rotations (RY(0.5 * 2^k) controlled on x_k) are each written as RY, CX, RY, CX, because Circuit.controlled(), which qpe() uses, does not accept CRY.
* The notebook's zero_oracle (within_apply with ind ^= (x == 0)) is written as S_0 = I - 2|0><0| on all four qubits: X on every qubit, a three-control phase pi with no ancillas, X on every qubit. Both give the same operator.
* The notebook's phase(pi) is the global phase of the Grover operator (Circuit.global_phase = pi). qpe() turns it into a controlled phase, which QPE needs.
* The notebook's text writes Q as -S_psi1 A^dag S_0 A. Its code runs invert(state_loading) first and state_loading last (within_apply), which is -A S_0 A^dag S_good. This file implements the code's order.
* Qubit labels: ind is qubit 0 of the system register and x bit k is qubit 1 + k, mirroring the notebook's state[0] and state[1:].
* The notebook's phase register is signed, QNum[3, SIGNED, 3]: bin 5/8 is read as -3/8 there. This file uses unsigned bins 0..7 (k/8) and checks the signed reading separately. sin^2(pi * phase) is the same for +3/8 and -3/8.
* The notebook's assertion |measured - exact| < 0.1 is its own bound, not noise-derived; it is repeated here for fidelity and adds no coverage beyond the two exact 1e-12 comparisons.
* The notebook's sampling settings are not printed. This file samples 4000 shots with seed 2026. Bins 3/8 and 5/8 have equal exact probability (about 0.49 each), so the top sampled bin can be either; both give the same amplitude sin^2(3 pi / 8).

Run: .venv/bin/mojo run -I . examples/frameworks/classiq/qmc_user_defined.mojo
example-meta: {"framework": "classiq", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from std.math import asin, cos, pi, sin, sqrt
from qumo.ir.circuit import Circuit
from qumo.ir.mc import append_mcp
from qumo.algorithms.library import bitstring_to_int, qpe
from qumo.algorithms.qml import append_amplitude_embedding_real
from qumo.sim.runner import evolve, sample
from qumo.testing.checks import CheckLog


comptime N_X = 3  # x register: 8 values on qubits 1..3
comptime N_QPE = 3  # evaluation register (the notebook's n_qpe)
comptime SHOTS = 4000


def prob(i: Int) -> Float64:
    """Probability p_i = i / 28: the notebook's linspace(0, 1, 8) divided by its sum."""
    return Float64(i) / 28.0


def f_good(i: Int) -> Float64:
    """Amplitude f_i = sin^2(0.25 i + 0.2) = sin^2(theta_i / 2) with theta_i = 0.5 i + 0.4."""
    var s = sin(0.25 * Float64(i) + 0.2)
    return s * s


def add_cry(mut c: Circuit, theta: Float64, control: Int, target: Int) raises:
    """Controlled RY(theta) as RY(theta/2), CX, RY(-theta/2), CX in time order."""
    c.ry(theta / 2.0, target)
    c.cx(control, target)
    c.ry(-theta / 2.0, target)
    c.cx(control, target)


def state_loading() raises -> Circuit:
    """A on 4 qubits: qubit 0 = ind, qubits 1..3 = x (bit k on qubit 1 + k).

    inplace_prepare_state(p) puts sqrt(p_i) on x. linear_pauli_rotations(Y, slope 0.5,
    offset 0.4) rotates ind by theta = 0.5 i + 0.4 for x = i: RY(0.4) on ind plus
    RY(0.5 * 2^k) controlled on x_k, so P(ind = 1 | x = i) = sin^2(theta / 2) = f_i.
    """
    var c = Circuit(4, name="state_loading")
    var amps = List[Float64]()
    for i in range(1 << N_X):
        amps.append(sqrt(prob(i)))
    var wires = List[Int]()  # wires[0] carries the most significant index bit
    wires.append(3)
    wires.append(2)
    wires.append(1)
    append_amplitude_embedding_real(c, amps, wires)
    c.ry(0.4, 0)
    for k in range(N_X):
        add_cry(c, 0.5 * Float64(1 << k), 1 + k, 0)
    return c^


def grover_operator() raises -> Circuit:
    """Q = -A S_0 A^dag S_good on the 4 system qubits.

    Time order, as in the notebook's code: S_good, A^dag, S_0, A, then the global phase -1.
    S_good = Z on ind (-1 on the good states ind = 1). S_0 = I - 2|0000><0000|.
    """
    var a = state_loading()
    var c = Circuit(4, name="qmci_grover")
    c.z(0)  # S_good
    c.compose(a.inverse())  # A^dag
    for q in range(4):  # S_0 = X^4 (three-control phase pi) X^4
        c.x(q)
    var ctrl = List[Int]()
    ctrl.append(0)
    ctrl.append(1)
    ctrl.append(2)
    append_mcp(c, pi, ctrl, 3)
    for q in range(4):
        c.x(q)
    c.compose(a)  # A
    c.global_phase = pi  # the notebook's phase(pi)
    return c^


def qpe_kernel(phi: Float64, k: Int, n: Int) -> Float64:
    """Ideal QPE probability of bin k for an eigenphase phi: |(1/N) sum_j exp(2 pi i j (phi - k/N))|^2, N = 2^n."""
    var nn = 1 << n
    var d = phi - Float64(k) / Float64(nn)
    var re = 0.0
    var im = 0.0
    for j in range(nn):
        var ang = 2.0 * pi * Float64(j) * d
        re += cos(ang)
        im += sin(ang)
    return (re * re + im * im) / Float64(nn * nn)


def main() raises:
    var log = CheckLog("classiq/qmc_user_defined")

    # ---- exact amplitude by brute force (independent of QUMO) ----
    var a_exact = 0.0
    for i in range(1 << N_X):
        a_exact += prob(i) * f_good(i)
    var theta = asin(sqrt(a_exact))  # sin^2(theta) = a; Q has eigenvalues exp(+-2 i theta)
    print("exact amplitude a =", a_exact, " eigenphase theta/pi =", theta / pi)
    _ = log.close("brute-force amplitude matches the notebook exact value", a_exact, 0.8338393824876795, 1e-12)

    # ---- A|0>: joint law of (x, ind), basis index = ind + 2 x ----
    var a = state_loading()
    var probs = evolve(a).probabilities()
    var dev = 0.0
    var p_good = 0.0
    for i in range(1 << N_X):
        var pg = probs[1 + 2 * i]
        var pb = probs[2 * i]
        dev = max(dev, abs(pg - prob(i) * f_good(i)))
        dev = max(dev, abs(pb - prob(i) * (1.0 - f_good(i))))
        p_good += pg
    _ = log.close("A|0> joint law of (x, ind) matches p_i f_i and p_i (1 - f_i) (max deviation)", dev, 0.0, 1e-12)
    _ = log.close("P(ind = 1) after A matches the notebook exact amplitude", p_good, 0.8338393824876795, 1e-12)

    # ---- QPE of Q: exact distribution of the 3-bit phase register ----
    var qsv = evolve(qpe(grover_operator(), N_QPE, a.copy(), False))
    var qprobs = qsv.probabilities()
    var dist = List[Float64](length=1 << N_QPE, fill=0.0)
    for idx in range(len(qprobs)):
        dist[idx % (1 << N_QPE)] += qprobs[idx]  # evaluation qubits are the low bits
    var phi = theta / pi
    var ideal = List[Float64]()
    var dev_qpe = 0.0
    for k in range(1 << N_QPE):
        # Q has eigenvalues exp(+-2 i theta), each with weight 1/2 in A|0>
        var pk = 0.5 * (qpe_kernel(phi, k, N_QPE) + qpe_kernel(-phi, k, N_QPE))
        ideal.append(pk)
        dev_qpe = max(dev_qpe, abs(dist[k] - pk))
    print("QUMO exact phase distribution:", dist[0], dist[1], dist[2], dist[3], dist[4], dist[5], dist[6], dist[7])
    print("ideal phase distribution:     ", ideal[0], ideal[1], ideal[2], ideal[3], ideal[4], ideal[5], ideal[6], ideal[7])
    _ = log.close("QPE phase distribution matches the ideal kernel (max deviation)", dev_qpe, 0.0, 1e-12)

    # ---- sampled phase register (the notebook's sample(qprog_3)) ----
    var res = sample(qpe(grover_operator(), N_QPE, a.copy(), True), SHOTS, UInt64(2026))
    var freq = List[Int](length=1 << N_QPE, fill=0)
    for kv in res.counts.items():
        freq[bitstring_to_int(kv.key)] += kv.value
    var top = 0
    for k in range(1 << N_QPE):
        if freq[k] > freq[top]:
            top = k
    print("sampled counts per phase bin (k/8):", freq[0], freq[1], freq[2], freq[3], freq[4], freq[5], freq[6], freq[7])
    _ = log.check("most frequent sampled phase bin is 3/8 or 5/8 (notebook prints 0.375)", top == 3 or top == 5, "top=" + String(top))
    for k in range(1 << N_QPE):
        var pk = ideal[k]
        var fk = Float64(freq[k]) / Float64(SHOTS)
        var sk = sqrt(pk * (1.0 - pk) / Float64(SHOTS))
        _ = log.close("sampled frequency of phase bin " + String(k) + "/8 matches the ideal (3 sigma)", fk, pk, 3.0 * sk)
    var p35 = ideal[3] + ideal[5]
    var f35 = Float64(freq[3] + freq[5]) / Float64(SHOTS)
    var sigma = sqrt(p35 * (1.0 - p35) / Float64(SHOTS))
    _ = log.close("sampled probability of bins 3 and 5 matches the ideal (3 sigma)", f35, p35, 3.0 * sigma)

    # ---- the notebook's measured amplitude sin^2(pi * phase) from the top bin ----
    var phase_top = Float64(top) / Float64(1 << N_QPE)
    var phase_signed = phase_top
    if top >= (1 << (N_QPE - 1)):
        phase_signed = phase_top - 1.0  # the notebook's SIGNED register: 5/8 reads as -3/8
    print("top bin as the notebook's signed phase:", phase_signed)
    _ = log.close("signed phase of the top bin is +-0.375 as in the notebook", abs(phase_signed), 0.375, 1e-12)
    var measured = sin(pi * phase_top) * sin(pi * phase_top)
    print("measured amplitude =", measured, " exact amplitude =", a_exact)
    _ = log.close("measured amplitude from the top bin matches the notebook print", measured, 0.8535533905932737, 1e-12)
    _ = log.check("notebook assertion |measured - exact| < 0.1 holds", abs(measured - a_exact) < 0.1, "diff=" + String(abs(measured - a_exact)))

    log.finish()
