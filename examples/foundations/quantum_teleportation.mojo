"""Quantum teleportation: the notebook's protocol with deferred and mid-circuit corrections.

Reproduces: Classiq library - Quantum Teleportation Algorithm (Protocol)
Source: https://github.com/Classiq/classiq-library/blob/main/algorithms/foundational/quantum_teleportation/quantum_teleportation.ipynb (Classiq Library, MIT)
Framework: Classiq - https://github.com/Classiq/classiq-library

The notebook prepares Alice's qubit in |+> (H), makes a Bell pair with Bob's qubit, applies CX(alice, bell) and H(alice), measures Alice's two qubits, and applies X to Bob if the bell-pair bit is 1 and Z if the alice bit is 1. Its Technical Notes give Bob's four branch states. This file checks QUMO's amplitudes against those states before any measurement (exact, 1e-12), checks that the corrections turn every branch into the input state (exact), and runs the measure + c_if circuit by trajectory sampling: each Alice branch occurs with probability 1/4, and Bob's Z, X and Y readouts in every branch match the input's Bloch components within 5 binomial standard errors (fixed seeds). The notebook's |+> input is included: Bob reads 0 with probability 1/2 in the Z basis, and never 1 in the X basis.

Deviations from the notebook:
  * The Classiq synthesis, platform links, show() and the cloud sample() are replaced by QUMO's local sample() with 20000 shots and fixed seeds. The notebook's counts (1036 / 1012) come from one Classiq run, so only the distribution is compared.
  * The exact checks use the deferred-measurement form, CX(bell, bob) and CZ(alice, bob) in place of measure + if_, because evolve() refuses classically conditioned gates. The mid-circuit form is checked by sampling only.
  * Qubits: alice_qubit = 0, bell_pair_qubit = 1, bob_qubit = 2. Clbits: 0 = alice_qubit, 1 = bell_pair_qubit, 2 = Bob's readout.
  * Added beyond the notebook: a generic input cos(theta/2)|0> + e^{i phi} sin(theta/2)|1> (theta = 1.1, phi = 0.7), and Bob read out in the X and Y bases. |+> is an eigenstate of X, so the notebook's input cannot detect a wrong X correction.

Run: .venv/bin/mojo run -I . examples/foundations/quantum_teleportation.mojo
example-meta: {"framework": "classiq", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from std.math import cos, sin, sqrt
from qumo.ir.circuit import Circuit
from qumo.sim.runner import evolve, sample, SampleResult
from qumo.testing.checks import CheckLog

comptime SHOTS = 20000
comptime Z_BOUND = 5.0  # sampled checks: at most this many binomial standard errors from the analytic value
comptime GEN_THETA = 1.1
comptime GEN_PHI = 0.7


# ---------------------------------------------------------------------------
# Circuits. Qubits: 0 alice_qubit, 1 bell_pair_qubit, 2 bob_qubit.
# ---------------------------------------------------------------------------
def prepare(mut c: Circuit, notebook: Bool):
    """Alice's qubit: the notebook's H|0> = |+>, or cos(theta/2)|0> + e^{i phi} sin(theta/2)|1>."""
    if notebook:
        c.h(0)
    else:
        c.ry(GEN_THETA, 0)
        c.p(GEN_PHI, 0)


def teleport_deferred(notebook: Bool, corrected: Bool) raises -> Circuit:
    """The protocol without measurement. The corrections are gates: CX(bell, bob) is the notebook's X
    (bell bit = 1), CZ(alice, bob) its Z (alice bit = 1), applied in the notebook's order."""
    var c = Circuit(3, name="teleport_deferred")
    prepare(c, notebook)
    c.h(1)
    c.cx(1, 2)
    c.cx(0, 1)
    c.h(0)
    if corrected:
        c.cx(1, 2)
        c.cz(0, 2)
    return c^


def teleport_feed_forward(notebook: Bool, readout: Int) raises -> Circuit:
    """The notebook's circuit with real mid-circuit measurements and classically controlled corrections.
    readout: the basis of Bob's last measurement, 0 = Z (the notebook's), 1 = X, 2 = Y."""
    var c = Circuit(3, num_clbits=3, name="teleport")
    prepare(c, notebook)
    c.h(1)
    c.cx(1, 2)
    c.cx(0, 1)
    c.h(0)
    c.measure(0, 0)  # alice_qubit -> clbit 0
    c.measure(1, 1)  # bell_pair_qubit -> clbit 1
    c.x(2)
    c.c_if(1, 1)  # X on Bob if the bell-pair bit is 1
    c.z(2)
    c.c_if(0, 1)  # Z on Bob if the alice bit is 1
    if readout == 1:
        c.h(2)  # X basis
    elif readout == 2:
        c.sdg(2)
        c.h(2)  # Y basis
    c.measure(2, 2)
    return c^


# ---------------------------------------------------------------------------
# Analytic references. A state is (Re alpha, Im alpha, Re beta, Im beta).
# ---------------------------------------------------------------------------
def input_amps(notebook: Bool) -> Tuple[Float64, Float64, Float64, Float64]:
    if notebook:
        var h = 1.0 / sqrt(2.0)
        return (h, 0.0, h, 0.0)
    return (cos(GEN_THETA / 2.0), 0.0, sin(GEN_THETA / 2.0) * cos(GEN_PHI), sin(GEN_THETA / 2.0) * sin(GEN_PHI))


def bloch(v: Tuple[Float64, Float64, Float64, Float64]) -> Tuple[Float64, Float64, Float64]:
    """(<X>, <Y>, <Z>) = (2 Re a*b, 2 Im a*b, |a|^2 - |b|^2)."""
    var ar = v[0]
    var ai = v[1]
    var br = v[2]
    var bi = v[3]
    return (2.0 * (ar * br + ai * bi), 2.0 * (ar * bi - ai * br), ar * ar + ai * ai - br * br - bi * bi)


def psi_on(v: Tuple[Float64, Float64, Float64, Float64], b: Int) -> Tuple[Float64, Float64]:
    """Amplitude of the input state on |b>."""
    if b == 0:
        return (v[0], v[1])
    return (v[2], v[3])


def branch_ref(a: Int, p: Int, b: Int, v: Tuple[Float64, Float64, Float64, Float64]) -> Tuple[Float64, Float64]:
    """Technical Notes, before any correction: the amplitude on Bob's |b> when Alice's pair is |a p>, with
    the 1/2 of the Bell-basis expansion. The term is alpha when p XOR b = 0 and beta otherwise, with a minus
    sign when a = 1 and p XOR b = 1: 00 -> alpha|0> + beta|1>, 01 -> beta|0> + alpha|1>,
    10 -> alpha|0> - beta|1>, 11 -> -beta|0> + alpha|1>."""
    var t = p ^ b
    var re = v[0] if t == 0 else v[2]
    var im = v[1] if t == 0 else v[3]
    var s = -0.5 if (a == 1 and t == 1) else 0.5
    return (s * re, s * im)


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------
def tally(memory: List[Int], mut n: List[Int], mut zeros: List[Int]):
    """Shots per branch (bit 0 = alice_qubit, bit 1 = bell_pair_qubit) and Bob's zeros per branch (clbit 2)."""
    for creg in memory:
        var br = creg & 3
        n[br] += 1
        if ((creg >> 2) & 1) == 0:
            zeros[br] += 1


def run_readout(notebook: Bool, readout: Int, seed: UInt64, mut n: List[Int], mut zeros: List[Int]) raises:
    var res = sample(teleport_feed_forward(notebook, readout), SHOTS, seed)
    tally(res.memory, n, zeros)


def worst_z(trials: List[Int], hits: List[Int], expect: Float64) -> Float64:
    """Largest |hits/trials - expect| in binomial standard errors over the branches (0 < expect < 1)."""
    var var_unit = expect * (1.0 - expect)
    var worst = 0.0
    for br in range(len(trials)):
        var f = Float64(hits[br]) / Float64(trials[br])
        worst = max(worst, abs(f - expect) / sqrt(var_unit / Float64(trials[br])))
    return worst


# ---------------------------------------------------------------------------
# Exact checks (state vectors)
# ---------------------------------------------------------------------------
def exact_checks(mut log: CheckLog, notebook: Bool, name: String) raises:
    var v = input_amps(notebook)

    # Before measurement: QUMO's amplitudes equal the notebook's four branch states.
    var pre = evolve(teleport_deferred(notebook, False))
    var worst = 0.0
    var worst_p = 0.0
    for a in range(2):
        for p in range(2):
            var prob = 0.0
            for b in range(2):
                var idx = a + 2 * p + 4 * b
                var want = branch_ref(a, p, b, v)
                worst = max(worst, abs(pre.re[idx] - want[0]) + abs(pre.im[idx] - want[1]))
                prob += pre.re[idx] * pre.re[idx] + pre.im[idx] * pre.im[idx]
            worst_p = max(worst_p, abs(prob - 0.25))
    _ = log.close(name + " states before measurement match the Technical Notes", worst, 0.0, 1e-12)
    _ = log.close(name + " each Alice branch has probability 1/4", worst_p, 0.0, 1e-12)

    # After the corrections: every branch leaves Bob in the input state, so the state is |+>|+> x input.
    var post = evolve(teleport_deferred(notebook, True))
    for a in range(2):
        for p in range(2):
            var dev = 0.0
            for b in range(2):
                var idx = a + 2 * p + 4 * b
                var want = psi_on(v, b)
                dev = max(dev, abs(post.re[idx] - 0.5 * want[0]) + abs(post.im[idx] - 0.5 * want[1]))
            _ = log.close(name + " corrected branch a=" + String(a) + " p=" + String(p) + " leaves the input on Bob", dev, 0.0, 1e-12)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() raises:
    var log = CheckLog("classiq/quantum_teleportation")

    # ---- exact: the notebook's |+> input and a generic input ----
    exact_checks(log, True, "notebook input")
    exact_checks(log, False, "generic input")

    # ---- sampled: the mid-circuit circuit with c_if, generic input, Bob read in Z, X and Y ----
    var bloch_in = bloch(input_amps(False))
    var full = List[Int](length=4, fill=SHOTS)
    var n_z = List[Int](length=4, fill=0)
    var zeros_z = List[Int](length=4, fill=0)
    run_readout(False, 0, UInt64(2026), n_z, zeros_z)
    print("generic input, Z readout: shots per branch", n_z[0], n_z[1], n_z[2], n_z[3], "of", SHOTS)
    var freq_worst = worst_z(full, n_z, 0.25)
    _ = log.within("generic input Alice branch frequencies within 5 standard errors (worst z)", freq_worst, 0.0, Z_BOUND)

    var z_expect = (1.0 + bloch_in[2]) / 2.0
    _ = log.within("generic input Bob Z readout within 5 standard errors in each branch (worst z)", worst_z(n_z, zeros_z, z_expect), 0.0, Z_BOUND)

    for readout in range(1, 3):
        var n = List[Int](length=4, fill=0)
        var zeros = List[Int](length=4, fill=0)
        run_readout(False, readout, UInt64(2026 + readout), n, zeros)
        var comp = bloch_in[0]
        var label = "X"
        if readout == 2:
            comp = bloch_in[1]
            label = "Y"
        var expect = (1.0 + comp) / 2.0
        _ = log.within("generic input Bob " + label + " readout within 5 standard errors in each branch (worst z)", worst_z(n, zeros, expect), 0.0, Z_BOUND)

    # ---- sampled: the notebook's |+> input, its Z readout (1036 of 2048 zeros on the page) and an X readout ----
    var nb_n = List[Int](length=4, fill=0)
    var nb_zeros = List[Int](length=4, fill=0)
    run_readout(True, 0, UInt64(2030), nb_n, nb_zeros)
    var nb_total = 0
    var nb_zero_total = 0
    for br in range(4):
        nb_total += nb_n[br]
        nb_zero_total += nb_zeros[br]
    print("notebook input, Z readout:", nb_zero_total, "zeros of", nb_total, "shots (notebook page: 1036 of 2048)")
    var f_z = Float64(nb_zero_total) / Float64(nb_total)
    var z_nb = abs(f_z - 0.5) / sqrt(0.25 / Float64(nb_total))
    _ = log.within("notebook input Bob reads 0 with probability 1/2 in the Z basis (z)", z_nb, 0.0, Z_BOUND)

    var nb_x_n = List[Int](length=4, fill=0)
    var nb_x_zeros = List[Int](length=4, fill=0)
    run_readout(True, 1, UInt64(2031), nb_x_n, nb_x_zeros)
    var x_ones = 0
    for br in range(4):
        x_ones += nb_x_n[br] - nb_x_zeros[br]
    _ = log.close("notebook input Bob reads 1 in no shot in the X basis", Float64(x_ones), 0.0, 0.0)

    log.finish()
