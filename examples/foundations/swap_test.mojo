"""Swap test: the notebook's overlap estimate for two random 3-qubit states on QUMO.

Reproduces: Classiq library - Swap Test Algorithm
Source: https://github.com/Classiq/classiq-library/blob/main/algorithms/quantum_primitives/swap_test/swap_test.ipynb (Classiq Library, MIT)
Framework: Classiq - https://github.com/Classiq/classiq-library

The notebook prepares two real 8-amplitude states (numpy seed 0, entries 1 - 2 rand(8), normalised), runs the swap test on them (H on a test qubit, a controlled-SWAP of each qubit pair of the two registers, H again) and reads the overlap from P(0) as |<psi1|psi2>| = sqrt(2 P(0) - 1). The exact value is P(0) = (1 + |<psi1|psi2>|^2) / 2. The checks compare the QUMO statevector with that value and with the brute-force overlap, which reproduces the notebook's printed 0.10746808371981542. They also check the amplitude preparation, the notebook's own ATOL = 0.05 test on 100000 sampled shots, and the two cases the notebook text describes: identical states give P(0) = 1 and orthogonal states give P(0) = 1/2.

Deviations from the notebook:
* Classiq's prepare_amplitudes is QUMO's append_amplitude_embedding_real (a Mottonen RY/CX tree), and Classiq's swap_test library function is written out as H, controlled-SWAP and H on the test qubit. The prepared states are the same; the signed real amplitudes are exact.
* QUMO has no numpy-compatible generator, so the 16 draws of np.random.seed(0) (1 - 2 rand(8), twice) are hardcoded here. The normalisation and the dot product are done in this file. The dot product matches the notebook's printed classical overlap to 1e-12.
* The notebook samples on Classiq's simulator as a platform job. QUMO samples its own statevector with a fixed seed, so its shot estimate differs from the notebook's printed swap-test overlap 0.0743 by shot noise, and that comparison uses a statistical tolerance.
* Extra checks beyond the notebook: the exact state of both registers, identical inputs and orthogonal inputs.
* No Classiq synthesis, show(), platform links or cloud job.

Run: .venv/bin/mojo run -I . examples/foundations/swap_test.mojo
example-meta: {"framework": "classiq", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from std.math import sqrt
from qumo.ir.circuit import Circuit
from qumo.algorithms.qml import append_amplitude_embedding_real
from qumo.sim.runner import evolve, sample
from qumo.testing.checks import CheckLog


comptime NQ = 3  # qubits per register (the notebook's NUM_QUBITS)
comptime DIM = 8  # 2^NQ amplitudes per state
comptime TEST = 2 * NQ  # the test qubit; the registers sit on 0..2 and 3..5
comptime NUM_SHOTS = 100_000  # the notebook's NUM_SHOTS
comptime ATOL = 0.05  # the notebook's ATOL
comptime NB_EXACT = 0.10746808371981542  # the notebook's printed classical overlap
comptime NB_SWAP = 0.07429670248402663  # the notebook's printed swap-test overlap


def raw_psi1() -> List[Float64]:
    """The first 8 draws 1 - 2 rand(8) of np.random.seed(0), before normalisation."""
    var v: List[Float64] = [
        -0.0976270078546495, -0.43037873274483895, -0.20552675214328775, -0.08976636599379373,
        0.15269040132219058, -0.29178822613331223, 0.12482557747461498, -0.7835460015641595,
    ]
    return v^


def raw_psi2() -> List[Float64]:
    """The next 8 draws 1 - 2 rand(8) of the same seed, before normalisation."""
    var v: List[Float64] = [
        -0.9273255210020586, 0.2331169623484446, -0.5834500761653292, -0.05778983950580896,
        -0.13608912218786462, -0.8511932765853221, 0.8579278836042261, 0.8257414005969186,
    ]
    return v^


def normalised(raw: List[Float64]) -> List[Float64]:
    var s = 0.0
    for v in raw:
        s += v * v
    var inv = 1.0 / sqrt(s)
    var out = List[Float64]()
    for v in raw:
        out.append(v * inv)
    return out^


def dot(a: List[Float64], b: List[Float64]) -> Float64:
    var s = 0.0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s


def basis_state(k: Int) -> List[Float64]:
    var v = List[Float64](length=DIM, fill=0.0)
    v[k] = 1.0
    return v^


def prep_register(mut c: Circuit, amps: List[Float64], base: Int) raises:
    """Amplitudes on qubits base..base+NQ-1, with qubit base+j carrying bit j of the index."""
    var wires = List[Int]()
    for i in range(NQ):
        wires.append(base + NQ - 1 - i)
    append_amplitude_embedding_real(c, amps, wires)


def swap_test(a: List[Float64], b: List[Float64], measure: Bool) raises -> Circuit:
    """Prepare a and b, then H on the test qubit, a controlled-SWAP of each qubit pair, H again, and optionally measure the test qubit into clbit 0."""
    var c = Circuit(2 * NQ + 1, 1 if measure else 0, "swap_test")
    prep_register(c, a, 0)
    prep_register(c, b, NQ)
    c.h(TEST)
    for i in range(NQ):
        c.cswap(TEST, i, NQ + i)
    c.h(TEST)
    if measure:
        c.measure(TEST, 0)
    return c^


def p_test_zero(probs: List[Float64]) -> Float64:
    """P(test = 0): the test qubit is the top qubit, so it is 0 on the basis states below 2^TEST."""
    var total = 0.0
    for i in range(1 << TEST):
        total += probs[i]
    return total


def product_prep_error(a: List[Float64], b: List[Float64]) raises -> Float64:
    """Largest deviation of the prepared 6-qubit state from a[k1] * b[k2] at index k1 + DIM * k2, including imaginary parts."""
    var c = Circuit(2 * NQ, 0, "prep_pair")
    prep_register(c, a, 0)
    prep_register(c, b, NQ)
    var sv = evolve(c)
    var worst = 0.0
    for k2 in range(DIM):
        for k1 in range(DIM):
            var idx = k1 + DIM * k2
            worst = max(worst, abs(sv.re[idx] - a[k1] * b[k2]))
            worst = max(worst, abs(sv.im[idx]))
    return worst


def binom_sigma(p: Float64, shots: Int) -> Float64:
    """Standard error of a frequency p estimated from `shots` samples."""
    return sqrt(p * (1.0 - p) / Float64(shots))


def main() raises:
    var log = CheckLog("classiq/swap_test")
    var a = normalised(raw_psi1())
    var b = normalised(raw_psi2())
    var c_ov = dot(a, b)

    # ---- exact: amplitude preparation and the swap test ----
    _ = log.close("prepared amplitudes of both registers match the normalised notebook vectors (max deviation)", product_prep_error(a, b), 0.0, 1e-12)
    _ = log.close("classical overlap |<psi1|psi2>| equals the notebook's printed 0.10746808371981542", abs(c_ov), NB_EXACT, 1e-12)
    var p0 = p_test_zero(evolve(swap_test(a, b, False)).probabilities())
    var p0_ref = 0.5 * (1.0 + c_ov * c_ov)
    print("exact: P(0) =", p0, " overlap =", sqrt(2.0 * p0 - 1.0))
    _ = log.close("exact P(test = 0) equals (1 + |<psi1|psi2>|^2) / 2", p0, p0_ref, 1e-12)
    _ = log.close("exact swap-test overlap sqrt(2 P(0) - 1) equals |<psi1|psi2>|", sqrt(2.0 * p0 - 1.0), abs(c_ov), 1e-9)

    # ---- the two cases the notebook text describes ----
    var p_same = p_test_zero(evolve(swap_test(a, a, False)).probabilities())
    _ = log.close("identical states give P(test = 0) = 1", p_same, 1.0, 1e-12)
    var p_orth = p_test_zero(evolve(swap_test(basis_state(0), basis_state(1), False)).probabilities())
    _ = log.close("orthogonal basis states |000> and |001> give P(test = 0) = 1/2", p_orth, 0.5, 1e-12)

    # ---- sampled: 100000 shots of the measured test qubit, as in the notebook ----
    var res = sample(swap_test(a, b, True), NUM_SHOTS, seed=UInt64(20261103))
    var zeros = 0
    for m in res.memory:
        if m == 0:
            zeros += 1
    var p0_s = Float64(zeros) / Float64(NUM_SHOTS)
    var overlap_s = sqrt(2.0 * p0_s - 1.0)
    var sig = binom_sigma(p0, NUM_SHOTS)
    var p0_nb = 0.5 * (1.0 + NB_SWAP * NB_SWAP)
    var sig_nb = binom_sigma(p0_nb, NUM_SHOTS)
    print("sampled from", NUM_SHOTS, "shots: P(0) =", p0_s, " overlap =", overlap_s, "| notebook: overlap =", NB_SWAP, " classical =", NB_EXACT)
    _ = log.close("sampled P(test = 0) agrees with the exact value within 4 standard errors", p0_s, p0, 4.0 * sig)
    _ = log.close("sampled overlap agrees with the classical overlap within the notebook's ATOL 0.05", overlap_s, abs(c_ov), ATOL)
    _ = log.close("sampled P(test = 0) agrees with the notebook's printed overlap within shot noise", p0_s, p0_nb, 4.0 * sqrt(sig * sig + sig_nb * sig_nb))

    log.finish()
