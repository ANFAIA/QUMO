"""Oblivious amplitude amplification of an LCU block encoding: the notebook's H tensor H example on QUMO.

Reproduces: Classiq library - Oblivious Amplitude Amplification
Source: https://github.com/Classiq/classiq-library/blob/main/algorithms/amplitude_amplification_and_estimation/oblivious_amplitude_amplification/oblivious_amplitude_amplification.ipynb (Classiq Library, MIT)

The notebook block-encodes A = (X+Z)/sqrt2 tensor (X+Z)/sqrt2 = H tensor H, written as the Pauli sum 0.5 (Z0 X1 + X0 Z1 + X0 X1 + Z0 Z1), with s = 2. It then wraps the block encoding W in one oblivious amplitude amplification step, Q = W R W^dag O, where R reflects about |00> on the block register only. Before the step the block-00 probability is ||A psi||^2 / 4 = 1/4 (the notebook prints 0.2314 from 2048 shots); after it, sin^2(3 arcsin(1/2)) = 1 and the data register holds A psi. The checks: the Pauli sum is unitary and equals H tensor H; the block-00 part of W equals A/2 on all four data inputs; the block-00 probabilities and the data law |A psi|^2 are exact; 2048 seeded shots of the amplified circuit always read block 00 and follow |A psi|^2; the notebook's printed frequencies lie within 4 sigma of the exact law.

Deviations from the notebook: (1) QUMO has no LCU, PREP/SELECT or grover_operator primitive. The LCU is built by hand from existing gates. PREP is H on each block qubit, since the four terms have equal weight 1/2. SELECT applies term t to the data when the block register reads t, using CCZ and CCX gates with the zero bits of t X-conjugated. The oracle O and the reflection R are both I - 2 P_0 on the block register, so Q = W R W^dag R. (2) The input psi is the notebook's np.random.seed(1) vector, normalised. Its four amplitudes are copied from numpy 2.5.1 output, since QUMO has no NumPy MT19937 generator; qpe_for_matrix does the same. (3) The notebook runs through Classiq synthesis and a cloud simulator with 2048 shots. Here the state vector is exact, and the shots come from QUMO's seeded sampler. The notebook's printed counts are compared only as 4-sigma windows around the exact law. (4) Qubit layout: data on qubits 0-1 (data value k is basis index k), block on qubits 2-3. The notebook fixes no layout. (5) The notebook's show() call and Classiq job links are not reproduced.

Run: .venv/bin/mojo run -I . examples/frameworks/classiq/oblivious_amplitude_amplification.mojo
example-meta: {"framework": "classiq", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from std.math import asin, sin, sqrt
from qumo.ir.circuit import Circuit
from qumo.sim.runner import evolve, sample
from qumo.algorithms.qml import append_amplitude_embedding_real
from qumo.testing.checks import CheckLog


# ---------------------------------------------------------------------------
# The notebook's operator: 0.5 (Z0 X1 + X0 Z1 + X0 X1 + Z0 Z1), data qubit 0 least significant.
# Term t is selected by the block value t.
# ---------------------------------------------------------------------------
def term_pauli(t: Int, q: Int) -> String:
    """Pauli on data qubit q (0 or 1) in term t of the notebook's Pauli sum."""
    if q == 0:
        if t == 0 or t == 3:
            return String("Z")
        return String("X")
    if t == 0 or t == 2:
        return String("X")
    return String("Z")


def pauli_entry(p: String, r: Int, c: Int) -> Float64:
    """Entry (r, c) of the single-qubit Pauli X or Z."""
    if p == "X":
        return 1.0 if r != c else 0.0
    if r != c:
        return 0.0
    return 1.0 if r == 0 else -1.0


def pauli_sum_matrix() -> List[Float64]:
    """A = 0.5 sum_t P_t as a row-major 4 x 4 real matrix, built from the Pauli strings alone."""
    var a = List[Float64](length=16, fill=0.0)
    for t in range(4):
        for r in range(4):
            for c in range(4):
                var v0 = pauli_entry(term_pauli(t, 0), r & 1, c & 1)
                var v1 = pauli_entry(term_pauli(t, 1), (r >> 1) & 1, (c >> 1) & 1)
                a[4 * r + c] += 0.5 * v0 * v1
    return a^


def hadamard_tensor_matrix() -> List[Float64]:
    """H tensor H as a row-major 4 x 4 real matrix: entry (-1)^(r.c) / 2 with the bits of r and c."""
    var h = List[Float64](length=16, fill=0.0)
    for r in range(4):
        for c in range(4):
            var sign = 1.0
            if ((r & c) & 1) == 1:
                sign = -sign
            if (((r >> 1) & (c >> 1)) & 1) == 1:
                sign = -sign
            h[4 * r + c] = 0.5 * sign
    return h^


def apply4(m: List[Float64], v: List[Float64]) -> List[Float64]:
    var out = List[Float64](length=4, fill=0.0)
    for r in range(4):
        for c in range(4):
            out[r] += m[4 * r + c] * v[c]
    return out^


def unitarity_error(m: List[Float64]) -> Float64:
    """Max deviation of M^T M from the identity."""
    var worst = 0.0
    for i in range(4):
        for j in range(4):
            var s = 0.0
            for k in range(4):
                s += m[4 * k + i] * m[4 * k + j]
            var want = 1.0 if i == j else 0.0
            worst = max(worst, abs(s - want))
    return worst


def notebook_psi() -> List[Float64]:
    """The notebook's input vector: np.random.seed(1), np.random.rand(4), normalised (numpy 2.5.1)."""
    var psi: List[Float64] = [0.47092437141245563, 0.8134303596981151, 0.0001291583857834318, 0.3414107052353368]
    return psi^


# ---------------------------------------------------------------------------
# Circuits. Data on qubits 0-1, block on qubits 2-3; index = data + 4 * block.
# ---------------------------------------------------------------------------
def data_wires() -> List[Int]:
    var w = List[Int]()
    w.append(0)
    w.append(1)
    return w^


def block_wires() -> List[Int]:
    var w = List[Int]()
    w.append(2)
    w.append(3)
    return w^


def block_encoding(data: List[Int], block: List[Int]) raises -> Circuit:
    """W = PREP^dag SELECT PREP on 4 qubits. PREP is H on each block qubit; SELECT applies term t to the
    data when the block register reads t (CCZ / CCX, with the zero bits of t X-conjugated). The block-00
    part of W is A / 2."""
    var c = Circuit(4, name="lcu_block_encoding")
    c.h(block[0])
    c.h(block[1])
    for t in range(4):
        for b in range(2):
            if ((t >> b) & 1) == 0:
                c.x(block[b])
        for q in range(2):
            if term_pauli(t, q) == "Z":
                c.ccz(block[0], block[1], data[q])
            else:
                c.ccx(block[0], block[1], data[q])
        for b in range(2):
            if ((t >> b) & 1) == 0:
                c.x(block[b])
    c.h(block[0])
    c.h(block[1])
    return c^


def reflect_block_zero(mut c: Circuit, block: List[Int]) raises:
    """I - 2 P_0 on the block register: a phase -1 on block value 0 for every data value (X-conjugated CZ)."""
    c.x(block[0])
    c.x(block[1])
    c.cz(block[0], block[1])
    c.x(block[0])
    c.x(block[1])


def oaa_circuit(psi: List[Float64], steps: Int, measure: Bool) raises -> Circuit:
    """Input psi on the data register, then W, then `steps` copies of Q = W R W^dag O (the notebook's
    grover_operator with the block oracle O)."""
    var data = data_wires()
    var block = block_wires()
    var w = block_encoding(data, block)
    var w_dag = w.inverse()
    var c = Circuit(4, name="oblivious_amplitude_amplification")
    var wires = List[Int]()
    wires.append(1)
    wires.append(0)
    append_amplitude_embedding_real(c, psi, wires)
    c.compose(w)
    for _ in range(steps):
        reflect_block_zero(c, block)  # O: marks block value 0
        c.compose(w_dag)
        reflect_block_zero(c, block)  # R: reflection about |0> on the block register
        c.compose(w)
    if measure:
        for q in range(4):
            c.measure(q, q)
    return c^


def max_z(counts: List[Int], shots: Int, p: List[Float64]) -> Float64:
    """Largest |frequency - p| / sqrt(p (1 - p) / shots) over the outcomes (all p > 0 here)."""
    var worst = 0.0
    for d in range(len(p)):
        var f = Float64(counts[d]) / Float64(shots)
        var sigma = sqrt(p[d] * (1.0 - p[d]) / Float64(shots))
        worst = max(worst, abs(f - p[d]) / sigma)
    return worst


def main() raises:
    var log = CheckLog("classiq/oblivious_amplitude_amplification")
    var shots = 2048

    var psi = notebook_psi()
    var norm_psi = 0.0
    for x in psi:
        norm_psi += x * x
    _ = log.close("notebook input vector is normalised", norm_psi, 1.0, 1e-12)

    # ---- the operator, from the Pauli strings alone ----
    var a = pauli_sum_matrix()
    var had = hadamard_tensor_matrix()
    _ = log.close("Pauli sum 0.5 (ZX + XZ + XX + ZZ) is unitary (max deviation of A^T A - I)", unitarity_error(a), 0.0, 1e-12)
    var had_dev = 0.0
    for i in range(16):
        had_dev = max(had_dev, abs(a[i] - had[i]))
    _ = log.close("Pauli sum equals H tensor H (max entry deviation)", had_dev, 0.0, 1e-12)

    # analytic values: A psi, the block-00 probability before the step, and the data law after it
    var a_psi = apply4(a, psi)
    var law_after = List[Float64]()
    var good_before_exact = 0.0
    for d in range(4):
        var p = a_psi[d] * a_psi[d]
        law_after.append(p)
        good_before_exact += p / 4.0
    var law_before = List[Float64]()
    for d in range(4):
        law_before.append(law_after[d] / 4.0)

    # ---- the block encoding W: block 00 equals A / 2 on every data input ----
    var w = block_encoding(data_wires(), block_wires())
    var dev_w = 0.0
    for k in range(4):
        var c = Circuit(4, name="block_encoding_input")
        for q in range(2):
            if ((k >> q) & 1) == 1:
                c.x(q)
        c.compose(w)
        var sv = evolve(c)
        for d in range(4):
            dev_w = max(dev_w, abs(sv.re[d] - a[4 * d + k] / 2.0))
            dev_w = max(dev_w, abs(sv.im[d]))
    _ = log.close("LCU block 00 equals A/2 on all four data inputs (max deviation)", dev_w, 0.0, 1e-12)

    # ---- before the amplitude amplification step ----
    var before = evolve(oaa_circuit(psi, 0, False)).probabilities()
    var good_before = before[0] + before[1] + before[2] + before[3]
    print("block-00 probability before one step:", good_before, " exact:", good_before_exact)
    _ = log.close("block-00 probability before one step equals ||A psi||^2 / 4", good_before, good_before_exact, 1e-12)
    var nb_before = List[Int]()
    nb_before.append(295)
    nb_before.append(48)
    nb_before.append(131)
    nb_before.append(0)
    _ = log.within(
        "notebook's printed block-00 frequencies before one step, max |z| (2048 shots)",
        max_z(nb_before, shots, law_before), 0.0, 4.0,
    )

    # ---- after one step: the good block is reached with certainty ----
    var after = evolve(oaa_circuit(psi, 1, False)).probabilities()
    var good_after = after[0] + after[1] + after[2] + after[3]
    var theta = asin(sqrt(good_before_exact))
    var expect_after = sin(3.0 * theta) * sin(3.0 * theta)
    print("block-00 probability after one step:", good_after, " exact:", expect_after)
    _ = log.close("block-00 probability after one step equals sin^2(3 theta) = 1", good_after, expect_after, 1e-12)
    var dev_law = 0.0
    for d in range(4):
        dev_law = max(dev_law, abs(after[d] - law_after[d]))
    _ = log.close("data law after one step equals |A psi|^2 (max deviation)", dev_law, 0.0, 1e-12)
    var nb_after = List[Int]()
    nb_after.append(1311)
    nb_after.append(252)
    nb_after.append(485)
    nb_after.append(0)
    _ = log.within(
        "notebook's printed data frequencies after one step, max |z| (2048 shots)",
        max_z(nb_after, shots, law_after), 0.0, 4.0,
    )

    # ---- sampled shots of the amplified circuit ----
    var res = sample(oaa_circuit(psi, 1, True), shots, UInt64(1234))
    var block0 = 0
    var cnt = List[Int](length=4, fill=0)
    for y in res.memory:
        if ((y >> 2) & 3) == 0:
            block0 += 1
            cnt[y & 3] += 1
    _ = log.check("sampled shots after one step all read block 00", block0 == shots, "block00=" + String(block0) + " of " + String(shots))
    _ = log.within("sampled data law after one step, max |z| against |A psi|^2", max_z(cnt, shots, law_after), 0.0, 4.0)

    log.finish()
