"""Deutsch-Jozsa and Bernstein-Vazirani, exact, sampled and compiled (Braket algorithm library).

Reproduces: Amazon Braket algorithm library - Deutsch-Jozsa Algorithm and Bernstein-Vazirani Algorithm
Source: https://github.com/amazon-braket/amazon-braket-algorithm-library/tree/main/notebooks/textbook (Deutsch_Jozsa_Algorithm.ipynb, Bernstein_Vazirani_Algorithm.ipynb; Apache-2.0)

The Braket notebooks run Deutsch-Jozsa on n = 3 query qubits (constant oracle:
P("000") = 1.0 on the local simulator, 10,000 shots) and Bernstein-Vazirani
with the hidden string "1010" (1,000 shots; character i = qubit i, i.e. the
QUMO integer 5). QUMO's library builders `deutsch_jozsa(n, mask, c)`
(f(x) = mask.x XOR c, ancilla in |->) and `bernstein_vazirani(n, secret)`
(phase-oracle form, no ancilla) are swept over n = 3..12, run exactly and
with 1,000 shots, and compiled for two vendor-neutral modality targets:
- `grid:3x4/sc_cz`: a 3x4 superconducting-style lattice, basis {rz, sx, x, cz};
- `trapped_ion:12`: all-to-all, Molmer-Sorensen basis.

Checks:
- constant oracles give |0...0> with probability 1, balanced oracles with
  probability 0 (1e-12; page n = 3: P("000") = 1.0);
- BV returns the secret with probability 1 (1e-12; page secret "1010"), and
  every one of the 1,000 shots equals it;
- every compiled circuit conforms to its target and is equivalent to the
  original through the initial/final layouts (overlap 1 within 1e-10, exact
  global phase) on |0...0> and on a random product input.

Run: .venv/bin/mojo run -I . examples/frameworks/braket/deutsch_jozsa_bernstein_vazirani.mojo
example-meta: {"framework": "braket", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from qumo.ir.circuit import Circuit
from qumo.sim.runner import evolve, sample
from qumo.sim.rng import Rng
from qumo.compiler.target import parse_target, Target
from qumo.compiler.pass_manager import transpile, conformance_error, CompileResult
from qumo.algorithms.library import deutsch_jozsa, bernstein_vazirani, bitstring_to_int
from qumo.testing.checks import CheckLog

comptime SHOTS = 1000


def braket_secret(n: Int) -> Int:
    """"1010..." in Braket notation (character i = qubit i): ones on even qubits."""
    var s = 0
    for i in range(0, n, 2):
        s |= 1 << i
    return s


def input_marginal(c: Circuit, n: Int) raises -> List[Float64]:
    var probs = evolve(c).probabilities()
    var out = List[Float64](length=1 << n, fill=0.0)
    for i in range(len(probs)):
        out[i & ((1 << n) - 1)] += probs[i]
    return out^


def routed_overlap(orig: Circuit, res: CompileResult, prep: Circuit) raises -> Tuple[Float64, Float64]:
    """<orig psi | compiled psi> with logical qubit l read at physical final_layout[l]."""
    var n = orig.num_qubits
    var pa = prep.copy()
    pa.compose(orig)
    var sa = evolve(pa)
    var qmap = List[Int]()
    for l in range(n):
        qmap.append(res.initial_layout[l])
    var pb = Circuit(res.circuit.num_qubits)
    pb.compose_mapped(prep, qmap)
    pb.compose(res.circuit)
    var sb = evolve(pb)
    var re = 0.0
    var im = 0.0
    for i in range(1 << n):
        var j = 0
        for l in range(n):
            if ((i >> l) & 1) == 1:
                j |= 1 << res.final_layout[l]
        re += sa.re[i] * sb.re[j] + sa.im[i] * sb.im[j]
        im += sa.re[i] * sb.im[j] - sa.im[i] * sb.re[j]
    return (re, im)


def random_product(n: Int, seed: UInt64) -> Circuit:
    var rng = Rng(seed)
    var c = Circuit(n)
    for q in range(n):
        c.ry(rng.uniform() * 3.0, q)
        c.rz(rng.uniform() * 6.0, q)
    return c^


def main() raises:
    var log = CheckLog("braket/deutsch_jozsa_bernstein_vazirani")
    var targets: List[String] = ["grid:3x4/sc_cz", "trapped_ion:12"]
    var tgs = List[Target]()
    for t in targets:
        tgs.append(parse_target(t))
    var dj_err = 0.0
    var bv_err = 0.0
    var shots_ok = True
    var equiv_err = 0.0
    var conform_ok = True
    var compiled = 0
    print("   n   DJ const  DJ bal   BV secret  P(secret)   compiled 2q gates (grid / ion)")
    for n in range(3, 13):
        # Deutsch-Jozsa: two constant and one balanced oracle
        var bal_mask = (1 << n) - 1
        var p_c0 = input_marginal(deutsch_jozsa(n, 0, 0), n)[0]
        var p_c1 = input_marginal(deutsch_jozsa(n, 0, 1), n)[0]
        var p_b = input_marginal(deutsch_jozsa(n, bal_mask, 1), n)[0]
        dj_err = max(dj_err, max(abs(p_c0 - 1.0), max(abs(p_c1 - 1.0), abs(p_b))))
        # Bernstein-Vazirani
        var s = braket_secret(n)
        var pb = input_marginal(bernstein_vazirani(n, s), n)[s]
        bv_err = max(bv_err, abs(pb - 1.0))
        var r = sample(bernstein_vazirani(n, s, measure=True), SHOTS, seed=UInt64(n))
        if bitstring_to_int(r.most_frequent()) != s or len(r.counts) != 1:
            shots_ok = False
        var rd = sample(deutsch_jozsa(n, bal_mask, 0, measure=True), SHOTS, seed=UInt64(50 + n))
        if r.counts.get(String("0") * n, 0) != 0 or rd.counts.get(String("0") * n, 0) != 0:
            shots_ok = False
        # Compile both algorithms for both modality targets
        var line = String("")
        for ti in range(len(tgs)):
            var circs = List[Circuit]()
            if n + 1 <= 12:
                circs.append(deutsch_jozsa(n, bal_mask, 1))
            circs.append(bernstein_vazirani(n, s))
            var twoq = String("")
            for c in circs:
                var res = transpile(c, tgs[ti], 2)
                compiled += 1
                if conformance_error(res.circuit, tgs[ti]).byte_length() > 0:
                    conform_ok = False
                var o0 = routed_overlap(c, res, Circuit(c.num_qubits))
                var o1 = routed_overlap(c, res, random_product(c.num_qubits, UInt64(7 * n + ti)))
                equiv_err = max(equiv_err, max(abs(o0[0] - 1.0) + abs(o0[1]), abs(o1[0] - 1.0) + abs(o1[1])))
                twoq += String(res.circuit.num_2q_gates()) + " "
            line += twoq + "/ " if ti == 0 else twoq
        print("  ", n, "  ", p_c0, "  ", p_b, "  ", s, "      ", pb, "  ", line)
    _ = log.close("DJ constant -> 1, balanced -> 0 (n=3..12; page n=3)", dj_err, 0.0, 1e-12)
    _ = log.close("BV returns the secret with P=1 (page 1010)", bv_err, 0.0, 1e-12)
    _ = log.check("all sampled runs are deterministic", shots_ok)
    _ = log.check("compiled circuits conform to grid:3x4/sc_cz and trapped_ion:12", conform_ok,
                  String(compiled) + " compilations")
    _ = log.close("compiled circuits are equivalent (overlap with exact phase)", equiv_err, 0.0, 1e-10)
    log.finish()
