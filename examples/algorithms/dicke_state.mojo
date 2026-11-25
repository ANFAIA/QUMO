"""Dicke states: an equal superposition of every n-bit string with k ones, without ancillas.

Reproduces: Bärtschi, Eidenbenz - Deterministic Preparation of Dicke States (the split-and-cyclic-shift circuit of Section 2)
Source: https://arxiv.org/abs/1904.07358
Framework: QUMO - https://github.com/ANFAIA/QUMO

|D_k^n> = C(n,k)^(-1/2) sum over the strings x with wt(x) = k of |x>. `dicke_state(n, k)`
starts from |0^(n-k) 1^k> and applies the split-and-cyclic-shift unitaries
SCS_{n,k}, ..., SCS_{k+1,k}, SCS_{k,k-1}, ..., SCS_{2,1}: one controlled RY per SCS
(gate (i)) and one CNOT-conjugated doubly controlled RY for each further weight
(gates (ii)). No ancillas, O(n min(k, n-k)) gates, O(n) depth. For 2k > n it builds
|D_{n-k}^n> and flips every qubit, as the paper's Theorem 1 suggests.

Motivation: arXiv:2611.27640 (2026) claims O(n) gates and O(log n) depth without
ancillas. That construction is NOT implemented here: it is a theory paper with no gate
constants; for n <= 31, and for the 31-qubit blocks it uses when n >= 32, the paper relies
on generic exact synthesis, so no explicit circuit is given.

Checks (references computed here, not with the library's `dicke_amplitudes`):
- every amplitude equals C(n,k)^(-1/2) on weight-k strings and 0 elsewhere, with
  C(n,k) from Pascal's triangle and the weight counted bit by bit;
- sampled bitstrings all have weight k, and each frequency is within 5 sigma of 1/C(n,k);
- the gate structure is the paper's (Section 3): n-1 gates (i) (one CRY each) and
  (n-r)(r-1) + sum_{i=3}^{r} (i-2) gates (ii) (four RY each here), r = min(k, n-k).
Gate count, CX count and depth are printed, not asserted (the paper bounds the CNOTs
only up to O(n) terms, and it cancels one CNOT per gate (ii) that QUMO keeps).

Run: .venv/bin/mojo run -I . examples/algorithms/dicke_state.mojo
example-meta: {"framework": "qumo", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from std.math import sqrt

from qumo.algorithms.library import dicke_state
from qumo.ir.gates import Op
from qumo.sim.runner import evolve, sample
from qumo.testing.checks import CheckLog

comptime SHOTS = 20000


def binomial(n: Int, k: Int) -> Int:
    """C(n, k) from Pascal's triangle."""
    var row = List[Int](length=n + 1, fill=0)
    row[0] = 1
    for i in range(1, n + 1):
        var j = i
        while j >= 1:
            row[j] = row[j] + row[j - 1]
            j -= 1
    return row[k]


def weight(x: Int, n: Int) -> Int:
    var w = 0
    for q in range(n):
        w += (x >> q) & 1
    return w


def run_case(mut log: CheckLog, n: Int, k: Int) raises:
    var tag = "n=" + String(n) + " k=" + String(k)
    var c = dicke_state(n, k)
    var size = binomial(n, k)
    var a = 1.0 / sqrt(Float64(size))

    # 1. state vector against the analytic amplitudes
    var sv = evolve(c)
    var err = 0.0
    var p_in = 0.0
    for i in range(1 << n):
        var expect = a if weight(i, n) == k else 0.0
        err = max(err, abs(sv.re[i] - expect))
        err = max(err, abs(sv.im[i]))
        if weight(i, n) == k:
            p_in += sv.re[i] * sv.re[i] + sv.im[i] * sv.im[i]
    _ = log.close("amplitudes equal C(n,k)^(-1/2) on weight-k strings (" + tag + ")", err, 0.0, 1e-12)
    _ = log.close("probability on weight k (" + tag + ")", p_in, 1.0, 1e-12)

    # 2. sampled counts: weight k only, each string near 1/C(n,k)
    var cm = c.copy()
    cm.measure_all()
    var r = sample(cm, SHOTS, seed=UInt64(100 * n + k))
    var off_weight = 0
    var worst_z = 0.0
    var p = 1.0 / Float64(size)
    var sigma = sqrt(p * (1.0 - p) / Float64(SHOTS))
    for kv in r.counts.items():
        var ones = 0
        for ch in kv.key.as_bytes():
            if Int(ch) == ord("1"):
                ones += 1
        if ones != k:
            off_weight += kv.value
        else:
            worst_z = max(worst_z, abs(Float64(kv.value) / Float64(SHOTS) - p) / sigma)
    _ = log.check("sampled strings all have weight k (" + tag + ")", off_weight == 0, "off-weight shots=" + String(off_weight))
    _ = log.check("sampled strings seen (" + tag + ")", len(r.counts) == size, "distinct=" + String(len(r.counts)) + " C(n,k)=" + String(size))
    _ = log.within("largest frequency deviation in sigmas (" + tag + ")", worst_z, 0.0, 5.0)

    # 3. gate structure of Section 3 of the paper
    var rr = min(k, n - k)
    var gates_ii = (n - rr) * (rr - 1)
    for i in range(3, rr + 1):
        gates_ii += i - 2
    _ = log.check("n-1 gates (i), one CRY each (" + tag + ")", c.count(Op.CRY) == n - 1, "CRY=" + String(c.count(Op.CRY)))
    _ = log.check("paper count of gates (ii), four RY each (" + tag + ")", c.count(Op.RY) == 4 * gates_ii, "RY=" + String(c.count(Op.RY)) + " gates (ii)=" + String(gates_ii))
    print("  D_" + String(k) + "^" + String(n) + ": C(n,k)=" + String(size) + " gates=" + String(len(c)) + " CX=" + String(c.count(Op.CX)) + " depth=" + String(c.depth()))


def main() raises:
    var log = CheckLog("algorithms/dicke_state")
    run_case(log, 6, 3)
    run_case(log, 8, 2)
    run_case(log, 7, 5)
    log.finish()
