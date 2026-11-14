"""Quantum volume: the Classiq notebook's heavy-output test on QUMO.

Reproduces: Classiq library - Quantum Volume
Framework: Classiq - https://github.com/Classiq/classiq-library
Source: https://github.com/Classiq/classiq-library/blob/main/applications/benchmarking/quantum_volume/quantum_volume.ipynb (Classiq Library, MIT)

The notebook builds the quantum-volume model circuit for n = 3..6 qubits (n layers, each a random pairing of the qubits with a random SU(4) gate per pair, one qubit idle when n is odd), samples 10 circuits per width with 100 shots on the ideal simulator, and computes the heavy-output probability with its own statistic: for each circuit it takes the median of the observed counts of the outcomes that appeared, and sums the counts of outcomes with count >= that median. It prints 0.872, 0.871, 0.906, 0.903 for n = 3..6 in one run and 0.846 (n = 4), 0.835 (n = 5), 0.861 (n = 6) in a second run, and the rule "mean - 2 sigma >= 2/3" gives quantum volume 64 both times.

QUMO runs the same experiment with qumo.bench.circuits.quantum_volume on the state-vector simulator and implements the notebook's statistic exactly (median of observed counts, count >= median). The checks are:
- Notebook statistic vs the notebook's printed values (run 2: n = 4..6; run 1: n = 3, 4 only, see the limits below). One notebook run is one draw of a random experiment, so the tolerance is 3 x the run-to-run standard deviation of that experiment, measured here by repeating the whole 10 x 100 experiment 30 times with QUMO (times sqrt(1 + 1/30) for the error of the QUMO mean). The notebook's two runs differ from each other by up to 0.07, which this spread has to (and does) cover; nothing is tuned.
- Notebook statistic of QUMO vs the same experiment run on an independent reference: the notebook's own model (Haar SU(4) from Gram-Schmidt of complex Gaussians, QR-equivalent) applied on a plain state vector inside this file, using QUMO only for its random numbers. Same for the exact heavy-output probability (200 circuits each).
- The standard definition (heavy = exact probability above the median of all 2^n probabilities) is checked on its own: the sampled heavy fraction against the exact value of the same circuits, with the binomial error computed from each circuit's exact heavy probability.
- Median: the brute-force median is checked on hand-computed lists, and for every circuit exactly half of the 2^n probabilities lie above it and half below (a counting check that does not use the sort).
- Normalisation of every state, and circuit structure: n layers x floor(n/2) SU(4) blocks (3 CX each), disjoint pairs within a layer, one idle qubit per layer for odd n.
- The notebook's 2/3 rule and the resulting quantum volume 64 (log2 = 6 = max_qubits).

Deviations from the notebook:
- The SU(4) blocks come from QUMO's KAK-form generator (random single-qubit unitaries around exp(i(x XX + y YY + z ZZ)) with x, y, z uniform on [0, pi/4], which is not the Haar measure on the KAK coordinates), not a Haar matrix from scipy/QR; the random pairings and matrices are QUMO's seeded stream, so individual numbers differ from the notebook. Haar-ness is only judged through statistics.
- Limit, generator: QUMO's exact heavy-output probability is 0.008 to 0.014 above the Haar SU(4) reference at every width (200 circuits each; 1.6 to 3.9 combined standard errors, so a small real offset, in the direction of the uniform KAK sampler being not Haar). The 5-sigma tolerance of that check is wide enough to contain it, so the check only catches gross errors, not a 0.01 bias. A Haar-weighted KAK sampler tried while writing this example gave a similar offset. The finite-size Haar-state value (1/d) sum_k sum_j 1/j (0.817, 0.831, 0.839, 0.843 for n = 3..6) is printed for information only: depth-n QV circuits are not Haar states, and QUMO and the Haar SU(4) model both sit above it.
- Limit, notebook run 1: its n = 5 and n = 6 values (0.906, 0.903) are not reproduced. QUMO gives 0.865 and 0.853, the independent Haar SU(4) model 0.856 and 0.851 (run-to-run sigma about 0.015), and the notebook's own run 2 prints 0.835 and 0.861; run 1 is 2.7 to 4.3 run-to-run sigma above QUMO and 3.2 to 3.7 above the Haar model. Those two numbers are printed with their distance but are not checked. The notebook's two runs disagree with each other by up to 0.07 at n = 5, so one printed run is not a precise reference; QUMO agrees with a faithful reimplementation of the model and with run 2.
- The 2/3 rule is satisfied by a wide margin (about 0.83 or more against 0.667), so quantum volume 64 does not test circuit correctness; the structure and statistics checks above do.
- The notebook's sigma follows its formula (std of per-circuit heavy counts divided by trials * shots), which is much smaller than the standard error of the mean; a second check uses the proper standard error.
- No hardware or noisy backends (Rigetti Aspen, IBM): the notebook runs them commented out.

Run: .venv/bin/mojo run -I . examples/frameworks/classiq/quantum_volume.mojo
example-meta: {"framework": "classiq", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from std.math import log, sqrt
from qumo.sim.runner import evolve, sample
from qumo.bench.circuits import quantum_volume
from qumo.ir.circuit import Circuit
from qumo.ir.gates import Op, op_arity
from qumo.algorithms.library import bitstring_to_int
from qumo.sim.rng import Rng
from qumo.testing.checks import CheckLog


def haar4(mut rng: Rng, mut ur: List[Float64], mut ui: List[Float64]):
    """Haar-random 4x4 unitary: Gram-Schmidt on complex Gaussian columns (row-major ur/ui)."""
    var cr = List[Float64](length=16, fill=0.0)  # column j stored at [j*4 + i]
    var ci = List[Float64](length=16, fill=0.0)
    for j in range(4):
        for i in range(4):
            cr[j * 4 + i] = rng.normal()
            ci[j * 4 + i] = rng.normal()
        for k in range(j):
            var pr = 0.0  # <col_k, col_j>
            var pi_ = 0.0
            for i in range(4):
                pr += cr[k * 4 + i] * cr[j * 4 + i] + ci[k * 4 + i] * ci[j * 4 + i]
                pi_ += cr[k * 4 + i] * ci[j * 4 + i] - ci[k * 4 + i] * cr[j * 4 + i]
            for i in range(4):
                cr[j * 4 + i] -= pr * cr[k * 4 + i] - pi_ * ci[k * 4 + i]
                ci[j * 4 + i] -= pr * ci[k * 4 + i] + pi_ * cr[k * 4 + i]
        var nrm = 0.0
        for i in range(4):
            nrm += cr[j * 4 + i] * cr[j * 4 + i] + ci[j * 4 + i] * ci[j * 4 + i]
        nrm = sqrt(nrm)
        for i in range(4):
            cr[j * 4 + i] /= nrm
            ci[j * 4 + i] /= nrm
    for i in range(4):
        for j in range(4):
            ur[i * 4 + j] = cr[j * 4 + i]
            ui[i * 4 + j] = ci[j * 4 + i]


def haar_qv_probs(n: Int, mut rng: Rng) -> List[Float64]:
    """Notebook's model by direct state-vector arithmetic: n layers, shuffled
    pairing, an independent Haar SU(4) on each pair; returns |amplitude|^2."""
    var d = 1 << n
    var re = List[Float64](length=d, fill=0.0)
    var im = List[Float64](length=d, fill=0.0)
    re[0] = 1.0
    var perm = List[Int](length=n, fill=0)
    var ur = List[Float64](length=16, fill=0.0)
    var ui = List[Float64](length=16, fill=0.0)
    for _ in range(n):
        for i in range(n):
            perm[i] = i
        var k = n - 1
        while k > 0:
            var j = rng.below(k + 1)
            var t = perm[k]
            perm[k] = perm[j]
            perm[j] = t
            k -= 1
        for idx in range(n // 2):
            var a = perm[idx]
            var b = perm[n // 2 + idx]
            haar4(rng, ur, ui)
            for base in range(d):
                if (base >> a) & 1 == 1 or (base >> b) & 1 == 1:
                    continue
                var ix = List[Int](length=4, fill=0)
                for s in range(4):
                    ix[s] = base | (((s >> 1) & 1) << a) | ((s & 1) << b)
                var vr = List[Float64](length=4, fill=0.0)
                var vi = List[Float64](length=4, fill=0.0)
                for r in range(4):
                    for c in range(4):
                        vr[r] += ur[r * 4 + c] * re[ix[c]] - ui[r * 4 + c] * im[ix[c]]
                        vi[r] += ur[r * 4 + c] * im[ix[c]] + ui[r * 4 + c] * re[ix[c]]
                for r in range(4):
                    re[ix[r]] = vr[r]
                    im[ix[r]] = vi[r]
    var p = List[Float64](length=d, fill=0.0)
    for i in range(d):
        p[i] = re[i] * re[i] + im[i] * im[i]
    return p^


def median_of(xs: List[Float64]) -> Float64:
    """Brute-force median: selection by insertion sort on a copy."""
    var s = xs.copy()
    for i in range(1, len(s)):
        var v = s[i]
        var j = i - 1
        while j >= 0 and s[j] > v:
            s[j + 1] = s[j]
            j -= 1
        s[j + 1] = v
    var m = len(s)
    if m % 2 == 1:
        return s[m // 2]
    return 0.5 * (s[m // 2 - 1] + s[m // 2])


def is_heavy(probs: List[Float64], med: Float64) -> List[Bool]:
    var out = List[Bool](length=len(probs), fill=False)
    for i in range(len(probs)):
        out[i] = probs[i] > med
    return out^


def exact_hop(probs: List[Float64], heavy: List[Bool]) -> Float64:
    var h = 0.0
    for i in range(len(probs)):
        if heavy[i]:
            h += probs[i]
    return h


def haar_state_hop(d: Int) -> Float64:
    """Expected heavy-output probability of a Haar-random state in dimension d.

    The probabilities are Dirichlet(1,...,1); the k-th smallest has mean
    (1/d) * sum_{j=d-k+1}^{d} 1/j. Heavy = the d/2 largest.
    """
    var tot = 0.0
    for k in range(d // 2 + 1, d + 1):
        var h = 0.0
        for j in range(d - k + 1, d + 1):
            h += 1.0 / Float64(j)
        tot += h / Float64(d)
    return tot


def notebook_hop_count(counts: List[Float64]) -> Float64:
    """The notebook's heavy_outputs_prob numerator: median of the observed
    counts, then the sum of the counts that are >= the median."""
    var med = median_of(counts)
    var tot = 0.0
    for c in counts:
        if c >= med:
            tot += c
    return tot


def haar_notebook_count(probs: List[Float64], shots: Int, mut rng: Rng) -> Float64:
    """Draw `shots` outcomes from probs (inverse CDF) and apply the notebook statistic."""
    var counts = List[Int](length=len(probs), fill=0)
    for _ in range(shots):
        var u = rng.uniform()
        var acc = 0.0
        var pick = len(probs) - 1
        for i in range(len(probs)):
            acc += probs[i]
            if u < acc:
                pick = i
                break
        counts[pick] += 1
    var seen = List[Float64]()
    for c in counts:
        if c > 0:
            seen.append(Float64(c))
    return notebook_hop_count(seen)


def seed_of(n: Int, t: Int, salt: Int) -> UInt64:
    return UInt64(7919 * n + 104729 * t + 15485863 * salt + 17)


def structure_ok(c: Circuit, n: Int) -> Bool:
    """n layers x floor(n/2) SU(4) blocks (3 CX each on one pair), disjoint pairs
    inside a layer (so one qubit idles per layer when n is odd), nothing else
    entangling."""
    var cx = List[Int]()
    var two_q = 0
    for i in range(len(c.ops)):
        if op_arity(c.ops[i].kind) == 2:
            two_q += 1
        if c.ops[i].kind == Op.CX:
            cx.append(i)
    var blocks_per_layer = n // 2
    if two_q != len(cx) or len(cx) != 3 * n * blocks_per_layer:
        return False
    for layer in range(n):
        var used = List[Bool](length=n, fill=False)
        for b in range(blocks_per_layer):
            var base = 3 * (layer * blocks_per_layer + b)
            var i0 = c.ops[cx[base]]
            var i1 = c.ops[cx[base + 1]]
            var i2 = c.ops[cx[base + 2]]
            # cx(b,a), cx(a,b), cx(b,a): one pair, alternating direction
            if i0.q0 != i2.q0 or i0.q1 != i2.q1 or i1.q0 != i0.q1 or i1.q1 != i0.q0:
                return False
            if used[i0.q0] or used[i0.q1] or i0.q0 == i0.q1:
                return False
            used[i0.q0] = True
            used[i0.q1] = True
    return True


def run_rep(n: Int, rep: Int, num_trials: Int, num_shots: Int,
            mut nb_hits: List[Float64], mut def_hits: List[Float64], mut exact_h: List[Float64]) raises:
    """One notebook experiment: num_trials circuits x num_shots shots.

    nb_hits: per-circuit numerator of the notebook statistic; def_hits: shots
    landing on an exact-heavy state; exact_h: exact heavy-output probability.
    """
    for t in range(num_trials):
        var c = quantum_volume(n, n, seed_of(n, t, 1000 + rep))
        var probs = evolve(c).probabilities()
        var heavy = is_heavy(probs, median_of(probs))
        exact_h[t] = exact_hop(probs, heavy)
        c.measure_all()
        var res = sample(c, num_shots, seed_of(n, t, 5000 + rep))
        var counts = List[Float64]()
        var hits = 0
        for kv in res.counts.items():
            counts.append(Float64(kv.value))
            if heavy[bitstring_to_int(kv.key)]:
                hits += kv.value
        nb_hits[t] = notebook_hop_count(counts)
        def_hits[t] = Float64(hits)


def main() raises:
    var log_ = CheckLog("classiq/quantum_volume")
    var num_trials = 10
    var num_shots = 100
    var min_qubits = 3
    var max_qubits = 6
    var reps = 30
    var n_exact = 200
    var width = max_qubits - min_qubits + 1
    var norm_shots = Float64(num_trials * num_shots)
    # notebook-printed values (None = not printed), n = 3..6
    var run1: List[Float64] = [0.872, 0.871, 0.906, 0.903]
    var run2: List[Float64] = [-1.0, 0.846, 0.835, 0.861]
    var sampled_mean = List[Float64](length=width, fill=0.0)
    var sampled_std = List[Float64](length=width, fill=0.0)

    # ---- the median helper on hand-computed lists ----
    var odd: List[Float64] = [5.0, 1.0, 3.0]
    var even: List[Float64] = [4.0, 1.0, 3.0, 2.0]
    _ = log_.close("median of [5,1,3] is 3", median_of(odd), 3.0, 0.0)
    _ = log_.close("median of [4,1,3,2] is 2.5", median_of(even), 2.5, 0.0)
    var tie: List[Float64] = [2.0, 2.0, 1.0, 5.0, 2.0]
    _ = log_.close("notebook statistic on counts [2,2,1,5,2]: median 2, sum of counts >= 2 is 11", notebook_hop_count(tie), 11.0, 0.0)

    for n in range(min_qubits, max_qubits + 1):
        var k = n - min_qubits
        var d = 1 << n
        # ---- exact heavy-output probability, 200 circuits ----
        var tot = 0.0
        var tot2 = 0.0
        var worst_norm = 0.0
        var half_ok = True
        var struct_all = True
        for t in range(n_exact):
            var c = quantum_volume(n, n, seed_of(n, t, 1))
            if not structure_ok(c, n):
                struct_all = False
            var probs = evolve(c).probabilities()
            var s = 0.0
            for p in probs:
                s += p
            worst_norm = max(worst_norm, abs(s - 1.0))
            var med = median_of(probs)
            var above = 0
            var below = 0
            for p in probs:
                if p > med:
                    above += 1
                elif p < med:
                    below += 1
            if above != d // 2 or below != d // 2:
                half_ok = False
            var h = exact_hop(probs, is_heavy(probs, med))
            tot += h
            tot2 += h * h
        var exact_mean = tot / Float64(n_exact)
        var sem = sqrt(max(tot2 / Float64(n_exact) - exact_mean * exact_mean, 0.0) / Float64(n_exact))
        var haar_state = haar_state_hop(d)
        # independent reference: the notebook's own model (Haar SU(4) by Gram-Schmidt) on a plain state vector
        var hrng = Rng(UInt64(9001 + n))
        var ht = 0.0
        var ht2 = 0.0
        for _ in range(n_exact):
            var hp = haar_qv_probs(n, hrng)
            var hh = exact_hop(hp, is_heavy(hp, median_of(hp)))
            ht += hh
            ht2 += hh * hh
        var haar_mean = ht / Float64(n_exact)
        var haar_sem = sqrt(max(ht2 / Float64(n_exact) - haar_mean * haar_mean, 0.0) / Float64(n_exact))
        print("n =", n, " exact HOP over", n_exact, "circuits: QUMO", exact_mean, "+-", sem, " Haar-SU(4) reference", haar_mean, "+-", haar_sem,
              " (Haar-state value", haar_state, ")")
        _ = log_.check("n=" + String(n) + " circuit structure: " + String(n) + " layers x " + String(n // 2) + " SU(4) blocks, disjoint pairs", struct_all)
        _ = log_.close("n=" + String(n) + " probabilities normalised", worst_norm, 0.0, 1e-9)
        _ = log_.check("n=" + String(n) + " median splits the " + String(d) + " probabilities in half (counted, no ties)", half_ok)
        _ = log_.close("n=" + String(n) + " exact HOP vs Haar-SU(4) reference model (5 sigma of the difference)", exact_mean, haar_mean,
                       5.0 * sqrt(sem * sem + haar_sem * haar_sem))

        # ---- the notebook experiment, repeated to measure its run-to-run spread ----
        var rep_mean = List[Float64](length=reps, fill=0.0)
        var first_def = 0.0
        var first_exact = 0.0
        var first_var = 0.0
        for r in range(reps):
            var nb = List[Float64](length=num_trials, fill=0.0)
            var dh = List[Float64](length=num_trials, fill=0.0)
            var eh = List[Float64](length=num_trials, fill=0.0)
            run_rep(n, r, num_trials, num_shots, nb, dh, eh)
            var sum_nb = 0.0
            for t in range(num_trials):
                sum_nb += nb[t]
            rep_mean[r] = sum_nb / norm_shots
            if r == 0:
                # the single "notebook run" used for the decision rule
                var mean_c = sum_nb / Float64(num_trials)
                var var_c = 0.0
                for t in range(num_trials):
                    var_c += (nb[t] - mean_c) * (nb[t] - mean_c)
                sampled_mean[k] = rep_mean[0]
                sampled_std[k] = sqrt(var_c / Float64(num_trials)) / norm_shots
                var sd = 0.0
                var se2 = 0.0
                var ex = 0.0
                for t in range(num_trials):
                    sd += dh[t]
                    ex += eh[t]
                    se2 += 100.0 * eh[t] * (1.0 - eh[t])  # Var of Binomial(100, h_t)
                first_def = sd / norm_shots
                first_exact = ex / Float64(num_trials)
                first_var = sqrt(se2) / norm_shots
        var m = 0.0
        for r in range(reps):
            m += rep_mean[r]
        m /= Float64(reps)
        var v = 0.0
        for r in range(reps):
            v += (rep_mean[r] - m) * (rep_mean[r] - m)
        var sigma_rep = sqrt(v / Float64(reps - 1))
        var tol = 3.0 * sigma_rep * sqrt(1.0 + 1.0 / Float64(reps))
        print("n =", n, " notebook statistic: mean over", reps, "experiments", m, " run-to-run sigma", sigma_rep,
              " first experiment", sampled_mean[k], " notebook-style sigma", sampled_std[k])
        # the same experiment on the Haar SU(4) reference model
        var hm = 0.0
        var hv = 0.0
        var h_rep = List[Float64](length=reps, fill=0.0)
        for r in range(reps):
            var sum_h = 0.0
            for _ in range(num_trials):
                sum_h += haar_notebook_count(haar_qv_probs(n, hrng), num_shots, hrng)
            h_rep[r] = sum_h / norm_shots
            hm += h_rep[r]
        hm /= Float64(reps)
        for r in range(reps):
            hv += (h_rep[r] - hm) * (h_rep[r] - hm)
        var h_sigma = sqrt(hv / Float64(reps - 1))
        print("n =", n, " notebook statistic on the Haar SU(4) reference model:", hm, " run-to-run sigma", h_sigma)
        _ = log_.close("n=" + String(n) + " notebook statistic: QUMO vs Haar SU(4) reference model (4 sigma of the difference of means)", m, hm,
                       4.0 * sqrt((sigma_rep * sigma_rep + h_sigma * h_sigma) / Float64(reps)))
        if n <= 4:
            _ = log_.close("n=" + String(n) + " notebook statistic vs notebook run 1 (3 run-to-run sigma)", m, run1[k], tol)
        else:
            # not checked: see the docstring (the independent Haar model agrees with QUMO, not with these two printed numbers)
            print("n =", n, " notebook run 1 printed", run1[k], "is", (run1[k] - m) / sigma_rep, "run-to-run sigma above QUMO and",
                  (run1[k] - hm) / h_sigma, "above the Haar reference (not reproduced, not checked)")
        if run2[k] > 0.0:
            _ = log_.close("n=" + String(n) + " notebook statistic vs notebook run 2 (3 run-to-run sigma)", m, run2[k], tol)
        # same circuits, standard definition: only shot noise (exact per-circuit binomial variance)
        _ = log_.close("n=" + String(n) + " sampled heavy fraction vs exact value of the same circuits (4 binomial sigma)", first_def, first_exact, 4.0 * first_var)

    # ---- the notebook's decision rule ----
    var qubit_v = 0
    for n in range(min_qubits, max_qubits + 1):
        var k = n - min_qubits
        if sampled_mean[k] - 2.0 * sampled_std[k] >= 2.0 / 3.0:
            qubit_v = n
        else:
            break
    var qv = 1 << qubit_v
    print("quantum volume:", qv)
    _ = log_.check("every width passes the 2/3 threshold, quantum volume is 64 as in the notebook", qv == 64, "qv=" + String(qv))
    var proper_ok = True
    for n in range(min_qubits, max_qubits + 1):
        var k = n - min_qubits
        var se = sqrt(sampled_mean[k] * (1.0 - sampled_mean[k]) / norm_shots)
        if sampled_mean[k] - 2.0 * se < 2.0 / 3.0:
            proper_ok = False
    _ = log_.check("2/3 threshold also holds with the proper 2 x standard error", proper_ok)
    log_.finish()
