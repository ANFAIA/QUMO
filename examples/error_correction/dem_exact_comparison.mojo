"""Exact comparison of detector error models: QUMO's extractor against a DEM worked out by hand.

Reproduces: Luo, Suzuki - DEMVal, exact comparison oracle for detector error models (Sec. 4.2, Alg. 1)
Source: https://arxiv.org/abs/2611.20501
Framework: QUMO - https://github.com/ANFAIA/QUMO

A detector error model (DEM) is a table of independent mechanisms, each a set of
flipped detectors and observables (its symptom class) with a probability. Two
models are the same when, after merging mechanisms of equal class with
x + y - 2xy, no class with p > 1e-15 is on one side only (u = 0) and every shared
class agrees to a relative tolerance (delta = max |pa - pb| / max(pa, pb)).
No sampling is involved, so a small rate error is seen at once.

The circuit: a distance-3 repetition code, one round of two Z-parity checks, then
the data read out. Its DEM is written below from the channel formulas alone, with
no QUMO or Stim code:

    X_ERROR(px) on data 0, 2, 4 before the round: D0 | D0 D1 | D1 L0, rate px
    X_ERROR(pm) on the checks before MR (a readout error): D0 D2 | D1 D3, rate pm
    DEPOLARIZE1(pd) on the data before M: X and Y flip the readout, Z does not.
        Stim's conversion gives each Pauli q = (1 - sqrt(1 - 4pd/3))/2, and the
        two merge to 2q(1 - q) = 2pd/3: D2 | D2 D3 | D3 L0

Then two of the paper's defect classes are seeded: depolarizing taken as p/3 per
Pauli (qdk-ec issue 68), which leaves the classes but moves delta to exactly pd/3,
and the observable lost from two classes (cudaqx issue 607), which gives u = 4.

Checks:
- the hand-written DEM has 8 classes;
- QUMO's detector_error_model agrees with it, u = 0 and delta <= 1e-12, with and without decomposition;
- the p/3 mutation is inconsistent with u = 0 and delta = pd/3;
- the lost observable is inconsistent with u = 4.

Run: .venv/bin/mojo run -I . examples/error_correction/dem_exact_comparison.mojo
example-meta: {"framework": "qumo", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from qumo.qec.dem import DetectorErrorModel, DemError, detector_error_model
from qumo.qec.dem_compare import DemComparison, compare_dem, gather_dem
from qumo.qec.stim_text import from_stim
from qumo.testing.checks import CheckLog


def hand_dem(px: Float64, pm: Float64, flip: Float64) -> DetectorErrorModel:
    """The repetition-code DEM from the channel formulas; `flip` is the X-or-Y rate of DEPOLARIZE1."""
    var m = DetectorErrorModel(4, 1)
    m.errors.append(DemError(px, [0], 0))
    m.errors.append(DemError(px, [0, 1], 0))
    m.errors.append(DemError(pm, [0, 2], 0))
    m.errors.append(DemError(px, [1], 1))
    m.errors.append(DemError(pm, [1, 3], 0))
    m.errors.append(DemError(flip, [2], 0))
    m.errors.append(DemError(flip, [2, 3], 0))
    m.errors.append(DemError(flip, [3], 1))
    return m^


def _summary(r: DemComparison) -> String:
    """One-line detail for a CHECK line (String(r) adds a line per mismatch)."""
    return "u=" + String(r.unmatched) + " delta=" + String(r.max_rel) + " shared=" + String(r.shared)


def main() raises:
    var log = CheckLog("dem_exact_comparison")
    var px = 0.01
    var pm = 0.005
    var pd = 0.03
    var text = String("""R 0 1 2 3 4
X_ERROR(0.01) 0 2 4
CX 0 1 2 3
CX 2 1 4 3
X_ERROR(0.005) 1 3
MR 1 3
DETECTOR rec[-2]
DETECTOR rec[-1]
DEPOLARIZE1(0.03) 0 2 4
M 0 2 4
DETECTOR rec[-3] rec[-2] rec[-5]
DETECTOR rec[-2] rec[-1] rec[-4]
OBSERVABLE_INCLUDE(0) rec[-1]
""")
    var circuit = from_stim(text).circuit.copy()
    var hand = hand_dem(px, pm, 2.0 * pd / 3.0)
    print("hand-derived DEM:")
    print(hand.to_text())
    _ = log.close("hand-derived DEM classes", Float64(gather_dem(hand).num_errors()), 8.0, 0.0)

    var qumo = detector_error_model(circuit, decompose=False)
    var r = compare_dem(qumo, hand, tol=1e-12)
    print(r)
    _ = log.check("QUMO extractor matches the hand-derived DEM", r.consistent, _summary(r))
    _ = log.close("one-sided classes u", Float64(r.unmatched), 0.0, 0.0)
    _ = log.within("largest relative difference delta", r.max_rel, 0.0, 1e-12)
    _ = log.check("same detector and observable counts", r.same_shape)
    var rd = compare_dem(detector_error_model(circuit), hand, tol=1e-12)
    _ = log.check("decomposed DEM compares equal too", rd.consistent, _summary(rd))

    # qdk-ec 68: each Pauli of DEPOLARIZE1 at p/3, merged: 2(p/3)(1 - p/3)
    var q3 = pd / 3.0
    var mutant = hand_dem(px, pm, 2.0 * q3 * (1.0 - q3))
    var rm = compare_dem(qumo, mutant)
    print(rm)
    _ = log.check("p/3 depolarizing is reported inconsistent", not rm.consistent)
    _ = log.close("p/3 mutation keeps the classes u", Float64(rm.unmatched), 0.0, 0.0)
    _ = log.close("p/3 mutation delta equals pd/3", rm.max_rel, pd / 3.0, 1e-12)

    # cudaqx 607: the observable lost from D1 L0 and D3 L0
    var lost = hand.copy()
    for ref e in lost.errors:
        e.obs = 0
    var rl = compare_dem(qumo, lost)
    print(rl)
    _ = log.check("lost observable is reported inconsistent", not rl.consistent)
    _ = log.close("lost observable one-sided classes u", Float64(rl.unmatched), 4.0, 0.0)
    log.finish()
