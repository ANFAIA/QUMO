"""Measurement reduction: naive vs qubit-wise-commuting vs commuting sets.

Reproduces: TKET - Advanced expectation values and measurement reduction
Source: https://docs.quantinuum.com/tket/user-guide/examples/algorithms_and_protocols/measurement_reduction_example.html (pytket user guide, Apache-2.0)

The page measures the 3-term operator
    O = 0.1 Z0 + 0.4 Y0 Z1 X2 X3 Y4 + 0.2 X0 X1
on the 5-qubit state V(q2) H(q4)|00000> (V = Rx(pi/2), up to phase). Exact
value: 0.1 (page). It compares measurement strategies:
- naive: one circuit per term (3);
- NonConflictingSets (qubit-wise commuting, only 1-qubit basis changes): 3;
- CommutingSets (general commutation, Clifford diagonalisation): 2, the second
  needing 1 CX (page).
QUMO's `PauliSum.group_qwc` is NonConflictingSets and `EstimatorPub` measures
each QWC group with one basis setting. CommutingSets is not in QUMO, so this
example adds a small version of it in-file: greedy general-commutation
grouping, then a search for CX gates (tracked in the symplectic picture with
Aaronson-Gottesman signs) that turn each group qubit-wise commuting.

Checks:
- the exact value is 0.1 (page, 1e-12);
- 3 naive circuits, 3 QWC groups (page NonConflictingSets: 3), 2 commuting
  sets (page CommutingSets: 2) whose second circuit has 1 CX (page);
- the exact value read through the commuting-set circuits is 0.1 (1e-12) and
  equals the direct expectation on random entangled states (1e-12), which
  validates the diagonalisation and its signs (on the page state two of the
  three terms vanish, so it alone would not);
- QWC and commuting-set shot estimates (10,000 shots per circuit, as on the
  page) are within 4 standard errors of 0.1.
Page shot values (AerBackend, 10,000 shots): 0.10184 (no reduction),
0.10088 (NonConflictingSets), 0.10388 (CommutingSets) - quoted, not asserted.

Run: .venv/bin/mojo run -I . examples/frameworks/tket/measurement_reduction.mojo
example-meta: {"framework": "tket", "tier": "fast", "timeout": 60, "requires": [], "args": []}
"""

from std.math import pi, sqrt
from qumo.ir.circuit import Circuit
from qumo.ir.pauli import PauliSum, PauliTerm, parse_pauli_sum
from qumo.sim.runner import evolve, sample
from qumo.exec.primitives import EstimatorPub, run_estimator, pop_parity
from qumo.algorithms.library import hardware_efficient_ansatz
from qumo.sim.rng import Rng
from qumo.ai.checks import CheckLog

comptime N = 5
comptime SHOTS = 10000


@fieldwise_init
struct SignedPauli(ImplicitlyCopyable):
    """Signed Pauli term: coeff * sign * P, P given by symplectic masks (x = z = 1 is Y)."""

    var x: UInt64
    var z: UInt64
    var sign: Float64
    var coeff: Float64


def _bit(m: UInt64, q: Int) -> Bool:
    return ((m >> UInt64(q)) & 1) == 1


def conj_cx(p: SignedPauli, c: Int, t: Int) -> SignedPauli:
    """CX(c, t) P CX(c, t) (Aaronson-Gottesman update, sign included)."""
    var xc = _bit(p.x, c)
    var zc = _bit(p.z, c)
    var xt = _bit(p.x, t)
    var zt = _bit(p.z, t)
    var out = p
    if xc and zt and (xt == zc):
        out.sign = -p.sign
    if xc:
        out.x = p.x ^ (UInt64(1) << UInt64(t))
    if zt:
        out.z = p.z ^ (UInt64(1) << UInt64(c))
    return out


def qwc(a: SignedPauli, b: SignedPauli) -> Bool:
    var supp = (a.x | a.z) & (b.x | b.z)
    return ((a.x ^ b.x) & supp) == 0 and ((a.z ^ b.z) & supp) == 0


def all_qwc(ps: List[SignedPauli]) -> Bool:
    for i in range(len(ps)):
        for j in range(i + 1, len(ps)):
            if not qwc(ps[i], ps[j]):
                return False
    return True


def commuting_sets(op: PauliSum) -> List[List[Int]]:
    """Greedy partition into mutually commuting sets (TKET CommutingSets idea)."""
    var groups = List[List[Int]]()
    for i in range(len(op.terms)):
        var placed = False
        for ref g in groups:
            var ok = True
            for j in g:
                if not op.terms[i].commutes_with(op.terms[j]):
                    ok = False
                    break
            if ok:
                g.append(i)
                placed = True
                break
        if not placed:
            var ng = List[Int]()
            ng.append(i)
            groups.append(ng^)
    return groups^


struct MeasurementCircuit(Movable):
    """Clifford C (CX gates + 1-qubit basis changes) that maps every term of a
    commuting set to +-Z on `masks[k]`: <P_k> = signs[k] * <Z_masks[k]> after C."""

    var circuit: Circuit
    var masks: List[Int]
    var signs: List[Float64]
    var coeffs: List[Float64]
    var num_cx: Int

    def __init__(out self, n: Int):
        self.circuit = Circuit(n)
        self.masks = List[Int]()
        self.signs = List[Float64]()
        self.coeffs = List[Float64]()
        self.num_cx = 0


def diagonalise(op: PauliSum, group: List[Int], n: Int) raises -> MeasurementCircuit:
    """Search 0, 1 or 2 CX gates that make the group qubit-wise commuting."""
    var ps = List[SignedPauli]()
    for i in group:
        ps.append(SignedPauli(op.terms[i].x, op.terms[i].z, 1.0, op.terms[i].coeff))
    var pairs = List[Tuple[Int, Int]]()
    for c in range(n):
        for t in range(n):
            if c != t:
                pairs.append((c, t))
    var chosen = List[Tuple[Int, Int]]()
    var cur = ps.copy()
    var found = all_qwc(cur)
    if not found:
        for a in pairs:
            var t1 = List[SignedPauli]()
            for p in cur:
                t1.append(conj_cx(p, a[0], a[1]))
            if all_qwc(t1):
                chosen.append(a)
                cur = t1^
                found = True
                break
    if not found:
        for a in pairs:
            for b in pairs:
                var t2 = List[SignedPauli]()
                for p in cur:
                    t2.append(conj_cx(conj_cx(p, a[0], a[1]), b[0], b[1]))
                if all_qwc(t2):
                    chosen.append(a)
                    chosen.append(b)
                    cur = t2^
                    found = True
                    break
            if found:
                break
    if not found:
        raise Error("no diagonalising circuit with <= 2 CX gates for this set")
    var mc = MeasurementCircuit(n)
    for g in chosen:
        mc.circuit.cx(g[0], g[1])
        mc.num_cx += 1
    # 1-qubit basis change on the union of the (now qubit-wise commuting) terms
    var ux: UInt64 = 0
    var uy: UInt64 = 0
    for p in cur:
        ux |= p.x
        uy |= p.x & p.z
    for q in range(n):
        if _bit(uy, q):
            mc.circuit.sdg(q)
            mc.circuit.h(q)
        elif _bit(ux, q):
            mc.circuit.h(q)
    for p in cur:
        mc.masks.append(Int(p.x | p.z))
        mc.signs.append(p.sign)
        mc.coeffs.append(p.coeff)
    return mc^


def exact_through_sets(prep: Circuit, op: PauliSum, groups: List[List[Int]]) raises -> Float64:
    """<O> read through the diagonalising circuits from exact probabilities."""
    var total = 0.0
    for gi in range(len(groups)):
        var mc = diagonalise(op, groups[gi], prep.num_qubits)
        var full = prep.copy()
        full.compose(mc.circuit)
        var probs = evolve(full).probabilities()
        for k in range(len(mc.masks)):
            var ev = 0.0
            for o in range(len(probs)):
                ev += probs[o] * (-1.0 if pop_parity(o & mc.masks[k]) else 1.0)
            total += mc.coeffs[k] * mc.signs[k] * ev
    return total


def prepare() -> Circuit:
    var c = Circuit(N, name="page_state")
    c.rx(pi / 2, 2)  # TKET V = Rx(1/2 half-turn)
    c.h(4)
    return c^


def main() raises:
    var log = CheckLog("tket/measurement_reduction")
    var op = parse_pauli_sum("0.1*Z0 + 0.4*Y0 Z1 X2 X3 Y4 + 0.2*X0 X1")
    var prep = prepare()
    var exact = evolve(prep).expectation(op)
    print("O =", op)
    print("exact <O> =", exact, "(page: 0.1)")
    _ = log.close("exact value (page 0.1)", exact, 0.1, 1e-12)

    var naive = len(op.terms)
    var g_qwc = op.group_qwc()
    var g_com = commuting_sets(op)
    print("measurement circuits: naive", naive, "| QWC (NonConflictingSets)", len(g_qwc),
          "| commuting sets (CommutingSets)", len(g_com))
    _ = log.check("naive strategy needs 3 circuits", naive == 3, "circuits=" + String(naive))
    _ = log.check("QWC groups = page NonConflictingSets (3)", len(g_qwc) == 3, "groups=" + String(len(g_qwc)))
    _ = log.check("commuting sets = page CommutingSets (2)", len(g_com) == 2, "groups=" + String(len(g_com)))

    # QWC: EstimatorPub measures one basis setting per QWC group.
    var obs = List[PauliSum]()
    obs.append(op.copy())
    var precision = 1.0 / sqrt(Float64(SHOTS) - 0.5)
    var est = run_estimator(EstimatorPub(prep.copy(), obs, List[List[Float64]](), precision, 2024))
    print("QWC estimate:", est.evs[0][0], "+/-", est.stds[0][0], "(", est.shots_per_set, "shots per group )")
    _ = log.close("QWC shot estimate within 4 sigma", est.evs[0][0], 0.1, 4.0 * est.stds[0][0])

    # Commuting sets: diagonalise each set, then sample once per set.
    var exact_cs = 0.0
    var shot_cs = 0.0
    var var_cs = 0.0
    var cx_counts = List[Int]()
    for gi in range(len(g_com)):
        var mc = diagonalise(op, g_com[gi], N)
        cx_counts.append(mc.num_cx)
        var full = prep.copy()
        full.compose(mc.circuit)
        var probs = evolve(full).probabilities()
        var meas = full.copy()
        meas.measure_all()
        var res = sample(meas, SHOTS, seed=UInt64(77 + gi))
        var desc = String()
        for k in range(len(mc.masks)):
            var ev = 0.0
            for o in range(len(probs)):
                ev += probs[o] * (-1.0 if pop_parity(o & mc.masks[k]) else 1.0)
            exact_cs += mc.coeffs[k] * mc.signs[k] * ev
            var s = 0.0
            for o in res.memory:
                s += -1.0 if pop_parity(o & mc.masks[k]) else 1.0
            var mean = s / Float64(SHOTS)
            shot_cs += mc.coeffs[k] * mc.signs[k] * mean
            var cov_free = max(0.0, 1.0 - mean * mean) / Float64(SHOTS)
            var_cs += mc.coeffs[k] * mc.coeffs[k] * cov_free
            desc += " Z-mask " + String(mc.masks[k]) + " sign " + String(mc.signs[k])
        print("  set", gi, ": terms", len(g_com[gi]), "| CX gates", mc.num_cx, "|", desc)
    # terms inside one set are read from the same shots; the variance above ignores
    # their covariance, so widen the band with the worst case (|c1| + |c2|)^2 instead
    var worst_sd = 0.0
    for gi in range(len(g_com)):
        var s = 0.0
        for i in g_com[gi]:
            s += abs(op.terms[i].coeff)
        worst_sd += s * s / Float64(SHOTS)
    worst_sd = sqrt(worst_sd)
    print("commuting-set exact =", exact_cs, " shot estimate =", shot_cs, "+/-", sqrt(var_cs),
          "(worst-case sd", worst_sd, ")")
    _ = log.check("second commuting set needs 1 CX (page)", len(cx_counts) == 2 and cx_counts[0] + cx_counts[1] == 1,
                  "cx per set=" + String(cx_counts[0]) + "," + String(cx_counts[1]))
    _ = log.close("exact value through commuting-set circuits", exact_cs, 0.1, 1e-12)
    _ = log.close("commuting-set shot estimate within 4 sigma", shot_cs, 0.1, 4.0 * worst_sd)
    # Sign validation on random entangled states (all three terms non-zero).
    var hea = hardware_efficient_ansatz(N, 2, "linear", "ry,rz")
    var rng = Rng(5)
    var worst = 0.0
    for _ in range(4):
        var x = List[Float64]()
        for _ in range(hea.num_params()):
            x.append(6.283185307179586 * rng.uniform())
        var st = hea.bind(x)
        var direct = evolve(st).expectation(op)
        worst = max(worst, abs(exact_through_sets(st, op, g_com) - direct))
    _ = log.close("commuting-set circuits on random states", worst, 0.0, 1e-12)
    print("page shot values (10,000 shots): naive 0.10184, NonConflictingSets 0.10088, CommutingSets 0.10388")
    log.finish()
