"""Entanglement swapping with mid-circuit measurement and conditional corrections.

Reproduces: TKET - Iterated entanglement swapping (single swap) and conditional gates
Source: https://docs.quantinuum.com/tket/user-guide/examples/algorithms_and_protocols/entanglement_swapping.html (pytket user guide, Apache-2.0)
Source: https://docs.quantinuum.com/tket/user-guide/examples/circuit_construction/conditional_gate_example.html (pytket user guide, Apache-2.0)

Ava (q0) shares a Bell pair with Bella (q1); Bella's second qubit (q2) shares
one with Charlie (q3). Bella teleports q1 to Charlie: CX(q1, q2), H(q1),
measure q1 -> c0 and q2 -> c1, then Charlie applies X if c1 and Z if c0 (the
page's corrections "X when d = 2 or 3, Z when d = 1 or 3" with d = c0 + 2 c1).
Ava and Charlie then hold |Phi+> although they never interacted. The page
verifies with ZZ statistics (about 50/50 between 00 and 11 over 2000 shots)
and tomography, and repeats it under a noise model (depolarising 0.03 on every
CX, readout error 0.05).

QUMO runs the feed-forward with trajectory sampling (`sample` switches to
per-shot trajectories for dynamic circuits). Corrections are written twice:
with single-bit conditions (`c_if`) and with TKET-style register conditions
(`c_if_register`, one gate per value of d).

Checks:
- with corrections, <Z0 Z3> = <X0 X3> = 1 and <Y0 Y3> = -1 on every shot
  (4000 shots per basis), for both conditional styles; Bell-state fidelity 1;
- without corrections the correlations vanish (within 4 sigma of 0);
- the 4 Bell-measurement outcomes are uniform (each within 4 sigma of 1/4);
- under the page's noise model (no number on the page) each correlator is
  within 4 sigma of the analytic Pauli-channel value: a 2-qubit depolarising
  error p scales every correlator by (1 - 16p/15) per CX and a readout error r
  by (1 - 2r) per record it touches, so <ZZ> = <XX> = (1-2r)^3 (1-16p/15)^3
  = 0.66123 and <YY> = -(1-2r)^4 (1-16p/15)^3 = -0.59511 (fidelity 0.72939;
  cross-checked with a 4-qubit density-matrix calculation).

Run: .venv/bin/mojo run -I . examples/frameworks/tket/entanglement_swapping.mojo
example-meta: {"framework": "tket", "tier": "fast", "timeout": 60, "requires": [], "args": []}
"""

from std.math import sqrt
from qumo.ir.circuit import Circuit
from qumo.sim.runner import sample
from qumo.testing.checks import CheckLog

comptime SHOTS = 4000
comptime DEP_ERR = 0.03  # page noise model: depolarising error on every CX
comptime RO_ERR = 0.05  # page noise model: readout error


def cx_noisy(mut c: Circuit, a: Int, b: Int, noisy: Bool):
    c.cx(a, b)
    if noisy:
        c.depolarize2(DEP_ERR, a, b)


def swap_circuit(corrections: Int, basis: String, noisy: Bool = False) raises -> Circuit:
    """Swap circuit. corrections: 0 none, 1 c_if bits, 2 c_if_register on d = c0 + 2 c1;
    basis: 'Z', 'X' or 'Y' readout of Ava (c2) and Charlie (c3)."""
    var c = Circuit(4, 4, name="ent_swap")
    c.h(0)
    cx_noisy(c, 0, 1, noisy)  # Ava - Bella[0]
    c.h(2)
    cx_noisy(c, 2, 3, noisy)  # Bella[1] - Charlie
    # Bella's Bell measurement (teleports q1 onto q3)
    cx_noisy(c, 1, 2, noisy)
    c.h(1)
    if noisy:
        c.x_error(RO_ERR, 1)
        c.x_error(RO_ERR, 2)
    c.measure(1, 0)
    c.measure(2, 1)
    if corrections == 1:
        c.x(3)
        c.c_if(1, 1)
        c.z(3)
        c.c_if(0, 1)
    elif corrections == 2:
        var bits: List[Int] = [0, 1]
        var d2: List[Int] = [0, 1]
        var d3: List[Int] = [1, 1]
        var d1: List[Int] = [1, 0]
        c.x(3)
        c.c_if_register(bits, d2)
        c.x(3)
        c.c_if_register(bits, d3)
        c.z(3)
        c.c_if_register(bits, d1)
        c.z(3)
        c.c_if_register(bits, d3)
    for q in [0, 3]:
        if basis == "X":
            c.h(q)
        elif basis == "Y":
            c.sdg(q)
            c.h(q)
    if noisy:
        c.x_error(RO_ERR, 0)
        c.x_error(RO_ERR, 3)
    c.measure(0, 2)
    c.measure(3, 3)
    return c^


def correlator(c: Circuit, seed: UInt64, mut bell_counts: List[Int]) raises -> Float64:
    """<P0 P3> from clbits 2 and 3; also tallies the Bell outcomes d = c0 + 2 c1."""
    var r = sample(c, SHOTS, seed=seed)
    var s = 0
    for m in r.memory:
        s += 1 if (((m >> 2) & 1) == ((m >> 3) & 1)) else -1
        bell_counts[m & 3] += 1
    return Float64(s) / Float64(SHOTS)


def main() raises:
    var log = CheckLog("tket/entanglement_swapping")
    var sigma = 1.0 / sqrt(Float64(SHOTS))
    var bases: List[String] = ["Z", "X", "Y"]
    var ideal: List[Float64] = [1.0, 1.0, -1.0]  # |Phi+>
    var style: List[String] = ["c_if bits", "c_if_register"]
    var bell = List[Int](length=4, fill=0)
    for s in range(2):
        var fid = 1.0
        for b in range(3):
            var c = swap_circuit(s + 1, bases[b])
            var e = correlator(c, UInt64(100 + 10 * s + b), bell)
            fid += e if b != 2 else -e
            print("corrections (", style[s], "):  <", bases[b], "0", bases[b], "3> =", e,
                  " method", sample(c, 1, seed=1).method)
            _ = log.close("<" + bases[b] + "0 " + bases[b] + "3> with " + style[s], e, ideal[b], 1e-12)
        fid /= 4.0
        print("  Bell-state fidelity (1 + <XX> - <YY> + <ZZ>)/4 =", fid)
        _ = log.close("Bell fidelity with " + style[s], fid, 1.0, 1e-12)
    var total = 0
    for k in range(4):
        total += bell[k]
    for k in range(4):
        var p = Float64(bell[k]) / Float64(total)
        _ = log.close("Bell outcome d=" + String(k) + " frequency", p, 0.25, 4.0 * sqrt(0.25 * 0.75 / Float64(total)))

    # Without the corrections Ava and Charlie hold an equal mixture of the four Bell states.
    for b in range(2):
        var unused = List[Int](length=4, fill=0)
        var e = correlator(swap_circuit(0, bases[b]), UInt64(300 + b), unused)
        print("no corrections:  <", bases[b], "0", bases[b], "3> =", e)
        _ = log.close("<" + bases[b] + "0 " + bases[b] + "3> without corrections", e, 0.0, 4.0 * sigma)

    # The page's noise model: depolarising 0.03 after every CX, readout error 0.05.
    var dep = (1.0 - 16.0 * DEP_ERR / 15.0) ** 3
    var ro = 1.0 - 2.0 * RO_ERR
    var model: List[Float64] = [ro ** 3 * dep, ro ** 3 * dep, -(ro ** 4) * dep]
    var fid_noisy = 1.0
    var fid_model = 1.0
    for b in range(3):
        var unused = List[Int](length=4, fill=0)
        var e = correlator(swap_circuit(1, bases[b], noisy=True), UInt64(500 + b), unused)
        fid_noisy += e if b != 2 else -e
        fid_model += model[b] if b != 2 else -model[b]
        print("noisy (dep 0.03, readout 0.05):  <", bases[b], "0", bases[b], "3> =", e, " model", model[b])
        var sd = sqrt((1.0 - model[b] * model[b]) / Float64(SHOTS))
        _ = log.close("noisy <" + bases[b] + "0 " + bases[b] + "3> vs Pauli-channel model", e, model[b], 4.0 * sd)
    fid_noisy /= 4.0
    fid_model /= 4.0
    print("  noisy Bell-state fidelity =", fid_noisy, " model", fid_model, "(page: 'reduced considerably', no number)")
    log.finish()
