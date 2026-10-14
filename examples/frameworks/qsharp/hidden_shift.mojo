"""Hidden shift of an inner-product bent function on 10 qubits (Q# sample).

Reproduces: Q# (Microsoft QDK) - Hidden shift sample
Source: https://github.com/microsoft/qdk/blob/main/samples/algorithms/HiddenShift.qs (MIT)

The Q# sample hides a shift s in g(x) = f(x XOR s), where f is the
inner-product bent function on 2u = 10 qubits,
    f(x) = sum_i x_i x_(i+u)  (mod 2),   u = 5,
applied as the phase (-1)^f(x) by CZ(i, i+5), i = 0..4. f is its own dual,
so the sample uses the same oracle for f* and runs
    H^n  O_g  H^n  O_f*  H^n,   O_g = X^s . O_f . X^s,
then measures every qubit; the little-endian result is s. It finds the
shifts 170, 512 and 999 ("Found <shift> successfully!").

QUMO writes the same circuit gate for gate; Q#'s ApplyXorInPlace and
ResultArrayAsInt are little-endian, like QUMO's integers, so the shifts are
the same numbers.

Checks (per shift 170, 512, 999):
- the exact probability of |s> is 1 (1e-12);
- every one of 100 sampled shots returns s (page: "Found <shift> successfully!").

Run: .venv/bin/mojo run -I . examples/frameworks/qsharp/hidden_shift.mojo
example-meta: {"framework": "qsharp", "tier": "fast", "timeout": 60, "requires": [], "args": []}
"""

from qumo.ir.circuit import Circuit
from qumo.sim.runner import evolve, sample
from qumo.algorithms.library import bitstring_to_int
from qumo.testing.checks import CheckLog

comptime NQ = 10
comptime U = NQ // 2
comptime SHOTS = 100


def bent(mut c: Circuit):
    """(-1)^IP(x): CZ between the two halves of the register."""
    for i in range(U):
        c.cz(i, i + U)


def shifted_bent(mut c: Circuit, shift: Int):
    for q in range(NQ):
        if ((shift >> q) & 1) == 1:
            c.x(q)
    bent(c)
    for q in range(NQ):
        if ((shift >> q) & 1) == 1:
            c.x(q)


def hidden_shift_circuit(shift: Int, measure: Bool) -> Circuit:
    var c = Circuit(NQ, name="hidden_shift")
    for q in range(NQ):
        c.h(q)
    shifted_bent(c, shift)
    for q in range(NQ):
        c.h(q)
    bent(c)  # dual f* = f for the inner-product function
    for q in range(NQ):
        c.h(q)
    if measure:
        for q in range(NQ):
            c.measure(q, q)
    return c^


def main() raises:
    var log = CheckLog("qsharp/hidden_shift")
    var shifts: List[Int] = [170, 512, 999]
    for s in shifts:
        var p = evolve(hidden_shift_circuit(s, False)).probabilities()[s]
        var r = sample(hidden_shift_circuit(s, True), SHOTS, seed=UInt64(s))
        var found = bitstring_to_int(r.most_frequent())
        print("shift", s, ": P(|s>) =", p, " sampled", r.counts.get(r.most_frequent(), 0), "/", SHOTS, "->", found,
              "Found " + String(found) + " successfully!" if found == s else "FAILED")
        _ = log.close("P(s) = 1 for shift " + String(s), p, 1.0, 1e-12)
        _ = log.check("all shots return " + String(s), found == s and len(r.counts) == 1,
                      String(len(r.counts)) + " distinct outcomes")
    log.finish()
