"""Reichardt QEC cycles, Iceberg Pauli exponentials, the Helix code.

Reproduces: Guppy FT 0.5 - QECStyle.Reichardt, Iceberg pauli_exp, Helix (arXiv:2609.03194, 1804.06995)
Source: https://github.com/Quantinuum/guppyft/blob/main/docs/CHANGELOG.md (Apache-2.0)

The `reichardt` style measures all six Steane stabilizers with 3 ancillas instead
of fresh logical blocks. Iceberg's block-wide Paulis are single weight-2 rotations
modulo the stabilizers. Helix [[20,2,6]] is the code definition only: QUMO has no
Helix architecture. With `method="auto"`, an encoded program past 26 qubits runs on
stabilizer branching instead of failing.

Checks:
- reichardt needs fewer qubits than steane and keeps the logical action;
- every seven-exponential gadget passes the Clifford verifier at theta = pi/2;
- Helix commutes, has rank 18 and distance 6;
- a 30-qubit encoded T program emulates, matching the unencoded circuit.

Run: .venv/bin/mojo run -I . examples/ft/reichardt_iceberg_helix.mojo
example-meta: {"framework": "guppyft", "tier": "fast", "timeout": 240, "requires": [], "args": []}
"""

from std.math import pi
from qumo.ir.circuit import Circuit
from qumo.ir.gates import Op
from qumo.ft.arch import SteaneBuilder, QECPolicy
from qumo.ft.code import helix_code, iceberg_code
from qumo.ft.encode import encode
from qumo.ft.emulate import run_encoded, reference_counts, total_variation
from qumo.ft.gadgets import iceberg_exp_kind, iceberg_rotation
from qumo.ft.logical import LOp
from qumo.ft.verify import valid_clifford_implementation
from qumo.testing.checks import CheckLog


def main() raises:
    var log = CheckLog("reichardt_iceberg_helix")
    var c = Circuit(1, 1)
    c.h(0)
    c.s(0)
    c.h(0)
    c.measure(0, 0)
    var steane = SteaneBuilder().with_qec_policy(QECPolicy("steane", 0.5)).build(2)
    var dense = SteaneBuilder().with_qec_policy(QECPolicy("reichardt", 0.5)).build(2)
    steane.qec.costs[LOp.H] = 1.0
    dense.qec.costs[LOp.H] = 1.0
    var a = encode(steane, c)
    var b = encode(dense, c)
    print("physical qubits: steane", a.circuit.num_qubits, "reichardt", b.circuit.num_qubits)
    _ = log.check("reichardt needs fewer qubits", b.circuit.num_qubits < a.circuit.num_qubits)
    var tv = total_variation(reference_counts(c, 4000, 5), run_encoded(b, 4000, 6))
    _ = log.close("logical action kept", tv, 0.0, 0.05)

    var k = 4
    var code = iceberg_code(k)
    var words: List[String] = ["XXXX", "YYYY", "ZZZZ", "XIXX", "ZZIZ", "XZXX", "ZXZZ"]
    var ok = 0
    for w in words:
        var sem = Circuit(k)
        var last = -1
        var bs = w.as_bytes()
        for i in range(k):
            if Int(bs[i]) != ord("I"):
                if Int(bs[i]) == ord("X"):
                    sem.h(i)
                elif Int(bs[i]) == ord("Y"):
                    sem.rx(pi / 2, i)
                if last >= 0:
                    sem.cx(last, i)
                last = i
        sem.rz(pi / 2, last)
        var chain = List[Int]()
        for i in range(k):
            if Int(bs[i]) != ord("I"):
                chain.append(i)
        var j = len(chain) - 1
        while j > 0:
            sem.cx(chain[j - 1], chain[j])
            j -= 1
        for i in range(k):
            if Int(bs[i]) == ord("X"):
                sem.h(i)
            elif Int(bs[i]) == ord("Y"):
                sem.rx(-pi / 2, i)
        var kind = iceberg_exp_kind(w)
        if valid_clifford_implementation(sem, iceberg_rotation(code.n, kind[0], kind[1], -1, pi / 2), code).ok:
            ok += 1
    _ = log.close("Iceberg exponentials verified", Float64(ok), 7.0, 0.0)

    var h = helix_code()
    h.validate()
    _ = log.close("Helix distance", Float64(h.css_distance()), 6.0, 0.0)

    var d = Circuit(4, 4)
    d.h(0)
    d.t(0)
    d.h(0)
    d.cx(0, 1)
    d.cx(1, 2)
    d.cx(2, 3)
    for q in range(4):
        d.measure(q, q)
    var prog = encode(SteaneBuilder().build(5), d)
    var r = run_encoded(prog, 400, 7)
    print(prog.circuit.num_qubits, "qubits, method", r.method)
    _ = log.check("wide programs run on branching", r.method == "branching" and prog.circuit.num_qubits > 26)
    _ = log.close("matches the unencoded circuit", total_variation(reference_counts(d, 8000, 5), r), 0.0, 0.1)
    log.finish()
