"""Stabilizer branching: near-Clifford circuits far past the state vector.

Reproduces: QDK 1.32 - stabilizer branching simulator
Source: https://github.com/microsoft/qdk/blob/main/samples/notebooks/stabilizer_branching.ipynb (MIT)

One Clifford frame and a sparse list of amplitudes: Clifford gates touch the frame
only, a rotation at most doubles the list, and measuring never grows it, so cost
follows the non-Clifford gates. The notebook's echo idea: T and Tdg on a wide GHZ
state cancel, so the data qubits return to 0.

Checks:
- the echo on 111 qubits gives one outcome;
- one T on 200 qubits gives the two expected outcomes with cos^2(pi/8) weights;
- a circuit with too many T gates reports QM-E412.

Run: .venv/bin/mojo run -I . examples/frameworks/qsharp/stabilizer_branching.mojo
example-meta: {"framework": "qsharp", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from std.math import cos, pi
from qumo.ir.circuit import Circuit
from qumo.sim.branching import sample_branching, BranchingSimulator
from qumo.sim.rng import Rng
from qumo.testing.checks import CheckLog


def main() raises:
    var log = CheckLog("stabilizer_branching")
    var n = 111
    var c = Circuit(n)
    c.h(0)
    for q in range(n - 1):
        c.cx(q, q + 1)
    for q in range(0, n, 11):
        c.h(q)
        c.t(q)
        c.h(q)
    for q in range(0, n, 11):
        c.h(q)
        c.tdg(q)
        c.h(q)
    for q in range(n - 2, -1, -1):
        c.cx(q, q + 1)
    c.h(0)
    for q in range(62):
        c.measure(q, q)
    var r = sample_branching(c, 200, 5)
    _ = log.close("the echo returns to one outcome", Float64(len(r.counts)), 1.0, 0.0)
    _ = log.check("data qubits back to 0", r.most_frequent().find("1") < 0)

    var w = Circuit(200, 1)
    w.h(0)
    w.t(0)
    w.h(0)
    w.measure(0, 0)
    var s = sample_branching(w, 20000, 7)
    var p0 = cos(pi / 8) * cos(pi / 8)
    _ = log.close("H T H on 200 qubits", s.probability("0"), p0, 0.015)

    var hard = Circuit(24)
    for q in range(24):
        hard.h(q)
        hard.t(q)
        hard.h(q)
    var sim = BranchingSimulator(24, 1)
    var rng = Rng(1)
    var code = String("")
    try:
        sim.run(hard, rng)
    except e:
        code = String(e)
    _ = log.check("the branch limit is reported", code.startswith("QM-E412"), code)
    log.finish()
