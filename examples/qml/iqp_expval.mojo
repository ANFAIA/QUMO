"""IQP circuits: expectation values without a state vector.

Reproduces: PennyLane 0.44 - qml.IQP, qml.qnn.iqp_expval (arXiv:2503.02934)
Source: https://docs.pennylane.ai/en/stable/code/api/pennylane.qnn.iqp_expval.html (PennyLane docs, Apache-2.0)
Framework: PennyLane (Xanadu) - https://github.com/PennyLaneAI/pennylane

Sampling an IQP circuit is believed hard, yet <Z_a> is an average of cosines over
random bitstrings, so training needs no quantum computer and the width is free.
PennyLane's 2-qubit demo has the exact value 0.17317819 for all three operators.

Checks:
- the state vector gives that value;
- Monte Carlo agrees within its standard error;
- a 2000-qubit model runs, with a gradient.

Run: .venv/bin/mojo run -I . examples/qml/iqp_expval.mojo
example-meta: {"framework": "pennylane", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from qumo.ir.circuit import Circuit
from qumo.ir.pauli import parse_pauli_sum
from qumo.sim.runner import evolve
from qumo.algorithms.iqp import IqpModel, append_iqp, iqp_expval
from qumo.testing.checks import CheckLog


def main() raises:
    var log = CheckLog("iqp_expval")
    var m = IqpModel(2)
    m.add_generator([0])
    m.add_generator([1])
    m.add_generator([0, 1])
    var w: List[Float64] = [1.0, 1.0, 1.0]
    var c = Circuit(2)
    append_iqp(c, m, w)
    print(c.draw())
    _ = log.close("state vector value", evolve(c).expectation(parse_pauli_sum("Z0")), 0.17317819, 1e-8)
    var ops = List[List[Int]]()
    ops.append([1])
    ops.append([0])
    ops.append([0, 1])
    var r = iqp_expval(m, w, ops, 200000, 3)
    print(r)
    for i in range(3):
        _ = log.close("Monte Carlo value " + String(i), r.means[i], 0.17317819, 5 * r.stds[i])
    var wide = IqpModel(2000)
    for q in range(1999):
        wide.add_generator([q, q + 1], 0)
    var g = iqp_expval(wide, [0.1], [[0, 1000], [1999]], 2000, 1, gradient=True)
    print(g.means[0], g.gradients[0][0])
    _ = log.close("2000 qubits, two operators", Float64(len(g.means)), 2.0, 0.0)
    log.finish()
