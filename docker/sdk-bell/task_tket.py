# Build a Bell circuit, optimize it for the cx, rz, rx gate set, simulate it.
import numpy as np
from pytket import Circuit, OpType
from pytket.passes import AutoRebase, FullPeepholeOptimise, SequencePass

c = Circuit(2)
c.H(0)
c.CX(0, 1)
SequencePass([FullPeepholeOptimise(), AutoRebase({OpType.CX, OpType.Rz, OpType.Rx})]).apply(c)
print("compiled:", c.n_gates, "gates,", c.n_gates_of_type(OpType.CX), "cx")
p = np.abs(c.get_statevector()) ** 2
shots = np.random.default_rng(7).multinomial(1000, p / p.sum())
print({format(i, "02b"): int(n) for i, n in enumerate(shots) if n})
