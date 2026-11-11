"""The names a first QUMO program needs, in one import.

    from qumo.prelude import *

    def main() raises:
        var c = parse_qasm("OPENQASM 3.0; include \"stdgates.inc\"; qubit[2] q; bit[2] c; h q[0]; cx q[0], q[1]; c = measure q;")
        print(sample_auto(c, 1000, seed=7))          # {"00": 503, "11": 497}
        print(statevector(c))                        # |00> and |11>
        print(expectation(c, parse_pauli_sum("Z0 Z1")))
        var r = transpile(c, parse_target("grid:2x2/sc_cz"))
        print(r)                                     # per-pass report
        print(to_qasm(r.circuit))

Explicit names work the same way: `from qumo.prelude import Circuit, sample_auto`.

What is here, and where it lives (import from there for anything else):

* build and read circuits - `Circuit`, `Op` (`qumo.ir.circuit`, `qumo.ir.gates`),
  `parse_qasm`, `to_qasm` (`qumo.ir.qasm`; `c.draw()` draws a circuit), `PauliSum`, `parse_pauli_sum`
  (`qumo.ir.pauli`);
* simulate - `sample_auto` (`qumo.exec.dispatch`: picks the simulator), `sample`,
  `evolve`, `statevector` (the same function as `evolve`, under the name the CLI
  uses), `expectation` (`qumo.sim.runner`), with the result types `SampleResult`
  and `StateVector`;
* compile - `Target`, `parse_target` (`qumo.compiler.target`), `transpile`
  (`qumo.compiler.pass_manager`) and its result type `CompileResult`.

It is a list of re-exports (plus the one-line `statevector` alias): no code
runs at import and a call through the prelude is the same call as through the
defining module, so it costs nothing at run time, and a program builds as fast
with it as with the deep imports. Variational algorithms, QEC, noise
models, fault tolerance, GPU and hardware clients are not here on purpose: they
stay one explicit import away (`qumo.algorithms.variational`, `qumo.qec`, ...)
so that the prelude stays small. Seeds: sampling defaults to 1234, `transpile`
to 7; pass `seed=` to choose.
"""

from qumo.ir.circuit import Circuit
from qumo.ir.gates import Op
from qumo.ir.pauli import PauliSum, parse_pauli_sum
from qumo.ir.qasm import parse_qasm, to_qasm
from qumo.sim.statevector import StateVector
from qumo.sim.runner import SampleResult, sample, evolve, expectation
from qumo.exec.dispatch import sample_auto
from qumo.compiler.target import Target, parse_target
from qumo.compiler.pass_manager import CompileResult, transpile


def statevector(circuit: Circuit, params: List[Float64] = List[Float64]()) raises -> StateVector:
    """The exact final state of `circuit`: `evolve`, under the name of the CLI command."""
    return evolve(circuit, params)
