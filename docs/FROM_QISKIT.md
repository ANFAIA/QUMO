# Coming from Qiskit, Cirq or PennyLane

This page maps what you already do in another SDK to QUMO, and shows how to move a circuit across with OpenQASM. For a step-by-step introduction read [TUTORIAL.md](TUTORIAL.md) first; it uses the same calls.

What was checked: the Qiskit column against Qiskit 2.6.1 and qiskit-aer 0.18.2, the Cirq column against cirq-core 1.7.0, and every QUMO cell and both round trips on this repository (Mojo 1.2.0). PennyLane is not installed in this project's Python environment, so the PennyLane section at the end is from PennyLane's documentation and was **not run here**.

## What is different

- **Three ways in.** QUMO is a Mojo library, a command line tool (`build/qumo`) and a Python package (`import qumo`, [PYTHON.md](PYTHON.md)). The Mojo API is the complete one. The CLI covers running, inspecting, converting, compiling (`transpile`), device emulation (`emulate`) and VQE; everything else is reachable as `build/qumo tool <name> '<json>'`. The Python package has a circuit builder whose gate methods have the names and argument order of the Mojo ones (angles first, then qubits: `c.rx(0.3, 0)`, where Qiskit also writes `qc.rx(0.3, 0)`), and functions with keyword arguments that return dicts and NumPy arrays; all the computation runs in the compiled Mojo core.
- **OpenQASM is the exchange format.** QUMO reads OpenQASM 2 and 3 and writes both. That is how circuits move between QUMO and any other SDK.
- **Same bit order as Qiskit.** Qubit 0 is the least significant bit; in a result key the leftmost character is the highest classical bit. Cirq orders the other way: its `LineQubit(0)` is the most significant bit of a state-vector index.
- **Observables are text.** `"Z0 Z1 + 0.5*X2"` means Z on qubits 0 and 1 plus half X on qubit 2. Dense labels such as `"ZZI"` also work, with the leftmost character on the highest qubit, like Qiskit's `SparsePauliOp("ZZI")`.
- **A device is data.** There are no backend objects. A `Target` is a coupling map, a basis and optional error rates, and it can be written as one line: `"grid:4x4/sc_cz"`, `"trapped_ion:8"`.
- **Noise lives in the circuit.** A noise channel is an instruction, as in Cirq and Stim, not a separate noise model attached to a simulator as in Aer. A helper turns a target's error rates into channels (`emulate`).
- **Everything is seeded.** Same seed, same counts.

## Mapping table

`c` is a QUMO circuit, `qc` a Qiskit `QuantumCircuit`, `circuit` a Cirq `Circuit`. Mojo imports are listed under the table.

| Task | Qiskit | Cirq | QUMO (Mojo) |
|---|---|---|---|
| Build a circuit | `qc = QuantumCircuit(2); qc.h(0); qc.cx(0, 1)` | `a, b = cirq.LineQubit.range(2); circuit = cirq.Circuit(cirq.H(a), cirq.CNOT(a, b))` | `var c = Circuit(2); c.h(0); c.cx(0, 1)` |
| Measure | `qc.measure_all()` | `circuit + cirq.measure(a, b, key="m")` | `c.measure_all()` or `c.measure(q, clbit)` |
| Sample counts | `StatevectorSampler(seed=7).run([qc], shots=1000).result()[0].data.meas.get_counts()` | `cirq.Simulator(seed=7).run(circuit, repetitions=1000).histogram(key="m")` | `sample(c, 1000, seed=7).counts` |
| State vector | `Statevector(qc)` | `cirq.final_state_vector(circuit)` | `var sv = evolve(c)`, then `sv.re[i]`, `sv.im[i]` |
| Expectation value | `Statevector(qc).expectation_value(SparsePauliOp("ZZ"))` or `StatevectorEstimator().run([(qc, op)])` | `cirq.Simulator().simulate_expectation_values(circuit, [cirq.Z(a) * cirq.Z(b)])` | `expectation(c, parse_pauli_sum("Z0 Z1"))` |
| Free parameter | `th = Parameter("theta"); qc.ry(th, 0)` | `cirq.ry(sympy.Symbol("theta"))(a)` | `var th = c.parameter("theta"); c.ry(th, 0)` |
| Bind parameters | `qc.assign_parameters({th: 1.0})` | `cirq.resolve_parameters(circuit, {"theta": 1.0})` | `c.bind([1.0])`, or pass values: `expectation(c, obs, [1.0])` |
| Compile to a coupling map and basis | `transpile(qc, coupling_map=[[0, 1]], basis_gates=["rz", "sx", "x", "cz"], optimization_level=2)` | `cirq.optimize_for_target_gateset(circuit, gateset=cirq.CZTargetGateset())` (gate set only; routing is separate) | `transpile(c, parse_target("line:2/rz,sx,x,cz"), 2)` |
| Noise | `NoiseModel()` + `add_all_qubit_quantum_error(depolarizing_error(0.02, 1), ["h"])`, run on `AerSimulator(noise_model=nm)` | `circuit.with_noise(cirq.depolarize(0.02))` | `c.depolarize1(0.02, q)` after a gate; `emulate(compiled, target, shots)` for a whole device model |
| Draw | `qc.draw()` | `print(circuit)` | `c.draw()` or `c.draw(style="unicode")` |
| OpenQASM out | `qasm3.dumps(qc)`, `qasm2.dumps(qc)` | `cirq.qasm(circuit)` (OpenQASM 2) | `to_qasm(c)` (3), `to_qasm2(c)` (2) |
| OpenQASM in | `qasm3.loads(text)`, `qasm2.loads(text)` | `cirq.contrib.qasm_import.circuit_from_qasm(text)` | `parse_qasm(text)`, `load_qasm(path)` (2 or 3) |

Mojo imports for the QUMO column:

```mojo
from qumo.ir.circuit import Circuit
from qumo.ir.pauli import parse_pauli_sum
from qumo.ir.qasm import parse_qasm, load_qasm, to_qasm
from qumo.ir.qasm2 import to_qasm2
from qumo.sim.runner import sample, evolve, expectation
from qumo.compiler.target import parse_target
from qumo.compiler.pass_manager import transpile
from qumo.exec.emulate import emulate
```

Notes on the Qiskit and Cirq cells: `qasm3.loads` needs the `qiskit-qasm3-import` package; `qc` in the "Sample counts" row must have measurements (`measure_all` names the register `meas`).

### The same tasks from the QUMO command line and from Python

| Task | QUMO CLI | QUMO Python (`import qumo`) |
|---|---|---|
| Build a circuit | write an OpenQASM file | `c = qumo.Circuit(2).h(0).cx(0, 1)` |
| Measure | `c = measure q;` in the file | `c.measure_all()` or `c.measure(q, clbit)` |
| Sample counts | `build/qumo run bell.qasm --shots 1000 --seed 7` | `qumo.sample(c, shots=1000, seed=7)` (a `dict` with `.method`) |
| State vector | `build/qumo statevector bell.qasm` | `qumo.statevector(c)` (NumPy `complex128`), `qumo.probabilities(c)` |
| Expectation value | `build/qumo expect bell.qasm "Z0 Z1"` | `qumo.expectation(c, "Z0 Z1")` or `qumo.expectation(c, [("ZZ", 1.0)])` |
| Free parameter | `input float theta;` in the file | `c.ry("theta", 0)`, or `th = c.parameter("theta"); c.ry(2 * th, 0)` |
| Bind parameters | `--params '{"theta": 1.0}'` or `--params '[1.0]'` | `c.bind(theta=1.0)`, or pass values: `qumo.expectation(c, obs, params={"theta": 1.0})` |
| Compile | `build/qumo transpile bell.qasm --target line:2/sc_cz` (`-o out.qasm` saves the result) | `r = qumo.transpile(c, "line:2/sc_cz", level=2)`, or `qumo.transpile(c, coupling_map=[(0, 1)], basis=["rz", "sx", "x", "cz"])`; then `r.circuit`, `r.initial_layout`, `r.final_layout`, `r.metrics` |
| Noise | `#pragma qumo depolarize1(0.02) q[0]` in the file | `c.depolarize1(0.02, 0)` after a gate |
| Device emulation | `build/qumo emulate bell.qasm --target superconducting_grid:2x2` | not in the package yet; the hardware plugin's local provider does it ([HARDWARE_PLUGINS.md](HARDWARE_PLUGINS.md)) |
| VQE | `build/qumo vqe "Z0 Z1 + 0.5*X0 + 0.5*X1"` | `qumo.vqe(qumo.hardware_efficient_ansatz(2, layers=2), "Z0 Z1 + 0.5*X0 + 0.5*X1")["energy"]` |
| Draw | `build/qumo draw bell.qasm [--unicode]` | `print(c)` or `c.draw(style="ascii")` |
| OpenQASM out | `build/qumo convert bell.qasm` (3), `--to qasm2` (2), `-o out.qasm` | `c.to_qasm()` (3), `c.to_qasm2()` (2) |
| OpenQASM in | every command reads OpenQASM 2 or 3; `-` reads stdin | `qumo.load(path_or_text)` (2 or 3) |
| Check a file | `build/qumo validate bell.qasm`, `build/qumo info bell.qasm` | `c.count_ops()`, `c.depth()`, `c.num_qubits` |

Python functions take keyword arguments, and the defaults are the CLI's (`shots=1024`, `seed=1234`; `level=2`, `seed=7` for `transpile`). Every Python cell above was run for this page; the full reference is [PYTHON.md](PYTHON.md). The CLI commands are described by `build/qumo <command> --help`, and an option a command does not know is an error (QM-E921), not ignored. Operations without a command of their own are reachable as `build/qumo tool <name> '<json>'` (`build/qumo tool list`).

### Qiskit primitives

QUMO has the same two primitives with the same "PUB" idea (a circuit plus what to evaluate) in `qumo.exec.primitives`: `SamplerPub` with `run_sampler`, and `EstimatorPub` with `run_estimator`. See the execution section of [ARCHITECTURE.md](ARCHITECTURE.md) and [examples/foundations/chsh.mojo](../examples/foundations/chsh.mojo).

## Moving a circuit with OpenQASM

### Qiskit to QUMO and back

This script takes a Qiskit circuit, loads it in QUMO, checks an expectation value against Qiskit, compiles it in QUMO for a 3-qubit line, loads the compiled circuit back in Qiskit and checks that it prepares the same state. Run it after `make py`, with a Python that has Qiskit and the `qumo` package (`make setup-py` creates `bench/.venv` with Qiskit and `make install-py` installs `qumo` into it):

```python
# Qiskit -> QUMO -> Qiskit through OpenQASM, checked with state vectors.
import qumo
from qiskit import QuantumCircuit, qasm2, qasm3
from qiskit.quantum_info import Statevector, SparsePauliOp

qc = QuantumCircuit(3)
qc.h(0); qc.cx(0, 1); qc.rz(0.3, 1); qc.ry(1.1, 2); qc.cz(1, 2)

# 1. Qiskit -> QUMO. Both dialects load with the same call.
c = qumo.load(qasm3.dumps(qc))            # or qasm2.dumps(qc)
obs = SparsePauliOp.from_sparse_list([("ZZ", [0, 1], 1.0), ("Z", [2], 0.5)], num_qubits=3)
print("QUMO  :", qumo.expectation(c, "Z0 Z1 + 0.5*Z2"))
print("Qiskit:", Statevector(qc).expectation_value(obs).real)

# 2. Compile in QUMO for a 3-qubit line with rz, sx, x, cz.
r = qumo.transpile(c, "line:3/sc_cz", level=2)

# 3. QUMO -> Qiskit. The compiled circuit acts on physical qubits: logical l is on final_layout[l].
back = qasm3.loads(r.circuit.to_qasm())   # or qasm2.loads(r.circuit.to_qasm2())
expected = QuantumCircuit(3)
expected.compose(qc, qubits=r.final_layout, inplace=True)
print("same state up to global phase:", Statevector(expected).equiv(Statevector(back)))
print("gates:", dict(back.count_ops()))
```

```
QUMO  : 1.2267980607127884
Qiskit: 1.2267980607127882
same state up to global phase: True
gates: {'rz': 7, 'sx': 5, 'cz': 2}
```

Both directions were also run with OpenQASM 2 (`qasm2.dumps` in, `to_qasm2` / `qasm2.loads` out) with the same result. State vectors compare directly, since the bit order is the same: `numpy.allclose(qumo.statevector(c), Statevector(qc).data)` is `True` here.

### Cirq to QUMO and back

Cirq writes and reads OpenQASM 2.

```python
# Cirq -> QUMO -> Cirq through OpenQASM 2, checked with state vectors.
import cirq
from cirq.contrib.qasm_import import circuit_from_qasm
import qumo

a, b = cirq.LineQubit.range(2)
circuit = cirq.Circuit(cirq.H(a), cirq.CNOT(a, b), cirq.rz(0.3)(b))

c = qumo.load(cirq.qasm(circuit))                    # Cirq writes OpenQASM 2
z = cirq.Z(a) * cirq.Z(b)
x = cirq.X(a) * cirq.X(b)
psi = cirq.final_state_vector(circuit)
qm = {a: 0, b: 1}
print("QUMO:", qumo.expectation(c, "Z0 Z1"), qumo.expectation(c, "X0 X1"))
print("Cirq:", z.expectation_from_state_vector(psi, qm).real, x.expectation_from_state_vector(psi, qm).real)

back = circuit_from_qasm(c.to_qasm2())               # QUMO -> Cirq
print("same unitary up to global phase:",
      cirq.equal_up_to_global_phase(cirq.unitary(back), cirq.unitary(circuit), atol=1e-9))
```

```
QUMO: 1.0 0.9553364890839552
Cirq: 0.9999999403953552 0.9553363919258118
same unitary up to global phase: True
```

The Cirq numbers differ in the eighth digit because Cirq's default simulator uses single precision.

### From the command line

Any SDK that can write OpenQASM to a file works with the CLI directly:

```bash
build/qumo validate my_circuit.qasm          # parse and lint, with line and column on errors
build/qumo run my_circuit.qasm --shots 1000 --seed 7
build/qumo convert my_circuit.qasm --to qasm2 -o for_cirq.qasm
```

### What does not cross

- **QUMO pragmas.** Noise channels, detectors and observables are written as `#pragma qumo ...` in OpenQASM 3. Other tools skip or reject pragma lines. OpenQASM 2 has no pragmas, so the OpenQASM 2 writer replaces each channel with a comment (`// noise channel not exported: ...`). Send the noiseless circuit and add noise on the other side.
- **Other tools' extensions.** QUMO skips `#pragma braket ...` and other foreign pragmas silently, and skips `@annotations` with warning QM-W052. Subroutines and pulse-level `cal` / `defcal` are not supported. The exact subset QUMO accepts is in [AI_NATIVE.md](AI_NATIVE.md).
- **Programs with `while` and classical variables** load with `load_any` in Mojo and `qumo.native.load_program` in Python, not with `parse_qasm` / `qumo.load` ([CONTROL_FLOW.md](CONTROL_FLOW.md)). Whether another SDK can read what QUMO writes for them depends on that SDK's OpenQASM 3 support; this was not tested here.
- **Layouts.** A compiled circuit is on physical qubits. Keep `r.initial_layout` and `r.final_layout` from `transpile` (the same names in Mojo and Python) to map results back, as the Qiskit script above does.
- **Global phase.** QUMO tracks it and writes `gphase(...)` in OpenQASM 3. It does not change any measurement.

## PennyLane (not run here)

PennyLane is not installed in this project's Python environment, so nothing in this section was executed. The names below are from PennyLane's documentation; check them against your version.

| Task | PennyLane | QUMO (Mojo) |
|---|---|---|
| Circuit and expectation value | a `@qml.qnode(dev)` function that applies `qml.Hadamard(wires=0)`, `qml.CNOT(wires=[0, 1])` and returns `qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))` | `expectation(c, parse_pauli_sum("Z0 Z1"))` |
| Samples | `qml.device("default.qubit", shots=1000)` with `return qml.counts()` | `sample(c, 1000, seed=7).counts` |
| Gradient | `qml.grad(circuit)(theta)` | `qumo.algorithms.gradients` (adjoint and parameter shift), see [ALGORITHMS.md](ALGORITHMS.md) |
| Import OpenQASM | `qml.from_qasm(text)` | `parse_qasm(text)` |

QUMO's QML examples follow PennyLane's demos and are checked against reference numbers recorded from PennyLane 0.45.1. Mind the wire order: PennyLane's wire 0 is the most significant bit, so PennyLane wire `w` is QUMO qubit `n-1-w`. Details are in [QML.md](QML.md).
