# Tutorial: your first hour with QUMO

This tutorial goes from a two-qubit circuit to a small algorithm in eight steps. Each step shows the same thing three ways where it exists today: the `qumo` command line, a Mojo script, and Python (the `qumo` package). You need no quantum hardware and no GPU.

Every command and script on this page was run on this repository (Apple M5, macOS, Mojo 1.2.0). The outputs shown are the real outputs; with the same seed you get the same counts (the millisecond timings vary from run to run).

If you already use Qiskit, Cirq or PennyLane, read [FROM_QISKIT.md](FROM_QISKIT.md) next to this page.

## Before you start

Install and build once ([GETTING_STARTED.md](GETTING_STARTED.md) has the details):

```bash
git clone https://github.com/ANFAIA/QUMO.git && cd QUMO
make setup        # Mojo toolchain in .venv
make cli          # build/qumo, the command line tool
make py           # build/qumo_py.so, the compiled extension (only for the Python snippets)
pip install -e .  # the Python package `qumo` (only for the Python snippets; or: uv pip install -e .)
```

Run everything from the repository root.

**Running a Mojo script.** Save a snippet as `my.mojo` anywhere and run:

```bash
.venv/bin/mojo run -I . my.mojo
```

`-I .` tells Mojo where the `qumo` package is, so the command must run from the repository root (or give the path to the root after `-I`). Three things to know:

- `mojo run` compiles the script first. On an Apple M5 the scripts on this page took 3 to 5 seconds each. The first script that imports the compiler (step 6) took 18 seconds; later runs took 5, because Mojo caches compiled modules.
- To skip that wait, build once and run the binary: `.venv/bin/mojo build -I . -o my my.mojo` (2.5 s for the step 1 script), then `./my` (0.16 s).
- Mojo prints `Failed to initialize Crashpad. Crash reporting will not be available. ...` as the first line. It is harmless: it only means the crash reporter is not installed. The outputs below leave that line out.

**Running the Python snippets.** They use the `qumo` package ([PYTHON.md](PYTHON.md)) with Python 3.12 and NumPy. After `make py` and `pip install -e .` a plain `import qumo` works from any directory and in Jupyter; type the snippets into `python`, a notebook, or a file. Without installing, `import qumo` still works in a `python` started at the repository root and in a script saved there.

**Running the Mojo snippets with less typing.** The scripts on this page import each name from the module that defines it, so you can see where things live. `from qumo.prelude import *` brings the common ones (`Circuit`, `sample`, `sample_auto`, `statevector`, `expectation`, `parse_pauli_sum`, `parse_target`, `transpile`, ...) in one line; see [GETTING_STARTED.md](GETTING_STARTED.md#mojo).

**Conventions.** Qubit 0 is the least significant bit. In a result such as `01`, the leftmost character is the highest classical bit, as in Qiskit. Angles are in radians. Every random process takes a seed; when you give none, sampling uses 1234 and the compiler and VQE use 7.

**When a command is wrong.** `build/qumo <command> --help` lists a command's options. A misspelt option is an error with a suggestion (`unknown option '--shot' for 'qumo run'`, `did you mean '--shots'?`), not something the tool ignores.

## 1. A Bell pair and its histogram

A Bell pair is two qubits that always give the same answer: half the time `00`, half the time `11`.

**CLI.** The circuit is in [examples/foundations/bell.qasm](../examples/foundations/bell.qasm), written in OpenQASM 3:

```
OPENQASM 3.0;
include "stdgates.inc";

qubit[2] q;
bit[2] c;

h q[0];
cx q[0], q[1];
c = measure q;
```

```bash
build/qumo run examples/foundations/bell.qasm --shots 1000 --seed 7
```

```
examples/foundations/bell.qasm: 2 qubits, 1000 shots, seed 7, statevector:static, 2.047 ms
  00         484  0.4840  ###################
  11         516  0.5160  #####################
```

The first line names the simulator that ran (`statevector:static`) and how long it took. Add `--json` to get the same result as JSON. Change `--seed` and the counts change; keep it and they do not.

**Mojo.**

```mojo
from qumo.ir.circuit import Circuit
from qumo.sim.runner import sample


def main() raises:
    var c = Circuit(2)          # 2 qubits, all in |0>
    c.h(0)                      # superposition on qubit 0
    c.cx(0, 1)                  # entangle qubit 1 with it
    c.measure_all()             # adds 2 classical bits and measures into them
    var r = sample(c, 1000, seed=7)
    for key in r.sorted_keys():
        print(key, r.counts[key], r.probability(key))
    print("method:", r.method)
```

```
00 484 0.484
11 516 0.516
method: statevector:static
```

`print(r)` prints the counts in one line, `{"00": 484, "11": 516}`, and `r.to_json()` adds the shots and the method.

**Python.**

```python
import qumo

bell = qumo.Circuit(2).h(0).cx(0, 1).measure_all()
counts = qumo.sample(bell, shots=1000, seed=7)
print(counts)                       # a small histogram
print(dict(counts), counts.method)  # Counts is a dict
```

```
1000 shots, 2 outcome(s), statevector:static
  11       516  0.5160  #####################
  00       484  0.4840  ###################
{'11': 516, '00': 484} statevector:static
```

Gate calls chain, and the names and argument order are the same as in Mojo. `qumo.load("examples/foundations/bell.qasm")` loads the same circuit from the file.

## 2. Draw the circuit

**CLI.**

```bash
build/qumo draw examples/foundations/bell.qasm
build/qumo draw examples/foundations/bell.qasm --unicode
```

```
q0: --H--@--M>c0--
         |
q1: -----+--M>c1--

q0: ──H──●──M>c0──
         │
q1: ─────⊕──M>c1──
```

`@` is the control of the CX, `+` its target, and `M>c0` a measurement into classical bit 0. `build/qumo info examples/foundations/bell.qasm` prints depth, gate counts and which simulator QUMO recommends.

**Mojo.** Add one line to the step 1 script, after `c.measure_all()`:

```mojo
    print(c.draw())                       # or c.draw(style="unicode")
```

**Python.**

```python
import qumo

bell = qumo.Circuit(2).h(0).cx(0, 1).measure_all()
print(bell.draw(style="ascii"))       # print(bell) draws with Unicode characters
print(repr(bell))
```

```
q0: --H--@--M>c0--
         |
q1: -----+--M>c1--

<qumo.Circuit 2 qubits, 2 clbits, 2 gates, depth 2>
```

In Jupyter, a circuit as the last expression of a cell shows its drawing.

## 3. Look at the state vector

Sampling shows counts. The state vector shows the exact amplitudes before measurement.

**CLI.**

```bash
build/qumo statevector examples/foundations/bell.qasm
```

```
2 qubit(s), 2 nonzero amplitude(s), norm 1.000000000000 (measurements/resets ignored)
  |00>  +0.7071067812 +0.0000000000i   p=0.500000
  |11>  +0.7071067812 +0.0000000000i   p=0.500000
```

**Mojo.**

```mojo
from qumo.ir.circuit import Circuit
from qumo.sim.runner import evolve, bitstring


def main() raises:
    var c = Circuit(2)
    c.h(0)
    c.cx(0, 1)
    var sv = evolve(c)          # exact state, no measurement
    print(sv)                   # the nonzero amplitudes, by bitstring
    for i in range(1 << c.num_qubits):
        print("|" + bitstring(i, 2) + ">", sv.re[i], sv.im[i])
```

```
StateVector(2 qubits)
  |00>: 0.7071067811865476 + 0.0j
  |11>: 0.7071067811865476 + 0.0j
|00> 0.7071067811865476 0.0
|01> 0.0 0.0
|10> 0.0 0.0
|11> 0.7071067811865476 0.0
```

`sv.re` and `sv.im` hold the real and imaginary parts. A state vector of n qubits takes 16 x 2^n bytes: 16 GB at 30 qubits. (`statevector(c)` from `qumo.prelude` is the same function as `evolve(c)`.)

**Python.**

```python
import qumo

c = qumo.Circuit(2).h(0).cx(0, 1)
psi = qumo.statevector(c)             # NumPy array, complex128; index i has qubit q in bit q
print(psi)
print(qumo.probabilities(c))
```

```
[0.70710678+0.j 0.        +0.j 0.        +0.j 0.70710678+0.j]
[0.5 0.  0.  0.5]
```

## 4. Parameters and expectation values

Variational algorithms use circuits with free angles and read out the average of an observable. An observable in QUMO is a sum of Pauli strings written as text: `"Z0"`, `"Z0 Z1"`, `"0.5*Z0 Z1 - X0"`.

**CLI.** Save this as `ry.qasm`. `input float theta;` declares the free angle.

```
OPENQASM 3.0;
include "stdgates.inc";
input float theta;
qubit[1] q;
bit[1] c;
ry(theta) q[0];
c = measure q;
```

```bash
build/qumo expect ry.qasm "Z0" --params '{"theta": 1.0}'
build/qumo run ry.qasm --params '{"theta": 1.0}' --shots 1000 --seed 7
```

```
<Z0> = 0.5403023058681398
ry.qasm: 1 qubits, 1000 shots, seed 7, statevector:static, 0.437 ms
  0          772  0.7720  ###############################
  1          228  0.2280  #########
```

`<Z0>` is exact: cos(1.0) = 0.5403. The sampled estimate from the counts is (772 - 228) / 1000 = 0.544. `--params '[1.0]'` (values in declaration order) also works. Without `--params` you get error `QM-E101` with a hint that says what to pass. Several observables in one call: `build/qumo expect ry.qasm --observable "Z0" --observable "X0" --params '[1.0]'`.

**Mojo.**

```mojo
from qumo.ir.circuit import Circuit
from qumo.ir.pauli import parse_pauli_sum
from qumo.sim.runner import expectation, sample


def main() raises:
    var c = Circuit(1)
    var theta = c.parameter("theta")   # symbolic angle
    c.ry(theta, 0)
    var z = parse_pauli_sum("Z0")
    var angles: List[Float64] = [0.0, 0.5, 1.0, 1.5707963267948966, 3.141592653589793]
    for a in angles:
        print("theta =", a, " <Z> =", expectation(c, z, [a]))   # cos(theta)
    var bound = c.bind([1.0])          # a new circuit with theta = 1.0
    bound.measure_all()
    print(sample(bound, 1000, seed=7).counts)
```

```
theta = 0.0  <Z> = 1.0
theta = 0.5  <Z> = 0.8775825618903726
theta = 1.0  <Z> = 0.5403023058681398
theta = 1.5707963267948966  <Z> = 2.220446049250313e-16
theta = 3.141592653589793  <Z> = -1.0
{1: 228, 0: 772}
```

**Python.**

```python
import math
import qumo

c = qumo.Circuit(1).ry("theta", 0)    # a string is a free parameter
for a in [0.0, 0.5, 1.0, math.pi / 2, math.pi]:
    print("theta =", a, " <Z> =", qumo.expectation(c, "Z0", params={"theta": a}))
bound = c.bind(theta=1.0).measure_all()
print(dict(qumo.sample(bound, shots=1000, seed=7)))
```

```
theta = 0.0  <Z> = 1.0
theta = 0.5  <Z> = 0.8775825618903726
theta = 1.0  <Z> = 0.5403023058681398
theta = 1.5707963267948966  <Z> = 2.220446049250313e-16
theta = 3.141592653589793  <Z> = -1.0
{'1': 228, '0': 772}
```

Gradients (parameter shift and adjoint) are in [ALGORITHMS.md](ALGORITHMS.md).

## 5. Add noise

Real devices make errors. QUMO puts noise in the circuit as extra instructions. In OpenQASM they are written as `#pragma qumo ...` lines, which other tools skip.

**CLI.** [examples/noise_and_mitigation/noisy_bell.qasm](../examples/noise_and_mitigation/noisy_bell.qasm) is the Bell pair with depolarizing noise after each gate:

```
h q[0];
#pragma qumo depolarize1(0.02) q[0]
cx q[0], q[1];
#pragma qumo depolarize2(0.05) q[0], q[1]
```

```bash
build/qumo run examples/noise_and_mitigation/noisy_bell.qasm --shots 1000 --seed 7
```

```
examples/noise_and_mitigation/noisy_bell.qasm: 2 qubits, 1000 shots, seed 7, stabilizer:pauli_frame, 1.884 ms
  00         515  0.5150  #####################
  01          19  0.0190  #
  10          22  0.0220  #
  11         444  0.4440  ##################
```

`01` and `10` now appear: about 4% of the shots are wrong. The CLI chose a different simulator, `stabilizer:pauli_frame`, because this circuit has only Clifford gates and Pauli noise, which is much cheaper to sample than a state vector.

**Mojo.**

```mojo
from qumo.ir.circuit import Circuit
from qumo.ir.kraus import amplitude_damping
from qumo.sim.runner import sample


def main() raises:
    var c = Circuit(2)
    c.h(0)
    c.depolarize1(0.02, 0)      # 2% depolarizing noise after the H
    c.cx(0, 1)
    c.depolarize2(0.05, 0, 1)   # 5% two-qubit depolarizing noise after the CX
    c.measure_all()
    var r = sample(c, 1000, seed=7)
    print(r.counts, r.method)

    var d = Circuit(1)
    d.x(0)                                       # prepare |1>
    d.apply_channel(amplitude_damping(0.3), 0)   # decays to |0> with probability 0.3
    d.measure_all()
    var rd = sample(d, 1000, seed=7)
    print(rd.counts, rd.method)
```

```
{00: 496, 11: 475, 10: 15, 01: 14} statevector:trajectories
{1: 708, 0: 292} statevector:trajectories
```

The counts differ from the CLI run because `sample` always uses the state vector (one noisy trajectory per shot), and the CLI picks the simulator for you. Both are correct samples of the same distribution. To let Mojo choose as the CLI does, use `sample_auto` from `qumo.exec.dispatch` with the same arguments.

The second circuit uses a channel that is not a Pauli error (amplitude damping, the T1 decay of a qubit). Every noise instruction and the pragma syntax are listed in [SIMULATION.md](SIMULATION.md#noise-channels-and-the-pragma-syntax); how to define a channel of your own is in [EXTENDING.md](EXTENDING.md#2-a-custom-noise-channel).

**Python.**

```python
import qumo

c = qumo.Circuit(2)
c.h(0).depolarize1(0.02, 0)           # 2% depolarizing noise after the H
c.cx(0, 1).depolarize2(0.05, 0, 1)    # 5% two-qubit depolarizing noise after the CX
c.measure_all()
counts = qumo.sample(c, shots=1000, seed=7)
print(dict(counts), counts.method)
print(qumo.expectation(c, "Z0 Z1"))   # exact, on the density matrix
```

```
{'00': 515, '11': 444, '10': 22, '01': 19} stabilizer:pauli_frame
0.9466666666666664
```

`qumo.sample` picks the simulator as the CLI does, so these are the CLI's counts. The exact `<Z0 Z1>` is below 1 because of the noise; an ideal Bell pair gives 1.

## 6. Compile for a device

A device has a fixed set of gates and only some pairs of qubits are connected. Compiling ("transpiling") rewrites your circuit to fit. In QUMO a device is a `Target`, and a target is plain data that you can write as one line of text:

| Spec | Meaning |
|---|---|
| `line:3/sc_cz` | 3 qubits in a line; gates rz, sx, x, cz |
| `grid:4x4/sc_cz` | 4 x 4 lattice, same gates |
| `heavy_hex:3/sc_ecr` | heavy-hex lattice with the ECR gate |
| `trapped_ion:8` | 8 ions, all connected; gates rz, rx, ry, rxx |
| `neutral_atom:3x3` | 3 x 3 atom array |
| `superconducting_grid:2x2` | 2 x 2 lattice with typical error rates filled in |

No vendor is a default. The full list is in [COMPILER.md](COMPILER.md).

The circuit below is a 3-qubit GHZ state. It has a CX between qubits 0 and 2, which are not neighbours on a line.

**Mojo.**

```mojo
from qumo.ir.circuit import Circuit
from qumo.ir.qasm import to_qasm
from qumo.compiler.target import parse_target
from qumo.compiler.pass_manager import transpile


def main() raises:
    var c = Circuit(3)
    c.h(0)
    c.cx(0, 1)
    c.cx(0, 2)                  # qubits 0 and 2 are not neighbours on a line
    c.measure_all()
    var target = parse_target("line:3/sc_cz")   # 3 qubits in a line; gates rz, sx, x, cz
    var r = transpile(c, target, 2)             # optimisation level 0..3
    print(r.summary())
    var out = r.circuit.copy()                  # the compiled circuit, on physical qubits
    print(out.draw())
    print(to_qasm(out))                         # OpenQASM 3 text of the compiled circuit
```

```
compiled 'circuit' for line_3 [rz,sx,x,cz]: gates 3 -> 15, 2q 2 -> 2, depth 3 -> 7, T 0 -> 0, 11 passes in 0.124 ms
q0: --RZ(pi/2)--SX---RZ(pi)---@--SX--RZ(pi/2)----M>c1----------
                              |
q1: --RZ(pi/2)--SX--RZ(pi/2)--@--@-----M>c0--------------------
                                 |
q2: --RZ(pi/2)--SX---RZ(pi)------@------SX-----RZ(pi/2)--M>c2--
```

(The OpenQASM text follows; it is left out here.) Read the result this way:

- The compiler needed no SWAP. It placed your qubit 0 on the middle physical qubit, q1, so both CX gates land on connected pairs. `r.initial_layout` records that placement, and the measurements still write the classical bits you asked for (`M>c0` is on q1).
- The two-qubit count stayed at 2. The gate count grew from 3 to 15 because H and CX were rewritten in rz, sx and cz.
- `r.table()` prints what each pass did.

**CLI.** Save the circuit as `ghz3.qasm`:

```
OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
bit[3] c;
h q[0];
cx q[0], q[1];
cx q[0], q[2];
c = measure q;
```

```bash
build/qumo transpile ghz3.qasm --target line:3/sc_cz
```

```
ghz3.qasm -> line_3 (3 qubits, coupling_map): level 2, seed 7, 1.305 ms
  pipeline: remove_identities,elide_swaps,pauli_simp,fixpoint(cancel,commute,fuse1q,max=4),route(sabre,layout=auto,trials=20,iterations=2,swap_trials=20),kak,basis,cancel,commute,fuse1q,check
                         before    after
  gates                       3       15
  two-qubit gates             2        2
  depth                       3        7
  swaps inserted                       0
  estimated fidelity            1.000000
  layout: initial [1, 0, 2], final [1, 0, 2]
  (-o out.qasm saves the compiled circuit, --qasm prints it, --json gives the per-pass report)
```

The numbers are those of the Mojo script. `--level 0` to `--level 3` chooses how hard the compiler tries, `-o compiled.qasm` writes the compiled circuit to a file that every other command can read, and `build/qumo transpile --help` lists the rest.

**Python.**

```python
import qumo

ghz = qumo.Circuit(3).h(0).cx(0, 1).cx(0, 2).measure_all()
r = qumo.transpile(ghz, "line:3/sc_cz", level=2)
print(r.circuit.count_ops(), r.initial_layout, r.final_layout)
print(r.metrics["gates"], r.metrics["two_qubit"], r.metrics["depth"])   # (before, after)

# the same device as explicit data
r2 = qumo.transpile(ghz, coupling_map=[(0, 1), (1, 2)], basis=["rz", "sx", "x", "cz"])
print(r2.circuit == r.circuit)
```

```
{'sx': 5, 'rz': 8, 'cz': 2, 'measure': 3} [1, 0, 2] [1, 0, 2]
(3, 15) (2, 2) (3, 7)
True
```

`r.initial_layout[l]` is the physical qubit of your qubit `l` at the start, `r.final_layout[l]` at the end. `r.circuit` is a circuit like any other: draw it, sample it, write it with `r.circuit.to_qasm()`.

### Run it on a noisy model of the device

`emulate` takes a compiled circuit and a target with error rates, adds the matching noise, and samples. Nothing touches a real device.

```mojo
from qumo.ir.circuit import Circuit
from qumo.compiler.target import parse_target
from qumo.compiler.pass_manager import transpile
from qumo.exec.emulate import emulate


def main() raises:
    var c = Circuit(3)
    c.h(0)
    c.cx(0, 1)
    c.cx(0, 2)
    c.measure_all()
    var target = parse_target("superconducting_grid:2x2")   # preset with typical error rates
    var r = transpile(c, target, 2)
    var noisy = emulate(r.circuit, target, 1000, seed=7)    # needs a circuit compiled for the target
    for key in noisy.sorted_keys():
        print(key, noisy.counts[key])
    print("method:", noisy.method)
```

```
000 507
001 6
010 13
011 8
100 11
101 13
110 6
111 436
method: stabilizer:pauli_frame
```

An ideal GHZ state gives only `000` and `111`. This preset has 0.1% error per one-qubit gate, 0.5% per two-qubit gate and 1% per readout, and about 6% of the shots come out wrong.

The same from the CLI, with the file of the previous step:

```bash
build/qumo emulate ghz3.qasm --target superconducting_grid:2x2 --shots 1000 --seed 7
```

```
ghz3.qasm on superconducting_grid_2x2: 4 qubits, 1000 shots, seed 7, stabilizer:pauli_frame, estimated fidelity 0.955827
  000        507  0.5070  ####################
  001          6  0.0060  
  010         13  0.0130  #
  011          8  0.0080  
  100         11  0.0110  
  101         13  0.0130  #
  110          6  0.0060  
  111        436  0.4360  #################
  note: the circuit did not fit the target as written, so it was compiled first (level 2). --no-compile forbids that; 'qumo transpile' shows what the compiler did.
```

The counts are those of the Mojo script. The first line also gives the estimated fidelity of the compiled circuit on this device, 0.956: the product of the success probabilities of its gates and readouts. `emulate` compiles for you when the circuit does not fit the target and says so; `--no-compile` turns that into an error. The Python package has no `emulate` function yet; from Python, write the noise in the circuit as in step 5.

To run on real hardware (IBM Quantum, Quantinuum), see [HARDWARE_PLUGINS.md](HARDWARE_PLUGINS.md). Submission is a dry run unless you pass `--yes`.

## 7. An algorithm: Grover search, then VQE

QUMO ships the textbook algorithms as functions that return circuits ([ALGORITHMS.md](ALGORITHMS.md)).

### Grover search

Grover's algorithm finds a marked item among N = 2^n with about sqrt(N) queries. Here n = 3 and the marked item is 5 (binary `101`).

```mojo
from qumo.algorithms.library import grover, grover_optimal_iterations
from qumo.sim.runner import sample


def main() raises:
    var marked: List[Int] = [5]                 # the item to find: 5 = binary 101
    var c = grover(3, marked, measure=True)     # 3 search qubits, optimal number of iterations
    print("iterations:", grover_optimal_iterations(3, 1), " gates:", len(c))
    var r = sample(c, 1000, seed=7)
    print("most frequent:", r.most_frequent(), " probability:", r.probability(r.most_frequent()))
```

```
iterations: 2  gates: 38
most frequent: 101  probability: 0.953
```

A random guess is right 1 time in 8. After two Grover iterations the answer comes out 95% of the time (the theory says 94.5%).

From the CLI, `build/qumo tool build_circuit '{"kind":"grover","args":{"marked":["101"]},"measure":true}'` returns the same circuit as OpenQASM inside a JSON object (`build/qumo tool build_circuit --help` lists the algorithms and their arguments). Saved to a file and run with `build/qumo run grover.qasm --shots 1000 --seed 7`, it gave `101` in 953 of 1000 shots.

### VQE

The variational quantum eigensolver looks for the lowest energy of a Hamiltonian: a classical optimiser tunes the angles of a circuit (the ansatz) to minimise an expectation value. Here the Hamiltonian is a two-qubit Ising model whose exact ground energy is -sqrt(2) = -1.41421356...

```mojo
from qumo.ir.pauli import parse_pauli_sum
from qumo.algorithms.library import hardware_efficient_ansatz
from qumo.algorithms.variational import vqe


def main() raises:
    var h = parse_pauli_sum("Z0 Z1 + 0.5*X0 + 0.5*X1")      # a 2-qubit Ising Hamiltonian
    var ansatz = hardware_efficient_ansatz(2, 2, "linear", "ry,rz")
    print("parameters:", ansatz.num_params())
    var r = vqe(ansatz, h, "bfgs")                          # adjoint gradients by default
    print("energy:", r.energy, " iterations:", r.nit, " circuit evaluations:", r.circuit_evals)
```

```
parameters: 12
energy: -1.414213562373095  iterations: 22  circuit evaluations: 44
```

`print(r.to_json())` gives the whole result (angles, parameter names, the energy after each iteration) as JSON.

**CLI.** `vqe` takes the Hamiltonian and builds the same hardware-efficient ansatz (2 layers, linear entanglement by default):

```bash
build/qumo vqe "Z0 Z1 + 0.5*X0 + 0.5*X1"
```

```
energy -1.414213562373095 (converged)
  ansatz     hardware_efficient(n=2, layers=2, linear), seed 7
  optimizer  bfgs, gradient adjoint
  cost       22 iterations, 44 circuit evaluations, 0.0001 s
  params     [-1.1014581880539795,-0.012659426601635852,-0.5659489073577528,0.08965618647478721,0.11821595416436555,0.10693905028881251,-0.9842885410886429,-0.08166120847874471,-0.5898849151254457,-0.04349615896006225,-0.5206691355758011,0.020344027166191863]
```

`--layers`, `--optimizer` and `--gradient` change the ansatz and the method; `--json` gives the result for a script.

**Python.**

```python
import qumo

h = "Z0 Z1 + 0.5*X0 + 0.5*X1"                          # or [("ZZ", 1.0), ("IX", 0.5), ("XI", 0.5)]
ansatz = qumo.hardware_efficient_ansatz(2, layers=2)
r = qumo.vqe(ansatz, h, method="bfgs")
print("parameters:", ansatz.num_parameters)
print("energy:", r["energy"], " iterations:", r["nit"], " exact:", qumo.ground_energy(h))
```

```
parameters: 12
energy: -1.414213562373095  iterations: 22  exact: -1.4142135623730951
```

A chemistry example with the UCCSD ansatz and four optimisers is [examples/chemistry_and_physics/vqe_h2.mojo](../examples/chemistry_and_physics/vqe_h2.mojo).

## 8. Where next

| You want to | Go to |
|---|---|
| Run more examples, by topic | [examples/README.md](../examples/README.md) ("Start here" lists five in order of difficulty) and [examples/CATALOG.md](../examples/CATALOG.md) |
| Use QUMO from Python: every function, NumPy, Jupyter, performance | [PYTHON.md](PYTHON.md) |
| Move code from Qiskit, Cirq or PennyLane | [FROM_QISKIT.md](FROM_QISKIT.md) |
| Loops, `while`, classical variables, mid-circuit measurement | [CONTROL_FLOW.md](CONTROL_FLOW.md) |
| Understand the simulators and their limits | [SIMULATION.md](SIMULATION.md), [GPU.md](GPU.md) |
| Control the compiler pass by pass | [COMPILER.md](COMPILER.md) |
| Error correction: surface codes, decoders | [QEC.md](QEC.md), [FT.md](FT.md) |
| Quantum machine learning | [QML.md](QML.md) |
| Add your own gate, noise channel, pass, decoder or target | [EXTENDING.md](EXTENDING.md) |
| Let an LLM agent drive QUMO | [AI_NATIVE.md](AI_NATIVE.md) |
| Look up an error code such as `QM-E101` | [ERRORS.md](ERRORS.md), or `build/qumo describe error QM-E101` |
| Write Mojo that compiles on the pinned toolchain | [MOJO_GUIDE.md](MOJO_GUIDE.md) |
| Cite QUMO or report a benchmark | [CITATION.cff](../CITATION.cff), [CONTRIBUTING.md](../CONTRIBUTING.md) |
