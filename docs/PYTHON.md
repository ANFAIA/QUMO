# QUMO from Python

`qumo` is a small pure-Python package over the compiled extension `qumo_py`
(Mojo, `python/qumo_py.mojo`). You write circuits and call functions with
keyword arguments; the simulators, the compiler, the optimisers and the QEC
pipeline run natively.

```python
import qumo

bell = qumo.Circuit(2).h(0).cx(0, 1).measure_all()
print(bell.draw())
counts = qumo.sample(bell, shots=1000, seed=7)      # {'11': 516, '00': 484}
print(counts)                                       # a small histogram
```

## Install

```bash
make setup            # once: the Mojo toolchain in .venv
make py               # build/qumo_py.so (about a minute the first time)
pip install -e .      # or: uv pip install -e .
python -c "import qumo; print(qumo.__version__)"
```

`pyproject.toml` is at the repository root. The install is editable because
the package does not contain the extension: `qumo_py.so` is built on your
machine and links the Modular runtime of the checkout's `.venv`, so `qumo`
loads it from `<checkout>/build/` when it is imported. No `sys.path` or
`PYTHONPATH` setting is needed. Rebuilding with `make py` is picked up by the
next Python process.

Where `qumo` looks for the extension, in order: a `qumo_py` module the process
already imported; `$QUMO_PY_PATH` (the `.so`, or a directory holding
`qumo_py.so`); next to the package; `<checkout>/build/qumo_py.so`. If none
exists you get `qumo.QumoNativeMissing` (an `ImportError`) that lists the
paths tried and tells you to run `make py`.

Without installing, two things work. In a `python` started at the repository
root, and in a script saved there, `import qumo` works as soon as `make py`
has run: Python finds the Mojo source directory `./qumo` first there, and
`qumo/__init__.py` redirects it to this package.
From any other directory, use `PYTHONPATH=/path/to/QUMO/python`. Installing is
still the recommended way: it works from every directory and in Jupyter.

`make install-py` runs the `uv pip install -e .` step into the Makefile's
`$(PY)` (default `bench/.venv/bin/python`, created by `make setup-py`).

NumPy is the only dependency.

## Quick start

```python
import math
import qumo

# Build. Angles first, qubits after, as in the Mojo API. Calls chain.
c = qumo.Circuit(3)
c.h(0).cx(0, 1).rz(0.25, 2).ccx(0, 1, 2)
c.measure_all()

# Inspect.
c.num_qubits, c.depth(), c.count_ops()       # 3, 3, {'h': 1, 'cx': 1, 'rz': 1, 'ccx': 1, 'measure': 3}
print(c.draw())                              # text drawing; c.draw(style="ascii")
c.to_qasm()                                  # OpenQASM 3 (to_qasm(2) for OpenQASM 2)

# Load: a path or OpenQASM 2/3 text.
c = qumo.load("examples/foundations/bell.qasm")
c = qumo.load("OPENQASM 3.0; qubit[2] q; bit[2] c; h q[0]; cx q[0], q[1]; c = measure q;")

# Sample. Defaults are the command line's: shots=1024, seed=1234.
counts = qumo.sample(c)
counts.method, counts.most_common(1), counts.probabilities()

# 40 qubits: Clifford circuits go to the stabilizer simulator on their own.
qumo.sample(qumo.ghz(40, measure=True), shots=100).method      # 'stabilizer:tableau'

# State vector and probabilities as NumPy arrays.
psi = qumo.statevector(qumo.Circuit(2).h(0).cx(0, 1))          # complex128, shape (4,)
p = qumo.probabilities(qumo.Circuit(2).h(0).cx(0, 1))          # float64

# Expectation values. A noisy circuit is evaluated on the density matrix.
qumo.expectation(qumo.ghz(3), "Z0 Z1")                         # 1.0
noisy = qumo.Circuit(2).h(0).cx(0, 1).depolarize2(0.05, 0, 1)
qumo.expectation(noisy, [("ZZ", 1.0)])                         # 0.9467

# Parameters by name.
ansatz = qumo.Circuit(1).ry("theta", 0).measure_all()
qumo.sample(ansatz, params={"theta": math.pi})                 # {'1': 1024}
bound = ansatz.bind({"theta": 0.3})                            # or ansatz.bind(theta=0.3)

# Compile for a target, with the same target strings as Mojo and the CLI.
r = qumo.transpile(c, "grid:2x2/sc_cz", level=2)
r.circuit, r.initial_layout, r.final_layout, r.metrics
r = qumo.transpile(c, coupling_map=[(0, 1), (1, 2)], basis=["rz", "sx", "x", "cz"])

# Research drivers.
vqe = qumo.vqe(qumo.h2_ansatz(), qumo.h2_sto3g())              # vqe["energy"] = -1.13731
qaoa = qumo.qaoa_maxcut([(0, 1), (1, 2), (2, 3), (3, 0)], p=2)
mem = qumo.memory_experiment("surface_z", distance=3, p=0.001, shots=100_000)
```

Every public function has a docstring with a runnable example:
`help(qumo.sample)`. The examples are executed by the test suite.

## Circuits

`qumo.Circuit(num_qubits)` is a builder. Gate methods return the circuit.

| Group | Methods |
|---|---|
| One qubit | `id x y z h s sdg t tdg sx sxdg` `(q)` |
| Rotations | `rx ry rz p` `(angle, q)`, `u(theta, phi, lam, q)` |
| Two qubits | `cx cy cz ch swap iswap ecr` `(a, b)` |
| Two-qubit rotations | `cp crx cry crz rxx ryy rzz` `(angle, a, b)` |
| Three qubits | `ccx cswap ccz` `(a, b, c)` |
| Non-unitary | `measure(q, c=None)`, `measure_all()`, `reset(q)`, `barrier()`, `c_if(clbit, value=1)` |
| Noise channels | `depolarize1(p, q)`, `depolarize2(p, a, b)`, `x_error y_error z_error (p, q)`, `pauli_channel_1(px, py, pz, q)` |
| General channels | `kraus1(name, matrices, q)`, `kraus2(name, matrices, q0, q1)`: Kraus operators (complex matrices, sum K^dagger K = I); written as `#pragma qumo kraus1 "name" (re, im, ...)` in QASM |
| Parameters | `parameter(name)`, `parameters(prefix, count)`, `bind(values)` |
| Inspection | `num_qubits`, `num_clbits`, `parameter_names`, `num_parameters` (properties); `len(c)`, `size()`, `depth()`, `depth_of(name)`, `count_ops()`, `count_gate(name)`, `num_2q_gates()`, `t_count()`, `two_qubit_edges()`, `has_noise()`, `has_measurements()`, `is_clifford()`, `is_dynamic()` |
| Transformations | `copy()`, `compose(other)`, `inverse()`, `bind(values)` (each returns a new circuit) |
| Text | `draw(width=80, style="unicode")`, `to_qasm(version=3)`, `to_qasm2()`, `to_qasm3()`, `to_stim()`, `text()` |

The names and argument order are those of the Mojo `Circuit`
(`qumo/ir/circuit.mojo`), so a circuit reads the same in both languages. A
test checks that every simple gate method of the Mojo builder exists in
Python.

**Angles** are numbers, `qumo.Parameter` objects, affine expressions of one
parameter (`2 * theta + 0.5`, `-theta / 2`), or parameter names as strings
(`c.rx("theta", 0)`). Parameters are matched by name. `parameter_names` is the
order positional values are read in.

**Classical bits.** `measure(q)` writes bit `q`; `measure(q, c)` writes bit
`c`; `measure_all()` appends `num_qubits` fresh bits. In result bitstrings bit
0 is the rightmost character. `c_if(clbit, value)` conditions the instruction
added just before it: `c.x(1).c_if(0, 1)`.

**Python protocol.** `repr(c)` is a one-line summary, `str(c)` the drawing.
In Jupyter a circuit displays as the summary and the drawing (text and HTML;
circuits past 400 instructions show the summary only). `==` compares
structure (same instructions in the same order), not unitary equivalence.
`copy.copy`, `copy.deepcopy` and `pickle` work; a pickle stores OpenQASM 3
text, so it survives a rebuild of the extension and can cross to worker
processes (`multiprocessing`, `joblib`, Dask).

**Loading.** `qumo.load(source)` (alias `qumo.load_qasm`) takes a
`pathlib.Path`, a path string, or OpenQASM 2 / 3 text. A string with a newline
or a `;` is text, anything else is a path; a missing file raises
`FileNotFoundError` that says so. `qumo.from_stim(text)` reads Stim circuits.

**Low level.** `c.native` is the `qumo_py.Circuit` behind a circuit and
`qumo.Circuit.from_native(obj)` wraps one; `qumo.native` is the `qumo_py`
module (hardware clients, programs with run-time control flow, Benchpress
helpers). `help(qumo.native.sample)` works: every function and method of the
extension has a docstring whose first line is its signature.

## Functions

The defaults of `sample`, `run` and `transpile` are those of the command line.

### Simulation

| Function | Returns |
|---|---|
| `sample(circuit, shots=1024, seed=1234, *, params=None, device="auto", precision="auto", method="auto")` | `Counts`: a `dict` `{bitstring: count}` with `.method`, `.shots`, `.seed`, `.note`, `.probabilities()`, `.most_common(n)` |
| `run(circuit, shots=1024, seed=1234, **options)` | the same (alias, the CLI's name) |
| `statevector(circuit, params=None)` | `numpy.ndarray`, `complex128`, `2**n` amplitudes |
| `probabilities(circuit, params=None)` | `numpy.ndarray`, `float64`, `2**n` values |
| `expectation(circuit, observable, params=None, *, noise="auto", device="cpu")` | `float` |
| `value_and_gradient(circuit, observable, params, *, device="cpu")` | `(float, ndarray)` |
| `batch_value_and_gradient(circuit, observable, param_sets)` | `(ndarray (S,), ndarray (S, P))` |
| `emulate(circuit, target, shots=1024, seed=1234, *, compile=None, level=2)` | `Emulation`: `.counts` (a `Counts`), `.method`, `.compiled`, `.estimated_fidelity`, `.target`, `.num_qubits` |

* `sample` dispatches as `qumo run` does: stabilizer tableau or Pauli frames
  for Clifford circuits (hundreds of qubits), state vector otherwise (32
  qubits at most), GPU when the build has it (`make py-gpu`) and the circuit
  is large enough. `method="statevector"`, `"ptsbe"` or `"branching"` forces
  one simulator. `device` is `"auto"`, `"cpu"` or `"gpu"`.
* Index `i` of a state vector has qubit `q` in bit `q` (qubit 0 is the least
  significant bit). Measurements and resets are skipped. A circuit with noise
  channels has no state vector: `statevector` raises QM-E414 and points at
  `probabilities` / `expectation`.
* An **observable** is a Pauli-sum string (`"Z0 Z1 + 0.5 * X0"`, or dense
  labels `"ZZ + 0.5*IX"`), or a list of `(label, coeff)` pairs
  (`[("ZZ", 1.0), ("IX", 0.5)]`; a dict works too). A dense label reads left
  to right from the highest qubit, as in Qiskit. Complex coefficients, as
  `SparsePauliOp.to_list()` gives them, are accepted when their imaginary part
  is zero (`[("ZZ", 1+0j)]`); a non-zero imaginary part raises QM-E913.
* `expectation` with `noise="auto"` evaluates a circuit that holds noise
  channels as `Tr(rho H)` on the density matrix (15 qubits at most; past that
  it raises QM-E410 instead of returning the ideal value). `noise="ignore"`
  gives the ideal value, `noise="error"` raises if there is noise.

* `emulate` is `qumo emulate`: the circuit is sampled under the noise of a
  device model built from the target spec (per-qubit T1/T2, gate and readout
  errors, per-edge two-qubit errors), with no device. A circuit that does not
  fit the target is compiled first (`compiled` says so); `compile=False`
  refuses it with QM-E615 instead, `compile=True` always compiles. The same
  seed gives the same counts as the command line.

### Compilation

| Function | Returns |
|---|---|
| `transpile(circuit, target=None, *, coupling_map=None, basis=None, num_qubits=None, level=2, seed=7)` | `TranspileResult` |
| `target_info(spec)` | `dict`: `name`, `modality`, `num_qubits`, `all_to_all`, `edges`, `basis` |

* `target` is the one-line spec of `parse_target` in Mojo and of
  `qumo compile --target`: `"<topology>:<size>[/<basis>]"`, for example
  `"grid:2x2/sc_cz"`, `"line:20"`, `"heavy_hex:5/rz,sx,x,ecr"`,
  `"trapped_ion:11"`. Topologies are in `qumo.TOPOLOGIES`.
* Or give explicit data: `coupling_map=[(0, 1), (1, 2)]` and
  `basis="sc_cz"` (a name from `qumo.BASIS_SETS`) or
  `basis=["rz", "sx", "x", "cz"]`. With neither, the target is all-to-all with
  any gate.
* `TranspileResult` has `.circuit`, `.initial_layout`, `.final_layout`
  (`layout[l]` = physical qubit of logical qubit `l`), `.metrics`
  (`gates`, `two_qubit`, `depth`, `t_count` as `(before, after)`,
  `est_fidelity`, `compile_ms`) and `.report` (the per-pass report). It
  unpacks as `circuit, initial, final = qumo.transpile(...)`.

### Research drivers

| Function | Returns |
|---|---|
| `vqe(ansatz, hamiltonian, *, method="bfgs", x0=None, maxiter=200, tol=1e-8, grad="adjoint", seed=7)` | `dict`: `energy`, `params` (ndarray), `parameters` (`{name: value}`), `history` (ndarray), `nit`, `nfev`, `ngrad`, `converged`, `wall_time_s`, ... |
| `qaoa_maxcut(edges, *, num_nodes=None, p=1, method="bfgs", restarts=4, shots=1024, seed=7, maxiter=200)` | `dict`: `expected_cut`, `max_cut`, `approx_ratio`, `gammas`, `betas` (ndarrays), `best_bitstring`, `best_cut`, `prob_optimal`, ... |
| `ground_energy(hamiltonian, num_qubits=None)` | `float` (exact, Lanczos) |
| `h2_sto3g(num_qubits=4, include_nuclear_repulsion=True)` | `[(label, coeff), ...]` |
| `transverse_field_ising(n, j=1.0, h=1.0, periodic=False)` | `[(label, coeff), ...]` |
| `heisenberg_xxz(n, j=1.0, delta=1.0, periodic=False, field=0.0)` | `[(label, coeff), ...]` |
| `fermi_hubbard(edges, *, num_sites=None, t=1.0, u=4.0, mu=0.0)` | `[(label, coeff), ...]` |
| `maxcut_hamiltonian(edges, *, num_nodes=None, include_offset=True)` | `[(label, coeff), ...]` |
| `ghz(n, measure=False)`, `qft(n, do_swaps=True)` | `Circuit` |
| `hardware_efficient_ansatz(n, layers=1, entanglement="linear", rotations="ry,rz")` | parametric `Circuit` |
| `qaoa_maxcut_ansatz(edges, p=1)`, `uccsd_ansatz(num_qubits, num_electrons)`, `h2_ansatz()` | parametric `Circuit` |
| `trotter(hamiltonian, time, steps=1, order=1, num_qubits=None)` | `Circuit` |

Optimiser names are in `qumo.OPTIMIZERS`; `grad` is `"adjoint"`,
`"parameter-shift"` or `"finite-diff"`. Hamiltonians are plain lists, so
`qiskit.quantum_info.SparsePauliOp.from_list(qumo.h2_sto3g())` works as is.

### Error correction

| Function | Returns |
|---|---|
| `memory_experiment(code="surface_z", distance=3, rounds=None, p=0.001, shots=100000, *, decoder="mwpm", seed=1, max_errors=None)` | `dict`: `shots`, `errors`, `logical_error_rate`, `std_error`, `wilson`, `per_round_error_rate`, `seconds`, `shots_per_second`, ... |
| `threshold_sweep(code, distances, ps, shots=100000, *, decoder="mwpm", seed=1, max_errors=None, rounds=None)` | `dict` of ndarrays of shape `(len(distances), len(ps))`: `logical_error_rate`, `std_error`, `wilson_lo`, `wilson_hi`, `errors`, `shots`, ... |
| `QecMemory(code, distance, rounds, p)` | streaming object: `.sample(shots, seed)` -> `(detectors, observables)`, `.decode(detectors)`, `.count_logical_errors(obs, pred)`, `.dem()`, `.circuit()` |

Codes are in `qumo.CODES`, decoders in `qumo.DECODERS`. `QecMemory` moves
detection events as bit-packed `uint8` arrays, one row per shot;
`numpy.unpackbits(dets, axis=1, bitorder="little")` expands them.

## Errors

Failures raise `qumo.QumoError`:

```python
try:
    qumo.sample(qumo.Circuit(2).h(0))
except qumo.QumoError as e:
    e.code      # 'QM-E402'
    e.message   # 'circuit has no measurements.'
    e.hint      # 'call circuit.measure_all() (or circuit.measure(q, c)) before sampling, or use ...'
    e.docs      # 'docs/ERRORS.md#qm-e402'
```

* `.code` is the QUMO code (`QM-Exxx`), the same one the CLI and the Mojo API
  report; [ERRORS.md](ERRORS.md) has an entry per code. `.hint` is written
  for Python: it names `qumo` functions and `Circuit` methods.
* A wrong argument type raises `qumo.QumoTypeError`, which is both a
  `QumoError` (QM-E911) and a `TypeError`: `qumo.sample(c, shots="100")` and
  `shots=100.5` are refused; they are not converted.
* Builder mistakes are reported at the call that made them: `c.h(5)` on a
  2-qubit circuit raises QM-E001 on that line, `c.cx(0, 0)` QM-E002,
  `c.rx(0, 0.5)` (angle and qubit swapped) a `TypeError`. A failed call
  leaves the circuit as it was.
* OpenQASM errors (QM-E5xx) keep the source excerpt with the caret in
  `.message`.
* A missing or stale extension raises `qumo.QumoNativeMissing` (QM-E950, an
  `ImportError`).

Calls made on `qumo.native` directly raise a bare `Exception` whose text
starts with the code; `qumo.errors.from_native(exc)` converts one.

## Performance

What stays in Mojo: gate application (fusion, cache blocking, SIMD,
threads), sampling, simulator dispatch, the compiler passes, the optimiser
loops and gradients of `vqe` / `qaoa_maxcut`, QEC sampling and decoding.
Python holds handles and receives results.

**Building is bulk.** A gate method checks its arguments and appends five
integers to a C array. The native circuit is extended in one call
(`Circuit.extend_packed`, which reads the array's memory) the next time
something needs it, and only with the gates added since the last time. After
that the handle is reused: sampling the same circuit again sends nothing.
There is no Python work per gate on the simulation path.

Measured on an Apple M-series laptop (Python 3.12, `python/tests/test_overhead.py`
prints these):

| What | Time |
|---|---|
| Build in Python, per gate | 0.3 to 0.4 us |
| Send 24 770 gates to the extension | 1.3 to 1.9 ms (one call) |
| `qumo.sample(bell, 1024)` vs the raw `qumo_py.sample_device` call | 30.1 us vs 29.4 us: 0.6 to 2 us per call |
| `qumo.sample` on 20 qubits, 460 gates, 1024 shots vs the raw call | 24 to 34 ms both: the difference is below timing noise; the package's own cost is 0.002 % of the run |
| `qumo.statevector(ghz(22))` (4 194 304 amplitudes, simulation + copy) | 18 ms |

**State vectors** are written once from the simulator's buffers into a NumPy
array, in parallel from 2^16 amplitudes; no Python list is built. The
simulator keeps real and imaginary parts in separate arrays, so this one copy
is needed: peak memory is twice the state (32 bytes per amplitude) for the
duration of the call.

**Counts** are filled by the extension directly into the `Counts` dict.

**Sweeps: one call, not a loop.** A Python loop pays one call per point, which
is fine for large circuits and slow for many small ones. Prefer:

```python
import numpy as np

# energies and gradients for many parameter sets: one native call, parallel up to 14 qubits
values, grads = qumo.batch_value_and_gradient(ansatz, hamiltonian, np.random.rand(1000, ansatz.num_parameters))

# a whole optimisation
result = qumo.vqe(ansatz, hamiltonian, method="bfgs")

# a threshold plot: every (distance, p) point sampled and decoded natively on all cores
sweep = qumo.threshold_sweep("surface_z", [3, 5, 7], np.logspace(-3, -2, 6), shots=200_000)
sweep["logical_error_rate"]          # ndarray (3, 6)

# sampling at many parameter values: params= binds inside the native call, no circuit copy
for theta in thetas:
    counts = qumo.sample(circuit, 1024, params=[theta])
```

For independent circuits in separate processes, circuits pickle (as OpenQASM
3), so `multiprocessing.Pool.map(qumo.sample, circuits)` works; each worker
loads the extension once. The simulators already use every core for circuits
past 14 qubits, so process pools pay off for many small circuits only.

**GPU.** `make py-gpu` builds `build/gpu/qumo_py.so`; point `QUMO_PY_PATH` at
it. `qumo.gpu_info()` reports what the build and the machine offer, and
`device="gpu"` falls back to the CPU with the reason in `counts.note`.

## Jupyter

```python
import qumo
c = qumo.Circuit(3).h(0).cx(0, 1).cx(1, 2).measure_all()
c                       # summary line + drawing
qumo.sample(c)          # histogram: outcome, count, fraction, bar
```

* A circuit as the last expression of a cell shows its drawing; `print(c)`
  does the same anywhere. `c.draw(width=0)` does not fold wide circuits.
* `Counts` prints as a text histogram and is a `dict`, so
  `pandas.Series(counts)` and `matplotlib.pyplot.bar(counts.keys(), counts.values())`
  take it as is.
* Arrays from `statevector`, `probabilities`, `threshold_sweep` are ordinary
  NumPy arrays.
* After `make py`, restart the kernel: a process keeps the extension it
  loaded.
* Install the package into the kernel's environment (`pip install -e .` with
  that environment's `pip`), or start Jupyter with `PYTHONPATH=python`.

## Tests

```bash
make test-python      # = PYTHONPATH=python bench/.venv/bin/python -m pytest -q python/tests
```

`make test-python PY=/path/to/python` uses another interpreter; it needs
NumPy and pytest.

They cover the builder, loading, pickling, every function above, the
docstring examples, the argument checks of the extension, and the overhead
bounds.
