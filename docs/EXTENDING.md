# Extending QUMO

How to add your own gate, noise channel, compiler pass, decoder or device target. Each section says first whether you can do it from your own files or whether it means editing QUMO's source.

| What | From your own code | By name everywhere (CLI, MCP, text pipelines) |
|---|---|---|
| Gate | yes: an OpenQASM `gate` definition, or a Mojo function that returns a `Circuit` | not needed for a composite gate; a new primitive opcode means editing core files |
| Noise channel | yes: `KrausChannel` in Mojo, `#pragma qumo kraus1` / `kraus2` in OpenQASM | not needed |
| Compiler pass | yes: a Mojo function `Circuit -> Circuit`, called between built-in pipelines | no: the pass names are a closed table in `qumo/compiler/pass_manager.mojo` |
| Decoder | yes: a struct with the `Decoder` trait, run with `sample_and_decode` | no: the decoder names are a closed table in `qumo/qec/experiments.mojo` |
| Target | yes: targets are data (edges, basis, error rates) | yes for data; a new one-line topology keyword means editing `qumo/compiler/target.mojo` |

There is no plugin loader: QUMO does not discover passes or decoders at run time. Mojo compiles everything ahead of time, so "from your own code" means your Mojo file imports `qumo` and is compiled with `-I .`.

Every snippet on this page was run on this repository (Mojo 1.2.0). Run Mojo snippets from the repository root with `.venv/bin/mojo run -I . my.mojo`; the first line Mojo prints (`Failed to initialize Crashpad...`) is harmless.

## 1. A custom gate

### In OpenQASM

A `gate` definition builds a new gate from existing ones. It can take angle parameters. QUMO expands it when it reads the file.

```
OPENQASM 3.0;
include "stdgates.inc";

// A gate of your own, built from standard gates. theta is its parameter.
gate entangle(theta) a, b {
  ry(theta) a;
  cx a, b;
}

qubit[3] q;
bit[3] c;

entangle(pi/2) q[0], q[1];
cx q[1], q[2];
c = measure q;
```

```bash
build/qumo run custom_gate.qasm --shots 1000 --seed 7
build/qumo draw custom_gate.qasm
```

```
custom_gate.qasm: 3 qubits, 1000 shots, seed 7, statevector:static, 0.809 ms
  000        484  0.4840  ###################
  111        516  0.5160  #####################
q0: --RY(pi/2)--@--M>c0--------
                |
q1: ------------+---@----M>c1--
                    |
q2: ----------------+----M>c2--
```

The drawing shows the expanded gates: QUMO's circuit is flat, so a user gate does not survive as a named box. The OpenQASM 3 modifiers work on user gates: `ctrl @ entangle(pi) q[0], q[1], q[2];`, `inv @ entangle(pi/2) q[1], q[2];` and `pow(2) @ entangle(pi/2) q[1], q[2];` were all run. Gate definitions may call other user gates (up to 64 levels deep); recursion and subroutines (`def`) are not supported.

### In Mojo

A composite gate is a function that returns a small `Circuit`. `compose` places it; `inverse()`, `power(k)` and `controlled(q)` derive new gates from it.

```mojo
from std.math import pi
from qumo.ir.circuit import Circuit
from qumo.ir.param import Param
from qumo.sim.runner import sample


def entangle(theta: Param) -> Circuit:
    """A reusable two-qubit block: the Mojo counterpart of a QASM `gate`."""
    var g = Circuit(2, name="entangle")
    g.ry(theta, 0)
    g.cx(0, 1)
    return g^


def main() raises:
    var c = Circuit(3)
    c.compose(entangle(pi / 2))                 # on qubits 0, 1
    c.compose(entangle(0.0), qubit_offset=1)    # on qubits 1, 2 (theta = 0: just the CX)
    c.measure_all()
    print(c.draw())
    print(sample(c, 1000, seed=7).counts)

    var inv = entangle(pi / 2).inverse()        # derived gates come for free
    var ctl = entangle(pi / 2).controlled(2)    # control on a new qubit 2
    print(len(inv), "gates in the inverse;", ctl.num_qubits, "qubits in the controlled version")
```

```
q0: --RY(pi/2)--@--M>c0------------
                |
q1: ------------+--RY(0)--@--M>c1--
                          |
q2: ----------------------+--M>c2--

{111: 516, 000: 484}
2 gates in the inverse; 3 qubits in the controlled version
```

`compose_mapped(other, qmap)` places a block on arbitrary qubits.

### A new primitive gate (core edit)

A gate that the simulators and the compiler treat as one operation (its own opcode, like `rzz` or `ecr`) is not something you can add from outside. It starts in `qumo/ir/gates.mojo` (opcode, name, arity, matrix) and continues through the OpenQASM reader and writers, the simulators, the GPU kernels and the compiler: `grep -rl 'Op.RZZ' qumo` shows every file that had to know about one existing two-qubit gate (36 files today). Run `build/qumo describe gates` to see what exists first: most gates you need are already there, and an arbitrary one- or two-qubit unitary can always be written as a composite gate.

## 2. A custom noise channel

QUMO has Pauli channels built in (`x_error`, `depolarize1`, ...; the full list is in [SIMULATION.md](SIMULATION.md#noise-channels-and-the-pragma-syntax)). Any other channel on one or two qubits is a list of Kraus matrices K_k with sum_k K_k^dagger K_k = identity.

### In Mojo

```mojo
from std.math import sqrt
from qumo.ir.circuit import Circuit
from qumo.ir.kraus import KrausChannel
from qumo.ir.qasm import to_qasm
from qumo.sim.runner import sample


def my_damping(gamma: Float64) raises -> KrausChannel:
    var ch = KrausChannel("my_damping", 1)          # name, number of qubits (1 or 2)
    ch.add_real([1.0, 0.0, 0.0, sqrt(1.0 - gamma)]) # K0, row-major 2x2
    ch.add_real([0.0, sqrt(gamma), 0.0, 0.0])       # K1
    ch.validate()                                   # QM-E414 unless sum K^dag K = I
    return ch^


def main() raises:
    var c = Circuit(1)
    c.x(0)
    c.apply_channel(my_damping(0.3), 0)
    c.measure_all()
    var r = sample(c, 10000, seed=7)
    print(r.counts, r.method)
    print(to_qasm(c))
```

```
{1: 7056, 0: 2944} statevector:trajectories
OPENQASM 3.0;
include "stdgates.inc";
qubit[1] q;
bit[1] c;
x q[0];
#pragma qumo kraus1 "my_damping" (1, 0, 0, 0, 0, 0, 0.8366600265340756, 0, 0, 0, 0.5477225575051661, 0, 0, 0, 0, 0) q[0];
c[0] = measure q[0];
```

The qubit starts in |1> and decays to |0> with probability 0.3: 2944 of 10000 shots. Use `ch.add(re, im)` for complex matrices. `amplitude_damping`, `phase_damping` and `thermal_relaxation` in `qumo.ir.kraus` are ready-made.

### In OpenQASM

The same channel as a pragma, as `to_qasm` wrote it above:

```
#pragma qumo kraus1 "<name>" (<re>, <im>, <re>, <im>, ...) q[i];
#pragma qumo kraus2 "<name>" (<re>, <im>, ...) q[i], q[j];
```

- The numbers are (real, imaginary) pairs, row by row, one matrix after the other. `kraus1` takes 8 numbers per matrix (2 x 2), `kraus2` takes 32 (4 x 4).
- Constant expressions are allowed: `sqrt(0.7)`, `pi/4`.
- The name is free text; it appears in drawings and error messages.

```
OPENQASM 3.0;
include "stdgates.inc";

qubit[1] q;
bit[1] c;

x q[0];
// amplitude damping with gamma = 0.3: K0 = [[1, 0], [0, sqrt(0.7)]], K1 = [[0, sqrt(0.3)], [0, 0]]
#pragma qumo kraus1 "my_damping" (1,0, 0,0, 0,0, sqrt(0.7),0,   0,0, sqrt(0.3),0, 0,0, 0,0) q[0];
c = measure q;
```

```bash
build/qumo run kraus.qasm --shots 10000 --seed 7
```

```
kraus.qasm: 1 qubits, 10000 shots, seed 7, statevector:trajectories, 4.599 ms
  0         2944  0.2944  ############
  1         7056  0.7056  ############################
```

A channel that does not preserve the trace is refused: `error[QM-E414]: Kraus channel 'bad' is not trace preserving`. A wrong number of values is `QM-E560`.

### Which simulators accept it

| Simulator | Kraus channels |
|---|---|
| `sample` (state vector) | yes, as quantum trajectories: one random Kraus operator per shot |
| `density_evolve` / `qumo_density` (density matrix, up to 15 qubits) | yes, exact, no sampling. For the file above the `qumo_density` tool returns probabilities 0.3 and 0.7 |
| stabilizer tableau, Pauli frames | no (QM-E414): they only track Pauli errors |
| `sample_ptsbe` (pattern grouping) | no |
| GPU state vector | not verified for this page; see [GPU.md](GPU.md) |

## 3. A compiler pass

The compiler runs a pipeline written as text: `"cancel,route(sabre,trials=8),basis,fuse1q,check"`. The names in that text are a closed table. `build/qumo describe passes` lists them, and [COMPILER.md](COMPILER.md) explains each one.

### Your own pass, from your own code

A pass is any function from a circuit to an equivalent circuit. You cannot give it a name inside a text pipeline, but you can call it before, after or between built-in pipelines, because `compile_spec` takes and returns circuits.

```mojo
from qumo.ir.circuit import Circuit
from qumo.ir.gates import Op
from qumo.compiler.target import parse_target
from qumo.compiler.pass_manager import compile_spec


def cancel_adjacent_h(c: Circuit) -> Circuit:
    """A pass of your own: drop H H pairs that are next to each other in the gate list."""
    var out = c.empty_like()
    for g in c.ops:
        var n = len(out.ops)
        if g.kind == Op.H and n > 0 and out.ops[n - 1].kind == Op.H and out.ops[n - 1].q0 == g.q0:
            _ = out.ops.pop()
        else:
            out.append(g)
    return out^


def main() raises:
    var c = Circuit(3)
    c.h(0)
    c.h(0)
    c.h(1)
    c.cx(1, 2)
    c.cx(0, 2)
    var target = parse_target("line:3/sc_cz")
    var mine = cancel_adjacent_h(c)                                   # your pass first
    print("my pass:", len(c), "->", len(mine), "gates")
    var r = compile_spec(mine, target, "route(sabre,trials=8),basis,fixpoint(cancel,commute,fuse1q),check")
    print(r.summary())                                                # then built-in passes, by name
    print(r.table())
```

```
my pass: 5 -> 3 gates
compiled 'circuit' for line_3 [rz,sx,x,cz]: gates 3 -> 10, 2q 2 -> 2, depth 3 -> 7, T 0 -> 0, 4 passes in 0.13 ms
pass                                              gates                 2q            depth    time(us)  note
route(sabre,trials=8)                            3 -> 3             2 -> 2           3 -> 3          58  vf2 exact layout, swaps=0
basis                                           3 -> 11             2 -> 2           3 -> 8          22  translated 3 gates to {x,sx,rz,cz}
fixpoint(cancel,commute,fuse1q)                11 -> 10             2 -> 2           8 -> 7          40  2 iters: cancel 0 commute -1 fuse1q 0
check                                          10 -> 10             2 -> 2           7 -> 7           3  conforms to line_3
```

Things to respect in a pass of your own:

- `c.ops` is a flat list of `Instruction` (`kind`, `q0`, `q1`, `q2`, parameters). `c.empty_like()` gives an empty circuit with the same qubits, classical bits and parameters.
- Keep the unitary exact and add any phase you drop to `Circuit.global_phase`. The built-in passes do, and the tests compare state vectors.
- If your pass runs after `route`, the circuit is on physical qubits: keep two-qubit gates on coupling edges, and finish with the `check` pass (it fails with QM-E615 otherwise).
- A pass that moves qubits must be applied before routing, or you must update `initial_layout` / `final_layout` yourself.
- Your pass does not appear in `r.table()`; only the passes inside `compile_spec` do.

This example pass is for illustration: the built-in `cancel` already removes H H pairs, including ones separated by commuting gates.

### A pass with a name in text pipelines (core edit)

To make `my_pass` usable in a pipeline string, from the CLI and from the MCP tool, edit `qumo/compiler/pass_manager.mojo`:

1. `struct PassId`: add a constant and increase `NUM`.
2. `pass_name`: the canonical name.
3. `pass_description`: one line; `qumo describe passes` and the MCP server print it.
4. `pass_id_from_name`: the name and any aliases. This is the only place where text becomes an id.
5. `_validate_args`: if the pass takes arguments, reject bad ones here (QM-E602).
6. `_apply`: the branch that runs the pass on the circuit and returns the note shown in the report.
7. Optionally `preset_pipeline`, if levels 0 to 3 should use it.

Also: `tests/test_ai_transpiler.mojo` asserts the value of `PassId.NUM`; `qumo/compiler/program_compile.mojo` lists the passes that change layouts and needs the new id if yours does; add a state-vector test in `tests/test_compiler.mojo` and a row in [COMPILER.md](COMPILER.md). This procedure was read from the source, not exercised for this page.

## 4. A decoder

A decoder turns the detectors that fired in one shot into a prediction of which logical observables flipped. Anything with the `Decoder` trait runs inside QUMO's parallel sample-and-decode loop:

```mojo
trait Decoder(Copyable, Deinitable):
    def decode(mut self, fired: List[Int]) -> UInt64:
        """Predicted observable-flip mask for one shot's fired detectors (sorted)."""
        ...

    def name(self) -> String:
        ...
```

The worked example is in the repository: [examples/error_correction/qec_custom_decoder.mojo](../examples/error_correction/qec_custom_decoder.mojo). It defines a toy decoder for the repetition code and compares it with the built-in matching decoder on the same shots:

```bash
.venv/bin/mojo run -I . examples/error_correction/qec_custom_decoder.mojo
```

```
toy decoder (illustration): RunStats(shots=500000, errors=90933, seconds=0.006757)  p_L = 0.181866
MWPM (sparse blossom)     : RunStats(shots=500000, errors=5538, seconds=0.038291)  p_L = 0.011076
```

The core of it:

```mojo
var cc = repetition_code_memory(d, d, NoiseParams.uniform(0.02))   # the noisy code circuit
var dem = detector_error_model(cc.circuit)                         # which errors fire which detectors
var prog = FrameProgram(cc.circuit)                                # the fast sampler
var stats = sample_and_decode(prog, MyDecoder(...), 500_000, 1)    # shots, seed
```

Notes:

- The loop copies your decoder once per worker thread (`dec.copy()`), so it must be `Copyable` and must not share mutable state between copies.
- `decode` is called once per shot on the hot path. Allocate buffers in `__init__`, not in `decode`.
- `fired` holds detector ids in ascending order. The result is a bit mask: bit k set means "observable k flipped".
- The detector error model (`dem`) gives you the graph to decode on; `MatchingGraph(dem)` is the matching form. See [QEC.md](QEC.md).

### A decoder with a name (core edit)

`memory_experiment(..., decoder="mwpm")`, the `qumo_qec_memory` tool and the Python `qec_memory` function choose a decoder by name. The names are a closed table in `qumo/qec/experiments.mojo`:

1. `canonical_decoder_name`: add the name and its aliases.
2. `make_decoder_and_run`: add the branch that builds your decoder from the `DetectorErrorModel` and calls `sample_and_decode`. Update the hint of error QM-E707 in the same function.
3. The text of QM-E707 in `qumo/ai/diagnostics.mojo` and the decoder list in the `qumo_qec_memory` tool description (`qumo/exec/tools.mojo`), so that agents see the new name. Regenerate [ERRORS.md](ERRORS.md) afterwards (it is generated from the catalogue).

This procedure was read from the source, not exercised for this page.

## 5. A target

A target is data: number of qubits, coupling edges, basis gates and, optionally, error rates. You never need to edit QUMO to describe a device.

```mojo
from qumo.ir.circuit import Circuit
from qumo.compiler.target import Target
from qumo.compiler.pass_manager import transpile
from qumo.exec.emulate import emulate


def main() raises:
    # A 4-qubit "T" device: qubit 1 in the middle. Edges are a flat list of pairs.
    var edges: List[Int] = [0, 1, 1, 2, 1, 3]
    var t = Target.from_edges(4, edges, "rz,sx,x,cz", name="my_t_device")
    # Calibration: 1q error per qubit, 2q error per edge (same order as edges), readout error per qubit.
    t.set_calibration([1e-3, 1e-3, 2e-3, 1e-3], [5e-3, 3e-2, 6e-3], [1e-2, 1e-2, 3e-2, 1e-2])
    print(t.describe())

    var c = Circuit(3)
    c.h(0)
    c.cx(0, 1)
    c.cx(0, 2)
    c.measure_all()
    var r = transpile(c, t, 2)
    print(r.summary())
    print("layout:", r.initial_layout, " estimated fidelity:", t.estimate_fidelity(r.circuit))
    var noisy = emulate(r.circuit, t, 2000, seed=7)
    print(noisy.counts, noisy.method)
```

```
{"name":"my_t_device","modality":"generic","num_qubits":4,"connectivity":"coupling_map","num_edges":3,"max_degree":3,"diameter":2,"connected":true,"basis":["rz","sx","x","cz"],"avg_error_1q":0.001,"avg_error_2q":0.014,"avg_readout_error":0.015,"avg_t1_us":0,"avg_t2_us":0,"edges":[[0,1],[1,2],[1,3]]}
compiled 'circuit' for my_t_device [rz,sx,x,cz]: gates 3 -> 15, 2q 2 -> 2, depth 3 -> 7, T 0 -> 0, 12 passes in 0.151 ms, est. fidelity 0.9597
layout: [1, 0, 3, 2]  estimated fidelity: 0.9596548199699999
{111: 999, 000: 936, 010: 9, 011: 14, 001: 13, 101: 11, 100: 12, 110: 6} stabilizer:pauli_frame
```

The edge between qubits 1 and 2 was given a 3% error, six times worse than the others. The compiler put the circuit on qubits 0, 1 and 3 and left qubit 2 unused. With different error rates per qubit or edge, the preset pipelines add a noise-aware placement pass; with uniform or no error data, placement only minimises SWAPs.

More ways to set the data: `set_edge_error(a, b, e)` by qubit pair, T1 and T2 as the fourth and fifth arguments of `set_calibration`, `set_angle_bounds` for gates with a limited angle range, `with_basis("...")` to swap the gate set. Basis names and presets are in [COMPILER.md](COMPILER.md).

From the command line, `build/qumo transpile <file> --target SPEC` takes the one-line specs (`--target line:4/rz,sx,x,cz`, or `--basis` to name the gates separately). It has no options for an edge list or calibration data. For those, call the same compiler as a tool, where the target is JSON; this is also the form an MCP client sends. With the GHZ circuit saved as `ghz3.qasm` (step 6 of the [tutorial](TUTORIAL.md)):

```bash
build/qumo tool transpile '{"file":"ghz3.qasm","num_qubits":4,"edges":[[0,1],[1,2],[1,3]],"basis":"rz,sx,x,cz","calibration":{"edge_errors":[[0,1,0.005],[1,2,0.03],[1,3,0.006]]}}'
```

It returned `"initial_layout":[1,0,3,2]`: the same placement, away from the bad edge. `build/qumo tool transpile --help` lists every argument (`"file"` is read by the CLI; an MCP client sends the text in `"qasm"`).

From Python, `qumo.transpile(c, coupling_map=[(0, 1), (1, 2), (1, 3)], basis=["rz", "sx", "x", "cz"], num_qubits=4)` takes the edges and basis directly. Calibration data goes through the low-level call: `qumo.native.transpile_calibrated(c.native, 4, [(0, 1), (1, 2), (1, 3)], ["rz", "sx", "x", "cz"], 2, 7, {"edge_errors": {"0-1": 0.005, "1-2": 0.03, "1-3": 0.006}})` returned the layout `[1, 0, 3, 2]` as well (`help(qumo.native.transpile_calibrated)` describes the dictionary).

### A new one-line keyword (core edit)

`parse_target("grid:4x4/sc_cz")` understands a fixed list of topology words (`line`, `ring`, `grid`, `torus`, `tree`, `heavy_hex`, `all_to_all`, and the modality presets). To add a word, add a constructor to `struct Target` and a branch in `_parse_target`, both in `qumo/compiler/target.mojo`, and extend the QM-E611 hint. Keep presets named by modality or topology, not by vendor: QUMO has no vendor default by design.

A real device from a provider (IBM Quantum, Quantinuum, IonQ) comes in through the hardware plugins, which build the target from the provider's data: [HARDWARE_PLUGINS.md](HARDWARE_PLUGINS.md), section "Adding a provider".

## Related extension points

- A new MCP tool or CLI command: [AI_NATIVE.md](AI_NATIVE.md), section 7 "Extending: register tools and commands".
- A new hardware provider: [HARDWARE_PLUGINS.md](HARDWARE_PLUGINS.md), section 9.
- A new example with machine-checked results: [CONTRIBUTING.md](../CONTRIBUTING.md).
- The Mojo dialect QUMO is written in: [MOJO_GUIDE.md](MOJO_GUIDE.md).
