# Control flow and classical variables

A `Circuit` is a flat list of operations. A `Program` adds what a flat list cannot
hold: `while`, `break`, `continue`, `for` and `switch` decided during the shot,
classical variables, and gate angles computed from measured results.

```
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
output uint[4] tries;
uint[4] k = 0;
while (!c[0] && k < 7) {
  reset q[0];
  h q[0];
  c[0] = measure q[0];
  k = k + 1;
}
tries = k;
```

```bash
build/qumo run rus.qasm --shots 1000 --max-iterations 1000   # counts, iterations, outputs
build/qumo draw rus.qasm                                     # blocks under indented headers
build/qumo validate rus.qasm                                 # QM-W054 and friends
build/qumo convert rus.qasm --to json                        # qumo.program.v1
```

```python
import sys; sys.path.insert(0, "build")
import qumo_py as q
p = q.load_program(open("rus.qasm").read())
r = q.sample_program(p, 1000, 7, max_iterations=1000)  # counts, outputs, iterations, method, warnings
out, initial, final, report = q.transpile_program(p, 7, [[i, i + 1] for i in range(6)], ["rz", "sx", "x", "cz"], 2, 7)
```

```mojo
from qumo.compiler.program_compile import compile_program
from qumo.compiler.target import parse_target
from qumo.ir.qasm import load_any, to_qasm
from qumo.sim.program_runner import sample_program

def main() raises:
    var p = load_any(text)                         # plain input stays what it always was
    var r = sample_program(p, 1000, 7)             # r.sample, r.int_output("tries"), r.iterations
    var c = compile_program(p, parse_target("heavy_hex:3/rz,sx,x,cz"), 2)
    print(c, to_qasm(c.program))
```

Worked examples, checked against closed forms:
[`repeat_until_success.mojo`](../examples/control_flow/repeat_until_success.mojo) and
[`iterative_phase_estimation.mojo`](../examples/control_flow/iterative_phase_estimation.mojo).

## Limits

These hold today, and each one fails loudly instead of doing less:

* **Nothing runs on hardware yet.** The hardware plugins take OpenQASM 2 payloads,
  and OpenQASM 2 has no loops. Text, a file or a `qumo_py.Program` with runtime
  regions is QM-E954. A program without runtime regions is a circuit and compiles.
* **The GPU is not used.** Each shot branches on its own, so `device="gpu"` runs on
  the CPU and reports QM-W020.
* **Detectors and observables are refused** in a program with runtime regions
  (QM-E586). They index measurements as if each ran once, and no decoder takes a
  syndrome of varying length.
* **`verify_program` does not check every block.** It leaves out blocks with a reset,
  noise or a conditional gate, and blocks over its budget, which also covers anything
  over 22 wires, ancillas included. See [Verification](#verification). The layouts on
  every edge are checked regardless.
* **`density`, `ptsbe` and `frame`** are QM-E483: each needs one fixed sequence of
  operations for all shots.
* **The AI presets** are QM-E630, as are synthesis passes, the learned router, a
  second `route` and a relabeling pass inside `fixpoint`.
* **A state vector or an expectation value** of a program is QM-E585: a shot-by-shot
  program has no single final state. So are density matrices, OpenQASM 2, resource
  estimates, device emulation, Pauli-based form and encoding.
* **Hexadecimal, octal and binary integer literals** are QM-E500. The OpenQASM 3
  specification allows them; QUMO does not implement them yet. It is an error, not
  a wrong answer.
* **Subroutines** stay QM-E504.
* **The flat lints do not run on a program.** They read the ops as one pass from
  start to end, which a loop is not. A program gets QM-W054 only.

## What a Program is

`Program` (`qumo/ir/program.mojo`) sits on top of `Circuit`, with no new opcode and
no change to `Instruction`. Every walker of `.ops` skips what it does not know, so a
`WHILE` opcode would be dropped in silence. With a type of its own, the Mojo compiler
rejects any path that was not adapted.

| Field | Holds |
|---|---|
| `body` | Arena `Circuit`. A quantum block is a range `[lo, hi)` of its ops, so conditions, Kraus channels and noise tables stay valid |
| `tape` | Statements in preorder with `end` links: OPS ASSIGN IF ELSE WHILE FOR SWITCH CASE BREAK CONTINUE END. There is no goto |
| `vars` | Name, type, storage, parameter slot, output flag. A `bit` variable is a range of clbits |
| `nodes` | Expressions in postorder, evaluated by one forward scan with no recursion and no allocation per shot |

`cfg_of(p)` derives basic blocks with JMP, BR, SWITCH and HALT terminators and marks
back-edges. The interpreter and the verifier both run on it.

A gate angle known only at run time is a symbolic `Param` on a slot of
`body.param_names` with a reserved name. An ASSIGN ends the block, so each slot has
one value per block and `apply_instruction(sv, g, params)` did not change.

`load_any` runs the flat parser first, so anything a `Circuit` can hold keeps its
exact meaning. A constant `for` stays unrolled, and stays on the tape when it has
`break` or `continue`, when a bound is known only at run time, when its body assigns
the loop variable, when its range does not fit the variable's type, or when an empty
range has a `bit` in its body.

## Surfaces

| Surface | Program support |
|---|---|
| Mojo | `load_any`, `parse_qasm_program`, `to_qasm`, `draw_program` (`qumo.ir.qasm`); `sample_program` (`qumo.sim.program_runner`); `compile_program`, `compile_spec_program`, `verify_program`, `program_metrics` (`qumo.compiler.program_compile`); `program_to_json`, `program_from_json` (`qumo.ai.describe`); `lint_program` (`qumo.ai.diagnostics`); `ProgramBuilder` (`qumo.ir.program`) |
| Python | `qumo_py.Program`, `load_program`, `program_from_json`, `sample_program`, `transpile_program`, `draw_program` |
| CLI | `qumo run` with `--max-iterations`; `qumo draw`, `validate`, `info`, `convert` |
| MCP | `qumo_run_qasm`, `qumo_run_device`, `qumo_transpile`, `qumo_draw`, `qumo_validate_qasm`, `qumo_circuit_info`, `qumo_convert` take `qasm`, `program` or `circuit`. Every tool that needs one circuit answers QM-E585 |

`arg_program` is how a tool reads a program, and `arg_circuit` is `arg_program` plus
QM-E585. A run reports `iterations` (mean and max back-edges per shot) and `outputs`
(mean, range and the first 16 shots of each declared output). A compile reports
`metrics`, `restored_swaps`, `blocks_checked` and `blocks_unchecked`.

## Classical semantics

The OpenQASM 3 specification leaves gaps, and QUMO fills them as below. Folding at
parse time gives what a run would compute, with **one deliberate exception**: an
integer constant with no width is exact. It can hold a value the run-time form would
wrap or refuse.

### Unsized constants are exact

This covers integer literals, `const int` and `const uint` without a width, unsized
declarations that fold to constants, and unsized unrolled loop variables, in both
parsers. A sized type (`int[8]`, `uint[8]`) wraps by the table, and evaluation at run
time does not change at all.

| Case | Rule |
|---|---|
| `/` and `%` | Still truncate, as below: `7 / 2` is 3 |
| A result past 64 bits | Does not wrap. It goes on as a float: `10000000000 * 10000000000` is 1e20, `float t = 2 ** 70` is 2^70, `rz(pi / 2 ** n)` with `n = 64` is pi / 2^64 |
| A negative exponent | Gives a float: `2 ** -1` is 0.5, `10 ** -3` is 0.001 |
| Negating or underflowing an unsized `uint` | Gives the negative number: `for uint k in [1:1] { rz(-k) q[0]; }` is -1 |
| `int` with `uint` | Mix with no QM-E580 when both are unsized constants: `const int a; const uint b; rz(a + b)`, `q[n - k]` |
| Into an unsized target, out of range | QM-E482, never a wrap: `int x = 18446744073709551615;`, `uint x = -1;`, `const int x = 2 ** 64;` |
| Into a sized target | Wraps by the table: `int[8] x = 300;` is 44. A value that went on as a float is out of range: `uint[8] x = 2 ** 70;` is QM-E482 |

Next to a value known only at run time, an unsized constant is an ordinary `int` or
`uint`, and the table below applies.

Qiskit's `qasm3.loads` agrees on this shape, measured on 2.6.1 by the coordinator of
this work: `rz(3/4*pi)` is 0.0 and `rz(10000000000*10000000000)` is 1e20.

### The table

| Case | Rule |
|---|---|
| Types | `bit[n]`, `bool`, `int[n]`, `uint[n]` (1 <= n <= 64), `float` (binary64), `angle[n]`. `float[32]` and a mutable `angle` without a width are QM-E582 |
| Overflow | Modulo 2^n, two's complement. `int[8]` 127 + 1 is -128 |
| `/`, `%` on integers | Truncate toward zero, literals included: `7 / 2` is 3, also in `float x = 7 / 2`, which is 3.0. `%` has the sign of the dividend: `-7 % 3` is -1 |
| `/`, `%` on floats | `%` truncates the quotient: `-7.5 % 2.0` is -1.5. `mod(-7.5, 2.0)` floors it: 0.5 |
| Division by zero | QM-E481 at run time, with shot and line. QM-E533 when the divisor is a constant zero. `mod(x, 0.0)` is QM-E481; `mod(1.0, 0.0)`, all constant, is QM-E533 |
| `**` on integers | Modulo 2^n. A negative exponent is QM-E481 at run time. On a sized constant it is QM-E533; on an unsized one, see above |
| Shifts | `bit`, `uint` and `angle` only; `int` is QM-E580. Shifting by the width or more gives 0. A count is a uint or a non-negative literal |
| `&&`, `\|\|` | Left to right, short-circuit. `false && (1 / i > 1)` does not divide. A constant zero divisor is still QM-E533, reached or not |
| `int` with `uint` | QM-E580, except a literal |
| `bool` and `bit` in arithmetic | Promote to 64-bit `int`, C99 style, and so does the narrow type beside them: `true + 1` is 2, `b + b` is 2, `c[0] + c[1]` is 2, `-true` is -1, `uint[8]` 255 + `true` is 256, `int[8]` 127 + `true` is 128. Next to a 64-bit `uint` they stay in it |
| A literal with a typed operand | Takes the operand's type when it fits. A literal too wide is QM-E580 in arithmetic and bitwise operators: `int[8] a; a + 1000`, with a hint to cast. Shift counts are exempt |
| Comparison with a literal | Exact, whatever the width: `uint[8] a; a < 1000` is true, `a == 2^64 - 1` is false |
| Float to integer | Truncates: 2.9 is 2, -2.9 is -2. NaN, infinity or out of range is QM-E482 at run time; a constant one is QM-E482 at parse time, reached or not |
| Float to `angle[n]` | Nearest step, a tie going to the even step, modulo a turn |
| Angle in a gate | `angle[n]` with bits v is 2 pi v / 2^n radians |
| Gate angle at run time | NaN, infinity or an overflow through `coeff * x` is QM-E482 at the gate that reads it. A NaN in a variable no gate reads is no error |
| `angle` to `int`, `uint` or `float` | QM-E580, as in the specification's cast table. `+` and `-` take two angles of one width |
| Casts to `bit[m]` | An explicit `bit[m](x)` needs x of m bits, or a literal that fits (QM-E580). Assigning to a bit register keeps the low bits: `bit[3] b; b = 13` holds 5 |
| Read before assignment | QM-E581, by analysis on the control-flow graph, gate angles included. `bit` starts at 0 and is exempt. The analysis does not look at constant conditions: `while (true) { x = 1; break; } r = x;` is QM-E581. Conservative, not a wrong answer |
| `bit` declared in a loop body | Restarts at 0 on each pass, and its name is scoped to the block. It keeps one clbit per declaration, even when the loop never runs |
| Loop variable | Counts in `int` or `uint` (QM-E580 otherwise; flat `parse_qasm` still takes `for float x`). It is declared first in the body, so redeclaring it is QM-E522. Assigning it is allowed and does not change the iteration |
| Ranges | A step of 0, or a step or bound with a fraction, is QM-E570. A constant float with no fraction counts as an integer, unrolled or on the tape: `[0.0:1]`. Stepping cannot overflow at the ends of int64, and when neither bound can be negative they compare unsigned, so `for uint i in [9223372036854775807:n]` with `n` = 2^63 runs twice |
| `bit` in a loop | Keeps its size on every pass: `for int i in [0:1] { bit[i + 1] b; }` is QM-E504 |
| Integer positions | A qubit index, a size or a range bound takes an integer, bool, bit or integral float constant. An `angle` is QM-E530 |
| Gate bodies | A name of a typed constant follows the table there too, and in `pow(k / 2) @` and `ctrl(k / 2) @`. Gate parameters are angles and stay affine |
| Symbolic constants | `input float t; float u = t * 2; float w = 1.0; rz(u + w) q[0];` is `rz(2*t + 1)`: they read as affine |
| Non-finite constants | In a flat declaration or gate argument, gate bodies included, QM-E533: `sqrt(-1.0)`, `ln(0.0)`, `1e308 * 10`. In a program, QM-E482 at the gate that reads one |
| `switch` | On an integer or bits; on a float it is QM-E580. A label of the wrong sign for the target is QM-E580 |
| Variables in a shot | Nothing survives from the shot before: variables, slots and bits start again |
| Loop budget | `max_iterations` back-edges per shot, 65536 by default; past it, QM-E480 |
| `input angle[n]` | A float in radians in both parsers, not quantised to n bits. In expressions it behaves as a float |

### Warnings

QM-W054 is raised when nothing in a `while` body writes what its condition reads and
no `break` leaves that loop. A write is an assignment, a `for` variable, or a
measurement into a bit it reads. Such a loop never runs, or runs until QM-E480.

QM-W055 is raised under an OPENQASM 3 header, by both parsers, because integer division
that loses a remainder is a trap where an angle is meant:

* a truncation in a classical constant used as an angle or a float: `float t = 3 / 4 * pi;`
  is 0, and so is `rz(i / 4 * pi)` for `i` up to 3 in an unrolled loop;
* a literal-only gate argument with real division: `rz(3/4*pi)` stays 3 pi / 4 here,
  where the OpenQASM 3 specification and Qiskit give 0.

The hint is to write `3.0 / 4`, which every tool reads alike. Literal-only arguments keep
their meaning, and OPENQASM 2 sources, or a source with no header, get no warning.

## Simulation

`sample_program(p, shots, seed, params, max_iterations, method)`. A program without
runtime regions goes to `sample`, so `Program.from_circuit` gives identical counts.
Otherwise a program of Clifford gates and Pauli noise goes to the tableau and the
rest to the state vector. Shots run in parallel up to 14 qubits, each on
`rng.split(shot)`, so the counts do not depend on the number of workers; above that,
shots run in series and the kernels in parallel.

Clifford status stops past 2^22 quarter turns. `theta / (pi/2)` is a rounded number,
and at 2^51, with one fractional bit left, it comes out whole for angles up to 0.57 rad
from every multiple of pi/2 (measured with exact arithmetic). This limit applies to
`Instruction.is_clifford`, `quarter_turns` and the compiler's copies, for circuits as
well as programs.

## Compiling

`compile_program(p, target, level, seed)` keeps one layout for a program:

1. It splits the preset pipeline at its `route`. Passes before and after it run on
   each block, without `elide_swaps`, `post_layout` or `star`, which relabel qubits
   going forward only.
2. It picks one layout from a summary circuit that weighs loop bodies.
3. It routes each block from `RouteOptions.initial_layout`.
4. **Invariant:** a region exits with the layout it entered with. Each branch and
   each `break` or `continue` restores the layout of its header with a SWAP network.
   If none exists, QM-E632.

A runtime angle on a gate with target angle bounds is QM-E633.

`program_metrics` gives `base + sum(per_iteration_i * N_i)`, the costlier branch of
each `if` counting. There is no `depth()`: one number would mislead.

## Verification

`verify_program` runs inside every compile, and a failure is QM-E631. It checks:

* the skeleton: the same statements in the same order, and the same classical tables;
* each source block a shot can reach has its compiled block at the same place in the
  skeleton, with the same source range, so a block that optimises to nothing stays as
  an empty one;
* every edge of the graph joins equal layouts, `break` and `continue` included;
* `l_out == perm(l_in)` on every block;
* a compiled block has no reset, noise or conditional gate that its source lacks;
* each block equals its source on two seeded random states.

That last check runs both sides through the state-vector kernels over the block's
wires, ancillas included, and each state must come out as the other side times one
unit complex number, the same for both states, within a distance of 1e-7. That covers
the global phase. A difference of any size changes the result on a random state with
probability 1, and comparing distances rather than overlaps keeps a small one visible:
a change of 1e-5 rad in one `rz` is caught. A measurement is replaced by a CX onto a
fresh ancilla, and ancillas are paired by (clbit, nth write to that clbit): the deferred
measurement principle. Both sides reduce huge angles first, so a phase of 1e12 is not a
false alarm.

The cost of a block is gates x 2^wires x 2 states. The budget is **2^27 per block and
2^29 per compile**, with at most **22 wires** for the memory of one state. Over budget
a block is unchecked and counted. A block whose source has a reset, noise or a
conditional gate stays unchecked too, because pairing them across a reordering could
raise false alarms. The result counts both: `blocks_checked` and `blocks_unchecked`.
A compiled program whose blocks all start with a reset, like iterative phase estimation
here, has its layouts and structure checked and none of its gates.

Paired compile times on this machine, in the foreground, against commit a55628b, which
checked only blocks without measurements: 120 small programs for `line:6` at level 2
take 0.063 to 0.068 s against 0.060 to 0.065 s, and a 200-gate loop body on 7 qubits with
3 measures, for `line:7` at level 2, takes 4.3 to 6.1 ms against 4.1 to 4.7 ms.

The first run of this check tripped on a valid compile, and it was right:
`transpile` at level 2 was miscompiling `p(2^63 - 1)`. Merging sums the angles, and
at that size the sum drops the smaller term and `wrap_angle` has no bits left.
`PassManager.run` now reduces constant angles of magnitude 1e4 or more through sin
and cos before any pass (`reduce_large_angles`). Compiled output changes only for
such angles, in plain circuits too. The offset of a symbolic angle reduces the same way.
A huge coefficient on a symbol stays a float limit, and so does binding: at 2^63 the sum
`t + offset` in `Param.resolve` drops `t`, so an uncompiled circuit loses `t` where a
compiled one keeps it.

## Cross-check

`bench/crosscheck/control_flow.py` shares no code with QUMO. It parses with the
`openqasm3` package, interprets the AST on numpy with exact branch weights, and
compares distributions under a total-variation limit fixed before running. Its
generator writes 3-qubit gates, runtime angles read under branches, `bit` declarations
in loops, narrow types with literals, unsigned values at or above 2^63, `bool` and
`bit` arithmetic, negative bases under `**` and `for` at the ends of int64, and 10% of
its programs end in a run-time error that both sides must raise with the same code.
It also writes gate arguments from unsized constants, with its own exact integers: past
64 bits, under negative exponents, as negated `uint` loop variables, in real arithmetic
when literal-only, and read in a gate body next to a parameter.

Measured at 20000 shots, with no program skipped by the oracle:

| Corpus | Programs | Both raised | Worst TV / limit | Blocks checked | Not checked |
|---|---|---|---|---|---|
| Source | 500 | 57 | 0.482 | n/a | n/a |
| `line:6`, level 0 | 200 | 17 | 0.400 | 1580 | 61 |
| `heavy_hex:3`, level 2 | 200 | 19 | 0.341 | 1661 | 56 |

Those 900 programs, source and compiled, also came back identical through JSON, and
60000 mutated documents went through the JSON loader with no crash and no QM-E631.
Aer is a partial reference only: it ran `while_loop`, `store` and logic, comparison
and bitwise expressions on registers, and failed on `expr.add` and on a free `Bool`.

## JSON: `qumo.program.v1`

The format wraps `qumo.circuit.v1`, which holds the arena.

```json
{
  "format": "qumo.program.v1",
  "circuit": {"format": "qumo.circuit.v1", "num_qubits": 2, "instructions": ["..."]},
  "num_store": 3,
  "user_params": [],
  "registers": [{"name": "c", "first": 0, "width": 2}],
  "variables": [{"name": "k", "type": "uint[4]", "store": 0, "slot": -1, "output": false}],
  "nodes": [{"op": "var", "type": "uint[4]", "x": 0}, {"op": "const", "type": "uint[4]", "i": 7}],
  "expressions": [[0, 2]],
  "for_ranges": [],
  "case_values": [],
  "statements": [{"kind": "while", "a": 0, "line": 8}, {"kind": "ops", "a": 0, "b": 3}, {"kind": "end"}]
}
```

`x`, `y`, `i` and `f` of a node are left out when zero. A constant JSON cannot write
is the string `"nan"`, `"inf"` or `"-inf"`, in a node and in an affine coefficient.
Integers keep 64 bits.

`program_from_json` trusts nothing in the file. Every index is checked, because a bad
one would be a crash inside a shot, and each node may read only what comes before it in
its expression and is not skipped by a short-circuit. Blocks cover the arena in order, and
a statement must hold the kind of value the interpreter reads from it. Anything wrong is
QM-E587. The links between statements are not in the file: they are built again from
the statements, so a wrong nesting is QM-E583. A read before assignment is QM-E581, as
from text.

## Accepted flat inputs that changed

`parse_qasm` output is unchanged except for what follows, against commit 11e15b5. The
golden corpus did not move. On the second review's differential, 700 random flat loop
programs are identical, and so are 1486 of 1500 random float-only programs: the other
14 are a NaN that is now an error, or an error reworded. Of 956 repository files and
edge cases, 902 are identical and the other 54 are what this section lists. `Before`
is 11e15b5, and every row was run on both trees.

**Typed declarations follow the classical table:**

| Input | Before | Now |
|---|---|---|
| `int[8] x = 300;` | 300 | 44 |
| `uint[8] x = -1;` | -1 | 255 |
| `angle[8] x = 0.1;`, `angle[32] t = 0.1;` | 0.1 | quantised to 8 or 32 bits |
| `const int m = -7 % 3;`, `float t = -7 % 3;` | 2 | -1 |
| `const int a = 7; const float x = a / 2;` | 3.5 | 3.0 |
| `float a = 7 / 2;` | 3.5 | 3.0 |
| `float t = 1/2;`, `float[16] t = 1/2;` | 0.5 | 0, with QM-W055 |
| `float t = 3 / 4 * pi;` | 3 pi / 4 | 0, with QM-W055 |
| `bool b = 2;` | 2 | 1 |
| `uint[1] i = 3; h q[i];` | q[3] | q[1] |
| `float t = 18446744073709551616;` | 0 | 2^64 |

**Typed names in expressions follow it too**, in gate arguments, gate bodies, `gphase`,
pragma values and the arguments of a gate call, wherever they are read:

| Input | Before | Now |
|---|---|---|
| `const int a = 3; gphase(a / 2);` | 1.5 | 1 |
| `const int a = 3; ctrl @ gphase(a / 2) q[0];` | `p(1.5)` | `p(1)` |
| `const int k = 3; #pragma qumo depolarize1(1 / k) q[0]` | 0.333 | 0 |
| `for int i in [0:3] { rz(i / 2) q[0]; }` | 0, 0.5, 1, 1.5 | 0, 0, 1, 1, with QM-W055 |
| `for int i in [0:3] { rz(i / 4 * pi) q[0]; }` | multiples of pi / 4 | 0, with QM-W055 |
| `const int k = 3; gate g(a) r { rx(k/2) r; } g(k/2) q[0];` | `rx(1.5)`, a = 1.5 | `rx(1)`, a = 1 |
| `input float t; const int k = 3; rz(t + k / 2) q[0];` | `t + 1.5` | `t + 1` |
| `const float k = -7.5; rz(k % 2) q[0];` | 0.5 | -1.5 |
| `const float k = 1.0; rz(k + 1/2) q[0];` | 1.5 | 1, with QM-W055 |
| `angle[8] a = pi/2; rz(-a) q[0];`, also under `ctrl @` | -pi / 2 | 3 pi / 2: an angle wraps |

Under rule R1 these did not change: `float t = 2 ** 70;`, `rz(pi / 2 ** n)`, `rz(-k)` for
an unrolled `uint k`, `rz(pi * 2.0 ** (-k))`, `q[n - k]` and `rz(a + b)` with `int` and
`uint` constants, `rx(2 ** -i)`, and `rz(10000000000*10000000000)`.

**Now refused:**

| Input | Before | Now |
|---|---|---|
| `int t = 18446744073709551615;` | -1 | QM-E482 |
| `uint x = -1;` | -1 | QM-E482 |
| `const int x = 2 ** 64;`, `const int x = int(1e30);` | 9.2e18 | QM-E482 |
| `float t = 1e400;`, `rz(arcsin(2.0))`, any NaN or infinity | ran on as NaN | QM-E533 |
| `angle[n] a = 1;` | 1 | QM-E580 |
| `angle[8] t = pi/4; float f = t;`, `rz(sin(t))`, `float t = angle(0.3);` | the radians | QM-E580: no cast from angle to float |
| `angle[8] t = pi/4; rz(t * 0.5) q[0];` | pi / 8 | QM-E580: an angle scales by a uint |
| An integer position that an expression computes past 2^53 | rounded | QM-E580: write it as one literal |
| `if (c[0] == 1) { qubit r; }` | accepted | QM-E504 |
| `for int[8] i in [0:300] { x q[0]; }` | 301 passes | QM-E504; `load_any` takes it, and the variable wraps |
| `for uint i in [2:-1:-1] { rz(i) q[0]; }` | ran with -1 | QM-E504; `load_any` takes it, and the variable wraps |
| `for int i in [1:0] { bit[2] b; }` | nothing | QM-E504: the bit still takes a clbit; `load_any` takes it |
| An `if` body that writes a bit it tests, with an operation after the write | wrong answer | QM-E541 |
| `int d; d = c;` with `c` measured again before the `switch` on `d` | wrong answer | QM-E541 |

**Now accepted:** `bool b = 1 < 2;`, `bool b = true && false;`, `float t = 1 << 3;`,
`float t = 2 ^ 3;`, `int n = "0101";`, `rz(int[8](k))`, and `c[0] == v && c[1] == 0` with a
typed `v`, which gave QM-E500 or QM-E541.

**Also changed:**

* `to_qasm` writes a condition on part of the bits as an `&&` chain, and `parse_condition`
  reads it. It used to drop the gate behind a comment, which changes the emitted text of
  three repository files and of a `switch` on a typed constant.
* Compiled output changes for constant angles of magnitude 1e4 or more, and for the
  offset of a symbolic angle of that size (`reduce_large_angles`).
* Past 2^22 quarter turns an angle is not Clifford: `sample_stabilizer` raises QM-E701,
  `choose_method` picks the state vector, and `t_count` counts it.
* JSON numbers of 19 digits are exact integers, and an affine coefficient that overflowed
  is written as a string. It used to come back as 1.
* `qumo_statevector` and the other circuit tools answer QM-E585 for a `while`, where they
  used to answer QM-E504. `qumo_validate_qasm` calls it valid.
* Some QM-E533 and QM-E522 messages are worded differently, under the same codes.

## Known differences from other tools

**Literal-only gate arguments use real arithmetic.** `rz(1/2)` is 0.5 and `rz(-7 % 2)`
is 1, in both parsers, as a constant angle always was in QUMO. `qiskit.qasm3.loads`
(Qiskit 2.6.1) gives `rz(1/2)` = 0 and `rz(7/2)` = 3, and refuses `%`. A gate argument
that names a typed constant or a loop variable follows the classical table, so
`const int a = 7; rz(a/2)` is 3.0 in both. QM-W055 points at the difference.

Float loop variables, the `angle` declaration of `input`, and the wider literal rule
are in the table above. `expr.add` on `Uint(8)` and 300 is a `TypeError` in Qiskit
2.6.1, and a literal that fits keeps `Uint(8)`, which is what QM-E580 matches.

## Error codes

| Code | Meaning |
|---|---|
| QM-E480 | a shot passed `max_iterations` |
| QM-E481 | division by zero, or an integer power with a negative exponent, in a shot |
| QM-E482 | a float reached an integer or a gate angle as NaN, infinite or out of range |
| QM-E483 | a method that cannot run a program |
| QM-E484 | an opcode a block cannot run |
| QM-E533 | a constant division by zero, a sized constant under a negative exponent, or a flat constant that is not finite |
| QM-E580 | classical type error |
| QM-E581 | a variable may be read before it is assigned |
| QM-E582 | a type that cannot be represented |
| QM-E583 | malformed control flow |
| QM-E584 | a runtime value where a constant is needed |
| QM-E585 | a program where one circuit is needed |
| QM-E586 | a detector or observable in a program with runtime regions |
| QM-E587 | program JSON that cannot be read |
| QM-E630 | an AI preset, synthesis pass, second route or relabeling inside fixpoint |
| QM-E631 | a compiled program failed verification |
| QM-E632 | no SWAP network restores a region's layout |
| QM-E633 | a runtime angle on a gate with target bounds |
| QM-W054 | a `while` body cannot change its condition |
| QM-W055 | integer division loses a remainder where an angle is meant, or a literal-only argument keeps real division |

Full text and fixes: [ERRORS.md](ERRORS.md).
