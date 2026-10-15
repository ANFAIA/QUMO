# Fault-tolerant programming in QUMO (`qumo.ft`)

`qumo.ft` takes a QEC-agnostic QUMO circuit and turns it into a fault-tolerant
physical program, or lets you write programs directly in a code's logical gate
set. It is a native Mojo re-implementation of the key ideas of
[Guppy FT](https://github.com/Quantinuum/guppyft) (Quantinuum, Apache-2.0),
adapted to QUMO's static circuit IR. No Python, Guppy or HUGR is involved.

```mojo
from qumo.ft.arch import SteaneBuilder
from qumo.ft.encode import encode
from qumo.ft.emulate import run_encoded

var steane = SteaneBuilder().build(3)          # Guppy FT: SteaneBuilder().build(n_blocks=3)
var prog = encode(steane, teleportation)       # computational -> logical -> physical
print(run_encoded(prog, 1000))                 # tableau or state-vector emulation
```

Run the tests and examples:

```bash
.venv/bin/mojo run -I . tests/test_ft_code.mojo     # Pauli algebra, code library, validation errors
.venv/bin/mojo run -I . tests/test_ft_verify.mojo   # every gadget verified, wrong gadgets caught
.venv/bin/mojo run -I . tests/test_ft_encode.mojo   # compile/encode/emulate vs unencoded programs
.venv/bin/mojo run -I . tests/test_ft_rz.mojo       # Clifford+T synthesis, comparator Rz
bench/.venv/bin/python examples/run_examples.py --only examples/ft --tier all
```

## Three abstraction levels and three passes

Guppy FT separates a program into three levels; QUMO keeps the same names.

| Level | Guppy FT | QUMO |
|---|---|---|
| computational | Guppy program, any gate set, noiseless | `qumo.ir.circuit.Circuit` (any gates, mid-circuit measurement, reset, `c_if`, `c_if_register`) |
| logical | HUGR built from the architecture's logical extension | `qumo.ft.logical.LogicalProgram` (handles from `alloc()` / `prep_t()`, logical ops, decoded clbits) |
| physical | HUGR with gadgets linked in | `qumo.ft.implement.EncodedProgram` (a plain `Circuit` + output/post-selection map) |

| Pass | Guppy FT | QUMO |
|---|---|---|
| `compile` | `ReplacementCompiler`: computational ops -> logical ops | `qumo.ft.encode.compile(arch, circuit)`: decompositions, Rz synthesis, lazy allocation, classical-qubit folding |
| `implement_ops` | links opaque logical ops to Guppy implementations, global `STATE` | `qumo.ft.implement.implement_ops(arch, lp)`: gadgets, allocator, factories, QEC policy, Pauli frame, decoding |
| `encode` | both | `qumo.ft.encode.encode(arch, circuit)` |

`check_may_encode(arch, circuit)` reports the first problem without encoding.

## Codes (`qumo.ft.code`)

`StabilizerCode` stores n, k, d, the n-k generators and the k logical X and Z
operators (Y is derived as i X Z); `css` is derived. `validate()` checks the
generator count, pairwise commutation, independence, logicals commuting with
the stabilizers, canonical X_i Z_j relations and logical independence (each
failure is a precise `QM-E721`). `min_distance()` brute-forces the distance of
small codes.

| Code | [[n,k,d]] | Source |
|---|---|---|
| `steane_code()` | [[7,1,3]] | Guppy FT `STEANE_DEF` |
| `iceberg_code(k)` | [[k+2,k,2]], k even | layout [t, b, d_0..d_{k-1}], X_i = X_t X_i, Z_i = Z_b Z_i (arXiv:2211.06703) |
| `css_422_code()` / `noncss_422_code()` | [[4,2,2]] | Guppy FT verifier tests (non-CSS from arXiv:2510.18753) |
| `five_qubit_code()` | [[5,1,3]] | perfect code (non-CSS) |
| `surface_code(d)` | [[d^2,1,d]] | adapter: checks read from `qumo.qec.codes.surface_code_memory` |
| `repetition_code(d)` | [[d,1,1]] | adapter over `repetition_code_memory` (bit-flip distance d) |

Pauli labels use the Guppy FT / Stim convention: character i is physical qubit i.

## Architectures (`qumo.ft.arch`, `qumo.ft.gadgets`)

An architecture = code + physical primitives + logical gate set + passes +
resource policies, built with Guppy-style builders:

```mojo
var steane = SteaneBuilder()
    .with_qec_policy(QECPolicy("knill", 2.0).with_cost("h", 1.0).with_cost("cx", 1.0))
    .with_zero_factory_conf(FactoryConf(2, 5))   # 2 states ahead of use, 5 RUS attempts
    .with_feed_forward("frame")                  # or "eager"
    .with_magic_prep("ft")                       # flag-verified T|+>
    .with_rz("static", 1e-2, 16)                 # or "comparator", "exact"
    .build(3)
var ice = IcebergBuilder(4).build(2)             # two [[6,4,2]] blocks = 8 logical qubits
```

| | Steane [[7,1,3]] | Iceberg [[k+2,k,2]] |
|---|---|---|
| logical gate set | X Y Z H S S^dag CX CZ (transversal; logical S = transversal S^dag), SWAP (relabel), T / T^dag by magic-state injection | X Y Z, RZ(a) = RZZ(a) on (b, d_i), RX(a) = RXX(a) on (t, d_i), RZZ / RXX between slots (weight 2) or blocks (4-body parity ladder), SWAP (relabel) |
| |0>_L | 8-CX encoder + 1 flag (weight-3 Z_L representative) | H + CX fan-out (GHZ) |
| magic state | |H> encoder (10.1038/srep19578 Fig. 3b) + logical S^dag H; `ft`: + logical-H measurement with a cat ancilla + flagged syndrome extraction (arXiv:1804.06995 Fig. 5), 8 flags, 3 reused ancillas | not needed (rotations are native) |
| measurement | transversal, Hamming lookup decoder (corrects 1 X error) | transversal, error detection (Z^n parity) -> post-selection |
| QEC cycle | `steane` (two one-bit teleportations through fresh |0>_L / |+>_L) or `knill` (Bell-pair teleportation) | `syndrome` (X^n and Z^n with 2 ancillas, detection flags) |

Gadgets are plain `Circuit` templates, so every one of them is verified on its
own (see Verification). `verify_steane_library` / `verify_iceberg_library`
re-run the whole library; the MCP tool `qumo_ft_codes` reports it.

## Runtime resource management, statically

| Guppy FT (runtime `STATE`) | QUMO (`qumo.ft.implement.Engine`) |
|---|---|
| stack of `(block, qubit)` addresses; `allocate_next_addr` pops, `free_addr` pushes | same LIFO stack, resolved at encode time; Iceberg slots share a block (k > 1); `QM-E725` when exhausted |
| `mem_swap` of logical blocks (teleportation EC, injection) | handles are re-pointed to other physical blocks: zero gates |
| `StateFactory` + `PreBlock.force_check`: batches of RUS preparations, flags read only when the state is needed | `FactoryConf(size, max_attempts)`: `size` states prepared ahead of use (refilled after each consumption, bounded by the remaining demand counted from the program); `max_attempts` flag-verified attempts unrolled under classical conditions; all attempts failing -> post-selection flag (Guppy: `exit`) |
| `QECPolicy`: per-block cost counters, cycle at threshold | identical counters, evaluated while encoding |
| `if decode(measure_z(a)): x(q)` | Pauli frame of classical expressions; physical corrections only when needed |

**What is static and what stays runtime.** Guppy programs have loops and
runtime control flow, so Guppy FT runs the allocator, the factories' queues
and the QEC counters on the device's classical processor. A QUMO circuit is a
static instruction list. Every decision that Guppy FT takes from *program
structure* (which address, which block, when a counter crosses its
threshold, when to refill a factory) is therefore taken at encode time, with
the same result. Every decision that depends on *measurement outcomes*
(decoded corrections, RUS retries, user `if`s, Iceberg detections) stays a
run-time classical condition inside the circuit. Consequences: RUS loops are
unrolled to `max_attempts` attempts (a static circuit contains every attempt;
a device skips the inactive ones because they are conditioned), and factory
"batches" become prefetching (states prepared earlier, in parallel with the
computation) rather than a runtime queue.

### Deferred reads, Pauli frame and the classical lane

QUMO conditions test bits (`c_if`) or ANDs of bits (`c_if_register`); the IR
has no classical arithmetic. `qumo.ft.classical.CExpr` represents any
classical value as an XOR of terms (bit, AND of literals, constant 1). A
decoded Steane outcome is `parity(raw) XOR [syndrome minterms]`; frame entries
and outputs are XORs of decoded values. Values are materialised only when a
quantum operation depends on them, by a *classical lane*: one qubit kept in a
computational basis state (reset, one X conditioned on each term, measure
into a fresh clbit). It never touches data qubits and is noise-free by
construction, so it is an exact stand-in for a classical register on every
QUMO backend.

Pauli corrections never cost gates in `frame` mode: they update a per-qubit
frame that is propagated through Clifford gates at encode time and folded
into decoded measurement results. A frame is flushed (turned into
conditioned physical Paulis) only before an operation it does not commute
with (rotations, conditioned non-Pauli gates). A classically conditioned S
whose condition equals the X frame (exactly the T-injection correction
`if r: X; S`) keeps the frame linear (`z ^= r`), so T gates never force a
flush. `eager` mode applies every correction immediately; tests check that
both modes give identical logical distributions.

Classical-only qubits of the computational circuit (touched only by reset,
X - possibly conditioned - and measure, e.g. a lane computing an OR) are not
encoded at all: `compile` folds them into `CSET` classical assignments.

## Rz synthesis (`qumo.ft.rz`)

| Path | When | Cost |
|---|---|---|
| exact | angle a multiple of pi/4 | T^m up to phase: <= 1 T |
| static search (`rz_mode="static"`) | known angle | fewest-T Clifford+T word within `rz_epsilon`, exhaustive over Matsumoto-Amano normal forms up to `rz_max_t` T gates (2^max_t words) |
| comparator RUS (`rz_mode="comparator"`, `append_comparator_rz`) | known or run-time angle | n = 1 + ceil(log2(1/eps)) register qubits + n - 1 workspace; T-count per attempt 4(n-2) + 14; success probability > 1/2 per attempt |

Static search precision (best distance d = sqrt(1 - |tr(U^dag V)|/2) for
Rz(0.3), measured with `rz_precision_table`):

| T budget | 0 | 5 | 7 | 9 | 10 | 13 | 14 | 16 |
|---|---|---|---|---|---|---|---|---|
| error | 0.106 | 0.104 | 0.050 | 0.033 | 0.032 | 0.029 | 0.018 | 0.016 |

This is optimal for the budget but scales exponentially; number-theoretic
synthesis (gridsynth, T ~ 3 log2(1/eps)) is the scalable alternative and is
not implemented. The achieved error of every synthesised rotation is written
to `LogicalProgram.notes`.

The comparator circuit (arXiv:2404.05618, as Guppy FT's
`decompose/_comparator_based_rz.py`) is re-derived as the carry of
a + (2^n - k): OR/AND ladders of Gidney temporary ANDs (4 T, measurement
uncompute) whose k-dependent gates are only X gates, i.e. Paulis
conditioned on the bits of k, which the Pauli frame absorbs once encoded.
With `angle_bits` the angle is read at run time from classical bits (the
test checks that circuits for two angles differ only in the angle register).
Attempts after the first act on a scratch qubit that a conditioned SWAP
exchanges with the target only when needed, so no T gate is ever
conditioned. On the state vector the circuit reproduces the angle encoded in
k to 1e-15 and theta within epsilon (`tests/test_ft_rz.mojo`).

## Verification (`qumo.ft.verify`)

* `valid_clifford_implementation(semantic, impl, code, num_blocks, num_ancillas)`:
  reads C^dag P C from the rows of `qumo.sim.stabilizer.TableauSimulator`
  (it stores the inverse tableau) and checks that every stabilizer maps into
  the stabilizer group and that every logical X_j / Z_j maps to the expanded
  semantic image times a stabilizer *with the right sign*. This fixes the
  logical action up to global phase, like Guppy FT's Choi-state comparison,
  without preparing Choi states. Clean ancillas must return to |0>.
* `valid_stabilizer_state_preparation(semantic, impl, code, ...)`: impl may use
  measurements, flags and resets; on every accepted run (forced-zero branch
  plus random seeds) each expected generator has expectation exactly +1.
* `valid_magic_state_preparation(impl, code, theta)` (state vector,
  post-selected flags): stabilizers +1 and logical Bloch vector
  (cos theta, sin theta, 0).

Failures name the operator, e.g. the incorrect non-CSS [[4,2,2]] gadget of
Guppy FT's tests is reported as "the right operator with the wrong sign".

## Emulation (`qumo.ft.emulate`)

`run_encoded(prog, shots, seed, noise)` runs per-shot trajectories on the
tableau (Clifford programs, any width) or the state vector (T injection and
other non-Clifford programs, <= 26 qubits), applies Stim-style circuit-level
noise (`NoiseParams`, lanes excluded) and returns `LogicalCounts` after
post-selection. `logical_to_circuit(lp)` gives the semantics of a logical
program as an unencoded circuit, `reference_counts` samples the source
circuit, `error_rate_sweep` compares logical and physical error rates.
`qumo.ft.tools.resource_report_json` combines encoder statistics with
`qumo.exec.resources.estimate_resources` of the physical circuit.

Measured with `examples/ft/logical_error_sweep.mojo` (3-qubit deterministic
Clifford program, Steane, no EC cycles, flagged |0>_L preparations
post-selected, 20000 shots per point, uniform circuit-level noise p):

| p | unencoded error | Steane-encoded error | acceptance |
|---|---|---|---|
| 1e-3 | 8.95e-3 | 1.53e-3 | 0.981 |
| 2e-3 | 1.55e-2 | 3.64e-3 | 0.961 |
| 4e-3 | 3.45e-2 | 1.44e-2 | 0.925 |

The encoded error grows ~9.4x for a 4x larger p (distance-3 behaviour); the
unencoded one ~3.9x. On Iceberg ([[6,4,2]], `examples/ft/iceberg_program.mojo`,
p = 2e-3) discarding detected errors lowers the encoded error from 0.153 to
0.027 at 80% acceptance, close to the unencoded 0.030: rotations on (b, d_i)
pairs are not fault tolerant, so a single 2-qubit fault can be a logical error.

## AI-native surface

* MCP / CLI tools: `qumo_ft_codes`, `qumo_ft_encode` (with optional
  emulation next to the unencoded reference), `qumo_ft_verify`.
* JSON: `StabilizerCode.to_json()`, `Architecture.to_json()`,
  `EncodedProgram.to_json()`, `LogicalProgram.summary_json()`.
* Errors `QM-E720`..`QM-E730` with hints ([ERRORS.md](ERRORS.md)).

## Mapping to Guppy FT source files

| Guppy FT | QUMO |
|---|---|
| `code_def.py` (`StabilizerCode`) | `qumo/ft/code.mojo`, `qumo/ft/pauli.mojo` |
| `code/steane/primitives.py` | `qumo/ft/gadgets.mojo` (Steane part) |
| `code/steane/logical.py`, `code/iceberg/logical.py`, `extensions/*` | `qumo/ft/logical.mojo` |
| `code/steane/encode.py` (`SteaneBuilder`, `QECPolicy`, `STATE`) | `qumo/ft/arch.mojo`, `qumo/ft/implement.mojo` |
| `std/state_factory.py`, `std/_logical_block.py` | `Engine._prepare_rus`, factories, blocks in `qumo/ft/implement.mojo` |
| `encode/__init__.py`, `_compile.py`, `_implement_ops.py` | `qumo/ft/encode.mojo`, `qumo/ft/implement.mojo` |
| `verify/_verify.py`, `_expansion.py`, `_state_gen.py` | `qumo/ft/verify.mojo` |
| `decompose/_comparator_based_rz.py`, `_passes.py` | `qumo/ft/rz.mojo` |
| `globals.py` (`with_global`, `map_global`) | not needed: state is encode-time (`Engine`) |

## Limits

* Static IR: loops and RUS are unrolled; factories prefetch instead of
  queueing at run time; the classical lane emulates classical logic with a
  qubit (one extra qubit, no noise).
* Steane: transversal Clifford gates, one lookup decoder (no flag-aware or
  correlated decoding); EC cycles are teleportation-based (as Guppy FT).
  Non-Clifford gates only through injection (conditioned T is `QM-E726`).
* Iceberg: error detection only (post-selection); within-block and
  cross-block rotations are not fault tolerant (as Guppy FT's Iceberg ops);
  non-destructive single-slot measurement uses one ancilla parity.
* Static Rz search is exponential in the T budget (no gridsynth); the
  comparator circuit is exact but expensive once encoded (Rz(0.7) Rz(-0.7)
  with epsilon = 0.1 and 2 attempts: 104 logical T, 92 physical qubits,
  3029 two-qubit gates, 2q depth 804), so encoded comparator programs are
  characterised by resource reports, not simulated.
* State-vector emulation of encoded non-Clifford programs is limited to
  ~26 physical qubits (one Steane data block + one magic block + lane).
* Non-CSS codes are supported by the code library and the verifier;
  architectures (encode / decode) are Steane and Iceberg.
