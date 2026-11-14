"""OpenQASM 3.0 (+ 2.0 compatibility) <-> QUMO `Circuit`.

Why OpenQASM is QUMO's LLM-facing circuit format: it is the lingua franca of
Qiskit, Braket, TKET, Cirq and the IBM/IonQ/Quantinuum hardware APIs, so
language models have seen orders of magnitude more QASM than any Python
builder API. QUMO therefore accepts the dialects models actually emit
(QASM 2 `qreg/creg/measure ->/if(c==1)`, QASM 3 `qubit[n]/bit[n]/c[i] = measure`,
`input float theta;`, `gate` definitions, `inv @`/`ctrl @`/`pow(k) @`
modifiers, constant `for` loops, unicode `π`/`θ`) and emits one canonical
QASM 3 form that other toolchains read back.

Implementation
--------------
* Lexer: one pass over the UTF-8 bytes, no regex, no per-character String
  allocation. Tokens are POD (kind, byte span, line, numeric value); columns
  are computed lazily (UTF-8 aware) only when a diagnostic is rendered.
* Parser: recursive descent over the token array. Parameter expressions are
  evaluated eagerly to QUMO's affine `Param` (`coeff * input + offset`); a
  product/division of inputs or a mix of two inputs is rejected with a precise
  error instead of silently building an expression tree.
* User `gate` bodies are inlined by re-walking their token range with a fresh
  closed scope (params -> Param, qubit args -> global indices). Modifiers are
  applied with the IR functors `Circuit.inverse()/controlled()/power()`.
* Definitions of gates QUMO already knows natively (e.g. the `gate rzz(...)`
  blocks the emitter writes for non-`stdgates.inc` gates) are treated as
  declarations: the native, exact opcode is used. This keeps emit -> parse
  round trips bit-exact while staying portable to other tools.
* QUMO-only IR (noise channels, QEC detectors/observables, ticks, original
  parameter names) is carried in `#pragma qumo ...` lines that this parser
  reads back and other tools ignore.

Every error is `QM-E5xx: <message> (line L, col C). Hint: <fix>` followed by
the offending source line and a caret, so both humans and agents can repair
the input (`qumo.ai.diagnostics.diagnostic_from_error` parses it back into a
structured `Diagnostic`).

The implementation is split by stage (this file re-exports every name, so callers do not change):

* `qumo/ir/qasm_lexer.mojo`: tokens, diagnostic helpers, the lexer and the builtin gate table.
* `qumo/ir/qasm_parser.mojo`: the recursive-descent parser (`QasmParser`) and the public parsing API.
* `qumo/ir/qasm_emit.mojo`: OpenQASM 3 output for circuits and programs.
"""

from qumo.ir.qasm_lexer import (
    decimal_to_bits, bits_to_decimal, TK, P, Token, format_qasm_error, suggest_names, tokenize_qasm, G,
    builtin_gate_table)
from qumo.ir.qasm_parser import (
    Register, GateDef, Scope, MAX_SOURCE_ITEMS, NK, QasmParser, parse_qasm_with_warnings, parse_qasm,
    parse_qasm_program, load_any, load_any_with_warnings, load_qasm)
from qumo.ir.qasm_emit import (
    register_condition_value, format_angle, format_param, to_qasm, _float_text, can_imply,
    draw_program)

