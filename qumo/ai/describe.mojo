"""Machine-readable self-description of QUMO (gates, errors, capabilities)
and the agent-facing summary of a circuit.

Inspiration: PennyLane's device capability TOML files, Amazon Braket's
`DeviceCapabilities`, and the `llms.txt` convention - an agent should be able
to ask the toolkit what it supports instead of guessing from docs. Every
document is generated from the same metadata the compiler and simulators use
(`qumo.ir.gates` opcode tables, `qumo.ai.diagnostics.error_catalog`), so it
cannot drift from the implementation. Output is deterministic (fixed key
order, sorted maps) to keep agent transcripts diffable and cacheable.

Circuit and program JSON (`qumo.circuit.v1`, `qumo.program.v1`) are IR serialisation and
live in `qumo/ir/json.mojo`; `write_circuit_json`, `circuit_to_json`, `circuit_from_json`,
`write_param_json`, `write_program_json`, `program_to_json` and `program_from_json` are
re-exported here for one release (kept so `from qumo.ai.describe import ...` keeps working).
"""

from std.math import isinf, isnan, pi

from qumo.ir.gates import (
    Op, op_name, op_arity, op_num_params, op_is_unitary, op_is_noise, op_is_annotation,
    op_is_clifford, op_is_diagonal, op_is_self_inverse, op_inverse_kind, op_is_symmetric,
    op_description, matrix_1q, matrix_2q, M2, M4,
)
from qumo.ir.circuit import UNUSED_WIRES, Circuit, Instruction
from qumo.ir.param import Param
from qumo.ir.kraus import KrausChannel
from qumo.ir.classical import CFunc, CNodeOp, CNode, CTypeKind, CType, type_name
from qumo.ir.program import CReg, CVar, ForRangeField, MAX_NESTING, Program, StmtKind, Stmt, TapeBuilder, check_program
from qumo.ir.qasm import builtin_gate_table, format_param
from qumo.core.json import JsonWriter, JsonValue, JsonKind, parse_json
# circuit and program JSON moved to qumo.ir.json; re-exported here for one release
from qumo.ir.json import (
    write_param_json, write_circuit_json, circuit_to_json, circuit_from_json,
    write_program_json, program_to_json, program_from_json,
)
from qumo.ai.diagnostics import error_catalog, error_ranges, range_index_of
from qumo.compiler.pass_manager import PassId, pass_decl, pass_description, arg_kind_name, ArgForm
from std.os import getenv
from std.os.path import exists

comptime QUMO_VERSION = "0.1.0"


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------
def all_opcodes() -> List[Int]:
    var out = List[Int]()
    for k in range(Op.NUM_OPS):
        if op_name(k) != "unknown":
            out.append(k)
    return out^


def op_kind_name(k: Int) -> String:
    if op_is_unitary(k): return "gate"
    if k == Op.MEASURE: return "measurement"
    if k == Op.RESET: return "reset"
    if op_is_noise(k): return "noise"
    return "annotation"


def _param_names(k: Int) -> List[String]:
    var out = List[String]()
    if k == Op.U:
        out.append("theta")
        out.append("phi")
        out.append("lambda")
    elif k == Op.PAULI_CHANNEL_1:
        out.append("px")
        out.append("py")
        out.append("pz")
    elif op_is_noise(k):
        out.append("p")
    elif k == Op.P or k == Op.CP:
        out.append("lambda")
    elif op_num_params(k) == 1:
        out.append("theta")
    return out^


def gate_qasm_example(k: Int) -> String:
    """How to write the opcode in QUMO-flavoured OpenQASM 3."""
    if k == Op.MEASURE: return "c[0] = measure q[0];"
    if k == Op.RESET: return "reset q[0];"
    if k == Op.BARRIER: return "barrier q;"
    if k == Op.DETECTOR: return "#pragma qumo detector 0, 1"
    if k == Op.OBSERVABLE: return "#pragma qumo observable(0) 2"
    if k == Op.TICK: return "#pragma qumo tick"
    var s = String()
    if op_is_noise(k):
        s += "#pragma qumo "
    s += "U" if k == Op.U else op_name(k)
    var pn = _param_names(k)
    if len(pn) > 0:
        s += "("
        for i in range(len(pn)):
            if i > 0:
                s += ", "
            s += pn[i]
        s += ")"
    for i in range(op_arity(k)):
        s += " q[" if i == 0 else ", q["
        s += String(i) + "]"
    return s + ";"


def _in_stdgates(k: Int) -> Bool:
    return op_is_unitary(k) and not (
        k == Op.SXDG or k == Op.ISWAP or k == Op.ISWAPDG or k == Op.ECR
        or k == Op.RXX or k == Op.RYY or k == Op.RZZ or k == Op.CCZ
    )


def _write_m2(mut w: JsonWriter, m: M2):
    w.begin_array()
    for r in range(2):
        w.begin_array()
        for c in range(2):
            w.begin_array()
            w.number(m.re[r * 2 + c])
            w.number(m.im[r * 2 + c])
            w.end_array()
        w.end_array()
    w.end_array()


def _write_m4(mut w: JsonWriter, m: M4):
    w.begin_array()
    for r in range(4):
        w.begin_array()
        for c in range(4):
            w.begin_array()
            w.number(m.re[r * 4 + c])
            w.number(m.im[r * 4 + c])
            w.end_array()
        w.end_array()
    w.end_array()


def write_gate_json(mut w: JsonWriter, k: Int, aliases: List[String]):
    w.begin_object()
    w.kv_string("name", op_name(k))
    w.kv_int("opcode", k)
    w.kv_string("kind", op_kind_name(k))
    w.kv_int("num_qubits", op_arity(k))
    w.kv_int("num_params", op_num_params(k))
    w.kv_string_list("param_names", _param_names(k))
    if op_is_unitary(k):
        w.kv_bool("clifford", op_is_clifford(k))
        w.kv_bool("diagonal", op_is_diagonal(k))
        w.kv_bool("self_inverse", op_is_self_inverse(k))
        w.kv_bool("symmetric", op_is_symmetric(k))
        if op_num_params(k) > 0:
            w.kv_string("inverse", op_name(k) + " with negated parameters" if k != Op.U else "U(-theta, -lambda, -phi)")
        else:
            w.kv_string("inverse", op_name(op_inverse_kind(k)))
        w.kv_bool("in_stdgates_inc", _in_stdgates(k))
    w.kv_string_list("aliases", aliases)
    w.kv_string("qasm", gate_qasm_example(k))
    w.kv_string("description", op_description(k))
    if op_is_unitary(k) and op_num_params(k) == 0:
        if op_arity(k) == 1:
            w.key("matrix")
            _write_m2(w, matrix_1q(k))
        elif op_arity(k) == 2:
            w.key("matrix")
            _write_m4(w, matrix_2q(k))
    w.end_object()


def _aliases_of(k: Int, table_keys: List[String], table_vals: List[Int]) -> List[String]:
    var out = List[String]()
    var canon = op_name(k)
    for i in range(len(table_keys)):
        if table_vals[i] == k and table_keys[i] != canon:
            out.append(table_keys[i])
    sort(out)
    return out^


def describe_gates_json() -> String:
    """Every opcode with arity, params, algebraic flags, QASM syntax, matrix."""
    var t = builtin_gate_table()
    var keys = List[String]()
    var vals = List[Int]()
    for kv in t.items():
        keys.append(kv.key)
        vals.append(kv.value)
    var w = JsonWriter()
    w.begin_object()
    w.kv_string("schema", "qumo.gates.v1")
    w.kv_string("matrix_convention", "row-major, entries [re, im]; basis index = b(q0) + 2*b(q1) (little-endian, Qiskit order); for controlled gates q0 is the control")
    w.key("gates")
    w.begin_array()
    for k in all_opcodes():
        write_gate_json(w, k, _aliases_of(k, keys, vals))
    w.end_array()
    w.key("composite_aliases")
    w.begin_object()
    w.kv_string("u2(phi, lambda)", "U(pi/2, phi, lambda) (Qiskit semantics, no extra global phase)")
    w.kv_string("u3(theta, phi, lambda)", "U(theta, phi, lambda) (Qiskit semantics, no extra global phase)")
    w.kv_string("cu(theta, phi, lambda, gamma) c, t", "p(gamma) c; ctrl @ U(theta, phi, lambda) c, t;")
    w.kv_string("cu3(theta, phi, lambda) c, t", "ctrl @ U(theta, phi, lambda) c, t;")
    w.kv_string("csx c, t", "ctrl @ sx c, t;")
    w.kv_string("gphase(gamma)", "global phase e^{i gamma}")
    w.end_object()
    w.kv_string_list("modifiers", ["inv @", "ctrl @", "ctrl(n) @", "negctrl @", "pow(k) @ (integer k)"])
    w.end_object()
    return w.finish()


def describe_gate_json(name: String) raises -> String:
    """One gate by name or alias (QM-E913 with suggestions when unknown)."""
    var t = builtin_gate_table()
    var code = t.get(name)
    var k = -1
    if code and code.value() >= 0:
        k = code.value()
    else:
        for op in all_opcodes():
            if op_name(op) == name:
                k = op
    if k < 0:
        raise Error("QM-E913: unknown gate '" + name + "'. Hint: call describe with topic 'gates' for the full list.")
    var keys = List[String]()
    var vals = List[Int]()
    for kv in t.items():
        keys.append(kv.key)
        vals.append(kv.value)
    var w = JsonWriter()
    write_gate_json(w, k, _aliases_of(k, keys, vals))
    return w.finish()


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
def describe_errors_json() -> String:
    var w = JsonWriter()
    w.begin_object()
    w.kv_string("schema", "qumo.errors.v1")
    w.kv_string("format", "QM-Exxx: <message> (line L, col C). Hint: <fix>  followed by an optional source snippet; QM-Wxxx are warnings (lint, execution notes, parser)")
    w.key("ranges")
    w.begin_array()
    for r in error_ranges():
        w.begin_object()
        w.kv_string("range", r.code)
        w.kv_string("category", r.category)
        w.kv_string("title", r.title)
        w.kv_string("owner", r.owner)
        w.end_object()
    w.end_array()
    w.key("codes")
    w.begin_array()
    for e in error_catalog():
        w.begin_object()
        w.kv_string("code", e.code)
        w.kv_string("category", e.category)
        w.kv_string("severity", "warning" if e.code.startswith("QM-W") else "error")
        w.kv_string("meaning", e.meaning)
        w.kv_string("hint", e.hint)
        w.end_object()
    w.end_array()
    w.end_object()
    return w.finish()


def describe_errors_markdown() -> String:
    """docs/ERRORS.md, generated from `error_ranges()` + `error_catalog()`.

    tests/test_mcp.mojo compares it with the checked-in file and rewrites the
    file when QUMO_UPDATE_DOCS=1. Every code gets a `### QM-Exxx` heading, so
    `Diagnostic.docs` links (`docs/ERRORS.md#qm-e510`) resolve."""
    var ranges = error_ranges()
    var cat = error_catalog()
    var s = String("# QUMO error and warning codes\n\n")
    s += "<!-- GENERATED FILE, do not edit by hand. Source of truth: error_catalog() and\n"
    s += "     error_ranges() in qumo/ai/diagnostics.mojo. Regenerate with\n"
    s += "     QUMO_UPDATE_DOCS=1 .venv/bin/mojo run -I . tests/test_mcp.mojo\n"
    s += "     (or build/qumo tool qumo_describe '{\"topic\":\"errors\",\"format\":\"markdown\"}').\n"
    s += "     tests/test_mcp.mojo fails when this file is stale. -->\n\n"
    s += "Every failure in QUMO is reported with a **stable code**. Codes never change\n"
    s += "meaning; message wording may improve. Agents should branch on the code,\n"
    s += "show the message, and apply the hint.\n\n"
    s += "```\nQM-E510: unknown gate 'cnto' (line 6, col 1). Hint: did you mean 'cnot'?\n 6 | cnto q[0], q[1];\n   | ^~~~\n```\n\n"
    s += "* **Text form** (any `Error` raised by QUMO): `QM-Exxx: <message> [(line L, col C)]. [Hint: <fix>]`,\n"
    s += "  optionally followed by a source snippet with a caret.\n"
    s += "* **Structured form** (CLI `--json`, MCP `structuredContent.error`, `qumo_validate_qasm`):\n"
    s += "  `qumo.ai.diagnostics.diagnostic_from_error(text)` ->\n"
    s += "  `{\"code\", \"severity\", \"message\", \"hint\", \"line\", \"col\", \"snippet\", \"instruction\", \"docs\"}`\n"
    s += "  (`line`/`col`/`snippet` only for source errors, `instruction` only for lint findings).\n"
    s += "* Errors without a QUMO code (e.g. from the Mojo standard library) map to `QM-E999`.\n"
    s += "* `QM-Wxxx` are warnings: lint findings from `qumo.ai.diagnostics.lint()`, execution notes\n"
    s += "  in tool results (e.g. QM-W020, GPU fallback) and OpenQASM parser warnings. Severity is\n"
    s += "  `warning` or `info` (QM-W014 is reported with severity `error`).\n"
    s += "* QM-E95x (hardware plugins) and QM-E96x (benchmark harness) are raised by the Python tools\n"
    s += "  of this repository, with the same text form and JSON shape.\n\n"
    s += "Machine-readable catalog: `qumo describe errors --json`, or the MCP tool\n"
    s += "`qumo_describe {\"topic\": \"errors\"}` (single code: `{\"topic\": \"error\", \"name\": \"QM-E510\"}`).\n\n"
    s += "## Code ranges\n\n| Range | Category | Owner |\n|---|---|---|\n"
    for r in ranges:
        s += "| `" + r.code + "` | " + r.category + " | " + r.owner + " |\n"
    s += "\nA code belongs to the narrowest range that contains it (QM-E472 is in QM-E47x, not QM-E4xx).\n\n"
    s += "Adding a code: raise `Error(\"QM-Exxx: <message>. Hint: <fix>.\")` from your module\n"
    s += "(inside your range), add one `_e(c, ...)` line to `error_catalog()` in\n"
    s += "`qumo/ai/diagnostics.mojo`, then regenerate this file (command in the header comment).\n"
    s += "`make test` fails while a raised code is missing from the catalogue or this file is stale.\n"
    for ri in range(len(ranges)):
        var r = ranges[ri].copy()
        s += "\n## " + r.code + " — " + r.title + "\n"
        if r.note.byte_length() > 0:
            s += "\n" + r.note + "\n"
        for e in cat:
            if range_index_of(e.code, ranges) != ri:
                continue
            s += "\n### " + e.code + "\n\n**" + e.meaning + "**  \nFix: " + e.hint + "\n"
    s += _agent_errors_markdown()
    return s^


def _agent_errors_markdown() -> String:
    """The `QA-E1xx` section: errors of the native agent harness (agents/mojo, qumo-agent).
    They are not QUMO diagnostics (no catalogue entry, no JSON shape); the texts are in
    agents/mojo/DESIGN.md Appendix B.2."""
    var s = String("\n## QA-E1xx — Agent harness (agents/mojo)\n\n")
    s += "Raised by `build/qumo-agent` (the native Mojo agents), in the text form\n"
    s += "`QA-Ennn: <message>. Hint: <fix>`. Exit code 3 for `QA-E140` and `QA-E105`, 1 for the rest.\n\n"
    s += "| Code | Raised when | Fix |\n|---|---|---|\n"
    s += "| `QA-E101` | The model endpoint answered non-2xx after the retries (`model request failed: HTTP <status> <reason>: <message>`). | Check the model id, the key and the endpoint in `agents/agents.json`. |\n"
    s += "| `QA-E102` | The prompt does not fit the model's context (HTTP 400/413/422 with an overflow marker) and summarization could not recover. | Shorten the task, or choose a model with a longer context. |\n"
    s += "| `QA-E103` | The model answered 200 with an error body, or without `choices`. | Retry; if it persists, try another model. |\n"
    s += "| `QA-E104` | Network failure: DNS, connect, TLS, timeout, a redirect from https to http, a truncated or oversized body. | Check the network; `build/qumo-agent doctor`. |\n"
    s += "| `QA-E105` | OpenSSL 3 or the CA certificates were not found (or lack a needed symbol). | Install OpenSSL 3, or set `SSL_CERT_FILE` / `QUMO_LIBSSL`. |\n"
    s += "| `QA-E110` | The recursion limit was reached without a final answer. | Raise `--recursion-limit`, or split the task. |\n"
    s += "| `QA-E120` | A subagent failed (`subagent <type> failed: ...`), or a subagent request or result was malformed. | Read the child's log on stderr; run with `--parallel 1` to see it inline. |\n"
    s += "| `QA-E130` | MCP transport or protocol error: an error response, no answer within the timeout, or the server closed the connection. | Rebuild `build/qumo` (`make cli`); check the server's stderr. |\n"
    s += "| `QA-E140` | Configuration: unknown agent, a key variable not set, `build/qumo` or a file not found, or a stale `agents/prompts/` export. | Follow the hint; `python -m qumo_agents export-prompts` refreshes the export. |\n"
    s += "| `QA-E150` | A generated tool was refused (not reviewed, hash mismatch, malformed manifest entry, unknown tool). | `qumo-agent tools verify`, then `tools review NAME`. |\n"
    s += "| `QA-E160` | Process or file-system failure (`[Errno N] ...`), or a child closed its input. | Check the path and permissions. |\n"
    s += "| `QA-E161` | A line from a child or MCP server was longer than the limit. | The peer is misbehaving; check its output. |\n"
    s += "| `QA-E162` | This libc has no `posix_spawn` chdir action. | Use glibc 2.29 or newer. |\n"
    s += "| `QA-E163` | No OS sandbox is available, or OS confinement failed to start. | Use macOS (`sandbox-exec`) or Linux with Landlock; a human can pass `--allow-unsandboxed-tool-tests`. |\n"
    s += "| `QA-E170` | The `--replay` file is unreadable or exhausted. | Add responses to the file, or drop `--replay`. |\n"
    return s^


def describe_error_json(code: String) raises -> String:
    var c = code.upper()
    for e in error_catalog():
        if e.code == c:
            var w = JsonWriter()
            w.begin_object()
            w.kv_string("code", e.code)
            w.kv_string("category", e.category)
            w.kv_string("meaning", e.meaning)
            w.kv_string("hint", e.hint)
            w.end_object()
            return w.finish()
    raise Error("QM-E913: unknown error code '" + code + "'. Hint: call describe with topic 'errors' for the catalog.")


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------
def describe_capabilities_json(tools: List[String], commands: List[String] = List[String]()) -> String:
    var w = JsonWriter()
    w.begin_object()
    w.kv_string("schema", "qumo.capabilities.v1")
    w.kv_string("name", "qumo")
    w.kv_string("version", QUMO_VERSION)
    w.kv_string("description", "QUMO: a fast, AI-native quantum software stack in Mojo (IR, OpenQASM 2/3, state-vector simulation, compiler, QEC, algorithms).")
    w.key("conventions")
    w.begin_object()
    w.kv_string("qubit_order", "little-endian: bit k of a basis-state index is qubit k")
    w.kv_string("bitstrings", "Qiskit-style: leftmost character is the highest classical bit (c[n-1] ... c[0])")
    w.kv_string("angles", "radians; U(theta, phi, lambda) follows OpenQASM 3")
    w.kv_string("determinism", "every random process takes an explicit seed (default 1234); identical inputs give identical counts")
    w.kv_string("parameters", "affine symbolic parameters coeff*input + offset ('input float theta;'); bind with 'params' as an array (declaration order) or a name->value object")
    w.kv_string("observables", "Pauli sums such as '0.5*Z0 Z1 - 1.2*X0 + 0.3' or dense labels '0.5*ZZI' (leftmost = highest qubit)")
    w.end_object()
    w.key("simulators")
    w.begin_array()
    w.begin_object()
    w.kv_string("name", "statevector")
    w.kv_string_list("methods", ["statevector:static", "statevector:trajectories", "statevector:exact"])
    w.kv_string("memory", "16 * 2^n bytes (n=24: 256 MiB, n=28: 4 GiB, n=30: 16 GiB)")
    w.kv_int("max_qubits_default", 28)
    w.kv_string("selection", "static circuits: one evolution + sampling; mid-circuit measurement, reset, feed-forward or noise: per-shot trajectories")
    w.kv_string_list("features", ["all QUMO gates", "measure", "reset", "feed-forward on a bit, a register or an && chain of bits", "Pauli noise channels", "Kraus channels (trajectories)", "symbolic parameters", "exact Pauli expectation values", "SIMD + multithreaded kernels"])
    w.end_object()
    w.begin_object()
    w.kv_string("name", "ptsbe")
    w.kv_string_list("methods", ["statevector:ptsbe"])
    w.kv_string("memory", "16 * 2^n bytes per trie level")
    w.kv_string("selection", "opt-in, sample_ptsbe: static circuits with Pauli noise, one simulation per distinct error pattern; emulate() uses it")
    w.kv_string_list("features", ["same distribution as trajectories", "cost follows distinct patterns, not shots"])
    w.end_object()
    w.begin_object()
    w.kv_string("name", "density_matrix")
    w.kv_string_list("methods", ["density_matrix"])
    w.kv_string("memory", "16 * 4^n bytes (n=12: 256 MiB, n=15: 16 GiB, the limit)")
    w.kv_int("max_qubits_default", 15)
    w.kv_string("selection", "opt-in: density_evolve, sample_density, MCP qumo_density")
    w.kv_string_list("features", ["exact noise", "Kraus channels", "reset", "non-selective measurement", "no classical conditions"])
    w.end_object()
    w.begin_object()
    w.kv_string("name", "stabilizer_branching")
    w.kv_string_list("methods", ["stabilizer:branching"])
    w.kv_string("memory", "at most 2^20 branches")
    w.kv_int("max_qubits_default", 512)
    w.kv_string("selection", "opt-in: sample_branching; qumo.ft run_physical picks it past 26 qubits")
    w.kv_string_list("features", ["cost follows non-Clifford gates, not qubits", "feed-forward", "Pauli noise"])
    w.end_object()
    w.end_array()
    w.key("qasm")
    w.begin_object()
    w.kv_string_list("input", ["OpenQASM 2.0", "OpenQASM 3.0"])
    w.kv_string("output", "OpenQASM 3.0 (stdgates.inc + exact gate definitions for non-standard natives); OpenQASM 2.0 with qumo_convert to='qasm2' (original qelib1.inc + exact definitions; bound parameters; loads in Qiskit strict/legacy, pytket, Cirq)")
    w.kv_string_list("supported", [
        "qubit[n]/qreg, bit[n]/creg (multiple registers are flattened)",
        "all QUMO gates and aliases (cnot, toffoli, u1/u2/u3, cu1/cphase, cu, cu3, csx, U, CX, phase, id)",
        "constant expressions: + - * / % ** pi tau euler sin cos tan arcsin arccos arctan exp ln sqrt abs floor ceil",
        "input float/angle parameters with affine expressions",
        "gate definitions (inlined) and modifiers inv @, ctrl(n) @, negctrl(n) @, pow(k) @ (any number of controls, any unitary gate)",
        "box and box[duration] (a barrier on each side, duration ignored), @annotation lines (ignored, QM-W052)",
        "switch on a bit register, an int mirroring one, or a constant: case lists, default, Qiskit's older 'case 0: {} break;'",
        "register broadcast, slices q[a:b] (inclusive), index sets q[{0,2}], negative indices, $n physical qubits",
        "measure q -> c; c = measure q; c[i] = measure q[j]; reset; barrier; gphase",
        "if (c[i]) / if (c[i] == v) / if (!c[i]) with else; if (creg == v) on a whole register of any size; if (c[i] == v && c[j] == w), no else on the last two",
        "const declarations and for loops over constant ranges/sets (unrolled)",
        "#pragma qumo: noise channels, detector, observable, tick, param",
        "programs (parse_qasm_program, load_any): while, break, continue, for and switch at run time, classical variables bit, bool, int[n], uint[n], float, angle[n], assignment, conditions on any value, runtime gate angles",
    ])
    w.kv_string_list("unsupported", [
        "def/extern subroutines", "hexadecimal, octal and binary integer literals",
        "opaque gates", "pulse-level cal/defcal", "delay timing (delay ignored)",
        "programs on hardware: the plugins refuse them", "detectors in a program with runtime regions (QM-E586)",
    ])
    w.end_object()
    w.key("error_codes")
    w.begin_array()
    for r in error_ranges():
        w.begin_object()
        w.kv_string("range", r.code)
        w.kv_string("owner", r.owner)
        w.end_object()
    w.end_array()
    w.kv_string_list("mcp_tools", tools)
    if len(commands) > 0:
        w.kv_string_list("cli_commands", commands)
    w.kv_string("mcp_protocol", "2025-06-18 (stdio, newline-delimited JSON-RPC 2.0)")
    w.end_object()
    return w.finish()


# ---------------------------------------------------------------------------
# Compiler passes and AI-transpiler models
# ---------------------------------------------------------------------------
def describe_passes_markdown() -> String:
    """docs/PASSES.md, generated from the pass table (`pass_decl` in
    qumo/compiler/pass_manager.mojo): every pass with its aliases and
    arguments. tests/test_mcp.mojo compares it with the checked-in file and
    rewrites the file when QUMO_UPDATE_DOCS=1."""
    var s = String("# QUMO pipeline passes: names, aliases and arguments\n\n")
    s += "<!-- GENERATED FILE, do not edit by hand. Source of truth: pass_decl() in\n"
    s += "     qumo/compiler/pass_manager.mojo. Regenerate with\n"
    s += "     QUMO_UPDATE_DOCS=1 .venv/bin/mojo run -I . tests/test_mcp.mojo -->\n\n"
    s += "What each pass does and where it comes from is in [COMPILER.md](COMPILER.md). This page is the reference of\n"
    s += "what a pipeline string may hold. An alias is another name for the same pass; `canonical()` prints the\n"
    s += "canonical one. An argument is written `key=value`; where the table says so the value alone is enough.\n"
    s += "When an argument is written twice the last one counts. A flag takes `0|1|true|false|on|off`.\n\n"
    s += "| Pass | Aliases | Arguments |\n|---|---|---|\n"
    for i in range(PassId.NUM):
        var d = pass_decl(i)
        var al = String()
        for a in d.alias_list():
            al += (", " if al.byte_length() > 0 else "") + "`" + a + "`"
        var names = String()
        for a in d.args:
            names += (", " if names.byte_length() > 0 else "") + "`" + a.name + "`"
        s += "| `" + d.name + "` | " + al + " | " + (names if names.byte_length() > 0 else String("none")) + " |\n"
    for i in range(PassId.NUM):
        var d = pass_decl(i)
        if len(d.args) == 0:
            continue
        s += "\n## " + d.name + "\n\n" + d.summary + ".\n\n"
        s += "| Argument | Written | Values | Default | Notes |\n|---|---|---|---|---|\n"
        for a in d.args:
            var written = String("`") + a.name + "=...`"
            if a.form == ArgForm.KEY_OR_BARE:
                written += " or the value alone"
            elif a.form == ArgForm.BARE:
                written = "the value alone"
            s += "| `" + a.name + "` | " + written + " | `" + a.values().replace("|", "\\|") + "` | "
            s += (("`" + a.default + "`") if a.default.byte_length() > 0 else String("")) + " | " + a.help.replace("|", "\\|") + " |\n"
    return s


def describe_passes_json() -> String:
    """Pass catalog (`qumo.passes.v1`): every pass of the text pipeline
    language with its description, aliases and arguments (from the pass table), the AI
    synthesis passes flagged, the
    preset families and the AI-transpiler error codes."""
    var w = JsonWriter()
    w.begin_object()
    w.kv_string("schema", "qumo.passes.v1")
    w.kv_string("grammar", "pipeline := item ((','|';') item)*; item := name [ '(' arg (',' arg)* ')' ]; arg := word | key=value | item")
    w.key("passes")
    w.begin_array()
    for i in range(PassId.NUM):
        var d = pass_decl(i)
        w.begin_object()
        w.kv_string("name", d.name)
        w.kv_string("description", pass_description(i))
        w.kv_bool("ai_synthesis", d.block_kind >= 0)
        w.kv_string_list("aliases", d.alias_list())
        w.key("args")
        w.begin_array()
        for a in d.args:
            w.begin_object()
            w.kv_string("name", a.name)
            w.kv_string("type", arg_kind_name(a.kind))
            w.kv_string("written", "key=value" if a.form == ArgForm.KEY else ("key=value or value" if a.form == ArgForm.KEY_OR_BARE else "value"))
            w.kv_string("values", a.values())
            w.kv_string("default", a.default)
            w.kv_string("help", a.help)
            w.end_object()
        w.end_array()
        w.end_object()
    w.end_array()
    w.key("route_ai_keys")
    w.begin_object()
    w.kv_string("policy", "off (default) | auto | <model id>: learned-policy layout + routing, kept only when verified and cheaper than SABRE")
    w.kv_string("ai_level", "1..3 effort (IBM AIRouting table: 8/2/2, 16/16/8, 32/16/8 runs/iterations/repetitions)")
    w.kv_string("layout_mode", "optimize | improve | keep (a user initial layout turns optimize into improve)")
    w.kv_string("model_dir", "directory with qumo.rlroute.v0 models (default: $QUMO_MODEL_DIR, then ./models/ai_transpiler)")
    w.end_object()
    w.key("presets")
    w.begin_object()
    w.kv_string("transpile(level=0..3)", "heuristic presets O0-O3 (O3 is a portfolio with O2)")
    w.kv_string("transpile_ai(level=1)", "O1 + learned-policy routing")
    w.kv_string("transpile_ai(level=2)", "O2 + policy routing + post-route linear_synth (forward, backward) + perm_synth")
    w.kv_string("transpile_ai(level=3)", "O3 + pre-route clifford_synth + policy routing + post-route fixpoint(linear_synth, clifford_synth, perm_synth, pauli_network); portfolio with transpile(level=3)")
    w.end_object()
    w.kv_string_list("ai_error_codes", ["QM-E616", "QM-E617", "QM-E618", "QM-E619", "QM-E620"])
    w.end_object()
    return w.finish()


def ai_model_search_dirs(explicit: String) -> List[String]:
    """Model directories in lookup order: explicit `model_dir`, then
    $QUMO_MODEL_DIR, then ./models/ai_transpiler (same order as
    `qumo.compiler.policy.model_search_dirs`)."""
    var out = List[String]()
    if explicit.byte_length() > 0:
        out.append(explicit)
    var env = getenv("QUMO_MODEL_DIR")
    if env.byte_length() > 0:
        out.append(env)
    out.append("models/ai_transpiler")
    return out^


def _str_field(v: JsonValue, k1: String, k2: String) -> String:
    try:
        var a = v.get(k1)
        if not a.is_defined():
            a = v.get(k2)
        if a.is_string():
            return a.as_string()
        if a.is_number():
            return String(a.as_int())
    except:
        pass
    return String()


def _write_model(mut w: JsonWriter, e: JsonValue, fallback_id: String, family: String, index: String):
    w.begin_object()
    var id = _str_field(e, "id", "name")
    w.kv_string("id", id if id.byte_length() > 0 else fallback_id)
    var kind = _str_field(e, "kind", "env")
    w.kv_string("kind", kind if kind.byte_length() > 0 else family)
    var n = _str_field(e, "num_qubits", "n")
    if n.byte_length() > 0:
        try:
            w.kv_int("num_qubits", Int(n))
        except:
            w.kv_string("num_qubits", n)
    var topo = _str_field(e, "topology", "graph")
    if topo.byte_length() > 0:
        w.kv_string("topology", topo)
    var fmt = _str_field(e, "format", "schema")
    if fmt.byte_length() > 0:
        w.kv_string("format", fmt)
    w.kv_string("index", index)
    w.kv_raw("entry", e.to_json())
    w.end_object()


def describe_models_json(model_dir: String = "") -> String:
    """Installed AI-transpiler models (`qumo.models.v1`), read from the
    INDEX.json registries (<dir>/INDEX.json, <dir>/synth/INDEX.json,
    <dir>/route/INDEX.json) of every search directory. Never raises: an
    unreadable registry is reported under `errors`."""
    var w = JsonWriter()
    w.begin_object()
    w.kv_string("schema", "qumo.models.v1")
    var dirs = ai_model_search_dirs(model_dir)
    w.kv_string_list("search_dirs", dirs)
    var seen = List[String]()
    var errors = List[String]()
    var count = 0
    w.key("models")
    w.begin_array()
    for d in dirs:
        for sub in ["", "/synth", "/route"]:
            var path = d + String(sub) + "/INDEX.json"
            var dup = False
            for p in seen:
                if p == path:
                    dup = True
            if dup or not exists(path):
                continue
            seen.append(path)
            var family = "synth" if String(sub) == "/synth" else ("route" if String(sub) == "/route" else "")
            try:
                var text: String
                with open(path, "r") as f:
                    text = f.read()
                var root = parse_json(text)
                var lst = root.get("models")
                if lst.is_array():
                    for i in range(len(lst)):
                        _write_model(w, lst.at(i), String(i), family, path)
                        count += 1
                elif lst.is_object():
                    for i in range(len(lst)):
                        _write_model(w, lst.at(i), lst.key_at(i), family, path)
                        count += 1
                elif root.is_array():
                    for i in range(len(root)):
                        _write_model(w, root.at(i), String(i), family, path)
                        count += 1
            except e:
                errors.append(path + ": " + String(e))
    w.end_array()
    w.kv_int("count", count)
    w.kv_string_list("errors", errors)
    w.kv_string("hint", "verify integrity with 'make models-verify'; retrain with 'make train-ai' (docs/AI_TRANSPILER.md)")
    w.end_object()
    return w.finish()


# ---------------------------------------------------------------------------
# Circuit summary (circuit JSON itself lives in qumo.ir.json)
# ---------------------------------------------------------------------------
def write_circuit_summary(circuit: Circuit, mut w: JsonWriter):
    var counts = List[Int](length=Op.NUM_OPS, fill=0)
    for g in circuit.ops:
        if g.kind >= 0 and g.kind < Op.NUM_OPS:
            counts[g.kind] += 1
    var names = List[String]()
    for k in range(Op.NUM_OPS):
        if counts[k] > 0:
            names.append(op_name(k))
    sort(names)
    w.begin_object()
    w.kv_string("name", circuit.name)
    w.kv_int("num_qubits", circuit.num_qubits)
    w.kv_int("num_clbits", circuit.num_clbits)
    w.kv_int("num_instructions", len(circuit.ops))
    w.kv_int("num_gates", circuit.num_unitary_gates())
    w.kv_int("depth", circuit.depth())
    w.kv_int("two_qubit_gates", circuit.num_2q_gates())
    w.kv_int("t_count", circuit.t_count())
    w.key("ops")
    w.begin_object()
    for nm in names:
        for k in range(Op.NUM_OPS):
            if op_name(k) == nm:
                w.kv_int(nm, counts[k])
    w.end_object()
    w.kv_int("num_measurements", circuit.num_measurements)
    w.kv_bool("is_clifford", circuit.is_clifford())
    w.kv_bool("is_dynamic", circuit.is_dynamic())
    w.kv_bool("has_noise", circuit.has_noise())
    w.kv_bool("is_parametric", circuit.is_parametric())
    w.kv_string_list("params", circuit.param_names)
    w.kv_float("global_phase", circuit.global_phase)
    w.kv_int("num_detectors", circuit.num_detectors)
    w.kv_int("num_observables", circuit.num_observables)
    if circuit.num_qubits < 60:
        w.kv_float("statevector_bytes", Float64(1 << circuit.num_qubits) * 16.0)
    var sim = String("statevector")
    if circuit.num_qubits > 28 and circuit.is_clifford():
        sim = "stabilizer (qumo.sim.stabilizer)"
    elif circuit.num_qubits > 28:
        sim = "none locally (exceeds dense state-vector memory)"
    w.kv_string("recommended_simulator", sim)
    w.end_object()


def circuit_summary_json(circuit: Circuit) -> String:
    var w = JsonWriter()
    write_circuit_summary(circuit, w)
    return w.finish()
