"""Machine-readable self-description of QUMO (gates, errors, capabilities)
and JSON views of circuits.

Inspiration: PennyLane's device capability TOML files, Amazon Braket's
`DeviceCapabilities`, and the `llms.txt` convention - an agent should be able
to ask the toolkit what it supports instead of guessing from docs. Every
document is generated from the same metadata the compiler and simulators use
(`qumo.ir.gates` opcode tables, `qumo.ai.diagnostics.error_catalog`), so it
cannot drift from the implementation. Output is deterministic (fixed key
order, sorted maps) to keep agent transcripts diffable and cacheable.

Circuit JSON (`format: "qumo.circuit.v1"`) round-trips through
`circuit_from_json`, giving agents a structured alternative to QASM text.
Program JSON (`format: "qumo.program.v1"`) wraps it, with the circuit as the
arena of quantum ops, and round-trips through `program_from_json`.
"""

from std.math import isinf, isnan, pi

from qumo.ir.gates import (
    Op, op_name, op_arity, op_num_params, op_is_unitary, op_is_noise, op_is_annotation,
    op_is_clifford, op_is_diagonal, op_is_self_inverse, op_inverse_kind, op_is_symmetric,
    op_description, matrix_1q, matrix_2q, M2, M4,
)
from qumo.ir.circuit import Circuit, Instruction
from qumo.ir.param import Param
from qumo.ir.kraus import KrausChannel
from qumo.ir.classical import CF, CN, CNode, CT, CType, type_name
from qumo.ir.program import CReg, CVar, FORR, Program, SK, Stmt, TapeBuilder, check_program
from qumo.ir.qasm import builtin_gate_table, format_param
from qumo.ai.json import JsonWriter, JsonValue, JsonKind, parse_json
from qumo.ai.diagnostics import error_catalog, error_ranges, range_index_of
from qumo.compiler.pass_manager import PassId, pass_name, pass_description, is_synth_pass
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
def describe_passes_json() -> String:
    """Pass catalog (`qumo.passes.v1`): every pass of the text pipeline
    language with its description, the AI synthesis passes flagged, the
    preset families and the AI-transpiler error codes."""
    var w = JsonWriter()
    w.begin_object()
    w.kv_string("schema", "qumo.passes.v1")
    w.kv_string("grammar", "pipeline := item ((','|';') item)*; item := name [ '(' arg (',' arg)* ')' ]; arg := word | key=value | item")
    w.key("passes")
    w.begin_array()
    for i in range(PassId.NUM):
        w.begin_object()
        w.kv_string("name", pass_name(i))
        w.kv_string("description", pass_description(i))
        w.kv_bool("ai_synthesis", is_synth_pass(i))
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
# Circuits
# ---------------------------------------------------------------------------
def _kv_number(mut w: JsonWriter, key: StringSlice, v: Float64):
    """JSON has no NaN or infinity, and an affine coefficient that overflowed is one."""
    if isnan(v) or isinf(v):
        w.kv_string(key, "nan" if isnan(v) else ("inf" if v > 0.0 else "-inf"))
    else:
        w.kv_float(key, v)


def _read_number(v: JsonValue, key: String, default: Float64) raises -> Float64:
    """Inverse of `_kv_number`."""
    var x = v.get(key)
    if x.is_string():
        var s = x.as_string()
        if s == "nan" or s == "inf" or s == "-inf":
            return Float64(s)
    return v.get_float(key, default)


def write_param_json(mut w: JsonWriter, p: Param, names: List[String]):
    if not p.is_symbolic():
        w.number(p.offset)
        return
    w.begin_object()
    w.kv_string("expr", format_param(p, names))
    w.kv_string("param", names[p.index] if p.index < len(names) else "$" + String(p.index))
    _kv_number(w, "coeff", p.coeff)
    _kv_number(w, "offset", p.offset)
    w.end_object()


def _write_condition_json(mut w: JsonWriter, circuit: Circuit, g: Instruction):
    """`condition`: {clbit, value} tests one bit; a register condition adds
    `bits` and `values`, every one of which must hold."""
    if g.cond < 0:
        return
    w.key("condition")
    w.begin_object()
    w.kv_int("clbit", g.cond)
    w.kv_int("value", g.cond_val)
    if g.cond_reg >= 0:
        var n = circuit.cond_table[g.cond_reg]
        w.key("bits")
        w.begin_array()
        for j in range(n):
            w.integer(circuit.cond_table[g.cond_reg + 1 + 2 * j])
        w.end_array()
        w.key("values")
        w.begin_array()
        for j in range(n):
            w.integer(circuit.cond_table[g.cond_reg + 2 + 2 * j])
        w.end_array()
    w.end_object()


def _read_condition_json(mut c: Circuit, g: JsonValue) raises:
    """Condition of the instruction just appended, as `_write_condition_json` wrote it."""
    var cond = g.get("condition")
    if not cond.is_object():
        return
    var bv = cond.get("bits")
    if not bv.is_defined():
        c.c_if(cond.get_int("clbit", 0), cond.get_int("value", 1))
        return
    var vv = cond.get("values")
    var bits = List[Int]()
    var vals = List[Int]()
    for j in range(len(bv)):
        bits.append(bv.at(j).as_int())
    for j in range(len(vv)):
        vals.append(vv.at(j).as_int())
    c.c_if_register(bits, vals)


def write_circuit_json(circuit: Circuit, mut w: JsonWriter):
    w.begin_object()
    w.kv_string("format", "qumo.circuit.v1")
    w.kv_string("name", circuit.name)
    w.kv_int("num_qubits", circuit.num_qubits)
    w.kv_int("num_clbits", circuit.num_clbits)
    w.kv_float("global_phase", circuit.global_phase)
    w.kv_string_list("params", circuit.param_names)
    w.key("instructions")
    w.begin_array()
    for g in circuit.ops:
        var k = g.kind
        w.begin_object()
        w.kv_string("op", op_name(k))
        if k == Op.DETECTOR or k == Op.OBSERVABLE:
            if k == Op.OBSERVABLE:
                w.kv_int("index", g.q2)
            w.key("records")
            w.begin_array()
            for j in range(g.q1):
                w.integer(circuit.records[g.q0 + j])
            w.end_array()
            w.end_object()
            continue
        if k == Op.CORRELATED_ERROR:
            w.kv_float("probability", g.p0.offset)
            w.kv_bool("else", g.q2 != 0)
            w.key("paulis")
            w.begin_array()
            for j in range(g.q1):
                w.begin_array()
                w.integer(circuit.pauli_table[g.q0 + 2 * j])
                w.integer(circuit.pauli_table[g.q0 + 2 * j + 1])
                w.end_array()
            w.end_array()
            _write_condition_json(w, circuit, g)
            w.end_object()
            continue
        var a = op_arity(k)
        if a > 0:
            w.key("qubits")
            w.begin_array()
            for j in range(a):
                w.integer(g.qubit(j))
            w.end_array()
        if k == Op.PAULI_CHANNEL_2:
            w.key("probabilities")
            w.begin_array()
            for j in range(15):
                w.number(circuit.noise_table[Int(g.p0.offset) + j])
            w.end_array()
        if k == Op.KRAUS1 or k == Op.KRAUS2:
            ref ch = circuit.kraus[Int(g.p0.offset)]
            w.kv_string("channel", ch.name)
            w.key("matrices")
            w.begin_array()
            for m in range(ch.count()):
                w.begin_array()
                for j in range(len(ch.re[m])):
                    w.number(ch.re[m][j])
                    w.number(ch.im[m][j])
                w.end_array()
            w.end_array()
        var np = op_num_params(k)
        if np > 0:
            w.key("params")
            w.begin_array()
            for j in range(np):
                write_param_json(w, g.param(j), circuit.param_names)
            w.end_array()
        if k == Op.MEASURE:
            w.kv_int("clbit", g.clbit)
        _write_condition_json(w, circuit, g)
        w.end_object()
    w.end_array()
    w.end_object()


def circuit_to_json(circuit: Circuit) -> String:
    var w = JsonWriter()
    write_circuit_json(circuit, w)
    return w.finish()


def _path_err(path: String, msg: String, hint: String) -> Error:
    return Error("QM-E911: " + path + ": " + msg + ". Hint: " + hint)


def circuit_from_json(v: JsonValue) raises -> Circuit:
    """Inverse of `circuit_to_json` (format qumo.circuit.v1). Gate names may
    be any QASM alias; params are numbers or {param, coeff, offset} objects."""
    if not v.is_object():
        raise _path_err("circuit", "expected an object", "pass the output of circuit_to_json / qumo_convert(to='json')")
    var nq = v.get_int("num_qubits", -1)
    if nq < 0:
        raise Error("QM-E910: circuit.num_qubits is required. Hint: e.g. {\"num_qubits\": 2, \"instructions\": [...]}.")
    var c = Circuit(nq, v.get_int("num_clbits", 0), v.get_string("name", "circuit"))
    c.global_phase = v.get_float("global_phase", 0.0)
    var ps = v.get("params")
    for i in range(len(ps)):
        _ = c.parameter(ps.at(i).as_string())
    var table = builtin_gate_table()
    var ins = v.get("instructions")
    if ins.is_defined() and not ins.is_array():
        raise _path_err("circuit.instructions", "expected an array", "list instruction objects like {\"op\": \"h\", \"qubits\": [0]}")
    for i in range(len(ins)):
        var path = "instructions[" + String(i) + "]"
        var g = ins.at(i)
        var nm = g.get_string("op", "")
        var k: Int
        if nm == "measure": k = Op.MEASURE
        elif nm == "reset": k = Op.RESET
        elif nm == "barrier": k = Op.BARRIER
        elif nm == "tick": k = Op.TICK
        elif nm == "detector": k = Op.DETECTOR
        elif nm == "observable": k = Op.OBSERVABLE
        elif nm == "correlated_error": k = Op.CORRELATED_ERROR
        elif nm == "pauli_channel_2": k = Op.PAULI_CHANNEL_2
        elif nm == "kraus1": k = Op.KRAUS1
        elif nm == "kraus2": k = Op.KRAUS2
        else:
            var code = table.get(nm)
            if not code or code.value() < 0:
                raise Error("QM-E913: " + path + ": unknown op '" + nm + "'. Hint: use a gate name from describe(gates); composite aliases (u2, u3, cu, csx) are QASM-only.")
            k = code.value()
        if k == Op.DETECTOR or k == Op.OBSERVABLE:
            var recs = List[Int]()
            var rv = g.get("records")
            for j in range(len(rv)):
                recs.append(rv.at(j).as_int())
            if k == Op.DETECTOR:
                c.detector(recs)
            else:
                c.observable(g.get_int("index", 0), recs)
            continue
        if k == Op.BARRIER or k == Op.TICK:
            c.add(k, -1)
            continue
        if k == Op.CORRELATED_ERROR:
            var eq = List[Int]()
            var el = List[Int]()
            var pv = g.get("paulis")
            for j in range(len(pv)):
                eq.append(pv.at(j).at(0).as_int())
                el.append(pv.at(j).at(1).as_int())
            c.correlated_error(g.get_float("probability", 0.0), eq, el, g.get_bool("else", False))
            _read_condition_json(c, g)
            continue
        var qv = g.get("qubits")
        var a = op_arity(k)
        if len(qv) != a:
            raise _path_err(path + ".qubits", "'" + nm + "' needs " + String(a) + " qubit(s), got " + String(len(qv)), "e.g. \"qubits\": [0, 1] for two-qubit gates")
        var qs = List[Int]()
        for j in range(a):
            qs.append(qv.at(j).as_int())
        if k == Op.PAULI_CHANNEL_2:
            var probs = List[Float64]()
            var pv = g.get("probabilities")
            for j in range(len(pv)):
                probs.append(pv.at(j).as_float())
            c.pauli_channel_2(probs, qs[0], qs[1])
        elif k == Op.KRAUS1 or k == Op.KRAUS2:
            var ch = KrausChannel(g.get_string("channel", "kraus"), a)
            var mv = g.get("matrices")
            for m in range(len(mv)):
                var re = List[Float64]()
                var im = List[Float64]()
                var row = mv.at(m)
                for j in range(0, len(row), 2):
                    re.append(row.at(j).as_float())
                    im.append(row.at(j + 1).as_float())
                ch.add(re^, im^)
            c.apply_channel(ch, qs[0], qs[1] if a == 2 else -1)
        elif k == Op.MEASURE:
            c.measure(qs[0], g.get_int("clbit", qs[0]))
        else:
            var pv = g.get("params")
            var np = op_num_params(k)
            if len(pv) != np:
                raise _path_err(path + ".params", "'" + nm + "' needs " + String(np) + " parameter(s), got " + String(len(pv)), "pass numbers or {\"param\": name, \"coeff\": 1, \"offset\": 0}")
            var pp = List[Param]()
            for j in range(np):
                var x = pv.at(j)
                if x.is_number():
                    pp.append(Param(x.as_float()))
                elif x.is_object():
                    var pn = x.get_string("param", "")
                    var idx = -1
                    for t in range(len(c.param_names)):
                        if c.param_names[t] == pn:
                            idx = t
                    if idx < 0:
                        raise Error("QM-E913: " + path + ".params[" + String(j) + "]: unknown parameter '" + pn + "'. Hint: list it in circuit.params.")
                    pp.append(Param(_read_number(x, "offset", 0.0), idx, _read_number(x, "coeff", 1.0)))
                else:
                    raise _path_err(path + ".params[" + String(j) + "]", "expected a number or parameter object", "e.g. 0.5 or {\"param\": \"theta\", \"coeff\": 2.0, \"offset\": 0.0}")
            while len(pp) < 3:
                pp.append(Param(0.0))
            c.add(k, qs[0] if a > 0 else -1, qs[1] if a > 1 else -1, qs[2] if a > 2 else -1, pp[0], pp[1], pp[2])
        _read_condition_json(c, g)
    c.validate()
    return c^


# ---------------------------------------------------------------------------
# Programs
# ---------------------------------------------------------------------------
comptime _PROGRAM_FORMAT = "qumo.program.v1"
comptime _MAX_STORE = 16777216  # a table this large is a bad file, not a program


def _node_names() -> List[String]:
    """Index = CN code."""
    return ["const", "var", "input", "bits", "neg", "lnot", "bnot", "add", "sub", "mul", "div", "mod", "pow",
            "shl", "shr", "and", "or", "xor", "eq", "ne", "lt", "le", "gt", "ge", "cast", "fn",
            "sc_and", "sc_or", "sc_end"]


def _stmt_names() -> List[String]:
    """Index = SK code."""
    return ["ops", "assign", "if", "else", "while", "for", "switch", "case", "break", "continue", "end"]


def write_program_json(p: Program, mut w: JsonWriter):
    var ops = _node_names()
    var kinds = _stmt_names()
    w.begin_object()
    w.kv_string("format", _PROGRAM_FORMAT)
    w.key("circuit")
    write_circuit_json(p.body, w)
    w.kv_int("num_store", p.num_store)
    w.kv_int_list("user_params", p.user_params)
    w.key("registers")
    w.begin_array()
    for r in p.cregs:
        w.begin_object()
        w.kv_string("name", r.name)
        w.kv_int("first", r.first)
        w.kv_int("width", r.width)
        w.end_object()
    w.end_array()
    w.key("variables")
    w.begin_array()
    for v in p.vars:
        w.begin_object()
        w.kv_string("name", v.name)
        w.kv_string("type", type_name(v.ty))
        w.kv_int("store", v.store)
        w.kv_int("slot", v.slot)
        w.kv_bool("output", v.is_output)
        w.end_object()
    w.end_array()
    w.key("nodes")
    w.begin_array()
    for n in p.nodes:
        w.begin_object()
        w.kv_string("op", ops[n.op])
        w.kv_string("type", type_name(n.ty))
        if n.x != 0: w.kv_int("x", n.x)
        if n.y != 0: w.kv_int("y", n.y)
        if n.i != 0: w.kv_int("i", n.i)
        if String(n.f) != "0.0":
            _kv_number(w, "f", n.f)
        w.end_object()
    w.end_array()
    w.key("expressions")
    w.begin_array()
    for i in range(len(p.expr_lo)):
        w.begin_array()
        w.integer(p.expr_lo[i])
        w.integer(p.expr_root[i])
        w.end_array()
    w.end_array()
    w.kv_int_list("for_ranges", p.for_ranges)
    w.kv_int_list("case_values", p.case_vals)
    w.key("statements")
    w.begin_array()
    for s in p.tape:
        w.begin_object()
        w.kv_string("kind", kinds[s.kind])
        if s.kind == SK.OPS or s.kind == SK.ASSIGN or s.kind == SK.FOR or s.kind == SK.CASE:
            w.kv_int("a", s.a)
            w.kv_int("b", s.b)
        elif s.kind == SK.IF or s.kind == SK.WHILE or s.kind == SK.SWITCH:
            w.kv_int("a", s.a)
        if s.line != 0:
            w.kv_int("line", s.line)
        w.end_object()
    w.end_array()
    w.end_object()


def program_to_json(p: Program) -> String:
    var w = JsonWriter()
    write_program_json(p, w)
    return w.finish()


def _bad_program(msg: String) -> Error:
    return Error("QM-E587: " + msg + ". Hint: pass the output of program_to_json or qumo_convert with to='json'.")


def _json_int(v: JsonValue, what: String) raises -> Int:
    if not v.is_number():
        raise _bad_program(what + " must be an integer, got " + v.kind_name())
    try:
        return v.as_int()
    except:
        raise _bad_program(what + " must be an integer")


def _json_int_or(v: JsonValue, key: String, default: Int, what: String) raises -> Int:
    var x = v.get(key)
    if not x.is_defined():
        return default
    return _json_int(x, what + "." + key)


def _json_ints(v: JsonValue, key: String) raises -> List[Int]:
    var a = v.get(key)
    if not a.is_defined():
        return List[Int]()
    if not a.is_array():
        raise _bad_program(key + " must be an array")
    var out = List[Int](capacity=len(a))
    for i in range(len(a)):
        out.append(_json_int(a.at(i), key + "[" + String(i) + "]"))
    return out^


def _json_string(v: JsonValue, key: String, what: String) raises -> String:
    var x = v.get(key)
    if not x.is_string():
        raise _bad_program(what + "." + key + " must be a string")
    return x.as_string()


def _json_array(v: JsonValue, key: String) raises -> JsonValue:
    var a = v.get(key)
    if not a.is_array():
        raise _bad_program("'" + key + "' must be an array")
    return a^


def _json_float(v: JsonValue, what: String) raises -> Float64:
    try:
        return _read_number(v, "f", 0.0)
    except:
        raise _bad_program(what + ".f must be a number, 'nan', 'inf' or '-inf'")


def _type_from_name(s: String, what: String) raises -> CType:
    """Inverse of `type_name`."""
    if s == "bool": return CType(CT.BOOL, 1, False)
    if s == "float": return CType(CT.FLOAT, 64, True)
    if s == "float[64]": return CType(CT.FLOAT, 64, False)
    if s == "bit": return CType(CT.BIT, 1, False)
    if s == "int": return CType(CT.INT, 64, True)
    if s == "uint": return CType(CT.UINT, 64, True)
    var open = s.find("[")
    if open > 0 and s.endswith("]"):
        var nm = String(s[byte=0:open])
        var width: Int
        try:
            width = Int(String(s[byte=open + 1:s.byte_length() - 1]))
        except:
            raise _bad_program(what + ": bad type '" + s + "'")
        var tag = CT.BIT if nm == "bit" else (CT.INT if nm == "int" else (CT.UINT if nm == "uint" else (CT.ANGLE if nm == "angle" else -1)))
        if tag >= 0 and width >= 1 and width <= 64:
            return CType(tag, width, False)
    raise _bad_program(what + ": bad type '" + s + "'")


def _is_float(t: CType) -> Bool:
    return t.tag == CT.FLOAT


def _check_operand(p: Program, pc: Int, at: Int, lo: Int, float_: Bool) raises:
    """Node `at` is an operand of node `pc` in an expression starting at `lo`."""
    if at < lo or at >= pc:
        raise _bad_program("nodes[" + String(pc) + "] has an operand outside its expression")
    if _is_float(p.nodes[at].ty) != float_:
        raise _bad_program("nodes[" + String(pc) + "] has an operand of the wrong kind")


def _check_expression(p: Program, lo: Int, root: Int) raises:
    """What `eval_nodes` takes on trust: operands before their node, in tables."""
    var pc = lo
    while pc <= root:
        ref n = p.nodes[pc]
        var op = n.op
        var fl = _is_float(n.ty)
        if op == CN.CONST:
            pass
        elif op == CN.VAR:
            if n.x < 0 or n.x >= p.num_store:
                raise _bad_program("nodes[" + String(pc) + "] reads a variable outside the store")
        elif op == CN.INPUT:
            if n.x < 0 or n.x >= p.body.num_params() or not fl:
                raise _bad_program("nodes[" + String(pc) + "] reads a parameter outside the circuit")
        elif op == CN.BITS:
            if n.x < 0 or n.y < 1 or n.y > 64 or n.x + n.y > p.num_clbits() or fl:
                raise _bad_program("nodes[" + String(pc) + "] reads bits outside the circuit")
        elif op == CN.CAST:
            if n.x < lo or n.x >= pc:
                raise _bad_program("nodes[" + String(pc) + "] has an operand outside its expression")
        elif op == CN.NEG:
            _check_operand(p, pc, n.x, lo, fl)
        elif op == CN.LNOT or op == CN.BNOT or op == CN.SC_END:
            _check_operand(p, pc, n.x, lo, False)
        elif op == CN.ADD or op == CN.SUB or op == CN.MUL or op == CN.DIV or op == CN.MOD or op == CN.POW:
            _check_operand(p, pc, n.x, lo, fl)
            _check_operand(p, pc, n.y, lo, fl)
        elif op == CN.SHL or op == CN.SHR or op == CN.AND or op == CN.OR or op == CN.XOR:
            _check_operand(p, pc, n.x, lo, False)
            _check_operand(p, pc, n.y, lo, False)
        elif op >= CN.EQ and op <= CN.GE:
            if n.x < lo or n.x >= pc:
                raise _bad_program("nodes[" + String(pc) + "] has an operand outside its expression")
            _check_operand(p, pc, n.y, lo, _is_float(p.nodes[n.x].ty))
        elif op == CN.SC_AND or op == CN.SC_OR:
            _check_operand(p, pc, n.x, lo, False)
            if n.y <= pc or n.y > root or p.nodes[n.y].op != CN.SC_END:
                raise _bad_program("nodes[" + String(pc) + "] skips to a node that does not end it")
        elif op == CN.FN:
            if n.i < 0 or n.i > CF.POPCOUNT:
                raise _bad_program("nodes[" + String(pc) + "] calls an unknown function")
            var on_ints = n.i == CF.POPCOUNT or (n.i == CF.ABS and not fl)
            _check_operand(p, pc, n.x, lo, not on_ints)
            if n.y != -1:
                _check_operand(p, pc, n.y, lo, not on_ints)
            elif n.i == CF.MOD:
                raise _bad_program("nodes[" + String(pc) + "] calls mod with one operand")
        pc += 1


def _check_tables(p: Program) raises:
    """Every index in the classical tables of a program read from a file. Nothing
    runs on trust in the file: a bad index here would be a crash in a shot."""
    var nv = p.num_store
    if nv < 0 or nv > _MAX_STORE:
        raise _bad_program("num_store is out of range")
    for i in p.user_params:
        if i < 0 or i >= p.body.num_params():
            raise _bad_program("user_params names a parameter outside the circuit")
    for r in p.cregs:
        if r.first < 0 or r.width < 0 or r.first + r.width > p.num_clbits():
            raise _bad_program("register '" + r.name + "' lies outside the circuit")
    for v in p.vars:
        if v.ty.tag == CT.BIT:
            if v.store < 0 or v.store + v.ty.width > p.num_clbits():
                raise _bad_program("variable '" + v.name + "' lies outside the circuit")
        elif v.store < 0 or v.store >= nv:
            raise _bad_program("variable '" + v.name + "' lies outside the store")
        if v.slot < -1 or v.slot >= p.body.num_params():
            raise _bad_program("variable '" + v.name + "' has a slot outside the circuit")
    if len(p.expr_lo) != len(p.expr_root):
        raise _bad_program("expressions must be [lo, root] pairs")
    for e in range(len(p.expr_lo)):
        var lo = p.expr_lo[e]
        var root = p.expr_root[e]
        if lo < 0 or root < lo or root >= len(p.nodes):
            raise _bad_program("expressions[" + String(e) + "] lies outside the nodes")
        _check_expression(p, lo, root)


def _check_statement(p: Program, s: Stmt, at: Int) raises:
    var ne = len(p.expr_lo)
    var what = "statements[" + String(at) + "]"
    if s.kind == SK.OPS:
        if s.a < 0 or s.b < s.a or s.b > len(p.body.ops):
            raise _bad_program(what + " lies outside the circuit")
    elif s.kind == SK.ASSIGN:
        if s.a < 0 or s.a >= len(p.vars) or s.b < 0 or s.b >= ne:
            raise _bad_program(what + " names a variable or expression that does not exist")
    elif s.kind == SK.IF or s.kind == SK.WHILE or s.kind == SK.SWITCH:
        if s.a < 0 or s.a >= ne:
            raise _bad_program(what + " names an expression that does not exist")
    elif s.kind == SK.CASE:
        if s.a != -1 and (s.a < 0 or s.b < s.a or s.b > len(p.case_vals)):
            raise _bad_program(what + " lies outside case_values")
    elif s.kind == SK.FOR:
        if s.a < 0 or s.a >= len(p.vars) or s.b < 0 or s.b + FORR.SIZE > len(p.for_ranges):
            raise _bad_program(what + " names a variable or range that does not exist")
        var store = p.for_ranges[s.b + FORR.STORE]
        if store < 0 or store + 3 > p.num_store:
            raise _bad_program(what + " keeps its state outside the store")
        var set_at = p.for_ranges[s.b + FORR.SET]
        if set_at >= 0:
            if set_at >= len(p.case_vals) or p.case_vals[set_at] < 0 or set_at + 1 + p.case_vals[set_at] > len(p.case_vals):
                raise _bad_program(what + " has a set outside case_values")
        else:
            for f in range(FORR.START, FORR.STOP + 1):
                var e = p.for_ranges[s.b + f]
                if e < 0 or e >= ne:
                    raise _bad_program(what + " names an expression that does not exist")


def program_from_json(v: JsonValue) raises -> Program:
    """Inverse of `program_to_json`. Every index is checked, and the control-flow
    links are built again from the statements (QM-E583 when they do not nest)."""
    if not v.is_object():
        raise _bad_program("a program is an object")
    if v.get_string("format", "") != _PROGRAM_FORMAT:
        raise _bad_program("format must be '" + _PROGRAM_FORMAT + "'")
    var p = Program(circuit_from_json(v.get("circuit")))
    p.num_store = _json_int_or(v, "num_store", 0, "program")
    p.user_params = _json_ints(v, "user_params")
    var regs = _json_array(v, "registers")
    for i in range(len(regs)):
        var r = regs.at(i)
        var what = "registers[" + String(i) + "]"
        p.cregs.append(CReg(_json_string(r, "name", what), _json_int_or(r, "first", -1, what), _json_int_or(r, "width", -1, what)))
    var vars = _json_array(v, "variables")
    for i in range(len(vars)):
        var x = vars.at(i)
        var what = "variables[" + String(i) + "]"
        p.vars.append(CVar(_json_string(x, "name", what), _type_from_name(_json_string(x, "type", what), what),
                           _json_int_or(x, "store", -1, what), _json_int_or(x, "slot", -1, what), x.get_bool("output", False)))
    var names = _node_names()
    var nodes = _json_array(v, "nodes")
    for i in range(len(nodes)):
        var x = nodes.at(i)
        var what = "nodes[" + String(i) + "]"
        var nm = _json_string(x, "op", what)
        var op = -1
        for k in range(len(names)):
            if names[k] == nm:
                op = k
        if op < 0:
            raise _bad_program(what + ": unknown op '" + nm + "'")
        p.nodes.append(CNode(op, _type_from_name(_json_string(x, "type", what), what), _json_int_or(x, "x", 0, what),
                             _json_int_or(x, "y", 0, what), _json_int_or(x, "i", 0, what), _json_float(x, what)))
    var exprs = _json_array(v, "expressions")
    for i in range(len(exprs)):
        var what = "expressions[" + String(i) + "]"
        var e = exprs.at(i)
        if not e.is_array() or len(e) != 2:
            raise _bad_program(what + " must be [lo, root]")
        p.expr_lo.append(_json_int(e.at(0), what))
        p.expr_root.append(_json_int(e.at(1), what))
    p.for_ranges = _json_ints(v, "for_ranges")
    p.case_vals = _json_ints(v, "case_values")
    _check_tables(p)
    var kinds = _stmt_names()
    var stmts = _json_array(v, "statements")
    var t = TapeBuilder()
    for i in range(len(stmts)):
        var x = stmts.at(i)
        var what = "statements[" + String(i) + "]"
        var nm = _json_string(x, "kind", what)
        var kind = -1
        for k in range(len(kinds)):
            if kinds[k] == nm:
                kind = k
        if kind < 0:
            raise _bad_program(what + ": unknown kind '" + nm + "'")
        var s = Stmt(kind, _json_int_or(x, "a", -1, what), _json_int_or(x, "b", -1, what), -1, -1, _json_int_or(x, "line", 0, what))
        _check_statement(p, s, i)
        if kind == SK.OPS or kind == SK.ASSIGN:
            t.tape.append(s)
        elif kind == SK.IF or kind == SK.WHILE or kind == SK.FOR or kind == SK.SWITCH:
            t.begin(kind, s.a, s.b, 0, s.line)
        elif kind == SK.ELSE:
            t.else_(0, s.line)
        elif kind == SK.CASE:
            t.case_(s.a, s.b, 0, s.line)
        elif kind == SK.BREAK or kind == SK.CONTINUE:
            t.jump(kind, 0, s.line)
        else:
            t.end(0, s.line)
    if len(t.open) != 0:
        raise Error("QM-E583: a region of the program is left open. Hint: every if, while, for and switch needs its end.")
    p.tape = t.tape.copy()
    _ = check_program(p)
    return p^


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
