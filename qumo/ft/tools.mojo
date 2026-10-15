"""AI-native surface of `qumo.ft`: JSON descriptions and MCP tools.

Tools (registered by `register_ft_tools`, listed by MCP `tools/list`,
callable as `build/qumo tool <name> '<json>'`):

* `qumo_ft_codes`    - the code library ([[n,k,d]], generators, logicals, CSS,
                        brute-force distance check) and, for an architecture,
                        the verification status of every gadget;
* `qumo_ft_encode`   - compile + implement_ops a circuit for an architecture
                        (Steane / Iceberg, QEC policy, factories, feed-forward,
                        Rz mode); returns the logical program, the resource
                        report and, with `shots`, an emulation of the encoded
                        program next to the unencoded reference (optional
                        circuit-level noise);
* `qumo_ft_verify`   - `valid_clifford_implementation` /
                        `valid_stabilizer_state_preparation` on OpenQASM inputs.

Every failure is a stable `QM-E72x` code with a hint (docs/ERRORS.md).

Idea from Guppy FT (Apache-2.0): the architecture/primitive split of
docs/architecture_dev.md, exposed here to agents.
"""

from std.math import pi

from qumo.ir.circuit import Circuit
from qumo.ir.qasm import to_qasm, parse_qasm_with_warnings
from qumo.ai.json import JsonWriter, JsonValue
from qumo.ai.tools import ToolRegistry, ToolResult, ToolContext, tool_ok, arg_circuit, arg_int, arg_required
from qumo.exec.resources import estimate_resources
from qumo.qec.codes import NoiseParams
from qumo.ft.code import StabilizerCode, code_by_name, code_names
from qumo.ft.arch import Architecture, SteaneBuilder, IcebergBuilder, QECPolicy, FactoryConf, architecture_by_name
from qumo.ft.logical import LogicalProgram, LOp, lop_name
from qumo.ft.encode import compile, encode, check_may_encode
from qumo.ft.implement import implement_ops, EncodedProgram
from qumo.ft.emulate import run_encoded, reference_counts, total_variation, LogicalCounts
from qumo.ft.verify import valid_clifford_implementation, valid_stabilizer_state_preparation, VerifyReport
from qumo.ft.gadgets import verify_steane_library, verify_iceberg_library, GadgetCheck


def default_code_names() -> List[String]:
    var l: List[String] = ["steane", "five_qubit", "css_422", "noncss_422", "iceberg_2", "iceberg_4", "surface_3", "repetition_3"]
    return l^


def ft_codes_json(names: List[String], check_distance: Bool) raises -> String:
    var w = JsonWriter()
    w.begin_object()
    w.key("codes")
    w.begin_array()
    for nm in names:
        var c = code_by_name(nm)
        w.begin_object()
        w.kv_raw("definition", c.to_json())
        w.kv_bool("valid", True)
        if check_distance and c.n <= 12:
            w.kv_int("verified_distance", c.min_distance(4))
        w.end_object()
    w.end_array()
    w.kv_string("families", code_names())
    w.kv_string("label_convention", "character i of a Pauli label is physical qubit i (Guppy FT / Stim convention)")
    w.end_object()
    return w.finish()


def gadget_checks_json(arch: Architecture) raises -> String:
    var checks: List[GadgetCheck]
    if arch.kind == "steane":
        checks = verify_steane_library(arch.code)
    else:
        checks = verify_iceberg_library(arch.code)
    var w = JsonWriter()
    w.begin_array()
    for c in checks:
        w.begin_object()
        w.kv_string("gadget", c.name)
        w.kv_bool("valid", c.report.ok)
        w.kv_int("operators_checked", c.report.checked)
        if not c.report.ok:
            w.kv_string("reason", c.report.reason)
        w.end_object()
    w.end_array()
    return w.finish()


def resource_report_json(prog: EncodedProgram) raises -> String:
    """Logical + physical resources of an encoded program (physical metrics
    from qumo.exec.resources.estimate_resources)."""
    var r = estimate_resources(prog.circuit)
    var w = JsonWriter()
    w.begin_object()
    w.key("logical")
    w.begin_object()
    w.kv_int("logical_qubits_peak", prog.stats.peak_logical)
    w.kv_int("logical_ops", len(prog.logical.ops))
    w.kv_int("logical_t_count", prog.logical.t_count())
    w.kv_int("qec_cycles", prog.stats.qec_cycles)
    w.kv_int("magic_states", prog.stats.magic_states)
    w.kv_int("zero_states", prog.stats.zero_states)
    w.end_object()
    w.key("physical")
    w.begin_object()
    w.kv_int("qubits", r.num_qubits)
    w.kv_int("lane_qubits", len(prog.lane_qubits))
    w.kv_int("clbits", r.num_clbits)
    w.kv_int("gates", r.gates_total)
    w.kv_int("two_qubit_gates", r.gates_2q)
    w.kv_int("measurements", r.measurements)
    w.kv_int("resets", r.resets)
    w.kv_int("depth", r.depth)
    w.kv_int("depth_2q", r.depth_2q)
    w.kv_int("non_clifford_gates", r.t_count)
    w.kv_bool("is_clifford", r.is_clifford)
    w.kv_string("recommended_simulator", String("tableau (qumo.ft.emulate.run_encoded)") if r.is_clifford else String("state vector trajectories"))
    w.end_object()
    w.key("encoder")
    prog.stats.write_json(w)
    w.end_object()
    return w.finish()


def _arch_from_args(args: JsonValue, default_blocks: Int) raises -> Architecture:
    var name = args.get_string("architecture", "steane").lower()
    var n_blocks = args.get_int("n_blocks", default_blocks)
    if n_blocks < 1 or n_blocks > 64:
        raise Error("QM-E913: argument 'n_blocks' = " + String(n_blocks) + " is out of range [1, 64]. Hint: one block per simultaneously live logical qubit (k per block).")
    var style_default = String("steane") if name == "steane" else String("syndrome")
    var pol = QECPolicy(args.get_string("qec_style", style_default), args.get_float("qec_threshold", 1.0), args.get_bool("qec_before_non_clifford", False))
    var costs = args.get("qec_costs")
    if costs.is_defined() and not costs.is_null():
        for k in costs.keys():
            pol = pol.with_cost(k, costs.get(k).as_float())
    if name == "steane":
        var b = SteaneBuilder().with_qec_policy(pol)
        b = b.with_feed_forward(args.get_string("feed_forward", "frame"))
        b = b.with_magic_prep(args.get_string("magic_prep", "nonft"))
        b = b.with_zero_prep(args.get_string("zero_prep", "ft"))
        b = b.with_zero_factory_conf(FactoryConf(args.get_int("zero_factory_size", 0), args.get_int("zero_factory_attempts", 1)))
        b = b.with_magic_factory_conf(FactoryConf(args.get_int("magic_factory_size", 0), args.get_int("magic_factory_attempts", 1)))
        b = b.with_rz(args.get_string("rz_mode", "static"), args.get_float("rz_epsilon", 1e-2), args.get_int("rz_max_t", 16), args.get_int("comparator_attempts", 4))
        return b.build(n_blocks)
    if name.startswith("iceberg_"):
        var kk: Int
        try:
            kk = Int(String(name[byte=8:]))
        except:
            raise Error("QM-E727: bad Iceberg size in '" + name + "'. Hint: iceberg_2, iceberg_4, ...")
        var ib = IcebergBuilder(kk).with_qec_policy(pol).with_feed_forward(args.get_string("feed_forward", "frame"))
        return ib.build(n_blocks)
    raise Error("QM-E727: unknown architecture '" + name + "'. Hint: 'steane' or 'iceberg_<even k>'.")


def _default_blocks(args: JsonValue, c: Circuit) raises -> Int:
    var name = args.get_string("architecture", "steane").lower()
    if name.startswith("iceberg_"):
        var kk = 2
        try:
            kk = Int(String(name[byte=8:]))
        except:
            pass
        return max(1, (c.num_qubits + kk - 1) // max(kk, 1))
    var extra = 1 if (c.t_count() > 0) else 0
    return max(1, c.num_qubits + extra)


def _counts_json(c: LogicalCounts) -> String:
    var w = JsonWriter()
    c.write_json(w)
    return w.finish()


def _ft_codes_tool(args: JsonValue, ctx: ToolContext) raises -> ToolResult:
    var names = List[String]()
    var one = args.get_string("code", "")
    if one.byte_length() > 0:
        names.append(one)
    else:
        names = default_code_names()
    var w = JsonWriter()
    w.begin_object()
    w.kv_raw("library", ft_codes_json(names, args.get_bool("check_distance", True)))
    var arch = args.get_string("architecture", "")
    if arch.byte_length() > 0:
        var a = architecture_by_name(arch, 1)
        w.kv_raw("architecture", a.to_json())
        if args.get_bool("verify_gadgets", True):
            w.kv_raw("gadget_verification", gadget_checks_json(a))
    w.end_object()
    return tool_ok(w.finish())


def _ft_encode_tool(args: JsonValue, ctx: ToolContext) raises -> ToolResult:
    var warnings = List[String]()
    var c = arg_circuit(args, warnings)
    var arch = _arch_from_args(args, _default_blocks(args, c))
    var lp = compile(arch, c)
    var prog = implement_ops(arch, lp)
    var w = JsonWriter()
    w.begin_object()
    w.kv_raw("architecture", arch.to_json())
    var lw = JsonWriter()
    lp.summary_json(lw)
    w.kv_raw("logical", lw.finish())
    if args.get_bool("include_logical", False):
        w.kv_string("logical_listing", String(lp))
    w.kv_raw("resources", resource_report_json(prog))
    w.kv_int_list("outputs", prog.outputs)
    w.kv_int_list("postselect_clbits", prog.postselect)
    var shots = arg_int(args, "shots", 0, 0, 100000)
    if shots > 0:
        var seed = UInt64(arg_int(args, "seed", 1234, 0, 1 << 62))
        var p = args.get_float("noise_p", 0.0)
        if p < 0.0 or p > 0.5:
            raise Error("QM-E913: argument 'noise_p' = " + String(p) + " is out of range [0, 0.5]. Hint: 1e-3 is a typical circuit-level noise strength.")
        var noise = NoiseParams(p, p, p, 0.0)
        var enc = run_encoded(prog, shots, seed, noise)
        var ref_ = reference_counts(c, shots, seed + 1, noise)
        w.key("emulation")
        w.begin_object()
        w.kv_float("noise_p", p)
        w.kv_raw("encoded", _counts_json(enc))
        w.kv_raw("unencoded", _counts_json(ref_))
        w.kv_float("total_variation_distance", total_variation(enc, ref_))
        w.end_object()
    if args.get_bool("include_qasm", False):
        if len(prog.circuit.ops) > 20000:
            w.kv_string("qasm_note", "physical circuit too large to inline (> 20000 ops)")
        else:
            w.kv_string("qasm", to_qasm(prog.circuit))
            w.kv_string("qasm_note", "QUMO's OpenQASM 3 export comments out AND-conditions over bit subsets (QM-E552); run the circuit natively for exact semantics")
    w.end_object()
    return tool_ok(w.finish())


def _ft_verify_tool(args: JsonValue, ctx: ToolContext) raises -> ToolResult:
    var code = code_by_name(args.get_string("code", "steane"))
    var warnings = List[String]()
    var sem = parse_qasm_with_warnings(arg_required(args, "semantic_qasm", "OpenQASM of the logical action on num_blocks*k qubits").as_string(), warnings)
    var imp = parse_qasm_with_warnings(arg_required(args, "impl_qasm", "OpenQASM of the physical gadget on num_blocks*n (+ancilla) qubits").as_string(), warnings)
    var nb = arg_int(args, "num_blocks", 1, 1, 4)
    var na = arg_int(args, "num_ancillas", 0, 0, 32)
    var kind = args.get_string("kind", "clifford")
    var rep: VerifyReport
    if kind == "clifford":
        rep = valid_clifford_implementation(sem, imp, code, nb, na)
    elif kind == "state":
        rep = valid_stabilizer_state_preparation(sem, imp, code, nb, na)
    else:
        raise Error("QM-E913: argument 'kind' must be 'clifford' or 'state' (got '" + kind + "').")
    var w = JsonWriter()
    w.begin_object()
    w.kv_string("code", code.name)
    w.kv_string("kind", kind)
    w.kv_raw("report", rep.to_json())
    w.end_object()
    return tool_ok(w.finish())


comptime _FT_CIRC = "\"qasm\":{\"type\":\"string\",\"description\":\"OpenQASM 2 or 3 program\"},\"circuit\":{\"type\":\"object\",\"description\":\"qumo.circuit.v1 JSON\"}"


def register_ft_tools(mut reg: ToolRegistry):
    reg.add(
        "qumo_ft_codes", "Fault-tolerant code library",
        "List QUMO's stabilizer codes for fault-tolerant programming (Steane [[7,1,3]], Iceberg [[k+2,k,2]], [[4,2,2]] CSS and non-CSS, [[5,1,3]], surface and repetition adapters) with generators, logical operators, CSS flag and a brute-force distance check; with 'architecture' ('steane' | 'iceberg_<k>') also describe the architecture (logical gate set, QEC policy, factories) and verify every gadget of its library with the stabilizer tableau (and state vector for magic states).",
        "{\"type\":\"object\",\"properties\":{\"code\":{\"type\":\"string\",\"description\":\"steane, five_qubit, css_422, noncss_422, iceberg_<k>, surface_<d>, repetition_<d>\"},\"check_distance\":{\"type\":\"boolean\",\"default\":true},\"architecture\":{\"type\":\"string\"},\"verify_gadgets\":{\"type\":\"boolean\",\"default\":true}},\"additionalProperties\":false}",
        _ft_codes_tool,
    )
    reg.add(
        "qumo_ft_encode", "Encode a circuit fault-tolerantly",
        "Encode a QEC-agnostic circuit (any gates, mid-circuit measurement, reset, c_if feed-forward) into a fault-tolerant physical program, Guppy FT style: compile (computational -> logical gate set of the architecture: Steane Clifford+T with magic-state injection and static Clifford+T or comparator RUS synthesis of Rz, or Iceberg native rotations) then implement_ops (logical -> physical gadgets, block allocation, state factories, QEC-cycle policy, Pauli-frame feed-forward, decoding). Returns the logical program summary, a resource report (physical qubits, depth, 2q gates, QEC cycles, magic states) and, with 'shots', the encoded program emulated next to the unencoded reference (optional circuit-level noise 'noise_p').",
        "{\"type\":\"object\",\"properties\":{" + _FT_CIRC + ",\"architecture\":{\"type\":\"string\",\"default\":\"steane\"},\"n_blocks\":{\"type\":\"integer\"},\"qec_style\":{\"type\":\"string\",\"enum\":[\"none\",\"steane\",\"knill\",\"syndrome\"]},\"qec_threshold\":{\"type\":\"number\",\"default\":1.0},\"qec_costs\":{\"type\":\"object\",\"description\":\"logical op name -> cost, e.g. {\\\"h\\\":1,\\\"cx\\\":1}\"},\"qec_before_non_clifford\":{\"type\":\"boolean\",\"default\":false},\"feed_forward\":{\"type\":\"string\",\"enum\":[\"frame\",\"eager\"],\"default\":\"frame\"},\"magic_prep\":{\"type\":\"string\",\"enum\":[\"ft\",\"nonft\"],\"default\":\"nonft\"},\"zero_prep\":{\"type\":\"string\",\"enum\":[\"ft\",\"nonft\"],\"default\":\"ft\"},\"zero_factory_size\":{\"type\":\"integer\",\"default\":0},\"zero_factory_attempts\":{\"type\":\"integer\",\"default\":1},\"magic_factory_size\":{\"type\":\"integer\",\"default\":0},\"magic_factory_attempts\":{\"type\":\"integer\",\"default\":1},\"rz_mode\":{\"type\":\"string\",\"enum\":[\"static\",\"comparator\",\"exact\"],\"default\":\"static\"},\"rz_epsilon\":{\"type\":\"number\",\"default\":0.01},\"rz_max_t\":{\"type\":\"integer\",\"default\":16},\"comparator_attempts\":{\"type\":\"integer\",\"default\":4},\"shots\":{\"type\":\"integer\",\"default\":0},\"seed\":{\"type\":\"integer\",\"default\":1234},\"noise_p\":{\"type\":\"number\",\"default\":0.0},\"include_logical\":{\"type\":\"boolean\",\"default\":false},\"include_qasm\":{\"type\":\"boolean\",\"default\":false}},\"additionalProperties\":false}",
        _ft_encode_tool,
    )
    reg.add(
        "qumo_ft_verify", "Verify a logical gadget",
        "Check that a physical gadget implements a logical operation on a stabilizer code, exactly, with the stabilizer tableau: kind 'clifford' (semantic Clifford on num_blocks*k qubits vs unitary implementation on num_blocks*n (+ clean ancillas): stabilizers preserved and logical Paulis mapped with the right signs) or 'state' (stabilizer-state preparation, flags/measurements allowed, accept branch). Returns valid + the first failing operator.",
        "{\"type\":\"object\",\"properties\":{\"code\":{\"type\":\"string\",\"default\":\"steane\"},\"kind\":{\"type\":\"string\",\"enum\":[\"clifford\",\"state\"],\"default\":\"clifford\"},\"semantic_qasm\":{\"type\":\"string\"},\"impl_qasm\":{\"type\":\"string\"},\"num_blocks\":{\"type\":\"integer\",\"default\":1},\"num_ancillas\":{\"type\":\"integer\",\"default\":0}},\"required\":[\"semantic_qasm\",\"impl_qasm\"],\"additionalProperties\":false}",
        _ft_verify_tool,
    )
