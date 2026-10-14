"""`qumo_build_circuit`: build an algorithm from QUMO's library (agent tool).

Agents (an MCP client, the AI selector of examples/heterogeneous) choose an
algorithm *by name* from an allow-list and get back the circuit as OpenQASM,
built by QUMO's own tested builders (qumo.algorithms.library), instead of
generating circuit code themselves.

Input  {"kind": K, "args": {...}, "measure": false, "format": "qasm3"}

| kind               | args (defaults)                                                         | builder |
|--------------------|-------------------------------------------------------------------------|---------|
| ghz                | n                                                                       | ghz |
| qft / iqft         | n, do_swaps (true), approximation_degree (0)                            | qft / iqft |
| grover             | marked ["101", ...] (Qiskit keys), iterations (-1 = optimal)            | grover_bitstrings |
| bernstein_vazirani | secret "1011" (key) or integer + n                                      | bernstein_vazirani |
| deutsch_jozsa      | n, balanced_mask (0 = constant), constant_value (0)                     | deutsch_jozsa |
| qaoa_maxcut        | graph "0-1 1-2 2-0:0.5", p (1)                                          | qaoa_maxcut_ansatz |
| hea                | n, layers (2), entanglement ("linear"), rotations ("ry,rz"), hamiltonian | hardware_efficient_ansatz |
| uccsd              | num_qubits, num_electrons, singles (true), doubles (true), hamiltonian  | uccsd_ansatz |
| h2_ansatz          | (none)                                                                  | h2_ansatz + h2_sto3g(4) |
| trotter            | hamiltonian, time (1.0), steps (1), order (1), num_qubits, param_name   | trotter |
| reuploading_classifier | n (1), layers (3), num_features (3)                                 | qml.append_rot (data re-uploading) |
| angle_kernel       | n, rotation ("x")                                                       | qml.append_angle_embedding + qkernels.kernel_circuit |

Output: `qasm` (symbolic parameters as `input float[64]`), `format`,
`num_qubits`, `num_clbits`, `parameters` (names in declaration order),
`num_parameters`, `gate_counts`, `two_qubit_gates`, `depth`, `observable`
(Pauli sum the circuit is meant to be measured against: the MaxCut cost
H = sum w/2 (Z_u Z_v - 1) with <H> = -cut for qaoa_maxcut, the molecular
Hamiltonian for h2_ansatz, the given `hamiltonian` for hea / uccsd / trotter,
the label-0 fidelity 0.5 + 0.5 Z0 for reuploading_classifier, |0..0><0..0|
(the kernel value) for angle_kernel; "" otherwise) and `notes`.

Size caps keep every result small enough to inspect and simulate; anything
over a cap, an unknown argument or a malformed value is a QM-E801 (option) or
QM-E802 (size) tool error. `format: "qasm2"` needs a circuit without symbolic
parameters (QM-E102 otherwise).
"""

from qumo.ir.circuit import Circuit
from qumo.ir.gates import Op, op_name
from qumo.ir.pauli import PauliSum, parse_pauli_sum
from qumo.ir.qasm import to_qasm
from qumo.ir.qasm2 import to_qasm2
from qumo.algorithms.hamiltonians import Graph, maxcut, h2_sto3g
from qumo.algorithms.library import (
    ghz, qft, iqft, grover_bitstrings, grover_optimal_iterations, bernstein_vazirani, deutsch_jozsa,
    qaoa_maxcut_ansatz, hardware_efficient_ansatz, uccsd_ansatz, h2_ansatz, trotter, bitstring_to_int,
)
from qumo.algorithms.qml import append_rot, append_angle_embedding, projector_zero
from qumo.algorithms.qkernels import kernel_circuit
from qumo.ai.json import JsonWriter, JsonValue
from qumo.ai.tools import ToolRegistry, ToolResult, ToolContext, tool_ok


comptime BUILD_CIRCUIT_KINDS = "ghz,qft,iqft,grover,bernstein_vazirani,deutsch_jozsa,qaoa_maxcut,hea,uccsd,h2_ansatz,trotter,reuploading_classifier,angle_kernel"

# Size caps (inclusive).
comptime CAP_WIDE = 64  # ghz, bernstein_vazirani, deutsch_jozsa, hea, qaoa nodes, trotter qubits
comptime CAP_QFT = 32
comptime CAP_GROVER = 16  # search qubits (iterations grow as 2^(n/2))
comptime CAP_GROVER_MARKED = 64
comptime CAP_QAOA_P = 10
comptime CAP_QAOA_EDGES = 2000
comptime CAP_HEA_LAYERS = 20
comptime CAP_UCCSD = 12
comptime CAP_TROTTER_STEPS = 200
comptime CAP_TROTTER_TERMS = 1000
comptime CAP_GATES = 200000
comptime CAP_QML_QUBITS = 12  # reuploading_classifier / angle_kernel (projector observable is dense in Z strings)
comptime CAP_QML_LAYERS = 20


def _known_kinds() -> List[String]:
    var out = List[String]()
    for k in String(BUILD_CIRCUIT_KINDS).split(","):
        out.append(String(k))
    return out^


def _check_args(args: JsonValue, kind: String, allowed: List[String]) raises:
    """Reject unknown argument names (agents must not rely on silently ignored keys)."""
    if not args.is_defined() or args.is_null():
        return
    if not args.is_object():
        raise Error("QM-E801: qumo_build_circuit 'args' must be an object, got " + args.kind_name() + ". Hint: e.g. {\"kind\": \"ghz\", \"args\": {\"n\": 3}}.")
    for k in args.keys():
        var ok = False
        for a in allowed:
            if a == k:
                ok = True
        if not ok:
            var allowed_text = String(", ").join(allowed) if len(allowed) > 0 else String("(none)")
            raise Error("QM-E801: unknown argument '" + k + "' for kind '" + kind + "'. Hint: allowed arguments: " + allowed_text + ".")


def _int_arg(args: JsonValue, key: String, default: Int, lo: Int, hi: Int, kind: String) raises -> Int:
    var v = args.get(key)
    if not v.is_defined() or v.is_null():
        if default < lo:
            raise Error("QM-E801: kind '" + kind + "' needs integer argument '" + key + "'. Hint: e.g. {\"" + key + "\": " + String(lo) + "}.")
        return default
    if not v.is_number():
        raise Error("QM-E801: argument '" + key + "' must be an integer, got " + v.kind_name() + ".")
    var f = v.as_float()
    var i = Int(f)
    if Float64(i) != f:
        raise Error("QM-E801: argument '" + key + "' must be an integer, got " + String(f) + ".")
    if i < lo or i > hi:
        raise Error("QM-E802: argument '" + key + "' = " + String(i) + " is outside [" + String(lo) + ", " + String(hi) + "] for kind '" + kind + "'. Hint: qumo_build_circuit caps sizes so results stay inspectable and simulable.")
    return i


def _bool_arg(args: JsonValue, key: String, default: Bool) raises -> Bool:
    var v = args.get(key)
    if not v.is_defined() or v.is_null():
        return default
    if not v.is_bool():
        raise Error("QM-E801: argument '" + key + "' must be a boolean, got " + v.kind_name() + ".")
    return v.as_bool()


def _str_arg(args: JsonValue, key: String, default: String) raises -> String:
    var v = args.get(key)
    if not v.is_defined() or v.is_null():
        return default
    if not v.is_string():
        raise Error("QM-E801: argument '" + key + "' must be a string, got " + v.kind_name() + ".")
    return v.as_string()


def _float_arg(args: JsonValue, key: String, default: Float64, lo: Float64, hi: Float64) raises -> Float64:
    var v = args.get(key)
    if not v.is_defined() or v.is_null():
        return default
    if not v.is_number():
        raise Error("QM-E801: argument '" + key + "' must be a number, got " + v.kind_name() + ".")
    var f = v.as_float()
    if not (f >= lo and f <= hi):
        raise Error("QM-E802: argument '" + key + "' = " + String(f) + " is outside [" + String(lo) + ", " + String(hi) + "].")
    return f


def _hamiltonian_arg(args: JsonValue, key: String, num_qubits: Int, required: Bool, kind: String) raises -> String:
    var v = args.get(key)
    if not v.is_defined() or v.is_null():
        if required:
            raise Error("QM-E801: kind '" + kind + "' needs a Pauli-sum argument '" + key + "'. Hint: e.g. \"0.5 * Z0 Z1 - 1.0 * X0\".")
        return String()
    if not v.is_string():
        raise Error("QM-E801: argument '" + key + "' must be a Pauli-sum string, got " + v.kind_name() + ".")
    var h = parse_pauli_sum(v.as_string())
    if len(h.terms) > CAP_TROTTER_TERMS:
        raise Error("QM-E802: '" + key + "' has " + String(len(h.terms)) + " terms; the cap is " + String(CAP_TROTTER_TERMS) + ".")
    if num_qubits > 0 and h.num_qubits() > num_qubits:
        raise Error("QM-E802: '" + key + "' acts on qubit " + String(h.num_qubits() - 1) + " but the circuit has " + String(num_qubits) + " qubit(s). Hint: Pauli indices must be < num_qubits.")
    return String(h)


def _count_ones(s: String) -> Int:
    var n = 0
    var bs = s.as_bytes()
    for i in range(s.byte_length()):
        if Int(bs[i]) == ord("1"):
            n += 1
    return n


struct BuiltCircuit(Movable):
    var circuit: Circuit
    var observable: String
    var notes: List[String]

    def __init__(out self, var circuit: Circuit, var observable: String, var notes: List[String]):
        self.circuit = circuit^
        self.observable = observable^
        self.notes = notes^


def build_algorithm_circuit(kind: String, args: JsonValue, measure: Bool) raises -> BuiltCircuit:
    """Build `kind` with `args` (see the module docstring). Raises QM-E801/E802."""
    var notes = List[String]()
    var observable = String()
    var c: Circuit
    var measured_by_builder = False
    if kind == "ghz":
        _check_args(args, kind, ["n"])
        c = ghz(_int_arg(args, "n", -1, 1, CAP_WIDE, kind))
    elif kind == "qft" or kind == "iqft":
        _check_args(args, kind, ["n", "do_swaps", "approximation_degree"])
        var n = _int_arg(args, "n", -1, 1, CAP_QFT, kind)
        var swaps = _bool_arg(args, "do_swaps", True)
        var deg = _int_arg(args, "approximation_degree", 0, 0, n, kind)
        c = qft(n, swaps, deg) if kind == "qft" else iqft(n, swaps, deg)
    elif kind == "grover":
        _check_args(args, kind, ["marked", "iterations"])
        var mv = args.get("marked")
        if not mv.is_defined() or not mv.is_array() or len(mv) == 0:
            raise Error("QM-E801: kind 'grover' needs 'marked': a non-empty array of bitstrings. Hint: e.g. {\"marked\": [\"101\"]} (leftmost character = highest qubit).")
        if len(mv) > CAP_GROVER_MARKED:
            raise Error("QM-E802: grover 'marked' has " + String(len(mv)) + " entries; the cap is " + String(CAP_GROVER_MARKED) + ".")
        var keys = List[String]()
        for i in range(len(mv)):
            if not mv.at(i).is_string():
                raise Error("QM-E801: grover 'marked' entries must be bitstrings like \"101\".")
            keys.append(mv.at(i).as_string())
        var n = keys[0].byte_length()
        if n < 1 or n > CAP_GROVER:
            raise Error("QM-E802: grover search width " + String(n) + " is outside [1, " + String(CAP_GROVER) + "].")
        for k in keys:
            _ = bitstring_to_int(k)  # validates characters (QM-E802)
        var opt = grover_optimal_iterations(n, len(keys))
        var it = _int_arg(args, "iterations", -1, -1, 4 * opt + 4, kind)
        c = grover_bitstrings(keys, it, measure)
        measured_by_builder = True
        notes.append("search qubits 0.." + String(n - 1) + "; " + String(c.num_qubits - n) + " clean ancilla(s); iterations = " + String(it if it >= 0 else opt) + " (optimal " + String(opt) + ")")
        if measure:
            notes.append("measure=true measures the search qubits only; count keys are the marked items")
    elif kind == "bernstein_vazirani":
        _check_args(args, kind, ["secret", "n"])
        var sv = args.get("secret")
        if not sv.is_defined() or sv.is_null():
            raise Error("QM-E801: kind 'bernstein_vazirani' needs 'secret' (bitstring like \"1011\" or an integer with 'n'). Hint: {\"secret\": \"1011\"}.")
        var n: Int
        var secret: Int
        if sv.is_string():
            var key = sv.as_string()
            n = key.byte_length()
            if n < 1 or n > CAP_WIDE:
                raise Error("QM-E802: bernstein_vazirani secret length " + String(n) + " is outside [1, " + String(CAP_WIDE) + "].")
            secret = bitstring_to_int(key)
            if args.has("n") and _int_arg(args, "n", n, 1, CAP_WIDE, kind) != n:
                raise Error("QM-E801: 'n' does not match the length of 'secret'.")
        else:
            n = _int_arg(args, "n", -1, 1, CAP_WIDE, kind)
            secret = _int_arg(args, "secret", -1, 0, (1 << min(n, 62)) - 1, kind)
        c = bernstein_vazirani(n, secret, measure)
        measured_by_builder = True
    elif kind == "deutsch_jozsa":
        _check_args(args, kind, ["n", "balanced_mask", "constant_value"])
        var n = _int_arg(args, "n", -1, 1, 62, kind)
        var mask = _int_arg(args, "balanced_mask", 0, 0, (1 << n) - 1, kind)
        var cv = _int_arg(args, "constant_value", 0, 0, 1, kind)
        c = deutsch_jozsa(n, mask, cv, measure)
        measured_by_builder = True
        notes.append("inputs 0.." + String(n - 1) + ", oracle ancilla " + String(n) + "; all-zero input readout <=> constant")
    elif kind == "qaoa_maxcut":
        _check_args(args, kind, ["graph", "p"])
        var gtext = _str_arg(args, "graph", "")
        if gtext.byte_length() == 0:
            raise Error("QM-E801: kind 'qaoa_maxcut' needs 'graph', an edge list such as \"0-1 1-2 2-0:0.5\".")
        var g = Graph.parse(gtext)
        if g.num_nodes > CAP_WIDE or len(g.edges) > CAP_QAOA_EDGES:
            raise Error("QM-E802: qaoa_maxcut graph has " + String(g.num_nodes) + " nodes / " + String(len(g.edges)) + " edges; caps are " + String(CAP_WIDE) + " / " + String(CAP_QAOA_EDGES) + ".")
        if len(g.edges) == 0:
            raise Error("QM-E801: qaoa_maxcut graph has no edges.")
        var p = _int_arg(args, "p", 1, 1, CAP_QAOA_P, kind)
        c = qaoa_maxcut_ansatz(g, p)
        observable = String(maxcut(g, True))
        notes.append("parameters gamma[0..p-1] then beta[0..p-1]; edge (u,v,w) -> RZZ(w*gamma) on (u,v); mixer RX(2*beta)")
        notes.append("observable H = sum_edges w/2 (Z_u Z_v - 1): <H> = -expected cut, ground energy = -max cut")
        notes.append("gamma and beta are shared across gates: use adjoint or gradient-free optimisation, not per-gate parameter shift")
    elif kind == "hea":
        _check_args(args, kind, ["n", "layers", "entanglement", "rotations", "hamiltonian"])
        var n = _int_arg(args, "n", -1, 1, CAP_WIDE, kind)
        var layers = _int_arg(args, "layers", 2, 0, CAP_HEA_LAYERS, kind)
        var ent = _str_arg(args, "entanglement", "linear")
        var rots = _str_arg(args, "rotations", "ry,rz")
        c = hardware_efficient_ansatz(n, layers, ent, rots)
        observable = _hamiltonian_arg(args, "hamiltonian", n, False, kind)
        notes.append("parameters theta[k], layer-major then qubit then rotation")
    elif kind == "uccsd":
        _check_args(args, kind, ["num_qubits", "num_electrons", "singles", "doubles", "hamiltonian"])
        var nq = _int_arg(args, "num_qubits", -1, 2, CAP_UCCSD, kind)
        var ne = _int_arg(args, "num_electrons", -1, 1, nq - 1, kind)
        c = uccsd_ansatz(nq, ne, _bool_arg(args, "singles", True), _bool_arg(args, "doubles", True))
        observable = _hamiltonian_arg(args, "hamiltonian", nq, False, kind)
        notes.append("Jordan-Wigner, interleaved spin orbitals; Hartree-Fock reference X on qubits 0.." + String(ne - 1))
    elif kind == "h2_ansatz":
        _check_args(args, kind, List[String]())
        c = h2_ansatz()
        observable = String(h2_sto3g(4))
        notes.append("H2 STO-3G at 0.735 A, 4 qubits (q0 g-up, q1 g-down, q2 u-up, q3 u-down); exact ground energy -1.1373060357533993 Ha")
    elif kind == "trotter":
        _check_args(args, kind, ["hamiltonian", "time", "steps", "order", "num_qubits", "param_name"])
        var nq = _int_arg(args, "num_qubits", 0, 0, CAP_WIDE, kind)
        var htext = _hamiltonian_arg(args, "hamiltonian", nq, True, kind)
        var h = parse_pauli_sum(htext)
        if h.num_qubits() > CAP_WIDE:
            raise Error("QM-E802: trotter Hamiltonian acts on " + String(h.num_qubits()) + " qubits; the cap is " + String(CAP_WIDE) + ".")
        var t = _float_arg(args, "time", 1.0, -1e6, 1e6)
        var steps = _int_arg(args, "steps", 1, 1, CAP_TROTTER_STEPS, kind)
        var order = _int_arg(args, "order", 1, 1, 2, kind)
        var pname = _str_arg(args, "param_name", "")
        c = trotter(h, t, steps, order, nq if nq > 0 else -1, pname)
        observable = htext
        notes.append("product formula of order " + String(order) + " with " + String(steps) + " step(s); identity terms become a global phase")
    elif kind == "reuploading_classifier":
        _check_args(args, kind, ["n", "layers", "num_features"])
        var n = _int_arg(args, "n", 1, 1, CAP_QML_QUBITS, kind)
        var layers = _int_arg(args, "layers", 3, 1, CAP_QML_LAYERS, kind)
        var nf = _int_arg(args, "num_features", 3, 1, 3 * n, kind)
        c = Circuit(n, 0, "reuploading_classifier")
        var w = c.parameters("w", 3 * n * layers)  # weights first
        var x = c.parameters("x", nf)  # then the data point, re-uploaded in every layer
        for l in range(layers):
            for q in range(n):
                append_rot(c, x[(3 * q) % nf], x[(3 * q + 1) % nf], x[(3 * q + 2) % nf], q)
                var b = 3 * (l * n + q)
                append_rot(c, w[b], w[b + 1], w[b + 2], q)
            if n > 1:
                for q in range(n - 1):
                    c.cz(q, q + 1)
        observable = "0.5 + 0.5 * Z0"
        notes.append("data re-uploading (Perez-Salinas et al.; PennyLane tutorial_data_reuploading_classifier): per layer and qubit Rot(x) then Rot(w) (Rot = RZ RY RZ), then a CZ chain when n > 1")
        notes.append("parameters w[0..3*n*layers-1] (layer-major, then qubit) then x[0..num_features-1]; qubit q uploads x[(3q+k) mod num_features], k = 0..2")
        notes.append("observable = fidelity of qubit 0 with |0> (label 0); predict label 1 when it is below 0.5. n=1, layers=3, num_features=3 is the PennyLane demo circuit")
    elif kind == "angle_kernel":
        _check_args(args, kind, ["n", "rotation"])
        var n = _int_arg(args, "n", -1, 1, CAP_QML_QUBITS, kind)
        var rot = _str_arg(args, "rotation", "x")
        var embed = Circuit(n, 0, "angle_embedding")
        var feats = embed.parameters("x", n)
        var wires = List[Int]()
        for q in range(n):
            wires.append(q)
        append_angle_embedding(embed, feats, wires, rot)
        c = kernel_circuit(embed, n)
        observable = String(projector_zero(n))
        notes.append("compute-uncompute kernel U(xa) U(xb)^dagger with PennyLane AngleEmbedding (rotation " + rot + "): k(a, b) = <|0..0><0..0|>")
        notes.append("parameters xa[0..n-1] then xb[0..n-1]; on hardware k(a, b) = fraction of all-zero shots")
    else:
        raise Error("QM-E801: unknown circuit kind '" + kind + "'. Hint: qumo_build_circuit kinds: " + String(BUILD_CIRCUIT_KINDS).replace(",", ", ") + ".")
    if len(c.ops) > CAP_GATES:
        raise Error("QM-E802: the circuit has " + String(len(c.ops)) + " instructions; the cap is " + String(CAP_GATES) + ". Hint: reduce the size or depth.")
    if measure and not measured_by_builder:
        c.measure_all()
    return BuiltCircuit(c^, observable^, notes^)


def write_built_circuit(b: BuiltCircuit, kind: String, format: String, mut w: JsonWriter) raises:
    var text: String
    if format == "qasm2":
        text = to_qasm2(b.circuit)
    else:
        text = to_qasm(b.circuit)
    var counts = List[Int](length=Op.NUM_OPS, fill=0)
    for g in b.circuit.ops:
        if g.kind >= 0 and g.kind < Op.NUM_OPS:
            counts[g.kind] += 1
    w.begin_object()
    w.kv_string("kind", kind)
    w.kv_string("format", format)
    w.kv_string("qasm", text)
    w.kv_int("num_qubits", b.circuit.num_qubits)
    w.kv_int("num_clbits", b.circuit.num_clbits)
    w.kv_string_list("parameters", b.circuit.param_names)
    w.kv_int("num_parameters", b.circuit.num_params())
    w.key("gate_counts")
    w.begin_object()
    for k in range(Op.NUM_OPS):
        if counts[k] > 0:
            w.kv_int(op_name(k), counts[k])
    w.end_object()
    w.kv_int("two_qubit_gates", b.circuit.num_2q_gates())
    w.kv_int("depth", b.circuit.depth())
    w.kv_string("observable", b.observable)
    w.kv_string("bit_order", "little-endian qubits (bit k of a basis index = qubit k); count keys: leftmost = highest clbit")
    w.kv_string_list("notes", b.notes)
    w.end_object()


def _build_circuit_tool(args: JsonValue, ctx: ToolContext) raises -> ToolResult:
    var kv = args.get("kind")
    if not kv.is_defined() or not kv.is_string():
        raise Error("QM-E801: qumo_build_circuit needs 'kind' (string). Hint: one of " + String(BUILD_CIRCUIT_KINDS).replace(",", ", ") + ".")
    for k in args.keys():
        if k != "kind" and k != "args" and k != "measure" and k != "format":
            raise Error("QM-E801: unknown field '" + k + "'. Hint: qumo_build_circuit takes kind, args, measure and format.")
    var kind = kv.as_string()
    var format = args.get_string("format", "qasm3")
    if format != "qasm3" and format != "qasm2":
        raise Error("QM-E801: format must be 'qasm3' or 'qasm2', got '" + format + "'.")
    var measure = args.get_bool("measure", False)
    var b = build_algorithm_circuit(kind, args.get("args"), measure)
    var w = JsonWriter()
    write_built_circuit(b, kind, format, w)
    return tool_ok(w.finish())


comptime BUILD_CIRCUIT_SCHEMA = "{\"type\":\"object\",\"properties\":{\"kind\":{\"type\":\"string\",\"enum\":[\"ghz\",\"qft\",\"iqft\",\"grover\",\"bernstein_vazirani\",\"deutsch_jozsa\",\"qaoa_maxcut\",\"hea\",\"uccsd\",\"h2_ansatz\",\"trotter\",\"reuploading_classifier\",\"angle_kernel\"],\"description\":\"Algorithm from QUMO's library\"},\"args\":{\"type\":\"object\",\"description\":\"Kind-specific arguments: ghz {n}; qft/iqft {n, do_swaps, approximation_degree}; grover {marked: [bitstrings], iterations}; bernstein_vazirani {secret: bitstring}; deutsch_jozsa {n, balanced_mask, constant_value}; qaoa_maxcut {graph: '0-1 1-2:0.5', p}; hea {n, layers, entanglement, rotations, hamiltonian}; uccsd {num_qubits, num_electrons, singles, doubles, hamiltonian}; h2_ansatz {}; trotter {hamiltonian, time, steps, order, num_qubits, param_name}; reuploading_classifier {n, layers, num_features}; angle_kernel {n, rotation}\"},\"measure\":{\"type\":\"boolean\",\"default\":false,\"description\":\"Append measurements (search/input qubits only for grover, bernstein_vazirani, deutsch_jozsa)\"},\"format\":{\"type\":\"string\",\"enum\":[\"qasm3\",\"qasm2\"],\"default\":\"qasm3\",\"description\":\"qasm2 needs a circuit without symbolic parameters\"}},\"required\":[\"kind\"],\"additionalProperties\":false}"


def register_algorithm_tools(mut reg: ToolRegistry):
    reg.add(
        "qumo_build_circuit", "Build an algorithm circuit",
        "Build a circuit from QUMO's algorithm library by name instead of writing gates: ghz, qft, iqft, grover (marked bitstrings), bernstein_vazirani, deutsch_jozsa, qaoa_maxcut (edge-list graph, depth p), hea (hardware-efficient ansatz), uccsd, h2_ansatz, trotter (Pauli-sum Hamiltonian), reuploading_classifier (data re-uploading QML classifier), angle_kernel (compute-uncompute quantum kernel). Returns OpenQASM 3 with symbolic parameters as 'input float' (or OpenQASM 2 for parameter-free circuits), parameter names in order, gate counts, two-qubit gates, depth, the observable to measure (MaxCut cost, molecular or given Hamiltonian) and notes. Sizes are capped; unknown arguments are rejected (QM-E801/QM-E802).",
        BUILD_CIRCUIT_SCHEMA,
        _build_circuit_tool,
    )
