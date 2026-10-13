"""Agent tools for compilation, resource estimation, variational algorithms,
QEC experiments and device-explicit simulation (CPU / GPU).

Registered in `qumo.ai.registry.full_registry()`, so they appear in MCP
`tools/list`, `qumo tool <name> '<json>'` and the capabilities document.
Targets are vendor-neutral one-line specs (`qumo.compiler.target.parse_target`)
or explicit coupling edges + basis names.
"""

from std.time import perf_counter_ns

from qumo.ir.circuit import Circuit
from qumo.ir.pauli import parse_pauli_sum
from qumo.ir.qasm import to_qasm
from qumo.ai.json import JsonWriter, JsonValue
from qumo.ai.tools import (
    ToolRegistry, ToolResult, ToolContext, tool_ok, arg_circuit, arg_int, arg_required, arg_params,
    write_warnings,
)
from qumo.exec.dispatch import sample_on
from qumo.gpu.device import query_gpu, GPU_MIN_QUBITS
from qumo.compiler.target import Target, parse_target
from qumo.compiler.pass_manager import transpile, compile_spec
from qumo.exec.resources import (
    estimate_resources, estimate_hardware, estimate_fault_tolerant, HardwareProfile,
)
from qumo.algorithms.hamiltonians import Graph
from qumo.algorithms.library import hardware_efficient_ansatz
from qumo.algorithms.variational import vqe, qaoa_maxcut
from qumo.algorithms.gradients import grad_method_from_name
from qumo.qec.experiments import memory_experiment
from qumo.qec.codes import NoiseParams, generate_code
from qumo.qec.dem import detector_error_model


def _target_from_args(args: JsonValue, n: Int) raises -> Target:
    var spec = args.get_string("target", "")
    if spec.byte_length() > 0:
        return parse_target(spec)
    var basis = args.get_string("basis", "any")
    var e = args.get("edges")
    var width = args.get_int("num_qubits", n)
    if e.is_defined() and not e.is_null() and len(e) > 0:
        var flat = List[Int]()
        for i in range(len(e)):
            var pair = e.at(i)
            flat.append(pair.at(0).as_int())
            flat.append(pair.at(1).as_int())
        return Target.from_edges(width, flat, basis)
    return Target.all_to_all(width, basis)


def _ms(t0: Int, t1: Int) -> Float64:
    return Float64(t1 - t0) / 1e6


def _transpile_tool(args: JsonValue, ctx: ToolContext) raises -> ToolResult:
    var warnings = List[String]()
    var c = arg_circuit(args, warnings)
    var t = _target_from_args(args, c.num_qubits)
    var level = arg_int(args, "level", 2, 0, 3)
    var seed = arg_int(args, "seed", 7, 0, 1 << 62)
    var t0 = perf_counter_ns()
    var spec = args.get_string("pipeline", "")
    var r = compile_spec(c, t, spec, seed) if spec.byte_length() > 0 else transpile(c, t, level, seed)
    var t1 = perf_counter_ns()
    var w = JsonWriter()
    w.begin_object()
    w.kv_raw("target", t.describe())
    w.kv_string("pipeline", r.pipeline)
    w.kv_int("num_qubits", r.circuit.num_qubits)
    w.kv_int("gates_before", r.gates_before)
    w.kv_int("gates_after", r.circuit.num_unitary_gates())
    w.kv_int("two_qubit_before", r.twoq_before)
    w.kv_int("two_qubit_after", r.circuit.num_2q_gates())
    w.kv_int("depth_before", r.depth_before)
    w.kv_int("depth_after", r.circuit.depth())
    w.kv_int("swaps", r.num_swaps())
    w.kv_float("estimated_fidelity", r.est_fidelity)
    w.kv_int_list("initial_layout", r.initial_layout)
    w.kv_int_list("final_layout", r.final_layout)
    w.kv_raw("report", r.to_json())
    if args.get_bool("include_qasm", True):
        w.kv_string("qasm", to_qasm(r.circuit))
    w.kv_float("compile_ms", _ms(t0, t1))
    w.end_object()
    return tool_ok(w.finish())


def _resources_tool(args: JsonValue, ctx: ToolContext) raises -> ToolResult:
    var warnings = List[String]()
    var c = arg_circuit(args, warnings)
    var w = JsonWriter()
    w.begin_object()
    w.kv_raw("logical", estimate_resources(c).to_json())
    var prof = args.get_string("profile", "superconducting")
    var hp = HardwareProfile.superconducting()
    if prof == "trapped_ion":
        hp = HardwareProfile.trapped_ion()
    elif prof == "neutral_atom":
        hp = HardwareProfile.neutral_atom()
    w.kv_raw("hardware", estimate_hardware(c, hp).to_json())
    var p = args.get_float("physical_error", 1e-3)
    var budget = args.get_float("error_budget", 1e-2)
    w.kv_raw("fault_tolerant", estimate_fault_tolerant(c, p, budget).to_json())
    w.end_object()
    return tool_ok(w.finish())


def _vqe_tool(args: JsonValue, ctx: ToolContext) raises -> ToolResult:
    var h = parse_pauli_sum(arg_required(args, "hamiltonian", "Pauli sum string").as_string())
    var n = max(h.num_qubits(), args.get_int("num_qubits", 0))
    var layers = arg_int(args, "layers", 2, 1, 50)
    var ent = args.get_string("entanglement", "linear")
    var ansatz = hardware_efficient_ansatz(n, layers, ent)
    var method = args.get_string("optimizer", "bfgs")
    var grad = grad_method_from_name(args.get_string("gradient", "adjoint"))
    var maxiter = arg_int(args, "maxiter", 200, 1, 100000)
    var seed = arg_int(args, "seed", 7, 0, 1 << 62)
    var r = vqe(ansatz, h, method, List[Float64](), maxiter, 1e-8, grad, UInt64(seed))
    var w = JsonWriter()
    w.begin_object()
    w.kv_float("energy", r.energy)
    w.key("params")
    w.begin_array()
    for v in r.params:
        w.number(v)
    w.end_array()
    w.kv_int("iterations", r.nit)
    w.kv_int("circuit_evaluations", r.circuit_evals)
    w.kv_bool("converged", r.converged)
    w.kv_string("optimizer", r.optimizer)
    w.kv_string("gradient", r.grad_method)
    w.kv_float("wall_time_s", r.wall_time_s)
    w.kv_string("ansatz", "hardware_efficient(n=" + String(n) + ", layers=" + String(layers) + ", " + ent + ")")
    w.end_object()
    return tool_ok(w.finish())


def _qaoa_tool(args: JsonValue, ctx: ToolContext) raises -> ToolResult:
    var g = Graph.parse(arg_required(args, "graph", "edge list like '0-1 1-2 2-0:0.5'").as_string())
    var p = arg_int(args, "p", 1, 1, 20)
    var seed = arg_int(args, "seed", 7, 0, 1 << 62)
    var shots = arg_int(args, "shots", 1024, 1, 1000000)
    var r = qaoa_maxcut(g, p, "bfgs", 4, shots, UInt64(seed))
    var w = JsonWriter()
    w.begin_object()
    w.kv_int("p", r.p)
    w.kv_float("expected_cut", r.expected_cut)
    w.kv_float("max_cut", r.max_cut)
    w.kv_float("approximation_ratio", r.approx_ratio)
    w.kv_string("best_bitstring", r.best_bitstring)
    w.kv_float("best_cut", r.best_cut)
    w.key("gammas")
    w.begin_array()
    for v in r.gammas:
        w.number(v)
    w.end_array()
    w.key("betas")
    w.begin_array()
    for v in r.betas:
        w.number(v)
    w.end_array()
    w.end_object()
    return tool_ok(w.finish())


def _qec_memory_tool(args: JsonValue, ctx: ToolContext) raises -> ToolResult:
    var code = args.get_string("code", "surface_z")
    var d = arg_int(args, "distance", 3, 2, 25)
    var rounds = arg_int(args, "rounds", -1, -1, 1000)
    var p = args.get_float("p", 0.001)
    var shots = arg_int(args, "shots", 100000, 1, 100000000)
    var decoder = args.get_string("decoder", "mwpm")
    var seed = arg_int(args, "seed", 1, 0, 1 << 62)
    var max_errors = arg_int(args, "max_errors", -1, -1, 1 << 40)
    var r = memory_experiment(code, d, rounds, p, shots, decoder, UInt64(seed), max_errors)
    return tool_ok(r.to_json())


def _qec_dem_tool(args: JsonValue, ctx: ToolContext) raises -> ToolResult:
    var code = args.get_string("code", "surface_z")
    var d = arg_int(args, "distance", 3, 2, 15)
    var rounds = arg_int(args, "rounds", -1, -1, 100)
    var p = args.get_float("p", 0.001)
    var cc = generate_code(code, d, rounds if rounds > 0 else d, NoiseParams.uniform(p))
    var dem = detector_error_model(cc.circuit)
    var w = JsonWriter()
    w.begin_object()
    w.kv_int("num_qubits", cc.circuit.num_qubits)
    w.kv_int("num_detectors", cc.circuit.num_detectors)
    w.kv_int("num_errors", len(dem.errors))
    if args.get_bool("include_stim", False):
        w.kv_string("stim_circuit", cc.to_stim())
    w.kv_string("dem", dem.to_text())
    w.end_object()
    return tool_ok(w.finish())


def _gpu_info_tool(args: JsonValue, ctx: ToolContext) raises -> ToolResult:
    return tool_ok(query_gpu().to_json())


def _run_device_tool(args: JsonValue, ctx: ToolContext) raises -> ToolResult:
    var t0 = perf_counter_ns()
    var warnings = List[String]()
    var c = arg_circuit(args, warnings)
    var shots = arg_int(args, "shots", 1024, 1, 10000000)
    var seed = arg_int(args, "seed", 1234, 0, 9223372036854775807)
    var device = args.get_string("device", "auto")
    var max_q = arg_int(args, "max_qubits", 30, 1, 40)
    if c.num_qubits > max_q and not (c.num_params() == 0 and c.is_clifford()):
        raise Error("QM-E930: circuit has " + String(c.num_qubits) + " qubits; dense state-vector simulation is limited to max_qubits=" + String(max_q) + ". Hint: raise 'max_qubits' if the device has the memory (8*2^n bytes in fp32, 16*2^n in fp64).")
    var params = arg_params(args, c)
    var auto = False
    if c.num_measurements == 0:
        c.measure_all()
        auto = True
    var note = String()
    var r = sample_on(c, shots, UInt64(seed), params, device, note)
    var t1 = perf_counter_ns()
    if note.byte_length() > 0:
        warnings.append("QM-W020: device 'gpu' requested but " + note + ". Hint: check qumo_gpu_info; GPU runs need a build made with -D QUMO_GPU (make cli-gpu) and a supported GPU.")
    if auto:
        warnings.append("QM-W004: circuit had no measurements; all qubits were measured into c[0..n-1]. Hint: add 'c = measure q;' to control readout.")
    var w = JsonWriter()
    w.begin_object()
    w.key("counts")
    w.begin_object()
    for k in r.sorted_keys():
        w.kv_int(k, r.counts.get(k, 0))
    w.end_object()
    w.kv_int("shots", shots)
    w.kv_int("seed", seed)
    w.kv_string("device", device)
    w.kv_string("device_used", "gpu" if r.method.startswith("gpu") else "cpu")
    w.kv_string("method", r.method)
    w.kv_int("num_qubits", c.num_qubits)
    w.kv_string("most_frequent", r.most_frequent())
    w.kv_string("bit_order", "leftmost character = highest classical bit c[n-1]")
    write_warnings(w, warnings)
    w.kv_float("total_ms", _ms(t0, t1))
    w.end_object()
    return tool_ok(w.finish())


comptime _CIRC = "\"qasm\":{\"type\":\"string\",\"description\":\"OpenQASM 2 or 3 program\"},\"circuit\":{\"type\":\"object\",\"description\":\"qumo.circuit.v1 JSON\"}"


def register_exec_tools(mut reg: ToolRegistry):
    reg.add(
        "qumo_transpile", "Compile for a target",
        "Optimize, map and route a circuit for ANY target (vendor-neutral): either a one-line spec 'target' like 'grid:4x4/sc_cz', 'heavy_hex:5/rz,sx,x,ecr', 'trapped_ion:11', 'neutral_atom:4x4', 'line:20/u,cx', or explicit 'edges' [[a,b],...] + 'basis' 'rz,sx,x,cz'. 'level' 0-3 picks a preset; 'pipeline' runs a text pass spec, e.g. 'cancel,merge,fuse1q,route(sabre),basis,cancel'. Returns compiled OpenQASM 3, per-pass report, layouts, swap count and estimated fidelity.",
        "{\"type\":\"object\",\"properties\":{" + _CIRC + ",\"target\":{\"type\":\"string\"},\"edges\":{\"type\":\"array\",\"items\":{\"type\":\"array\",\"items\":{\"type\":\"integer\"}}},\"num_qubits\":{\"type\":\"integer\"},\"basis\":{\"type\":\"string\",\"default\":\"any\"},\"level\":{\"type\":\"integer\",\"minimum\":0,\"maximum\":3,\"default\":2},\"pipeline\":{\"type\":\"string\"},\"seed\":{\"type\":\"integer\",\"default\":7},\"include_qasm\":{\"type\":\"boolean\",\"default\":true}},\"additionalProperties\":false}",
        _transpile_tool,
    )
    reg.add(
        "qumo_estimate_resources", "Estimate resources",
        "Plan before running: logical metrics (gates, depth, 2q, T-count, Clifford, memory, recommended simulator), NISQ estimate for a modality profile ('superconducting' | 'trapped_ion' | 'neutral_atom': duration, success probability) and a surface-code fault-tolerant estimate (code distance, physical qubits, T factories, runtime) for a physical error rate and error budget.",
        "{\"type\":\"object\",\"properties\":{" + _CIRC + ",\"profile\":{\"type\":\"string\",\"enum\":[\"superconducting\",\"trapped_ion\",\"neutral_atom\"],\"default\":\"superconducting\"},\"physical_error\":{\"type\":\"number\",\"default\":0.001},\"error_budget\":{\"type\":\"number\",\"default\":0.01}},\"additionalProperties\":false}",
        _resources_tool,
    )
    reg.add(
        "qumo_vqe", "Variational eigensolver",
        "Find the ground-state energy of a Pauli-sum Hamiltonian (e.g. '-1.05*II + 0.39*ZI - 0.39*IZ - 0.01*ZZ + 0.18*XX') with a hardware-efficient ansatz, exact adjoint gradients and BFGS/Adam/SPSA/Nelder-Mead. Deterministic for a seed.",
        "{\"type\":\"object\",\"properties\":{\"hamiltonian\":{\"type\":\"string\"},\"num_qubits\":{\"type\":\"integer\"},\"layers\":{\"type\":\"integer\",\"default\":2},\"entanglement\":{\"type\":\"string\",\"enum\":[\"linear\",\"circular\",\"full\",\"none\"],\"default\":\"linear\"},\"optimizer\":{\"type\":\"string\",\"enum\":[\"bfgs\",\"adam\",\"gd\",\"spsa\",\"nelder-mead\"],\"default\":\"bfgs\"},\"gradient\":{\"type\":\"string\",\"enum\":[\"adjoint\",\"parameter-shift\",\"finite-diff\"],\"default\":\"adjoint\"},\"maxiter\":{\"type\":\"integer\",\"default\":200},\"seed\":{\"type\":\"integer\",\"default\":7}},\"required\":[\"hamiltonian\"],\"additionalProperties\":false}",
        _vqe_tool,
    )
    reg.add(
        "qumo_qaoa_maxcut", "QAOA MaxCut",
        "Solve MaxCut with depth-p QAOA on a graph given as an edge list '0-1 1-2 2-3:0.5' (optional :weight). Returns expected cut, brute-force optimum, approximation ratio, best sampled bitstring and optimal angles.",
        "{\"type\":\"object\",\"properties\":{\"graph\":{\"type\":\"string\"},\"p\":{\"type\":\"integer\",\"default\":1},\"shots\":{\"type\":\"integer\",\"default\":1024},\"seed\":{\"type\":\"integer\",\"default\":7}},\"required\":[\"graph\"],\"additionalProperties\":false}",
        _qaoa_tool,
    )
    reg.add(
        "qumo_qec_memory", "QEC memory experiment",
        "Sinter-style logical error rate of a memory experiment: code 'surface_z' | 'surface_x' | 'repetition', distance d, rounds (default d), uniform circuit-level noise p (Stim SI1000-like parameters), decoder 'mwpm' (default: exact minimum-weight perfect matching by sparse blossom, PyMatching 2's algorithm; aliases 'matching', 'pymatching') | 'sparse_blossom' (same decoder) | 'mwpm_dense' (exact MWPM by all-pairs shortest paths + dense blossom; small codes, d <= 5) | 'uf' (union-find; fast, slightly lower threshold) | 'lookup' (lookup table; tiny codes, <= 64 detectors). Bit-packed Pauli-frame sampling; returns logical error rate, std error, Wilson interval, per-round rate, timing and the decoder that ran.",
        "{\"type\":\"object\",\"properties\":{\"code\":{\"type\":\"string\",\"enum\":[\"surface_z\",\"surface_x\",\"repetition\"],\"default\":\"surface_z\"},\"distance\":{\"type\":\"integer\",\"default\":3},\"rounds\":{\"type\":\"integer\",\"default\":-1},\"p\":{\"type\":\"number\",\"default\":0.001},\"shots\":{\"type\":\"integer\",\"default\":100000},\"decoder\":{\"type\":\"string\",\"enum\":[\"mwpm\",\"sparse_blossom\",\"mwpm_dense\",\"uf\",\"lookup\"],\"default\":\"mwpm\"},\"seed\":{\"type\":\"integer\",\"default\":1},\"max_errors\":{\"type\":\"integer\",\"default\":-1}},\"additionalProperties\":false}",
        _qec_memory_tool,
    )
    reg.add(
        "qumo_qec_dem", "Detector error model",
        "Generate a QEC memory circuit and return its detector error model in Stim DEM text ('error(p) D0 D1 L0', with '^' decompositions), optionally the Stim circuit text for cross-checking with Stim/PyMatching.",
        "{\"type\":\"object\",\"properties\":{\"code\":{\"type\":\"string\",\"enum\":[\"surface_z\",\"surface_x\",\"repetition\"],\"default\":\"surface_z\"},\"distance\":{\"type\":\"integer\",\"default\":3},\"rounds\":{\"type\":\"integer\",\"default\":-1},\"p\":{\"type\":\"number\",\"default\":0.001},\"include_stim\":{\"type\":\"boolean\",\"default\":false}},\"additionalProperties\":false}",
        _qec_dem_tool,
    )
    reg.add(
        "qumo_gpu_info", "GPU status",
        "Report whether this QUMO build has GPU support (-D QUMO_GPU) and a usable device: name, API (cuda/hip/metal), vendor, default precision (fp32 on Apple, fp64 on NVIDIA/AMD), fp64 support and the qubit threshold above which automatic dispatch uses the GPU. Call it before qumo_run_device with device 'gpu'.",
        "{\"type\":\"object\",\"properties\":{},\"additionalProperties\":false}",
        _gpu_info_tool,
    )
    reg.add(
        "qumo_run_device", "Run circuit on a device",
        "Like qumo_run_qasm but with an explicit simulator device (vendor-neutral: NVIDIA CUDA, AMD ROCm or Apple Metal through MAX). 'auto' (default) = qumo_run_qasm's dispatch: noisy Clifford circuits without feed-forward -> CPU Pauli frames, other wide/noisy/dynamic Clifford circuits -> stabilizer tableau, static non-Clifford circuits with >= " + String(GPU_MIN_QUBITS) + " qubits -> GPU state vector when this build has GPU support (-D QUMO_GPU) and a usable device (CPU on any GPU failure), everything else -> CPU state vector. 'cpu' = the same choice restricted to CPU simulators. 'gpu' = Clifford circuits without parameters or classical feed-forward (noise, mid-circuit measurement and reset allowed) -> GPU Pauli frames ('gpu:pauli_frame'); other static, noiseless circuits -> GPU state vector ('gpu:statevector:fp32' on Apple, fp64 on NVIDIA/AMD); dynamic or noisy non-Clifford circuits, builds without -D QUMO_GPU, or no usable device -> run on the CPU as with 'cpu' plus warning QM-W020 giving the reason. Returns counts, 'method' (the simulator that ran), 'device_used' and structured 'warnings'.",
        "{\"type\":\"object\",\"properties\":{" + _CIRC + ",\"shots\":{\"type\":\"integer\",\"minimum\":1,\"default\":1024},\"seed\":{\"type\":\"integer\",\"minimum\":0,\"default\":1234},\"params\":{\"description\":\"Values for 'input float' parameters: array in declaration order or object name->value\"},\"device\":{\"type\":\"string\",\"enum\":[\"auto\",\"cpu\",\"gpu\"],\"default\":\"auto\"},\"max_qubits\":{\"type\":\"integer\",\"default\":30}},\"additionalProperties\":false}",
        _run_device_tool,
    )
