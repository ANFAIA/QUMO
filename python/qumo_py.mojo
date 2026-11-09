"""Python bindings for QUMO (CPython extension built from Mojo).

    mojo build -I . --emit shared-lib -o build/qumo_py.so python/qumo_py.mojo

Exposes opaque `Circuit` and `Program` handles plus free functions. Kept
deliberately small and stable: build/load circuits, compile them for any target
(coupling map + basis as plain Python data, vendor-neutral), inspect
statistics, simulate. A `Program` adds classical variables and control flow at
run time: `load_program`, `sample_program`, `transpile_program`. Used by the
Benchpress `qumo_gym` and by `python/qumo/__init__.py`.
"""

from std.python import PythonObject, Python
from std.python.bindings import PythonModuleBuilder
from std.os import abort, setenv, unsetenv
from std.time import perf_counter_ns

from qumo.ir.circuit import Circuit
from qumo.ir.mc import append_mcx
from qumo.sim.ptsbe import sample_ptsbe, plan_trajectories
from qumo.ir.noise_inject import inject_noise
from qumo.exec.quantinuum_api import qtm_native_circuit, qtm_hqc, qtm_submit, qtm_status, qtm_result, qtm_devices
from qumo.exec.ibm_api import ibm_backends, ibm_target, ibm_isa_payload, ibm_submit, ibm_status, ibm_result
from qumo.exec.speqtrum_api import (
    SpeQtrumAuth, checked_base, payload_document, unsafe_construct, decode_result,
    speqtrum_devices, speqtrum_execute, speqtrum_status, speqtrum_result,
)
from qumo.exec.ionq_api import ionq_request, ionq_counts, ionq_submit, ionq_status, ionq_result, ionq_backends
from qumo.sim.runner import SampleResult
from qumo.sim.branching import sample_branching
from qumo.sim.density import density_evolve
from qumo.compiler.pbc import to_pbc
from qumo.qec.characterization import ReadoutCalibration
from qumo.qec.mthree import mitigate_counts_m3
from qumo.qec.codes import to_stim
from qumo.qec.dem import detector_error_model
from qumo.qec.decoders import MatchingGraph
from qumo.qec.sparse_blossom import SparseBlossomDecoder
from qumo.qec.correlated import CorrelatedMatchingDecoder
from qumo.qec.stim_text import from_stim, parse_dem, sample_dem
from qumo.qec.lint import missing_detectors, count_determined_measurements
from qumo.ai.json import JsonWriter
from std.collections import Dict
from qumo.ir.gates import Op, op_name, op_arity, op_is_unitary, op_from_name
from qumo.ir.pauli import parse_pauli_sum, PauliSum
from qumo.sim.runner import sample, evolve
from qumo.ir.program import Program
from qumo.ir.qasm import draw_program, parse_qasm, load_any_with_warnings, load_qasm, to_qasm
from qumo.ir.qasm2 import to_qasm2
from qumo.sim.program_runner import DEFAULT_MAX_ITERATIONS, sample_program
from qumo.compiler.program_compile import compile_program
from qumo.ai.describe import program_from_json, program_to_json
from qumo.ai.json import parse_json
from qumo.ai.diagnostics import diagnostic_from_error, lint_program
from qumo.compiler.target import Target
from qumo.exec.dispatch import sample_on
from qumo.gpu.device import query_gpu
from qumo.compiler.pass_manager import transpile, compile_spec, transpile_ai, CompileResult
from qumo.algorithms.gradients import batch_value_and_gradient, Grad
from qumo.algorithms.gradients_device import value_and_gradient_on, expectations_on
from qumo.algorithms.sqd import subspace_ground_state
from qumo.exec.orchestrator import Workflow, Device
from qumo.qec.codes import generate_code, NoiseParams, CodeCircuit
from qumo.qec.dem import detector_error_model, DetectorErrorModel
from qumo.qec.decoders import MatchingGraph, decode_samples, graphlike_dem
from qumo.qec.sparse_blossom import SparseBlossomDecoder
from qumo.sim.frame import FrameSimulator, DetectorSamples, BATCH, FW
from qumo.gpu.frame_gpu import GpuFrameSampler
from qumo.gpu.device import gpu_available
from std.sys.defines import is_defined
from qumo.bench.circuits import (
    quantum_volume, efficient_su2, dtc_unitary, random_clifford_circuit,
    bv_all_ones, trivial_bvlike_circuit, multi_control_circuit, pauli_twirl_2q,
    hamiltonian_circuit_from_labels,
)


struct PyCircuit(Movable, Writable):
    var c: Circuit

    def __init__(out self, var c: Circuit):
        self.c = c^

    @staticmethod
    def _get(py_self: PythonObject) raises -> Pointer[PyCircuit, MutAnyOrigin]:
        return py_self.downcast_value_ptr[Self]().as_unsafe_any_origin()

    @staticmethod
    def num_qubits(py_self: PythonObject) raises -> PythonObject:
        return Self._get(py_self)[].c.num_qubits

    @staticmethod
    def num_clbits(py_self: PythonObject) raises -> PythonObject:
        return Self._get(py_self)[].c.num_clbits

    @staticmethod
    def size(py_self: PythonObject) raises -> PythonObject:
        return Self._get(py_self)[].c.num_unitary_gates()

    @staticmethod
    def num_parameters(py_self: PythonObject) raises -> PythonObject:
        return Self._get(py_self)[].c.num_params()

    @staticmethod
    def depth(py_self: PythonObject) raises -> PythonObject:
        return Self._get(py_self)[].c.depth()

    @staticmethod
    def count_ops(py_self: PythonObject) raises -> PythonObject:
        ref c = Self._get(py_self)[].c
        var counts = List[Int](length=Op.NUM_OPS, fill=0)
        for g in c.ops:
            counts[g.kind] += 1
        var d = Python.dict()
        for k in range(Op.NUM_OPS):
            if counts[k] > 0:
                d[op_name(k)] = counts[k]
        return d

    @staticmethod
    def count_gate(py_self: PythonObject, name: PythonObject) raises -> PythonObject:
        var k = op_from_name(String(py=name))
        if k < 0:
            return 0
        return Self._get(py_self)[].c.count(k)

    @staticmethod
    def depth_of(py_self: PythonObject, name: PythonObject) raises -> PythonObject:
        """Depth counting only gates named `name` (e.g. 2q depth)."""
        ref c = Self._get(py_self)[].c
        var k = op_from_name(String(py=name))
        var level = List[Int](length=c.num_qubits, fill=0)
        var d = 0
        for g in c.ops:
            if not op_is_unitary(g.kind):
                continue
            var a = op_arity(g.kind)
            var l = 0
            for j in range(a):
                l = max(l, level[g.qubit(j)])
            if g.kind == k:
                l += 1
            for j in range(a):
                level[g.qubit(j)] = l
            d = max(d, l)
        return d

    @staticmethod
    def two_qubit_edges(py_self: PythonObject) raises -> PythonObject:
        """List of (q0, q1) for every 2-qubit unitary (for validation)."""
        ref c = Self._get(py_self)[].c
        var out = Python.list()
        for g in c.ops:
            if op_is_unitary(g.kind) and op_arity(g.kind) == 2:
                out.append(Python.tuple(g.q0, g.q1))
        return out

    @staticmethod
    def copy(py_self: PythonObject) raises -> PythonObject:
        return PythonObject(alloc=PyCircuit(Self._get(py_self)[].c.copy()))

    @staticmethod
    def compose(py_self: PythonObject, other: PythonObject) raises -> PythonObject:
        """Return self followed by other (new circuit)."""
        var out = Self._get(py_self)[].c.copy()
        out.compose(Self._get(other)[].c)
        return PythonObject(alloc=PyCircuit(out^))

    @staticmethod
    def bind(py_self: PythonObject, values: PythonObject) raises -> PythonObject:
        var vals = List[Float64]()
        for v in values:
            vals.append(Float64(py=v))
        return PythonObject(alloc=PyCircuit(Self._get(py_self)[].c.bind(vals)))

    @staticmethod
    def to_qasm2(py_self: PythonObject) raises -> PythonObject:
        return to_qasm2(Self._get(py_self)[].c)

    @staticmethod
    def to_qasm3(py_self: PythonObject) raises -> PythonObject:
        return to_qasm(Self._get(py_self)[].c)

    @staticmethod
    def text(py_self: PythonObject) raises -> PythonObject:
        return String(Self._get(py_self)[].c)

    def write_to(self, mut w: Some[Writer]):
        w.write("<qumo.Circuit qubits=", self.c.num_qubits, " ops=", len(self.c.ops), ">")

    def write_repr_to(self, mut w: Some[Writer]):
        self.write_to(w)


struct PyProgram(Movable, Writable):
    """A circuit with classical variables and control flow at run time."""

    var p: Program
    var warnings: List[String]  # of the parser that read it

    def __init__(out self, var p: Program, var warnings: List[String] = List[String]()):
        self.p = p^
        self.warnings = warnings^

    @staticmethod
    def _get(py_self: PythonObject) raises -> Pointer[PyProgram, MutAnyOrigin]:
        return py_self.downcast_value_ptr[Self]().as_unsafe_any_origin()

    @staticmethod
    def num_qubits(py_self: PythonObject) raises -> PythonObject:
        return Self._get(py_self)[].p.num_qubits()

    @staticmethod
    def num_clbits(py_self: PythonObject) raises -> PythonObject:
        return Self._get(py_self)[].p.num_clbits()

    @staticmethod
    def num_parameters(py_self: PythonObject) raises -> PythonObject:
        return Self._get(py_self)[].p.num_params()

    @staticmethod
    def has_runtime_regions(py_self: PythonObject) raises -> PythonObject:
        """False when the program is one plain circuit."""
        return Self._get(py_self)[].p.has_runtime_region()

    @staticmethod
    def to_circuit(py_self: PythonObject) raises -> PythonObject:
        """The plain circuit of a program without runtime regions (QM-E585 otherwise)."""
        var c = Self._get(py_self)[].p.try_flatten()
        if not c:
            raise Error("QM-E585: this program has classical variables or control flow at run time, so it is not one Circuit. Hint: use sample_program and transpile_program.")
        return PythonObject(alloc=PyCircuit(c.take()))

    @staticmethod
    def to_qasm3(py_self: PythonObject) raises -> PythonObject:
        return to_qasm(Self._get(py_self)[].p)

    @staticmethod
    def to_json(py_self: PythonObject) raises -> PythonObject:
        """Program JSON (qumo.program.v1) as a string."""
        return program_to_json(Self._get(py_self)[].p)

    @staticmethod
    def lint(py_self: PythonObject) raises -> PythonObject:
        """Warnings of the parser and of the program as [(code, message), ...]."""
        var out = Python.list()
        for s in Self._get(py_self)[].warnings:
            var d = diagnostic_from_error(s)
            out.append(Python.tuple(d.code, d.message))
        for d in lint_program(Self._get(py_self)[].p):
            out.append(Python.tuple(d.code, d.message))
        return out

    @staticmethod
    def text(py_self: PythonObject) raises -> PythonObject:
        return String(Self._get(py_self)[].p)

    def write_to(self, mut w: Some[Writer]):
        w.write("<qumo.Program qubits=", self.p.num_qubits(), " statements=", len(self.p.tape), ">")

    def write_repr_to(self, mut w: Some[Writer]):
        self.write_to(w)


def _wrap(var c: Circuit) raises -> PythonObject:
    return PythonObject(alloc=PyCircuit(c^))


def py_load_program(text: PythonObject) raises -> PythonObject:
    """Parse OpenQASM 2/3 into a Program: anything a Circuit holds keeps its exact
    meaning, and while, break, continue and classical variables are added."""
    var warnings = List[String]()
    var p = load_any_with_warnings(String(py=text), warnings)
    return PythonObject(alloc=PyProgram(p^, warnings^))


def py_program_from_json(text: PythonObject) raises -> PythonObject:
    """Inverse of Program.to_json. Malformed input raises QM-E587."""
    return PythonObject(alloc=PyProgram(program_from_json(parse_json(String(py=text)))))


def py_draw_program(prog: PythonObject, width: PythonObject, style: PythonObject) raises -> PythonObject:
    """Text drawing by blocks: (program, width, style) -> str."""
    return PythonObject(draw_program(PyProgram._get(prog)[].p, Int(py=width), String(py=style)))


def py_sample_program(prog: PythonObject, shots: PythonObject, seed: PythonObject,
                      var **kwargs: PythonObject) raises -> PythonObject:
    """Sample a program -> dict with counts, method, iterations (back-edges per
    shot), outputs ({name: [value per shot]}) and warnings.

    Python: sample_program(prog, shots, seed, params=[], max_iterations=65536, method="auto").
    params: values of the `input` declarations, in order. Past max_iterations a shot
    raises QM-E480. method: 'auto' | 'statevector' | 'tableau' | 'branching' | 'gpu'
    ('gpu' runs on the CPU with QM-W020; 'density', 'ptsbe' and 'frame' are QM-E483).
    """
    var params = List[Float64]()
    var max_iterations = DEFAULT_MAX_ITERATIONS
    var method = String("auto")
    for kv in kwargs.items():
        if kv.key == "params":
            params = _floats(kv.value)
        elif kv.key == "max_iterations":
            max_iterations = Int(py=kv.value)
        elif kv.key == "method":
            method = String(py=kv.value)
        else:
            raise Error("QM-E913: sample_program() got an unexpected keyword argument '" + kv.key + "'. Hint: the keyword arguments are params, max_iterations and method.")
    var r = sample_program(PyProgram._get(prog)[].p, Int(py=shots), UInt64(Int(py=seed)), params, max_iterations, method)
    var counts = Python.dict()
    for kv in r.sample.counts.items():
        counts[kv.key] = kv.value
    var iterations = Python.list()
    for v in r.iterations:
        iterations.append(v)
    var outputs = Python.dict()
    for j in range(len(r.out_names)):
        var vals = Python.list()
        if r.out_is_real[j]:
            for v in r.real_output(r.out_names[j]):
                vals.append(v)
        elif r.out_is_unsigned[j]:
            for v in r.uint_output(r.out_names[j]):
                vals.append(v)
        else:
            for v in r.int_output(r.out_names[j]):
                vals.append(v)
        outputs[r.out_names[j]] = vals
    var warnings = Python.list()
    for w in r.warnings:
        warnings.append(w)
    var d = Python.dict()
    d["counts"] = counts
    d["method"] = r.sample.method
    d["iterations"] = iterations
    d["outputs"] = outputs
    d["warnings"] = warnings
    return d


def py_transpile_program(prog: PythonObject, num_qubits: PythonObject, edges: PythonObject,
                         basis: PythonObject, level: PythonObject, seed: PythonObject) raises -> PythonObject:
    """Compile: (program, num_qubits, edges, basis, level, seed) -> (program, initial, final, report).

    Same arguments as `transpile`. Every block lives on one layout, and each region
    leaves on the layout it entered with. The result has passed verify_program;
    report holds metrics, restored_swaps, blocks_checked and blocks_unchecked."""
    var t = _target(num_qubits, edges, basis)
    var r = compile_program(PyProgram._get(prog)[].p, t, Int(py=level), Int(py=seed))
    var init = Python.list()
    for v in r.initial_layout:
        init.append(v)
    var fin = Python.list()
    for v in r.final_layout:
        fin.append(v)
    var d = Python.dict()
    d["metrics"] = String(r.metrics)
    d["restored_swaps"] = r.restored_swaps
    d["blocks_checked"] = r.blocks_checked
    d["blocks_unchecked"] = r.blocks_unchecked
    return Python.tuple(PythonObject(alloc=PyProgram(r.program.copy())), init, fin, d)


def py_quantum_volume(n: PythonObject, depth: PythonObject, seed: PythonObject) raises -> PythonObject:
    return _wrap(quantum_volume(Int(py=n), Int(py=depth), UInt64(Int(py=seed))))


def py_efficient_su2(n: PythonObject, reps: PythonObject, ent: PythonObject) raises -> PythonObject:
    return _wrap(efficient_su2(Int(py=n), Int(py=reps), String(py=ent)))


def py_dtc_unitary(n: PythonObject, g: PythonObject, seed: PythonObject) raises -> PythonObject:
    return _wrap(dtc_unitary(Int(py=n), Float64(py=g), UInt64(Int(py=seed))))


def py_random_clifford(n: PythonObject, num_gates: PythonObject, seed: PythonObject) raises -> PythonObject:
    return _wrap(random_clifford_circuit(Int(py=n), Int(py=num_gates), UInt64(Int(py=seed))))


def py_bv_all_ones(n: PythonObject) raises -> PythonObject:
    return _wrap(bv_all_ones(Int(py=n)))


def py_bvlike(n: PythonObject) raises -> PythonObject:
    return _wrap(trivial_bvlike_circuit(Int(py=n)))


def py_multi_control(n: PythonObject) raises -> PythonObject:
    return _wrap(multi_control_circuit(Int(py=n)))


def py_empty(n: PythonObject) raises -> PythonObject:
    return _wrap(Circuit(Int(py=n)))


def py_twirl(circ: PythonObject, seed: PythonObject) raises -> PythonObject:
    return _wrap(pauli_twirl_2q(PyCircuit._get(circ)[].c, UInt64(Int(py=seed))))


def py_hamiltonian_circuit(labels: PythonObject, coeffs: PythonObject, time: PythonObject) raises -> PythonObject:
    var ls = List[String]()
    for l in labels:
        ls.append(String(py=l))
    var cs = List[Float64]()
    for c in coeffs:
        cs.append(Float64(py=c))
    return _wrap(hamiltonian_circuit_from_labels(ls, cs, Float64(py=time)))


comptime QUMO_VERSION = "0.1.0"


def py_version() raises -> PythonObject:
    return QUMO_VERSION


def py_load_qasm(text: PythonObject) raises -> PythonObject:
    return _wrap(parse_qasm(String(py=text)))


def py_load_qasm_file(path: PythonObject) raises -> PythonObject:
    return _wrap(load_qasm(String(py=path)))


def _basis_spec(basis: PythonObject) raises -> String:
    var spec = String()
    for b in basis:
        if spec.byte_length() > 0:
            spec += ","
        spec += String(py=b)
    return spec


def _target(num_qubits: PythonObject, edges: PythonObject, basis: PythonObject) raises -> Target:
    var n = Int(py=num_qubits)
    var spec = _basis_spec(basis)
    var flat = List[Int]()
    for e in edges:
        flat.append(Int(py=e[0]))
        flat.append(Int(py=e[1]))
    if len(flat) == 0:
        return Target.all_to_all(n, spec)
    return Target.from_edges(n, flat, spec)


def py_transpile(circ: PythonObject, num_qubits: PythonObject, edges: PythonObject,
                 basis: PythonObject, level: PythonObject, seed: PythonObject) raises -> PythonObject:
    """Compile: (circuit, num_qubits, edges, basis, level, seed) -> (circuit, initial, final).

    edges: list of (a, b) physical couplings ([] = all-to-all); basis: gate names.
    initial[l] / final[l]: physical qubit of logical l before / after routing.
    """
    var t = _target(num_qubits, edges, basis)
    var r = transpile(PyCircuit._get(circ)[].c, t, Int(py=level), Int(py=seed))
    var init = Python.list()
    for v in r.initial_layout:
        init.append(v)
    var fin = Python.list()
    for v in r.final_layout:
        fin.append(v)
    return Python.tuple(_wrap(r.circuit.copy()), init, fin)


def _apply_calibration(mut t: Target, cal: PythonObject) raises:
    """Keys, all optional: error_1q, readout, t1_us, t2_us (one value per qubit),
    edge_errors ({"a-b": e}), angle_bounds ({"rzz": (lo, hi)})."""
    var empty = List[Float64]()
    var e1 = _floats(cal["error_1q"]) if cal.__contains__("error_1q") else empty.copy()
    var ro = _floats(cal["readout"]) if cal.__contains__("readout") else empty.copy()
    var t1 = _floats(cal["t1_us"]) if cal.__contains__("t1_us") else empty.copy()
    var t2 = _floats(cal["t2_us"]) if cal.__contains__("t2_us") else empty.copy()
    t.set_calibration(e1, empty, ro, t1, t2)
    if cal.__contains__("edge_errors"):
        var d = cal["edge_errors"]
        for key in d:
            var parts = String(py=key).split("-")
            t.set_edge_error(Int(parts[0]), Int(parts[1]), Float64(py=d[key]))
    if cal.__contains__("angle_bounds"):
        var d = cal["angle_bounds"]
        for key in d:
            var k = op_from_name(String(py=key))
            if k < 0:
                raise Error("QM-E610: unknown gate '" + String(py=key) + "' in angle_bounds. Hint: use gate names such as rzz.")
            t.set_angle_bounds(k, Float64(py=d[key][0]), Float64(py=d[key][1]))


def py_transpile_calibrated(circ: PythonObject, num_qubits: PythonObject, edges: PythonObject,
                            basis: PythonObject, level: PythonObject, seed: PythonObject,
                            calibration: PythonObject) raises -> PythonObject:
    """`transpile` with device data: (..., seed, calibration) -> (circuit, initial, final, est_fidelity).

    calibration: dict with optional error_1q, readout, t1_us, t2_us (per qubit),
    edge_errors ({"a-b": e}) and angle_bounds ({"rzz": (lo, hi)}). Uniform data
    changes nothing; non-uniform data adds `post_layout`, bounds add `wrap_angles`.
    """
    var t = _target(num_qubits, edges, basis)
    _apply_calibration(t, calibration)
    var r = transpile(PyCircuit._get(circ)[].c, t, Int(py=level), Int(py=seed))
    var lay = _layouts(r)
    return Python.tuple(_wrap(r.circuit.copy()), lay[0], lay[1], t.estimate_fidelity(r.circuit))


def py_mcx_circuit(num_controls: PythonObject, mode: PythonObject, num_ancillas: PythonObject,
                   clean: PythonObject) raises -> PythonObject:
    """Multi-controlled X on qubits 0..k-1 -> k, ancillas after: (k, mode, ancillas, clean) -> circuit."""
    var k = Int(py=num_controls)
    var na = Int(py=num_ancillas)
    var controls = List[Int]()
    for i in range(k):
        controls.append(i)
    var ancillas = List[Int]()
    for i in range(na):
        ancillas.append(k + 1 + i)
    var c = Circuit(k + 1 + na, name="mcx")
    append_mcx(c, controls, k, ancillas, String(py=mode), Bool(py=clean))
    return _wrap(c^)


def py_draw(circ: PythonObject, width: PythonObject, style: PythonObject) raises -> PythonObject:
    """Text drawing: (circuit, width, style) -> str; style 'ascii' or 'unicode', width 0 = no fold."""
    return PythonObject(PyCircuit._get(circ)[].c.draw(Int(py=width), String(py=style)))


def _counts_dict(r: SampleResult) raises -> PythonObject:
    var d = Python.dict()
    for kv in r.counts.items():
        d[kv.key] = kv.value
    return d


def py_sample_ptsbe(circ: PythonObject, shots: PythonObject, seed: PythonObject) raises -> PythonObject:
    """Static noisy circuit, one simulation per distinct Pauli error pattern -> counts."""
    return _counts_dict(sample_ptsbe(PyCircuit._get(circ)[].c, Int(py=shots), UInt64(Int(py=seed))))


def py_inject_noise(text: PythonObject, p1: PythonObject, p2: PythonObject, pm: PythonObject) raises -> PythonObject:
    """OpenQASM 2 text + gate/measure rates -> (text with #pragma qumo channels, inserted 1q, 2q, readout)."""
    var r = inject_noise(String(py=text), Float64(py=p1), Float64(py=p2), Float64(py=pm))
    return Python.tuple(String(r.text), r.depolarize1, r.depolarize2, r.x_error)


def py_ionq_request(text: PythonObject, shots: PythonObject, noise: PythonObject, seed: PythonObject,
                    name: PythonObject) raises -> PythonObject:
    """OpenQASM 2 -> (IonQ ionq.circuit.v1 JSON, clbit->qubit map, num_clbits, dense qubits)."""
    var r = ionq_request(String(py=text), Int(py=shots), String(py=noise), Int(py=seed), String(py=name))
    var qmap = Python.list()
    for v in r.clbit_qubit:
        qmap.append(v)
    return Python.tuple(String(r.body), qmap, r.num_clbits, r.dense_qubits)


def py_ionq_counts(keys: PythonObject, probs: PythonObject, clbit_qubit: PythonObject, shots: PythonObject,
                   seed: PythonObject) raises -> PythonObject:
    """IonQ histogram (lists of outcome ints and probabilities) -> counts dict over the clbits."""
    var p = List[Float64]()
    for v in probs:
        p.append(Float64(py=v))
    return _py_counts(ionq_counts(_int_list(keys), p, _int_list(clbit_qubit), Int(py=shots), UInt64(Int(py=seed))))


def _int_list(obj: PythonObject) raises -> List[Int]:
    var out = List[Int]()
    for v in obj:
        out.append(Int(py=v))
    return out^


def _py_int_list(vals: List[Int]) raises -> PythonObject:
    var out = Python.list()
    for v in vals:
        out.append(v)
    return out


def _py_counts(counts: Dict[String, Int]) raises -> PythonObject:
    var d = Python.dict()
    for kv in counts.items():
        d[kv.key] = kv.value
    return d


def py_ionq_submit(text: PythonObject, shots: PythonObject, noise: PythonObject, seed: PythonObject,
                   name: PythonObject) raises -> PythonObject:
    """Native IonQ submission (POST /jobs; key read from IONQ_API_KEY in Mojo) -> (job id, clbit->qubit map, num_clbits, dense qubits)."""
    var r = ionq_submit(String(py=text), Int(py=shots), String(py=noise), Int(py=seed), String(py=name))
    return Python.tuple(String(r.job_id), _py_int_list(r.request.clbit_qubit), r.request.num_clbits,
                        r.request.dense_qubits)


def py_ionq_status(job_id: PythonObject) raises -> PythonObject:
    """Native GET /jobs/<id> -> QUMO state name."""
    return PythonObject(ionq_status(String(py=job_id)))


def py_ionq_result(job_id: PythonObject, clbit_qubit: PythonObject, shots: PythonObject,
                   seed: PythonObject) raises -> PythonObject:
    """Native GET of the probability histogram, sampled into counts."""
    return _py_counts(ionq_result(String(py=job_id), _int_list(clbit_qubit), Int(py=shots), UInt64(Int(py=seed))))


def py_ionq_backends() raises -> PythonObject:
    """Native GET /backends -> list of (name, status, qubits)."""
    var out = Python.list()
    for r in ionq_backends():
        out.append(Python.tuple(String(r.name), String(r.status), r.qubits))
    return out


def _spq_auth(base: PythonObject, user: PythonObject, key: PythonObject, aud: PythonObject) raises -> SpeQtrumAuth:
    return SpeQtrumAuth(checked_base(String(py=base)), String(py=user), String(py=key), String(py=aud))


def py_spq_payload_document(backend: PythonObject, shots: PythonObject, qasm: PythonObject, nq: PythonObject,
                            nc: PythonObject, initial: PythonObject, final: PythonObject, mclbits: PythonObject,
                            mqubits: PythonObject, auto: PythonObject) raises -> PythonObject:
    """qumo.payload.v1 document (byte-identical to the Python fallback)."""
    return PythonObject(payload_document(String(py=backend), Int(py=shots), String(py=qasm), Int(py=nq), Int(py=nc),
                                         _int_list(initial), _int_list(final), _int_list(mclbits), _int_list(mqubits),
                                         Bool(py=auto)))


def py_spq_unsafe_construct(text: PythonObject) raises -> PythonObject:
    return PythonObject(unsafe_construct(String(py=text)))


def py_spq_decode_result(encoded: PythonObject, job_id: PythonObject) raises -> PythonObject:
    return _py_counts(decode_result(String(py=encoded), String(py=job_id)))


def py_spq_devices(base: PythonObject, user: PythonObject, key: PythonObject, aud: PythonObject) raises -> PythonObject:
    var out = Python.list()
    var auth = _spq_auth(base, user, key, aud)
    for d in speqtrum_devices(auth):
        out.append(Python.tuple(String(d.code), d.nqubits, String(d.status), String(d.notes)))
    return out


def py_spq_execute(base: PythonObject, user: PythonObject, key: PythonObject, aud: PythonObject, device: PythonObject,
                   document: PythonObject, shots: PythonObject) raises -> PythonObject:
    var auth = _spq_auth(base, user, key, aud)
    return PythonObject(speqtrum_execute(auth, String(py=device), String(py=document), Int(py=shots)))


def py_spq_status(base: PythonObject, user: PythonObject, key: PythonObject, aud: PythonObject,
                  job_id: PythonObject) raises -> PythonObject:
    var auth = _spq_auth(base, user, key, aud)
    return PythonObject(speqtrum_status(auth, String(py=job_id)))


def py_spq_result(base: PythonObject, user: PythonObject, key: PythonObject, aud: PythonObject,
                  job_id: PythonObject) raises -> PythonObject:
    var auth = _spq_auth(base, user, key, aud)
    return _py_counts(speqtrum_result(auth, String(py=job_id)))


def _str_list(obj: PythonObject) raises -> List[String]:
    var out = List[String]()
    for v in obj:
        out.append(String(py=v))
    return out^


def _py_float_list(vals: List[Float64]) raises -> PythonObject:
    var out = Python.list()
    for v in vals:
        out.append(v)
    return out


def py_ibm_backends(account: PythonObject, instance: PythonObject) raises -> PythonObject:
    var out = Python.list()
    for r in ibm_backends(String(py=account), String(py=instance)):
        out.append(Python.tuple(String(r.name), r.qubits, String(r.status), r.pending))
    return out


def py_ibm_target(account: PythonObject, instance: PythonObject, backend: PythonObject,
                  fractional: PythonObject) raises -> PythonObject:
    var t = ibm_target(String(py=account), String(py=instance), String(py=backend), Bool(py=fractional))
    var d = Python.dict()
    d["name"] = String(t.name)
    d["num_qubits"] = t.num_qubits
    var basis = Python.list()
    for g in t.basis:
        basis.append(String(g))
    d["basis"] = basis
    d["edges"] = _py_int_list(t.edges)
    d["edge_errors"] = _py_float_list(t.edge_errors)
    d["symmetric"] = t.symmetric
    d["err_1q"] = t.err_1q
    d["err_2q"] = t.err_2q
    d["err_ro"] = t.err_ro
    d["dur_1q"] = t.dur_1q
    d["dur_2q"] = t.dur_2q
    d["dur_ro"] = t.dur_ro
    d["readout"] = _py_float_list(t.readout)
    d["max_shots"] = t.max_shots
    d["simulator"] = t.simulator
    d["processor"] = String(t.processor)
    d["dt"] = t.dt
    d["rep_delay"] = t.rep_delay
    d["fractional"] = t.fractional
    d["native_2q"] = String(t.native_2q)
    d["faulty_qubits"] = _py_int_list(t.faulty_qubits)
    d["faulty_edges"] = t.faulty_edges
    return d


def py_ibm_isa_payload(qasm: PythonObject, num_qubits: PythonObject, basis: PythonObject, edges: PythonObject,
                       symmetric: PythonObject, d1: PythonObject, d2: PythonObject, dr: PythonObject) raises -> PythonObject:
    var p = ibm_isa_payload(String(py=qasm), Int(py=num_qubits), _str_list(basis), _int_list(edges), Bool(py=symmetric),
                            Float64(py=d1), Float64(py=d2), Float64(py=dr))
    return Python.tuple(String(p.qasm3), p.duration_s, p.two_qubit_gates, p.active_qubits)


def py_ibm_submit(account: PythonObject, instance: PythonObject, backend: PythonObject, qasm3: PythonObject,
                  shots: PythonObject, cap: PythonObject) raises -> PythonObject:
    return PythonObject(ibm_submit(String(py=account), String(py=instance), String(py=backend), String(py=qasm3),
                                   Int(py=shots), Int(py=cap)))


def py_ibm_status(account: PythonObject, instance: PythonObject, job_id: PythonObject) raises -> PythonObject:
    var st = ibm_status(String(py=account), String(py=instance), String(py=job_id))
    return Python.tuple(String(st.state), String(st.reason))


def py_ibm_result(account: PythonObject, instance: PythonObject, job_id: PythonObject,
                  num_clbits: PythonObject) raises -> PythonObject:
    return _py_counts(ibm_result(String(py=account), String(py=instance), String(py=job_id), Int(py=num_clbits)))


def py_qtm_native_circuit(qasm: PythonObject) raises -> PythonObject:
    """Compiled OpenQASM 2 -> (pytket circuit JSON, n_qubits, n_clbits, n1q, n2q, n_measure_reset)."""
    var c = qtm_native_circuit(String(py=qasm))
    var regs = Python.list()
    for r in c.registers:
        regs.append(String(r))
    return Python.tuple(String(c.json), c.n_qubits, c.n_clbits, c.n1q, c.n2q, c.n_measure_reset, regs)


def py_qtm_submit(device: PythonObject, project: PythonObject, name: PythonObject, qasm: PythonObject,
                  shots: PythonObject, max_cost: PythonObject) raises -> PythonObject:
    var r = qtm_submit(String(py=device), String(py=project), String(py=name), String(py=qasm), Int(py=shots),
                       Float64(py=max_cost))
    return Python.tuple(String(r.job_id), String(r.project_id), String(r.circuit_id))


def py_qtm_status(job_id: PythonObject) raises -> PythonObject:
    var st = qtm_status(String(py=job_id))
    return Python.tuple(String(st.state), String(st.detail))


def py_qtm_result(job_id: PythonObject, num_clbits: PythonObject) raises -> PythonObject:
    return _py_counts(qtm_result(String(py=job_id), Int(py=num_clbits)))


def py_qtm_devices() raises -> PythonObject:
    var out = Python.list()
    for d in qtm_devices():
        out.append(Python.tuple(String(d.name), d.qubits))
    return out


def py_plan_trajectories(circ: PythonObject, shots: PythonObject, seed: PythonObject) raises -> PythonObject:
    """What sample_ptsbe would run: (distinct patterns, noise sites, error-free shots)."""
    var p = plan_trajectories(PyCircuit._get(circ)[].c, Int(py=shots), UInt64(Int(py=seed)))
    return Python.tuple(p.unique, p.sites, p.error_free)


def py_sample_branching(circ: PythonObject, shots: PythonObject, seed: PythonObject) raises -> PythonObject:
    """Stabilizer branching sampler: near-Clifford circuits on up to 512 qubits -> counts."""
    return _counts_dict(sample_branching(PyCircuit._get(circ)[].c, Int(py=shots), UInt64(Int(py=seed))))


def py_density_probabilities(circ: PythonObject) raises -> PythonObject:
    """Exact outcome probabilities of all qubits from the density matrix."""
    var out = Python.list()
    for p in density_evolve(PyCircuit._get(circ)[].c).probabilities():
        out.append(p)
    return out


def py_density_expectation(circ: PythonObject, observable: PythonObject) raises -> PythonObject:
    """Tr(rho H) for a Pauli-sum string."""
    return density_evolve(PyCircuit._get(circ)[].c).expectation(parse_pauli_sum(String(py=observable)))


def py_mitigate_m3(counts: PythonObject, p01: PythonObject, p10: PythonObject, hamming: PythonObject) raises -> PythonObject:
    """M3 readout mitigation: ({bitstring: count}, p01, p10, hamming) -> {bitstring: quasi-probability}."""
    var c = Dict[String, Int]()
    for key in counts:
        c[String(py=key)] = Int(py=counts[key])
    var e01 = _floats(p01)
    var e10 = _floats(p10)
    var cal = ReadoutCalibration(len(e01))
    cal.p01 = e01^
    cal.p10 = e10^
    var out = Python.dict()
    var m = mitigate_counts_m3(c, cal, Int(py=hamming))
    for kv in m.items():
        out[kv.key] = kv.value
    return out


def py_to_pbc(circ: PythonObject, fix_clifford: PythonObject) raises -> PythonObject:
    """Pauli-based form as JSON text: rotations and measurements in the input frame."""
    var w = JsonWriter()
    to_pbc(PyCircuit._get(circ)[].c, Bool(py=fix_clifford)).write_json(w)
    return PythonObject(w.finish())


def py_from_stim(text: PythonObject) raises -> PythonObject:
    """Stim circuit text -> Circuit."""
    return _wrap(from_stim(String(py=text)).circuit.copy())


def py_to_stim(circ: PythonObject) raises -> PythonObject:
    """Clifford + Pauli-noise Circuit -> Stim circuit text."""
    return PythonObject(to_stim(PyCircuit._get(circ)[].c))


def py_dem_text(circ: PythonObject) raises -> PythonObject:
    """Detector error model of a noisy Clifford circuit, in Stim DEM text."""
    return PythonObject(detector_error_model(PyCircuit._get(circ)[].c).to_text())


def py_parse_dem(text: PythonObject) raises -> PythonObject:
    """Round trip Stim DEM text through QUMO: (num_detectors, num_observables, num_errors, text)."""
    var d = parse_dem(String(py=text))
    return Python.tuple(d.num_detectors, d.num_observables, d.num_errors(), d.to_text())


def py_sample_dem(text: PythonObject, shots: PythonObject, seed: PythonObject) raises -> PythonObject:
    """Detector firing rates of `shots` samples of a DEM: list per detector, then per observable."""
    var d = parse_dem(String(py=text))
    var n = Int(py=shots)
    var s = sample_dem(d, n, UInt64(Int(py=seed)))
    var out = Python.list()
    for k in range(d.num_detectors):
        var c = 0
        for shot in range(n):
            c += 1 if s.get(shot, k) else 0
        out.append(Float64(c) / Float64(n))
    for k in range(d.num_observables):
        var c = 0
        for shot in range(n):
            c += 1 if s.observable(shot, k) else 0
        out.append(Float64(c) / Float64(n))
    return out


def py_decode_dem(text: PythonObject, shots: PythonObject, decoder: PythonObject) raises -> PythonObject:
    """Decode with a DEM given as text: (dem_text, [[fired detector ids], ...], 'mwpm' or
    'mwpm_correlated') -> list of observable masks."""
    var d = parse_dem(String(py=text))
    var name = String(py=decoder)
    var out = Python.list()
    if name == "mwpm_correlated":
        var dec = CorrelatedMatchingDecoder(d)
        for shot in shots:
            var f = List[Int]()
            for x in shot:
                f.append(Int(py=x))
            out.append(Int(dec.decode(f)))
        return out
    var dec = SparseBlossomDecoder(MatchingGraph(d))
    for shot in shots:
        var f = List[Int]()
        for x in shot:
            f.append(Int(py=x))
        out.append(Int(dec.decode(f)))
    return out


def py_missing_detectors(circ: PythonObject, unknown_input: PythonObject) raises -> PythonObject:
    """Deterministic record parities the circuit does not declare: list of record lists."""
    var out = Python.list()
    for recs in missing_detectors(PyCircuit._get(circ)[].c, Bool(py=unknown_input)):
        var row = Python.list()
        for r in recs:
            row.append(r)
        out.append(row)
    return out


def py_count_determined(circ: PythonObject, unknown_input: PythonObject) raises -> PythonObject:
    """Stim's count_determined_measurements."""
    return count_determined_measurements(PyCircuit._get(circ)[].c, Bool(py=unknown_input))


def py_translate(circ: PythonObject, basis: PythonObject) raises -> PythonObject:
    """Basis translation only (no layout/routing/optimisation)."""
    ref c = PyCircuit._get(circ)[].c
    var t = Target.all_to_all(max(c.num_qubits, 1), _basis_spec(basis))
    var r = compile_spec(c, t, "basis", 7)
    return _wrap(r.circuit.copy())


def py_compile_spec(circ: PythonObject, num_qubits: PythonObject, edges: PythonObject,
                    basis: PythonObject, spec: PythonObject) raises -> PythonObject:
    """Run an explicit text pipeline; returns (circuit, report_json)."""
    var t = _target(num_qubits, edges, basis)
    var r = compile_spec(PyCircuit._get(circ)[].c, t, String(py=spec), 7)
    return Python.tuple(_wrap(r.circuit.copy()), r.to_json())


def _layouts(r: CompileResult) raises -> Tuple[PythonObject, PythonObject]:
    var init = Python.list()
    for v in r.initial_layout:
        init.append(v)
    var fin = Python.list()
    for v in r.final_layout:
        fin.append(v)
    return (init, fin)


def py_transpile_ai(circ: PythonObject, num_qubits: PythonObject, edges: PythonObject,
                    basis: PythonObject, level: PythonObject, seed: PythonObject) raises -> PythonObject:
    """Hybrid heuristic + AI compile (`transpile_ai`, AI level 1..3):
    (circuit, num_qubits, edges, basis, level, seed) -> (circuit, initial, final).

    Same conventions as `transpile`. Models are looked up in $QUMO_MODEL_DIR,
    then ./models/ai_transpiler; without models exact / greedy synthesis and
    SABRE routing still run. Every learned proposal is verified exactly.
    """
    var t = _target(num_qubits, edges, basis)
    var r = transpile_ai(PyCircuit._get(circ)[].c, t, Int(py=level), Int(py=seed))
    var lay = _layouts(r)
    return Python.tuple(_wrap(r.circuit.copy()), lay[0], lay[1])


def py_transpile_ai_report(circ: PythonObject, num_qubits: PythonObject, edges: PythonObject,
                           basis: PythonObject, level: PythonObject, seed: PythonObject) raises -> PythonObject:
    """`transpile_ai` with the per-pass report:
    -> (circuit, initial, final, report_json). At AI level 3 the report is the
    portfolio winner's (its pipeline names the preset that won)."""
    var t = _target(num_qubits, edges, basis)
    var r = transpile_ai(PyCircuit._get(circ)[].c, t, Int(py=level), Int(py=seed))
    var lay = _layouts(r)
    return Python.tuple(_wrap(r.circuit.copy()), lay[0], lay[1], r.to_json())


def py_compile_spec_full(circ: PythonObject, num_qubits: PythonObject, edges: PythonObject,
                         basis: PythonObject, spec: PythonObject, seed: PythonObject) raises -> PythonObject:
    """Explicit text pipeline with a seed and layouts:
    -> (circuit, initial, final, report_json)."""
    var t = _target(num_qubits, edges, basis)
    var r = compile_spec(PyCircuit._get(circ)[].c, t, String(py=spec), Int(py=seed))
    var lay = _layouts(r)
    return Python.tuple(_wrap(r.circuit.copy()), lay[0], lay[1], r.to_json())


def py_sample(circ: PythonObject, shots: PythonObject, seed: PythonObject) raises -> PythonObject:
    var r = sample(PyCircuit._get(circ)[].c, Int(py=shots), UInt64(Int(py=seed)))
    var d = Python.dict()
    for kv in r.counts.items():
        d[kv.key] = kv.value
    return d


def py_sample_device(circ: PythonObject, shots: PythonObject, seed: PythonObject, device: PythonObject,
                     var **kwargs: PythonObject) raises -> PythonObject:
    """Sample on an explicit device -> (counts, method, note).

    Python: sample_device(circ, shots, seed, device, precision="auto").

    device: 'auto' | 'cpu' | 'gpu'. 'gpu' falls back to the CPU when the build
    lacks -D QUMO_GPU or no device is usable; `note` then says why.
    precision (keyword only): 'auto' (QUMO_GPU_PRECISION when set, else the
    build default: fp32 on every GPU) | 'fp32' | 'fp64', for GPU state-vector
    runs; `method` names the precision that ran ('gpu:statevector:fp64').
    Errors: QM-E940 unknown device, QM-E941 unknown precision, QM-E472 fp64
    on an Apple GPU.
    """
    var precision = String("auto")
    for kv in kwargs.items():
        if kv.key == "precision":
            precision = String(py=kv.value)
        else:
            raise Error("QM-E913: sample_device() got an unexpected keyword argument '" + kv.key + "'. Hint: the only keyword argument is precision='auto'|'fp32'|'fp64'.")
    var note = String()
    var r = sample_on(PyCircuit._get(circ)[].c, Int(py=shots), UInt64(Int(py=seed)), List[Float64](), String(py=device), note,
                      precision)
    var d = Python.dict()
    for kv in r.counts.items():
        d[kv.key] = kv.value
    return Python.tuple(d, r.method, note)


def py_gpu_info() raises -> PythonObject:
    """GPU status: compiled (-D QUMO_GPU), available, name, api, vendor,
    precision (what 'auto' runs), precision_build_default, precision_env
    (QUMO_GPU_PRECISION), precisions (selectable), precision_note."""
    var g = query_gpu()
    var d = Python.dict()
    d["compiled"] = g.compiled
    d["available"] = g.available
    d["name"] = g.name
    d["api"] = g.api
    d["vendor"] = g.vendor
    d["precision"] = g.precision
    d["precision_build_default"] = g.precision_build_default
    d["precision_env"] = g.precision_env
    var ps = Python.list()
    for p in g.precisions:
        ps.append(p)
    d["precisions"] = ps
    d["precision_note"] = g.precision_note
    d["fp64_supported"] = g.fp64_supported
    d["min_qubits_auto"] = g.min_qubits_auto
    d["error"] = g.error
    return d


def py_expectation(circ: PythonObject, observable: PythonObject) raises -> PythonObject:
    var sv = evolve(PyCircuit._get(circ)[].c)
    return sv.expectation(parse_pauli_sum(String(py=observable)))


# ---------------------------------------------------------------------------
# Device-placed values and gradients, SQD subspace solver, HEFT schedules,
# streaming QEC (used by examples/heterogeneous/{sqd,cutting,realtime_qec,
# qaoa_escalation}). All keyword arguments are optional and keyword-only.
# ---------------------------------------------------------------------------
def _floats(values: PythonObject) raises -> List[Float64]:
    var out = List[Float64]()
    for v in values:
        out.append(Float64(py=v))
    return out^


def _float_list(values: List[Float64]) raises -> PythonObject:
    var out = Python.list()
    for v in values:
        out.append(v)
    return out


def py_expectation_device(circ: PythonObject, observable: PythonObject, device: PythonObject,
                          var **kwargs: PythonObject) raises -> PythonObject:
    """<observable> of a bound circuit on a device -> (value, method, note).

    Python: expectation_device(circ, observable, device, precision="auto").
    device 'auto' | 'cpu' | 'gpu' (GPU state vector, fp32 by default; falls
    back to the CPU with a note when the build or the machine has no GPU).
    """
    var prec = String("auto")
    for kv in kwargs.items():
        if kv.key == "precision":
            prec = String(py=kv.value)
        else:
            raise Error("QM-E913: expectation_device() got an unexpected keyword argument '" + kv.key + "'. Hint: the only keyword argument is precision='auto'|'fp32'|'fp64'.")
    var obs = List[PauliSum]()
    obs.append(parse_pauli_sum(String(py=observable)))
    var note = String()
    var method = String()
    var vals = expectations_on(PyCircuit._get(circ)[].c, obs, List[Float64](), String(py=device), note, method, prec)
    return Python.tuple(vals[0], method, note)


def py_expectations_device(circ: PythonObject, observables: PythonObject, device: PythonObject,
                           var **kwargs: PythonObject) raises -> PythonObject:
    """Many observables of one bound circuit -> (values, method, note): one
    simulation, all Pauli terms reduced in one batch on the GPU.

    Python: expectations_device(circ, [obs, ...], device, precision="auto").
    """
    var prec = String("auto")
    for kv in kwargs.items():
        if kv.key == "precision":
            prec = String(py=kv.value)
        else:
            raise Error("QM-E913: expectations_device() got an unexpected keyword argument '" + kv.key + "'. Hint: the only keyword argument is precision='auto'|'fp32'|'fp64'.")
    var obs = List[PauliSum]()
    for o in observables:
        obs.append(parse_pauli_sum(String(py=o)))
    var note = String()
    var method = String()
    var vals = expectations_on(PyCircuit._get(circ)[].c, obs, List[Float64](), String(py=device), note, method, prec)
    return Python.tuple(_float_list(vals), method, note)


def py_value_and_gradient(circ: PythonObject, observable: PythonObject, params: PythonObject,
                          var **kwargs: PythonObject) raises -> PythonObject:
    """Exact E(params) and its adjoint gradient -> (value, grad, method, note).

    Python: value_and_gradient(circ, observable, params, device="cpu", precision="auto").
    device 'cpu' (fp64 adjoint), 'gpu' (GPU adjoint through a doubled register,
    fp32 by default; CPU fallback with a note), 'auto' (GPU from the dispatch
    threshold). Shared and affine parameters (QAOA angles) are exact: the
    chain rule sums every occurrence. Errors: QM-E804, QM-E805, QM-E940, QM-E941.
    """
    var dev = String("cpu")
    var prec = String("auto")
    for kv in kwargs.items():
        if kv.key == "device":
            dev = String(py=kv.value)
        elif kv.key == "precision":
            prec = String(py=kv.value)
        else:
            raise Error("QM-E913: value_and_gradient() got an unexpected keyword argument '" + kv.key + "'. Hint: keyword arguments: device, precision.")
    var note = String()
    var r = value_and_gradient_on(PyCircuit._get(circ)[].c, parse_pauli_sum(String(py=observable)), _floats(params),
                                  dev, note, prec)
    return Python.tuple(r.value, _float_list(r.grad), r.method, note)


def py_batch_value_and_gradient(circ: PythonObject, observable: PythonObject, param_sets: PythonObject) raises -> PythonObject:
    """CPU adjoint values and gradients for many parameter sets -> (values, grads)
    (parallel over sets for <= 14 qubits; bit-identical to one-by-one calls)."""
    var sets = List[List[Float64]]()
    for ps in param_sets:
        sets.append(_floats(ps))
    var r = batch_value_and_gradient(PyCircuit._get(circ)[].c, parse_pauli_sum(String(py=observable)), sets,
                                     Grad.ADJOINT)
    var gs = Python.list()
    for g in r[1]:
        gs.append(_float_list(g))
    return Python.tuple(_float_list(r[0]), gs)


def py_subspace_ground_state(observable: PythonObject, basis_states: PythonObject, num_qubits: PythonObject,
                             var **kwargs: PythonObject) raises -> PythonObject:
    """Lowest eigenpair of a Pauli-sum Hamiltonian projected onto basis states
    (the SQD / SKQD classical stage) -> dict with energy, occupations (<n_q>),
    dimension, nonzeros, matvecs, restarts, residual, converged, seconds and
    probabilities (|c_k|^2 per basis state, same order).

    Python: subspace_ground_state(observable, states, n, tol=1e-9, krylov=40).
    """
    var tol = 1e-9
    var krylov = 40
    for kv in kwargs.items():
        if kv.key == "tol":
            tol = Float64(py=kv.value)
        elif kv.key == "krylov":
            krylov = Int(py=kv.value)
        else:
            raise Error("QM-E913: subspace_ground_state() got an unexpected keyword argument '" + kv.key + "'. Hint: keyword arguments: tol, krylov.")
    var states = List[Int]()
    for b in basis_states:
        states.append(Int(py=b))
    var t0 = perf_counter_ns()
    var r = subspace_ground_state(parse_pauli_sum(String(py=observable)), states, Int(py=num_qubits), tol, krylov)
    var secs = Float64(perf_counter_ns() - t0) / 1e9
    var d = Python.dict()
    d["energy"] = r.energy
    d["occupations"] = _float_list(r.occupations)
    var probs = List[Float64](capacity=len(r.re))
    for k in range(len(r.re)):
        probs.append(r.re[k] * r.re[k] + r.im[k] * r.im[k])
    d["probabilities"] = _float_list(probs)
    d["dimension"] = r.dimension
    d["nonzeros"] = r.nonzeros
    d["matvecs"] = r.matvecs
    d["restarts"] = r.restarts
    d["residual"] = r.residual
    d["converged"] = r.converged
    d["seconds"] = secs
    return d


def _dget(d: PythonObject, key: String, default: PythonObject) raises -> PythonObject:
    return d.get(key, default)


def py_schedule(jobs: PythonObject, devices: PythonObject, var **kwargs: PythonObject) raises -> PythonObject:
    """HEFT plan of quantum jobs over a device pool (qumo.exec.orchestrator),
    without running them -> dict(device_of, start, finish, cost, makespan,
    total_cost, notes).

    Python: schedule(jobs, devices, cost_weight=0.0)
    jobs: [{"name", "circuit" (Circuit), "kind": "sample"|"estimate", "shots",
            "observable" (estimate), "deps": [earlier job indices], "device": pin}]
    devices: [{"name", "kind": "cpu"|"gpu"|"qpu"|"stabilizer", "max_qubits",
               "queue_s", "shot_time_s", "cost_per_task", "cost_per_shot",
               "precision"}]
    Errors: QM-E460..QM-E464 (empty pool, nothing fits, bad deps, unknown or
    too small pinned device).
    """
    var cw = 0.0
    for kv in kwargs.items():
        if kv.key == "cost_weight":
            cw = Float64(py=kv.value)
        else:
            raise Error("QM-E913: schedule() got an unexpected keyword argument '" + kv.key + "'. Hint: keyword argument: cost_weight.")
    var pool = List[Device]()
    for d in devices:
        var name = String(py=d["name"])
        var kind = String(py=_dget(d, "kind", "cpu"))
        var mq = Int(py=_dget(d, "max_qubits", 30))
        if kind == "cpu":
            pool.append(Device.cpu(name, mq))
        elif kind == "gpu":
            pool.append(Device.gpu(name, mq, String(py=_dget(d, "precision", "auto"))))
        elif kind == "stabilizer":
            pool.append(Device.stabilizer(name))
        elif kind == "qpu":
            pool.append(Device.qpu(name, mq, Float64(py=_dget(d, "queue_s", 0.0)), Float64(py=_dget(d, "shot_time_s", 1e-4)),
                                   Float64(py=_dget(d, "cost_per_task", 0.0)), Float64(py=_dget(d, "cost_per_shot", 0.0))))
        else:
            raise Error("QM-E913: device '" + name + "' has unknown kind '" + kind + "'. Hint: use cpu, gpu, qpu or stabilizer.")
    var wf = Workflow()
    for j in jobs:
        var name = String(py=j["name"])
        ref c = PyCircuit._get(j["circuit"])[].c
        var kind = String(py=_dget(j, "kind", "sample"))
        var deps = List[Int]()
        for x in _dget(j, "deps", Python.list()):
            deps.append(Int(py=x))
        var pin = String(py=_dget(j, "device", ""))
        if kind == "estimate":
            _ = wf.add_estimate(name, c, parse_pauli_sum(String(py=_dget(j, "observable", "Z0"))), List[Float64](), deps,
                                1234, "auto", pin)
        else:
            _ = wf.add_sample(name, c, Int(py=_dget(j, "shots", 1000)), List[Float64](), deps, 1234, "auto", pin)
    var sch = wf.plan(pool, cw)
    var out = Python.dict()
    var dev = Python.list()
    for k in range(len(sch.device_of)):
        dev.append(pool[sch.device_of[k]].name)
    out["device_of"] = dev
    out["start"] = _float_list(sch.start)
    out["finish"] = _float_list(sch.finish)
    out["cost"] = _float_list(sch.cost)
    out["makespan"] = sch.makespan
    out["total_cost"] = sch.total_cost
    var notes = Python.list()
    for line in sch.notes:
        notes.append(line)
    out["notes"] = notes
    return out


comptime _HEXD = "0123456789abcdef"


def _hex_nibble(c: Int) raises -> Int:
    if c >= 48 and c <= 57:
        return c - 48
    if c >= 97 and c <= 102:
        return c - 87
    if c >= 65 and c <= 70:
        return c - 55
    raise Error("QM-E913: invalid hex digit in a packed detection-event string. Hint: pass the string returned by QecMemory.sample().")


struct PyQecMemory(Movable, Writable):
    """A QEC memory experiment for streaming: syndrome sampling (CPU or GPU
    Pauli frames) and batch decoding (sparse blossom) on packed strings.

    Packed format (hex, like Stim's b8): one record of ceil(D/8) bytes per
    shot, bit k of byte j = detector 8j + k; observables likewise."""

    var cc: CodeCircuit
    var dem: DetectorErrorModel
    var decoder: SparseBlossomDecoder
    var frames: FrameSimulator
    var num_detectors: Int
    var num_observables: Int

    def __init__(out self, var cc: CodeCircuit) raises:
        self.dem = detector_error_model(cc.circuit, True)
        self.decoder = SparseBlossomDecoder(MatchingGraph(self.dem))
        self.frames = FrameSimulator(cc.circuit)
        self.num_detectors = self.dem.num_detectors
        self.num_observables = self.dem.num_observables
        self.cc = cc^

    @staticmethod
    def _get(py_self: PythonObject) raises -> Pointer[PyQecMemory, MutAnyOrigin]:
        return py_self.downcast_value_ptr[Self]().as_unsafe_any_origin()

    @staticmethod
    def info(py_self: PythonObject) raises -> PythonObject:
        ref m = Self._get(py_self)[]
        var d = Python.dict()
        d["code"] = m.cc.code
        d["basis"] = m.cc.basis
        d["distance"] = m.cc.distance
        d["rounds"] = m.cc.rounds
        d["num_qubits"] = m.cc.circuit.num_qubits
        d["num_detectors"] = m.num_detectors
        d["num_observables"] = m.num_observables
        d["dem_errors"] = m.dem.num_errors()
        d["p"] = m.cc.noise.after_clifford_depolarization
        return d

    @staticmethod
    def dem_text(py_self: PythonObject) raises -> PythonObject:
        # graph-like form: the same decomposition QUMO's matching decoders use,
        # so PyMatching built from this text matches them shot for shot
        return graphlike_dem(Self._get(py_self)[].dem).to_text()

    @staticmethod
    def stim_text(py_self: PythonObject) raises -> PythonObject:
        return Self._get(py_self)[].cc.to_stim()

    @staticmethod
    def sample(py_self: PythonObject, shots: PythonObject, seed: PythonObject, device: PythonObject,
               threads: PythonObject) raises -> PythonObject:
        """-> (dets_hex, obs_hex, method, note, seconds). device 'cpu' | 'gpu'
        ('gpu' = GPU Pauli frames; CPU fallback with a note); threads: CPU
        frame workers (0 = all cores)."""
        ref m = Self._get(py_self)[]
        var n = Int(py=shots)
        var dev = String(py=device)
        if dev != "cpu" and dev != "gpu" and dev != "auto":
            raise Error("QM-E940: unknown device '" + dev + "'. Hint: use 'auto', 'cpu' or 'gpu'.")
        var note = String()
        var method = String("cpu:pauli_frame")
        var t0 = perf_counter_ns()
        var samples: DetectorSamples
        var on_gpu = False
        if dev != "cpu":
            comptime if is_defined["QUMO_GPU"]():
                var ok: Bool
                try:
                    ok = gpu_available()
                except:
                    ok = False
                on_gpu = ok
                if not ok and dev == "gpu":
                    note = "no usable GPU device: ran on the CPU"
            else:
                if dev == "gpu":
                    note = "GPU support not compiled (build with -D QUMO_GPU): ran on the CPU"
        if on_gpu:
            var gs = GpuFrameSampler(m.cc.circuit)
            samples = gs.sample_detectors(n, UInt64(Int(py=seed)))
            method = gs.method_name()
        else:
            samples = m.frames.sample_detectors(n, UInt64(Int(py=seed)), Int(py=threads))
        var secs = Float64(perf_counter_ns() - t0) / 1e9
        var D = m.num_detectors
        var K = m.num_observables
        var bd = (D + 7) // 8
        var bk = (K + 7) // 8
        var digits = _HEXD.as_bytes()
        var hd = List[UInt8](capacity=2 * bd * n)
        var hk = List[UInt8](capacity=2 * bk * n)
        for s in range(n):
            for j in range(bd):
                var byte = 0
                for k in range(8):
                    var det = 8 * j + k
                    if det < D and samples.get(s, det):
                        byte |= 1 << k
                hd.append(digits[byte >> 4])
                hd.append(digits[byte & 15])
            for j in range(bk):
                var byte = 0
                for k in range(8):
                    var o = 8 * j + k
                    if o < K and samples.observable(s, o):
                        byte |= 1 << k
                hk.append(digits[byte >> 4])
                hk.append(digits[byte & 15])
        return Python.tuple(String(unsafe_from_utf8=Span(hd)), String(unsafe_from_utf8=Span(hk)), method, note, secs)

    @staticmethod
    def decode(py_self: PythonObject, dets_hex: PythonObject, shots: PythonObject, threads: PythonObject) raises -> PythonObject:
        """Sparse-blossom predictions for packed detection events ->
        (predictions_hex, seconds, detection_events). threads=1: this thread
        only (one decoder worker); 0: every core (decode_samples)."""
        ref m = Self._get(py_self)[]
        var text = String(py=dets_hex)
        var bs = text.as_bytes()
        var n = Int(py=shots)
        var D = m.num_detectors
        var K = m.num_observables
        var bd = (D + 7) // 8
        var bk = (K + 7) // 8
        if text.byte_length() != 2 * bd * n:
            raise Error("QM-E913: packed detection events have " + String(text.byte_length()) + " hex digits, expected " + String(2 * bd * n) + " for " + String(n) + " shots of " + String(D) + " detectors. Hint: decode with the QecMemory that sampled them.")
        var t0 = perf_counter_ns()
        var preds = List[UInt64](length=max(1, n), fill=0)
        var events = 0
        if Int(py=threads) == 1:
            var fired = List[Int]()
            for s in range(n):
                fired.clear()
                var base = 2 * bd * s
                for j in range(bd):
                    var byte = _hex_nibble(Int(bs[base + 2 * j])) * 16 + _hex_nibble(Int(bs[base + 2 * j + 1]))
                    if byte == 0:
                        continue
                    for k in range(8):
                        if (byte >> k) & 1 == 1:
                            fired.append(8 * j + k)
                events += len(fired)
                preds[s] = m.decoder.decode(fired)
        else:
            var smp = DetectorSamples(n, D, K)
            for s in range(n):
                var base = 2 * bd * s
                var b = s // BATCH
                var w = (s % BATCH) >> 6
                var bit = UInt64(1) << UInt64(s & 63)
                for j in range(bd):
                    var byte = _hex_nibble(Int(bs[base + 2 * j])) * 16 + _hex_nibble(Int(bs[base + 2 * j + 1]))
                    if byte == 0:
                        continue
                    for k in range(8):
                        if (byte >> k) & 1 == 1:
                            smp.dets[(b * D + 8 * j + k) * FW + w] |= bit
                            events += 1
            preds = decode_samples(m.decoder, smp)
        var secs = Float64(perf_counter_ns() - t0) / 1e9
        var digits = _HEXD.as_bytes()
        var hk = List[UInt8](capacity=2 * bk * n)
        for s in range(n):
            for j in range(bk):
                var byte = Int((preds[s] >> UInt64(8 * j)) & 255)
                hk.append(digits[byte >> 4])
                hk.append(digits[byte & 15])
        return Python.tuple(String(unsafe_from_utf8=Span(hk)), secs, events)

    def write_to(self, mut w: Some[Writer]):
        w.write("<qumo.QecMemory ", self.cc.code, " d=", self.cc.distance, " rounds=", self.cc.rounds, " detectors=",
                self.num_detectors, ">")

    def write_repr_to(self, mut w: Some[Writer]):
        self.write_to(w)


def py_qec_memory(code: PythonObject, distance: PythonObject, rounds: PythonObject, p: PythonObject) raises -> PythonObject:
    """QecMemory for a generated memory circuit ('surface_z', 'surface_x',
    'repetition') under uniform circuit noise p (Stim/Sinter convention)."""
    var cc = generate_code(String(py=code), Int(py=distance), Int(py=rounds), NoiseParams.uniform(Float64(py=p)))
    return PythonObject(alloc=PyQecMemory(cc^))


# Environment variables the Mojo runtime sets at the C level when its Python
# interop starts: it discovers *a* Python on the machine (e.g. a Homebrew
# 3.14) and exports PYTHONEXECUTABLE, PYTHONPATH (":"-prefixed) and
# MOJO_PYTHON_LIBRARY. Inside a host interpreter these are wrong, and every
# child process (subprocess, multiprocessing spawn) inherits them: a
# subprocess started with sys.executable then runs the other Python.


def _restore_host_environment():
    """Put those variables back to the host interpreter's values.

    `os.environ` is a snapshot taken when the interpreter started, so it holds
    the user's original values: restore each one, or unset it if the user had
    not set it. Errors are ignored: the module must still import."""
    try:
        var environ = Python.import_module("os").environ
        var keys: List[String] = ["PYTHONEXECUTABLE", "PYTHONPATH", "PYTHONHOME", "MOJO_PYTHON_LIBRARY"]
        for key in keys:
            if Bool(environ.__contains__(key)):
                _ = setenv(key, String(py=environ[key]), True)
            else:
                _ = unsetenv(key)
    except:
        pass


@export
def PyInit_qumo_py() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("qumo_py")
        _ = (
            m.add_type[PyCircuit]("Circuit")
            .def_method[PyCircuit.num_qubits]("num_qubits")
            .def_method[PyCircuit.num_clbits]("num_clbits")
            .def_method[PyCircuit.size]("size")
            .def_method[PyCircuit.num_parameters]("num_parameters")
            .def_method[PyCircuit.depth]("depth")
            .def_method[PyCircuit.count_ops]("count_ops")
            .def_method[PyCircuit.count_gate]("count_gate")
            .def_method[PyCircuit.depth_of]("depth_of")
            .def_method[PyCircuit.two_qubit_edges]("two_qubit_edges")
            .def_method[PyCircuit.copy]("copy")
            .def_method[PyCircuit.compose]("compose")
            .def_method[PyCircuit.bind]("bind")
            .def_method[PyCircuit.text]("text")
            .def_method[PyCircuit.to_qasm2]("to_qasm2")
            .def_method[PyCircuit.to_qasm3]("to_qasm3")
        )
        _ = (
            m.add_type[PyProgram]("Program")
            .def_method[PyProgram.num_qubits]("num_qubits")
            .def_method[PyProgram.num_clbits]("num_clbits")
            .def_method[PyProgram.num_parameters]("num_parameters")
            .def_method[PyProgram.has_runtime_regions]("has_runtime_regions")
            .def_method[PyProgram.to_circuit]("to_circuit")
            .def_method[PyProgram.to_qasm3]("to_qasm3")
            .def_method[PyProgram.to_json]("to_json")
            .def_method[PyProgram.lint]("lint")
            .def_method[PyProgram.text]("text")
        )
        m.def_function[py_load_program]("load_program")
        m.def_function[py_program_from_json]("program_from_json")
        m.def_function[py_draw_program]("draw_program")
        m.def_function[py_sample_program]("sample_program")
        m.def_function[py_transpile_program]("transpile_program")
        m.def_function[py_empty]("circuit", docstring="Empty circuit with n qubits")
        m.def_function[py_quantum_volume]("quantum_volume")
        m.def_function[py_efficient_su2]("efficient_su2")
        m.def_function[py_dtc_unitary]("dtc_unitary")
        m.def_function[py_random_clifford]("random_clifford_circuit")
        m.def_function[py_bv_all_ones]("bv_all_ones")
        m.def_function[py_bvlike]("trivial_bvlike_circuit")
        m.def_function[py_multi_control]("multi_control_circuit")
        m.def_function[py_twirl]("pauli_twirl_2q")
        m.def_function[py_hamiltonian_circuit]("hamiltonian_circuit")
        m.def_function[py_version]("version")
        m.def_function[py_transpile_calibrated]("transpile_calibrated")
        m.def_function[py_mcx_circuit]("mcx_circuit")
        m.def_function[py_draw]("draw")
        m.def_function[py_sample_ptsbe]("sample_ptsbe")
        m.def_function[py_plan_trajectories]("plan_trajectories")
        m.def_function[py_inject_noise]("inject_noise")
        m.def_function[py_ionq_request]("ionq_request")
        m.def_function[py_ionq_counts]("ionq_counts")
        m.def_function[py_ionq_submit]("ionq_submit")
        m.def_function[py_ionq_status]("ionq_status")
        m.def_function[py_ionq_result]("ionq_result")
        m.def_function[py_ionq_backends]("ionq_backends")
        m.def_function[py_spq_payload_document]("speqtrum_payload_document")
        m.def_function[py_spq_unsafe_construct]("speqtrum_unsafe_construct")
        m.def_function[py_spq_decode_result]("speqtrum_decode_result")
        m.def_function[py_spq_devices]("speqtrum_devices")
        m.def_function[py_spq_execute]("speqtrum_execute")
        m.def_function[py_spq_status]("speqtrum_status")
        m.def_function[py_spq_result]("speqtrum_result")
        m.def_function[py_ibm_backends]("ibm_backends")
        m.def_function[py_ibm_target]("ibm_target")
        m.def_function[py_ibm_isa_payload]("ibm_isa_payload")
        m.def_function[py_ibm_submit]("ibm_submit")
        m.def_function[py_ibm_status]("ibm_status")
        m.def_function[py_ibm_result]("ibm_result")
        m.def_function[py_qtm_native_circuit]("quantinuum_native_circuit")
        m.def_function[py_qtm_submit]("quantinuum_submit")
        m.def_function[py_qtm_status]("quantinuum_status")
        m.def_function[py_qtm_result]("quantinuum_result")
        m.def_function[py_qtm_devices]("quantinuum_devices")
        m.def_function[py_sample_branching]("sample_branching")
        m.def_function[py_density_probabilities]("density_probabilities")
        m.def_function[py_density_expectation]("density_expectation")
        m.def_function[py_mitigate_m3]("mitigate_m3")
        m.def_function[py_to_pbc]("to_pbc")
        m.def_function[py_from_stim]("from_stim")
        m.def_function[py_to_stim]("to_stim")
        m.def_function[py_dem_text]("dem_text")
        m.def_function[py_parse_dem]("parse_dem")
        m.def_function[py_sample_dem]("sample_dem")
        m.def_function[py_decode_dem]("decode_dem")
        m.def_function[py_missing_detectors]("missing_detectors")
        m.def_function[py_count_determined]("count_determined_measurements")
        m.def_function[py_load_qasm]("load_qasm")
        m.def_function[py_load_qasm_file]("load_qasm_file")
        m.def_function[py_transpile]("transpile")
        m.def_function[py_translate]("translate")
        m.def_function[py_compile_spec]("compile_spec")
        m.def_function[py_transpile_ai]("transpile_ai")
        m.def_function[py_compile_spec_full]("compile_spec_full")
        m.def_function[py_transpile_ai_report]("transpile_ai_report")
        m.def_function[py_sample]("sample")
        m.def_function[py_expectation]("expectation")
        m.def_function[py_sample_device]("sample_device")
        m.def_function[py_gpu_info]("gpu_info")
        _ = (
            m.add_type[PyQecMemory]("QecMemory")
            .def_method[PyQecMemory.info]("info")
            .def_method[PyQecMemory.dem_text]("dem_text")
            .def_method[PyQecMemory.stim_text]("stim_text")
            .def_method[PyQecMemory.sample]("sample")
            .def_method[PyQecMemory.decode]("decode")
        )
        m.def_function[py_expectation_device]("expectation_device")
        m.def_function[py_expectations_device]("expectations_device")
        m.def_function[py_value_and_gradient]("value_and_gradient")
        m.def_function[py_batch_value_and_gradient]("batch_value_and_gradient")
        m.def_function[py_subspace_ground_state]("subspace_ground_state")
        m.def_function[py_schedule]("schedule")
        m.def_function[py_qec_memory]("qec_memory")
        var mod = m.finalize()
        _restore_host_environment()
        return mod
    except e:
        abort(String("error creating qumo_py module: ", e))
