"""Resource estimation *before* execution (plan, then spend).

Best-of synthesis:
  * Q# / QDK Resource Estimator (QRE): fault-tolerant footprint from logical
    counts + physical error model (surface code distance, T factories).
  * PennyLane `qp.specs`, Qiskit `count_ops`/depth: logical metrics.
  * Braket `Tracker` / emulator: cost & duration on a concrete device profile.
  * Simulation planning: memory + time on the local simulators, with a
    recommended method and the reason (explainability for agents).

The FT model is deliberately simple and documented (Beverland et al. 2022
"Assessing requirements to scale to practical quantum advantage";
Litinski 2019 "Magic state distillation: not as costly as you think"):
  p_L(d) = 0.03 * (p / p_th)^((d+1)/2), p_th = 0.01, 2 d^2 physical qubits per
  logical patch, logical layout Q = 2n + ceil(sqrt(8n)) + 1, one logical cycle
  = d code cycles, 15-to-1 distillation p_out = 35 p_in^3.
"""

from std.math import sqrt, ceil, log, exp
from qumo.ir.circuit import Circuit
from qumo.ir.gates import Op, op_arity, op_is_unitary, op_name


struct ResourceEstimate(Copyable, Writable):
    var num_qubits: Int
    var num_clbits: Int
    var num_params: Int
    var gates_total: Int
    var gates_1q: Int
    var gates_2q: Int
    var gates_3q: Int
    var measurements: Int
    var resets: Int
    var depth: Int
    var depth_2q: Int
    var t_count: Int
    var t_depth: Int
    var is_clifford: Bool
    var is_dynamic: Bool
    var has_noise: Bool
    var sv_memory_bytes: Float64
    var recommended_method: String
    var reason: String

    def __init__(out self):
        self.num_qubits = 0
        self.num_clbits = 0
        self.num_params = 0
        self.gates_total = 0
        self.gates_1q = 0
        self.gates_2q = 0
        self.gates_3q = 0
        self.measurements = 0
        self.resets = 0
        self.depth = 0
        self.depth_2q = 0
        self.t_count = 0
        self.t_depth = 0
        self.is_clifford = True
        self.is_dynamic = False
        self.has_noise = False
        self.sv_memory_bytes = 0
        self.recommended_method = ""
        self.reason = ""

    def to_json(self) -> String:
        var s = String("{")
        s += "\"num_qubits\":" + String(self.num_qubits)
        s += ",\"num_clbits\":" + String(self.num_clbits)
        s += ",\"num_params\":" + String(self.num_params)
        s += ",\"gates_total\":" + String(self.gates_total)
        s += ",\"gates_1q\":" + String(self.gates_1q)
        s += ",\"gates_2q\":" + String(self.gates_2q)
        s += ",\"gates_3q\":" + String(self.gates_3q)
        s += ",\"measurements\":" + String(self.measurements)
        s += ",\"resets\":" + String(self.resets)
        s += ",\"depth\":" + String(self.depth)
        s += ",\"depth_2q\":" + String(self.depth_2q)
        s += ",\"t_count\":" + String(self.t_count)
        s += ",\"t_depth\":" + String(self.t_depth)
        s += ",\"is_clifford\":" + ("true" if self.is_clifford else "false")
        s += ",\"is_dynamic\":" + ("true" if self.is_dynamic else "false")
        s += ",\"has_noise\":" + ("true" if self.has_noise else "false")
        s += ",\"statevector_memory_bytes\":" + String(self.sv_memory_bytes)
        s += ",\"recommended_method\":\"" + self.recommended_method + "\""
        s += ",\"reason\":\"" + self.reason + "\"}"
        return s

    def write_to(self, mut w: Some[Writer]):
        w.write("ResourceEstimate(qubits=", self.num_qubits, ", gates=", self.gates_total,
                " [1q=", self.gates_1q, " 2q=", self.gates_2q, " 3q=", self.gates_3q, "]",
                ", depth=", self.depth, ", 2q-depth=", self.depth_2q, ", T=", self.t_count,
                ", T-depth=", self.t_depth, ", clifford=", self.is_clifford,
                ", dynamic=", self.is_dynamic, ", sv_mem=", format_bytes(self.sv_memory_bytes),
                ", method=", self.recommended_method, " (", self.reason, "))")


def format_bytes(b: Float64) -> String:
    if b < 1024: return String(Int(b)) + " B"
    if b < 1024 * 1024: return String(Int(b / 1024)) + " KiB"
    if b < 1024 * 1024 * 1024: return String(Int(b / (1024 * 1024))) + " MiB"
    if b < 1024.0 * 1024 * 1024 * 1024: return String(Int(b / (1024 * 1024 * 1024))) + " GiB"
    return String(Int(b / (1024.0 * 1024 * 1024 * 1024))) + " TiB"


def _is_t_like(c: Circuit, i: Int) -> Bool:
    var g = c.ops[i]
    if g.kind == Op.T or g.kind == Op.TDG:
        return True
    if (g.kind == Op.RZ or g.kind == Op.P or g.kind == Op.RX or g.kind == Op.RY) and not g.is_clifford():
        return True
    return False


def estimate_resources(c: Circuit, available_memory_bytes: Float64 = 16e9) -> ResourceEstimate:
    var r = ResourceEstimate()
    r.num_qubits = c.num_qubits
    r.num_clbits = c.num_clbits
    r.num_params = c.num_params()
    var lvl2 = List[Int](length=c.num_qubits, fill=0)
    var lvlt = List[Int](length=c.num_qubits, fill=0)
    for i in range(len(c.ops)):
        var g = c.ops[i]
        var k = g.kind
        if k == Op.MEASURE:
            r.measurements += 1
            continue
        if k == Op.RESET:
            r.resets += 1
            continue
        if not op_is_unitary(k) or k == Op.I:
            continue
        r.gates_total += 1
        var a = op_arity(k)
        if a == 1: r.gates_1q += 1
        elif a == 2: r.gates_2q += 1
        else: r.gates_3q += 1
        if not g.is_clifford():
            r.is_clifford = False
        if a >= 2:
            var l = 0
            for j in range(a):
                l = max(l, lvl2[g.qubit(j)])
            for j in range(a):
                lvl2[g.qubit(j)] = l + 1
            r.depth_2q = max(r.depth_2q, l + 1)
        var t_like = _is_t_like(c, i)
        if t_like:
            r.t_count += 1
        var lt = 0
        for j in range(a):
            lt = max(lt, lvlt[g.qubit(j)])
        if t_like:
            lt += 1
            r.t_depth = max(r.t_depth, lt)
        for j in range(a):
            lvlt[g.qubit(j)] = lt
        if k == Op.CCX or k == Op.CCZ or k == Op.CSWAP:
            r.t_count += 7  # standard Toffoli decomposition
    r.depth = c.depth()
    r.is_dynamic = c.is_dynamic()
    r.has_noise = c.has_noise()
    r.sv_memory_bytes = 16.0 * Float64(1 << min(c.num_qubits, 62))
    if r.is_clifford:
        r.recommended_method = "stabilizer"
        r.reason = "circuit is Clifford: tableau/Pauli-frame simulation is polynomial"
    elif r.sv_memory_bytes <= available_memory_bytes:
        r.recommended_method = "statevector:trajectories" if (r.is_dynamic or r.has_noise) else "statevector"
        r.reason = "fits in memory (" + format_bytes(r.sv_memory_bytes) + ")"
    else:
        r.recommended_method = "hardware_or_tensor_network"
        r.reason = "state vector needs " + format_bytes(r.sv_memory_bytes) + " > available " + format_bytes(available_memory_bytes)
    return r^


# ---------------------------------------------------------------------------
# NISQ hardware estimate
# ---------------------------------------------------------------------------
@fieldwise_init
struct HardwareProfile(ImplicitlyCopyable, Writable):
    var name: String
    var t_1q_ns: Float64
    var t_2q_ns: Float64
    var t_meas_ns: Float64
    var err_1q: Float64
    var err_2q: Float64
    var err_meas: Float64
    var t1_us: Float64
    var t2_us: Float64

    @staticmethod
    def superconducting() -> HardwareProfile:
        return HardwareProfile("superconducting", 30, 70, 1000, 2.5e-4, 3e-3, 1e-2, 250, 150)

    @staticmethod
    def trapped_ion() -> HardwareProfile:
        return HardwareProfile("trapped_ion", 5000, 250000, 150000, 3e-5, 1e-3, 3e-3, 1e8, 1e6)

    @staticmethod
    def neutral_atom() -> HardwareProfile:
        return HardwareProfile("neutral_atom", 1000, 400, 500000, 5e-4, 5e-3, 1e-2, 4e6, 1e6)

    def write_to(self, mut w: Some[Writer]):
        w.write("HardwareProfile(", self.name, ")")


struct HardwareEstimate(Copyable, Writable):
    var duration_us: Float64
    var success_probability: Float64
    var gate_error_budget: Float64
    var decoherence_error: Float64
    var profile: String

    def __init__(out self):
        self.duration_us = 0
        self.success_probability = 1
        self.gate_error_budget = 0
        self.decoherence_error = 0
        self.profile = ""

    def to_json(self) -> String:
        return ("{\"profile\":\"" + self.profile + "\",\"duration_us\":" + String(self.duration_us)
                + ",\"success_probability\":" + String(self.success_probability)
                + ",\"gate_error\":" + String(self.gate_error_budget)
                + ",\"decoherence_error\":" + String(self.decoherence_error) + "}")

    def write_to(self, mut w: Some[Writer]):
        w.write("HardwareEstimate(", self.profile, ": duration=", self.duration_us,
                " us, success~", self.success_probability, ")")


def estimate_hardware(c: Circuit, hw: HardwareProfile) -> HardwareEstimate:
    """Critical-path duration and an ESP-style (estimated success probability) figure."""
    var r = HardwareEstimate()
    r.profile = hw.name
    var clock = List[Float64](length=c.num_qubits, fill=0.0)
    var log_ok = 0.0
    for g in c.ops:
        var k = g.kind
        var t: Float64
        var e = 0.0
        if k == Op.MEASURE:
            t = hw.t_meas_ns
            e = hw.err_meas
        elif op_is_unitary(k) and k != Op.I:
            var a = op_arity(k)
            if a == 1:
                t = hw.t_1q_ns
                e = hw.err_1q
                if k == Op.RZ or k == Op.P or k == Op.Z or k == Op.S or k == Op.SDG or k == Op.T or k == Op.TDG:
                    t = 0.0  # virtual-Z
                    e = 0.0
            elif a == 2:
                t = hw.t_2q_ns
                e = hw.err_2q
            else:
                t = 6 * hw.t_2q_ns
                e = 1.0 - (1.0 - hw.err_2q) ** 6
        elif k == Op.RESET:
            t = hw.t_meas_ns
        else:
            continue
        log_ok += log(1.0 - e) if e < 1.0 else -1e9
        var a2 = op_arity(k)
        var start = 0.0
        for j in range(a2):
            start = max(start, clock[g.qubit(j)])
        for j in range(a2):
            clock[g.qubit(j)] = start + t
    var total = 0.0
    for q in range(c.num_qubits):
        total = max(total, clock[q])
    r.duration_us = total / 1000.0
    r.gate_error_budget = 1.0 - _exp(log_ok)
    # idle decoherence: each qubit idles for the whole program
    var t_us = r.duration_us
    var p_idle = 1.0 - _exp(-t_us / hw.t1_us) * _exp(-t_us / hw.t2_us)
    r.decoherence_error = 1.0 - (1.0 - p_idle) ** Float64(c.num_qubits)
    r.success_probability = _exp(log_ok) * (1.0 - r.decoherence_error)
    return r^


@always_inline
def _exp(x: Float64) -> Float64:
    return exp(x)


# ---------------------------------------------------------------------------
# Fault-tolerant estimate (surface code)
# ---------------------------------------------------------------------------
struct FTEstimate(Copyable, Writable):
    var logical_qubits: Int
    var code_distance: Int
    var logical_cycles: Int
    var physical_qubits_algorithm: Int
    var t_factories: Int
    var factory_distance: Int
    var distillation_rounds: Int
    var physical_qubits_factories: Int
    var physical_qubits_total: Int
    var runtime_seconds: Float64
    var logical_error_per_cycle: Float64

    def __init__(out self):
        self.logical_qubits = 0
        self.code_distance = 0
        self.logical_cycles = 0
        self.physical_qubits_algorithm = 0
        self.t_factories = 0
        self.factory_distance = 0
        self.distillation_rounds = 0
        self.physical_qubits_factories = 0
        self.physical_qubits_total = 0
        self.runtime_seconds = 0
        self.logical_error_per_cycle = 0

    def to_json(self) -> String:
        return ("{\"logical_qubits\":" + String(self.logical_qubits)
                + ",\"code_distance\":" + String(self.code_distance)
                + ",\"logical_cycles\":" + String(self.logical_cycles)
                + ",\"physical_qubits_algorithm\":" + String(self.physical_qubits_algorithm)
                + ",\"t_factories\":" + String(self.t_factories)
                + ",\"factory_distance\":" + String(self.factory_distance)
                + ",\"distillation_rounds\":" + String(self.distillation_rounds)
                + ",\"physical_qubits_factories\":" + String(self.physical_qubits_factories)
                + ",\"physical_qubits_total\":" + String(self.physical_qubits_total)
                + ",\"runtime_seconds\":" + String(self.runtime_seconds)
                + ",\"logical_error_per_cycle\":" + String(self.logical_error_per_cycle) + "}")

    def write_to(self, mut w: Some[Writer]):
        w.write("FTEstimate(logical_qubits=", self.logical_qubits, ", d=", self.code_distance,
                ", cycles=", self.logical_cycles, ", factories=", self.t_factories,
                ", physical_qubits=", self.physical_qubits_total,
                ", runtime=", self.runtime_seconds, " s)")


def _p_logical(p: Float64, d: Int) -> Float64:
    return 0.03 * (p / 0.01) ** (Float64(d + 1) / 2.0)


def estimate_fault_tolerant(c: Circuit, physical_error: Float64 = 1e-3,
                            error_budget: Float64 = 1e-2,
                            cycle_time_ns: Float64 = 1000.0) raises -> FTEstimate:
    if physical_error >= 0.01:
        raise Error("QM-E450: physical error " + String(physical_error) + " is above the surface-code threshold (~1e-2). Hint: FT estimation needs p < 1e-2.")
    var res = estimate_resources(c)
    var r = FTEstimate()
    var n = max(c.num_qubits, 1)
    r.logical_qubits = 2 * n + Int(ceil(sqrt(8.0 * Float64(n)))) + 1
    # one T (or non-Clifford rotation ~ synthesised T sequence) per logical cycle
    var rot = res.t_count
    r.logical_cycles = max(res.depth, rot)
    var budget_logical = error_budget / 3.0
    var budget_t = error_budget / 3.0
    var target = budget_logical / Float64(max(r.logical_qubits * r.logical_cycles, 1))
    var d = 3
    while _p_logical(physical_error, d) > target and d < 101:
        d += 2
    r.code_distance = d
    r.logical_error_per_cycle = _p_logical(physical_error, d)
    r.physical_qubits_algorithm = r.logical_qubits * 2 * d * d
    if rot > 0:
        var p_t_target = budget_t / Float64(rot)
        var p_in = physical_error
        var rounds = 0
        while p_in > p_t_target and rounds < 4:
            p_in = 35.0 * p_in * p_in * p_in
            rounds += 1
        r.distillation_rounds = rounds
        var df = 3
        while _p_logical(physical_error, df) * 15.0 * Float64(rounds) > p_t_target and df < d:
            df += 2
        r.factory_distance = df
        # a 15-to-1 unit uses ~16 patches and takes ~6 df cycles; need 1 T per logical cycle (d code cycles)
        var per_factory_qubits = 16 * 2 * df * df * max(rounds, 1)
        r.t_factories = max(1, Int(ceil(6.0 * Float64(df) * Float64(max(rounds, 1)) / Float64(d))))
        r.physical_qubits_factories = r.t_factories * per_factory_qubits
    r.physical_qubits_total = r.physical_qubits_algorithm + r.physical_qubits_factories
    r.runtime_seconds = Float64(r.logical_cycles) * Float64(d) * cycle_time_ns * 1e-9
    return r^
