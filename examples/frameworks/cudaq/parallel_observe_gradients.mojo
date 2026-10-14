"""Batched observe over 10,000 parameter sets, HEFT placement, parallel gradients.

Reproduces: CUDA-Q - Multi-GPU workflows (mqpu observe_async batching) and VQE parallel gradients
Source: https://nvidia.github.io/cuda-quantum/0.14.0/using/examples/multi_gpu_workflows.html (CUDA-Q docs, Apache-2.0)
Source: https://nvidia.github.io/cuda-quantum/0.14.0/applications/python/vqe_advanced.html (CUDA-Q docs, Apache-2.0)

Part A (multi_gpu_workflows, "circuit batching"): a 5-qubit kernel
rx(params[i], q[i]) observed against H = Z0 for 10,000 parameter sets drawn
uniformly from [0, 1) with seed 13; one `cudaq.observe` call took 3.185 s on
one GPU and four `observe_async` batches of 2500 on four virtual QPUs 1.175 s
(page). QUMO: ONE `EstimatorPub` (Qiskit PUB / Braket ProgramSet style) holds
all 10,000 sets; `run_estimator` evaluates them in parallel threads. It is
compared with a sequential loop over a reused `ExpectationEngine` and with the
analytic value <Z0> = cos(params[0]).

Part B (mqpu device pool): a `Workflow` (qumo/exec/orchestrator.mojo, HEFT
list scheduling with a cost model) receives small 5-qubit and wide 18-qubit
estimate jobs over the pool [Device.cpu(), Device.gpu()]; the schedule is
printed with the reason for every placement. Without -D QUMO_GPU, jobs placed
on the GPU run on the CPU and their backend says "(gpu fallback: ...)".

Part C (vqe_advanced, parallel parameter-shift): CUDA-Q sends the shifted
circuits of every parameter to `observe_async` round-robin over QPUs. QUMO
runs (i) all 2P shifted parameter sets of a 12-qubit hardware-efficient
ansatz as one parallel EstimatorPub, (ii) `batch_gradient` with the
parameter-shift rule and (iii) the adjoint method (one forward and one
backward sweep). Note: the page's formula (E+ - E-) / (2 eps) with
eps = pi/4 equals the exact derivative times sin(eps)/eps (about 0.900); the
exact two-term rule divides by 2 sin(eps), as QUMO does (eps = pi/2).

Checks:
- parallel PUB results equal the sequential loop and cos(params[0]) (1e-12);
- every Workflow job is placed on a device with enough qubits, and its value
  equals a direct evaluation (1e-12 on CPU backends, 1e-4 on fp32 GPU ones);
- PUB parameter-shift, batch_gradient parameter-shift and adjoint gradients
  agree (1e-10). Speedups are printed, not asserted.

Run: .venv/bin/mojo run -I . examples/frameworks/cudaq/parallel_observe_gradients.mojo
example-meta: {"framework": "cudaq", "tier": "slow", "timeout": 300, "requires": [], "args": []}
"""

from std.math import cos, pi
from std.time import perf_counter_ns
from qumo.ir.circuit import Circuit
from qumo.ir.pauli import PauliSum, parse_pauli_sum
from qumo.algorithms.library import hardware_efficient_ansatz
from qumo.algorithms.hamiltonians import transverse_field_ising, heisenberg_xxz
from qumo.algorithms.gradients import ExpectationEngine, Grad, batch_gradient
from qumo.exec.primitives import EstimatorPub, run_estimator
from qumo.exec.orchestrator import Workflow, Device
from qumo.sim.rng import Rng
from qumo.testing.checks import CheckLog

comptime NSETS = 10000


def seconds_since(t0: Int) -> Float64:
    return Float64(perf_counter_ns() - t0) / 1e9


def rx_kernel() -> Circuit:
    var c = Circuit(5, name="rx_kernel")
    var p = c.parameters("params", 5)
    for i in range(5):
        c.rx(p[i], i)
    return c^


def main() raises:
    var log = CheckLog("cudaq/parallel_observe_gradients")

    # ---------------- Part A: one PUB with 10,000 parameter sets ----------------
    var kern = rx_kernel()
    var h = parse_pauli_sum("Z0")
    var rng = Rng(13)
    var sets = List[List[Float64]](capacity=NSETS)
    for _ in range(NSETS):
        var s = List[Float64]()
        for _ in range(5):
            s.append(rng.uniform())
        sets.append(s^)
    var obs = List[PauliSum]()
    obs.append(h.copy())
    var t0 = perf_counter_ns()
    var res = run_estimator(EstimatorPub(kern, obs, sets))
    var t_par = seconds_since(t0)
    var eng = ExpectationEngine(kern, h)
    t0 = perf_counter_ns()
    var seq = List[Float64](capacity=NSETS)
    for s in sets:
        seq.append(eng.value(s))
    var t_seq = seconds_since(t0)
    var d_seq = 0.0
    var d_ana = 0.0
    for b in range(NSETS):
        d_seq = max(d_seq, abs(res.evs[b][0] - seq[b]))
        d_ana = max(d_ana, abs(res.evs[b][0] - cos(sets[b][0])))
    print("Part A: 10,000 parameter sets, 5 qubits, H = Z0")
    print("  one EstimatorPub (parallel):", t_par, "s | sequential loop:", t_seq, "s | speedup", t_seq / t_par,
          "(page: 3.185 s -> 1.175 s on 4 GPUs)")
    _ = log.close("PUB equals sequential loop", d_seq, 0.0, 1e-12)
    _ = log.close("PUB equals cos(params[0])", d_ana, 0.0, 1e-12)

    # ---------------- Part B: HEFT placement over a CPU + GPU pool ----------------
    var wf = Workflow()
    var small = rx_kernel()
    var wide = hardware_efficient_ansatz(18, 2, "linear", "ry,rz")
    var h_wide = transverse_field_ising(18, 1.0, 1.0)
    var wide_params = List[List[Float64]]()
    for k in range(4):
        _ = wf.add_estimate("rx5_batch" + String(k), small, h, sets[k * 2500])
        var r = rng.split(100 + k)
        var p = List[Float64]()
        for _ in range(wide.num_params()):
            p.append(2.0 * pi * r.uniform())
        _ = wf.add_estimate("hea18_" + String(k), wide, h_wide, p)
        wide_params.append(p^)
    var devices = List[Device]()
    devices.append(Device.cpu())
    devices.append(Device.gpu())
    var sch = wf.plan(devices)
    print("Part B:", sch)
    var results = wf.run(devices)
    var placed_ok = True
    var worst_cpu = 0.0
    var worst_gpu = 0.0
    var n_gpu = 0
    var eng_wide = ExpectationEngine(wide, h_wide)
    for j in range(len(wf.jobs)):
        ref dev = devices[sch.device_of[j]]
        placed_ok = placed_ok and dev.max_qubits >= wf.jobs[j].circuit.num_qubits and results[j].ok
        var ref_value: Float64
        if j % 2 == 0:
            ref_value = cos(sets[(j // 2) * 2500][0])
        else:
            ref_value = eng_wide.value(wide_params[j // 2])
        var d = abs(results[j].expectation - ref_value)
        if results[j].backend.startswith("gpu"):
            n_gpu += 1
            worst_gpu = max(worst_gpu, d / max(1.0, abs(ref_value)))
        else:
            worst_cpu = max(worst_cpu, d)
        print("  job", j, wf.jobs[j].name, "->", results[j].device, "|", results[j].backend, "|",
              results[j].seconds, "s | <H> =", results[j].expectation)
    _ = log.check("every job placed on a capable device and ran", placed_ok, String(len(wf.jobs)) + " jobs")
    _ = log.close("CPU-backend job values", worst_cpu, 0.0, 1e-12)
    if n_gpu > 0:
        _ = log.close("GPU-backend job values (fp32, relative)", worst_gpu, 0.0, 1e-4)
    else:
        _ = log.check("GPU-backend job values", True, "no job ran on a GPU here; fallbacks are named in the backend column")

    # ---------------- Part C: parallel parameter-shift vs adjoint ----------------
    var hea = hardware_efficient_ansatz(12, 3, "linear", "ry,rz")
    var h12 = heisenberg_xxz(12, 1.0, 1.0)
    var np_ = hea.num_params()
    var points = List[List[Float64]]()
    for k in range(4):
        var r = rng.split(200 + k)
        var p = List[Float64]()
        for _ in range(np_):
            p.append(2.0 * pi * r.uniform())
        points.append(p^)
    print("Part C: 12-qubit HEA,", np_, "parameters,", len(hea), "gates, H = Heisenberg(12),", len(points), "points")
    t0 = perf_counter_ns()
    var g_adj = batch_gradient(hea, h12, points, Grad.ADJOINT)
    var t_adj = seconds_since(t0)
    t0 = perf_counter_ns()
    var g_ps = batch_gradient(hea, h12, points, Grad.PARAM_SHIFT)
    var t_ps = seconds_since(t0)
    # all 2P shifted circuits of every point in ONE parallel PUB (CUDA-Q: observe_async round-robin)
    var shifted = List[List[Float64]]()
    for p in points:
        for i in range(np_):
            var plus = p.copy()
            plus[i] += pi / 2
            var minus = p.copy()
            minus[i] -= pi / 2
            shifted.append(plus^)
            shifted.append(minus^)
    var obs12 = List[PauliSum]()
    obs12.append(h12.copy())
    t0 = perf_counter_ns()
    var pub = run_estimator(EstimatorPub(hea, obs12, shifted))
    var t_pub = seconds_since(t0)
    var d_ps = 0.0
    var d_pub = 0.0
    for k in range(len(points)):
        for i in range(np_):
            d_ps = max(d_ps, abs(g_ps[k][i] - g_adj[k][i]))
            var gi = 0.5 * (pub.evs[2 * (k * np_ + i)][0] - pub.evs[2 * (k * np_ + i) + 1][0])
            d_pub = max(d_pub, abs(gi - g_adj[k][i]))
    print("  adjoint:", t_adj, "s | batch_gradient parameter-shift:", t_ps, "s | one PUB of", len(shifted),
          "shifted circuits:", t_pub, "s")
    print("  adjoint speedup over parameter-shift:", t_ps / t_adj, "x; PUB speedup over sequential shifts:", t_ps / t_pub, "x")
    _ = log.close("batch_gradient parameter-shift equals adjoint", d_ps, 0.0, 1e-10)
    _ = log.close("parallel PUB parameter-shift equals adjoint", d_pub, 0.0, 1e-10)
    log.finish()
