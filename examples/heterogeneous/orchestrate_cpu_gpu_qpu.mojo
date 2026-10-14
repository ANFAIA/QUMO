"""Mojo twin of ai_gpu_qpu.py: rule-based selection, HEFT over CPU + GPU, final job on a QPU pool.

Reproduces: heterogeneous CPU+GPU+QPU workflow pattern - CUDA-Q multi-QPU (mqpu) asynchronous execution
Source: https://nvidia.github.io/cuda-quantum/latest/using/backends/sims/mqpusims.html
Source: https://github.com/NVIDIA/cuda-quantum (Apache-2.0); QUMO design notes in qumo/exec/orchestrator.mojo

What it shows (all offline, deterministic):
1. Offline rules select QAOA (p=1) for the deterministic 16-node 3-regular graph of
   problems/maxcut_16_3reg.json.
2. Workflow 1: 64 ESTIMATE jobs (an 8x8 gamma/beta grid) over the pool [CPU, GPU];
   HEFT places each job and `Schedule` explains why. On -D QUMO_GPU builds with a usable
   device the GPU jobs run on the GPU state vector (method gpu:statevector:fp32); otherwise
   they fall back to the CPU and say so.
3. CPU refinement: Nelder-Mead on exact energies (ExpectationEngine) from the best grid point.
4. Workflow 2: one SAMPLE job on a QPU-only pool, Device.qpu("superconducting-sim", 133, ...).
   QPU jobs run on the local emulator and are flagged `emulated`. (Phase 1 workaround: a
   separate QPU-only workflow, because jobs cannot be pinned to a device yet; TODO phase 2:
   add_sample(..., device="superconducting-sim") in a single workflow.)
5. The bound OpenQASM 3 payload is written to build/hetero/payload.qasm, ready for
   `python -m qumo_hardware run build/hetero/payload.qasm -p local -b <backend>` (dry run by
   default for remote backends).

Checks: p=1 approximation ratio >= 0.6924 (Farhi et al. bound for 3-regular graphs), every
grid job ran on a capable device, the QPU job is flagged emulated, the payload parses back to
the same state (1e-12), GPU method string on GPU builds, best sampled cut = brute-force max.

Run:  .venv/bin/mojo run -I . examples/heterogeneous/orchestrate_cpu_gpu_qpu.mojo
      .venv/bin/mojo run -I . -D QUMO_GPU examples/heterogeneous/orchestrate_cpu_gpu_qpu.mojo
example-meta: {"framework": "qumo", "tier": "fast", "timeout": 600, "requires": [], "args": []}
"""

from std.math import pi
from std.os import makedirs
from std.os.path import exists
from std.sys.defines import is_defined
from std.time import perf_counter_ns

from qumo.testing.checks import CheckLog
from qumo.algorithms.hamiltonians import Graph, maxcut
from qumo.algorithms.library import qaoa_maxcut_ansatz
from qumo.algorithms.optimizers import NelderMead
from qumo.algorithms.variational import VQEObjective
from qumo.exec.orchestrator import Workflow, Device, JobResult
from qumo.ir.qasm import parse_qasm, to_qasm
from qumo.sim.runner import evolve

comptime GRAPH_16 = "0-6 0-9 0-11 1-2 1-3 1-7 2-6 2-13 3-4 3-12 4-9 4-15 5-7 5-11 5-15 6-13 7-14 8-10 8-12 8-13 9-11 10-12 10-14 14-15"
comptime FARHI_P1_BOUND = 0.6924


def cut_of_key(g: Graph, key: String) raises -> Float64:
    """Cut value of a Qiskit-style count key (leftmost character = highest qubit)."""
    var n = key.byte_length()
    var bs = key.as_bytes()
    var a = 0
    for i in range(n):
        if Int(bs[i]) == ord("1"):
            a |= 1 << (n - 1 - i)
    return g.cut_value(a)


def main() raises:
    var log = CheckLog("heterogeneous/orchestrate_cpu_gpu_qpu")
    var t_start = perf_counter_ns()

    # ---- 1. rule-based selection --------------------------------------------------------
    var g = Graph.parse(GRAPH_16)
    var n = g.num_nodes
    var p = 1  # the Mojo twin keeps p=1 (Farhi bound); the Python driver uses p=2 for n <= 16
    print("[plan] offline rules: problem kind maxcut ->", "qaoa_maxcut p=" + String(p), "on", n, "nodes,",
          len(g), "edges")
    var mc = g.max_cut()
    var max_cut = mc[0]
    var ansatz = qaoa_maxcut_ansatz(g, p)
    var h = maxcut(g, True)  # <H> = -expected cut

    # ---- 2. workflow 1: landscape scan over [CPU, GPU] ----------------------------------
    var wf = Workflow()
    var grid = 8
    for i in range(grid):
        for j in range(grid):
            var gm = pi * (Float64(i) + 0.5) / Float64(grid)
            var bt = (pi / 2.0) * (Float64(j) + 0.5) / Float64(grid)
            var prm: List[Float64] = [gm, bt]
            _ = wf.add_estimate("E(" + String(i) + "," + String(j) + ")", ansatz, h, prm)
    var pool = List[Device]()
    pool.append(Device.cpu("cpu0"))
    pool.append(Device.gpu("gpu0"))
    var sched = wf.plan(pool)
    print("[gpu/cpu] HEFT schedule for", len(wf.jobs), "estimate jobs:")
    var shown = 0
    for line in sched.notes:
        if shown < 6:
            print("   ", line)
        shown += 1
    if shown > 6:
        print("    ... (" + String(shown - 6) + " more lines)")
    var t0 = perf_counter_ns()
    var rs = wf.run(pool)
    var t_scan = Float64(perf_counter_ns() - t0) / 1e9
    var best_k = 0
    var on_gpu = 0
    var capable = True
    var gpu_method = String()
    for k in range(len(rs)):
        if not rs[k].ok:
            capable = False
        if rs[k].device == "gpu0":
            on_gpu += 1
            if rs[k].backend.startswith("gpu:"):
                gpu_method = rs[k].backend
        if rs[k].expectation < rs[best_k].expectation:
            best_k = k
    var bi = best_k // grid
    var bj = best_k % grid
    var g0 = pi * (Float64(bi) + 0.5) / Float64(grid)
    var b0 = (pi / 2.0) * (Float64(bj) + 0.5) / Float64(grid)
    print("[gpu/cpu] scan:", len(rs), "jobs in", t_scan, "s;", on_gpu, "placed on gpu0; backend of job 0:",
          rs[0].backend)
    print("[gpu/cpu] best grid point gamma =", g0, "beta =", b0, "expected cut =", -rs[best_k].expectation)

    # ---- 3. CPU refinement (exact energies, Nelder-Mead) ----------------------------------
    var obj = VQEObjective(ansatz, h)
    var x0: List[Float64] = [g0, b0]
    var t1 = perf_counter_ns()
    var opt = NelderMead(maxiter=300, initial_step=0.1).minimize(obj, x0)
    var t_cpu = Float64(perf_counter_ns() - t1) / 1e9
    var ideal_cut = -opt.fun
    var ratio = ideal_cut / max_cut
    print("[cpu] Nelder-Mead:", opt.nfev, "evaluations in", t_cpu, "s; gamma =", opt.x[0], "beta =", opt.x[1],
          "expected cut =", ideal_cut, "/", max_cut, "ratio =", ratio)

    # ---- 4. workflow 2: final sampling on a QPU-only pool ----------------------------------
    var meas = ansatz.copy()
    meas.measure_all()
    var wf2 = Workflow()
    _ = wf2.add_sample("final sampling", meas, 2000, opt.x, List[Int](), UInt64(7))
    var qpool = List[Device]()
    qpool.append(Device.qpu("superconducting-sim", 133, 60.0, 1e-4, 0.0, 0.0))
    var sched2 = wf2.plan(qpool)
    for line in sched2.notes:
        print("[qpu]", line)
    var rs2 = wf2.run(qpool)
    var qr = rs2[0].copy()
    var best_cut = 0.0
    var mean_cut = 0.0
    var shots = 0
    for kv in qr.samples.counts.items():
        var c = cut_of_key(g, kv.key)
        mean_cut += c * Float64(kv.value)
        shots += kv.value
        if c > best_cut:
            best_cut = c
    mean_cut /= Float64(max(shots, 1))
    print("[qpu] device", qr.device, "emulated =", qr.emulated, "backend =", qr.backend, "shots =", shots,
          "mean cut =", mean_cut, "best cut =", best_cut)

    # ---- 5. payload for qumo_hardware ------------------------------------------------------
    var bound = meas.bind(opt.x)
    var payload = to_qasm(bound)
    if not exists("build/hetero"):
        makedirs("build/hetero", exist_ok=True)
    with open("build/hetero/payload.qasm", "w") as f:
        f.write(payload)
    print("[qpu] wrote build/hetero/payload.qasm (" + String(payload.byte_length()) + " bytes): "
          + "python -m qumo_hardware run build/hetero/payload.qasm -p local -b 'qumo-sim:grid:4x4/sc_cz+noise:superconducting'")
    var back = parse_qasm(payload)
    var s_ref = evolve(bound)
    var s_back = evolve(back)
    var worst = 0.0
    for i in range(s_ref.size()):
        worst = max(worst, abs(s_ref.re[i] - s_back.re[i]) + abs(s_ref.im[i] - s_back.im[i]))

    # ---- checks ------------------------------------------------------------------------------
    _ = log.within("p=1 approximation ratio (Farhi bound 0.6924)", ratio, FARHI_P1_BOUND, 1.0)
    _ = log.check("every scan job ran on a capable device", capable and len(rs) == grid * grid,
                  String(len(rs)) + " jobs, " + String(on_gpu) + " on gpu0")
    _ = log.within("refinement improved the best grid point", -opt.fun, -rs[best_k].expectation - 1e-12,
                   max_cut)
    _ = log.check("qpu job is flagged emulated", qr.emulated and qr.ok, "device=" + qr.device)
    _ = log.check("qpu best sampled cut equals brute-force max cut", best_cut == max_cut,
                  "best=" + String(best_cut) + " max=" + String(max_cut))
    _ = log.close("payload round trip (state vector)", worst, 0.0, 1e-12)
    comptime if is_defined["QUMO_GPU"]():
        if on_gpu > 0 and gpu_method.byte_length() > 0:
            _ = log.check("gpu jobs ran on the GPU state vector", gpu_method.startswith("gpu:statevector:"),
                          "method=" + gpu_method)
        else:
            _ = log.check("gpu build: jobs on gpu0 fell back with a reason", on_gpu == 0 or
                          rs[0].backend.find("fallback") >= 0 or rs[0].device != "gpu0", "backend=" + rs[0].backend)
    else:
        _ = log.check("cpu build: gpu0 jobs fall back to the CPU and say why", on_gpu == 0 or
                      _first_gpu_backend(rs).find("gpu fallback") >= 0, "backend=" + _first_gpu_backend(rs))
    print("total", Float64(perf_counter_ns() - t_start) / 1e9, "s")
    log.finish()


def _first_gpu_backend(rs: List[JobResult]) -> String:
    for r in rs:
        if r.device == "gpu0":
            return r.backend
    return "(no job on gpu0)"
