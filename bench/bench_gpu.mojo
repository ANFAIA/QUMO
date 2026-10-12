"""GPU vs CPU state-vector throughput (same layered circuit as bench_sv.mojo).

    make bench-gpu                      # = build with -D QUMO_GPU, run up to 26 qubits
    ./build/gpu/bench_gpu 28 5          # nmax, repetitions (median reported)

Per width n: the CPU time of `evolve` (fused plan, SIMD + threads + cache
blocking) and the GPU time of `reset + evolve + synchronize` on a
pre-allocated `GpuStateVector` (device allocation and the one-off
DeviceContext / kernel-module load are excluded; a warm-up run precedes the
timed runs), plus GPU expectation (sum of n ZZ terms) and 10k-shot sampling.
Correctness is checked once per width against the CPU (fidelity).

Without -D QUMO_GPU the binary only reports that GPU support is missing (and
runs a tiny emulator smoke check so the harness itself stays tested).
"""

from std.time import perf_counter_ns
from std.sys import argv
from std.sys.defines import is_defined

from qumo.ir.circuit import Circuit
from qumo.ir.pauli import PauliSum
from qumo.sim.runner import evolve
from qumo.gpu.statevector_gpu import GpuStateVector
from qumo.gpu.device import gpu_info, gpu_available, GPU_DEFAULT_DTYPE, precision_name


def layered(n: Int, depth: Int) -> Circuit:
    var c = Circuit(n)
    for d in range(depth):
        for q in range(n):
            c.ry(0.1 + 0.01 * Float64(q + d), q)
            c.rz(0.2 + 0.03 * Float64(q * d), q)
        for q in range(d % 2, n - 1, 2):
            c.cx(q, q + 1)
    return c^


def zz_chain(n: Int) -> PauliSum:
    var h = PauliSum()
    for q in range(n - 1):
        h.add(1.0, 0, (UInt64(3) << UInt64(q)))
    return h^


def median(var xs: List[Float64]) -> Float64:
    sort(xs)
    return xs[len(xs) // 2]


def secs(t0: Int) -> Float64:
    return Float64(perf_counter_ns() - t0) / 1e9


def bench_width[dt: DType](n: Int, reps: Int, emulate: Bool) raises:
    var c = layered(n, 10)
    var h = zz_chain(n)
    var cpu_t = List[Float64]()
    var cpu_e = 0.0
    for _ in range(reps):
        var t0 = perf_counter_ns()
        var sv = evolve(c)
        cpu_t.append(secs(t0))
        cpu_e = sv.expectation(h)
    var g = GpuStateVector[dt](n, emulate)
    g.evolve(c)  # warm-up (module load, allocator)
    g.synchronize()
    var gpu_t = List[Float64]()
    for _ in range(reps):
        var t0 = perf_counter_ns()
        g.reset()
        g.evolve(c)
        g.synchronize()
        gpu_t.append(secs(t0))
    var t1 = perf_counter_ns()
    var gpu_e = g.expectation(h)
    var exp_s = secs(t1)
    var t2 = perf_counter_ns()
    _ = g.sample(10000, 7)
    var smp_s = secs(t2)
    var fid = evolve(c).fidelity(g.to_host())
    var tc = median(cpu_t^)
    var tg = median(gpu_t^)
    print(n, ",", len(c), ",", tc, ",", tg, ",", tc / tg, ",", Float64(len(c)) / tg, ",",
          exp_s, ",", smp_s, ",", abs(cpu_e - gpu_e), ",", fid, ",", g.method_name())


def main() raises:
    var args = argv()
    var nmin = 16
    var nmax = 26
    var reps = 3
    if len(args) > 1:
        nmax = Int(String(args[1]))
    if len(args) > 2:
        reps = Int(String(args[2]))
    print("#", gpu_info())
    comptime if is_defined["QUMO_GPU"]():
        if not gpu_available():
            print("# no usable GPU device")
            return
        print("qubits,gates,cpu_s,gpu_s,speedup,gpu_gates_per_s,gpu_expect_s,gpu_sample10k_s,abs_err_expect,fidelity,method")
        for n in range(nmin, nmax + 1, 2):
            bench_width[GPU_DEFAULT_DTYPE](n, reps, False)
    else:
        print("# GPU support not compiled: run `make bench-gpu` (builds with -D QUMO_GPU).")
        print("# emulator smoke check (timings are NOT GPU timings):")
        print("qubits,gates,cpu_s,emu_s,ratio,emu_gates_per_s,emu_expect_s,emu_sample10k_s,abs_err_expect,fidelity,method")
        bench_width[DType.float32](12, 1, True)
