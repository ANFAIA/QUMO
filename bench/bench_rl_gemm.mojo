"""Single-core f32 GEMM throughput of the native RL stack (performance gate G1, MOJO_PORT.md section 15).

Usage:  bench_rl_gemm [m=512] [n=512] [k=256] [reps=20]

Measures gemm_nn, gemm_nt (including its per-call weight transpose) and gemm_tn on one core, prints one line per
variant (GFLOP/s, median over `reps` timed calls after a warm-up) and one JSON line with the three numbers. Gate G1:
every variant >= 60 GFLOP/s on the reference machine (4x16 register tile, no packing); below 40 triggers the
fallbacks of 17-R1.

Build:  .venv/bin/mojo build -I . -o build/bench_rl_gemm bench/bench_rl_gemm.mojo && build/bench_rl_gemm
"""

from std.time import perf_counter_ns
from std.sys import argv
from qumo.rl.tensor import FPtr, gemm_nn, gemm_nt, gemm_tn
from qumo.sim.rng import Rng


def _median(mut v: List[Float64]) -> Float64:
    sort(v)
    return v[len(v) // 2]


def _fill(mut v: List[Float32], seed: UInt64):
    var r = Rng(seed)
    for i in range(len(v)):
        v[i] = Float32(r.uniform()) - 0.5


def main() raises:
    var args = argv()
    var m = 512
    var n = 512
    var k = 256
    var reps = 20
    if len(args) > 1:
        m = Int(args[1])
    if len(args) > 2:
        n = Int(args[2])
    if len(args) > 3:
        k = Int(args[3])
    if len(args) > 4:
        reps = Int(args[4])
    var flops = 2.0 * Float64(m) * Float64(n) * Float64(k)

    # Operand storage for the three variants:
    #   nn: A[m,k] B[k,n]   nt: A[m,k] B[n,k]   tn: A[k,m] B[k,n]
    var a_mk = List[Float32](length=m * k, fill=0.0)
    var a_km = List[Float32](length=k * m, fill=0.0)
    var b_kn = List[Float32](length=k * n, fill=0.0)
    var b_nk = List[Float32](length=n * k, fill=0.0)
    var c = List[Float32](length=m * n, fill=0.0)
    _fill(a_mk, 1)
    _fill(a_km, 2)
    _fill(b_kn, 3)
    _fill(b_nk, 4)
    var pa_mk = a_mk.unsafe_ptr().as_unsafe_any_origin()
    var pa_km = a_km.unsafe_ptr().as_unsafe_any_origin()
    var pb_kn = b_kn.unsafe_ptr().as_unsafe_any_origin()
    var pb_nk = b_nk.unsafe_ptr().as_unsafe_any_origin()
    var pc = c.unsafe_ptr().as_unsafe_any_origin()

    var gf = List[Float64]()
    for variant in range(3):
        var times = List[Float64]()
        for rep in range(reps + 10):
            var t0 = perf_counter_ns()
            if variant == 0:
                gemm_nn(m, n, k, pa_mk, k, pb_kn, n, pc, n, False)
            elif variant == 1:
                gemm_nt(m, n, k, pa_mk, k, pb_nk, k, pc, n, False)
            else:
                gemm_tn(m, n, k, pa_km, m, pb_kn, n, pc, n, False)
            var t1 = perf_counter_ns()
            if rep >= 10:
                times.append(Float64(t1 - t0) * 1e-9)
        var med = _median(times)
        var g = flops / med * 1e-9
        gf.append(g)
        var name = "gemm_nn"
        if variant == 1:
            name = "gemm_nt"
        elif variant == 2:
            name = "gemm_tn"
        print(name, "m=" + String(m), "n=" + String(n), "k=" + String(k), "median_ms=" + String(med * 1e3), "GFLOP/s=" + String(g))
    # keep the operand lists alive until here
    _ = len(a_mk) + len(a_km) + len(b_kn) + len(b_nk) + len(c)
    print(
        '{"bench":"rl_gemm","m":' + String(m) + ',"n":' + String(n) + ',"k":' + String(k)
        + ',"gflops_nn":' + String(gf[0]) + ',"gflops_nt":' + String(gf[1]) + ',"gflops_tn":' + String(gf[2]) + "}"
    )
