"""Performance gates of the native RL stack (MOJO_PORT.md section 15), one JSON line per gate.

Usage:  bench_rl [G1 G2 G3 G2p G4 G5 G6 G7 G8 G9 G10 G11 ...]      (no argument: every gate)

Build and run (on an otherwise idle machine; two processes share the same pool threads and slow each other):

    .venv/bin/mojo build -I . -o build/bench_rl bench/bench_rl.mojo && build/bench_rl

Each line is {"gate","metric","value","unit","gate_value","gate_op","stretch","pass","stretch_met","note"}. G10 (peak
resident set size) is measured first because the peak only grows, so it is the peak of its own workload only when the
other gates have not run yet. Run `bench_rl G10` alone for a clean figure. G11 is measured and reported, never a gate.
Timing is wall clock (perf_counter_ns); nothing here feeds a numeric result.

Measured configurations (MOJO_PORT.md 15):
  G1    single-core 512 x 512 x 256 f32 GEMM, nn / nt / tn (the worst of the three is reported)
  G2    one learn_step of the lf_line6 configuration (obs 36, E 512, C 256, A 10, 1024 episodes, difficulty 100, 4 cores),
        both evaluations every iteration; reference: the shipped Python-stack iteration of that model (about 6 s)
  G3    the same with the sampled evaluation every 5 iterations (the default)
  G2p   the pauli_line4 configuration (difficulty 200, both evaluations every iteration; reference 6.3 s)
  G4    collector decisions per second (4 cores, lf_line6 net, 1024 episodes at difficulty 100)
  G5    integrated ppo_update row-epochs per second (forward, loss, backward, reduction, Adam; 10 epochs; 4 cores)
  G6    router BC, 168k training rows x 30 epochs, hidden 64, 4 threads (planted-optimum teacher data)
  G7    models_verify on the committed model tree; SHA-256 throughput
  G8    build_jobs(31, 1000, max_n 30, scale 0.5) plus its JSON serialisation (the gen-jobs work, in-process)
  G9    export of one synthesis model (lf_line6 architecture, f16) including checks, vectors and rollouts
  G10   peak resident set size with N = 130k rows (collect plus a 2-epoch update on lf_line6)
  G11   iteration time at 2 versus 4 cores (reported only)
"""

from std.time import perf_counter_ns
from std.sys import argv
from std.sys.info import CompilationTarget
from std.ffi import external_call, c_int
from std.os import makedirs
from std.os.path import exists
from qumo.sim.rng import Rng
from qumo.compiler.policy import model_kind_code
from qumo.rl.tensor import gemm_nn, gemm_nt, gemm_tn
from qumo.rl.topology import parse_topology
from qumo.rl.gateset import EnvConfig, build_gateset
from qumo.rl.synth_env import TrainEnv
from qumo.rl.nn import NetSpec, BasicNet
from qumo.rl.optim import Adam
from qumo.rl.ppo import PPOConfig, Trainer, collect, evaluate, ppo_update
from qumo.rl.teacher_data import TeacherData, parse_teacher_bytes, encode_teacher_record
from qumo.rl.bc import BCConfig, train_bc
from qumo.rl.container import verify_models, remove_tree
from qumo.rl.digest import sha256_span_hex
from qumo.rl.jobs import build_jobs
from qumo.rl.targets import RTopo
from qumo.rl.checkpoint import (
    new_run_info, net_spec_for, write_run_json, write_best, write_summary, config_hash_of, TrainState, RunSummary,
)
from qumo.rl.export_synth import export_synth_run


def _now() -> Float64:
    return Float64(perf_counter_ns()) * 1e-9


def _median(mut v: List[Float64]) -> Float64:
    sort(v)
    return v[len(v) // 2]


def _fmt(x: Float64) -> String:
    """Compact JSON number (6 significant digits would hide a 1 percent change; keep the shortest round trip)."""
    return String(x)


def emit(
    gate: String, metric: String, value: Float64, unit: String, gate_value: Float64, higher_better: Bool,
    stretch: Float64, note: String,
):
    """One JSON line. `gate_op` is ">=" when higher is better, else "<="; `stretch` < 0 means none."""
    var ok = value >= gate_value if higher_better else value <= gate_value
    var st = False
    if stretch >= 0.0:
        st = value >= stretch if higher_better else value <= stretch
    var line = String('{"gate":"') + gate + '","metric":"' + metric + '","value":' + _fmt(value) + ',"unit":"' + unit
    line += '","gate_value":' + _fmt(gate_value) + ',"gate_op":"' + (">=" if higher_better else "<=") + '"'
    line += ',"stretch":' + (_fmt(stretch) if stretch >= 0.0 else String("null"))
    line += ',"pass":' + ("true" if ok else "false") + ',"stretch_met":' + ("true" if st else "false")
    line += ',"note":"' + note + '"}'
    print(line, flush=True)


def report(gate: String, metric: String, value: Float64, unit: String, note: String):
    """A measured figure that is not a gate (G11)."""
    var line = String('{"gate":"') + gate + '","metric":"' + metric + '","value":' + _fmt(value) + ',"unit":"' + unit
    line += '","pass":null,"note":"' + note + '"}'
    print(line, flush=True)


def peak_rss_mb() -> Float64:
    """Peak resident set size of this process in MB (getrusage ru_maxrss: bytes on macOS, kilobytes elsewhere)."""
    var buf = List[Int64](length=64, fill=0)
    var rc = external_call["getrusage", c_int](c_int(0), buf.unsafe_ptr())
    var raw = Float64(buf[4])  # struct rusage: ru_utime (16 bytes), ru_stime (16 bytes), then ru_maxrss
    _ = len(buf)
    if rc != 0:
        return -1.0
    comptime if CompilationTarget.is_macos():
        return raw / (1024.0 * 1024.0)
    else:
        return raw / 1024.0


# ------------------------------------------------------------------------------------------------ setup helpers


def make_env(kind_name: String, topo: String, difficulty: Int) raises -> TrainEnv:
    var t = parse_topology(topo)
    var cfg = EnvConfig()
    cfg.kind = model_kind_code(kind_name)
    cfg.n = t.n
    cfg.actions = build_gateset(cfg.kind, t.n, t.edges)
    cfg.difficulty = difficulty
    return TrainEnv(cfg, 1)


def make_net(env: TrainEnv) raises -> BasicNet:
    var tw = env.twists()
    var s = NetSpec()
    s.obs_size = env.obs_size()
    s.num_actions = env.num_actions()
    s.emb = 512
    s.common = [256]
    s.obs_perms = tw.obs_perms.copy()
    s.act_perms = tw.act_perms.copy()
    return BasicNet(s, 1)


def iteration_seconds(kind: String, topo: String, diff: Int, cores: Int, sampled_every: Int, iters: Int) raises -> Float64:
    """Mean wall time of `iters` learn_step calls after one warm-up call. With sampled_every = 5 the timed window starts
    at iteration 5, so it holds exactly one sampled evaluation per five iterations."""
    var env = make_env(kind, topo, diff)
    var net = make_net(env)
    var cfg = PPOConfig()
    cfg.num_cores = cores
    cfg.eval_sampled_every = sampled_every
    var tr = Trainer(env, net^, cfg)
    _ = tr.learn_step()  # warm-up (page faults, caches)
    if sampled_every > 1:
        tr.iteration = sampled_every
    var total = 0.0
    for _ in range(iters):
        var rep = tr.learn_step()
        total += rep.step_s
    return total / Float64(iters)


# ------------------------------------------------------------------------------------------------ gates


def gate_g1() raises:
    var m = 512
    var n = 512
    var k = 256
    var reps = 20
    var flops = 2.0 * Float64(m) * Float64(n) * Float64(k)
    var a_mk = List[Float32](length=m * k, fill=0.0)
    var a_km = List[Float32](length=k * m, fill=0.0)
    var b_kn = List[Float32](length=k * n, fill=0.0)
    var b_nk = List[Float32](length=n * k, fill=0.0)
    var c = List[Float32](length=m * n, fill=0.0)
    var r = Rng(1)
    for i in range(m * k):
        a_mk[i] = Float32(r.uniform()) - 0.5
        a_km[i] = Float32(r.uniform()) - 0.5
    for i in range(k * n):
        b_kn[i] = Float32(r.uniform()) - 0.5
        b_nk[i] = Float32(r.uniform()) - 0.5
    var pa_mk = a_mk.unsafe_ptr().as_unsafe_any_origin()
    var pa_km = a_km.unsafe_ptr().as_unsafe_any_origin()
    var pb_kn = b_kn.unsafe_ptr().as_unsafe_any_origin()
    var pb_nk = b_nk.unsafe_ptr().as_unsafe_any_origin()
    var pc = c.unsafe_ptr().as_unsafe_any_origin()
    var worst = 1e30
    var names: List[String] = ["gemm_nn", "gemm_nt", "gemm_tn"]
    var detail = String()
    for variant in range(3):
        var times = List[Float64]()
        for rep in range(reps + 10):
            var t0 = _now()
            if variant == 0:
                gemm_nn(m, n, k, pa_mk, k, pb_kn, n, pc, n, False)
            elif variant == 1:
                gemm_nt(m, n, k, pa_mk, k, pb_nk, k, pc, n, False)
            else:
                gemm_tn(m, n, k, pa_km, m, pb_kn, n, pc, n, False)
            var t1 = _now()
            if rep >= 10:
                times.append(t1 - t0)
        var g = flops / _median(times) * 1e-9
        detail += names[variant] + " " + String(Int(g)) + "; "
        worst = min(worst, g)
    _ = len(a_mk) + len(a_km) + len(b_kn) + len(b_nk) + len(c)
    emit("G1", "single-core GEMM 512x512x256, worst of nn/nt/tn", worst, "GFLOP/s", 60.0, True, 90.0, detail)


def gate_g4_g5() raises:
    var env = make_env("linear_function", "line:6", 100)
    var net = make_net(env)
    var cfg = PPOConfig()
    cfg.num_cores = 4
    # G4: collector throughput, median of 3 after a warm-up
    _ = collect(env, net, cfg, 0)
    var rates = List[Float64]()
    var last_rows = 0
    var keep = collect(env, net, cfg, 1)
    for rep in range(3):
        var t0 = _now()
        var ro = collect(env, net, cfg, 2 + rep)
        var dt = _now() - t0
        rates.append(Float64(ro.n) / dt)
        last_rows = ro.n
        if rep == 2:
            keep = ro^
    emit("G4", "collector decisions per second, 4 cores, lf_line6 net, 1024 episodes", _median(rates), "decisions/s",
         400000.0, True, 800000.0, "rows per iteration " + String(last_rows))
    # G5: integrated ppo_update, median of 3
    var opt = Adam(net.param_count(), cfg.lr)
    _ = ppo_update(net, opt, keep, cfg)  # warm-up
    var ups = List[Float64]()
    for _ in range(3):
        var t0 = _now()
        _ = ppo_update(net, opt, keep, cfg)
        var dt = _now() - t0
        ups.append(Float64(keep.n * cfg.num_epochs) / dt)
    emit("G5", "integrated ppo_update row-epochs per second, 4 cores", _median(ups), "row-epochs/s", 250000.0, True,
         400000.0, "rows " + String(keep.n) + ", epochs " + String(cfg.num_epochs))


def gate_g2_g3_g2p_g11() raises:
    var g2 = iteration_seconds("linear_function", "line:6", 100, 4, 1, 3)
    emit("G2", "lf_line6 iteration, both evaluations every iteration, 4 cores", g2, "s", 9.0, False, 6.0,
         "gate = 1.5 x the Python reference of about 6 s; stretch = 1.0 x")
    var g3 = iteration_seconds("linear_function", "line:6", 100, 4, 5, 5)
    emit("G3", "lf_line6 iteration, sampled evaluation every 5 iterations, 4 cores", g3, "s", 6.0, False, 3.6,
         "gate = 1.0 x the Python reference; stretch = 0.6 x")
    var g2p = iteration_seconds("pauli_network", "line:4", 200, 4, 1, 3)
    emit("G2p", "pauli_line4 iteration, both evaluations every iteration, 4 cores", g2p, "s", 9.4, False, 6.3,
         "gate = 1.5 x the Python reference 6.3 s; stretch = 1.0 x")
    var c2 = iteration_seconds("linear_function", "line:6", 100, 2, 1, 2)
    report("G11", "lf_line6 iteration at 2 cores (both evals)", c2, "s", "not a gate")
    report("G11", "lf_line6 iteration at 4 cores (both evals)", g2, "s", "ratio 2 vs 4 cores " + String(c2 / g2))


def _planted(rows: Int, seed: UInt64) -> List[UInt8]:
    var r = Rng(seed)
    var buf = List[UInt8]()
    for row in range(rows):
        var a = 2 + r.below(15)
        var best = r.below(a)
        var obs = List[Int](length=128, fill=0)
        var q = List[Float32]()
        for k in range(a):
            obs[8 * k] = (1 + r.below(3)) if k == best else -(1 + r.below(3))
            for d in range(1, 8):
                var u = r.below(3)
                obs[8 * k + d] = (u - 1) * (1 + r.below(2))
            q.append(0.0 if k == best else Float32(4.0 + 4.0 * r.uniform()))
        encode_teacher_record(buf, obs, a, best, row, best, q)
    return buf^


def gate_g6() raises:
    # 177,000 records: 168,150 training rows after the 5 percent validation hold-out, as in the shipped campaign
    var buf = _planted(177000, 1)
    var td = TeacherData()
    parse_teacher_bytes(buf, td, "bench")
    var cfg = BCConfig()
    cfg.threads = 4
    cfg.epochs = 30
    cfg.hidden = 64
    var t0 = _now()
    var out = train_bc(cfg, td, None)
    var dt = _now() - t0
    emit("G6", "router BC, 168k rows x 30 epochs, hidden 64, 4 threads", dt, "s", 53.0, False, 35.0,
         "samples " + String(out[1].samples) + ", final val_agree " + String(out[1].log[29].val_agree))


def gate_g7() raises:
    var best = 1e30
    var rep = verify_models("models/ai_transpiler")
    for _ in range(5):
        var t0 = _now()
        rep = verify_models("models/ai_transpiler")
        best = min(best, _now() - t0)
    emit("G7", "models_verify on models/ai_transpiler (in-process, best of 5)", best, "s", 1.0, False, -1.0,
         String(rep.n_files) + " files ok, " + String(rep.n_problems) + " problems, " + String(rep.total_bytes) + " bytes")
    # SHA-256 throughput over 32 MiB, best of 3
    var data = List[UInt8](length=32 * 1024 * 1024, fill=0)
    for i in range(len(data)):
        data[i] = UInt8(i & 255)
    var bt = 1e30
    for _ in range(3):
        var t0 = _now()
        var h = sha256_span_hex(Span(data))
        bt = min(bt, _now() - t0)
        if h.byte_length() != 64:
            print("bad digest")
    emit("G7", "SHA-256 throughput", 32.0 * 1.048576 / bt, "MB/s", 100.0, True, -1.0, "32 MiB buffer, best of 3")


def gate_g8() raises:
    var best = 1e30
    var n_jobs = 0
    var bytes = 0
    for _ in range(3):
        var t0 = _now()
        var js = build_jobs(31, 1000, List[RTopo](), List[String](), 30, 0.5, 0.4)
        var text = js.to_json()
        best = min(best, _now() - t0)
        n_jobs = js.num_jobs()
        bytes = text.byte_length()
    emit("G8", "build_jobs(31, 1000, max_n 30, scale 0.5) + JSON, in-process, best of 3", best, "s", 3.0, False, -1.0,
         String(n_jobs) + " jobs, " + String(bytes) + " bytes of JSON; the CLI adds process start")


def gate_g9() raises:
    var work = String("build/bench_rl_work")
    remove_tree(work)
    makedirs(work, exist_ok=True)
    var run = work + "/run"
    var kind = model_kind_code("linear_function")
    var info = new_run_info("bench_lf_line6", kind, "line:6")
    info.emb = 512
    info.cores = 4
    var net = BasicNet(net_spec_for(info), 1)
    write_run_json(run, info)
    var st = TrainState()
    st.it = 7
    st.difficulty = 5
    st.seed = 1
    st.config_hash = config_hash_of(info)
    _ = write_best(run, net, info, st)
    var s = RunSummary()
    s.finished = True
    s.iterations = 7
    s.final_difficulty = 5
    s.cores = 4
    s.seed = 1
    s.topology = "line:6"
    write_summary(run, s)
    var best = 1e30
    for _ in range(3):
        var t0 = _now()
        _ = export_synth_run(run, "best", "bench_lf_line6", work + "/root", "f16", False)
        best = min(best, _now() - t0)
    emit("G9", "export of one synthesis model (lf_line6 architecture, f16, with checks, vectors, rollouts), best of 3",
         best, "s", 3.0, False, -1.0, "in-process")
    remove_tree(work)


def gate_g10() raises:
    var env = make_env("linear_function", "line:6", 100)
    var net = make_net(env)
    var cfg = PPOConfig()
    cfg.num_cores = 4
    cfg.num_epochs = 2
    var ro = collect(env, net, cfg, 0)
    var opt = Adam(net.param_count(), cfg.lr)
    _ = ppo_update(net, opt, ro, cfg)
    emit("G10", "peak resident set size, collect + 2-epoch update at about 130k rows", peak_rss_mb(), "MB", 2048.0, False,
         1024.0, "rows " + String(ro.n) + "; only meaningful when no other gate ran before it in this process")


def wanted(args: List[String], g: String) -> Bool:
    if len(args) == 0:
        return True
    for a in args:
        if a == g:
            return True
    return False


def main() raises:
    var raw = argv()
    var args = List[String]()
    for i in range(1, len(raw)):
        args.append(String(raw[i]))
    # G10 first: the resident-set peak only grows
    if wanted(args, "G10"):
        gate_g10()
    if wanted(args, "G1"):
        gate_g1()
    if wanted(args, "G4") or wanted(args, "G5"):
        gate_g4_g5()
    if wanted(args, "G2") or wanted(args, "G3") or wanted(args, "G2p") or wanted(args, "G11"):
        gate_g2_g3_g2p_g11()
    if wanted(args, "G6"):
        gate_g6()
    if wanted(args, "G7"):
        gate_g7()
    if wanted(args, "G8"):
        gate_g8()
    if wanted(args, "G9"):
        gate_g9()
