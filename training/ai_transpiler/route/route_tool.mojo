"""Training / evaluation driver for QUMO's learned router (front 'router').

Subcommands (run from the repo root with `.venv/bin/mojo run -I . <this file> ...`):

  gen  <jobs.json> <out.bin> <model_dir|-> <beta> <seed> <rollout:heuristic|model> <G>
      Behaviour-cloning / DAgger data. For every job (circuit, target, seed)
      one episode is played; at each policy decision the teacher scores every
      active swap k by Q(k) = routing overhead (CX units) after applying k and
      rolling out 2 seeded episodes (heuristic or the model) until G more 2q
      gates have executed (min over the 2 rollouts; +1000 + front distance if
      a rollout does not get there). The behaviour action is the teacher's
      argmin with probability beta, else the model's argmax (or the heuristic
      when no model is given). Record (little-endian, 328 bytes):
        int16 obs[128], int16 A, int16 action, int16 job, int16 teacher,
        float32 Q[16] (1e9 for invalid k).

  eval <jobs.json> <out.jsonl> <model_dir|-> <level>
      For every job: QUMO SABRE (route_circuit with the O2 / O3 route
      options), the policy alone (portfolio off), the portfolio, the
      heuristic env and the scored-SABRE ablation; routed_cost (CX units,
      2q depth), wall time; every AI result statevector-free verified
      structurally inside ai_route.

  episode <jobs.json> <model_dir|-> <out.json>
      Replays one episode per job (policy mode, from the job's layout and
      seed) and writes the action trace + overhead (Python/Mojo parity).

jobs.json: {"targets": [{"name", "n", "edges": [[a,b],...]}],
            "circuits": [{"name", "n", "ops": [[kind,a,b],...]} | {"name", "qasm": path}],
            "jobs": [[circuit, target, seed, layout?], ...]}
Op kinds: cx cz swap u2 (rzz 0.3) u1 (h) measure barrier.
"""

from std.sys import argv
from std.os import getenv
from std.time import perf_counter_ns
from std.math import exp
from qumo.ai.json import parse_json, JsonValue
from qumo.ir.circuit import Circuit, Instruction
from qumo.ir.gates import Op
from qumo.ir.qasm import load_qasm
from qumo.compiler.target import Target
from qumo.compiler.routing import RouteOptions, route_circuit
from qumo.compiler.basis import unroll_3q
from qumo.compiler.ai_route import (
    AIRouteOptions, ai_route_with_model, routed_cost, RouteModel, load_route_model, RProblem, Episode,
    episode_max_steps, reversed_ops, MODE_POLICY, MODE_HEURISTIC, MODE_SCORED, NUM_ACTIVE_SWAPS, OBS_SIZE,
    _random_layout, _complete,
)
from qumo.sim.rng import Rng
from max.algorithm import parallelize


def circuit_from_json(cj: JsonValue) raises -> Circuit:
    if cj.has("qasm"):
        return load_qasm(cj.get("qasm").as_string())
    var n = cj.get("n").as_int()
    var nm = 0
    var ops = cj.get("ops")
    for i in range(len(ops)):
        if ops.at(i).at(0).as_string() == "measure":
            nm += 1
    var c = Circuit(n, nm)
    var m = 0
    for i in range(len(ops)):
        var o = ops.at(i)
        var k = o.at(0).as_string()
        var a = o.at(1).as_int()
        var b = o.at(2).as_int()
        if k == "cx":
            c.cx(a, b)
        elif k == "cz":
            c.cz(a, b)
        elif k == "swap":
            c.swap(a, b)
        elif k == "u2":
            c.rzz(0.3, a, b)
        elif k == "u1":
            c.h(a)
        elif k == "measure":
            c.measure(a, m)
            m += 1
        elif k == "barrier":
            c.barrier()
        else:
            raise Error("QM-E602: unknown op kind '" + k + "' in jobs file. Hint: use cx cz swap u2 u1 measure barrier.")
    return c^


def target_from_json(tj: JsonValue) raises -> Target:
    var n = tj.get("n").as_int()
    var e = List[Int]()
    var ej = tj.get("edges")
    for i in range(len(ej)):
        e.append(ej.at(i).at(0).as_int())
        e.append(ej.at(i).at(1).as_int())
    var t = Target.from_edges(n, e, "any", tj.get_string("name", "custom"))
    return t^


def load_model_arg(s: String) raises -> RouteModel:
    if s == "-" or s == "":
        return RouteModel()
    return load_route_model(s)


def put_i16(mut b: List[UInt8], v: Int):
    var x = max(-32768, min(32767, v))
    var u = UInt16(x & 0xFFFF)
    b.append(UInt8(u & 0xFF))
    b.append(UInt8(u >> 8))


def put_f32(mut b: List[UInt8], v: Float32):
    from std.memory import bitcast
    var u = bitcast[DType.uint32](v)
    b.append(UInt8(u & 0xFF))
    b.append(UInt8((u >> 8) & 0xFF))
    b.append(UInt8((u >> 16) & 0xFF))
    b.append(UInt8(u >> 24))


comptime REC_BYTES = 328


def teacher_q(ep: Episode, prob: RProblem, target: Target, model: RouteModel, rmode: Int, G: Int,
              k: Int, seed: UInt64) -> Float64:
    """Q of active swap k: overhead after 2 seeded rollouts of G executed 2q gates."""
    var best = 1e18
    var limit = 8  # rollouts take the fallback swap early instead of wandering
    for r in range(2):
        var c = ep.copy()
        c.rng = Rng(seed + UInt64(r) * 7919)
        var base = ep.overhead()
        c.actions.clear()
        c.apply_swap(prob, target, c.active[k])
        var goal = ep.exec2q + G
        var steps = 0
        var ok = False
        while True:
            if c.finished(prob) or c.exec2q >= goal:
                ok = True
                break
            if not c.step(prob, target, model, rmode, 1.0, limit):
                ok = not c.failed
                break
            steps += 1
            if steps > 8 * G + 16:
                break
        var q = Float64(c.overhead() - base)
        if not ok:
            q += 1000.0 + Float64(c.fsum(prob, target))
        best = min(best, q)
    return best


def cmd_gen(jobs_path: String, out_path: String, model_arg: String, beta: Float64, seed: Int,
            rollout: String, G: Int) raises:
    var text: String
    with open(jobs_path, "r") as f:
        text = f.read()
    var j = parse_json(text)
    var targets = List[Target]()
    var tj = j.get("targets")
    for i in range(len(tj)):
        targets.append(target_from_json(tj.at(i)))
    var circs = List[Circuit]()
    var cj = j.get("circuits")
    for i in range(len(cj)):
        circs.append(circuit_from_json(cj.at(i)))
    var jj = j.get("jobs")
    var J = len(jj)
    var jc = List[Int]()
    var jt = List[Int]()
    var js = List[Int]()
    for i in range(J):
        jc.append(jj.at(i).at(0).as_int())
        jt.append(jj.at(i).at(1).as_int())
        js.append(jj.at(i).at(2).as_int())
    var model = load_model_arg(model_arg)
    var rmode = MODE_POLICY if (rollout == "model" and model.ok) else MODE_HEURISTIC
    comptime CAP = 256
    var out = List[UInt8]()
    var t0 = perf_counter_ns()
    var total = 0
    var B = 160
    var timing = getenv("ROUTE_TOOL_TIMING").byte_length() > 0
    var cap = CAP
    var capenv = getenv("ROUTE_TOOL_CAP")
    if capenv.byte_length() > 0:
        cap = min(CAP, Int(capenv))
    var start = 0
    while start < J:
        var nb = min(B, J - start)
        var buf = List[UInt8](length=nb * CAP * REC_BYTES, fill=0)
        var cnt = List[Int](length=nb, fill=0)
        var bp = buf.unsafe_ptr().as_unsafe_any_origin()
        var cp = cnt.unsafe_ptr().as_unsafe_any_origin()

        def task(t: Int) {imm circs, imm targets, imm jc, imm jt, imm js, imm model, imm rmode, imm beta, imm seed, imm G, imm start, imm bp, imm cp, imm timing, imm cap}:
            var job = start + t
            var tj0 = perf_counter_ns()
            var c = circs[jc[job]].copy()
            var target = targets[jt[job]].copy()
            var n = c.num_qubits
            var N = target.num_qubits
            var prob = RProblem(c.ops, n)
            var rng = Rng(UInt64(seed) * 1000003 + UInt64(js[job]))
            var l2p = _random_layout(N, rng)
            if js[job] % 2 == 1:
                # half of the episodes start from a reverse-pass placement
                var rev = RProblem(reversed_ops(c.ops), n)
                var e0 = Episode(rev, target, l2p, rng.next_u64(), False)
                e0.run(rev, target, model, MODE_HEURISTIC, 1.0, episode_max_steps(rev, target))
                if not e0.failed:
                    l2p = e0.l2p.copy()
            var l2p0 = l2p.copy()
            var ep = Episode(prob, target, l2p, rng.next_u64(), True)
            ep.rec_ops = False
            var limit = max(4, 2 * n)
            var maxs = episode_max_steps(prob, target)
            var recs = 0
            var mixr = Rng(UInt64(seed) * 31 + UInt64(job))
            var last_e = -1
            while recs < cap:
                var ex = ep.execute(prob, target)
                if ep.finished(prob):
                    break
                var f = ep.fsum(prob, target)
                if ex > 0:
                    ep.noprog = 0
                    ep.best_fsum = f
                    ep.fallback = False
                else:
                    if f < ep.best_fsum:
                        ep.best_fsum = f
                        ep.noprog = 0
                    else:
                        ep.noprog += 1
                    if ep.noprog >= limit:
                        ep.fallback = True
                ep.steps += 1
                if ep.steps > maxs:
                    break
                ep.build_layers(prob)
                ep.build_active(prob, target)
                var A = len(ep.active)
                if ep.fallback or A == 0:
                    var e = ep.fallback_swap(prob, target)
                    ep.n_fallback += 1
                    if e < 0:
                        break
                    last_e = e
                    ep.apply_swap(prob, target, e)
                    continue
                ep.build_obs(prob, target)
                var qs = List[Float64](length=NUM_ACTIVE_SWAPS, fill=1e9)
                var tb = 0
                var nl = len(ep.layer_start) - 1
                var qseed = mixr.next_u64()  # common random numbers across k
                for k in range(A):
                    # rollout cost; ties broken towards the SABRE-style score, and
                    # undoing the previous swap is discouraged (it only pays off
                    # through rollout noise and makes behaviour episodes oscillate)
                    qs[k] = teacher_q(ep, prob, target, model, rmode, G, k, qseed)
                    qs[k] += 0.01 * ep.heuristic_score(k, nl)
                    if ep.active[k] == last_e:
                        qs[k] += 1.5
                    if qs[k] < qs[tb]:
                        tb = k
                var act = tb
                if mixr.uniform() >= beta:
                    act = ep.choose(prob, target, model, MODE_POLICY if model.ok else MODE_HEURISTIC, 1.0)
                if recs < cap:
                    var off = (t * CAP + recs) * REC_BYTES
                    var rb = List[UInt8]()
                    for i in range(OBS_SIZE):
                        put_i16(rb, ep.obs[i])
                    put_i16(rb, A)
                    put_i16(rb, act)
                    put_i16(rb, job)
                    put_i16(rb, tb)
                    for k in range(NUM_ACTIVE_SWAPS):
                        put_f32(rb, Float32(qs[k]))
                    for i in range(REC_BYTES):
                        bp[unsafe_offset=off + i] = rb[i]
                    recs += 1
                last_e = ep.active[act]
                ep.apply_swap(prob, target, ep.active[act])
            cp[unsafe_offset=t] = recs
            if timing:
                var hov = 0
                for hr in range(4):
                    var eh = Episode(prob, target, l2p0, UInt64(hr) + 77, True)
                    eh.rec_ops = False
                    eh.run(prob, target, model, MODE_HEURISTIC, 1.0, maxs)
                    hov += eh.overhead()
                print("OVH", ep.overhead(), Float64(hov) / 4.0, "job", job, "n", n, "N", N, "nodes", prob.num_nodes(), "recs", recs, "steps", ep.steps, "ms", Float64(perf_counter_ns() - tj0) / 1e6)

        parallelize(task, nb)
        _ = len(buf)
        _ = len(cnt)
        for t in range(nb):
            for i in range(cnt[t] * REC_BYTES):
                out.append(buf[t * CAP * REC_BYTES + i])
            total += cnt[t]
        start += nb
        print("gen: jobs", start, "/", J, "records", total, "elapsed s", Float64(perf_counter_ns() - t0) / 1e9)
    with open(out_path, "w") as f:
        f.write_bytes(Span(out))
    print("gen done:", total, "records ->", out_path)


def _ms(t0: Int) -> Float64:
    return Float64(perf_counter_ns() - t0) / 1e6


def cmd_eval(jobs_path: String, out_path: String, model_arg: String, level: Int) raises:
    var text: String
    with open(jobs_path, "r") as f:
        text = f.read()
    var j = parse_json(text)
    var targets = List[Target]()
    var tnames = List[String]()
    var tj = j.get("targets")
    for i in range(len(tj)):
        targets.append(target_from_json(tj.at(i)))
        tnames.append(tj.at(i).get_string("name", "t" + String(i)))
    var cnames = List[String]()
    var circs = List[Circuit]()
    var cj = j.get("circuits")
    for i in range(len(cj)):
        circs.append(circuit_from_json(cj.at(i)))
        cnames.append(cj.at(i).get_string("name", "c" + String(i)))
    var model = load_model_arg(model_arg)
    var jj = j.get("jobs")
    var lines = String()
    var configs = String(getenv("ROUTE_EVAL_CONFIGS"))
    if configs.byte_length() == 0:
        configs = "sabre_o3,policy,portfolio,heuristic,scored"
    for i in range(len(jj)):
        var ci = jj.at(i).at(0).as_int()
        var ti = jj.at(i).at(1).as_int()
        var sd = jj.at(i).at(2).as_int()
        var c = circs[ci].copy()
        _ = unroll_3q(c)
        var t = targets[ti].copy()
        if c.num_qubits > t.num_qubits:
            continue
        var rec = String("{\"circuit\":\"") + cnames[ci] + "\",\"target\":\"" + tnames[ti] + "\",\"n\":" + String(c.num_qubits) + ",\"seed\":" + String(sd)
        # SABRE O2 / O3 route options
        for lv in range(2):
            if lv == 1 and configs.find("sabre_o3") < 0:
                continue
            var ro = RouteOptions()
            ro.seed = sd
            ro.trials = 20 if lv == 0 else 40
            ro.iterations = 2 if lv == 0 else 3
            ro.swap_trials = 20 if lv == 0 else 40
            var sc = c.copy()
            var il = List[Int]()
            var fl = List[Int]()
            var note = String()
            var t0 = perf_counter_ns()
            _ = route_circuit(sc, t, ro, il, fl, note)
            var ms = _ms(t0)
            var rc = routed_cost(sc, t)
            var tag = "sabre_o2" if lv == 0 else "sabre_o3"
            rec += ",\"" + tag + "\":[" + String(rc[0]) + "," + String(rc[1]) + "," + String(ms) + "]"
        var pols: List[String] = ["policy", "portfolio", "heuristic", "scored"]
        for pi_ in range(len(pols)):
            var p = pols[pi_]
            if (p == "policy" or p == "portfolio" or p == "scored") and not model.ok:
                continue
            if configs.find(p) < 0:
                continue
            var ao = AIRouteOptions()
            ao.level = level
            ao.seed = sd
            ao.portfolio = p == "portfolio"
            ao.policy = "x" if (p == "policy" or p == "portfolio") else p
            var ro = RouteOptions()
            ro.seed = sd
            ro.trials = 20
            ro.iterations = 2
            ro.swap_trials = 20
            var out = Circuit(1)
            var il = List[Int]()
            var fl = List[Int]()
            var note = String()
            var t0 = perf_counter_ns()
            var ok = ai_route_with_model(c, t, ro, ao, model, out, il, fl, note)
            var ms = _ms(t0)
            if not ok:
                rec += ",\"" + p + "\":null"
                continue
            var rc = routed_cost(out, t)
            var won = 1 if note.find("policy won") >= 0 else 0
            var rejected = 1 if note.find("rejected by verification") >= 0 else 0
            rec += ",\"" + p + "\":[" + String(rc[0]) + "," + String(rc[1]) + "," + String(ms) + "," + String(won) + "," + String(rejected) + "]"
        rec += "}\n"
        lines += rec
        print(rec, end="")
    with open(out_path, "w") as f:
        f.write(lines)


def cmd_episode(jobs_path: String, model_arg: String, out_path: String) raises:
    var text: String
    with open(jobs_path, "r") as f:
        text = f.read()
    var j = parse_json(text)
    var model = load_model_arg(model_arg)
    var tj = j.get("targets")
    var cj = j.get("circuits")
    var jj = j.get("jobs")
    var s = String("[")
    for i in range(len(jj)):
        var job = jj.at(i)
        var c = circuit_from_json(cj.at(job.at(0).as_int()))
        var t = target_from_json(tj.at(job.at(1).as_int()))
        var sd = job.at(2).as_int()
        var lj = job.at(3)
        var l2p = List[Int]()
        for k in range(len(lj)):
            l2p.append(lj.at(k).as_int())
        var prob = RProblem(c.ops, c.num_qubits)
        var ep = Episode(prob, t, l2p, UInt64(sd), True)
        ep.run(prob, t, model, MODE_POLICY if model.ok else MODE_HEURISTIC, 1.0, episode_max_steps(prob, t))
        if i > 0:
            s += ","
        s += "{\"actions\":["
        for k in range(len(ep.actions)):
            if k > 0:
                s += ","
            s += String(ep.actions[k])
        s += "],\"overhead\":" + String(ep.overhead()) + ",\"final\":["
        for k in range(len(ep.l2p)):
            if k > 0:
                s += ","
            s += String(ep.l2p[k])
        s += "],\"fallback\":" + String(ep.n_fallback) + "}"
    s += "]"
    with open(out_path, "w") as f:
        f.write(s)
    print("episodes ->", out_path)


def main() raises:
    var args = argv()
    if len(args) < 2:
        print("usage: route_tool.mojo gen|eval|episode ...")
        return
    var cmd = String(args[1])
    if cmd == "gen":
        cmd_gen(String(args[2]), String(args[3]), String(args[4]), Float64(String(args[5])), Int(String(args[6])), String(args[7]), Int(String(args[8])))
    elif cmd == "eval":
        cmd_eval(String(args[2]), String(args[3]), String(args[4]), Int(String(args[5])))
    elif cmd == "episode":
        cmd_episode(String(args[2]), String(args[3]), String(args[4]))
    else:
        print("unknown command", cmd)
