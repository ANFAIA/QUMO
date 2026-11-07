"""Tests for the neutral topologies, the circuit-family job generator, the evaluation sets and the eval summary.

Covers (MOJO_PORT.md 13.2, `test_rl_jobs`): `topologies.json` match for all 35 topologies, connectivity, determinism of
`build_jobs`, structural invariants of every circuit family, JSON loading through the `route_tool` helpers, the
evaluation-set rules (seeds >= 910000, CRC-32 placement), and distribution drift of 2000 Mojo jobs per scale against
`route/jobs_stats.json` (chi-square for categorical features, two-sample KS for continuous ones, 4 sigma).
"""

from std.collections import List
from std.os import makedirs
from std.os.path import exists
from qumo.ai.json import JsonValue, parse_json
from qumo.compiler.target import Target
from qumo.ir.qasm import load_qasm
from qumo.rl.testutil import (
    load_fixture,
    ints,
    floats,
    chi_square_homogeneity,
    ks_against_knots,
    ks_critical,
)
from qumo.rl.digest import crc32_string
from qumo.rl.targets import (
    RTopo,
    neutral_topology,
    train_topologies,
    eval_topologies,
    topology_by_name,
)
from qumo.rl.jobs import (
    JobSet,
    JobCircuit,
    OP_CX,
    OP_CZ,
    OP_SWAP,
    OP_U2,
    OP_U1,
    OP_MEASURE,
    OP_BARRIER,
    build_jobs,
    family_names,
    eval_synth_set,
    eval_bench_set,
    eval_sets,
    summarize,
    fixed,
    count_qreg_qubits,
    circuit_from_json,
    target_from_json,
    put_i16,
    put_f32,
)

comptime OUT_DIR = "build/test_rl_jobs"


def check(cond: Bool, msg: String) raises:
    if not cond:
        raise Error("test_rl_jobs failed: " + msg)


# ------------------------------------------------------------------------------------------------ topologies


def edges_of(ej: JsonValue) raises -> List[Int]:
    var e = List[Int]()
    for k in range(len(ej)):
        e.append(ej.at(k).at(0).as_int())
        e.append(ej.at(k).at(1).as_int())
    return e^


def is_connected(t: RTopo) -> Bool:
    var adj = List[List[Int]]()
    for _ in range(t.n):
        adj.append(List[Int]())
    var k = 0
    while k + 1 < len(t.edges):
        adj[t.edges[k]].append(t.edges[k + 1])
        adj[t.edges[k + 1]].append(t.edges[k])
        k += 2
    var seen = List[Bool](length=t.n, fill=False)
    var stack = List[Int]()
    stack.append(0)
    seen[0] = True
    var cnt = 1
    while len(stack) > 0:
        var v = stack.pop()
        for w in adj[v]:
            if not seen[w]:
                seen[w] = True
                cnt += 1
                stack.append(w)
    return cnt == t.n


def check_topology_shape(t: RTopo) raises:
    check(t.n >= 2, t.name + ": n >= 2")
    check(len(t.edges) % 2 == 0 and len(t.edges) > 0, t.name + ": edge list is pairs")
    var seen = List[Int]()
    var k = 0
    while k + 1 < len(t.edges):
        var a = t.edges[k]
        var b = t.edges[k + 1]
        check(a >= 0 and a < t.n and b >= 0 and b < t.n, t.name + ": endpoint in range")
        check(a != b, t.name + ": no self loop")
        seen.append(min(a, b) * t.n + max(a, b))
        k += 2
    sort(seen)
    for i in range(1, len(seen)):
        check(seen[i] != seen[i - 1], t.name + ": no duplicate undirected edge")
    check(is_connected(t), t.name + ": connected")


def test_topologies() raises:
    var fx = load_fixture("topologies.json")
    var train = train_topologies()
    var evalt = eval_topologies()
    check(len(train) == 25, "25 training topologies")
    check(len(evalt) == 10, "10 evaluation topologies")
    check(len(fx.get("train")) == 25 and len(fx.get("eval")) == 10, "fixture sizes")
    for s in range(2):
        var group = fx.get("train") if s == 0 else fx.get("eval")
        for i in range(len(group)):
            var j = group.at(i)
            var t = train[i].copy() if s == 0 else evalt[i].copy()
            var what = String("train " if s == 0 else "eval ") + String(i) + " " + t.name
            check(t.name == j.get("name").as_string(), what + ": name " + t.name + " vs " + j.get("name").as_string())
            check(t.n == j.get("n").as_int(), what + ": n")
            check(t.family == j.get("family").as_string(), what + ": family")
            var fe = edges_of(j.get("edges"))
            check(len(fe) == len(t.edges), what + ": edge count")
            for k in range(len(fe)):
                check(fe[k] == t.edges[k], what + ": edge entry " + String(k) + " (emission order)")
            # the builder reached through its (kind, size) parameters gives the same topology
            var size = j.get("size")
            var u = neutral_topology(j.get("kind").as_string(), size.at(0).as_int(), size.at(1).as_int()) if size.is_array() else neutral_topology(j.get("kind").as_string(), size.as_int())
            check(u.name == t.name and u.n == t.n and len(u.edges) == len(t.edges), what + ": neutral_topology")
            for k in range(len(fe)):
                check(u.edges[k] == fe[k], what + ": neutral_topology edges")
            check_topology_shape(t)
            var byname = topology_by_name(t.name)
            check(byname.n == t.n and len(byname.edges) == len(t.edges), what + ": topology_by_name")
    # a minimum qubit count rounds up to the real size of the family
    check(neutral_topology("tree", 16).n == 31, "tree 16 -> 31 qubits")
    check(neutral_topology("torus", 10).n == 16, "torus 10 -> 16 qubits")
    check(neutral_topology("torus", 2).n == 9, "torus side >= 3")
    check(neutral_topology("diag", 16).n == 18, "diag 16 -> 18")
    check(neutral_topology("octagonal", 17).n == 32, "octagonal 17 -> 3 octagons in a 2x2 tiling -> 32")
    check(neutral_topology("heavy_hex", 20).n == 57, "heavy-hex 20 -> d=5 (57 qubits)")
    check(neutral_topology("ionring", 5).n == 6, "ion ring: 3 zones of 2")
    check(neutral_topology("grid", 3).n == 9, "grid with one size is square")
    var raised = False
    try:
        _ = neutral_topology("hexagon", 4)
    except:
        raised = True
    check(raised, "unknown kind raises")
    raised = False
    try:
        _ = neutral_topology("ring", 2)
    except:
        raised = True
    check(raised, "ring of 2 raises")
    raised = False
    try:
        _ = topology_by_name("nope9")
    except:
        raised = True
    check(raised, "unknown topology name raises")


# ------------------------------------------------------------------------------------------------ build_jobs


def is_2q(kind: Int) -> Bool:
    return kind == OP_CX or kind == OP_CZ or kind == OP_SWAP or kind == OP_U2


def tail_int(token: StringSlice) raises -> Int:
    """Integer after the one-letter prefix of a name token, e.g. `d4` -> 4."""
    var t = String(token)
    return Int(String(t[byte=1:]))



def check_job_set(js: JobSet, seed: Int, max_n: Int, scale: Float64, min_frac: Float64, what: String) raises:
    check(js.num_jobs() == len(js.circuits), what + ": one circuit per job")
    check(js.num_jobs() > 0, what + ": some jobs")
    for j in range(js.num_jobs()):
        var ci = js.jobs[3 * j]
        var ti = js.jobs[3 * j + 1]
        var c = js.circuits[ci].copy()
        var t = js.targets[ti].copy()
        var tag = what + " job " + String(j) + " " + c.name
        check(ci == j, tag + ": circuit index")
        check(c.n >= 3 and c.n <= t.n, tag + ": circuit fits the topology")
        check(c.n <= max(max_n, 4), tag + ": size bound")
        # router seed and name suffix `_s<seed>_<job index>`
        var parts = c.name.split("_")
        var jidx = Int(String(parts[len(parts) - 1]))
        var sd = String(parts[len(parts) - 2])
        check(sd == "s" + String(seed), tag + ": seed in name")
        check(js.jobs[3 * j + 2] == seed * 100000 + jidx, tag + ": job seed")
        var fam = c.family
        check(c.name.startswith(fam + "_"), tag + ": family prefix")
        var nops = c.num_ops()
        check(nops > 0 or fam == "rand2q", tag + ": has ops")
        var n_u1 = 0
        var n_2q = 0
        var n_meas = 0
        for o in range(nops):
            var kind = c.ops[3 * o]
            var a = c.ops[3 * o + 1]
            var b = c.ops[3 * o + 2]
            check(kind != OP_BARRIER and kind != OP_SWAP, tag + ": generated op kinds")
            check(a >= 0 and a < c.n, tag + ": qubit a in range")
            if is_2q(kind):
                check(b >= 0 and b < c.n and a != b, tag + ": 2q operands distinct and in range")
                n_2q += 1
            else:
                check(b == -1, tag + ": 1q op has b = -1")
                if kind == OP_U1:
                    n_u1 += 1
                else:
                    check(kind == OP_MEASURE, tag + ": 1q kinds")
                    n_meas += 1
        # per-family structure from the circuit name
        if fam == "qft":
            check(nops == c.n + c.n * (c.n - 1) // 2, tag + ": qft op count")
            check(n_2q == c.n * (c.n - 1) // 2 and n_u1 == c.n, tag + ": qft gates")
        elif fam == "qv":
            var d = tail_int(parts[2])
            check(nops == d * (c.n // 2) * 6, tag + ": qv op count")
            check(n_2q == d * (c.n // 2) * 3, tag + ": qv 2q count")
        elif fam == "rand2q":
            var d = tail_int(parts[2])
            check(n_2q <= d * (c.n // 2) and nops <= 2 * d * (c.n // 2), tag + ": rand2q bounds")
            check(d >= 2, tag + ": rand2q depth >= 2")
        elif fam == "sparse":
            var m = tail_int(parts[2])
            check(n_2q == m and m >= 4, tag + ": sparse 2q count")
            check(n_meas <= c.n, tag + ": sparse measurements")
        elif fam == "trotter":
            var s = tail_int(parts[2])
            check(n_u1 == s * c.n and s >= 1, tag + ": trotter 1q layers")
            check(n_2q >= s * (c.n - 1), tag + ": trotter 2q count")
        elif fam == "qaoa":
            var p = tail_int(parts[2])
            check(c.n >= 4 and p >= 1, tag + ": qaoa size")
            var edges_x3 = n_2q // (2 * p)  # 2 CX per edge per layer
            check(n_2q == 2 * p * edges_x3, tag + ": qaoa cx pairs")
            check(n_u1 == c.n + p * (edges_x3 + c.n), tag + ": qaoa 1q count")
            check(edges_x3 <= 3 * c.n // 2, tag + ": qaoa edge count of a 3-regular-ish graph")
        else:
            check(False, tag + ": unknown family " + fam)


def test_build_jobs() raises:
    var empty_t = List[RTopo]()
    var empty_f = List[String]()
    var a = build_jobs(31, 400, empty_t, empty_f, 30, 0.5, 0.4)
    var b = build_jobs(31, 400, empty_t, empty_f, 30, 0.5, 0.4)
    check(a.to_json() == b.to_json(), "build_jobs is deterministic")
    var c = build_jobs(32, 400, empty_t, empty_f, 30, 0.5, 0.4)
    check(a.to_json() != c.to_json(), "different seeds differ")
    check(len(a.targets) == 25, "default topologies are the 25 training ones")
    check_job_set(a, 31, 30, 0.5, 0.4, "scale 0.5")
    var d = build_jobs(7, 400, empty_t, empty_f, 24, 1.0, 0.4)
    check_job_set(d, 7, 24, 1.0, 0.4, "scale 1.0")
    # all six families appear, in roughly equal numbers
    var counts = List[Int](length=6, fill=0)
    var fams = family_names()
    for i in range(len(a.circuits)):
        for k in range(6):
            if a.circuits[i].family == fams[k]:
                counts[k] += 1
    for k in range(6):
        check(counts[k] > 40, "family " + fams[k] + " drawn (" + String(counts[k]) + " of 400)")
    # restricted topologies and families
    var one = List[RTopo]()
    one.append(topology_by_name("ring14"))
    var qft_only: List[String] = ["qft"]
    var q = build_jobs(3, 30, one, qft_only, 14, 1.0, 0.4)
    check(len(q.targets) == 1 and q.num_jobs() == 30, "single topology, qft only")
    for i in range(len(q.circuits)):
        check(q.circuits[i].family == "qft" and q.circuits[i].n <= 14, "qft only")
        check(q.jobs[3 * i + 1] == 0, "target index 0")
    # size window of the generator: lo = max(3, int(N * min_frac)), hi = min(N, max_n)
    var w = build_jobs(5, 200, one, List[String](), 9, 1.0, 0.6)
    for i in range(len(w.circuits)):
        var cc = w.circuits[i].copy()
        if cc.family != "qft" and cc.family != "qaoa":
            check(cc.n >= 8 and cc.n <= 9, "size window [int(14*0.6), 9] -> [8, 9], got " + String(cc.n))
    # errors
    var bad_fams: List[String] = ["nope"]
    var raised = False
    try:
        _ = build_jobs(1, 5, empty_t, bad_fams, 30, 1.0, 0.4)
    except:
        raised = True
    check(raised, "unknown family raises")
    raised = False
    try:
        _ = build_jobs(-1, 5, empty_t, empty_f, 30, 1.0, 0.4)
    except:
        raised = True
    check(raised, "negative seed raises")
    raised = False
    try:
        _ = build_jobs(1, 5, empty_t, empty_f, 2, 1.0, 0.4)
    except:
        raised = True
    check(raised, "max_n below 3 raises")


def test_json_roundtrip() raises:
    var js = build_jobs(11, 60, List[RTopo](), List[String](), 20, 0.5, 0.4)
    var doc = parse_json(js.to_json())
    check(len(doc.get("targets")) == 25, "json targets")
    check(len(doc.get("circuits")) == len(js.circuits), "json circuits")
    check(len(doc.get("jobs")) == js.num_jobs(), "json jobs")
    for i in range(len(js.targets)):
        var t = target_from_json(doc.get("targets").at(i))
        check(t.num_qubits == js.targets[i].n, "target_from_json qubits")
        check(doc.get("targets").at(i).get_string("family", "") == js.targets[i].family, "json target family")
    for i in range(len(js.circuits)):
        var cj = doc.get("circuits").at(i)
        var cc = circuit_from_json(cj)
        check(cc.num_qubits == js.circuits[i].n, "circuit_from_json qubits: " + js.circuits[i].name)
        check(len(cc.ops) == js.circuits[i].num_ops(), "circuit_from_json op count: " + js.circuits[i].name)
        check(cj.get("family").as_string() == js.circuits[i].family, "json circuit family")
    for j in range(js.num_jobs()):
        var jj = doc.get("jobs").at(j)
        check(jj.at(0).as_int() == js.jobs[3 * j] and jj.at(1).as_int() == js.jobs[3 * j + 1] and jj.at(2).as_int() == js.jobs[3 * j + 2], "json job row")


def test_helpers() raises:
    var b = List[UInt8]()
    put_i16(b, 258)
    put_i16(b, -2)
    put_i16(b, 70000)
    put_i16(b, -70000)
    var expect: List[Int] = [2, 1, 254, 255, 255, 127, 0, 128]
    check(len(b) == 8, "put_i16 length")
    for i in range(8):
        check(Int(b[i]) == expect[i], "put_i16 byte " + String(i) + ": " + String(Int(b[i])))
    var f = List[UInt8]()
    put_f32(f, Float32(1.0))
    put_f32(f, Float32(-2.5))
    var fe: List[Int] = [0, 0, 128, 63, 0, 0, 32, 192]
    for i in range(8):
        check(Int(f[i]) == fe[i], "put_f32 byte " + String(i))
    check(fixed(1.0005, 3) == "1.001" or fixed(1.0005, 3) == "1.000", "fixed rounds to 3 digits")
    check(fixed(2.0, 3) == "2.000", "fixed 2.0")
    check(fixed(0.05, 1) == "0.1", "fixed half rounds up: " + fixed(0.05, 1))
    check(fixed(12.34, 1) == "12.3", "fixed 12.34")
    check(fixed(0.0, 3) == "0.000", "fixed zero")
    check(count_qreg_qubits("OPENQASM 2.0;\nqreg q[5];\nqreg  anc[2];\ncreg c[5];\n") == 7, "qreg sum")
    check(count_qreg_qubits("// no registers\ncreg c[4];") == 0, "no qreg")
    check(count_qreg_qubits("qreg\tq0_a[12];") == 12, "tab separator")


# ------------------------------------------------------------------------------------------------ evaluation sets


def read_json_file(path: String) raises -> JsonValue:
    var text: String
    with open(path, "r") as f:
        text = f.read()
    return parse_json(text)


def test_crc32() raises:
    # reference values of zlib.crc32 (checked with an independent gzip trailer)
    check(crc32_string("123456789") == UInt32(0xCBF43926), "crc32 check value")
    check(crc32_string("feynman__W-state.qasm") == UInt32(0xB45C9159), "crc32 W-state")
    check(crc32_string("feynman__adder.qasm") == UInt32(0xEAEFBDC0), "crc32 adder")
    check(crc32_string("feynman__barenco_tof_3.qasm") == UInt32(0x07B6122A), "crc32 barenco_tof_3")


def test_eval_sets() raises:
    var s = eval_synth_set()
    check(len(s.targets) == 10, "synth: 10 targets")
    var eval_t = eval_topologies()
    for i in range(10):
        check(s.targets[i].name == eval_t[i].name, "synth: target order")
    var per_topo = List[Int](length=10, fill=0)
    for j in range(s.num_jobs()):
        var ti = s.jobs[3 * j + 1]
        per_topo[ti] += 1
        var seed = s.jobs[3 * j + 2]
        check(seed // 100000 == 910000 + ti, "synth: build seed 910000 + topology index, got " + String(seed))
        check(seed // 100000 >= 910000, "synth: seeds >= 910000 (training uses < 800000)")
        var c = s.circuits[s.jobs[3 * j]].copy()
        check(c.n <= min(20, s.targets[ti].n) and c.n >= 3, "synth: circuit size")
    for ti in range(10):
        check(per_topo[ti] >= 6 and per_topo[ti] <= 8, "synth: about 8 jobs per topology, got " + String(per_topo[ti]))
    check(eval_synth_set().to_json() == s.to_json(), "synth set is deterministic")

    var bset = eval_bench_set()
    check(len(bset.circuits) > 40, "bench: circuits found (" + String(len(bset.circuits)) + ")")
    check(bset.num_jobs() == 3 * len(bset.circuits), "bench: 3 jobs per circuit")
    var known_names: List[String] = ["feynman__W-state.qasm", "feynman__adder.qasm", "feynman__barenco_tof_3.qasm"]
    var known_crc: List[UInt32] = [UInt32(0xB45C9159), UInt32(0xEAEFBDC0), UInt32(0x07B6122A)]
    var found = 0
    for ci in range(len(bset.circuits)):
        var c = bset.circuits[ci].copy()
        check(c.n >= 3 and c.n <= 14, "bench: 3 <= n <= 14 for " + c.name)
        check(c.qasm == "bench/neutral/circuits/" + c.name + ".qasm", "bench: qasm path " + c.qasm)
        check(exists(c.qasm), "bench: file exists " + c.qasm)
        var h = crc32_string(c.name + ".qasm")
        for k in range(3):
            var row = 3 * (3 * ci + k)
            check(bset.jobs[row] == ci, "bench: circuit index")
            check(bset.jobs[row + 1] == (Int(h) + 3 * k) % 10, "bench: topology (crc32 + 3k) % 10 for " + c.name)
            check(bset.jobs[row + 2] == 7, "bench: seed 7")
        for i in range(3):
            if c.name + ".qasm" == known_names[i]:
                found += 1
                check(h == known_crc[i], "bench: known crc")
    check(found == 3, "bench: the three reference files are in the set (" + String(found) + ")")
    # the qubit counts agree with the QASM parser for the circuits that are in (and a few that are out)
    for ci in range(0, len(bset.circuits), 6):
        var c = bset.circuits[ci].copy()
        check(load_qasm(c.qasm).num_qubits == c.n, "bench: parser qubit count of " + c.name)
    # eval_sets writes both files, loadable as jobs files
    eval_sets(OUT_DIR + "/sets")
    var ds = read_json_file(OUT_DIR + "/sets/eval_synth.json")
    var db = read_json_file(OUT_DIR + "/sets/eval_bench.json")
    check(len(ds.get("jobs")) == s.num_jobs() and len(ds.get("targets")) == 10, "eval_synth.json content")
    check(len(db.get("jobs")) == bset.num_jobs() and len(db.get("circuits")) == len(bset.circuits), "eval_bench.json content")
    var c0 = circuit_from_json(db.get("circuits").at(0))
    check(c0.num_qubits == bset.circuits[0].n, "bench circuit loads through circuit_from_json")


# ------------------------------------------------------------------------------------------------ summarize


def test_summarize() raises:
    makedirs(OUT_DIR, exist_ok=True)
    # one grid job: policy costs twice the base (19+1 vs 9+1), same depth, twice the time
    var rows = String()
    rows += "{\"circuit\":\"c0\",\"target\":\"grid4x4\",\"n\":5,\"seed\":1,\"sabre_o2\":[9,4,2.0],\"policy\":[19,4,4.0,0,0],\"portfolio\":[9,3,3.0,1,0]}\n"
    rows += "{\"circuit\":\"c1\",\"target\":\"grid4x4\",\"n\":5,\"seed\":2,\"sabre_o2\":[3,2,1.0],\"policy\":[1,2,1.0,0,0],\"portfolio\":null}\n"
    with open(OUT_DIR + "/ev.jsonl", "w") as f:
        f.write(rows)
    var text = summarize(OUT_DIR + "/ev.jsonl")
    var lines = text.split("\n")
    # job 0 cost ratio 20/10 = 2, job 1 cost ratio 2/4 = 0.5: geometric mean 1.000; time ratios 2 and 1 -> 1.414
    var want_policy = String("grid           policy     n=  2 cx=1.000 d2=1.000 t=   1.4x W/L=1/1")
    var want_all = String("ALL            policy     n=  2 cx=1.000 d2=1.000 t=   1.4x W/L=1/1")
    var want_portfolio = String("grid           portfolio  n=  1 cx=1.000 d2=0.800 t=   1.5x W/L=0/0 pol_won=1")
    var got = List[String]()
    for ln in lines:
        got.append(String(ln))
    check(got[0] == want_policy, "summarize policy line: '" + got[0] + "'")
    check(got[1] == want_portfolio, "summarize portfolio line: '" + got[1] + "'")
    check(got[2] == want_all, "summarize ALL policy line: '" + got[2] + "'")
    var md = summarize(OUT_DIR + "/ev.jsonl", String(), "sabre_o2", True)
    check(md.startswith("| family | config | jobs |"), "markdown header")
    check(md.find("| grid | policy | 2 | 1.000 | 1.000 | 1.4x | 1/1 |  |") >= 0, "markdown policy row")
    check(md.find("| ALL | portfolio | 1 | 1.000 | 0.800 | 1.5x | 0/0 | 1 |") >= 0, "markdown portfolio row")


# ------------------------------------------------------------------------------------------------ distribution drift


struct Features(Copyable):
    var n: List[Float64]
    var ops: List[Float64]
    var twoq: List[Float64]
    var depth: List[Float64]
    var nover: List[Float64]

    def __init__(out self):
        self.n = List[Float64]()
        self.ops = List[Float64]()
        self.twoq = List[Float64]()
        self.depth = List[Float64]()
        self.nover = List[Float64]()


def add_features(mut f: Features, c: JobCircuit, topo_n: Int):
    var nops = c.num_ops()
    var lvl = List[Int](length=c.n, fill=0)
    var n2 = 0
    var dmax = 0
    for o in range(nops):
        var a = c.ops[3 * o + 1]
        var b = c.ops[3 * o + 2]
        var m = lvl[a]
        if b >= 0:
            n2 += 1
            m = max(m, lvl[b])
        m += 1
        lvl[a] = m
        if b >= 0:
            lvl[b] = m
        dmax = max(dmax, m)
    f.n.append(Float64(c.n))
    f.ops.append(Float64(nops))
    f.twoq.append(Float64(n2) / Float64(nops) if nops > 0 else 0.0)
    f.depth.append(Float64(dmax))
    f.nover.append(Float64(c.n) / Float64(topo_n))


def ks_check(sample: List[Float64], knots: JsonValue, what: String, mut worst: Float64) raises:
    var d = ks_against_knots(sample, floats(knots.get("x")), floats(knots.get("F")))
    var lim = ks_critical(knots.get("n").as_int(), len(sample)) + 1.0 / 32.0
    var ratio = d / lim
    if ratio > worst:
        worst = ratio
    check(d <= lim, "KS drift " + what + ": D = " + String(d) + " > " + String(lim))


def hist_counts(cells: List[String], table: JsonValue) raises -> List[Int]:
    var out = List[Int]()
    for c in cells:
        var v = table.get(c)
        out.append(v.as_int() if v.is_defined() else 0)
    return out^


def chi_check(mojo: List[Int], reference: List[Int], what: String, mut worst_z: Float64) raises:
    var r = chi_square_homogeneity(mojo, reference)
    if r.z > worst_z:
        worst_z = r.z
    check(r.z <= 4.0, "chi-square drift " + what + ": z = " + String(r.z) + " (stat " + String(r.stat) + ", dof " + String(r.dof) + ")")


def test_distribution_drift() raises:
    var stats = load_fixture("route/jobs_stats.json")
    var scales = stats.get("scales")
    var fams = family_names()
    var worst_z: Float64 = -100.0
    var worst_ks: Float64 = 0.0
    for si in range(len(scales)):
        var sc = scales.at(si)
        var scale = sc.get("scale").as_float()
        var js = build_jobs(sc.get("seed").as_int(), sc.get("jobs_requested").as_int(), List[RTopo](), List[String](), sc.get("max_n").as_int(), scale, 0.4)
        var tag = "scale " + String(scale)
        var kept_ref = sc.get("jobs_kept").as_int()
        check(js.num_jobs() >= kept_ref - 20 and js.num_jobs() <= sc.get("jobs_requested").as_int(), tag + ": jobs kept " + String(js.num_jobs()) + " vs " + String(kept_ref))
        # categorical: family, topology, topology family
        var fam_cells = fams.copy()
        var topo_cells = List[String]()
        for t in js.targets:
            topo_cells.append(t.name)
        var tf_cells = sc.get("topology_family_counts").keys()
        var fam_m = List[Int](length=len(fam_cells), fill=0)
        var topo_m = List[Int](length=len(topo_cells), fill=0)
        var tf_m = List[Int](length=len(tf_cells), fill=0)
        for j in range(js.num_jobs()):
            var c = js.circuits[js.jobs[3 * j]].copy()
            var ti = js.jobs[3 * j + 1]
            for k in range(len(fam_cells)):
                if fam_cells[k] == c.family:
                    fam_m[k] += 1
            topo_m[ti] += 1
            for k in range(len(tf_cells)):
                if tf_cells[k] == js.targets[ti].family:
                    tf_m[k] += 1
        chi_check(fam_m, hist_counts(fam_cells, sc.get("family_counts")), tag + " family mix", worst_z)
        chi_check(topo_m, hist_counts(topo_cells, sc.get("topology_counts")), tag + " topology mix", worst_z)
        chi_check(tf_m, hist_counts(tf_cells, sc.get("topology_family_counts")), tag + " topology-family mix", worst_z)
        # continuous: all jobs, then per family
        var all = Features()
        var per = List[Features]()
        for _ in range(len(fams)):
            per.append(Features())
        for j in range(js.num_jobs()):
            var c = js.circuits[js.jobs[3 * j]].copy()
            var tn = js.targets[js.jobs[3 * j + 1]].n
            add_features(all, c, tn)
            for k in range(len(fams)):
                if fams[k] == c.family:
                    add_features(per[k], c, tn)
        var ra = sc.get("all")
        ks_check(all.n, ra.get("n"), tag + " all n", worst_ks)
        ks_check(all.ops, ra.get("ops"), tag + " all ops", worst_ks)
        ks_check(all.twoq, ra.get("twoq_frac"), tag + " all 2q fraction", worst_ks)
        ks_check(all.depth, ra.get("depth"), tag + " all depth", worst_ks)
        ks_check(all.nover, ra.get("n_over_topology_n"), tag + " all n/N", worst_ks)
        for k in range(len(fams)):
            var rf = sc.get("per_family").get(fams[k])
            var ftag = tag + " " + fams[k]
            ks_check(per[k].ops, rf.get("ops"), ftag + " ops", worst_ks)
            ks_check(per[k].twoq, rf.get("twoq_frac"), ftag + " 2q fraction", worst_ks)
            ks_check(per[k].depth, rf.get("depth"), ftag + " depth", worst_ks)
            ks_check(per[k].nover, rf.get("n_over_topology_n"), ftag + " n/N", worst_ks)
            # exact histogram of n per family
            var ncells = List[String]()
            var nm = List[Int](length=31, fill=0)
            for v in range(31):
                ncells.append(String(v))
            for x in per[k].n:
                nm[Int(x)] += 1
            chi_check(nm, hist_counts(ncells, rf.get("n_hist")), ftag + " n histogram", worst_z)
    print("drift: worst chi-square z =", worst_z, "; worst KS statistic / limit =", worst_ks)


def main() raises:
    makedirs(OUT_DIR, exist_ok=True)
    test_topologies()
    test_crc32()
    test_helpers()
    test_build_jobs()
    test_json_roundtrip()
    test_eval_sets()
    test_summarize()
    test_distribution_drift()
    print("test_rl_jobs OK")
