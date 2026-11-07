"""Routing job generator (circuit families), evaluation sets, eval summary and the JSON glue shared with route_tool.

A job set is the jobs file consumed by `training/ai_transpiler/route/route_tool.mojo`:

    {"targets":  [{"name", "n", "edges": [[a, b], ...], "family"}, ...],
     "circuits": [{"name", "n", "ops": [[kind, a, b], ...], "family"} | {"name", "qasm": path, "n"}, ...],
     "jobs":     [[circuit_index, target_index, seed], ...]}

Op kinds: cx cz swap u2 u1 measure barrier (b is -1 for one-qubit ops). The circuit families are six seeded
distributions (rand2q, qv, qft, trotter, qaoa, sparse); topologies come from `qumo.rl.targets`. All randomness is the
seeded `qumo.sim.rng.Rng` (xoshiro256**), so a job set is a pure function of its arguments.

Also holds the held-out evaluation sets (`eval_sets`), the geometric-mean summary of a `route_tool eval` JSONL file
(`summarize`) and the four helpers shared with `route_tool` (`circuit_from_json`, `target_from_json`, `put_i16`,
`put_f32`).
"""

from std.collections import List
from std.math import exp, log, sqrt
from std.memory import bitcast
from std.os import listdir, makedirs
from qumo.ai.json import JsonValue, parse_json, json_quote
from qumo.ir.circuit import Circuit
from qumo.ir.qasm import load_qasm
from qumo.compiler.target import Target
from qumo.sim.rng import Rng
from qumo.rl.targets import RTopo, train_topologies, eval_topologies
from qumo.rl.digest import crc32_string

# ------------------------------------------------------------------------------------------------ op kinds

comptime OP_CX = 0
comptime OP_CZ = 1
comptime OP_SWAP = 2
comptime OP_U2 = 3
comptime OP_U1 = 4
comptime OP_MEASURE = 5
comptime OP_BARRIER = 6

comptime JOBS_TAG = 0x4A4F42535F524C31  # mixed into the seed so job streams differ from other seeded consumers
comptime DEFAULT_CIRCUITS_DIR = "bench/neutral/circuits"


def op_name(kind: Int) -> String:
    if kind == OP_CX:
        return "cx"
    if kind == OP_CZ:
        return "cz"
    if kind == OP_SWAP:
        return "swap"
    if kind == OP_U2:
        return "u2"
    if kind == OP_U1:
        return "u1"
    if kind == OP_MEASURE:
        return "measure"
    return "barrier"


def family_names() -> List[String]:
    var f: List[String] = ["rand2q", "qv", "qft", "trotter", "qaoa", "sparse"]
    return f^


# ------------------------------------------------------------------------------------------------ data model


struct JobCircuit(Copyable):
    """One circuit of a job set: generated ops (flat `kind, a, b` triples) or a QASM file reference."""

    var name: String
    var n: Int
    var family: String
    var ops: List[Int]
    var qasm: String

    def __init__(out self):
        self.name = String()
        self.n = 0
        self.family = String()
        self.ops = List[Int]()
        self.qasm = String()

    def num_ops(self) -> Int:
        return len(self.ops) // 3


struct JobSet(Copyable):
    """A generated set of routing jobs, serialisable to the jobs JSON consumed by `route_tool`.

    `jobs` holds flat `(circuit index, target index, seed)` triples.
    """

    var targets: List[RTopo]
    var circuits: List[JobCircuit]
    var jobs: List[Int]

    def __init__(out self):
        self.targets = List[RTopo]()
        self.circuits = List[JobCircuit]()
        self.jobs = List[Int]()

    def num_jobs(self) -> Int:
        return len(self.jobs) // 3

    def to_json(self) -> String:
        var s = String("{\"targets\":[")
        for i in range(len(self.targets)):
            if i > 0:
                s += ","
            ref t = self.targets[i]
            s += "{\"name\":" + json_quote(t.name) + ",\"n\":" + String(t.n) + ",\"edges\":["
            var k = 0
            while k + 1 < len(t.edges):
                if k > 0:
                    s += ","
                s += "[" + String(t.edges[k]) + "," + String(t.edges[k + 1]) + "]"
                k += 2
            s += "],\"family\":" + json_quote(t.family) + "}"
        s += "],\"circuits\":["
        for i in range(len(self.circuits)):
            if i > 0:
                s += ","
            ref c = self.circuits[i]
            if c.qasm.byte_length() > 0:
                s += "{\"name\":" + json_quote(c.name) + ",\"qasm\":" + json_quote(c.qasm) + ",\"n\":" + String(c.n) + "}"
                continue
            s += "{\"name\":" + json_quote(c.name) + ",\"n\":" + String(c.n) + ",\"ops\":["
            var o = 0
            while o + 2 < len(c.ops):
                if o > 0:
                    s += ","
                s += "[\"" + op_name(c.ops[o]) + "\"," + String(c.ops[o + 1]) + "," + String(c.ops[o + 2]) + "]"
                o += 3
            s += "],\"family\":" + json_quote(c.family) + "}"
        s += "],\"jobs\":["
        var j = 0
        while j + 2 < len(self.jobs):
            if j > 0:
                s += ","
            s += "[" + String(self.jobs[j]) + "," + String(self.jobs[j + 1]) + "," + String(self.jobs[j + 2]) + "]"
            j += 3
        s += "]}"
        return s^


# ------------------------------------------------------------------------------------------------ random helpers


def job_rng(seed: Int) raises -> Rng:
    """The stream of one `build_jobs` call: `Rng` seeded by a mix of `seed` (non-negative) and a fixed tag."""
    if seed < 0:
        raise Error("QM-E640: job seed must be >= 0, got " + String(seed) + ". Hint: pass a non-negative integer seed.")
    return Rng(UInt64(seed) * 0x9E3779B97F4A7C15 + UInt64(JOBS_TAG))


@always_inline
def _randint(mut r: Rng, lo: Int, hi: Int) -> Int:
    """Uniform integer in [lo, hi] (both inclusive)."""
    return lo + r.below(hi - lo + 1)


def _shuffle(mut r: Rng, mut xs: List[Int]):
    """In-place Fisher-Yates, `i = len - 1 .. 1`, `j = below(i + 1)`."""
    var i = len(xs) - 1
    while i > 0:
        var j = r.below(i + 1)
        var t = xs[i]
        xs[i] = xs[j]
        xs[j] = t
        i -= 1


def _iota(n: Int) -> List[Int]:
    var v = List[Int](capacity=n)
    for i in range(n):
        v.append(i)
    return v^


def _isqrt(n: Int) -> Int:
    """Floor of the square root of a non-negative integer."""
    var c = Int(sqrt(Float64(n)))
    while c * c > n:
        c -= 1
    while (c + 1) * (c + 1) <= n:
        c += 1
    return c


@always_inline
def _op(mut ops: List[Int], kind: Int, a: Int, b: Int):
    ops.append(kind)
    ops.append(a)
    ops.append(b)


def _maybe_u1(mut ops: List[Int], mut r: Rng, q: Int):
    if r.uniform() < 0.5:
        _op(ops, OP_U1, q, -1)


# ------------------------------------------------------------------------------------------------ circuit families


def gen_rand2q(n: Int, depth: Int, mut r: Rng) -> List[Int]:
    """Random 2q layers: per layer a random pairing, each pair active with probability 0.7."""
    var ops = List[Int]()
    for _ in range(depth):
        var qs = _iota(n)
        _shuffle(r, qs)
        var k = 0
        while k < n - 1:
            if r.uniform() < 0.7:
                var a = qs[k]
                var b = qs[k + 1]
                _maybe_u1(ops, r, a)
                if r.uniform() < 0.8:
                    _op(ops, OP_CX, a, b)
                else:
                    _op(ops, OP_CZ, a, b)
            k += 2
    return ops^


def gen_qv(n: Int, depth: Int, mut r: Rng) -> List[Int]:
    """Quantum-volume-like: random pairing per layer, three CX blocks (random orientation) with 1q gates."""
    var ops = List[Int]()
    for _ in range(depth):
        var qs = _iota(n)
        _shuffle(r, qs)
        var k = 0
        while k < n - 1:
            var a = qs[k]
            var b = qs[k + 1]
            for _ in range(3):
                _op(ops, OP_U1, a, -1)
                if r.uniform() < 0.5:
                    _op(ops, OP_CX, a, b)
                else:
                    _op(ops, OP_CX, b, a)
            k += 2
    return ops^


def gen_qft(n: Int) -> List[Int]:
    """QFT-like all-pairs controlled-phase ladder (deterministic)."""
    var ops = List[Int]()
    for i in range(n):
        _op(ops, OP_U1, i, -1)
        for j in range(i + 1, n):
            _op(ops, OP_U2, j, i)
    return ops^


def gen_trotter(n: Int, steps: Int, mut r: Rng) -> List[Int]:
    """Heisenberg / Ising Trotter steps on a logical line or square-ish lattice with a random labelling."""
    var ea = List[Int]()
    var eb = List[Int]()
    if r.uniform() < 0.5:
        for i in range(n - 1):
            ea.append(i)
            eb.append(i + 1)
    else:
        var c = max(2, _isqrt(n))
        for q in range(n):
            if (q % c) + 1 < c and q + 1 < n:
                ea.append(q)
                eb.append(q + 1)
            if q + c < n:
                ea.append(q)
                eb.append(q + c)
    var perm = _iota(n)
    _shuffle(r, perm)
    var ops = List[Int]()
    for _ in range(steps):
        for e in range(len(ea)):
            var a = perm[ea[e]]
            var b = perm[eb[e]]
            var reps = 1 if r.below(2) == 0 else 3
            for _ in range(reps):
                _op(ops, OP_U2, a, b)
        for q in range(n):
            _op(ops, OP_U1, q, -1)
    return ops^


def gen_qaoa(n_in: Int, layers: Int, mut r: Rng) -> Tuple[List[Int], Int]:
    """QAOA on a random 3-regular-ish graph (configuration pairing, up to 50 attempts; the last attempt's edges are
    kept when none is simple, like the reference). Returns `(ops, n)` with `n` raised to at least 4."""
    var n = max(n_in, 4)
    var keys = List[Int]()  # edge keys a * n + b, a < b
    for _ in range(50):
        var stubs = List[Int](capacity=3 * n)
        for q in range(n):
            for _ in range(3):
                stubs.append(q)
        _shuffle(r, stubs)
        keys = List[Int]()
        var ok = True
        var k = 0
        while k < len(stubs) - 1:
            var a = stubs[k]
            var b = stubs[k + 1]
            var key = min(a, b) * n + max(a, b)
            var dup = False
            for x in keys:
                if x == key:
                    dup = True
                    break
            if a == b or dup:
                ok = False
                break
            keys.append(key)
            k += 2
        if ok:
            break
    sort(keys)
    var ops = List[Int]()
    for q in range(n):
        _op(ops, OP_U1, q, -1)
    for _ in range(layers):
        var order = _iota(len(keys))
        _shuffle(r, order)
        for idx in order:
            var a = keys[idx] // n
            var b = keys[idx] % n
            _op(ops, OP_CX, a, b)
            _op(ops, OP_U1, b, -1)
            _op(ops, OP_CX, a, b)
        for q in range(n):
            _op(ops, OP_U1, q, -1)
    return (ops^, n)


def gen_sparse(n: Int, m: Int, mut r: Rng) -> List[Int]:
    """`m` random long-range CX gates (optional 1q gate before each) and random final measurements."""
    var ops = List[Int]()
    for _ in range(m):
        var a = r.below(n)
        var b = (a + 1 + r.below(n - 1)) % n
        _maybe_u1(ops, r, a)
        _op(ops, OP_CX, a, b)
    for q in range(n):
        if r.uniform() < 0.3:
            _op(ops, OP_MEASURE, q, -1)
    return ops^


def make_circuit(family: String, n: Int, mut r: Rng, scale: Float64) raises -> JobCircuit:
    """One circuit of `family` on about `n` qubits; `scale` shrinks or grows the depth (curriculum)."""
    var c = JobCircuit()
    c.family = family
    if family == "rand2q":
        var d = max(2, Int(Float64(_randint(r, 3, 3 + n)) * scale))
        c.name = "rand2q_n" + String(n) + "_d" + String(d)
        c.n = n
        c.ops = gen_rand2q(n, d, r)
    elif family == "qv":
        var d = max(1, Int(Float64(_randint(r, 2, max(3, n // 2 + 1))) * scale))
        c.name = "qv_n" + String(n) + "_d" + String(d)
        c.n = n
        c.ops = gen_qv(n, d, r)
    elif family == "qft":
        var m = max(3, min(n, Int(Float64(n) * min(1.0, scale + 0.3))))
        c.name = "qft_n" + String(m)
        c.n = m
        c.ops = gen_qft(m)
    elif family == "trotter":
        var s = max(1, Int(Float64(_randint(r, 1, 3)) * scale))
        c.name = "trotter_n" + String(n) + "_s" + String(s)
        c.n = n
        c.ops = gen_trotter(n, s, r)
    elif family == "qaoa":
        var p = max(1, Int(Float64(_randint(r, 1, 2)) * scale))
        var res = gen_qaoa(n, p, r)
        c.name = "qaoa_n" + String(res[1]) + "_p" + String(p)
        c.n = res[1]
        c.ops = res[0].copy()
    elif family == "sparse":
        var m2 = max(4, Int(Float64(_randint(r, n, 4 * n)) * scale))
        c.name = "sparse_n" + String(n) + "_m" + String(m2)
        c.n = n
        c.ops = gen_sparse(n, m2, r)
    else:
        raise Error("QM-E640: unknown circuit family '" + family + "'. Hint: use rand2q qv qft trotter qaoa sparse.")
    return c^


def build_jobs(
    seed: Int,
    n_jobs: Int,
    topologies: List[RTopo],
    families: List[String],
    max_n: Int = 30,
    scale: Float64 = 1.0,
    min_frac: Float64 = 0.4,
) raises -> JobSet:
    """Random (topology, family, size) job mixes. An empty `topologies` means the 25 training topologies and an empty
    `families` all six families. A job whose circuit has more qubits than its topology is skipped (only the QAOA
    minimum of 4 qubits and QFT rounding can cause it), so fewer than `n_jobs` jobs may come back.

    Job `j` carries the router seed `seed * 100000 + j` and its circuit is named `<family>_..._s<seed>_<j>`.
    """
    var rng = job_rng(seed)
    var js = JobSet()
    if len(topologies) == 0:
        js.targets = train_topologies()
    else:
        js.targets = topologies.copy()
    var fams = families.copy() if len(families) > 0 else family_names()
    var known = family_names()
    for f in fams:
        var found = False
        for k in known:
            if k == f:
                found = True
        if not found:
            raise Error("QM-E640: unknown circuit family '" + f + "'. Hint: use rand2q qv qft trotter qaoa sparse.")
    var nt = len(js.targets)
    for j in range(n_jobs):
        var ti = rng.below(nt)
        var N = js.targets[ti].n
        var hi = min(N, max_n)
        var lo = max(3, min(hi, Int(Float64(N) * min_frac)))
        if hi < lo:
            raise Error("QM-E640: no circuit size fits topology '" + js.targets[ti].name + "' with max_n=" + String(max_n) + ". Hint: use max_n >= 3.")
        var n = _randint(rng, lo, hi)
        var fam = fams[rng.below(len(fams))]
        var c = make_circuit(fam, n, rng, scale)
        if c.n > N:
            continue
        c.name = c.name + "_s" + String(seed) + "_" + String(j)
        js.circuits.append(c^)
        js.jobs.append(len(js.circuits) - 1)
        js.jobs.append(ti)
        js.jobs.append(seed * 100000 + j)
    return js^


# ------------------------------------------------------------------------------------------------ evaluation sets


def eval_synth_set() raises -> JobSet:
    """Per evaluation topology, 8 jobs of the family mix at build seeds 910000 + topology index (training uses
    seeds below 800000), circuits of at most min(20, topology size) qubits."""
    var topos = eval_topologies()
    var out = JobSet()
    out.targets = topos.copy()
    for ti in range(len(topos)):
        var one = List[RTopo]()
        one.append(topos[ti].copy())
        var sub = build_jobs(910000 + ti, 8, one, List[String](), min(20, topos[ti].n), 1.0, 0.4)
        for k in range(sub.num_jobs()):
            out.circuits.append(sub.circuits[sub.jobs[3 * k]].copy())
            out.jobs.append(len(out.circuits) - 1)
            out.jobs.append(ti)
            out.jobs.append(sub.jobs[3 * k + 2])
    return out^


def _is_space(c: Int) -> Bool:
    return c == 32 or (c >= 9 and c <= 13)


def _is_word(c: Int) -> Bool:
    return (c >= 48 and c <= 57) or (c >= 65 and c <= 90) or (c >= 97 and c <= 122) or c == 95


def count_qreg_qubits(text: String) -> Int:
    """Sum of the sizes of all `qreg name[size]` declarations (the total qubit count of an OpenQASM 2 file)."""
    var bs = text.as_bytes()
    var nb = len(bs)
    var total = 0
    var pos = 0
    while pos + 4 <= nb:
        if (
            Int(bs[pos]) == ord("q")
            and Int(bs[pos + 1]) == ord("r")
            and Int(bs[pos + 2]) == ord("e")
            and Int(bs[pos + 3]) == ord("g")
        ):
            var p = pos + 4
            var w0 = p
            while p < nb and _is_space(Int(bs[p])):
                p += 1
            if p > w0:
                var n0 = p
                while p < nb and _is_word(Int(bs[p])):
                    p += 1
                if p > n0 and p < nb and Int(bs[p]) == ord("["):
                    p += 1
                    var d0 = p
                    var v = 0
                    while p < nb and Int(bs[p]) >= 48 and Int(bs[p]) <= 57:
                        v = v * 10 + (Int(bs[p]) - 48)
                        p += 1
                    if p > d0 and p < nb and Int(bs[p]) == ord("]"):
                        total += v
                        pos = p + 1
                        continue
        pos += 1
    return total


def eval_bench_set(circuits_dir: String = DEFAULT_CIRCUITS_DIR) raises -> JobSet:
    """The benchmark circuits (`*.qasm` of `circuits_dir`) with 3 <= qubits <= 14, each on 3 of the 10 evaluation
    topologies chosen by `(crc32(file name) + 3 k) % 10`, router seed 7. Files are visited in sorted order."""
    var topos = eval_topologies()
    var out = JobSet()
    out.targets = topos.copy()
    var names = listdir(circuits_dir)
    sort(names)
    for fi in range(len(names)):
        var f = names[fi]
        if not f.endswith(".qasm"):
            continue
        var path = circuits_dir + "/" + f
        var text: String
        with open(path, "r") as fh:
            text = fh.read()
        var n = count_qreg_qubits(text)
        if n < 3 or n > 14:
            continue
        var c = JobCircuit()
        c.name = String(f[byte = 0 : f.byte_length() - 5])
        c.qasm = path
        c.n = n
        out.circuits.append(c^)
        var ci = len(out.circuits) - 1
        var h = Int(crc32_string(f))
        for k in range(3):
            out.jobs.append(ci)
            out.jobs.append((h + 3 * k) % len(topos))
            out.jobs.append(7)
    return out^


def eval_sets(out_dir: String, circuits_dir: String = DEFAULT_CIRCUITS_DIR) raises:
    """Writes `eval_synth.json` and `eval_bench.json` (held-out router evaluation sets) into `out_dir` (created)."""
    makedirs(out_dir, exist_ok=True)
    var s = eval_synth_set()
    var b = eval_bench_set(circuits_dir)
    with open(out_dir + "/eval_synth.json", "w") as f:
        f.write(s.to_json())
    with open(out_dir + "/eval_bench.json", "w") as f:
        f.write(b.to_json())
    print("synth jobs", s.num_jobs(), "bench circuits", len(b.circuits), "bench jobs", b.num_jobs())


# ------------------------------------------------------------------------------------------------ eval summary


def _pad_right(s: String, w: Int) -> String:
    var out = s
    var k = s.byte_length()
    while k < w:
        out += " "
        k += 1
    return out^


def _pad_left(s: String, w: Int) -> String:
    var out = String()
    var k = s.byte_length()
    while k < w:
        out += " "
        k += 1
    out += s
    return out^


def fixed(x: Float64, decimals: Int) -> String:
    """`x` with exactly `decimals` digits after the point (round half away from zero), like `%.<d>f`."""
    var neg = x < 0.0
    var ax = -x if neg else x
    var p = 1
    for _ in range(decimals):
        p *= 10
    var scaled = Int(ax * Float64(p) + 0.5)
    var ip = scaled // p
    var fp = scaled % p
    var out = String("-") if (neg and scaled > 0) else String()
    out += String(ip)
    if decimals > 0:
        var fs = String(fp)
        while fs.byte_length() < decimals:
            fs = String("0") + fs
        out += "." + fs
    return out^


def family_of_target_name(name: String) -> String:
    """Family label of a neutral topology name (used when no jobs file is available to `summarize`)."""
    if name.startswith("heavy_hex"):
        return "heavy_hex"
    if name.startswith("atom_r2"):
        return "neutral_atom"
    if name.startswith("ionring"):
        return "ion_ring"
    if name.startswith("diag"):
        return "diag_grid"
    if name.startswith("oct"):
        return "octagonal"
    if name.startswith("grid"):
        return "grid"
    if name.startswith("line"):
        return "line"
    if name.startswith("ring"):
        return "ring"
    if name.startswith("tree"):
        return "tree"
    if name.startswith("torus"):
        return "torus"
    return name


struct _Acc(Copyable):
    var cx: Float64
    var d2: Float64
    var t: Float64
    var n: Int
    var wins: Int
    var losses: Int
    var won: Int

    def __init__(out self):
        self.cx = 0.0
        self.d2 = 0.0
        self.t = 0.0
        self.n = 0
        self.wins = 0
        self.losses = 0
        self.won = 0


def summarize(path: String, jobs_path: String = String(), base: String = "sabre_o2", md: Bool = False) raises -> String:
    """Geometric-mean ratios of each routing configuration against `base` from a `route_tool eval` JSONL file.

    Ratios use `(cx_units + 1)` and `(2q depth + 1)` so zero-swap jobs count as ties; the time ratio is the ratio of
    wall times (floored at 1e-3 ms). Rows are grouped by topology family: from `jobs_path` when given, else derived
    from the topology name. `md` selects a Markdown table instead of the plain text lines.
    """
    var fam = Dict[String, String]()
    if jobs_path.byte_length() > 0:
        var jtxt: String
        with open(jobs_path, "r") as f:
            jtxt = f.read()
        var jj = parse_json(jtxt)
        var tj = jj.get("targets")
        for i in range(len(tj)):
            var nm = tj.at(i).get_string("name", "")
            fam[nm] = tj.at(i).get_string("family", nm)
    var text: String
    with open(path, "r") as f:
        text = f.read()
    var rows = List[JsonValue]()
    for ln in text.split("\n"):
        var line = String(ln.strip())
        if line.byte_length() > 0:
            rows.append(parse_json(line))
    var all_cols: List[String] = ["policy", "portfolio", "heuristic", "scored", "sabre_o3", "sabre_o2"]
    var cols = List[String]()
    for c in all_cols:
        if c == base:
            continue
        var any_row = False
        for r in rows:
            var v = r.get(c)
            if v.is_array() and len(v) > 0:
                any_row = True
                break
        if any_row:
            cols.append(c)
    var keys = List[String]()
    var accs = Dict[String, _Acc]()  # key: family + "|" + config
    for r in rows:
        var tname = r.get_string("target", "")
        var b = r.get(base)
        if not (b.is_array() and len(b) >= 3):
            raise Error("QM-E640: row for circuit '" + r.get_string("circuit", "?") + "' has no '" + base + "' entry. Hint: pass --base with a configuration present in the file.")
        var b0 = b.at(0).as_float()
        var b1 = b.at(1).as_float()
        var b2 = b.at(2).as_float()
        var f = fam.get(tname, family_of_target_name(tname))
        var seen = False
        for k in keys:
            if k == f:
                seen = True
        if not seen:
            keys.append(f)
        for c in cols:
            var v = r.get(c)
            if not (v.is_array() and len(v) > 0):
                continue
            var v0 = v.at(0).as_float()
            var v1 = v.at(1).as_float()
            var v2 = v.at(2).as_float()
            for gi in range(2):
                var key = (f if gi == 0 else String("ALL")) + "|" + c
                var a = accs.get(key, _Acc())
                a.cx += log((v0 + 1.0) / (b0 + 1.0))
                a.d2 += log((v1 + 1.0) / (b1 + 1.0))
                a.t += log(max(v2, 1e-3) / max(b2, 1e-3))
                a.n += 1
                if v0 < b0:
                    a.wins += 1
                if v0 > b0:
                    a.losses += 1
                if len(v) > 3:
                    a.won += v.at(3).as_int()
                accs[key] = a.copy()
    sort(keys)
    keys.append("ALL")
    var out = String()
    if md:
        out += "| family | config | jobs | 2q cost ratio | 2q depth ratio | time ratio | wins/losses vs " + base + " | policy won (portfolio) |\n"
        out += "|---|---|---|---|---|---|---|---|\n"
    for k in keys:
        for c in cols:
            var a = accs.get(k + "|" + c, _Acc())
            if a.n == 0:
                continue
            var n = Float64(a.n)
            var gcx = exp(a.cx / n)
            var gd2 = exp(a.d2 / n)
            var gt = exp(a.t / n)
            var wl = String(a.wins) + "/" + String(a.losses)
            if md:
                out += "| " + k + " | " + c + " | " + String(a.n) + " | " + fixed(gcx, 3) + " | " + fixed(gd2, 3) + " | " + fixed(gt, 1) + "x | " + wl + " | " + (String(a.won) if c == "portfolio" else String()) + " |\n"
            else:
                out += _pad_right(k, 14) + " " + _pad_right(c, 10) + " n=" + _pad_left(String(a.n), 3) + " cx=" + fixed(gcx, 3) + " d2=" + fixed(gd2, 3) + " t=" + _pad_left(fixed(gt, 1), 6) + "x W/L=" + wl
                if c == "portfolio":
                    out += " pol_won=" + String(a.won)
                out += "\n"
    return out^


# ------------------------------------------------------------------------------------------------ route_tool helpers


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


def put_i16(mut b: List[UInt8], v: Int):
    var x = max(-32768, min(32767, v))
    var u = UInt16(x & 0xFFFF)
    b.append(UInt8(u & 0xFF))
    b.append(UInt8(u >> 8))


def put_f32(mut b: List[UInt8], v: Float32):
    var u = bitcast[DType.uint32](v)
    b.append(UInt8(u & 0xFF))
    b.append(UInt8((u >> 8) & 0xFF))
    b.append(UInt8((u >> 16) & 0xFF))
    b.append(UInt8(u >> 24))
