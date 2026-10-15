# Evaluate one exported synthesis model against QUMO's own synthesizers on
# seeded random blocks of the model's coupling graph.
#
#   .venv/bin/mojo run -I . training/ai_transpiler/synth/eval_blocks.mojo \
#       <model_root> <model_id> <num_instances> <out.json>
#
# <model_root> is a directory with INDEX.json + <id>/ (e.g. models/ai_transpiler/synth).
# Proposers: policy (beams=1 greedy ensemble; beams=8 + 4 samples), block_synth
# greedy / auto / exact (exact only where `exact_supported`). Every output is
# verified with blockalg (Perm / GF2Mat / signed SympTab equality; Pauli
# networks by dense statevector comparison for n <= 10). Costs: 2q cost in CX
# units (CX = CZ = 1, SWAP = 3) and 2q depth. Writes a JSON summary.
from std.sys import argv
from std.time import perf_counter_ns
from qumo.ir.circuit import Instruction
from qumo.ir.gates import Op
from qumo.sim.rng import Rng
from qumo.ai.json import JsonWriter
from qumo.compiler.blockalg import (
    SubGraph, BlockKind, Perm, GF2Mat, SympTab, PauliNet, dense_equiv, ops_depth2q,
)
from qumo.compiler.block_synth import (
    synth_perm, synth_linear, synth_clifford, synth_pauli_network, exact_supported,
)
from qumo.compiler.policy import load_model_file, ModelFile
from qumo.compiler.policy_synth import PolicySession


def cost2(ops: List[Instruction]) -> Int:
    var c = 0
    for o in ops:
        if o.kind == Op.SWAP:
            c += 3
        elif o.kind == Op.CX or o.kind == Op.CZ:
            c += 1
    return c


struct Inst(Copyable):
    var src: List[Instruction]

    def __init__(out self, var src: List[Instruction]):
        self.src = src^


def rand_2q(g: SubGraph, mut rng: Rng, kind: Int) -> Instruction:
    var e = rng.below(g.num_edges())
    var a = g.edges[2 * e]
    var b = g.edges[2 * e + 1]
    if rng.below(2) == 1:
        var t = a
        a = b
        b = t
    return Instruction.make(kind, a, b)


def make_instance(kind: Int, n: Int, g: SubGraph, mut rng: Rng) -> List[Instruction]:
    """Seeded random block on the complete graph (so the model graph must route it)."""
    var full = SubGraph.complete(n)
    var ops = List[Instruction]()
    if kind == BlockKind.PERM:
        var p = List[Int]()
        for i in range(n):
            p.append(i)
        for i in range(n - 1, 0, -1):
            var j = rng.below(i + 1)
            var t = p[i]
            p[i] = p[j]
            p[j] = t
        # realise p as a SWAP circuit on the complete graph (cycle decomposition)
        var cur = List[Int]()
        for i in range(n):
            cur.append(i)
        var pos = List[Int]()  # pos[i] = where state i is
        for i in range(n):
            pos.append(i)
        var who = List[Int]()
        for i in range(n):
            who.append(i)
        for i in range(n):
            # bring state s with p[s] == i to position i
            var s = -1
            for k in range(n):
                if p[k] == i:
                    s = k
            var at = pos[s]
            if at != i:
                ops.append(Instruction.make(Op.SWAP, min(at, i), max(at, i)))
                var other = who[i]
                who[i] = s
                who[at] = other
                pos[s] = i
                pos[other] = at
        _ = cur^
        return ops^
    if kind == BlockKind.LINEAR:
        for _ in range(3 * n * n):
            ops.append(rand_2q(full, rng, Op.CX))
        return ops^
    var depth = 6 * n * n if kind == BlockKind.CLIFFORD else 2 * n * n
    var nrot = 0
    var want_rot = 2 + rng.below(4) if kind == BlockKind.PAULI else 0
    for _ in range(depth):
        var r = rng.below(10)
        if r < 2:
            ops.append(Instruction.make(Op.H, rng.below(n)))
        elif r < 4:
            ops.append(Instruction.make(Op.S, rng.below(n)))
        else:
            ops.append(rand_2q(full, rng, Op.CX))
        if kind == BlockKind.PAULI and nrot < want_rot and rng.below(depth // want_rot + 1) == 0:
            var ax = rng.below(3)
            var k = Op.RZ if ax == 0 else (Op.RX if ax == 1 else Op.RY)
            ops.append(Instruction.make(k, rng.below(n), -1, -1, 0.1 + 0.7 * Float64(nrot)))
            nrot += 1
    while kind == BlockKind.PAULI and nrot < want_rot:
        ops.append(Instruction.make(Op.RZ, rng.below(n), -1, -1, 0.3 + Float64(nrot)))
        nrot += 1
        for _ in range(n):
            ops.append(rand_2q(full, rng, Op.CX))
    return ops^


def verify(kind: Int, n: Int, src: List[Instruction], res: List[Instruction], g: SubGraph) raises -> Bool:
    for o in res:
        if o.kind == Op.SWAP or o.kind == Op.CX or o.kind == Op.CZ:
            if not g.has_edge(min(o.q0, o.q1), max(o.q0, o.q1)):
                return False
    if kind == BlockKind.PERM:
        return Perm.from_ops(res, n) == Perm.from_ops(src, n)
    if kind == BlockKind.LINEAR:
        return GF2Mat.from_ops(res, n) == GF2Mat.from_ops(src, n)
    if kind == BlockKind.CLIFFORD:
        return SympTab.from_ops(res, n) == SympTab.from_ops(src, n)
    var ph: Float64 = 0.0
    return dense_equiv(src, res, n, ph)


struct Acc(Copyable):
    var name: String
    var cost: List[Int]  # -1 = no result
    var depth: List[Int]
    var ns: Int
    var bad: Int

    def __init__(out self, name: String):
        self.name = name
        self.cost = List[Int]()
        self.depth = List[Int]()
        self.ns = 0
        self.bad = 0


def run_one(name: String, kind: Int, n: Int, src: List[Instruction], g: SubGraph, mut acc: Acc,
            mut sess: PolicySession, model: String, seed: Int) raises:
    var t0 = perf_counter_ns()
    var res: List[Instruction]
    if name == "policy_greedy" or name == "policy_beam8":
        var beams = 1 if name == "policy_greedy" else 8
        var searches = 0 if name == "policy_greedy" else 4
        if kind == BlockKind.PERM:
            res = sess.synth_perm(Perm.from_ops(src, n), g, model, beams, searches, seed)
        elif kind == BlockKind.LINEAR:
            res = sess.synth_linear(GF2Mat.from_ops(src, n), g, model, beams, searches, seed)
        elif kind == BlockKind.CLIFFORD:
            res = sess.synth_clifford(SympTab.from_ops(src, n), g, model, beams, searches, seed)
        else:
            res = sess.synth_pauli_network(PauliNet.from_ops(src, n), g, model, beams, searches, seed, src)
    else:
        var method = String(name)
        if kind == BlockKind.PERM:
            res = synth_perm(Perm.from_ops(src, n), g, method, 0, seed)
        elif kind == BlockKind.LINEAR:
            res = synth_linear(GF2Mat.from_ops(src, n), g, method, 0, seed)
        elif kind == BlockKind.CLIFFORD:
            res = synth_clifford(SympTab.from_ops(src, n), g, method, 0, seed)
        else:
            res = synth_pauli_network(PauliNet.from_ops(src, n), g, method, 0, seed, src)
    acc.ns += Int(perf_counter_ns() - t0)
    if len(res) == 0:
        acc.cost.append(-1)
        acc.depth.append(-1)
        return
    if not verify(kind, n, src, res, g):
        acc.bad += 1
        acc.cost.append(-1)
        acc.depth.append(-1)
        return
    acc.cost.append(cost2(res))
    acc.depth.append(ops_depth2q(res, n))


def main() raises:
    var root = String(argv()[1])
    var id = String(argv()[2])
    var ninst = Int(String(argv()[3]))
    var outp = String(argv()[4])
    var m = load_model_file(root + "/" + id)
    var n = m.num_qubits
    var kind = m.kind_code
    var g = SubGraph(n, m.edges)
    var dirs = List[String]()
    dirs.append(root + "/..")
    var sess = PolicySession(dirs)
    var names: List[String] = ["policy_greedy", "policy_beam8", "greedy", "auto"]
    if exact_supported(kind, n):
        names.append("exact")
    var accs = List[Acc]()
    for nm in names:
        accs.append(Acc(nm))
    var rng = Rng(20261014)
    for i in range(ninst):
        var src = make_instance(kind, n, g, rng)
        for k in range(len(names)):
            run_one(names[k], kind, n, src, g, accs[k], sess, id, 7 + i)
    # summary
    var w = JsonWriter()
    w.begin_object()
    w.kv_string("model", id)
    w.kv_int("num_qubits", n)
    w.kv_int("instances", ninst)
    w.key("methods")
    w.begin_object()
    for k in range(len(names)):
        ref a = accs[k]
        var solved = 0
        var csum = 0
        var dsum = 0
        for j in range(ninst):
            if a.cost[j] >= 0:
                solved += 1
                csum += a.cost[j]
                dsum += a.depth[j]
        w.key(a.name)
        w.begin_object()
        w.kv_float("solved", Float64(solved) / Float64(ninst))
        w.kv_float("mean_cost", Float64(csum) / Float64(max(solved, 1)))
        w.kv_float("mean_depth2q", Float64(dsum) / Float64(max(solved, 1)))
        w.kv_float("ms_per_block", Float64(a.ns) / 1e6 / Float64(ninst))
        w.kv_int("verify_failures", a.bad)
        w.end_object()
    w.end_object()
    # head-to-head of the best policy setting vs greedy / auto / exact (instances both solved)
    var pi = 1
    for ref_name in ["greedy", "auto", "exact"]:
        var ri = -1
        for k in range(len(names)):
            if names[k] == ref_name:
                ri = k
        if ri < 0:
            continue
        var win = 0
        var tie = 0
        var both = 0
        var ps = 0
        var rs = 0
        for j in range(ninst):
            var pc = accs[pi].cost[j]
            var rc = accs[ri].cost[j]
            if pc < 0 or rc < 0:
                continue
            both += 1
            ps += pc
            rs += rc
            if pc < rc:
                win += 1
            elif pc == rc:
                tie += 1
        w.key("policy_vs_" + String(ref_name))
        w.begin_object()
        w.kv_int("compared", both)
        w.kv_float("strict_win", Float64(win) / Float64(max(both, 1)))
        w.kv_float("win_or_tie", Float64(win + tie) / Float64(max(both, 1)))
        w.kv_float("cost_ratio", Float64(ps) / Float64(max(rs, 1)))
        w.end_object()
    w.end_object()
    var js = w.finish()
    with open(outp, "w") as f:
        f.write(js)
    print(js)
