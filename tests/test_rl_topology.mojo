"""Tests for qumo.rl.topology and qumo.rl.symmetry (component C3).

Exact checks against golden data produced offline by the Python stack this module replaces:
  * the spec grammar against the 44 configs of envs/gateset_twists.json and the nine shipped manifests;
  * `twists` / `pauli_qubit_perms_and_act_perms` against gateset_twists.json (obs_perms, act_perms, qubit_perms);
  * `twists` against the `symmetries` of the nine shipped synth manifests (order included), `pauli_twists`
    against the Pauli manifest;
  * `pauli_twists` against envs/pauli_twists.json on six topologies (ring:5, ring:4, star:4 have non-involutive
    automorphisms, so a wrong direction of the relation to the env tables fails);
  * the Weisfeiler-Lehman hash of every manifest topology against INDEX.json;
  * automorphism counts, structure, QM-E641 rejections, slugs and connectivity;
  * the 35 ordered edge lists of topologies.json (normalisation, connectivity, automorphism sanity).
"""

from std.collections import List
from qumo.ai.json import JsonValue, parse_json
from qumo.compiler.policy import (
    MODEL_KIND_CLIFFORD,
    MODEL_KIND_LINEAR,
    MODEL_KIND_PAULI,
    MODEL_KIND_PERM,
    gate_code,
    load_model_file,
    model_kind_code,
    read_model_index,
    wl_hash,
)
from qumo.rl.symmetry import (
    Twists,
    automorphisms,
    pauli_qubit_perms_and_act_perms,
    pauli_twists,
    twists,
)
from qumo.rl.testutil import ints, ints2, load_fixture, strings
from qumo.rl.topology import Topology, is_connected, parse_topology, topology_slug

comptime SYNTH_ROOT = "models/ai_transpiler"


def expect(cond: Bool, msg: String) raises:
    if not cond:
        raise Error("test_rl_topology: " + msg)


def same_ints(a: List[Int], b: List[Int]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def same_lists(a: List[List[Int]], b: List[List[Int]]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if not same_ints(a[i], b[i]):
            return False
    return True


def flat_pairs(v: JsonValue) raises -> List[Int]:
    """JSON `[[a, b], ...]` -> flat pairs."""
    var out = List[Int]()
    for i in range(len(v)):
        out.append(v.at(i).at(0).as_int())
        out.append(v.at(i).at(1).as_int())
    return out^


def actions_from_json(v: JsonValue) raises -> List[Int]:
    """JSON `[[name, q0, q1], ...]` -> flat (opcode, q0, q1) triples."""
    var out = List[Int]()
    for i in range(len(v)):
        var code = gate_code(v.at(i).at(0).as_string())
        if code < 0:
            raise Error("unknown gate " + v.at(i).at(0).as_string())
        out.append(code)
        out.append(v.at(i).at(1).as_int())
        out.append(v.at(i).at(2).as_int())
    return out^


def invert(p: List[Int]) -> List[Int]:
    var inv = List[Int](length=len(p), fill=0)
    for i in range(len(p)):
        inv[p[i]] = i
    return inv^


def is_involution(p: List[Int]) -> Bool:
    for i in range(len(p)):
        if p[p[i]] != i:
            return False
    return True


def fails_with_e641(spec: String) -> Bool:
    try:
        _ = parse_topology(spec)
    except e:
        return String(e).startswith("QM-E641")
    return False


# ------------------------------------------------------------------------------------------------ grammar


def test_grammar() raises:
    var t = parse_topology("line:4")
    expect(t.n == 4 and same_ints(t.edges, [0, 1, 1, 2, 2, 3]), "line:4")
    t = parse_topology("ring:4")
    expect(t.n == 4 and same_ints(t.edges, [0, 1, 0, 3, 1, 2, 2, 3]), "ring:4 is sorted, a < b")
    t = parse_topology("grid:2x3")
    expect(t.n == 6 and same_ints(t.edges, [0, 1, 0, 3, 1, 2, 1, 4, 2, 5, 3, 4, 4, 5]), "grid:2x3")
    t = parse_topology("grid:2X3")
    expect(t.n == 6 and t.num_edges() == 7, "grid accepts an upper case X")
    t = parse_topology("king:2x2")
    expect(t.n == 4 and same_ints(t.edges, [0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3]), "king:2x2 is complete")
    t = parse_topology("king:3x4")
    expect(t.n == 12 and t.num_edges() == 17 + 12, "king:3x4 has 17 lattice and 12 diagonal edges")
    t = parse_topology("complete:5")
    expect(t.n == 5 and t.num_edges() == 10, "complete:5")
    t = parse_topology("star:5")
    expect(t.n == 5 and same_ints(t.edges, [0, 1, 0, 2, 0, 3, 0, 4]), "star:5")
    t = parse_topology("heavyhex_frag:H7")
    expect(t.n == 7 and same_ints(t.edges, [0, 1, 1, 2, 1, 3, 3, 5, 4, 5, 5, 6]), "H7")
    t = parse_topology("heavyhex_frag:T5")
    expect(t.n == 5 and same_ints(t.edges, [0, 1, 1, 2, 2, 3, 2, 4]), "T5")
    t = parse_topology("heavyhex_frag:cell12")
    expect(t.n == 12 and t.num_edges() == 12 and is_connected(t.n, t.edges), "cell12 is a 12-cycle")
    t = parse_topology("heavyhex_frag:ring10s2")
    expect(t.n == 12 and t.num_edges() == 12 and same_ints([t.edges[4], t.edges[5], t.edges[16], t.edges[17]], [0, 10, 5, 11]), "ring10s2")
    t = parse_topology("edges:0-1,1-2,1-3")
    expect(t.n == 4 and same_ints(t.edges, [0, 1, 1, 2, 1, 3]), "edges")
    t = parse_topology("edges:3-1, 2-1 ,1-0,0-1,2-2")
    expect(t.n == 4 and same_ints(t.edges, [0, 1, 1, 2, 1, 3]), "edges: unordered, duplicated, self loop, spaces")
    # degenerate but accepted, exactly as the Python helper
    t = parse_topology("line:1")
    expect(t.n == 1 and t.num_edges() == 0, "line:1")
    t = parse_topology("ring:2")
    expect(t.n == 2 and same_ints(t.edges, [0, 1]), "ring:2 has one edge")
    t = parse_topology("ring:1")
    expect(t.n == 1 and t.num_edges() == 0, "ring:1")
    # default-constructed Topology is empty
    var e = Topology()
    expect(e.n == 0 and len(e.edges) == 0, "Topology()")


def test_bad_specs() raises:
    var bad = List[String]()
    bad.append("")
    bad.append("line")
    bad.append("line:")
    bad.append("line:x")
    bad.append("line:0")
    bad.append("line:-3")
    bad.append("line:4.5")
    bad.append("line:99999")
    bad.append("ring:abc")
    bad.append("grid:3")
    bad.append("grid:2y3")
    bad.append("grid:0x3")
    bad.append("grid:2x3x4")
    bad.append("grid:100x100")
    bad.append("king:x")
    bad.append("complete:0")
    bad.append("complete:5000")
    bad.append("star:-1")
    bad.append("heavyhex_frag:nope")
    bad.append("heavyhex_frag:")
    bad.append("edges:")
    bad.append("edges:0")
    bad.append("edges:0-1-2")
    bad.append("edges:0-1,")
    bad.append("edges:a-b")
    bad.append("edges:2-2")
    bad.append("edges:0-9999")
    bad.append("torus:4")
    bad.append("LINE:4")
    for i in range(len(bad)):
        expect(fails_with_e641(bad[i]), "spec '" + bad[i] + "' must raise QM-E641")
    var failed = False
    try:
        _ = topology_slug("nocolon")
    except e:
        failed = String(e).startswith("QM-E641")
    expect(failed, "topology_slug of a spec without ':' raises QM-E641")


def test_slug() raises:
    expect(topology_slug("grid:2x3") == "grid2x3", "slug grid")
    expect(topology_slug("grid:2X3") == "grid2x3", "slug lower-cases the argument")
    expect(topology_slug("line:4") == "line4", "slug line")
    expect(topology_slug("king:3x4") == "king3x4", "slug king")
    expect(topology_slug("heavyhex_frag:ring10s2") == "hhring10s2", "slug heavy-hex")
    expect(topology_slug("heavyhex_frag:H7") == "hhH7", "slug H7")
    var a = topology_slug("edges:0-1,1-2,1-3")
    var b = topology_slug("edges:3-1,1-0,2-1,1-2")  # same normalised graph
    var c = topology_slug("edges:0-1,1-2,2-3")
    expect(a.byte_length() == 13 and a.startswith("edges"), "edges slug is 'edges' + 8 hex digits: " + a)
    expect(a == b, "edges slug depends only on the normalised graph")
    expect(a != c, "edges slug differs for another graph")
    var bs = a.as_bytes()
    for i in range(5, 13):
        var ch = Int(bs[i])
        expect((ch >= ord("0") and ch <= ord("9")) or (ch >= ord("a") and ch <= ord("f")), "edges slug is lower hex")


def test_connectivity() raises:
    expect(is_connected(4, [0, 1, 1, 2, 2, 3]), "path")
    expect(not is_connected(4, [0, 1, 2, 3]), "two components")
    expect(not is_connected(3, [0, 1]), "isolated vertex")
    expect(is_connected(1, List[Int]()), "single vertex")
    expect(not is_connected(2, List[Int]()), "two isolated vertices")
    expect(not is_connected(0, List[Int]()), "empty graph, as in the Python helper")
    expect(not is_connected(3, [0, 1, 1, 7]), "endpoint out of range")
    expect(is_connected(5, [4, 3, 3, 2, 2, 1, 1, 0]), "reverse-listed path")
    var specs = List[String]()
    specs.append("line:7")
    specs.append("ring:6")
    specs.append("grid:3x4")
    specs.append("king:3x3")
    specs.append("complete:6")
    specs.append("star:6")
    specs.append("heavyhex_frag:H7")
    specs.append("heavyhex_frag:T5")
    specs.append("heavyhex_frag:cell12")
    specs.append("heavyhex_frag:ring10s2")
    for i in range(len(specs)):
        var t = parse_topology(specs[i])
        expect(is_connected(t.n, t.edges), specs[i] + " is connected")


# ------------------------------------------------------------------------------------------------ automorphisms


def check_automorphisms(n: Int, edges: List[Int], what: String) raises -> Int:
    """Structural checks on the automorphism list; returns its length."""
    var autos = automorphisms(n, edges)
    var ne = len(edges) // 2
    var present = List[Bool](length=n * n, fill=False)
    for e in range(ne):
        present[edges[2 * e] * n + edges[2 * e + 1]] = True
        present[edges[2 * e + 1] * n + edges[2 * e]] = True
    expect(len(autos) >= 1, what + ": at least the identity")
    for p in range(len(autos)):
        var perm = autos[p].copy()
        expect(len(perm) == n, what + ": length")
        var seen = List[Bool](length=n, fill=False)
        for i in range(n):
            expect(perm[i] >= 0 and perm[i] < n and not seen[perm[i]], what + ": a permutation")
            seen[perm[i]] = True
        for e in range(ne):
            expect(present[perm[edges[2 * e]] * n + perm[edges[2 * e + 1]]], what + ": preserves edges")
        if p > 0:
            # strictly increasing in lexicographic order (this also proves uniqueness)
            var prev = autos[p - 1].copy()
            var less = False
            for i in range(n):
                if prev[i] != perm[i]:
                    less = prev[i] < perm[i]
                    break
            expect(less, what + ": sorted lexicographically")
    for i in range(n):
        expect(autos[0][i] == i, what + ": identity first")
    return len(autos)


def count_for(spec: String) raises -> Int:
    var t = parse_topology(spec)
    return check_automorphisms(t.n, t.edges, spec)


def test_automorphism_counts() raises:
    expect(count_for("line:2") == 2, "line:2")
    expect(count_for("line:3") == 2, "line:3")
    expect(count_for("line:4") == 2, "line:4")
    expect(count_for("line:6") == 2, "line:6")
    expect(count_for("line:9") == 2, "line:n -> 2")
    expect(count_for("ring:4") == 8, "ring:4")
    expect(count_for("ring:5") == 10, "ring:5")
    expect(count_for("ring:6") == 12, "ring:6")
    expect(count_for("ring:12") == 24, "ring:12")
    expect(count_for("star:4") == 6, "star:4")
    expect(count_for("star:5") == 24, "star:5")
    expect(count_for("complete:4") == 24, "complete:4")
    expect(count_for("complete:5") == 120, "complete:5")
    expect(count_for("grid:2x3") == 4, "grid:2x3")
    expect(count_for("grid:3x3") == 8, "grid:3x3")
    expect(count_for("grid:3x4") == 4, "grid:3x4")
    expect(count_for("grid:2x2") == 8, "grid:2x2")
    expect(count_for("king:3x4") == 4, "king:3x4")
    expect(count_for("heavyhex_frag:H7") == 8, "H7")
    expect(count_for("heavyhex_frag:T5") == 2, "T5")
    expect(count_for("heavyhex_frag:cell12") == 24, "cell12")
    expect(count_for("heavyhex_frag:ring10s2") == 4, "ring10s2")
    expect(count_for("edges:0-1,1-2,1-3") == 6, "edges: star with 3 leaves")
    expect(count_for("line:1") == 1, "single vertex")
    # no edge at all: every permutation (n!)
    expect(check_automorphisms(4, List[Int](), "edgeless 4") == 24, "edgeless graph has n! automorphisms")
    expect(check_automorphisms(1, List[Int](), "edgeless 1") == 1, "edgeless n=1")
    expect(len(automorphisms(0, List[Int]())) == 0, "n=0 has none")
    # input hygiene: self loops, duplicates, out-of-range endpoints and reversed pairs do not matter
    var messy: List[Int] = [2, 1, 1, 0, 0, 1, 1, 1, 5, 9, 1, 2]
    var cleaned = automorphisms(3, messy)
    var plain = automorphisms(3, [0, 1, 1, 2])
    expect(same_lists(cleaned, plain), "messy edge list equals the clean path")
    # explicit lexicographic order on a known case: ring:4
    var r4 = automorphisms(4, parse_topology("ring:4").edges)
    var want = List[List[Int]]()
    want.append([0, 1, 2, 3])
    want.append([0, 3, 2, 1])
    want.append([1, 0, 3, 2])
    want.append([1, 2, 3, 0])
    want.append([2, 1, 0, 3])
    want.append([2, 3, 0, 1])
    want.append([3, 0, 1, 2])
    want.append([3, 2, 1, 0])
    expect(same_lists(r4, want), "ring:4 automorphism list")


def brute_force_count(n: Int, edges: List[Int]) -> Int:
    """Number of automorphisms by testing all n! permutations (Heap-free recursion over a used mask)."""
    var present = List[Bool](length=n * n, fill=False)
    for e in range(len(edges) // 2):
        present[edges[2 * e] * n + edges[2 * e + 1]] = True
        present[edges[2 * e + 1] * n + edges[2 * e]] = True
    var perm = List[Int](length=n, fill=0)
    for i in range(n):
        perm[i] = i
    var total = 0
    # iterate permutations in lexicographic order with next_permutation
    while True:
        var ok = True
        for a in range(n):
            for b in range(n):
                if present[a * n + b] != present[perm[a] * n + perm[b]]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            total += 1
        # next permutation
        var i = n - 2
        while i >= 0 and perm[i] > perm[i + 1]:
            i -= 1
        if i < 0:
            break
        var j = n - 1
        while perm[j] < perm[i]:
            j -= 1
        var tmp = perm[i]
        perm[i] = perm[j]
        perm[j] = tmp
        var lo = i + 1
        var hi = n - 1
        while lo < hi:
            var t2 = perm[lo]
            perm[lo] = perm[hi]
            perm[hi] = t2
            lo += 1
            hi -= 1
    return total


def test_against_brute_force() raises:
    """The backtracking enumeration is complete: it matches an exhaustive n! search on pseudo-random graphs."""
    var state: UInt64 = 0x9E3779B97F4A7C15
    var checked = 0
    var nontrivial = 0
    for trial in range(160):
        var n = 3 + trial % 5  # 3..7
        var edges = List[Int]()
        # xorshift64 edge selection with a varying density
        var dens = 2 + (trial // 5) % 5
        for a in range(n):
            for b in range(a + 1, n):
                state ^= state << 13
                state ^= state >> 7
                state ^= state << 17
                if Int(state % 8) < dens:
                    edges.append(a)
                    edges.append(b)
        var fast = automorphisms(n, edges)
        var slow = brute_force_count(n, edges)
        expect(len(fast) == slow, "random graph trial " + String(trial) + ": " + String(len(fast)) + " vs " + String(slow))
        if slow > 1:
            nontrivial += 1
        checked += 1
    expect(checked == 160 and nontrivial > 40, "enough non-trivial random graphs: " + String(nontrivial))


# ------------------------------------------------------------------------------------------------ gateset_twists.json


def test_gateset_twists() raises:
    var g = load_fixture("envs/gateset_twists.json")
    expect(len(g) == 44, "44 configs")
    var plain_checked = 0
    var pauli_checked = 0
    for ci in range(len(g)):
        var c = g.at(ci)
        var what = c.get("kind").as_string() + " " + c.get("topology").as_string()
        var kind = model_kind_code(c.get("kind").as_string())
        expect(kind >= 0, what + ": kind")
        var n = c.get("num_qubits").as_int()
        # the spec grammar reproduces the golden qubit count and edge list
        var topo = parse_topology(c.get("topology").as_string())
        expect(topo.n == n, what + ": qubit count")
        expect(same_ints(topo.edges, flat_pairs(c.get("edges"))), what + ": edges")
        var actions = actions_from_json(c.get("gateset"))
        expect(len(actions) // 3 == c.get("num_actions").as_int(), what + ": num_actions")
        # the coupling graph recovered from the gateset equals the topology, so counts match
        if kind == MODEL_KIND_PAULI:
            var t = twists(kind, n, actions)
            expect(len(t.obs_perms) == 0 and len(t.act_perms) == 0, what + ": Pauli twists() is empty")
            var tab = pauli_qubit_perms_and_act_perms(n, actions)
            expect(same_lists(tab.obs_perms, ints2(c.get("qubit_perms"))), what + ": qubit_perms")
            expect(same_lists(tab.act_perms, ints2(c.get("act_perms"))), what + ": pauli act_perms")
            expect(len(tab.obs_perms) == len(automorphisms(n, topo.edges)), what + ": all automorphisms kept")
            pauli_checked += 1
        else:
            var t = twists(kind, n, actions)
            expect(same_lists(t.obs_perms, ints2(c.get("obs_perms"))), what + ": obs_perms")
            expect(same_lists(t.act_perms, ints2(c.get("act_perms"))), what + ": act_perms")
            plain_checked += 1
    expect(plain_checked > 0 and pauli_checked > 0 and plain_checked + pauli_checked == 44, "kinds covered")
    print("  gateset_twists: plain", plain_checked, "pauli", pauli_checked)


def test_twists_synthetic() raises:
    """Cases the goldens do not reach: a gateset that lacks the images of some automorphisms."""
    # a single SWAP on two qubits: identity and the exchange
    var one_swap = List[Int]()
    one_swap.append(20)
    one_swap.append(0)
    one_swap.append(1)
    var t = twists(MODEL_KIND_PERM, 2, one_swap)
    expect(len(t.act_perms) == 2, "single SWAP on 2 qubits: identity and the exchange")
    # three qubits, edges {0-1, 1-2} but only SWAP(0,1) and SWAP(1,2) given -> symmetric: 2 symmetries
    var two = List[Int]()
    two.append(20)
    two.append(0)
    two.append(1)
    two.append(20)
    two.append(1)
    two.append(2)
    t = twists(MODEL_KIND_PERM, 3, two)
    expect(same_lists(t.act_perms, [[0, 1], [1, 0]]), "line:3 swaps map to each other under reversal")
    # a gateset with an asymmetric extra action loses the reversal (image missing) but keeps the identity
    var asym = List[Int]()
    asym.append(16)  # CX 0 1
    asym.append(0)
    asym.append(1)
    asym.append(16)  # CX 1 0
    asym.append(1)
    asym.append(0)
    asym.append(16)  # CX 1 2
    asym.append(1)
    asym.append(2)  # CX 2 1 missing
    t = twists(MODEL_KIND_LINEAR, 3, asym)
    expect(len(t.act_perms) == 1 and same_ints(t.act_perms[0], [0, 1, 2]), "reversal dropped, identity kept")
    expect(same_ints(t.obs_perms[0], [0, 1, 2, 3, 4, 5, 6, 7, 8]), "identity observation map")
    # SWAP keys are unordered, CX keys are ordered: a reversed SWAP listing still maps onto itself
    var rev = List[Int]()
    rev.append(20)
    rev.append(1)
    rev.append(0)
    rev.append(20)
    rev.append(2)
    rev.append(1)
    t = twists(MODEL_KIND_PERM, 3, rev)
    expect(same_lists(t.act_perms, [[0, 1], [1, 0]]), "unordered SWAP keys")
    # routing and unknown kinds produce nothing
    expect(len(twists(4, 3, two).act_perms) == 0, "routing kind has no twists")
    expect(len(twists(-1, 3, two).act_perms) == 0, "unknown kind has no twists")
    # clifford observation map: m(i) = perm[i] for i < n else n + perm[i - n]
    var cl = List[Int]()
    cl.append(4)  # H 0
    cl.append(0)
    cl.append(-1)
    cl.append(4)  # H 1
    cl.append(1)
    cl.append(-1)
    cl.append(16)  # CX 0 1
    cl.append(0)
    cl.append(1)
    cl.append(16)  # CX 1 0
    cl.append(1)
    cl.append(0)
    t = twists(MODEL_KIND_CLIFFORD, 2, cl)
    expect(len(t.obs_perms) == 2 and len(t.obs_perms[1]) == 16, "clifford 2 qubits")
    # perm = (1, 0): m = [1, 0, 3, 2]; entry r*4 + c -> m[r]*4 + m[c]
    expect(t.obs_perms[1][0] == 1 * 4 + 1 and t.obs_perms[1][1] == 1 * 4 + 0 and t.obs_perms[1][4] == 0 * 4 + 1, "m()")
    expect(t.obs_perms[1][2] == 1 * 4 + 3 and t.obs_perms[1][10] == 3 * 4 + 3, "m() on the destabiliser block")
    expect(same_ints(t.act_perms[1], [1, 0, 3, 2]), "clifford act_perm swaps H0<->H1, CX01<->CX10")


# ------------------------------------------------------------------------------------------------ manifests


def check_manifest(dir_name: String, expect_pauli: Bool) raises:
    var m = load_model_file(SYNTH_ROOT + "/synth/" + dir_name)
    var man = parse_json(m.manifest)
    var spec = man.get("training").get("topology_spec").as_string()
    var topo = parse_topology(spec)
    expect(topo.n == m.num_qubits, dir_name + ": qubit count from topology_spec")
    expect(same_ints(topo.edges, m.edges), dir_name + ": edges from topology_spec equal the manifest")
    expect(is_connected(topo.n, topo.edges), dir_name + ": connected")
    # the coupling edges implied by the gateset equal the topology
    if m.kind_code == MODEL_KIND_PAULI:
        expect(expect_pauli, dir_name + ": unexpected Pauli")
        var t = pauli_twists(m.num_qubits, m.actions, m.max_rotations)
        expect(len(t.act_perms) > 0, dir_name + ": pauli_twists nonempty")
        expect(same_lists(t.obs_perms, m.obs_perms), dir_name + ": pauli_twists obs_perms equal the manifest")
        expect(same_lists(t.act_perms, m.act_perms), dir_name + ": pauli_twists act_perms equal the manifest")
        # the plain-kind entry point yields nothing for Pauli
        expect(len(twists(m.kind_code, m.num_qubits, m.actions).act_perms) == 0, dir_name + ": twists() empty for Pauli")
    else:
        var t = twists(m.kind_code, m.num_qubits, m.actions)
        expect(len(t.act_perms) > 0, dir_name + ": twists nonempty")
        expect(same_lists(t.obs_perms, m.obs_perms), dir_name + ": obs_perms equal the manifest")
        expect(same_lists(t.act_perms, m.act_perms), dir_name + ": act_perms equal the manifest")
    # every manifest permutation is a table of the right size
    for p in range(len(m.act_perms)):
        expect(len(m.act_perms[p]) == m.num_actions(), dir_name + ": act perm length")


def test_manifests() raises:
    var ids = List[String]()
    ids.append("lf_grid2x3")
    ids.append("lf_hhH7")
    ids.append("lf_line6")
    ids.append("lf_ring6")
    ids.append("pauli_line4")
    ids.append("perm_grid3x3")
    ids.append("perm_grid3x4")
    ids.append("perm_hhring10s2")
    ids.append("perm_king3x4")
    var pauli_seen = 0
    for i in range(len(ids)):
        check_manifest(ids[i], ids[i].startswith("pauli"))
        if ids[i].startswith("pauli"):
            pauli_seen += 1
    expect(pauli_seen == 1, "one Pauli manifest")

    # WL hash of the parsed topology equals the INDEX row, for every model
    var rows = read_model_index(SYNTH_ROOT, "synth")
    expect(len(rows) == 9, "INDEX has 9 rows")
    for r in range(len(rows)):
        var m = load_model_file(rows[r].path)
        var man = parse_json(m.manifest)
        var topo = parse_topology(man.get("training").get("topology_spec").as_string())
        var h = wl_hash(topo.n, topo.edges)
        expect(h == rows[r].wl_hash, rows[r].id + ": wl_hash " + h + " vs INDEX " + rows[r].wl_hash)
        expect(same_ints(topo.edges, rows[r].edges), rows[r].id + ": INDEX edges")
        expect(topo.n == rows[r].num_qubits, rows[r].id + ": INDEX qubits")
        # the slug is usable as a directory name and distinct per spec
        var slug = topology_slug(man.get("training").get("topology_spec").as_string())
        expect(slug.byte_length() > 0, rows[r].id + ": slug")


# ------------------------------------------------------------------------------------------------ pauli_twists.json


def test_pauli_twists_fixture() raises:
    var f = load_fixture("envs/pauli_twists.json")
    var cases = f.get("cases")
    expect(len(cases) == 6, "six topologies")
    var non_involutive_total = 0
    var direction_discriminated = 0
    for ci in range(len(cases)):
        var c = cases.at(ci)
        var spec = c.get("topology").as_string()
        var n = c.get("num_qubits").as_int()
        var max_rot = c.get("max_rotations").as_int()
        var actions = actions_from_json(c.get("actions"))
        var topo = parse_topology(spec)
        expect(topo.n == n, spec + ": qubits")
        expect(same_ints(topo.edges, flat_pairs(c.get("edges"))), spec + ": edges")
        var autos = automorphisms(n, topo.edges)
        expect(same_lists(autos, ints2(c.get("automorphisms"))), spec + ": automorphism list")
        var non_inv = 0
        for p in range(len(autos)):
            if not is_involution(autos[p]):
                non_inv += 1
        expect(non_inv == c.get("non_involutive_automorphisms").as_int(), spec + ": non-involutive count")
        non_involutive_total += non_inv

        var t = pauli_twists(n, actions, max_rot)
        var want_obs = ints2(c.get("obs_perms"))
        var want_act = ints2(c.get("act_perms"))
        expect(len(want_obs) == len(autos) and len(want_act) == len(autos), spec + ": fixture sizes")
        expect(len(want_obs[0]) == c.get("obs_size").as_int(), spec + ": obs_size")
        expect(same_lists(t.obs_perms, want_obs), spec + ": pauli_twists obs_perms")
        expect(same_lists(t.act_perms, want_act), spec + ": pauli_twists act_perms")

        # relation to the env tables: act_perms is the inverse of the env's direct table, obs_perms is built from the
        # inverse of [perm, n + perm]
        var env = pauli_qubit_perms_and_act_perms(n, actions)
        expect(len(env.act_perms) == len(autos), spec + ": env tables for every automorphism")
        var differs = False
        for p in range(len(autos)):
            expect(same_ints(t.act_perms[p], invert(env.act_perms[p])), spec + ": export act_perm = inverse of env act_perm")
            expect(same_ints(env.obs_perms[p], autos[p]), spec + ": env qubit perm is the automorphism")
            if not same_ints(env.act_perms[p], t.act_perms[p]):
                differs = True
        if non_inv > 0:
            # a wrong direction (using the env table directly) would be wrong here
            expect(differs, spec + ": direct and inverse tables must differ when an automorphism is non-involutive")
            direction_discriminated += 1
        else:
            expect(not differs, spec + ": involutions only -> direct and inverse tables coincide")
    expect(non_involutive_total == 4 + 2 + 2 + 0 + 2 + 0, "non-involutive total over the six topologies")
    expect(direction_discriminated == 4, "four topologies discriminate the direction (ring:5, ring:4, star:4, H7)")


# ------------------------------------------------------------------------------------------------ topologies.json


def test_topologies_json() raises:
    var f = load_fixture("topologies.json")
    var total = 0
    var groups = List[String]()
    groups.append("train")
    groups.append("eval")
    var expected = List[Int]()
    expected.append(25)
    expected.append(10)
    for g in range(len(groups)):
        var arr = f.get(groups[g])
        expect(len(arr) == expected[g], groups[g] + ": count")
        for i in range(len(arr)):
            var t = arr.at(i)
            var name = t.get("name").as_string()
            var n = t.get("n").as_int()
            var raw = flat_pairs(t.get("edges"))
            # same graph through the explicit-edge grammar: sorted, a < b, unique; n = largest index + 1
            var text = String("edges:")
            for e in range(len(raw) // 2):
                if e > 0:
                    text += ","
                text += String(raw[2 * e]) + "-" + String(raw[2 * e + 1])
            var topo = parse_topology(text)
            expect(topo.n == n, name + ": n")
            expect(topo.num_edges() == len(raw) // 2, name + ": no duplicate or loop in the golden list")
            expect(is_connected(topo.n, topo.edges), name + ": connected")
            # every edge of the golden list is in the normalised list
            var key = List[Int]()
            for e in range(topo.num_edges()):
                key.append(topo.edges[2 * e] * 4096 + topo.edges[2 * e + 1])
            for e in range(len(raw) // 2):
                var a = min(raw[2 * e], raw[2 * e + 1])
                var b = max(raw[2 * e], raw[2 * e + 1])
                var found = False
                for k in range(len(key)):
                    if key[k] == a * 4096 + b:
                        found = True
                        break
                expect(found, name + ": edge in normalised list")
            total += 1
            # structure-aware spec checks for the families with a direct grammar
            var fam = t.get("kind").as_string()
            if fam == "line":
                var s = parse_topology("line:" + String(n))
                expect(same_ints(s.edges, topo.edges), name + ": line grammar")
            if fam == "ring":
                var s = parse_topology("ring:" + String(n))
                expect(same_ints(s.edges, topo.edges), name + ": ring grammar")
            if fam == "grid":
                var size = t.get("size")
                var s = parse_topology("grid:" + String(size.at(0).as_int()) + "x" + String(size.at(1).as_int()))
                expect(same_ints(s.edges, topo.edges), name + ": grid grammar")
    expect(total == 35, "35 edge lists")

    # automorphism sanity on a few of the larger graphs (identity first, valid, sorted, known orders)
    var by_name = List[String]()
    var counts = List[Int]()
    by_name.append("line8")
    counts.append(2)
    by_name.append("line20")
    counts.append(2)
    by_name.append("ring8")
    counts.append(16)
    by_name.append("ring20")
    counts.append(40)
    by_name.append("grid3x3")
    counts.append(8)
    by_name.append("grid4x4")
    counts.append(8)
    by_name.append("grid3x5")
    counts.append(4)
    by_name.append("tree15")
    counts.append(128)
    by_name.append("heavy_hex19")
    counts.append(2)
    var tr = f.get("train")
    for k in range(len(by_name)):
        var found = False
        for i in range(len(tr)):
            if tr.at(i).get("name").as_string() == by_name[k]:
                var edges = flat_pairs(tr.at(i).get("edges"))
                var got = check_automorphisms(tr.at(i).get("n").as_int(), edges, by_name[k])
                expect(got == counts[k], by_name[k] + ": automorphism count " + String(got) + " != " + String(counts[k]))
                found = True
                break
        expect(found, by_name[k] + " present in topologies.json")


def main() raises:
    test_grammar()
    test_bad_specs()
    test_slug()
    test_connectivity()
    test_automorphism_counts()
    test_against_brute_force()
    test_gateset_twists()
    test_twists_synthetic()
    test_manifests()
    test_pauli_twists_fixture()
    test_topologies_json()
    print("test_rl_topology OK")
