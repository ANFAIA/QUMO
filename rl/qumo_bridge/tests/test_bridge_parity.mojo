"""Bridge test: the pieces qrl copied from QUMO still behave exactly like QUMO's originals.

qrl imports nothing from `qumo.*`, so it carries copies of what training shares with the runtime. Each must stay
identical to its QUMO source, or fixtures, checkpoints and exported models drift:

  * `qrl.rng.Rng` vs `qumo.sim.rng.Rng`: the same xoshiro256** stream (next_u64, uniform, below, normal, split,
    bernoulli_mask) for many seeds;
  * `qrl.gates` vs `qumo.ir.gates`: every opcode value and name;
  * `qrl.model` vs `qumo.compiler.policy`: kind codes and names, gate codes, FNV-1a 64 (bytes and words), hex16, the
    WL hash on random graphs, f16 -> f32 on all 65,536 bit patterns, format strings;
  * `qrl.env_core.PolicyEnv` vs `qumo.compiler.policy_synth.PolicyEnv`: random action walks from training resets
    of all four kinds on several topologies (states, emitted gates, observations, solved flags, keys, inverses);
  * `qrl.json` vs `qumo.core.json`: pretty printing and compact re-serialisation of every JSON fixture, number and
    string formatting.

Run from the QUMO root with `-I . -I rl` (rl/Makefile `test-bridge`).
"""

from std.os import listdir
from std.os.path import isdir
from std.memory import bitcast
from qrl.rng import Rng as QRng
from qrl.gates import Op as QOp, op_name as q_op_name
from qrl.json import parse_json as q_parse, pretty_json as q_pretty, json_number as q_number, json_quote as q_quote
from qrl.model import (
    ModelFile as QModelFile,
    model_kind_code as q_kind_code,
    model_kind_name as q_kind_name,
    gate_code as q_gate_code,
    fnv1a64_bytes as q_fnv_bytes,
    fnv_words as q_fnv_words,
    hex16 as q_hex16,
    wl_hash as q_wl_hash,
    f16_to_f32 as q_f16,
    MODEL_FORMAT_SYNTH as Q_FMT_SYNTH,
    MODEL_FORMAT_ROUTE as Q_FMT_ROUTE,
    MODEL_KIND_PAULI,
)
from qrl.env_core import PolicyEnv as QEnv
from qrl.gateset import EnvConfig, build_gateset, shell_model
from qrl.synth_env import TrainEnv
from qrl.topology import parse_topology
from qrl.testutil import fixture_root, read_text
from qumo.sim.rng import Rng as URng
from qumo.ir.gates import Op as UOp, op_name as u_op_name
from qumo.core.json import parse_json as u_parse, pretty_json as u_pretty, json_number as u_number, json_quote as u_quote
from qumo.compiler.policy import (
    ModelFile as UModelFile,
    model_kind_code as u_kind_code,
    model_kind_name as u_kind_name,
    gate_code as u_gate_code,
    fnv1a64_bytes as u_fnv_bytes,
    fnv_words as u_fnv_words,
    hex16 as u_hex16,
    wl_hash as u_wl_hash,
    f16_to_f32 as u_f16,
    MODEL_FORMAT_SYNTH as U_FMT_SYNTH,
    MODEL_FORMAT_ROUTE as U_FMT_ROUTE,
)
from qumo.compiler.policy_synth import PolicyEnv as UEnv


def check(cond: Bool, msg: String) raises:
    if not cond:
        raise Error("test_bridge_parity: " + msg)


def _same(a: List[UInt64], b: List[UInt64]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def _same_ints(a: List[Int], b: List[Int]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


# ------------------------------------------------------------------------------------------------ Rng


def test_rng() raises:
    var seeds: List[UInt64] = [0, 1, 2, 7, 42, 424242, 0xFFFFFFFFFFFFFFFF, 0x9E3779B97F4A7C15, 20261014]
    var draws = 0
    for s in seeds:
        var a = QRng(s)
        var b = URng(s)
        for _ in range(500):
            check(a.next_u64() == b.next_u64(), "next_u64 stream, seed " + String(s))
        for _ in range(200):
            check(bitcast[DType.uint64](a.uniform()) == bitcast[DType.uint64](b.uniform()), "uniform, seed " + String(s))
        for k in range(1, 200):
            check(a.below(k) == b.below(k), "below, seed " + String(s))
        for _ in range(100):
            check(bitcast[DType.uint64](a.normal()) == bitcast[DType.uint64](b.normal()), "normal, seed " + String(s))
        for p in [0.0, 0.01, 0.3, 0.5, 0.97, 1.0]:
            check(a.bernoulli_mask(p) == b.bernoulli_mask(p), "bernoulli_mask, seed " + String(s))
        for k in range(5):
            var sa = a.split(k)
            var sb = b.split(k)
            for _ in range(20):
                check(sa.next_u64() == sb.next_u64(), "split stream, seed " + String(s))
        var na = a.normal_list(17, 0.5)
        var nb = b.normal_list(17, 0.5)
        for i in range(17):
            check(bitcast[DType.uint64](na[i]) == bitcast[DType.uint64](nb[i]), "normal_list")
        draws += 500 + 200 + 199 + 100 + 6 + 100 + 17
    print("  Rng: " + String(len(seeds)) + " seeds, " + String(draws) + " draws identical")


# ------------------------------------------------------------------------------------------------ gates and model helpers


def test_gates_and_model_helpers() raises:
    var qv: List[Int] = [QOp.I, QOp.X, QOp.Y, QOp.Z, QOp.H, QOp.S, QOp.SDG, QOp.T, QOp.TDG, QOp.SX, QOp.SXDG, QOp.CX, QOp.CZ, QOp.SWAP]
    var uv: List[Int] = [UOp.I, UOp.X, UOp.Y, UOp.Z, UOp.H, UOp.S, UOp.SDG, UOp.T, UOp.TDG, UOp.SX, UOp.SXDG, UOp.CX, UOp.CZ, UOp.SWAP]
    for i in range(len(qv)):
        check(qv[i] == uv[i], "opcode " + String(i) + ": " + String(qv[i]) + " vs " + String(uv[i]))
        check(q_op_name(qv[i]) == u_op_name(uv[i]), "op_name " + u_op_name(uv[i]))
    check(String(Q_FMT_SYNTH) == String(U_FMT_SYNTH) and String(Q_FMT_ROUTE) == String(U_FMT_ROUTE), "format strings")
    for name in ["clifford", "linear_function", "permutation", "pauli_network", "routing", "nope"]:
        check(q_kind_code(name) == u_kind_code(name), "kind code " + String(name))
    for c in range(-1, 6):
        check(q_kind_name(c) == u_kind_name(c), "kind name " + String(c))
    for g in ["SWAP", "cx", "CZ", "h", "S", "sdg", "SX", "SXDG", "T", "rz", ""]:
        check(q_gate_code(g) == u_gate_code(g), "gate code " + String(g))
    for h in range(65536):
        var a = bitcast[DType.uint32](q_f16(UInt16(h)))
        var b = bitcast[DType.uint32](u_f16(UInt16(h)))
        check(a == b, "f16_to_f32 of pattern " + String(h))
    var r = QRng(99)
    for t in range(200):
        var bytes = List[UInt8]()
        var words = List[UInt64]()
        for _ in range(t):
            bytes.append(UInt8(r.below(256)))
            words.append(r.next_u64())
        check(q_fnv_bytes(bytes) == u_fnv_bytes(bytes), "fnv1a64_bytes")
        check(q_fnv_words(words) == u_fnv_words(words), "fnv_words")
        check(q_hex16(q_fnv_words(words)) == u_hex16(u_fnv_words(words)), "hex16")
    for t in range(300):
        var n = 2 + r.below(12)
        var e = List[Int]()
        for a in range(n):
            for b in range(a + 1, n):
                if r.below(3) == 0:
                    e.append(a)
                    e.append(b)
        check(q_wl_hash(n, e) == u_wl_hash(n, e), "wl_hash on random graph " + String(t))
        check(q_wl_hash(n, e, 1) == u_wl_hash(n, e, 1), "wl_hash, 1 iteration")
    print("  gates, kind / gate codes, f16 (65,536 patterns), FNV-1a, hex16, WL hash (300 graphs) identical")


# ------------------------------------------------------------------------------------------------ PolicyEnv


def _qumo_shell(m: QModelFile) -> UModelFile:
    var u = UModelFile()
    u.kind_code = m.kind_code
    u.kind = m.kind
    u.num_qubits = m.num_qubits
    u.actions = m.actions.copy()
    u.max_depth = m.max_depth
    u.max_rotations = m.max_rotations
    u.obs_size = m.obs_size
    return u^


def _walk(kind_name: String, topo: String, difficulty: Int, seed: UInt64, steps: Int) raises -> Int:
    var t = parse_topology(topo)
    var cfg = EnvConfig()
    cfg.kind = q_kind_code(kind_name)
    cfg.n = t.n
    cfg.actions = build_gateset(cfg.kind, t.n, t.edges)
    cfg.difficulty = difficulty
    var env = TrainEnv(cfg, seed)
    env.reseed(seed)
    env.reset()
    var st = env.state_words()
    var qm = shell_model(cfg)
    var qe = QEnv(cfg.kind, cfg.n, qm)
    var ue = UEnv(cfg.kind, cfg.n, _qumo_shell(qm))
    if cfg.kind == MODEL_KIND_PAULI:
        var R = (len(st) - 2 * cfg.n - 1) // 4
        qe.init_pauli(R, st)
        ue.init_pauli(R, st)
        check(qe.R == ue.R and len(qe.anti) == len(ue.anti), "Pauli init")
        for i in range(len(qe.anti)):
            check(qe.anti[i] == ue.anti[i], "Pauli anticommutation table")
    var sq = st.copy()
    var su = st.copy()
    var r = QRng(seed * 31 + 7)
    var A = qe.num_actions()
    check(A == ue.num_actions(), "action count")
    var what = kind_name + " on " + topo
    for k in range(steps):
        var oq = List[Int]()
        var ou = List[Int]()
        qe.observe(sq, oq)
        ue.observe(su, ou)
        check(_same_ints(oq, ou), what + ": observation at step " + String(k))
        check(qe.solved(sq) == ue.solved(su), what + ": solved flag at step " + String(k))
        check(qe.key(sq) == ue.key(su), what + ": state key at step " + String(k))
        var iq = List[UInt64]()
        var iu = List[UInt64]()
        var okq = qe.inverse(sq, iq)
        var oku = ue.inverse(su, iu)
        check(okq == oku and (not okq or _same(iq, iu)), what + ": inverse at step " + String(k))
        var a = r.below(A)
        check(_same(qe.step(sq, a), ue.step(su, a)), what + ": step() at step " + String(k))
        var eq = List[Int]()
        var eu = List[Int]()
        qe.apply(sq, a, eq, True)
        ue.apply(su, a, eu, True)
        check(_same(sq, su), what + ": state after action " + String(a) + " at step " + String(k))
        check(_same_ints(eq, eu), what + ": emitted gates at step " + String(k))
    return steps


def test_policy_env() raises:
    var n = 0
    var specs: List[String] = [
        "permutation|line:4", "permutation|ring:5", "permutation|grid:2x3", "permutation|star:4",
        "linear_function|line:4", "linear_function|ring:5", "linear_function|grid:2x3",
        "clifford|line:3", "clifford|ring:4", "clifford|line:2",
        "pauli_network|line:3", "pauli_network|ring:4", "pauli_network|line:2",
    ]
    for s in specs:
        var parts = s.split("|")
        for seed in range(1, 5):
            for d in [1, 4, 16]:
                n += _walk(String(parts[0]), String(parts[1]), d, UInt64(seed), 60)
    print("  PolicyEnv: " + String(len(specs)) + " kind / topology pairs, " + String(n) + " random steps identical")


# ------------------------------------------------------------------------------------------------ JSON


def _json_files(dir: String, mut out: List[String]) raises:
    for name in listdir(dir):
        var p = dir + "/" + name
        if isdir(p):
            _json_files(p, out)
        elif name.endswith(".json"):
            out.append(p)


def test_json() raises:
    var files = List[String]()
    _json_files(fixture_root(), files)
    _json_files("models/ai_transpiler", files)
    for p in files:
        var text = read_text(p)
        check(q_pretty(text, 1) == u_pretty(text, 1), "pretty_json(1) of " + p)
        check(q_pretty(text) == u_pretty(text), "pretty_json(2) of " + p)
        check(q_parse(text).to_json() == u_parse(text).to_json(), "compact re-serialisation of " + p)
    var r = QRng(5)
    for _ in range(5000):
        var v = (r.uniform() - 0.5) * Float64(10 ** r.below(12)) / Float64(10 ** r.below(12))
        check(q_number(v) == u_number(v), "json_number " + String(v))
    for s in ["", "plain", 'quote " and \\ back', "tab\tnl\ncr\r", "\x01\x1f", "utf8 é ✓"]:
        check(q_quote(s) == u_quote(s), "json_quote")
    print("  JSON: " + String(len(files)) + " files pretty-printed and re-serialised identically, 5000 numbers")


def main() raises:
    test_rng()
    test_gates_and_model_helpers()
    test_policy_env()
    test_json()
    print("test_bridge_parity OK")
