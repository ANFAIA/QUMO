"""Learned-policy models for the AI transpiler: container loader, MLP inference,
model lookup by coupling-graph isomorphism.

Hardware-neutral, native re-implementation of the inference side of
qiskit-ibm-transpiler's RL synthesis (arXiv:2405.13196) and of twisterl's
`BasicPolicy` forward pass (twisterl 0.5.1, Apache-2.0). The weights are
QUMO's own (trained locally with qiskit-gym 0.4.1, Apache-2.0); no IBM model
or code is used. A policy only *proposes* gate sequences: callers verify every
proposal exactly before using it.

Container (`qumo.rlsynth.v0` / `qumo.rlroute.v0`, written by
training/ai_transpiler/common/export.py): a directory with `manifest.json`
(format, id, kind, num_qubits, edges, actions, obs, arch, symmetries, tensors,
weights_bytes, fnv1a64, sha256, test_vectors, rollouts, training, license) and
`weights.bin` (little-endian f32/f16 tensors, 64-byte aligned). The FNV-1a 64
checksum of weights.bin is verified on load (QM-E617 on any mismatch).

Model lookup: `<root>/synth/INDEX.json` and `<root>/route/INDEX.json` list the
models of a root directory; roots come from `model_search_dirs` (explicit
`model_dir=`, then $QUMO_MODEL_DIR, then ./models/ai_transpiler).

Forward pass (twisterl `Policy::_raw_predict`, embedding-bag first layer):
    h = relu(b_e + sum_{i in obs} W_eT[i, :]);  t = relu(W_c h + b_c) (each common layer)
    logits = W_a t + b_a;  value = W_v t + b_v
Symmetry ensemble (`full_predict`): for p = 0..P-1, obs -> obs_perms[p][obs],
logits_p[a] = raw[act_perms[p][a]]; mean over p in that fixed order.
Everything is single-threaded and in a fixed summation order: bit-identical runs.
"""

from std.collections import List, Dict
from std.os import getenv
from std.os.path import exists
from std.memory import bitcast
from qumo.ai.json import parse_json, JsonValue
from qumo.ir.gates import Op
from qumo.compiler.blockalg import SubGraph, BlockKind


comptime MODEL_FORMAT_SYNTH = "qumo.rlsynth.v0"
comptime MODEL_FORMAT_ROUTE = "qumo.rlroute.v0"

# kind codes (synthesis kinds mirror blockalg.BlockKind)
comptime MODEL_KIND_CLIFFORD = 0
comptime MODEL_KIND_LINEAR = 1
comptime MODEL_KIND_PERM = 2
comptime MODEL_KIND_PAULI = 3
comptime MODEL_KIND_ROUTING = 4


def model_kind_code(kind: String) -> Int:
    """Manifest `kind` string -> code (-1 if unknown)."""
    if kind == "clifford":
        return MODEL_KIND_CLIFFORD
    if kind == "linear_function":
        return MODEL_KIND_LINEAR
    if kind == "permutation":
        return MODEL_KIND_PERM
    if kind == "pauli_network":
        return MODEL_KIND_PAULI
    if kind == "routing":
        return MODEL_KIND_ROUTING
    return -1


def model_kind_name(code: Int) -> String:
    if code == MODEL_KIND_CLIFFORD:
        return "clifford"
    if code == MODEL_KIND_LINEAR:
        return "linear_function"
    if code == MODEL_KIND_PERM:
        return "permutation"
    if code == MODEL_KIND_PAULI:
        return "pauli_network"
    if code == MODEL_KIND_ROUTING:
        return "routing"
    return "unknown"


def gate_code(name: String) -> Int:
    """Gate name used in manifests ("SWAP", "CX", "H", ...) -> opcode (-1 if unknown)."""
    var u = name.upper()
    if u == "SWAP":
        return Op.SWAP
    if u == "CX":
        return Op.CX
    if u == "CZ":
        return Op.CZ
    if u == "H":
        return Op.H
    if u == "S":
        return Op.S
    if u == "SDG":
        return Op.SDG
    if u == "SX":
        return Op.SX
    if u == "SXDG":
        return Op.SXDG
    return -1


# ----------------------------------------------------------------- checksums
comptime FNV_OFFSET: UInt64 = 0xCBF29CE484222325
comptime FNV_PRIME: UInt64 = 0x100000001B3


def fnv1a64_bytes(data: List[UInt8]) -> UInt64:
    var h = FNV_OFFSET
    for i in range(len(data)):
        h ^= UInt64(data[i])
        h *= FNV_PRIME
    return h


def fnv_words(words: List[UInt64]) -> UInt64:
    """FNV-1a over the little-endian 8-byte encoding of each word (matches
    training/ai_transpiler/common/topology.py `fnv_words`)."""
    var h = FNV_OFFSET
    for w in words:
        var u = w
        for k in range(8):
            h ^= (u >> UInt64(8 * k)) & 0xFF
            h *= FNV_PRIME
    return h


def hex16(v: UInt64) -> String:
    var digits = "0123456789abcdef"
    var bs = digits.as_bytes()
    var out = String()
    for k in range(15, -1, -1):
        var d = Int((v >> UInt64(4 * k)) & 0xF)
        out += String(unsafe_from_utf8=bs[d:d + 1])
    return out^


def wl_hash(n: Int, edges: List[Int], iters: Int = 3) -> String:
    """Weisfeiler-Lehman colour-refinement hash of an undirected graph given as
    flat pairs (prefilter only; collisions are resolved by an exact matcher).
    Same rule as training/ai_transpiler/common/topology.py `wl_hash`."""
    var adj = List[List[Int]]()
    for _ in range(n):
        adj.append(List[Int]())
    var ne = len(edges) // 2
    for e in range(ne):
        var a = edges[2 * e]
        var b = edges[2 * e + 1]
        adj[a].append(b)
        adj[b].append(a)
    var col = List[UInt64]()
    for v in range(n):
        col.append(UInt64(len(adj[v])))
    for _ in range(iters):
        var nxt = List[UInt64]()
        for v in range(n):
            var nb = List[UInt64]()
            for w in adj[v]:
                nb.append(col[w])
            sort(nb)
            var words = List[UInt64]()
            words.append(col[v])
            for x in nb:
                words.append(x)
            nxt.append(fnv_words(words))
        col = nxt^
    var s = col.copy()
    sort(s)
    var words = List[UInt64]()
    words.append(UInt64(n))
    words.append(UInt64(ne))
    for x in s:
        words.append(x)
    return hex16(fnv_words(words))


# ----------------------------------------------------------------- f16 / f32
def f32_from_bits(u: UInt32) -> Float32:
    return bitcast[DType.float32](u)


def f16_to_f32(h: UInt16) -> Float32:
    """IEEE binary16 -> binary32 (exact; handles subnormals, inf, nan)."""
    var sign = UInt32(h >> 15) & 1
    var exp = Int((h >> 10) & 0x1F)
    var man = UInt32(h & 0x3FF)
    var bits: UInt32
    if exp == 0:
        if man == 0:
            bits = sign << 31
        else:
            # subnormal: normalise
            var e = -1
            var m = man
            while (m & 0x400) == 0:
                m <<= 1
                e -= 1
            m &= 0x3FF
            bits = (sign << 31) | (UInt32(e + 114) << 23) | (m << 13)
    elif exp == 31:
        bits = (sign << 31) | (UInt32(0xFF) << 23) | (man << 13)
    else:
        bits = (sign << 31) | (UInt32(exp - 15 + 127) << 23) | (man << 13)
    return f32_from_bits(bits)


def _u32_at(bytes: List[UInt8], off: Int) -> UInt32:
    return UInt32(bytes[off]) | (UInt32(bytes[off + 1]) << 8) | (UInt32(bytes[off + 2]) << 16) | (UInt32(bytes[off + 3]) << 24)


def _u16_at(bytes: List[UInt8], off: Int) -> UInt16:
    return UInt16(bytes[off]) | (UInt16(bytes[off + 1]) << 8)


def _e617(dir: String, what: String) -> Error:
    return Error("QM-E617: model '" + dir + "': " + what + ". Hint: re-export the model with training/ai_transpiler/common/export.py, or remove it from INDEX.json / pass model=off.")


# ----------------------------------------------------------------- ModelFile
struct ModelFile(Movable):
    """A parsed model container: manifest fields plus decoded f32 tensors.

    actions: flat triples (opcode, q0, q1) in the model's exact action order
    (q1 = -1 for 1-qubit gates). edges: flat undirected pairs (a < b).
    obs_perms / act_perms: the symmetry ensemble (may be empty).
    """

    var id: String
    var kind: String
    var kind_code: Int
    var format: String
    var dir: String
    var num_qubits: Int
    var manifest: String
    var names: List[String]
    var tensors: List[List[Float32]]
    var shapes: List[List[Int]]
    var edges: List[Int]
    var actions: List[Int]
    var obs_size: Int
    var obs_layout: String
    var max_depth: Int
    var max_rotations: Int
    var obs_perms: List[List[Int]]
    var act_perms: List[List[Int]]

    def __init__(out self):
        self.id = String()
        self.kind = String()
        self.kind_code = -1
        self.format = String()
        self.dir = String()
        self.num_qubits = 0
        self.manifest = String()
        self.names = List[String]()
        self.tensors = List[List[Float32]]()
        self.shapes = List[List[Int]]()
        self.edges = List[Int]()
        self.actions = List[Int]()
        self.obs_size = 0
        self.obs_layout = String()
        self.max_depth = 128
        self.max_rotations = 0
        self.obs_perms = List[List[Int]]()
        self.act_perms = List[List[Int]]()

    def num_actions(self) -> Int:
        return len(self.actions) // 3

    def has_tensor(self, name: String) -> Bool:
        for i in range(len(self.names)):
            if self.names[i] == name:
                return True
        return False

    def tensor_index(self, name: String) raises -> Int:
        for i in range(len(self.names)):
            if self.names[i] == name:
                return i
        raise _e617(self.dir, "missing tensor '" + name + "'")

    def tensor(self, name: String) raises -> List[Float32]:
        return self.tensors[self.tensor_index(name)].copy()

    def shape(self, name: String) raises -> List[Int]:
        return self.shapes[self.tensor_index(name)].copy()


def _int_list(v: JsonValue) raises -> List[Int]:
    var out = List[Int]()
    for i in range(len(v)):
        out.append(v.at(i).as_int())
    return out^


def load_model_file(dir: String) raises -> ModelFile:
    """Load `<dir>/manifest.json` + weights, verify format and checksum.

    Raises QM-E617 on a missing file, unparsable manifest, unknown format,
    size/offset error or FNV-1a 64 mismatch."""
    var mpath = dir + "/manifest.json"
    if not exists(mpath):
        raise _e617(dir, "manifest.json not found")
    var text: String
    with open(mpath, "r") as f:
        text = f.read()
    var man: JsonValue
    try:
        man = parse_json(text)
    except e:
        raise _e617(dir, "manifest.json is not valid JSON (" + String(e) + ")")
    var m = ModelFile()
    m.dir = dir
    m.manifest = text
    try:
        m.format = man.get("format").as_string()
    except:
        raise _e617(dir, "manifest has no 'format'")
    if m.format != MODEL_FORMAT_SYNTH and m.format != MODEL_FORMAT_ROUTE:
        raise _e617(dir, "unsupported format '" + m.format + "' (expected " + MODEL_FORMAT_SYNTH + " or " + MODEL_FORMAT_ROUTE + ")")
    var wbytes: List[UInt8]
    try:
        m.id = man.get("id").as_string()
        m.kind = man.get("kind").as_string()
        m.kind_code = model_kind_code(m.kind)
        m.num_qubits = man.get_int("num_qubits", 0)
        if man.has("edges"):
            var ej = man.get("edges")
            for i in range(len(ej)):
                var a = ej.at(i).at(0).as_int()
                var b = ej.at(i).at(1).as_int()
                m.edges.append(min(a, b))
                m.edges.append(max(a, b))
        if man.has("actions"):
            var aj = man.get("actions")
            for i in range(len(aj)):
                var t = aj.at(i)
                var g = gate_code(t.at(0).as_string())
                if g < 0:
                    raise _e617(dir, "unknown action gate '" + t.at(0).as_string() + "'")
                m.actions.append(g)
                m.actions.append(t.at(1).as_int())
                m.actions.append(t.at(2).as_int() if len(t) > 2 else -1)
        if man.has("obs"):
            var oj = man.get("obs")
            m.obs_size = oj.get_int("size", 0)
            m.obs_layout = oj.get_string("layout", "")
            m.max_rotations = oj.get_int("max_rotations", 0)
        m.max_depth = man.get_int("max_depth", 128)
        if man.has("symmetries"):
            var sj = man.get("symmetries")
            if sj.has("obs_perms"):
                var op = sj.get("obs_perms")
                for i in range(len(op)):
                    m.obs_perms.append(_int_list(op.at(i)))
            if sj.has("act_perms"):
                var ap = sj.get("act_perms")
                for i in range(len(ap)):
                    m.act_perms.append(_int_list(ap.at(i)))
        var wfile = man.get_string("weights_file", "weights.bin")
        var wpath = dir + "/" + wfile
        if not exists(wpath):
            raise _e617(dir, wfile + " not found")
        with open(wpath, "r") as f:
            wbytes = f.read_bytes()
        var want = man.get_int("weights_bytes", -1)
        if want >= 0 and want != len(wbytes):
            raise _e617(dir, "weights.bin has " + String(len(wbytes)) + " bytes, manifest says " + String(want))
        var fnv = hex16(fnv1a64_bytes(wbytes))
        var fwant = man.get_string("fnv1a64", "").lower()
        if fnv != fwant:
            raise _e617(dir, "checksum mismatch (fnv1a64 " + fnv + " != manifest " + fwant + ")")
        var tj = man.get("tensors")
        for i in range(len(tj)):
            var t = tj.at(i)
            var name = t.get("name").as_string()
            var shape = _int_list(t.get("shape"))
            var cnt = 1
            for s in shape:
                cnt *= s
            var dt = t.get_string("dtype", "f32")
            var off = t.get("offset").as_int()
            var esz = 4 if dt == "f32" else 2
            if dt != "f32" and dt != "f16":
                raise _e617(dir, "tensor '" + name + "' has unsupported dtype '" + dt + "'")
            if off < 0 or off + cnt * esz > len(wbytes):
                raise _e617(dir, "tensor '" + name + "' lies outside weights.bin")
            var data = List[Float32](capacity=cnt)
            if esz == 4:
                for k in range(cnt):
                    data.append(f32_from_bits(_u32_at(wbytes, off + 4 * k)))
            else:
                for k in range(cnt):
                    data.append(f16_to_f32(_u16_at(wbytes, off + 2 * k)))
            m.names.append(name)
            m.shapes.append(shape^)
            m.tensors.append(data^)
    except e:
        var s = String(e)
        if s.startswith("QM-E617"):
            raise Error(s)
        raise _e617(dir, "malformed manifest (" + s + ")")
    return m^


# ----------------------------------------------------------------- search dirs
def model_search_dirs(explicit: String) -> List[String]:
    """Existing model roots in lookup order: explicit `model_dir=`, then
    $QUMO_MODEL_DIR, then ./models/ai_transpiler. Each root may hold `synth/`
    and `route/` subdirectories with an INDEX.json."""
    var out = List[String]()
    var cand = List[String]()
    if explicit.byte_length() > 0:
        cand.append(explicit)
    var env = getenv("QUMO_MODEL_DIR")
    if env.byte_length() > 0:
        cand.append(env)
    cand.append("models/ai_transpiler")
    for c in cand:
        if not exists(c):
            continue
        var dup = False
        for o in out:
            if o == c:
                dup = True
        if not dup:
            out.append(c)
    return out^


struct ModelIndexEntry(Copyable):
    """One INDEX.json row; `path` is the model directory (root/sub/dir)."""

    var id: String
    var kind_code: Int
    var num_qubits: Int
    var edges: List[Int]
    var wl_hash: String
    var path: String

    def __init__(out self):
        self.id = String()
        self.kind_code = -1
        self.num_qubits = 0
        self.edges = List[Int]()
        self.wl_hash = String()
        self.path = String()


def read_model_index(root: String, sub: String) -> List[ModelIndexEntry]:
    """Rows of `<root>/<sub>/INDEX.json` (empty when absent or unreadable)."""
    var out = List[ModelIndexEntry]()
    var ipath = root + "/" + sub + "/INDEX.json"
    if not exists(ipath):
        return out^
    try:
        var text: String
        with open(ipath, "r") as f:
            text = f.read()
        var j = parse_json(text)
        for i in range(len(j)):
            var r = j.at(i)
            var e = ModelIndexEntry()
            e.id = r.get("id").as_string()
            e.kind_code = model_kind_code(r.get("kind").as_string())
            e.num_qubits = r.get("num_qubits").as_int()
            var ej = r.get("edges")
            for k in range(len(ej)):
                var a = ej.at(k).at(0).as_int()
                var b = ej.at(k).at(1).as_int()
                e.edges.append(min(a, b))
                e.edges.append(max(a, b))
            e.wl_hash = r.get_string("wl_hash", "")
            e.path = root + "/" + sub + "/" + r.get("dir").as_string()
            out.append(e^)
    except:
        return List[ModelIndexEntry]()
    return out^


# ----------------------------------------------------------------- MLP policy
@fieldwise_init
struct _Dense(Copyable):
    var w: List[Float32]  # [n_out, n_in] row-major
    var b: List[Float32]
    var n_out: Int
    var n_in: Int


def _dense(layer: _Dense, x: List[Float32], relu: Bool) -> List[Float32]:
    """y = W x + b (8-lane SIMD dot products, 2 accumulators, scalar tail)."""
    comptime W = 8
    var n_in = layer.n_in
    var y = List[Float32](length=layer.n_out, fill=0.0)
    var pw = layer.w.unsafe_ptr().as_unsafe_any_origin()
    var px = x.unsafe_ptr().as_unsafe_any_origin()
    var main = (n_in // (2 * W)) * (2 * W)
    for o in range(layer.n_out):
        var base = o * n_in
        var a0 = SIMD[DType.float32, W](0)
        var a1 = SIMD[DType.float32, W](0)
        var k = 0
        while k < main:
            a0 += pw.unsafe_load[width=W](base + k) * px.unsafe_load[width=W](k)
            a1 += pw.unsafe_load[width=W](base + k + W) * px.unsafe_load[width=W](k + W)
            k += 2 * W
        var v = (a0 + a1).reduce_add()
        while k < n_in:
            v += pw[unsafe_offset=base + k] * px[unsafe_offset=k]
            k += 1
        v += layer.b[o]
        if relu and v < 0.0:
            v = 0.0
        y[o] = v
    _ = len(layer.w)
    _ = len(x)
    return y^


def _dense_batch(layer: _Dense, x: List[Float32], B: Int, relu: Bool) -> List[Float32]:
    """Row-batched `_dense`: x is [B, n_in] row-major, returns [B, n_out]. Each
    weight row is read once for all B vectors; per-vector arithmetic is the same
    as `_dense` (bit-identical results)."""
    comptime W = 8
    var n_in = layer.n_in
    var n_out = layer.n_out
    var y = List[Float32](length=B * n_out, fill=0.0)
    var pw = layer.w.unsafe_ptr().as_unsafe_any_origin()
    var px = x.unsafe_ptr().as_unsafe_any_origin()
    var py = y.unsafe_ptr().as_unsafe_any_origin()
    var main = (n_in // (2 * W)) * (2 * W)
    for o in range(n_out):
        var base = o * n_in
        var bo = layer.b[o]
        for b in range(B):
            var xb = b * n_in
            var a0 = SIMD[DType.float32, W](0)
            var a1 = SIMD[DType.float32, W](0)
            var k = 0
            while k < main:
                a0 += pw.unsafe_load[width=W](base + k) * px.unsafe_load[width=W](xb + k)
                a1 += pw.unsafe_load[width=W](base + k + W) * px.unsafe_load[width=W](xb + k + W)
                k += 2 * W
            var v = (a0 + a1).reduce_add()
            while k < n_in:
                v += pw[unsafe_offset=base + k] * px[unsafe_offset=xb + k]
                k += 1
            v += bo
            if relu and v < 0.0:
                v = 0.0
            py[unsafe_offset=b * n_out + o] = v
    _ = len(layer.w)
    _ = len(x)
    _ = len(y)
    return y^


struct MLPPolicy(Movable):
    """twisterl BasicPolicy inference: sparse embedding bag + ReLU MLP torso,
    linear action / value heads (heads may have hidden ReLU layers)."""

    var obs_size: Int
    var emb: Int
    var we_t: List[Float32]  # [obs_size, emb]
    var be: List[Float32]
    var common: List[_Dense]
    var action: List[_Dense]
    var value: List[_Dense]
    var num_actions: Int
    var obs_perms: List[List[Int]]
    var act_perms: List[List[Int]]

    def __init__(out self):
        self.obs_size = 0
        self.emb = 0
        self.we_t = List[Float32]()
        self.be = List[Float32]()
        self.common = List[_Dense]()
        self.action = List[_Dense]()
        self.value = List[_Dense]()
        self.num_actions = 0
        self.obs_perms = List[List[Int]]()
        self.act_perms = List[List[Int]]()

    @staticmethod
    def _seq(m: ModelFile, prefix: String) raises -> List[_Dense]:
        var out = List[_Dense]()
        var k = 0
        while m.has_tensor(prefix + "." + String(k) + ".weight"):
            var wn = prefix + "." + String(k) + ".weight"
            var sh = m.shape(wn)
            if len(sh) != 2:
                raise _e617(m.dir, "tensor '" + wn + "' is not 2-D")
            var b = m.tensor(prefix + "." + String(k) + ".bias")
            if len(b) != sh[0]:
                raise _e617(m.dir, "bias of '" + wn + "' has the wrong size")
            out.append(_Dense(m.tensor(wn), b^, sh[0], sh[1]))
            k += 1
        return out^

    @staticmethod
    def from_model(m: ModelFile) raises -> MLPPolicy:
        var p = MLPPolicy()
        var sh = m.shape("embeddings.weight")  # stored transposed [OBS, EMB]
        if len(sh) != 2:
            raise _e617(m.dir, "embeddings.weight is not 2-D")
        p.obs_size = sh[0]
        p.emb = sh[1]
        p.we_t = m.tensor("embeddings.weight")
        p.be = m.tensor("embeddings.bias")
        if len(p.be) != p.emb:
            raise _e617(m.dir, "embeddings.bias has the wrong size")
        p.common = MLPPolicy._seq(m, "common")
        p.action = MLPPolicy._seq(m, "action")
        p.value = MLPPolicy._seq(m, "value")
        if len(p.action) == 0:
            raise _e617(m.dir, "no action head")
        var prev = p.emb
        for l in p.common:
            if l.n_in != prev:
                raise _e617(m.dir, "common layer shapes do not chain")
            prev = l.n_out
        var prev_a = prev
        for l in p.action:
            if l.n_in != prev_a:
                raise _e617(m.dir, "action head shapes do not chain")
            prev_a = l.n_out
        var prev_v = prev
        for l in p.value:
            if l.n_in != prev_v:
                raise _e617(m.dir, "value head shapes do not chain")
            prev_v = l.n_out
        p.num_actions = prev_a
        if m.num_actions() > 0 and m.num_actions() != p.num_actions:
            raise _e617(m.dir, "action head size " + String(p.num_actions) + " != " + String(m.num_actions()) + " manifest actions")
        if m.obs_size > 0 and m.obs_size != p.obs_size:
            raise _e617(m.dir, "obs size " + String(p.obs_size) + " != manifest obs.size " + String(m.obs_size))
        if len(m.obs_perms) != len(m.act_perms):
            raise _e617(m.dir, "obs_perms / act_perms count mismatch")
        for i in range(len(m.obs_perms)):
            if len(m.obs_perms[i]) != p.obs_size or len(m.act_perms[i]) != p.num_actions:
                raise _e617(m.dir, "symmetry " + String(i) + " has the wrong size")
        p.obs_perms = m.obs_perms.copy()
        p.act_perms = m.act_perms.copy()
        return p^

    def num_symmetries(self) -> Int:
        return len(self.obs_perms)

    def _raw(self, obs_idx: List[Int], perm: Int, mut logits: List[Float32]) -> Float32:
        """Raw forward; `perm` >= 0 relabels the observation with obs_perms[perm]
        (logits stay in the model's frame). Out-of-range indices are ignored."""
        comptime W = 8
        var E = self.emb
        var h = self.be.copy()
        var ph = h.unsafe_ptr().as_unsafe_any_origin()
        var pw = self.we_t.unsafe_ptr().as_unsafe_any_origin()
        var mainE = (E // W) * W
        for j in range(len(obs_idx)):
            var idx = obs_idx[j]
            if perm >= 0:
                idx = self.obs_perms[perm][idx]
            if idx < 0 or idx >= self.obs_size:
                continue
            var base = idx * E
            var k = 0
            while k < mainE:
                ph.unsafe_store(k, ph.unsafe_load[width=W](k) + pw.unsafe_load[width=W](base + k))
                k += W
            while k < E:
                ph[unsafe_offset=k] = ph[unsafe_offset=k] + pw[unsafe_offset=base + k]
                k += 1
        _ = len(self.we_t)
        for e in range(E):
            if h[e] < 0.0:
                h[e] = 0.0
        var t = h^
        for l in self.common:
            t = _dense(l, t, True)
        var a = t.copy()
        for i in range(len(self.action)):
            a = _dense(self.action[i], a, i + 1 < len(self.action))
        var v: Float32 = 0.0
        if len(self.value) > 0:
            var vv = t^
            for i in range(len(self.value)):
                vv = _dense(self.value[i], vv, i + 1 < len(self.value))
            v = vv[0]
        logits = a^
        return v

    def raw_batch(self, obs: List[List[Int]], perms: List[Int], mut logits: List[Float32], mut values: List[Float32]):
        """Batched raw forward: item b uses obs[b] relabelled by obs_perms[perms[b]]
        (perms[b] < 0: identity); logits come back flat [B, A] in the model frame
        (no act_perms applied). Bit-identical to calling `_raw` per item."""
        comptime W = 8
        var B = len(obs)
        var E = self.emb
        var h = List[Float32](length=B * E, fill=0.0)
        var ph = h.unsafe_ptr().as_unsafe_any_origin()
        var pw = self.we_t.unsafe_ptr().as_unsafe_any_origin()
        var pb = self.be.unsafe_ptr().as_unsafe_any_origin()
        var mainE = (E // W) * W
        for b in range(B):
            var hb = b * E
            for e in range(E):
                ph[unsafe_offset=hb + e] = pb[unsafe_offset=e]
            var p = perms[b]
            for j in range(len(obs[b])):
                var idx = obs[b][j]
                if p >= 0:
                    idx = self.obs_perms[p][idx]
                if idx < 0 or idx >= self.obs_size:
                    continue
                var base = idx * E
                var k = 0
                while k < mainE:
                    ph.unsafe_store(hb + k, ph.unsafe_load[width=W](hb + k) + pw.unsafe_load[width=W](base + k))
                    k += W
                while k < E:
                    ph[unsafe_offset=hb + k] = ph[unsafe_offset=hb + k] + pw[unsafe_offset=base + k]
                    k += 1
            for e in range(E):
                if ph[unsafe_offset=hb + e] < 0.0:
                    ph[unsafe_offset=hb + e] = 0.0
        _ = len(self.we_t)
        _ = len(self.be)
        var t = h^
        for l in self.common:
            t = _dense_batch(l, t, B, True)
        var a = t.copy()
        for i in range(len(self.action)):
            a = _dense_batch(self.action[i], a, B, i + 1 < len(self.action))
        values = List[Float32](length=B, fill=0.0)
        if len(self.value) > 0:
            var vv = t^
            for i in range(len(self.value)):
                vv = _dense_batch(self.value[i], vv, B, i + 1 < len(self.value))
            var nv = self.value[len(self.value) - 1].n_out
            for b in range(B):
                values[b] = vv[b * nv]
        logits = a^

    def forward(self, obs_idx: List[Int], mut logits: List[Float32]) -> Float32:
        """Identity-frame forward pass; fills `logits` (num_actions), returns the value."""
        return self._raw(obs_idx, -1, logits)

    def forward_sym(self, obs_idx: List[Int], p: Int, mut logits: List[Float32]) -> Float32:
        """Forward through symmetry p: logits_out[a] = raw[act_perms[p][a]]."""
        var raw = List[Float32]()
        var v = self._raw(obs_idx, p, raw)
        logits = List[Float32](length=self.num_actions, fill=0.0)
        for a in range(self.num_actions):
            logits[a] = raw[self.act_perms[p][a]]
        return v

    def forward_ensemble(self, obs_idx: List[Int], mut logits: List[Float32]) -> Float32:
        """Mean over all symmetries (fixed order 0..P-1, each term divided by P
        before accumulation, as twisterl `full_predict`). Falls back to the
        identity forward when the model has no symmetries."""
        var P = len(self.obs_perms)
        if P == 0:
            return self._raw(obs_idx, -1, logits)
        var obs = List[List[Int]]()
        var perms = List[Int]()
        for p in range(P):
            obs.append(obs_idx.copy())
            perms.append(p)
        var raw = List[Float32]()
        var vals = List[Float32]()
        self.raw_batch(obs, perms, raw, vals)
        var A = self.num_actions
        logits = List[Float32](length=A, fill=0.0)
        var v: Float32 = 0.0
        var fp = Float32(P)
        for p in range(P):
            v += vals[p] / fp
            for a in range(A):
                logits[a] += raw[p * A + self.act_perms[p][a]] / fp
        return v


# ----------------------------------------------------------------- model lookup
struct ModelMatch(Copyable):
    """Result of `find_model`. perm[model qubit] = local block qubit."""

    var ok: Bool
    var id: String
    var dir: String
    var perm: List[Int]
    var exact: Bool  # True for an isomorphism, False for a monomorphism

    def __init__(out self):
        self.ok = False
        self.id = String()
        self.dir = String()
        self.perm = List[Int]()
        self.exact = False


def _adj_matrix(n: Int, edges: List[Int]) -> List[Bool]:
    var m = List[Bool](length=n * n, fill=False)
    for e in range(len(edges) // 2):
        var a = edges[2 * e]
        var b = edges[2 * e + 1]
        if a >= 0 and b >= 0 and a < n and b < n and a != b:
            m[a * n + b] = True
            m[b * n + a] = True
    return m^


def _num_undirected(n: Int, adj: List[Bool]) -> Int:
    var c = 0
    for a in range(n):
        for b in range(a + 1, n):
            if adj[a * n + b]:
                c += 1
    return c


def _embed_rec(n: Int, order: List[Int], pos: Int, madj: List[Bool], gadj: List[Bool],
               mdeg: List[Int], gdeg: List[Int], mut phi: List[Int], mut used: List[Bool],
               mut steps: Int, cap: Int) -> Bool:
    if pos == n:
        return True
    steps += 1
    if steps > cap:
        return False
    var u = order[pos]
    for v in range(n):
        if used[v] or gdeg[v] < mdeg[u]:
            continue
        var ok = True
        for k in range(pos):
            var w = order[k]
            if madj[u * n + w] and not gadj[v * n + phi[w]]:
                ok = False
                break
        if not ok:
            continue
        phi[u] = v
        used[v] = True
        if _embed_rec(n, order, pos + 1, madj, gadj, mdeg, gdeg, phi, used, steps, cap):
            return True
        used[v] = False
        phi[u] = -1
    return False


def graph_embedding(n: Int, model_edges: List[Int], g_edges: List[Int], mut phi: List[Int]) -> Bool:
    """Find an injective map phi (model node -> g node, both on n nodes) such that
    every model edge maps onto a g edge (a monomorphism; an isomorphism when
    the edge counts are equal). Deterministic backtracking: model nodes in BFS
    order from the highest-degree node, candidates in increasing index,
    degree pruning; capped at 200000 search nodes."""
    var madj = _adj_matrix(n, model_edges)
    var gadj = _adj_matrix(n, g_edges)
    var mdeg = List[Int](length=n, fill=0)
    var gdeg = List[Int](length=n, fill=0)
    for a in range(n):
        for b in range(n):
            if madj[a * n + b]:
                mdeg[a] += 1
            if gadj[a * n + b]:
                gdeg[a] += 1
    if _num_undirected(n, madj) > _num_undirected(n, gadj):
        return False
    # order: BFS from highest-degree node (ties -> lowest index), components in turn
    var order = List[Int]()
    var seen = List[Bool](length=n, fill=False)
    while len(order) < n:
        var start = -1
        for v in range(n):
            if not seen[v] and (start < 0 or mdeg[v] > mdeg[start]):
                start = v
        seen[start] = True
        var q = List[Int]()
        q.append(start)
        var head = 0
        while head < len(q):
            var x = q[head]
            head += 1
            order.append(x)
            for y in range(n):
                if madj[x * n + y] and not seen[y]:
                    seen[y] = True
                    q.append(y)
    phi = List[Int](length=n, fill=-1)
    var used = List[Bool](length=n, fill=False)
    var steps = 0
    return _embed_rec(n, order, 0, madj, gadj, mdeg, gdeg, phi, used, steps, 200000)


def load_synth_index(dirs: List[String]) -> List[ModelIndexEntry]:
    """INDEX rows of `<root>/synth` for every root, in `dirs` order."""
    var rows = List[ModelIndexEntry]()
    for root in dirs:
        var r = read_model_index(root, "synth")
        for e in r:
            rows.append(e.copy())
    return rows^


def find_model(kind: Int, g: SubGraph, request: String, dirs: List[String]) -> ModelMatch:
    """Find a synthesis model of `kind` (BlockKind) whose coupling graph embeds
    into the block graph `g` with the same number of qubits (reads the INDEX
    files of `dirs`; see `find_model_rows`)."""
    return find_model_rows(kind, g, request, load_synth_index(dirs))


def find_model_rows(kind: Int, g: SubGraph, request: String, rows: List[ModelIndexEntry]) -> ModelMatch:
    """`find_model` over pre-loaded INDEX rows.

    request: "auto" (any model), "off" (never matches) or a model id.
    Preference: an exact isomorphism (WL-hash prefilter, confirmed by the
    matcher), then the monomorphism with the most model edges (valid because
    the policy then only uses a subset of g's couplings), then the earlier row
    (roots in `dirs` order, rows sorted by id). perm[model qubit] = block qubit."""
    var res = ModelMatch()
    if request == "off" or g.n <= 0:
        return res^
    var g_hash = wl_hash(g.n, g.edges)
    var g_ne = len(g.edges) // 2
    var best_score = -1
    for r in rows:
        if r.kind_code != kind or r.num_qubits != g.n:
            continue
        if request != "auto" and r.id != request:
            continue
        var m_ne = len(r.edges) // 2
        if m_ne > g_ne:
            continue
        var iso = m_ne == g_ne
        if iso and r.wl_hash.byte_length() > 0 and r.wl_hash != g_hash:
            continue  # same edge count but not isomorphic (WL differs)
        var score = (1 << 20) * (1 if iso else 0) + m_ne
        if score <= best_score:
            continue
        var phi = List[Int]()
        if not graph_embedding(g.n, r.edges, g.edges, phi):
            continue
        best_score = score
        res.ok = True
        res.id = r.id
        res.dir = r.path
        res.perm = phi^
        res.exact = iso
    return res^
