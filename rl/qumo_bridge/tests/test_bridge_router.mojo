"""Bridge tests of the router network against QUMO's runtime router (`qumo.compiler.ai_route`).

The halves of the former `test_rl_router` that need QUMO: `RouterNet` forward against the independent float64
`RouteModel.forward` of the runtime (random weights, hidden sizes that are not multiples of the SIMD width, and the
shipped model), and QUMO's `load_route_model` accepting a behaviour-cloning run directory as a router model.
Run from the QUMO root with `-I . -I rl` (rl/Makefile `test-bridge`).
"""

from std.os import makedirs
from qrl.rng import Rng
from qrl.model import load_model_file
from qrl.router_net import RouterNet, RouterCache
from qrl.teacher_data import read_teacher_files, encode_teacher_record
from qrl.bc import BCConfig, train_bc, write_bc_run
from qrl.testutil import assert_close, model_root
from qumo.compiler.ai_route import RouteModel, load_route_model

comptime OUT_DIR = "build/test_bridge_router"


def _random_net(h: Int, seed: UInt64) -> RouterNet:
    var net = RouterNet(h, seed)
    var r = Rng(seed + 99)
    for i in range(len(net.params)):
        net.params[i] = Float32(r.uniform() - 0.5)
    return net^


def _mutp[T: DType](mut l: List[Scalar[T]]) -> MutPointer[Scalar[T], MutAnyOrigin]:
    return l.unsafe_ptr().as_unsafe_any_origin()


def _route_model_of(net: RouterNet) -> RouteModel:
    var m = RouteModel()
    m.ok = True
    m.hidden = net.hidden
    for i in range(128 * net.hidden):
        m.emb.append(net.params[net.off_emb() + i])
    for i in range(net.hidden):
        m.b0.append(net.params[net.off_b0() + i])
    for i in range(16 * net.hidden):
        m.w1.append(net.params[net.off_w1() + i])
    for i in range(16):
        m.b1.append(net.params[net.off_b1() + i])
    return m^


def test_forward_vs_runtime() raises:
    # the runtime RouteModel.forward is an independent float64 implementation of the same network
    for hi in range(3):
        var h = 64 if hi == 0 else (24 if hi == 1 else 5)
        var net = _random_net(h, UInt64(10 + hi))
        var model = _route_model_of(net)
        var rows = 37
        var rng = Rng(77)
        var sign = List[Int8](length=rows * 128, fill=0)
        var obs_all = List[List[Int]]()
        for r in range(rows):
            var obs = List[Int]()
            for i in range(128):
                var u = rng.below(5)
                var o = 0 if u < 2 else (u - 3)  # mostly 0, some -1/1
                obs.append(o * (1 + rng.below(3)))
                sign[r * 128 + i] = Int8(1 if o > 0 else (-1 if o < 0 else 0))
            obs_all.append(obs^)
        var cc = RouterCache(64, h)
        net.forward_chunk(_mutp(sign), rows, cc, True)
        var hid = List[Float64](length=h, fill=0.0)
        var logits = List[Float64](length=16, fill=0.0)
        for r in range(rows):
            model.forward(obs_all[r], hid, logits)
            for k in range(16):
                assert_close(Float64(cc.z[r * 16 + k]), logits[k], 2e-5, 2e-5, "logit h=" + String(h) + " row " + String(r))
            var v = Float64(net.params[net.off_bv()])
            for j in range(h):
                if hid[j] > 0.0:
                    v += Float64(net.params[net.off_wv() + j]) * hid[j]
            assert_close(Float64(cc.v[r]), v, 2e-5, 2e-5, "value h=" + String(h))


def test_shipped_model_vs_runtime() raises:
    var dir = model_root() + "/route/route_mlp_h64_v1"
    var mf = load_model_file(dir)
    var net = RouterNet.from_model(mf)
    var rm = load_route_model(dir)
    var rows = 20
    var rng = Rng(5)
    var sign = List[Int8](length=rows * 128, fill=0)
    var obs_all = List[List[Int]]()
    for r in range(rows):
        var obs = List[Int]()
        for i in range(128):
            var o = rng.below(7) - 3
            if rng.below(3) != 0:
                o = 0
            obs.append(o)
            sign[r * 128 + i] = Int8(1 if o > 0 else (-1 if o < 0 else 0))
        obs_all.append(obs^)
    var cc = RouterCache(32, 64)
    net.forward_chunk(_mutp(sign), rows, cc)
    var hid = List[Float64](length=64, fill=0.0)
    var logits = List[Float64](length=16, fill=0.0)
    for r in range(rows):
        rm.forward(obs_all[r], hid, logits)
        for k in range(16):
            assert_close(Float64(cc.z[r * 16 + k]), logits[k], 1e-4, 1e-4, "shipped router logit")


def _write(path: String, buf: List[UInt8]) raises:
    with open(path, "w") as f:
        f.write_bytes(Span(buf))


def _planted_file(rows: Int, seed: UInt64) -> List[UInt8]:
    """Teacher rows whose optimum is planted: slot `best` carries a +1 marker in its feature 0, every other valid slot
    a -1, the rest of the features are noise; the best slot costs 0, the others between 4 and 8."""
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


def test_bc_run_loads_in_runtime() raises:
    # a small behaviour-cloning run on a planted teacher file (as in the core test); QUMO's loader must accept the run
    # directory (f32 container) as a router model of the same hidden size
    makedirs(OUT_DIR, exist_ok=True)
    var paths = List[String]()
    var path = OUT_DIR + "/planted_run.bin"
    _write(path, _planted_file(3000, 11))
    paths.append(path)
    var td = read_teacher_files(paths)
    var cfg = BCConfig()
    cfg.hidden = 16
    cfg.epochs = 1
    cfg.seed = 21
    var out = train_bc(cfg, td, None)
    var run = OUT_DIR + "/run_a"
    write_bc_run(run, out[0], out[1].copy(), cfg, paths, "")
    var rm = load_route_model(run)
    if rm.hidden != 16:
        raise Error("route model hidden")


def main() raises:
    test_forward_vs_runtime()
    test_shipped_model_vs_runtime()
    test_bc_run_loads_in_runtime()
    print("test_bridge_router OK")
