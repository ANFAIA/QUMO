"""Tests for the QML layer: qumo.sim.rng normals, qumo.algorithms.datasets,
qumo.algorithms.qml (templates, embeddings, observables), the batched /
initial-state gradients of qumo.algorithms.gradients and
qumo.algorithms.qml_train (step optimizers, batches, losses, vjp).

Reference numbers come from PennyLane 0.45.1 (default.qubit), sklearn 1.9.1,
numpy and Qiskit 2.5.2; the generating scripts are summarised in docs/QML.md.
Prints `test_qml OK`."""

from std.math import pi, sqrt, sin, cos
from std.time import perf_counter_ns
from std.os import remove, rmdir
from std.tempfile import mkdtemp
from std.testing import assert_true, assert_equal

from qumo.ir.circuit import Circuit
from qumo.ir.param import Param
from qumo.ir.pauli import PauliSum, parse_pauli_sum
from qumo.ir.gates import u3_matrix
from qumo.sim.rng import Rng
from qumo.sim.statevector import StateVector
from qumo.sim.runner import evolve, evolve_into
from qumo.algorithms.hamiltonians import PauliOperator
from qumo.algorithms.gradients import (
    ExpectationEngine, Grad, gradient, batch_value_and_gradient, batch_gradient,
)
from qumo.algorithms.datasets import (
    parity_dataset, double_cake, circle_dataset, moons_dataset, bars_and_stripes,
    load_csv, load_json_matrix, standard_scale, apply_standard_scale, l2_normalize_rows,
    pad_rows, train_test_split, take_rows, take_values,
)
from qumo.algorithms.qml import (
    append_rot, append_angle_embedding, append_basis_embedding, iqp_embedding, iqp_feature_values,
    zz_feature_map, zz_feature_values, mottonen_angles, mottonen_tree_angles,
    append_amplitude_embedding_real, amplitude_state, append_strongly_entangling_layers,
    append_basic_entangler_layers, pauli_words_gray, append_arbitrary_unitary, append_controlled_u3,
    append_qcnn_conv, append_qcnn_pool, diagonal_observable, projector_zero, projector_basis,
    local_projector_zero,
)
from qumo.algorithms.qml_train import (
    StepOptimizer, cosine_lr, step_lr, Batcher, square_loss, hinge_loss, l1_loss, cross_entropy,
    fidelity_loss, accuracy, vjp, expectations,
)
from qumo.algorithms.qkernels import kernel_matrix


def _code(msg: String) -> String:
    return String(msg[byte=0:7]) if msg.byte_length() >= 7 else msg


def _close(a: Float64, b: Float64, tol: Float64, what: String) raises:
    assert_true(abs(a - b) <= tol, what + ": " + String(a) + " vs " + String(b) + " (|diff| " + String(abs(a - b)) + " > " + String(tol) + ")")


def _wires(n: Int) -> List[Int]:
    var w = List[Int]()
    for q in range(n):
        w.append(q)
    return w^


def _rev_wires(n: Int) -> List[Int]:
    var w = List[Int]()
    var q = n - 1
    while q >= 0:
        w.append(q)
        q -= 1
    return w^


# ---------------------------------------------------------------------------
# Rng.normal
# ---------------------------------------------------------------------------
def test_rng_normal() raises:
    var rng = Rng(2024)
    var n = 1000000
    var s = 0.0
    var s2 = 0.0
    for _ in range(n):
        var v = rng.normal()
        s += v
        s2 += v * v
    var mean = s / Float64(n)
    var var_ = s2 / Float64(n) - mean * mean
    assert_true(abs(mean) < 3.0 / sqrt(Float64(n)), "normal mean " + String(mean))
    assert_true(abs(var_ - 1.0) < 3.0 * sqrt(2.0 / Float64(n)), "normal variance " + String(var_))
    # same seed -> same stream; normal_list(scale) = scale * normal()
    var a = Rng(7)
    var b = Rng(7)
    var la = a.normal_list(5, 0.01)
    for i in range(5):
        assert_equal(la[i], 0.01 * b.normal())
    # split streams are independent (sample correlation ~ 0)
    var root = Rng(99)
    var r1 = root.split(1)
    var r2 = root.split(2)
    var m = 200000
    var c = 0.0
    for _ in range(m):
        c += r1.normal() * r2.normal()
    assert_true(abs(c / Float64(m)) < 4.0 / sqrt(Float64(m)), "split streams correlated: " + String(c / Float64(m)))


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
def test_datasets() raises:
    var p = parity_dataset(4)
    var PX = p[0].copy()
    var Py = p[1].copy()
    assert_equal(len(PX), 16)
    # row 5 = "0101" (first character = wire 0), odd parity... 0+1+0+1 = 2 -> -1
    assert_equal(PX[5][0], 0.0)
    assert_equal(PX[5][1], 1.0)
    assert_equal(PX[5][3], 1.0)
    assert_equal(Py[5], -1.0)
    assert_equal(Py[1], 1.0)
    # double_cake == PennyLane make_double_cake_data(3) (p2 fixture)
    var dc = double_cake(3)
    var DX = dc[0].copy()
    var Dy = dc[1].copy()
    var ref_x: List[Float64] = [0.35000000000000003, 0.606217782649107, -0.7, 8.572527594031472e-17, 0.34999999999999953, -0.6062177826491073, 0.17500000000000002, 0.3031088913245535, -0.35, 4.286263797015736e-17, 0.17499999999999977, -0.30310889132455365]
    var ref_y: List[Float64] = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
    for i in range(6):
        _close(DX[i][0], ref_x[2 * i], 1e-15, "double_cake x")
        _close(DX[i][1], ref_x[2 * i + 1], 1e-15, "double_cake y")
        assert_equal(Dy[i], ref_y[i])
    # bars and stripes (tutorial_qcbm order, 3x3: 14 patterns)
    var bs = bars_and_stripes(3, 3)
    assert_equal(len(bs), 14)
    assert_equal(bs[0], 0)
    assert_equal(bs[1], 0b100100100)
    assert_equal(bs[13], 511)
    assert_equal(len(bars_and_stripes(2, 3)), 10)
    # seeded random datasets: deterministic per seed
    var c1 = circle_dataset(50, 3)
    var c2 = circle_dataset(50, 3)
    var c3 = circle_dataset(50, 4)
    var X1 = c1[0].copy()
    var X2 = c2[0].copy()
    var X3 = c3[0].copy()
    var y1 = c1[1].copy()
    assert_equal(X1[17][1], X2[17][1])
    assert_true(X1[17][1] != X3[17][1])
    for i in range(50):
        var inside = sqrt(X1[i][0] * X1[i][0] + X1[i][1] * X1[i][1]) < sqrt(2.0 / pi)
        assert_equal(y1[i], 1.0 if inside else 0.0)
    var mo = moons_dataset(40, 0.1, 42)
    var MX = mo[0].copy()
    var My = mo[1].copy()
    assert_equal(len(MX), 40)
    var ones = 0
    for v in My:
        if v == 1.0:
            ones += 1
    assert_equal(ones, 20)
    # standard scaling == numpy (population std)
    var S: List[List[Float64]] = [[1.0, 2.0], [3.0, 5.0], [4.0, -1.0], [0.0, 0.5]]
    var ss = standard_scale(S)
    var Xs = ss[0].copy()
    var mean = ss[1].copy()
    var std = ss[2].copy()
    _close(mean[1], 1.625, 1e-15, "mean")
    _close(std[0], 1.5811388300841898, 1e-15, "std0")
    _close(std[1], 2.218529918662356, 1e-15, "std1")
    _close(Xs[1][1], 1.5212776585113297, 1e-15, "scaled")
    _close(Xs[3][1], -0.50709255283711, 1e-15, "scaled")
    var again = apply_standard_scale(S, mean, std)
    assert_equal(again[2][0], Xs[2][0])
    var nr = l2_normalize_rows(S)
    _close(nr[1][0] * nr[1][0] + nr[1][1] * nr[1][1], 1.0, 1e-15, "l2 norm")
    var pr = pad_rows(S, 4, 0.5)
    assert_equal(len(pr[0]), 4)
    assert_equal(pr[0][3], 0.5)
    var tt = train_test_split(100, 0.25, 42)
    var tr = tt[0].copy()
    var te = tt[1].copy()
    assert_equal(len(tr), 75)
    assert_equal(len(te), 25)
    var seen = List[Int](length=100, fill=0)
    for i in tr:
        seen[i] += 1
    for i in te:
        seen[i] += 1
    for v in seen:
        assert_equal(v, 1)
    var sub = take_rows(S, [3, 0])
    assert_equal(sub[0][1], 0.5)
    assert_equal(take_values([5.0, 6.0, 7.0], [2, 2])[1], 7.0)
    # CSV / JSON round trip through a temporary directory
    var d = mkdtemp()
    var csv = d + "/data.csv"
    var js = d + "/data.json"
    with open(csv, "w") as f:
        f.write("f0,f1,label\n# comment\n0.5,-1.25,1\n2,3e-2,-1\n\n")
    with open(js, "w") as f:
        f.write("{\"probe\": {\"X\": [[1, 2.5], [-3, 4e-1, 7]]}, \"flat\": [1, 2, 3]}")
    var l = load_csv(csv)
    var LX = l[0].copy()
    var Ly = l[1].copy()
    assert_equal(len(LX), 2)
    assert_equal(LX[1][1], 0.03)
    assert_equal(Ly[0], 1.0)
    var l2 = load_csv(csv, label_column=0)
    assert_equal(l2[1].copy()[1], 2.0)
    var M = load_json_matrix(js, "probe/X")
    assert_equal(len(M), 2)
    assert_equal(M[1][2], 7.0)
    assert_equal(load_json_matrix(js, "flat")[0][2], 3.0)
    var code = String()
    try:
        _ = load_csv(d + "/missing.csv")
    except e:
        code = _code(String(e))
    assert_equal(code, "QM-E834")
    code = String()
    try:
        _ = load_json_matrix(js, "nope")
    except e:
        code = _code(String(e))
    assert_equal(code, "QM-E834")
    remove(csv)
    remove(js)
    rmdir(d)
    code = String()
    try:
        _ = take_rows(S, [9])
    except e:
        code = _code(String(e))
    assert_equal(code, "QM-E831")


# ---------------------------------------------------------------------------
# Templates and embeddings
# ---------------------------------------------------------------------------
def _parity_classifier() raises -> Circuit:
    """tutorial_variational_classifier: BasisState + 2 x [Rot on every wire + CNOT ring]."""
    var c = Circuit(4, 0, "parity_classifier")
    var w = c.parameters("w", 24)
    var x = c.parameters("x", 4)
    append_basis_embedding(c, x, _wires(4))
    for l in range(2):
        for q in range(4):
            var o = l * 12 + q * 3
            append_rot(c, w[o], w[o + 1], w[o + 2], q)
        c.cx(0, 1)
        c.cx(1, 2)
        c.cx(2, 3)
        c.cx(3, 0)
    return c^


def test_parity_classifier() raises:
    var c = _parity_classifier()
    var p = List[Float64]()
    for l in range(2):
        for q in range(4):
            for k in range(3):
                p.append(0.1 * Float64(l + 1) + 0.01 * Float64(q) + 0.001 * Float64(k))
    for b in [0.0, 1.0, 0.0, 1.0]:
        p.append(b)
    var eng = ExpectationEngine(c, parse_pauli_sum("Z0"))
    var g = List[Float64]()
    var e = eng.value_and_gradient(p, g)
    _close(e, 0.9065742515591111, 1e-12, "parity <Z0>")
    _close(g[1], -0.10562131348797613, 1e-12, "parity dE/dw1")
    _close(g[4], -0.12685347415571607, 1e-12, "parity dE/dw4")
    _close(g[10], -0.12189911931880341, 1e-12, "parity dE/dw10")


def test_mottonen_and_amplitudes() raises:
    var x: List[Float64] = [0.53896774, 0.79503606, 0.27826503, 0.0]
    var ang = mottonen_angles(x)
    var ref_: List[Float64] = [0.5639746511208602, -0.0, 0.0, -0.9750460406377358, 0.9750460406377358]
    assert_equal(len(ang), 5)
    for i in range(5):
        _close(ang[i], ref_[i], 1e-12, "mottonen_angles[" + String(i) + "]")
    assert_equal(len(mottonen_tree_angles(x)), 3)
    # the tree reproduces the (normalised) amplitudes, in QUMO index order for wires [n-1..0]
    var c = Circuit(2)
    append_amplitude_embedding_real(c, x, _rev_wires(2))
    var sv = evolve(c)
    var ref_sv: List[Float64] = [0.5389677431315174, 0.7950360646179606, 0.27826503161645105, 0.0]
    for k in range(4):
        _close(sv.re[k], ref_sv[k], 1e-12, "amplitude[" + String(k) + "]")
        _close(sv.im[k], 0.0, 1e-15, "amplitude imag")
    # signed 6-qubit vector, padding: tree == amplitude_state, 2^n - 2 CX
    var a = List[Float64]()
    for k in range(50):
        a.append(Float64((k * 37) % 11) - 5.0 + 0.25 * sin(Float64(k)))
    var c6 = Circuit(6)
    append_amplitude_embedding_real(c6, a, _rev_wires(6), True, 0.0)
    assert_equal(c6.num_2q_gates(), 62)
    var s6 = evolve(c6)
    var r6 = amplitude_state(a, 6)
    for k in range(64):
        _close(s6.re[k], r6.re[k], 1e-12, "tree vs amplitude_state")
        _close(s6.im[k], 0.0, 1e-12, "tree imag")
    var code = String()
    try:
        var c2 = Circuit(2)
        append_amplitude_embedding_real(c2, [0.6, 0.6, 0.6, 0.6], _wires(2), False)
    except e:
        code = _code(String(e))
    assert_equal(code, "QM-E830")


def test_layer_templates() raises:
    # StronglyEntanglingLayers / BasicEntanglerLayers / AngleEmbedding == PennyLane 0.45.1
    var c = Circuit(3)
    var w = c.parameters("w", 18)
    append_strongly_entangling_layers(c, w, _wires(3))
    var p = List[Float64]()
    for l in range(2):
        for i in range(3):
            for k in range(3):
                p.append(sin(0.3 * Float64(l) + 0.7 * Float64(i) + 1.1 * Float64(k) + 0.2))
    var sv = evolve(c, p)
    _close(sv.expectation(parse_pauli_sum("Z0 Z2")), 0.5068288125525742, 1e-12, "SEL <Z0Z2>")
    _close(sv.expectation(parse_pauli_sum("X1")), 0.10223294126435284, 1e-12, "SEL <X1>")
    _close(sv.expectation(parse_pauli_sum("Y0")), -0.2280048980605785, 1e-12, "SEL <Y0>")
    var c2 = Circuit(3)
    var w2 = c2.parameters("w", 18)
    append_strongly_entangling_layers(c2, w2, _wires(3), [2, 1], "cz")
    var sv2 = evolve(c2, p)
    _close(sv2.expectation(parse_pauli_sum("Z0 Z2")), 0.10200884299523899, 1e-12, "SEL ranges <Z0Z2>")
    _close(sv2.expectation(parse_pauli_sum("Y0")), 0.557731076259504, 1e-12, "SEL ranges <Y0>")
    var cb = Circuit(3)
    var bw = cb.parameters("b", 9)
    var bx = cb.parameters("x", 3)
    append_angle_embedding(cb, bx, _wires(3), "y")
    append_basic_entangler_layers(cb, bw, _wires(3), "y")
    var pb = List[Float64]()
    for l in range(3):
        for i in range(3):
            pb.append(cos(0.4 * Float64(l) + 0.9 * Float64(i) + 0.1))
    for v in [0.3, -1.1, 2.0]:
        pb.append(v)
    var svb = evolve(cb, pb)
    _close(svb.expectation(parse_pauli_sum("Z1")), -0.33326326783648486, 1e-12, "BEL <Z1>")
    _close(svb.expectation(parse_pauli_sum("X0 X2")), -0.38548688962852484, 1e-12, "BEL <X0X2>")
    var c22 = Circuit(2)
    var w22 = c22.parameters("w", 4)
    append_basic_entangler_layers(c22, w22, _wires(2))
    var sv22 = evolve(c22, [0.2, -0.4, 1.3, 0.7])
    _close(sv22.expectation(parse_pauli_sum("Z0")), 0.2621666615466402, 1e-12, "BEL2 <Z0>")
    _close(sv22.expectation(parse_pauli_sum("Y1")), -0.07905130911680425, 1e-12, "BEL2 <Y1>")
    # ArbitraryUnitary: PennyLane Gray-code word order + PauliRot semantics
    var words = pauli_words_gray(2)
    var expect: List[String] = ["XI", "YI", "ZI", "ZX", "IX", "XX", "YX", "YY", "ZY", "IY", "XY", "XZ", "YZ", "ZZ", "IZ"]
    assert_equal(len(words), 15)
    for i in range(15):
        assert_equal(words[i], expect[i])
    assert_equal(len(pauli_words_gray(3)), 63)
    var ca = Circuit(2)
    var aw = ca.parameters("a", 15)
    ca.h(0)
    append_arbitrary_unitary(ca, aw, _wires(2))
    var pa = List[Float64]()
    for i in range(15):
        pa.append(sin(0.5 * Float64(i) + 0.3))
    var sva = evolve(ca, pa)
    _close(sva.expectation(parse_pauli_sum("X0 Z1")), -0.13925118099049347, 1e-12, "ArbitraryUnitary <X0Z1>")
    _close(sva.expectation(parse_pauli_sum("Y1")), 0.462408441890908, 1e-12, "ArbitraryUnitary <Y1>")
    # Qiskit ZZFeatureMap(3, reps=2) (qiskit 2.5 zz_feature_map)
    var zz = zz_feature_map(3)
    var zx: List[Float64] = [0.3, -1.1, 2.0]
    var zy: List[Float64] = [1.2, 0.4, -0.7]
    var szx = evolve(zz, zz_feature_values(zx))
    var szy = evolve(zz, zz_feature_values(zy))
    var qk: List[Float64] = [0.424533091180221, 0.07578954115519028, 0.12861547288786584, 0.1803411758267776, 0.12443275297258286, 0.022390493142058862, 0.031476165825379925, 0.012421307009922854]
    var probs = szx.probabilities()
    for k in range(8):
        _close(probs[k], qk[k], 1e-12, "ZZFeatureMap prob[" + String(k) + "]")
    _close(szx.fidelity(szy), 0.07593161074622826, 1e-12, "ZZFeatureMap kernel")
    # AngleEmbedding kernel = prod cos^2(delta/2); IQP(n_repeats=2) kernel (iris pair, p2 fixture)
    var ae = Circuit(4, 0, "angle")
    var af = ae.parameters("x", 4)
    append_angle_embedding(ae, af, _wires(4))
    var xa: List[Float64] = [0.3586633205653358, 2.7312471783076395, -0.9436431058276776, -0.6864416474328721]
    var xb: List[Float64] = [-0.7376874409881049, 0.8418371395091182, -1.0823121882417375, -0.8642762711201448]
    var KA: List[List[Float64]] = [xa.copy()]
    var KB: List[List[Float64]] = [xb.copy()]
    var k_ang = kernel_matrix(ae, KA, KB)
    _close(k_ang[0], 0.24695866074501246, 1e-12, "angle kernel")
    var analytic = 1.0
    for i in range(4):
        analytic *= cos((xa[i] - xb[i]) / 2.0) ** 2
    _close(k_ang[0], analytic, 1e-14, "angle kernel analytic")
    var iqp = iqp_embedding(4, 2)
    assert_equal(iqp.num_params(), 10)
    var IA: List[List[Float64]] = [iqp_feature_values(xa)]
    var IB: List[List[Float64]] = [iqp_feature_values(xb)]
    _close(kernel_matrix(iqp, IA, IB)[0], 0.280132222844837, 1e-12, "IQP kernel")
    var code = String()
    try:
        var cc = Circuit(3)
        var ww = cc.parameters("w", 10)
        append_strongly_entangling_layers(cc, ww, _wires(3))
    except e:
        code = _code(String(e))
    assert_equal(code, "QM-E830")


def test_controlled_u3() raises:
    # dense reference on a random 3-qubit state, control 2, target 0
    var th = 0.7
    var ph = -1.3
    var la = 2.1
    var prep = Circuit(3)
    prep.h(0)
    prep.ry(0.4, 1)
    prep.h(2)
    prep.cx(2, 1)
    prep.rz(0.9, 2)
    prep.rx(-0.3, 0)
    var psi = evolve(prep)
    var c = Circuit(3)
    append_controlled_u3(c, 2, 0, th, ph, la)
    var out = psi.copy()
    evolve_into(out, c, List[Float64]())
    var m = u3_matrix(th, ph, la)
    for i in range(8):
        if (i & 4) == 0 or (i & 1) == 1:
            continue
        var j = i | 1
        var a0r = psi.re[i]
        var a0i = psi.im[i]
        var a1r = psi.re[j]
        var a1i = psi.im[j]
        var r0 = m.re[0] * a0r - m.im[0] * a0i + m.re[1] * a1r - m.im[1] * a1i
        var i0 = m.re[0] * a0i + m.im[0] * a0r + m.re[1] * a1i + m.im[1] * a1r
        var r1 = m.re[2] * a0r - m.im[2] * a0i + m.re[3] * a1r - m.im[3] * a1i
        var i1 = m.re[2] * a0i + m.im[2] * a0r + m.re[3] * a1i + m.im[3] * a1r
        _close(out.re[i], r0, 1e-14, "CU3 re")
        _close(out.im[i], i0, 1e-14, "CU3 im")
        _close(out.re[j], r1, 1e-14, "CU3 re")
        _close(out.im[j], i1, 1e-14, "CU3 im")
    for i in range(8):
        if (i & 4) == 0:
            _close(out.re[i], psi.re[i], 1e-15, "CU3 control 0")
            _close(out.im[i], psi.im[i], 1e-15, "CU3 control 0")
    # symbolic angles: adjoint gradient == parameter shift
    var s = Circuit(2)
    var q = s.parameters("u", 3)
    s.h(0)
    s.ry(0.3, 1)
    append_controlled_u3(s, 0, 1, q[0], q[1], q[2])
    var h = parse_pauli_sum("X1 + 0.5 * Y0 Y1 + 0.2 * Z0")
    var e = ExpectationEngine(s, h)
    var pv: List[Float64] = [0.7, -1.3, 2.1]
    var ga = e.gradient(pv, Grad.ADJOINT)
    var gp = e.gradient(pv, Grad.PARAM_SHIFT)
    for i in range(3):
        _close(ga[i], gp[i], 1e-12, "CU3 adjoint vs parameter shift")


def test_observables() raises:
    var rng = Rng(5)
    var vals = rng.normal_list(16, 1.0)
    var d = diagonal_observable(vals)
    for i in range(16):
        _close(StateVector.basis(4, i).expectation(d), vals[i], 1e-13, "diagonal_observable")
    var op = PauliOperator(d, 4)
    var b = StateVector.basis(4, 11)
    _close(op.expectation(b), vals[11], 1e-13, "diagonal_observable via PauliOperator")
    var prep = Circuit(5)
    for q in range(5):
        prep.ry(0.3 + 0.2 * Float64(q), q)
    prep.cx(0, 3)
    prep.rx(0.8, 4)
    var psi = evolve(prep)
    var p0 = psi.re[0] * psi.re[0] + psi.im[0] * psi.im[0]
    _close(psi.expectation(projector_zero(5)), p0, 1e-14, "projector_zero")
    var p13 = psi.re[13] * psi.re[13] + psi.im[13] * psi.im[13]
    _close(psi.expectation(projector_basis(5, 13)), p13, 1e-14, "projector_basis")
    var loc = 0.0
    for j in range(5):
        loc += 1.0 - psi.prob_one(j)
    _close(psi.expectation(local_projector_zero(5)), loc / 5.0, 1e-14, "local_projector_zero")
    var code = String()
    try:
        _ = diagonal_observable(List[Float64](length=1 << 13, fill=1.0))
    except e:
        code = _code(String(e))
    assert_equal(code, "QM-E830")


def _qcnn() raises -> Circuit:
    """tutorial_learning_few_data QCNN on 6 qubits (PennyLane wire w = qubit 5-w).
    Parameters: W (18 x 2, W[k][j] at 2k + j), then last (15)."""
    var c = Circuit(6, 0, "qcnn")
    var W = c.parameters("W", 36)
    var L = c.parameters("L", 15)
    var wires = _rev_wires(6)
    for j in range(2):
        var kw = List[Param]()
        for k in range(15):
            kw.append(W[2 * k + j])
        append_qcnn_conv(c, kw, wires, j != 0)
        var pw: List[Param] = [W[30 + j], W[32 + j], W[34 + j]]
        var sinks = List[Int]()
        var sources = List[Int]()
        for i in range(1, len(wires), 2):
            sinks.append(wires[i - 1])
            sources.append(wires[i])
        append_qcnn_pool(c, pw, sinks, sources)
        var nw = List[Int]()
        for i in range(0, len(wires), 2):
            nw.append(wires[i])
        wires = nw^
    append_arbitrary_unitary(c, L, wires)
    return c^


def _qcnn_features() -> List[Float64]:
    var f: List[Float64] = [0.0, 0.0, 0.0902403594605542, 0.2346249345974409, 0.16243264702899754, 0.01804807189211084, 0.0, 0.0, 0.0, 0.0, 0.2346249345974409, 0.2707210783816626, 0.1804807189211084, 0.2707210783816626, 0.0902403594605542, 0.0, 0.0, 0.05414421567633252, 0.2707210783816626, 0.03609614378422168, 0.0, 0.19852879081321922, 0.1443845751368867, 0.0, 0.0, 0.07219228756844336, 0.21657686270533008, 0.0, 0.0, 0.1443845751368867, 0.1443845751368867, 0.0, 0.0, 0.0902403594605542, 0.1443845751368867, 0.0, 0.0, 0.16243264702899754, 0.1443845751368867, 0.0, 0.0, 0.07219228756844336, 0.19852879081321922, 0.0, 0.01804807189211084, 0.21657686270533008, 0.12633650324477588, 0.0, 0.0, 0.03609614378422168, 0.25267300648955177, 0.0902403594605542, 0.1804807189211084, 0.21657686270533008, 0.0, 0.0, 0.0, 0.0, 0.10828843135266504, 0.2346249345974409, 0.1804807189211084, 0.0, 0.0, 0.0]
    return f^


def test_qcnn() raises:
    var c = _qcnn()
    var p = List[Float64]()
    for k in range(18):
        for j in range(2):
            p.append(sin(0.37 * Float64(k) + 1.3 * Float64(j) + 0.1))
    for i in range(15):
        p.append(cos(0.21 * Float64(i) + 0.4))
    var feats = _qcnn_features()
    var eng = ExpectationEngine(c, parse_pauli_sum("0.5 + 0.5 * Z5"))  # P(PennyLane wire 0 = 0)
    eng.set_initial_state(amplitude_state(feats, 6))
    var g = List[Float64]()
    var p0 = eng.value_and_gradient(p, g)
    _close(p0, 0.4247292672981652, 1e-11, "QCNN p(0)")
    _close(1.0 - p0, 0.5752707327018325, 1e-11, "QCNN p(1)")
    var gW: List[Float64] = [0.11922381454904189, 0.0, -0.013352097461732264, 0.0, -0.020694974010971097, 0.0]
    var gL: List[Float64] = [-0.008553214411840619, 0.056564330260893775, -0.022044646147057638, 0.07061076156027338, 0.0905113327880776, -0.1732023239343262]
    for i in range(6):
        _close(g[i], gW[i], 1e-10, "QCNN dp/dW")
        _close(g[36 + i], gL[i], 1e-10, "QCNN dp/dlast")
    # the Mottonen-prepared circuit gives the same value and weight gradient
    var full = Circuit(6, 0, "qcnn_mottonen")
    append_amplitude_embedding_real(full, feats, _rev_wires(6))
    full.compose(c)
    var e2 = ExpectationEngine(full, parse_pauli_sum("0.5 + 0.5 * Z5"))
    var g2 = List[Float64]()
    _close(e2.value_and_gradient(p, g2), p0, 1e-12, "QCNN Mottonen value")
    for i in range(len(g)):
        _close(g2[i], g[i], 1e-12, "QCNN Mottonen gradient")
    # finite differences through the initial-state engine
    var h = 1e-6
    for i in [0, 7, 40]:
        var pp = p.copy()
        pp[i] += h
        var fp = eng.value(pp)
        pp[i] -= 2.0 * h
        var fm = eng.value(pp)
        _close(g[i], (fp - fm) / (2.0 * h), 1e-7, "QCNN initial-state FD")
    # width mismatch
    var code = String()
    try:
        eng.set_initial_state(StateVector(5))
    except e:
        code = _code(String(e))
    assert_equal(code, "QM-E804")
    eng.clear_initial_state()
    var zero_start = eng.value(p)
    var plain = ExpectationEngine(c, parse_pauli_sum("0.5 + 0.5 * Z5"))
    _close(zero_start, plain.value(p), 1e-15, "clear_initial_state")


# ---------------------------------------------------------------------------
# Batched value and gradient
# ---------------------------------------------------------------------------
def _batch_case(n: Int, layers: Int, nb: Int) raises:
    var c = Circuit(n, 0, "sel")
    var w = c.parameters("w", layers * n * 3)
    var x = c.parameters("x", n)
    append_angle_embedding(c, x, _wires(n))
    append_strongly_entangling_layers(c, w, _wires(n))
    var h = parse_pauli_sum("Z0 + 0.5 * X1 Z0 - 0.25 * Y" + String(n - 1))
    var rng = Rng(11)
    var sets = List[List[Float64]]()
    for _ in range(nb):
        sets.append(rng.normal_list(c.num_params(), 1.0))
    var t0 = perf_counter_ns()
    var r = batch_value_and_gradient(c, h, sets)
    var t1 = perf_counter_ns()
    var eng = ExpectationEngine(c, h)
    var g = List[Float64]()
    var md = 0.0
    for b in range(nb):
        var v = eng.value_and_gradient(sets[b], g)
        md = max(md, abs(v - r[0][b]))
        for k in range(len(g)):
            md = max(md, abs(g[k] - r[1][b][k]))
    var t2 = perf_counter_ns()
    assert_true(md <= 1e-14, "parallel != sequential: " + String(md))
    var gs = batch_gradient(c, h, sets)
    for k in range(c.num_params()):
        assert_equal(gs[nb - 1][k], r[1][nb - 1][k])
    print("  batch_value_and_gradient", n, "qubits,", c.num_params(), "params,", nb, "sets: parallel",
          Float64(t1 - t0) / 1e6, "ms, sequential", Float64(t2 - t1) / 1e6, "ms")


def test_batch_value_and_gradient() raises:
    _batch_case(4, 2, 64)
    _batch_case(10, 2, 64)
    # one initial state per set (amplitude-embedded data) == per-set engines
    var c = Circuit(3, 0, "on_data")
    var w = c.parameters("w", 9)
    append_strongly_entangling_layers(c, w, _wires(3))
    var h = parse_pauli_sum("Z0 Z1 + X2")
    var rng = Rng(3)
    var sets = List[List[Float64]]()
    var states = List[StateVector]()
    for _ in range(8):
        sets.append(rng.normal_list(9, 1.0))
        states.append(amplitude_state(rng.normal_list(8, 1.0), 3))
    var r = batch_value_and_gradient(c, h, sets, Grad.ADJOINT, states)
    var eng = ExpectationEngine(c, h)
    for b in range(8):
        eng.set_initial_state(states[b])
        var g = List[Float64]()
        _close(eng.value_and_gradient(sets[b], g), r[0][b], 1e-14, "initial-state batch value")
        for k in range(9):
            _close(g[k], r[1][b][k], 1e-14, "initial-state batch gradient")
        var gps = eng.gradient(sets[b], Grad.PARAM_SHIFT)
        for k in range(9):
            _close(gps[k], r[1][b][k], 1e-12, "initial-state parameter shift")
    var code = String()
    try:
        _ = batch_value_and_gradient(c, h, sets, Grad.ADJOINT, [StateVector(3)])
    except e:
        code = _code(String(e))
    assert_equal(code, "QM-E804")


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------
def _quad_grad(x: List[Float64]) -> List[Float64]:
    var c: List[Float64] = [1.0, 3.0, 0.5]
    var t: List[Float64] = [0.1, 0.2, -0.4]
    var g = List[Float64]()
    for i in range(3):
        g.append(2.0 * c[i] * (x[i] - t[i]))
    return g^


def _check_traj(var opt: StepOptimizer, ref_: List[Float64], tol: Float64, name: String) raises:
    var x: List[Float64] = [0.3, -0.7, 1.2]
    for _ in range(3):
        var la = opt.lookahead(x)
        opt.step(x, _quad_grad(la))
    for i in range(3):
        _close(x[i], ref_[i], tol, name)


def test_step_optimizers() raises:
    # 3 steps on sum c_i (x_i - t_i)^2 vs PennyLane 0.45.1 optimizers / the optax Adam rule
    _check_traj(StepOptimizer.gd(0.1), [0.2024, 0.14240000000000003, 0.7664], 1e-15, "gd")
    _check_traj(StepOptimizer.momentum(0.1, 0.9), [0.1124, 0.9686, 0.37759999999999994], 1e-15, "momentum")
    _check_traj(StepOptimizer.nesterov(0.1, 0.9), [0.13558399999999998, 0.36790399999999995, 0.4281439999999999], 1e-15, "nesterov")
    _check_traj(StepOptimizer.adam(0.1), [0.03245877419914536, -0.40175465414636075, 0.9007749993384057], 1e-14, "adam pennylane")
    _check_traj(StepOptimizer.adam(0.05, 0.8, 0.999, 1e-6), [0.1553246367269318, -0.5506670650058085, 1.0503502321214155], 1e-14, "adam pennylane b2")
    _check_traj(StepOptimizer.adam(0.1, 0.9, 0.999, 1e-8, "optax"), [0.03281908063038802, -0.4018543153451126, 0.9008293223237975], 1e-14, "adam optax")
    _check_traj(StepOptimizer.adagrad(0.1), [0.13127974975805226, -0.48153803160310904, 0.9769449103310481], 1e-14, "adagrad")
    _check_traj(StepOptimizer.rmsprop(0.1), [0.07751177849012493, -0.08456719076036893, 0.522465054366575], 1e-14, "rmsprop")
    _close(cosine_lr(37, 100, 0.1, 0.01), 0.07287165507856512, 1e-16, "cosine_lr")
    _close(cosine_lr(150, 100, 0.1, 0.01), 0.010000000000000002, 1e-16, "cosine_lr end")
    _close(step_lr(25, 10, 0.5, 0.2), 0.05, 1e-16, "step_lr")
    var opt = StepOptimizer.adam(0.1)
    var x: List[Float64] = [1.0, 2.0]
    opt.step(x, [0.1, 0.2])
    var code = String()
    try:
        var y: List[Float64] = [1.0, 2.0, 3.0]
        opt.step(y, [0.1, 0.2, 0.3])
    except e:
        code = _code(String(e))
    assert_equal(code, "QM-E833")
    opt.reset()
    var y2: List[Float64] = [1.0, 2.0, 3.0]
    opt.step(y2, [0.1, 0.2, 0.3])
    assert_equal(opt.t, 1)


def test_parity_training() raises:
    """30 full-batch Nesterov(0.5, m=0.9) steps on all 16 parity strings from the
    PennyLane seed-0 init: the cost trajectory equals PennyLane's."""
    var c = _parity_classifier()
    var h = parse_pauli_sum("Z0")
    var ds = parity_dataset(4)
    var X = ds[0].copy()
    var Y = ds[1].copy()
    var theta: List[Float64] = [0.017640523459676642, 0.004001572083672233, 0.009787379841057393, 0.022408931992014578, 0.018675579901499675, -0.00977277879876411, 0.009500884175255894, -0.001513572082976979, -0.0010321885179355784, 0.004105985019383723, 0.0014404357116087798, 0.014542735069629751, 0.007610377251469934, 0.001216750164928284, 0.004438632327454257, 0.0033367432737426684, 0.01494079073157606, -0.0020515826376580086, 0.003130677016509014, -0.008540957393017247, -0.025529898158340787, 0.006536185954403606, 0.008644361988595054, -0.007421650204064417, 0.0]
    var pl: List[Float64] = [1.9992624749350514, 1.993817588690902, 1.9394271984930849, 1.5299241662064444, 0.9963491157316887, 1.0094989265575647, 1.0428524518719673, 1.0198221507792637, 1.0003709327546735, 1.012033128139989, 1.0152117273073973, 0.9991487683559752, 0.9844859792386106, 0.9449075787970994, 0.7593303839875251, 0.12905922347167068, 0.000610489698550953, 0.07821549413988442, 0.16268511908059444, 0.08595922375968176, 0.02637935153099016, 0.006630865864154612, 0.0017412270592543915, 0.0008081983064602694, 0.0011830165693697402, 0.002331396605205784, 0.0034836878524683586, 0.0037880034663540377, 0.003287231942591003, 0.002525821086497758]
    var opt = StepOptimizer.nesterov(0.5, 0.9)
    var t0 = perf_counter_ns()
    var last = 0.0
    for step in range(30):
        var la = opt.lookahead(theta)
        var sets = List[List[Float64]]()
        for s in range(16):
            var p = List[Float64]()
            for i in range(24):
                p.append(la[i])
            for q in range(4):
                p.append(X[s][q])
            sets.append(p^)
        var r = batch_value_and_gradient(c, h, sets)
        var preds = List[Float64]()
        for s in range(16):
            preds.append(r[0][s] + la[24])
        var L = square_loss(preds, Y)
        var dl = L[1].copy()
        _close(L[0], pl[step], 1e-10, "parity Nesterov cost at step " + String(step))
        last = L[0]
        var grad = List[Float64](length=25, fill=0.0)
        for s in range(16):
            for i in range(24):
                grad[i] += dl[s] * r[1][s][i]
            grad[24] += dl[s]
        opt.step(theta, grad)
    _close(pl[0], 1.9992624749350514, 0.0, "first value")
    _close(last, 0.002525821086497758, 1e-10, "final value")
    print("  parity Nesterov 30 steps x 16 adjoint gradients:", Float64(perf_counter_ns() - t0) / 1e6, "ms")


def test_batcher() raises:
    var rec: List[List[Int]] = [[7, 12, 2, 0, 0], [4, 5, 5, 6, 8], [4, 1, 15, 4, 9]]
    var b = Batcher.replay(rec)
    for k in range(4):
        var got = b.next()
        var want = rec[k % 3].copy()
        assert_equal(len(got), 5)
        for i in range(5):
            assert_equal(got[i], want[i])
    # epochs: each epoch is a permutation, deterministic per seed, last partial batch kept
    var e1 = Batcher(10, 4, 5)
    var e2 = Batcher(10, 4, 5)
    assert_equal(e1.batches_per_epoch(), 3)
    var seen = List[Int](length=10, fill=0)
    for k in range(3):
        var x1 = e1.next()
        var x2 = e2.next()
        assert_equal(len(x1), 4 if k < 2 else 2)
        for i in range(len(x1)):
            assert_equal(x1[i], x2[i])
            seen[x1[i]] += 1
    for v in seen:
        assert_equal(v, 1)
    var dl = Batcher(10, 4, 5, True, True)
    assert_equal(dl.batches_per_epoch(), 2)
    for _ in range(5):
        assert_equal(len(dl.next()), 4)
    var wr = Batcher(10, 7, 1, True, False, True)
    for i in wr.next():
        assert_true(i >= 0 and i < 10)


def _fd_check(name: String, kind: Int, pred: List[Float64], y: List[Float64]) raises:
    var r = _loss(kind, pred, y)
    var g = r[1].copy()
    var h = 1e-6
    for i in range(len(pred)):
        var pp = pred.copy()
        pp[i] += h
        var fp = _loss(kind, pp, y)[0]
        pp[i] -= 2.0 * h
        var fm = _loss(kind, pp, y)[0]
        _close(g[i], (fp - fm) / (2.0 * h), 1e-8, name + " gradient")


def _loss(kind: Int, pred: List[Float64], y: List[Float64]) raises -> Tuple[Float64, List[Float64]]:
    if kind == 0:
        return square_loss(pred, y)
    if kind == 1:
        return hinge_loss(pred, y)
    if kind == 2:
        return l1_loss(pred, y)
    return fidelity_loss(pred)


def test_losses_and_vjp() raises:
    var pred: List[Float64] = [0.3, -0.8, 0.45, 1.7, -0.2]
    var y: List[Float64] = [1.0, -1.0, -1.0, 1.0, 1.0]
    _close(square_loss(pred, y)[0], (0.49 + 0.04 + 2.1025 + 0.49 + 1.44) / 5.0, 1e-15, "square_loss")
    _close(hinge_loss(pred, y)[0], (0.7 + 0.2 + 1.45 + 0.0 + 1.2) / 5.0, 1e-15, "hinge_loss")
    _close(l1_loss(pred, y)[0], (0.7 + 0.2 + 1.45 + 0.7 + 1.2) / 5.0, 1e-15, "l1_loss")
    _fd_check("square_loss", 0, pred, y)
    _fd_check("hinge_loss", 1, pred, y)
    _fd_check("l1_loss", 2, pred, y)
    _fd_check("fidelity_loss", 3, [0.9, 0.3, 0.55], y)
    var probs: List[List[Float64]] = [[0.7, 0.2, 0.1], [0.1, 0.6, 0.3]]
    var ce = cross_entropy(probs, [0, 2])
    _close(ce[0], (0.35667494393873245 + 1.2039728043259361) / 2.0, 1e-9, "cross_entropy")  # std.math.log is accurate to ~1e-9 here
    var cg = ce[1].copy()
    _close(cg[0], -1.0 / (2.0 * 0.7), 1e-15, "cross_entropy grad")
    _close(cg[5], -1.0 / (2.0 * 0.3), 1e-15, "cross_entropy grad")
    _close(cg[1], 0.0, 0.0, "cross_entropy grad off-label")
    _close(accuracy(pred, y), 0.6, 1e-15, "accuracy")
    var code = String()
    try:
        _ = square_loss(pred, [1.0])
    except e:
        code = _code(String(e))
    assert_equal(code, "QM-E831")
    # vjp == sum_j u_j grad <O_j>, from one adjoint pass
    var c = Circuit(3, 0, "layer")
    var w = c.parameters("w", 9)
    var x = c.parameters("x", 3)
    append_angle_embedding(c, x, _wires(3))
    append_strongly_entangling_layers(c, w, _wires(3))
    var obs: List[PauliSum] = [parse_pauli_sum("Z0"), parse_pauli_sum("X1 Z2"), parse_pauli_sum("Y2")]
    var u: List[Float64] = [0.7, -1.2, 0.4]
    var rng = Rng(8)
    var p = rng.normal_list(12, 1.0)
    var v = vjp(c, obs, p, u)
    var ref_ = List[Float64](length=12, fill=0.0)
    for j in range(3):
        var gj = gradient(c, obs[j], p)
        for k in range(12):
            ref_[k] += u[j] * gj[k]
    for k in range(12):
        _close(v[k], ref_[k], 1e-13, "vjp")
    var ev = expectations(c, obs, p)
    var sv = evolve(c, p)
    _close(ev[1], sv.expectation(obs[1]), 1e-15, "expectations")


def main() raises:
    test_rng_normal()
    test_datasets()
    test_parity_classifier()
    test_mottonen_and_amplitudes()
    test_layer_templates()
    test_controlled_u3()
    test_observables()
    test_qcnn()
    test_batch_value_and_gradient()
    test_step_optimizers()
    test_parity_training()
    test_batcher()
    test_losses_and_vjp()
    print("test_qml OK")
