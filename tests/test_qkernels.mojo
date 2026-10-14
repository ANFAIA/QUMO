"""Tests for qumo.algorithms.qkernels: fidelity kernels (state, compute-uncompute
and sampled routes), kernel-target alignment and its gradient, SMO C-SVC and
kernel ridge regression. References: PennyLane 0.45.1 tutorial_kernels_module
(DoubleCake, seed 1359 init) and sklearn 1.9.1 SVC(kernel='precomputed').
Prints `test_qkernels OK`."""

from std.math import sqrt
from std.time import perf_counter_ns
from std.testing import assert_true, assert_equal

from qumo.ir.circuit import Circuit
from qumo.algorithms.datasets import double_cake
from qumo.algorithms.qkernels import (
    embedding_states, kernel_matrix, kernel_matrix_symmetric, kernel_circuit, kernel_circuit_params,
    kernel_value_and_grad, kernel_matrix_symmetric_and_grad, sampled_kernel_entry, target_alignment,
    target_alignment_grad, SVCModel, svc_fit, svc_decision, svc_predict, kernel_ridge_fit,
    kernel_ridge_predict,
)


def _code(msg: String) -> String:
    return String(msg[byte=0:7]) if msg.byte_length() >= 7 else msg


def _close(a: Float64, b: Float64, tol: Float64, what: String) raises:
    assert_true(abs(a - b) <= tol, what + ": " + String(a) + " vs " + String(b) + " (|diff| " + String(abs(a - b)) + " > " + String(tol) + ")")


def _doublecake_embedding() raises -> Circuit:
    """5 qubits x 6 layers of [H, RZ(x_(i mod 2)), RY(theta)] + CRZ ring; params th (60) then x (2)."""
    var c = Circuit(5, 0, "doublecake")
    var th = c.parameters("th", 60)
    var x = c.parameters("x", 2)
    for l in range(6):
        var i = l * 5
        for j in range(5):
            c.h(j)
            c.rz(x[i % 2], j)
            i += 1
            c.ry(th[l * 10 + j], j)
        for j in range(5):
            c.crz(th[l * 10 + 5 + j], j, (j + 1) % 5)
    return c^


def _init_params() -> List[Float64]:
    var p: List[Float64] = [1.0910562437274183, 5.948599338277643, 1.603961248971189, 3.63190940614966, 4.463391241811286, 4.292595025147145, 3.7073368426921007, 3.8112748112170416, 4.157595557710465, 2.177539977909702, 3.194026548176765, 5.523164674933418, 3.4167377571667013, 2.1144344907019286, 1.3819313700691926, 2.7236254508595805, 4.296258240927783, 4.412381402947051, 1.423170759539888, 1.1338939278448925, 0.9571446674014675, 5.624858798772517, 4.635974295732184, 5.355488870828087, 3.397777498325954, 5.603960814642693, 4.491950445484586, 0.37382161831498073, 6.243919261142066, 1.4055277655994451, 5.134851630375424, 1.3299072661207187, 5.546753081592854, 0.969198519385721, 5.5903323452999665, 2.120136694440747, 2.343401587633935, 4.184221975226072, 3.1883317980581767, 0.4990291679048266, 5.15044509842968, 0.36158135571754374, 5.888539749629254, 0.40702702766435944, 5.6556515290069544, 4.889906066060002, 0.4993484082942385, 4.8872361761201955, 4.450194405556316, 4.625501892435892, 0.35306536448107007, 6.139193246656743, 0.26864776175842503, 1.2863454348380872, 4.124184207598183, 6.0709985373553765, 6.151781504710909, 5.501276474502562, 0.2827118569135146, 1.729785632063598]
    return p^


def test_doublecake() raises:
    var P = _init_params()
    var Kpl: List[Float64] = [1.0, 0.029983453404801606, 0.08562305857611457, 0.6334793498796302, 0.03201456164738065, 0.06621282145765385, 0.029983453404801606, 1.0, 0.12136184717299739, 0.14067907067668126, 0.7013529406387848, 0.18019388683467208, 0.08562305857611457, 0.12136184717299739, 1.0, 0.13731144462501604, 0.25606969220476405, 0.7630558960833492, 0.6334793498796302, 0.14067907067668126, 0.13731144462501604, 1.0, 0.33146356902302626, 0.2736804486512212, 0.03201456164738065, 0.7013529406387848, 0.25606969220476405, 0.33146356902302626, 1.0, 0.47521000917295353, 0.06621282145765385, 0.18019388683467208, 0.7630558960833492, 0.2736804486512212, 0.47521000917295353, 1.0]
    var dc = double_cake(3)
    var X = dc[0].copy()
    var Y = dc[1].copy()
    var e = _doublecake_embedding()
    var K = kernel_matrix_symmetric(e, X, P)
    _close(K[1], 0.029983453404801606, 1e-12, "k01")
    for i in range(36):
        _close(K[i], Kpl[i], 1e-13, "K_init[" + String(i) + "]")
    var K2 = kernel_matrix(e, X, X, P)
    for i in range(36):
        _close(K2[i], Kpl[i], 1e-13, "kernel_matrix[" + String(i) + "]")
    var kc = kernel_circuit(e, 2)
    assert_equal(kc.num_params(), 64)
    assert_equal(kc.param_names[61], "xa[1]")
    assert_equal(kc.param_names[62], "xb[0]")
    var kg = kernel_value_and_grad(kc, P, X[0], X[1])
    var g = kg[1].copy()
    assert_equal(len(g), 60)
    _close(kg[0], 0.029983453404801606, 1e-12, "k01 circuit route")
    var dref: List[Float64] = [0.052756475018638685, -0.009458429342994176, -0.03658170785490919, 0.10397971732213496, 7.249159282602356e-05, -0.01985706213103956, 0.0024777098233813677, -0.04083350723915904, 0.022834177834201656, -0.004356310652688828]
    for i in range(10):
        _close(g[i], dref[i], 1e-12, "dk01/dtheta[" + String(i) + "]")
    var r = kernel_matrix_symmetric_and_grad(kc, P, X)
    var Kc = r[0].copy()
    for i in range(36):
        _close(Kc[i], K[i], 1e-14, "compute-uncompute == state route")
    var kta = target_alignment(Kc, Y, True)
    _close(kta, 0.07433831454242083, 1e-12, "KTA_init")
    _close(target_alignment(K, Y, True), kta, 1e-14, "KTA state route")
    var dkta = target_alignment_grad(Kc, r[1], Y, True)
    var negref: List[Float64] = [-0.0013314018593972468, -0.001152780240660053, 0.006750920338074614, -0.004562968571560648, -0.009073859916854901, -0.0008528821401623445, -0.0013627708365610428, -0.006901040002608863, 0.0033127891955509626, 0.0018904726981384666]
    for i in range(10):
        _close(dkta[i], -negref[i], 1e-12, "dKTA/dtheta[" + String(i) + "]")
    # finite-difference check of the full-matrix KTA gradient (no normalisation assumption)
    var h = 1e-6
    var pp = P.copy()
    pp[7] += h
    var kp = target_alignment(kernel_matrix_symmetric(e, X, pp), Y)
    pp[7] -= 2.0 * h
    var km = target_alignment(kernel_matrix_symmetric(e, X, pp), Y)
    _close(target_alignment_grad(Kc, r[1], Y)[7], (kp - km) / (2.0 * h), 1e-8, "dKTA FD")
    # sampled (QPU) route within 4 sigma
    var shots = 20000
    var s = sampled_kernel_entry(kc, kernel_circuit_params(P, X[0], X[1]), shots, 7)
    var k01 = 0.029983453404801606
    var sigma = sqrt(k01 * (1.0 - k01) / Float64(shots))
    assert_true(abs(s - k01) < 4.0 * sigma, "sampled kernel " + String(s) + " vs " + String(k01))
    # timing (printed, not asserted)
    var t0 = perf_counter_ns()
    var acc = 0.0
    for it in range(3000):
        acc += kernel_value_and_grad(kc, P, X[it % 6], X[(it + 1) % 6])[0]
    var t1 = perf_counter_ns()
    print("  DoubleCake: 3000 kernel-entry adjoint gradients (5 qubits, 64 params):", Float64(t1 - t0) / 1e9, "s")
    var code = String()
    try:
        _ = kernel_matrix(e, [[0.1, 0.2, 0.3]], X, P)
    except e2:
        code = _code(String(e2))
    assert_equal(code, "QM-E831")


def test_svc() raises:
    var K_sep: List[Float64] = [1.0, 0.6376281516217733, 0.5945205479701944, 0.014920786069067842, 0.018224289258677303, 0.009466462401710323, 0.6376281516217733, 1.0, 0.6126263941844161, 0.11474993459727875, 0.09208897957928115, 0.10126646185388335, 0.5945205479701944, 0.6126263941844161, 1.0, 0.09975880065361863, 0.16121764412977677, 0.046887695219988486, 0.014920786069067842, 0.11474993459727875, 0.09975880065361863, 1.0, 0.7710515858035663, 0.8146473164114145, 0.018224289258677303, 0.09208897957928115, 0.16121764412977677, 0.7710515858035663, 1.0, 0.3965314190749929, 0.009466462401710323, 0.10126646185388335, 0.046887695219988486, 0.8146473164114145, 0.3965314190749929, 1.0]
    var y_sep: List[Float64] = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
    var Kt_sep: List[Float64] = [0.7788007830714049, 0.9048374180359595, 0.8436648165963837, 0.09026549560978427, 0.09975880065361867, 0.06020499239237354, 0.03877420783172201, 0.20189651799465547, 0.2165356673160071, 0.9185122844014573, 0.8650222931107413, 0.6376281516217732, 0.23692775868212176, 0.6376281516217733, 0.5945205479701944, 0.48432456895536236, 0.4653339309743134, 0.34645581033005735]
    var dt_sep: List[Float64] = [-1.1625587950287324, 0.9502926804883934, -0.1684266136994529]
    var dtr_sep: List[Float64] = [-0.9999999945163394, -1.0000000177861583, -1.0000000073717927, 1.136538850361648, 0.9999999924008869, 0.9999999869547208]
    var m_sep = svc_fit(K_sep, y_sep, 1.0, 1e-12)
    assert_true(m_sep.converged)
    var d1_sep = svc_decision(m_sep, Kt_sep)
    var d2_sep = svc_decision(m_sep, K_sep)
    for i in range(len(d1_sep)):
        _close(d1_sep[i], dt_sep[i], 1e-6, "svc sep test decision")
    for i in range(len(d2_sep)):
        _close(d2_sep[i], dtr_sep[i], 1e-6, "svc sep train decision")
    var K_soft: List[Float64] = [1.0, 0.6376281516217733, 0.5945205479701944, 0.014920786069067842, 0.018224289258677303, 0.009466462401710323, 0.8146473164114145, 0.03254928098910343, 0.6376281516217733, 1.0, 0.6126263941844161, 0.11474993459727875, 0.09208897957928115, 0.10126646185388335, 0.9185122844014574, 0.19691167520419406, 0.5945205479701944, 0.6126263941844161, 1.0, 0.09975880065361863, 0.16121764412977677, 0.046887695219988486, 0.7985162187593771, 0.17118678648494096, 0.014920786069067842, 0.11474993459727875, 0.09975880065361863, 1.0, 0.7710515858035663, 0.8146473164114145, 0.07730474044329971, 0.9607894391523232, 0.018224289258677303, 0.09208897957928115, 0.16121764412977677, 0.7710515858035663, 1.0, 0.3965314190749929, 0.0820849986238988, 0.7710515858035664, 0.009466462401710323, 0.10126646185388335, 0.046887695219988486, 0.8146473164114145, 0.3965314190749929, 1.0, 0.05366469191273011, 0.7672059499758556, 0.8146473164114145, 0.9185122844014574, 0.7985162187593771, 0.07730474044329971, 0.0820849986238988, 0.05366469191273011, 1.0, 0.14085842092104503, 0.03254928098910343, 0.19691167520419406, 0.17118678648494096, 0.9607894391523232, 0.7710515858035664, 0.7672059499758556, 0.14085842092104503, 1.0]
    var y_soft: List[Float64] = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0]
    var Kt_soft: List[Float64] = [0.7788007830714049, 0.9048374180359595, 0.8436648165963837, 0.09026549560978427, 0.09975880065361867, 0.06020499239237354, 0.9950124791926823, 0.16121764412977677, 0.03877420783172201, 0.20189651799465547, 0.2165356673160071, 0.9185122844014573, 0.8650222931107413, 0.6376281516217732, 0.1580253208896478, 0.9753099120283327, 0.23692775868212176, 0.6376281516217733, 0.5945205479701944, 0.48432456895536236, 0.4653339309743134, 0.34645581033005735, 0.5683601467575405, 0.6537697851298473]
    var dt_soft: List[Float64] = [-0.7379758328517094, 0.5569994865672746, -0.1455222790525082]
    var dtr_soft: List[Float64] = [-0.720055949079277, -0.6266106661484179, -0.6523130707975717, 0.7200559552338195, 0.5872063665064017, 0.6536720431146734, -0.7460762906158591, 0.6032924407280994]
    var m_soft = svc_fit(K_soft, y_soft, 0.5, 1e-12)
    assert_true(m_soft.converged)
    var d1_soft = svc_decision(m_soft, Kt_soft)
    var d2_soft = svc_decision(m_soft, K_soft)
    for i in range(len(d1_soft)):
        _close(d1_soft[i], dt_soft[i], 1e-6, "svc soft test decision")
    for i in range(len(d2_soft)):
        _close(d2_soft[i], dtr_soft[i], 1e-6, "svc soft train decision")
    # XOR with a precomputed (1 + x.y)^2 kernel: accuracy 1
    var xs: List[List[Float64]] = [[1.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]]
    var ys: List[Float64] = [1.0, 1.0, -1.0, -1.0]
    var Kx = List[Float64]()
    for i in range(4):
        for j in range(4):
            var d = 1.0 + xs[i][0] * xs[j][0] + xs[i][1] * xs[j][1]
            Kx.append(d * d)
    var mx = svc_fit(Kx, ys, 10.0)
    var pr = svc_predict(mx, Kx)
    for i in range(4):
        assert_equal(pr[i], ys[i])
    assert_equal(mx.num_support(), 4)
    var code = String()
    try:
        _ = svc_fit(Kx, [1.0, -1.0], 1.0)
    except e:
        code = _code(String(e))
    assert_equal(code, "QM-E831")


def test_kernel_ridge() raises:
    var K: List[Float64] = [2.0, 0.5, 0.1, 0.5, 1.5, 0.3, 0.1, 0.3, 1.0]
    var y: List[Float64] = [1.0, -2.0, 0.5]
    var lam = 0.1
    var a = kernel_ridge_fit(K, y, lam)
    for i in range(3):
        var s = lam * a[i]
        for j in range(3):
            s += K[i * 3 + j] * a[j]
        _close(s, y[i], 1e-10, "(K + lam I) alpha = y")
    var pred = kernel_ridge_predict(a, K)
    _close(pred[1] + lam * a[1], y[1], 1e-10, "kernel_ridge_predict")


def main() raises:
    test_doublecake()
    test_svc()
    test_kernel_ridge()
    print("test_qkernels OK")
