"""qumo.sample / run / statevector / probabilities / expectation."""

import inspect
import math
import pickle

import numpy as np
import pytest

import qumo


def ghz(n: int) -> qumo.Circuit:
    c = qumo.Circuit(n).h(0)
    for i in range(n - 1):
        c.cx(i, i + 1)
    return c.measure_all()


def test_defaults_equal_the_cli():
    for fn in (qumo.sample, qumo.run):
        sig = inspect.signature(fn)
        assert sig.parameters["shots"].default == 1024
        assert sig.parameters["seed"].default == 1234
    sig = inspect.signature(qumo.transpile)
    assert sig.parameters["level"].default == 2 and sig.parameters["seed"].default == 7


def test_sample_bell(bell):
    counts = qumo.sample(bell)
    assert isinstance(counts, dict) and isinstance(counts, qumo.Counts)
    assert set(counts) == {"00", "11"} and sum(counts.values()) == 1024
    assert counts.shots == 1024 and counts.seed == 1234 and counts.method.startswith("statevector")
    assert counts == qumo.sample(bell, shots=1024, seed=1234) == qumo.run(bell)
    assert counts != qumo.sample(bell, seed=1)
    assert abs(counts.probabilities()["00"] - 0.5) < 0.1
    assert counts.most_common(1)[0][1] == max(counts.values())
    assert "1024 shots" in str(counts) and "#" in str(counts)
    back = pickle.loads(pickle.dumps(counts))
    assert back == counts and back.method == counts.method and back.shots == 1024


def test_sample_matches_raw_extension(q, bell):
    assert dict(qumo.sample(bell, 2000, 9)) == dict(q.sample_device(bell.native, 2000, 9, "auto")[0])
    assert dict(qumo.sample(bell, 2000, 9, method="statevector")) == dict(q.sample(bell.native, 2000, 9))


def test_ghz_40_qubits_uses_automatic_dispatch(q):
    """Regression: qumo_py.sample fails with QM-E404 at 40 qubits; the facade dispatches like the CLI."""
    c = ghz(40)
    with pytest.raises(Exception, match="QM-E404"):
        q.sample(c.native, 100, 1)
    counts = qumo.sample(c, shots=200, seed=3)
    assert set(counts) == {"0" * 40, "1" * 40} and sum(counts.values()) == 200
    assert counts.method.startswith("stabilizer") or counts.method.startswith("frame")
    assert qumo.sample(qumo.ghz(40, measure=True), shots=50).shots == 50


def test_non_clifford_40_qubits_explains_the_limit():
    c = qumo.Circuit(40).h(0).t(0).rx(0.3, 1).cx(0, 1).measure_all()
    with pytest.raises(qumo.QumoError) as e:
        qumo.sample(c, method="statevector")
    assert e.value.code == "QM-E404" and "qumo.sample()" in e.value.hint


def test_no_measurements_hint_is_python(q):
    c = qumo.Circuit(2).h(0).cx(0, 1)
    with pytest.raises(qumo.QumoError) as e:
        qumo.sample(c)
    assert e.value.code == "QM-E402"
    assert "circuit.measure_all()" in e.value.hint and "qumo.statevector" in e.value.hint
    assert "evolve()" not in e.value.hint
    wide = qumo.Circuit(40).h(0)  # the stabilizer path returns {'': shots} natively: the facade refuses it
    with pytest.raises(qumo.QumoError) as e:
        qumo.sample(wide)
    assert e.value.code == "QM-E402"


@pytest.mark.parametrize("shots", ["100", 100.5, None, [100]])
def test_shots_must_be_an_int(q, bell, shots):
    """Regression: "100" and 100.5 were converted silently."""
    with pytest.raises(TypeError) as e:
        qumo.sample(bell, shots=shots)
    assert isinstance(e.value, qumo.QumoError) and e.value.code == "QM-E911" and "shots" in str(e.value)
    for call in (lambda: q.sample(bell.native, shots, 1), lambda: q.sample_device(bell.native, shots, 1, "auto"),
                 lambda: q.sample_ptsbe(bell.native, shots, 1), lambda: q.sample_branching(bell.native, shots, 1)):
        with pytest.raises(Exception, match="QM-E911.*'shots' must be an int"):
            call()


def test_seed_must_be_an_int(bell):
    with pytest.raises(TypeError):
        qumo.sample(bell, seed="7")
    with pytest.raises(TypeError):
        qumo.sample(bell, seed=1.5)
    assert sum(qumo.sample(bell, shots=np.int64(64), seed=np.int32(5)).values()) == 64


def test_negative_shots_and_unknown_options(bell):
    with pytest.raises(qumo.QumoError) as e:
        qumo.sample(bell, shots=-1)
    assert e.value.code == "QM-E913"
    with pytest.raises(qumo.QumoError) as e:
        qumo.sample(bell, device="quantum")
    assert e.value.code == "QM-E940"
    with pytest.raises(qumo.QumoError):
        qumo.sample(bell, method="magic")
    with pytest.raises(TypeError):
        qumo.sample("OPENQASM 3.0; qubit[1] q;")


def test_sample_with_parameters():
    c = qumo.Circuit(1).ry("theta", 0).measure_all()
    assert qumo.sample(c, shots=50, params={"theta": math.pi}) == {"1": 50}
    assert qumo.sample(c, shots=50, params=[0.0]) == {"0": 50}
    assert qumo.sample(c, shots=50, params={"theta": math.pi}, method="statevector") == {"1": 50}
    with pytest.raises(qumo.QumoError) as e:
        qumo.sample(c)
    assert e.value.code == "QM-E101" and "params=" in e.value.hint
    with pytest.raises(qumo.QumoError) as e:
        qumo.sample(c, params={"phi": 1.0})
    assert e.value.code == "QM-E101" and "theta" in str(e.value)


def test_sampling_methods_agree_on_noisy_circuit():
    c = qumo.Circuit(2).h(0).depolarize1(0.1, 0).cx(0, 1).depolarize2(0.1, 0, 1).measure_all()
    shots = 20000
    auto = qumo.sample(c, shots, 5)
    ptsbe = qumo.sample(c, shots, 5, method="ptsbe")
    assert auto.method and ptsbe.method == "statevector:ptsbe"
    exact = qumo.probabilities(c)  # density matrix
    for i, key in enumerate(("00", "01", "10", "11")):
        for counts in (auto, ptsbe):
            assert abs(counts.get(key, 0) / shots - exact[i]) < 0.02, (key, counts.method)
    cliff = qumo.Circuit(3).h(0).cx(0, 1).cx(1, 2).measure_all()
    assert set(qumo.sample(cliff, 100, method="branching")) == {"000", "111"}


def test_dynamic_circuit_feed_forward():
    c = qumo.Circuit(2).h(0).measure(0, 0).x(1).c_if(0, 1).measure(1, 1)
    assert set(qumo.sample(c, 500)) == {"00", "11"}


def test_statevector_is_a_numpy_array():
    psi = qumo.statevector(qumo.Circuit(2).h(0).cx(0, 1))
    assert isinstance(psi, np.ndarray) and psi.dtype == np.complex128 and psi.shape == (4,)
    assert psi.flags["C_CONTIGUOUS"] and psi.flags["OWNDATA"] and psi.flags["WRITEABLE"]
    assert np.allclose(psi, np.array([1, 0, 0, 1]) / math.sqrt(2))
    # qubit 0 is the least significant bit of the index
    assert np.allclose(qumo.statevector(qumo.Circuit(2).x(0)), [0, 1, 0, 0])
    assert np.allclose(qumo.statevector(qumo.Circuit(1).s(0).h(0).t(0).p(0.3, 0).x(0)),
                       np.array([np.exp(1j * (math.pi / 4 + 0.3)), 1]) / math.sqrt(2))


def test_statevector_matches_matrix_product():
    rng = np.random.default_rng(4)
    n = 5
    c = qumo.Circuit(n)
    ref = np.zeros(2 ** n, dtype=complex)
    ref[0] = 1
    for _ in range(30):
        q0, theta = int(rng.integers(n)), float(rng.uniform(-3, 3))
        c.ry(theta, q0)
        cos, sin = math.cos(theta / 2), math.sin(theta / 2)
        ref = ref.reshape([2] * n)
        axis = n - 1 - q0
        a, b = np.take(ref, 0, axis=axis), np.take(ref, 1, axis=axis)
        ref = np.stack([cos * a - sin * b, sin * a + cos * b], axis=axis).reshape(-1)
        q1 = int((q0 + 1 + rng.integers(n - 1)) % n)
        c.cz(q0, q1)
        idx = np.arange(2 ** n)
        ref = np.where(((idx >> q0) & 1) & ((idx >> q1) & 1), -ref, ref)
    assert np.allclose(qumo.statevector(c), ref, atol=1e-12)


def test_statevector_large_and_parallel_copy():
    n = 18  # past the threshold where the copy into NumPy runs on several threads
    psi = qumo.statevector(qumo.ghz(n))
    assert psi.shape == (2 ** n,)
    assert math.isclose(abs(psi[0]) ** 2, 0.5, rel_tol=1e-12) and math.isclose(abs(psi[-1]) ** 2, 0.5, rel_tol=1e-12)
    assert np.count_nonzero(psi) == 2
    uniform = qumo.Circuit(n)
    for i in range(n):
        uniform.h(i)
    assert np.allclose(qumo.statevector(uniform), 2 ** (-n / 2))
    assert np.allclose(qumo.probabilities(uniform), 2.0 ** -n)


def test_statevector_parameters_and_noise():
    c = qumo.Circuit(1).ry("theta", 0)
    assert np.allclose(qumo.statevector(c, {"theta": math.pi}), [0, 1])
    with pytest.raises(qumo.QumoError) as e:
        qumo.statevector(c)
    assert e.value.code == "QM-E101"
    with pytest.raises(qumo.QumoError) as e:
        qumo.statevector(qumo.Circuit(1).h(0).depolarize1(0.1, 0))
    assert e.value.code == "QM-E414" and "qumo.probabilities" in e.value.hint


def test_probabilities():
    p = qumo.probabilities(qumo.Circuit(2).h(0).cx(0, 1))
    assert isinstance(p, np.ndarray) and p.dtype == np.float64
    assert np.allclose(p, [0.5, 0, 0, 0.5])
    noisy = qumo.probabilities(qumo.Circuit(1).x(0).x_error(0.25, 0))
    assert np.allclose(noisy, [0.25, 0.75])
    assert np.allclose(qumo.probabilities(qumo.Circuit(1).ry("t", 0), [math.pi / 2]), [0.5, 0.5])


def test_expectation_observable_forms():
    bell = qumo.Circuit(2).h(0).cx(0, 1)
    assert math.isclose(qumo.expectation(bell, "Z0 Z1"), 1.0, abs_tol=1e-12)
    assert math.isclose(qumo.expectation(bell, "ZZ + 0.5*XX"), 1.5, abs_tol=1e-12)
    assert math.isclose(qumo.expectation(bell, [("ZZ", 1.0), ("XX", 0.5)]), 1.5, abs_tol=1e-12)
    assert math.isclose(qumo.expectation(bell, [("Z0 Z1", 2.0)]), 2.0, abs_tol=1e-12)
    assert math.isclose(qumo.expectation(bell, {"ZZ": 1.0, "YY": 1.0}), 0.0, abs_tol=1e-12)
    assert math.isclose(qumo.expectation(bell, [(1.0, "ZZ")]), 1.0, abs_tol=1e-12)  # (coeff, label) too
    one = qumo.Circuit(2).x(0)  # dense labels read from the highest qubit
    assert math.isclose(qumo.expectation(one, [("IZ", 1.0)]), -1.0) and math.isclose(qumo.expectation(one, "ZI"), 1.0)
    assert isinstance(qumo.expectation(bell, "Z0"), float)
    with pytest.raises(TypeError):
        qumo.expectation(bell, 5)
    with pytest.raises(qumo.QumoError) as e:
        qumo.expectation(bell, [("ZQ", 1.0)])
    assert e.value.code == "QM-E301"
    with pytest.raises(TypeError):
        qumo.expectation(bell, [])


def test_expectation_does_not_ignore_noise(q):
    """Regression: qumo_py.expectation returns 1.0 on a noisy Bell pair; density gives 0.9467."""
    noisy = qumo.Circuit(2).h(0).cx(0, 1).depolarize2(0.05, 0, 1)
    assert math.isclose(q.expectation(noisy.native, "Z0 Z1"), 1.0, abs_tol=1e-12)  # the raw call skips channels
    value = qumo.expectation(noisy, "Z0 Z1")
    assert math.isclose(value, 1 - 0.05 * 16 / 15, abs_tol=1e-9) and round(value, 4) == 0.9467
    assert math.isclose(value, q.density_expectation(noisy.native, "Z0 Z1"), abs_tol=1e-12)
    assert math.isclose(qumo.expectation(noisy, "Z0 Z1", noise="ignore"), 1.0, abs_tol=1e-12)
    with pytest.raises(qumo.QumoError) as e:
        qumo.expectation(noisy, "Z0 Z1", noise="error")
    assert e.value.code == "QM-E414"
    with pytest.raises(qumo.QumoError):
        qumo.expectation(noisy, "Z0 Z1", noise="maybe")
    big = qumo.Circuit(16).h(0).depolarize1(0.01, 0)  # past the density-matrix limit: an error, not a wrong number
    with pytest.raises(qumo.QumoError) as e:
        qumo.expectation(big, "Z0")
    assert e.value.code == "QM-E410"


def test_expectation_with_parameters():
    c = qumo.Circuit(1).ry("theta", 0)
    for theta in (0.0, 0.7, 2.0):
        assert math.isclose(qumo.expectation(c, "Z0", {"theta": theta}), math.cos(theta), abs_tol=1e-12)
        assert math.isclose(qumo.expectation(c, "Z0", [theta], device="auto"), math.cos(theta), abs_tol=1e-6)
    noisy = qumo.Circuit(1).ry("theta", 0).z_error(0.1, 0)
    assert math.isclose(qumo.expectation(noisy, "X0", {"theta": math.pi / 2}), 0.8, abs_tol=1e-9)
    with pytest.raises(qumo.QumoError) as e:
        qumo.expectation(c, "Z0")
    assert e.value.code == "QM-E101"


def test_gradients_single_and_batch():
    c = qumo.Circuit(2).ry("a", 0).ry("b", 1).cx(0, 1)
    value, grad = qumo.value_and_gradient(c, "Z0 Z1", {"a": 0.3, "b": 0.9})
    assert math.isclose(value, math.cos(0.9), abs_tol=1e-12)
    assert isinstance(grad, np.ndarray) and np.allclose(grad, [0.0, -math.sin(0.9)], atol=1e-12)
    grid = np.array([[0.3, b] for b in np.linspace(0, math.pi, 9)])
    values, grads = qumo.batch_value_and_gradient(c, "Z0 Z1", grid)
    assert values.shape == (9,) and grads.shape == (9, 2)
    assert np.allclose(values, np.cos(grid[:, 1]), atol=1e-12)
    assert np.allclose(grads[:, 1], -np.sin(grid[:, 1]), atol=1e-12)
    with pytest.raises(qumo.QumoError):
        qumo.batch_value_and_gradient(c, "Z0", np.zeros((3, 5)))
