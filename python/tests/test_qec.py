"""QEC drivers: memory experiments, threshold sweeps, packed streaming."""

import numpy as np
import pytest

import qumo


def test_memory_experiment():
    r = qumo.memory_experiment("repetition", distance=3, p=0.01, shots=20000)
    assert r["code"] == "repetition" and r["distance"] == 3 and r["rounds"] == 3 and r["decoder"] == "mwpm"
    assert r["shots"] == 20000 and 0 < r["errors"] < 400
    assert r["logical_error_rate"] == r["errors"] / r["shots"]
    lo, hi = r["wilson"]
    assert lo < r["logical_error_rate"] < hi
    assert r["shots_per_second"] > 0 and r["num_detectors"] > 0
    again = qumo.memory_experiment("repetition", distance=3, p=0.01, shots=20000)
    assert again["errors"] == r["errors"]  # deterministic for a seed, whatever the thread count
    assert qumo.memory_experiment("repetition", 3, p=0.01, shots=20000, seed=2)["errors"] != r["errors"]


def test_memory_experiment_options():
    surf = qumo.memory_experiment("surface_z", distance=3, rounds=2, p=0.002, shots=4096, decoder="uf")
    assert surf["rounds"] == 2 and surf["num_qubits"] == 17 and surf["decoder"] == "uf"
    early = qumo.memory_experiment("repetition", 3, p=0.05, shots=10 ** 7, max_errors=50)
    assert early["shots"] < 10 ** 7 and early["errors"] >= 50
    with pytest.raises(qumo.QumoError) as e:
        qumo.memory_experiment("toric", 3)
    assert e.value.code == "QM-E706"
    with pytest.raises(qumo.QumoError) as e:
        qumo.memory_experiment("repetition", 3, decoder="oracle")
    assert e.value.code == "QM-E707"
    with pytest.raises(TypeError):
        qumo.memory_experiment("repetition", 3, shots="many")
    assert set(qumo.CODES) == {"surface_z", "surface_x", "repetition"} and "mwpm" in qumo.DECODERS


def test_threshold_sweep_returns_arrays():
    s = qumo.threshold_sweep("repetition", [3, 5, 7], [0.01, 0.03], shots=20000)
    for key in ("logical_error_rate", "std_error", "wilson_lo", "wilson_hi", "per_round_error_rate", "seconds",
                "errors", "shots", "rounds"):
        assert isinstance(s[key], np.ndarray) and s[key].shape == (3, 2), key
    assert s["errors"].dtype == np.int64 and s["logical_error_rate"].dtype == np.float64
    assert s["distances"].tolist() == [3, 5, 7] and s["ps"].tolist() == [0.01, 0.03]
    assert (s["shots"] == 20000).all() and (s["rounds"] == np.array([[3], [5], [7]])).all()
    rate = s["logical_error_rate"]
    assert (rate[:, 1] > rate[:, 0]).all()  # more noise, more logical errors
    assert rate[2, 0] < rate[0, 0]  # below threshold a larger distance helps
    single = qumo.memory_experiment("repetition", 5, p=0.03, shots=20000)
    assert single["errors"] == s["errors"][1, 1]
    with pytest.raises(qumo.QumoError):
        qumo.threshold_sweep("repetition", [], [0.01])


def test_streaming_memory_stays_packed():
    m = qumo.QecMemory("repetition", distance=5, rounds=5, p=0.02)
    assert m.num_detectors > 0 and m.num_observables == 1 and "repetition" in repr(m)
    dets, obs = m.sample(shots=4000, seed=5)
    width = (m.num_detectors + 7) // 8
    assert dets.dtype == np.uint8 and dets.shape == (4000, width) and obs.shape == (4000, 1)
    predictions = m.decode(dets)
    assert predictions.shape == obs.shape and predictions.dtype == np.uint8
    errors = m.count_logical_errors(obs, predictions)
    assert 0 <= errors < 400
    assert (m.decode(dets, threads=1) == predictions).all()
    bits = np.unpackbits(dets, axis=1, bitorder="little")[:, :m.num_detectors]
    assert 0.0 < bits.mean() < 0.2
    again, _ = m.sample(shots=4000, seed=5)
    assert (again == dets).all()
    with pytest.raises(qumo.QumoError):
        m.decode(dets[:, :1])
    assert "error(" in m.dem() and m.circuit().num_qubits == m.info["num_qubits"]


def test_decode_dem_argument_errors(q):
    """Regression: decode_dem(text, 20000, 'mwpm') said "'int' object is not iterable"."""
    dem = qumo.QecMemory("repetition", 3, 3, 0.01).dem()
    with pytest.raises(Exception) as e:
        q.decode_dem(dem, 20000, "mwpm")
    text = str(e.value)
    assert text.startswith("QM-E911") and "'shots' must be a list" in text and "sample_dem" in text
    assert "not iterable" not in text
    with pytest.raises(Exception, match="QM-E911"):
        q.decode_dem(dem, "0 1", "mwpm")
    with pytest.raises(Exception, match="QM-E911"):
        q.decode_dem(dem, [[0, 1]], 3)
    with pytest.raises(Exception, match="QM-E707"):
        q.decode_dem(dem, [[0, 1]], "oracle")
    assert list(q.decode_dem(dem, [[0, 1], []], "mwpm")) == [0, 0]
    assert len(q.decode_dem(dem, [[0], [1], []], "mwpm_correlated")) == 3
