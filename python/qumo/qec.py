"""Quantum error correction: memory experiments, threshold sweeps, streaming decode.

Sampling (bit-packed Pauli frames) and decoding (sparse blossom) are fused in
the native library and run on every core. Python receives totals, or
bit-packed NumPy arrays; there is never a Python list per shot.
"""

from __future__ import annotations

from ._native import native as _q
from .circuit import Circuit, _call
from .errors import QumoError

__all__ = ["memory_experiment", "threshold_sweep", "QecMemory", "CODES", "DECODERS"]

CODES = ("surface_z", "surface_x", "repetition")
DECODERS = ("mwpm", "mwpm_correlated", "mwpm_dense", "uf", "lookup")


def memory_experiment(code: str = "surface_z", distance: int = 3, rounds: int | None = None, p: float = 0.001,
                      shots: int = 100_000, *, decoder: str = "mwpm", seed: int = 1,
                      max_errors: int | None = None) -> dict:
    """Logical error rate of a QEC memory experiment under uniform circuit noise `p`.

    Args:
        code: ``'surface_z'`` | ``'surface_x'`` | ``'repetition'``.
        distance: code distance. rounds: syndrome rounds (default: `distance`).
        p: physical error rate of every operation.
        shots: shots to sample and decode (default 100000).
        decoder: ``'mwpm'`` (sparse blossom, default) | ``'mwpm_correlated'`` |
            ``'mwpm_dense'`` | ``'uf'`` | ``'lookup'``.
        seed: results are identical for a seed whatever the thread count.
        max_errors: stop early once this many logical errors were seen.

    Returns:
        dict with ``shots``, ``errors``, ``logical_error_rate``, ``std_error``,
        ``wilson`` (95 % interval), ``per_round_error_rate``, ``seconds``,
        ``shots_per_second``, ``num_qubits``, ``num_detectors``, ...

    Example:
        >>> import qumo
        >>> r = qumo.memory_experiment("repetition", distance=3, p=0.01, shots=20000)
        >>> r["shots"], 0.0 < r["logical_error_rate"] < 0.02
        (20000, True)
    """
    r = dict(_call(_q.memory_experiment, code, distance, -1 if rounds is None else rounds, float(p), shots,
                   decoder, seed, -1 if max_errors is None else max_errors))
    r["wilson"] = tuple(r["wilson"])
    return r


def threshold_sweep(code: str, distances, ps, shots: int = 100_000, *, decoder: str = "mwpm", seed: int = 1,
                    max_errors: int | None = None, rounds: int | None = None) -> dict:
    """Grid of memory experiments over code distances and physical error rates, in one native call.

    Returns:
        dict of NumPy arrays of shape ``(len(distances), len(ps))``:
        ``logical_error_rate``, ``std_error``, ``wilson_lo``, ``wilson_hi``,
        ``per_round_error_rate``, ``errors``, ``shots``, ``seconds``,
        ``rounds``; plus ``distances`` and ``ps`` (1-D arrays), ``code``,
        ``decoder``.

    Example:
        >>> import qumo
        >>> s = qumo.threshold_sweep("repetition", [3, 5], [0.01, 0.02], shots=5000)
        >>> s["logical_error_rate"].shape
        (2, 2)
    """
    import numpy as np

    ds = [int(d) for d in distances]
    pl = [float(x) for x in ps]
    if not ds or not pl:
        raise QumoError("threshold_sweep needs at least one distance and one error rate", "QM-E913",
                        "e.g. distances=[3, 5], ps=[0.001, 0.002].")
    r = dict(_call(_q.threshold_sweep, code, ds, pl, shots, decoder, seed, -1 if max_errors is None else max_errors,
                   -1 if rounds is None else rounds))
    shape = (len(ds), len(pl))
    out = {"code": str(r["code"]), "decoder": str(r["decoder"]),
           "distances": np.asarray(ds, dtype=np.int64), "ps": np.asarray(pl, dtype=np.float64)}
    for key in ("shots", "errors", "rounds"):
        out[key] = np.asarray(r[key], dtype=np.int64).reshape(shape)
    for key in ("logical_error_rate", "std_error", "wilson_lo", "wilson_hi", "per_round_error_rate", "seconds"):
        out[key] = np.asarray(r[key], dtype=np.float64).reshape(shape)
    return out


class QecMemory:
    """A memory experiment kept open for streaming: sample syndromes, decode them, repeat.

    Detection events travel as bit-packed ``uint8`` arrays of shape
    ``(shots, ceil(num_detectors / 8))``; bit ``k`` of byte ``j`` is detector
    ``8 j + k`` (``numpy.unpackbits(..., bitorder="little")`` expands them).

    Example:
        >>> import qumo
        >>> m = qumo.QecMemory("repetition", distance=3, rounds=3, p=0.01)
        >>> dets, obs = m.sample(shots=2000, seed=5)
        >>> predictions = m.decode(dets)
        >>> dets.shape[0], predictions.shape == obs.shape
        (2000, True)
        >>> errors = m.count_logical_errors(obs, predictions)
    """

    def __init__(self, code: str = "surface_z", distance: int = 3, rounds: int | None = None, p: float = 0.001):
        self._m = _call(_q.qec_memory, code, distance, distance if rounds is None else rounds, float(p))
        info = dict(self._m.info())
        self.info = info
        self.num_detectors = int(info["num_detectors"])
        self.num_observables = int(info["num_observables"])
        self.last_method = ""
        self.last_seconds = 0.0

    def sample(self, shots: int, seed: int = 1, *, device: str = "cpu", threads: int = 0):
        """Sample -> ``(detectors, observables)`` as bit-packed ``uint8`` arrays, one row per shot."""
        import numpy as np

        dets, obs, method, _note, seconds = _call(self._m.sample, shots, seed, device, threads)
        self.last_method, self.last_seconds = str(method), float(seconds)
        d = np.frombuffer(bytes.fromhex(dets), dtype=np.uint8).reshape(shots, -1)
        o = np.frombuffer(bytes.fromhex(obs), dtype=np.uint8).reshape(shots, -1)
        return d, o

    def decode(self, detectors, *, threads: int = 0):
        """Decode packed detection events -> predicted observable flips, bit-packed like `sample`'s."""
        import numpy as np

        d = np.ascontiguousarray(detectors, dtype=np.uint8)
        width = (self.num_detectors + 7) // 8
        if d.ndim != 2 or d.shape[1] != width:
            raise QumoError(f"detectors has shape {d.shape}; expected (shots, {width})", "QM-E913",
                            "pass the first array QecMemory.sample() returned.")
        shots = d.shape[0]
        pred, seconds, _events = _call(self._m.decode, d.tobytes().hex(), shots, threads)
        self.last_seconds = float(seconds)
        return np.frombuffer(bytes.fromhex(pred), dtype=np.uint8).reshape(shots, -1)

    @staticmethod
    def count_logical_errors(observables, predictions) -> int:
        """Shots whose predicted observable flips differ from the sampled ones."""
        import numpy as np

        return int(np.any(np.asarray(observables) != np.asarray(predictions), axis=1).sum())

    def dem(self) -> str:
        """The detector error model the decoder uses, in Stim DEM text."""
        return str(self._m.dem_text())

    def circuit(self) -> Circuit:
        """The noisy memory circuit."""
        return Circuit.from_native(_call(_q.from_stim, self._m.stim_text()))

    def to_stim(self) -> str:
        return str(self._m.stim_text())

    def __repr__(self) -> str:
        i = self.info
        return (f"<qumo.QecMemory {i['code']} d={i['distance']} rounds={i['rounds']} p={i['p']} "
                f"detectors={self.num_detectors}>")
