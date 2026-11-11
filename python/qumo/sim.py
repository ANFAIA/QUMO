"""Simulation: sampling, state vectors, probabilities, expectation values.

Each function is one call into the native library; the work (gate fusion,
cache blocking, threads, stabilizer / Pauli-frame / GPU dispatch) stays in
Mojo. Defaults equal the command line's (`qumo run`): shots=1024, seed=1234.
"""

from __future__ import annotations

from ._native import native as _q
from .circuit import Circuit, _call, _native_of, param_values
from .errors import QumoError, QumoTypeError, from_native

__all__ = [
    "Counts", "sample", "run", "statevector", "probabilities", "expectation",
    "value_and_gradient", "batch_value_and_gradient",
]

_METHODS = ("auto", "statevector", "ptsbe", "branching")


class Counts(dict):
    """Measurement counts ``{bitstring: count}``: a dict with the run's metadata.

    Bit 0 of the classical register is the rightmost character of a key.

    Attributes:
        shots: number of shots requested.
        seed: the seed used.
        method: the simulator that ran, e.g. ``'statevector'``, ``'stabilizer:tableau'``.
        note: why a requested device was not used, or ``''``.
    """

    __slots__ = ("shots", "seed", "method", "note")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shots = sum(self.values())
        self.seed = None
        self.method = ""
        self.note = ""

    def probabilities(self) -> dict[str, float]:
        """``{bitstring: count / shots}``."""
        total = sum(self.values()) or 1
        return {k: v / total for k, v in self.items()}

    def most_common(self, n: int | None = None) -> list[tuple[str, int]]:
        """Outcomes by decreasing count (ties in bitstring order)."""
        ranked = sorted(self.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked if n is None else ranked[:n]

    def _repr_pretty_(self, printer, cycle):
        printer.text(str(self))

    def __str__(self) -> str:
        total = sum(self.values()) or 1
        head = f"{total} shots, {len(self)} outcome(s)" + (f", {self.method}" if self.method else "")
        lines = [head]
        for key, count in self.most_common(32):
            frac = count / total
            lines.append(f"  {key}  {count:>8}  {frac:6.4f}  {'#' * round(40 * frac)}")
        if len(self) > 32:
            lines.append(f"  ... {len(self) - 32} more outcome(s)")
        return "\n".join(lines)

    def __reduce__(self):
        return (_rebuild_counts, (dict(self), self.shots, self.seed, self.method, self.note))


def _rebuild_counts(data, shots, seed, method, note):
    out = Counts(data)
    out.shots, out.seed, out.method, out.note = shots, seed, method, note
    return out


def sample(circuit, shots: int = 1024, seed: int = 1234, *, params=None, device: str = "auto",
           precision: str = "auto", method: str = "auto") -> Counts:
    """Run `circuit` `shots` times and count the measured bitstrings.

    The simulator is chosen as `qumo run` chooses it: Clifford circuits go to
    the stabilizer tableau or Pauli frames (hundreds of qubits), the rest to
    the state vector (up to 32 qubits), on the GPU when the build has it and
    the circuit is large enough. Noise channels and mid-circuit measurement
    with feed-forward are sampled as trajectories.

    Args:
        circuit: a `qumo.Circuit` with at least one measurement.
        shots: number of repetitions (default 1024, as the CLI).
        seed: random seed (default 1234, as the CLI); same seed, same counts.
        params: values for symbolic parameters: ``{name: value}`` or a
            sequence in `circuit.parameter_names` order.
        device: ``'auto'`` | ``'cpu'`` | ``'gpu'``.
        precision: ``'auto'`` | ``'fp32'`` | ``'fp64'`` for GPU state vectors.
        method: ``'auto'`` (dispatch above) | ``'statevector'`` (always the
            CPU state vector) | ``'ptsbe'`` (static Pauli-noise circuits: one
            simulation per distinct error pattern) | ``'branching'``
            (near-Clifford circuits on up to 512 qubits).

    Returns:
        `Counts`: a dict ``{bitstring: count}`` with `.method`, `.shots`,
        `.seed`, `.note`, `.probabilities()` and `.most_common()`. Bit 0 is
        the rightmost character.

    Raises:
        QumoError: QM-E402 no measurements, QM-E404 too many qubits for the
            state vector, QM-E101 unbound parameters.
        TypeError: `shots` or `seed` is not an int (also a `QumoError`, QM-E911).

    Example:
        >>> import qumo
        >>> bell = qumo.Circuit(2).h(0).cx(0, 1).measure_all()
        >>> counts = qumo.sample(bell, shots=1000, seed=7)
        >>> sorted(counts), counts.shots
        (['00', '11'], 1000)
        >>> ghz = qumo.Circuit(40).h(0)
        >>> for q in range(39):
        ...     _ = ghz.cx(q, q + 1)
        >>> qumo.sample(ghz.measure_all(), shots=100).method     # 40 qubits: stabilizer
        'stabilizer:tableau'
    """
    nat = circuit._flush() if circuit.__class__ is Circuit else _native_of(circuit)
    out = Counts()
    try:
        if method == "auto":
            if params is None:
                if precision == "auto":
                    _, used, note = _q.sample_device(nat, shots, seed, device, counts=out)
                else:
                    _, used, note = _q.sample_device(nat, shots, seed, device, counts=out, precision=precision)
            else:
                _, used, note = _q.sample_device(nat, shots, seed, device, counts=out, precision=precision,
                                                 params=param_values(circuit, params))
        else:
            used, note = _sample_method(nat, circuit, shots, seed, params, method, out)
    except QumoError:
        raise
    except Exception as exc:
        raise from_native(exc) from None
    if "" in out and nat.num_measurements() == 0:
        raise QumoError("the circuit has no measurements", "QM-E402",
                        "call circuit.measure_all() (or circuit.measure(q, c)) before sampling, or use "
                        "qumo.statevector(circuit) / qumo.expectation(circuit, observable).")
    out.shots = shots
    out.seed = seed
    out.method = str(used)
    out.note = str(note)
    return out


def _sample_method(nat, circuit, shots, seed, params, method, out):
    if method not in _METHODS:
        raise QumoError(f"unknown sampling method '{method}'", "QM-E913",
                        "use 'auto', 'statevector', 'ptsbe' or 'branching'.")
    if params is not None:
        nat = nat.bind(param_values(circuit, params))
    if method == "statevector":
        out.update(_q.sample(nat, shots, seed))
        return "statevector", ""
    if method == "ptsbe":
        out.update(_q.sample_ptsbe(nat, shots, seed))
        return "statevector:ptsbe", ""
    out.update(_q.sample_branching(nat, shots, seed))
    return "stabilizer:branching", ""


def run(circuit, shots: int = 1024, seed: int = 1234, **options) -> Counts:
    """Alias of `sample` (the name the command line uses: `qumo run`).

    Example:
        >>> import qumo
        >>> qumo.run(qumo.Circuit(1).x(0).measure_all(), shots=10)
        {'1': 10}
    """
    return sample(circuit, shots, seed, **options)


def statevector(circuit, params=None):
    """Final state vector as a NumPy ``complex128`` array of ``2**n`` amplitudes.

    Amplitude ``i`` has qubit ``q`` in bit ``q`` of ``i`` (qubit 0 is the least
    significant bit). Measurements and resets are skipped. The simulator's
    state is written once into the array; no Python list is built.

    Args:
        circuit: a noiseless `qumo.Circuit` on up to 32 qubits.
        params: values for symbolic parameters (dict or sequence).

    Raises:
        QumoError: QM-E404 past 32 qubits; QM-E414 when the circuit holds
            noise channels (a noisy circuit has no state vector: use
            `probabilities` or `expectation`, which go through the density
            matrix).

    Example:
        >>> import qumo
        >>> qumo.statevector(qumo.Circuit(2).h(0).cx(0, 1)).round(3)
        array([0.707+0.j, 0.   +0.j, 0.   +0.j, 0.707+0.j])
    """
    nat = _native_of(circuit)
    if nat.has_noise():
        raise QumoError("the circuit has noise channels, so it has no state vector", "QM-E414",
                        "use qumo.probabilities(circuit) or qumo.expectation(circuit, observable) (density "
                        "matrix), or qumo.sample(circuit) (trajectories).")
    if params is None and nat.num_parameters() == 0:
        return _call(_q.statevector, nat)
    return _call(_q.statevector, nat, params=param_values(circuit, params))


def probabilities(circuit, params=None):
    """Outcome probabilities of all qubits as a NumPy ``float64`` array of ``2**n`` values.

    Index ``i`` has qubit ``q`` in bit ``q``. A circuit with noise channels is
    simulated as a density matrix (up to 15 qubits), so the result includes
    the noise.

    Example:
        >>> import qumo
        >>> qumo.probabilities(qumo.Circuit(2).h(0).cx(0, 1)).round(3)
        array([0.5, 0. , 0. , 0.5])
    """
    nat = _native_of(circuit)
    if params is None and nat.num_parameters() == 0:
        return _call(_q.probabilities, nat)
    return _call(_q.probabilities, nat, params=param_values(circuit, params))


def _observable(observable):
    """A str passes through; (label, coeff) pairs become a list the extension reads."""
    if isinstance(observable, str):
        return observable
    if isinstance(observable, dict):
        return list(observable.items())
    try:
        return list(observable)
    except TypeError:
        raise QumoTypeError(
            f"an observable is a Pauli-sum str or a list of (label, coeff) pairs, got {type(observable).__name__}",
            "QM-E911", "e.g. 'Z0 Z1 + 0.5 * X0' or [('ZZ', 1.0), ('XI', 0.5)].") from None


def expectation(circuit, observable, params=None, *, noise: str = "auto", device: str = "cpu") -> float:
    """Expectation value ``<H>`` of a Pauli-sum observable in the circuit's final state.

    Args:
        circuit: a `qumo.Circuit`.
        observable: a Pauli-sum str (``"Z0 Z1 + 0.5 * X0"`` or dense labels
            ``"ZZ + 0.5*IX"``), or a list of ``(label, coeff)`` pairs such as
            ``[("ZZ", 1.0), ("IX", 0.5)]`` (what `qumo.h2_sto3g()` returns).
            A dense label reads left to right from the highest qubit.
        params: values for symbolic parameters (dict or sequence).
        noise: ``'auto'`` (default): a circuit with noise channels is
            evaluated exactly on the density matrix, ``Tr(rho H)`` (up to 15
            qubits); ``'ignore'``: skip the channels (the ideal value);
            ``'error'``: raise QM-E414 if the circuit has noise.
        device: ``'cpu'`` (default) | ``'gpu'`` | ``'auto'`` for the noiseless path.

    Returns:
        float.

    Example:
        >>> import qumo
        >>> bell = qumo.Circuit(2).h(0).cx(0, 1)
        >>> round(qumo.expectation(bell, "Z0 Z1"), 6)
        1.0
        >>> noisy = qumo.Circuit(2).h(0).cx(0, 1).depolarize2(0.05, 0, 1)
        >>> round(qumo.expectation(noisy, [("ZZ", 1.0)]), 4)     # density matrix: noise counts
        0.9467
    """
    nat = _native_of(circuit)
    obs = _observable(observable)
    if noise not in ("auto", "ignore", "error"):
        raise QumoError(f"unknown noise option '{noise}'", "QM-E913", "use 'auto', 'ignore' or 'error'.")
    if noise != "ignore" and nat.has_noise():
        if noise == "error":
            raise QumoError("the circuit has noise channels", "QM-E414",
                            "pass noise='auto' for the density-matrix value or noise='ignore' for the ideal one.")
        if params is not None or nat.num_parameters():
            nat = _call(nat.bind, param_values(circuit, params))
        return float(_call(_q.density_expectation, nat, obs))
    if device != "cpu":
        if params is not None or nat.num_parameters():
            nat = _call(nat.bind, param_values(circuit, params))
        return float(_call(_q.expectation_device, nat, obs, device)[0])
    if params is None and nat.num_parameters() == 0:
        return float(_call(_q.expectation, nat, obs))
    return float(_call(_q.expectation, nat, obs, params=param_values(circuit, params)))


def value_and_gradient(circuit, observable, params, *, device: str = "cpu"):
    """Exact ``E(params)`` and its gradient by the adjoint method, in one native call.

    Returns:
        ``(value, gradient)``: a float and a NumPy array with one entry per
        parameter, in `circuit.parameter_names` order.

    Example:
        >>> import qumo
        >>> c = qumo.Circuit(1).ry("theta", 0)
        >>> e, g = qumo.value_and_gradient(c, "Z0", {"theta": 0.5})
        >>> round(e, 4), g.round(4).tolist()
        (0.8776, [-0.4794])
    """
    import numpy as np

    nat = _native_of(circuit)
    if nat.has_noise():
        raise QumoError("the adjoint gradient needs a noiseless circuit", "QM-E414", "remove the noise channels.")
    value, grad, _, _ = _call(_q.value_and_gradient, nat, _observable(observable), param_values(circuit, params),
                              device=device)
    return float(value), np.asarray(grad, dtype=np.float64)


def batch_value_and_gradient(circuit, observable, param_sets):
    """Values and adjoint gradients for many parameter sets in ONE native call.

    This is the way to sweep: the loop over sets runs in Mojo (in parallel up
    to 14 qubits) instead of one Python call per point.

    Args:
        param_sets: array-like of shape ``(num_sets, num_parameters)``, columns
            in `circuit.parameter_names` order.

    Returns:
        ``(values, gradients)``: NumPy arrays of shape ``(num_sets,)`` and
        ``(num_sets, num_parameters)``.

    Example:
        >>> import numpy as np, qumo
        >>> c = qumo.Circuit(1).ry("theta", 0)
        >>> values, grads = qumo.batch_value_and_gradient(c, "Z0", np.linspace(0, np.pi, 5)[:, None])
        >>> values.round(3).tolist()
        [1.0, 0.707, 0.0, -0.707, -1.0]
    """
    import numpy as np

    nat = _native_of(circuit)
    sets = np.asarray(param_sets, dtype=np.float64)
    if sets.ndim != 2 or sets.shape[1] != nat.num_parameters():
        raise QumoError(
            f"param_sets has shape {sets.shape}; expected (num_sets, {nat.num_parameters()})", "QM-E101",
            "one row per parameter set, columns in circuit.parameter_names order.")
    values, grads = _call(_q.batch_value_and_gradient, nat, _observable(observable), sets.tolist())
    return (np.asarray(values, dtype=np.float64),
            np.asarray(grads, dtype=np.float64).reshape(sets.shape[0], sets.shape[1]))
