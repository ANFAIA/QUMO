"""`Circuit`: build, inspect and exchange quantum circuits.

The builder mirrors the Mojo `Circuit` API (`qumo/ir/circuit.mojo`): the same
gate names, angles first and qubits after (``c.rx(theta, q)``).

Performance: a gate method only appends five integers to a C array held in
Python. The native circuit is extended in ONE call the next time something
needs it (sampling, drawing, compiling, ...), and only with the gates added
since the last time. Nothing on the simulation path runs per gate in Python.
"""

from __future__ import annotations

import os
from array import array
from collections.abc import Mapping
from html import escape as _escape
from numbers import Real

from ._native import native as _q
from .errors import QumoError, QumoTypeError, from_native

__all__ = ["Circuit", "Parameter", "load", "load_qasm", "from_stim"]

_OPS = _q.op_table()  # name -> (kind, arity, num_params)
_C_IF = -1
_MEASURE = _OPS["measure"][0]
_NativeCircuit = (_q.Circuit,)  # a tuple: doctest cannot introspect the extension type as a module global


class Parameter:
    """A named symbolic angle, optionally scaled and shifted: ``coeff * name + offset``.

    Parameters are matched by name, so the same object (or the same name as a
    plain string) can be used in several gates and several circuits.

    Example:
        >>> import qumo
        >>> theta = qumo.Parameter("theta")
        >>> c = qumo.Circuit(1).rx(2 * theta + 0.5, 0)
        >>> c.parameter_names
        ['theta']
    """

    __slots__ = ("name", "coeff", "offset")

    def __init__(self, name: str, coeff: float = 1.0, offset: float = 0.0):
        if not isinstance(name, str) or not name:
            raise QumoTypeError("a parameter name is a non-empty str", "QM-E911", "e.g. qumo.Parameter('theta').")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "coeff", float(coeff))
        object.__setattr__(self, "offset", float(offset))

    def __setattr__(self, key, value):
        raise AttributeError("Parameter is immutable")

    def __reduce__(self):
        return (Parameter, (self.name, self.coeff, self.offset))

    def __neg__(self):
        return Parameter(self.name, -self.coeff, -self.offset)

    def __pos__(self):
        return self

    def __add__(self, other):
        if isinstance(other, Real):
            return Parameter(self.name, self.coeff, self.offset + other)
        if isinstance(other, Parameter) and other.name == self.name:
            return Parameter(self.name, self.coeff + other.coeff, self.offset + other.offset)
        if isinstance(other, Parameter):
            raise QumoError(
                f"cannot add parameters '{self.name}' and '{other.name}': a gate angle is an affine function of "
                "one parameter", "QM-E913",
                "declare a third parameter for the sum, or split the rotation into two gates.")
        return NotImplemented

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, (Real, Parameter)):
            return self + (-other)
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, Real):
            return (-self) + other
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Real):
            return Parameter(self.name, self.coeff * other, self.offset * other)
        return NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, Real):
            return Parameter(self.name, self.coeff / other, self.offset / other)
        return NotImplemented

    def __eq__(self, other):
        return (isinstance(other, Parameter) and other.name == self.name and other.coeff == self.coeff
                and other.offset == self.offset)

    def __hash__(self):
        return hash((self.name, self.coeff, self.offset))

    def __repr__(self):
        text = self.name if self.coeff == 1.0 else f"{self.coeff:g}*{self.name}"
        if self.offset:
            text += f" {'+' if self.offset > 0 else '-'} {abs(self.offset):g}"
        return f"Parameter({text})"


def _fixed1(name: str, what: str):
    kind = _OPS[name][0]

    def gate(self, q: int):
        if q.__class__ is not int:
            q = self._index(name, q)
        if not 0 <= q < self._n:
            self._bad_qubit(name, q)
        self._ops.extend((kind, q, -1, -1, -1))
        return self

    gate.__name__ = gate.__qualname__ = name
    gate.__doc__ = f"{what} on qubit `q`. Returns the circuit, so calls chain."
    return gate


def _fixed2(name: str, what: str, a: str = "a", b: str = "b"):
    kind = _OPS[name][0]

    def gate(self, q0: int, q1: int):
        if q0.__class__ is not int or q1.__class__ is not int:
            q0 = self._index(name, q0)
            q1 = self._index(name, q1)
        n = self._n
        if not (0 <= q0 < n and 0 <= q1 < n) or q0 == q1:
            self._bad_qubits(name, (q0, q1))
        self._ops.extend((kind, q0, q1, -1, -1))
        return self

    gate.__name__ = gate.__qualname__ = name
    gate.__doc__ = f"{what} ({a} = first argument, {b} = second). Returns the circuit."
    return gate


def _fixed3(name: str, what: str):
    kind = _OPS[name][0]

    def gate(self, q0: int, q1: int, q2: int):
        if q0.__class__ is not int or q1.__class__ is not int or q2.__class__ is not int:
            q0 = self._index(name, q0)
            q1 = self._index(name, q1)
            q2 = self._index(name, q2)
        n = self._n
        if not (0 <= q0 < n and 0 <= q1 < n and 0 <= q2 < n) or q0 == q1 or q0 == q2 or q1 == q2:
            self._bad_qubits(name, (q0, q1, q2))
        self._ops.extend((kind, q0, q1, q2, -1))
        return self

    gate.__name__ = gate.__qualname__ = name
    gate.__doc__ = f"{what}. Returns the circuit."
    return gate


def _rot1(name: str, what: str, angle: str = "theta"):
    kind = _OPS[name][0]

    def gate(self, theta, q: int):
        if q.__class__ is not int:
            q = self._index(name, q)
        if not 0 <= q < self._n:
            self._bad_qubit(name, q)
        self._par.extend((theta, -1.0, 0.0) if theta.__class__ is float else self._angle(name, theta))
        self._ops.extend((kind, q, -1, -1, -1))
        return self

    gate.__name__ = gate.__qualname__ = name
    gate.__doc__ = (f"{what} by `{angle}` on qubit `q`. The angle is a number, a `Parameter` (or an affine "
                    "expression of one, e.g. ``2 * theta + 0.5``) or a parameter name. Returns the circuit.")
    return gate


def _rot2(name: str, what: str):
    kind = _OPS[name][0]

    def gate(self, theta, q0: int, q1: int):
        if q0.__class__ is not int or q1.__class__ is not int:
            q0 = self._index(name, q0)
            q1 = self._index(name, q1)
        n = self._n
        if not (0 <= q0 < n and 0 <= q1 < n) or q0 == q1:
            self._bad_qubits(name, (q0, q1))
        self._par.extend((theta, -1.0, 0.0) if theta.__class__ is float else self._angle(name, theta))
        self._ops.extend((kind, q0, q1, -1, -1))
        return self

    gate.__name__ = gate.__qualname__ = name
    gate.__doc__ = (f"{what} by `theta` on qubits `q0`, `q1`. The angle is a number, a `Parameter` or a "
                    "parameter name. Returns the circuit.")
    return gate


def _channel(name: str, what: str, arity: int):
    kind = _OPS[name][0]

    if arity == 1:
        def gate(self, p: float, q: int):
            if q.__class__ is not int:
                q = self._index(name, q)
            if not 0 <= q < self._n:
                self._bad_qubit(name, q)
            self._par.extend((self._probability(name, p), -1.0, 0.0))
            self._ops.extend((kind, q, -1, -1, -1))
            return self
    else:
        def gate(self, p: float, q0: int, q1: int):
            if q0.__class__ is not int or q1.__class__ is not int:
                q0 = self._index(name, q0)
                q1 = self._index(name, q1)
            n = self._n
            if not (0 <= q0 < n and 0 <= q1 < n) or q0 == q1:
                self._bad_qubits(name, (q0, q1))
            self._par.extend((self._probability(name, p), -1.0, 0.0))
            self._ops.extend((kind, q0, q1, -1, -1))
            return self

    gate.__name__ = gate.__qualname__ = name
    gate.__doc__ = f"Noise channel: {what} with probability `p`. Returns the circuit."
    return gate


class Circuit:
    """A quantum circuit on `num_qubits` qubits.

    Gate methods return the circuit, so they chain. Angles come first, qubits
    after, as in the Mojo API. Classical bits appear as measurements name them.

    Example:
        >>> import qumo
        >>> bell = qumo.Circuit(2).h(0).cx(0, 1).measure_all()
        >>> print(bell.draw())               # doctest: +SKIP
        >>> sorted(qumo.sample(bell, shots=1000, seed=7))
        ['00', '11']

    Parameters by name:
        >>> c = qumo.Circuit(1).ry("theta", 0).measure_all()
        >>> c.parameter_names
        ['theta']
        >>> counts = qumo.sample(c, shots=100, params={"theta": 3.14159265})
    """

    __slots__ = ("_n", "_nat", "_ops", "_par", "_new", "_pindex", "_nclbits", "__weakref__")

    def __init__(self, num_qubits: int):
        if isinstance(num_qubits, bool) or not isinstance(num_qubits, int):
            try:
                num_qubits = num_qubits.__index__()
            except AttributeError:
                raise QumoTypeError(
                    f"Circuit(num_qubits) takes an int, got {type(num_qubits).__name__}", "QM-E911",
                    "e.g. qumo.Circuit(2); to read a file or OpenQASM text use qumo.load(...).") from None
        if num_qubits < 1:
            raise QumoError(f"a circuit needs at least one qubit, got {num_qubits}", "QM-E913",
                            "e.g. qumo.Circuit(2).")
        self._n = num_qubits
        self._nat = _q.circuit(num_qubits)
        self._ops = array("q")
        self._par = array("d")
        self._new = []
        self._pindex = {}
        self._nclbits = 0

    # ------------------------------------------------------------------ native bridge
    @classmethod
    def from_native(cls, nat) -> "Circuit":
        """Wrap a `qumo_py.Circuit` (shared, not copied)."""
        if not isinstance(nat, _NativeCircuit):
            raise QumoTypeError(f"expected a qumo_py.Circuit, got {type(nat).__name__}", "QM-E911",
                                "build one with qumo.Circuit(n) or qumo.load(...).")
        self = object.__new__(cls)
        self._n = nat.num_qubits()
        self._nat = nat
        self._ops = array("q")
        self._par = array("d")
        self._new = []
        self._pindex = {name: i for i, name in enumerate(nat.parameter_names())}
        self._nclbits = nat.num_clbits()
        return self

    def _flush(self):
        """The native circuit, brought up to date in one call."""
        ops = self._ops
        if ops:
            par = self._par
            new = self._new
            self._ops = array("q")
            self._par = array("d")
            self._new = []
            try:
                self._nat.extend_packed(ops.buffer_info()[0], len(ops) // 5, par.buffer_info()[0], len(par), new)
            except Exception as exc:
                raise from_native(exc) from None
        elif self._new:
            new, self._new = self._new, []
            self._nat.extend_packed(0, 0, 0, 0, new)
        return self._nat

    @property
    def native(self):
        """The underlying `qumo_py.Circuit`, for the low-level `qumo_py` functions."""
        return self._flush()

    # ------------------------------------------------------------------ argument checks
    def _bad_qubit(self, gate: str, q):
        raise QumoError(
            f"{gate}() uses qubit {q} but the circuit has {self._n} qubit(s)", "QM-E001",
            f"valid qubits are 0..{self._n - 1}; create a wider circuit with qumo.Circuit({max(int(q) + 1, 1)}).")

    def _bad_qubits(self, gate: str, qs):
        for q in qs:
            if not 0 <= q < self._n:
                self._bad_qubit(gate, q)
        raise QumoError(f"{gate}() repeats a qubit: {tuple(int(q) for q in qs)}", "QM-E002",
                        "a multi-qubit gate acts on distinct qubits.")

    def _probability(self, gate: str, p) -> float:
        if isinstance(p, bool) or not isinstance(p, Real):
            raise QumoTypeError(f"{gate}(): the probability is a number, got {type(p).__name__}", "QM-E911",
                                f"e.g. c.{gate}(0.01, 0): probability first, qubits after.")
        p = float(p)
        if not 0.0 <= p <= 1.0:
            raise QumoError(f"{gate}(): probability {p} is outside [0, 1]", "QM-E913", "pass a probability.")
        return p

    def _index(self, gate: str, q) -> int:
        """A qubit number as an int (NumPy integers pass; bool, float and str do not)."""
        if isinstance(q, bool) or not hasattr(q, "__index__"):
            raise QumoTypeError(
                f"{gate}(): a qubit is an int, got {type(q).__name__} ({q!r})", "QM-E911",
                f"qubits are numbered 0..{self._n - 1}; angles come before qubits, e.g. c.rx(0.5, 0).")
        return q.__index__()

    def _angle(self, gate: str, theta) -> tuple[float, float, float]:
        """One angle as (offset, parameter index, coeff); index -1 = a constant."""
        if isinstance(theta, Parameter):
            return (theta.offset, float(self._declare(theta.name)), theta.coeff)
        if isinstance(theta, str):
            return (0.0, float(self._declare(theta)), 1.0)
        if isinstance(theta, Real) and not isinstance(theta, bool):
            return (float(theta), -1.0, 0.0)
        else:
            raise QumoTypeError(
                f"{gate}(): an angle is a number, a qumo.Parameter or a parameter name, got "
                f"{type(theta).__name__}", "QM-E911", f"e.g. c.{gate}(0.5, ...) or c.{gate}('theta', ...).")

    def _declare(self, name: str) -> int:
        index = self._pindex.get(name)
        if index is None:
            if not name:
                raise QumoError("a parameter name cannot be empty", "QM-E913", "e.g. 'theta'.")
            index = len(self._pindex)
            self._pindex[name] = index
            self._new.append(name)
        return index

    # ------------------------------------------------------------------ parameters
    def parameter(self, name: str) -> Parameter:
        """Declare (or fetch) the symbolic parameter `name`.

        Declaring fixes its position in `parameter_names`; using a new name
        directly in a gate (``c.rx("theta", 0)``) declares it too.
        """
        if not isinstance(name, str):
            raise QumoTypeError(f"parameter(name) takes a str, got {type(name).__name__}", "QM-E911",
                                "e.g. theta = c.parameter('theta').")
        self._declare(name)
        return Parameter(name)

    def parameters(self, prefix: str, count: int) -> list[Parameter]:
        """Declare `count` parameters named ``prefix[0]`` .. ``prefix[count-1]``."""
        return [self.parameter(f"{prefix}[{i}]") for i in range(count)]

    # ------------------------------------------------------------------ gates
    id = _fixed1("id", "Identity")
    x = _fixed1("x", "Pauli X")
    y = _fixed1("y", "Pauli Y")
    z = _fixed1("z", "Pauli Z")
    h = _fixed1("h", "Hadamard")
    s = _fixed1("s", "S = sqrt(Z)")
    sdg = _fixed1("sdg", "S dagger")
    t = _fixed1("t", "T = Z^(1/4)")
    tdg = _fixed1("tdg", "T dagger")
    sx = _fixed1("sx", "sqrt(X)")
    sxdg = _fixed1("sxdg", "sqrt(X) dagger")
    reset = _fixed1("reset", "Reset to |0>")

    rx = _rot1("rx", "Rotation about X")
    ry = _rot1("ry", "Rotation about Y")
    rz = _rot1("rz", "Rotation about Z")
    p = _rot1("p", "Phase gate diag(1, e^{i lam})", "lam")

    cx = _fixed2("cx", "Controlled X (CNOT)", "control", "target")
    cy = _fixed2("cy", "Controlled Y", "control", "target")
    cz = _fixed2("cz", "Controlled Z")
    ch = _fixed2("ch", "Controlled Hadamard", "control", "target")
    swap = _fixed2("swap", "SWAP")
    iswap = _fixed2("iswap", "iSWAP")
    ecr = _fixed2("ecr", "Echoed cross-resonance")

    cp = _rot2("cp", "Controlled phase")
    crx = _rot2("crx", "Controlled RX")
    cry = _rot2("cry", "Controlled RY")
    crz = _rot2("crz", "Controlled RZ")
    rxx = _rot2("rxx", "XX rotation exp(-i theta/2 XX)")
    ryy = _rot2("ryy", "YY rotation exp(-i theta/2 YY)")
    rzz = _rot2("rzz", "ZZ rotation exp(-i theta/2 ZZ)")

    ccx = _fixed3("ccx", "Toffoli: X on the third qubit controlled by the first two")
    cswap = _fixed3("cswap", "Fredkin: swap the last two qubits controlled by the first")
    ccz = _fixed3("ccz", "Doubly controlled Z")

    depolarize1 = _channel("depolarize1", "a uniformly random Pauli on one qubit", 1)
    depolarize2 = _channel("depolarize2", "a uniformly random two-qubit Pauli", 2)
    x_error = _channel("x_error", "an X flip", 1)
    y_error = _channel("y_error", "a Y flip", 1)
    z_error = _channel("z_error", "a Z flip", 1)

    def u(self, theta, phi, lam, q: int):
        """General one-qubit gate U(theta, phi, lam) on qubit `q` (OpenQASM 3 convention)."""
        if q.__class__ is not int:
            q = self._index("u", q)
        if not 0 <= q < self._n:
            self._bad_qubit("u", q)
        angles = self._angle("u", theta) + self._angle("u", phi) + self._angle("u", lam)
        self._par.extend(angles)
        self._ops.extend((_OPS["u"][0], q, -1, -1, -1))
        return self

    def pauli_channel_1(self, px: float, py: float, pz: float, q: int):
        """Noise channel: X, Y or Z on qubit `q` with probabilities `px`, `py`, `pz`."""
        if q.__class__ is not int:
            q = self._index("pauli_channel_1", q)
        if not 0 <= q < self._n:
            self._bad_qubit("pauli_channel_1", q)
        probs = [self._probability("pauli_channel_1", p) for p in (px, py, pz)]
        if sum(probs) > 1.0 + 1e-12:
            raise QumoError(f"pauli_channel_1(): probabilities sum to {sum(probs)}", "QM-E913", "px + py + pz <= 1.")
        self._par.extend((probs[0], -1.0, 0.0, probs[1], -1.0, 0.0, probs[2], -1.0, 0.0))
        self._ops.extend((_OPS["pauli_channel_1"][0], q, -1, -1, -1))
        return self

    def kraus1(self, name: str, matrices, q: int):
        """General noise channel on qubit `q` given by its Kraus operators (non-Pauli noise).

        `matrices` is a list of 2x2 complex matrices K_k with sum_k K_k^dagger K_k = I
        (the channel is rho -> sum_k K_k rho K_k^dagger). `name` labels the channel in
        drawings and in the OpenQASM text, where it is written as
        ``#pragma qumo kraus1 "name" (re, im, ...) q[q];``. Density-matrix and
        trajectory simulation take it; the stabilizer and Pauli-frame samplers do not.
        Returns the circuit.

        Example:
            >>> import qumo
            >>> damp = [[[1, 0], [0, (1 - 0.3) ** 0.5]], [[0, 0.3 ** 0.5], [0, 0]]]
            >>> float(round(qumo.probabilities(qumo.Circuit(1).x(0).kraus1("damp", damp, 0))[1], 6))
            0.7
        """
        return self._kraus("kraus1", name, matrices, 1, (q,))

    def kraus2(self, name: str, matrices, q0: int, q1: int):
        """General two-qubit noise channel on `q0`, `q1` given by its Kraus operators.

        `matrices` is a list of 4x4 complex matrices, row-major, with
        sum_k K_k^dagger K_k = I. See `kraus1`. Returns the circuit.
        """
        return self._kraus("kraus2", name, matrices, 2, (q0, q1))

    def _kraus(self, gate: str, name, matrices, arity: int, qubits):
        if not isinstance(name, str):
            raise QumoTypeError(f"{gate}(): the channel name is a str, got {type(name).__name__}", "QM-E911",
                                f"e.g. c.{gate}('damp', [K0, K1], 0).")
        if not name or '"' in name:
            raise QumoError(f"{gate}(): the channel name must be non-empty and contain no double quote", "QM-E913",
                            "e.g. 'amplitude_damping'.")
        qs = tuple(self._index(gate, q) for q in qubits)
        for q in qs:
            if not 0 <= q < self._n:
                self._bad_qubit(gate, q)
        if arity == 2 and qs[0] == qs[1]:
            self._bad_qubits(gate, qs)
        dim = 1 << arity
        try:
            ops = list(matrices)
        except TypeError:
            raise QumoTypeError(f"{gate}(): matrices is a list of Kraus operators, got {type(matrices).__name__}",
                                "QM-E911", f"e.g. [[[1, 0], [0, 0.8]], [[0, 0.6], [0, 0]]] for {gate}.") from None
        if not ops:
            raise QumoError(f"{gate}(): the channel has no Kraus operators", "QM-E414",
                            f"pass at least one {dim}x{dim} matrix.")
        data = array("d")
        for m in ops:
            try:
                rows = [list(row) for row in m]
            except TypeError:
                raise QumoTypeError(f"{gate}(): each Kraus operator is a {dim}x{dim} matrix", "QM-E911",
                                    f"e.g. [[1, 0], [0, 0]] for kraus1.") from None
            if len(rows) != dim or any(len(row) != dim for row in rows):
                raise QumoError(f"{gate}(): a Kraus operator has the wrong shape for {arity} qubit(s); expected "
                                f"{dim}x{dim}", "QM-E414", f"matrices are {dim}x{dim} for {gate}.")
            for row in rows:
                for x in row:
                    try:
                        z = complex(x)
                    except (TypeError, ValueError):
                        raise QumoTypeError(f"{gate}(): matrix entries are numbers, got {x!r}", "QM-E911",
                                            "e.g. 1, 0.5j or 0.5+0.5j.") from None
                    data.append(z.real)
                    data.append(z.imag)
        nat = self._flush()
        _call(nat.add_kraus, name, arity, data, qs[0], qs[1] if arity == 2 else -1)
        return self

    def barrier(self):
        """A barrier over all qubits: the compiler does not optimise across it."""
        self._ops.extend((_OPS["barrier"][0], -1, -1, -1, -1))
        return self

    def measure(self, q: int, c: int | None = None):
        """Measure qubit `q` into classical bit `c` (default: the bit with the same number).

        The classical register grows to hold bit `c`. In the bitstrings that
        `qumo.sample` returns, bit 0 is the rightmost character.
        """
        if q.__class__ is not int:
            q = self._index("measure", q)
        if not 0 <= q < self._n:
            self._bad_qubit("measure", q)
        if c is None:
            c = q
        else:
            if c.__class__ is not int:
                c = self._index("measure", c)
            if c < 0:
                raise QumoError(f"measure(): classical bit {c} is negative", "QM-E913",
                                "classical bits start at 0: c.measure(0, 0), or c.measure_all().")
        self._ops.extend((_MEASURE, q, -1, -1, c))
        if c >= self._nclbits:
            self._nclbits = c + 1
        return self

    def measure_all(self):
        """Measure every qubit into fresh classical bits (qubit q -> the next free bit + q)."""
        base = self._nclbits
        ops = self._ops
        for q in range(self._n):
            ops.extend((_MEASURE, q, -1, -1, base + q))
        self._nclbits = base + self._n
        return self

    def c_if(self, clbit: int, value: int = 1):
        """Condition the most recently added instruction on classical bit `clbit` being `value`.

        Example:
            >>> import qumo
            >>> c = qumo.Circuit(2).h(0).measure(0, 0).x(1).c_if(0, 1).measure(1, 1)
        """
        if not self._ops and self._nat.num_ops() == 0:
            raise QumoError("c_if() has no instruction to condition", "QM-E913",
                            "add the gate first: c.x(1).c_if(0, 1).")
        if clbit.__class__ is not int:
            clbit = self._index("c_if", clbit)
        if not 0 <= clbit < self._nclbits:
            raise QumoError(
                f"c_if() conditions on classical bit {clbit!r} but the circuit has {self._nclbits}", "QM-E003",
                "measure into that bit first: c.measure(q, clbit).")
        if value not in (0, 1):
            raise QumoError(f"c_if() value must be 0 or 1, got {value!r}", "QM-E913", "one classical bit is 0 or 1.")
        self._ops.extend((_C_IF, clbit, int(value), -1, -1))
        return self

    # ------------------------------------------------------------------ inspection
    @property
    def num_qubits(self) -> int:
        return self._n

    @property
    def num_clbits(self) -> int:
        return self._nclbits

    @property
    def parameter_names(self) -> list[str]:
        """Names of the unbound parameters, in the order positional values are read."""
        return list(self._pindex)

    @property
    def num_parameters(self) -> int:
        return len(self._pindex)

    def __len__(self) -> int:
        """Number of instructions of every kind."""
        return self._flush().num_ops()

    def size(self) -> int:
        """Number of unitary gates."""
        return self._flush().size()

    def depth(self) -> int:
        """Circuit depth over unitary gates."""
        return self._flush().depth()

    def depth_of(self, name) -> int:
        """Depth counting only the gates called `name` (a name or a list of names).

        Example:
            >>> import qumo
            >>> qumo.Circuit(3).cz(0, 1).h(1).cz(1, 2).depth_of("cz")
            2
        """
        return _call(self._flush().depth_of, name)

    def count_ops(self) -> dict[str, int]:
        """``{instruction name: count}``."""
        return dict(self._flush().count_ops())

    def count_gate(self, name: str) -> int:
        if name not in _OPS:
            raise QumoError(f"unknown gate name '{name}'", "QM-E913", "names are lower case: 'cx', 'cz', 'rz', ...")
        return self._flush().count_gate(name)

    def num_2q_gates(self) -> int:
        return self._flush().num_2q_gates()

    def t_count(self) -> int:
        return self._flush().t_count()

    def two_qubit_edges(self) -> list[tuple[int, int]]:
        """``[(q0, q1), ...]`` for every two-qubit gate, in order."""
        return list(self._flush().two_qubit_edges())

    def has_noise(self) -> bool:
        return bool(self._flush().has_noise())

    def has_measurements(self) -> bool:
        return self._flush().num_measurements() > 0

    def is_clifford(self) -> bool:
        return bool(self._flush().is_clifford())

    def is_dynamic(self) -> bool:
        """True with classical conditions, resets or gates after a measurement."""
        return bool(self._flush().is_dynamic())

    # ------------------------------------------------------------------ transformations
    def copy(self) -> "Circuit":
        """An independent copy."""
        return Circuit.from_native(self._flush().copy())

    def compose(self, other: "Circuit") -> "Circuit":
        """A new circuit: this one followed by `other` (same qubits)."""
        return Circuit.from_native(_call(self._flush().compose, _native_of(other)))

    def inverse(self) -> "Circuit":
        """The inverse of a unitary circuit (no measurements)."""
        return Circuit.from_native(_call(self._flush().inverse))

    def bind(self, values=None, **named) -> "Circuit":
        """A new circuit with the parameters replaced by numbers.

        `values` is a dict ``{name: value}`` (every parameter must be named) or
        a sequence in `parameter_names` order; keyword arguments work when the
        names are identifiers.

        Example:
            >>> import qumo
            >>> c = qumo.Circuit(1).rx("theta", 0)
            >>> c.bind({"theta": 0.5}).num_parameters
            0
            >>> c.bind(theta=0.5).num_parameters
            0
        """
        if named:
            if values is not None:
                raise QumoError("bind() takes a dict/sequence or keyword arguments, not both", "QM-E913",
                                "e.g. c.bind({'theta': 0.5}).")
            values = named
        nat = self._flush()
        return Circuit.from_native(_call(nat.bind, param_values(self, values)))

    # ------------------------------------------------------------------ text forms
    def draw(self, width: int = 80, style: str = "unicode") -> str:
        """Text drawing. `style`: 'unicode' or 'ascii'; `width` 0 = do not fold.

        Example:
            >>> import qumo
            >>> print(qumo.Circuit(2).h(0).cx(0, 1).draw(style="ascii"))   # doctest: +SKIP
        """
        return str(_call(_q.draw, self._flush(), width, style))

    def to_qasm(self, version: int = 3) -> str:
        """OpenQASM text: `version` 3 (default; `qumo.load` reads it back exactly) or 2."""
        if version == 3:
            return str(_call(self._flush().to_qasm3))
        if version == 2:
            return str(_call(self._flush().to_qasm2))
        raise QumoError(f"to_qasm(version={version!r}): unknown OpenQASM version", "QM-E913", "use 2 or 3.")

    def to_qasm2(self) -> str:
        return self.to_qasm(2)

    def to_qasm3(self) -> str:
        return self.to_qasm(3)

    def to_stim(self) -> str:
        """Stim circuit text (Clifford gates and Pauli noise)."""
        return str(_call(_q.to_stim, self._flush()))

    def text(self) -> str:
        """QUMO's own listing, one instruction per line."""
        return str(self._flush().text())

    # ------------------------------------------------------------------ Python protocol
    def __repr__(self) -> str:
        nat = self._flush()
        parts = [f"{self._n} qubit{'s' if self._n != 1 else ''}"]
        if self._nclbits:
            parts.append(f"{self._nclbits} clbit{'s' if self._nclbits != 1 else ''}")
        parts.append(f"{nat.size()} gates")
        parts.append(f"depth {nat.depth()}")
        if self._pindex:
            parts.append("parameters " + ", ".join(self._pindex))
        return "<qumo.Circuit " + ", ".join(parts) + ">"

    def __str__(self) -> str:
        if self._flush().num_ops() > _DRAW_LIMIT:
            return repr(self)
        return self.draw()

    def _repr_pretty_(self, printer, cycle):  # IPython / Jupyter text display
        printer.text(repr(self) if cycle else repr(self) + ("\n" + self.draw() if len(self) <= _DRAW_LIMIT else ""))

    def _repr_html_(self) -> str:  # Jupyter rich display
        head = _escape(repr(self))
        if len(self) > _DRAW_LIMIT:
            return f"<code>{head}</code>"
        body = _escape(self.draw(width=0))
        return (f"<div><code>{head}</code>"
                f"<pre style=\"line-height:1.15;overflow-x:auto;white-space:pre\">{body}</pre></div>")

    def __eq__(self, other) -> bool:
        """Same qubits, parameters and instructions in the same order (not unitary equivalence)."""
        if not isinstance(other, Circuit):
            return NotImplemented
        return self.to_qasm(3) == other.to_qasm(3)

    __hash__ = None

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy()

    def __reduce__(self):
        # OpenQASM 3 text is the stable exchange form: pickles survive rebuilds of the extension
        return (_from_qasm3, (self.to_qasm(3),))


_DRAW_LIMIT = 400  # instructions past which str() and the notebook display show the summary only


def _from_qasm3(text: str) -> Circuit:
    return Circuit.from_native(_call(_q.load_qasm, text))


def _call(fn, *args, **kwargs):
    """Call into the extension, turning its bare Exception into a QumoError."""
    try:
        return fn(*args, **kwargs)
    except QumoError:
        raise
    except Exception as exc:
        raise from_native(exc) from None


def _native_of(circuit):
    """The up-to-date `qumo_py.Circuit` behind a `qumo.Circuit` (or a raw one)."""
    if circuit.__class__ is Circuit:
        return circuit._flush()
    if isinstance(circuit, Circuit):
        return circuit._flush()
    if isinstance(circuit, _NativeCircuit):
        return circuit
    if isinstance(circuit, (str, os.PathLike)):
        raise QumoTypeError(
            f"expected a qumo.Circuit, got {type(circuit).__name__}", "QM-E911",
            "read OpenQASM text or a file first: circuit = qumo.load(...).")
    raise QumoTypeError(f"expected a qumo.Circuit, got {type(circuit).__name__}", "QM-E911",
                        "build one with qumo.Circuit(n) or qumo.load(...).")


def param_values(circuit, values) -> list[float]:
    """Parameter values as the list the extension takes, from a dict or a sequence."""
    nat = _native_of(circuit)
    names = list(circuit._pindex) if isinstance(circuit, Circuit) else list(nat.parameter_names())
    if values is None:
        values = ()
    if isinstance(values, Mapping):
        missing = [n for n in names if n not in values]
        extra = [k for k in values if k not in names]
        if missing or extra:
            problem = []
            if missing:
                problem.append("no value for " + ", ".join(repr(n) for n in missing))
            if extra:
                problem.append("unknown parameter " + ", ".join(repr(k) for k in extra))
            raise QumoError("; ".join(problem), "QM-E101",
                            f"the circuit's parameters are {names}." if names else "the circuit has no parameters.")
        out = [values[n] for n in names]
    else:
        if isinstance(values, (str, bytes)) or isinstance(values, Real):
            raise QumoTypeError(
                f"parameter values are a dict {{name: value}} or a sequence, got {type(values).__name__}",
                "QM-E911", f"the circuit's parameters are {names}.")
        out = list(values)
        if len(out) != len(names):
            raise QumoError(
                f"the circuit has {len(names)} parameter(s) {names} but {len(out)} value(s) were given", "QM-E101",
                "pass one value per parameter, or a dict {name: value}.")
    try:
        return [float(v) for v in out]
    except (TypeError, ValueError):
        raise QumoTypeError("parameter values must be numbers", "QM-E911",
                            "e.g. params={'theta': 0.5}.") from None


def _looks_like_qasm(text: str) -> bool:
    return "\n" in text or ";" in text or text.lstrip().startswith(("OPENQASM", "//"))


def load(source) -> Circuit:
    """Read a circuit from a file path or from OpenQASM 2 / 3 text.

    A `pathlib.Path` is always a file. A str is OpenQASM text when it has a
    newline or a ``;``; otherwise it is a path.

    Example:
        >>> import qumo
        >>> c = qumo.load("OPENQASM 3.0; qubit[2] q; bit[2] c; h q[0]; cx q[0], q[1]; c = measure q;")
        >>> c.num_qubits
        2
        >>> c = qumo.load("examples/foundations/bell.qasm")          # doctest: +SKIP

    Raises:
        FileNotFoundError: the path does not exist.
        QumoError: the text is not valid OpenQASM (QM-E5xx, with line and column).
    """
    if isinstance(source, Circuit):
        return source.copy()
    if isinstance(source, os.PathLike):
        path = os.fspath(source)
    elif isinstance(source, str):
        if _looks_like_qasm(source):
            return Circuit.from_native(_call(_q.load_qasm, source))
        path = source
    else:
        raise QumoTypeError(f"load() takes a path or OpenQASM text, got {type(source).__name__}", "QM-E911",
                            "e.g. qumo.load('bell.qasm').")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"no such file: {path!r}. qumo.load() takes a file path or OpenQASM text; text has a newline or ';' "
            "(e.g. 'OPENQASM 3.0; qubit[1] q; h q[0];').")
    return Circuit.from_native(_call(_q.load_qasm_file, path))


load_qasm = load


def from_stim(text: str) -> Circuit:
    """Read Stim circuit text (Clifford gates, Pauli noise, detectors, observables)."""
    return Circuit.from_native(_call(_q.from_stim, text))
