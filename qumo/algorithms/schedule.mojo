"""Time-dependent analog schedules: H(t) = sum_k c_k(t) H_k.

Each named term is a PauliSum H_k with a coefficient c_k(t) given by knots
(times, values). Between knots the coefficient is linear (INTERP_LINEAR) or
held at the earlier knot (INTERP_STEP). Outside the knot range the value is
held at the nearest end.

`evolve_schedule` uses the midpoint rule with piecewise-constant steps: each
step applies exp(-i H(t_mid) dt) through `exact_evolve`. This is Trotter-free.
For smooth coefficients the error is second order in dt. For non-commuting
terms the error is not zero even when every step is exact.

Error codes: QM-E813 (schedule or knot errors), QM-E814 (qubit-count or step
mismatch).
"""

from std.math import isfinite
from qumo.ir.pauli import PauliSum
from qumo.sim.statevector import StateVector
from qumo.algorithms.hamiltonians import exact_evolve

comptime INTERP_LINEAR = 0  # linear between knots
comptime INTERP_STEP = 1  # the earlier knot holds until the next one


@fieldwise_init
struct _ScheduleTerm(Copyable):
    var label: String
    var h: PauliSum
    var times: List[Float64]
    var values: List[Float64]
    var interpolation: Int


struct AnalogSchedule(Copyable, Writable):
    """Named Hamiltonian terms with time-dependent coefficients on [0, total_time]."""

    var total_time: Float64
    var width: Int  # qubits of the schedule; every term acts on at most this many
    var terms: List[_ScheduleTerm]

    def __init__(out self, total_time: Float64, num_qubits: Int) raises:
        if not (total_time > 0.0) or not isfinite(total_time):
            raise Error("QM-E813: AnalogSchedule total_time must be > 0 and finite, got " + String(total_time) + ". Hint: pass the schedule duration in the same units as the Hamiltonian.")
        if num_qubits < 1 or num_qubits > 64:
            raise Error("QM-E814: AnalogSchedule needs 1 <= num_qubits <= 64, got " + String(num_qubits) + ". Hint: PauliSum masks hold 64 qubits.")
        self.total_time = total_time
        self.width = num_qubits
        self.terms = List[_ScheduleTerm]()

    def num_qubits(self) -> Int:
        """The qubit count the schedule was built for (its state must have this many)."""
        return self.width

    def add_term(mut self, label: String, h: PauliSum, times: List[Float64], values: List[Float64], interpolation: Int) raises:
        """Add the term label with H = h and coefficient c_label(t) given by knots.

        Raises QM-E813 for a duplicate label, an unknown interpolation code, a
        knot count mismatch, empty knots, or knot times that are not strictly
        increasing or lie outside [0, total_time]. Raises QM-E814 when h acts
        on more qubits than the schedule; a term on fewer qubits is padded
        with identities, so a one-qubit drive and a two-qubit coupling share
        one schedule.
        """
        for i in range(len(self.terms)):
            if self.terms[i].label == label:
                raise Error("QM-E813: duplicate schedule term '" + label + "'. Hint: give each term a unique label.")
        if interpolation != INTERP_LINEAR and interpolation != INTERP_STEP:
            raise Error("QM-E813: unknown interpolation code " + String(interpolation) + " for '" + label + "'. Hint: use INTERP_LINEAR (0) or INTERP_STEP (1).")
        var n = len(times)
        if n < 1 or n != len(values):
            raise Error("QM-E813: term '" + label + "' has " + String(n) + " knot times and " + String(len(values)) + " values. Hint: give at least one knot and one value per knot time.")
        for i in range(n):
            if not (times[i] >= 0.0 and times[i] <= self.total_time):
                raise Error("QM-E813: knot time " + String(times[i]) + " of '" + label + "' is outside [0, " + String(self.total_time) + "]. Hint: keep every knot inside the schedule.")
            if i > 0 and not (times[i] > times[i - 1]):
                raise Error("QM-E813: knot times of '" + label + "' are not strictly increasing at " + String(times[i]) + ". Hint: sort the knots and remove repeated times.")
            if not isfinite(values[i]):
                raise Error("QM-E813: knot value " + String(values[i]) + " of '" + label + "' is not finite. Hint: use finite coefficient values.")
        var hq = h.num_qubits()
        if hq > self.width:
            raise Error("QM-E814: term '" + label + "' acts on " + String(hq) + " qubits but the schedule has " + String(self.width) + ". Hint: build the schedule with AnalogSchedule(total_time, num_qubits) on at least " + String(hq) + " qubits.")
        self.terms.append(_ScheduleTerm(label.copy(), h.copy(), times.copy(), values.copy(), interpolation))

    def coefficient(self, label_index: Int, t: Float64) -> Float64:
        """c_label(t) for term number label_index (order of add_term).

        Outside the knot range the value is held at the nearest end knot.
        """
        ref term = self.terms[label_index]
        var n = len(term.times)
        if t <= term.times[0]:
            return term.values[0]
        if t >= term.times[n - 1]:
            return term.values[n - 1]
        var i = 0
        while i + 1 < n and term.times[i + 1] <= t:
            i += 1
        if term.interpolation == INTERP_STEP:
            return term.values[i]
        var frac = (t - term.times[i]) / (term.times[i + 1] - term.times[i])
        return term.values[i] + frac * (term.values[i + 1] - term.values[i])

    def hamiltonian_at(self, t: Float64) raises -> PauliSum:
        """H(t) = sum_k c_k(t) H_k, merged through PauliSum.add."""
        var out = PauliSum()
        for k in range(len(self.terms)):
            var c = self.coefficient(k, t)
            for term in self.terms[k].h.terms:
                out.add(c * term.coeff, term.x, term.z)
        return out^

    def write_to(self, mut w: Some[Writer]):
        w.write("AnalogSchedule(total_time=", self.total_time, ", terms=[")
        for i in range(len(self.terms)):
            if i > 0:
                w.write(", ")
            w.write(self.terms[i].label, " (", len(self.terms[i].times), " knots)")
        w.write("])")


def evolve_schedule(sched: AnalogSchedule, psi: StateVector, steps: Int) raises -> StateVector:
    """Evolve psi under H(t) by the midpoint rule with `steps` piecewise-constant steps.

    dt = total_time / steps; step k applies exp(-i H((k + 1/2) dt) dt) through
    `exact_evolve`. For smooth coefficients the global error is second order in
    dt. This is the Trotter-free piecewise-constant scheme: it is not exact for
    non-commuting time dependence. With commuting terms it is exact when each
    coefficient is linear on every step (the midpoint rule integrates a linear
    function exactly).

    Raises QM-E814 when steps < 1 or when psi.num_qubits differs from the
    schedule's qubit count.
    """
    if steps < 1:
        raise Error("QM-E814: evolve_schedule() needs steps >= 1, got " + String(steps) + ". Hint: use at least one step; more steps reduce the error.")
    var n = sched.num_qubits()
    if psi.num_qubits != n:
        raise Error("QM-E814: state has " + String(psi.num_qubits) + " qubits but the schedule acts on " + String(n) + ". Hint: prepare the state on the schedule's qubit count.")
    var dt = sched.total_time / Float64(steps)
    var state = psi.copy()
    for k in range(steps):
        var t_mid = (Float64(k) + 0.5) * dt
        state = exact_evolve(sched.hamiltonian_at(t_mid), state, dt)
    return state^
