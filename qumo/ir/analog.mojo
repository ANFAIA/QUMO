"""Analog neutral-atom programs: an atom register driven by global Rydberg pulses.

The program model of analog Rydberg devices: atoms at fixed
positions in a plane, one global laser channel coupling |g> and |r>, and a sequence of
pulses. Each pulse has an amplitude waveform Omega(t) and a detuning waveform delta(t)
(rad/us) over a duration in ns, and a fixed phase phi (rad). The Hamiltonian is

    H(t) = sum_i Omega(t)/2 (cos(phi) X_i - sin(phi) Y_i) - delta(t) sum_i n_i
           + sum_{i<j} C6 / r_ij^6 n_i n_j,          n_i = |r><r|_i,

the convention of Pulser's ground-rydberg basis. Measurement reads each atom as 1 when it
is in |r>. Bitstrings list the atoms in register order, atom 0 first (leftmost).

`qumo.sim.rydberg` simulates programs locally. This file is the neutral program model only
and imports nothing: a vendor's wire format and device specs live with that vendor's client,
outside `qumo/` (Pasqal / Pulser: `plugins/hardware/native/qumo_hw/pasqal_pulser.mojo`, which
has `pulser_json`, `AnalogDevice` and `check_against_device`).
"""

comptime E_ANALOG = "QM-E955"


# ---------------------------------------------------------------------------
# Waveforms
# ---------------------------------------------------------------------------
struct WaveformKind:
    comptime CONSTANT = 0
    comptime RAMP = 1
    comptime INTERPOLATED = 2  # values at evenly spaced fractions of the duration, linear in between


struct Waveform(Copyable):
    var kind: Int
    var duration: Int  # ns
    var values: List[Float64]  # constant: [v]; ramp: [start, stop]; interpolated: [v0, ..., vk]

    def __init__(out self, kind: Int, duration: Int, var values: List[Float64]):
        self.kind = kind
        self.duration = duration
        self.values = values^

    @staticmethod
    def constant(duration: Int, value: Float64) -> Waveform:
        var v = List[Float64]()
        v.append(value)
        return Waveform(WaveformKind.CONSTANT, duration, v^)

    @staticmethod
    def ramp(duration: Int, start: Float64, stop: Float64) -> Waveform:
        var v = List[Float64]()
        v.append(start)
        v.append(stop)
        return Waveform(WaveformKind.RAMP, duration, v^)

    @staticmethod
    def interpolated(duration: Int, var values: List[Float64]) -> Waveform:
        return Waveform(WaveformKind.INTERPOLATED, duration, values^)

    def at(self, t_ns: Float64) -> Float64:
        """Value at time t in ns, 0 <= t <= duration."""
        if self.kind == WaveformKind.CONSTANT:
            return self.values[0]
        var f = t_ns / Float64(self.duration) if self.duration > 0 else 0.0
        f = min(max(f, 0.0), 1.0)
        if self.kind == WaveformKind.RAMP:
            return self.values[0] + (self.values[1] - self.values[0]) * f
        var n = len(self.values)
        if n == 1:
            return self.values[0]
        var pos = f * Float64(n - 1)
        var i = Int(pos)
        if i >= n - 1:
            return self.values[n - 1]
        var w = pos - Float64(i)
        return self.values[i] * (1.0 - w) + self.values[i + 1] * w

    def max_abs(self) -> Float64:
        var m = 0.0
        for v in self.values:
            m = max(m, abs(v))
        return m

    def mean(self) -> Float64:
        """Time average (exact for these piecewise-linear shapes)."""
        if self.kind == WaveformKind.CONSTANT:
            return self.values[0]
        if self.kind == WaveformKind.RAMP:
            return (self.values[0] + self.values[1]) / 2.0
        var n = len(self.values)
        if n == 1:
            return self.values[0]
        var s = 0.0
        for i in range(n - 1):
            s += (self.values[i] + self.values[i + 1]) / 2.0
        return s / Float64(n - 1)


struct AnalogPulse(Copyable):
    var amplitude: Waveform  # Omega, rad/us
    var detuning: Waveform  # delta, rad/us
    var phase: Float64  # rad

    def __init__(out self, var amplitude: Waveform, var detuning: Waveform, phase: Float64 = 0.0):
        self.amplitude = amplitude^
        self.detuning = detuning^
        self.phase = phase

    def duration(self) -> Int:
        return self.amplitude.duration


# ---------------------------------------------------------------------------
# Program
# ---------------------------------------------------------------------------
struct AnalogProgram(Copyable):
    var name: String
    var xs: List[Float64]  # atom coordinates, um
    var ys: List[Float64]
    var pulses: List[AnalogPulse]

    def __init__(out self, name: String = "qumo-analog"):
        self.name = name
        self.xs = List[Float64]()
        self.ys = List[Float64]()
        self.pulses = List[AnalogPulse]()

    def add_atom(mut self, x: Float64, y: Float64):
        self.xs.append(x)
        self.ys.append(y)

    def add_pulse(mut self, var p: AnalogPulse) raises:
        if p.amplitude.duration != p.detuning.duration:
            raise Error(E_ANALOG + ": pulse amplitude and detuning must have the same duration")
        self.pulses.append(p^)

    def num_atoms(self) -> Int:
        return len(self.xs)

    def duration(self) -> Int:
        var t = 0
        for p in self.pulses:
            t += p.duration()
        return t


def adiabatic_sweep(mut prog: AnalogProgram, omega: Float64, delta_start: Float64, delta_stop: Float64,
                    rise_ns: Int, sweep_ns: Int, fall_ns: Int) raises:
    """Adiabatic preparation of the blockade ground state (MIS): ramp Omega up at delta_start,
    sweep delta to delta_stop at constant Omega, ramp Omega down at delta_stop."""
    prog.add_pulse(AnalogPulse(Waveform.ramp(rise_ns, 0.0, omega), Waveform.constant(rise_ns, delta_start)))
    prog.add_pulse(AnalogPulse(Waveform.constant(sweep_ns, omega), Waveform.ramp(sweep_ns, delta_start, delta_stop)))
    prog.add_pulse(AnalogPulse(Waveform.ramp(fall_ns, omega, 0.0), Waveform.constant(fall_ns, delta_stop)))


def blockade_radius(c6: Float64, omega: Float64) -> Float64:
    """r_b = (C6 / Omega)^(1/6), um."""
    return (c6 / omega) ** (1.0 / 6.0)


def c6_for_level(n: Int) raises -> Float64:
    """C6 / hbar (rad um^6 / us) of the n S_1/2 Rydberg level of 87Rb, as Pulser tabulates it."""
    if n == 60:
        return 865723.02
    if n == 70:
        return 5420158.53
    raise Error(E_ANALOG + ": no C6 coefficient for Rydberg level " + String(n))
