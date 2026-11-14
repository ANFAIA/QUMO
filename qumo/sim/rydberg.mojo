"""State-vector simulation of analog Rydberg programs (qumo.ir.analog), for small registers.

    H(t) = sum_i Omega(t)/2 (cos(phi) X_i - sin(phi) Y_i) - delta(t) sum_i n_i + sum_{i<j} C6/r_ij^6 n_i n_j

Basis state bit i = 1 means atom i is in |r>. Time evolution uses a second-order Trotter
splitting per step: half the diagonal part (detuning + van der Waals), the drive as exact
single-atom rotations, the other half of the diagonal. Times are converted from ns to us,
so energies are in rad/us. The reference has no noise (no SPAM, dephasing or relaxation),
which is why devices and noisy emulators give broader distributions.
"""

from std.collections import Dict
from std.math import cos, sin, sqrt

from qumo.ir.analog import AnalogProgram
from qumo.sim.rng import Rng

comptime MAX_ATOMS = 20


struct RydbergState(Movable):
    var n: Int
    var re: List[Float64]
    var im: List[Float64]

    def __init__(out self, n: Int):
        self.n = n
        var dim = 1 << n
        self.re = List[Float64](length=dim, fill=0.0)
        self.im = List[Float64](length=dim, fill=0.0)
        self.re[0] = 1.0  # all atoms in |g>

    def probabilities(self) -> List[Float64]:
        var out = List[Float64](capacity=len(self.re))
        for i in range(len(self.re)):
            out.append(self.re[i] * self.re[i] + self.im[i] * self.im[i])
        return out^


def _interaction_diagonal(prog: AnalogProgram, c6: Float64) -> List[Float64]:
    """sum_{i<j} C6 / r^6 over the excited pairs of each basis state."""
    var n = prog.num_atoms()
    var dim = 1 << n
    var v = List[Float64](length=dim, fill=0.0)
    for i in range(n):
        for j in range(i + 1, n):
            var dx = prog.xs[i] - prog.xs[j]
            var dy = prog.ys[i] - prog.ys[j]
            var r2 = dx * dx + dy * dy
            var vij = c6 / (r2 * r2 * r2)
            var mask = (1 << i) | (1 << j)
            for b in range(dim):
                if (b & mask) == mask:
                    v[b] += vij
    return v^


def _popcounts(n: Int) -> List[Int]:
    var dim = 1 << n
    var out = List[Int](length=dim, fill=0)
    for b in range(dim):
        var c = 0
        var x = b
        while x > 0:
            c += x & 1
            x >>= 1
        out[b] = c
    return out^


def _diag_phase(mut st: RydbergState, vdiag: List[Float64], pop: List[Int], delta: Float64, dt_us: Float64):
    for b in range(len(st.re)):
        var e = vdiag[b] - delta * Float64(pop[b])
        var c = cos(-e * dt_us)
        var s = sin(-e * dt_us)
        var r = st.re[b]
        var im = st.im[b]
        st.re[b] = r * c - im * s
        st.im[b] = r * s + im * c


def _drive(mut st: RydbergState, omega: Float64, phi: Float64, dt_us: Float64):
    """exp(-i dt Omega/2 (cos phi X - sin phi Y)) on every atom: [[c, -i s e^{i phi}], [-i s e^{-i phi}, c]]
    in the basis (|g>, |r>) = (bit 0, bit 1)."""
    var theta = omega * dt_us / 2.0
    var c = cos(theta)
    var s = sin(theta)
    # off-diagonal elements: from |r> to |g>: -i s e^{i phi}; from |g> to |r>: -i s e^{-i phi}
    var a_re = s * sin(phi)  # (-i s e^{i phi}) = s sin(phi) - i s cos(phi)
    var a_im = -s * cos(phi)
    var b_re = -s * sin(phi)  # (-i s e^{-i phi}) = -s sin(phi) - i s cos(phi)
    var b_im = -s * cos(phi)
    var dim = len(st.re)
    for q in range(st.n):
        var m = 1 << q
        for g in range(dim):
            if (g & m) != 0:
                continue
            var r = g | m
            var g_re = st.re[g]
            var g_im = st.im[g]
            var r_re = st.re[r]
            var r_im = st.im[r]
            st.re[g] = c * g_re + (a_re * r_re - a_im * r_im)
            st.im[g] = c * g_im + (a_re * r_im + a_im * r_re)
            st.re[r] = c * r_re + (b_re * g_re - b_im * g_im)
            st.im[r] = c * r_im + (b_re * g_im + b_im * g_re)


def simulate_analog(prog: AnalogProgram, c6: Float64, step_ns: Float64 = 1.0) raises -> List[Float64]:
    """Final measurement probabilities over the 2^n basis states (bit i = atom i in |r>)."""
    var n = prog.num_atoms()
    if n == 0 or n > MAX_ATOMS:
        raise Error("QM-E956: the local Rydberg simulator takes 1.." + String(MAX_ATOMS) + " atoms")
    var st = RydbergState(n)
    var vdiag = _interaction_diagonal(prog, c6)
    var pop = _popcounts(n)
    for p in prog.pulses:
        var dur = Float64(p.duration())
        var steps = max(1, Int(dur / step_ns + 0.5))
        var h = dur / Float64(steps)
        var dt_us = h / 1000.0
        for k in range(steps):
            var tm = (Float64(k) + 0.5) * h  # midpoint rule for the time dependence
            var om = p.amplitude.at(tm)
            var de = p.detuning.at(tm)
            _diag_phase(st, vdiag, pop, de, dt_us / 2.0)
            _drive(st, om, p.phase, dt_us)
            _diag_phase(st, vdiag, pop, de, dt_us / 2.0)
    return st.probabilities()


def bitstring_of(index: Int, n: Int) -> String:
    """Basis index -> Pulser bitstring: atom 0 is the leftmost character."""
    var s = String()
    for i in range(n):
        s += "1" if ((index >> i) & 1) == 1 else "0"
    return s^


def sample_analog(probs: List[Float64], n: Int, shots: Int, seed: UInt64) -> Dict[String, Int]:
    var cdf = List[Float64](capacity=len(probs))
    var acc = 0.0
    for p in probs:
        acc += p
        cdf.append(acc)
    var rng = Rng(seed)
    var out = Dict[String, Int]()
    for _ in range(shots):
        var u = rng.uniform() * acc
        var lo = 0
        var hi = len(cdf) - 1
        while lo < hi:
            var mid = (lo + hi) // 2
            if cdf[mid] > u:
                hi = mid
            else:
                lo = mid + 1
        var key = bitstring_of(lo, n)
        out[key] = out.get(key, 0) + 1
    return out^
