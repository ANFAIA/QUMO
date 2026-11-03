"""QUBO model, Ising conversion, brute force and simulated annealing.

Energy convention: x_i in {0, 1}, bit i of a mask is x_i, qubit i is bit i:

    E(x) = offset + sum_i linear[i] x_i + sum_{i<j} Q_ij x_i x_j

`qubo_to_ising` uses x_i = (1 - z_i) / 2 with z_i the Pauli Z on qubit i, so the
basis state |b> has <b|H|b> = qubo_energy(q, b). `ising_to_qubo` inverts it for
diagonal sums (constants, Z and Z Z terms).

Error codes: QM-E802 (index or size), QM-E811 (brute force above 24 variables),
QM-E812 (Pauli sum that is not a QUBO), QM-E822 (annealing settings).
"""

from std.bit import pop_count
from std.collections import Dict
from std.math import exp, log
from qumo.ir.pauli import PauliSum
from qumo.sim.rng import Rng

comptime QUBO_MAX_VARS = 64  # masks are UInt64, as are the Pauli bit masks
comptime QUBO_BRUTE_MAX = 24


@always_inline
def _x(mask: UInt64, i: Int) -> Bool:
    return ((mask >> UInt64(i)) & 1) == 1


def _lowest_bit(z: UInt64) -> Int:
    for q in range(64):
        if ((z >> UInt64(q)) & 1) == 1:
            return q
    return -1


def _trailing_zeros(k: Int) -> Int:
    """Position of the lowest set bit of k >= 1."""
    var t = k
    var i = 0
    while (t & 1) == 0:
        t >>= 1
        i += 1
    return i


struct Qubo(Copyable, Writable):
    """offset + linear.x + sum_{i<j} Q_ij x_i x_j over num_vars binary variables."""

    var num_vars: Int
    var offset: Float64
    var linear: List[Float64]
    var quadratic: List[Tuple[Int, Int, Float64]]  # (i, j, coeff), i < j, one per pair
    var _slot: Dict[Int, Int]  # pair key i * num_vars + j -> index into quadratic

    def __init__(out self, num_vars: Int) raises:
        if num_vars < 0 or num_vars > QUBO_MAX_VARS:
            raise Error(
                "QM-E802: Qubo needs 0 <= num_vars <= 64, got " + String(num_vars)
                + ". Hint: a QUBO holds at most 64 variables, the width of a bit mask."
            )
        self.num_vars = num_vars
        self.offset = 0.0
        self.linear = List[Float64](length=num_vars, fill=0.0)
        self.quadratic = List[Tuple[Int, Int, Float64]]()
        self._slot = Dict[Int, Int]()

    def _check_index(self, i: Int) raises:
        if i < 0 or i >= self.num_vars:
            raise Error(
                "QM-E802: variable " + String(i) + " is outside [0, " + String(self.num_vars)
                + "). Hint: variables are numbered 0 to num_vars - 1."
            )

    def add_linear(mut self, i: Int, coeff: Float64) raises:
        """Add coeff to the linear term of x_i."""
        self._check_index(i)
        self.linear[i] += coeff

    def add_quadratic(mut self, i: Int, j: Int, coeff: Float64) raises:
        """Add coeff to Q_ij (repeated pairs are summed)."""
        if i == j:
            raise Error(
                "QM-E802: add_quadratic needs two distinct variables, got " + String(i)
                + " twice. Hint: x_i^2 = x_i, so put a diagonal term in add_linear."
            )
        self._check_index(i)
        self._check_index(j)
        var a = min(i, j)
        var b = max(i, j)
        var key = a * self.num_vars + b
        if key in self._slot:
            var k = self._slot[key]
            self.quadratic[k] = (a, b, self.quadratic[k][2] + coeff)
        else:
            self._slot[key] = len(self.quadratic)
            self.quadratic.append((a, b, coeff))

    def write_to(self, mut w: Some[Writer]):
        w.write(
            "Qubo(num_vars=", self.num_vars, ", offset=", self.offset,
            ", quadratic_terms=", len(self.quadratic), ")",
        )


@fieldwise_init
struct QuboSolution(Copyable, Writable):
    var mask: UInt64
    var energy: Float64
    var num_vars: Int

    def bit(self, i: Int) -> Int:
        """Value x_i of variable i in this solution."""
        if i < 0 or i >= QUBO_MAX_VARS:
            return 0
        return Int((self.mask >> UInt64(i)) & 1)

    def write_to(self, mut w: Some[Writer]):
        w.write(
            "QuboSolution(mask=", self.mask, ", energy=", self.energy,
            ", num_vars=", self.num_vars, ")",
        )


def qubo_energy(q: Qubo, mask: UInt64) -> Float64:
    """Energy of the assignment x_i = bit i of mask."""
    var e = q.offset
    for i in range(q.num_vars):
        if _x(mask, i):
            e += q.linear[i]
    for t in q.quadratic:
        if _x(mask, t[0]) and _x(mask, t[1]):
            e += t[2]
    return e


def qubo_to_ising(q: Qubo) raises -> PauliSum:
    """Pauli-Z form with x_i = (1 - z_i) / 2.

    Every variable gets a Z term, even at coefficient 0, so the sum keeps
    num_vars qubits and ising_to_qubo recovers the same width.
    """
    var n = q.num_vars
    var constant = q.offset
    var z = List[Float64](length=n, fill=0.0)
    for i in range(n):
        constant += 0.5 * q.linear[i]
        z[i] -= 0.5 * q.linear[i]
    for t in q.quadratic:
        var c = 0.25 * t[2]
        constant += c
        z[t[0]] -= c
        z[t[1]] -= c
    var h = PauliSum()
    h.add(constant, 0, 0)
    for i in range(n):
        h.add(z[i], 0, UInt64(1) << UInt64(i))
    for t in q.quadratic:
        h.add(0.25 * t[2], 0, (UInt64(1) << UInt64(t[0])) | (UInt64(1) << UInt64(t[1])))
    return h^


def ising_to_qubo(h: PauliSum) raises -> Qubo:
    """Inverse of qubo_to_ising for diagonal sums, with z_i = 1 - 2 x_i.

    The width is h.num_qubits(), so a sum from qubo_to_ising gives back num_vars.
    """
    var q = Qubo(h.num_qubits())
    for t in h.terms:
        if t.x != 0:
            raise Error(
                "QM-E812: Pauli term has an X or Y factor, so the sum is not diagonal and not a QUBO."
                " Hint: diagonalise the problem first, or pass only Z and ZZ terms to ising_to_qubo."
            )
        var w = Int(pop_count(t.z))
        if w == 0:
            q.offset += t.coeff
        elif w == 1:
            var i = _lowest_bit(t.z)
            q.offset += t.coeff
            q.linear[i] -= 2.0 * t.coeff
        elif w == 2:
            var i = _lowest_bit(t.z)
            var j = _lowest_bit(t.z & (t.z - 1))
            q.offset += t.coeff
            q.linear[i] -= 2.0 * t.coeff
            q.linear[j] -= 2.0 * t.coeff
            q.add_quadratic(i, j, 4.0 * t.coeff)
        else:
            raise Error(
                "QM-E812: Z product of weight " + String(w) + " is not quadratic."
                " Hint: ising_to_qubo accepts constants, Z and Z Z terms only."
            )
    return q^


def qubo_energy_of_ising(h: PauliSum, mask: UInt64) raises -> Float64:
    """Eigenvalue of a diagonal Pauli sum on the basis state mask."""
    var e = 0.0
    for t in h.terms:
        if t.x != 0:
            raise Error(
                "QM-E812: Pauli term has an X or Y factor, so the sum is not diagonal."
                " Hint: evaluate only Z and ZZ sums with qubo_energy_of_ising."
            )
        if (Int(pop_count(mask & t.z)) & 1) == 1:
            e -= t.coeff
        else:
            e += t.coeff
    return e


def _dense_couplings(q: Qubo) -> List[Float64]:
    """Row-major symmetric Q with zero diagonal."""
    var n = q.num_vars
    var d = List[Float64](length=n * n, fill=0.0)
    for t in q.quadratic:
        d[t[0] * n + t[1]] = t[2]
        d[t[1] * n + t[0]] = t[2]
    return d^


def _fields(q: Qubo, dense: List[Float64], mask: UInt64) -> List[Float64]:
    """h_i = linear_i + sum_j Q_ij x_j. Flipping x_i changes the energy by (1 - 2 x_i) h_i."""
    var n = q.num_vars
    var h = List[Float64](length=n, fill=0.0)
    for i in range(n):
        var s = q.linear[i]
        for j in range(n):
            if _x(mask, j):
                s += dense[i * n + j]
        h[i] = s
    return h^


def _scale(q: Qubo) -> Float64:
    var s = 1.0 + abs(q.offset)
    for v in q.linear:
        s += abs(v)
    for t in q.quadratic:
        s += abs(t[2])
    return s


def qubo_brute_force(q: Qubo) raises -> QuboSolution:
    """Exact minimum by enumeration; ties go to the smallest mask.

    Masks are visited in Gray-code order with incremental energies. The
    incremental value is re-checked exactly whenever it is within a small
    tolerance of the best, so the reported energy and the tie rule are exact.
    """
    var n = q.num_vars
    if n > QUBO_BRUTE_MAX:
        raise Error(
            "QM-E811: qubo_brute_force on " + String(n) + " variables is too large."
            " Hint: brute force takes at most 24 variables; use qubo_simulated_annealing."
        )
    var dense = _dense_couplings(q)
    var tol = 1e-9 * _scale(q)
    var mask: UInt64 = 0
    var h = _fields(q, dense, mask)
    var approx = q.offset
    var best = qubo_energy(q, mask)
    var best_mask: UInt64 = 0
    var total = 1 << n
    for k in range(1, total):
        var i = _trailing_zeros(k)
        var s = 1.0 - 2.0 * Float64(Int((mask >> UInt64(i)) & 1))
        approx += s * h[i]
        mask ^= UInt64(1) << UInt64(i)
        for j in range(n):
            h[j] += s * dense[j * n + i]
        if (k & 1023) == 0:
            # resync to bound the rounding drift of the incremental sums
            approx = qubo_energy(q, mask)
            h = _fields(q, dense, mask)
        if approx <= best + tol:
            var e = qubo_energy(q, mask)
            if e < best or (e == best and mask < best_mask):
                best = e
                best_mask = mask
    return QuboSolution(best_mask, best, n)


def qubo_simulated_annealing(
    q: Qubo,
    num_reads: Int,
    num_sweeps: Int,
    beta_min: Float64,
    beta_max: Float64,
    seed: UInt64,
) raises -> QuboSolution:
    """Best assignment over num_reads annealing runs.

    Each read starts from a random mask drawn from Rng(seed).split(read). A
    sweep tries every variable once with a Metropolis flip. beta rises
    geometrically from beta_min (first sweep) to beta_max (last sweep); one
    sweep uses beta_min. The result is deterministic for a fixed seed.
    """
    if num_reads < 1 or num_sweeps < 1 or not (beta_min > 0.0 and beta_min <= beta_max):
        raise Error(
            "QM-E822: qubo_simulated_annealing needs num_reads >= 1, num_sweeps >= 1 and"
            " 0 < beta_min <= beta_max, got " + String(num_reads) + " reads, "
            + String(num_sweeps) + " sweeps, beta " + String(beta_min) + " to " + String(beta_max)
            + ". Hint: use num_reads >= 1, num_sweeps >= 1 and 0 < beta_min <= beta_max."
        )
    var n = q.num_vars
    var dense = _dense_couplings(q)
    var base = Rng(seed)
    var log_ratio = log(beta_max / beta_min)
    var best_mask: UInt64 = 0
    var best = 0.0
    for r in range(num_reads):
        var rng = base.split(r)
        var mask = rng.next_u64()
        if n < 64:
            mask &= (UInt64(1) << UInt64(n)) - 1
        var h = _fields(q, dense, mask)
        var energy = qubo_energy(q, mask)
        var read_mask = mask
        var read_best = energy
        for sweep in range(num_sweeps):
            var frac = 0.0
            if num_sweeps > 1:
                frac = Float64(sweep) / Float64(num_sweeps - 1)
            var beta = beta_min * exp(frac * log_ratio)
            for i in range(n):
                var s = 1.0 - 2.0 * Float64(Int((mask >> UInt64(i)) & 1))
                var de = s * h[i]
                if de <= 0.0 or rng.uniform() < exp(-beta * de):
                    mask ^= UInt64(1) << UInt64(i)
                    energy += de
                    for j in range(n):
                        h[j] += s * dense[j * n + i]
                    if energy < read_best:
                        read_best = energy
                        read_mask = mask
            # resync once per sweep so incremental drift cannot accumulate
            energy = qubo_energy(q, mask)
            h = _fields(q, dense, mask)
        var exact = qubo_energy(q, read_mask)
        if r == 0 or exact < best or (exact == best and read_mask < best_mask):
            best = exact
            best_mask = read_mask
    return QuboSolution(best_mask, best, n)
