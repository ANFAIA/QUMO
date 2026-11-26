"""Pauli-noise semantics: one definition of what each Pauli noise opcode draws.

Every per-shot sampler (tableau, state-vector trajectories, the FT emulator)
asks this module which Pauli to apply and then applies it its own way, so the
meaning of X_ERROR / Y_ERROR / Z_ERROR / DEPOLARIZE1 / DEPOLARIZE2 /
PAULI_CHANNEL_1 / PAULI_CHANNEL_2 / E / ELSE_CORRELATED_ERROR, the order of
the random draws and the Stim letter order live in one place.

Letters: 0 = I, 1 = X, 2 = Y, 3 = Z. A draw is returned as one code:
letter on q0 = code & 3, letter on q1 = code >> 2 (0 = no error).

The random stream is part of the contract (seeded results must not move):
  * X/Y/Z_ERROR(p): one uniform u, fires when u < p.
  * DEPOLARIZE1(p): one uniform, then below(3) when it fires.
  * PAULI_CHANNEL_1(px, py, pz): one uniform against the cumulative sums.
  * DEPOLARIZE2(p): one uniform, then r = 1 + below(15) when it fires:
    letter r & 3 on q0, r >> 2 on q1 (uniform over the 15 non-identity pairs).
  * PAULI_CHANNEL_2: one uniform against the 15 table entries in Stim's order
    (IX, IY, IZ, XI, ..., ZZ): entry i is (first, second) = ((i+1) >> 2, (i+1) & 3),
    first letter on q0.
  * E / ELSE_CORRELATED_ERROR: `correlated_error_fires`; an E (q2 == 0) starts
    a chain, an ELSE fires only when nothing earlier in its chain fired.

Exact (density-matrix) and analytic (DEM, PTSBE) consumers do not draw; the
probabilities they use are checked against these draws in
tests/test_pauli_noise_semantics.mojo.
"""

from qumo.ir.gates import Op
from qumo.ir.circuit import Circuit, Instruction
from qumo.ir.param import Param
from qumo.core.rng import Rng


@always_inline
def _prob(p: Param, params: List[Float64]) -> Float64:
    # value() when no parameters are bound (the tableau's historical reading),
    # resolve() otherwise; equal for every numeric probability.
    if len(params) == 0:
        return p.value()
    return p.resolve(params)


@always_inline
def pauli_code(first: Int, second: Int) -> Int:
    """Code with `first` on q0 and `second` on q1."""
    return first | (second << 2)


def draw_pauli_noise(g: Instruction, mut rng: Rng, params: List[Float64] = List[Float64]()) -> Int:
    """Sampled Pauli code for the 1-/2-qubit Pauli opcodes that carry their
    probabilities inline (X/Y/Z_ERROR, DEPOLARIZE1/2, PAULI_CHANNEL_1).
    Returns 0 for no error and for any other opcode (no draw then)."""
    var k = g.kind
    if k == Op.X_ERROR:
        return 1 if rng.uniform() < _prob(g.p0, params) else 0
    if k == Op.Y_ERROR:
        return 2 if rng.uniform() < _prob(g.p0, params) else 0
    if k == Op.Z_ERROR:
        return 3 if rng.uniform() < _prob(g.p0, params) else 0
    if k == Op.DEPOLARIZE1:
        if rng.uniform() < _prob(g.p0, params):
            return 1 + rng.below(3)
        return 0
    if k == Op.PAULI_CHANNEL_1:
        var u = rng.uniform()
        var px = _prob(g.p0, params)
        var py = _prob(g.p1, params)
        var pz = _prob(g.p2, params)
        if u < px: return 1
        if u < px + py: return 2
        if u < px + py + pz: return 3
        return 0
    if k == Op.DEPOLARIZE2:
        if rng.uniform() < _prob(g.p0, params):
            return 1 + rng.below(15)
        return 0
    return 0


def pauli_channel_2_code(i: Int) -> Int:
    """Code of entry i (0..14) of a PAULI_CHANNEL_2 table, Stim's order."""
    return pauli_code((i + 1) >> 2, (i + 1) & 3)


def draw_pauli_channel_2(c: Circuit, g: Instruction, mut rng: Rng) -> Int:
    """Sampled code of a PAULI_CHANNEL_2 (its 15 probabilities are in c.noise_table)."""
    var off = Int(g.p0.value())
    var u = rng.uniform()
    for i in range(15):
        var p = c.noise_table[off + i]
        if u < p:
            return pauli_channel_2_code(i)
        u -= p
    return 0


def correlated_error_fires(g: Instruction, mut rng: Rng, mut fired: Bool) -> Bool:
    """E / ELSE_CORRELATED_ERROR: whether this one fires. `fired` carries the
    chain state between calls. The product is c.pauli_table[g.q0 + 2*i] (qubit)
    and [g.q0 + 2*i + 1] (letter) for i < g.q1."""
    if g.q2 == 0:
        fired = False
    if not fired and rng.uniform() < g.p0.value():
        fired = True
        return True
    return False
