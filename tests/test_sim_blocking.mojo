from qumo.ir.circuit import Circuit
from qumo.ir.gates import Op
from qumo.sim.statevector import StateVector
from qumo.sim.runner import evolve, apply_instruction
from qumo.sim.rng import Rng
from std.testing import assert_almost_equal


def main() raises:
    # random circuit on 21 qubits mixing every 1q/2q kernel kind; blocked
    # execution (evolve) must equal plain gate-by-gate application
    var n = 21
    var rng = Rng(5)
    var c = Circuit(n)
    var kinds: List[Int] = [Op.H, Op.RX, Op.RZ, Op.T, Op.U, Op.CX, Op.CZ, Op.CP, Op.CRY, Op.RZZ, Op.SWAP, Op.RXX, Op.ISWAP, Op.CCX]
    for _ in range(400):
        var k = kinds[rng.below(len(kinds))]
        var a = rng.below(n)
        var b = rng.below(n - 1)
        if b >= a:
            b += 1
        var d = rng.below(n)
        while d == a or d == b:
            d = rng.below(n)
        c.add(k, a, b, d, rng.uniform() * 3, rng.uniform(), rng.uniform())
    var fast = evolve(c)
    var slow = StateVector(n)
    for g in c.ops:
        apply_instruction(slow, g, List[Float64]())
    assert_almost_equal(fast.fidelity(slow), 1.0, atol=1e-9)
    print("test_sim_blocking OK")
