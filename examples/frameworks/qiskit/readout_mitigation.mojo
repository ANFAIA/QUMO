"""T-REX and M3: readout mitigation past the 24-qubit dense limit.

Reproduces: Qiskit - qiskit_mitigation.TREX and mthree (arXiv:2012.09738, PRX Quantum 2, 040326)
Source: https://github.com/Qiskit/qiskit-addon-mthree (Apache-2.0)

T-REX twirls each measurement with a random X and a classical flip, so asymmetric
readout noise only rescales each Pauli, and calibrating on |0...0> measures that
factor. M3 solves the assignment matrix in the subspace of observed bitstrings, so
memory follows the shots. bench/crosscheck/tier1.py m3 matches mthree 3.0 to 5e-9.

Checks:
- with flips p01 = 0.02, p10 = 0.12, raw <Z0 Z1 + Z2> is off; T-REX recovers it;
- M3 restores most of the GHZ weight lost to readout on 40 qubits.

Run: .venv/bin/mojo run -I . examples/frameworks/qiskit/readout_mitigation.mojo
example-meta: {"framework": "qiskit", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from std.collections import Dict
from qumo.ir.circuit import Circuit
from qumo.ir.pauli import parse_pauli_sum
from qumo.sim.runner import evolve, sample, SampleResult
from qumo.sim.rng import Rng
from qumo.algorithms.mitigation import trex_expectation
from qumo.qec.characterization import ReadoutCalibration
from qumo.qec.mthree import mitigate_counts_m3
from qumo.testing.checks import CheckLog


def device(c: Circuit, shots: Int, seed: UInt64) raises -> SampleResult:
    """A perfect device with asymmetric readout: only a classical channel can do this."""
    var r = sample(c, shots, seed)
    var rng = Rng(seed ^ 0xABCDEF)
    var out = SampleResult(c.num_clbits, shots, "readout")
    for creg in r.memory:
        var o = 0
        for b in range(c.num_clbits):
            var bit = (creg >> b) & 1
            var flip = rng.uniform() < (0.02 if bit == 0 else 0.12)
            if (bit == 1) != flip:
                o |= 1 << b
        out.record(o)
    return out^


def main() raises:
    var log = CheckLog("readout_mitigation")
    var c = Circuit(3)
    c.ry(0.7, 0)
    c.cx(0, 1)
    c.ry(1.1, 2)
    var h = parse_pauli_sum("0.5*Z0 Z1 + Z2")
    var r = trex_expectation(device, c, h, 40000, 3)
    var exact = evolve(c).expectation(h)
    print("exact", exact, "raw", r.raw, "T-REX", r.value, "+-", r.std)
    _ = log.check("raw value is off", abs(r.raw - exact) > 5 * r.std)
    _ = log.close("T-REX recovers it", r.value, exact, 5 * r.std + 0.01)

    var n = 40
    var cal = ReadoutCalibration(n)
    for q in range(n):
        cal.p01[q] = 0.005
        cal.p10[q] = 0.01
    var rng = Rng(8)
    var counts = Dict[String, Int]()
    var shots = 20000
    for _ in range(shots):
        var truth = 0 if rng.uniform() < 0.5 else 1
        var key = String("")
        for _ in range(n):
            var flip = rng.uniform() < (0.005 if truth == 0 else 0.01)
            key += "1" if (truth == 1) != flip else "0"
        counts[key] = counts.get(key, 0) + 1
    var zeros = String("0") * n
    var ones = String("1") * n
    var raw = Float64(counts.get(zeros, 0) + counts.get(ones, 0)) / Float64(shots)
    var m3 = mitigate_counts_m3(counts, cal)
    var fixed = m3.get(zeros, 0.0) + m3.get(ones, 0.0)
    print("40 qubits,", len(counts), "strings: raw", raw, "M3", fixed)
    _ = log.within("M3 restores the GHZ weight", fixed, 0.9, 1.05)
    log.finish()
