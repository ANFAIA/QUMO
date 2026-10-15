"""Multi-logical-qubit Iceberg program: native rotations, block allocation and error detection.

Reproduces: Guppy FT - Iceberg logical API (Block[N], rotations, measure_all, measure_syndrome)
Source: https://github.com/Quantinuum/guppyft/blob/main/src/guppyft/code/iceberg/logical.py (Apache-2.0)
Source: https://arxiv.org/abs/2211.06703 (Self, Benedetti, Amaro - Iceberg code)

Guppy FT's Iceberg architecture keeps N logical qubits in one [[N+2,N,2]]
block (`Block[N]`), applies logical rotations as physical ZZ/XX-phase gates on
pairs (`rz(i, phase)`, `zz_phase(i, j, phase)`, `xx_phase`, cross-block
`zz_phase_between_blocks`), measures a whole block destructively
(`measure_all` -> `LogicalMeasurement[N]`) and detects errors with
`measure_syndrome`. QUMO's `IcebergBuilder(k)` does the same with a
(block, slot) address stack: logical qubits fill slots 0..k-1 of block 0
first, rotations within a block are weight-2 RZZ/RXX gates, rotations
between blocks use a 4-body parity ladder, a final readout of a whole block
is one transversal measurement with one shared detection flag, and syndrome
cycles (X^n, Z^n) are inserted by a cost policy. Detected errors are
discarded (post-selection).

Checks: a 4-qubit rotation circuit (RZZ, RX, CX, RZ, H, RXX) encoded in one
[[6,4,2]] block and in two [[4,2,2]] blocks reproduces the unencoded
distribution (state vector, TVD <= 0.09); the [[6,4,2]] encoding uses 6 + 1
lane + ancilla qubits; with circuit-level noise p = 2e-3, discarding shots
with detected errors at least halves the encoded error and is not worse than
the unencoded program. (Iceberg rotations on (b, d_i) pairs are not fault
tolerant: a single two-qubit fault can be a logical error, so the gain over
the unencoded program is small; Guppy FT's Iceberg ops have the same property.)

Run: .venv/bin/mojo run -I . examples/ft/iceberg_program.mojo
example-meta: {"framework": "guppyft", "tier": "fast", "timeout": 900, "requires": [], "args": []}
"""

from qumo.ir.circuit import Circuit
from qumo.qec.codes import NoiseParams
from qumo.ft.arch import IcebergBuilder, QECPolicy
from qumo.ft.encode import compile, encode
from qumo.ft.emulate import run_encoded, run_physical, add_noise, reference_counts, total_variation, LogicalCounts
from qumo.testing.checks import CheckLog


def rotations() -> Circuit:
    var c = Circuit(4, 0, "iceberg_rotations")
    c.h(0)
    c.rzz(0.9, 0, 1)
    c.rx(0.4, 2)
    c.cx(2, 3)
    c.rz(1.1, 3)
    c.h(1)
    c.rxx(0.7, 1, 3)
    c.measure_all()
    return c^


def ghz_parity() -> Circuit:
    """Deterministic: GHZ on 4 qubits, back to |0000>, X on all -> 1111."""
    var c = Circuit(4, 0, "ghz_roundtrip")
    c.h(0)
    for q in range(3):
        c.cx(q, q + 1)
    var q = 2
    while q >= 0:
        c.cx(q, q + 1)
        q -= 1
    c.h(0)
    for i in range(4):
        c.x(i)
    c.measure_all()
    return c^


def err(r: LogicalCounts, key: String) -> Float64:
    return 1.0 - r.probability(key)


def main() raises:
    var log = CheckLog("iceberg_program")
    var c = rotations()
    var ref_ = reference_counts(c, 1000, 3)
    print("unencoded:", ref_)
    var one = IcebergBuilder(4).build(1)
    var lp = compile(one, c)
    print(lp)
    var p1 = encode(one, c)
    print("[[6,4,2]] x1:", p1)
    _ = log.check("one [[6,4,2]] block + lane + 1 ancilla", p1.circuit.num_qubits <= 8, String(p1.circuit.num_qubits) + " qubits")
    var r1 = run_encoded(p1, 1000, 4)
    print("   encoded:", r1)
    _ = log.within("[[6,4,2]] TVD vs unencoded", total_variation(r1, ref_), 0.0, 0.09)
    var two = IcebergBuilder(2).build(2)
    var p2 = encode(two, c)
    print("[[4,2,2]] x2:", p2)
    var r2 = run_encoded(p2, 1000, 5)
    print("   encoded:", r2)
    _ = log.within("2 x [[4,2,2]] TVD vs unencoded (cross-block rotations)", total_variation(r2, ref_), 0.0, 0.09)

    # error detection under circuit-level noise, with syndrome cycles after every CX
    var g = ghz_parity()
    var noise = NoiseParams(2e-3, 2e-3, 2e-3, 0.0)
    var phys = reference_counts(g, 4000, 6, noise)
    var pol = QECPolicy("syndrome", 2.0).with_cost("rzz", 1.0)
    var prog = encode(IcebergBuilder(4).with_qec_policy(pol).build(1), g)
    var enc = run_encoded(prog, 4000, 7, noise)
    # same noisy shots without discarding detected errors
    var raw = run_physical(add_noise(prog.circuit, noise, prog.lane_qubits), prog.outputs, List[Int](), 4000, 7)
    print("noisy unencoded:", err(phys, "1111"), " encoded without post-selection:", err(raw, "1111"),
          " encoded with post-selection:", err(enc, "1111"), " acceptance:", enc.acceptance())
    _ = log.check("post-selection on detections at least halves the encoded error", err(enc, "1111") < 0.5 * err(raw, "1111"),
                  "without " + String(err(raw, "1111")) + " with " + String(err(enc, "1111")))
    _ = log.check("detected encoding not worse than unencoded", err(enc, "1111") <= err(phys, "1111"),
                  "unencoded " + String(err(phys, "1111")) + " encoded " + String(err(enc, "1111")))
    _ = log.within("acceptance", enc.acceptance(), 0.7, 1.0)
    log.finish()
