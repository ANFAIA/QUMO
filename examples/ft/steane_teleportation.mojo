"""Encoded teleportation with feed-forward on the Steane code, three QEC architectures.

Reproduces: Guppy FT - Getting started (encoding with Steane) and Writing logical programs directly
Source: https://github.com/Quantinuum/guppyft/blob/main/docs/getting_started.md (Apache-2.0)
Source: https://github.com/Quantinuum/guppyft/blob/main/examples/logical_program.ipynb (Apache-2.0)

Guppy FT's first example writes teleportation as a QEC-agnostic program
(source in |1>, Bell pair, Bell measurement, `if measure(src): z(tgt)`,
`if measure(tmp): x(tgt)`), builds `SteaneBuilder().build(n_blocks=3)` and
encodes it. The logical-program notebook then compares a default Steane
instance with one using Knill-style QEC cycles (threshold 2, cost 1 for H and
CX) and a factory of 2 states, and observes the larger two-qubit depth.

QUMO does the same with `qumo.ft`: the circuit uses `c_if` feed-forward, the
encoder keeps the X / Z corrections in a Pauli frame (no physical correction
gates; decoded measurement values are read only at the end), and the encoded
programs are emulated per shot on QUMO's stabilizer tableau. Variants: frame
vs eager feed-forward, Knill cycles + factory, Steane-style teleportation EC.

Checks: the logical target is 1 in every shot of every variant; encoded
distributions match the unencoded one (total variation <= 0.08 over 1000
shots); frame mode applies 0 physical corrections; the Knill variant inserts
2 QEC cycles and has a larger 2-qubit depth; with circuit-level noise
p = 1e-3 the encoded target stays correct in > 99% of accepted shots.

Run: .venv/bin/mojo run -I . examples/ft/steane_teleportation.mojo
example-meta: {"framework": "guppyft", "tier": "fast", "timeout": 600, "requires": [], "args": []}
"""

from qumo.ir.circuit import Circuit
from qumo.qec.codes import NoiseParams
from qumo.exec.resources import estimate_resources
from qumo.ft.arch import SteaneBuilder, QECPolicy, FactoryConf, Architecture
from qumo.ft.encode import compile, encode
from qumo.ft.implement import implement_ops
from qumo.ft.emulate import run_encoded, reference_counts, total_variation, LogicalCounts
from qumo.testing.checks import CheckLog


def teleportation() -> Circuit:
    var c = Circuit(3, 3, "teleportation")
    c.x(0)  # source |1>
    c.h(1)
    c.cx(1, 2)  # Bell pair (tmp, tgt)
    c.cx(0, 1)
    c.h(0)
    c.measure(0, 0)
    c.measure(1, 1)
    c.z(2)
    c.c_if(0, 1)
    c.x(2)
    c.c_if(1, 1)
    c.measure(2, 2)
    return c^


def target_ok(r: LogicalCounts) -> Float64:
    var ok = 0
    for k in r.keys():
        if k.startswith("1"):  # leftmost char = clbit 2 = tgt
            ok += r.counts.get(k, 0)
    return Float64(ok) / Float64(max(r.accepted, 1))


def main() raises:
    var log = CheckLog("steane_teleportation")
    var c = teleportation()
    var ref_ = reference_counts(c, 1000, 1)
    print("unencoded:", ref_)

    var default = SteaneBuilder().build(3)
    var lp = compile(default, c)
    print(lp)
    var names: List[String] = ["default (frame)", "eager feed-forward", "knill qec + factory", "steane-style qec"]
    var depth_default = 0
    for i in range(len(names)):
        var arch: Architecture
        if i == 0:
            arch = default.copy()
        elif i == 1:
            arch = SteaneBuilder().with_feed_forward("eager").build(3)
        elif i == 2:
            var pol = QECPolicy("knill", 2.0).with_cost("h", 1.0).with_cost("cx", 1.0)
            arch = SteaneBuilder().with_qec_policy(pol).with_zero_factory_conf(FactoryConf(2, 5)).build(3)
        else:
            arch = SteaneBuilder().with_qec_policy(QECPolicy("steane", 1.0).with_cost("cx", 1.0)).build(3)
        var prog = implement_ops(arch, lp)
        var res = estimate_resources(prog.circuit)
        print(names[i], ":", prog)
        print("   physical qubits", res.num_qubits, " depth", res.depth, " 2q depth", res.depth_2q, " 2q gates", res.gates_2q)
        var r = run_encoded(prog, 1000, 7 + UInt64(i))
        print("   encoded:", r)
        _ = log.check(names[i] + " target always 1", target_ok(r) == 1.0, String(target_ok(r)))
        _ = log.within(names[i] + " TVD vs unencoded", total_variation(r, ref_), 0.0, 0.08)
        if i == 0:
            depth_default = res.depth_2q
            _ = log.check("frame mode applies no physical corrections", prog.stats.frame_flushes == 0 and prog.stats.frame_corrections == 2,
                          "corrections in frame " + String(prog.stats.frame_corrections))
        if i == 2:
            _ = log.check("knill policy inserted 2 QEC cycles", prog.stats.qec_cycles == 2, String(prog.stats.qec_cycles))
            _ = log.check("QEC cycles increase the 2q depth", res.depth_2q > depth_default,
                          String(depth_default) + " -> " + String(res.depth_2q))

    # circuit-level noise (Stim-style p on gates, measurements, resets)
    var noisy = run_encoded(encode(default, c), 2000, 11, NoiseParams(1e-3, 1e-3, 1e-3, 0.0))
    print("noisy p=1e-3:", noisy)
    _ = log.within("noisy encoded target correct", target_ok(noisy), 0.99, 1.0)
    _ = log.within("noisy acceptance (flagged preparations discarded)", noisy.acceptance(), 0.9, 1.0)
    log.finish()
