"""Emulating a device from its target, and splitting a parameter sweep.

Reproduces: Braket SDK - device emulators (1.112), ProgramSet.split (1.120)
Source: https://github.com/amazon-braket/amazon-braket-sdk-python/tree/main/src/braket/emulation (Apache-2.0)
Framework: Amazon Braket - https://github.com/amazon-braket/amazon-braket-sdk-python

`NoiseModel.from_target` turns a target's calibration into channels and `emulate`
samples a compiled circuit under them, nothing leaves the machine. `split_pub`
breaks a sweep for a device limit; seeds follow the original index, so the split
run is bit-identical to the unsplit one.

Checks:
- each modality's emulated GHZ weight tracks estimate_fidelity;
- an uncompiled circuit is refused with QM-E615;
- split results equal the unsplit ones for every parameter set.

Run: .venv/bin/mojo run -I . examples/simulation/emulate_and_split.mojo
example-meta: {"framework": "braket", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from qumo.ir.circuit import Circuit
from qumo.compiler.target import parse_target
from qumo.compiler.pass_manager import transpile
from qumo.exec.emulate import emulate
from qumo.exec.primitives import SamplerPub, run_sampler, split_pub, PubSlot
from qumo.testing.checks import CheckLog


def main() raises:
    var log = CheckLog("emulate_and_split")
    var specs: List[String] = ["superconducting_grid:3x3", "trapped_ion:5", "neutral_atom:3x3"]
    for spec in specs:
        var t = parse_target(spec)
        var c = Circuit(5)
        c.h(0)
        for q in range(4):
            c.cx(q, q + 1)
        c.measure_all()
        var r = transpile(c, t, 2, 7)
        var e = emulate(r.circuit, t, 20000, 3)
        var keep = e.probability("00000") + e.probability("11111")
        print(spec, keep, t.estimate_fidelity(r.circuit), e.method)
        _ = log.close("GHZ weight on " + spec, keep, t.estimate_fidelity(r.circuit), 0.08)
    var refused = String("")
    try:
        var raw = Circuit(4)
        raw.h(0)
        raw.cx(0, 3)
        raw.measure_all()
        _ = emulate(raw, parse_target("line:4/sc_cz"), 10)
    except e:
        refused = String(e)
    _ = log.check("uncompiled circuits are refused", refused.startswith("QM-E615"), refused)

    var c = Circuit(2)
    var th = c.parameter("t")
    c.ry(th, 0)
    c.cx(0, 1)
    c.measure_all()
    var sets = List[List[Float64]]()
    for s in range(7):
        sets.append([0.4 * Float64(s)])
    var pub = SamplerPub(c, 500, sets, 42)
    var whole = run_sampler(pub)
    var slots = List[PubSlot]()
    var parts = split_pub(pub, 3, slots)
    var same = True
    for s in range(7):
        var r = run_sampler(parts[slots[s].pub])[slots[s].index].copy()
        for kv in whole[s].counts.items():
            same = same and r.counts.get(kv.key, -1) == kv.value
    _ = log.check("split equals unsplit", same)
    log.finish()
