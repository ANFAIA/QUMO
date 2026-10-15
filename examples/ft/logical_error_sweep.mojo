"""Logical versus physical error rate of an encoded Steane program under circuit-level noise.

Reproduces: Guppy FT - Steane encoding emulated with Stim (logical vs unencoded behaviour)
Source: https://github.com/Quantinuum/guppyft/blob/main/examples/steane_encoding.ipynb (Apache-2.0)
Source: https://github.com/Quantinuum/guppyft/blob/main/docs/getting_started.md (Apache-2.0)

Guppy FT emulates encoded programs with Stim through Selene. QUMO emulates
them on its own tableau simulator: the encoded circuit gets Stim-style
circuit-level noise (DEPOLARIZE1/2 after gates, X errors before measurements
and after resets; the classical lane is noise-free) and every shot is decoded
by the circuit itself (lookup decoding on the classical lane, flagged |0>_L
preparations post-selected).

The program is deterministic (X, a CX chain and an H-S-S-H round trip on
three qubits, measured: 111), so the error rate is P(outcome != 111). We
sweep p and compare the unencoded program with the same noise against the
Steane encoding: the encoded error should be much smaller at p = 1e-3 and
scale roughly like p^2 (distance 3), the unencoded one like p.

Checks: encoded error < unencoded error at p = 1e-3 and 2e-3; noiseless
encoded run is exactly 111; the encoded error grows faster than linearly
between p = 1e-3 and 4e-3 (ratio > 4 for a 4x larger p, within sampling).

Run: .venv/bin/mojo run -I . examples/ft/logical_error_sweep.mojo
example-meta: {"framework": "guppyft", "tier": "fast", "timeout": 900, "requires": [], "args": []}
"""

from qumo.ir.circuit import Circuit
from qumo.ft.arch import SteaneBuilder
from qumo.ft.encode import encode
from qumo.ft.emulate import run_encoded, error_rate_sweep
from qumo.testing.checks import CheckLog


def program() -> Circuit:
    var c = Circuit(3, 0, "deterministic")
    c.x(0)
    c.cx(0, 1)
    c.cx(1, 2)
    c.h(1)
    c.s(1)
    c.s(1)
    c.h(1)  # H Z H = X: qubit 1 flips back to 0 ... then CX makes it 1 again
    c.cx(2, 1)
    c.measure_all()
    return c^


def main() raises:
    var log = CheckLog("logical_error_sweep")
    var c = program()
    var steane = SteaneBuilder().build(3)
    var clean = run_encoded(encode(steane, c), 200, 1)
    print("noiseless encoded:", clean)
    var key = clean.most_frequent()
    _ = log.check("noiseless encoded run is deterministic", clean.counts.get(key, 0) == 200, key)
    var ps: List[Float64] = [1e-3, 2e-3, 4e-3]
    var pts = error_rate_sweep(steane, c, ps, 20000, 3)
    print("p, unencoded error, encoded error, acceptance")
    for pt in pts:
        print(pt.p, pt.physical_error, pt.logical_error, pt.acceptance)
    _ = log.check("encoded beats unencoded at p=1e-3", pts[0].logical_error < pts[0].physical_error,
                  String(pts[0].logical_error) + " < " + String(pts[0].physical_error))
    _ = log.check("encoded beats unencoded at p=2e-3", pts[1].logical_error < pts[1].physical_error,
                  String(pts[1].logical_error) + " < " + String(pts[1].physical_error))
    var ratio = pts[2].logical_error / max(pts[0].logical_error, 1e-6)
    _ = log.within("encoded error growth for 4x p (superlinear)", ratio, 4.0, 1e6)
    log.finish()
