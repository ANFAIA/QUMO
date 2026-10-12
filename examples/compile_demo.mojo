"""Compile demo: QFT + Clifford+T circuits for several hardware modalities.

    .venv/bin/mojo run -I . examples/compile_demo.mojo
"""

from std.math import pi
from qumo.ir.circuit import Circuit
from qumo.compiler.target import Target, parse_target
from qumo.compiler.pass_manager import transpile, compile_spec, preset_pipeline


def qft(n: Int) -> Circuit:
    var c = Circuit(n, name="qft" + String(n))
    for j in range(n):
        c.h(j)
        for k in range(j + 1, n):
            c.cp(pi / Float64(1 << (k - j)), k, j)
    for j in range(n // 2):
        c.swap(j, n - 1 - j)
    return c^


def toffoli_ladder(n: Int) -> Circuit:
    """Clifford+T workload: a compute / uncompute Toffoli ladder (lots of T)."""
    var c = Circuit(n, name="toffoli_ladder" + String(n))
    for q in range(n):
        c.h(q)
    for q in range(n - 2):
        c.ccx(q, q + 1, q + 2)
    for q in range(n - 3, -1, -1):
        c.ccx(q, q + 1, q + 2)
    c.t(0)
    c.cx(0, 1)
    c.tdg(1)
    c.cx(0, 1)
    return c^


def main() raises:
    var target = Target.superconducting_grid(3, 4)  # rz, sx, x, cz on a 3x4 lattice
    print("target:", target.describe())
    print()

    var c = qft(8)
    print("== QFT-8 at O3 on", target.name, "==")
    print("pipeline:", preset_pipeline(3, target))
    var res = transpile(c, target, 3)
    print(res.table())
    print(res.summary())
    print("initial layout:", res.to_json().split("\"initial_layout\":")[1].split("]")[0] + "]")
    print()

    var t = toffoli_ladder(6)
    print("== Toffoli ladder (Clifford+T) on", target.name, "==")
    var spec = "unroll3q,fixpoint(cancel,commute,phase_fold),route(sabre,trials=8),basis,fixpoint(cancel,commute,eject_z,fuse1q),check"
    var r2 = compile_spec(t, target, spec)
    print(r2.table())
    print(r2.summary())
    print()

    print("== Same QFT-8, every modality at O2 ==")
    var specs: List[String] = [
        "all_to_all:8", "line:8/generic", "heavy_hex:3/sc_ecr", "superconducting_grid_iswap:2x4",
        "trapped_ion:8", "trapped_ion:8/ion_zz", "neutral_atom:3x3",
    ]
    for s in specs:
        var tg = parse_target(s)
        var r = transpile(c, tg, 2)
        print("  ", s, "->", r.summary())
