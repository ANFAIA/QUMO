"""Magic-state T gates on Steane, and Rz synthesis for known and run-time angles.

Reproduces: Guppy FT - Custom T injection (logical program) and Comparator-based Rz synthesis with the Steane architecture
Source: https://github.com/Quantinuum/guppyft/blob/main/examples/logical_program.ipynb (Apache-2.0)
Source: https://github.com/Quantinuum/guppyft/blob/main/examples/comparator_rz_steane.ipynb (Apache-2.0)
Source: https://arxiv.org/abs/2404.05618 (single-qubit rotation algorithm with logarithmic Toffoli count)

Part 1 (logical program notebook): a qubit is put in |+>_L, four logical T
gates are applied by magic-state injection (prepare T|+>_L, CX, measure,
corrections X then S), which act as a Z; H and a Z measurement must give 1.
QUMO writes it with the logical API (`prep_t`, `inject_t`), encodes it with
`implement_ops`, and emulates the encoded program on the state vector
(T is not Clifford): one Steane data block + one magic block + a classical
lane = 15 qubits. The S corrections stay in the Pauli frame (no flushes).

Part 2 (comparator notebook): Rz with an angle known only at run time. The
notebook draws a random angle, applies Rz(theta) Rz(-theta) and expects 0;
QUMO builds the repeat-until-success comparator circuit whose angle is read
from classical bits, checks it on the state vector (unencoded), compiles it
for Steane and reports the encoded resources (the notebook likewise only
reports two-qubit depth: "prohibitive on current hardware"). Known angles
use the static fewest-T Clifford+T search; its T-count / error table is
printed.

Checks: T^4 = Z on |+>_L gives 1 in every shot; encoded T|+> then H matches
cos^2(pi/8) = 0.854 within 0.12 (40 shots); comparator Rz (eps = 0.1)
reproduces the angle encoded in k to 1e-9 and theta within eps, for a fixed
angle and for a run-time angle register; static synthesis reaches <= 0.02 at
<= 16 T for theta = 1.0; the encoded comparator needs > 40 physical qubits.

Run: .venv/bin/mojo run -I . examples/ft/magic_t_and_comparator_rz.mojo
example-meta: {"framework": "guppyft", "tier": "slow", "timeout": 1200, "requires": [], "args": []}
"""

from std.math import pi, cos

from qumo.ir.circuit import Circuit
from qumo.exec.resources import estimate_resources
from qumo.ft.logical import LogicalProgram
from qumo.ft.arch import SteaneBuilder
from qumo.ft.encode import compile, encode
from qumo.ft.implement import implement_ops
from qumo.ft.emulate import run_encoded, comparator_rz_check
from qumo.ft.rz import synthesize_rz, rz_precision_table
from qumo.testing.checks import CheckLog


def main() raises:
    var log = CheckLog("magic_t_and_comparator_rz")
    # ---- Part 1: logical program with explicit injections
    var lp = LogicalProgram("custom_t_example")
    var q = lp.alloc()
    lp.h(q)
    for _ in range(4):
        var m = lp.prep_t()
        lp.inject_t(q, m)
    lp.h(q)
    lp.measure(q, 0)
    print(lp)
    var steane = SteaneBuilder().build(2)
    var prog = implement_ops(steane, lp)
    print(prog)
    _ = log.check("T corrections kept in the Pauli frame", prog.stats.frame_flushes == 0, String(prog.stats.frame_corrections) + " frame corrections")
    var r = run_encoded(prog, 16, 1)
    print("T^4 on |+>_L then H:", r)
    _ = log.check("T^4 = Z (every shot reads 1)", r.counts.get("1", 0) == 16)
    var c = Circuit(1, 1, "t_plus")
    c.h(0)
    c.t(0)
    c.h(0)
    c.measure(0, 0)
    var r2 = run_encoded(encode(steane, c), 40, 2)
    print("H T H |0>:", r2)
    _ = log.close("P(0) after H T H", r2.probability("0"), cos(pi / 8) ** 2, 0.12)

    # ---- Part 2: Rz synthesis
    print("static Clifford+T search for Rz(0.3): T budget -> error")
    for s in rz_precision_table(0.3, 12):
        print("   ", s.t_count, s.error)
    var syn = synthesize_rz(1.0, 2e-2, 16)
    print(syn)
    _ = log.within("static Rz(1.0) error", syn.error, 0.0, 0.02)
    _ = log.within("static Rz(1.0) T-count", Float64(syn.t_count), 0.0, 16.0)
    var fixed = comparator_rz_check(0.7, 0.1, 3, 16, 3)
    print(fixed)
    _ = log.within("comparator Rz(0.7) vs encoded angle", fixed.max_eff_error, 0.0, 1e-9)
    _ = log.within("comparator Rz(0.7) vs theta", fixed.max_angle_error, 0.0, 0.1)
    var runtime = comparator_rz_check(2.2, 0.1, 3, 16, 4, runtime_angle=True)
    print(runtime)
    _ = log.within("run-time angle register vs encoded angle", runtime.max_eff_error, 0.0, 1e-9)
    _ = log.within("run-time angle register vs theta", runtime.max_angle_error, 0.0, 0.1)
    # encoded comparator: resources only
    var rc = Circuit(1, 1, "comparator_rz_steane")
    rc.h(0)
    rc.rz(0.7, 0)
    rc.rz(-0.7, 0)
    rc.h(0)
    rc.measure(0, 0)
    var arch = SteaneBuilder().with_rz("comparator", 0.1, 16, 2).build(16)
    var lpc = compile(arch, rc)
    var enc = encode(arch, rc)
    var res = estimate_resources(enc.circuit)
    print("encoded comparator Rz: logical T =", lpc.t_count(), " physical qubits", res.num_qubits, " 2q gates", res.gates_2q, " 2q depth", res.depth_2q)
    _ = log.check("encoded comparator is large", res.num_qubits > 40, String(res.num_qubits) + " qubits")
    log.finish()
