"""Vendor-neutral resource estimation: logical counts, NISQ per modality, fault tolerant (Q# samples).

Reproduces: Q# (Microsoft QDK) - Resource estimation samples (ShorRE.qs, estimation-ising-2D.ipynb)
Source: https://github.com/microsoft/qdk/blob/main/samples/estimation/ShorRE.qs (MIT)
Source: https://github.com/microsoft/qdk/blob/main/samples/estimation/estimation-ising-2D.ipynb (MIT)

The Q# samples feed the Microsoft Resource Estimator with the frequency-
estimation core of Shor's algorithm (ShorRE.qs: 31-bit modulus, semiclassical
QFT, modular multiplication by constants) and with a 2D transverse-field
Ising evolution (N x N lattice, J = g = 1, dt = 0.9, 4th-order product
formula, error budget 0.01). Neither prints its results in the repository.

QUMO estimates explicit circuits with three read-only estimators from
qumo/exec/resources.mojo:
- `estimate_resources`: logical counts, depth, T-count (T/Tdg, 7 per
  Toffoli-class gate, 1 per non-Clifford rotation), recommended simulator;
- `estimate_hardware`: NISQ duration and estimated success probability per
  modality preset (superconducting, trapped_ion, neutral_atom; none is a default);
- `estimate_fault_tolerant`: surface-code distance, factories, physical
  qubits and runtime for a physical error rate and an error budget.
Workloads (small instances of the same kinds of kernels):
1. a controlled modular multiplication x -> 2x mod 21 (the generic
   permutation oracle of qumo/algorithms/library.mojo, 9 qubits);
2. the full N = 21 order-finding circuit (16 qubits, 8 evaluation bits);
3. an 8-qubit QFT with every CP written as P / CX rotations;
4. one first-order Trotter step of the 2D Ising model on a 10 x 10 lattice
   (RX(2 g dt) on every site, CX . RZ(2 J dt) . CX on all 180 bonds).
For the fault-tolerant estimate each modality supplies its physical
two-qubit error (3e-3, 1e-3, 5e-3) and a surface-code cycle of
4 t_2q + t_meas; error budgets 1e-2, 1e-3 and 1e-4.

Out of scope, stated: count-only (circuit-free) estimates in the style of
RSA-2048 would need a counts-based estimator in qumo/exec/resources.mojo,
a file owned by another workstream; and the Q# qubit models (Majorana,
floquet codes) are not modelled.

Checks:
- the T-count equals an explicit count: Toffoli-class gates expanded into
  Clifford+T (7 T each, equivalence checked on the state vector at 1e-12) plus
  non-Clifford rotations counted one by one;
- the decomposed QFT equals the library qft(8) (1e-12);
- every code distance is >= 3, and physical qubits never decrease as the error budget tightens;
- distances are ordered by the presets' physical error: trapped_ion <= superconducting <= neutral_atom;
- NISQ durations are ordered superconducting < neutral_atom < trapped_ion.

Run: .venv/bin/mojo run -I . examples/frameworks/qsharp/resource_estimation.mojo
example-meta: {"framework": "qsharp", "tier": "fast", "timeout": 90, "requires": [], "args": []}
"""

from std.math import pi
from qumo.ir.circuit import Circuit
from qumo.ir.gates import Op
from qumo.sim.rng import Rng
from qumo.sim.runner import evolve
from qumo.sim.statevector import StateVector
from qumo.exec.resources import (
    estimate_resources, estimate_hardware, estimate_fault_tolerant, HardwareProfile,
)
from qumo.algorithms.library import append_controlled_modmul, order_finding, qft, num_mcz_ancillas
from qumo.ai.checks import CheckLog


def ccz_clifford_t(mut c: Circuit, a: Int, b: Int, t: Int):
    """CCZ with 7 T/Tdg gates (Nielsen & Chuang Fig. 4.9 without the H pair)."""
    c.cx(b, t)
    c.tdg(t)
    c.cx(a, t)
    c.t(t)
    c.cx(b, t)
    c.tdg(t)
    c.cx(a, t)
    c.t(b)
    c.t(t)
    c.cx(a, b)
    c.t(a)
    c.tdg(b)
    c.cx(a, b)


def expand_clifford_t(c: Circuit) raises -> Circuit:
    """Replace CCX / CCZ / CSWAP by Clifford+T; everything else is copied."""
    var out = c.empty_like()
    for g in c.ops:
        if g.kind == Op.CCZ:
            ccz_clifford_t(out, g.q0, g.q1, g.q2)
        elif g.kind == Op.CCX:
            out.h(g.q2)
            ccz_clifford_t(out, g.q0, g.q1, g.q2)
            out.h(g.q2)
        elif g.kind == Op.CSWAP:
            out.cx(g.q2, g.q1)
            out.h(g.q2)
            ccz_clifford_t(out, g.q0, g.q1, g.q2)
            out.h(g.q2)
            out.cx(g.q2, g.q1)
        else:
            out.append(g)
    return out^


def explicit_t_count(c: Circuit) -> Int:
    """T/Tdg gates plus non-Clifford single-qubit rotations, counted one by one."""
    var n = 0
    for g in c.ops:
        if g.kind == Op.T or g.kind == Op.TDG:
            n += 1
        elif (g.kind == Op.RZ or g.kind == Op.RX or g.kind == Op.RY or g.kind == Op.P) and not g.is_clifford():
            n += 1
        elif g.kind == Op.CCX or g.kind == Op.CCZ or g.kind == Op.CSWAP:
            n += 1000000  # must have been expanded
    return n


def qft_rotations(n: Int) -> Circuit:
    """QFT (same convention as qumo.algorithms.library.qft) with CP(l) = P(l/2) CX P(-l/2) CX P(l/2)."""
    var c = Circuit(n, name="qft_rot")
    var j = n - 1
    while j >= 0:
        c.h(j)
        var k = j - 1
        while k >= 0:
            var lam = pi * (2.0 ** Float64(k - j))
            c.p(lam / 2.0, j)
            c.cx(j, k)
            c.p(-lam / 2.0, k)
            c.cx(j, k)
            c.p(lam / 2.0, k)
            k -= 1
        j -= 1
    for i in range(n // 2):
        c.swap(i, n - 1 - i)
    return c^


def ising_step(rows: Int, cols: Int, j: Float64, g: Float64, dt: Float64) -> Circuit:
    var c = Circuit(rows * cols, name="ising2d_step")
    for q in range(rows * cols):
        c.rx(2.0 * g * dt, q)
    for r in range(rows):
        for cc in range(cols):
            var v = r * cols + cc
            if cc + 1 < cols:
                c.cx(v, v + 1)
                c.rz(2.0 * j * dt, v + 1)
                c.cx(v, v + 1)
            if r + 1 < rows:
                c.cx(v, v + cols)
                c.rz(2.0 * j * dt, v + cols)
                c.cx(v, v + cols)
    return c^


def equivalent(a: Circuit, b: Circuit, seed: UInt64) raises -> Float64:
    """Max amplitude difference of a and b applied to a random product state."""
    var rng = Rng(seed)
    var p = Circuit(a.num_qubits)
    for q in range(a.num_qubits):
        p.ry(rng.uniform() * 3.0, q)
        p.rz(rng.uniform() * 6.0, q)
    var pa = p.copy()
    pa.compose(a)
    var pb = p.copy()
    pb.compose(b)
    var sa = evolve(pa)
    var sb = evolve(pb)
    var d = 0.0
    for i in range(sa.size()):
        d = max(d, max(abs(sa.re[i] - sb.re[i]), abs(sa.im[i] - sb.im[i])))
    return d


def main() raises:
    var log = CheckLog("qsharp/resource_estimation")
    # ----- workloads -----
    var mm = Circuit(1 + 5 + num_mcz_ancillas(6), name="cmodmul_2_mod_21")
    var tg: List[Int] = [1, 2, 3, 4, 5]
    var anc: List[Int] = [6, 7, 8]
    append_controlled_modmul(mm, 0, tg, 2, 21, anc)
    var circuits = List[Circuit]()
    circuits.append(mm^)
    circuits.append(order_finding(2, 21, 8))
    circuits.append(qft_rotations(8))
    circuits.append(ising_step(10, 10, 1.0, 1.0, 0.9))

    # QFT decomposition is the library QFT
    _ = log.close("rotation-level QFT equals qft(8)", equivalent(circuits[2], qft(8), 3), 0.0, 1e-12)

    # ----- logical counts and the explicit T-count -----
    var t_ok = True
    var equiv_err = 0.0
    print("workload                qubits  gates   2q   3q   depth  T-count  explicit  recommended")
    for i in range(len(circuits)):
        ref c = circuits[i]
        var re = estimate_resources(c)
        var expanded = expand_clifford_t(c)
        var tc = explicit_t_count(expanded)
        if i == 0:  # the Toffoli expansion is checked on the state vector (9 qubits)
            equiv_err = equivalent(c, expanded, 11)
        if tc != re.t_count:
            t_ok = False
        print("  ", c.name, "  ", re.num_qubits, "  ", re.gates_total, "  ", re.gates_2q, "  ", re.gates_3q, "  ",
              re.depth, "  ", re.t_count, "  ", tc, "  ", re.recommended_method)
    _ = log.close("Clifford+T expansion of the modmul oracle is exact", equiv_err, 0.0, 1e-12)
    _ = log.check("estimate_resources T-count equals the explicit count", t_ok)

    # ----- NISQ estimates per modality preset -----
    var profiles = List[HardwareProfile]()
    profiles.append(HardwareProfile.superconducting())
    profiles.append(HardwareProfile.neutral_atom())
    profiles.append(HardwareProfile.trapped_ion())
    var dur_ok = True
    print("NISQ estimates (duration us / estimated success probability):")
    for i in range(len(circuits)):
        var line = String("  ") + circuits[i].name + ":"
        var prev = -1.0
        for prof in profiles:
            var he = estimate_hardware(circuits[i], prof)
            line += "  " + prof.name + " " + String(he.duration_us) + " / " + String(he.success_probability)
            if he.duration_us <= prev:
                dur_ok = False
            prev = he.duration_us
        print(line)
    _ = log.check("durations ordered superconducting < neutral_atom < trapped_ion", dur_ok)

    # ----- fault-tolerant estimates -----
    var budgets: List[Float64] = [1e-2, 1e-3, 1e-4]
    var ft_profiles = List[HardwareProfile]()
    ft_profiles.append(HardwareProfile.trapped_ion())
    ft_profiles.append(HardwareProfile.superconducting())
    ft_profiles.append(HardwareProfile.neutral_atom())
    var d_ok = True
    var mono_ok = True
    var order_ok = True
    print("fault-tolerant estimates (surface code):")
    for i in range(len(circuits)):
        print("  ", circuits[i].name)
        for b in budgets:
            var prev_d = 0
            for prof in ft_profiles:
                var cycle = 4.0 * prof.t_2q_ns + prof.t_meas_ns
                var ft = estimate_fault_tolerant(circuits[i], prof.err_2q, b, cycle)
                if ft.code_distance < 3:
                    d_ok = False
                if ft.code_distance < prev_d:
                    order_ok = False
                prev_d = ft.code_distance
                print("     budget", b, prof.name, "(p =", prof.err_2q, "): d =", ft.code_distance, " logical", ft.logical_qubits,
                      " factories", ft.t_factories, " physical", ft.physical_qubits_total, " runtime", ft.runtime_seconds, "s")
        for prof in ft_profiles:
            var cycle = 4.0 * prof.t_2q_ns + prof.t_meas_ns
            var prev_q = 0
            for b in budgets:
                var q = estimate_fault_tolerant(circuits[i], prof.err_2q, b, cycle).physical_qubits_total
                if q < prev_q:
                    mono_ok = False
                prev_q = q
    _ = log.check("every code distance >= 3", d_ok)
    _ = log.check("physical qubits never decrease as the budget tightens", mono_ok)
    _ = log.check("distance ordered trapped_ion <= superconducting <= neutral_atom", order_ok)
    log.finish()
