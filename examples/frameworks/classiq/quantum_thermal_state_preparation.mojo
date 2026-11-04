"""Quantum thermal state preparation: the Gibbs state of Z1 Z2 + Z1 at beta = 0.2 from the notebook's block-encoded Linbladian.

Reproduces: Classiq library - Quantum Thermal State Preparation Algorithm Implementation
Source: https://github.com/Classiq/classiq-library/blob/main/algorithms/quantum_state_preparation/gibbs/quantum_thermal_state_preparation.ipynb (Classiq Library, MIT)

The notebook runs 50 delta steps of the Chen, Kastoryano, Brandao and Gilyen Gibbs sampler on two qubits. One step writes the energy differences of a jump operator (X0, Y0, X1 or Y1, chosen by a 2-qubit register) into a 4-qubit Bohr-frequency register with an operator Fourier transform (Gaussian window, controlled Hamiltonian evolution, inverse QFT); writes Glauber amplitudes sqrt(1/(exp(beta w0 f) + 1)) on a Boltzmann qubit; rotates a flag qubit where the Boltzmann qubit is 0; undoes the block encoding where the flag is 0; and resets the ancillas. QUMO builds the same 10-qubit step unitary and checks its parts against closed forms: the Gaussian window, the 16 Glauber amplitudes, the 64 amplitudes of the operator Fourier transform with O = H x H (a discrete Fourier sum), and trace preservation of the step channel. The 50 steps are then evaluated exactly, and the final populations are compared with the notebook's printed 1000-shot sample (three binomial standard errors), with its printed Gibbs weights (1e-7), and with its own assertion (atol 0.1).

Deviations from the notebook:
* Exact, not sampled. Each step resets its ancillas, so on the two system qubits it is one fixed channel. QUMO runs the step unitary once from each of the four system basis states, reads the 256 Kraus operators K_k off the statevector (k = ancilla outcome), and applies sum_k K_k rho K_k^dag fifty times to the 4 x 4 density matrix. The result is the exact mean of the notebook's RESET-and-shots estimate. The density-matrix simulator would hold 4^10 entries and pass over them for every gate, so it is not used.
* Gaussian window: Classiq's inplace_prepare_amplitudes is QUMO's Mottonen amplitude embedding of the same real vector.
* Glauber lookup: assign_amplitude_table followed by X is written as 16 multi-controlled RY gates on the Boltzmann qubit, one per Bohr value, leaving the same amplitudes on |0> and |1>.
* Hamiltonian evolution: the notebook's order-1 Suzuki-Trotter step (one repetition) of Z0 Z1 + Z1 is exact here because the two terms commute, so RZZ and RZ gates replace it, controlled with controlled_by.
* control(), within_apply() and prepare_uniform_trimmed_state(4) are written out: H on the two jump qubits, then X or Y on system qubit 0 or 1 under each jump value (controlled_by with X-flips). invert(qft) is QUMO's iqft; the closed-form OFT check confirms its phase sign.
* The Bohr register is read as a signed value in the Glauber lookup, as the notebook declares it SIGNED.

Run: .venv/bin/mojo run -I . examples/frameworks/classiq/quantum_thermal_state_preparation.mojo
example-meta: {"framework": "classiq", "tier": "fast", "timeout": 120, "requires": [], "args": []}
"""

from std.math import acos, asin, cos, exp, pi, sin, sqrt
from qumo.ir.circuit import Circuit
from qumo.ir.mc import controlled_by
from qumo.ir.pauli import PauliSum, PauliTerm
from qumo.algorithms.library import append_pauli_exponential, iqft
from qumo.algorithms.qml import append_amplitude_embedding_real
from qumo.sim.runner import evolve, expectation
from qumo.testing.checks import CheckLog


# Qubit layout of one step: system 0-1, jump 2-3, Bohr frequency 4-7 (bit j on qubit 4 + j),
# Boltzmann 8, flag 9. The ancillas (qubits 2-9) are reset after every step.
comptime NQ = 10
comptime JUMP0 = 2
comptime BOHR0 = 4
comptime NBOHR = 4
comptime BOLTZ = 8
comptime FLAG = 9
comptime NKRAUS = 256  # ancilla outcomes: 2 + 4 + 1 + 1 qubits
comptime STEPS = 50
comptime SHOTS = 1000


def bohr_qubit(j: Int) -> Int:
    return BOHR0 + j


def bohr_qubits() -> List[Int]:
    var out = List[Int]()
    for j in range(NBOHR):
        out.append(bohr_qubit(j))
    return out^


def gaussian_window() -> List[Float64]:
    """The notebook's amplitudes exp(-x^2) on linspace(-2, 2, 16), normalised; entry k belongs to Bohr value k."""
    var out = List[Float64]()
    var norm2 = 0.0
    var n = 1 << NBOHR
    for k in range(n):
        var x = -2.0 + 4.0 * Float64(k) / Float64(n - 1)
        var g = exp(-x * x)
        out.append(g)
        norm2 += g * g
    var inv = 1.0 / sqrt(norm2)
    for k in range(n):
        out[k] = out[k] * inv
    return out^


def energy_scale() -> Tuple[Float64, Float64]:
    """Return omega_0 and t_0 of the notebook's get_fourier_parameters(4, 4)."""
    var n = 1 << NBOHR
    var w0 = (2.0 + 0.5) * 4.0 / Float64(n)
    var t0 = 2.0 * pi / (w0 * Float64(n))
    return (w0, t0)


def hamiltonian_step(tau: Float64) raises -> Circuit:
    """Evolve exp(-i tau H), H = Z0 Z1 + Z1 on the system qubits; the two terms commute, so this is exact."""
    var c = Circuit(NQ, name="hamiltonian_step")
    append_pauli_exponential(c, PauliTerm(1.0, UInt64(0), UInt64(3)), tau)
    append_pauli_exponential(c, PauliTerm(1.0, UInt64(0), UInt64(2)), tau)
    return c^


def controlled_evolution(t0: Float64) raises -> Circuit:
    """The notebook's controlled_hamiltonian_simulation: Bohr bit i controls exp(-i 2^i t0 H)."""
    var c = Circuit(NQ, name="controlled_evolution")
    for i in range(NBOHR):
        var ctrl = List[Int]()
        ctrl.append(bohr_qubit(i))
        c.compose(controlled_by(hamiltonian_step(Float64(1 << i) * t0), ctrl))
    return c^


def compose_when_equal(mut c: Circuit, sub: Circuit, qubits: List[Int], value: Int) raises:
    """Append `sub` under the condition that the register `qubits` (bit j on qubits[j]) holds `value`."""
    for j in range(len(qubits)):
        if ((value >> j) & 1) == 0:
            c.x(qubits[j])
    c.compose(controlled_by(sub, qubits))
    for j in range(len(qubits)):
        if ((value >> j) & 1) == 0:
            c.x(qubits[j])


def jump_operator() raises -> Circuit:
    """The notebook's pauli_jump_operators: the jump register is uniform over X0, Y0, X1, Y1
    (value j acts on system qubit j // 2, with X for even j and Y for odd j), and the Pauli is applied
    controlled on the value."""
    var c = Circuit(NQ, name="jump_operator")
    var jump = List[Int]()
    jump.append(JUMP0)
    jump.append(JUMP0 + 1)
    c.h(JUMP0)
    c.h(JUMP0 + 1)
    for j in range(4):
        var sub = Circuit(NQ)
        if j % 2 == 0:
            sub.x(j // 2)
        else:
            sub.y(j // 2)
        compose_when_equal(c, sub, jump, j)
    return c^


def operator_fourier_transform(t0: Float64, jumps: Bool) raises -> Circuit:
    """The notebook's operator_fourier_transform: Gaussian window on the Bohr register, controlled evolution,
    the operator, the inverse of the controlled evolution, then the inverse QFT (invert(qft)).
    With jumps = False the operator is H x H on the system, for the closed-form check."""
    var c = Circuit(NQ, name="operator_fourier_transform")
    var wires = List[Int]()
    for j in range(NBOHR - 1, -1, -1):
        wires.append(bohr_qubit(j))
    append_amplitude_embedding_real(c, gaussian_window(), wires)
    var fwd = controlled_evolution(t0)
    c.compose(fwd)
    if jumps:
        c.compose(jump_operator())
    else:
        c.h(0)
        c.h(1)
    c.compose(fwd.inverse())
    c.compose(iqft(NBOHR), BOHR0)
    return c^


def glauber_lookup(beta: Float64, w0: Float64) raises -> Circuit:
    """The notebook's apply_boltzmann_weight: the Boltzmann qubit gets amplitude sqrt(1/(exp(beta w0 f) + 1))
    on |0>, where f is the signed Bohr value. Written as one multi-controlled RY per Bohr value."""
    var c = Circuit(NQ, name="glauber_lookup")
    var n = 1 << NBOHR
    for p in range(n):
        var f = Float64(p) if p < n // 2 else Float64(p - n)
        var a = sqrt(1.0 / (exp(beta * w0 * f) + 1.0))
        var sub = Circuit(NQ)
        sub.ry(2.0 * acos(a), BOLTZ)
        compose_when_equal(c, sub, bohr_qubits(), p)
    return c^


def block_encoding(beta: Float64, w0: Float64, t0: Float64) raises -> Circuit:
    """The notebook's block_encode_linbladian: the operator Fourier transform, then the Glauber lookup."""
    var c = Circuit(NQ, name="block_encoding")
    c.compose(operator_fourier_transform(t0, True))
    c.compose(glauber_lookup(beta, w0))
    return c^


def delta_step(block: Circuit, delta: Float64) raises -> Circuit:
    """The unitary part of the notebook's delta_step: the block encoding; RY(asin(2 sqrt(delta))) on the flag
    where the Boltzmann qubit is 0; the inverse block encoding where the flag is 0. The resets follow in the Kraus step."""
    var c = Circuit(NQ, name="delta_step")
    c.compose(block)
    var rot = Circuit(NQ)
    rot.ry(asin(2.0 * sqrt(delta)), FLAG)
    var ctrl_b = List[Int]()
    ctrl_b.append(BOLTZ)
    c.x(BOLTZ)
    c.compose(controlled_by(rot, ctrl_b))
    c.x(BOLTZ)
    var ctrl_f = List[Int]()
    ctrl_f.append(FLAG)
    c.x(FLAG)
    c.compose(controlled_by(block.inverse(), ctrl_f))
    c.x(FLAG)
    return c^


def basis_state(s: Int) raises -> Circuit:
    """|s> on the system qubits (state index s = q0 + 2 q1)."""
    var c = Circuit(NQ, name="basis_state")
    for q in range(2):
        if ((s >> q) & 1) == 1:
            c.x(q)
    return c^


def kraus_operators(step: Circuit, mut kre: List[Float64], mut kim: List[Float64]) raises:
    """K_k[s][i] = <s, k| step |i, 0> for system states s, i and ancilla outcome k (qubits 2-9),
    stored at k * 16 + s * 4 + i. The ancillas start in |0> and are traced out, which is the reset."""
    kre = List[Float64](length=NKRAUS * 16, fill=0.0)
    kim = List[Float64](length=NKRAUS * 16, fill=0.0)
    for i in range(4):
        var prep = basis_state(i)
        prep.compose(step)
        var sv = evolve(prep)
        for k in range(NKRAUS):
            for s in range(4):
                kre[k * 16 + s * 4 + i] = sv.re[s + 4 * k]
                kim[k * 16 + s * 4 + i] = sv.im[s + 4 * k]


def trace_preservation_error(kre: List[Float64], kim: List[Float64]) -> Float64:
    """Largest |(sum_k K_k^dag K_k)_ij - delta_ij| over i, j."""
    var worst = 0.0
    for i in range(4):
        for j in range(4):
            var ar = 0.0
            var ai = 0.0
            for k in range(NKRAUS):
                for s in range(4):
                    var xr = kre[k * 16 + s * 4 + i]
                    var xi = -kim[k * 16 + s * 4 + i]
                    var yr = kre[k * 16 + s * 4 + j]
                    var yi = kim[k * 16 + s * 4 + j]
                    ar += xr * yr - xi * yi
                    ai += xr * yi + xi * yr
            var want = 1.0 if i == j else 0.0
            worst = max(worst, abs(ar - want), abs(ai))
    return worst


def apply_channel(
    kre: List[Float64], kim: List[Float64], rre: List[Float64], rim: List[Float64],
    mut nre: List[Float64], mut nim: List[Float64],
):
    """Channel update rho' = sum_k K_k rho K_k^dag on the 4 x 4 system density matrix (entry s * 4 + t)."""
    for q in range(16):
        nre[q] = 0.0
        nim[q] = 0.0
    for k in range(NKRAUS):
        var base = k * 16
        var tre = List[Float64](length=16, fill=0.0)
        var tim = List[Float64](length=16, fill=0.0)
        for s in range(4):
            for j in range(4):
                for i in range(4):
                    var ar = kre[base + s * 4 + i]
                    var ai = kim[base + s * 4 + i]
                    var br = rre[i * 4 + j]
                    var bi = rim[i * 4 + j]
                    tre[s * 4 + j] += ar * br - ai * bi
                    tim[s * 4 + j] += ar * bi + ai * br
        for s in range(4):
            for t in range(4):
                for j in range(4):
                    var ar = tre[s * 4 + j]
                    var ai = tim[s * 4 + j]
                    var br = kre[base + t * 4 + j]
                    var bi = -kim[base + t * 4 + j]
                    nre[s * 4 + t] += ar * br - ai * bi
                    nim[s * 4 + t] += ar * bi + ai * br


def evolve_channel(
    kre: List[Float64], kim: List[Float64], steps: Int, mut rre: List[Float64], mut rim: List[Float64],
):
    """Apply the step channel `steps` times to rre + i rim, starting from the state passed in."""
    for _ in range(steps):
        var nre = List[Float64](length=16, fill=0.0)
        var nim = List[Float64](length=16, fill=0.0)
        apply_channel(kre, kim, rre, rim, nre, nim)
        rre = nre^
        rim = nim^


def main() raises:
    var log = CheckLog("classiq/quantum_thermal_state_preparation")
    var beta = 0.2
    var delta = 0.1
    var scale = energy_scale()
    var w0 = scale[0]
    var t0 = scale[1]
    print("omega_0 =", w0, " t_0 =", t0, " steps =", STEPS, " delta =", delta, " beta =", beta)

    # ---- the notebook's matrix H = [[2,0,0,0],[0,0,0,0],[0,0,-2,0],[0,0,0,0]] and its printed expected_probs ----
    var printed_energy = List[Float64]()
    printed_energy.append(2.0)
    printed_energy.append(0.0)
    printed_energy.append(-2.0)
    printed_energy.append(0.0)
    var printed_gibbs = List[Float64]()
    printed_gibbs.append(0.16105159)
    printed_gibbs.append(0.24026075)
    printed_gibbs.append(0.35842691)
    printed_gibbs.append(0.24026075)
    var printed_sampled = List[Float64]()
    printed_sampled.append(0.18)
    printed_sampled.append(0.253)
    printed_sampled.append(0.337)
    printed_sampled.append(0.23)

    var h = PauliSum()
    h.add(1.0, UInt64(0), UInt64(3))  # Z0 Z1
    h.add(1.0, UInt64(0), UInt64(2))  # Z1
    var energy = List[Float64]()
    for s in range(4):
        energy.append(expectation(basis_state(s), h))
    var gibbs = List[Float64]()
    var zsum = 0.0
    for s in range(4):
        var wgt = exp(-beta * energy[s])
        gibbs.append(wgt)
        zsum += wgt
    for s in range(4):
        gibbs[s] = gibbs[s] / zsum
    for s in range(4):
        _ = log.close("energy of state " + String(s) + " matches the notebook matrix diagonal", energy[s], printed_energy[s], 1e-12)
    for s in range(4):
        _ = log.close("Gibbs weight of state " + String(s) + " matches the notebook expected_probs", gibbs[s], printed_gibbs[s], 1e-7)

    # ---- the Gaussian window and the Glauber lookup against their closed forms ----
    var g = gaussian_window()
    var gc = Circuit(NQ, name="gaussian_only")
    var wires = List[Int]()
    for j in range(NBOHR - 1, -1, -1):
        wires.append(bohr_qubit(j))
    append_amplitude_embedding_real(gc, g, wires)
    var gsv = evolve(gc)
    var gdev = 0.0
    for k in range(1 << NBOHR):
        gdev = max(gdev, abs(gsv.re[k << BOHR0] - g[k]), abs(gsv.im[k << BOHR0]))
    _ = log.close("Gaussian window amplitudes match exp(-x^2) on linspace(-2, 2, 16)", gdev, 0.0, 1e-12)

    var bl = glauber_lookup(beta, w0)
    var bdev = 0.0
    for p in range(1 << NBOHR):
        var prep = Circuit(NQ, name="glauber_input")
        for j in range(NBOHR):
            if ((p >> j) & 1) == 1:
                prep.x(bohr_qubit(j))
        prep.compose(bl)
        var bsv = evolve(prep)
        var f = Float64(p) if p < (1 << (NBOHR - 1)) else Float64(p - (1 << NBOHR))
        var want = sqrt(1.0 / (exp(beta * w0 * f) + 1.0))
        bdev = max(bdev, abs(bsv.re[p << BOHR0] - want), abs(bsv.im[p << BOHR0]))
    _ = log.close("Glauber amplitudes on the 16 Bohr values match sqrt(1/(exp(beta w0 f)+1))", bdev, 0.0, 1e-12)

    # ---- operator Fourier transform with O = H x H: the closed-form discrete sum ----
    # O|00> = (1/2) sum_x |x>, E_in = E(|00>) = 2, Delta_x = E_x - E_in, and the Bohr amplitude is
    # (1/2)(1/4) sum_k g_k exp(2 pi i k (Delta_x / w0 - m) / 16).
    var oft_sv = evolve(operator_fourier_transform(t0, False))
    var odev = 0.0
    for x in range(4):
        var u0 = (printed_energy[x] - printed_energy[0]) / w0
        for m in range(1 << NBOHR):
            var sre = 0.0
            var sim = 0.0
            for k in range(1 << NBOHR):
                var th = 2.0 * pi * Float64(k) * (u0 - Float64(m)) / Float64(1 << NBOHR)
                sre += g[k] * cos(th)
                sim += g[k] * sin(th)
            var idx = x + (m << BOHR0)
            odev = max(odev, abs(oft_sv.re[idx] - 0.125 * sre), abs(oft_sv.im[idx] - 0.125 * sim))
    _ = log.close("operator Fourier transform amplitudes match the closed-form window sum on 64 entries", odev, 0.0, 1e-12)

    # ---- one delta step as Kraus operators on the system, and their trace preservation ----
    var step = delta_step(block_encoding(beta, w0, t0), delta)
    print("delta step gates:", len(step.ops), " qubits:", step.num_qubits)
    var kre = List[Float64]()
    var kim = List[Float64]()
    kraus_operators(step, kre, kim)
    _ = log.close("step channel is trace preserving (sum_k K^dag K = I)", trace_preservation_error(kre, kim), 0.0, 1e-12)

    # ---- 50 steps on the system density matrix, from |00> ----
    var rre = List[Float64](length=16, fill=0.0)
    var rim = List[Float64](length=16, fill=0.0)
    rre[0] = 1.0
    evolve_channel(kre, kim, STEPS, rre, rim)
    var pop = List[Float64]()
    var total = 0.0
    var top = 0
    for s in range(4):
        var p = rre[s * 4 + s]
        pop.append(p)
        total += p
        if p > pop[top]:
            top = s
    print("state  QUMO after 50 steps  notebook sample  Gibbs")
    for s in range(4):
        print("  ", s, "  ", pop[s], "  ", printed_sampled[s], "  ", gibbs[s])
    _ = log.close("populations after 50 steps sum to 1", total, 1.0, 1e-12)
    for s in range(4):
        var se = sqrt(printed_sampled[s] * (1.0 - printed_sampled[s]) / Float64(SHOTS))
        _ = log.close("population of state " + String(s) + " within three standard errors of the notebook sample", pop[s], printed_sampled[s], 3.0 * se)
    for s in range(4):
        _ = log.close("population of state " + String(s) + " within the notebook atol 0.1 of Gibbs", pop[s], gibbs[s], 0.1)
    _ = log.check("most populated state is the Gibbs mode (state 2)", top == 2, "top=" + String(top))

    log.finish()
