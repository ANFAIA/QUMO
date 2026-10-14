"""Sample-based Krylov quantum diagonalization (SKQD) of a Heisenberg chain.

Reproduces: Qiskit - Sample-based Krylov quantum diagonalization (method), on the CUDA-Q SKQD Heisenberg instance at 12 qubits
Source: https://quantum.cloud.ibm.com/docs/tutorials/sample-based-krylov-quantum-diagonalization (IBM Quantum documentation; tutorial code Apache-2.0, text CC BY-SA 4.0)
Source: https://nvidia.github.io/cuda-quantum/0.14.0/applications/python/skqd.html (CUDA-Q documentation, Apache-2.0)

SKQD is the "quantum sampling + classical HPC" pattern:
1. prepare Krylov states |psi_k> = U^k |psi_0>, k = 0..D-1, with U a
   Trotterised exp(-i H dt) (`trotter`, second order, 2 slices per step);
2. sample each state in the computational basis (`run_sampler`, 4000 shots);
3. pool the distinct bitstrings into a subspace basis;
4. project H onto that basis and diagonalise classically
   (`projected_hamiltonian` / `subspace_ground_energy`, sparse Lanczos for
   large subspaces).

The instance differs from both pages, as stated here:
- the IBM tutorial applies SKQD/SQD to a single-impurity Anderson model
  (16 and 40 qubits, qiskit-addon-sqd with configuration recovery);
- the CUDA-Q page uses a 22-qubit Heisenberg chain, Jx = Jy = Jz = 1 with
  fields hx = hy = hz = 1, a Neel initial state and dt = pi/8.
This example keeps the CUDA-Q Hamiltonian and dt at 12 qubits (open chain),
where the exact ground energy is known by Lanczos; neither page prints its
final SKQD energy, so the reference is exact diagonalisation. The X and Y
fields break magnetisation conservation, so the full 4096-state space is
in play and the projected matrix is complex (Y terms).

Checks:
- variational: E_SKQD(D) >= E_exact - 1e-9 for every D, and non-increasing in D;
- relative error <= 1% at D = 8 with 4000 shots per Krylov state;
- the full basis (4096 states) reproduces the exact ground energy (1e-10).

Run: .venv/bin/mojo run -I . examples/frameworks/qiskit/skqd_heisenberg.mojo
example-meta: {"framework": "qiskit", "tier": "slow", "timeout": 300, "requires": [], "args": []}
"""

from std.math import pi
from std.time import perf_counter_ns
from qumo.ir.circuit import Circuit
from qumo.ir.pauli import PauliSum
from qumo.exec.primitives import SamplerPub, run_sampler
from qumo.algorithms.hamiltonians import heisenberg_xxz, ground_energy, subspace_ground_energy
from qumo.algorithms.library import trotter, bitstring_to_int
from qumo.ai.checks import CheckLog

comptime NQ = 12
comptime D_MAX = 8
comptime SHOTS = 4000
comptime SLICES = 2  # Trotter slices per Krylov step


def cudaq_heisenberg(n: Int) raises -> PauliSum:
    """J sum (XX + YY + ZZ) on an open chain + h sum (X + Y + Z), J = h = 1."""
    var h = heisenberg_xxz(n, 1.0, 1.0, False, 1.0)  # includes the Z field
    for q in range(n):
        var b = UInt64(1) << UInt64(q)
        h.add(1.0, b, 0)  # X_q
        h.add(1.0, b, b)  # Y_q
    return h^


def main() raises:
    var log = CheckLog("qiskit/skqd_heisenberg")
    var h = cudaq_heisenberg(NQ)
    var t0 = perf_counter_ns()
    var e_exact = ground_energy(h, NQ)
    print("H: 12-qubit Heisenberg chain + (X, Y, Z) fields,", len(h), "Pauli terms; exact E0 =", e_exact,
          "(Lanczos,", Float64(perf_counter_ns() - t0) / 1e9, "s)")

    var dt = pi / 8.0
    var step = trotter(h, dt, SLICES, 2, NQ)
    print("Krylov step U = Strang trotter(dt = pi/8, 2 slices):", len(step), "gates")
    var pool = List[Int]()
    var seen = List[Bool](length=1 << NQ, fill=False)
    var c = Circuit(NQ, name="krylov")
    for q in range(NQ):
        if q % 2 == 1:
            c.x(q)  # Neel reference state
    var prev = 1e300
    var variational = True
    var monotone = True
    var e_last = 0.0
    print("   D   subspace dim   E_SKQD               rel. error")
    for k in range(D_MAX):
        if k > 0:
            c.compose(step)
        var m = c.copy()
        m.measure_all()
        var res = run_sampler(SamplerPub(m, SHOTS, seed=UInt64(1000 + k)))
        for kv in res[0].counts.items():
            var b = bitstring_to_int(kv.key)
            if not seen[b]:
                seen[b] = True
                pool.append(b)
        sort(pool)
        var e = subspace_ground_energy(h, pool, NQ)
        var rel = (e - e_exact) / abs(e_exact)
        print("  ", k + 1, "  ", len(pool), "          ", e, "  ", rel)
        if e < e_exact - 1e-9:
            variational = False
        if e > prev + 1e-9:
            monotone = False
        prev = e
        e_last = e
    _ = log.check("variational at every D", variational)
    _ = log.check("energy non-increasing in D", monotone)
    var rel8 = (e_last - e_exact) / abs(e_exact)
    _ = log.within("relative error at D=8", rel8, 0.0, 0.01)

    var all_states = List[Int](capacity=1 << NQ)
    for i in range(1 << NQ):
        all_states.append(i)
    var e_full = subspace_ground_energy(h, all_states, NQ)
    print("full basis (4096 states):", e_full, " exact:", e_exact)
    _ = log.close("full-basis projection equals exact", e_full, e_exact, 1e-10)
    print("total time", Float64(perf_counter_ns() - t0) / 1e9, "s")
    log.finish()
