"""Cross-checks of the 2026-11 framework-update features against the frameworks
they came from. Every section prints what it compared; numbers quoted in docs/
come from here.

    PYTHONPATH=build:plugins/hardware bench/.venv/bin/python bench/crosscheck/tier1.py [section ...]

Sections: mcx wrap layout pbc tcount density stim dem lint correlated m3 ptsbe branching
`m3` needs mthree, which `make setup-py` does not install. QUMO_REF_VENV selects a
second Python for it (default: skip).
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, "build")
import qumo_py as q  # noqa: E402


def header(name: str) -> None:
    print(f"\n## {name}\n", flush=True)


def to_qiskit(c):
    from qiskit import QuantumCircuit

    return QuantumCircuit.from_qasm_str(c.to_qasm2())


def section_mcx() -> None:
    header("MCX: CX counts after {u, cx}, against Qiskit's synthesis")
    from qiskit import transpile
    from qiskit.quantum_info import Operator
    from qiskit.synthesis import synth_mcx_1_clean_kg24, synth_mcx_n_clean_m15, synth_mcx_n_dirty_i15, synth_mcx_noaux_hp24, synth_mcx_noaux_v24

    def cx_qiskit(c):
        return transpile(c, basis_gates=["u", "cx"], optimization_level=1).count_ops().get("cx", 0)

    def cx_qumo(c):
        return dict(q.translate(c, ["u", "cx"]).count_ops()).get("cx", 0)

    print("| k | QUMO noaux | Qiskit v24 | Qiskit hp24 | QUMO 1 clean | Qiskit kg24 | QUMO k-2 dirty | Qiskit i15 |")
    print("|---|---|---|---|---|---|---|---|")
    for k in (3, 4, 5, 6, 7, 8, 10, 12, 16, 20):
        row = [k, cx_qumo(q.mcx_circuit(k, "noaux", 0, False)), cx_qiskit(synth_mcx_noaux_v24(k)), cx_qiskit(synth_mcx_noaux_hp24(k)),
               cx_qumo(q.mcx_circuit(k, "clean", 1, True)), cx_qiskit(synth_mcx_1_clean_kg24(k)),
               cx_qumo(q.mcx_circuit(k, "dirty", k - 2, False)), cx_qiskit(synth_mcx_n_dirty_i15(k))]
        print("| " + " | ".join(map(str, row)) + " |", flush=True)
    bad = 0
    from qiskit.circuit.library import MCXGate
    from qiskit import QuantumCircuit

    for k in range(1, 11):
        ref = QuantumCircuit(k + 1)
        ref.append(MCXGate(k), list(range(k + 1)))
        got = to_qiskit(q.mcx_circuit(k, "noaux", 0, False))
        if not Operator(got).equiv(Operator(ref)):
            bad += 1
    print(f"\nunitary against Qiskit MCXGate, k = 1..10: {10 - bad} of 10 equal")
    assert bad == 0


def section_wrap() -> None:
    header("wrap_angles against FoldRzzAngle (qiskit-ibm-runtime)")
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from qiskit.transpiler import PassManager
    from qiskit_ibm_runtime.transpiler.passes import FoldRzzAngle

    rng = np.random.default_rng(3)
    same = 0
    extra_qumo = extra_ibm = 0
    trials = 60
    for _ in range(trials):
        theta = float(rng.uniform(-12, 12))
        c = q.load_qasm(f'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; rzz({theta!r}) q[0],q[1];')
        out, report = q.compile_spec(c, 2, [], ["rzz", "x", "z"], "wrap_angles(rzz[0:pi/2])")
        ours = to_qiskit(out)
        ref = QuantumCircuit(2)
        ref.rzz(theta, 0, 1)
        theirs = PassManager([FoldRzzAngle()]).run(ref)
        assert Operator(ours).equiv(Operator(ref)), theta
        assert Operator(theirs).equiv(Operator(ref)), theta
        a = [float(i.operation.params[0]) for i in ours.data if i.operation.name == "rzz"]
        b = [float(i.operation.params[0]) for i in theirs.data if i.operation.name == "rzz"]
        assert len(a) == 1 and 0 <= a[0] <= math.pi / 2 + 1e-12
        same += abs(a[0] - b[0]) < 1e-9
        extra_qumo += len(ours.data) - 1
        extra_ibm += len([i for i in theirs.data if i.operation.num_qubits > 0]) - 1
    print(f"{trials} random angles: every output equals the rzz as an operator, up to a global phase")
    print(f"same rzz angle as FoldRzzAngle in {same} of {trials}; Pauli corrections per gate: QUMO {extra_qumo / trials:.2f}, IBM {extra_ibm / trials:.2f}")


def _true_fidelity(circ, spec, err1) -> float:
    """Estimated success under the device's real calibration, from the compiled gates."""
    log_f = 0.0
    qc = circ
    for inst in qc.data:
        name = inst.operation.name
        qs = [qc.find_bit(x).index for x in inst.qubits]
        if name in ("barrier", "rz", "id"):
            continue
        if name == "measure":
            log_f += math.log(1 - (spec.readout_errors[qs[0]] or 0.01))
        elif len(qs) == 1:
            log_f += math.log(1 - err1.get(qs[0], 3e-4))
        else:
            e = spec.edge_errors.get(f"{min(qs)}-{max(qs)}", spec.errors.get("2q", 0.01))
            log_f += math.log(max(1 - e, 1e-9))
    return math.exp(log_f)


def section_layout() -> None:
    header("Noise-aware layout on calibrated fake backends (estimated success, higher is better)")
    from qiskit import QuantumCircuit, transpile
    from qiskit_ibm_runtime.fake_provider import FakeFez, FakeMarrakesh, FakeTorino

    sys.path.insert(0, "plugins/hardware")
    from qumo_hardware.compile import compile_for_target
    from qumo_hardware.ibm import target_from_qiskit

    circuits = {}
    for n in (5, 10, 16):
        c = QuantumCircuit(n)
        c.h(0)
        for i in range(n - 1):
            c.cx(i, i + 1)
        c.measure_all()
        circuits[f"ghz_{n}"] = c
    for n in (6, 8):
        c = QuantumCircuit(n)
        for i in range(n):
            c.h(i)
            for j in range(i + 1, n):
                c.cp(math.pi / 2 ** (j - i), j, i)
        c.measure_all()
        circuits[f"qft_{n}"] = c
    c = QuantumCircuit(12)
    for i in range(11):
        c.h(i)
        c.cx(i, 11)
    c.measure_all()
    circuits["bv_12"] = c
    print("| backend | circuit | whole device | select_region | calibrated | calibrated / best | Qiskit O3 |")
    print("|---|---|---|---|---|---|---|")
    ratios = []
    ratios_plain = []
    for backend in (FakeTorino(), FakeFez(), FakeMarrakesh()):
        spec = target_from_qiskit(backend, False, "ibm")
        err1 = {}
        for qargs, props in backend.target["sx"].items():
            if props is not None and props.error is not None:
                err1[qargs[0]] = props.error
        for name, qc in circuits.items():
            text = __import__("qiskit.qasm2", fromlist=["dumps"]).dumps(qc)
            plain = compile_for_target(text, spec, level=2, seed=1, region="none", calibrate=False)
            auto = compile_for_target(text, spec, level=2, seed=1, region="auto", calibrate=False)
            cal = compile_for_target(text, spec, level=2, seed=1, region="auto", calibrate=True)
            fp = _true_fidelity(QuantumCircuit.from_qasm_str(plain.qasm), spec, err1)
            fr = _true_fidelity(QuantumCircuit.from_qasm_str(auto.qasm), spec, err1)
            fa = max(fp, fr)
            fb = _true_fidelity(QuantumCircuit.from_qasm_str(cal.qasm), spec, err1)
            ref = transpile(qc, backend, optimization_level=3, seed_transpiler=1)
            fq = _true_fidelity(ref, spec, err1)
            ratios.append(fb / fa)
            ratios_plain.append(fb / fp)
            print(f"| {backend.name} | {name} | {fp:.4g} | {fr:.4g} | {fb:.4g} | {fb / fa:.2f} | {fq:.4g} |", flush=True)
    print(f"\ngeomean calibrated / best of today's two: {math.exp(np.mean(np.log(ratios))):.3f}; calibrated / whole device: {math.exp(np.mean(np.log(ratios_plain))):.3f}")


def section_pbc() -> None:
    header("to_pbc against Qiskit LitinskiTransformation")
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from qiskit.transpiler import PassManager

    try:
        from qiskit.transpiler.passes import LitinskiTransformation
    except ImportError:
        print("LitinskiTransformation is not in this Qiskit")
        return
    rng = np.random.default_rng(1)
    agree = 0
    trials = 40
    for t in range(trials):
        n = 4
        qc = QuantumCircuit(n)
        for _ in range(30):
            r = rng.integers(0, 5)
            a = int(rng.integers(0, n))
            b = int((a + 1 + rng.integers(0, n - 1)) % n)
            if r == 0:
                qc.h(a)
            elif r == 1:
                qc.s(a)
            elif r == 2:
                qc.t(a)
            elif r == 3:
                qc.tdg(a)
            else:
                qc.cx(a, b)
        text = __import__("qiskit.qasm2", fromlist=["dumps"]).dumps(qc)
        ours = json.loads(q.to_pbc(q.load_qasm(text), True))
        rot = [(o["pauli"], round(o["angle"], 9)) for o in ours["ops"]]
        out = PassManager([LitinskiTransformation()]).run(qc)
        theirs = []
        for inst in out.data:
            op = inst.operation
            if op.name == "PauliEvolution":
                letters, where, coeff = op.operator.to_sparse_list()[0]
                full = ["I"] * n
                idx = [out.find_bit(x).index for x in inst.qubits]
                for ch, pos in zip(letters, where):
                    full[idx[pos]] = ch
                theirs.append(("".join(reversed(full)), round(2 * float(np.real(coeff)) * float(op.time), 9)))
        # same multiset of Pauli rotations up to the sign convention of P
        key = lambda x: (x[0], abs(x[1]))  # noqa: E731
        agree += sorted(map(key, rot)) == sorted(map(key, theirs))
    print(f"{agree} of {trials} random Clifford+T circuits: same Pauli rotations (as multisets)")


def section_tcount() -> None:
    header("pauli_simp(objective=t) on the Feynman benchmarks")
    names = ["tof_3", "barenco_tof_3", "mod5_4", "rc_adder_6", "gf2p4_mult", "csum_mux_9", "qcla_com_7", "mod_mult_55", "adder_8", "ham15-low"]
    fold = "unroll3q,fixpoint(cancel,merge,commute,phase_fold,cancel,merge,max=6)"
    with_t = "unroll3q,pauli_simp(objective=t),fixpoint(cancel,merge,commute,phase_fold,cancel,merge,max=6)"
    print("| circuit | T in | fold | fold + objective=t | upstream fold (report) |")
    print("|---|---|---|---|---|")
    upstream = {"tof_3": 15, "barenco_tof_3": 16, "mod5_4": 8, "rc_adder_6": 47, "gf2p4_mult": 68, "csum_mux_9": 84, "qcla_com_7": 95, "mod_mult_55": 35, "adder_8": 173, "ham15-low": 97}
    for name in names:
        path = f"bench/neutral/circuits/feynman__{name}.qasm"
        if not os.path.exists(path):
            continue
        c = q.load_qasm(open(path).read())
        n = int(c.num_qubits())
        a, _ = q.compile_spec(c, n, [], [], fold)
        b, _ = q.compile_spec(c, n, [], [], with_t)
        def tc(x):
            # T gates and rotations that are not Clifford
            total = 0
            for inst in to_qiskit(x).data:
                name_ = inst.operation.name
                if name_ in ("t", "tdg"):
                    total += 1
                elif name_ in ("rz", "p", "u1"):
                    turns = float(inst.operation.params[0]) / (math.pi / 2)
                    total += abs(turns - round(turns)) > 1e-9
            return total
        print(f"| {name} | {tc(q.compile_spec(c, n, [], [], 'unroll3q')[0])} | {tc(a)} | {tc(b)} | {upstream.get(name, '-')} |", flush=True)


def section_density() -> None:
    header("Density matrix against Qiskit Aer and Cirq")
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    rng = np.random.default_rng(2)
    worst = 0.0
    for t in range(6):
        n = 5
        qasm = ['OPENQASM 3.0; include "stdgates.inc";', f"qubit[{n}] q;"]
        for _ in range(30):
            r = rng.integers(0, 4)
            a = int(rng.integers(0, n))
            b = int((a + 1 + rng.integers(0, n - 1)) % n)
            if r == 0:
                qasm.append(f"rx({float(rng.uniform(0, 3))!r}) q[{a}];")
            elif r == 1:
                qasm.append(f"rz({float(rng.uniform(0, 3))!r}) q[{a}];")
            elif r == 2:
                qasm.append(f"h q[{a}];")
            else:
                qasm.append(f"cx q[{a}], q[{b}];")
                qasm.append(f"#pragma qumo depolarize2(0.02) q[{a}], q[{b}];")
        ours = np.array(q.density_probabilities(q.load_qasm("\n".join(qasm))))
        # the same circuit in Aer, noise after each cx
        qc = QuantumCircuit(n)
        for line in qasm[2:]:
            parts = line.replace(";", "").split()
            if line.startswith("#"):
                continue
            if parts[0].startswith("rx("):
                qc.rx(float(parts[0][3:-1]), int(parts[1][2:-1]))
            elif parts[0].startswith("rz("):
                qc.rz(float(parts[0][3:-1]), int(parts[1][2:-1]))
            elif parts[0] == "h":
                qc.h(int(parts[1][2:-1]))
            else:
                qc.cx(int(parts[1][2:-2]), int(parts[2][2:-1]))
        nm = NoiseModel()
        # QUMO: Stim's convention, each of 15 Paulis with p/15; Qiskit: lambda = p * 16/15
        nm.add_all_qubit_quantum_error(depolarizing_error(0.02 * 16 / 15, 2), ["cx"])
        qc.save_probabilities()
        sim = AerSimulator(method="density_matrix", noise_model=nm)
        probs = np.array(sim.run(qc.decompose(), shots=1).result().data()["probabilities"])
        worst = max(worst, float(np.abs(ours - probs).max()))
    print(f"6 random 5-qubit noisy circuits: largest probability difference with Aer {worst:.2e}")
    assert worst < 1e-9
    import cirq

    a, b = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(a), cirq.CNOT(a, b), cirq.amplitude_damp(0.3).on(a), cirq.phase_damp(0.2).on(b), cirq.rx(0.7).on(a))
    rho = cirq.DensityMatrixSimulator().simulate(circuit).final_density_matrix
    print("Cirq amplitude + phase damping: probabilities", np.round(np.real(np.diag(rho)), 6), "(tests/test_noise_sim.mojo checks the analytic formulas)")


def section_stim() -> None:
    header("from_stim against Stim")
    import stim

    # every single-qubit and two-qubit gate name, as a unitary up to phase
    gates1 = ["I", "X", "Y", "Z", "H", "H_XZ", "H_YZ", "H_XY", "S", "S_DAG", "SQRT_X", "SQRT_X_DAG", "SQRT_Y", "SQRT_Y_DAG", "SQRT_Z", "SQRT_Z_DAG", "C_XYZ", "C_ZYX"]
    gates2 = ["CX", "CY", "CZ", "CNOT", "ZCX", "ZCY", "ZCZ", "SWAP", "ISWAP", "ISWAP_DAG", "XCX", "XCY", "XCZ", "YCX", "YCY", "YCZ", "SQRT_XX", "SQRT_XX_DAG", "SQRT_YY", "SQRT_YY_DAG", "SQRT_ZZ", "SQRT_ZZ_DAG"]
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator

    def same(name, text, n):
        got = q.from_stim(text + (" 0 1" if n == 2 else " 0"))
        mat = stim.Tableau.from_named_gate(name).to_unitary_matrix(endian="little")
        qc = to_qiskit_any(got)
        return Operator(qc).equiv(Operator(mat))

    def to_qiskit_any(c):
        from qiskit import qasm3

        text = c.to_qasm3()
        return qasm3.loads(text)

    bad = [g for g in gates1 if not same(g, g, 1)] + [g for g in gates2 if not same(g, g, 2)]
    print(f"{len(gates1) + len(gates2) - len(bad)} of {len(gates1) + len(gates2)} Stim gates equal their unitary up to a phase; different: {bad}")
    worst = 0.0
    count = 0
    for task in ("surface_code:rotated_memory_z", "surface_code:rotated_memory_x", "surface_code:unrotated_memory_z", "repetition_code:memory", "color_code:memory_xyz"):
        for d in (3, 5):
            kw = dict(distance=d, rounds=d, after_clifford_depolarization=0.002, before_round_data_depolarization=0.001, before_measure_flip_probability=0.003, after_reset_flip_probability=0.002)
            c = stim.Circuit.generated(task, **kw)
            dem_ref = c.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True)
            got = q.dem_text(q.from_stim(str(c)))
            dem_got = stim.DetectorErrorModel(got)
            # same mechanisms up to the DEM's own order
            def table(model):
                out = {}
                for inst in model.flattened():
                    if inst.type != "error":
                        continue
                    symptom = set()
                    for t in inst.targets_copy():
                        if not t.is_separator():
                            symptom ^= {str(t)}
                    key = tuple(sorted(symptom))
                    p = inst.args_copy()[0]
                    out[key] = out.get(key, 0.0) * (1 - p) + p * (1 - out.get(key, 0.0))
                return out

            a = table(dem_ref)
            b = table(dem_got)
            only = set(a) ^ set(b)
            assert not only, sorted(only)[:4]
            for k in a:
                worst = max(worst, abs(a[k] - b[k]) / a[k])
            count += len(a)
            assert stim.Circuit(q.to_stim(q.from_stim(str(c)))).num_measurements == c.num_measurements
    print(f"generated circuits through from_stim: {count} DEM mechanisms compared, worst relative probability difference {worst:.2e}")
    pc = stim.Circuit("R 0 1 2\nPAULI_CHANNEL_2(0.01, 0, 0, 0.02, 0, 0, 0.03, 0, 0.01, 0, 0, 0, 0, 0, 0.04) 0 1\nH 2\nCZ 0 2\nE(0.05) X0 Z2\nH 2\nM 0 1 2\nDETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-1]")
    ref = pc.detector_error_model(approximate_disjoint_errors=True)
    print("PAULI_CHANNEL_2 + E DEM, Stim :", str(ref).replace("\n", "; "))
    print("PAULI_CHANNEL_2 + E DEM, QUMO :", q.dem_text(q.from_stim(str(pc))).strip().replace("\n", "; "))


def section_dem() -> None:
    header("parse_dem and sample_dem against Stim")
    import stim

    c = stim.Circuit.generated("surface_code:rotated_memory_z", distance=5, rounds=5, after_clifford_depolarization=0.004, before_measure_flip_probability=0.002)
    dem = c.detector_error_model(decompose_errors=True)
    nd, nk, ne, text = q.parse_dem(str(dem))
    back = stim.DetectorErrorModel(text)
    assert back.num_detectors == dem.num_detectors and back.num_errors == dem.num_errors
    print(f"parse_dem: {nd} detectors, {nk} observables, {ne} errors; Stim reads the text back with the same counts")
    shots = 1_000_000
    ours = np.array(q.sample_dem(str(dem), shots, 3))
    d, l, _ = dem.compile_sampler(seed=5).sample(shots)
    ref = np.concatenate([d.mean(axis=0), l.mean(axis=0)])
    sigma = np.sqrt(np.maximum(ref * (1 - ref), 1e-9) * 2 / shots)
    z = np.abs(ours - ref) / sigma
    print(f"sample_dem marginals over {len(ref)} detectors and observables, {shots} shots: largest |z| {z.max():.2f}, largest difference {np.abs(ours - ref).max():.5f}")


def section_lint() -> None:
    header("missing_detectors against Stim 1.16")
    import stim

    for d in (3, 5):
        c = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=d)
        cq = q.from_stim(str(c))
        a = len(c.missing_detectors())
        b = len(q.missing_detectors(cq, False))
        ca = c.count_determined_measurements()
        cb = q.count_determined_measurements(cq, False)
        print(f"rotated memory z, d={d}: missing {a} (Stim) {b} (QUMO); determined {ca} (Stim) {cb} (QUMO)")
        text = str(c.flattened())
        lines = text.splitlines()
        drop = [i for i, x in enumerate(lines) if x.startswith("DETECTOR")][len(lines) // 200]
        cut = stim.Circuit("\n".join(lines[:drop] + lines[drop + 1:]))
        cc = q.from_stim(str(cut))
        print(f"  one detector deleted: missing {len(cut.missing_detectors())} (Stim) {len(q.missing_detectors(cc, False))} (QUMO)")
        print(f"  unknown_input: determined {c.count_determined_measurements(unknown_input=True)} (Stim) {q.count_determined_measurements(cq, True)} (QUMO)")


def section_correlated() -> None:
    header("mwpm_correlated against PyMatching 2.4 on identical Stim syndromes")
    import pymatching
    import stim

    print("| d | p | shots | plain Py | corr Py | plain QUMO | corr QUMO | plain/corr QUMO | plain/corr Py |")
    print("|---|---|---|---|---|---|---|---|---|")
    for d, p, shots in ((3, 0.003, 600000), (5, 0.003, 600000), (5, 0.005, 300000), (7, 0.005, 300000)):
        c = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=d, after_clifford_depolarization=p, before_round_data_depolarization=p, before_measure_flip_probability=p, after_reset_flip_probability=p)
        dem = c.detector_error_model(decompose_errors=True)
        dets, obs = c.compile_detector_sampler(seed=11).sample(shots, separate_observables=True)
        m = pymatching.Matching.from_detector_error_model(dem)
        pa = np.count_nonzero(m.decode_batch(dets)[:, 0] != obs[:, 0])
        mc = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
        pb = np.count_nonzero(mc.decode_batch(dets, enable_correlations=True)[:, 0] != obs[:, 0])
        fired = [np.flatnonzero(row).tolist() for row in dets]
        qa = np.array(q.decode_dem(str(dem), fired, "mwpm"), dtype=np.uint64) & 1
        t0 = time.perf_counter()
        qb = np.array(q.decode_dem(str(dem), fired, "mwpm_correlated"), dtype=np.uint64) & 1
        t = time.perf_counter() - t0
        ea = np.count_nonzero(qa != obs[:, 0].astype(np.uint64))
        eb = np.count_nonzero(qb != obs[:, 0].astype(np.uint64))
        print(f"| {d} | {p} | {shots} | {pa} | {pb} | {ea} | {eb} | {ea / max(eb, 1):.3f} | {pa / max(pb, 1):.3f} |", flush=True)


def section_m3() -> None:
    header("M3 against mthree 3.0")
    py = os.environ.get("QUMO_REF_VENV")
    if not py:
        print("skipped: set QUMO_REF_VENV to a Python with mthree installed")
        return
    rng = np.random.default_rng(4)
    for n, strings in ((12, 200), (30, 400)):
        p01 = rng.uniform(0.002, 0.02, n)
        p10 = rng.uniform(0.01, 0.04, n)
        counts = {}
        for _ in range(4000):
            truth = int(rng.integers(0, strings)) * 2654435761 % (1 << n)
            s = "".join("1" if ((truth >> (n - 1 - i)) & 1) ^ (rng.uniform() < (p10[n - 1 - i] if (truth >> (n - 1 - i)) & 1 else p01[n - 1 - i])) else "0" for i in range(n))
            counts[s] = counts.get(s, 0) + 1
        ours = q.mitigate_m3(counts, list(p01), list(p10), 3)
        script = """
import json, sys, numpy as np, mthree
d = json.load(open(sys.argv[1]))
n = d["n"]
m = mthree.M3Mitigation()
m.single_qubit_cals = [np.array([[1 - d["p01"][i], d["p10"][i]], [d["p01"][i], 1 - d["p10"][i]]], dtype=np.float32) for i in range(n)]
m.faulty_qubits = []
m.num_qubits = n
m.cal_shots = 100000
q = m.apply_correction(d["counts"], list(range(n)), distance=3, method="direct" if len(d["counts"]) < 2000 else "iterative", return_mitigation_overhead=False)
print(json.dumps({k: float(v) for k, v in q.items()}))
"""
        with tempfile.TemporaryDirectory() as td:
            data = os.path.join(td, "in.json")
            json.dump({"n": n, "p01": list(map(float, p01)), "p10": list(map(float, p10)), "counts": counts}, open(data, "w"))
            run = os.path.join(td, "run.py")
            open(run, "w").write(script)
            out = subprocess.run([py, "-I", run, data], capture_output=True, text=True, check=False)
        if out.returncode != 0:
            print("mthree failed:", out.stderr.strip().splitlines()[-1:])
            continue
        theirs = json.loads(out.stdout.strip().splitlines()[-1])
        keys = list(theirs)
        a = np.array([ours[k] for k in keys])
        b = np.array([theirs[k] for k in keys])
        print(f"n = {n}: {len(keys)} strings, largest quasi-probability difference {np.abs(a - b).max():.2e}")


def section_ptsbe() -> None:
    header("sample_ptsbe against per-shot trajectories, 16-qubit layered circuit, 20000 shots")

    def text(n, p):
        lines = ['OPENQASM 3.0; include "stdgates.inc";', f"qubit[{n}] q;", f"bit[{n}] c;"]
        for d in range(6):
            for i in range(n):
                lines.append(f"ry({0.1 + 0.01 * (i + d)}) q[{i}];")
                lines.append(f"#pragma qumo depolarize1({p}) q[{i}];")
            for i in range(d % 2, n - 1, 2):
                lines.append(f"cx q[{i}], q[{i + 1}];")
                lines.append(f"#pragma qumo depolarize2({p}) q[{i}], q[{i + 1}];")
        lines.append("c = measure q;")
        return "\n".join(lines)

    print("| p | distinct patterns | error-free shots | trajectories s | ptsbe s | speedup |")
    print("|---|---|---|---|---|---|")
    for p in (1e-4, 1e-3, 1e-2):
        c = q.load_qasm(text(16, p))
        unique, sites, clean = q.plan_trajectories(c, 20000, 3)
        t0 = time.perf_counter()
        q.sample_ptsbe(c, 20000, 3)
        tb = time.perf_counter() - t0
        t0 = time.perf_counter()
        q.sample(c, 2000, 3)
        ta = (time.perf_counter() - t0) * 10
        print(f"| {p} | {unique} | {clean} | {ta:.1f} (scaled from 2000 shots) | {tb:.2f} | {ta / tb:.0f}x |", flush=True)


def section_branching() -> None:
    header("Stabilizer branching: throughput against the width limit of the state vector")
    print("| qubits | T gates | shots | seconds |")
    print("|---|---|---|---|")
    for n, t in ((30, 4), (60, 8), (120, 10), (400, 12)):
        lines = ['OPENQASM 3.0; include "stdgates.inc";', f"qubit[{n}] q;", f"bit[{min(n, 32)}] c;", "h q[0];"]
        for i in range(n - 1):
            lines.append(f"cx q[{i}], q[{i + 1}];")
        for i in range(t):
            lines.append(f"t q[{(i * n) // t}];")
            lines.append(f"h q[{(i * n) // t}];")
        for i in range(min(n, 32)):
            lines.append(f"c[{i}] = measure q[{i}];")
        c = q.load_qasm("\n".join(lines))
        t0 = time.perf_counter()
        q.sample_branching(c, 200, 1)
        print(f"| {n} | {t} | 200 | {time.perf_counter() - t0:.2f} |", flush=True)


SECTIONS = {
    "mcx": section_mcx, "wrap": section_wrap, "layout": section_layout, "pbc": section_pbc, "tcount": section_tcount,
    "density": section_density, "stim": section_stim, "dem": section_dem, "lint": section_lint,
    "correlated": section_correlated, "m3": section_m3, "ptsbe": section_ptsbe, "branching": section_branching,
}

if __name__ == "__main__":
    for name in sys.argv[1:] or list(SECTIONS):
        SECTIONS[name]()
