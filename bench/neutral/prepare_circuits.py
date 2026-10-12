"""Build the shared input set: one normalised OpenQASM 2 file per circuit.

Sources: Benchpress' copies of QASMBench (small/medium) and Feynman circuits
(community benchmark sets). Normalisation is done once, identically for all
SDKs: drop measurements/barriers/resets, reject classically-controlled
circuits, unroll custom gate definitions to a common qelib1 subset that every
parser supports. Every SDK then parses the same text.

Usage: python bench/neutral/prepare_circuits.py <benchpress_qasm_dir> <out_dir> [max_qubits]
"""
import os
import sys

from qiskit import QuantumCircuit, qasm2, transpile

COMMON_BASIS = ["x", "y", "z", "h", "s", "sdg", "t", "tdg", "rx", "ry", "rz",
                "cx", "cz", "swap", "ccx", "cswap", "cp", "crz"]


def normalise(path):
    qc = qasm2.load(path, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    for inst in qc.data:
        if getattr(inst.operation, "condition", None) is not None or inst.operation.name in ("if_else", "while_loop"):
            return None, "classically controlled"
    out = QuantumCircuit(qc.num_qubits)
    for inst in qc.data:
        if inst.operation.name in ("measure", "barrier", "reset", "delay"):
            continue
        out.append(inst.operation, [out.qubits[qc.find_bit(q).index] for q in inst.qubits])
    out = transpile(out, basis_gates=COMMON_BASIS, optimization_level=0)
    # drop idle qubits? no: keep width identical to the source
    return out, None


def main():
    src, dst = sys.argv[1], sys.argv[2]
    max_q = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    os.makedirs(dst, exist_ok=True)
    picked = 0
    for sub in ("qasmbench-small", "qasmbench-medium", "feynman"):
        base = os.path.join(src, sub)
        for root, _, files in os.walk(base):
            for f in sorted(files):
                if not f.endswith(".qasm") or "transpiled" in f:
                    continue
                path = os.path.join(root, f)
                try:
                    qc, why = normalise(path)
                except Exception as e:  # unparsable source
                    print(f"skip {sub}/{f}: {type(e).__name__}")
                    continue
                if qc is None:
                    print(f"skip {sub}/{f}: {why}")
                    continue
                if qc.num_qubits > max_q or qc.size() == 0:
                    continue
                name = f"{sub.replace('qasmbench-', 'qb_')}__{f[:-5]}".replace("^", "p")
                with open(os.path.join(dst, name + ".qasm"), "w") as fh:
                    fh.write(qasm2.dumps(qc))
                picked += 1
    print(f"wrote {picked} circuits to {dst}")


if __name__ == "__main__":
    main()
