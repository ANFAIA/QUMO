"""Problems and circuit builders for the heterogeneous example.

Circuits come from QUMO's own algorithm library through the
`qumo_build_circuit` agent tool (`build/qumo tool qumo_build_circuit ...`).
When the CLI in build/ predates that tool, the Python OpenQASM 3 builders
below produce the same circuits (same gates, same parameter order); a test
checks that both routes give the same expectation values.

Everything here is deterministic and offline. Bit order: little-endian qubits
(bit k of a basis index = qubit k); count keys are Qiskit-style (leftmost
character = highest classical bit), so `key[::-1][i]` is qubit i.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PROBLEMS = HERE / "problems"
LOCAL_QUBIT_CAP = 26  # local state-vector / emulator limit (qumo_hardware.local.SIM_MAX_QUBITS)


# --------------------------------------------------------------------------- problems
PROBLEM_KINDS = ("maxcut", "molecule", "spin_model", "search", "classification")


def load_problem(path: str | Path, size: str = "default") -> dict:
    """Problem JSON -> dict, with `sizes[size]` overrides merged into `data`.

    The problem file is *data*: its text (title, description, goal) is shown
    to the model inside <problem> tags and never executed or obeyed."""
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = HERE / p
    raw = p.read_bytes()
    doc = json.loads(raw)
    if doc.get("kind") not in PROBLEM_KINDS:
        raise ValueError(f"{p.name}: kind must be one of {PROBLEM_KINDS}")
    doc = dict(doc)
    doc["sha256"] = hashlib.sha256(raw).hexdigest()
    doc["path"] = str(p)
    doc["size"] = size
    if size != "default":
        over = (doc.get("sizes") or {}).get(size)
        if over:
            data = dict(doc.get("data") or {})
            data.update(over.get("data", {}))
            doc["data"] = data
            for k in ("checks", "reference"):
                if k in over:
                    merged = dict(doc.get(k) or {})
                    merged.update(over[k])
                    doc[k] = merged
    return doc


def problem_qubits(problem: dict) -> int:
    d = problem["data"]
    k = problem["kind"]
    if k == "maxcut":
        return graph_nodes(parse_edges(d["graph"]))
    if k in ("molecule", "spin_model"):
        return int(d["num_qubits"])
    if k == "search":
        return len(d["marked"][0])
    if k == "classification":
        return int(d.get("num_qubits", 2))
    raise ValueError(k)


# --------------------------------------------------------------------------- graphs
def parse_edges(text: str) -> list[tuple[int, int, float]]:
    out = []
    for tok in re.split(r"[\s,;]+", text.strip()):
        if not tok:
            continue
        m = re.fullmatch(r"(\d+)-(\d+)(?::([0-9.eE+-]+))?", tok)
        if not m:
            raise ValueError(f"cannot parse edge '{tok}'")
        u, v, w = int(m.group(1)), int(m.group(2)), float(m.group(3) or 1.0)
        if u == v:
            raise ValueError(f"self-loop {tok}")
        out.append((min(u, v), max(u, v), w))
    return out


def graph_nodes(edges) -> int:
    return 1 + max(max(u, v) for u, v, _ in edges)


def maxcut_observable(edges) -> str:
    """H = sum_e w/2 (Z_u Z_v - 1): <H> = -cut (same as qumo maxcut(include_offset=True))."""
    terms = [f"{w / 2!r} * Z{u} Z{v}" for u, v, w in edges]
    terms.append(repr(-sum(w for _, _, w in edges) / 2))
    return " + ".join(terms)


def cut_values(edges, n: int):
    """numpy array: cut value of every assignment 0..2^n-1 (bit k = node k)."""
    import numpy as np

    idx = np.arange(1 << n, dtype=np.int64)
    cut = np.zeros(1 << n, dtype=np.float64)
    for u, v, w in edges:
        cut += w * (((idx >> u) & 1) ^ ((idx >> v) & 1))
    return cut


def brute_force_maxcut(edges, n: int) -> tuple[float, list[int]]:
    """(max cut, all optimal assignments as integers) by enumeration (numpy, n <= 26)."""
    import numpy as np

    if n > LOCAL_QUBIT_CAP:
        raise ValueError("brute force limited to 26 nodes")
    cut = cut_values(edges, n)
    best = float(cut.max())
    return best, [int(i) for i in np.flatnonzero(np.abs(cut - best) < 1e-9)]


def cut_of_key(key: str, edges) -> float:
    bits = key[::-1]
    return sum(w for u, v, w in edges if bits[u] != bits[v])


# --------------------------------------------------------------------------- CLI tool route
def cli_path() -> Path | None:
    env = os.environ.get("QUMO_CLI")
    cands = [Path(env)] if env else []
    cands += [REPO / "build" / "qumo", REPO / "build" / "gpu" / "qumo"]
    for c in cands:
        if c.is_file() and os.access(c, os.X_OK):
            return c
    return None


def cli_tool(name: str, args: dict, timeout: float = 120.0) -> dict | None:
    """`build/qumo tool <name> '<json>'` -> parsed JSON, or None when the CLI or
    the tool is missing (older build). Other tool errors raise RuntimeError."""
    exe = cli_path()
    if exe is None:
        return None
    try:
        r = subprocess.run([str(exe), "tool", name, json.dumps(args, sort_keys=True)], capture_output=True,
                           text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    try:
        doc = json.loads(r.stdout)
    except ValueError:
        return None
    if isinstance(doc, dict) and "error" in doc and isinstance(doc["error"], dict):
        if doc["error"].get("code") == "QM-E912":  # unknown tool: this CLI predates it
            return None
        raise RuntimeError(f"{name}: {doc['error'].get('code')}: {doc['error'].get('message')}")
    return doc


def cli_describe_capabilities() -> dict | None:
    exe = cli_path()
    if exe is None:
        return None
    try:
        r = subprocess.run([str(exe), "describe", "capabilities", "--json"], capture_output=True, text=True,
                           timeout=60)
        return json.loads(r.stdout)
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None


# --------------------------------------------------------------------------- built circuits
@dataclass
class Built:
    """A circuit ready for the stages: unmeasured QASM (expectations), measured
    QASM (sampling), parameter names in order, observable and provenance."""

    algorithm: str
    num_qubits: int
    qasm: str
    qasm_measured: str
    parameters: list[str]
    observable: str = ""
    source: str = "python-builder"  # "qumo_build_circuit" | "python-builder"
    notes: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {"algorithm": self.algorithm, "num_qubits": self.num_qubits, "num_parameters": len(self.parameters),
                "parameters": self.parameters, "source": self.source, "observable_terms":
                    (self.observable.count(" + ") + 1) if self.observable else 0, **self.stats}


def _header(params: list[str]) -> list[str]:
    return ['OPENQASM 3.0;', 'include "stdgates.inc";'] + [f"input float[64] {p};" for p in params]


def _measured(qasm: str, n_meas: int, n_total: int) -> str:
    lines = qasm.rstrip().splitlines()
    out = []
    for ln in lines:
        out.append(ln)
        if ln.startswith("qubit["):
            out.append(f"bit[{n_meas}] c;")
    if n_meas == n_total:
        out.append("c = measure q;")
    else:
        out += [f"c[{i}] = measure q[{i}];" for i in range(n_meas)]
    return "\n".join(out) + "\n"


def _from_tool(kind: str, args: dict, algorithm: str, measure_qubits: int | None = None) -> Built | None:
    try:
        plain = cli_tool("qumo_build_circuit", {"kind": kind, "args": args, "measure": False})
        if plain is None:
            return None
        meas = cli_tool("qumo_build_circuit", {"kind": kind, "args": args, "measure": True})
    except RuntimeError:
        raise
    if meas is None:
        return None
    stats = {"gate_counts": plain.get("gate_counts", {}), "two_qubit_gates": plain.get("two_qubit_gates"),
             "depth": plain.get("depth")}
    return Built(algorithm, int(plain["num_qubits"]), plain["qasm"], meas["qasm"], list(plain["parameters"]),
                 plain.get("observable", ""), "qumo_build_circuit", list(plain.get("notes", [])), stats)


# ---- QAOA MaxCut ----------------------------------------------------------------------
def qaoa_maxcut_python(edges, n: int, p: int) -> Built:
    params = [f"gamma_{l}" for l in range(p)] + [f"beta_{l}" for l in range(p)]
    L = _header(params) + [f"qubit[{n}] q;"] + [f"h q[{i}];" for i in range(n)]
    for l in range(p):
        L += [f"rzz({w!r} * gamma_{l}) q[{u}], q[{v}];" for u, v, w in edges]
        L += [f"rx(2.0 * beta_{l}) q[{i}];" for i in range(n)]
    qasm = "\n".join(L) + "\n"
    return Built("qaoa_maxcut", n, qasm, _measured(qasm, n, n), params, maxcut_observable(edges),
                 notes=["python builder (same gates as qumo qaoa_maxcut_ansatz)"])


def build_qaoa(edges, n: int, p: int, prefer_tool: bool = True) -> Built:
    if prefer_tool:
        g = " ".join(f"{u}-{v}" + (f":{w!r}" if w != 1.0 else "") for u, v, w in edges)
        b = _from_tool("qaoa_maxcut", {"graph": g, "p": p}, "qaoa_maxcut")
        if b is not None:
            return b
    return qaoa_maxcut_python(edges, n, p)


# ---- Pauli rotations (UCCSD, Hamiltonian evolution) -------------------------------------
def _pauli_rotation(L: list[str], xmask: int, zmask: int, angle: str) -> None:
    """exp(-i angle/2 P), mirroring qumo.algorithms.library.append_pauli_rotation."""
    support = [q for q in range(64) if ((xmask | zmask) >> q) & 1]
    if not support:
        return
    for q in support:
        hx, hz = (xmask >> q) & 1, (zmask >> q) & 1
        if hx and hz:
            L += [f"sdg q[{q}];", f"h q[{q}];"]
        elif hx:
            L.append(f"h q[{q}];")
    if len(support) == 1:
        L.append(f"rz({angle}) q[{support[0]}];")
    elif len(support) == 2:
        L.append(f"rzz({angle}) q[{support[0]}], q[{support[1]}];")
    else:
        for i in range(len(support) - 1):
            L.append(f"cx q[{support[i]}], q[{support[i + 1]}];")
        L.append(f"rz({angle}) q[{support[-1]}];")
        for i in range(len(support) - 2, -1, -1):
            L.append(f"cx q[{support[i]}], q[{support[i + 1]}];")
    for q in support:
        hx, hz = (xmask >> q) & 1, (zmask >> q) & 1
        if hx and hz:
            L += [f"h q[{q}];", f"s q[{q}];"]
        elif hx:
            L.append(f"h q[{q}];")


def _zstring(lo: int, hi: int) -> int:
    m = 0
    for q in range(lo + 1, hi):
        m |= 1 << q
    return m


def uccsd_python(num_qubits: int, num_electrons: int) -> Built:
    """Same excitations, order and decomposition as qumo uccsd_ansatz (JW, interleaved spins)."""
    n, ne = num_qubits, num_electrons
    body: list[str] = [f"x q[{q}];" for q in range(ne)]
    params: list[str] = []
    for i in range(ne):
        for a in range(ne, n):
            if i % 2 == a % 2:
                name = f"s_{i}_{a}"
                params.append(name)
                bp, bq, zs = 1 << i, 1 << a, _zstring(i, a)
                _pauli_rotation(body, bp | bq, zs | bq, name)
                _pauli_rotation(body, bp | bq, zs | bp, f"-{name}")
    pats = [0b0001, 0b0010, 0b0100, 0b0111, 0b1000, 0b1011, 0b1101, 0b1110]
    signs = [-1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0]
    for i in range(ne):
        for j in range(i + 1, ne):
            for a in range(ne, n):
                for b in range(a + 1, n):
                    up_occ = (i % 2 == 0) + (j % 2 == 0)
                    up_vir = (a % 2 == 0) + (b % 2 == 0)
                    if up_occ != up_vir:
                        continue
                    name = f"d_{i}_{j}_{a}_{b}"
                    params.append(name)
                    zs = _zstring(i, j) | _zstring(a, b)
                    qs = [i, j, a, b]
                    for k in range(8):
                        x, z = 0, zs
                        for jj in range(4):
                            bit = 1 << qs[jj]
                            x |= bit
                            if (pats[k] >> (3 - jj)) & 1:
                                z |= bit
                        _pauli_rotation(body, x, z, f"{signs[k] / 4!r} * {name}")
    qasm = "\n".join(_header(params) + [f"qubit[{n}] q;"] + body) + "\n"
    return Built("vqe_uccsd", n, qasm, _measured(qasm, n, n), params,
                 notes=["python builder (same excitations as qumo uccsd_ansatz; parameter names sanitised)"])


def build_uccsd(problem: dict, prefer_tool: bool = True) -> Built:
    d = problem["data"]
    n, ne = int(d["num_qubits"]), int(d["num_electrons"])
    b = None
    if prefer_tool:
        kind, args = ("h2_ansatz", {}) if d.get("molecule") == "H2" and n == 4 else (
            "uccsd", {"num_qubits": n, "num_electrons": ne, "hamiltonian": d["hamiltonian"]})
        b = _from_tool(kind, args, "vqe_uccsd")
    if b is None:
        b = uccsd_python(n, ne)
    b.algorithm = "vqe_uccsd"
    b.observable = d["hamiltonian"]  # the problem's Hamiltonian is authoritative
    return b


# ---- hardware-efficient ansatz -----------------------------------------------------------
def hea_python(n: int, layers: int, entanglement: str = "linear", rotations: str = "ry,rz",
               init_x: list[int] | None = None) -> Built:
    rots = [r.strip() for r in rotations.split(",") if r.strip()]
    params: list[str] = []
    body: list[str] = [f"x q[{q}];" for q in (init_x or [])]
    k = 0

    def layer():
        nonlocal k
        for q in range(n):
            for r in rots:
                name = f"theta_{k}"
                params.append(name)
                body.append(f"{r}({name}) q[{q}];")
                k += 1

    for _ in range(layers):
        layer()
        if entanglement in ("linear", "circular"):
            body.extend(f"cx q[{i}], q[{i + 1}];" for i in range(n - 1))
            if entanglement == "circular" and n > 2:
                body.append(f"cx q[{n - 1}], q[0];")
        elif entanglement == "full":
            body.extend(f"cx q[{i}], q[{j}];" for i in range(n) for j in range(i + 1, n))
    layer()
    qasm = "\n".join(_header(params) + [f"qubit[{n}] q;"] + body) + "\n"
    return Built("vqe_hea", n, qasm, _measured(qasm, n, n), params,
                 notes=[f"hardware-efficient ansatz, {layers} layer(s), {entanglement} CX entangler, rotations {rotations}"])


def build_hea(problem: dict, layers: int, prefer_tool: bool = True) -> Built:
    d = problem["data"]
    n = int(d["num_qubits"])
    init_x = list(d.get("initial_x", []))
    b = None
    if prefer_tool and not init_x:
        b = _from_tool("hea", {"n": n, "layers": layers, "entanglement": "linear", "rotations": "ry,rz",
                               "hamiltonian": d["hamiltonian"]}, "vqe_hea")
    if b is None:
        b = hea_python(n, layers, "linear", "ry,rz", init_x)
    b.algorithm = "vqe_hea"
    b.observable = d["hamiltonian"]
    return b


# ---- Grover --------------------------------------------------------------------------------
def _mcz(L: list[str], qs: list[int], anc: list[int]) -> None:
    """Multi-controlled Z, mirroring qumo append_mcz (CCX V-chain, CCZ = H.CCX.H)."""
    k = len(qs)

    def ccz(a, b, c):
        L.extend([f"h q[{c}];", f"ccx q[{a}], q[{b}], q[{c}];", f"h q[{c}];"])

    if k == 1:
        L.append(f"z q[{qs[0]}];")
        return
    if k == 2:
        L.append(f"cz q[{qs[0]}], q[{qs[1]}];")
        return
    if k == 3:
        ccz(qs[0], qs[1], qs[2])
        return
    L.append(f"ccx q[{qs[0]}], q[{qs[1]}], q[{anc[0]}];")
    for i in range(2, k - 2):
        L.append(f"ccx q[{qs[i]}], q[{anc[i - 2]}], q[{anc[i - 1]}];")
    ccz(anc[k - 4], qs[k - 2], qs[k - 1])
    for i in range(k - 3, 1, -1):
        L.append(f"ccx q[{qs[i]}], q[{anc[i - 2]}], q[{anc[i - 1]}];")
    L.append(f"ccx q[{qs[0]}], q[{qs[1]}], q[{anc[0]}];")


def grover_optimal_iterations(n: int, m: int) -> int:
    if m <= 0 or m >= (1 << n):
        return 0
    theta = math.asin(math.sqrt(m / (1 << n)))
    return max(int(math.floor(math.pi / (4 * theta))), 0)


def grover_success_probability(n: int, m: int, k: int) -> float:
    theta = math.asin(math.sqrt(min(m / (1 << n), 1.0)))
    return math.sin((2 * k + 1) * theta) ** 2


def grover_python(marked: list[str], iterations: int = -1) -> Built:
    n = len(marked[0])
    nanc = max(0, n - 3)
    total = n + nanc
    qs, anc = list(range(n)), list(range(n, total))
    k = iterations if iterations >= 0 else grover_optimal_iterations(n, len(marked))
    body = [f"h q[{q}];" for q in qs]
    for _ in range(k):
        for key in marked:
            m = int(key, 2)
            flips = [f"x q[{q}];" for q in qs if not (m >> q) & 1]
            body += flips
            _mcz(body, qs, anc)
            body += flips
        body += [f"h q[{q}];" for q in qs] + [f"x q[{q}];" for q in qs]
        _mcz(body, qs, anc)
        body += [f"x q[{q}];" for q in qs] + [f"h q[{q}];" for q in qs]
    qasm = "\n".join(_header([]) + [f"qubit[{total}] q;"] + body) + "\n"
    return Built("grover_search", total, qasm, _measured(qasm, n, total), [],
                 notes=[f"{n} search qubits + {nanc} ancilla(s); {k} iteration(s)"], extra={"iterations": k,
                                                                                              "search_qubits": n})


def build_grover(problem: dict, prefer_tool: bool = True) -> Built:
    marked = list(problem["data"]["marked"])
    n = len(marked[0])
    k = grover_optimal_iterations(n, len(marked))
    b = _from_tool("grover", {"marked": marked}, "grover_search") if prefer_tool else None
    if b is None:
        b = grover_python(marked)
    b.extra.update({"iterations": k, "search_qubits": n})
    return b


# ---- classification: angle-embedding kernel and data re-uploading classifier ----------------
def angle_embedding_lines(xnames: list[str], pnames: list[str], n: int, reps: int) -> list[str]:
    """ZZ-feature-map style embedding, repeated `reps` times: H, RZ(2 x_q), then
    RZZ(2 p_q) on (q, q+1) with the pair feature p_q = (pi - x_q)(pi - x_{q+1}).

    Parameters are affine in QUMO, so the pair products are passed as their own
    parameters (`kernel_feature_values` computes them)."""
    L = []
    for _ in range(reps):
        L += [f"h q[{q}];" for q in range(n)]
        L += [f"rz(2.0 * {xnames[q]}) q[{q}];" for q in range(n)]
        L += [f"rzz(2.0 * {pnames[q]}) q[{q}], q[{q + 1}];" for q in range(n - 1)]
    return L


def kernel_feature_values(x: list[float]) -> list[float]:
    """[x_0..x_{n-1}, p_0..p_{n-2}] for angle_embedding_lines."""
    n = len(x)
    return list(map(float, x)) + [float((math.pi - x[q]) * (math.pi - x[q + 1])) for q in range(n - 1)]


def kernel_circuit(n: int, reps: int) -> Built:
    """Compute-uncompute kernel circuit U(a) U(b)^dagger: P(0^n) = |<phi(b)|phi(a)>|^2.

    Parameters: a features, a pair features, then the same for b. U(b)^dagger is
    written out gate by gate (reverse order, negated angles; H is self-inverse)."""
    a, ap = [f"a_{i}" for i in range(n)], [f"ap_{i}" for i in range(n - 1)]
    b, bp = [f"b_{i}" for i in range(n)], [f"bp_{i}" for i in range(n - 1)]
    fwd = angle_embedding_lines(a, ap, n, reps)
    inv_b = []
    for line in reversed(angle_embedding_lines(b, bp, n, reps)):
        m = re.match(r"(\w+)\((.*)\) (.*);", line)
        inv_b.append(f"{m.group(1)}(-{m.group(2)}) {m.group(3)};" if m else line)
    qasm = "\n".join(_header(a + ap + b + bp) + [f"qubit[{n}] q;"] + fwd + inv_b) + "\n"
    proj = " + ".join([f"{1 / (1 << n)!r}"] + [
        f"{1 / (1 << n)!r} * " + " ".join(f"Z{q}" for q in range(n) if (s >> q) & 1) for s in range(1, 1 << n)])
    return Built("quantum_kernel_svm", n, qasm, _measured(qasm, n, n), a + ap + b + bp, proj,
                 notes=["kernel(a, b) = <|0..0><0..0|> (observable written as a sum of Z strings)",
                        "embedding: H, RZ(2x), RZZ(2(pi-x_q)(pi-x_q+1)) (ZZ feature map), compute-uncompute"])


def reuploading_circuit(n: int, layers: int, num_features: int) -> Built:
    """Data re-uploading classifier: per layer, features via RY(x) then trainable RY/RZ and a CZ chain.

    Parameters: weights w_0.. first, then features x_0.. (affine parameters)."""
    w = []
    body = []
    x = [f"x_{i}" for i in range(num_features)]
    k = 0
    for _ in range(layers):
        for q in range(n):
            body.append(f"ry({x[q % num_features]}) q[{q}];")
            for r in ("ry", "rz"):
                name = f"w_{k}"
                w.append(name)
                body.append(f"{r}({name}) q[{q}];")
                k += 1
        body += [f"cz q[{q}], q[{q + 1}];" for q in range(n - 1)]
    qasm = "\n".join(_header(w + x) + [f"qubit[{n}] q;"] + body) + "\n"
    return Built("vqc_reuploading", n, qasm, _measured(qasm, 1, n), w + x, "Z0",
                 notes=["prediction f(x) = <Z0>; class = sign(f)"], extra={"num_weights": len(w)})


# ---- Hamiltonians ---------------------------------------------------------------------------
def heisenberg_observable(n: int, j: float = 1.0, delta: float = 1.0) -> str:
    terms = []
    for i in range(n - 1):
        terms += [f"{j!r} * X{i} X{i + 1}", f"{j!r} * Y{i} Y{i + 1}", f"{j * delta!r} * Z{i} Z{i + 1}"]
    return " + ".join(terms)


def parse_pauli_terms(obs: str) -> list[tuple[float, dict[int, str]]]:
    """'0.5 * Z0 Z1 + -1.0 * X0 + 0.25' -> [(coeff, {qubit: 'X'|'Y'|'Z'}), ...]."""
    out = []
    for part in re.split(r"\s\+\s", obs.strip()):
        part = part.strip()
        if not part:
            continue
        if "*" in part:
            c, rest = part.split("*", 1)
            coeff = float(c)
        elif re.fullmatch(r"[-+0-9.eE]+", part):
            coeff, rest = float(part), ""
        else:
            coeff, rest = 1.0, part
        ops = {}
        for m in re.finditer(r"([XYZ])(\d+)", rest):
            ops[int(m.group(2))] = m.group(1)
        out.append((coeff, ops))
    return out


def qwc_groups(terms: list[tuple[float, dict[int, str]]]) -> list[list[int]]:
    """Greedy qubit-wise-commuting groups (same order as qumo PauliSum.qwc_groups)."""
    groups: list[list[int]] = []
    for i, (_, ops) in enumerate(terms):
        if not ops:
            continue
        placed = False
        for g in groups:
            if all(all(terms[j][1].get(q, p) == p for q, p in ops.items()) for j in g):
                g.append(i)
                placed = True
                break
        if not placed:
            groups.append([i])
    return groups


def basis_rotation_lines(bases: dict[int, str]) -> list[str]:
    """Rotate each measured qubit into the Z basis (X: H, Y: Sdg H)."""
    L = []
    for q, p in sorted(bases.items()):
        if p == "X":
            L.append(f"h q[{q}];")
        elif p == "Y":
            L += [f"sdg q[{q}];", f"h q[{q}];"]
    return L
