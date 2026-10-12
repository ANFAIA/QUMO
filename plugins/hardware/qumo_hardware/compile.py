"""Compile circuits with QUMO itself for a provider's `TargetSpec`.

Pipeline (identical for every provider, which is what keeps QUMO neutral):

1. `qumo_py.load_qasm` parses OpenQASM 2 or 3 (registers are flattened in
   declaration order: that flat order *is* the user's logical order).
2. If the circuit has no measurement, every logical qubit l is measured into
   clbit l (like `measure_all`), so results always come back in logical order.
3. `qumo_py.transpile(circ, width, edges, basis, level, seed)` does layout,
   routing, optimisation and basis translation; it returns the physical circuit
   plus `initial_layout[l]` / `final_layout[l]` (physical qubit of logical l).
4. The physical circuit is exported as OpenQASM 2. Measurements in it read the
   *physical* qubit that holds the logical qubit at that point and write the
   user's original clbit, so the payload's classical bits are already in the
   user's order. `measure_map` (clbit -> physical qubit) is recorded so that
   providers returning per-physical-qubit data can be remapped
   (`physical_counts_to_clbits`).
"""

from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .base import BIT_ORDER, TWO_QUBIT_GATES, TargetSpec
from .errors import E_COMPILE, E_QUMO_PY, HardwareError

_QUMO: Any = None

# Basis presets, same names and contents as qumo.compiler.target.parse_basis.
BASIS_PRESETS: dict[str, list[str]] = {
    "any": [],
    "generic": ["u", "cx"],
    "sc_cx": ["rz", "sx", "x", "cx"],
    "sc_cz": ["rz", "sx", "x", "cz"],
    "sc_ecr": ["rz", "sx", "x", "ecr"],
    "sc_iswap": ["rz", "rx", "ry", "iswap"],
    "ion_ms": ["rz", "rx", "ry", "rxx"],
    "ion_zz": ["rz", "ry", "rzz"],
    "atom_cz": ["rz", "rx", "ry", "cz"],
    "clifford_t": ["h", "s", "sdg", "t", "tdg", "x", "y", "z", "cx"],
}

NON_GATE_OPS = {"measure", "reset", "barrier", "delay"}


def repo_root() -> Path | None:
    """The QUMO checkout this package lives in (plugins/hardware/qumo_hardware), or None
    when the package was installed elsewhere (e.g. a non-editable pip install)."""
    here = Path(__file__).resolve()
    root = here.parents[3]
    if here.parents[1].name == "hardware" and here.parents[2].name == "plugins" and (root / "qumo").is_dir():
        return root
    return None


def _candidate_paths() -> list[Path]:
    """Files that may hold the QUMO extension, in priority order.

    Security: only an explicit QUMO_PY_PATH (a .so file or a directory holding
    qumo_py.so) and the `build/` directory of the checkout that contains this
    package are considered. The current working directory and the rest of
    sys.path are never searched, so running the CLI inside an untrusted
    directory cannot load a planted qumo_py.so."""
    out: list[Path] = []
    env = os.environ.get("QUMO_PY_PATH")
    if env:
        p = Path(env).expanduser()
        out.append(p if p.suffix in (".so", ".pyd", ".dylib") else p / "qumo_py.so")
    root = repo_root()
    if root is not None:
        out.append(root / "build" / "qumo_py.so")
    return out


def load_qumo_py():
    """Load the QUMO CPython extension from an explicit path (see _candidate_paths). Cached."""
    global _QUMO
    if _QUMO is not None:
        return _QUMO
    if "qumo_py" in sys.modules:  # already loaded by the host application
        _QUMO = sys.modules["qumo_py"]
        return _QUMO
    tried = _candidate_paths()
    path = next((p for p in tried if p.is_file()), None)
    if path is None:
        where = ", ".join(str(p) for p in tried) or "no candidate (package not inside a QUMO checkout)"
        raise HardwareError(E_QUMO_PY, f"qumo_py.so not found (looked at: {where})")
    try:
        import importlib.machinery
        import importlib.util

        loader = importlib.machinery.ExtensionFileLoader("qumo_py", str(path))
        spec = importlib.util.spec_from_file_location("qumo_py", str(path), loader=loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules["qumo_py"] = module
        loader.exec_module(module)
    except Exception as exc:  # broken build, wrong Python ABI, ...
        sys.modules.pop("qumo_py", None)
        raise HardwareError(E_QUMO_PY, f"cannot load {path} ({type(exc).__name__}: {exc})") from None
    _QUMO = module
    return module


def expand_basis(basis: str | Sequence[str]) -> list[str]:
    """'sc_cz' / 'rz,sx,x,cz' / ['rz', ...] -> explicit gate list ([] = any)."""
    if isinstance(basis, str):
        key = basis.strip().lower()
        if key in BASIS_PRESETS:
            return list(BASIS_PRESETS[key])
        parts = [p.strip().lower() for p in re.split(r"[,;+\s]+", key) if p.strip()]
    else:
        parts = [str(p).strip().lower() for p in basis]
    out: list[str] = []
    for p in parts:
        for g in BASIS_PRESETS.get(p, [p]):
            if g not in out:
                out.append(g)
    return out


# --------------------------------------------------------------------- QASM 2 text helpers
_RE_QREG = re.compile(r"^\s*qreg\s+(\w+)\s*\[\s*(\d+)\s*\]\s*;", re.M)
_RE_CREG = re.compile(r"^\s*creg\s+(\w+)\s*\[\s*(\d+)\s*\]\s*;", re.M)
_RE_MEASURE = re.compile(r"measure\s+(\w+)\s*\[\s*(\d+)\s*\]\s*->\s*(\w+)\s*\[\s*(\d+)\s*\]\s*;")
_RE_QREF = re.compile(r"\bq\[(\d+)\]")


def _offsets(regex: re.Pattern, text: str) -> tuple[dict[str, int], int]:
    offs: dict[str, int] = {}
    total = 0
    for m in regex.finditer(text):
        offs[m.group(1)] = total
        total += int(m.group(2))
    return offs, total


def parse_measurements(qasm2: str) -> list[tuple[int, int]]:
    """[(flat_qubit, flat_clbit), ...] in program order."""
    qoff, _ = _offsets(_RE_QREG, qasm2)
    coff, _ = _offsets(_RE_CREG, qasm2)
    out = []
    for m in _RE_MEASURE.finditer(qasm2):
        qr, qi, cr, ci = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
        if qr in qoff and cr in coff:
            out.append((qoff[qr] + qi, coff[cr] + ci))
    return out


def _body_lines(qasm2: str) -> list[str]:
    lines = []
    for raw in qasm2.splitlines():
        line = raw.split("//", 1)[0].strip()
        if not line or line.startswith(("OPENQASM", "include", "qreg", "creg", "gate ", "opaque")):
            continue
        lines.append(line)
    return lines


def _add_measure_all(qasm2: str, n: int) -> str:
    """QUMO-exported QASM 2 without measurements -> measure q[l] -> c[l] for all l."""
    if "if(" in qasm2.replace(" ", ""):
        raise HardwareError(
            E_COMPILE,
            "the circuit has classical conditions but no measurements",
            "add explicit measurements to the circuit",
        )
    lines = [ln for ln in qasm2.splitlines() if not _RE_CREG.match(ln)]
    out = []
    inserted = False
    for ln in lines:
        out.append(ln)
        if not inserted and _RE_QREG.match(ln):
            out.append(f"creg c[{n}];")
            inserted = True
    out.extend(f"measure q[{i}] -> c[{i}];" for i in range(n))
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------- result
@dataclass
class CompiledCircuit:
    """QUMO-compiled physical circuit plus everything needed to read results."""

    qasm: str  # physical circuit, OpenQASM 2 (single qreg q, single creg c)
    source_qasm: str  # logical circuit that was compiled (after measure-all), OpenQASM 2
    num_qubits: int  # physical register width
    num_logical: int
    num_clbits: int
    initial_layout: list[int]
    final_layout: list[int]
    measure_map: dict[int, int]  # clbit -> physical qubit (last write)
    logical_measure_map: dict[int, int]  # clbit -> logical qubit (last write)
    auto_measured: bool
    ops: dict[str, int]
    two_qubit_gates: int
    depth: int
    level: int
    seed: int
    target: TargetSpec
    compile_ms: float
    qumo_version: str
    active_qubits: list[int] = field(default_factory=list)
    layout_check: str = "consistent"
    region: list[int] | None = None  # physical qubits QUMO was restricted to (None = whole device)

    bit_order: str = BIT_ORDER

    def to_dict(self, include_qasm: bool = True) -> dict:
        d = asdict(self)
        d["target"] = self.target.to_dict(compact=True)
        d["measure_map"] = {str(k): v for k, v in self.measure_map.items()}
        d["logical_measure_map"] = {str(k): v for k, v in self.logical_measure_map.items()}
        if not include_qasm:
            d.pop("qasm")
            d.pop("source_qasm")
        return d

    def summary(self) -> dict:
        return {
            "num_logical": self.num_logical,
            "num_physical": self.num_qubits,
            "active_qubits": len(self.active_qubits),
            "num_clbits": self.num_clbits,
            "ops": self.ops,
            "two_qubit_gates": self.two_qubit_gates,
            "depth": self.depth,
            "initial_layout": self.initial_layout,
            "final_layout": self.final_layout,
            "auto_measured": self.auto_measured,
            "region": self.region,
            "level": self.level,
            "seed": self.seed,
            "compile_ms": round(self.compile_ms, 3),
            "qumo_version": self.qumo_version,
        }

    def estimated_success(self) -> float | None:
        """Crude success-probability estimate from the target's median errors:
        prod (1 - e) over 1q gates (rz is virtual and free), 2q gates and measurements."""
        e = self.target.errors
        if not e:
            return None
        one_q = sum(v for k, v in self.ops.items() if k not in ("rz", "measure", "reset", "barrier", "delay")
                    and k not in TWO_QUBIT_GATES)
        p = (1 - e.get("1q", 0.0)) ** one_q
        p *= (1 - e.get("2q", 0.0)) ** self.two_qubit_gates
        p *= (1 - e.get("readout", 0.0)) ** self.ops.get("measure", 0)
        return p

    def result_meta(self) -> dict:
        """What the job store keeps to post-process results later."""
        return {
            "num_clbits": self.num_clbits,
            "measure_map": {str(k): v for k, v in self.measure_map.items()},
            "logical_measure_map": {str(k): v for k, v in self.logical_measure_map.items()},
            "initial_layout": self.initial_layout,
            "final_layout": self.final_layout,
            "num_logical": self.num_logical,
            "num_physical": self.num_qubits,
            "auto_measured": self.auto_measured,
        }


SIMULATION_ONLY = {"depolarize1", "depolarize2", "x_error", "y_error", "z_error", "pauli_channel_1",
                   "detector", "observable", "tick"}
_RE_INPUT = re.compile(r"^\s*input\s+(?:float|angle)(?:\s*\[\s*\d+\s*\])?\s+(\w+)\s*;", re.M)


def _qumo_error(exc: Exception, what: str, hint: str | None = None) -> HardwareError:
    first = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
    msg, _, qumo_hint = first.partition(" Hint:")
    return HardwareError(E_COMPILE, f"{what}: {msg.rstrip('.')}",
                         hint or qumo_hint.strip() or "check the circuit with `build/qumo validate <file>`")


def _load_circuit(source: Any) -> tuple[Any, str]:
    """-> (qumo_py.Circuit, source text or '')."""
    q = load_qumo_py()
    if isinstance(source, str):
        text = source
        if "\n" not in source and source.endswith((".qasm", ".qasm2", ".qasm3")) and os.path.exists(source):
            text = Path(source).read_text()
        try:
            return q.load_qasm(text), text
        except Exception as exc:
            raise _qumo_error(exc, "cannot parse the circuit") from None
    return source, ""  # assume a qumo_py.Circuit


def _bind(circ: Any, text: str, params: Any):
    """Bind symbolic parameters: a list (declaration order) or a {name: value} dict
    (names from OpenQASM 3 `input float/angle` declarations)."""
    if isinstance(params, dict):
        names = _RE_INPUT.findall(text)
        missing = [n for n in names if n not in params]
        if not names or missing or len(names) != int(circ.num_parameters()):
            raise HardwareError(E_COMPILE, f"cannot bind parameters by name (declared {names}, missing {missing})",
                                "pass a list of values in declaration order")
        params = [params[n] for n in names]
    values = [float(v) for v in params]
    if len(values) != int(circ.num_parameters()):
        raise HardwareError(E_COMPILE, f"{len(values)} parameter values given, circuit has {int(circ.num_parameters())}")
    try:
        return circ.bind(values)
    except Exception as exc:
        raise _qumo_error(exc, "parameter binding failed") from None


def select_region(spec: TargetSpec, k: int, starts: int = 24) -> list[int] | None:
    """Pick k connected physical qubits with the lowest calibrated error.

    Neutral (uses only TargetSpec data): greedy growth from the best `starts`
    seed qubits; a qubit's cost is its readout error plus the error of the
    cheapest coupling that attaches it to the region (2q errors weighted x2).
    Returns None when the target is all-to-all, too small or has no error data.
    """
    if spec.all_to_all or k >= spec.num_qubits or not spec.edges:
        return None
    ro = list(spec.readout_errors or [])
    if not ro and not spec.edge_errors:
        return None
    known_ro = [x for x in ro if x is not None]
    med_ro = sorted(known_ro)[len(known_ro) // 2] if known_ro else 0.0
    q_cost = [(ro[i] if i < len(ro) and ro[i] is not None else med_ro) for i in range(spec.num_qubits)]
    med_2q = spec.errors.get("2q", 0.01)
    adj: dict[int, list[tuple[int, float]]] = {}
    for a, b in spec.edges:
        e = spec.edge_errors.get(f"{min(a, b)}-{max(a, b)}", med_2q)
        e = 1.0 if e is None else min(float(e), 1.0)
        adj.setdefault(a, []).append((b, e))
        adj.setdefault(b, []).append((a, e))
    seeds = sorted(adj, key=lambda q: q_cost[q] + 2.0 * min(e for _, e in adj[q]))[:starts]
    best, best_score = None, float("inf")
    for s0 in seeds:
        region, inreg, score = [s0], {s0}, q_cost[s0]
        frontier: dict[int, float] = {}
        for nb, e in adj[s0]:
            frontier[nb] = min(frontier.get(nb, 9.0), q_cost[nb] + 2.0 * e)
        while len(region) < k and frontier:
            q = min(frontier, key=frontier.get)
            score += frontier.pop(q)
            region.append(q)
            inreg.add(q)
            for nb, e in adj[q]:
                if nb not in inreg:
                    frontier[nb] = min(frontier.get(nb, 9.0), q_cost[nb] + 2.0 * e)
            if score >= best_score:
                break
        if len(region) == k and score < best_score:
            best, best_score = sorted(region), score
    return best


def _sub_target(spec: TargetSpec, region: list[int]) -> TargetSpec:
    pos = {p: i for i, p in enumerate(region)}
    edges = [(pos[a], pos[b]) for a, b in spec.edges if a in pos and b in pos]
    return TargetSpec(f"{spec.name}[region]", spec.provider, len(region), edges, list(spec.basis), False,
                      dict(spec.errors), dict(spec.durations), spec.max_shots)


def _widen_qasm(qasm: str, region: list[int], width: int) -> str:
    """Relabel q[i] -> q[region[i]] and widen the register to the full device."""
    lines = []
    for ln in qasm.splitlines():
        if _RE_QREG.match(ln):
            lines.append(f"qreg q[{width}];")
            continue
        lines.append(_RE_QREF.sub(lambda m: f"q[{region[int(m.group(1))]}]", ln))
    return "\n".join(lines) + "\n"


def compile_for_target(
    source: Any,
    spec: TargetSpec,
    level: int = 2,
    seed: int = 1,
    measure_all_if_missing: bool = True,
    region: str | Sequence[int] | None = "auto",
    params: Sequence[float] | dict | None = None,
) -> CompiledCircuit:
    """Compile `source` (OpenQASM 2/3 text, a path, or a qumo_py.Circuit) for `spec`.

    `region`: "auto" restricts large coupling-map devices to the best
    calibrated connected subset of qubits (`select_region`) before QUMO places
    and routes; None / "none" uses the whole device; a list of physical qubits
    pins the region explicitly. `params` binds symbolic parameters first."""
    q = load_qumo_py()
    circ, text = _load_circuit(source)
    if params is not None:
        circ = _bind(circ, text, params)
    n_log = int(circ.num_qubits())
    if n_log == 0:
        raise HardwareError(E_COMPILE, "the circuit has no qubits")
    if n_log > spec.num_qubits:
        raise HardwareError(
            E_COMPILE,
            f"the circuit needs {n_log} qubits but backend '{spec.name}' has {spec.num_qubits}",
        )
    if int(circ.num_parameters()) > 0:
        raise HardwareError(E_COMPILE, "the circuit has unbound symbolic parameters",
                            "pass values with --params '[...]' (CLI) or params=[...] (API)")
    sim_only = sorted(SIMULATION_ONLY & set(dict(circ.count_ops())))
    if sim_only:
        raise HardwareError(E_COMPILE, f"the circuit contains simulation-only instructions {sim_only}",
                            "remove `#pragma qumo` noise channels / detectors: hardware brings its own noise")
    try:
        logical_qasm = circ.to_qasm2()
    except Exception as exc:
        raise _qumo_error(exc, "the circuit cannot be exported to OpenQASM 2 (needed by the provider bridges)",
                          "condition on whole classical registers (if(c==k)); single-bit conditions are not "
                          "supported by the hardware plugins yet") from None
    auto = False
    if "measure" not in dict(circ.count_ops()):
        if not measure_all_if_missing:
            raise HardwareError(E_COMPILE, "the circuit has no measurements",
                                "add measurements, or allow automatic measure-all")
        logical_qasm = _add_measure_all(logical_qasm, n_log)
        circ = q.load_qasm(logical_qasm)
        auto = True
    n_cl = int(circ.num_clbits())

    chosen: list[int] | None = None
    if isinstance(region, str):
        if region == "auto":
            chosen = select_region(spec, min(spec.num_qubits, n_log + max(3, n_log // 2)))
        elif region not in ("none", ""):
            raise HardwareError(E_COMPILE, f"unknown region mode '{region}'", "use 'auto', 'none' or a qubit list")
    elif region is not None:
        chosen = sorted({int(x) for x in region})
        if len(chosen) < n_log or any(x < 0 or x >= spec.num_qubits for x in chosen):
            raise HardwareError(E_COMPILE, "the requested region is too small or out of range")
    work = _sub_target(spec, chosen) if chosen else spec

    width = n_log if work.all_to_all else work.num_qubits
    t0 = time.perf_counter()
    try:
        out, initial, final = q.transpile(circ, width, work.qumo_edges(), list(work.basis), int(level), int(seed))
    except Exception as exc:
        raise HardwareError(E_COMPILE, f"QUMO compilation failed: {str(exc).splitlines()[0]}") from None
    ms = (time.perf_counter() - t0) * 1e3
    _check_conformance(out, work)

    try:
        qasm = out.to_qasm2()
    except Exception as exc:
        raise _qumo_error(exc, "the compiled circuit cannot be exported to OpenQASM 2") from None
    initial = [int(v) for v in list(initial)[:n_log]]
    final = [int(v) for v in list(final)[:n_log]]
    two_q_edges = [(int(a), int(b)) for a, b in out.two_qubit_edges()]
    num_phys = int(out.num_qubits())
    if chosen:
        qasm = _widen_qasm(qasm, chosen, spec.num_qubits)
        initial = [chosen[v] for v in initial]
        final = [chosen[v] for v in final]
        two_q_edges = [(chosen[a], chosen[b]) for a, b in two_q_edges]
        num_phys = spec.num_qubits

    measure_map = {c: p for p, c in parse_measurements(qasm)}
    logical_map = {c: l for l, c in parse_measurements(logical_qasm)}
    if set(measure_map) != set(logical_map):
        raise HardwareError(E_COMPILE, "internal error: compiled measurements do not cover the same clbits")

    active = set(measure_map.values()) | set(final)
    for a, b in two_q_edges:
        active.update((a, b))
    for line in _body_lines(qasm):
        active.update(int(x) for x in _RE_QREF.findall(line))

    check = "consistent"
    for c, l in logical_map.items():
        if measure_map[c] != final[l]:
            check = "measured-before-final-permutation"  # mid-circuit measurement; still correct by clbit

    ops = {str(k): int(v) for k, v in dict(out.count_ops()).items()}
    return CompiledCircuit(
        qasm=qasm,
        source_qasm=logical_qasm,
        num_qubits=num_phys,
        num_logical=n_log,
        num_clbits=n_cl,
        initial_layout=initial,
        final_layout=final,
        measure_map=measure_map,
        logical_measure_map=logical_map,
        auto_measured=auto,
        ops=ops,
        two_qubit_gates=len(two_q_edges),
        depth=int(out.depth()),
        level=int(level),
        seed=int(seed),
        target=spec,
        compile_ms=ms,
        qumo_version=str(q.version()),
        active_qubits=sorted(active),
        layout_check=check,
        region=chosen,
    )


def _check_conformance(out: Any, spec: TargetSpec) -> None:
    """Neutral pre-flight check: native gates only, 2q gates on couplings."""
    if spec.basis:
        allowed = set(spec.basis) | NON_GATE_OPS
        bad = sorted(set(dict(out.count_ops())) - allowed)
        if bad:
            raise HardwareError(E_COMPILE, f"QUMO left non-native gates {bad} for basis {spec.basis}")
    if not spec.all_to_all:
        edges = {tuple(sorted(e)) for e in spec.edges}
        for a, b in out.two_qubit_edges():
            if tuple(sorted((int(a), int(b)))) not in edges:
                raise HardwareError(E_COMPILE, f"QUMO emitted a two-qubit gate on uncoupled qubits ({a}, {b})")


# --------------------------------------------------------------------- bit order helpers
def key_from_bits(bits: Sequence[int]) -> str:
    """bits[k] = value of clbit k  ->  QUMO key (leftmost = highest clbit)."""
    return "".join("1" if b else "0" for b in reversed(list(bits)))


def reorder_counts(
    counts: Mapping[Any, int],
    order: Sequence[int],
    width: int,
    leftmost_is_first: bool = False,
) -> dict[str, int]:
    """Normalise provider counts to QUMO keys.

    Each source key is a bitstring (or tuple of bits). Its i-th position (from
    the left if `leftmost_is_first`, else from the right like Qiskit) holds the
    value of clbit `order[i]`. Clbits not listed are 0.
    """
    out: dict[str, int] = {}
    for key, n in counts.items():
        bits_src = [int(b) for b in (key if not isinstance(key, str) else key.replace(" ", ""))]
        if not leftmost_is_first:
            bits_src = bits_src[::-1]
        if len(bits_src) != len(order):
            raise ValueError(f"key {key!r} has {len(bits_src)} bits, expected {len(order)}")
        bits = [0] * width
        for i, b in enumerate(bits_src):
            bits[order[i]] = b
        k = key_from_bits(bits)
        out[k] = out.get(k, 0) + int(n)
    return out


def physical_counts_to_clbits(
    counts: Mapping[Any, int],
    measured_qubits: Sequence[int],
    compiled: CompiledCircuit | Mapping[str, Any],
    leftmost_is_first: bool = True,
) -> dict[str, int]:
    """Remap counts keyed by *physical* qubits (Braket/IonQ style: position i =
    `measured_qubits[i]`) to the user's clbits via the compiled measure map."""
    if isinstance(compiled, CompiledCircuit):
        mmap = compiled.measure_map
        width = compiled.num_clbits
    else:
        mmap = {int(k): int(v) for k, v in compiled["measure_map"].items()}
        width = int(compiled["num_clbits"])
    phys_to_clbit: dict[int, list[int]] = {}
    for c, p in mmap.items():
        phys_to_clbit.setdefault(p, []).append(c)
    out: dict[str, int] = {}
    for key, n in counts.items():
        src = [int(b) for b in (key if not isinstance(key, str) else key)]
        if not leftmost_is_first:
            src = src[::-1]
        bits = [0] * width
        for i, p in enumerate(measured_qubits):
            for c in phys_to_clbit.get(int(p), []):
                bits[c] = src[i]
        k = key_from_bits(bits)
        out[k] = out.get(k, 0) + int(n)
    return out


def compress_qasm(compiled: CompiledCircuit) -> tuple[str, list[int]]:
    """Restrict the physical circuit to its active qubits (for simulation).

    Returns (qasm, kept) where kept[i] = physical qubit now at index i.
    Classical bits are untouched, so counts keep the user's bit order."""
    kept = sorted(set(compiled.active_qubits))
    pos = {p: i for i, p in enumerate(kept)}
    lines = []
    for ln in compiled.qasm.splitlines():
        if _RE_QREG.match(ln):
            lines.append(f"qreg q[{max(len(kept), 1)}];")
            continue
        lines.append(_RE_QREF.sub(lambda m: f"q[{pos[int(m.group(1))]}]", ln))
    return "\n".join(lines) + "\n", kept


def counts_total(counts: Mapping[str, int]) -> int:
    return int(sum(counts.values()))


def as_iterable(x: Iterable | None) -> list:
    return list(x) if x is not None else []
