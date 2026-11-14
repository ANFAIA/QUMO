"""Compilation: `transpile` a circuit for a target (layout, routing, basis, optimisation).

Targets are vendor-neutral. A target is either the one-line spec the Mojo API
and the command line take (``"grid:2x2/sc_cz"``), or an explicit coupling map
and basis.
"""

from __future__ import annotations

import json

from ._native import native as _q
from .circuit import Circuit, _call, _native_of
from .errors import QumoError, QumoTypeError

__all__ = ["transpile", "TranspileResult", "target_info", "BASIS_SETS", "TOPOLOGIES"]

BASIS_SETS = ("any", "generic", "sc_cx", "sc_cz", "sc_ecr", "sc_iswap", "ion_ms", "ion_zz", "atom_cz", "clifford_t")
TOPOLOGIES = ("all_to_all", "line", "ring", "grid", "torus", "tree", "heavy_hex", "superconducting_heavy_hex",
              "superconducting_grid", "superconducting_grid_iswap", "trapped_ion", "neutral_atom")


class TranspileResult:
    """What `transpile` returns.

    Attributes:
        circuit: the compiled `qumo.Circuit` on the target's physical qubits.
        initial_layout: ``initial_layout[l]`` = physical qubit of logical qubit ``l`` at the start.
        final_layout: the same after routing (SWAPs move qubits).
        metrics: ``{"gates": (before, after), "two_qubit": ..., "depth": ...,
            "t_count": ..., "est_fidelity": float, "compile_ms": float}``.
        report: the full per-pass report (dict): target, basis, pipeline, passes.
    """

    __slots__ = ("circuit", "initial_layout", "final_layout", "_report_json", "_report")

    def __init__(self, circuit: Circuit, initial_layout, final_layout, report_json: str):
        self.circuit = circuit
        self.initial_layout = list(initial_layout)
        self.final_layout = list(final_layout)
        self._report_json = report_json
        self._report = None

    @property
    def report(self) -> dict:
        if self._report is None:
            self._report = json.loads(self._report_json)
        return self._report

    @property
    def metrics(self) -> dict:
        r = self.report
        return {
            "gates": tuple(r["gates"]),
            "two_qubit": tuple(r["2q"]),
            "depth": tuple(r["depth"]),
            "t_count": tuple(r["t_count"]),
            "est_fidelity": r["est_fidelity"],
            "compile_ms": r["total_us"] / 1000.0,
        }

    def __iter__(self):
        """Unpacks like the low-level tuple: ``circuit, initial, final = qumo.transpile(...)``."""
        yield self.circuit
        yield self.initial_layout
        yield self.final_layout

    def __repr__(self) -> str:
        m = self.metrics
        return (f"<qumo.TranspileResult target={self.report['target']} basis={self.report['basis']} "
                f"gates {m['gates'][0]}->{m['gates'][1]}, 2q {m['two_qubit'][0]}->{m['two_qubit'][1]}, "
                f"depth {m['depth'][0]}->{m['depth'][1]}>")


def _basis_arg(basis):
    if basis is None:
        return "any"
    if isinstance(basis, str):
        return basis
    names = list(basis)
    if not all(isinstance(b, str) for b in names):
        raise QumoTypeError("basis is a basis-set name or a list of gate names", "QM-E911",
                            f"e.g. basis='sc_cz' or basis=['rz', 'sx', 'x', 'cz']; basis sets: {', '.join(BASIS_SETS)}.")
    return names


def _edges_arg(coupling_map):
    edges = []
    for e in coupling_map:
        try:
            a, b = e
            edges.append((a.__index__(), b.__index__()))
        except (TypeError, ValueError, AttributeError):
            raise QumoTypeError(f"coupling_map entries are pairs of qubit numbers, got {e!r}", "QM-E911",
                                "e.g. coupling_map=[(0, 1), (1, 2)].") from None
    return edges


def transpile(circuit, target: str | None = None, *, coupling_map=None, basis=None, num_qubits: int | None = None,
              level: int = 2, seed: int = 7) -> TranspileResult:
    """Compile `circuit` for a target: layout, routing, basis translation, optimisation.

    Give the target in one of two ways:

    * `target`: a one-line spec ``"<topology>:<size>[/<basis>]"``, exactly as
      `parse_target` in Mojo and ``qumo compile --target`` take it:
      ``"grid:2x2/sc_cz"``, ``"line:20"``, ``"heavy_hex:5/rz,sx,x,ecr"``,
      ``"trapped_ion:11"``, ``"all_to_all:8/sc_cx"``.
    * `coupling_map` and/or `basis`: explicit data. `coupling_map` is a list of
      ``(a, b)`` physical couplings (omit for all-to-all); `basis` is a
      basis-set name (``"sc_cz"``, see `qumo.BASIS_SETS`) or a list of gate
      names (``["rz", "sx", "x", "cz"]``); `num_qubits` defaults to what the
      coupling map or the circuit needs.

    With neither, the circuit is optimised for an all-to-all device with any gate.

    Args:
        level: optimisation level 0..3 (default 2, as the CLI). Level 3 also
            runs level 2 and keeps the better result.
        seed: seed of the randomised passes (default 7, as the CLI).

    Returns:
        `TranspileResult` with `.circuit`, `.initial_layout`, `.final_layout`,
        `.metrics` and `.report`. It also unpacks as
        ``circuit, initial_layout, final_layout``.

    Example:
        >>> import qumo
        >>> ghz = qumo.Circuit(4).h(0).cx(0, 1).cx(0, 2).cx(0, 3)
        >>> r = qumo.transpile(ghz, "grid:2x2/sc_cz")
        >>> sorted(r.circuit.count_ops())
        ['cz', 'rz', 'sx']
        >>> r2 = qumo.transpile(ghz, coupling_map=[(0, 1), (1, 2), (2, 3)], basis="sc_cz", level=3)
        >>> all(abs(a - b) == 1 for a, b in r2.circuit.two_qubit_edges())
        True
    """
    nat = _native_of(circuit)
    if isinstance(target, str):
        if coupling_map is not None or basis is not None or num_qubits is not None:
            raise QumoError("give either a target spec or coupling_map/basis/num_qubits, not both", "QM-E913",
                            "put the basis in the spec: 'grid:2x2/sc_cz'.")
        out, initial, final, report = _call(_q.transpile_target, nat, target, level, seed)
    elif target is not None:
        raise QumoTypeError(f"target is a spec string such as 'grid:2x2/sc_cz', got {type(target).__name__}",
                            "QM-E911", "pass explicit data as coupling_map=[(0, 1), ...] and basis=[...].")
    else:
        edges = _edges_arg(coupling_map) if coupling_map is not None else []
        if num_qubits is None:
            num_qubits = max([nat.num_qubits()] + [max(e) + 1 for e in edges])
        out, initial, final, report = _call(_q.transpile_report, nat, num_qubits, edges, _basis_arg(basis), level, seed)
    return TranspileResult(Circuit.from_native(out), initial, final, str(report))


def target_info(spec: str) -> dict:
    """Describe a one-line target spec: name, modality, num_qubits, all_to_all, edges, basis.

    Example:
        >>> import qumo
        >>> t = qumo.target_info("grid:2x2/sc_cz")
        >>> t["num_qubits"], t["edges"], t["basis"]
        (4, [(0, 1), (0, 2), (1, 3), (2, 3)], ['rz', 'sx', 'x', 'cz'])
    """
    return dict(_call(_q.parse_target, spec))
