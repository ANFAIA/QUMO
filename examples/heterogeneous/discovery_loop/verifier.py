"""The external verifier: builds each candidate as a QUMO program and runs it.

Agents hand it DATA (a genome: five fields), never code and never a cost. It owns the readout
circuits, draws the marked items from its own seed, and measures:

  zero error     every shot of the program returns a marked item
  termination    every shot leaves the loop within the iteration budget
  exact          P(hit) on the first weak check, from the state vector (CPU or GPU), equal to the
                 value the screening model predicts for the same instance
  cost           mean query rounds over the shots, within 5 standard errors of that prediction

This is numerical testing at sizes a simulator can hold. It is not a proof: the paper's verifier
is the Lean kernel against a frozen specification, which covers all sizes. Nothing here does.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

import screening as S

HERE = Path(__file__).resolve().parent
SPEC_SHA256 = "816c0ac28771b49833eb2ecf20b776d406e876f084b0a93d22a42b840c5f14b8"


def load_spec(path: Path = HERE / "spec.json") -> dict:
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SPEC_SHA256:
        raise ValueError(f"spec.json changed (sha256 {digest[:12]}...): the specification is frozen")
    return json.loads(raw)


def validate_genome(g: object, spec: dict) -> dict:
    """Exact keys, enumerated choices and bounded numbers. Anything else is refused."""
    rules = spec["genome"]
    if not isinstance(g, dict) or set(g) != set(rules):
        raise ValueError(f"a candidate has exactly the fields {sorted(rules)}")
    for key in ("family", "diffusion", "readout"):
        if g[key] not in rules[key]:
            raise ValueError(f"{key} must be one of {rules[key]}")
    for key in ("a", "c"):
        v = g[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not rules[key][0] <= v <= rules[key][1]:
            raise ValueError(f"{key} must be a number in {rules[key]}")
    if not S.supported(g):
        raise ValueError("a weak check needs joint diffusion")
    return g


# ------------------------------------------------------------------ circuits (OpenQASM 3)
class _Builder:
    def __init__(self, n: int, m: int, marked: list[int]):
        self.n, self.m, self.marked = n, m, marked
        self.lines: list[str] = []

    def flag(self, s: int) -> int:
        return s * (self.n + 1)

    def addr(self, s: int) -> list[int]:
        return [s * (self.n + 1) + 1 + b for b in range(self.n)]

    def gate(self, name: str, *qs: int) -> None:
        self.lines.append(f"{name} " + ", ".join(f"q[{i}]" for i in qs) + ";")

    def mcx(self, controls: list[int], target: int) -> None:
        mod = f"ctrl({len(controls)}) @ x " if len(controls) > 1 else "cx "
        self.lines.append(mod + ", ".join(f"q[{i}]" for i in controls + [target]) + ";")

    def oracle(self) -> None:
        """One parallel query round: every slot flips its flag if its address is marked."""
        for s in range(self.m):
            for item in self.marked:
                zeros = [q for b, q in enumerate(self.addr(s)) if not (item >> b) & 1]
                for q in zeros:
                    self.gate("x", q)
                self.mcx(self.addr(s), self.flag(s))
                for q in zeros:
                    self.gate("x", q)

    def phase(self) -> None:
        """Sign (-1)^(marked slots) on flags |0...0> (W O W with W = XHX)."""
        for half in range(2):
            for s in range(self.m):
                for name in ("x", "h", "x"):
                    self.gate(name, self.flag(s))
            if half == 0:
                self.oracle()

    def reflect(self, wires: list[int]) -> None:
        for q in wires:
            self.gate("h", q)
            self.gate("x", q)
        self.gate("h", wires[-1])
        self.mcx(wires[:-1], wires[-1])
        self.gate("h", wires[-1])
        for q in wires:
            self.gate("x", q)
            self.gate("h", q)

    def diffuse(self, kind: str) -> None:
        if kind == "joint":
            self.reflect([q for s in range(self.m) for q in self.addr(s)])
        else:
            for s in range(self.m):
                self.reflect(self.addr(s))

    def amplify(self, steps: int, kind: str) -> None:
        for s in range(self.m):
            for q in self.addr(s):
                self.gate("h", q)
        for _ in range(steps):
            self.phase()
            self.diffuse(kind)

    def chord(self, ratio: float) -> None:
        """U = u Z(flag 0) + v X(all flags), with v^2 - u^2 = -ratio."""
        u, v = math.sqrt((1 + ratio) / 2), math.sqrt((1 - ratio) / 2)
        for s in range(1, self.m):
            self.gate("cx", self.flag(0), self.flag(s))
        self.gate("z", self.flag(0))
        self.lines.append(f"ry({2 * math.atan2(v, u):.17g}) q[{self.flag(0)}];")
        for s in range(1, self.m):
            self.gate("cx", self.flag(0), self.flag(s))

    def weak_check(self, ratio: float) -> None:
        self.chord(ratio)
        self.phase()
        self.chord(ratio)

    def source(self, clbits: int, extra_decl: str = "") -> str:
        head = ('OPENQASM 3.0;\ninclude "stdgates.inc";\n'
                f"qubit[{self.m * (self.n + 1)}] q;\nbit[{clbits}] c;\n" + extra_decl)
        return head + "\n".join(self.lines) + "\n"


def program_source(g: dict, n: int, m: int, marked: list[int]) -> str:
    """c[0] = 'done'; then, from bit 1, all qubits (checked) or slot 0's address (trusted)."""
    K, ratio = S.schedule(g, n, m, len(marked))
    b = _Builder(n, m, marked)
    width = m * (n + 1)
    if g["family"] == "weak":
        b.amplify(K + 1, "joint")
        b.lines.append("while (!c[0]) {")
        b.weak_check(ratio)
        b.lines.append(f"c[0] = measure q[{b.flag(0)}];")
        b.lines.append("if (!c[0]) {")
        b.diffuse("joint")
        b.lines.append("}\n}")
        if g["readout"] == "checked":
            for s in range(m):
                b.gate("x", b.flag(s))  # a hit leaves every flag at 1
            b.oracle()
    else:
        b.lines.append("while (!c[0]) {")
        for q in range(width):
            b.lines.append(f"reset q[{q}];")
        b.amplify(K, g["diffusion"])
        if g["readout"] == "checked":
            b.oracle()
            for s in range(m):
                b.lines.append(f"c[{1 + b.flag(s)}] = measure q[{b.flag(s)}];")
            op = " ^ " if g["diffusion"] == "joint" else " || "
            bits = [f"c[{1 + b.flag(s)}]" for s in range(m)]
            if op == " || ":
                bits = [f"bool({x})" for x in bits]
            b.lines.append("c[0] = " + op.join(bits) + ";")
        else:
            b.lines.append("c[0] = true;")  # stop after one attempt, believe it
        b.lines.append("}")
    if g["readout"] == "checked":
        for q in range(width):
            b.lines.append(f"c[{1 + q}] = measure q[{q}];")
    else:
        for i, q in enumerate(b.addr(0)):
            b.lines.append(f"c[{1 + i}] = measure q[{q}];")
    return b.source(1 + width)


def _answers(g: dict, n: int, m: int, key: str) -> list[int]:
    """The addresses a shot returns, from its counts key (highest classical bit first)."""
    read = int(key, 2) >> 1
    if g["readout"] == "trusted":
        return [read & ((1 << n) - 1)]
    flagged = [(read >> (s * (n + 1) + 1)) & ((1 << n) - 1) for s in range(m) if (read >> (s * (n + 1))) & 1]
    return flagged[:1]


def first_hit_circuit(g: dict, n: int, m: int, marked: list[int]) -> str:
    K, ratio = S.schedule(g, n, m, len(marked))
    b = _Builder(n, m, marked)
    b.amplify(K + 1, "joint")
    b.weak_check(ratio)
    return b.source(1)


class Verifier:
    def __init__(self, qumo, spec: dict, seed: int, shots: int = 800, budget: int = 200):
        self.q, self.spec, self.shots, self.budget = qumo, spec, shots, budget
        self._rng = random.Random(f"verifier-{seed}")  # private: agents never see the marked items
        self.seed = seed
        self.calls = 0

    def marked_for(self, n: int, k: int) -> list[int]:
        return sorted(self._rng.sample(range(1 << n), k))

    def exact_first_hit(self, g: dict, n: int, m: int, k: int, device: str = "cpu") -> dict:
        marked = self.marked_for(n, k)
        circuit = self.q.load_qasm(first_hit_circuit(g, n, m, marked))
        width = m * (n + 1)
        z0 = "I" * (width - 1) + "Z"  # qubit 0 is flag 0, rightmost in the label
        value, method, note = self.q.expectation_device(circuit, [(z0, 1.0)], device)
        measured = (1.0 - value) / 2.0
        predicted = S.rounds(g, n, m, k)[1]
        return {"n": n, "m": m, "k": k, "qubits": width, "measured": measured, "predicted": predicted,
                "method": method, "note": note}

    def verify(self, g: object) -> dict:
        """{'accepted': bool, 'reason': str, 'cells': [...], 'cost': screened cost if accepted}."""
        self.calls += 1
        try:
            g = validate_genome(g, self.spec)
        except ValueError as exc:
            return {"accepted": False, "reason": f"schema: {exc}", "cells": [], "cost": None}
        cells = []
        for n, m, k in self.spec["verification_cells"]:
            marked = self.marked_for(n, k)
            predicted, _ = S.rounds(g, n, m, k)
            if not math.isfinite(predicted):
                return {"accepted": False, "reason": f"does not terminate at n={n}, m={m}, k={k}",
                        "cells": cells, "cost": None}
            if g["family"] == "weak":
                exact = self.exact_first_hit(g, n, m, k)
                if abs(exact["measured"] - exact["predicted"]) > 1e-6:
                    return {"accepted": False, "cells": cells, "cost": None,
                            "reason": f"P(hit) {exact['measured']:.6f} differs from the model {exact['predicted']:.6f}"}
            try:
                run = self.q.sample_program(self.q.load_program(program_source(g, n, m, marked)),
                                            self.shots, self.seed + self.calls,
                                            max_iterations=self.budget)
            except Exception as exc:  # QM-E480: the loop did not end within the budget
                return {"accepted": False, "reason": f"run failed: {str(exc)[:160]}", "cells": cells, "cost": None}
            wrong = 0
            for key, count in run["counts"].items():
                got = _answers(g, n, m, key)
                if not got or got[0] not in marked:
                    wrong += count
            loops = list(run["iterations"])
            mean = sum(loops) / len(loops)
            spread = math.sqrt(sum((x - mean) ** 2 for x in loops) / max(len(loops) - 1, 1))
            K, _ = S.schedule(g, n, m, k)
            confirm = 1 if g["readout"] == "checked" else 0
            if g["family"] == "weak":
                measured, error = K + 1 + mean + confirm, spread / math.sqrt(len(loops))
            else:
                per = K + confirm
                measured, error = per * mean, per * spread / math.sqrt(len(loops))
            cells.append({"n": n, "m": m, "k": k, "wrong_shots": wrong, "shots": self.shots,
                          "measured_rounds": measured, "predicted_rounds": predicted, "std_error": error})
            if wrong:
                return {"accepted": False, "cells": cells, "cost": None,
                        "reason": f"zero error violated: {wrong}/{self.shots} wrong answers at n={n}, m={m}, k={k}"}
            if abs(measured - predicted) > 5 * error + 1e-9:
                return {"accepted": False, "cells": cells, "cost": None,
                        "reason": f"measured {measured:.3f} rounds, model says {predicted:.3f} (> 5 std errors)"}
        cost = S.cost(g, self.spec["screening_n"], self.spec["screening_cells"])
        return {"accepted": True, "reason": "ok", "cells": cells, "cost": cost}
