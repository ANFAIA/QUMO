#!/usr/bin/env python3
"""Exact fault-tolerance counts of two Steane |0>_L preparations, computed with Stim alone.

Reproduces: Schmid, Wille - Definition 1 (t-fault-tolerant post-selected state preparation), independent reference for examples/ft/ft_state_prep_check.mojo
Source: https://arxiv.org/abs/2611.25107
Framework: QUMO - https://github.com/ANFAIA/QUMO

The independent reference for QUMO's `fault_tolerant_state_preparation`. It does not
import QUMO. Each fault set is written into the circuit as Pauli gates and run on
`stim.TableauSimulator`; a set is detected when a measurement differs from the fault-free
run. For an undetected set the residual error is read from the state, not propagated:
the signs of the Z-type state generators give the syndrome of the X part, the X-type
generators that of the Z part, and the reduced weight is the smallest weight with that
syndrome (the state group is maximal, so the syndrome fixes the error up to the group).

Fault locations, in QUMO's order: the initial |0> of every qubit (X), X/Y/Z after each H,
the 15 two-qubit Paulis after each CX (first qubit outer, I X Y Z), an outcome flip before
each measurement. The X/Z-only model keeps X and Z after H and {XI, IX, XX, ZI, IZ, ZZ}
(in that I X Y Z order) after each CX.

Checks: the counts (fault sets, undetected, dangerous) for t = 1 and the undetected sets
at order 2 equal the numbers QUMO's example prints, for the encoder alone, the flagged
encoder (steane_prep_zero_ft) and the negative control (flag on the stabilizer Z0 Z1 Z2 Z3).

Run: bench/.venv/bin/python examples/ft/ft_state_prep_check_stim.py
example-meta: {"framework": "qumo", "tier": "fast", "timeout": 120, "requires": ["stim"], "args": []}
"""

from __future__ import annotations

import itertools
import os
import sys

import stim

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _checks import CheckLog  # noqa: E402

N = 7
PAIRS = [0, 1, 4, 5, 6, 3, 6, 5, 4, 2, 0, 3, 4, 1, 3, 2]  # qumo/ft/gadgets.mojo steane_encoder_zero
Z_STATE = ["ZZZZIII", "IZZIZZI", "IIZZIZZ", "ZZZZZZZ"]  # code Z generators and Z_L: |0>_L
X_STATE = ["XXXXIII", "IXXIXXI", "IIXXIXX"]


def encoder():
    ops = [("H", 0), ("H", 4), ("H", 6)]
    for i in range(8):
        ops.append(("CX", PAIRS[2 * i], PAIRS[2 * i + 1]))
    return ops


def flagged(flag):
    ops = encoder()
    for q in flag:
        ops.append(("CX", q, 7))
    ops.append(("M", 7))
    return ops


def locations(ops, nq, full):
    one = "XYZ" if full else "XZ"
    two = [(a, b) for a in "IXYZ" for b in "IXYZ" if (a, b) != ("I", "I")]
    if not full:
        two = [(a, b) for (a, b) in two if set(a + b) <= set("IX") or set(a + b) <= set("IZ")]
    locs = [[(-1, "P", ((q, "X"),))] for q in range(nq)]
    for i, op in enumerate(ops):
        if op[0] == "H":
            locs.append([(i, "P", ((op[1], p),)) for p in one])
        elif op[0] == "CX":
            locs.append([(i, "P", tuple((q, p) for q, p in zip(op[1:], ab) if p != "I")) for ab in two])
        elif op[0] == "M":
            locs.append([(i, "F", None)])
    return locs


def run(ops, nq, faults):
    s = stim.TableauSimulator()
    s.set_num_qubits(nq)

    def inject(data):
        for q, p in data:
            {"X": s.x, "Y": s.y, "Z": s.z}[p](q)

    for pos, kind, data in faults:
        if pos == -1:
            inject(data)
    rec = []
    for i, op in enumerate(ops):
        if op[0] == "H":
            s.h(op[1])
        elif op[0] == "CX":
            s.cnot(op[1], op[2])
        elif op[0] == "M":
            m = s.measure(op[1])
            if any(p == i and k == "F" for p, k, _ in faults):
                m = not m
            rec.append(int(m))
        for pos, kind, data in faults:
            if pos == i and kind == "P":
                inject(data)
    return s, rec


def min_weight(gens):
    """syndrome w.r.t. gens -> smallest weight of an error of the other type with it."""
    tab = {}
    for bits in itertools.product([0, 1], repeat=N):
        syn = tuple(sum(b for b, c in zip(bits, g) if c != "I") % 2 for g in gens)
        tab[syn] = min(tab.get(syn, N + 1), sum(bits))
    return tab


MIN_X = min_weight(Z_STATE)
MIN_Z = min_weight(X_STATE)


def counts(ops, nq, size, full):
    locs = locations(ops, nq, full)
    _, ref = run(ops, nq, [])
    sets = undetected = dangerous = 0
    pad = "I" * (nq - N)
    for combo in itertools.combinations(range(len(locs)), size):
        for choice in itertools.product(*[range(len(locs[c])) for c in combo]):
            sets += 1
            s, rec = run(ops, nq, [locs[c][k] for c, k in zip(combo, choice)])
            if rec != ref:
                continue
            undetected += 1
            sx = tuple(0 if s.peek_observable_expectation(stim.PauliString(g + pad)) == 1 else 1 for g in Z_STATE)
            sz = tuple(0 if s.peek_observable_expectation(stim.PauliString(g + pad)) == 1 else 1 for g in X_STATE)
            if MIN_X[sx] > size or MIN_Z[sz] > size:
                dangerous += 1
    return sets, undetected, dangerous


# (name, ops, qubits, model) -> (t = 1: sets, undetected, dangerous), undetected at order 2
EXPECTED = {
    ("encoder", True): ((136, 136, 29), 8331),
    ("encoder", False): ((61, 61, 8), 1707),
    ("flagged", True): ((183, 90, 0), 7735),
    ("flagged", False): ((81, 54, 0), 1702),
    ("stabilizer flag", True): ((198, 122, 21), 9466),
    ("stabilizer flag", False): ((87, 65, 6), 2200),
}


def main() -> None:
    log = CheckLog("ft_state_prep_check_stim")
    print("stim", stim.__version__)
    circuits = [("encoder", encoder(), 7), ("flagged", flagged((1, 3, 5)), 8),
                ("stabilizer flag", flagged((0, 1, 2, 3)), 8)]
    for name, ops, nq in circuits:
        for full in (True, False):
            model = "circuit_level" if full else "xz_only"
            one = counts(ops, nq, 1, full)
            two = counts(ops, nq, 2, full)[1]
            print(f"  {name:16s} {model:14s} t=1 sets/undetected/dangerous = {one}, undetected at order 2 = {two}")
            exp_one, exp_two = EXPECTED[(name, full)]
            log.check(f"{name} {model} t=1 counts", one == exp_one, f"stim={one} qumo={exp_one}")
            log.check(f"{name} {model} order-2 undetected", two == exp_two, f"stim={two} qumo={exp_two}")
    log.check("flagged preparation is 1-fault tolerant", counts(flagged((1, 3, 5)), 8, 1, True)[2] == 0)
    log.finish()


if __name__ == "__main__":
    main()
