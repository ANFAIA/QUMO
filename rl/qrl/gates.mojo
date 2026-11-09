# Copyright 2026 The QUMO Authors. Licensed under the Apache License, Version 2.0 (see LICENSE).
# Opcode subset copied from QUMO qumo/ir/gates.mojo so that qrl imports nothing from `qumo.*`. The numeric values
# MUST stay equal to QUMO's (`actions` triples, checkpoints and fixtures carry them); rl/qumo_bridge/tests/
# test_bridge_parity.mojo checks every constant and name below against qumo.ir.gates.
"""Gate opcodes of the synthesis environments (the Clifford generators, CX / CZ / SWAP and the Pauli gates).

Only the gates a synthesis model can act with, the Paulis the Pauli-network conjugation rules need, and T / TDG
(the non-Clifford gates the action validators must reject). The numbers
are QUMO's opcodes; `op_name` gives QUMO's lower-case gate name.
"""


struct Op:
    comptime I = 0
    comptime X = 1
    comptime Y = 2
    comptime Z = 3
    comptime H = 4
    comptime S = 5
    comptime SDG = 6
    comptime T = 7  # not a synthesis action (validators reject it)
    comptime TDG = 8
    comptime SX = 9
    comptime SXDG = 10
    comptime CX = 16
    comptime CZ = 18
    comptime SWAP = 20


def op_name(k: Int) -> String:
    """QUMO's gate name of an opcode of this subset ("unknown" otherwise)."""
    if k == Op.I:
        return "id"
    if k == Op.X:
        return "x"
    if k == Op.Y:
        return "y"
    if k == Op.Z:
        return "z"
    if k == Op.H:
        return "h"
    if k == Op.S:
        return "s"
    if k == Op.SDG:
        return "sdg"
    if k == Op.T:
        return "t"
    if k == Op.TDG:
        return "tdg"
    if k == Op.SX:
        return "sx"
    if k == Op.SXDG:
        return "sxdg"
    if k == Op.CX:
        return "cx"
    if k == Op.CZ:
        return "cz"
    if k == Op.SWAP:
        return "swap"
    return "unknown"
