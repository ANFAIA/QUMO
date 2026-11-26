"""Clifford conjugation of Paulis: one table for every Clifford opcode.

`clifford_heisenberg(kind, idx, ...)` gives G^dag A G and `clifford_image`
gives G A G^dag, for A one of X_a, Z_a, X_b, Z_b (idx 0..3; qubit a is the
first operand, the control of CX / CY), in XZ form i^k X_a^xa Z_a^za X_b^xb Z_b^zb
(Y = i X Z). `clifford_sign` turns that phase into the sign bit of a
Hermitian Pauli, the form tableau rows use.

Opcodes: I, X, Y, Z, H, S, SDG, SX, SXDG, CX, CY, CZ, SWAP, ISWAP, ISWAPDG, ECR.
The table is checked against U A U^dag of the opcode matrices, sign included,
and the specialised word-parallel loops (the tableau simulator's rows, the
Clifford-block tableau of qumo.compiler.blockalg) are checked against the table,
in tests/test_clifford_table.mojo.
"""

from qumo.ir.gates import Op, op_inverse_kind


def clifford_heisenberg(kind: Int, idx: Int, mut xa: Int, mut za: Int, mut xb: Int, mut zb: Int, mut k: Int) -> Bool:
    """G^dag A G for A = (X_a, Z_a, X_b, Z_b)[idx], as i^k X_a^xa Z_a^za X_b^xb Z_b^zb.
    False (and the identity image) for an opcode the table does not hold."""
    xa = 0; za = 0; xb = 0; zb = 0; k = 0
    if kind == Op.H:
        if idx == 0: za = 1
        else: xa = 1
        return True
    if kind == Op.S:
        if idx == 0: xa = 1; za = 1; k = 3
        else: za = 1
        return True
    if kind == Op.SDG:
        if idx == 0: xa = 1; za = 1; k = 1
        else: za = 1
        return True
    if kind == Op.X:
        if idx == 0: xa = 1
        else: za = 1; k = 2
        return True
    if kind == Op.Y:
        if idx == 0: xa = 1; k = 2
        else: za = 1; k = 2
        return True
    if kind == Op.Z:
        if idx == 0: xa = 1; k = 2
        else: za = 1
        return True
    if kind == Op.SX:
        if idx == 0: xa = 1
        else: xa = 1; za = 1; k = 1
        return True
    if kind == Op.SXDG:
        if idx == 0: xa = 1
        else: xa = 1; za = 1; k = 3
        return True
    if kind == Op.I:
        if idx == 0: xa = 1
        else: za = 1
        return True
    if kind == Op.CX:
        if idx == 0: xa = 1; xb = 1
        elif idx == 1: za = 1
        elif idx == 2: xb = 1
        else: za = 1; zb = 1
        return True
    if kind == Op.CZ:
        if idx == 0: xa = 1; zb = 1
        elif idx == 1: za = 1
        elif idx == 2: za = 1; xb = 1
        else: zb = 1
        return True
    if kind == Op.CY:
        if idx == 0: xa = 1; xb = 1; zb = 1; k = 1
        elif idx == 1: za = 1
        elif idx == 2: za = 1; xb = 1
        else: za = 1; zb = 1
        return True
    if kind == Op.SWAP:
        if idx == 0: xb = 1
        elif idx == 1: zb = 1
        elif idx == 2: xa = 1
        else: za = 1
        return True
    if kind == Op.ISWAP or kind == Op.ISWAPDG:
        var kk = 3 if kind == Op.ISWAP else 1
        if idx == 0: za = 1; xb = 1; zb = 1; k = kk
        elif idx == 1: zb = 1
        elif idx == 2: xa = 1; za = 1; zb = 1; k = kk
        else: za = 1
        return True
    if kind == Op.ECR:
        if idx == 0: xa = 1; za = 1; xb = 1; k = 3
        elif idx == 1: za = 1; k = 2
        elif idx == 2: xb = 1
        else: za = 1; xb = 1; zb = 1; k = 1
        return True
    return False


def clifford_image(kind: Int, idx: Int, mut xa: Int, mut za: Int, mut xb: Int, mut zb: Int, mut k: Int) -> Bool:
    """G A G^dag (the Schroedinger direction): the Heisenberg table of G^dag."""
    return clifford_heisenberg(op_inverse_kind(kind), idx, xa, za, xb, zb, k)


def clifford_sign(xa: Int, za: Int, xb: Int, zb: Int, k: Int) -> Int:
    """Sign bit r of i^k X^x Z^z written as (-1)^r times a product of X, Y, Z."""
    return ((k - (xa & za) - (xb & zb)) & 3) >> 1
