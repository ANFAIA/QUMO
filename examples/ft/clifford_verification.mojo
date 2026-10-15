"""Verifying logical Clifford gadgets and state preparations with the stabilizer tableau.

Reproduces: Guppy FT - Clifford verification
Source: https://github.com/Quantinuum/guppyft/blob/main/examples/clifford_verification.ipynb (Apache-2.0)
Source: https://github.com/Quantinuum/guppyft/blob/main/tests/verifier/ops/nonCSS_4q.py (Apache-2.0)

The notebook checks physical gadgets against their intended logical action with
`valid_clifford_implementation(semantic, impl, code)` and
`valid_stabilizer_state_preparation(...)`: transversal gates on Steane, a
two-block CX, [[4,2,2]] gadgets that are pure qubit permutations, logical
state preparation, and a deliberately incorrect gadget for the non-CSS
[[4,2,2]] code ("Table 2, row 4", which sends a logical Z to minus the right
operator). QUMO's `qumo.ft.verify` gives the same verdicts from the rows of its
inverse-tableau simulator (Heisenberg picture) instead of Choi states, and
names the failing operator.

Checks: every Steane / Iceberg library gadget VALID; the wrong Steane S
(transversal S instead of S^dag) and the incorrect row-4 gadget are caught as
sign errors; a gate on 6 of 7 qubits is caught as leaving the code space; the
Steane |0>_L encoder (with and without its flag) is a valid |0>_L preparation
but not a |+>_L preparation; T|+>_L magic states pass the state-vector check.

Run: .venv/bin/mojo run -I . examples/ft/clifford_verification.mojo
example-meta: {"framework": "guppyft", "tier": "fast", "timeout": 300, "requires": [], "args": []}
"""

from qumo.ir.circuit import Circuit
from qumo.ir.gates import Op
from qumo.ft.code import steane_code, iceberg_code, css_422_code, noncss_422_code
from qumo.ft.verify import valid_clifford_implementation, valid_stabilizer_state_preparation
from qumo.ft.gadgets import (
    verify_steane_library, verify_iceberg_library, transversal_1q, transversal_2q,
    steane_encoder_zero, steane_prep_zero_ft, steane_prep_t_ft, valid_magic_state_preparation,
)
from qumo.testing.checks import CheckLog


def main() raises:
    var log = CheckLog("clifford_verification")
    var st = steane_code()
    print(st.to_json())

    # 1. the whole gadget libraries
    var bad = 0
    var total = 0
    for c in verify_steane_library(st):
        print("  steane  ", c)
        total += 1
        if not c.report.ok:
            bad += 1
    for c in verify_iceberg_library(iceberg_code(4)):
        total += 1
        if not c.report.ok:
            bad += 1
    _ = log.check("all library gadgets valid", bad == 0, String(total) + " gadgets, " + String(bad) + " invalid")

    # 2. notebook cases: transversal H, two-block CX
    var sh = Circuit(1)
    sh.h(0)
    _ = log.check("steane transversal H", valid_clifford_implementation(sh, transversal_1q(Op.H, 7), st).ok)
    var scx = Circuit(2)
    scx.cx(0, 1)
    _ = log.check("steane transversal CX (two blocks)", valid_clifford_implementation(scx, transversal_2q(Op.CX, 7), st, 2).ok)

    # 3. wrong gadgets are caught, with a reason
    var ss = Circuit(1)
    ss.s(0)
    var r1 = valid_clifford_implementation(ss, transversal_1q(Op.S, 7), st)
    print("  wrong S:", r1)
    _ = log.check("transversal S is not logical S (sign)", (not r1.ok) and r1.reason.find("wrong sign") >= 0)
    var h6 = Circuit(7)
    for q in range(6):
        h6.h(q)
    var r2 = valid_clifford_implementation(sh, h6, st)
    print("  H on 6 qubits:", r2)
    _ = log.check("partial H leaves the code space", (not r2.ok) and r2.reason.find("code space") >= 0)

    # 4. [[4,2,2]]: permutations implement logical gates (Guppy FT test ops)
    var c4 = css_422_code()
    var sem = Circuit(2)
    sem.cx(0, 1)
    var imp = Circuit(4)
    imp.swap(3, 1)
    _ = log.check("css [[4,2,2]] intra-block CX = qubit swap", valid_clifford_implementation(sem, imp, c4).ok)
    var nc = noncss_422_code()
    var s4 = Circuit(2)
    s4.cx(0, 1)
    s4.swap(0, 1)
    var wrong = Circuit(4)
    wrong.h(0)
    wrong.s(0)
    wrong.sdg(1)
    wrong.h(1)
    wrong.sdg(2)
    wrong.h(2)
    wrong.h(3)
    wrong.s(3)
    var r3 = valid_clifford_implementation(s4, wrong, nc)
    print("  non-CSS row 4 (incorrect):", r3)
    _ = log.check("non-CSS row 4 incorrect gadget caught", (not r3.ok) and r3.reason.find("wrong sign") >= 0)
    var right = Circuit(4)
    right.h(0)
    right.sdg(0)
    right.sdg(1)
    right.h(1)
    right.s(2)
    right.h(2)
    right.h(3)
    right.s(3)
    _ = log.check("non-CSS row 4 correct gadget valid", valid_clifford_implementation(s4, right, nc).ok)

    # 5. state preparation
    _ = log.check("steane |0>_L encoder", valid_stabilizer_state_preparation(Circuit(1), steane_encoder_zero(), st).ok)
    _ = log.check("steane |0>_L encoder + flag", valid_stabilizer_state_preparation(Circuit(1), steane_prep_zero_ft(), st, 1, 1).ok)
    var plus = Circuit(1)
    plus.h(0)
    _ = log.check("|0>_L encoder is not |+>_L", not valid_stabilizer_state_preparation(plus, steane_encoder_zero(), st).ok)
    var mt = valid_magic_state_preparation(steane_prep_t_ft(), st)
    print("  FT T|+>_L:", mt)
    _ = log.check("flag-verified T|+>_L magic state", mt.ok)
    log.finish()
