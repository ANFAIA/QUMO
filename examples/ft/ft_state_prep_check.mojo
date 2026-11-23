"""Exact fault-tolerance check of flagged Steane state preparation.

Reproduces: Schmid, Wille - Definition 1, t-fault-tolerant post-selected state preparation (Sec. 2)
Source: https://arxiv.org/abs/2611.25107
Framework: QUMO - https://github.com/ANFAIA/QUMO

A post-selected preparation is t-fault tolerant if every set F of at most t faults is
detected or leaves a data error whose X and Z parts each have weight at most |F| after
reduction by the prepared state's stabilizers. `valid_stabilizer_state_preparation`
checks the circuit without faults only; `fault_tolerant_state_preparation` enumerates
every fault set under a circuit-level model (initial |0> and resets, X/Y/Z after 1q
gates, 15 Paulis after each CX, outcome flips) and reports the first dangerous set.

The Steane |0>_L encoder alone is not 1-fault tolerant: an X on qubit 3 at the start
spreads to X2 X3. With one flag measuring Z1 Z3 Z5 (a weight-3 representative of Z_L,
steane_prep_zero_ft) it is. A flag on the stabilizer Z0 Z1 Z2 Z3 is a valid preparation
without faults but not fault tolerant. Every count below is reproduced independently by
examples/ft/ft_state_prep_check_stim.py, which uses Stim alone.

Checks: counts for the three circuits under the circuit-level and X/Z-only models
(fault sets, undetected, dangerous at t = 1; undetected sets at order 2), the verdicts,
the witness of the encoder, and CX count and depth.

Run: .venv/bin/mojo run -I . examples/ft/ft_state_prep_check.mojo
example-meta: {"framework": "qumo", "tier": "fast", "timeout": 300, "requires": [], "args": []}
"""

from qumo.ir.circuit import Circuit
from qumo.ft.code import steane_code
from qumo.ft.verify import valid_stabilizer_state_preparation
from qumo.ft.gadgets import steane_encoder_zero, steane_prep_zero_ft
from qumo.ft.faults import (
    FaultModel, FaultToleranceReport, fault_tolerant_state_preparation, count_undetected,
)
from qumo.testing.checks import CheckLog


def _stab_flag() -> Circuit:
    """Encoder + flag on the code stabilizer Z0 Z1 Z2 Z3 (valid without faults, not fault tolerant)."""
    var c = Circuit(8, 1, "stab_flag")
    c.compose(steane_encoder_zero())
    c.cx(0, 7)
    c.cx(1, 7)
    c.cx(2, 7)
    c.cx(3, 7)
    c.measure(7, 0)
    return c^


def _check_one(mut log: CheckLog, name: String, impl: Circuit, num_ancillas: Int, model: FaultModel,
               fault_sets: Int, undetected: Int, dangerous: Int, order2: Int) raises -> FaultToleranceReport:
    var st = steane_code()
    var r = fault_tolerant_state_preparation(Circuit(1), impl, st, num_ancillas, -1, model)
    print("  ", name, ":", r)
    var u2 = count_undetected(Circuit(1), impl, st, num_ancillas, 2, model)
    print("  ", name, model.name(), "order-2 undetected:", u2)
    _ = log.check(name + " " + model.name() + " t=1 counts",
                  r.fault_sets == fault_sets and r.undetected == undetected and r.dangerous == dangerous,
                  "(" + String(r.fault_sets) + ", " + String(r.undetected) + ", " + String(r.dangerous) + ")")
    _ = log.check(name + " " + model.name() + " order-2 undetected", u2 == order2, String(u2))
    return r^


def main() raises:
    var log = CheckLog("ft_state_prep_check")
    var st = steane_code()
    var enc = steane_encoder_zero()
    var flg = steane_prep_zero_ft()
    var stab = _stab_flag()

    var rE = _check_one(log, "encoder", enc, 0, FaultModel.circuit_level(), 136, 136, 29, 8331)
    _ = _check_one(log, "encoder", enc, 0, FaultModel.xz_only(), 61, 61, 8, 1707)
    var rF = _check_one(log, "flagged", flg, 1, FaultModel.circuit_level(), 183, 90, 0, 7735)
    _ = _check_one(log, "flagged", flg, 1, FaultModel.xz_only(), 81, 54, 0, 1702)
    var rN = _check_one(log, "stabilizer flag", stab, 1, FaultModel.circuit_level(), 198, 122, 21, 9466)
    _ = _check_one(log, "stabilizer flag", stab, 1, FaultModel.xz_only(), 87, 65, 6, 2200)

    _ = log.check("encoder is not 1-fault tolerant", not rE.ok)
    _ = log.check("flagged preparation is 1-fault tolerant", rF.ok)
    _ = log.check("stabilizer flag is not 1-fault tolerant", not rN.ok)
    _ = log.check("encoder witness is the initial X on qubit 3", rE.witness.find("init q3: X") >= 0, rE.witness)
    _ = log.check("flagged CX count and depth", rF.operations == 12 and rF.cx_depth == 6)
    _ = log.check("fault-free check alone accepts the stabilizer flag",
                  valid_stabilizer_state_preparation(Circuit(1), stab, st, 1, 1).ok)
    print(rF.to_json())
    log.finish()
