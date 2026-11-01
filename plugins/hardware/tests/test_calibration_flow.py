"""Calibration and angle bounds reach QUMO's compiler through the neutral TargetSpec."""

from __future__ import annotations

import math

import pytest

from qumo_hardware.base import TargetSpec
from qumo_hardware.compile import _sub_target, compile_for_target

BELL = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg c[2]; h q[0]; cx q[0],q[1]; measure q -> c;'


def line_spec(**kw) -> TargetSpec:
    base = dict(name="line5", provider="test", num_qubits=5, edges=[(0, 1), (1, 2), (2, 3), (3, 4)],
                basis=["rz", "sx", "x", "cz"], errors={"2q": 0.01, "readout": 0.01})
    base.update(kw)
    return TargetSpec(**base)


def test_spec_with_no_data_passes_nothing():
    assert line_spec().calibration() is None


def test_calibration_dict():
    spec = line_spec(edge_errors={"0-1": 0.05, "1-2": None}, readout_errors=[0.01, None, 0.02, 0.01, 0.01],
                     angle_bounds={"rzz": (0.0, 1.5)})
    cal = spec.calibration()
    assert cal["edge_errors"] == {"0-1": 0.05}
    assert cal["readout"][1] == 0.01  # unknown qubit: the median
    assert cal["angle_bounds"] == {"rzz": (0.0, 1.5)}


def test_sub_target_keeps_the_calibration():
    spec = line_spec(edge_errors={"2-3": 0.003, "3-4": 0.004}, readout_errors=[0.1, 0.2, 0.3, 0.4, 0.5],
                     angle_bounds={"rzz": (0.0, 1.5)})
    sub = _sub_target(spec, [2, 3, 4])
    assert sub.edge_errors == {"0-1": 0.003, "1-2": 0.004}
    assert sub.readout_errors == [0.3, 0.4, 0.5]
    assert sub.angle_bounds == {"rzz": (0.0, 1.5)}


def test_calibrated_compile_lands_on_the_good_edge():
    pytest.importorskip("qumo_py")
    spec = line_spec(edge_errors={"0-1": 0.2, "1-2": 0.2, "2-3": 0.2, "3-4": 0.001})
    out = compile_for_target(BELL, spec, level=2, seed=1, region="none")
    assert set(out.measure_map.values()) == {3, 4}
    plain = compile_for_target(BELL, spec, level=2, seed=1, region="none", calibrate=False)
    assert set(plain.measure_map.values()) != {3, 4}


def test_angle_bounds_hold_after_compilation():
    pytest.importorskip("qumo_py")
    qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg c[2]; rzz(2.5) q[0],q[1]; measure q -> c;'
    spec = line_spec(basis=["rz", "sx", "x", "cz", "rzz"], angle_bounds={"rzz": (0.0, math.pi / 2)})
    out = compile_for_target(qasm, spec, level=1, seed=1)
    import re

    calls = [ln for ln in out.qasm.splitlines() if ln.startswith("rzz(")]
    angles = [float(eval(re.match(r"rzz\(([^)]*)\)", ln).group(1), {"pi": math.pi})) for ln in calls]
    assert angles and all(0 <= a <= math.pi / 2 + 1e-9 for a in angles)
