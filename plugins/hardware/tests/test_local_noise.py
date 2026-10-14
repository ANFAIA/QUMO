"""Vendor-neutral noisy local emulator: `qumo-sim:<topology>[/<basis>]+noise:<modality>`."""

from __future__ import annotations

import math

import pytest
from conftest import BELL

from qumo_hardware import runner
from qumo_hardware.errors import E_UNKNOWN_BACKEND, HardwareError
from qumo_hardware.local import NOISE_PRESETS, inject_noise, parse_noise, parse_sim_backend

SHOTS = 20000


def _odd(counts: dict) -> float:
    return sum(v for k, v in counts.items() if k.count("1") % 2 == 1) / sum(counts.values())


def test_presets_mirror_hardware_profiles():
    # qumo/exec/resources.mojo HardwareProfile presets (1q, 2q, readout)
    assert NOISE_PRESETS["superconducting"] == (2.5e-4, 3e-3, 1e-2)
    assert NOISE_PRESETS["trapped_ion"] == (3e-5, 1e-3, 3e-3)
    assert NOISE_PRESETS["neutral_atom"] == (5e-4, 5e-3, 1e-2)
    n = parse_noise("custom(1e-4, 2e-3, 0.01)")
    assert (n["modality"], n["p1"], n["p2"], n["readout"]) == ("custom", 1e-4, 2e-3, 0.01)


def test_bell_parity_error_matches_theory(qumo):
    """One CZ with depolarize2(p2) then 1q Cliffords: a uniformly random non-identity
    2-qubit Pauli flips the ZZ parity with probability 8/15; readout flips add
    r = 2 pm (1 - pm). P(odd) = q (1 - r) + r (1 - q), q = 8 p2 / 15."""
    p2, pm = 0.06, 0.02
    out = runner.run(BELL, "local", f"qumo-sim:line:2/sc_cz+noise:custom(0,{p2},{pm})", shots=SHOTS, seed=11)
    assert out["compiled"]["two_qubit_gates"] == 1
    assert out["finishing"]["noisy"] is True
    assert out["finishing"]["noise_channels_inserted"] == {"depolarize1": 0, "depolarize2": 1, "x_error": 2}
    q = 8 * p2 / 15
    r = 2 * pm * (1 - pm)
    expect = q * (1 - r) + r * (1 - q)
    sigma = math.sqrt(expect * (1 - expect) / SHOTS)
    got = _odd(out["counts"])
    assert abs(got - expect) <= 3 * sigma, (got, expect, sigma)


def test_bell_fidelity_drops_with_modality(qumo):
    ideal = runner.run(BELL, "local", "qumo-sim:line:2/sc_cz", shots=SHOTS, seed=5)
    assert _odd(ideal["counts"]) == 0.0
    sc = runner.run(BELL, "local", "qumo-sim:line:2/sc_cz+noise:superconducting", shots=SHOTS, seed=5)
    ion = runner.run(BELL, "local", "qumo-sim:all:2/ion_zz+noise:trapped_ion", shots=SHOTS, seed=5)
    p_sc, p_ion = _odd(sc["counts"]), _odd(ion["counts"])
    # leading order: readout 2 pm + 8/15 p2 (+ small 1q terms)
    for got, (p1, p2, pm) in ((p_sc, NOISE_PRESETS["superconducting"]), (p_ion, NOISE_PRESETS["trapped_ion"])):
        lead = 2 * pm + 8 * p2 / 15
        sigma = math.sqrt(lead / SHOTS)
        assert lead - 3 * sigma - 1e-3 <= got <= lead + 3 * sigma + 8 * p1 + 1e-3, (got, lead)
    assert p_ion < p_sc


def test_deterministic_per_seed(qumo):
    be = "qumo-sim:line:3/sc_cz+noise:neutral_atom"
    src = BELL.replace("qreg q[2];", "qreg q[3];").replace("creg c[2];", "creg c[3];").replace(
        "measure q[1] -> c[1];", "cx q[1],q[2];\nmeasure q[1] -> c[1];\nmeasure q[2] -> c[2];")
    a = runner.run(src, "local", be, shots=4000, seed=9)["counts"]
    b = runner.run(src, "local", be, shots=4000, seed=9)["counts"]
    c = runner.run(src, "local", be, shots=4000, seed=10)["counts"]
    assert a == b
    assert a != c


def test_noise_free_backend_unchanged(qumo):
    base = runner.run(BELL, "local", "qumo-sim:line:2/sc_cz", shots=3000, seed=4)
    zero = runner.run(BELL, "local", "qumo-sim:line:2/sc_cz+noise:custom(0,0,0)", shots=3000, seed=4)
    assert base["counts"] == zero["counts"]
    assert "noisy" not in base["finishing"]
    plan = runner.plan(BELL, "local", "qumo-sim:line:2/sc_cz")
    assert "#pragma qumo" not in plan.payload


def test_unknown_modality_raises():
    for bad in ("qumo-sim:line:2+noise:ibm_fez", "qumo-sim+noise:custom(1,2)", "qumo-sim+noise:custom(0.9,0,0)",
                "qumo-sim+noise:custom(a,b,c)"):
        with pytest.raises(HardwareError) as exc:
            parse_sim_backend(bad)
        assert exc.value.code == E_UNKNOWN_BACKEND, bad


def test_plan_is_local_and_free(qumo):
    p = runner.plan(BELL, "local", "qumo-sim:grid:2x2/sc_cz+noise:superconducting", shots=1000)
    d = p.to_dict()
    assert d["local"] is True and d["requires_confirmation"] is False
    assert d["cost"]["unit"] == "free" and d["cost"]["amount"] == 0.0
    assert d["target"]["errors"] == {"1q": 2.5e-4, "2q": 3e-3, "readout": 1e-2}
    assert d["target"]["extras"]["noise"]["modality"] == "superconducting"
    assert any("not a device calibration" in n for n in d.get("notes", []))
    assert "#pragma qumo depolarize2" in p.payload and "#pragma qumo x_error" in p.payload


def test_backend_listed():
    rows = runner.list_backends("local")
    assert any("+noise:<modality>" in r["name"] for r in rows)


def test_inject_noise_skips_gate_definitions_and_headers():
    text = ('OPENQASM 2.0;\ninclude "qelib1.inc";\ngate g2 a,b\n{\n  cx a,b;\n}\nqreg q[3];\ncreg c[3];\n'
            "barrier q[0],q[1];\nccx q[0],q[1],q[2];\ng2 q[0],q[2];\nmeasure q[2] -> c[0];\n")
    out, ins = inject_noise(text, parse_noise("custom(0.001,0.01,0.02)"))
    assert "  cx a,b;\n#pragma" not in out  # nothing inside the gate body
    assert ins == {"depolarize1": 3, "depolarize2": 1, "x_error": 1}
    assert out.index("#pragma qumo x_error(0.02) q[2]") < out.index("measure q[2] -> c[0];")


def test_error_pattern_sampling_matches_qumo_trajectories(qumo):
    """sample_noisy draws the same distribution as QUMO's per-shot pragma trajectories."""
    from qumo_hardware.local import sample_noisy

    src = BELL.replace("qreg q[2];", "qreg q[3];").replace("creg c[2];", "creg c[3];").replace(
        "measure q[1] -> c[1];", "ry(0.7) q[2];\ncx q[1],q[2];\nmeasure q[1] -> c[1];\nmeasure q[2] -> c[2];")
    p = runner.plan(src, "local", "qumo-sim:line:3/sc_cz+noise:custom(0.02,0.08,0.03)", shots=8000)
    fast, info = sample_noisy(qumo, p.payload, 8000, 21)
    assert info["distinct_error_patterns"] > 10 and info["readout_flips_on_clbits"] == 3
    slow = {str(k): int(v) for k, v in dict(qumo.sample(qumo.load_qasm(p.payload), 8000, 22)).items()}
    keys = set(fast) | set(slow)
    tvd = 0.5 * sum(abs(fast.get(k, 0) - slow.get(k, 0)) / 8000 for k in keys)
    assert tvd < 0.03, (tvd, fast, slow)
