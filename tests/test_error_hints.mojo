"""Error hints and tool-schema enums say what their validators accept.

Each test calls a validator with a value it refuses and with every value it
accepts, and checks that the hint (the raised text and the catalogue entry of
`qumo.ai.diagnostics`, from which docs/ERRORS.md is generated) names the accepted
values: QM-E707 (decoder names), QM-E709 (noise model), QM-E736 (sliding
window), QM-E722 (code names) and the `qec_style` enum of `qumo_ft_encode`.
"""

from std.testing import assert_equal, assert_true

from qumo.ai.describe import describe_error_json
from qumo.ai.diagnostics import default_hint
from qumo.ai.ft_tools import register_ft_tools
from qumo.ai.tools import ToolRegistry
from qumo.core.json import parse_json
from qumo.ft.code import code_by_name, code_names
from qumo.noise.characterization import NoiseModel
from qumo.qec.codes import NoiseParams, generate_code
from qumo.qec.dem import DetectorErrorModel, detector_error_model
from qumo.qec.experiments import canonical_decoder_name, memory_experiment
from qumo.qec.window import plan_windows, detector_rounds


def _has(text: String, part: String, what: String) raises:
    assert_true(text.find(part) >= 0, what + " does not mention '" + part + "': " + text)


def _lacks(text: String, part: String, what: String) raises:
    assert_true(text.find(part) < 0, what + " still says '" + part + "': " + text)


# --- QM-E707 ----------------------------------------------------------------
def _decoder_said(name: String) -> String:
    try:
        _ = memory_experiment("repetition", 3, 1, 0.01, 64, name, 1, -1)
    except e:
        return String(e)
    return ""


def test_decoder_names() raises:
    var canon: List[String] = ["mwpm", "sparse_blossom", "mwpm_correlated", "mwpm_dense", "uf", "lookup"]
    var aliases: List[String] = [
        "matching", "pymatching", "sparse-blossom", "sparseblossom", "sb", "mwpm-dense", "dense_mwpm", "dense",
        "correlated", "pymatching-correlated", "union_find", "unionfind", "lut", "table",
    ]
    var raised = _decoder_said("no_such_decoder")
    assert_true(raised.startswith("QM-E707"), raised)
    var hint = default_hint("QM-E707")
    for name in canon:
        assert_equal(canonical_decoder_name(name), name)
        assert_equal(_decoder_said(name), "", "decoder " + name + " runs")
        _has(raised, "'" + name + "'", "QM-E707 raised text")
        _has(hint, "'" + name + "'", "QM-E707 catalogue hint")
    for a in aliases:
        assert_true(canonical_decoder_name(a) != "", "alias " + a)
        _has(hint, "'" + a + "'", "QM-E707 catalogue hint")
    assert_equal(canonical_decoder_name("no_such_decoder"), "")


# --- QM-E709 ----------------------------------------------------------------
def _model_said(t1: Float64, t2: Float64, e1: Float64, e2: Float64, ro: Float64) -> String:
    try:
        _ = NoiseModel.uniform(2, t1, t2, e1, e2, ro, 0.035, 0.3, 1.0)
    except e:
        return String(e)
    return ""


def test_noise_model_ranges() raises:
    # accepted: the ends of each interval, and T1 / T2 <= 0 ("infinite")
    assert_equal(_model_said(100.0, 80.0, 0.0, 0.0, 0.0), "")
    assert_equal(_model_said(100.0, 200.0, 0.5, 0.75, 0.5), "")
    assert_equal(_model_said(0.0, 0.0, 0.01, 0.6, 0.01), "")
    assert_equal(_model_said(-1.0, 50.0, 0.01, 0.01, 0.01), "")
    assert_equal(_model_said(100.0, 0.0, 0.01, 0.01, 0.01), "")
    # refused
    var e1 = _model_said(100.0, 80.0, 0.51, 0.01, 0.01)
    var ro = _model_said(100.0, 80.0, 0.01, 0.01, 0.51)
    var e2 = _model_said(100.0, 80.0, 0.01, 0.76, 0.01)
    var t2 = _model_said(100.0, 201.0, 0.01, 0.01, 0.01)
    for said in [e1, ro, e2, t2, _model_said(100.0, 80.0, -0.01, 0.01, 0.01), _model_said(100.0, 80.0, 0.01, -0.01, 0.01)]:
        assert_true(said.startswith("QM-E709"), said)
    _has(e1, "[0, 0.5]", "QM-E709 one-qubit text")
    _has(ro, "[0, 0.5]", "QM-E709 readout text")
    _has(e2, "[0, 0.75]", "QM-E709 two-qubit text")
    _has(t2, "2*T1", "QM-E709 T2 text")
    var hint = default_hint("QM-E709")
    _has(hint, "[0, 0.5]", "QM-E709 catalogue hint")
    _has(hint, "[0, 0.75]", "QM-E709 catalogue hint")
    _has(hint, "T2 <= 2*T1", "QM-E709 catalogue hint")
    _has(hint, "<= 0", "QM-E709 catalogue hint")
    _lacks(hint, "[0, 1]", "QM-E709 catalogue hint")
    _lacks(hint, "T1/T2 > 0", "QM-E709 catalogue hint")
    # the code lives in qumo.noise
    assert_equal(parse_json(describe_error_json("QM-E709")).get_string("category", ""), "noise")
    assert_equal(parse_json(describe_error_json("QM-E710")).get_string("category", ""), "noise")


# --- QM-E736 ----------------------------------------------------------------
def _window_said(dem: DetectorErrorModel, rounds_of: List[Int], window: Int, step: Int) -> String:
    try:
        _ = plan_windows(dem, rounds_of, window, step)
    except e:
        return String(e)
    return ""


def test_window_rule() raises:
    var cc = generate_code("repetition", 3, 6, NoiseParams.uniform(0.01))
    var dem = detector_error_model(cc.circuit)
    dem.detector_coords = cc.detector_coords.copy()
    var rounds_of = detector_rounds(dem)
    var rounds = 0
    for r in rounds_of:
        rounds = max(rounds, r + 1)
    assert_true(rounds >= 6, "rounds " + String(rounds))
    # accepted: 0 < step <= window, and a whole number of steps when window < rounds
    assert_equal(_window_said(dem, rounds_of, rounds - 2, 1), "")
    assert_equal(_window_said(dem, rounds_of, rounds - 2, 2), "")
    assert_equal(_window_said(dem, rounds_of, rounds, 3), "")
    # a window wider than the experiment is one window, whatever the step
    assert_equal(_window_said(dem, rounds_of, rounds + 5, 3), "")
    assert_equal(len(plan_windows(dem, rounds_of, rounds + 5, 3).dems), 1)
    # refused
    var bad: List[String] = [
        _window_said(dem, rounds_of, 3, 0),
        _window_said(dem, rounds_of, 2, 3),
        _window_said(dem, rounds_of, rounds - 3, 2),
    ]
    var hint = default_hint("QM-E736")
    for said in bad:
        assert_true(said.startswith("QM-E736"), said)
        _has(said, "0 < step <= window", "QM-E736 raised text")
        _has(said, "window < rounds", "QM-E736 raised text")
    _has(hint, "0 < step <= window", "QM-E736 catalogue hint")
    _has(hint, "window < rounds", "QM-E736 catalogue hint")
    _lacks(hint, "window <= rounds", "QM-E736 catalogue hint")


# --- QM-E722 ----------------------------------------------------------------
def _code_said(name: String) -> String:
    try:
        _ = code_by_name(name)
    except e:
        return String(e)
    return ""


def test_code_names() raises:
    # (name accepted, how the hint spells it)
    var accepted: List[String] = ["steane", "five_qubit", "css_422", "noncss_422", "helix", "helix_10_2_3", "iceberg_4", "surface_3", "repetition_3", "identity_2"]
    var spelled: List[String] = ["steane", "five_qubit", "css_422", "noncss_422", "helix", "helix_10_2_3", "iceberg_<even k>", "surface_<d>", "repetition_<d>", "identity_<k>"]
    var raised = _code_said("no_such_code")
    assert_true(raised.startswith("QM-E722"), raised)
    var hint = default_hint("QM-E722")
    for i in range(len(accepted)):
        assert_equal(_code_said(accepted[i]), "", "code " + accepted[i])
        _has(code_names(), spelled[i], "code_names()")
        _has(raised, spelled[i], "QM-E722 raised text")
        _has(hint, spelled[i], "QM-E722 catalogue hint")
    # 'helix' must be listed as a name of its own, not only as the prefix of helix_10_2_3
    _has(code_names(), "helix,", "code_names()")
    _has(hint, "helix,", "QM-E722 catalogue hint")
    for alias in ["steane_713", "513", "perfect", "422", "helix_20_2_6"]:
        assert_equal(_code_said(String(alias)), "", "alias " + String(alias))
    assert_true(_code_said("iceberg_3").startswith("QM-E"), "odd Iceberg k")


# --- qec_style of qumo_ft_encode -----------------------------------------------
def test_ft_encode_qec_style_enum() raises:
    var reg = ToolRegistry()
    register_ft_tools(reg)
    var schema = parse_json(reg.tools[reg.find("qumo_ft_encode")].input_schema)
    var en = schema.get("properties").get("qec_style").get("enum")
    var listed = List[String]()
    for i in range(len(en)):
        listed.append(en.at(i).as_string())
    var q = "OPENQASM 3.0; include \\\"stdgates.inc\\\"; qubit[2] q; bit[2] c; h q[0]; cx q[0], q[1]; c[0] = measure q[0]; c[1] = measure q[1];"
    # every style the policy accepts is in the enum, and the tool takes each one
    var styles: List[String] = ["none", "steane", "knill", "reichardt", "syndrome"]
    for st in styles:
        assert_true(st in listed, "qec_style enum lacks '" + st + "': " + en.to_json())
        var arch = String("iceberg_2") if st == "syndrome" else String("steane")
        var r = reg.call("qumo_ft_encode", parse_json("{\"qasm\":\"" + q + "\",\"architecture\":\"" + arch + "\",\"qec_style\":\"" + st + "\",\"qec_costs\":{\"h\":1,\"cx\":1}}"))
        assert_true(not r.is_error, st + ": " + r.structured)
    assert_equal(len(listed), len(styles))
    # reichardt inserts cycles and the encoded circuit still gives the Bell statistics
    var rr = reg.call("qumo_ft_encode", parse_json("{\"qasm\":\"" + q + "\",\"architecture\":\"steane\",\"qec_style\":\"reichardt\",\"qec_costs\":{\"h\":1,\"cx\":1},\"shots\":100}"))
    assert_true(not rr.is_error, rr.structured)
    var out = parse_json(rr.structured)
    assert_true(out.get("resources").get("logical").get_int("qec_cycles", 0) > 0, rr.structured)
    assert_true(out.get("emulation").get_float("total_variation_distance", 1.0) < 0.25, rr.structured)
    # a style outside the enum is refused, and the text names the five
    var bad = reg.call("qumo_ft_encode", parse_json("{\"qasm\":\"" + q + "\",\"qec_style\":\"bogus\"}"))
    assert_true(bad.is_error)
    _has(bad.structured, "QM-E727", "unknown style")
    for st in styles:
        _has(bad.structured, st, "QM-E727 text")


def main() raises:
    test_decoder_names()
    test_noise_model_ranges()
    test_window_rule()
    test_code_names()
    test_ft_encode_qec_style_enum()
    print("test_error_hints OK")
