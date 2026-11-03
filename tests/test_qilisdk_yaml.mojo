"""Tests for qumo.ai.yaml_io: flow-style YAML for Pauli sums, and refusal of unsafe YAML."""

from std.testing import assert_equal, assert_true

from qumo.ai.json import parse_json
from qumo.ai.yaml_io import pauli_sum_from_yaml, pauli_sum_to_yaml, refuse_unsafe, yaml_load_flow
from qumo.ir.pauli import PauliSum


def _three_qubit_sum() raises -> PauliSum:
    """A constant, a Z, an XZY term and a fractional coefficient on 3 qubits."""
    var h = PauliSum()
    h.add_label(0.75, "III")
    h.add_label(-1.5, "IIZ")
    h.add_label(0.125, "XZY")
    h.add_label(1.0 / 3.0, "ZXI")
    return h^


def _refusal(text: String) -> String:
    """The message pauli_sum_from_yaml raises for `text`, or "" when it accepts it."""
    try:
        _ = pauli_sum_from_yaml(text)
    except e:
        return String(e)
    return String()


def _expect_refused(text: String, frag: String) raises:
    var msg = _refusal(text)
    assert_true(msg.startswith("QM-E915: "), "expected QM-E915 for " + text + ", got: " + msg)
    assert_true(msg.find(frag) >= 0, "expected '" + frag + "' in: " + msg)
    assert_true(msg.find("Hint: ") > 0, "no hint in: " + msg)


def test_round_trip() raises:
    var h = _three_qubit_sum()
    var text = pauli_sum_to_yaml(h)
    var back = pauli_sum_from_yaml(text)
    assert_equal(len(back), len(h))
    assert_equal(back.num_qubits(), 3)
    for t in h.terms:
        var found = False
        for u in back.terms:
            if u.x == t.x and u.z == t.z:
                found = True
                assert_true(abs(u.coeff - t.coeff) <= 1e-15, "coefficient drift in " + text)
        assert_true(found, "term lost in " + text)


def test_round_trip_edge_widths() raises:
    """An empty sum and a constant-only sum both need a width of 1 in the document."""
    var empty = PauliSum()
    var t0 = pauli_sum_to_yaml(empty)
    assert_equal(t0, '{"num_qubits":1,"terms":[]}')
    assert_equal(len(pauli_sum_from_yaml(t0)), 0)

    var c = PauliSum()
    c.add_label(2.5, "I")
    var t1 = pauli_sum_to_yaml(c)
    assert_equal(t1, '{"num_qubits":1,"terms":[{"coeff":2.5,"paulis":"I"}]}')
    var back = pauli_sum_from_yaml(t1)
    assert_equal(len(back), 1)
    assert_equal(back.terms[0].coeff, 2.5)
    assert_equal(back.terms[0].x, 0)
    assert_equal(back.terms[0].z, 0)


def test_output_is_parseable_with_two_keys() raises:
    var text = pauli_sum_to_yaml(_three_qubit_sum())
    var v = parse_json(text)
    var keys = v.keys()
    assert_equal(len(keys), 2)
    assert_equal(keys[0], "num_qubits")
    assert_equal(keys[1], "terms")
    assert_equal(v.get("num_qubits").as_int(), 3)
    assert_equal(len(v.get("terms")), 4)
    assert_true(text.startswith('{"num_qubits":3,"terms":[{"coeff":0.75,"paulis":"III"},'), text)


def test_refuses_tags_anchors_and_aliases() raises:
    _expect_refused('!DigitalPropagation {"num_qubits": 1, "terms": []}', "tag !DigitalPropagation:")
    assert_equal(
        _refusal('!DigitalPropagation {"num_qubits": 1, "terms": []}'),
        "QM-E915: tag !DigitalPropagation: QUMO does not read tags, because a tag can name a class. "
        + "Hint: write the document in flow style without tags.",
    )
    _expect_refused('!!python/object/apply:os.system ["echo x"]', "tag !!python/object/apply:os.system:")
    _expect_refused('&a {"num_qubits": 1, "terms": []}', "anchor &a:")
    _expect_refused('{"num_qubits": 1, "terms": []} *a', "alias *a:")
    _expect_refused('{"num_qubits": 1, "terms": [], "x": *a}', "alias *a:")
    _expect_refused('{"num_qubits": 1, "terms": [], "k": "x\\\\", "t": !tag}', "tag !tag:")


def test_refuses_block_style_separators_and_scalars() raises:
    _expect_refused("num_qubits: 2\nterms: []", "block style")
    _expect_refused("{} --- {}", "document separator ---")
    _expect_refused("--- {\"num_qubits\": 1, \"terms\": []}", "document separator ---")
    _expect_refused("{} {}", "second top-level document")
    _expect_refused("3", "bare scalar")
    _expect_refused("# note\n{\"num_qubits\": 1, \"terms\": []}", "a comment before the document")
    _expect_refused("%YAML 1.2\n{\"num_qubits\": 1, \"terms\": []}", "a directive")
    _expect_refused("", "empty document")


def test_tag_and_anchor_characters_inside_strings_are_accepted() raises:
    var text = '{"num_qubits": 1, "terms": [{"coeff": 1.0, "paulis": "Z"}], "note": "a!b&c*d"}'
    assert_equal(len(pauli_sum_from_yaml(text)), 1)
    assert_equal(yaml_load_flow(text).get("note").as_string(), "a!b&c*d")

    # escaped quotes do not end the string early, and an escaped backslash does not escape the quote after it
    var esc = '{"num_qubits": 1, "terms": [], "note": "say \\"!x\\" &y", "k": "x\\\\", "t": "!*&"}'
    assert_equal(len(pauli_sum_from_yaml(esc)), 0)
    assert_equal(yaml_load_flow(esc).get("note").as_string(), 'say "!x" &y')
    assert_equal(yaml_load_flow(esc).get("k").as_string(), "x\\")
    assert_equal(len(pauli_sum_from_yaml('{"num_qubits": 1, "terms": [], "k": "\\"!"}')), 0)


def test_wrong_shapes_are_refused() raises:
    _expect_refused('{"num_qubits": 3, "terms": [{"coeff": 1.0, "paulis": "ZZ"}]}', "terms[0].paulis")
    _expect_refused('{"num_qubits": 3, "terms": [{"coeff": 1.0, "paulis": "XQZ"}]}', "not I, X, Y or Z")
    _expect_refused('{"num_qubits": 0, "terms": []}', "num_qubits")
    _expect_refused('{"num_qubits": 65, "terms": []}', "num_qubits")
    _expect_refused('{"num_qubits": 2.5, "terms": []}', "num_qubits")
    _expect_refused('{"num_qubits": 2}', "key 'terms'")
    _expect_refused('{"num_qubits": 2, "terms": [{"coeff": "1.0", "paulis": "ZZ"}]}', "terms[0].coeff")
    _expect_refused('{"num_qubits": 1, "terms": [{"coeff": 1e999, "paulis": "Z"}]}', "terms[0].coeff")


def test_writer_refuses_a_non_finite_coefficient() raises:
    var h = PauliSum()
    h.add_label(Float64(1.0) / Float64(0.0), "Z")
    try:
        _ = pauli_sum_to_yaml(h)
        raise Error("wrote an infinite coefficient")
    except e:
        assert_true(String(e).startswith("QM-E913"), String(e))


def test_generic_flow_reader() raises:
    var v = yaml_load_flow('\n  {"a": [1, 2.5, "x", true, null]}\n')
    assert_equal(len(v.get("a")), 5)
    assert_equal(v.get("a").at(1).as_float(), 2.5)
    refuse_unsafe('\n  {"num_qubits": 1,\n "terms": []}\n')
    try:
        refuse_unsafe("!x {}")
        raise Error("accepted a tag")
    except e:
        assert_true(String(e).startswith("QM-E915"), String(e))
    assert_equal(len(pauli_sum_from_yaml('\n  {"num_qubits": 1,\n "terms": []}\n')), 0)


def main() raises:
    test_round_trip()
    test_round_trip_edge_widths()
    test_output_is_parseable_with_two_keys()
    test_refuses_tags_anchors_and_aliases()
    test_refuses_block_style_separators_and_scalars()
    test_tag_and_anchor_characters_inside_strings_are_accepted()
    test_wrong_shapes_are_refused()
    test_writer_refuses_a_non_finite_coefficient()
    test_generic_flow_reader()
    print("test_qilisdk_yaml OK")
