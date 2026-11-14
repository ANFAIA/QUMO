"""Tests harness/texts.mojo: every constant equals tests/golden/deepagents_golden.json."""

from std.testing import assert_equal, assert_true

from qumo.ai.json import JsonValue, parse_json

from qumo_agent.harness.texts import (
    text_table, count_parameters, MEDIA_EXTENSIONS, MEDIA_TYPES, MEDIA_MIMES,
)

comptime GOLDEN = "agents/mojo/tests/golden/deepagents_golden.json"


def _read(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def _lookup(table: List[Tuple[String, String]], name: String) raises -> String:
    for i in range(len(table)):
        if table[i][0] == name:
            return table[i][1]
    raise Error("texts.mojo has no constant " + name)


def main() raises:
    var g = parse_json(_read(GOLDEN))
    var table = text_table()
    var n = 0
    var texts = g.get("texts")
    for i in range(len(texts)):
        var name = texts.key_at(i)
        assert_equal(_lookup(table, name), texts.at(i).as_string(), name)
        n += 1
    var schemas = g.get("schemas")
    for i in range(len(schemas)):
        var name = schemas.key_at(i)
        # Compact JSON of the parsed value: same keys in the same order, same values.
        assert_equal(parse_json(_lookup(table, name)).to_json(), schemas.at(i).to_json(), name)
        n += 1
    var cp = g.get("count_parameters")
    for i in range(len(cp)):
        var tool = cp.key_at(i)
        assert_equal(parse_json(count_parameters(tool)).to_json(), cp.at(i).to_json(), tool)
        n += 1
    assert_equal(count_parameters("echo"), "")
    var media = g.get("media")
    var exts = materialize[MEDIA_EXTENSIONS]()
    var types = materialize[MEDIA_TYPES]()
    var mimes = materialize[MEDIA_MIMES]()
    assert_equal(len(exts), len(media))
    for i in range(len(media)):
        assert_equal(exts[i], media.key_at(i))
        assert_equal(types[i], media.at(i).at(0).as_string())
        assert_equal(mimes[i], media.at(i).at(1).as_string())
    # Every constant in the table comes from the golden (no stale extras).
    assert_equal(len(table), n)
    assert_true(n > 90)
    print("test_harness_texts OK")
