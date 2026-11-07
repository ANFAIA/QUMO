"""Tests harness/validate.mojo against pydantic through LangGraph
(tests/golden/pydantic_errors.json, Appendix C.2)."""

from std.testing import assert_equal, assert_true

from qumo.ai.json import parse_json

from qumo_agent.harness.validate import validate_args, task_extra_key_errors, invocation_error_text

comptime GOLDEN = "agents/mojo/tests/golden/pydantic_errors.json"


def _read(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def main() raises:
    var g = parse_json(_read(GOLDEN))
    var cases = g.get("cases")
    for i in range(len(cases)):
        var c = cases.at(i)
        var label = c.get("tool").as_string() + " " + c.get("args").to_json()
        var chk = validate_args(c.get("parameters").to_json(), c.get("args"))
        var exp = c.get("errors")
        if len(chk.errors) != len(exp):
            print(label, String("|").join(chk.errors))
        assert_equal(len(chk.errors), len(exp), label)
        for j in range(len(exp)):
            assert_equal(chk.errors[j], exp.at(j).as_string(), label)
        assert_equal(chk.ok, len(exp) == 0, label)
        if chk.ok:
            assert_equal(chk.args.to_json(), c.get("coerced").to_json(), label)
    var extra = task_extra_key_errors(parse_json("{\"description\":\"d\",\"subagent_type\":\"x\",\"prompt\":1,\"a\":2}"))
    assert_equal(len(extra), 2)
    assert_equal(extra[0], "a: Value error, Unexpected argument 'a'; put all instructions for the subagent in `description`.")
    assert_equal(extra[1], "prompt: Value error, Unexpected argument 'prompt'; put all instructions for the subagent in `description`.")
    var errs: List[String] = ["text: Field required"]
    assert_equal(invocation_error_text("qumo_echo", parse_json("{\"bad\": 1}"), errs),
                 "Error invoking tool 'qumo_echo' with kwargs {'bad': 1} with error:\n text: Field required\n Please fix the error and try again.")
    print("test_harness_validate OK (" + String(len(cases)) + " cases)")
