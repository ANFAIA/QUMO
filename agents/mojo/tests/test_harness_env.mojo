"""Tests harness/env.mojo: the subagent child protocol round trip (Appendix B.7)."""

from std.testing import assert_equal, assert_true, assert_false

from qumo.ai.json import parse_json

from qumo_agent.core.stats import zero_stats
from qumo_agent.harness.env import (
    SubagentRequest, SubagentResult, subagent_request_from_json, subagent_result_from_json,
)


def test_request_round_trip() raises:
    var anc: List[String] = ["orchestrator"]
    var files = "{\"/notes.md\":{\"content\":\"x\",\"encoding\":\"utf-8\",\"created_at\":\"2026-11-06T12:00:00+00:00\",\"modified_at\":\"2026-11-06T12:00:00+00:00\"}}"
    var req = SubagentRequest("orchestrator", "m/x", "research-watch", "Find é \"q\"\n", "call_7",
                              anc^, files, "", "call_7")
    var text = req.to_json()
    assert_true(text.startswith("{\"schema\":\"qumo.subagent_request.v1\",\"parent_agent\":\"orchestrator\""))
    assert_false("\n" in text)
    var back = subagent_request_from_json(text)
    assert_equal(back.parent_model, "m/x")
    assert_equal(back.subagent_type, "research-watch")
    assert_equal(back.description, "Find é \"q\"\n")
    assert_equal(back.tool_call_id, "call_7")
    assert_equal(len(back.ancestors), 1)
    assert_equal(back.ancestors[0], "orchestrator")
    assert_equal(back.files_json, files)
    assert_equal(back.replay_at, "call_7")
    var bad = False
    try:
        _ = subagent_request_from_json("{\"schema\":\"other\"}")
    except e:
        bad = String(e).startswith("QA-E120: malformed subagent request")
    assert_true(bad)


def test_result_round_trip() raises:
    var st = zero_stats()
    st.model_calls = 7
    st.tool_calls = 12
    var res = SubagentResult(True, "Report saved", False, "{\"/a\":1}", "", st^, "")
    var text = res.to_json()
    var doc = parse_json(text)
    assert_equal(doc.get("schema").as_string(), "qumo.subagent_result.v1")
    assert_true(doc.get("files").is_null())
    var back = subagent_result_from_json(text)
    assert_true(back.ok)
    assert_equal(back.content, "Report saved")
    assert_false(back.files_changed)
    assert_equal(back.files_json, "")
    assert_equal(back.stats.model_calls, 7)
    assert_equal(back.stats.tool_calls, 12)
    var st2 = zero_stats()
    var res2 = SubagentResult(True, "", True, "{\"/b\":{\"content\":\"y\",\"encoding\":\"utf-8\",\"created_at\":\"t\",\"modified_at\":\"t\"}}", "/tmp/sb", st2^, "")
    var back2 = subagent_result_from_json(res2.to_json())
    assert_true(back2.files_changed)
    assert_true(back2.files_json.startswith("{\"/b\""))
    assert_equal(back2.sandbox, "/tmp/sb")
    var fail = subagent_result_from_json("{\"ok\": false, \"error\": \"QA-E140: x\"}")
    assert_false(fail.ok)
    assert_equal(fail.error, "QA-E140: x")


def main() raises:
    test_request_round_trip()
    test_result_round_trip()
    print("test_harness_env OK")
