"""Portions ported from deepagents 0.7.32 and langchain 1.4.8 / langchain-core 1.7.4
(Copyright (c) LangChain, Inc., MIT License; see agents/mojo/LICENSES/langchain-MIT.txt).

Tool-call patching (DESIGN.md 2.7.8, [ctx §9]): deepagents' PatchToolCallsMiddleware
(once per run, before the agent) and langchain's `_patch_invalid_tool_calls` (every model
request, request copy only). Both are idempotent.
"""

from qumo_agent.core.messages import Message, ROLE_ASSISTANT, ROLE_TOOL, tool_message
from qumo_agent.core.toolspec import error_outcome
from qumo_agent.harness._fmt import fmt2
from qumo_agent.harness.texts import ERR_MALFORMED_TOOL_CALL, ERR_DANGLING_TOOL_CALL


def _answered(messages: List[Message]) -> List[String]:
    var ids = List[String]()
    for m in messages:
        if m.role == ROLE_TOOL:
            ids.append(m.tool_call_id)
    return ids^


def _contains(ids: List[String], id: String) -> Bool:
    for x in ids:
        if x == id:
            return True
    return False


def _name_or_unknown(name: String) -> String:
    return name if name.byte_length() > 0 else String("unknown")


def malformed_call_message(name: String, call_id: String) -> Message:
    """The error tool message for an invalid call (A.9 ERR_MALFORMED_TOOL_CALL)."""
    var n = _name_or_unknown(name)
    return tool_message(call_id, n, error_outcome(fmt2(ERR_MALFORMED_TOOL_CALL, "name", n, "tool_call_id", call_id)))


def patch_dangling_tool_calls(messages: List[Message]) -> List[Message]:
    """Every valid and invalid call id with no tool message anywhere gets an error tool
    message right after its assistant message (valid calls first)."""
    var answered = _answered(messages)
    var out = List[Message]()
    for m in messages:
        out.append(m.copy())
        if m.role != ROLE_ASSISTANT:
            continue
        for c in m.tool_calls:
            if _contains(answered, c.id):
                continue
            var n = _name_or_unknown(c.name)
            out.append(tool_message(c.id, n, error_outcome(fmt2(ERR_DANGLING_TOOL_CALL, "name", n, "tool_call_id", c.id))))
        for c in m.invalid_tool_calls:
            if _contains(answered, c.id):
                continue
            out.append(malformed_call_message(c.name, c.id))
    return out^


def patch_invalid_tool_calls(messages: List[Message]) -> List[Message]:
    """Every unanswered invalid call gets the malformed-arguments message right after its
    assistant message."""
    var answered = _answered(messages)
    var out = List[Message]()
    for m in messages:
        out.append(m.copy())
        if m.role != ROLE_ASSISTANT:
            continue
        for c in m.invalid_tool_calls:
            if _contains(answered, c.id):
                continue
            out.append(malformed_call_message(c.name, c.id))
            answered.append(c.id)
    return out^
