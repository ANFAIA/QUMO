"""Portions ported from deepagents 0.7.32 and langchain 1.4.8 / langchain-core 1.7.4
(Copyright (c) LangChain, Inc., MIT License; see agents/mojo/LICENSES/langchain-MIT.txt).

The `task` tool and subagent isolation (DESIGN.md 2.7.10, 1.4 "Subagent isolation",
[sub §3], [fs §8]).
"""

from qumo.ai.json import JsonWriter

from qumo_agent.core.messages import Message, ROLE_ASSISTANT
from qumo_agent.core.toolspec import ToolSpec, TOOL_HARNESS
from qumo_agent.util.pyfmt import py_rstrip
from qumo_agent.harness._fmt import fmt1, fmt2
from qumo_agent.harness.env import AgentDef, SubagentResult
from qumo_agent.harness.texts import (
    TASK_DESCRIPTION_TEMPLATE, TASK_PARAMETERS, TASK_UNKNOWN_SUBAGENT_ERROR,
    GENERAL_PURPOSE_DESCRIPTION, DEFAULT_SUBAGENT_PROMPT, COUNT_PARAMETERS_TASK,
)
from qumo_agent.harness.vfs import VfsTx, vfs_from_json

comptime GENERAL_PURPOSE = "general-purpose"
comptime SUBAGENT_RECURSION_LIMIT = 9999


def count_json_for(name: String, description: String, count_parameters: String) -> String:
    """`{"name", "description", "parameters"}` as Python's token counter sees a tool."""
    var w = JsonWriter()
    w.begin_object()
    w.kv_string("name", name)
    w.kv_string("description", description)
    w.key("parameters")
    w.raw(count_parameters)
    w.end_object()
    return w.finish()


def task_tool_spec(names: List[String], descriptions: List[String]) -> ToolSpec:
    """The `task` tool; `names` lists general-purpose first, then the named subagents in
    spec order."""
    var lines = List[String]()
    for i in range(len(names)):
        lines.append("- " + names[i] + ": " + descriptions[i])
    var desc = fmt1(TASK_DESCRIPTION_TEMPLATE, "available_agents", String("\n").join(lines))
    return ToolSpec("task", desc, TASK_PARAMETERS, TOOL_HARNESS, False,
                    count_json_for("task", desc, COUNT_PARAMETERS_TASK))


def general_purpose_def(parent: AgentDef) -> AgentDef:
    return AgentDef(GENERAL_PURPOSE, GENERAL_PURPOSE_DESCRIPTION, DEFAULT_SUBAGENT_PROMPT,
                    parent.model, List[String](), SUBAGENT_RECURSION_LIMIT)


def unknown_subagent_text(subagent_type: String, allowed: List[String]) -> String:
    """A.4 error; `allowed` back-quoted, joined with ", " (general-purpose first)."""
    var quoted = List[String]()
    for a in allowed:
        quoted.append("`" + a + "`")
    return fmt2(TASK_UNKNOWN_SUBAGENT_ERROR, "subagent_type", subagent_type, "allowed_types", String(", ").join(quoted))


def child_replay_at(parent_at: String, call_id: String) -> String:
    if parent_at.byte_length() == 0:
        return call_id
    return parent_at + "/" + call_id


def subagent_final_text(messages: List[Message]) -> String:
    """The last assistant message whose text has a non-empty rstrip(), right-stripped; "" if none."""
    var i = len(messages) - 1
    while i >= 0:
        if messages[i].role == ROLE_ASSISTANT:
            var t = py_rstrip(messages[i].text())
            if t.byte_length() > 0:
                return t
        i -= 1
    return String()


def merge_subagent_files(res: SubagentResult, mut tx: VfsTx) raises:
    """The child's whole file dict into `tx`, key by key (right wins), when it changed.
    Files the child deleted are not deleted in the parent [fs §8]."""
    if not res.files_changed:
        return
    var child = vfs_from_json(res.files_json)
    for item in child.files.items():
        tx.set(item.key, item.value)
