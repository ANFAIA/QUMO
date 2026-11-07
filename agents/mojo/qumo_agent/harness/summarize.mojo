"""Portions ported from deepagents 0.7.32 and langchain 1.4.8 / langchain-core 1.7.4
(Copyright (c) LangChain, Inc., MIT License; see agents/mojo/LICENSES/langchain-MIT.txt).

Summarization pieces (DESIGN.md 2.7.7, 3.3, [ctx §1-§7]): the effective message list,
tool-argument truncation, the safe cutoff, the XML transcript, the summary prompt and
message, the history file section and the context-overflow tail clip. The orchestration
(model calls and retries) lives in loop.mojo.
"""

from qumo.ai.json import JsonValue, json_quote

from qumo_agent.core.messages import (
    Message, ToolCall, ROLE_SYSTEM, ROLE_USER, ROLE_ASSISTANT, ROLE_TOOL, user_message,
)
from qumo_agent.core.toolspec import ToolSpec
from qumo_agent.util.jsonx import json_with
from qumo_agent.util.pyfmt import cp_len, cp_head, cp_slice, py_rstrip, py_json_dumps, xml_escape, xml_quoteattr
from qumo_agent.harness._fmt import fmt, fmt1, fmt2
from qumo_agent.harness.eviction import offload_text
from qumo_agent.harness.texts import (
    SUMMARY_PROMPT_TEMPLATE, SUMMARY_WITH_FILE, SUMMARY_WITHOUT_FILE, SUMMARY_HISTORY_SECTION,
    SUMMARY_OVERFLOW_MIDDLE, ARG_TRUNCATION_SUFFIX, OVERFLOW_READ_FILE_NOTICE,
)
from qumo_agent.harness.tokens import count_tokens_list
from qumo_agent.harness.vfs import Vfs, VfsTx

comptime SUMMARY_TRIGGER_TOKENS = 170000
comptime SUMMARY_KEEP_MESSAGES = 6
comptime TRUNCATE_TRIGGER_MESSAGES = 20
comptime TRUNCATE_KEEP_MESSAGES = 20
comptime TRUNCATE_MAX_ARG_CHARS = 2000
comptime _ARG_KEEP_CHARS = 20
comptime _XML_ID_CHARS = 64
comptime _OVERFLOW_READ_KEEP_CHARS = 4000
comptime SUMMARY_SOURCE = "summarization"


@fieldwise_init
struct SummaryEvent(Copyable):
    var cutoff_index: Int
    var summary_message: Message
    var file_path: String            # "" when the history offload failed
    var created_at_index: Int


def apply_event(messages: List[Message], event: Optional[SummaryEvent]) -> List[Message]:
    """The effective list: the summary message, then messages[cutoff:] [ctx §2]."""
    if not event:
        return messages.copy()
    var ev = event.value().copy()
    var out = List[Message]()
    out.append(ev.summary_message.copy())
    if ev.cutoff_index > len(messages):
        return out^
    for i in range(ev.cutoff_index, len(messages)):
        out.append(messages[i].copy())
    return out^


def truncate_tool_args(messages: List[Message]) -> Tuple[List[Message], Bool]:
    """From 20 messages on, string arguments longer than 2000 code points of older
    `write_file`/`edit_file` calls become their first 20 code points + the suffix."""
    if len(messages) < TRUNCATE_TRIGGER_MESSAGES or len(messages) <= TRUNCATE_KEEP_MESSAGES:
        return (messages.copy(), False)
    var cutoff = len(messages) - TRUNCATE_KEEP_MESSAGES
    var out = List[Message]()
    var changed = False
    for i in range(len(messages)):
        var m = messages[i].copy()
        if i < cutoff and m.role == ROLE_ASSISTANT:
            for ci in range(len(m.tool_calls)):
                var name = m.tool_calls[ci].name
                if name != "write_file" and name != "edit_file":
                    continue
                var args = m.tool_calls[ci].args.copy()
                var modified = False
                for k in range(len(args)):
                    var v = args.at(k)
                    if v.is_string():
                        var s = String()
                        try:
                            s = v.as_string()
                        except:
                            pass
                        if cp_len(s) > TRUNCATE_MAX_ARG_CHARS:
                            try:
                                args = json_with(args, args.key_at(k), json_quote(cp_head(s, _ARG_KEEP_CHARS) + ARG_TRUNCATION_SUFFIX))
                                modified = True
                            except:
                                pass
                if modified:
                    m.tool_calls[ci].args = args^
                    changed = True
        out.append(m^)
    return (out^, changed)


def reported_tokens_trigger(messages: List[Message]) -> Bool:
    """The last assistant message reported usage.total_tokens >= 170000."""
    var i = len(messages) - 1
    while i >= 0:
        if messages[i].role == ROLE_ASSISTANT:
            return messages[i].usage_total_tokens >= SUMMARY_TRIGGER_TOKENS
        i -= 1
    return False


def safe_cutoff(messages: List[Message], keep: Int) -> Int:
    """LC _find_safe_cutoff: never split an assistant message from its tool results."""
    if len(messages) <= keep:
        return 0
    var cutoff = len(messages) - keep
    if cutoff >= len(messages) or messages[cutoff].role != ROLE_TOOL:
        return cutoff
    var ids = List[String]()
    var idx = cutoff
    while idx < len(messages) and messages[idx].role == ROLE_TOOL:
        if messages[idx].tool_call_id.byte_length() > 0:
            ids.append(messages[idx].tool_call_id)
        idx += 1
    var i = cutoff - 1
    while i >= 0:
        var m = messages[i].copy()
        if m.role == ROLE_ASSISTANT and len(m.tool_calls) > 0:
            for c in m.tool_calls:
                for x in ids:
                    if c.id.byte_length() > 0 and c.id == x:
                        return i
        i -= 1
    return idx


def state_cutoff(prior: Optional[SummaryEvent], effective_cutoff: Int) -> Int:
    """An effective-list cutoff as a state index (the summary message is effective index 0)."""
    if not prior:
        return effective_cutoff
    return prior.value().cutoff_index + effective_cutoff - 1


def _xml_type(m: Message) -> String:
    if m.role == ROLE_USER:
        return "human"
    if m.role == ROLE_ASSISTANT:
        return "ai"
    if m.role == ROLE_SYSTEM:
        return "system"
    return "tool"


def _display_id(id: String) -> String:
    if cp_len(id) > _XML_ID_CHARS:
        return cp_head(id, _XML_ID_CHARS) + "..."
    return id


def xml_transcript(messages: List[Message]) -> String:
    """get_buffer_string(messages, format="xml") (CU:447-509)."""
    var out = List[String]()
    for m in messages:
        var content_parts = List[String]()
        if m.has_parts:
            for p in m.parts:
                if p.byte_length() > 0:
                    content_parts.append(xml_escape(p))
        elif m.content.byte_length() > 0:
            content_parts.append(xml_escape(m.content))
        var ty = xml_quoteattr(_xml_type(m))
        if m.role == ROLE_ASSISTANT and len(m.tool_calls) > 0:
            var parts = List[String]()
            parts.append("<message type=" + ty + ">")
            if len(content_parts) > 0:
                parts.append("  <content>" + String(" ").join(content_parts) + "</content>")
            for c in m.tool_calls:
                parts.append("  <tool_call id=" + xml_quoteattr(_display_id(c.id)) + " name=" + xml_quoteattr(c.name) + ">"
                             + xml_escape(py_json_dumps(c.args, False, -1)) + "</tool_call>")
            parts.append("</message>")
            out.append(String("\n").join(parts))
        else:
            out.append("<message type=" + ty + ">" + String(" ").join(content_parts) + "</message>")
    return String("\n").join(out)


def summary_prompt_text(transcript: String) -> String:
    """SUMMARY_PROMPT_TEMPLATE.format(messages=transcript).rstrip()."""
    return py_rstrip(fmt1(SUMMARY_PROMPT_TEMPLATE, "messages", transcript))


def overflow_retry_prompt(messages: List[Message]) -> Optional[String]:
    """LC _overflow_retry_prompt: the summary prompt over the transcript with its middle
    cut to fit half its token count; None when nothing fits."""
    var t = xml_transcript(messages)
    var one = List[Message]()
    one.append(user_message(t))
    var tokens = count_tokens_list(one, List[ToolSpec](), False)
    var budget = tokens // 2
    var denom = tokens if tokens > 1 else 1
    var keep = cp_len(t) * budget // denom
    if keep <= 0:
        return None
    var head = keep // 4
    var tail = keep - head
    var n = cp_len(t)
    var cut = cp_head(t, head) + SUMMARY_OVERFLOW_MIDDLE + cp_slice(t, n - tail, n)
    return summary_prompt_text(cut)


def summary_message(summary: String, file_path: String) -> Message:
    """The summary user message (A.7), tagged lc_source = "summarization"."""
    var text: String
    if file_path.byte_length() > 0:
        text = fmt2(SUMMARY_WITH_FILE, "file_path", file_path, "summary", summary)
    else:
        text = fmt1(SUMMARY_WITHOUT_FILE, "summary", summary)
    var m = user_message(text)
    m.lc_source = SUMMARY_SOURCE
    return m^


def history_section(timestamp: String, messages: List[Message]) -> String:
    """One section of /conversation_history/<session>.md; earlier summary messages are left out."""
    var kept = List[Message]()
    for m in messages:
        if m.role == ROLE_USER and m.lc_source == SUMMARY_SOURCE:
            continue
        kept.append(m.copy())
    return fmt2(SUMMARY_HISTORY_SECTION, "timestamp", timestamp, "xml", xml_transcript(kept))


def _call_path(messages: List[Message], call_id: String) -> String:
    """The `file_path` (else `path`) argument of the read_file call `call_id`, or ""."""
    for m in messages:
        if m.role != ROLE_ASSISTANT:
            continue
        for c in m.tool_calls:
            if c.id != call_id or c.name != "read_file":
                continue
            var keys: List[String] = ["file_path", "path"]
            for key in keys:
                var v = c.args.get(key)
                if v.is_string():
                    try:
                        var s = v.as_string()
                        if s.byte_length() > 0:
                            return s
                    except:
                        pass
            return String()
    return String()


def clip_tail_for_overflow(messages: List[Message], base: Vfs, mut tx: VfsTx, now: String) -> Tuple[List[Message], List[Int]]:
    """_clip_overflow_tail with keep ("tokens", 1): the trailing run of tool messages is
    always clipped. A read_file result keeps its first 4,000 code points plus the A.12
    notice; any other is offloaded to /large_tool_results/ like an evicted result. Every
    replacement is a plain-string message. Returns (clipped list, indices replaced)."""
    var out = messages.copy()
    var replaced = List[Int]()
    if len(messages) == 0 or messages[len(messages) - 1].role != ROLE_TOOL:
        return (out^, replaced^)
    var start = len(messages) - 1
    while start > 0 and messages[start - 1].role == ROLE_TOOL:
        start -= 1
    for i in range(start, len(messages)):
        var m = messages[i].copy()
        var text = String("\n").join(m.parts) if m.has_parts else m.content
        var path = _call_path(messages, m.tool_call_id)
        var nm = m.copy()
        if path.byte_length() > 0:
            nm.content = cp_head(text, _OVERFLOW_READ_KEEP_CHARS) + fmt1(OVERFLOW_READ_FILE_NOTICE, "original_path", path)
        else:
            nm.content = offload_text(m.tool_call_id, text, m.is_error, base, tx, now, "").text
        nm.parts = List[String]()
        nm.has_parts = False
        out[i] = nm^
        replaced.append(i)
    return (out^, replaced^)

