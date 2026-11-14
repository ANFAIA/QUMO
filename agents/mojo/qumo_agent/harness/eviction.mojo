"""Portions ported from deepagents 0.7.32 and langchain 1.4.8 / langchain-core 1.7.4
(Copyright (c) LangChain, Inc., MIT License; see agents/mojo/LICENSES/langchain-MIT.txt).

Large-result eviction (DESIGN.md 2.7.5, [fs §7]): tool results longer than 80,000 code
points go to `/large_tool_results/<id>` and the model sees a stub with a head/tail
preview; the last user message longer than 200,000 code points goes to
`/conversation_history/<uuid>.md` the same way.

An evicted parts result becomes plain string content (`_build_evicted_content` returns
the replacement text when the list holds no non-text block, and this port has none, X9).
"""

from qumo_agent.core.toolspec import ToolOutcome
from qumo_agent.util.pyfmt import cp_len, cp_head, py_splitlines
from qumo_agent.harness._fmt import fmt, fmt1
from qumo_agent.util.sha256 import sha256_hex_text
from qumo_agent.harness.fs_tools import format_with_line_numbers
from qumo_agent.harness.texts import (
    TOO_LARGE_TOOL_MSG, TOO_LARGE_HUMAN_MSG, PREVIEW_NOTE_PLAIN, PREVIEW_NOTE_HEAD_TAIL,
    CAVEAT_OMITTED_LINES, CAVEAT_CLIPPED_LINES, TRUNCATION_MARKER,
)
from qumo_agent.harness.vfs import Vfs, VfsTx, FileData, new_file_data, updated_file_data

comptime TOOL_EVICT_CHARS = 80000          # 4 * 20000, strictly greater evicts [critic A2]
comptime HUMAN_EVICT_CHARS = 200000        # 4 * 50000
comptime PREVIEW_LINE_CHARS = 1000
comptime _HEAD_LINES = 5
comptime _TAIL_LINES = 5
comptime _MAX_ID_BYTES = 128
comptime _VISIBLE_ID_CHARS = 32
comptime LARGE_TOOL_RESULTS_PREFIX = "/large_tool_results"


def is_eviction_exempt(tool: String) -> Bool:
    """deepagents TOOLS_EXCLUDED_FROM_EVICTION."""
    return (tool == "ls" or tool == "glob" or tool == "grep" or tool == "read_file"
            or tool == "edit_file" or tool == "write_file" or tool == "delete")


def sanitize_tool_call_id(id: String) -> String:
    """`.`, `/` and `\\` become `_`; more than 128 bytes gives "call-" + sha256 of the id."""
    var s = id.replace(".", "_").replace("/", "_").replace("\\", "_")
    if s.byte_length() > _MAX_ID_BYTES:
        return "call-" + sha256_hex_text(id)
    return s^


def display_tool_call_id(id: String) -> String:
    """The first 32 code points + "..." when longer."""
    if cp_len(id) > _VISIBLE_ID_CHARS:
        return cp_head(id, _VISIBLE_ID_CHARS) + "..."
    return id


@fieldwise_init
struct Preview(Copyable):
    var text: String
    var lines_omitted: Bool
    var lines_clipped: Bool


def _clip(lines: List[String], a: Int, b: Int, mut clipped: Bool) -> List[String]:
    var out = List[String]()
    for i in range(a, b):
        if cp_len(lines[i]) > PREVIEW_LINE_CHARS:
            clipped = True
            out.append(cp_head(lines[i], PREVIEW_LINE_CHARS))
        else:
            out.append(lines[i])
    return out^


def content_preview(text: String) -> Preview:
    """_create_content_preview: all lines when at most 10, else 5 head + marker + 5 tail."""
    var lines = py_splitlines(text)
    var n = len(lines)
    var clipped = False
    if n <= _HEAD_LINES + _TAIL_LINES:
        var shown = _clip(lines, 0, n, clipped)
        return Preview(format_with_line_numbers(shown, 1), False, clipped)
    var head = _clip(lines, 0, _HEAD_LINES, clipped)
    var tail = _clip(lines, n - _TAIL_LINES, n, clipped)
    var marker = fmt1(TRUNCATION_MARKER, "omitted_lines", String(n - _HEAD_LINES - _TAIL_LINES))
    var out = format_with_line_numbers(head, 1) + "\n" + marker + "\n" + format_with_line_numbers(tail, n - _TAIL_LINES + 1)
    return Preview(out^, True, clipped)


def preview_note(p: Preview, subject: String) -> String:
    """`Here is a preview ...[ (caveats)]:` for `subject` "result" or "content"."""
    var note = fmt1(PREVIEW_NOTE_HEAD_TAIL if p.lines_omitted else PREVIEW_NOTE_PLAIN, "subject", subject)
    var caveats = List[String]()
    if p.lines_omitted:
        caveats.append(CAVEAT_OMITTED_LINES)
    if p.lines_clipped:
        caveats.append(CAVEAT_CLIPPED_LINES)
    if len(caveats) > 0:
        note += " (" + String("; ").join(caveats) + ")"
    return note + ":"


def large_tool_result_text(call_id: String, file_path: String, full_text: String) -> String:
    """TOO_LARGE_TOOL_MSG for `full_text` saved at `file_path`."""
    var p = content_preview(full_text)
    var keys: List[String] = ["tool_call_id", "file_path", "preview_note", "content_sample"]
    var values: List[String] = [display_tool_call_id(call_id), file_path, preview_note(p, "result"), p.text]
    return fmt(TOO_LARGE_TOOL_MSG, keys, values)


def human_message_stub(file_path: String, full_text: String) -> String:
    """TOO_LARGE_HUMAN_MSG for a user message saved at `file_path`."""
    var p = content_preview(full_text)
    var keys: List[String] = ["file_path", "preview_note", "content_sample"]
    var values: List[String] = [file_path, preview_note(p, "content"), p.text]
    return fmt(TOO_LARGE_HUMAN_MSG, keys, values)


def write_overwrite(base: Vfs, mut tx: VfsTx, path: String, content: String, now: String):
    """StateBackend.write: create, or replace keeping created_at."""
    var existing = tx.get(base, path)
    var fd: FileData
    if existing:
        fd = updated_file_data(existing.value(), content, now)
    else:
        fd = new_file_data(content, now)
    fd.encoding = "utf-8"
    tx.set(path, fd)


def offload_text(call_id: String, text: String, is_error: Bool, base: Vfs, mut tx: VfsTx, now: String, fallback_id: String) -> ToolOutcome:
    """_offload_tool_message_content: writes `text` to /large_tool_results/<id> and returns
    the plain-string stub outcome (regardless of size or tool name)."""
    var sanitized = sanitize_tool_call_id(call_id) if call_id.byte_length() > 0 else "unknown-" + fallback_id
    var path = String(LARGE_TOOL_RESULTS_PREFIX) + "/" + sanitized
    write_overwrite(base, tx, path, text, now)
    return ToolOutcome(large_tool_result_text(call_id, path, text), List[String](), False, is_error)


def evict_tool_outcome(tool: String, call_id: String, outcome: ToolOutcome, base: Vfs, mut tx: VfsTx, now: String, fallback_id: String) -> Tuple[ToolOutcome, Bool]:
    """Tool-result eviction [fs §7]: exempt tools and results of at most 80,000 code
    points are unchanged; otherwise the result is offloaded and replaced by the stub.
    `fallback_id` (the first 8 hex digits of a uuid4) names results without an id."""
    if is_eviction_exempt(tool):
        return (outcome.copy(), False)
    var text = outcome.joined_text()
    if cp_len(text) <= TOOL_EVICT_CHARS:
        return (outcome.copy(), False)
    return (offload_text(call_id, text, outcome.is_error, base, tx, now, fallback_id), True)
