"""Portions ported from deepagents 0.7.32 and langchain 1.4.8 / langchain-core 1.7.4
(Copyright (c) LangChain, Inc., MIT License; see agents/mojo/LICENSES/langchain-MIT.txt).

The virtual file tools over the StateBackend (DESIGN.md 2.7.3): `ls`, `read_file`,
`write_file`, `edit_file`, `delete`, `glob`, `grep`, and the hidden `execute`.

Each function takes the turn-start snapshot `base` and the calling tool's transaction
`tx` (reads see `tx` over `base`; writes go to `tx`), and returns the exact text the
deepagents tool returns [fs §3, §4]. Every length is counted in code points.
"""

from std.collections import Dict, Optional

from qumo_agent.core.toolspec import ToolOutcome, text_outcome, error_outcome
from qumo_agent.util.pyfmt import (
    cp_len, cp_head, cp_slice, py_str_repr, py_list_repr, py_strip,
)
from qumo_agent.harness._fmt import fmt1, fmt2, fmt3
from qumo_agent.native.fnmatch import wc_glob_match
from qumo_agent.harness.texts import (
    ERR_FILE_NOT_FOUND, ERR_OFFSET_PAST_END, ERR_EXECUTE_UNAVAILABLE,
    OUT_UPDATED_FILE, OUT_DELETED, OUT_NO_FILES, OUT_NO_MATCHES,
    EMPTY_CONTENT_WARNING, NO_LINES_REQUESTED_WARNING, READ_NEGATIVE_OFFSET_NOTICE,
    READ_FILE_TRUNCATION_MSG, READ_FIELD_LINES, READ_FIELD_NEXT, READ_FIELD_TRUNCATED,
    READ_FIELD_MIDLINE, READ_FIELD_MIDLINE_COUNT,
    EDIT_EMPTY_OLD_STRING, EDIT_NOT_FOUND, EDIT_EOF_NEWLINE_UNIQUE, EDIT_EOF_NEWLINE_AMBIGUOUS,
    EDIT_MULTIPLE, EDIT_SUCCESS, TRUNCATION_GUIDANCE, GREP_TRUNCATION_NOTE, GREP_REGEX_HINT,
    MEDIA_PLACEHOLDER, MEDIA_EXTENSIONS, MEDIA_TYPES, MEDIA_MIMES,
)
from qumo_agent.harness.vfs import (
    Vfs, VfsTx, FileData, new_file_data, updated_file_data, validate_vfs_path,
)

comptime TOOL_RESULT_CHARS = 80000      # TOOL_RESULT_TOKEN_LIMIT (20000) * 4
comptime READ_BUDGET_CHARS = 80000      # NUM_CHARS_PER_TOKEN (4) * tool_token_limit_before_evict (20000)
comptime MAX_LINE_LENGTH = 5000
comptime GREP_DEFAULT_MAX_COUNT = 1000


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def truncate_list_repr(items: List[String]) -> String:
    """str(truncate_if_too_long(list)): each item costs len(repr(item)) + 2."""
    var budget = TOOL_RESULT_CHARS - cp_len(py_str_repr(TRUNCATION_GUIDANCE)) - 2
    var used = 0
    for kept in range(len(items)):
        used += cp_len(py_str_repr(items[kept])) + 2
        if used > budget:
            var cut = List[String]()
            for i in range(kept):
                cut.append(items[i])
            cut.append(TRUNCATION_GUIDANCE)
            return py_list_repr(cut)
    return py_list_repr(items)


def truncate_text(s: String) -> String:
    """truncate_if_too_long(str)."""
    if cp_len(s) > TOOL_RESULT_CHARS:
        var g = String(TRUNCATION_GUIDANCE)
        return cp_head(s, TOOL_RESULT_CHARS - cp_len(g) - 1) + "\n" + g
    return s


def _format_file_paths(paths: List[String]) -> String:
    if len(paths) == 0:
        return OUT_NO_FILES
    return truncate_list_repr(paths)


def _rjust(s: String, width: Int) -> String:
    var pad = width - cp_len(s)
    if pad <= 0:
        return s
    var out = String()
    for _ in range(pad):
        out += " "
    return out + s


def format_with_line_numbers(lines: List[String], start_line: Int) -> String:
    """deepagents format_content_with_line_numbers for a list of lines (BU:363-413):
    lines longer than 5000 characters continue on `N.1`, `N.2` rows; the gutter is
    right-aligned to the widest marker and separated by two spaces."""
    var markers = List[String]()
    var chunks = List[String]()
    var width = 0
    for i in range(len(lines)):
        var line_num = i + start_line
        var n = cp_len(lines[i])
        var ci = 0
        var s = 0
        while ci == 0 or s < n:
            var chunk = lines[i] if n <= MAX_LINE_LENGTH else cp_slice(lines[i], s, s + MAX_LINE_LENGTH)
            var marker = String(line_num) if ci == 0 else String(line_num) + "." + String(ci)
            if cp_len(marker) > width:
                width = cp_len(marker)
            markers.append(marker^)
            chunks.append(chunk^)
            ci += 1
            s += MAX_LINE_LENGTH
    var out = String()
    for i in range(len(markers)):
        if i > 0:
            out += "\n"
        out += _rjust(markers[i], width) + "  " + chunks[i]
    return out^


def _is_line_break(cp: Int) -> Bool:
    return ((cp >= 0x0A and cp <= 0x0D) or (cp >= 0x1C and cp <= 0x1E) or cp == 0x85
            or cp == 0x2028 or cp == 0x2029)


def splitlines_keepends(s: String) -> List[String]:
    """str.splitlines(keepends=True)."""
    var out = List[String]()
    var bs = s.as_bytes()
    var n = len(bs)
    var start = 0
    var i = 0
    while i < n:
        var b0 = Int(bs[i])
        var cp = b0
        var w = 1
        if b0 >= 0xF0:
            w = 4
            cp = 0x10000
        elif b0 >= 0xE0 and i + 2 < n:
            w = 3
            cp = ((b0 & 0x0F) << 12) | ((Int(bs[i + 1]) & 0x3F) << 6) | (Int(bs[i + 2]) & 0x3F)
        elif b0 >= 0xC0 and i + 1 < n:
            w = 2
            cp = ((b0 & 0x1F) << 6) | (Int(bs[i + 1]) & 0x3F)
        if _is_line_break(cp):
            var j = i + w
            if cp == 0x0D and j < n and Int(bs[j]) == 0x0A:
                j += 1
            out.append(String(unsafe_from_utf8=bs[start:j]))
            start = j
            i = j
        else:
            i += w
    if start < n:
        out.append(String(unsafe_from_utf8=bs[start:n]))
    return out^


def _error(prefix_msg: String) -> ToolOutcome:
    return error_outcome(prefix_msg)


# ---------------------------------------------------------------------------
# ls
# ---------------------------------------------------------------------------
def vfs_ls(base: Vfs, tx: VfsTx, path: String) -> ToolOutcome:
    var p: String
    try:
        p = validate_vfs_path(path)
    except e:
        return _error("Error: " + String(e))
    var view = tx.view(base)
    var prefix = p if p.endswith("/") else p + "/"
    var entries = List[String]()
    var subdirs = List[String]()
    for k in view.files.keys():
        if not k.startswith(prefix):
            continue
        var rel = String(k[byte=prefix.byte_length():])
        var slash = rel.find("/")
        if slash >= 0:
            var d = prefix + String(rel[byte=0:slash]) + "/"
            var seen = False
            for s in subdirs:
                if s == d:
                    seen = True
                    break
            if not seen:
                subdirs.append(d)
            continue
        entries.append(k)
    for d in subdirs:
        entries.append(d)
    sort(entries)
    return text_outcome(_format_file_paths(entries))


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------
def _suffix_lower(path: String) -> String:
    """PurePosixPath(path).suffix.lower()."""
    var name = path
    var slash = path.rfind("/")
    if slash >= 0:
        name = String(path[byte=slash + 1:])
    var dot = name.rfind(".")
    if dot > 0 and dot < name.byte_length() - 1:
        return String(name[byte=dot:]).lower()
    return String()


def media_kind(path: String) -> Tuple[String, String]:
    """(content type, mime) for an Appendix A.11 extension, or ("", "") for text."""
    var suffix = _suffix_lower(path)
    if suffix.byte_length() == 0:
        return (String(), String())
    var exts = materialize[MEDIA_EXTENSIONS]()
    for i in range(len(exts)):
        if exts[i] == suffix:
            return (materialize[MEDIA_TYPES]()[i], materialize[MEDIA_MIMES]()[i])
    return (String(), String())


def _is_blank(s: String) -> Bool:
    return s.byte_length() == 0 or py_strip(s).byte_length() == 0


def _lines_field(start: Int, end: Int, total: Int) -> String:
    return fmt3(READ_FIELD_LINES, "start", String(start), "end", String(end), "total", String(total))


def _assemble_len(notices: List[String], fields: List[String], body_len: Int) -> Int:
    """len("\\n".join([*notices, "@@ " + " | ".join(fields) + " @@", body]))."""
    var n = 0
    for x in notices:
        n += cp_len(x) + 1
    n += 6
    for i in range(len(fields)):
        if i > 0:
            n += 3
        n += cp_len(fields[i])
    return n + 1 + body_len


def _assemble(notices: List[String], fields: List[String], body: String) -> String:
    var out = String()
    for x in notices:
        out += x + "\n"
    out += "@@ " + String(" | ").join(fields) + " @@\n"
    out += body
    return out^


def vfs_read(base: Vfs, tx: VfsTx, file_path: String, offset: Int, limit: Int) -> ToolOutcome:
    var p: String
    try:
        p = validate_vfs_path(file_path)
    except e:
        return _error("Error: " + String(e))
    var fd_opt = tx.get(base, p)
    if not fd_opt:
        return _error(fmt1(ERR_FILE_NOT_FOUND, "file_path", p))
    var content = fd_opt.value().content
    var media = media_kind(p)
    if media[0].byte_length() > 0:
        # Non-text files are returned whole; the placeholder replaces the media block (X8).
        if _is_blank(content):
            return text_outcome(EMPTY_CONTENT_WARNING)
        return text_outcome(fmt3(MEDIA_PLACEHOLDER, "path", p, "type", media[0], "mime", media[1]))
    var off = offset if offset > 0 else 0
    var lim = limit if limit > 0 else 0
    if _is_blank(content):
        return text_outcome(EMPTY_CONTENT_WARNING)
    if lim == 0:
        return text_outcome(fmt1(NO_LINES_REQUESTED_WARNING, "limit", String(limit)))
    var lines = splitlines_keepends(content)
    var total = len(lines)
    if off >= total:
        return _error(fmt2(ERR_OFFSET_PAST_END, "offset", String(off), "total_lines", String(total)))
    var end_idx = off + lim
    if end_idx > total:
        end_idx = total
    var sliced = String()
    for i in range(off, end_idx):
        sliced += lines[i]
    sliced = sliced.replace("\r\n", "\n").replace("\r", "\n")
    var start_line = off + 1
    var end_line = end_idx
    var whole = start_line == 1 and end_line == total
    if _is_blank(sliced) and whole:
        return text_outcome(EMPTY_CONTENT_WARNING)
    # _prepare_read_window: the verbatim rows, padded back to the window size.
    var rows = List[String]()
    for r in sliced.split("\n"):
        rows.append(String(r))
    if len(rows) > 0 and rows[len(rows) - 1].byte_length() == 0:
        _ = rows.pop()
    while len(rows) < end_line - start_line + 1:
        rows.append(String())
    var notices = List[String]()
    if offset < 0:
        notices.append(fmt1(READ_NEGATIVE_OFFSET_NOTICE, "offset", String(offset)))
    var fields = List[String]()
    fields.append(_lines_field(start_line, end_line, total))
    if end_idx < total:
        fields.append(fmt1(READ_FIELD_NEXT, "next_offset", String(end_idx)))
    var row_lens = List[Int]()
    var body_len = 0
    for i in range(len(rows)):
        var l = cp_len(rows[i])
        row_lens.append(l)
        body_len += l
        if i > 0:
            body_len += 1
    if _assemble_len(notices, fields, body_len) < READ_BUDGET_CHARS:
        return text_outcome(_assemble(notices, fields, String("\n").join(rows)))
    # _truncate_paginated_read: cut at the last whole source line that fits.
    var notices2 = notices.copy()
    notices2.append(py_strip(fmt1(READ_FILE_TRUNCATION_MSG, "file_path", p)))
    var boundaries = List[Int]()
    var position = 0
    for i in range(len(rows)):
        if start_line + i > end_line:
            break
        position += row_lens[i]
        boundaries.append(position)
        position += 1
    var k = len(boundaries) - 1
    while k >= 0:
        var shown_end = start_line + k
        var f2 = List[String]()
        f2.append(_lines_field(start_line, shown_end, total))
        if shown_end < total:
            f2.append(fmt1(READ_FIELD_NEXT, "next_offset", String(shown_end)))
        f2.append(READ_FIELD_TRUNCATED)
        if _assemble_len(notices2, f2, boundaries[k]) <= READ_BUDGET_CHARS:
            var body = String()
            for i in range(k + 1):
                if i > 0:
                    body += "\n"
                body += rows[i]
            return text_outcome(_assemble(notices2, f2, body))
        k -= 1
    # Not even one line fits: show part of the first one.
    var oversized = row_lens[0] if len(row_lens) > 0 else 0
    var f3 = List[String]()
    f3.append(_lines_field(start_line, start_line, total))
    f3.append(READ_FIELD_MIDLINE)
    f3.append(fmt2(READ_FIELD_MIDLINE_COUNT, "shown", String(oversized), "oversized", String(oversized)))
    var reserved = _assemble_len(notices2, f3, 0)
    var shown = READ_BUDGET_CHARS - reserved
    if shown > oversized:
        shown = oversized
    if shown < 0:
        shown = 0
    f3[2] = fmt2(READ_FIELD_MIDLINE_COUNT, "shown", String(shown), "oversized", String(oversized))
    var joined = String("\n").join(rows)
    return text_outcome(_assemble(notices2, f3, cp_head(joined, shown)))


# ---------------------------------------------------------------------------
# write_file / edit_file / delete
# ---------------------------------------------------------------------------
def vfs_write(base: Vfs, mut tx: VfsTx, file_path: String, content: String, now: String) -> ToolOutcome:
    var p: String
    try:
        p = validate_vfs_path(file_path)
    except e:
        return _error("Error: " + String(e))
    var existing = tx.get(base, p)
    var fd: FileData
    if existing:
        fd = updated_file_data(existing.value(), content, now)
    else:
        fd = new_file_data(content, now)
    fd.encoding = "utf-8"
    tx.set(p, fd)
    return text_outcome(fmt1(OUT_UPDATED_FILE, "path", p))


def count_occurrences(haystack: String, needle: String) -> Int:
    """str.count for a non-empty needle (non-overlapping)."""
    var n = 0
    var start = 0
    var step = needle.byte_length()
    while True:
        var i = haystack.find(needle, start)
        if i < 0:
            return n
        n += 1
        start = i + step


def vfs_edit(base: Vfs, mut tx: VfsTx, file_path: String, old: String, new: String, replace_all: Bool, now: String) -> ToolOutcome:
    var p: String
    try:
        p = validate_vfs_path(file_path)
    except e:
        return _error("Error: " + String(e))
    var existing = tx.get(base, p)
    if not existing:
        return _error(fmt1(ERR_FILE_NOT_FOUND, "file_path", p))
    var fd = existing.value().copy()
    var content = fd.content
    if old.byte_length() == 0:
        return _error(EDIT_EMPTY_OLD_STRING)
    var occurrences = count_occurrences(content, old)
    if occurrences == 0:
        if old.endswith("\n") and cp_len(old) > 1:
            var stripped = String(old[byte=0:old.byte_length() - 1])
            if content.endswith(stripped):
                var stripped_count = count_occurrences(content, stripped)
                if stripped_count == 1:
                    return _error(EDIT_EOF_NEWLINE_UNIQUE)
                return _error(fmt1(EDIT_EOF_NEWLINE_AMBIGUOUS, "stripped_count", String(stripped_count)))
        return _error(fmt1(EDIT_NOT_FOUND, "old_string", old))
    if occurrences > 1 and not replace_all:
        return _error(fmt2(EDIT_MULTIPLE, "old_string", old, "occurrences", String(occurrences)))
    var new_content = content.replace(old, new)
    tx.set(p, updated_file_data(fd, new_content, now))
    return text_outcome(fmt2(EDIT_SUCCESS, "occurrences", String(occurrences), "path", p))


def vfs_delete(base: Vfs, mut tx: VfsTx, file_path: String) -> ToolOutcome:
    var p: String
    try:
        p = validate_vfs_path(file_path)
    except e:
        return _error("Error: " + String(e))
    var b = p
    while b.endswith("/"):
        var _t = String(b[byte=0:b.byte_length() - 1])
        b = _t^
    var prefix = b + "/"
    var view = tx.view(base)
    var doomed = List[String]()
    for k in view.files.keys():
        if k == b or k.startswith(prefix):
            doomed.append(k)
    if len(doomed) == 0:
        return _error(fmt1(ERR_FILE_NOT_FOUND, "file_path", p))
    for k in doomed:
        tx.delete(k)
    return text_outcome(fmt1(OUT_DELETED, "path", p))


# ---------------------------------------------------------------------------
# glob / grep
# ---------------------------------------------------------------------------
def _normalize_search_root(path: Optional[String]) -> String:
    """deepagents _normalize_path (None and "" give "/")."""
    var p = String("/")
    if path:
        if path.value().byte_length() > 0:
            p = path.value()
    if not p.startswith("/"):
        p = "/" + p
    if p != "/":
        while p.endswith("/") and p != "/":
            var _t = String(p[byte=0:p.byte_length() - 1])
            p = _t^
    return p^


def _filter_by_root(view: Vfs, root: String) -> List[String]:
    """deepagents _filter_files_by_path: the exact file, else every key below root."""
    var out = List[String]()
    if root in view.files:
        out.append(root)
        return out^
    var prefix = "/" if root == "/" else root + "/"
    for k in view.files.keys():
        if k.startswith(prefix):
            out.append(k)
    return out^


def _relative_to_root(file_path: String, root: String) -> String:
    if root == "/":
        return String(file_path[byte=1:])
    if file_path == root:
        var slash = file_path.rfind("/")
        return String(file_path[byte=slash + 1:])
    return String(file_path[byte=root.byte_length() + 1:])


def vfs_glob(base: Vfs, tx: VfsTx, pattern: String, path: Optional[String]) -> ToolOutcome:
    var permission_path: String
    try:
        permission_path = validate_vfs_path(path.value() if path else String("/"))
    except e:
        return _error("Error: " + String(e))
    var backend_path = Optional[String](permission_path) if path else Optional[String](None)
    var root = _normalize_search_root(backend_path)
    var view = tx.view(base)
    var candidates = _filter_by_root(view, root)
    try:
        _ = wc_glob_match("", pattern)    # a refused pattern is an error even with no files
    except e:
        return _error("Error: " + String(e))
    var matched = List[String]()
    var stamps = List[String]()
    for k in candidates:
        var ok: Bool
        try:
            ok = wc_glob_match(_relative_to_root(k, root), pattern)
        except e:
            return _error("Error: " + String(e))
        if ok:
            matched.append(k)
            var fd = view.get(k)
            stamps.append(fd.value().modified_at if fd else String())
    # Stable sort by modified_at, newest first (Python sort(reverse=True) is stable).
    for i in range(1, len(matched)):
        var j = i
        while j > 0 and stamps[j - 1] < stamps[j]:
            var ts = stamps[j - 1]
            stamps[j - 1] = stamps[j]
            stamps[j] = ts
            var tp = matched[j - 1]
            matched[j - 1] = matched[j]
            matched[j] = tp
            j -= 1
    return text_outcome(_format_file_paths(matched))


def looks_like_regex(pattern: String) -> Bool:
    """deepagents _REGEX_SIGNAL_RE: `|`, `.*`, `.+`, or a backslash before a regex metacharacter."""
    if "|" in pattern or ".*" in pattern or ".+" in pattern:
        return True
    var bs = pattern.as_bytes()
    var meta = String(".wWdDsSbB(){}[]|+*?^$").as_bytes()
    for i in range(len(bs) - 1):
        if Int(bs[i]) == 0x5C:
            var c = bs[i + 1]
            for m in meta:
                if m == c:
                    return True
    return False


def vfs_grep(base: Vfs, tx: VfsTx, pattern: String, path: Optional[String], glob: Optional[String], output_mode: String, max_count: Optional[Int]) -> ToolOutcome:
    var search = Optional[String](None)
    if path:
        try:
            search = Optional[String](validate_vfs_path(path.value()))
        except e:
            return _error("Error: " + String(e))
    var root = _normalize_search_root(search.copy() if search else Optional[String](String("/")))
    var view = tx.view(base)
    var candidates = _filter_by_root(view, root)
    if glob and glob.value().byte_length() > 0:
        var g = glob.value()
        try:
            _ = wc_glob_match("", g)
        except e:
            return _error(String(e))
        var kept = List[String]()
        for k in candidates:
            var ok: Bool
            try:
                ok = wc_glob_match(_relative_to_root(k, root), g)
            except e:
                return _error(String(e))
            if ok:
                kept.append(k)
        candidates = kept^
    var cap = max_count.value() if max_count else GREP_DEFAULT_MAX_COUNT
    var m_paths = List[String]()
    var m_lines = List[Int]()
    var m_texts = List[String]()
    var truncated = False
    for k in candidates:
        if truncated:
            break
        var fd = view.get(k)
        if not fd:
            continue
        var line_num = 0
        for raw in fd.value().content.split("\n"):
            line_num += 1
            var line = String(raw)
            if pattern in line:
                if len(m_paths) >= cap:
                    truncated = True
                    break
                m_paths.append(k)
                m_lines.append(line_num)
                m_texts.append(line)
    var formatted: String
    if len(m_paths) == 0:
        formatted = OUT_NO_MATCHES
    else:
        # Group by path, then sort the paths.
        var order = List[String]()
        for p in m_paths:
            var seen = False
            for o in order:
                if o == p:
                    seen = True
                    break
            if not seen:
                order.append(p)
        sort(order)
        var out_lines = List[String]()
        for p in order:
            if output_mode == "files_with_matches":
                out_lines.append(p)
            elif output_mode == "count":
                var n = 0
                for q in m_paths:
                    if q == p:
                        n += 1
                out_lines.append(p + ": " + String(n))
            else:
                out_lines.append(p + ":")
                for i in range(len(m_paths)):
                    if m_paths[i] == p:
                        out_lines.append("  " + String(m_lines[i]) + ": " + m_texts[i])
        formatted = String("\n").join(out_lines)
    formatted = truncate_text(formatted)
    var notes = List[String]()
    if truncated:
        notes.append(GREP_TRUNCATION_NOTE)
    if not truncated and len(m_paths) == 0 and looks_like_regex(pattern):
        notes.append(GREP_REGEX_HINT)
    if len(notes) > 0:
        return text_outcome(formatted + "\n\n" + String("\n\n").join(notes))
    return text_outcome(formatted)


def execute_unavailable() -> ToolOutcome:
    return error_outcome(ERR_EXECUTE_UNAVAILABLE)
