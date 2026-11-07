"""Portions ported from deepagents 0.7.32 and langchain 1.4.8 / langchain-core 1.7.4
(Copyright (c) LangChain, Inc., MIT License; see agents/mojo/LICENSES/langchain-MIT.txt).

The merged `read_file` / `write_file` / `edit_file` tools (DESIGN.md 2.7.4, D2a): one
tool per name, routed by path, accepting both argument vocabularies (deepagents'
`file_path`/`offset`/`limit`/`old_string`/`new_string` and QUMO's
`path`/`start_line`/`max_lines`/`old`/`new`).

| route | rule |
|---|---|
| `/...` | virtual file system (deepagents texts) |
| relative, `read_file`, `routes.repo_read` | `env.relative_file_op` (repository or open sandbox) |
| relative, write/edit, `routes.sandbox_write` | `env.relative_file_op` (open sandbox) |
| relative, write/edit, `routes.repo_read` only | refused with WRITE_RELATIVE_REFUSED |
| relative, V0 agents | virtual, normalized by validate_vfs_path |
"""

from qumo.ai.json import JsonValue, JsonWriter

from qumo_agent.core.toolspec import ToolSpec, ToolOutcome, TOOL_HARNESS, error_outcome
from qumo_agent.harness.env import (
    AgentEnv, FileRoutes, RelativeFileOp, FILE_READ, FILE_WRITE, FILE_EDIT,
)
from qumo_agent.harness.fs_tools import vfs_read, vfs_write, vfs_edit
from qumo_agent.harness.subagents import count_json_for
from qumo_agent.harness.texts import (
    READ_FILE_DESCRIPTION, WRITE_FILE_DESCRIPTION, EDIT_FILE_DESCRIPTION,
    READ_FILE_NOTES_V0, READ_FILE_NOTES_V12, WRITE_FILE_NOTES_V0, WRITE_FILE_NOTES_V1,
    WRITE_FILE_NOTES_V2, EDIT_FILE_NOTES_V0, EDIT_FILE_NOTES_V1, EDIT_FILE_NOTES_V2,
    READ_FILE_PARAMETERS_V0, READ_FILE_PARAMETERS_V12, WRITE_FILE_PARAMETERS_V0,
    WRITE_FILE_PARAMETERS_V1, WRITE_FILE_PARAMETERS_V2, EDIT_FILE_PARAMETERS_V0,
    EDIT_FILE_PARAMETERS_V1, EDIT_FILE_PARAMETERS_V2, WRITE_RELATIVE_REFUSED, FILE_PATH_CONFLICT,
    COUNT_PARAMETERS_READ_FILE, COUNT_PARAMETERS_WRITE_FILE, COUNT_PARAMETERS_EDIT_FILE,
)
from qumo_agent.harness.validate import invocation_error_text
from qumo_agent.native.fs import normpath
from qumo_agent.harness.vfs import Vfs, VfsTx, validate_vfs_path

comptime _READ_DEFAULT_OFFSET = 0
comptime _READ_DEFAULT_LIMIT = 100
comptime _REL_DEFAULT_START = 1
comptime _REL_DEFAULT_MAX = 400


def merged_file_tools(routes: FileRoutes) -> List[ToolSpec]:
    """read_file, write_file, edit_file for the agent's route variant (Appendix B.1).
    The token counter sees deepagents' own tools in these slots (count_json)."""
    var read_notes = READ_FILE_NOTES_V12 if routes.repo_read else READ_FILE_NOTES_V0
    var read_params = READ_FILE_PARAMETERS_V12 if routes.repo_read else READ_FILE_PARAMETERS_V0
    var write_notes: String
    var write_params: String
    var edit_notes: String
    var edit_params: String
    if routes.sandbox_write:
        write_notes = WRITE_FILE_NOTES_V2
        write_params = WRITE_FILE_PARAMETERS_V2
        edit_notes = EDIT_FILE_NOTES_V2
        edit_params = EDIT_FILE_PARAMETERS_V2
    elif routes.repo_read:
        write_notes = WRITE_FILE_NOTES_V1
        write_params = WRITE_FILE_PARAMETERS_V1
        edit_notes = EDIT_FILE_NOTES_V1
        edit_params = EDIT_FILE_PARAMETERS_V1
    else:
        write_notes = WRITE_FILE_NOTES_V0
        write_params = WRITE_FILE_PARAMETERS_V0
        edit_notes = EDIT_FILE_NOTES_V0
        edit_params = EDIT_FILE_PARAMETERS_V0
    var out = List[ToolSpec]()
    out.append(ToolSpec("read_file", String(READ_FILE_DESCRIPTION) + "\n\n" + read_notes, read_params, TOOL_HARNESS, False,
                        count_json_for("read_file", READ_FILE_DESCRIPTION, COUNT_PARAMETERS_READ_FILE)))
    out.append(ToolSpec("write_file", String(WRITE_FILE_DESCRIPTION) + "\n\n" + write_notes, write_params, TOOL_HARNESS, False,
                        count_json_for("write_file", WRITE_FILE_DESCRIPTION, COUNT_PARAMETERS_WRITE_FILE)))
    out.append(ToolSpec("edit_file", String(EDIT_FILE_DESCRIPTION) + "\n\n" + edit_notes, edit_params, TOOL_HARNESS, False,
                        count_json_for("edit_file", EDIT_FILE_DESCRIPTION, COUNT_PARAMETERS_EDIT_FILE)))
    return out^


def _str_arg(args: JsonValue, key: String) -> Optional[String]:
    var v = args.get(key)
    if v.is_string():
        try:
            return v.as_string()
        except:
            pass
    return None


def _int_arg(args: JsonValue, key: String) -> Optional[Int]:
    var v = args.get(key)
    if v.is_number():
        try:
            return v.as_int()
        except:
            pass
    return None


def file_path_arg(args: JsonValue) raises -> Tuple[String, String]:
    """(path, error_text): `file_path`, else `path`. Both given and different gives the
    FILE_PATH_CONFLICT error; neither gives ("", "")."""
    var fp = _str_arg(args, "file_path")
    var p = _str_arg(args, "path")
    if fp and p and fp.value() != p.value():
        return (String(), String(FILE_PATH_CONFLICT))
    if fp:
        return (fp.value(), String())
    if p:
        return (p.value(), String())
    return (String(), String())


def _is_relative(path: String) -> Bool:
    return not path.startswith("/")


def mutation_key(tool: String, args: JsonValue, routes: FileRoutes) -> String:
    """D2d: the key of a file mutation, following the route (not the spelling): virtual
    targets give "vfs:" + validate_vfs_path(p), relative ones handed to the env give
    "rel:" + normpath(p); a refused route, an invalid path or a non-mutation gives ""."""
    if tool != "write_file" and tool != "edit_file" and tool != "delete":
        return String()
    var p: String
    try:
        var r = file_path_arg(args)
        if r[1].byte_length() > 0:
            return String()
        p = r[0]
    except:
        return String()
    if p.byte_length() == 0 and not args.has("file_path") and not args.has("path"):
        return String()
    if tool != "delete" and _is_relative(p):
        if routes.sandbox_write:
            return "rel:" + normpath(p)
        if routes.repo_read:
            return String()
    try:
        return "vfs:" + validate_vfs_path(p)
    except:
        return String()


def run_file_tool[E: AgentEnv](mut env: E, tool: String, args: JsonValue, original_args: JsonValue, base: Vfs, mut tx: VfsTx, routes: FileRoutes) raises -> ToolOutcome:
    """Runs a merged file tool. `args` are the validated (coerced) arguments; whether an
    optional argument was given is decided from `original_args`."""
    var r = file_path_arg(args)
    if r[1].byte_length() > 0:
        return error_outcome(r[1])
    if not args.has("file_path") and not args.has("path"):
        var errs: List[String] = ["file_path: Field required"]
        return error_outcome(invocation_error_text(tool, original_args, errs))
    var path = r[0]
    var relative = _is_relative(path)
    if tool == "read_file":
        if relative and routes.repo_read:
            var start = _REL_DEFAULT_START
            if original_args.has("start_line") and _int_arg(args, "start_line"):
                start = _int_arg(args, "start_line").value()
            elif original_args.has("offset") and _int_arg(args, "offset"):
                start = _int_arg(args, "offset").value() + 1
            var max_lines = _REL_DEFAULT_MAX
            if original_args.has("max_lines") and _int_arg(args, "max_lines"):
                max_lines = _int_arg(args, "max_lines").value()
            elif original_args.has("limit") and _int_arg(args, "limit"):
                max_lines = _int_arg(args, "limit").value()
            return env.relative_file_op(RelativeFileOp(FILE_READ, path, start, max_lines, "", "", "", False))
        var offset = _READ_DEFAULT_OFFSET
        if original_args.has("offset") and _int_arg(args, "offset"):
            offset = _int_arg(args, "offset").value()
        elif original_args.has("start_line") and _int_arg(args, "start_line"):
            offset = _int_arg(args, "start_line").value() - 1
        var limit = _READ_DEFAULT_LIMIT
        if original_args.has("limit") and _int_arg(args, "limit"):
            limit = _int_arg(args, "limit").value()
        elif original_args.has("max_lines") and _int_arg(args, "max_lines"):
            limit = _int_arg(args, "max_lines").value()
        return vfs_read(base, tx, path, offset, limit)
    if tool == "write_file":
        var content = _str_arg(args, "content")
        if not content:
            var errs: List[String] = ["content: Field required"]
            return error_outcome(invocation_error_text(tool, original_args, errs))
        if relative and routes.sandbox_write:
            return env.relative_file_op(RelativeFileOp(FILE_WRITE, path, 0, 0, content.value(), "", "", False))
        if relative and routes.repo_read:
            return error_outcome(WRITE_RELATIVE_REFUSED)
        return vfs_write(base, tx, path, content.value(), env.now_utc_iso())
    # edit_file
    var old = _str_arg(args, "old_string")
    if not old:
        old = _str_arg(args, "old")
    var new = _str_arg(args, "new_string")
    if not new:
        new = _str_arg(args, "new")
    if not old or not new:
        var errs = List[String]()
        if not old:
            errs.append("old_string: Field required")
        if not new:
            errs.append("new_string: Field required")
        return error_outcome(invocation_error_text(tool, original_args, errs))
    var replace_all = False
    var ra = args.get("replace_all")
    if ra.is_bool():
        replace_all = ra.as_bool()
    if relative and routes.sandbox_write:
        return env.relative_file_op(RelativeFileOp(FILE_EDIT, path, 0, 0, "", old.value(), new.value(), replace_all))
    if relative and routes.repo_read:
        return error_outcome(WRITE_RELATIVE_REFUSED)
    return vfs_edit(base, tx, path, old.value(), new.value(), replace_all, env.now_utc_iso())
