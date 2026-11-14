"""Golden data for the Mojo harness (DESIGN.md 8.1 and Appendix D). Dev-time only.

Run from the repository root:

    agents/.venv/bin/python -I agents/mojo/tests/golden/gen_golden.py [section ...]

Sections (default: all): texts fs previews tokens summaries pydantic wire.
Writes into agents/mojo/tests/golden/:

- deepagents_golden.json  Appendix A texts and schemas read from the installed packages
                          (checked against DESIGN.md Appendix A), plus the B.1 texts
                          taken from DESIGN.md, and COUNT_PARAMETERS.
- fs_outputs.json         deepagents StateBackend tool outputs, recorded through a real
                          create_deep_agent run driven by a scripted model.
- previews.json           _create_content_preview, the preview note and both stubs.
- token_counts.json       count_tokens_approximately (scaling on) for message lists.
- summaries.json          XML transcripts, the summary prompt, safe cutoffs, the overflow
                          retry prompt and _truncate_args results.
- pydantic_errors.json    Appendix C.2 validation messages (pydantic through LangGraph).
- wire_S1..S6.json        exact request bodies of deepagents driven by a scripted model,
                          with uuids and timestamps replaced by placeholders.

Portions of the behaviour recorded here belong to deepagents 0.7.32 and langchain 1.4.8 /
langchain-core 1.7.4 (Copyright (c) LangChain, Inc., MIT License).
"""

import asyncio
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
DESIGN = REPO / "agents" / "mojo" / "DESIGN.md"
SP = REPO / "agents" / ".venv" / "lib" / "python3.12" / "site-packages"

sys.path.insert(0, str(REPO / "agents"))

import httpx  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage  # noqa: E402
from langchain_core.messages.utils import count_tokens_approximately, get_buffer_string  # noqa: E402
from langchain_core.tools import ToolException, tool  # noqa: E402
from langchain_core.utils.function_calling import convert_to_openai_tool  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

from deepagents import CompiledSubAgent, create_deep_agent  # noqa: E402
from deepagents.backends import utils as bu  # noqa: E402
from deepagents.middleware import _message_eviction as me  # noqa: E402
from deepagents.middleware import filesystem as fs  # noqa: E402
from deepagents.middleware import subagents as sa  # noqa: E402
from deepagents.middleware import summarization as dsm  # noqa: E402
from langgraph.prebuilt import tool_node as tn  # noqa: E402

MODEL_ID = "anthropic/claude-opus-5.5"


def write(name, data):
    path = HERE / name
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(REPO)}")


# ---------------------------------------------------------------------------
# DESIGN.md appendices
# ---------------------------------------------------------------------------
def design_blocks(heading_prefix):
    """The ```json blocks of the DESIGN.md section whose heading starts with `heading_prefix`."""
    text = DESIGN.read_text(encoding="utf-8")
    start = text.index("\n### " + heading_prefix)
    end = text.find("\n### ", start + 5)
    end2 = text.find("\n## ", start + 5)
    if end < 0 or (0 <= end2 < end):
        end = end2
    section = text[start:end]
    return [json.loads(m) for m in re.findall(r"```json\n(.*?)\n```", section, re.S)]


# ---------------------------------------------------------------------------
# texts
# ---------------------------------------------------------------------------
FILE_TOOLS = ["ls", "read_file", "write_file", "edit_file", "delete", "glob", "grep"]


def strip_title_description(schema):
    return {k: v for k, v in schema.items() if k not in ("title", "description")}


def gen_texts():
    texts = {}
    schemas = {}
    m = fs.FilesystemMiddleware()
    tools = {t.name: t for t in m.tools}
    count_parameters = {}
    for name in FILE_TOOLS:
        f = convert_to_openai_tool(tools[name])["function"]
        desc = f["description"]
        if name == "grep":
            desc = fs._GREP_TOOL_DESCRIPTION_WITHOUT_EXECUTE.strip()
        texts[name.upper() + "_DESCRIPTION"] = desc
        schemas[name.upper() + "_PARAMETERS"] = f["parameters"]
        count_parameters[name] = strip_title_description(tools[name].tool_call_schema.model_json_schema())
    texts["TOO_LARGE_TOOL_MSG"] = me._TOO_LARGE_TOOL_MSG
    texts["PREVIEW_NOTE_PLAIN"] = me._PREVIEW_NOTE_PLAIN
    texts["PREVIEW_NOTE_HEAD_TAIL"] = me._PREVIEW_NOTE_HEAD_TAIL
    texts["CAVEAT_OMITTED_LINES"] = me._CAVEAT_OMITTED_LINES
    texts["CAVEAT_CLIPPED_LINES"] = me._CAVEAT_CLIPPED_LINES
    texts["TRUNCATION_MARKER"] = bu.TRUNCATION_MARKER_TEMPLATE
    texts["TOO_LARGE_HUMAN_MSG"] = fs._TOO_LARGE_HUMAN_MSG
    texts["TASK_DESCRIPTION_TEMPLATE"] = sa.TASK_TOOL_DESCRIPTION
    task_params = task_parameters_from_request()
    schemas["TASK_PARAMETERS"] = task_params
    count_parameters["task"] = strip_title_description(sa.TaskToolSchema.model_json_schema())
    texts["TASK_UNKNOWN_SUBAGENT_ERROR"] = "We cannot invoke subagent {subagent_type} because it does not exist, the only allowed types are {allowed_types}"
    texts["TASK_UNEXPECTED_ARGUMENT_MSG"] = "Value error, Unexpected argument '{key}'; put all instructions for the subagent in `description`."
    texts["GENERAL_PURPOSE_DESCRIPTION"] = sa.DEFAULT_GENERAL_PURPOSE_DESCRIPTION
    texts["DEFAULT_SUBAGENT_PROMPT"] = sa.DEFAULT_SUBAGENT_PROMPT
    texts["SUMMARY_PROMPT_TEMPLATE"] = dsm.DEEPAGENTS_DEFAULT_SUMMARY_PROMPT
    # A.7: rendered through the real code with placeholder values.
    texts["SUMMARY_WITH_FILE"] = summary_message_text("{summary}", "{file_path}")
    texts["SUMMARY_WITHOUT_FILE"] = summary_message_text("{summary}", None)
    texts["SUMMARY_NO_HISTORY"] = "No previous conversation history."
    texts["SUMMARY_TOO_LONG"] = "Previous conversation was too long to summarize."
    texts["SUMMARY_HISTORY_SECTION"] = "## Summarized at {timestamp}\n\n{xml}\n\n"
    texts["SUMMARY_OVERFLOW_MIDDLE"] = "\n\n[... middle of conversation omitted ...]\n\n"
    texts["ARG_TRUNCATION_SUFFIX"] = "...(argument truncated)"
    texts["INVALID_TOOL_NAME"] = tn.INVALID_TOOL_NAME_ERROR_TEMPLATE
    texts["TOOL_INVOCATION_ERROR"] = tn.TOOL_INVOCATION_ERROR_TEMPLATE
    # A.9: checked against the source where the text is a literal.
    for name, value in design_blocks("A.9")[0].items():
        texts[name] = value
    for name, value in design_blocks("A.10")[0].items():
        texts[name] = value
    texts["MEDIA_PLACEHOLDER"] = "[read_file: {path} was not attached because this model does not support {type} content ({mime}).]"
    a12 = design_blocks("A.12")[0]
    for name in ("OVERFLOW_READ_FILE_NOTICE", "OVERFLOW_RECOVERY_EXHAUSTED", "OVERFLOW_HISTORY_OFFLOAD_FAILED"):
        texts[name] = a12[name]
    b1 = design_blocks("B.1")
    for name, value in b1[0].items():
        texts[name] = value
    for name, value in b1[1].items():
        schemas[name] = value
    media = design_blocks("A.11")[0]

    check_sources(texts)
    check_against_design(texts, schemas, media, a12)
    return {
        "texts": texts,
        "schemas": schemas,
        "media": media,
        "overflow_markers": a12["OVERFLOW_MARKERS"],
        "count_parameters": count_parameters,
    }


def summary_message_text(summary, file_path):
    mw = dsm.SummarizationMiddleware.__new__(dsm.SummarizationMiddleware)
    msg = dsm.SummarizationMiddleware._build_new_messages_with_path(mw, summary, file_path)[0]
    return msg.content


def source(rel):
    return (SP / rel).read_text(encoding="utf-8")


def check_sources(texts):
    """Texts copied from DESIGN.md must appear in the installed sources."""
    pyfs = source("deepagents/middleware/filesystem.py") + source("deepagents/backends/utils.py") + source("deepagents/backends/state.py")
    checks = {
        "ERR_PATH_TRAVERSAL": "Path traversal not allowed: {path}",
        "ERR_GLOB_TRAVERSAL": "Path traversal not allowed in glob pattern {pattern!r}",
        "ERR_GLOB_INVALID": "Invalid glob pattern {pattern!r}: {exc}",
        "OUT_UPDATED_FILE": "Updated file {res.path}",
        "OUT_DELETED": "Deleted {res.path}",
        "OUT_NO_FILES": '"No files found"',
        "OUT_NO_MATCHES": '"No matches found"',
        "EMPTY_CONTENT_WARNING": texts["EMPTY_CONTENT_WARNING"],
        "TRUNCATION_GUIDANCE": texts["TRUNCATION_GUIDANCE"],
    }
    for name, needle in checks.items():
        if needle not in pyfs:
            raise SystemExit(f"{name}: {needle!r} not found in the deepagents sources")
    for name in ("GREP_TRUNCATION_NOTE", "READ_FILE_TRUNCATION_MSG", "NO_LINES_REQUESTED_WARNING"):
        if texts[name] != getattr(fs, name):
            raise SystemExit(f"{name} differs from deepagents")
    if bu.regex_literal_hint("a|b") != texts["GREP_REGEX_HINT"]:
        raise SystemExit("GREP_REGEX_HINT differs from deepagents")
    if texts["TRUNCATION_GUIDANCE"] != bu.TRUNCATION_GUIDANCE:
        raise SystemExit("TRUNCATION_GUIDANCE differs")
    if texts["EMPTY_CONTENT_WARNING"] != bu.EMPTY_CONTENT_WARNING:
        raise SystemExit("EMPTY_CONTENT_WARNING differs")


def check_against_design(texts, schemas, media, a12):
    """Every Appendix A value read from the packages must equal DESIGN.md."""
    problems = []
    a1 = design_blocks("A.1")[0]
    for name in FILE_TOOLS:
        if a1[name]["description"] != texts[name.upper() + "_DESCRIPTION"]:
            problems.append(name + " description")
        if a1[name]["parameters"] != schemas[name.upper() + "_PARAMETERS"]:
            problems.append(name + " parameters")
        if list(a1[name]["parameters"]["properties"]) != list(schemas[name.upper() + "_PARAMETERS"]["properties"]):
            problems.append(name + " parameter order")
    for sec in ("A.2", "A.3", "A.5", "A.6", "A.7", "A.8"):
        for name, value in design_blocks(sec)[0].items():
            if texts.get(name) != value:
                problems.append(f"{sec} {name}")
    a4 = design_blocks("A.4")[0]
    for name, value in a4.items():
        got = schemas.get(name, texts.get(name))
        if got != value:
            problems.append(f"A.4 {name}")
    if bu._EXTENSION_TO_FILE_TYPE.keys() - media.keys():
        problems.append("A.11 extensions")
    if problems:
        raise SystemExit("DESIGN.md Appendix A differs from the packages: " + ", ".join(problems))


# ---------------------------------------------------------------------------
# scripted model
# ---------------------------------------------------------------------------
def completion(content=None, tool_calls=None, total_tokens=2):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {
        "id": "x", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if tool_calls else "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": total_tokens},
    }


def tc(i, name, args):
    return {"id": i, "type": "function", "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}


def fake_model(respond, log=None):
    """ChatOpenAI against a mock transport; `respond(body)` gives the response dict."""

    def handler(request):
        body = json.loads(request.content)
        if log is not None:
            log.append(body)
        return httpx.Response(200, json=respond(body))

    async def ahandler(request):
        await request.aread()
        return handler(request)

    return ChatOpenAI(
        model=MODEL_ID, base_url="https://openrouter.ai/api/v1", api_key="sk-test-not-real",
        default_headers={"X-Title": "QUMO agents"}, timeout=300, max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        http_async_client=httpx.AsyncClient(transport=httpx.MockTransport(ahandler)),
    )


def task_parameters_from_request():
    log = []
    agent = create_deep_agent(model=fake_model(lambda b: completion("done"), log), tools=[], system_prompt="P", name="probe")
    asyncio.run(agent.ainvoke({"messages": [{"role": "user", "content": "go"}]}))
    for t in log[0]["tools"]:
        if t["function"]["name"] == "task":
            return t["function"]["parameters"]
    raise SystemExit("no task tool in the request")


# ---------------------------------------------------------------------------
# fake user tools (FakeEnv in harness/testing.mojo mirrors these)
# ---------------------------------------------------------------------------
@tool
def echo(text: str) -> str:
    """Echo the text back."""
    return text


@tool
def big(n: int) -> str:
    """Return n numbered lines."""
    return "\n".join(f"line{i}" for i in range(1, n + 1))


@tool
def parts(items: list[str]) -> list:
    """Return each item as a text part."""
    return [{"type": "text", "text": s} for s in items]


@tool
def fail() -> str:
    """Always fails."""
    raise ToolException("boom")


fail.handle_tool_error = True
FAKE_TOOLS = [echo, big, parts, fail]


# ---------------------------------------------------------------------------
# fs outputs
# ---------------------------------------------------------------------------
LONG_LINES = "\n".join("L%04d " % i + "x" * 994 for i in range(1, 121)) + "\n"   # 120 lines of 1000 chars
ONE_HUGE_LINE = "y" * 90000
FS_STEPS = [
    [("ls", {"path": "/"})],
    [("read_file", {"file_path": "/missing.txt"})],
    [("edit_file", {"file_path": "/notes.md", "old_string": "a", "new_string": "b"})],
    [("write_file", {"file_path": "/notes.md", "content": "line one\nline two\nalpha beta\n"})],
    [("read_file", {"file_path": "/notes.md"})],
    [("read_file", {"file_path": "/notes.md", "offset": 1, "limit": 1})],
    [("read_file", {"file_path": "/notes.md", "offset": 3})],
    [("read_file", {"file_path": "/notes.md", "offset": 9})],
    [("read_file", {"file_path": "/notes.md", "offset": -2, "limit": 2})],
    [("read_file", {"file_path": "/notes.md", "limit": 0})],
    [("read_file", {"file_path": "/notes.md", "limit": -1})],
    [("edit_file", {"file_path": "/notes.md", "old_string": "alpha", "new_string": "gamma"})],
    [("edit_file", {"file_path": "/notes.md", "old_string": "zzz", "new_string": "q"})],
    [("edit_file", {"file_path": "/notes.md", "old_string": "line", "new_string": "row"})],
    [("edit_file", {"file_path": "/notes.md", "old_string": "line", "new_string": "row", "replace_all": True})],
    [("edit_file", {"file_path": "/notes.md", "old_string": "", "new_string": "row"})],
    [("write_file", {"file_path": "/eof.txt", "content": "a\nb\nb"})],
    [("edit_file", {"file_path": "/eof.txt", "old_string": "a\nb\nb\n", "new_string": "z"})],
    [("edit_file", {"file_path": "/eof.txt", "old_string": "b\n", "new_string": "z"})],
    [("edit_file", {"file_path": "/eof.txt", "old_string": "\n", "new_string": "z"})],
    [("write_file", {"file_path": "/notes.md", "content": "again"})],
    [("write_file", {"file_path": "/src/a.py", "content": "print('alpha')\n"})],
    [("write_file", {"file_path": "/src/app/main.py", "content": "alpha = 1\nbeta = 2\nalpha_beta = 3\n"})],
    [("write_file", {"file_path": "/.github/workflows/ci.yml", "content": "on: push\n"})],
    [("write_file", {"file_path": "/top.py", "content": "x = 'alpha'\n"})],
    [("write_file", {"file_path": "/conf.yaml", "content": "a: 1\n"})],
    [("ls", {"path": "/"})],
    [("ls", {"path": "/src"})],
    [("ls", {"path": "/src/"})],
    [("ls", {"path": "src"})],
    [("ls", {"path": "/src/a.py"})],
    [("ls", {"path": "../etc"})],
    [("ls", {"path": "C:/x"})],
    [("glob", {"pattern": "*.py"})],
    [("glob", {"pattern": "**/*.py"})],
    [("glob", {"pattern": "/*.py"})],
    [("glob", {"pattern": "src/**/*.py"})],
    [("glob", {"pattern": "*.{yml,yaml}"})],
    [("glob", {"pattern": "**/*.yml"})],
    [("glob", {"pattern": ".github/**/*.yml"})],
    [("glob", {"pattern": "*.py", "path": "/src"})],
    [("glob", {"pattern": "*.py", "path": "/src/a.py"})],
    [("glob", {"pattern": "../*.py"})],
    [("glob", {"pattern": "*.rs"})],
    [("glob", {"pattern": "m?in.py"})],
    [("glob", {"pattern": "[ab].py"})],
    [("grep", {"pattern": "alpha"})],
    [("grep", {"pattern": "alpha", "output_mode": "content"})],
    [("grep", {"pattern": "alpha", "output_mode": "count"})],
    [("grep", {"pattern": "alpha", "path": "/src"})],
    [("grep", {"pattern": "alpha", "glob": "*.py", "output_mode": "content"})],
    [("grep", {"pattern": "alpha", "glob": "app/*.py"})],
    [("grep", {"pattern": "alpha", "glob": "../x"})],
    [("grep", {"pattern": "alpha", "max_count": 2, "output_mode": "content"})],
    [("grep", {"pattern": "alpha", "max_count": 4, "output_mode": "count"})],
    [("grep", {"pattern": "nomatch"})],
    [("grep", {"pattern": "foo|bar"})],
    [("grep", {"pattern": "a.*b"})],
    [("grep", {"pattern": "alpha", "path": "../x"})],
    [("read_file", {"file_path": "relative.txt"})],
    [("read_file", {"file_path": "src/a.py"})],
    [("write_file", {"file_path": "/empty.txt", "content": ""})],
    [("read_file", {"file_path": "/empty.txt"})],
    [("read_file", {"file_path": "/empty.txt", "limit": 0})],
    [("write_file", {"file_path": "/ws.txt", "content": "  \n\t\n"})],
    [("read_file", {"file_path": "/ws.txt"})],
    [("write_file", {"file_path": "/blank.txt", "content": "a\n\n\nb\n"})],
    [("read_file", {"file_path": "/blank.txt", "offset": 1, "limit": 2})],
    [("write_file", {"file_path": "/crlf.txt", "content": "one\r\ntwo\rthree\n\u2028four"})],
    [("read_file", {"file_path": "/crlf.txt"})],
    [("read_file", {"file_path": "/crlf.txt", "offset": 1, "limit": 2})],
    [("grep", {"pattern": "two", "path": "/crlf.txt", "output_mode": "content"})],
    [("write_file", {"file_path": "/long.txt", "content": LONG_LINES})],
    [("read_file", {"file_path": "/long.txt", "limit": 200})],
    [("read_file", {"file_path": "/long.txt", "offset": 100, "limit": 200})],
    [("read_file", {"file_path": "/long.txt", "offset": -5, "limit": 100})],
    [("write_file", {"file_path": "/huge.txt", "content": ONE_HUGE_LINE})],
    [("read_file", {"file_path": "/huge.txt"})],
    [("delete", {"file_path": "/src/a.py"})],
    [("delete", {"file_path": "/nope"})],
    [("delete", {"file_path": "/src/"})],
    [("ls", {"path": "/"})],
    [("execute", {"command": "ls"})],
    [("write_file", {"file_path": "/p.txt", "content": "1"}), ("edit_file", {"file_path": "/p.txt", "old_string": "1", "new_string": "2"})],
    [("write_file", {"file_path": "/q.txt", "content": "1"}), ("write_file", {"file_path": "q.txt", "content": "2"}),
     ("delete", {"file_path": "/./q.txt"}), ("write_file", {"file_path": "/r.txt", "content": "3"})],
    [("read_file", {"file_path": "/q.txt"})],
    [("task", {"description": "x", "subagent_type": "nobody"})],
    [("task", {"description": "x", "subagent_type": "general-purpose", "prompt": "extra", "a": 1})],
    [("grep", {"pattern": "x", "max_count": 0})],
    [("grep", {"pattern": "x", "output_mode": "lines"})],
    [("edit_file", {"file_path": "/notes.md", "old_string": "a", "new_string": "b", "replace_all": "maybe"})],
    [("edit_file", {"file_path": "/notes.md", "old_string": "again", "new_string": "b", "replace_all": "yes"})],
    [("ls", {})],
    [("no_such_tool", {"x": 1})],
]


def gen_fs():
    step = {"i": 0, "n": 0}

    def respond(body):
        i = step["i"]
        step["i"] += 1
        if i < len(FS_STEPS):
            calls = []
            for name, args in FS_STEPS[i]:
                step["n"] += 1
                calls.append(tc(f"call_{step['n']}", name, args))
            return completion(None, calls)
        return completion("done")

    agent = create_deep_agent(model=fake_model(respond), tools=[], system_prompt="P", name="probe")
    res = asyncio.run(agent.ainvoke({"messages": [{"role": "user", "content": "go"}]}, config={"recursion_limit": 1000}))
    out = []
    pending = {}
    for m in res["messages"]:
        if m.type == "ai":
            for c in m.tool_calls:
                pending[c["id"]] = (c["name"], c["args"])
        if m.type == "tool":
            out.append({"id": m.tool_call_id, "call": pending.get(m.tool_call_id), "status": m.status, "content": m.content})
    files = {k: v["content"] for k, v in res["files"].items()}
    steps = [[[n, a] for n, a in turn] for turn in FS_STEPS]
    write("fs_outputs.json", {"steps": steps, "results": out, "final_files": files})


# ---------------------------------------------------------------------------
# previews
# ---------------------------------------------------------------------------
def gen_previews():
    inputs = [
        "short",
        "",
        "\n".join(f"line{i}" for i in range(1, 11)),
        "\n".join(f"line{i}" for i in range(1, 12)),
        "\n".join(f"line{i}" for i in range(1, 23)),
        "\n".join(("z" * 1001 if i % 3 == 0 else f"l{i}") for i in range(1, 15)),
        "a" * 1001,
        "one\r\ntwo\r\nthree",
        "a\u2028b\u2029c\x85d\ne",
        "\n".join(f"row {i}" for i in range(1, 120)) + "\n",
    ]
    cases = []
    for s in inputs:
        p = me._create_content_preview(s)
        cases.append({
            "input": s,
            "text": p.text,
            "lines_omitted": p.lines_omitted,
            "lines_clipped": p.lines_clipped,
            "note_result": me._preview_note(lines_omitted=p.lines_omitted, lines_clipped=p.lines_clipped),
            "note_content": me._preview_note(lines_omitted=p.lines_omitted, lines_clipped=p.lines_clipped, subject="content"),
            "tool_stub": me._render_too_large_tool_msg(tool_call_id="call_" + "q" * 40, file_path="/large_tool_results/x", content_str=s),
            "human_stub": me._render_preview_stub(fs._TOO_LARGE_HUMAN_MSG, p, subject="content", file_path="/conversation_history/u.md"),
        })
    ids = ["call.1/2\\3", "toolu_01ABC", "é" * 70, "x" * 129, "x" * 128]
    write("previews.json", {
        "cases": cases,
        "sanitize": [[i, bu.sanitize_tool_call_id(i)] for i in ids],
        "display": [[i, me._visible_tool_call_id(i)] for i in ["a" * 32, "b" * 33, "é" * 40]],
    })


# ---------------------------------------------------------------------------
# token counts
# ---------------------------------------------------------------------------
def ai(content, calls=(), name=None, total=None):
    kw = {}
    if total is not None:
        kw["usage_metadata"] = {"input_tokens": total - 1, "output_tokens": 1, "total_tokens": total}
    kw["response_metadata"] = {"model_provider": "openai"}
    return AIMessage(content=content, name=name,
                     tool_calls=[{"name": n, "args": a, "id": i, "type": "tool_call"} for i, n, a in calls], **kw)


def msg_json(m):
    """A message in the shape test_harness_tokens rebuilds."""
    d = {"type": m.type}
    if isinstance(m.content, str):
        d["content"] = m.content
    else:
        d["parts"] = [b["text"] for b in m.content]
    if getattr(m, "name", None):
        d["name"] = m.name
    if m.type == "ai":
        d["tool_calls"] = [{"id": c["id"], "name": c["name"], "args": c["args"]} for c in m.tool_calls]
        if m.usage_metadata:
            d["total_tokens"] = m.usage_metadata["total_tokens"]
    if m.type == "tool":
        d["tool_call_id"] = m.tool_call_id
    return d


def token_lists():
    big_text = "word " * 3000
    return [
        [SystemMessage("SYS"), HumanMessage("hello")],
        [SystemMessage("SYS"), HumanMessage("hello"), ai("", [("call_1", "ls", {"path": "/"})], name="research-watch"),
         ToolMessage(content="No files found", name="ls", tool_call_id="call_1")],
        [SystemMessage("SYS"), HumanMessage("hello"), ai("ok", name="a", total=110)],
        [SystemMessage("SYS"), HumanMessage("hi é 日本"), ai("x", [("c1", "write_file", {"file_path": "/é.md", "content": "naïve — \U0001F600", "n": 1.5, "b": True, "z": None})], name="o"),
         ToolMessage(content="Updated file /é.md", name="write_file", tool_call_id="c1"), ai("done", name="o", total=500)],
        [SystemMessage("P"), HumanMessage(big_text), ai("", [("c1", "echo", {"text": "a"}), ("c2", "big", {"n": 3})], name="o", total=2000),
         ToolMessage(content="a", name="echo", tool_call_id="c1"), ToolMessage(content="line1\nline2\nline3", name="big", tool_call_id="c2")],
        [SystemMessage("P"), HumanMessage("x"), ai("", [("c1", "parts", {"items": ["p", "q"]})], name="o"),
         ToolMessage(content=[{"type": "text", "text": "pp"}, {"type": "text", "text": "qqq"}], name="parts", tool_call_id="c1")],
        [SystemMessage("P"), HumanMessage("x"), ai("a", name="o", total=10), HumanMessage("y" * 400), ai("b", name="o", total=50)],
        [SystemMessage("P"), HumanMessage("x"), ai("a", name="o", total=1000)],
        [SystemMessage(""), HumanMessage("")],
        [SystemMessage("P"), HumanMessage("x"), ai("", [("c1", "task", {"description": "d", "subagent_type": "general-purpose"})], name="o"),
         ToolMessage(content="answer", tool_call_id="c1")],
    ]


def gen_tokens():
    m = fs.FilesystemMiddleware()
    file_tools = [t for t in m.tools if t.name in FILE_TOOLS]
    task = None
    log = []
    agent = create_deep_agent(model=fake_model(lambda b: completion("done"), log), tools=[], system_prompt="P", name="probe")
    asyncio.run(agent.ainvoke({"messages": [{"role": "user", "content": "go"}]}))
    task_wire = [t for t in log[0]["tools"] if t["function"]["name"] == "task"][0]
    tool_sets = {"none": [], "fake": FAKE_TOOLS, "files": file_tools}
    cases = []
    for li, msgs in enumerate(token_lists()):
        for ts_name, ts in tool_sets.items():
            if ts_name != "none" and li not in (0, 1, 4):
                continue
            cases.append({
                "messages": [msg_json(x) for x in msgs],
                "tools": ts_name,
                "scaled": count_tokens_approximately(msgs, tools=ts or None, use_usage_metadata_scaling=True),
                "unscaled": count_tokens_approximately(msgs, tools=ts or None),
            })
    count_json = {}
    for t in list(file_tools) + FAKE_TOOLS:
        schema = t.tool_call_schema.model_json_schema()
        params = strip_title_description(schema)
        count_json[t.name] = {"name": t.name, "description": t.description, "parameters": params}
    write("token_counts.json", {"cases": cases, "count_json": count_json, "task_wire": task_wire["function"]})


# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------
def gen_summaries():
    from langchain.agents.middleware.summarization import SummarizationMiddleware as LSM
    lists = [
        [HumanMessage('hi <x> & "q"'), ai("thinking", [("call_1", "ls", {"path": "/é"})]), ToolMessage(content="a\nb", tool_call_id="call_1"), ai("done")],
        [HumanMessage("task"), ai("", [("c" * 70, "echo", {"text": "<&>"})], name="o"), ToolMessage(content="<&>", name="echo", tool_call_id="c" * 70),
         ai("", [("c2", "parts", {"items": ["a"]})], name="o"),
         ToolMessage(content=[{"type": "text", "text": "x"}, {"type": "text", "text": "y"}], name="parts", tool_call_id="c2"), ai("end", name="o")],
        [SystemMessage("sys"), HumanMessage("a"), ai("b")],
    ]
    xml = [{"messages": [msg_json(m) for m in ms], "xml": get_buffer_string(ms, format="xml")} for ms in lists]
    prompt = dsm.DEEPAGENTS_DEFAULT_SUMMARY_PROMPT.format(messages=xml[0]["xml"]).rstrip()

    lsm = LSM.__new__(LSM)
    cut_lists = []
    calls = lambda i: [(f"t{i}a", "echo", {"text": "a"}), (f"t{i}b", "echo", {"text": "b"})]
    base = [HumanMessage("u0")]
    for i in range(6):
        base += [ai("", calls(i), name="o"), ToolMessage("ra", name="echo", tool_call_id=f"t{i}a"), ToolMessage("rb", name="echo", tool_call_id=f"t{i}b")]
    base += [ai("final", name="o")]
    for n in (3, 6, 7, 8, 9, 10, 11, 12, len(base)):
        ms = base[:n]
        cut_lists.append({"messages": [msg_json(m) for m in ms], "cutoff": LSM._find_safe_cutoff(lsm, ms, 6)})
    orphan = [HumanMessage("a"), ai("b"), ToolMessage("x", tool_call_id="zz"), ToolMessage("y", tool_call_id="zz2"), ai("c"), HumanMessage("d"), ai("e"), HumanMessage("f"), ai("g")]
    cut_lists.append({"messages": [msg_json(m) for m in orphan], "cutoff": LSM._find_safe_cutoff(lsm, orphan, 6)})

    # overflow retry prompt: LC:995-1026 with count_tokens_approximately (no scaling)
    from langchain_core.messages.utils import count_tokens_approximately as cta
    overflow = []
    for text in [xml[0]["xml"], xml[1]["xml"] * 7, "x"]:
        tokens = cta([HumanMessage(content=text)])
        budget = tokens // 2
        keep = len(text) * budget // max(tokens, 1)
        if keep <= 0:
            overflow.append({"transcript": text, "prompt": None})
            continue
        head = keep // 4
        tail = keep - head
        overflow.append({"transcript": text, "prompt": f"{text[:head]}\n\n[... middle of conversation omitted ...]\n\n{text[-tail:] if tail else ''}"})

    # _truncate_args
    dmw = dsm.SummarizationMiddleware.__new__(dsm.SummarizationMiddleware)
    dmw._truncate_args_trigger = ("messages", 20)
    dmw._truncate_args_keep = ("messages", 20)
    dmw._max_arg_length = 2000
    dmw._truncation_text = "...(argument truncated)"
    longs = []
    ms = [HumanMessage("start")]
    for i in range(12):
        ms.append(ai("", [(f"w{i}", "write_file", {"file_path": f"/f{i}.md", "content": "c" * 2001, "short": "s"}),
                          (f"e{i}", "edit_file", {"file_path": f"/f{i}.md", "old_string": "o" * 2500, "new_string": "n" * 10}),
                          (f"x{i}", "echo", {"text": "t" * 3000})], name="o"))
        ms.append(ToolMessage("ok", name="write_file", tool_call_id=f"w{i}"))
        ms.append(ToolMessage("ok", name="edit_file", tool_call_id=f"e{i}"))
        ms.append(ToolMessage("ok", name="echo", tool_call_id=f"x{i}"))
    truncated = []
    for n in (19, 20, 21, 30, len(ms)):
        sub = ms[:n]
        out, changed = dsm.SummarizationMiddleware._truncate_args(dmw, sub, 0)
        truncated.append({"n": n, "changed": changed, "messages": [msg_json(m) for m in out]})
    write("summaries.json", {
        "xml": xml,
        "summary_prompt_first": prompt,
        "cutoffs": cut_lists,
        "overflow_prompts": overflow,
        "truncate_args": {"input": [msg_json(m) for m in ms], "results": truncated},
    })


# ---------------------------------------------------------------------------
# pydantic errors (Appendix C.2 and more)
# ---------------------------------------------------------------------------
def gen_pydantic():
    from pydantic import ValidationError
    from qumo_agents.tools import repo
    tools = {"read_file": repo.read_file, "run_check": repo.run_check, "write_report": repo.write_report}
    m = fs.FilesystemMiddleware()
    for t in m.tools:
        if t.name in ("grep", "edit_file", "glob", "ls"):
            tools["da_" + t.name] = t
    tools["echo"] = echo
    tools["big"] = big
    tools["parts"] = parts
    cases = [
        ("read_file", {}), ("read_file", {"path": 5}), ("read_file", {"path": None}),
        ("read_file", {"path": "x", "start_line": "abc"}), ("read_file", {"path": "x", "start_line": 1.5}),
        ("read_file", {"path": "x", "start_line": "7"}), ("read_file", {"path": "x", "start_line": 2.0}),
        ("read_file", {"path": "x", "extra": 1}), ("read_file", {"path": "x", "start_line": True}),
        ("read_file", {"path": "x", "start_line": " 12 "}), ("read_file", {"path": "x", "max_lines": "1e3"}),
        ("read_file", {"path": "x", "max_lines": [1]}), ("read_file", {"path": ["a"], "start_line": "q"}),
        ("run_check", {"argv": "git"}), ("run_check", {"argv": [1]}), ("run_check", {"argv": ["git", None, 3]}),
        ("run_check", {"argv": ["git", "status"], "timeout_s": "30"}), ("run_check", {}),
        ("da_grep", {"pattern": "x", "max_count": 0}), ("da_grep", {"pattern": "x", "output_mode": "lines"}),
        ("da_grep", {"pattern": "x", "max_count": "3"}), ("da_grep", {"pattern": "x", "max_count": None}),
        ("da_grep", {"pattern": "x", "path": 3}), ("da_grep", {"pattern": "x", "max_count": -1, "output_mode": 5}),
        ("da_edit_file", {"file_path": "/a", "old_string": "a", "new_string": "b", "replace_all": "maybe"}),
        ("da_edit_file", {"file_path": "/a", "old_string": "a", "new_string": "b", "replace_all": "yes"}),
        ("da_edit_file", {"file_path": "/a", "old_string": "a", "new_string": "b", "replace_all": 1}),
        ("da_edit_file", {"file_path": "/a", "old_string": "a", "new_string": "b", "replace_all": 2}),
        ("da_edit_file", {"file_path": "/a", "old_string": "a", "new_string": "b", "replace_all": "OFF"}),
        ("da_edit_file", {"file_path": "/a"}),
        ("da_glob", {"pattern": "*", "path": None}), ("da_ls", {"path": 1.5}),
        ("write_report", {}), ("echo", {"bad": 1}), ("echo", {"text": 1}), ("big", {"n": "12"}), ("big", {"n": 1e20}),
        ("parts", {"items": ["a", 2]}), ("parts", {"items": "ab"}),
    ]
    out = []
    for name, args in cases:
        t = tools[name]
        params = convert_to_openai_tool(t)["function"]["parameters"]
        errors = []
        coerced = None
        try:
            v = t.args_schema.model_validate(args)
            coerced = v.model_dump() if hasattr(v, "model_dump") else dict(v)
        except ValidationError as e:
            for err in e.errors():
                errors.append(".".join(str(x) for x in err["loc"]) + ": " + err["msg"])
        out.append({"tool": name, "parameters": params, "args": args, "errors": errors, "coerced": coerced})
    write("pydantic_errors.json", {"cases": out})


# ---------------------------------------------------------------------------
# wire scenarios
# ---------------------------------------------------------------------------
UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|session_[0-9a-f]{32}")
TS_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{6})?\+00:00")


def placeholders(obj):
    if isinstance(obj, str):
        return TS_RE.sub("<TS>", UUID_RE.sub("<UUID>", obj))
    if isinstance(obj, list):
        return [placeholders(x) for x in obj]
    if isinstance(obj, dict):
        return {k: placeholders(v) for k, v in obj.items()}
    return obj


def run_wire(name, respond, at_of=lambda body: ""):
    """Runs the scenario and writes wire_<name>.json (request bodies, placeholders applied)
    and wire_<name>.replay.jsonl (the same responses as ReplayModel lines, routed by `at`)."""
    log = []
    replay = []

    def recording(body):
        resp = respond(body)
        replay.append({"at": at_of(body), "response": resp})
        return resp

    model = fake_model(recording, log)
    sub = create_deep_agent(model=model, tools=FAKE_TOOLS, system_prompt="SUB PROMPT", name="research-watch")
    main = create_deep_agent(
        model=model, tools=FAKE_TOOLS, system_prompt="MAIN PROMPT",
        subagents=[CompiledSubAgent(name="research-watch", description="Watches research.", runnable=sub)],
        name="orchestrator",
    )
    res = asyncio.run(main.ainvoke({"messages": [{"role": "user", "content": "TASK"}]}, config={"recursion_limit": 200}))
    answer = res["messages"][-1].content
    write(f"wire_{name}.json", placeholders({"requests": log, "answer": answer,
                                             "files": sorted(res.get("files", {}).keys())}))
    path = HERE / f"wire_{name}.replay.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in replay), encoding="utf-8")
    print(f"wrote {path.relative_to(REPO)}")


def first_user(body):
    for m in body["messages"]:
        if m["role"] == "user":
            return m["content"]
    return None


def system_of(body):
    m = body["messages"][0]
    return m["content"] if m["role"] == "system" else None


def turn_of(body):
    """Number of assistant messages already in the request."""
    return sum(1 for m in body["messages"] if m["role"] == "assistant")


def gen_wire():
    # S1: one tool call, then the answer.
    def s1(b):
        if turn_of(b) == 0:
            return completion(None, [tc("c1", "echo", {"text": "héllo"})])
        return completion("final S1")
    run_wire("S1", s1)

    # S2: four parallel calls: echo ok, ls, unknown name, validation error; plus fail.
    def s2(b):
        if turn_of(b) == 0:
            return completion(None, [tc("c1", "echo", {"text": "a"}), tc("c2", "ls", {"path": "/"}), tc("c3", "no_such_tool", {}),
                                     tc("c4", "echo", {"bad": 1}), tc("c5", "fail", {})])
        return completion("final S2")
    run_wire("S2", s2)

    # S3: two parallel task calls (named and general-purpose); answers keyed by system prompt.
    def s3(b):
        sp = system_of(b)
        if sp == "SUB PROMPT":
            return completion("sub-answer-1")
        if sp == sa.DEFAULT_SUBAGENT_PROMPT:
            return completion("sub-answer-2")
        if turn_of(b) == 0:
            return completion(None, [tc("t1", "task", {"description": "Do sub work", "subagent_type": "research-watch"}),
                                     tc("t2", "task", {"description": "GP work", "subagent_type": "general-purpose"})])
        return completion("final S3")

    def s3_at(b):
        sp = system_of(b)
        if sp == "SUB PROMPT":
            return "t1"
        if sp == sa.DEFAULT_SUBAGENT_PROMPT:
            return "t2"
        return ""
    run_wire("S3", s3, s3_at)

    # S4: eviction of a large result, read of the evicted path, a large parts result.
    def s4(b):
        t = turn_of(b)
        if t == 0:
            return completion(None, [tc("c1", "big", {"n": 12000})])
        if t == 1:
            return completion(None, [tc("c2", "read_file", {"file_path": "/large_tool_results/c1", "offset": 10, "limit": 5})])
        if t == 2:
            return completion(None, [tc("c3", "parts", {"items": ["a" * 45000, "b" * 45000]})])
        return completion("final S4")
    run_wire("S4", s4)

    # S5: summarization triggered by the reported usage.total_tokens = 170000.
    s5_calls = {"n": 0}

    def s5(b):
        if "tools" not in b:
            return completion("SUMMARY TEXT")
        t = s5_calls["n"]
        s5_calls["n"] += 1
        if t < 4:
            return completion(None, [tc(f"c{t}", "echo", {"text": f"step {t}"})], total_tokens=100 if t < 3 else 170000)
        return completion("final S5")
    run_wire("S5", s5)

    # S6: general-purpose subagent that calls a tool.
    def s6(b):
        sp = system_of(b)
        if sp == sa.DEFAULT_SUBAGENT_PROMPT:
            if turn_of(b) == 0:
                return completion(None, [tc("g1", "echo", {"text": "from gp"})])
            return completion("gp done  \n")
        if turn_of(b) == 0:
            return completion(None, [tc("t1", "task", {"description": "Use echo", "subagent_type": "general-purpose"})])
        return completion("final S6")
    run_wire("S6", s6, lambda b: "t1" if system_of(b) == sa.DEFAULT_SUBAGENT_PROMPT else "")


SECTIONS = {
    "texts": lambda: write("deepagents_golden.json", gen_texts()),
    "fs": gen_fs,
    "previews": gen_previews,
    "tokens": gen_tokens,
    "summaries": gen_summaries,
    "pydantic": gen_pydantic,
    "wire": gen_wire,
}

if __name__ == "__main__":
    for s in (sys.argv[1:] or list(SECTIONS)):
        SECTIONS[s]()
