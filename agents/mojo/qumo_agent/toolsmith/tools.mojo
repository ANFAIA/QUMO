"""The toolsmith tools: toolsmith_new, toolsmith_build, toolsmith_test and
toolsmith_publish (DESIGN.md 6.3, 6.4), plus the rebuild that `qumo-agent tools review`
runs (6.6 step 2).

Every tool works on `<sandbox>/agents/mojo/tools/<name>/` and needs an open sandbox
and a valid name. Model-written code only ever runs confined (6.8):

- builds copy `tool.mojo` alone into a fresh temporary directory and compile it with
  the main checkout's include paths (`-I <repo_root> -I <repo_root>/agents/mojo`),
  never the sandbox's, under `CONF_BUILD`, with a fresh `MODULAR_CACHE_DIR` inside
  that directory;
- tests run a private copy of the binary per case, with an empty environment, in a
  fresh temporary directory, under `CONF_TOOL`.

Confinement convention of the module-level functions: `hermetic_build` confines the
compiler when `self_binary` (the `__confine` trampoline) is not empty and runs it
unconfined when it is empty; `run_cases` confines when `confined` is true. Only a
human's `--allow-unsandboxed-tool-tests` may lead a caller to the unconfined forms.

The tools return text for the model and never raise: refusals start with
`refused: `, unexpected failures with `error: `.
"""

from std.os import getenv

from qumo.ai.json import JsonValue, JsonWriter, parse_json

from qumo_agent.core.errors import E_TOOL_REFUSED
from qumo_agent.core.toolspec import ToolSpec, TOOL_TOOLSMITH
from qumo_agent.native.clock import monotonic_ms, utc_iso_now
from qumo_agent.native.fs import (
    copy_file, exists, is_file, join_path, make_dirs, make_temp_dir, read_text_lossy,
    realpath_strict, remove_tree, resolve_nonstrict, write_text_atomic,
)
from qumo_agent.native.sandbox import Confinement, CONF_BUILD, CONF_TOOL, confine, modular_lib_dir
from qumo_agent.native.spawn import (
    SpawnSpec, STDIO_NULL, STDIO_PIPE, minimal_child_env, run_capture, which,
)
from qumo_agent.mcp.stdio_client import McpStdioSession, StdioServerParams
from qumo_agent.util.pyfmt import cp_head, cp_len, cp_tail, py_str_repr
from qumo_agent.util.sha256 import sha256_file_hex
from qumo_agent.toolsmith.manifest import (
    BuildRecord, ToolEntry, DEFAULT_MAX_OUTPUT_BYTES, DEFAULT_TOOL_TIMEOUT_S,
    build_record_from_json, build_record_from_value, check_entry_shape, current_platform,
    entry_dir, is_inside, load_manifest, sdk_sha256, valid_tool_name, valid_tool_version,
    verify_entry,
)
from qumo_agent.toolsmith.templates import cases_template, tool_template

comptime CASES_SCHEMA = "qumo.tool_cases.v1"
comptime TOOLS_REL_DIR = "agents/mojo/tools"
comptime PUBLISHED_BY = "mojo-toolsmith"
# The sandbox kind a caller passes when a human gave --allow-unsandboxed-tool-tests on a
# machine without an OS sandbox: build and test unconfined (6.8).
comptime SANDBOX_KIND_UNSANDBOXED = "unsandboxed"

comptime REPORT_CAP = 12000                     # code points, build and test results
comptime STDERR_TAIL_CAP = 2000                 # code points of a failing case's stderr
comptime BUILD_TIMEOUT_MIN = 60
comptime BUILD_TIMEOUT_MAX = 1800
comptime TEST_TIMEOUT_MIN = 1
comptime TEST_TIMEOUT_MAX = 120
comptime BUILD_FSIZE_BYTES = 1073741824         # RLIMIT_FSIZE of a confined build (1 GiB)
comptime TOOL_FSIZE_BYTES = 67108864            # RLIMIT_FSIZE of a confined tool (64 MiB)
comptime CAPTURE_MAX_BYTES = 4194304            # compiler output kept per stream
comptime TOOL_MAX_MESSAGE_BYTES = 4194304       # one JSON-RPC line from a tool under test
comptime MAX_CASES = 200

comptime REFUSE_NO_SANDBOX = "refused: call open_sandbox first; the main checkout is read-only for agents"
comptime REFUSE_NAME = "refused: tool names must match gen_[a-z0-9_]{2,40}"
comptime REFUSE_VERSION = "refused: version must look like 1.2.3 or 1.2.3-rc1"
comptime REFUSE_STALE = "refused: tool.mojo, cases.json or the binary changed after the last passing toolsmith_test; build and test again"
comptime PUBLISH_CASE_TIMEOUT_S = 30    # per case, for the re-run inside toolsmith_publish
comptime REFUSE_REBUILD = "refused: build the tool again"
comptime REFUSE_NO_OS_SANDBOX_TEST = "refused: no OS sandbox on this machine, so unreviewed tools cannot be tested. Hint: a human can pass --allow-unsandboxed-tool-tests"
comptime REFUSE_NO_OS_SANDBOX_BUILD = "refused: no OS sandbox on this machine, so model-written code cannot be built. Hint: a human can pass --allow-unsandboxed-tool-tests"


# ---------------------------------------------------------------------------
# Tool specs (Appendix B.4)
# ---------------------------------------------------------------------------
def toolsmith_specs() -> List[ToolSpec]:
    var specs = List[ToolSpec]()
    specs.append(ToolSpec(
        "toolsmith_new",
        "Create a generated Mojo tool in the open sandbox: agents/mojo/tools/<name>/tool.mojo (the program, built on qumo_agent.tool_sdk) and agents/mojo/tools/<name>/cases.json (its test cases). Without source or cases it writes templates to finish with edit_file. Call open_sandbox first. Names start with gen_ and use a-z, 0-9 and _.",
        "{\"properties\":{\"name\":{\"description\":\"Tool name: gen_ followed by 2-40 characters from a-z, 0-9 and _.\",\"type\":\"string\"},\"description\":{\"description\":\"One paragraph: what the tool computes, its inputs and its output.\",\"type\":\"string\"},\"source\":{\"default\":\"\",\"description\":\"Complete Mojo source of tool.mojo; empty writes the template.\",\"type\":\"string\"},\"cases\":{\"default\":\"\",\"description\":\"Content of cases.json (schema qumo.tool_cases.v1); empty writes the template.\",\"type\":\"string\"},\"overwrite\":{\"default\":false,\"description\":\"Replace files that already exist.\",\"type\":\"boolean\"}},\"required\":[\"name\",\"description\"],\"type\":\"object\"}",
        TOOL_TOOLSMITH, True, String()))
    specs.append(ToolSpec(
        "toolsmith_build",
        "Compile agents/mojo/tools/<name>/tool.mojo in the open sandbox with .venv/bin/mojo build. Returns the exit code and the compiler diagnostics (last 12000 characters) and, on success, the binary path and its sha256.",
        "{\"properties\":{\"name\":{\"description\":\"Tool name, as given to toolsmith_new.\",\"type\":\"string\"},\"timeout_s\":{\"default\":600,\"description\":\"Build timeout in seconds (60-1800).\",\"type\":\"integer\"}},\"required\":[\"name\"],\"type\":\"object\"}",
        TOOL_TOOLSMITH, True, String()))
    specs.append(ToolSpec(
        "toolsmith_test",
        "Run the built tool against its cases.json: each case starts the binary with an empty environment in a fresh temporary directory, calls it over MCP and checks the expectation. Returns PASS or FAIL with one line per case. A passing run is recorded for toolsmith_publish.",
        "{\"properties\":{\"name\":{\"description\":\"Tool name, as given to toolsmith_new.\",\"type\":\"string\"},\"timeout_s\":{\"default\":30,\"description\":\"Timeout per case in seconds (1-120).\",\"type\":\"integer\"}},\"required\":[\"name\"],\"type\":\"object\"}",
        TOOL_TOOLSMITH, True, String()))
    specs.append(ToolSpec(
        "toolsmith_publish",
        "Add the tested tool to the generated-tool manifest with the sha256 of its source, cases and binary, marked reviewed: false. Needs a passing toolsmith_test of the current binary. Agents can use it only after a human runs `qumo-agent tools review <name>`.",
        "{\"properties\":{\"name\":{\"description\":\"Tool name, as given to toolsmith_new.\",\"type\":\"string\"},\"version\":{\"default\":\"0.1.0\",\"description\":\"Version recorded in the manifest: MAJOR.MINOR.PATCH with up to 4 digits each, optionally followed by . or - and up to 16 letters or digits (e.g. 0.1.0, 1.2.3-rc1).\",\"type\":\"string\"}},\"required\":[\"name\"],\"type\":\"object\"}",
        TOOL_TOOLSMITH, True, String()))
    return specs^


# ---------------------------------------------------------------------------
# Test cases (qumo.tool_cases.v1)
# ---------------------------------------------------------------------------
@fieldwise_init
struct ToolCase(Copyable):
    var name: String
    var arguments_json: String
    var expect_error: Bool
    var has_text_equals: Bool
    var text_equals: String
    var text_contains: List[String]
    var has_json_equals: Bool
    var json_equals: String          # JSON text of the expected value


def _string_member(v: JsonValue, key: String, what: String) raises -> String:
    var f = v.get(key)
    if not f.is_string():
        raise Error(what + " must be a string")
    return f.as_string()


def parse_cases(text: String, tool_name: String) raises -> List[ToolCase]:
    """Parse and check a cases.json text; raises a plain message on any problem."""
    var v: JsonValue
    try:
        v = parse_json(text)
    except e:
        raise Error("not valid JSON: " + String(e))
    if not v.is_object():
        raise Error("must be a JSON object")
    var schema = v.get("schema")
    if not schema.is_string() or schema.as_string() != CASES_SCHEMA:
        raise Error("schema must be \"" + String(CASES_SCHEMA) + "\"")
    var tool = v.get("tool")
    if not tool.is_string() or tool.as_string() != tool_name:
        raise Error("tool must be \"" + tool_name + "\"")
    var cases = v.get("cases")
    if not cases.is_array() or len(cases) == 0:
        raise Error("cases must be a non-empty list")
    if len(cases) > MAX_CASES:
        raise Error("at most " + String(MAX_CASES) + " cases")
    var out_cases = List[ToolCase]()
    for i in range(len(cases)):
        var c = cases.at(i)
        var where = "case " + String(i + 1)
        if not c.is_object():
            raise Error(where + " must be an object")
        var name = _string_member(c, "name", where + ": name")
        if name.byte_length() == 0:
            raise Error(where + ": name must not be empty")
        where = "case " + py_str_repr(name)
        for j in range(len(out_cases)):
            if out_cases[j].name == name:
                raise Error(where + " appears twice")
        for key in c.keys():
            if key != "name" and key != "arguments" and key != "expect":
                raise Error(where + ": unknown key " + py_str_repr(key))
        var args = c.get("arguments")
        var args_json = String("{}")
        if args.is_defined():
            if not args.is_object():
                raise Error(where + ": arguments must be an object")
            args_json = args.to_json()
        var expect = c.get("expect")
        var expect_error = False
        var has_text_equals = False
        var text_equals = String()
        var contains = List[String]()
        var has_json_equals = False
        var json_equals = String()
        if expect.is_defined():
            if not expect.is_object():
                raise Error(where + ": expect must be an object")
            for key in expect.keys():
                if key == "is_error":
                    var b = expect.get(key)
                    if not b.is_bool():
                        raise Error(where + ": expect.is_error must be true or false")
                    expect_error = b.as_bool()
                elif key == "text_equals":
                    text_equals = _string_member(expect, key, where + ": expect.text_equals")
                    has_text_equals = True
                elif key == "text_contains":
                    var t = expect.get(key)
                    if t.is_string():
                        contains.append(t.as_string())
                    elif t.is_array():
                        for k in range(len(t)):
                            if not t.at(k).is_string():
                                raise Error(where + ": expect.text_contains must be a string or a list of strings")
                            contains.append(t.at(k).as_string())
                    else:
                        raise Error(where + ": expect.text_contains must be a string or a list of strings")
                elif key == "json_equals":
                    json_equals = expect.get(key).to_json()
                    has_json_equals = True
                else:
                    raise Error(where + ": unknown expect key " + py_str_repr(key) + " (use is_error, text_equals, text_contains, json_equals)")
        out_cases.append(ToolCase(name, args_json, expect_error, has_text_equals, text_equals,
                                  contains^, has_json_equals, json_equals))
    return out_cases^


def _is_integer_text(t: String) -> Bool:
    for b in t.as_bytes():
        var c = Int(b)
        if c == 0x2E or c == 0x65 or c == 0x45:
            return False
    return True


def json_equal(a: JsonValue, b: JsonValue) raises -> Bool:
    """Python `==` on json.loads results: object key order ignored, numbers compared
    exactly (integers as integers, otherwise as floats, so 1 == 1.0)."""
    var ka = a.kind()
    var kb = b.kind()
    if a.is_number() and b.is_number():
        var ta = a.number_text()
        var tb = b.number_text()
        if _is_integer_text(ta) and _is_integer_text(tb):
            return ta == tb
        return a.as_float() == b.as_float()
    if ka != kb:
        return False
    if a.is_null():
        return True
    if a.is_bool():
        return a.as_bool() == b.as_bool()
    if a.is_string():
        return a.as_string() == b.as_string()
    if a.is_array():
        if len(a) != len(b):
            return False
        for i in range(len(a)):
            if not json_equal(a.at(i), b.at(i)):
                return False
        return True
    if a.is_object():
        var keys_a = a.keys()
        var keys_b = b.keys()
        if len(keys_a) != len(keys_b):
            return False
        for k in keys_a:
            if not b.has(k):
                return False
            if not json_equal(a.get(k), b.get(k)):
                return False
        return True
    return False


def _quote_head(text: String, n: Int) -> String:
    if cp_len(text) > n:
        return py_str_repr(cp_head(text, n)) + "..."
    return py_str_repr(text)


def check_expectation(c: ToolCase, is_error: Bool, text: String) -> String:
    """"" when the result meets the case's expectation, else the reason."""
    if is_error != c.expect_error:
        var want = "true" if c.expect_error else "false"
        var got = "true" if is_error else "false"
        return "expected is_error " + want + ", got " + got + ": " + _quote_head(text, 300)
    if c.has_text_equals and text != c.text_equals:
        return "text is " + _quote_head(text, 300) + ", expected " + _quote_head(c.text_equals, 300)
    for s in c.text_contains:
        if text.find(s) < 0:
            return "text does not contain " + _quote_head(s, 200) + ": " + _quote_head(text, 300)
    if c.has_json_equals:
        var got: JsonValue
        try:
            got = parse_json(text)
        except:
            return "text is not JSON: " + _quote_head(text, 300)
        try:
            if not json_equal(got, parse_json(c.json_equals)):
                return "JSON differs: got " + _quote_head(got.to_json(), 300) + ", expected " + _quote_head(c.json_equals, 300)
        except e:
            return "JSON comparison failed: " + String(e)
    return String()


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _dirname(path: String) -> String:
    var i = path.rfind("/")
    if i <= 0:
        return "/" if i == 0 else "."
    return String(path[byte=0:i])


def _one_line(s: String) -> String:
    """`s` with control characters replaced by spaces (one result line per case)."""
    var res = String()
    var bs = s.as_bytes()
    var run = 0
    for i in range(len(bs)):
        var c = Int(bs[i])
        if c < 0x20 or c == 0x7F:
            res += String(unsafe_from_utf8=bs[run:i])
            res += " "
            run = i + 1
    res += String(unsafe_from_utf8=bs[run:len(bs)])
    return res^


def _clamp(v: Int, lo: Int, hi: Int) -> Int:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def _home_realpath() -> String:
    var home = getenv("HOME", "")
    if home.byte_length() == 0:
        return String()
    try:
        return realpath_strict(home)
    except:
        return home


def _env_with(base: List[String], name: String, value: String) -> List[String]:
    """`base` without any `name=` entry, plus `name=value`."""
    var res = List[String]()
    var prefix = name + "="
    for e in base:
        if not e.startswith(prefix):
            res.append(e)
    res.append(prefix + value)
    return res^


def _drop_crashpad_lines(text: String) -> String:
    """The Modular toolchain prints a harmless "Failed to initialize Crashpad" line
    on every run (docs/MOJO_GUIDE.md); drop it from diagnostics."""
    if text.find("Failed to initialize Crashpad") < 0:
        return text
    var kept = List[String]()
    for line in text.split("\n"):
        if not String(line).startswith("Failed to initialize Crashpad"):
            kept.append(String(line))
    return String("\n").join(kept)


def mojo_version(mojo_bin: String) -> String:
    """First line of `mojo --version`, e.g. "Mojo 1.2.0 (2016d0cd)"; "" on failure.
    The compiler itself is trusted, so this runs unconfined (fixed arguments)."""
    try:
        var spec = SpawnSpec(mojo_bin, [mojo_bin, "--version"], minimal_child_env(), String(),
                             STDIO_NULL, STDIO_PIPE, STDIO_PIPE, True)
        var res = run_capture(spec, 60000, 65536, String())
        if res.exit_code != 0:
            return String()
        for line in _drop_crashpad_lines(res.stdout).split("\n"):
            var s = String(String(line).strip())
            if s.byte_length() > 0:
                return s
    except:
        pass
    return String()


# ---------------------------------------------------------------------------
# Hermetic build (6.4)
# ---------------------------------------------------------------------------
def _build_read_allow(repo_root: String) -> List[String]:
    """Subtrees the compiler reads that must stay readable even below a denied $HOME
    directory: the main checkout and its toolchain (`.venv` may be a symlink)."""
    var out = List[String]()
    for p in [repo_root, join_path(repo_root, ".venv")]:
        try:
            out.append(realpath_strict(p))
        except:
            pass
    return out^


def hermetic_build(repo_root: String, mojo_bin: String, source_path: String, name: String, timeout_s: Int, self_binary: String) raises -> Tuple[Int, String, String, BuildRecord]:
    """Compile `source_path` alone in a fresh temporary directory against the main
    checkout's include paths. Returns (exit code, diagnostics, temporary binary path,
    build record without binary_sha256). Diagnostics are the last 12,000 characters of
    stdout+stderr with the temporary directory shown as `agents/mojo/tools/<name>/`.
    The caller removes the temporary directory (the parent of the binary path).
    Confined under CONF_BUILD when `self_binary` is not empty."""
    var tmp = make_temp_dir("qumo-build-")
    try:
        var tmp_real = realpath_strict(tmp)
        var src = join_path(tmp_real, "tool.mojo")
        copy_file(source_path, src, 0o600)
        var source_sha = sha256_file_hex(src)
        var sdk = sdk_sha256(repo_root)
        var version = mojo_version(mojo_bin)
        var cache = join_path(tmp_real, "modular-cache")
        make_dirs(cache)
        var out_bin = join_path(tmp_real, name)
        var argv: List[String] = [
            mojo_bin, "build", "-I", repo_root, "-I", join_path(repo_root, "agents/mojo"),
            "-o", out_bin, src,
        ]
        var env = _env_with(minimal_child_env(), "TMPDIR", tmp_real + "/")
        env = _env_with(env, "MODULAR_CACHE_DIR", cache)
        var spec = SpawnSpec(mojo_bin, argv^, env^, tmp_real, STDIO_NULL, STDIO_PIPE, STDIO_PIPE, True)
        if self_binary.byte_length() > 0:
            var conf = Confinement(CONF_BUILD, [tmp_real], _build_read_allow(repo_root), _home_realpath(), 0, BUILD_FSIZE_BYTES)
            spec = confine(spec, conf, self_binary)
        var t = _clamp(timeout_s, BUILD_TIMEOUT_MIN, BUILD_TIMEOUT_MAX)
        var res = run_capture(spec, t * 1000, CAPTURE_MAX_BYTES, String())
        var text = res.stdout
        if text.byte_length() > 0 and res.stderr.byte_length() > 0 and not text.endswith("\n"):
            text += "\n"
        text += res.stderr
        text = _drop_crashpad_lines(text)
        var shown = String(TOOLS_REL_DIR) + "/" + name + "/"
        text = text.replace(tmp_real + "/", shown)
        if tmp != tmp_real:
            text = text.replace(tmp + "/", shown)
        if res.timed_out:
            text += "\n[build timed out after " + String(t) + " s]"
        var rc = res.exit_code
        if rc == 0 and not is_file(out_bin):
            rc = 1
            text += "\n[the compiler exited 0 but wrote no binary]"
        var rec = BuildRecord(source_sha, sdk, String(), version, utc_iso_now())
        return (rc, cp_tail(String(text.strip()), REPORT_CAP), out_bin, rec^)
    except e:
        try:
            remove_tree(tmp)
        except:
            pass
        raise e^


# ---------------------------------------------------------------------------
# Running test cases (6.4, 6.8)
# ---------------------------------------------------------------------------
@fieldwise_init
struct CaseResult(Copyable):
    var name: String
    var ok: Bool
    var detail: String       # "" when ok
    var ms: Int


@fieldwise_init
struct CaseRun(Copyable):
    var results: List[CaseResult]
    var tool_description: String     # from tools/list ("" when no case got that far)

    def passed(self) -> Int:
        var n = 0
        for r in self.results:
            if r.ok:
                n += 1
        return n

    def lines(self) -> String:
        var out_lines = List[String]()
        for r in self.results:
            if r.ok:
                out_lines.append("- " + _one_line(r.name) + ": ok (" + String(r.ms) + " ms)")
            else:
                out_lines.append("- " + _one_line(r.name) + ": " + r.detail)
        return String("\n").join(out_lines)


def tool_read_allow(repo_root: String) raises -> List[String]:
    """CONF_TOOL readable subtrees under $HOME: the Modular runtime libraries."""
    return [modular_lib_dir(repo_root)]


def _repo_root_from_binary(self_binary: String) raises -> String:
    """The nearest ancestor of `self_binary` that holds the Mojo runtime libraries
    (`.venv/lib/python3.*/site-packages/modular/lib`)."""
    var d = _dirname(realpath_strict(self_binary))
    while True:
        try:
            _ = modular_lib_dir(d)
            return d
        except:
            pass
        if d == "/" or d.byte_length() == 0:
            break
        d = _dirname(d)
    raise Error("QA-E163: OS confinement failed: no .venv with the Modular runtime libraries above " + self_binary)


def _run_one_case(binary: String, name: String, c: ToolCase, timeout_s: Int, expected_sha: String,
                  self_binary: String, confined: Bool, read_allow: List[String],
                  mut description: String) -> CaseResult:
    var t0 = monotonic_ms()
    var detail: String
    var tail = String()
    var tmp = String()
    try:
        tmp = make_temp_dir("qumo-tool-")
        var tmp_real = realpath_strict(tmp)
        var copy = join_path(tmp_real, name)
        copy_file(binary, copy, 0o700)
        if sha256_file_hex(copy) != expected_sha:
            detail = "the binary changed while testing; build and test again"
        else:
            var conf = Optional[Confinement](None)
            var trampoline = String()
            if confined:
                conf = Confinement(CONF_TOOL, [tmp_real], read_allow.copy(), _home_realpath(),
                                   4 * timeout_s + 10, TOOL_FSIZE_BYTES)
                trampoline = self_binary
            var params = StdioServerParams(copy, List[String](), List[String](), tmp_real,
                                           timeout_s * 1000, TOOL_MAX_MESSAGE_BYTES, STDIO_PIPE,
                                           conf^, trampoline)
            var session = McpStdioSession(params)
            try:
                session.initialize("qumo-toolsmith", "0.1.0")
                var tools = session.list_tools()
                if len(tools) != 1 or tools[0].name != name:
                    var names = List[String]()
                    for t in tools:
                        names.append(t.name)
                    detail = "tools/list must return exactly one tool named " + name + ", got " + py_str_repr(String(", ").join(names))
                else:
                    if description.byte_length() == 0:
                        description = tools[0].description
                    var r = session.call_tool(name, c.arguments_json)
                    detail = check_expectation(c, r.is_error, String("\n").join(r.parts))
            except e:
                detail = String(e)
            try:
                _ = session.close()
            except:
                pass
            tail = session.stderr_tail()
    except e:
        detail = String(e)
    if tmp.byte_length() > 0:
        try:
            remove_tree(tmp)
        except:
            pass
    var ok = detail.byte_length() == 0
    if not ok:
        detail = _one_line(detail)
        var t = String(tail.strip())
        if t.byte_length() > 0:
            detail += "; stderr: " + _one_line(cp_tail(t, STDERR_TAIL_CAP))
    return CaseResult(c.name, ok, detail, monotonic_ms() - t0)


def run_case_list(binary: String, name: String, cases: List[ToolCase], timeout_s: Int, expected_sha: String,
                  self_binary: String, confined: Bool, repo_root: String) raises -> CaseRun:
    """Every case against a fresh private copy of `binary` (whose sha256 must stay
    `expected_sha`), confined under CONF_TOOL when `confined`."""
    var read_allow = List[String]()
    if confined:
        read_allow = tool_read_allow(repo_root)
    var t = _clamp(timeout_s, TEST_TIMEOUT_MIN, TEST_TIMEOUT_MAX)
    var results = List[CaseResult]()
    var description = String()
    for c in cases:
        results.append(_run_one_case(binary, name, c, t, expected_sha, self_binary, confined, read_allow, description))
    return CaseRun(results^, description)


def run_cases(binary: String, name: String, cases_json: String, timeout_s: Int, self_binary: String, confined: Bool) raises -> Tuple[Int, Int, String]:
    """(passed, total, one line per case). The repository whose runtime libraries a
    confined tool may read is found above `self_binary`."""
    var cases: List[ToolCase]
    try:
        cases = parse_cases(cases_json, name)
    except e:
        raise Error(String(E_TOOL_REFUSED) + ": " + name + ": cases.json " + String(e))
    var repo_root = String()
    if confined:
        repo_root = _repo_root_from_binary(self_binary)
    var run = run_case_list(binary, name, cases, timeout_s, sha256_file_hex(binary), self_binary, confined, repo_root)
    return (run.passed(), len(run.results), run.lines())


def _case_run_json(run: CaseRun, rec: BuildRecord, cases_sha: String) -> String:
    var w = JsonWriter()
    w.begin_object()
    w.key("build")
    rec.write_json(w)
    w.kv_string("cases_sha256", cases_sha)
    w.kv_int("passed", run.passed())
    w.kv_int("total", len(run.results))
    w.key("results")
    w.begin_array()
    for r in run.results:
        w.begin_object()
        w.kv_string("name", r.name)
        w.kv_bool("ok", r.ok)
        w.kv_string("detail", r.detail)
        w.kv_int("ms", r.ms)
        w.end_object()
    w.end_array()
    w.kv_string("description", run.tool_description)
    w.kv_string("at", utc_iso_now())
    w.end_object()
    return w.finish()


# ---------------------------------------------------------------------------
# The four tools
# ---------------------------------------------------------------------------
def _rel(name: String, file: String) -> String:
    return String(TOOLS_REL_DIR) + "/" + name + "/" + file


def _publish_branch(sandbox: String) -> String:
    """`git -C <sandbox> rev-parse --abbrev-ref HEAD`, or "" on failure."""
    try:
        var env = minimal_child_env()
        var path_env = String()
        for e in env:
            if e.startswith("PATH="):
                path_env = String(e[byte=5:])
        var git = which("git", path_env)
        if git.byte_length() == 0:
            return String()
        var spec = SpawnSpec(git, [git, "-C", sandbox, "rev-parse", "--abbrev-ref", "HEAD"], env^, String(),
                             STDIO_NULL, STDIO_PIPE, STDIO_PIPE, True)
        var res = run_capture(spec, 30000, 65536, String())
        if res.exit_code != 0:
            return String()
        return String(res.stdout.strip())
    except:
        return String()


struct Toolsmith(Copyable):
    var repo_root: String           # the main checkout: never written by agents
    var store: String
    var mojo_bin: String            # <repo_root>/.venv/bin/mojo
    var self_binary: String
    var sandbox_kind: String
    var allow_unsandboxed: Bool     # --allow-unsandboxed-tool-tests

    def __init__(out self, repo_root: String, store: String, self_binary: String, sandbox_kind: String, allow_unsandboxed: Bool):
        self.repo_root = repo_root
        self.store = store
        self.mojo_bin = join_path(repo_root, ".venv/bin/mojo")
        self.self_binary = self_binary
        self.sandbox_kind = sandbox_kind
        self.allow_unsandboxed = allow_unsandboxed

    # -- shared checks -----------------------------------------------------
    def _refusal(self, sandbox: String, name: String) -> String:
        if sandbox.byte_length() == 0:
            return REFUSE_NO_SANDBOX
        try:
            if realpath_strict(sandbox) == realpath_strict(self.repo_root):
                return REFUSE_NO_SANDBOX
        except e:
            return "error: " + String(e)
        if not valid_tool_name(name):
            return REFUSE_NAME
        return String()

    def _path(self, sandbox: String, name: String, file: String) raises -> String:
        """`<sandbox>/agents/mojo/tools/<name>/<file>`, refused when it resolves
        outside the sandbox (a symlink planted in the sandbox)."""
        var p = join_path(sandbox, _rel(name, file)) if file.byte_length() > 0 else join_path(sandbox, String(TOOLS_REL_DIR) + "/" + name)
        if not is_inside(resolve_nonstrict(p), realpath_strict(sandbox)):
            raise Error("refused: " + (_rel(name, file) if file.byte_length() > 0 else String(TOOLS_REL_DIR) + "/" + name) + " resolves outside the sandbox")
        return p

    def _confined(self) -> Bool:
        """Whether model-written code runs under the OS sandbox; False only when there
        is none and a human allowed unsandboxed runs (callers check that first)."""
        return self.sandbox_kind != "none" and self.sandbox_kind != SANDBOX_KIND_UNSANDBOXED

    def _may_run_code(self) -> Bool:
        return self._confined() or self.allow_unsandboxed or self.sandbox_kind == SANDBOX_KIND_UNSANDBOXED

    # -- toolsmith_new -----------------------------------------------------
    def new_tool(self, sandbox: String, name: String, description: String, source: String, cases: String, overwrite: Bool) -> String:
        var r = self._refusal(sandbox, name)
        if r.byte_length() > 0:
            return r
        if String(description.strip()).byte_length() == 0:
            return "refused: description must not be empty"
        try:
            var cases_text = cases if cases.byte_length() > 0 else cases_template(name)
            var parsed: List[ToolCase]
            try:
                parsed = parse_cases(cases_text, name)
            except e:
                return "refused: cases.json " + String(e)
            var source_text = source if source.byte_length() > 0 else tool_template(name, description)
            var tool_path = self._path(sandbox, name, "tool.mojo")
            var cases_path = self._path(sandbox, name, "cases.json")
            if not overwrite:
                if exists(tool_path):
                    return "refused: " + _rel(name, "tool.mojo") + " already exists; pass overwrite=true to replace it"
                if exists(cases_path):
                    return "refused: " + _rel(name, "cases.json") + " already exists; pass overwrite=true to replace it"
            make_dirs(self._path(sandbox, name, ""))
            # Re-check after creating the directories (nothing may have moved them).
            tool_path = self._path(sandbox, name, "tool.mojo")
            cases_path = self._path(sandbox, name, "cases.json")
            write_text_atomic(tool_path, source_text)
            write_text_atomic(cases_path, cases_text)
            return "created " + _rel(name, "tool.mojo") + " and cases.json (" + String(len(parsed)) + " cases)"
        except e:
            var t = String(e)
            if t.startswith("refused: "):
                return t
            return "error: " + t

    # -- toolsmith_build ---------------------------------------------------
    def build(self, sandbox: String, name: String, timeout_s: Int) -> String:
        var r = self._refusal(sandbox, name)
        if r.byte_length() > 0:
            return r
        if not self._may_run_code():
            return REFUSE_NO_OS_SANDBOX_BUILD
        try:
            var src = self._path(sandbox, name, "tool.mojo")
            if not is_file(src):
                return "refused: " + _rel(name, "tool.mojo") + " does not exist; call toolsmith_new first"
            var trampoline = self.self_binary if self._confined() else String()
            var res = hermetic_build(self.repo_root, self.mojo_bin, src, name,
                                     _clamp(timeout_s, BUILD_TIMEOUT_MIN, BUILD_TIMEOUT_MAX), trampoline)
            var rc = res[0]
            var tmp_bin = res[2]
            var rec = res[3].copy()
            var text = "exit " + String(rc) + "\n" + res[1]
            try:
                if rc == 0:
                    make_dirs(self._path(sandbox, name, "build"))
                    var dest = self._path(sandbox, name, "build/" + name)
                    copy_file(tmp_bin, dest, 0o755)
                    rec.binary_sha256 = sha256_file_hex(dest)
                    write_text_atomic(self._path(sandbox, name, "build/build.json"), rec.to_json() + "\n")
                    text += "\nbinary: " + _rel(name, "build/" + name) + "\nsha256: " + rec.binary_sha256
            finally:
                remove_tree(_dirname(tmp_bin))
            return text^
        except e:
            var t = String(e)
            if t.startswith("refused: "):
                return t
            return "error: " + t

    # -- toolsmith_test ----------------------------------------------------
    def test(self, sandbox: String, name: String, timeout_s: Int) -> String:
        var r = self._refusal(sandbox, name)
        if r.byte_length() > 0:
            return r
        if not self._may_run_code():
            return REFUSE_NO_OS_SANDBOX_TEST
        try:
            var binary = self._path(sandbox, name, "build/" + name)
            var record_path = self._path(sandbox, name, "build/build.json")
            if not is_file(binary) or not is_file(record_path):
                return REFUSE_REBUILD
            var rec = build_record_from_json(read_text_lossy(record_path))
            var bin_sha = sha256_file_hex(binary)
            if bin_sha != rec.binary_sha256:
                return REFUSE_REBUILD
            var cases_path = self._path(sandbox, name, "cases.json")
            if not is_file(cases_path):
                return "refused: " + _rel(name, "cases.json") + " does not exist; call toolsmith_new first"
            var cases_text = read_text_lossy(cases_path)
            var cases_sha = sha256_file_hex(cases_path)
            var cases: List[ToolCase]
            try:
                cases = parse_cases(cases_text, name)
            except e:
                return "refused: cases.json " + String(e)
            var run = run_case_list(binary, name, cases, _clamp(timeout_s, TEST_TIMEOUT_MIN, TEST_TIMEOUT_MAX),
                                    bin_sha, self.self_binary, self._confined(), self.repo_root)
            write_text_atomic(self._path(sandbox, name, "build/last_test.json"), _case_run_json(run, rec, cases_sha) + "\n")
            var p = run.passed()
            var total = len(run.results)
            var head = ("PASS " if p == total else "FAIL ") + String(p) + "/" + String(total)
            return cp_head(head + "\n" + run.lines(), REPORT_CAP)
        except e:
            var t = String(e)
            if t.startswith("refused: "):
                return t
            return "error: " + t

    # -- toolsmith_publish -------------------------------------------------
    def publish(self, sandbox: String, name: String, version: String) -> String:
        var r = self._refusal(sandbox, name)
        if r.byte_length() > 0:
            return r
        if not valid_tool_version(version):
            return REFUSE_VERSION
        try:
            var last_path = self._path(sandbox, name, "build/last_test.json")
            var src = self._path(sandbox, name, "tool.mojo")
            var cases_path = self._path(sandbox, name, "cases.json")
            var binary = self._path(sandbox, name, "build/" + name)
            if not is_file(last_path):
                return "refused: no toolsmith_test result for " + name + "; build and test the tool first"
            if not self._may_run_code():
                return REFUSE_NO_OS_SANDBOX_TEST
            # last_test.json is an ordinary file in the open sandbox, so the model can write
            # it by hand. It only proves that toolsmith_test was called; its hashes bind the
            # current files, and nothing it says about results or the description is trusted.
            var last = parse_json(read_text_lossy(last_path))
            var rec = build_record_from_value(last.get("build"))
            if not is_file(src) or not is_file(cases_path) or not is_file(binary):
                return REFUSE_STALE
            var bin_sha = sha256_file_hex(binary)
            var src_sha = sha256_file_hex(src)
            var cases_sha = sha256_file_hex(cases_path)
            if bin_sha != rec.binary_sha256 or src_sha != rec.source_sha256 or cases_sha != last.get_string("cases_sha256", ""):
                return REFUSE_STALE
            # The pass count and the description come from a fresh, confined run of the
            # stored cases against the binary that will be published (its tools/list answer).
            var cases: List[ToolCase]
            try:
                cases = parse_cases(read_text_lossy(cases_path), name)
            except err:
                return "refused: cases.json " + String(err)
            var run = run_case_list(binary, name, cases, PUBLISH_CASE_TIMEOUT_S, bin_sha, self.self_binary,
                                    self._confined(), self.repo_root)
            var passed = run.passed()
            var total = len(run.results)
            if total <= 0 or passed != total:
                return "refused: publish re-ran the cases and " + String(passed) + "/" + String(total) + " passed; fix the tool and run toolsmith_test again\n" + cp_head(run.lines(), REPORT_CAP)
            if sha256_file_hex(binary) != bin_sha or sha256_file_hex(src) != src_sha or sha256_file_hex(cases_path) != cases_sha:
                return REFUSE_STALE
            make_dirs(self.store)
            var store_real = realpath_strict(self.store)
            var e = ToolEntry(
                name, version, run.tool_description, entry_dir(name, version, bin_sha),
                src_sha, cases_sha, bin_sha, rec.sdk_sha256, rec.mojo_version, current_platform(),
                DEFAULT_TOOL_TIMEOUT_S, DEFAULT_MAX_OUTPUT_BYTES, utc_iso_now(), PUBLISHED_BY,
                _publish_branch(sandbox), passed, total, False, String(), String())
            var shape = check_entry_shape(store_real, e)
            if not shape.ok:
                return "refused: " + String("; ").join(shape.problems)
            var target = join_path(store_real, e.dir)
            make_dirs(target)
            if not is_inside(realpath_strict(target), store_real):
                return "refused: " + e.dir + " resolves outside the tool store"
            copy_file(src, join_path(target, "tool.mojo"), 0o644)
            copy_file(cases_path, join_path(target, "cases.json"), 0o644)
            copy_file(binary, join_path(target, name), 0o755)
            var check = verify_entry(store_real, e)
            if not check.ok:
                try:
                    remove_tree(target)
                except:
                    pass
                return REFUSE_STALE
            var manifest = load_manifest(self.store)
            var old = manifest.get(name)
            manifest.put(e)
            manifest.save()
            if old:
                var o = old.value().copy()
                if o.dir != e.dir and check_entry_shape(store_real, o).ok:
                    try:
                        remove_tree(join_path(store_real, o.dir))
                    except:
                        pass
            return "published " + name + " " + version + " (unreviewed). A human must run: qumo-agent tools review " + name
        except e:
            var t = String(e)
            if t.startswith("refused: "):
                return t
            return "error: " + t


# ---------------------------------------------------------------------------
# Review rebuild (6.6 step 2)
# ---------------------------------------------------------------------------
def review_rebuild(repo_root: String, store: String, e: ToolEntry, self_binary: String, sandbox_kind: String) raises -> ToolEntry:
    """Rebuild the stored tool.mojo hermetically, re-run the stored cases against the
    new binary and move the tool into `entry_dir(name, version, new sha256)`. Returns
    the updated entry (binary_sha256, dir, sdk_sha256, mojo_version, tests); the
    caller marks it reviewed and saves the manifest. Raises (nothing changed) when the
    entry does not verify, the build fails or a case fails. `sandbox_kind` "none"
    raises QA-E163; pass SANDBOX_KIND_UNSANDBOXED only for a human's
    --allow-unsandboxed-tool-tests."""
    var check = verify_entry(store, e)
    if not check.ok:
        raise Error(String("; ").join(check.problems))
    if sandbox_kind == "none":
        raise Error("QA-E163: no OS sandbox is available on this machine. Hint: a human can pass --allow-unsandboxed-tool-tests")
    var confined = sandbox_kind != SANDBOX_KIND_UNSANDBOXED
    var store_real = realpath_strict(store)
    var old_dir = join_path(store_real, e.dir)
    var stored_src = join_path(old_dir, "tool.mojo")
    var stored_cases = join_path(old_dir, "cases.json")
    var cases = parse_cases(read_text_lossy(stored_cases), e.name)
    var res = hermetic_build(repo_root, join_path(repo_root, ".venv/bin/mojo"), stored_src, e.name,
                             600, self_binary if confined else String())
    var tmp_bin = res[2]
    var tmp_dir = _dirname(tmp_bin)
    try:
        if res[0] != 0:
            raise Error(String(E_TOOL_REFUSED) + ": " + e.name + ": rebuild failed (exit " + String(res[0]) + ")\n" + res[1])
        var rec = res[3].copy()
        if rec.source_sha256 != e.source_sha256:
            raise Error(String(E_TOOL_REFUSED) + ": " + e.name + ": tool.mojo changed during the rebuild")
        var new_sha = sha256_file_hex(tmp_bin)
        var timeout = e.timeout_s if e.timeout_s > 0 else DEFAULT_TOOL_TIMEOUT_S
        var run = run_case_list(tmp_bin, e.name, cases, _clamp(timeout, TEST_TIMEOUT_MIN, TEST_TIMEOUT_MAX),
                                new_sha, self_binary, confined, repo_root)
        var passed = run.passed()
        if passed != len(run.results):
            raise Error(String(E_TOOL_REFUSED) + ": " + e.name + ": the rebuilt binary passes " + String(passed) + "/" + String(len(run.results)) + " cases\n" + run.lines())
        var updated = e.copy()
        updated.binary_sha256 = new_sha
        updated.dir = entry_dir(e.name, e.version, new_sha)
        updated.sdk_sha256 = rec.sdk_sha256
        updated.mojo_version = rec.mojo_version
        updated.tests_passed = passed
        updated.tests_total = len(run.results)
        var shape = check_entry_shape(store_real, updated)
        if not shape.ok:
            raise Error(String("; ").join(shape.problems))
        var target = join_path(store_real, updated.dir)
        make_dirs(target)
        copy_file(stored_src, join_path(target, "tool.mojo"), 0o644)
        copy_file(stored_cases, join_path(target, "cases.json"), 0o644)
        copy_file(tmp_bin, join_path(target, e.name), 0o755)
        var verified = verify_entry(store_real, updated)
        if not verified.ok:
            if updated.dir != e.dir:
                try:
                    remove_tree(target)
                except:
                    pass
            raise Error(String("; ").join(verified.problems))
        if updated.dir != e.dir:
            remove_tree(old_dir)
        return updated^
    finally:
        try:
            remove_tree(tmp_dir)
        except:
            pass
